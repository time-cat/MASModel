import concurrent.futures
import contextlib
import os
import random
import traceback
from typing import Any, Dict, List, Optional, Union

from langfuse._client.span import LangfuseSpan
from pydantic import BaseModel
from tqdm import tqdm

from agent_scaling.agents import AgentSystem
from agent_scaling.config.run import RunConfig
from agent_scaling.datasets import Dataset, DatasetInstance
from agent_scaling.logger import logger
from agent_scaling.utils import write_json, write_yaml


class InstanceSave(BaseModel):
    inp: Dict[str, Any]
    output: Dict[str, Any]
    # Dataset outputs may contain structured audit data (for example a
    # PlanCraft action trace), while metrics remain scalar by convention.
    metrics: Dict[str, Any]
    expected_output: Optional[Any] = None


class ExperimentRunner:
    def __init__(self, config: RunConfig):
        self.config = config
        self.log_langfuse = config.log_langfuse
        self.output_dir = config.save_dir
        self.dataset: Dataset = self.config.dataset.dataset
        self.lf_dataset = self.config.dataset.langfuse_dataset
        self.agent: AgentSystem = self.config.get_agent()
        logger.debug(f"Logging to langfuse: {self.log_langfuse}")

    def _get_context_manager(self, index: int) -> contextlib.AbstractContextManager:
        if self.log_langfuse:
            assert self.lf_dataset is not None, (
                "Langfuse dataset must be set for logging"
            )
            return self.lf_dataset.items[index].run(
                run_name=self.config.run_name,
                run_description=self.config.llm.model,
                run_metadata=self.config.get_run_metadata(),
            )
        return contextlib.nullcontext()

    def _get_instances(self) -> List[DatasetInstance]:
        max_instances = 10 if self.config.debug else len(self.dataset.instances)
        max_instances = (
            min(max_instances, self.config.max_instances)
            if self.config.max_instances is not None
            else max_instances
        )

        if self.config.dataset.dataset_filter is not None:
            instances = [
                instance
                for i, instance in enumerate(self.dataset.instances)
                if eval(self.config.dataset.dataset_filter, {}, {"x": instance, "i": i})
            ]
        else:
            instances = list(self.dataset.instances)

        # Deterministic shuffle for SWE-bench to avoid alphabetical bias (all astropy first)
        # Skip shuffle for other datasets where ordering is already reasonable
        if "swebench" in self.dataset.dataset_id:
            rng = random.Random(42)
            rng.shuffle(instances)

        instances = instances[:max_instances]
        return instances

    def _failure_metrics(
        self, instance: Optional[DatasetInstance] = None
    ) -> Dict[str, Any]:
        """Create dataset-compatible metrics for an instance-level failure."""
        dataset_id = self.dataset.dataset_id
        if dataset_id in {"synthetic_dag", "synthetic-dag"}:
            return {
                "submitted_score": -1,
                "submitted": False,
                "success": False,
                "parallelism": float(getattr(instance, "parallelism", 0.0)),
                "critical_path": float(getattr(instance, "critical_path", 0.0)),
            }
        if dataset_id == "plancraft-test":
            return {"success": False, "num_steps": 0}
        if "swebench" in dataset_id or "terminalbench" in dataset_id:
            return {"resolved": 0, "num_steps": 0, "success": 0}
        if "browsecomp" in dataset_id:
            return {"is_correct": False, "confidence": 0.0}
        if dataset_id in {"simpleqa", "multiagent_simpleqa"}:
            return {"grade": "NOT_ATTEMPTED", "confidence": 0.0}
        # Most remaining aggregators either use these keys or defensively read
        # optional correctness fields with dict.get()/membership checks.
        return {"resolved": 0, "num_steps": 0, "success": 0}

    def run(self):
        metrics = []
        instances = self._get_instances()

        if self.log_langfuse:
            iterator = tqdm(
                enumerate(instances),
                total=len(instances),
                desc=f"Evaluating {self.dataset.dataset_id} dataset instances",
            )
        else:
            iterator = enumerate(instances)

        consecutive_failures = 0
        max_consecutive_failures = 3
        for i, instance in iterator:
            if self.config.debug and i >= 10:
                break
            instance_dir = None
            if self.output_dir is not None:
                instance_dir = os.path.join(
                    self.output_dir, "instance_runs", f"{i:04d}"
                )
                os.makedirs(instance_dir, exist_ok=True)

            # Use the core worker function
            try:
                inst_metrics = self._process_single_instance(i, instance, instance_dir)
                metrics.append(inst_metrics)
                consecutive_failures = 0
            except Exception as e:
                logger.error(f"Instance {i} failed with error: {e}")
                # Record failure metrics so experiment continues
                # Use "resolved" key to match dataset get_metrics expectations
                metrics.append(self._failure_metrics(instance))
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(
                        f"Aborting experiment: {consecutive_failures} consecutive failures. "
                        f"Last error: {e}"
                    )
                    break

        all_metrics = self.dataset.get_metrics(metrics)
        if self.output_dir is not None:
            write_json(
                all_metrics,
                os.path.join(self.output_dir, "dataset_eval_metrics.json"),
                indent=True,
            )
        return all_metrics

    def _process_single_instance(
        self,
        i: int,
        instance,
        instance_dir: Optional[str] = None,
    ) -> Dict[str, Union[int, float]]:
        """
        Core worker function to process a single instance.
        This can be reused for both sequential and parallel processing.

        Args:
            i: Index of the instance
            instance: Dataset instance to process
            instance_dir: Optional directory to save outputs

        Returns:
            Tuple of (index, metrics) to maintain order
        """

        context_manager = self._get_context_manager(i)
        with context_manager as span:
            agent = self.config.get_agent()
            try:
                output = agent.run_agent(
                    instance,
                    instance_dir=instance_dir,
                    llm_params=self.config.llm.params,
                    instance_idx=i,
                )
                inst_metrics = self.dataset.get_instance_eval_metrics(output)
                inst_output = self.dataset.get_instance_eval_output(output)
                if isinstance(span, LangfuseSpan):
                    span.update(
                        input=instance.get_prompt_info(),
                        output=inst_output,
                    )
                    for name, metric in inst_metrics.items():
                        span.score_trace(
                            name=name,
                            value=metric,
                        )

                if instance_dir is not None:
                    output_save = InstanceSave(
                        inp=instance.get_prompt_info(),
                        output=inst_output,
                        metrics=inst_metrics,  # type: ignore
                        expected_output=instance.expected_output,
                    )
                    write_yaml(
                        output_save.model_dump(),
                        os.path.join(instance_dir, "instance_save.yaml"),
                        indent=True,
                    )

                return inst_metrics
            finally:
                # Cleanup Docker containers — handle both single and multi-agent
                self._cleanup_agent(agent)

    def _cleanup_agent(self, agent) -> None:
        """Clean up Docker containers for any agent type."""
        # Single-agent: agent.env._cleanup()
        if hasattr(agent, "env") and hasattr(agent.env, "_cleanup"):
            try:
                agent.env._cleanup()
            except Exception as e:
                logger.warning(f"Docker cleanup error: {e}")

        # Multi-agent: iterate over subagents
        lead = getattr(agent, "lead_agent", None)
        if lead is not None:
            subagents = getattr(lead, "subagents", {})
            for agent_id, sub in subagents.items():
                env = getattr(sub, "env", None)
                if env is not None and hasattr(env, "_cleanup"):
                    try:
                        env._cleanup()
                    except Exception as e:
                        logger.warning(f"Docker cleanup error for {agent_id}: {e}")

    def run_parallel(self, num_workers: int = 4):
        """
        Run the experiment in parallel using multiprocessing.

        Args:
            num_workers: Maximum number of worker processes
        """
        # Prepare work items
        instances = self._get_instances()

        work_items = []
        for i, instance in enumerate(instances):
            instance_dir = None
            if self.output_dir is not None:
                instance_dir = os.path.join(
                    self.output_dir, "instance_runs", f"{i:04d}"
                )
                os.makedirs(instance_dir, exist_ok=True)

            work_items.append((i, instance, instance_dir))

        # Process in parallel
        metrics: List[Dict[str, Any]] = [
            self._failure_metrics(instance) for _, instance, _ in work_items
        ]  # Pre-allocate valid failures to maintain order

        # Parallel processing with progress bar

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    self._process_single_instance,
                    *work_item,
                ): i
                for i, work_item in enumerate(work_items)
            }
            with tqdm(
                total=len(work_items),
                desc=f"Evaluating {self.dataset.dataset_id} dataset instances (parallel)",
            ) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    try:
                        result = future.result()
                        metrics[i] = result
                    except Exception as exc:
                        tb_str = "".join(
                            traceback.format_exception(
                                type(exc), exc, exc.__traceback__
                            )
                        )
                        logger.error(
                            f"Error generating output: {exc}\nTraceback:\n{tb_str}"
                        )
                        metrics[i] = self._failure_metrics(instances[i])

                    pbar.update(1)

        all_metrics = self.dataset.get_metrics(metrics)
        if self.output_dir is not None:
            write_json(
                all_metrics,
                os.path.join(self.output_dir, "dataset_eval_metrics.json"),
                indent=True,
            )
        return all_metrics
