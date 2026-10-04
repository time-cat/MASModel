"""Independent parallel ensemble with contract-aware aggregation.

All workers run independently.  The aggregation layer preserves a complete
text synthesis for free-form tasks, performs exact-value voting for scalar
tasks, and selects one successful environment trajectory for executable plans.
No incompatible environment states are merged.
"""
import asyncio
import os.path as osp
import time
from typing import Dict, List, Optional, Tuple

from agent_scaling.agents.base import AgentSystemWithTools
from agent_scaling.config.llm import LLMParams
from agent_scaling.datasets import DatasetInstance, DatasetInstanceOutputWithTrajectory
from agent_scaling.logger import logger
from agent_scaling.utils import write_yaml

from .multiagent_components.mas_subagent import WorkerSubagent
from .multiagent_components.memory import EnhancedMemory
from .multiagent_utils.aggregation import AggregationRequest, aggregate
from .multiagent_utils.result_selection import (
    task_output_type,
    submit_tool_call,
)
from .registry import register_agent
from agent_scaling.budget import per_round_cap


@register_agent("multi-agent-independent")
class IndependentMultiAgentSystem(AgentSystemWithTools):
    """N agents run in parallel with no inter-agent communication.

    Aggregation is delegated to the shared type-aware layer. The original
    worker transcripts remain in ``raw_candidates`` for auditability.

    `max_iterations_per_agent` is an optional legacy safety cap. Canonical
    budget experiments omit it and use the per-worker lifetime budget.
    """

    required_prompts = ["subagent"]

    def __init__(
        self,
        *args,
        n_base_agents: int = 3,
        min_iterations_per_agent: int = 3,
        max_iterations_per_agent: int | None = None,
        total_decision_budget: int | None = 32,
        use_remaining_budget_for_verification: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory = EnhancedMemory()
        self.n_base_agents = n_base_agents
        self.min_iterations_per_agent = min_iterations_per_agent
        self.max_iterations_per_agent = max_iterations_per_agent
        self.total_decision_budget = total_decision_budget
        self.use_remaining_budget_for_verification = (
            use_remaining_budget_for_verification
        )
        self.per_agent_decision_budget = (
            None
            if total_decision_budget is None
            else total_decision_budget // max(1, n_base_agents)
        )
        self.subagents: Dict[str, WorkerSubagent] = {}

        logger.info(
            f"IndependentMultiAgentSystem: {n_base_agents} agents in parallel, "
            f"type-aware aggregation"
        )

    def _create_subagents(self, task_instance: DatasetInstance):
        """Create N fully independent agents, each working on the full task."""
        self.subagents = {}
        for i in range(self.n_base_agents):
            agent_id = f"agent_{i + 1}"
            subagent = WorkerSubagent.init_from_agent(
                agent=self,
                agent_id=agent_id,
                objective="Solve the task completely on your own",
                original_query=self.memory.original_task,
                strategy=f"Independent agent {i + 1}",
                task_instance=task_instance,
                min_iterations_per_agent=self.min_iterations_per_agent,
                max_iterations_per_agent=self.max_iterations_per_agent,
                decision_budget=self.per_agent_decision_budget,
                decision_budget_per_round=per_round_cap(
                    self.per_agent_decision_budget, None
                ),
                use_remaining_budget_for_verification=(
                    self.use_remaining_budget_for_verification
                ),
            )
            self.subagents[agent_id] = subagent
        logger.info(
            f"Created {len(self.subagents)} independent agents (no coordination)"
        )

    def _synthesize_only(self) -> Tuple[str, List[str]]:
        """Concatenate each sub-agent's final answer without rewriting it.

        Returns (synthesized_answer, contributing_agent_ids).
        """
        parts: List[str] = []
        contributing: List[str] = []
        for agent_id, agent in self.subagents.items():
            answer = agent.conv_history.last_outgoing_external_message or ""
            if not answer:
                continue
            parts.append(f"=== {agent_id} ===\n{answer}")
            contributing.append(agent_id)
        return ("\n\n".join(parts), contributing)

    def _auto_submit(self, synthesized_answer: str) -> str:
        """Submit the synthesized answer via the first available agent's env."""
        for agent_id, agent in self.subagents.items():
            env = agent.env
            if env.env_done():
                continue
            tool_call = submit_tool_call(
                env, synthesized_answer, "synthesis_submit_independent"
            )
            if tool_call:
                logger.info(
                    f"Submitting synthesis via {agent_id} using {tool_call['name']}"
                )
                try:
                    tool_msg = env.execute_tool(tool_call)
                    return str(tool_msg.content)
                except Exception as e:
                    logger.warning(f"Synthesis auto-submit via {agent_id} failed: {e}")
        return ""

    def run_agent(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
    ) -> DatasetInstanceOutputWithTrajectory:
        return asyncio.run(
            self.run_agent_async(instance, instance_dir, llm_params, instance_idx)
        )

    async def run_agent_async(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
    ) -> DatasetInstanceOutputWithTrajectory:
        start_time = time.time()

        shared_prompt_templates = self.get_dataset_prompt_templates(
            dataset_instance=instance
        )
        self.memory.original_task = shared_prompt_templates.get(
            "task_instance", str(instance.get_prompt_info())
        )

        self._create_subagents(instance)

        # Single message: each agent works fully independently on the task.
        message = (
            "Solve the task completely on your own. Work systematically and submit "
            "when done.\n\n"
            "CRITICAL: You MUST make changes and submit within your iteration budget. "
            "Do NOT spend more than half your iterations on exploration/analysis. "
            "After understanding the problem, immediately start implementing the fix. "
            "If you reach 50% of your budget without making changes, STOP exploring "
            "and START implementing your best solution immediately."
        )

        tasks: List[Tuple[str, asyncio.Task]] = []
        for agent_id, subagent in self.subagents.items():
            task = asyncio.create_task(
                asyncio.to_thread(subagent.process_orchestrator_message, message)
            )
            tasks.append((agent_id, task))

        # Per-task time budget: only enforced if the dataset instance sets a
        # positive `time_limit`. Defaults to no limit so all N agents run to
        # completion before synthesis_only aggregates their answers.
        time_limit = getattr(instance, "time_limit", None)
        if time_limit is not None and time_limit > 0:
            done, pending = await asyncio.wait(
                [t for _, t in tasks], timeout=time_limit
            )
        else:
            done, pending = await asyncio.wait(
                [t for _, t in tasks], return_when=asyncio.ALL_COMPLETED
            )
        for t in pending:
            t.cancel()

        # Drain each task (logs failures but does not stop synthesis).
        for agent_id, task in tasks:
            if task in done:
                try:
                    await task
                except Exception as e:
                    logger.warning(f"Agent {agent_id} failed: {e}")

        # Preserve every raw candidate for auditability.  The canonical output
        # is produced exclusively by the shared type-aware aggregation layer.
        synthesized_answer, contributing_ids = self._synthesize_only()
        raw_candidates = {
            aid: agent.conv_history.last_outgoing_external_message or ""
            for aid, agent in self.subagents.items()
        }
        result_candidates = (
            list(raw_candidates.items())
            if task_output_type(self) in {"scalar_exact", "executable_plan", "patch_or_state"}
            else [("synthesis", synthesized_answer)]
        )
        aggregation_result = aggregate(
            AggregationRequest(
                output_type=task_output_type(self),
                candidates=result_candidates,
                agents=tuple(self.subagents.values()),
                plan_strategy="successful",
            )
        )
        canonical_answer = aggregation_result.output
        selected_agent_id = aggregation_result.selected_agent
        candidate_submissions = aggregation_result.candidate_submissions
        vote_counts = aggregation_result.vote_counts

        # Give the canonical result to an available environment only when the
        # workers did not already reach a terminal state.
        submission_response = ""
        if canonical_answer and not any(agent.env.env_done() for agent in self.subagents.values()):
            submission_response = self._auto_submit(canonical_answer)

        # Canonical env status. For text-output benchmarks (Finance-Agent,
        # WorkBench, BrowseComp-Plus, PlanCraft) the grader reads
        # agent_output, so env_status is only infrastructure metadata. For
        # environment-state benchmarks evaluated against a Docker container's
        # terminal state (SWE-bench Verified, Terminal-Bench), independent
        # container states cannot be physically merged across sub-agent runs,
        # so we deterministically report the first sub-agent's env status in
        # insertion order, unconditionally and without any success-based
        # selection. This is not first-to-succeed: env_done() is not
        # consulted as a filter, consistent with the synthesis-only contract
        # that prohibits voting or cross-validation. Under the canonical
        # all-sub-agents-complete runs reported in the paper this resolves
        # to the first sub-agent's status; the unconditional rule makes the
        # behaviour identical under non-canonical reuse (e.g., early
        # termination of a later sub-agent), where it remains positional
        # rather than success-based.
        agents_in_order = list(self.subagents.values())
        if selected_agent_id and selected_agent_id in self.subagents:
            final_env_status = self.subagents[selected_agent_id].env.env_status()
        else:
            final_env_status = (
                agents_in_order[0].env.env_status() if agents_in_order else None
            )

        final_answer = canonical_answer or submission_response
        selected_env = (
            self.subagents[selected_agent_id].env
            if selected_agent_id in self.subagents
            else (agents_in_order[0].env if agents_in_order else None)
        )
        canonical_submission = aggregation_result.canonical_submission

        execution_time = time.time() - start_time
        total_decision_calls = sum(
            a.budget.snapshot()["used_decision_calls"] for a in self.subagents.values()
        )
        logger.info(
            f"Independent aggregation completed in {execution_time:.2f}s "
            f"with {total_decision_calls} worker decisions across {len(self.subagents)} agents. "
            f"Contributing agents: {contributing_ids}"
        )

        if instance_dir is not None:
            output_data = {
                "architecture": "independent",
                "aggregator": "type_aware_aggregation",
                "n_agents": self.n_base_agents,
                "contributing_agents": contributing_ids,
                "total_iterations": total_decision_calls,
                "total_decision_calls": total_decision_calls,
                "execution_time": execution_time,
                "synthesized_answer": synthesized_answer,
                "raw_candidates": raw_candidates,
                "aggregated_output": canonical_answer,
                "canonical_submission": canonical_submission,
                "output_type": task_output_type(self),
                "selected_agent": selected_agent_id,
                "candidate_submissions": candidate_submissions,
                "vote_counts": vote_counts,
                "tie_break_reason": aggregation_result.tie_break_reason,
                "aggregation_invalid_reason": aggregation_result.invalid_reason,
                "canonical_agent_output": final_answer,
                "evaluation_semantics": (
                    "success_environment_state_selection"
                    if task_output_type(self) in {"executable_plan", "patch_or_state"}
                    else "task_contract_submission"
                ),
                "selected_environment_trajectory": (
                    getattr(selected_env, "get_action_trace", lambda: [])()
                    if selected_env is not None
                    else []
                ),
                "total_decision_budget": self.total_decision_budget,
                "per_agent_budgets": {
                    aid: agent.budget.snapshot()
                    for aid, agent in self.subagents.items()
                },
                "per_agent_verification": {
                    aid: agent.submission_policy.snapshot()
                    for aid, agent in self.subagents.items()
                },
            }
            write_yaml(
                output_data,
                osp.join(instance_dir, "multi_agent_output.yaml"),
                use_long_str_representer=True,
                truncate_floats=False,
            )

        return DatasetInstanceOutputWithTrajectory(
            data_instance=instance,
            agent_output=final_answer,
            canonical_submission=canonical_submission,
            trajectory=(
                getattr(selected_env, "get_action_trace", lambda: [])()
                if selected_env is not None
                else []
            ),
            final_env_output=final_env_status,
        )
