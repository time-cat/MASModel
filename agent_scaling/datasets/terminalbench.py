"""Terminal-Bench 2.0 dataset classes."""

from typing import Any, Dict, List, Literal, Optional, Union

from agent_scaling.datasets.base import Dataset, DatasetInstance, DatasetInstanceOutput
from agent_scaling.datasets.registry import register_dataset, register_dataset_instance

DATASET_IDS = ["terminalbench"]


@register_dataset_instance(DATASET_IDS)
class TerminalBenchInstance(DatasetInstance):
    """A single Terminal-Bench task instance."""

    task_id: str
    instruction: str
    docker_image: str
    time_limit: int = 600  # seconds
    test_script: Optional[str] = None  # Verification wrapper script (run-tests.sh)
    test_script_pytest: Optional[str] = None  # Actual pytest test file content
    reference_solution: Optional[str] = None  # Oracle solution
    category: Optional[str] = None
    difficulty: Optional[str] = None

    def model_post_init(self, __context):
        self.expected_output = self.reference_solution

    def get_prompt_info(self) -> Dict[str, str]:
        return {
            "instruction": self.instruction,
        }


@register_dataset(DATASET_IDS)
class TerminalBenchDataset(Dataset):
    output_type: Literal["patch_or_state"] = "patch_or_state"
    """Terminal-Bench 2.0 dataset (89 Linux terminal tasks)."""

    dataset_id: str = "terminalbench"
    instances: List[TerminalBenchInstance]

    def get_instance_eval_output(
        self, instance_output: DatasetInstanceOutput[TerminalBenchInstance]
    ) -> Dict[str, Any]:
        return {
            "success": instance_output.final_env_output.success or False,
            "num_steps": instance_output.final_env_output.num_steps or -1,
            "task_id": instance_output.data_instance.task_id,
        }

    def get_instance_eval_metrics(
        self, instance_output: DatasetInstanceOutput[TerminalBenchInstance]
    ) -> Dict[str, Union[int, float]]:
        return {
            "resolved": int(instance_output.final_env_output.success or False),
            "num_steps": instance_output.final_env_output.num_steps or -1,
        }

    def get_metrics(self, eval_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        num_instances = len(eval_outputs)
        num_resolved = sum(e["resolved"] for e in eval_outputs)
        return {
            "resolution_rate": num_resolved / num_instances if num_instances > 0 else 0.0,
            "num_resolved": num_resolved,
            "num_instances": num_instances,
            "avg_num_steps": (
                sum(e["num_steps"] for e in eval_outputs) / num_instances
                if num_instances > 0
                else 0.0
            ),
        }
