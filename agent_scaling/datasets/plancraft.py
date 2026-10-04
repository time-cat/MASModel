from typing import Any, Dict, List, Literal

from plancraft.simple import PlancraftExample

from agent_scaling.datasets.base import (
    Dataset,
    DatasetInstance,
    DatasetInstanceOutputWithTrajectory,
)
from agent_scaling.datasets.registry import register_dataset, register_dataset_instance

DATASET_IDS = ["plancraft-test"]


@register_dataset_instance(DATASET_IDS)
class PlancraftInstance(PlancraftExample, DatasetInstance):
    def model_post_init(self, context: Any) -> None:
        self.expected_output = self.target
        self.slotted_inventory = {int(k): v for k, v in self.slotted_inventory.items()}

    def get_prompt_info(self) -> Dict[str, Any]:
        return {
            "inventory": self.inventory,
            "target": self.target,
        }


@register_dataset(DATASET_IDS)
class PlancraftDataset(Dataset):
    """PlanCraft is evaluated from environment state, not plan-text matching.

    Multi-agent runs select one worker's environment (success-first); the
    recorded action trace is diagnostic evidence of that selected execution.
    """
    dataset_id: str = "plancraft-test"
    output_type: Literal["executable_plan"] = "executable_plan"
    instances: List[PlancraftInstance]

    def get_instance_eval_output(
        self, instance_output: DatasetInstanceOutputWithTrajectory[PlancraftInstance]
    ) -> Dict[str, Any]:
        return {
            "success": instance_output.final_env_output.success
            if instance_output.final_env_output
            else False,
            "num_steps": instance_output.final_env_output.num_steps
            if instance_output.final_env_output
            else -1,
            "evaluation_semantics": "environment_state_execution",
            "submission_semantics": "selected_environment_action_trace",
            "action_trace": [step.model_dump() for step in instance_output.trajectory],
        }

    def get_instance_eval_metrics(
        self, instance_output: DatasetInstanceOutputWithTrajectory[PlancraftInstance]
    ) -> Dict[str, Any]:
        # Metrics are scalar values consumed by Langfuse and dataset-level
        # aggregation.  Keep the full action trace in the instance output and
        # trajectory audit fields, not in this scalar metrics dictionary.
        evaluation = self.get_instance_eval_output(instance_output)
        return {
            "success": evaluation["success"],
            "num_steps": evaluation["num_steps"],
        }

    def get_metrics(self, eval_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "avg_success": sum(e["success"] for e in eval_outputs) / len(eval_outputs),
            "avg_num_steps": sum(e["num_steps"] for e in eval_outputs)
            / len(eval_outputs),
            "num_instances": len(eval_outputs),
        }
