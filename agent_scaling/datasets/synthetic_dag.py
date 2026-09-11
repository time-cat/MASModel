"""Deterministic, partially-observable DAG tasks for coordination experiments.

Each instance exposes a dependency graph but hides node values behind the
environment's ``inspect_node`` tool. A node can only be inspected after all
of its predecessors have been inspected, creating controllable sequential
dependencies while independent branches can be explored in parallel. The
arithmetic difficulty profile changes the post-observation computation while
leaving the structural inspection threshold unchanged.
"""

import re
from typing import Any, Dict, List

from pydantic import Field, model_validator

from agent_scaling.datasets.base import Dataset, DatasetInstance, DatasetInstanceOutput
from agent_scaling.datasets.registry import register_dataset, register_dataset_instance
from agent_scaling.datasets.synthetic_dag_rules import (
    compute_target_score,
    get_difficulty_profile,
)

DATASET_IDS = ["synthetic_dag", "synthetic-dag"]


@register_dataset_instance(DATASET_IDS)
class SyntheticDAGInstance(DatasetInstance):
    task_id: str
    nodes: List[str]
    edges: List[List[str]] = Field(default_factory=list)
    node_values: Dict[str, int]
    target_score: int
    parallelism: float = 0.0
    critical_path: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self.expected_output = self.target_score

    @model_validator(mode="after")
    def validate_generated_instance(self) -> "SyntheticDAGInstance":
        if set(self.nodes) != set(self.node_values):
            raise ValueError("node_values must contain exactly one value per node")
        positions = {node: i for i, node in enumerate(self.nodes)}
        for u, v in self.edges:
            if u not in positions or v not in positions:
                raise ValueError(f"edge {u}->{v} references an unknown node")
            if positions[u] >= positions[v]:
                raise ValueError("nodes must be stored in topological order")
        difficulty = str(self.metadata.get("difficulty", "easy"))
        expected = compute_target_score(
            self.nodes,
            self.edges,
            self.node_values,
            difficulty,
        )
        if expected != self.target_score:
            raise ValueError("target_score does not match the generated DAG")
        return self

    def get_prompt_info(self) -> Dict[str, str]:
        edge_text = ", ".join(f"{u}->{v}" for u, v in self.edges) or "none"
        return {
            "task_id": self.task_id,
            "nodes": ", ".join(self.nodes),
            "edges": edge_text,
            "parallelism": f"{self.parallelism:.2f}",
            "critical_path": str(self.critical_path),
            "difficulty": str(self.metadata.get("difficulty", "easy")),
            "computation_rule": get_difficulty_profile(
                str(self.metadata.get("difficulty", "easy"))
            )["rule_text"],
        }


@register_dataset(DATASET_IDS)
class SyntheticDAGDataset(Dataset):
    dataset_id: str = "synthetic_dag"
    instances: List[SyntheticDAGInstance]

    def get_instance_eval_output(
        self, instance_output: DatasetInstanceOutput[SyntheticDAGInstance]
    ) -> Dict[str, Any]:
        raw = str(instance_output.agent_output or "")
        labelled = re.findall(
            r"(?:final\s+score|final\s+answer|answer|score)\s*[:=]\s*(-?\d+)",
            raw,
            flags=re.IGNORECASE,
        )
        integers = re.findall(r"-?\d+", raw)
        # ``InstanceSave.metrics`` accepts scalar numeric values but not None.
        # Use -1 as an explicitly invalid score when a run never submitted an
        # answer (for example, when the decision budget is exhausted before the
        # first tool call). This preserves failure semantics while allowing the
        # runner to persist the per-instance result.
        submitted = int(labelled[-1]) if labelled else (
            int(integers[-1]) if integers else -1
        )
        target = instance_output.data_instance.target_score
        return {
            "submitted_score": submitted,
            "submitted": submitted >= 0,
            "target_score": target,
            "success": submitted == target,
            "parallelism": instance_output.data_instance.parallelism,
            "critical_path": instance_output.data_instance.critical_path,
            "num_nodes": len(instance_output.data_instance.nodes),
            "structure": instance_output.data_instance.metadata.get("structure", "unknown"),
            "difficulty": instance_output.data_instance.metadata.get("difficulty", "easy"),
            # Keep the structural threshold attached to each evaluation so
            # threshold-response analyses do not need to rejoin the raw
            # dataset by task id after an experiment.
            "theoretical_min_budget": instance_output.data_instance.metadata.get(
                "theoretical_min_budget", len(instance_output.data_instance.nodes) + 1
            ),
            "budget_family": instance_output.data_instance.metadata.get(
                "budget_family", "default"
            ),
        }

    def get_instance_eval_metrics(
        self, instance_output: DatasetInstanceOutput[SyntheticDAGInstance]
    ) -> Dict[str, Any]:
        return self.get_instance_eval_output(instance_output)

    def get_metrics(self, eval_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(eval_outputs)
        if n == 0:
            return {"accuracy": 0.0, "num_instances": 0}
        return {
            "accuracy": sum(bool(e.get("success")) for e in eval_outputs) / n,
            "num_instances": n,
            "avg_parallelism": sum(float(e.get("parallelism", 0.0)) for e in eval_outputs) / n,
            "avg_critical_path": sum(float(e.get("critical_path", 0.0)) for e in eval_outputs) / n,
        }
