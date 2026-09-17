"""Environment for the synthetic budgeted-coordination DAG benchmark."""

import re
from typing import Dict

from agent_scaling.datasets.synthetic_dag import SyntheticDAGInstance
from agent_scaling.env.base import AgentEnvironmentTools
from agent_scaling.env.tools import cls_tool

from .registry import register_env


@register_env("synthetic-dag")
class SyntheticDAGEnvironment(AgentEnvironmentTools):
    """Reveal dependency-ordered node values and accept an exact score."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_instance: SyntheticDAGInstance
        # Deliberately local to one worker. Sharing this set would create an
        # implicit blackboard and confound comparisons between independent,
        # centralized, and decentralized architectures.
        self.revealed: set[str] = set()
        self.is_done = False
        self.success = False
        self.submitted_score = None

    def get_instance_prompt_info(self) -> Dict[str, str]:
        instance = self.dataset_instance
        return {
            **instance.get_prompt_info(),
            "tools_description": self.tools_description,
        }

    def env_done(self) -> bool:
        return self.is_done

    def _predecessors(self, node_id: str) -> list[str]:
        return [u for u, v in self.dataset_instance.edges if v == node_id]

    @cls_tool
    def inspect_node(self, node_id: str) -> str:
        """Reveal one node's value. Prerequisite nodes must be inspected first."""
        instance = self.dataset_instance
        if node_id not in instance.nodes:
            return f"ERROR: unknown node {node_id}. Choose one of {', '.join(instance.nodes)}."
        if node_id in self.revealed:
            return f"{node_id} was already inspected; value={instance.node_values[node_id]}"
        missing = [p for p in self._predecessors(node_id) if p not in self.revealed]
        if missing:
            return f"BLOCKED: inspect prerequisites first: {', '.join(missing)}"
        self.revealed.add(node_id)
        return f"{node_id}: value={instance.node_values[node_id]}; inspected={len(self.revealed)}/{len(instance.nodes)}"

    @cls_tool
    def submit(self, answer: int | None = None, reasoning: str = "") -> str:
        """Submit the final DAG score as an integer.

        ``reasoning`` is accepted for compatibility with the runner's generic
        budget-exhaustion auto-submit path.
        """
        if answer is None:
            # Prefer an explicitly labelled final value.  The previous parser
            # used the first integer in arbitrary reasoning, so text such as
            # ``b0=43 ... Final Answer: 877`` was submitted as 43.
            labelled = re.findall(
                r"(?:final\s+(?:answer|score)|answer|score)\s*[:=]\s*(-?\d+)",
                reasoning,
                flags=re.IGNORECASE,
            )
            match = labelled[-1] if labelled else None
            if match is None:
                integers = re.findall(r"-?\d+", reasoning)
                match = integers[-1] if integers else None
            if match is None:
                # The generic runner auto-submits with a textual reason when
                # the decision budget is exhausted. Record that as an explicit
                # invalid score so the run is persisted as a normal failure.
                answer = -1
            else:
                answer = int(match)
        self.submitted_score = int(answer)
        self.success = self.submitted_score == self.dataset_instance.target_score
        self.is_done = True
        return f"FINAL_SCORE: {self.submitted_score}"
