"""Shared deferred-submission policy for budgeted ReAct agents."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping


DEFAULT_TERMINAL_TOOLS = frozenset({"done", "submit", "submit_patch"})


@dataclass(frozen=True)
class ProvisionalCandidate:
    """One terminal action proposed before the final decision call."""

    tool_name: str
    args: dict[str, Any]


@dataclass
class DeferredSubmissionPolicy:
    """Turn early terminal actions into task-agnostic verification passes.

    The policy never evaluates candidate correctness. It only delays execution of
    known terminal tools while the agent still has decision calls available.
    """

    enabled: bool = True
    terminal_tools: frozenset[str] = DEFAULT_TERMINAL_TOOLS
    candidates: list[ProvisionalCandidate] = field(default_factory=list)

    def is_terminal_tool(self, tool_name: str) -> bool:
        return tool_name in self.terminal_tools

    def defer_if_needed(
        self,
        tool_call: Mapping[str, Any],
        remaining_decision_calls: int | None,
    ) -> str | None:
        """Record and describe an early terminal call, or return ``None``."""

        tool_name = str(tool_call.get("name", ""))
        if (
            not self.enabled
            or not self.is_terminal_tool(tool_name)
            or remaining_decision_calls is None
            or remaining_decision_calls <= 0
        ):
            return None

        raw_args = tool_call.get("args", {})
        args = deepcopy(dict(raw_args)) if isinstance(raw_args, Mapping) else {}
        self.candidates.append(ProvisionalCandidate(tool_name=tool_name, args=args))
        pass_num = len(self.candidates)
        remaining_text = (
            "Your next decision is the final decision."
            if remaining_decision_calls == 1
            else f"You have {remaining_decision_calls} decision calls remaining."
        )
        return (
            f"PROVISIONAL CANDIDATE RECORDED (verification pass {pass_num}). "
            "The terminal action was not executed. "
            f"{remaining_text}\n\n"
            "Independently audit the candidate against the original task and the "
            "complete trajectory. Check task coverage, evidence grounding, state "
            "consistency, tool/action correctness, answer correctness, and output "
            "format. Do not merely repeat the previous reasoning. Keep the candidate "
            "unless you identify a concrete, evidence-based error. Use another "
            "task-relevant tool if evidence is missing; otherwise call the terminal "
            "tool again with the revised or best-supported candidate."
        )

    def latest_tool_call(self, *, call_id: str) -> dict[str, Any] | None:
        """Build a fresh tool call for the most recent provisional candidate."""

        if not self.candidates:
            return None
        candidate = self.candidates[-1]
        return {
            "name": candidate.tool_name,
            "args": deepcopy(candidate.args),
            "id": call_id,
            "type": "tool_call",
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "deferred_terminal_calls": len(self.candidates),
            "candidate_tool_names": [c.tool_name for c in self.candidates],
        }
