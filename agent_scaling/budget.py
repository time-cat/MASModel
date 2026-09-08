"""Instance-level budgets for controlled agent-system experiments."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional
import math


@dataclass
class BudgetLedger:
    """Thread-safe lifetime budget for one agent on one task instance."""

    max_decision_calls: Optional[int] = None
    used_decision_calls: int = 0
    auxiliary_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __post_init__(self) -> None:
        if self.max_decision_calls is not None and self.max_decision_calls < 0:
            raise ValueError("max_decision_calls must be non-negative")
        self._lock = threading.Lock()

    @property
    def remaining_decision_calls(self) -> Optional[int]:
        with self._lock:
            if self.max_decision_calls is None:
                return None
            return max(0, self.max_decision_calls - self.used_decision_calls)

    def try_consume_decision(self) -> bool:
        """Atomically reserve one worker decision call."""
        with self._lock:
            if (
                self.max_decision_calls is not None
                and self.used_decision_calls >= self.max_decision_calls
            ):
                return False
            self.used_decision_calls += 1
            return True

    def record_auxiliary_call(self) -> None:
        with self._lock:
            self.auxiliary_calls += 1

    def record_response(self, response: object) -> None:
        """Record provider usage when LiteLLM exposes it on an AIMessage."""
        metadata = getattr(response, "response_metadata", {}) or {}
        raw = metadata.get("litellm_response", metadata)
        usage = getattr(raw, "usage", None)
        if usage is None and isinstance(raw, dict):
            usage = raw.get("usage", {})
        # LangChain providers may expose the same counters under
        # ``token_usage`` instead of LiteLLM's response object.
        if not usage and isinstance(metadata, dict):
            usage = metadata.get("token_usage") or metadata.get("usage")
        if usage is None:
            return

        def get_value(name: str) -> int:
            value = getattr(usage, name, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(name, 0)
            return int(value or 0)

        with self._lock:
            self.prompt_tokens += get_value("prompt_tokens")
            self.completion_tokens += get_value("completion_tokens")

    def snapshot(self) -> Dict[str, Optional[int]]:
        with self._lock:
            remaining = (
                None
                if self.max_decision_calls is None
                else max(0, self.max_decision_calls - self.used_decision_calls)
            )
            return {
                "max_decision_calls": self.max_decision_calls,
                "used_decision_calls": self.used_decision_calls,
                "remaining_decision_calls": remaining,
                "auxiliary_calls": self.auxiliary_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            }


def per_round_cap(per_agent_budget: Optional[int], rounds: Optional[int]) -> Optional[int]:
    """Derive a local round quota without changing the lifetime budget."""
    if per_agent_budget is None:
        return None
    if rounds is None or rounds <= 0:
        return per_agent_budget
    return max(1, math.ceil(per_agent_budget / rounds))
