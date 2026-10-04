"""Type-aware aggregation for multi-agent task outputs.

This module is the single contract boundary between agent transcripts and
dataset evaluators.  A candidate is never treated as an untyped string at the
aggregation boundary: the task output type determines extraction, selection,
canonicalization, and evaluation semantics.

The layer is deliberately deterministic.  An optional synthesizer hook can be
provided by a caller for free-form text, but it is never used for exact scalar
answers or executable environment plans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence


OutputType = Literal[
    "scalar_exact", "free_text", "executable_plan", "patch_or_state"
]
PlanStrategy = Literal["majority", "successful"]
TiePolicy = Literal["first", "abstain", "lexical"]


@dataclass
class CandidateRecord:
    """One agent contribution as seen by the aggregation layer."""

    agent_id: str
    text: str = ""
    agent: Any | None = None
    env: Any | None = None
    terminal_submission: Any | None = None

    def __post_init__(self) -> None:
        self.agent_id = str(self.agent_id)
        self.text = str(self.text or "")
        if self.env is None and self.agent is not None:
            self.env = getattr(self.agent, "env", None)
        if self.terminal_submission is None and self.agent is not None:
            call = getattr(self.agent, "last_terminal_tool_call", None)
            if isinstance(call, Mapping):
                self.terminal_submission = dict(call.get("args") or {})


@dataclass
class AggregationRequest:
    """Normalized request passed to the type-aware aggregator."""

    output_type: OutputType
    candidates: Sequence[CandidateRecord | tuple[str, str]]
    agents: Sequence[Any] = field(default_factory=tuple)
    plan_strategy: PlanStrategy = "majority"
    # An exact-value tie is not evidence.  Abstaining avoids making agent
    # insertion order an unreported experimental variable.
    tie_policy: TiePolicy = "abstain"
    synthesizer: Callable[[list[CandidateRecord]], str] | None = None


@dataclass
class AggregationResult:
    """Stable result consumed by architecture logging and dataset adapters."""

    output: str = ""
    output_type: OutputType | None = None
    canonical_submission: Any | None = None
    selected_agent: str | None = None
    candidate_submissions: dict[str, Any] = field(default_factory=dict)
    vote_counts: dict[str, int] = field(default_factory=dict)
    tie_break_reason: str | None = None
    evaluation_semantics: str = "task_contract_submission"
    selected_environment_trajectory: list[dict[str, Any]] = field(default_factory=list)
    invalid_reason: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Return serializable audit metadata shared by all architectures."""
        return {
            "aggregation_output_type": self.output_type,
            "canonical_submission": self.canonical_submission,
            "selected_agent": self.selected_agent,
            "candidate_submissions": self.candidate_submissions,
            "vote_counts": self.vote_counts,
            "tie_break_reason": self.tie_break_reason,
            "evaluation_semantics": self.evaluation_semantics,
            "selected_environment_trajectory": self.selected_environment_trajectory,
            "aggregation_invalid_reason": self.invalid_reason,
        }


def _records(
    candidates: Iterable[CandidateRecord | tuple[str, str]],
    agents: Sequence[Any],
) -> list[CandidateRecord]:
    agent_map = {str(getattr(agent, "agent_id", "")): agent for agent in agents}
    result: list[CandidateRecord] = []
    for item in candidates:
        if isinstance(item, CandidateRecord):
            result.append(item)
        else:
            agent_id, text = item
            agent = agent_map.get(str(agent_id))
            result.append(CandidateRecord(str(agent_id), str(text or ""), agent=agent))
    return result


def _normalize_scalar(value: Any) -> str | None:
    raw = str(value).strip()
    if not raw:
        return None
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None
    if not raw.lstrip("-").replace(".", "", 1).isdigit():
        return None
    return str(int(number)) if number.is_integer() else format(number, "g")


def _candidate_scalar(record: CandidateRecord) -> str | None:
    if isinstance(record.terminal_submission, Mapping):
        value = _normalize_scalar(record.terminal_submission.get("answer"))
        if value is not None:
            return value
    # Keep the mature explicit-label parser in one low-level module; this layer
    # decides when and how its result is admissible for a task contract.
    from .result_selection import explicit_scalar_candidate

    value = explicit_scalar_candidate(record.agent, record.text)
    return _normalize_scalar(value)


def _env_success(record: CandidateRecord) -> bool:
    try:
        status = record.env.env_status() if record.env is not None else None
        return bool(getattr(status, "success", False))
    except Exception:
        return False


def _env_trace(record: CandidateRecord) -> list[dict[str, Any]]:
    try:
        trace = record.env.get_action_trace() if record.env is not None else []
        return [dict(item) for item in (trace or [])]
    except Exception:
        return []


def _select_vote(
    counts: dict[str, int], records: list[CandidateRecord], values: dict[str, str | None],
    tie_policy: TiePolicy,
) -> tuple[str | None, str | None]:
    if not counts:
        return None, "no_valid_candidate"
    maximum = max(counts.values())
    tied = [value for value, count in counts.items() if count == maximum]
    if len(tied) == 1:
        return tied[0], "majority"
    if tie_policy == "abstain":
        return None, "tie_abstained"
    if tie_policy == "lexical":
        return sorted(tied)[0], "tie_lexical_order"
    # Compatibility mode: first appearance remains deterministic, but the
    # reason is explicit in metadata so it cannot be mistaken for a majority.
    for record in records:
        if values.get(record.agent_id) in tied:
            return values[record.agent_id], "tie_first_candidate"
    return tied[0], "tie_first_candidate"


def _aggregate_scalar(request: AggregationRequest, records: list[CandidateRecord]) -> AggregationResult:
    values = {record.agent_id: _candidate_scalar(record) for record in records}
    counts: dict[str, int] = {}
    for value in values.values():
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    selected, reason = _select_vote(counts, records, values, request.tie_policy)
    result = AggregationResult(
        candidate_submissions=values,
        vote_counts=counts,
        tie_break_reason=reason,
    )
    if selected is None:
        result.invalid_reason = reason
        return result
    from .result_selection import append_final_scalar, submission_adapter

    for record in records:
        if values.get(record.agent_id) == selected:
            result.output = append_final_scalar(record.text, selected)
            result.selected_agent = record.agent_id
            break
    result.canonical_submission = submission_adapter("scalar_exact", selected)
    return result


def _aggregate_plan(request: AggregationRequest, records: list[CandidateRecord]) -> AggregationResult:
    from .result_selection import (
        _plan_payload,
        normalized_action_sequence,
        plan_is_structurally_complete,
        plancraft_submission_from_env,
        submission_adapter,
    )

    result = AggregationResult(evaluation_semantics="success_environment_state_selection")
    executable_records = [
        record for record in records
        if record.text.strip() or _env_success(record) or _env_trace(record)
    ]
    if not executable_records:
        result.invalid_reason = "no_candidate"
        result.canonical_submission = {"action": "plan", "plan": []}
        return result

    selected: CandidateRecord | None = None
    if request.plan_strategy == "successful":
        selected = next((record for record in executable_records if _env_success(record)), None)
        if selected is None:
            selected = next(
                (record for record in executable_records if plan_is_structurally_complete(record.text)),
                None,
            )
    else:
        groups: dict[str, list[CandidateRecord]] = {}
        for record in executable_records:
            key = normalized_action_sequence(record.text) or _plan_payload(record.text).lower()
            groups.setdefault(key, []).append(record)
        selected = max(groups.values(), key=lambda group: len(group))[0]
    selected = selected or executable_records[0]
    result.output = _plan_payload(selected.text)
    result.selected_agent = selected.agent_id
    result.selected_environment_trajectory = _env_trace(selected)
    result.canonical_submission = (
        plancraft_submission_from_env(selected.env, result.output)
        if selected.env is not None and _env_trace(selected)
        else submission_adapter("executable_plan", result.output)
    )
    return result


def _aggregate_text(request: AggregationRequest, records: list[CandidateRecord]) -> AggregationResult:
    nonempty = [record for record in records if record.text.strip()]
    result = AggregationResult()
    if not nonempty:
        result.invalid_reason = "no_candidate"
        return result
    if request.synthesizer is not None:
        output = str(request.synthesizer(nonempty) or "")
        result.tie_break_reason = "llm_synthesizer"
    else:
        # A contract-aware default does not invent a summary.  It preserves a
        # single complete candidate, which is safe for patch/state outputs and
        # keeps free-text aggregation auditable.
        selected = next((record for record in nonempty if _env_success(record)), nonempty[0])
        output = selected.text
        result.selected_agent = selected.agent_id
        result.tie_break_reason = "successful_environment" if _env_success(selected) else "first_nonempty"
    result.output = output
    result.canonical_submission = output or None
    return result


def _aggregate_patch_or_state(
    request: AggregationRequest, records: list[CandidateRecord]
) -> AggregationResult:
    """Select one coherent patch/state candidate with its environment.

    Independent containers cannot be merged.  A successful environment is
    therefore authoritative; the associated patch/state text follows the same
    candidate.  Without success, preserve a structurally non-empty candidate
    and mark the result as provisional in the tie reason.
    """
    nonempty = [record for record in records if record.text.strip()]
    selected = next((record for record in nonempty if _env_success(record)), None)
    selected = selected or (nonempty[0] if nonempty else None)
    result = AggregationResult()
    if selected is None:
        result.invalid_reason = "no_candidate"
        return result
    result.output = selected.text
    result.selected_agent = selected.agent_id
    result.canonical_submission = selected.text
    result.selected_environment_trajectory = _env_trace(selected)
    result.tie_break_reason = (
        "successful_environment" if _env_success(selected) else "first_nonempty_provisional"
    )
    result.evaluation_semantics = "selected_environment_state_and_submission"
    return result


def aggregate(request: AggregationRequest) -> AggregationResult:
    """Aggregate candidates according to one explicit task contract."""
    records = _records(request.candidates, tuple(request.agents))
    if request.output_type == "scalar_exact":
        result = _aggregate_scalar(request, records)
    elif request.output_type == "executable_plan":
        result = _aggregate_plan(request, records)
    elif request.output_type == "patch_or_state":
        result = _aggregate_patch_or_state(request, records)
    elif request.output_type == "free_text":
        result = _aggregate_text(request, records)
    else:
        raise ValueError(f"Unsupported output_type: {request.output_type}")
    result.output_type = request.output_type
    return result
