"""Shared result propagation helpers for multi-agent systems.

The worker environments are intentionally isolated, so an MAS needs an explicit
policy for choosing the answer exposed to the dataset evaluator and the matching
environment status.  These helpers keep that plumbing architecture-independent;
their scalar selection is deterministic and based only on explicit answer labels.
"""

from __future__ import annotations

import re
import json
from collections.abc import Mapping
from typing import Any, Iterable

from .aggregation import OutputType



def task_output_type(obj: Any) -> OutputType:
    """Read the explicit task contract from a dataset/system."""
    value = getattr(obj, "output_type", None)
    if value in {"scalar_exact", "free_text", "executable_plan", "patch_or_state"}:
        return value
    dataset = getattr(obj, "dataset", None)
    value = getattr(dataset, "output_type", None)
    if value in {"scalar_exact", "free_text", "executable_plan", "patch_or_state"}:
        return value
    raise ValueError("output_type must be explicitly configured for every task")


# Numeric answers are often wrapped in Markdown/math delimiters by the model
# (for example ``Final answer: **105**`` or ``Score: $105$``).  Keep this
# parser deliberately restricted to explicit answer labels so intermediate
# node ids, budgets, and arithmetic values are not mistaken for submissions.
_LABEL_MARKER_RE = re.compile(
    r"(?:final\s+(?:answer|score)|(?<!final\s)(?:answer|score))"
    r"\s*(?::|=|\bis\b)",
    flags=re.IGNORECASE,
)
_JSON_SCALAR_RE = re.compile(
    r"[\"']answer[\"']\s*:\s*(?:\*\*|__|`|\$|\s)*"
    r"(-?\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)


def last_external_answer(agent: Any) -> str:
    """Return the worker's latest externally visible answer text."""
    return str(getattr(getattr(agent, "conv_history", None),
                       "last_outgoing_external_message", None) or "")


def first_successful_agent(agents: Iterable[Any]) -> Any | None:
    """Return the first successful worker in stable insertion order."""
    for agent in agents:
        status = agent.env.env_status()
        if bool(getattr(status, "success", False)):
            return agent
    return None


def first_nonempty_answer(agents: Iterable[Any]) -> str:
    for agent in agents:
        answer = last_external_answer(agent)
        if answer.strip():
            return answer
    return ""


def labelled_scalar_answers(text: str) -> list[str]:
    """Return explicitly labelled numeric candidates in appearance order."""
    if not text:
        return []
    matches: list[str] = []
    # Work line-by-line. Ordinary labelled prose takes the first number after
    # the marker; formula-like lines may use their final explicit equality.
    for line in str(text).splitlines():
        marker = _LABEL_MARKER_RE.search(line)
        if marker:
            # Ignore node ids or progress counters that occur before the
            # label (e.g. ``Inspect N2 ... final score:``).
            suffix = line[marker.end():]
            # A planning line such as ``Final score = sum(...) mod 1009`` is
            # not a submitted value.  If it has no subsequent equality (the
            # usual sign of a computed result), discard it.
            is_formula = bool(
                re.search(r"\b(?:sum|mod|formula|calculate)\b", suffix, re.I)
            )
            if is_formula and not re.search(r"(?:\d|\))\s*=\s*-?\d", suffix):
                continue
            numbers = re.findall(r"-?\d+(?:\.\d+)?", suffix)
            if numbers:
                matches.append(numbers[-1] if is_formula else numbers[0])
        # Tool-call traces frequently contain a JSON ``answer`` argument
        # without a prose label.  It is still explicit, unlike arbitrary
        # prose numbers, so include it in its original line position.
        matches.extend(match.group(1) for match in _JSON_SCALAR_RE.finditer(line))
    return matches


def last_labelled_scalar(text: str) -> str:
    """Return the last explicit numeric answer label, or ``""``."""
    matches = labelled_scalar_answers(text)
    if matches:
        return matches[-1]
    # Some agents introduce a ``Final score`` section and put the computed
    # value on the following LaTeX lines (``... = 399 mod 1009`` / ``= 399``).
    # Treat only equalities after that explicit section header as candidates;
    # ordinary node/base-value arithmetic remains out of scope.
    headers = list(
        re.finditer(r"final\s+(?:answer|score)", str(text), flags=re.IGNORECASE)
    )
    for header in reversed(headers):
        tail = str(text)[header.end():]
        computed = re.findall(
            r"(?:^|\n)\s*=\s*(-?\d+(?:\.\d+)?)", tail
        )
        if computed:
            return computed[-1]
    return ""


def _normalize_scalar(value: Any) -> str | None:
    raw = str(value).strip()
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", raw):
        return None
    try:
        numeric = float(raw)
        return str(int(numeric)) if numeric.is_integer() else format(numeric, "g")
    except ValueError:
        return raw


def explicit_scalar_candidate(agent: Any, text: str) -> str | None:
    """Prefer an actual terminal tool argument over prose reconstruction."""
    terminal_call = getattr(agent, "last_terminal_tool_call", None)
    if isinstance(terminal_call, Mapping):
        args = terminal_call.get("args", {})
        if isinstance(args, Mapping) and "answer" in args:
            value = _normalize_scalar(args.get("answer"))
            if value is not None:
                return value
    policy = getattr(agent, "submission_policy", None)
    for candidate in reversed(getattr(policy, "candidates", []) or []):
        args = getattr(candidate, "args", {})
        if isinstance(args, Mapping) and "answer" in args:
            value = _normalize_scalar(args.get("answer"))
            if value is not None:
                return value
    return _normalize_scalar(last_labelled_scalar(str(text or "")))


def majority_labelled_scalar(answers: Iterable[str]) -> str:
    """Choose a stable majority among per-agent explicit scalar answers.

    Each agent contributes at most its last labelled value.  This prevents a
    long chain-of-thought from receiving extra votes merely because it repeats
    an intermediate value several times.  Ties are resolved by first
    appearance, matching the deterministic ordering used by the aggregators.
    """
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for answer in answers:
        value = last_labelled_scalar(str(answer or ""))
        if not value:
            continue
        # Canonicalize ``105.0`` and ``105`` while preserving non-integral
        # answers for environments whose schema is ``number``.
        try:
            numeric = float(value)
            key = str(int(numeric)) if numeric.is_integer() else format(numeric, "g")
        except ValueError:
            key = value.strip()
        counts[key] = counts.get(key, 0) + 1
        first_seen.setdefault(key, order)
        order += 1
    if not counts:
        return ""
    return max(counts, key=lambda key: (counts[key], -first_seen[key]))


def append_final_scalar(text: str, value: str) -> str:
    """Keep full reasoning while adding an unambiguous final scalar marker."""
    text = str(text or "")
    if not value:
        return text
    if re.search(
        rf"(?:final\s+(?:answer|score)|answer|score)\s*[:=\s]+"
        rf"(?:\*\*|__|`|\$|\s)*{re.escape(str(value))}(?!\d)",
        text,
        flags=re.IGNORECASE,
    ):
        return text
    return f"{text}\n\nFinal answer: {value}" if text.strip() else f"Final answer: {value}"


_PLAN_ACTION_NAMES = ("move", "smelt", "impossible", "stop")


def _extract_call_actions(text: str) -> list[str]:
    """Extract balanced tool calls, including calls spanning multiple lines."""
    source = str(text or "")
    actions: list[str] = []
    pattern = re.compile(r"\b(?:move|smelt|impossible|stop)\s*\(", re.I)
    for match in pattern.finditer(source):
        depth = 0
        quote: str | None = None
        escaped = False
        end = None
        for index in range(match.start(), len(source)):
            char = source[index]
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in "'\"":
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is not None:
            actions.append(re.sub(r"\s+", " ", source[match.start():end]).strip())
    return actions


def _extract_json_actions(text: str) -> list[str]:
    """Extract common JSON/tool-call action objects as a fallback."""
    source = str(text or "")
    candidates = re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", source, re.S)
    actions: list[str] = []
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        stack = [value]
        while stack:
            item = stack.pop(0)
            if isinstance(item, list):
                stack[0:0] = item
                continue
            if not isinstance(item, dict):
                continue
            name = item.get("action") or item.get("name") or item.get("tool")
            if isinstance(name, str) and name.lower() in _PLAN_ACTION_NAMES:
                actions.append(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            for nested in item.values():
                if isinstance(nested, (dict, list)):
                    stack.append(nested)
    return actions


def _plan_actions(text: str) -> list[str]:
    return _extract_call_actions(text) or _extract_json_actions(text)


def _plan_payload(text: str) -> str:
    """Extract a single executable payload without merging peer reasoning.

    PlanCraft tool traces are often embedded in otherwise verbose responses.
    Keep the complete response when no structured action list is available;
    callers still submit through the environment's schema-aware tools.
    """
    text = str(text or "").strip()
    if not text:
        return ""
    fenced = re.findall(r"```(?:json|python|text)?\s*(.*?)```", text, re.I | re.S)
    for block in fenced:
        if re.search(r"\b(?:move|smelt|impossible|stop)\s*\(", block, re.I):
            return block.strip()
    return text


def plan_is_structurally_complete(text: str) -> bool:
    """Return whether a candidate looks like one complete PlanCraft plan."""
    payload = _plan_payload(text)
    if not payload:
        return False
    return bool(_plan_actions(payload))


def normalized_action_sequence(text: str) -> str:
    """Canonicalize only the executable action portion of a plan response."""
    payload = _plan_payload(text)
    actions = _plan_actions(payload)
    return "\n".join(re.sub(r"\s+", " ", action).strip().lower() for action in actions)


def submission_adapter(output_type: OutputType, value: Any) -> Any:
    """Convert an aggregated value into the task's canonical submission shape."""
    if output_type == "executable_plan":
        payload = _plan_payload(str(value or ""))
        actions = _plan_actions(payload)
        return {
            "action": "plan",
            "plan": [re.sub(r"\s+", " ", action).strip() for action in actions],
        }
    if output_type == "scalar_exact":
        raw = str(value or "").strip()
        scalar = _normalize_scalar(raw) or last_labelled_scalar(raw)
        return scalar or None
    return str(value or "")


def plancraft_submission_from_env(env: Any, fallback: Any = "") -> dict[str, Any]:
    """Build the PlanCraft union submission from the selected env trace."""
    trace = list(getattr(env, "action_trace", []) or [])
    if trace:
        last = trace[-1]
        if last.get("tool") == "impossible":
            return {
                "action": "impossible",
                "reason": str((last.get("args") or {}).get("reason", "")),
            }
        return {
            "action": "plan",
            "plan": [
                {
                    "name": item.get("tool", ""),
                    "args": dict(item.get("args") or {}),
                }
                for item in trace
                if item.get("tool") in {"move", "smelt"}
            ],
        }
    adapted = submission_adapter("executable_plan", fallback)
    return adapted if isinstance(adapted, dict) else {"action": "plan", "plan": []}


def final_answer_from_text(text: str) -> str:
    """Extract a scalar final answer only when a tool explicitly asks for one.

    Only explicitly labelled scalar answers are valid. Unlabelled integers in
    reasoning are not submissions.
    """
    labelled = labelled_scalar_answers(text)
    if labelled:
        return labelled[-1]
    return ""


def labelled_answer_from_text(text: str) -> str:
    """Extract a free-form value from a conventional final-answer label."""
    matches = re.findall(
        r"(?:final\s+answer|answer)\s*[:=]\s*([^\r\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    return matches[-1].strip() if matches else text.strip()


def _schema_scalar_type(schema: Any) -> str | None:
    if not isinstance(schema, Mapping):
        return None
    direct = schema.get("type")
    if direct:
        return str(direct)
    for option in schema.get("anyOf", []) or []:
        option_type = _schema_scalar_type(option)
        if option_type and option_type != "null":
            return option_type
    return None


def submit_tool_call(env: Any, text: str, call_id: str) -> dict[str, Any] | None:
    """Build a generic terminal call from the selected tool's argument schema.

    ``reasoning`` tools receive the full answer text.  Tools with an explicit
    scalar ``answer`` field receive the labelled/last scalar answer.  This avoids
    the old policy of feeding arbitrary prose to environments that parse the
    first integer they encounter.
    """
    tool_name = None
    for candidate in ("submit_patch", "submit", "done"):
        if candidate in env.tools:
            tool_name = candidate
            break
    if tool_name is None:
        return None
    tool = env.tools[tool_name]
    schema = getattr(tool, "args_schema", None)
    properties: Mapping[str, Any] = {}
    required: set[str] = set()
    if isinstance(schema, Mapping):
        properties = schema.get("properties", schema.get("args", {})) or {}
        required = set(schema.get("required", []) or [])
    else:
        try:
            json_schema = schema.model_json_schema() if schema is not None else {}
            properties = json_schema.get("properties", {}) or {}
            required = set(json_schema.get("required", []) or [])
        except Exception:
            pass

    args: dict[str, Any]
    if "answer" in properties or "answer" in required:
        answer_schema = properties.get("answer", {})
        answer_type = _schema_scalar_type(answer_schema)
        if answer_type in {"integer", "number"}:
            answer = final_answer_from_text(text)
            if not answer:
                return None
            value: Any = int(answer) if answer_type == "integer" else float(answer)
        else:
            value = labelled_answer_from_text(text)
            if not value:
                return None
        args = {"answer": value}
        if "confidence_score" in properties or "confidence_score" in required:
            args["confidence_score"] = 100
    elif "reasoning" in properties or "reasoning" in required or not properties:
        args = {"reasoning": text}
    else:
        return None
    return {"name": tool_name, "args": args, "id": call_id, "type": "tool_call"}
