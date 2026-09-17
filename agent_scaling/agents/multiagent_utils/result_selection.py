"""Shared result propagation helpers for multi-agent systems.

The worker environments are intentionally isolated, so an MAS needs an explicit
policy for choosing the answer exposed to the dataset evaluator and the matching
environment status.  These helpers keep that plumbing architecture-independent;
they do not compare or score answers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Iterable


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


def final_answer_from_text(text: str) -> str:
    """Extract a scalar final answer only when a tool explicitly asks for one.

    Labelled answers are preferred.  The last integer fallback is deliberately
    used only for structured ``answer`` parameters, never for free-form
    ``reasoning`` parameters.
    """
    labelled = re.findall(
        r"(?:final\s+(?:answer|score)|answer|score)\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if labelled:
        return labelled[-1]
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else ""


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
