"""Regression tests for architecture-independent MAS result plumbing."""

from types import SimpleNamespace

from agent_scaling.agents.multiagent_utils.result_selection import (
    final_answer_from_text,
    first_successful_agent,
    submit_tool_call,
)


def _agent(success: bool):
    status = SimpleNamespace(success=success, num_steps=1)
    return SimpleNamespace(
        env=SimpleNamespace(env_status=lambda: status, env_done=lambda: success)
    )


def test_first_successful_agent_is_stable_and_does_not_score_answers():
    first = _agent(False)
    second = _agent(True)
    assert first_successful_agent([first, second]) is second


def test_final_answer_prefers_labelled_answer_and_last_integer_fallback():
    assert final_answer_from_text("b0=43; Final Answer: 877") == "877"
    assert final_answer_from_text("computed values 43, 14, 67") == "67"


class _Tool:
    def __init__(self, schema):
        self.args_schema = schema


class _AnswerSchema:
    @classmethod
    def model_json_schema(cls):
        return {"properties": {"answer": {"type": "integer"}}, "required": ["answer"]}


class _ReasoningSchema:
    @classmethod
    def model_json_schema(cls):
        return {"properties": {"reasoning": {"type": "string"}}, "required": ["reasoning"]}


class _StringAnswerSchema:
    @classmethod
    def model_json_schema(cls):
        return {
            "properties": {
                "answer": {"type": "string"},
                "confidence_score": {"type": "integer"},
            },
            "required": ["answer", "confidence_score"],
        }


def test_submit_tool_call_uses_structured_answer_instead_of_first_integer():
    env = SimpleNamespace(
        tools={"submit": _Tool(_AnswerSchema)},
    )
    call = submit_tool_call(env, "b0=43\nFinal Answer: 877", "test")
    assert call is not None
    assert call["args"] == {"answer": 877}


def test_submit_tool_call_preserves_full_reasoning_for_text_tasks():
    text = "Final Answer: a long task response with details"
    env = SimpleNamespace(tools={"submit": _Tool(_ReasoningSchema)})
    call = submit_tool_call(env, text, "test")
    assert call is not None
    assert call["args"] == {"reasoning": text}


def test_submit_tool_call_supports_string_answer_tools():
    env = SimpleNamespace(tools={"done": _Tool(_StringAnswerSchema)})
    call = submit_tool_call(env, "Final Answer: Paris", "test")
    assert call is not None
    assert call["args"] == {"answer": "Paris", "confidence_score": 100}
