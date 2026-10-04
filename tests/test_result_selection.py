"""Regression tests for architecture-independent MAS result plumbing."""

from types import SimpleNamespace

from agent_scaling.agents.multiagent_utils.result_selection import (
    final_answer_from_text,
    first_successful_agent,
    plan_is_structurally_complete,
    submission_adapter,
    submit_tool_call,
)
from agent_scaling.agents.multiagent_utils.aggregation import (
    AggregationRequest,
    aggregate,
)


def _agent(success: bool, agent_id: str = ""):
    status = SimpleNamespace(success=success, num_steps=1)
    return SimpleNamespace(
        agent_id=agent_id,
        env=SimpleNamespace(env_status=lambda: status, env_done=lambda: success)
    )


def test_first_successful_agent_is_stable_and_does_not_score_answers():
    first = _agent(False)
    second = _agent(True)
    assert first_successful_agent([first, second]) is second


def test_final_answer_requires_an_explicit_label():
    assert final_answer_from_text("b0=43; Final Answer: 877") == "877"
    assert final_answer_from_text("computed values 43, 14, 67") == ""


def test_scalar_parser_does_not_use_later_node_ids():
    text = "The final score is 150. I still need to inspect N2."
    assert final_answer_from_text(text) == "150"


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


def test_submit_tool_call_refuses_unlabelled_integer_from_reasoning():
    env = SimpleNamespace(tools={"submit": _Tool(_AnswerSchema)})
    assert submit_tool_call(env, "I inspected N3 but have not finished", "test") is None


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


def test_executable_plan_aggregation_never_concatenates_candidates():
    move = 'move(slot_from="[I1]", slot_to="[I2]", quantity=1)'
    smelt = 'smelt(slot_from="[I1]", slot_to="[I3]", quantity=1)'
    result = aggregate(AggregationRequest(
        output_type="executable_plan",
        candidates=[("agent_1", move), ("agent_2", smelt)],
    ))
    selected, agent_id = result.output, result.selected_agent
    assert agent_id == "agent_1"
    assert selected == move
    assert "smelt" not in selected
    assert plan_is_structurally_complete(selected)
    assert submission_adapter("executable_plan", selected) == {
        "action": "plan",
        "plan": [move],
    }


def test_independent_plan_strategy_prefers_success_then_structure():
    incomplete = "I will solve this soon"
    multiline = 'move(\n  slot_from="[I1]",\n  slot_to="[I2]",\n  quantity=1\n)'
    result = aggregate(AggregationRequest(
        output_type="executable_plan",
        candidates=[("agent_1", incomplete), ("agent_2", multiline)],
        agents=(_agent(False, "agent_1"), _agent(True, "agent_2")),
        plan_strategy="successful",
    ))
    selected, agent_id = result.output, result.selected_agent
    assert selected.startswith("move(")
    assert plan_is_structurally_complete(selected)
    assert agent_id == "agent_2"


def test_plan_adapter_accepts_json_action_schema():
    value = '{"action":"move","arguments":{"slot_from":"[I1]","slot_to":"[I2]","quantity":1}}'
    assert submission_adapter("executable_plan", value)["plan"]


def test_scalar_adapter_is_contract_specific():
    assert submission_adapter("scalar_exact", "reasoning\nFinal answer: 105") == "105"
    assert submission_adapter("scalar_exact", "No candidate was submitted; N3 pending") is None
    assert submission_adapter("free_text", "Final answer: 105") == "Final answer: 105"


def test_scalar_aggregation_prefers_terminal_tool_argument():
    first = _agent(False, "agent_1")
    first.last_terminal_tool_call = {"name": "submit", "args": {"answer": 105}}
    result = aggregate(AggregationRequest(
        output_type="scalar_exact",
        candidates=[("agent_1", "No explicit answer; N3 pending"), ("agent_2", "Final answer: 5")],
        agents=(first, _agent(False, "agent_2")),
    ))
    selected, agent_id = result.output, result.selected_agent
    assert agent_id == "agent_1"
    assert "Final answer: 105" in selected


def test_scalar_aggregation_without_candidate_is_explicitly_invalid():
    result = aggregate(AggregationRequest(
        output_type="scalar_exact",
        candidates=[("agent_1", "I inspected N3 but did not finish")],
    ))
    assert result.canonical_submission is None
    assert result.invalid_reason == "no_valid_candidate"
    assert result.candidate_submissions == {"agent_1": None}


def test_scalar_tie_reason_is_auditable():
    result = aggregate(AggregationRequest(
        output_type="scalar_exact",
        candidates=[("agent_1", "Final answer: 10"), ("agent_2", "Final answer: 20")],
    ))
    assert result.canonical_submission is None
    assert result.tie_break_reason == "tie_abstained"
    assert result.vote_counts == {"10": 1, "20": 1}


def test_free_text_is_never_reparsed_as_a_scalar():
    text = "The report contains revenue 105 and should remain unchanged."
    result = aggregate(AggregationRequest(
        output_type="free_text", candidates=[("agent_1", text)]
    ))
    assert result.canonical_submission == text
