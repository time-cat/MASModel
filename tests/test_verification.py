"""Tests for the shared ReAct deferred-submission policy."""

from agent_scaling.verification import DeferredSubmissionPolicy


def _call(name: str, **args):
    return {"name": name, "args": args, "id": "call-1", "type": "tool_call"}


def test_non_terminal_tool_is_never_deferred():
    policy = DeferredSubmissionPolicy()

    assert policy.defer_if_needed(_call("inspect", target="x"), 3) is None
    assert policy.candidates == []


def test_terminal_tool_executes_on_final_decision():
    policy = DeferredSubmissionPolicy()

    assert policy.defer_if_needed(_call("submit", answer=42), 0) is None
    assert policy.candidates == []


def test_early_terminal_tool_becomes_generic_verification_pass():
    policy = DeferredSubmissionPolicy()

    feedback = policy.defer_if_needed(_call("submit", answer=42), 2)

    assert feedback is not None
    assert "original task" in feedback
    assert "complete trajectory" in feedback
    assert "evidence" in feedback
    assert "node" not in feedback.lower()
    assert "code" not in feedback.lower()
    assert policy.snapshot()["deferred_terminal_calls"] == 1


def test_latest_candidate_is_used_for_budget_exhaustion_fallback():
    policy = DeferredSubmissionPolicy()
    first = {"answer": 41, "reasoning": "first"}
    second = {"answer": 42, "reasoning": "verified"}

    policy.defer_if_needed(_call("submit", **first), 2)
    policy.defer_if_needed(_call("submit", **second), 1)
    fallback = policy.latest_tool_call(call_id="fallback")

    assert fallback == {
        "name": "submit",
        "args": second,
        "id": "fallback",
        "type": "tool_call",
    }


def test_candidates_are_copied_and_policy_can_be_disabled():
    args = {"answer": {"value": 42}}
    call = _call("done", **args)
    policy = DeferredSubmissionPolicy()
    policy.defer_if_needed(call, 1)
    args["answer"]["value"] = 0

    assert policy.candidates[0].args["answer"]["value"] == 42
    assert DeferredSubmissionPolicy(enabled=False).defer_if_needed(call, 3) is None


def test_each_budget_above_first_candidate_adds_one_verification_pass():
    # Models on a threshold-5 task first propose submit on decision 5. The
    # policy converts exactly the additional B-5 decisions into verification.
    for total_budget, expected_passes in ((5, 0), (6, 1), (7, 2)):
        policy = DeferredSubmissionPolicy()
        for used_calls in range(5, total_budget + 1):
            policy.defer_if_needed(
                _call("submit", answer=used_calls),
                total_budget - used_calls,
            )

        assert len(policy.candidates) == expected_passes
