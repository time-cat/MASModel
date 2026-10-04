"""Decentralized multi-agent system: peer debate over d sequential rounds with consensus aggregation.

Implements Du et al. 2023 (arXiv:2305.14325) "Improving Factuality and Reasoning in
Language Models through Multi-Agent Debate" adapted to tool-using LLM agents.

Algorithm (matches the Methods section of the revised manuscript, Section 6.1):
    1. Round 1: each of N agents independently produces a candidate answer.
    2. Rounds 2..d: each agent receives the FULL set of peer responses from the
       previous round (no truncation) and is asked to critique / refine / replace
       its own answer.
    3. Consensus aggregation: majority vote over the final-round answers; the
       deterministic tie-break is the first-submitted answer that matches.
"""
import asyncio
import os.path as osp
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

from agent_scaling.agents.base import AgentSystemWithTools
from agent_scaling.agents.multiagent_utils.communication_strategy import (
    ConsensusStrategy,
    create_communication_strategy,
)
from agent_scaling.config.llm import LLMParams
from agent_scaling.datasets import DatasetInstance, DatasetInstanceOutputWithTrajectory
from agent_scaling.logger import logger
from agent_scaling.utils import write_yaml
from agent_scaling.budget import per_round_cap

from .multiagent_components.conversation import SubAgentRoundResult
from .multiagent_components.mas_subagent import WorkerSubagent
from .multiagent_components.memory import EnhancedMemory
from .multiagent_utils.aggregation import AggregationRequest, CandidateRecord, aggregate
from .multiagent_utils.result_selection import (
    append_final_scalar,
    explicit_scalar_candidate,
    first_successful_agent,
    last_external_answer,
    task_output_type,
    submit_tool_call,
)
from .registry import register_agent


@register_agent("multi-agent-decentralized")
class DecentralizedMultiAgentSystem(AgentSystemWithTools):
    """Decentralized multi-agent system with peer debate and consensus voting.

    No orchestrator. N peer agents iterate over budget-available sequential
    debate rounds, exchanging full prior-round responses. After the last round, a
    consensus vote over the agents' final answers selects the system output.

    `max_rounds` is a protocol horizon, not a worker-computation budget. The
    canonical configuration fixes it to 10 so that the lifetime worker budget
    is spread across communication rounds; passing ``None`` is supported for
    an explicitly unbounded protocol horizon.
    """

    required_prompts = ["subagent"]

    def __init__(
        self,
        *args,
        n_base_agents: int = 3,
        min_iterations_per_agent: int = 3,
        max_iterations_per_agent: int | None = None,
        max_rounds: int | None = 10,
        consensus_threshold: float = 0.5,
        total_decision_budget: int | None = 32,
        use_remaining_budget_for_verification: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory = EnhancedMemory()
        self.n_base_agents = n_base_agents
        self.min_iterations_per_agent = min_iterations_per_agent
        self.max_iterations_per_agent = max_iterations_per_agent
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.total_decision_budget = total_decision_budget
        self.use_remaining_budget_for_verification = (
            use_remaining_budget_for_verification
        )
        self.per_agent_decision_budget = (
            None
            if total_decision_budget is None
            else total_decision_budget // max(1, n_base_agents)
        )
        self.subagents: Dict[str, WorkerSubagent] = {}
        self.consensus = create_communication_strategy(
            "consensus", {"consensus_threshold": consensus_threshold}
        )

        logger.info(
            f"DecentralizedMultiAgentSystem: N={n_base_agents} agents, "
            f"d={max_rounds} debate rounds, consensus_threshold={consensus_threshold}"
        )

    def _create_subagents(self, task_instance: DatasetInstance):
        """Create N peer agents that will each produce a candidate answer per round."""
        self.subagents = {}
        for i in range(self.n_base_agents):
            agent_id = f"agent_{i + 1}"
            strategies = [
                "Analyze the problem systematically and produce a complete solution",
                "Explore the problem broadly, identify the root cause, then solve it",
                "Focus on the target outcome and work backwards to a solution",
            ]
            subagent = WorkerSubagent.init_from_agent(
                agent=self,
                agent_id=agent_id,
                objective="Independently produce your best answer to the task",
                original_query=self.memory.original_task,
                strategy=strategies[i % len(strategies)],
                task_instance=task_instance,
                min_iterations_per_agent=self.min_iterations_per_agent,
                max_iterations_per_agent=self.max_iterations_per_agent,
                decision_budget=self.per_agent_decision_budget,
                decision_budget_per_round=per_round_cap(
                    self.per_agent_decision_budget, self.max_rounds
                ),
                use_remaining_budget_for_verification=(
                    self.use_remaining_budget_for_verification
                ),
            )
            self.subagents[agent_id] = subagent
        logger.info(f"Created {len(self.subagents)} peer agents for debate")

    def _build_full_debate_context(
        self, current_agent_id: str, round_num: int
    ) -> str:
        """Build the FULL set of peer responses from the previous round (no truncation)."""
        peer_blocks: List[str] = []
        for agent_id, agent in self.subagents.items():
            if agent_id == current_agent_id:
                continue
            answer = agent.conv_history.last_outgoing_external_message
            if not answer:
                continue
            peer_blocks.append(
                f"--- Peer response from {agent_id} (round {round_num - 1}) ---\n"
                f"{answer}"
            )
        if not peer_blocks:
            return "(no peer responses available from the previous round)"
        return "\n\n".join(peer_blocks)

    def _round_prompt(self, current_agent_id: str, round_num: int) -> str:
        """Compose the message handed to an agent at the start of a debate round."""
        if round_num == 1:
            return (
                "Round 1: produce your best candidate answer to the task independently. "
                "Use the available tools and submit your final answer when ready."
            )
        peer_context = self._build_full_debate_context(current_agent_id, round_num)
        return (
            f"Debate round {round_num} of {self.max_rounds}.\n\n"
            "Below are your peers' answers from the previous round. Read them carefully. "
            "Identify points where you agree, points where they erred, and points where "
            "you missed something. Then produce your updated final answer for this round. "
            "You may defend your previous answer, refine it, or replace it.\n\n"
            f"{peer_context}\n\n"
            "Produce your updated final answer now."
        )

    def _consensus_vote(
        self, final_round_answers: List[Tuple[str, str]]
    ) -> Tuple[str, Optional[str]]:
        """Majority-vote over final-round answers using the ConsensusStrategy.

        Each distinct answer is submitted as a candidate finding; every agent
        then votes yes if it gave that answer, no otherwise. The
        ConsensusStrategy approves findings whose yes/total ratio meets the
        consensus_threshold. The winner is the candidate with the most yes
        votes; ties break to the first answer in iteration order.

        Returns (winning_answer, winning_agent_id). When no answers exist,
        returns ("", None).
        """
        non_empty = [(aid, ans) for aid, ans in final_round_answers if ans]
        if not non_empty:
            return "", None

        # Exact and executable contracts are delegated to the shared layer;
        # the debate/consensus object remains responsible only for exposing
        # peer findings and round state.
        if task_output_type(self) in {
            "scalar_exact", "executable_plan", "patch_or_state"
        }:
            result = aggregate(
                AggregationRequest(
                    output_type=task_output_type(self),
                    candidates=non_empty,
                    agents=tuple(self.subagents.values()),
                    plan_strategy="majority",
                )
            )
            return result.output, result.selected_agent

        # Vote on an explicit scalar value when one is present, rather than on
        # the entire chain-of-thought string.  Agents commonly reach the same
        # answer with different formatting or arithmetic explanations.
        answer_to_supporters: Dict[Tuple[str, str], List[str]] = {}
        representative: Dict[Tuple[str, str], str] = {}
        scalar_for_key: Dict[Tuple[str, str], str] = {}
        use_scalar_contract = task_output_type(self) == "scalar_exact"
        for aid, ans in non_empty:
            scalar = (
                explicit_scalar_candidate(self.subagents.get(aid), ans)
                if use_scalar_contract
                else ""
            )
            if scalar:
                try:
                    numeric = float(scalar)
                    normalized = (
                        str(int(numeric))
                        if numeric.is_integer()
                        else format(numeric, "g")
                    )
                except ValueError:
                    normalized = scalar.strip()
                key = ("scalar", normalized)
                scalar_for_key[key] = normalized
            else:
                # Preserve the original exact-string consensus semantics for
                # free-form answers (and its deterministic first-seen tie-break).
                key = ("text", ans)
            answer_to_supporters.setdefault(key, []).append(aid)
            representative.setdefault(key, ans)

        all_agent_ids = [aid for aid, _ in non_empty]
        for key, supporters in answer_to_supporters.items():
            ans = representative[key]
            proposer = supporters[0]
            finding_id = len(self.consensus.pending_findings)
            self.consensus.share_finding(
                proposer,
                {
                    "answer_excerpt": ans[:200],
                    "supporters": list(supporters),
                },
            )
            for aid in all_agent_ids:
                if aid == proposer:
                    continue  # proposer's yes vote is recorded by share_finding
                self.consensus.vote_on_finding(aid, finding_id, aid in supporters)
        self.consensus.synchronize()

        counts = Counter(
            {key: len(supporters) for key, supporters in answer_to_supporters.items()}
        )
        winning_key, vote_count = counts.most_common(1)[0]
        winning_agent = answer_to_supporters[winning_key][0]
        winning_answer = representative[winning_key]
        if use_scalar_contract and winning_key in scalar_for_key:
            winning_answer = append_final_scalar(
                winning_answer, scalar_for_key[winning_key]
            )
        share = vote_count / len(non_empty)
        logger.info(
            f"Consensus vote: '{winning_answer[:80]}...' wins with "
            f"{vote_count}/{len(non_empty)} votes ({share:.0%}); "
            f"threshold={self.consensus_threshold}; winner={winning_agent}"
        )
        return winning_answer, winning_agent

    def _auto_submit_consensus(
        self, consensus_answer: str, winning_agent_id: Optional[str]
    ) -> str:
        """Ensure the consensus answer is submitted via at least one agent's env.

        If `winning_agent_id`'s env has not yet submitted, call env.submit there.
        Otherwise the winning answer is already on record in that env.
        """
        if not consensus_answer:
            return ""

        target_id = winning_agent_id or next(iter(self.subagents), None)
        if target_id is None:
            return consensus_answer
        agent = self.subagents[target_id]
        env = agent.env
        if env.env_done():
            return consensus_answer

        tool_call = submit_tool_call(
            env, consensus_answer, "consensus_submit_decentralized"
        )
        if not tool_call:
            return consensus_answer

        logger.info(
            f"Submitting consensus answer via {target_id} using {tool_call['name']}"
        )
        try:
            tool_msg = env.execute_tool(tool_call)
            return str(tool_msg.content)
        except Exception as e:
            logger.warning(f"Consensus auto-submit failed: {e}")
            return consensus_answer

    def _select_environment_agent(self, winning_agent_id: Optional[str]):
        """Choose one worker as the source of all environment-backed outputs.

        Worker environments are isolated.  If the textual consensus winner did
        not complete the task, its answer cannot be paired with another
        worker's successful environment without producing a misleading result.
        """
        winning_agent = (
            self.subagents.get(winning_agent_id)
            if winning_agent_id is not None
            else None
        )
        if winning_agent is not None:
            status = winning_agent.env.env_status()
            if bool(getattr(status, "success", False)):
                return winning_agent

        successful_agent = first_successful_agent(self.subagents.values())
        if successful_agent is not None:
            return successful_agent
        if winning_agent is not None:
            return winning_agent
        return next(iter(self.subagents.values()), None)

    def run_agent(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
    ) -> DatasetInstanceOutputWithTrajectory:
        return asyncio.run(
            self.run_agent_async(instance, instance_dir, llm_params, instance_idx)
        )

    async def run_agent_async(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
    ) -> DatasetInstanceOutputWithTrajectory:
        start_time = time.time()

        # Reset per-instance consensus state. The canonical runner constructs a
        # fresh system per instance, so this is defensive only; it makes the
        # class safe under non-canonical reuse (e.g., a notebook reusing the
        # same instance across tasks, or future test harnesses), where stale
        # pending_findings/votes/approved_findings could otherwise leak across
        # calls.
        self.consensus = create_communication_strategy(
            "consensus", {"consensus_threshold": self.consensus_threshold}
        )

        shared_prompt_templates = self.get_dataset_prompt_templates(
            dataset_instance=instance
        )
        self.memory.original_task = shared_prompt_templates.get(
            "task_instance", str(instance.get_prompt_info())
        )

        self._create_subagents(instance)

        # Track each round's per-agent final answer for inspection / output.
        per_round_answers: List[List[Tuple[str, str]]] = []
        # Per-task time budget: only enforce if the dataset instance explicitly
        # sets a positive `time_limit`. Defaults to no limit so that the d
        # debate rounds always complete.
        time_limit = getattr(instance, "time_limit", None)

        round_num = 0
        while self.max_rounds is None or round_num < self.max_rounds:
            round_num += 1
            if time_limit is not None and time_limit > 0 and time.time() - start_time > time_limit:
                logger.warning(
                    f"Execution timeout reached before round {round_num}"
                )
                break
            logger.info(
                f"\n=== Decentralized debate round {round_num}/{self.max_rounds} ==="
            )

            # All d debate rounds run to completion. Agents whose environment
            # already terminated do not produce new responses, but their last
            # answer is preserved in conv_history and re-emitted in the
            # per-round snapshot below, so the final consensus vote always
            # sees every agent's contribution regardless of individual
            # termination.
            active_agents = [
                aid
                for aid, a in self.subagents.items()
                if a.conv_history.status == "active"
                and not a.should_stop_due_to_rate_limiting()
            ]

            if not active_agents:
                logger.info("No active agents remain; stopping debate")
                break

            tasks: List[Tuple[str, asyncio.Task]] = []
            for agent_id in active_agents:
                subagent = self.subagents[agent_id]
                msg = self._round_prompt(agent_id, round_num)
                task = asyncio.create_task(
                    asyncio.to_thread(subagent.process_orchestrator_message, msg)
                )
                tasks.append((agent_id, task))

            # No asyncio.wait timeout — let each debate round complete fully so
            # the consensus aggregation sees real answers from every agent.
            if tasks:
                done, pending = await asyncio.wait(
                    [t for _, t in tasks], return_when=asyncio.ALL_COMPLETED
                )
                for t in pending:
                    t.cancel()
            else:
                done, pending = set(), set()

            round_results: Dict[str, SubAgentRoundResult] = {}
            for agent_id, task in tasks:
                if task in done:
                    try:
                        round_results[agent_id] = await task
                    except Exception as e:
                        logger.warning(
                            f"Agent {agent_id} failed in round {round_num}: {e}"
                        )

            for agent_id, result in round_results.items():
                if result.findings:
                    self.memory.add_findings(agent_id, result.findings)

            # Snapshot each agent's last outgoing answer at this round's close
            round_answers: List[Tuple[str, str]] = []
            for agent_id in self.subagents:
                ans = self.subagents[agent_id].conv_history.last_outgoing_external_message or ""
                round_answers.append((agent_id, ans))
            per_round_answers.append(round_answers)

            # NOTE: we do NOT exit early when any single agent's env is done.
            # All `d` debate rounds run so the consensus aggregation has full
            # information from each agent. Agents whose env terminated early
            # simply keep their last answer in subsequent rounds.

        # Consensus aggregation over the FINAL round's per-agent answers.
        final_round_answers = per_round_answers[-1] if per_round_answers else []
        consensus_answer, winning_agent = self._consensus_vote(final_round_answers)

        # Make sure the consensus answer reaches an environment for scoring.
        submission_response = self._auto_submit_consensus(
            consensus_answer, winning_agent
        )

        output_type = task_output_type(self)
        environment_backed_output = output_type in {"executable_plan", "patch_or_state"}

        # Pick the canonical env_status using the existing fallback behavior.
        final_env_status = None
        if winning_agent is not None:
            final_env_status = self.subagents[winning_agent].env.env_status()
        if not getattr(final_env_status, "success", False):
            successful_agent = first_successful_agent(self.subagents.values())
            if successful_agent is not None:
                final_env_status = successful_agent.env.env_status()
        if final_env_status is None:
            for agent in self.subagents.values():
                if agent.env.env_done():
                    final_env_status = agent.env.env_status()
                    break
        if final_env_status is None:
            for agent in self.subagents.values():
                final_env_status = agent.env.env_status()
                break

        final_answer = consensus_answer or submission_response
        selected_agent = self.subagents.get(winning_agent) if winning_agent else None
        if selected_agent is None:
            selected_agent = first_successful_agent(self.subagents.values())
        selected_env = selected_agent.env if selected_agent is not None else None

        if environment_backed_output:
            # Keep status, trajectory, and canonical submission attached to the
            # same worker.  A successful fallback worker must also supply the
            # final text; otherwise the log would describe one worker while
            # the evaluator scores another worker's environment.
            selected_agent = self._select_environment_agent(winning_agent)
            selected_env = selected_agent.env if selected_agent is not None else None
            final_env_status = (
                selected_env.env_status() if selected_env is not None else None
            )
            if selected_agent is not None and selected_agent.agent_id != winning_agent:
                selected_answer = last_external_answer(selected_agent)
                if selected_answer:
                    final_answer = selected_answer
            if not final_answer and selected_agent is not None:
                final_answer = last_external_answer(selected_agent)

        aggregation_candidates = (
            [CandidateRecord(selected_agent.agent_id, final_answer, agent=selected_agent)]
            if final_answer and selected_agent is not None
            else final_round_answers
        )
        aggregation_result = aggregate(
            AggregationRequest(
                output_type=output_type,
                candidates=aggregation_candidates,
                agents=tuple(self.subagents.values()),
                plan_strategy="majority",
            )
        )
        canonical_submission = aggregation_result.canonical_submission

        execution_time = time.time() - start_time
        total_decision_calls = sum(
            a.budget.snapshot()["used_decision_calls"] for a in self.subagents.values()
        )
        logger.info(
            f"Decentralized debate completed in {execution_time:.2f}s with "
            f"{total_decision_calls} worker decisions across {len(self.subagents)} agents "
            f"and {len(per_round_answers)} debate rounds; winner={winning_agent}"
        )

        if instance_dir is not None:
            output_data = {
                "architecture": "decentralized",
                "algorithm": "multi_agent_debate_with_consensus",
                "n_agents": self.n_base_agents,
                "max_rounds": self.max_rounds,
                "per_round_decision_budget": per_round_cap(
                    self.per_agent_decision_budget, self.max_rounds
                ),
                "rounds_executed": len(per_round_answers),
                "consensus_threshold": self.consensus_threshold,
                "total_decision_budget": self.total_decision_budget,
                "total_decision_calls": total_decision_calls,
                "per_agent_budgets": {
                    aid: agent.budget.snapshot()
                    for aid, agent in self.subagents.items()
                },
                "per_agent_verification": {
                    aid: agent.submission_policy.snapshot()
                    for aid, agent in self.subagents.items()
                },
                "winning_agent": winning_agent,
                "selected_agent": (
                    selected_agent.agent_id if selected_agent is not None else None
                ),
                "canonical_agent_output": final_answer,
                "raw_candidates": {
                    aid: ans for aid, ans in final_round_answers
                },
                "aggregated_output": final_answer,
                "canonical_submission": canonical_submission,
                "candidate_submissions": aggregation_result.candidate_submissions,
                "vote_counts": aggregation_result.vote_counts,
                "tie_break_reason": aggregation_result.tie_break_reason,
                "aggregation_invalid_reason": aggregation_result.invalid_reason,
                "output_type": task_output_type(self),
                "evaluation_semantics": (
                    "success_environment_state_selection"
                    if task_output_type(self) in {"executable_plan", "patch_or_state"}
                    else "task_contract_submission"
                ),
                "selected_environment_trajectory": (
                    getattr(selected_env, "get_action_trace", lambda: [])()
                    if selected_env is not None
                    else []
                ),
                "total_iterations": total_decision_calls,
                "execution_time": execution_time,
                "agent_findings": {
                    aid: findings
                    for aid, findings in self.memory.agent_findings.items()
                },
                "per_round_answers": [
                    {aid: ans for aid, ans in round_pairs}
                    for round_pairs in per_round_answers
                ],
                "consensus_record": {
                    "approved_findings": self.consensus.approved_findings,
                    "pending_findings": self.consensus.pending_findings,
                },
            }
            write_yaml(
                output_data,
                osp.join(instance_dir, "multi_agent_output.yaml"),
                use_long_str_representer=True,
                truncate_floats=False,
            )

        return DatasetInstanceOutputWithTrajectory(
            data_instance=instance,
            agent_output=final_answer,
            canonical_submission=canonical_submission,
            trajectory=(
                getattr(selected_env, "get_action_trace", lambda: [])()
                if selected_env is not None
                else []
            ),
            final_env_output=final_env_status,
        )
