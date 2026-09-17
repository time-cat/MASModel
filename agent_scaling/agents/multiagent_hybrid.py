"""Hybrid multi-agent system: centralized coordination + peer-to-peer communication."""
import asyncio
import os.path as osp
import time
from typing import Dict, List, Optional

from agent_scaling.agents.base import AgentSystemWithTools
from agent_scaling.config.llm import LLMParams
from agent_scaling.datasets import DatasetInstance, DatasetInstanceOutputWithTrajectory
from agent_scaling.logger import logger
from agent_scaling.utils import write_yaml

from .multiagent_components.conversation import OrchestrationResult, SubAgentRoundResult
from .multiagent_components.mas_lead_agent import LeadAgent
from .multiagent_components.mas_subagent import WorkerSubagent
from .multiagent_components.memory import EnhancedMemory
from .multiagent_utils.result_selection import (
    first_nonempty_answer,
    first_successful_agent,
    last_external_answer,
    submit_tool_call,
)
from .registry import register_agent


@register_agent("multi-agent-hybrid")
class HybridMultiAgentSystem(AgentSystemWithTools):
    """Hybrid multi-agent system: orchestrator coordinates + agents share peer findings.

    Combines centralized orchestration with peer-to-peer knowledge sharing.
    The lead agent plans and coordinates, while agents directly receive
    findings from peers in their prompts (not just orchestrator summaries).
    """

    required_prompts = ["lead_agent", "subagent"]

    def __init__(
        self,
        *args,
        n_base_agents: int = 3,
        min_iterations_per_agent: int = 3,
        max_iterations_per_agent: int | None = None,
        enable_peer_communication: bool = True,
        total_decision_budget: int | None = 32,
        use_remaining_budget_for_verification: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory = EnhancedMemory()
        self.n_base_agents = n_base_agents
        self.min_iterations_per_agent = min_iterations_per_agent
        self.max_iterations_per_agent = max_iterations_per_agent
        self.enable_peer_communication = enable_peer_communication
        self.total_decision_budget = total_decision_budget
        per_agent_budget = (
            None
            if total_decision_budget is None
            else total_decision_budget // max(1, n_base_agents)
        )

        self.lead_agent = LeadAgent(
            *args,
            memory=self.memory,
            min_iterations_per_agent=min_iterations_per_agent,
            max_iterations_per_agent=max_iterations_per_agent,
            decision_budget_per_agent=per_agent_budget,
            num_base_agents=n_base_agents,
            max_rounds=kwargs.get("max_rounds", 10),
            max_execution_time=kwargs.get("max_execution_time", 600),
            use_remaining_budget_for_verification=(
                use_remaining_budget_for_verification
            ),
            domain_config={"task_blurb": kwargs.get("task_blurb", "hybrid coordinator")},
            **{k: v for k, v in kwargs.items() if k not in ("max_rounds", "max_execution_time", "task_blurb")},
        )

        logger.info(
            f"HybridMultiAgentSystem: {n_base_agents} agents, "
            f"peer_communication={enable_peer_communication}"
        )

    def _auto_submit(self, result: OrchestrationResult, synthesized_answer: str) -> str:
        """Auto-submit after orchestration if no subagent already submitted."""
        for agent_id, agent in self.lead_agent.subagents.items():
            env = agent.env
            if env.env_done():
                continue
            tool_call = submit_tool_call(
                env, synthesized_answer or "Hybrid multi-agent synthesis",
                "auto_submit_hybrid",
            )
            if tool_call:
                logger.info(
                    f"Auto-submitting via {agent_id} using {tool_call['name']}"
                )
                try:
                    tool_msg = env.execute_tool(tool_call)
                    return str(tool_msg.content)
                except Exception as e:
                    logger.warning(f"Auto submission via {agent_id} failed: {e}")
        return synthesized_answer

    def _inject_peer_findings(
        self, agent: WorkerSubagent, all_subagents: Dict[str, WorkerSubagent]
    ) -> str:
        """Build direct peer-to-peer context for an agent.

        Unlike centralized which only passes orchestrator summaries,
        hybrid injects raw peer findings directly into agent context.
        """
        if not self.enable_peer_communication:
            return ""

        peer_sections = []
        for other_id, other_agent in all_subagents.items():
            if other_id == agent.agent_id:
                continue
            # Get the peer's actual findings (not orchestrator summary)
            peer_findings = other_agent.conv_history.last_outgoing_external_message
            if peer_findings and len(peer_findings.strip()) > 10:
                peer_sections.append(
                    f"[PEER {other_id}]: {peer_findings[:400]}"
                )

        if peer_sections:
            return (
                "\n\n## Direct Peer Findings (shared peer-to-peer)\n"
                + "\n".join(peer_sections)
            )
        return ""

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
        llm_params_dict = llm_params.model_dump() if llm_params else {}

        logger.info("Starting hybrid multi-agent processing")

        # Use task-specific time limit if available
        self.lead_agent.max_execution_time = getattr(instance, "time_limit", 600)
        # Use lead agent's orchestration for planning
        processing_result: OrchestrationResult = await self.lead_agent.orchestrate_work(
            task_instance=instance,
            llm_params_dict=llm_params_dict,
        )

        final_answer = processing_result.synthesized_answer

        # After orchestration rounds, run a peer-enhanced final round
        # where agents get direct peer findings
        if self.enable_peer_communication:
            active_agents = [
                aid for aid, a in self.lead_agent.subagents.items()
                if not a.env.env_done()
            ]
            if active_agents:
                logger.info(f"Running peer-enhanced round with {len(active_agents)} agents")
                tasks = []
                for agent_id in active_agents:
                    subagent = self.lead_agent.subagents[agent_id]
                    peer_context = self._inject_peer_findings(
                        subagent, self.lead_agent.subagents
                    )
                    message = (
                        f"Final round with peer insights. Review your peers' work "
                        f"and finalize your solution.{peer_context}"
                    )
                    task = asyncio.create_task(
                        asyncio.to_thread(
                            subagent.process_orchestrator_message, message
                        )
                    )
                    tasks.append((agent_id, task))

                done, pending = await asyncio.wait(
                    [t for _, t in tasks], timeout=300
                )
                for t in pending:
                    t.cancel()

                for agent_id, task in tasks:
                    if task in done:
                        try:
                            result = await task
                            if result.findings:
                                self.memory.add_findings(agent_id, result.findings)
                        except Exception as e:
                            logger.warning(f"Peer round agent {agent_id} failed: {e}")

        agents = list(self.lead_agent.subagents.values())
        successful_agent = first_successful_agent(agents)
        if not final_answer:
            final_answer = (
                last_external_answer(successful_agent)
                if successful_agent is not None
                else first_nonempty_answer(agents)
            )

        # Auto-submit only when no worker reached a terminal state, using the
        # terminal tool's schema to preserve task-specific answer semantics.
        any_done = any(agent.env.env_done() for agent in agents)
        if not any_done:
            # If synthesis is empty, build reasoning from sub-agent findings
            submit_reasoning = final_answer or ""
            if not submit_reasoning.strip():
                all_findings = []
                for agent_id, findings in self.memory.agent_findings.items():
                    for f in findings:
                        if f and len(f.strip()) > 20:
                            all_findings.append(f"{agent_id}: {f[:200]}")
                if all_findings:
                    submit_reasoning = "Multi-agent findings:\n" + "\n".join(all_findings[-3:])
                else:
                    submit_reasoning = "Hybrid multi-agent synthesis (no explicit findings collected)"
                logger.warning(
                    f"Synthesis was empty, using sub-agent findings for auto-submit: {submit_reasoning[:100]}..."
                )
            submission_response = self._auto_submit(
                processing_result, submit_reasoning
            )
            if not final_answer:
                final_answer = submission_response

        execution_time = time.time() - start_time
        total_decision_calls = sum(
            a.budget.snapshot()["used_decision_calls"]
            for a in self.lead_agent.subagents.values()
        )
        logger.info(
            f"Hybrid processing completed in {execution_time:.2f}s "
            f"with {total_decision_calls} worker decisions across "
            f"{len(self.lead_agent.subagents)} agents"
        )

        if instance_dir is not None:
            output_data = processing_result.model_dump()
            output_data["canonical_agent_output"] = final_answer
            output_data["canonical_agent_id"] = (
                next(
                    (
                        aid
                        for aid, agent in self.lead_agent.subagents.items()
                        if agent is successful_agent
                    ),
                    None,
                )
                if successful_agent is not None
                else None
            )
            output_data["total_decision_budget"] = self.total_decision_budget
            output_data["total_decision_calls"] = total_decision_calls
            output_data["max_rounds"] = self.lead_agent.max_rounds
            output_data["per_round_decision_budget"] = (
                self.lead_agent.decision_budget_per_round
            )
            output_data["per_agent_budgets"] = {
                aid: agent.budget.snapshot()
                for aid, agent in self.lead_agent.subagents.items()
            }
            output_data["per_agent_verification"] = {
                aid: agent.submission_policy.snapshot()
                for aid, agent in self.lead_agent.subagents.items()
            }
            write_yaml(
                output_data,
                osp.join(instance_dir, "multi_agent_output.yaml"),
                use_long_str_representer=True,
                truncate_floats=False,
            )

        # Get final env status
        final_env_status = (
            successful_agent.env.env_status() if successful_agent is not None else None
        )
        if final_env_status is None:
            for agent in agents:
                if agent.env.env_done():
                    final_env_status = agent.env.env_status()
                    break
        if final_env_status is None:
            final_env_status = processing_result.combined_env_status

        return DatasetInstanceOutputWithTrajectory(
            data_instance=instance,
            agent_output=final_answer,
            trajectory=[],
            final_env_output=final_env_status,
        )
