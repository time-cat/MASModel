import asyncio
import os.path as osp
import time
from typing import Optional

from agent_scaling.agents.base import AgentSystemWithTools
from agent_scaling.config.llm import LLMParams
from agent_scaling.datasets import DatasetInstance, DatasetInstanceOutputWithTrajectory
from agent_scaling.logger import logger
from agent_scaling.utils import write_yaml

from .multiagent_components.conversation import OrchestrationResult
from .multiagent_components.mas_lead_agent import LeadAgent
from .multiagent_components.memory import EnhancedMemory
from .registry import register_agent


@register_agent("multi-agent-centralized")
class CentralizedMultiAgentSystem(AgentSystemWithTools):
    """Centralized multi-agent system with orchestrator coordinating workers"""

    required_prompts = ["lead_agent", "subagent"]

    def __init__(
        self,
        *args,
        n_base_agents: int = 3,
        min_iterations_per_agent: int = 3,
        max_iterations_per_agent: int | None = None,
        total_decision_budget: int | None = 32,
        use_remaining_budget_for_verification: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory = EnhancedMemory()

        self.n_base_agents = n_base_agents
        self.min_iterations_per_agent = min_iterations_per_agent
        self.max_iterations_per_agent = max_iterations_per_agent
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
            domain_config={"task_blurb": kwargs.get("task_blurb", "task coordinator")},
            **{k: v for k, v in kwargs.items() if k not in ("max_rounds", "max_execution_time", "task_blurb")},
        )

        logger.info(
            f"CentralizedMultiAgentSystem initialized with: {n_base_agents} agents, {min_iterations_per_agent} min iterations per agent (adaptive orchestration)"
        )
        logger.info(
            "Using prompt compilation with dataset-shared templates like single-agent system"
        )

    def _auto_submit(
        self, result: OrchestrationResult, synthesized_answer: str
    ) -> str:
        """Auto-submit after orchestration if no subagent already submitted.

        Uses agent_1's environment to trigger the submit/submit_patch tool,
        passing the synthesized answer as reasoning.
        """
        # Pick the first subagent's environment for submission
        for agent_id, agent in self.lead_agent.subagents.items():
            env = agent.env
            reasoning = (synthesized_answer or "Multi-agent synthesis")[:500]
            if env.env_done():
                break
            # Use execute_tool with a synthetic ToolCall
            submit_tool_name = None
            if "submit_patch" in env.tools:
                submit_tool_name = "submit_patch"
            elif "submit" in env.tools:
                submit_tool_name = "submit"
            if submit_tool_name:
                logger.info(f"Auto-submitting via {agent_id} using {submit_tool_name}")
                try:
                    tool_call = {
                        "name": submit_tool_name,
                        "args": {"reasoning": reasoning},
                        "id": "auto_submit",
                        "type": "tool_call",
                    }
                    tool_msg = env.execute_tool(tool_call)
                    return str(tool_msg.content)
                except Exception as e:
                    logger.warning(f"Auto {submit_tool_name} failed: {e}")
            break  # Only try the first subagent
        return synthesized_answer

    def run_agent(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
    ) -> DatasetInstanceOutputWithTrajectory:
        """Synchronous wrapper for backward compatibility"""
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
        """Async version of run_agent for better performance"""
        start_time = time.time()
        logger.info(f"Starting multi-agent processing for instance {instance_idx}")
        logger.info(
            f"Configuration: {self.n_base_agents} agents, {self.min_iterations_per_agent} min iterations per agent"
        )
        # Process llm_params like single_agent.py
        llm_params_dict = llm_params.model_dump() if llm_params else {}

        # Use task-specific time limit if available
        self.lead_agent.max_execution_time = getattr(instance, "time_limit", 600)
        logger.info("Starting lead agent orchestration...")
        processing_result: OrchestrationResult = await self.lead_agent.orchestrate_work(
            task_instance=instance,
            llm_params_dict=llm_params_dict,
        )

        final_answer = processing_result.synthesized_answer
        if final_answer is not None:
            logger.info(
                f"Final answer extracted: {final_answer[:200]}..."
                if len(str(final_answer)) > 200
                else f"Final answer extracted: {final_answer}"
            )

        # Auto-submit if no subagent already submitted
        # This ensures evaluation runs even when subagents only explored
        any_done = any(
            agent.env.env_done()
            for agent in self.lead_agent.subagents.values()
        )
        if not any_done:
            # If synthesis is empty, build reasoning from sub-agent findings
            submit_reasoning = final_answer or ""
            if not submit_reasoning.strip():
                # Gather findings from all sub-agents as fallback reasoning
                all_findings = []
                for agent_id, findings in self.memory.agent_findings.items():
                    for f in findings:
                        if f and len(f.strip()) > 20:
                            all_findings.append(f"{agent_id}: {f[:200]}")
                if all_findings:
                    submit_reasoning = "Multi-agent findings:\n" + "\n".join(all_findings[-3:])
                else:
                    submit_reasoning = "Multi-agent synthesis (no explicit findings collected)"
                logger.warning(
                    f"Synthesis was empty, using sub-agent findings for auto-submit: {submit_reasoning[:100]}..."
                )
            final_answer = self._auto_submit(
                processing_result, submit_reasoning
            )

        execution_time = time.time() - start_time
        total_decision_calls = sum(
            agent.budget.snapshot()["used_decision_calls"]
            for agent in self.lead_agent.subagents.values()
        )

        logger.info(
            f"Processing completed in {execution_time:.2f}s with "
            f"{total_decision_calls} worker decisions across "
            f"{len(processing_result.subagent_conversations)} agents"
        )

        if instance_dir is not None:
            output_data = processing_result.model_dump()
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

        # Re-fetch env status after potential auto-submit
        final_env_status = None
        for agent_id, agent in self.lead_agent.subagents.items():
            if agent.env.env_done():
                final_env_status = agent.env.env_status()
                break
        if final_env_status is None:
            final_env_status = processing_result.combined_env_status

        # Return DatasetInstanceOutputWithTrajectory like single_agent.py
        return DatasetInstanceOutputWithTrajectory(
            data_instance=instance,
            agent_output=final_answer,
            trajectory=[],  # Multi-agent doesn't have a single trajectory
            final_env_output=final_env_status,
        )
