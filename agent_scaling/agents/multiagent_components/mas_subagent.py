import threading
import traceback
from typing import Any, cast

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.utils import convert_to_openai_messages

from agent_scaling.agents.base import BaseAgentWithTools
from agent_scaling.datasets import DatasetInstance
from agent_scaling.logger import logger
from agent_scaling.budget import BudgetLedger
from agent_scaling.verification import DeferredSubmissionPolicy

from .conversation import SubAgentConversationHistory, SubAgentRoundResult


class WorkerSubagent(BaseAgentWithTools):
    """Generic worker subagent that works with proper environment access"""

    required_prompts = ["subagent"]

    def __init__(
        self,
        agent_id: str,
        objective: str,
        original_query: str,
        strategy: str,
        task_instance: DatasetInstance,
        min_iterations_per_agent: int = 3,
        max_iterations_per_agent: int | None = None,
        decision_budget: int | None = None,
        decision_budget_per_round: int | None = None,
        use_remaining_budget_for_verification: bool = True,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.agent_id = agent_id
        # Core agent attributes
        self.objective = objective
        self.original_query = original_query
        self.strategy = strategy
        self.min_iterations_per_agent = min_iterations_per_agent
        self.max_iterations_per_agent = max_iterations_per_agent
        self.decision_budget_per_round = decision_budget_per_round
        self.use_remaining_budget_for_verification = (
            use_remaining_budget_for_verification
        )
        self.budget = BudgetLedger(
            decision_budget if decision_budget is not None else max_iterations_per_agent
        )
        if self.budget.max_decision_calls is None:
            raise ValueError("WorkerSubagent requires a decision budget")
        self.min_iterations_per_agent = min(
            min_iterations_per_agent, self.budget.max_decision_calls
        )
        self.task_instance = task_instance
        self.env, self.llm_w_tools = self.init_environment(task_instance, agent_id)
        self.shared_prompt_templates = self.get_dataset_prompt_templates(self.env)
        # Conversation management
        self.conv_history = SubAgentConversationHistory(agent_id=agent_id)
        self.submission_policy = DeferredSubmissionPolicy(
            enabled=use_remaining_budget_for_verification
        )
        self._terminal_action_executed = False
        self.last_terminal_tool_call: dict[str, Any] | None = None

        self._execution_lock = threading.Lock()

        logger.info(
            f"WorkerSubagent {agent_id} initialized with objective: {objective[:100]}..."
        )

    @classmethod
    def init_from_agent(
        cls,
        agent: BaseAgentWithTools,
        agent_id: str,
        objective: str,
        original_query: str,
        strategy: str,
        min_iterations_per_agent: int = 3,
        max_iterations_per_agent: int | None = None,
        **kwargs,
    ):
        return cls(
            agent_id=agent_id,
            objective=objective,
            original_query=original_query,
            strategy=strategy,
            min_iterations_per_agent=min_iterations_per_agent,
            max_iterations_per_agent=max_iterations_per_agent,
            llm=agent.llm,
            dataset=agent.dataset,
            prompts=agent.prompts,
            tools=agent.tools,
            env=agent.env_name,
            env_prompts=agent.env_prompts,
            **kwargs,
        )

    def _run_one_round(self, message: str) -> SubAgentRoundResult:
        """Process the task by calling tools with proper conversation state management"""
        logger.info(f"Agent {self.agent_id} starting process")
        logger.info(f"Agent {self.agent_id} objective: {self.objective}")

        # Compile the prompt using shared templates exactly like single agent
        messages = (
            self.prompts["subagent"]
            .get_template("start_with_orchestrator_guidance")
            .compile(
                orchestrator_objective=self.objective,
                orchestrator_guidance=message,
                **self.shared_prompt_templates,
            )
        )
        # Continue from previous conversation state if exists and valid
        if len(self.conv_history.internal_comms) > 0:
            logger.info(
                f"Agent {self.agent_id} continuing from previous conversation state with {len(self.conv_history.internal_comms)} messages"
            )
            last_n_messages = self.conv_history.last_n_iterations_messages(n=20)
            messages.extend(last_n_messages)

        # Simple iteration loop with conversation persistence
        curr_iteration = 0
        recent_actions: list = []
        recent_observations: list = []
        iteration = 0
        round_cap = self.decision_budget_per_round
        if round_cap is None:
            round_cap = self.max_iterations_per_agent
        while (
            round_cap is None
            or iteration < round_cap
        ):
            if not self.budget.try_consume_decision():
                logger.info(
                    f"Agent {self.agent_id} lifetime decision budget exhausted"
                )
                break
            curr_iteration += 1
            iteration += 1
            logger.info(
                f"Agent {self.agent_id} iteration {iteration}/{round_cap or 'budget'}"
            )
            # Invoke LLM with tools using retry logic
            response: AIMessage = cast(
                AIMessage,
                self.llm_w_tools.invoke(
                    messages,  # type: ignore
                    num_retries=2,
                    **self._get_llm_params_dict(),
                ),
            )
            self.budget.record_response(response)
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                response.tool_calls = [response.tool_calls[0]]
                tool_name = ""

                response_msg = convert_to_openai_messages(response)
                messages.append(response_msg)  # type: ignore
                self.conv_history.add_internal_message(
                    message=response_msg,  # type: ignore
                    iteration_num=curr_iteration,
                )
                try:
                    # Execute tool with retry logic
                    tool_name = tool_call["name"]
                    remaining = self.budget.remaining_decision_calls
                    deferred_content = self.submission_policy.defer_if_needed(
                        tool_call, remaining
                    )
                    was_deferred = deferred_content is not None
                    if deferred_content is not None:
                        tool_resp = ToolMessage(
                            content=deferred_content,
                            tool_call_id=tool_call["id"],
                            name=tool_name,
                        )
                    else:
                        tool_resp = self.env.execute_tool(tool_call)
                        if self.submission_policy.is_terminal_tool(tool_name):
                            self._terminal_action_executed = True
                            self.last_terminal_tool_call = dict(tool_call)
                except Exception as e:
                    # Add error message to conversation state (consistent with single_agent.py)
                    error_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["name"],
                        "content": f"ERROR: Tool **{tool_call['name']}** failed with error: {str(e)}. Please check the tool call.",
                    }
                    messages.append(error_msg)  # type: ignore
                    self.conv_history.add_internal_message(
                        message=error_msg,
                        iteration_num=curr_iteration,  # type: ignore
                    )
                    logger.warning(
                        f"Tool **{tool_name}** failed with error: {str(e)}\n{traceback.format_exc()}"
                    )
                else:
                    # Add tool response to messages and state
                    tool_msg = convert_to_openai_messages(tool_resp)
                    messages.append(tool_msg)  # type: ignore
                    self.conv_history.add_internal_message(
                        message=tool_msg,  # type: ignore
                        iteration_num=curr_iteration,
                    )
                    # Track action and observation for loop detection
                    tool_input = tool_call.get("args", {})
                    action_str = f"{tool_name}({', '.join([f'{k}={v}' for k, v in tool_input.items()])})"
                    obs_str = str(tool_resp.content) if tool_resp else ""
                    recent_actions.append(action_str)
                    recent_observations.append(obs_str)
                    if len(recent_actions) > 3:
                        recent_actions.pop(0)
                        recent_observations.pop(0)
                    # Loop detection — warn agent if stuck
                    if len(recent_actions) >= 3:
                        last_actions = recent_actions[-3:]
                        last_obs = recent_observations[-3:]
                        loop_warn = None
                        if len(set(last_actions)) == 1 and last_actions[0] != "":
                            loop_warn = {
                                "role": "user",
                                "content": (
                                    "WARNING: You have repeated the same action 3 times in a row with identical results. "
                                    "This approach is not working. Try a DIFFERENT strategy:\n"
                                    "- If editing a file fails repeatedly, try reading the full file first and rewriting it completely\n"
                                    "- If a command produces wrong output, change your approach rather than re-running it\n"
                                    "- Consider whether you're solving the right problem"
                                ),
                            }
                        elif len(set(last_obs)) == 1 and last_obs[0] != "":
                            loop_warn = {
                                "role": "user",
                                "content": (
                                    "WARNING: The last 3 tool calls produced identical output. "
                                    "You appear to be stuck in a loop. Change your approach."
                                ),
                            }
                        # Similarity-based loop detection: catch near-identical actions
                        elif all(a != "" for a in last_actions):
                            tool_names = [a.split("(")[0] if "(" in a else a for a in last_actions]
                            if len(set(tool_names)) == 1:
                                all_errors = all(
                                    "error" in o.lower() or "No such file" in o or "Exit code:" in o
                                    for o in last_obs if o
                                )
                                if all_errors:
                                    loop_warn = {
                                        "role": "user",
                                        "content": (
                                            "WARNING: You have called the same tool 3 times in a row and all "
                                            "returned errors. This approach is failing. Stop and try a completely "
                                            "different strategy. Do NOT continue with variations of the same command."
                                        ),
                                    }
                        if loop_warn:
                            messages.append(loop_warn)  # type: ignore
                            self.conv_history.add_internal_message(
                                message=loop_warn,  # type: ignore
                                iteration_num=curr_iteration,
                            )
                    # Check if done
                    if (
                        not was_deferred
                        and self.submission_policy.is_terminal_tool(tool_name)
                    ):
                        logger.info(
                            f"Agent {self.agent_id} executed terminal tool {tool_name}"
                        )
                        break
                    elif self.env.env_done():
                        logger.info(
                            f"Environment {self.env_name} is done for agent {self.agent_id}"
                        )
                        break
            else:
                # No tool calls - add error message to conversation state
                error_msg = {
                    "role": "user",
                    "content": "ERROR: No tool calls found. Please use the tools to solve the task.",
                }
                messages.append(error_msg)  # type: ignore
                self.conv_history.add_internal_message(
                    message=error_msg,  # type: ignore
                    iteration_num=curr_iteration,
                )
                logger.warning(
                    f"Agent {self.agent_id}: No tool calls found in iteration {iteration}"
                )

        # If the final model call did not terminate, submit the latest verified
        # candidate rather than discarding it at lifetime budget exhaustion.
        remaining = self.budget.remaining_decision_calls
        if (
            remaining == 0
            and not self.env.env_done()
            and not self._terminal_action_executed
        ):
            fallback_tool_call = self.submission_policy.latest_tool_call(
                call_id=f"{self.agent_id}_auto_submit_latest_candidate"
            )
            if fallback_tool_call is not None:
                try:
                    fallback_msg = self.env.execute_tool(fallback_tool_call)  # type: ignore[arg-type]
                    self._terminal_action_executed = True
                    self.last_terminal_tool_call = dict(fallback_tool_call)
                    converted_fallback = convert_to_openai_messages(fallback_msg)
                    messages.append(converted_fallback)  # type: ignore
                    self.conv_history.add_internal_message(
                        message=converted_fallback,  # type: ignore
                        iteration_num=max(1, curr_iteration),
                    )
                    logger.info(
                        f"Agent {self.agent_id} auto-submitted its latest verified candidate"
                    )
                except Exception as e:
                    logger.warning(
                        f"Agent {self.agent_id} failed to auto-submit its latest candidate: {e}"
                    )
        curr_iteration += 1
        findings_message = (
            self.prompts["subagent"]
            .get_template("summarize_findings", with_base=False)
            .compile()
        )[0]
        messages.append(findings_message)  # type: ignore
        self.conv_history.add_internal_message(
            message=findings_message,  # type: ignore
            iteration_num=curr_iteration,
        )

        self.budget.record_auxiliary_call()
        llm_response = self.llm.invoke(messages)
        self.budget.record_response(llm_response)

        # Update agent's conversation
        self.conv_history.add_internal_message(
            message=convert_to_openai_messages(llm_response),  # type: ignore
            iteration_num=curr_iteration,
        )
        logger.info(
            f"Agent {self.agent_id} completed round {self.conv_history.current_round} (n_iterations={self.conv_history.curr_iteration}). Findings:\n{llm_response.text()} "
        )

        return SubAgentRoundResult(
            agent_id=self.agent_id,
            findings=llm_response.text(),
            env_status=self.env.env_status(),
        )

    def process_orchestrator_message(self, message: str) -> SubAgentRoundResult:
        """Process a message from orchestrator with conversation persistence and proper error handling"""
        # Increment round counter for new orchestrator message
        self.conv_history.start_new_round()
        logger.info(
            f"Agent {self.agent_id} ({self.strategy}) processing message for round {self.conv_history.current_round}"
        )
        self.conv_history.add_external_message("lead_agent", message)
        round_result = self._run_one_round(message)
        self.conv_history.add_external_message("subagent", round_result.findings)

        if self.env.env_done() or self._terminal_action_executed:
            self.conv_history.status = "completed"
        elif self.budget.remaining_decision_calls == 0:
            self.conv_history.status = "budget_exhausted"
        elif self.should_stop_due_to_rate_limiting():
            self.conv_history.status = "rate_limited"

        return round_result

    def should_stop_due_to_rate_limiting(self) -> bool:
        """Check if agent should stop due to excessive rate limiting"""
        if hasattr(self.env, "should_stop_due_to_rate_limiting"):
            return self.env.should_stop_due_to_rate_limiting()
        return False
