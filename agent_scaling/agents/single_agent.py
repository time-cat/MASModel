import os.path as osp
import traceback
from typing import List, Optional, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,  # type: ignore
    ToolMessage,
)
from langchain_core.messages.utils import convert_to_openai_messages  # type: ignore

from agent_scaling.agents.base import AgentSystemWithTools
from agent_scaling.config.llm import LLMParams
from agent_scaling.datasets import (
    DatasetInstance,
    DatasetInstanceOutputWithTrajectory,
    TrajectoryStep,
)
from agent_scaling.env import AgentEnvironment
from agent_scaling.logger import logger
from agent_scaling.utils import write_yaml
from agent_scaling.budget import BudgetLedger
from agent_scaling.verification import DeferredSubmissionPolicy

from .registry import register_agent
from .multiagent_utils.aggregation import AggregationRequest, CandidateRecord, aggregate
from .multiagent_utils.result_selection import (
    task_output_type,
)


@register_agent("single-agent")
class SingleAgent(AgentSystemWithTools[AgentEnvironment]):
    """
    A single agent that can interact with tools. Developed in SWE-Agent framework.
    """

    required_prompts = ["main"]

    def __init__(
        self,
        max_steps: Optional[int] = None,
        total_decision_budget: Optional[int] = 32,
        use_remaining_budget_for_verification: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_steps = max_steps
        self.total_decision_budget = total_decision_budget
        self.use_remaining_budget_for_verification = (
            use_remaining_budget_for_verification
        )
        self.budget: Optional[BudgetLedger] = None
        self.submission_policy = DeferredSubmissionPolicy(
            enabled=use_remaining_budget_for_verification
        )
        self.last_terminal_tool_call: dict | None = None

    def run_agent(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
    ) -> DatasetInstanceOutputWithTrajectory:
        llm_params_dict = llm_params.model_dump() if llm_params else {}
        self.budget = BudgetLedger(self.total_decision_budget)
        self.submission_policy = DeferredSubmissionPolicy(
            enabled=self.use_remaining_budget_for_verification
        )
        self.last_terminal_tool_call = None
        max_steps = self.max_steps
        if max_steps is None:
            max_steps = self.budget.remaining_decision_calls
        if max_steps is None:
            raise ValueError("Single-agent requires total_decision_budget or max_steps")
        if self.budget.remaining_decision_calls is not None:
            max_steps = min(max_steps, self.budget.remaining_decision_calls)
        env, llm_w_tools = self.init_environment(instance)
        shared_prompt_templates = self.get_dataset_prompt_templates(env)

        messages = cast(
            list,
            self.prompts["main"].compile(**shared_prompt_templates),
        )
        trajectory: List[TrajectoryStep] = []
        final_answer = ""
        final_env_output = {}
        is_done = False
        for step in range(max_steps):
            if not self.budget.try_consume_decision():
                logger.info("Single-agent lifetime decision budget exhausted")
                break
            response: BaseMessage = llm_w_tools.invoke(messages, **llm_params_dict)  # type: ignore
            self.budget.record_response(response)
            response = cast(AIMessage, response)
            if response.tool_calls:
                response.tool_calls = [response.tool_calls[0]]

            messages.append(convert_to_openai_messages(response))
            tool_resp: ToolMessage | None = None
            was_deferred = False
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_name = ""
                try:
                    tool_name = tool_call["name"]
                    tool_input = tool_call["args"]
                    action = f"{tool_name}({', '.join([f'{k}={v}' for k, v in tool_input.items()])})"
                    remaining_steps = max_steps - step - 1
                    deferred_content = self.submission_policy.defer_if_needed(
                        tool_call, remaining_steps
                    )
                    if deferred_content is not None:
                        was_deferred = True
                        tool_resp = ToolMessage(
                            content=deferred_content,
                            tool_call_id=tool_call["id"],
                            name=tool_name,
                        )
                    else:
                        tool_resp = env.execute_tool(tool_call)
                        if self.submission_policy.is_terminal_tool(tool_name):
                            self.last_terminal_tool_call = dict(tool_call)
                    messages.append(convert_to_openai_messages(tool_resp))
                    is_done = (
                        not was_deferred
                        and self.submission_policy.is_terminal_tool(tool_name)
                    )
                except Exception as e:
                    action = ""
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": tool_call["name"],
                            "content": f"ERROR: Tool **{tool_call['name']}** failed with error: {str(e)}. Please check the tool call.",
                        }
                    )
                    logger.warning(
                        f"Tool **{tool_name}** failed with error: {str(e)}\n{traceback.format_exc()}"
                    )
            else:
                action = ""
                messages.append(
                    {
                        "role": "user",
                        "content": "ERROR: No tool calls found. Please use the tools to solve the task.",
                    }
                )
                logger.warning("No tool calls found in the response.")
            trajectory.append(
                TrajectoryStep(
                    action=action,
                    observation=str(tool_resp.content) if tool_resp else "",
                    response=str(response.content),
                    thought=str(response.content),
                )
            )
            # Loop detection — warn agent if stuck
            if len(trajectory) >= 3:
                last_actions = [t.action for t in trajectory[-3:]]
                last_obs = [t.observation for t in trajectory[-3:]]

                if len(set(last_actions)) == 1 and last_actions[0] != "":
                    messages.append({
                        "role": "user",
                        "content": (
                            "WARNING: You have repeated the same action 3 times in a row with identical results. "
                            "This approach is not working. Try a DIFFERENT strategy:\n"
                            "- If editing a file fails repeatedly, try reading the full file first and rewriting it completely\n"
                            "- If a command produces wrong output, change your approach rather than re-running it\n"
                            "- Consider whether you're solving the right problem"
                        ),
                    })
                elif len(set(last_obs)) == 1 and last_obs[0] != "":
                    messages.append({
                        "role": "user",
                        "content": (
                            "WARNING: The last 3 tool calls produced identical output. "
                            "You appear to be stuck in a loop. Change your approach."
                        ),
                    })
                # Similarity-based loop detection: catch near-identical actions
                # (e.g., path traversal with varying depth)
                elif len(last_actions) >= 3 and all(a != "" for a in last_actions):
                    # Check if actions share the same tool name and similar structure
                    tool_names = [a.split("(")[0] if "(" in a else a for a in last_actions]
                    if len(set(tool_names)) == 1:
                        # Same tool called 3 times - check if observations are all errors
                        all_errors = all(
                            "error" in o.lower() or "No such file" in o or "Exit code:" in o
                            for o in last_obs if o
                        )
                        if all_errors:
                            messages.append({
                                "role": "user",
                                "content": (
                                    "WARNING: You have called the same tool 3 times in a row and all "
                                    "returned errors. This approach is failing. Stop and try a completely "
                                    "different strategy. Do NOT continue with variations of the same command."
                                ),
                            })
            if is_done or env.env_done():
                final_answer = trajectory[-1].observation
                break
        else:
            # Step budget exhausted — force submit if env supports it and hasn't submitted
            if not env.env_done():
                logger.warning(
                    f"Step budget exhausted ({max_steps} steps). Auto-submitting..."
                )
                submit_tool_name = None
                if "submit_patch" in env.tools:
                    submit_tool_name = "submit_patch"
                elif "submit" in env.tools:
                    submit_tool_name = "submit"
                fallback_tool_call = self.submission_policy.latest_tool_call(
                    call_id="auto_submit_latest_candidate"
                )
                if fallback_tool_call is not None or submit_tool_name is not None:
                    try:
                        tool_call = fallback_tool_call or {
                            "name": submit_tool_name,
                            "args": {"reasoning": "Auto-submit: step budget exhausted"},
                            "id": "auto_submit_budget",
                            "type": "tool_call",
                        }
                        tool_msg = env.execute_tool(tool_call)
                        self.last_terminal_tool_call = dict(tool_call)
                        final_answer = str(tool_msg.content)
                    except Exception as e:
                        logger.warning(f"Auto-submit on budget exhaustion failed: {e}")
        final_env_output = env.env_status()
        output_type = task_output_type(self)
        aggregation_result = aggregate(
            AggregationRequest(
                output_type=output_type,
                candidates=[CandidateRecord("single_agent", final_answer, agent=self, env=env)],
                agents=(self,),
                plan_strategy="successful",
            )
        )
        canonical_submission = aggregation_result.canonical_submission
        if instance_dir is not None:
            out = {
                "trajectory": [t.model_dump() for t in trajectory],
                "final_answer": final_answer,
                "canonical_submission": canonical_submission,
                "output_type": output_type,
                "evaluation_semantics": (
                    "environment_state_execution"
                    if output_type in {"executable_plan", "patch_or_state"}
                    else "task_contract_submission"
                ),
                "environment_action_trace": getattr(env, "get_action_trace", lambda: [])(),
                "budget": self.budget.snapshot() if self.budget else None,
                "verification": self.submission_policy.snapshot(),
            }
            out.update(aggregation_result.to_metadata())
            write_yaml(
                out,
                osp.join(instance_dir, "agent_output.yaml"),
                use_long_str_representer=True,
            )
        return DatasetInstanceOutputWithTrajectory(
            data_instance=instance,
            agent_output=final_answer,
            canonical_submission=canonical_submission,
            trajectory=trajectory,
            final_env_output=final_env_output,
        )
