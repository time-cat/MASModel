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

from .registry import register_agent


@register_agent("single-agent")
class SingleAgent(AgentSystemWithTools[AgentEnvironment]):
    """
    A single agent that can interact with tools. Developed in SWE-Agent framework.
    """

    required_prompts = ["main"]

    def __init__(self, max_steps: int = 100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = max_steps

    def run_agent(
        self,
        instance: DatasetInstance,
        instance_dir: Optional[str] = None,
        llm_params: Optional[LLMParams] = None,
        instance_idx: Optional[int] = None,
    ) -> DatasetInstanceOutputWithTrajectory:
        llm_params_dict = llm_params.model_dump() if llm_params else {}
        # Scale step budget with task time limit for datasets like TerminalBench
        # that have varying time limits (600–2400s). Minimum of configured max_steps.
        time_limit = getattr(instance, "time_limit", None)
        if time_limit is not None:
            scaled_steps = max(self.max_steps, time_limit // 5)
            if scaled_steps != self.max_steps:
                logger.info(
                    f"Scaling max_steps from {self.max_steps} to {scaled_steps} "
                    f"based on task time_limit={time_limit}s"
                )
            max_steps = scaled_steps
        else:
            max_steps = self.max_steps
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
            response: BaseMessage = llm_w_tools.invoke(messages, **llm_params_dict)  # type: ignore
            response = cast(AIMessage, response)
            if response.tool_calls:
                response.tool_calls = [response.tool_calls[0]]

            messages.append(convert_to_openai_messages(response))
            tool_resp: ToolMessage | None = None
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_name = ""
                try:
                    tool_resp = env.execute_tool(tool_call)
                    tool_name = tool_call["name"]
                    tool_input = tool_call["args"]
                    action = f"{tool_name}({', '.join([f'{k}={v}' for k, v in tool_input.items()])})"
                    messages.append(convert_to_openai_messages(tool_resp))
                    is_done = tool_name == "done"
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
            # Budget warning — alert agent when running low on steps
            remaining_steps = max_steps - step - 1
            if remaining_steps == max_steps // 4:
                messages.append({
                    "role": "user",
                    "content": (
                        f"BUDGET WARNING: You have only {remaining_steps} steps remaining out of {max_steps}. "
                        "If you haven't already, make your code changes NOW and submit. "
                        "Do not spend remaining steps on exploration — focus on implementation and submission."
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
                if submit_tool_name:
                    try:
                        tool_call = {
                            "name": submit_tool_name,
                            "args": {"reasoning": "Auto-submit: step budget exhausted"},
                            "id": "auto_submit_budget",
                            "type": "tool_call",
                        }
                        tool_msg = env.execute_tool(tool_call)
                        final_answer = str(tool_msg.content)
                    except Exception as e:
                        logger.warning(f"Auto-submit on budget exhaustion failed: {e}")
        final_env_output = env.env_status()
        if instance_dir is not None:
            out = {
                "trajectory": [t.model_dump() for t in trajectory],
                "final_answer": final_answer,
            }
            write_yaml(
                out,
                osp.join(instance_dir, "agent_output.yaml"),
                use_long_str_representer=True,
            )
        return DatasetInstanceOutputWithTrajectory(
            data_instance=instance,
            agent_output=final_answer,
            trajectory=trajectory,
            final_env_output=final_env_output,
        )
