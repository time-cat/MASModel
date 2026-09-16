from typing import Any, Dict, Optional, Self

from litellm import _logging as litellm_logging
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_scaling.agents import AgentSystem, get_agent_cls
from agent_scaling.langfuse_client import get_lf_client
from agent_scaling.utils import (
    disable_local_cache,
    enable_local_cache,
    enable_local_logging,
)

from .dataset import DatasetConfig
from .llm import LLMConfig
from .prompts import Prompt


class AgentConfig(BaseModel):
    """Agent configuration that captures all YAML fields and passes them to the agent."""

    model_config = ConfigDict(extra="allow")

    name: str
    prompts: Dict[str, Prompt] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_prompts(self) -> Self:
        agent_cls = get_agent_cls(self.name)
        agent_cls.check_required_prompts(self.prompts)
        return self

    @model_validator(mode="before")
    @classmethod
    def add_prompt_names(cls, data: Any) -> Dict[str, Any]:
        data = dict(data)
        prompts = dict(data.get("prompts", {}))
        for k, prompt in prompts.items():
            prompt = dict(prompt)
            if prompt.get("name") is None:
                prompt["name"] = k
            prompts[k] = prompt
        data["prompts"] = prompts
        return data

    def get_agent_kwargs(self) -> Dict[str, Any]:
        """Get all extra fields from YAML config to pass to the agent constructor."""
        if not self.__pydantic_extra__:
            return {}
        # Convert OmegaConf DictConfig/ListConfig to plain Python types
        result = {}
        for k, v in self.__pydantic_extra__.items():
            try:
                from omegaconf import OmegaConf
                if OmegaConf.is_config(v):
                    v = OmegaConf.to_container(v, resolve=True)
            except ImportError:
                pass
            result[k] = v
        return result

    def get_run_metadata(self) -> Dict[str, Any]:
        prompts = {}
        assert self.prompts is not None, "Prompts must be defined in AgentConfig"
        for k, prompt in self.prompts.items():
            prompts[k] = {
                "name": prompt.name,
            }
        metadata: Dict[str, Any] = {
            "name": self.name,
            "prompts": prompts,
        }
        agent_kwargs = self.get_agent_kwargs()
        if agent_kwargs:
            metadata["agent_params"] = agent_kwargs
        return metadata


class RunConfig(BaseModel):
    agent: AgentConfig
    dataset: DatasetConfig
    llm: LLMConfig
    run_name: str
    save_dir: Optional[str] = None
    log_langfuse: bool = True
    use_disk_cache: bool = False
    debug: bool = False
    max_instances: Optional[int] = None
    num_workers: int = 1
    console_log_level: Optional[str] = None

    @property
    def run_parallel(self) -> bool:
        return self.num_workers > 1

    @property
    def effective_console_log_level(self) -> str:
        if self.console_log_level is not None:
            return self.console_log_level.upper()
        if self.agent.name.startswith("multi-agent-"):
            return "WARNING"
        return "INFO"

    def model_post_init(self, context: Any) -> None:
        client = get_lf_client()
        self.log_langfuse = (
            self.log_langfuse and client is not None
            # and self.dataset.langfuse_dataset is not None
        )
        if not self.log_langfuse and not self.run_parallel:
            enable_local_logging(prompt_only=True)
        litellm_logging._disable_debugging()  # type: ignore
        # Disable litellm debugging logs
        if self.use_disk_cache:
            enable_local_cache()
        else:
            disable_local_cache()

    def get_agent(self) -> AgentSystem:
        return get_agent_cls(self.agent.name).from_config(
            llm_config=self.llm,
            dataset_config=self.dataset,
            prompts=self.agent.prompts,
            **self.agent.get_agent_kwargs(),
        )

    def get_run_metadata(self) -> Dict[str, Any]:
        ret: Dict[str, Any] = {
            "agent": self.agent.get_run_metadata(),
            "llm": self.llm.model_dump(exclude_none=True),
            "dataset": self.dataset.model_dump(exclude_none=True),
        }
        if self.save_dir is not None:
            ret["save_dir"] = self.save_dir
        ret["run_name"] = self.run_name
        ret["num_workers"] = self.num_workers
        ret["console_log_level"] = self.effective_console_log_level
        return ret
