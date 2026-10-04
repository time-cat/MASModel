"""SWE-Bench Verified and Pro dataset classes."""

import json
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import Field

from agent_scaling.datasets.base import Dataset, DatasetInstance, DatasetInstanceOutput
from agent_scaling.datasets.registry import register_dataset, register_dataset_instance

DATASET_IDS = ["swebench-verified", "swebench-pro"]


@register_dataset_instance(DATASET_IDS)
class SWEBenchInstance(DatasetInstance):
    """A single SWE-Bench instance (bug fix task)."""

    instance_id: str
    repo: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: Optional[str] = ""
    fail_to_pass: str = "[]"  # JSON string of test list
    pass_to_pass: str = "[]"  # JSON string of test list
    environment_setup_commit: Optional[str] = None  # Verified
    dockerhub_tag: Optional[str] = None  # Pro
    difficulty: Optional[str] = None  # Verified only
    repo_language: Optional[str] = None  # Pro only
    requirements: Optional[str] = None  # Pro only
    version: Optional[str] = None
    created_at: Optional[str] = None

    def model_post_init(self, __context):
        self.expected_output = self.patch
        # Normalize field casing (some HF datasets use UPPER_CASE)
        if hasattr(self, "FAIL_TO_PASS") and self.fail_to_pass == "[]":
            self.fail_to_pass = getattr(self, "FAIL_TO_PASS", "[]")
        if hasattr(self, "PASS_TO_PASS") and self.pass_to_pass == "[]":
            self.pass_to_pass = getattr(self, "PASS_TO_PASS", "[]")

    @property
    def docker_image(self) -> str:
        """Compute the Docker image name for this instance.

        SWE-bench Docker Hub uses '_1776_' separator in image names
        (not '__' as in instance_ids). Format:
        swebench/sweb.eval.x86_64.{instance_id with __ -> _1776_}:latest
        """
        if self.dockerhub_tag:
            # Pro: use dockerhub_tag directly
            tag = self.dockerhub_tag.lower()
            return f"swebench/sweb.eval.x86_64.{tag}"
        # Verified: convert instance_id to Docker Hub format
        # e.g. astropy__astropy-12907 -> astropy_1776_astropy-12907
        image_tag = self.instance_id.replace("__", "_1776_")
        return f"swebench/sweb.eval.x86_64.{image_tag}:latest"

    @property
    def fail_to_pass_tests(self) -> List[str]:
        """Parse fail_to_pass JSON string into list."""
        return json.loads(self.fail_to_pass)

    @property
    def pass_to_pass_tests(self) -> List[str]:
        """Parse pass_to_pass JSON string into list."""
        return json.loads(self.pass_to_pass)

    def get_prompt_info(self) -> Dict[str, Any]:
        # Pre-format fail_to_pass as a bulleted list string for template insertion
        ftp_tests = self.fail_to_pass_tests
        if ftp_tests:
            ftp_str = "\n".join(f"  - {t}" for t in ftp_tests)
        else:
            ftp_str = ""
        return {
            "problem_statement": self.problem_statement,
            "hints_text": self.hints_text or "",
            "repo": self.repo,
            "base_commit": self.base_commit,
            "fail_to_pass_tests": ftp_str,
        }


@register_dataset(DATASET_IDS)
class SWEBenchDataset(Dataset):
    output_type: Literal["patch_or_state"] = "patch_or_state"
    """SWE-Bench Verified or Pro dataset."""

    dataset_id: str = "swebench-verified"
    instances: List[SWEBenchInstance]

    def get_instance_eval_output(
        self, instance_output: DatasetInstanceOutput[SWEBenchInstance]
    ) -> Dict[str, Any]:
        return {
            "success": instance_output.final_env_output.success or False,
            "num_steps": instance_output.final_env_output.num_steps or -1,
            "instance_id": instance_output.data_instance.instance_id,
            "model_patch": instance_output.submission or "",
        }

    def get_instance_eval_metrics(
        self, instance_output: DatasetInstanceOutput[SWEBenchInstance]
    ) -> Dict[str, Union[int, float]]:
        return {
            "resolved": int(instance_output.final_env_output.success or False),
            "num_steps": instance_output.final_env_output.num_steps or -1,
        }

    def get_metrics(self, eval_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        num_instances = len(eval_outputs)
        num_resolved = sum(e["resolved"] for e in eval_outputs)
        return {
            "resolution_rate": num_resolved / num_instances if num_instances > 0 else 0.0,
            "num_resolved": num_resolved,
            "num_instances": num_instances,
            "avg_num_steps": (
                sum(e["num_steps"] for e in eval_outputs) / num_instances
                if num_instances > 0
                else 0.0
            ),
        }
