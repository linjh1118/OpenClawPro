"""StructuredMemoryConfig for structured state tracker."""

from dataclasses import dataclass, field
from .update_policy import UpdatePolicy


@dataclass
class StructuredMemoryConfig:
    """Configuration for structured state tracking (H2 verification).

    This harness verifies H2: agent memory failures stem from not
    transforming state into decision-relevant representations.

    The structured tracker maintains four state slots that are
    actively UPDATED as the agent works, not just appended to.
    """

    enabled: bool = False

    # Slot limits
    max_constraints: int = 20
    max_derived_facts: int = 50
    max_pending_subgoals: int = 30
    max_artifact_paths: int = 100

    # Update policy — how to handle slot updates
    update_policy: UpdatePolicy = field(default_factory=lambda: UpdatePolicy.ACTIVE)

    # Whether to auto-derive facts from tool results
    auto_derive_facts: bool = True

    # Whether to track artifact paths
    track_artifacts: bool = True

    # Whether to detect subgoal completion
    detect_subgoal_completion: bool = True

    # Truncation: when a slot exceeds limit, keep most recent N
    keep_most_recent: bool = True

    # LLM config for state extraction (initialize_from_task and periodic update)
    llm_model: str = "glm-4"
    llm_api_url: str = ""
    llm_api_key: str = ""

    # LLM 更新频率控制（防止过度调用）
    # 每隔多少 iteration 做一次 LLM 状态更新（0=只用 initialize，禁用周期性更新）
    llm_update_interval: int = 4
    # 每次 LLM 更新时，收集最近多少条工具结果（buffer 大小）
    llm_update_buffer_size: int = 10

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "max_constraints": self.max_constraints,
            "max_derived_facts": self.max_derived_facts,
            "max_pending_subgoals": self.max_pending_subgoals,
            "max_artifact_paths": self.max_artifact_paths,
            "update_policy": self.update_policy.value,
            "auto_derive_facts": self.auto_derive_facts,
            "track_artifacts": self.track_artifacts,
            "detect_subgoal_completion": self.detect_subgoal_completion,
            "keep_most_recent": self.keep_most_recent,
            "llm_model": self.llm_model,
            "llm_api_url": self.llm_api_url,
            "llm_update_interval": self.llm_update_interval,
            "llm_update_buffer_size": self.llm_update_buffer_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StructuredMemoryConfig":
        data = data.copy()
        if "update_policy" in data and isinstance(data["update_policy"], str):
            data["update_policy"] = UpdatePolicy(data["update_policy"])
        return cls(**data)
