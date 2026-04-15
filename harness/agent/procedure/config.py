"""
Procedural Config - Recipe T4: Procedural Support

T4a: Program Support Cards (Skill Availability)
  - Pre-compiled domain program cards injected via dense passage retrieval
  - Each card: (1) program goal, (2) prerequisites, (3) step-by-step instructions, (4) common pitfalls

T4b: Skill Activation Prompts (Skill Utilization)
  - Prompts agent to enumerate/evaluate/select/verify skills
  - Injected at task start and re-triggered on unexpected tool results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ─────────────────────────────────────────────
# T4a: Program Support Cards
# ─────────────────────────────────────────────

@dataclass
class SkillCard:
    """Program Support Card aligned with T4a.

    Contains domain-specific structured knowledge:
    (1) program_goal     - What the program achieves
    (2) prerequisites    - Pre-checks before execution
    (3) steps            - Step-by-step instructions
    (4) common_pitfalls  - Pitfalls and how to avoid them

    Additionally supports:
    - domain: which benchmark domain this card belongs to
    - trigger_keywords: for backward-compatible keyword matching
    - trigger_retrieval: whether to use dense retrieval (default True)
    """
    name: str
    description: str
    domain: str = ""
    # T4a components
    program_goal: str = ""
    prerequisites: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    common_pitfalls: List[str] = field(default_factory=list)
    # Backward-compatible fields
    trigger_keywords: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    compact: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "program_goal": self.program_goal,
            "prerequisites": self.prerequisites,
            "steps": self.steps,
            "common_pitfalls": self.common_pitfalls,
            "trigger_keywords": self.trigger_keywords,
            "examples": self.examples,
            "compact": self.compact,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SkillCard":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            domain=data.get("domain", ""),
            program_goal=data.get("program_goal", ""),
            prerequisites=data.get("prerequisites", []),
            steps=data.get("steps", []),
            common_pitfalls=data.get("common_pitfalls", []),
            trigger_keywords=data.get("trigger_keywords", []),
            examples=data.get("examples", []),
            compact=data.get("compact", ""),
        )


@dataclass
class RetrievalConfig:
    """Dense passage retrieval config for T4a program support cards.

    Uses a frozen BERT-based bi-encoder to compute cosine similarity
    between task description embeddings and card embeddings.
    """
    # Embedding model (sentence-transformers model name or HuggingFace path)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Device for embedding model: "cpu", "cuda", or "mps"
    device: str = "cpu"
    # Number of top cards to retrieve
    top_k: int = 3
    # Batch size for embedding computation
    batch_size: int = 16
    # Whether to use cache for embeddings
    cache_embeddings: bool = True


@dataclass
class ProgramSupportConfig:
    """Configuration for T4a: Program Support Cards."""
    enabled: bool = False
    # Directory containing card files (YAML/JSON)
    cards_dir: str = ""
    # Retrieval config
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    # Domain-specific card sets (domain_name -> list of card names)
    # If empty, cards are shared across all domains
    domain_card_sets: dict = field(default_factory=dict)
    # Fallback: whether to use keyword trigger when retrieval is disabled
    use_keyword_fallback: bool = True


# ─────────────────────────────────────────────
# T4b: Skill Activation Prompts
# ─────────────────────────────────────────────

@dataclass
class SkillActivationComponent:
    """One component of a skill activation prompt (T4b).

    Types:
      - skill_inventory: List all available tools with descriptions
      - skill_selection: Checklist for reasoning about skill relevance
      - skill_verification: Prompt to verify skill output before continuing
    """
    component_type: str  # "skill_inventory" | "skill_selection" | "skill_verification"
    content: str
    # When to inject: "task_start" | "on_unexpected_result"
    inject_trigger: str = "task_start"


@dataclass
class SkillActivationConfig:
    """Configuration for T4b: Skill Activation Prompts.

    Three components injected at task start and re-triggered on unexpected results:
    (a) Skill inventory: list of all available tools with descriptions
    (b) Skill selection checklist: explicit reasoning to pick the right skill
    (c) Skill verification: validate skill output before proceeding
    """
    enabled: bool = False
    # Inject at task start (first iteration)
    inject_at_start: bool = True
    # Re-trigger when tool returns unexpected result
    retrigger_on_unexpected: bool = True
    # Keywords that indicate an unexpected tool result (for re-trigger detection)
    unexpected_keywords: List[str] = field(default_factory=lambda: [
        "error", "failed", "none", "empty", "not found",
        "invalid", "unexpected", "timeout", "exception",
    ])
    # Number of unexpected results before re-trigger
    unexpected_threshold: int = 1
    # Whether to include skill inventory
    include_inventory: bool = True
    # Whether to include skill selection checklist
    include_selection: bool = True
    # Whether to include skill verification
    include_verification: bool = True
    # Custom skill inventory (if empty, built from tool definitions at runtime)
    custom_inventory: List[SkillActivationComponent] = field(default_factory=list)


# ─────────────────────────────────────────────
# Unified Procedural Config (T4a + T4b)
# ─────────────────────────────────────────────

@dataclass
class ProceduralConfig:
    """Unified configuration for procedural support module (T4).

    T4a: Program Support Cards (skill availability)
    T4b: Skill Activation Prompts (skill utilization)
    """
    enabled: bool = False
    # T4a: Program Support Cards
    program_support: ProgramSupportConfig = field(default_factory=ProgramSupportConfig)
    # T4b: Skill Activation Prompts
    skill_activation: SkillActivationConfig = field(default_factory=SkillActivationConfig)
    # Backward-compatible fields (redirected to T4a)
    cards_dir: str = ""
    max_expansions_per_iteration: int = 3
    show_skill_list: bool = True
    cache_triggers: bool = True

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "program_support": {
                "enabled": self.program_support.enabled,
                "cards_dir": self.program_support.cards_dir or self.cards_dir,
                "retrieval": {
                    "embedding_model": self.program_support.retrieval.embedding_model,
                    "device": self.program_support.retrieval.device,
                    "top_k": self.program_support.retrieval.top_k,
                    "batch_size": self.program_support.retrieval.batch_size,
                    "cache_embeddings": self.program_support.retrieval.cache_embeddings,
                },
                "domain_card_sets": self.program_support.domain_card_sets,
                "use_keyword_fallback": self.program_support.use_keyword_fallback,
            },
            "skill_activation": {
                "enabled": self.skill_activation.enabled,
                "inject_at_start": self.skill_activation.inject_at_start,
                "retrigger_on_unexpected": self.skill_activation.retrigger_on_unexpected,
                "unexpected_keywords": self.skill_activation.unexpected_keywords,
                "unexpected_threshold": self.skill_activation.unexpected_threshold,
                "include_inventory": self.skill_activation.include_inventory,
                "include_selection": self.skill_activation.include_selection,
                "include_verification": self.skill_activation.include_verification,
            },
            "cards_dir": self.cards_dir,
            "max_expansions_per_iteration": self.max_expansions_per_iteration,
            "show_skill_list": self.show_skill_list,
            "cache_triggers": self.cache_triggers,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProceduralConfig":
        ps_data = data.get("program_support", {})
        sa_data = data.get("skill_activation", {})

        retrieval_data = ps_data.get("retrieval", {})
        retrieval_cfg = RetrievalConfig(
            embedding_model=retrieval_data.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            device=retrieval_data.get("device", "cpu"),
            top_k=retrieval_data.get("top_k", 3),
            batch_size=retrieval_data.get("batch_size", 16),
            cache_embeddings=retrieval_data.get("cache_embeddings", True),
        )

        program_support = ProgramSupportConfig(
            enabled=ps_data.get("enabled", False),
            cards_dir=ps_data.get("cards_dir", data.get("cards_dir", "")),
            retrieval=retrieval_cfg,
            domain_card_sets=ps_data.get("domain_card_sets", {}),
            use_keyword_fallback=ps_data.get("use_keyword_fallback", True),
        )

        skill_activation = SkillActivationConfig(
            enabled=sa_data.get("enabled", False),
            inject_at_start=sa_data.get("inject_at_start", True),
            retrigger_on_unexpected=sa_data.get("retrigger_on_unexpected", True),
            unexpected_keywords=sa_data.get("unexpected_keywords", []),
            unexpected_threshold=sa_data.get("unexpected_threshold", 1),
            include_inventory=sa_data.get("include_inventory", True),
            include_selection=sa_data.get("include_selection", True),
            include_verification=sa_data.get("include_verification", True),
        )

        return cls(
            enabled=data.get("enabled", False),
            program_support=program_support,
            skill_activation=skill_activation,
            cards_dir=data.get("cards_dir", ""),
            max_expansions_per_iteration=data.get("max_expansions_per_iteration", 3),
            show_skill_list=data.get("show_skill_list", True),
            cache_triggers=data.get("cache_triggers", True),
        )
