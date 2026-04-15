"""
Procedural Support module - Recipe T4: Procedural Support

T4a: Program Support Cards (Skill Availability)
  - Dense passage retrieval using frozen BERT bi-encoder
  - Cards with: Goal, Prerequisites, Steps, Common Pitfalls

T4b: Skill Activation Prompts (Skill Utilization)
  - Skill inventory, selection checklist, verification
  - Injected at task start; re-triggered on unexpected tool results
"""

from .config import (
    ProceduralConfig,
    ProgramSupportConfig,
    RetrievalConfig,
    SkillActivationConfig,
    SkillActivationComponent,
    SkillCard,
)
from .event import ProceduralEvent
from .expander import ProceduralExpander
from .retriever import DenseRetriever
from .store import ProceduralStore
from .trigger import ProceduralTrigger

__all__ = [
    # Config
    "ProceduralConfig",
    "ProgramSupportConfig",
    "RetrievalConfig",
    "SkillActivationConfig",
    "SkillActivationComponent",
    "SkillCard",
    # Event
    "ProceduralEvent",
    # Core
    "ProceduralStore",
    "ProceduralTrigger",
    "ProceduralExpander",
    "DenseRetriever",
    # Utilities
    "get_procedure_summary",
]


def get_procedure_summary(
    trigger_events: list[ProceduralEvent],
    expander_events: list[ProceduralEvent],
) -> dict:
    """Get a summary dict from procedural events.

    Args:
        trigger_events: List of ProceduralEvent from triggers
        expander_events: List of ProceduralEvent from expanders

    Returns:
        dict with event counts and triggered skills
    """
    all_events = trigger_events + expander_events

    triggered_skills = set()
    retrieved_cards = set()
    unexpected_triggers = 0
    skill_activations = 0

    for event in trigger_events:
        if event.event_type == "program_card_retrieved":
            triggered_skills.add(event.skill_name)
            retrieved_cards.add(event.skill_name)
        elif event.event_type == "skill_activation_retriggered":
            unexpected_triggers += 1

    for event in expander_events:
        if event.event_type == "program_card_expanded":
            triggered_skills.add(event.skill_name)
        elif event.event_type == "skill_activation_injected":
            skill_activations += 1

    return {
        "total_events": len(all_events),
        "triggered_skills": list(triggered_skills),
        "retrieved_cards": list(retrieved_cards),
        "trigger_count": len(trigger_events),
        "expansion_count": len(expander_events),
        "unexpected_triggers": unexpected_triggers,
        "skill_activations": skill_activations,
    }
