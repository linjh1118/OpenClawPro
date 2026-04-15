"""
Procedural Expander - Recipe T4: Procedural Support

Formats skill card content for prompt injection.

T4a: Formats program support cards with the 4-component structure:
     (1) Program Goal, (2) Prerequisites, (3) Steps, (4) Common Pitfalls

T4b: Formats skill activation prompts with 3 components:
     (a) Skill Inventory, (b) Skill Selection Checklist, (c) Skill Verification
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .config import ProceduralConfig, SkillActivationConfig, SkillCard
from .event import ProceduralEvent

logger = logging.getLogger("agent.procedure.expander")


class ProceduralExpander:
    """Formats T4a/T4b content for prompt injection."""

    def __init__(self):
        self._events: List[ProceduralEvent] = []

    # ─────────────────────────────────────────
    # T4a: Program Support Card Formatting
    # ─────────────────────────────────────────

    def format(
        self,
        card: SkillCard,
        include_pitfalls: bool = True,
    ) -> str:
        """Format a program support card (T4a) for injection into the prompt.

        The card follows the 4-component structure from the paper:
        (1) Program Goal
        (2) Prerequisites
        (3) Step-by-step Instructions
        (4) Common Pitfalls

        Args:
            card: The skill card to format
            include_pitfalls: Whether to include common pitfalls

        Returns:
            Formatted string suitable for prompt injection
        """
        lines = [
            f"\n## Program Support Card: {card.name}",
            f"**Description**: {card.description}",
        ]

        # (1) Program Goal
        if card.program_goal:
            lines.append("")
            lines.append("**Goal**:")
            lines.append(f"  {card.program_goal}")

        # (2) Prerequisites
        if card.prerequisites:
            lines.append("")
            lines.append("**Prerequisites (Pre-checks)**:")
            for i, prereq in enumerate(card.prerequisites, 1):
                lines.append(f"  {i}. {prereq}")

        # (3) Step-by-step Instructions
        if card.steps:
            lines.append("")
            lines.append("**Steps**:")
            for i, step in enumerate(card.steps, 1):
                lines.append(f"  {i}. {step}")

        # (4) Common Pitfalls
        if include_pitfalls and card.common_pitfalls:
            lines.append("")
            lines.append("**Common Pitfalls & How to Avoid**:")
            for pitfall in card.common_pitfalls:
                lines.append(f"  - {pitfall}")

        # Backward compatibility: also show examples if present
        if card.examples:
            lines.append("")
            lines.append("**Examples**:")
            for example in card.examples:
                lines.append(f"  - {example}")

        formatted = "\n".join(lines)

        event = ProceduralEvent(
            event_type="program_card_expanded",
            skill_name=card.name,
            data={
                "formatted_length": len(formatted),
                "has_goal": bool(card.program_goal),
                "has_prerequisites": bool(card.prerequisites),
                "step_count": len(card.steps),
                "has_pitfalls": bool(card.common_pitfalls),
            },
        )
        self._events.append(event)

        logger.debug(f"[ProceduralExpander] Expanded card '{card.name}' ({len(formatted)} chars)")

        return formatted

    def format_multiple(
        self,
        cards: List[SkillCard],
        include_pitfalls: bool = True,
        header: str = "## Available Program Support Cards",
        subheader: str = "(Use these procedures when relevant to the task)",
    ) -> str:
        """Format multiple program support cards for injection.

        Args:
            cards: List of skill cards to format
            include_pitfalls: Whether to include common pitfalls
            header: Section header
            subheader: Section subheader

        Returns:
            Combined formatted string
        """
        if not cards:
            return ""

        formatted_parts = []
        for card in cards:
            formatted_parts.append(self.format(card, include_pitfalls=include_pitfalls))

        return "\n".join([
            f"\n{header}",
            f"{subheader}\n",
        ] + formatted_parts)

    def format_skill_list(
        self,
        cards: List[SkillCard],
    ) -> str:
        """Format a compact skill list for display.

        Args:
            cards: List of skill cards

        Returns:
            Compact list as string
        """
        if not cards:
            return ""

        lines = ["\n## Available Skills"]
        for card in cards:
            compact = card.compact or card.description[:50]
            lines.append(f"- **{card.name}**: {compact}")

        return "\n".join(lines)

    # ─────────────────────────────────────────
    # T4b: Skill Activation Prompt Formatting
    # ─────────────────────────────────────────

    def format_skill_activation(
        self,
        config: SkillActivationConfig,
        tool_definitions: List[Dict[str, Any]] | None = None,
    ) -> str:
        """Format skill activation prompt (T4b) for injection.

        Three components:
        (a) Skill Inventory: list all available tools with descriptions
        (b) Skill Selection Checklist: explicit reasoning for picking skills
        (c) Skill Verification: validate output before proceeding

        Args:
            config: Skill activation config
            tool_definitions: Available tool definitions (used to build inventory)

        Returns:
            Formatted skill activation prompt string
        """
        lines = [
            "\n## Skill Activation Prompt (T4b)",
            "",
            "### Critical: Follow these steps for every tool use:",
        ]

        # (a) Skill Inventory
        if config.include_inventory:
            lines.append("")
            lines.append("**(a) Skill Inventory**: Before acting, enumerate ALL available skills/tools:")
            if tool_definitions:
                for tool in tool_definitions:
                    name = tool.get("name", "unknown")
                    desc = tool.get("description", "no description")
                    lines.append(f"  - **{name}**: {desc}")
            elif config.custom_inventory:
                for comp in config.custom_inventory:
                    if comp.component_type == "skill_inventory":
                        lines.append(f"  {comp.content}")
            else:
                lines.append("  (Tool inventory auto-populated at runtime)")

        # (b) Skill Selection Checklist
        if config.include_selection:
            lines.append("")
            lines.append("**(b) Skill Selection Checklist**: Explicitly reason about skill relevance:")
            lines.append("  1. What is the current sub-task?")
            lines.append("  2. Which skill/tool is most relevant?")
            lines.append("  3. Are there prerequisites for this skill?")
            lines.append("  4. What parameters should I pass?")
            lines.append("  5. Is the selected skill the BEST choice among alternatives?")

        # (c) Skill Verification
        if config.include_verification:
            lines.append("")
            lines.append("**(c) Skill Execution Verification**: After each tool call:")
            lines.append("  1. Is the output format as expected?")
            lines.append("  2. Does the output align with the task goal?")
            lines.append("  3. If unexpected: re-evaluate skill selection and try alternatives.")
            lines.append("  4. Only proceed to the next step when the output is valid and useful.")

        formatted = "\n".join(lines)

        event = ProceduralEvent(
            event_type="skill_activation_injected",
            skill_name="",
            data={"formatted_length": len(formatted)},
        )
        self._events.append(event)

        logger.debug(f"[ProceduralExpander] Skill activation prompt formatted ({len(formatted)} chars)")

        return formatted

    def format_skill_activation_retrigger(
        self,
        tool_name: str,
        result_summary: str,
    ) -> str:
        """Format re-trigger prompt when unexpected tool result is detected.

        Args:
            tool_name: Name of the tool that returned unexpected result
            result_summary: Summary of the unexpected result

        Returns:
            Formatted re-trigger prompt string
        """
        lines = [
            "\n## Skill Activation Re-trigger (Unexpected Result Detected)",
            "",
            f"Tool **'{tool_name}'** returned an unexpected result:",
            f'  "{result_summary}"',
            "",
            "Re-evaluate your skill selection:",
            "  1. Was this the right skill for the sub-task?",
            "  2. Are there alternative skills that might work better?",
            "  3. Check if prerequisites were met before calling the skill.",
            "  4. Verify the parameters passed to the skill were correct.",
            "  5. Consider consulting the available program support cards for guidance.",
        ]
        return "\n".join(lines)

    def get_events(self) -> List[ProceduralEvent]:
        """Get all events recorded."""
        return self._events.copy()
