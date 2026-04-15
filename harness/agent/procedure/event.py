"""
Procedural Event - Recipe T4: Procedural Support

Event types:
  - T4a: program_card_retrieved, program_card_expanded
  - T4b: skill_activation_injected, skill_activation_retriggered
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProceduralEvent:
    """Structured event for procedural support tracking.

    T4a events:
      - program_card_retrieved: dense retrieval matched a card
      - program_card_expanded: card was formatted and injected

    T4b events:
      - skill_activation_injected: skill activation prompt injected at start
      - skill_activation_retriggered: re-triggered on unexpected tool result
      - unexpected_tool_result: detected unexpected result from a tool
    """
    event_type: str
    skill_name: str = ""
    matched_keywords: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float | None = None

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "skill_name": self.skill_name,
            "matched_keywords": self.matched_keywords,
            "data": self.data,
            "timestamp": self.timestamp,
        }
