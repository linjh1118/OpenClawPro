"""StateSlot dataclass for structured memory."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class StateSlot:
    """Single slot in structured state tracking.

    Slots:
    - constraints: task constraints extracted from tool results
    - derived_facts: facts derived from tool results (not directly observed)
    - pending_subgoals: subgoals pending completion
    - artifact_paths: paths to files created or modified
    """

    slot_id: str
    slot_type: str  # "constraints" | "derived_facts" | "pending_subgoals" | "artifact_paths"
    content: str
    derived: bool = False  # True for derived_facts
    iteration_created: int = 0
    iteration_updated: int = 0
    source_event: str = ""  # tool_name or "derived" for derived facts
    version: int = 1
    active: bool = True  # False means marked complete/obsolete
    tags: List[str] = field(default_factory=list)  # For cross-referencing

    def touch(self, iteration: int, source_event: str = "") -> None:
        """Update the slot's last-modified metadata."""
        self.iteration_updated = iteration
        self.version += 1
        if source_event:
            self.source_event = source_event

    def to_dict(self) -> dict:
        return {
            "slot_id": self.slot_id,
            "slot_type": self.slot_type,
            "content": self.content,
            "derived": self.derived,
            "iteration_created": self.iteration_created,
            "iteration_updated": self.iteration_updated,
            "source_event": self.source_event,
            "version": self.version,
            "active": self.active,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StateSlot":
        return cls(**data)
