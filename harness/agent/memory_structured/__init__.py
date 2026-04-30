"""
StructuredMemoryStore module for Recipe H2 — Structured State Tracker.

Provides structured state tracking with four slots:
- constraints: task constraints extracted from tool results
- derived_facts: facts derived from tool results (not directly observed)
- pending_subgoals: subgoals pending completion
- artifact_paths: paths to files created or modified

This harness verifies H2: agent memory failures stem from not transforming
state into decision-relevant representations.
"""

from .slot import StateSlot
from .store import StructuredMemoryStore
from .config import StructuredMemoryConfig
from .update_policy import UpdatePolicy, SlotUpdateResult

__all__ = [
    "StateSlot",
    "StructuredMemoryStore",
    "StructuredMemoryConfig",
    "UpdatePolicy",
    "SlotUpdateResult",
]
