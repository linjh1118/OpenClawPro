"""
Procedural Trigger - Recipe T4: Procedural Support

T4a: Matches task description against program support cards via dense retrieval.
T4b: Detects unexpected tool results that should re-trigger skill activation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Set, Tuple

from .config import ProceduralConfig, SkillActivationConfig, SkillCard
from .event import ProceduralEvent
from .retriever import DenseRetriever
from .store import ProceduralStore

logger = logging.getLogger("agent.procedure.trigger")


class ProceduralTrigger:
    """Handles T4a retrieval and T4b unexpected-result detection.

    T4a: Uses dense passage retrieval (BERT bi-encoder) to identify
         the top-k most relevant program support cards for the task.

    T4b: Monitors tool results for unexpected outputs and triggers
         skill activation re-injection when threshold is met.
    """

    def __init__(
        self,
        config: ProceduralConfig,
        store: ProceduralStore,
    ):
        self.config = config
        self.store = store
        self._events: List[ProceduralEvent] = []
        self._iteration = 0

        # T4a: Dense retriever
        self._retriever: DenseRetriever | None = None
        self._retriever_initialized = False

        # T4b: Unexpected result tracking
        self._unexpected_count = 0
        self._unexpected_triggered = False

        # Backward-compatible keyword trigger cache
        self._cached_triggers: Set[str] = set()

        # Index cards for dense retrieval
        if config.enabled and config.program_support.enabled:
            self._init_retriever()

    def _init_retriever(self) -> None:
        """Initialize dense retriever and index cards."""
        if self._retriever_initialized:
            return

        try:
            self._retriever = DenseRetriever(self.config.program_support)
            all_cards = self.store.get_all()
            if all_cards:
                self._retriever.index_cards(all_cards)
                self._retriever_initialized = True
                logger.info(f"[ProceduralTrigger] Dense retriever initialized with {len(all_cards)} cards")
        except Exception as e:
            logger.warning(f"[ProceduralTrigger] Failed to initialize dense retriever: {e}. Falling back to keyword matching.")
            self._retriever = None
            self._retriever_initialized = True

    def increment_iteration(self) -> None:
        """Increment iteration counter."""
        self._iteration += 1

    # ─────────────────────────────────────────
    # T4a: Program Support Card Retrieval
    # ─────────────────────────────────────────

    def retrieve_cards(
        self,
        task: str,
        context: str | None = None,
        domain: str | None = None,
    ) -> List[Tuple[SkillCard, float, List[str]]]:
        """Retrieve top-k relevant program support cards via dense retrieval.

        Args:
            task: The current task description
            context: Optional additional context
            domain: Optional domain to filter cards

        Returns:
            List of (SkillCard, similarity_score, matched_keywords) tuples
        """
        if not self.config.enabled or not self.config.program_support.enabled:
            return []

        self._init_retriever()

        query = task
        if context:
            query = f"{task} {context}"

        results: List[Tuple[SkillCard, float, List[str]]] = []

        # Try dense retrieval first
        if self._retriever is not None:
            retrieved = self._retriever.retrieve(query)
            for card_name_or_result, score, matched in retrieved:
                # Handle both old-style (name) and new-style (name,) returns
                if isinstance(card_name_or_result, str):
                    card = self.store.get(card_name_or_result)
                else:
                    card = card_name_or_result

                if card is None:
                    continue

                # Domain filter
                if domain and card.domain and card.domain != domain:
                    continue

                results.append((card, score, matched))

                event = ProceduralEvent(
                    event_type="program_card_retrieved",
                    skill_name=card.name,
                    matched_keywords=matched,
                    data={
                        "similarity": score,
                        "task_preview": task[:100],
                        "domain": domain or "",
                    },
                )
                self._events.append(event)

            if results:
                logger.info(
                    f"[ProceduralTrigger] Retrieved {len(results)} cards via dense retrieval "
                    f"(top score: {results[0][1]:.4f})"
                )

        # Keyword fallback
        if not results and self.config.program_support.use_keyword_fallback:
            results = self._keyword_fallback(task, context, domain)

        return results

    def _keyword_fallback(
        self,
        task: str,
        context: str | None,
        domain: str | None,
    ) -> List[Tuple[SkillCard, float, List[str]]]:
        """Fallback keyword-based trigger matching."""
        text_to_check = task + (" " + context if context else "")
        text_lower = text_to_check.lower()

        results: List[Tuple[SkillCard, float, List[str]]] = []

        for card in self.store.get_all():
            if domain and card.domain and card.domain != domain:
                continue

            matched_keywords = self._find_matched_keywords(card, text_lower)
            if matched_keywords:
                results.append((card, 0.5, matched_keywords))
                self._cached_triggers.add(card.name)

                event = ProceduralEvent(
                    event_type="program_card_retrieved",
                    skill_name=card.name,
                    matched_keywords=matched_keywords,
                    data={"fallback": True, "task_preview": task[:100]},
                )
                self._events.append(event)

        results.sort(key=lambda x: len(x[2]), reverse=True)
        return results[: self.config.max_expansions_per_iteration]

    def _find_matched_keywords(self, card: SkillCard, text_lower: str) -> List[str]:
        """Find which keywords from a card match in the text."""
        matched = []
        for keyword in card.trigger_keywords:
            keyword_lower = keyword.lower()
            pattern = r'\b' + re.escape(keyword_lower) + r'\b'
            if re.search(pattern, text_lower):
                matched.append(keyword)
            elif len(keyword_lower) >= 5 and keyword_lower in text_lower:
                matched.append(keyword)
        return matched

    # ─────────────────────────────────────────
    # T4b: Unexpected Result Detection
    # ─────────────────────────────────────────

    def check_unexpected_result(
        self,
        tool_name: str,
        tool_result: str | dict | None,
    ) -> bool:
        """Check if a tool result is unexpected and should trigger re-injection.

        Args:
            tool_name: Name of the tool that was called
            tool_result: The result returned by the tool

        Returns:
            True if skill activation should be re-triggered
        """
        if not self.config.enabled or not self.config.skill_activation.enabled:
            return False

        if not self.config.skill_activation.retrigger_on_unexpected:
            return False

        result_str = ""
        if isinstance(tool_result, dict):
            result_str = json.dumps(tool_result, ensure_ascii=False).lower()
        elif isinstance(tool_result, str):
            result_str = tool_result.lower()
        else:
            result_str = str(tool_result).lower()

        unexpected_kw = self.config.skill_activation.unexpected_keywords
        found_keywords = [
            kw for kw in unexpected_kw
            if kw.lower() in result_str
        ]

        if found_keywords:
            self._unexpected_count += 1
            event = ProceduralEvent(
                event_type="unexpected_tool_result",
                skill_name=tool_name,
                data={
                    "unexpected_count": self._unexpected_count,
                    "matched_keywords": found_keywords,
                    "threshold": self.config.skill_activation.unexpected_threshold,
                },
            )
            self._events.append(event)

            threshold = self.config.skill_activation.unexpected_threshold
            if self._unexpected_count >= threshold and not self._unexpected_triggered:
                self._unexpected_triggered = True
                retrigger_event = ProceduralEvent(
                    event_type="skill_activation_retriggered",
                    skill_name="",
                    data={
                        "trigger": "unexpected_tool_result",
                        "unexpected_count": self._unexpected_count,
                    },
                )
                self._events.append(retrigger_event)
                logger.info(
                    f"[ProceduralTrigger] T4b re-trigger: unexpected result detected "
                    f"(count={self._unexpected_count}, tool={tool_name})"
                )
                return True

        return False

    def reset_unexpected_tracking(self) -> None:
        """Reset unexpected result counter."""
        self._unexpected_count = 0
        self._unexpected_triggered = False

    def is_unexpected_triggered(self) -> bool:
        """Check if skill activation re-trigger was just triggered this iteration."""
        return self._unexpected_triggered

    def clear_unexpected_triggered(self) -> None:
        """Clear the unexpected triggered flag (call after handling)."""
        self._unexpected_triggered = False

    # ─────────────────────────────────────────
    # Common
    # ─────────────────────────────────────────

    def get_triggered_skills(self) -> Set[str]:
        """Get set of skill names that have been triggered."""
        return self._cached_triggers.copy()

    def get_events(self) -> List[ProceduralEvent]:
        """Get all events recorded."""
        return self._events.copy()

    def reset(self) -> None:
        """Reset trigger state."""
        self._cached_triggers.clear()
        self._events.clear()
        self._iteration = 0
        self._unexpected_count = 0
        self._unexpected_triggered = False
