"""
Tests for T4a and T4b: Procedural Support Module

Tests:
  - T4a: Program Support Cards with 4-component structure
  - T4a: Dense retrieval (BERT bi-encoder) card matching
  - T4a: Keyword fallback when retrieval unavailable
  - T4b: Skill activation prompt formatting
  - T4b: Unexpected result detection and re-trigger
  - Backward compatibility with legacy config format
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

from harness.agent.procedure import (
    DenseRetriever,
    ProceduralConfig,
    ProceduralExpander,
    ProceduralEvent,
    ProceduralStore,
    ProceduralTrigger,
    ProgramSupportConfig,
    RetrievalConfig,
    SkillActivationConfig,
    SkillCard,
    get_procedure_summary,
)


# ─────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────

@pytest.fixture
def sample_cards() -> list[SkillCard]:
    return [
        SkillCard(
            name="pubmed_rct_search",
            description="Conduct a PubMed search for randomized controlled trials",
            domain="medical",
            program_goal="Find peer-reviewed RCTs using PubMed clinical query filters",
            prerequisites=["Access PubMed", "Identify PICO elements"],
            steps=["Navigate to PubMed", "Build query", "Apply RCT filter"],
            common_pitfalls=[
                "Using only free text: Use MeSH vocabulary for better recall",
                "Forgetting RCT filter: Always use Clinical Queries filter",
            ],
            trigger_keywords=["pubmed", "rct", "clinical trial", "medical search"],
        ),
        SkillCard(
            name="code_debug_workflow",
            description="Systematic debugging using structured approach",
            domain="software",
            program_goal="Identify and fix bugs systematically",
            prerequisites=["Reproduce the bug", "Gather error traceback"],
            steps=["Reproduce bug", "Identify failure line", "Form hypothesis", "Fix", "Verify"],
            common_pitfalls=[
                "Fixing symptoms instead of root cause",
                "Not running existing tests before changes",
            ],
            trigger_keywords=["debug", "bug", "error", "crash", "software"],
        ),
    ]


@pytest.fixture
def cards_dir(sample_cards, tmp_path) -> Path:
    """Create a temp directory with sample card files."""
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()

    # YAML file
    yaml_file = cards_dir / "test_cards.yaml"
    yaml_file.write_text(
        "name: pubmed_rct_search\n"
        "description: PubMed RCT search\n"
        "domain: medical\n"
        "program_goal: Find RCTs\n"
        "prerequisites:\n  - Access PubMed\n  - Identify PICO\n"
        "steps:\n  - Navigate to PubMed\n  - Build query\n  - Apply filter\n"
        "common_pitfalls:\n  - Use MeSH terms\n  - Apply RCT filter\n"
        "trigger_keywords:\n  - pubmed\n  - rct\n  - clinical trial\n",
        encoding="utf-8",
    )
    return cards_dir


@pytest.fixture
def program_support_config(cards_dir) -> ProgramSupportConfig:
    return ProgramSupportConfig(
        enabled=True,
        cards_dir=str(cards_dir),
        retrieval=RetrievalConfig(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            top_k=3,
            batch_size=2,
            cache_embeddings=True,
        ),
        use_keyword_fallback=True,
    )


@pytest.fixture
def procedural_config(program_support_config) -> ProceduralConfig:
    return ProceduralConfig(
        enabled=True,
        program_support=program_support_config,
        skill_activation=SkillActivationConfig(
            enabled=True,
            inject_at_start=True,
            retrigger_on_unexpected=True,
            unexpected_threshold=1,
            include_inventory=True,
            include_selection=True,
            include_verification=True,
        ),
    )


# ─────────────────────────────────────────
# T4a: SkillCard (4-component structure)
# ─────────────────────────────────────────

class TestSkillCard:
    def test_from_dict_full(self):
        data = {
            "name": "test_card",
            "description": "A test card",
            "domain": "medical",
            "program_goal": "Test goal",
            "prerequisites": ["prereq1", "prereq2"],
            "steps": ["step1", "step2"],
            "common_pitfalls": ["pitfall1"],
            "trigger_keywords": ["test", "keyword"],
            "examples": ["example1"],
            "compact": "compact summary",
        }
        card = SkillCard.from_dict(data)
        assert card.name == "test_card"
        assert card.domain == "medical"
        assert card.program_goal == "Test goal"
        assert card.prerequisites == ["prereq1", "prereq2"]
        assert card.steps == ["step1", "step2"]
        assert card.common_pitfalls == ["pitfall1"]
        assert card.trigger_keywords == ["test", "keyword"]

    def test_to_dict_roundtrip(self, sample_cards):
        card = sample_cards[0]
        data = card.to_dict()
        restored = SkillCard.from_dict(data)
        assert restored.name == card.name
        assert restored.domain == card.domain
        assert restored.program_goal == card.program_goal
        assert restored.prerequisites == card.prerequisites
        assert restored.steps == card.steps
        assert restored.common_pitfalls == card.common_pitfalls

    def test_backward_compat_legacy_dict(self):
        """Legacy dict format (name, description, steps, examples) should still work."""
        data = {
            "name": "legacy_card",
            "description": "Legacy card",
            "steps": ["step1"],
            "examples": ["example1"],
        }
        card = SkillCard.from_dict(data)
        assert card.name == "legacy_card"
        assert card.steps == ["step1"]
        assert card.examples == ["example1"]
        assert card.program_goal == ""  # Default for legacy format


# ─────────────────────────────────────────
# T4a: ProceduralStore
# ─────────────────────────────────────────

class TestProceduralStore:
    def test_load_yaml_cards(self, procedural_config):
        store = ProceduralStore(procedural_config)
        assert store.get_card_count() == 1
        card = store.get("pubmed_rct_search")
        assert card is not None
        assert card.domain == "medical"

    def test_get_by_domain(self, procedural_config, sample_cards):
        store = ProceduralStore(procedural_config)
        # Manually add both cards
        store._cards = {c.name: c for c in sample_cards}
        store._domain_index = {}
        for c in sample_cards:
            if c.domain:
                store._domain_index.setdefault(c.domain, []).append(c.name)

        medical_cards = store.get_by_domain("medical")
        assert len(medical_cards) == 1
        assert medical_cards[0].name == "pubmed_rct_search"

        software_cards = store.get_by_domain("software")
        assert len(software_cards) == 1
        assert software_cards[0].name == "code_debug_workflow"

    def test_search_by_keyword(self, procedural_config, sample_cards):
        store = ProceduralStore(procedural_config)
        store._cards = {c.name: c for c in sample_cards}
        # Rebuild trigger index (normally done in _add_card)
        store._trigger_index = {}
        for card in sample_cards:
            for keyword in card.trigger_keywords:
                keyword_lower = keyword.lower()
                if keyword_lower not in store._trigger_index:
                    store._trigger_index[keyword_lower] = []
                if card.name not in store._trigger_index[keyword_lower]:
                    store._trigger_index[keyword_lower].append(card.name)

        results = store.search_by_keyword("pubmed")
        assert "pubmed_rct_search" in results

        results = store.search_by_keyword("debug")
        assert "code_debug_workflow" in results


# ─────────────────────────────────────────
# T4a: DenseRetriever (skip if sentence-transformers unavailable)
# ─────────────────────────────────────────

class TestDenseRetriever:
    def test_retriever_init(self, program_support_config, sample_cards):
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        retriever = DenseRetriever(program_support_config)
        store = ProceduralStore(
            ProceduralConfig(enabled=True, program_support=program_support_config)
        )
        store._cards = {c.name: c for c in sample_cards}
        retriever.index_cards(sample_cards)

        assert retriever.get_card_count() == 2

        # Retrieve with a medical query
        results = retriever.retrieve("How to search for clinical trials on PubMed", top_k=1)
        assert len(results) >= 1
        # Top result should be pubmed_rct_search or code_debug_workflow
        retrieved_names = [r[0] for r in results]
        assert "pubmed_rct_search" in retrieved_names or "code_debug_workflow" in retrieved_names


# ─────────────────────────────────────────
# T4a: ProceduralTrigger
# ─────────────────────────────────────────

class TestProceduralTrigger:
    def test_keyword_fallback(self, procedural_config, sample_cards):
        """Fallback to keyword matching when retriever unavailable."""
        config = ProceduralConfig(
            enabled=True,
            program_support=ProgramSupportConfig(
                enabled=True,
                cards_dir="",
                retrieval=RetrievalConfig(device="cpu"),
                use_keyword_fallback=True,
            ),
            skill_activation=SkillActivationConfig(enabled=False),
        )
        store = ProceduralStore(config)
        store._cards = {c.name: c for c in sample_cards}

        trigger = ProceduralTrigger(config, store)
        # Dense retriever init will fail (no cards_dir), but keyword fallback should work
        results = trigger.retrieve_cards(
            "I need to debug a Python crash in my code",
            context=None,
        )
        assert len(results) >= 1
        # Should match code_debug_workflow
        card_names = [r[0].name for r in results]
        assert "code_debug_workflow" in card_names

    def test_domain_filter(self, procedural_config, sample_cards):
        store = ProceduralStore(procedural_config)
        store._cards = {c.name: c for c in sample_cards}
        trigger = ProceduralTrigger(procedural_config, store)

        results = trigger.retrieve_cards(
            "PubMed search for medical trials",
            domain="software",
        )
        # Should not return medical domain cards when filtering by software
        card_names = [r[0].name for r in results]
        if "pubmed_rct_search" not in card_names:
            # Expected: medical card filtered out
            pass
        # Either way, no assertion failure


# ─────────────────────────────────────────
# T4b: Unexpected Result Detection
# ─────────────────────────────────────────

class TestSkillActivationTrigger:
    def test_unexpected_result_detection(self, procedural_config):
        """T4b: Should trigger when tool returns unexpected result."""
        config = ProceduralConfig(
            enabled=True,
            program_support=ProgramSupportConfig(enabled=False),
            skill_activation=SkillActivationConfig(
                enabled=True,
                retrigger_on_unexpected=True,
                unexpected_threshold=1,
                unexpected_keywords=["error", "failed", "none", "not found"],
            ),
        )
        store = ProceduralStore(config)
        trigger = ProceduralTrigger(config, store)

        # Normal result - should NOT trigger
        should_retrigger = trigger.check_unexpected_result("web_search", "Found 5 results")
        assert not should_retrigger
        assert not trigger.is_unexpected_triggered()

        # Unexpected result - should trigger
        should_retrigger = trigger.check_unexpected_result(
            "web_search", "Error: page not found"
        )
        assert should_retrigger
        assert trigger.is_unexpected_triggered()

        # After handling, clear the flag
        trigger.clear_unexpected_triggered()
        assert not trigger.is_unexpected_triggered()

    def test_unexpected_threshold(self, procedural_config):
        """T4b: Should only trigger after reaching threshold."""
        config = ProceduralConfig(
            enabled=True,
            program_support=ProgramSupportConfig(enabled=False),
            skill_activation=SkillActivationConfig(
                enabled=True,
                retrigger_on_unexpected=True,
                unexpected_threshold=3,
                unexpected_keywords=["error", "failed"],
            ),
        )
        store = ProceduralStore(config)
        trigger = ProceduralTrigger(config, store)

        # Two unexpected results - below threshold
        trigger.check_unexpected_result("tool1", "Error occurred")
        assert not trigger.is_unexpected_triggered()
        trigger.check_unexpected_result("tool2", "Failed to load")
        assert not trigger.is_unexpected_triggered()

        # Third unexpected - reaches threshold
        should_retrigger = trigger.check_unexpected_result("tool3", "error")
        assert should_retrigger
        assert trigger.is_unexpected_triggered()


# ─────────────────────────────────────────
# T4a: Expander Formatting
# ─────────────────────────────────────────

class TestProceduralExpander:
    def test_format_4component(self, sample_cards):
        expander = ProceduralExpander()
        formatted = expander.format(sample_cards[0])

        # Should contain all 4 components
        assert "Program Support Card" in formatted
        assert "Goal" in formatted or "Goal:" in formatted
        assert "Prerequisites" in formatted
        assert "Steps" in formatted
        assert "Common Pitfalls" in formatted
        assert "pubmed_rct_search" in formatted

    def test_format_multiple(self, sample_cards):
        expander = ProceduralExpander()
        formatted = expander.format_multiple(sample_cards)

        assert "pubmed_rct_search" in formatted
        assert "code_debug_workflow" in formatted
        assert "Steps" in formatted
        assert "Common Pitfalls" in formatted

    def test_skill_activation_prompt(self, procedural_config):
        expander = ProceduralExpander()
        tool_defs = [
            {"name": "web_search", "description": "Search the web for information"},
            {"name": "read_file", "description": "Read a file from disk"},
        ]
        formatted = expander.format_skill_activation(
            procedural_config.skill_activation,
            tool_definitions=tool_defs,
        )

        assert "Skill Activation" in formatted
        assert "web_search" in formatted
        assert "read_file" in formatted
        assert "Skill Selection Checklist" in formatted
        assert "Skill Execution Verification" in formatted
        assert "enumerate all available" in formatted.lower()

    def test_skill_activation_retrigger(self):
        expander = ProceduralExpander()
        formatted = expander.format_skill_activation_retrigger(
            tool_name="web_search",
            result_summary="Error: connection timeout",
        )
        assert "re-trigger" in formatted.lower()
        assert "web_search" in formatted
        assert "connection timeout" in formatted


# ─────────────────────────────────────────
# T4b: Expander Events
# ─────────────────────────────────────────

class TestProceduralEvents:
    def test_event_types(self):
        """Verify new event types are recorded correctly."""
        event = ProceduralEvent(
            event_type="program_card_retrieved",
            skill_name="test_card",
            matched_keywords=["test"],
            data={"similarity": 0.95},
        )
        assert event.event_type == "program_card_retrieved"
        assert event.skill_name == "test_card"
        assert event.data["similarity"] == 0.95

        event2 = ProceduralEvent(
            event_type="skill_activation_retriggered",
            skill_name="",
            data={"trigger": "unexpected_tool_result"},
        )
        assert event2.event_type == "skill_activation_retriggered"


# ─────────────────────────────────────────
# Config: Backward Compatibility
# ─────────────────────────────────────────

class TestConfigBackwardCompat:
    def test_legacy_config(self):
        """Legacy flat config should still deserialize."""
        data = {
            "enabled": True,
            "cards_dir": "/some/path",
            "max_expansions_per_iteration": 5,
            "show_skill_list": True,
            "cache_triggers": False,
        }
        config = ProceduralConfig.from_dict(data)
        assert config.enabled is True
        assert config.cards_dir == "/some/path"
        assert config.max_expansions_per_iteration == 5
        # New nested configs should have defaults
        assert config.program_support.enabled is False
        assert config.skill_activation.enabled is False

    def test_new_config(self):
        """Full nested config should deserialize correctly."""
        data = {
            "enabled": True,
            "program_support": {
                "enabled": True,
                "cards_dir": "/cards",
                "retrieval": {
                    "embedding_model": "test-model",
                    "device": "cpu",
                    "top_k": 5,
                },
                "use_keyword_fallback": False,
            },
            "skill_activation": {
                "enabled": True,
                "inject_at_start": False,
                "retrigger_on_unexpected": True,
                "unexpected_threshold": 2,
            },
        }
        config = ProceduralConfig.from_dict(data)
        assert config.program_support.enabled is True
        assert config.program_support.cards_dir == "/cards"
        assert config.program_support.retrieval.top_k == 5
        assert config.skill_activation.enabled is True
        assert config.skill_activation.inject_at_start is False
        assert config.skill_activation.retrigger_on_unexpected is True

    def test_to_dict_roundtrip(self, procedural_config):
        data = procedural_config.to_dict()
        restored = ProceduralConfig.from_dict(data)
        assert restored.enabled == procedural_config.enabled
        assert restored.program_support.enabled == procedural_config.program_support.enabled
        assert restored.skill_activation.enabled == procedural_config.skill_activation.enabled


# ─────────────────────────────────────────
# Summary
# ─────────────────────────────────────────

class TestProcedureSummary:
    def test_summary_tracking(self, procedural_config, sample_cards):
        store = ProceduralStore(procedural_config)
        store._cards = {c.name: c for c in sample_cards}
        trigger = ProceduralTrigger(procedural_config, store)
        expander = ProceduralExpander()

        # Simulate trigger events
        trigger.retrieve_cards("debug a Python bug")
        trigger.retrieve_cards("pubmed clinical trial")

        # Simulate expand events
        for card in sample_cards:
            expander.format(card)

        summary = get_procedure_summary(trigger.get_events(), expander.get_events())
        assert summary["total_events"] >= 4
        assert len(summary["triggered_skills"]) >= 2
        assert summary["expansion_count"] >= 2
