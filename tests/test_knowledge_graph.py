"""Unit tests for the collision knowledge graph module.

Tests cover:
- Graph construction from default seed data and from dict
- Node/edge counts and types
- GraphML export/import round-trip
- Pattern retrieval scoring logic
- Regulation retrieval filtering
- RAG context formatting
- Verification misidentification check formatting
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from video_analysis_poc.video_llm.knowledge_graph.graph import (
    CollisionKnowledgeGraph,
)
from video_analysis_poc.video_llm.knowledge_graph.retriever import (
    KGRetriever,
    RetrievalWeights,
)
from video_analysis_poc.video_llm.knowledge_graph.formatter import (
    format_rag_context,
    format_verification_misid_check,
)
from video_analysis_poc.video_llm.knowledge_graph.models import (
    CollisionPattern,
    ScoredPattern,
    TrafficRegulation,
    ReasoningTrace,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def default_kg() -> CollisionKnowledgeGraph:
    """Build the default knowledge graph from embedded seed data."""
    return CollisionKnowledgeGraph.from_defaults()


@pytest.fixture
def retriever(default_kg: CollisionKnowledgeGraph) -> KGRetriever:
    """Create a retriever with the default KG."""
    return KGRetriever(default_kg)


# ── Graph construction tests ──────────────────────────────────────────


class TestCollisionKnowledgeGraph:
    def test_from_defaults_creates_non_empty_graph(self, default_kg):
        assert default_kg.num_nodes > 0
        assert default_kg.num_edges > 0

    def test_from_defaults_has_expected_node_types(self, default_kg):
        node_types = {
            d.get("node_type")
            for _, d in default_kg.graph.nodes(data=True)
        }
        assert "vehicle_type" in node_types
        assert "road_geometry" in node_types
        assert "collision_pattern" in node_types
        assert "regulation" in node_types

    def test_from_defaults_pattern_count(self, default_kg):
        patterns = default_kg.pattern_nodes()
        assert len(patterns) == 5

    def test_from_defaults_regulation_count(self, default_kg):
        regulations = default_kg.regulation_nodes()
        assert len(regulations) == 4

    def test_from_dict_with_minimal_data(self):
        data = {
            "vehicle_types": [
                {"id": "car", "label": "Car", "swept_path_risk": "low"}
            ],
            "road_types": [
                {"id": "urban_road", "multi_lane": False, "turning_required": False}
            ],
            "patterns": [
                {
                    "pattern_id": "test_pattern",
                    "pattern_name": "Test",
                    "description": "A test pattern.",
                    "vehicle_types": ["car"],
                    "road_types": ["urban_road"],
                    "keywords": ["test"],
                }
            ],
            "regulations": [],
        }
        kg = CollisionKnowledgeGraph.from_dict(data)
        assert kg.num_nodes == 3  # 1 vtype + 1 road + 1 pattern
        assert kg.num_edges == 2  # INVOLVES + OCCURS_ON

    def test_successors_by_relation(self, default_kg):
        # The roundabout_sideswipe_hgv pattern should INVOLVE truck and car
        involves = default_kg.successors_by_relation(
            "pattern:roundabout_sideswipe_hgv", "INVOLVES"
        )
        involved_ids = {n.split(":")[1] for n in involves}
        assert "truck" in involved_ids
        assert "car" in involved_ids

    def test_causal_edge_exists(self, default_kg):
        followers = default_kg.successors_by_relation(
            "pattern:roundabout_sideswipe_hgv", "OFTEN_FOLLOWED_BY"
        )
        assert "pattern:hit_and_run_scrape" in followers

    def test_graphml_roundtrip(self, default_kg):
        with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
            path = Path(f.name)

        try:
            default_kg.to_graphml(path)
            assert path.exists()
            assert path.stat().st_size > 0

            reloaded = CollisionKnowledgeGraph.from_graphml(path)
            assert reloaded.num_nodes == default_kg.num_nodes
            assert reloaded.num_edges == default_kg.num_edges
        finally:
            path.unlink(missing_ok=True)


# ── Retriever tests ───────────────────────────────────────────────────


class TestKGRetriever:
    def test_retrieve_patterns_roundabout_truck_car(self, retriever):
        results = retriever.retrieve_patterns(
            detected_vehicle_types=["truck", "car"],
            road_type="roundabout",
        )
        assert len(results) > 0
        # The HGV sideswipe pattern should rank highest
        assert results[0].pattern.pattern_id == "roundabout_sideswipe_hgv"

    def test_retrieve_patterns_with_keywords_boosts_score(self, retriever):
        without_kw = retriever.retrieve_patterns(
            detected_vehicle_types=["truck", "car"],
            road_type="roundabout",
        )
        with_kw = retriever.retrieve_patterns(
            detected_vehicle_types=["truck", "car"],
            road_type="roundabout",
            keyword_hints=["sideswipe", "roundabout", "truck"],
        )
        # Score should be higher with keyword matches
        assert with_kw[0].relevance_score > without_kw[0].relevance_score

    def test_retrieve_patterns_returns_empty_for_no_match(self, retriever):
        results = retriever.retrieve_patterns(
            detected_vehicle_types=["bicycle"],
            road_type="rural_road",
        )
        assert len(results) == 0

    def test_retrieve_patterns_top_k(self, retriever):
        results = retriever.retrieve_patterns(
            detected_vehicle_types=["car", "truck", "van"],
            road_type="roundabout",
            top_k=2,
        )
        assert len(results) <= 2

    def test_retrieve_regulations_uk_roundabout(self, retriever):
        regs = retriever.retrieve_regulations(
            road_type="roundabout", jurisdiction="UK"
        )
        reg_ids = {r.reg_id for r in regs}
        assert "uk_roundabout_priority" in reg_ids
        assert "uk_hgv_swept_path" in reg_ids

    def test_retrieve_regulations_includes_general(self, retriever):
        regs = retriever.retrieve_regulations(
            road_type="roundabout", jurisdiction="UK"
        )
        reg_ids = {r.reg_id for r in regs}
        # uk_hit_and_run is category "general" — should be included
        assert "uk_hit_and_run" in reg_ids

    def test_retrieve_regulations_wrong_jurisdiction(self, retriever):
        regs = retriever.retrieve_regulations(
            road_type="roundabout", jurisdiction="US"
        )
        assert len(regs) == 0

    def test_custom_weights(self, default_kg):
        heavy_road = RetrievalWeights(
            vehicle_type_match=1.0,
            road_geometry_match=10.0,
            keyword_match=0.0,
        )
        retriever = KGRetriever(default_kg, weights=heavy_road)
        results = retriever.retrieve_patterns(
            detected_vehicle_types=["car"],
            road_type="roundabout",
        )
        # With road weight at 10, roundabout patterns should dominate
        roundabout_pattern_ids = {
            r.pattern.pattern_id
            for r in results[:3]
            if "roundabout" in r.pattern.road_types
        }
        assert len(roundabout_pattern_ids) > 0


# ── Formatter tests ───────────────────────────────────────────────────


class TestFormatter:
    def _make_scored_pattern(self, pid: str = "test") -> ScoredPattern:
        return ScoredPattern(
            pattern=CollisionPattern(
                pattern_id=pid,
                pattern_name=f"Test Pattern {pid}",
                description="A test pattern.",
                visual_signatures=["sig1", "sig2"],
                causal_chain="A -> B -> C",
                common_misidentifications=["WRONG: error1", "WRONG: error2"],
                keywords=["test"],
            ),
            relevance_score=5.0,
        )

    def _make_regulation(self) -> TrafficRegulation:
        return TrafficRegulation(
            reg_id="test_reg",
            jurisdiction="UK",
            category="general",
            rule_text="Test rule text.",
            analysis_implication="Test implication.",
        )

    def test_format_rag_context_includes_pattern_name(self):
        sp = self._make_scored_pattern()
        text = format_rag_context([sp], [])
        assert "Test Pattern test" in text

    def test_format_rag_context_includes_errors_to_avoid(self):
        sp = self._make_scored_pattern()
        text = format_rag_context([sp], [])
        assert "ERRORS TO AVOID" in text
        assert "error1" in text

    def test_format_rag_context_includes_regulations(self):
        reg = self._make_regulation()
        text = format_rag_context([], [reg])
        assert "APPLICABLE REGULATIONS" in text
        assert "Test rule text" in text

    def test_format_rag_context_respects_max_patterns(self):
        patterns = [self._make_scored_pattern(f"p{i}") for i in range(5)]
        text = format_rag_context(patterns, [], max_patterns=2)
        assert "Test Pattern p0" in text
        assert "Test Pattern p1" in text
        assert "Test Pattern p2" not in text

    def test_format_verification_misid_check_with_pattern(self):
        sp = self._make_scored_pattern()
        text = format_verification_misid_check(sp)
        assert "KB CHECK" in text
        assert "Test Pattern test" in text
        assert "error1" in text

    def test_format_verification_misid_check_none(self):
        text = format_verification_misid_check(None)
        assert text == ""


# ── Model tests ───────────────────────────────────────────────────────


class TestModels:
    def test_reasoning_trace_validation(self):
        rt = ReasoningTrace(
            global_perception="Scene overview",
            vehicle_inventory="V1: truck, V2: car",
            interaction_analysis="V1 encroaches on V2 lane",
            causal_conclusion="Sideswipe caused by HGV swept path",
        )
        assert rt.global_perception == "Scene overview"

    def test_scored_pattern_properties(self):
        sp = ScoredPattern(
            pattern=CollisionPattern(
                pattern_id="test_id",
                pattern_name="Test Name",
                description="Desc",
            ),
            relevance_score=7.5,
        )
        assert sp.pattern_id == "test_id"
        assert sp.pattern_name == "Test Name"
