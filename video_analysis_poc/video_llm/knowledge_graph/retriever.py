"""Scored graph-traversal retriever for the collision knowledge graph.

``KGRetriever`` ranks collision patterns by matching detected vehicle types,
road geometry, and keyword hints extracted from the scene.  It also retrieves
applicable traffic regulations filtered by road type and jurisdiction.

Scoring heuristic (configurable via constructor weights):
    +2  per matching vehicle type  (pattern INVOLVES vtype in detected set)
    +3  if road geometry matches   (pattern OCCURS_ON detected road type)
    +1  per matching keyword hint
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from video_analysis_poc.video_llm.knowledge_graph.graph import (
    CollisionKnowledgeGraph,
)
from video_analysis_poc.video_llm.knowledge_graph.models import (
    CollisionPattern,
    ScoredPattern,
    TrafficRegulation,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievalWeights:
    """Scoring weights for pattern retrieval."""

    vehicle_type_match: float = 2.0
    road_geometry_match: float = 3.0
    keyword_match: float = 1.0


class KGRetriever:
    """Retrieves and ranks collision patterns from the knowledge graph.

    Parameters:
        kg: A populated ``CollisionKnowledgeGraph`` instance.
        weights: Optional scoring weights override.
    """

    def __init__(
        self,
        kg: CollisionKnowledgeGraph,
        weights: RetrievalWeights | None = None,
    ) -> None:
        self._kg = kg
        self._weights = weights or RetrievalWeights()

    def retrieve_patterns(
        self,
        detected_vehicle_types: list[str],
        road_type: str | None = None,
        keyword_hints: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[ScoredPattern]:
        """Retrieve and rank collision patterns by scene metadata.

        Parameters:
            detected_vehicle_types: Vehicle types observed in the scene
                (e.g. ``["truck", "car", "van"]``).
            road_type: Detected road geometry (e.g. ``"roundabout"``).
            keyword_hints: Optional keyword cues extracted from a preliminary
                scene description or metadata.
            top_k: If set, return only the top-k highest-scoring patterns.

        Returns:
            A list of ``ScoredPattern`` instances sorted by descending
            relevance score.  Only patterns with score > 0 are returned.
        """
        results: list[ScoredPattern] = []
        w = self._weights

        for node_id, attrs in self._kg.pattern_nodes():
            score = 0.0

            # Vehicle type matching
            involves_nodes = self._kg.successors_by_relation(node_id, "INVOLVES")
            matched_vtypes = [
                n.split(":")[1]
                for n in involves_nodes
                if n.split(":")[1] in detected_vehicle_types
            ]
            score += len(matched_vtypes) * w.vehicle_type_match

            # Road geometry matching
            if road_type:
                occurs_on_nodes = self._kg.successors_by_relation(node_id, "OCCURS_ON")
                if any(road_type in n for n in occurs_on_nodes):
                    score += w.road_geometry_match

            # Keyword matching
            if keyword_hints:
                pattern_keywords = set(attrs.get("keywords", []))
                score += len(set(keyword_hints) & pattern_keywords) * w.keyword_match

            if score > 0:
                pattern = CollisionPattern(
                    pattern_id=attrs.get("pattern_id", ""),
                    pattern_name=attrs.get("pattern_name", ""),
                    description=attrs.get("description", ""),
                    visual_signatures=attrs.get("visual_signatures", []),
                    causal_chain=attrs.get("causal_chain", ""),
                    common_misidentifications=attrs.get(
                        "common_misidentifications", []
                    ),
                    vehicle_types=attrs.get("vehicle_types", []),
                    road_types=attrs.get("road_types", []),
                    keywords=attrs.get("keywords", []),
                    example_narrative=attrs.get("example_narrative", ""),
                )
                results.append(ScoredPattern(pattern=pattern, relevance_score=score))

        results.sort(key=lambda sp: sp.relevance_score, reverse=True)

        if top_k is not None:
            results = results[:top_k]

        logger.info(
            "Retrieved %d collision pattern(s) (top score: %.1f).",
            len(results),
            results[0].relevance_score if results else 0.0,
        )
        return results

    def retrieve_regulations(
        self,
        road_type: str | None = None,
        jurisdiction: str = "UK",
    ) -> list[TrafficRegulation]:
        """Retrieve applicable traffic regulations.

        Returns regulations that either apply to the given road type via an
        ``APPLIES_TO`` edge, or have ``category == "general"``.

        Parameters:
            road_type: The detected road geometry.
            jurisdiction: Jurisdiction filter (default ``"UK"``).

        Returns:
            A list of ``TrafficRegulation`` instances.
        """
        results: list[TrafficRegulation] = []

        for node_id, attrs in self._kg.regulation_nodes():
            if attrs.get("jurisdiction") != jurisdiction:
                continue

            applies_to = self._kg.successors_by_relation(node_id, "APPLIES_TO")
            is_applicable = attrs.get("category") == "general" or (
                road_type and any(road_type in n for n in applies_to)
            )

            if is_applicable:
                results.append(
                    TrafficRegulation(
                        reg_id=attrs.get("reg_id", ""),
                        jurisdiction=attrs.get("jurisdiction", ""),
                        category=attrs.get("category", ""),
                        rule_text=attrs.get("rule_text", ""),
                        analysis_implication=attrs.get("analysis_implication", ""),
                    )
                )

        logger.info(
            "Retrieved %d regulation(s) for jurisdiction=%s, road=%s.",
            len(results),
            jurisdiction,
            road_type,
        )
        return results
