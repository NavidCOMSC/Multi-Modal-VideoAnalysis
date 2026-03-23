"""Pydantic and dataclass models for the collision knowledge graph.

Defines the typed containers used throughout the KG module:
- Domain entities (``CollisionPattern``, ``TrafficRegulation``)
- Graph metadata types (``VehicleTypeNode``, ``RoadGeometryNode``)
- Retrieval results (``ScoredPattern``)
- Reasoning trace extension for chain-of-thought output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pydantic import BaseModel, Field


# ── Graph node value objects ──────────────────────────────────────────


@dataclass(frozen=True)
class VehicleTypeNode:
    """A vehicle-type node in the knowledge graph."""

    id: str
    label: str
    swept_path_risk: str  # "low" | "medium" | "high" | "none"


@dataclass(frozen=True)
class RoadGeometryNode:
    """A road-geometry node in the knowledge graph."""

    id: str
    multi_lane: bool
    turning_required: bool


# ── Domain entities ───────────────────────────────────────────────────


@dataclass
class CollisionPattern:
    """An archetype collision pattern stored as a graph node.

    Each pattern encodes visual signatures, a causal chain, and — crucially —
    a list of ``common_misidentifications`` that LLMs frequently make when
    analysing footage matching this pattern.
    """

    pattern_id: str
    pattern_name: str
    description: str
    visual_signatures: list[str] = field(default_factory=list)
    causal_chain: str = ""
    common_misidentifications: list[str] = field(default_factory=list)
    vehicle_types: list[str] = field(default_factory=list)
    road_types: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    severity_range: str = "low to medium"
    example_narrative: str = ""


@dataclass
class TrafficRegulation:
    """A jurisdiction-specific traffic regulation node."""

    reg_id: str
    jurisdiction: str
    category: str  # "roundabout" | "general" | ...
    rule_text: str
    analysis_implication: str


# ── Retrieval result ──────────────────────────────────────────────────


@dataclass
class ScoredPattern:
    """A collision pattern annotated with a retrieval relevance score."""

    pattern: CollisionPattern
    relevance_score: float

    @property
    def pattern_id(self) -> str:
        return self.pattern.pattern_id

    @property
    def pattern_name(self) -> str:
        return self.pattern.pattern_name


# ── Reasoning trace (chain-of-thought extension) ─────────────────────


class ReasoningTrace(BaseModel):
    """Structured reasoning trace populated by the LLM before the
    main scene-analysis output.

    Forces a perception-to-cognition chain:
    1. Global perception — scene-level observations.
    2. Vehicle inventory — all vehicles with assigned IDs.
    3. Interaction analysis — trajectory relationships and proximity.
    4. Causal conclusion — final event classification with evidence.
    """

    global_perception: str = Field(
        ..., description="Scene-level observations."
    )
    vehicle_inventory: str = Field(
        ..., description="All vehicles with IDs and descriptions."
    )
    interaction_analysis: str = Field(
        ..., description="Vehicle interaction analysis."
    )
    causal_conclusion: str = Field(
        ..., description="Final causal attribution with temporal evidence."
    )
