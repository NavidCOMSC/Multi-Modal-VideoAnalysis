"""Collision knowledge graph construction and persistence.

The ``CollisionKnowledgeGraph`` class wraps a ``networkx.DiGraph`` with
typed helper methods for adding collision patterns, vehicle types, road
geometries, and traffic regulations — plus the edges that connect them.

Two factory methods are provided:

- ``from_dict`` — builds the graph from an in-memory dictionary (the
  format used in the PoC notebooks).
- ``from_graphml`` — loads a previously-exported GraphML file (suitable
  for CI fixtures or production configuration).

The graph can be exported to GraphML via ``to_graphml`` for ingestion
into Neo4j, Azure Cosmos DB Gremlin, or other graph databases.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

# ── Default KG seed data ─────────────────────────────────────────────
# Embedded as a module constant so the graph can be built without any
# external file dependency.  In production this would be loaded from a
# managed configuration store or graph database.

_DEFAULT_KG_DATA: dict[str, Any] = {
    "vehicle_types": [
        {"id": "car", "label": "Car / SUV", "swept_path_risk": "low"},
        {"id": "truck", "label": "Truck / HGV / Lorry", "swept_path_risk": "high"},
        {"id": "van", "label": "Van / LCV", "swept_path_risk": "medium"},
        {"id": "bus", "label": "Bus / Coach", "swept_path_risk": "high"},
        {"id": "motorcycle", "label": "Motorcycle", "swept_path_risk": "low"},
        {"id": "pedestrian", "label": "Pedestrian", "swept_path_risk": "none"},
    ],
    "road_types": [
        {"id": "roundabout", "multi_lane": True, "turning_required": True},
        {"id": "motorway", "multi_lane": True, "turning_required": False},
        {"id": "intersection", "multi_lane": True, "turning_required": True},
        {"id": "urban_road", "multi_lane": False, "turning_required": False},
    ],
    "patterns": [
        {
            "pattern_id": "roundabout_sideswipe_hgv",
            "pattern_name": "HGV Sideswipe on Roundabout",
            "description": (
                "A large vehicle (truck/HGV) enters a roundabout and its swept "
                "turning path encroaches on an adjacent smaller vehicle, causing "
                "a side scrape."
            ),
            "visual_signatures": [
                "Large vehicle and smaller vehicle entering roundabout side-by-side",
                "Smaller vehicle suddenly stops on roundabout after truck passes",
                "Truck continues without stopping (scrape-and-leave)",
                "No rear-end impact visible - damage is lateral",
                "Gap between vehicles narrows as they navigate the curve",
            ],
            "causal_chain": (
                "HGV enters outer lane -> turning arc exceeds lane width -> "
                "trailer encroaches inner lane -> contact -> car stops, HGV continues"
            ),
            "common_misidentifications": [
                "WRONG: Vehicle described as inexplicably stopped (the stop was CAUSED by the sideswipe)",
                "WRONG: Later-arriving vehicle blamed for collision (they merely navigated around)",
                "WRONG: Impact classified as rear-end when actually lateral",
                "WRONG: Truck not identified as involved because it left the scene",
            ],
            "vehicle_types": ["truck", "car"],
            "road_types": ["roundabout"],
            "keywords": [
                "sideswipe",
                "swept path",
                "roundabout",
                "truck",
                "HGV",
                "scrape",
            ],
            "example_narrative": (
                "A green Yodel truck and black SUV approach a multi-lane roundabout. "
                "Truck in outer lane, SUV in inner lane. Both enter at ~15s. Between "
                "17-19s the truck rear swings into SUV lane. SUV stops at ~20s after "
                "being scraped. Truck continues. Later vehicles (van ~25s, red SUV "
                "~38s) just navigate around the stopped SUV."
            ),
        },
        {
            "pattern_id": "rear_end_stationary",
            "pattern_name": "Rear-End Collision with Stationary Vehicle",
            "description": (
                "A moving vehicle collides with the rear of a stationary vehicle. "
                "Classic failure-to-observe."
            ),
            "visual_signatures": [
                "Clear closing speed between moving and stationary vehicle",
                "Impact as sudden deceleration at point of contact",
                "Both vehicles in same lane before impact",
                "Post-impact: both stationary, striking vehicle directly behind",
            ],
            "causal_chain": (
                "Lead vehicle stationary -> following vehicle fails to observe -> "
                "closing speed -> rear-end impact"
            ),
            "common_misidentifications": [
                "WRONG: Vehicle passing a stopped car confused with rear-end impact",
                "WRONG: Vehicle slowing near stopped car assumed to have hit it",
            ],
            "vehicle_types": ["car", "truck", "van"],
            "road_types": ["roundabout", "motorway", "urban_road"],
            "keywords": ["rear-end", "stationary", "following distance", "braking"],
        },
        {
            "pattern_id": "hit_and_run_scrape",
            "pattern_name": "Hit-and-Run / Scrape-and-Leave",
            "description": (
                "A vehicle makes contact and continues driving without stopping. "
                "Common with large vehicles."
            ),
            "visual_signatures": [
                "One vehicle stops after contact; the other continues",
                "Departing vehicle shows no behaviour change",
                "Victim vehicle stops in unusual position mid-lane",
            ],
            "causal_chain": (
                "Contact -> victim stops -> striking vehicle continues -> "
                "scene appears as if victim stopped for no reason"
            ),
            "common_misidentifications": [
                "WRONG: Victim described as inexplicably stopped because striking vehicle left frame",
                "WRONG: Later-arriving vehicle blamed because actual culprit is gone",
            ],
            "vehicle_types": ["truck", "car", "van", "bus"],
            "road_types": ["roundabout", "motorway", "urban_road", "intersection"],
            "keywords": ["hit and run", "scrape and leave", "departed scene"],
        },
        {
            "pattern_id": "roundabout_failure_yield",
            "pattern_name": "Failure to Yield at Roundabout Entry",
            "description": "Vehicle enters roundabout without giving way to circulating traffic.",
            "visual_signatures": [
                "Vehicle enters without slowing while another circulates",
                "Near-miss or impact at entry point",
                "Circulating vehicle brakes hard or swerves",
            ],
            "causal_chain": (
                "Vehicle A circulating -> Vehicle B enters without yielding -> "
                "paths converge -> impact or avoidance"
            ),
            "common_misidentifications": [
                "WRONG: Normal give-way waiting confused with near-miss",
            ],
            "vehicle_types": ["car", "van", "truck"],
            "road_types": ["roundabout"],
            "keywords": ["give way", "yield", "roundabout entry", "side impact"],
        },
        {
            "pattern_id": "lane_change_sideswipe",
            "pattern_name": "Lane Change Sideswipe",
            "description": (
                "Vehicle changes lane without checking blind spot, contacting "
                "adjacent vehicle."
            ),
            "visual_signatures": [
                "Vehicle drifts laterally into occupied lane",
                "Side-to-side contact, same direction of travel",
                "No significant speed differential",
            ],
            "causal_chain": (
                "Vehicle A in lane 1 -> lane change into lane 2 -> "
                "Vehicle B in lane 2 -> lateral contact"
            ),
            "common_misidentifications": [
                "WRONG: Misclassified as head-on when both moving same direction",
            ],
            "vehicle_types": ["car", "van", "truck"],
            "road_types": ["motorway", "urban_road"],
            "keywords": ["lane change", "blind spot", "sideswipe", "merge"],
        },
    ],
    "regulations": [
        {
            "reg_id": "uk_roundabout_priority",
            "jurisdiction": "UK",
            "category": "roundabout",
            "rule_text": (
                "Give way to traffic from the right when entering a roundabout "
                "(Highway Code Rule 185)."
            ),
            "analysis_implication": (
                "Vehicles on the roundabout have priority. Entering vehicles must yield."
            ),
        },
        {
            "reg_id": "uk_lane_discipline",
            "jurisdiction": "UK",
            "category": "roundabout",
            "rule_text": (
                "Use left lane for left/straight; right lane for right/full circle "
                "(Highway Code Rule 186)."
            ),
            "analysis_implication": "Lane position at entry indicates intended exit.",
        },
        {
            "reg_id": "uk_hgv_swept_path",
            "jurisdiction": "UK",
            "category": "general",
            "rule_text": (
                "Large vehicles may need more than one lane when turning "
                "(Highway Code Rule 221)."
            ),
            "analysis_implication": (
                "HGVs encroaching into adjacent lanes is a known sideswipe risk."
            ),
        },
        {
            "reg_id": "uk_hit_and_run",
            "jurisdiction": "UK",
            "category": "general",
            "rule_text": (
                "Failing to stop and report an accident is an offence under "
                "Road Traffic Act 1988 s.170."
            ),
            "analysis_implication": (
                "If a vehicle departs after contact, flag as potential offence."
            ),
        },
    ],
    # Explicit causal edges between patterns
    "causal_edges": [
        {
            "source": "roundabout_sideswipe_hgv",
            "target": "hit_and_run_scrape",
            "relation": "OFTEN_FOLLOWED_BY",
        },
    ],
    # Regulation → road-geometry applicability edges
    "regulation_edges": [
        {"reg_id": "uk_roundabout_priority", "road_type": "roundabout"},
        {"reg_id": "uk_lane_discipline", "road_type": "roundabout"},
        {"reg_id": "uk_hgv_swept_path", "road_type": "roundabout"},
    ],
}


class CollisionKnowledgeGraph:
    """In-process property graph encoding collision domain expertise.

    Node types:
        ``vehicle_type``, ``road_geometry``, ``collision_pattern``, ``regulation``

    Edge relations:
        ``INVOLVES``, ``OCCURS_ON``, ``APPLIES_TO``, ``OFTEN_FOLLOWED_BY``

    Parameters:
        graph: An optional pre-built ``nx.DiGraph``. If *None*, an empty
            graph is created — populate it via ``from_dict`` or ``from_graphml``.
    """

    def __init__(self, graph: nx.DiGraph | None = None) -> None:
        self._graph: nx.DiGraph = graph or nx.DiGraph()

    # ── Properties ────────────────────────────────────────────────────

    @property
    def graph(self) -> nx.DiGraph:
        """The underlying NetworkX directed graph."""
        return self._graph

    @property
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    # ── Factory methods ───────────────────────────────────────────────

    @classmethod
    def from_defaults(cls) -> CollisionKnowledgeGraph:
        """Build the graph from the embedded default seed data."""
        return cls.from_dict(_DEFAULT_KG_DATA)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CollisionKnowledgeGraph:
        """Build the graph from a dictionary matching the PoC JSON schema.

        Parameters:
            data: Dictionary with keys ``vehicle_types``, ``road_types``,
                ``patterns``, ``regulations``, and optionally ``causal_edges``
                and ``regulation_edges``.

        Returns:
            A fully populated ``CollisionKnowledgeGraph``.
        """
        graph_construct = nx.DiGraph()

        # Vehicle type nodes
        for vt in data.get("vehicle_types", []):
            graph_construct.add_node(
                f"vtype:{vt['id']}",
                node_type="vehicle_type",
                **vt,
            )

        # Road geometry nodes
        for rt in data.get("road_types", []):
            graph_construct.add_node(
                f"road:{rt['id']}",
                node_type="road_geometry",
                **rt,
            )

        # Collision pattern nodes + INVOLVES / OCCURS_ON edges
        for p in data.get("patterns", []):
            pid = p["pattern_id"]
            graph_construct.add_node(
                f"pattern:{pid}", node_type="collision_pattern", **p
            )
            for vt in p.get("vehicle_types", []):
                graph_construct.add_edge(
                    f"pattern:{pid}", f"vtype:{vt}", relation="INVOLVES"
                )
            for rt in p.get("road_types", []):
                graph_construct.add_edge(
                    f"pattern:{pid}", f"road:{rt}", relation="OCCURS_ON"
                )

        # Inter-pattern causal edges
        for ce in data.get("causal_edges", []):
            graph_construct.add_edge(
                f"pattern:{ce['source']}",
                f"pattern:{ce['target']}",
                relation=ce.get("relation", "OFTEN_FOLLOWED_BY"),
            )

        # Regulation nodes
        for r in data.get("regulations", []):
            graph_construct.add_node(f"reg:{r['reg_id']}", node_type="regulation", **r)

        # Regulation → road applicability edges
        for re in data.get("regulation_edges", []):
            graph_construct.add_edge(
                f"reg:{re['reg_id']}",
                f"road:{re['road_type']}",
                relation="APPLIES_TO",
            )

        instance = cls(graph=graph_construct)
        logger.info(
            "Knowledge graph built: %d nodes, %d edges.",
            instance.num_nodes,
            instance.num_edges,
        )
        return instance

    @classmethod
    def from_graphml(cls, path: str | Path) -> CollisionKnowledgeGraph:
        """Load a previously exported GraphML file.

        JSON-serialised list/dict attributes are automatically deserialised
        back into Python objects.

        Parameters:
            path: Filesystem path to the ``.graphml`` file.

        Returns:
            A ``CollisionKnowledgeGraph`` populated from the file.
        """
        graph_reader = nx.read_graphml(str(path))
        # Deserialise JSON-encoded attributes
        for _, attrs in graph_reader.nodes(data=True):
            for k, v in list(attrs.items()):
                if isinstance(v, str) and v.startswith("["):
                    try:
                        attrs[k] = json.loads(v)
                    except json.JSONDecodeError:
                        pass
                elif isinstance(v, str) and v in ("True", "False"):
                    attrs[k] = v == "True"
        logger.info("Knowledge graph loaded from %s.", path)
        return cls(graph=graph_reader)

    # ── Export ─────────────────────────────────────────────────────────

    def to_graphml(self, path: str | Path) -> None:
        """Export the graph to GraphML for Neo4j / Cosmos DB ingestion.

        List and dict attributes are JSON-serialised; booleans are
        converted to strings (GraphML only supports scalar types).

        Parameters:
            path: Destination file path.
        """
        g_exp = self._graph.copy()
        for _, attrs in g_exp.nodes(data=True):
            for k, v in list(attrs.items()):
                if isinstance(v, (list, dict)):
                    attrs[k] = json.dumps(v)
                elif isinstance(v, bool):
                    attrs[k] = str(v)
        nx.write_graphml(g_exp, str(path))
        logger.info("Knowledge graph exported to %s.", path)

    # ── Query helpers (used by KGRetriever) ───────────────────────────

    def pattern_nodes(self) -> list[tuple[str, dict[str, Any]]]:
        """Return all collision-pattern nodes as ``(node_id, attrs)`` pairs."""
        return [
            (n, d)
            for n, d in self._graph.nodes(data=True)
            if d.get("node_type") == "collision_pattern"
        ]

    def regulation_nodes(self) -> list[tuple[str, dict[str, Any]]]:
        """Return all regulation nodes as ``(node_id, attrs)`` pairs."""
        return [
            (n, d)
            for n, d in self._graph.nodes(data=True)
            if d.get("node_type") == "regulation"
        ]

    def successors_by_relation(self, node_id: str, relation: str) -> list[str]:
        """Return successor node IDs filtered by edge relation type."""
        return [
            n
            for n in self._graph.successors(node_id)
            if self._graph.edges[node_id, n].get("relation") == relation
        ]
