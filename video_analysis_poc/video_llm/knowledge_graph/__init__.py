"""Knowledge graph module for collision pattern retrieval and RAG augmentation.

This module provides:
- ``CollisionKnowledgeGraph``: A NetworkX-backed property graph encoding
  collision archetypes, vehicle types, road geometries, and traffic regulations.
- ``KGRetriever``: Scored graph-traversal retrieval that ranks collision
  patterns by detected vehicle types, road geometry, and keyword hints.
- ``format_rag_context``: Prompt-ready serialisation of retrieved patterns
  and applicable regulations.

Production deployment note:
    The in-process NetworkX graph is suitable for PoC and low-traffic services.
    For production at scale, swap the backend for Neo4j (Bolt driver) or
    Azure Cosmos DB Gremlin — the ``CollisionKnowledgeGraph`` interface is
    designed to be replaceable via dependency injection.
"""

from video_analysis_poc.video_llm.knowledge_graph.graph import (
    CollisionKnowledgeGraph,
)
from video_analysis_poc.video_llm.knowledge_graph.retriever import KGRetriever
from video_analysis_poc.video_llm.knowledge_graph.formatter import (
    format_rag_context,
)

__all__ = [
    "CollisionKnowledgeGraph",
    "KGRetriever",
    "format_rag_context",
]
