"""
Prompt-ready formatting of KG retrieval results.

``format_rag_context`` converts scored collision patterns and traffic
regulations into a structured text block suitable for injection into an
LLM prompt.  The output includes:

- Pattern name, description, visual signatures, causal chain
- **ERRORS TO AVOID** — directly from ``common_misidentifications``
- Example narratives (when available)
- Applicable regulations with their analysis implications
"""

from __future__ import annotations

from video_analysis_poc.video_llm.knowledge_graph.models import (
    ScoredPattern,
    TrafficRegulation,
)


def format_rag_context(
    scored_patterns: list[ScoredPattern],
    regulations: list[TrafficRegulation],
    max_patterns: int = 3,
) -> str:
    """Format KG retrieval results as a prompt-injectable text block.

    Parameters:
        scored_patterns: Ranked collision patterns from ``KGRetriever``.
        regulations: Applicable traffic regulations.
        max_patterns: Maximum number of patterns to include (default 3).

    Returns:
        A multi-line string ready for insertion into the LLM task prompt.
    """
    lines: list[str] = ["DOMAIN EXPERTISE FROM KNOWLEDGE BASE:", ""]

    for sp in scored_patterns[:max_patterns]:
        p = sp.pattern
        lines.append(f"### Pattern: {p.pattern_name}")
        lines.append(f"Description: {p.description}")

        if p.visual_signatures:
            sigs = "; ".join(p.visual_signatures)
            lines.append(f"Visual signatures: {sigs}")

        if p.causal_chain:
            lines.append(f"Causal chain: {p.causal_chain}")

        if p.common_misidentifications:
            misids = "; ".join(p.common_misidentifications)
            lines.append(f"ERRORS TO AVOID: {misids}")

        if p.example_narrative:
            lines.append(f"Example: {p.example_narrative}")

        lines.append("")

    if regulations:
        lines.append("APPLICABLE REGULATIONS:")
        for r in regulations:
            lines.append(f"- {r.rule_text}")
            lines.append(f"  Implication: {r.analysis_implication}")

    return "\n".join(lines)


def format_verification_misid_check(
    top_pattern: ScoredPattern | None,
) -> str:
    """Build a verification check block from the top-ranked pattern's
    common misidentification list.

    Used in Pass 2 (verification) to cross-check the initial analysis
    against the most likely error classes for the scene.

    Parameters:
        top_pattern: The highest-scoring pattern, or *None*.

    Returns:
        A text block for the verification prompt, or an empty string
        if no pattern is provided.
    """
    if top_pattern is None:
        return ""

    p = top_pattern.pattern
    if not p.common_misidentifications:
        return ""

    misids = "; ".join(p.common_misidentifications)
    return (
        f"\nKB CHECK: Top pattern is '{p.pattern_name}'. "
        f"Common errors: {misids}\n"
        f"Verify you did NOT make these errors.\n"
    )
