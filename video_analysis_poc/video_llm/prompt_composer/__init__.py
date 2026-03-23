"""Composable prompt architecture for video analysis.

This module assembles the complete LLM prompt from modular components:
- Camera context (from ``cv_preprocessing``)
- Retrieved collision patterns and regulations (from ``knowledge_graph``)
- Chain-of-thought reasoning structure
- Output JSON schema
- Verification pass instructions

The ``PromptComposer`` class provides a clean interface for building
both Pass 1 (KG-augmented analysis) and Pass 2 (verification) prompts.
"""

from video_analysis_poc.video_llm.prompt_composer.composer import (
    PromptComposer,
)

__all__ = ["PromptComposer"]
