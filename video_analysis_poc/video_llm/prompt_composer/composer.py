"""Composable prompt builder for KG-augmented video scene analysis.

``PromptComposer`` assembles system prompts, task prompts, and
verification prompts from:
- Camera profile context
- Retrieved KG patterns (with ERRORS TO AVOID)
- Applicable regulations
- Chain-of-thought reasoning structure
- Structured JSON output schema

This replaces the hardcoded prompt strings in the PoC notebooks with a
programmatic, testable composition layer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from video_analysis_poc.video_llm.cv_preprocessing.camera_detection import (
    CameraProfile,
)
from video_analysis_poc.video_llm.knowledge_graph.formatter import (
    format_rag_context,
    format_verification_misid_check,
)
from video_analysis_poc.video_llm.knowledge_graph.models import (
    ScoredPattern,
    TrafficRegulation,
)

logger = logging.getLogger(__name__)


# ── Prompt template fragments ─────────────────────────────────────────

_CAUSAL_PRINCIPLES = """\
ANALYSIS PRINCIPLES:
- Establish CAUSAL relationships between events. If a vehicle is stopped \
in an unusual location, determine what caused it to stop by examining the \
preceding seconds — do NOT assume it was inexplicably stationary.
- Large vehicles (trucks, HGVs) have wide turning arcs that frequently \
cause sideswipe collisions with adjacent vehicles on roundabouts and \
tight curves. Look carefully for this pattern whenever a truck and a \
smaller vehicle enter a roundabout side by side.
- The striking vehicle may leave the scene after a collision (hit-and-run \
or scrape-and-leave). Do not assume the first vehicle you see near a \
stopped car is the one that struck it — trace the timeline backwards.
- Do NOT confuse vehicles that arrive AFTER a collision with the vehicle \
that CAUSED it. Temporal ordering is critical: establish which \
interaction happened first.
- Still frames are provided at key timestamps alongside the video. Use \
them to verify spatial relationships between vehicles at critical moments."""

_COT_STRUCTURE = """\
REASONING CHAIN (populate reasoning_trace FIRST, THEN structured output):
1. GLOBAL PERCEPTION: Overall scene description
2. VEHICLE INVENTORY: Every vehicle with an assigned ID
3. INTERACTION ANALYSIS: Trajectory relationships and proximity
4. CAUSAL CONCLUSION: Event classification with temporal evidence"""

_TASK_INSTRUCTIONS = """\
Instructions:
- FIRST populate reasoning_trace, THEN the structured output sections.
- Check your conclusions against the ERRORS TO AVOID from the knowledge base.
- For crash/near-miss events, ALWAYS populate witnessed_crash_detail.
- For non-crash events, set witnessed_crash_detail to null.
- If the striking vehicle left the scene, note this explicitly.
- List EVERY distinct object you can identify, not just vehicles.
- Include static infrastructure: signs, traffic lights, barriers, road markings.
- Provide timestamps relative to the start of the video.
- Be conservative with confidence scores.
- Return ONLY the JSON object, no markdown fences or commentary."""

_OUTPUT_SCHEMA = """\
{
  "reasoning_trace": {
    "global_perception": "<scene observations>",
    "vehicle_inventory": "<all vehicles with IDs>",
    "interaction_analysis": "<trajectory interactions, proximity>",
    "causal_conclusion": "<causal attribution with temporal evidence>"
  },
  "environment": {"weather": "...", "lighting": "...", "road_type": "...",
                  "road_surface": "...", "traffic_density": "...", "additional_notes": "..."},
  "detected_objects": [{"object_type": "...", "description": "...",
    "first_seen_seconds": 0.0, "last_seen_seconds": 0.0,
    "position_in_frame": "...", "confidence": 0.0}],
  "events": [{"event_type": "...", "timestamp_seconds": 0.0,
    "timestamp_display": "HH:MM:SS", "description": "...",
    "involved_objects": ["..."], "severity": "...", "confidence": 0.0,
    "witnessed_crash_detail": {
      "impact_type": "...",
      "involved_parties": [{"road_user_type": "...", "description": "...",
        "action_before_event": "...", "action_after_event": "...",
        "apparent_damage": "..."}],
      "probable_cause": "...", "fault_indicators": ["..."],
      "sequence_of_events": "...", "post_crash_actions": "...",
      "estimated_speed_range": "...", "dashcam_vehicle_involved": false}}],
  "summary": "<narrative>"
}"""

_VERIFICATION_TEMPLATE = """\
You previously produced:
{pass1_json}
{misid_check}
Verify:
1) Causality of stopped vehicles — trace backwards to find what caused them to stop.
2) Temporal ordering — which vehicle interactions happened first?
3) Truck swept-path — examine frames at the roundabout entry for encroachment.
4) Later arrivals — did vehicles arriving after the incident actually make contact, \
or did they merely navigate around the already-stopped vehicle?

Provide a CORRECTED JSON output using the exact same schema. If your original \
analysis was correct, return it unchanged. Return ONLY JSON."""


class PromptComposer:
    """Assembles complete prompts for KG-augmented video scene analysis.

    Parameters:
        camera_profile: Detected camera motion profile.
        scored_patterns: Ranked collision patterns from KG retrieval.
        regulations: Applicable traffic regulations.
    """

    def __init__(
        self,
        camera_profile: CameraProfile | None = None,
        scored_patterns: list[ScoredPattern] | None = None,
        regulations: list[TrafficRegulation] | None = None,
    ) -> None:
        self._camera_profile = camera_profile
        self._scored_patterns = scored_patterns or []
        self._regulations = regulations or []

    def build_system_prompt(self) -> str:
        """Compose the system prompt with camera context and causal principles.

        Returns:
            The complete system prompt string.
        """
        parts: list[str] = [
            "You are an expert multimodal video analyst for road scene "
            "understanding. Provide comprehensive structured descriptions "
            "of dashcam/CCTV footage.",
            "",
        ]

        if self._camera_profile:
            parts.append(f"CAMERA: {self._camera_profile.prompt_context}")
            parts.append("")

        parts.append(_CAUSAL_PRINCIPLES)
        parts.append("")
        parts.append(_COT_STRUCTURE)

        return "\n".join(parts)

    def build_task_prompt(self, user_query: str | None = None) -> str:
        """Compose the task prompt with RAG context, schema, and instructions.

        Parameters:
            user_query: Optional incident-specific instruction from the caller
                (e.g. ``"Focus on the interaction between the truck and the
                black SUV at 15-20s"``).  Appended after the schema and
                instructions so the model treats it as additional guidance.

        Returns:
            The complete task prompt string.
        """
        parts: list[str] = []

        # RAG context from knowledge graph
        if self._scored_patterns or self._regulations:
            rag_text = format_rag_context(
                self._scored_patterns, self._regulations
            )
            parts.append(rag_text)
            parts.append("\n---\n")

        parts.append(
            "Analyse the video and still frames using the domain patterns above."
        )
        parts.append(
            "Follow the perception-to-cognition chain. Return JSON:"
        )
        parts.append("")
        parts.append(_OUTPUT_SCHEMA)
        parts.append("")
        parts.append(_TASK_INSTRUCTIONS)

        if user_query:
            parts.append("")
            parts.append(f"ADDITIONAL CONTEXT FROM THE USER:\n{user_query}")

        return "\n".join(parts)

    def build_verification_prompt(self, pass1_output: dict) -> str:
        """Compose the Pass 2 verification prompt.

        Parameters:
            pass1_output: The raw JSON dict from the Pass 1 LLM response.

        Returns:
            The complete verification prompt string.
        """
        misid_check = format_verification_misid_check(
            self._scored_patterns[0] if self._scored_patterns else None
        )

        return _VERIFICATION_TEMPLATE.format(
            pass1_json=json.dumps(pass1_output, indent=2),
            misid_check=misid_check,
        )

    def build_keyframe_annotation(self, timestamp_seconds: float) -> str:
        """Build the text annotation for a supplementary key frame.

        Parameters:
            timestamp_seconds: The video timestamp of the frame.

        Returns:
            Annotation text for the content part.
        """
        return (
            f"[STILL FRAME at {timestamp_seconds:.0f}s] — Examine this frame "
            f"carefully. Check vehicle positions and spatial relationships."
        )
