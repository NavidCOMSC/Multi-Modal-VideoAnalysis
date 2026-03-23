"""Module for executing video-based LLM analysis calls with prompt management.

Provides three orchestration functions:

- ``execute_video_analysis_stage`` — crash-only detection (original).
- ``execute_video_scene_analysis_stage`` — full scene description (non-KG).
- ``execute_video_scene_analysis_kg_stage`` — KG-augmented two-pass analysis
  with chain-of-thought reasoning and verification.
"""

from __future__ import annotations

import logging
from typing import List, Literal

from langfuse import Langfuse

from video_analysis_poc.connection.langfuse.prompts import collate_prompt
from video_analysis_poc.video_llm.knowledge_graph import (
    CollisionKnowledgeGraph,
)
from video_analysis_poc.video_llm.llm_wrapper import VideoLLM
from video_analysis_poc.video_llm.schema.video import (
    CameraSource,
    VideoAgentTemplate,
    VideoAnalysisOutput,
    VideoSceneAnalysisOutput,
)
from video_analysis_poc.video_llm.use_case_config import MetadataTemplate

logger = logging.getLogger(__name__)


async def execute_video_analysis_stage(
    metadata_instance: MetadataTemplate,
    uuid_tag: str,
    langfuse_client: Langfuse,
    video_agent: VideoAgentTemplate,
    camera_sources: List[CameraSource],
    user_query: str | None = None,
    **prompt_kwargs,
) -> VideoAnalysisOutput:
    """Execute a video crash-detection analysis with structured output.

    This function mirrors the calling convention of ``execute_llm_stage`` in
    :pymod:`llm_services.llm_caller` but is specialised for multimodal video
    input targeting Gemini 2.5 Pro.

    Args:
        metadata_instance: Use case metadata containing LLM configuration.
        uuid_tag: Unique identifier for the session/process.
        langfuse_client: Langfuse client for tracing.
        video_agent: Video agent configuration (model name, task template, params).
        camera_sources: List of CameraSource objects describing each video file.
        user_query: Optional user query/instruction to augment the prompt.
        **prompt_kwargs: Additional keyword arguments forwarded to the prompt
                         template.

    Returns:
        VideoAnalysisOutput containing detected crash events, timestamps,
        confidence scores, and an overall summary.
    """
    prompt_text, prompt_object = await collate_prompt(
        metadata_instance=metadata_instance,
        langfuse=langfuse_client,
        prompt_type="task",
        prompt_template_name=video_agent.task_template_name,
        uuid_tag=uuid_tag,
        user_query=user_query,
        **prompt_kwargs,
    )

    prompt_text_str: str = (
        prompt_text if isinstance(prompt_text, str) else str(prompt_text)
    )

    video_llm_client = VideoLLM(
        metadata_instance=metadata_instance,
        llm_agent=video_agent,
        langfuse=langfuse_client,
    )

    video_analysis_output = await video_llm_client.async_analyse_crash_footage(
        camera_sources=camera_sources,
        prompt_text=prompt_text_str,
        prompt_object=prompt_object,
        task_name=video_agent.task_template_name,
        temperature=video_agent.temperature,
        reasoning_effort=video_agent.reasoning_effort,
    )

    logger.info(
        "Video analysis complete for %s — %d crash event(s) detected.",
        uuid_tag,
        len(video_analysis_output.crash_events),
    )

    return video_analysis_output


async def execute_video_scene_analysis_stage(
    metadata_instance: MetadataTemplate,
    uuid_tag: str,
    langfuse_client: Langfuse,
    video_agent: VideoAgentTemplate,
    camera_sources: List[CameraSource],
    user_query: str | None = None,
    **prompt_kwargs,
) -> VideoSceneAnalysisOutput:
    """Execute full video scene analysis (non-KG path).

    Returns a comprehensive scene description including all detected objects,
    environmental context, and notable events with forensic crash details.

    Args:
        metadata_instance: Use case metadata containing LLM configuration.
        uuid_tag: Unique identifier for the session/process.
        langfuse_client: Langfuse client for tracing.
        video_agent: Video agent configuration.
        camera_sources: List of CameraSource objects.
        user_query: Optional user query/instruction.
        **prompt_kwargs: Additional prompt template arguments.

    Returns:
        VideoSceneAnalysisOutput with environment, objects, events, summary.
    """
    prompt_text, prompt_object = await collate_prompt(
        metadata_instance=metadata_instance,
        langfuse=langfuse_client,
        prompt_type="task",
        prompt_template_name=video_agent.task_template_name,
        uuid_tag=uuid_tag,
        user_query=user_query,
        **prompt_kwargs,
    )

    prompt_text_str: str = (
        prompt_text if isinstance(prompt_text, str) else str(prompt_text)
    )

    video_llm_client = VideoLLM(
        metadata_instance=metadata_instance,
        llm_agent=video_agent,
        langfuse=langfuse_client,
    )

    output = await video_llm_client.async_analyse_scene_footage(
        camera_sources=camera_sources,
        prompt_text=prompt_text_str,
        prompt_object=prompt_object,
        task_name=video_agent.task_template_name,
        temperature=video_agent.temperature,
        reasoning_effort=video_agent.reasoning_effort,
    )

    logger.info(
        "Scene analysis complete for %s — %d object(s), %d event(s).",
        uuid_tag,
        len(output.detected_objects),
        len(output.events),
    )

    return output


async def execute_video_scene_analysis_kg_stage(
    metadata_instance: MetadataTemplate,
    uuid_tag: str,
    langfuse_client: Langfuse,
    video_agent: VideoAgentTemplate,
    camera_sources: List[CameraSource],
    detected_vehicle_types: list[str],
    detected_road_type: str | None = None,
    keyword_hints: list[str] | None = None,
    key_frame_timestamps: list[float] | None = None,
    knowledge_graph: CollisionKnowledgeGraph | None = None,
    run_verification_pass: bool = True,
    jurisdiction: str = "UK",
    user_query: str | None = None,
    **prompt_kwargs,
) -> VideoSceneAnalysisOutput:
    """Execute KG-augmented two-pass video scene analysis.

    This is the most capable analysis mode, orchestrating:

    1. Camera auto-detection via optical flow
    2. Knowledge graph pattern retrieval (vehicle types, road geometry, keywords)
    3. Key frame extraction at critical timestamps
    4. Composable prompt assembly (camera context + RAG patterns + CoT)
    5. Pass 1 — KG-augmented analysis with reasoning trace
    6. Pass 2 — KG-grounded verification against common misidentifications

    If no ``knowledge_graph`` is provided, the default embedded KG is used
    (5 collision patterns, 4 UK regulations, typed vehicle/road nodes).

    Args:
        metadata_instance: Use case metadata containing LLM configuration.
        uuid_tag: Unique identifier for the session/process.
        langfuse_client: Langfuse client for tracing.
        video_agent: Video agent configuration.
        camera_sources: List of CameraSource objects.
        detected_vehicle_types: Vehicle types observed in the scene
            (e.g. ``["truck", "car", "van"]``).
        detected_road_type: Road geometry type (e.g. ``"roundabout"``).
        keyword_hints: Optional keyword cues for KG retrieval.
        key_frame_timestamps: Timestamps (seconds) for supplementary frame
            extraction.  If None, no key frames are added.
        knowledge_graph: A populated CollisionKnowledgeGraph.  If None,
            the default embedded graph is built.
        run_verification_pass: Whether to run Pass 2 verification (default True).
        jurisdiction: Jurisdiction for regulation retrieval (default ``"UK"``).
        user_query: Optional user query/instruction.
        **prompt_kwargs: Additional prompt template arguments.

    Returns:
        VideoSceneAnalysisOutput with reasoning trace, environment, objects,
        events (with forensic crash details), and summary.
    """
    if knowledge_graph is None:
        knowledge_graph = CollisionKnowledgeGraph.from_defaults()

    video_llm_client = VideoLLM(
        metadata_instance=metadata_instance,
        llm_agent=video_agent,
        langfuse=langfuse_client,
    )

    output = await video_llm_client.async_analyse_scene_footage_kg(
        camera_sources=camera_sources,
        knowledge_graph=knowledge_graph,
        detected_vehicle_types=detected_vehicle_types,
        detected_road_type=detected_road_type,
        keyword_hints=keyword_hints,
        key_frame_timestamps=key_frame_timestamps,
        task_name=video_agent.task_template_name,
        temperature=video_agent.temperature,
        reasoning_effort=video_agent.reasoning_effort,
        run_verification_pass=run_verification_pass,
        jurisdiction=jurisdiction,
    )

    n_crash = sum(
        1 for e in output.events if e.event_type in ("collision", "near_miss")
    )
    logger.info(
        "KG-augmented analysis complete for %s — %d object(s), %d event(s) "
        "(%d crash/near-miss), verification=%s.",
        uuid_tag,
        len(output.detected_objects),
        len(output.events),
        n_crash,
        "yes" if run_verification_pass else "no",
    )

    return output
