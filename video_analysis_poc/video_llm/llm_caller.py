"""Module for executing video-based LLM analysis calls with prompt management."""

import logging
from typing import List
from langfuse import Langfuse
from video_analysis_poc.connection.langfuse.prompts import collate_prompt
from video_analysis_poc.video_llm.llm_wrapper import VideoLLM
from video_analysis_poc.video_llm.use_case_config import MetadataTemplate
from video_analysis_poc.video_llm.schema.video import (
    VideoAgentTemplate,
    VideoAnalysisOutput,
    CameraSource,
)

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
        camera_sources: List of CameraSource objects describing each video file
                        (dashcam front, dashcam rear, CCTV, etc.).
        user_query: Optional user query/instruction to augment the prompt.
        **prompt_kwargs: Additional keyword arguments forwarded to the prompt
                         template (e.g., ``incident_ref="INC-12345"``).

    Returns:
        VideoAnalysisOutput containing detected crash events, timestamps,
        confidence scores, and an overall summary.
    """

    # Collate the prompt from the Langfuse template
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

    # Initialise the Video LLM client
    video_llm_client = VideoLLM(
        metadata_instance=metadata_instance,
        llm_agent=video_agent,
        langfuse=langfuse_client,
    )

    # Execute the video analysis
    video_analysis_output = await video_llm_client.analyse_crash_footage(
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
