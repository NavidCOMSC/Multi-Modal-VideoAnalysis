"""Pydantic models for video analysis agents."""

from pydantic import BaseModel, Field
from typing import Literal


class CameraSource(BaseModel):
    """Describes a single camera source in the video data."""

    camera_type: Literal["dashcam_front", "dashcam_rear", "cctv"] = Field(
        ..., description="Type of camera that recorded the footage."
    )
    video_file_name: str = Field(
        ..., description="Name or path of the source video file."
    )
    duration_seconds: float | None = Field(
        None, description="Duration of the video in seconds."
    )


class CrashEvent(BaseModel):
    """A single detected crash event within a video."""

    camera_type: str = Field(..., description="Camera source that captured the event.")
    video_file_name: str = Field(
        ..., description="Video file name in which the crash was detected."
    )
    crash_timestamp_seconds: float = Field(
        ...,
        description="Time offset (in seconds) from start of video when crash occurred.",
    )
    crash_timestamp_display: str = Field(
        ..., description="Human-readable timestamp, e.g. '00:01:34'."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence score for the detection."
    )
    description: str | None = Field(
        None,
        description="Brief description of the crash event as observed by the model.",
    )


class VideoAnalysisOutput(BaseModel):
    """Structured output from video crash analysis."""

    task_name: str | None = Field(
        None, description="Logical label for this analysis run."
    )
    crash_events: list[CrashEvent] = Field(
        default_factory=list, description="List of detected crash events."
    )
    summary: str | None = Field(
        None, description="Overall summary of the video analysis."
    )
    raw_output: dict | None = Field(
        None, description="Raw parsed JSON output from the model."
    )


class VideoAgentTemplate(BaseModel):
    """Configuration template for a video analysis LLM agent."""

    model_name: str = Field(
        default="gemini/gemini-2.5-pro",
        description="LiteLLM model identifier for Gemini 2.5 Pro.",
    )
    task_template_name: str = Field(
        ..., description="Langfuse prompt template name for the video analysis task."
    )
    temperature: float | None = Field(default=0.0, description="Sampling temperature.")
    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "default"] | None
    ) = Field(default=None, description="Reasoning effort level.")
    api_version: str | None = Field(default=None, description="API version override.")
    max_video_size_mb: float = Field(
        default=100.0, description="Maximum allowed video file size in MB."
    )
    supported_formats: list[str] = Field(
        default_factory=lambda: ["mp4", "avi", "mov", "mkv"],
        description="Accepted video file formats.",
    )
