"""Pydantic models for video analysis agents.

This module defines the complete schema hierarchy for both crash-only
and full scene analysis pipelines, including:

- ``CameraSource`` / ``VideoAgentTemplate`` — input configuration
- ``CrashEvent`` / ``VideoAnalysisOutput`` — crash-only pipeline output
- ``VideoObjectObservation`` / ``SceneEvent`` / ``VideoSceneAnalysisOutput``
  — full scene analysis output with forensic crash details
- ``ReasoningTrace`` — chain-of-thought trace for KG-augmented analysis
"""

from __future__ import annotations

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


class VideoObjectObservation(BaseModel):
    """A single object or road user detected in the video."""

    object_type: str = Field(
        ...,
        description=(
            "Category of the detected object, e.g. 'car', 'truck', 'bus', "
            "'motorcycle', 'bicycle', 'pedestrian', 'traffic_sign', "
            "'traffic_light', 'lane_marking', 'road_barrier', 'building', "
            "'tree', 'animal', 'debris', 'other'."
        ),
    )
    description: str = Field(
        ...,
        description="Brief natural-language description of the object, including colour, size, behaviour, or state.",
    )
    first_seen_seconds: float | None = Field(
        None,
        description="Approximate timestamp (seconds from video start) when the object first appears.",
    )
    last_seen_seconds: float | None = Field(
        None,
        description="Approximate timestamp (seconds from video start) when the object leaves the frame.",
    )
    position_in_frame: str | None = Field(
        None,
        description="Spatial location hint, e.g. 'centre', 'left lane', 'right shoulder', 'far background'.",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence for this detection."
    )


class CrashInvolvedParty(BaseModel):
    """A single road user involved in a witnessed collision or near-miss."""

    road_user_type: str = Field(
        ...,
        description="Type of road user, e.g. 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian', 'other'.",
    )
    description: str = Field(
        ...,
        description="Brief description: colour, make/model if visible, lane position, direction of travel.",
    )
    action_before_event: str | None = Field(
        None,
        description="What this party was doing immediately before the event, e.g. 'turning left', 'stationary', 'accelerating'.",
    )
    action_after_event: str | None = Field(
        None,
        description="What this party did after the event, e.g. 'stopped', 'continued driving', 'pulled over'.",
    )
    apparent_damage: str | None = Field(
        None,
        description="Visible damage if any, e.g. 'front bumper dented', 'windshield cracked', 'none visible'.",
    )


class WitnessedCrashDetail(BaseModel):
    """Rich detail for a collision or near-miss witnessed by the dashcam/CCTV.

    This is populated ONLY when event_type is 'collision' or 'near_miss'.
    """

    impact_type: str | None = Field(
        None,
        description=(
            "Nature of the impact: 'rear_end', 'head_on', 'side_swipe', "
            "'t_bone', 'rollover', 'run_off_road', 'pedestrian_strike', "
            "'cyclist_strike', 'multi_vehicle_pileup', 'none_near_miss', 'other'."
        ),
    )
    involved_parties: list[CrashInvolvedParty] = Field(
        default_factory=list,
        description="Detailed description of each road user involved.",
    )
    probable_cause: str | None = Field(
        None,
        description="Best assessment of the primary cause, e.g. 'failure to yield', 'distracted driving', 'running red light'.",
    )
    fault_indicators: list[str] = Field(
        default_factory=list,
        description="Observable cues suggesting fault, e.g. 'ran red light', 'crossed centre line', 'excessive speed'.",
    )
    sequence_of_events: str | None = Field(
        None,
        description="Step-by-step narrative of how the crash unfolded.",
    )
    post_crash_actions: str | None = Field(
        None,
        description="What happened after the crash: vehicles stopping, emergency response, debris on road, etc.",
    )
    estimated_speed_range: str | None = Field(
        None,
        description="Rough speed estimate if inferable, e.g. 'low (< 20 mph)', 'moderate (20-40 mph)', 'high (> 40 mph)'.",
    )
    dashcam_vehicle_involved: bool = Field(
        default=False,
        description="True if the dashcam's own vehicle was involved in the crash; False if it was a bystander witness.",
    )


class SceneEvent(BaseModel):
    """A notable event observed in the video — not limited to crashes."""

    event_type: str = Field(
        ...,
        description=(
            "Category of event, e.g. 'collision', 'near_miss', 'hard_braking', "
            "'lane_change', 'pedestrian_crossing', 'traffic_light_change', "
            "'vehicle_overtake', 'vehicle_turning', 'emergency_vehicle', "
            "'road_obstruction', 'other'."
        ),
    )
    timestamp_seconds: float = Field(
        ..., description="Time offset (seconds) from start of video."
    )
    timestamp_display: str = Field(
        ..., description="Human-readable timestamp, e.g. '00:01:34'."
    )
    description: str = Field(..., description="Concise description of what happened.")
    involved_objects: list[str] = Field(
        default_factory=list,
        description="List of object types involved in this event.",
    )
    severity: str | None = Field(
        None,
        description="Severity or significance: 'low', 'medium', 'high', or None if not applicable.",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence for this event."
    )
    witnessed_crash_detail: WitnessedCrashDetail | None = Field(
        None,
        description=(
            "Populated ONLY when event_type is 'collision' or 'near_miss'. "
            "Contains rich crash-specific metadata: impact type, involved parties, "
            "probable cause, fault indicators, and post-crash actions."
        ),
    )


class EnvironmentContext(BaseModel):
    """High-level environmental and road context."""

    weather: str | None = Field(
        None,
        description="Observed weather: 'clear', 'rain', 'fog', 'snow', 'overcast', etc.",
    )
    lighting: str | None = Field(
        None,
        description="Lighting conditions: 'daylight', 'dusk', 'night', 'artificial', etc.",
    )
    road_type: str | None = Field(
        None,
        description="Road type: 'motorway', 'urban', 'suburban', 'rural', 'roundabout', etc.",
    )
    road_surface: str | None = Field(
        None, description="Surface condition: 'dry', 'wet', 'icy', 'gravel', etc."
    )
    traffic_density: str | None = Field(
        None,
        description="Traffic level: 'empty', 'light', 'moderate', 'heavy', 'congested'.",
    )
    additional_notes: str | None = Field(
        None, description="Any other environmental observations."
    )


class ReasoningTrace(BaseModel):
    """Structured chain-of-thought reasoning trace.

    Populated by the LLM *before* the main scene-analysis output when
    using the KG-augmented pipeline.  Forces a perception-to-cognition
    chain that makes causal reasoning explicit.
    """

    global_perception: str = Field(..., description="Scene-level observations.")
    vehicle_inventory: str = Field(
        ..., description="All vehicles with IDs and descriptions."
    )
    interaction_analysis: str = Field(..., description="Vehicle interaction analysis.")
    causal_conclusion: str = Field(
        ..., description="Final causal attribution with temporal evidence."
    )


class VideoSceneAnalysisOutput(BaseModel):
    """Structured output from full video scene analysis.

    When using the KG-augmented pipeline, the ``reasoning_trace`` field
    is populated with the model's explicit chain-of-thought reasoning.
    """

    task_name: str | None = Field(
        None, description="Logical label for this analysis run."
    )
    camera_type: str | None = Field(None, description="Camera source type.")
    video_file_name: str | None = Field(None, description="Source video file name.")
    environment: EnvironmentContext | None = Field(
        None, description="Environmental and road context."
    )
    detected_objects: list[VideoObjectObservation] = Field(
        default_factory=list, description="All objects detected in the footage."
    )
    events: list[SceneEvent] = Field(
        default_factory=list, description="Notable events observed in the footage."
    )
    summary: str | None = Field(
        None, description="Overall narrative summary of the footage."
    )
    reasoning_trace: ReasoningTrace | None = Field(
        None,
        description=(
            "Chain-of-thought reasoning trace from the KG-augmented pipeline. "
            "None when using the basic (non-KG) analysis path."
        ),
    )
    raw_output: dict | None = Field(
        None, description="Raw parsed JSON output from the model."
    )
