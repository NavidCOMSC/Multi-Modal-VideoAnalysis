"""Module for video-based LLM inference using Gemini 2.5 Pro via LiteLLM.

Supports both the original crash-only analysis pipeline and the extended
scene-description pipeline that captures all detected objects, environmental
context, and rich event metadata (including forensic-level crash details).
"""

import litellm
from litellm.files.main import ModelResponse
import json
import base64
import logging
import os
from pathlib import Path
from langfuse import Langfuse
from langfuse.model import TextPromptClient
from langfuse_schema import ObservationPayload
from use_case_config.metadata import MetadataTemplate
from schema.video import (
    # Original crash-only models (backward compatibility)
    VideoAgentTemplate,
    VideoAnalysisOutput,
    CrashEvent,
    CameraSource,
    # Extended scene-description models
    DetectedObject,
    CrashInvolvedParty,
    WitnessedCrashDetail,
    SceneEvent,
    EnvironmentContext,
    VideoSceneAnalysisOutput,
)

from video_analysis_poc.connection.langfuse.tracing_connection import (
    langfuse_observation,
)
from typing import Literal


logger = logging.getLogger(__name__)

# Mapping of file extensions to MIME types for video content
_VIDEO_MIME_TYPES = {
    "mp4": "video/mp4",
    "avi": "video/x-msvideo",
    "mov": "video/quicktime",
    "mkv": "video/x-matroska",
    "webm": "video/webm",
}

# ── System prompts ────────────────────────────────────────────────────

_CRASH_ONLY_SYSTEM_PROMPT = (
    "You are an expert video analyst specialising in road traffic "
    "collision detection. Analyse the provided dashcam and CCTV footage "
    "for crash events, near-misses, impact moments, sequence of events, "
    "road-user involvement (vehicles, pedestrians, cyclists), probable "
    "cause, and confidence per event. Include timestamps and concise "
    "descriptions, then return your findings as a JSON object."
)

_SCENE_ANALYSIS_SYSTEM_PROMPT = (
    "You are an expert multimodal video analyst specialising in road scene "
    "understanding from dashcam and CCTV footage. Your task is to provide a "
    "comprehensive, structured description of EVERYTHING visible in the video: "
    "all vehicles, pedestrians, cyclists, road infrastructure, signage, lane "
    "markings, buildings, vegetation, weather and lighting conditions, and any "
    "notable events (lane changes, braking, turns, pedestrian crossings, etc.). "
    "Pay special attention to any collisions or near-miss events — whether the "
    "dashcam vehicle is directly involved OR merely witnesses a crash between "
    "other road users. For every collision or near-miss, provide a detailed "
    "forensic-level breakdown: impact type, each involved party's description "
    "and actions before/after, probable cause, fault indicators, sequence of "
    "events, post-crash actions, and estimated speeds. Be thorough and precise "
    "with timestamps. Return your findings as a structured JSON object."
)


class VideoLLM:
    """Async wrapper around LiteLLM for video-based analysis with Langfuse instrumentation.

    Responsibilities:
    - Configure LiteLLM global settings from a MetadataTemplate instance.
    - Accept one or more video sources (dashcam front/rear, CCTV).
    - Encode video data and send multimodal requests to Gemini 2.5 Pro.
    - Parse structured output from model responses into Pydantic models.
    - Log inputs/outputs and usage statistics to Langfuse.

    Two analysis modes are supported:
    - ``analyse_crash_footage``: Original crash-only detection returning
      ``VideoAnalysisOutput`` (retained for backward compatibility).
    - ``analyse_scene_footage``: Full scene description returning
      ``VideoSceneAnalysisOutput`` with objects, environment, and rich
      event metadata including forensic crash details.
    """

    def __init__(
        self,
        llm_agent: VideoAgentTemplate,
        langfuse: Langfuse,
        metadata_instance: MetadataTemplate | None = None,
    ) -> None:
        """
        Parameters:
            metadata_instance: Configuration object containing LLM + Langfuse settings.
            llm_agent: Video agent configuration including model name and parameters.
            langfuse: Langfuse client instance used for tracing generations.
        """
        self.metadata_instance = metadata_instance
        self.langfuse = langfuse
        self.llm_agent = llm_agent

        # Configure LiteLLM client settings (global module vars in litellm)
        litellm.api_key = getattr(
            getattr(self.metadata_instance, "llm", None), "client_secret", None
        ) or os.getenv("LITELLM_SECRET")
        litellm.api_base = getattr(
            getattr(self.metadata_instance, "llm", None), "endpoint", None
        ) or os.getenv("LITELLM_URL")
        litellm.api_version = self.llm_agent.api_version
        litellm.suppress_debug_info = True

    # ── Static helpers ────────────────────────────────────────────────

    @staticmethod
    def _encode_video_to_base64(video_path: str) -> str:
        """Read a video file and return its base64-encoded string.

        Parameters:
            video_path: Filesystem path to the video file.

        Returns:
            Base64-encoded string of the video bytes.

        Raises:
            FileNotFoundError: If the video file does not exist.
            ValueError: If the file extension is unsupported.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        ext = path.suffix.lstrip(".").lower()
        if ext not in _VIDEO_MIME_TYPES:
            raise ValueError(
                f"Unsupported video format '.{ext}'. Supported: {list(_VIDEO_MIME_TYPES.keys())}"
            )

        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _get_mime_type(video_path: str) -> str:
        """Resolve MIME type from a video file path."""
        ext = Path(video_path).suffix.lstrip(".").lower()
        return _VIDEO_MIME_TYPES.get(ext, "video/mp4")

    # ── Validation ────────────────────────────────────────────────────

    def _validate_video_sources(self, camera_sources: list[CameraSource]) -> None:
        """Validate that all video sources meet size and format constraints.

        Parameters:
            camera_sources: List of camera source descriptors.

        Raises:
            ValueError: If any file exceeds size limit or has unsupported format.
            FileNotFoundError: If a referenced file does not exist.
        """
        for source in camera_sources:
            path = Path(source.video_file_name)
            if not path.exists():
                raise FileNotFoundError(
                    f"Video file not found: {source.video_file_name}"
                )

            ext = path.suffix.lstrip(".").lower()
            if ext not in self.llm_agent.supported_formats:
                raise ValueError(
                    f"Unsupported format '.{ext}' for {source.video_file_name}. "
                    f"Supported: {self.llm_agent.supported_formats}"
                )

            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self.llm_agent.max_video_size_mb:
                raise ValueError(
                    f"Video {source.video_file_name} is {size_mb:.1f} MB, "
                    f"exceeding limit of {self.llm_agent.max_video_size_mb} MB."
                )

    # ── Payload construction ──────────────────────────────────────────

    def _build_video_content_parts(
        self, camera_sources: list[CameraSource]
    ) -> list[dict]:
        """Build multimodal content parts for the LiteLLM messages payload.

        Each video is base64-encoded and wrapped in the format expected by
        LiteLLM's multimodal API for Gemini models.

        Parameters:
            camera_sources: List of camera source descriptors.

        Returns:
            List of content-part dicts compatible with LiteLLM message format.
        """
        parts = []
        for source in camera_sources:
            mime = self._get_mime_type(source.video_file_name)
            b64 = self._encode_video_to_base64(source.video_file_name)

            parts.append(
                {
                    "type": "text",
                    "text": f"[{source.camera_type}] Video file: {Path(source.video_file_name).name}",
                }
            )
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}",
                    },
                }
            )

        return parts

    # ── LLM call + Langfuse instrumentation (shared) ──────────────────

    async def _call_llm(
        self,
        messages: list[dict],
        prompt_text: str,
        prompt_object: TextPromptClient | None,
        task_name: str | None,
        temperature: float | None,
        reasoning_effort: (
            Literal["none", "minimal", "low", "medium", "high", "default"] | None
        ),
    ) -> tuple[dict, str]:
        """Send a multimodal request to the LLM and return the parsed JSON and model name.

        This method encapsulates the LiteLLM call, response validation,
        Langfuse generation logging, and JSON parsing — shared across both
        the crash-only and scene-analysis pipelines.

        Parameters:
            messages: Fully assembled messages payload (system + user).
            prompt_text: The rendered prompt text (for Langfuse input logging).
            prompt_object: Langfuse prompt object for tracing lineage.
            task_name: Logical label for this generation.
            temperature: Sampling temperature override.
            reasoning_effort: Reasoning effort level override.

        Returns:
            A tuple of (parsed_json_dict, model_name_str).

        Raises:
            FileNotFoundError: If any video file is missing.
            ValueError: If video size/format constraints are violated.
            TypeError: If the response is not a ModelResponse.
            Exception: Re-raises any exception from litellm.acompletion.
        """
        if temperature is None:
            temperature = getattr(self.llm_agent, "temperature", 0.0)
        if reasoning_effort is None:
            reasoning_effort = getattr(self.llm_agent, "reasoning_effort", None)

        llm_payload = ObservationPayload(
            name=task_name or "video_analysis",
            prompt=prompt_object,
            as_type="generation",
        )

        with langfuse_observation(
            langfuse=self.langfuse, payload=llm_payload
        ) as generation:
            try:
                raw_response = await litellm.acompletion(
                    model=self.llm_agent.model_name,
                    messages=messages,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    response_format={"type": "json_object"},
                    cache={"no-store": True},
                )
            except Exception as e:
                generation.update(level="ERROR", status_message=str(e))
                logger.error("Video LLM call failed: %s", e)
                raise

            # Narrow the union type to ModelResponse for attribute access
            if not isinstance(raw_response, ModelResponse):
                err_msg = (
                    f"Expected ModelResponse but got {type(raw_response).__name__}. "
                    "Streaming responses are not supported."
                )
                generation.update(level="ERROR", status_message=err_msg)
                raise TypeError(err_msg)

            response: ModelResponse = raw_response

            # Validate choices exist
            if not response.choices:
                err_msg = "LLM returned no choices."
                generation.update(level="ERROR", status_message=err_msg)
                raise ValueError(err_msg)

            first_choice = response.choices[0]

            # choices[0] could be StreamingChoices; guard for the non-streaming type
            chat_response: str | None = getattr(
                getattr(first_choice, "message", None), "content", None
            )
            model_name: str = response.model or "unknown"
            usage = getattr(response, "usage", None)

            # Guard against None or empty content
            if not chat_response:
                err_msg = "LLM returned empty or null content."
                generation.update(level="ERROR", status_message=err_msg)
                raise ValueError(err_msg)

            # Convert Usage object to dict for Langfuse
            usage_dict: dict[str, int] | None = None
            if usage is not None:
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                }

            # Respect Langfuse capture settings
            llm_input = (
                prompt_text
                if getattr(
                    getattr(self.metadata_instance, "langfuse", None),
                    "capture_input",
                    False,
                )
                else None
            )
            llm_output = (
                chat_response
                if getattr(
                    getattr(self.metadata_instance, "langfuse", None),
                    "capture_output",
                    False,
                )
                else None
            )

            generation.update(
                input=llm_input,
                output=llm_output,
                model=model_name,
                usage_details=usage_dict,
                model_parameters={
                    k: str(v)
                    for k, v in {
                        "temperature": temperature,
                        "reasoning_effort": reasoning_effort,
                    }.items()
                    if v is not None
                },
            )

            parsed = json.loads(chat_response)

        return parsed, model_name

    # ── Response parsing helpers ──────────────────────────────────────

    @staticmethod
    def _parse_crash_events(parsed: dict) -> list[CrashEvent]:
        """Parse crash events from the raw JSON into CrashEvent models.

        Parameters:
            parsed: Raw parsed JSON dict from the LLM response.

        Returns:
            List of CrashEvent instances.
        """
        return [
            CrashEvent(**event) for event in parsed.get("crash_events", [])
        ]

    @staticmethod
    def _parse_environment(parsed: dict) -> EnvironmentContext | None:
        """Parse environmental context from the raw JSON.

        Parameters:
            parsed: Raw parsed JSON dict from the LLM response.

        Returns:
            An EnvironmentContext instance, or None if not present.
        """
        env_raw = parsed.get("environment")
        if env_raw and isinstance(env_raw, dict):
            return EnvironmentContext(**env_raw)
        return None

    @staticmethod
    def _parse_detected_objects(parsed: dict) -> list[DetectedObject]:
        """Parse detected objects from the raw JSON.

        Parameters:
            parsed: Raw parsed JSON dict from the LLM response.

        Returns:
            List of DetectedObject instances.
        """
        return [
            DetectedObject(**obj) for obj in parsed.get("detected_objects", [])
        ]

    @staticmethod
    def _parse_scene_events(parsed: dict) -> list[SceneEvent]:
        """Parse scene events from the raw JSON, including nested crash details.

        For events with ``event_type`` of ``'collision'`` or ``'near_miss'``,
        the nested ``witnessed_crash_detail`` object is hydrated into
        ``WitnessedCrashDetail`` (with its own ``CrashInvolvedParty`` list).

        Parameters:
            parsed: Raw parsed JSON dict from the LLM response.

        Returns:
            List of SceneEvent instances.
        """
        events: list[SceneEvent] = []

        for evt_raw in parsed.get("events", []):
            # Work on a copy to avoid mutating the raw_output reference
            evt_data = dict(evt_raw)
            crash_detail_raw = evt_data.pop("witnessed_crash_detail", None)

            crash_detail: WitnessedCrashDetail | None = None
            if crash_detail_raw is not None and isinstance(crash_detail_raw, dict):
                detail_data = dict(crash_detail_raw)
                parties_raw = detail_data.pop("involved_parties", [])
                parties = [CrashInvolvedParty(**p) for p in parties_raw]
                crash_detail = WitnessedCrashDetail(
                    involved_parties=parties, **detail_data
                )

            events.append(
                SceneEvent(witnessed_crash_detail=crash_detail, **evt_data)
            )

        return events

    # ── Public analysis methods ───────────────────────────────────────

    async def analyse_crash_footage(
        self,
        camera_sources: list[CameraSource],
        prompt_text: str,
        prompt_object: TextPromptClient | None = None,
        task_name: str | None = None,
        temperature: float | None = None,
        reasoning_effort: (
            Literal["none", "minimal", "low", "medium", "high", "default"] | None
        ) = None,
    ) -> VideoAnalysisOutput:
        """Analyse one or more video files to detect crash timestamps.

        Sends a multimodal request containing video data and a text prompt to
        Gemini 2.5 Pro via LiteLLM. The model is instructed to return structured
        JSON with crash event timestamps and descriptions.

        .. note::
            This method is retained for backward compatibility with the
            crash-only pipeline. For richer scene-level analysis, use
            :meth:`analyse_scene_footage` instead.

        Parameters:
            camera_sources: List of CameraSource objects describing each video.
            prompt_text: The fully rendered prompt instructing the model on the
                         crash detection task.
            prompt_object: Langfuse prompt object for tracing lineage.
            task_name: Logical label for this generation (e.g. 'crash_detection').
            temperature: Sampling temperature override.
            reasoning_effort: Reasoning effort level override.

        Returns:
            VideoAnalysisOutput containing detected crash events and summary.

        Raises:
            FileNotFoundError: If any video file is missing.
            ValueError: If video size/format constraints are violated.
            TypeError: If the response is not a ModelResponse.
            Exception: Re-raises any exception from litellm.acompletion.
        """
        self._validate_video_sources(camera_sources)

        # Build multimodal message content
        video_parts = self._build_video_content_parts(camera_sources)
        video_parts.append({"type": "text", "text": prompt_text})

        messages = [
            {"role": "system", "content": _CRASH_ONLY_SYSTEM_PROMPT},
            {"role": "user", "content": video_parts},
        ]

        parsed, _ = await self._call_llm(
            messages=messages,
            prompt_text=prompt_text,
            prompt_object=prompt_object,
            task_name=task_name or "video_crash_analysis",
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

        crash_events = self._parse_crash_events(parsed)

        output = VideoAnalysisOutput(
            task_name=task_name or "video_crash_analysis",
            crash_events=crash_events,
            summary=parsed.get("summary"),
            raw_output=parsed,
        )

        return output

    async def analyse_scene_footage(
        self,
        camera_sources: list[CameraSource],
        prompt_text: str,
        prompt_object: TextPromptClient | None = None,
        task_name: str | None = None,
        temperature: float | None = None,
        reasoning_effort: (
            Literal["none", "minimal", "low", "medium", "high", "default"] | None
        ) = None,
        system_prompt: str | None = None,
    ) -> VideoSceneAnalysisOutput:
        """Analyse one or more video files for full scene description.

        Sends a multimodal request containing video data and a text prompt to
        Gemini 2.5 Pro via LiteLLM. The model returns a comprehensive structured
        JSON covering all detected objects, environmental context, and notable
        events — including forensic-level crash details when collisions or
        near-misses are observed.

        Parameters:
            camera_sources: List of CameraSource objects describing each video.
            prompt_text: The fully rendered prompt instructing the model on the
                         scene analysis task.
            prompt_object: Langfuse prompt object for tracing lineage.
            task_name: Logical label for this generation (e.g. 'scene_analysis').
            temperature: Sampling temperature override.
            reasoning_effort: Reasoning effort level override.
            system_prompt: Optional system prompt override. Defaults to the
                           built-in scene analysis system prompt.

        Returns:
            VideoSceneAnalysisOutput containing environment context, detected
            objects, scene events (with nested crash details where applicable),
            and an overall summary.

        Raises:
            FileNotFoundError: If any video file is missing.
            ValueError: If video size/format constraints are violated.
            TypeError: If the response is not a ModelResponse.
            Exception: Re-raises any exception from litellm.acompletion.
        """
        self._validate_video_sources(camera_sources)

        # Build multimodal message content
        video_parts = self._build_video_content_parts(camera_sources)
        video_parts.append({"type": "text", "text": prompt_text})

        messages = [
            {
                "role": "system",
                "content": system_prompt or _SCENE_ANALYSIS_SYSTEM_PROMPT,
            },
            {"role": "user", "content": video_parts},
        ]

        parsed, _ = await self._call_llm(
            messages=messages,
            prompt_text=prompt_text,
            prompt_object=prompt_object,
            task_name=task_name or "video_scene_analysis",
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

        # Parse each section of the response
        environment = self._parse_environment(parsed)
        detected_objects = self._parse_detected_objects(parsed)
        events = self._parse_scene_events(parsed)

        # Derive camera_type and video_file_name from the first source
        primary_camera_type = camera_sources[0].camera_type if camera_sources else None
        primary_video_file = (
            Path(camera_sources[0].video_file_name).name if camera_sources else None
        )

        output = VideoSceneAnalysisOutput(
            task_name=task_name or "video_scene_analysis",
            camera_type=primary_camera_type,
            video_file_name=primary_video_file,
            environment=environment,
            detected_objects=detected_objects,
            events=events,
            summary=parsed.get("summary"),
            raw_output=parsed,
        )

        logger.info(
            "Scene analysis complete for task '%s' — %d object(s), %d event(s) "
            "(%d crash/near-miss).",
            output.task_name,
            len(output.detected_objects),
            len(output.events),
            sum(
                1
                for e in output.events
                if e.event_type in ("collision", "near_miss")
            ),
        )

        return output
