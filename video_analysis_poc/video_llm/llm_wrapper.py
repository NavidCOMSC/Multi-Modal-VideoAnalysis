"""Module for video-based LLM inference using Gemini 2.5 Pro via LiteLLM.

Supports three analysis modes:

1. **Crash-only** (``async_analyse_crash_footage``) — original pipeline,
   returns ``VideoAnalysisOutput``.
2. **Full scene** (``async_analyse_scene_footage``) — comprehensive scene
   description, returns ``VideoSceneAnalysisOutput``.
3. **KG-augmented scene** (``async_analyse_scene_footage_kg``) — two-pass
   analysis with knowledge-graph RAG, key-frame supplementation, camera
   auto-detection, chain-of-thought reasoning, and KG-grounded verification.

Async optimisations
-------------------
- Video file I/O (validation, base64 encoding) is offloaded to the thread
  pool via ``asyncio.to_thread``.
- Multi-camera payloads are built concurrently with ``asyncio.gather``.
- Batch methods process multiple camera sources in parallel, bounded by
  an ``asyncio.Semaphore``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
from typing import Literal

import litellm
from langfuse import Langfuse
from langfuse.model import TextPromptClient
from litellm.files.main import ModelResponse

from video_analysis_poc.connection.langfuse.tracing_connection import (
    langfuse_observation,
)
from video_analysis_poc.video_llm.cv_preprocessing import (
    CameraProfileDetector,
    KeyFrameExtractor,
    KeyFrame,
)
from video_analysis_poc.video_llm.knowledge_graph import (
    CollisionKnowledgeGraph,
    KGRetriever,
)
from video_analysis_poc.video_llm.langfuse_schema import ObservationPayload
from video_analysis_poc.video_llm.prompt_composer import PromptComposer
from video_analysis_poc.video_llm.schema.video import (
    CameraSource,
    CrashEvent,
    CrashInvolvedParty,
    EnvironmentContext,
    ReasoningTrace,
    SceneEvent,
    VideoAgentTemplate,
    VideoAnalysisOutput,
    VideoObjectObservation,
    VideoSceneAnalysisOutput,
    WitnessedCrashDetail,
)
from video_analysis_poc.video_llm.use_case_config.metadata import MetadataTemplate

logger = logging.getLogger(__name__)

# Mapping of file extensions to MIME types for video content
_VIDEO_MIME_TYPES = {
    "mp4": "video/mp4",
    "avi": "video/x-msvideo",
    "mov": "video/quicktime",
    "mkv": "video/x-matroska",
    "webm": "video/webm",
}

# Default ceiling for concurrent LLM requests in batch methods
_DEFAULT_MAX_CONCURRENT_REQUESTS = 3


# ── System prompts (non-KG paths) ────────────────────────────────────────────────────

_CRASH_ONLY_SYSTEM_PROMPT = (
    "You are an expert video analyst specialising in road traffic "
    "collision detection. Analyse the provided dashcam and CCTV footage "
    "for crash events, near-misses, impact moments, sequence of events, "
    "road-user involvement (vehicles, pedestrians, cyclists), probable "
    "cause, and confidence per event. Include timestamps and concise "
    "descriptions, then return your findings as a JSON object.\n"
    "IMPORTANT ANALYSIS PRINCIPLES:\n"
    "- Always establish CAUSAL relationships between events. If a vehicle is "
    "stopped in an unusual location, determine what caused it to stop by "
    "examining the preceding seconds — do NOT assume it was inexplicably "
    "stationary without a reason.\n"
    "- Large vehicles like trucks and HGVs have wide turning arcs that "
    "frequently cause sideswipe collisions with adjacent vehicles on "
    "roundabouts and tight curves. Look carefully for this pattern whenever "
    "a truck and a smaller vehicle enter a roundabout side by side.\n"
    "- The striking vehicle may leave the scene after a collision (hit-and-run "
    "or scrape-and-leave). Do not assume the first vehicle you see near a "
    "stopped car is the one that struck it — trace the timeline backwards.\n"
    "- Do NOT confuse vehicles that arrive AFTER a collision with the vehicle "
    "that CAUSED it. Temporal ordering is critical: establish which interaction "
    "happened first.\n"
    "- Still frames are provided at key timestamps alongside the video. Use "
    "them to verify spatial relationships between vehicles at critical moments."
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
    """Async wrapper around LiteLLM for video-based crash detection with Langfuse instrumentation.

    Responsibilities:
    - Configure LiteLLM global settings from a MetadataTemplate instance.
    - Accept one or more video sources (dashcam front/rear, CCTV).
    - Encode video data and send multimodal requests to Gemini 2.5 Pro.
    - Parse structured output from model responses into Pydantic models.
    - Log inputs/outputs and usage statistics to Langfuse.

    Parameters:
        llm_agent: Video agent configuration including model name and parameters.
        langfuse: Langfuse client instance used for tracing generations.
        metadata_instance: Configuration object containing LLM + Langfuse settings.
        max_concurrent_requests: Maximum parallel LLM requests for batch methods.

    Three public analysis methods are provided:
    - ``analyse_crash_footage``: Original crash-only detection returning
      ``VideoAnalysisOutput`` (retained for backward compatibility).
    - ``analyse_scene_footage``: Full scene description returning
      ``VideoSceneAnalysisOutput`` with objects, environment, and rich
      event metadata including forensic crash details.
    - ``analyse_scene_footage_batch``: Runs ``analyse_scene_footage`` for
      each camera source concurrently, bounded by a semaphore.
    """

    def __init__(
        self,
        llm_agent: VideoAgentTemplate,
        langfuse: Langfuse,
        metadata_instance: MetadataTemplate | None = None,
        max_concurrent_requests: int = _DEFAULT_MAX_CONCURRENT_REQUESTS,
    ) -> None:
        self.metadata_instance = metadata_instance
        self.langfuse = langfuse
        self.llm_agent = llm_agent
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

        # Configure LiteLLM client settings (global module vars in litellm)
        litellm.api_key = getattr(
            getattr(self.metadata_instance, "llm", None), "client_secret", None
        ) or os.getenv("LITELLM_SECRET")
        litellm.api_base = getattr(
            getattr(self.metadata_instance, "llm", None), "endpoint", None
        ) or os.getenv("LITELLM_URL")
        litellm.api_version = self.llm_agent.api_version
        litellm.suppress_debug_info = True

    # ── Async file I/O helpers ─────────────────────────────────────────────

    @staticmethod
    def _encode_video_to_base64_sync(video_path: str) -> str:
        """Synchronous base64 encoding — called via ``asyncio.to_thread``.

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

    @classmethod
    async def _async_encode_video_to_base64(cls, video_path: str) -> str:
        """Non-blocking base64 encoding of a video file.

        Offloads the blocking file read to the default thread-pool executor
        so the event loop remains responsive while large video files are
        read from disk.

        Parameters:
            video_path: Filesystem path to the video file.

        Returns:
            Base64-encoded string of the video bytes.
        """
        return await asyncio.to_thread(cls._encode_video_to_base64_sync, video_path)

    @staticmethod
    def _get_mime_type(video_path: str) -> str:
        """Resolve MIME type from a video file path."""
        ext = Path(video_path).suffix.lstrip(".").lower()
        return _VIDEO_MIME_TYPES.get(ext, "video/mp4")

    # ── Validation ────────────────────────────────────────────────────

    @staticmethod
    def _validate_single_source_sync(
        source: CameraSource,
        supported_formats: list[str],
        max_video_size_mb: float,
    ) -> None:
        """Validate a single camera source (synchronous, for thread offloading).

        Raises:
            ValueError: If file exceeds size limit or has unsupported format.
            FileNotFoundError: If file does not exist.
        """
        path = Path(source.video_file_name)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {source.video_file_name}")

        ext = path.suffix.lstrip(".").lower()
        if ext not in supported_formats:
            raise ValueError(
                f"Unsupported format '.{ext}' for {source.video_file_name}. "
                f"Supported: {supported_formats}"
            )

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_video_size_mb:
            raise ValueError(
                f"Video {source.video_file_name} is {size_mb:.1f} MB, "
                f"exceeding limit of {max_video_size_mb} MB."
            )

    async def _async_validate_video_sources(
        self, camera_sources: list[CameraSource]
    ) -> None:
        """Validate all video sources concurrently.

        Each source's existence, format, and size check is offloaded to the
        thread pool so that multiple ``stat()`` calls do not serialize on
        the event loop.

        Parameters:
            camera_sources: List of camera source descriptors.

        Raises:
            ValueError: If any file exceeds size limit or has unsupported format.
            FileNotFoundError: If a referenced file does not exist.
        """
        await asyncio.gather(
            *(
                asyncio.to_thread(
                    self._validate_single_source_sync,
                    source,
                    self.llm_agent.supported_formats,
                    self.llm_agent.max_video_size_mb,
                )
                for source in camera_sources
            )
        )

    # ── Payload construction ──────────────────────────────────────────

    async def _async_build_single_source_parts(
        self, source: CameraSource
    ) -> list[dict]:
        """Encode one camera source and return its content-part dicts.

        Parameters:
            source: A single camera source descriptor.

        Returns:
            A two-element list: [text-label part, base64-video part].
        """
        mime = self._get_mime_type(source.video_file_name)
        b64 = await self._async_encode_video_to_base64(source.video_file_name)

        return [
            {
                "type": "text",
                "text": (
                    f"[{source.camera_type}] Video file: "
                    f"{Path(source.video_file_name).name}"
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{b64}",
                },
            },
        ]

    async def _async_build_video_content_parts(
        self, camera_sources: list[CameraSource]
    ) -> list[dict]:
        """Build multimodal content parts for the LiteLLM messages payload.

        All camera sources are encoded concurrently via ``asyncio.gather``.

        Parameters:
            camera_sources: List of camera source descriptors.

        Returns:
            List of content-part dicts compatible with LiteLLM message format.
        """
        nested = await asyncio.gather(
            *(self._async_build_single_source_parts(s) for s in camera_sources)
        )
        # Flatten [[text, video], [text, video], …] → [text, video, text, video, …]
        parts: list[dict] = []
        for pair in nested:
            parts.extend(pair)
        return parts

    @staticmethod
    def _build_keyframe_content_parts(
        key_frames: list[KeyFrame],
        prompt_composer: PromptComposer,
    ) -> list[dict]:
        """Build multimodal content parts for supplementary key frames."""
        parts: list[dict] = []
        for kf in key_frames:
            parts.append(
                {
                    "type": "text",
                    "text": prompt_composer.build_keyframe_annotation(
                        kf.timestamp_seconds
                    ),
                }
            )
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{kf.base64_jpeg}"},
                }
            )
        return parts

    # ── LLM call + Langfuse instrumentation (shared) ──────────────────

    async def _async_call_llm(
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
        return [CrashEvent(**event) for event in parsed.get("crash_events", [])]

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
        return print("no environment information")

    @staticmethod
    def _parse_detected_objects(parsed: dict) -> list[VideoObjectObservation]:
        """Parse detected objects from the raw JSON.

        Parameters:
            parsed: Raw parsed JSON dict from the LLM response.

        Returns:
            List of VideoObjectObservation instances.
        """
        return [
            VideoObjectObservation(**obj) for obj in parsed.get("detected_objects", [])
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

        for event_raw in parsed.get("events", []):
            # Work on a copy to avoid mutating the raw_output reference
            evt_data = dict(event_raw)
            crash_detail_raw = evt_data.pop("witnessed_crash_detail", None)

            crash_detail: WitnessedCrashDetail | None = None
            if crash_detail_raw is not None and isinstance(crash_detail_raw, dict):
                detail_data = dict(crash_detail_raw)
                parties_raw = detail_data.pop("involved_parties", [])
                parties = [CrashInvolvedParty(**p) for p in parties_raw]
                crash_detail = WitnessedCrashDetail(
                    involved_parties=parties, **detail_data
                )

            events.append(SceneEvent(witnessed_crash_detail=crash_detail, **evt_data))

        return events

    @staticmethod
    def _parse_reasoning_trace(parsed: dict) -> ReasoningTrace | None:
        rt_raw = parsed.get("reasoning_trace")
        if rt_raw and isinstance(rt_raw, dict):
            return ReasoningTrace(**rt_raw)
        return None

    # ── Public analysis methods ───────────────────────────────────────

    async def async_analyse_crash_footage(
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
        await self._async_validate_video_sources(camera_sources)

        # Build multimodal message content
        video_parts = await self._async_build_video_content_parts(camera_sources)
        video_parts.append({"type": "text", "text": prompt_text})

        messages = [
            {"role": "system", "content": _CRASH_ONLY_SYSTEM_PROMPT},
            {"role": "user", "content": video_parts},
        ]

        parsed, _ = await self._async_call_llm(
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

    async def async_analyse_scene_footage(
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
        """Analyse one or more video files for full scene description (non-KG Pipeline).

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
        await self._async_validate_video_sources(camera_sources)

        # Build multimodal message content
        video_parts = await self._async_build_video_content_parts(camera_sources)
        video_parts.append({"type": "text", "text": prompt_text})

        messages = [
            {
                "role": "system",
                "content": system_prompt or _SCENE_ANALYSIS_SYSTEM_PROMPT,
            },
            {"role": "user", "content": video_parts},
        ]

        parsed, _ = await self._async_call_llm(
            messages=messages,
            prompt_text=prompt_text,
            prompt_object=prompt_object,
            task_name=task_name or "video_scene_analysis",
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

        primary_camera_type = camera_sources[0].camera_type if camera_sources else None
        primary_video_file = (
            Path(camera_sources[0].video_file_name).name if camera_sources else None
        )

        return VideoSceneAnalysisOutput(
            task_name=task_name or "video_scene_analysis",
            camera_type=primary_camera_type,
            video_file_name=primary_video_file,
            environment=self._parse_environment(parsed),
            detected_objects=self._parse_detected_objects(parsed),
            events=self._parse_scene_events(parsed),
            summary=parsed.get("summary"),
            reasoning_trace=self._parse_reasoning_trace(parsed),
            raw_output=parsed,
        )

    async def async_analyse_scene_footage_kg(
        self,
        camera_sources: list[CameraSource],
        knowledge_graph: CollisionKnowledgeGraph,
        detected_vehicle_types: list[str],
        detected_road_type: str | None = None,
        keyword_hints: list[str] | None = None,
        key_frame_timestamps: list[float] | None = None,
        prompt_object: TextPromptClient | None = None,
        task_name: str | None = None,
        temperature: float | None = None,
        reasoning_effort: (
            Literal["none", "minimal", "low", "medium", "high", "default"] | None
        ) = None,
        run_verification_pass: bool = True,
        jurisdiction: str = "UK",
        user_query: str | None = None,
    ) -> VideoSceneAnalysisOutput:
        """KG-augmented two-pass scene analysis with chain-of-thought reasoning.

        This is the most capable analysis mode, combining:
        - Camera auto-detection via optical flow
        - Knowledge graph pattern retrieval and regulation lookup
        - Key frame supplementation for spatial resolution
        - Composable prompt architecture with CoT reasoning
        - Two-pass verification against KG common-misidentification patterns

        Parameters:
            camera_sources: Camera source descriptors.
            knowledge_graph: Populated collision knowledge graph.
            detected_vehicle_types: Vehicle types observed (e.g. from YOLO
                or manual annotation).
            detected_road_type: Road geometry type.
            keyword_hints: Optional keyword cues for KG retrieval.
            key_frame_timestamps: Timestamps for supplementary frame extraction.
                If None, no key frames are added.
            prompt_object: Langfuse prompt object for tracing.
            task_name: Logical label for this generation.
            temperature: Sampling temperature override.
            reasoning_effort: Reasoning effort level override.
            run_verification_pass: Whether to run Pass 2 verification.
            jurisdiction: Jurisdiction for regulation retrieval.
            user_query: Optional incident-specific instruction to augment
                the KG-composed prompt (e.g. ``"Focus on the truck and SUV
                interaction at 15-20s"``).

        Returns:
            VideoSceneAnalysisOutput with reasoning trace and verified events.
        """
        await self._async_validate_video_sources(camera_sources)

        # Step 1: Camera auto-detection
        primary_video = camera_sources[0].video_file_name
        camera_detector = CameraProfileDetector()
        camera_profile = await camera_detector.async_detect(primary_video)

        # Step 2: KG retrieval
        retriever = KGRetriever(knowledge_graph)
        scored_patterns = retriever.retrieve_patterns(
            detected_vehicle_types=detected_vehicle_types,
            road_type=detected_road_type,
            keyword_hints=keyword_hints,
        )
        regulations = retriever.retrieve_regulations(
            road_type=detected_road_type,
            jurisdiction=jurisdiction,
        )

        # Step 3: Compose prompts
        composer = PromptComposer(
            camera_profile=camera_profile,
            scored_patterns=scored_patterns,
            regulations=regulations,
        )
        system_prompt = composer.build_system_prompt()
        task_prompt = composer.build_task_prompt(user_query=user_query)

        # Step 4: Build multimodal payload
        video_parts = await self._async_build_video_content_parts(camera_sources)

        # Step 5: Key frame extraction and supplementation
        if key_frame_timestamps:
            extractor = KeyFrameExtractor()
            key_frames = await extractor.async_extract(
                primary_video, key_frame_timestamps
            )
            kf_parts = self._build_keyframe_content_parts(key_frames, composer)
            video_parts.extend(kf_parts)

        video_parts.append({"type": "text", "text": task_prompt})

        messages_pass1 = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": video_parts},
        ]

        # Step 6: Pass 1 — KG-augmented analysis
        parsed_pass1, _ = await self._async_call_llm(
            messages=messages_pass1,
            prompt_text=task_prompt,
            prompt_object=prompt_object,
            task_name=f"{task_name or 'video_scene_analysis_kg'}_pass1",
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

        # Step 7: Pass 2 — Verification (optional)
        if run_verification_pass:
            verification_prompt = composer.build_verification_prompt(parsed_pass1)
            verify_parts = video_parts[:-1] + [
                {"type": "text", "text": verification_prompt}
            ]
            messages_pass2 = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": verify_parts},
            ]

            parsed_final, _ = await self._async_call_llm(
                messages=messages_pass2,
                prompt_text=verification_prompt,
                prompt_object=prompt_object,
                task_name=f"{task_name or 'video_scene_analysis_kg'}_pass2",
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
        else:
            parsed_final = parsed_pass1

        # Step 8: Parse into structured output
        primary_camera_type = camera_sources[0].camera_type if camera_sources else None
        primary_video_file = (
            Path(camera_sources[0].video_file_name).name if camera_sources else None
        )

        output = VideoSceneAnalysisOutput(
            task_name=task_name or "video_scene_analysis_kg",
            camera_type=primary_camera_type,
            video_file_name=primary_video_file,
            environment=self._parse_environment(parsed_final),
            detected_objects=self._parse_detected_objects(parsed_final),
            events=self._parse_scene_events(parsed_final),
            summary=parsed_final.get("summary"),
            reasoning_trace=self._parse_reasoning_trace(parsed_final),
            raw_output=parsed_final,
        )

        n_crash = sum(
            1 for e in output.events if e.event_type in ("collision", "near_miss")
        )
        logger.info(
            "KG-augmented analysis complete for task '%s' — %d object(s), "
            "%d event(s) (%d crash/near-miss), verification=%s.",
            output.task_name,
            len(output.detected_objects),
            len(output.events),
            n_crash,
            "yes" if run_verification_pass else "no",
        )

        return output

    async def async_analyse_scene_footage_batch(
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
        return_exceptions: bool = False,
    ) -> list[VideoSceneAnalysisOutput | BaseException]:
        """Analyse each camera source independently and concurrently.

        Each ``CameraSource`` is submitted as a separate
        ``async_analyse_scene_footage`` call.  An ``asyncio.Semaphore`` (configured
        via the constructor's ``max_concurrent_requests``) limits how many
        LLM requests are in-flight at once, preventing rate-limit exhaustion.

        Parameters:
            camera_sources: One CameraSource per video file to analyse.
            prompt_text: The fully rendered prompt for scene analysis.
            prompt_object: Langfuse prompt object for tracing lineage.
            task_name: Base task name — suffixed with the camera type for
                       each concurrent call.
            temperature: Sampling temperature override.
            reasoning_effort: Reasoning effort level override.
            system_prompt: Optional system prompt override.
            return_exceptions: If ``True``, exceptions from individual calls
                are returned in the results list instead of being raised.
                Mirrors ``asyncio.gather(return_exceptions=...)``.

        Returns:
            A list of ``VideoSceneAnalysisOutput`` (one per source), in the
            same order as ``camera_sources``.  If ``return_exceptions`` is
            ``True``, failed entries are ``BaseException`` instances.
        """

        async def _async_guarded_analyse(
            source: CameraSource,
        ) -> VideoSceneAnalysisOutput:
            """Run a single analysis under the semaphore."""
            async with self._semaphore:
                suffix = f"_{source.camera_type}" if task_name else ""
                return await self.async_analyse_scene_footage(
                    camera_sources=[source],
                    prompt_text=prompt_text,
                    prompt_object=prompt_object,
                    task_name=f"{task_name or 'video_scene_analysis'}{suffix}",
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    system_prompt=system_prompt,
                )

        results = await asyncio.gather(
            *(_async_guarded_analyse(src) for src in camera_sources),
            return_exceptions=return_exceptions,
        )

        succeeded = sum(1 for r in results if not isinstance(r, BaseException))
        logger.info(
            "Batch scene analysis: %d/%d sources completed successfully.",
            succeeded,
            len(camera_sources),
        )

        return list(results)
