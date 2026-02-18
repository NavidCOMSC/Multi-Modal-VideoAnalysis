"""Module for video-based LLM inference using Gemini 2.5 Pro via LiteLLM."""

import litellm
from litellm.files.main import ModelResponse
import json
import asyncio
import base64
import logging
import os
from pathlib import Path
from langfuse import Langfuse
from langfuse.model import TextPromptClient
from langfuse_schema import ObservationPayload
from use_case_config.metadata import MetadataTemplate
from schema.video import (
    VideoAgentTemplate,
    VideoAnalysisOutput,
    CrashEvent,
    CameraSource,
)

from video_analysis_poc.connection.langfuse.tracing_connection import (
    langfuse_observation,
)
from typing import Optional, List, Literal


logger = logging.getLogger(__name__)

# Mapping of file extensions to MIME types for video content
_VIDEO_MIME_TYPES = {
    "mp4": "video/mp4",
    "avi": "video/x-msvideo",
    "mov": "video/quicktime",
    "mkv": "video/x-matroska",
    "webm": "video/webm",
}


class VideoLLM:
    """Async wrapper around LiteLLM for video-based crash detection with Langfuse instrumentation.

    Responsibilities:
    - Configure LiteLLM global settings from a MetadataTemplate instance.
    - Accept one or more video sources (dashcam front/rear, CCTV).
    - Encode video data and send multimodal requests to Gemini 2.5 Pro.
    - Parse structured crash-event timestamps from model responses.
    - Log inputs/outputs and usage statistics to Langfuse.
    """

    def __init__(
        self,
        llm_agent: VideoAgentTemplate,
        langfuse: Langfuse,
        metadata_instance: Optional[MetadataTemplate] = None,
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

    def _validate_video_sources(self, camera_sources: List[CameraSource]) -> None:
        """Validate that all video sources meet size and format constraints.

        Parameters:
            camera_sources: List of camera source descriptors.

        Raises:
            ValueError: If any file exceeds size limit or has unsupported format.
            FileNotFoundError: If a referenced file does not exist.
        """
        for source in camera_sources:
            path = Path(source.file_name)
            if not path.exists():
                raise FileNotFoundError(f"Video file not found: {source.file_name}")

            ext = path.suffix.lstrip(".").lower()
            if ext not in self.llm_agent.supported_formats:
                raise ValueError(
                    f"Unsupported format '.{ext}' for {source.file_name}. "
                    f"Supported: {self.llm_agent.supported_formats}"
                )

            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self.llm_agent.max_video_size_mb:
                raise ValueError(
                    f"Video {source.file_name} is {size_mb:.1f} MB, "
                    f"exceeding limit of {self.llm_agent.max_video_size_mb} MB."
                )

    def _build_video_content_parts(
        self, camera_sources: List[CameraSource]
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
            mime = self._get_mime_type(source.file_name)
            b64 = self._encode_video_to_base64(source.file_name)

            parts.append(
                {
                    "type": "text",
                    "text": f"[{source.camera_type}] Video file: {Path(source.file_name).name}",
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

    async def analyse_crash_footage(
        self,
        camera_sources: List[CameraSource],
        prompt_text: str,
        prompt_object: TextPromptClient | None = None,
        task_name: str | None = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[
            Literal["none", "minimal", "low", "medium", "high", "default"]
        ] = None,
    ) -> VideoAnalysisOutput:
        """Analyse one or more video files to detect crash timestamps.

        Sends a multimodal request containing video data and a text prompt to
        Gemini 2.5 Pro via LiteLLM. The model is instructed to return structured
        JSON with crash event timestamps and descriptions.

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
        # Validate all inputs before making the API call
        self._validate_video_sources(camera_sources)

        if temperature is None:
            temperature = getattr(self.llm_agent, "temperature", 0.0)
        if reasoning_effort is None:
            reasoning_effort = getattr(self.llm_agent, "reasoning_effort", None)

        # Build multimodal message content
        video_parts = self._build_video_content_parts(camera_sources)

        # Append the text prompt as the final user content part
        video_parts.append({"type": "text", "text": prompt_text})

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert video analyst specialising in road traffic "
                    "collision detection. Analyse the provided dashcam and CCTV footage "
                    "carefully. Return your findings as a JSON object."
                ),
            },
            {
                "role": "user",
                "content": video_parts,
            },
        ]

        llm_payload = ObservationPayload(
            name=task_name or "video_crash_analysis",
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

            # Parse structured output
            parsed = json.loads(chat_response)
            crash_events = [
                CrashEvent(**event) for event in parsed.get("crash_events", [])
            ]

            output = VideoAnalysisOutput(
                task_name=task_name or "video_crash_analysis",
                crash_events=crash_events,
                summary=parsed.get("summary"),
                raw_output=parsed,
            )

        return output
