"""
Camera motion classification via optical flow stability analysis.

Analyses the first few seconds of a video to classify the camera as:
- **static** (CCTV) — very low global optical flow; all motion is real.
- **ego_forward** (dashcam) — consistent forward-motion flow pattern;
  background motion is ego-vehicle movement.
- **handheld** — high flow variance (jitter); need to distinguish
  camera shake from vehicle behaviour.

The resulting ``CameraProfile`` includes a ``prompt_context`` string
that is injected into the LLM system prompt to calibrate the model's
interpretation of motion in the video.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraMotionType(str, Enum):
    """Enumeration of camera motion categories."""

    STATIC = "static"
    EGO_FORWARD = "ego_forward"
    HANDHELD = "handheld"


@dataclass(frozen=True)
class CameraProfile:
    """Result of camera motion classification.

    Attributes:
        motion_type: Detected camera motion category.
        avg_stability: Stability score in [0, 1] — higher means more static.
        prompt_context: Human-readable context string for injection into
            the LLM system prompt.
    """

    motion_type: CameraMotionType
    avg_stability: float
    prompt_context: str


# ── Prompt context strings ────────────────────────────────────────────

_PROMPT_CONTEXT_STATIC = (
    "Fixed CCTV. No ego-motion. All motion in the frame represents "
    "real vehicle and pedestrian movement."
)

_PROMPT_CONTEXT_EGO_FORWARD = (
    "Front-facing dashcam on a moving vehicle. ALL background motion is "
    "caused by the ego-vehicle's movement. Focus on RELATIVE motion "
    "between vehicles to identify interactions."
)

_PROMPT_CONTEXT_HANDHELD = (
    "Handheld or body-worn camera. Significant jitter present. "
    "Distinguish actual vehicle behaviour from camera shake."
)

_PROMPT_CONTEXT_FALLBACK = "Camera type could not be determined. Assume dashcam."


class CameraProfileDetector:
    """Classifies camera motion type from a video file.

    Uses Farneback dense optical flow on the first ``sample_seconds``
    seconds of the video.  The mean flow magnitude and its standard
    deviation are used to distinguish the three camera types.

    Parameters:
        sample_seconds: Number of seconds to analyse (default 5).
        max_frames: Hard cap on frames to process (prevents runaway
            on very high-FPS video).
        static_threshold: Stability value above which the camera is
            classified as static CCTV.
        jitter_ratio_threshold: Coefficient of variation threshold
            above which the camera is classified as handheld.
    """

    def __init__(
        self,
        sample_seconds: int = 5,
        max_frames: int = 150,
        static_threshold: float = 0.85,
        jitter_ratio_threshold: float = 1.5,
    ) -> None:
        self._sample_seconds = sample_seconds
        self._max_frames = max_frames
        self._static_threshold = static_threshold
        self._jitter_ratio_threshold = jitter_ratio_threshold

    def _detect_sync(self, video_path: Path) -> CameraProfile:
        """Synchronous detection — called via ``asyncio.to_thread``."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            logger.warning("Could not read first frame from %s.", video_path)
            return CameraProfile(
                CameraMotionType.EGO_FORWARD, 0.5, _PROMPT_CONTEXT_FALLBACK
            )

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        flow_mags: list[float] = []

        n_frames = min(int(self._sample_seconds * fps), self._max_frames)
        for _ in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                np.zeros((*prev_gray.shape, 2), dtype=np.float32),
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
            flow_mags.append(mag)
            prev_gray = gray

        cap.release()

        if not flow_mags:
            return CameraProfile(
                CameraMotionType.EGO_FORWARD, 0.5, _PROMPT_CONTEXT_FALLBACK
            )

        avg_flow = float(np.mean(flow_mags))
        stability = 1.0 / (1.0 + avg_flow)
        flow_std = float(np.std(flow_mags))

        if stability > self._static_threshold:
            return CameraProfile(
                CameraMotionType.STATIC, stability, _PROMPT_CONTEXT_STATIC
            )
        elif flow_std / (avg_flow + 1e-6) > self._jitter_ratio_threshold:
            return CameraProfile(
                CameraMotionType.HANDHELD, stability, _PROMPT_CONTEXT_HANDHELD
            )
        else:
            return CameraProfile(
                CameraMotionType.EGO_FORWARD, stability, _PROMPT_CONTEXT_EGO_FORWARD
            )

    def detect(self, video_path: str | Path) -> CameraProfile:
        """Classify the camera motion type (synchronous).

        Parameters:
            video_path: Path to the video file.

        Returns:
            A ``CameraProfile`` with the detected motion type.
        """
        profile = self._detect_sync(Path(video_path))
        logger.info(
            "Camera detected: %s (stability=%.3f) for %s.",
            profile.motion_type.value,
            profile.avg_stability,
            video_path,
        )
        return profile

    async def async_detect(self, video_path: str | Path) -> CameraProfile:
        """Classify the camera motion type (async, thread-offloaded).

        Parameters:
            video_path: Path to the video file.

        Returns:
            A ``CameraProfile`` with the detected motion type.
        """
        profile = await asyncio.to_thread(self._detect_sync, Path(video_path))
        logger.info(
            "Camera detected: %s (stability=%.3f) for %s.",
            profile.motion_type.value,
            profile.avg_stability,
            video_path,
        )
        return profile
