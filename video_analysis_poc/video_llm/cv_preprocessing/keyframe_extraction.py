"""Key frame extraction at critical timestamps.

Extracts and base64-encodes still images from a video at specified
timestamps.  These frames supplement the full video in the LLM payload,
providing pixel-level spatial resolution at the moments when vehicles
interact closely (e.g. a truck and car entering a roundabout side-by-side).

The extractor supports both synchronous and async (thread-offloaded)
interfaces to avoid blocking the event loop during disk I/O.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KeyFrame:
    """A single extracted key frame.

    Attributes:
        timestamp_seconds: The video timestamp this frame was captured at.
        base64_jpeg: Base64-encoded JPEG bytes.
    """

    timestamp_seconds: float
    base64_jpeg: str


class KeyFrameExtractor:
    """Extracts and encodes key frames from a video file.

    Parameters:
        max_dimension: Maximum width or height (whichever is larger) for
            the extracted frame.  Frames exceeding this are downscaled
            while preserving aspect ratio.
        jpeg_quality: JPEG encoding quality (0-100).
    """

    def __init__(
        self,
        max_dimension: int = 768,
        jpeg_quality: int = 85,
    ) -> None:
        self._max_dim = max_dimension
        self._jpeg_quality = jpeg_quality

    def _extract_sync(
        self, video_path: Path, timestamps: list[float]
    ) -> list[KeyFrame]:
        """Synchronous extraction — called via ``asyncio.to_thread``."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            logger.warning("Could not determine FPS for %s.", video_path)
            cap.release()
            return []

        frames: list[KeyFrame] = []
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
            ret, frame = cap.read()
            if not ret:
                logger.warning("Could not read frame at %.1fs from %s.", ts, video_path)
                continue

            h, w = frame.shape[:2]
            scale = min(self._max_dim / w, self._max_dim / h, 1.0)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            )
            if not ok:
                logger.warning("JPEG encoding failed for frame at %.1fs.", ts)
                continue

            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            frames.append(KeyFrame(timestamp_seconds=ts, base64_jpeg=b64))

        cap.release()
        logger.info(
            "Extracted %d/%d key frames from %s.",
            len(frames),
            len(timestamps),
            video_path.name,
        )
        return frames

    def extract(
        self, video_path: str | Path, timestamps: list[float]
    ) -> list[KeyFrame]:
        """Extract key frames (synchronous).

        Parameters:
            video_path: Path to the video file.
            timestamps: List of timestamps (in seconds) to extract.

        Returns:
            List of ``KeyFrame`` instances (in timestamp order).
        """
        return self._extract_sync(Path(video_path), timestamps)

    async def async_extract(
        self, video_path: str | Path, timestamps: list[float]
    ) -> list[KeyFrame]:
        """Extract key frames (async, thread-offloaded).

        Parameters:
            video_path: Path to the video file.
            timestamps: List of timestamps (in seconds) to extract.

        Returns:
            List of ``KeyFrame`` instances (in timestamp order).
        """
        return await asyncio.to_thread(
            self._extract_sync, Path(video_path), timestamps
        )
