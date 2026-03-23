"""Unit tests for the CV preprocessing module.

Tests cover:
- Camera profile detector with synthetic video data
- Key frame extractor with synthetic video data
- Edge cases (empty video, unsupported format)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from video_analysis_poc.video_llm.cv_preprocessing.camera_detection import (
    CameraMotionType,
    CameraProfile,
    CameraProfileDetector,
)
from video_analysis_poc.video_llm.cv_preprocessing.keyframe_extraction import (
    KeyFrame,
    KeyFrameExtractor,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def static_video_path() -> Path:
    """Create a short synthetic video with no motion (simulating static CCTV)."""
    path = Path(tempfile.mktemp(suffix=".mp4"))
    h, w, fps, duration = 240, 320, 30, 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))

    # Write identical frames (no motion)
    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    for _ in range(fps * duration):
        writer.write(frame)

    writer.release()
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def moving_video_path() -> Path:
    """Create a short synthetic video with horizontal motion (simulating dashcam)."""
    path = Path(tempfile.mktemp(suffix=".mp4"))
    h, w, fps, duration = 240, 320, 30, 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))

    for i in range(fps * duration):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Draw a moving rectangle to simulate ego-motion
        x_offset = (i * 3) % w
        cv2.rectangle(frame, (x_offset, 50), (x_offset + 80, 190), (255, 255, 255), -1)
        writer.write(frame)

    writer.release()
    yield path
    path.unlink(missing_ok=True)


# ── Camera detection tests ────────────────────────────────────────────


class TestCameraProfileDetector:
    def test_static_video_detected_as_static(self, static_video_path):
        detector = CameraProfileDetector(sample_seconds=1)
        profile = detector.detect(static_video_path)
        assert isinstance(profile, CameraProfile)
        assert profile.motion_type == CameraMotionType.STATIC
        assert profile.avg_stability > 0.8

    def test_moving_video_not_static(self, moving_video_path):
        detector = CameraProfileDetector(sample_seconds=1)
        profile = detector.detect(moving_video_path)
        assert isinstance(profile, CameraProfile)
        assert profile.motion_type != CameraMotionType.STATIC

    def test_prompt_context_is_non_empty(self, static_video_path):
        detector = CameraProfileDetector(sample_seconds=1)
        profile = detector.detect(static_video_path)
        assert len(profile.prompt_context) > 10

    def test_nonexistent_file_returns_fallback(self):
        detector = CameraProfileDetector(sample_seconds=1)
        profile = detector.detect("/nonexistent/video.mp4")
        assert profile.motion_type == CameraMotionType.EGO_FORWARD

    @pytest.mark.asyncio
    async def test_async_detect(self, static_video_path):
        detector = CameraProfileDetector(sample_seconds=1)
        profile = await detector.async_detect(static_video_path)
        assert isinstance(profile, CameraProfile)


# ── Key frame extraction tests ────────────────────────────────────────


class TestKeyFrameExtractor:
    def test_extract_returns_frames(self, moving_video_path):
        extractor = KeyFrameExtractor(max_dimension=160)
        frames = extractor.extract(moving_video_path, [0.0, 0.5, 1.0])
        assert len(frames) > 0
        assert all(isinstance(f, KeyFrame) for f in frames)

    def test_extract_frame_has_base64_data(self, moving_video_path):
        extractor = KeyFrameExtractor()
        frames = extractor.extract(moving_video_path, [0.5])
        assert len(frames) == 1
        assert len(frames[0].base64_jpeg) > 100

    def test_extract_out_of_range_timestamp(self, moving_video_path):
        extractor = KeyFrameExtractor()
        frames = extractor.extract(moving_video_path, [999.0])
        # Should skip the out-of-range timestamp
        assert len(frames) == 0

    def test_extract_preserves_timestamp(self, moving_video_path):
        extractor = KeyFrameExtractor()
        frames = extractor.extract(moving_video_path, [0.5])
        assert frames[0].timestamp_seconds == 0.5

    @pytest.mark.asyncio
    async def test_async_extract(self, moving_video_path):
        extractor = KeyFrameExtractor(max_dimension=160)
        frames = await extractor.async_extract(moving_video_path, [0.0, 0.5])
        assert len(frames) > 0
