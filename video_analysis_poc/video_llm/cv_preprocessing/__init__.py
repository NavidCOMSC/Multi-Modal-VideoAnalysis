"""Computer vision pre-processing for video analysis.

This module provides:
- ``CameraProfileDetector``: Optical-flow-based camera motion classification
  (static CCTV, forward dashcam, handheld).
- ``KeyFrameExtractor``: Targeted frame extraction at critical timestamps
  for supplementary spatial resolution in the LLM payload.
"""

from video_analysis_poc.video_llm.cv_preprocessing.camera_detection import (
    CameraMotionType,
    CameraProfile,
    CameraProfileDetector,
)
from video_analysis_poc.video_llm.cv_preprocessing.keyframe_extraction import (
    KeyFrame,
    KeyFrameExtractor,
)

__all__ = [
    "CameraMotionType",
    "CameraProfile",
    "CameraProfileDetector",
    "KeyFrame",
    "KeyFrameExtractor",
]
