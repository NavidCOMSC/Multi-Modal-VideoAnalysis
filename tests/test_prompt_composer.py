"""Unit tests for the prompt composer module.

Tests cover:
- System prompt assembly with/without camera profile
- Task prompt assembly with/without RAG context
- Verification prompt formatting
- Key frame annotation
"""

from __future__ import annotations

import json

import pytest

from video_analysis_poc.video_llm.cv_preprocessing.camera_detection import (
    CameraMotionType,
    CameraProfile,
)
from video_analysis_poc.video_llm.knowledge_graph.models import (
    CollisionPattern,
    ScoredPattern,
    TrafficRegulation,
)
from video_analysis_poc.video_llm.prompt_composer.composer import PromptComposer


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def camera_profile() -> CameraProfile:
    return CameraProfile(
        motion_type=CameraMotionType.EGO_FORWARD,
        avg_stability=0.42,
        prompt_context="Front-facing dashcam. Focus on relative motion.",
    )


@pytest.fixture
def scored_patterns() -> list[ScoredPattern]:
    return [
        ScoredPattern(
            pattern=CollisionPattern(
                pattern_id="roundabout_sideswipe_hgv",
                pattern_name="HGV Sideswipe on Roundabout",
                description="HGV swept path encroaches on adjacent vehicle.",
                visual_signatures=["Truck and car side-by-side"],
                causal_chain="HGV turning arc -> encroachment -> contact",
                common_misidentifications=[
                    "WRONG: Vehicle described as inexplicably stopped"
                ],
                keywords=["sideswipe", "roundabout"],
                example_narrative="Green truck scrapes black SUV on roundabout.",
            ),
            relevance_score=11.0,
        )
    ]


@pytest.fixture
def regulations() -> list[TrafficRegulation]:
    return [
        TrafficRegulation(
            reg_id="uk_roundabout_priority",
            jurisdiction="UK",
            category="roundabout",
            rule_text="Give way to traffic from the right (Rule 185).",
            analysis_implication="Vehicles on roundabout have priority.",
        )
    ]


# ── Tests ─────────────────────────────────────────────────────────────


class TestPromptComposer:
    def test_system_prompt_includes_camera_context(
        self, camera_profile, scored_patterns, regulations
    ):
        composer = PromptComposer(
            camera_profile=camera_profile,
            scored_patterns=scored_patterns,
            regulations=regulations,
        )
        prompt = composer.build_system_prompt()
        assert "Front-facing dashcam" in prompt
        assert "CAUSAL" in prompt

    def test_system_prompt_without_camera(self):
        composer = PromptComposer()
        prompt = composer.build_system_prompt()
        assert "CAMERA:" not in prompt
        assert "CAUSAL" in prompt

    def test_system_prompt_includes_cot_structure(self, camera_profile):
        composer = PromptComposer(camera_profile=camera_profile)
        prompt = composer.build_system_prompt()
        assert "GLOBAL PERCEPTION" in prompt
        assert "VEHICLE INVENTORY" in prompt
        assert "INTERACTION ANALYSIS" in prompt
        assert "CAUSAL CONCLUSION" in prompt

    def test_task_prompt_includes_rag_context(
        self, camera_profile, scored_patterns, regulations
    ):
        composer = PromptComposer(
            camera_profile=camera_profile,
            scored_patterns=scored_patterns,
            regulations=regulations,
        )
        prompt = composer.build_task_prompt()
        assert "DOMAIN EXPERTISE" in prompt
        assert "HGV Sideswipe on Roundabout" in prompt
        assert "ERRORS TO AVOID" in prompt
        assert "APPLICABLE REGULATIONS" in prompt

    def test_task_prompt_includes_schema(self, scored_patterns):
        composer = PromptComposer(scored_patterns=scored_patterns)
        prompt = composer.build_task_prompt()
        assert "reasoning_trace" in prompt
        assert "detected_objects" in prompt
        assert "witnessed_crash_detail" in prompt

    def test_task_prompt_without_rag(self):
        composer = PromptComposer()
        prompt = composer.build_task_prompt()
        assert "DOMAIN EXPERTISE" not in prompt
        assert "reasoning_trace" in prompt

    def test_verification_prompt_includes_pass1_output(self, scored_patterns):
        composer = PromptComposer(scored_patterns=scored_patterns)
        pass1 = {"events": [], "summary": "No collisions detected."}
        prompt = composer.build_verification_prompt(pass1)
        assert "No collisions detected" in prompt
        assert "Verify" in prompt
        assert "KB CHECK" in prompt

    def test_verification_prompt_without_patterns(self):
        composer = PromptComposer()
        pass1 = {"events": [], "summary": "Test."}
        prompt = composer.build_verification_prompt(pass1)
        assert "Verify" in prompt
        # No KB CHECK since no patterns
        assert "KB CHECK" not in prompt

    def test_keyframe_annotation(self):
        composer = PromptComposer()
        annotation = composer.build_keyframe_annotation(17.0)
        assert "17s" in annotation
        assert "STILL FRAME" in annotation

    def test_task_prompt_includes_user_query(self, scored_patterns):
        composer = PromptComposer(scored_patterns=scored_patterns)
        prompt = composer.build_task_prompt(
            user_query="Focus on the truck and SUV interaction at 15-20s"
        )
        assert "ADDITIONAL CONTEXT FROM THE USER" in prompt
        assert "Focus on the truck and SUV interaction" in prompt

    def test_task_prompt_omits_user_query_section_when_none(self, scored_patterns):
        composer = PromptComposer(scored_patterns=scored_patterns)
        prompt = composer.build_task_prompt(user_query=None)
        assert "ADDITIONAL CONTEXT FROM THE USER" not in prompt
