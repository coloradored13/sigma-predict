"""Contract tests for pipeline model changes and warnings_out wiring."""

from __future__ import annotations

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.models import (
    ChallengeEntry,
    HumanReview,
    PredictionRecord,
    RunResult,
    VerificationEntry,
)
from src.registry.store import RegistryStore


# ---------------------------------------------------------------------------
# PredictionRecord field defaults
# ---------------------------------------------------------------------------

class TestPredictionRecordDefaults:
    def test_pipeline_warnings_default(self):
        rec = PredictionRecord()
        assert rec.pipeline_warnings == []

    def test_close_date_default(self):
        rec = PredictionRecord()
        assert rec.close_date == ""

    def test_tags_default(self):
        rec = PredictionRecord()
        assert rec.tags == []

    def test_tags_are_independent(self):
        rec1 = PredictionRecord()
        rec2 = PredictionRecord()
        rec1.tags.append("foo")
        assert rec2.tags == []

    def test_pipeline_warnings_are_independent(self):
        rec1 = PredictionRecord()
        rec2 = PredictionRecord()
        rec1.pipeline_warnings.append("warning")
        assert rec2.pipeline_warnings == []


# ---------------------------------------------------------------------------
# RunResult field defaults
# ---------------------------------------------------------------------------

class TestRunResultDefaults:
    def test_parse_failed_default(self):
        rr = RunResult()
        assert rr.parse_failed is False

    def test_provider_default(self):
        rr = RunResult()
        assert rr.provider == ""

    def test_falsification_anchors_default(self):
        rr = RunResult()
        assert rr.falsification_anchors == []

    def test_falsification_anchors_are_independent(self):
        rr1 = RunResult()
        rr2 = RunResult()
        rr1.falsification_anchors.append("anchor")
        assert rr2.falsification_anchors == []


# ---------------------------------------------------------------------------
# HumanReview field default
# ---------------------------------------------------------------------------

class TestHumanReviewDefaults:
    def test_actor_analysis_consulted_default(self):
        hr = HumanReview()
        assert hr.actor_analysis_consulted is False


# ---------------------------------------------------------------------------
# JSONL roundtrip with new fields via RegistryStore
# ---------------------------------------------------------------------------

class TestJSONLRoundtrip:
    def test_new_fields_survive_roundtrip(self, tmp_path):
        store = RegistryStore(str(tmp_path / "test.jsonl"))
        rec = PredictionRecord(
            prediction_id="test-roundtrip",
            close_date="2026-06-01",
            tags=["geopolitics", "iran"],
            pipeline_warnings=["decomposition_parse_failure"],
        )
        store.save(rec)
        loaded = store.load_all()
        assert len(loaded) == 1
        r = loaded[0]
        assert r.close_date == "2026-06-01"
        assert r.tags == ["geopolitics", "iran"]
        assert r.pipeline_warnings == ["decomposition_parse_failure"]

    def test_run_result_new_fields_survive_roundtrip(self, tmp_path):
        store = RegistryStore(str(tmp_path / "test2.jsonl"))
        rr = RunResult(
            provider="anthropic",
            parse_failed=True,
            falsification_anchors=["anchor1"],
        )
        rec = PredictionRecord(prediction_id="test-rr", runs=[rr])
        store.save(rec)
        loaded = store.load_all()
        assert len(loaded) == 1
        r = loaded[0].runs[0]
        assert r.provider == "anthropic"
        assert r.parse_failed is True
        assert r.falsification_anchors == ["anchor1"]

    def test_old_record_missing_new_fields_loads_with_defaults(self, tmp_path):
        """Records written before the schema update load correctly with defaults."""
        registry_path = tmp_path / "legacy.jsonl"
        # Write a minimal record without new fields
        minimal = {
            "prediction_id": "legacy-1",
            "question_id": "q1",
            "platform": "metaculus",
            "question_text": "Legacy question",
            "question_type": "binary",
            "domain": "",
            "horizon_days": None,
            "created_at": "2026-01-01T00:00:00+00:00",
            "decomposition": {"sub_questions": [], "reference_class": "", "base_rate": 0.5, "base_rate_source": "", "actor_analysis": {"actors_involved": False, "actors": [], "level_2_3_present": False, "peripheral_actors": [], "probability_space_constraints": []}},
            "runs": [],
            "aggregation": {"method": "single_run", "raw_aggregate": 0.5, "stdev": 0.0, "inter_model_agreement": 1.0, "effective_n": 1, "deliberation_triggered": False, "deliberation_result": None},
            "calibration": {"platt_params_used": None, "calibrated_probability": 0.5, "confidence_interval": [0.0, 1.0]},
            "platform_signals": {"prices_at_prediction_time": {}, "cross_platform_divergence": None, "anomaly_detected": False, "anomaly_details": None},
            "human_review": {"reviewed": False, "human_adjustment": None, "human_reasoning": None, "flags_raised": []},
            "submitted": None,
            "resolution": None,
            "verification": {"enabled": False, "providers_available": [], "stages": []},
            "cost": {"total_tokens_in": 0, "total_tokens_out": 0, "total_cost_usd": 0.0, "latency_seconds": 0.0},
        }
        registry_path.write_text(json.dumps(minimal) + "\n", encoding="utf-8")
        store = RegistryStore(str(registry_path))
        loaded = store.load_all()
        assert len(loaded) == 1
        r = loaded[0]
        assert r.close_date == ""
        assert r.tags == []
        assert r.pipeline_warnings == []


# ---------------------------------------------------------------------------
# Literal validation: ChallengeEntry.vulnerability
# ---------------------------------------------------------------------------

class TestLiteralValidation:
    def test_valid_vulnerability(self):
        ce = ChallengeEntry(vulnerability="high")
        assert ce.vulnerability == "high"

    def test_valid_vulnerability_low(self):
        ce = ChallengeEntry(vulnerability="low")
        assert ce.vulnerability == "low"

    def test_invalid_vulnerability_raises(self):
        with pytest.raises(ValidationError):
            ChallengeEntry(vulnerability="critical")

    def test_invalid_vulnerability_uppercase_raises(self):
        with pytest.raises(ValidationError):
            ChallengeEntry(vulnerability="HIGH")

    def test_vulnerability_default(self):
        ce = ChallengeEntry()
        assert ce.vulnerability == "medium"

    def test_valid_assessment(self):
        ve = VerificationEntry(assessment="agree", confidence="high")
        assert ve.assessment == "agree"
        assert ve.confidence == "high"

    def test_invalid_assessment_raises(self):
        with pytest.raises(ValidationError):
            VerificationEntry(assessment="yes")

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValidationError):
            VerificationEntry(confidence="very_high")


# ---------------------------------------------------------------------------
# warnings_out wiring: decompose_question parse failure path
# ---------------------------------------------------------------------------

class TestWarningsOutWiring:
    def test_decompose_question_appends_warning_on_parse_failure(self):
        """When LLM returns unparseable JSON, warnings_out gets populated."""
        from src.pipeline.decomposer import decompose_question
        from src.models import PlatformQuestion, QuestionType, Platform

        question = PlatformQuestion(
            question_id="q-test",
            platform=Platform.METACULUS,
            title="Test question",
            question_type=QuestionType.BINARY,
        )

        mock_config = MagicMock()
        mock_config.primary_model.provider = "anthropic"
        mock_config.primary_model.model_id = "claude-3-5-sonnet"
        mock_config.primary_model.max_tokens = 1024
        mock_config.prompts_dir = "/nonexistent"

        mock_router = MagicMock()
        mock_router.call.return_value = ("not valid json at all !!!", 10, 10)

        warnings: list[str] = []

        with patch("src.pipeline.decomposer._load_prompt", return_value="system prompt"):
            result = decompose_question(
                question=question,
                config=mock_config,
                router=mock_router,
                warnings_out=warnings,
            )

        assert "decomposition_parse_failure" in warnings
        # Should return default Decomposition on parse failure
        assert result.sub_questions == []

    def test_decompose_question_no_warning_on_success(self):
        """When LLM returns valid JSON, warnings_out stays empty."""
        from src.pipeline.decomposer import decompose_question
        from src.models import PlatformQuestion, QuestionType, Platform

        question = PlatformQuestion(
            question_id="q-test",
            platform=Platform.METACULUS,
            title="Test question",
            question_type=QuestionType.BINARY,
        )

        mock_config = MagicMock()
        mock_config.primary_model.provider = "anthropic"
        mock_config.primary_model.model_id = "claude-3-5-sonnet"
        mock_config.primary_model.max_tokens = 1024
        mock_config.prompts_dir = "/nonexistent"

        mock_router = MagicMock()
        valid_json = json.dumps({
            "sub_questions": ["Will X happen?"],
            "reference_class": "historical base rate",
            "base_rate": 0.3,
            "base_rate_source": "test",
            "actor_analysis": {
                "actors_involved": False,
                "actors": [],
                "peripheral_actors": [],
                "probability_space_constraints": [],
            },
        })
        mock_router.call.return_value = (valid_json, 10, 10)

        warnings: list[str] = []

        with patch("src.pipeline.decomposer._load_prompt", return_value="system prompt"):
            result = decompose_question(
                question=question,
                config=mock_config,
                router=mock_router,
                warnings_out=warnings,
            )

        assert warnings == []
        assert result.sub_questions == ["Will X happen?"]

    def test_warnings_out_none_does_not_raise(self):
        """When warnings_out is None (default), parse failure doesn't raise."""
        from src.pipeline.decomposer import decompose_question
        from src.models import PlatformQuestion, QuestionType, Platform

        question = PlatformQuestion(
            question_id="q-test",
            platform=Platform.METACULUS,
            title="Test question",
            question_type=QuestionType.BINARY,
        )

        mock_config = MagicMock()
        mock_config.primary_model.provider = "anthropic"
        mock_config.primary_model.model_id = "claude-3-5-sonnet"
        mock_config.primary_model.max_tokens = 1024

        mock_router = MagicMock()
        mock_router.call.return_value = ("not valid json", 10, 10)

        with patch("src.pipeline.decomposer._load_prompt", return_value="system prompt"):
            result = decompose_question(
                question=question,
                config=mock_config,
                router=mock_router,
                # warnings_out omitted → defaults to None
            )

        assert result.sub_questions == []
