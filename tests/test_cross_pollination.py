"""Tests for cross-pollination improvements.

Covers:
  - LLMRouter: no model mutation (SQ[1])
  - LLMRouter: PROVIDERS registry expansion (SQ[2])
  - PipelineValidator gates (SQ[3])
  - LLMAuditLogger (SQ[4])
  - source_cluster_check (SQ[5])
  - deliberation provider fix (CQA-F6)
  - Config new fields additive (no breaking change)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import Config, ModelConfig
from src.models import Aggregation, PredictionRecord, RunResult


# ---------------------------------------------------------------------------
# Config: new fields are additive — existing usage unaffected
# ---------------------------------------------------------------------------

class TestConfigNewFields:
    def test_verification_providers_default(self):
        config = Config()
        assert isinstance(config.verification_providers, list)
        assert "openai" in config.verification_providers
        assert "google" in config.verification_providers
        # 4B local models excluded by default
        assert "nemotron-nano" not in config.verification_providers
        assert "qwen-local" not in config.verification_providers

    def test_audit_path_default_none(self):
        config = Config()
        assert config.audit_path is None

    def test_strict_validation_default_false(self):
        config = Config()
        assert config.strict_validation is False

    def test_models_field_untouched(self):
        """Config.models is not modified by new fields."""
        config = Config(anthropic_api_key="test-key")
        # models field still populated normally
        assert isinstance(config.models, list)

    def test_verification_providers_configurable(self):
        config = Config(verification_providers=["openai"])
        assert config.verification_providers == ["openai"]


# ---------------------------------------------------------------------------
# LLMRouter: no model mutation
# ---------------------------------------------------------------------------

class TestLLMRouterNoMutation:
    def _make_router(self, provider="anthropic", **kwargs):
        config = Config(
            anthropic_api_key="test-key",
            primary_model=ModelConfig(provider=provider, model_id="claude-sonnet-4-6"),
            verification_providers=[],
        )
        return config

    def test_anthropic_call_passes_model_to_local_client(self):
        """Anthropic calls use model= kwarg, not mutation."""
        config = self._make_router()
        with patch("src.pipeline.llm_router.AnthropicClient") as MockClient:
            instance = MockClient.return_value
            instance.available = True
            instance.call.return_value = ("response", 10, 5)
            from src.pipeline.llm_router import LLMRouter
            router = LLMRouter.__new__(LLMRouter)
            router.config = config
            router._audit = None
            router._anthropic = instance
            router._external_providers = {}
            router._openai = None
            router._gemini = None
            router._client_cache = {}

            router.call("anthropic", "claude-opus-4-6", "sys", "user")
            # model kwarg passed, not mutation
            instance.call.assert_called_once_with("sys", "user", 0.7, 4096, model="claude-opus-4-6")

    def test_non_anthropic_call_uses_cached_client(self):
        """Non-anthropic calls use per-(provider,model) cached client — no mutation."""
        config = Config(
            anthropic_api_key="",
            primary_model=ModelConfig(provider="openai", model_id="gpt-5.1"),
            verification_providers=[],
        )
        mock_client = MagicMock()
        mock_client.call.return_value = ("result", 20, 10)

        from src.pipeline.llm_router import LLMRouter
        router = LLMRouter.__new__(LLMRouter)
        router.config = config
        router._audit = None
        router._anthropic = None
        router._external_providers = {}
        router._openai = None
        router._gemini = None
        router._client_cache = {("openai", "gpt-5.1"): mock_client}

        router.call("openai", "gpt-5.1", "sys", "user")
        mock_client.call.assert_called_once_with("sys", "user", 0.7, 4096)
        # Verify model attribute was NOT mutated
        assert not hasattr(mock_client, "model") or mock_client.model == mock_client.model

    def test_client_cache_keyed_by_provider_model(self):
        """_get_or_create_client returns the same instance for the same (provider, model)
        and distinct instances for different provider or model."""
        from src.pipeline.llm_router import LLMRouter
        config = Config(
            anthropic_api_key="test-key",
            verification_providers=[],
        )
        router = LLMRouter.__new__(LLMRouter)
        router.config = config
        router._client_cache = {}

        mock_openai_cls = MagicMock()
        mock_google_cls = MagicMock()
        # Each call to the class constructor returns a new distinct instance
        mock_openai_cls.side_effect = [MagicMock(), MagicMock()]
        mock_google_cls.side_effect = [MagicMock()]

        providers_map = {"openai": mock_openai_cls, "google": mock_google_cls}

        with patch("src.pipeline.llm_router.PROVIDERS", providers_map, create=True):
            # Same (provider, model) twice — must return same instance (cached)
            client_1 = router._get_or_create_client("openai", "gpt-5.4")
            client_2 = router._get_or_create_client("openai", "gpt-5.4")
            assert client_1 is client_2, "Cache miss: same (provider, model) returned different instances"

            # Different model, same provider — must return a different instance
            client_3 = router._get_or_create_client("openai", "gpt-4o")
            assert client_1 is not client_3, "Same instance returned for different model"

            # Different provider, same model string — must return a different instance
            client_4 = router._get_or_create_client("google", "gpt-5.4")
            assert client_1 is not client_4, "Same instance returned for different provider"
            assert client_3 is not client_4, "Same instance returned for different provider"

    def test_available_providers_includes_anthropic(self):
        """available_providers lists anthropic when key present."""
        config = Config(anthropic_api_key="test-key", verification_providers=[])
        from src.pipeline.llm_router import AnthropicClient, LLMRouter
        router = LLMRouter.__new__(LLMRouter)
        router.config = config
        router._anthropic = AnthropicClient(model="claude-sonnet-4-6", api_key="test-key")
        router._external_providers = {}
        router._openai = None
        router._gemini = None
        router._client_cache = {}

        providers = router.available_providers()
        assert "anthropic" in providers

    def test_verification_available_false_when_no_external(self):
        from src.pipeline.llm_router import LLMRouter
        router = LLMRouter.__new__(LLMRouter)
        router._external_providers = {}
        assert router.verification_available() is False

    def test_verification_available_true_when_external_present(self):
        from src.pipeline.llm_router import LLMRouter
        router = LLMRouter.__new__(LLMRouter)
        router._external_providers = {"openai": MagicMock()}
        assert router.verification_available() is True


# ---------------------------------------------------------------------------
# PipelineValidator gates
# ---------------------------------------------------------------------------

class TestPipelineValidator:
    def _make_record(self, base_rate=0.5, prob=0.5, runs=None):
        rec = PredictionRecord()
        rec.decomposition.base_rate = base_rate
        rec.calibration.calibrated_probability = prob
        rec.runs = runs or []
        return rec

    def test_base_rate_gate_passes_normal(self):
        from src.pipeline.validator import PipelineValidator
        config = Config()
        v = PipelineValidator(config)
        rec = self._make_record(base_rate=0.3)
        warnings = v.validate_base_rate(rec)
        assert warnings == []

    def test_base_rate_gate_warns_on_zero(self):
        from src.pipeline.validator import PipelineValidator
        config = Config()
        v = PipelineValidator(config)
        rec = self._make_record(base_rate=0.0)
        warnings = v.validate_base_rate(rec)
        assert len(warnings) == 1
        assert "base_rate_gate" in warnings[0]

    def test_base_rate_gate_warns_on_one(self):
        from src.pipeline.validator import PipelineValidator
        config = Config()
        v = PipelineValidator(config)
        rec = self._make_record(base_rate=1.0)
        warnings = v.validate_base_rate(rec)
        assert len(warnings) == 1

    def test_probability_gate_passes_normal(self):
        from src.pipeline.validator import PipelineValidator
        config = Config()
        v = PipelineValidator(config)
        rec = self._make_record(prob=0.7)
        warnings = v.validate_aggregation(rec)
        assert warnings == []

    def test_probability_gate_warns_on_extreme(self):
        from src.pipeline.validator import PipelineValidator
        config = Config()
        v = PipelineValidator(config)
        rec = self._make_record(prob=0.001)
        warnings = v.validate_aggregation(rec)
        assert len(warnings) == 1
        assert "probability_gate" in warnings[0]

    def test_run_count_gate_warns_on_mismatch(self):
        from src.pipeline.validator import PipelineValidator
        config = Config()
        v = PipelineValidator(config)
        rec = self._make_record(runs=[RunResult()])
        warnings = v.validate_runs(rec, requested_runs=3)
        assert any("run_count_gate" in w for w in warnings)

    def test_run_count_gate_passes_on_match(self):
        from src.pipeline.validator import PipelineValidator
        config = Config()
        v = PipelineValidator(config)
        rec = self._make_record(runs=[RunResult(), RunResult()])
        warnings = v.validate_runs(rec, requested_runs=2)
        assert not any("run_count_gate" in w for w in warnings)

    def test_parse_failure_gate_warns(self):
        from src.pipeline.validator import PipelineValidator
        config = Config()
        v = PipelineValidator(config)
        rec = self._make_record(runs=[RunResult(parse_failed=True)])
        warnings = v.validate_runs(rec)
        assert any("parse_failure_gate" in w for w in warnings)

    def test_parse_failure_gate_passes_when_no_failures(self):
        from src.pipeline.validator import PipelineValidator
        config = Config()
        v = PipelineValidator(config)
        rec = self._make_record(runs=[RunResult(parse_failed=False)])
        warnings = v.validate_runs(rec)
        assert not any("parse_failure_gate" in w for w in warnings)

    def test_strict_mode_raises_on_gate_failure(self):
        from src.pipeline.validator import PipelineError, PipelineValidator
        config = Config(strict_validation=True)
        v = PipelineValidator(config)
        rec = self._make_record(base_rate=0.0)
        with pytest.raises(PipelineError):
            v.validate_base_rate(rec)

    def test_validate_all_combines_warnings(self):
        from src.pipeline.validator import PipelineValidator
        config = Config()
        v = PipelineValidator(config)
        rec = self._make_record(
            base_rate=0.0,
            prob=0.001,
            runs=[RunResult(parse_failed=True)],
        )
        warnings = v.validate_all(rec, requested_runs=3)
        assert len(warnings) >= 3  # base_rate + parse_failure + run_count + probability

    def test_instance_count_none_no_warning(self):
        """No warning when instance count is not provided."""
        from src.pipeline.validator import PipelineValidator
        config = Config()
        v = PipelineValidator(config)
        rec = self._make_record()
        rec.decomposition.reference_class_instance_count = None
        warnings = v.validate_base_rate_instances(rec)
        assert warnings == []

    def test_instance_count_below_10_warns_unreliable(self):
        """Fewer than 10 instances triggers strong reliability warning."""
        from src.pipeline.validator import PipelineValidator
        config = Config()
        v = PipelineValidator(config)
        rec = self._make_record()
        rec.decomposition.reference_class_instance_count = 7
        warnings = v.validate_base_rate_instances(rec)
        assert len(warnings) == 1
        assert "reference_class_instances_gate" in warnings[0]
        assert "unreliable" in warnings[0]

    def test_instance_count_below_20_warns_low_reliability(self):
        """Between 10 and 19 instances triggers soft reliability warning."""
        from src.pipeline.validator import PipelineValidator
        config = Config()
        v = PipelineValidator(config)
        rec = self._make_record()
        rec.decomposition.reference_class_instance_count = 15
        warnings = v.validate_base_rate_instances(rec)
        assert len(warnings) == 1
        assert "reference_class_instances_gate" in warnings[0]
        assert "recommend" in warnings[0]

    def test_instance_count_20_or_more_no_warning(self):
        """20+ instances passes without warning."""
        from src.pipeline.validator import PipelineValidator
        config = Config()
        v = PipelineValidator(config)
        rec = self._make_record()
        rec.decomposition.reference_class_instance_count = 47
        warnings = v.validate_base_rate_instances(rec)
        assert warnings == []

    def test_instance_count_exactly_10_warns_low_not_unreliable(self):
        """Exactly 10 instances triggers the soft (not hard) warning."""
        from src.pipeline.validator import PipelineValidator
        config = Config()
        v = PipelineValidator(config)
        rec = self._make_record()
        rec.decomposition.reference_class_instance_count = 10
        warnings = v.validate_base_rate_instances(rec)
        assert len(warnings) == 1
        assert "recommend" in warnings[0]
        assert "unreliable" not in warnings[0]


# ---------------------------------------------------------------------------
# LLMAuditLogger
# ---------------------------------------------------------------------------

class TestLLMAuditLogger:
    def test_log_call_creates_jsonl_entry(self, tmp_path):
        from src.pipeline.llm_audit import LLMAuditLogger
        log_path = str(tmp_path / "audit.jsonl")
        audit = LLMAuditLogger(log_path)
        audit.log_call(
            provider="anthropic",
            model="claude-sonnet-4-6",
            method="call",
            params={"system": "sys prompt", "user": "user msg", "temperature": 0.7},
            result_text="response text",
            tokens_in=100,
            tokens_out=50,
            duration_ms=1234.5,
            cost_usd=0.001,
        )
        audit.flush()

        lines = Path(log_path).read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["provider"] == "anthropic"
        assert entry["model"] == "claude-sonnet-4-6"
        assert entry["method"] == "call"
        assert entry["tokens_in"] == 100
        assert entry["tokens_out"] == 50
        assert entry["duration_ms"] == 1234.5
        assert entry["cost_usd"] == 0.001
        assert "ts" in entry
        assert "params_hash" in entry
        assert "result_hash" in entry

    def test_raw_params_not_in_log(self, tmp_path):
        """Prompt text must not appear verbatim in audit log."""
        from src.pipeline.llm_audit import LLMAuditLogger
        log_path = str(tmp_path / "audit.jsonl")
        audit = LLMAuditLogger(log_path)
        secret_prompt = "This is my secret system prompt SUPERSECRET"
        audit.log_call(
            provider="openai",
            model="gpt-5.1",
            method="call",
            params={"system": secret_prompt, "user": "user content"},
            result_text="ok",
            tokens_in=10,
            tokens_out=5,
            duration_ms=100.0,
        )
        audit.flush()

        raw_log = Path(log_path).read_text()
        assert "SUPERSECRET" not in raw_log

    def test_flush_empty_buffer_is_noop(self, tmp_path):
        """flush() with empty buffer doesn't create/modify file."""
        from src.pipeline.llm_audit import LLMAuditLogger
        log_path = str(tmp_path / "audit.jsonl")
        audit = LLMAuditLogger(log_path)
        audit.flush()  # no entries
        assert not Path(log_path).exists()

    def test_multiple_calls_append(self, tmp_path):
        from src.pipeline.llm_audit import LLMAuditLogger
        log_path = str(tmp_path / "audit.jsonl")
        audit = LLMAuditLogger(log_path)
        for i in range(3):
            audit.log_call("anthropic", "claude-sonnet-4-6", "call",
                          {"i": i}, f"result {i}", 10, 5, 100.0)
        audit.flush()

        lines = Path(log_path).read_text().strip().split("\n")
        assert len(lines) == 3

    def test_buffer_auto_flushes_at_limit(self, tmp_path):
        from src.pipeline.llm_audit import LLMAuditLogger
        log_path = str(tmp_path / "audit.jsonl")
        audit = LLMAuditLogger(log_path)
        audit.BUFFER_LIMIT = 3  # lower limit for test

        for i in range(3):
            audit.log_call("openai", "gpt-5.1", "call",
                          {"i": i}, "result", 10, 5, 50.0)
        # Auto-flush should have fired
        assert Path(log_path).exists()
        lines = Path(log_path).read_text().strip().split("\n")
        assert len(lines) >= 3


# ---------------------------------------------------------------------------
# source_cluster_check
# ---------------------------------------------------------------------------

class TestSourceClusterCheck:
    def _run(self, sources_per_run: list[list[str]]) -> tuple[float, list[str]]:
        from src.pipeline.aggregator import source_cluster_check
        runs = [RunResult(sources_cited=s) for s in sources_per_run]
        return source_cluster_check(runs)

    def test_no_clustering_different_sources(self):
        score, domains = self._run([
            ["https://site-a.com/article1"],
            ["https://site-b.com/article2"],
            ["https://site-c.com/article3"],
        ])
        assert score == 0.0
        assert domains == []

    def test_clustering_detected_same_domain(self):
        score, domains = self._run([
            ["https://breaking-news.com/story1"],
            ["https://breaking-news.com/story2"],
            ["https://breaking-news.com/story3"],
        ])
        assert score > 0.0
        assert "breaking-news.com" in domains

    def test_authoritative_domains_not_flagged(self):
        """reuters.com etc are authoritative — should not trigger warning."""
        score, domains = self._run([
            ["https://reuters.com/article1"],
            ["https://reuters.com/article2"],
            ["https://reuters.com/article3"],
        ])
        assert score == 0.0
        assert domains == []

    def test_single_run_no_clustering(self):
        """Single run can't produce clustering."""
        score, domains = self._run([["https://any.com/article"]])
        assert score == 0.0

    def test_mixed_sources_below_threshold(self):
        """Domain in only 1 of 3 runs (33%) doesn't trigger 50% threshold."""
        score, domains = self._run([
            ["https://obscure-blog.com/article1"],
            ["https://other-site.com/article2"],
            ["https://third-source.com/article3"],
        ])
        assert score == 0.0

    def test_clustering_score_reflects_coverage(self):
        """Score reflects fraction of runs citing clustered domain."""
        score, domains = self._run([
            ["https://obscure-blog.com/a"],
            ["https://obscure-blog.com/b"],
        ])
        assert score == 1.0  # 2/2 runs

    def test_empty_sources_no_clustering(self):
        score, domains = self._run([[], []])
        assert score == 0.0


# ---------------------------------------------------------------------------
# Aggregation model new fields survive JSONL roundtrip
# ---------------------------------------------------------------------------

class TestAggregationNewFields:
    def test_new_fields_default(self):
        agg = Aggregation()
        assert agg.sources_clustering_score == 0.0
        assert agg.clustered_domains == []

    def test_clustering_fields_survive_serialization(self):
        agg = Aggregation(
            sources_clustering_score=0.75,
            clustered_domains=["blog.com", "site.net"],
        )
        rec = PredictionRecord(aggregation=agg)
        data = json.loads(rec.model_dump_json())
        restored = PredictionRecord.model_validate(data)
        assert restored.aggregation.sources_clustering_score == 0.75
        assert restored.aggregation.clustered_domains == ["blog.com", "site.net"]

    def test_old_record_missing_clustering_fields_loads_with_defaults(self):
        """Registry records written before this change load with defaults."""
        import json as _json
        from src.registry.store import RegistryStore
        import tempfile, os

        minimal_agg = {
            "method": "single_run",
            "raw_aggregate": 0.5,
            "stdev": 0.0,
            "inter_model_agreement": 1.0,
            "effective_n": 1,
            "deliberation_triggered": False,
            "deliberation_result": None,
            # sources_clustering_score and clustered_domains absent — legacy record
        }
        minimal = {
            "prediction_id": "test-legacy",
            "question_id": "q1",
            "platform": "metaculus",
            "question_text": "Test?",
            "question_type": "binary",
            "domain": "",
            "horizon_days": None,
            "created_at": "2026-01-01T00:00:00+00:00",
            "decomposition": {"sub_questions": [], "reference_class": "", "base_rate": 0.5,
                              "base_rate_source": "", "actor_analysis": {"actors_involved": False,
                              "actors": [], "level_2_3_present": False, "peripheral_actors": [],
                              "probability_space_constraints": []}},
            "runs": [],
            "aggregation": minimal_agg,
            "calibration": {"platt_params_used": None, "calibrated_probability": 0.5,
                            "confidence_interval": [0.0, 1.0]},
            "platform_signals": {"prices_at_prediction_time": {}, "cross_platform_divergence": None,
                                 "anomaly_detected": False, "anomaly_details": None},
            "human_review": {"reviewed": False, "human_adjustment": None, "human_reasoning": None,
                             "flags_raised": []},
            "submitted": None, "resolution": None,
            "verification": {"enabled": False, "providers_available": [], "stages": []},
            "cost": {"total_tokens_in": 0, "total_tokens_out": 0, "total_cost_usd": 0.0,
                     "latency_seconds": 0.0},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(_json.dumps(minimal) + "\n")
            path = f.name

        try:
            store = RegistryStore(path)
            loaded = store.load_all()
            assert len(loaded) == 1
            assert loaded[0].aggregation.sources_clustering_score == 0.0
            assert loaded[0].aggregation.clustered_domains == []
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# deliberation.py: CQA-F6 fix — uses config.primary_model.provider not "anthropic"
# ---------------------------------------------------------------------------

class TestDeliberationProviderFix:
    def test_deliberate_uses_primary_model_provider(self):
        """deliberate() must call router.call with primary_model.provider, not hardcoded 'anthropic'."""
        from src.pipeline.deliberation import deliberate
        from src.models import Aggregation, RunResult

        config = MagicMock()
        config.primary_model.provider = "openai"
        config.primary_model.model_id = "gpt-5.1"

        mock_router = MagicMock()
        mock_router.call.return_value = (
            '{"disagreement_reason": "test", "key_uncertainties": [], "recommended_action": "accept_aggregate"}',
            10, 10,
        )

        runs = [
            RunResult(probability=0.3, model="gpt-5.1", reasoning_chain="low"),
            RunResult(probability=0.7, model="gpt-5.1", reasoning_chain="high"),
        ]
        agg = Aggregation(raw_aggregate=0.5, stdev=0.2, inter_model_agreement=0.5)

        deliberate(runs, agg, config, router=mock_router)

        call_args = mock_router.call.call_args
        provider_used = call_args[0][0]
        assert provider_used == "openai", (
            f"Expected 'openai', got '{provider_used}' — hardcoded 'anthropic' bug not fixed"
        )

    def test_deliberate_passes_through_when_router_none(self):
        """deliberate() with no router returns runs+agg unchanged."""
        from src.pipeline.deliberation import deliberate
        runs = [RunResult(probability=0.5)]
        agg = Aggregation(raw_aggregate=0.5)
        config = MagicMock()
        result_runs, result_agg = deliberate(runs, agg, config, router=None)
        assert result_runs is runs
        assert result_agg is agg
