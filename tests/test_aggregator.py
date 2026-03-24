"""Tests for the statistical aggregation module."""

import math

import numpy as np
import pytest

from src.models import RunResult
from src.pipeline.aggregator import aggregate_runs, _extremize


def _make_run(probability: float, model: str = "test-model") -> RunResult:
    """Helper to create a RunResult with just a probability."""
    return RunResult(probability=probability, model=model)


class TestAggregateRuns:
    def test_empty_runs(self):
        result = aggregate_runs([])
        assert result.raw_aggregate == 0.5
        assert result.effective_n <= 1

    def test_single_run(self):
        result = aggregate_runs([_make_run(0.7)])
        assert result.method == "single_run"
        assert result.raw_aggregate == 0.7
        assert result.stdev == 0.0
        assert result.effective_n == 1

    def test_two_identical_runs(self):
        result = aggregate_runs([_make_run(0.6), _make_run(0.6)])
        assert result.stdev == 0.0
        assert result.inter_model_agreement == 1.0

    def test_high_agreement(self):
        runs = [_make_run(p) for p in [0.60, 0.62, 0.58, 0.61, 0.59]]
        result = aggregate_runs(runs)
        assert result.stdev < 0.05
        assert result.inter_model_agreement > 0.9

    def test_low_agreement(self):
        runs = [_make_run(p) for p in [0.2, 0.5, 0.8, 0.3, 0.9]]
        result = aggregate_runs(runs)
        assert result.stdev > 0.2
        assert result.inter_model_agreement < 0.5

    def test_extremized_mean_pushes_from_center(self):
        runs = [_make_run(p) for p in [0.6, 0.65, 0.62, 0.58, 0.64]]
        result = aggregate_runs(runs, method="extremized_trimmed_mean")
        mean_val = np.mean([0.6, 0.65, 0.62, 0.58, 0.64])
        # Extremized should push further from 0.5 than the raw mean
        assert result.raw_aggregate > mean_val

    def test_extremized_mean_below_center(self):
        runs = [_make_run(p) for p in [0.3, 0.35, 0.28, 0.32, 0.31]]
        result = aggregate_runs(runs, method="extremized_trimmed_mean")
        mean_val = np.mean([0.3, 0.35, 0.28, 0.32, 0.31])
        # For p < 0.5, extremizing should push lower
        assert result.raw_aggregate < mean_val

    def test_probability_bounds(self):
        runs = [_make_run(0.99), _make_run(0.98), _make_run(0.97)]
        result = aggregate_runs(runs, extremization_factor=3.0)
        assert 0.01 <= result.raw_aggregate <= 0.99

    def test_method_mean(self):
        runs = [_make_run(p) for p in [0.3, 0.5, 0.7]]
        result = aggregate_runs(runs, method="mean")
        assert abs(result.raw_aggregate - 0.5) < 0.01

    def test_method_median(self):
        runs = [_make_run(p) for p in [0.2, 0.5, 0.9]]
        result = aggregate_runs(runs, method="median")
        assert abs(result.raw_aggregate - 0.5) < 0.01


class TestNDistinctProvidersGate:
    def test_single_provider_no_extremization(self):
        # With n_distinct_providers=1, extremized_trimmed_mean should NOT extremize
        # Use values clearly above 0.5 so trimmed_mean is > 0.5
        runs = [_make_run(p) for p in [0.65, 0.67, 0.63, 0.66, 0.64]]
        result_single = aggregate_runs(
            runs, method="extremized_trimmed_mean", n_distinct_providers=1
        )
        result_multi = aggregate_runs(
            runs, method="extremized_trimmed_mean", n_distinct_providers=2
        )
        # Single provider: raw_aggregate = trimmed_mean (no extremization)
        # Multi provider: raw_aggregate is extremized (pushed further from 0.5)
        assert result_multi.raw_aggregate > result_single.raw_aggregate

    def test_two_providers_extremizes(self):
        runs = [_make_run(p) for p in [0.6, 0.65, 0.62, 0.58, 0.64]]
        result = aggregate_runs(
            runs, method="extremized_trimmed_mean", n_distinct_providers=2
        )
        mean_val = float(sum([0.6, 0.65, 0.62, 0.58, 0.64]) / 5)
        # With extremization, aggregate should be pushed further from 0.5
        assert result.raw_aggregate > mean_val

    def test_single_provider_aggregate_equals_trimmed_mean(self):
        # n=5, trim_fraction=0.1 → trim_count=max(1,0)=1 → trimmed 3 middle values
        runs = [_make_run(p) for p in [0.5, 0.6, 0.7, 0.8, 0.9]]
        result = aggregate_runs(
            runs, method="extremized_trimmed_mean",
            trim_fraction=0.1, n_distinct_providers=1
        )
        # trimmed: [0.6, 0.7, 0.8] → mean = 0.7
        assert result.raw_aggregate == pytest.approx(0.7, abs=0.01)

    def test_default_n_distinct_providers_is_one(self):
        # Default (n_distinct_providers=1) should NOT extremize
        runs = [_make_run(p) for p in [0.65, 0.67, 0.63, 0.66, 0.64]]
        result_default = aggregate_runs(runs, method="extremized_trimmed_mean")
        result_explicit_1 = aggregate_runs(
            runs, method="extremized_trimmed_mean", n_distinct_providers=1
        )
        assert result_default.raw_aggregate == pytest.approx(
            result_explicit_1.raw_aggregate, abs=1e-9
        )


class TestExtremize:
    def test_identity_at_center(self):
        # At p=0.5, extremization should have no effect
        assert _extremize(0.5, 1.0) == pytest.approx(0.5)
        assert _extremize(0.5, 2.0) == pytest.approx(0.5)

    def test_pushes_high_higher(self):
        assert _extremize(0.7, 1.5) > 0.7

    def test_pushes_low_lower(self):
        assert _extremize(0.3, 1.5) < 0.3

    def test_factor_one_is_identity(self):
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert _extremize(p, 1.0) == pytest.approx(p, abs=1e-10)

    def test_extreme_bounds(self):
        # Should not go below 0.01 or above 0.99
        result = _extremize(0.01, 5.0)
        assert result == 0.01  # clamped at boundary
