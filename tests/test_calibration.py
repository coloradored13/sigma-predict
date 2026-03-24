"""Tests for the calibration module."""

import pytest

from src.pipeline.calibration import (
    PlattParams,
    apply_hedging_correction,
    apply_hedging_correction_scaled,
    calibrate,
    fit_isotonic_logit,
    fit_platt,
    isotonic_transform,
    platt_transform,
)
from src.config import Config


class TestHedgingCorrection:
    def test_pushes_high_higher(self):
        result = apply_hedging_correction(0.7, 0.04)
        assert result == pytest.approx(0.74)

    def test_pushes_low_lower(self):
        result = apply_hedging_correction(0.3, 0.04)
        assert result == pytest.approx(0.26)

    def test_no_change_at_center(self):
        result = apply_hedging_correction(0.5, 0.04)
        assert result == 0.5

    def test_clamps_high(self):
        result = apply_hedging_correction(0.98, 0.05)
        assert result == 0.99

    def test_clamps_low(self):
        result = apply_hedging_correction(0.02, 0.05)
        assert result == 0.01


class TestPlattTransform:
    def test_identity_params(self):
        # a=1, b=0 should be roughly identity
        params = PlattParams(a=1.0, b=0.0)
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert platt_transform(p, params) == pytest.approx(p, abs=0.01)

    def test_shifts_up(self):
        params = PlattParams(a=1.0, b=0.5)
        result = platt_transform(0.5, params)
        assert result > 0.5

    def test_shifts_down(self):
        params = PlattParams(a=1.0, b=-0.5)
        result = platt_transform(0.5, params)
        assert result < 0.5

    def test_bounds(self):
        params = PlattParams(a=5.0, b=10.0)
        result = platt_transform(0.99, params)
        assert 0.01 <= result <= 0.99


class TestFitPlatt:
    def test_well_calibrated_data(self):
        # Already calibrated predictions should produce near-identity params
        preds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 10
        # Outcomes that roughly match predictions
        import random
        random.seed(42)
        outcomes = [1.0 if random.random() < p else 0.0 for p in preds]

        params = fit_platt(preds, outcomes)
        # a should be near 1.0, b near 0.0
        assert 0.5 < params.a < 2.0
        assert -1.0 < params.b < 1.0

    def test_too_few_samples(self):
        params = fit_platt([0.5, 0.5], [1.0, 0.0])
        # Should return identity
        assert params.a == 1.0
        assert params.b == 0.0


class TestHedgingCorrectionScaled:
    def test_no_shift_at_center(self):
        # At 0.5, distance=0, effective_shift=0 → no change
        result = apply_hedging_correction_scaled(0.5, 0.04)
        assert result == 0.5

    def test_pushes_high_higher(self):
        # p=0.75, distance=0.25, effective_shift=0.04*2*0.25=0.02 → 0.77
        result = apply_hedging_correction_scaled(0.75, 0.04)
        assert result == pytest.approx(0.77)

    def test_pushes_low_lower(self):
        # p=0.25, distance=0.25, effective_shift=0.02 → 0.23
        result = apply_hedging_correction_scaled(0.25, 0.04)
        assert result == pytest.approx(0.23)

    def test_larger_distance_larger_shift(self):
        # p=0.45 (distance=0.05): effective=0.04*2*0.05=0.004, result=0.446, shift=0.004
        shift_near = abs(0.45 - apply_hedging_correction_scaled(0.45, 0.04))
        # p=0.1 (distance=0.4): effective=0.04*2*0.4=0.032, result=0.068, shift=0.032
        shift_far = abs(0.1 - apply_hedging_correction_scaled(0.1, 0.04))
        assert shift_far > shift_near  # bigger distance → bigger shift (absolute)

    def test_clamps_high(self):
        result = apply_hedging_correction_scaled(0.999, 0.5)
        assert result == 0.99

    def test_clamps_low(self):
        result = apply_hedging_correction_scaled(0.001, 0.5)
        assert result == 0.01


class TestIsotonicTransform:
    def _breakpoints(self):
        # Simple monotone map: logit(-2)→0.2, logit(0)→0.5, logit(2)→0.8
        import math
        return [
            (math.log(0.12 / 0.88), 0.2),
            (0.0, 0.5),
            (math.log(0.88 / 0.12), 0.8),
        ]

    def test_below_range_clamps(self):
        bps = self._breakpoints()
        result = isotonic_transform(0.001, bps)
        assert result == pytest.approx(0.2, abs=0.01)

    def test_above_range_clamps(self):
        bps = self._breakpoints()
        result = isotonic_transform(0.999, bps)
        assert result == pytest.approx(0.8, abs=0.01)

    def test_midpoint_interpolation(self):
        bps = self._breakpoints()
        # p=0.5 → logit=0 → should return 0.5
        result = isotonic_transform(0.5, bps)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_result_in_bounds(self):
        bps = self._breakpoints()
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = isotonic_transform(p, bps)
            assert 0.01 <= result <= 0.99

    def test_empty_breakpoints_returns_input(self):
        result = isotonic_transform(0.6, [])
        assert result == 0.6


class TestCalibrate:
    def test_phase0_uses_scaled_hedging(self):
        config = Config()
        # n_resolved=5 < min_resolved_for_isotonic=20 → Phase 0
        prob, params_dict = calibrate(0.6, config, n_resolved=5)
        assert prob > 0.6  # pushed further from 0.5
        assert params_dict is None

    def test_phase1_uses_isotonic(self):
        config = Config()
        # Build enough isotonic data
        isotonic_data = [(0.3, 0.0)] * 5 + [(0.7, 1.0)] * 5
        # n_resolved=25: 20<=25<50 → Phase 1
        prob, params_dict = calibrate(
            0.7, config, n_resolved=25, isotonic_data=isotonic_data
        )
        assert 0.01 <= prob <= 0.99
        assert params_dict is None

    def test_phase1_fallback_when_insufficient_isotonic(self):
        config = Config()
        # Isotonic data with only 3 pairs (< 5 min) → falls back to scaled hedging
        prob, params_dict = calibrate(
            0.7, config, n_resolved=25, isotonic_data=[(0.6, 1.0), (0.4, 0.0), (0.5, 1.0)]
        )
        assert prob > 0.7  # scaled hedging pushes higher
        assert params_dict is None

    def test_phase2_uses_platt(self):
        config = Config()
        platt = PlattParams(a=1.0, b=0.2)
        prob, params_dict = calibrate(
            0.5, config, platt_params=platt, n_resolved=100
        )
        assert prob > 0.5  # b=0.2 should shift up
        assert params_dict is not None
        assert params_dict["a"] == 1.0

    def test_uses_prior_correction_when_few_resolved(self):
        config = Config()
        prob, params_dict = calibrate(0.6, config, n_resolved=10)
        # Should apply scaled hedging correction
        assert prob > 0.6
        assert params_dict is None

    def test_uses_platt_when_enough_resolved(self):
        config = Config()
        platt = PlattParams(a=1.0, b=0.2)
        prob, params_dict = calibrate(
            0.5, config, platt_params=platt, n_resolved=100
        )
        assert prob > 0.5  # b=0.2 should shift up
        assert params_dict is not None
        assert params_dict["a"] == 1.0
