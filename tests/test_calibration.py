"""Tests for the calibration module."""

import pytest

from src.pipeline.calibration import (
    PlattParams,
    apply_hedging_correction,
    calibrate,
    fit_platt,
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


class TestCalibrate:
    def test_uses_prior_correction_when_few_resolved(self):
        config = Config()
        prob, params_dict = calibrate(0.6, config, n_resolved=10)
        # Should apply hedging correction
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
