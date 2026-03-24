"""Calibration module: Platt scaling and prior corrections.

Phase 0 (N<20):  apply scaled hedging correction (distance-weighted).
Phase 1 (N=20-49): isotonic regression on logit scale.
Phase 2 (N>=50): Platt scaling on historical data.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import os
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from src.config import Config

logger = logging.getLogger(__name__)


@dataclass
class PlattParams:
    a: float  # slope
    b: float  # intercept

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, d: dict) -> PlattParams:
        return cls(a=d["a"], b=d["b"])


def calibrate(
    probability: float,
    config: Config,
    platt_params: PlattParams | None = None,
    n_resolved: int = 0,
    isotonic_data: list[tuple[float, float]] | None = None,
) -> tuple[float, dict | None]:
    """Apply calibration to a raw probability estimate.

    3-phase gate:
      Phase 2 (N>=50): Platt scaling.
      Phase 1 (N=20-49): isotonic regression on logit scale (if data available).
      Phase 0 (N<20): scaled hedging correction.

    Args:
        probability: Raw aggregate probability
        config: Pipeline config
        platt_params: Fitted Platt parameters (if available)
        n_resolved: Number of resolved predictions in dataset
        isotonic_data: List of (predicted, outcome) pairs for isotonic fitting

    Returns:
        tuple of (calibrated_probability, platt_params_dict_or_None)
    """
    if n_resolved >= config.calibration.min_resolved_for_platt and platt_params:
        # Phase 2: Platt scaling
        calibrated = platt_transform(probability, platt_params)
        logger.info(
            "Platt calibration: %.3f -> %.3f (a=%.3f, b=%.3f)",
            probability, calibrated, platt_params.a, platt_params.b,
        )
        return calibrated, platt_params.to_dict()
    elif n_resolved >= config.calibration.min_resolved_for_isotonic:
        # Phase 1: isotonic regression
        if isotonic_data and len(isotonic_data) >= 5:
            preds, outs = zip(*isotonic_data)
            breakpoints = fit_isotonic_logit(list(preds), list(outs))
            calibrated = isotonic_transform(probability, breakpoints)
            logger.info(
                "Isotonic calibration: %.3f -> %.3f (n_resolved=%d)",
                probability, calibrated, n_resolved,
            )
            return calibrated, None
        # Fall back to scaled hedging if not enough isotonic data
        calibrated = apply_hedging_correction_scaled(
            probability, config.calibration.prior_hedging_shift
        )
        logger.info(
            "Scaled hedging (isotonic fallback): %.3f -> %.3f (n_resolved=%d)",
            probability, calibrated, n_resolved,
        )
        return calibrated, None
    else:
        # Phase 0: scaled hedging correction
        calibrated = apply_hedging_correction_scaled(
            probability, config.calibration.prior_hedging_shift
        )
        logger.info(
            "Scaled hedging correction: %.3f -> %.3f (shift=%.3f, n_resolved=%d/%d for isotonic)",
            probability, calibrated, config.calibration.prior_hedging_shift,
            n_resolved, config.calibration.min_resolved_for_isotonic,
        )
        return calibrated, None


def apply_hedging_correction(probability: float, shift: float) -> float:
    """Shift probability away from 0.5 to correct for LLM hedging bias.

    LLMs tend to hedge toward 50%, so we push estimates away from 0.5.
    """
    if probability > 0.5:
        corrected = min(0.99, probability + shift)
    elif probability < 0.5:
        corrected = max(0.01, probability - shift)
    else:
        corrected = probability
    return corrected


def apply_hedging_correction_scaled(probability: float, base_shift: float = 0.04) -> float:
    """Distance-weighted hedging correction: shift scales with distance from 0.5.

    Probabilities further from 0.5 receive a larger absolute shift, which
    avoids over-correcting near-center estimates while still addressing
    confident-but-hedged predictions.
    """
    distance = abs(probability - 0.5)
    effective_shift = base_shift * 2 * distance
    if probability > 0.5:
        return min(0.99, probability + effective_shift)
    elif probability < 0.5:
        return max(0.01, probability - effective_shift)
    return probability


def fit_isotonic_logit(
    predictions: list[float], outcomes: list[float]
) -> list[tuple[float, float]]:
    """Fit isotonic regression on logit-transformed predictions.

    Returns a sorted list of (logit_x, prob_y) breakpoints for interpolation.
    Caps the number of breakpoints at max(3, N//5) to avoid overfitting.
    """
    from scipy.optimize import isotonic_regression as _isotonic

    n = len(predictions)
    preds = np.clip(np.array(predictions, dtype=float), 0.001, 0.999)
    logits = np.log(preds / (1.0 - preds))
    acts = np.array(outcomes, dtype=float)

    # Sort by logit value
    order = np.argsort(logits)
    logits_sorted = logits[order]
    acts_sorted = acts[order]

    # Isotonic regression (pool adjacent violators, minimises MSE)
    iso_probs = _isotonic(acts_sorted).x

    # Cap breakpoints
    max_bps = max(3, n // 5)
    if len(iso_probs) > max_bps:
        # Subsample uniformly
        indices = np.round(np.linspace(0, len(iso_probs) - 1, max_bps)).astype(int)
        logits_bp = logits_sorted[indices]
        probs_bp = iso_probs[indices]
    else:
        logits_bp = logits_sorted
        probs_bp = iso_probs

    # Return as sorted list of (logit_x, prob_y) tuples
    breakpoints = sorted(zip(logits_bp.tolist(), probs_bp.tolist()), key=lambda t: t[0])
    return breakpoints


def isotonic_transform(
    probability: float, isotonic_data: list[tuple[float, float]]
) -> float:
    """Apply isotonic calibration via linear interpolation on the logit scale.

    Args:
        probability: Raw probability to calibrate.
        isotonic_data: Sorted (logit_x, prob_y) breakpoints from fit_isotonic_logit.

    Returns:
        Calibrated probability clamped to [0.01, 0.99].
    """
    if not isotonic_data:
        return probability

    p = max(0.001, min(0.999, probability))
    logit_p = math.log(p / (1.0 - p))

    xs = [t[0] for t in isotonic_data]
    ys = [t[1] for t in isotonic_data]

    # Clamp to breakpoint range
    if logit_p <= xs[0]:
        result = ys[0]
    elif logit_p >= xs[-1]:
        result = ys[-1]
    else:
        # Linear interpolation between adjacent breakpoints
        idx = bisect.bisect_right(xs, logit_p) - 1
        x0, x1 = xs[idx], xs[idx + 1]
        y0, y1 = ys[idx], ys[idx + 1]
        t = (logit_p - x0) / (x1 - x0) if x1 != x0 else 0.0
        result = y0 + t * (y1 - y0)

    return max(0.01, min(0.99, result))


def platt_transform(probability: float, params: PlattParams) -> float:
    """Apply Platt scaling: P_cal = 1 / (1 + exp(-(a * logit(p) + b)))."""
    # Clamp to avoid log(0)
    p = max(0.001, min(0.999, probability))
    logit_p = math.log(p / (1.0 - p))
    exponent = -(params.a * logit_p + params.b)
    calibrated = 1.0 / (1.0 + math.exp(exponent))
    return max(0.01, min(0.99, calibrated))


def fit_platt(
    predictions: list[float],
    outcomes: list[float],
) -> PlattParams:
    """Fit Platt scaling parameters on historical (predicted, actual) pairs.

    Uses maximum likelihood estimation for logistic regression.

    Args:
        predictions: Raw predicted probabilities
        outcomes: Actual outcomes (0 or 1 for binary)

    Returns:
        Fitted PlattParams
    """
    if len(predictions) < 10:
        logger.warning("Too few data points (%d) for Platt fitting", len(predictions))
        return PlattParams(a=1.0, b=0.0)  # identity transform

    preds = np.array(predictions)
    acts = np.array(outcomes)

    # Clamp predictions to avoid log(0)
    preds = np.clip(preds, 0.001, 0.999)

    # Convert to logits
    logits = np.log(preds / (1.0 - preds))

    # Negative log-likelihood for logistic regression
    def neg_log_likelihood(params):
        a, b = params
        z = a * logits + b
        # Numerically stable log-likelihood
        ll = acts * z - np.log1p(np.exp(z))
        # Handle large z values
        large = z > 20
        ll[large] = acts[large] * z[large] - z[large]
        return -np.sum(ll)

    # Optimize
    result = minimize(
        neg_log_likelihood,
        x0=[1.0, 0.0],
        method="Nelder-Mead",
        options={"maxiter": 1000},
    )

    a, b = result.x
    logger.info("Platt parameters fitted: a=%.4f, b=%.4f", a, b)

    return PlattParams(a=float(a), b=float(b))


def save_platt_params(params: PlattParams, config: Config, label: str = "global") -> None:
    """Save Platt parameters to disk."""
    path = os.path.join(config.calibration_dir, f"platt_{label}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(params.to_dict(), f)


def load_platt_params(config: Config, label: str = "global") -> PlattParams | None:
    """Load Platt parameters from disk."""
    path = os.path.join(config.calibration_dir, f"platt_{label}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return PlattParams.from_dict(data)
