"""Calibration module: Platt scaling and prior corrections.

Until N>=50 resolved predictions: apply prior correction (LLM hedging bias).
After N>=50: fit Platt scaling parameters on historical data.
"""

from __future__ import annotations

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
) -> tuple[float, dict | None]:
    """Apply calibration to a raw probability estimate.

    Args:
        probability: Raw aggregate probability
        config: Pipeline config
        platt_params: Fitted Platt parameters (if available)
        n_resolved: Number of resolved predictions in dataset

    Returns:
        tuple of (calibrated_probability, platt_params_dict_or_None)
    """
    if n_resolved >= config.calibration.min_resolved_for_platt and platt_params:
        # Apply Platt scaling
        calibrated = platt_transform(probability, platt_params)
        logger.info(
            "Platt calibration: %.3f -> %.3f (a=%.3f, b=%.3f)",
            probability, calibrated, platt_params.a, platt_params.b,
        )
        return calibrated, platt_params.to_dict()
    else:
        # Apply prior correction: shift away from 0.5 to counter LLM hedging
        calibrated = apply_hedging_correction(
            probability, config.calibration.prior_hedging_shift
        )
        logger.info(
            "Prior correction: %.3f -> %.3f (shift=%.3f, n_resolved=%d/%d for Platt)",
            probability, calibrated, config.calibration.prior_hedging_shift,
            n_resolved, config.calibration.min_resolved_for_platt,
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
