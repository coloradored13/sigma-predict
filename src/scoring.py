"""Scoring functions shared across scripts."""

from __future__ import annotations

import math


def brier_score(predicted: float, actual: float) -> float:
    """Compute Brier score: (predicted - actual)^2."""
    return (predicted - actual) ** 2


def log_score(predicted: float, actual: float) -> float:
    """Compute log score: -log(predicted) if actual=1, -log(1-predicted) if actual=0."""
    p = max(0.001, min(0.999, predicted))
    if actual >= 0.5:
        return -math.log(p)
    else:
        return -math.log(1.0 - p)
