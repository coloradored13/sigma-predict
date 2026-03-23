"""Statistical aggregation of multiple estimation runs.

Pure Python — no LLM calls. Computes trimmed mean, extremized mean,
inter-model agreement, and effective-N.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.models import Aggregation, RunResult


def aggregate_runs(
    runs: list[RunResult],
    method: str = "extremized_trimmed_mean",
    trim_fraction: float = 0.1,
    extremization_factor: float = 1.5,
) -> Aggregation:
    """Aggregate multiple run estimates into a single probability.

    Args:
        runs: List of completed forecast runs
        method: Aggregation method to use
        trim_fraction: Fraction to trim from each end (for trimmed mean)
        extremization_factor: Factor to push estimate away from 0.5
            Values > 1.0 push away from 0.5 (extremize)
            Values < 1.0 pull toward 0.5 (moderate)

    Returns:
        Aggregation with computed statistics
    """
    if not runs:
        return Aggregation(method=method)

    probabilities = np.array([r.probability for r in runs])
    n = len(probabilities)

    if n == 1:
        p = float(probabilities[0])
        return Aggregation(
            method="single_run",
            raw_aggregate=p,
            stdev=0.0,
            inter_model_agreement=1.0,
            effective_n=1,
        )

    # Compute various aggregation statistics
    raw_mean = float(np.mean(probabilities))
    raw_median = float(np.median(probabilities))
    stdev = float(np.std(probabilities, ddof=1))

    # Trimmed mean: remove top/bottom trim_fraction
    trim_count = max(1, int(n * trim_fraction))
    if n > 2 * trim_count:
        sorted_p = np.sort(probabilities)
        trimmed = sorted_p[trim_count:-trim_count]
        trimmed_mean = float(np.mean(trimmed))
    else:
        trimmed_mean = raw_mean

    # Extremized mean: push the trimmed mean away from 0.5
    # Using the log-odds transformation
    if method == "extremized_trimmed_mean":
        raw_aggregate = _extremize(trimmed_mean, extremization_factor)
    elif method == "trimmed_mean":
        raw_aggregate = trimmed_mean
    elif method == "median":
        raw_aggregate = raw_median
    elif method == "mean":
        raw_aggregate = raw_mean
    else:
        raw_aggregate = _extremize(trimmed_mean, extremization_factor)

    # Clamp to valid probability range
    raw_aggregate = max(0.01, min(0.99, raw_aggregate))

    # Inter-model agreement: 1 - normalized stdev
    # stdev of 0 = perfect agreement (1.0), stdev of 0.5 = minimum agreement (0.0)
    inter_model_agreement = max(0.0, 1.0 - 2.0 * stdev)

    # Effective N: accounts for correlation between runs
    # If runs are perfectly correlated, effective_n = 1
    # If perfectly independent, effective_n = n
    # Estimate via the ratio of mean variance to individual variances
    effective_n = _estimate_effective_n(probabilities)

    return Aggregation(
        method=method,
        raw_aggregate=raw_aggregate,
        stdev=round(stdev, 4),
        inter_model_agreement=round(inter_model_agreement, 4),
        effective_n=effective_n,
    )


def _extremize(p: float, factor: float) -> float:
    """Push probability away from 0.5 using log-odds transformation.

    extremize(p, factor) = logistic(factor * logit(p))
    """
    if p <= 0.01 or p >= 0.99:
        return p

    # logit transform
    logit_p = math.log(p / (1.0 - p))

    # Apply extremization factor
    extremized_logit = factor * logit_p

    # Inverse logit (logistic)
    result = 1.0 / (1.0 + math.exp(-extremized_logit))

    return result


def _estimate_effective_n(probabilities: np.ndarray) -> int:
    """Estimate effective number of independent estimates.

    Uses the variance ratio method: if all estimates are the same,
    effective_n = 1. If they're fully spread, effective_n = n.
    """
    n = len(probabilities)
    if n <= 1:
        return n

    # Simple heuristic: effective_n based on spread relative to expected
    # For n truly independent estimates from a beta distribution,
    # we'd expect a certain spread. Compare actual spread to expected.
    spread = float(np.ptp(probabilities))  # range
    mean_p = float(np.mean(probabilities))

    # Expected spread for n independent draws from Beta with mean=mean_p
    # This is a rough heuristic
    expected_var_single = mean_p * (1 - mean_p) * 0.1  # rough estimate
    if expected_var_single > 0:
        actual_var = float(np.var(probabilities, ddof=1))
        # If actual variance is close to expected_var/n, estimates are independent
        ratio = actual_var / expected_var_single if expected_var_single > 0 else 1.0
        effective_n = max(1, min(n, int(round(n * min(1.0, ratio * n)))))
    else:
        effective_n = 1

    return max(1, min(n, effective_n))
