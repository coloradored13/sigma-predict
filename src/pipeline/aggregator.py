"""Statistical aggregation of multiple estimation runs.

Pure Python — no LLM calls. Computes trimmed mean, extremized mean,
inter-model agreement, and effective-N.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np

from src.models import Aggregation, RunResult

# Domains exempt from clustering warnings — authoritative sources that
# legitimately appear across many independent searches.
_AUTHORITATIVE_DOMAINS = frozenset({
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "nytimes.com", "wsj.com", "ft.com", "economist.com",
    "who.int", "cdc.gov", "bls.gov", "census.gov", "nih.gov", "nasa.gov",
    "data.worldbank.org", "fred.stlouisfed.org", "imf.org", "oecd.org",
    "europa.eu", "un.org",
    "wikipedia.org", "metaculus.com", "polymarket.com",
})

# Two-part TLD suffixes that require three hostname parts for the apex domain.
# Without this, "bbc.co.uk" would incorrectly return "co.uk" as the apex.
_TWO_PART_TLDS = frozenset({
    "co.uk", "com.au", "co.jp", "co.kr", "com.br",
    "co.in", "co.nz", "org.uk", "ac.uk",
})


def aggregate_runs(
    runs: list[RunResult],
    method: str = "extremized_trimmed_mean",
    trim_fraction: float = 0.1,
    extremization_factor: float = 1.5,
    n_distinct_providers: int = 1,
) -> Aggregation:
    """Aggregate multiple run estimates into a single probability.

    Args:
        runs: List of completed forecast runs
        method: Aggregation method to use
        trim_fraction: Fraction to trim from each end (for trimmed mean)
        extremization_factor: Factor to push estimate away from 0.5
            Values > 1.0 push away from 0.5 (extremize)
            Values < 1.0 pull toward 0.5 (moderate)
        n_distinct_providers: Number of distinct model providers in runs.
            Extremization is only applied when this is >= 2.

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

    # Extremized mean: push the trimmed mean away from 0.5 (only with multiple providers)
    # Using the log-odds transformation
    if method == "extremized_trimmed_mean":
        if n_distinct_providers >= 2:
            raw_aggregate = _extremize(trimmed_mean, extremization_factor)
        else:
            raw_aggregate = trimmed_mean
    elif method == "trimmed_mean":
        raw_aggregate = trimmed_mean
    elif method == "median":
        raw_aggregate = raw_median
    elif method == "mean":
        raw_aggregate = raw_mean
    else:
        if n_distinct_providers >= 2:
            raw_aggregate = _extremize(trimmed_mean, extremization_factor)
        else:
            raw_aggregate = trimmed_mean

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

    clustering_score, clustered_domains = source_cluster_check(runs)

    return Aggregation(
        method=method,
        raw_aggregate=raw_aggregate,
        stdev=round(stdev, 4),
        inter_model_agreement=round(inter_model_agreement, 4),
        effective_n=effective_n,
        sources_clustering_score=round(clustering_score, 4),
        clustered_domains=clustered_domains,
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


def source_cluster_check(runs: list[RunResult]) -> tuple[float, list[str]]:
    """Detect apex-domain clustering across forecast runs.

    Recency bias or filter bubble risk: when N/N runs cite the same news source,
    the aggregate probability estimate has reduced independence value.

    Returns:
        (clustering_score, clustered_domains)
        clustering_score: 0.0 = no clustering, 1.0 = all runs cite same apex domain
        clustered_domains: apex domains appearing in >=50% of runs (non-authoritative only)
    """
    if len(runs) < 2:
        return 0.0, []

    def apex_domain(url: str) -> str:
        try:
            host = urlparse(url).netloc.lower()
            parts = host.split(".")
            if len(parts) >= 3 and ".".join(parts[-2:]) in _TWO_PART_TLDS:
                return ".".join(parts[-3:])
            return ".".join(parts[-2:]) if len(parts) >= 2 else host
        except Exception:
            return url

    n_runs = len(runs)
    threshold = 0.5  # domain must appear in >=50% of runs to be flagged

    # Build per-run apex domain sets
    run_domains: list[set[str]] = []
    for run in runs:
        domains = {apex_domain(url) for url in run.sources_cited if url}
        run_domains.append(domains)

    # Count how many runs cite each domain
    all_domains: set[str] = set()
    for ds in run_domains:
        all_domains.update(ds)

    domain_counts: dict[str, int] = {}
    for domain in all_domains:
        domain_counts[domain] = sum(1 for ds in run_domains if domain in ds)

    # Identify non-authoritative domains cited by >= threshold of runs
    clustered = [
        d for d, count in domain_counts.items()
        if count / n_runs >= threshold and d not in _AUTHORITATIVE_DOMAINS
    ]

    if not clustered:
        return 0.0, []

    # Score = fraction of runs sharing the most-clustered domain
    max_count = max(domain_counts[d] for d in clustered)
    score = max_count / n_runs

    return score, sorted(clustered)


def get_aggregate_run(runs: list[RunResult], aggregation: Aggregation) -> RunResult:
    """Construct a synthetic RunResult representing the aggregate of all runs.

    Used by challenge_forecast to challenge the aggregate probability rather
    than the last individual run. Summarizes reasoning from all runs.

    Args:
        runs: All completed forecast runs
        aggregation: The computed aggregation from aggregate_runs()

    Returns:
        Synthetic RunResult with aggregate probability and summarized reasoning
    """
    if not runs:
        return RunResult(
            model="aggregate",
            probability=aggregation.raw_aggregate,
            reasoning_chain="No runs to aggregate.",
        )

    # Summarize reasoning chains from all runs
    reasoning_parts = []
    for i, run in enumerate(runs):
        reasoning_parts.append(
            f"Run {i + 1} (model={run.model}, p={run.probability:.3f}): "
            f"{run.reasoning_chain[:200]}"
        )
    combined_reasoning = "\n\n".join(reasoning_parts)

    # Use mean of base rates and inside-view adjustments
    mean_base_rate = float(np.mean([r.base_rate_used for r in runs]))
    mean_adjustment = float(np.mean([r.inside_view_adjustment for r in runs]))

    # Represent the worst pre-mortem outcome across runs
    max_pre_mortem_delta = max(
        (abs(r.pre_mortem_delta) for r in runs), default=0.0
    )
    worst_run = max(runs, key=lambda r: abs(r.pre_mortem_delta), default=runs[0])

    return RunResult(
        model="aggregate",
        search_persona="aggregate",
        base_rate_used=round(mean_base_rate, 4),
        inside_view_adjustment=round(mean_adjustment, 4),
        adjustment_reasoning=f"Aggregate of {len(runs)} run(s)",
        pre_mortem_scenario=worst_run.pre_mortem_scenario,
        pre_mortem_changed_estimate=any(r.pre_mortem_changed_estimate for r in runs),
        pre_mortem_delta=round(max_pre_mortem_delta, 4),
        probability=aggregation.raw_aggregate,
        confidence_self_score=int(
            round(float(np.mean([r.confidence_self_score for r in runs])))
        ),
        reasoning_chain=combined_reasoning,
    )
