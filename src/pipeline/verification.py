"""Cross-model verification at each pipeline stage."""

from __future__ import annotations

import logging

import numpy as np

from src.config import Config
from src.models import (
    ActorAnalysis,
    ChallengeEntry,
    RunResult,
    StageVerification,
    VerificationEntry,
)
from src.pipeline.llm_router import LLMRouter

logger = logging.getLogger(__name__)


def _result_to_entry(result) -> VerificationEntry:
    """Convert a sigma-verify VerificationResult or error dict to VerificationEntry."""
    if hasattr(result, "to_dict"):
        d = result.to_dict()
    else:
        d = result if isinstance(result, dict) else {}
    return VerificationEntry(
        provider=d.get("provider", ""),
        model=d.get("model", ""),
        assessment=d.get("assessment", ""),
        confidence=d.get("confidence", ""),
        reasoning=d.get("reasoning", ""),
        counter_evidence=d.get("counter_evidence", ""),
        status=d.get("status", "success"),
        error_class=d.get("error_class", ""),
    )


def _compute_agreement(entries: list[VerificationEntry]) -> str:
    """Compute agreement level across verification entries."""
    successes = [e for e in entries if e.status == "success"]
    if not successes:
        return "no_data"
    assessments = {e.assessment for e in successes}
    if len(assessments) == 1:
        return "unanimous"
    if len(assessments) == len(successes):
        return "none"
    return "partial"


def verify_base_rate(
    base_rate: float,
    ref_class: str,
    question_text: str,
    router: LLMRouter,
    config: Config,
) -> StageVerification:
    """Verify the base rate estimate using external models.

    Formats the base rate and reference class as a finding,
    calls router.cross_verify(), and returns a StageVerification.
    """
    finding = (
        f"Base rate estimate: {base_rate:.3f} for reference class '{ref_class}'. "
        f"Question: {question_text[:200]}"
    )
    context = (
        f"This is a statistical base rate estimate for a forecasting question. "
        f"The reference class is '{ref_class}'. The estimate was made in isolation "
        f"before considering any inside-view evidence or current events."
    )

    results = router.cross_verify(finding, context)
    entries = [_result_to_entry(r) for r in results]
    agreement = _compute_agreement(entries)

    logger.info(
        "Base rate verification: %d results, agreement=%s",
        len(entries), agreement,
    )

    return StageVerification(
        stage="base_rate",
        finding_summary=f"Base rate {base_rate:.3f} (ref class: {ref_class})",
        verifications=entries,
        agreement=agreement,
    )


def verify_actors(
    actor_analysis: ActorAnalysis,
    question_text: str,
    router: LLMRouter,
    config: Config,
) -> StageVerification | None:
    """Verify actor analysis using external models.

    Only runs if level_2_3_present (identity-constrained or divergent-preference
    actors are present). Returns None if no L2/3 actors.
    """
    if not actor_analysis.level_2_3_present:
        return None

    # Build a summary of the actor analysis
    actor_summaries = []
    for actor in actor_analysis.actors:
        if actor.level >= 2:
            actor_summaries.append(
                f"- {actor.name}: Level {actor.level} ({actor.level_label}), "
                f"motivation='{actor.motivation}', influence={actor.influence_estimate}"
            )

    finding = (
        f"Actor analysis for question: {question_text[:200]}\n"
        f"Level 2/3 actors identified:\n"
        + "\n".join(actor_summaries)
    )
    context = (
        f"This actor analysis classifies key actors in a forecasting question "
        f"using a 3-level motivation hierarchy: Level 1 (convergent rational), "
        f"Level 2 (identity-constrained), Level 3 (divergent preference). "
        f"Level 2/3 actors introduce non-rational dynamics that affect probabilities."
    )

    results = router.cross_verify(finding, context)
    entries = [_result_to_entry(r) for r in results]
    agreement = _compute_agreement(entries)

    logger.info(
        "Actor verification: %d results, agreement=%s",
        len(entries), agreement,
    )

    return StageVerification(
        stage="actors",
        finding_summary=f"{len(actor_summaries)} Level 2/3 actors identified",
        verifications=entries,
        agreement=agreement,
    )


def verify_forecast_run(
    run: RunResult,
    question_text: str,
    router: LLMRouter,
    config: Config,
) -> StageVerification:
    """Verify a single forecast run's conclusion using external models."""
    finding = (
        f"Forecast conclusion: probability={run.probability:.3f} "
        f"(base_rate={run.base_rate_used:.3f}, adjustment={run.inside_view_adjustment:+.3f}). "
        f"Model: {run.model}, persona: {run.search_persona}. "
        f"Reasoning: {run.reasoning_chain[:300]}"
    )
    context = (
        f"Question: {question_text[:200]}. "
        f"This is one forecaster run in a multi-run prediction pipeline. "
        f"The forecaster started from a locked base rate of {run.base_rate_used:.3f} "
        f"and adjusted based on evidence from independent search results."
    )

    results = router.cross_verify(finding, context)
    entries = [_result_to_entry(r) for r in results]
    agreement = _compute_agreement(entries)

    logger.info(
        "Forecast run verification (model=%s): %d results, agreement=%s",
        run.model, len(entries), agreement,
    )

    return StageVerification(
        stage=f"forecast_run_{run.run_id[:8]}",
        finding_summary=f"P={run.probability:.3f} (model={run.model}, persona={run.search_persona})",
        verifications=entries,
        agreement=agreement,
    )


def should_verify_run(
    run: RunResult,
    all_runs: list[RunResult],
    config: Config,
) -> bool:
    """Determine if a run should be verified based on outlier detection.

    Returns True if:
    - verify_forecasts is "all"
    - verify_forecasts is "outliers" AND this run's probability is
      further from the mean than outlier_threshold
    Returns False if verify_forecasts is "none".
    """
    verify_setting = config.verification.verify_forecasts

    if verify_setting == "none":
        return False
    if verify_setting == "all":
        return True

    # "outliers" — check if this run is an outlier
    if len(all_runs) < 2:
        # Can't compute meaningful outlier with < 2 runs; verify single runs
        return True

    probs = np.array([r.probability for r in all_runs])
    mean_p = np.mean(probs)
    deviation = abs(run.probability - mean_p)

    return bool(deviation > config.verification.outlier_threshold)


def challenge_forecast(
    run: RunResult,
    question_text: str,
    router: LLMRouter,
    config: Config,
) -> ChallengeEntry | None:
    """Use router.challenge() to stress-test a forecast run.

    Returns a ChallengeEntry or None if no external client is available.
    """
    claim = (
        f"The probability of '{question_text[:200]}' is {run.probability:.3f}. "
        f"Reasoning: {run.reasoning_chain[:400]}"
    )
    evidence = (
        f"Base rate: {run.base_rate_used:.3f}. "
        f"Inside-view adjustment: {run.inside_view_adjustment:+.3f}. "
        f"Pre-mortem scenario: {run.pre_mortem_scenario[:200]}. "
        f"Pre-mortem changed estimate: {run.pre_mortem_changed_estimate} "
        f"(delta={run.pre_mortem_delta:+.3f}). "
        f"Search persona: {run.search_persona}."
    )

    result = router.challenge(claim, evidence)
    if result is None:
        return None

    return ChallengeEntry(
        provider=result.get("provider", ""),
        model=result.get("model", ""),
        counter_argument=result.get("counter_argument", ""),
        logical_gaps=result.get("logical_gaps", ""),
        evidence_needed=result.get("evidence_needed", ""),
        vulnerability=result.get("vulnerability", ""),
        status=result.get("status", "success"),
    )
