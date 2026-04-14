"""Conditional deliberation module (Phase 3).

Triggered when:
- Cross-model disagreement is high (inter_model_agreement < threshold), OR
- Challenge vulnerability is "high", OR
- |pre_mortem_delta| > 0.1 across any run.

A supervisor LLM reviews disagreements, focusing on "why do runs disagree?"
"""

from __future__ import annotations

import json
import logging

from src.config import Config
from src.models import Aggregation, ChallengeEntry, RunResult

logger = logging.getLogger(__name__)

# Trigger thresholds
_INTER_MODEL_AGREEMENT_THRESHOLD = 0.6  # below this → deliberate
_PRE_MORTEM_DELTA_THRESHOLD = 0.1       # above this (abs) → deliberate


def should_deliberate(
    aggregation: Aggregation,
    config: Config,
    challenge_entry: ChallengeEntry | None = None,
    pre_mortem_delta: float = 0.0,
) -> bool:
    """Check if deliberation should be triggered.

    Triggers on any of three conditions:
    1. Cross-model disagreement: inter_model_agreement < threshold
    2. Challenge vulnerability == "high"
    3. |pre_mortem_delta| > 0.1
    """
    # Condition 1: cross-model disagreement
    if aggregation.inter_model_agreement < _INTER_MODEL_AGREEMENT_THRESHOLD:
        logger.info(
            "Deliberation condition met: inter_model_agreement=%.3f < %.3f",
            aggregation.inter_model_agreement,
            _INTER_MODEL_AGREEMENT_THRESHOLD,
        )
        return True

    # Condition 2: high challenge vulnerability
    if challenge_entry is not None and challenge_entry.vulnerability == "high":
        logger.info("Deliberation condition met: challenge vulnerability=high")
        return True

    # Condition 3: large pre-mortem delta
    if abs(pre_mortem_delta) > _PRE_MORTEM_DELTA_THRESHOLD:
        logger.info(
            "Deliberation condition met: |pre_mortem_delta|=%.3f > %.3f",
            abs(pre_mortem_delta),
            _PRE_MORTEM_DELTA_THRESHOLD,
        )
        return True

    return False


def deliberate(
    runs: list[RunResult],
    aggregation: Aggregation,
    config: Config,
    router=None,
) -> tuple[list[RunResult], Aggregation]:
    """Run conditional deliberation on high-disagreement predictions.

    Uses a minimal supervisor LLM call to identify why runs disagree and
    produce a deliberation note. Does not modify individual run probabilities
    but records the deliberation result on the aggregation.

    Args:
        runs: All completed forecast runs
        aggregation: Current aggregation result
        config: Pipeline config
        router: LLMRouter instance (if None, deliberation is skipped with a warning)

    Returns:
        (runs, updated_aggregation) — runs unchanged, aggregation has
        deliberation_result populated
    """
    if router is None:
        logger.warning(
            "Deliberation triggered (stdev=%.3f) but no router provided — skipping",
            aggregation.stdev,
        )
        return runs, aggregation

    if not runs:
        return runs, aggregation

    # Build disagreement summary for supervisor
    run_summaries = []
    for i, run in enumerate(runs):
        run_summaries.append(
            f"Run {i + 1}: model={run.model}, p={run.probability:.3f}, "
            f"base_rate={run.base_rate_used:.3f}, adj={run.inside_view_adjustment:+.3f}\n"
            f"  Reasoning: {run.reasoning_chain[:300]}"
        )

    probabilities = [r.probability for r in runs]
    system_prompt = (
        "You are a deliberation supervisor for a forecasting system. "
        "Multiple independent forecasters have produced divergent probability estimates. "
        "Your task: identify WHY they disagree (not who is right) and note any "
        "structural blind spots or conflicting evidence. "
        "Be concise. Output JSON with keys: "
        '"disagreement_reason" (string), "key_uncertainties" (list of strings), '
        '"recommended_action" (string: "accept_aggregate" | "investigate_further").'
    )

    user_content = (
        f"Forecaster run summaries:\n\n"
        + "\n\n".join(run_summaries)
        + f"\n\nAggregate probability: {aggregation.raw_aggregate:.3f} "
        f"(stdev={aggregation.stdev:.3f}, inter_model_agreement={aggregation.inter_model_agreement:.3f})\n"
        f"Range: {min(probabilities):.3f} – {max(probabilities):.3f}\n\n"
        "Why do these runs disagree? Produce your JSON response now."
    )

    try:
        raw, _, _ = router.call(
            config.primary_model.provider,
            config.primary_model.model_id,
            system_prompt,
            user_content,
            temperature=0.3,
            max_tokens=512,
        )

        # Parse JSON from response
        deliberation_result = _parse_deliberation_json(raw)
        logger.info(
            "Deliberation complete: action=%s, reason=%s",
            deliberation_result.get("recommended_action", "unknown"),
            deliberation_result.get("disagreement_reason", "")[:100],
        )
    except Exception as e:
        logger.warning("Deliberation LLM call failed: %s", e)
        deliberation_result = {
            "disagreement_reason": f"Deliberation failed: {e}",
            "key_uncertainties": [],
            "recommended_action": "accept_aggregate",
        }

    aggregation.deliberation_result = deliberation_result
    return runs, aggregation


def _parse_deliberation_json(text: str) -> dict:
    """Extract JSON from deliberation response."""
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()
    else:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "disagreement_reason": text[:200] if text else "Parse failure",
            "key_uncertainties": [],
            "recommended_action": "accept_aggregate",
        }
