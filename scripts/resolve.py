"""CLI: Process resolved questions and compute scoring.

Usage:
    python -m scripts.resolve --check-all
    python -m scripts.resolve --prediction-id <id>
"""

from __future__ import annotations

import logging
import sys

import click

from src.config import Config
from src.models import PredictionRecord, Resolution
from src.platforms.metaculus import MetaculusClient
from src.registry.store import RegistryStore
from src.registry.query import RegistryQuery
from src.scoring import brier_score, log_score


def classify_error(record: PredictionRecord, outcome: float) -> list[str]:
    """Classify prediction errors into diagnostic categories.

    Returns a list of zero or more error class strings:
      CONFABULATION       — base rate strongly disagrees with outcome + low search quality
      PROCESS             — parse failures present in any run
      CALIBRATION_FAILURE — high Brier score and large drift from base rate
      SEARCH_BLINDSPOT    — very low search quality scores
      ACTOR_MISCLASSIFICATION — L2/L3 actors present but outcome contradicts calibrated prob
    """
    errors: list[str] = []
    cal_prob = record.calibration.calibrated_probability
    base_rate = record.decomposition.base_rate

    # CONFABULATION: base rate strongly disagrees with outcome AND search quality low
    if abs(base_rate - outcome) > 0.5 and any(
        r.confidence_self_score < 3 for r in record.runs
    ):
        errors.append("CONFABULATION")

    # PROCESS: parse failures present
    if any(getattr(r, "parse_failed", False) for r in record.runs):
        errors.append("PROCESS")

    # CALIBRATION_FAILURE: high Brier AND large delta from base rate
    brier = (cal_prob - outcome) ** 2
    if brier > 0.25 and abs(cal_prob - base_rate) > 0.4:
        errors.append("CALIBRATION_FAILURE")

    # SEARCH_BLINDSPOT: very low search quality scores
    if any(getattr(r, "search_quality_score", 5) < 2 for r in record.runs):
        errors.append("SEARCH_BLINDSPOT")

    # ACTOR_MISCLASSIFICATION: L2/L3 actors present but outcome contradicts calibrated prob
    if (
        record.decomposition.actor_analysis
        and record.decomposition.actor_analysis.level_2_3_present
    ):
        if abs(cal_prob - outcome) > 0.3:
            errors.append("ACTOR_MISCLASSIFICATION")

    return errors


@click.command()
@click.option("--check-all", is_flag=True, help="Check all unresolved predictions")
@click.option("--prediction-id", type=str, help="Check a specific prediction")
@click.option("--verbose", "-v", is_flag=True)
def main(check_all: bool, prediction_id: str | None, verbose: bool):
    """Process resolved questions and compute accuracy scores."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")

    config = Config()
    store = RegistryStore(config.registry_path)
    query = RegistryQuery(store)
    metaculus = MetaculusClient(config)

    records = store.load_all()
    if prediction_id:
        records = [r for r in records if r.prediction_id == prediction_id]

    updated = 0
    for record in records:
        if record.resolution is not None:
            continue
        if not record.submitted:
            continue

        # Try to get resolution from platform
        if record.platform == "metaculus":
            resolution = metaculus.get_resolution(record.question_id)
        else:
            click.echo(f"  Skipping {record.prediction_id} — resolution check not implemented for {record.platform}")
            continue

        if resolution is None:
            if verbose:
                click.echo(f"  {record.question_id}: not yet resolved")
            continue

        # Compute scores
        submitted_p = record.submitted.probability
        outcome = resolution.outcome
        if outcome is not None:
            resolution.brier_score = round(brier_score(submitted_p, outcome), 6)
            resolution.log_score = round(log_score(submitted_p, outcome), 6)
            resolution.error_class_auto = classify_error(record, outcome)

        record.resolution = resolution
        store.update(record)
        updated += 1

        click.echo(
            f"  Resolved: {record.question_text[:60]}... "
            f"predicted={submitted_p:.3f} actual={outcome} "
            f"brier={resolution.brier_score:.4f}"
        )

    click.echo(f"\nUpdated {updated} predictions with resolution data.")
    click.echo(f"Total resolved: {query.count_resolved()}")


if __name__ == "__main__":
    main()
