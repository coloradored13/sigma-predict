"""CLI: Process resolved questions and compute scoring.

Usage:
    python -m scripts.resolve --check-all
    python -m scripts.resolve --prediction-id <id>
"""

from __future__ import annotations

import logging
import math
import sys

import click

from src.config import Config
from src.models import Resolution
from src.platforms.metaculus import MetaculusClient
from src.registry.store import RegistryStore
from src.registry.query import RegistryQuery


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
