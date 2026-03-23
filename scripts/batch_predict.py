"""CLI: Batch predictions across platforms.

Usage:
    python -m scripts.batch_predict --platform metaculus --limit 5
    python -m scripts.batch_predict --platform metaculus --tournament-id 3672
"""

from __future__ import annotations

import logging
import sys

import click

from src.config import Config
from src.models import SearchPersona
from src.pipeline.orchestrator import Orchestrator
from src.platforms.metaculus import MetaculusClient
from src.platforms.polymarket import PolymarketClient
from src.platforms.kalshi import KalshiClient


@click.command()
@click.option(
    "--platform", "-p",
    type=click.Choice(["metaculus", "polymarket", "kalshi"]),
    default="metaculus",
)
@click.option("--limit", "-l", type=int, default=5, help="Number of questions to predict on")
@click.option("--tournament-id", "-t", type=int, help="Metaculus tournament ID")
@click.option("--search", "-s", type=str, help="Search term for questions")
@click.option("--runs", "-n", type=int, default=1, help="Estimation runs per question")
@click.option("--verbose", "-v", is_flag=True)
def main(
    platform: str,
    limit: int,
    tournament_id: int | None,
    search: str | None,
    runs: int,
    verbose: bool,
):
    """Make predictions on multiple questions from a platform."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")

    config = Config()
    missing = config.validate()
    if missing:
        click.echo(f"Missing config: {', '.join(missing)}", err=True)
        sys.exit(1)

    # Get client
    clients = {
        "metaculus": lambda: MetaculusClient(config),
        "polymarket": lambda: PolymarketClient(config),
        "kalshi": lambda: KalshiClient(config),
    }
    client = clients[platform]()

    # Fetch questions
    click.echo(f"Fetching up to {limit} questions from {platform}...")
    questions = client.get_questions(
        limit=limit,
        tournament_id=tournament_id,
        search=search,
    )

    if not questions:
        click.echo("No questions found.")
        sys.exit(0)

    click.echo(f"Found {len(questions)} questions.\n")
    for i, q in enumerate(questions, 1):
        click.echo(f"  {i}. {q.title}")

    click.echo(f"\nProceeding with {len(questions)} predictions (runs={runs} each).\n")

    orchestrator = Orchestrator(config)
    total_cost = 0.0
    submitted = 0

    for i, question in enumerate(questions, 1):
        click.echo(f"\n{'='*60}")
        click.echo(f"[{i}/{len(questions)}] {question.title}")
        click.echo(f"{'='*60}")

        try:
            record = orchestrator.predict(
                question=question,
                platform_client=client,
                num_runs=runs,
            )
            total_cost += record.cost.total_cost_usd
            if record.submitted:
                submitted += 1
        except Exception as e:
            click.echo(f"  ERROR: {e}", err=True)
            continue

    click.echo(f"\n{'='*60}")
    click.echo(f"Batch complete: {submitted}/{len(questions)} submitted")
    click.echo(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
