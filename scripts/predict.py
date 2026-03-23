"""CLI: Make a prediction on a question.

Usage:
    python -m scripts.predict --platform metaculus --question-id 12345
    python -m scripts.predict --platform metaculus --search "AI progress"
    python -m scripts.predict --url "https://www.metaculus.com/questions/12345/"
"""

from __future__ import annotations

import logging
import sys

import click

from src.config import Config
from src.models import Platform, SearchPersona
from src.pipeline.orchestrator import Orchestrator
from src.platforms.metaculus import MetaculusClient
from src.platforms.polymarket import PolymarketClient
from src.platforms.kalshi import KalshiClient


def _get_platform_client(platform: str, config: Config):
    """Get the appropriate platform client."""
    clients = {
        "metaculus": lambda: MetaculusClient(config),
        "polymarket": lambda: PolymarketClient(config),
        "kalshi": lambda: KalshiClient(config),
    }
    factory = clients.get(platform)
    if not factory:
        raise click.BadParameter(f"Unknown platform: {platform}")
    return factory()


@click.command()
@click.option(
    "--platform", "-p",
    type=click.Choice(["metaculus", "polymarket", "kalshi"]),
    default="metaculus",
    help="Prediction platform",
)
@click.option("--question-id", "-q", type=str, help="Question ID to predict on")
@click.option("--search", "-s", type=str, help="Search for questions matching this term")
@click.option("--url", "-u", type=str, help="Question URL (extracts platform and ID)")
@click.option(
    "--runs", "-n",
    type=int,
    default=1,
    help="Number of independent estimation runs (default: 1 for Phase 0)",
)
@click.option(
    "--personas",
    type=str,
    default=None,
    help="Comma-separated personas: academic,news,contrarian,domain_expert,meta_forecaster",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
@click.option("--debug", is_flag=True, help="Debug logging")
def main(
    platform: str,
    question_id: str | None,
    search: str | None,
    url: str | None,
    runs: int,
    personas: str | None,
    verbose: bool,
    debug: bool,
):
    """Make a prediction on a forecasting question."""
    # Configure logging
    level = logging.DEBUG if debug else logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = Config()

    # Validate config
    missing = config.validate()
    if missing:
        click.echo(f"Missing required config: {', '.join(missing)}", err=True)
        click.echo("Set these as environment variables.", err=True)
        sys.exit(1)

    # Parse URL if provided
    if url:
        platform, question_id = _parse_url(url)

    # Get platform client
    client = _get_platform_client(platform, config)

    # Get question
    if question_id:
        click.echo(f"Fetching question {question_id} from {platform}...")
        question = client.get_question(question_id)
    elif search:
        click.echo(f"Searching {platform} for: {search}")
        questions = client.get_questions(limit=10, search=search)
        if not questions:
            click.echo("No questions found.", err=True)
            sys.exit(1)

        # Let user pick
        click.echo(f"\nFound {len(questions)} questions:\n")
        for i, q in enumerate(questions, 1):
            community = f" (community: {q.community_prediction:.3f})" if q.community_prediction else ""
            click.echo(f"  {i}. {q.title}{community}")
            click.echo(f"     ID: {q.question_id} | Close: {q.close_date[:10] if q.close_date else 'N/A'}")

        while True:
            choice = click.prompt("\nSelect question number", type=int)
            if 1 <= choice <= len(questions):
                question = questions[choice - 1]
                break
            click.echo(f"Enter 1-{len(questions)}")
    else:
        click.echo("Provide --question-id, --search, or --url", err=True)
        sys.exit(1)

    click.echo(f"\nPredicting: {question.title}\n")

    # Parse personas
    persona_list = None
    if personas:
        persona_list = [SearchPersona(p.strip()) for p in personas.split(",")]

    # Run pipeline
    orchestrator = Orchestrator(config)
    record = orchestrator.predict(
        question=question,
        platform_client=client,
        num_runs=runs,
        personas=persona_list,
    )

    # Summary
    if record.submitted:
        click.echo(f"\nPrediction submitted: {record.submitted.probability:.3f}")
        click.echo(f"Registry ID: {record.prediction_id}")
        click.echo(f"Cost: ${record.cost.total_cost_usd:.4f}")
    elif record.human_review.human_reasoning and "REJECTED" in (record.human_review.human_reasoning or ""):
        click.echo("\nPrediction rejected. Logged to registry.")
    else:
        click.echo("\nPrediction logged (not submitted).")


def _parse_url(url: str) -> tuple[str, str]:
    """Extract platform and question ID from a URL."""
    if "metaculus.com" in url:
        # https://www.metaculus.com/questions/12345/slug/
        parts = url.rstrip("/").split("/")
        for i, part in enumerate(parts):
            if part == "questions" and i + 1 < len(parts):
                return "metaculus", parts[i + 1]
        raise click.BadParameter(f"Could not parse Metaculus URL: {url}")

    elif "polymarket.com" in url:
        parts = url.rstrip("/").split("/")
        return "polymarket", parts[-1]

    elif "kalshi.com" in url:
        parts = url.rstrip("/").split("/")
        return "kalshi", parts[-1]

    else:
        raise click.BadParameter(f"Unknown platform URL: {url}")


if __name__ == "__main__":
    main()
