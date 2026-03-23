"""CLI: Run platform behavioral signal monitor (Phase 3).

Usage:
    python -m scripts.monitor --interval 3600
"""

from __future__ import annotations

import click


@click.command()
@click.option("--interval", type=int, default=3600, help="Poll interval in seconds")
def main(interval: int):
    """Run the platform behavioral signal monitor.

    Phase 3 feature — not yet implemented.
    """
    click.echo("Platform monitor is a Phase 3 feature and not yet implemented.")
    click.echo("It will poll Metaculus, Polymarket, and Kalshi APIs at regular intervals")
    click.echo("to detect anomalous price movements and cross-platform divergence.")


if __name__ == "__main__":
    main()
