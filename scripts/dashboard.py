"""CLI: Show calibration dashboard.

Usage:
    python -m scripts.dashboard
"""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.table import Table

from src.config import Config
from src.registry.store import RegistryStore
from src.registry.query import RegistryQuery
from src.scoring import brier_score

console = Console()


@click.command()
def main():
    """Display calibration dashboard for prediction performance."""
    config = Config()
    store = RegistryStore(config.registry_path)
    query = RegistryQuery(store)

    all_records = store.load_all()
    resolved = query.get_resolved()

    console.print()
    console.rule("[bold blue]Sigma-Predict Dashboard[/bold blue]")
    console.print()

    # Overview
    console.print(f"[bold]Total predictions:[/bold] {len(all_records)}")
    console.print(f"[bold]Resolved:[/bold] {len(resolved)}")
    console.print(f"[bold]Pending:[/bold] {len(all_records) - len(resolved)}")

    if not resolved:
        console.print("\n[yellow]No resolved predictions yet. Submit predictions and wait for resolution.[/yellow]")
        return

    # Overall Brier score
    pairs = query.get_calibration_pairs()
    if pairs:
        brier_scores = [brier_score(p, a) for p, a in pairs]
        avg_brier = sum(brier_scores) / len(brier_scores)
        console.print(f"\n[bold]Average Brier Score:[/bold] {avg_brier:.4f}")
        console.print(f"[dim](Lower is better. Random baseline = 0.25, perfect = 0.0)[/dim]")

    # By domain
    domains: dict[str, list[tuple[float, float]]] = {}
    for r in resolved:
        if r.resolution and r.resolution.outcome is not None and r.submitted:
            d = r.domain or "unknown"
            domains.setdefault(d, []).append((r.submitted.probability, r.resolution.outcome))

    if domains:
        console.print()
        domain_table = Table(title="Brier Score by Domain")
        domain_table.add_column("Domain")
        domain_table.add_column("N", justify="right")
        domain_table.add_column("Brier", justify="right")

        for domain, domain_pairs in sorted(domains.items()):
            scores = [brier_score(p, a) for p, a in domain_pairs]
            domain_table.add_row(
                domain,
                str(len(domain_pairs)),
                f"{sum(scores)/len(scores):.4f}",
            )
        console.print(domain_table)

    # By platform
    platforms: dict[str, list[tuple[float, float]]] = {}
    for r in resolved:
        if r.resolution and r.resolution.outcome is not None and r.submitted:
            platforms.setdefault(r.platform, []).append(
                (r.submitted.probability, r.resolution.outcome)
            )

    if platforms:
        console.print()
        plat_table = Table(title="Brier Score by Platform")
        plat_table.add_column("Platform")
        plat_table.add_column("N", justify="right")
        plat_table.add_column("Brier", justify="right")

        for plat, plat_pairs in sorted(platforms.items()):
            scores = [brier_score(p, a) for p, a in plat_pairs]
            plat_table.add_row(
                plat,
                str(len(plat_pairs)),
                f"{sum(scores)/len(scores):.4f}",
            )
        console.print(plat_table)

    # Cost summary
    total_cost = sum(r.cost.total_cost_usd for r in all_records)
    console.print(f"\n[bold]Total cost:[/bold] ${total_cost:.4f}")
    if all_records:
        console.print(f"[bold]Avg cost/prediction:[/bold] ${total_cost/len(all_records):.4f}")

    console.print()


if __name__ == "__main__":
    main()
