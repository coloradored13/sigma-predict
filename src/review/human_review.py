"""Terminal-based human review interface.

Every prediction is presented for human review before submission.
Uses rich for formatted terminal output.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.models import HumanReview, PlatformQuestion, PredictionRecord

console = Console()


def present_for_review(
    record: PredictionRecord,
    question: PlatformQuestion,
) -> HumanReview:
    """Present a prediction for human review and collect decision.

    Displays:
    - Question text and key context
    - Per-run estimates with reasoning
    - Aggregate estimate with confidence
    - Actor analysis summary
    - Platform signals
    - Flags

    Returns:
        HumanReview with decision
    """
    console.print()
    console.rule("[bold blue]PREDICTION REVIEW[/bold blue]")
    console.print()

    # Question panel
    q_text = Text()
    q_text.append(question.title, style="bold")
    q_text.append(f"\n\n{question.description[:500]}" if question.description else "")
    if question.resolution_criteria:
        q_text.append(f"\n\nResolution: {question.resolution_criteria[:300]}")
    q_text.append(f"\n\nPlatform: {question.platform.value}")
    q_text.append(f" | Type: {question.question_type.value}")
    q_text.append(f" | Close: {question.close_date[:10] if question.close_date else 'N/A'}")
    console.print(Panel(q_text, title="Question", border_style="cyan"))

    # Decomposition summary
    decomp = record.decomposition
    if decomp.sub_questions:
        console.print("\n[bold]Sub-questions:[/bold]")
        for sq in decomp.sub_questions:
            console.print(f"  - {sq}")

    console.print(f"\n[bold]Base Rate:[/bold] {decomp.base_rate:.3f} ({decomp.base_rate_source})")
    console.print(f"[bold]Reference Class:[/bold] {decomp.reference_class}")

    # Actor analysis
    aa = decomp.actor_analysis
    if aa.actors_involved and aa.actors:
        console.print("\n[bold yellow]Actor Analysis:[/bold yellow]")
        for actor in aa.actors:
            level_style = {1: "green", 2: "yellow", 3: "red"}.get(actor.level, "white")
            console.print(
                f"  [{level_style}]Level {actor.level}[/{level_style}] "
                f"{actor.name}: {actor.motivation}"
            )
            if actor.level >= 2:
                console.print(f"    Influence: {actor.influence_estimate}")
                console.print(f"    Implication: {actor.implication_for_estimate}")
        if aa.peripheral_actors:
            console.print("\n  [bold cyan]Peripheral Actors:[/bold cyan]")
            for pa in aa.peripheral_actors:
                cat_labels = ", ".join(c.value for c in pa.categories)
                console.print(f"    {pa.name} [{cat_labels}] — {pa.motivation}")
                if pa.leverage:
                    console.print(f"      Leverage: {pa.leverage}")
                if pa.feedback_mechanism:
                    console.print(f"      Feedback: {pa.feedback_mechanism}")

        if aa.probability_space_constraints:
            console.print("\n  [bold]Constraints:[/bold]")
            for c in aa.probability_space_constraints:
                console.print(f"    - {c}")

    # Per-run estimates
    if record.runs:
        console.print()
        run_table = Table(title="Estimation Runs", show_header=True)
        run_table.add_column("Model", style="cyan")
        run_table.add_column("Persona", style="green")
        run_table.add_column("Base Rate", justify="right")
        run_table.add_column("Adjustment", justify="right")
        run_table.add_column("Pre-mortem", justify="right")
        run_table.add_column("Final P", justify="right", style="bold")
        run_table.add_column("Confidence", justify="right")

        for run in record.runs:
            pm_str = f"{run.pre_mortem_delta:+.3f}" if run.pre_mortem_changed_estimate else "—"
            run_table.add_row(
                run.model,
                run.search_persona,
                f"{run.base_rate_used:.3f}",
                f"{run.inside_view_adjustment:+.3f}",
                pm_str,
                f"{run.probability:.3f}",
                f"{run.confidence_self_score}/10",
            )
        console.print(run_table)

    # Aggregation
    agg = record.aggregation
    console.print()
    agg_panel_text = (
        f"Method: {agg.method}\n"
        f"Raw Aggregate: {agg.raw_aggregate:.3f}\n"
        f"Std Dev: {agg.stdev:.3f}\n"
        f"Inter-model Agreement: {agg.inter_model_agreement:.2f}\n"
        f"Effective N: {agg.effective_n}"
    )
    if agg.deliberation_triggered:
        agg_panel_text += "\n[yellow]Deliberation was triggered[/yellow]"

    console.print(Panel(agg_panel_text, title="Aggregation", border_style="green"))

    # Calibration
    cal = record.calibration
    console.print(
        f"\n[bold]Calibrated Probability:[/bold] "
        f"[bold white on blue] {cal.calibrated_probability:.3f} [/bold white on blue]"
    )
    if cal.platt_params_used:
        console.print(f"  (Platt scaled: a={cal.platt_params_used['a']:.3f}, b={cal.platt_params_used['b']:.3f})")
    else:
        console.print("  (Prior hedging correction applied — Platt not yet available)")
    console.print(
        f"  Confidence interval: [{cal.confidence_interval[0]:.3f}, {cal.confidence_interval[1]:.3f}]"
    )

    # Platform signals
    ps = record.platform_signals
    prices = ps.prices_at_prediction_time
    has_prices = any(v is not None for v in prices.values())
    if has_prices:
        console.print("\n[bold]Platform Prices:[/bold]")
        for platform, price in prices.items():
            if price is not None:
                console.print(f"  {platform}: {price:.3f}")

    if ps.anomaly_detected:
        console.print(f"\n[bold red]ANOMALY DETECTED:[/bold red] {ps.anomaly_details}")

    # Verification summary
    _display_verification_summary(record)

    # Flags
    flags = _generate_flags(record, question)
    if flags:
        console.print()
        for flag in flags:
            console.print(f"  [bold yellow]FLAG:[/bold yellow] {flag}")

    # Cost
    console.print(f"\n[dim]Cost: ${record.cost.total_cost_usd:.4f} | "
                  f"Latency: {record.cost.latency_seconds:.1f}s | "
                  f"Tokens: {record.cost.total_tokens_in}in/{record.cost.total_tokens_out}out[/dim]")

    # Collect decision
    console.print()
    console.rule("[bold]Decision[/bold]")
    console.print(
        "[bold green]a[/bold green]ccept | "
        "[bold yellow]j[/bold yellow]ust (adjust) | "
        "[bold red]r[/bold red]eject"
    )

    while True:
        choice = console.input("\n> ").strip().lower()

        if choice in ("a", "accept"):
            return HumanReview(
                reviewed=True,
                flags_raised=flags,
            )

        elif choice in ("j", "adjust"):
            while True:
                try:
                    adj_str = console.input("New probability (0.01-0.99): ").strip()
                    new_p = float(adj_str)
                    if 0.01 <= new_p <= 0.99:
                        break
                    console.print("[red]Must be between 0.01 and 0.99[/red]")
                except ValueError:
                    console.print("[red]Enter a decimal number[/red]")

            reasoning = console.input("Reasoning for adjustment: ").strip()
            adjustment = new_p - cal.calibrated_probability

            return HumanReview(
                reviewed=True,
                human_adjustment=adjustment,
                human_reasoning=reasoning,
                flags_raised=flags,
            )

        elif choice in ("r", "reject"):
            reasoning = console.input("Reason for rejection (optional): ").strip()
            return HumanReview(
                reviewed=True,
                human_adjustment=None,
                human_reasoning=f"REJECTED: {reasoning}" if reasoning else "REJECTED",
                flags_raised=flags,
            )

        else:
            console.print("[red]Enter 'a' (accept), 'j' (adjust), or 'r' (reject)[/red]")


def _display_verification_summary(record: PredictionRecord) -> None:
    """Display cross-model verification results."""
    v = record.verification
    if not v.enabled or not v.stages:
        if v.enabled and not v.stages:
            console.print("\n[dim]Verification: enabled but no stages executed[/dim]")
        return

    console.print()
    console.rule("[bold magenta]Cross-Model Verification[/bold magenta]")
    console.print(f"  Providers: {', '.join(v.providers_available)}")

    for stage in v.stages:
        # Agreement styling
        agreement_style = {
            "unanimous": "bold green",
            "partial": "bold yellow",
            "none": "bold red",
            "no_data": "dim",
        }.get(stage.agreement, "dim")

        console.print(
            f"\n  [{agreement_style}][{stage.agreement.upper()}][/{agreement_style}] "
            f"[bold]{stage.stage}[/bold]: {stage.finding_summary}"
        )

        # Show verifications
        for entry in stage.verifications:
            if entry.status != "success":
                console.print(
                    f"    [dim]{entry.provider}/{entry.model}: "
                    f"FAILED ({entry.error_class})[/dim]"
                )
                continue

            assess_style = {
                "agree": "green",
                "disagree": "red",
                "partial": "yellow",
                "uncertain": "dim",
            }.get(entry.assessment, "white")

            console.print(
                f"    [{assess_style}]{entry.assessment.upper()}[/{assess_style}] "
                f"({entry.confidence}) {entry.provider}/{entry.model}"
            )
            if entry.counter_evidence and entry.assessment in ("disagree", "partial"):
                console.print(
                    f"      Counter-evidence: {entry.counter_evidence[:200]}"
                )

        # Show challenges
        for challenge in stage.challenges:
            if challenge.status != "success":
                console.print(
                    f"    [dim]Challenge failed: {challenge.provider}/{challenge.model}[/dim]"
                )
                continue

            vuln_style = {
                "high": "bold red",
                "medium": "yellow",
                "low": "green",
            }.get(challenge.vulnerability, "white")

            console.print(
                f"    Challenge [{vuln_style}]{challenge.vulnerability.upper()} vulnerability"
                f"[/{vuln_style}] ({challenge.provider}/{challenge.model})"
            )
            if challenge.counter_argument:
                console.print(
                    f"      Counter-argument: {challenge.counter_argument[:200]}"
                )
            if challenge.logical_gaps:
                console.print(
                    f"      Logical gaps: {challenge.logical_gaps[:200]}"
                )


def _generate_flags(record: PredictionRecord, question: PlatformQuestion) -> list[str]:
    """Generate review flags based on prediction characteristics."""
    flags = []

    # High uncertainty
    if record.aggregation.stdev > 0.15:
        flags.append(f"High disagreement between runs (stdev={record.aggregation.stdev:.3f})")

    # Actor motivation risk
    aa = record.decomposition.actor_analysis
    if aa.level_2_3_present:
        l23_actors = [a.name for a in aa.actors if a.level >= 2]
        flags.append(f"Level 2/3 actors present: {', '.join(l23_actors)}")

    # Extreme probability
    cal_p = record.calibration.calibrated_probability
    if cal_p < 0.05 or cal_p > 0.95:
        flags.append(f"Extreme probability ({cal_p:.3f}) — verify confidence")

    # Large divergence from community
    ps = record.platform_signals.prices_at_prediction_time
    community = ps.get("metaculus_community")
    if community is not None and abs(cal_p - community) > 0.15:
        flags.append(
            f"Diverges from community by {abs(cal_p - community):.3f} "
            f"(ours={cal_p:.3f}, community={community:.3f})"
        )

    # Low effective N
    if record.aggregation.effective_n < 3 and len(record.runs) >= 3:
        flags.append("Low effective N — runs may be correlated")

    # Platform anomaly
    if record.platform_signals.anomaly_detected:
        flags.append("Platform behavioral anomaly detected")

    # Verification-based flags
    v = record.verification
    if v.enabled and v.stages:
        for stage in v.stages:
            # Flag disagreements
            for entry in stage.verifications:
                if entry.status == "success" and entry.assessment == "disagree":
                    flags.append(
                        f"External models DISAGREE with {stage.stage}: "
                        f"{entry.provider}/{entry.model}"
                    )

            # Flag disputes (partial agreement from multiple models)
            partial_count = sum(
                1 for e in stage.verifications
                if e.status == "success" and e.assessment == "partial"
            )
            if partial_count > 1:
                flags.append(
                    f"Multiple external models partially dispute {stage.stage}"
                )

            # Flag high-vulnerability challenges
            for challenge in stage.challenges:
                if challenge.status == "success" and challenge.vulnerability == "high":
                    flags.append(
                        f"HIGH vulnerability challenge on {stage.stage}: "
                        f"{challenge.provider}/{challenge.model}"
                    )

            # Flag verification gaps (providers that failed)
            failed_count = sum(
                1 for e in stage.verifications
                if e.status != "success"
            )
            if failed_count > 0:
                flags.append(
                    f"Verification gap: {failed_count} provider(s) failed "
                    f"for {stage.stage}"
                )

    return flags
