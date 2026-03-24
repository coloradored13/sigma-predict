"""Plotly chart builders for sigma-predict Streamlit app."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------

def calibration_curve(pairs: list[tuple[float, float]], n_bins: int = 10) -> go.Figure:
    """Decile bucket calibration plot vs perfect diagonal.

    Args:
        pairs: list of (predicted_prob, outcome) tuples
        n_bins: number of equal-width buckets
    """
    if not pairs:
        fig = go.Figure()
        fig.add_annotation(text="No resolved predictions yet", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Calibration Curve", xaxis_title="Predicted", yaxis_title="Actual")
        return fig

    bucket_size = 1.0 / n_bins
    buckets: dict[int, list[float]] = defaultdict(list)
    for pred, outcome in pairs:
        bucket = min(int(pred / bucket_size), n_bins - 1)
        buckets[bucket].append(outcome)

    xs, ys = [], []
    for b in range(n_bins):
        if b in buckets:
            mid = (b + 0.5) * bucket_size
            actual = sum(buckets[b]) / len(buckets[b])
            xs.append(mid)
            ys.append(actual)

    fig = go.Figure()

    # Perfect diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray"), name="Perfect calibration"
    ))

    # Over-confidence shading
    fig.add_trace(go.Scatter(
        x=[0, 0.5, 1, 0], y=[0, 0, 1, 0],
        fill="toself", fillcolor="rgba(255,100,100,0.08)",
        line=dict(width=0), name="Overconfident region", showlegend=True
    ))

    # Under-confidence shading
    fig.add_trace(go.Scatter(
        x=[0, 0.5, 1, 0], y=[0, 1, 1, 0],
        fill="toself", fillcolor="rgba(100,100,255,0.08)",
        line=dict(width=0), name="Underconfident region", showlegend=True
    ))

    # Actual calibration
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers",
        marker=dict(size=8), line=dict(color="#1f77b4", width=2),
        name="Actual calibration"
    ))

    fig.update_layout(
        title="Calibration Curve",
        xaxis_title="Predicted probability",
        yaxis_title="Actual frequency",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    return fig


# ---------------------------------------------------------------------------
# Brier timeline
# ---------------------------------------------------------------------------

def brier_timeline(records: list, window: int = 7) -> go.Figure:
    """Rolling average Brier score over time."""
    resolved = [
        r for r in records
        if getattr(r, "resolution", None) is not None
        and getattr(r.resolution, "brier_score", None) is not None
    ]
    resolved.sort(key=lambda r: getattr(r, "created_at", ""))

    if not resolved:
        fig = go.Figure()
        fig.add_annotation(text="No resolved predictions yet", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Brier Score Timeline")
        return fig

    dates = []
    scores = []
    rolling = []

    for r in resolved:
        dates.append(getattr(r, "created_at", ""))
        scores.append(r.resolution.brier_score)

    for i in range(len(scores)):
        start = max(0, i - window + 1)
        rolling.append(sum(scores[start:i + 1]) / (i - start + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=scores, mode="markers",
        marker=dict(size=5, color="rgba(100,150,200,0.5)"),
        name="Individual Brier"
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=rolling, mode="lines",
        line=dict(color="#1f77b4", width=2),
        name=f"{window}-prediction rolling avg"
    ))
    fig.update_layout(
        title="Brier Score Timeline",
        xaxis_title="Date",
        yaxis_title="Brier score (lower = better)",
    )
    return fig


# ---------------------------------------------------------------------------
# Domain Brier bar chart
# ---------------------------------------------------------------------------

def domain_brier_chart(records: list) -> go.Figure:
    """Average Brier score by domain."""
    domain_scores: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if getattr(r, "resolution", None) is None:
            continue
        brier = getattr(r.resolution, "brier_score", None)
        if brier is None:
            continue
        domain = getattr(r, "domain", "general") or "general"
        domain_scores[domain].append(brier)

    if not domain_scores:
        fig = go.Figure()
        fig.add_annotation(text="No resolved predictions yet", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Brier by Domain")
        return fig

    domains = list(domain_scores.keys())
    avgs = [sum(v) / len(v) for v in domain_scores.values()]
    counts = [len(v) for v in domain_scores.values()]

    fig = go.Figure(go.Bar(
        x=domains, y=avgs,
        text=[f"n={c}" for c in counts],
        textposition="outside",
        marker_color="#1f77b4",
    ))
    fig.update_layout(
        title="Average Brier Score by Domain",
        xaxis_title="Domain",
        yaxis_title="Avg Brier score",
    )
    return fig


# ---------------------------------------------------------------------------
# Cost stacked bar (by month)
# ---------------------------------------------------------------------------

def cost_stacked_bar(records: list) -> go.Figure:
    """Monthly total cost stacked bar."""
    monthly: dict[str, float] = defaultdict(float)
    for r in records:
        created = getattr(r, "created_at", "") or ""
        cost_obj = getattr(r, "cost", None)
        if not cost_obj or not created:
            continue
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            month_key = dt.strftime("%Y-%m")
        except (ValueError, TypeError):
            continue
        monthly[month_key] += getattr(cost_obj, "total_cost_usd", 0.0)

    if not monthly:
        fig = go.Figure()
        fig.add_annotation(text="No cost data yet", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Monthly Cost")
        return fig

    months = sorted(monthly.keys())
    costs = [monthly[m] for m in months]

    fig = go.Figure(go.Bar(x=months, y=costs, marker_color="#2ca02c"))
    fig.update_layout(
        title="Monthly API Cost (USD)",
        xaxis_title="Month",
        yaxis_title="Cost (USD)",
    )
    return fig


# ---------------------------------------------------------------------------
# Error class frequency
# ---------------------------------------------------------------------------

def error_class_freq_chart(records: list) -> go.Figure:
    """Frequency bar chart of error classes across resolved predictions."""
    freq: dict[str, int] = defaultdict(int)
    for r in records:
        res = getattr(r, "resolution", None)
        if res is None:
            continue
        for ec in getattr(res, "error_class_auto", []):
            freq[ec] += 1
        for ec in getattr(res, "error_class_manual", []):
            freq[ec] += 1

    if not freq:
        fig = go.Figure()
        fig.add_annotation(text="No error class data yet", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Error Class Frequency")
        return fig

    labels = sorted(freq.keys(), key=lambda k: freq[k], reverse=True)
    values = [freq[l] for l in labels]

    fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#d62728"))
    fig.update_layout(
        title="Error Class Frequency",
        xaxis_title="Error class",
        yaxis_title="Count",
    )
    return fig


# ---------------------------------------------------------------------------
# Human adjustment scatter
# ---------------------------------------------------------------------------

def adjustment_scatter(records: list) -> go.Figure:
    """Scatter: human adjustment vs Brier score delta (post-review improvement)."""
    xs, ys, labels = [], [], []
    for r in records:
        res = getattr(r, "resolution", None)
        if res is None:
            continue
        brier = getattr(res, "brier_score", None)
        if brier is None:
            continue
        review = getattr(r, "human_review", None)
        if review is None:
            continue
        adj = getattr(review, "human_adjustment", None)
        if adj is None:
            continue
        cal_prob = getattr(getattr(r, "calibration", None), "calibrated_probability", None)
        if cal_prob is None:
            continue
        # Brier delta: lower is better, so a negative delta = improvement
        outcome = getattr(res, "outcome", None)
        if outcome is None:
            continue
        pre_brier = (cal_prob - outcome) ** 2
        post_prob = max(0.01, min(0.99, cal_prob + adj))
        post_brier = (post_prob - outcome) ** 2
        delta = post_brier - pre_brier  # negative = improved

        xs.append(adj)
        ys.append(delta)
        labels.append(getattr(r, "prediction_id", "")[:8])

    if not xs:
        fig = go.Figure()
        fig.add_annotation(text="No human adjustment data yet", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Human Adjustment vs Brier Delta")
        return fig

    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers",
        text=labels, hoverinfo="text+x+y",
        marker=dict(size=8, color="#9467bd"),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Human Adjustment vs Brier Delta",
        xaxis_title="Human adjustment (delta from calibrated)",
        yaxis_title="Brier delta (negative = improved)",
    )
    return fig


# ---------------------------------------------------------------------------
# Probability waterfall
# ---------------------------------------------------------------------------

def probability_waterfall(run) -> go.Figure:
    """Waterfall: base_rate → adjustment → pre_mortem → final."""
    base_rate = getattr(run, "base_rate_used", 0.5)
    inside_adj = getattr(run, "inside_view_adjustment", 0.0)
    pre_mortem_delta = getattr(run, "pre_mortem_delta", 0.0)
    final = getattr(run, "probability", 0.5)

    # Derive stages
    after_adj = base_rate + inside_adj
    after_pre = after_adj + pre_mortem_delta

    stages = ["Base rate", "Inside view adjustment", "Pre-mortem delta", "Final"]
    measures = ["absolute", "relative", "relative", "total"]
    values = [base_rate, inside_adj, pre_mortem_delta, final]

    colors = []
    for i, (m, v) in enumerate(zip(measures, values)):
        if m == "absolute" or m == "total":
            colors.append("#1f77b4")
        elif v >= 0:
            colors.append("#2ca02c")
        else:
            colors.append("#d62728")

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=stages,
        y=values,
        connector=dict(line=dict(color="gray")),
        decreasing=dict(marker_color="#d62728"),
        increasing=dict(marker_color="#2ca02c"),
        totals=dict(marker_color="#1f77b4"),
    ))
    fig.update_layout(
        title="Probability Waterfall",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
    )
    return fig
