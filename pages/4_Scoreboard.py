"""Scoreboard page — calibration and learning loop surfaces."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from app import get_config, get_registry_cache
from components.charts import (
    adjustment_scatter,
    brier_timeline,
    calibration_curve,
    cost_stacked_bar,
    domain_brier_chart,
    error_class_freq_chart,
)
from components.state import KEY_SCOREBOARD_DATE_RANGE, KEY_SCOREBOARD_DOMAINS, init_state

init_state()

st.title("Scoreboard")

config = get_config()
cache = get_registry_cache(config.registry_path)
records = cache.load()

resolved = [r for r in records if getattr(r, "resolution", None) is not None]

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

col1, col2 = st.columns(2)
with col1:
    date_range = st.date_input("Date range (created_at)", value=[], key="sb_date_range")
    st.session_state[KEY_SCOREBOARD_DATE_RANGE] = date_range if len(date_range) == 2 else None

with col2:
    all_domains = sorted({getattr(r, "domain", "") for r in records if getattr(r, "domain", "")})
    sel_domains = st.multiselect("Domains", all_domains, key="sb_domains")
    st.session_state[KEY_SCOREBOARD_DOMAINS] = sel_domains


def _passes_filters(record) -> bool:
    if sel_domains and getattr(record, "domain", "") not in sel_domains:
        return False
    if date_range and len(date_range) == 2:
        created = getattr(record, "created_at", "") or ""
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00")).date()
            if not (date_range[0] <= dt <= date_range[1]):
                return False
        except (ValueError, TypeError):
            pass
    return True


filtered_resolved = [r for r in resolved if _passes_filters(r)]
filtered_all = [r for r in records if _passes_filters(r)]

st.caption(f"{len(filtered_resolved)} resolved predictions in view")

st.divider()

# ---------------------------------------------------------------------------
# PRIMARY: Calibration curve
# ---------------------------------------------------------------------------

st.subheader("Calibration")

cal_pairs = []
for r in filtered_resolved:
    cal = getattr(r, "calibration", None)
    prob = getattr(cal, "calibrated_probability", None) if cal else None
    outcome = getattr(r.resolution, "outcome", None)
    if prob is not None and outcome is not None:
        cal_pairs.append((prob, float(outcome)))

fig_cal = calibration_curve(cal_pairs)
st.plotly_chart(fig_cal, use_container_width=True, key="cal_curve")

# Brier decomposition (from RCA — bracket notation, placeholder if insufficient data)
st.markdown("**Brier Decomposition**")
if len(cal_pairs) >= 10:
    try:
        # Attempt decomposition using sklearn or manual
        probs = [p for p, _ in cal_pairs]
        outcomes = [o for _, o in cal_pairs]
        n = len(probs)
        mean_outcome = sum(outcomes) / n

        # Murphy (1973) decomposition: BS = reliability - resolution + uncertainty
        # uncertainty = mean_outcome * (1 - mean_outcome)
        uncertainty = mean_outcome * (1 - mean_outcome)

        # reliability and resolution via binning
        from collections import defaultdict
        bins: dict[int, list] = defaultdict(list)
        for p, o in cal_pairs:
            b = min(int(p * 10), 9)
            bins[b].append((p, o))

        reliability = 0.0
        resolution = 0.0
        for b_vals in bins.values():
            n_k = len(b_vals)
            bar_p = sum(p for p, _ in b_vals) / n_k
            bar_o = sum(o for _, o in b_vals) / n_k
            reliability += (n_k / n) * (bar_p - bar_o) ** 2
            resolution += (n_k / n) * (bar_o - mean_outcome) ** 2

        decomp = {
            "reliability": round(reliability, 4),
            "resolution": round(resolution, 4),
            "uncertainty": round(uncertainty, 4),
        }

        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Reliability (lower=better)", decomp["reliability"])
        dc2.metric("Resolution (higher=better)", decomp["resolution"])
        dc3.metric("Uncertainty", decomp["uncertainty"])
    except Exception:
        st.info("Could not compute Brier decomposition.")
else:
    st.info(f"Brier decomposition requires ≥10 resolved predictions (have {len(cal_pairs)}).")

st.divider()

# ---------------------------------------------------------------------------
# Brier timeline
# ---------------------------------------------------------------------------

st.subheader("Brier Score Timeline")
fig_brier = brier_timeline(filtered_resolved)
st.plotly_chart(fig_brier, use_container_width=True, key="brier_timeline")

st.divider()

# ---------------------------------------------------------------------------
# Domain bar chart
# ---------------------------------------------------------------------------

st.subheader("Brier by Domain")
fig_domain = domain_brier_chart(filtered_resolved)
st.plotly_chart(fig_domain, use_container_width=True, key="domain_brier")

st.divider()

# ---------------------------------------------------------------------------
# Cost stacked bar
# ---------------------------------------------------------------------------

st.subheader("Monthly Cost")
fig_cost = cost_stacked_bar(filtered_all)
st.plotly_chart(fig_cost, use_container_width=True, key="cost_bar")

st.divider()

# ---------------------------------------------------------------------------
# Learning loop surfaces
# ---------------------------------------------------------------------------

st.subheader("Learning Loop")

ll1, ll2 = st.columns(2)

with ll1:
    st.markdown("**Error Class Frequency**")
    fig_err = error_class_freq_chart(filtered_resolved)
    st.plotly_chart(fig_err, use_container_width=True, key="error_freq")

with ll2:
    st.markdown("**Human Adjustment vs Brier Delta**")
    fig_adj = adjustment_scatter(filtered_resolved)
    st.plotly_chart(fig_adj, use_container_width=True, key="adj_scatter")

st.divider()

# Lessons feed table
st.markdown("**Lessons Feed**")
lessons_rows = []
for r in filtered_resolved:
    res = getattr(r, "resolution", None)
    if res is None:
        continue
    lessons = getattr(res, "structured_lessons", [])
    q_text = getattr(r, "question_text", "")[:60]
    pid = getattr(r, "prediction_id", "")[:8]
    for lesson in lessons:
        lessons_rows.append({
            "Prediction": pid,
            "Question": q_text,
            "Lesson": lesson,
        })

if lessons_rows:
    import pandas as pd
    st.dataframe(pd.DataFrame(lessons_rows), use_container_width=True)
else:
    st.info("No lessons recorded yet.")

st.divider()

# Pre-mortem effectiveness
st.markdown("**Pre-mortem Effectiveness**")
pm_changed = [
    r for r in filtered_resolved
    if any(getattr(run, "pre_mortem_changed_estimate", False) for run in getattr(r, "runs", []))
]
pm_total = [r for r in filtered_resolved if getattr(r, "runs", [])]

if pm_total:
    pct = 100 * len(pm_changed) / len(pm_total)
    st.metric("Pre-mortem changed estimate", f"{pct:.1f}%", help="% of predictions where pre-mortem altered the forecast")

    # Compare Brier: with vs without pre-mortem change
    with_change = [
        r.resolution.brier_score for r in pm_changed
        if getattr(r.resolution, "brier_score", None) is not None
    ]
    without_change = [
        r.resolution.brier_score for r in pm_total
        if r not in pm_changed and getattr(r.resolution, "brier_score", None) is not None
    ]
    if with_change and without_change:
        avg_with = sum(with_change) / len(with_change)
        avg_without = sum(without_change) / len(without_change)
        c1, c2 = st.columns(2)
        c1.metric("Avg Brier (pre-mortem changed)", f"{avg_with:.3f}")
        c2.metric("Avg Brier (no pre-mortem change)", f"{avg_without:.3f}")
else:
    st.info("No pre-mortem data available yet.")

st.divider()

# Actor signal test: Brier when L2/3 present vs absent
st.markdown("**Actor Signal Test** (Brier: L2/3 present vs absent)")

with_actors = []
without_actors = []
for r in filtered_resolved:
    decomp = getattr(r, "decomposition", None)
    aa = getattr(decomp, "actor_analysis", None) if decomp else None
    l23 = getattr(aa, "level_2_3_present", False) if aa else False
    brier = getattr(getattr(r, "resolution", None), "brier_score", None)
    if brier is None:
        continue
    if l23:
        with_actors.append(brier)
    else:
        without_actors.append(brier)

if with_actors or without_actors:
    c1, c2 = st.columns(2)
    if with_actors:
        c1.metric(f"Avg Brier (L2/3 present, n={len(with_actors)})", f"{sum(with_actors)/len(with_actors):.3f}")
    else:
        c1.metric("Avg Brier (L2/3 present)", "—")
    if without_actors:
        c2.metric(f"Avg Brier (L2/3 absent, n={len(without_actors)})", f"{sum(without_actors)/len(without_actors):.3f}")
    else:
        c2.metric("Avg Brier (L2/3 absent)", "—")
else:
    st.info("No actor analysis data available yet.")
