"""Dashboard page — active predictions overview."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from app import get_config, get_registry_cache
from components.state import (
    KEY_PIPELINE_STATE,
    KEY_SELECTED_PREDICTION_ID,
    init_state,
)

init_state()

st.title("Dashboard")

config = get_config()
cache = get_registry_cache(config.registry_path)
records = cache.load()


def _in_current_month(record, now: datetime) -> bool:
    created = getattr(record, "created_at", "") or ""
    try:
        dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        return dt.year == now.year and dt.month == now.month
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# 4-metric strip
# ---------------------------------------------------------------------------

active = [r for r in records if getattr(r, "resolution", None) is None]
resolved = [r for r in records if getattr(r, "resolution", None) is not None]

brier_scores = [
    r.resolution.brier_score
    for r in resolved
    if getattr(r.resolution, "brier_score", None) is not None
]
avg_brier = sum(brier_scores) / len(brier_scores) if brier_scores else None

# Monthly spend — current calendar month
now = datetime.now(timezone.utc)
monthly_cost = sum(
    getattr(getattr(r, "cost", None), "total_cost_usd", 0.0)
    for r in records
    if _in_current_month(r, now)
)

pending_review = [
    r for r in records
    if not getattr(getattr(r, "human_review", None), "reviewed", False)
    and getattr(r, "resolution", None) is None
]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active predictions", len(active))
col2.metric("Avg Brier (resolved)", f"{avg_brier:.3f}" if avg_brier is not None else "—")
col3.metric("Monthly spend", f"${monthly_cost:.2f}")
col4.metric("Pending review", len(pending_review))

st.divider()

# ---------------------------------------------------------------------------
# Active predictions table
# ---------------------------------------------------------------------------

st.subheader("Active Predictions")

if not active:
    st.info("No active predictions yet. Use the Predict page to get started.")
else:
    # Build display rows
    rows = []
    for r in active:
        close_date_str = ""
        days_remaining = None
        question = getattr(r, "question_text", "")
        close_date_str = getattr(r, "close_date", "") if hasattr(r, "close_date") else ""

        # Try to get close_date from the record (may not exist on PredictionRecord)
        # horizon_days is a stored approximation
        horizon = getattr(r, "horizon_days", None)

        rows.append({
            "ID": getattr(r, "prediction_id", "")[:8],
            "Question": question[:80] + ("..." if len(question) > 80 else ""),
            "Platform": getattr(r, "platform", ""),
            "Domain": getattr(r, "domain", ""),
            "Probability": getattr(getattr(r, "calibration", None), "calibrated_probability", 0.5),
            "Horizon (days)": horizon if horizon is not None else "—",
            "Created": getattr(r, "created_at", "")[:10],
            "_prediction_id": getattr(r, "prediction_id", ""),
        })

    import pandas as pd
    df = pd.DataFrame(rows)

    display_cols = ["ID", "Question", "Platform", "Domain", "Probability", "Horizon (days)", "Created"]

    event = st.dataframe(
        df[display_cols],
        use_container_width=True,
        column_config={
            "Probability": st.column_config.ProgressColumn(
                "Probability",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            )
        },
        on_select="rerun",
        selection_mode="single-row",
        key="dashboard_table",
    )

    # Row click → navigate to Review
    selected_rows = event.selection.get("rows", []) if event.selection else []
    if selected_rows:
        idx = selected_rows[0]
        prediction_id = df.iloc[idx]["_prediction_id"]
        st.session_state[KEY_SELECTED_PREDICTION_ID] = prediction_id
        st.switch_page("pages/3_Review.py")

# ---------------------------------------------------------------------------
# Auto-rerun during pipeline run
# ---------------------------------------------------------------------------

if st.session_state.get(KEY_PIPELINE_STATE) == "running":
    st.rerun()

# Manual refresh
if st.button("Refresh"):
    cache.invalidate()
    st.rerun()
