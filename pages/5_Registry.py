"""Registry page — full prediction history with filtering."""

from __future__ import annotations

import io
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import streamlit as st

from app import get_config, get_registry_cache
from components.state import KEY_SELECTED_PREDICTION_ID, init_state

init_state()

st.title("Registry")

config = get_config()
cache = get_registry_cache(config.registry_path)
records = cache.load()

# ---------------------------------------------------------------------------
# Filter bar
# ---------------------------------------------------------------------------

with st.expander("Filters", expanded=True):
    col1, col2, col3 = st.columns(3)

    all_platforms = sorted({getattr(r, "platform", "") for r in records if getattr(r, "platform", "")})
    all_domains = sorted({getattr(r, "domain", "") for r in records if getattr(r, "domain", "")})

    # Status options
    STATUS_OPTIONS = ["active", "resolved", "pending_review"]

    with col1:
        sel_platforms = st.multiselect("Platform", all_platforms)
        sel_domains = st.multiselect("Domain", all_domains)

    with col2:
        sel_statuses = st.multiselect("Status", STATUS_OPTIONS)
        date_range = st.date_input("Created date range", value=[], key="registry_date_range")

    with col3:
        text_search = st.text_input("Search question text")

# ---------------------------------------------------------------------------
# Filter logic
# ---------------------------------------------------------------------------

def _get_status(record) -> str:
    if getattr(record, "resolution", None) is not None:
        return "resolved"
    reviewed = getattr(getattr(record, "human_review", None), "reviewed", False)
    if not reviewed:
        return "pending_review"
    return "active"


def _date_in_range(record, start, end) -> bool:
    created = getattr(record, "created_at", "") or ""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(created.replace("Z", "+00:00")).date()
        return start <= dt <= end
    except (ValueError, TypeError):
        return True


filtered = records

if sel_platforms:
    filtered = [r for r in filtered if getattr(r, "platform", "") in sel_platforms]
if sel_domains:
    filtered = [r for r in filtered if getattr(r, "domain", "") in sel_domains]
if sel_statuses:
    filtered = [r for r in filtered if _get_status(r) in sel_statuses]
if text_search:
    q = text_search.lower()
    filtered = [r for r in filtered if q in getattr(r, "question_text", "").lower()]
if date_range and len(date_range) == 2:
    start, end = date_range
    filtered = [
        r for r in filtered
        if _date_in_range(r, start, end)
    ]


st.caption(f"{len(filtered)} records shown")

# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

if not filtered:
    st.info("No records match the current filters.")
else:
    rows = []
    for r in filtered:
        res = getattr(r, "resolution", None)
        brier = getattr(res, "brier_score", None) if res else None
        review = getattr(r, "human_review", None)
        flags = getattr(review, "flags_raised", []) if review else []
        cost_obj = getattr(r, "cost", None)
        total_cost = getattr(cost_obj, "total_cost_usd", 0.0) if cost_obj else 0.0

        q_text = getattr(r, "question_text", "")
        rows.append({
            "ID": getattr(r, "prediction_id", "")[:8],
            "Question": q_text[:80] + ("..." if len(q_text) > 80 else ""),
            "Platform": getattr(r, "platform", ""),
            "Domain": getattr(r, "domain", ""),
            "Created": getattr(r, "created_at", "")[:10],
            "Probability": getattr(getattr(r, "calibration", None), "calibrated_probability", 0.5),
            "Brier": f"{brier:.3f}" if brier is not None else "—",
            "Flags": len(flags),
            "Cost ($)": f"{total_cost:.4f}",
            "Status": _get_status(r),
            "_prediction_id": getattr(r, "prediction_id", ""),
        })

    df = pd.DataFrame(rows)
    display_cols = ["ID", "Question", "Platform", "Domain", "Created", "Probability", "Brier", "Flags", "Cost ($)", "Status"]

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
        key="registry_table",
    )

    selected_rows = event.selection.get("rows", []) if event.selection else []
    if selected_rows:
        idx = selected_rows[0]
        prediction_id = df.iloc[idx]["_prediction_id"]
        st.session_state[KEY_SELECTED_PREDICTION_ID] = prediction_id
        st.switch_page("pages/3_Review.py")

    # CSV download
    csv_buf = io.StringIO()
    df[display_cols].to_csv(csv_buf, index=False)
    st.download_button(
        label="Download CSV",
        data=csv_buf.getvalue(),
        file_name="sigma_predict_registry.csv",
        mime="text/csv",
    )
