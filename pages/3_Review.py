"""Review page — detailed prediction inspection across 4 tabs."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from app import get_config, get_registry_cache
from components.charts import probability_waterfall
from components.state import KEY_SELECTED_PREDICTION_ID, init_state

init_state()

st.title("Review")

config = get_config()
cache = get_registry_cache(config.registry_path)
records = cache.load()

if not records:
    st.info("No predictions in the registry yet.")
    st.stop()

# ---------------------------------------------------------------------------
# Prediction selector
# ---------------------------------------------------------------------------

options = {
    getattr(r, "prediction_id", ""): f"{getattr(r, 'prediction_id', '')[:8]} — {getattr(r, 'question_text', '')[:60]}"
    for r in records
}
prediction_ids = list(options.keys())

# Pre-select from session state if set
default_id = st.session_state.get(KEY_SELECTED_PREDICTION_ID)
default_idx = prediction_ids.index(default_id) if default_id in prediction_ids else 0

selected_id = st.selectbox(
    "Prediction",
    options=prediction_ids,
    format_func=lambda x: options[x],
    index=default_idx,
    key="review_selector",
)
st.session_state[KEY_SELECTED_PREDICTION_ID] = selected_id

record = cache.get(selected_id)
if record is None:
    st.error("Prediction not found.")
    st.stop()

# ---------------------------------------------------------------------------
# 4 Tabs
# ---------------------------------------------------------------------------

tab_summary, tab_runs, tab_verification, tab_resolution = st.tabs(
    ["Summary", "Runs", "Verification", "Resolution"]
)

# ===========================================================================
# TAB: Summary
# ===========================================================================
with tab_summary:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Question:** {getattr(record, 'question_text', '—')}")
        st.markdown(f"**Platform:** {getattr(record, 'platform', '—')} &nbsp;|&nbsp; **Domain:** {getattr(record, 'domain', '—')} &nbsp;|&nbsp; **Type:** {getattr(record, 'question_type', '—')}")
        st.markdown(f"**Created:** {getattr(record, 'created_at', '—')[:19]}")
    with col2:
        cal = getattr(record, "calibration", None)
        cal_prob = getattr(cal, "calibrated_probability", None)
        agg = getattr(record, "aggregation", None)
        raw_agg = getattr(agg, "raw_aggregate", None)
        cost_obj = getattr(record, "cost", None)
        total_cost = getattr(cost_obj, "total_cost_usd", None)

        st.metric("Calibrated probability", f"{cal_prob:.3f}" if cal_prob is not None else "—")
        st.metric("Raw aggregate", f"{raw_agg:.3f}" if raw_agg is not None else "—")
        st.metric("Total cost", f"${total_cost:.4f}" if total_cost is not None else "—")

    # Decomposition expander
    decomp = getattr(record, "decomposition", None)
    if decomp:
        with st.expander("Decomposition", expanded=False):
            sub_qs = getattr(decomp, "sub_questions", [])
            if sub_qs:
                st.markdown("**Sub-questions:**")
                for sq in sub_qs:
                    st.markdown(f"- {sq}")
            ref_class = getattr(decomp, "reference_class", "")
            base_rate = getattr(decomp, "base_rate", None)
            base_source = getattr(decomp, "base_rate_source", "")
            if ref_class:
                st.markdown(f"**Reference class:** {ref_class}")
            if base_rate is not None:
                st.markdown(f"**Base rate:** {base_rate:.3f} ({base_source})")

    # Flags
    review = getattr(record, "human_review", None)
    flags = getattr(review, "flags_raised", []) if review else []
    if flags:
        for flag in flags:
            st.warning(f"Flag: {flag}")

    # Actor analysis (collapsed)
    actor_analysis = getattr(decomp, "actor_analysis", None) if decomp else None
    if actor_analysis and getattr(actor_analysis, "actors_involved", False):
        with st.expander("Actor Analysis", expanded=False):
            actors = getattr(actor_analysis, "actors", [])
            for actor in actors:
                name = getattr(actor, "name", "")
                level = getattr(actor, "level_label", "")
                motivation = getattr(actor, "motivation", "")
                influence = getattr(actor, "influence_estimate", "")
                st.markdown(f"**{name}** ({level}) — influence: {influence}")
                st.markdown(f"Motivation: {motivation}")
                implication = getattr(actor, "implication_for_estimate", "")
                if implication:
                    st.markdown(f"Implication: {implication}")
                st.divider()

            peripheral = getattr(actor_analysis, "peripheral_actors", [])
            if peripheral:
                st.markdown("**Peripheral actors:**")
                for pa in peripheral:
                    st.markdown(f"- **{getattr(pa, 'name', '')}** — {getattr(pa, 'feedback_mechanism', '')}")

            constraints = getattr(actor_analysis, "probability_space_constraints", [])
            if constraints:
                st.markdown("**Probability space constraints:**")
                for c in constraints:
                    st.markdown(f"- {c}")

# ===========================================================================
# TAB: Runs
# ===========================================================================
with tab_runs:
    runs = getattr(record, "runs", [])
    if not runs:
        st.info("No runs recorded.")
    else:
        # Comparison dataframe
        import pandas as pd
        run_rows = []
        for run in runs:
            run_rows.append({
                "Run ID": getattr(run, "run_id", "")[:8],
                "Persona": getattr(run, "search_persona", ""),
                "Model": getattr(run, "model", ""),
                "Base rate": getattr(run, "base_rate_used", None),
                "Adjustment": getattr(run, "inside_view_adjustment", None),
                "Pre-mortem Δ": getattr(run, "pre_mortem_delta", None),
                "Probability": getattr(run, "probability", None),
                "Confidence": getattr(run, "confidence_self_score", None),
            })
        st.dataframe(pd.DataFrame(run_rows), use_container_width=True)

        st.divider()

        # Per-run expanders with waterfall
        for run in runs:
            run_id = getattr(run, "run_id", "")[:8]
            prob = getattr(run, "probability", None)
            with st.expander(f"Run {run_id} — P={prob:.3f}" if prob is not None else f"Run {run_id}", expanded=False):
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.markdown(f"**Persona:** {getattr(run, 'search_persona', '—')}")
                    st.markdown(f"**Model:** {getattr(run, 'model', '—')}")
                    reasoning = getattr(run, "reasoning_chain", "")
                    if reasoning:
                        st.markdown("**Reasoning:**")
                        st.text(reasoning[:500] + ("..." if len(reasoning) > 500 else ""))
                    adj_reasoning = getattr(run, "adjustment_reasoning", "")
                    if adj_reasoning:
                        st.markdown("**Adjustment reasoning:**")
                        st.text(adj_reasoning[:300] + ("..." if len(adj_reasoning) > 300 else ""))
                    pm_scenario = getattr(run, "pre_mortem_scenario", "")
                    if pm_scenario:
                        st.markdown("**Pre-mortem scenario:**")
                        st.text(pm_scenario[:300] + ("..." if len(pm_scenario) > 300 else ""))
                with c2:
                    fig = probability_waterfall(run)
                    st.plotly_chart(fig, use_container_width=True, key=f"waterfall_{run_id}")

# ===========================================================================
# TAB: Verification
# ===========================================================================
with tab_verification:
    verification = getattr(record, "verification", None)
    if verification is None or not getattr(verification, "enabled", False):
        st.info("Verification was not enabled for this prediction.")
    else:
        stages = getattr(verification, "stages", [])
        if not stages:
            st.info("No verification stages recorded.")
        else:
            for stage in stages:
                stage_name = getattr(stage, "stage", "unknown")
                agreement = getattr(stage, "agreement", "no_data")
                summary = getattr(stage, "finding_summary", "")

                badge_color = {"unanimous": "green", "partial": "orange", "none": "red"}.get(agreement, "gray")

                with st.expander(f"{stage_name} — agreement: :{badge_color}[{agreement}]", expanded=False):
                    if summary:
                        st.markdown(f"**Finding:** {summary}")

                    verifications = getattr(stage, "verifications", [])
                    for v in verifications:
                        provider = getattr(v, "provider", "")
                        assessment = getattr(v, "assessment", "")
                        confidence = getattr(v, "confidence", "")
                        reasoning = getattr(v, "reasoning", "")
                        status = getattr(v, "status", "success")

                        badge = {"agree": "green", "disagree": "red", "partial": "orange", "uncertain": "gray"}.get(assessment, "gray")
                        st.markdown(f"**{provider}** — :{badge}[{assessment}] (confidence: {confidence}, status: {status})")
                        if reasoning:
                            st.text(reasoning[:300] + ("..." if len(reasoning) > 300 else ""))
                        counter = getattr(v, "counter_evidence", "")
                        if counter:
                            st.markdown(f"Counter evidence: {counter}")

                    challenges = getattr(stage, "challenges", [])
                    for ch in challenges:
                        provider = getattr(ch, "provider", "")
                        vulnerability = getattr(ch, "vulnerability", "")
                        counter_arg = getattr(ch, "counter_argument", "")
                        st.markdown(f"**Challenge from {provider}** — vulnerability: {vulnerability}")
                        if counter_arg:
                            st.text(counter_arg[:300] + ("..." if len(counter_arg) > 300 else ""))

# ===========================================================================
# TAB: Resolution
# ===========================================================================
with tab_resolution:
    resolution = getattr(record, "resolution", None)

    if resolution is None:
        st.info("This prediction has not been resolved yet.")
    else:
        outcome = getattr(resolution, "outcome", None)
        brier = getattr(resolution, "brier_score", None)
        log_score = getattr(resolution, "log_score", None)
        resolved_at = getattr(resolution, "resolved_at", "")

        col1, col2, col3 = st.columns(3)
        col1.metric("Outcome", str(outcome) if outcome is not None else "—")
        col2.metric("Brier score", f"{brier:.4f}" if brier is not None else "—")
        col3.metric("Log score", f"{log_score:.4f}" if log_score is not None else "—")

        if resolved_at:
            st.markdown(f"**Resolved:** {resolved_at[:19]}")

        error_auto = getattr(resolution, "error_class_auto", [])
        error_manual = getattr(resolution, "error_class_manual", [])
        if error_auto or error_manual:
            st.markdown(f"**Error classes (auto):** {', '.join(error_auto) if error_auto else '—'}")
            st.markdown(f"**Error classes (manual):** {', '.join(error_manual) if error_manual else '—'}")

        lessons = getattr(resolution, "structured_lessons", [])
        if lessons:
            st.markdown("**Lessons learned:**")
            for lesson in lessons:
                st.markdown(f"- {lesson}")

    st.divider()

    # Add lesson form
    with st.form("add_lesson_form"):
        st.markdown("**Add lesson**")
        new_lesson = st.text_area("Lesson text", placeholder="What did this prediction teach us?")
        submitted = st.form_submit_button("Add lesson")
        if submitted and new_lesson.strip():
            if resolution is None:
                st.warning("Cannot add lesson — prediction not yet resolved.")
            else:
                lessons_list = getattr(resolution, "structured_lessons", [])
                lessons_list.append(new_lesson.strip())
                resolution.structured_lessons = lessons_list
                store = cache._store
                store.update(record)
                cache.invalidate()
                st.success("Lesson added.")
                st.rerun()
