"""Predict page — search, configure, and run the prediction pipeline."""

from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from app import get_completed_slot, get_config, get_pipeline_queue, get_registry_cache
from components.state import (
    KEY_PIPELINE_CLIENT,
    KEY_PIPELINE_QUESTION,
    KEY_PIPELINE_RECORD,
    KEY_PIPELINE_STATE,
    KEY_SEARCH_RESULTS,
    KEY_SELECTED_PLATFORM,
    init_state,
)
from src.models import HumanReview, Platform, SearchPersona

init_state()

st.title("Predict")

config = get_config()

# ---------------------------------------------------------------------------
# Platform clients (lazy import to avoid mandatory API keys at boot)
# ---------------------------------------------------------------------------

def _get_platform_client(platform_name: str):
    """Return a platform client or None if not configured."""
    try:
        if platform_name == "metaculus":
            from src.platforms.metaculus import MetaculusClient
            return MetaculusClient(config)
        elif platform_name == "polymarket":
            from src.platforms.polymarket import PolymarketClient
            return PolymarketClient(config)
        elif platform_name == "kalshi":
            from src.platforms.kalshi import KalshiClient
            return KalshiClient(config)
    except Exception as e:
        st.warning(f"Could not initialise {platform_name} client: {e}")
    return None


# ---------------------------------------------------------------------------
# Threading helpers
# ---------------------------------------------------------------------------

def _run_pipeline(config, question, client, num_runs, personas, q, completed_slot):
    """Run full pipeline in background thread.  DA[#1] fix: per-thread Orchestrator."""
    try:
        # TODO: replace with actual Orchestrator.predict_until_review() when TA finalises API
        # For now wraps orchestrator.predict() with auto_submit=True then queues sentinel
        from src.pipeline.orchestrator import Orchestrator
        orchestrator = Orchestrator(config)  # NEW per-thread instance

        q.put("Starting pipeline...")
        q.put(f"Question: {question.title[:80]}")
        q.put(f"Runs: {num_runs}, Personas: {[p.value for p in personas]}")

        # TODO: switch to orchestrator.predict_until_review() once available
        record = orchestrator.predict(
            question=question,
            platform_client=client,
            num_runs=num_runs,
            personas=personas,
            auto_submit=True,  # skip CLI human review — UI handles it
        )

        completed_slot.clear()
        completed_slot.append(record)
    except Exception as e:
        q.put(f"ERROR: {e}")
        completed_slot.clear()
    finally:
        q.put(None)  # sentinel — signals completion


# ---------------------------------------------------------------------------
# Left / Right layout
# ---------------------------------------------------------------------------

left_col, right_col = st.columns([35, 65])

# ===========================================================================
# LEFT: Platform + search
# ===========================================================================
with left_col:
    st.subheader("Question")

    platform_name = st.selectbox(
        "Platform",
        options=[p.value for p in Platform],
        key=KEY_SELECTED_PLATFORM,
    )

    input_mode = st.radio("Find question by", ["Search", "ID", "URL"], horizontal=True)

    client = _get_platform_client(platform_name)

    if input_mode == "Search":
        search_query = st.text_input("Search query", placeholder="e.g. US election 2026")
        if st.button("Search", key="search_btn"):
            if client is None:
                st.error(f"No client available for {platform_name}. Check API key.")
            elif search_query.strip():
                with st.spinner("Searching..."):
                    try:
                        results = client.get_questions(limit=20, search=search_query.strip())
                        st.session_state[KEY_SEARCH_RESULTS] = results
                        st.session_state[KEY_PIPELINE_CLIENT] = client
                    except Exception as e:
                        st.error(f"Search failed: {e}")
                        st.session_state[KEY_SEARCH_RESULTS] = []

        results = st.session_state.get(KEY_SEARCH_RESULTS, [])
        if results:
            import pandas as pd
            rows = [
                {
                    "Title": getattr(q, "title", "")[:70],
                    "Close": getattr(q, "close_date", "")[:10],
                    "Community P": getattr(q, "community_prediction", None),
                    "_question_id": getattr(q, "question_id", ""),
                }
                for q in results
            ]
            df = pd.DataFrame(rows)

            event = st.dataframe(
                df[["Title", "Close", "Community P"]],
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row",
                key="search_results_table",
            )
            sel = event.selection.get("rows", []) if event.selection else []
            if sel:
                idx = sel[0]
                selected_q = results[idx]
                st.session_state[KEY_PIPELINE_QUESTION] = selected_q
                st.success(f"Selected: {getattr(selected_q, 'title', '')[:60]}")

    elif input_mode == "ID":
        qid = st.text_input("Question ID")
        if st.button("Fetch", key="fetch_id_btn"):
            if client is None:
                st.error(f"No client available for {platform_name}.")
            elif qid.strip():
                with st.spinner("Fetching..."):
                    try:
                        q = client.get_question(qid.strip())
                        st.session_state[KEY_PIPELINE_QUESTION] = q
                        st.session_state[KEY_PIPELINE_CLIENT] = client
                        st.success(f"Loaded: {getattr(q, 'title', '')[:60]}")
                    except Exception as e:
                        st.error(f"Fetch failed: {e}")

    elif input_mode == "URL":
        url = st.text_input("Question URL")
        if st.button("Fetch from URL", key="fetch_url_btn"):
            # Extract question ID from URL — platform-specific heuristic
            if not url.strip():
                st.warning("Enter a URL.")
            elif client is None:
                st.error(f"No client available for {platform_name}.")
            else:
                # Naive ID extraction: last path segment
                qid_from_url = url.rstrip("/").split("/")[-1]
                with st.spinner("Fetching..."):
                    try:
                        q = client.get_question(qid_from_url)
                        st.session_state[KEY_PIPELINE_QUESTION] = q
                        st.session_state[KEY_PIPELINE_CLIENT] = client
                        st.success(f"Loaded: {getattr(q, 'title', '')[:60]}")
                    except Exception as e:
                        st.error(f"Fetch failed: {e}")

# ===========================================================================
# RIGHT: Config + pipeline + inline review
# ===========================================================================
with right_col:
    st.subheader("Run Configuration")

    num_runs = st.number_input("Number of runs", min_value=1, max_value=10, value=1, step=1)

    all_personas = [p.value for p in SearchPersona]
    sel_persona_vals = st.multiselect(
        "Search personas",
        options=all_personas,
        default=[SearchPersona.NEWS.value],
        help="One persona will be used per run (rotated if fewer than num_runs)",
    )

    verification_toggle = st.toggle(
        "Enable cross-model verification",
        value=config.verification.enabled,
    )

    # Show selected question summary
    question = st.session_state.get(KEY_PIPELINE_QUESTION)
    pipeline_client = st.session_state.get(KEY_PIPELINE_CLIENT, client)
    pipeline_state = st.session_state.get(KEY_PIPELINE_STATE, "idle")

    if question:
        st.markdown("---")
        st.markdown(f"**Question:** {getattr(question, 'title', '')}")
        st.markdown(f"**Platform:** {getattr(question, 'platform', '')} &nbsp;|&nbsp; **Close:** {getattr(question, 'close_date', '—')[:10]}")
        community_p = getattr(question, "community_prediction", None)
        if community_p is not None:
            st.markdown(f"**Community prediction:** {community_p:.3f}")

    st.markdown("---")

    # Launch button
    can_launch = (
        question is not None
        and pipeline_client is not None
        and pipeline_state == "idle"
    )

    if st.button("Run Pipeline", disabled=not can_launch, type="primary"):
        # Override verification config from toggle
        config.verification.enabled = verification_toggle

        # Build personas list
        personas = [SearchPersona(v) for v in sel_persona_vals] if sel_persona_vals else [SearchPersona.NEWS]

        # Clear queue and slot
        q = get_pipeline_queue()
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
        completed_slot = get_completed_slot()
        completed_slot.clear()

        st.session_state[KEY_PIPELINE_STATE] = "running"
        st.session_state[KEY_PIPELINE_RECORD] = None

        thread = threading.Thread(
            target=_run_pipeline,
            args=(config, question, pipeline_client, int(num_runs), personas, q, completed_slot),
            daemon=True,
        )
        thread.start()
        st.rerun()

    # ---------------------------------------------------------------------------
    # Pipeline progress fragment
    # ---------------------------------------------------------------------------

    @st.fragment(run_every=1)
    def pipeline_progress_fragment():
        if st.session_state.get(KEY_PIPELINE_STATE) != "running":
            return
        q = get_pipeline_queue()
        msgs = []
        while True:
            try:
                msg = q.get_nowait()
                if msg is None:
                    # Thread complete — read from cache_resource slot (main thread safe)
                    slot = get_completed_slot()
                    if slot:
                        st.session_state[KEY_PIPELINE_RECORD] = slot[0]
                        st.session_state[KEY_PIPELINE_STATE] = "awaiting_review"
                        # Invalidate registry cache so Dashboard/Registry pick up new record
                        cache = get_registry_cache(config.registry_path)
                        cache.invalidate()
                    else:
                        st.session_state[KEY_PIPELINE_STATE] = "idle"
                    st.rerun()
                    return
                msgs.append(msg)
            except queue.Empty:
                break
        if msgs:
            with st.status("Running pipeline...", expanded=True):
                for m in msgs:
                    st.write(m)

    pipeline_progress_fragment()

    # ---------------------------------------------------------------------------
    # Inline review (visible when pipeline_state == "awaiting_review")
    # ---------------------------------------------------------------------------

    if st.session_state.get(KEY_PIPELINE_STATE) == "awaiting_review":
        record = st.session_state.get(KEY_PIPELINE_RECORD)

        if record:
            st.markdown("---")
            st.subheader("Human Review")

            cal = getattr(record, "calibration", None)
            cal_prob = getattr(cal, "calibrated_probability", 0.5) if cal else 0.5
            ci = getattr(cal, "confidence_interval", [0.0, 1.0]) if cal else [0.0, 1.0]

            st.markdown(f"**Calibrated probability:** {cal_prob:.3f}")
            st.markdown(f"**Confidence interval:** [{ci[0]:.3f}, {ci[1]:.3f}]")

            decision = st.radio(
                "Decision",
                options=["Accept", "Adjust", "Reject"],
                horizontal=True,
                key="review_decision",
            )

            adjusted_prob = None
            adjust_reasoning = ""

            if decision == "Adjust":
                adjusted_prob = st.slider(
                    "Adjusted probability",
                    min_value=0.01,
                    max_value=0.99,
                    value=float(cal_prob),
                    step=0.01,
                    key="review_adjusted_prob",
                )
                adjust_reasoning = st.text_area(
                    "Reasoning for adjustment",
                    placeholder="Why are you adjusting from the model estimate?",
                    key="review_reasoning",
                )

            flags_input = st.text_input(
                "Flags (comma-separated, optional)",
                placeholder="e.g. high_uncertainty, actor_conflict",
                key="review_flags",
            )

            if st.button("Confirm Review", type="primary", key="confirm_review_btn"):
                flags = [f.strip() for f in flags_input.split(",") if f.strip()] if flags_input else []

                if decision == "Accept":
                    human_review = HumanReview(
                        reviewed=True,
                        human_adjustment=None,
                        human_reasoning="Accepted",
                        flags_raised=flags,
                    )
                elif decision == "Adjust":
                    delta = (adjusted_prob - cal_prob) if adjusted_prob is not None else 0.0
                    human_review = HumanReview(
                        reviewed=True,
                        human_adjustment=round(delta, 4),
                        human_reasoning=adjust_reasoning or "Adjusted",
                        flags_raised=flags,
                    )
                else:  # Reject
                    human_review = HumanReview(
                        reviewed=True,
                        human_adjustment=None,
                        human_reasoning="REJECTED",
                        flags_raised=flags,
                    )

                record.human_review = human_review

                # TODO: call orchestrator.submit_after_review(record, human_review) when TA finalises API
                # For now: update the record in registry directly
                cache = get_registry_cache(config.registry_path)
                cache._store.update(record)
                cache.invalidate()

                st.session_state[KEY_PIPELINE_STATE] = "idle"
                st.session_state[KEY_PIPELINE_RECORD] = None
                st.success(f"Review submitted: {decision}")
                st.rerun()
