"""Session state key constants and initializer for sigma-predict Streamlit app."""

from __future__ import annotations

import streamlit as st

# Session state keys
KEY_SELECTED_PREDICTION_ID = "selected_prediction_id"   # str | None
KEY_PIPELINE_STATE = "pipeline_state"                   # "idle" | "running" | "awaiting_review"
KEY_PIPELINE_RECORD = "pipeline_record"                 # PredictionRecord | None
KEY_PIPELINE_QUESTION = "pipeline_question"             # PlatformQuestion | None
KEY_PIPELINE_CLIENT = "pipeline_client"                 # PlatformClient | None
KEY_SEARCH_RESULTS = "search_results"                   # list[PlatformQuestion]
KEY_SELECTED_PLATFORM = "selected_platform"             # str
KEY_REGISTRY_FILTERS = "registry_filters"               # dict
KEY_SCOREBOARD_DATE_RANGE = "scoreboard_date_range"     # tuple | None
KEY_SCOREBOARD_DOMAINS = "scoreboard_domains"           # list[str]

_DEFAULTS: dict = {
    KEY_SELECTED_PREDICTION_ID: None,
    KEY_PIPELINE_STATE: "idle",
    KEY_PIPELINE_RECORD: None,
    KEY_PIPELINE_QUESTION: None,
    KEY_PIPELINE_CLIENT: None,
    KEY_SEARCH_RESULTS: [],
    KEY_SELECTED_PLATFORM: "metaculus",
    KEY_REGISTRY_FILTERS: {},
    KEY_SCOREBOARD_DATE_RANGE: None,
    KEY_SCOREBOARD_DOMAINS: [],
}


def init_state() -> None:
    """Set session state defaults for any missing keys."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default
