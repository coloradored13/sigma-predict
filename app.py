"""sigma-predict Streamlit app entry point."""

from __future__ import annotations

import queue
import sys
from pathlib import Path

# Ensure project root is on path so `src` imports work
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from components.cache import RegistryCache
from components.state import init_state
from src.config import Config
from src.registry.store import RegistryStore

st.set_page_config(
    page_title="sigma-predict",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Shared cache_resource singletons
# ---------------------------------------------------------------------------

@st.cache_resource
def get_config() -> Config:
    return Config()


@st.cache_resource
def get_registry_store(path: str) -> RegistryStore:
    return RegistryStore(path)


@st.cache_resource
def get_registry_cache(path: str) -> RegistryCache:
    store = get_registry_store(path)
    return RegistryCache(store)


@st.cache_resource
def get_pipeline_queue() -> queue.Queue:
    return queue.Queue()


@st.cache_resource
def get_completed_slot() -> list:
    """Single-item list used as a thread→UI bridge for completed PredictionRecord."""
    return []


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def main() -> None:
    init_state()

    config = get_config()

    pages = [
        st.Page("pages/1_Dashboard.py", title="Dashboard", icon="📊"),
        st.Page("pages/2_Predict.py", title="Predict", icon="🔮"),
        st.Page("pages/3_Review.py", title="Review", icon="🔍"),
        st.Page("pages/4_Scoreboard.py", title="Scoreboard", icon="📈"),
        st.Page("pages/5_Registry.py", title="Registry", icon="🗂️"),
    ]

    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()
