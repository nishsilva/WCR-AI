# apps/web/pages/1_Chat.py

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Make sure "src/" is importable when Streamlit runs from project root
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from wcr_agent.data_access.census import load_census, get_census_summary
from wcr_agent.agent.orchestrator import orchestrate_query

st.set_page_config(page_title="WCR Chat", layout="wide")
st.title("WCR Chat")
st.caption("Prototype natural-language interface for the Warm Core Ring census dataset")


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return load_census()


try:
    df = load_data()
    dataset_summary = get_census_summary()
except Exception as exc:
    st.error(f"Failed to load processed census data: {exc}")
    st.info("Run `python scripts/build_wcr_census.py` first.")
    st.stop()

if df.empty:
    st.warning("The processed census dataset is empty.")
    st.stop()


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Ask a question about the WCR census dataset. "
                "This chat now routes requests through the orchestrator and tool registry."
            ),
        }
    ]

if "last_result_df" not in st.session_state:
    st.session_state.last_result_df = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def add_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def render_messages() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def build_preview_df(result_df: pd.DataFrame) -> pd.DataFrame:
    preview_cols = [
        "row_id",
        "ring_id",
        "date_first_seen",
        "area_km2",
        "date_last_seen",
        "lifetime_days",
        "displacement_km",
        "birth_region",
        "death_region",
        "record_status",
    ]
    preview_cols = [c for c in preview_cols if c in result_df.columns]

    preview_df = result_df[preview_cols].head(20).copy()
    for col in ["date_first_seen", "date_last_seen"]:
        if col in preview_df.columns:
            preview_df[col] = pd.to_datetime(preview_df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return preview_df


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Prototype status")
    st.write(
        "This page now uses the orchestrator, which sits between the UI and the reusable tool registry."
    )

    st.markdown("### Dataset snapshot")
    st.json(dataset_summary)

    if st.button("Clear chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared. Ask a question about the WCR census dataset.",
            }
        ]
        st.session_state.last_result_df = None
        st.rerun()


# -----------------------------------------------------------------------------
# Render chat history
# -----------------------------------------------------------------------------
render_messages()

# -----------------------------------------------------------------------------
# Suggested prompts
# -----------------------------------------------------------------------------
with st.expander("Example prompts"):
    st.markdown(
        """
- summary of the dataset
- summary of rings after 2000
- map births after 2000
- map deaths of complete records
- birth-to-death segments for duplicates
- lifetime distribution of large rings
- area distribution before 2000
- birth counts by year after 1990
- birth vs absorption counts
- compare lifetime by birth region
- compare area by record status
- compare displacement by death region
- compare lifetime by duplicate flag
- compare lifetime early vs late
- compare area small vs large rings
"""
    )

# -----------------------------------------------------------------------------
# Chat input
# -----------------------------------------------------------------------------
user_prompt = st.chat_input("Ask about the WCR census dataset...")

if user_prompt:
    add_message("user", user_prompt)

    with st.chat_message("user"):
        st.markdown(user_prompt)

    try:
        result = orchestrate_query(user_prompt, df, use_llm_parser=True)
        st.session_state.last_result_df = result.data
    except Exception as exc:
        error_text = f"I ran into an error while processing that request: `{exc}`"
        add_message("assistant", error_text)
        with st.chat_message("assistant"):
            st.markdown(error_text)
        st.stop()

    assistant_text = result.response_text
    add_message("assistant", assistant_text)

    with st.chat_message("assistant"):
        st.markdown(assistant_text)

        if result.note:
            st.caption(result.note)

        if result.figure is not None:
            st.plotly_chart(result.figure, width="stretch")

        if result.extra_figure is not None:
            st.plotly_chart(result.extra_figure, width="stretch")

        if result.table is not None:
            st.dataframe(result.table, width="stretch", hide_index=True)

        if result.data is not None:
            result_df = result.data
            st.caption(f"Matched rows: {len(result_df):,}")

            preview_df = build_preview_df(result_df)
            with st.expander("Preview matched rows"):
                st.dataframe(preview_df, width="stretch", hide_index=True)

            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download matched rows as CSV",
                data=csv_bytes,
                file_name="wcr_chat_results.csv",
                mime="text/csv",
            )