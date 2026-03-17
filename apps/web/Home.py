# apps/web/Home.py

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Make sure "src/" is importable when Streamlit runs from project root
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from wcr_agent.data_access.census import get_census_summary, load_census


st.set_page_config(
    page_title="WCR Agent",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 AI-powered WCR Explorer")
st.caption("Warm Core Ring census exploration and analysis workspace assisted by an LLM agent.")

st.markdown(
    """
**Warm Core Rings (WCRs)** are coherent clockwise rotating ocean eddies that pinches off from the Gulf Stream. They are characterized by warm water at their center. 
They play an important role in ocean circulation and heat transport.
"""
)

st.divider()


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return load_census()


try:
    df = load_data()
    summary = get_census_summary()
except Exception as exc:
    st.error(
        "Could not load the processed WCR census dataset.\n\n"
        f"Details: {exc}"
    )
    st.info(
        "Make sure you have already run:\n"
        "`python scripts/build_wcr_census.py`"
    )
    st.stop()


st.markdown(
    """
This app is based on a WCR census dataset that has been created by digitizing physical Gulf Stream Charts and recording the birth and demise characteristics of individual WCRs from 1980 to 2017.
Where each row represents one warm core ring record with:

- birth date and birth location
- absorption date and demise location
- area and derived metrics such as lifetime and displacement

Use the pages in the left sidebar to explore the dataset.
"""
)

# -----------------------------------------------------------------------------
# Top summary metrics
# -----------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{summary['n_rows']:,}")
c2.metric("Unique ring IDs", f"{summary['n_unique_ring_id']:,}")
c3.metric("Duplicate rows", f"{summary['n_duplicate_rows']:,}")
c4.metric("Missing absorption dates", f"{summary['n_missing_absorption_dates']:,}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Birth date min", summary["birth_date_min"] or "—")
c6.metric("Birth date max", summary["birth_date_max"] or "—")
c7.metric("Death date min", summary["death_date_min"] or "—")
c8.metric("Death date max", summary["death_date_max"] or "—")

st.divider()

# -----------------------------------------------------------------------------
# Navigation cards / quick guide
# -----------------------------------------------------------------------------
left, right = st.columns([1.15, 1])

with left:
    st.subheader("Application Capabilities")

    st.markdown(
        """
### 1. Chat
Natural-language interface for asking analysis questions.

Examples:
- “Show rings born after 2000 with lifetime over 30 days”
- “Map birth locations of the largest rings”
- “Compare birth and absorption counts by year”

### 2. Census Explorer
Interactive filters, maps, distributions, yearly counts, and downloadable tables.

### 3. Ring Detail
Focused view for inspecting an individual ring record or duplicated ring ID group.
"""
    )

with right:
    st.subheader("Current dataset snapshot")
    preview_cols = [
        "row_id",
        "ring_id",
        "date_first_seen",
        "lat_birth",
        "lon_birth",
        "area_km2",
        "date_last_seen",
        "lifetime_days",
        "displacement_km",
        "record_status",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    preview_df = df[preview_cols].head(10).copy()

    for col in ["date_first_seen", "date_last_seen"]:
        if col in preview_df.columns:
            preview_df[col] = pd.to_datetime(preview_df[col], errors="coerce").dt.strftime("%Y-%m-%d")

    st.dataframe(preview_df, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------------------------------------------------------
# Notes / assumptions
# -----------------------------------------------------------------------------
st.subheader("Notes")
st.markdown(
    """
- This dataset is currently treated as a **census table**, not a full trajectory table.
- Each row corresponds to one ring-level record.
- Birth and death coordinates are point summaries; they do not represent the full track.
- Duplicate `ring_id` values are preserved and flagged rather than silently removed.
"""
)

with st.expander("Full dataset summary"):
    st.json(summary)

st.divider()

st.subheader("Dataset Citation")
st.markdown(
    """
Gangopadhyay, A., Gawarkiewicz, G. (2020). Yearly census of Gulf Stream Warm Core Ring formation from 1980 to 2017. 
Biological and Chemical Oceanography Data Management Office (BCO-DMO). (Version 1) Version Date 2020-05-06. 
https://doi.org/10.26008/1912/bco-dmo.810182.1
"""
)

st.divider()
st.caption("Created by Nish Etige, Ph.D. | [Homepage](https://www.imnish.org) | [Email](mailto:your.nishetige@gmail.com) | [GitHub](https://github.com/nishsilva)")