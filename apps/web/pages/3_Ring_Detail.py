# apps/web/pages/3_Ring_Detail.py

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

from wcr_agent.data_access.census import (
    load_census,
    get_rows_by_ring_id,
    get_ring_by_row_id,
    get_duplicate_groups,
)
from wcr_agent.plotting.maps import (
    plot_birth_and_death_locations,
    plot_birth_to_death_segments,
)
from wcr_agent.plotting.distributions import (
    plot_histogram,
    plot_boxplot,
)


st.set_page_config(page_title="WCR Ring Detail", layout="wide")
st.title("WCR Ring Detail")
st.caption("Inspect a single record or all rows associated with a duplicated ring ID.")


@st.cache_data(show_spinner=False)
def load_census_data() -> pd.DataFrame:
    return load_census()


def format_dates_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["date_first_seen", "date_last_seen"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def format_scalar(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}"


def safe_int_str(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{int(value):,}"


def describe_differences(df_group: pd.DataFrame) -> pd.DataFrame:
    """
    For a duplicate group, show which columns vary across rows.
    """
    if df_group.empty or len(df_group) < 2:
        return pd.DataFrame(columns=["column", "n_distinct_non_null", "values"])

    cols_to_check = [
        "ring_id",
        "date_first_seen",
        "lat_birth",
        "lon_birth",
        "area_km2",
        "date_last_seen",
        "lat_death",
        "lon_death",
        "lifetime_days",
        "displacement_km",
        "record_status",
    ]
    cols_to_check = [c for c in cols_to_check if c in df_group.columns]

    records = []
    for col in cols_to_check:
        values = df_group[col].dropna()
        n_unique = values.nunique()

        if n_unique > 1:
            display_values = []
            for v in values.unique().tolist():
                if isinstance(v, pd.Timestamp):
                    display_values.append(v.strftime("%Y-%m-%d"))
                else:
                    display_values.append(str(v))

            records.append(
                {
                    "column": col,
                    "n_distinct_non_null": int(n_unique),
                    "values": " | ".join(display_values),
                }
            )

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
try:
    df = load_census_data()
except Exception as exc:
    st.error(f"Failed to load processed census data: {exc}")
    st.stop()

if df.empty:
    st.warning("The processed census dataset is empty.")
    st.stop()

duplicate_df = get_duplicate_groups()
duplicate_ring_ids = (
    sorted(duplicate_df["ring_id"].dropna().astype(str).unique().tolist())
    if not duplicate_df.empty
    else []
)

all_ring_ids = sorted(df["ring_id"].dropna().astype(str).unique().tolist())

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header("Selection")

selection_mode = st.sidebar.radio(
    "Select by",
    options=["row_id", "ring_id", "duplicate ring_id"],
    index=0,
)

selected_row_id = None
selected_ring_id = None

if selection_mode == "row_id":
    row_ids = sorted(pd.to_numeric(df["row_id"], errors="coerce").dropna().astype(int).tolist())
    selected_row_id = st.sidebar.selectbox("Choose row_id", options=row_ids)

elif selection_mode == "ring_id":
    selected_ring_id = st.sidebar.selectbox("Choose ring_id", options=all_ring_ids)

else:
    if duplicate_ring_ids:
        selected_ring_id = st.sidebar.selectbox("Choose duplicate ring_id", options=duplicate_ring_ids)
    else:
        st.sidebar.info("No duplicate ring IDs found in the dataset.")

# -----------------------------------------------------------------------------
# Retrieve selection
# -----------------------------------------------------------------------------
try:
    if selection_mode == "row_id":
        record = get_ring_by_row_id(int(selected_row_id))
        selected_df = pd.DataFrame([record])
        page_title_value = f"row_id = {int(selected_row_id)}"
    else:
        selected_df = get_rows_by_ring_id(str(selected_ring_id))
        page_title_value = f"ring_id = {selected_ring_id}"
except Exception as exc:
    st.error(f"Could not load selection: {exc}")
    st.stop()

if selected_df.empty:
    st.warning("No matching records found.")
    st.stop()

# -----------------------------------------------------------------------------
# Header and overview
# -----------------------------------------------------------------------------
st.subheader(page_title_value)

is_duplicate_group = len(selected_df) > 1

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows in selection", safe_int_str(len(selected_df)))
c2.metric(
    "Unique ring IDs",
    safe_int_str(selected_df["ring_id"].nunique(dropna=True)) if "ring_id" in selected_df.columns else "—",
)
c3.metric(
    "Median lifetime (days)",
    format_scalar(pd.to_numeric(selected_df.get("lifetime_days"), errors="coerce").median(), digits=1)
    if "lifetime_days" in selected_df.columns
    else "—",
)
c4.metric(
    "Median area (km²)",
    format_scalar(pd.to_numeric(selected_df.get("area_km2"), errors="coerce").median(), digits=1)
    if "area_km2" in selected_df.columns
    else "—",
)
c5.metric(
    "Median displacement (km)",
    format_scalar(pd.to_numeric(selected_df.get("displacement_km"), errors="coerce").median(), digits=1)
    if "displacement_km" in selected_df.columns
    else "—",
)

if is_duplicate_group:
    st.warning("This ring_id has multiple rows in the source data. The table below shows all associated records.")

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Record Table", "Maps", "Comparison", "Download"]
)

# -----------------------------------------------------------------------------
# Record Table
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### Selected records")

    display_cols_default = [
        "row_id",
        "ring_id",
        "date_first_seen",
        "lat_birth",
        "lon_birth",
        "area_km2",
        "date_last_seen",
        "lat_death",
        "lon_death",
        "lifetime_days",
        "displacement_km",
        "bearing_birth_to_death",
        "record_status",
        "duplicate_ring_id_flag",
        "duplicate_group_size",
    ]
    display_cols_default = [c for c in display_cols_default if c in selected_df.columns]

    selected_cols = st.multiselect(
        "Columns to display",
        options=list(selected_df.columns),
        default=display_cols_default,
    )

    display_df = selected_df[selected_cols].copy() if selected_cols else selected_df.copy()
    display_df = format_dates_for_display(display_df)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    if is_duplicate_group:
        st.markdown("### Fields that differ across rows")
        diff_df = describe_differences(selected_df)
        if diff_df.empty:
            st.info("No differing non-null fields detected across duplicate rows.")
        else:
            st.dataframe(diff_df, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Maps
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### Spatial view")

    map_type = st.selectbox(
        "Map type",
        options=[
            "Birth and death locations",
            "Birth-to-death segments",
        ],
        index=0,
        key="detail_map_type",
    )

    try:
        if map_type == "Birth and death locations":
            fig = plot_birth_and_death_locations(
                selected_df,
                title=f"Birth and death locations: {page_title_value}",
            )
        else:
            fig = plot_birth_to_death_segments(
                selected_df,
                title=f"Birth-to-death segments: {page_title_value}",
                max_segments=len(selected_df),
            )

        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Could not render map: {exc}")

# -----------------------------------------------------------------------------
# Comparison / summaries
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### Selection summary")

    numeric_cols = [c for c in ["area_km2", "lifetime_days", "displacement_km", "radius_equiv_km"] if c in selected_df.columns]

    if numeric_cols:
        summary_rows = []
        for col in numeric_cols:
            s = pd.to_numeric(selected_df[col], errors="coerce").dropna()
            summary_rows.append(
                {
                    "variable": col,
                    "count_non_null": int(s.shape[0]),
                    "mean": None if s.empty else float(s.mean()),
                    "median": None if s.empty else float(s.median()),
                    "min": None if s.empty else float(s.min()),
                    "max": None if s.empty else float(s.max()),
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if len(selected_df) >= 2 and "lifetime_days" in selected_df.columns:
        try:
            st.markdown("### Lifetime distribution within selection")
            fig = plot_histogram(
                selected_df,
                column="lifetime_days",
                nbins=min(10, max(5, len(selected_df))),
                title=f"Lifetime distribution: {page_title_value}",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    if len(selected_df) >= 2 and "area_km2" in selected_df.columns:
        try:
            st.markdown("### Area boxplot within selection")
            fig = plot_boxplot(
                selected_df,
                column="area_km2",
                title=f"Area boxplot: {page_title_value}",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### Download selected records")

    download_df = format_dates_for_display(selected_df)
    csv_bytes = download_df.to_csv(index=False).encode("utf-8")

    default_name = (
        f"wcr_detail_row_{int(selected_row_id)}.csv"
        if selection_mode == "row_id"
        else f"wcr_detail_{str(selected_ring_id)}.csv"
    )

    st.download_button(
        label="Download selection as CSV",
        data=csv_bytes,
        file_name=default_name,
        mime="text/csv",
    )

    st.markdown("### Quick preview")
    st.dataframe(download_df, use_container_width=True, hide_index=True)