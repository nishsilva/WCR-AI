# apps/web/pages/2_Census_Explorer.py

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from wcr_agent.analysis.compare_groups import (
    compare_groups,
    compare_by_birth_region,
    compare_by_death_region,
    compare_by_record_status,
    compare_by_duplicate_flag,
    compare_early_vs_late_period,
    compare_small_vs_large_rings,
)

from wcr_agent.plotting.comparisons import (
    plot_group_metric_bar,
    plot_group_metric_dot,
    plot_group_metric_box_from_raw,
    plot_group_metric_violin_from_raw,
    plot_two_metric_scatter,
    get_plottable_comparison_value_columns,
)

# -----------------------------------------------------------------------------
# Make sure "src/" is importable when Streamlit runs from project root
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from wcr_agent.data_access.census import load_census, get_census_summary
from wcr_agent.analysis.filter_census import filter_rings
from wcr_agent.analysis.summarize_census import (
    summarize_subset,
    summarize_yearly_counts,
)
from wcr_agent.analysis.yearly_counts import (
    birth_yearly_counts,
    death_yearly_counts,
    compare_birth_vs_death_yearly_counts,
)
from wcr_agent.plotting.distributions import (
    plot_histogram,
    plot_boxplot,
    plot_yearly_counts_bar,
    plot_birth_vs_death_counts,
)
from wcr_agent.plotting.maps import (
    plot_birth_locations,
    plot_death_locations,
    plot_birth_and_death_locations,
    plot_birth_to_death_segments,
    plot_displacement_bubble_map,
)


st.set_page_config(page_title="WCR Census Explorer", layout="wide")
st.title("WCR Census Explorer")
st.caption("Explore the Warm Core Ring census dataset with filters, maps, and summary plots.")


@st.cache_data(show_spinner=False)
def load_census_data() -> pd.DataFrame:
    return load_census()


def fmt_number(value, digits: int = 1):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}"


def fmt_int(value):
    if value is None or pd.isna(value):
        return "—"
    return f"{int(value):,}"


def build_download_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["date_first_seen", "date_last_seen"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
try:
    df = load_census_data()
    dataset_summary = get_census_summary()
except Exception as exc:
    st.error(f"Failed to load processed census data: {exc}")
    st.stop()

if df.empty:
    st.warning("The processed census dataset is empty.")
    st.stop()


# -----------------------------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")

birth_min = pd.to_datetime(df["date_first_seen"], errors="coerce").min()
birth_max = pd.to_datetime(df["date_first_seen"], errors="coerce").max()

death_min = pd.to_datetime(df["date_last_seen"], errors="coerce").min()
death_max = pd.to_datetime(df["date_last_seen"], errors="coerce").max()

area_min = float(df["area_km2"].min()) if "area_km2" in df.columns else 0.0
area_max = float(df["area_km2"].max()) if "area_km2" in df.columns else 1.0

life_valid = pd.to_numeric(df["lifetime_days"], errors="coerce").dropna()
life_min = int(life_valid.min()) if not life_valid.empty else 0
life_max = int(life_valid.max()) if not life_valid.empty else 1

disp_valid = pd.to_numeric(df["displacement_km"], errors="coerce").dropna()
disp_min = float(disp_valid.min()) if not disp_valid.empty else 0.0
disp_max = float(disp_valid.max()) if not disp_valid.empty else 1.0

lat_birth_min = float(df["lat_birth"].min())
lat_birth_max = float(df["lat_birth"].max())
lon_birth_min = float(df["lon_birth"].min())
lon_birth_max = float(df["lon_birth"].max())

lat_death_valid = pd.to_numeric(df["lat_death"], errors="coerce").dropna()
lon_death_valid = pd.to_numeric(df["lon_death"], errors="coerce").dropna()
lat_death_min = float(lat_death_valid.min()) if not lat_death_valid.empty else -90.0
lat_death_max = float(lat_death_valid.max()) if not lat_death_valid.empty else 90.0
lon_death_min = float(lon_death_valid.min()) if not lon_death_valid.empty else -180.0
lon_death_max = float(lon_death_valid.max()) if not lon_death_valid.empty else 180.0

available_statuses = sorted(df["record_status"].dropna().astype(str).unique().tolist())
available_duplicate_options = ["All", "Only duplicates", "Only non-duplicates"]

with st.sidebar:
    st.subheader("Date filters")
    birth_range = st.date_input(
        "Birth date range",
        value=(birth_min.date(), birth_max.date()) if pd.notna(birth_min) and pd.notna(birth_max) else None,
    )
    death_range = st.date_input(
        "Absorption date range",
        value=(death_min.date(), death_max.date()) if pd.notna(death_min) and pd.notna(death_max) else None,
    )

    st.subheader("Scalar filters")
    area_range = st.slider(
        "Area (km²)",
        min_value=float(area_min),
        max_value=float(area_max),
        value=(float(area_min), float(area_max)),
    )
    lifetime_range = st.slider(
        "Lifetime (days)",
        min_value=int(life_min),
        max_value=int(life_max),
        value=(int(life_min), int(life_max)),
    )
    displacement_range = st.slider(
        "Displacement (km)",
        min_value=float(disp_min),
        max_value=float(disp_max),
        value=(float(disp_min), float(disp_max)),
    )

    st.subheader("Birth location bounding box")
    birth_lat_range = st.slider(
        "Birth latitude",
        min_value=float(lat_birth_min),
        max_value=float(lat_birth_max),
        value=(float(lat_birth_min), float(lat_birth_max)),
    )
    birth_lon_range = st.slider(
        "Birth longitude",
        min_value=float(lon_birth_min),
        max_value=float(lon_birth_max),
        value=(float(lon_birth_min), float(lon_birth_max)),
    )

    st.subheader("Death location bounding box")
    death_lat_range = st.slider(
        "Death latitude",
        min_value=float(lat_death_min),
        max_value=float(lat_death_max),
        value=(float(lat_death_min), float(lat_death_max)),
    )
    death_lon_range = st.slider(
        "Death longitude",
        min_value=float(lon_death_min),
        max_value=float(lon_death_max),
        value=(float(lon_death_min), float(lon_death_max)),
    )

    st.subheader("Record filters")
    selected_statuses = st.multiselect(
        "Record status",
        options=available_statuses,
        default=available_statuses,
    )
    duplicate_mode = st.radio(
        "Duplicate rows",
        options=available_duplicate_options,
        index=0,
    )

    sort_by = st.selectbox(
        "Sort filtered table by",
        options=[
            "date_first_seen",
            "date_last_seen",
            "area_km2",
            "lifetime_days",
            "displacement_km",
            "ring_id",
            "row_id",
        ],
        index=0,
    )
    ascending = st.toggle("Sort ascending", value=True)


# -----------------------------------------------------------------------------
# Normalize date inputs
# -----------------------------------------------------------------------------
def unpack_date_range(value):
    if value is None:
        return None, None

    if isinstance(value, tuple) and len(value) == 2:
        return pd.to_datetime(value[0]), pd.to_datetime(value[1])

    if isinstance(value, list) and len(value) == 2:
        return pd.to_datetime(value[0]), pd.to_datetime(value[1])

    # single date selected
    one = pd.to_datetime(value)
    return one, one


birth_start, birth_end = unpack_date_range(birth_range)
death_start, death_end = unpack_date_range(death_range)

duplicate_flag = None
if duplicate_mode == "Only duplicates":
    duplicate_flag = True
elif duplicate_mode == "Only non-duplicates":
    duplicate_flag = False

selected_statuses = selected_statuses if selected_statuses else None


# -----------------------------------------------------------------------------
# Apply filters
# -----------------------------------------------------------------------------
try:
    filtered = filter_rings(
        df,
        birth_date_start=birth_start,
        birth_date_end=birth_end,
        death_date_start=death_start,
        death_date_end=death_end,
        min_area_km2=area_range[0],
        max_area_km2=area_range[1],
        min_lifetime_days=lifetime_range[0],
        max_lifetime_days=lifetime_range[1],
        min_displacement_km=displacement_range[0],
        max_displacement_km=displacement_range[1],
        min_lat_birth=birth_lat_range[0],
        max_lat_birth=birth_lat_range[1],
        min_lon_birth=birth_lon_range[0],
        max_lon_birth=birth_lon_range[1],
        min_lat_death=death_lat_range[0],
        max_lat_death=death_lat_range[1],
        min_lon_death=death_lon_range[0],
        max_lon_death=death_lon_range[1],
        record_statuses=selected_statuses,
        duplicate_ring_id_flag=duplicate_flag,
        sort_by=sort_by,
        ascending=ascending,
    )
except Exception as exc:
    st.error(f"Failed to apply filters: {exc}")
    st.stop()


# -----------------------------------------------------------------------------
# Top metrics
# -----------------------------------------------------------------------------
overview = summarize_subset(filtered)["overview"] if not filtered.empty else {
    "n_rows": 0,
    "n_unique_ring_id": 0,
    "birth_date_min": None,
    "birth_date_max": None,
    "death_date_min": None,
    "death_date_max": None,
    "n_duplicate_rows": 0,
    "n_missing_absorption_dates": 0,
}

numeric_summary = summarize_subset(filtered)["numeric"] if not filtered.empty else {}

st.subheader("Filtered overview")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Rows", fmt_int(overview.get("n_rows")))
m2.metric("Unique ring IDs", fmt_int(overview.get("n_unique_ring_id")))
m3.metric(
    "Median lifetime (days)",
    fmt_number(
        numeric_summary.get("lifetime_days", {}).get("median") if numeric_summary else None,
        digits=1,
    ),
)
m4.metric(
    "Median area (km²)",
    fmt_number(
        numeric_summary.get("area_km2", {}).get("median") if numeric_summary else None,
        digits=1,
    ),
)
m5.metric(
    "Median displacement (km)",
    fmt_number(
        numeric_summary.get("displacement_km", {}).get("median") if numeric_summary else None,
        digits=1,
    ),
)

with st.expander("Dataset-level summary"):
    st.json(dataset_summary)


if filtered.empty:
    st.warning("No rows match the current filter settings.")
    st.stop()


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Maps", "Distributions", "Yearly Counts", "Data Table", "Comparisons"]
)

# -----------------------------------------------------------------------------
# Maps tab
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Maps")

    map_col1, map_col2 = st.columns([1, 1])
    with map_col1:
        map_type = st.selectbox(
            "Map type",
            options=[
                "Birth locations",
                "Death locations",
                "Birth and death locations",
                "Birth-to-death segments",
                "Birth locations sized by displacement",
            ],
            index=0,
        )
    with map_col2:
        color_options = [None, "record_status", "lifetime_days", "area_km2", "displacement_km"]
        color_label = st.selectbox(
            "Color by",
            options=["None"] + [c for c in color_options if c is not None],
            index=0,
        )
        color_col = None if color_label == "None" else color_label

    max_map_rows = st.slider(
        "Maximum rows to plot on map",
        min_value=50,
        max_value=min(1000, max(100, len(filtered))),
        value=min(300, len(filtered)),
        step=25,
    )

    map_df = filtered.head(max_map_rows).copy()

    try:
        if map_type == "Birth locations":
            fig = plot_birth_locations(
                map_df,
                color_col=color_col,
                title=f"WCR Birth Locations (n={len(map_df):,})",
            )
        elif map_type == "Death locations":
            fig = plot_death_locations(
                map_df,
                color_col=color_col,
                title=f"WCR Death Locations (n={len(map_df):,})",
            )
        elif map_type == "Birth and death locations":
            fig = plot_birth_and_death_locations(
                map_df,
                title=f"WCR Birth and Death Locations (n={len(map_df):,})",
            )
        elif map_type == "Birth-to-death segments":
            fig = plot_birth_to_death_segments(
                map_df,
                color_by=color_col,
                title=f"WCR Birth-to-Death Segments (n={len(map_df):,})",
                max_segments=max_map_rows,
            )
        else:
            fig = plot_displacement_bubble_map(
                map_df,
                color_col=color_col if color_col is not None else "lifetime_days",
                title=f"Birth Locations Sized by Displacement (n={len(map_df):,})",
            )

        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Could not render map: {exc}")


# -----------------------------------------------------------------------------
# Distributions tab
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Distributions")

    dist_col1, dist_col2 = st.columns([1, 1])
    with dist_col1:
        variable = st.selectbox(
            "Variable",
            options=["lifetime_days", "area_km2", "displacement_km", "radius_equiv_km"],
            index=0,
        )
    with dist_col2:
        plot_kind = st.selectbox(
            "Plot type",
            options=["Histogram", "Boxplot"],
            index=0,
        )

    if plot_kind == "Histogram":
        nbins = st.slider("Number of bins", min_value=5, max_value=80, value=25)
        try:
            fig = plot_histogram(
                filtered,
                column=variable,
                nbins=nbins,
                title=f"{variable} distribution",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not render histogram: {exc}")
    else:
        try:
            fig = plot_boxplot(
                filtered,
                column=variable,
                title=f"{variable} boxplot",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not render boxplot: {exc}")

    st.markdown("### Numeric summary")
    num_stats = summarize_subset(filtered)["numeric"].get(variable, {})
    if num_stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", fmt_number(num_stats.get("mean")))
        c2.metric("Median", fmt_number(num_stats.get("median")))
        c3.metric("Min", fmt_number(num_stats.get("min")))
        c4.metric("Max", fmt_number(num_stats.get("max")))
    else:
        st.info("No numeric summary available for the selected variable.")


# -----------------------------------------------------------------------------
# Yearly counts tab
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Yearly counts")

    yc1, yc2 = st.columns([1, 1])
    with yc1:
        count_mode = st.selectbox(
            "Count mode",
            options=["Birth counts", "Death counts", "Birth vs death counts"],
            index=0,
        )
    with yc2:
        chart_mode = st.selectbox(
            "Chart style",
            options=["Bar", "Line"],
            index=0,
        )

    try:
        if count_mode == "Birth counts":
            counts_df = birth_yearly_counts(filtered)
            if chart_mode == "Bar":
                fig = plot_yearly_counts_bar(
                    counts_df,
                    year_column="birth_year",
                    count_column="count",
                    title="Birth counts by year",
                )
            else:
                from wcr_agent.plotting.distributions import plot_yearly_counts_line
                fig = plot_yearly_counts_line(
                    counts_df,
                    year_column="birth_year",
                    count_column="count",
                    title="Birth counts by year",
                )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(counts_df, use_container_width=True, hide_index=True)

        elif count_mode == "Death counts":
            counts_df = death_yearly_counts(filtered)
            if chart_mode == "Bar":
                fig = plot_yearly_counts_bar(
                    counts_df,
                    year_column="death_year",
                    count_column="count",
                    title="Absorption counts by year",
                )
            else:
                from wcr_agent.plotting.distributions import plot_yearly_counts_line
                fig = plot_yearly_counts_line(
                    counts_df,
                    year_column="death_year",
                    count_column="count",
                    title="Absorption counts by year",
                )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(counts_df, use_container_width=True, hide_index=True)

        else:
            compare_df = compare_birth_vs_death_yearly_counts(filtered)
            fig = plot_birth_vs_death_counts(compare_df, title="Birth vs absorption counts by year")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error(f"Could not render yearly counts: {exc}")


# -----------------------------------------------------------------------------
# Data table tab
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Filtered data table")

    show_cols_default = [
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
        "record_status",
        "duplicate_ring_id_flag",
    ]
    show_cols_default = [c for c in show_cols_default if c in filtered.columns]

    selected_cols = st.multiselect(
        "Columns to show",
        options=list(filtered.columns),
        default=show_cols_default,
    )

    display_df = build_download_df(filtered[selected_cols] if selected_cols else filtered)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered CSV",
        data=csv_bytes,
        file_name="wcr_census_filtered.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------------
# Comparisons tab
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("Group comparisons")

    compare_mode = st.selectbox(
        "Comparison mode",
        options=[
            "By birth region",
            "By death region",
            "By record status",
            "By duplicate flag",
            "Early vs late period",
            "Small vs large rings",
            "Custom group column",
        ],
        index=0,
    )

    metric_options = [
        c for c in [
            "area_km2",
            "radius_equiv_km",
            "lifetime_days",
            "displacement_km",
            "lat_birth",
            "lon_birth",
            "lat_death",
            "lon_death",
        ]
        if c in filtered.columns
    ]

    selected_metrics = st.multiselect(
        "Metrics to compare",
        options=metric_options,
        default=[m for m in ["lifetime_days", "area_km2", "displacement_km"] if m in metric_options],
    )

    agg_options = ["count", "mean", "median", "std", "min", "max"]
    selected_aggs = st.multiselect(
        "Aggregations",
        options=agg_options,
        default=["count", "mean", "median"],
    )

    sort_candidates = [
        "group_count",
        "lifetime_days_mean",
        "lifetime_days_median",
        "area_km2_mean",
        "area_km2_median",
        "displacement_km_mean",
        "displacement_km_median",
    ]
    selected_sort = st.selectbox(
        "Sort output by",
        options=["Auto"] + sort_candidates,
        index=0,
    )
    sort_by = None if selected_sort == "Auto" else selected_sort

    raw_group_col_for_distribution = None

    try:
        compare_df = None

        if compare_mode == "By birth region":
            raw_group_col_for_distribution = "birth_region"
            compare_df = compare_by_birth_region(
                filtered,
                metrics=selected_metrics or None,
                aggregations=selected_aggs or None,
                sort_by=sort_by or "group_count",
                ascending=False,
            )

        elif compare_mode == "By death region":
            raw_group_col_for_distribution = "death_region"
            compare_df = compare_by_death_region(
                filtered,
                metrics=selected_metrics or None,
                aggregations=selected_aggs or None,
                sort_by=sort_by or "group_count",
                ascending=False,
            )

        elif compare_mode == "By record status":
            raw_group_col_for_distribution = "record_status"
            compare_df = compare_by_record_status(
                filtered,
                metrics=selected_metrics or None,
                aggregations=selected_aggs or None,
                sort_by=sort_by or "group_count",
                ascending=False,
            )

        elif compare_mode == "By duplicate flag":
            raw_group_col_for_distribution = "duplicate_ring_id_flag"
            compare_df = compare_by_duplicate_flag(
                filtered,
                metrics=selected_metrics or None,
                aggregations=selected_aggs or None,
                sort_by=sort_by or "group_count",
                ascending=False,
            )

        elif compare_mode == "Early vs late period":
            valid_birth_years = pd.to_numeric(filtered["birth_year"], errors="coerce").dropna()
            if valid_birth_years.empty:
                st.warning("No valid birth_year values are available for this comparison.")
                st.stop()

            split_year = st.slider(
                "Split year",
                min_value=int(valid_birth_years.min()),
                max_value=int(valid_birth_years.max()),
                value=int(valid_birth_years.median()),
            )

            # build a labeled raw frame too, for box/violin charts from raw data
            raw_group_col_for_distribution = "time_period"
            comparison_raw_df = filtered.copy()
            comparison_raw_df["birth_year"] = pd.to_numeric(comparison_raw_df["birth_year"], errors="coerce")
            comparison_raw_df = comparison_raw_df.dropna(subset=["birth_year"]).copy()
            comparison_raw_df["time_period"] = comparison_raw_df["birth_year"].apply(
                lambda y: "early" if int(y) < split_year else "late"
            )

            compare_df = compare_early_vs_late_period(
                filtered,
                split_year=split_year,
                metrics=selected_metrics or None,
                aggregations=selected_aggs or None,
            )

        elif compare_mode == "Small vs large rings":
            valid_areas = pd.to_numeric(filtered["area_km2"], errors="coerce").dropna()
            if valid_areas.empty:
                st.warning("No valid area_km2 values are available for this comparison.")
                st.stop()

            threshold_area = st.slider(
                "Area threshold (km²)",
                min_value=float(valid_areas.min()),
                max_value=float(valid_areas.max()),
                value=float(valid_areas.median()),
            )

            raw_group_col_for_distribution = "size_class"
            comparison_raw_df = filtered.copy()
            comparison_raw_df["area_km2"] = pd.to_numeric(comparison_raw_df["area_km2"], errors="coerce")
            comparison_raw_df = comparison_raw_df.dropna(subset=["area_km2"]).copy()
            comparison_raw_df["size_class"] = comparison_raw_df["area_km2"].apply(
                lambda a: "small" if float(a) < threshold_area else "large"
            )

            compare_df = compare_small_vs_large_rings(
                filtered,
                threshold_area_km2=threshold_area,
                metrics=selected_metrics or None,
                aggregations=selected_aggs or None,
            )

        else:
            custom_group_options = [
                c for c in [
                    "birth_region",
                    "death_region",
                    "record_status",
                    "duplicate_ring_id_flag",
                    "birth_year",
                    "death_year",
                    "birth_month",
                    "death_month",
                ]
                if c in filtered.columns
            ]

            custom_group_col = st.selectbox(
                "Group by column",
                options=custom_group_options,
                index=0,
            )
            raw_group_col_for_distribution = custom_group_col

            compare_df = compare_groups(
                filtered,
                group_by=custom_group_col,
                metrics=selected_metrics or None,
                aggregations=selected_aggs or None,
                sort_by=sort_by,
                ascending=False,
            )

        if compare_df is None or compare_df.empty:
            st.info("No comparison output is available for the current selection.")
        else:
            st.markdown("### Comparison table")
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

            csv_bytes = compare_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download comparison CSV",
                data=csv_bytes,
                file_name="wcr_group_comparison.csv",
                mime="text/csv",
            )

            st.markdown("### Comparison chart")

            group_col = compare_df.columns[0]
            plottable_cols = get_plottable_comparison_value_columns(compare_df)

            chart_type = st.selectbox(
                "Chart type",
                options=[
                    "Bar chart",
                    "Dot chart",
                    "Two-metric scatter",
                    "Raw boxplot",
                    "Raw violin plot",
                ],
                index=0,
            )

            if chart_type in {"Bar chart", "Dot chart"}:
                default_value_col = "group_count" if "group_count" in plottable_cols else (plottable_cols[0] if plottable_cols else None)

                if default_value_col is None:
                    st.info("No plottable numeric columns are available in the comparison output.")
                else:
                    value_col = st.selectbox(
                        "Value column",
                        options=plottable_cols,
                        index=plottable_cols.index(default_value_col) if default_value_col in plottable_cols else 0,
                    )

                    if chart_type == "Bar chart":
                        fig = plot_group_metric_bar(
                            compare_df,
                            group_col=group_col,
                            value_col=value_col,
                            title=f"{value_col} by {group_col}",
                        )
                    else:
                        fig = plot_group_metric_dot(
                            compare_df,
                            group_col=group_col,
                            value_col=value_col,
                            title=f"{value_col} by {group_col}",
                        )

                    st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Two-metric scatter":
                metric_cols = [c for c in plottable_cols if c != "group_count"]
                if len(metric_cols) < 2:
                    st.info("Need at least two numeric comparison columns for a two-metric scatter.")
                else:
                    x_col = st.selectbox("X column", options=metric_cols, index=0)
                    y_default_index = 1 if len(metric_cols) > 1 else 0
                    y_col = st.selectbox("Y column", options=metric_cols, index=y_default_index)

                    size_col_options = ["None"] + plottable_cols
                    size_col_choice = st.selectbox(
                        "Bubble size column",
                        options=size_col_options,
                        index=size_col_options.index("group_count") if "group_count" in size_col_options else 0,
                    )
                    size_col = None if size_col_choice == "None" else size_col_choice

                    fig = plot_two_metric_scatter(
                        compare_df,
                        group_col=group_col,
                        x_col=x_col,
                        y_col=y_col,
                        size_col=size_col,
                        title=f"{y_col} vs {x_col} by {group_col}",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            else:
                # Raw distribution plots require a raw grouped dataframe
                metric_for_raw = st.selectbox(
                    "Raw metric column",
                    options=metric_options,
                    index=metric_options.index("lifetime_days") if "lifetime_days" in metric_options else 0,
                )

                if compare_mode in {"Early vs late period", "Small vs large rings"}:
                    raw_df_for_plot = comparison_raw_df.copy()
                else:
                    raw_df_for_plot = filtered.copy()

                if raw_group_col_for_distribution is None or raw_group_col_for_distribution not in raw_df_for_plot.columns:
                    st.info("Raw grouped distributions are not available for this comparison.")
                else:
                    if chart_type == "Raw boxplot":
                        fig = plot_group_metric_box_from_raw(
                            raw_df_for_plot,
                            group_col=raw_group_col_for_distribution,
                            metric_col=metric_for_raw,
                            title=f"{metric_for_raw} by {raw_group_col_for_distribution}",
                        )
                    else:
                        fig = plot_group_metric_violin_from_raw(
                            raw_df_for_plot,
                            group_col=raw_group_col_for_distribution,
                            metric_col=metric_for_raw,
                            title=f"{metric_for_raw} by {raw_group_col_for_distribution}",
                        )

                    st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.error(f"Could not compute group comparison: {exc}")