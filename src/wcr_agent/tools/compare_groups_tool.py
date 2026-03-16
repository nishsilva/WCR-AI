# src/wcr_agent/tools/compare_groups_tool.py

from __future__ import annotations

import pandas as pd

from wcr_agent.analysis.compare_groups import (
    compare_by_birth_region,
    compare_by_death_region,
    compare_by_record_status,
    compare_by_duplicate_flag,
    compare_early_vs_late_period,
    compare_small_vs_large_rings,
    compare_groups,
)
from wcr_agent.plotting.comparisons import (
    plot_group_metric_bar,
    plot_group_metric_box_from_raw,
)


def run_compare_groups_tool(
    df: pd.DataFrame,
    *,
    comparison_mode: str,
    metric: str = "lifetime_days",
    aggregations: list[str] | None = None,
    custom_group_col: str | None = None,
) -> dict:
    """
    Build a comparison table and companion charts for a filtered subset.
    """
    if aggregations is None:
        aggregations = ["count", "mean", "median"]

    raw_group_col = None
    compare_df = None
    raw_df_for_plot = df.copy()
    comparison_note = None

    if comparison_mode == "birth_region":
        raw_group_col = "birth_region"
        compare_df = compare_by_birth_region(
            df,
            metrics=[metric],
            aggregations=aggregations,
            sort_by="group_count",
            ascending=False,
        )

    elif comparison_mode == "death_region":
        raw_group_col = "death_region"
        compare_df = compare_by_death_region(
            df,
            metrics=[metric],
            aggregations=aggregations,
            sort_by="group_count",
            ascending=False,
        )

    elif comparison_mode == "record_status":
        raw_group_col = "record_status"
        compare_df = compare_by_record_status(
            df,
            metrics=[metric],
            aggregations=aggregations,
            sort_by="group_count",
            ascending=False,
        )

    elif comparison_mode == "duplicate_flag":
        raw_group_col = "duplicate_ring_id_flag"
        compare_df = compare_by_duplicate_flag(
            df,
            metrics=[metric],
            aggregations=aggregations,
            sort_by="group_count",
            ascending=False,
        )

    elif comparison_mode == "early_vs_late":
        birth_years = pd.to_numeric(df["birth_year"], errors="coerce").dropna()
        if birth_years.empty:
            raise ValueError("No valid birth_year values available for early_vs_late comparison.")

        split_year = int(birth_years.median())
        raw_group_col = "time_period"

        raw_df_for_plot = df.copy()
        raw_df_for_plot["birth_year"] = pd.to_numeric(raw_df_for_plot["birth_year"], errors="coerce")
        raw_df_for_plot = raw_df_for_plot.dropna(subset=["birth_year"]).copy()
        raw_df_for_plot["time_period"] = raw_df_for_plot["birth_year"].apply(
            lambda y: "early" if int(y) < split_year else "late"
        )

        compare_df = compare_early_vs_late_period(
            df,
            split_year=split_year,
            metrics=[metric],
            aggregations=aggregations,
        )
        comparison_note = f"Split year used: {split_year}"

    elif comparison_mode == "small_vs_large":
        areas = pd.to_numeric(df["area_km2"], errors="coerce").dropna()
        if areas.empty:
            raise ValueError("No valid area_km2 values available for small_vs_large comparison.")

        threshold = float(areas.median())
        raw_group_col = "size_class"

        raw_df_for_plot = df.copy()
        raw_df_for_plot["area_km2"] = pd.to_numeric(raw_df_for_plot["area_km2"], errors="coerce")
        raw_df_for_plot = raw_df_for_plot.dropna(subset=["area_km2"]).copy()
        raw_df_for_plot["size_class"] = raw_df_for_plot["area_km2"].apply(
            lambda a: "small" if float(a) < threshold else "large"
        )

        compare_df = compare_small_vs_large_rings(
            df,
            threshold_area_km2=threshold,
            metrics=[metric],
            aggregations=aggregations,
        )
        comparison_note = f"Area threshold used: {threshold:,.1f} km²"

    elif comparison_mode == "custom":
        if not custom_group_col:
            raise ValueError("custom_group_col is required when comparison_mode='custom'.")

        raw_group_col = custom_group_col
        compare_df = compare_groups(
            df,
            group_by=custom_group_col,
            metrics=[metric],
            aggregations=aggregations,
            sort_by="group_count",
            ascending=False,
        )

    else:
        raise ValueError(f"Unsupported comparison_mode: {comparison_mode}")

    if compare_df is None or compare_df.empty:
        return {
            "tool_name": "compare_groups",
            "comparison_mode": comparison_mode,
            "metric": metric,
            "table": compare_df,
            "bar_figure": None,
            "box_figure": None,
            "data": df,
            "note": comparison_note,
        }

    group_col = compare_df.columns[0]
    preferred_value_col = f"{metric}_mean"
    value_col = preferred_value_col if preferred_value_col in compare_df.columns else "group_count"

    bar_fig = plot_group_metric_bar(
        compare_df,
        group_col=group_col,
        value_col=value_col,
        title=f"{value_col} by {group_col}",
    )

    box_fig = None
    if raw_group_col in raw_df_for_plot.columns and metric in raw_df_for_plot.columns:
        box_fig = plot_group_metric_box_from_raw(
            raw_df_for_plot,
            group_col=raw_group_col,
            metric_col=metric,
            title=f"{metric} distribution by {raw_group_col}",
        )

    return {
        "tool_name": "compare_groups",
        "comparison_mode": comparison_mode,
        "metric": metric,
        "table": compare_df,
        "bar_figure": bar_fig,
        "box_figure": box_fig,
        "data": df,
        "note": comparison_note,
    }