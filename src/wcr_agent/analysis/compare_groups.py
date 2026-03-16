# src/wcr_agent/analysis/compare_groups.py

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


DEFAULT_METRICS = [
    "area_km2",
    "radius_equiv_km",
    "lifetime_days",
    "displacement_km",
]

VALID_AGGREGATIONS = {"count", "mean", "median", "std", "min", "max"}


def _validate_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            "The dataframe is missing required columns:\n" + "\n".join(missing)
        )


def _clean_metrics(df: pd.DataFrame, metrics: Optional[Iterable[str]]) -> list[str]:
    metric_list = list(metrics) if metrics is not None else DEFAULT_METRICS
    metric_list = [m for m in metric_list if m in df.columns]

    if not metric_list:
        raise ValueError("No valid metric columns were found in the dataframe.")

    return metric_list


def _clean_aggregations(aggregations: Optional[Iterable[str]]) -> list[str]:
    agg_list = list(aggregations) if aggregations is not None else [
        "count",
        "mean",
        "median",
        "std",
        "min",
        "max",
    ]

    invalid = [a for a in agg_list if a not in VALID_AGGREGATIONS]
    if invalid:
        raise ValueError(
            f"Invalid aggregation(s): {invalid}. "
            f"Expected only: {sorted(VALID_AGGREGATIONS)}"
        )

    return agg_list


def compare_groups(
    df: pd.DataFrame,
    *,
    group_by: str,
    metrics: Optional[Iterable[str]] = None,
    aggregations: Optional[Iterable[str]] = None,
    dropna_group: bool = False,
    sort_by: Optional[str] = None,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Compare groups using summary statistics over one or more numeric metrics.

    Parameters
    ----------
    df
        Input dataframe.
    group_by
        Column to group by, e.g. 'birth_region', 'death_region', 'record_status',
        'duplicate_ring_id_flag', 'birth_year', etc.
    metrics
        Numeric columns to summarize.
    aggregations
        Summary statistics to compute. Supported:
        count, mean, median, std, min, max
    dropna_group
        If True, rows with missing group labels are dropped.
    sort_by
        Optional output column to sort by after aggregation.
        For flattened output, examples include:
        - "lifetime_days_mean"
        - "area_km2_median"
        - "group_count"
    ascending
        Sort order.

    Returns
    -------
    pd.DataFrame
        Flattened group comparison table.
    """
    _validate_columns(df, [group_by])

    metrics_clean = _clean_metrics(df, metrics)
    aggs_clean = _clean_aggregations(aggregations)

    work = df[[group_by] + metrics_clean].copy()

    if dropna_group:
        work = work.dropna(subset=[group_by])

    if work.empty:
        return pd.DataFrame(columns=[group_by])

    # Convert metrics to numeric safely
    for metric in metrics_clean:
        work[metric] = pd.to_numeric(work[metric], errors="coerce")

    grouped = work.groupby(group_by, dropna=not dropna_group)

    agg_dict = {metric: aggs_clean for metric in metrics_clean}
    out = grouped.agg(agg_dict)

    # Flatten MultiIndex columns
    out.columns = [f"{metric}_{agg}" for metric, agg in out.columns]
    out = out.reset_index()

    # Add explicit group size
    group_counts = (
        work.groupby(group_by, dropna=not dropna_group)
        .size()
        .rename("group_count")
        .reset_index()
    )
    out = out.merge(group_counts, on=group_by, how="left")

    # Reorder to put group_count after group label
    ordered_cols = [group_by, "group_count"] + [c for c in out.columns if c not in {group_by, "group_count"}]
    out = out[ordered_cols]

    if sort_by is not None:
        if sort_by not in out.columns:
            raise ValueError(
                f"sort_by='{sort_by}' not found in output columns. "
                f"Available columns: {list(out.columns)}"
            )
        out = out.sort_values(sort_by, ascending=ascending, na_position="last")
    else:
        out = out.sort_values(group_by, ascending=True, na_position="last")

    return out.reset_index(drop=True)


def compare_two_subsets(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    *,
    left_label: str = "left_subset",
    right_label: str = "right_subset",
    metrics: Optional[Iterable[str]] = None,
    aggregations: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Compare two already-defined subsets by assigning each subset a label
    and then running compare_groups().

    Useful when the subsets were built by separate filters.
    """
    left = df_left.copy()
    right = df_right.copy()

    left["_comparison_group"] = left_label
    right["_comparison_group"] = right_label

    combined = pd.concat([left, right], ignore_index=True)

    return compare_groups(
        combined,
        group_by="_comparison_group",
        metrics=metrics,
        aggregations=aggregations,
        dropna_group=True,
    )


def compare_by_birth_region(
    df: pd.DataFrame,
    *,
    metrics: Optional[Iterable[str]] = None,
    aggregations: Optional[Iterable[str]] = None,
    sort_by: Optional[str] = "group_count",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper for comparing by birth_region.
    """
    return compare_groups(
        df,
        group_by="birth_region",
        metrics=metrics,
        aggregations=aggregations,
        dropna_group=False,
        sort_by=sort_by,
        ascending=ascending,
    )


def compare_by_death_region(
    df: pd.DataFrame,
    *,
    metrics: Optional[Iterable[str]] = None,
    aggregations: Optional[Iterable[str]] = None,
    sort_by: Optional[str] = "group_count",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper for comparing by death_region.
    """
    return compare_groups(
        df,
        group_by="death_region",
        metrics=metrics,
        aggregations=aggregations,
        dropna_group=False,
        sort_by=sort_by,
        ascending=ascending,
    )


def compare_by_record_status(
    df: pd.DataFrame,
    *,
    metrics: Optional[Iterable[str]] = None,
    aggregations: Optional[Iterable[str]] = None,
    sort_by: Optional[str] = "group_count",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper for comparing by record_status.
    """
    return compare_groups(
        df,
        group_by="record_status",
        metrics=metrics,
        aggregations=aggregations,
        dropna_group=False,
        sort_by=sort_by,
        ascending=ascending,
    )


def compare_by_duplicate_flag(
    df: pd.DataFrame,
    *,
    metrics: Optional[Iterable[str]] = None,
    aggregations: Optional[Iterable[str]] = None,
    sort_by: Optional[str] = "group_count",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper for comparing by duplicate_ring_id_flag.
    """
    return compare_groups(
        df,
        group_by="duplicate_ring_id_flag",
        metrics=metrics,
        aggregations=aggregations,
        dropna_group=False,
        sort_by=sort_by,
        ascending=ascending,
    )


def compare_early_vs_late_period(
    df: pd.DataFrame,
    *,
    split_year: int,
    year_column: str = "birth_year",
    early_label: str = "early",
    late_label: str = "late",
    metrics: Optional[Iterable[str]] = None,
    aggregations: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Compare rows before and after a split year.

    Rows with year < split_year -> early_label
    Rows with year >= split_year -> late_label
    """
    _validate_columns(df, [year_column])

    work = df.copy()
    work[year_column] = pd.to_numeric(work[year_column], errors="coerce")
    work = work.dropna(subset=[year_column])

    if work.empty:
        return pd.DataFrame(columns=["time_period"])

    work["time_period"] = work[year_column].apply(
        lambda y: early_label if int(y) < split_year else late_label
    )

    return compare_groups(
        work,
        group_by="time_period",
        metrics=metrics,
        aggregations=aggregations,
        dropna_group=True,
    )


def compare_small_vs_large_rings(
    df: pd.DataFrame,
    *,
    threshold_area_km2: float,
    area_column: str = "area_km2",
    small_label: str = "small",
    large_label: str = "large",
    metrics: Optional[Iterable[str]] = None,
    aggregations: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Compare rings below vs above an area threshold.
    """
    _validate_columns(df, [area_column])

    work = df.copy()
    work[area_column] = pd.to_numeric(work[area_column], errors="coerce")
    work = work.dropna(subset=[area_column])

    if work.empty:
        return pd.DataFrame(columns=["size_class"])

    work["size_class"] = work[area_column].apply(
        lambda a: small_label if float(a) < threshold_area_km2 else large_label
    )

    return compare_groups(
        work,
        group_by="size_class",
        metrics=metrics,
        aggregations=aggregations,
        dropna_group=True,
    )