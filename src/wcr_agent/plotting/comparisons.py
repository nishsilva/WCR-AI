# src/wcr_agent/plotting/comparisons.py

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


DEFAULT_TEMPLATE = "plotly_white"


def _validate_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe.")


def _prepare_plot_df(
    df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
) -> pd.DataFrame:
    _validate_column(df, group_col)
    _validate_column(df, value_col)

    out = df[[group_col, value_col]].copy()
    out[group_col] = out[group_col].astype("string").fillna("Missing")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[value_col])

    if out.empty:
        raise ValueError(
            f"No valid rows available for plotting using '{group_col}' and '{value_col}'."
        )

    return out


def plot_group_metric_bar(
    compare_df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    title: Optional[str] = None,
    template: str = DEFAULT_TEMPLATE,
) -> go.Figure:
    """
    Plot a bar chart from a grouped comparison table.

    Example:
    - group_col='birth_region'
    - value_col='lifetime_days_mean'
    """
    plot_df = _prepare_plot_df(compare_df, group_col=group_col, value_col=value_col)

    fig = px.bar(
        plot_df,
        x=group_col,
        y=value_col,
        title=title or f"{value_col} by {group_col}",
        template=template,
    )
    fig.update_layout(
        xaxis_title=group_col,
        yaxis_title=value_col,
    )
    return fig


def plot_group_metric_dot(
    compare_df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    title: Optional[str] = None,
    template: str = DEFAULT_TEMPLATE,
) -> go.Figure:
    """
    Plot a dot chart from a grouped comparison table.
    """
    plot_df = _prepare_plot_df(compare_df, group_col=group_col, value_col=value_col)
    plot_df = plot_df.sort_values(value_col, ascending=False)

    fig = px.scatter(
        plot_df,
        x=value_col,
        y=group_col,
        title=title or f"{value_col} by {group_col}",
        template=template,
    )
    fig.update_layout(
        xaxis_title=value_col,
        yaxis_title=group_col,
    )
    return fig


def plot_group_metric_box_from_raw(
    df: pd.DataFrame,
    *,
    group_col: str,
    metric_col: str,
    points: str = "outliers",
    title: Optional[str] = None,
    template: str = DEFAULT_TEMPLATE,
) -> go.Figure:
    """
    Plot a boxplot directly from the raw filtered dataframe.

    This is useful when you want to compare full distributions across groups,
    rather than already-aggregated comparison outputs.
    """
    _validate_column(df, group_col)
    _validate_column(df, metric_col)

    plot_df = df[[group_col, metric_col]].copy()
    plot_df[group_col] = plot_df[group_col].astype("string").fillna("Missing")
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric_col])

    if plot_df.empty:
        raise ValueError(
            f"No valid rows available for plotting using '{group_col}' and '{metric_col}'."
        )

    fig = px.box(
        plot_df,
        x=group_col,
        y=metric_col,
        points=points,
        title=title or f"{metric_col} by {group_col}",
        template=template,
    )
    fig.update_layout(
        xaxis_title=group_col,
        yaxis_title=metric_col,
    )
    return fig


def plot_group_metric_violin_from_raw(
    df: pd.DataFrame,
    *,
    group_col: str,
    metric_col: str,
    box: bool = True,
    points: Optional[str] = None,
    title: Optional[str] = None,
    template: str = DEFAULT_TEMPLATE,
) -> go.Figure:
    """
    Plot a violin plot directly from the raw filtered dataframe.
    """
    _validate_column(df, group_col)
    _validate_column(df, metric_col)

    plot_df = df[[group_col, metric_col]].copy()
    plot_df[group_col] = plot_df[group_col].astype("string").fillna("Missing")
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric_col])

    if plot_df.empty:
        raise ValueError(
            f"No valid rows available for plotting using '{group_col}' and '{metric_col}'."
        )

    fig = px.violin(
        plot_df,
        x=group_col,
        y=metric_col,
        box=box,
        points=points,
        title=title or f"{metric_col} by {group_col}",
        template=template,
    )
    fig.update_layout(
        xaxis_title=group_col,
        yaxis_title=metric_col,
    )
    return fig


def plot_two_metric_scatter(
    compare_df: pd.DataFrame,
    *,
    group_col: str,
    x_col: str,
    y_col: str,
    size_col: Optional[str] = "group_count",
    title: Optional[str] = None,
    template: str = DEFAULT_TEMPLATE,
) -> go.Figure:
    """
    Plot a scatter chart from an aggregated comparison table.

    Useful for comparisons like:
    - x = lifetime_days_mean
    - y = area_km2_mean
    - size = group_count
    """
    _validate_column(compare_df, group_col)
    _validate_column(compare_df, x_col)
    _validate_column(compare_df, y_col)

    cols = [group_col, x_col, y_col]
    if size_col is not None:
        _validate_column(compare_df, size_col)
        cols.append(size_col)

    plot_df = compare_df[cols].copy()
    plot_df[group_col] = plot_df[group_col].astype("string").fillna("Missing")
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    if size_col is not None:
        plot_df[size_col] = pd.to_numeric(plot_df[size_col], errors="coerce")

    needed = [x_col, y_col] + ([size_col] if size_col is not None else [])
    plot_df = plot_df.dropna(subset=[c for c in needed if c is not None])

    if plot_df.empty:
        raise ValueError("No valid rows available for two-metric scatter plot.")

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        size=size_col if size_col is not None else None,
        text=group_col,
        hover_name=group_col,
        title=title or f"{y_col} vs {x_col} by {group_col}",
        template=template,
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    return fig


def get_plottable_comparison_value_columns(compare_df: pd.DataFrame) -> list[str]:
    """
    Return numeric-ish columns from a comparison dataframe that are useful for plotting.
    Excludes the first group label column by convention.
    """
    if compare_df.empty:
        return []

    out: list[str] = []
    for col in compare_df.columns[1:]:
        converted = pd.to_numeric(compare_df[col], errors="coerce")
        if converted.notna().any():
            out.append(col)

    return out