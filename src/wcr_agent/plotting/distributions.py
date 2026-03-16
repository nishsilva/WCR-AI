# src/wcr_agent/plotting/distributions.py

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


DEFAULT_TEMPLATE = "plotly_white"


def _validate_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe.")


def _prepare_numeric_series(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    _validate_column(df, column)

    out = df[[column]].copy()
    out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=[column])

    if out.empty:
        raise ValueError(f"Column '{column}' has no valid numeric values to plot.")

    return out


def plot_histogram(
    df: pd.DataFrame,
    *,
    column: str,
    nbins: int = 30,
    title: Optional[str] = None,
    marginal: Optional[str] = None,
    opacity: float = 0.85,
    template: str = DEFAULT_TEMPLATE,
) -> go.Figure:
    """
    Plot a histogram for a numeric column.
    """
    plot_df = _prepare_numeric_series(df, column)

    fig = px.histogram(
        plot_df,
        x=column,
        nbins=nbins,
        marginal=marginal,
        opacity=opacity,
        title=title or f"Distribution of {column}",
        template=template,
    )

    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Count",
    )
    return fig


def plot_boxplot(
    df: pd.DataFrame,
    *,
    column: str,
    by: Optional[str] = None,
    points: str = "outliers",
    title: Optional[str] = None,
    template: str = DEFAULT_TEMPLATE,
) -> go.Figure:
    """
    Plot a boxplot for a numeric column, optionally grouped by a category.
    """
    _validate_column(df, column)
    plot_df = df.copy()
    plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=[column])

    if plot_df.empty:
        raise ValueError(f"Column '{column}' has no valid numeric values to plot.")

    if by is not None:
        _validate_column(plot_df, by)
        plot_df[by] = plot_df[by].astype("string").fillna("Missing")

        fig = px.box(
            plot_df,
            x=by,
            y=column,
            points=points,
            title=title or f"{column} by {by}",
            template=template,
        )
        fig.update_layout(
            xaxis_title=by,
            yaxis_title=column,
        )
    else:
        fig = px.box(
            plot_df,
            y=column,
            points=points,
            title=title or f"Boxplot of {column}",
            template=template,
        )
        fig.update_layout(
            xaxis_title="",
            yaxis_title=column,
        )

    return fig


def plot_violin(
    df: pd.DataFrame,
    *,
    column: str,
    by: Optional[str] = None,
    box: bool = True,
    points: Optional[str] = None,
    title: Optional[str] = None,
    template: str = DEFAULT_TEMPLATE,
) -> go.Figure:
    """
    Plot a violin plot for a numeric column, optionally grouped by a category.
    """
    _validate_column(df, column)
    plot_df = df.copy()
    plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=[column])

    if plot_df.empty:
        raise ValueError(f"Column '{column}' has no valid numeric values to plot.")

    if by is not None:
        _validate_column(plot_df, by)
        plot_df[by] = plot_df[by].astype("string").fillna("Missing")

        fig = px.violin(
            plot_df,
            x=by,
            y=column,
            box=box,
            points=points,
            title=title or f"{column} by {by}",
            template=template,
        )
        fig.update_layout(
            xaxis_title=by,
            yaxis_title=column,
        )
    else:
        fig = px.violin(
            plot_df,
            y=column,
            box=box,
            points=points,
            title=title or f"Distribution of {column}",
            template=template,
        )
        fig.update_layout(
            xaxis_title="",
            yaxis_title=column,
        )

    return fig


def plot_yearly_counts_bar(
    counts_df: pd.DataFrame,
    *,
    year_column: str,
    count_column: str = "count",
    title: Optional[str] = None,
    template: str = DEFAULT_TEMPLATE,
) -> go.Figure:
    """
    Plot yearly counts from a precomputed counts dataframe.
    """
    _validate_column(counts_df, year_column)
    _validate_column(counts_df, count_column)

    plot_df = counts_df.copy()
    plot_df[year_column] = pd.to_numeric(plot_df[year_column], errors="coerce")
    plot_df[count_column] = pd.to_numeric(plot_df[count_column], errors="coerce")
    plot_df = plot_df.dropna(subset=[year_column, count_column])

    if plot_df.empty:
        raise ValueError("The yearly counts dataframe has no valid values to plot.")

    fig = px.bar(
        plot_df,
        x=year_column,
        y=count_column,
        title=title or f"Yearly Counts by {year_column}",
        template=template,
    )

    fig.update_layout(
        xaxis_title=year_column,
        yaxis_title=count_column,
    )
    return fig


def plot_yearly_counts_line(
    counts_df: pd.DataFrame,
    *,
    year_column: str,
    count_column: str = "count",
    markers: bool = True,
    title: Optional[str] = None,
    template: str = DEFAULT_TEMPLATE,
) -> go.Figure:
    """
    Plot yearly counts as a line chart from a precomputed counts dataframe.
    """
    _validate_column(counts_df, year_column)
    _validate_column(counts_df, count_column)

    plot_df = counts_df.copy()
    plot_df[year_column] = pd.to_numeric(plot_df[year_column], errors="coerce")
    plot_df[count_column] = pd.to_numeric(plot_df[count_column], errors="coerce")
    plot_df = plot_df.dropna(subset=[year_column, count_column]).sort_values(year_column)

    if plot_df.empty:
        raise ValueError("The yearly counts dataframe has no valid values to plot.")

    fig = px.line(
        plot_df,
        x=year_column,
        y=count_column,
        markers=markers,
        title=title or f"Yearly Counts by {year_column}",
        template=template,
    )

    fig.update_layout(
        xaxis_title=year_column,
        yaxis_title=count_column,
    )
    return fig


def plot_birth_vs_death_counts(
    compare_df: pd.DataFrame,
    *,
    year_column: str = "year",
    birth_count_column: str = "birth_count",
    death_count_column: str = "death_count",
    title: Optional[str] = None,
    template: str = DEFAULT_TEMPLATE,
) -> go.Figure:
    """
    Plot birth and death yearly counts together as two lines.
    """
    for col in [year_column, birth_count_column, death_count_column]:
        _validate_column(compare_df, col)

    plot_df = compare_df.copy()
    plot_df[year_column] = pd.to_numeric(plot_df[year_column], errors="coerce")
    plot_df[birth_count_column] = pd.to_numeric(plot_df[birth_count_column], errors="coerce")
    plot_df[death_count_column] = pd.to_numeric(plot_df[death_count_column], errors="coerce")
    plot_df = plot_df.dropna(subset=[year_column]).sort_values(year_column)

    if plot_df.empty:
        raise ValueError("The comparison dataframe has no valid years to plot.")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df[year_column],
            y=plot_df[birth_count_column],
            mode="lines+markers",
            name="Birth count",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df[year_column],
            y=plot_df[death_count_column],
            mode="lines+markers",
            name="Death count",
        )
    )

    fig.update_layout(
        title=title or "Birth vs Death Counts by Year",
        xaxis_title=year_column,
        yaxis_title="Count",
        template=template,
    )
    return fig


def plot_grouped_distribution_histogram(
    df: pd.DataFrame,
    *,
    column: str,
    group_by: str,
    nbins: int = 30,
    barmode: str = "overlay",
    opacity: float = 0.7,
    title: Optional[str] = None,
    template: str = DEFAULT_TEMPLATE,
) -> go.Figure:
    """
    Plot a grouped histogram for a numeric variable split by a categorical column.
    """
    _validate_column(df, column)
    _validate_column(df, group_by)

    plot_df = df[[column, group_by]].copy()
    plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df[group_by] = plot_df[group_by].astype("string").fillna("Missing")
    plot_df = plot_df.dropna(subset=[column])

    if plot_df.empty:
        raise ValueError("No valid rows remain after cleaning for grouped histogram.")

    fig = px.histogram(
        plot_df,
        x=column,
        color=group_by,
        nbins=nbins,
        barmode=barmode,
        opacity=opacity,
        title=title or f"{column} grouped by {group_by}",
        template=template,
    )

    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Count",
    )
    return fig