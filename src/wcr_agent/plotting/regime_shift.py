# src/wcr_agent/plotting/regime_shift.py

from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go

from wcr_agent.analysis.regime_shift import RegimeShiftResult

DEFAULT_TEMPLATE = "plotly_white"

_REGIME_COLORS = [
    "rgba(99,  110, 250, 0.15)",
    "rgba(239, 85,  59,  0.15)",
    "rgba(0,   204, 150, 0.15)",
    "rgba(171, 99,  250, 0.15)",
    "rgba(255, 161, 90,  0.15)",
]


def plot_regime_shift(
    result: RegimeShiftResult,
    *,
    title: Optional[str] = None,
    template: str = DEFAULT_TEMPLATE,
) -> go.Figure:
    counts_df = result.counts_df
    year_col = result.year_column
    years = counts_df[year_col].tolist()
    counts = counts_df["count"].tolist()

    fig = go.Figure()

    # Shaded regime bands
    for i, row in result.segments_df.iterrows():
        color = _REGIME_COLORS[int(row["regime"] - 1) % len(_REGIME_COLORS)]
        fig.add_vrect(
            x0=row["start_year"] - 0.5,
            x1=row["end_year"] + 0.5,
            fillcolor=color,
            line_width=0,
            layer="below",
            annotation_text=f"Regime {int(row['regime'])}<br>μ={row['mean_count']}",
            annotation_position="top left",
            annotation_font_size=11,
        )

    # Raw annual counts bar
    fig.add_trace(
        go.Bar(
            x=years,
            y=counts,
            name="Annual count",
            marker_color="rgba(100, 149, 237, 0.6)",
            marker_line_color="rgba(100, 149, 237, 1.0)",
            marker_line_width=0.5,
        )
    )

    # Rolling mean
    if not result.rolling_mean.empty:
        fig.add_trace(
            go.Scatter(
                x=result.rolling_mean.index.tolist(),
                y=result.rolling_mean.tolist(),
                mode="lines",
                name=f"{result.rolling_window}-yr rolling mean",
                line=dict(color="rgba(50, 50, 50, 0.8)", width=2, dash="dot"),
            )
        )

    # Segment means as horizontal step lines
    for _, row in result.segments_df.iterrows():
        fig.add_shape(
            type="line",
            x0=row["start_year"] - 0.5,
            x1=row["end_year"] + 0.5,
            y0=row["mean_count"],
            y1=row["mean_count"],
            line=dict(color="crimson", width=2),
        )

    # Changepoint vertical lines
    for cp_year in result.changepoint_years:
        fig.add_vline(
            x=cp_year - 0.5,
            line_color="crimson",
            line_dash="dash",
            line_width=1.5,
            annotation_text=str(cp_year),
            annotation_position="top right",
            annotation_font_color="crimson",
            annotation_font_size=11,
        )

    label = "Birth" if year_col == "birth_year" else "Absorption"
    fig.update_layout(
        title=title or f"Regime Shift Analysis — Annual {label} Counts",
        xaxis_title="Year",
        yaxis_title="Annual ring count",
        template=template,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.15,
    )
    return fig
