# src/wcr_agent/plotting/maps.py

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


DEFAULT_TEMPLATE = "plotly_white"


def _validate_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe.")


def _validate_lat_lon_columns(df: pd.DataFrame, lat_col: str, lon_col: str) -> None:
    _validate_column(df, lat_col)
    _validate_column(df, lon_col)


def _prepare_point_map_df(
    df: pd.DataFrame,
    *,
    lat_col: str,
    lon_col: str,
    color_col: Optional[str] = None,
    hover_name: Optional[str] = None,
    hover_data: Optional[list[str]] = None,
) -> pd.DataFrame:
    _validate_lat_lon_columns(df, lat_col, lon_col)

    cols = [lat_col, lon_col]
    if color_col is not None:
        _validate_column(df, color_col)
        cols.append(color_col)
    if hover_name is not None:
        _validate_column(df, hover_name)
        cols.append(hover_name)
    if hover_data is not None:
        for col in hover_data:
            _validate_column(df, col)
        cols.extend(hover_data)

    cols = list(dict.fromkeys(cols))
    out = df[cols].copy()

    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    out = out.dropna(subset=[lat_col, lon_col])

    if out.empty:
        raise ValueError(f"No valid rows available for map using {lat_col}/{lon_col}.")

    return out


def _default_hover_data(df: pd.DataFrame) -> list[str]:
    preferred = [
        "row_id",
        "ring_id",
        "date_first_seen",
        "date_last_seen",
        "area_km2",
        "lifetime_days",
        "displacement_km",
        "record_status",
    ]
    return [col for col in preferred if col in df.columns]


def plot_point_map(
    df: pd.DataFrame,
    *,
    lat_col: str,
    lon_col: str,
    color_col: Optional[str] = None,
    hover_name: Optional[str] = "ring_id",
    hover_data: Optional[list[str]] = None,
    title: Optional[str] = None,
    zoom: float = 2.5,
    height: int = 650,
    map_style: str = "carto-positron",
) -> go.Figure:
    """
    Generic point map for latitude/longitude columns.
    """
    if hover_data is None:
        hover_data = _default_hover_data(df)

    plot_df = _prepare_point_map_df(
        df,
        lat_col=lat_col,
        lon_col=lon_col,
        color_col=color_col,
        hover_name=hover_name if hover_name in df.columns else None,
        hover_data=hover_data,
    )

    fig = px.scatter_map(
        plot_df,
        lat=lat_col,
        lon=lon_col,
        color=color_col,
        hover_name=hover_name if hover_name in plot_df.columns else None,
        hover_data=hover_data,
        zoom=zoom,
        height=height,
        title=title or f"Point Map: {lat_col} / {lon_col}",
        map_style=map_style,
    )

    fig.update_layout(template=DEFAULT_TEMPLATE, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_birth_locations(
    df: pd.DataFrame,
    *,
    color_col: Optional[str] = None,
    hover_name: str = "ring_id",
    hover_data: Optional[list[str]] = None,
    title: Optional[str] = None,
    zoom: float = 2.5,
    height: int = 650,
    map_style: str = "carto-positron",
) -> go.Figure:
    """
    Plot birth locations of WCR records.
    """
    return plot_point_map(
        df,
        lat_col="lat_birth",
        lon_col="lon_birth",
        color_col=color_col,
        hover_name=hover_name,
        hover_data=hover_data,
        title=title or "WCR Birth Locations",
        zoom=zoom,
        height=height,
        map_style=map_style,
    )


def plot_death_locations(
    df: pd.DataFrame,
    *,
    color_col: Optional[str] = None,
    hover_name: str = "ring_id",
    hover_data: Optional[list[str]] = None,
    title: Optional[str] = None,
    zoom: float = 2.5,
    height: int = 650,
    map_style: str = "carto-positron",
) -> go.Figure:
    """
    Plot demise/absorption locations of WCR records.
    """
    return plot_point_map(
        df,
        lat_col="lat_death",
        lon_col="lon_death",
        color_col=color_col,
        hover_name=hover_name,
        hover_data=hover_data,
        title=title or "WCR Death Locations",
        zoom=zoom,
        height=height,
        map_style=map_style,
    )


def plot_birth_and_death_locations(
    df: pd.DataFrame,
    *,
    hover_name: str = "ring_id",
    hover_data: Optional[list[str]] = None,
    title: str = "WCR Birth and Death Locations",
    zoom: float = 2.5,
    height: int = 700,
    map_style: str = "carto-positron",
) -> go.Figure:
    """
    Plot birth and death points together on one map.
    """
    required = ["lat_birth", "lon_birth", "lat_death", "lon_death"]
    for col in required:
        _validate_column(df, col)

    if hover_data is None:
        hover_data = _default_hover_data(df)

    records = []

    for stage, lat_col, lon_col in [
        ("Birth", "lat_birth", "lon_birth"),
        ("Death", "lat_death", "lon_death"),
    ]:
        cols = [lat_col, lon_col]
        if hover_name in df.columns:
            cols.append(hover_name)
        cols.extend([c for c in hover_data if c in df.columns])
        cols = list(dict.fromkeys(cols))

        temp = df[cols].copy()
        temp[lat_col] = pd.to_numeric(temp[lat_col], errors="coerce")
        temp[lon_col] = pd.to_numeric(temp[lon_col], errors="coerce")
        temp = temp.dropna(subset=[lat_col, lon_col])

        if temp.empty:
            continue

        temp = temp.rename(columns={lat_col: "latitude", lon_col: "longitude"})
        temp["stage"] = stage
        records.append(temp)

    if not records:
        raise ValueError("No valid birth or death coordinates available to plot.")

    plot_df = pd.concat(records, ignore_index=True)

    fig = px.scatter_map(
        plot_df,
        lat="latitude",
        lon="longitude",
        color="stage",
        hover_name=hover_name if hover_name in plot_df.columns else None,
        hover_data=[c for c in hover_data if c in plot_df.columns],
        zoom=zoom,
        height=height,
        title=title,
        map_style=map_style,
    )

    fig.update_layout(template=DEFAULT_TEMPLATE, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_birth_to_death_segments(
    df: pd.DataFrame,
    *,
    hover_name_col: str = "ring_id",
    color_by: Optional[str] = None,
    show_birth_points: bool = True,
    show_death_points: bool = True,
    title: str = "Birth-to-Death Segments",
    zoom: float = 2.5,
    height: int = 700,
    map_style: str = "carto-positron",
    max_segments: Optional[int] = None,
) -> go.Figure:
    """
    Plot one line segment per record from birth to death location.
    """
    required = ["lat_birth", "lon_birth", "lat_death", "lon_death"]
    for col in required:
        _validate_column(df, col)

    work = df.copy()
    for col in required:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=required)

    if work.empty:
        raise ValueError("No valid rows with both birth and death coordinates.")

    if max_segments is not None:
        work = work.head(max_segments).copy()

    fig = go.Figure()

    # Segment lines
    if color_by is not None and color_by in work.columns:
        grouped = work.groupby(color_by, dropna=False)
        for group_name, g in grouped:
            lats = []
            lons = []
            texts = []

            for _, row in g.iterrows():
                label = f"{hover_name_col}: {row[hover_name_col]}" if hover_name_col in g.columns else ""
                lats.extend([row["lat_birth"], row["lat_death"], None])
                lons.extend([row["lon_birth"], row["lon_death"], None])
                texts.extend([label, label, None])

            fig.add_trace(
                go.Scattermap(
                    lat=lats,
                    lon=lons,
                    mode="lines",
                    name=str(group_name),
                    text=texts,
                    hoverinfo="text",
                )
            )
    else:
        lats = []
        lons = []
        texts = []

        for _, row in work.iterrows():
            label_parts = []
            if hover_name_col in work.columns:
                label_parts.append(f"{hover_name_col}: {row[hover_name_col]}")
            if "date_first_seen" in work.columns:
                label_parts.append(f"Birth: {row['date_first_seen']}")
            if "date_last_seen" in work.columns:
                label_parts.append(f"Death: {row['date_last_seen']}")
            label = "<br>".join(map(str, label_parts))

            lats.extend([row["lat_birth"], row["lat_death"], None])
            lons.extend([row["lon_birth"], row["lon_death"], None])
            texts.extend([label, label, None])

        fig.add_trace(
            go.Scattermap(
                lat=lats,
                lon=lons,
                mode="lines",
                name="Birth → Death",
                text=texts,
                hoverinfo="text",
            )
        )

    # Birth points
    if show_birth_points:
        fig.add_trace(
            go.Scattermap(
                lat=work["lat_birth"],
                lon=work["lon_birth"],
                mode="markers",
                name="Birth",
                text=work[hover_name_col] if hover_name_col in work.columns else None,
                hoverinfo="text",
            )
        )

    # Death points
    if show_death_points:
        fig.add_trace(
            go.Scattermap(
                lat=work["lat_death"],
                lon=work["lon_death"],
                mode="markers",
                name="Death",
                text=work[hover_name_col] if hover_name_col in work.columns else None,
                hoverinfo="text",
            )
        )

    center_lat = pd.concat([work["lat_birth"], work["lat_death"]]).mean()
    center_lon = pd.concat([work["lon_birth"], work["lon_death"]]).mean()

    fig.update_layout(
        title=title,
        template=DEFAULT_TEMPLATE,
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_layout(
        map=dict(
            style=map_style,
            center=dict(lat=float(center_lat), lon=float(center_lon)),
            zoom=zoom,
        )
    )

    return fig


def plot_displacement_bubble_map(
    df: pd.DataFrame,
    *,
    lat_col: str = "lat_birth",
    lon_col: str = "lon_birth",
    size_col: str = "displacement_km",
    color_col: Optional[str] = "lifetime_days",
    hover_name: str = "ring_id",
    hover_data: Optional[list[str]] = None,
    title: str = "Birth Locations Sized by Displacement",
    zoom: float = 2.5,
    height: int = 650,
    map_style: str = "carto-positron",
    size_max: int = 30,
) -> go.Figure:
    """
    Plot points where marker size reflects displacement.
    """
    _validate_column(df, size_col)

    if hover_data is None:
        hover_data = _default_hover_data(df)

    cols = [lat_col, lon_col, size_col]
    if color_col is not None and color_col in df.columns:
        cols.append(color_col)
    if hover_name in df.columns:
        cols.append(hover_name)
    cols.extend([c for c in hover_data if c in df.columns])
    cols = list(dict.fromkeys(cols))

    plot_df = df[cols].copy()
    plot_df[lat_col] = pd.to_numeric(plot_df[lat_col], errors="coerce")
    plot_df[lon_col] = pd.to_numeric(plot_df[lon_col], errors="coerce")
    plot_df[size_col] = pd.to_numeric(plot_df[size_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[lat_col, lon_col, size_col])

    if plot_df.empty:
        raise ValueError("No valid rows available for displacement bubble map.")

    fig = px.scatter_map(
        plot_df,
        lat=lat_col,
        lon=lon_col,
        size=size_col,
        color=color_col if color_col in plot_df.columns else None,
        hover_name=hover_name if hover_name in plot_df.columns else None,
        hover_data=[c for c in hover_data if c in plot_df.columns],
        zoom=zoom,
        height=height,
        title=title,
        map_style=map_style,
        size_max=size_max,
    )

    fig.update_layout(template=DEFAULT_TEMPLATE, margin=dict(l=10, r=10, t=50, b=10))
    return fig