# src/wcr_agent/data_access/regions.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class BoundingBoxRegion:
    """
    Simple rectangular geographic region.

    Longitude is assumed to use the [-180, 180] convention.
    """

    name: str
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float

    def contains(self, lon: float, lat: float) -> bool:
        return (
            self.min_lon <= lon <= self.max_lon
            and self.min_lat <= lat <= self.max_lat
        )


# -----------------------------------------------------------------------------
# Region definitions
# -----------------------------------------------------------------------------
# These are intentionally simple first-pass regions for exploratory analysis.
# You can refine or replace them later with polygons or shapefiles.
# -----------------------------------------------------------------------------
DEFAULT_REGIONS: list[BoundingBoxRegion] = [
    BoundingBoxRegion(
        name="Shelf/Slope",
        min_lon=-78.0,
        max_lon=-65.0,
        min_lat=32.0,
        max_lat=45.5,
    ),
    BoundingBoxRegion(
        name="Gulf Stream Corridor",
        min_lon=-77.0,
        max_lon=-55.0,
        min_lat=34.0,
        max_lat=42.5,
    ),
    BoundingBoxRegion(
        name="Sargasso Sea",
        min_lon=-75.0,
        max_lon=-50.0,
        min_lat=22.0,
        max_lat=36.0,
    ),
    BoundingBoxRegion(
        name="Northwest Atlantic",
        min_lon=-80.0,
        max_lon=-45.0,
        min_lat=36.0,
        max_lat=50.0,
    ),
    BoundingBoxRegion(
        name="Western Tropical Atlantic",
        min_lon=-70.0,
        max_lon=-45.0,
        min_lat=10.0,
        max_lat=22.0,
    ),
]


def classify_point_to_region(
    lon: float | int | None,
    lat: float | int | None,
    *,
    regions: Optional[list[BoundingBoxRegion]] = None,
    return_unclassified_label: str = "Unclassified",
) -> str | None:
    """
    Assign a simple region label to one point using bounding-box rules.

    Parameters
    ----------
    lon, lat
        Longitude and latitude values.
    regions
        Optional custom list of BoundingBoxRegion objects.
        If omitted, DEFAULT_REGIONS is used.
    return_unclassified_label
        Label to return when a valid point does not fall into any region.

    Returns
    -------
    str | None
        Region name, unclassified label, or None if lon/lat are missing.
    """
    if pd.isna(lon) or pd.isna(lat):
        return None

    lon_f = float(lon)
    lat_f = float(lat)

    region_list = regions if regions is not None else DEFAULT_REGIONS

    for region in region_list:
        if region.contains(lon_f, lat_f):
            return region.name

    return return_unclassified_label


def assign_region_series(
    lon: pd.Series,
    lat: pd.Series,
    *,
    regions: Optional[list[BoundingBoxRegion]] = None,
    return_unclassified_label: str = "Unclassified",
) -> pd.Series:
    """
    Vectorized-ish rowwise region assignment for paired lon/lat series.
    """
    if len(lon) != len(lat):
        raise ValueError("Longitude and latitude series must have the same length.")

    region_list = regions if regions is not None else DEFAULT_REGIONS

    return pd.Series(
        [
            classify_point_to_region(
                lo,
                la,
                regions=region_list,
                return_unclassified_label=return_unclassified_label,
            )
            for lo, la in zip(lon, lat)
        ],
        index=lon.index,
        dtype="object",
    )


def assign_birth_death_regions(
    df: pd.DataFrame,
    *,
    birth_lon_col: str = "lon_birth",
    birth_lat_col: str = "lat_birth",
    death_lon_col: str = "lon_death",
    death_lat_col: str = "lat_death",
    regions: Optional[list[BoundingBoxRegion]] = None,
    return_unclassified_label: str = "Unclassified",
    birth_output_col: str = "birth_region",
    death_output_col: str = "death_region",
) -> pd.DataFrame:
    """
    Add/overwrite birth_region and death_region columns in a dataframe.
    """
    required = [birth_lon_col, birth_lat_col, death_lon_col, death_lat_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns for region assignment:\n"
            + "\n".join(missing)
        )

    out = df.copy()

    out[birth_output_col] = assign_region_series(
        out[birth_lon_col],
        out[birth_lat_col],
        regions=regions,
        return_unclassified_label=return_unclassified_label,
    )

    out[death_output_col] = assign_region_series(
        out[death_lon_col],
        out[death_lat_col],
        regions=regions,
        return_unclassified_label=return_unclassified_label,
    )

    return out


def list_region_names(regions: Optional[list[BoundingBoxRegion]] = None) -> list[str]:
    """
    Return the region names in the order they are applied.
    """
    region_list = regions if regions is not None else DEFAULT_REGIONS
    return [region.name for region in region_list]


def get_region_definitions_df(
    regions: Optional[list[BoundingBoxRegion]] = None,
) -> pd.DataFrame:
    """
    Return region definitions as a dataframe for display/debugging.
    """
    region_list = regions if regions is not None else DEFAULT_REGIONS

    return pd.DataFrame(
        [
            {
                "name": r.name,
                "min_lon": r.min_lon,
                "max_lon": r.max_lon,
                "min_lat": r.min_lat,
                "max_lat": r.max_lat,
            }
            for r in region_list
        ]
    )