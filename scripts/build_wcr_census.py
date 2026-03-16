# scripts/build_wcr_census.py

from __future__ import annotations

from pathlib import Path
import math
import pandas as pd
import numpy as np
from wcr_agent.data_access.regions import assign_birth_death_regions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "WCR_Master_Dataset_as_at_061319.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PARQUET = PROCESSED_DIR / "wcr_census.parquet"
OUTPUT_CSV = PROCESSED_DIR / "wcr_census_clean.csv"
VALIDATION_REPORT_CSV = PROCESSED_DIR / "wcr_census_validation_report.csv"


RAW_TO_INTERNAL_COLUMNS = {
    "WCR_name": "ring_id",
    "Date.of.Birth": "date_first_seen",
    "Latitude.x": "lat_birth",
    "Longitude.x": "lon_birth",
    "Area.sq.km..x": "area_km2",
    "Date.of.Absorption": "date_last_seen",
    "Latitude.y": "lat_death",
    "Longitude.y": "lon_death",
}


def haversine_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    """
    Vectorized great-circle distance in kilometers.
    Returns NaN if any coordinate needed for a row is missing.
    """
    r_earth_km = 6371.0088

    lat1_rad = np.radians(lat1.astype(float))
    lon1_rad = np.radians(lon1.astype(float))
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    return r_earth_km * c


def initial_bearing_degrees(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    """
    Vectorized initial bearing from point 1 to point 2 in degrees [0, 360).
    Returns NaN where coordinates are missing.
    """
    lat1_rad = np.radians(lat1.astype(float))
    lon1_rad = np.radians(lon1.astype(float))
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))

    dlon = lon2_rad - lon1_rad

    x = np.sin(dlon) * np.cos(lat2_rad)
    y = (
        np.cos(lat1_rad) * np.sin(lat2_rad)
        - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    )

    bearings = np.degrees(np.arctan2(x, y))
    return (bearings + 360.0) % 360.0


def compute_record_status(df: pd.DataFrame) -> pd.Series:
    """
    Assign a simple status label for each record.
    Priority order:
      1. duplicate_ring_id
      2. missing_absorption_and_demise_location
      3. missing_absorption
      4. missing_demise_location
      5. invalid_negative_lifetime
      6. complete
    """
    status = pd.Series("complete", index=df.index, dtype="object")

    missing_absorption = df["date_last_seen"].isna()
    missing_demise_location = df["lat_death"].isna() | df["lon_death"].isna()
    negative_lifetime = df["lifetime_days"].notna() & (df["lifetime_days"] < 0)
    duplicate_ring_id = df["duplicate_ring_id_flag"]

    status.loc[negative_lifetime] = "invalid_negative_lifetime"
    status.loc[missing_demise_location] = "missing_demise_location"
    status.loc[missing_absorption] = "missing_absorption"
    status.loc[missing_absorption & missing_demise_location] = (
        "missing_absorption_and_demise_location"
    )
    status.loc[duplicate_ring_id] = "duplicate_ring_id"

    return status


def build_validation_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a compact validation report as a two-column table.
    """
    metrics = {
        "n_rows": len(df),
        "n_unique_ring_id": df["ring_id"].nunique(dropna=True),
        "n_duplicate_rows_by_ring_id": int(df["duplicate_ring_id_flag"].sum()),
        "n_duplicate_ring_ids": int(df.loc[df["duplicate_ring_id_flag"], "ring_id"].nunique()),
        "n_missing_date_first_seen": int(df["date_first_seen"].isna().sum()),
        "n_missing_date_last_seen": int(df["date_last_seen"].isna().sum()),
        "n_missing_lat_birth": int(df["lat_birth"].isna().sum()),
        "n_missing_lon_birth": int(df["lon_birth"].isna().sum()),
        "n_missing_lat_death": int(df["lat_death"].isna().sum()),
        "n_missing_lon_death": int(df["lon_death"].isna().sum()),
        "n_missing_area_km2": int(df["area_km2"].isna().sum()),
        "n_negative_lifetime": int((df["lifetime_days"] < 0).fillna(False).sum()),
        "birth_date_min": df["date_first_seen"].min(),
        "birth_date_max": df["date_first_seen"].max(),
        "death_date_min": df["date_last_seen"].min(),
        "death_date_max": df["date_last_seen"].max(),
        "area_km2_min": df["area_km2"].min(),
        "area_km2_max": df["area_km2"].max(),
        "lifetime_days_min": df["lifetime_days"].min(),
        "lifetime_days_max": df["lifetime_days"].max(),
    }

    report = pd.DataFrame(
        {"metric": list(metrics.keys()), "value": list(metrics.values())}
    )
    return report


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Raw input file not found:\n{RAW_CSV}\n\n"
            "Place the raw dataset in data/raw/ before running this script."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw dataset from: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)

    expected_cols = set(RAW_TO_INTERNAL_COLUMNS.keys())
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            "The raw CSV is missing expected columns:\n"
            + "\n".join(sorted(missing_cols))
        )

    # Keep only expected columns in a controlled order, then rename
    df = df[list(RAW_TO_INTERNAL_COLUMNS.keys())].rename(columns=RAW_TO_INTERNAL_COLUMNS)

    # Add a stable row identifier
    df.insert(0, "row_id", np.arange(1, len(df) + 1))

    # Strip whitespace in ring_id
    df["ring_id"] = df["ring_id"].astype(str).str.strip()

    # Parse dates
    df["date_first_seen"] = pd.to_datetime(df["date_first_seen"], errors="coerce")
    df["date_last_seen"] = pd.to_datetime(df["date_last_seen"], errors="coerce")

    # Convert numerics
    numeric_cols = [
        "lat_birth",
        "lon_birth",
        "area_km2",
        "lat_death",
        "lon_death",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Duplicate flags
    df["duplicate_ring_id_flag"] = df["ring_id"].duplicated(keep=False)
    duplicate_group_sizes = df.groupby("ring_id")["ring_id"].transform("size")
    df["duplicate_group_size"] = np.where(
        df["duplicate_ring_id_flag"], duplicate_group_sizes, 1
    )

    # Time-derived columns
    df["lifetime_days"] = (df["date_last_seen"] - df["date_first_seen"]).dt.days
    df["birth_year"] = df["date_first_seen"].dt.year
    df["birth_month"] = df["date_first_seen"].dt.month
    df["death_year"] = df["date_last_seen"].dt.year
    df["death_month"] = df["date_last_seen"].dt.month

    # Spatial deltas
    df["delta_lat"] = df["lat_death"] - df["lat_birth"]
    df["delta_lon"] = df["lon_death"] - df["lon_birth"]

    # Equivalent radius from area
    df["radius_equiv_km"] = np.sqrt(df["area_km2"] / math.pi)

    # Geodesic summaries
    coord_cols = ["lat_birth", "lon_birth", "lat_death", "lon_death"]
    has_all_coords = df[coord_cols].notna().all(axis=1)

    df["displacement_km"] = np.nan
    df.loc[has_all_coords, "displacement_km"] = haversine_km(
        df.loc[has_all_coords, "lat_birth"],
        df.loc[has_all_coords, "lon_birth"],
        df.loc[has_all_coords, "lat_death"],
        df.loc[has_all_coords, "lon_death"],
    )

    df["bearing_birth_to_death"] = np.nan
    df.loc[has_all_coords, "bearing_birth_to_death"] = initial_bearing_degrees(
        df.loc[has_all_coords, "lat_birth"],
        df.loc[has_all_coords, "lon_birth"],
        df.loc[has_all_coords, "lat_death"],
        df.loc[has_all_coords, "lon_death"],
    )

    # Placeholder region columns
    df = assign_birth_death_regions(df)

    # Record status
    df["record_status"] = compute_record_status(df)

    # Sort for readability
    df = df.sort_values(["date_first_seen", "ring_id", "row_id"], na_position="last").reset_index(drop=True)

    # Validation report
    report = build_validation_report(df)

    # Write outputs
    print(f"Writing cleaned parquet to: {OUTPUT_PARQUET}")
    df.to_parquet(OUTPUT_PARQUET, index=False)

    print(f"Writing cleaned CSV to: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Writing validation report to: {VALIDATION_REPORT_CSV}")
    report.to_csv(VALIDATION_REPORT_CSV, index=False)

    print("\nBuild complete.")
    print(f"Rows: {len(df)}")
    print(f"Unique ring IDs: {df['ring_id'].nunique(dropna=True)}")
    print(f"Duplicate-ring rows: {int(df['duplicate_ring_id_flag'].sum())}")
    print(f"Missing absorption dates: {int(df['date_last_seen'].isna().sum())}")
    print(
        "Missing demise coordinates: "
        f"{int((df['lat_death'].isna() | df['lon_death'].isna()).sum())}"
    )


if __name__ == "__main__":
    main()