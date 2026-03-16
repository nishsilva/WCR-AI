# src/wcr_agent/data_access/census.py

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CENSUS_PATH = PROJECT_ROOT / "data" / "processed" / "wcr_census.parquet"


REQUIRED_COLUMNS = [
    "row_id",
    "ring_id",
    "date_first_seen",
    "lat_birth",
    "lon_birth",
    "area_km2",
    "date_last_seen",
    "lat_death",
    "lon_death",
    "duplicate_ring_id_flag",
    "duplicate_group_size",
    "lifetime_days",
    "birth_year",
    "birth_month",
    "death_year",
    "death_month",
    "delta_lat",
    "delta_lon",
    "radius_equiv_km",
    "displacement_km",
    "bearing_birth_to_death",
    "birth_region",
    "death_region",
    "record_status",
]


def _validate_census_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Processed census dataset is missing required columns:\n"
            + "\n".join(missing)
        )


def _normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic dtype normalization after loading.
    """
    out = df.copy()

    if "row_id" in out.columns:
        out["row_id"] = pd.to_numeric(out["row_id"], errors="coerce").astype("Int64")

    if "ring_id" in out.columns:
        out["ring_id"] = out["ring_id"].astype("string").str.strip()

    date_cols = ["date_first_seen", "date_last_seen"]
    for col in date_cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    bool_cols = ["duplicate_ring_id_flag"]
    for col in bool_cols:
        if col in out.columns:
            out[col] = out[col].astype("boolean")

    return out


@lru_cache(maxsize=4)
def load_census(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the processed WCR census dataset from parquet.

    Parameters
    ----------
    path : str, optional
        Custom path to the parquet file. If not provided, uses the default
        project location: data/processed/wcr_census.parquet

    Returns
    -------
    pd.DataFrame
        Loaded census dataframe with normalized dtypes.

    Notes
    -----
    This function is cached in memory. If the underlying parquet file changes
    during a session, call clear_census_cache() before reloading.
    """
    census_path = Path(path) if path is not None else DEFAULT_CENSUS_PATH

    if not census_path.exists():
        raise FileNotFoundError(
            f"Census parquet file not found:\n{census_path}\n\n"
            "Run scripts/build_wcr_census.py first."
        )

    df = pd.read_parquet(census_path)
    _validate_census_columns(df)
    df = _normalize_dtypes(df)

    return df


def clear_census_cache() -> None:
    """
    Clear the in-memory cache for load_census().
    Useful after rebuilding the parquet file.
    """
    load_census.cache_clear()


def get_census_shape(path: Optional[str] = None) -> tuple[int, int]:
    """
    Return (n_rows, n_columns) for the census dataset.
    """
    df = load_census(path=path)
    return df.shape


def get_census_summary(path: Optional[str] = None) -> dict:
    """
    Return a compact summary of the processed census dataset.
    """
    df = load_census(path=path)

    summary = {
        "n_rows": int(len(df)),
        "n_unique_ring_id": int(df["ring_id"].nunique(dropna=True)),
        "n_duplicate_rows": int(df["duplicate_ring_id_flag"].fillna(False).sum()),
        "n_duplicate_ring_ids": int(
            df.loc[df["duplicate_ring_id_flag"].fillna(False), "ring_id"]
            .nunique(dropna=True)
        ),
        "n_missing_absorption_dates": int(df["date_last_seen"].isna().sum()),
        "n_missing_demise_coordinates": int(
            (df["lat_death"].isna() | df["lon_death"].isna()).sum()
        ),
        "birth_date_min": (
            None if df["date_first_seen"].dropna().empty
            else df["date_first_seen"].min().date().isoformat()
        ),
        "birth_date_max": (
            None if df["date_first_seen"].dropna().empty
            else df["date_first_seen"].max().date().isoformat()
        ),
        "death_date_min": (
            None if df["date_last_seen"].dropna().empty
            else df["date_last_seen"].min().date().isoformat()
        ),
        "death_date_max": (
            None if df["date_last_seen"].dropna().empty
            else df["date_last_seen"].max().date().isoformat()
        ),
    }
    return summary


def get_ring_by_row_id(row_id: int, path: Optional[str] = None) -> pd.Series:
    """
    Retrieve exactly one census record by unique row_id.

    Raises
    ------
    KeyError
        If no matching row_id exists.
    """
    df = load_census(path=path)
    match = df.loc[df["row_id"] == row_id]

    if match.empty:
        raise KeyError(f"No census record found for row_id={row_id}")

    return match.iloc[0]


def get_rows_by_ring_id(ring_id: str, path: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieve all rows associated with a given scientific ring_id.

    This may return multiple rows if the ring_id is duplicated in the source data.
    """
    df = load_census(path=path)
    ring_id_clean = str(ring_id).strip()

    out = df.loc[df["ring_id"] == ring_id_clean].copy()

    if out.empty:
        raise KeyError(f"No census records found for ring_id='{ring_id_clean}'")

    return out.sort_values(["date_first_seen", "date_last_seen", "row_id"])


def ring_id_exists(ring_id: str, path: Optional[str] = None) -> bool:
    """
    Check whether at least one row exists for a ring_id.
    """
    df = load_census(path=path)
    ring_id_clean = str(ring_id).strip()
    return bool((df["ring_id"] == ring_id_clean).any())


def row_id_exists(row_id: int, path: Optional[str] = None) -> bool:
    """
    Check whether a row_id exists.
    """
    df = load_census(path=path)
    return bool((df["row_id"] == row_id).any())


def get_duplicate_groups(path: Optional[str] = None) -> pd.DataFrame:
    """
    Return all rows whose ring_id is duplicated, sorted by ring_id and dates.
    """
    df = load_census(path=path)

    out = df.loc[df["duplicate_ring_id_flag"].fillna(False)].copy()
    if out.empty:
        return out

    return out.sort_values(["ring_id", "date_first_seen", "date_last_seen", "row_id"])


def get_duplicate_group_sizes(path: Optional[str] = None) -> pd.DataFrame:
    """
    Return one row per duplicated ring_id with its duplicate group size.
    """
    duplicates = get_duplicate_groups(path=path)
    if duplicates.empty:
        return pd.DataFrame(columns=["ring_id", "duplicate_group_size"])

    out = (
        duplicates.groupby("ring_id", as_index=False)["duplicate_group_size"]
        .max()
        .sort_values(["duplicate_group_size", "ring_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return out


def get_complete_records(path: Optional[str] = None) -> pd.DataFrame:
    """
    Return rows labeled as complete.
    """
    df = load_census(path=path)
    return df.loc[df["record_status"] == "complete"].copy()


def get_records_by_status(record_status: str, path: Optional[str] = None) -> pd.DataFrame:
    """
    Return rows for a given record_status value.
    """
    df = load_census(path=path)
    status_clean = str(record_status).strip()
    return df.loc[df["record_status"] == status_clean].copy()


def list_record_status_counts(path: Optional[str] = None) -> pd.DataFrame:
    """
    Return counts of each record_status value.
    """
    df = load_census(path=path)

    out = (
        df["record_status"]
        .value_counts(dropna=False)
        .rename_axis("record_status")
        .reset_index(name="count")
    )
    return out


def get_birth_date_range(path: Optional[str] = None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """
    Return the min and max birth dates.
    """
    df = load_census(path=path)

    valid = df["date_first_seen"].dropna()
    if valid.empty:
        return None, None

    return valid.min(), valid.max()


def get_death_date_range(path: Optional[str] = None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """
    Return the min and max absorption dates.
    """
    df = load_census(path=path)

    valid = df["date_last_seen"].dropna()
    if valid.empty:
        return None, None

    return valid.min(), valid.max()