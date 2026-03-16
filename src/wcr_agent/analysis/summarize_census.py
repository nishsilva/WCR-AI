# src/wcr_agent/analysis/summarize_census.py

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


NUMERIC_SUMMARY_COLUMNS = [
    "area_km2",
    "radius_equiv_km",
    "lifetime_days",
    "displacement_km",
    "lat_birth",
    "lon_birth",
    "lat_death",
    "lon_death",
]

DATE_SUMMARY_COLUMNS = [
    "date_first_seen",
    "date_last_seen",
]

CATEGORICAL_SUMMARY_COLUMNS = [
    "record_status",
    "birth_region",
    "death_region",
    "duplicate_ring_id_flag",
]


def _safe_date_iso(value) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _safe_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _safe_int(value) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _numeric_summary(series: pd.Series) -> dict:
    """
    Return robust summary statistics for one numeric column.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()

    if s.empty:
        return {
            "count_non_null": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
        }

    return {
        "count_non_null": int(s.shape[0]),
        "mean": _safe_float(s.mean()),
        "std": _safe_float(s.std(ddof=1)) if len(s) > 1 else None,
        "min": _safe_float(s.min()),
        "q25": _safe_float(s.quantile(0.25)),
        "median": _safe_float(s.median()),
        "q75": _safe_float(s.quantile(0.75)),
        "max": _safe_float(s.max()),
    }


def _date_summary(series: pd.Series) -> dict:
    """
    Return min/max summary for one datetime-like column.
    """
    s = pd.to_datetime(series, errors="coerce").dropna()

    if s.empty:
        return {
            "count_non_null": 0,
            "min": None,
            "max": None,
        }

    return {
        "count_non_null": int(s.shape[0]),
        "min": _safe_date_iso(s.min()),
        "max": _safe_date_iso(s.max()),
    }


def _value_counts_summary(
    series: pd.Series,
    top_n: int = 10,
    dropna: bool = False,
) -> list[dict]:
    """
    Return top-N value counts as a list of dictionaries.
    """
    counts = (
        series.value_counts(dropna=dropna)
        .head(top_n)
        .rename_axis("value")
        .reset_index(name="count")
    )

    out = []
    for _, row in counts.iterrows():
        value = row["value"]
        if pd.isna(value):
            value = None
        elif isinstance(value, (np.bool_, bool)):
            value = bool(value)
        else:
            value = str(value)

        out.append(
            {
                "value": value,
                "count": int(row["count"]),
            }
        )
    return out


def summarize_rings(df: pd.DataFrame) -> dict:
    """
    Return a compact high-level summary for a census subset.
    """
    n_rows = len(df)

    summary = {
        "n_rows": int(n_rows),
        "n_unique_ring_id": int(df["ring_id"].nunique(dropna=True)) if "ring_id" in df.columns else 0,
        "n_duplicate_rows": int(df["duplicate_ring_id_flag"].fillna(False).sum())
        if "duplicate_ring_id_flag" in df.columns
        else 0,
        "n_duplicate_ring_ids": int(
            df.loc[df["duplicate_ring_id_flag"].fillna(False), "ring_id"].nunique(dropna=True)
        )
        if {"duplicate_ring_id_flag", "ring_id"}.issubset(df.columns)
        else 0,
        "n_missing_absorption_dates": int(df["date_last_seen"].isna().sum())
        if "date_last_seen" in df.columns
        else 0,
        "n_missing_demise_coordinates": int(
            (df["lat_death"].isna() | df["lon_death"].isna()).sum()
        )
        if {"lat_death", "lon_death"}.issubset(df.columns)
        else 0,
    }

    if "date_first_seen" in df.columns:
        valid_birth = pd.to_datetime(df["date_first_seen"], errors="coerce").dropna()
        summary["birth_date_min"] = _safe_date_iso(valid_birth.min()) if not valid_birth.empty else None
        summary["birth_date_max"] = _safe_date_iso(valid_birth.max()) if not valid_birth.empty else None
    else:
        summary["birth_date_min"] = None
        summary["birth_date_max"] = None

    if "date_last_seen" in df.columns:
        valid_death = pd.to_datetime(df["date_last_seen"], errors="coerce").dropna()
        summary["death_date_min"] = _safe_date_iso(valid_death.min()) if not valid_death.empty else None
        summary["death_date_max"] = _safe_date_iso(valid_death.max()) if not valid_death.empty else None
    else:
        summary["death_date_min"] = None
        summary["death_date_max"] = None

    return summary


def summarize_numeric_columns(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
) -> dict:
    """
    Return numeric summary statistics for selected columns.
    """
    columns_to_use = list(columns) if columns is not None else NUMERIC_SUMMARY_COLUMNS

    out = {}
    for col in columns_to_use:
        if col in df.columns:
            out[col] = _numeric_summary(df[col])

    return out


def summarize_date_columns(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
) -> dict:
    """
    Return date summaries for selected date columns.
    """
    columns_to_use = list(columns) if columns is not None else DATE_SUMMARY_COLUMNS

    out = {}
    for col in columns_to_use:
        if col in df.columns:
            out[col] = _date_summary(df[col])

    return out


def summarize_categorical_columns(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    *,
    top_n: int = 10,
    dropna: bool = False,
) -> dict:
    """
    Return top-N category counts for selected categorical columns.
    """
    columns_to_use = list(columns) if columns is not None else CATEGORICAL_SUMMARY_COLUMNS

    out = {}
    for col in columns_to_use:
        if col in df.columns:
            out[col] = _value_counts_summary(df[col], top_n=top_n, dropna=dropna)

    return out


def summarize_yearly_counts(
    df: pd.DataFrame,
    *,
    column: str = "birth_year",
    sort_ascending: bool = True,
) -> pd.DataFrame:
    """
    Return counts by year for birth_year or death_year.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe.")

    out = (
        df[column]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index(ascending=sort_ascending)
        .rename_axis(column)
        .reset_index(name="count")
    )

    return out


def summarize_status_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return counts by record_status.
    """
    if "record_status" not in df.columns:
        raise ValueError("Column 'record_status' not found in dataframe.")

    return (
        df["record_status"]
        .value_counts(dropna=False)
        .rename_axis("record_status")
        .reset_index(name="count")
    )


def summarize_duplicate_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per duplicated ring_id with the group size.
    """
    required = {"ring_id", "duplicate_ring_id_flag", "duplicate_group_size"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    dup_df = df.loc[df["duplicate_ring_id_flag"].fillna(False)].copy()
    if dup_df.empty:
        return pd.DataFrame(columns=["ring_id", "duplicate_group_size"])

    out = (
        dup_df.groupby("ring_id", as_index=False)["duplicate_group_size"]
        .max()
        .sort_values(["duplicate_group_size", "ring_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return out


def summarize_subset(
    df: pd.DataFrame,
    *,
    numeric_columns: Optional[Iterable[str]] = None,
    date_columns: Optional[Iterable[str]] = None,
    categorical_columns: Optional[Iterable[str]] = None,
    categorical_top_n: int = 10,
) -> dict:
    """
    Return a complete summary bundle for a filtered subset.
    """
    return {
        "overview": summarize_rings(df),
        "numeric": summarize_numeric_columns(df, columns=numeric_columns),
        "dates": summarize_date_columns(df, columns=date_columns),
        "categorical": summarize_categorical_columns(
            df,
            columns=categorical_columns,
            top_n=categorical_top_n,
        ),
    }