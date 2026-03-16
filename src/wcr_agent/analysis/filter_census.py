# src/wcr_agent/analysis/filter_census.py

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


VALID_COMPARISON_OPERATORS = {"and", "or"}


def _normalize_iterable_strings(values: Optional[Iterable[str]]) -> list[str]:
    """
    Normalize an optional iterable of strings:
    - drop None
    - strip whitespace
    - remove empty strings
    """
    if values is None:
        return []

    out: list[str] = []
    for value in values:
        if value is None:
            continue
        value_clean = str(value).strip()
        if value_clean:
            out.append(value_clean)
    return out


def _apply_min_filter(
    df: pd.DataFrame,
    column: str,
    value: float | int | None,
) -> pd.Series:
    if value is None:
        return pd.Series(True, index=df.index)
    return df[column] >= value


def _apply_max_filter(
    df: pd.DataFrame,
    column: str,
    value: float | int | None,
) -> pd.Series:
    if value is None:
        return pd.Series(True, index=df.index)
    return df[column] <= value


def _apply_date_start_filter(
    df: pd.DataFrame,
    column: str,
    value: str | pd.Timestamp | None,
) -> pd.Series:
    if value is None:
        return pd.Series(True, index=df.index)

    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"Invalid start date for {column}: {value}")

    return df[column] >= timestamp


def _apply_date_end_filter(
    df: pd.DataFrame,
    column: str,
    value: str | pd.Timestamp | None,
) -> pd.Series:
    if value is None:
        return pd.Series(True, index=df.index)

    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"Invalid end date for {column}: {value}")

    return df[column] <= timestamp


def _apply_exact_match_filter(
    df: pd.DataFrame,
    column: str,
    value: str | None,
) -> pd.Series:
    if value is None:
        return pd.Series(True, index=df.index)

    value_clean = str(value).strip()
    if not value_clean:
        return pd.Series(True, index=df.index)

    return df[column].astype("string").str.strip() == value_clean


def _apply_membership_filter(
    df: pd.DataFrame,
    column: str,
    values: Optional[Iterable[str]],
) -> pd.Series:
    values_clean = _normalize_iterable_strings(values)
    if not values_clean:
        return pd.Series(True, index=df.index)

    return df[column].astype("string").str.strip().isin(values_clean)


def _apply_bbox_filter(
    df: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    min_lon: float | None = None,
    max_lon: float | None = None,
    min_lat: float | None = None,
    max_lat: float | None = None,
) -> pd.Series:
    mask = pd.Series(True, index=df.index)

    if min_lon is not None:
        mask &= df[lon_col] >= min_lon
    if max_lon is not None:
        mask &= df[lon_col] <= max_lon
    if min_lat is not None:
        mask &= df[lat_col] >= min_lat
    if max_lat is not None:
        mask &= df[lat_col] <= max_lat

    return mask


def _combine_masks(
    masks: list[pd.Series],
    how: str = "and",
) -> pd.Series:
    if how not in VALID_COMPARISON_OPERATORS:
        raise ValueError(
            f"Invalid mask combination operator '{how}'. "
            f"Expected one of: {sorted(VALID_COMPARISON_OPERATORS)}"
        )

    if not masks:
        raise ValueError("No masks were provided to combine.")

    combined = masks[0].copy()

    for mask in masks[1:]:
        if how == "and":
            combined &= mask
        else:
            combined |= mask

    return combined


def filter_rings(
    df: pd.DataFrame,
    *,
    row_ids: Optional[Iterable[int]] = None,
    ring_id: str | None = None,
    ring_ids: Optional[Iterable[str]] = None,
    record_status: str | None = None,
    record_statuses: Optional[Iterable[str]] = None,
    duplicate_ring_id_flag: bool | None = None,
    birth_date_start: str | pd.Timestamp | None = None,
    birth_date_end: str | pd.Timestamp | None = None,
    death_date_start: str | pd.Timestamp | None = None,
    death_date_end: str | pd.Timestamp | None = None,
    min_area_km2: float | None = None,
    max_area_km2: float | None = None,
    min_radius_equiv_km: float | None = None,
    max_radius_equiv_km: float | None = None,
    min_lifetime_days: int | None = None,
    max_lifetime_days: int | None = None,
    min_displacement_km: float | None = None,
    max_displacement_km: float | None = None,
    min_lon_birth: float | None = None,
    max_lon_birth: float | None = None,
    min_lat_birth: float | None = None,
    max_lat_birth: float | None = None,
    min_lon_death: float | None = None,
    max_lon_death: float | None = None,
    min_lat_death: float | None = None,
    max_lat_death: float | None = None,
    birth_region: str | None = None,
    death_region: str | None = None,
    birth_regions: Optional[Iterable[str]] = None,
    death_regions: Optional[Iterable[str]] = None,
    birth_year_min: int | None = None,
    birth_year_max: int | None = None,
    death_year_min: int | None = None,
    death_year_max: int | None = None,
    sort_by: str | None = "date_first_seen",
    ascending: bool = True,
    combine_masks_with: str = "and",
) -> pd.DataFrame:
    """
    Filter the processed WCR census dataframe.

    Parameters
    ----------
    df
        Processed census dataframe.
    row_ids
        Optional list of unique row identifiers to keep.
    ring_id, ring_ids
        Filter by one or more scientific ring IDs.
    record_status, record_statuses
        Filter by one or more record_status values.
    duplicate_ring_id_flag
        Filter duplicate vs non-duplicate records.
    birth_date_start, birth_date_end
        Inclusive date range for date_first_seen.
    death_date_start, death_date_end
        Inclusive date range for date_last_seen.
    min_area_km2, max_area_km2
        Area bounds.
    min_radius_equiv_km, max_radius_equiv_km
        Equivalent-radius bounds.
    min_lifetime_days, max_lifetime_days
        Lifetime bounds.
    min_displacement_km, max_displacement_km
        Birth-to-death displacement bounds.
    min_lon_birth, max_lon_birth, min_lat_birth, max_lat_birth
        Bounding box for birth location.
    min_lon_death, max_lon_death, min_lat_death, max_lat_death
        Bounding box for demise location.
    birth_region, death_region
        Exact match for a single region label.
    birth_regions, death_regions
        Membership match for multiple region labels.
    birth_year_min, birth_year_max, death_year_min, death_year_max
        Year bounds.
    sort_by
        Column to sort output by. Pass None to skip sorting.
    ascending
        Sort order.
    combine_masks_with
        Either 'and' or 'or'. In almost all cases you want 'and'.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe copy.
    """
    if df.empty:
        return df.copy()

    masks: list[pd.Series] = []

    # row_ids
    if row_ids is not None:
        row_ids_set = set(row_ids)
        masks.append(df["row_id"].isin(row_ids_set))

    # ring IDs
    if ring_id is not None:
        masks.append(_apply_exact_match_filter(df, "ring_id", ring_id))

    if ring_ids is not None:
        masks.append(_apply_membership_filter(df, "ring_id", ring_ids))

    # record status
    if record_status is not None:
        masks.append(_apply_exact_match_filter(df, "record_status", record_status))

    if record_statuses is not None:
        masks.append(_apply_membership_filter(df, "record_status", record_statuses))

    # duplicate flag
    if duplicate_ring_id_flag is not None:
        masks.append(df["duplicate_ring_id_flag"] == duplicate_ring_id_flag)

    # dates
    masks.append(_apply_date_start_filter(df, "date_first_seen", birth_date_start))
    masks.append(_apply_date_end_filter(df, "date_first_seen", birth_date_end))
    masks.append(_apply_date_start_filter(df, "date_last_seen", death_date_start))
    masks.append(_apply_date_end_filter(df, "date_last_seen", death_date_end))

    # scalar bounds
    masks.append(_apply_min_filter(df, "area_km2", min_area_km2))
    masks.append(_apply_max_filter(df, "area_km2", max_area_km2))
    masks.append(_apply_min_filter(df, "radius_equiv_km", min_radius_equiv_km))
    masks.append(_apply_max_filter(df, "radius_equiv_km", max_radius_equiv_km))
    masks.append(_apply_min_filter(df, "lifetime_days", min_lifetime_days))
    masks.append(_apply_max_filter(df, "lifetime_days", max_lifetime_days))
    masks.append(_apply_min_filter(df, "displacement_km", min_displacement_km))
    masks.append(_apply_max_filter(df, "displacement_km", max_displacement_km))
    masks.append(_apply_min_filter(df, "birth_year", birth_year_min))
    masks.append(_apply_max_filter(df, "birth_year", birth_year_max))
    masks.append(_apply_min_filter(df, "death_year", death_year_min))
    masks.append(_apply_max_filter(df, "death_year", death_year_max))

    # bounding boxes
    masks.append(
        _apply_bbox_filter(
            df,
            lon_col="lon_birth",
            lat_col="lat_birth",
            min_lon=min_lon_birth,
            max_lon=max_lon_birth,
            min_lat=min_lat_birth,
            max_lat=max_lat_birth,
        )
    )
    masks.append(
        _apply_bbox_filter(
            df,
            lon_col="lon_death",
            lat_col="lat_death",
            min_lon=min_lon_death,
            max_lon=max_lon_death,
            min_lat=min_lat_death,
            max_lat=max_lat_death,
        )
    )

    # regions
    if birth_region is not None:
        masks.append(_apply_exact_match_filter(df, "birth_region", birth_region))
    if death_region is not None:
        masks.append(_apply_exact_match_filter(df, "death_region", death_region))
    if birth_regions is not None:
        masks.append(_apply_membership_filter(df, "birth_region", birth_regions))
    if death_regions is not None:
        masks.append(_apply_membership_filter(df, "death_region", death_regions))

    combined_mask = _combine_masks(masks, how=combine_masks_with)
    out = df.loc[combined_mask].copy()

    if sort_by is not None:
        if sort_by not in out.columns:
            raise ValueError(
                f"sort_by='{sort_by}' is not a valid column. "
                f"Available columns include: {list(out.columns)}"
            )
        out = out.sort_values(sort_by, ascending=ascending, na_position="last")

    return out.reset_index(drop=True)


def filter_complete_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience filter for rows with record_status == 'complete'.
    """
    return filter_rings(df, record_status="complete")


def filter_duplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience filter for rows with duplicated ring IDs.
    """
    return filter_rings(df, duplicate_ring_id_flag=True)


def filter_by_birth_bbox(
    df: pd.DataFrame,
    *,
    min_lon: float | None = None,
    max_lon: float | None = None,
    min_lat: float | None = None,
    max_lat: float | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper for filtering by birth-location bounding box.
    """
    return filter_rings(
        df,
        min_lon_birth=min_lon,
        max_lon_birth=max_lon,
        min_lat_birth=min_lat,
        max_lat_birth=max_lat,
    )


def filter_by_death_bbox(
    df: pd.DataFrame,
    *,
    min_lon: float | None = None,
    max_lon: float | None = None,
    min_lat: float | None = None,
    max_lat: float | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper for filtering by demise-location bounding box.
    """
    return filter_rings(
        df,
        min_lon_death=min_lon,
        max_lon_death=max_lon,
        min_lat_death=min_lat,
        max_lat_death=max_lat,
    )


def filter_by_birth_year_range(
    df: pd.DataFrame,
    *,
    year_start: int | None = None,
    year_end: int | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper for filtering by birth year.
    """
    return filter_rings(
        df,
        birth_year_min=year_start,
        birth_year_max=year_end,
    )


def filter_by_lifetime(
    df: pd.DataFrame,
    *,
    min_lifetime_days: int | None = None,
    max_lifetime_days: int | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper for filtering by lifetime.
    """
    return filter_rings(
        df,
        min_lifetime_days=min_lifetime_days,
        max_lifetime_days=max_lifetime_days,
    )