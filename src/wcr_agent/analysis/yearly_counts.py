# src/wcr_agent/analysis/yearly_counts.py

from __future__ import annotations

from typing import Optional

import pandas as pd


VALID_YEAR_COLUMNS = {"birth_year", "death_year"}


def _validate_year_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe.")
    if column not in VALID_YEAR_COLUMNS:
        raise ValueError(
            f"Unsupported year column '{column}'. Expected one of: {sorted(VALID_YEAR_COLUMNS)}"
        )


def yearly_counts(
    df: pd.DataFrame,
    *,
    year_column: str = "birth_year",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    fill_missing_years: bool = True,
    sort_ascending: bool = True,
    count_column_name: str = "count",
) -> pd.DataFrame:
    """
    Compute counts by year for birth_year or death_year.

    Parameters
    ----------
    df
        Input census dataframe.
    year_column
        Either 'birth_year' or 'death_year'.
    start_year, end_year
        Optional year bounds applied after extracting the year column.
    fill_missing_years
        If True, insert missing years between min/max (or start/end if provided)
        with zero counts.
    sort_ascending
        Whether to sort years ascending.
    count_column_name
        Name of the count column in the output.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [year_column, count_column_name].
    """
    _validate_year_column(df, year_column)

    year_series = pd.to_numeric(df[year_column], errors="coerce").dropna().astype(int)

    if start_year is not None:
        year_series = year_series[year_series >= int(start_year)]
    if end_year is not None:
        year_series = year_series[year_series <= int(end_year)]

    if year_series.empty:
        return pd.DataFrame(columns=[year_column, count_column_name])

    counts = (
        year_series.value_counts()
        .sort_index(ascending=sort_ascending)
        .rename_axis(year_column)
        .reset_index(name=count_column_name)
    )

    if not fill_missing_years:
        return counts.reset_index(drop=True)

    range_start = int(start_year) if start_year is not None else int(year_series.min())
    range_end = int(end_year) if end_year is not None else int(year_series.max())

    all_years = pd.DataFrame({year_column: range(range_start, range_end + 1)})
    out = all_years.merge(counts, on=year_column, how="left")
    out[count_column_name] = out[count_column_name].fillna(0).astype(int)

    if not sort_ascending:
        out = out.sort_values(year_column, ascending=False)

    return out.reset_index(drop=True)


def birth_yearly_counts(
    df: pd.DataFrame,
    *,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    fill_missing_years: bool = True,
) -> pd.DataFrame:
    """
    Convenience wrapper for yearly counts by birth year.
    """
    return yearly_counts(
        df,
        year_column="birth_year",
        start_year=start_year,
        end_year=end_year,
        fill_missing_years=fill_missing_years,
    )


def death_yearly_counts(
    df: pd.DataFrame,
    *,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    fill_missing_years: bool = True,
) -> pd.DataFrame:
    """
    Convenience wrapper for yearly counts by death year.
    """
    return yearly_counts(
        df,
        year_column="death_year",
        start_year=start_year,
        end_year=end_year,
        fill_missing_years=fill_missing_years,
    )


def yearly_counts_by_category(
    df: pd.DataFrame,
    *,
    year_column: str = "birth_year",
    category_column: str = "record_status",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    fill_missing_years: bool = True,
    fill_missing_categories: bool = True,
) -> pd.DataFrame:
    """
    Compute counts by year and category.

    Returns
    -------
    pd.DataFrame
        Columns: [year_column, category_column, count]
    """
    _validate_year_column(df, year_column)

    if category_column not in df.columns:
        raise ValueError(f"Category column '{category_column}' not found in dataframe.")

    work = df[[year_column, category_column]].copy()
    work[year_column] = pd.to_numeric(work[year_column], errors="coerce")
    work = work.dropna(subset=[year_column])
    work[year_column] = work[year_column].astype(int)

    if start_year is not None:
        work = work.loc[work[year_column] >= int(start_year)]
    if end_year is not None:
        work = work.loc[work[year_column] <= int(end_year)]

    if work.empty:
        return pd.DataFrame(columns=[year_column, category_column, "count"])

    counts = (
        work.groupby([year_column, category_column], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values([year_column, category_column], ascending=[True, True])
        .reset_index(drop=True)
    )

    if not fill_missing_years and not fill_missing_categories:
        return counts

    range_start = int(start_year) if start_year is not None else int(work[year_column].min())
    range_end = int(end_year) if end_year is not None else int(work[year_column].max())
    all_years = list(range(range_start, range_end + 1))

    categories = (
        work[category_column].drop_duplicates().tolist()
        if fill_missing_categories
        else counts[category_column].drop_duplicates().tolist()
    )

    if fill_missing_years or fill_missing_categories:
        grid = pd.MultiIndex.from_product(
            [all_years, categories],
            names=[year_column, category_column],
        ).to_frame(index=False)

        counts = grid.merge(
            counts,
            on=[year_column, category_column],
            how="left",
        )
        counts["count"] = counts["count"].fillna(0).astype(int)

    return counts.sort_values([year_column, category_column]).reset_index(drop=True)


def cumulative_yearly_counts(
    df: pd.DataFrame,
    *,
    year_column: str = "birth_year",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    fill_missing_years: bool = True,
) -> pd.DataFrame:
    """
    Compute cumulative counts over time.
    """
    counts = yearly_counts(
        df,
        year_column=year_column,
        start_year=start_year,
        end_year=end_year,
        fill_missing_years=fill_missing_years,
        sort_ascending=True,
    )

    if counts.empty:
        counts["cumulative_count"] = pd.Series(dtype="int64")
        return counts

    counts["cumulative_count"] = counts["count"].cumsum().astype(int)
    return counts


def compare_birth_vs_death_yearly_counts(
    df: pd.DataFrame,
    *,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    fill_missing_years: bool = True,
) -> pd.DataFrame:
    """
    Return a combined table of yearly birth and death counts.
    """
    births = birth_yearly_counts(
        df,
        start_year=start_year,
        end_year=end_year,
        fill_missing_years=fill_missing_years,
    ).rename(columns={"birth_year": "year", "count": "birth_count"})

    deaths = death_yearly_counts(
        df,
        start_year=start_year,
        end_year=end_year,
        fill_missing_years=fill_missing_years,
    ).rename(columns={"death_year": "year", "count": "death_count"})

    if births.empty and deaths.empty:
        return pd.DataFrame(columns=["year", "birth_count", "death_count"])

    out = births.merge(deaths, on="year", how="outer").sort_values("year").reset_index(drop=True)
    out["birth_count"] = out["birth_count"].fillna(0).astype(int)
    out["death_count"] = out["death_count"].fillna(0).astype(int)

    return out 