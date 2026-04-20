# src/wcr_agent/analysis/regime_shift.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from wcr_agent.analysis.yearly_counts import yearly_counts


@dataclass
class RegimeShiftResult:
    counts_df: pd.DataFrame
    changepoint_years: list[int]
    segments_df: pd.DataFrame
    rolling_mean: pd.Series
    rolling_window: int
    year_column: str


def _rss(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.sum((values - values.mean()) ** 2))


def _find_best_split(values: np.ndarray, min_size: int) -> tuple[int, float]:
    n = len(values)
    total_rss = _rss(values)
    best_idx, best_gain = -1, 0.0
    for i in range(min_size, n - min_size + 1):
        gain = total_rss - (_rss(values[:i]) + _rss(values[i:]))
        if gain > best_gain:
            best_gain, best_idx = gain, i
    return best_idx, best_gain


def _binary_segmentation(
    values: np.ndarray,
    offset: int,
    min_size: int,
    penalty: float,
    breakpoints: list[int],
    depth: int,
    max_depth: int,
) -> None:
    if depth >= max_depth or len(values) < 2 * min_size:
        return
    idx, gain = _find_best_split(values, min_size)
    if idx == -1 or gain < penalty:
        return
    breakpoints.append(offset + idx)
    _binary_segmentation(values[:idx], offset, min_size, penalty, breakpoints, depth + 1, max_depth)
    _binary_segmentation(values[idx:], offset + idx, min_size, penalty, breakpoints, depth + 1, max_depth)


def detect_changepoints(
    values: np.ndarray,
    min_segment_size: int = 5,
    max_breakpoints: int = 4,
) -> list[int]:
    """
    Binary segmentation changepoint detection using RSS gain.
    Returns sorted list of breakpoint indices into `values`.
    Penalty is adaptive: 2 * var(values) * log(n) to avoid over-segmentation.
    """
    n = len(values)
    if n < 2 * min_segment_size:
        return []

    var = float(np.var(values))
    penalty = max(var * np.log(n) * 2, 1e-6)

    breakpoints: list[int] = []
    _binary_segmentation(values, 0, min_segment_size, penalty, breakpoints, 0, max_breakpoints)
    return sorted(breakpoints)


def regime_shift_analysis(
    df: pd.DataFrame,
    *,
    year_column: str = "birth_year",
    min_segment_size: int = 5,
    max_breakpoints: int = 4,
    rolling_window: int = 5,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> RegimeShiftResult:
    counts_df = yearly_counts(
        df,
        year_column=year_column,
        start_year=start_year,
        end_year=end_year,
        fill_missing_years=True,
        sort_ascending=True,
    )

    if counts_df.empty or len(counts_df) < 2 * min_segment_size:
        return RegimeShiftResult(
            counts_df=counts_df,
            changepoint_years=[],
            segments_df=pd.DataFrame(columns=["regime", "start_year", "end_year", "mean_count", "total_rings", "n_years"]),
            rolling_mean=pd.Series(dtype=float),
            rolling_window=rolling_window,
            year_column=year_column,
        )

    values = counts_df["count"].to_numpy(dtype=float)
    years = counts_df[year_column].to_numpy(dtype=int)

    bp_indices = detect_changepoints(values, min_segment_size=min_segment_size, max_breakpoints=max_breakpoints)

    changepoint_years = [int(years[i]) for i in bp_indices]

    # Build segment boundaries (as index positions)
    boundaries = [0] + bp_indices + [len(values)]
    segments = []
    for regime_num, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        seg_values = values[lo:hi]
        seg_years = years[lo:hi]
        segments.append({
            "regime": regime_num,
            "start_year": int(seg_years[0]),
            "end_year": int(seg_years[-1]),
            "mean_count": round(float(seg_values.mean()), 1),
            "total_rings": int(seg_values.sum()),
            "n_years": int(hi - lo),
        })

    segments_df = pd.DataFrame(segments)

    rolling_mean = (
        counts_df.set_index(year_column)["count"]
        .rolling(window=rolling_window, center=True, min_periods=1)
        .mean()
        .rename("rolling_mean")
    )

    return RegimeShiftResult(
        counts_df=counts_df,
        changepoint_years=changepoint_years,
        segments_df=segments_df,
        rolling_mean=rolling_mean,
        rolling_window=rolling_window,
        year_column=year_column,
    )
