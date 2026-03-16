# src/wcr_agent/tools/filter_rings_tool.py

from __future__ import annotations

import pandas as pd

from wcr_agent.analysis.filter_census import filter_rings


def run_filter_rings_tool(
    df: pd.DataFrame,
    **filters,
) -> dict:
    """
    Apply the standard census filter tool and return a structured payload.
    """
    filtered = filter_rings(df, **filters)

    return {
        "tool_name": "filter_rings",
        "n_rows_matched": int(len(filtered)),
        "data": filtered,
        "filters_used": filters,
    }