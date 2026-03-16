# src/wcr_agent/tools/summarize_rings_tool.py

from __future__ import annotations

import pandas as pd

from wcr_agent.analysis.summarize_census import summarize_subset


def run_summarize_rings_tool(
    df: pd.DataFrame,
) -> dict:
    """
    Summarize the provided census subset.
    """
    summary = summarize_subset(df)

    return {
        "tool_name": "summarize_rings",
        "n_rows_matched": int(len(df)),
        "summary": summary,
        "data": df,
    }