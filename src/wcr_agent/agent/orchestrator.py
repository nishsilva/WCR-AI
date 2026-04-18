from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import pandas as pd

from wcr_agent.logging_utils import get_logger

logger = get_logger(__name__)

from wcr_agent.agent.client import ParsedQuery, get_default_intent_parser
from wcr_agent.agent.tool_registry import get_tool
from wcr_agent.analysis.yearly_counts import (
    birth_yearly_counts,
    death_yearly_counts,
    compare_birth_vs_death_yearly_counts,
)
from wcr_agent.analysis.regime_shift import regime_shift_analysis
from wcr_agent.plotting.distributions import (
    plot_histogram,
    plot_yearly_counts_bar,
    plot_birth_vs_death_counts,
)
from wcr_agent.plotting.maps import (
    plot_birth_locations,
    plot_death_locations,
    plot_birth_to_death_segments,
)
from wcr_agent.plotting.regime_shift import plot_regime_shift

filter_tool = get_tool("filter_rings")
summarize_tool = get_tool("summarize_rings")
compare_tool = get_tool("compare_groups")


@dataclass
class OrchestratorResult:
    intent: str
    response_text: str
    data: Optional[pd.DataFrame] = None
    table: Optional[pd.DataFrame] = None
    figure: Any = None
    extra_figure: Any = None
    note: Optional[str] = None
    filters_used: Optional[dict] = None


def _run_comparison(parsed: ParsedQuery, subset: pd.DataFrame) -> OrchestratorResult:
    metric = parsed.metric or "lifetime_days"
    comparison_mode = parsed.comparison_mode

    if comparison_mode is None:
        return OrchestratorResult(
            intent="compare_groups",
            response_text="I understood this as a comparison request, but I could not determine the comparison mode.",
            data=subset,
            filters_used=parsed.filters,
        )

    payload = compare_tool(
        subset,
        comparison_mode=comparison_mode,
        metric=metric,
        custom_group_col=parsed.custom_group_col,
    )

    response_map = {
        "birth_region": f"Comparing **{metric}** by **birth region**.",
        "death_region": f"Comparing **{metric}** by **death region**.",
        "record_status": f"Comparing **{metric}** by **record status**.",
        "duplicate_flag": f"Comparing **{metric}** by **duplicate flag**.",
        "early_vs_late": f"Comparing **{metric}** for **early vs late** periods.",
        "small_vs_large": f"Comparing **{metric}** for **small vs large rings**.",
        "custom": f"Comparing **{metric}** by **{parsed.custom_group_col}**.",
    }

    response_text = response_map.get(
        comparison_mode,
        f"Comparing **{metric}** across groups.",
    )

    if payload.get("note"):
        response_text += f" {payload['note']}"

    return OrchestratorResult(
        intent="compare_groups",
        response_text=response_text,
        data=subset,
        table=payload.get("table"),
        figure=payload.get("bar_figure"),
        extra_figure=payload.get("box_figure"),
        note=payload.get("note"),
        filters_used=parsed.filters,
    )


def _run_summary(parsed: ParsedQuery, subset: pd.DataFrame) -> OrchestratorResult:
    summary_payload = summarize_tool(subset)
    overview = summary_payload["summary"]["overview"]

    response = (
        f"Found **{overview['n_rows']:,}** rows and **{overview['n_unique_ring_id']:,}** unique ring IDs.\n\n"
        f"- Duplicate rows: **{overview['n_duplicate_rows']:,}**\n"
        f"- Missing absorption dates: **{overview['n_missing_absorption_dates']:,}**\n"
        f"- Missing demise coordinates: **{overview['n_missing_demise_coordinates']:,}**\n"
        f"- Birth date range: **{overview['birth_date_min']}** to **{overview['birth_date_max']}**\n"
        f"- Death date range: **{overview['death_date_min']}** to **{overview['death_date_max']}**"
    )

    return OrchestratorResult(
        intent="summary",
        response_text=response,
        data=subset,
        filters_used=parsed.filters,
    )


def _run_map_births(parsed: ParsedQuery, subset: pd.DataFrame) -> OrchestratorResult:
    fig = plot_birth_locations(
        subset.head(500),
        color_col="lifetime_days" if "lifetime_days" in subset.columns else None,
        title=f"WCR Birth Locations (n={min(len(subset), 500):,})",
    )
    return OrchestratorResult(
        intent="map_births",
        response_text=f"Showing birth locations for **{min(len(subset), 500):,}** rows.",
        data=subset,
        figure=fig,
        filters_used=parsed.filters,
    )


def _run_map_deaths(parsed: ParsedQuery, subset: pd.DataFrame) -> OrchestratorResult:
    fig = plot_death_locations(
        subset.head(500),
        color_col="lifetime_days" if "lifetime_days" in subset.columns else None,
        title=f"WCR Death Locations (n={min(len(subset), 500):,})",
    )
    return OrchestratorResult(
        intent="map_deaths",
        response_text=f"Showing death locations for **{min(len(subset), 500):,}** rows.",
        data=subset,
        figure=fig,
        filters_used=parsed.filters,
    )


def _run_map_segments(parsed: ParsedQuery, subset: pd.DataFrame) -> OrchestratorResult:
    fig = plot_birth_to_death_segments(
        subset.head(300),
        title=f"WCR Birth-to-Death Segments (n={min(len(subset), 300):,})",
        max_segments=min(len(subset), 300),
    )
    return OrchestratorResult(
        intent="map_segments",
        response_text=f"Showing birth-to-death segments for **{min(len(subset), 300):,}** rows.",
        data=subset,
        figure=fig,
        filters_used=parsed.filters,
    )


def _run_lifetime_distribution(parsed: ParsedQuery, subset: pd.DataFrame) -> OrchestratorResult:
    fig = plot_histogram(
        subset,
        column="lifetime_days",
        nbins=25,
        title="Lifetime Distribution",
    )
    return OrchestratorResult(
        intent="lifetime_distribution",
        response_text=f"Showing lifetime distribution for **{len(subset):,}** rows.",
        data=subset,
        figure=fig,
        filters_used=parsed.filters,
    )


def _run_area_distribution(parsed: ParsedQuery, subset: pd.DataFrame) -> OrchestratorResult:
    fig = plot_histogram(
        subset,
        column="area_km2",
        nbins=25,
        title="Area Distribution",
    )
    return OrchestratorResult(
        intent="area_distribution",
        response_text=f"Showing area distribution for **{len(subset):,}** rows.",
        data=subset,
        figure=fig,
        filters_used=parsed.filters,
    )


def _run_birth_year_counts(parsed: ParsedQuery, subset: pd.DataFrame) -> OrchestratorResult:
    counts_df = birth_yearly_counts(subset)
    fig = plot_yearly_counts_bar(
        counts_df,
        year_column="birth_year",
        count_column="count",
        title="Birth Counts by Year",
    )
    return OrchestratorResult(
        intent="birth_year_counts",
        response_text=f"Showing yearly birth counts for **{len(subset):,}** rows.",
        data=subset,
        table=counts_df,
        figure=fig,
        filters_used=parsed.filters,
    )


def _run_death_year_counts(parsed: ParsedQuery, subset: pd.DataFrame) -> OrchestratorResult:
    counts_df = death_yearly_counts(subset)
    fig = plot_yearly_counts_bar(
        counts_df,
        year_column="death_year",
        count_column="count",
        title="Absorption Counts by Year",
    )
    return OrchestratorResult(
        intent="death_year_counts",
        response_text=f"Showing yearly absorption counts for **{len(subset):,}** rows.",
        data=subset,
        table=counts_df,
        figure=fig,
        filters_used=parsed.filters,
    )


def _run_birth_vs_death_counts(parsed: ParsedQuery, subset: pd.DataFrame) -> OrchestratorResult:
    compare_df = compare_birth_vs_death_yearly_counts(subset)
    fig = plot_birth_vs_death_counts(compare_df, title="Birth vs Absorption Counts by Year")
    return OrchestratorResult(
        intent="birth_vs_death_counts",
        response_text="Showing yearly birth vs absorption counts.",
        data=subset,
        table=compare_df,
        figure=fig,
        filters_used=parsed.filters,
    )


def _run_regime_shift(parsed: ParsedQuery, subset: pd.DataFrame) -> OrchestratorResult:
    result = regime_shift_analysis(subset, year_column="birth_year")

    if not result.changepoint_years:
        response_text = (
            "No significant regime shifts detected in annual birth counts "
            f"across **{len(result.counts_df)}** years of data. "
            "The series appears statistically homogeneous."
        )
    else:
        cp_str = ", ".join(str(y) for y in result.changepoint_years)
        n_regimes = len(result.segments_df)
        regime_lines = []
        for _, row in result.segments_df.iterrows():
            regime_lines.append(
                f"- **Regime {int(row['regime'])}** ({int(row['start_year'])}–{int(row['end_year'])}): "
                f"mean **{row['mean_count']}** rings/yr, {int(row['total_rings'])} total"
            )
        response_text = (
            f"Detected **{len(result.changepoint_years)}** regime shift(s) at: **{cp_str}**. "
            f"This divides the record into **{n_regimes}** activity regimes:\n\n"
            + "\n".join(regime_lines)
        )

    fig = plot_regime_shift(result)

    return OrchestratorResult(
        intent="regime_shift",
        response_text=response_text,
        data=subset,
        table=result.segments_df,
        figure=fig,
        filters_used=parsed.filters,
    )


def _run_fallback(parsed: ParsedQuery, subset: pd.DataFrame) -> OrchestratorResult:
    summary_payload = summarize_tool(subset)
    overview = summary_payload["summary"]["overview"]

    response = (
        f"I applied the current prototype filters and found **{overview['n_rows']:,}** rows.\n\n"
        "Try asking for:\n"
        "- `summary of rings after 2000`\n"
        "- `map births after 2000`\n"
        "- `map deaths of complete records`\n"
        "- `birth-to-death segments for duplicates`\n"
        "- `lifetime distribution of large rings`\n"
        "- `birth counts by year after 1990`\n"
        "- `compare lifetime by birth region`\n"
        "- `compare area by record status`\n"
        "- `compare displacement by death region`\n"
        "- `compare lifetime early vs late`\n"
    )

    return OrchestratorResult(
        intent="fallback",
        response_text=response,
        data=subset,
        filters_used=parsed.filters,
    )


def orchestrate_query(
    user_query: str,
    df: pd.DataFrame,
    *,
    use_llm_parser: bool = False,
) -> OrchestratorResult:
    logger.info("Received query: %s", user_query)

    llm_parser_error: str | None = None

    if use_llm_parser:
        try:
            parser = get_default_intent_parser(use_llm=True)
            parsed = parser.parse(user_query)
            logger.info("LLM parser succeeded | intent=%s | filters=%s", parsed.intent, parsed.filters)
        except Exception as exc:
            llm_parser_error = str(exc)
            logger.warning("LLM parser failed, using rule-based fallback: %s", exc)
            parser = get_default_intent_parser(use_llm=False)
            parsed = parser.parse(user_query)
            logger.info("Fallback parser result | intent=%s | filters=%s", parsed.intent, parsed.filters)
    else:
        parser = get_default_intent_parser(use_llm=False)
        parsed = parser.parse(user_query)
        logger.info("Rule-based parser result | intent=%s | filters=%s", parsed.intent, parsed.filters)

    filtered_payload = filter_tool(df, **parsed.filters) if parsed.filters else filter_tool(df)
    subset = filtered_payload["data"]

    logger.info("Filter result | matched_rows=%s", len(subset))

    if parsed.intent == "compare_groups":
        result = _run_comparison(parsed, subset)
    elif parsed.intent == "summary":
        result = _run_summary(parsed, subset)
    elif parsed.intent == "map_births":
        result = _run_map_births(parsed, subset)
    elif parsed.intent == "map_deaths":
        result = _run_map_deaths(parsed, subset)
    elif parsed.intent == "map_segments":
        result = _run_map_segments(parsed, subset)
    elif parsed.intent == "lifetime_distribution":
        result = _run_lifetime_distribution(parsed, subset)
    elif parsed.intent == "area_distribution":
        result = _run_area_distribution(parsed, subset)
    elif parsed.intent == "birth_year_counts":
        result = _run_birth_year_counts(parsed, subset)
    elif parsed.intent == "death_year_counts":
        result = _run_death_year_counts(parsed, subset)
    elif parsed.intent == "birth_vs_death_counts":
        result = _run_birth_vs_death_counts(parsed, subset)
    elif parsed.intent == "regime_shift":
        result = _run_regime_shift(parsed, subset)
    else:
        result = _run_fallback(parsed, subset)

    result.filters_used = parsed.filters

    if llm_parser_error:
        fallback_note = "Used fallback parser for this request."
        if result.note:
            result.note = f"{result.note} | {fallback_note}"
        else:
            result.note = fallback_note

    logger.info("Returning result | intent=%s", result.intent)
    return result