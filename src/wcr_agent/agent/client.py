# src/wcr_agent/agent/client.py

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI

from wcr_agent.agent.prompts import (
    SYSTEM_PROMPT,
    TOOL_SELECTION_PROMPT,
    INTENT_SCHEMA_DESCRIPTION,
    SUPPORTED_FILTER_FIELDS,
    SUPPORTED_GROUP_COLUMNS,
    SUPPORTED_METRICS,
)


@dataclass
class ParsedQuery:
    intent: str
    filters: dict[str, Any]
    comparison_mode: str | None = None
    metric: str | None = None
    custom_group_col: str | None = None
    response_hint: str | None = None
    rolling_window: int = 5
    rolling_window_explicit: bool = False


class BaseIntentParser:
    def parse(self, user_query: str) -> ParsedQuery:
        raise NotImplementedError


class RuleBasedIntentParser(BaseIntentParser):
    def parse(self, user_query: str) -> ParsedQuery:
        q = user_query.lower().strip()

        filters = self._parse_simple_filters(q)
        metric = self._detect_metric(q)

        if "compare" in q:
            if "birth region" in q:
                return ParsedQuery(
                    intent="compare_groups",
                    filters=filters,
                    comparison_mode="birth_region",
                    metric=metric,
                    response_hint="compare metric by birth region",
                )

            if "death region" in q:
                return ParsedQuery(
                    intent="compare_groups",
                    filters=filters,
                    comparison_mode="death_region",
                    metric=metric,
                    response_hint="compare metric by death region",
                )

            if "record status" in q:
                return ParsedQuery(
                    intent="compare_groups",
                    filters=filters,
                    comparison_mode="record_status",
                    metric=metric,
                    response_hint="compare metric by record status",
                )

            if "duplicate flag" in q or "duplicate" in q:
                return ParsedQuery(
                    intent="compare_groups",
                    filters=filters,
                    comparison_mode="duplicate_flag",
                    metric=metric,
                    response_hint="compare metric by duplicate flag",
                )

            if "early vs late" in q or "early and late" in q:
                return ParsedQuery(
                    intent="compare_groups",
                    filters=filters,
                    comparison_mode="early_vs_late",
                    metric=metric,
                    response_hint="compare metric for early versus late periods",
                )

            if "small vs large" in q or "small and large" in q:
                return ParsedQuery(
                    intent="compare_groups",
                    filters=filters,
                    comparison_mode="small_vs_large",
                    metric=metric,
                    response_hint="compare metric for small versus large rings",
                )

        if any(word in q for word in ["birth map", "map births", "birth locations", "formation map"]):
            return ParsedQuery(
                intent="map_births",
                filters=filters,
                response_hint="show birth locations",
            )

        if any(word in q for word in ["death map", "map deaths", "death locations", "absorption map"]):
            return ParsedQuery(
                intent="map_deaths",
                filters=filters,
                response_hint="show death locations",
            )

        if any(word in q for word in ["segment", "birth-to-death", "birth to death", "displacement map"]):
            return ParsedQuery(
                intent="map_segments",
                filters=filters,
                response_hint="show birth to death segments",
            )

        if any(word in q for word in ["lifetime", "distribution", "histogram"]) and "year" not in q and "compare" not in q:
            return ParsedQuery(
                intent="lifetime_distribution",
                filters=filters,
                response_hint="show lifetime distribution",
            )

        if "area" in q and any(word in q for word in ["distribution", "histogram"]) and "compare" not in q:
            return ParsedQuery(
                intent="area_distribution",
                filters=filters,
                response_hint="show area distribution",
            )

        if "birth vs death" in q or "birth vs absorption" in q or "compare birth and death" in q:
            return ParsedQuery(
                intent="birth_vs_death_counts",
                filters=filters,
                response_hint="compare yearly birth and absorption counts",
            )

        if any(word in q for word in ["death year", "absorption year", "deaths by year", "absorptions by year"]):
            return ParsedQuery(
                intent="death_year_counts",
                filters=filters,
                response_hint="show yearly absorption counts",
            )

        if any(word in q for word in ["regime shift", "regime change", "changepoint", "structural break", "step change"]):
            window = self._detect_rolling_window(q)
            return ParsedQuery(
                intent="regime_shift",
                filters=filters,
                rolling_window=window,
                rolling_window_explicit=window != 5 or any(w in q for w in ["rolling", "window", "mean"]),
                response_hint="detect regime shifts in annual ring formation counts",
            )

        if "birth" in q and "year" in q:
            return ParsedQuery(
                intent="birth_year_counts",
                filters=filters,
                response_hint="show yearly birth counts",
            )

        if any(word in q for word in ["summary", "summarize", "overview", "how many", "count"]):
            return ParsedQuery(
                intent="summary",
                filters=filters,
                response_hint="summarize filtered subset",
            )

        return ParsedQuery(
            intent="fallback",
            filters=filters,
            response_hint="fallback response",
        )

    def _parse_years_from_text(self, text: str) -> list[int]:
        years = []
        for token in text.replace(",", " ").split():
            token = token.strip(" .;:()[]{}!?")
            if token.isdigit() and len(token) == 4:
                year = int(token)
                if 1900 <= year <= 2100:
                    years.append(year)
        return years

    def _parse_simple_filters(self, text: str) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        years = self._parse_years_from_text(text)

        if "after" in text and years:
            filters["birth_date_start"] = f"{years[0]}-01-01"
        if "before" in text and years:
            filters["birth_date_end"] = f"{years[0]}-12-31"

        if "duplicate" in text and "compare" not in text:
            filters["duplicate_ring_id_flag"] = True

        if "complete" in text:
            filters["record_status"] = "complete"

        if "missing absorption" in text:
            filters["record_status"] = "missing_absorption"

        if "missing demise" in text:
            filters["record_status"] = "missing_demise_location"

        if "long-lived" in text or "long lived" in text:
            filters["min_lifetime_days"] = 60

        if "large" in text and "compare" not in text:
            filters["min_area_km2"] = 50000

        return filters

    def _detect_rolling_window(self, text: str) -> int:
        import re
        # Match patterns like "7-year rolling", "10 year window", "rolling mean of 8", "window of 12"
        for pattern in [
            r"(\d+)[- ]year(?:s)?\s+rolling",
            r"rolling[^0-9]*(\d+)[- ]year",
            r"rolling\s+(?:mean|window)\s+(?:of\s+)?(\d+)",
            r"window\s+(?:of\s+)?(\d+)",
            r"(\d+)[- ]yr\s+rolling",
        ]:
            m = re.search(pattern, text)
            if m:
                val = int(m.group(1))
                if 2 <= val <= 20:
                    return val
        return 5

    def _detect_metric(self, text: str) -> str:
        if "lifetime" in text:
            return "lifetime_days"
        if "displacement" in text:
            return "displacement_km"
        if "radius" in text:
            return "radius_equiv_km"
        if "area" in text or "size" in text:
            return "area_km2"
        return "lifetime_days"


class LLMIntentParser(BaseIntentParser):
    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        api_key_env_var: str = "OPENAI_API_KEY",
    ) -> None:
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise ValueError(
                f"Environment variable '{api_key_env_var}' is not set. "
                "Set your OpenAI API key before using LLMIntentParser."
            )

        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def parse(self, user_query: str) -> ParsedQuery:
        schema = {
            "name": "wcr_intent_parse",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": [
                            "summary",
                            "map_births",
                            "map_deaths",
                            "map_segments",
                            "lifetime_distribution",
                            "area_distribution",
                            "birth_year_counts",
                            "death_year_counts",
                            "birth_vs_death_counts",
                            "compare_groups",
                            "regime_shift",
                            "fallback",
                        ],
                    },
                    "filters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "ring_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "record_status": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "duplicate_ring_id_flag": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
                            "birth_date_start": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "birth_date_end": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "death_date_start": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "death_date_end": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "min_area_km2": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                            "max_area_km2": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                            "min_radius_equiv_km": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                            "max_radius_equiv_km": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                            "min_lifetime_days": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                            "max_lifetime_days": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                            "min_displacement_km": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                            "max_displacement_km": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                            "min_lon_birth": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                            "max_lon_birth": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                            "min_lat_birth": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                            "max_lat_birth": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                            "min_lon_death": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                            "max_lon_death": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                            "min_lat_death": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                            "max_lat_death": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                            "birth_region": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "death_region": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "birth_year_min": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                            "birth_year_max": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                            "death_year_min": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                            "death_year_max": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                        },
                        "required": [
                            "ring_id",
                            "record_status",
                            "duplicate_ring_id_flag",
                            "birth_date_start",
                            "birth_date_end",
                            "death_date_start",
                            "death_date_end",
                            "min_area_km2",
                            "max_area_km2",
                            "min_radius_equiv_km",
                            "max_radius_equiv_km",
                            "min_lifetime_days",
                            "max_lifetime_days",
                            "min_displacement_km",
                            "max_displacement_km",
                            "min_lon_birth",
                            "max_lon_birth",
                            "min_lat_birth",
                            "max_lat_birth",
                            "min_lon_death",
                            "max_lon_death",
                            "min_lat_death",
                            "max_lat_death",
                            "birth_region",
                            "death_region",
                            "birth_year_min",
                            "birth_year_max",
                            "death_year_min",
                            "death_year_max",
                        ],
                    },
                    "comparison_mode": {
                        "anyOf": [
                            {"type": "string", "enum": [
                                "birth_region",
                                "death_region",
                                "record_status",
                                "duplicate_flag",
                                "early_vs_late",
                                "small_vs_large",
                                "custom",
                            ]},
                            {"type": "null"},
                        ]
                    },
                    "metric": {
                        "anyOf": [
                            {"type": "string", "enum": SUPPORTED_METRICS},
                            {"type": "null"},
                        ]
                    },
                    "custom_group_col": {
                        "anyOf": [
                            {"type": "string", "enum": SUPPORTED_GROUP_COLUMNS},
                            {"type": "null"},
                        ]
                    },
                    "response_hint": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ]
                    },
                    "rolling_window": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "null"},
                        ]
                    },
                },
                "required": [
                    "intent",
                    "filters",
                    "comparison_mode",
                    "metric",
                    "custom_group_col",
                    "response_hint",
                    "rolling_window",
                ],
            },
            "strict": True,
        }

        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": SYSTEM_PROMPT.strip()},
                        {"type": "input_text", "text": TOOL_SELECTION_PROMPT.strip()},
                        {"type": "input_text", "text": INTENT_SCHEMA_DESCRIPTION.strip()},
                        {
                            "type": "input_text",
                            "text": (
                                "Supported filter fields:\n- "
                                + "\n- ".join(SUPPORTED_FILTER_FIELDS)
                                + "\n\nSupported grouping columns:\n- "
                                + "\n- ".join(SUPPORTED_GROUP_COLUMNS)
                                + "\n\nSupported metrics:\n- "
                                + "\n- ".join(SUPPORTED_METRICS)
                            ),
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_query}],
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema["name"],
                    "schema": schema["schema"],
                    "strict": True,
                }
            },
        )

        parsed_json = json.loads(response.output_text)
        return self.validate_parsed_output(parsed_json)

    @staticmethod
    def validate_parsed_output(parsed: dict[str, Any]) -> ParsedQuery:
        intent = parsed.get("intent", "fallback")
        filters = parsed.get("filters", {}) or {}
        comparison_mode = parsed.get("comparison_mode")
        metric = parsed.get("metric")
        custom_group_col = parsed.get("custom_group_col")
        response_hint = parsed.get("response_hint")
        raw_window = parsed.get("rolling_window")
        rolling_window_explicit = raw_window is not None and 2 <= int(raw_window) <= 20
        rolling_window = int(raw_window) if rolling_window_explicit else 5

        filters = {
            k: v for k, v in filters.items()
            if k in SUPPORTED_FILTER_FIELDS and v is not None
        }

        if metric is not None and metric not in SUPPORTED_METRICS:
            metric = None

        if custom_group_col is not None and custom_group_col not in SUPPORTED_GROUP_COLUMNS:
            custom_group_col = None

        if comparison_mode == "custom" and custom_group_col is None:
            comparison_mode = None

        return ParsedQuery(
            intent=intent,
            filters=filters,
            comparison_mode=comparison_mode,
            metric=metric,
            custom_group_col=custom_group_col,
            response_hint=response_hint,
            rolling_window=rolling_window,
            rolling_window_explicit=rolling_window_explicit,
        )


def get_default_intent_parser(use_llm: bool = False) -> BaseIntentParser:
    if use_llm:
        return LLMIntentParser()
    return RuleBasedIntentParser()