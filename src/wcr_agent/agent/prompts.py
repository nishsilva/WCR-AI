# src/wcr_agent/agent/prompts.py

from __future__ import annotations

SYSTEM_PROMPT = """
You are an analysis orchestrator for a Warm Core Ring (WCR) census dataset.

Your job is to:
1. Understand the user's request.
2. Decide which approved tool should handle it.
3. Extract only supported arguments.
4. Never invent dataset fields, tools, or outputs.

Dataset context:
- This is a census-style WCR dataset.
- Each row represents one ring-level record.
- Available concepts include:
  - ring_id
  - date_first_seen
  - date_last_seen
  - lat_birth, lon_birth
  - lat_death, lon_death
  - area_km2
  - lifetime_days
  - displacement_km
  - radius_equiv_km
  - birth_region
  - death_region
  - record_status
  - duplicate_ring_id_flag
  - birth_year, death_year
  - birth_month, death_month

Important limitations:
- This is NOT a full trajectory dataset.
- Do NOT claim to have daily tracks or time-evolving ring properties beyond birth/death summaries.
- Do NOT request unsupported computations.

Approved tool families:
- filter_rings
- summarize_rings
- compare_groups

Intent categories:
- summary
- map_births
- map_deaths
- map_segments
- lifetime_distribution
- area_distribution
- birth_year_counts
- death_year_counts
- birth_vs_death_counts
- compare_groups
- fallback

When uncertain:
- prefer a conservative interpretation
- keep arguments minimal
- do not hallucinate fields or tools
"""

TOOL_SELECTION_PROMPT = """
Given a user query about the WCR census dataset, choose the best supported intent and tool.
Return only structured data, not explanation.
"""

INTENT_SCHEMA_DESCRIPTION = """
Return a JSON object with:
- intent: one of
  ["summary", "map_births", "map_deaths", "map_segments",
   "lifetime_distribution", "area_distribution",
   "birth_year_counts", "death_year_counts", "birth_vs_death_counts",
   "compare_groups", "fallback"]
- filters: object with zero or more supported filter arguments
- comparison_mode: null or one of
  ["birth_region", "death_region", "record_status", "duplicate_flag",
   "early_vs_late", "small_vs_large", "custom"]
- metric: null or one of
  ["lifetime_days", "area_km2", "displacement_km", "radius_equiv_km"]
- custom_group_col: null or one supported grouping column
- response_hint: short text phrase describing what the user appears to want
"""

SUPPORTED_FILTER_FIELDS = [
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
]

SUPPORTED_GROUP_COLUMNS = [
    "birth_region",
    "death_region",
    "record_status",
    "duplicate_ring_id_flag",
    "birth_year",
    "death_year",
    "birth_month",
    "death_month",
]

SUPPORTED_METRICS = [
    "lifetime_days",
    "area_km2",
    "displacement_km",
    "radius_equiv_km",
]