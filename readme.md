# WCR Agent

WCR Agent is an analysis platform for a **Warm Core Ring (WCR) census dataset**.

It combines:

- a preprocessing pipeline that converts the raw master CSV into clean analysis tables
- a multi-page Streamlit app for exploratory analysis and visualization
- a FastAPI service for programmatic filtering and summarization
- an agent-oriented orchestration layer that maps natural language requests to approved analysis tools

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Project File Structure](#project-file-structure)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Prepare the Dataset](#prepare-the-dataset)
- [Run the Streamlit App](#run-the-streamlit-app)
- [Run the API](#run-the-api)
- [Chat and Orchestration Notes](#chat-and-orchestration-notes)
- [API Endpoints](#api-endpoints)
- [Example API Requests](#example-api-requests)
- [Data Processing Details](#data-processing-details)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Development Notes](#development-notes)

---

## What This Project Does

From the project root, a typical workflow is:

1. Place the raw file in `data/raw/`
2. Build the processed dataset with `scripts/build_wcr_census.py`
3. Explore data in Streamlit (`apps/web/Home.py`)
4. Query the dataset from API routes (`apps/api/main.py`)
5. Use natural-language prompts in the Chat page, routed through the orchestrator and tool registry

---

## Project File Structure

```text
WCR_Agent/
|-- apps/
|   |-- api/
|   |   `-- main.py
|   `-- web/
|       |-- Home.py
|       `-- pages/
|           |-- 1_Chat.py
|           |-- 2_Census_Explorer.py
|           `-- 3_Ring_Detail.py
|-- data/
|   |-- raw/
|   |   `-- WCR_Master_Dataset_as_at_061319.csv
|   |-- processed/
|   |   |-- wcr_census.parquet
|   |   |-- wcr_census_clean.csv
|   |   `-- wcr_census_validation_report.csv
|   `-- reference/
|       `-- regions.geojson
|-- scripts/
|   `-- build_wcr_census.py
|-- src/
|   `-- wcr_agent/
|       |-- __init__.py
|       |-- logging_utils.py
|       |-- agent/
|       |   |-- client.py
|       |   |-- orchestrator.py
|       |   |-- prompts.py
|       |   `-- tool_registry.py
|       |-- analysis/
|       |   |-- compare_groups.py
|       |   |-- displacement.py
|       |   |-- filter_census.py
|       |   |-- summarize_census.py
|       |   `-- yearly_counts.py
|       |-- data_access/
|       |   |-- census.py
|       |   |-- io_utils.py
|       |   `-- regions.py
|       |-- plotting/
|       |   |-- comparisons.py
|       |   |-- distributions.py
|       |   `-- maps.py
|       |-- schemas/
|       |   |-- census.py
|       |   |-- filters.py
|       |   `-- outputs.py
|       `-- tools/
|           |-- compare_groups_tool.py
|           |-- export_results_tool.py
|           |-- filter_rings_tool.py
|           |-- map_births_tool.py
|           |-- map_deaths_tool.py
|           |-- map_segments_tool.py
|           `-- summarize_rings_tool.py
|-- logs/
|   `-- wcr_agent.log
|-- pyproject.toml
|-- requirements.txt
`-- readme.md
```

---

## Technology Stack

- **pandas** and **numpy** for data manipulation
- **pyarrow** for parquet read/write
- **Streamlit** for the interactive web interface
- **Plotly** for charts and geospatial visualizations
- **FastAPI** for service endpoints
- **Pydantic** for request models
- **OpenAI Python SDK** for LLM-assisted intent parsing

---

## Installation

### Option 1: editable install (recommended)

```bash
pip install -e .
```

### Option 2: use requirements file

```bash
pip install -r requirements.txt
```

### Optional development tools

```bash
pip install pytest black ruff
```

---

## Quick Start

```bash
pip install -e .
python scripts/build_wcr_census.py
streamlit run apps/web/Home.py
```

For API mode:

```bash
uvicorn apps.api.main:app --reload
```

---

## Prepare the Dataset

Place the raw source file at:

```text
data/raw/WCR_Master_Dataset_as_at_061319.csv
```

Run preprocessing:

```bash
python scripts/build_wcr_census.py
```

Generated outputs:

- `data/processed/wcr_census.parquet`
- `data/processed/wcr_census_clean.csv`
- `data/processed/wcr_census_validation_report.csv`

---

## Run the Streamlit App

```bash
streamlit run apps/web/Home.py
```

Available pages:

- **Home**: overview, key metrics, dataset preview
- **Chat**: natural-language query interface via orchestrator
- **Census Explorer**: interactive filters, maps, distributions, yearly counts, downloads
- **Ring Detail**: focused inspection of one `row_id`, one `ring_id`, or duplicate groups

---

## Run the API

```bash
uvicorn apps.api.main:app --reload
```

Open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

---

## Chat and Orchestration Notes

- The chat page routes user prompts through `src/wcr_agent/agent/orchestrator.py`.
- The orchestrator applies filters and delegates to approved tools from the tool registry.
- The parser attempts an LLM intent parse first (`use_llm_parser=True`) and falls back to rule-based parsing if LLM parsing fails.
- If you want LLM parsing to work, set:

```bash
export OPENAI_API_KEY=<your_key>
```

Without an API key, the app still works using fallback parsing.

---

## API Endpoints

### GET `/`

API root information.

### GET `/health`

Service and dataset load health.

### GET `/dataset/summary`

High-level summary for the processed census dataset.

### POST `/analysis/filter-rings`

Returns rows matching filter criteria.

### POST `/analysis/summarize`

Returns summary statistics for a filtered subset.

### GET `/rings/{row_id}`

Returns one row by `row_id`; optionally include all rows with the same `ring_id`.

---

## Example API Requests

### Filter rings born after 2000 with lifetime >= 30 days

```json
{
  "birth_date_start": "2000-01-01",
  "min_lifetime_days": 30,
  "limit": 100
}
```

### Filter duplicate rows

```json
{
  "duplicate_ring_id_flag": true,
  "limit": 20
}
```

### Summarize large rings

```json
{
  "min_area_km2": 50000
}
```

### Summarize complete records only

```json
{
  "record_status": "complete"
}
```

---

## Data Processing Details

Raw-to-standard column mapping:

- `WCR_name` -> `ring_id`
- `Date.of.Birth` -> `date_first_seen`
- `Latitude.x` -> `lat_birth`
- `Longitude.x` -> `lon_birth`
- `Area.sq.km..x` -> `area_km2`
- `Date.of.Absorption` -> `date_last_seen`
- `Latitude.y` -> `lat_death`
- `Longitude.y` -> `lon_death`

Derived fields include:

- `row_id`
- `duplicate_ring_id_flag`, `duplicate_group_size`
- `lifetime_days`
- `birth_year`, `birth_month`, `death_year`, `death_month`
- `delta_lat`, `delta_lon`
- `radius_equiv_km`
- `displacement_km`
- `bearing_birth_to_death`
- `birth_region`, `death_region` (assigned via bounding-box logic)
- `record_status`

Record status values:

- `complete`
- `missing_absorption`
- `missing_demise_location`
- `missing_absorption_and_demise_location`
- `invalid_negative_lifetime`
- `duplicate_ring_id`

---

## Known Limitations

1. The dataset is census-style, not a full trajectory product.
2. Region assignment is currently simple bounding-box classification, not polygon-based geospatial zoning.
3. Duplicate rows are flagged and preserved; no automatic deduplication policy is applied.
4. Automated tests are still minimal and should be expanded.

---

## Troubleshooting

### `streamlit: command not found`

```bash
pip install streamlit
```

### `ModuleNotFoundError: No module named 'wcr_agent'`

```bash
pip install -e .
```

### Missing processed parquet file

```bash
python scripts/build_wcr_census.py
```

Expected file:

```text
data/processed/wcr_census.parquet
```

### Parquet read/write errors

```bash
pip install pyarrow
```

### Chat does not use LLM parsing

Set an OpenAI key:

```bash
export OPENAI_API_KEY=<your_key>
```

---

## Development Notes

The project is intentionally split into layers:

- `data_access`: loading and low-level data retrieval
- `analysis`: deterministic analytical logic
- `plotting`: figure generation only
- `tools`: reusable analysis operations exposed to orchestrator
- `apps`: Streamlit and FastAPI interfaces
- `agent`: intent parsing and tool orchestration

This keeps analysis logic out of UI code and prompts, which improves testing, reproducibility, and maintainability.