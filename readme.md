# WCR Agent

An interactive analysis platform for exploring a **Warm Core Ring (WCR) census dataset**.

This project currently supports:

- preprocessing a raw WCR master CSV into a cleaned analysis-ready dataset
- exploring the dataset through a **Streamlit** web app
- accessing the same dataset and analysis functions through a **FastAPI** backend
- providing a foundation for a future **AI agent** that can translate natural-language questions into approved analysis workflows

---

## Table of Contents

- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Prepare the Dataset](#prepare-the-dataset)
- [What the preprocessing script does](#what-the-preprocessing-script-does)
- [Run the Streamlit App](#run-the-streamlit-app)
- [Available pages](#available-pages)
- [Run the API](#run-the-api)
- [API Endpoints](#api-endpoints)
- [Example API Requests](#example-api-requests)
- [Streamlit Pages](#streamlit-pages)
- [Data Processing Details](#data-processing-details)
- [Duplicate Ring IDs](#duplicate-ring-ids)
- [Known Limitations](#known-limitations)
- [Development Notes](#development-notes)
- [Suggested Next Steps](#suggested-next-steps)
- [Troubleshooting](#troubleshooting)
- [Quick Start Summary](#quick-start-summary)

---

## Technology Stack

### Python

- **pandas** for tabular data handling
- **numpy** for numeric operations
- **pyarrow** for parquet support
- **Streamlit** for the interactive web interface
- **Plotly** for interactive maps and charts
- **FastAPI** for the API backend
- **Pydantic** for request validation

---

## Installation

### Option 1: editable install from the repo root

```bash
pip install -e .
```

### Option 2: install core dependencies manually

```bash
pip install pandas numpy streamlit plotly pyarrow fastapi "uvicorn[standard]" pydantic
```

### Optional development tools

```bash
pip install pytest black ruff
```

---

## Getting Started

From the project root:

1. Place the raw dataset in `data/raw/`
2. Run the preprocessing script
3. Launch Streamlit or the API

---

## Prepare the Dataset

Place the raw CSV file here:

```
data/raw/WCR_Master_Dataset_as_at_061319.csv
```

Then run:

```bash
python scripts/build_wcr_census.py
```

This creates:

- `data/processed/wcr_census.parquet`
- `data/processed/wcr_census_clean.csv`
- `data/processed/wcr_census_validation_report.csv`

---

## What the preprocessing script does

The script:

- reads the raw dataset
- renames source columns to standardized internal names
- parses dates
- coerces numeric columns
- adds a unique `row_id`
- flags duplicate ring IDs
- computes lifetime in days
- computes birth and death year/month
- computes equivalent radius from area
- computes geodesic displacement
- computes initial bearing from birth to death
- assigns a simple `record_status`
- writes processed outputs

---

## Run the Streamlit App

From the repository root:

```bash
streamlit run apps/web/Home.py
```

This will open the Streamlit app in your browser.

### Available pages

- **Home** — Project overview, key metrics, dataset preview
- **Chat** — Prototype natural-language interface
- **Census Explorer** — Interactive exploration with filters
- **Ring Detail** — Detailed inspection of individual records

---

## Run the API

From the repository root:

```bash
uvicorn apps.api.main:app --reload
```

Then open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

---

## API Endpoints

### GET `/`

Basic API root response.

### GET `/health`

Returns API health and dataset load status.

### GET `/dataset/summary`

Returns summary information for the processed WCR census dataset.

### POST `/analysis/filter-rings`

Returns rows matching a set of filters.

### POST `/analysis/summarize`

Returns summary statistics for a filtered subset.

### GET `/rings/{row_id}`

Returns one row by `row_id`, with an option to include all rows sharing the same `ring_id`.

---

## Example API Requests

### Filter rings born after 2000 with lifetime at least 30 days

```json
{
  "birth_date_start": "2000-01-01",
  "min_lifetime_days": 30,
  "limit": 100
}
```

### Filter only duplicate rows

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

## Streamlit Pages

### Home

The landing page for the app.

Shows:

- a project overview
- key dataset metrics
- a preview of the processed dataset
- notes about assumptions and limitations

### Chat

A prototype natural-language interface.

Current status:

- rule-based
- not yet wired to a full LLM tool-calling backend
- useful for testing a future conversational workflow

Example prompts:

- summary of the dataset
- map births after 2000
- birth-to-death segments for duplicates
- birth counts by year after 1990

### Census Explorer

The main interactive exploration page.

Supports:

- date filtering
- area/lifetime/displacement filtering
- location bounding boxes
- record status filters
- duplicate-only filtering
- maps
- distributions
- yearly counts
- filtered data table download

### Ring Detail

Detailed inspection page for:

- a specific `row_id`
- a specific `ring_id`
- a duplicated `ring_id` group

Useful for:

- checking duplicate records
- comparing differing values across duplicate rows
- inspecting one record cleanly
- downloading selected rows

---

## Data Processing Details

### Raw source columns

The raw CSV currently contains fields like:

- `WCR_name`
- `Date.of.Birth`
- `Latitude.x`
- `Longitude.x`
- `Area.sq.km..x`
- `Date.of.Absorption`
- `Latitude.y`
- `Longitude.y`

### Standardized internal names

These are mapped to:

- `WCR_name` → `ring_id`
- `Date.of.Birth` → `date_first_seen`
- `Latitude.x` → `lat_birth`
- `Longitude.x` → `lon_birth`
- `Area.sq.km..x` → `area_km2`
- `Date.of.Absorption` → `date_last_seen`
- `Latitude.y` → `lat_death`
- `Longitude.y` → `lon_death`

### Derived metrics

**lifetime_days**

Computed as:

```
date_last_seen - date_first_seen
```

**radius_equiv_km**

Equivalent radius from area:

```
sqrt(area_km2 / pi)
```

**displacement_km**

Great-circle distance from birth coordinates to death coordinates.

**bearing_birth_to_death**

Initial bearing from birth location to death location.

### Record status rules

The preprocessing script currently assigns one of:

- `complete`
- `missing_absorption`
- `missing_demise_location`
- `missing_absorption_and_demise_location`
- `invalid_negative_lifetime`
- `duplicate_ring_id`

This gives a simple QA layer for downstream analysis.

---

## Duplicate Ring IDs

The source dataset contains a small number of duplicated `ring_id` values.

These are **not** silently removed.

Instead, the preprocessing script adds:

- `duplicate_ring_id_flag`
- `duplicate_group_size`

This allows:

- explicit filtering of duplicates
- inspection of duplicate groups in the Ring Detail page
- future development of a duplicate resolution policy

### Current policy

The current project preserves all source rows and marks them as duplicates.

**No automatic deduplication is performed.**

That is the safest choice during early development.

---

## Known Limitations

### 1. Census-only dataset

This project currently uses a census-style dataset, not a full track dataset.

### 2. No region assignment logic yet

`birth_region` and `death_region` are placeholders unless region logic is added.

### 3. No full LLM orchestration yet

The chat page is currently rule-based and does not yet use a true tool-calling agent.

### 4. Duplicate resolution is not yet implemented

Duplicates are flagged but not reconciled.

### 5. No automated tests yet

The current version is functional but still needs formal unit and integration tests.

---

## Development Notes

### Why the architecture is split this way

The project is intentionally organized so that:

- data access only loads and retrieves data
- analysis contains deterministic scientific logic
- plotting renders figures
- apps handle UI and API layers
- the future agent layer only orchestrates approved tools

This avoids putting scientific logic inside prompts or page code.

### Why this matters

That design makes the system:

- easier to test
- easier to debug
- more reproducible
- easier to extend into an AI-driven assistant later

---

## Suggested Next Steps

The most useful next steps are:

### Immediate

- add `.gitignore`
- add automated tests
- verify all API endpoints from `/docs`
- validate Streamlit pages against the real processed dataset

### Near term

- add region assignment logic
- add `tests/integration/test_api_routes.py`
- improve output schemas in the API
- connect the chat page to reusable tool functions rather than embedded routing logic

### Longer term

- implement a true agent orchestrator
- expose approved analysis tools through a tool registry
- add richer comparison analyses
- support full ring trajectories if a track dataset becomes available
- add environmental overlays and composite analyses

---

## Troubleshooting

### streamlit: command not found

Install Streamlit:

```bash
pip install streamlit
```

### ModuleNotFoundError: No module named 'wcr_agent'

From the repo root, install the package in editable mode:

```bash
pip install -e .
```

### FileNotFoundError for the processed parquet

Run the preprocessing script first:

```bash
python scripts/build_wcr_census.py
```

### Parquet write/read errors

Install parquet support:

```bash
pip install pyarrow
```

### API loads but returns dataset errors

Check that this file exists:

```
data/processed/wcr_census.parquet
```

### Maps do not render as expected

Check your local Plotly version and confirm it supports the current map functions used in `plotting/maps.py`.

---

## Quick Start Summary

**For the Streamlit app:**

```bash
pip install -e .
python scripts/build_wcr_census.py
streamlit run apps/web/Home.py
```

**For the API:**

```bash
uvicorn apps.api.main:app --reload
```