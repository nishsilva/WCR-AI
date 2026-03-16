from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from wcr_agent.analysis.filter_census import filter_rings
from wcr_agent.analysis.summarize_census import summarize_subset
from wcr_agent.data_access.census import get_census_summary, load_census

app = FastAPI(
    title="WCR Agent API",
    version="0.1.0",
    description="API for Warm Core Ring census exploration and analysis",
)


class FilterRingsRequest(BaseModel):
    ring_id: Optional[str] = None
    record_status: Optional[str] = None
    duplicate_ring_id_flag: Optional[bool] = None

    birth_date_start: Optional[str] = None
    birth_date_end: Optional[str] = None
    death_date_start: Optional[str] = None
    death_date_end: Optional[str] = None

    min_area_km2: Optional[float] = None
    max_area_km2: Optional[float] = None
    min_radius_equiv_km: Optional[float] = None
    max_radius_equiv_km: Optional[float] = None
    min_lifetime_days: Optional[int] = None
    max_lifetime_days: Optional[int] = None
    min_displacement_km: Optional[float] = None
    max_displacement_km: Optional[float] = None

    min_lon_birth: Optional[float] = None
    max_lon_birth: Optional[float] = None
    min_lat_birth: Optional[float] = None
    max_lat_birth: Optional[float] = None

    min_lon_death: Optional[float] = None
    max_lon_death: Optional[float] = None
    min_lat_death: Optional[float] = None
    max_lat_death: Optional[float] = None

    birth_region: Optional[str] = None
    death_region: Optional[str] = None

    birth_year_min: Optional[int] = None
    birth_year_max: Optional[int] = None
    death_year_min: Optional[int] = None
    death_year_max: Optional[int] = None

    sort_by: Optional[str] = "date_first_seen"
    ascending: bool = True
    limit: int = Field(default=200, ge=1, le=5000)


class SummarizeRequest(FilterRingsRequest):
    pass


def _load_df() -> pd.DataFrame:
    try:
        return load_census()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load census data: {exc}") from exc


def _apply_filters(df: pd.DataFrame, req: FilterRingsRequest) -> pd.DataFrame:
    try:
        filtered = filter_rings(
            df,
            ring_id=req.ring_id,
            record_status=req.record_status,
            duplicate_ring_id_flag=req.duplicate_ring_id_flag,
            birth_date_start=req.birth_date_start,
            birth_date_end=req.birth_date_end,
            death_date_start=req.death_date_start,
            death_date_end=req.death_date_end,
            min_area_km2=req.min_area_km2,
            max_area_km2=req.max_area_km2,
            min_radius_equiv_km=req.min_radius_equiv_km,
            max_radius_equiv_km=req.max_radius_equiv_km,
            min_lifetime_days=req.min_lifetime_days,
            max_lifetime_days=req.max_lifetime_days,
            min_displacement_km=req.min_displacement_km,
            max_displacement_km=req.max_displacement_km,
            min_lon_birth=req.min_lon_birth,
            max_lon_birth=req.max_lon_birth,
            min_lat_birth=req.min_lat_birth,
            max_lat_birth=req.max_lat_birth,
            min_lon_death=req.min_lon_death,
            max_lon_death=req.max_lon_death,
            min_lat_death=req.min_lat_death,
            max_lat_death=req.max_lat_death,
            birth_region=req.birth_region,
            death_region=req.death_region,
            birth_year_min=req.birth_year_min,
            birth_year_max=req.birth_year_max,
            death_year_min=req.death_year_min,
            death_year_max=req.death_year_max,
            sort_by=req.sort_by,
            ascending=req.ascending,
        )
        return filtered
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid filter request: {exc}") from exc


def _serialize_df_for_json(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.where(pd.notnull(out), None)
    return out.to_dict(orient="records")


@app.get("/")
def root() -> dict:
    return {
        "message": "WCR Agent API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict:
    try:
        summary = get_census_summary()
        return {
            "status": "ok",
            "dataset_loaded": True,
            "n_rows": summary.get("n_rows", 0),
            "n_unique_ring_id": summary.get("n_unique_ring_id", 0),
        }
    except Exception as exc:
        return {
            "status": "error",
            "dataset_loaded": False,
            "detail": str(exc),
        }


@app.get("/dataset/summary")
def dataset_summary() -> dict:
    try:
        return get_census_summary()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to summarize dataset: {exc}") from exc


@app.post("/analysis/filter-rings")
def analysis_filter_rings(req: FilterRingsRequest) -> dict:
    df = _load_df()
    filtered = _apply_filters(df, req)

    limited = filtered.head(req.limit).copy()

    return {
        "n_rows_matched": int(len(filtered)),
        "n_rows_returned": int(len(limited)),
        "limit": req.limit,
        "rows": _serialize_df_for_json(limited),
    }


@app.post("/analysis/summarize")
def analysis_summarize(req: SummarizeRequest) -> dict:
    df = _load_df()
    filtered = _apply_filters(df, req)

    try:
        summary = summarize_subset(filtered)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to summarize filtered data: {exc}") from exc

    return {
        "n_rows_matched": int(len(filtered)),
        "summary": summary,
    }


@app.get("/rings/{row_id}")
def get_ring_by_row_id(
    row_id: int,
    include_all_same_ring_id: bool = Query(default=False),
) -> dict:
    df = _load_df()

    match = df.loc[df["row_id"] == row_id].copy()
    if match.empty:
        raise HTTPException(status_code=404, detail=f"No record found for row_id={row_id}")

    row = match.iloc[0]
    ring_id = row["ring_id"]

    response = {
        "row": _serialize_df_for_json(match)[0],
    }

    if include_all_same_ring_id:
        same_ring = df.loc[df["ring_id"] == ring_id].copy()
        response["same_ring_id_rows"] = _serialize_df_for_json(same_ring)

    return response