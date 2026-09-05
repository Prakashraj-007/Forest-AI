"""
Analysis Service -- Arbitrary AOI Biomass Estimation (Phase 4).

Wraps the existing Phase 1/2 Earth Engine + Random Forest pipeline so that
any user-drawn GeoJSON polygon can be used as the Area of Interest instead of
the hardcoded Bandipur bounding box.

Design constraints
------------------
* Does NOT modify any Phase 1/2/3 service modules.
* Reuses:  initialize_earth_engine, build_biomass_feature_stack (ee_service)
*          load_biomass_model (biomass_model)
*          get_demo_biomass_result (demo_data)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import (
    ALL_STACK_BANDS,
    BIOMASS_UNIT,
    DEMO_MODE,
)
from demo_data import get_demo_biomass_result
from ee_service import (
    build_biomass_feature_stack,
    initialize_earth_engine,
)
from biomass_model import load_biomass_model, load_biomass_model_v2

logger = logging.getLogger("analysis_service")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _calculate_area_ha(geojson_coords: List) -> float:
    """Estimate polygon area in hectares using the Shoelace formula."""
    ring = geojson_coords[0]
    if len(ring) < 3:
        return 0.0
    lat_avg = sum(p[1] for p in ring) / len(ring)
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_avg))
    meters_per_deg_lat = 111_320.0
    area_sq_deg = 0.0
    n = len(ring)
    for i in range(n - 1):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[i + 1][0], ring[i + 1][1]
        area_sq_deg += x1 * y2 - x2 * y1
    area_sq_deg = abs(area_sq_deg) / 2.0
    area_sq_m = area_sq_deg * meters_per_deg_lon * meters_per_deg_lat
    return round(area_sq_m / 10_000.0, 4)


def _validate_geometry(geometry: Dict) -> Tuple[bool, str]:
    """Basic GeoJSON polygon validation."""
    if not isinstance(geometry, dict):
        return False, "geometry must be a JSON object"
    if geometry.get("type") != "Polygon":
        return False, f"geometry.type must be 'Polygon', got '{geometry.get('type')}'"
    coords = geometry.get("coordinates")
    if not coords or not isinstance(coords, list):
        return False, "geometry.coordinates is missing or empty"
    ring = coords[0]
    if len(ring) < 4:
        return False, "Polygon ring must have at least 4 coordinate pairs"
    area_ha = _calculate_area_ha(coords)
    if area_ha > 50_000_000:
        return False, f"Area ({area_ha:.0f} ha) exceeds maximum (50,000,000 ha)"
    if area_ha < 0.01:
        return False, f"Area ({area_ha:.4f} ha) is too small. Draw a larger polygon."
    return True, ""


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_aoi_biomass_analysis(
    geometry: Dict,
    start_date: str = "2023-01-01",
    end_date: str = "2023-04-30",
    force_demo: bool = False,
) -> Dict[str, Any]:
    """Run biomass estimation for an arbitrary user-drawn polygon AOI."""
    valid, err = _validate_geometry(geometry)
    if not valid:
        return {"status": "error", "mode": "ERROR", "error": err}

    coords = geometry["coordinates"]
    if coords[0][0] != coords[0][-1]:
        coords[0] = coords[0] + [coords[0][0]]

    area_ha = _calculate_area_ha(coords)

    if force_demo or DEMO_MODE:
        demo = get_demo_biomass_result()
        demo["aoi_area_ha"] = area_ha
        demo["aoi_geometry"] = geometry
        demo["start_date"] = start_date
        demo["end_date"] = end_date
        demo["mode"] = "DEMO_MODE"
        demo["prediction_statistics"]["study_area"] = f"User-selected AOI ({area_ha:.1f} ha)"
        return demo

    try:
        if not initialize_earth_engine():
            demo = get_demo_biomass_result()
            demo["aoi_area_ha"] = area_ha
            demo["aoi_geometry"] = geometry
            demo["mode"] = "DEMO_MODE"
            demo["fallback_reason"] = "Earth Engine initialization failed"
            return demo

        import ee
        import pandas as pd

        aoi = ee.Geometry.Polygon(coords)
        logger.info(f"Building EE feature stack for AOI ({area_ha:.1f} ha)...")
        feature_stack, stack_meta = build_biomass_feature_stack(
            aoi=aoi, start_date=start_date, end_date=end_date
        )

        # Load multi-biome v2 model if available, fallback to v1 model
        model, model_meta = load_biomass_model_v2()
        if model is None:
            logger.info("v2 model not found; falling back to v1 biomass model...")
            model, model_meta = load_biomass_model()

        if model is None:
            demo = get_demo_biomass_result()
            demo["aoi_area_ha"] = area_ha
            demo["aoi_geometry"] = geometry
            demo["mode"] = "DEMO_MODE"
            demo["fallback_reason"] = "No trained model found."
            return demo

        # Adaptive 15x15 grid over the AOI bbox (225 sample points)
        bbox = aoi.bounds().getInfo()["coordinates"][0]
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        min_lon, max_lon = min(xs), max(xs)
        min_lat, max_lat = min(ys), max(ys)
        
        num_steps = 15
        grid_step_lon = max((max_lon - min_lon) / num_steps, 0.001)
        grid_step_lat = max((max_lat - min_lat) / num_steps, 0.001)
        lons = [min_lon + (i + 0.5) * grid_step_lon for i in range(num_steps)]
        lats = [min_lat + (j + 0.5) * grid_step_lat for j in range(num_steps)]

        pts = [ee.Feature(ee.Geometry.Point([float(lo), float(la)])) for lo in lons for la in lats]
        if not pts:
            raise ValueError("AOI too small to sample grid points.")

        pts_fc = ee.FeatureCollection(pts)
        res = feature_stack.sampleRegions(collection=pts_fc, scale=100, geometries=False).getInfo()
        records = [f["properties"] for f in res.get("features", [])]
        if not records:
            raise ValueError("No EE feature values retrieved for the AOI.")

        df = pd.DataFrame(records)
        default_band_values = {
            "canopy_height": 0.0,
            "NDVI": 0.05,
            "EVI": 0.05,
            "SAVI": 0.05,
            "NDWI": -0.10,
            "VV": -16.0,
            "VH": -23.0,
            "VV_VH_diff": 7.0,
            "VV_VH_ratio_lin": 5.01,
            "RVI": 0.66,
            "B2": 1000.0,
            "B3": 1200.0,
            "B4": 1400.0,
            "B8": 1600.0,
            "B11": 2000.0,
            "B12": 2200.0
        }
        for b in ALL_STACK_BANDS:
            def_val = default_band_values.get(b, 0.0)
            if b not in df.columns:
                df[b] = def_val
            else:
                med_val = df[b].median()
                df[b] = df[b].fillna(med_val if (not df[b].isna().all() and not pd.isna(med_val)) else def_val)

        preds = np.clip(model.predict(df[ALL_STACK_BANDS]), a_min=0.0, a_max=None)
        pred_stats = {
            "min": round(float(np.min(preds)), 2),
            "max": round(float(np.max(preds)), 2),
            "mean": round(float(np.mean(preds)), 2),
            "median": round(float(np.median(preds)), 2),
            "std": round(float(np.std(preds)), 2),
            "valid_cells": int(len(preds)),
            "unit": BIOMASS_UNIT,
            "study_area": f"User-selected AOI ({area_ha:.1f} ha)",
        }

        # Calculate actual AOI feature means
        ch_mean = float(df["canopy_height"].mean()) if "canopy_height" in df.columns else 0.0
        ndvi_mean = float(df["NDVI"].mean()) if "NDVI" in df.columns else 0.0
        vh_mean = float(df["VH"].mean()) if "VH" in df.columns else 0.0

        feature_means = {
            "canopy_height_mean": round(ch_mean, 3),
            "ndvi_mean": round(ndvi_mean, 3),
            "vh_mean": round(vh_mean, 3),
        }

        is_arid = (ch_mean < 1.0) and (ndvi_mean < 0.20)
        biome_context = {
            "detected_biome": "arid / desert ecosystem" if is_arid else "vegetated / forest ecosystem",
            "is_arid": is_arid,
            "disclaimer": (
                "Notice: This AOI exhibits low canopy height (<1.0m) and low NDVI (<0.20), characteristic of arid/desert regions. "
                "Biomass predictions are produced by the multi-biome RF model (v2) trained on NASA/ORNL DAAC global carbon density data. "
                "All biomass estimates assume a 0.50 carbon fraction (IPCC Tier 1 standard)."
            ) if is_arid else None
        }

        return {
            "status": "success",
            "mode": "LIVE_EARTH_ENGINE",
            "aoi_area_ha": area_ha,
            "aoi_geometry": geometry,
            "start_date": start_date,
            "end_date": end_date,
            "biomass_unit": BIOMASS_UNIT,
            "model_status": model_meta.get("model_version", "v1-cached-rf"),
            "model_version": model_meta.get("model_version", "v1.0"),
            "reference_dataset": model_meta.get("reference_dataset", "NASA/ORNL DAAC"),
            "model_config": model_meta.get("hyperparameters", {}),
            "validation_metrics": model_meta.get("overall_validation_metrics", model_meta.get("validation_metrics", {})),
            "per_biome_metrics": model_meta.get("per_biome_metrics", {}),
            "feature_importance": model_meta.get("feature_importance", []),
            "feature_means": feature_means,
            "biome_context": biome_context,
            "quality_interpretation": model_meta.get("quality_summary", model_meta.get("quality_interpretation", "")),
            "prediction_statistics": pred_stats,
        }

    except Exception as exc:
        logger.error(f"AOI biomass analysis failed: {exc}", exc_info=True)
        if DEMO_MODE:
            demo = get_demo_biomass_result()
            demo["aoi_area_ha"] = area_ha
            demo["aoi_geometry"] = geometry
            demo["mode"] = "DEMO_MODE"
            demo["fallback_reason"] = f"EE error: {str(exc)}"
            demo["prediction_statistics"]["study_area"] = f"User-selected AOI ({area_ha:.1f} ha)"
            return demo
        return {
            "status": "error",
            "mode": "ERROR",
            "error": f"Earth Engine biomass analysis failed for this region: {str(exc)}",
            "aoi_area_ha": area_ha,
            "aoi_geometry": geometry
        }

