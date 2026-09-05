"""
FastAPI Application exposing Earth Engine Data Pipeline and Diagnostic Endpoints.
Phases 1-4: Satellite Data Acquisition, Biomass Estimation, Tree Detection, AOI Analysis.
"""

from typing import Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config import GCP_PROJECT, ALL_STACK_BANDS, FEATURE_DESCRIPTIONS, DEMO_MODE
from ee_service import verify_pipeline

app = FastAPI(
    title="Forest AI - Tree Counting & Biomass Estimation API",
    description="Earth Engine Data Acquisition & Preprocessing Engine for Global Forest Biomass Analysis",
    version="2.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "forest-ai-biomass-ee", "gcp_project": GCP_PROJECT}


@app.get("/api/v1/features", tags=["Metadata"])
def get_features_info():
    """
    Returns descriptions of all optical, vegetation index, radar SAR,
    and canopy height features in the unified stack.
    """
    return {
        "total_features": len(ALL_STACK_BANDS),
        "bands": ALL_STACK_BANDS,
        "descriptions": FEATURE_DESCRIPTIONS
    }


@app.get("/api/v1/ee-test", tags=["Earth Engine Verification"])
def run_ee_verification(
    start_date: str = Query("2023-01-01", description="Observation window start date (YYYY-MM-DD)"),
    end_date: str = Query("2023-04-30", description="Observation window end date (YYYY-MM-DD)"),
    demo: Optional[bool] = Query(False, description="Force demo mode fallback")
):
    """
    Performs live Earth Engine verification:
      - Validates Earth Engine initialization with GCP project
      - Verifies Sentinel-2 SR Harmonized imagery & cloud masking
      - Derives NDVI, EVI, SAVI, NDWI vegetation indices
      - Verifies Sentinel-1 GRD imagery & extracts VV, VH and derived features
      - Verifies ETH 10m Canopy Height & GEDI companion layer
      - Checks combined multi-band feature stack integrity
    """
    report = verify_pipeline(start_date=start_date, end_date=end_date, force_demo=bool(demo))
    return report


# =========================================================================
# Phase 2: Forest Biomass Estimation Endpoints
# =========================================================================

from biomass_service import run_biomass_pipeline
from biomass_model import load_biomass_model
from demo_data import get_demo_biomass_result


@app.get("/api/v1/biomass", tags=["Biomass Estimation"])
def get_biomass_report(
    demo: Optional[bool] = Query(False, description="Force demo mode fallback"),
    retrain: Optional[bool] = Query(False, description="Force model retraining"),
    dataset: Optional[str] = Query("whrc", description="Biomass reference dataset ('whrc', 'gedi_l4a', 'nasa_ornl')")
):
    """
    Executes or retrieves the complete Aboveground Biomass (AGB) pipeline:
      - Trained Random Forest Regression model metadata
      - Multi-sensor feature importances (optical, radar SAR, canopy height)
      - Independent validation metrics (R², RMSE, MAE in Mg/ha)
      - Area-wide prediction raster summary statistics (min, max, mean, median, std in Mg/ha)
      - Georeferenced GeoTIFF output path
    """
    report = run_biomass_pipeline(
        force_retrain=bool(retrain),
        force_demo=bool(demo),
        dataset_key=dataset or "whrc"
    )
    return report


@app.get("/api/v1/biomass/model", tags=["Biomass Estimation"])
def get_biomass_model_info(
    demo: Optional[bool] = Query(False, description="Force demo mode fallback")
):
    """
    Returns the Random Forest biomass model specifications, hyperparameters,
    validation metrics (R², RMSE, MAE), and sorted feature importances.
    """
    if demo:
        demo_rep = get_demo_biomass_result()
        return {
            "mode": "DEMO_MODE",
            "model_config": demo_rep["model_config"],
            "sample_statistics": demo_rep["sample_statistics"],
            "validation_metrics": demo_rep["validation_metrics"],
            "feature_importance": demo_rep["feature_importance"],
            "quality_interpretation": demo_rep["quality_interpretation"],
        }

    model, meta = load_biomass_model()
    if meta is None:
        # Train if not already saved
        report = run_biomass_pipeline(force_retrain=False, force_demo=False)
        return {
            "mode": report.get("mode"),
            "model_config": report.get("model_config"),
            "sample_statistics": report.get("sample_statistics"),
            "validation_metrics": report.get("validation_metrics"),
            "feature_importance": report.get("feature_importance"),
            "quality_interpretation": report.get("quality_interpretation"),
        }

    return {
        "mode": "LIVE_CACHED_MODEL",
        "model_config": meta.get("hyperparameters", {}),
        "sample_statistics": meta.get("sample_statistics", {}),
        "validation_metrics": meta.get("validation_metrics", {}),
        "feature_importance": meta.get("feature_importance", []),
        "quality_interpretation": meta.get("quality_interpretation", ""),
    }


@app.get("/api/v1/biomass/statistics", tags=["Biomass Estimation"])
def get_biomass_statistics(
    demo: Optional[bool] = Query(False, description="Force demo mode fallback")
):
    """
    Returns summary statistics of predicted aboveground biomass (Mg/ha)
    across the default study area (minimum, maximum, mean, median, standard deviation).
    """
    report = run_biomass_pipeline(force_retrain=False, force_demo=bool(demo))
    return {
        "biomass_unit": report.get("biomass_unit", "Mg/ha"),
        "statistics": report.get("prediction_statistics", {}),
        "raster_metadata": report.get("raster_output", {}),
        "mode": report.get("mode", "LIVE_EARTH_ENGINE")
    }


@app.post("/api/v1/biomass/train", tags=["Biomass Estimation"])
def train_biomass_endpoint(
    n_estimators: Optional[int] = Query(200, description="Number of trees"),
    sample_count: Optional[int] = Query(1000, description="Number of training samples"),
    dataset: Optional[str] = Query("whrc", description="Biomass reference dataset ('whrc', 'gedi_l4a')")
):
    """
    Triggers on-demand training of the Random Forest biomass regression model.
    """
    report = run_biomass_pipeline(
        force_retrain=True,
        force_demo=False,
        dataset_key=dataset or "whrc",
        n_estimators=n_estimators or 200,
        sample_count=sample_count or 1000
    )
    return {
        "status": "training_complete",
        "validation_metrics": report.get("validation_metrics"),
        "quality_interpretation": report.get("quality_interpretation"),
        "sample_statistics": report.get("sample_statistics"),
        "feature_importance": report.get("feature_importance", [])[:5]
    }


@app.post("/api/v1/biomass/retrain-multibiome", tags=["Biomass Estimation"])
def retrain_multibiome_endpoint():
    """
    Triggers multi-biome v2 Random Forest model training using NASA/ORNL DAAC reference dataset
    across tropical forest, semi-arid savanna, and arid desert ecosystems.
    Thar Desert samples are strictly held out for final validation.
    """
    from biomass_service import run_multibiome_pipeline
    report = run_multibiome_pipeline()
    return report


# =========================================================================
# Phase 3: Individual Tree Detection Endpoints (DeepForest / NEON AOP)
# =========================================================================

import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from fastapi import File, Form, HTTPException, UploadFile
from tree_detection.deepforest_service import run_tree_detection, validate_geotiff
from neon_demo_setup import setup_neon_proxy_tile


@app.get("/api/v1/geocode", tags=["Geocoding"])
def geocode_location(q: str = Query(..., description="Location or forest name to search")):
    """
    Geocode an arbitrary global location or forest query using OpenStreetMap Nominatim.
    Complies with OSM Nominatim usage policy (identifying User-Agent, timeout, low frequency).
    Returns latitude, longitude, bounding box, and display name for any global location.
    """
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter 'q' cannot be empty")

    try:
        url = f"https://nominatim.openstreetmap.org/search?format=json&q={urllib.parse.quote(query)}&limit=1"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "ForestAI-BiomassPlatform/2.0 (contact: hackathon-eval@forestai.app)",
                "Accept-Language": "en"
            }
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode())
            if data and len(data) > 0:
                item = data[0]
                lat = float(item["lat"])
                lon = float(item["lon"])
                bbox_str = item.get("boundingbox", [])
                bbox = [float(bbox_str[2]), float(bbox_str[0]), float(bbox_str[3]), float(bbox_str[1])] if len(bbox_str) == 4 else [lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1]
                return {
                    "success": True,
                    "name": item.get("name") or item.get("display_name", "").split(",")[0],
                    "display_name": item.get("display_name", query),
                    "lat": lat,
                    "lon": lon,
                    "bbox": bbox,
                    "source": "nominatim"
                }
            else:
                return {
                    "success": False,
                    "message": f"No location found for '{query}'. Please verify spelling or try another location name.",
                    "source": "nominatim"
                }
    except Exception as exc:
        return {
            "success": False,
            "message": f"Geocoding service unavailable: {str(exc)}",
            "source": "error"
        }


@app.get("/api/v1/trees", tags=["Tree Detection"])
@app.post("/api/v1/trees/detect", tags=["Tree Detection"])
def get_tree_detections(
    demo: Optional[bool] = Query(False, description="Force demo mode fallback"),
    conf: Optional[float] = Query(None, description="Detection confidence threshold (0-1)"),
    score_thresh: Optional[float] = Query(None, description="Alias for conf"),
    tiff: Optional[str]   = Query(None, description="Path to input GeoTIFF (default: NEON proxy tile)")
):
    """
    Runs the Phase 3 individual tree detection pipeline on a high-resolution
    GeoTIFF using the DeepForest pretrained RetinaNet tree-crown model.
    Defaults to the NEON AOP Florida proxy tile when no input is provided.
    NOTE: This is a PROXY DEMO using NEON AOP imagery from Florida, USA. It is NOT
    associated with any user-selected location on the map.
    """
    threshold = conf if conf is not None else score_thresh
    result = run_tree_detection(
        tiff_path=tiff or None,
        conf_thresh=threshold,
        force_demo=bool(demo),
        is_user_upload=False
    )
    return result


@app.post("/api/v1/trees/detect/upload", tags=["Tree Detection"])
async def upload_tree_detection_endpoint(
    file: UploadFile = File(...),
    score_thresh: Optional[float] = Form(None),
    conf: Optional[float] = Form(None),
    aoi_geojson: Optional[str] = Form(None),
    demo: Optional[bool] = Form(False)
):
    """
    Path A: Receives an uploaded high-resolution georeferenced RGB GeoTIFF,
    validates spatial reference and image structure, runs DeepForest individual
    tree-crown detection, and returns geographic tree predictions.
    The uploaded imagery must be a real high-resolution (ideally < 1m/px) RGB GeoTIFF.
    """
    if not file.filename.lower().endswith((".tif", ".tiff", ".geotiff")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a georeferenced RGB GeoTIFF (.tif or .tiff)."
        )

    threshold = conf if conf is not None else score_thresh
    if threshold is None:
        threshold = 0.15

    aoi_geom = None
    if aoi_geojson:
        try:
            aoi_data = json.loads(aoi_geojson)
            if isinstance(aoi_data, dict):
                aoi_geom = aoi_data.get("geometry", aoi_data)
        except Exception:
            pass

    temp_dir = tempfile.mkdtemp(prefix="forest_tree_upload_")
    temp_file_path = os.path.join(temp_dir, f"upload_{file.filename}")

    try:
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = run_tree_detection(
            tiff_path=temp_file_path,
            conf_thresh=threshold,
            force_demo=bool(demo),
            is_user_upload=True,
            aoi_geometry=aoi_geom
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error", "Tree detection failed."))

        return result

    finally:
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass


@app.get("/api/v1/trees/demo-tile", tags=["Tree Detection"])
def get_demo_tile_info(
    regenerate: Optional[bool] = Query(False, description="Force re-copy of the NEON proxy tile")
):
    """
    Sets up (or returns cached metadata for) the NEON AOP proxy demonstration
    tile used for Phase 3 tree detection.

    Tile properties:
      - Dataset: NEON AOP Orthorectified Camera Imagery
      - Site: Ordway-Swisher Biological Station (OSBS), Florida, USA
      - GSD: ~0.10 m/pixel (10 cm)
      - CRS: UTM (site-dependent EPSG)
      - License: CC0 1.0 Universal (Public Domain)
      - Source: Bundled via `deepforest` package sample data (weecology/DeepForest)
      - PROXY DEMO: This is a proxy demonstration tile. It is NOT imagery of the
        user's selected location and should never be presented as such.
    """
    meta = setup_neon_proxy_tile(force=bool(regenerate))
    return meta


# =========================================================================
# Phase 4: Arbitrary AOI Biomass Analysis & Status Endpoints
# =========================================================================

from analysis_service import run_aoi_biomass_analysis


class AOIGeometry(BaseModel):
    type: str
    coordinates: list


class AOIBiomassRequest(BaseModel):
    geometry: AOIGeometry
    start_date: Optional[str] = "2023-01-01"
    end_date: Optional[str] = "2023-04-30"


@app.post("/api/v1/analysis/biomass", tags=["Analysis"])
def analyze_aoi_biomass(request: AOIBiomassRequest):
    """
    Runs the complete biomass estimation pipeline for a user-drawn polygon AOI.

    Accepts a GeoJSON Polygon geometry (WGS-84 / EPSG:4326 lon/lat), builds the
    Phase 1 Sentinel-1 + Sentinel-2 + canopy-height feature stack for that area,
    and applies the cached Phase 2 Random Forest model to produce biomass estimates.

    Falls back to DEMO_MODE if Earth Engine is unavailable or the model is not cached.
    """
    return run_aoi_biomass_analysis(
        geometry=request.geometry.model_dump(),
        start_date=request.start_date or "2023-01-01",
        end_date=request.end_date or "2023-04-30",
    )


# =========================================================================
# Dynamic EE Satellite Layer Tile URLs
# =========================================================================

class EETileRequest(BaseModel):
    layer: str  # "s2_rgb" | "s2_ndvi" | "s1_vv" | "s1_vh" | "canopy_height" | "biomass"
    geometry: Optional[AOIGeometry] = None
    start_date: Optional[str] = "2023-01-01"
    end_date: Optional[str] = "2023-04-30"


@app.post("/api/v1/ee/tile-url", tags=["Satellite Visualization"])
def get_ee_tile_url(request: EETileRequest):
    """
    Returns a dynamic Google Earth Engine tile URL for the requested satellite layer
    clipped to the user's current AOI. The tile URL can be added directly to a
    Leaflet map as a TileLayer.

    Supported layers:
      - s2_rgb:        Sentinel-2 True Color RGB (B4/B3/B2)
      - s2_ndvi:       Sentinel-2 NDVI (Normalized Difference Vegetation Index)
      - s1_vv:         Sentinel-1 VV backscatter (dB)
      - s1_vh:         Sentinel-1 VH backscatter (dB)
      - canopy_height: ETH 10m Global Canopy Height
      - biomass:       Estimated AGB (Mg/ha) from RF model

    Falls back gracefully to DEMO_MODE if EE is unavailable.
    """
    from ee_service import get_ee_map_tile_url
    geometry = request.geometry.model_dump() if request.geometry else None
    result = get_ee_map_tile_url(
        layer=request.layer,
        geometry=geometry,
        start_date=request.start_date or "2023-01-01",
        end_date=request.end_date or "2023-04-30",
    )
    return result


@app.get("/api/v1/status", tags=["System"])
def get_status():
    """
    Returns backend service status, demo mode flag, and phase availability.
    Used by the Phase 4 frontend for startup health checks.
    """
    return {
        "backend": "healthy",
        "service": "forest-ai-biomass-ee",
        "gcp_project": GCP_PROJECT,
        "demo_mode": DEMO_MODE,
        "phases": {
            "phase1": "Earth Engine feature stack (global AOI support)",
            "phase2": "Random Forest biomass estimation (Sentinel-1 + Sentinel-2 + Canopy Height)",
            "phase3": "DeepForest RetinaNet tree detection (user upload or NEON AOP proxy demo)",
            "phase4": "Interactive AOI analysis dashboard with dynamic EE satellite layers",
        },
        "neon_proxy_disclaimer": (
            "Individual tree detection uses NEON AOP imagery from "
            "Ordway-Swisher Biological Station, Florida, USA. "
            "This is a PROXY DEMO — it is NOT associated with the user's selected location."
        ),
        "scientific_note": (
            "Biomass estimation uses Sentinel-1 + Sentinel-2 + Canopy Height → Random Forest. "
            "Individual tree detection uses high-resolution RGB → DeepForest RetinaNet. "
            "These are independent pipelines. Sentinel-1/2 do NOT count individual trees."
        ),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_excludes=["lightning_logs*", "outputs*", "data*", "frontend*", "*.pt", "*.tif", "*.geojson"]
    )
