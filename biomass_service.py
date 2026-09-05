"""
Biomass Estimation Service.
Orchestrates training sample extraction, Random Forest training, evaluation,
biomass prediction raster generation, and summary statistics.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from PIL import Image
import ee

from config import (
    GCP_PROJECT,
    BANDIPUR_METADATA,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    DEFAULT_BIOMASS_DATASET,
    BIOMASS_UNIT,
    ALL_STACK_BANDS,
    FEATURE_DESCRIPTIONS,
    SAMPLE_COUNT,
    SAMPLING_SCALE,
    RANDOM_SEED,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_RANDOM_STATE,
    TEST_SPLIT_RATIO,
    MODEL_PATH,
    MODEL_META_PATH,
    OUTPUT_DIR,
    RASTER_PATH,
    RASTER_META_PATH,
    DEMO_MODE,
)
from ee_service import (
    initialize_earth_engine,
    get_bandipur_polygon,
    build_biomass_feature_stack,
)
from biomass_reference import get_biomass_reference
from biomass_model import (
    train_and_evaluate_rf,
    load_biomass_model,
)
from demo_data import get_demo_biomass_result

logger = logging.getLogger("biomass_service")


def extract_training_samples(
    aoi: ee.Geometry,
    feature_stack: ee.Image,
    biomass_img: ee.Image,
    sample_count: int = SAMPLE_COUNT,
    scale: int = SAMPLING_SCALE,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Combines the 16-band feature stack with the reference biomass target image,
    spatially samples pixels across the AOI in Earth Engine, converts to a cleaned
    pandas DataFrame, filters out missing or invalid values, and returns (X, y, metadata).
    """
    logger.info(f"Extracting up to {sample_count} training samples over Bandipur (scale={scale}m, seed={seed})...")

    # Combine predictors and target band
    target_band_name = "biomass_target"
    combined_img = feature_stack.addBands(biomass_img.select([target_band_name]))

    # Mask to ensure only valid target pixels are sampled
    valid_sample_img = combined_img.updateMask(biomass_img.select(target_band_name).mask())

    # Spatially distributed random sampling in Earth Engine
    samples_fc = valid_sample_img.sample(
        region=aoi,
        scale=scale,
        numPixels=sample_count,
        seed=seed,
        geometries=False
    )

    fc_info = samples_fc.getInfo()
    features = fc_info.get("features", [])
    raw_count = len(features)
    logger.info(f"Retrieved {raw_count} raw sample points from Earth Engine.")

    if raw_count == 0:
        raise ValueError("Earth Engine returned 0 samples. Verify AOI boundaries and mask validity.")

    # Convert to DataFrame
    records = [f["properties"] for f in features]
    df = pd.DataFrame(records)

    # Validate that all required predictor bands and target are present
    missing_cols = [b for b in ALL_STACK_BANDS if b not in df.columns]
    if missing_cols:
        raise ValueError(f"Sample data is missing required predictor columns: {missing_cols}")

    if target_band_name not in df.columns:
        raise ValueError(f"Sample data is missing target column '{target_band_name}'")

    # Drop nulls, NaNs, infinities
    df_clean = df.dropna(subset=ALL_STACK_BANDS + [target_band_name])
    df_clean = df_clean[np.isfinite(df_clean[target_band_name])]
    for b in ALL_STACK_BANDS:
        df_clean = df_clean[np.isfinite(df_clean[b])]

    valid_count = len(df_clean)
    logger.info(f"Valid samples after cleaning: {valid_count} (dropped {raw_count - valid_count} invalid).")

    if valid_count < 20:
        raise ValueError(f"Insufficient valid samples after cleaning ({valid_count}). Need at least 20.")

    X = df_clean[ALL_STACK_BANDS]
    y = df_clean[target_band_name]

    sample_meta = {
        "requested_sample_count": sample_count,
        "raw_samples_retrieved": raw_count,
        "valid_samples_cleaned": valid_count,
        "dropped_invalid_samples": raw_count - valid_count,
        "sampling_scale_meters": scale,
        "random_seed": seed,
        "target_min": round(float(y.min()), 2),
        "target_max": round(float(y.max()), 2),
        "target_mean": round(float(y.mean()), 2),
        "target_median": round(float(y.median()), 2),
    }

    return X, y, sample_meta


def generate_biomass_raster_grid(
    model: Any,
    aoi: ee.Geometry,
    feature_stack: ee.Image,
    raster_path: str = RASTER_PATH,
    meta_path: str = RASTER_META_PATH,
    grid_step_deg: float = 0.01  # ~1 km grid resolution for rapid, reliable MVP calculation
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Generates a regular grid of biomass predictions across the Bandipur AOI bounding box,
    computes spatial statistics, writes a georeferenced GeoTIFF (EPSG:4326), and saves metadata.
    """
    bbox = BANDIPUR_METADATA["bbox"]  # [min_lon, min_lat, max_lon, max_lat]
    min_lon, min_lat, max_lon, max_lat = bbox

    lons = np.arange(min_lon, max_lon + grid_step_deg/2, grid_step_deg)
    lats = np.arange(min_lat, max_lat + grid_step_deg/2, grid_step_deg)

    grid_cols = len(lons)
    grid_rows = len(lats)
    total_cells = grid_rows * grid_cols

    logger.info(f"Creating regular prediction grid: {grid_rows} rows x {grid_cols} cols ({total_cells} cells)...")

    pts = [ee.Feature(ee.Geometry.Point([float(lon), float(lat)])) for lon in lons for lat in lats]
    pts_fc = ee.FeatureCollection(pts)

    sample_fc = feature_stack.sampleRegions(collection=pts_fc, scale=100, geometries=True)
    res = sample_fc.getInfo()
    features = res.get("features", [])

    if not features:
        raise ValueError("Failed to extract feature values across prediction grid.")

    records = []
    coords = []
    for f in features:
        props = f.get("properties", {})
        c = f.get("geometry", {}).get("coordinates", [0.0, 0.0])
        records.append(props)
        coords.append(c)

    df_grid = pd.DataFrame(records)
    # Fill any minor missing bands with column median to prevent prediction dropouts
    for b in ALL_STACK_BANDS:
        if b not in df_grid.columns:
            df_grid[b] = 0.0
        else:
            df_grid[b] = df_grid[b].fillna(df_grid[b].median())

    # Predict using Random Forest model
    X_grid = df_grid[ALL_STACK_BANDS]
    preds = model.predict(X_grid)
    # Biomass cannot be negative
    preds = np.clip(preds, a_min=0.0, a_max=None)

    # Compute comprehensive raster statistics
    stats = {
        "min": round(float(np.min(preds)), 2),
        "max": round(float(np.max(preds)), 2),
        "mean": round(float(np.mean(preds)), 2),
        "median": round(float(np.median(preds)), 2),
        "std": round(float(np.std(preds)), 2),
        "valid_cells": len(preds),
        "unit": BIOMASS_UNIT,
        "study_area": BANDIPUR_METADATA["name"]
    }

    # Reshape predictions into a 2D array matching latitude/longitude bounds
    # Note: image rows correspond to decreasing latitude (top to bottom)
    grid_2d = np.full((grid_rows, grid_cols), np.nan, dtype=np.float32)

    # Map each point to the nearest grid cell
    for idx, (lon_val, lat_val) in enumerate(coords):
        c_idx = int(round((lon_val - min_lon) / grid_step_deg))
        r_idx = int(round((max_lat - lat_val) / grid_step_deg))  # row 0 = top (max_lat)
        if 0 <= r_idx < grid_rows and 0 <= c_idx < grid_cols:
            grid_2d[r_idx, c_idx] = preds[idx]

    # Fill any unfilled edge cells with median
    median_val = stats["median"]
    grid_2d = np.nan_to_num(grid_2d, nan=median_val)

    # Save as GeoTIFF with EPSG:4326 tags
    os.makedirs(os.path.dirname(raster_path), exist_ok=True)
    try:
        tif_img = Image.fromarray(grid_2d.astype(np.float32))
        # Standard GeoTIFF tags for EPSG:4326:
        # 33550: ModelPixelScaleTag (pixel_x_deg, pixel_y_deg, 0.0)
        # 33922: ModelTiepointTag (0, 0, 0, min_lon, max_lat, 0.0)
        # 34735: GeoKeyDirectoryTag for WGS-84 EPSG:4326
        tiff_tags = {
            33550: (float(grid_step_deg), float(grid_step_deg), 0.0),
            33922: (0.0, 0.0, 0.0, float(min_lon), float(max_lat), 0.0),
            34735: (1, 1, 0, 7, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326)
        }
        tif_img.save(raster_path, tiffinfo=tiff_tags)
        logger.info(f"Saved georeferenced GeoTIFF to {raster_path}")

        # Also save compressed numpy array for rapid numerical loading
        npz_path = raster_path.replace(".tif", ".npz")
        np.savez_compressed(
            npz_path,
            biomass=grid_2d,
            lons=lons,
            lats=lats,
            bbox=bbox,
            crs="EPSG:4326",
            unit=BIOMASS_UNIT
        )
    except Exception as e:
        logger.warning(f"Could not write GeoTIFF with PIL: {e}")

    raster_metadata = {
        "geotiff_path": raster_path,
        "crs": "EPSG:4326",
        "bounds": bbox,
        "resolution_deg": [grid_step_deg, grid_step_deg],
        "grid_shape": [grid_rows, grid_cols],
        "total_cells": total_cells,
        "unit": BIOMASS_UNIT,
        "statistics": stats
    }

    with open(meta_path, "w") as f:
        json.dump(raster_metadata, f, indent=2)

    return stats, raster_metadata


def run_biomass_pipeline(
    force_retrain: bool = False,
    force_demo: bool = False,
    dataset_key: str = DEFAULT_BIOMASS_DATASET,
    n_estimators: int = RF_N_ESTIMATORS,
    sample_count: int = SAMPLE_COUNT
) -> Dict[str, Any]:
    """
    Main entry point for the biomass estimation pipeline:
      1. Initializes Earth Engine (or falls back to DEMO_MODE)
      2. Loads or trains Random Forest Regressor
      3. Validates model and computes feature importances
      4. Generates raster prediction and area statistics
      5. Returns comprehensive JSON-serializable report
    """
    if force_demo or DEMO_MODE:
        logger.info("Running biomass pipeline in DEMO_MODE.")
        return get_demo_biomass_result()

    try:
        # Step 1: Initialize Earth Engine
        if not initialize_earth_engine():
            logger.warning("Earth Engine initialization failed; falling back to DEMO_MODE.")
            demo_res = get_demo_biomass_result()
            demo_res["fallback_reason"] = "EE initialization failed"
            return demo_res

        aoi = get_bandipur_polygon()

        # Step 2: Build or reuse Phase 1 feature stack
        logger.info("Building multi-sensor 16-band feature stack...")
        feature_stack, stack_meta = build_biomass_feature_stack(aoi)

        # Step 3: Check if model already trained and cached
        model = None
        model_meta = None
        if not force_retrain:
            model, model_meta = load_biomass_model()

        # Step 4: Train if not cached or force_retrain requested
        if model is None or model_meta is None:
            logger.info(f"Model not cached or force_retrain={force_retrain}. Initiating training...")

            # Retrieve biomass reference dataset
            biomass_img, ref_meta = get_biomass_reference(aoi, dataset_key=dataset_key)

            # Sample training points
            X, y, sample_meta = extract_training_samples(
                aoi=aoi,
                feature_stack=feature_stack,
                biomass_img=biomass_img,
                sample_count=sample_count,
                scale=SAMPLING_SCALE,
                seed=RANDOM_SEED
            )

            # Train Random Forest
            model, model_meta = train_and_evaluate_rf(
                X=X,
                y=y,
                feature_names=ALL_STACK_BANDS,
                n_estimators=n_estimators,
                max_depth=RF_MAX_DEPTH,
                min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                random_state=RF_RANDOM_STATE,
                test_size=TEST_SPLIT_RATIO,
                save_model=True
            )
            model_meta["reference_dataset"] = ref_meta
            model_meta["sampling"] = sample_meta

        # Step 5: Generate prediction grid and statistics
        logger.info("Generating biomass prediction grid and summary statistics...")
        pred_stats, raster_meta = generate_biomass_raster_grid(
            model=model,
            aoi=aoi,
            feature_stack=feature_stack
        )

        return {
            "status": "success",
            "mode": "LIVE_EARTH_ENGINE",
            "study_area": BANDIPUR_METADATA["name"],
            "biomass_unit": BIOMASS_UNIT,
            "model_status": "trained_and_cached",
            "reference_dataset": model_meta.get("reference_dataset", {
                "dataset_id": "WHRC/biomass/tropical",
                "units": BIOMASS_UNIT
            }),
            "sample_statistics": model_meta.get("sample_statistics", {}),
            "model_config": model_meta.get("hyperparameters", {}),
            "validation_metrics": model_meta.get("validation_metrics", {}),
            "feature_importance": model_meta.get("feature_importance", []),
            "quality_interpretation": model_meta.get("quality_interpretation", ""),
            "prediction_statistics": pred_stats,
            "raster_output": raster_meta
        }

    except Exception as e:
        logger.error(f"Live biomass pipeline failed: {e}", exc_info=True)
        demo_res = get_demo_biomass_result()
        demo_res["status"] = "fallback_success"
        demo_res["fallback_reason"] = f"Runtime Earth Engine error: {str(e)}"
        return demo_res


def run_multibiome_pipeline() -> Dict[str, Any]:
    """
    Triggers multi-biome v2 Random Forest model training and evaluation using NASA/ORNL.
    """
    from multibiome_training import run_multibiome_training
    try:
        model, metadata = run_multibiome_training()
        return {
            "status": "success",
            "message": "Multi-biome v2 biomass model successfully trained and serialized.",
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"Multi-biome pipeline training failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }

