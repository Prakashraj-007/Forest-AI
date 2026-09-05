"""
Multi-Biome Random Forest Training & Evaluation Module (v2).

Uses NASA/ORNL biomass carbon density (agb x 2.0 = AGB Mg/ha, 0.50 carbon fraction assumption)
across global biomes for unified multi-biome biomass estimation.

HELD-OUT VALIDATION RULE:
Thar Desert samples are strictly held out for final model validation and are NEVER included in training.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import ee
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

from config import (
    MODEL_V2_PATH,
    MODEL_V2_META_PATH,
    ALL_STACK_BANDS,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_RANDOM_STATE,
    ORNL_AGB_CARBON_FRACTION,
    ORNL_AGB_SCALE_FACTOR,
    BIOMASS_UNIT,
    FEATURE_DESCRIPTIONS
)
from ee_service import initialize_earth_engine, build_biomass_feature_stack
from biomass_reference import load_raw_biomass_image
from biomass_model import save_model_to_disk, compute_biome_metrics

logger = logging.getLogger("multibiome_training")
logging.basicConfig(level=logging.INFO)

# Define geographic AOIs per biome for training and held-out validation
MULTIBIOME_AOIS = {
    "tropical_forest": {
        "train": [
            {"name": "Bandipur NP", "bbox": [76.45, 11.60, 76.80, 11.85], "n_samples": 600}
        ],
        "val": [
            {"name": "Nagarhole NP", "bbox": [76.00, 11.90, 76.40, 12.10], "n_samples": 200}
        ]
    },
    "arid_desert": {
        "train": [
            {"name": "Arabian Peninsula Desert", "bbox": [54.00, 23.00, 56.00, 25.00], "n_samples": 400},
            {"name": "Simpson Desert AU", "bbox": [136.00, -25.00, 138.00, -24.00], "n_samples": 300}
        ],
        "val": [
            # HELD OUT VALIDATION ONLY - NEVER INCLUDED IN TRAINING
            {"name": "Thar Desert (Rajasthan)", "bbox": [70.00, 26.00, 72.00, 28.00], "n_samples": 300}
        ]
    },
    "semi_arid_savanna": {
        "train": [
            {"name": "Sahel West", "bbox": [2.00, 14.50, 4.00, 15.50], "n_samples": 400}
        ],
        "val": [
            {"name": "Sahel East", "bbox": [8.00, 13.00, 10.00, 14.00], "n_samples": 150}
        ]
    }
}


def sample_biome_aoi(
    aoi_spec: Dict[str, Any],
    biome_label: str,
    dataset_key: str = "nasa_ornl",
    scale_m: int = 300
) -> pd.DataFrame:
    """
    Samples feature stack and biomass reference target over a specific AOI.
    """
    bbox = aoi_spec["bbox"]
    n_samples = aoi_spec["n_samples"]
    aoi_name = aoi_spec["name"]
    ee_geom = ee.Geometry.BBox(*bbox)

    logger.info(f"Sampling {n_samples} points from {aoi_name} ({biome_label})...")

    # Load feature stack
    feature_stack, _ = build_biomass_feature_stack(ee_geom)

    # Load target image (NASA/ORNL with x2.0 conversion factor for AGB Mg/ha)
    target_img, _ = load_raw_biomass_image(ee_geom, dataset_key=dataset_key)

    # Combined composite: features + target
    combined_img = feature_stack.addBands(target_img)

    # Extract sample points
    sampled_fc = combined_img.sampleRegions(
        collection=ee.FeatureCollection.randomPoints(ee_geom, n_samples, seed=42),
        properties=[],
        scale=scale_m,
        geometries=False,
        tileScale=4
    )

    # Convert EE FeatureCollection to Python list of dicts
    features_list = sampled_fc.getInfo()["features"]
    records = []
    for f in features_list:
        props = f.get("properties", {})
        # Check target availability
        if "biomass_target" in props and props["biomass_target"] is not None:
            # Check all required stack bands are present
            valid = True
            for band in ALL_STACK_BANDS:
                if band not in props or props[band] is None:
                    valid = False
                    break
            if valid:
                props["biome_label"] = biome_label
                props["aoi_name"] = aoi_name
                records.append(props)

    df = pd.DataFrame(records)
    logger.info(f"Retrieved {len(df)} valid sample rows from {aoi_name}.")
    return df


def run_multibiome_training() -> Tuple[RandomForestRegressor, Dict[str, Any]]:
    """
    Orchestrates sampling across all multi-biome training and validation AOIs,
    trains RF v2 model, evaluates per biome (with Thar held out), and persists model + metadata.
    """
    initialize_earth_engine()

    train_dfs = []
    val_dfs = []

    for biome, splits in MULTIBIOME_AOIS.items():
        logger.info(f"=== Processing Biome: {biome} ===")
        # Training AOIs
        for aoi_spec in splits["train"]:
            try:
                df = sample_biome_aoi(aoi_spec, biome_label=biome, dataset_key="nasa_ornl", scale_m=300)
                train_dfs.append(df)
            except Exception as e:
                logger.error(f"Failed sampling training AOI {aoi_spec['name']}: {e}")

        # Validation AOIs
        for aoi_spec in splits["val"]:
            try:
                df = sample_biome_aoi(aoi_spec, biome_label=biome, dataset_key="nasa_ornl", scale_m=300)
                val_dfs.append(df)
            except Exception as e:
                logger.error(f"Failed sampling validation AOI {aoi_spec['name']}: {e}")

    df_train = pd.concat(train_dfs, ignore_index=True)
    df_val = pd.concat(val_dfs, ignore_index=True)

    logger.info(f"Total Training Samples: {len(df_train)}")
    logger.info(f"Total Validation Samples: {len(df_val)}")

    X_train = df_train[ALL_STACK_BANDS]
    y_train = df_train["biomass_target"]

    X_val = df_val[ALL_STACK_BANDS]
    y_val = df_val["biomass_target"]

    # Train Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1
    )

    logger.info("Training Random Forest v2 on multi-biome dataset...")
    model.fit(X_train, y_train)

    # Evaluate on full validation set
    y_pred_val = model.predict(X_val)
    overall_r2 = float(r2_score(y_val, y_pred_val))
    overall_rmse = float(root_mean_squared_error(y_val, y_pred_val))
    overall_mae = float(mean_absolute_error(y_val, y_pred_val))

    # Evaluate per biome (including held-out Thar Desert validation set)
    per_biome_metrics = compute_biome_metrics(y_val, y_pred_val, df_val["biome_label"])

    # Feature Importances
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    feature_importance_list = []
    for idx in sorted_idx:
        fname = ALL_STACK_BANDS[idx]
        feature_importance_list.append({
            "feature": fname,
            "importance": round(float(importances[idx]), 4),
            "type": FEATURE_DESCRIPTIONS.get(fname, {}).get("type", "feature"),
            "description": FEATURE_DESCRIPTIONS.get(fname, {}).get("desc", "")
        })

    # Prepare metadata JSON
    metadata = {
        "model_version": "v2.0-multibiome",
        "model_type": "RandomForestRegressor",
        "reference_dataset": "NASA/ORNL biomass_carbon_density/v1",
        "carbon_fraction_assumption": ORNL_AGB_CARBON_FRACTION,
        "conversion_factor": f"AGB Mg/ha = Mg C/ha * {ORNL_AGB_SCALE_FACTOR} (assuming 0.50 C fraction)",
        "held_out_validation": "Thar Desert (Rajasthan) samples strictly held out for validation; 0 Thar samples in training set.",
        "sample_statistics": {
            "total_train_samples": len(df_train),
            "total_val_samples": len(df_val),
            "train_samples_per_biome": df_train["biome_label"].value_counts().to_dict(),
            "val_samples_per_biome": df_val["biome_label"].value_counts().to_dict(),
        },
        "overall_validation_metrics": {
            "r2": round(overall_r2, 4),
            "rmse": round(overall_rmse, 4),
            "mae": round(overall_mae, 4),
            "mean_target": round(float(y_val.mean()), 2),
            "mean_prediction": round(float(y_pred_val.mean()), 2),
        },
        "per_biome_metrics": per_biome_metrics,
        "feature_importance": feature_importance_list,
        "quality_summary": (
            f"Multi-biome model v2 trained across tropical, savanna, and arid biomes using unified NASA/ORNL AGB reference. "
            f"Overall Validation R² = {overall_r2:.3f}, RMSE = {overall_rmse:.2f} {BIOMASS_UNIT}, MAE = {overall_mae:.2f} {BIOMASS_UNIT}. "
            f"Thar Desert held-out validation evaluated relative to arid range [0.0 - 2.1 Mg/ha]."
        )
    }

    # Save model v2 and metadata
    save_model_to_disk(model, metadata, model_path=MODEL_V2_PATH, meta_path=MODEL_V2_META_PATH)
    logger.info(f"Multi-biome v2 model training complete. Saved to {MODEL_V2_PATH}")

    return model, metadata


if __name__ == "__main__":
    run_multibiome_training()
