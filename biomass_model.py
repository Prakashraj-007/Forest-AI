"""
Biomass Random Forest Model Module.
Trains, evaluates, serializes, and inspects Random Forest regression for aboveground biomass estimation.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

from config import (
    MODEL_DIR,
    MODEL_PATH,
    MODEL_META_PATH,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_RANDOM_STATE,
    TEST_SPLIT_RATIO,
    BIOMASS_UNIT,
    FEATURE_DESCRIPTIONS,
    ALL_STACK_BANDS,
)

logger = logging.getLogger("biomass_model")


def train_and_evaluate_rf(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str] = ALL_STACK_BANDS,
    n_estimators: int = RF_N_ESTIMATORS,
    max_depth: int = RF_MAX_DEPTH,
    min_samples_leaf: int = RF_MIN_SAMPLES_LEAF,
    random_state: int = RF_RANDOM_STATE,
    test_size: float = TEST_SPLIT_RATIO,
    save_model: bool = True
) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
    """
    Trains a Random Forest Regressor on the provided feature DataFrame and target series,
    evaluates on the validation set, extracts feature importances, and optionally persists to disk.
    """
    if len(X) < 20:
        raise ValueError(f"Insufficient samples for training ({len(X)} samples). Minimum 20 required.")

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X[feature_names],
        y,
        test_size=test_size,
        random_state=random_state
    )

    logger.info(
        f"Training Random Forest Regressor: n_estimators={n_estimators}, max_depth={max_depth}, "
        f"train_samples={len(X_train)}, val_samples={len(X_val)}"
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Validation predictions & metrics
    y_pred = model.predict(X_val)
    r2 = float(r2_score(y_val, y_pred))
    rmse = float(root_mean_squared_error(y_val, y_pred))
    mae = float(mean_absolute_error(y_val, y_pred))

    # Feature importances
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    feature_importance_list = []
    for idx in sorted_idx:
        fname = feature_names[idx]
        fdesc = FEATURE_DESCRIPTIONS.get(fname, {}).get("desc", "")
        ftype = FEATURE_DESCRIPTIONS.get(fname, {}).get("type", "feature")
        feature_importance_list.append({
            "feature": fname,
            "importance": round(float(importances[idx]), 4),
            "type": ftype,
            "description": fdesc
        })

    # Scientific interpretation
    if r2 > 0.70:
        quality_interpretation = (
            f"Strong regression fit (R² = {r2:.3f}). The multi-sensor optical, radar, and "
            f"canopy height features explain over {r2*100:.1f}% of aboveground biomass variance "
            f"across Bandipur National Park (RMSE = {rmse:.2f} {BIOMASS_UNIT}, MAE = {mae:.2f} {BIOMASS_UNIT})."
        )
    elif r2 > 0.40:
        quality_interpretation = (
            f"Moderate regression fit (R² = {r2:.3f}). S1/S2 and height features provide informative predictive "
            f"capability for biomass estimation, with remaining residual variance attributable to sub-pixel "
            f"crown heterogeneity or sensor resolution differences (RMSE = {rmse:.2f} {BIOMASS_UNIT})."
        )
    else:
        quality_interpretation = (
            f"Low predictive fit (R² = {r2:.3f}, RMSE = {rmse:.2f} {BIOMASS_UNIT}). Target dataset variance "
            f"or sparse sampling tracks may limit model calibration."
        )

    metadata: Dict[str, Any] = {
        "model_type": "RandomForestRegressor",
        "library": "scikit-learn",
        "target_variable": "Aboveground Biomass (AGB)",
        "unit": BIOMASS_UNIT,
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
            "test_split_ratio": test_size,
        },
        "sample_statistics": {
            "total_samples": len(X),
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "features_count": len(feature_names),
        },
        "validation_metrics": {
            "r2": round(r2, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mean_actual_val": round(float(y_val.mean()), 2),
            "mean_pred_val": round(float(y_pred.mean()), 2),
        },
        "feature_importance": feature_importance_list,
        "quality_interpretation": quality_interpretation,
    }

    if save_model:
        save_model_to_disk(model, metadata)

    return model, metadata


def save_model_to_disk(model: RandomForestRegressor, metadata: Dict[str, Any], model_path: str = MODEL_PATH, meta_path: str = MODEL_META_PATH):
    """
    Saves the trained model via joblib and its associated metadata JSON to disk.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Model saved to {model_path}, metadata saved to {meta_path}")


def load_biomass_model(model_path: str = MODEL_PATH, meta_path: str = MODEL_META_PATH) -> Tuple[Optional[RandomForestRegressor], Optional[Dict[str, Any]]]:
    """
    Loads serialized model and metadata from disk if available (v1 model).
    """
    if os.path.exists(model_path) and os.path.exists(meta_path):
        try:
            model = joblib.load(model_path)
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            logger.info(f"Loaded existing trained biomass model from {model_path}")
            return model, metadata
        except Exception as e:
            logger.warning(f"Error loading saved model from {model_path}: {e}")
            return None, None
    return None, None


def load_biomass_model_v2(model_path: str = None, meta_path: str = None) -> Tuple[Optional[RandomForestRegressor], Optional[Dict[str, Any]]]:
    """
    Loads serialized multi-biome v2 model and metadata from disk if available.
    """
    from config import MODEL_V2_PATH, MODEL_V2_META_PATH
    m_path = model_path or MODEL_V2_PATH
    meta_p = meta_path or MODEL_V2_META_PATH
    if os.path.exists(m_path) and os.path.exists(meta_p):
        try:
            model = joblib.load(m_path)
            with open(meta_p, "r") as f:
                metadata = json.load(f)
            logger.info(f"Loaded trained multi-biome v2 biomass model from {m_path}")
            return model, metadata
        except Exception as e:
            logger.warning(f"Error loading saved v2 model from {m_path}: {e}")
            return None, None
    return None, None


def compute_biome_metrics(y_true: pd.Series, y_pred: np.ndarray, biome_series: pd.Series) -> Dict[str, Dict[str, Any]]:
    """
    Computes per-biome evaluation metrics (R2, RMSE, MAE, sample count, target range, prediction range, mean target, mean pred).
    Evaluates errors relative to each biome's target range without imposing a fixed RMSE cutoff.
    """
    biome_metrics = {}
    df_eval = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "biome": biome_series})

    for biome, group in df_eval.groupby("biome"):
        n_samples = len(group)
        if n_samples == 0:
            continue
        yt = group["y_true"]
        yp = group["y_pred"]

        r2 = float(r2_score(yt, yp)) if n_samples > 1 and yt.nunique() > 1 else 0.0
        rmse = float(root_mean_squared_error(yt, yp))
        mae = float(mean_absolute_error(yt, yp))

        target_min, target_max = float(yt.min()), float(yt.max())
        pred_min, pred_max = float(yp.min()), float(yp.max())
        mean_target = float(yt.mean())
        mean_pred = float(yp.mean())
        target_range = target_max - target_min

        # Relative error interpretation
        rel_rmse_pct = (rmse / target_range * 100.0) if target_range > 0 else 0.0
        rel_mae_pct = (mae / (mean_target if mean_target > 0 else 1.0) * 100.0)

        interpretation = (
            f"Biome '{biome}': N={n_samples}, Target Range=[{target_min:.2f}, {target_max:.2f}] Mg/ha, "
            f"Pred Range=[{pred_min:.2f}, {pred_max:.2f}] Mg/ha, Mean Target={mean_target:.2f}, Mean Pred={mean_pred:.2f}. "
            f"RMSE={rmse:.2f} Mg/ha ({rel_rmse_pct:.1f}% of target range), MAE={mae:.2f} Mg/ha, R²={r2:.3f}."
        )

        biome_metrics[str(biome)] = {
            "sample_count": n_samples,
            "r2": round(r2, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "target_range": [round(target_min, 2), round(target_max, 2)],
            "prediction_range": [round(pred_min, 2), round(pred_max, 2)],
            "mean_target": round(mean_target, 2),
            "mean_prediction": round(mean_pred, 2),
            "relative_rmse_pct_of_range": round(rel_rmse_pct, 2),
            "interpretation": interpretation
        }

    return biome_metrics

