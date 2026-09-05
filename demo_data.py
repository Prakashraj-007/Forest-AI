"""
Demo Data Module for Offline / Fallback Operation.
Provides realistic pre-computed metadata and statistics for Bandipur National Park.
"""

from typing import Dict, Any
from config import (
    BANDIPUR_METADATA,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    ALL_STACK_BANDS,
    FEATURE_DESCRIPTIONS
)


def get_demo_pipeline_result(start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE) -> Dict[str, Any]:
    """
    Returns realistic diagnostic verification results for Bandipur National Park
    without requiring an active Earth Engine connection.
    """
    return {
        "status": "success",
        "mode": "DEMO_MODE",
        "ee_initialized": True,
        "study_area": BANDIPUR_METADATA,
        "date_range": {
            "start_date": start_date,
            "end_date": end_date
        },
        "datasets": {
            "sentinel2": {
                "collection": "COPERNICUS/S2_SR_HARMONIZED",
                "available": True,
                "image_count": 125,
                "cloud_filter_threshold_pct": 20,
                "masking_method": "SCL (Scene Classification Layer: shadows, cloud medium/high, cirrus)",
                "extracted_bands": ["B2", "B3", "B4", "B8", "B11", "B12"],
                "computed_indices": ["NDVI", "EVI", "SAVI", "NDWI"]
            },
            "sentinel1": {
                "collection": "COPERNICUS/S1_GRD",
                "available": True,
                "image_count": 31,
                "mode": "IW (Interferometric Wide)",
                "polarizations": ["VV", "VH"],
                "features": ["VV", "VH", "VV_VH_diff", "VV_VH_ratio_lin", "RVI"]
            },
            "canopy_height": {
                "primary_dataset": "users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1",
                "description": "ETH Zurich 10m GEDI-calibrated continuous canopy height",
                "available": True,
                "resolution_meters": 10,
                "companion_gedi_dataset": "LARSE/GEDI/GEDI02_A_002_MONTHLY",
                "companion_available": True
            }
        },
        "feature_stack": {
            "valid": True,
            "total_bands": len(ALL_STACK_BANDS),
            "band_names": ALL_STACK_BANDS,
            "descriptions": FEATURE_DESCRIPTIONS
        },
        "statistics": {
            "NDVI": {
                "min": -0.6744,
                "max": 0.8762,
                "mean": 0.4552,
                "unit": "unitless index [-1, 1]"
            },
            "EVI": {
                "min": -0.5210,
                "max": 0.9480,
                "mean": 0.3820,
                "unit": "unitless index [-1, 1]"
            },
            "SAVI": {
                "min": -0.4100,
                "max": 0.7250,
                "mean": 0.3200,
                "unit": "unitless index [-1, 1]"
            },
            "NDWI": {
                "min": -0.7500,
                "max": 0.6500,
                "mean": -0.1250,
                "unit": "unitless index [-1, 1]"
            },
            "VV": {
                "min": -23.2716,
                "max": 4.3719,
                "mean": -9.7448,
                "unit": "dB (decibels)"
            },
            "VH": {
                "min": -29.9950,
                "max": -3.0618,
                "mean": -16.3591,
                "unit": "dB (decibels)"
            },
            "VV_VH_diff": {
                "min": 1.2050,
                "max": 15.4820,
                "mean": 6.6143,
                "unit": "dB difference (VV_dB - VH_dB)"
            },
            "VV_VH_ratio_lin": {
                "min": 1.3190,
                "max": 35.3350,
                "mean": 4.5860,
                "unit": "linear ratio (10^((VV-VH)/10))"
            },
            "canopy_height": {
                "min": 0.0,
                "max": 37.0,
                "mean": 11.5072,
                "unit": "meters (m)"
            }
        },
        "summary": "Bandipur National Park Earth Engine data pipeline simulated successfully in DEMO_MODE."
    }


def get_demo_biomass_result() -> Dict[str, Any]:
    """
    Returns deterministic, realistic simulated biomass regression and raster prediction results
    for Bandipur National Park when operating in DEMO_MODE.
    Clearly labeled as DEMO / SIMULATED RESULTS.
    """
    return {
        "status": "DEMO / SIMULATED RESULTS",
        "mode": "DEMO_MODE",
        "model_status": "trained_demo",
        "study_area": BANDIPUR_METADATA["name"],
        "biomass_unit": "Mg/ha",
        "reference_dataset": {
            "name": "WHRC Pantropical Biomass (Simulated)",
            "dataset_id": "WHRC/biomass/tropical",
            "target_band": "Mg",
            "units": "Mg/ha",
            "spatial_resolution_m": 500,
            "citation": "Baccini et al. (2012), Woods Hole Research Center"
        },
        "sample_statistics": {
            "total_samples": 997,
            "training_samples": 797,
            "validation_samples": 200,
            "features_count": len(ALL_STACK_BANDS),
            "predictor_features": ALL_STACK_BANDS,
        },
        "model_config": {
            "model_type": "RandomForestRegressor",
            "library": "scikit-learn",
            "n_estimators": 200,
            "max_depth": 16,
            "random_state": 42,
            "test_split_ratio": 0.20
        },
        "validation_metrics": {
            "r2": 0.7873,
            "rmse": 19.1951,
            "mae": 13.4822,
            "mean_actual_val": 67.89,
            "mean_pred_val": 66.45,
            "unit": "Mg/ha"
        },
        "feature_importance": [
            {"feature": "canopy_height", "importance": 0.3842, "type": "structure_height", "description": "ETH Global Canopy Height (10m) calibrated with GEDI LiDAR"},
            {"feature": "B11", "importance": 0.1215, "type": "optical", "description": "SWIR-1 (1610 nm) - Canopy water absorption and dry biomass proxy"},
            {"feature": "VH", "importance": 0.0984, "type": "radar_sar", "description": "S1 C-band VH backscatter in dB - Volume scattering from branches and canopy"},
            {"feature": "NDVI", "importance": 0.0872, "type": "vegetation_index", "description": "Normalized Difference Veg Index (B8-B4)/(B8+B4) - Green canopy density"},
            {"feature": "EVI", "importance": 0.0714, "type": "vegetation_index", "description": "Enhanced Veg Index - Sensitive in high-biomass dense canopy without saturating"},
            {"feature": "VV_VH_diff", "importance": 0.0521, "type": "radar_sar", "description": "VV_dB - VH_dB backscatter difference (dB)"},
            {"feature": "B8", "importance": 0.0463, "type": "optical", "description": "NIR (842 nm) - Leaf cellular structure and high canopy scattering"},
            {"feature": "B12", "importance": 0.0385, "type": "optical", "description": "SWIR-2 (2190 nm) - Lignin, cellulose, and moisture sensitivity"},
            {"feature": "SAVI", "importance": 0.0312, "type": "vegetation_index", "description": "Soil-Adjusted Veg Index - Minimizes soil background brightness effects"},
            {"feature": "VV", "importance": 0.0245, "type": "radar_sar", "description": "S1 C-band VV backscatter in dB - Ground-trunk double bounce interaction"},
            {"feature": "RVI", "importance": 0.0189, "type": "radar_sar", "description": "Radar Vegetation Index: volume scattering fraction indicator"},
            {"feature": "VV_VH_ratio_lin", "importance": 0.0124, "type": "radar_sar", "description": "Linear polarization ratio 10^((VV-VH)/10)"},
            {"feature": "NDWI", "importance": 0.0078, "type": "vegetation_index", "description": "Normalized Difference Water Index - Canopy moisture content"},
            {"feature": "B4", "importance": 0.0032, "type": "optical", "description": "Red (665 nm) - Chlorophyll absorption band"},
            {"feature": "B3", "importance": 0.0016, "type": "optical", "description": "Green (560 nm) - Chlorophyll reflectance peak"},
            {"feature": "B2", "importance": 0.0008, "type": "optical", "description": "Blue (490 nm) - Atmospheric baseline and shadow differentiation"}
        ],
        "quality_interpretation": (
            "Strong regression fit (R² = 0.787). The multi-sensor optical, radar, and canopy height features "
            "explain over 78.7% of aboveground biomass variance across Bandipur National Park "
            "(RMSE = 19.20 Mg/ha, MAE = 13.48 Mg/ha). Structural canopy height and Sentinel-1 VH radar "
            "cross-polarization are the strongest predictors."
        ),
        "prediction_statistics": {
            "min": 0.0,
            "max": 204.5,
            "mean": 67.89,
            "median": 64.12,
            "std": 34.50,
            "valid_cells": 875,
            "unit": "Mg/ha",
            "study_area": BANDIPUR_METADATA["name"]
        },
        "raster_output": {
            "geotiff_path": "outputs/bandipur_biomass_prediction.tif (simulated)",
            "crs": "EPSG:4326",
            "bounds": BANDIPUR_METADATA["bbox"],
            "resolution_deg": [0.01, 0.01],
            "grid_shape": [25, 35]
        }
    }
