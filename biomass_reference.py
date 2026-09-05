"""
Biomass Reference & Target Data Module.
Handles Earth Engine-accessible biomass reference datasets for supervised training.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import ee

from config import DATASETS, DEFAULT_BIOMASS_DATASET, BIOMASS_UNIT, ORNL_AGB_CARBON_FRACTION, ORNL_AGB_SCALE_FACTOR

logger = logging.getLogger("biomass_reference")

# Metadata catalog for Earth Engine-accessible biomass datasets
BIOMASS_DATASET_SPECS = {
    "whrc": {
        "dataset_id": DATASETS["whrc_biomass"],
        "asset_type": "Image",
        "native_band": "Mg",
        "target_band_name": "biomass_target",
        "units": "Mg/ha",
        "spatial_resolution_m": 500,
        "citation": "Baccini et al. (2012), Woods Hole Research Center Pantropical National Level Carbon Stock",
        "description": "Pantropical aboveground live woody dry biomass density map based on multi-sensor satellite data and LiDAR calibration.",
        "temporal_range": "Circa 2000-2010 benchmark",
    },
    "gedi_l4a": {
        "dataset_id": DATASETS["gedi_l4a_monthly"],
        "asset_type": "ImageCollection",
        "native_band": "agbd",
        "target_band_name": "biomass_target",
        "units": "Mg/ha",
        "spatial_resolution_m": 1000,
        "citation": "Dubayah et al. (2022), NASA GEDI L4A Footprint Level Aboveground Biomass Density Gridded",
        "description": "Spaceborne full-waveform LiDAR derived Aboveground Biomass Density (AGBD) from the International Space Station.",
        "temporal_range": "2019-2023 Monthly",
    },
    "nasa_ornl": {
        "dataset_id": DATASETS["nasa_ornl_biomass"],
        "asset_type": "ImageCollection",
        "native_band": "agb",
        "target_band_name": "biomass_target",
        "units": "Mg/ha (AGB derived from Mg C/ha x 2.0)",
        "spatial_resolution_m": 300,
        "carbon_fraction_assumption": ORNL_AGB_CARBON_FRACTION,
        "conversion_scale_factor": ORNL_AGB_SCALE_FACTOR,
        "citation": "Spawn & Gibbs (2020), Global Aboveground and Belowground Biomass Carbon Density (NASA ORNL DAAC)",
        "description": f"Global aboveground biomass carbon density converted from Mg C/ha to AGB Mg/ha via x{ORNL_AGB_SCALE_FACTOR} factor based on assumed {ORNL_AGB_CARBON_FRACTION} carbon fraction (IPCC Tier 1 standard).",
        "temporal_range": "Circa 2010",
    }
}


def load_raw_biomass_image(aoi: ee.Geometry, dataset_key: str = "whrc") -> Tuple[ee.Image, Dict[str, Any]]:
    """
    Loads and clips the raw Earth Engine biomass reference image for the given AOI.
    For nasa_ornl, converts native biomass carbon density (Mg C/ha) to AGB (Mg/ha) using
    a 2.0x multiplier based on an assumed 0.50 carbon fraction (IPCC Tier 1 standard).
    """
    if dataset_key not in BIOMASS_DATASET_SPECS:
        raise ValueError(f"Unknown biomass reference dataset key '{dataset_key}'. Supported keys: {list(BIOMASS_DATASET_SPECS.keys())}")

    spec = BIOMASS_DATASET_SPECS[dataset_key]
    dataset_id = spec["dataset_id"]
    native_band = spec["native_band"]
    target_name = spec["target_band_name"]

    if spec["asset_type"] == "Image":
        raw_img = ee.Image(dataset_id).select(native_band).clip(aoi).rename(target_name)
    elif spec["asset_type"] == "ImageCollection":
        col = ee.ImageCollection(dataset_id).filterBounds(aoi).select(native_band)
        count = int(col.size().getInfo())
        if count == 0:
            raise ValueError(f"No scenes in ImageCollection '{dataset_id}' intersecting AOI.")
        raw_img = col.mean().clip(aoi).select(native_band)
        if dataset_key == "nasa_ornl":
            # Native unit is Mg C/ha. Multiply by 2.0 (carbon fraction = 0.50 assumption) -> AGB Mg/ha
            raw_img = raw_img.multiply(ORNL_AGB_SCALE_FACTOR)
        raw_img = raw_img.rename(target_name)
    else:
        raise ValueError(f"Unsupported asset type: {spec['asset_type']}")

    return raw_img, spec


def get_biomass_reference(
    aoi: ee.Geometry,
    dataset_key: str = DEFAULT_BIOMASS_DATASET,
    fallback_allowed: bool = True
) -> Tuple[ee.Image, Dict[str, Any]]:
    """
    Retrieves the biomass reference image and metadata for supervised training.
    Validates accessibility and band existence.
    If the requested dataset fails and fallback_allowed is True, tries alternative datasets.
    """
    attempted_keys = [dataset_key]
    if fallback_allowed:
        for k in BIOMASS_DATASET_SPECS:
            if k not in attempted_keys:
                attempted_keys.append(k)

    last_error = None
    for k in attempted_keys:
        try:
            logger.info(f"Attempting to load biomass reference dataset: {k} ({BIOMASS_DATASET_SPECS[k]['dataset_id']})")
            img, spec = load_raw_biomass_image(aoi, k)

            # Quick verification: check band name and test reduction
            band_names = img.bandNames().getInfo()
            if spec["target_band_name"] not in band_names:
                raise ValueError(f"Target band '{spec['target_band_name']}' not found in image: {band_names}")

            metadata = {
                "dataset_key": k,
                "dataset_id": spec["dataset_id"],
                "native_band": spec["native_band"],
                "target_band": spec["target_band_name"],
                "units": spec["units"],
                "spatial_resolution_m": spec["spatial_resolution_m"],
                "citation": spec["citation"],
                "description": spec["description"],
                "temporal_range": spec["temporal_range"],
                "status": "active"
            }
            logger.info(f"Successfully loaded biomass reference dataset: {k}")
            return img, metadata

        except Exception as e:
            logger.warning(f"Biomass dataset '{k}' failed: {e}")
            last_error = e

    raise RuntimeError(
        f"Failed to access any suitable Earth Engine biomass reference dataset over AOI. "
        f"Attempted: {attempted_keys}. Last error: {last_error}"
    )


def verify_biomass_reference_datasets(aoi: ee.Geometry) -> Dict[str, Any]:
    """
    Diagnostic tool to probe all candidate biomass reference datasets over the AOI.
    Returns accessibility and metadata report for each candidate.
    """
    results = {}
    for key, spec in BIOMASS_DATASET_SPECS.items():
        try:
            img, loaded_spec = load_raw_biomass_image(aoi, key)
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), sharedInputs=True),
                geometry=aoi,
                scale=500,
                maxPixels=1e7
            ).getInfo()

            results[key] = {
                "dataset_id": spec["dataset_id"],
                "accessible": True,
                "units": spec["units"],
                "resolution_m": spec["spatial_resolution_m"],
                "sample_stats": stats,
                "citation": spec["citation"]
            }
        except Exception as e:
            results[key] = {
                "dataset_id": spec["dataset_id"],
                "accessible": False,
                "error": str(e)
            }

    return results
