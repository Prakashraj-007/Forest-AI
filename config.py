"""
Configuration module for Earth Engine Forest Biomass & Tree Counting Pipeline.
Phase 1: Data Acquisition & Testing.
"""

import os
from typing import Dict, List, Any

# Google Cloud Project configured for Earth Engine
GCP_PROJECT: str = os.getenv("EE_PROJECT", "forest-ai-507705")

# Demo mode flag: fallback to realistic pre-computed data if EE is offline or unavailable
DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").lower() in ("true", "1", "yes")

# Fixed study area: Bandipur National Park, Karnataka, India
# Spans ~750 km² in southern Karnataka (Chamarajanagar district)
BANDIPUR_METADATA: Dict[str, Any] = {
    "name": "Bandipur National Park",
    "state": "Karnataka",
    "country": "India",
    "description": "Demonstration study area for AI-based biomass estimation and tree monitoring",
    "center": [76.625, 11.725],  # [Longitude, Latitude]
    "approx_area_sq_km": 750.0,
    "bbox": [76.45, 11.60, 76.80, 11.85],  # [min_lon, min_lat, max_lon, max_lat]
    "coordinates": [
        [76.45, 11.60],
        [76.80, 11.60],
        [76.80, 11.85],
        [76.45, 11.85],
        [76.45, 11.60],
    ]
}

# Default observation date range (dry/post-monsoon season for minimal cloud cover)
DEFAULT_START_DATE: str = "2023-01-01"
DEFAULT_END_DATE: str = "2023-04-30"

# Earth Engine Dataset Catalog Identifiers
DATASETS = {
    "sentinel2_sr": "COPERNICUS/S2_SR_HARMONIZED",
    "sentinel1_grd": "COPERNICUS/S1_GRD",
    "eth_canopy_height": "users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1",
    "gedi_l2a_monthly": "LARSE/GEDI/GEDI02_A_002_MONTHLY",
    "gedi_l4a_monthly": "LARSE/GEDI/GEDI04_A_002_MONTHLY",
    "whrc_biomass": "WHRC/biomass/tropical",
    "nasa_ornl_biomass": "NASA/ORNL/biomass_carbon_density/v1",
}

# Phase 2 Biomass Reference & Target Configuration
DEFAULT_BIOMASS_DATASET: str = os.getenv("BIOMASS_DATASET", "whrc")
BIOMASS_UNIT: str = "Mg/ha"  # Megagrams per hectare (metric tons/ha)

# Sampling Configuration
SAMPLE_COUNT: int = int(os.getenv("SAMPLE_COUNT", "1000"))
SAMPLING_SCALE: int = int(os.getenv("SAMPLING_SCALE", "100"))  # spatial scale in meters
RANDOM_SEED: int = 42

# Random Forest Hyperparameters & Validation Configuration
RF_N_ESTIMATORS: int = int(os.getenv("RF_N_ESTIMATORS", "200"))
RF_MAX_DEPTH: int = int(os.getenv("RF_MAX_DEPTH", "16"))
RF_MIN_SAMPLES_LEAF: int = 2
RF_RANDOM_STATE: int = 42
TEST_SPLIT_RATIO: float = 0.20

# Storage Directories & Relative File Paths
MODEL_DIR: str = "models"
MODEL_PATH: str = os.path.join("models", "biomass_rf_model.joblib")
MODEL_META_PATH: str = os.path.join("models", "biomass_model_meta.json")
MODEL_V2_PATH: str = os.path.join("models", "biomass_rf_model_v2_multibiome.joblib")
MODEL_V2_META_PATH: str = os.path.join("models", "biomass_model_v2_meta.json")

# NASA/ORNL Carbon Conversion Parameters
# NASA/ORNL biomass carbon density (agb band) unit is Mg C/ha.
# Conversion to Aboveground Biomass (AGB Mg/ha) uses a scale factor of 2.0 (assuming 0.50 carbon fraction, IPCC Tier 1 standard).
ORNL_AGB_CARBON_FRACTION: float = 0.50
ORNL_AGB_SCALE_FACTOR: float = 2.0  # AGB (Mg/ha) = Mg C/ha * (1 / 0.50)

# Multi-Biome Classification Labels
BIOME_LABELS: List[str] = ["tropical_forest", "arid_desert", "semi_arid_savanna"]

OUTPUT_DIR: str = "outputs"
RASTER_PATH: str = os.path.join("outputs", "bandipur_biomass_prediction.tif")
RASTER_META_PATH: str = os.path.join("outputs", "bandipur_biomass_raster_meta.json")

# Phase 3: High-Resolution Tree Detection & Counting Configuration (NEON Proxy)
HIGHRES_DATA_DIR: str = os.path.join("data", "demo_highres")
HIGHRES_DEMO_TIFF: str = os.path.join("data", "demo_highres", "neon_proxy_osbs.tif")
HIGHRES_META_PATH: str = os.path.join("data", "demo_highres", "neon_proxy_meta.json")

# DeepForest pretrained model (RetinaNet) is downloaded on first use to a cache dir
TREE_DETECTION_CONF_THRESH: float = float(os.getenv("TREE_CONF_THRESH", "0.25"))
TREE_DETECTION_IOU_THRESH: float = 0.40
TILE_SIZE: int = 400
TILE_OVERLAP: int = 0.05  # DeepForest uses percentage for overlap


# Feature band groups
S2_OPTICAL_BANDS: List[str] = ["B2", "B3", "B4", "B8", "B11", "B12"]
S2_INDICES: List[str] = ["NDVI", "EVI", "SAVI", "NDWI"]

S1_RADAR_BANDS: List[str] = [
    "VV",                 # Vertical-Vertical backscatter (dB)
    "VH",                 # Vertical-Horizontal backscatter (dB, volume scattering)
    "VV_VH_diff",         # dB difference: VV_dB - VH_dB
    "VV_VH_ratio_lin",    # Linear power polarization ratio: 10^((VV-VH)/10)
    "RVI",                # Dual-pol Radar Vegetation Index
]

HEIGHT_BANDS: List[str] = [
    "canopy_height",      # ETH Zurich GEDI-calibrated 10m wall-to-wall canopy height (m)
    "gedi_rh98",          # GEDI L2A relative height at 98% (canopy top height)
]

ALL_STACK_BANDS: List[str] = (
    S2_OPTICAL_BANDS +
    S2_INDICES +
    S1_RADAR_BANDS +
    ["canopy_height"]
)

# Feature dictionary describing relevance for biomass estimation
FEATURE_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "B2": {"type": "optical", "desc": "Blue (490 nm) - Atmospheric baseline and shadow differentiation"},
    "B3": {"type": "optical", "desc": "Green (560 nm) - Chlorophyll reflectance peak"},
    "B4": {"type": "optical", "desc": "Red (665 nm) - Chlorophyll absorption band"},
    "B8": {"type": "optical", "desc": "NIR (842 nm) - Leaf cellular structure and high canopy scattering"},
    "B11": {"type": "optical", "desc": "SWIR-1 (1610 nm) - Canopy water absorption and dry biomass proxy"},
    "B12": {"type": "optical", "desc": "SWIR-2 (2190 nm) - Lignin, cellulose, and moisture sensitivity"},
    "NDVI": {"type": "vegetation_index", "desc": "Normalized Difference Veg Index (B8-B4)/(B8+B4) - Green canopy density"},
    "EVI": {"type": "vegetation_index", "desc": "Enhanced Veg Index - Sensitive in high-biomass dense canopy without saturating"},
    "SAVI": {"type": "vegetation_index", "desc": "Soil-Adjusted Veg Index - Minimizes soil background brightness effects"},
    "NDWI": {"type": "vegetation_index", "desc": "Normalized Difference Water Index (B8-B11)/(B8+B11) - Canopy moisture content"},
    "VV": {"type": "radar_sar", "desc": "S1 C-band VV backscatter in dB - Ground-trunk double bounce interaction"},
    "VH": {"type": "radar_sar", "desc": "S1 C-band VH backscatter in dB - Volume scattering from branches, leaves, and canopy"},
    "VV_VH_diff": {"type": "radar_sar", "desc": "VV_dB - VH_dB backscatter difference (dB)"},
    "VV_VH_ratio_lin": {"type": "radar_sar", "desc": "Linear polarization ratio 10^((VV-VH)/10)"},
    "RVI": {"type": "radar_sar", "desc": "Radar Vegetation Index: volume scattering fraction indicator"},
    "canopy_height": {"type": "structure_height", "desc": "ETH Global Canopy Height (10m) calibrated with GEDI LiDAR (meters)"},
    "gedi_rh98": {"type": "structure_height", "desc": "GEDI L2A spaceborne LiDAR relative height at 98% (canopy top height, m)"}
}
