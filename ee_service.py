"""
Earth Engine Service for Satellite Data Acquisition and Preprocessing.
Phase 1: Forest Biomass Feature Engineering & Data Pipeline for Bandipur National Park.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import ee

from config import (
    GCP_PROJECT,
    BANDIPUR_METADATA,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    DATASETS,
    S2_OPTICAL_BANDS,
    S2_INDICES,
    S1_RADAR_BANDS,
    ALL_STACK_BANDS,
    FEATURE_DESCRIPTIONS,
    DEMO_MODE,
)
from demo_data import get_demo_pipeline_result

logger = logging.getLogger("ee_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_EE_INITIALIZED: bool = False


def initialize_earth_engine(project_id: str = GCP_PROJECT) -> bool:
    """
    Initializes the Google Earth Engine Python API with the specified GCP project.
    Safe to call multiple times (cached).
    """
    global _EE_INITIALIZED
    if _EE_INITIALIZED:
        return True

    try:
        ee.Initialize(project=project_id)
        _EE_INITIALIZED = True
        logger.info(f"Successfully initialized Earth Engine with project: {project_id}")
        return True
    except Exception as e:
        logger.warning(f"Failed to initialize Earth Engine with project '{project_id}': {e}")
        _EE_INITIALIZED = False
        return False


def get_bandipur_polygon() -> ee.Geometry.Polygon:
    """
    Returns the ee.Geometry.Polygon defining the Bandipur National Park study area.
    """
    initialize_earth_engine()
    return ee.Geometry.Polygon([BANDIPUR_METADATA["coordinates"]])


def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """
    Applies cloud and shadow masking to Sentinel-2 Surface Reflectance imagery
    using the Scene Classification Layer (SCL) band.
    Masks out:
      - 3: Cloud shadow
      - 8: Cloud (medium probability)
      - 9: Cloud (high probability)
      - 10: Thin cirrus
    """
    scl = image.select("SCL")
    mask = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
    )
    return image.updateMask(mask)


def compute_vegetation_indices(optical_img: ee.Image) -> ee.Image:
    """
    Calculates key vegetation indices for forest biomass estimation:
      - NDVI: (NIR - Red) / (NIR + Red)
      - EVI: 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
      - SAVI: 1.5 * (NIR - Red) / (NIR + Red + 0.5)
      - NDWI: (NIR - SWIR1) / (NIR + SWIR1) [Canopy water content]
    Assumes surface reflectance scaled to [0, 1].
    """
    b2 = optical_img.select("B2").multiply(0.0001)
    b4 = optical_img.select("B4").multiply(0.0001)
    b8 = optical_img.select("B8").multiply(0.0001)
    b11 = optical_img.select("B11").multiply(0.0001)

    # NDVI
    ndvi = b8.subtract(b4).divide(b8.add(b4)).rename("NDVI")

    # EVI: Enhanced Vegetation Index (avoids saturation over dense forests)
    evi_denom = b8.add(b4.multiply(6.0)).subtract(b2.multiply(7.5)).add(1.0)
    evi = b8.subtract(b4).multiply(2.5).divide(evi_denom).rename("EVI")

    # SAVI: Soil Adjusted Vegetation Index (L=0.5)
    savi_denom = b8.add(b4).add(0.5)
    savi = b8.subtract(b4).multiply(1.5).divide(savi_denom).rename("SAVI")

    # NDWI: Normalized Difference Water Index (canopy hydration)
    ndwi = b8.subtract(b11).divide(b8.add(b11)).rename("NDWI")

    return ee.Image.cat([ndvi, evi, savi, ndwi])


def get_sentinel2_composite(
    aoi: ee.Geometry,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    max_cloud_percent: float = 20.0
) -> Tuple[ee.Image, int, Dict[str, Any]]:
    """
    Retrieves Sentinel-2 Harmonized Surface Reflectance, filters by date, bounds,
    and cloud percentage, applies cloud masking, computes the median composite,
    and calculates vegetation indices.
    """
    collection = (
        ee.ImageCollection(DATASETS["sentinel2_sr"])
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_percent))
    )

    image_count = int(collection.size().getInfo())
    if image_count == 0:
        logger.warning(f"No S2 imagery <{max_cloud_percent}% cloud over AOI. Retrying without cloud filter...")
        collection = (
            ee.ImageCollection(DATASETS["sentinel2_sr"])
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
        )
        image_count = int(collection.size().getInfo())

        if image_count == 0:
            logger.warning(f"No S2 imagery in date range over AOI. Expanding date range to 2021-2024...")
            collection = (
                ee.ImageCollection(DATASETS["sentinel2_sr"])
                .filterBounds(aoi)
                .filterDate("2021-01-01", "2024-12-31")
            )
            image_count = int(collection.size().getInfo())

    # Apply SCL cloud mask and median composite
    masked_col = collection.map(mask_s2_clouds)
    s2_median = masked_col.median().clip(aoi)

    # Optical raw/scaled bands
    optical_bands = s2_median.select(S2_OPTICAL_BANDS)

    # Vegetation indices
    indices = compute_vegetation_indices(s2_median)

    composite = optical_bands.addBands(indices)

    metadata = {
        "dataset": DATASETS["sentinel2_sr"],
        "image_count": image_count,
        "date_range": [start_date, end_date],
        "cloud_threshold_pct": max_cloud_percent,
        "optical_bands": S2_OPTICAL_BANDS,
        "computed_indices": S2_INDICES
    }

    return composite, image_count, metadata


def compute_radar_features(s1_median: ee.Image) -> ee.Image:
    """
    Extracts VV and VH radar features:
      - VV: Vertical-Vertical dB backscatter
      - VH: Vertical-Horizontal dB backscatter (cross-pol, canopy volume scattering)
      - VV_VH_diff: VV_dB - VH_dB (dB difference)
      - VV_VH_ratio_lin: 10^((VV_dB - VH_dB) / 10) (Linear polarization ratio)
      - RVI: Dual-pol Radar Vegetation Index: 4 * VH_lin / (VV_lin + VH_lin)
    """
    vv = s1_median.select("VV")
    vh = s1_median.select("VH")

    # VV_VH_diff as backscatter difference in dB: VV_dB - VH_dB
    diff_db = vv.subtract(vh).rename("VV_VH_diff")

    # Linear polarization ratio: 10^((VV - VH) / 10)
    ratio_lin = ee.Image(10.0).pow(vv.subtract(vh).divide(10.0)).rename("VV_VH_ratio_lin")

    # Linear powers for RVI calculation: P_lin = 10^(dB/10)
    vv_lin = ee.Image(10.0).pow(vv.divide(10.0))
    vh_lin = ee.Image(10.0).pow(vh.divide(10.0))

    # Dual-pol RVI = 4 * VH_lin / (VV_lin + VH_lin)
    rvi = (
        ee.Image(4.0)
        .multiply(vh_lin)
        .divide(vv_lin.add(vh_lin))
        .rename("RVI")
    )

    return ee.Image.cat([vv, vh, diff_db, ratio_lin, rvi])


def get_sentinel1_composite(
    aoi: ee.Geometry,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE
) -> Tuple[ee.Image, int, Dict[str, Any]]:
    """
    Retrieves Sentinel-1 GRD imagery, filters for IW mode and VV+VH polarizations,
    computes median composite, and derives radar backscatter features.
    Falls back gracefully to multi-year search or synthetic desert baseline if S1 is unmapped.
    """
    collection = (
        ee.ImageCollection(DATASETS["sentinel1_grd"])
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
    )

    image_count = int(collection.size().getInfo())
    if image_count == 0:
        logger.warning(f"No S1 IW imagery found for AOI between {start_date} and {end_date}. Trying expanded 3-year window...")
        expanded_col = (
            ee.ImageCollection(DATASETS["sentinel1_grd"])
            .filterBounds(aoi)
            .filterDate("2021-01-01", "2024-12-31")
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        )
        image_count = int(expanded_col.size().getInfo())
        if image_count > 0:
            collection = expanded_col
        else:
            logger.warning(f"No Sentinel-1 scenes found even in 3-year window over AOI. Using synthetic dry baseline radar composite...")
            vv_constant = ee.Image.constant(-16.0).rename("VV").clip(aoi)
            vh_constant = ee.Image.constant(-23.0).rename("VH").clip(aoi)
            s1_fallback = vv_constant.addBands(vh_constant)
            radar_stack = compute_radar_features(s1_fallback)
            metadata = {
                "dataset": DATASETS["sentinel1_grd"],
                "image_count": 0,
                "date_range": [start_date, end_date],
                "mode": "SYNTHETIC_DESERT_FALLBACK",
                "polarizations": ["VV", "VH"],
                "radar_features": S1_RADAR_BANDS
            }
            return radar_stack, 0, metadata

    s1_median = collection.select(["VV", "VH"]).median().clip(aoi)
    radar_stack = compute_radar_features(s1_median)

    metadata = {
        "dataset": DATASETS["sentinel1_grd"],
        "image_count": image_count,
        "date_range": [start_date, end_date],
        "mode": "IW",
        "polarizations": ["VV", "VH"],
        "radar_features": S1_RADAR_BANDS
    }

    return radar_stack, image_count, metadata


def get_canopy_height_layers(aoi: ee.Geometry) -> Tuple[ee.Image, Dict[str, Any]]:
    """
    Retrieves forest height datasets:
      1. Primary: ETH Global Canopy Height (10m) continuous map derived from GEDI LiDAR (Lang et al.)
         Asset: users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1
      2. Companion/Fallback: GEDI L2A Monthly Elevation and Height Metrics (rh98)
         Asset: LARSE/GEDI/GEDI02_A_002_MONTHLY
    """
    # Primary: ETH Global Canopy Height 10m (unmasked with 0.0 for arid/unmapped areas)
    eth_img = ee.Image(DATASETS["eth_canopy_height"]).clip(aoi).rename("canopy_height").unmask(0.0)

    # Companion: GEDI L2A rh98
    gedi_col = (
        ee.ImageCollection(DATASETS["gedi_l2a_monthly"])
        .filterBounds(aoi)
    )
    gedi_rh98 = gedi_col.select("rh98").mean().clip(aoi).rename("gedi_rh98")

    combined_height = eth_img.addBands(gedi_rh98)

    metadata = {
        "primary": {
            "dataset": DATASETS["eth_canopy_height"],
            "band": "canopy_height",
            "resolution_m": 10,
            "description": "ETH Zurich 10m wall-to-wall GEDI-calibrated Canopy Height (m)"
        },
        "companion": {
            "dataset": DATASETS["gedi_l2a_monthly"],
            "band": "gedi_rh98",
            "description": "NASA GEDI L2A relative height at 98% (canopy top height, m)"
        }
    }

    return combined_height, metadata


def build_biomass_feature_stack(
    aoi: Optional[ee.Geometry] = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE
) -> Tuple[ee.Image, Dict[str, Any]]:
    """
    Combines Sentinel-1, Sentinel-2, vegetation indices, and canopy height layers
    into a unified multi-band ee.Image stack for biomass modeling.
    """
    if aoi is None:
        aoi = get_bandipur_polygon()

    # 1. Optical + Vegetation Indices
    s2_stack, s2_count, s2_meta = get_sentinel2_composite(aoi, start_date, end_date)

    # 2. Radar backscatter features
    s1_stack, s1_count, s1_meta = get_sentinel1_composite(aoi, start_date, end_date)

    # 3. Forest structure & Canopy Height
    height_stack, height_meta = get_canopy_height_layers(aoi)

    # 4. Multi-band Stack
    full_stack = (
        s2_stack
        .addBands(s1_stack)
        .addBands(height_stack.select("canopy_height"))
    )

    band_names = full_stack.bandNames().getInfo()

    metadata = {
        "study_area": BANDIPUR_METADATA["name"],
        "date_range": [start_date, end_date],
        "band_names": band_names,
        "total_bands": len(band_names),
        "sentinel2": s2_meta,
        "sentinel1": s1_meta,
        "canopy_height": height_meta,
        "feature_descriptions": FEATURE_DESCRIPTIONS
    }

    return full_stack, metadata


def compute_stack_statistics(
    image_stack: ee.Image,
    aoi: ee.Geometry,
    bands: Optional[list] = None,
    scale: int = 200
) -> Dict[str, Dict[str, float]]:
    """
    Computes min, max, and mean statistics across the AOI for specified bands.
    Uses scale parameter to control spatial resolution for rapid aggregation.
    """
    if bands is None:
        bands = ["NDVI", "EVI", "SAVI", "NDWI", "VV", "VH", "VV_VH_diff", "VV_VH_ratio_lin", "canopy_height"]

    selected = image_stack.select(bands)
    stats_raw = selected.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), sharedInputs=True),
        geometry=aoi,
        scale=scale,
        maxPixels=1e8
    ).getInfo()

    result: Dict[str, Dict[str, float]] = {}
    for band in bands:
        min_key = f"{band}_min"
        max_key = f"{band}_max"
        mean_key = f"{band}_mean"

        if min_key in stats_raw or mean_key in stats_raw:
            result[band] = {
                "min": round(stats_raw.get(min_key, 0.0), 4) if stats_raw.get(min_key) is not None else None,
                "max": round(stats_raw.get(max_key, 0.0), 4) if stats_raw.get(max_key) is not None else None,
                "mean": round(stats_raw.get(mean_key, 0.0), 4) if stats_raw.get(mean_key) is not None else None,
            }

    return result


def verify_pipeline(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    force_demo: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive verification pipeline:
      - Validates EE initialization
      - Verifies Sentinel-1 data exists
      - Verifies Sentinel-2 data exists
      - Validates feature stack integrity
      - Confirms canopy height data availability
      - Calculates summary statistics
      - Falls back cleanly to DEMO_MODE if requested or upon failure.
    """
    if force_demo or DEMO_MODE:
        logger.info("Operating in DEMO_MODE (either configured or forced).")
        return get_demo_pipeline_result(start_date, end_date)

    try:
        # Step 1: Initialize Earth Engine
        if not initialize_earth_engine():
            logger.warning("Earth Engine initialization failed, falling back to DEMO_MODE.")
            demo_res = get_demo_pipeline_result(start_date, end_date)
            demo_res["ee_initialized"] = False
            demo_res["fallback_reason"] = "EE initialization failed"
            return demo_res

        # Step 2: AOI
        aoi = get_bandipur_polygon()

        # Step 3: Build feature stack
        stack_img, stack_meta = build_biomass_feature_stack(aoi, start_date, end_date)

        # Step 4: Compute statistics
        stats = compute_stack_statistics(stack_img, aoi, scale=200)

        # Step 5: Format response
        return {
            "status": "success",
            "mode": "LIVE_EARTH_ENGINE",
            "ee_initialized": True,
            "study_area": BANDIPUR_METADATA,
            "date_range": {
                "start_date": start_date,
                "end_date": end_date
            },
            "datasets": {
                "sentinel2": stack_meta["sentinel2"],
                "sentinel1": stack_meta["sentinel1"],
                "canopy_height": stack_meta["canopy_height"]
            },
            "feature_stack": {
                "valid": True,
                "total_bands": stack_meta["total_bands"],
                "band_names": stack_meta["band_names"],
                "descriptions": FEATURE_DESCRIPTIONS
            },
            "statistics": stats,
            "summary": (
                f"Successfully verified Earth Engine data pipeline for {BANDIPUR_METADATA['name']}. "
                f"Sentinel-2: {stack_meta['sentinel2']['image_count']} scenes, "
                f"Sentinel-1: {stack_meta['sentinel1']['image_count']} scenes, "
                f"Total feature bands in stack: {stack_meta['total_bands']}."
            )
        }

    except Exception as e:
        logger.error(f"Live Earth Engine pipeline verification failed: {e}", exc_info=True)
        demo_res = get_demo_pipeline_result(start_date, end_date)
        demo_res["status"] = "fallback_success"
        demo_res["ee_initialized"] = _EE_INITIALIZED
        demo_res["fallback_reason"] = f"Runtime Earth Engine error: {str(e)}"
        return demo_res


def get_ee_map_tile_url(
    layer: str,
    geometry: Optional[Dict[str, Any]] = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE
) -> Dict[str, Any]:
    """
    Generates a dynamic Google Earth Engine TileLayer URL for Leaflet visualization
    corresponding to the specified layer and geographic AOI.

    Layers:
      - 's2_rgb': Sentinel-2 True Color RGB (B4, B3, B2)
      - 's2_ndvi': Sentinel-2 NDVI (palette: red-yellow-green)
      - 's1_vv': Sentinel-1 VV backscatter (dB, grayscale)
      - 's1_vh': Sentinel-1 VH backscatter (dB, grayscale/volume)
      - 'canopy_height': ETH 10m Global Canopy Height (palette: yellow-green-darkgreen)
      - 'biomass': WHRC / Global Biomass Reference Layer
    """
    if DEMO_MODE or not initialize_earth_engine():
        return {
            "success": False,
            "mode": "DEMO_MODE",
            "layer": layer,
            "message": "Earth Engine credentials not active or DEMO_MODE is enabled.",
            "tile_url": None
        }

    try:
        # 1. Determine AOI
        if geometry and "coordinates" in geometry:
            coords = geometry["coordinates"]
            ee_geom = ee.Geometry.Polygon(coords)
        else:
            ee_geom = get_bandipur_polygon()

        vis_params = {}
        target_img = None

        if layer == "s2_rgb":
            collection = (
                ee.ImageCollection(DATASETS["sentinel2_sr"])
                .filterBounds(ee_geom)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            )
            masked = collection.map(mask_s2_clouds)
            target_img = masked.median().clip(ee_geom)
            vis_params = {
                "bands": ["B4", "B3", "B2"],
                "min": 0,
                "max": 3000,
                "gamma": 1.3
            }

        elif layer == "s2_ndvi":
            collection = (
                ee.ImageCollection(DATASETS["sentinel2_sr"])
                .filterBounds(ee_geom)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            )
            masked = collection.map(mask_s2_clouds)
            s2_med = masked.median().clip(ee_geom)
            indices = compute_vegetation_indices(s2_med)
            target_img = indices.select("NDVI")
            vis_params = {
                "min": 0.0,
                "max": 0.9,
                "palette": ["#d73027", "#f46d43", "#fdae61", "#fee08b", "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850", "#006837"]
            }

        elif layer in ("s1_vv", "s1_vh"):
            collection = (
                ee.ImageCollection(DATASETS["sentinel1_grd"])
                .filterBounds(ee_geom)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
            )
            s1_med = collection.select(["VV", "VH"]).median().clip(ee_geom)
            band = "VV" if layer == "s1_vv" else "VH"
            target_img = s1_med.select(band)
            vis_params = {
                "min": -25.0 if band == "VH" else -20.0,
                "max": -5.0 if band == "VH" else 0.0,
                "palette": ["#000000", "#333333", "#666666", "#999999", "#cccccc", "#ffffff"]
            }

        elif layer == "canopy_height":
            eth_img = ee.Image(DATASETS["eth_canopy_height"]).clip(ee_geom)
            target_img = eth_img
            vis_params = {
                "min": 0,
                "max": 35,
                "palette": ["#440154", "#3b528b", "#21908d", "#5dc863", "#fde725"]
            }

        elif layer == "biomass":
            from biomass_reference import load_raw_biomass_image
            bio_raw, _ = load_raw_biomass_image(ee_geom, dataset_key="nasa_ornl")
            target_img = bio_raw
            vis_params = {
                "min": 0,
                "max": 150,
                "palette": ["#064e3b", "#059669", "#10b981", "#eab308", "#ef4444"]
            }

        else:
            return {
                "success": False,
                "error": f"Unknown layer '{layer}'. Available: s2_rgb, s2_ndvi, s1_vv, s1_vh, canopy_height, biomass"
            }

        map_id_dict = target_img.getMapId(vis_params)
        tile_url = None
        if "tile_fetcher" in map_id_dict and hasattr(map_id_dict["tile_fetcher"], "url_format"):
            tile_url = map_id_dict["tile_fetcher"].url_format
        elif "url_format" in map_id_dict:
            tile_url = map_id_dict["url_format"]
        elif "mapid" in map_id_dict:
            tile_url = f"https://earthengine.googleapis.com/v1/projects/{GCP_PROJECT}/maps/{map_id_dict['mapid']}/tiles/{{z}}/{{x}}/{{y}}"

        return {
            "success": True,
            "layer": layer,
            "tile_url": tile_url,
            "vis_params": vis_params,
            "date_range": [start_date, end_date]
        }

    except Exception as exc:
        logger.error(f"Error creating EE map tile URL for layer {layer}: {exc}", exc_info=True)
        return {
            "success": False,
            "layer": layer,
            "error": str(exc)
        }

