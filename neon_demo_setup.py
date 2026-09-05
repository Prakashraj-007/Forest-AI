"""
Phase 3 NEON Proxy Demo Setup
================================
Fetches a real NEON AOP high-resolution RGB GeoTIFF tile to be used as a
proxy demonstration dataset for individual tree detection.

Because real sub-meter open drone/aerial imagery for Bandipur National Park is
restricted by forestry regulations, we use a real 10cm/px NEON AOP tile
(OSBS_029.tif) from the Ordway-Swisher Biological Station (Florida, USA)
as a proxy to demonstrate the capability of the DeepForest pipeline.

The DeepForest package conveniently bundles this tile as an example,
so we extract it directly without requiring a NEON API token or download script.

License: NEON AOP data is distributed under a CC0 1.0 Universal (Public Domain) license.
Citation: National Ecological Observatory Network. Airborne Observation Platform.
"""

import os
import json
import shutil

try:
    import deepforest
    from deepforest import get_data
    HAS_DEEPFOREST = True
except ImportError:
    HAS_DEEPFOREST = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

import config

def setup_neon_proxy_tile(force: bool = False) -> dict:
    """
    Copies the OSBS_029.tif sample from deepforest installation to the project data dir.
    Returns metadata dict.
    """
    os.makedirs(config.HIGHRES_DATA_DIR, exist_ok=True)
    out_tiff = config.HIGHRES_DEMO_TIFF
    out_meta = config.HIGHRES_META_PATH

    if os.path.exists(out_tiff) and not force:
        if os.path.exists(out_meta):
            with open(out_meta) as f:
                return json.load(f)

    if not HAS_DEEPFOREST:
        raise ImportError("deepforest is not installed. Run: pip install deepforest")
    if not HAS_RASTERIO:
        raise ImportError("rasterio is not installed. Run: pip install rasterio")

    # Locate the built-in DeepForest sample
    src_tiff = get_data("OSBS_029.tif")
    if not src_tiff or not os.path.exists(src_tiff):
        raise FileNotFoundError("Could not find OSBS_029.tif in deepforest package data.")

    print(f"Copying NEON AOP proxy tile from: {src_tiff}")
    shutil.copy2(src_tiff, out_tiff)

    # Extract metadata using rasterio
    with rasterio.open(out_tiff) as src:
        width = src.width
        height = src.height
        transform = src.transform
        gsd_x = abs(transform.a)
        gsd_y = abs(transform.e)
        crs_str = str(src.crs) if src.crs else "Unknown"
        bounds = src.bounds

    # Compute area in hectares
    area_ha = (width * gsd_x * height * gsd_y) / 10_000.0

    meta = {
        "dataset_name": "NEON AOP High-Resolution Orthorectified Camera Imagery",
        "site_name": "Ordway-Swisher Biological Station (OSBS), Florida, USA",
        "proxy_note": "PROXY DATASET: Used to demonstrate Phase 3 pipeline since real Bandipur sub-meter open aerial data is restricted.",
        "license": "CC0 1.0 Universal (Public Domain)",
        "source": "National Ecological Observatory Network (NEON)",
        "access_requirements": "None (Bundled via deepforest sample data; normally requires NEON Data Portal account/API token for bulk download)",
        "resolution_m_per_px": gsd_x,
        "crs": crs_str,
        "tile_width_px": width,
        "tile_height_px": height,
        "tile_area_ha": round(area_ha, 4),
        "bounds": {
            "left": bounds.left,
            "bottom": bounds.bottom,
            "right": bounds.right,
            "top": bounds.top
        },
        "output_file": out_tiff
    }

    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    return meta

if __name__ == "__main__":
    meta = setup_neon_proxy_tile(force=True)
    print("NEON AOP Proxy Tile Setup Complete:")
    print(json.dumps(meta, indent=2))
