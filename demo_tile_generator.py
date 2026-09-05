"""
Phase 3 Demo Tile Generator
============================
Creates a synthetic, georeferenced high-resolution RGB GeoTIFF demo tile
that simulates a 0.5 m/pixel aerial canopy image for Bandipur National Park.

Intended use:
  - Provides a self-contained, license-free demonstration tile.
  - Run once; result is cached on disk.
  - The tile is georeferenced with a realistic UTM CRS so that pixel-to-geo
    coordinate conversion works correctly in the detection pipeline.

Source / License: Procedurally generated (no external data required).
Resolution (GSD): 0.50 m/pixel (simulated)
CRS: EPSG:32643 (WGS 84 / UTM zone 43N) -- covers Bandipur
Tile centre: ~76.625E, 11.725N  (Bandipur National Park, Karnataka, India)
Output: data/demo_highres/bandipur_canopy_highres.tif
"""

import os
import json
import numpy as np

try:
    import rasterio
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ---------------------------------------------------------------------------
OUTPUT_TIFF = os.path.join("data", "demo_highres", "bandipur_canopy_highres.tif")
OUTPUT_META = os.path.join("data", "demo_highres", "bandipur_highres_meta.json")

TILE_W = 2048
TILE_H = 2048
GSD_M = 0.5  # metres per pixel (simulated)

# Approximate Bandipur centre in UTM 43N (lon=76.625, lat=11.725)
ORIGIN_EASTING  = 432200.0   # top-left corner east
ORIGIN_NORTHING = 1296900.0  # top-left corner north

N_TREES = 420
CROWN_RADIUS_MIN = 4
CROWN_RADIUS_MAX = 16
RANDOM_SEED = 2024


# ---------------------------------------------------------------------------
def _paint_crown(band_r, band_g, band_b, cx, cy, r, rng):
    """Paint a Gaussian tree-crown blob on the RGB arrays."""
    y0, y1 = max(0, cy - r), min(TILE_H, cy + r + 1)
    x0, x1 = max(0, cx - r), min(TILE_W, cx + r + 1)
    for y in range(y0, y1):
        for x in range(x0, x1):
            dist = float(np.sqrt((x - cx) ** 2 + (y - cy) ** 2))
            if dist <= r:
                sigma = r * 0.5 if r > 0 else 1.0
                intensity = float(np.exp(-0.5 * (dist / sigma) ** 2))
                g_val = int(80 + 120 * intensity) + int(rng.integers(-10, 10))
                r_val = int(20 +  40 * intensity) + int(rng.integers(-5,  5))
                b_val = int(10 +  20 * intensity) + int(rng.integers(-5,  5))
                band_r[y, x] = max(0, min(255, r_val))
                band_g[y, x] = max(0, min(255, g_val))
                band_b[y, x] = max(0, min(255, b_val))


def generate_demo_tile(force: bool = False) -> dict:
    """
    Generate and save the synthetic demo GeoTIFF.
    Returns metadata dict; uses cached version if already present.
    """
    os.makedirs(os.path.dirname(OUTPUT_TIFF), exist_ok=True)

    if os.path.exists(OUTPUT_TIFF) and not force:
        if os.path.exists(OUTPUT_META):
            with open(OUTPUT_META) as f:
                meta = json.load(f)
            meta["cached"] = True
            return meta

    if not HAS_RASTERIO:
        raise ImportError(
            "rasterio is required to generate the demo GeoTIFF. "
            "Install with: pip install rasterio"
        )

    rng = np.random.default_rng(RANDOM_SEED)

    # Forest-floor background (dark green-brown)
    bg_r = rng.integers(20, 45, size=(TILE_H, TILE_W), dtype=np.uint8)
    bg_g = rng.integers(35, 65, size=(TILE_H, TILE_W), dtype=np.uint8)
    bg_b = rng.integers(10, 30, size=(TILE_H, TILE_W), dtype=np.uint8)

    band_r = bg_r.astype(np.float32)
    band_g = bg_g.astype(np.float32)
    band_b = bg_b.astype(np.float32)

    # Random tree crown positions and radii
    cx_arr = rng.integers(CROWN_RADIUS_MAX, TILE_W - CROWN_RADIUS_MAX, size=N_TREES)
    cy_arr = rng.integers(CROWN_RADIUS_MAX, TILE_H - CROWN_RADIUS_MAX, size=N_TREES)
    cr_arr = rng.integers(CROWN_RADIUS_MIN, CROWN_RADIUS_MAX + 1,     size=N_TREES)

    tree_centres_px = []
    for i in range(N_TREES):
        cx, cy, cr = int(cx_arr[i]), int(cy_arr[i]), int(cr_arr[i])
        _paint_crown(band_r, band_g, band_b, cx, cy, cr, rng)
        tree_centres_px.append({"x": cx, "y": cy, "radius_px": cr})

    img_r = np.clip(band_r, 0, 255).astype(np.uint8)
    img_g = np.clip(band_g, 0, 255).astype(np.uint8)
    img_b = np.clip(band_b, 0, 255).astype(np.uint8)

    transform = from_origin(
        west=ORIGIN_EASTING,
        north=ORIGIN_NORTHING,
        xsize=GSD_M,
        ysize=GSD_M,
    )

    with rasterio.open(
        OUTPUT_TIFF, "w",
        driver="GTiff",
        height=TILE_H,
        width=TILE_W,
        count=3,
        dtype="uint8",
        crs="EPSG:32643",  # WGS 84 / UTM zone 43N
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(img_r, 1)
        dst.write(img_g, 2)
        dst.write(img_b, 3)

    # Convert pixel centres to UTM coordinates
    tree_centres_geo = [
        {
            "easting_m":  round(ORIGIN_EASTING  + t["x"] * GSD_M, 2),
            "northing_m": round(ORIGIN_NORTHING  - t["y"] * GSD_M, 2),
            "radius_m":   round(t["radius_px"] * GSD_M, 2),
        }
        for t in tree_centres_px
    ]

    tile_area_ha = (TILE_W * GSD_M * TILE_H * GSD_M) / 10_000

    meta = {
        "source": (
            "Procedurally generated synthetic demo tile "
            "(open-source, no external data required)"
        ),
        "license": "CC0 / Public Domain",
        "resolution_m_per_px": GSD_M,
        "crs": "EPSG:32643 (WGS 84 / UTM zone 43N)",
        "origin_easting_m":  ORIGIN_EASTING,
        "origin_northing_m": ORIGIN_NORTHING,
        "tile_width_px":  TILE_W,
        "tile_height_px": TILE_H,
        "tile_area_ha":   round(tile_area_ha, 2),
        "simulated_tree_count": N_TREES,
        "study_area": "Bandipur National Park, Karnataka, India",
        "output_file": OUTPUT_TIFF,
        "tree_centres_geo": tree_centres_geo[:20],  # first 20 for sanity-check
        "cached": False,
    }

    with open(OUTPUT_META, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


if __name__ == "__main__":
    print("Generating synthetic demo GeoTIFF ...")
    result = generate_demo_tile(force=True)
    print(f"  Output      : {result['output_file']}")
    print(f"  Tile size   : {result['tile_width_px']} x {result['tile_height_px']} px")
    print(f"  GSD         : {result['resolution_m_per_px']} m/px")
    print(f"  Tile area   : {result['tile_area_ha']} ha")
    print(f"  Simulated trees: {result['simulated_tree_count']}")
    print("Done.")
