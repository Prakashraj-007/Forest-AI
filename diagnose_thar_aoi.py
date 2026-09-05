"""
Thar Desert AOI Diagnostic Script
===================================
Investigates why the Thar Desert AOI returns:
  - Mean canopy height: 14.6 m  (scientifically wrong for a desert)
  - Mean biomass: ~28-29 Mg/ha  (should be near-zero for an arid dune ecosystem)

Investigation targets:
  1. ETH canopy height raster — is it masked to the AOI? Values? 
  2. Sentinel-2 NDVI statistics — appropriate for desert?
  3. Full feature range table sent to the Random Forest
  4. WHRC biomass reference domain — does it cover arid Thar?
  5. NaN-fill strategy in analysis_service.py (fillna with median — could this propagate forest values?)
  6. Whether the model's training distribution (Bandipur, ~68 Mg/ha mean) biases predictions
     when queried outside its training domain
"""

import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np

# ── Setup EE ──────────────────────────────────────────────────────────────────
from ee_service import initialize_earth_engine, mask_s2_clouds, compute_vegetation_indices
from config import DATASETS, ALL_STACK_BANDS, DEFAULT_START_DATE, DEFAULT_END_DATE

print("="*72)
print("  THAR DESERT AOI — SCIENTIFIC DIAGNOSTIC INVESTIGATION")
print("="*72)

if not initialize_earth_engine():
    print("ERROR: Cannot initialize Earth Engine. Exiting.")
    sys.exit(1)

import ee

# ── AOI definition (approximation of the polygon visible in the screenshot) ───
# Jaisalmer / Desert National Park area in Rajasthan, India
# From the screenshot: a quadrilateral polygon near Jaisalmer (~26.9–27.4 N, 70.6–71.2 E)
THAR_COORDS = [
    [70.72, 27.35],
    [71.18, 27.30],
    [71.10, 26.88],
    [70.68, 26.92],
    [70.72, 27.35],
]

aoi = ee.Geometry.Polygon([THAR_COORDS])
START_DATE = "2023-01-01"
END_DATE   = "2023-04-30"

# Compute area
area_ha = float(aoi.area(maxError=100).getInfo()) / 10_000
print(f"\n[AOI] Area          : {area_ha:,.0f} ha  ({area_ha/100:.1f} km²)")
print(f"[AOI] Centroid      : ~27.1°N, 70.9°E  (Jaisalmer / Desert National Park)")
print(f"[AOI] Dates         : {START_DATE} to {END_DATE}")

# ── 1. ETH Global Canopy Height ─────────────────────────────────────────────
print("\n" + "─"*72)
print("  1. ETH GLOBAL CANOPY HEIGHT (users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1)")
print("─"*72)

eth_img = ee.Image(DATASETS["eth_canopy_height"]).clip(aoi)

# Check mask and pixel count
eth_stats = eth_img.reduceRegion(
    reducer=ee.Reducer.mean()
        .combine(ee.Reducer.min(), sharedInputs=True)
        .combine(ee.Reducer.max(), sharedInputs=True)
        .combine(ee.Reducer.median(), sharedInputs=True)
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.count(), sharedInputs=True),
    geometry=aoi,
    scale=10,       # native 10m resolution
    maxPixels=1e9
).getInfo()

print(f"  Band: 'b1' (raw band name from ETH asset)")
print(f"  Stats at native 10m scale:")
for k, v in sorted(eth_stats.items()):
    print(f"    {k:<25s} : {v}")

# Get valid pixel count vs total possible
total_pixels_approx = int(area_ha * 10_000 / 100)   # at 10m: ~10000 px/ha
valid_px = eth_stats.get("b1_count", "unknown")
print(f"\n  Valid pixels (count)    : {valid_px}")
print(f"  Approx expected at 10m  : {total_pixels_approx:,}")

# Check un-masked mean at 10m
eth_mean_10m = eth_stats.get("b1_mean")
print(f"\n  CONCLUSION — ETH mean at native 10m : {eth_mean_10m} m")
if eth_mean_10m is not None and eth_mean_10m > 2.0:
    print("  ⚠  ANOMALY: ETH mean > 2m in a desert. Possible causes:")
    print("     (a) Dataset has non-zero background fill values (e.g. 0 or last valid pixel),")
    print("     (b) Scale mismatch — sampling at 100m resamples to forest regions,")
    print("     (c) The ETH raster is not properly masked to bare desert (it maps height=0 as valid).")
else:
    print("  ✓  ETH correctly shows low canopy height for arid region.")

# ── 2. ETH sampled at 100m (what analysis_service actually uses) ─────────────
print("\n" + "─"*72)
print("  2. ETH CANOPY HEIGHT SAMPLED AT 100m  (what sampleRegions uses)")
print("─"*72)

eth_100m = eth_img.reduceRegion(
    reducer=ee.Reducer.mean()
        .combine(ee.Reducer.min(), sharedInputs=True)
        .combine(ee.Reducer.max(), sharedInputs=True)
        .combine(ee.Reducer.median(), sharedInputs=True),
    geometry=aoi,
    scale=100,
    maxPixels=1e8
).getInfo()

for k, v in sorted(eth_100m.items()):
    print(f"  {k:<25s} : {v}")

# ── 3. Sentinel-2 NDVI and optical bands ─────────────────────────────────────
print("\n" + "─"*72)
print("  3. SENTINEL-2 OPTICAL & NDVI (Cloud ≤20%, Median Composite)")
print("─"*72)

col = (
    ee.ImageCollection(DATASETS["sentinel2_sr"])
    .filterBounds(aoi)
    .filterDate(START_DATE, END_DATE)
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
)
s2_count = int(col.size().getInfo())
print(f"  S2 scenes available  : {s2_count}")

if s2_count == 0:
    print("  WARNING: No Sentinel-2 scenes found! Trying cloud=60%...")
    col = (
        ee.ImageCollection(DATASETS["sentinel2_sr"])
        .filterBounds(aoi)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
    )
    s2_count = int(col.size().getInfo())
    print(f"  S2 scenes at 60% cloud: {s2_count}")

s2_med = col.map(mask_s2_clouds).median().clip(aoi)
indices = compute_vegetation_indices(s2_med)
s2_full = s2_med.addBands(indices)

s2_stats = s2_full.reduceRegion(
    reducer=ee.Reducer.mean()
        .combine(ee.Reducer.min(), sharedInputs=True)
        .combine(ee.Reducer.max(), sharedInputs=True),
    geometry=aoi,
    scale=100,
    maxPixels=1e8
).getInfo()

print(f"\n  Band statistics at 100m scale:")
for band in ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "EVI", "SAVI", "NDWI"]:
    mean_k = f"{band}_mean"
    min_k  = f"{band}_min"
    max_k  = f"{band}_max"
    mean_v = s2_stats.get(mean_k, "N/A")
    min_v  = s2_stats.get(min_k, "N/A")
    max_v  = s2_stats.get(max_k, "N/A")
    print(f"  {band:<6s} : mean={mean_v!s:<12} min={min_v!s:<12} max={max_v!s}")

ndvi_mean = s2_stats.get("NDVI_mean")
if ndvi_mean is not None:
    print(f"\n  NDVI Mean: {ndvi_mean:.4f}")
    if ndvi_mean < 0.15:
        print("  ✓  NDVI < 0.15 — correctly represents bare arid/desert land")
    elif ndvi_mean < 0.25:
        print("  ⚠  NDVI 0.15–0.25 — marginal/sparse vegetation (dry scrub)")
    else:
        print("  ⚠⚠ NDVI > 0.25 — unexpectedly high for desert; check if cloud contamination")

# ── 4. Sentinel-1 SAR ────────────────────────────────────────────────────────
print("\n" + "─"*72)
print("  4. SENTINEL-1 GRD SAR BACKSCATTER")
print("─"*72)

s1_col = (
    ee.ImageCollection(DATASETS["sentinel1_grd"])
    .filterBounds(aoi)
    .filterDate(START_DATE, END_DATE)
    .filter(ee.Filter.eq("instrumentMode", "IW"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
)
s1_count = int(s1_col.size().getInfo())
print(f"  S1 scenes available  : {s1_count}")

s1_med = s1_col.select(["VV", "VH"]).median().clip(aoi)
s1_stats = s1_med.reduceRegion(
    reducer=ee.Reducer.mean()
        .combine(ee.Reducer.min(), sharedInputs=True)
        .combine(ee.Reducer.max(), sharedInputs=True),
    geometry=aoi,
    scale=100,
    maxPixels=1e8
).getInfo()

for band in ["VV", "VH"]:
    print(f"  {band}: mean={s1_stats.get(band+'_mean','N/A'):.3f} dB | "
          f"min={s1_stats.get(band+'_min','N/A'):.3f} | "
          f"max={s1_stats.get(band+'_max','N/A'):.3f}")

# ── 5. Full Feature Vector — what the RF actually receives ───────────────────
print("\n" + "─"*72)
print("  5. FULL FEATURE VECTOR AT SAMPLE POINTS (what RF receives)")
print("─"*72)

from ee_service import build_biomass_feature_stack, compute_radar_features
feature_stack, _ = build_biomass_feature_stack(aoi=aoi, start_date=START_DATE, end_date=END_DATE)

# Create a regular grid of sample points (same as analysis_service.py)
bbox_info = aoi.bounds().getInfo()["coordinates"][0]
xs = [p[0] for p in bbox_info]
ys = [p[1] for p in bbox_info]
min_lon, max_lon = min(xs), max(xs)
min_lat, max_lat = min(ys), max(ys)
grid_step = max(0.003, min((max_lon - min_lon) / 20, (max_lat - min_lat) / 20))
lons = list(np.arange(min_lon + grid_step / 2, max_lon, grid_step))
lats = list(np.arange(min_lat + grid_step / 2, max_lat, grid_step))

pts = [ee.Feature(ee.Geometry.Point([float(lo), float(la)])) for lo in lons for la in lats]
pts_fc = ee.FeatureCollection(pts)
res = feature_stack.sampleRegions(collection=pts_fc, scale=100, geometries=False).getInfo()
records = [f["properties"] for f in res.get("features", [])]

print(f"  Grid sample points   : {len(lons)} × {len(lats)} = {len(pts)} pts")
print(f"  Records returned     : {len(records)}")

import pandas as pd
df = pd.DataFrame(records)
print(f"\n  Feature value ranges at sample grid (BEFORE NaN fill):")
print(f"  {'Band':<20} {'count':>8} {'NaN':>6} {'min':>10} {'mean':>10} {'median':>10} {'max':>10}")
print(f"  {'-'*76}")
nan_summary = {}
for b in ALL_STACK_BANDS:
    if b in df.columns:
        col_data = df[b]
        nan_count = col_data.isna().sum()
        nan_summary[b] = nan_count
        valid = col_data.dropna()
        if len(valid) > 0:
            print(f"  {b:<20} {len(col_data):>8} {nan_count:>6} {valid.min():>10.3f} {valid.mean():>10.3f} {valid.median():>10.3f} {valid.max():>10.3f}")
        else:
            print(f"  {b:<20} {len(col_data):>8} {nan_count:>6} {'ALL NaN':>10}")
    else:
        print(f"  {b:<20} {'MISSING':>8}")

print(f"\n  NaN counts per band:")
for b, n in nan_summary.items():
    if n > 0:
        pct = 100 * n / max(len(df), 1)
        print(f"  ⚠  {b:<20} : {n}/{len(df)} NaN ({pct:.1f}%)")

# ── 6. After NaN fill — what RF actually predicts on ─────────────────────────
print("\n" + "─"*72)
print("  6. AFTER NaN FILL — canopy_height distribution sent to RF")
print("─"*72)

df_filled = df.copy()
for b in ALL_STACK_BANDS:
    if b not in df_filled.columns:
        df_filled[b] = 0.0
    else:
        median_val = df_filled[b].median() if not df_filled[b].isna().all() else 0.0
        df_filled[b] = df_filled[b].fillna(median_val)

ch_after = df_filled["canopy_height"]
print(f"  canopy_height after fill: min={ch_after.min():.2f} mean={ch_after.mean():.2f} median={ch_after.median():.2f} max={ch_after.max():.2f}")
print(f"  NaN remaining: {ch_after.isna().sum()}")

# ── 7. RF prediction ─────────────────────────────────────────────────────────
print("\n" + "─"*72)
print("  7. RANDOM FOREST PREDICTION ON THAR FEATURES")
print("─"*72)

from biomass_model import load_biomass_model
model, model_meta = load_biomass_model()

if model is None:
    print("  No cached model found. Skipping RF prediction test.")
else:
    X = df_filled[ALL_STACK_BANDS]
    preds = np.clip(model.predict(X), 0, None)
    print(f"  Predictions (Mg/ha): min={preds.min():.2f} mean={preds.mean():.2f} median={np.median(preds):.2f} max={preds.max():.2f} std={preds.std():.2f}")
    print(f"  Unique values (sample): {sorted(set(np.round(preds, 1)))[:10]}...")
    
    # Compare to training domain
    training_stats = model_meta.get("sample_statistics", {})
    print(f"\n  Model training domain (Bandipur):")
    for k, v in training_stats.items():
        print(f"    {k}: {v}")

# ── 8. WHRC biomass reference over Thar ──────────────────────────────────────
print("\n" + "─"*72)
print("  8. WHRC BIOMASS REFERENCE OVER THAR DESERT AOI")
print("─"*72)

whrc_img = ee.Image(DATASETS["whrc_biomass"])
whrc_bands = whrc_img.bandNames().getInfo()
print(f"  WHRC band names: {whrc_bands}")

whrc_clipped = whrc_img.clip(aoi)
for band in whrc_bands[:3]:
    try:
        stats = whrc_clipped.select(band).reduceRegion(
            reducer=ee.Reducer.mean()
                .combine(ee.Reducer.min(), sharedInputs=True)
                .combine(ee.Reducer.max(), sharedInputs=True)
                .combine(ee.Reducer.count(), sharedInputs=True),
            geometry=aoi,
            scale=500,
            maxPixels=1e7
        ).getInfo()
        print(f"\n  WHRC band '{band}':")
        for k, v in sorted(stats.items()):
            print(f"    {k}: {v}")
    except Exception as e:
        print(f"  WHRC band '{band}' error: {e}")

# ── 9. Root cause summary ─────────────────────────────────────────────────────
print("\n" + "="*72)
print("  DIAGNOSIS SUMMARY")
print("="*72)

print("""
WHAT TO LOOK FOR:
─────────────────
  A) canopy_height (ETH) mean for Thar:
     - Expected: 0–1 m (Thar Desert is bare sand dunes with sparse shrubs)
     - If you see ~14.6 m → model is using a cached result from Bandipur training
       OR the NaN-fill strategy is imputing the Bandipur-median canopy height

  B) NDVI for Thar:
     - Expected: 0.05–0.15 (bare soil reflectance)
     - If > 0.20 → suspect cloud contamination or seasonal vegetation (post-monsoon)

  C) NaN fill in analysis_service.py line 165:
     df[b] = df[b].fillna(df[b].median() if not df[b].isna().all() else 0.0)
     ISSUE: If canopy_height returns NaN (masked desert pixels), the median of
     the NON-NaN values is used. If ETH has NO data over this bare-sand AOI,
     df[b].median() returns NaN, then the else-branch returns 0.0.
     BUT if some pixels are valid (e.g. 0), the median=0 is fine.
     Key question: does ETH have valid data (0 m) or NaN over Thar?

  D) RF extrapolation bias:
     - The RF was trained on Bandipur (mean AGB ~68 Mg/ha, canopy ~12-15m)
     - When Thar features are fed in, the model extrapolates into a domain 
       it has never seen (arid, no canopy, low NDVI)
     - RF cannot extrapolate below its training range minimum;
       all predictions are clipped to the training minimum (~22 Mg/ha)
     - So even with perfect feature values, the RF outputs ≥22 Mg/ha
       because it was trained only on tropical forest data

  E) WHRC domain coverage:
     - WHRC/biomass/tropical covers tropical forests (lat ~±30°)
     - Thar Desert (Rajasthan, 27°N) is within the spatial domain
       but is NOT a forest — WHRC maps it as 0 or masked
     - If WHRC returns 0 or masked for Thar, training data for this
       biome is absent → RF cannot generalize
""")
print("="*72)
print("Script complete.")
