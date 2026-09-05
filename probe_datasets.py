"""
Dataset Candidate Probe — Multi-Biome Training Data Research
=============================================================
Probes every candidate biomass reference dataset over:
  (A) Bandipur tropical forest AOI (existing training domain)
  (B) Thar Desert AOI (proposed arid domain)
  (C) Sahel semi-arid AOI (global arid proxy)

Reports: pixel count, min/mean/median/max, unit notes, coverage status.
"""

import sys, warnings
warnings.filterwarnings("ignore")

from ee_service import initialize_earth_engine
from config import DATASETS

print("=" * 70)
print("  DATASET CANDIDATE PROBE")
print("=" * 70)

if not initialize_earth_engine():
    print("EE init failed."); sys.exit(1)

import ee

# ── Reference AOIs ──────────────────────────────────────────────────────────
BANDIPUR_POLY = [
    [76.45, 11.60], [76.80, 11.60], [76.80, 11.85],
    [76.45, 11.85], [76.45, 11.60]
]
THAR_POLY = [
    [70.72, 27.35], [71.18, 27.30], [71.10, 26.88],
    [70.68, 26.92], [70.72, 27.35]
]
# Sahel dryland proxy (Mali/Niger border region, 15N) — arid savanna, ~5-15 Mg/ha
SAHEL_POLY = [
    [2.0, 14.5], [4.0, 14.5], [4.0, 15.5],
    [2.0, 15.5], [2.0, 14.5]
]
# Australian arid outback (Simpson Desert) — extreme arid, ~0-5 Mg/ha
SIMPSON_POLY = [
    [136.0, -25.5], [138.0, -25.5], [138.0, -24.5],
    [136.0, -24.5], [136.0, -25.5]
]

AOIs = {
    "Bandipur (tropical forest)": ee.Geometry.Polygon([BANDIPUR_POLY]),
    "Thar Desert (arid)":          ee.Geometry.Polygon([THAR_POLY]),
    "Sahel (semi-arid savanna)":   ee.Geometry.Polygon([SAHEL_POLY]),
    "Simpson Desert (arid)":       ee.Geometry.Polygon([SIMPSON_POLY]),
}


def probe_image(img, aoi_name, aoi, band, scale, max_px=1e8):
    """Reduce a single-band image over an AOI and print statistics."""
    try:
        stats = img.select(band).reduceRegion(
            reducer=ee.Reducer.mean()
                .combine(ee.Reducer.min(), sharedInputs=True)
                .combine(ee.Reducer.max(), sharedInputs=True)
                .combine(ee.Reducer.median(), sharedInputs=True)
                .combine(ee.Reducer.count(), sharedInputs=True),
            geometry=aoi,
            scale=scale,
            maxPixels=max_px
        ).getInfo()
        count = stats.get(f"{band}_count", 0)
        mean  = stats.get(f"{band}_mean")
        mn    = stats.get(f"{band}_min")
        mx    = stats.get(f"{band}_max")
        med   = stats.get(f"{band}_median")
        mean_str = f"{mean:.3f}" if mean is not None else "None"
        mn_str   = f"{mn:.3f}"   if mn   is not None else "None"
        mx_str   = f"{mx:.3f}"   if mx   is not None else "None"
        med_str  = f"{med:.3f}"  if med  is not None else "None"
        print(f"    {aoi_name:<32s} count={count:>8}  mean={mean_str:>8}  "
              f"min={mn_str:>8}  median={med_str:>8}  max={mx_str:>8}")
    except Exception as e:
        print(f"    {aoi_name:<32s} ERROR: {e}")


# ── 1. WHRC Pantropical (existing) ──────────────────────────────────────────
print("\n[1] WHRC/biomass/tropical — band='Mg', scale=500m, unit=Mg/ha (woody AGB)")
whrc = ee.Image(DATASETS["whrc_biomass"])
for name, aoi in AOIs.items():
    probe_image(whrc, name, aoi, "Mg", 500)

# ── 2. NASA ORNL Spawn & Gibbs 2020 ─────────────────────────────────────────
print("\n[2] NASA/ORNL/biomass_carbon_density/v1 — band='agb', scale=300m, unit=Mg C/ha")
print("    NOTE: 'agb' is CARBON density (Mg C/ha). Multiply x2.0 => AGB dry mass (Mg/ha)")
ornl = ee.ImageCollection(DATASETS["nasa_ornl_biomass"]).first()
for name, aoi in AOIs.items():
    probe_image(ornl, name, aoi, "agb", 300)

# ── 3. GEDI L4A Monthly (2021 — good global coverage year) ──────────────────
print("\n[3] LARSE/GEDI/GEDI04_A_002_MONTHLY — band='agbd', scale=1000m, unit=Mg/ha")
print("    Filtering: l4_quality_flag==1, year=2021, summing all months")
gedi_col = (
    ee.ImageCollection(DATASETS["gedi_l4a_monthly"])
    .filterDate("2021-01-01", "2022-01-01")
    .select(["agbd", "l4_quality_flag"])
)
# Apply quality flag mask across the collection, then mosaic
def apply_quality(img):
    qf = img.select("l4_quality_flag").eq(1)
    return img.select("agbd").updateMask(qf)

gedi_masked = gedi_col.map(apply_quality).mean()
for name, aoi in AOIs.items():
    probe_image(gedi_masked, name, aoi, "agbd", 1000)

# ── 4. Also check GEDI L4A over Thar at 25m footprint level ─────────────────
print("\n[4] GEDI L4A footprint count over Thar (quality-filtered, 2020-2023):")
gedi_thar_col = (
    ee.ImageCollection(DATASETS["gedi_l4a_monthly"])
    .filterBounds(ee.Geometry.Polygon([THAR_POLY]))
    .filterDate("2020-01-01", "2023-12-31")
    .select(["agbd", "l4_quality_flag"])
)
thar_count = int(gedi_thar_col.size().getInfo())
print(f"    GEDI L4A monthly images available over Thar (2020-2023): {thar_count}")

# Merge and probe
gedi_thar_merged = gedi_thar_col.map(apply_quality).mean()
probe_image(gedi_thar_merged, "Thar GEDI combined", ee.Geometry.Polygon([THAR_POLY]), "agbd", 1000)

# ── 5. ORNL AGB conversion check on Bandipur ────────────────────────────────
print("\n[5] ORNL 'agb' x2.0 => AGB Mg/ha conversion check (Bandipur):")
ornl_bandipur = (
    ee.ImageCollection(DATASETS["nasa_ornl_biomass"])
    .first()
    .select("agb")
    .multiply(2.0)
)
try:
    s = ornl_bandipur.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
        geometry=ee.Geometry.Polygon([BANDIPUR_POLY]),
        scale=300, maxPixels=1e7
    ).getInfo()
    print(f"    Bandipur ORNL*2.0: mean={s.get('agb_mean'):.2f} Mg/ha, count={s.get('agb_count')}")
except Exception as e:
    print(f"    ERROR: {e}")

# ── 6. Check ORNL AGB over Thar ──────────────────────────────────────────────
print("\n[6] ORNL AGB over Thar Desert (raw carbon density then x2.0):")
ornl_thar = (
    ee.ImageCollection(DATASETS["nasa_ornl_biomass"])
    .first()
    .select("agb")
)
thar_aoi = ee.Geometry.Polygon([THAR_POLY])
try:
    s = ornl_thar.reduceRegion(
        reducer=ee.Reducer.mean()
            .combine(ee.Reducer.min(), sharedInputs=True)
            .combine(ee.Reducer.max(), sharedInputs=True)
            .combine(ee.Reducer.count(), sharedInputs=True),
        geometry=thar_aoi, scale=300, maxPixels=1e8
    ).getInfo()
    raw_mean = s.get("agb_mean")
    count    = s.get("agb_count")
    raw_min  = s.get("agb_min")
    raw_max  = s.get("agb_max")
    print(f"    Thar ORNL raw (Mg C/ha): mean={raw_mean}, min={raw_min}, max={raw_max}, count={count}")
    if raw_mean is not None:
        print(f"    Thar ORNL x2.0 (Mg/ha AGB): mean={raw_mean*2:.3f}, max={raw_max*2:.3f}")
except Exception as e:
    print(f"    ERROR: {e}")

# ── 7. Summary recommendation ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  INTERPRETATION GUIDE")
print("=" * 70)
print("""
Dataset selection criteria:
  [A] Has VALID (non-None, non-zero count) pixels over BOTH Bandipur AND Thar
  [B] Unit is Mg/ha AGB or convertible with a known factor
  [C] Global coverage (not pantropical-only like WHRC)
  [D] Year ~2010-2023 (modern satellite era)
  [E] Publicly licensed (CC, NASA open, ESA open)

Expected results:
  WHRC       — count=0 over Thar (confirmed in prior diagnostic) => FAIL [A]
  NASA ORNL  — global, includes non-forest (desert=0 or near-0 carbon) => LIKELY PASS [A]
  GEDI L4A   — LiDAR transects; may have sparse coverage in deep desert => CHECK [A]

Best choice for arid reference data:
  If ORNL has valid pixels (count>0) over Thar => use ORNL as the multi-biome reference
  and convert from Mg C/ha -> Mg/ha AGB using factor 2.0 (IPCC default C fraction 0.5)

  If ORNL Thar count=0 => use GEDI L4A (quality-filtered footprints over Thar + Sahel)
  as the arid reference, supplemented by ORNL over Bandipur
""")
print("=" * 70)
print("Probe complete.")
