"""
CLI Test Script: Verify Earth Engine Data Acquisition & Feature Pipeline for Bandipur National Park.
Validates all requirements for Phase 1.
"""

import sys
import argparse
import json
from ee_service import verify_pipeline, initialize_earth_engine, get_bandipur_polygon, build_biomass_feature_stack, compute_stack_statistics
from config import BANDIPUR_METADATA, GCP_PROJECT


def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f"  {title.upper()}")
    print("=" * 75)


def print_check(name: str, passed: bool, details: str = ""):
    status_str = "[PASS]" if passed else "[FAIL]"
    print(f"  {status_str:<8} | {name:<35} | {details}")


def run_tests(demo_mode: bool = False, start_date: str = "2023-01-01", end_date: str = "2023-04-30"):
    print_banner(f"Forest Biomass AI - Earth Engine Pipeline Verification ({'DEMO MODE' if demo_mode else 'LIVE GEE'})")
    print(f"Target Study Area : {BANDIPUR_METADATA['name']}, {BANDIPUR_METADATA['state']}, {BANDIPUR_METADATA['country']}")
    print(f"Bounding Box      : {BANDIPUR_METADATA['bbox']}")
    print(f"Target Project    : {GCP_PROJECT}")
    print(f"Date Window       : {start_date} to {end_date}")
    print("-" * 75)

    print("\n[Phase 1] Executing Verification Pipeline...")
    report = verify_pipeline(start_date=start_date, end_date=end_date, force_demo=demo_mode)

    print_banner("Requirement Verification Results")

    # Req 1: Earth Engine Initialization
    ee_init = report.get("ee_initialized", False)
    print_check("1. Earth Engine Initialization", ee_init, f"Project: {GCP_PROJECT}")

    # Req 2: Sentinel-2 Data
    s2_info = report.get("datasets", {}).get("sentinel2", {})
    s2_count = s2_info.get("image_count", 0)
    s2_pass = s2_count > 0
    print_check("2. Sentinel-2 SR Data Available", s2_pass, f"{s2_count} scenes found (cloud < {s2_info.get('cloud_threshold_pct', 20)}%)")

    # Req 3: Cloud Mask & Vegetation Indices (NDVI, etc.)
    indices = s2_info.get("computed_indices", [])
    indices_pass = "NDVI" in indices and len(indices) >= 4
    print_check("3. Cloud Mask & Indices Computed", indices_pass, f"Indices: {', '.join(indices)}")

    # Req 4: Sentinel-1 Radar Data
    s1_info = report.get("datasets", {}).get("sentinel1", {})
    s1_count = s1_info.get("image_count", 0)
    s1_pass = s1_count > 0
    print_check("4. Sentinel-1 GRD Data Available", s1_pass, f"{s1_count} scenes found (IW mode, VV+VH)")

    # Req 5: Radar Features (VV, VH, VV_VH_diff, VV_VH_ratio_lin, RVI)
    r_features = s1_info.get("radar_features", s1_info.get("features", []))
    diff_present = "VV_VH_diff" in r_features
    print_check("5. Radar Backscatter Features", diff_present, f"Features: {', '.join(r_features)}")

    # Req 6: Forest Height / Structure (ETH 10m + GEDI companion)
    height_info = report.get("datasets", {}).get("canopy_height", {})
    height_pass = "primary" in height_info or height_info.get("available", False)
    primary_name = height_info.get("primary", {}).get("dataset", height_info.get("primary_dataset", "Unknown"))
    print_check("6. Canopy Height / Structure Layer", height_pass, f"{primary_name}")

    # Req 7: Combined Feature Stack
    stack_info = report.get("feature_stack", {})
    stack_valid = stack_info.get("valid", False)
    total_bands = stack_info.get("total_bands", 0)
    band_names = stack_info.get("band_names", [])
    print_check("7. Unified Feature Stack Valid", stack_valid and total_bands > 0, f"{total_bands} bands assembled")

    # Req 8: Summary Statistics
    stats = report.get("statistics", {})
    stats_pass = len(stats) > 0 and "NDVI" in stats and "canopy_height" in stats
    print_check("8. Region Statistics Calculated", stats_pass, f"{len(stats)} bands aggregated over study area")

    print_banner("Feature Stack Band Composition")
    for i, bname in enumerate(band_names, start=1):
        desc = stack_info.get("descriptions", {}).get(bname, {}).get("desc", "N/A")
        btype = stack_info.get("descriptions", {}).get(bname, {}).get("type", "feature")
        print(f"  {i:>2}. [{btype:<16}] {bname:<16} : {desc}")

    print_banner("Study Area Statistical Summary (Bandipur)")
    print(f"  {'Band / Feature':<18} | {'Min':<12} | {'Mean':<12} | {'Max':<12}")
    print("  " + "-" * 60)
    for band_name, stat_vals in stats.items():
        min_v = stat_vals.get("min", "N/A")
        mean_v = stat_vals.get("mean", "N/A")
        max_v = stat_vals.get("max", "N/A")
        print(f"  {band_name:<18} | {str(min_v):<12} | {str(mean_v):<12} | {str(max_v):<12}")

    print("\n" + "=" * 75)
    all_passed = ee_init and s2_pass and indices_pass and s1_pass and diff_present and height_pass and stack_valid and stats_pass
    if all_passed:
        print("  ALL VERIFICATION CRITERIA PASSED SUCCESSFULLY! [PHASE 1 COMPLETE]")
    else:
        print("  SOME CRITERIA FAILED OR COMPLETED WITH FALLBACK.")
    print("=" * 75 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Earth Engine Data Acquisition for Bandipur National Park.")
    parser.add_argument("--demo", action="store_true", help="Force DEMO_MODE fallback test")
    parser.add_argument("--start-date", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2023-04-30", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    exit_code = run_tests(demo_mode=args.demo, start_date=args.start_date, end_date=args.end_date)
    sys.exit(exit_code)
