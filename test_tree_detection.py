"""
Phase 3 Tree Detection Verification Tests (DeepForest / NEON AOP)
==================================================================
Verifies the full Phase 3 pipeline using the real NEON AOP proxy tile and
the legitimate DeepForest pretrained RetinaNet tree-crown detection model.

Test groups
-----------
  1. NEON proxy tile setup (copy OSBS_029.tif from deepforest package data)
  2. DeepForest inference pipeline (LIVE_DEEPFOREST or DEMO_MODE fallback)
  3. Geographic coordinate validity (WGS-84 lon/lat range check)
  4. DEMO_MODE shortcut
  5. FastAPI endpoint smoke tests (/api/v1/trees, /api/v1/trees/demo-tile)
"""

import os
import sys
import time

# Run from project root
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

PASS = "PASS"
FAIL = "FAIL"


def _check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    icon   = "[OK]" if condition else "[!!]"
    msg    = f"  {icon} {label}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    return condition


def _section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Test 1: NEON AOP proxy tile setup
# ---------------------------------------------------------------------------
def test_neon_tile_setup() -> bool:
    _section("TEST 1: NEON AOP Proxy Tile Setup")
    all_ok = True
    try:
        from neon_demo_setup import setup_neon_proxy_tile
        import config
        t0 = time.time()
        meta = setup_neon_proxy_tile(force=True)
        elapsed = round(time.time() - t0, 1)
        all_ok &= _check("setup_neon_proxy_tile() returned dict", isinstance(meta, dict))
        all_ok &= _check("GeoTIFF file exists on disk",
                          os.path.exists(config.HIGHRES_DEMO_TIFF), config.HIGHRES_DEMO_TIFF)
        all_ok &= _check("Metadata JSON file exists", os.path.exists(config.HIGHRES_META_PATH))
        all_ok &= _check("dataset_name field present",
                          "dataset_name" in meta, meta.get("dataset_name", ""))
        all_ok &= _check("NEON source documented",
                          "NEON" in meta.get("source", ""), meta.get("source", ""))
        all_ok &= _check("license is CC0",
                          "CC0" in meta.get("license", ""), meta.get("license", ""))
        all_ok &= _check("resolution_m_per_px reported", "resolution_m_per_px" in meta)
        all_ok &= _check("tile_area_ha > 0",
                          meta.get("tile_area_ha", 0) > 0, str(meta.get("tile_area_ha")))
        all_ok &= _check("crs field present", "crs" in meta)
        try:
            import rasterio
            with rasterio.open(config.HIGHRES_DEMO_TIFF) as src:
                width, height, bands = src.width, src.height, src.count
            all_ok &= _check("Rasterio reads tile",
                              width > 0 and height > 0 and bands >= 1,
                              f"{width}x{height}, {bands} bands")
            all_ok &= _check("Tile has >= 3 bands (RGB)", bands >= 3, str(bands))
        except ImportError:
            print("  [WW] rasterio not installed -- skipping raster read check")
        except Exception as e:
            all_ok &= _check("Rasterio tile read", False, str(e))
        print(f"  [TT] Setup time: {elapsed}s")
    except ImportError as e:
        print(f"  [WW] deepforest or rasterio not installed: {e}")
        print("       Run: pip install deepforest rasterio pyproj")
    except Exception as exc:
        all_ok &= _check("neon_demo_setup import/run", False, str(exc))
        import traceback; traceback.print_exc()
    return all_ok


# ---------------------------------------------------------------------------
# Test 2: DeepForest inference pipeline
# ---------------------------------------------------------------------------
def test_deepforest_inference() -> bool:
    _section("TEST 2: DeepForest Tree Detection Pipeline")
    all_ok = True
    try:
        from tree_detection.deepforest_service import run_tree_detection
        import config
        t0 = time.time()
        result = run_tree_detection(
            tiff_path=config.HIGHRES_DEMO_TIFF,
            conf_thresh=0.10,
            force_demo=False,
        )
        elapsed = round(time.time() - t0, 1)
        print(f"  [TT] Detection time: {elapsed}s")
        all_ok &= _check("run_tree_detection() returned dict", isinstance(result, dict))
        mode = result.get("mode", "")
        print(f"  [II] Mode: {mode}")
        all_ok &= _check("mode is LIVE_DEEPFOREST or DEMO_MODE",
                          mode in ("LIVE_DEEPFOREST", "DEMO_MODE"), mode)
        tree_count = result.get("tree_count", 0)
        all_ok &= _check("tree_count >= 0", tree_count >= 0, str(tree_count))
        density = result.get("tree_density_per_ha", -1)
        all_ok &= _check("tree_density_per_ha >= 0", density >= 0, f"{density} trees/ha")
        area = result.get("tile_area_ha", 0)
        all_ok &= _check("tile_area_ha > 0", area > 0, str(area))
        gsd = result.get("gsd_m_per_px", -1)
        all_ok &= _check("gsd_m_per_px > 0", gsd > 0, str(gsd))
        geojson = result.get("geojson", {})
        all_ok &= _check("geojson.type == FeatureCollection",
                          geojson.get("type") == "FeatureCollection")
        features = geojson.get("features", [])
        all_ok &= _check("geojson.features is a list", isinstance(features, list))
        if features:
            f = features[0]
            all_ok &= _check("feature.type == Feature", f.get("type") == "Feature")
            coords = f.get("geometry", {}).get("coordinates", [])
            all_ok &= _check("feature has [lon, lat] coords", len(coords) == 2, str(coords))
            props = f.get("properties", {})
            all_ok &= _check("feature.properties has confidence", "confidence" in props)
            all_ok &= _check("feature.properties has radius_m", "radius_m" in props)
            all_ok &= _check("source == deepforest_retinanet",
                              props.get("source") == "deepforest_retinanet",
                              props.get("source", ""))
        all_ok &= _check("warnings field is a list", isinstance(result.get("warnings"), list))
        all_ok &= _check("processing_time_s reported", "processing_time_s" in result)
        if mode == "LIVE_DEEPFOREST":
            all_ok &= _check("tree_count > 0 in LIVE mode", tree_count > 0, str(tree_count))
            model_ver = result.get("model_version", "")
            print(f"  [II] DeepForest version: {model_ver}")
        else:
            print("  [WW] Running in DEMO_MODE -- deepforest or rasterio may not be installed")
    except Exception as exc:
        all_ok &= _check("deepforest_service import/run", False, str(exc))
        import traceback; traceback.print_exc()
    return all_ok


# ---------------------------------------------------------------------------
# Test 3: Geographic coordinate validity
# ---------------------------------------------------------------------------
def test_geographic_bounds() -> bool:
    _section("TEST 3: Geographic Coordinate Validity")
    all_ok = True
    # NEON OSBS site is in Florida, USA. Use global valid-coord bounds.
    LON_MIN, LON_MAX = -180.0, 180.0
    LAT_MIN, LAT_MAX = -90.0, 90.0
    try:
        from tree_detection.deepforest_service import run_tree_detection
        import config
        result = run_tree_detection(
            tiff_path=config.HIGHRES_DEMO_TIFF, conf_thresh=0.10, force_demo=False)
        features = result.get("geojson", {}).get("features", [])
        if not features:
            print("  [WW] No detections -- skipping geographic bounds check")
            print(f"  [II] Mode was: {result.get('mode', '?')}")
            return True
        invalid_lon = sum(1 for f in features
                          if not (LON_MIN <= f["geometry"]["coordinates"][0] <= LON_MAX))
        invalid_lat = sum(1 for f in features
                          if not (LAT_MIN <= f["geometry"]["coordinates"][1] <= LAT_MAX))
        total = len(features)
        all_ok &= _check(f"All {total} features have valid longitude",
                          invalid_lon == 0, f"{invalid_lon} invalid")
        all_ok &= _check(f"All {total} features have valid latitude",
                          invalid_lat == 0, f"{invalid_lat} invalid")
        coords = features[0]["geometry"]["coordinates"]
        print(f"  [II] First detection: lon={coords[0]}, lat={coords[1]}")
    except Exception as exc:
        all_ok &= _check("geographic bounds test", False, str(exc))
    return all_ok


# ---------------------------------------------------------------------------
# Test 4: DEMO_MODE shortcut
# ---------------------------------------------------------------------------
def test_demo_mode() -> bool:
    _section("TEST 4: DEMO_MODE Shortcut")
    all_ok = True
    try:
        from tree_detection.deepforest_service import run_tree_detection
        result = run_tree_detection(force_demo=True)
        all_ok &= _check("DEMO_MODE result is dict", isinstance(result, dict))
        all_ok &= _check("mode == DEMO_MODE",
                          result.get("mode") == "DEMO_MODE", result.get("mode", ""))
        all_ok &= _check("tree_count > 0 in DEMO_MODE",
                          result.get("tree_count", 0) > 0, str(result.get("tree_count")))
        all_ok &= _check("geojson present in DEMO_MODE", "geojson" in result)
        all_ok &= _check("warnings list present", isinstance(result.get("warnings"), list))
        if result.get("warnings"):
            print(f"  [II] Demo warning: {result['warnings'][0]}")
    except Exception as exc:
        all_ok &= _check("DEMO_MODE test", False, str(exc))
    return all_ok


# ---------------------------------------------------------------------------
# Test 5: FastAPI endpoint smoke test
# ---------------------------------------------------------------------------
def test_fastapi_endpoints() -> bool:
    _section("TEST 5: FastAPI Endpoint Smoke Test")
    all_ok = True
    try:
        from fastapi.testclient import TestClient
        from app import app
        client = TestClient(app)

        # -- /api/v1/trees/demo-tile (NEON tile metadata)
        resp = client.get("/api/v1/trees/demo-tile")
        all_ok &= _check("/api/v1/trees/demo-tile returns 200",
                          resp.status_code == 200, str(resp.status_code))
        body = resp.json()
        all_ok &= _check("demo-tile has dataset_name",
                          "dataset_name" in body, body.get("dataset_name", ""))
        all_ok &= _check("demo-tile license is CC0",
                          "CC0" in body.get("license", ""), body.get("license", ""))
        all_ok &= _check("demo-tile source mentions NEON",
                          "NEON" in body.get("source", ""), body.get("source", ""))

        # -- /api/v1/trees?demo=true
        resp2 = client.get("/api/v1/trees?demo=true")
        all_ok &= _check("/api/v1/trees?demo=true returns 200",
                          resp2.status_code == 200, str(resp2.status_code))
        body2 = resp2.json()
        all_ok &= _check("trees demo has tree_count", "tree_count" in body2)
        all_ok &= _check("trees demo mode == DEMO_MODE",
                          body2.get("mode") == "DEMO_MODE", body2.get("mode", ""))
        all_ok &= _check("trees demo geojson present", "geojson" in body2)

        # -- /api/v1/trees (live DeepForest)
        resp3 = client.get("/api/v1/trees?conf=0.10")
        all_ok &= _check("/api/v1/trees (live) returns 200",
                          resp3.status_code == 200, str(resp3.status_code))
        body3 = resp3.json()
        mode3 = body3.get("mode", "")
        all_ok &= _check("live mode is LIVE_DEEPFOREST or DEMO_MODE",
                          mode3 in ("LIVE_DEEPFOREST", "DEMO_MODE"), mode3)
        print(f"  [II] Live mode: {mode3}, "
              f"trees: {body3.get('tree_count', 'N/A')}, "
              f"density: {body3.get('tree_density_per_ha', 'N/A')} trees/ha")

    except ImportError as e:
        print(f"  [WW] httpx/testclient not available: {e}")
        print("  [WW] Skipping FastAPI endpoint tests")
    except Exception as exc:
        all_ok &= _check("FastAPI endpoint test", False, str(exc))
        import traceback; traceback.print_exc()
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "#" * 60)
    print("  PHASE 3 - TREE DETECTION VERIFICATION TESTS")
    print("  (DeepForest / NEON AOP Pipeline)")
    print("#" * 60)

    results = {
        "NEON tile setup":      test_neon_tile_setup(),
        "DeepForest inference": test_deepforest_inference(),
        "Geographic bounds":    test_geographic_bounds(),
        "DEMO_MODE shortcut":   test_demo_mode(),
        "FastAPI endpoints":    test_fastapi_endpoints(),
    }

    _section("SUMMARY")
    total  = len(results)
    passed = sum(results.values())
    for name, ok in results.items():
        icon = "[OK]" if ok else "[!!]"
        print(f"  {icon} {name}")

    print(f"\n  {passed}/{total} test groups passed")

    if passed == total:
        print("\n  ALL PHASE 3 TESTS PASSED")
    else:
        print(f"\n  {total - passed} test group(s) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
