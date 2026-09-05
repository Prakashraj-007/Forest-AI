"""
Phase 3 Tree Detection Service
================================
Sliding-window YOLOv8-OBB inference pipeline for individual tree detection
on high-resolution (0.5 m/px) georeferenced aerial imagery.

Architecture overview
---------------------
1. Load the GeoTIFF with rasterio; read CRS and affine transform.
2. Slide a 640x640 pixel window across the image with configurable overlap.
3. Run YOLOv8-OBB inference on each chip.
4. Convert oriented bounding-box centres from pixel coordinates to geographic
   coordinates using the GeoTIFF affine transform.
5. Apply Non-Maximum Suppression (NMS) across overlapping chip detections to
   remove duplicate trees.
6. Return results as GeoJSON (FeatureCollection) + summary statistics.

Model strategy
--------------
* Primary  : custom-trained YOLOv8-OBB tree-detection weights (``tree_yolov8_obb.pt``)
  -- if present at config.YOLO_OBB_MODEL_PATH.
* Fallback  : Segment-anything-based blob detector (fast Laplacian-of-Gaussian)
  -- no GPU / model weights required; runs on CPU with scipy/skimage.

The fallback is clearly labelled in the response so downstream users understand
that results come from classical CV rather than the neural model.

DEMO_MODE
---------
If config.DEMO_MODE is set, the service skips all rasterio / model inference and
returns a pre-computed GeoJSON result derived from the demo tile generator
ground-truth tree centres.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy dependencies
# ---------------------------------------------------------------------------
try:
    import rasterio
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    from skimage.feature import blob_log
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

import config

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pixel_to_geo(row: float, col: float, transform) -> Tuple[float, float]:
    """Convert (row, col) pixel coordinates to (lon, lat) WGS-84 via rasterio transform."""
    # rasterio transform: x = transform.c + col * transform.a,  y = transform.f + row * transform.e
    x = transform.c + col * transform.a + row * transform.b
    y = transform.f + col * transform.d + row * transform.e
    return x, y  # may be in projected CRS (UTM)


def _project_xy_to_lonlat(x: float, y: float, src_crs_str: str) -> Tuple[float, float]:
    """Re-project (x, y) from source CRS to WGS-84 lon/lat if needed."""
    if not HAS_PYPROJ:
        # Best-effort: assume source is already lon/lat (EPSG:4326)
        return x, y
    try:
        src_crs = CRS.from_string(src_crs_str) if HAS_RASTERIO else None
        if src_crs is None or src_crs.is_geographic:
            return x, y
        transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        return lon, lat
    except Exception:
        return x, y


def _iou_centres(c1: Tuple[float, float, float], c2: Tuple[float, float, float]) -> float:
    """Simple proximity-based duplicate suppression.
    Returns 1.0 if centre distance < sum of radii (overlap), else 0.0.
    c = (cx_px, cy_px, radius_px)
    """
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    dist = math.sqrt(dx * dx + dy * dy)
    return 1.0 if dist < (c1[2] + c2[2]) * 0.75 else 0.0


def _nms_detections(detections: List[Dict], iou_thresh: float) -> List[Dict]:
    """
    Non-Maximum Suppression across chip detections in global-pixel space.
    Sorts by confidence descending; suppresses overlapping lower-conf detections.
    """
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
    kept: List[Dict] = []
    suppressed = [False] * len(dets)
    for i, d in enumerate(dets):
        if suppressed[i]:
            continue
        kept.append(d)
        c1 = (d["cx_global"], d["cy_global"], d.get("radius_px", 5))
        for j in range(i + 1, len(dets)):
            if suppressed[j]:
                continue
            c2 = (dets[j]["cx_global"], dets[j]["cy_global"], dets[j].get("radius_px", 5))
            if _iou_centres(c1, c2) > 0:
                suppressed[j] = True
    return kept


# ---------------------------------------------------------------------------
# YOLO-OBB inference on a single chip
# ---------------------------------------------------------------------------

def _run_yolo_on_chip(
    model,
    chip_rgb: np.ndarray,
    conf_thresh: float,
    x_offset: int,
    y_offset: int,
) -> List[Dict]:
    """Run YOLOv8-OBB on a single 640x640 RGB chip.
    Returns list of raw detection dicts with global-pixel coordinates.
    All numeric values are cast to native Python float/int to ensure
    JSON serializability.
    """
    results = model.predict(
        source=chip_rgb,
        conf=conf_thresh,
        iou=config.TREE_DETECTION_IOU_THRESH,
        verbose=False,
        save=False,
    )

    detections = []
    for r in results:
        # OBB results: r.obb
        if r.obb is None:
            continue
        boxes = r.obb
        # boxes.xywhr: [cx, cy, w, h, angle] in pixel space of the chip
        for k in range(len(boxes)):
            try:
                xywhr = boxes.xywhr[k].cpu().numpy()
                conf  = float(boxes.conf[k].cpu().numpy())
                cls   = int(boxes.cls[k].cpu().numpy())
                # Explicitly cast each element to Python float
                cx_chip = float(xywhr[0])
                cy_chip = float(xywhr[1])
                w_px    = float(xywhr[2])
                h_px    = float(xywhr[3])
                angle   = float(xywhr[4])
                cx_global = cx_chip + float(x_offset)
                cy_global = cy_chip + float(y_offset)
                radius_px = max(w_px, h_px) / 2.0
                detections.append({
                    "cx_global":  cx_global,
                    "cy_global":  cy_global,
                    "radius_px":  radius_px,
                    "width_px":   w_px,
                    "height_px":  h_px,
                    "angle_deg":  angle,
                    "confidence": conf,
                    "class_id":   cls,
                    "source":     "yolo_obb",
                })
            except Exception:
                continue
    return detections


# ---------------------------------------------------------------------------
# Blob fallback (Laplacian-of-Gaussian)
# ---------------------------------------------------------------------------

def _run_blob_fallback(
    img_rgb: np.ndarray,
    min_sigma: float = 3.0,
    max_sigma: float = 12.0,
    threshold: float = 0.06,
    x_offset: int = 0,
    y_offset: int = 0,
) -> List[Dict]:
    """Detect tree crown blobs via Laplacian-of-Gaussian on the green channel.
    Works without any trained model weights.
    All numeric values are explicitly cast to native Python float to ensure
    JSON serializability (blob_log returns numpy.float32/float64).
    """
    if not HAS_SKIMAGE:
        return []
    green = img_rgb[:, :, 1].astype(np.float32) / 255.0
    blobs = blob_log(
        green,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=6,
        threshold=threshold,
    )
    detections = []
    for blob in blobs:
        # blob_log returns numpy scalars -- cast everything to Python float
        row      = float(blob[0])
        col      = float(blob[1])
        sigma    = float(blob[2])
        radius_px = sigma * math.sqrt(2)  # already Python float after cast
        detections.append({
            "cx_global":  col + float(x_offset),
            "cy_global":  row + float(y_offset),
            "radius_px":  radius_px,
            "width_px":   radius_px * 2.0,
            "height_px":  radius_px * 2.0,
            "angle_deg":  0.0,
            "confidence": 0.80,   # fixed pseudo-confidence for blob detector
            "class_id":   0,
            "source":     "blob_log_fallback",
        })
    return detections


# ---------------------------------------------------------------------------
# Tiled inference over full image
# ---------------------------------------------------------------------------

def _tiled_inference(
    img_rgb: np.ndarray,
    model,
    use_blob_fallback: bool,
    conf_thresh: float,
    tile_size: int,
    tile_overlap: int,
) -> List[Dict]:
    """Slide a window over the full image and collect all raw detections."""
    h, w = img_rgb.shape[:2]
    stride = tile_size - tile_overlap
    all_detections: List[Dict] = []

    x_starts = list(range(0, max(w - tile_size, 0) + 1, stride))
    y_starts = list(range(0, max(h - tile_size, 0) + 1, stride))
    if not x_starts or x_starts[-1] + tile_size < w:
        x_starts.append(max(0, w - tile_size))
    if not y_starts or y_starts[-1] + tile_size < h:
        y_starts.append(max(0, h - tile_size))

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            chip = img_rgb[y0:y1, x0:x1]

            if use_blob_fallback:
                chip_dets = _run_blob_fallback(chip, x_offset=x0, y_offset=y0)
            else:
                # Pad chip to tile_size x tile_size if needed
                if chip.shape[0] < tile_size or chip.shape[1] < tile_size:
                    padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    padded[:chip.shape[0], :chip.shape[1]] = chip
                    chip = padded
                chip_dets = _run_yolo_on_chip(model, chip, conf_thresh, x0, y0)

            all_detections.extend(chip_dets)

    return all_detections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_tree_detection(
    tiff_path: Optional[str] = None,
    conf_thresh: Optional[float] = None,
    force_demo: bool = False,
) -> Dict[str, Any]:
    """
    Run the full tree-detection pipeline on a georeferenced GeoTIFF.

    Parameters
    ----------
    tiff_path : str, optional
        Path to input GeoTIFF. Defaults to config.HIGHRES_DEMO_TIFF.
    conf_thresh : float, optional
        Detection confidence threshold. Defaults to config.TREE_DETECTION_CONF_THRESH.
    force_demo : bool
        If True, returns pre-computed demo result without running inference.

    Returns
    -------
    dict
        {
          "mode": "LIVE_YOLO" | "LIVE_BLOB_FALLBACK" | "DEMO_MODE",
          "tree_count": int,
          "tree_density_per_ha": float,
          "tile_area_ha": float,
          "gsd_m_per_px": float,
          "crs": str,
          "geojson": GeoJSON FeatureCollection,
          "summary_statistics": {...},
          "processing_time_s": float,
          "warnings": [str],
        }
    """
    start_time = time.time()
    warnings: List[str] = []

    if conf_thresh is None:
        conf_thresh = config.TREE_DETECTION_CONF_THRESH
    if tiff_path is None:
        tiff_path = config.HIGHRES_DEMO_TIFF

    # ---- DEMO_MODE shortcut ------------------------------------------------
    if force_demo or config.DEMO_MODE:
        return _build_demo_result(warnings, start_time)

    # ---- Load GeoTIFF -------------------------------------------------------
    if not HAS_RASTERIO:
        warnings.append("rasterio not installed -- generating demo tile result.")
        return _build_demo_result(warnings, start_time)

    if not os.path.exists(tiff_path):
        warnings.append(
            f"GeoTIFF not found at '{tiff_path}'. "
            "Run demo_tile_generator.generate_demo_tile() first."
        )
        return _build_demo_result(warnings, start_time)

    try:
        with rasterio.open(tiff_path) as src:
            transform = src.transform
            crs_str   = str(src.crs) if src.crs else "EPSG:4326"
            img_data  = src.read()        # shape: (bands, H, W)
            gsd_m     = abs(float(transform.a))  # pixel width in map units (m)
    except Exception as exc:
        warnings.append(f"Failed to read GeoTIFF: {exc}")
        return _build_demo_result(warnings, start_time)

    # Convert to (H, W, 3) uint8 RGB
    if img_data.shape[0] >= 3:
        img_rgb = np.moveaxis(img_data[:3], 0, -1).astype(np.uint8)
    else:
        # Grayscale → repeat to RGB
        gray = img_data[0]
        img_rgb = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)

    h_px, w_px = img_rgb.shape[:2]
    tile_area_ha = (w_px * gsd_m * h_px * gsd_m) / 10_000

    # ---- Choose inference engine -------------------------------------------
    model = None
    use_blob_fallback = False

    if HAS_YOLO and os.path.exists(config.YOLO_OBB_MODEL_PATH):
        try:
            model = YOLO(config.YOLO_OBB_MODEL_PATH)
        except Exception as exc:
            warnings.append(f"Failed to load YOLO model: {exc}. Using blob fallback.")
            use_blob_fallback = True
    else:
        if not HAS_YOLO:
            warnings.append("ultralytics not installed. Using blob fallback.")
        else:
            warnings.append(
                f"YOLO weights not found at '{config.YOLO_OBB_MODEL_PATH}'. "
                "Using Laplacian-of-Gaussian blob fallback."
            )
        use_blob_fallback = True

    if use_blob_fallback and not HAS_SKIMAGE:
        warnings.append(
            "skimage not installed. Cannot run blob fallback. "
            "Returning demo result."
        )
        return _build_demo_result(warnings, start_time)

    # ---- Tiled inference ---------------------------------------------------
    raw_dets = _tiled_inference(
        img_rgb=img_rgb,
        model=model,
        use_blob_fallback=use_blob_fallback,
        conf_thresh=conf_thresh,
        tile_size=config.TILE_SIZE,
        tile_overlap=config.TILE_OVERLAP,
    )

    # ---- NMS ---------------------------------------------------------------
    kept_dets = _nms_detections(raw_dets, iou_thresh=config.TREE_DETECTION_IOU_THRESH)

    # ---- Build GeoJSON -----------------------------------------------------
    features = []
    confidences = []
    radii_m = []

    for det in kept_dets:
        cx_px = det["cx_global"]
        cy_px = det["cy_global"]
        radius_px = det.get("radius_px", 5.0)

        # Pixel → map coordinates (UTM or lon/lat depending on CRS)
        x_map, y_map = _pixel_to_geo(cy_px, cx_px, transform)

        # Reproject to WGS-84 lon/lat for GeoJSON
        lon, lat = _project_xy_to_lonlat(x_map, y_map, crs_str)

        radius_m = radius_px * gsd_m
        conf     = det.get("confidence", 0.0)
        confidences.append(conf)
        radii_m.append(radius_m)

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(lon, 7), round(lat, 7)]
            },
            "properties": {
                "confidence":    round(conf, 4),
                "radius_m":      round(radius_m, 2),
                "crown_area_m2": round(math.pi * radius_m ** 2, 2),
                "source":        det.get("source", "unknown"),
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    # ---- Summary statistics ------------------------------------------------
    tree_count = len(kept_dets)
    density_per_ha = round(tree_count / tile_area_ha, 1) if tile_area_ha > 0 else 0.0
    mean_conf = round(float(np.mean(confidences)), 4) if confidences else 0.0
    mean_radius_m = round(float(np.mean(radii_m)), 2) if radii_m else 0.0
    raw_count = len(raw_dets)

    mode = "LIVE_BLOB_FALLBACK" if use_blob_fallback else "LIVE_YOLO"

    elapsed = round(time.time() - start_time, 2)
    return {
        "mode":                   mode,
        "tree_count":             tree_count,
        "raw_detections_before_nms": raw_count,
        "tree_density_per_ha":    density_per_ha,
        "tile_area_ha":           round(tile_area_ha, 2),
        "gsd_m_per_px":           gsd_m,
        "crs":                    crs_str,
        "mean_confidence":        mean_conf,
        "mean_crown_radius_m":    mean_radius_m,
        "confidence_threshold":   conf_thresh,
        "geojson":                geojson,
        "summary_statistics": {
            "min_confidence":  round(float(min(confidences)), 4) if confidences else 0.0,
            "max_confidence":  round(float(max(confidences)), 4) if confidences else 0.0,
            "mean_confidence": mean_conf,
            "min_crown_radius_m": round(float(min(radii_m)), 2) if radii_m else 0.0,
            "max_crown_radius_m": round(float(max(radii_m)), 2) if radii_m else 0.0,
            "mean_crown_radius_m": mean_radius_m,
        },
        "processing_time_s": elapsed,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Demo fallback result
# ---------------------------------------------------------------------------

def _build_demo_result(warnings: List[str], start_time: float) -> Dict[str, Any]:
    """Return a pre-computed demo result based on the tile generator metadata."""
    from demo_tile_generator import (
        OUTPUT_META, N_TREES, TILE_W, TILE_H, GSD_M,
        ORIGIN_EASTING, ORIGIN_NORTHING
    )

    demo_meta_path = OUTPUT_META
    tile_area_ha = (TILE_W * GSD_M * TILE_H * GSD_M) / 10_000

    features = []
    if os.path.exists(demo_meta_path):
        try:
            with open(demo_meta_path) as f:
                dm = json.load(f)
            centres = dm.get("tree_centres_geo", [])
        except Exception:
            centres = []
    else:
        # Regenerate first 20 synthetically
        rng = np.random.default_rng(2024)
        cx_arr = rng.integers(16, TILE_W - 16, size=N_TREES)
        cy_arr = rng.integers(16, TILE_H - 16, size=N_TREES)
        cr_arr = rng.integers(4,  17,           size=N_TREES)
        centres = [
            {
                "easting_m":  round(ORIGIN_EASTING  + int(cx_arr[i]) * GSD_M, 2),
                "northing_m": round(ORIGIN_NORTHING  - int(cy_arr[i]) * GSD_M, 2),
                "radius_m":   round(int(cr_arr[i]) * GSD_M, 2),
            }
            for i in range(min(20, N_TREES))
        ]

    for centre in centres:
        east = centre["easting_m"]
        north = centre["northing_m"]
        r_m  = centre["radius_m"]

        # Convert UTM 43N to approx lon/lat for GeoJSON
        lon, lat = _utm43n_to_lonlat(east, north)

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(lon, 7), round(lat, 7)]},
            "properties": {
                "confidence":    0.85,
                "radius_m":      r_m,
                "crown_area_m2": round(math.pi * r_m ** 2, 2),
                "source":        "demo_mode",
            }
        })

    warnings.append(
        "DEMO_MODE: Results are derived from synthetic tile ground truth, "
        "not from live model inference."
    )

    elapsed = round(time.time() - start_time, 2)
    return {
        "mode":                      "DEMO_MODE",
        "tree_count":                N_TREES,
        "raw_detections_before_nms": N_TREES,
        "tree_density_per_ha":       round(N_TREES / tile_area_ha, 1),
        "tile_area_ha":              round(tile_area_ha, 2),
        "gsd_m_per_px":              GSD_M,
        "crs":                       "EPSG:32643 (WGS 84 / UTM zone 43N)",
        "mean_confidence":           0.85,
        "mean_crown_radius_m":       GSD_M * 10,  # median crown ~5px * 0.5 m/px
        "confidence_threshold":      config.TREE_DETECTION_CONF_THRESH,
        "geojson":                   {"type": "FeatureCollection", "features": features},
        "summary_statistics": {
            "min_confidence": 0.80,
            "max_confidence": 0.95,
            "mean_confidence": 0.85,
            "min_crown_radius_m": GSD_M * 4,
            "max_crown_radius_m": GSD_M * 16,
            "mean_crown_radius_m": GSD_M * 10,
        },
        "processing_time_s": elapsed,
        "warnings": warnings,
    }


def _utm43n_to_lonlat(easting: float, northing: float) -> Tuple[float, float]:
    """Approximate UTM zone 43N to WGS-84 lon/lat.
    Uses pyproj if available, else a simple linear approximation.
    """
    if HAS_PYPROJ:
        try:
            transformer = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(easting, northing)
            return lon, lat
        except Exception:
            pass
    # Fallback: central meridian of UTM 43N = 75E, scale factor 0.9996
    # Very rough -- good enough for DEMO_MODE display
    central_meridian = 75.0
    k0 = 0.9996
    R = 6_371_000.0
    lat = math.degrees(northing / R)
    lon = central_meridian + math.degrees(easting / (k0 * R * math.cos(math.radians(lat))))
    return lon, lat
