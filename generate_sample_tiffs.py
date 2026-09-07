"""
Generate Sample GeoTIFF Files for Multiple Geographic Locations
===============================================================
Generates ready-to-use, georeferenced GeoTIFF (.tif) files from distinct
ecosystems and geographic locations across the world with embedded CRS/transforms:

1. yellowstone_conifer_usa.tif   -> Yellowstone National Park, Wyoming, USA (EPSG:32612)
2. california_sierra_soap.tif    -> Sierra National Forest, California, USA (EPSG:32611)
3. bandipur_canopy_india.tif     -> Bandipur National Park, Karnataka, India (EPSG:32643)
4. florida_everglades_osbs.tif   -> Ordway-Swisher / Florida, USA (EPSG:32617)
"""

import os
import sys
import shutil
import json
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import deepforest
    from deepforest import get_data
    HAS_DEEPFOREST = True
except ImportError:
    HAS_DEEPFOREST = False


def create_sample_tiffs():
    out_dir = os.path.join(os.path.dirname(__file__), "sample_tiffs")
    os.makedirs(out_dir, exist_ok=True)
    
    # Also copy to frontend public directory if it exists
    frontend_public = os.path.join(os.path.dirname(__file__), "frontend", "public", "sample_tiffs")
    os.makedirs(frontend_public, exist_ok=True)

    results = []

    # 1. Florida OSBS
    if HAS_DEEPFOREST:
        osbs_src = get_data("OSBS_029.tif")
        if os.path.exists(osbs_src):
            dst1 = os.path.join(out_dir, "florida_osbs_usa.tif")
            shutil.copy2(osbs_src, dst1)
            shutil.copy2(osbs_src, os.path.join(frontend_public, "florida_osbs_usa.tif"))
            results.append({
                "filename": "florida_osbs_usa.tif",
                "path": dst1,
                "location": "Ordway-Swisher Biological Station, Florida, USA",
                "coordinates": "29.689° N, 81.993° W",
                "crs": "EPSG:32617 (UTM Zone 17N)",
                "ecosystem": "Subtropical Longleaf Pine & Mixed Oak Woodland",
                "file_size": f"{os.path.getsize(dst1)/1024:.1f} KB"
            })

    # 2. Yellowstone National Park, Wyoming (2019_YELL)
    if HAS_DEEPFOREST and HAS_RASTERIO:
        yell_src = get_data("2019_YELL_2_528000_4978000_image_crop2.png")
        if os.path.exists(yell_src):
            img = Image.open(yell_src).convert("RGB")
            arr = np.array(img)  # H x W x 3
            h, w, c = arr.shape
            
            # Yellowstone NEON coordinates: UTM Zone 12N, Easting: 528000, Northing: 4978000 + height * 0.1
            gsd = 0.1  # 10cm / pixel
            origin_x = 528000.0
            origin_y = 4978000.0 + (h * gsd)
            transform = from_origin(origin_x, origin_y, gsd, gsd)
            
            dst2 = os.path.join(out_dir, "yellowstone_conifer_usa.tif")
            with rasterio.open(
                dst2,
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=3,
                dtype=arr.dtype,
                crs="EPSG:32612",  # UTM Zone 12N (Wyoming / Yellowstone)
                transform=transform,
                compress="lzw"
            ) as dst:
                for band_idx in range(3):
                    dst.write(arr[:, :, band_idx], band_idx + 1)
            
            shutil.copy2(dst2, os.path.join(frontend_public, "yellowstone_conifer_usa.tif"))
            results.append({
                "filename": "yellowstone_conifer_usa.tif",
                "path": dst2,
                "location": "Yellowstone National Park, Wyoming, USA",
                "coordinates": "44.954° N, 110.645° W",
                "crs": "EPSG:32612 (UTM Zone 12N)",
                "ecosystem": "Temperate Coniferous Forest (Lodgepole Pine / Fir)",
                "file_size": f"{os.path.getsize(dst2)/1024:.1f} KB"
            })

    # 3. Sierra National Forest / Soaproot Saddle, California, USA (SOAP_061)
    if HAS_DEEPFOREST and HAS_RASTERIO:
        soap_src = get_data("SOAP_061.png")
        if os.path.exists(soap_src):
            img = Image.open(soap_src).convert("RGB")
            arr = np.array(img)
            h, w, c = arr.shape
            
            # Soaproot NEON coordinates: UTM Zone 11N, Easting: 297500, Northing: 4100000
            gsd = 0.1  # 10cm / pixel
            origin_x = 297500.0
            origin_y = 4100000.0 + (h * gsd)
            transform = from_origin(origin_x, origin_y, gsd, gsd)
            
            dst3 = os.path.join(out_dir, "california_sierra_soap.tif")
            with rasterio.open(
                dst3,
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=3,
                dtype=arr.dtype,
                crs="EPSG:32611",  # UTM Zone 11N (California Sierra Nevada)
                transform=transform,
                compress="lzw"
            ) as dst:
                for band_idx in range(3):
                    dst.write(arr[:, :, band_idx], band_idx + 1)
            
            shutil.copy2(dst3, os.path.join(frontend_public, "california_sierra_soap.tif"))
            results.append({
                "filename": "california_sierra_soap.tif",
                "path": dst3,
                "location": "Soaproot Saddle, Sierra National Forest, California, USA",
                "coordinates": "37.031° N, 119.261° W",
                "crs": "EPSG:32611 (UTM Zone 11N)",
                "ecosystem": "Sierra Nevada Mixed Montane Conifer & Oak Woodland",
                "file_size": f"{os.path.getsize(dst3)/1024:.1f} KB"
            })

    # 4. Bandipur National Park, Karnataka, India
    bandipur_existing = os.path.join(os.path.dirname(__file__), "data", "demo_highres", "bandipur_canopy_highres.tif")
    if os.path.exists(bandipur_existing):
        dst4 = os.path.join(out_dir, "bandipur_canopy_india.tif")
        shutil.copy2(bandipur_existing, dst4)
        shutil.copy2(bandipur_existing, os.path.join(frontend_public, "bandipur_canopy_india.tif"))
        results.append({
            "filename": "bandipur_canopy_india.tif",
            "path": dst4,
            "location": "Bandipur National Park, Karnataka, India",
            "coordinates": "11.725° N, 76.381° E",
            "crs": "EPSG:32643 (UTM Zone 43N)",
            "ecosystem": "Tropical Moist & Dry Deciduous Canopy (Teak, Rosewood)",
            "file_size": f"{os.path.getsize(dst4)/(1024*1024):.2f} MB"
        })

    # Write summary metadata JSON
    manifest_path = os.path.join(out_dir, "sample_tiffs_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Generated {len(results)} Sample GeoTIFF files:")
    for item in results:
        print(f"  - [{item['filename']}] Location: {item['location']} | CRS: {item['crs']} | Coords: {item['coordinates']}")

    return results

if __name__ == "__main__":
    create_sample_tiffs()
