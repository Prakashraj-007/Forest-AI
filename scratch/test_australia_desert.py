"""
Test script for remote Australian Desert AOI biomass estimation.
"""

import sys
sys.path.append(".")

from analysis_service import run_aoi_biomass_analysis

australia_desert_geom = {
    "type": "Polygon",
    "coordinates": [[
        [130.00, -25.00],
        [130.50, -25.00],
        [130.50, -24.50],
        [130.00, -24.50],
        [130.00, -25.00]
    ]]
}

print("Running Australian Desert AOI analysis...")
res = run_aoi_biomass_analysis(australia_desert_geom)
print("Status:", res.get("status"))
print("Mode:", res.get("mode"))
print("Model Version:", res.get("model_version"))
print("Prediction Statistics:", res.get("prediction_statistics"))
print("Feature Means:", res.get("feature_means"))
print("Arid Flag:", res.get("biome_context", {}).get("is_arid"))
print("Disclaimer:", res.get("biome_context", {}).get("disclaimer"))

assert res.get("status") == "success"
assert res.get("mode") == "LIVE_EARTH_ENGINE"
mean_agb = res.get("prediction_statistics", {}).get("mean")
print(f"Verified Australian Desert Mean Biomass: {mean_agb} Mg/ha")
assert mean_agb < 5.0, f"Expected desert biomass < 5.0 Mg/ha, got {mean_agb}"
print("=== AUSTRALIAN DESERT TEST PASSED! ===")
