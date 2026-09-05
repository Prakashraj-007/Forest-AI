"""
Comprehensive Verification Script for Forest AI Biomass Estimation Platform v2.
"""

import sys
sys.path.append(".")

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_pipeline():
    print("--- 1. Health Check ---")
    res = client.get("/health")
    print("Health response:", res.status_code, res.json())
    assert res.status_code == 200

    print("\n--- 2. Biomass v1 API Endpoint ---")
    res = client.get("/api/v1/biomass?demo=true")
    print("Biomass v1 demo response status:", res.status_code, res.json().get("status"))
    assert res.status_code == 200

    print("\n--- 3. Thar Desert AOI Analysis (Multi-Biome v2) ---")
    thar_payload = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [71.00, 27.00],
                [71.10, 27.00],
                [71.10, 27.10],
                [71.00, 27.10],
                [71.00, 27.00]
            ]]
        },
        "start_date": "2023-01-01",
        "end_date": "2023-04-30"
    }
    res = client.post("/api/v1/analysis/biomass", json=thar_payload)
    data = res.json()
    print("Thar AOI Analysis Status:", data.get("status"))
    print("Model Version:", data.get("model_version"))
    print("Predicted Mean Biomass:", data.get("prediction_statistics", {}).get("mean"), "Mg/ha")
    print("Feature Means:", data.get("feature_means"))
    print("Biome Context Arid Flag:", data.get("biome_context", {}).get("is_arid"))
    print("Disclaimer:", data.get("biome_context", {}).get("disclaimer"))
    print("Per-Biome Metrics Keys:", list(data.get("per_biome_metrics", {}).keys()))

    assert res.status_code == 200
    assert data.get("status") == "success"
    assert data.get("model_version") == "v2.0-multibiome"
    assert data.get("biome_context", {}).get("is_arid") is True
    # Verify Thar biomass mean is dramatically lower than v1's 29.1 Mg/ha
    mean_agb = data.get("prediction_statistics", {}).get("mean")
    assert mean_agb < 5.0, f"Expected Thar mean biomass < 5.0 Mg/ha, got {mean_agb}"

    print("\n--- 4. EE Biomass Tile URL Endpoint ---")
    tile_payload = {
        "layer": "biomass",
        "geometry": thar_payload["geometry"]
    }
    res = client.post("/api/v1/ee/tile-url", json=tile_payload)
    print("Tile URL Response:", res.status_code, res.json().get("success"), res.json().get("tile_url"))
    assert res.status_code == 200
    assert res.json().get("success") is True

    print("\n=== ALL VERIFICATION TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    test_pipeline()
