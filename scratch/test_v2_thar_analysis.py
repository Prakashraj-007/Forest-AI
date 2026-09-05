"""
Script to test multi-biome v2 analysis on Thar Desert AOI.
"""

import json
from analysis_service import run_aoi_biomass_analysis

# Thar Desert test geometry
thar_geom = {
    "type": "Polygon",
    "coordinates": [[
        [71.00, 27.00],
        [71.10, 27.00],
        [71.10, 27.10],
        [71.00, 27.10],
        [71.00, 27.00]
    ]]
}

res = run_aoi_biomass_analysis(thar_geom)
print(json.dumps(res, indent=2))
