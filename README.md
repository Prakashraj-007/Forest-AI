# Forest AI: Tree Counting & Biomass Estimation
## Phase 1: Earth Engine Data Acquisition & Preprocessing
## Phase 2: Aboveground Biomass (AGB) Estimation Pipeline

This repository contains an end-to-end cloud and satellite remote sensing pipeline for estimating Aboveground Biomass (AGB in Mg/ha) and monitoring forest structure for **Bandipur National Park, Karnataka, India**.

---

### Study Area: Bandipur National Park, Karnataka, India
- **Bounding Box**: `[76.45°E, 11.60°N]` to `[76.80°E, 11.85°N]`
- **Approximate Area**: ~750 km²
- **GCP Project**: `forest-ai-507705`
- **Forest Characteristics**: Dry deciduous, moist deciduous, and scrub forest eco-regions in the Western Ghats complex.

---

### Pipeline Architecture

```
Sentinel-1 GRD (VV, VH, VV_VH_diff, VV_VH_ratio_lin, RVI)
       +
Sentinel-2 SR (B2, B3, B4, B8, B11, B12)
       +
Vegetation Indices (NDVI, EVI, SAVI, NDWI)
       +
Canopy Height (ETH Global Canopy Height 10m - GEDI calibrated)
       ↓
16-Band Multi-Sensor Feature Stack
       ↓
Biomass Reference Data (WHRC Pantropical Biomass / GEDI L4A AGBD in Mg/ha)
       ↓
Spatially Distributed Sampling (1000 points over Bandipur, 100m scale)
       ↓
Random Forest Regressor (200 trees, 80/20 train/validation split)
       ↓
Independent Validation (R² = 0.791, RMSE = 19.02 Mg/ha, MAE = 13.37 Mg/ha)
       ↓
Georeferenced Biomass Prediction Raster (GeoTIFF, EPSG:4326)
       ↓
FastAPI Backend API Endpoints & CLI Verification Suite
```

---

### Phase 2: Biomass Modeling Details

#### 1. Biomass Reference / Target Datasets
- **Primary Reference**: WHRC Pantropical Biomass Dataset (`WHRC/biomass/tropical`)
  - **Band**: `Mg` (Mg/ha of dry aboveground live woody biomass)
  - **Spatial Resolution**: 500 meters
  - **Citation**: Baccini et al. (2012), *Estimated carbon stocks in tropical forests with GLAS lidar data*, Nature Climate Change.
- **Companion / Alternative Reference**: NASA GEDI L4A Footprint AGBD Gridded Monthly (`LARSE/GEDI/GEDI04_A_002_MONTHLY`)
  - **Band**: `agbd` (Aboveground Biomass Density in Mg/ha)
  - **Spatial Resolution**: 1000 meters
  - **Citation**: Dubayah et al. (2022), *The Global Ecosystem Dynamics Investigation: High-resolution waveform lidar for carbon and ecosystem structure*.
- **Target Unit**: **Mg/ha** (Megagrams of dry biomass per hectare, equivalent to metric tons/ha).

#### 2. Random Forest Regression Configuration
- **Model**: `sklearn.ensemble.RandomForestRegressor`
- **Number of Estimators**: 200 trees
- **Max Depth**: 16
- **Min Samples Leaf**: 2
- **Split Ratio**: 80% Training (797 samples) / 20% Validation (200 samples)
- **Random State**: 42 (fully reproducible)
- **Features Used (16)**:
  - Optical Reflectance: `B2`, `B3`, `B4`, `B8`, `B11`, `B12`
  - Vegetation Indices: `NDVI`, `EVI`, `SAVI`, `NDWI`
  - Radar SAR: `VV`, `VH`, `VV_VH_diff` ($VV_{dB} - VH_{dB}$), `VV_VH_ratio_lin`, `RVI`
  - Forest Structure: `canopy_height` (ETH Zurich GEDI-calibrated 10m)

#### 3. Actual Empirical Validation Performance
- **$R^2$ Score**: **0.7911** (79.1% of biomass variance explained across Bandipur National Park)
- **Root Mean Squared Error (RMSE)**: **19.02 Mg/ha**
- **Mean Absolute Error (MAE)**: **13.37 Mg/ha**

#### 4. Top Feature Importances
1. **`canopy_height`** (0.5970 / 59.7%): Strongest structural predictor of timber volume and biomass.
2. **`B3`** (0.1542 / 15.4%): Green chlorophyll reflectance peak.
3. **`B4`** (0.0467 / 4.7%): Red absorption band.
4. **`B12`** (0.0286 / 2.9%): SWIR-2 lignin and cellulose proxy.
5. **`B2`** (0.0238 / 2.4%): Blue atmospheric baseline.
6. **`VH`** (0.0182 / 1.8%): Sentinel-1 C-band cross-polarization (canopy volume scattering).
7. **`RVI`** (0.0167 / 1.7%): Dual-polarization Radar Vegetation Index.
8. **`VV_VH_diff`** (0.0159 / 1.6%): Decibel difference ($VV_{dB} - VH_{dB}$).

---

### Scientific Transparency & Limitations

> [!IMPORTANT]
> - **Model Estimates vs. Field Truth**: Remote sensing biomass estimates are statistical model predictions calibrated with reference spaceborne LiDAR datasets, not direct in-situ forest plot inventories.
> - **Sensor Scale Discrepancies**: Sentinel-2 optical bands (10-20m), Sentinel-1 SAR (10m), and WHRC reference biomass (500m) operate at differing native spatial resolutions. Resampling over 100m pixels mitigates spatial mismatch while capturing landscape-level forest density.
> - **Canopy Height Dominance**: Structural canopy height derived from GEDI LiDAR calibration contributes ~60% of predictive power, confirming that 3D vertical forest structure is more strongly correlated with total aboveground dry mass than 2D optical reflectance alone.
> - **Hackathon Prototype Scope**: This implementation is designed as a high-performance demonstration and research prototype for regional forest monitoring, not an accredited UN-REDD+ carbon credit auditing system.

---

### Quick Start & CLI Verification

#### 1. Run Phase 1 Earth Engine Pipeline Test
```bash
python test_ee_pipeline.py
```

#### 2. Run Phase 2 Biomass Pipeline Test
```bash
python test_biomass.py
```

#### 3. Test Offline DEMO_MODE
```bash
python test_biomass.py --demo
```

#### 4. Run the FastAPI Web Server
```bash
python app.py
```
Or:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Interactive documentation: `http://127.0.0.1:8000/docs`

#### Available API Endpoints:
- `GET /health`: Service health check and GCP project status.
- `GET /api/v1/study-area`: GeoJSON polygon & bounding box for Bandipur National Park.
- `GET /api/v1/features`: Descriptions of all 16 multi-sensor features.
- `GET /api/v1/ee-test`: Live Earth Engine verification diagnostic endpoint.
- `GET /api/v1/biomass`: Full biomass pipeline report (model status, validation metrics, feature importances, prediction raster stats).
- `GET /api/v1/biomass/model`: Model specifications, hyperparameters, metrics, and sorted feature importances.
- `GET /api/v1/biomass/statistics`: Area-wide biomass summary statistics (min, max, mean, median, std, Mg/ha).
- `POST /api/v1/biomass/train`: On-demand training/retraining endpoint.

### Phase 3: Individual Tree Crown Detection (DeepForest / NEON AOP)

- **Model Architecture**: DeepForest RetinaNet (v2.1.0 pretrained individual tree crown detector)
- **Proxy Demonstration Dataset**: Real high-resolution (0.10 m/pixel) airborne RGB imagery from **Ordway-Swisher Biological Station (OSBS), Florida, USA** via NEON Airborne Observation Platform (AOP).
- **License**: CC0 1.0 Universal (Public Domain).
- **Reason for Proxy**: Sub-meter aerial drone/aircraft RGB imagery over Indian national protected reserves is restricted; NEON AOP provides legitimate open airborne calibration data.
- **Inference Outputs**: Bounding boxes, geographic WGS-84 centroids, crown area ($m^2$), confidence scores, tree counts, and canopy density (trees/ha).

---

### Phase 4: Interactive Forest AI Web Dashboard

A modern, responsive single-page web dashboard built with **Vite + Mapbox GL JS + Turf.js**.

- **Interactive Map**: Multi-layer Mapbox GL JS map with satellite basemap, Bandipur boundary polygon, biomass heatmap grid, and DeepForest tree crown overlays.
- **Custom AOI Polygon Drawing**: Interactive drawing tool with real-time area calculation (ha / km²) powered by Turf.js.
- **Arbitrary AOI Biomass Analysis**: Sends custom drawn GeoJSON polygons to `POST /api/v1/analysis/biomass` for on-demand satellite stack extraction and Random Forest inference.
- **Mode Switcher**: Clean separation between landscape-scale **Biomass & Carbon Stock Analysis** and stand-scale **Tree Crown Detection**.
- **Prominent NEON Proxy Notice**: Dedicated warning banners and attribution ensuring transparency that tree detection uses the NEON AOP demonstration proxy.

---

### Quick Start & Running the Application

#### 1. Start the FastAPI Backend
```bash
python app.py
```
Backend runs on `http://127.0.0.1:8000` (API Docs at `http://127.0.0.1:8000/docs`).

#### 2. Start the Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
```
Frontend runs on `http://127.0.0.1:5173`.

#### 3. Run Test Verification Suites
```bash
# Phase 1 Earth Engine test
python test_ee_pipeline.py

# Phase 2 Biomass pipeline test
python test_biomass.py

# Phase 3 Tree detection test
python test_tree_detection.py
```

---

### Complete API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status and GCP project check |
| `GET` | `/api/v1/status` | System status, demo mode flag, and phase capability report |
| `GET` | `/api/v1/study-area` | Bandipur National Park boundary coordinates & metadata |
| `GET` | `/api/v1/features` | Full descriptions of 16 multi-sensor feature stack bands |
| `GET` | `/api/v1/biomass` | Area-wide biomass model predictions and statistics |
| `GET` | `/api/v1/biomass/model` | Model hyperparameters and feature importance weights |
| `GET` | `/api/v1/biomass/statistics` | Summary stats (min, mean, max, std Mg/ha) for Bandipur |
| `POST` | `/api/v1/analysis/biomass` | On-demand biomass estimation for arbitrary user-drawn AOI polygons |
| `GET` | `/api/v1/trees` | DeepForest tree detection inference on high-res GeoTIFF |
| `POST` | `/api/v1/trees/detect` | Alias for tree detection with confidence threshold filter |
| `GET` | `/api/v1/trees/demo-tile` | Metadata for NEON AOP proxy demonstration tile |

   DEMO VIDEO:https://drive.google.com/drive/folders/1gEUh1wkOrLl1D27mwTpVXPslSFHkbRIR?usp=sharing
