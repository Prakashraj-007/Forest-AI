/**
 * Forest AI Dashboard — Main Coordinator (Leaflet + OpenStreetMap)
 */

import './style.css';
import { ForestMap, NEON_TILE_BOUNDS } from './map.js';
import { DashboardPanels } from './panels.js';
import {
  fetchStatus,
  fetchAOIBiomass,
  fetchBiomassPreset,
  fetchTreeDetection,
  uploadTreeDetection,
  fetchEETileUrl
} from './api.js';

class ForestApp {
  constructor() {
    this.currentMode = 'biomass';
    this.currentAOI = {
      type: 'none',
      geometry: null,
      areaHa: '—',
      areaSqKm: '—',
      center: ['—', '—']
    };
    this.selectedTreeFile = null;
    this.currentEELayer = 'none';   // Biomass panel EE layer
    this.neonEELayer = 'none';       // NEON proxy panel EE layer (independent)
    this.lastNeonDetectionsGeoJSON = null; // Latest proxy detections for polygon stats

    this.panels = new DashboardPanels();
    this.map = new ForestMap('leaflet-map', {
      onAOIChange: (aoiData) => this.handleAOIChange(aoiData),
      onNeonPolygonChange: (polygonGeoJSON) => this.handleNeonPolygonChange(polygonGeoJSON)
    });

    this.initEventListeners();
    this.checkSystemStatus();
  }

  async checkSystemStatus() {
    try {
      const status = await fetchStatus();
      this.panels.setBackendStatus(status);
    } catch (err) {
      console.warn('Backend status check error:', err);
      this.panels.setBackendStatus({ backend: 'offline', demo_mode: true });
    }
  }

  handleAOIChange(aoiData) {
    this.currentAOI = aoiData;
    this.panels.updateAOIInfo(aoiData);

    // If an EE layer is active, refresh the tile with the new AOI
    if (this.currentEELayer && this.currentEELayer !== 'none') {
      this.loadEELayer(this.currentEELayer);
    }
  }

  initEventListeners() {
    // 1. Top Mode Switcher Tabs
    const btnModeBiomass = document.getElementById('mode-biomass-btn');
    const btnModeTrees = document.getElementById('mode-trees-btn');
    const panelBiomass = document.getElementById('panel-biomass');
    const panelTrees = document.getElementById('panel-trees');

    btnModeBiomass.addEventListener('click', () => {
      this.currentMode = 'biomass';
      btnModeBiomass.classList.add('active');
      btnModeTrees.classList.remove('active');
      panelBiomass.classList.add('active');
      panelTrees.classList.remove('active');
      this.map.setMode('biomass');
      this.panels.updateLegendForLayer(this.currentEELayer);
    });

    btnModeTrees.addEventListener('click', () => {
      this.currentMode = 'trees';
      btnModeTrees.classList.add('active');
      btnModeBiomass.classList.remove('active');
      panelTrees.classList.add('active');
      panelBiomass.classList.remove('active');
      this.map.setMode('trees');
      this.panels.updateLegendForLayer('trees');
    });

    // 2. Search Navigation
    const searchInput = document.getElementById('location-search-input');
    const btnSearchGo = document.getElementById('btn-search-go');

    const handleSearch = async () => {
      const query = searchInput.value;
      if (!query.trim()) return;
      this.panels.showSearchFeedback('Searching OpenStreetMap location...', true);
      const res = await this.map.searchLocation(query);
      if (res.success) {
        this.panels.showSearchFeedback(`📍 Navigated to: ${res.name}`, true);
      } else {
        this.panels.showSearchFeedback(res.message, false);
      }
    };

    btnSearchGo.addEventListener('click', handleSearch);
    searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleSearch();
      }
    });

    // 3. AOI Draw & Edit Buttons
    const btnDrawPolygon = document.getElementById('btn-draw-polygon');
    const btnEditPolygon = document.getElementById('btn-edit-polygon');
    const btnClearDraw = document.getElementById('btn-clear-draw');

    btnDrawPolygon.addEventListener('click', () => {
      this.map.startDrawPolygon();
    });

    btnEditPolygon.addEventListener('click', () => {
      this.map.toggleEditMode();
    });

    btnClearDraw.addEventListener('click', () => {
      this.map.clearDrawnAOI();
      this.map.removeEETileLayer();
    });

    // 4. Dynamic Earth Engine Layers Radio Switcher
    const eeLayerRadios = document.querySelectorAll('input[name="ee-layer-select"]');
    eeLayerRadios.forEach(radio => {
      radio.addEventListener('change', (e) => {
        this.loadEELayer(e.target.value);
      });
    });

    // 5. Run Biomass Estimation Button
    const btnRunBiomass = document.getElementById('btn-run-biomass');
    btnRunBiomass.addEventListener('click', () => this.runBiomassAnalysis());

    // 6. Tree Detection Sub-Tabs (Upload, Provider, Proxy Demo)
    const tabUpload = document.getElementById('tab-path-upload');
    const tabProvider = document.getElementById('tab-path-provider');
    const tabProxy = document.getElementById('tab-path-proxy');
    const sectionUpload = document.getElementById('section-path-upload');
    const sectionProvider = document.getElementById('section-path-provider');
    const sectionProxy = document.getElementById('section-path-proxy');

    const switchTreeTab = (activeTab, activeSection) => {
      [tabUpload, tabProvider, tabProxy].forEach(t => t.classList.remove('active'));
      [sectionUpload, sectionProvider, sectionProxy].forEach(s => s.classList.remove('active'));
      activeTab.classList.add('active');
      activeSection.classList.add('active');
    };

    tabUpload.addEventListener('click', () => switchTreeTab(tabUpload, sectionUpload));
    tabProvider.addEventListener('click', () => switchTreeTab(tabProvider, sectionProvider));
    tabProxy.addEventListener('click', () => switchTreeTab(tabProxy, sectionProxy));

    // 7. Tree Detection - Path A (GeoTIFF Upload)
    const uploadDropzone = document.getElementById('upload-dropzone');
    const fileInput = document.getElementById('geotiff-file-input');
    const selectedFileName = document.getElementById('selected-file-name');
    const btnUploadDetect = document.getElementById('btn-upload-detect');
    const uploadScoreSlider = document.getElementById('upload-score-thresh');
    const uploadScoreValLabel = document.getElementById('upload-score-thresh-val');

    uploadDropzone.addEventListener('click', () => fileInput.click());
    uploadDropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadDropzone.classList.add('dragover');
    });
    uploadDropzone.addEventListener('dragleave', () => {
      uploadDropzone.classList.remove('dragover');
    });
    uploadDropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadDropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        this.handleFileSelected(e.dataTransfer.files[0], selectedFileName, btnUploadDetect);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        this.handleFileSelected(e.target.files[0], selectedFileName, btnUploadDetect);
      }
    });

    uploadScoreSlider.addEventListener('input', (e) => {
      uploadScoreValLabel.textContent = Number(e.target.value).toFixed(2);
    });

    btnUploadDetect.addEventListener('click', () => this.runUploadedTreeDetection());

    // Layer View Mode Switcher: Original Image, Blue Crowns, or Both
    document.querySelectorAll('.btn-view-mode, .map-view-toggle-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const mode = btn.getAttribute('data-view-mode') || 'both';
        this.setViewMode(mode);
      });
    });

    // 8. Tree Detection - Path B (Commercial Provider Search)
    const btnSearchProvider = document.getElementById('btn-search-provider');
    const providerApiKey = document.getElementById('provider-api-key');
    const providerSelect = document.getElementById('provider-select');
    const providerSearchResult = document.getElementById('provider-search-result');

    btnSearchProvider.addEventListener('click', () => {
      const key = providerApiKey.value.trim();
      const provider = providerSelect.value;
      providerSearchResult.classList.remove('hidden');

      if (!key) {
        providerSearchResult.innerHTML = `⚠️ <strong>Missing Credentials:</strong> Please enter your ${provider.toUpperCase()} API key to search commercial catalog.`;
        return;
      }

      if (this.currentAOI.type === 'none') {
        providerSearchResult.innerHTML = `⚠️ <strong>No AOI Selected:</strong> Please search or draw an Area of Interest on the map before querying ${provider.toUpperCase()} imagery.`;
        return;
      }

      providerSearchResult.innerHTML = `🛰️ <strong>Querying ${provider.toUpperCase()} Catalog:</strong> Searching recent sub-meter ortho-imagery for AOI (${this.currentAOI.areaHa} ha)...<br><span style="color: var(--text-muted); font-size: 0.7rem; margin-top: 4px; display: block;">No commercial imagery subscription active for this key. Tree crown detection requires sub-meter imagery (< 1m/px) to detect individual trees.</span>`;
    });

    // 9. Tree Detection - Path C (NEON Florida Proxy Demo)
    const treeSlider = document.getElementById('tree-score-thresh');
    const scoreValLabel = document.getElementById('score-thresh-val');
    treeSlider.addEventListener('input', (e) => {
      scoreValLabel.textContent = Number(e.target.value).toFixed(2);
    });

    const btnRunTrees = document.getElementById('btn-run-tree-detection');
    btnRunTrees.addEventListener('click', () => this.runProxyTreeDetection());

    // 10. NEON Context Satellite Layer Picker (proxy panel — independent of Biomass EE)
    const neonLayerRadios = document.querySelectorAll('input[name="neon-ee-layer-select"]');
    neonLayerRadios.forEach(radio => {
      radio.addEventListener('change', (e) => {
        this.loadNeonEELayer(e.target.value);
      });
    });

    // 10b. Upload Area Earth Engine Satellite Context (Sentinel-1 & Sentinel-2)
    const uploadEELayerRadios = document.querySelectorAll('input[name="upload-ee-layer-select"]');
    uploadEELayerRadios.forEach(radio => {
      radio.addEventListener('change', (e) => {
        this.loadUploadEELayer(e.target.value);
      });
    });

    // 11. NEON Polygon Draw & Clear
    const btnDrawNeonPolygon = document.getElementById('btn-draw-neon-polygon');
    const btnClearNeonPolygon = document.getElementById('btn-clear-neon-polygon');

    btnDrawNeonPolygon.addEventListener('click', () => {
      this.map.flyToNeon();
      // Show the site boundary if not already visible
      this.map.showNeonSiteBoundary();
      // Short delay so the map settles before activating draw mode
      setTimeout(() => this.map.startNeonPolygonDraw(), 500);
      btnDrawNeonPolygon.textContent = '✏️ Drawing… click inside the amber boundary';
      btnDrawNeonPolygon.disabled = true;
      setTimeout(() => {
        btnDrawNeonPolygon.textContent = '✏️ Draw Analysis Polygon';
        btnDrawNeonPolygon.disabled = false;
      }, 15000);
    });

    btnClearNeonPolygon.addEventListener('click', () => {
      this.map.clearNeonPolygon();
      const totalCount = this.lastNeonDetectionsGeoJSON ? (this.lastNeonDetectionsGeoJSON.features || []).length : 0;
      this.panels.clearNeonPolygonStats(totalCount);
      btnClearNeonPolygon.classList.add('hidden');
      document.getElementById('neon-polygon-badge').textContent = 'No Polygon';
    });

    // 10. Floating Map Controls
    document.getElementById('btn-fly-neon').addEventListener('click', () => {
      this.map.flyToNeon();
    });

    document.getElementById('btn-reset-view').addEventListener('click', () => {
      this.map.resetView();
      const mapBar = document.getElementById('map-overlay-view-bar');
      if (mapBar) mapBar.style.display = 'none';
    });

    // 11. Toggle Tree Crown Visibility
    const chkShowBoxes = document.getElementById('chk-show-boxes');
    if (chkShowBoxes) {
      chkShowBoxes.addEventListener('change', (e) => {
        if (e.target.checked) {
          this.map.map.addLayer(this.map.treeLayerGroup);
        } else {
          this.map.map.removeLayer(this.map.treeLayerGroup);
        }
      });
    }
  }

  handleFileSelected(file, nameElement, submitBtn) {
    const accepted = /\.(tif|tiff|geotiff|png|jpg|jpeg|webp|bmp)$/i;
    if (!accepted.test(file.name)) {
      alert('Unsupported format. Please select a GeoTIFF, PNG, JPG, WEBP, or BMP image.');
      return;
    }
    this.selectedTreeFile = file;
    nameElement.textContent = `📄 ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
    nameElement.classList.remove('hidden');
    submitBtn.removeAttribute('disabled');

    // Clear old image overlay when a new file is chosen
    this.map.clearImageOverlay();
  }

  // -------------------------------------------------------------------------
  // NEON EE & High-Res Context Layers (proxy panel)
  // -------------------------------------------------------------------------

  /**
   * Load and display a NEON satellite or high-res aerial context layer.
   *
   * @param {string} layer            — Layer key ('aerial_highres', 's2_rgb', 's2_ndvi', etc.)
   * @param {Object} [polygonGeoJSON] — Optional GeoJSON Polygon for clipping
   */
  async loadNeonEELayer(layer, polygonGeoJSON = null) {
    this.neonEELayer = layer;
    const statusEl = document.getElementById('neon-ee-layer-status');

    if (layer === 'none') {
      this.map.removeNeonEETileLayer();
      this.map.removeNeonAerialLayer();
      if (statusEl) statusEl.classList.add('hidden');
      return;
    }

    if (layer === 'aerial_highres') {
      this.map.removeNeonEETileLayer();
      this.map.setNeonAerialLayer();
      this.map.flyToNeon();
      if (statusEl) {
        statusEl.textContent = '📸 NEON Aerial RGB (0.10m/px) — High-Resolution Orthophoto';
        statusEl.className = 'ee-layer-status';
        statusEl.classList.remove('hidden');
      }
      return;
    }

    // Otherwise, it is an Earth Engine context layer
    this.map.removeNeonAerialLayer();

    // Determine geometry: prefer polygon, active polygon layer, or NEON footprint
    let eeGeometry = null;
    if (polygonGeoJSON) {
      const g = polygonGeoJSON.type === 'Feature' ? polygonGeoJSON.geometry : polygonGeoJSON;
      eeGeometry = g;
    } else if (this.map.neonPolygonLayer) {
      const existing = this.map.neonPolygonLayer.toGeoJSON();
      eeGeometry = existing.type === 'Feature' ? existing.geometry : existing;
    } else {
      eeGeometry = {
        type: 'Polygon',
        coordinates: [[
          [-81.9900959, 29.6923218],
          [-81.9896825, 29.6923249],
          [-81.9896860, 29.6926859],
          [-81.9900994, 29.6926828],
          [-81.9900959, 29.6923218]
        ]]
      };
    }

    if (statusEl) {
      statusEl.textContent = `Loading ${layer} satellite context...`;
      statusEl.className = 'ee-layer-status loading';
      statusEl.classList.remove('hidden');
    }

    try {
      const res = await fetchEETileUrl(layer, eeGeometry);
      if (res.success && res.tile_url) {
        this.map.setNeonEETileLayer(res.tile_url, eeGeometry);
        if (statusEl) {
          statusEl.textContent = `🛰️ ${layer.toUpperCase()} Context Layer Active`;
          statusEl.className = 'ee-layer-status';
        }
      } else {
        if (statusEl) {
          statusEl.textContent = `Notice: ${res.message || res.error || 'Layer unavailable'}`;
          statusEl.className = 'ee-layer-status';
        }
      }
    } catch (err) {
      console.warn('NEON EE layer fetch error:', err);
      if (statusEl) {
        statusEl.textContent = `Layer notice: ${err.message}`;
        statusEl.className = 'ee-layer-status';
      }
    }
  }

  // -------------------------------------------------------------------------
  // Upload Area Earth Engine Layer (Sentinel-1 / Sentinel-2 / Canopy Height)
  // -------------------------------------------------------------------------
  async loadUploadEELayer(layer) {
    const statusEl = document.getElementById('upload-ee-layer-status');
    if (layer === 'none') {
      this.map.removeEETileLayer();
      if (statusEl) statusEl.classList.add('hidden');
      this.panels.updateLegendForLayer('biomass');
      return;
    }

    // Determine geometry from uploaded image bounds or current AOI
    let eeGeometry = null;
    if (this.lastUploadedBounds && this.lastUploadedBounds.length === 4) {
      const [w, s, e, n] = this.lastUploadedBounds;
      eeGeometry = {
        type: 'Polygon',
        coordinates: [[
          [w, s],
          [e, s],
          [e, n],
          [w, n],
          [w, s]
        ]]
      };
    } else if (this.currentAOI && this.currentAOI.geometry) {
      eeGeometry = this.currentAOI.geometry;
    }

    if (statusEl) {
      statusEl.textContent = `Fetching Copernicus ${layer.toUpperCase()} Earth Engine layer...`;
      statusEl.className = 'ee-layer-status loading';
      statusEl.classList.remove('hidden');
    }

    try {
      const res = await fetchEETileUrl(layer, eeGeometry);
      if (res.success && res.tile_url) {
        this.map.setEETileLayer(layer, res.tile_url);
        this.panels.updateLegendForLayer(layer);
        if (statusEl) {
          statusEl.textContent = `🛰️ ${layer.toUpperCase()} Layer Active`;
          statusEl.className = 'ee-layer-status';
        }
      } else {
        if (statusEl) {
          statusEl.textContent = `Notice: ${res.message || res.error || 'Layer unavailable'}`;
          statusEl.className = 'ee-layer-status';
        }
      }
    } catch (err) {
      console.warn('Upload EE layer fetch error:', err);
      if (statusEl) {
        statusEl.textContent = `Layer notice: ${err.message}`;
        statusEl.className = 'ee-layer-status';
      }
    }
  }

  // -------------------------------------------------------------------------
  // NEON Polygon Change Handler
  // -------------------------------------------------------------------------
  handleNeonPolygonChange(polygonGeoJSON) {
    const btnClearNeonPolygon = document.getElementById('btn-clear-neon-polygon');
    const badgeEl = document.getElementById('neon-polygon-badge');

    if (!polygonGeoJSON) {
      this.map.highlightTreesInPolygon(null);
      this.map.removeNeonEEMask();
      const totalCount = this.lastNeonDetectionsGeoJSON ? (this.lastNeonDetectionsGeoJSON.features || []).length : 0;
      this.panels.clearNeonPolygonStats(totalCount);
      if (btnClearNeonPolygon) btnClearNeonPolygon.classList.add('hidden');
      if (badgeEl) badgeEl.textContent = 'No Polygon';
      return;
    }

    // Polygon exists — show clear button
    if (btnClearNeonPolygon) btnClearNeonPolygon.classList.remove('hidden');
    if (badgeEl) badgeEl.textContent = 'Polygon Active';

    // Highlight tree points on map
    this.map.highlightTreesInPolygon(polygonGeoJSON);

    // If an EE satellite layer is active (and not aerial), update clip
    if (this.neonEELayer && this.neonEELayer !== 'none' && this.neonEELayer !== 'aerial_highres') {
      this.map.createNeonEEMask(polygonGeoJSON);
    }

    // Compute tree stats inside the polygon from existing detections
    this.updateNeonPolygonStats(polygonGeoJSON);
  }

  /**
   * Compute and render polygon statistics from the latest NEON detections.
   * Called whenever the polygon changes OR a new detection run completes.
   */
  updateNeonPolygonStats(polygonGeoJSON) {
    if (!polygonGeoJSON && this.map.neonPolygonLayer) {
      polygonGeoJSON = this.map.neonPolygonLayer.toGeoJSON();
    }
    if (!polygonGeoJSON) return;

    const detectionsGeoJSON = this.lastNeonDetectionsGeoJSON;
    const totalDetectionsCount = detectionsGeoJSON && detectionsGeoJSON.features ? detectionsGeoJSON.features.length : 0;
    const noDetections = totalDetectionsCount === 0;

    const stats = this.map.computeNeonPolygonStats(polygonGeoJSON, detectionsGeoJSON || { type: 'FeatureCollection', features: [] });
    this.panels.renderNeonPolygonStats(stats, noDetections, totalDetectionsCount);
    this.map.highlightTreesInPolygon(polygonGeoJSON);
  }

  // -------------------------------------------------------------------------
  // EE Layer (Biomass panel)
  // -------------------------------------------------------------------------
  async loadEELayer(layer) {
    this.currentEELayer = layer;
    if (layer === 'none') {
      this.map.removeEETileLayer();
      this.panels.setEELayerStatus(null);
      this.panels.updateLegendForLayer('biomass');
      return;
    }

    try {
      this.panels.setEELayerStatus(`Fetching Earth Engine layer: ${layer}...`, true);
      const res = await fetchEETileUrl(layer, this.currentAOI.geometry);
      
      if (res.success && res.tile_url) {
        this.map.setEETileLayer(layer, res.tile_url);
        this.panels.setEELayerStatus(`Active EE Layer: ${layer}`);
        this.panels.updateLegendForLayer(layer);
      } else {
        this.panels.setEELayerStatus(`Notice: ${res.message || res.error || 'Layer unavailable'}`, false);
      }
    } catch (err) {
      console.warn('EE layer fetch error:', err);
      this.panels.setEELayerStatus(`Layer notice: ${err.message}`, false);
    }
  }

  async runBiomassAnalysis() {
    if (this.currentAOI.type === 'none' || !this.currentAOI.geometry) {
      alert('Please draw an Area of Interest (AOI) polygon on the map or search for a location first.');
      return;
    }

    let isLoading = true;
    let timer1 = null;
    let timer2 = null;

    try {
      this.panels.resetBiomassResultsUI();
      this.panels.setBiomassLoading(true, 1);
      timer1 = setTimeout(() => {
        if (isLoading) this.panels.setBiomassLoading(true, 2);
      }, 700);
      timer2 = setTimeout(() => {
        if (isLoading) this.panels.setBiomassLoading(true, 3);
      }, 1800);

      const data = await fetchAOIBiomass(this.currentAOI.geometry);
      isLoading = false;
      clearTimeout(timer1);
      clearTimeout(timer2);
      this.panels.setBiomassLoading(false);
      this.panels.renderBiomassResults(data);

      // Auto-load biomass overlay if practical
      if (this.currentEELayer && this.currentEELayer !== 'none') {
        this.loadEELayer(this.currentEELayer);
      }
    } catch (err) {
      isLoading = false;
      clearTimeout(timer1);
      clearTimeout(timer2);
      this.panels.setBiomassLoading(false);
      console.error('Biomass analysis error:', err);
      alert(`Biomass Analysis Notice: ${err.message}`);
    } finally {
      isLoading = false;
      clearTimeout(timer1);
      clearTimeout(timer2);
      this.panels.setBiomassLoading(false);
    }
  }


  async runUploadedTreeDetection() {
    if (!this.selectedTreeFile) {
      alert('Please choose an aerial image or GeoTIFF file first.');
      return;
    }

    const scoreThresh = parseFloat(document.getElementById('upload-score-thresh').value || 0.15);

    // Use map CENTER ± a sensible small area (~0.005° ≈ 500m radius) so the
    // uploaded image is placed at a real-world location near where the user is
    // looking, at a zoom level where individual trees are visible.
    // For true GeoTIFFs (with embedded CRS) the backend will use real coords.
    const [centerLon, centerLat] = this.map.getMapCenter();
    const halfDeg = 0.005; // ~500m at most latitudes — sensible drone/aerial coverage
    const localBbox = [
      centerLon - halfDeg,
      centerLat - halfDeg,
      centerLon + halfDeg,
      centerLat + halfDeg
    ];

    try {
      this.panels.setTreeLoading(true, 'Running DeepForest Tree Crown Detection...', 'Detecting individual tree crowns in uploaded imagery');

      // Clear previous uploaded image overlay before running new detection
      this.map.clearImageOverlay();

      const data = await uploadTreeDetection(this.selectedTreeFile, scoreThresh, this.currentAOI.geometry, localBbox);
      this.panels.setTreeLoading(false);

      this.panels.renderTreeResults(data, `Uploaded Image: ${this.selectedTreeFile.name}`);

      const detGeoJSON = data.geojson || data.detections_geojson;

      // Determine bounding box for image overlay placement.
      // For true GeoTIFFs (embedded CRS): use real geographic bounds from response.
      // For non-georeferenced images: use localBbox (same bbox backend used for scaling).
      const imageBounds = (data.is_georeferenced && data.image_bounds_wgs84)
        ? data.image_bounds_wgs84
        : localBbox;

      this.lastUploadedBounds = imageBounds;

      // Use the backend-generated JPEG data URL — the only format that browsers can
      // render for TIFF uploads (browsers do not support TIFF natively).
      // For PNG/JPG uploads, also prefer the backend version to guarantee alignment.
      const imageDataUrl = data.image_data_url;

      if (imageDataUrl) {
        // Show the image on the map with canvas circles drawn precisely on it
        this.map.showUploadedImageOverlay(imageDataUrl, imageBounds, detGeoJSON);
      } else {
        // Fallback: try reading the file directly (works for PNG/JPG, not TIFF)
        const reader = new FileReader();
        reader.onload = (e) => {
          this.map.showUploadedImageOverlay(e.target.result, imageBounds, detGeoJSON);
        };
        reader.readAsDataURL(this.selectedTreeFile);
      }

      // Also add invisible Leaflet circles for click-to-popup interaction
      if (detGeoJSON) {
        this.map.setTreeDetectionsGeoJSON(detGeoJSON, `Upload: ${this.selectedTreeFile.name}`, true);
      }

      // Show floating view switcher and default to 'both'
      const mapBar = document.getElementById('map-overlay-view-bar');
      if (mapBar) mapBar.style.display = 'flex';
      this.setViewMode('both');
    } catch (err) {
      console.error('Upload tree detection error:', err);
      this.panels.setTreeLoading(false);
      alert(`Tree Detection Error: ${err.message}`);
    }
  }

  /**
   * Switch active display mode: 'both' (Image + Crowns), 'image' (Original Image), or 'crowns' (Blue Crowns)
   * @param {'both'|'image'|'crowns'} mode
   */
  setViewMode(mode = 'both') {
    this.map.setOverlayDisplayMode(mode);

    // Sync sidebar buttons in Card 02
    document.querySelectorAll('.btn-view-mode').forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-view-mode') === mode);
    });

    // Sync floating toolbar on the map
    document.querySelectorAll('.map-view-toggle-btn').forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-view-mode') === mode);
    });
  }

  async runProxyTreeDetection() {
    const scoreThresh = parseFloat(document.getElementById('tree-score-thresh').value || 0.15);

    try {
      this.panels.setTreeLoading(true, 'Running DeepForest on NEON Florida Proxy Tile...', 'Detecting tree crowns in 0.10m/px aerial RGB');
      const data = await fetchTreeDetection(scoreThresh);
      this.panels.setTreeLoading(false);

      this.panels.renderTreeResults(data, 'NEON Florida Proxy Demo (0.10m AOP)');

      // Fly to the exact tile location and show the boundary
      this.map.flyToNeon();
      this.map.showNeonSiteBoundary();

      if (data.geojson || data.detections_geojson) {
        const detGeoJSON = data.geojson || data.detections_geojson;
        this.map.setTreeDetectionsGeoJSON(detGeoJSON, 'NEON AOP Florida Proxy Demo');
        // Store for polygon analysis
        this.lastNeonDetectionsGeoJSON = detGeoJSON;
        // Recalculate polygon stats if a polygon is already drawn
        this.updateNeonPolygonStats(null);
      } else {
        this.lastNeonDetectionsGeoJSON = null;
      }
    } catch (err) {
      console.error('Tree detection error:', err);
      this.panels.setTreeLoading(false);
      alert(`Tree Detection Notice: ${err.message}`);
    }
  }
}

// Initialize on DOM load
window.addEventListener('DOMContentLoaded', () => {
  window.app = new ForestApp();
});
