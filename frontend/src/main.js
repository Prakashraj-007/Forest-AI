/**
 * Forest AI Dashboard — Main Coordinator (Leaflet + OpenStreetMap)
 */

import './style.css';
import { ForestMap } from './map.js';
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
    this.currentEELayer = 'none';

    this.panels = new DashboardPanels();
    this.map = new ForestMap('leaflet-map', {
      onAOIChange: (aoiData) => this.handleAOIChange(aoiData)
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

    // 10. Floating Map Controls
    document.getElementById('btn-fly-neon').addEventListener('click', () => {
      this.map.flyToNeon();
    });

    document.getElementById('btn-reset-view').addEventListener('click', () => {
      this.map.resetView();
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
    if (!file.name.match(/\.(tif|tiff|geotiff)$/i)) {
      alert('Please select a valid GeoTIFF (.tif or .tiff) file.');
      return;
    }
    this.selectedTreeFile = file;
    nameElement.textContent = `📄 ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
    nameElement.classList.remove('hidden');
    submitBtn.removeAttribute('disabled');
  }

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
      alert('Please choose a high-resolution GeoTIFF file first.');
      return;
    }

    const scoreThresh = parseFloat(document.getElementById('upload-score-thresh').value || 0.15);

    try {
      this.panels.setTreeLoading(true, 'Running DeepForest on Uploaded GeoTIFF...', 'Parsing geospatial bounds and detecting crowns');
      const data = await uploadTreeDetection(this.selectedTreeFile, scoreThresh, this.currentAOI.geometry);
      this.panels.setTreeLoading(false);

      this.panels.renderTreeResults(data, `Uploaded GeoTIFF: ${this.selectedTreeFile.name}`);

      if (data.geojson || data.detections_geojson) {
        this.map.setTreeDetectionsGeoJSON(data.geojson || data.detections_geojson, `Upload: ${this.selectedTreeFile.name}`);
      }
    } catch (err) {
      console.error('Upload tree detection error:', err);
      this.panels.setTreeLoading(false);
      alert(`Tree Detection Error: ${err.message}`);
    }
  }

  async runProxyTreeDetection() {
    const scoreThresh = parseFloat(document.getElementById('tree-score-thresh').value || 0.15);

    try {
      this.panels.setTreeLoading(true, 'Running DeepForest on NEON Florida Proxy Tile...', 'Detecting tree crowns in 0.10m/px aerial RGB');
      const data = await fetchTreeDetection(scoreThresh);
      this.panels.setTreeLoading(false);

      this.panels.renderTreeResults(data, 'NEON Florida Proxy Demo (0.10m AOP)');

      this.map.flyToNeon();
      if (data.geojson || data.detections_geojson) {
        this.map.setTreeDetectionsGeoJSON(data.geojson || data.detections_geojson, 'NEON AOP Florida Proxy Demo');
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
