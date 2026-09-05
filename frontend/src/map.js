import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import '@geoman-io/leaflet-geoman-free';
import '@geoman-io/leaflet-geoman-free/dist/leaflet-geoman.css';
import * as turf from '@turf/turf';

// Fix Leaflet's default marker icon URLs broken in Vite builds
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

// Location-Neutral Initial Center (Global Overview)
export const GLOBAL_CENTER = [20.0, 0.0];
export const GLOBAL_DEFAULT_ZOOM = 3;

// NEON Proxy Tile (Ordway-Swisher Biological Station, Florida, USA)
export const NEON_PROXY_CENTER = [29.6893, -81.9959];

export class ForestMap {
  constructor(containerId, options = {}) {
    this.containerId = containerId;
    this.options = options;
    this.map = null;
    this.drawnLayer = null;
    this.activeEELayer = null;
    this.treeLayerGroup = null;
    this.searchMarker = null;
    this.currentMode = 'biomass'; // 'biomass' | 'trees'
    this.onAOIChangeCallback = options.onAOIChange || (() => {});

    this.initMap();
  }

  initMap() {
    this.map = L.map(this.containerId, {
      center: GLOBAL_CENTER,
      zoom: GLOBAL_DEFAULT_ZOOM,
      zoomControl: true,
      attributionControl: true
    });

    // 1. OpenStreetMap Tile Layer (No API Key Required)
    const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(this.map);

    // 2. Satellite Tile Layer option (Esri World Imagery, open access)
    const esriSatellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 19,
      attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and GIS User Community'
    });

    // Layer Switcher
    const baseMaps = {
      '🗺️ OpenStreetMap': osmLayer,
      '🛰️ Satellite (Esri)': esriSatellite
    };
    L.control.layers(baseMaps, null, { position: 'bottomleft' }).addTo(this.map);
    L.control.scale({ imperial: false, metric: true, position: 'bottomleft' }).addTo(this.map);

    // 3. Initialize Drawing Controls (Geoman)
    this.initDrawingControls();

    // 4. Setup Tree Layer Group
    this.treeLayerGroup = L.featureGroup().addTo(this.map);
  }

  initDrawingControls() {
    this.map.pm.addControls({
      position: 'topleft',
      drawMarker: false,
      drawCircleMarker: false,
      drawPolyline: false,
      drawRectangle: true,
      drawPolygon: true,
      drawCircle: false,
      drawText: false,
      editMode: true,
      dragMode: true,
      cutPolygon: false,
      removalMode: true,
    });

    // Customize drawing styles
    this.map.pm.setPathOptions({
      color: '#f59e0b',
      fillColor: '#f59e0b',
      fillOpacity: 0.25,
      weight: 3,
      dashArray: '3, 3'
    });

    // Drawing lifecycle events
    this.map.on('pm:create', (e) => {
      if (this.drawnLayer && this.drawnLayer !== e.layer) {
        this.map.removeLayer(this.drawnLayer);
      }
      this.drawnLayer = e.layer;
      this.updateDrawnAOI(this.drawnLayer);

      this.drawnLayer.on('pm:edit', () => this.updateDrawnAOI(this.drawnLayer));
      this.drawnLayer.on('pm:dragend', () => this.updateDrawnAOI(this.drawnLayer));
    });

    this.map.on('pm:remove', (e) => {
      if (this.drawnLayer === e.layer) {
        this.drawnLayer = null;
        this.handleDrawDelete();
      }
    });
  }

  updateDrawnAOI(layer) {
    if (!layer) return;
    const geojson = layer.toGeoJSON();
    
    if (geojson && geojson.geometry && geojson.geometry.type === 'Polygon') {
      const areaSqMeters = turf.area(geojson);
      const areaHa = areaSqMeters / 10000;
      const areaSqKm = areaSqMeters / 1000000;
      const center = turf.center(geojson).geometry.coordinates;

      this.onAOIChangeCallback({
        type: 'custom',
        geometry: geojson.geometry,
        areaHa: areaHa.toFixed(2),
        areaSqKm: areaSqKm.toFixed(2),
        center: [center[0].toFixed(4), center[1].toFixed(4)]
      });
    }
  }

  handleDrawDelete() {
    this.onAOIChangeCallback({
      type: 'none',
      geometry: null,
      areaHa: '—',
      areaSqKm: '—',
      center: ['—', '—']
    });
  }

  startDrawPolygon() {
    if (this.drawnLayer) {
      this.map.removeLayer(this.drawnLayer);
      this.drawnLayer = null;
    }
    this.map.pm.enableDraw('Polygon', {
      snappable: true,
      snapDistance: 20,
    });
  }

  toggleEditMode() {
    this.map.pm.toggleGlobalEditMode();
  }

  clearDrawnAOI() {
    if (this.drawnLayer) {
      this.map.removeLayer(this.drawnLayer);
      this.drawnLayer = null;
    }
    this.map.pm.disableDraw();
    this.map.pm.disableGlobalEditMode();
    this.handleDrawDelete();
  }

  // -------------------------------------------------------------------------
  // Location Search & Navigation (Pure OpenStreetMap Geocoding)
  // -------------------------------------------------------------------------
  async searchLocation(query) {
    if (!query || !query.trim()) return { success: false, message: 'Please enter a search query' };

    try {
      const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query.trim())}&limit=1`;
      const res = await fetch(url, {
        headers: {
          'Accept-Language': 'en',
          'User-Agent': 'ForestAI-BiomassPlatform/2.0'
        }
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (data && data.length > 0) {
        const item = data[0];
        const lat = parseFloat(item.lat);
        const lon = parseFloat(item.lon);
        const displayName = item.display_name;

        this.navigateToLocation(lat, lon, 12, displayName);
        return { success: true, name: displayName, source: 'nominatim' };
      } else {
        return { success: false, message: `No location found for "${query}". Please check spelling.` };
      }
    } catch (err) {
      console.warn('Geocoding fetch failed:', err);
      return { 
        success: false, 
        message: `Search error (${err.message}). Try entering coordinates or another query.` 
      };
    }
  }

  navigateToLocation(lat, lon, zoom = 12, title = '') {
    this.map.flyTo([lat, lon], zoom, { duration: 1.5 });

    if (this.searchMarker) {
      this.map.removeLayer(this.searchMarker);
    }

    this.searchMarker = L.marker([lat, lon])
      .addTo(this.map)
      .bindPopup(`<strong>📍 ${title}</strong><br>Coordinates: ${lat.toFixed(4)}°N, ${lon.toFixed(4)}°E`)
      .openPopup();
  }

  // -------------------------------------------------------------------------
  // Dynamic Earth Engine Tile Layers
  // -------------------------------------------------------------------------
  setEETileLayer(layerName, tileUrl) {
    this.removeEETileLayer();
    if (!tileUrl) return;

    this.activeEELayer = L.tileLayer(tileUrl, {
      maxZoom: 20,
      opacity: 0.85,
      attribution: 'Google Earth Engine &copy; Copernicus / ETH'
    }).addTo(this.map);
  }

  removeEETileLayer() {
    if (this.activeEELayer) {
      this.map.removeLayer(this.activeEELayer);
      this.activeEELayer = null;
    }
  }

  // -------------------------------------------------------------------------
  // Tree Detection Layer
  // -------------------------------------------------------------------------
  setTreeDetectionsGeoJSON(geojson, sourceLabel = 'DeepForest RetinaNet') {
    this.treeLayerGroup.clearLayers();
    if (!geojson || !geojson.features || geojson.features.length === 0) return;

    const bounds = L.latLngBounds([]);

    geojson.features.forEach((feat, index) => {
      const coords = feat.geometry.coordinates; // [lon, lat]
      const props = feat.properties || {};
      const score = Number(props.confidence || props.score || 0.65).toFixed(3);
      const radiusM = Number(props.radius_m || 2.1).toFixed(1);
      const areaM2 = (Math.PI * Math.pow(parseFloat(radiusM), 2)).toFixed(1);

      const latlng = [coords[1], coords[0]];
      bounds.extend(latlng);

      const circle = L.circle(latlng, {
        radius: Math.max(1.0, parseFloat(radiusM) * 1.2),
        color: '#22d3ee',
        weight: 1.8,
        fillColor: '#06b6d4',
        fillOpacity: 0.4
      });

      circle.bindPopup(`
        <div style="font-family: var(--font-sans); font-size: 0.8rem;">
          <div style="font-weight: 700; color: #06b6d4; margin-bottom: 4px;">🌲 DeepForest Tree Crown #${index + 1}</div>
          <div>Confidence: <b>${score}</b></div>
          <div>Crown Radius: <b>${radiusM} m</b> (~${areaM2} m²)</div>
          <div>Coordinates: <b>${coords[1].toFixed(6)}°N, ${coords[0].toFixed(6)}°E</b></div>
          <div style="font-size: 0.7rem; color: #9ca3af; margin-top: 4px;">Source: ${sourceLabel}</div>
        </div>
      `);

      this.treeLayerGroup.addLayer(circle);
    });

    if (bounds.isValid()) {
      this.map.fitBounds(bounds, { maxZoom: 18, padding: [30, 30] });
    }
  }

  // -------------------------------------------------------------------------
  // Mode Management & Quick Views
  // -------------------------------------------------------------------------
  setMode(mode) {
    this.currentMode = mode;
    if (mode === 'biomass') {
      if (this.treeLayerGroup) this.map.removeLayer(this.treeLayerGroup);
      if (this.activeEELayer) this.map.addLayer(this.activeEELayer);
    } else if (mode === 'trees') {
      if (this.activeEELayer) this.map.removeLayer(this.activeEELayer);
      if (this.treeLayerGroup) this.map.addLayer(this.treeLayerGroup);
    }
  }

  flyToNeon() {
    this.map.flyTo(NEON_PROXY_CENTER, 18, { duration: 2.0 });
  }

  resetView() {
    if (this.drawnLayer) {
      const bounds = this.drawnLayer.getBounds();
      this.map.fitBounds(bounds, { padding: [40, 40] });
    } else {
      this.map.flyTo(GLOBAL_CENTER, GLOBAL_DEFAULT_ZOOM, { duration: 1.5 });
    }
  }
}
