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

// NEON Proxy Tile — Ordway-Swisher Biological Station (OSBS), Florida, USA
// Source: NEON AOP sample tile OSBS_029.tif (CC0 1.0 Public Domain)
// CRS: EPSG:32617 (UTM Zone 17N)
// Tile: 400×400 px at 0.10 m/px = 40m × 40m = 0.16 ha
export const NEON_PROXY_CENTER = [29.6925, -81.9899];

/**
 * Exact WGS-84 boundary of the NEON OSBS_029.tif GeoTIFF.
 * [longitude, latitude] corners in clockwise order.
 * Converted from EPSG:32617 UTM bounds:
 *   left=404211.9, bottom=3285102.9, right=404251.9, top=3285142.9
 */
export const NEON_TILE_BOUNDS = {
  sw: [-81.9900959, 29.6923218],  // [lon, lat]
  ne: [-81.9896860, 29.6926859],
  nw: [-81.9900994, 29.6926828],
  se: [-81.9896825, 29.6923249],
  center: [-81.9898910, 29.6925038],
  // Leaflet-style ring: [lat, lon] clockwise
  ring: [
    [29.6926828, -81.9900994],  // NW
    [29.6926859, -81.9896860],  // NE
    [29.6923249, -81.9896825],  // SE
    [29.6923218, -81.9900959],  // SW
  ],
  areaHa: 0.16,
  widthM: 40,
  heightM: 40
};

export class ForestMap {
  constructor(containerId, options = {}) {
    this.containerId = containerId;
    this.options = options;
    this.map = null;
    this.drawnLayer = null;
    this.activeEELayer = null;        // Biomass panel EE layer
    this.neonEELayer = null;           // NEON proxy EE tile layer (independent)
    this._eeClipGeom = null;           // GeoJSON Polygon geometry used for CSS clip-path on neonEELayer
    this.neonSiteBoundaryLayer = null; // Fixed NEON GeoTIFF footprint boundary polygon
    this.neonPolygonLayer = null;      // User-drawn analysis polygon
    this.neonDetectionsGeoJSON = null; // Latest NEON proxy detection GeoJSON
    this.neonAerialOverlay = null;     // High-resolution NEON aerial image overlay
    this.uploadedImageOverlay = null;  // User-uploaded image shown as map overlay
    this.treeCircleLayers = [];        // Array of { circle, feat, index } for dynamic styling
    this.treeLayerGroup = null;
    this.searchMarker = null;
    this.currentMode = 'biomass'; // 'biomass' | 'trees'
    this.onAOIChangeCallback = options.onAOIChange || (() => {});
    this.onNeonPolygonChangeCallback = options.onNeonPolygonChange || (() => {});
    this._neonPolygonDrawing = false;

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

    // 5. Setup NEON analysis polygon layer group
    this.neonPolygonLayerGroup = L.featureGroup().addTo(this.map);
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

    // Drawing lifecycle events — route to NEON polygon or main AOI based on flag
    this.map.on('pm:create', (e) => {
      if (this._neonPolygonDrawing) {
        // ---- NEON analysis polygon ----
        this._neonPolygonDrawing = false;
        this.map.pm.disableDraw();

        // Style the NEON polygon distinctly (teal border)
        e.layer.setStyle({
          color: '#06b6d4',
          fillColor: '#06b6d4',
          fillOpacity: 0.12,
          weight: 2,
          dashArray: '4, 4'
        });

        if (this.neonPolygonLayer) {
          this.neonPolygonLayerGroup.removeLayer(this.neonPolygonLayer);
        }
        this.neonPolygonLayer = e.layer;
        this.neonPolygonLayerGroup.addLayer(this.neonPolygonLayer);

        e.layer.on('pm:edit', () => this._fireNeonPolygonChange());
        e.layer.on('pm:dragend', () => this._fireNeonPolygonChange());
        this._fireNeonPolygonChange();
      } else {
        // ---- Main AOI draw ----
        if (this.drawnLayer && this.drawnLayer !== e.layer) {
          this.map.removeLayer(this.drawnLayer);
        }
        this.drawnLayer = e.layer;
        this.updateDrawnAOI(this.drawnLayer);

        this.drawnLayer.on('pm:edit', () => this.updateDrawnAOI(this.drawnLayer));
        this.drawnLayer.on('pm:dragend', () => this.updateDrawnAOI(this.drawnLayer));
      }
    });

    this.map.on('pm:remove', (e) => {
      if (this.drawnLayer === e.layer) {
        this.drawnLayer = null;
        this.handleDrawDelete();
      } else if (this.neonPolygonLayer === e.layer) {
        this.neonPolygonLayer = null;
        this.onNeonPolygonChangeCallback(null);
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
  // Dynamic Earth Engine Tile Layers — Biomass Panel (activeEELayer)
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
  // -------------------------------------------------------------------------
  // NEON Proxy Context Satellite & High-Res Aerial Layers
  // -------------------------------------------------------------------------

  /**
   * Display the high-resolution RGB aerial orthophoto for the NEON site.
   */
  setNeonAerialLayer() {
    this.removeNeonEETileLayer();
    this.removeNeonAerialLayer();

    // Bounds in Leaflet [[south, west], [north, east]]
    const bounds = [
      [NEON_TILE_BOUNDS.sw[1], NEON_TILE_BOUNDS.sw[0]],
      [NEON_TILE_BOUNDS.ne[1], NEON_TILE_BOUNDS.ne[0]]
    ];

    this.neonAerialOverlay = L.imageOverlay('/neon_proxy_osbs.png', bounds, {
      opacity: 0.95,
      interactive: false,
      zIndex: 400
    }).addTo(this.map);
  }

  /**
   * Remove the high-resolution aerial image overlay.
   */
  removeNeonAerialLayer() {
    if (this.neonAerialOverlay) {
      this.map.removeLayer(this.neonAerialOverlay);
      this.neonAerialOverlay = null;
    }
  }

  /**
   * Set the NEON EE satellite tile layer.
   * Clips the tile layer to the user's polygon or the NEON site footprint.
   */
  setNeonEETileLayer(tileUrl, clipGeometry = null) {
    this.removeNeonEETileLayer();
    this.removeNeonAerialLayer();
    if (!tileUrl) return;

    this.neonEELayer = L.tileLayer(tileUrl, {
      maxZoom: 20,
      opacity: 0.85,
      attribution: 'Google Earth Engine &copy; Copernicus / ESA / ETH'
    }).addTo(this.map);

    // If a polygon is drawn, clip to it; otherwise clip to clipGeometry or NEON bounds
    const targetGeom = this.neonPolygonLayer ? this.neonPolygonLayer.toGeoJSON() : clipGeometry;
    if (targetGeom) {
      this.createNeonEEMask(targetGeom);
    }
  }

  /**
   * Remove the NEON EE tile layer AND its polygon clip mask.
   */
  removeNeonEETileLayer() {
    if (this.neonEELayer) {
      this.map.removeLayer(this.neonEELayer);
      this.neonEELayer = null;
    }
    this.removeNeonEEMask();
  }

  /**
   * Clip the NEON EE satellite tile layer to a polygon using CSS clip-path.
   *
   * @param {Object} polygonGeoJSON — GeoJSON Feature or Geometry (Polygon)
   */
  createNeonEEMask(polygonGeoJSON) {
    this.removeNeonEEMask();
    if (!polygonGeoJSON || !this.neonEELayer) return;

    const geom = polygonGeoJSON.type === 'Feature'
      ? polygonGeoJSON.geometry
      : polygonGeoJSON;

    if (!geom || geom.type !== 'Polygon' || !geom.coordinates || !geom.coordinates[0]) return;

    this._eeClipGeom = geom;
    this._applyEETileClipPath();
    this.map.on('move zoom moveend zoomend viewreset', this._applyEETileClipPath, this);
  }

  _applyEETileClipPath() {
    if (!this._eeClipGeom || !this.neonEELayer) return;

    const container = this.neonEELayer.getContainer();
    if (!container) return;

    const coords = this._eeClipGeom.coordinates[0].map(c => {
      const pt = this.map.latLngToContainerPoint(L.latLng(c[1], c[0]));
      return `${pt.x}px ${pt.y}px`;
    });

    container.style.clipPath = `polygon(${coords.join(', ')})`;
  }

  removeNeonEEMask() {
    this.map.off('move zoom moveend zoomend viewreset', this._applyEETileClipPath, this);

    if (this.neonEELayer) {
      const container = this.neonEELayer.getContainer();
      if (container) container.style.clipPath = 'none';
    }

    this._eeClipGeom = null;
  }

  // -------------------------------------------------------------------------
  // Map Viewport Helpers
  // -------------------------------------------------------------------------

  /**
   * Returns current map viewport as [minLon, minLat, maxLon, maxLat].
   * Used to georeference uploaded images that have no embedded CRS.
   */
  getMapBounds() {
    const b = this.map.getBounds();
    return [
      b.getWest(),
      b.getSouth(),
      b.getEast(),
      b.getNorth()
    ];
  }

  /**
   * Returns the current map center as [lon, lat].
   */
  getMapCenter() {
    const c = this.map.getCenter();
    return [c.lng, c.lat];
  }

  // -------------------------------------------------------------------------
  // Uploaded Image Overlay + Canvas Detection Circles
  // -------------------------------------------------------------------------

  /**
   * Display the uploaded image (as a browser-renderable base64 data URL) as a
   * Leaflet ImageOverlay and draw tree crown circles on a synchronized canvas.
   *
   * The backend converts TIFF → JPEG (base64) because browsers cannot render TIFFs.
   * Normalized pixel coordinates on each feature drive canvas circle sizing so
   * circles are proportionally identical to the NEON proxy demo appearance.
   *
   * @param {string}   imageDataUrl - data:image/jpeg;base64,... from backend response
   * @param {number[]} bounds       - [minLon, minLat, maxLon, maxLat]
   * @param {Object}   geojson      - FeatureCollection with pixel_*_norm properties
   */
  showUploadedImageOverlay(imageDataUrl, bounds, geojson = null) {
    this.clearImageOverlay();

    const [minLon, minLat, maxLon, maxLat] = bounds;
    const leafletBounds = [[minLat, minLon], [maxLat, maxLon]];

    // Use the data URL directly — works for JPEG, PNG, and any browser-displayable format.
    // No URL.createObjectURL needed (which fails for TIFF).
    this.uploadedImageOverlay = L.imageOverlay(imageDataUrl, leafletBounds, {
      opacity: 1.0,
      interactive: false,
      zIndex: 200,
      className: 'uploaded-image-overlay'
    }).addTo(this.map);

    // Fly to image — use high maxZoom so even a small tile fills the screen
    this.map.fitBounds(leafletBounds, { maxZoom: 21, padding: [40, 40] });

    // 2. Canvas overlay — transparent, same bounds, draws circles on top
    this._uploadCanvasGeojson = geojson;
    this._uploadCanvasBounds  = leafletBounds;

    this._uploadCanvas = document.createElement('canvas');
    this._uploadCanvas.style.position      = 'absolute';
    this._uploadCanvas.style.pointerEvents = 'none';
    this._uploadCanvas.style.zIndex        = '500';

    // Mount canvas into the Leaflet pane so it scrolls/zooms with the map
    this.map.getPanes().overlayPane.appendChild(this._uploadCanvas);

    // Sync canvas on every map move/zoom
    this._syncCanvasToOverlay = () => this._redrawDetectionCanvas();
    this.map.on('move zoom moveend zoomend viewreset resize', this._syncCanvasToOverlay);

    // Pre-load the image to get natural dimensions for radius scaling
    const img = new Image();
    img.onload = () => {
      this._uploadImageNaturalW = img.naturalWidth;
      this._uploadImageNaturalH = img.naturalHeight;
      this._redrawDetectionCanvas();
    };
    img.src = imageDataUrl;
  }

  /** Reproject canvas position/size to match current Leaflet image overlay position. */
  _redrawDetectionCanvas() {
    const canvas = this._uploadCanvas;
    const bounds = this._uploadCanvasBounds;
    if (!canvas || !bounds) return;

    // NW (top-left) and SE (bottom-right) in Leaflet layer coordinates
    // (matches the exact coordinate system of L.imageOverlay inside overlayPane)
    const nw = this.map.latLngToLayerPoint(L.latLng(bounds[1][0], bounds[0][1]));
    const se = this.map.latLngToLayerPoint(L.latLng(bounds[0][0], bounds[1][1]));

    const left   = Math.min(nw.x, se.x);
    const top    = Math.min(nw.y, se.y);
    const width  = Math.abs(se.x - nw.x);
    const height = Math.abs(se.y - nw.y);

    if (width < 1 || height < 1) return;

    // Position canvas exactly over the image overlay
    canvas.style.left   = `${left}px`;
    canvas.style.top    = `${top}px`;
    canvas.width  = Math.ceil(width);
    canvas.height = Math.ceil(height);

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const geojson = this._uploadCanvasGeojson;
    if (!geojson || !geojson.features || geojson.features.length === 0) return;

    // The image natural dimensions (for correct scaling of r_norm)
    const imgW = this._uploadImageNaturalW || 400;
    const imgH = this._uploadImageNaturalH || 400;

    geojson.features.forEach((feat) => {
      const p = feat.properties || {};
      const cxN = p.pixel_cx_norm;
      const cyN = p.pixel_cy_norm;
      const rN  = p.pixel_r_norm;

      // Skip if normalized coords missing (e.g., demo result without pixel info)
      if (cxN == null || cyN == null || rN == null) return;

      // Map 0-1 normalized to canvas pixels
      const cx = cxN * width;
      const cy = cyN * height;
      // Scale radius by canvas width (since r_norm is relative to min(imgW,imgH))
      const r  = rN * Math.min(width, height);

      if (r < 0.5) return;

      const score = Number(p.confidence || 0.65);

      // Outer glow
      ctx.beginPath();
      ctx.arc(cx, cy, r + 2, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(34,211,238,0.35)';
      ctx.lineWidth   = 4;
      ctx.stroke();

      // Main circle fill
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle   = `rgba(6,182,212,${0.25 + score * 0.3})`;
      ctx.fill();

      // Circle stroke
      ctx.strokeStyle = '#22d3ee';
      ctx.lineWidth   = Math.max(1.5, r * 0.08);
      ctx.stroke();
    });
  }

  /**
   * Set display mode for the image overlay and detection circles:
   *  - 'both': original image visible + blue/cyan detection circles drawn on top
   *  - 'image': original aerial/drone image only (circles hidden)
   *  - 'crowns': blue/cyan tree crown circles only (original image hidden)
   *
   * @param {'both'|'image'|'crowns'} mode
   */
  setOverlayDisplayMode(mode = 'both') {
    this.overlayDisplayMode = mode;

    // 1. Image overlay opacity
    if (this.uploadedImageOverlay) {
      if (mode === 'crowns') {
        this.uploadedImageOverlay.setOpacity(0.0);
      } else {
        this.uploadedImageOverlay.setOpacity(1.0);
      }
    }

    // 2. Canvas visibility
    if (this._uploadCanvas) {
      if (mode === 'image') {
        this._uploadCanvas.style.display = 'none';
      } else {
        this._uploadCanvas.style.display = 'block';
        this._redrawDetectionCanvas();
      }
    }

    // 3. Leaflet vector layer styling
    if (this.treeLayerGroup) {
      this.treeLayerGroup.eachLayer((layer) => {
        if (mode === 'image') {
          layer.setStyle({ opacity: 0, fillOpacity: 0 });
        } else if (mode === 'crowns') {
          layer.setStyle({
            color: '#22d3ee',
            fillColor: '#06b6d4',
            fillOpacity: 0.5,
            opacity: 1,
            weight: 2
          });
        } else {
          // In 'both' mode, canvas handles high-precision circle rendering,
          // so Leaflet vector circles remain transparent for click popups.
          layer.setStyle({
            color: 'transparent',
            fillColor: 'transparent',
            fillOpacity: 0,
            opacity: 0,
            weight: 0
          });
        }
      });
    }
  }

  /**
   * Remove the uploaded image overlay and detection canvas from the map.
   */
  clearImageOverlay() {
    if (this._syncCanvasToOverlay) {
      this.map.off('move zoom moveend zoomend viewreset resize', this._syncCanvasToOverlay);
      this._syncCanvasToOverlay = null;
    }
    if (this._uploadCanvas && this._uploadCanvas.parentNode) {
      this._uploadCanvas.parentNode.removeChild(this._uploadCanvas);
      this._uploadCanvas = null;
    }
    if (this.uploadedImageOverlay) {
      this.uploadedImageOverlay.remove();
      this.uploadedImageOverlay = null;
    }
    this._uploadCanvasGeojson = null;
    this._uploadCanvasBounds  = null;
    this.overlayDisplayMode   = 'both';
  }


  // -------------------------------------------------------------------------
  // Tree Detection Layer & Dynamic Highlighting
  // -------------------------------------------------------------------------
  /**
   * @param {Object}  geojson      - GeoJSON FeatureCollection
   * @param {string}  sourceLabel  - Display label
   * @param {boolean} overlayMode  - If true, Leaflet circles are invisible (canvas draws visuals)
   */
  setTreeDetectionsGeoJSON(geojson, sourceLabel = 'DeepForest RetinaNet', overlayMode = false) {
    this.treeLayerGroup.clearLayers();
    this.treeCircleLayers = [];
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

      // In overlayMode the canvas draws the visible circles; Leaflet circles
      // are transparent/invisible but still handle popup clicks.
      const circle = L.circle(latlng, {
        radius: Math.max(1.0, parseFloat(radiusM) * 1.2),
        color: overlayMode ? 'transparent' : '#22d3ee',
        weight: overlayMode ? 0 : 1.8,
        fillColor: overlayMode ? 'transparent' : '#06b6d4',
        fillOpacity: overlayMode ? 0 : 0.45
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
      this.treeCircleLayers.push({ circle, feat, index });
    });

    if (bounds.isValid() && !this.uploadedImageOverlay) {
      this.map.fitBounds(bounds, { maxZoom: 18, padding: [30, 30] });
    }

    this.neonDetectionsGeoJSON = geojson;

    // Apply polygon highlight if active
    if (this.neonPolygonLayer) {
      this.highlightTreesInPolygon(this.neonPolygonLayer.toGeoJSON());
    }
  }

  /**
   * Visually highlights tree circles that fall inside the polygon GeoJSON,
   * while dimming tree circles that fall outside it.
   */
  highlightTreesInPolygon(polygonGeoJSON) {
    if (!this.treeCircleLayers || this.treeCircleLayers.length === 0) return;

    if (!polygonGeoJSON) {
      // Reset all circles to default vibrant cyan
      this.treeCircleLayers.forEach(({ circle }) => {
        circle.setStyle({
          color: '#22d3ee',
          fillColor: '#06b6d4',
          fillOpacity: 0.45,
          weight: 1.8
        });
      });
      return;
    }

    const polyFeature = polygonGeoJSON.type === 'Feature'
      ? polygonGeoJSON
      : { type: 'Feature', geometry: polygonGeoJSON, properties: {} };

    this.treeCircleLayers.forEach(({ circle, feat }) => {
      const pt = turf.point(feat.geometry.coordinates);
      let isInside = false;
      try {
        isInside = turf.booleanPointInPolygon(pt, polyFeature);
      } catch (_) {
        isInside = false;
      }

      if (isInside) {
        circle.setStyle({
          color: '#10b981',      // vibrant glowing emerald green
          fillColor: '#34d399',
          fillOpacity: 0.85,
          weight: 2.8
        });
      } else {
        circle.setStyle({
          color: '#64748b',      // muted slate
          fillColor: '#334155',
          fillOpacity: 0.15,
          weight: 1.0
        });
      }
    });
  }

  // -------------------------------------------------------------------------
  // NEON Polygon Draw & Analysis
  // -------------------------------------------------------------------------

  startNeonPolygonDraw() {
    this._neonPolygonDrawing = true;
    this.map.pm.enableDraw('Polygon', {
      snappable: true,
      snapDistance: 20,
      pathOptions: {
        color: '#06b6d4',
        fillColor: '#06b6d4',
        fillOpacity: 0.12,
        weight: 2,
        dashArray: '4, 4'
      }
    });
  }

  clearNeonPolygon() {
    if (this.neonPolygonLayer) {
      this.neonPolygonLayerGroup.removeLayer(this.neonPolygonLayer);
      this.neonPolygonLayer = null;
    }
    this._neonPolygonDrawing = false;
    this.map.pm.disableDraw();
    this.highlightTreesInPolygon(null);
    this.removeNeonEEMask();
    this.onNeonPolygonChangeCallback(null);
  }

  /**
   * Internal: fire the polygon-change callback with the current polygon GeoJSON.
   */
  _fireNeonPolygonChange() {
    if (!this.neonPolygonLayer) return;
    const geojson = this.neonPolygonLayer.toGeoJSON();
    this.onNeonPolygonChangeCallback(geojson);
  }

  /**
   * Compute statistics for tree-crown detections that fall inside the given
   * polygon GeoJSON feature.  Uses Turf.js `pointsWithinPolygon`.
   *
   * @param {Object} polygonGeoJSON  – GeoJSON Feature (Polygon)
   * @param {Object} detectionsGeoJSON – GeoJSON FeatureCollection of Point detections
   * @returns {Object} stats
   */
  computeNeonPolygonStats(polygonGeoJSON, detectionsGeoJSON) {
    const empty = {
      count: 0,
      areaHa: 0,
      densityPerHa: 0,
      confidenceMin: null,
      confidenceMax: null,
      crownRadiusMin: null,
      crownRadiusMax: null
    };

    if (!polygonGeoJSON || !detectionsGeoJSON || !detectionsGeoJSON.features ||
        detectionsGeoJSON.features.length === 0) {
      return empty;
    }

    // Filter points to only Point geometry features (ignore Polygons etc.)
    const pointsFC = {
      type: 'FeatureCollection',
      features: detectionsGeoJSON.features.filter(
        f => f.geometry && f.geometry.type === 'Point'
      )
    };

    if (pointsFC.features.length === 0) return empty;

    // Ensure we have a proper Feature wrapper for the polygon
    const polyFeature = polygonGeoJSON.type === 'Feature'
      ? polygonGeoJSON
      : { type: 'Feature', geometry: polygonGeoJSON, properties: {} };

    let inside;
    try {
      inside = turf.pointsWithinPolygon(pointsFC, polyFeature);
    } catch (err) {
      console.warn('Turf pointsWithinPolygon error:', err);
      return empty;
    }

    const count = inside.features.length;

    // Calculate polygon area in hectares
    let areaHa = 0;
    try {
      areaHa = turf.area(polyFeature) / 10000;
    } catch (_) {}

    const densityPerHa = areaHa > 0 ? count / areaHa : 0;

    const confidences = inside.features
      .map(f => Number(f.properties.confidence || f.properties.score || 0))
      .filter(v => !isNaN(v) && v > 0);

    const radii = inside.features
      .map(f => Number(f.properties.radius_m || 0))
      .filter(v => !isNaN(v) && v > 0);

    return {
      count,
      areaHa,
      densityPerHa,
      confidenceMin: confidences.length ? Math.min(...confidences) : null,
      confidenceMax: confidences.length ? Math.max(...confidences) : null,
      crownRadiusMin: radii.length ? Math.min(...radii) : null,
      crownRadiusMax: radii.length ? Math.max(...radii) : null
    };
  }

  // -------------------------------------------------------------------------
  // NEON Site Boundary (fixed GeoTIFF footprint)
  // -------------------------------------------------------------------------

  /**
   * Draw the fixed NEON GeoTIFF footprint as a styled boundary polygon.
   * This shows the user exactly where they can draw their analysis polygon
   * to count trees from the proxy dataset.
   */
  showNeonSiteBoundary() {
    if (this.neonSiteBoundaryLayer) return; // already shown

    // Amber dashed outer boundary polygon (the GeoTIFF coverage footprint)
    this.neonSiteBoundaryLayer = L.polygon(NEON_TILE_BOUNDS.ring, {
      color: '#f59e0b',
      weight: 3,
      dashArray: '8, 5',
      fillColor: '#f59e0b',
      fillOpacity: 0.06,
      interactive: false
    }).addTo(this.map);

    // Label tooltip that always shows
    this.neonSiteBoundaryLayer.bindTooltip(
      [
        '<div style="font-family:var(--font-sans);font-size:0.72rem;line-height:1.4;">',
        '  <b style="color:#f59e0b;">📍 NEON GeoTIFF Boundary</b><br>',
        '  Ordway-Swisher Biological Station, FL<br>',
        '  <span style="color:#9ca3af;">40 m × 40 m • 0.16 ha • 0.10 m/px</span><br>',
        '  <span style="color:#06b6d4;">✏️ Draw your analysis polygon inside this area</span>',
        '</div>'
      ].join(''),
      {
        permanent: true,
        direction: 'top',
        offset: [0, -6],
        opacity: 1,
        className: 'neon-boundary-tooltip'
      }
    ).openTooltip();
  }

  /**
   * Remove the NEON site boundary polygon from the map.
   */
  hideNeonSiteBoundary() {
    if (this.neonSiteBoundaryLayer) {
      this.map.removeLayer(this.neonSiteBoundaryLayer);
      this.neonSiteBoundaryLayer = null;
    }
  }

  // -------------------------------------------------------------------------
  // Mode Management & Quick Views
  // -------------------------------------------------------------------------
  setMode(mode) {
    this.currentMode = mode;
    if (mode === 'biomass') {
      if (this.treeLayerGroup) this.map.removeLayer(this.treeLayerGroup);
      if (this.neonPolygonLayerGroup) this.map.removeLayer(this.neonPolygonLayerGroup);
      if (this.neonEELayer) this.map.removeLayer(this.neonEELayer);
      if (this.neonAerialOverlay) this.map.removeLayer(this.neonAerialOverlay);
      if (this.activeEELayer) this.map.addLayer(this.activeEELayer);
      this.hideNeonSiteBoundary();
    } else if (mode === 'trees') {
      if (this.activeEELayer) this.map.removeLayer(this.activeEELayer);
      if (this.treeLayerGroup) this.map.addLayer(this.treeLayerGroup);
      if (this.neonPolygonLayerGroup) this.map.addLayer(this.neonPolygonLayerGroup);
      if (this.neonAerialOverlay) this.map.addLayer(this.neonAerialOverlay);
      if (this.neonEELayer) {
        this.map.addLayer(this.neonEELayer);
        // Re-apply clip-path after the layer is re-added to the DOM
        if (this._eeClipGeom) {
          requestAnimationFrame(() => this._applyEETileClipPath());
        }
      }
    }
  }

  flyToNeon() {
    // Fly to the exact center of the NEON OSBS_029.tif GeoTIFF tile
    // [lat, lon] — NEON_PROXY_CENTER is [lat, lon] for Leaflet
    const [lat, lon] = [NEON_TILE_BOUNDS.center[1], NEON_TILE_BOUNDS.center[0]];
    this.map.flyTo([lat, lon], 21, { duration: 2.0 });
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
