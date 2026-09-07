/**
 * API client module for Forest AI backend endpoints.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

async function handleResponse(res) {
  if (!res.ok) {
    let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
    try {
      const err = await res.json();
      if (err.detail) errorDetail = err.detail;
    } catch (_) {}
    throw new Error(errorDetail);
  }
  return res.json();
}

export async function fetchStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/v1/status`);
    return await handleResponse(res);
  } catch (err) {
    // fallback check health endpoint
    const res = await fetch(`${API_BASE}/health`);
    return await handleResponse(res);
  }
}

export async function fetchStudyArea() {
  const res = await fetch(`${API_BASE}/api/v1/study-area`);
  return handleResponse(res);
}

export async function fetchFeatures() {
  const res = await fetch(`${API_BASE}/api/v1/features`);
  return handleResponse(res);
}

export async function fetchBiomassPreset(startDate = '2023-01-01', endDate = '2023-04-30') {
  const params = new URLSearchParams({ start_date: startDate, end_date: endDate });
  const res = await fetch(`${API_BASE}/api/v1/biomass?${params}`);
  return handleResponse(res);
}

export async function fetchBiomassStats() {
  const res = await fetch(`${API_BASE}/api/v1/biomass/stats`);
  return handleResponse(res);
}

export async function geocodeQuery(query) {
  const params = new URLSearchParams({ q: query });
  const res = await fetch(`${API_BASE}/api/v1/geocode?${params}`);
  return handleResponse(res);
}

export async function fetchAOIBiomass(geometry, startDate = '2023-01-01', endDate = '2023-04-30') {
  const res = await fetch(`${API_BASE}/api/v1/analysis/biomass`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      geometry,
      start_date: startDate,
      end_date: endDate
    })
  });
  return handleResponse(res);
}

export async function fetchTreeDetection(scoreThreshold = 0.15) {
  const params = new URLSearchParams({ score_thresh: scoreThreshold });
  const res = await fetch(`${API_BASE}/api/v1/trees/detect?${params}`, {
    method: 'POST'
  });
  return handleResponse(res);
}

export async function uploadTreeDetection(file, scoreThreshold = 0.15, aoiGeometry = null, mapBounds = null) {
  const formData = new FormData();
  formData.append('file', file);
  if (scoreThreshold != null) {
    formData.append('score_thresh', scoreThreshold.toString());
  }
  if (aoiGeometry) {
    formData.append('aoi_geojson', JSON.stringify(aoiGeometry));
  }
  // Pass current map viewport bounds so the backend can place detections at
  // the correct geographic location for non-georeferenced images (e.g. screenshots).
  if (mapBounds && mapBounds.length === 4) {
    formData.append('map_bounds', JSON.stringify(mapBounds));
  }
  const res = await fetch(`${API_BASE}/api/v1/trees/detect/upload`, {
    method: 'POST',
    body: formData
  });
  return handleResponse(res);
}

export async function fetchDemoTile() {
  const res = await fetch(`${API_BASE}/api/v1/trees/demo-tile`);
  return handleResponse(res);
}

export async function fetchEETileUrl(layer, geometry = null, startDate = '2023-01-01', endDate = '2023-04-30') {
  const res = await fetch(`${API_BASE}/api/v1/ee/tile-url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      layer,
      geometry,
      start_date: startDate,
      end_date: endDate
    })
  });
  return handleResponse(res);
}

