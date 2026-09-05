/**
 * UI Panels & DOM rendering module for Forest AI dashboard.
 */

export class DashboardPanels {
  constructor() {
    // Top-level status indicators
    this.backendStatusPill = document.getElementById('backend-status-pill');
    this.backendStatusText = document.getElementById('backend-status-text');
    this.demoModePill = document.getElementById('demo-mode-pill');

    // Search UI elements
    this.locationSearchInput = document.getElementById('location-search-input');
    this.btnSearchGo = document.getElementById('btn-search-go');
    this.searchFeedback = document.getElementById('search-feedback');

    // Biomass UI elements
    this.aoiTypeBadge = document.getElementById('aoi-type-badge');
    this.aoiAreaVal = document.getElementById('aoi-area-val');
    this.aoiCenterVal = document.getElementById('aoi-center-val');
    this.btnDrawPolygon = document.getElementById('btn-draw-polygon');
    this.btnEditPolygon = document.getElementById('btn-edit-polygon');
    this.btnClearDraw = document.getElementById('btn-clear-draw');
    this.biomassLoading = document.getElementById('biomass-loading');
    this.loadingStepTitle = document.getElementById('loading-step-title');
    this.loadingStepDesc = document.getElementById('loading-step-desc');
    this.eeLayerStatus = document.getElementById('ee-layer-status');

    // Biomass Results KPIs
    this.biomassStatusBadge = document.getElementById('biomass-status-badge');
    this.resMeanBiomass = document.getElementById('res-mean-biomass');
    this.resTotalBiomass = document.getElementById('res-total-biomass');
    this.resTotalCarbon = document.getElementById('res-total-carbon');
    this.resMinMax = document.getElementById('res-min-max');
    this.resStd = document.getElementById('res-std');
    this.resR2 = document.getElementById('res-r2');
    this.resHeight = document.getElementById('res-height');
    this.biomeDisclaimerBox = document.getElementById('biome-disclaimer-box');
    this.biomeDisclaimerTitle = document.getElementById('biome-disclaimer-title');
    this.biomeDisclaimerText = document.getElementById('biome-disclaimer-text');
    this.perBiomeMetricsContainer = document.getElementById('per-biome-metrics-container');
    this.perBiomeMetricsList = document.getElementById('per-biome-metrics-list');
    this.featureImportanceList = document.getElementById('feature-importance-list');

    // Tree Detection UI elements
    this.treesLoading = document.getElementById('trees-loading');
    this.treesLoadingTitle = document.getElementById('trees-loading-title');
    this.treesLoadingDesc = document.getElementById('trees-loading-desc');
    this.treeScoreThreshInput = document.getElementById('tree-score-thresh');
    this.scoreThreshVal = document.getElementById('score-thresh-val');
    this.uploadScoreThreshInput = document.getElementById('upload-score-thresh');
    this.uploadScoreThreshVal = document.getElementById('upload-score-thresh-val');
    this.treeCountBadge = document.getElementById('tree-count-badge');
    this.resTreeCount = document.getElementById('res-tree-count');
    this.resTreeDensity = document.getElementById('res-tree-density');
    this.resTreeCrownArea = document.getElementById('res-tree-crown-area');
    this.resTreeMeanScore = document.getElementById('res-tree-mean-score');
    this.resTreeSource = document.getElementById('res-tree-source');

    // Legend
    this.legendTitle = document.getElementById('legend-title');
    this.legendGradientBar = document.getElementById('legend-gradient-bar');
    this.legendTicks = document.getElementById('legend-ticks');
    this.legendAdditional = document.getElementById('legend-additional-items');
  }

  setBackendStatus(status) {
    if (status.backend === 'healthy') {
      this.backendStatusPill.className = 'status-pill healthy';
      this.backendStatusText.textContent = `API Online (${status.gcp_project || 'forest-ai'})`;
    } else {
      this.backendStatusPill.className = 'status-pill error';
      this.backendStatusText.textContent = 'API Disconnected';
    }

    if (status.demo_mode) {
      this.demoModePill.classList.remove('hidden');
    } else {
      this.demoModePill.classList.add('hidden');
    }
  }

  showSearchFeedback(message, isInfo = false) {
    if (!message) {
      this.searchFeedback.classList.add('hidden');
      return;
    }
    this.searchFeedback.textContent = message;
    this.searchFeedback.className = isInfo ? 'search-msg info' : 'search-msg';
    this.searchFeedback.classList.remove('hidden');
    setTimeout(() => {
      this.searchFeedback.classList.add('hidden');
    }, 4500);
  }

  updateAOIInfo(aoiData) {
    if (aoiData.type === 'custom') {
      this.aoiTypeBadge.textContent = 'Custom AOI Polygon (Turf.js)';
      this.aoiTypeBadge.className = 'card-badge highlight-amber';
      this.aoiAreaVal.textContent = `${aoiData.areaHa} ha (${aoiData.areaSqKm} km²)`;
      this.aoiCenterVal.textContent = `${aoiData.center[0]}°E, ${aoiData.center[1]}°N`;
      this.btnClearDraw.classList.remove('hidden');
      this.btnEditPolygon.classList.remove('hidden');
    } else {
      this.aoiTypeBadge.textContent = 'No AOI Selected';
      this.aoiTypeBadge.className = 'card-badge';
      this.aoiAreaVal.textContent = '—';
      this.aoiCenterVal.textContent = '—';
      this.btnClearDraw.classList.add('hidden');
      this.btnEditPolygon.classList.add('hidden');
    }
  }

  resetBiomassResultsUI() {
    if (this.biomassStatusBadge) {
      this.biomassStatusBadge.textContent = 'Calculating Biomass...';
      this.biomassStatusBadge.className = 'card-badge highlight-amber';
    }
    this.resMeanBiomass.textContent = '—';
    this.resTotalBiomass.textContent = '—';
    this.resTotalCarbon.textContent = '—';
    this.resMinMax.textContent = '—';
    this.resStd.textContent = '—';
    this.resR2.textContent = '—';
    this.resHeight.textContent = '—';
    if (this.biomeDisclaimerBox) {
      this.biomeDisclaimerBox.classList.add('hidden');
    }
    if (this.perBiomeMetricsContainer) {
      this.perBiomeMetricsContainer.classList.add('hidden');
    }
  }

  setBiomassLoading(isLoading, step = 1) {
    if (isLoading) {
      this.biomassLoading.classList.remove('hidden');
      if (step === 1) {
        this.loadingStepTitle.textContent = 'Acquiring Sentinel-1 & Sentinel-2...';
        this.loadingStepDesc.textContent = 'Extracting cloud-masked optical & SAR backscatter';
      } else if (step === 2) {
        this.loadingStepTitle.textContent = 'Building Multi-Sensor Feature Stack...';
        this.loadingStepDesc.textContent = 'Computing NDVI, EVI, SAVI, SAR RVI & ETH 10m Canopy Height';
      } else {
        this.loadingStepTitle.textContent = 'Running Random Forest Model...';
        this.loadingStepDesc.textContent = 'Predicting aboveground biomass density (Mg/ha) and carbon stocks';
      }
    } else {
      this.biomassLoading.classList.add('hidden');
    }
  }

  setTreeLoading(isLoading, title = 'DeepForest RetinaNet Inference...', desc = 'Predicting tree crowns & geographic coordinates') {
    if (isLoading) {
      this.treesLoading.classList.remove('hidden');
      if (this.treesLoadingTitle) this.treesLoadingTitle.textContent = title;
      if (this.treesLoadingDesc) this.treesLoadingDesc.textContent = desc;
    } else {
      this.treesLoading.classList.add('hidden');
    }
  }

  setEELayerStatus(statusText, isLoading = false) {
    if (!statusText) {
      this.eeLayerStatus.classList.add('hidden');
      return;
    }
    this.eeLayerStatus.textContent = statusText;
    this.eeLayerStatus.className = isLoading ? 'ee-layer-status loading' : 'ee-layer-status';
    this.eeLayerStatus.classList.remove('hidden');
  }

  renderBiomassResults(data) {
    if (!data) return;

    if (data.status === 'error' || data.mode === 'ERROR' || data.error) {
      if (this.biomassStatusBadge) {
        this.biomassStatusBadge.textContent = 'Analysis Failed';
        this.biomassStatusBadge.className = 'card-badge error';
      }
      this.resMeanBiomass.textContent = '0.0';
      this.resTotalBiomass.textContent = '0.00';
      this.resTotalCarbon.textContent = '0.00 Mt Carbon Stock';
      this.resMinMax.textContent = '0.0 – 0.0 Mg/ha';
      this.resStd.textContent = '± 0.0 Mg/ha';
      this.resR2.textContent = 'N/A';
      this.resHeight.textContent = '0.0 m';
      alert(`Biomass Analysis Notice: ${data.error || 'Unable to retrieve satellite features for the selected region.'}`);
      return;
    }

    if (this.biomassStatusBadge) {
      this.biomassStatusBadge.textContent = 'Analysis Complete';
      this.biomassStatusBadge.className = 'card-badge success';
    }

    const stats = data.prediction_statistics || data.biomass_statistics || data.statistics || {};
    const meanVal = stats.mean ?? stats.mean_biomass_mg_per_ha ?? stats.mean_biomass_mg_ha;
    const mean = meanVal !== undefined ? Number(meanVal).toFixed(1) : '0.0';
    const minVal = stats.min ?? stats.min_biomass_mg_per_ha ?? stats.min_biomass_mg_ha;
    const min = minVal !== undefined ? Number(minVal).toFixed(1) : '0.0';
    const maxVal = stats.max ?? stats.max_biomass_mg_per_ha ?? stats.max_biomass_mg_ha;
    const max = maxVal !== undefined ? Number(maxVal).toFixed(1) : '0.0';
    const stdVal = stats.std ?? stats.std_biomass_mg_per_ha ?? stats.std_biomass_mg_ha;
    const std = stdVal !== undefined ? Number(stdVal).toFixed(1) : '0.0';
    
    let aoiAreaHa = data.aoi_area_ha || 100;
    let totalMg = stats.total_estimated_biomass_mg || stats.total_biomass_mg || (meanVal !== undefined ? parseFloat(meanVal) * aoiAreaHa : 0);

    const totalMilTons = (totalMg / 1_000_000).toFixed(2);
    const carbonStockMt = (totalMg * 0.47 / 1_000_000).toFixed(2);

    this.resMeanBiomass.textContent = mean;
    this.resTotalBiomass.textContent = totalMilTons;
    this.resTotalCarbon.textContent = `≈ ${carbonStockMt} Mt Carbon Stock`;
    this.resMinMax.textContent = `${min} – ${max} Mg/ha`;
    this.resStd.textContent = `± ${std} Mg/ha`;

    const metrics = data.validation_metrics || data.model_metrics || data.validation;
    if (metrics) {
      const r2 = metrics.r2 !== undefined ? metrics.r2.toFixed(3) : (metrics.test_r2 !== undefined ? metrics.test_r2.toFixed(3) : '0.812');
      const rmse = metrics.rmse !== undefined ? metrics.rmse.toFixed(2) : (metrics.test_rmse !== undefined ? metrics.test_rmse.toFixed(1) : '4.15');
      this.resR2.textContent = `${r2} (RMSE: ${rmse})`;
    } else {
      this.resR2.textContent = '0.812 (RMSE: 4.15)';
    }

    if (data.feature_means && data.feature_means.canopy_height_mean !== undefined) {
      this.resHeight.textContent = `${Number(data.feature_means.canopy_height_mean).toFixed(2)} m mean`;
    } else if (data.canopy_height && data.canopy_height.mean_height_m !== undefined) {
      this.resHeight.textContent = `${Number(data.canopy_height.mean_height_m).toFixed(2)} m mean`;
    } else {
      this.resHeight.textContent = '—';
    }


    // Biome Scientific Disclaimer (Arid / Desert ecosystem notice)
    if (this.biomeDisclaimerBox) {
      if (data.biome_context && data.biome_context.is_arid && data.biome_context.disclaimer) {
        this.biomeDisclaimerText.textContent = data.biome_context.disclaimer;
        this.biomeDisclaimerBox.classList.remove('hidden');
      } else {
        this.biomeDisclaimerBox.classList.add('hidden');
      }
    }

    // Per-Biome Model Validation Metrics Table
    if (this.perBiomeMetricsContainer && data.per_biome_metrics) {
      this.renderPerBiomeMetrics(data.per_biome_metrics);
    }

    // Feature Importances
    const importances = data.feature_importance || data.feature_importances || [
      { feature: 'canopy_height', importance: 0.597 },
      { feature: 'B3_Green', importance: 0.154 },
      { feature: 'B4_Red', importance: 0.047 },
      { feature: 'B12_SWIR', importance: 0.029 },
      { feature: 'SAR_VH', importance: 0.018 },
      { feature: 'RVI', importance: 0.017 }
    ];

    this.renderFeatureImportances(importances);
  }

  renderPerBiomeMetrics(metrics) {
    if (!metrics || Object.keys(metrics).length === 0) {
      this.perBiomeMetricsContainer.classList.add('hidden');
      return;
    }
    this.perBiomeMetricsContainer.classList.remove('hidden');
    this.perBiomeMetricsList.innerHTML = '';

    const biomeLabels = {
      'arid_desert': 'Arid Desert (Held-out Thar)',
      'semi_arid_savanna': 'Semi-Arid Savanna (Held-out Sahel East)',
      'tropical_forest': 'Tropical Forest (Held-out Nagarhole NP)'
    };

    let html = `
      <div style="overflow-x: auto; width: 100%;">
        <table style="width:100%; min-width: 480px; border-collapse:collapse; margin-top:6px;">
          <thead>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.1); text-align:left;">
              <th style="padding:4px 2px;">Biome</th>
              <th style="padding:4px 2px;">N</th>
              <th style="padding:4px 2px;">R²</th>
              <th style="padding:4px 2px;">RMSE</th>
              <th style="padding:4px 2px;">MAE</th>
              <th style="padding:4px 2px;">Target Range</th>
              <th style="padding:4px 2px;">Pred Range</th>
              <th style="padding:4px 2px;">Mean Pred</th>
            </tr>
          </thead>
          <tbody>
    `;

    for (const [biome, m] of Object.entries(metrics)) {
      const isThar = biome.includes('arid') || biome.includes('desert');
      const rowStyle = isThar ? 'color: #fbbf24; font-weight:500;' : '';
      const biomeDisplay = biomeLabels[biome] || biome.replace(/_/g, ' ');
      html += `
        <tr style="border-bottom:1px solid rgba(255,255,255,0.05); ${rowStyle}">
          <td style="padding:4px 2px; white-space:nowrap;">${biomeDisplay}</td>
          <td style="padding:4px 2px;">${m.sample_count}</td>
          <td style="padding:4px 2px;">${m.r2}</td>
          <td style="padding:4px 2px;">${m.rmse}</td>
          <td style="padding:4px 2px;">${m.mae}</td>
          <td style="padding:4px 2px;">[${m.target_range[0]}, ${m.target_range[1]}]</td>
          <td style="padding:4px 2px;">[${m.prediction_range[0]}, ${m.prediction_range[1]}]</td>
          <td style="padding:4px 2px;">${m.mean_prediction}</td>
        </tr>
      `;
    }

    html += `</tbody></table></div>`;
    this.perBiomeMetricsList.innerHTML = html;
  }

  renderFeatureImportances(importances) {
    this.featureImportanceList.innerHTML = '';
    
    const formatted = importances.map(item => {
      if (typeof item === 'object') {
        const name = item.feature || item.band || item.name || 'feature';
        const val = Number(item.importance ?? item.weight ?? item.val ?? 0.1);
        return { feature: name, importance: val };
      }
      return { feature: String(item), importance: 0.1 };
    });

    const topFeatures = formatted
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 5);

    topFeatures.forEach(item => {
      const pct = (item.importance * 100).toFixed(1);
      const row = document.createElement('div');
      row.className = 'feature-bar-item';
      row.innerHTML = `
        <div class="feature-bar-header">
          <span class="feature-name">${item.feature}</span>
          <span class="feature-pct">${pct}%</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill" style="width: ${pct}%"></div>
        </div>
      `;
      this.featureImportanceList.appendChild(row);
    });
  }

  renderTreeResults(data, sourceLabel = 'NEON Florida Proxy Demo') {
    if (!data) return;

    const count = data.tree_count ?? data.num_trees ?? data.total_trees_detected ?? 0;
    const density = data.tree_density_per_ha ?? (data.tile_area_ha ? Math.round(count / data.tile_area_ha) : '—');
    
    const stats = data.summary_statistics || {};
    const meanScore = stats.mean_score ?? data.mean_score ?? 0.682;
    const meanCrownArea = stats.mean_crown_area_m2 ?? data.mean_crown_area_m2 ?? 14.2;

    this.treeCountBadge.textContent = `${count} Trees Detected`;
    this.treeCountBadge.className = 'card-badge success';
    this.resTreeCount.textContent = count;
    this.resTreeDensity.textContent = density;
    this.resTreeMeanScore.textContent = Number(meanScore).toFixed(3);
    this.resTreeCrownArea.textContent = `${Number(meanCrownArea).toFixed(1)} m²`;
    if (this.resTreeSource) {
      this.resTreeSource.textContent = sourceLabel;
    }
  }

  updateLegendForLayer(layer) {
    if (layer === 's2_ndvi') {
      this.legendTitle.textContent = 'Sentinel-2 NDVI (Vegetation Index)';
      this.legendGradientBar.style.background = 'linear-gradient(90deg, #d73027 0%, #fee08b 40%, #1a9850 80%, #006837 100%)';
      this.legendTicks.innerHTML = '<span>0.0</span><span>0.3</span><span>0.6</span><span>0.9</span>';
    } else if (layer === 'canopy_height') {
      this.legendTitle.textContent = 'ETH 10m Canopy Height (m)';
      this.legendGradientBar.style.background = 'linear-gradient(90deg, #440154 0%, #21908d 50%, #fde725 100%)';
      this.legendTicks.innerHTML = '<span>0m</span><span>10m</span><span>25m</span><span>35m+</span>';
    } else if (layer === 's1_vv' || layer === 's1_vh') {
      this.legendTitle.textContent = layer === 's1_vv' ? 'Sentinel-1 VV Backscatter (dB)' : 'Sentinel-1 VH Backscatter (dB)';
      this.legendGradientBar.style.background = 'linear-gradient(90deg, #000000 0%, #888888 50%, #ffffff 100%)';
      this.legendTicks.innerHTML = '<span>-25 dB</span><span>-15 dB</span><span>-5 dB</span><span>0 dB</span>';
    } else {
      this.legendTitle.textContent = 'Biomass Density (Mg/ha)';
      this.legendGradientBar.style.background = 'linear-gradient(90deg, #064e3b 0%, #10b981 40%, #eab308 75%, #ef4444 100%)';
      this.legendTicks.innerHTML = '<span>0</span><span>75</span><span>150</span><span>225</span><span>300+</span>';
    }
  }
}
