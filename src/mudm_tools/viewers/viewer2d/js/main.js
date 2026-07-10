import { asyncBufferFromUrl, parquetMetadataAsync, parquetReadObjects } from "./vendor/hyparquet.min.js";

function _cellExprRamp(t){
  t = Math.max(0, Math.min(1, Math.cbrt(t)));
  // jet colormap: dark-blue -> blue -> cyan -> green -> yellow -> red -> dark-red
  const s=[[0,0,143],[0,0,255],[0,255,255],[0,255,0],[255,255,0],[255,0,0],[128,0,0]];
  const f=t*(s.length-1), i=Math.min(Math.floor(f), s.length-2), g=f-i, a=s[i], b=s[i+1];
  return 'rgb('+Math.round(a[0]+(b[0]-a[0])*g)+','+Math.round(a[1]+(b[1]-a[1])*g)+','+Math.round(a[2]+(b[2]-a[2])*g)+')';
}
const _CELL_EXPR_MAX = 300;  // total_counts normalization for the cbrt ramp (breast cells ~10-300)
// legacy inline fallback: pull one gene's count out of a sparse {"GENE":n,...} expression tag (no full parse)
function _geneCountInline(exprStr, g){
  if (!exprStr || !g) return 0;
  const key = '"' + g + '":';
  const i = exprStr.indexOf(key);
  if (i < 0) return 0;
  let j = i + key.length, n = '';
  while (j < exprStr.length) { const ch = exprStr[j]; if (ch >= '0' && ch <= '9') { n += ch; j++; } else break; }
  return n ? +n : 0;
}
function _facetParquetHref(facets, facetName){
  const a = ((facets && facets.assets) || []).find(x => x.facet === facetName && String(x.media_type||'').indexOf('parquet') >= 0);
  return a ? a.href : null;
}
function _statToStr(v){
  if (v == null) return null;
  if (typeof v === 'string') return v;
  if (v instanceof Uint8Array) return new TextDecoder().decode(v);
  return String(v);
}
// Range-read one gene's (cell_id -> count) from facets/expression.parquet. The file is sorted by `gene`,
// so we read ONLY the row-group span whose [min,max] gene stats include the target (full read if stats absent).
async function _facetGeneMap(datasetId, facets, gene){
  window._facetCache = window._facetCache || {};
  if (window._facetCache[gene]) return window._facetCache[gene];
  const href = _facetParquetHref(facets, 'gene');
  const m = new Map();
  if (!href || !gene) return m;
  const url = `${BASE_URL}/tiles2d/${datasetId}/${href}`;
  const file = await asyncBufferFromUrl({ url });
  const md = await parquetMetadataAsync(file);
  const rgs = md.row_groups || [];
  let cum = 0, rowStart = -1, rowEnd = -1;
  for (const rg of rgs){
    const n = Number(rg.num_rows);
    const col = (rg.columns||[]).find(c => (((c.meta_data && c.meta_data.path_in_schema)||[]).join('.')) === 'gene');
    const st = col && col.meta_data && col.meta_data.statistics;
    const lo = st ? _statToStr(st.min_value != null ? st.min_value : st.min) : null;
    const hi = st ? _statToStr(st.max_value != null ? st.max_value : st.max) : null;
    const hit = (lo == null || hi == null) ? true : (gene >= lo && gene <= hi);
    if (hit){ if (rowStart < 0) rowStart = cum; rowEnd = cum + n; }
    cum += n;
  }
  if (rowStart < 0){ window._facetCache[gene] = m; return m; }
  const rows = await parquetReadObjects({ file, metadata: md, columns: ['cell_id','gene','count'], rowStart, rowEnd });
  for (const r of rows){ if (r.gene === gene) m.set(String(r.cell_id), Number(r.count)); }
  window._facetCache[gene] = m;
  return m;
}
// Locate the WIDE-layout facet asset (one float column per marker, keyed by cell_id). Selected by the
// asset's `layout`, not by a facet name (a wide store has no per-row `facet`/`gene` discriminator column).
function _facetWideHref(facets){
  const a = ((facets && facets.assets) || []).find(x => x.layout === 'wide');
  return a ? a.href : null;
}
// Wide layout: read ONE marker COLUMN (+ the cell_id join column ONCE, cached) and join by cell_id. Reading a
// single column keeps the over-the-wire cost ~1 column even on a many-marker store; the column-store BSS
// encoding means each marker is its own contiguous span. Cached per marker (window._facetCache[marker]).
async function _facetMarkerMap(datasetId, facets, marker){
  window._facetCache = window._facetCache || {};
  if (window._facetCache[marker]) return window._facetCache[marker];
  const href = _facetWideHref(facets);
  const m = new Map();
  if (!href || !marker) return m;
  const url = `${BASE_URL}/tiles2d/${datasetId}/${href}`;
  const file = await asyncBufferFromUrl({ url });
  const md = await parquetMetadataAsync(file);
  if (!window._facetCellIds){  // read the cell_id join column ONCE, reuse for every marker
    const idr = await parquetReadObjects({ file, metadata: md, columns: ['cell_id'] });
    window._facetCellIds = idr.map(r => String(r.cell_id));
  }
  const col = await parquetReadObjects({ file, metadata: md, columns: [marker] });
  for (let i = 0; i < col.length; i++) m.set(window._facetCellIds[i], Number(col[i][marker]));
  window._facetCache[marker] = m;
  return m;
}
// Prefetch the WHOLE wide store (small datasets only — gated on facets.prefetch). After this, every marker
// switch is purely client-side (read window._facetWide.get(cid)[marker]); no further network reads.
async function _facetPrefetchAll(datasetId, facets){
  if (window._facetWide) return;
  const href = _facetWideHref(facets);
  if (!href) return;
  const url = `${BASE_URL}/tiles2d/${datasetId}/${href}`;
  const file = await asyncBufferFromUrl({ url });
  const md = await parquetMetadataAsync(file);
  const rows = await parquetReadObjects({ file, metadata: md });  // all columns, once
  window._facetWide = new Map(rows.map(r => [String(r.cell_id), r]));
}
// Wide-store helper: build the selected marker's (cell_id -> value) map. When the whole store is prefetched,
// this is a client-side projection of window._facetWide; otherwise it range-reads the single marker column.
async function _facetWideMap(datasetId, facets, marker){
  if (!marker) return new Map();
  if (facets && facets.prefetch){
    await _facetPrefetchAll(datasetId, facets);
    const m = new Map();
    if (window._facetWide){ window._facetWide.forEach(function(row, cid){ m.set(cid, Number(row[marker])); }); }
    return m;
  }
  return _facetMarkerMap(datasetId, facets, marker);
}
window._facetDebug = (function(){ try { return new URLSearchParams(location.search).has('debug'); } catch(_){ return false; } })();
// Build the muDM Feature JSON for a hovered cell AS IT IS DEFINED: type/id/geometry + the inline scalar
// PROPERTIES (storage: inline) + the declared facets as DEFERRED REFERENCES ($ref to the Asset, keyed by the
// feature id). Nothing is resolved/inlined here — the deferral is the point, and showing it as a reference is
// both faithful and instant (reads props + the declared schema; no store fetch). Resolve a value via the $ref
// (e.g. DuckDB `WHERE <key>=<keyValue>`); the active gene's value is also shown dereferenced on the line above.
function _muDMFeatureJSON(props){
  const f = metadata.facets, cid = String(props.cell_id);
  const inline = {}; for (const k in props){ if (k !== 'layer_type') inline[k] = props[k]; }  // storage: inline
  const facets = {};
  (f.assets || []).forEach(function(a){
    const vocab = (f.fieldenums && f.fieldenums[a.facet]) || null;
    facets[a.facet] = {
      $ref: `${BASE_URL}/tiles2d/${window._facetDatasetId}/${a.href}`,
      mediaType: a.media_type, storage: f.storage,
      key: a.key || f.key, keyValue: cid, layout: a.layout,
      vocabulary: vocab ? { field: a.facet, size: vocab.length, declaredIn: 'TileLayer.fieldenums' } : undefined
    };
  });
  return { type: 'Feature', id: props.cell_id, featureClass: 'cell',
           geometry: { type: 'Polygon', '$tiled': `${metadata.vectors.path} + features.parquet (muDM binary tile geometry)` },
           properties: inline, facets: facets };
}
// WIDE-facet per-cell hover: the markers were STRIPPED from the tiles (surviving tile props are just
// cell_id + cell_type), so the full per-cell marker vector is read from the PREFETCHED wide store
// (window._facetWide, a cid -> {marker: value, …} Map populated by _facetPrefetchAll). We emit ONE summary
// property — every marker value, value-sorted high->low (like the old inline m_* panel) — restricted to the
// declared marker columns (Object.keys(metadata.facets.fields)). Returns true when it augmented `out`.
function _withFacetMarkers(out, cid){
  const f = metadata.facets;
  if (!f || f.layout !== 'wide' || !window._facetWide) return false;
  const row = window._facetWide.get(cid);
  if (!row) return false;
  const names = Object.keys(f.fields || {});
  const _fmt = function(v){ v = Number(v); return Math.abs(v) >= 1 ? v.toFixed(2) : Number(v.toPrecision(2)); };
  const pairs = names.map(function(k){ return [k, Number(row[k])]; })
                     .filter(function(p){ return isFinite(p[1]); })
                     .sort(function(a, b){ return b[1] - a[1]; });  // value-sorted high->low
  if (!pairs.length) return false;
  out['markers (facet)'] = pairs.map(function(p){ return p[0] + ' ' + _fmt(p[1]); }).join('  ·  ');
  return true;
}
// Surface facet data on HOVER. ONLY for cells of a facet dataset; non-facet datasets (CODEX/CosMx markers,
// transcripts, …) and non-cell features pass through unchanged — their per-object data is inline in the tile.
// LONG facets (Xenium): the selected gene's value is dereferenced inline (quick readout — never hundreds of
// genes). WIDE facets (CODEX/CosMx markers): the whole per-cell marker vector is shown from the prefetched
// store (the markers are no longer in the tile). ?debug=1 also shows the full muDM Feature JSON with the
// facets AS DEFERRED REFERENCES (instant — no whole-store load).
function _withFacets(props){
  const f = metadata.facets;
  if (!f || !props || props.cell_id == null || props.layer_type !== f.layer) return props;  // gated: non-facet → unchanged
  const cid = String(props.cell_id);
  const out = Object.assign({}, props);
  if (f.layout === 'wide'){
    _withFacetMarkers(out, cid);                      // WIDE: full per-cell marker vector from the prefetched store
  } else {
    const g = window._exprGene;                       // LONG: only the selected gene's value (no gene dump)
    if (g && window._facetMap){ const v = window._facetMap.get(cid); out[g + ' (facet)'] = (v == null ? 0 : v); }
  }
  if (window._facetDebug){ out['muDM (debug)'] = JSON.stringify(_muDMFeatureJSON(props)); }
  return out;
}
function _setupCellExpressionLayer(datasetId, imageBounds, maxZoom){
  const facets = metadata.facets;
  const legacyInline = !facets && metadata.expression_lod && metadata.expression_lod.enabled;
  if (!facets && !legacyInline) return;
  window._facetDatasetId = datasetId;  // used to build the absolute $ref href in the ?debug=1 feature JSON
  const _layer = (facets && facets.layer) || (metadata.expression_lod && metadata.expression_lod.layer) || 'cells';
  const _store = !!facets && facets.storage !== 'inline';   // true => parquet store-join; false => inline tag
  // WIDE facets (CODEX/CosMx/merscope): markers live as one float column each in the wide store; metadata
  // .facets.fields keys ARE the marker names. The categorical #gene-filter holds CELL TYPES (gene_list.json),
  // not markers, so wide datasets get a SEPARATE marker dropdown to drive the expression overlay (below);
  // #gene-filter keeps driving the cell-type coloring. LONG facets (Xenium): #gene-filter == the gene panel.
  const _isWide = !!(facets && facets.layout === 'wide');
  const _hide = { fill:false, stroke:false, weight:0, fillOpacity:0, opacity:0, radius:0 };
  const _dim = { fill:true, fillColor:'#1a1a1a', fillOpacity:0.12, weight:0, stroke:false };
  if (window._geneMax == null) window._geneMax = 1;
  window._exprGene = window._exprGene || '';

  function _cellColor(props){
    const cid = props.cell_id != null ? String(props.cell_id) : null;
    const g = window._exprGene;
    if (g){  // selected-gene mode
      let v;
      if (_store){ v = (window._facetMap && cid != null) ? window._facetMap.get(cid) : undefined; }
      else { v = props.expression != null ? _geneCountInline(props.expression, g) : undefined; }
      if (v == null || v === 0) return _dim;
      return { fill:true, fillColor:_cellExprRamp(v / window._geneMax), fillOpacity:0.9, weight:0, stroke:false };
    }
    if (props.total_counts == null) return _hide;  // all-genes: color by the inline total_counts scalar
    return { fill:true, fillColor:_cellExprRamp((+props.total_counts) / _CELL_EXPR_MAX), fillOpacity:0.85, weight:0, stroke:false };
  }

  const _styles = {};
  (metadata.vectors.layers || []).forEach(function(ld){
    _styles[ld.id] = (ld.id === _layer) ? _cellColor : function(){ return _hide; };
  });
  // Overlay tile-request floor: honor the faceted layer's declared min_zoom (e.g. merscope cells=5), so
  // enabling Cell Expression on a huge dataset doesn't brute-force its heavy low-zoom tiles. Floor at 3
  // (the overlay's prior default) so normal datasets are unchanged. Matches the base grid's gridMinZoom.
  // BUT clamp to maxZoom: a small dataset (e.g. 2.5D nuclei, maxZoom=2) has no zoom>=3, so an unclamped
  // floor of 3 makes minZoom>maxZoom -> the overlay requests ZERO tiles and never paints. Clamping to
  // maxZoom lets small sets load the overlay at their finest zoom while leaving big sets at the floor.
  const _ovFloor = Math.max(3, (((metadata.vectors.layers || []).find(function(l){ return l.id === _layer; }) || {}).min_zoom) || 0);
  const _ovMinZoom = Math.min(maxZoom, _ovFloor);
  window._cellExprGrid = L.vectorGrid.protobuf(
    _vectorUrl(datasetId, metadata),
    // getFeatureId populates tile._features so the join-by-cell_id recolor can repaint.
    { vectorTileLayerStyles: _styles, interactive:false, rendererFactory: L.canvas.tile, maxZoom: maxZoom, minZoom: _ovMinZoom, bounds: imageBounds,
      getFeatureId: function(f){ return (f.properties && f.properties.cell_id != null) ? 'ce_' + f.properties.cell_id : null; } }
  );

  // Fetch the selected gene's (cell_id->count) map BEFORE (re)rendering, so the overlay paints correct colors
  // in a SINGLE pass. Fetching first is what kills the flicker: adding the layer and THEN redraw()ing tears
  // every tile down and reloads it (a blank frame). 'All genes' needs no fetch (uses the inline total_counts).
  async function _ensureData(){
    if (window._exprGene){
      if (_store){
        // Wide layout (markers): read one marker column (or project the prefetched whole store) and join by
        // cell_id. Long layout (Xenium genes): range-read facets/expression.parquet by the sorted `gene` column.
        if (facets && facets.layout === 'wide'){
          window._facetMap = await _facetWideMap(datasetId, facets, window._exprGene);
        } else {
          window._facetMap = await _facetGeneMap(datasetId, facets, window._exprGene);
        }
        let mx = 1; window._facetMap.forEach(function(v){ if (v > mx) mx = v; }); window._geneMax = mx;
      } else {
        window._geneMax = _geneMaxInline(window._exprGene);
      }
    }
  }
  // Gene change while the overlay is already on: fetch the new gene, THEN repaint once.
  async function _recolor(){
    if (!(window._cellExprGrid && map.hasLayer(window._cellExprGrid))) return;
    await _ensureData();
    if (window._cellExprGrid && map.hasLayer(window._cellExprGrid)) window._cellExprGrid.redraw();
  }
  function _geneMaxInline(g){ let mx=1; if(!g) return mx;
    const vts=(window._cellExprGrid && window._cellExprGrid._vectorTiles)||{};
    for (const k in vts){ const fs=vts[k]._features||{}; for (const id in fs){ const p=fs[id].feature&&fs[id].feature.properties;
      if(p&&p.expression){ const v=_geneCountInline(p.expression,g); if(v>mx)mx=v; } } } return mx; }

  const _gf = document.getElementById('gene-filter');
  // LONG facets only: cell coloring follows the gene filter (a selected gene -> that gene's per-cell count,
  // range-read from the store; nothing selected -> the inline total_counts). For WIDE facets the #gene-filter
  // is the CELL-TYPE selector — it must NOT touch window._exprGene (a dedicated marker dropdown does that).
  if (!_isWide && _gf) _gf.addEventListener('change', function(){ window._exprGene = _gf.value || ''; _recolor(); });

  // WIDE facets: a DEDICATED marker dropdown (NOT the cell-type #gene-filter) drives the expression overlay.
  // Populated from the wide store's field names (== markers); on change the chosen marker becomes the active
  // _exprGene and the existing recolor runs (reusing _facetWideMap/_facetPrefetchAll). The whole wide store is
  // also prefetched eagerly so the per-cell marker HOVER (_withFacetMarkers) finds window._facetWide populated.
  let _mf = null;
  if (_isWide){
    const _names = Object.keys(facets.fields || {}).slice().sort();
    const _panel = document.getElementById('layer-panel');
    if (_names.length && _panel){
      const _div = document.createElement('div'); _div.className = 'panel-divider'; _panel.appendChild(_div);
      const _t = document.createElement('div'); _t.className = 'panel-title'; _t.textContent = 'Marker (Cell Expression)';
      _panel.appendChild(_t);
      _mf = document.createElement('select'); _mf.id = 'marker-filter';
      _mf.innerHTML = '<option value="">Select marker…</option>';
      _names.forEach(function(n){ const o = document.createElement('option'); o.value = n; o.textContent = n; _mf.appendChild(o); });
      const _markerFilterChange = function(){ window._exprGene = _mf.value || ''; _recolor(); };
      _mf.addEventListener('change', _markerFilterChange);
      _panel.appendChild(_mf);
    }
    if (facets.prefetch) _facetPrefetchAll(datasetId, facets);  // eager: hover is synchronous, store must be ready
  }
  const _exprSource = function(){ return (_isWide ? (_mf && _mf.value) : (_gf && _gf.value)) || ''; };

  const _tc = document.getElementById('layer-toggles');
  if (!_tc) return;
  const _w = document.createElement('label'); _w.className = 'layer-toggle';
  const _cb = document.createElement('input'); _cb.type = 'checkbox'; _cb.checked = false;
  _cb.addEventListener('change', async function(){
    if (_cb.checked) {
      window._exprGene = _exprSource();                  // WIDE: the marker dropdown; LONG: the gene filter
      await _ensureData();                               // fetch FIRST -> overlay renders once, correctly (no redraw flicker)
      if (_cb.checked) window._cellExprGrid.addTo(map);  // re-check: user may have toggled off during the fetch
    } else {
      map.removeLayer(window._cellExprGrid);
    }
  });
  const _sw = document.createElement('span'); _sw.className = 'layer-swatch'; _sw.style.background = 'linear-gradient(90deg,#00007f,#0000ff,#00ffff,#00ff00,#ffff00,#ff0000,#7f0000)';
  _w.appendChild(_cb); _w.appendChild(_sw); _w.appendChild(document.createTextNode('Cell Expression'));
  _tc.appendChild(_w);
}
function _rasterOf(m){
  if (m && m.raster) return m.raster;
  const b = (m && m.bounds_um) || [0,0,1,1]; const upp = (m && m.um_per_px) || 1;
  let mz = 0; (((m && m.vectors && m.vectors.layers) || [])).forEach(function(l){ mz = Math.max(mz, l.max_zoom||0); });
  return {image_size_px:[Math.ceil((b[2]-b[0])/upp), Math.ceil((b[3]-b[1])/upp)], max_zoom:mz, tile_size:256, path:null};
}
// viewer2d/js/main.js
import { LayerPanel } from "./LayerPanel.js";
import { InfoPanel } from "./InfoPanel.js";
import { loadDescriptor } from "./descriptor.js";
import { TileLoadCounter } from "./TileLoadCounter.mjs";
import { LoadingIndicator } from "./LoadingIndicator.js";

let map;
let metadata;
let vectorGridLayer;
let rasterLayer;
let hiddenLayers = new Set();
let hoveredFeatureId = null;
let geneColorMap = null;  // { gene_name → hex color }
let currentDatasetId = null;

// --- Tile-loading indicator (non-blocking) ---
const _tileCounter = new TileLoadCounter();
const _tileIndicator = new LoadingIndicator(document.getElementById("tile-loading"));
window._tileLoadCounter = _tileCounter;      // exposed for the Playwright gate
window._tileLoadIndicator = _tileIndicator;
let _tilesRequested = 0, _tilesResolved = 0;
const _loadingLayers = new Set();            // GridLayers still mid-load
function _pushLoading() {
    _tileIndicator.update(_tileCounter.setOutstanding(Math.max(0, _tilesRequested - _tilesResolved)));
}
function _resetLoading() {
    _tilesRequested = 0; _tilesResolved = 0; _loadingLayers.clear();
    _tileIndicator.update(_tileCounter.reset());
}
// Aggregate GridLayer tile events (raster + vector) into ONE outstanding count. `tileerror`
// counts as resolved so an aborted tile can't strand the pill. When every wired layer has
// fired `load` (fully idle) we zero the tallies to reconcile any tiles dropped without a
// tileload/tileerror (fast pan-away).
function _wireTileCounting(layer) {
    layer.on("loading", () => {
        if (layer !== rasterLayer && layer !== vectorGridLayer) return;
        _loadingLayers.add(layer);
    });
    layer.on("tileloadstart", () => {
        if (layer !== rasterLayer && layer !== vectorGridLayer) return;
        _tilesRequested++; _pushLoading();
    });
    layer.on("tileload", () => {
        if (layer !== rasterLayer && layer !== vectorGridLayer) return;
        _tilesResolved++; _pushLoading();
    });
    layer.on("tileerror", () => {
        if (layer !== rasterLayer && layer !== vectorGridLayer) return;
        _tilesResolved++; _pushLoading();
    });
    layer.on("load", () => {
        if (layer !== rasterLayer && layer !== vectorGridLayer) return;
        _loadingLayers.delete(layer);
        if (_loadingLayers.size === 0) { _tilesRequested = 0; _tilesResolved = 0; }
        _pushLoading();
    });
}

const BASE_URL = "";

// --- 2.5D multichannel + Z-slice controls (inert unless metadata.depth + metadata.rasters) ---
function _sliceTok() { return 'z' + String(window.__curZ || 0).padStart(3, '0'); }
function _rasterUrl(id, m) {
    if (m && m.rasters && m.rasters.length) {
        var ci = (window.__curCh | 0); if (ci < 0 || ci >= m.rasters.length) ci = 0;
        return `${BASE_URL}/tiles2d/${id}/` + m.rasters[ci].path.replace('{d}', _sliceTok());
    }
    return `${BASE_URL}/tiles2d/${id}/raster/{z}/{x}/{y}.png`;
}
function _vectorUrl(id, m) {
    var p = (m && m.vectors && m.vectors.path) || 'vectors/{z}/{x}/{y}.pbf';
    return `${BASE_URL}/tiles2d/${id}/` + (p.indexOf('{d}') >= 0 ? p.replace('{d}', _sliceTok()) : p);
}
function _init25DControls(m, map, id) {
    window._map = map;                          // debug / Playwright hook (all datasets)
    if (!m || !m.depth || !m.rasters) return;   // no-op for normal single-plane 2D datasets
    // Small-image edge case: _clamp_min_zoom_to_data can leave minZoom > maxZoom (a small image's fit
    // zoom exceeds the raster native max) → blank map. Give overzoom headroom so tiles render + zoom-in works.
    // Small-image fix: _clamp_min_zoom_to_data leaves the map at a fractional fit-zoom (e.g. 2.15) that
    // exceeds the native tile max. That makes tile grids hide (zoom > their maxZoom) OR request a fractional
    // {z} (→404) — including the lazily-created Cell-Expression overlay grid. Snap the map to the INTEGER
    // native zoom and cap zoom-in there so EVERY grid (raster, outlines, cell-expr) requests integer tiles.
    var _nat = Math.max(0, Math.round(map.getMinZoom()));
    // Finest zoom the tiles ACTUALLY support (from metadata), NOT the fit zoom. For a small image the fit
    // ≈ the native max so capping zoom-in at the fit was harmless; for a large one (merscope 89k×61k,
    // max_zoom 9) the fit is ~3 and capping there wrongly blocked zoom-in. Cap zoom-in at the data max.
    var _dmax = 0;
    (m.rasters || []).forEach(function (r) { _dmax = Math.max(_dmax, r.max_zoom || 0); });
    (((m.vectors && m.vectors.layers) || [])).forEach(function (l) { _dmax = Math.max(_dmax, l.max_zoom || 0); });
    if (!_dmax) { _dmax = _nat; }
    map.setMinZoom(Math.max(0, _nat - 2));      // allow zoom-out (padding) in integer steps
    map.setMaxZoom(_dmax);                       // allow zoom-IN to the finest available tiles
    map.setZoom(Math.min(_nat, _dmax));          // start framed at the fit (never past the data max)
    var nuc = m.rasters.findIndex(function (r) { return String(r.id) === '405'; });
    window.__curCh = nuc >= 0 ? nuc : 0;                       // default to the nuclear (405) channel
    window.__curZ = Math.floor((m.depth.slice_count || 1) / 2); // start mid-stack (dense tissue)
    var refresh = function () {
        if (rasterLayer) rasterLayer.setUrl(_rasterUrl(id, m));
        if (vectorGridLayer && vectorGridLayer.setUrl) vectorGridLayer.setUrl(_vectorUrl(id, m));
        if (window._cellExprGrid && window._cellExprGrid.setUrl) window._cellExprGrid.setUrl(_vectorUrl(id, m));
    };
    var ctrl = L.control({ position: 'topleft' });   // topleft (under zoom); bottomleft is the InfoPanel
    ctrl.onAdd = function () {
        var d = L.DomUtil.create('div', 'zslice-ctrl');
        d.style.cssText = 'background:rgba(0,0,0,0.65);color:#fff;padding:6px 8px;border-radius:4px;font:12px sans-serif;';
        var opts = m.rasters.map(function (r, i) {
            return '<option value="' + i + '"' + (i === window.__curCh ? ' selected' : '') + '>' + r.name + '</option>';
        }).join('');
        d.innerHTML = 'Channel <select id="ch-sel">' + opts + '</select><br>' +
            'Z <input id="z-sl" type="range" min="0" max="' + ((m.depth.slice_count || 1) - 1) +
            '" value="' + window.__curZ + '" style="vertical-align:middle"> <span id="z-lab">' + window.__curZ + '</span>';
        L.DomEvent.disableClickPropagation(d);
        return d;
    };
    ctrl.addTo(map);
    document.getElementById('ch-sel').addEventListener('change', function (e) { window.__curCh = +e.target.value; refresh(); });
    document.getElementById('z-sl').addEventListener('input', function (e) {
        window.__curZ = +e.target.value; document.getElementById('z-lab').textContent = e.target.value; refresh();
    });
    refresh();  // apply the mid-stack / 405 defaults
}


async function loadDatasets() {
    const resp = await fetch(`${BASE_URL}/tiles2d/datasets.json`);
    const datasets = await resp.json();
    const select = document.getElementById("dataset-select");
    select.innerHTML = "";
    datasets.forEach((ds) => {
        const opt = document.createElement("option");
        opt.value = ds.id;
        opt.textContent = ds.name;
        select.appendChild(opt);
    });
    // Honor ?dataset=<id> (catalogue deep-link) as the initial load; hide the dropdown when
    // deep-linked (selection is the linking page's job). No/unknown ?dataset= → first dataset +
    // dropdown stays for standalone browsing. loadDataset() guards duplicate ids, so the deep-link
    // shim's 'change' is a no-op — no race / double-load.
    select.addEventListener("change", () => loadDataset(select.value));
    if (datasets.length > 0) {
        const wantedId = new URLSearchParams(location.search).get("dataset");
        const wanted = wantedId ? datasets.find((d) => d.id === wantedId) : null;
        const initial = wanted || datasets[0];
        select.value = initial.id;
        if (wanted) {
            const box = document.getElementById("dataset-selector");
            if (box) box.style.display = "none";
        }
        await loadDataset(initial.id);
    }
}

async function loadDataset(datasetId) {
    if (datasetId === currentDatasetId) return;  // ignore duplicate re-selects (deep-link shim's 'change')
    currentDatasetId = datasetId;
    _resetLoading();  // new dataset → drop any stale in-flight tallies
    // Consume the muDM TileModel (tilejson.json); metadata.json is a transition fallback. `vm.meta` is
    // the metadata-shaped view the raster/2.5D/facet machinery reads — one descriptor, one fetch.
    const vm = await loadDescriptor(BASE_URL, datasetId);
    metadata = vm.meta;

    if (rasterLayer) { map.removeLayer(rasterLayer); }
    if (vectorGridLayer) { map.removeLayer(vectorGridLayer); }
    hiddenLayers.clear();
    hoveredFeatureId = null;

    // Load gene colormap if available
    geneColorMap = null;
    try {
        const cmResp = await fetch(`${BASE_URL}/tiles2d/${datasetId}/gene_colormap.json`);
        if (cmResp.ok) {
            const cm = await cmResp.json();
            geneColorMap = {};
            for (const [catName, catDef] of Object.entries(cm.categories)) {
                for (const gene of catDef.genes) {
                    geneColorMap[gene] = catDef.color;
                }
            }
            geneColorMap._default = cm.default_color || "#888888";
            geneColorMap._categories = cm.categories;
        }
    } catch (_) {}

    const umPerPx = metadata.um_per_px;
    const [imgW, imgH] = _rasterOf(metadata).image_size_px;
    const rasterMaxZoom = _rasterOf(metadata).max_zoom;

    const southWest = map.unproject([0, imgH], rasterMaxZoom);
    const northEast = map.unproject([imgW, 0], rasterMaxZoom);
    const imageBounds = L.latLngBounds(southWest, northEast);

    const [bxMin, byMin, bxMax, byMax] = metadata.bounds_um;
    const dataSW = map.unproject([bxMin / umPerPx, byMax / umPerPx], rasterMaxZoom);
    const dataNE = map.unproject([bxMax / umPerPx, byMin / umPerPx], rasterMaxZoom);
    const dataBounds = L.latLngBounds(dataSW, dataNE);

    const vectorMaxZoom = Math.max(...metadata.vectors.layers.map(l => l.max_zoom));
    rasterLayer = L.tileLayer(
        _rasterUrl(datasetId, metadata),
        {
            minZoom: 0,
            maxNativeZoom: rasterMaxZoom,
            maxZoom: vectorMaxZoom,
            tileSize: _rasterOf(metadata).tile_size || 256,
            noWrap: true,
            bounds: imageBounds,
        }
    );
    _wireTileCounting(rasterLayer);
    rasterLayer.addTo(map);

    map.fitBounds(dataBounds);
    // muDM-min-zoom: frame the tissue at the fractional data-fit zoom + floor minZoom there, so
    // zoom-out can't reveal the empty black margin of the square tile world around the slide.
    // Compute the fractional fit by briefly disabling zoomSnap, then RESTORE it (integer zoom
    // interaction — leaving zoomSnap:0 on would thrash/flicker the canvas expression overlay).
    const _zs = map.options.zoomSnap; map.options.zoomSnap = 0;
    const _fitZoom = map.getBoundsZoom(dataBounds); map.options.zoomSnap = _zs;
    if (isFinite(_fitZoom)) { map.setMinZoom(_fitZoom); map.setZoom(_fitZoom); }
    map.setMaxBounds(dataBounds.pad(0.2));
    map.setMaxZoom(Math.max(rasterMaxZoom, vectorMaxZoom));  // muDM-cap-maxzoom: block zoom past available tiles

    setupVectorLayer(datasetId, imageBounds, vectorMaxZoom);
    LayerPanel.init(metadata, rasterLayer, vectorGridLayer, hiddenLayers, datasetId);
    _init25DControls(metadata, map, datasetId);
    _setupCellExpressionLayer(datasetId, imageBounds, vectorMaxZoom);  // facet-store join overlay + toggle
    _updateZoomUI();  // initialize the zoom readout + per-layer badges for the freshly-built panel
}

// Reflect the current tile zoom in the layer panel: a "Zoom z / max" readout plus per-layer presence
// badges that grey out when the current zoom is below a layer's min_zoom (i.e. it isn't rendered yet).
// DOM-driven (reads elements LayerPanel builds), so a single persistent zoomend handler covers every
// dataset. `Math.round(map.getZoom())` is the tile z VectorGrid actually requests (== `coords.z`), so
// the badge active-state matches the display gate exactly.
function _updateZoomUI() {
    if (!map || !metadata) return;
    const z = Math.round(map.getZoom());
    const rmax = (_rasterOf(metadata) && _rasterOf(metadata).max_zoom) ||
        Math.max(...metadata.vectors.layers.map((l) => l.max_zoom || 0));
    const readout = document.getElementById("zoom-readout");
    if (readout) readout.textContent = `Zoom ${z} / ${rmax}`;
    document.querySelectorAll(".layer-badge").forEach((b) => {
        const mz = parseInt(b.dataset.minZoom || "0", 10);
        const active = z >= mz;
        b.style.opacity = active ? "1" : "0.4";
        b.style.color = active ? "#63b3ed" : "#718096";
    });
}

function setupVectorLayer(datasetId, imageBounds, maxZoom) {
    const layerLookup = {};
    for (const layerDef of metadata.vectors.layers) {
        layerLookup[layerDef.id] = layerDef;
    }

    const styles = {};
    const geneSelect = document.getElementById("gene-filter");
    for (const layerDef of metadata.vectors.layers) {
        styles[layerDef.id] = function (properties, zoom) {
            // Per-layer display floor: below the layer's declared min_zoom, skip the feature entirely.
            // Returning [] makes VectorGrid `continue` (no path created/painted) — cheaper than an
            // invisible style, and it makes each layer's metadata min_zoom AUTHORITATIVE for display
            // (e.g. transcripts wait for max_zoom-1) independent of what the tiler baked into low-zoom
            // tiles. `zoom` is the TILE's z, so this is robust to the fractional map zoom set by the
            // min-zoom-clamp patch.
            if (zoom < (layerDef.min_zoom || 0)) {
                return [];
            }
            if (hiddenLayers.has(layerDef.id)) {
                return { opacity: 0, fillOpacity: 0, radius: 0, weight: 0 };
            }
            // _muDMpolyfilter: hide non-matching cells (polygons) by cell_type — ONLY for datasets
            // whose cells carry a category (CODEX/segmentation). A category-less cell (e.g. Xenium,
            // where the dropdown is a GENE filter for transcripts) must be left untouched, else the
            // gene filter wrongly blanks the cell boundaries.
            if (layerDef.type === "polygon" && geneSelect.value) {
                const _cv = properties.cell_type || properties.cluster || properties.class || properties.category;
                if (_cv != null && _cv !== geneSelect.value) {
                    return { opacity: 0, fillOpacity: 0, radius: 0, weight: 0 };
                }
            }
            // Gene filter: hide non-matching transcripts
            if (layerDef.type === "point" && geneSelect.value) {
                if (properties.gene_name !== geneSelect.value) {
                    return { opacity: 0, fillOpacity: 0, radius: 0, weight: 0 };
                }
            }
            // Color-by-gene for transcripts
            if (layerDef.type === "point" && geneColorMap && properties.gene_name) {
                // Check category filter
                if (window._hiddenGeneCategories && window._hiddenGeneCategories.size > 0) {
                    const cat = (window._geneCategoryMap && window._geneCategoryMap[properties.gene_name]) || "Other";
                    if (window._hiddenGeneCategories.has(cat)) {
                        return { opacity: 0, fillOpacity: 0, radius: 0, weight: 0 };
                    }
                }
                const color = geneColorMap[properties.gene_name] || geneColorMap._default;
                return {
                    radius: 5, weight: 1,
                    color: color, fillColor: color,
                    fillOpacity: 0.7, fill: true,
                    interactive: true,
                };
            }
            if (geneColorMap) {
                const _muDMcat = properties.cell_type || properties.cluster || properties.class || properties.category || properties.gene_name;
                if (_muDMcat && geneColorMap[_muDMcat]) {
                    if (window._hiddenGeneCategories && window._hiddenGeneCategories.has(_muDMcat)) {
                        return { opacity: 0, fillOpacity: 0, radius: 0, weight: 0 };
                    }
                    const _c = geneColorMap[_muDMcat];
                    return { color: _c, fillColor: _c, weight: 1, fillOpacity: 0.45, fill: true, radius: 4, interactive: true };
                }
            }
            return getLayerStyle(layerDef);
        };
    }

    // Tile-request floor: don't fetch vector tiles below the EARLIEST layer's declared min_zoom.
    // Keeps full-fidelity geometry (no simplification) but stops the viewer brute-force-loading the
    // heavy low-zoom tiles of huge datasets (e.g. MERSCOPE's ~5.5 gigapixel image → 11-21 MB z2 cell
    // tiles); the raster is the overview there and full-detail cells load on zoom-in. Default 0 (=
    // unchanged) for the common case where a layer is visible from z0.
    const gridMinZoom = Math.min(...metadata.vectors.layers.map((l) => l.min_zoom || 0));

    vectorGridLayer = L.vectorGrid.protobuf(
        _vectorUrl(datasetId, metadata),
        {
            vectorTileLayerStyles: styles,
            interactive: true,
            // Canvas renderer (not the default SVG): heavy datasets put 10k–100k+ full-fidelity polygons in
            // view (81k SVG <path> nodes at z6 on a Prime 5K breast slide) — SVG creates one DOM node per
            // feature, which is what freezes/locks the tab and forbids low zooms. Canvas draws to a single
            // element (no DOM bloat), so cells can render far earlier and stay responsive. Interactivity +
            // setFeatureStyle recolor are supported by VectorGrid's canvas.tile renderer.
            rendererFactory: L.canvas.tile,
            maxZoom: maxZoom,
            minZoom: gridMinZoom,
            bounds: imageBounds,
            // Every feature needs an ID for restyleAll() to work.
            // Polygons use layer_type + cell_id (cross-tile highlight).
            // Points get a unique counter ID.
            getFeatureId: (() => {
                let counter = 0;
                return (f) => {
                    if (f.properties.cell_id) {
                        return f.properties.layer_type + "_" + f.properties.cell_id;
                    }
                    return "_pt_" + (counter++);
                };
            })(),
        }
    );

    vectorGridLayer.on("mouseover", (e) => {
        const props = e.layer.properties;
        const def = layerLookup[props.layer_type] || {};
        InfoPanel.show(_withFacets(props), def);

        // Cross-tile polygon highlight
        const fid = props.cell_id
            ? props.layer_type + "_" + props.cell_id
            : null;
        if (fid && def.type === "polygon") {
            hoveredFeatureId = fid;
            vectorGridLayer.setFeatureStyle(fid, {
                weight: 2,
                color: def.color,
                fillOpacity: 0.3,
                fillColor: def.color,
                fill: true,
                opacity: 1,
                interactive: true,
            });
        }
    });

    vectorGridLayer.on("mouseout", () => {
        InfoPanel.clear();
        if (hoveredFeatureId) {
            vectorGridLayer.resetFeatureStyle(hoveredFeatureId);
            hoveredFeatureId = null;
        }
    });

    vectorGridLayer.on("click", (e) => {
        const props = e.layer.properties;
        const def = layerLookup[props.layer_type] || {};
        InfoPanel.pin(_withFacets(props), def);
    });

    _wireTileCounting(vectorGridLayer);
    vectorGridLayer.addTo(map);
}

function getLayerStyle(layerDef) {
    if (layerDef.type === "polygon") {
        return {
            weight: 1,
            color: layerDef.color,
            fillOpacity: 0,
            fill: true,
            opacity: 0.8,
            interactive: true,
        };
    }
    // Points: interactive: true is REQUIRED here — VectorGrid's PointSymbolizer
    // skips L.CircleMarker's constructor, so options.interactive is unset unless
    // we include it in the style. Without it, Canvas hit detection ignores points.
    return {
        radius: 5,
        weight: 2,
        color: layerDef.color,
        fillColor: layerDef.color,
        fillOpacity: 0.7,
        fill: true,
        interactive: true,
    };
}

function initMap() {
    map = L.map("map", {
        crs: L.CRS.Simple,
        minZoom: 0,
        maxZoom: 10,
        zoomControl: true,
        attributionControl: false,
    });

    // Custom scale bar in microns (Leaflet's built-in assumes meters)
    const scaleDiv = L.DomUtil.create("div", "micron-scale-bar");
    const scaleControl = L.control({ position: "bottomright" });
    scaleControl.onAdd = () => scaleDiv;
    scaleControl.addTo(map);

    map.on("zoomend", _updateZoomUI);  // keep the zoom readout + layer badges in sync as the user zooms

    map.on("zoomend moveend", () => {
        if (!metadata) return;
        const umPerPx = metadata.um_per_px;
        const rasterMaxZoom = _rasterOf(metadata).max_zoom;
        // Pixels per CSS pixel at current zoom
        const scale = Math.pow(2, rasterMaxZoom - map.getZoom());
        // Target ~100px bar width
        const barPx = 100;
        const barUm = barPx * scale * umPerPx;
        // Round to a nice number
        const nice = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
        const niceUm = nice.reduce((prev, n) => Math.abs(n - barUm) < Math.abs(prev - barUm) ? n : prev);
        const nicePx = Math.round(niceUm / (scale * umPerPx));
        scaleDiv.innerHTML =
            `<div style="width:${nicePx}px;border-bottom:2px solid #a0aec0;margin-bottom:2px;"></div>` +
            `<div style="text-align:center;font-size:0.75rem;color:#a0aec0;">${niceUm} µm</div>`;
    });

    map.setView([0, 0], 0);
}

initMap();
loadDatasets();

export { map, metadata, vectorGridLayer, hiddenLayers };
