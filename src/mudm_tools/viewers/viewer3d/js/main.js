/**
 * Three.js 3D Tiles viewer — main entry point.
 *
 * Z-up coordinate system (matching MicroJSON world coords).
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TileManager } from './TileManager.js';
import { FeatureSelector } from './FeatureSelector.js';
import { InfoPanel } from './InfoPanel.js';
import { PyramidSelector } from './PyramidSelector.js';
import { SlicePlanePanel } from './SlicePlanePanel.js';
import { OverviewPanel } from './OverviewPanel.js';
import { ScaleBar } from './ScaleBar.js';
import { AxisGizmo } from './AxisGizmo.js';

// --- Scene setup ---
const canvas = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: false });
// NOTE: Do NOT use setPixelRatio — causes rendering offset on macOS Retina.
// DPR is handled manually in onResize() instead.
// preserveDrawingBuffer:false — takeScreenshot() forces a fresh renderer.render()
// before reading the canvas, so the back buffer never needs to be retained.
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

// --- Camera (Z-up) ---
const camera = new THREE.PerspectiveCamera(50, 1, 1, 2000000);
camera.up.set(0, 0, 1);

// --- Controls ---
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.minDistance = 10;
controls.maxDistance = 1e9;

// --- Lighting ---
scene.add(new THREE.AmbientLight(0xffffff, 0.6));

const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(1, -0.5, 1).normalize();
scene.add(dirLight);

const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
dirLight2.position.set(-1, 0.5, -0.5).normalize();
scene.add(dirLight2);

// --- Scale Bar ---
const scaleBar = new ScaleBar(canvas);

// --- Axis Gizmo ---
const axisGizmo = new AxisGizmo();

// --- Tile Manager (baseUrl set after pyramid selection) ---
let tileManager = new TileManager(scene, '/tiles/default/');

// --- Slice Plane ---
const sliceContainer = document.getElementById('slice-controls');
const slicePlanePanel = new SlicePlanePanel(sliceContainer, { renderer, scene });

// --- Info Panel ---
const infoPanel = new InfoPanel(camera, scene, canvas);
infoPanel.slicePanel = slicePlanePanel;

// --- Overview Panel ---
const overviewPanel = new OverviewPanel({
    onSelectionChange: (worldCenter, ring, box) => {
        // Overview is always on — clicking it focuses the main view (spatial
        // filter) on the clicked region and reframes the camera so the orange
        // box's midpoint lands at the center of the view.
        tileManager.setSpatialFilter(worldCenter, ring);
        if (box) {
            // Fit + center on the selection box (same framing as frameZoomRegion).
            // A pure pan that preserved the old oblique offset left the box
            // visibly translated under the perspective camera.
            frameBox(box);
        } else {
            const offset = camera.position.clone().sub(controls.target);
            controls.target.copy(worldCenter);
            camera.position.copy(worldCenter).add(offset);
            controls.update();
        }
    },
});
overviewPanel.initDOM();
overviewPanel.enabled = true;   // overview is always on (no toggle)

// --- Feature Selector ---
const selectorContainer = document.getElementById('feature-selector');
const featureSelector = new FeatureSelector(selectorContainer, (selected) => {
    tileManager.setSelectedFeatures(selected);
    overviewPanel.setSelectedFeatures(selected);
});

// --- Color By ---
const colorBySelect = document.getElementById('color-by-select');
const colorLegend = document.getElementById('color-legend');

// 20 visually distinct colors for categorical palettes
const PALETTE_COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9a6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9',
];

/**
 * Extract the filter schema + facet-store descriptor for the currently loaded pyramid, if any.
 * Both come from the muDM TileModel (TileManager.descriptor, set in TileManager.init()): the
 * per-feature filter-field schema lives at vector_layers[0], and the columnar facet-store parquet
 * is declared as a role="facets" asset. Either/both can be null (older, un-migrated pyramids),
 * in which case callers fall back to the features.json-derived path.
 */
function facetContext(tm) {
    const desc = tm.descriptor;
    const schema = desc?.vector_layers?.[0] || null;
    const fa = (desc?.assets || []).find(a => a.role === 'facets');
    const facetStore = fa ? { baseUrl: tm.baseUrl, href: fa.href, key: fa.key } : null;
    return { schema, facetStore };
}

/**
 * Discover colorable attributes from feature index.
 * Excludes structural keys and id_fields from features.json config.
 * Returns [{key, label, type: 'categorical'|'numeric', values: string[], numericRange: [min,max]|null}].
 */
function discoverAttributes(featureIndex, idFields) {
    const SKIP = new Set(['color', 'tiles', 'acronym']);
    if (idFields) for (const f of idFields) SKIP.add(f);

    const attrMeta = {};  // key → {values: Set, allNumeric: bool, min, max}
    for (const feat of Object.values(featureIndex)) {
        for (const [key, val] of Object.entries(feat)) {
            if (SKIP.has(key)) continue;
            if (val === null || val === undefined || typeof val === 'object') continue;
            if (!attrMeta[key]) attrMeta[key] = { values: new Set(), allNumeric: true, min: Infinity, max: -Infinity };
            const num = typeof val === 'number' ? val : Number(val);
            if (isNaN(num)) {
                attrMeta[key].allNumeric = false;
                attrMeta[key].values.add(String(val));
            } else {
                attrMeta[key].values.add(String(val));
                attrMeta[key].min = Math.min(attrMeta[key].min, num);
                attrMeta[key].max = Math.max(attrMeta[key].max, num);
            }
        }
    }

    const result = [];
    for (const [key, meta] of Object.entries(attrMeta)) {
        if (meta.values.size < 2) continue;
        const label = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        const type = meta.allNumeric ? 'numeric' : 'categorical';
        const values = [...meta.values].sort((a, b) =>
            type === 'numeric' ? Number(a) - Number(b) : a.localeCompare(b));
        const numericRange = type === 'numeric' ? [meta.min, meta.max] : null;
        result.push({ key, label, type, values, numericRange });
    }
    return result.sort((a, b) => a.key.localeCompare(b.key));
}

/**
 * Build a color palette for a set of values.
 * Returns Map<string, string> (value → hex color).
 */
function buildPalette(values) {
    const palette = new Map();
    for (let i = 0; i < values.length; i++) {
        if (i < PALETTE_COLORS.length) {
            palette.set(values[i], PALETTE_COLORS[i]);
        } else {
            // Fall back to HSL for large palettes
            const hue = (i * 137.508) % 360;  // golden angle for max spread
            palette.set(values[i], `hsl(${hue.toFixed(0)}, 65%, 55%)`);
        }
    }
    return palette;
}

/**
 * Populate the color-by dropdown from feature index.
 */
function populateColorByDropdown(featureIndex, idFields) {
    // Clear existing options except "Original"
    while (colorBySelect.options.length > 1) {
        colorBySelect.remove(1);
    }
    const attrs = discoverAttributes(featureIndex, idFields);
    for (const attr of attrs) {
        const opt = document.createElement('option');
        opt.value = attr.key;
        const suffix = attr.type === 'numeric' ? ' (range)' : ` (${attr.values.length})`;
        opt.textContent = attr.label + suffix;
        colorBySelect.appendChild(opt);
    }
    colorBySelect.value = '';
    _cachedAttributes = attrs;
}
let _cachedAttributes = [];

// Numeric color-by range controls
const colorByRangeContainer = document.getElementById('color-by-range');
const COLOR_MATCH = '#6fdfaf';
const COLOR_NO_MATCH = '#444444';
// Continuous value ramp for numeric color-by: blue (low) -> red (high). Replaces the old binary
// in-range/out-of-range highlight, which painted every mesh one flat color at full range so
// "color by a numeric feature" looked like it did nothing.
function _rampColor(t) { t = Math.max(0, Math.min(1, t)); return `hsl(${((1 - t) * 240).toFixed(0)}, 75%, 52%)`; }

function showColorByRange(attrInfo) {
    colorByRangeContainer.innerHTML = '';
    if (!attrInfo || attrInfo.type !== 'numeric') {
        colorByRangeContainer.style.display = 'none';
        return;
    }
    colorByRangeContainer.style.display = '';
    const [dataMin, dataMax] = attrInfo.numericRange;

    const row = document.createElement('div');
    row.className = 'filter-numeric-row';

    const makeInput = (placeholder) => {
        const inp = document.createElement('input');
        inp.type = 'number';
        inp.className = 'filter-numeric-input';
        inp.placeholder = placeholder;
        inp.step = 'any';
        return inp;
    };

    const minLabel = document.createElement('span');
    minLabel.textContent = '\u2265'; // ≥
    minLabel.className = 'filter-numeric-label';
    const minInput = makeInput(String(dataMin));

    const maxLabel = document.createElement('span');
    maxLabel.textContent = '\u2264'; // ≤
    maxLabel.className = 'filter-numeric-label';
    const maxInput = makeInput(String(dataMax));

    const update = () => {
        const rMin = minInput.value !== '' ? parseFloat(minInput.value) : null;
        const rMax = maxInput.value !== '' ? parseFloat(maxInput.value) : null;
        applyNumericColorBy(attrInfo, rMin, rMax);
    };
    minInput.addEventListener('input', update);
    maxInput.addEventListener('input', update);

    row.appendChild(minLabel);
    row.appendChild(minInput);
    row.appendChild(maxLabel);
    row.appendChild(maxInput);

    const hint = document.createElement('div');
    hint.className = 'filter-range-hint';
    hint.textContent = `Range: ${dataMin.toLocaleString()} \u2013 ${dataMax.toLocaleString()}`;

    colorByRangeContainer.appendChild(row);
    colorByRangeContainer.appendChild(hint);
}

function applyNumericColorBy(attrInfo, rangeMin, rangeMax) {
    const attr = attrInfo.key;
    // Build palette: each unique value → match or no-match color
    const palette = new Map();
    const [rMin, rMax] = attrInfo.numericRange || [0, 1];
    const span = (rMax - rMin) || 1;
    for (const val of attrInfo.values) {
        const num = Number(val);
        let matches = true;
        if (rangeMin !== null && num < rangeMin) matches = false;
        if (rangeMax !== null && num > rangeMax) matches = false;
        // in filter range -> position on the value ramp; outside -> dimmed
        palette.set(val, matches ? _rampColor((num - rMin) / span) : COLOR_NO_MATCH);
    }

    tileManager.setColorBy(attr, palette);

    // Build name → color map for sidebar swatches
    const nameColorMap = new Map();
    for (const [name, feat] of Object.entries(tileManager.featureIndex)) {
        const val = String(feat[attr] ?? '');
        nameColorMap.set(name, palette.get(val) || COLOR_NO_MATCH);
    }
    featureSelector.updateSwatchColors(nameColorMap);

    // Legend: a 5-stop gradient across the data range (matches the mesh ramp), plus a dimmed
    // "out of range" swatch only when a min/max filter is active.
    const items = new Map();
    for (let i = 0; i <= 4; i++) {
        const t = i / 4;
        items.set((rMin + t * span).toLocaleString(undefined, { maximumFractionDigits: 2 }), _rampColor(t));
    }
    if (rangeMin !== null || rangeMax !== null) items.set('out of range', COLOR_NO_MATCH);
    updateLegend(attr, items);
}

/**
 * Update the color legend overlay.
 */
function updateLegend(attribute, palette) {
    if (!attribute || !palette || palette.size === 0) {
        colorLegend.style.display = 'none';
        return;
    }
    const attr = _cachedAttributes.find(a => a.key === attribute);
    const label = attr?.label || attribute;

    let html = `<h4>${label}</h4>`;
    for (const [val, color] of palette) {
        html += `<div class="legend-item">` +
            `<span class="legend-swatch" style="background:${color}"></span>` +
            `<span class="legend-label" title="${val}">${val}</span></div>`;
    }
    colorLegend.innerHTML = html;
    colorLegend.style.display = '';
}

colorBySelect.addEventListener('change', () => {
    const attr = colorBySelect.value;
    if (!attr) {
        tileManager.setColorBy(null, null);
        featureSelector.updateSwatchColors(null);
        updateLegend(null, null);
        showColorByRange(null);
        return;
    }
    const attrInfo = _cachedAttributes.find(a => a.key === attr);
    if (!attrInfo) return;

    if (attrInfo.type === 'numeric') {
        // Numeric: show range controls, default to full range (all match)
        showColorByRange(attrInfo);
        applyNumericColorBy(attrInfo, null, null);
    } else {
        // Categorical: discrete palette
        showColorByRange(null);
        const palette = buildPalette(attrInfo.values);
        tileManager.setColorBy(attr, palette);
        const nameColorMap = new Map();
        for (const [name, feat] of Object.entries(tileManager.featureIndex)) {
            const val = String(feat[attr] ?? '');
            nameColorMap.set(name, palette.get(val) || '#555555');
        }
        featureSelector.updateSwatchColors(nameColorMap);
        updateLegend(attr, palette);
    }
});

// --- Screenshot ---
function takeScreenshot() {
    // Render current frame to ensure buffer is fresh
    renderer.render(scene, camera);

    const srcCanvas = renderer.domElement;
    const w = srcCanvas.width;
    const h = srcCanvas.height;

    // Create compositing canvas
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = w;
    tmpCanvas.height = h;
    const ctx = tmpCanvas.getContext('2d');

    // Draw 3D render
    ctx.drawImage(srcCanvas, 0, 0);

    // Composite scale bar
    const scaleBarLine = document.querySelector('#scale-bar .scale-bar-line');
    const scaleBarLabel = document.querySelector('#scale-bar .scale-bar-label');
    if (scaleBarLine && scaleBarLabel) {
        const barPx = scaleBarLine.offsetWidth;
        const labelText = scaleBarLabel.textContent;
        const dpr = renderer.getPixelRatio();
        const x = 16 * dpr;
        const y = h - 40 * dpr;

        // Bar line
        ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
        ctx.fillRect(x, y, barPx * dpr, 3 * dpr);
        // End caps
        ctx.fillRect(x, y - 4 * dpr, 2 * dpr, 11 * dpr);
        ctx.fillRect(x + barPx * dpr - 2 * dpr, y - 4 * dpr, 2 * dpr, 11 * dpr);

        // Label
        ctx.font = `${12 * dpr}px system-ui, sans-serif`;
        ctx.textAlign = 'center';
        ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
        ctx.shadowBlur = 3 * dpr;
        ctx.fillText(labelText, x + barPx * dpr / 2, y + 18 * dpr);
        ctx.shadowBlur = 0;
    }

    // Download
    const link = document.createElement('a');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    link.download = `mudm-${timestamp}.png`;
    link.href = tmpCanvas.toDataURL('image/png');
    link.click();
}

document.getElementById('screenshot-btn').addEventListener('click', takeScreenshot);

// --- Background Color ---
const BG_COLORS = {
    dark:  0x1a1a2e,
    light: 0x4a4a5e,
    white: 0xffffff,
};

const bgRadios = document.querySelectorAll('input[name="bg-color"]');
function setBackground(value) {
    scene.background = new THREE.Color(BG_COLORS[value] ?? BG_COLORS.dark);
    localStorage.setItem('mudm-bg', value);
    for (const r of bgRadios) r.checked = (r.value === value);
}
for (const radio of bgRadios) {
    radio.addEventListener('change', () => {
        if (radio.checked) setBackground(radio.value);
    });
}
// Restore saved preference
const savedBg = localStorage.getItem('mudm-bg');
if (savedBg && BG_COLORS[savedBg]) setBackground(savedBg);

// --- Hover Highlight ---
let _hoveredFeature = null;
let _hoverRafPending = false;
const _hoverRaycaster = new THREE.Raycaster();
const _hoverMouse = new THREE.Vector2();

canvas.addEventListener('mousemove', (e) => {
    if (_hoverRafPending) return;
    _hoverRafPending = true;
    requestAnimationFrame(() => {
        _hoverRafPending = false;
        _updateHover(e);
    });
});

canvas.addEventListener('mouseleave', () => {
    _setHover(null);
});

function _updateHover(event) {
    const rect = canvas.getBoundingClientRect();
    _hoverMouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    _hoverMouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    _hoverRaycaster.setFromCamera(_hoverMouse, camera);
    const intersects = _hoverRaycaster.intersectObjects(scene.children, true);

    for (const hit of intersects) {
        if (hit.object.userData?._isSliceHelper) continue;
        if (slicePlanePanel?.enabled && slicePlanePanel.clipPlane.distanceToPoint(hit.point) < 0) continue;
        // BatchedMesh raycast returns the hit instance in hit.batchId and only reports
        // VISIBLE instances; legacy line/point meshes resolve via the parent userData.
        const name = hit.object.isBatchedMesh
            ? hit.object.featureByInstance?.get(hit.batchId)
            : (hit.object.visible ? _findFeatureName(hit.object) : null);
        if (name) {
            _setHover(name);
            return;
        }
    }
    _setHover(null);
}

function _findFeatureName(object) {
    let current = object;
    while (current) {
        if (current.userData?._featureName) return current.userData._featureName;
        current = current.parent;
    }
    return null;
}

function _setHover(featureName) {
    if (featureName === _hoveredFeature) return;

    // Restore previous + highlight new — TileManager owns the batches (per-instance
    // color brighten/restore), so the highlight lives there now.
    if (_hoveredFeature) tileManager.setFeatureHighlight(_hoveredFeature, false);
    _hoveredFeature = featureName;
    if (_hoveredFeature) {
        tileManager.setFeatureHighlight(_hoveredFeature, true);
        canvas.style.cursor = 'pointer';
    } else {
        canvas.style.cursor = '';
    }
    requestRender();   // hover highlight changed → repaint (render-on-demand)
}

// --- Stats ---
const statLoaded = document.getElementById('stat-loaded');
const statVisible = document.getElementById('stat-visible');
const statMemory = document.getElementById('stat-memory');
const statFPS = document.getElementById('stat-fps');
const loadingEl = document.getElementById('loading');

let lastTime = performance.now();
let frameCount = 0;
let fps = 0;

// --- GPU Budget Slider ---
const gpuSlider = document.getElementById('gpu-budget-slider');
const gpuLabel = document.getElementById('gpu-budget-label');
gpuSlider.addEventListener('input', () => {
    const mb = parseInt(gpuSlider.value);
    gpuLabel.textContent = mb;
    tileManager.maxGpuMB = mb;
});

// --- Zoom level (always a manually-selected level; no dynamic LOD) ---
const zoomSlider = document.getElementById('zoom-slider');
const zoomLabel = document.getElementById('zoom-label');
const zoomDistEl = document.getElementById('zoom-distribution');
tileManager.lodMode = 'forced';

zoomSlider.addEventListener('input', () => {
    const z = parseInt(zoomSlider.value);
    zoomLabel.textContent = z;
    tileManager.forcedZoom = z;
    // Zoom level = a centered spatial zoom into the orange box at the new zoom: sync the
    // overview's zoom first, then focus the main view + spatial filter on that box so the
    // loaded region stays centered (matches the overview-click + initial-load behavior).
    overviewPanel.currentZoom = z;
    syncOverviewRingMax();
    overviewPanel._updateOverlays();
    focusOverviewBox();
    requestRender();
});

// --- Opacity (main-view neuron transparency) ---
const opacitySlider = document.getElementById('opacity-slider');
const opacityLabel = document.getElementById('opacity-label');
opacitySlider.addEventListener('input', () => {
    const pct = parseInt(opacitySlider.value);
    opacityLabel.textContent = pct + '%';
    tileManager.setOpacity(pct / 100);
    requestRender();
});

// --- Overview Controls (overview is always on; selector lives above the panels) ---
const overviewRingSlider = document.getElementById('overview-ring-slider');
const overviewRingLabel = document.getElementById('overview-ring-label');
const overviewAxesRadios = document.querySelectorAll('input[name="overview-axes"]');

for (const radio of overviewAxesRadios) {
    radio.addEventListener('change', () => {
        if (!radio.checked) return;
        overviewPanel.setAxisPair(radio.value);
    });
}

overviewRingSlider.addEventListener('input', () => {
    const r = parseInt(overviewRingSlider.value);
    overviewRingLabel.textContent = r;
    overviewPanel.setRing(r);
});

// --- Resize ---
function onResize() {
    const sidebarWidth = 300;
    const overviewWidth = overviewPanel.enabled ? 350 : 0;
    const w = window.innerWidth - sidebarWidth - overviewWidth;
    const h = window.innerHeight;
    // Cap effective DPR at 1.5 — on a 2x/3x Retina display this cuts fragment
    // work up to ~4x with negligible visual loss for solid-colored meshes.
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();

    // Handle DPR manually (setPixelRatio causes offset on macOS Retina)
    renderer.setSize(w * dpr, h * dpr, false);
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    canvas.style.left = sidebarWidth + 'px';

    // Reposition info panel
    const infoEl = document.getElementById('info-panel');
    if (infoEl) infoEl.style.right = (overviewWidth + 12) + 'px';

    // Reposition stats, scale bar, and legend
    const statsEl = document.getElementById('stats');
    if (statsEl) statsEl.style.left = (sidebarWidth + 12) + 'px';
    const scaleEl = document.getElementById('scale-bar');
    if (scaleEl) scaleEl.style.left = (sidebarWidth + 12) + 'px';
    if (colorLegend) colorLegend.style.left = (sidebarWidth + 12) + 'px';

    overviewPanel.resize();
}
window.addEventListener('resize', onResize);
onResize();

// --- Keyboard Shortcuts ---
window.addEventListener('keydown', (e) => {
    // Ignore when typing in input/select elements
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;

    switch (e.key.toLowerCase()) {
        case 'r':
            resetCamera();
            break;
        case 'a':
            featureSelector._selectAllVisible();
            break;
        case 'escape':
            featureSelector._clearAll();
            document.getElementById('info-panel').style.display = 'none';
            break;
        case 's':
            takeScreenshot();
            break;
        case 'f': {
            // Focus: zoom to fit selected features' bounding boxes
            if (!tileManager.root?.box3 || tileManager.selectedFeatures.size === 0) break;
            const fitBox = new THREE.Box3();
            for (const name of tileManager.selectedFeatures) {
                const feat = tileManager.featureIndex[name];
                if (!feat) continue;
                for (const uris of Object.values(feat.tiles)) {
                    for (const uri of uris) {
                        const node = tileManager.nodeByUri.get(uri);
                        if (node?.box3) fitBox.union(node.box3);
                    }
                }
            }
            if (!fitBox.isEmpty()) {
                const center = new THREE.Vector3();
                fitBox.getCenter(center);
                const size = new THREE.Vector3();
                fitBox.getSize(size);
                const maxDim = Math.max(size.x, size.y, size.z);
                camera.position.set(
                    center.x + maxDim * 0.6,
                    center.y - maxDim * 0.6,
                    center.z + maxDim * 0.5,
                );
                controls.target.copy(center);
                controls.update();
            }
            break;
        }
    }
});

// --- Render loop (render-on-demand) ---
// Idle frames are skipped entirely: the heavy work (tileManager.update +
// renderer.render + the two overview panels + gizmo) only runs when something
// actually changed. A 9k–23k-mesh scene that sat at a busy-looping 60fps now
// costs ~0 when nothing is happening. needsRender is raised by:
//   • OrbitControls 'change' (orbit / zoom / pan), and damping settles via
//     controls.update()'s return value;
//   • any sidebar interaction (sliders, selects, checkboxes, buttons, feature
//     toggles) via cheap capture-phase document listeners;
//   • hover-highlight changes (_setHover) and window resize;
//   • tiles still streaming in (tileManager._pendingLoads) or a just-completed
//     async load/unload (tileManager._dirty).
let animating = false;
let needsRender = true;
function requestRender() { needsRender = true; }
controls.addEventListener('change', requestRender);
window.addEventListener('resize', requestRender);
// Catch-all for sidebar UI + overview interactions + keyboard shortcuts —
// these mutate the scene/overview without moving the main camera. Capture phase
// + a plain boolean flip → negligible cost. ('wheel' covers the overview's own
// zoom; 'keydown' covers reset/select-all/focus/clear shortcuts.)
document.addEventListener('input', requestRender, true);
document.addEventListener('change', requestRender, true);
document.addEventListener('click', requestRender, true);
document.addEventListener('wheel', requestRender, true);
document.addEventListener('keydown', requestRender, true);

function animate() {
    requestAnimationFrame(animate);
    const moved = controls.update();   // true while inertial damping is settling
    const busy = tileManager._pendingLoads > 0 || tileManager._dirty;
    if (!(needsRender || moved || busy)) return;   // idle → skip the whole frame
    needsRender = false;
    tileManager._dirty = false;

    tileManager.update(camera);
    renderer.render(scene, camera);

    // Render overview panels (skip when the panel is collapsed/disabled —
    // it would otherwise cost two extra full-scene renders per frame)
    if (overviewPanel.enabled) overviewPanel.render();

    // Update scale bar and axis gizmo
    scaleBar.update(camera, controls);
    axisGizmo.render(renderer, camera);

    // Stats + FPS (counted over rendered frames only)
    frameCount++;
    const now = performance.now();
    if (now - lastTime >= 1000) {
        fps = Math.round(frameCount * 1000 / (now - lastTime));
        frameCount = 0;
        lastTime = now;
        statFPS.textContent = `${fps} · ${renderer.info.render.calls} draws`;
    }
    statLoaded.textContent = tileManager.loadedCount;
    statVisible.textContent = tileManager.visibleCount;
    statMemory.textContent = tileManager.gpuMB;
    zoomDistEl.textContent = tileManager.zoomDistribution
        ? `LOD: ${tileManager.zoomDistribution}` : '';
}

/**
 * Reset camera to frame the tileset bounding volume.
 */
/**
 * Frame the main camera so an arbitrary world-space box fills the view, centered.
 * Uses the same fixed oblique angle as frameZoomRegion so overview clicks and the
 * zoom slider agree. The box's center projects to screen-center after this.
 */
function frameBox(box) {
    if (!box) return;
    const center = new THREE.Vector3(); box.getCenter(center);
    const size = new THREE.Vector3(); box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    camera.position.set(
        center.x + maxDim * 0.6,
        center.y - maxDim * 0.6,
        center.z + maxDim * 0.5,
    );
    camera.lookAt(center);
    controls.target.copy(center);
    controls.update();
}

/**
 * Frame the main camera on a CENTERED region sized to a zoom level: the region is
 * the full volume scaled by 1/2^z (one octree tile's footprint at that zoom). So
 * z0 frames the whole volume, and each finer level zooms into a centered detail.
 */
function frameZoomRegion(z) {
    const box = tileManager.root?.box3;
    if (!box) return;
    const center = new THREE.Vector3(); box.getCenter(center);
    const size = new THREE.Vector3(); box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    const ring = parseInt(overviewRingSlider.value) || 0;
    // Region extent (incl. neighbor ring) as a fraction of the full volume — matches
    // the overview's selection box: (2*ring+1) tiles at this zoom.
    const f = (2 * ring + 1) / Math.pow(2, Math.max(0, z));
    camera.position.set(
        center.x + maxDim * 0.6 * f,
        center.y - maxDim * 0.6 * f,
        center.z + maxDim * 0.5 * f,
    );
    camera.lookAt(center);
    controls.target.copy(center);
    controls.update();
}

/**
 * Focus the main view + spatial filter on the overview's CURRENT selection box (the
 * orange box) so the loaded tiles land exactly at the center of the view. This is the
 * single source of truth shared by initial load, the zoom slider, and overview clicks
 * — framing the raw volume center instead would be off by up to half a tile (the
 * crosshair snaps to a tile center, which for an even grid is offset from the volume
 * center). Falls back to the raw volume center only if the overview box isn't ready.
 */
function focusOverviewBox() {
    const ring = parseInt(overviewRingSlider.value) || 0;
    const box = overviewPanel.bounds ? overviewPanel._computeSelectionBox() : null;
    if (box) {
        tileManager.setSpatialFilter(box.getCenter(new THREE.Vector3()), ring);
        frameBox(box);
    } else if (tileManager.root?.box3) {
        const c = new THREE.Vector3();
        tileManager.root.box3.getCenter(c);
        tileManager.setSpatialFilter(c, ring);
        frameZoomRegion(tileManager.forcedZoom);
    }
}

function resetCamera() {
    if (!tileManager.root?.box3) return;
    const box = tileManager.root.box3;
    const center = new THREE.Vector3();
    box.getCenter(center);
    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);

    // Adapt clipping planes + zoom limits to dataset scale. Coordinates range from
    // nanometers (EM/connectome) to meters (anatomy); a fixed minDistance would clamp
    // zoom-in on small-coordinate datasets (e.g. a ~1.8-unit HRA body vs the old minDistance 10).
    camera.near = maxDim * 0.0001;
    camera.far = maxDim * 10;
    camera.updateProjectionMatrix();
    controls.minDistance = maxDim * 0.0005;

    // Scale bar reads real-world units from tilejson3d (meters_per_unit); defaults to nm (EM).
    scaleBar.setMetersPerUnit(tileManager.metersPerUnit ?? 1e-9);
    overviewPanel.metersPerUnit = tileManager.metersPerUnit ?? 1e-9;  // overview scale bar too

    // Constrain loading to EXACTLY the centered region the overview's orange box shows,
    // and frame the camera on that box's center — from the very first frame, not just
    // after the user touches a control. (Requires overviewPanel.setBounds() to have run
    // first; loadPyramid/init order this before resetCamera.)
    focusOverviewBox();

    slicePlanePanel.updateBounds(box);
}

/**
 * Sync zoom slider to tile manager state after init/switch.
 */
function syncZoomSlider() {
    zoomSlider.max = tileManager.maxZoom;
    // Default to a MID zoom (e.g. z2 for a 0-4 pyramid): open on a centered detail,
    // not the whole volume at the finest level.
    const mid = Math.round(tileManager.maxZoom / 2);
    zoomSlider.value = mid;
    zoomLabel.textContent = mid;
    tileManager.forcedZoom = mid;
    overviewPanel.currentZoom = mid;   // keep the overview selection box in sync
}

function syncOverviewRingMax() {
    const z = tileManager.maxZoom;
    const maxRing = Math.min(2, Math.pow(2, z) - 1);
    overviewRingSlider.max = maxRing;
    if (parseInt(overviewRingSlider.value) > maxRing) {
        overviewRingSlider.value = maxRing;
        overviewRingLabel.textContent = maxRing;
        overviewPanel.setRing(maxRing);
    }
}

/**
 * Load a pyramid: init tile manager + feature selector, reset camera.
 */
async function loadPyramid(pyramid) {
    loadingEl.style.display = '';
    loadingEl.textContent = `Loading ${pyramid.label}...`;

    const baseUrl = `/tiles/${pyramid.id}/`;
    const featuresUrl = `/tiles/${pyramid.id}/features.json`;

    // Fetch + parse features.json ONCE and share it with both consumers
    // (the payload can be hundreds of MB at 100k+ features).
    const featuresData = await (await fetch(featuresUrl)).json();
    await tileManager.switchPyramid(baseUrl, featuresData);
    const idFieldsArr = tileManager.idFields ? [...tileManager.idFields] : [];
    const { schema, facetStore } = facetContext(tileManager);
    await featureSelector.init(featuresData, idFieldsArr, schema, facetStore);
    syncZoomSlider();
    // Set up the overview (bounds + crosshair snapped to the mid-zoom tile center) BEFORE
    // framing, so resetCamera centers the main view on the overview's selection box.
    overviewPanel.setFeatureIndex(tileManager.featureIndex);
    overviewPanel.setBounds(tileManager.root.box3, tileManager.maxZoom, baseUrl);
    syncOverviewRingMax();
    resetCamera();
    overviewPanel.loadTiles().then(requestRender);   // repaint once z0 tiles arrive (render-on-demand)
    overviewPanel.setSelectedFeatures(featureSelector.selected);

    // Reset color-by state for new pyramid
    populateColorByDropdown(tileManager.featureIndex, tileManager.idFields);
    tileManager.setColorBy(null, null);
    featureSelector.updateSwatchColors(null);
    updateLegend(null, null);
    showColorByRange(null);

    loadingEl.style.display = 'none';
    console.log(`Switched to pyramid "${pyramid.id}": ${Object.keys(tileManager.featureIndex).length} features`);
}

// --- Pyramid Selector ---
const pyramidContainer = document.getElementById('pyramid-selector');
const pyramidSelector = new PyramidSelector(pyramidContainer, async (pyramid) => {
    await loadPyramid(pyramid);
});

// --- Init ---
async function init() {
    try {
        const defaultPyramid = await pyramidSelector.init('/tiles/pyramids.json');

        let baseUrl;
        if (defaultPyramid) {
            // Use the first pyramid from manifest
            baseUrl = `/tiles/${defaultPyramid.id}/`;
            tileManager = new TileManager(scene, baseUrl);
            // Fetch + parse features.json ONCE and share it with both consumers.
            const featuresData = await (await fetch(`/tiles/${defaultPyramid.id}/features.json`)).json();
            await tileManager.init(featuresData);
            const idFieldsArr = tileManager.idFields ? [...tileManager.idFields] : [];
            const { schema, facetStore } = facetContext(tileManager);
            await featureSelector.init(featuresData, idFieldsArr, schema, facetStore);
        } else {
            // Fallback: no manifest, try legacy single-pyramid path
            baseUrl = '/tiles/';
            tileManager = new TileManager(scene, baseUrl);
            const featuresData = await (await fetch('/tiles/features.json')).json();
            await tileManager.init(featuresData);
            const { schema, facetStore } = facetContext(tileManager);
            await featureSelector.init(featuresData, [], schema, facetStore);
        }

        // The GPU-budget slider is the single source of truth. Sync the freshly-created
        // TileManager's maxGpuMB from it on load so the constructor default (1024) can no
        // longer disagree with the value shown in the UI.
        tileManager.maxGpuMB = parseInt(gpuSlider.value);

        syncZoomSlider();
        // Set up the overview (bounds + crosshair snapped to the mid-zoom tile center)
        // BEFORE framing, so resetCamera centers the main view on the overview's
        // selection box rather than the raw volume center.
        overviewPanel.setFeatureIndex(tileManager.featureIndex);
        overviewPanel.setBounds(tileManager.root.box3, tileManager.maxZoom, baseUrl);
        syncOverviewRingMax();
        resetCamera();
        overviewPanel.loadTiles().then(requestRender);   // repaint once z0 tiles arrive (render-on-demand)

        // Populate color-by dropdown
        populateColorByDropdown(tileManager.featureIndex, tileManager.idFields);

        loadingEl.style.display = 'none';

        if (!animating) {
            animating = true;
            animate();
        }

        console.log(`Loaded feature index: ${Object.keys(tileManager.featureIndex).length} features`);
    } catch (err) {
        loadingEl.textContent = `Error: ${err.message}`;
        console.error('Init failed:', err);
    }
}

init();
