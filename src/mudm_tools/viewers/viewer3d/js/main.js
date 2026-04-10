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
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
// NOTE: Do NOT use setPixelRatio — causes rendering offset on macOS Retina.
// DPR is handled manually in onResize() instead.
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
    onSelectionChange: (worldCenter, ring) => {
        // Only apply spatial filter when overview is actually enabled
        const toggle = document.getElementById('overview-toggle');
        if (toggle && toggle.checked) {
            tileManager.setSpatialFilter(worldCenter, ring);
            // Recenter main camera on clicked position
            const offset = camera.position.clone().sub(controls.target);
            controls.target.copy(worldCenter);
            camera.position.copy(worldCenter).add(offset);
            controls.update();
        }
    },
});
overviewPanel.initDOM();

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
    for (const val of attrInfo.values) {
        const num = Number(val);
        let matches = true;
        if (rangeMin !== null && num < rangeMin) matches = false;
        if (rangeMax !== null && num > rangeMax) matches = false;
        palette.set(val, matches ? COLOR_MATCH : COLOR_NO_MATCH);
    }

    tileManager.setColorBy(attr, palette);

    // Build name → color map for sidebar swatches
    const nameColorMap = new Map();
    for (const [name, feat] of Object.entries(tileManager.featureIndex)) {
        const val = String(feat[attr] ?? '');
        nameColorMap.set(name, palette.get(val) || COLOR_NO_MATCH);
    }
    featureSelector.updateSwatchColors(nameColorMap);

    // Legend: show criteria
    const items = new Map();
    let matchLabel = 'In range';
    if (rangeMin !== null && rangeMax !== null) {
        matchLabel = `${rangeMin.toLocaleString()} \u2013 ${rangeMax.toLocaleString()}`;
    } else if (rangeMin !== null) {
        matchLabel = `\u2265 ${rangeMin.toLocaleString()}`;
    } else if (rangeMax !== null) {
        matchLabel = `\u2264 ${rangeMax.toLocaleString()}`;
    }
    items.set(matchLabel, COLOR_MATCH);
    items.set('Other', COLOR_NO_MATCH);
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
        if (!hit.object.visible) continue;
        if (hit.object.userData?._isSliceHelper) continue;
        if (slicePlanePanel?.enabled && slicePlanePanel.clipPlane.distanceToPoint(hit.point) < 0) continue;
        const name = _findFeatureName(hit.object);
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

    // Restore previous
    if (_hoveredFeature) {
        _setFeatureEmissive(_hoveredFeature, 0x000000);
    }

    _hoveredFeature = featureName;

    // Highlight new
    if (_hoveredFeature) {
        _setFeatureEmissive(_hoveredFeature, 0x333333);
        canvas.style.cursor = 'pointer';
    } else {
        canvas.style.cursor = '';
    }
}

function _setFeatureEmissive(name, color) {
    for (const node of tileManager.nodeByUri.values()) {
        if (!node.object3D) continue;
        const meshes = node.meshByFeature[name];
        if (!meshes) continue;
        for (const mesh of meshes) {
            if (mesh.material) {
                mesh.material.emissive.set(color);
            }
        }
    }
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

// --- LOD Controls ---
const lodRadios = document.querySelectorAll('input[name="lod-mode"]');
const zoomSlider = document.getElementById('zoom-slider');
const zoomLabel = document.getElementById('zoom-label');
const zoomDistEl = document.getElementById('zoom-distribution');

for (const radio of lodRadios) {
    radio.addEventListener('change', () => {
        if (!radio.checked) return;  // ignore the unchecked radio's event
        tileManager.lodMode = radio.value;
        zoomSlider.disabled = (radio.value === 'dynamic');
        if (radio.value === 'forced') {
            tileManager.forcedZoom = parseInt(zoomSlider.value);
        }
        console.log(`[LOD] mode=${radio.value} forcedZoom=${tileManager.forcedZoom}`);

        // Update overview zoom reference
        if (radio.value === 'forced') {
            overviewPanel.currentZoom = tileManager.forcedZoom;
        } else {
            overviewPanel.currentZoom = tileManager.maxZoom;
        }
        syncOverviewRingMax();
        overviewPanel._updateOverlays();
    });
}
zoomSlider.addEventListener('input', () => {
    const z = parseInt(zoomSlider.value);
    zoomLabel.textContent = z;
    tileManager.forcedZoom = z;
    overviewPanel.currentZoom = z;
    syncOverviewRingMax();
    overviewPanel._updateOverlays();
    overviewPanel._fireSelectionChange();
});

// --- Overview Controls ---
const overviewToggle = document.getElementById('overview-toggle');
const overviewOptions = document.getElementById('overview-options');
const overviewContainer = document.getElementById('overview-container');
const overviewRingSlider = document.getElementById('overview-ring-slider');
const overviewRingLabel = document.getElementById('overview-ring-label');
const overviewAxesRadios = document.querySelectorAll('input[name="overview-axes"]');

overviewToggle.addEventListener('change', () => {
    overviewPanel.enabled = overviewToggle.checked;
    overviewOptions.style.display = overviewToggle.checked ? '' : 'none';
    overviewContainer.style.display = overviewToggle.checked ? '' : 'none';

    if (!overviewToggle.checked) {
        tileManager.setSpatialFilter(null, 0);
    } else {
        overviewPanel._fireSelectionChange();
    }

    onResize();
});

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
    const dpr = window.devicePixelRatio || 1;
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

// --- Render loop ---
let animating = false;
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    tileManager.update(camera);
    renderer.render(scene, camera);

    // Render overview panels
    overviewPanel.render();

    // Update scale bar and axis gizmo
    scaleBar.update(camera, controls);
    axisGizmo.render(renderer, camera);

    // Stats + FPS
    frameCount++;
    const now = performance.now();
    if (now - lastTime >= 1000) {
        fps = Math.round(frameCount * 1000 / (now - lastTime));
        frameCount = 0;
        lastTime = now;
        statFPS.textContent = fps;
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
function resetCamera() {
    if (!tileManager.root?.box3) return;
    const box = tileManager.root.box3;
    const center = new THREE.Vector3();
    box.getCenter(center);
    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);

    // Adapt clipping planes to dataset scale
    camera.near = maxDim * 0.0001;
    camera.far = maxDim * 10;
    camera.updateProjectionMatrix();

    camera.position.set(
        center.x + maxDim * 0.6,
        center.y - maxDim * 0.6,
        center.z + maxDim * 0.5,
    );
    camera.lookAt(center);
    controls.target.copy(center);
    controls.update();

    slicePlanePanel.updateBounds(box);
}

/**
 * Sync zoom slider to tile manager state after init/switch.
 */
function syncZoomSlider() {
    zoomSlider.max = tileManager.maxZoom;
    zoomSlider.value = tileManager.maxZoom;
    zoomLabel.textContent = tileManager.maxZoom;
    tileManager.forcedZoom = tileManager.maxZoom;
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

    await tileManager.switchPyramid(baseUrl);
    const idFieldsArr = tileManager.idFields ? [...tileManager.idFields] : [];
    await featureSelector.init(featuresUrl, idFieldsArr);
    syncZoomSlider();
    resetCamera();

    // Update overview panel
    overviewPanel.setFeatureIndex(tileManager.featureIndex);
    overviewPanel.setBounds(tileManager.root.box3, tileManager.maxZoom, baseUrl);
    overviewPanel.loadTiles();
    syncOverviewRingMax();
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
            await tileManager.init();
            const idFieldsArr = tileManager.idFields ? [...tileManager.idFields] : [];
            await featureSelector.init(`/tiles/${defaultPyramid.id}/features.json`, idFieldsArr);
        } else {
            // Fallback: no manifest, try legacy single-pyramid path
            baseUrl = '/tiles/';
            tileManager = new TileManager(scene, baseUrl);
            await tileManager.init();
            await featureSelector.init('/tiles/features.json');
        }

        syncZoomSlider();
        resetCamera();

        // Update overview panel
        overviewPanel.setFeatureIndex(tileManager.featureIndex);
        overviewPanel.setBounds(tileManager.root.box3, tileManager.maxZoom, baseUrl);
        overviewPanel.loadTiles();
        syncOverviewRingMax();

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
