/**
 * OverviewPanel — two orthographic overview panels with crosshair-based
 * tile selection.
 *
 * Manages its own THREE.Scene, two renderers, two orthographic cameras,
 * crosshair + selection box overlays, and zoom-0 tile loading.
 *
 * Z-up coordinate system (matching MicroJSON world coords).
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';

// "Nice" numbers for scale bar labels
const NICE = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
              10000, 20000, 50000, 100000, 200000, 500000, 1000000];

function formatDistance(val) {
    if (val >= 1000000) return (val / 1000000).toFixed(val % 1000000 === 0 ? 0 : 1) + ' mm';
    if (val >= 1000) return (val / 1000).toFixed(val % 1000 === 0 ? 0 : 1) + ' \u00B5m';
    return val + ' nm';
}

/** Axis configuration per projection plane. */
const AXIS_CFG = {
    xy: { hAxis: 'x', vAxis: 'y', camDir: [0, 0,  1], up: [0, 1, 0], label: 'XY' },
    xz: { hAxis: 'x', vAxis: 'z', camDir: [0, -1, 0], up: [0, 0, 1], label: 'XZ' },
    yz: { hAxis: 'y', vAxis: 'z', camDir: [1, 0,  0], up: [0, 0, 1], label: 'YZ' },
};

/** Supported axis pair presets. */
const AXIS_PAIRS = {
    'xy-xz': ['xy', 'xz'],
    'xy-yz': ['xy', 'yz'],
    'xz-yz': ['xz', 'yz'],
};

export class OverviewPanel {
    /**
     * @param {Object} opts
     * @param {function(THREE.Vector3, number): void} opts.onSelectionChange
     *   Callback fired when crosshair moves or ring changes.
     *   Receives (worldCenter, ring).
     */
    constructor({ onSelectionChange }) {
        this.onSelectionChange = onSelectionChange || (() => {});

        // World bounds + zoom metadata
        this.bounds = null;       // THREE.Box3
        this.maxZoom = 3;
        this.baseUrl = '';

        // Crosshair world position (center of selection)
        this.crosshairPos = new THREE.Vector3();

        // Zoom factor for orthographic cameras (world-units visible)
        this.zoomFactor = 1.0;

        // Current zoom level (set externally by main.js from TileManager state)
        this.currentZoom = 0;

        // Ring (0, 1, or 2 neighbor tiles around center)
        this._ring = 0;

        // Axis pair
        this._axisPair = 'xy-xz';

        // Feature visibility
        this._selectedFeatures = new Set();
        this._featureIndex = {};

        // --- Three.js objects (created in initDOM) ---
        this._scene = null;
        this._renderers = [null, null];
        this._cameras = [null, null];
        this._canvases = [null, null];
        this._labels = [null, null];

        // Overlay groups (layers 1 and 2)
        this._overlayGroups = [null, null];

        // Tile meshes group (layer 0)
        this._tileGroup = null;

        // GLTFLoader with decoders
        this._loader = null;

        // Loaded mesh tracking: featureName -> [Mesh, ...]
        this._meshByFeature = {};

        // Track loaded state
        this._tilesLoaded = false;

        // Scale bar elements (set in initDOM)
        this._scaleBars = [null, null];

        // Axis indicator canvases (set in initDOM)
        this._axisCanvases = [null, null];
    }

    // ---------------------------------------------------------------
    // Public API
    // ---------------------------------------------------------------

    /**
     * Bind to DOM canvas/label elements, create renderers and cameras.
     */
    initDOM() {
        this._canvases[0] = document.getElementById('overview-canvas-0');
        this._canvases[1] = document.getElementById('overview-canvas-1');
        this._labels[0] = document.getElementById('overview-label-0');
        this._labels[1] = document.getElementById('overview-label-1');

        // Scene with its own lighting
        this._scene = new THREE.Scene();
        this._scene.background = new THREE.Color(0x12121f);
        this._scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(1, -0.5, 1).normalize();
        this._scene.add(dirLight);

        // Tile mesh group (layer 0 — both cameras see it)
        this._tileGroup = new THREE.Group();
        this._tileGroup.layers.set(0);
        this._scene.add(this._tileGroup);

        // Overlay groups on layers 1 and 2 (one per panel)
        for (let i = 0; i < 2; i++) {
            const g = new THREE.Group();
            g.layers.set(i + 1);
            this._overlayGroups[i] = g;
            this._scene.add(g);
        }

        // Renderers
        for (let i = 0; i < 2; i++) {
            const r = new THREE.WebGLRenderer({
                canvas: this._canvases[i],
                antialias: true,
                alpha: false,
            });
            // NOTE: Do NOT use setPixelRatio — causes offset on macOS Retina.
            // DPR is handled manually in resize() instead.
            r.outputColorSpace = THREE.SRGBColorSpace;
            this._renderers[i] = r;
        }

        // Cameras (placeholder frustum — updated in _updateCameras)
        for (let i = 0; i < 2; i++) {
            const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, -1e7, 1e7);
            cam.layers.enable(0);       // tile meshes
            cam.layers.enable(i + 1);   // own overlay layer
            this._cameras[i] = cam;
        }

        // GLTFLoader with Draco + Meshopt decoders
        this._loader = new GLTFLoader();
        const dracoLoader = new DRACOLoader();
        dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.7/');
        dracoLoader.setDecoderConfig({ type: 'wasm' });
        this._loader.setDRACOLoader(dracoLoader);
        this._loader.setMeshoptDecoder(MeshoptDecoder);

        // Scale bar elements
        for (let i = 0; i < 2; i++) {
            this._scaleBars[i] = document.getElementById(`overview-scale-${i}`);
        }

        // Axis indicator canvases
        for (let i = 0; i < 2; i++) {
            const container = document.getElementById(`overview-axis-${i}`);
            if (container) {
                this._axisCanvases[i] = container.querySelector('canvas');
            }
        }
        this._drawAxisIndicators();

        // Event listeners
        for (let i = 0; i < 2; i++) {
            this._canvases[i].addEventListener('click', (e) => this._onClick(i, e));
            this._canvases[i].addEventListener('wheel', (e) => this._onWheel(e), { passive: false });
        }
    }

    /**
     * Set world bounds, max zoom, and base URL. Reset crosshair to center.
     * @param {THREE.Box3} box3
     * @param {number} maxZoom
     * @param {string} baseUrl - e.g. '/tiles/2020-11-26/'
     */
    setBounds(box3, maxZoom, baseUrl) {
        this.bounds = box3.clone();
        this.maxZoom = maxZoom;
        this.baseUrl = baseUrl.endsWith('/') ? baseUrl : baseUrl + '/';

        // Center crosshair and snap to tile center
        this.bounds.getCenter(this.crosshairPos);
        this._snapToTileCenter();

        // Default zoom factor: show entire bounds
        const size = new THREE.Vector3();
        this.bounds.getSize(size);
        this.zoomFactor = Math.max(size.x, size.y, size.z) * 0.6;

        this._tilesLoaded = false;
        this._updateCameras();
        this._updateOverlays();
    }

    /**
     * Set the axis pair preset.
     * @param {'xy-xz' | 'xy-yz' | 'xz-yz'} pair
     */
    setAxisPair(pair) {
        if (!AXIS_PAIRS[pair]) return;
        this._axisPair = pair;
        this._updateLabels();
        this._updateCameras();
        this._updateOverlays();
    }

    /**
     * Set the neighbor ring (0, 1, or 2).
     * @param {number} ring
     */
    setRing(ring) {
        this._ring = Math.max(0, Math.min(2, ring));
        this._updateOverlays();
        this._fireSelectionChange();
    }

    /**
     * Set visible features.
     * @param {Set<string>} names
     */
    setSelectedFeatures(names) {
        this._selectedFeatures = new Set(names);
        this._applyFeatureVisibility();
    }

    /**
     * Set feature index metadata.
     * @param {Object} featureIndex - name -> {color, acronym, ...}
     */
    setFeatureIndex(featureIndex) {
        this._featureIndex = featureIndex || {};
    }

    /**
     * Load zoom-0 GLBs from the feature index into the overview scene.
     */
    async loadTiles() {
        if (!this._featureIndex || !this.baseUrl) return;

        // Clear existing tile meshes
        this._clearTiles();

        // Collect all unique zoom-0 URIs across all features
        const uris = new Set();
        for (const feat of Object.values(this._featureIndex)) {
            const z0 = feat.tiles?.['0'] || feat.tiles?.[0] || [];
            for (const uri of z0) uris.add(uri);
        }

        if (uris.size === 0) {
            console.log('[OverviewPanel] No zoom-0 tiles found');
            return;
        }

        console.log(`[OverviewPanel] Loading ${uris.size} zoom-0 tiles`);

        const promises = [...uris].map(uri => this._loadGLB(uri));
        await Promise.allSettled(promises);

        this._tilesLoaded = true;
        this._applyFeatureVisibility();
        console.log(`[OverviewPanel] Loaded. Features: ${Object.keys(this._meshByFeature).length}`);
    }

    /**
     * Render both panels. Call from the animation loop.
     */
    render() {
        for (let i = 0; i < 2; i++) {
            const r = this._renderers[i];
            const cam = this._cameras[i];
            if (!r || !cam) continue;
            r.render(this._scene, cam);
            this._updateScaleBar(i);
        }
    }

    /**
     * Update cameras and renderers on window resize.
     */
    resize() {
        const dpr = window.devicePixelRatio || 1;
        for (let i = 0; i < 2; i++) {
            const canvas = this._canvases[i];
            if (!canvas) continue;
            const rect = canvas.getBoundingClientRect();
            const w = Math.floor(rect.width);
            const h = Math.floor(rect.height);
            if (w === 0 || h === 0) continue;
            this._renderers[i].setSize(w * dpr, h * dpr, false);
        }
        this._updateCameras();
    }

    /**
     * Fire the selection change callback with current crosshair + ring.
     */
    _fireSelectionChange() {
        this.onSelectionChange(this.crosshairPos.clone(), this._ring);
    }

    // ---------------------------------------------------------------
    // Camera + overlay updates
    // ---------------------------------------------------------------

    /**
     * Reconfigure both orthographic cameras based on axis pair, crosshair
     * position, and zoom factor.
     */
    _updateCameras() {
        if (!this.bounds) return;
        const axes = AXIS_PAIRS[this._axisPair];
        if (!axes) return;

        for (let i = 0; i < 2; i++) {
            const cfg = AXIS_CFG[axes[i]];
            const cam = this._cameras[i];
            const canvas = this._canvases[i];
            if (!cam || !canvas) continue;

            const rect = canvas.getBoundingClientRect();
            const aspect = rect.width / (rect.height || 1);

            const halfH = this.zoomFactor;
            const halfW = halfH * aspect;

            cam.left = -halfW;
            cam.right = halfW;
            cam.top = halfH;
            cam.bottom = -halfH;
            cam.updateProjectionMatrix();

            // Position camera along projection direction, centered on crosshair
            const cp = this.crosshairPos;
            cam.position.set(
                cp.x + cfg.camDir[0] * 1e6,
                cp.y + cfg.camDir[1] * 1e6,
                cp.z + cfg.camDir[2] * 1e6,
            );
            cam.up.set(cfg.up[0], cfg.up[1], cfg.up[2]);
            cam.lookAt(cp.x, cp.y, cp.z);
        }
    }

    /**
     * Rebuild crosshair + selection box geometry for both panels.
     */
    _updateOverlays() {
        if (!this.bounds) return;
        const axes = AXIS_PAIRS[this._axisPair];
        if (!axes) return;

        for (let i = 0; i < 2; i++) {
            const group = this._overlayGroups[i];
            // Clear existing overlay children
            while (group.children.length > 0) {
                const child = group.children[0];
                if (child.geometry) child.geometry.dispose();
                if (child.material) child.material.dispose();
                group.remove(child);
            }

            const cfg = AXIS_CFG[axes[i]];
            const cp = this.crosshairPos;

            // --- Crosshair lines ---
            const crosshairMat = new THREE.LineBasicMaterial({
                color: 0x6fdfaf,
                depthTest: false,
            });

            const size = new THREE.Vector3();
            this.bounds.getSize(size);
            const extent = Math.max(size.x, size.y, size.z) * 2;

            // Horizontal line (along hAxis)
            const hPts = this._crosshairLine(cfg, cp, cfg.hAxis, extent);
            const hGeo = new THREE.BufferGeometry().setFromPoints(hPts);
            const hLine = new THREE.LineSegments(hGeo, crosshairMat);
            hLine.renderOrder = 999;
            hLine.layers.set(i + 1);
            group.add(hLine);

            // Vertical line (along vAxis)
            const vPts = this._crosshairLine(cfg, cp, cfg.vAxis, extent);
            const vGeo = new THREE.BufferGeometry().setFromPoints(vPts);
            const vLine = new THREE.LineSegments(vGeo, crosshairMat);
            vLine.renderOrder = 999;
            vLine.layers.set(i + 1);
            group.add(vLine);

            // --- Selection box ---
            const selBox = this._computeSelectionBox();
            if (selBox) {
                const boxMat = new THREE.LineBasicMaterial({
                    color: 0xffaa44,
                    depthTest: false,
                });
                const boxPts = this._selectionBoxPoints(cfg, selBox);
                const boxGeo = new THREE.BufferGeometry().setFromPoints(boxPts);
                const boxLine = new THREE.LineSegments(boxGeo, boxMat);
                boxLine.renderOrder = 1000;
                boxLine.layers.set(i + 1);
                group.add(boxLine);
            }
        }
    }

    /**
     * Create two points for a crosshair line through the crosshair position
     * along the specified axis.
     */
    _crosshairLine(cfg, center, axis, extent) {
        const p1 = center.clone();
        const p2 = center.clone();
        p1[axis] -= extent;
        p2[axis] += extent;
        return [p1, p2];
    }

    /**
     * Compute the 3D selection box in world coordinates based on
     * currentZoom, ring, and crosshairPos.
     * @returns {THREE.Box3|null}
     */
    _computeSelectionBox() {
        if (!this.bounds) return null;
        const z = this.currentZoom;
        const ring = this._ring;
        const n = Math.pow(2, z);

        const min = this.bounds.min;
        const max = this.bounds.max;
        const rangeX = max.x - min.x;
        const rangeY = max.y - min.y;
        const rangeZ = max.z - min.z;

        if (rangeX <= 0 || rangeY <= 0 || rangeZ <= 0) return null;

        // Center tile indices
        const tx = Math.min(Math.max(Math.floor((this.crosshairPos.x - min.x) / rangeX * n), 0), n - 1);
        const ty = Math.min(Math.max(Math.floor((this.crosshairPos.y - min.y) / rangeY * n), 0), n - 1);
        const td = Math.min(Math.max(Math.floor((this.crosshairPos.z - min.z) / rangeZ * n), 0), n - 1);

        // Selection tile range (clamped)
        const x0 = Math.max(tx - ring, 0);
        const x1 = Math.min(tx + ring + 1, n);
        const y0 = Math.max(ty - ring, 0);
        const y1 = Math.min(ty + ring + 1, n);
        const d0 = Math.max(td - ring, 0);
        const d1 = Math.min(td + ring + 1, n);

        // Convert tile range back to world coordinates
        const selBox = new THREE.Box3(
            new THREE.Vector3(
                min.x + (x0 / n) * rangeX,
                min.y + (y0 / n) * rangeY,
                min.z + (d0 / n) * rangeZ,
            ),
            new THREE.Vector3(
                min.x + (x1 / n) * rangeX,
                min.y + (y1 / n) * rangeY,
                min.z + (d1 / n) * rangeZ,
            ),
        );
        return selBox;
    }

    /**
     * Build LineSegments points for the 2D footprint of the selection box
     * as seen from the panel's projection axis.
     */
    _selectionBoxPoints(cfg, box) {
        const h = cfg.hAxis;
        const v = cfg.vAxis;

        // Get the 2D rectangle extents
        const hMin = box.min[h];
        const hMax = box.max[h];
        const vMin = box.min[v];
        const vMax = box.max[v];

        // Build 4 edges as line segments (8 points for LineSegments)
        // We need to place these in 3D at the crosshair's depth on the
        // third axis (the one not visible in this panel).
        const makePoint = (hVal, vVal) => {
            const p = this.crosshairPos.clone();
            p[h] = hVal;
            p[v] = vVal;
            return p;
        };

        const tl = makePoint(hMin, vMax);
        const tr = makePoint(hMax, vMax);
        const br = makePoint(hMax, vMin);
        const bl = makePoint(hMin, vMin);

        // LineSegments: pairs of vertices
        return [
            tl, tr,  // top
            tr, br,  // right
            br, bl,  // bottom
            bl, tl,  // left
        ];
    }

    // ---------------------------------------------------------------
    // Labels
    // ---------------------------------------------------------------

    /**
     * Update the scale bar for an overview panel based on its orthographic camera.
     */
    _updateScaleBar(panelIdx) {
        const el = this._scaleBars[panelIdx];
        if (!el) return;
        const cam = this._cameras[panelIdx];
        const canvas = this._canvases[panelIdx];
        if (!cam || !canvas) return;

        const rect = canvas.getBoundingClientRect();
        const widthPx = rect.width;
        if (widthPx === 0) return;

        // World units visible across the canvas width
        const worldWidth = cam.right - cam.left;
        const worldPerPx = worldWidth / widthPx;
        const targetPx = 100;
        const targetWorld = worldPerPx * targetPx;

        // Find largest nice number <= targetWorld
        let best = NICE[0];
        for (const n of NICE) {
            if (n <= targetWorld) best = n;
            else break;
        }

        const barPx = best / worldPerPx;
        el.querySelector('.scale-bar-line').style.width = barPx + 'px';
        el.querySelector('.scale-bar-label').textContent = formatDistance(best);
    }

    _updateLabels() {
        const axes = AXIS_PAIRS[this._axisPair];
        if (!axes) return;
        for (let i = 0; i < 2; i++) {
            if (this._labels[i]) {
                this._labels[i].textContent = AXIS_CFG[axes[i]].label;
            }
        }
        this._drawAxisIndicators();
    }

    /**
     * Draw small axis direction indicators on each overview panel.
     * Shows colored arrows for the horizontal and vertical axes.
     */
    _drawAxisIndicators() {
        const axes = AXIS_PAIRS[this._axisPair];
        if (!axes) return;

        const COLORS = { x: '#e74c3c', y: '#2ecc71', z: '#3498db' };

        for (let i = 0; i < 2; i++) {
            const canvas = this._axisCanvases[i];
            if (!canvas) continue;

            const ctx = canvas.getContext('2d');
            const w = canvas.width;
            const h = canvas.height;
            ctx.clearRect(0, 0, w, h);

            const cfg = AXIS_CFG[axes[i]];
            const cx = 30;
            const cy = 30;
            const len = 20;

            // Horizontal axis arrow (right)
            const hColor = COLORS[cfg.hAxis];
            ctx.strokeStyle = hColor;
            ctx.fillStyle = hColor;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(cx + len, cy);
            ctx.stroke();
            // Arrowhead
            ctx.beginPath();
            ctx.moveTo(cx + len + 4, cy);
            ctx.lineTo(cx + len - 2, cy - 3);
            ctx.lineTo(cx + len - 2, cy + 3);
            ctx.fill();
            // Label
            ctx.font = 'bold 10px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(cfg.hAxis.toUpperCase(), cx + len + 2, cy - 6);

            // Vertical axis arrow (up)
            const vColor = COLORS[cfg.vAxis];
            ctx.strokeStyle = vColor;
            ctx.fillStyle = vColor;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(cx, cy - len);
            ctx.stroke();
            // Arrowhead
            ctx.beginPath();
            ctx.moveTo(cx, cy - len - 4);
            ctx.lineTo(cx - 3, cy - len + 2);
            ctx.lineTo(cx + 3, cy - len + 2);
            ctx.fill();
            // Label
            ctx.textAlign = 'center';
            ctx.fillText(cfg.vAxis.toUpperCase(), cx + 8, cy - len - 2);

            // Origin dot
            ctx.fillStyle = '#aaa';
            ctx.beginPath();
            ctx.arc(cx, cy, 2, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    // ---------------------------------------------------------------
    // Event handlers
    // ---------------------------------------------------------------

    /**
     * Click in a panel: map pixel -> NDC -> world coords, snap to tile
     * center, and update the two visible axes of crosshairPos.
     */
    _onClick(panelIdx, event) {
        const canvas = this._canvases[panelIdx];
        const rect = canvas.getBoundingClientRect();
        const cam = this._cameras[panelIdx];

        // Pixel to NDC [-1, 1]
        const ndcX = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        const ndcY = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        // NDC to world using the orthographic camera's inverse projection
        const worldPos = new THREE.Vector3(ndcX, ndcY, 0).unproject(cam);

        // Determine which axes this panel controls
        const axes = AXIS_PAIRS[this._axisPair];
        const cfg = AXIS_CFG[axes[panelIdx]];

        // Update only the two visible axes
        this.crosshairPos[cfg.hAxis] = worldPos[cfg.hAxis];
        this.crosshairPos[cfg.vAxis] = worldPos[cfg.vAxis];

        // Snap crosshair to tile center so camera targets the tile middle
        this._snapToTileCenter();

        this._updateCameras();
        this._updateOverlays();
        this._fireSelectionChange();
    }

    /**
     * Snap crosshairPos to the center of the tile cell it falls in.
     */
    _snapToTileCenter() {
        if (!this.bounds) return;
        const n = Math.pow(2, this.currentZoom);
        const min = this.bounds.min;
        const max = this.bounds.max;

        for (const axis of ['x', 'y', 'z']) {
            const range = max[axis] - min[axis];
            if (range <= 0) continue;
            const t = Math.min(Math.max(
                Math.floor((this.crosshairPos[axis] - min[axis]) / range * n),
                0), n - 1);
            this.crosshairPos[axis] = min[axis] + (t + 0.5) / n * range;
        }
    }

    /**
     * Mouse wheel on either canvas: adjust shared zoom factor.
     */
    _onWheel(event) {
        event.preventDefault();
        const delta = event.deltaY > 0 ? 1.1 : 0.9;
        this.zoomFactor = Math.max(0.5, Math.min(50 * this._getMaxExtent(), this.zoomFactor * delta));
        this._updateCameras();
    }

    _getMaxExtent() {
        if (!this.bounds) return 1;
        const size = new THREE.Vector3();
        this.bounds.getSize(size);
        return Math.max(size.x, size.y, size.z);
    }

    // ---------------------------------------------------------------
    // Tile loading
    // ---------------------------------------------------------------

    _clearTiles() {
        // Remove all children from tileGroup
        while (this._tileGroup.children.length > 0) {
            const child = this._tileGroup.children[0];
            child.traverse(c => {
                if (c.isMesh) {
                    c.geometry?.dispose();
                    c.material?.dispose();
                }
            });
            this._tileGroup.remove(child);
        }
        this._meshByFeature = {};
        this._tilesLoaded = false;
    }

    /**
     * Load a single GLB into the overview scene.
     */
    async _loadGLB(uri) {
        try {
            const tileUri = uri.includes('.') ? uri : uri + '.glb';
            const url = this.baseUrl + tileUri + '?v=' + Date.now();
            const gltf = await this._loader.loadAsync(url);
            const group = gltf.scene;

            group.traverse(child => {
                if (!child.isMesh) return;
                // Assign to layer 0 so both cameras see it
                child.layers.set(0);

                const name = this._resolveName(child);
                if (!name) return;

                if (!this._meshByFeature[name]) this._meshByFeature[name] = [];
                this._meshByFeature[name].push(child);

                // Apply color from feature index
                const feat = this._featureIndex[name];
                const props = this._findProps(child);
                const color = feat?.color || props?.color;
                if (color) {
                    child.material = child.material.clone();
                    child.material.color.set(color);
                }
            });

            // Set group layer and add to tile group
            group.layers.set(0);
            group.traverse(c => { c.layers.set(0); });
            this._tileGroup.add(group);
        } catch (e) {
            console.warn(`[OverviewPanel] Failed to load ${uri}:`, e);
        }
    }

    /**
     * Resolve feature name from a mesh, matching TileManager priority:
     * 1. userData.name (if not matching /^feature_\d+$/)
     * 2. userData.acronym
     * 3. userData.instance
     * 4. userData.body_id (stringified)
     */
    _resolveName(obj) {
        const props = this._findProps(obj);
        if (!props) return null;

        const rawName = props.name;
        const name = (rawName && !/^feature_\d+$/.test(rawName) ? rawName : null)
            || props.acronym
            || props.instance
            || (props.body_id != null ? String(props.body_id) : '');

        return name || null;
    }

    /**
     * Walk up the scene graph to find userData with identifying properties.
     * Matches TileManager._findProps.
     */
    _findProps(obj) {
        let cur = obj;
        while (cur) {
            if (cur.userData && (cur.userData.name || cur.userData.acronym ||
                cur.userData.instance || cur.userData.body_id)) {
                return cur.userData;
            }
            cur = cur.parent;
        }
        return null;
    }

    // ---------------------------------------------------------------
    // Feature visibility
    // ---------------------------------------------------------------

    _applyFeatureVisibility() {
        const nSelected = this._selectedFeatures.size;
        const nTotal = Object.keys(this._featureIndex).length;
        // Show all when nothing selected OR when everything is selected
        const showAll = nSelected === 0 || nSelected >= nTotal;

        for (const [name, meshes] of Object.entries(this._meshByFeature)) {
            const visible = showAll || this._selectedFeatures.has(name);
            for (const mesh of meshes) {
                mesh.visible = visible;
            }
        }
    }
}
