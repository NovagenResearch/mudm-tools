/**
 * Feature-level LOD tile manager.
 *
 * Two modes:
 * - "dynamic": SSE-based zoom selection per feature, with hysteresis
 * - "forced": all features at a user-chosen zoom level
 *
 * Each selected feature is displayed at ONE zoom level (no mixed
 * resolutions within a single brain region). During zoom transitions,
 * old tiles stay visible until new tiles finish loading.
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { ogcBoxToBox3, computeSSE } from './BoundingVolume.js';

const MAX_CONCURRENT_LOADS = 6;
const DEFAULT_GPU_MB = 1024;
const SSE_THRESHOLD = 300;
const HYSTERESIS_FRAMES = 10;  // ~0.17s at 60fps
const STALE_FRAMES = 120;      // ~2s for stale cleanup
const SELECTION_WARN_THRESHOLD = 5000;  // warn above this many selected w/o spatial filter

/** Load states for tiles. */
const UNLOADED = 0, LOADING = 1, LOADED = 2, FAILED = 3;

class TileNode {
    constructor(json, depth, parent) {
        this.depth = depth;
        this.parent = parent;
        this.uri = json.content?.uri ?? null;
        this.geometricError = json.geometricError ?? 0;
        this.box3 = json.boundingVolume?.box
            ? ogcBoxToBox3(json.boundingVolume.box) : null;
        this.children = (json.children ?? [])
            .map(c => new TileNode(c, depth + 1, this));

        // Runtime
        this.loadState = UNLOADED;
        this.object3D = null;
        this.meshByFeature = {};   // featureName → [Mesh, ...]
        this.gpuBytes = 0;
        this.lastUsedFrame = 0;
    }
}

export class TileManager {
    constructor(scene, baseUrl) {
        this.scene = scene;
        this.baseUrl = baseUrl.endsWith('/') ? baseUrl : baseUrl + '/';
        this.loader = new GLTFLoader();

        // Configure Draco decoder for KHR_draco_mesh_compression GLBs
        const dracoLoader = new DRACOLoader();
        dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.7/');
        dracoLoader.setDecoderConfig({ type: 'wasm' });
        this.loader.setDRACOLoader(dracoLoader);

        // Configure meshoptimizer decoder for EXT_meshopt_compression GLBs
        this.loader.setMeshoptDecoder(MeshoptDecoder);

        /** Feature index: name → {color, acronym, ccf_id, tiles: {zoom: [uri]}} */
        this.featureIndex = {};
        this.maxZoom = 3;

        /** Real-world meters per world unit (for the scale bar); nm by default (EM datasets). */
        this.metersPerUnit = 1e-9;

        /** URI → TileNode for all nodes in the hierarchy. Indexed under multiple
         * keys per node (full uri, bare id, canonical z/x/y/d), so NEVER iterate
         * .values() for counting — use _allNodes (deduplicated) instead. */
        this.nodeByUri = new Map();
        /** Deduplicated set of every TileNode (one entry per node) for iteration
         * and byte/loaded-count stats. Avoids the 2-3x over-count that iterating
         * nodeByUri.values() would cause. */
        this._allNodes = new Set();

        /** Currently selected feature names. */
        this.selectedFeatures = new Set();

        /** LOD mode: 'dynamic' (SSE-based) or 'forced' (user-chosen zoom). */
        this.lodMode = 'forced';
        /** Zoom level for forced mode. */
        this.forcedZoom = 3;

        /** Main-view neuron opacity (0..1). 1 = solid. Applied per material. */
        this._opacity = 1.0;

        /** GPU memory budget in MB (configurable via slider). */
        this.maxGpuMB = DEFAULT_GPU_MB;

        // Feature-level LOD state
        this._featureState = new Map();  // name → {activeZoom, targetZoom, stableCount}

        // Geometric error per zoom level (precomputed from tileset hierarchy)
        this._geoErrorByZoom = [];

        // Tiles protected from eviction (active + transition target)
        this._protectedUris = new Set();
        // Estimated resident bytes of the protected set this frame (admission control).
        this._protectedBytes = 0;
        // Running-average decoded GPU bytes per tile, used to budget not-yet-loaded
        // tiles before they are fetched. Seeded at 2 MB; self-corrects as tiles load.
        this._avgTileBytes = 2 * 1024 * 1024;
        this._avgSamples = 0;

        // Traversal state
        this.root = null;
        this._frustum = new THREE.Frustum();
        this._projScreenMatrix = new THREE.Matrix4();
        this._pendingLoads = 0;
        this._loadQueue = [];
        this._frameNumber = 0;
        // Render-on-demand: set true whenever an async load/unload mutates the
        // scene, so the main loop knows to repaint the just-streamed tile.
        this._dirty = false;

        // Stats
        this.loadedCount = 0;
        this.visibleCount = 0;
        this.gpuMB = '0.0';
        this.zoomDistribution = '';  // e.g. "z2:3 z3:12"

        // Spatial filter (set by overview panel)
        this._spatialFilter = null;  // { center: THREE.Vector3, ring: number } or null

        // Color-by-attribute state
        this.colorByAttribute = null;    // attribute key (e.g. 'cell_type') or null
        this._colorPalette = new Map();  // attribute value → '#rrggbb'
    }

    /**
     * @param {object|null} featuresData - already-parsed features.json document.
     *   When provided, it is reused instead of re-fetching/re-parsing (the same
     *   payload is also consumed by FeatureSelector — see main.js loadPyramid).
     */
    async init(featuresData = null) {
        const tilesetResp = await fetch(this.baseUrl + 'tileset.json');
        const tileset = await tilesetResp.json();
        this.root = new TileNode(tileset.root, 0, null);
        this._indexNodes(this.root);
        this._precomputeGeoErrors();

        // Load tilejson3d.json for pyramid metadata (optional)
        try {
            const tjResp = await fetch(this.baseUrl + 'tilejson3d.json');
            if (tjResp.ok) {
                const tjData = await tjResp.json();
                this.maxZoom = tjData.maxzoom ?? this.maxZoom;
                this.metersPerUnit = tjData.meters_per_unit ?? this.metersPerUnit;
                this.idFields = new Set(tjData.id_fields ?? []);
                this._encodings = tjData.encodings ?? null;
            }
        } catch (_) {
            // tilejson3d.json not available — fall back to features.json metadata
        }

        let data = featuresData;
        if (!data) {
            const featResp = await fetch(this.baseUrl + 'features.json');
            data = await featResp.json();
        }
        // Fall back to features.json for maxZoom and idFields if tilejson3d not loaded
        if (!this.idFields || this.idFields.size === 0) {
            const collProps = data.properties ?? {};
            this.maxZoom = collProps.max_zoom ?? data.max_zoom ?? this.maxZoom;
            this.idFields = new Set(collProps.id_fields ?? data.id_fields ?? []);
        }
        if (Array.isArray(data.features)) {
            // MicroJSON format: array of {type, id, geometry, properties}
            this.featureIndex = {};
            for (const feat of data.features) {
                const name = feat.id ?? feat.properties?.name ?? '';
                if (!name) continue;
                // Build zoom-keyed tile dict from geometry.tiles (flat list)
                const tiles = feat.geometry?.tiles ?? [];
                const byZoom = {};
                for (const t of tiles) {
                    const z = t.split('/')[0];
                    (byZoom[z] ??= []).push(t);
                }
                this.featureIndex[name] = { ...feat.properties, tiles: byZoom };
            }
        } else {
            // Legacy format: dict keyed by name
            this.featureIndex = data.features;
        }

        console.log(
            `[TileManager] init: ${this.nodeByUri.size} tiles indexed, ` +
            `${Object.keys(this.featureIndex).length} features, ` +
            `maxZoom=${this.maxZoom}, ` +
            `geoErrors=[${this._geoErrorByZoom.map(e => e.toFixed(0)).join(', ')}]`
        );
    }

    /**
     * Switch to a different pyramid. Unloads everything and re-inits.
     * @param {string} newBaseUrl - new base URL for tiles (e.g. '/tiles/2020-11-26/3dtiles/')
     * @param {object|null} featuresData - already-parsed features.json to reuse (optional)
     */
    async switchPyramid(newBaseUrl, featuresData = null) {
        // Unload all tiles
        for (const node of this._allNodes) {
            if (node.loadState === LOADED) {
                this._unloadNode(node);
            }
        }

        // Clear all state
        this.nodeByUri.clear();
        this._allNodes.clear();
        this.root = null;
        this.featureIndex = {};
        this.selectedFeatures = new Set();
        this._featureState.clear();
        this._protectedUris.clear();
        this._spatialFilter = null;
        this._loadQueue = [];
        this._pendingLoads = 0;
        this._frameNumber = 0;
        this._geoErrorByZoom = [];
        this.colorByAttribute = null;
        this._colorPalette = new Map();

        // Set new URL and re-init
        this.baseUrl = newBaseUrl.endsWith('/') ? newBaseUrl : newBaseUrl + '/';
        await this.init(featuresData);
    }

    _indexNodes(node) {
        if (node.uri) {
            this._allNodes.add(node);
            this.nodeByUri.set(node.uri, node);
            // Also index by bare tile ID (without extension) for new format
            const bareId = node.uri.replace(/\.[^.]+$/, '');
            this.nodeByUri.set(bareId, node);
            // And by the canonical "z/x/y/d" tile id, stripping any directory prefix the
            // tileset bakes into content.uri. A static-serving root tileset roots its URIs
            // under "3dtiles/" (so plain HTTP/S3 can resolve the GLB without /tiles routing),
            // but features.json references tiles by bare "z/x/y/d". Indexing the canonical id
            // lets feature->tile lookups resolve regardless of how the tileset is rooted,
            // while node.uri (possibly prefixed) is still used to fetch the GLB.
            const canon = bareId.match(/(\d+\/\d+\/\d+\/\d+)$/);
            if (canon && canon[1] !== bareId) this.nodeByUri.set(canon[1], node);
        }
        for (const child of node.children) this._indexNodes(child);
    }

    _precomputeGeoErrors() {
        this._geoErrorByZoom = [];
        const collect = (node, depth) => {
            while (this._geoErrorByZoom.length <= depth) this._geoErrorByZoom.push(0);
            this._geoErrorByZoom[depth] = Math.max(
                this._geoErrorByZoom[depth], node.geometricError,
            );
            for (const child of node.children) collect(child, depth + 1);
        };
        collect(this.root, 0);
    }

    /**
     * Called when feature selection changes.
     */
    setSelectedFeatures(selectedNames) {
        this.selectedFeatures = new Set(selectedNames);
        // Clean up state for deselected features
        for (const name of this._featureState.keys()) {
            if (!this.selectedFeatures.has(name)) {
                this._featureState.delete(name);
            }
        }
    }

    /**
     * Set main-view neuron opacity (0..1). Applies to all loaded meshes and is
     * remembered for tiles loaded later (see _loadTile). 1 = solid/opaque.
     */
    setOpacity(o) {
        this._opacity = o;
        const transparent = o < 1;
        for (const node of this._allNodes) {
            node.object3D?.traverse(c => {
                if (!c.isMesh || !c.material) return;
                c.material.transparent = transparent;
                c.material.opacity = o;
                c.material.depthWrite = !transparent;
                c.material.needsUpdate = true;
            });
        }
        this._dirty = true;
    }

    /**
     * Set spatial filter for tile loading. When set, only tiles within
     * the selection cube (center tile + ring neighbors) are loaded.
     * @param {THREE.Vector3|null} worldCenter - crosshair world position, or null to clear
     * @param {number} ring - neighbor ring count (0, 1, or 2)
     */
    setSpatialFilter(worldCenter, ring) {
        const wasActive = !!this._spatialFilter;
        this._spatialFilter = worldCenter ? { center: worldCenter, ring } : null;
        const isActive = !!this._spatialFilter;

        // When switching filter on/off, reset per-feature LOD state so
        // stale activeZoom values from the previous mode don't persist.
        if (wasActive !== isActive) {
            for (const state of this._featureState.values()) {
                state.activeZoom = null;
                state.targetZoom = null;
                state.stableCount = 0;
            }
        }
    }

    /**
     * Parse tile coordinates from a URI like "3/2/1/5.glb".
     * @returns {{z: number, x: number, y: number, d: number}|null}
     */
    _parseTileUri(uri) {
        const match = uri.match(/^(\d+)\/(\d+)\/(\d+)\/(\d+)(?:\.glb)?$/);
        if (!match) return null;
        return { z: +match[1], x: +match[2], y: +match[3], d: +match[4] };
    }

    /**
     * Check if a tile URI is within the spatial selection cube.
     * Converts the world-space crosshair to tile coordinates at the tile's zoom level.
     */
    _isTileInSelection(uri) {
        if (!this._spatialFilter || !this.root?.box3) return true;
        const tile = this._parseTileUri(uri);
        if (!tile) return true;

        const bounds = this.root.box3;
        const min = bounds.min;
        const range = new THREE.Vector3();
        bounds.getSize(range);
        const n = Math.pow(2, tile.z);

        const cx = Math.min(Math.floor((this._spatialFilter.center.x - min.x) / range.x * n), n - 1);
        const cy = Math.min(Math.floor((this._spatialFilter.center.y - min.y) / range.y * n), n - 1);
        const cd = Math.min(Math.floor((this._spatialFilter.center.z - min.z) / range.z * n), n - 1);

        const r = this._spatialFilter.ring;
        return Math.abs(tile.x - cx) <= r
            && Math.abs(tile.y - cy) <= r
            && Math.abs(tile.d - cd) <= r;
    }

    /**
     * Set color-by-attribute mode. Pass null to restore original colors.
     * @param {string|null} attribute - feature metadata key to color by
     * @param {Map<string, string>|null} palette - value → hex color map
     */
    setColorBy(attribute, palette) {
        this.colorByAttribute = attribute;
        this._colorPalette = palette || new Map();
        this._recolorAll();
    }

    /**
     * Get the effective color for a feature, considering color-by-attribute.
     * @returns {string|null} hex color or null if no color determined
     */
    _getFeatureColor(name) {
        if (this.colorByAttribute) {
            const feat = this.featureIndex[name];
            if (feat) {
                const val = String(feat[this.colorByAttribute] ?? '');
                if (this._colorPalette.has(val)) {
                    return this._colorPalette.get(val);
                }
            }
            return '#555555'; // no value for this attribute
        }
        // Original color
        const feat = this.featureIndex[name];
        return feat?.color || null;
    }

    /**
     * Recolor all loaded meshes based on current color-by setting.
     */
    _recolorAll() {
        for (const node of this._allNodes) {
            if (node.loadState !== LOADED) continue;
            for (const [name, meshes] of Object.entries(node.meshByFeature)) {
                const color = this._getFeatureColor(name);
                if (!color) continue;
                for (const mesh of meshes) {
                    if (mesh.material) {
                        mesh.material.color.set(color);
                    }
                }
            }
        }
    }

    _getFeatureState(name) {
        if (!this._featureState.has(name)) {
            this._featureState.set(name, {
                activeZoom: null,
                targetZoom: null,
                stableCount: 0,
            });
        }
        return this._featureState.get(name);
    }

    // ----- Per-frame update ---------------------------------------------------

    update(camera) {
        if (!this.root) return;
        this._frameNumber++;

        // Ensure camera matrices are current (OrbitControls.update() does NOT
        // call updateMatrixWorld, so matrixWorldInverse can be stale).
        camera.updateMatrixWorld(true);

        // Update frustum from current camera matrices
        this._projScreenMatrix.multiplyMatrices(
            camera.projectionMatrix, camera.matrixWorldInverse,
        );
        this._frustum.setFromProjectionMatrix(this._projScreenMatrix);

        const screenHeight = window.innerHeight;
        const fov = camera.fov * Math.PI / 180;

        // Phase 0: hide all loaded tiles + meshes
        for (const node of this._allNodes) {
            if (!node.object3D) continue;
            node.object3D.visible = false;
            for (const meshes of Object.values(node.meshByFeature)) {
                for (const mesh of meshes) mesh.visible = false;
            }
        }

        this.visibleCount = 0;
        const zoomCounts = {};
        this._protectedUris.clear();
        this._protectedBytes = 0;
        const maxBytes = this.maxGpuMB * 1024 * 1024;

        // Step 8: rebuild the load queue from scratch each frame so it only ever
        // holds THIS frame's still-wanted tiles — no unbounded/stale backlog, and
        // it re-prioritizes naturally as the camera/selection change. Tiles that
        // were merely QUEUED (not yet in-flight) revert to UNLOADED so this frame
        // can re-decide whether they are still wanted; in-flight loads (counted in
        // _pendingLoads, already removed from the queue) are untouched.
        for (const node of this._loadQueue) {
            if (node.loadState === LOADING) node.loadState = UNLOADED;
        }
        this._loadQueue.length = 0;

        // Step 6: at very large selections with no spatial filter, warn. Admission
        // control below still bounds GPU memory, but a spatial filter loads faster
        // and more focused. Throttled so it does not spam the console.
        if (!this._spatialFilter &&
            this.selectedFeatures.size > SELECTION_WARN_THRESHOLD &&
            this._frameNumber % 180 === 1) {
            console.warn(
                `[TileManager] ${this.selectedFeatures.size} features selected with no ` +
                `spatial filter. GPU residency is capped at ${this.maxGpuMB} MB (nearest ` +
                `tiles win); enable a spatial filter for faster, more focused loading.`
            );
        }

        // Phase 1: per-feature zoom decision + tile display
        for (const name of this.selectedFeatures) {
            const feat = this.featureIndex[name];
            if (!feat) continue;

            const state = this._getFeatureState(name);

            // Compute target zoom
            let committedZoom;
            if (this.lodMode === 'forced') {
                committedZoom = this.forcedZoom;
                state.targetZoom = committedZoom;
                state.stableCount = HYSTERESIS_FRAMES;
            } else {
                const idealZoom = this._computeIdealZoom(
                    name, camera.position, screenHeight, fov,
                );
                committedZoom = this._applyHysteresis(state, idealZoom);
            }

            // Find best available zoom for this feature
            let desiredZoom = this._bestAvailableZoom(feat, committedZoom);
            if (desiredZoom === null) continue;

            // Step 7: admission control. Estimate the cost of this feature's
            // in-frustum tiles at the desired zoom. If admitting them would exceed
            // the GPU budget, downgrade to the coarsest available zoom (far fewer
            // tiles); if even that will not fit, admit nothing new this frame so
            // resident memory stays bounded (already-loaded tiles still show but
            // remain evictable). This is what stops the zoomed-out + everything-
            // selected case from streaming toward an OOM.
            let desiredUris = this._frustumFilter(feat.tiles[String(desiredZoom)] || []);
            let cost = this._newCost(desiredUris);
            let admit = true;
            if (this._protectedBytes + cost > maxBytes) {
                const coarsest = this._coarsestZoom(feat);
                if (coarsest !== null && coarsest !== desiredZoom) {
                    const coarseUris = this._frustumFilter(feat.tiles[String(coarsest)] || []);
                    const coarseCost = this._newCost(coarseUris);
                    if (coarseUris.length && this._protectedBytes + coarseCost <= maxBytes) {
                        desiredZoom = coarsest;
                        desiredUris = coarseUris;
                        cost = coarseCost;
                    } else {
                        admit = false;
                    }
                } else {
                    admit = false;
                }
            }

            if (!admit) {
                // Over budget: do not protect or enqueue new tiles for this feature.
                // Keep showing whatever is already loaded so it does not vanish, but
                // leave it evictable so the budget can be honored.
                this._showAnyLoadedZoom(feat, name, state);
                const dz = state.activeZoom ?? desiredZoom;
                zoomCounts[dz] = (zoomCounts[dz] || 0) + 1;
                continue;
            }

            // Protect the admitted (in-frustum, budgeted) tiles from eviction,
            // billing each unique tile to the budget exactly once.
            for (const uri of desiredUris) {
                if (!this._protectedUris.has(uri)) {
                    this._protectedUris.add(uri);
                    this._protectedBytes += this._estBytesOne(uri);
                }
            }
            // When a spatial filter is active, also protect the active-zoom fallback
            // we may still be showing during a transition.
            if (!this._spatialFilter || state.activeZoom >= desiredZoom) {
                this._protectTiles(feat, state.activeZoom);
            }

            // Are all in-frustum tiles at desired zoom loaded?
            const allLoaded = desiredUris.length > 0 && desiredUris.every(uri => {
                const node = this.nodeByUri.get(uri);
                return node?.loadState === LOADED;
            });

            if (allLoaded) {
                // Transition complete
                state.activeZoom = desiredZoom;
                for (const uri of desiredUris) {
                    this._showTileFeature(uri, name);
                }
            } else {
                // Enqueue desired tiles for loading
                for (const uri of desiredUris) {
                    const node = this.nodeByUri.get(uri);
                    if (node?.loadState === UNLOADED) this._enqueueLoad(node);
                }

                // Show any already-loaded desired-zoom tiles (progressive).
                // This updates their lastUsedFrame so they survive eviction
                // and gives the user visual progress during long transitions.
                for (const uri of desiredUris) {
                    this._showTileFeature(uri, name);
                }

                // Also show fallback at current activeZoom for full coverage.
                // When spatial filter is active, skip coarser zoom fallbacks —
                // they cover the entire dataset and would show content outside
                // the selection region.
                if (state.activeZoom !== null &&
                    (!this._spatialFilter || state.activeZoom >= desiredZoom)) {
                    const fallbackUris = this._frustumFilter(
                        feat.tiles[String(state.activeZoom)] || [],
                    );
                    for (const uri of fallbackUris) {
                        this._showTileFeature(uri, name);
                    }
                } else if (!this._spatialFilter) {
                    // No activeZoom yet — show any loaded zoom as initial fallback
                    // (only without spatial filter, to avoid showing global tiles)
                    this._showAnyLoadedZoom(feat, name, state);
                }
            }

            // Track zoom distribution
            const displayZoom = state.activeZoom ?? desiredZoom;
            zoomCounts[displayZoom] = (zoomCounts[displayZoom] || 0) + 1;
        }

        // Build zoom stats string
        this.zoomDistribution = Object.entries(zoomCounts)
            .sort((a, b) => a[0] - b[0])
            .map(([z, n]) => `z${z}:${n}`)
            .join(' ');

        this._processQueue();
        this._evict();
        this._recalcStats();

        // Debug log once per second
        if (this._frameNumber % 60 === 0 && this.selectedFeatures.size > 0) {
            const firstName = [...this.selectedFeatures][0];
            const fs = this._featureState.get(firstName);
            console.log(
                `[TileManager] mode=${this.lodMode} forcedZoom=${this.forcedZoom} ` +
                `selected=${this.selectedFeatures.size} ` +
                `loaded=${this.loadedCount} visible=${this.visibleCount} ` +
                `pending=${this._pendingLoads} queue=${this._loadQueue.length} ` +
                `protected=${this._protectedUris.size} ` +
                `| "${firstName}": active=${fs?.activeZoom} target=${fs?.targetZoom} ` +
                `dist=${this.zoomDistribution}`
            );
        }
    }

    /**
     * Mark tiles for a feature at a given zoom as protected from eviction.
     * When a spatial filter is active, only tiles inside the selection are protected.
     */
    _protectTiles(feat, zoom) {
        if (zoom === null || zoom === undefined) return;
        const uris = feat.tiles[String(zoom)];
        if (!uris) return;
        for (const uri of uris) {
            if (this._isTileInSelection(uri)) {
                this._protectedUris.add(uri);
            }
        }
    }

    /**
     * Filter URIs to only those whose tile bounding box intersects the frustum
     * AND falls within the spatial selection (if active).
     */
    _frustumFilter(uris) {
        return uris.filter(uri => {
            const node = this.nodeByUri.get(uri);
            if (!node?.box3 || !this._frustum.intersectsBox(node.box3)) return false;
            return this._isTileInSelection(uri);
        });
    }

    /**
     * Show a specific feature's meshes within a tile.
     */
    _showTileFeature(uri, featureName) {
        const node = this.nodeByUri.get(uri);
        if (!node || node.loadState !== LOADED || !node.object3D) return;

        node.lastUsedFrame = this._frameNumber;
        node.object3D.visible = true;

        const meshes = node.meshByFeature[featureName];
        if (meshes) {
            for (const mesh of meshes) {
                mesh.visible = true;
                this.visibleCount++;
            }
        }
    }

    /**
     * Fallback: find any loaded zoom for a feature and show it.
     */
    _showAnyLoadedZoom(feat, name, state) {
        // Prefer higher zoom levels (more detail)
        const zooms = Object.keys(feat.tiles).map(Number).sort((a, b) => b - a);
        for (const z of zooms) {
            const uris = this._frustumFilter(feat.tiles[String(z)] || []);
            const loadedUris = uris.filter(uri => {
                const node = this.nodeByUri.get(uri);
                return node?.loadState === LOADED;
            });
            if (loadedUris.length > 0) {
                state.activeZoom = z;
                for (const uri of loadedUris) {
                    this._showTileFeature(uri, name);
                }
                return;
            }
        }
    }

    // ----- LOD computation ----------------------------------------------------

    /**
     * Compute ideal zoom for a feature based on camera distance (dynamic mode).
     */
    _computeIdealZoom(name, cameraPos, screenHeight, fov) {
        const feat = this.featureIndex[name];

        // Find minimum distance to any tile of this feature
        let minDist = Infinity;
        for (const uris of Object.values(feat.tiles)) {
            for (const uri of uris) {
                const node = this.nodeByUri.get(uri);
                if (node?.box3) {
                    const d = node.box3.distanceToPoint(cameraPos);
                    if (d < minDist) minDist = d;
                }
            }
        }
        if (minDist === Infinity) return this.maxZoom;
        minDist = Math.max(minDist, 1.0);

        // Walk from coarsest to finest: first zoom where SSE <= threshold
        for (let z = 0; z <= this.maxZoom; z++) {
            const geoError = this._geoErrorByZoom[z] ?? 0;
            const sse = computeSSE(geoError, minDist, screenHeight, fov);
            if (sse <= SSE_THRESHOLD) return z;
        }
        return this.maxZoom;
    }

    /**
     * Prevent zoom flickering by requiring a stable target for several frames.
     */
    _applyHysteresis(state, idealZoom) {
        if (state.targetZoom === null) {
            state.targetZoom = idealZoom;
            state.stableCount = HYSTERESIS_FRAMES;
            return idealZoom;
        }

        if (idealZoom === state.targetZoom) {
            state.stableCount = Math.min(state.stableCount + 1, HYSTERESIS_FRAMES);
        } else {
            state.targetZoom = idealZoom;
            state.stableCount = 0;
        }

        if (state.stableCount >= HYSTERESIS_FRAMES) {
            return state.targetZoom;
        }

        // Not stable yet: keep current active zoom
        return state.activeZoom !== null ? state.activeZoom : state.targetZoom;
    }

    /**
     * Find the closest available zoom level for a feature.
     */
    _bestAvailableZoom(feat, targetZoom) {
        const available = Object.keys(feat.tiles).map(Number);
        if (available.length === 0) return null;

        let best = available[0];
        let bestDiff = Math.abs(best - targetZoom);
        for (const z of available) {
            const diff = Math.abs(z - targetZoom);
            if (diff < bestDiff || (diff === bestDiff && z > best)) {
                best = z;
                bestDiff = diff;
            }
        }
        return best;
    }

    /** Estimated resident GPU bytes for ONE tile (real size if loaded, else the
     * running average). */
    _estBytesOne(uri) {
        const node = this.nodeByUri.get(uri);
        return (node && node.loadState === LOADED && node.gpuBytes > 0)
            ? node.gpuBytes : this._avgTileBytes;
    }

    /** Cost (estimated bytes) of admitting these URIs, counting only tiles NOT
     * already protected this frame. A single tile holds many features' meshes, so
     * a shared tile must be billed to the budget once, not once per feature. */
    _newCost(uris) {
        let bytes = 0;
        for (const uri of uris) {
            if (!this._protectedUris.has(uri)) bytes += this._estBytesOne(uri);
        }
        return bytes;
    }

    /** Coarsest (numerically smallest) zoom level available for a feature. */
    _coarsestZoom(feat) {
        let min = null;
        for (const z of Object.keys(feat.tiles)) {
            const n = Number(z);
            if (min === null || n < min) min = n;
        }
        return min;
    }

    // ----- Tile loading -------------------------------------------------------

    _enqueueLoad(node) {
        if (node.loadState !== UNLOADED) return;
        node.loadState = LOADING;
        this._loadQueue.push(node);
    }

    _processQueue() {
        while (this._loadQueue.length > 0 && this._pendingLoads < MAX_CONCURRENT_LOADS) {
            const node = this._loadQueue.shift();
            if (node.loadState !== LOADING) continue;
            this._pendingLoads++;
            this._loadTile(node);
        }
    }

    async _loadTile(node) {
        try {
            const gltf = await this.loader.loadAsync(this.baseUrl + node.uri);
            const group = gltf.scene;

            // Index meshes by feature name + apply colors
            let meshCount = 0, namedCount = 0;
            group.traverse(child => {
                if (!child.isMesh) return;
                child.visible = false; // hide all meshes by default
                // Neuron meshes never move — freeze the local matrix so three.js
                // skips the per-frame world-matrix recompute across thousands of
                // static objects. (Transforms are already baked by the loader.)
                child.matrixAutoUpdate = false;
                child.updateMatrix();
                meshCount++;
                const props = this._findProps(child);
                if (!props) return;
                const rawName = props.name;
                const name = (rawName && !/^feature_\d+$/.test(rawName) ? rawName : null)
                    || props.acronym || props.instance
                    || (props.body_id != null ? String(props.body_id) : '');
                if (!name) return;
                namedCount++;

                if (!node.meshByFeature[name]) node.meshByFeature[name] = [];
                node.meshByFeature[name].push(child);
                child.userData._featureName = name;

                // Color: use palette if color-by is active, otherwise original.
                // Also apply the current global opacity (clone so it's per-mesh).
                const color = this._getFeatureColor(name) || props.color;
                if (color || this._opacity < 1) {
                    child.material = child.material.clone();
                    if (color) child.material.color.set(color);
                    if (this._opacity < 1) {
                        child.material.transparent = true;
                        child.material.opacity = this._opacity;
                        child.material.depthWrite = false;
                    }
                }

                // GPU accounting
                if (child.geometry) {
                    for (const attr of Object.values(child.geometry.attributes)) {
                        node.gpuBytes += attr.array.byteLength;
                    }
                    if (child.geometry.index) {
                        node.gpuBytes += child.geometry.index.array.byteLength;
                    }
                }
            });

            console.log(`[debug] ${node.uri}: ${meshCount} meshes, ${namedCount} named, features: [${Object.keys(node.meshByFeature).slice(0,3).join(', ')}${Object.keys(node.meshByFeature).length > 3 ? '...' : ''}]`);
            group.visible = false;
            this.scene.add(group);
            node.object3D = group;
            node.loadState = LOADED;
            // Maintain a running average tile size for admission-control estimates.
            if (node.gpuBytes > 0) {
                this._avgSamples++;
                this._avgTileBytes += (node.gpuBytes - this._avgTileBytes) / this._avgSamples;
            }
            // Give newly loaded tiles a recent timestamp so they survive
            // stale eviction long enough for transition to complete.
            node.lastUsedFrame = this._frameNumber;
        } catch (e) {
            console.warn(`Failed to load ${node.uri}:`, e);
            node.loadState = FAILED;
        } finally {
            this._pendingLoads--;
            this._dirty = true;   // async load completed → repaint once
        }
    }

    _findProps(obj) {
        let cur = obj;
        while (cur) {
            if (cur.userData && (cur.userData.name || cur.userData.acronym || cur.userData.instance || cur.userData.body_id)) {
                return cur.userData;
            }
            cur = cur.parent;
        }
        return null;
    }

    // ----- Eviction -----------------------------------------------------------

    _evict() {
        // Pass 0: when spatial filter is active, immediately unload tiles
        // outside the selection to free GPU memory for large datasets.
        if (this._spatialFilter) {
            for (const node of this._allNodes) {
                if (node.loadState !== LOADED) continue;
                if (this._protectedUris.has(node.uri)) continue;
                if (!this._isTileInSelection(node.uri)) {
                    this._unloadNode(node);
                }
            }
        }

        // Pass 1: unload stale tiles NOT protected by active transitions
        for (const node of this._allNodes) {
            if (node.loadState !== LOADED) continue;
            // Never evict tiles needed for active display or pending transition
            if (this._protectedUris.has(node.uri)) continue;
            if (this._frameNumber - node.lastUsedFrame > STALE_FRAMES) {
                this._unloadNode(node);
            }
        }

        // Pass 2: LRU evict if over GPU budget (still respects protection)
        let total = 0;
        const loaded = [];
        for (const node of this._allNodes) {
            if (node.loadState === LOADED) {
                total += node.gpuBytes;
                loaded.push(node);
            }
        }
        const maxBytes = this.maxGpuMB * 1024 * 1024;
        if (total <= maxBytes) return;

        // Evict UNPROTECTED tiles (LRU first) until under budget. Admission control
        // guarantees the protected set already fits the budget, so anything
        // unprotected is expendable — even if it was shown this frame (a non-admitted
        // feature's leftover tiles). The old `lastUsedFrame < frameNumber` guard made
        // shown-but-unprotected tiles un-evictable, which defeated the budget.
        loaded
            .filter(n => !this._protectedUris.has(n.uri))
            .sort((a, b) => a.lastUsedFrame - b.lastUsedFrame)
            .forEach(node => {
                if (total <= maxBytes) return;
                total -= node.gpuBytes;
                this._unloadNode(node);
            });
    }

    _unloadNode(node) {
        if (node.object3D) {
            node.object3D.traverse(c => {
                if (c.isMesh) {
                    c.geometry?.dispose();
                    c.material?.dispose();
                }
            });
            this.scene.remove(node.object3D);
            node.object3D = null;
        }
        node.meshByFeature = {};
        node.gpuBytes = 0;
        node.loadState = UNLOADED;
        this._dirty = true;   // scene changed → repaint
    }

    _recalcStats() {
        let total = 0, count = 0;
        for (const node of this._allNodes) {
            if (node.loadState === LOADED) { total += node.gpuBytes; count++; }
        }
        this.loadedCount = count;
        this.gpuMB = (total / (1024 * 1024)).toFixed(1);
    }
}
