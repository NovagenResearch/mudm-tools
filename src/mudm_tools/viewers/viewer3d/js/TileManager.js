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

const _batchColor = new THREE.Color();        // scratch for setColorAt
const _white = new THREE.Color(0xffffff);     // scratch brighten-target for hover highlight
const _hl = new THREE.Color();                // scratch for hover highlight
const MAX_DRAW_OBJECTS_DEFAULT = 40000;       // default rendered-instance ceiling (tunable via slider)
const MAX_INSTANCES_PER_BATCH = 50000;        // refuse batching a single tile larger than this (z0 guard)
const MAX_BATCH_BYTES = 256 * 1024 * 1024;    // and refuse if its combined vertex+index buffer exceeds this
                                              // — a BatchedMesh pre-allocates one contiguous buffer, so an
                                              // oversized coarse tile would throw "Array buffer allocation failed".

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
        // --- BatchedMesh path (triangle features) ---
        this.batched = [];            // [THREE.BatchedMesh] — usually 1 per tile
        this.instanceByFeature = {};  // featureName -> [{ batch, id }]  (>=1; regions can split)
        this.instanceCount = 0;       // total admitted draw-objects in this tile (for the budget)
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

        // --- Per-frame hot-path caches (decouple update() cost from selection size) ---
        // Parsed tile coords per URI (a URI->{z,x,y,d} mapping never changes).
        this._coordCache = new Map();
        // Spatial-filter center tile {cx,cy,cd} per zoom; rebuilt when the filter moves.
        this._selCenterByZoom = [];
        // Pre-culled list of selected features that actually have a tile inside the
        // current spatial box at their display zoom — the only features that can
        // render under a filter. null = "no filter, iterate every selected feature".
        // Recomputed only when the selection / filter / forced zoom changes, NOT per
        // frame, so the per-frame loop is O(features-in-box) not O(features-selected).
        this._activeFeatures = null;
        this._activeDirty = true;
        this._activeForcedZoom = -1;
        this._filterToken = 0;        // bumped on every setSpatialFilter (incl. moves)
        this._activeFilterToken = -1;

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
        // Hot-path caches keyed by the old pyramid's URIs/features — drop them.
        this._coordCache.clear();
        this._selCenterByZoom = [];
        this._activeFeatures = null;
        this._activeDirty = true;
        this._filterToken++;

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
        this._activeDirty = true;   // selection changed → re-cull active features
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
            for (const batch of node.batched) {
                const m = batch.material;
                m.transparent = transparent; m.opacity = o; m.depthWrite = !transparent; m.needsUpdate = true;
            }
            node.object3D?.traverse(c => { if (c.isMesh && c.material) {
                c.material.transparent = transparent; c.material.opacity = o; c.material.depthWrite = !transparent; c.material.needsUpdate = true; } });
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

        // The box moved (or toggled): invalidate the per-zoom center cache and the
        // pre-culled active-feature list so both are rebuilt against the new box.
        this._selCenterByZoom = [];
        this._filterToken++;
        this._activeDirty = true;

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

    /** Parse + memoize tile coords for a URI (a URI's z/x/y/d never change), so the
     * per-frame selection/eviction loops don't re-run the regex each call. */
    _coords(uri) {
        let c = this._coordCache.get(uri);
        if (c === undefined) {
            c = this._parseTileUri(uri);
            this._coordCache.set(uri, c);
        }
        return c;
    }

    /** Spatial-filter center tile {cx,cy,cd} at a zoom level, cached per filter move
     * (the crosshair is fixed within a frame, so this is computed at most once per
     * zoom per box rather than once per tile per frame). */
    _selCenterAtZoom(z) {
        let c = this._selCenterByZoom[z];
        if (c) return c;
        const bounds = this.root.box3;
        const min = bounds.min;
        const sx = bounds.max.x - min.x, sy = bounds.max.y - min.y, sz = bounds.max.z - min.z;
        const n = Math.pow(2, z);
        const center = this._spatialFilter.center;
        c = {
            cx: Math.min(Math.floor((center.x - min.x) / sx * n), n - 1),
            cy: Math.min(Math.floor((center.y - min.y) / sy * n), n - 1),
            cd: Math.min(Math.floor((center.z - min.z) / sz * n), n - 1),
        };
        this._selCenterByZoom[z] = c;
        return c;
    }

    /**
     * Check if a tile URI is within the spatial selection cube.
     * Converts the world-space crosshair to tile coordinates at the tile's zoom level.
     */
    _isTileInSelection(uri) {
        if (!this._spatialFilter || !this.root?.box3) return true;
        const tile = this._coords(uri);
        if (!tile) return true;

        const c = this._selCenterAtZoom(tile.z);
        const r = this._spatialFilter.ring;
        return Math.abs(tile.x - c.cx) <= r
            && Math.abs(tile.y - c.cy) <= r
            && Math.abs(tile.d - c.cd) <= r;
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
            for (const [name, insts] of Object.entries(node.instanceByFeature)) {
                const color = this._getFeatureColor(name);
                if (!color) continue;
                for (const { batch, id } of insts) {
                    batch.setColorAt(id, _batchColor.set(color));
                    batch.baseColorByInstance.set(id, color);
                }
            }
            // legacy meshes
            for (const [name, meshes] of Object.entries(node.meshByFeature)) {
                const color = this._getFeatureColor(name);
                if (color) for (const m of meshes) m.material?.color.set(color);
            }
        }
        this._dirty = true;
    }

    /**
     * Hover highlight for a feature: brighten its batched instance colors (on) or
     * restore them from baseColorByInstance (off). Legacy line/point meshes fall back
     * to material.emissive. Called by the viewer's hover handler.
     */
    setFeatureHighlight(name, on) {
        for (const node of this._allNodes) {
            const insts = node.instanceByFeature[name];
            if (insts) {
                for (const { batch, id } of insts) {
                    const base = batch.baseColorByInstance.get(id) || '#cccccc';
                    if (on) batch.setColorAt(id, _hl.set(base).lerp(_white, 0.5));
                    else    batch.setColorAt(id, _batchColor.set(base));
                }
            }
            const meshes = node.meshByFeature[name];
            if (meshes) for (const m of meshes) m.material?.emissive?.set(on ? 0x333333 : 0x000000);
        }
        this._dirty = true;
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

    /**
     * Rebuild the pre-culled list of selected features that can actually render
     * under the current spatial filter — i.e. those with at least one tile inside
     * the selection box at their display zoom. Excluded features produce no visible
     * geometry (the per-frame loop's frustum/box filter would empty them anyway), so
     * skipping them per frame is the difference between O(features-selected) and
     * O(features-in-box) work — the fix for ~1 FPS at 139k selected.
     *
     * Only meaningful with a spatial filter in forced-zoom mode (the focused
     * workflow). Without a filter, any selected feature may be visible anywhere, so
     * we fall back to iterating the full selection (null = "use selectedFeatures").
     *
     * Soundness: a feature's display tile is at `dz = bestAvailableZoom(forcedZoom)`,
     * and the per-zoom box shrinks as zoom increases, so a finer box ⊆ the dz box —
     * testing dz alone correctly admits the finer-fallback path too. The one path it
     * does NOT mirror is the over-budget `_showAnyLoadedZoom` coarse fallback (coarser
     * box is larger): under GPU pressure a boundary feature that would have shown a
     * coarse tile bleeding outside the box is now skipped. That aligns with the
     * filter's stated intent (don't show content outside the selection) and only
     * occurs when already over budget, so it is an acceptable, beneficial divergence.
     */
    _recomputeActiveFeatures() {
        this._activeForcedZoom = this.forcedZoom;
        this._activeFilterToken = this._filterToken;
        this._activeDirty = false;

        if (!this._spatialFilter || this.lodMode !== 'forced') {
            this._activeFeatures = null;
            return;
        }

        const active = [];
        for (const name of this.selectedFeatures) {
            const feat = this.featureIndex[name];
            if (!feat) continue;
            const dz = this._bestAvailableZoom(feat, this.forcedZoom);
            if (dz === null) continue;
            const uris = feat.tiles[String(dz)];
            if (!uris) continue;
            for (let i = 0; i < uris.length; i++) {
                if (this._isTileInSelection(uris[i])) { active.push(name); break; }
            }
        }
        this._activeFeatures = active;
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
            for (const batch of node.batched) {
                batch.visible = false;
                const n = batch.instanceCount;       // BatchedMesh.instanceCount = # instances
                for (let i = 0; i < n; i++) batch.setVisibleAt(i, false);
            }
            if (!node.object3D) continue;            // legacy line/point path
            node.object3D.visible = false;
            for (const meshes of Object.values(node.meshByFeature))
                for (const mesh of meshes) mesh.visible = false;
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

        // Phase 1: per-feature zoom decision + tile display. Iterate only the
        // features that can actually render under the current spatial filter — the
        // pre-culled list is recomputed lazily here whenever the selection, the
        // filter box, or the forced zoom changed (never per frame), so this loop is
        // O(features-in-box) instead of O(features-selected).
        if (this._activeDirty
            || this.forcedZoom !== this._activeForcedZoom
            || this._filterToken !== this._activeFilterToken) {
            this._recomputeActiveFeatures();
        }
        const featureList = this._activeFeatures || this.selectedFeatures;
        for (const name of featureList) {
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
        if (!node || node.loadState !== LOADED) return;
        node.lastUsedFrame = this._frameNumber;
        const insts = node.instanceByFeature[featureName];
        if (insts) {
            for (const { batch, id } of insts) { batch.visible = true; batch.setVisibleAt(id, true); this.visibleCount++; }
        }
        // legacy line/point meshes for this feature
        const meshes = node.meshByFeature[featureName];
        if (meshes && node.object3D) {
            node.object3D.visible = true;
            for (const mesh of meshes) { mesh.visible = true; this.visibleCount++; }
        }
    }

    /**
     * Fallback: find any loaded zoom for a feature and show it.
     */
    _showAnyLoadedZoom(feat, name, state) {
        // Prefer higher zoom levels (more detail): cached keys are ascending, walk down.
        const zk = this._zoomKeysOf(feat);
        for (let i = zk.length - 1; i >= 0; i--) {
            const z = zk[i];
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

    /** Numeric tile-zoom levels available for a feature, ascending, cached on the
     * feature (feat.tiles is built once and never mutated). Avoids the per-call
     * Object.keys(...).map(Number) allocation in the per-frame hot loop. */
    _zoomKeysOf(feat) {
        let zk = feat._zoomKeys;
        if (!zk) {
            zk = Object.keys(feat.tiles).map(Number).sort((a, b) => a - b);
            feat._zoomKeys = zk;
        }
        return zk;
    }

    /**
     * Find the closest available zoom level for a feature.
     */
    _bestAvailableZoom(feat, targetZoom) {
        const available = this._zoomKeysOf(feat);
        if (available.length === 0) return null;

        let best = available[0];
        let bestDiff = Math.abs(best - targetZoom);
        for (let i = 1; i < available.length; i++) {
            const z = available[i];
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
        const zk = this._zoomKeysOf(feat);   // ascending → first is coarsest
        return zk.length ? zk[0] : null;
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

            this._buildBatchedTile(node, group);
            // Legacy line/point meshes (if any) still live in the group; add it only if it
            // has renderable legacy children. BatchedMeshes were added to the scene directly.
            const hasLegacy = Object.keys(node.meshByFeature).length > 0;
            if (hasLegacy) { group.visible = false; this.scene.add(group); node.object3D = group; }
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

    /** Resolve a feature's logical name from its GLB node userData (extras). */
    _resolveFeatureName(props) {
        if (!props) return '';
        const rawName = props.name;
        return (rawName && !/^feature_\d+$/.test(rawName) ? rawName : null)
            || props.acronym || props.instance
            || (props.body_id != null ? String(props.body_id) : '');
    }

    /** One shared material for a batch, cloned from a representative loaded mesh
     * material so lighting/appearance matches the current per-mesh look. Per-feature
     * color is applied per-instance via setColorAt, not on the material. */
    _makeBatchMaterial(sampleMaterial) {
        const m = sampleMaterial ? sampleMaterial.clone() : new THREE.MeshStandardMaterial();
        m.color.set('#ffffff');           // white base; instance color multiplies in
        m.vertexColors = false;           // BatchedMesh injects per-instance color itself
        if (this._opacity < 1) { m.transparent = true; m.opacity = this._opacity; m.depthWrite = false; }
        return m;
    }

    /**
     * Build the renderable objects for a freshly loaded tile group: triangle
     * features become a single THREE.BatchedMesh (one draw call per tile);
     * line/point features fall back to the legacy per-mesh meshByFeature path.
     */
    _buildBatchedTile(node, group) {
        const triFeats = [];          // {name, geometry, color, props}
        let sampleMat = null;
        group.traverse(child => {
            const isTri = child.isMesh && !child.isLine && !child.isPoints;
            const props = this._findProps(child);
            const name = this._resolveFeatureName(props);
            if (!name) return;
            if (!isTri) {             // line/point feature -> legacy per-mesh path
                child.matrixAutoUpdate = false; child.updateMatrix(); child.visible = false;
                (node.meshByFeature[name] ??= []).push(child);
                child.userData._featureName = name;
                return;
            }
            if (!sampleMat) sampleMat = child.material;
            triFeats.push({ name, geometry: child.geometry, props,
                            color: this._getFeatureColor(name) || props.color });
        });

        if (triFeats.length === 0) return;   // pure line/point tile: legacy path already populated

        // Size the batch from the decoded geometries (loader has already decoded them).
        let vtot = 0, itot = 0;
        for (const f of triFeats) {
            vtot += f.geometry.attributes.position.count;
            itot += f.geometry.index ? f.geometry.index.count : 0;
        }

        // Oversized-tile guard: a BatchedMesh pre-allocates one contiguous vertex+index
        // buffer, so a coarse mega-tile (e.g. the z0 tile holding ~all features) would
        // throw "Array buffer allocation failed". Refuse to batch it — it is never the
        // intended resident working set (the object/byte budgets keep finer in-frustum
        // tiles instead, and the overview poster is the see-everything view). The tile
        // loads but renders nothing rather than crashing.
        const estBytes = vtot * 12 + itot * 4;   // position(3×f32) + index(u32)
        if (triFeats.length > MAX_INSTANCES_PER_BATCH || estBytes > MAX_BATCH_BYTES) {
            console.warn(
                `[TileManager] not batching oversized tile ${node.uri}: ${triFeats.length} ` +
                `features, ~${(estBytes / 1048576).toFixed(0)} MB buffer (over cap). Tile will ` +
                `not render — rely on finer in-frustum tiles / the overview.`
            );
            return;
        }

        const batch = new THREE.BatchedMesh(triFeats.length, vtot, itot, this._makeBatchMaterial(sampleMat));
        // Tile-level frustum cull already runs (_frustumFilter); per-instance cull + sort are
        // O(n) CPU per frame and become the bottleneck at this scale (three.js #28776).
        batch.perObjectFrustumCulled = false;
        batch.sortObjects = false;
        batch.frustumCulled = false;          // we manage tile visibility ourselves
        batch.featureByInstance = new Map();  // id -> name
        batch.propsByInstance = new Map();    // id -> props (for InfoPanel)
        batch.baseColorByInstance = new Map();// id -> '#rrggbb' (for hover restore)
        batch.userData._tileNode = node;

        for (const f of triFeats) {
            const gid = batch.addGeometry(f.geometry);
            const iid = batch.addInstance(gid);   // identity matrix (geometry is absolute-coords)
            const hex = f.color || '#cccccc';
            batch.setColorAt(iid, _batchColor.set(hex));
            batch.setVisibleAt(iid, false);       // hidden until selected
            (node.instanceByFeature[f.name] ??= []).push({ batch, id: iid });
            batch.featureByInstance.set(iid, f.name);
            batch.propsByInstance.set(iid, f.props);
            batch.baseColorByInstance.set(iid, hex);
            node.gpuBytes += f.geometry.attributes.position.array.byteLength
                + (f.geometry.index ? f.geometry.index.array.byteLength : 0);
        }
        node.instanceCount += triFeats.length;
        batch.visible = false;
        this.scene.add(batch);
        node.batched.push(batch);
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
        for (const batch of node.batched) {
            batch.dispose();             // frees the batch's shared buffers
            this.scene.remove(batch);
        }
        node.batched = [];
        node.instanceByFeature = {};
        node.instanceCount = 0;
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
