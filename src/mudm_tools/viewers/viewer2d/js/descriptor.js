// viewer2d/js/descriptor.js
//
// The 2D viewer's SINGLE descriptor loader. Every viewer consumes the muDM `TileModel` (served as
// tilejson.json) — NOT the bespoke metadata.json. This module fetches tilejson.json, and (for the migration
// window) falls back to metadata.json if the muDM descriptor isn't present yet, so the live site never breaks.
//
// It exposes two pure adapters (testable in Node, no fetch):
//   tileModelToMetadata(tj)   muDM TileModel  → the metadata.json-shaped object the viewer engine reads
//   metadataToViewModel(meta) metadata shape  → a stable, camelCase ViewModel the UI code reads
// and one async entry point:
//   loadDescriptor(baseUrl, id) -> ViewModel   (with `.meta` = the metadata-shaped view, for the engine)
//
// Building the ViewModel FROM the reconstructed metadata shape means the ViewModel is identical whether the
// source was tilejson.json or a raw metadata.json — the rest of the viewer is descriptor-shape-agnostic.

// --- muDM TileModel → metadata.json shape --------------------------------------------------------------
// Maps the muDM first-class + sanctioned foreign members back onto the flat metadata shape the viewer's
// (raster / vector / 2.5D / facet) machinery already consumes. Contract:
// .claude/reference/mudm-viewer-descriptor.md (in mudm-data).
export function tileModelToMetadata(tj) {
    const assets = tj.assets || [];
    const ms = tj.multiscale || {};
    const axes = ms.axes || [];
    const scaleT = (ms.coordinateTransformations || []).find((t) => t.type === "scale");
    const scale = scaleT ? scaleT.scale : null;
    const umPerPx = scale ? scale[0] : (tj["mudm:um_per_px"] != null ? tj["mudm:um_per_px"] : 1);
    const hasZ = axes.some((a) => a.name === "z");

    const rasters = assets.filter((a) => a.role === "raster").map((a) => ({
        id: a.title || a.channel, name: a.title || a.channel, channel: a.channel,
        path: a.href, image_size_px: a.image_size_px, max_zoom: a.max_zoom,
        tile_size: a.tile_size || 256, is_default: !!a.is_default,
    }));
    const vec = assets.find((a) => a.role === "vector");
    const layers = (tj.vector_layers || []).map((l) => ({
        id: l.id, name: l["mudm:label"] || l.id, type: l["mudm:type"] || l.description || "polygon",
        color: l["mudm:color"] || null, min_zoom: l.minzoom || 0,
        max_zoom: (l.maxzoom != null ? l.maxzoom : undefined),
        feature_count: l["mudm:feature_count"],
    }));

    const meta = {
        name: tj.name,
        platform: tj["mudm:platform"] || null,
        um_per_px: umPerPx,
        bounds_um: tj.bounds || null,
        vectors: {
            path: vec ? vec.href : "vectors/{z}/{x}/{y}.pbf",
            layers: layers,
            feature_label: tj["mudm:feature_label"] || undefined,
            feature_label_plural: tj["mudm:feature_label_plural"] || undefined,
        },
    };
    const dflt = rasters.find((r) => r.is_default) || rasters[0];
    // singular `raster` for the common single-channel, non-2.5D case (the viewer's simple path); `rasters`
    // for multichannel / 2.5D (drives the channel selector). Carry the channel as the toggle `label`.
    if (rasters.length === 1 && !hasZ) {
        // NB: do NOT set `label` here — a channel-less singular raster's channel is the generic "morphology"
        // placeholder; leaving label unset lets LayerPanel._rasterLabel pick the accurate per-platform default
        // (xenium→DAPI, visium→H&E, …), matching what the raw metadata.json path produces.
        meta.raster = { path: dflt.path, image_size_px: dflt.image_size_px, max_zoom: dflt.max_zoom,
                        tile_size: dflt.tile_size, min_zoom: 0 };
    }
    if (rasters.length) meta.rasters = rasters;
    if (hasZ && ms["mudm:slice_count"] != null) {
        meta.depth = { slice_count: ms["mudm:slice_count"], z_index_format: ms["mudm:z_index_format"],
                       z_spacing_um: (scale && scale.length > 2) ? scale[2] : undefined };
    }
    // facet store: rebuild the block from the facet asset(s) (which carry layer/storage/prefetch/fields +
    // per-asset facet/layout/key/columns) + the facet layer's fieldenums (the panel — kept off the asset).
    const facetAssets = assets.filter((a) => a.role === "facets");
    if (facetAssets.length) {
        const a0 = facetAssets[0];
        const facetLayer = a0.layer || (layers[0] && layers[0].id);
        const tl = (tj.vector_layers || []).find((l) => l.id === facetLayer) || {};
        meta.facets = {
            layer: a0.layer || facetLayer, key: a0.key, storage: a0.storage || "asset",
            layout: a0.layout, prefetch: !!a0.prefetch,
            fields: a0.fields || tl.fields || {}, fieldenums: tl.fieldenums || undefined,
            assets: facetAssets.map((a) => {
                const o = { role: "facets", href: a.href, media_type: a.media_type };
                ["facet", "layout", "key", "columns", "sorted_by", "rows"].forEach((k) => {
                    if (a[k] !== undefined) o[k] = a[k];
                });
                return o;
            }),
        };
    }
    const gl = assets.find((a) => a.role === "gene_list");
    const cm = assets.find((a) => a.role === "colormap");
    if (gl) meta._gene_list_href = gl.href;
    if (cm) meta._colormap_href = cm.href;
    return meta;
}

// --- metadata.json shape → ViewModel -------------------------------------------------------------------
export function metadataToViewModel(meta) {
    const rastersRaw = meta.rasters
        || (meta.raster ? [Object.assign({}, meta.raster,
                { channel: meta.raster.label || meta.raster.channel || "morphology", is_default: true })]
            : []);
    const rasters = rastersRaw.map((r) => ({
        channel: r.channel || r.name || r.id || "morphology",
        pathTemplate: r.path,
        imageSizePx: r.image_size_px || null,
        maxZoom: r.max_zoom != null ? r.max_zoom : null,
        tileSize: r.tile_size || 256,
        isDefault: !!r.is_default,
    }));
    // exactly one default channel (fallback: the first) so the viewer always has a channel to show
    if (rasters.length && !rasters.some((r) => r.isDefault)) rasters[0].isDefault = true;

    const vlayers = (meta.vectors && meta.vectors.layers) || [];
    const layers = vlayers.map((l) => ({
        id: l.id, type: l.type, color: l.color || null, label: l.name || l.id,
        minZoom: l.min_zoom || 0, maxZoom: l.max_zoom != null ? l.max_zoom : null,
        featureCount: l.feature_count,
    }));
    const rmax = rasters.reduce((m, r) => Math.max(m, r.maxZoom || 0), 0);
    const lmax = layers.reduce((m, l) => Math.max(m, l.maxZoom || 0), 0);

    const f = meta.facets;
    const featureLabel = (meta.vectors && meta.vectors.feature_label) || "Gene";
    return {
        name: meta.name,
        platform: meta.platform || null,
        umPerPx: meta.um_per_px,
        boundsUm: meta.bounds_um || null,
        zoomMin: 0,
        zoomMax: Math.max(rmax, lmax),
        rasters,
        vectorPath: (meta.vectors && meta.vectors.path) || "vectors/{z}/{x}/{y}.pbf",
        layers,
        featureLabel,
        featureLabelPlural: (meta.vectors && meta.vectors.feature_label_plural)
            || (featureLabel.toLowerCase() + "s"),
        depth: meta.depth ? {
            sliceCount: meta.depth.slice_count,
            zSpacingUm: meta.depth.z_spacing_um,
            zIndexFormat: meta.depth.z_index_format || "z{:03d}",
        } : null,
        facets: f ? {
            layer: f.layer, key: f.key, layout: f.layout, storage: f.storage || "asset",
            prefetch: !!f.prefetch, fields: f.fields || {}, fieldenums: f.fieldenums || null,
            assetHref: (f.assets && f.assets[0] && f.assets[0].href) || null,
        } : null,
        geneListHref: meta._gene_list_href || "gene_list.json",
        colormapHref: meta._colormap_href || "gene_colormap.json",
    };
}

// --- async loader: tilejson.json (the muDM descriptor), else metadata.json (transition fallback) -------
export async function loadDescriptor(baseUrl, id) {
    let meta = null;
    try {
        const r = await fetch(`${baseUrl}/tiles2d/${id}/tilejson.json`);
        if (r.ok) {
            meta = tileModelToMetadata(await r.json());
            meta._source = "tilejson";
        }
    } catch (_) { /* fall through to metadata.json */ }
    if (!meta) {
        const r = await fetch(`${baseUrl}/tiles2d/${id}/metadata.json`);
        meta = await r.json();
        meta._source = "metadata";
    }
    const vm = metadataToViewModel(meta);
    vm.meta = meta;   // the metadata-shaped view the 2.5D / facet engine reads (same descriptor, one fetch)
    return vm;
}
