// viewer2d/js/main.js
import { LayerPanel } from "./LayerPanel.js";
import { InfoPanel } from "./InfoPanel.js";

let map;
let metadata;
let vectorGridLayer;
let rasterLayer;
let hiddenLayers = new Set();
let hoveredFeatureId = null;
let geneColorMap = null;  // { gene_name → hex color }

const BASE_URL = "";

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
    if (datasets.length > 0) {
        await loadDataset(datasets[0].id);
    }
    select.addEventListener("change", () => loadDataset(select.value));
}

async function loadDataset(datasetId) {
    const resp = await fetch(`${BASE_URL}/tiles2d/${datasetId}/metadata.json`);
    metadata = await resp.json();

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
    const [imgW, imgH] = metadata.raster.image_size_px;
    const rasterMaxZoom = metadata.raster.max_zoom;

    const southWest = map.unproject([0, imgH], rasterMaxZoom);
    const northEast = map.unproject([imgW, 0], rasterMaxZoom);
    const imageBounds = L.latLngBounds(southWest, northEast);

    const [bxMin, byMin, bxMax, byMax] = metadata.bounds_um;
    const dataSW = map.unproject([bxMin / umPerPx, byMax / umPerPx], rasterMaxZoom);
    const dataNE = map.unproject([bxMax / umPerPx, byMin / umPerPx], rasterMaxZoom);
    const dataBounds = L.latLngBounds(dataSW, dataNE);

    const vectorMaxZoom = Math.max(...metadata.vectors.layers.map(l => l.max_zoom));
    rasterLayer = L.tileLayer(
        `${BASE_URL}/tiles2d/${datasetId}/raster/{z}/{x}/{y}.png`,
        {
            minZoom: 0,
            maxNativeZoom: rasterMaxZoom,
            maxZoom: vectorMaxZoom,
            tileSize: metadata.raster.tile_size || 256,
            noWrap: true,
            bounds: imageBounds,
        }
    );
    rasterLayer.addTo(map);

    map.fitBounds(dataBounds);
    map.setMaxBounds(dataBounds.pad(0.2));

    setupVectorLayer(datasetId, imageBounds, vectorMaxZoom);
    LayerPanel.init(metadata, rasterLayer, vectorGridLayer, hiddenLayers, datasetId);
}

function setupVectorLayer(datasetId, imageBounds, maxZoom) {
    const layerLookup = {};
    for (const layerDef of metadata.vectors.layers) {
        layerLookup[layerDef.id] = layerDef;
    }

    const styles = {};
    const geneSelect = document.getElementById("gene-filter");
    for (const layerDef of metadata.vectors.layers) {
        styles[layerDef.id] = function (properties) {
            if (hiddenLayers.has(layerDef.id)) {
                return { opacity: 0, fillOpacity: 0, radius: 0, weight: 0 };
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
            return getLayerStyle(layerDef);
        };
    }

    vectorGridLayer = L.vectorGrid.protobuf(
        `${BASE_URL}/tiles2d/${datasetId}/${metadata.vectors.path}`,
        {
            vectorTileLayerStyles: styles,
            interactive: true,
            maxZoom: maxZoom,
            minZoom: 0,
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
        InfoPanel.show(props, def);

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
        InfoPanel.pin(props, def);
    });

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

    map.on("zoomend moveend", () => {
        if (!metadata) return;
        const umPerPx = metadata.um_per_px;
        const rasterMaxZoom = metadata.raster.max_zoom;
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
