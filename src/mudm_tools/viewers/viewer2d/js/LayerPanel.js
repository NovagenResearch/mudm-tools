// viewer2d/js/LayerPanel.js

const LAYER_LABELS = {
    cells: "Cell Boundaries",
    nuclei: "Nucleus Boundaries",
    transcripts: "Transcripts",
};

export const LayerPanel = {
    init(metadata, rasterLayer, vectorGridLayer, hiddenLayers, datasetId) {
        const container = document.getElementById("layer-toggles");
        container.innerHTML = "";

        // DAPI raster toggle
        this._addToggle(container, "DAPI Image", "#888888", true, (checked) => {
            if (checked) rasterLayer.setOpacity(1);
            else rasterLayer.setOpacity(0);
        });

        // Vector layer toggles
        metadata.vectors.layers.forEach((layerDef) => {
            this._addToggle(
                container,
                LAYER_LABELS[layerDef.id] || layerDef.id,
                layerDef.color,
                true,
                (checked) => {
                    if (checked) {
                        hiddenLayers.delete(layerDef.id);
                    } else {
                        hiddenLayers.add(layerDef.id);
                    }
                    vectorGridLayer.restyleAll();
                },
                layerDef.type === "point" ? "circle" : "square"
            );
        });

        // Gene filter — load from sidecar or collect from hover events
        this._setupGeneFilter(vectorGridLayer, datasetId);

        // Color legend from colormap
        this._loadColorLegend(datasetId, vectorGridLayer);
    },

    _addToggle(container, label, color, checked, onChange, shape = "square") {
        const wrapper = document.createElement("label");
        wrapper.className = "layer-toggle";
        wrapper.style.setProperty("--layer-color", color);

        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = checked;
        checkbox.addEventListener("change", () => onChange(checkbox.checked));

        const swatch = document.createElement("span");
        swatch.className = "layer-swatch";
        swatch.style.background = color;
        if (shape === "circle") swatch.style.borderRadius = "50%";

        const text = document.createTextNode(label);

        wrapper.appendChild(checkbox);
        wrapper.appendChild(swatch);
        wrapper.appendChild(text);
        container.appendChild(wrapper);
    },

    async _setupGeneFilter(vectorGridLayer, datasetId) {
        const select = document.getElementById("gene-filter");
        if (!vectorGridLayer) {
            select.style.display = "none";
            return;
        }

        // Try to load gene list from sidecar file
        try {
            const resp = await fetch(`/tiles2d/${datasetId}/gene_list.json`);
            if (resp.ok) {
                const genes = await resp.json();
                select.innerHTML = '<option value="">All genes</option>';
                genes.forEach((g) => {
                    const opt = document.createElement("option");
                    opt.value = g;
                    opt.textContent = g;
                    select.appendChild(opt);
                });
            }
        } catch (_) {
            // Fallback: collect from hover events
            const genes = new Set();
            vectorGridLayer.on("mouseover", (e) => {
                const name = e.layer.properties && e.layer.properties.gene_name;
                if (name && !genes.has(name)) {
                    genes.add(name);
                    const current = select.value;
                    select.innerHTML = '<option value="">All genes</option>';
                    [...genes].sort().forEach((g) => {
                        const opt = document.createElement("option");
                        opt.value = g;
                        opt.textContent = g;
                        select.appendChild(opt);
                    });
                    select.value = current;
                }
            });
        }

        select.addEventListener("change", () => {
            vectorGridLayer.restyleAll();
        });
    },

    async _loadColorLegend(datasetId, vectorGridLayer) {
        const container = document.getElementById("color-legend");
        container.innerHTML = "";
        try {
            const resp = await fetch(`/tiles2d/${datasetId}/gene_colormap.json`);
            if (!resp.ok) return;
            const cm = await resp.json();

            const hiddenCategories = new Set();
            const checkboxes = [];

            // Debounced redraw — batches rapid checkbox changes
            let redrawTimer = null;
            const scheduleRedraw = () => {
                if (redrawTimer) clearTimeout(redrawTimer);
                redrawTimer = setTimeout(() => vectorGridLayer.restyleAll(), 150);
            };

            // Toggle all button
            const toggleRow = document.createElement("div");
            toggleRow.style.cssText = "display:flex;gap:6px;margin-bottom:6px;";
            const toggleBtn = document.createElement("button");
            toggleBtn.textContent = "Toggle all";
            toggleBtn.style.cssText = "background:#2d3748;color:#e2e8f0;border:1px solid #4a5568;border-radius:3px;padding:2px 8px;font-size:0.7rem;cursor:pointer;";
            let allChecked = true;
            toggleBtn.addEventListener("click", () => {
                allChecked = !allChecked;
                checkboxes.forEach(({ cb, catName }) => {
                    cb.checked = allChecked;
                    if (allChecked) {
                        hiddenCategories.delete(catName);
                    } else {
                        hiddenCategories.add(catName);
                    }
                });
                window._hiddenGeneCategories = hiddenCategories;
                scheduleRedraw();
            });
            toggleRow.appendChild(toggleBtn);
            container.appendChild(toggleRow);

            const entries = [...Object.entries(cm.categories),
                             ["Other", { color: cm.default_color || "#888888", genes: [] }]];

            for (const [catName, catDef] of entries) {
                const item = document.createElement("label");
                item.style.cssText = "display:flex;align-items:center;gap:6px;margin-bottom:4px;cursor:pointer;font-size:0.8rem;";

                const cb = document.createElement("input");
                cb.type = "checkbox";
                cb.checked = true;
                cb.style.accentColor = catDef.color;

                const swatch = document.createElement("span");
                swatch.style.cssText = `display:inline-block;width:10px;height:10px;border-radius:50%;background-color:${catDef.color};flex-shrink:0;`;

                const label = document.createTextNode(catName);

                cb.addEventListener("change", () => {
                    if (cb.checked) {
                        hiddenCategories.delete(catName);
                    } else {
                        hiddenCategories.add(catName);
                    }
                    window._hiddenGeneCategories = hiddenCategories;
                    scheduleRedraw();
                });

                checkboxes.push({ cb, catName });
                item.appendChild(cb);
                item.appendChild(swatch);
                item.appendChild(label);
                container.appendChild(item);
            }

            // Build gene → category lookup
            const geneToCat = {};
            for (const [catName, catDef] of Object.entries(cm.categories)) {
                for (const gene of catDef.genes) {
                    geneToCat[gene] = catName;
                }
            }
            window._geneCategoryMap = geneToCat;
            window._hiddenGeneCategories = hiddenCategories;

        } catch (_) {}
    },
};
