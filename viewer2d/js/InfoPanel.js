// viewer2d/js/InfoPanel.js

let pinned = false;

export const InfoPanel = {
    show(properties, layerDef) {
        if (pinned) return;
        this._render(properties, layerDef);
    },

    pin(properties, layerDef) {
        pinned = true;
        this._render(properties, layerDef);
    },

    clear() {
        if (pinned) return;
        document.getElementById("info-content").innerHTML =
            '<span class="info-placeholder">Hover over a feature</span>';
    },

    unpin() {
        pinned = false;
        this.clear();
    },

    _render(properties, layerDef) {
        const container = document.getElementById("info-content");
        container.innerHTML = "";

        // Layer type
        this._addRow(container, "Layer", layerDef.name);

        // All properties
        Object.entries(properties).forEach(([key, value]) => {
            if (value !== null && value !== undefined) {
                this._addRow(container, key, String(value));
            }
        });

        // Unpin on click outside
        if (pinned) {
            const unpin = document.createElement("div");
            unpin.style.cssText = "font-size:0.7rem; color:#718096; margin-top:6px; cursor:pointer;";
            unpin.textContent = "Click to unpin";
            unpin.addEventListener("click", () => this.unpin());
            container.appendChild(unpin);
        }
    },

    _addRow(container, label, value) {
        const row = document.createElement("div");
        row.className = "info-row";
        row.innerHTML = `<span class="info-label">${label}:</span> ${value}`;
        container.appendChild(row);
    },
};
