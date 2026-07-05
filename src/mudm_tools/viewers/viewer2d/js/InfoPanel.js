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
        const _mk = Object.entries(properties).filter(([k]) => k.startsWith("m_"));
        Object.entries(properties).forEach(([key, value]) => {
            if (value !== null && value !== undefined && !key.startsWith("m_")) {
                this._addRow(container, key, String(value));
            }
        });
        if (_mk.length) {
            const _fmt = (v) => (typeof v === "number") ? (Math.abs(v) >= 1 ? v.toFixed(2) : Number(v.toPrecision(2))) : v;
            const _sorted = _mk.slice().sort((a, b) => (Number(b[1]) || 0) - (Number(a[1]) || 0));
            const _sum = document.createElement("div");
            _sum.className = "info-row"; _sum.style.cursor = "pointer";
            const _list = document.createElement("div");
            _list.style.cssText = "display:none;margin:3px 0 0 8px;font-size:0.78rem;line-height:1.4;max-height:220px;overflow:auto;";
            _list.innerHTML = _sorted.map(([k, v]) => '<span class="info-label">' + k.slice(2) + '</span> ' + _fmt(v)).join(" &middot; ");
            let _open = false;
            const _draw = () => { _sum.innerHTML = '<span class="info-label">markers:</span> ' + _mk.length + (_open ? " \u25be" : " \u25b8 (click to expand)"); _list.style.display = _open ? "block" : "none"; };
            _draw();
            _sum.addEventListener("click", (e) => { e.stopPropagation(); _open = !_open; _draw(); });
            container.appendChild(_sum); container.appendChild(_list);
        }

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
