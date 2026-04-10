/**
 * Pyramid selector — dropdown to switch between tile pyramids.
 *
 * Fetches pyramids.json manifest and renders a <select> dropdown.
 * Fires a callback when the user selects a different pyramid.
 */

export class PyramidSelector {
    /**
     * @param {HTMLElement} container - DOM element to render into
     * @param {function(object): void} onPyramidChange - callback with pyramid entry
     */
    constructor(container, onPyramidChange) {
        this.container = container;
        this.onPyramidChange = onPyramidChange;
        this.pyramids = [];
        this.currentId = null;
    }

    async init(manifestUrl) {
        const resp = await fetch(manifestUrl);
        const data = await resp.json();
        this.pyramids = data.pyramids || [];

        if (this.pyramids.length === 0) {
            this.container.style.display = 'none';
            return null;
        }

        this._render();

        // Return the first pyramid as default
        this.currentId = this.pyramids[0].id;
        return this.pyramids[0];
    }

    _render() {
        this.container.innerHTML = '';

        const label = document.createElement('h3');
        label.textContent = 'Pyramid';
        label.style.cssText = 'font-size:11px;color:#999;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;';
        this.container.appendChild(label);

        const select = document.createElement('select');
        select.style.cssText = `
            width: 100%;
            padding: 7px 10px;
            border-radius: 5px;
            border: 1px solid rgba(255,255,255,0.15);
            background: rgba(255,255,255,0.06);
            color: #eee;
            font-size: 13px;
            outline: none;
            cursor: pointer;
        `;

        for (const p of this.pyramids) {
            const opt = document.createElement('option');
            opt.value = p.id;
            opt.textContent = `${p.label} (${p.feature_count ?? p.features ?? p.featureCount ?? '?'} regions)`;
            select.appendChild(opt);
        }

        select.addEventListener('change', () => {
            const pyramid = this.pyramids.find(p => p.id === select.value);
            if (pyramid && pyramid.id !== this.currentId) {
                this.currentId = pyramid.id;
                this.onPyramidChange(pyramid);
            }
        });

        this.container.appendChild(select);
    }
}
