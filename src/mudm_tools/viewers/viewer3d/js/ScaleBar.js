/**
 * Adaptive scale bar overlay for the 3D viewer.
 *
 * Renders a horizontal bar with a label showing the real-world distance it
 * represents. Updates each frame based on camera distance and FOV. Works in
 * real meters via `metersPerUnit` (set per dataset from tilejson3d's
 * `meters_per_unit`); defaults to 1e-9 (nm) so EM/connectome datasets that
 * predate the field keep their nanometer scale. Picks an appropriate unit
 * automatically (nm, µm, mm, cm, m).
 */

// "Nice" lengths in METERS, 1 nm … 50 m (decade × {1,2,5}).
const NICE_M = [];
for (let e = -9; e <= 1; e++) for (const m of [1, 2, 5]) NICE_M.push(m * Math.pow(10, e));

export class ScaleBar {
    /**
     * @param {HTMLCanvasElement} canvas - The main WebGL canvas (for sizing)
     */
    constructor(canvas) {
        this.canvas = canvas;
        this.targetPx = 150; // desired bar width in pixels
        this.metersPerUnit = 1e-9; // world-unit → meters; nm default (EM datasets)

        // Create DOM elements
        this.container = document.createElement('div');
        this.container.id = 'scale-bar';
        this.container.innerHTML = `
            <div class="scale-bar-line"></div>
            <div class="scale-bar-label"></div>
        `;
        document.body.appendChild(this.container);

        this.barEl = this.container.querySelector('.scale-bar-line');
        this.labelEl = this.container.querySelector('.scale-bar-label');
    }

    /** Set real-world meters per world unit (from tilejson3d `meters_per_unit`). */
    setMetersPerUnit(m) {
        this.metersPerUnit = (typeof m === 'number' && m > 0) ? m : 1e-9;
    }

    /**
     * Update scale bar for current camera state.
     * @param {THREE.PerspectiveCamera} camera
     * @param {THREE.OrbitControls} controls
     */
    update(camera, controls) {
        const dist = camera.position.distanceTo(controls.target);
        const vFov = camera.fov * Math.PI / 180;
        const heightWorld = 2 * dist * Math.tan(vFov / 2);
        const heightPx = this.canvas.clientHeight;
        if (heightPx === 0) return;

        // Convert to real meters per pixel.
        const metersPerPx = (heightWorld / heightPx) * this.metersPerUnit;
        const targetMeters = metersPerPx * this.targetPx;

        // Largest "nice" length (meters) that fits the target pixel width.
        let best = NICE_M[0];
        for (const n of NICE_M) {
            if (n <= targetMeters) best = n;
            else break;
        }

        this.barEl.style.width = (best / metersPerPx) + 'px';
        this.labelEl.textContent = this._format(best);
    }

    /** Format a length in METERS to a human-readable string (nm/µm/mm/cm/m). */
    _format(m) {
        if (m >= 1) return this._n(m) + ' m';
        if (m >= 1e-2) return this._n(m * 1e2) + ' cm';
        if (m >= 1e-3) return this._n(m * 1e3) + ' mm';
        if (m >= 1e-6) return this._n(m * 1e6) + ' µm';
        return this._n(m * 1e9) + ' nm';
    }

    /** Round float-scale artifacts to a clean integer or 1 decimal. */
    _n(v) {
        const r = Math.round(v * 10) / 10;
        return (r % 1 === 0) ? r.toFixed(0) : r.toFixed(1);
    }
}
