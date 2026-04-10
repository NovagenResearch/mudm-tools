/**
 * Adaptive scale bar overlay for the 3D viewer.
 *
 * Renders a horizontal bar with a label showing the real-world distance
 * it represents. Updates each frame based on camera distance and FOV.
 * Automatically selects appropriate units (nm, um, mm).
 */

// "Nice" numbers for scale bar labels
const NICE = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
              10000, 20000, 50000, 100000, 200000, 500000, 1000000];

export class ScaleBar {
    /**
     * @param {HTMLCanvasElement} canvas - The main WebGL canvas (for sizing)
     */
    constructor(canvas) {
        this.canvas = canvas;
        this.targetPx = 150; // desired bar width in pixels

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

    /**
     * Update scale bar for current camera state.
     * @param {THREE.PerspectiveCamera} camera
     * @param {THREE.OrbitControls} controls
     */
    update(camera, controls) {
        // World units per pixel at the focus distance
        const dist = camera.position.distanceTo(controls.target);
        const vFov = camera.fov * Math.PI / 180;
        const heightWorld = 2 * dist * Math.tan(vFov / 2);
        const heightPx = this.canvas.clientHeight;
        if (heightPx === 0) return;

        const worldPerPx = heightWorld / heightPx;
        const targetWorld = worldPerPx * this.targetPx;

        // Find the largest "nice" number <= targetWorld
        let best = NICE[0];
        for (const n of NICE) {
            if (n <= targetWorld) best = n;
            else break;
        }

        const barPx = best / worldPerPx;
        this.barEl.style.width = barPx + 'px';
        this.labelEl.textContent = this._format(best);
    }

    /**
     * Format a distance in world units (assumed nm) to a human-readable string.
     */
    _format(val) {
        if (val >= 1000000) return (val / 1000000).toFixed(val % 1000000 === 0 ? 0 : 1) + ' mm';
        if (val >= 1000) return (val / 1000).toFixed(val % 1000 === 0 ? 0 : 1) + ' \u00B5m';
        return val + ' nm';
    }
}
