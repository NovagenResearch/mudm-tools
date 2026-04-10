/**
 * Slice plane panel — adds a clipping plane to the renderer.
 *
 * Uses renderer.clippingPlanes (global) so all meshes, including
 * future tiles loaded by TileManager, are automatically clipped.
 */
import * as THREE from 'three';

export class SlicePlanePanel {
    /**
     * @param {HTMLElement} container  DOM element to render controls into
     * @param {{renderer: THREE.WebGLRenderer, scene: THREE.Scene}} opts
     */
    constructor(container, { renderer, scene }) {
        this.renderer = renderer;
        this.scene = scene;

        // Enable local clipping on renderer
        this.renderer.localClippingEnabled = true;

        // State
        this.enabled = false;
        this.axis = 'z';      // 'x' | 'y' | 'z'
        this.flipped = false;
        this.bounds = null;    // THREE.Box3

        // Clipping plane (normal points toward kept half)
        this.clipPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);

        // Visual helper
        this.helperMesh = this._createHelper();
        this.helperMesh.visible = false;
        this.scene.add(this.helperMesh);

        // Build DOM
        this._buildUI(container);
    }

    // --- Public API ---

    /**
     * Update world bounds (called after camera reset / pyramid switch).
     * Auto-sizes the helper plane and resets slider to center.
     */
    updateBounds(box3) {
        this.bounds = box3.clone();
        this._slider.value = 50;
        this._sliderLabel.textContent = '50%';
        this._updatePlane();
    }

    // --- Internals ---

    _createHelper() {
        const DIVISIONS = 24;
        const verts = [];
        // Unit grid from -0.5 to 0.5 in XY plane
        for (let i = 0; i <= DIVISIONS; i++) {
            const t = i / DIVISIONS - 0.5;
            // lines along X
            verts.push(-0.5, t, 0, 0.5, t, 0);
            // lines along Y
            verts.push(t, -0.5, 0, t, 0.5, 0);
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));

        const mat = new THREE.LineBasicMaterial({
            color: 0x6fdfaf,
            transparent: true,
            opacity: 0.35,
            depthWrite: false,
            clippingPlanes: [],  // exempt from global clipping
        });
        const grid = new THREE.LineSegments(geo, mat);
        grid.userData._isSliceHelper = true;
        grid.renderOrder = 999;
        return grid;
    }

    _buildUI(container) {
        // Title
        const h3 = document.createElement('h3');
        h3.textContent = 'Slice Plane';
        container.appendChild(h3);

        // Toggle row
        const toggleRow = document.createElement('div');
        toggleRow.style.cssText = 'display:flex;align-items:center;gap:8px;margin-bottom:8px;';
        const toggleCb = document.createElement('input');
        toggleCb.type = 'checkbox';
        toggleCb.id = 'slice-toggle';
        toggleCb.style.accentColor = '#6fdfaf';
        const toggleLabel = document.createElement('label');
        toggleLabel.htmlFor = 'slice-toggle';
        toggleLabel.textContent = 'Enable clipping';
        toggleLabel.style.cssText = 'cursor:pointer;color:#ccc;';
        toggleRow.appendChild(toggleCb);
        toggleRow.appendChild(toggleLabel);
        container.appendChild(toggleRow);

        // Axis row
        const axisRow = document.createElement('div');
        axisRow.style.cssText = 'display:flex;gap:14px;margin-bottom:8px;';
        for (const ax of ['x', 'y', 'z']) {
            const label = document.createElement('label');
            label.style.cssText = 'display:flex;align-items:center;gap:4px;cursor:pointer;color:#ccc;';
            const radio = document.createElement('input');
            radio.type = 'radio';
            radio.name = 'slice-axis';
            radio.value = ax;
            radio.checked = (ax === 'z');
            radio.style.accentColor = '#6fdfaf';
            radio.addEventListener('change', () => {
                if (!radio.checked) return;
                this.axis = ax;
                this._slider.value = 50;
                this._sliderLabel.textContent = '50%';
                this._updatePlane();
            });
            label.appendChild(radio);
            label.appendChild(document.createTextNode(ax.toUpperCase()));
            axisRow.appendChild(label);
        }
        container.appendChild(axisRow);

        // Slider row
        const sliderRow = document.createElement('div');
        sliderRow.style.cssText = 'display:flex;align-items:center;gap:8px;margin-bottom:8px;';
        const posLabel = document.createElement('span');
        posLabel.textContent = 'Pos:';
        posLabel.style.cssText = 'color:#999;font-size:12px;';
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = '0';
        slider.max = '100';
        slider.step = '1';
        slider.value = '50';
        slider.style.cssText = 'flex:1;height:4px;accent-color:#6fdfaf;cursor:pointer;';
        const valSpan = document.createElement('span');
        valSpan.textContent = '50%';
        valSpan.style.cssText = 'color:#6fdfaf;font-weight:600;min-width:32px;font-size:12px;';
        slider.addEventListener('input', () => {
            valSpan.textContent = `${slider.value}%`;
            this._updatePlane();
        });
        sliderRow.appendChild(posLabel);
        sliderRow.appendChild(slider);
        sliderRow.appendChild(valSpan);
        container.appendChild(sliderRow);

        this._slider = slider;
        this._sliderLabel = valSpan;

        // Flip button
        const flipBtn = document.createElement('button');
        flipBtn.textContent = 'Flip';
        flipBtn.style.cssText =
            'padding:3px 10px;border-radius:4px;border:1px solid rgba(255,255,255,0.15);' +
            'background:rgba(255,255,255,0.06);color:#ccc;font-size:11px;cursor:pointer;';
        flipBtn.addEventListener('click', () => {
            this.flipped = !this.flipped;
            this._updatePlane();
        });
        flipBtn.addEventListener('mouseenter', () => { flipBtn.style.background = 'rgba(255,255,255,0.12)'; });
        flipBtn.addEventListener('mouseleave', () => { flipBtn.style.background = 'rgba(255,255,255,0.06)'; });
        container.appendChild(flipBtn);

        // Toggle handler (last so all refs exist)
        toggleCb.addEventListener('change', () => {
            this.enabled = toggleCb.checked;
            this._updatePlane();
        });
    }

    /**
     * Recompute clip plane position/normal and helper mesh transform.
     */
    _updatePlane() {
        if (!this.bounds) return;

        const pct = parseInt(this._slider.value) / 100;
        const min = this.bounds.min;
        const max = this.bounds.max;
        const size = new THREE.Vector3();
        this.bounds.getSize(size);
        const center = new THREE.Vector3();
        this.bounds.getCenter(center);

        // Plane normal (before flip) and position along axis
        let normal, pos, helperRotation, helperSize;

        switch (this.axis) {
            case 'x': {
                const x = min.x + (max.x - min.x) * pct;
                normal = new THREE.Vector3(-1, 0, 0);
                pos = x;
                helperRotation = new THREE.Euler(0, Math.PI / 2, 0);
                helperSize = [size.y * 1.05, size.z * 1.05];
                break;
            }
            case 'y': {
                const y = min.y + (max.y - min.y) * pct;
                normal = new THREE.Vector3(0, -1, 0);
                pos = y;
                helperRotation = new THREE.Euler(Math.PI / 2, 0, 0);
                helperSize = [size.x * 1.05, size.z * 1.05];
                break;
            }
            case 'z':
            default: {
                const z = min.z + (max.z - min.z) * pct;
                normal = new THREE.Vector3(0, 0, -1);
                pos = z;
                helperRotation = new THREE.Euler(0, 0, 0);
                helperSize = [size.x * 1.05, size.y * 1.05];
                break;
            }
        }

        if (this.flipped) normal.negate();

        // Update THREE.Plane: normal dot point = constant
        this.clipPlane.normal.copy(normal);
        this.clipPlane.constant = -normal.dot(
            this.axis === 'x' ? new THREE.Vector3(pos, 0, 0) :
            this.axis === 'y' ? new THREE.Vector3(0, pos, 0) :
                                new THREE.Vector3(0, 0, pos)
        );

        // Toggle renderer clipping
        if (this.enabled) {
            this.renderer.clippingPlanes = [this.clipPlane];
        } else {
            this.renderer.clippingPlanes = [];
        }

        // Update helper mesh
        this.helperMesh.visible = this.enabled;
        if (this.enabled) {
            // Position helper slightly into the kept half-space so global
            // clipping doesn't discard fragments right on the boundary.
            const helperPos = center.clone();
            if (this.axis === 'x') helperPos.x = pos;
            else if (this.axis === 'y') helperPos.y = pos;
            else helperPos.z = pos;
            const eps = Math.max(size.x, size.y, size.z) * 0.002;
            helperPos.addScaledVector(this.clipPlane.normal, eps);

            this.helperMesh.position.copy(helperPos);
            this.helperMesh.rotation.copy(helperRotation);
            this.helperMesh.scale.set(helperSize[0], helperSize[1], 1);
        }
    }
}
