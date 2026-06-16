/**
 * Info panel for displaying clicked mesh properties.
 *
 * Uses raycasting to detect clicks on meshes, then displays
 * feature properties from node.userData (set by glTF extras).
 */
import * as THREE from 'three';

const DISPLAY_FIELDS = [
    ['name', 'Name'],
    ['acronym', 'Acronym'],
    ['cell_type', 'Cell Type'],
    ['body_id', 'Body ID'],
    ['brain_regions', 'Brain Regions'],
    ['pre', 'Pre-synapses'],
    ['post', 'Post-synapses'],
    ['status', 'Status'],
    ['status_label', 'Status Label'],
    ['soma_radius', 'Soma Radius'],
    ['size_voxels', 'Size (voxels)'],
    ['ccf_id', 'CCF ID'],
    ['parent_name', 'Parent'],
    ['vertex_count', 'Vertices'],
    ['face_count', 'Faces'],
];

export class InfoPanel {
    constructor(camera, scene, canvas) {
        this.camera = camera;
        this.scene = scene;
        this.canvas = canvas;
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        this.slicePanel = null;  // set from main.js

        this.panel = document.getElementById('info-panel');
        this.title = document.getElementById('info-title');
        this.tableBody = document.querySelector('#info-table tbody');

        canvas.addEventListener('click', e => this._onClick(e));
    }

    _onClick(event) {
        // Ignore if clicking on the panel itself
        if (this.panel.contains(event.target)) return;

        const rect = this.canvas.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.scene.children, true);

        // Find first intersection that resolves to a feature.
        for (const hit of intersects) {
            // Skip the slice plane helper mesh
            if (hit.object.userData?._isSliceHelper) continue;
            // Skip hits behind the clip plane (clipped geometry)
            if (this.slicePanel?.enabled && this.slicePanel.clipPlane.distanceToPoint(hit.point) < 0) continue;
            const props = this._propsForHit(hit);
            if (props) {
                this._showPanel(props);
                return;
            }
        }

        // Clicked empty space
        this.panel.style.display = 'none';
    }

    /**
     * Resolve a raycast hit to feature properties. BatchedMesh hits carry the instance
     * id in hit.batchId (and raycast only reports visible instances); legacy line/point
     * meshes walk the parent chain for userData.
     */
    _propsForHit(hit) {
        const obj = hit.object;
        if (obj.isBatchedMesh && hit.batchId != null && obj.propsByInstance) {
            return obj.propsByInstance.get(hit.batchId) || null;
        }
        if (!obj.visible) return null;
        return this._findProperties(obj);
    }

    /**
     * Walk up the parent chain to find a node with meaningful userData.
     */
    _findProperties(object) {
        let current = object;
        while (current) {
            if (current.userData && (current.userData.name || current.userData.acronym)) {
                return current.userData;
            }
            current = current.parent;
        }
        return null;
    }

    _showPanel(props) {
        this.title.textContent = props.name || props.acronym || 'Unknown';
        this.tableBody.innerHTML = '';

        // No whitelist — display ALL metadata. DISPLAY_FIELDS is used only to give
        // known fields friendly labels and a preferred order; every other field is
        // shown too, with a humanized label. Object/array values (e.g. the internal
        // `tiles` list) are skipped since they aren't scalar metadata.
        const LABELS = Object.fromEntries(DISPLAY_FIELDS);
        const humanize = k => k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        const order = DISPLAY_FIELDS.map(([k]) => k).filter(k => k in props);
        for (const k of Object.keys(props)) if (!order.includes(k)) order.push(k);

        for (const key of order) {
            const val = props[key];
            if (val === undefined || val === null || typeof val === 'object') continue;

            const tr = document.createElement('tr');
            const tdLabel = document.createElement('td');
            tdLabel.textContent = LABELS[key] || humanize(key);
            const tdVal = document.createElement('td');

            if (key === 'color') {
                const swatch = document.createElement('span');
                swatch.className = 'color-swatch';
                swatch.style.backgroundColor = val;
                tdVal.appendChild(swatch);
                tdVal.appendChild(document.createTextNode(val));
            } else if (typeof val === 'number') {
                tdVal.textContent = val.toLocaleString();
            } else {
                tdVal.textContent = val;
            }

            tr.appendChild(tdLabel);
            tr.appendChild(tdVal);
            this.tableBody.appendChild(tr);
        }

        this.panel.style.display = 'block';
    }
}
