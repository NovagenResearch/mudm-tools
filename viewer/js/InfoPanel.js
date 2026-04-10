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

        // Find first VISIBLE intersection with userData
        for (const hit of intersects) {
            // Skip invisible meshes (non-selected features still in scene)
            if (!hit.object.visible) continue;
            // Skip the slice plane helper mesh
            if (hit.object.userData?._isSliceHelper) continue;
            // Skip hits behind the clip plane (clipped geometry)
            if (this.slicePanel?.enabled && this.slicePanel.clipPlane.distanceToPoint(hit.point) < 0) continue;
            const props = this._findProperties(hit.object);
            if (props) {
                this._showPanel(props);
                return;
            }
        }

        // Clicked empty space
        this.panel.style.display = 'none';
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

        for (const [key, label] of DISPLAY_FIELDS) {
            const val = props[key];
            if (val === undefined || val === null) continue;

            const tr = document.createElement('tr');
            const tdLabel = document.createElement('td');
            tdLabel.textContent = label;
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
