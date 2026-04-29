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
    ['neuron_name', 'Neuron'],
    ['neuron_id', 'Neuron ID'],
    ['compartment', 'Compartment'],
    ['source', 'Source'],
    ['archive', 'Archive'],
    ['species', 'Species'],
    ['brain_region', 'Brain Region'],
    ['brain_regions', 'Brain Regions'],
    ['cell_type', 'Cell Type'],
    ['body_id', 'Body ID'],
    ['pre', 'Pre-synapses'],
    ['post', 'Post-synapses'],
    ['status', 'Status'],
    ['status_label', 'Status Label'],
    ['soma_radius', 'Soma Radius'],
    ['surface_m', 'Surface (µm²)'],
    ['volume_m', 'Volume (µm³)'],
    ['length', 'Length (µm)'],
    ['n_bifs', 'Bifurcations'],
    ['n_branch', 'Branches'],
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
        // Optional lookup: (name) => {...props from features.json} | undefined
        // set by main.js so clicks can show richer per-feature metadata than
        // what's baked into GLB node.extras. Leaves existing deployments
        // (mouselight etc.) unchanged when no lookup is wired.
        this.featureLookup = null;

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
            // Skip the atlas overlay (CCF isocortex) — it's foreground
            // chrome, not selectable. Walk up so a hit on any descendant
            // mesh of the overlay group is skipped, since `_isAtlasOverlay`
            // is set on the leaf meshes during traversal in main.js.
            let _ovr = hit.object;
            while (_ovr) {
                if (_ovr.userData?._isAtlasOverlay) break;
                _ovr = _ovr.parent;
            }
            if (_ovr) continue;
            // Skip hits behind the clip plane (clipped geometry)
            if (this.slicePanel?.enabled && this.slicePanel.clipPlane.distanceToPoint(hit.point) < 0) continue;
            const props = this._findProperties(hit.object);
            if (props) {
                // Merge per-feature sidecar props if a lookup is wired up.
                let merged = props;
                if (this.featureLookup) {
                    const extra = this.featureLookup(props.name || props.acronym);
                    if (extra && typeof extra === 'object') {
                        // userData fields take precedence (they're mesh-accurate);
                        // features.json fills in the rest.
                        merged = { ...extra, ...props };
                    }
                }
                this._showPanel(merged);
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
