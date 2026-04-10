/**
 * Orientation gizmo for the main 3D viewport.
 *
 * Renders colored axis arrows (X=red, Y=green, Z=blue) as a viewport
 * inset on the main renderer — no extra WebGL context needed.
 */
import * as THREE from 'three';

const AXIS_COLORS = {
    x: 0xe74c3c,  // red
    y: 0x2ecc71,  // green
    z: 0x3498db,  // blue
};

const AXIS_DIRS = [
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(0, 0, 1),
];

const AXIS_LABELS = ['X', 'Y', 'Z'];

export class AxisGizmo {
    /**
     * @param {number} size - Pixel size of the gizmo viewport
     */
    constructor(size = 120) {
        this.size = size;

        // Own scene for the gizmo
        this.scene = new THREE.Scene();

        // Orthographic camera
        const d = 2.5;
        this.camera = new THREE.OrthographicCamera(-d, d, d, -d, 0.1, 100);
        this.camera.up.set(0, 0, 1);

        this._buildAxes();
        this._buildLabels();
    }

    _buildAxes() {
        const axisLength = 1.2;
        const colors = [AXIS_COLORS.x, AXIS_COLORS.y, AXIS_COLORS.z];

        for (let i = 0; i < 3; i++) {
            const dir = AXIS_DIRS[i];
            const color = colors[i];

            // Shaft
            const geo = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(0, 0, 0),
                dir.clone().multiplyScalar(axisLength),
            ]);
            const mat = new THREE.LineBasicMaterial({ color, linewidth: 2, depthTest: false });
            const line = new THREE.Line(geo, mat);
            line.renderOrder = 1;
            this.scene.add(line);

            // Arrowhead
            const coneGeo = new THREE.ConeGeometry(0.08, 0.25, 8);
            const coneMat = new THREE.MeshBasicMaterial({ color, depthTest: false });
            const cone = new THREE.Mesh(coneGeo, coneMat);
            cone.position.copy(dir.clone().multiplyScalar(axisLength));
            cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
            cone.renderOrder = 1;
            this.scene.add(cone);
        }

        // Origin dot
        const sphereGeo = new THREE.SphereGeometry(0.06, 8, 8);
        const sphereMat = new THREE.MeshBasicMaterial({ color: 0xaaaaaa, depthTest: false });
        const sphere = new THREE.Mesh(sphereGeo, sphereMat);
        sphere.renderOrder = 1;
        this.scene.add(sphere);
    }

    _buildLabels() {
        const axisLength = 1.2;
        const colors = [AXIS_COLORS.x, AXIS_COLORS.y, AXIS_COLORS.z];

        for (let i = 0; i < 3; i++) {
            const sprite = this._makeTextSprite(AXIS_LABELS[i], colors[i]);
            sprite.position.copy(AXIS_DIRS[i].clone().multiplyScalar(axisLength + 0.35));
            sprite.scale.set(0.5, 0.5, 1);
            sprite.renderOrder = 2;
            this.scene.add(sprite);
        }
    }

    _makeTextSprite(text, color) {
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 64;
        const ctx = canvas.getContext('2d');
        ctx.font = 'bold 48px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#' + new THREE.Color(color).getHexString();
        ctx.fillText(text, 32, 32);

        const tex = new THREE.CanvasTexture(canvas);
        tex.minFilter = THREE.LinearFilter;
        const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false });
        return new THREE.Sprite(mat);
    }

    /**
     * Render the gizmo as a viewport inset on the given renderer.
     * Call AFTER the main scene render.
     * @param {THREE.WebGLRenderer} renderer - The main renderer
     * @param {THREE.PerspectiveCamera} mainCamera - The main camera (for rotation)
     */
    render(renderer, mainCamera) {
        // Match main camera rotation
        const dist = 5;
        const offset = new THREE.Vector3(0, 0, dist).applyQuaternion(mainCamera.quaternion);
        this.camera.position.copy(offset);
        this.camera.quaternion.copy(mainCamera.quaternion);
        this.camera.up.copy(mainCamera.up);
        this.camera.lookAt(0, 0, 0);

        // Render as inset viewport (top-left of canvas)
        const px = renderer.getPixelRatio();
        const s = this.size * px;
        renderer.setViewport(0, renderer.domElement.height - s, s, s);
        renderer.setScissor(0, renderer.domElement.height - s, s, s);
        renderer.setScissorTest(true);
        renderer.setClearColor(0x000000, 0);
        renderer.clearDepth();
        renderer.render(this.scene, this.camera);

        // Restore full viewport
        renderer.setScissorTest(false);
        renderer.setViewport(0, 0, renderer.domElement.width, renderer.domElement.height);
    }
}
