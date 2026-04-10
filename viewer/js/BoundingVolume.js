/**
 * OGC 3D Tiles bounding volume utilities.
 *
 * OGC axis-aligned box format (12 floats):
 *   [cx, cy, cz, hx, 0, 0, 0, hy, 0, 0, 0, hz]
 */
import * as THREE from 'three';

/**
 * Convert OGC 12-float box to THREE.Box3.
 */
export function ogcBoxToBox3(box12) {
    const cx = box12[0], cy = box12[1], cz = box12[2];
    const hx = box12[3];
    const hy = box12[7];
    const hz = box12[11];
    return new THREE.Box3(
        new THREE.Vector3(cx - hx, cy - hy, cz - hz),
        new THREE.Vector3(cx + hx, cy + hy, cz + hz),
    );
}

/**
 * Screen-space error in pixels.
 */
export function computeSSE(geometricError, distance, screenHeight, fov) {
    if (distance <= 0) return Infinity;
    return (geometricError / distance) * screenHeight / (2 * Math.tan(fov / 2));
}
