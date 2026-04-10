/// OGC 3D Tiles 1.1 `tileset.json` generation.
///
/// Produces a hierarchical tileset descriptor with axis-aligned bounding
/// volumes and geometric error that decreases with zoom level.

use serde_json::{json, Value};
use std::collections::HashSet;

/// Build an OGC 3D Tiles oriented bounding box (12 floats, axis-aligned).
fn box_volume(
    xmin: f64, ymin: f64, zmin: f64,
    xmax: f64, ymax: f64, zmax: f64,
) -> Vec<f64> {
    let cx = (xmin + xmax) / 2.0;
    let cy = (ymin + ymax) / 2.0;
    let cz = (zmin + zmax) / 2.0;
    let hx = (xmax - xmin) / 2.0;
    let hy = (ymax - ymin) / 2.0;
    let hz = (zmax - zmin) / 2.0;
    vec![cx, cy, cz, hx, 0.0, 0.0, 0.0, hy, 0.0, 0.0, 0.0, hz]
}

/// Compute geometric error for a zoom level.
///
/// At max_zoom the error is 0. At each higher level the error
/// doubles, representing the spatial resolution loss.
fn geometric_error(
    world_bounds: &(f64, f64, f64, f64, f64, f64),
    zoom: u32,
    max_zoom: u32,
) -> f64 {
    if zoom >= max_zoom {
        return 0.0;
    }
    let (xmin, ymin, zmin, xmax, ymax, zmax) = *world_bounds;
    let dx = xmax - xmin;
    let dy = ymax - ymin;
    let dz = zmax - zmin;
    let diagonal = (dx * dx + dy * dy + dz * dz).sqrt();
    diagonal / (1u64 << max_zoom) as f64 * (1u64 << (max_zoom - zoom)) as f64
}

/// Compute world-space bounding box for a tile address.
fn tile_bounds_world(
    z: u32, x: u32, y: u32, d: u32,
    world_bounds: &(f64, f64, f64, f64, f64, f64),
) -> (f64, f64, f64, f64, f64, f64) {
    let (xmin, ymin, zmin, xmax, ymax, zmax) = *world_bounds;
    let dx = xmax - xmin;
    let dy = ymax - ymin;
    let dz = zmax - zmin;
    let n = (1u32 << z) as f64;

    let wx0 = xmin + (x as f64 / n) * dx;
    let wy0 = ymin + (y as f64 / n) * dy;
    let wz0 = zmin + (d as f64 / n) * dz;
    let wx1 = xmin + ((x + 1) as f64 / n) * dx;
    let wy1 = ymin + ((y + 1) as f64 / n) * dy;
    let wz1 = zmin + ((d + 1) as f64 / n) * dz;

    (wx0, wy0, wz0, wx1, wy1, wz1)
}

/// Generate OGC 3D Tiles 1.1 tileset.json structure.
///
/// Args:
///   tile_keys: list of (z, x, y, d) tile addresses that have content.
///   world_bounds: (xmin, ymin, zmin, xmax, ymax, zmax) in world coordinates.
///   min_zoom, max_zoom: zoom range.
///
/// Returns: serde_json Value ready for serialization.
pub(crate) fn generate_tileset_json(
    tile_keys: &[(u32, u32, u32, u32)],
    world_bounds: &(f64, f64, f64, f64, f64, f64),
    min_zoom: u32,
    max_zoom: u32,
) -> Value {
    let key_set: HashSet<(u32, u32, u32, u32)> = tile_keys.iter().copied().collect();

    let root_error = geometric_error(world_bounds, min_zoom, max_zoom);
    let root_box = box_volume(
        world_bounds.0, world_bounds.1, world_bounds.2,
        world_bounds.3, world_bounds.4, world_bounds.5,
    );

    // Build root-level tiles
    let mut root_children: Vec<Value> = Vec::new();
    for &(z, x, y, d) in tile_keys {
        if z == min_zoom {
            if let Some(node) = build_node(z, x, y, d, &key_set, world_bounds, max_zoom) {
                root_children.push(node);
            }
        }
    }

    let root = if root_children.len() == 1 {
        let mut r = root_children.into_iter().next().unwrap();
        let obj = r.as_object_mut().unwrap();
        obj.insert("refine".to_string(), json!("REPLACE"));
        obj.insert("geometricError".to_string(), json!(root_error));
        obj.insert(
            "boundingVolume".to_string(),
            json!({"box": root_box}),
        );
        r
    } else {
        json!({
            "boundingVolume": {"box": root_box},
            "geometricError": root_error,
            "refine": "REPLACE",
            "children": root_children,
        })
    };

    json!({
        "asset": {"version": "1.1"},
        "geometricError": root_error,
        "root": root,
    })
}

/// Recursively build a tileset node.
fn build_node(
    z: u32, x: u32, y: u32, d: u32,
    key_set: &HashSet<(u32, u32, u32, u32)>,
    world_bounds: &(f64, f64, f64, f64, f64, f64),
    max_zoom: u32,
) -> Option<Value> {
    if !key_set.contains(&(z, x, y, d)) {
        return None;
    }

    let tb = tile_bounds_world(z, x, y, d, world_bounds);
    let bv = box_volume(tb.0, tb.1, tb.2, tb.3, tb.4, tb.5);
    let geo_err = geometric_error(world_bounds, z, max_zoom);

    let mut node = json!({
        "boundingVolume": {"box": bv},
        "geometricError": geo_err,
        "content": {"uri": format!("{}/{}/{}/{}.glb", z, x, y, d)},
    });

    if z < max_zoom {
        let mut children: Vec<Value> = Vec::new();
        let nz = z + 1;
        for cx in [x * 2, x * 2 + 1] {
            for cy in [y * 2, y * 2 + 1] {
                for cd in [d * 2, d * 2 + 1] {
                    if let Some(child) =
                        build_node(nz, cx, cy, cd, key_set, world_bounds, max_zoom)
                    {
                        children.push(child);
                    }
                }
            }
        }
        if !children.is_empty() {
            node.as_object_mut()
                .unwrap()
                .insert("children".to_string(), json!(children));
        }
    }

    Some(node)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_volume() {
        let bv = box_volume(0.0, 0.0, 0.0, 10.0, 20.0, 30.0);
        assert_eq!(bv.len(), 12);
        // Center
        assert_eq!(bv[0], 5.0);
        assert_eq!(bv[1], 10.0);
        assert_eq!(bv[2], 15.0);
        // Half-extents
        assert_eq!(bv[3], 5.0);
        assert_eq!(bv[7], 10.0);
        assert_eq!(bv[11], 15.0);
    }

    #[test]
    fn test_geometric_error() {
        let bounds = (0.0, 0.0, 0.0, 100.0, 100.0, 100.0);
        let err_z0 = geometric_error(&bounds, 0, 3);
        let err_z1 = geometric_error(&bounds, 1, 3);
        let err_z3 = geometric_error(&bounds, 3, 3);

        assert!(err_z0 > err_z1);
        assert_eq!(err_z3, 0.0);
        // Each level halves
        assert!((err_z0 / err_z1 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_generate_tileset_json() {
        let keys = vec![(0, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 1), (1, 1, 0, 0)];
        let bounds = (0.0, 0.0, 0.0, 100.0, 100.0, 100.0);
        let tileset = generate_tileset_json(&keys, &bounds, 0, 1);

        assert_eq!(tileset["asset"]["version"], "1.1");
        assert!(tileset["root"]["content"]["uri"].is_string());
        assert!(tileset["root"]["children"].is_array());

        let children = tileset["root"]["children"].as_array().unwrap();
        assert_eq!(children.len(), 3);
    }
}
