/// Streaming 3D tile generator — processes features one at a time.
///
/// Memory model:
/// - During ingestion (add_feature): O(1 feature) — clip through octree, write fragments
/// - During encoding (generate_pbf3): O(all fragments) — read from disk, transform, encode
///
/// Architecture:
///   For each feature: clip through octree → write binary fragments to temp file
///   For each tile (rayon parallel): read fragments → transform → encode → write .pbf3
///
/// # API differences from StreamingTileGenerator2D (streaming2d.rs)
///
/// **Constructor**: 3D has additional params: `extent`, `extent_z`, `base_cells`, `num_buckets`.
/// Both share: `min_zoom`, `max_zoom`, `buffer`, `temp_dir`.
///
/// **generate_pbf3 vs generate_mvt**: 3D encodes PBF3/GLB/Neuroglancer formats with Z
/// coordinates and 3D geometry types (TIN, PolyhedralSurface). 2D encodes MVT (Mapbox
/// Vector Tiles) with XY-only geometry. Unifying signatures would require breaking changes.
///
/// **Precision**: 3D uses `extent` (default 4096) and `extent_z` for quantization.
/// 2D uses `extent` only. A future `precision` parameter could unify these.
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use rayon::prelude::*;
use ahash::AHashMap;

use crate::types::BBox3D;
use crate::fragment::{Fragment3D, Fragment3DWriter, Fragment3DReader};
use crate::encoder_pbf3;
use crate::encoder_glb::{self, GlbFeature};
use crate::tileset_json;
use crate::tile_transform;
use crate::obj_parser;
use crate::simplify;

// Geometry type constants matching protobuf GeomType.
const POINT3D: u8 = 1;
const LINESTRING3D: u8 = 2;
const POLYGON3D: u8 = 3;
const POLYHEDRALSURFACE: u8 = 4;
const TIN: u8 = 5;

/// Small epsilon for boundary points (matches Python octree._EPS).
const EPS: f64 = 1e-10;

/// Check if a triangle (given as indices into a positions array) is non-degenerate.
/// Returns false if indices are duplicate, altitude < 1% of longest edge
/// (sliver artifact), or longest edge exceeds `max_edge_sq` (oversized artifact
/// from QEM simplification at coarse zoom levels).
#[inline]
fn is_valid_triangle(positions: &[f32], tri: &[u32; 3], max_edge_sq: f32) -> bool {
    if tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] {
        return false;
    }
    let p0 = &positions[(tri[0] as usize * 3)..][..3];
    let p1 = &positions[(tri[1] as usize * 3)..][..3];
    let p2 = &positions[(tri[2] as usize * 3)..][..3];
    let e0 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let e1 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let cx = e0[1] * e1[2] - e0[2] * e1[1];
    let cy = e0[2] * e1[0] - e0[0] * e1[2];
    let cz = e0[0] * e1[1] - e0[1] * e1[0];
    let area_sq = (cx * cx + cy * cy + cz * cz) as f64;
    let e2 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    let longest_edge_sq = ((e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2]) as f64)
        .max((e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]) as f64)
        .max((e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]) as f64);
    // Reject oversized triangles (edge > threshold)
    if longest_edge_sq > max_edge_sq as f64 {
        return false;
    }
    // Reject slivers (altitude < 1% of longest edge)
    longest_edge_sq <= 0.0 || area_sq >= 1e-4 * longest_edge_sq * longest_edge_sq
}

/// Remove oversized triangles from an index buffer after QEM simplification.
/// Triangles with longest edge exceeding `max_edge_sq` are dropped in-place.
fn filter_oversized_triangles(positions: &[f32], indices: &mut Vec<u32>, max_edge_sq: f32) {
    let n_tris = indices.len() / 3;
    let mut write = 0usize;
    for t in 0..n_tris {
        let tri = [indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]];
        if is_valid_triangle(positions, &tri, max_edge_sq) {
            if write != t * 3 {
                indices[write] = tri[0];
                indices[write + 1] = tri[1];
                indices[write + 2] = tri[2];
            }
            write += 3;
        }
    }
    indices.truncate(write);
}

/// Compute max allowed edge² for a tile at a given zoom level (normalized [0,1]³ coords).
/// Threshold = (3 × tile diagonal)².
#[inline]
fn tile_max_edge_sq_normalized(base_cells: u32, zoom: u32) -> f32 {
    let cells = base_cells as f64 * (1u64 << zoom) as f64;
    let cell_size = 1.0 / cells;
    // 3× tile diagonal: 9 × 3 × cell_size²
    (27.0 * cell_size * cell_size) as f32
}

/// Compute max allowed edge² for a tile at a given zoom level (world coords).
/// Threshold = (3 × tile diagonal)².
#[inline]
fn tile_max_edge_sq_world(base_cells: u32, zoom: u32, dx: f64, dy: f64, dz: f64) -> f32 {
    let cells = base_cells as f64 * (1u64 << zoom) as f64;
    let cx = dx / cells;
    let cy = dy / cells;
    let cz = dz / cells;
    (9.0 * (cx * cx + cy * cy + cz * cz)) as f32
}

/// Unique generator ID counter for temp file naming.
static GENERATOR_ID: AtomicU32 = AtomicU32::new(0);

// ---------------------------------------------------------------------------
// Tag value type for dictionary-encoded properties
// ---------------------------------------------------------------------------

/// Tag value matching protobuf Value variants.
#[derive(Debug, Clone)]
pub(crate) enum TagValue {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

// ---------------------------------------------------------------------------
// Intermediate feature for pure-Rust octree clipping
// ---------------------------------------------------------------------------

/// Feature in normalized [0,1]³ space with bounding box for clip acceleration.
#[derive(Debug, Clone)]
struct ClipFeature {
    xy: Vec<f64>,
    z: Vec<f64>,
    ring_lengths: Vec<u32>,
    geom_type: u8,
    bbox: BBox3D,
}

fn compute_bbox_rs(xy: &[f64], z: &[f64]) -> BBox3D {
    let mut bb = BBox3D::empty();
    let n = z.len();
    for j in 0..n {
        bb.expand(xy[j * 2], xy[j * 2 + 1], z[j]);
    }
    bb
}

// ---------------------------------------------------------------------------
// Pre-simplify helpers: build indexed mesh, convert back to ClipFeature,
// and simplify-then-clip per zoom level
// ---------------------------------------------------------------------------

/// Convert OBJ vertices + faces to an indexed f32 mesh in [0,1]³ normalized coords.
///
/// Returns (positions, indices, bbox) where:
/// - positions: flat f32 [x,y,z, x,y,z, ...] in [0,1]³
/// - indices: flat u32 triangle indices [i0,i1,i2, ...]
/// - bbox: bounding box of the normalized positions
fn build_indexed_mesh(
    vertices: &[[f64; 3]],
    faces: &[[u32; 3]],
    xmin: f64, ymin: f64, zmin: f64,
    dx: f64, dy: f64, dz: f64,
) -> (Vec<f32>, Vec<u32>, BBox3D) {
    let mut positions = Vec::with_capacity(vertices.len() * 3);
    let mut bb = BBox3D::empty();

    for v in vertices {
        let px = ((v[0] - xmin) / dx) as f32;
        let py = ((v[1] - ymin) / dy) as f32;
        let pz = ((v[2] - zmin) / dz) as f32;
        positions.push(px);
        positions.push(py);
        positions.push(pz);
        bb.expand(px as f64, py as f64, pz as f64);
    }

    let mut indices = Vec::with_capacity(faces.len() * 3);
    for face in faces {
        indices.push(face[0]);
        indices.push(face[1]);
        indices.push(face[2]);
    }

    (positions, indices, bb)
}

/// Convert an indexed f32 mesh back to ring-based ClipFeature format.
///
/// Each triangle becomes a 4-vertex ring (v0, v1, v2, v0) with f64 coords,
/// matching the format expected by `octree_clip()`.
fn indexed_mesh_to_clip_feature(
    positions: &[f32],
    indices: &[u32],
    bbox: BBox3D,
) -> ClipFeature {
    let n_tris = indices.len() / 3;
    let n_verts_total = n_tris * 4; // 4 verts per ring (closed triangle)
    let mut xy = Vec::with_capacity(n_verts_total * 2);
    let mut z = Vec::with_capacity(n_verts_total);
    let ring_lengths = vec![4u32; n_tris];

    for tri in 0..n_tris {
        let i0 = indices[tri * 3] as usize;
        let i1 = indices[tri * 3 + 1] as usize;
        let i2 = indices[tri * 3 + 2] as usize;
        // 4 vertices: v0, v1, v2, v0 (closed ring)
        for &vi in &[i0, i1, i2, i0] {
            xy.push(positions[vi * 3] as f64);
            xy.push(positions[vi * 3 + 1] as f64);
            z.push(positions[vi * 3 + 2] as f64);
        }
    }

    ClipFeature { xy, z, ring_lengths, geom_type: TIN, bbox }
}

/// Pre-simplify a mesh and clip per zoom level using cascaded simplification.
///
/// Cascading: simplify full mesh → zoom max-1, then that result → zoom max-2, etc.
/// Each step starts from the previous (smaller) mesh, so QEM runs on progressively
/// fewer triangles. For a 10M-tri mesh with max_zoom=4:
///   zoom 4: 10M tris (full)
///   zoom 3: simplify 10M → 150K (expensive, but only once)
///   zoom 2: simplify 150K → 20K (fast)
///   zoom 1: simplify 20K → 2K (instant)
///   zoom 0: simplify 2K → 600 (instant)
fn simplify_and_clip_per_zoom(
    positions: &[f32],
    indices: &[u32],
    bbox: BBox3D,
    min_zoom: u32,
    max_zoom: u32,
    base_cells: u32,
    buffer: f64,
) -> Vec<((u32, u32, u32, u32), ClipFeature)> {
    let mut result = Vec::new();
    let full_index_count = indices.len();

    // Start with full mesh, cascade downward
    let mut cur_pos = positions.to_vec();
    let mut cur_idx = indices.to_vec();

    for zoom in (min_zoom..=max_zoom).rev() {
        if zoom < max_zoom {
            // Cascade: simplify from current (already reduced) mesh
            let target_idx = simplify::compute_target_index_count(
                base_cells, zoom, max_zoom, full_index_count,
            );
            let target_tris = target_idx / 3;
            if target_tris == 0 {
                continue;
            }
            let (sp, mut si) = simplify::simplify_mesh(&cur_pos, &cur_idx, target_tris);
            if si.is_empty() {
                continue;
            }
            filter_oversized_triangles(&sp, &mut si, tile_max_edge_sq_normalized(base_cells, zoom));
            if si.is_empty() {
                continue;
            }
            cur_pos = sp;
            cur_idx = si;
        }

        let clip_feat = indexed_mesh_to_clip_feature(&cur_pos, &cur_idx, bbox);
        let mut clipped = octree_clip(&clip_feat, zoom, zoom, buffer);
        result.append(&mut clipped);
    }

    result
}

// ---------------------------------------------------------------------------
// Pure-Rust clip functions (matching clip3d.py / clip.rs logic)
// ---------------------------------------------------------------------------

/// Clip a list of features to [k1, k2) along axis (0=X, 1=Y, 2=Z).
fn clip_features(features: &[ClipFeature], k1: f64, k2: f64, axis: u8) -> Vec<ClipFeature> {
    let mut result = Vec::new();
    for feat in features {
        let (a_min, a_max) = match axis {
            0 => (feat.bbox.min_x, feat.bbox.max_x),
            1 => (feat.bbox.min_y, feat.bbox.max_y),
            _ => (feat.bbox.min_z, feat.bbox.max_z),
        };

        // Trivial reject — half-open interval [k1, k2)
        if a_min >= k2 || a_max < k1 {
            continue;
        }

        // Trivial accept — fully within [k1, k2)
        if a_min >= k1 && a_max < k2 {
            result.push(feat.clone());
            continue;
        }

        // Type-specific clipping
        match feat.geom_type {
            TIN | POLYHEDRALSURFACE => {
                if let Some(clipped) = clip_surface_rs(feat, k1, k2, axis) {
                    result.push(clipped);
                }
            }
            POINT3D => {
                result.extend(clip_points_rs(feat, k1, k2, axis));
            }
            LINESTRING3D => {
                result.extend(clip_line_rs(feat, k1, k2, axis));
            }
            POLYGON3D => {
                // Trivial accept (same as Python _clip_polygon)
                result.push(feat.clone());
            }
            _ => {}
        }
    }
    result
}

/// Clip TIN/PolyhedralSurface per-face — keep faces overlapping [k1, k2).
fn clip_surface_rs(feat: &ClipFeature, k1: f64, k2: f64, axis: u8) -> Option<ClipFeature> {
    let ring_lengths = if feat.ring_lengths.is_empty() {
        vec![feat.z.len() as u32]
    } else {
        feat.ring_lengths.clone()
    };

    let mut out_xy = Vec::new();
    let mut out_z = Vec::new();
    let mut out_rl = Vec::new();

    let mut offset = 0usize;
    for &rl in &ring_lengths {
        let rl = rl as usize;
        let mut f_min = f64::INFINITY;
        let mut f_max = f64::NEG_INFINITY;

        for j in 0..rl {
            let val = match axis {
                0 => feat.xy[(offset + j) * 2],
                1 => feat.xy[(offset + j) * 2 + 1],
                _ => feat.z[offset + j],
            };
            if val < f_min { f_min = val; }
            if val > f_max { f_max = val; }
        }

        if f_min >= k2 || f_max < k1 {
            offset += rl;
            continue;
        }

        out_xy.extend_from_slice(&feat.xy[offset * 2..(offset + rl) * 2]);
        out_z.extend_from_slice(&feat.z[offset..offset + rl]);
        out_rl.push(rl as u32);
        offset += rl;
    }

    if out_rl.is_empty() {
        return None;
    }

    let bbox = compute_bbox_rs(&out_xy, &out_z);
    Some(ClipFeature {
        xy: out_xy,
        z: out_z,
        ring_lengths: out_rl,
        geom_type: feat.geom_type,
        bbox,
    })
}

/// Clip LineString3D to [k1, k2) — returns zero or more line segments.
fn clip_line_rs(feat: &ClipFeature, k1: f64, k2: f64, axis: u8) -> Vec<ClipFeature> {
    let xy = &feat.xy;
    let z = &feat.z;
    let n = z.len();

    if n < 2 {
        return vec![];
    }

    let axis_val = |idx: usize| -> f64 {
        match axis {
            0 => xy[idx * 2],
            1 => xy[idx * 2 + 1],
            _ => z[idx],
        }
    };

    let intersect_at = |i: usize, j: usize, k: f64| -> (f64, f64, f64) {
        let ax = xy[i * 2]; let ay = xy[i * 2 + 1]; let az = z[i];
        let bx = xy[j * 2]; let by = xy[j * 2 + 1]; let bz = z[j];
        let t = match axis {
            0 => { let d = bx - ax; if d != 0.0 { (k - ax) / d } else { 0.0 } }
            1 => { let d = by - ay; if d != 0.0 { (k - ay) / d } else { 0.0 } }
            _ => { let d = bz - az; if d != 0.0 { (k - az) / d } else { 0.0 } }
        };
        (ax + (bx - ax) * t, ay + (by - ay) * t, az + (bz - az) * t)
    };

    let mut segments: Vec<ClipFeature> = Vec::new();
    let mut out_xy: Vec<f64> = Vec::new();
    let mut out_z: Vec<f64> = Vec::new();

    let flush = |out_xy: &mut Vec<f64>, out_z: &mut Vec<f64>, segments: &mut Vec<ClipFeature>| {
        if out_z.len() >= 2 {
            let bbox = compute_bbox_rs(out_xy, out_z);
            segments.push(ClipFeature {
                xy: std::mem::take(out_xy),
                z: std::mem::take(out_z),
                ring_lengths: vec![],
                geom_type: LINESTRING3D,
                bbox,
            });
        } else {
            out_xy.clear();
            out_z.clear();
        }
    };

    for i in 0..n - 1 {
        let a_val = axis_val(i);
        let b_val = axis_val(i + 1);
        let a_in = k1 <= a_val && a_val < k2;

        if a_in {
            out_xy.push(xy[i * 2]);
            out_xy.push(xy[i * 2 + 1]);
            out_z.push(z[i]);
        }

        let cross_k1 = (a_val < k1 && b_val > k1) || (b_val < k1 && a_val > k1);
        let cross_k2 = (a_val < k2 && b_val >= k2) || (b_val < k2 && a_val >= k2);

        let mut crossings: Vec<(f64, f64, f64, f64, bool)> = Vec::new();
        if cross_k1 {
            let (ix, iy, iz) = intersect_at(i, i + 1, k1);
            let d = b_val - a_val;
            let t = if d != 0.0 { (k1 - a_val) / d } else { 0.0 };
            let entering = b_val > a_val;
            crossings.push((t, ix, iy, iz, entering));
        }
        if cross_k2 {
            let (ix, iy, iz) = intersect_at(i, i + 1, k2);
            let d = b_val - a_val;
            let t = if d != 0.0 { (k2 - a_val) / d } else { 0.0 };
            let entering = b_val < a_val;
            crossings.push((t, ix, iy, iz, entering));
        }

        crossings.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for (_, ix, iy, iz, entering) in crossings {
            if entering {
                flush(&mut out_xy, &mut out_z, &mut segments);
                out_xy.push(ix);
                out_xy.push(iy);
                out_z.push(iz);
            } else {
                out_xy.push(ix);
                out_xy.push(iy);
                out_z.push(iz);
                flush(&mut out_xy, &mut out_z, &mut segments);
            }
        }
    }

    // Last point
    if n > 0 {
        let last_val = axis_val(n - 1);
        if k1 <= last_val && last_val < k2 {
            out_xy.push(xy[(n - 1) * 2]);
            out_xy.push(xy[(n - 1) * 2 + 1]);
            out_z.push(z[n - 1]);
        }
    }

    flush(&mut out_xy, &mut out_z, &mut segments);
    segments
}

/// Clip points to [k1, k2) — filter by axis value.
fn clip_points_rs(feat: &ClipFeature, k1: f64, k2: f64, axis: u8) -> Vec<ClipFeature> {
    let mut results = Vec::new();
    let n = feat.z.len();

    for i in 0..n {
        let val = match axis {
            0 => feat.xy[i * 2],
            1 => feat.xy[i * 2 + 1],
            _ => feat.z[i],
        };
        if k1 <= val && val < k2 {
            let px = feat.xy[i * 2];
            let py = feat.xy[i * 2 + 1];
            let pz = feat.z[i];
            results.push(ClipFeature {
                xy: vec![px, py],
                z: vec![pz],
                ring_lengths: vec![],
                geom_type: POINT3D,
                bbox: BBox3D {
                    min_x: px, min_y: py, min_z: pz,
                    max_x: px, max_y: py, max_z: pz,
                },
            });
        }
    }
    results
}

// ---------------------------------------------------------------------------
// Octree traversal — clip one feature across all zoom levels
// ---------------------------------------------------------------------------

/// Clip a feature through the octree, producing (tile_key, clipped_feature) pairs.
///
/// Matches the stack-based octree in `octree.py`:
/// - Split X → Y → Z at each level
/// - Store tile at each zoom in [min_zoom, max_zoom]
/// - Pad upper boundaries with EPS for boundary inclusion
fn octree_clip(
    feat: &ClipFeature,
    min_zoom: u32,
    max_zoom: u32,
    buffer: f64,
) -> Vec<((u32, u32, u32, u32), ClipFeature)> {
    let mut result = Vec::new();

    // Stack: (zoom, x, y, d, features_at_this_node)
    let mut stack: Vec<(u32, u32, u32, u32, Vec<ClipFeature>)> = vec![
        (0, 0, 0, 0, vec![feat.clone()])
    ];

    while let Some((z, x, y, d, feats)) = stack.pop() {
        if feats.is_empty() {
            continue;
        }

        // Store at this level if within zoom range
        if z >= min_zoom {
            for f in &feats {
                result.push(((z, x, y, d), f.clone()));
            }
        }

        if z >= max_zoom {
            continue;
        }

        // Split into 8 octants
        let nz = z + 1;
        let n = 1u32 << nz;

        let x0 = (x * 2) as f64 / n as f64;
        let x1 = (x * 2 + 1) as f64 / n as f64;
        let x2 = (x * 2 + 2) as f64 / n as f64;
        let y0 = (y * 2) as f64 / n as f64;
        let y1 = (y * 2 + 1) as f64 / n as f64;
        let y2 = (y * 2 + 2) as f64 / n as f64;
        let d0 = (d * 2) as f64 / n as f64;
        let d1 = (d * 2 + 1) as f64 / n as f64;
        let d2 = (d * 2 + 2) as f64 / n as f64;

        let buf = if buffer > 0.0 { buffer / n as f64 } else { 0.0 };

        let x2_pad = if x * 2 + 2 == n { x2 + EPS } else { x2 };
        let y2_pad = if y * 2 + 2 == n { y2 + EPS } else { y2 };
        let d2_pad = if d * 2 + 2 == n { d2 + EPS } else { d2 };

        // Clip X → two halves
        let left = clip_features(&feats, x0 - buf, x1 + buf, 0);
        let right = clip_features(&feats, x1 - buf, x2_pad + buf, 0);

        // Clip Y → four quadrants
        for (x_feats, nx) in [(left, x * 2), (right, x * 2 + 1)] {
            if x_feats.is_empty() {
                continue;
            }
            let bottom = clip_features(&x_feats, y0 - buf, y1 + buf, 1);
            let top = clip_features(&x_feats, y1 - buf, y2_pad + buf, 1);

            // Clip Z → eight octants
            for (y_feats, ny) in [(bottom, y * 2), (top, y * 2 + 1)] {
                if y_feats.is_empty() {
                    continue;
                }
                let front = clip_features(&y_feats, d0 - buf, d1 + buf, 2);
                let back = clip_features(&y_feats, d1 - buf, d2_pad + buf, 2);

                for (z_feats, nd) in [(front, d * 2), (back, d * 2 + 1)] {
                    if !z_feats.is_empty() {
                        stack.push((nz, nx, ny, nd, z_feats));
                    }
                }
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Tile encoding from fragments (pure Rust, no GIL)
// ---------------------------------------------------------------------------

/// Encode a single tag value to protobuf bytes.
fn encode_tag_value(val: &TagValue) -> Vec<u8> {
    let mut buf = Vec::new();
    match val {
        TagValue::Bool(b) => {
            encoder_pbf3::write_varint_field(&mut buf, encoder_pbf3::VALUE_BOOL, if *b { 1 } else { 0 });
        }
        TagValue::Str(s) => {
            encoder_pbf3::write_bytes_field(&mut buf, encoder_pbf3::VALUE_STRING, s.as_bytes());
        }
        TagValue::Float(d) => {
            encoder_pbf3::write_tag(&mut buf, encoder_pbf3::VALUE_DOUBLE, 1); // wire type 1 = 64-bit
            buf.extend_from_slice(&d.to_le_bytes());
        }
        TagValue::Int(i) => {
            if *i < 0 {
                encoder_pbf3::write_tag(&mut buf, encoder_pbf3::VALUE_SINT, 0);
                encoder_pbf3::encode_varint(&mut buf, encoder_pbf3::zigzag(*i));
            } else {
                encoder_pbf3::write_varint_field(&mut buf, encoder_pbf3::VALUE_UINT, *i as u64);
            }
        }
    }
    buf
}

/// Transform fragment geometry to tile-local integers and encode as protobuf tile.
fn encode_tile_from_fragments(
    frags: &[Fragment3D],
    tags_registry: &HashMap<u32, Vec<(String, TagValue)>>,
    tz: u32, tx: u32, ty: u32, td: u32,
    extent: u32, extent_z: u32,
    layer_name: &str,
) -> Vec<u8> {
    let n = 1u32 << tz;
    let x0 = tx as f64 / n as f64;
    let y0 = ty as f64 / n as f64;
    let z0 = td as f64 / n as f64;
    let scale_x = n as f64;
    let scale_y = n as f64;
    let scale_z = n as f64;

    // Dictionary-encoded tags (shared across features in this tile)
    let mut key_indices: AHashMap<String, u32> = AHashMap::new();
    let mut value_indices: AHashMap<String, u32> = AHashMap::new();
    let mut keys_list: Vec<String> = Vec::new();
    let mut values_encoded: Vec<Vec<u8>> = Vec::new();

    let mut features_encoded: Vec<Vec<u8>> = Vec::new();

    for (feat_idx, frag) in frags.iter().enumerate() {
        let mut feat_buf = Vec::new();

        // id
        encoder_pbf3::write_varint_field(&mut feat_buf, encoder_pbf3::FEAT_ID, feat_idx as u64);

        // type
        encoder_pbf3::write_varint_field(&mut feat_buf, encoder_pbf3::FEAT_TYPE, frag.geom_type as u64);

        // tags (dictionary-encoded from registry)
        if let Some(tags) = tags_registry.get(&frag.feature_id) {
            let mut tag_indices: Vec<u32> = Vec::new();
            for (key, val) in tags {
                // Key index
                let ki = match key_indices.get(key) {
                    Some(&idx) => idx,
                    None => {
                        let idx = key_indices.len() as u32;
                        key_indices.insert(key.clone(), idx);
                        keys_list.push(key.clone());
                        idx
                    }
                };
                tag_indices.push(ki);

                // Value index (use repr for dedup)
                let v_repr = match val {
                    TagValue::Str(s) => format!("'{}'", s),
                    TagValue::Int(i) => i.to_string(),
                    TagValue::Float(f) => format!("{:?}", f),
                    TagValue::Bool(b) => b.to_string(),
                };
                let vi = match value_indices.get(&v_repr) {
                    Some(&idx) => idx,
                    None => {
                        let idx = value_indices.len() as u32;
                        value_indices.insert(v_repr, idx);
                        values_encoded.push(encode_tag_value(val));
                        idx
                    }
                };
                tag_indices.push(vi);
            }
            encoder_pbf3::encode_packed_uint32(&mut feat_buf, encoder_pbf3::FEAT_TAGS, &tag_indices);
        }

        // Transform geometry to tile-local integers
        let nv = frag.z.len();
        let mut new_xy: Vec<i64> = Vec::with_capacity(nv * 2);
        let mut new_z: Vec<i64> = Vec::with_capacity(nv);

        for i in 0..nv {
            let lx = (frag.xy[i * 2] as f64 - x0) * scale_x;
            let ly = (frag.xy[i * 2 + 1] as f64 - y0) * scale_y;
            let lz = (frag.z[i] as f64 - z0) * scale_z;
            new_xy.push(tile_transform::round_half_to_even(lx * extent as f64));
            new_xy.push(tile_transform::round_half_to_even(ly * extent as f64));
            new_z.push(tile_transform::round_half_to_even(lz * extent_z as f64));
        }

        // Encode geometry by type
        let gt = frag.geom_type;
        if gt == 1 {
            // POINT3D
            let geom = encoder_pbf3::encode_point_geometry(&new_xy);
            encoder_pbf3::encode_packed_uint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY, &geom);
            let z_enc = encoder_pbf3::encode_z(&new_z);
            encoder_pbf3::encode_packed_sint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY_Z, &z_enc);
        } else if gt == 2 {
            // LINESTRING3D
            let geom = encoder_pbf3::encode_line_geometry(&new_xy);
            encoder_pbf3::encode_packed_uint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY, &geom);
            let z_enc = encoder_pbf3::encode_z(&new_z);
            encoder_pbf3::encode_packed_sint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY_Z, &z_enc);
        } else if gt == 4 || gt == 5 {
            // TIN / POLYHEDRALSURFACE — indexed mesh
            let rls: Vec<usize> = if frag.ring_lengths.is_empty() {
                vec![new_xy.len() / 2]
            } else {
                frag.ring_lengths.iter().map(|&r| r as usize).collect()
            };

            let mut vertex_map: AHashMap<(i64, i64, i64), u32> = AHashMap::new();
            let mut positions: Vec<f32> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();
            let mut offset = 0usize;

            for &rl in &rls {
                let nv = if rl >= 4 { 3 } else { rl };
                if nv == 3 {
                    let mut tri = [0u32; 3];
                    for vi_off in 0..3 {
                        let vi = offset + vi_off;
                        let x = new_xy[vi * 2];
                        let y = new_xy[vi * 2 + 1];
                        let zc = if vi < new_z.len() { new_z[vi] } else { 0 };
                        let key = (x, y, zc);
                        let idx = match vertex_map.get(&key) {
                            Some(&idx) => idx,
                            None => {
                                let idx = vertex_map.len() as u32;
                                vertex_map.insert(key, idx);
                                positions.push(x as f32);
                                positions.push(y as f32);
                                positions.push(zc as f32);
                                idx
                            }
                        };
                        tri[vi_off] = idx;
                    }
                    indices.extend_from_slice(&tri);
                }
                offset += rl;
            }

            let pos_bytes: Vec<u8> = positions.iter().flat_map(|f| f.to_le_bytes()).collect();
            let idx_bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
            encoder_pbf3::write_bytes_field(&mut feat_buf, encoder_pbf3::FEAT_MESH_POSITIONS, &pos_bytes);
            encoder_pbf3::write_bytes_field(&mut feat_buf, encoder_pbf3::FEAT_MESH_INDICES, &idx_bytes);
        } else if gt == 3 {
            // POLYGON3D
            let rls: Vec<usize> = if frag.ring_lengths.is_empty() {
                vec![new_xy.len() / 2]
            } else {
                frag.ring_lengths.iter().map(|&r| r as usize).collect()
            };
            let geom = encoder_pbf3::encode_polygon_geometry(&new_xy, &rls);
            encoder_pbf3::encode_packed_uint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY, &geom);
            let z_enc = encoder_pbf3::encode_z(&new_z);
            encoder_pbf3::encode_packed_sint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY_Z, &z_enc);
        } else {
            // Default: line encoding
            let geom = encoder_pbf3::encode_line_geometry(&new_xy);
            encoder_pbf3::encode_packed_uint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY, &geom);
            let z_enc = encoder_pbf3::encode_z(&new_z);
            encoder_pbf3::encode_packed_sint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY_Z, &z_enc);
        }

        features_encoded.push(feat_buf);
    }

    // --- Encode layer ---
    let mut layer_buf = Vec::new();
    encoder_pbf3::write_varint_field(&mut layer_buf, encoder_pbf3::LAYER_VERSION, 3);
    encoder_pbf3::write_bytes_field(&mut layer_buf, encoder_pbf3::LAYER_NAME, layer_name.as_bytes());
    for feat_buf in &features_encoded {
        encoder_pbf3::write_bytes_field(&mut layer_buf, encoder_pbf3::LAYER_FEATURES, feat_buf);
    }
    for key in &keys_list {
        encoder_pbf3::write_bytes_field(&mut layer_buf, encoder_pbf3::LAYER_KEYS, key.as_bytes());
    }
    for val_buf in &values_encoded {
        encoder_pbf3::write_bytes_field(&mut layer_buf, encoder_pbf3::LAYER_VALUES, val_buf);
    }
    encoder_pbf3::write_varint_field(&mut layer_buf, encoder_pbf3::LAYER_EXTENT, extent as u64);
    encoder_pbf3::write_varint_field(&mut layer_buf, encoder_pbf3::LAYER_EXTENT_Z, extent_z as u64);

    // --- Encode tile ---
    let mut tile_buf = Vec::new();
    encoder_pbf3::write_bytes_field(&mut tile_buf, encoder_pbf3::TILE_LAYERS, &layer_buf);
    tile_buf
}

// ---------------------------------------------------------------------------
// Per-feature PBF3 encoding (world coordinates, single feature per tile)
// ---------------------------------------------------------------------------

/// Encode a single feature's max-zoom fragments as one PBF3 tile in world coordinates.
///
/// Unlike tile-centric PBF3 (which uses tile-local integer coordinates), this
/// produces a feature-centric PBF3 where mesh positions are float32 world
/// coordinates. Non-mesh geometry (point, line, polygon) is encoded with
/// MVT commands in a bbox-local integer space.
///
/// Returns `(pbf3_bytes, feature_bbox)` where feature_bbox is [xmin,ymin,zmin,xmax,ymax,zmax].
fn encode_feature_pbf3(
    frags: &[&Fragment3D],
    tags: Option<&Vec<(String, TagValue)>>,
    world_bounds: &(f64, f64, f64, f64, f64, f64),
    layer_name: &str,
    extent: u32,
    extent_z: u32,
) -> (Vec<u8>, [f64; 6]) {
    let (xmin, ymin, zmin, xmax, ymax, zmax) = *world_bounds;
    let dx = if xmax != xmin { xmax - xmin } else { 1.0 };
    let dy = if ymax != ymin { ymax - ymin } else { 1.0 };
    let dz = if zmax != zmin { zmax - zmin } else { 1.0 };

    // Track feature bounding box in world coords
    let mut bb_min = [f64::INFINITY; 3];
    let mut bb_max = [f64::NEG_INFINITY; 3];

    // Dictionary-encoded tags
    let mut key_indices: AHashMap<String, u32> = AHashMap::new();
    let mut value_indices: AHashMap<String, u32> = AHashMap::new();
    let mut keys_list: Vec<String> = Vec::new();
    let mut values_encoded: Vec<Vec<u8>> = Vec::new();

    // Separate fragments by geometry type
    let mut mesh_frags: Vec<&Fragment3D> = Vec::new();
    let mut other_frags: Vec<&Fragment3D> = Vec::new();

    for frag in frags {
        match frag.geom_type {
            4 | 5 => mesh_frags.push(frag),
            _ => other_frags.push(frag),
        }
    }

    let mut features_encoded: Vec<Vec<u8>> = Vec::new();
    let mut feat_idx = 0u64;

    // --- Encode TIN/PolyhedralSurface as a single merged indexed mesh ---
    if !mesh_frags.is_empty() {
        let mut feat_buf = Vec::new();
        encoder_pbf3::write_varint_field(&mut feat_buf, encoder_pbf3::FEAT_ID, feat_idx);
        feat_idx += 1;

        // Use geom_type from first mesh fragment
        let geom_type = mesh_frags[0].geom_type;
        encoder_pbf3::write_varint_field(&mut feat_buf, encoder_pbf3::FEAT_TYPE, geom_type as u64);

        // Tags
        if let Some(tag_list) = tags {
            let mut tag_indices_vec: Vec<u32> = Vec::new();
            for (key, val) in tag_list {
                let ki = match key_indices.get(key) {
                    Some(&idx) => idx,
                    None => {
                        let idx = key_indices.len() as u32;
                        key_indices.insert(key.clone(), idx);
                        keys_list.push(key.clone());
                        idx
                    }
                };
                tag_indices_vec.push(ki);

                let v_repr = match val {
                    TagValue::Str(s) => format!("'{}'", s),
                    TagValue::Int(i) => i.to_string(),
                    TagValue::Float(f) => format!("{:?}", f),
                    TagValue::Bool(b) => b.to_string(),
                };
                let vi = match value_indices.get(&v_repr) {
                    Some(&idx) => idx,
                    None => {
                        let idx = value_indices.len() as u32;
                        value_indices.insert(v_repr, idx);
                        values_encoded.push(encode_tag_value(val));
                        idx
                    }
                };
                tag_indices_vec.push(vi);
            }
            encoder_pbf3::encode_packed_uint32(&mut feat_buf, encoder_pbf3::FEAT_TAGS, &tag_indices_vec);
        }

        // Merge geometry with vertex dedup (float32 bit-level, like Neuroglancer)
        let mut vertex_map: AHashMap<(i64, i64, i64), u32> = AHashMap::new();
        let mut positions: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for frag in &mesh_frags {
            let n_verts = frag.z.len();
            let ring_lengths: Vec<usize> = if frag.ring_lengths.is_empty() {
                vec![n_verts]
            } else {
                frag.ring_lengths.iter().map(|&r| r as usize).collect()
            };

            let mut offset = 0usize;
            for &rl in &ring_lengths {
                let nv = if rl >= 4 { 3 } else { rl };
                if nv < 3 || offset + 2 >= n_verts {
                    offset += rl;
                    continue;
                }

                let mut tri = [0u32; 3];
                for vi_off in 0..3 {
                    let vi = offset + vi_off;
                    let wx = xmin + frag.xy[vi * 2] as f64 * dx;
                    let wy = ymin + frag.xy[vi * 2 + 1] as f64 * dy;
                    let wz = zmin + frag.z[vi] as f64 * dz;

                    // Update bbox
                    if wx < bb_min[0] { bb_min[0] = wx; }
                    if wy < bb_min[1] { bb_min[1] = wy; }
                    if wz < bb_min[2] { bb_min[2] = wz; }
                    if wx > bb_max[0] { bb_max[0] = wx; }
                    if wy > bb_max[1] { bb_max[1] = wy; }
                    if wz > bb_max[2] { bb_max[2] = wz; }

                    let wx_f32 = wx as f32;
                    let wy_f32 = wy as f32;
                    let wz_f32 = wz as f32;

                    let key = (
                        wx_f32.to_bits() as i64,
                        wy_f32.to_bits() as i64,
                        wz_f32.to_bits() as i64,
                    );

                    let idx = match vertex_map.get(&key) {
                        Some(&idx) => idx,
                        None => {
                            let idx = vertex_map.len() as u32;
                            vertex_map.insert(key, idx);
                            positions.push(wx_f32);
                            positions.push(wy_f32);
                            positions.push(wz_f32);
                            idx
                        }
                    };
                    tri[vi_off] = idx;
                }

                // Skip degenerate triangles
                if is_valid_triangle(&positions, &tri, f32::MAX) {
                    indices.extend_from_slice(&tri);
                }

                offset += rl;
            }
        }

        if !positions.is_empty() && !indices.is_empty() {
            let pos_bytes: Vec<u8> = positions.iter().flat_map(|f| f.to_le_bytes()).collect();
            let idx_bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
            encoder_pbf3::write_bytes_field(&mut feat_buf, encoder_pbf3::FEAT_MESH_POSITIONS, &pos_bytes);
            encoder_pbf3::write_bytes_field(&mut feat_buf, encoder_pbf3::FEAT_MESH_INDICES, &idx_bytes);
        }

        features_encoded.push(feat_buf);
    }

    // --- Encode non-mesh geometry (point, line, polygon) ---
    for frag in &other_frags {
        let n_verts = frag.z.len();
        if n_verts == 0 {
            continue;
        }

        // Compute world coordinates and update bbox
        for i in 0..n_verts {
            let wx = xmin + frag.xy[i * 2] as f64 * dx;
            let wy = ymin + frag.xy[i * 2 + 1] as f64 * dy;
            let wz = zmin + frag.z[i] as f64 * dz;
            if wx < bb_min[0] { bb_min[0] = wx; }
            if wy < bb_min[1] { bb_min[1] = wy; }
            if wz < bb_min[2] { bb_min[2] = wz; }
            if wx > bb_max[0] { bb_max[0] = wx; }
            if wy > bb_max[1] { bb_max[1] = wy; }
            if wz > bb_max[2] { bb_max[2] = wz; }
        }

        // Encode in bbox-local integer space using extent
        let feat_xmin = bb_min[0];
        let feat_ymin = bb_min[1];
        let feat_zmin = bb_min[2];
        let feat_dx = if bb_max[0] != bb_min[0] { bb_max[0] - bb_min[0] } else { 1.0 };
        let feat_dy = if bb_max[1] != bb_min[1] { bb_max[1] - bb_min[1] } else { 1.0 };
        let feat_dz = if bb_max[2] != bb_min[2] { bb_max[2] - bb_min[2] } else { 1.0 };

        let mut new_xy: Vec<i64> = Vec::with_capacity(n_verts * 2);
        let mut new_z: Vec<i64> = Vec::with_capacity(n_verts);

        for i in 0..n_verts {
            let wx = xmin + frag.xy[i * 2] as f64 * dx;
            let wy = ymin + frag.xy[i * 2 + 1] as f64 * dy;
            let wz = zmin + frag.z[i] as f64 * dz;
            let lx = (wx - feat_xmin) / feat_dx;
            let ly = (wy - feat_ymin) / feat_dy;
            let lz = (wz - feat_zmin) / feat_dz;
            new_xy.push(tile_transform::round_half_to_even(lx * extent as f64));
            new_xy.push(tile_transform::round_half_to_even(ly * extent as f64));
            new_z.push(tile_transform::round_half_to_even(lz * extent_z as f64));
        }

        let mut feat_buf = Vec::new();
        encoder_pbf3::write_varint_field(&mut feat_buf, encoder_pbf3::FEAT_ID, feat_idx);
        feat_idx += 1;
        encoder_pbf3::write_varint_field(&mut feat_buf, encoder_pbf3::FEAT_TYPE, frag.geom_type as u64);

        // Tags (only on first non-mesh feature to avoid duplication)
        if features_encoded.is_empty() {
            if let Some(tag_list) = tags {
                let mut tag_indices_vec: Vec<u32> = Vec::new();
                for (key, val) in tag_list {
                    let ki = match key_indices.get(key) {
                        Some(&idx) => idx,
                        None => {
                            let idx = key_indices.len() as u32;
                            key_indices.insert(key.clone(), idx);
                            keys_list.push(key.clone());
                            idx
                        }
                    };
                    tag_indices_vec.push(ki);

                    let v_repr = match val {
                        TagValue::Str(s) => format!("'{}'", s),
                        TagValue::Int(i) => i.to_string(),
                        TagValue::Float(f) => format!("{:?}", f),
                        TagValue::Bool(b) => b.to_string(),
                    };
                    let vi = match value_indices.get(&v_repr) {
                        Some(&idx) => idx,
                        None => {
                            let idx = value_indices.len() as u32;
                            value_indices.insert(v_repr, idx);
                            values_encoded.push(encode_tag_value(val));
                            idx
                        }
                    };
                    tag_indices_vec.push(vi);
                }
                encoder_pbf3::encode_packed_uint32(&mut feat_buf, encoder_pbf3::FEAT_TAGS, &tag_indices_vec);
            }
        }

        // Encode geometry by type
        let gt = frag.geom_type;
        if gt == 1 {
            let geom = encoder_pbf3::encode_point_geometry(&new_xy);
            encoder_pbf3::encode_packed_uint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY, &geom);
            let z_enc = encoder_pbf3::encode_z(&new_z);
            encoder_pbf3::encode_packed_sint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY_Z, &z_enc);
        } else if gt == 2 {
            let geom = encoder_pbf3::encode_line_geometry(&new_xy);
            encoder_pbf3::encode_packed_uint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY, &geom);
            let z_enc = encoder_pbf3::encode_z(&new_z);
            encoder_pbf3::encode_packed_sint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY_Z, &z_enc);
        } else if gt == 3 {
            let rls: Vec<usize> = if frag.ring_lengths.is_empty() {
                vec![new_xy.len() / 2]
            } else {
                frag.ring_lengths.iter().map(|&r| r as usize).collect()
            };
            let geom = encoder_pbf3::encode_polygon_geometry(&new_xy, &rls);
            encoder_pbf3::encode_packed_uint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY, &geom);
            let z_enc = encoder_pbf3::encode_z(&new_z);
            encoder_pbf3::encode_packed_sint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY_Z, &z_enc);
        } else {
            let geom = encoder_pbf3::encode_line_geometry(&new_xy);
            encoder_pbf3::encode_packed_uint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY, &geom);
            let z_enc = encoder_pbf3::encode_z(&new_z);
            encoder_pbf3::encode_packed_sint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY_Z, &z_enc);
        }

        features_encoded.push(feat_buf);
    }

    // --- Encode layer ---
    let mut layer_buf = Vec::new();
    encoder_pbf3::write_varint_field(&mut layer_buf, encoder_pbf3::LAYER_VERSION, 3);
    encoder_pbf3::write_bytes_field(&mut layer_buf, encoder_pbf3::LAYER_NAME, layer_name.as_bytes());
    for feat_buf in &features_encoded {
        encoder_pbf3::write_bytes_field(&mut layer_buf, encoder_pbf3::LAYER_FEATURES, feat_buf);
    }
    for key in &keys_list {
        encoder_pbf3::write_bytes_field(&mut layer_buf, encoder_pbf3::LAYER_KEYS, key.as_bytes());
    }
    for val_buf in &values_encoded {
        encoder_pbf3::write_bytes_field(&mut layer_buf, encoder_pbf3::LAYER_VALUES, val_buf);
    }
    encoder_pbf3::write_varint_field(&mut layer_buf, encoder_pbf3::LAYER_EXTENT, extent as u64);
    encoder_pbf3::write_varint_field(&mut layer_buf, encoder_pbf3::LAYER_EXTENT_Z, extent_z as u64);

    // --- Encode tile ---
    let mut tile_buf = Vec::new();
    encoder_pbf3::write_bytes_field(&mut tile_buf, encoder_pbf3::TILE_LAYERS, &layer_buf);

    let bbox = [bb_min[0], bb_min[1], bb_min[2], bb_max[0], bb_max[1], bb_max[2]];
    (tile_buf, bbox)
}

// ---------------------------------------------------------------------------
// Multi-LOD per-feature PBF3 encoder
// ---------------------------------------------------------------------------

/// Encode a feature's fragments across all zoom levels into a multi-LOD PBF3 tile.
///
/// Each zoom level becomes a separate Layer named "lod_0" (finest) through "lod_N" (coarsest).
/// - At max_zoom: exact float32 vertex dedup (same as single-LOD encode_feature_pbf3)
/// - At coarser levels: grid-based vertex clustering (cell_size = world_extent / (base_cells * 2^zoom))
///
/// Returns (pbf3_bytes, finest_lod_bbox).
fn encode_feature_pbf3_multilod(
    all_frags: &[&Fragment3D],
    tags: Option<&Vec<(String, TagValue)>>,
    world_bounds: &(f64, f64, f64, f64, f64, f64),
    max_zoom: u32,
    extent: u32,
    extent_z: u32,
    base_cells: u32,
) -> (Vec<u8>, [f64; 6]) {
    let (xmin, ymin, zmin, xmax, ymax, zmax) = *world_bounds;
    let dx = if xmax != xmin { xmax - xmin } else { 1.0 };
    let dy = if ymax != ymin { ymax - ymin } else { 1.0 };
    let dz = if zmax != zmin { zmax - zmin } else { 1.0 };

    // Group fragments by tile_z (zoom level)
    let mut by_zoom: std::collections::BTreeMap<u32, Vec<&Fragment3D>> = std::collections::BTreeMap::new();
    for frag in all_frags {
        by_zoom.entry(frag.tile_z).or_default().push(frag);
    }

    // Track finest-LOD bounding box (from max_zoom fragments)
    let mut bb_min = [f64::INFINITY; 3];
    let mut bb_max = [f64::NEG_INFINITY; 3];

    let mut layer_bufs: Vec<Vec<u8>> = Vec::new();

    // Process each zoom level: max_zoom → 0 produces lod_0 → lod_N
    for (&zoom, zoom_frags) in by_zoom.iter().rev() {
        let lod_index = max_zoom.saturating_sub(zoom);
        let layer_name = format!("lod_{}", lod_index);
        let is_finest = zoom == max_zoom;
        let do_simplify = zoom < max_zoom;

        // Separate mesh vs non-mesh fragments
        let mut mesh_frags: Vec<&Fragment3D> = Vec::new();
        let mut other_frags: Vec<&Fragment3D> = Vec::new();
        for frag in zoom_frags {
            match frag.geom_type {
                4 | 5 => mesh_frags.push(frag),
                _ => other_frags.push(frag),
            }
        }

        // Dictionary-encoded tags (per-layer)
        let mut key_indices: AHashMap<String, u32> = AHashMap::new();
        let mut value_indices: AHashMap<String, u32> = AHashMap::new();
        let mut keys_list: Vec<String> = Vec::new();
        let mut values_encoded: Vec<Vec<u8>> = Vec::new();
        let mut features_encoded: Vec<Vec<u8>> = Vec::new();
        let mut feat_idx = 0u64;

        // --- Encode TIN/PolyhedralSurface ---
        if !mesh_frags.is_empty() {
            let mut feat_buf = Vec::new();
            encoder_pbf3::write_varint_field(&mut feat_buf, encoder_pbf3::FEAT_ID, feat_idx);
            feat_idx += 1;
            let geom_type = mesh_frags[0].geom_type;
            encoder_pbf3::write_varint_field(&mut feat_buf, encoder_pbf3::FEAT_TYPE, geom_type as u64);

            // Tags (only on first feature)
            if let Some(tag_list) = tags {
                let mut tag_indices_vec: Vec<u32> = Vec::new();
                for (key, val) in tag_list {
                    let ki = *key_indices.entry(key.clone()).or_insert_with(|| {
                        let idx = keys_list.len() as u32;
                        keys_list.push(key.clone());
                        idx
                    });
                    tag_indices_vec.push(ki);

                    let v_repr = match val {
                        TagValue::Str(s) => format!("'{}'", s),
                        TagValue::Int(i) => i.to_string(),
                        TagValue::Float(f) => format!("{:?}", f),
                        TagValue::Bool(b) => b.to_string(),
                    };
                    let vi = *value_indices.entry(v_repr).or_insert_with(|| {
                        let idx = values_encoded.len() as u32;
                        values_encoded.push(encode_tag_value(val));
                        idx
                    });
                    tag_indices_vec.push(vi);
                }
                encoder_pbf3::encode_packed_uint32(&mut feat_buf, encoder_pbf3::FEAT_TAGS, &tag_indices_vec);
            }

            // Build indexed mesh with exact f32 vertex dedup (all zoom levels)
            let mut positions: Vec<f32> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();
            {
                let mut vertex_map: AHashMap<(i64, i64, i64), u32> = AHashMap::new();

                for frag in &mesh_frags {
                    let n_verts = frag.z.len();
                    let ring_lengths: Vec<usize> = if frag.ring_lengths.is_empty() {
                        vec![n_verts]
                    } else {
                        frag.ring_lengths.iter().map(|&r| r as usize).collect()
                    };

                    let mut offset = 0usize;
                    for &rl in &ring_lengths {
                        let nv = if rl >= 4 { 3 } else { rl };
                        if nv < 3 || offset + 2 >= n_verts {
                            offset += rl;
                            continue;
                        }

                        let mut tri = [0u32; 3];
                        for vi_off in 0..3 {
                            let vi = offset + vi_off;
                            let wx_f32 = (xmin + frag.xy[vi * 2] as f64 * dx) as f32;
                            let wy_f32 = (ymin + frag.xy[vi * 2 + 1] as f64 * dy) as f32;
                            let wz_f32 = (zmin + frag.z[vi] as f64 * dz) as f32;

                            let key = (
                                wx_f32.to_bits() as i64,
                                wy_f32.to_bits() as i64,
                                wz_f32.to_bits() as i64,
                            );

                            let idx = match vertex_map.get(&key) {
                                Some(&idx) => idx,
                                None => {
                                    let idx = vertex_map.len() as u32;
                                    vertex_map.insert(key, idx);
                                    positions.push(wx_f32);
                                    positions.push(wy_f32);
                                    positions.push(wz_f32);
                                    idx
                                }
                            };
                            tri[vi_off] = idx;
                        }

                        if is_valid_triangle(&positions, &tri, f32::MAX) {
                            indices.extend_from_slice(&tri);
                        }
                        offset += rl;
                    }
                }
            }

            // QEM simplification for coarser LODs
            if do_simplify && indices.len() > 12 {
                let target_idx = simplify::compute_target_index_count(
                    base_cells, zoom, max_zoom, indices.len(),
                );
                let target_tris = target_idx / 3;
                let (sp, mut si) = simplify::simplify_mesh(&positions, &indices, target_tris);
                filter_oversized_triangles(&sp, &mut si, tile_max_edge_sq_world(base_cells, zoom, dx, dy, dz));
                positions = sp;
                indices = si;
            }

            // Update finest-LOD bbox from positions
            if is_finest {
                let n_verts = positions.len() / 3;
                for i in 0..n_verts {
                    let wx = positions[i * 3] as f64;
                    let wy = positions[i * 3 + 1] as f64;
                    let wz = positions[i * 3 + 2] as f64;
                    if wx < bb_min[0] { bb_min[0] = wx; }
                    if wy < bb_min[1] { bb_min[1] = wy; }
                    if wz < bb_min[2] { bb_min[2] = wz; }
                    if wx > bb_max[0] { bb_max[0] = wx; }
                    if wy > bb_max[1] { bb_max[1] = wy; }
                    if wz > bb_max[2] { bb_max[2] = wz; }
                }
            }

            if !positions.is_empty() && !indices.is_empty() {
                let pos_bytes: Vec<u8> = positions.iter().flat_map(|f| f.to_le_bytes()).collect();
                let idx_bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
                encoder_pbf3::write_bytes_field(&mut feat_buf, encoder_pbf3::FEAT_MESH_POSITIONS, &pos_bytes);
                encoder_pbf3::write_bytes_field(&mut feat_buf, encoder_pbf3::FEAT_MESH_INDICES, &idx_bytes);
            }

            features_encoded.push(feat_buf);
        }

        // --- Encode non-mesh geometry (point, line, polygon) ---
        for frag in &other_frags {
            let n_verts = frag.z.len();
            if n_verts == 0 { continue; }

            // Compute world coordinates for bbox
            let mut local_bb_min = [f64::INFINITY; 3];
            let mut local_bb_max = [f64::NEG_INFINITY; 3];
            for i in 0..n_verts {
                let wx = xmin + frag.xy[i * 2] as f64 * dx;
                let wy = ymin + frag.xy[i * 2 + 1] as f64 * dy;
                let wz = zmin + frag.z[i] as f64 * dz;
                if wx < local_bb_min[0] { local_bb_min[0] = wx; }
                if wy < local_bb_min[1] { local_bb_min[1] = wy; }
                if wz < local_bb_min[2] { local_bb_min[2] = wz; }
                if wx > local_bb_max[0] { local_bb_max[0] = wx; }
                if wy > local_bb_max[1] { local_bb_max[1] = wy; }
                if wz > local_bb_max[2] { local_bb_max[2] = wz; }
            }

            if is_finest {
                for d in 0..3 {
                    if local_bb_min[d] < bb_min[d] { bb_min[d] = local_bb_min[d]; }
                    if local_bb_max[d] > bb_max[d] { bb_max[d] = local_bb_max[d]; }
                }
            }

            // Encode in bbox-local integer space
            let feat_dx = if local_bb_max[0] != local_bb_min[0] { local_bb_max[0] - local_bb_min[0] } else { 1.0 };
            let feat_dy = if local_bb_max[1] != local_bb_min[1] { local_bb_max[1] - local_bb_min[1] } else { 1.0 };
            let feat_dz = if local_bb_max[2] != local_bb_min[2] { local_bb_max[2] - local_bb_min[2] } else { 1.0 };

            let mut new_xy: Vec<i64> = Vec::with_capacity(n_verts * 2);
            let mut new_z: Vec<i64> = Vec::with_capacity(n_verts);
            for i in 0..n_verts {
                let wx = xmin + frag.xy[i * 2] as f64 * dx;
                let wy = ymin + frag.xy[i * 2 + 1] as f64 * dy;
                let wz = zmin + frag.z[i] as f64 * dz;
                let lx = (wx - local_bb_min[0]) / feat_dx;
                let ly = (wy - local_bb_min[1]) / feat_dy;
                let lz = (wz - local_bb_min[2]) / feat_dz;
                new_xy.push(tile_transform::round_half_to_even(lx * extent as f64));
                new_xy.push(tile_transform::round_half_to_even(ly * extent as f64));
                new_z.push(tile_transform::round_half_to_even(lz * extent_z as f64));
            }

            let mut feat_buf = Vec::new();
            encoder_pbf3::write_varint_field(&mut feat_buf, encoder_pbf3::FEAT_ID, feat_idx);
            feat_idx += 1;
            encoder_pbf3::write_varint_field(&mut feat_buf, encoder_pbf3::FEAT_TYPE, frag.geom_type as u64);

            // Tags only on first non-mesh feature if no mesh features exist
            if features_encoded.is_empty() {
                if let Some(tag_list) = tags {
                    let mut tag_indices_vec: Vec<u32> = Vec::new();
                    for (key, val) in tag_list {
                        let ki = *key_indices.entry(key.clone()).or_insert_with(|| {
                            let idx = keys_list.len() as u32;
                            keys_list.push(key.clone());
                            idx
                        });
                        tag_indices_vec.push(ki);

                        let v_repr = match val {
                            TagValue::Str(s) => format!("'{}'", s),
                            TagValue::Int(i) => i.to_string(),
                            TagValue::Float(f) => format!("{:?}", f),
                            TagValue::Bool(b) => b.to_string(),
                        };
                        let vi = *value_indices.entry(v_repr).or_insert_with(|| {
                            let idx = values_encoded.len() as u32;
                            values_encoded.push(encode_tag_value(val));
                            idx
                        });
                        tag_indices_vec.push(vi);
                    }
                    encoder_pbf3::encode_packed_uint32(&mut feat_buf, encoder_pbf3::FEAT_TAGS, &tag_indices_vec);
                }
            }

            let gt = frag.geom_type;
            if gt == 1 {
                let geom = encoder_pbf3::encode_point_geometry(&new_xy);
                encoder_pbf3::encode_packed_uint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY, &geom);
                let z_enc = encoder_pbf3::encode_z(&new_z);
                encoder_pbf3::encode_packed_sint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY_Z, &z_enc);
            } else if gt == 2 {
                let geom = encoder_pbf3::encode_line_geometry(&new_xy);
                encoder_pbf3::encode_packed_uint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY, &geom);
                let z_enc = encoder_pbf3::encode_z(&new_z);
                encoder_pbf3::encode_packed_sint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY_Z, &z_enc);
            } else if gt == 3 {
                let rls: Vec<usize> = if frag.ring_lengths.is_empty() {
                    vec![new_xy.len() / 2]
                } else {
                    frag.ring_lengths.iter().map(|&r| r as usize).collect()
                };
                let geom = encoder_pbf3::encode_polygon_geometry(&new_xy, &rls);
                encoder_pbf3::encode_packed_uint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY, &geom);
                let z_enc = encoder_pbf3::encode_z(&new_z);
                encoder_pbf3::encode_packed_sint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY_Z, &z_enc);
            } else {
                let geom = encoder_pbf3::encode_line_geometry(&new_xy);
                encoder_pbf3::encode_packed_uint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY, &geom);
                let z_enc = encoder_pbf3::encode_z(&new_z);
                encoder_pbf3::encode_packed_sint32(&mut feat_buf, encoder_pbf3::FEAT_GEOMETRY_Z, &z_enc);
            }

            features_encoded.push(feat_buf);
        }

        // --- Encode layer ---
        let mut layer_buf = Vec::new();
        encoder_pbf3::write_varint_field(&mut layer_buf, encoder_pbf3::LAYER_VERSION, 3);
        encoder_pbf3::write_bytes_field(&mut layer_buf, encoder_pbf3::LAYER_NAME, layer_name.as_bytes());
        for feat_buf in &features_encoded {
            encoder_pbf3::write_bytes_field(&mut layer_buf, encoder_pbf3::LAYER_FEATURES, feat_buf);
        }
        for key in &keys_list {
            encoder_pbf3::write_bytes_field(&mut layer_buf, encoder_pbf3::LAYER_KEYS, key.as_bytes());
        }
        for val_buf in &values_encoded {
            encoder_pbf3::write_bytes_field(&mut layer_buf, encoder_pbf3::LAYER_VALUES, val_buf);
        }
        encoder_pbf3::write_varint_field(&mut layer_buf, encoder_pbf3::LAYER_EXTENT, extent as u64);
        encoder_pbf3::write_varint_field(&mut layer_buf, encoder_pbf3::LAYER_EXTENT_Z, extent_z as u64);

        layer_bufs.push(layer_buf);
    }

    // --- Assemble tile with all layers ---
    let mut tile_buf = Vec::new();
    for layer_buf in &layer_bufs {
        encoder_pbf3::write_bytes_field(&mut tile_buf, encoder_pbf3::TILE_LAYERS, layer_buf);
    }

    let bbox = [bb_min[0], bb_min[1], bb_min[2], bb_max[0], bb_max[1], bb_max[2]];
    (tile_buf, bbox)
}

// ---------------------------------------------------------------------------
// GLB tile encoding from fragments (pure Rust, no GIL)
// ---------------------------------------------------------------------------

/// Default base grid resolution at zoom 0 (matches simplify.rs).
const DEFAULT_BASE_CELLS: u32 = 10;

/// Encode a single tile's fragments as GLB binary with vertex-clustering simplification.
///
/// For TIN/PolyhedralSurface at non-leaf zoom levels, vertices are snapped to a
/// world-aligned grid (BASE_CELLS * 2^zoom cells per axis) to produce LOD.
/// At max_zoom, exact vertex dedup is used instead.
/// Convert a single fragment into a GlbFeature (vertex dedup + QEM simplification).
///
/// Pure function with no shared state — safe to call from `par_iter`.
fn fragment_to_glb_feature(
    frag: &Fragment3D,
    tags_registry: &HashMap<u32, Vec<(String, TagValue)>>,
    xmin: f64, ymin: f64, zmin: f64,
    dx: f64, dy: f64, dz: f64,
    do_simplify: bool,
    max_zoom: u32,
    base_cells: u32,
    tz: u32,
) -> Option<GlbFeature> {
    let n_verts = frag.z.len();
    if n_verts == 0 {
        return None;
    }

    // Build tags as JSON extras
    let extras = tags_registry.get(&frag.feature_id).map(|tags| {
        let obj: serde_json::Map<String, serde_json::Value> = tags
            .iter()
            .map(|(k, v)| {
                let jv = match v {
                    TagValue::Str(s) => serde_json::json!(s),
                    TagValue::Int(i) => serde_json::json!(i),
                    TagValue::Float(f) => serde_json::json!(f),
                    TagValue::Bool(b) => serde_json::json!(b),
                };
                (k.clone(), jv)
            })
            .collect();
        serde_json::Value::Object(obj)
    });

    match frag.geom_type {
        5 | 4 => {
            // TIN or PolyhedralSurface — build indexed mesh with f32 dedup
            let rls: Vec<usize> = if frag.ring_lengths.is_empty() {
                vec![n_verts]
            } else {
                frag.ring_lengths.iter().map(|&r| r as usize).collect()
            };
            let n_faces = rls.len();
            let do_simplify_this = do_simplify && n_faces > 4;

            // Build indexed mesh with exact f32 vertex dedup
            let mut positions: Vec<f32> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();
            {
                let mut vertex_map: AHashMap<(u32, u32, u32), u32> = AHashMap::new();
                let mut offset = 0usize;
                for &rl in &rls {
                    let nv = if rl >= 4 { 3 } else { rl };
                    if nv == 3 && offset + 2 < n_verts {
                        let mut tri = [0u32; 3];
                        for vi_off in 0..3 {
                            let vi = offset + vi_off;
                            let wx = (xmin + frag.xy[vi * 2] as f64 * dx) as f32;
                            let wy = (ymin + frag.xy[vi * 2 + 1] as f64 * dy) as f32;
                            let wz = (zmin + frag.z[vi] as f64 * dz) as f32;
                            let key = (wx.to_bits(), wy.to_bits(), wz.to_bits());
                            let idx = match vertex_map.get(&key) {
                                Some(&idx) => idx,
                                None => {
                                    let idx = (positions.len() / 3) as u32;
                                    vertex_map.insert(key, idx);
                                    positions.push(wx);
                                    positions.push(wy);
                                    positions.push(wz);
                                    idx
                                }
                            };
                            tri[vi_off] = idx;
                        }
                        if is_valid_triangle(&positions, &tri, f32::MAX) {
                            indices.extend_from_slice(&tri);
                        }
                    }
                    offset += rl;
                }
            }

            // QEM simplification for coarser LODs
            if do_simplify_this && indices.len() > 12 {
                let target_idx = simplify::compute_target_index_count(
                    base_cells, tz, max_zoom, indices.len(),
                );
                let target_tris = target_idx / 3;
                let (sp, mut si) = simplify::simplify_mesh(&positions, &indices, target_tris);
                filter_oversized_triangles(&sp, &mut si, tile_max_edge_sq_world(base_cells, tz, dx, dy, dz));
                positions = sp;
                indices = si;
            }

            if !positions.is_empty() && !indices.is_empty() {
                Some(GlbFeature {
                    positions,
                    indices,
                    mode: encoder_glb::MODE_TRIANGLES,
                    extras,
                })
            } else {
                None
            }
        }
        2 => {
            // LineString — line segments
            let mut positions: Vec<f32> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();

            for i in 0..n_verts {
                let wx = (xmin + frag.xy[i * 2] as f64 * dx) as f32;
                let wy = (ymin + frag.xy[i * 2 + 1] as f64 * dy) as f32;
                let wz = (zmin + frag.z[i] as f64 * dz) as f32;
                positions.push(wx);
                positions.push(wy);
                positions.push(wz);
            }

            for i in 0..n_verts.saturating_sub(1) {
                indices.push(i as u32);
                indices.push((i + 1) as u32);
            }

            if !positions.is_empty() && !indices.is_empty() {
                Some(GlbFeature {
                    positions,
                    indices,
                    mode: encoder_glb::MODE_LINES,
                    extras,
                })
            } else {
                None
            }
        }
        1 => {
            // Point
            let mut positions: Vec<f32> = Vec::new();
            for i in 0..n_verts {
                let wx = (xmin + frag.xy[i * 2] as f64 * dx) as f32;
                let wy = (ymin + frag.xy[i * 2 + 1] as f64 * dy) as f32;
                let wz = (zmin + frag.z[i] as f64 * dz) as f32;
                positions.push(wx);
                positions.push(wy);
                positions.push(wz);
            }

            if !positions.is_empty() {
                Some(GlbFeature {
                    positions,
                    indices: vec![],
                    mode: encoder_glb::MODE_POINTS,
                    extras,
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Bucketed redistribution for streaming tile generation
// ---------------------------------------------------------------------------

/// Redistribute fragments from ingestion shards into bucket files keyed by
/// tile hash. This allows processing one bucket at a time instead of loading
/// all fragments into memory.
///
/// Parallelized: each ingestion shard is read on a separate rayon thread,
/// writing to per-shard bucket files. The final bucket for each index is
/// the collection of per-shard files for that bucket.
///
/// Returns the paths of the merged bucket files (one per bucket).
/// Simplify a TIN/PolyhedralSurface fragment via QEM for coarse zoom levels.
/// Builds indexed mesh, runs QEM, converts back to ring format.
fn simplify_fragment(frag: Fragment3D, base_cells: u32, max_zoom: u32) -> Fragment3D {
    let n_verts = frag.z.len();
    if n_verts == 0 || (frag.geom_type != 4 && frag.geom_type != 5) {
        return frag;
    }

    let rls: Vec<usize> = if frag.ring_lengths.is_empty() {
        vec![n_verts]
    } else {
        frag.ring_lengths.iter().map(|&r| r as usize).collect()
    };
    let n_faces = rls.len();
    if n_faces <= 4 {
        return frag;
    }

    // Build indexed mesh with f32 vertex dedup (same logic as fragment_to_glb_feature)
    let mut positions: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    {
        let mut vertex_map: AHashMap<(u32, u32, u32), u32> = AHashMap::new();
        let mut offset = 0usize;
        for &rl in &rls {
            let nv = if rl >= 4 { 3 } else { rl };
            if nv == 3 && offset + 2 < n_verts {
                let mut tri = [0u32; 3];
                for vi_off in 0..3 {
                    let vi = offset + vi_off;
                    // Keep in [0,1] normalized coords (not world coords)
                    let px = frag.xy[vi * 2];
                    let py = frag.xy[vi * 2 + 1];
                    let pz = frag.z[vi];
                    let key = (px.to_bits(), py.to_bits(), pz.to_bits());
                    let idx = match vertex_map.get(&key) {
                        Some(&idx) => idx,
                        None => {
                            let idx = (positions.len() / 3) as u32;
                            vertex_map.insert(key, idx);
                            positions.push(px);
                            positions.push(py);
                            positions.push(pz);
                            idx
                        }
                    };
                    tri[vi_off] = idx;
                }
                if is_valid_triangle(&positions, &tri, f32::MAX) {
                    indices.extend_from_slice(&tri);
                }
            }
            offset += rl;
        }
    }

    if indices.len() <= 12 {
        return frag;
    }

    // QEM simplification
    let target_idx = simplify::compute_target_index_count(
        base_cells, frag.tile_z, max_zoom, indices.len(),
    );
    let target_tris = target_idx / 3;
    let (sp, mut si) = simplify::simplify_mesh(&positions, &indices, target_tris);
    filter_oversized_triangles(&sp, &mut si, tile_max_edge_sq_normalized(base_cells, frag.tile_z));

    if si.is_empty() {
        return frag;
    }

    // Convert back to ring format (4 vertices per triangle: v0, v1, v2, v0)
    let n_tris = si.len() / 3;
    let mut new_xy: Vec<f32> = Vec::with_capacity(n_tris * 4 * 2);
    let mut new_z: Vec<f32> = Vec::with_capacity(n_tris * 4);
    let new_ring_lengths: Vec<u32> = vec![4; n_tris];

    for tri_idx in 0..n_tris {
        let i0 = si[tri_idx * 3] as usize;
        let i1 = si[tri_idx * 3 + 1] as usize;
        let i2 = si[tri_idx * 3 + 2] as usize;
        for &vi in &[i0, i1, i2, i0] {
            new_xy.push(sp[vi * 3]);
            new_xy.push(sp[vi * 3 + 1]);
            new_z.push(sp[vi * 3 + 2]);
        }
    }

    Fragment3D {
        feature_id: frag.feature_id,
        tile_z: frag.tile_z,
        tile_x: frag.tile_x,
        tile_y: frag.tile_y,
        tile_d: frag.tile_d,
        geom_type: frag.geom_type,
        xy: new_xy,
        z: new_z,
        ring_lengths: new_ring_lengths,
    }
}

fn redistribute_fragments_to_buckets(
    frag_dir: &Path,
    num_buckets: usize,
    max_zoom: u32,
    base_cells: u32,
) -> io::Result<Vec<PathBuf>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let bucket_dir = frag_dir.join("buckets");
    std::fs::create_dir_all(&bucket_dir)?;

    // Discover ingestion shard files (exclude the buckets subdirectory)
    let mut shard_paths: Vec<PathBuf> = std::fs::read_dir(frag_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "mjf"))
        .collect();
    shard_paths.sort();

    if shard_paths.is_empty() {
        // No fragments — create empty bucket directories
        let mut bucket_dirs = Vec::with_capacity(num_buckets);
        for i in 0..num_buckets {
            let bdir = bucket_dir.join(format!("b_{:04}", i));
            std::fs::create_dir_all(&bdir)?;
            bucket_dirs.push(bdir);
        }
        return Ok(bucket_dirs);
    }

    // Phase 1: Each shard writes to its own set of bucket files in parallel.
    // shard S writes to buckets/shard_{S}_bucket_{B}.mjf
    let n_shards = shard_paths.len();
    let errors: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

    shard_paths.par_iter().enumerate().for_each(|(shard_idx, shard_path)| {
        // Each thread creates its own bucket writers
        let mut writers: Vec<Option<Fragment3DWriter>> = (0..num_buckets).map(|_| None).collect();

        let mut reader = match Fragment3DReader::new(shard_path) {
            Ok(r) => r,
            Err(e) => {
                errors.lock().unwrap().push(format!("shard {}: {}", shard_idx, e));
                return;
            }
        };

        loop {
            match reader.read_next() {
                Ok(Some(frag)) => {
                    // Simplify coarse zoom meshes before writing to bucket
                    let frag = if (frag.geom_type == 4 || frag.geom_type == 5)
                                  && frag.tile_z < max_zoom {
                        simplify_fragment(frag, base_cells, max_zoom)
                    } else {
                        frag
                    };

                    let mut hasher = DefaultHasher::new();
                    (frag.tile_z, frag.tile_x, frag.tile_y, frag.tile_d).hash(&mut hasher);
                    let bucket = (hasher.finish() as usize) % num_buckets;

                    // Lazy-init writer for this bucket
                    if writers[bucket].is_none() {
                        let path = bucket_dir.join(
                            format!("shard_{:03}_bucket_{:04}.mjf", shard_idx, bucket)
                        );
                        match Fragment3DWriter::new(&path) {
                            Ok(w) => writers[bucket] = Some(w),
                            Err(e) => {
                                errors.lock().unwrap().push(format!(
                                    "shard {} bucket {}: {}", shard_idx, bucket, e
                                ));
                                return;
                            }
                        }
                    }

                    if let Some(ref mut w) = writers[bucket] {
                        if let Err(e) = w.write(&frag) {
                            errors.lock().unwrap().push(format!(
                                "shard {} bucket {} write: {}", shard_idx, bucket, e
                            ));
                            return;
                        }
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    errors.lock().unwrap().push(format!("shard {} read: {}", shard_idx, e));
                    return;
                }
            }
        }

        // Flush all writers for this shard
        for w in writers.into_iter().flatten() {
            // Drop handles flush via the Drop impl
            drop(w);
        }
    });

    let errs = errors.into_inner().unwrap();
    if !errs.is_empty() {
        return Err(io::Error::new(io::ErrorKind::Other, errs.join("; ")));
    }

    // Phase 2: For each bucket index, collect all per-shard files into a
    // single Fragment3DReader path list. We don't need to merge them into one
    // file — Fragment3DReader::open_dir reads multiple files sequentially.
    // Create per-bucket subdirectories with the per-shard files.
    let mut bucket_dirs: Vec<PathBuf> = Vec::with_capacity(num_buckets);
    for b in 0..num_buckets {
        let bdir = bucket_dir.join(format!("b_{:04}", b));
        std::fs::create_dir_all(&bdir)?;

        // Move per-shard bucket files into this directory
        for s in 0..n_shards {
            let src = bucket_dir.join(format!("shard_{:03}_bucket_{:04}.mjf", s, b));
            if src.exists() {
                let dst = bdir.join(format!("shard_{:03}.mjf", s));
                std::fs::rename(&src, &dst)?;
            }
        }
        bucket_dirs.push(bdir);
    }

    Ok(bucket_dirs)
}

/// Memory-adaptive pipeline: per-zoom batched read → group → simplify → encode.
///
/// Processes each zoom level separately, with hash-based spatial batching to
/// stay within the given memory budget. This scales from small datasets on
/// laptops (128 GB) to 100+ TB datasets on large machines (768 GB+).
///
/// For each zoom level, estimates how many batches are needed to fit within
/// `max_memory_bytes`, then for each batch reads all fragment files (keeping
/// only matching zoom + batch), groups by tile key, simplifies, and encodes.
///
/// Fragment files are re-read for each (zoom, batch) pass. Sequential I/O and
/// OS page cache make re-reads fast.
///
/// Returns (tile_count, tile_keys).
fn read_group_simplify_encode(
    frag_dir: &Path,
    tags_registry: &HashMap<u32, Vec<(String, TagValue)>>,
    out_dir: &Path,
    xmin: f64, ymin: f64, zmin: f64,
    dx: f64, dy: f64, dz: f64,
    max_zoom: u32,
    base_cells: u32,
    effective_compression: &str,
    max_memory_bytes: usize,
) -> io::Result<(u32, Vec<(u32, u32, u32, u32)>)> {
    use dashmap::DashMap;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Discover fragment files
    let mut all_paths: Vec<PathBuf> = std::fs::read_dir(frag_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "mjf"))
        .collect();
    all_paths.sort();

    if all_paths.is_empty() {
        return Ok((0, Vec::new()));
    }

    // Partition fragment files by zoom level based on naming convention:
    //   "frag_XXXXX.mjf"       → plain: contains ALL zoom levels (small neurons)
    //   "frag_XXXXX_zN.mjf"    → zoom-specific: coarse zoom N only (large neurons)
    //   "frag_XXXXX_cXXXX.mjf" → chunk: max_zoom only (large neurons)
    let mut files_per_zoom: Vec<Vec<PathBuf>> = (0..=max_zoom).map(|_| Vec::new()).collect();
    let mut mixed_files: Vec<PathBuf> = Vec::new();

    for path in &all_paths {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if let Some(pos) = stem.rfind("_z") {
            // Zoom-specific file: frag_XXXXX_zN
            if let Ok(z) = stem[pos + 2..].parse::<u32>() {
                if z <= max_zoom {
                    files_per_zoom[z as usize].push(path.clone());
                }
            } else {
                mixed_files.push(path.clone());
            }
        } else if stem.contains("_c") {
            // Chunk file: frag_XXXXX_cXXXX → max_zoom only
            // Verify it's actually a chunk pattern (ends with digits after _c)
            if let Some(pos) = stem.rfind("_c") {
                if stem[pos + 2..].parse::<u32>().is_ok() {
                    files_per_zoom[max_zoom as usize].push(path.clone());
                } else {
                    mixed_files.push(path.clone());
                }
            }
        } else {
            // Plain file: contains all zoom levels
            mixed_files.push(path.clone());
        }
    }

    // Add mixed (all-zoom) files to every zoom level
    for zoom_files in &mut files_per_zoom {
        zoom_files.extend(mixed_files.iter().cloned());
        zoom_files.sort(); // maintain deterministic order
    }

    // Log partition info
    for (z, files) in files_per_zoom.iter().enumerate() {
        let disk_bytes: u64 = files.iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();
        eprintln!(
            "[encode] zoom {}: {} files ({:.1} GB on disk)",
            z, files.len(), disk_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }

    // Process all zoom levels in parallel.
    // Each zoom reads only its own files, groups fragments, and encodes GLBs.
    let results: Vec<io::Result<(u32, Vec<(u32, u32, u32, u32)>)>> =
        files_per_zoom.into_par_iter().enumerate().map(|(zoom_idx, zoom_files)| {
            let zoom = zoom_idx as u32;

            // Estimate memory for this zoom from actual file sizes
            let zoom_disk_bytes: u64 = zoom_files.iter()
                .filter_map(|p| std::fs::metadata(p).ok())
                .map(|m| m.len())
                .sum();
            let zoom_estimate = (zoom_disk_bytes as f64 * 3.0) as usize; // ZSTD ~3x

            // Batching: split into batches if estimated memory exceeds budget
            let effective_budget = (max_memory_bytes as f64 * 0.8) as usize;
            let n_batches = ((zoom_estimate + effective_budget - 1) / effective_budget).max(1);

            let mut zoom_count = 0u32;
            let mut zoom_keys = Vec::new();

            for batch in 0..n_batches {
                let groups: DashMap<(u32, u32, u32, u32), Vec<Fragment3D>> = DashMap::new();
                let n_b = n_batches;

                zoom_files.par_iter().for_each(|path| {
                    let mut reader = match Fragment3DReader::new(path) {
                        Ok(r) => r,
                        Err(_) => return,
                    };

                    let mut local: AHashMap<(u32, u32, u32, u32), Vec<Fragment3D>> = AHashMap::new();

                    loop {
                        match reader.read_next() {
                            Ok(Some(frag)) => {
                                // Filter: only this zoom level (needed for mixed files)
                                if frag.tile_z != zoom {
                                    continue;
                                }

                                // Filter: only this batch (by tile key hash)
                                if n_b > 1 {
                                    let mut hasher = DefaultHasher::new();
                                    (frag.tile_x, frag.tile_y, frag.tile_d).hash(&mut hasher);
                                    if (hasher.finish() as usize) % n_b != batch {
                                        continue;
                                    }
                                }

                                let key = (frag.tile_z, frag.tile_x, frag.tile_y, frag.tile_d);
                                local.entry(key).or_default().push(frag);
                            }
                            Ok(None) => break,
                            Err(_) => break,
                        }
                    }

                    // Batch-insert into shared DashMap
                    for (key, mut frags) in local {
                        groups.entry(key).or_default().extend(frags.drain(..));
                    }
                });

                // Encode this batch's tile groups
                let owned: AHashMap<(u32, u32, u32, u32), Vec<Fragment3D>> =
                    groups.into_iter().collect();

                if !owned.is_empty() {
                    let (count, keys) = _encode_grouped_fragments(
                        owned, tags_registry, out_dir,
                        xmin, ymin, zmin, dx, dy, dz,
                        max_zoom, base_cells, effective_compression,
                    )?;
                    zoom_count += count;
                    zoom_keys.extend(keys);
                }
            }

            eprintln!("[encode] zoom {} done: {} tiles", zoom, zoom_count);
            Ok((zoom_count, zoom_keys))
        }).collect();

    // Aggregate results from all zoom levels
    let mut total_count = 0u32;
    let mut all_tile_keys = Vec::new();
    for result in results {
        let (count, keys) = result?;
        total_count += count;
        all_tile_keys.extend(keys);
    }

    Ok((total_count, all_tile_keys))
}

/// Encode all tiles from a bucket directory into GLB files.
///
/// The bucket_dir contains one or more `.mjf` shard files for this bucket.
///
/// Returns (tile_count, tile_keys).
fn encode_bucket_to_3dtiles(
    bucket_dir: &Path,
    tags_registry: &HashMap<u32, Vec<(String, TagValue)>>,
    out_dir: &Path,
    xmin: f64, ymin: f64, zmin: f64,
    dx: f64, dy: f64, dz: f64,
    max_zoom: u32,
    base_cells: u32,
    effective_compression: &str,
) -> io::Result<(u32, Vec<(u32, u32, u32, u32)>)> {
    // Read all fragments from this bucket's shard files and group by tile key
    let mut reader = Fragment3DReader::open_dir(bucket_dir)?;
    let groups = reader.read_all_grouped()?;
    _encode_grouped_fragments(groups, tags_registry, out_dir,
        xmin, ymin, zmin, dx, dy, dz, max_zoom, base_cells, effective_compression)
}

/// Encode a single bucket file into GLB tiles.
fn encode_bucket_file_to_3dtiles(
    bucket_file: &Path,
    tags_registry: &HashMap<u32, Vec<(String, TagValue)>>,
    out_dir: &Path,
    xmin: f64, ymin: f64, zmin: f64,
    dx: f64, dy: f64, dz: f64,
    max_zoom: u32,
    base_cells: u32,
    effective_compression: &str,
) -> io::Result<(u32, Vec<(u32, u32, u32, u32)>)> {
    let mut reader = Fragment3DReader::new(bucket_file)?;
    let groups = reader.read_all_grouped()?;
    _encode_grouped_fragments(groups, tags_registry, out_dir,
        xmin, ymin, zmin, dx, dy, dz, max_zoom, base_cells, effective_compression)
}

/// Shared encoding logic for grouped fragments.
fn _encode_grouped_fragments(
    groups: AHashMap<(u32, u32, u32, u32), Vec<Fragment3D>>,
    tags_registry: &HashMap<u32, Vec<(String, TagValue)>>,
    out_dir: &Path,
    xmin: f64, ymin: f64, zmin: f64,
    dx: f64, dy: f64, dz: f64,
    max_zoom: u32,
    base_cells: u32,
    effective_compression: &str,
) -> io::Result<(u32, Vec<(u32, u32, u32, u32)>)> {

    let mut tiles: Vec<((u32, u32, u32, u32), Vec<Fragment3D>)> = groups.into_iter().collect();
    tiles.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    let tile_keys: Vec<(u32, u32, u32, u32)> = tiles.iter().map(|(k, _)| *k).collect();

    const INNER_PAR_THRESHOLD: usize = 16;

    let count: u32 = tiles.par_iter()
        .map(|((tz, tx, ty, td), frags)| {
            // Simplification already done during redistribution — skip here
            let do_simplify = false;

            let features: Vec<GlbFeature> = if frags.len() >= INNER_PAR_THRESHOLD {
                frags.par_iter().filter_map(|frag| {
                    fragment_to_glb_feature(
                        frag, tags_registry, xmin, ymin, zmin, dx, dy, dz,
                        do_simplify, max_zoom, base_cells, *tz,
                    )
                }).collect()
            } else {
                frags.iter().filter_map(|frag| {
                    fragment_to_glb_feature(
                        frag, tags_registry, xmin, ymin, zmin, dx, dy, dz,
                        do_simplify, max_zoom, base_cells, *tz,
                    )
                }).collect()
            };

            let data = match effective_compression {
                "draco" => encoder_glb::encode_glb_draco(&features, 50),
                "meshopt" => encoder_glb::encode_glb_meshopt(&features, 50),
                _ => encoder_glb::encode_glb(&features),
            };
            let tile_path = out_dir
                .join(tz.to_string())
                .join(tx.to_string())
                .join(ty.to_string())
                .join(format!("{}.glb", td));
            if let Some(parent) = tile_path.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            std::fs::write(&tile_path, data).ok();
            1u32
        })
        .sum();

    Ok((count, tile_keys))
}

// ---------------------------------------------------------------------------
// Parquet data collection (pure Rust, no GIL)
// ---------------------------------------------------------------------------

/// One row in the Parquet output — one fragment (feature × tile).
struct ParquetRow {
    zoom: u8,
    tile_x: u16,
    tile_y: u16,
    tile_d: u16,
    feature_id: u32,
    geom_type: u8,
    positions: Vec<u8>,  // raw LE float32 bytes [x,y,z,...]
    indices: Vec<u8>,    // raw LE uint32 bytes
}

/// Process all tile groups into ParquetRow structs.
///
/// For TIN/PolyhedralSurface at non-max zoom, applies vertex clustering
/// (same algorithm as `encode_glb_tile_from_fragments`). At max_zoom,
/// writes per-face vertices directly. Lines and points are transformed
/// to world coords without clustering.
fn collect_parquet_rows(
    tiles: &[((u32, u32, u32, u32), Vec<Fragment3D>)],
    world_bounds: &(f64, f64, f64, f64, f64, f64),
    max_zoom: u32,
    base_cells: u32,
) -> Vec<ParquetRow> {
    let (xmin, ymin, zmin, xmax, ymax, zmax) = *world_bounds;
    let dx = if xmax != xmin { xmax - xmin } else { 1.0 };
    let dy = if ymax != ymin { ymax - ymin } else { 1.0 };
    let dz = if zmax != zmin { zmax - zmin } else { 1.0 };

    tiles.par_iter()
        .flat_map(|((tz, tx, ty, td), frags)| {
            let tz = *tz;
            let do_simplify = tz < max_zoom;

            let mut rows: Vec<ParquetRow> = Vec::new();

            for frag in frags {
                let n_verts = frag.z.len();
                if n_verts == 0 {
                    continue;
                }

                let (pos_f32, idx_u32) = match frag.geom_type {
                    5 | 4 => {
                        // TIN / PolyhedralSurface
                        let rls: Vec<usize> = if frag.ring_lengths.is_empty() {
                            vec![n_verts]
                        } else {
                            frag.ring_lengths.iter().map(|&r| r as usize).collect()
                        };
                        let n_faces = rls.len();
                        let do_simplify_this = do_simplify && n_faces > 4;

                        if do_simplify_this {
                            parquet_tin_simplified(
                                frag, &rls, n_verts,
                                xmin, ymin, zmin, dx, dy, dz,
                                base_cells, tz, max_zoom,
                            )
                        } else {
                            parquet_tin_direct(
                                frag, &rls, n_verts,
                                xmin, ymin, zmin, dx, dy, dz,
                            )
                        }
                    }
                    2 => {
                        // LineString
                        let mut positions: Vec<f32> = Vec::with_capacity(n_verts * 3);
                        let mut indices: Vec<u32> = Vec::with_capacity(n_verts.saturating_sub(1) * 2);
                        for i in 0..n_verts {
                            positions.push((xmin + frag.xy[i * 2] as f64 * dx) as f32);
                            positions.push((ymin + frag.xy[i * 2 + 1] as f64 * dy) as f32);
                            positions.push((zmin + frag.z[i] as f64 * dz) as f32);
                        }
                        for i in 0..n_verts.saturating_sub(1) {
                            indices.push(i as u32);
                            indices.push((i + 1) as u32);
                        }
                        (positions, indices)
                    }
                    1 => {
                        // Point
                        let mut positions: Vec<f32> = Vec::with_capacity(n_verts * 3);
                        for i in 0..n_verts {
                            positions.push((xmin + frag.xy[i * 2] as f64 * dx) as f32);
                            positions.push((ymin + frag.xy[i * 2 + 1] as f64 * dy) as f32);
                            positions.push((zmin + frag.z[i] as f64 * dz) as f32);
                        }
                        (positions, Vec::new())
                    }
                    _ => continue,
                };

                if pos_f32.is_empty() {
                    continue;
                }

                // Convert to LE bytes
                let pos_bytes: Vec<u8> = pos_f32.iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect();
                let idx_bytes: Vec<u8> = idx_u32.iter()
                    .flat_map(|i| i.to_le_bytes())
                    .collect();

                rows.push(ParquetRow {
                    zoom: tz as u8,
                    tile_x: *tx as u16,
                    tile_y: *ty as u16,
                    tile_d: *td as u16,
                    feature_id: frag.feature_id,
                    geom_type: frag.geom_type,
                    positions: pos_bytes,
                    indices: idx_bytes,
                });
            }

            rows
        })
        .collect()
}

/// TIN mesh processing with QEM simplification (coarse zoom levels).
/// Builds an indexed mesh via `parquet_tin_direct`, then applies QEM.
fn parquet_tin_simplified(
    frag: &Fragment3D,
    rls: &[usize],
    n_verts: usize,
    xmin: f64, ymin: f64, zmin: f64,
    dx: f64, dy: f64, dz: f64,
    base_cells: u32, zoom: u32, max_zoom: u32,
) -> (Vec<f32>, Vec<u32>) {
    let (positions, indices) = parquet_tin_direct(
        frag, rls, n_verts,
        xmin, ymin, zmin, dx, dy, dz,
    );

    if indices.len() <= 12 {
        return (positions, indices);
    }

    let target_idx = simplify::compute_target_index_count(
        base_cells, zoom, max_zoom, indices.len(),
    );
    let target_tris = target_idx / 3;
    let (sp, mut si) = simplify::simplify_mesh(&positions, &indices, target_tris);
    filter_oversized_triangles(&sp, &mut si, tile_max_edge_sq_world(base_cells, zoom, dx, dy, dz));
    (sp, si)
}

/// TIN mesh processing without simplification (max_zoom level).
/// Deduplicates shared vertices via position hash map for indexed mesh output.
fn parquet_tin_direct(
    frag: &Fragment3D,
    rls: &[usize],
    n_verts: usize,
    xmin: f64, ymin: f64, zmin: f64,
    dx: f64, dy: f64, dz: f64,
) -> (Vec<f32>, Vec<u32>) {
    let mut vertex_map: AHashMap<(u32, u32, u32), u32> = AHashMap::new();
    let mut positions: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut offset = 0usize;

    for &rl in rls {
        let nv = if rl >= 4 { 3 } else { rl };
        if nv == 3 && offset + 2 < n_verts {
            for vi_off in 0..3 {
                let vi = offset + vi_off;
                let wx = (xmin + frag.xy[vi * 2] as f64 * dx) as f32;
                let wy = (ymin + frag.xy[vi * 2 + 1] as f64 * dy) as f32;
                let wz = (zmin + frag.z[vi] as f64 * dz) as f32;
                let key = (wx.to_bits(), wy.to_bits(), wz.to_bits());
                let idx = match vertex_map.get(&key) {
                    Some(&idx) => idx,
                    None => {
                        let idx = (positions.len() / 3) as u32;
                        vertex_map.insert(key, idx);
                        positions.push(wx);
                        positions.push(wy);
                        positions.push(wz);
                        idx
                    }
                };
                indices.push(idx);
            }
        }
        offset += rl;
    }

    (positions, indices)
}

// ---------------------------------------------------------------------------
// PyO3 helpers: extract features and tags from Python dicts
// ---------------------------------------------------------------------------

pub(crate) fn extract_tags(feat: &Bound<'_, PyDict>) -> PyResult<Vec<(String, TagValue)>> {
    let tags_obj = feat.get_item("tags")?;
    let mut result = Vec::new();

    if let Some(tags) = tags_obj {
        if let Ok(tags_dict) = tags.downcast::<PyDict>() {
            for (k, v) in tags_dict.iter() {
                if v.is_none() {
                    continue;
                }
                let key: String = k.extract()?;

                // Check bool before int (bool is subclass of int in Python)
                if v.is_instance_of::<PyBool>() {
                    let b: bool = v.extract()?;
                    result.push((key, TagValue::Bool(b)));
                } else if v.is_instance_of::<PyString>() {
                    let s: String = v.extract()?;
                    result.push((key, TagValue::Str(s)));
                } else if v.is_instance_of::<PyFloat>() {
                    let f: f64 = v.extract()?;
                    result.push((key, TagValue::Float(f)));
                } else if v.is_instance_of::<PyInt>() {
                    let i: i64 = v.extract()?;
                    result.push((key, TagValue::Int(i)));
                }
            }
        }
    }

    Ok(result)
}

fn extract_clip_feature(feat: &Bound<'_, PyDict>) -> PyResult<ClipFeature> {
    let xy: Vec<f64> = feat.get_item("geometry")?.unwrap().extract()?;
    let z: Vec<f64> = feat.get_item("geometry_z")?.unwrap().extract()?;
    let geom_type: u8 = feat.get_item("type")?.unwrap().extract()?;

    let ring_lengths: Vec<u32> = if let Some(rl) = feat.get_item("ring_lengths")? {
        if rl.is_none() {
            vec![]
        } else {
            rl.extract()?
        }
    } else {
        vec![]
    };

    let bbox = BBox3D {
        min_x: feat.get_item("minX")?.unwrap().extract()?,
        min_y: feat.get_item("minY")?.unwrap().extract()?,
        min_z: feat.get_item("minZ")?.unwrap().extract()?,
        max_x: feat.get_item("maxX")?.unwrap().extract()?,
        max_y: feat.get_item("maxY")?.unwrap().extract()?,
        max_z: feat.get_item("maxZ")?.unwrap().extract()?,
    };

    Ok(ClipFeature { xy, z, ring_lengths, geom_type, bbox })
}

// ---------------------------------------------------------------------------
// StreamingTileGenerator — PyO3 class
// ---------------------------------------------------------------------------

/// Streaming 3D tile generator.
///
/// Processes features one at a time via ``add_feature()``, clipping through
/// an octree and writing binary fragments to a temp file. Call
/// ``generate_pbf3()`` to read fragments back, transform to tile-local
/// integers, encode to protobuf, and write ``.pbf3`` tiles in parallel.
///
/// Memory: O(1 feature) during ingestion, O(fragments) during encoding.
#[pyclass]
pub struct StreamingTileGenerator {
    min_zoom: u32,
    max_zoom: u32,
    extent: u32,
    extent_z: u32,
    buffer: f64,
    base_cells: u32,
    feature_count: u32,
    tags_registry: HashMap<u32, Vec<(String, TagValue)>>,
    fragment_writer: Option<Fragment3DWriter>,
    frag_dir: PathBuf,
    shard_counter: std::sync::atomic::AtomicUsize,
    num_buckets: usize,
    tiles_written: u32,
    fragment_reader: Option<Fragment3DReader>,
    parquet_stream_active: bool,
}

#[pymethods]
impl StreamingTileGenerator {
    /// Create a new streaming tile generator.
    ///
    /// Args:
    ///   min_zoom: Minimum zoom level to generate tiles for.
    ///   max_zoom: Maximum zoom level to generate tiles for.
    ///   extent: XY tile extent in integer coordinates (default 4096).
    ///   extent_z: Z tile extent in integer coordinates (default 4096).
    ///   buffer: Tile buffer in normalized space (fraction of tile size).
    ///   base_cells: Grid resolution at zoom 0 for vertex clustering (default 10).
    ///     Higher values preserve finer detail at lower zoom levels.
    ///     Solid regions: 10. Thin branching structures (neurons): 50-200.
    #[new]
    #[pyo3(signature = (min_zoom=0, max_zoom=4, extent=4096, extent_z=4096, buffer=0.0, base_cells=10, temp_dir=None, num_buckets=256))]
    fn new(
        min_zoom: u32,
        max_zoom: u32,
        extent: u32,
        extent_z: u32,
        buffer: f64,
        base_cells: u32,
        temp_dir: Option<&str>,
        num_buckets: usize,
    ) -> PyResult<Self> {
        let gen_id = GENERATOR_ID.fetch_add(1, Ordering::Relaxed);
        let base_tmp = match temp_dir {
            Some(d) => PathBuf::from(d),
            None => std::env::temp_dir(),
        };
        let frag_dir = base_tmp
            .join(format!("microjson_frags_{}_{}", std::process::id(), gen_id));
        std::fs::create_dir_all(&frag_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        // Create shard 0 for the single-feature add_feature() path
        let shard_path = frag_dir.join("shard_000.mjf");
        let writer = Fragment3DWriter::new(&shard_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        Ok(Self {
            min_zoom,
            max_zoom,
            extent,
            extent_z,
            buffer,
            base_cells: if base_cells == 0 { DEFAULT_BASE_CELLS } else { base_cells },
            feature_count: 0,
            tags_registry: HashMap::new(),
            fragment_writer: Some(writer),
            frag_dir,
            shard_counter: std::sync::atomic::AtomicUsize::new(1), // shard 0 already created
            num_buckets: if num_buckets == 0 { 256 } else { num_buckets },
            tiles_written: 0,
            fragment_reader: None,
            parquet_stream_active: false,
        })
    }

    /// Add a single intermediate feature (already projected to [0,1]³).
    ///
    /// The feature dict must have keys: geometry, geometry_z, type,
    /// minX/minY/minZ/maxX/maxY/maxZ, and optionally tags, ring_lengths.
    ///
    /// Returns the assigned feature ID.
    fn add_feature(&mut self, feat: &Bound<'_, PyDict>) -> PyResult<u32> {
        let fid = self.feature_count;
        self.feature_count += 1;

        // Store tags in registry
        let tags = extract_tags(feat)?;
        self.tags_registry.insert(fid, tags);

        // Extract geometry to ClipFeature
        let clip_feat = extract_clip_feature(feat)?;

        // Clip through octree at all zoom levels
        let fragments = octree_clip(&clip_feat, self.min_zoom, self.max_zoom, self.buffer);

        // Write fragments to disk
        let writer = self.fragment_writer.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot add features after generate_pbf3()"))?;

        for ((tz, tx, ty, td), cf) in fragments {
            let frag = Fragment3D {
                feature_id: fid,
                tile_z: tz,
                tile_x: tx,
                tile_y: ty,
                tile_d: td,
                geom_type: cf.geom_type,
                xy: cf.xy.iter().map(|&v| v as f32).collect(),
                z: cf.z.iter().map(|&v| v as f32).collect(),
                ring_lengths: cf.ring_lengths,
            };
            writer.write(&frag)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        Ok(fid)
    }

    /// Generate .pbf3 tiles from accumulated fragments.
    ///
    /// Reads all fragments from the temp file, groups by tile key,
    /// transforms geometry to tile-local integers, encodes to protobuf,
    /// and writes tiles to ``output_dir/{z}/{x}/{y}/{d}.pbf3``.
    ///
    /// Uses rayon for parallel tile encoding (GIL released).
    ///
    /// Returns the number of tiles written.
    #[pyo3(signature = (output_dir, layer_name="default"))]
    fn generate_pbf3(&mut self, py: Python<'_>, output_dir: &str, layer_name: &str) -> PyResult<u32> {
        // Flush and close the fragment writer
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        // Read all fragments grouped by tile key
        let mut reader = Fragment3DReader::open_dir(&self.frag_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let groups = reader.read_all_grouped()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        // Collect for rayon
        let tiles: Vec<((u32, u32, u32, u32), Vec<Fragment3D>)> = groups.into_iter().collect();
        let tags_ref = &self.tags_registry;
        let extent = self.extent;
        let extent_z = self.extent_z;
        let out_dir = PathBuf::from(output_dir);
        let layer = layer_name.to_string();

        // Parallel tile encoding (GIL released)
        let count = py.allow_threads(|| {
            tiles.par_iter()
                .map(|((tz, tx, ty, td), frags)| {
                    let data = encode_tile_from_fragments(
                        frags, tags_ref,
                        *tz, *tx, *ty, *td,
                        extent, extent_z, &layer,
                    );
                    let tile_path = out_dir
                        .join(tz.to_string())
                        .join(tx.to_string())
                        .join(ty.to_string())
                        .join(format!("{}.pbf3", td));
                    if let Some(parent) = tile_path.parent() {
                        std::fs::create_dir_all(parent).ok();
                    }
                    std::fs::write(&tile_path, data).ok();
                    1u32
                })
                .sum::<u32>()
        });

        self.tiles_written = count;
        Ok(count)
    }

    /// Generate Parquet files from accumulated fragments using arrow-rs + parquet-rs.
    ///
    /// Reads fragments, transforms to world coords (parallel via Rayon),
    /// writes partitioned Parquet files per zoom level.
    ///
    /// Returns total number of rows written.
    #[pyo3(signature = (output_dir, world_bounds, simplify=true, compression="zstd"))]
    fn generate_parquet_native(
        &mut self,
        py: Python<'_>,
        output_dir: &str,
        world_bounds: (f64, f64, f64, f64, f64, f64),
        simplify: bool,
        compression: &str,
    ) -> PyResult<u64> {
        // Flush fragment writer
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        // Read all fragments grouped by tile
        let mut reader = Fragment3DReader::open_dir(&self.frag_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let groups = reader.read_all_grouped()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let tiles: Vec<((u32, u32, u32, u32), Vec<Fragment3D>)> = groups.into_iter().collect();
        let max_zoom = self.max_zoom;
        let base_cells = self.base_cells;
        let tags_snapshot: HashMap<u32, Vec<(String, TagValue)>> = self.tags_registry.clone();
        let out_dir = output_dir.to_string();
        let comp = compression.to_string();

        let total = py.allow_threads(|| {
            write_parquet_native_3d(
                &tiles, &out_dir, &world_bounds, max_zoom, base_cells,
                simplify, &tags_snapshot, &comp,
            )
        }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

        Ok(total)
    }

    /// Number of tiles written by the last generate_pbf3() call.
    fn tile_count(&self) -> u32 {
        self.tiles_written
    }

    /// Number of features added so far.
    fn feature_count_val(&self) -> u32 {
        self.feature_count
    }

    /// Generate .glb tiles for OGC 3D Tiles from accumulated fragments.
    ///
    /// Uses bucketed streaming: redistributes fragments into bucket files
    /// grouped by tile-key hash, then encodes one bucket at a time.
    /// Peak memory is O(fragments_per_bucket) instead of O(all_fragments),
    /// enabling datasets that exceed available RAM.
    ///
    /// Writes tiles to ``output_dir/{z}/{x}/{y}/{d}.glb`` and
    /// ``tileset.json`` to ``output_dir/tileset.json``.
    ///
    /// Uses rayon for parallel tile encoding within each bucket (GIL released).
    ///
    /// Returns the number of tiles written.
    #[pyo3(signature = (output_dir, world_bounds, layer_name="default", compression="none", use_draco=false, max_concurrent_buckets=8, max_memory_gb=0))]
    fn generate_3dtiles(
        &mut self,
        py: Python<'_>,
        output_dir: &str,
        world_bounds: (f64, f64, f64, f64, f64, f64),
        #[allow(unused)]
        layer_name: &str,
        compression: &str,
        use_draco: bool,
        #[allow(unused)]
        max_concurrent_buckets: usize,
        max_memory_gb: usize,
    ) -> PyResult<u32> {
        // Backward compat: use_draco=True with compression="none" → "draco"
        let effective_compression = if use_draco && compression == "none" {
            "draco"
        } else {
            compression
        };

        // Flush and close the fragment writer
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        let wb = world_bounds;
        let (xmin, ymin, zmin, xmax, ymax, zmax) = wb;
        let dx = if xmax != xmin { xmax - xmin } else { 1.0 };
        let dy = if ymax != ymin { ymax - ymin } else { 1.0 };
        let dz = if zmax != zmin { zmax - zmin } else { 1.0 };
        let out_dir = PathBuf::from(output_dir);
        let max_zoom = self.max_zoom;
        let base_cells = self.base_cells;
        let tags_ref = &self.tags_registry;
        let comp = effective_compression.to_string();

        // Memory budget: if 0, auto-detect from system RAM (80% of total)
        let max_memory_bytes = if max_memory_gb > 0 {
            max_memory_gb * 1024 * 1024 * 1024
        } else {
            // Read /proc/meminfo for total RAM, use 80%
            let total_ram = std::fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|s| {
                    s.lines()
                        .find(|l| l.starts_with("MemTotal:"))
                        .and_then(|l| l.split_whitespace().nth(1))
                        .and_then(|v| v.parse::<usize>().ok())
                })
                .unwrap_or(8 * 1024 * 1024); // fallback: 8 GB in KB
            (total_ram * 1024) * 4 / 5 // 80% of total, convert KB to bytes
        };

        // Per-zoom batched in-memory pipeline
        let frag_dir = self.frag_dir.clone();
        let (total_count, all_tile_keys) = py.allow_threads(|| {
            read_group_simplify_encode(
                &frag_dir, tags_ref, &out_dir,
                xmin, ymin, zmin, dx, dy, dz,
                max_zoom, base_cells, &comp,
                max_memory_bytes,
            )
        }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        // Write tileset.json
        let tileset = tileset_json::generate_tileset_json(
            &all_tile_keys, &wb, self.min_zoom, self.max_zoom,
        );
        let tileset_path = out_dir.join("tileset.json");
        let tileset_str = serde_json::to_string_pretty(&tileset)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        std::fs::write(&tileset_path, tileset_str)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        self.tiles_written = total_count;
        Ok(total_count)
    }

    /// Collect tile-centric Parquet data from all fragments.
    ///
    /// Flushes the fragment writer, reads all fragments grouped by tile key,
    /// processes meshes in parallel (vertex clustering at coarse zooms, world-coord
    /// transform), and returns columnar data as a Python dict of lists suitable
    /// for constructing a PyArrow table.
    ///
    /// Returns dict with keys: zoom, tile_x, tile_y, tile_d, feature_id,
    /// geom_type, positions (bytes), indices (bytes), tags (list of (str,str) tuples).
    #[pyo3(signature = (world_bounds,))]
    fn _collect_parquet_data(
        &mut self,
        py: Python<'_>,
        world_bounds: (f64, f64, f64, f64, f64, f64),
    ) -> PyResult<PyObject> {
        // Flush and close the fragment writer
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        // Read all fragments grouped by tile key
        let mut reader = Fragment3DReader::open_dir(&self.frag_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let groups = reader.read_all_grouped()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        // Collect for rayon
        let tiles: Vec<((u32, u32, u32, u32), Vec<Fragment3D>)> = groups.into_iter().collect();
        let wb = world_bounds;
        let max_zoom = self.max_zoom;
        let base_cells = self.base_cells;

        // GIL-released parallel mesh processing
        let rows = py.allow_threads(|| {
            collect_parquet_rows(&tiles, &wb, max_zoom, base_cells)
        });

        // Build Python dict of lists
        let tags_ref = &self.tags_registry;
        let n = rows.len();

        let zoom_list = pyo3::types::PyList::empty(py);
        let tx_list = pyo3::types::PyList::empty(py);
        let ty_list = pyo3::types::PyList::empty(py);
        let td_list = pyo3::types::PyList::empty(py);
        let fid_list = pyo3::types::PyList::empty(py);
        let gt_list = pyo3::types::PyList::empty(py);
        let pos_list = pyo3::types::PyList::empty(py);
        let idx_list = pyo3::types::PyList::empty(py);
        let tags_list = pyo3::types::PyList::empty(py);

        for row in &rows {
            zoom_list.append(row.zoom)?;
            tx_list.append(row.tile_x)?;
            ty_list.append(row.tile_y)?;
            td_list.append(row.tile_d)?;
            fid_list.append(row.feature_id)?;
            gt_list.append(row.geom_type)?;
            pos_list.append(pyo3::types::PyBytes::new(py, &row.positions))?;
            idx_list.append(pyo3::types::PyBytes::new(py, &row.indices))?;

            // Tags: list of (key, value_as_string) tuples for pa.map_ input
            let tag_pairs = pyo3::types::PyList::empty(py);
            if let Some(tags) = tags_ref.get(&row.feature_id) {
                for (k, v) in tags {
                    let vs = match v {
                        TagValue::Str(s) => s.clone(),
                        TagValue::Int(i) => i.to_string(),
                        TagValue::Float(f) => f.to_string(),
                        TagValue::Bool(b) => b.to_string(),
                    };
                    tag_pairs.append((k.as_str(), vs.as_str()))?;
                }
            }
            tags_list.append(tag_pairs)?;
        }

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("zoom", zoom_list)?;
        dict.set_item("tile_x", tx_list)?;
        dict.set_item("tile_y", ty_list)?;
        dict.set_item("tile_d", td_list)?;
        dict.set_item("feature_id", fid_list)?;
        dict.set_item("geom_type", gt_list)?;
        dict.set_item("positions", pos_list)?;
        dict.set_item("indices", idx_list)?;
        dict.set_item("tags", tags_list)?;
        dict.set_item("row_count", n)?;

        Ok(dict.into())
    }

    // ------------------------------------------------------------------
    // Streaming Parquet batch API — O(batch_size) memory
    // ------------------------------------------------------------------

    /// Initialize the streaming Parquet iterator.
    ///
    /// Flushes the fragment writer and opens a Fragment3DReader for sequential
    /// batch reading. Must be called before `_next_parquet_batch()`.
    fn _init_parquet_stream(&mut self) -> PyResult<()> {
        if self.parquet_stream_active {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Parquet stream already active — call _close_parquet_stream() first",
            ));
        }

        // Flush and close the fragment writer
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        // Open reader
        let reader = Fragment3DReader::open_dir(&self.frag_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        self.fragment_reader = Some(reader);
        self.parquet_stream_active = true;

        Ok(())
    }

    /// Read the next batch of fragments and return processed Parquet rows.
    ///
    /// Returns a Python dict with the same format as `_collect_parquet_data()`,
    /// or `None` when all fragments have been consumed (EOF).
    ///
    /// Memory: O(batch_size) — only `batch_size` fragments are loaded at a time.
    #[pyo3(signature = (batch_size, world_bounds, max_batch_bytes=2_000_000_000))]
    fn _next_parquet_batch(
        &mut self,
        py: Python<'_>,
        batch_size: usize,
        world_bounds: (f64, f64, f64, f64, f64, f64),
        max_batch_bytes: usize,
    ) -> PyResult<Option<PyObject>> {
        if !self.parquet_stream_active {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Parquet stream not active — call _init_parquet_stream() first",
            ));
        }

        let reader = self.fragment_reader.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Fragment3D reader is None")
        })?;

        // Read up to batch_size fragments, stopping early if byte budget exceeded.
        // Always read at least 1 fragment so we make progress.
        let mut groups: AHashMap<(u32, u32, u32, u32), Vec<Fragment3D>> = AHashMap::new();
        let mut count = 0usize;
        let mut batch_bytes = 0usize;
        while count < batch_size && (count == 0 || batch_bytes < max_batch_bytes) {
            match reader.read_next() {
                Ok(Some(frag)) => {
                    batch_bytes += frag.estimate_bytes();
                    let key = (frag.tile_z, frag.tile_x, frag.tile_y, frag.tile_d);
                    groups.entry(key).or_default().push(frag);
                    count += 1;
                }
                Ok(None) => break,  // EOF
                Err(e) => return Err(pyo3::exceptions::PyIOError::new_err(e.to_string())),
            }
        }

        // EOF — no fragments read
        if count == 0 {
            return Ok(None);
        }

        // Process fragments into ParquetRows (GIL-released, parallel)
        let tiles: Vec<((u32, u32, u32, u32), Vec<Fragment3D>)> = groups.into_iter().collect();
        let wb = world_bounds;
        let max_zoom = self.max_zoom;
        let base_cells = self.base_cells;

        let rows = py.allow_threads(|| {
            collect_parquet_rows(&tiles, &wb, max_zoom, base_cells)
        });

        // Build Python dict (same format as _collect_parquet_data)
        let tags_ref = &self.tags_registry;
        let n = rows.len();

        let zoom_list = pyo3::types::PyList::empty(py);
        let tx_list = pyo3::types::PyList::empty(py);
        let ty_list = pyo3::types::PyList::empty(py);
        let td_list = pyo3::types::PyList::empty(py);
        let fid_list = pyo3::types::PyList::empty(py);
        let gt_list = pyo3::types::PyList::empty(py);
        let pos_list = pyo3::types::PyList::empty(py);
        let idx_list = pyo3::types::PyList::empty(py);
        let tags_list = pyo3::types::PyList::empty(py);

        for row in &rows {
            zoom_list.append(row.zoom)?;
            tx_list.append(row.tile_x)?;
            ty_list.append(row.tile_y)?;
            td_list.append(row.tile_d)?;
            fid_list.append(row.feature_id)?;
            gt_list.append(row.geom_type)?;
            pos_list.append(pyo3::types::PyBytes::new(py, &row.positions))?;
            idx_list.append(pyo3::types::PyBytes::new(py, &row.indices))?;

            let tag_pairs = pyo3::types::PyList::empty(py);
            if let Some(tags) = tags_ref.get(&row.feature_id) {
                for (k, v) in tags {
                    let vs = match v {
                        TagValue::Str(s) => s.clone(),
                        TagValue::Int(i) => i.to_string(),
                        TagValue::Float(f) => f.to_string(),
                        TagValue::Bool(b) => b.to_string(),
                    };
                    tag_pairs.append((k.as_str(), vs.as_str()))?;
                }
            }
            tags_list.append(tag_pairs)?;
        }

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("zoom", zoom_list)?;
        dict.set_item("tile_x", tx_list)?;
        dict.set_item("tile_y", ty_list)?;
        dict.set_item("tile_d", td_list)?;
        dict.set_item("feature_id", fid_list)?;
        dict.set_item("geom_type", gt_list)?;
        dict.set_item("positions", pos_list)?;
        dict.set_item("indices", idx_list)?;
        dict.set_item("tags", tags_list)?;
        dict.set_item("row_count", n)?;

        Ok(Some(dict.into()))
    }

    /// Close the streaming Parquet iterator and release resources.
    fn _close_parquet_stream(&mut self) -> PyResult<()> {
        self.fragment_reader.take();
        self.parquet_stream_active = false;
        Ok(())
    }

    /// Add an OBJ file directly — parse, project, clip, and write fragments in Rust.
    ///
    /// This avoids creating Python intermediate objects entirely.
    /// Memory: O(1 mesh) — the OBJ vertices/faces are freed after clipping.
    ///
    /// Args:
    ///   path: Path to the OBJ file.
    ///   bounds: World bounding box (xmin, ymin, zmin, xmax, ymax, zmax).
    ///   tags: Property dict to attach to the feature.
    ///
    /// Returns the assigned feature ID.
    fn add_obj_file(
        &mut self,
        path: &str,
        bounds: (f64, f64, f64, f64, f64, f64),
        tags: &Bound<'_, PyDict>,
    ) -> PyResult<u32> {
        let (xmin, ymin, zmin, xmax, ymax, zmax) = bounds;
        let dx = if xmax != xmin { xmax - xmin } else { 1.0 };
        let dy = if ymax != ymin { ymax - ymin } else { 1.0 };
        let dz = if zmax != zmin { zmax - zmin } else { 1.0 };

        // Parse OBJ in Rust
        let (vertices, faces) = obj_parser::parse_obj(path)?;

        // Assign feature ID and store tags
        let fid = self.feature_count;
        self.feature_count += 1;

        let mut tag_vec: Vec<(String, TagValue)> = Vec::new();
        for (k, v) in tags.iter() {
            if v.is_none() {
                continue;
            }
            let key: String = k.extract()?;
            if v.is_instance_of::<PyBool>() {
                tag_vec.push((key, TagValue::Bool(v.extract()?)));
            } else if v.is_instance_of::<PyString>() {
                tag_vec.push((key, TagValue::Str(v.extract()?)));
            } else if v.is_instance_of::<PyFloat>() {
                tag_vec.push((key, TagValue::Float(v.extract()?)));
            } else if v.is_instance_of::<PyInt>() {
                tag_vec.push((key, TagValue::Int(v.extract()?)));
            }
        }
        self.tags_registry.insert(fid, tag_vec);

        // Build indexed mesh in [0,1]³ normalized coords
        let (positions, indices, bb) = build_indexed_mesh(
            &vertices, &faces, xmin, ymin, zmin, dx, dy, dz,
        );
        drop(vertices);
        drop(faces);

        // Pre-simplify per zoom level, then clip each LOD to its zoom
        let fragments = simplify_and_clip_per_zoom(
            &positions, &indices, bb,
            self.min_zoom, self.max_zoom, self.base_cells, self.buffer,
        );
        drop(positions);
        drop(indices);

        // Write fragments to disk
        let writer = self.fragment_writer.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot add features after generate"))?;

        for ((tz, tx, ty, td), cf) in fragments {
            let frag = Fragment3D {
                feature_id: fid,
                tile_z: tz,
                tile_x: tx,
                tile_y: ty,
                tile_d: td,
                geom_type: cf.geom_type,
                xy: cf.xy.iter().map(|&v| v as f32).collect(),
                z: cf.z.iter().map(|&v| v as f32).collect(),
                ring_lengths: cf.ring_lengths,
            };
            writer.write(&frag)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        Ok(fid)
    }

    /// Add multiple OBJ files in parallel using rayon.
    ///
    /// Tags are extracted from Python dicts while holding the GIL,
    /// then the GIL is released for parallel Rust processing.
    /// Each rayon thread parses an OBJ, projects to [0,1]³, clips through
    /// the octree, and writes fragments to disk under a mutex.
    ///
    /// Memory: O(N_cores * 1 mesh) — only in-flight meshes are in memory.
    ///
    /// Args:
    ///   paths: List of OBJ file paths.
    ///   bounds: World bounding box (xmin, ymin, zmin, xmax, ymax, zmax).
    ///   tags_list: List of property dicts (one per file).
    ///
    /// Returns the list of assigned feature IDs.
    #[pyo3(signature = (paths, bounds, tags_list, ingest_threads=0))]
    fn add_obj_files(
        &mut self,
        py: Python<'_>,
        paths: Vec<String>,
        bounds: (f64, f64, f64, f64, f64, f64),
        tags_list: &Bound<'_, PyList>,
        ingest_threads: usize,
    ) -> PyResult<Vec<u32>> {
        let n_files = paths.len();
        if n_files != tags_list.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "paths and tags_list must have the same length"));
        }

        let (xmin, ymin, zmin, xmax, ymax, zmax) = bounds;
        let dx = if xmax != xmin { xmax - xmin } else { 1.0 };
        let dy = if ymax != ymin { ymax - ymin } else { 1.0 };
        let dz = if zmax != zmin { zmax - zmin } else { 1.0 };

        // Pre-assign sequential feature IDs
        let base_fid = self.feature_count;
        self.feature_count += n_files as u32;
        let fids: Vec<u32> = (base_fid..base_fid + n_files as u32).collect();

        // Extract all tags from Python dicts (GIL held)
        for (i, item) in tags_list.iter().enumerate() {
            let tags_dict: &Bound<'_, PyDict> = item.downcast()?;
            let mut tag_vec: Vec<(String, TagValue)> = Vec::new();
            for (k, v) in tags_dict.iter() {
                if v.is_none() { continue; }
                let key: String = k.extract()?;
                if v.is_instance_of::<PyBool>() {
                    tag_vec.push((key, TagValue::Bool(v.extract()?)));
                } else if v.is_instance_of::<PyString>() {
                    tag_vec.push((key, TagValue::Str(v.extract()?)));
                } else if v.is_instance_of::<PyFloat>() {
                    tag_vec.push((key, TagValue::Float(v.extract()?)));
                } else if v.is_instance_of::<PyInt>() {
                    tag_vec.push((key, TagValue::Int(v.extract()?)));
                }
            }
            self.tags_registry.insert(fids[i], tag_vec);
        }

        // Close the single-feature writer (shard 0) — add_obj_files uses
        // per-thread sharded writers for full parallelism.
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        let min_zoom = self.min_zoom;
        let max_zoom = self.max_zoom;
        let buffer = self.buffer;
        let base_cells = self.base_cells;

        // Each file gets its own fragment output file — completely lock-free.
        // Files are named per input index: frag_{index}.mjf
        // Encoding phase reads all fragment files and groups by tile key.
        let frag_dir = self.frag_dir.clone();

        let mut indexed_paths: Vec<(usize, &String)> = paths.iter().enumerate().collect();
        indexed_paths.sort_by(|a, b| {
            let size_a = std::fs::metadata(a.1).map(|m| m.len()).unwrap_or(0);
            let size_b = std::fs::metadata(b.1).map(|m| m.len()).unwrap_or(0);
            size_b.cmp(&size_a)
        });

        // Release GIL — parallel parse + project + clip + write
        // with_min_len(1) forces rayon to steal one file at a time.
        // No locks: each thread writes to its own file.
        let n_threads = ingest_threads;
        py.allow_threads(|| {
            let do_ingest = |indexed_paths: &[(usize, &String)]| {
            indexed_paths.par_iter().with_min_len(1).for_each(|(i, path)| {
                let fid = base_fid + *i as u32;

                // Parse OBJ in Rust
                let (vertices, faces) = match obj_parser::parse_obj(path) {
                    Ok(vf) => vf,
                    Err(_) => return, // skip broken files
                };

                // Build indexed mesh in [0,1]³ normalized coords
                let (positions, indices, bb) = build_indexed_mesh(
                    &vertices, &faces, xmin, ymin, zmin, dx, dy, dz,
                );
                drop(vertices);
                drop(faces);

                let n_tris = indices.len() / 3;
                const PARALLEL_CLIP_THRESHOLD: usize = 500_000;
                const CLIP_CHUNK_SIZE: usize = 50_000;

                // Write helper closure for fragment output
                let write_clip_results = |clip_results: Vec<((u32,u32,u32,u32), ClipFeature)>,
                                          frag_path: PathBuf,
                                          fid: u32| {
                    let mut writer = match Fragment3DWriter::new(&frag_path) {
                        Ok(w) => w,
                        Err(_) => return,
                    };
                    for ((tz, tx_coord, ty, td), cf) in clip_results {
                        let frag = Fragment3D {
                            feature_id: fid,
                            tile_z: tz, tile_x: tx_coord, tile_y: ty, tile_d: td,
                            geom_type: cf.geom_type,
                            xy: cf.xy.iter().map(|&v| v as f32).collect(),
                            z: cf.z.iter().map(|&v| v as f32).collect(),
                            ring_lengths: cf.ring_lengths,
                        };
                        writer.write(&frag).ok();
                    }
                    writer.flush().ok();
                };

                if n_tris >= PARALLEL_CLIP_THRESHOLD {
                    // Large file: cascaded pre-simplify for coarse zooms, chunk only max_zoom
                    let full_index_count = indices.len();

                    // Coarse zooms: cascade simplify (each step from previous result)
                    let mut cur_pos = positions.clone();
                    let mut cur_idx = indices.clone();
                    for zoom in (min_zoom..max_zoom).rev() {
                        let target_idx = simplify::compute_target_index_count(
                            base_cells, zoom, max_zoom, full_index_count,
                        );
                        let target_tris = target_idx / 3;
                        if target_tris == 0 { continue; }
                        let (sp, mut si) = simplify::simplify_mesh(&cur_pos, &cur_idx, target_tris);
                        if si.is_empty() { continue; }
                        filter_oversized_triangles(&sp, &mut si, tile_max_edge_sq_normalized(base_cells, zoom));
                        if si.is_empty() { continue; }
                        let cf = indexed_mesh_to_clip_feature(&sp, &si, bb);
                        let clip_results = octree_clip(&cf, zoom, zoom, buffer);
                        let frag_path = frag_dir.join(
                            format!("frag_{:05}_z{}.mjf", i, zoom)
                        );
                        write_clip_results(clip_results, frag_path, fid);
                        cur_pos = sp;
                        cur_idx = si;
                    }

                    // Max zoom: chunk the full-resolution indexed mesh for parallel clipping
                    let chunk_count = (n_tris + CLIP_CHUNK_SIZE - 1) / CLIP_CHUNK_SIZE;
                    let indices_ref = &indices;
                    let positions_ref = &positions;
                    (0..chunk_count).into_par_iter().for_each(|chunk_idx| {
                        let start = chunk_idx * CLIP_CHUNK_SIZE * 3;
                        let end = (start + CLIP_CHUNK_SIZE * 3).min(indices_ref.len());
                        let chunk_indices = &indices_ref[start..end];
                        // Build ClipFeature directly from indexed chunk
                        let cn = chunk_indices.len() / 3;
                        let mut xy = Vec::with_capacity(cn * 4 * 2);
                        let mut z_vec = Vec::with_capacity(cn * 4);
                        let ring_lengths: Vec<u32> = vec![4; cn];
                        let mut cbb = BBox3D::empty();
                        for tri in 0..cn {
                            let i0 = chunk_indices[tri * 3] as usize;
                            let i1 = chunk_indices[tri * 3 + 1] as usize;
                            let i2 = chunk_indices[tri * 3 + 2] as usize;
                            for &vi in &[i0, i1, i2, i0] {
                                let px = positions_ref[vi * 3] as f64;
                                let py = positions_ref[vi * 3 + 1] as f64;
                                let pz = positions_ref[vi * 3 + 2] as f64;
                                xy.push(px);
                                xy.push(py);
                                z_vec.push(pz);
                                cbb.expand(px, py, pz);
                            }
                        }
                        let clip_feat = ClipFeature {
                            xy, z: z_vec, ring_lengths,
                            geom_type: TIN, bbox: cbb,
                        };
                        let clip_results = octree_clip(&clip_feat, max_zoom, max_zoom, buffer);
                        let frag_path = frag_dir.join(
                            format!("frag_{:05}_c{:04}.mjf", i, chunk_idx)
                        );
                        write_clip_results(clip_results, frag_path, fid);
                    });
                } else {
                    // Small file: pre-simplify per zoom, clip each LOD
                    let clip_results = simplify_and_clip_per_zoom(
                        &positions, &indices, bb,
                        min_zoom, max_zoom, base_cells, buffer,
                    );

                    let frag_path = frag_dir.join(format!("frag_{:05}.mjf", i));
                    write_clip_results(clip_results, frag_path, fid);
                }
            });
            };

            if n_threads > 0 {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n_threads)
                    .build()
                    .expect("Failed to build ingest thread pool");
                pool.install(|| do_ingest(&indexed_paths));
            } else {
                do_ingest(&indexed_paths);
            }
        });

        Ok(fids)
    }

    /// Add meshes from a Parquet file with pre-processed binary geometry.
    ///
    /// Each row is one mesh feature with flat binary columns for vertex
    /// positions (f32 LE, xyz triplets) and triangle indices (u32 LE).
    /// An optional string property column can provide per-feature tags.
    ///
    /// Processing is parallelised across rows using rayon. Each thread
    /// normalises the mesh into [0,1]³, runs cascaded QEM simplification
    /// and octree clipping per zoom level, then writes fragments to disk.
    ///
    /// Returns the number of features ingested.
    #[pyo3(signature = (path, positions_col="positions", indices_col="indices", prop_col=None, prop_name=None, layer_type="meshes", bounds=(0.0,0.0,0.0,1.0,1.0,1.0)))]
    fn add_parquet_meshes(
        &mut self,
        py: Python<'_>,
        path: &str,
        positions_col: &str,
        indices_col: &str,
        prop_col: Option<&str>,
        prop_name: Option<&str>,
        layer_type: &str,
        bounds: (f64, f64, f64, f64, f64, f64),
    ) -> PyResult<u32> {
        use arrow::array::{Array, LargeBinaryArray, BinaryArray, StringArray, LargeStringArray};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use std::fs::File;
        use std::sync::Mutex;

        let (xmin, ymin, zmin, xmax, ymax, zmax) = bounds;
        let dx = if xmax != xmin { xmax - xmin } else { 1.0 };
        let dy = if ymax != ymin { ymax - ymin } else { 1.0 };
        let dz = if zmax != zmin { zmax - zmin } else { 1.0 };

        // Close the single-feature writer; we'll use per-thread shard writers
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        let min_zoom = self.min_zoom;
        let max_zoom = self.max_zoom;
        let buffer = self.buffer;
        let base_cells = self.base_cells;

        let positions_col = positions_col.to_string();
        let indices_col = indices_col.to_string();
        let prop_col = prop_col.map(|s| s.to_string());
        let prop_name_str = prop_name.unwrap_or("name").to_string();
        let layer_type_str = layer_type.to_string();
        let path = path.to_string();

        let fid_counter = AtomicU32::new(self.feature_count);
        let fid_counter_ref = &fid_counter;

        // Per-thread fragment writers for lock-free parallelism
        let n_writers = rayon::current_num_threads() + 1;
        let writers: Vec<Mutex<Fragment3DWriter>> = (0..n_writers)
            .map(|_| {
                let shard_id = self.shard_counter.fetch_add(1, Ordering::Relaxed);
                let shard_path = self.frag_dir.join(format!("shard_{:03}.mjf", shard_id));
                Mutex::new(
                    Fragment3DWriter::new(&shard_path)
                        .expect("Failed to create fragment shard file")
                )
            })
            .collect();
        let writers_ref = &writers;

        // Read Parquet and process batches — GIL released
        let all_tags: Vec<Vec<(u32, Vec<(String, TagValue)>)>> = py.allow_threads(|| {
            let file = File::open(&path).expect("Cannot open parquet file");
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)
                .expect("Parquet reader error");
            let reader = builder.build().expect("Parquet build error");

            let mut batch_tags = Vec::new();

            for batch_result in reader {
                let batch = match batch_result {
                    Ok(b) => b,
                    Err(_) => continue,
                };

                let pos_col_idx = batch.schema().index_of(&positions_col)
                    .expect("positions column not found");
                let idx_col_idx = batch.schema().index_of(&indices_col)
                    .expect("indices column not found");
                let pos_arr = batch.column(pos_col_idx);
                let idx_arr = batch.column(idx_col_idx);

                // Optional property column
                let prop_arr = prop_col.as_ref().map(|col_name| {
                    let col_idx = batch.schema().index_of(col_name)
                        .expect("property column not found");
                    batch.column(col_idx).clone()
                });

                let n = batch.num_rows();

                // Collect row data for parallel processing
                struct RowData {
                    positions_bytes: Vec<u8>,
                    indices_bytes: Vec<u8>,
                    prop_value: Option<String>,
                }

                let mut rows: Vec<RowData> = Vec::with_capacity(n);

                for i in 0..n {
                    // Extract positions bytes
                    let pos_bytes = if let Some(a) = pos_arr.as_any().downcast_ref::<LargeBinaryArray>() {
                        if a.is_null(i) { continue; }
                        a.value(i).to_vec()
                    } else if let Some(a) = pos_arr.as_any().downcast_ref::<BinaryArray>() {
                        if a.is_null(i) { continue; }
                        a.value(i).to_vec()
                    } else {
                        continue;
                    };

                    // Extract indices bytes
                    let idx_bytes = if let Some(a) = idx_arr.as_any().downcast_ref::<LargeBinaryArray>() {
                        if a.is_null(i) { continue; }
                        a.value(i).to_vec()
                    } else if let Some(a) = idx_arr.as_any().downcast_ref::<BinaryArray>() {
                        if a.is_null(i) { continue; }
                        a.value(i).to_vec()
                    } else {
                        continue;
                    };

                    // Extract optional property
                    let prop_value = prop_arr.as_ref().and_then(|arr| {
                        if arr.is_null(i) { return None; }
                        if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
                            Some(a.value(i).to_string())
                        } else if let Some(a) = arr.as_any().downcast_ref::<LargeStringArray>() {
                            Some(a.value(i).to_string())
                        } else {
                            None
                        }
                    });

                    rows.push(RowData { positions_bytes: pos_bytes, indices_bytes: idx_bytes, prop_value });
                }

                // Process rows in parallel
                let this_tags: Vec<(u32, Vec<(String, TagValue)>)> = rows.par_iter()
                    .filter_map(|row| {
                        // Deserialize positions: flat f32 LE bytes → Vec<f32>
                        let n_floats = row.positions_bytes.len() / 4;
                        if n_floats < 3 || n_floats % 3 != 0 { return None; }
                        let mut positions = Vec::with_capacity(n_floats);
                        for chunk in row.positions_bytes.chunks_exact(4) {
                            positions.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                        }

                        // Deserialize indices: flat u32 LE bytes → Vec<u32>
                        let n_ints = row.indices_bytes.len() / 4;
                        if n_ints < 3 || n_ints % 3 != 0 { return None; }
                        let mut indices = Vec::with_capacity(n_ints);
                        for chunk in row.indices_bytes.chunks_exact(4) {
                            indices.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                        }

                        let n_verts = n_floats / 3;

                        // Validate: all indices must be in range
                        if indices.iter().any(|&idx| idx as usize >= n_verts) {
                            return None;
                        }

                        // Normalize positions to [0,1]³ in-place
                        let mut bb = BBox3D::empty();
                        for v in 0..n_verts {
                            let px = ((positions[v * 3] as f64 - xmin) / dx) as f32;
                            let py = ((positions[v * 3 + 1] as f64 - ymin) / dy) as f32;
                            let pz = ((positions[v * 3 + 2] as f64 - zmin) / dz) as f32;
                            positions[v * 3] = px;
                            positions[v * 3 + 1] = py;
                            positions[v * 3 + 2] = pz;
                            bb.expand(px as f64, py as f64, pz as f64);
                        }

                        // Assign feature ID
                        let fid = fid_counter_ref.fetch_add(1, Ordering::Relaxed);

                        // Build tags
                        let mut tags: Vec<(String, TagValue)> = Vec::with_capacity(2);
                        tags.push(("layer_type".to_string(), TagValue::Str(layer_type_str.clone())));
                        if let Some(ref val) = row.prop_value {
                            tags.push((prop_name_str.clone(), TagValue::Str(val.clone())));
                        }

                        // Simplify + clip per zoom level
                        let fragments = simplify_and_clip_per_zoom(
                            &positions, &indices, bb,
                            min_zoom, max_zoom, base_cells, buffer,
                        );

                        // Write fragments to shard
                        let writer_idx = rayon::current_thread_index().unwrap_or(n_writers - 1);
                        let mut w = writers_ref[writer_idx].lock().unwrap();
                        for ((tz, tx, ty, td), cf) in fragments {
                            let frag = Fragment3D {
                                feature_id: fid,
                                tile_z: tz,
                                tile_x: tx,
                                tile_y: ty,
                                tile_d: td,
                                geom_type: cf.geom_type,
                                xy: cf.xy.iter().map(|&v| v as f32).collect(),
                                z: cf.z.iter().map(|&v| v as f32).collect(),
                                ring_lengths: cf.ring_lengths,
                            };
                            w.write(&frag).ok();
                        }

                        Some((fid, tags))
                    })
                    .collect();

                batch_tags.push(this_tags);
            }

            batch_tags
        });

        // Flush shard writers
        for w in &writers {
            w.lock().unwrap().flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        // Collect tags into registry and update feature count
        let mut total_count: u32 = 0;
        for batch in all_tags {
            for (fid, tags) in batch {
                self.tags_registry.insert(fid, tags);
                total_count += 1;
            }
        }

        self.feature_count = fid_counter.load(Ordering::Relaxed);

        Ok(total_count)
    }

    /// Generate Neuroglancer precomputed legacy meshes from accumulated fragments.
    ///
    /// Unlike PBF3/3D Tiles which are tile-centric (one tile → many features),
    /// Neuroglancer meshes are segment-centric (one mesh per feature ID).
    ///
    /// For each feature:
    /// 1. Collect all fragments at max_zoom (highest resolution)
    /// 2. Merge geometry, unproject [0,1]³ → world coordinates
    /// 3. Vertex-dedup via hash map
    /// 4. Write binary mesh + JSON manifest + info file
    ///
    /// Directory structure:
    ///   {output_dir}/info                    — JSON {"@type":"neuroglancer_legacy_mesh"}
    ///   {output_dir}/{segment_id}            — binary mesh data
    ///   {output_dir}/{segment_id}:0          — JSON fragment manifest
    ///   {output_dir}/segment_properties/info — segment properties from tags
    ///
    /// Returns the number of segments written.
    #[pyo3(signature = (output_dir, world_bounds))]
    fn generate_neuroglancer(
        &mut self,
        py: Python<'_>,
        output_dir: &str,
        world_bounds: (f64, f64, f64, f64, f64, f64),
    ) -> PyResult<u32> {
        // Flush and close the fragment writer
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        // Read all fragments grouped by feature_id
        let mut reader = Fragment3DReader::open_dir(&self.frag_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let feature_groups = reader.read_all_grouped_by_feature()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let (xmin, ymin, zmin, xmax, ymax, zmax) = world_bounds;
        let dx = if xmax != xmin { xmax - xmin } else { 1.0 };
        let dy = if ymax != ymin { ymax - ymin } else { 1.0 };
        let dz = if zmax != zmin { zmax - zmin } else { 1.0 };

        let max_zoom = self.max_zoom;
        let out_dir = PathBuf::from(output_dir);
        let tags_ref = &self.tags_registry;

        // Process features in parallel
        let features: Vec<(u32, Vec<Fragment3D>)> = feature_groups.into_iter().collect();

        let count = py.allow_threads(|| {
            // Create output directory
            std::fs::create_dir_all(&out_dir).ok();

            // Write info file
            let info = serde_json::json!({
                "@type": "neuroglancer_legacy_mesh",
                "segment_properties": "segment_properties"
            });
            let info_str = serde_json::to_string_pretty(&info).unwrap_or_default();
            std::fs::write(out_dir.join("info"), info_str).ok();

            // Process each feature in parallel
            let segment_count: u32 = features.par_iter()
                .map(|(feature_id, frags)| {
                    // Filter to max_zoom fragments only (highest resolution)
                    let max_zoom_frags: Vec<&Fragment3D> = frags.iter()
                        .filter(|f| f.tile_z == max_zoom)
                        .collect();

                    if max_zoom_frags.is_empty() {
                        return 0u32;
                    }

                    // Merge geometry with vertex deduplication
                    let mut vertex_map: AHashMap<(i64, i64, i64), u32> = AHashMap::new();
                    let mut positions: Vec<f32> = Vec::new();
                    let mut indices: Vec<u32> = Vec::new();

                    for frag in &max_zoom_frags {
                        let n_verts = frag.z.len();
                        let ring_lengths: Vec<usize> = if frag.ring_lengths.is_empty() {
                            vec![n_verts]
                        } else {
                            frag.ring_lengths.iter().map(|&r| r as usize).collect()
                        };

                        match frag.geom_type {
                            5 | 4 => {
                                // TIN / PolyhedralSurface
                                let mut offset = 0usize;
                                for &rl in &ring_lengths {
                                    let nv = if rl >= 4 { 3 } else { rl };
                                    if nv < 3 || offset + 2 >= n_verts {
                                        offset += rl;
                                        continue;
                                    }

                                    let mut tri = [0u32; 3];
                                    for vi_off in 0..3 {
                                        let vi = offset + vi_off;
                                        // Unproject to world coordinates
                                        let wx = xmin + frag.xy[vi * 2] as f64 * dx;
                                        let wy = ymin + frag.xy[vi * 2 + 1] as f64 * dy;
                                        let wz = zmin + frag.z[vi] as f64 * dz;

                                        // Quantize to float32 precision for dedup
                                        let wx_f32 = wx as f32;
                                        let wy_f32 = wy as f32;
                                        let wz_f32 = wz as f32;

                                        let key = (
                                            wx_f32.to_bits() as i64,
                                            wy_f32.to_bits() as i64,
                                            wz_f32.to_bits() as i64,
                                        );

                                        let idx = match vertex_map.get(&key) {
                                            Some(&idx) => idx,
                                            None => {
                                                let idx = vertex_map.len() as u32;
                                                vertex_map.insert(key, idx);
                                                positions.push(wx_f32);
                                                positions.push(wy_f32);
                                                positions.push(wz_f32);
                                                idx
                                            }
                                        };
                                        tri[vi_off] = idx;
                                    }

                                    // Skip degenerate triangles
                                    if is_valid_triangle(&positions, &tri, f32::MAX) {
                                        indices.extend_from_slice(&tri);
                                    }

                                    offset += rl;
                                }
                            }
                            _ => {
                                // Skip non-triangle geometry for mesh output
                            }
                        }
                    }

                    if positions.is_empty() || indices.is_empty() {
                        return 0u32;
                    }

                    let num_vertices = positions.len() / 3;

                    // Encode as Neuroglancer legacy mesh binary
                    // Layout: uint32 num_vertices | float32[N*3] xyz | uint32[M] indices
                    let mut buf: Vec<u8> = Vec::with_capacity(
                        4 + positions.len() * 4 + indices.len() * 4
                    );
                    buf.extend_from_slice(&(num_vertices as u32).to_le_bytes());
                    for &p in &positions {
                        buf.extend_from_slice(&p.to_le_bytes());
                    }
                    for &i in &indices {
                        buf.extend_from_slice(&i.to_le_bytes());
                    }

                    // Write binary mesh file
                    let seg_path = out_dir.join(feature_id.to_string());
                    std::fs::write(&seg_path, &buf).ok();

                    // Write fragment manifest JSON
                    let manifest = format!(
                        "{{\"fragments\":[\"{}\"]}}",
                        feature_id
                    );
                    let manifest_path = out_dir.join(format!("{}:0", feature_id));
                    std::fs::write(&manifest_path, manifest).ok();

                    1u32
                })
                .sum::<u32>();

            // Write segment_properties/info from tags_registry
            let sp_dir = out_dir.join("segment_properties");
            std::fs::create_dir_all(&sp_dir).ok();

            let mut ids: Vec<String> = Vec::new();
            let mut all_keys: Vec<String> = Vec::new();
            let mut seen_keys: AHashMap<String, usize> = AHashMap::new();
            let mut columns: Vec<Vec<serde_json::Value>> = Vec::new();

            // Collect all feature IDs and property keys
            let mut sorted_features: Vec<&(u32, Vec<Fragment3D>)> = features.iter().collect();
            sorted_features.sort_by_key(|(fid, _)| *fid);

            for (fid, _) in &sorted_features {
                ids.push(fid.to_string());
                if let Some(tags) = tags_ref.get(fid) {
                    for (key, _) in tags {
                        if !seen_keys.contains_key(key) {
                            let col_idx = all_keys.len();
                            seen_keys.insert(key.clone(), col_idx);
                            all_keys.push(key.clone());
                            columns.push(Vec::new());
                        }
                    }
                }
            }

            // Fill column values
            for (fid, _) in &sorted_features {
                let tags = tags_ref.get(fid);
                for (ki, key) in all_keys.iter().enumerate() {
                    let val = tags.and_then(|t| {
                        t.iter().find(|(k, _)| k == key).map(|(_, v)| match v {
                            TagValue::Str(s) => serde_json::json!(s),
                            TagValue::Int(i) => serde_json::json!(i),
                            TagValue::Float(f) => serde_json::json!(f),
                            TagValue::Bool(b) => serde_json::json!(b),
                        })
                    }).unwrap_or(serde_json::json!(""));
                    columns[ki].push(val);
                }
            }

            // Build properties array
            let mut properties = Vec::new();
            for (ki, key) in all_keys.iter().enumerate() {
                let values = &columns[ki];
                // Determine type
                let all_numeric = values.iter().all(|v| v.is_number() || v == "");
                if all_numeric {
                    properties.push(serde_json::json!({
                        "id": key,
                        "type": "label",
                        "values": values
                    }));
                } else {
                    properties.push(serde_json::json!({
                        "id": key,
                        "type": "label",
                        "values": values
                    }));
                }
            }

            let sp_info = serde_json::json!({
                "@type": "neuroglancer_segment_properties",
                "inline": {
                    "ids": ids,
                    "properties": properties
                }
            });
            let sp_str = serde_json::to_string_pretty(&sp_info).unwrap_or_default();
            std::fs::write(sp_dir.join("info"), sp_str).ok();

            segment_count
        });

        self.tiles_written = count;
        Ok(count)
    }

    /// Generate per-feature .pbf3 files from accumulated fragments.
    ///
    /// Unlike tile-centric PBF3 (one tile → many features), this produces
    /// feature-centric PBF3 (one .pbf3 file per feature) for O(1) segment retrieval.
    ///
    /// When `multilod=True` (default), each .pbf3 contains one Layer per zoom level
    /// with genuine vertex reduction at coarser levels via grid-based clustering.
    /// When `multilod=False`, only max_zoom fragments are used (single-LOD, backward compat).
    ///
    /// Also writes `manifest.json` with per-feature bboxes, tags, and LOD count.
    ///
    /// Returns the number of feature files written.
    #[pyo3(signature = (output_dir, world_bounds, layer_name="default", multilod=true))]
    fn generate_feature_pbf3(
        &mut self,
        py: Python<'_>,
        output_dir: &str,
        world_bounds: (f64, f64, f64, f64, f64, f64),
        layer_name: &str,
        multilod: bool,
    ) -> PyResult<u32> {
        // Flush and close the fragment writer
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        // Read all fragments grouped by feature_id
        let mut reader = Fragment3DReader::open_dir(&self.frag_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let feature_groups = reader.read_all_grouped_by_feature()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let max_zoom = self.max_zoom;
        let extent = self.extent;
        let extent_z = self.extent_z;
        let base_cells = self.base_cells;
        let out_dir = PathBuf::from(output_dir);
        let tags_ref = &self.tags_registry;
        let wb = world_bounds;
        let layer = layer_name.to_string();

        let features: Vec<(u32, Vec<Fragment3D>)> = feature_groups.into_iter().collect();

        // Results include lod_count for manifest
        let count = py.allow_threads(|| {
            std::fs::create_dir_all(&out_dir).ok();

            // Process each feature in parallel → collect results for manifest
            let results: Vec<(u32, [f64; 6], usize, u32)> = features.par_iter()
                .filter_map(|(feature_id, frags)| {
                    if multilod {
                        // Multi-LOD: use all zoom levels
                        let all_refs: Vec<&Fragment3D> = frags.iter().collect();
                        if all_refs.is_empty() {
                            return None;
                        }

                        let tags = tags_ref.get(feature_id);
                        let (pbf3_bytes, bbox) = encode_feature_pbf3_multilod(
                            &all_refs, tags, &wb, max_zoom, extent, extent_z, base_cells,
                        );

                        if pbf3_bytes.is_empty() {
                            return None;
                        }

                        // Count distinct zoom levels
                        let mut zoom_set: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
                        for f in frags { zoom_set.insert(f.tile_z); }
                        let lod_count = zoom_set.len() as u32;
                        let byte_count = pbf3_bytes.len();

                        let feat_path = out_dir.join(format!("{}.pbf3", feature_id));
                        std::fs::write(&feat_path, pbf3_bytes).ok();

                        Some((*feature_id, bbox, byte_count, lod_count))
                    } else {
                        // Single-LOD: only max_zoom fragments (backward compat)
                        let max_zoom_frags: Vec<&Fragment3D> = frags.iter()
                            .filter(|f| f.tile_z == max_zoom)
                            .collect();

                        if max_zoom_frags.is_empty() {
                            return None;
                        }

                        let tags = tags_ref.get(feature_id);
                        let (pbf3_bytes, bbox) = encode_feature_pbf3(
                            &max_zoom_frags, tags, &wb, &layer, extent, extent_z,
                        );

                        if pbf3_bytes.is_empty() {
                            return None;
                        }

                        let byte_count = pbf3_bytes.len();
                        let feat_path = out_dir.join(format!("{}.pbf3", feature_id));
                        std::fs::write(&feat_path, pbf3_bytes).ok();

                        Some((*feature_id, bbox, byte_count, 1u32))
                    }
                })
                .collect();

            // Write manifest.json
            let (wb_xmin, wb_ymin, wb_zmin, wb_xmax, wb_ymax, wb_zmax) = wb;
            let mut features_json = serde_json::Map::new();
            for (fid, bbox, _byte_count, lod_count) in &results {
                let mut entry = serde_json::Map::new();
                entry.insert("bbox".to_string(), serde_json::json!([
                    bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]
                ]));
                if multilod {
                    entry.insert("lod_count".to_string(), serde_json::json!(lod_count));
                }
                // Include tags
                if let Some(tag_list) = tags_ref.get(fid) {
                    let mut tags_obj = serde_json::Map::new();
                    for (k, v) in tag_list {
                        let jv = match v {
                            TagValue::Str(s) => serde_json::json!(s),
                            TagValue::Int(i) => serde_json::json!(i),
                            TagValue::Float(f) => serde_json::json!(f),
                            TagValue::Bool(b) => serde_json::json!(b),
                        };
                        tags_obj.insert(k.clone(), jv);
                    }
                    entry.insert("tags".to_string(), serde_json::Value::Object(tags_obj));
                }
                features_json.insert(fid.to_string(), serde_json::Value::Object(entry));
            }

            let version = if multilod { 2 } else { 1 };
            let mut manifest_map = serde_json::Map::new();
            manifest_map.insert("format".to_string(), serde_json::json!("mudm_feature_pbf3"));
            manifest_map.insert("version".to_string(), serde_json::json!(version));
            manifest_map.insert("feature_count".to_string(), serde_json::json!(results.len()));
            manifest_map.insert("world_bounds".to_string(), serde_json::json!([wb_xmin, wb_ymin, wb_zmin, wb_xmax, wb_ymax, wb_zmax]));
            if multilod {
                manifest_map.insert("multilod".to_string(), serde_json::json!(true));
            }
            manifest_map.insert("features".to_string(), serde_json::Value::Object(features_json));

            let manifest = serde_json::Value::Object(manifest_map);
            let manifest_str = serde_json::to_string_pretty(&manifest).unwrap_or_default();
            std::fs::write(out_dir.join("manifest.json"), manifest_str).ok();

            results.len() as u32
        });

        self.tiles_written = count;
        Ok(count)
    }

    /// Generate Neuroglancer multilod_draco output from accumulated fragments.
    ///
    /// Produces one `.index` manifest and one Draco fragment data file per segment.
    /// All zoom levels are used — lod 0 = max_zoom (finest), lod N = zoom 0 (coarsest).
    /// Fragment positions within each LOD are sorted in Z-curve (Morton) order.
    ///
    /// Output structure:
    ///   {output_dir}/info                    (@type: neuroglancer_multilod_draco)
    ///   {output_dir}/{seg_id}.index          (binary manifest)
    ///   {output_dir}/{seg_id}                (concatenated Draco fragments)
    ///   {output_dir}/segment_properties/info (metadata from tags)
    ///
    /// Returns the number of segments written.
    #[pyo3(signature = (output_dir, world_bounds, vertex_quantization_bits=10))]
    fn generate_neuroglancer_multilod(
        &mut self,
        py: Python<'_>,
        output_dir: &str,
        world_bounds: (f64, f64, f64, f64, f64, f64),
        vertex_quantization_bits: u8,
    ) -> PyResult<u32> {
        // Flush and close the fragment writer
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        // Read all fragments grouped by feature_id
        let mut reader = Fragment3DReader::open_dir(&self.frag_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let feature_groups = reader.read_all_grouped_by_feature()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let (xmin, ymin, zmin, xmax, ymax, zmax) = world_bounds;
        let dx = if xmax != xmin { xmax - xmin } else { 1.0 };
        let dy = if ymax != ymin { ymax - ymin } else { 1.0 };
        let dz = if zmax != zmin { zmax - zmin } else { 1.0 };

        let max_zoom = self.max_zoom;
        let qbits = vertex_quantization_bits;
        let qmax = ((1u32 << qbits) - 1) as f64;
        let out_dir = PathBuf::from(output_dir);
        let tags_ref = &self.tags_registry;

        let features: Vec<(u32, Vec<Fragment3D>)> = feature_groups.into_iter().collect();

        // chunk_shape = world extent / 2^max_zoom (finest tile size in world coords)
        let n_tiles = (1u32 << max_zoom) as f64;
        let chunk_shape = [dx / n_tiles, dy / n_tiles, dz / n_tiles];
        let grid_origin = [xmin as f32, ymin as f32, zmin as f32];

        let count = py.allow_threads(|| {
            std::fs::create_dir_all(&out_dir).ok();

            // Write info file
            // Build the 4x3 transform matrix: identity scaling from stored model to model coords
            // stored_model_coord = grid_origin + vertex_offset + chunk_shape * 2^lod * (frag_pos + v/(2^bits-1))
            // The info transform maps stored model → model (identity for us since we use world coords)
            let transform = [
                1.0f64, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ];
            let info = serde_json::json!({
                "@type": "neuroglancer_multilod_draco",
                "vertex_quantization_bits": qbits,
                "transform": transform,
                "lod_scale_multiplier": 1.0,
                "segment_properties": "segment_properties"
            });
            let info_str = serde_json::to_string_pretty(&info).unwrap_or_default();
            std::fs::write(out_dir.join("info"), info_str).ok();

            // Process each feature (segment) in parallel
            let segment_count: u32 = features.par_iter()
                .map(|(feature_id, frags)| {
                    // Group fragments by tile_z → BTreeMap for ordered iteration
                    let mut by_zoom: std::collections::BTreeMap<u32, Vec<&Fragment3D>> = std::collections::BTreeMap::new();
                    for frag in frags {
                        if frag.geom_type == 4 || frag.geom_type == 5 {
                            by_zoom.entry(frag.tile_z).or_default().push(frag);
                        }
                    }

                    if by_zoom.is_empty() {
                        return 0u32;
                    }

                    let num_lods = by_zoom.len() as u32;
                    let mut all_draco_data: Vec<u8> = Vec::new();
                    let mut lod_scales: Vec<f32> = Vec::new();
                    let mut vertex_offsets: Vec<f32> = Vec::new();
                    let mut num_fragments_per_lod: Vec<u32> = Vec::new();
                    // Per-LOD fragment positions and sizes
                    let mut all_frag_positions: Vec<Vec<[u32; 3]>> = Vec::new();
                    let mut all_frag_offsets: Vec<Vec<u32>> = Vec::new();

                    // Process each zoom level: max_zoom → 0 = lod 0 → lod N
                    for (&zoom, zoom_frags) in by_zoom.iter().rev() {
                        let lod = max_zoom.saturating_sub(zoom);
                        let lod_scale = (1u32 << lod) as f32;
                        lod_scales.push(lod_scale);

                        // Vertex offsets (zero for now since we quantize per-fragment)
                        vertex_offsets.push(0.0);
                        vertex_offsets.push(0.0);
                        vertex_offsets.push(0.0);

                        // Group fragments by tile position (tile_x, tile_y, tile_d)
                        let mut by_tile: std::collections::BTreeMap<(u32, u32, u32), Vec<&Fragment3D>> = std::collections::BTreeMap::new();
                        for frag in zoom_frags {
                            by_tile.entry((frag.tile_x, frag.tile_y, frag.tile_d)).or_default().push(frag);
                        }

                        // Sort by Morton code
                        let mut tile_keys: Vec<(u32, u32, u32)> = by_tile.keys().cloned().collect();
                        tile_keys.sort_by(|a, b| crate::morton::morton_cmp(*a, *b));

                        let mut frag_positions: Vec<[u32; 3]> = Vec::new();
                        let mut frag_offsets: Vec<u32> = Vec::new();

                        for (tx, ty, td) in &tile_keys {
                            let tile_frags = &by_tile[&(*tx, *ty, *td)];

                            // Merge all TIN fragments in this tile into one mesh
                            let mut vertex_map: AHashMap<(i64, i64, i64), u32> = AHashMap::new();
                            let mut positions: Vec<f32> = Vec::new();
                            let mut indices: Vec<u32> = Vec::new();

                            for frag in tile_frags {
                                let n_verts = frag.z.len();
                                let ring_lengths: Vec<usize> = if frag.ring_lengths.is_empty() {
                                    vec![n_verts]
                                } else {
                                    frag.ring_lengths.iter().map(|&r| r as usize).collect()
                                };

                                let mut offset = 0usize;
                                for &rl in &ring_lengths {
                                    let nv = if rl >= 4 { 3 } else { rl };
                                    if nv < 3 || offset + 2 >= n_verts {
                                        offset += rl;
                                        continue;
                                    }

                                    let mut tri = [0u32; 3];
                                    for vi_off in 0..3 {
                                        let vi = offset + vi_off;
                                        let wx = xmin + frag.xy[vi * 2] as f64 * dx;
                                        let wy = ymin + frag.xy[vi * 2 + 1] as f64 * dy;
                                        let wz = zmin + frag.z[vi] as f64 * dz;

                                        let wx_f32 = wx as f32;
                                        let wy_f32 = wy as f32;
                                        let wz_f32 = wz as f32;

                                        let key = (
                                            wx_f32.to_bits() as i64,
                                            wy_f32.to_bits() as i64,
                                            wz_f32.to_bits() as i64,
                                        );

                                        let idx = match vertex_map.get(&key) {
                                            Some(&idx) => idx,
                                            None => {
                                                let idx = vertex_map.len() as u32;
                                                vertex_map.insert(key, idx);
                                                positions.push(wx_f32);
                                                positions.push(wy_f32);
                                                positions.push(wz_f32);
                                                idx
                                            }
                                        };
                                        tri[vi_off] = idx;
                                    }

                                    if is_valid_triangle(&positions, &tri, f32::MAX) {
                                        indices.extend_from_slice(&tri);
                                    }
                                    offset += rl;
                                }
                            }

                            if positions.is_empty() || indices.is_empty() {
                                continue;
                            }

                            // Quantize world-coord positions to [0, 2^bits) within this tile's bounding box
                            let tile_scale = (1u32 << zoom) as f64;
                            let tile_xmin = xmin + (*tx as f64 / tile_scale) * dx;
                            let tile_ymin = ymin + (*ty as f64 / tile_scale) * dy;
                            let tile_zmin = zmin + (*td as f64 / tile_scale) * dz;
                            let tile_dx = dx / tile_scale;
                            let tile_dy = dy / tile_scale;
                            let tile_dz = dz / tile_scale;

                            let n_mesh_verts = positions.len() / 3;
                            let mut quant_positions: Vec<u32> = Vec::with_capacity(n_mesh_verts * 3);
                            for i in 0..n_mesh_verts {
                                let wx = positions[i * 3] as f64;
                                let wy = positions[i * 3 + 1] as f64;
                                let wz = positions[i * 3 + 2] as f64;
                                let qx = ((wx - tile_xmin) / tile_dx * qmax).round().max(0.0).min(qmax) as u32;
                                let qy = ((wy - tile_ymin) / tile_dy * qmax).round().max(0.0).min(qmax) as u32;
                                let qz = ((wz - tile_zmin) / tile_dz * qmax).round().max(0.0).min(qmax) as u32;
                                quant_positions.push(qx);
                                quant_positions.push(qy);
                                quant_positions.push(qz);
                            }

                            // Encode with Draco
                            match crate::encoder_draco::encode_draco_mesh(&quant_positions, &indices) {
                                Ok(draco_bytes) => {
                                    frag_positions.push([*tx, *ty, *td]);
                                    frag_offsets.push(draco_bytes.len() as u32);
                                    all_draco_data.extend_from_slice(&draco_bytes);
                                }
                                Err(_) => {
                                    // Skip this fragment on encoding error
                                }
                            }
                        }

                        num_fragments_per_lod.push(frag_positions.len() as u32);
                        all_frag_positions.push(frag_positions);
                        all_frag_offsets.push(frag_offsets);
                    }

                    if all_draco_data.is_empty() {
                        return 0u32;
                    }

                    // Write .index binary manifest
                    let mut index_buf: Vec<u8> = Vec::new();
                    // chunk_shape: 3 x float32le
                    for &cs in &chunk_shape {
                        index_buf.extend_from_slice(&(cs as f32).to_le_bytes());
                    }
                    // grid_origin: 3 x float32le
                    for &go in &grid_origin {
                        index_buf.extend_from_slice(&go.to_le_bytes());
                    }
                    // num_lods: uint32le
                    index_buf.extend_from_slice(&num_lods.to_le_bytes());
                    // lod_scales: num_lods x float32le
                    for &ls in &lod_scales {
                        index_buf.extend_from_slice(&ls.to_le_bytes());
                    }
                    // vertex_offsets: num_lods * 3 x float32le (C-order [num_lods, 3])
                    for &vo in &vertex_offsets {
                        index_buf.extend_from_slice(&vo.to_le_bytes());
                    }
                    // num_fragments_per_lod: num_lods x uint32le
                    for &nf in &num_fragments_per_lod {
                        index_buf.extend_from_slice(&nf.to_le_bytes());
                    }
                    // Per-LOD: fragment_positions [3, N] uint32le then fragment_offsets [N] uint32le
                    for lod_idx in 0..num_lods as usize {
                        let fp = &all_frag_positions[lod_idx];
                        let fo = &all_frag_offsets[lod_idx];
                        // fragment_positions: [3, num_fragments] uint32le — column-major per spec
                        // x values
                        for pos in fp { index_buf.extend_from_slice(&pos[0].to_le_bytes()); }
                        // y values
                        for pos in fp { index_buf.extend_from_slice(&pos[1].to_le_bytes()); }
                        // z values
                        for pos in fp { index_buf.extend_from_slice(&pos[2].to_le_bytes()); }
                        // fragment_offsets: [num_fragments] uint32le
                        for &off in fo { index_buf.extend_from_slice(&off.to_le_bytes()); }
                    }

                    let index_path = out_dir.join(format!("{}.index", feature_id));
                    std::fs::write(&index_path, &index_buf).ok();

                    // Write concatenated Draco data
                    let data_path = out_dir.join(feature_id.to_string());
                    std::fs::write(&data_path, &all_draco_data).ok();

                    1u32
                })
                .sum::<u32>();

            // Write segment_properties/info
            let sp_dir = out_dir.join("segment_properties");
            std::fs::create_dir_all(&sp_dir).ok();

            let mut ids: Vec<String> = Vec::new();
            let mut all_keys: Vec<String> = Vec::new();
            let mut seen_keys: AHashMap<String, usize> = AHashMap::new();
            let mut columns: Vec<Vec<serde_json::Value>> = Vec::new();

            let mut sorted_features: Vec<&(u32, Vec<Fragment3D>)> = features.iter().collect();
            sorted_features.sort_by_key(|(fid, _)| *fid);

            for (fid, _) in &sorted_features {
                ids.push(fid.to_string());
                if let Some(tags) = tags_ref.get(fid) {
                    for (key, _) in tags {
                        if !seen_keys.contains_key(key) {
                            let col_idx = all_keys.len();
                            seen_keys.insert(key.clone(), col_idx);
                            all_keys.push(key.clone());
                            columns.push(Vec::new());
                        }
                    }
                }
            }

            for (fid, _) in &sorted_features {
                let tags = tags_ref.get(fid);
                for (ki, key) in all_keys.iter().enumerate() {
                    let val = tags.and_then(|t| {
                        t.iter().find(|(k, _)| k == key).map(|(_, v)| match v {
                            TagValue::Str(s) => serde_json::json!(s),
                            TagValue::Int(i) => serde_json::json!(i),
                            TagValue::Float(f) => serde_json::json!(f),
                            TagValue::Bool(b) => serde_json::json!(b),
                        })
                    }).unwrap_or(serde_json::json!(""));
                    columns[ki].push(val);
                }
            }

            let mut properties = Vec::new();
            for (ki, key) in all_keys.iter().enumerate() {
                properties.push(serde_json::json!({
                    "id": key,
                    "type": "label",
                    "values": columns[ki]
                }));
            }

            let sp_info = serde_json::json!({
                "@type": "neuroglancer_segment_properties",
                "inline": {
                    "ids": ids,
                    "properties": properties
                }
            });
            let sp_str = serde_json::to_string_pretty(&sp_info).unwrap_or_default();
            std::fs::write(sp_dir.join("info"), sp_str).ok();

            segment_count
        });

        self.tiles_written = count;
        Ok(count)
    }

    /// Write tilejson3d.json metadata file.
    ///
    /// Args:
    ///   path: Output file path.
    ///   bounds: World bounds (xmin, ymin, zmin, xmax, ymax, zmax).
    ///   layer_name: Layer name (default "default").
    ///   encodings: Optional list of encoding dicts (format, compression, path, extension).
    ///   zoom_counts: Optional dict of zoom level (str) -> tile count.
    ///   id_fields: Optional list of identifier field names.
    #[pyo3(signature = (path, bounds, layer_name="default", encodings=None, zoom_counts=None, id_fields=None))]
    fn write_tilejson3d(
        &self,
        path: &str,
        bounds: (f64, f64, f64, f64, f64, f64),
        layer_name: &str,
        encodings: Option<Vec<std::collections::HashMap<String, String>>>,
        zoom_counts: Option<std::collections::HashMap<String, u64>>,
        id_fields: Option<Vec<String>>,
    ) -> PyResult<()> {
        let (xmin, ymin, zmin, xmax, ymax, zmax) = bounds;
        let center = [
            (xmin + xmax) / 2.0,
            (ymin + ymax) / 2.0,
            (zmin + zmax) / 2.0,
            self.min_zoom as f64,
        ];

        let mut json = serde_json::json!({
            "tilejson": "3.0.0",
            "tiles": ["{z}/{x}/{y}/{d}"],
            "name": layer_name,
            "minzoom": self.min_zoom,
            "maxzoom": self.max_zoom,
            "bounds3d": [xmin, ymin, zmin, xmax, ymax, zmax],
            "center3d": center,
            "depthsize": self.extent_z,
            "vector_layers": [{
                "id": layer_name,
                "fields": {},
                "minzoom": self.min_zoom,
                "maxzoom": self.max_zoom,
            }],
        });

        if let Some(enc) = encodings {
            json["encodings"] = serde_json::to_value(enc)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        }
        if let Some(zc) = zoom_counts {
            json["zoom_counts"] = serde_json::to_value(zc)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        }
        if let Some(ids) = id_fields {
            json["id_fields"] = serde_json::to_value(ids)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        }

        let json_str = serde_json::to_string_pretty(&json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        std::fs::write(path, json_str)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }
}

/// Clean up temp fragment directory on drop.
impl Drop for StreamingTileGenerator {
    fn drop(&mut self) {
        // Drop the writer and reader to release file handles
        self.fragment_writer.take();
        self.fragment_reader.take();
        std::fs::remove_dir_all(&self.frag_dir).ok();
    }
}

// ---------------------------------------------------------------------------
// Parquet generation (pure Rust, parallel row collection + sequential write)
// ---------------------------------------------------------------------------

fn write_parquet_native_3d(
    tiles: &[((u32, u32, u32, u32), Vec<Fragment3D>)],
    output_dir: &str,
    world_bounds: &(f64, f64, f64, f64, f64, f64),
    max_zoom: u32,
    base_cells: u32,
    _simplify: bool,
    tags: &HashMap<u32, Vec<(String, TagValue)>>,
    compression: &str,
) -> Result<u64, String> {
    use arrow::array::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;

    if tiles.is_empty() {
        return Ok(0);
    }

    // Group tiles by zoom level
    let min_zoom = tiles.iter().map(|((z, _, _, _), _)| *z).min().unwrap_or(0);
    let max_z = tiles.iter().map(|((z, _, _, _), _)| *z).max().unwrap_or(0);

    let mut tiles_by_zoom: Vec<Vec<&((u32, u32, u32, u32), Vec<Fragment3D>)>> =
        vec![Vec::new(); (max_z as usize) + 1];
    for entry in tiles {
        tiles_by_zoom[entry.0 .0 as usize].push(entry);
    }

    // Arrow schema for 3D tiles (no ring_lengths, has tile_d)
    let schema = Schema::new(vec![
        Field::new("tile_x", DataType::UInt16, false),
        Field::new("tile_y", DataType::UInt16, false),
        Field::new("tile_d", DataType::UInt16, false),
        Field::new("feature_id", DataType::UInt32, false),
        Field::new("geom_type", DataType::UInt8, false),
        Field::new("positions", DataType::LargeBinary, false),
        Field::new("indices", DataType::LargeBinary, false),
        Field::new("tags",
            DataType::Map(
                std::sync::Arc::new(Field::new("entries", DataType::Struct(
                    vec![
                        Field::new("keys", DataType::Utf8, false),
                        Field::new("values", DataType::Utf8, true),
                    ].into()
                ), false)),
                false,
            ),
            false),
    ]);
    let schema_ref = std::sync::Arc::new(schema);

    let comp = match compression {
        "zstd" => Compression::ZSTD(Default::default()),
        "lz4" => Compression::LZ4_RAW,
        "snappy" => Compression::SNAPPY,
        _ => Compression::UNCOMPRESSED,
    };

    let props = WriterProperties::builder()
        .set_compression(comp)
        .build();

    let out_path = std::path::Path::new(output_dir);
    std::fs::create_dir_all(out_path).map_err(|e| format!("mkdir: {}", e))?;

    let total_rows = std::sync::atomic::AtomicU64::new(0);
    let total_rows_ref = &total_rows;

    // Process one zoom level at a time
    for zoom in min_zoom..=max_z {
        let zoom_tiles = &tiles_by_zoom[zoom as usize];
        if zoom_tiles.is_empty() {
            continue;
        }

        let zoom_dir = out_path.join(format!("zoom={}", zoom));
        std::fs::create_dir_all(&zoom_dir).map_err(|e| format!("mkdir: {}", e))?;

        // Split tiles into chunks -> each chunk collects rows + writes its own part file in parallel
        let n_parts = rayon::current_num_threads().max(1);
        let chunk_size = (zoom_tiles.len() + n_parts - 1) / n_parts;

        let chunks: Vec<(usize, &[&((u32, u32, u32, u32), Vec<Fragment3D>)])> =
            zoom_tiles.chunks(chunk_size).enumerate().collect();
        chunks.par_iter().try_for_each(|(part_idx, tile_chunk)| -> Result<(), String> {
            let part_idx = *part_idx;
            // Collect rows for this chunk of tiles
            let tiles_owned: Vec<((u32, u32, u32, u32), Vec<Fragment3D>)> = tile_chunk
                .iter()
                .map(|&entry| entry.clone())
                .collect();
            let rows = collect_parquet_rows(&tiles_owned, world_bounds, max_zoom, base_cells);
            drop(tiles_owned);

            if rows.is_empty() { return Ok(()); }

            let file_path = zoom_dir.join(format!("part_{:03}.parquet", part_idx));
            let file = std::fs::File::create(&file_path)
                .map_err(|e| format!("create {}: {}", file_path.display(), e))?;
            let mut pq_writer = ArrowWriter::try_new(file, schema_ref.clone(), Some(props.clone()))
                .map_err(|e| format!("ArrowWriter: {}", e))?;

            let n = rows.len();
            let tile_x: UInt16Array = rows.iter().map(|r| Some(r.tile_x)).collect();
            let tile_y: UInt16Array = rows.iter().map(|r| Some(r.tile_y)).collect();
            let tile_d: UInt16Array = rows.iter().map(|r| Some(r.tile_d)).collect();
            let feature_id: UInt32Array = rows.iter().map(|r| Some(r.feature_id)).collect();
            let geom_type: UInt8Array = rows.iter().map(|r| Some(r.geom_type)).collect();
            let positions: LargeBinaryArray = rows.iter().map(|r| Some(r.positions.as_slice())).collect();
            let indices: LargeBinaryArray = rows.iter().map(|r| Some(r.indices.as_slice())).collect();

            let key_builder = StringBuilder::with_capacity(n * 2, n * 32);
            let val_builder = StringBuilder::with_capacity(n * 2, n * 32);
            let mut map_builder = MapBuilder::new(None, key_builder, val_builder);

            for row in &rows {
                if let Some(tag_vec) = tags.get(&row.feature_id) {
                    for (k, v) in tag_vec {
                        map_builder.keys().append_value(k);
                        match v {
                            TagValue::Str(s) => map_builder.values().append_value(s),
                            TagValue::Int(i) => map_builder.values().append_value(i.to_string()),
                            TagValue::Float(f) => map_builder.values().append_value(f.to_string()),
                            TagValue::Bool(b) => map_builder.values().append_value(b.to_string()),
                        }
                    }
                }
                map_builder.append(true).map_err(|e| format!("map: {}", e))?;
            }

            let batch = arrow::record_batch::RecordBatch::try_new(
                schema_ref.clone(),
                vec![
                    std::sync::Arc::new(tile_x),
                    std::sync::Arc::new(tile_y),
                    std::sync::Arc::new(tile_d),
                    std::sync::Arc::new(feature_id),
                    std::sync::Arc::new(geom_type),
                    std::sync::Arc::new(positions),
                    std::sync::Arc::new(indices),
                    std::sync::Arc::new(map_builder.finish()),
                ],
            ).map_err(|e| format!("RecordBatch: {}", e))?;

            pq_writer.write(&batch).map_err(|e| format!("write: {}", e))?;
            pq_writer.close().map_err(|e| format!("close: {}", e))?;

            total_rows_ref.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        })?;
    }

    Ok(total_rows.load(std::sync::atomic::Ordering::Relaxed))
}

/// Scan a list of OBJ files and return the combined world bounding box.
///
/// Uses rayon to scan files in parallel across CPU cores.
/// Only reads vertex data — skips face parsing for speed.
/// Returns (xmin, ymin, zmin, xmax, ymax, zmax).
#[pyfunction]
pub fn scan_obj_bounds(py: Python<'_>, paths: Vec<String>) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
    py.allow_threads(|| {
        scan_bounds_parallel(&paths)
    }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))
}

/// Inner parallel bounds scan (no GIL).
fn scan_bounds_parallel(paths: &[String]) -> Result<(f64, f64, f64, f64, f64, f64), String> {
    use std::io::BufRead;

    // Each file scanned independently in parallel
    let per_file: Result<Vec<([f64; 3], [f64; 3])>, String> = paths.par_iter().with_min_len(1).map(|path| {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Cannot open {}: {}", path, e))?;
        let reader = std::io::BufReader::new(file);

        let mut fmin = [f64::INFINITY; 3];
        let mut fmax = [f64::NEG_INFINITY; 3];

        for line in reader.lines() {
            let line = line.map_err(|e| e.to_string())?;
            let line = line.trim_start();
            if !line.starts_with("v ") {
                continue;
            }
            let mut parts = line.split_whitespace();
            parts.next(); // skip "v"
            if let (Some(xs), Some(ys), Some(zs)) = (parts.next(), parts.next(), parts.next()) {
                if let (Ok(x), Ok(y), Ok(z)) = (xs.parse::<f64>(), ys.parse::<f64>(), zs.parse::<f64>()) {
                    if x < fmin[0] { fmin[0] = x; }
                    if y < fmin[1] { fmin[1] = y; }
                    if z < fmin[2] { fmin[2] = z; }
                    if x > fmax[0] { fmax[0] = x; }
                    if y > fmax[1] { fmax[1] = y; }
                    if z > fmax[2] { fmax[2] = z; }
                }
            }
        }

        Ok((fmin, fmax))
    }).collect();

    let per_file = per_file?;

    // Merge per-file bounds
    let mut gmin = [f64::INFINITY; 3];
    let mut gmax = [f64::NEG_INFINITY; 3];

    for (fmin, fmax) in per_file {
        for i in 0..3 {
            if fmin[i] < gmin[i] { gmin[i] = fmin[i]; }
            if fmax[i] > gmax[i] { gmax[i] = fmax[i]; }
        }
    }

    Ok((gmin[0], gmin[1], gmin[2], gmax[0], gmax[1], gmax[2]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_surface_basic() {
        // A single triangle face at x=[0.1, 0.9], should be kept when clipping [0, 0.5)
        let feat = ClipFeature {
            xy: vec![0.1, 0.2, 0.9, 0.3, 0.5, 0.8],
            z: vec![0.1, 0.2, 0.3],
            ring_lengths: vec![3],
            geom_type: TIN,
            bbox: BBox3D {
                min_x: 0.1, min_y: 0.2, min_z: 0.1,
                max_x: 0.9, max_y: 0.8, max_z: 0.3,
            },
        };

        // Should keep the face (it straddles [0, 0.5))
        let result = clip_surface_rs(&feat, 0.0, 0.5, 0);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.z.len(), 3);
    }

    #[test]
    fn test_clip_surface_reject() {
        // Triangle entirely at x=[0.6, 0.9], clipping [0, 0.5) should reject
        let feat = ClipFeature {
            xy: vec![0.6, 0.2, 0.9, 0.3, 0.7, 0.8],
            z: vec![0.1, 0.2, 0.3],
            ring_lengths: vec![3],
            geom_type: TIN,
            bbox: BBox3D {
                min_x: 0.6, min_y: 0.2, min_z: 0.1,
                max_x: 0.9, max_y: 0.8, max_z: 0.3,
            },
        };

        let result = clip_surface_rs(&feat, 0.0, 0.5, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_octree_clip_single_point() {
        let feat = ClipFeature {
            xy: vec![0.3, 0.4],
            z: vec![0.5],
            ring_lengths: vec![],
            geom_type: POINT3D,
            bbox: BBox3D {
                min_x: 0.3, min_y: 0.4, min_z: 0.5,
                max_x: 0.3, max_y: 0.4, max_z: 0.5,
            },
        };

        let result = octree_clip(&feat, 0, 1, 0.0);
        // Should appear at zoom 0 (one tile) and zoom 1 (one tile)
        assert!(result.len() >= 2);

        // At zoom 0, should be (0,0,0,0)
        let zoom0: Vec<_> = result.iter().filter(|((z, _, _, _), _)| *z == 0).collect();
        assert_eq!(zoom0.len(), 1);
        assert_eq!(zoom0[0].0, (0, 0, 0, 0));

        // At zoom 1, point at (0.3, 0.4, 0.5) should be in tile (1, 0, 0, 1)
        // x=0.3 → left half (x=0), y=0.4 → bottom half (y=0), z=0.5 → back half (d=1)
        let zoom1: Vec<_> = result.iter().filter(|((z, _, _, _), _)| *z == 1).collect();
        assert_eq!(zoom1.len(), 1);
        assert_eq!(zoom1[0].0, (1, 0, 0, 1));
    }

    #[test]
    fn test_clip_line_basic() {
        // Line from x=0.1 to x=0.9
        let feat = ClipFeature {
            xy: vec![0.1, 0.5, 0.9, 0.5],
            z: vec![0.5, 0.5],
            ring_lengths: vec![],
            geom_type: LINESTRING3D,
            bbox: BBox3D {
                min_x: 0.1, min_y: 0.5, min_z: 0.5,
                max_x: 0.9, max_y: 0.5, max_z: 0.5,
            },
        };

        // Clip to [0, 0.5) along X
        let result = clip_line_rs(&feat, 0.0, 0.5, 0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].z.len(), 2); // start + intersection at x=0.5
    }

    #[test]
    fn test_encode_tag_value_roundtrip() {
        let tag = TagValue::Int(42);
        let encoded = encode_tag_value(&tag);
        assert!(!encoded.is_empty());

        let tag = TagValue::Str("hello".to_string());
        let encoded = encode_tag_value(&tag);
        assert!(!encoded.is_empty());
    }
}
