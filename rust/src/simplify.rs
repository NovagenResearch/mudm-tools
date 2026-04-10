/// QEM (Quadric Error Metrics) mesh simplification.
///
/// Rust port of Fast-Quadric-Mesh-Simplification (sp4cerat/Meshlab).
/// Iteratively collapses the lowest-cost edge until the target triangle
/// count is reached.  Preserves surface topology and sharp features far
/// better than the previous grid-based vertex clustering.
use pyo3::prelude::*;
use pyo3::types::PyList;
use ahash::AHashMap;

// ---------------------------------------------------------------------------
// Symmetric 4×4 matrix (10 unique values)
// ---------------------------------------------------------------------------

/// Symmetric 4×4 matrix stored as 10 unique f64 values.
///
/// Layout (row-major upper triangle):
///   m0  m1  m2  m3
///       m4  m5  m6
///           m7  m8
///               m9
#[derive(Debug, Clone, Copy)]
struct SymmetricMatrix {
    m: [f64; 10],
}

impl SymmetricMatrix {
    #[inline]
    fn zero() -> Self {
        Self { m: [0.0; 10] }
    }

    /// Build from plane equation ax + by + cz + d = 0.
    #[inline]
    fn from_plane(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            m: [
                a * a, a * b, a * c, a * d,
                       b * b, b * c, b * d,
                              c * c, c * d,
                                     d * d,
            ],
        }
    }

    #[inline]
    fn add(&self, other: &Self) -> Self {
        let mut r = [0.0; 10];
        for i in 0..10 {
            r[i] = self.m[i] + other.m[i];
        }
        Self { m: r }
    }

    /// Determinant of 3×3 sub-matrix formed by the given indices.
    #[inline]
    fn det(&self, a11: usize, a12: usize, a13: usize,
                   a21: usize, a22: usize, a23: usize,
                   a31: usize, a32: usize, a33: usize) -> f64 {
        self.m[a11] * self.m[a22] * self.m[a33]
      + self.m[a13] * self.m[a21] * self.m[a32]
      + self.m[a12] * self.m[a23] * self.m[a31]
      - self.m[a13] * self.m[a22] * self.m[a31]
      - self.m[a11] * self.m[a23] * self.m[a32]
      - self.m[a12] * self.m[a21] * self.m[a33]
    }
}

// ---------------------------------------------------------------------------
// Internal QEM data structures
// ---------------------------------------------------------------------------

struct QemVertex {
    p: [f64; 3],
    q: SymmetricMatrix,
    tstart: usize,
    tcount: usize,
    border: bool,
}

struct QemTriangle {
    v: [usize; 3],
    err: [f64; 4], // err[0..2] = edge errors, err[3] = min
    deleted: bool,
    dirty: bool,
    n: [f64; 3],
}

/// Vertex → triangle reference.
#[derive(Clone)]
struct Ref {
    tid: usize,
    tvertex: usize,
}

// ---------------------------------------------------------------------------
// QEM core functions
// ---------------------------------------------------------------------------

/// Compute the error (cost) of collapsing vertex id_v1 onto id_v2,
/// and the optimal placement of the resulting vertex.
fn calculate_error(
    vertices: &[QemVertex],
    id_v1: usize,
    id_v2: usize,
) -> (f64, [f64; 3]) {
    let q = vertices[id_v1].q.add(&vertices[id_v2].q);
    let border = vertices[id_v1].border && vertices[id_v2].border;
    let det = q.det(0, 1, 2, 1, 4, 5, 2, 5, 7);

    if det.abs() > 1e-30 && !border {
        // Optimal vertex position via inverse of the quadric
        let inv_det = 1.0 / det;
        let x = -inv_det * q.det(1, 2, 3, 4, 5, 6, 5, 7, 8);
        let y =  inv_det * q.det(0, 2, 3, 1, 5, 6, 2, 7, 8);
        let z = -inv_det * q.det(0, 1, 3, 1, 4, 6, 2, 5, 8);
        let p = [x, y, z];
        let error = vertex_error(&q, x, y, z);
        (error, p)
    } else {
        // Determinant is zero or both vertices are border — pick best of
        // v1, v2, or midpoint.
        let p1 = vertices[id_v1].p;
        let p2 = vertices[id_v2].p;
        let p3 = [
            (p1[0] + p2[0]) * 0.5,
            (p1[1] + p2[1]) * 0.5,
            (p1[2] + p2[2]) * 0.5,
        ];
        let e1 = vertex_error(&q, p1[0], p1[1], p1[2]);
        let e2 = vertex_error(&q, p2[0], p2[1], p2[2]);
        let e3 = vertex_error(&q, p3[0], p3[1], p3[2]);
        let emin = e1.min(e2).min(e3);
        if emin == e1 { (e1, p1) }
        else if emin == e2 { (e2, p2) }
        else { (e3, p3) }
    }
}

/// Evaluate quadric error for a point (x,y,z).
#[inline]
fn vertex_error(q: &SymmetricMatrix, x: f64, y: f64, z: f64) -> f64 {
    q.m[0] * x * x + 2.0 * q.m[1] * x * y + 2.0 * q.m[2] * x * z
  + 2.0 * q.m[3] * x +       q.m[4] * y * y + 2.0 * q.m[5] * y * z
  + 2.0 * q.m[6] * y +       q.m[7] * z * z + 2.0 * q.m[8] * z
  +       q.m[9]
}

// ---------------------------------------------------------------------------
// Vector math helpers
// ---------------------------------------------------------------------------

#[inline]
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn normalize(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-30 {
        return [0.0; 3];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

// ---------------------------------------------------------------------------
// Main simplification
// ---------------------------------------------------------------------------

/// QEM mesh simplification — main entry point.
///
/// Takes flat f32 position array `[x,y,z, ...]` and u32 triangle indices,
/// returns simplified (positions, indices).
pub(crate) fn simplify_mesh(
    positions: &[f32],
    indices: &[u32],
    target_count: usize,
) -> (Vec<f32>, Vec<u32>) {
    let n_verts = positions.len() / 3;
    let n_tris = indices.len() / 3;

    if n_tris == 0 || n_verts == 0 || target_count >= n_tris {
        return (positions.to_vec(), indices.to_vec());
    }

    let target_count = target_count.max(1);

    // Build internal vertex array (promote f32 → f64 for precision)
    let mut vertices: Vec<QemVertex> = (0..n_verts)
        .map(|i| QemVertex {
            p: [
                positions[i * 3] as f64,
                positions[i * 3 + 1] as f64,
                positions[i * 3 + 2] as f64,
            ],
            q: SymmetricMatrix::zero(),
            tstart: 0,
            tcount: 0,
            border: false,
        })
        .collect();

    // Build internal triangle array
    let mut triangles: Vec<QemTriangle> = (0..n_tris)
        .map(|i| QemTriangle {
            v: [
                indices[i * 3] as usize,
                indices[i * 3 + 1] as usize,
                indices[i * 3 + 2] as usize,
            ],
            err: [0.0; 4],
            deleted: false,
            dirty: false,
            n: [0.0; 3],
        })
        .collect();

    let mut refs: Vec<Ref> = Vec::new();

    // Initial pass: compute normals, quadrics, and edge errors
    compute_normals(&mut triangles, &vertices);
    compute_quadrics(&triangles, &mut vertices);
    compute_all_errors(&mut triangles, &vertices);

    // Build refs once
    update_refs(&mut triangles, &mut vertices, &mut refs, true);

    let aggressiveness: f64 = 7.0;
    let mut deleted_triangles = 0usize;

    // Main iteration loop
    for iteration in 0..100 {
        if n_tris.saturating_sub(deleted_triangles) <= target_count {
            break;
        }

        // Periodically update mesh connectivity
        if iteration % 5 == 0 {
            update_refs(&mut triangles, &mut vertices, &mut refs, iteration == 0);
        }

        // Target threshold: increases each iteration
        let threshold = 0.000000001 * ((iteration as f64 + 3.0).powf(aggressiveness));

        // Scan all triangles for collapses below threshold
        for ti in 0..triangles.len() {
            if triangles[ti].deleted || triangles[ti].dirty {
                continue;
            }
            if triangles[ti].err[3] > threshold {
                continue;
            }

            for j in 0..3 {
                if triangles[ti].err[j] > threshold {
                    continue;
                }

                let i0 = triangles[ti].v[j];
                let i1 = triangles[ti].v[(j + 1) % 3];

                // Border edges: skip
                if vertices[i0].border != vertices[i1].border {
                    continue;
                }

                // Compute collapse target position
                let (_error, p) = calculate_error(&vertices, i0, i1);

                // Check for flipped triangles
                let (mut deleted0, flipped0) = check_flipped(
                    &p, i0, i1, &vertices, &triangles, &refs,
                );
                if flipped0 { continue; }

                let (mut deleted1, flipped1) = check_flipped(
                    &p, i1, i0, &vertices, &triangles, &refs,
                );
                if flipped1 { continue; }

                // Perform the collapse: move i0 to optimal position,
                // update all triangles referencing i1 → i0
                vertices[i0].p = p;
                vertices[i0].q = vertices[i0].q.add(&vertices[i1].q);

                // Update triangle references: iterate i0's refs then i1's refs,
                // appending surviving refs to build i0's new ref list.
                let tstart = refs.len();
                deleted_triangles += update_tris(
                    i0, i0, &mut deleted0, &mut triangles, &vertices, &mut refs,
                );
                deleted_triangles += update_tris(
                    i0, i1, &mut deleted1, &mut triangles, &vertices, &mut refs,
                );

                let tcount = refs.len() - tstart;
                vertices[i0].tstart = tstart;
                vertices[i0].tcount = tcount;

                break; // restart scanning from next triangle
            }
        }
    }

    // Compact: collect non-deleted triangles and remap vertices
    compact_mesh(&triangles, &vertices)
}

/// Compute face normals for all non-deleted triangles.
fn compute_normals(triangles: &mut [QemTriangle], vertices: &[QemVertex]) {
    for t in triangles.iter_mut() {
        if t.deleted { continue; }
        let p0 = vertices[t.v[0]].p;
        let p1 = vertices[t.v[1]].p;
        let p2 = vertices[t.v[2]].p;
        let d1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let d2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        t.n = normalize(cross(&d1, &d2));
    }
}

/// Accumulate quadric error matrices from triangle plane equations.
fn compute_quadrics(triangles: &[QemTriangle], vertices: &mut [QemVertex]) {
    for v in vertices.iter_mut() {
        v.q = SymmetricMatrix::zero();
    }
    for t in triangles {
        if t.deleted { continue; }
        let n = t.n;
        let p = vertices[t.v[0]].p;
        // Plane equation: n·x + d = 0  →  d = -n·p
        let d = -(n[0] * p[0] + n[1] * p[1] + n[2] * p[2]);
        let sm = SymmetricMatrix::from_plane(n[0], n[1], n[2], d);
        for &vi in &t.v {
            vertices[vi].q = vertices[vi].q.add(&sm);
        }
    }
}

/// Compute edge errors for all non-deleted triangles.
fn compute_all_errors(triangles: &mut [QemTriangle], vertices: &[QemVertex]) {
    for t in triangles.iter_mut() {
        if t.deleted { continue; }
        t.err[0] = calculate_error(vertices, t.v[0], t.v[1]).0;
        t.err[1] = calculate_error(vertices, t.v[1], t.v[2]).0;
        t.err[2] = calculate_error(vertices, t.v[2], t.v[0]).0;
        t.err[3] = t.err[0].min(t.err[1]).min(t.err[2]);
    }
}

/// Build/rebuild vertex→triangle reference lists.  On first call (or
/// when `detect_borders` is true), also identify border vertices.
fn update_refs(
    triangles: &mut [QemTriangle],
    vertices: &mut [QemVertex],
    refs: &mut Vec<Ref>,
    detect_borders: bool,
) {
    // Count references per vertex
    for v in vertices.iter_mut() {
        v.tstart = 0;
        v.tcount = 0;
    }
    for t in triangles.iter() {
        if t.deleted { continue; }
        for &vi in &t.v {
            vertices[vi].tcount += 1;
        }
    }

    // Allocate offsets
    let mut offset = 0usize;
    for v in vertices.iter_mut() {
        v.tstart = offset;
        offset += v.tcount;
        v.tcount = 0;
    }

    refs.clear();
    refs.resize(offset, Ref { tid: 0, tvertex: 0 });

    // Fill references
    for (ti, t) in triangles.iter().enumerate() {
        if t.deleted { continue; }
        for (j, &vi) in t.v.iter().enumerate() {
            let start = vertices[vi].tstart;
            let count = vertices[vi].tcount;
            refs[start + count] = Ref { tid: ti, tvertex: j };
            vertices[vi].tcount += 1;
        }
    }

    // Border detection
    if detect_borders {
        for v in vertices.iter_mut() {
            v.border = false;
        }

        // An edge is border if it appears in exactly one triangle.
        // A vertex is border if any of its edges are border.
        let mut edge_count: AHashMap<(usize, usize), usize> = AHashMap::new();
        for t in triangles.iter() {
            if t.deleted { continue; }
            for j in 0..3 {
                let mut a = t.v[j];
                let mut b = t.v[(j + 1) % 3];
                if a > b { std::mem::swap(&mut a, &mut b); }
                *edge_count.entry((a, b)).or_insert(0) += 1;
            }
        }

        for (&(a, b), &count) in &edge_count {
            if count == 1 {
                vertices[a].border = true;
                vertices[b].border = true;
            }
        }
    }

    // Clear dirty flags
    for t in triangles.iter_mut() {
        t.dirty = false;
    }
}

/// Check if collapsing vertex `i0` toward `i1` (placing the result at `p`)
/// would flip any triangle incident on `i0`.
fn check_flipped(
    p: &[f64; 3],
    i0: usize,
    i1: usize,
    vertices: &[QemVertex],
    triangles: &[QemTriangle],
    refs: &[Ref],
) -> (Vec<bool>, bool) {
    let tstart = vertices[i0].tstart;
    let tcount = vertices[i0].tcount;
    let mut deleted = vec![false; tcount];

    for k in 0..tcount {
        let r = &refs[tstart + k];
        let tid = r.tid;
        if triangles[tid].deleted {
            continue;
        }

        let s = r.tvertex;
        let id1 = triangles[tid].v[(s + 1) % 3];
        let id2 = triangles[tid].v[(s + 2) % 3];

        if id1 == i1 || id2 == i1 {
            deleted[k] = true;
            continue;
        }

        let d1 = normalize([
            vertices[id1].p[0] - p[0],
            vertices[id1].p[1] - p[1],
            vertices[id1].p[2] - p[2],
        ]);
        let d2 = normalize([
            vertices[id2].p[0] - p[0],
            vertices[id2].p[1] - p[1],
            vertices[id2].p[2] - p[2],
        ]);

        // Degenerate edge check
        if d1[0].abs() + d1[1].abs() + d1[2].abs() < 1e-30 {
            return (deleted, true);
        }
        if d2[0].abs() + d2[1].abs() + d2[2].abs() < 1e-30 {
            return (deleted, true);
        }

        let n = normalize(cross(&d1, &d2));
        if dot(&n, &triangles[tid].n) < 0.2 {
            return (deleted, true);
        }
    }

    (deleted, false)
}

/// Update triangles after a collapse, iterating over `source` vertex's refs
/// and repointing surviving triangles to `target`.  Surviving refs are
/// appended to the refs vector (building `target`'s new ref list).
/// Returns the number of newly deleted triangles.
fn update_tris(
    target: usize,
    source: usize,
    deleted: &mut [bool],
    triangles: &mut [QemTriangle],
    vertices: &[QemVertex],
    refs: &mut Vec<Ref>,
) -> usize {
    let tstart = vertices[source].tstart;
    let tcount = vertices[source].tcount;
    let mut del_count = 0usize;

    for k in 0..tcount {
        let r = refs[tstart + k].clone();
        let tid = r.tid;

        if triangles[tid].deleted {
            continue;
        }

        if deleted[k] {
            triangles[tid].deleted = true;
            del_count += 1;
            continue;
        }

        // Repoint to target vertex
        triangles[tid].v[r.tvertex] = target;
        triangles[tid].dirty = true;

        // Recompute edge errors
        triangles[tid].err[0] = calculate_error(vertices, triangles[tid].v[0], triangles[tid].v[1]).0;
        triangles[tid].err[1] = calculate_error(vertices, triangles[tid].v[1], triangles[tid].v[2]).0;
        triangles[tid].err[2] = calculate_error(vertices, triangles[tid].v[2], triangles[tid].v[0]).0;
        triangles[tid].err[3] = triangles[tid].err[0]
            .min(triangles[tid].err[1])
            .min(triangles[tid].err[2]);

        // Append surviving ref for target's new ref list
        refs.push(r);
    }

    del_count
}

/// Compact the mesh: remove deleted and degenerate triangles, remap vertex indices.
fn compact_mesh(triangles: &[QemTriangle], vertices: &[QemVertex]) -> (Vec<f32>, Vec<u32>) {
    let n_verts = vertices.len();
    let mut remap = vec![u32::MAX; n_verts];
    let mut out_positions: Vec<f32> = Vec::new();
    let mut out_indices: Vec<u32> = Vec::new();
    let mut next_idx = 0u32;

    for t in triangles {
        if t.deleted { continue; }

        // Filter degenerate triangles: skip if minimum altitude < 1% of longest edge.
        // These render as visible line artifacts, especially at coarse zoom levels.
        let p0 = vertices[t.v[0]].p;
        let p1 = vertices[t.v[1]].p;
        let p2 = vertices[t.v[2]].p;
        let e0 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e1 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let cx = cross(&e0, &e1);
        let area_sq = cx[0] * cx[0] + cx[1] * cx[1] + cx[2] * cx[2]; // = 4 * area²
        let e2 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        let max_edge_sq = (e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2])
            .max(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2])
            .max(e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]);
        if max_edge_sq > 0.0 && area_sq < 1e-4 * max_edge_sq * max_edge_sq {
            continue; // degenerate — altitude < 1% of longest edge
        }

        for &vi in &t.v {
            if remap[vi] == u32::MAX {
                remap[vi] = next_idx;
                out_positions.push(vertices[vi].p[0] as f32);
                out_positions.push(vertices[vi].p[1] as f32);
                out_positions.push(vertices[vi].p[2] as f32);
                next_idx += 1;
            }
            out_indices.push(remap[vi]);
        }
    }

    (out_positions, out_indices)
}

// ---------------------------------------------------------------------------
// Public helper: target triangle count from base_cells/zoom
// ---------------------------------------------------------------------------

/// Compute target index count (= target_triangles * 3) from `base_cells`,
/// `zoom`, and `max_zoom`.  Uses the stricter of two strategies:
///
/// 1. **Formula-based** (for large meshes): surface cells ≈ 6N² where N = base_cells * 2^zoom
/// 2. **Ratio-based** (for small meshes): reduce to 1/3 per zoom level below max_zoom
///
/// This ensures meaningful reduction even when the mesh has fewer triangles
/// than the formula-based budget.
pub(crate) fn compute_target_index_count(
    base_cells: u32,
    zoom: u32,
    max_zoom: u32,
    current_index_count: usize,
) -> usize {
    // Formula-based target (scales with tile grid resolution)
    let cells = base_cells as u64 * (1u64 << zoom);
    let target_faces = 6 * cells * cells;
    let formula_target = (target_faces * 3) as usize;

    // Ratio-based target: keep 1/3 per zoom level below max_zoom
    let zoom_diff = max_zoom.saturating_sub(zoom);
    let mut ratio_target = current_index_count;
    for _ in 0..zoom_diff {
        ratio_target /= 3;
    }

    // Take the stricter (lower) of the two, clamped to [3, current]
    formula_target.min(ratio_target).min(current_index_count).max(3)
}

// ---------------------------------------------------------------------------
// PyO3 wrapper: `decimate_tin` — same Python API, QEM internals
// ---------------------------------------------------------------------------

/// Decimate TIN coordinates using QEM simplification.
///
/// Args:
///   coordinates: TIN coords as list of [[[v0, v1, v2, v0]]] (closed triangle rings)
///   target_ratio: fraction of faces to keep (0.0–1.0), >= 1.0 returns input unchanged
///   world_bounds: optional (xmin, ymin, zmin, xmax, ymax, zmax) — unused in QEM but kept for API compat
///   zoom: current zoom level — unused in QEM but kept for API compat
///
/// Returns: simplified TIN coordinates in same format
#[pyfunction]
#[pyo3(signature = (coordinates, target_ratio, world_bounds=None, zoom=0))]
pub fn decimate_tin<'py>(
    py: Python<'py>,
    coordinates: &Bound<'py, PyList>,
    target_ratio: f64,
    world_bounds: Option<(f64, f64, f64, f64, f64, f64)>,
    zoom: u32,
) -> PyResult<PyObject> {
    let _ = (world_bounds, zoom); // API compat — not used by QEM

    let n_faces = coordinates.len();

    if target_ratio >= 1.0 || n_faces <= 4 {
        return Ok(coordinates.into_py(py));
    }

    // Extract triangle vertices as flat f32 arrays
    let mut positions: Vec<f32> = Vec::with_capacity(n_faces * 9);
    let mut indices: Vec<u32> = Vec::with_capacity(n_faces * 3);

    // First pass: collect all vertices, deduplicate
    let mut vertex_map: AHashMap<(u32, u32, u32), u32> = AHashMap::new();

    for i in 0..n_faces {
        let face: &Bound<'_, PyList> = &coordinates.get_item(i)?.downcast_into()?;
        let ring: &Bound<'_, PyList> = &face.get_item(0)?.downcast_into()?;
        for j in 0..3.min(ring.len()) {
            let v: Vec<f64> = ring.get_item(j)?.extract()?;
            let x = v[0] as f32;
            let y = if v.len() > 1 { v[1] as f32 } else { 0.0 };
            let z = if v.len() > 2 { v[2] as f32 } else { 0.0 };

            let key = (x.to_bits(), y.to_bits(), z.to_bits());
            let idx = match vertex_map.get(&key) {
                Some(&idx) => idx,
                None => {
                    let idx = (positions.len() / 3) as u32;
                    vertex_map.insert(key, idx);
                    positions.push(x);
                    positions.push(y);
                    positions.push(z);
                    idx
                }
            };
            indices.push(idx);
        }
    }

    let target_count = ((n_faces as f64 * target_ratio).ceil() as usize).max(1);

    let (simp_pos, simp_idx) = simplify_mesh(&positions, &indices, target_count);

    // Convert back to TIN coordinate format
    let result = PyList::empty(py);
    let n_simp_tris = simp_idx.len() / 3;

    for i in 0..n_simp_tris {
        let i0 = simp_idx[i * 3] as usize;
        let i1 = simp_idx[i * 3 + 1] as usize;
        let i2 = simp_idx[i * 3 + 2] as usize;

        let v0 = [simp_pos[i0 * 3] as f64, simp_pos[i0 * 3 + 1] as f64, simp_pos[i0 * 3 + 2] as f64];
        let v1 = [simp_pos[i1 * 3] as f64, simp_pos[i1 * 3 + 1] as f64, simp_pos[i1 * 3 + 2] as f64];
        let v2 = [simp_pos[i2 * 3] as f64, simp_pos[i2 * 3 + 1] as f64, simp_pos[i2 * 3 + 2] as f64];

        let ring = PyList::new(py, &[
            PyList::new(py, v0.as_slice())?,
            PyList::new(py, v1.as_slice())?,
            PyList::new(py, v2.as_slice())?,
            PyList::new(py, v0.as_slice())?, // closing vertex
        ])?;
        let face = PyList::new(py, &[ring])?;
        result.append(face)?;
    }

    if result.is_empty() {
        // All faces removed — return minimal subset
        let subset = PyList::empty(py);
        let limit = n_faces.min(4);
        for i in 0..limit {
            subset.append(coordinates.get_item(i)?)?;
        }
        return Ok(subset.into());
    }

    Ok(result.into())
}
