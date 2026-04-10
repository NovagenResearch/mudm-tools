/// 3D axis-parallel clipping for octree tiling.
///
/// Port of clip3d_cy.pyx — clips features against axis-aligned half-open
/// intervals [k1, k2) for axis 0 (X), 1 (Y), or 2 (Z).
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::types::BBox3D;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn axis_val(xy: &[f64], z: &[f64], idx: usize, axis: i32) -> f64 {
    match axis {
        0 => xy[idx * 2],
        1 => xy[idx * 2 + 1],
        _ => z[idx],
    }
}

#[inline]
fn intersect(
    xy: &[f64], z: &[f64],
    i: usize, j: usize,
    k: f64, axis: i32,
) -> (f64, f64, f64) {
    let ax = xy[i * 2];
    let ay = xy[i * 2 + 1];
    let az = z[i];
    let bx = xy[j * 2];
    let by = xy[j * 2 + 1];
    let bz = z[j];

    let t = match axis {
        0 => { let d = bx - ax; if d != 0.0 { (k - ax) / d } else { 0.0 } }
        1 => { let d = by - ay; if d != 0.0 { (k - ay) / d } else { 0.0 } }
        _ => { let d = bz - az; if d != 0.0 { (k - az) / d } else { 0.0 } }
    };

    (
        ax + (bx - ax) * t,
        ay + (by - ay) * t,
        az + (bz - az) * t,
    )
}

fn compute_bbox(xy: &[f64], z: &[f64]) -> BBox3D {
    let mut bb = BBox3D::empty();
    let n = z.len();
    for j in 0..n {
        bb.expand(xy[j * 2], xy[j * 2 + 1], z[j]);
    }
    bb
}

// ---------------------------------------------------------------------------
// clip_surface — TIN/PolyhedralSurface per-face clipping
// ---------------------------------------------------------------------------

/// Clip a TIN/PolyhedralSurface per-face — keep faces overlapping [k1, k2).
///
/// Args:
///   feat: dict with geometry, geometry_z, ring_lengths, type, tags
///   k1, k2: clip bounds
///   axis: 0=X, 1=Y, 2=Z
///
/// Returns: clipped dict or None
#[pyfunction]
pub fn clip_surface<'py>(
    py: Python<'py>,
    feat: &Bound<'py, PyDict>,
    k1: f64, k2: f64, axis: i32,
) -> PyResult<Option<PyObject>> {
    let xy_obj = feat.get_item("geometry")?.unwrap();
    let z_obj = feat.get_item("geometry_z")?.unwrap();
    let xy: Vec<f64> = xy_obj.extract()?;
    let z: Vec<f64> = z_obj.extract()?;

    let ring_lengths: Vec<usize> = if let Some(rl) = feat.get_item("ring_lengths")? {
        if rl.is_none() {
            vec![z.len()]
        } else {
            rl.extract()?
        }
    } else {
        vec![z.len()]
    };

    let mut out_xy: Vec<f64> = Vec::new();
    let mut out_z: Vec<f64> = Vec::new();
    let mut out_ring_lengths: Vec<usize> = Vec::new();

    let mut offset: usize = 0;
    for &rl in &ring_lengths {
        // Compute face min/max along clip axis
        let mut f_min = f64::INFINITY;
        let mut f_max = f64::NEG_INFINITY;

        for j in 0..rl {
            let val = match axis {
                0 => xy[(offset + j) * 2],
                1 => xy[(offset + j) * 2 + 1],
                _ => z[offset + j],
            };
            if val < f_min { f_min = val; }
            if val > f_max { f_max = val; }
        }

        // Half-open interval [k1, k2): reject if fully outside
        if f_min >= k2 || f_max < k1 {
            offset += rl;
            continue;
        }

        // Face overlaps — keep it whole
        out_xy.extend_from_slice(&xy[offset * 2..(offset + rl) * 2]);
        out_z.extend_from_slice(&z[offset..offset + rl]);
        out_ring_lengths.push(rl);
        offset += rl;
    }

    if out_ring_lengths.is_empty() {
        return Ok(None);
    }

    let bb = compute_bbox(&out_xy, &out_z);

    let result = PyDict::new(py);
    result.set_item("geometry", out_xy)?;
    result.set_item("geometry_z", out_z)?;
    result.set_item("ring_lengths", out_ring_lengths)?;
    result.set_item("type", feat.get_item("type")?.unwrap().extract::<i32>()?)?;
    result.set_item("tags", feat.get_item("tags").ok().flatten().unwrap_or_else(|| PyDict::new(py).into_any()))?;
    result.set_item("minX", bb.min_x)?;
    result.set_item("minY", bb.min_y)?;
    result.set_item("minZ", bb.min_z)?;
    result.set_item("maxX", bb.max_x)?;
    result.set_item("maxY", bb.max_y)?;
    result.set_item("maxZ", bb.max_z)?;

    Ok(Some(result.into()))
}

// ---------------------------------------------------------------------------
// clip_line — LineString3D clipping to [k1, k2)
// ---------------------------------------------------------------------------

/// Clip a LineString3D to [k1, k2) along axis.
/// Returns a list of clipped line segment dicts.
#[pyfunction]
pub fn clip_line<'py>(
    py: Python<'py>,
    feat: &Bound<'py, PyDict>,
    k1: f64, k2: f64, axis: i32,
) -> PyResult<Vec<PyObject>> {
    let xy: Vec<f64> = feat.get_item("geometry")?.unwrap().extract()?;
    let z: Vec<f64> = feat.get_item("geometry_z")?.unwrap().extract()?;
    let n = z.len();

    if n < 2 {
        return Ok(vec![]);
    }

    let tags = feat.get_item("tags").ok().flatten()
        .unwrap_or_else(|| PyDict::new(py).into_any());
    let ring_lengths = feat.get_item("ring_lengths").ok().flatten();

    let mut segments: Vec<PyObject> = Vec::new();
    let mut out_xy: Vec<f64> = Vec::new();
    let mut out_z: Vec<f64> = Vec::new();

    let flush = |out_xy: &mut Vec<f64>, out_z: &mut Vec<f64>, segments: &mut Vec<PyObject>, tags: &Bound<'_, pyo3::PyAny>, ring_lengths: &Option<Bound<'_, pyo3::PyAny>>| -> PyResult<()> {
        if out_z.len() >= 2 {
            let bb = compute_bbox(out_xy, out_z);
            let d = PyDict::new(py);
            d.set_item("geometry", std::mem::take(out_xy))?;
            d.set_item("geometry_z", std::mem::take(out_z))?;
            d.set_item("type", 2i32)?;
            d.set_item("tags", tags)?;
            if let Some(rl) = ring_lengths {
                d.set_item("ring_lengths", rl)?;
            }
            d.set_item("minX", bb.min_x)?;
            d.set_item("minY", bb.min_y)?;
            d.set_item("minZ", bb.min_z)?;
            d.set_item("maxX", bb.max_x)?;
            d.set_item("maxY", bb.max_y)?;
            d.set_item("maxZ", bb.max_z)?;
            segments.push(d.into());
        } else {
            out_xy.clear();
            out_z.clear();
        }
        Ok(())
    };

    for i in 0..n - 1 {
        let a_val = axis_val(&xy, &z, i, axis);
        let b_val = axis_val(&xy, &z, i + 1, axis);
        let a_in = k1 <= a_val && a_val < k2;

        if a_in {
            out_xy.push(xy[i * 2]);
            out_xy.push(xy[i * 2 + 1]);
            out_z.push(z[i]);
        }

        // Detect boundary crossings
        let cross_k1 = (a_val < k1 && b_val > k1) || (b_val < k1 && a_val > k1);
        let cross_k2 = (a_val < k2 && b_val >= k2) || (b_val < k2 && a_val >= k2);

        // Collect crossings with parameter t
        let mut crossings: Vec<(f64, f64, f64, f64, bool)> = Vec::new();
        if cross_k1 {
            let (ix, iy, iz) = intersect(&xy, &z, i, i + 1, k1, axis);
            let d = b_val - a_val;
            let t = if d != 0.0 { (k1 - a_val) / d } else { 0.0 };
            let entering = b_val > a_val;
            crossings.push((t, ix, iy, iz, entering));
        }
        if cross_k2 {
            let (ix, iy, iz) = intersect(&xy, &z, i, i + 1, k2, axis);
            let d = b_val - a_val;
            let t = if d != 0.0 { (k2 - a_val) / d } else { 0.0 };
            let entering = b_val < a_val;
            crossings.push((t, ix, iy, iz, entering));
        }

        // Sort by parameter t
        crossings.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for (_, ix, iy, iz, entering) in crossings {
            if entering {
                flush(&mut out_xy, &mut out_z, &mut segments, &tags, &ring_lengths)?;
                out_xy.push(ix);
                out_xy.push(iy);
                out_z.push(iz);
            } else {
                out_xy.push(ix);
                out_xy.push(iy);
                out_z.push(iz);
                flush(&mut out_xy, &mut out_z, &mut segments, &tags, &ring_lengths)?;
            }
        }
    }

    // Last point
    if n > 0 {
        let last_val = axis_val(&xy, &z, n - 1, axis);
        if k1 <= last_val && last_val < k2 {
            out_xy.push(xy[(n - 1) * 2]);
            out_xy.push(xy[(n - 1) * 2 + 1]);
            out_z.push(z[n - 1]);
        }
    }

    flush(&mut out_xy, &mut out_z, &mut segments, &tags, &ring_lengths)?;
    Ok(segments)
}

// ---------------------------------------------------------------------------
// clip_points — Point3D clipping to [k1, k2)
// ---------------------------------------------------------------------------

/// Clip points to [k1, k2) along axis.
/// Returns a list of point dicts.
#[pyfunction]
pub fn clip_points<'py>(
    py: Python<'py>,
    feat: &Bound<'py, PyDict>,
    k1: f64, k2: f64, axis: i32,
) -> PyResult<Vec<PyObject>> {
    let xy: Vec<f64> = feat.get_item("geometry")?.unwrap().extract()?;
    let z: Vec<f64> = feat.get_item("geometry_z")?.unwrap().extract()?;
    let tags = feat.get_item("tags").ok().flatten()
        .unwrap_or_else(|| PyDict::new(py).into_any());

    let n = z.len();
    let mut results: Vec<PyObject> = Vec::new();

    for i in 0..n {
        let val = axis_val(&xy, &z, i, axis);
        if k1 <= val && val < k2 {
            let px = xy[i * 2];
            let py_val = xy[i * 2 + 1];
            let pz = z[i];
            let d = PyDict::new(py);
            d.set_item("geometry", vec![px, py_val])?;
            d.set_item("geometry_z", vec![pz])?;
            d.set_item("type", 1i32)?;
            d.set_item("tags", &tags)?;
            d.set_item("minX", px)?;
            d.set_item("minY", py_val)?;
            d.set_item("minZ", pz)?;
            d.set_item("maxX", px)?;
            d.set_item("maxY", py_val)?;
            d.set_item("maxZ", pz)?;
            results.push(d.into());
        }
    }

    Ok(results)
}
