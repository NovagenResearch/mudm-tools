/// Transform tile features from normalized [0,1] to integer coords.
///
/// Port of tile3d_cy.pyx — pre-computes scale/offset and uses typed
/// loops for arithmetic. Uses Python round() semantics (banker's rounding).
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Round half to even (banker's rounding) to match Python's round().
#[inline]
pub(crate) fn round_half_to_even(x: f64) -> i64 {
    let rounded = x.round();
    // Check if exactly halfway
    let frac = (x - x.floor()).abs();
    if (frac - 0.5).abs() < 1e-15 {
        // Exactly halfway — round to even
        let r = rounded as i64;
        if r % 2 != 0 {
            if x > 0.0 { r - 1 } else { r + 1 }
        } else {
            r
        }
    } else {
        rounded as i64
    }
}

/// Transform tile features from normalized [0,1] to integer coords.
#[pyfunction]
#[pyo3(signature = (tile, extent=4096, extent_z=4096))]
pub fn transform_tile_3d<'py>(
    py: Python<'py>,
    tile: &Bound<'py, PyDict>,
    extent: i64,
    extent_z: i64,
) -> PyResult<PyObject> {
    let z: i64 = tile.get_item("z")?.unwrap().extract()?;
    let tx: i64 = tile.get_item("x")?.unwrap().extract()?;
    let ty: i64 = tile.get_item("y")?.unwrap().extract()?;
    let td: i64 = tile.get_item("d")?.unwrap().extract()?;

    let n = 1i64 << z; // 2^z
    let x0 = tx as f64 / n as f64;
    let y0 = ty as f64 / n as f64;
    let z0 = td as f64 / n as f64;
    let scale_x = n as f64;
    let scale_y = n as f64;
    let scale_z = n as f64;

    let features: Vec<Bound<'py, PyDict>> = tile.get_item("features")?.unwrap().extract()?;
    let new_features = PyList::empty(py);
    let mut total_points: i64 = 0;

    for feat in &features {
        let xy: Vec<f64> = feat.get_item("geometry")?.unwrap().extract()?;
        let zz: Vec<f64> = feat.get_item("geometry_z")?.unwrap().extract()?;
        let nv = zz.len();

        let mut new_xy: Vec<i64> = Vec::with_capacity(nv * 2);
        let mut new_z: Vec<i64> = Vec::with_capacity(nv);

        for i in 0..nv {
            let lx = (xy[i * 2] - x0) * scale_x;
            let ly = (xy[i * 2 + 1] - y0) * scale_y;
            let lz = (zz[i] - z0) * scale_z;

            new_xy.push(round_half_to_even(lx * extent as f64));
            new_xy.push(round_half_to_even(ly * extent as f64));
            new_z.push(round_half_to_even(lz * extent_z as f64));
        }

        total_points += nv as i64;

        let new_feat = PyDict::new(py);
        new_feat.set_item("geometry", new_xy)?;
        new_feat.set_item("geometry_z", new_z)?;
        new_feat.set_item("type", feat.get_item("type")?.unwrap())?;
        new_feat.set_item("tags", feat.get_item("tags").ok().flatten()
            .unwrap_or_else(|| PyDict::new(py).into_any()))?;

        // Preserve ring_lengths and radii keys if present in source (even if None)
        if let Some(rl) = feat.get_item("ring_lengths")? {
            new_feat.set_item("ring_lengths", rl)?;
        }
        if let Some(radii) = feat.get_item("radii")? {
            new_feat.set_item("radii", radii)?;
        }

        new_features.append(new_feat)?;
    }

    let result = PyDict::new(py);
    result.set_item("features", new_features)?;
    result.set_item("z", z)?;
    result.set_item("x", tx)?;
    result.set_item("y", ty)?;
    result.set_item("d", td)?;
    result.set_item("num_features", features.len())?;
    result.set_item("num_points", total_points)?;
    result.set_item("extent", extent)?;
    result.set_item("extent_z", extent_z)?;

    Ok(result.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_half_to_even() {
        assert_eq!(round_half_to_even(0.5), 0);
        assert_eq!(round_half_to_even(1.5), 2);
        assert_eq!(round_half_to_even(2.5), 2);
        assert_eq!(round_half_to_even(3.5), 4);
        assert_eq!(round_half_to_even(-0.5), 0);
        assert_eq!(round_half_to_even(-1.5), -2);
        assert_eq!(round_half_to_even(1.3), 1);
        assert_eq!(round_half_to_even(1.7), 2);
    }
}
