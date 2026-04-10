/// Draco mesh encoder for Neuroglancer multilod_draco format.
///
/// Encodes pre-quantized integer positions and triangle indices into Draco binary format.
/// Positions are already quantized to [0, 2^bits) by the caller — Draco must NOT
/// apply additional quantization.
///
/// Uses draco-oxide's OBJ loader as a bridge since AttributeDomain is not public.

use draco_oxide::prelude::*;
use draco_oxide::encode;
use std::sync::atomic::{AtomicU64, Ordering};

static DRACO_TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Encode a triangle mesh with pre-quantized u32 positions to Draco binary.
///
/// Arguments:
/// - `positions`: flat array of pre-quantized position components [x0,y0,z0, x1,y1,z1, ...]
///   Each value in [0, 2^quantization_bits). Values must be < 2^24 for exact f32 representation.
/// - `indices`: triangle indices (length must be multiple of 3)
///
/// Returns Draco-encoded bytes.
pub fn encode_draco_mesh(
    positions: &[u32],
    indices: &[u32],
) -> Result<Vec<u8>, String> {
    let n_verts = positions.len() / 3;
    if n_verts == 0 || indices.is_empty() {
        return Err("Empty mesh".to_string());
    }
    if positions.len() % 3 != 0 {
        return Err("positions length must be multiple of 3".to_string());
    }
    if indices.len() % 3 != 0 {
        return Err("indices length must be multiple of 3".to_string());
    }

    let n_faces = indices.len() / 3;

    // Write a temp OBJ file — draco-oxide requires this since AttributeDomain is pub(crate).
    // Positions are cast to float (exact for values < 2^24, which covers 16-bit quantization).
    let tmp_dir = std::env::temp_dir();
    let seq = DRACO_TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tid = std::thread::current().id();
    let tmp_path = tmp_dir.join(format!(
        "draco_tmp_{:?}_{}.obj",
        tid, seq,
    ));

    {
        use std::io::Write;
        let mut f = std::fs::File::create(&tmp_path)
            .map_err(|e| format!("Cannot create temp OBJ: {}", e))?;

        for i in 0..n_verts {
            writeln!(
                f,
                "v {} {} {}",
                positions[i * 3] as f32,
                positions[i * 3 + 1] as f32,
                positions[i * 3 + 2] as f32,
            )
            .map_err(|e| format!("Write error: {}", e))?;
        }

        for i in 0..n_faces {
            // OBJ faces are 1-indexed
            writeln!(
                f,
                "f {} {} {}",
                indices[i * 3] + 1,
                indices[i * 3 + 1] + 1,
                indices[i * 3 + 2] + 1,
            )
            .map_err(|e| format!("Write error: {}", e))?;
        }
    }

    // Load and encode in a catch_unwind block since draco-oxide can panic
    // on edge cases (e.g., meshes with unused vertices after dedup).
    let result = std::panic::catch_unwind(|| {
        let mesh = draco_oxide::io::obj::load_obj(&tmp_path)
            .map_err(|e| format!("OBJ load error: {:?}", e))?;

        let mut buffer: Vec<u8> = Vec::new();
        encode::encode(mesh, &mut buffer, encode::Config::default())
            .map_err(|e| format!("Draco encode error: {:?}", e))?;

        Ok::<Vec<u8>, String>(buffer)
    });

    // Clean up temp file
    std::fs::remove_file(&tmp_path).ok();

    match result {
        Ok(Ok(buffer)) => Ok(buffer),
        Ok(Err(e)) => Err(e),
        Err(_) => Err("Draco encoding panicked (likely degenerate mesh)".to_string()),
    }
}

/// Encode a triangle mesh with f32 world-space positions to Draco binary.
///
/// Unlike `encode_draco_mesh` (which takes pre-quantized u32), this function
/// takes raw f32 positions in world coordinates. Draco's internal quantization
/// will handle compression while preserving the coordinate space.
///
/// Used by the GLB encoder for 3D Tiles — the viewer expects world-space
/// positions back from Draco decode.
pub fn encode_draco_mesh_f32(
    positions: &[f32],
    indices: &[u32],
) -> Result<Vec<u8>, String> {
    let n_verts = positions.len() / 3;
    if n_verts == 0 || indices.is_empty() {
        return Err("Empty mesh".to_string());
    }
    if positions.len() % 3 != 0 {
        return Err("positions length must be multiple of 3".to_string());
    }
    if indices.len() % 3 != 0 {
        return Err("indices length must be multiple of 3".to_string());
    }

    let n_faces = indices.len() / 3;

    let tmp_dir = std::env::temp_dir();
    let seq = DRACO_TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tid = std::thread::current().id();
    let tmp_path = tmp_dir.join(format!(
        "draco_f32_{:?}_{}.obj",
        tid, seq,
    ));

    {
        use std::io::Write;
        let mut f = std::fs::File::create(&tmp_path)
            .map_err(|e| format!("Cannot create temp OBJ: {}", e))?;

        for i in 0..n_verts {
            writeln!(
                f,
                "v {} {} {}",
                positions[i * 3],
                positions[i * 3 + 1],
                positions[i * 3 + 2],
            )
            .map_err(|e| format!("Write error: {}", e))?;
        }

        for i in 0..n_faces {
            writeln!(
                f,
                "f {} {} {}",
                indices[i * 3] + 1,
                indices[i * 3 + 1] + 1,
                indices[i * 3 + 2] + 1,
            )
            .map_err(|e| format!("Write error: {}", e))?;
        }
    }

    let result = std::panic::catch_unwind(|| {
        let mesh = draco_oxide::io::obj::load_obj(&tmp_path)
            .map_err(|e| format!("OBJ load error: {:?}", e))?;

        let mut buffer: Vec<u8> = Vec::new();
        encode::encode(mesh, &mut buffer, encode::Config::default())
            .map_err(|e| format!("Draco encode error: {:?}", e))?;

        Ok::<Vec<u8>, String>(buffer)
    });

    std::fs::remove_file(&tmp_path).ok();

    match result {
        Ok(Ok(buffer)) => Ok(buffer),
        Ok(Err(e)) => Err(e),
        Err(_) => Err("Draco encoding panicked (likely degenerate mesh)".to_string()),
    }
}

/// Python-exposed: encode float32 positions + u32 indices → Draco bytes.
/// Quantizes positions to `qbits` bits relative to the mesh bounding box.
#[pyo3::pyfunction]
#[pyo3(signature = (positions, indices, qbits=10))]
pub fn draco_encode_mesh(
    positions: Vec<f32>,
    indices: Vec<u32>,
    qbits: u8,
) -> pyo3::PyResult<pyo3::Py<pyo3::types::PyBytes>> {
    let n_verts = positions.len() / 3;
    if n_verts == 0 || indices.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Empty mesh"));
    }

    // Compute bounding box
    let mut mins = [f32::INFINITY; 3];
    let mut maxs = [f32::NEG_INFINITY; 3];
    for i in 0..n_verts {
        for d in 0..3 {
            let v = positions[i * 3 + d];
            if v < mins[d] { mins[d] = v; }
            if v > maxs[d] { maxs[d] = v; }
        }
    }

    let qmax = ((1u32 << qbits) - 1) as f64;
    let mut quant: Vec<u32> = Vec::with_capacity(positions.len());
    for i in 0..n_verts {
        for d in 0..3 {
            let range = (maxs[d] - mins[d]) as f64;
            let q = if range > 0.0 {
                ((positions[i * 3 + d] as f64 - mins[d] as f64) / range * qmax)
                    .round().max(0.0).min(qmax) as u32
            } else {
                0
            };
            quant.push(q);
        }
    }

    let bytes = encode_draco_mesh(&quant, &indices)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    pyo3::Python::with_gil(|py| {
        Ok(pyo3::types::PyBytes::new(py, &bytes).into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_single_triangle() {
        let positions = vec![0u32, 0, 0, 100, 0, 0, 50, 100, 0];
        let indices = vec![0u32, 1, 2];

        let result = encode_draco_mesh(&positions, &indices);
        assert!(result.is_ok(), "Draco encoding failed: {:?}", result.err());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty(), "Draco output is empty");
        assert_eq!(&bytes[..5], b"DRACO", "Missing DRACO magic");
    }

    #[test]
    fn test_encode_two_triangles() {
        let positions = vec![
            0u32, 0, 0,
            100, 0, 0,
            50, 100, 0,
            100, 100, 50,
        ];
        let indices = vec![0, 1, 2, 1, 2, 3];

        let result = encode_draco_mesh(&positions, &indices);
        assert!(result.is_ok(), "Draco encoding failed: {:?}", result.err());
        assert!(result.unwrap().len() > 5);
    }

    #[test]
    fn test_encode_empty_fails() {
        let result = encode_draco_mesh(&[], &[]);
        assert!(result.is_err());
    }
}
