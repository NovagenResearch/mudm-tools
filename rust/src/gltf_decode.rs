use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

/// Decode a GLB byte buffer; return a dict with concatenated positions and indices
/// across all primitives of all meshes. Indices are offset by the running vertex
/// count so the concatenated index buffer is self-consistent.
///
/// **Holds the GIL for the duration of the decode**; intended for single-threaded
/// benchmark use, not as a production decoding path.
#[pyfunction]
pub fn decode_glb_buffer<'py>(
    py: Python<'py>,
    data: &Bound<'_, PyBytes>,
) -> PyResult<Bound<'py, PyDict>> {
    let bytes = data.as_bytes();
    let (document, buffers, _images) = gltf::import_slice(bytes)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("gltf import: {e}")))?;

    let mut positions: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut vertex_offset: u32 = 0;

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            let mut prim_vertex_count: u32 = 0;
            if let Some(pos_iter) = reader.read_positions() {
                for p in pos_iter {
                    positions.extend_from_slice(&p);
                    prim_vertex_count += 1;
                }
            }
            if let Some(idx_iter) = reader.read_indices() {
                for i in idx_iter.into_u32() {
                    indices.push(i + vertex_offset);
                }
            }
            vertex_offset += prim_vertex_count;
        }
    }

    let result = PyDict::new(py);
    result.set_item("positions", positions)?;
    result.set_item("indices", indices)?;
    Ok(result)
}
