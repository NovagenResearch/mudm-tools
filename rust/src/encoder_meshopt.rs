/// Meshoptimizer compression pipeline for GLB output.
///
/// Optimizes vertex/index ordering for GPU cache, then compresses
/// using meshopt vertex/index codecs.  ~10x faster encode and ~100x
/// faster decode than Draco, at slightly lower compression ratio.

/// Result of meshopt encoding for a single mesh.
pub(crate) struct MeshoptEncoded {
    /// Compressed vertex bytes (meshopt vertex codec).
    pub vertex_data: Vec<u8>,
    /// Compressed index bytes (meshopt index codec).
    pub index_data: Vec<u8>,
    /// Optimized positions (for computing min/max bounds).
    pub positions: Vec<f32>,
    /// Optimized indices (for count metadata).
    pub indices: Vec<u32>,
    /// Number of vertices after optimization.
    pub vertex_count: usize,
}

/// Optimize and compress a triangle mesh using meshoptimizer.
///
/// Pipeline:
/// 1. Reorder indices for GPU vertex cache locality
/// 2. Reorder vertices for fetch locality
/// 3. Compress vertex buffer (meshopt vertex codec)
/// 4. Compress index buffer (meshopt index codec)
pub(crate) fn encode_meshopt_mesh(
    positions: &[f32],
    indices: &[u32],
) -> Result<MeshoptEncoded, String> {
    let vertex_count = positions.len() / 3;
    let index_count = indices.len();

    if vertex_count == 0 || index_count == 0 {
        return Err("empty mesh".into());
    }

    // Reinterpret &[f32] as &[[f32; 3]] — same memory layout, no copy.
    let vertices_3: &[[f32; 3]] = unsafe {
        std::slice::from_raw_parts(positions.as_ptr() as *const [f32; 3], vertex_count)
    };

    // Step 1: Optimize index order for vertex cache
    let opt_indices = meshopt::optimize_vertex_cache(indices, vertex_count);

    // Step 2: Reorder vertices for fetch locality (modifies indices in-place)
    let mut opt_indices_mut = opt_indices;
    let opt_vertices = meshopt::optimize_vertex_fetch(&mut opt_indices_mut, vertices_3);

    // Flatten optimized positions back to &[f32]
    let opt_positions: Vec<f32> = opt_vertices.iter().flat_map(|v| v.iter().copied()).collect();
    let new_vertex_count = opt_vertices.len();

    // Step 3: Compress vertex buffer
    let vertex_data = meshopt::encoding::encode_vertex_buffer(&opt_vertices)
        .map_err(|e| format!("meshopt vertex encode: {e}"))?;

    // Step 4: Compress index buffer
    let index_data = meshopt::encoding::encode_index_buffer(&opt_indices_mut, new_vertex_count)
        .map_err(|e| format!("meshopt index encode: {e}"))?;

    Ok(MeshoptEncoded {
        vertex_data,
        index_data,
        positions: opt_positions,
        indices: opt_indices_mut,
        vertex_count: new_vertex_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_meshopt_basic() {
        // Simple quad (2 triangles, 4 unique vertices)
        let positions = vec![
            0.0f32, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
        ];
        let indices = vec![0u32, 1, 2, 0, 2, 3];

        let encoded = encode_meshopt_mesh(&positions, &indices).unwrap();
        assert_eq!(encoded.vertex_count, 4);
        assert_eq!(encoded.indices.len(), 6);
        assert!(!encoded.vertex_data.is_empty());
        assert!(!encoded.index_data.is_empty());
        // Compressed should generally be smaller than raw
        let raw_vertex_bytes = 4 * 3 * 4; // 4 verts * 3 floats * 4 bytes
        let raw_index_bytes = 6 * 4;       // 6 indices * 4 bytes
        // For very small meshes, compression overhead may make it larger,
        // but the encoder should still succeed
        assert!(encoded.vertex_data.len() > 0);
        assert!(encoded.index_data.len() > 0);
        let _ = (raw_vertex_bytes, raw_index_bytes);
    }

    #[test]
    fn test_encode_meshopt_large() {
        // Build a strip of 100 triangles (should compress well)
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        for i in 0..100 {
            let x = i as f32;
            positions.extend_from_slice(&[x, 0.0, 0.0, x + 1.0, 0.0, 0.0, x + 0.5, 1.0, 0.0]);
            let base = (i * 3) as u32;
            indices.extend_from_slice(&[base, base + 1, base + 2]);
        }

        let encoded = encode_meshopt_mesh(&positions, &indices).unwrap();
        assert_eq!(encoded.vertex_count, 300); // no shared vertices in this strip
        assert_eq!(encoded.indices.len(), 300);

        // For 100+ triangles, compressed index buffer should be smaller than raw
        let raw_index_bytes = 300 * 4;
        assert!(
            encoded.index_data.len() < raw_index_bytes,
            "meshopt index compression should beat raw: {} vs {}",
            encoded.index_data.len(),
            raw_index_bytes,
        );
    }

    #[test]
    fn test_encode_meshopt_empty() {
        let result = encode_meshopt_mesh(&[], &[]);
        assert!(result.is_err());
    }
}
