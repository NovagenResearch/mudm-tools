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

/// Result of meshopt encoding with u16-quantized positions (KHR_mesh_quantization).
pub(crate) struct MeshoptQuantEncoded {
    /// Compressed vertex bytes (meshopt vertex codec over `[u16;4]`, stride 8).
    pub vertex_data: Vec<u8>,
    /// Compressed index bytes (meshopt index codec).
    pub index_data: Vec<u8>,
    /// Optimized indices (for count metadata).
    pub indices: Vec<u32>,
    /// Number of vertices after optimization.
    pub vertex_count: usize,
    /// Node-TRS dequant: `world = translation + scale ⊙ quantized`.
    pub translation: [f32; 3],
    pub scale: [f32; 3],
    /// Quantized integer extents → glTF position accessor `min`/`max`.
    pub quant_min: [u16; 3],
    pub quant_max: [u16; 3],
}

/// meshopt pipeline (cache-opt → fetch-opt) then u16-quantize the optimized
/// positions and run the vertex codec on the `[u16;4]` buffer. The dequant
/// transform rides on the glTF node (KHR_mesh_quantization). Indices unchanged.
pub(crate) fn encode_meshopt_mesh_quantized(
    positions: &[f32],
    indices: &[u32],
    qbits: u8,
) -> Result<MeshoptQuantEncoded, String> {
    let vertex_count = positions.len() / 3;
    if vertex_count == 0 || indices.is_empty() {
        return Err("empty mesh".into());
    }
    let vertices_3: &[[f32; 3]] = unsafe {
        std::slice::from_raw_parts(positions.as_ptr() as *const [f32; 3], vertex_count)
    };
    let opt_indices = meshopt::optimize_vertex_cache(indices, vertex_count);
    let mut opt_indices_mut = opt_indices;
    let opt_vertices = meshopt::optimize_vertex_fetch(&mut opt_indices_mut, vertices_3);
    let new_vertex_count = opt_vertices.len();

    let opt_flat: Vec<f32> = opt_vertices.iter().flat_map(|v| v.iter().copied()).collect();
    let (quant, translation, scale) = quantize_positions(&opt_flat, qbits);

    let mut quant_min = [u16::MAX; 3];
    let mut quant_max = [0u16; 3];
    for q in &quant {
        for d in 0..3 {
            if q[d] < quant_min[d] { quant_min[d] = q[d]; }
            if q[d] > quant_max[d] { quant_max[d] = q[d]; }
        }
    }

    let vertex_data = meshopt::encoding::encode_vertex_buffer(&quant)
        .map_err(|e| format!("meshopt vertex encode (u16): {e}"))?;
    let index_data = meshopt::encoding::encode_index_buffer(&opt_indices_mut, new_vertex_count)
        .map_err(|e| format!("meshopt index encode: {e}"))?;

    Ok(MeshoptQuantEncoded {
        vertex_data,
        index_data,
        indices: opt_indices_mut,
        vertex_count: new_vertex_count,
        translation,
        scale,
        quant_min,
        quant_max,
    })
}

/// Quantize f32 positions to `[u16;4]` (stride-8, meshopt-codec-friendly) over
/// the per-mesh bbox — the KHR_mesh_quantization position layout. Returns the
/// quantized vertices plus the dequant transform stored on the glTF node:
/// `world = translation + scale ⊙ quantized`. `qbits <= 16`. Degenerate axes
/// (range 0) get `scale = 1` and quantize to 0 so the node transform never
/// carries a zero scale (`world = min` exactly).
pub(crate) fn quantize_positions(
    positions: &[f32],
    qbits: u8,
) -> (Vec<[u16; 4]>, [f32; 3], [f32; 3]) {
    let n = positions.len() / 3;
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in 0..n {
        for d in 0..3 {
            let val = positions[v * 3 + d];
            if val < min[d] { min[d] = val; }
            if val > max[d] { max[d] = val; }
        }
    }
    let qmax = ((1u32 << qbits) - 1) as f32;
    let mut scale = [0f32; 3];
    let mut inv = [0f32; 3]; // qmax / range, or 0 for degenerate axes
    for d in 0..3 {
        let range = max[d] - min[d];
        if range > 0.0 {
            scale[d] = range / qmax;
            inv[d] = qmax / range;
        } else {
            scale[d] = 1.0; // no 0-scale node transform; q stays 0 → world = min
            inv[d] = 0.0;
        }
    }
    let mut out = Vec::with_capacity(n);
    for v in 0..n {
        let mut q = [0u16; 4];
        for d in 0..3 {
            let qf = ((positions[v * 3 + d] - min[d]) * inv[d] + 0.5).floor();
            q[d] = qf.clamp(0.0, qmax) as u16;
        }
        out.push(q);
    }
    (out, min, scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_positions_roundtrip_within_half_step() {
        // x range 200, y range 50, z degenerate (all 5.0).
        let positions = vec![
            -100.0f32, 0.0, 5.0,
            100.0, 50.0, 5.0,
            0.0, 25.0, 5.0,
            37.5, 12.3, 5.0,
        ];
        let n = positions.len() / 3;
        let ranges = [200.0f32, 50.0, 0.0];
        for qbits in [8u8, 14, 16] {
            let qmax = ((1u32 << qbits) - 1) as f32;
            let (q, t, s) = quantize_positions(&positions, qbits);
            assert_eq!(q.len(), n);
            for v in 0..n {
                for d in 0..3 {
                    let deq = t[d] + s[d] * q[v][d] as f32;
                    let half = if ranges[d] > 0.0 { ranges[d] / qmax / 2.0 } else { 0.0 };
                    let slack = 1e-3 * ranges[d].max(1.0);
                    assert!(
                        (deq - positions[v * 3 + d]).abs() <= half + slack,
                        "qbits={qbits} v={v} d={d}: deq={deq} want~{} (half={half})",
                        positions[v * 3 + d]
                    );
                }
            }
            // degenerate z axis: scale=1, quant=0, dequant=5.0 exactly (no 0-scale)
            assert_eq!(s[2], 1.0, "degenerate axis must not get a 0 scale");
            assert_eq!(t[2], 5.0);
            for v in 0..n {
                assert_eq!(q[v][2], 0);
                assert_eq!(q[v][3], 0, "4th lane is padding");
            }
        }
    }

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
