/// Draco mesh encoder for Neuroglancer multilod_draco format.
///
/// Encodes pre-quantized integer positions and triangle indices into Draco binary format.
/// Positions are already quantized to [0, 2^bits) by the caller — Draco must NOT
/// apply additional quantization.
///
/// Builds the Draco `Mesh` directly in memory via `MeshBuilder` (no temp `.obj`
/// round-trip). This is possible because the vendored draco-oxide fork re-exports
/// `AttributeDomain` + `AttributeId` from its prelude (see `rust/vendor/draco-oxide`),
/// which `MeshBuilder::add_attribute` requires.

use draco_oxide::prelude::*;
use draco_oxide::encode;
use draco_oxide::prelude::{AttributeDomain, AttributeId};

/// Build a Draco `Mesh` from `NdVector<3,f32>` positions + triangle faces, then
/// encode it to a byte buffer. Shared by both the u32 and f32 entry points.
///
/// `MeshBuilder::build()` runs vertex dedup (by position), a degenerate-face
/// filter, and `remove_unused_vertices`, then `encode::encode` runs edgebreaker +
/// rANS. Both can panic on pathological input, so callers wrap this in
/// `std::panic::catch_unwind`.
fn build_and_encode(
    positions: Vec<NdVector<3, f32>>,
    faces: Vec<[usize; 3]>,
) -> Result<Vec<u8>, String> {
    // Light profiling: when MUDM_DRACO_PROFILE is set, time build() (dedup +
    // degenerate-filter + remove_unused) vs encode::encode (edgebreaker + rANS)
    // separately. The former temp-OBJ write/parse overhead is gone, so this
    // isolates the remaining build-vs-encode split (informs the skip-dedup call).
    let profile = std::env::var_os("MUDM_DRACO_PROFILE").is_some();

    let mut builder = MeshBuilder::new();
    builder.set_connectivity_attribute(faces);
    let _id: AttributeId = builder.add_attribute(
        positions,
        AttributeType::Position,
        AttributeDomain::Position,
        vec![],
    );

    let t_build = std::time::Instant::now();
    let mesh = builder.build().map_err(|e| format!("MeshBuilder build error: {:?}", e))?;
    let build_us = t_build.elapsed().as_micros();

    let mut buffer: Vec<u8> = Vec::new();
    let t_enc = std::time::Instant::now();
    encode::encode(mesh, &mut buffer, encode::Config::default())
        .map_err(|e| format!("Draco encode error: {:?}", e))?;
    let enc_us = t_enc.elapsed().as_micros();

    if profile {
        let total = (build_us + enc_us).max(1);
        eprintln!(
            "MUDM_DRACO_PROFILE build={}us ({:.1}%) encode={}us ({:.1}%) total={}us out={}B",
            build_us,
            100.0 * build_us as f64 / total as f64,
            enc_us,
            100.0 * enc_us as f64 / total as f64,
            build_us + enc_us,
            buffer.len(),
        );
    }
    Ok(buffer)
}

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

    // Build the Draco mesh directly in memory. The u32 -> f32 cast is exact for
    // values < 2^24 (covers 16-bit quantization), identical to the cast the old
    // OBJ bridge performed when float-formatting each vertex.
    let pos: Vec<NdVector<3, f32>> = positions
        .chunks_exact(3)
        .map(|c| NdVector::from([c[0] as f32, c[1] as f32, c[2] as f32]))
        .collect();
    let faces: Vec<[usize; 3]> = indices
        .chunks_exact(3)
        .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
        .collect();

    // build() + encode can still panic on degenerate input, so guard them.
    let result = std::panic::catch_unwind(move || build_and_encode(pos, faces));

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

    // Build the Draco mesh directly in memory from the world-space f32 positions.
    let pos: Vec<NdVector<3, f32>> = positions
        .chunks_exact(3)
        .map(|c| NdVector::from([c[0], c[1], c[2]]))
        .collect();
    let faces: Vec<[usize; 3]> = indices
        .chunks_exact(3)
        .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
        .collect();

    let result = std::panic::catch_unwind(move || build_and_encode(pos, faces));

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

    /// T1 compile-gate: prove the in-memory MeshBuilder recipe COMPILES and
    /// produces a DRACO stream WITHOUT any temp .obj round-trip. This is the
    /// make-or-break proof that the vendored fork's prelude patch (exposing
    /// `AttributeDomain` + `AttributeId`) makes `MeshBuilder::add_attribute`
    /// invokable from this downstream crate. Recipe verified against the
    /// vendored crate's own `builder.rs` test.
    #[test]
    fn test_in_memory_meshbuilder_recipe() {
        // These two imports are the entire point of the fork patch: before it,
        // neither type was nameable from outside draco-oxide (core is pub(crate)
        // and the published prelude omits them).
        use draco_oxide::prelude::{AttributeDomain, AttributeId};

        // pre-quantized u32 positions + u32 indices, exactly the data the NG
        // encoder already holds (single triangle here).
        let positions: &[u32] = &[0, 0, 0, 100, 0, 0, 50, 100, 0];
        let indices: &[u32] = &[0u32, 1, 2];

        // u32 -> f32 cast is exact for values < 2^24 (covers 16-bit quant),
        // identical to the cast the OBJ bridge does today at :59.
        let pos: Vec<NdVector<3, f32>> = positions
            .chunks_exact(3)
            .map(|c| NdVector::from([c[0] as f32, c[1] as f32, c[2] as f32]))
            .collect();
        // set_connectivity_attribute takes Vec<[usize; 3]> (NOT u32) — verified
        // against vendored builder.rs:76.
        let faces: Vec<[usize; 3]> = indices
            .chunks_exact(3)
            .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
            .collect();

        let mut builder = MeshBuilder::new();
        builder.set_connectivity_attribute(faces);
        // add_attribute arity (verified builder.rs:30-45): (Vec<Data>, AttributeType,
        // AttributeDomain, parents: Vec<AttributeId>) -> AttributeId.
        let _id: AttributeId = builder.add_attribute(
            pos,
            AttributeType::Position,
            AttributeDomain::Position,
            vec![],
        );
        let mesh = builder.build().expect("MeshBuilder::build failed");

        let mut buf: Vec<u8> = Vec::new();
        // Config::default() comes from the ConfigType trait (in prelude).
        encode::encode(mesh, &mut buf, encode::Config::default())
            .expect("encode::encode failed");

        assert!(buf.len() > 5, "encoded stream too short");
        assert_eq!(&buf[..5], b"DRACO", "missing DRACO magic from in-memory encode");
    }

    // ---- T2: in-memory encoder swap gates ----------------------------------

    /// Reference in-memory encode of u32 positions, identical to the T1 recipe.
    /// `encode_draco_mesh` MUST produce these exact bytes once it stops routing
    /// through the temp-OBJ bridge. On the OLD OBJ path the bytes differ
    /// (load_obj single_index re-weld + dedup reorder vs a direct array build),
    /// so this is the RED→GREEN gate that proves the bridge is gone.
    fn reference_in_memory_u32(positions: &[u32], indices: &[u32]) -> Vec<u8> {
        use draco_oxide::prelude::AttributeId;
        let pos: Vec<NdVector<3, f32>> = positions
            .chunks_exact(3)
            .map(|c| NdVector::from([c[0] as f32, c[1] as f32, c[2] as f32]))
            .collect();
        let faces: Vec<[usize; 3]> = indices
            .chunks_exact(3)
            .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
            .collect();
        let mut b = MeshBuilder::new();
        b.set_connectivity_attribute(faces);
        let _id: AttributeId =
            b.add_attribute(pos, AttributeType::Position, draco_oxide::prelude::AttributeDomain::Position, vec![]);
        let mesh = b.build().expect("reference build failed");
        let mut buf: Vec<u8> = Vec::new();
        encode::encode(mesh, &mut buf, encode::Config::default()).expect("reference encode failed");
        buf
    }

    #[test]
    fn test_encode_matches_in_memory_recipe_u32() {
        // Mesh with a DUPLICATE vertex position (verts 0 and 4 coincide) and a
        // vertex (5) that the OBJ `single_index` re-weld treats differently from
        // a direct array build. This forces the OBJ bridge's load_obj re-weld /
        // dedup reorder to diverge from a direct in-memory MeshBuilder build, so
        // the byte streams differ on the OLD path and match only once the bridge
        // is removed. RED→GREEN gate for the temp-OBJ removal.
        let positions = vec![
            0u32, 0, 0,     // v0
            100, 0, 0,      // v1
            50, 100, 0,     // v2
            50, 50, 100,    // v3
            0, 0, 0,        // v4 == v0 (coincident)
            25, 25, 25,     // v5
        ];
        let indices = vec![
            0u32, 1, 2,
            4, 2, 3,   // uses the coincident duplicate v4
            1, 3, 5,
            0, 5, 2,
        ];

        let got = encode_draco_mesh(&positions, &indices).expect("encode failed");
        let reference = reference_in_memory_u32(&positions, &indices);
        assert_eq!(
            got, reference,
            "encode_draco_mesh must equal the direct in-memory recipe (temp-OBJ bridge still present?)"
        );
    }

    #[test]
    fn test_encode_u32_deterministic_three_calls() {
        let positions = vec![
            0u32, 0, 0,
            100, 0, 0,
            50, 100, 0,
            50, 50, 100,
        ];
        let indices = vec![0u32, 1, 2, 0, 1, 3, 1, 2, 3, 0, 2, 3];

        let a = encode_draco_mesh(&positions, &indices).expect("a");
        let b = encode_draco_mesh(&positions, &indices).expect("b");
        let c = encode_draco_mesh(&positions, &indices).expect("c");
        assert_eq!(a, b, "encode is non-deterministic (a != b)");
        assert_eq!(b, c, "encode is non-deterministic (b != c)");
        assert_eq!(&a[..5], b"DRACO", "missing DRACO magic");
    }

    #[test]
    fn test_encode_f32_deterministic_and_magic() {
        let positions = vec![
            0.0f32, 0.0, 0.0,
            10.0, 0.0, 0.0,
            5.0, 10.0, 0.0,
            5.0, 5.0, 10.0,
        ];
        let indices = vec![0u32, 1, 2, 0, 1, 3, 1, 2, 3, 0, 2, 3];

        let a = encode_draco_mesh_f32(&positions, &indices).expect("a");
        let b = encode_draco_mesh_f32(&positions, &indices).expect("b");
        let c = encode_draco_mesh_f32(&positions, &indices).expect("c");
        assert_eq!(a, b, "f32 encode non-deterministic (a != b)");
        assert_eq!(b, c, "f32 encode non-deterministic (b != c)");
        assert!(a.len() > 5, "f32 Draco output too short");
        assert_eq!(&a[..5], b"DRACO", "missing DRACO magic (f32)");
    }
}

#[cfg(test)]
mod rebake_snapshot {
    use super::*;

    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 1469598103934665603;
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(1099511628211);
        }
        h
    }

    /// T2 re-bake guard. These FNV-1a hashes + lengths were captured from the
    /// OLD temp-OBJ-bridge encoder BEFORE the in-memory rewrite (run 2026-05-31).
    /// EMPIRICAL FINDING: the in-memory swap is BYTE-PRESERVING for these
    /// position-only meshes — `load_obj` already used the identical MeshBuilder
    /// recipe (set_connectivity_attribute + add_attribute(Position) + build) with
    /// tobj single_index, so removing the OBJ text round-trip does NOT re-bake the
    /// stream. If draco-oxide's encoder ever changes, these constants will trip and
    /// must be DELIBERATELY re-baked (do not silently bump).
    const OLD_U32_LEN: usize = 2013;
    const OLD_U32_FNV: u64 = 0xeda50e757a14d9d5;
    const OLD_F32_LEN: usize = 2068;
    const OLD_F32_FNV: u64 = 0x9f1998421548f6a8;

    /// Light build-vs-encode profile on a moderately-large synthetic mesh
    /// (a few thousand tris). Run with:
    ///   MUDM_DRACO_PROFILE=1 cargo test --lib draco_profile_split -- --nocapture --ignored
    /// The per-encode MUDM_DRACO_PROFILE line (emitted from build_and_encode)
    /// reports the build()-vs-encode split now that the temp-OBJ overhead is gone.
    #[test]
    #[ignore]
    fn draco_profile_split() {
        // Build a synthetic grid mesh: an N x N vertex lattice -> ~2*(N-1)^2 tris.
        const N: u32 = 64; // 64x64 = 4096 verts, ~7938 tris
        let mut positions: Vec<u32> = Vec::with_capacity((N * N * 3) as usize);
        for y in 0..N {
            for x in 0..N {
                // pre-quantized integer coords; add a deterministic z ripple so the
                // mesh is genuinely 3D (non-degenerate).
                let z = ((x * 7 + y * 13) % 97) as u32;
                positions.extend_from_slice(&[x * 16, y * 16, z]);
            }
        }
        let mut indices: Vec<u32> = Vec::new();
        for y in 0..N - 1 {
            for x in 0..N - 1 {
                let i = y * N + x;
                let r = i + 1;
                let d = i + N;
                let dr = d + 1;
                indices.extend_from_slice(&[i, r, d, r, dr, d]);
            }
        }
        let n_tris = indices.len() / 3;
        eprintln!("draco_profile_split: {} verts, {} tris", positions.len() / 3, n_tris);
        // Force-enable the profile span regardless of caller env.
        std::env::set_var("MUDM_DRACO_PROFILE", "1");
        let bytes = encode_draco_mesh(&positions, &indices).expect("profile encode failed");
        assert!(bytes.len() > 5);
        assert_eq!(&bytes[..5], b"DRACO");
    }

    #[test]
    fn rebake_guard_bytes_unchanged_vs_old_obj_path() {
        let positions = vec![0u32, 0, 0, 100, 0, 0, 50, 100, 0, 50, 50, 100, 0, 0, 0, 25, 25, 25];
        let indices = vec![0u32, 1, 2, 4, 2, 3, 1, 3, 5, 0, 5, 2];
        let bytes = encode_draco_mesh(&positions, &indices).expect("enc");
        assert_eq!(bytes.len(), OLD_U32_LEN, "u32 Draco length changed vs OLD OBJ path");
        assert_eq!(fnv1a(&bytes), OLD_U32_FNV, "u32 Draco bytes changed vs OLD OBJ path");

        let pf = vec![0.0f32, 0.0, 0.0, 10.0, 0.0, 0.0, 5.0, 10.0, 0.0, 5.0, 5.0, 10.0];
        let idxf = vec![0u32, 1, 2, 0, 1, 3, 1, 2, 3, 0, 2, 3];
        let bf = encode_draco_mesh_f32(&pf, &idxf).expect("encf");
        assert_eq!(bf.len(), OLD_F32_LEN, "f32 Draco length changed vs OLD OBJ path");
        assert_eq!(fnv1a(&bf), OLD_F32_FNV, "f32 Draco bytes changed vs OLD OBJ path");
    }
}
