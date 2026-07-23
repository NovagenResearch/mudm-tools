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
use std::sync::atomic::{AtomicU64, Ordering};

// Aggregate build-vs-encode profiling (MUDM_DRACO_PROFILE). Per-call printing is
// far too noisy for the NG path (tens of thousands of encodes); instead we
// accumulate and emit a running split every 2000 calls, so even a killed run
// shows the stabilized ratio.
static PROF_ADD_US: AtomicU64 = AtomicU64::new(0);
static PROF_BUILD_US: AtomicU64 = AtomicU64::new(0);
static PROF_ENC_US: AtomicU64 = AtomicU64::new(0);
static PROF_CALLS: AtomicU64 = AtomicU64::new(0);

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
    config: encode::Config,
) -> Result<Vec<u8>, String> {
    // Light profiling: when MUDM_DRACO_PROFILE is set, time build() (dedup +
    // degenerate-filter + remove_unused) vs encode::encode (edgebreaker + rANS)
    // separately. The former temp-OBJ write/parse overhead is gone, so this
    // isolates the remaining build-vs-encode split (informs the skip-dedup call).
    let profile = std::env::var_os("MUDM_DRACO_PROFILE").is_some();
    let n_tris = faces.len();
    let n_verts = positions.len();

    // Time attribute creation SEPARATELY from build(): Attribute::from() runs
    // remove_duplicate_values (an UNCONDITIONAL O(V^2) all-pairs scan) here,
    // BEFORE build() — so without this split it was untimed and invisible.
    let t_add = std::time::Instant::now();
    let mut builder = MeshBuilder::new();
    builder.set_connectivity_attribute(faces);
    let _id: AttributeId = builder.add_attribute(
        positions,
        AttributeType::Position,
        AttributeDomain::Position,
        vec![],
    );
    let add_us = t_add.elapsed().as_micros();

    let t_build = std::time::Instant::now();
    let mesh = builder.build().map_err(|e| format!("MeshBuilder build error: {:?}", e))?;
    let build_us = t_build.elapsed().as_micros();

    let mut buffer: Vec<u8> = Vec::new();
    let t_enc = std::time::Instant::now();
    encode::encode(mesh, &mut buffer, config)
        .map_err(|e| format!("Draco encode error: {:?}", e))?;
    let enc_us = t_enc.elapsed().as_micros();

    if profile {
        let calls = PROF_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        let a = PROF_ADD_US.fetch_add(add_us as u64, Ordering::Relaxed) + add_us as u64;
        let b = PROF_BUILD_US.fetch_add(build_us as u64, Ordering::Relaxed) + build_us as u64;
        let e = PROF_ENC_US.fetch_add(enc_us as u64, Ordering::Relaxed) + enc_us as u64;
        // Flag a pathologically slow SINGLE encode (catches the "stuck on one
        // mesh" case the aggregate misses) — names the phase + mesh size so we
        // can tell build()-dedup vs encode()-compression apart immediately.
        if add_us > 200_000 || build_us > 200_000 || enc_us > 200_000 {
            eprintln!(
                "MUDM_DRACO_PROFILE SLOW call#{}: tris={} verts={} add={}ms build={}ms encode={}ms",
                calls, n_tris, n_verts, add_us / 1000, build_us / 1000, enc_us / 1000,
            );
        }
        if calls % 200 == 0 {
            let tot = (a + b + e).max(1);
            eprintln!(
                "MUDM_DRACO_PROFILE agg: calls={} add={}ms ({:.1}%) build={}ms ({:.1}%) encode={}ms ({:.1}%)",
                calls,
                a / 1000, 100.0 * a as f64 / tot as f64,
                b / 1000, 100.0 * b as f64 / tot as f64,
                e / 1000, 100.0 * e as f64 / tot as f64,
            );
        }
    }
    Ok(buffer)
}

/// Encode a triangle mesh with pre-quantized u32 positions to Draco binary.
///
/// Arguments:
/// - `positions`: flat array of pre-quantized position components [x0,y0,z0, x1,y1,z1, ...]
///   Each value in [0, 2^qbits). Values must be < 2^24 for exact f32 representation.
/// - `indices`: triangle indices (length must be multiple of 3)
/// - `qbits`: the SAME `vertex_quantization_bits` the caller pre-quantized the
///   positions with. Threaded into the Draco encoder so the lossless grid-identity
///   QuantizationCoordinateWise transform spans exactly `[0, 2^qbits - 1]`.
///
/// Returns Draco-encoded bytes.
pub fn encode_draco_mesh(
    positions: &[u32],
    indices: &[u32],
    qbits: u8,
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
    // Sanitize the mesh so the vendored Draco edgebreaker never sees an unused
    // vertex (its corner-table panics on those: draco-oxide core/corner_table).
    // Two ways a vertex gets orphaned: (1) it's simply not referenced by any face;
    // (2) a triangle has two vertices at the SAME QUANTIZED position — `build()`
    // dedups coincident positions, collapsing that triangle to a line, and its
    // degenerate-filter then drops it, freeing the third vertex. The upstream
    // pre-quantization filter (streaming.rs:is_valid_triangle) can't catch case 2.
    // So: drop quant-degenerate faces here, then compact the now-unused vertices.
    let same_pos = |a: usize, b: usize| {
        positions[a * 3] == positions[b * 3]
            && positions[a * 3 + 1] == positions[b * 3 + 1]
            && positions[a * 3 + 2] == positions[b * 3 + 2]
    };
    let mut kept: Vec<[usize; 3]> = Vec::with_capacity(indices.len() / 3);
    for c in indices.chunks_exact(3) {
        let (a, b, cc) = (c[0] as usize, c[1] as usize, c[2] as usize);
        if a >= n_verts || b >= n_verts || cc >= n_verts {
            return Err(format!("index out of range (n_verts={})", n_verts));
        }
        // Keep only triangles with three distinct vertices AND three distinct
        // quantized positions (so they survive build()'s position-dedup).
        if a != b && b != cc && a != cc && !same_pos(a, b) && !same_pos(b, cc) && !same_pos(a, cc) {
            kept.push([a, b, cc]);
        }
    }
    if kept.is_empty() {
        return Err("mesh has no non-degenerate faces after quantization".to_string());
    }
    let mut used = vec![false; n_verts];
    for t in &kept {
        used[t[0]] = true;
        used[t[1]] = true;
        used[t[2]] = true;
    }
    let (pos, faces): (Vec<NdVector<3, f32>>, Vec<[usize; 3]>) = if used.iter().all(|&u| u) {
        let pos = positions
            .chunks_exact(3)
            .map(|c| NdVector::from([c[0] as f32, c[1] as f32, c[2] as f32]))
            .collect();
        (pos, kept)
    } else {
        let mut remap = vec![0usize; n_verts];
        let mut pos: Vec<NdVector<3, f32>> = Vec::with_capacity(n_verts);
        for v in 0..n_verts {
            if used[v] {
                remap[v] = pos.len();
                pos.push(NdVector::from([
                    positions[v * 3] as f32,
                    positions[v * 3 + 1] as f32,
                    positions[v * 3 + 2] as f32,
                ]));
            }
        }
        let faces = kept.iter().map(|t| [remap[t[0]], remap[t[1]], remap[t[2]]]).collect();
        (pos, faces)
    };

    // A1 LOSSLESS, libdraco-conformant NG path:
    // The positions arrive pre-quantized to the integer grid [0, 2^qbits - 1]
    // (the NG assembler quantizes with this same `qbits`). We keep
    // QuantizationCoordinateWise (portabilization id=2, which libdraco/DracoPy and
    // the Neuroglancer WASM viewer decode today) but force its quantization
    // transform to be the IDENTITY over that grid: min=0, range=(2^qbits - 1),
    // bits=qbits. Then encode is round((v-0)/(2^qbits-1) * (2^qbits-1)) = v and
    // libdraco dequant is v * (2^qbits-1)/(2^qbits-1) + 0 = v — EXACT (500 decodes
    // as 500.0, not 500.244). The wire format/metadata layout is unchanged (only
    // the metadata VALUES), so the stream stays conformant.
    //
    // qbits is threaded from the NG caller's `vertex_quantization_bits` (NOT
    // hardcoded) so the grid range always matches the pre-quantization.
    //
    // (The earlier T2b ToBits route is BLOCKED — its id=1 bitstream is not
    // libdraco-decodable; this A1 grid-identity QCW supersedes it.)
    let config = encode::Config::with_ng_lossless(qbits);

    // build() + encode can still panic on degenerate input, so guard them.
    let result = std::panic::catch_unwind(move || build_and_encode(pos, faces, config));

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

    // GLB / 3DTiles consumer (encoder_glb.rs): the viewer wants world-space
    // positions back via Draco's built-in quantization, so this path MUST keep
    // the quantizing Config::default() (QuantizationCoordinateWise). Do NOT route
    // it to ToBits — that would change the GLB bytes and stop re-baking world
    // coords on decode.
    let config = encode::Config::default();

    let result = std::panic::catch_unwind(move || build_and_encode(pos, faces, config));

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

    // Pass the SAME qbits used to bbox-quantize above into the encoder, so the
    // lossless grid-identity transform spans exactly [0, 2^qbits - 1].
    let bytes = encode_draco_mesh(&quant, &indices, qbits)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    pyo3::Python::with_gil(|py| {
        Ok(pyo3::types::PyBytes::new(py, &bytes).into())
    })
}

/// Test-only direct entry into the NG (u32) Draco path.
///
/// Unlike `draco_encode_mesh` (which bbox-quantizes f32 first), this passes the
/// caller's pre-quantized integers straight to `encode_draco_mesh` — exactly what
/// the Neuroglancer multilod assembly does. `qbits` is the
/// `vertex_quantization_bits` the integers were pre-quantized with (default 10,
/// matching `generate_neuroglancer_multilod`). Used by the A1 lossless round-trip
/// tests to prove the NG path stores integers LOSSLESSLY via the grid-identity
/// QuantizationCoordinateWise transform.
#[pyo3::pyfunction]
#[pyo3(signature = (positions, indices, qbits=10))]
pub fn draco_encode_ng_u32(
    positions: Vec<u32>,
    indices: Vec<u32>,
    qbits: u8,
) -> pyo3::PyResult<pyo3::Py<pyo3::types::PyBytes>> {
    let bytes = encode_draco_mesh(&positions, &indices, qbits)
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

        let result = encode_draco_mesh(&positions, &indices, 10);
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

        let result = encode_draco_mesh(&positions, &indices, 10);
        assert!(result.is_ok(), "Draco encoding failed: {:?}", result.err());
        assert!(result.unwrap().len() > 5);
    }

    #[test]
    fn test_encode_empty_fails() {
        let result = encode_draco_mesh(&[], &[], 10);
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

    /// Reference in-memory encode of u32 positions, identical to the production
    /// `encode_draco_mesh` recipe. `encode_draco_mesh` MUST produce these exact
    /// bytes. `qbits` mirrors the production caller so the grid-identity config
    /// matches.
    fn reference_in_memory_u32(positions: &[u32], indices: &[u32], qbits: u8) -> Vec<u8> {
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
        // A1: the NG path uses the LOSSLESS grid-identity QuantizationCoordinateWise
        // (Config::with_ng_lossless(qbits)). Reference must use the same config to
        // byte-match encode_draco_mesh.
        encode::encode(mesh, &mut buf, encode::Config::with_ng_lossless(qbits))
            .expect("reference encode failed");
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

        let got = encode_draco_mesh(&positions, &indices, 10).expect("encode failed");
        let reference = reference_in_memory_u32(&positions, &indices, 10);
        assert_eq!(
            got, reference,
            "encode_draco_mesh must equal the direct in-memory recipe (with_ng_lossless)"
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

        let a = encode_draco_mesh(&positions, &indices, 10).expect("a");
        let b = encode_draco_mesh(&positions, &indices, 10).expect("b");
        let c = encode_draco_mesh(&positions, &indices, 10).expect("c");
        assert_eq!(a, b, "encode is non-deterministic (a != b)");
        assert_eq!(b, c, "encode is non-deterministic (b != c)");
        assert_eq!(&a[..5], b"DRACO", "missing DRACO magic");
    }

    #[test]
    fn test_encode_draco_mesh_compacts_unused_vertices() {
        // Vertex 0 is UNUSED — no triangle references it. This mirrors the NG
        // fragment path, where the degenerate-triangle filter (streaming.rs) can
        // orphan a vertex while it stays in the position array. The vendored Draco
        // edgebreaker corner-table panics on unused vertices; encode_draco_mesh must
        // compact them out and still encode the remaining triangle.
        let positions: &[u32] = &[
            500, 500, 500,   // v0 — UNUSED (orphan)
            0, 0, 0,         // v1
            100, 0, 0,       // v2
            50, 100, 0,      // v3
        ];
        let indices: &[u32] = &[1, 2, 3]; // references v1,v2,v3 only; v0 orphaned
        let got = encode_draco_mesh(positions, indices, 10);
        assert!(
            got.is_ok(),
            "encode_draco_mesh must compact unused vertices, got err: {:?}",
            got.err()
        );
        assert_eq!(&got.unwrap()[..5], b"DRACO", "missing DRACO magic");

        // The real NG pathology: a triangle with two vertices at the SAME quantized
        // position. build()'s position-dedup collapses it to a line, its degenerate-
        // filter drops it, and the third vertex is orphaned → corner-table panic —
        // unless we drop the quant-degenerate face first. Here face 0 (v0==v1) is
        // dropped while the good face 1 still encodes.
        let mix_pos: &[u32] = &[10, 10, 10, 10, 10, 10, 100, 0, 0, 50, 100, 0, 200, 50, 0];
        let mix_idx: &[u32] = &[0, 1, 2, 2, 3, 4];
        let mixed = encode_draco_mesh(mix_pos, mix_idx, 10);
        assert!(
            mixed.is_ok(),
            "must drop quant-degenerate face + keep the good one, got {:?}",
            mixed.err()
        );
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

    /// Re-bake guards.
    ///
    /// f32 / GLB path (FROZEN): captured from the OLD temp-OBJ-bridge encoder
    /// (run 2026-05-31) and CONFIRMED unchanged by the T2 in-memory swap. The A1
    /// lossless-NG change does NOT touch the f32 path (`encode_draco_mesh_f32`
    /// keeps `Config::default()` and never sets the grid-identity override), so
    /// these MUST stay byte-identical. This is the backstop proving the f32 path
    /// is unaffected by construction.
    const OLD_F32_LEN: usize = 2068;
    const OLD_F32_FNV: u64 = 0x9f1998421548f6a8;

    /// u32 / NG path (RE-BAKED for A1 on 2026-05-31): the NG stream bytes
    /// LEGITIMATELY changed when the Position transform became the grid-identity
    /// QuantizationCoordinateWise (Config::with_ng_lossless(qbits)) instead of the
    /// data-bbox quantization. The old (pre-A1, data-bbox) constants were
    /// len=2013 / fnv=0xeda50e757a14d9d5. These NEW constants were deliberately
    /// re-baked ONLY AFTER the f32 guard above was confirmed still GREEN (so the
    /// re-bake cannot mask an unintended f32-path change). qbits=10 here matches
    /// the NG default. If the encoder changes again, these trip and must be
    /// DELIBERATELY re-baked (do not silently bump).
    const OLD_U32_LEN: usize = 101;
    const OLD_U32_FNV: u64 = 0xc933a9922fea5fec;

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
        let bytes = encode_draco_mesh(&positions, &indices, 10).expect("profile encode failed");
        assert!(bytes.len() > 5);
        assert_eq!(&bytes[..5], b"DRACO");
    }

    /// f32 / GLB guard — the HARD gate that must be GREEN before (and after) the
    /// u32 re-bake. The f32 path is untouched by A1.
    #[test]
    fn rebake_guard_f32_byte_identical() {
        let pf = vec![0.0f32, 0.0, 0.0, 10.0, 0.0, 0.0, 5.0, 10.0, 0.0, 5.0, 5.0, 10.0];
        let idxf = vec![0u32, 1, 2, 0, 1, 3, 1, 2, 3, 0, 2, 3];
        let bf = encode_draco_mesh_f32(&pf, &idxf).expect("encf");
        assert_eq!(bf.len(), OLD_F32_LEN, "f32 Draco length changed (must stay byte-identical)");
        assert_eq!(fnv1a(&bf), OLD_F32_FNV, "f32 Draco bytes changed (must stay byte-identical)");
    }

    /// u32 / NG guard — re-baked for A1 (lossless grid-identity QCW). qbits=10.
    #[test]
    fn rebake_guard_u32_ng_lossless() {
        let positions = vec![0u32, 0, 0, 100, 0, 0, 50, 100, 0, 50, 50, 100, 0, 0, 0, 25, 25, 25];
        let indices = vec![0u32, 1, 2, 4, 2, 3, 1, 3, 5, 0, 5, 2];
        let bytes = encode_draco_mesh(&positions, &indices, 10).expect("enc");
        assert_eq!(bytes.len(), OLD_U32_LEN, "u32 Draco length changed unexpectedly");
        assert_eq!(fnv1a(&bytes), OLD_U32_FNV, "u32 Draco bytes changed unexpectedly");
    }

    /// DUPLICATE-HEAVY + unused-vertex guard. The unique-vertex lattice guards
    /// above barely exercise build()'s dedup compaction (remap_attribute /
    /// remove_unused_vertices) — that path only runs when vertices coincide, which
    /// NG position-quantization makes the norm. The vertex array here is a 16x16
    /// base grid duplicated 2x (every vertex i has a coincident twin i+K); ~half
    /// the triangles reference the twin copy, and twins referenced by no face are
    /// dropped by remove_unused_vertices. This LOCKS the encoded bytes so the
    /// O(V^2)->O(V) compaction rewrite of remap_attribute / remove_unused_vertices
    /// stays byte-identical (captured on the post-dedup-fix, pre-compaction-fix
    /// encoder).
    const DUP_HEAVY_LEN: usize = 287;
    const DUP_HEAVY_FNV: u64 = 0x699c351327d1db9f;
    #[test]
    fn rebake_guard_u32_duplicate_heavy() {
        const N: u32 = 16;
        let mut base: Vec<[u32; 3]> = Vec::new();
        for y in 0..N {
            for x in 0..N {
                let z = ((x * 7 + y * 13) % 97) as u32;
                base.push([x * 16, y * 16, z]);
            }
        }
        let k = base.len() as u32;
        let mut positions: Vec<u32> = Vec::with_capacity(base.len() * 6);
        for p in &base {
            positions.extend_from_slice(p);
        }
        for p in &base {
            positions.extend_from_slice(p); // coincident twin at index +k
        }
        let mut indices: Vec<u32> = Vec::new();
        let mut t: u32 = 0;
        for y in 0..N - 1 {
            for x in 0..N - 1 {
                let i = y * N + x;
                let r = i + 1;
                let d = i + N;
                let dr = d + 1;
                // On alternating quads, route the i and d corners to their twin
                // copy so faces reference duplicates; base[i]/[r]/[d]/[dr] are
                // distinct so triangles stay non-degenerate after the collapse.
                let use_twin = t % 2 == 0;
                let tw = |v: u32| if use_twin { v + k } else { v };
                indices.extend_from_slice(&[tw(i), r, tw(d), r, dr, tw(d)]);
                t += 1;
            }
        }
        let bytes = encode_draco_mesh(&positions, &indices, 10).expect("dup-heavy enc");
        assert_eq!(
            bytes.len(),
            DUP_HEAVY_LEN,
            "dup-heavy Draco length changed (build() compaction must stay byte-identical)"
        );
        assert_eq!(
            fnv1a(&bytes),
            DUP_HEAVY_FNV,
            "dup-heavy Draco bytes changed (build() compaction must stay byte-identical)"
        );
    }
}
