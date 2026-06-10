//! MMC v1 ("mudm mesh codec") — experimental position-only triangle-mesh codec.
//!
//! A from-scratch alternative to the Draco (NG multilod) and meshopt (GLB) paths,
//! designed around what those paths actually carry: POSITION + indices, nothing
//! else. Keep the conformant Draco/meshopt outputs for stock viewers; use MMC
//! where we control the decoder.
//!
//! Architecture
//! ============
//! * Connectivity: the meshopt index codec (already a dependency; ~1 B/tri
//!   pre-LZ; WASM/JS decoder already ships in viewer ecosystems), wrapped in
//!   zstd. Post-zstd: 0.00–0.27 B/tri on benchmark meshes.
//! * Positions: parallelogram prediction driven by the *decoded* (canonical)
//!   triangle stream — no edgebreaker, no corner table, no panics. Vertices are
//!   emitted in first-seen order of the canonical stream. Each new vertex is
//!   predicted as `pred = pos[o1] + pos[o2] - pos[w]` where (o1, o2) are the
//!   already-known vertices of its triangle and `w` is the vertex opposite the
//!   shared edge (o2, o1) recorded from an earlier triangle; fallbacks: known
//!   neighbor -> last decoded vertex -> 0. Residuals are zigzagged and stored
//!   as byte-planes (component-major), then zstd.
//!
//! Both the NG path (pre-quantized u32 grid, LOSSLESS — `encode_mmc_u32`) and
//! the GLB path (f32 world coords, bbox-quantized with the dequant transform in
//! the header — `encode_mmc_f32`) are supported.
//!
//! Reference implementation + format spec: tools/mmc/mmc.py (Python, bit-exact
//! on decode). Benchmarks vs libdraco cl=7 and the current meshopt pipeline:
//! docs/mesh_codec_review.md. Headline (q10 terrain tile, 130k tris): MMC
//! 18.8 KB vs Draco 33.1 KB vs meshopt-raw-f32 328.7 KB.
//!
//! Layout (little-endian)
//! ======================
//! ```text
//! offset size  field
//! 0      4     magic "MMC1"
//! 4      1     version (1)
//! 5      1     flags: bit0 = HAS_DEQUANT
//! 6      1     qbits (1..=24)
//! 7      1     zstd level used (informational)
//! 8      4     n_verts u32 (post dedup/compaction)
//! 12     4     n_tris  u32 (post degenerate-drop)
//! [HAS_DEQUANT: 24 bytes = min xyz f32le, scale xyz f32le; world = min + q*scale]
//! 4            idx_zstd_len u32
//! 4            pos_zstd_len u32
//! ...          idx zstd frame (meshopt index codec) | pos zstd frame (planes)
//! ```
//!
//! Residual planes: `B = ceil((qbits + 2) / 8)` bytes per component. Stored as
//! `3 * B` planes of `n_verts` bytes: component-major, plane-minor,
//! vertex-major within a plane.

use ahash::AHashMap;

const MAGIC: &[u8; 4] = b"MMC1";
const VERSION: u8 = 1;
const FLAG_DEQUANT: u8 = 1;
#[allow(dead_code)]
pub const DEFAULT_ZSTD_LEVEL: i32 = 3;

/// Decoded MMC mesh.
pub enum MmcMesh {
    /// NG mode: lossless quantized grid positions (flat xyz) + triangle indices.
    Grid { positions: Vec<u32>, indices: Vec<u32>, qbits: u8 },
    /// Dequant mode: world-space f32 positions (flat xyz) + triangle indices.
    World { positions: Vec<f32>, indices: Vec<u32>, qbits: u8 },
}

// ---------------------------------------------------------------------------
// sanitize: dedup coincident quantized vertices, drop degenerate faces,
// compact unused vertices in FIRST-USE order of the index stream.
// Mirrors the semantics of the existing Draco NG path (encoder_draco.rs).
// First-use order matters: the meshopt index codec encodes ascending-first-use
// ids most cheaply (it is what optimize_vertex_fetch produces).
// ---------------------------------------------------------------------------
fn sanitize(positions: &[u32], indices: &[u32]) -> Result<(Vec<[u32; 3]>, Vec<u32>), String> {
    let n_verts = positions.len() / 3;
    if positions.len() % 3 != 0 {
        return Err("positions length must be multiple of 3".into());
    }
    if indices.len() % 3 != 0 {
        return Err("indices length must be multiple of 3".into());
    }

    // dedup rank by first occurrence in the vertex array
    let mut rank_of: AHashMap<[u32; 3], u32> = AHashMap::with_capacity(n_verts);
    let mut vert_rank: Vec<u32> = Vec::with_capacity(n_verts);
    let mut uniq: Vec<[u32; 3]> = Vec::with_capacity(n_verts);
    for v in 0..n_verts {
        let key = [positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]];
        let rank = *rank_of.entry(key).or_insert_with(|| {
            uniq.push(key);
            (uniq.len() - 1) as u32
        });
        vert_rank.push(rank);
    }

    // degenerate drop on dedup ranks (same rank == same quantized position)
    let mut kept: Vec<[u32; 3]> = Vec::with_capacity(indices.len() / 3);
    for t in indices.chunks_exact(3) {
        let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
        if a >= n_verts || b >= n_verts || c >= n_verts {
            return Err(format!("index out of range (n_verts={})", n_verts));
        }
        let (ra, rb, rc) = (vert_rank[a], vert_rank[b], vert_rank[c]);
        if ra != rb && rb != rc && ra != rc {
            kept.push([ra, rb, rc]);
        }
    }
    if kept.is_empty() {
        return Err("mesh has no non-degenerate faces after quantization".into());
    }

    // first-use compaction
    let mut slot_of: Vec<u32> = vec![u32::MAX; uniq.len()];
    let mut verts: Vec<[u32; 3]> = Vec::with_capacity(uniq.len());
    let mut tris: Vec<u32> = Vec::with_capacity(kept.len() * 3);
    for t in &kept {
        for &r in t {
            let s = slot_of[r as usize];
            let s = if s == u32::MAX {
                let s = verts.len() as u32;
                verts.push(uniq[r as usize]);
                slot_of[r as usize] = s;
                s
            } else {
                s
            };
            tris.push(s);
        }
    }
    Ok((verts, tris))
}

// ---------------------------------------------------------------------------
// prediction walk (shared by encode/decode)
//
// Walks the canonical triangle stream in order. The k-th distinct vertex seen
// corresponds to the k-th residual slot. `resolve(slot, pred)` returns the
// actual position: encode records the residual and returns the true position,
// decode adds the residual to the prediction.
// ---------------------------------------------------------------------------
fn predict_walk<F: FnMut(usize, [i64; 3]) -> [i64; 3]>(
    canon: &[u32],
    n_verts: usize,
    qbits: u8,
    mut resolve: F,
) -> Result<Vec<[i64; 3]>, String> {
    let qmax = (1i64 << qbits) - 1;
    let mut pos: Vec<[i64; 3]> = vec![[0; 3]; n_verts];
    let mut known: Vec<bool> = vec![false; n_verts];
    let mut edge: AHashMap<(u32, u32), u32> = AHashMap::with_capacity(canon.len());
    let mut last = [0i64; 3];
    let mut n_seen = 0usize;

    for t in canon.chunks_exact(3) {
        let (ta, tb, tc) = (t[0], t[1], t[2]);
        for &(x, o1, o2) in &[(ta, tb, tc), (tb, tc, ta), (tc, ta, tb)] {
            let xi = x as usize;
            if xi >= n_verts || (o1 as usize) >= n_verts || (o2 as usize) >= n_verts {
                return Err("decoded index out of range".into());
            }
            if !known[xi] {
                let (o1i, o2i) = (o1 as usize, o2 as usize);
                let mut p = if known[o1i] && known[o2i] {
                    match edge.get(&(o2, o1)) {
                        Some(&w) if known[w as usize] => {
                            let (a, b, w) = (pos[o1i], pos[o2i], pos[w as usize]);
                            [a[0] + b[0] - w[0], a[1] + b[1] - w[1], a[2] + b[2] - w[2]]
                        }
                        _ => pos[o1i],
                    }
                } else if known[o1i] {
                    pos[o1i]
                } else if known[o2i] {
                    pos[o2i]
                } else {
                    last
                };
                for c in &mut p {
                    *c = (*c).clamp(0, qmax);
                }
                let actual = resolve(n_seen, p);
                pos[xi] = actual;
                known[xi] = true;
                last = actual;
                n_seen += 1;
            }
        }
        edge.entry((ta, tb)).or_insert(tc);
        edge.entry((tb, tc)).or_insert(ta);
        edge.entry((tc, ta)).or_insert(tb);
    }

    if n_seen != n_verts {
        return Err("index stream does not reference all vertices".into());
    }
    Ok(pos)
}

fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

fn unzigzag(z: u64) -> i64 {
    ((z >> 1) as i64) ^ -((z & 1) as i64)
}

fn plane_bytes(qbits: u8) -> usize {
    (qbits as usize + 2 + 7) / 8
}

// ---------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------
fn encode_core(
    verts: &[[u32; 3]],
    tris: &[u32],
    qbits: u8,
    dequant: Option<([f32; 3], [f32; 3])>,
    level: i32,
) -> Result<Vec<u8>, String> {
    let n_verts = verts.len();
    let n_tris = tris.len() / 3;

    let ib = meshopt::encoding::encode_index_buffer(tris, n_verts)
        .map_err(|e| format!("meshopt index encode: {e}"))?;
    // canonical stream: what every decoder will see (meshopt may rotate tris)
    let canon: Vec<u32> = meshopt::encoding::decode_index_buffer::<u32>(&ib, tris.len())
        .map_err(|e| format!("meshopt index decode (canon): {e}"))?;

    // prediction walk -> residuals in emission-slot order
    let mut residuals: Vec<[i64; 3]> = vec![[0; 3]; n_verts];
    {
        // slot -> true position lookup is resolved during the walk: we need the
        // true position of vertex `x`, which is verts[x] (ids are stable).
        // resolve() receives only (slot, pred), so map slot->vertex by walking:
        // the walk visits vertices in canonical first-seen order; we recover the
        // vertex id from the canon stream by tracking it ourselves.
        let mut seen: Vec<bool> = vec![false; n_verts];
        let mut slot_vertex: Vec<u32> = Vec::with_capacity(n_verts);
        for t in canon.chunks_exact(3) {
            for &(x, _, _) in &[(t[0], t[1], t[2]), (t[1], t[2], t[0]), (t[2], t[0], t[1])] {
                let xi = x as usize;
                if xi >= n_verts {
                    return Err("canonical index out of range".into());
                }
                if !seen[xi] {
                    seen[xi] = true;
                    slot_vertex.push(x);
                }
            }
        }
        if slot_vertex.len() != n_verts {
            return Err("index stream does not reference all vertices".into());
        }
        predict_walk(&canon, n_verts, qbits, |slot, pred| {
            let v = verts[slot_vertex[slot] as usize];
            let actual = [v[0] as i64, v[1] as i64, v[2] as i64];
            residuals[slot] = [
                actual[0] - pred[0],
                actual[1] - pred[1],
                actual[2] - pred[2],
            ];
            actual
        })?;
    }

    // byte-plane transpose
    let bpc = plane_bytes(qbits);
    let mut pos_raw: Vec<u8> = vec![0; 3 * bpc * n_verts];
    for c in 0..3 {
        for b in 0..bpc {
            let plane = &mut pos_raw[(c * bpc + b) * n_verts..(c * bpc + b + 1) * n_verts];
            for (i, r) in residuals.iter().enumerate() {
                plane[i] = ((zigzag(r[c]) >> (8 * b)) & 0xFF) as u8;
            }
        }
    }

    let idx_z = zstd::bulk::compress(&ib, level).map_err(|e| format!("zstd idx: {e}"))?;
    let pos_z = zstd::bulk::compress(&pos_raw, level).map_err(|e| format!("zstd pos: {e}"))?;

    let mut out: Vec<u8> = Vec::with_capacity(40 + idx_z.len() + pos_z.len());
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.push(if dequant.is_some() { FLAG_DEQUANT } else { 0 });
    out.push(qbits);
    out.push(level.clamp(0, 255) as u8);
    out.extend_from_slice(&(n_verts as u32).to_le_bytes());
    out.extend_from_slice(&(n_tris as u32).to_le_bytes());
    if let Some((mins, scale)) = dequant {
        for v in mins.iter().chain(scale.iter()) {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out.extend_from_slice(&(idx_z.len() as u32).to_le_bytes());
    out.extend_from_slice(&(pos_z.len() as u32).to_le_bytes());
    out.extend_from_slice(&idx_z);
    out.extend_from_slice(&pos_z);
    Ok(out)
}

/// NG path: LOSSLESS encode of pre-quantized u32 grid positions in
/// `[0, 2^qbits)`. Same input contract as `encode_draco_mesh`.
pub fn encode_mmc_u32(
    positions: &[u32],
    indices: &[u32],
    qbits: u8,
    level: i32,
) -> Result<Vec<u8>, String> {
    if positions.is_empty() || indices.is_empty() {
        return Err("Empty mesh".into());
    }
    if !(1..=24).contains(&qbits) {
        return Err("qbits must be in 1..=24".into());
    }
    let limit = 1u32 << qbits;
    if positions.iter().any(|&p| p >= limit) {
        return Err("position exceeds 2^qbits".into());
    }
    let (verts, tris) = sanitize(positions, indices)?;
    encode_core(&verts, &tris, qbits, None, level)
}

/// GLB path: bbox-quantize f32 world positions to `qbits` (lossy, like Draco's
/// internal quantization) and store the dequant transform in the header.
/// Quantization: `q = clamp(floor((v - min)/range * (2^qbits - 1) + 0.5), 0, qmax)`.
pub fn encode_mmc_f32(
    positions: &[f32],
    indices: &[u32],
    qbits: u8,
    level: i32,
) -> Result<Vec<u8>, String> {
    if positions.is_empty() || indices.is_empty() {
        return Err("Empty mesh".into());
    }
    if !(1..=24).contains(&qbits) {
        return Err("qbits must be in 1..=24".into());
    }
    if positions.len() % 3 != 0 {
        return Err("positions length must be multiple of 3".into());
    }
    if positions.iter().any(|v| !v.is_finite()) {
        return Err("non-finite position".into());
    }
    let n = positions.len() / 3;
    let mut mins = [f32::INFINITY; 3];
    let mut maxs = [f32::NEG_INFINITY; 3];
    for i in 0..n {
        for d in 0..3 {
            let v = positions[i * 3 + d];
            if v < mins[d] {
                mins[d] = v;
            }
            if v > maxs[d] {
                maxs[d] = v;
            }
        }
    }
    let qmax = ((1u32 << qbits) - 1) as f64;
    let mut rng = [0f64; 3];
    let mut scale = [0f32; 3];
    for d in 0..3 {
        rng[d] = maxs[d] as f64 - mins[d] as f64;
        scale[d] = if rng[d] > 0.0 { (rng[d] / qmax) as f32 } else { 0.0 };
    }
    let mut quant: Vec<u32> = Vec::with_capacity(positions.len());
    for i in 0..n {
        for d in 0..3 {
            let q = if rng[d] > 0.0 {
                (((positions[i * 3 + d] as f64 - mins[d] as f64) / rng[d]) * qmax + 0.5)
                    .floor()
                    .clamp(0.0, qmax) as u32
            } else {
                0
            };
            quant.push(q);
        }
    }
    let (verts, tris) = sanitize(&quant, indices)?;
    encode_core(&verts, &tris, qbits, Some((mins, scale)), level)
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------
fn read_u32(data: &[u8], off: usize) -> Result<u32, String> {
    data.get(off..off + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or_else(|| "truncated stream".to_string())
}

fn read_f32(data: &[u8], off: usize) -> Result<f32, String> {
    read_u32(data, off).map(f32::from_bits)
}

/// Decode an MMC stream. Returns grid u32 positions (NG mode) or world f32
/// positions (dequant mode), plus the canonical triangle indices.
pub fn decode_mmc(data: &[u8]) -> Result<MmcMesh, String> {
    if data.len() < 24 || &data[0..4] != MAGIC {
        return Err("bad magic".into());
    }
    if data[4] != VERSION {
        return Err(format!("unsupported version {}", data[4]));
    }
    let flags = data[5];
    let qbits = data[6];
    if !(1..=24).contains(&qbits) {
        return Err("bad qbits".into());
    }
    let n_verts = read_u32(data, 8)? as usize;
    let n_tris = read_u32(data, 12)? as usize;
    if n_verts == 0 || n_tris == 0 || n_verts > (1 << 31) || n_tris > (1 << 31) {
        return Err("bad counts".into());
    }
    let mut off = 16usize;
    let dequant = if flags & FLAG_DEQUANT != 0 {
        let mut mins = [0f32; 3];
        let mut scale = [0f32; 3];
        for d in 0..3 {
            mins[d] = read_f32(data, off + 4 * d)?;
            scale[d] = read_f32(data, off + 12 + 4 * d)?;
        }
        off += 24;
        Some((mins, scale))
    } else {
        None
    };
    let idx_len = read_u32(data, off)? as usize;
    let pos_len = read_u32(data, off + 4)? as usize;
    off += 8;
    let idx_end = off.checked_add(idx_len).ok_or("overflow")?;
    let pos_end = idx_end.checked_add(pos_len).ok_or("overflow")?;
    if pos_end > data.len() {
        return Err("truncated stream".into());
    }

    // connectivity
    let ib = zstd::bulk::decompress(&data[off..idx_end], 16 * n_tris + 1024)
        .map_err(|e| format!("zstd idx: {e}"))?;
    let canon: Vec<u32> = meshopt::encoding::decode_index_buffer::<u32>(&ib, 3 * n_tris)
        .map_err(|e| format!("meshopt index decode: {e}"))?;

    // positions
    let bpc = plane_bytes(qbits);
    let pos_raw = zstd::bulk::decompress(&data[idx_end..pos_end], 3 * bpc * n_verts)
        .map_err(|e| format!("zstd pos: {e}"))?;
    if pos_raw.len() != 3 * bpc * n_verts {
        return Err("position payload size mismatch".into());
    }
    // un-transpose into slot-ordered residuals
    let mut residuals: Vec<[i64; 3]> = vec![[0; 3]; n_verts];
    for c in 0..3 {
        for b in 0..bpc {
            let plane = &pos_raw[(c * bpc + b) * n_verts..(c * bpc + b + 1) * n_verts];
            for (i, &byte) in plane.iter().enumerate() {
                residuals[i][c] |= (byte as i64) << (8 * b);
            }
        }
    }
    for r in &mut residuals {
        for c in r.iter_mut() {
            *c = unzigzag(*c as u64);
        }
    }

    let pos = predict_walk(&canon, n_verts, qbits, |slot, pred| {
        let r = residuals[slot];
        [pred[0] + r[0], pred[1] + r[1], pred[2] + r[2]]
    })?;

    // walk emits positions per vertex id; flatten id-ordered
    let qmax = (1i64 << qbits) - 1;
    if let Some((mins, scale)) = dequant {
        let mut out: Vec<f32> = Vec::with_capacity(n_verts * 3);
        for p in &pos {
            for d in 0..3 {
                let q = p[d].clamp(0, qmax) as f64;
                out.push((mins[d] as f64 + q * scale[d] as f64) as f32);
            }
        }
        Ok(MmcMesh::World { positions: out, indices: canon, qbits })
    } else {
        let mut out: Vec<u32> = Vec::with_capacity(n_verts * 3);
        for p in &pos {
            for d in 0..3 {
                if p[d] < 0 || p[d] > qmax {
                    return Err("decoded position out of grid range".into());
                }
                out.push(p[d] as u32);
            }
        }
        Ok(MmcMesh::Grid { positions: out, indices: canon, qbits })
    }
}

// ---------------------------------------------------------------------------
// Python API
// ---------------------------------------------------------------------------

/// Python-exposed: encode f32 positions + u32 indices -> MMC bytes (lossy,
/// bbox-quantized to `qbits`, dequant transform stored). GLB-path analogue of
/// `draco_encode_mesh`.
#[pyo3::pyfunction]
#[pyo3(signature = (positions, indices, qbits=14, level=3))]
pub fn mmc_encode_mesh(
    positions: Vec<f32>,
    indices: Vec<u32>,
    qbits: u8,
    level: i32,
) -> pyo3::PyResult<pyo3::Py<pyo3::types::PyBytes>> {
    let bytes = encode_mmc_f32(&positions, &indices, qbits, level)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    pyo3::Python::with_gil(|py| Ok(pyo3::types::PyBytes::new(py, &bytes).into()))
}

/// Python-exposed: LOSSLESS encode of pre-quantized u32 grid positions —
/// NG-path analogue of `draco_encode_ng_u32`.
#[pyo3::pyfunction]
#[pyo3(signature = (positions, indices, qbits=10, level=3))]
pub fn mmc_encode_ng_u32(
    positions: Vec<u32>,
    indices: Vec<u32>,
    qbits: u8,
    level: i32,
) -> pyo3::PyResult<pyo3::Py<pyo3::types::PyBytes>> {
    let bytes = encode_mmc_u32(&positions, &indices, qbits, level)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    pyo3::Python::with_gil(|py| Ok(pyo3::types::PyBytes::new(py, &bytes).into()))
}

/// Python-exposed: decode MMC bytes -> dict with `positions`, `indices`,
/// `lossless` (true = u32 grid positions), `qbits`.
#[pyo3::pyfunction]
pub fn mmc_decode(
    py: pyo3::Python<'_>,
    data: Vec<u8>,
) -> pyo3::PyResult<pyo3::Py<pyo3::types::PyDict>> {
    use pyo3::types::{PyDict, PyDictMethods};
    let mesh = decode_mmc(&data).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let dict = PyDict::new(py);
    match mesh {
        MmcMesh::Grid { positions, indices, qbits } => {
            dict.set_item("positions", positions)?;
            dict.set_item("indices", indices)?;
            dict.set_item("lossless", true)?;
            dict.set_item("qbits", qbits)?;
        }
        MmcMesh::World { positions, indices, qbits } => {
            dict.set_item("positions", positions)?;
            dict.set_item("indices", indices)?;
            dict.set_item("lossless", false)?;
            dict.set_item("qbits", qbits)?;
        }
    }
    Ok(dict.unbind())
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Canonical mesh form independent of vertex labels and triangle rotation:
    /// set of triangles as cyclically-minimal position triples (winding kept).
    fn canon_set(positions: &[u32], indices: &[u32]) -> BTreeSet<[[u32; 3]; 3]> {
        let p = |i: u32| -> [u32; 3] {
            let i = i as usize;
            [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]]
        };
        let mut out = BTreeSet::new();
        for t in indices.chunks_exact(3) {
            let tri = [p(t[0]), p(t[1]), p(t[2])];
            let rots = [
                [tri[0], tri[1], tri[2]],
                [tri[1], tri[2], tri[0]],
                [tri[2], tri[0], tri[1]],
            ];
            out.insert(*rots.iter().min().unwrap());
        }
        out
    }

    fn nondegenerate_set(positions: &[u32], indices: &[u32]) -> BTreeSet<[[u32; 3]; 3]> {
        let p = |i: u32| -> [u32; 3] {
            let i = i as usize;
            [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]]
        };
        let mut out = BTreeSet::new();
        for t in indices.chunks_exact(3) {
            let tri = [p(t[0]), p(t[1]), p(t[2])];
            if tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2] {
                let rots = [
                    [tri[0], tri[1], tri[2]],
                    [tri[1], tri[2], tri[0]],
                    [tri[2], tri[0], tri[1]],
                ];
                out.insert(*rots.iter().min().unwrap());
            }
        }
        out
    }

    fn lattice(n: u32) -> (Vec<u32>, Vec<u32>) {
        let mut pos = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let z = (x * 7 + y * 13) % 97;
                pos.extend_from_slice(&[x * 16, y * 16, z]);
            }
        }
        let mut idx = Vec::new();
        for y in 0..n - 1 {
            for x in 0..n - 1 {
                let i = y * n + x;
                idx.extend_from_slice(&[i, i + 1, i + n, i + 1, i + n + 1, i + n]);
            }
        }
        (pos, idx)
    }

    #[test]
    fn test_roundtrip_single_triangle() {
        let pos = vec![0u32, 0, 0, 100, 0, 0, 50, 100, 0];
        let idx = vec![0u32, 1, 2];
        let blob = encode_mmc_u32(&pos, &idx, 10, 3).expect("encode");
        assert_eq!(&blob[..4], b"MMC1");
        match decode_mmc(&blob).expect("decode") {
            MmcMesh::Grid { positions, indices, qbits } => {
                assert_eq!(qbits, 10);
                assert_eq!(canon_set(&pos, &idx), canon_set(&positions, &indices));
            }
            _ => panic!("expected grid mode"),
        }
    }

    #[test]
    fn test_roundtrip_lattice_lossless() {
        let (pos, idx) = lattice(64);
        let blob = encode_mmc_u32(&pos, &idx, 10, 3).expect("encode");
        match decode_mmc(&blob).expect("decode") {
            MmcMesh::Grid { positions, indices, .. } => {
                assert_eq!(canon_set(&pos, &idx), canon_set(&positions, &indices));
            }
            _ => panic!("expected grid mode"),
        }
        // size sanity: this is the draco_profile_split mesh at 1/1 scale; the
        // Draco NG encoder produces ~2 KB here, MMC must stay well under raw.
        assert!(blob.len() < 2000, "lattice blob unexpectedly large: {}", blob.len());
    }

    #[test]
    fn test_duplicate_orphan_degenerate_pathology() {
        // encoder_draco.rs pathologies: coincident dup + orphan + quant-degenerate
        let pos = vec![
            10u32, 10, 10, // v0
            10, 10, 10, // v1 == v0 (coincident dup)
            100, 0, 0, // v2
            50, 100, 0, // v3
            200, 50, 0, // v4
            500, 500, 500, // v5 orphan
        ];
        let idx = vec![0u32, 1, 2, 2, 3, 4]; // face0 degenerate after dedup
        let blob = encode_mmc_u32(&pos, &idx, 10, 3).expect("encode");
        match decode_mmc(&blob).expect("decode") {
            MmcMesh::Grid { positions, indices, .. } => {
                assert_eq!(indices.len(), 3, "degenerate face must drop");
                assert_eq!(positions.len(), 9, "orphans + dups must compact");
                assert_eq!(canon_set(&positions, &indices), nondegenerate_set(&pos, &idx));
            }
            _ => panic!("expected grid mode"),
        }
    }

    #[test]
    fn test_random_soup_lossless() {
        // deterministic LCG soup, q16
        let mut state = 0x12345678u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let n = 3000usize;
        let pos: Vec<u32> = (0..n * 3).map(|_| next() & 0xFFFF).collect();
        let idx: Vec<u32> = (0..3 * 2500).map(|_| next() % n as u32).collect();
        let blob = encode_mmc_u32(&pos, &idx, 16, 3).expect("encode");
        match decode_mmc(&blob).expect("decode") {
            MmcMesh::Grid { positions, indices, .. } => {
                assert_eq!(canon_set(&positions, &indices), nondegenerate_set(&pos, &idx));
            }
            _ => panic!("expected grid mode"),
        }
    }

    #[test]
    fn test_deterministic() {
        let (pos, idx) = lattice(16);
        let a = encode_mmc_u32(&pos, &idx, 10, 3).expect("a");
        let b = encode_mmc_u32(&pos, &idx, 10, 3).expect("b");
        assert_eq!(a, b, "encode must be deterministic");
    }

    #[test]
    fn test_f32_dequant_roundtrip() {
        let pos = vec![0.0f32, 0.0, 0.0, 10.0, 0.0, 0.0, 5.0, 10.0, 0.0, 5.0, 5.0, 10.0];
        let idx = vec![0u32, 1, 2, 0, 1, 3, 1, 2, 3, 0, 2, 3];
        let blob = encode_mmc_f32(&pos, &idx, 14, 3).expect("encode");
        match decode_mmc(&blob).expect("decode") {
            MmcMesh::World { positions, indices, qbits } => {
                assert_eq!(qbits, 14);
                assert_eq!(indices.len(), 12);
                assert_eq!(positions.len(), 12);
                // every decoded coordinate within one quantization step
                let step = 10.0f32 / ((1 << 14) - 1) as f32;
                for chunk in positions.chunks_exact(3) {
                    assert!(chunk.iter().all(|v| (-step..=10.0 + step).contains(v)));
                }
            }
            _ => panic!("expected world mode"),
        }
    }

    #[test]
    fn test_empty_and_garbage() {
        assert!(encode_mmc_u32(&[], &[], 10, 3).is_err());
        assert!(decode_mmc(b"nope").is_err());
        assert!(decode_mmc(b"MMC1\x01\x00\x0a\x03").is_err());
        // all-degenerate mesh
        let pos = vec![1u32, 1, 1, 1, 1, 1, 1, 1, 1];
        let idx = vec![0u32, 1, 2];
        assert!(encode_mmc_u32(&pos, &idx, 10, 3).is_err());
    }

    // -- cross-implementation vectors (encoded by the Python reference) -------

    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 1469598103934665603;
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(1099511628211);
        }
        h
    }

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// 8x8 ripple lattice, q10 lossless, encoded by tools/mmc/mmc.py.
    /// Asserts the Rust decoder reproduces the Python decode bit-for-bit.
    #[test]
    fn test_python_vector_lattice() {
        let blob = unhex(
            "4d4d433101000a0340000000620000003c0000002d00000028b52ffd20749d01007402e1f010\
             1000fe001c001b001a00190018001700f20017ed007687566778a986658968980169000003008d\
             79401057b0d30228b52ffd6080001d0100880020000e1a00c1c10022000000c1c1001e06000168\
             d88c18b82e2087134c0a780a86",
        );
        match decode_mmc(&blob).expect("decode python vector") {
            MmcMesh::Grid { positions, indices, qbits } => {
                assert_eq!(qbits, 10);
                assert_eq!(positions.len() / 3, 64);
                assert_eq!(indices.len() / 3, 98);
                let pb: Vec<u8> = positions.iter().flat_map(|v| v.to_le_bytes()).collect();
                let ib: Vec<u8> = indices.iter().flat_map(|v| v.to_le_bytes()).collect();
                assert_eq!(fnv1a(&pb), 0x661baebbe40bde33, "vertex bytes diverge from reference");
                assert_eq!(fnv1a(&ib), 0x26bba83d3f6eb7bc, "index bytes diverge from reference");
            }
            _ => panic!("expected grid mode"),
        }
    }

    /// dup/orphan/degenerate pathology, encoded by the Python reference.
    #[test]
    fn test_python_vector_pathology() {
        let blob = unhex(
            "4d4d433101000a0303000000010000001b0000001b00000028b52ffd2012910000e1f000768756\
             6778a986658968980169000028b52ffd2012910000c863c800000000c864000000000000000000",
        );
        match decode_mmc(&blob).expect("decode python vector") {
            MmcMesh::Grid { positions, indices, .. } => {
                assert_eq!(positions, vec![100, 0, 0, 50, 100, 0, 200, 50, 0]);
                assert_eq!(indices, vec![0, 1, 2]);
            }
            _ => panic!("expected grid mode"),
        }
    }

    /// f32 dequant tetrahedron, encoded by the Python reference.
    #[test]
    fn test_python_vector_f32() {
        let blob = unhex(
            "4d4d433101010e0304000000040000000000000000000000000000008002203a8002203a800220\
             3a200000002100000028b52ffd2017b90000e1f0fe015f3206007687566778a986658968980169\
             000028b52ffd2018c1000000fe0000007f40400000fe0000007f40000000fe0000007f",
        );
        match decode_mmc(&blob).expect("decode python vector") {
            MmcMesh::World { positions, indices, qbits } => {
                assert_eq!(qbits, 14);
                let expect = [
                    0.0f32, 0.0, 0.0,
                    10.0, 0.0, 0.0,
                    5.00030517578125, 10.0, 0.0,
                    5.00030517578125, 5.00030517578125, 10.0,
                ];
                assert_eq!(positions.len(), expect.len());
                for (got, want) in positions.iter().zip(expect.iter()) {
                    assert_eq!(got.to_bits(), want.to_bits(), "f32 bits diverge");
                }
                assert_eq!(indices, vec![0, 1, 2, 3, 0, 1, 3, 1, 2, 0, 2, 3]);
            }
            _ => panic!("expected world mode"),
        }
    }
}
