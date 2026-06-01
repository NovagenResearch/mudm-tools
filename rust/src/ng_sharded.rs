//! Neuroglancer `neuroglancer_uint64_sharded_v1` support.
//!
//! Step 1: the label -> (shard, minishard) hashing, matching neuroglancer /
//! CloudVolume EXACTLY. CloudVolume computes the hash as
//! `uint64(mmh3.hash64(uint64(key).tobytes(), x64arch=False)[0])` — i.e.
//! MurmurHash3_x86_128 (seed 0) over the 8 little-endian bytes of the key, then
//! the LOW 64 bits of the 128-bit result. We implement MurmurHash3_x86_128
//! directly (pure Rust, no new dependency) and pin it against vectors generated
//! from `mmh3` (the very library CloudVolume uses), so the bucketing is provably
//! conformant. The end-to-end proof is a CloudVolume read of the emitted shards
//! (later step).
//!
//! Subsequent steps add the shard-file writer (two-level minishard index + gzip
//! framing) and the opt-in wiring into `generate_neuroglancer_multilod`.

/// MurmurHash3_x86_128 (Austin Appleby's reference algorithm). Returns the four
/// 32-bit lanes (h1, h2, h3, h4); the 128-bit digest is their little-endian
/// concatenation.
fn murmurhash3_x86_128(data: &[u8], seed: u32) -> (u32, u32, u32, u32) {
    let nblocks = data.len() / 16;
    let mut h1 = seed;
    let mut h2 = seed;
    let mut h3 = seed;
    let mut h4 = seed;
    const C1: u32 = 0x239b961b;
    const C2: u32 = 0xab0e9789;
    const C3: u32 = 0x38b34ae5;
    const C4: u32 = 0xa1e38b93;

    // body
    for i in 0..nblocks {
        let b = i * 16;
        let mut k1 = u32::from_le_bytes([data[b], data[b + 1], data[b + 2], data[b + 3]]);
        let mut k2 = u32::from_le_bytes([data[b + 4], data[b + 5], data[b + 6], data[b + 7]]);
        let mut k3 = u32::from_le_bytes([data[b + 8], data[b + 9], data[b + 10], data[b + 11]]);
        let mut k4 = u32::from_le_bytes([data[b + 12], data[b + 13], data[b + 14], data[b + 15]]);

        k1 = k1.wrapping_mul(C1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(C2);
        h1 ^= k1;
        h1 = h1.rotate_left(19);
        h1 = h1.wrapping_add(h2);
        h1 = h1.wrapping_mul(5).wrapping_add(0x561ccd1b);

        k2 = k2.wrapping_mul(C2);
        k2 = k2.rotate_left(16);
        k2 = k2.wrapping_mul(C3);
        h2 ^= k2;
        h2 = h2.rotate_left(17);
        h2 = h2.wrapping_add(h3);
        h2 = h2.wrapping_mul(5).wrapping_add(0x0bcaa747);

        k3 = k3.wrapping_mul(C3);
        k3 = k3.rotate_left(17);
        k3 = k3.wrapping_mul(C4);
        h3 ^= k3;
        h3 = h3.rotate_left(15);
        h3 = h3.wrapping_add(h4);
        h3 = h3.wrapping_mul(5).wrapping_add(0x96cd1c35);

        k4 = k4.wrapping_mul(C4);
        k4 = k4.rotate_left(18);
        k4 = k4.wrapping_mul(C1);
        h4 ^= k4;
        h4 = h4.rotate_left(13);
        h4 = h4.wrapping_add(h1);
        h4 = h4.wrapping_mul(5).wrapping_add(0x32ac3b17);
    }

    // tail (replicates the C switch fall-through)
    let tail = &data[nblocks * 16..];
    let rem = data.len() & 15;
    let mut k1: u32 = 0;
    let mut k2: u32 = 0;
    let mut k3: u32 = 0;
    let mut k4: u32 = 0;

    if rem >= 15 {
        k4 ^= (tail[14] as u32) << 16;
    }
    if rem >= 14 {
        k4 ^= (tail[13] as u32) << 8;
    }
    if rem >= 13 {
        k4 ^= tail[12] as u32;
        k4 = k4.wrapping_mul(C4);
        k4 = k4.rotate_left(18);
        k4 = k4.wrapping_mul(C1);
        h4 ^= k4;
    }
    if rem >= 12 {
        k3 ^= (tail[11] as u32) << 24;
    }
    if rem >= 11 {
        k3 ^= (tail[10] as u32) << 16;
    }
    if rem >= 10 {
        k3 ^= (tail[9] as u32) << 8;
    }
    if rem >= 9 {
        k3 ^= tail[8] as u32;
        k3 = k3.wrapping_mul(C3);
        k3 = k3.rotate_left(17);
        k3 = k3.wrapping_mul(C4);
        h3 ^= k3;
    }
    if rem >= 8 {
        k2 ^= (tail[7] as u32) << 24;
    }
    if rem >= 7 {
        k2 ^= (tail[6] as u32) << 16;
    }
    if rem >= 6 {
        k2 ^= (tail[5] as u32) << 8;
    }
    if rem >= 5 {
        k2 ^= tail[4] as u32;
        k2 = k2.wrapping_mul(C2);
        k2 = k2.rotate_left(16);
        k2 = k2.wrapping_mul(C3);
        h2 ^= k2;
    }
    if rem >= 4 {
        k1 ^= (tail[3] as u32) << 24;
    }
    if rem >= 3 {
        k1 ^= (tail[2] as u32) << 16;
    }
    if rem >= 2 {
        k1 ^= (tail[1] as u32) << 8;
    }
    if rem >= 1 {
        k1 ^= tail[0] as u32;
        k1 = k1.wrapping_mul(C1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(C2);
        h1 ^= k1;
    }

    // finalization
    let len = data.len() as u32;
    h1 ^= len;
    h2 ^= len;
    h3 ^= len;
    h4 ^= len;
    h1 = h1.wrapping_add(h2);
    h1 = h1.wrapping_add(h3);
    h1 = h1.wrapping_add(h4);
    h2 = h2.wrapping_add(h1);
    h3 = h3.wrapping_add(h1);
    h4 = h4.wrapping_add(h1);
    h1 = fmix32(h1);
    h2 = fmix32(h2);
    h3 = fmix32(h3);
    h4 = fmix32(h4);
    h1 = h1.wrapping_add(h2);
    h1 = h1.wrapping_add(h3);
    h1 = h1.wrapping_add(h4);
    h2 = h2.wrapping_add(h1);
    h3 = h3.wrapping_add(h1);
    h4 = h4.wrapping_add(h1);
    (h1, h2, h3, h4)
}

#[inline]
fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    h
}

/// The 64-bit neuroglancer/CloudVolume sharding hash of a label: the LOW 64 bits
/// of MurmurHash3_x86_128(seed=0) over the label's 8 little-endian bytes.
/// Equals `uint64(mmh3.hash64(uint64(key).tobytes(), x64arch=False)[0])`.
pub fn ng_shard_hash(key: u64) -> u64 {
    let (h1, h2, _h3, _h4) = murmurhash3_x86_128(&key.to_le_bytes(), 0);
    (h1 as u64) | ((h2 as u64) << 32)
}

/// `neuroglancer_uint64_sharded_v1` sharding parameters (the subset we emit).
#[derive(Debug, Clone, Copy)]
pub struct ShardingSpec {
    pub preshift_bits: u32,
    pub minishard_bits: u32,
    pub shard_bits: u32,
}

#[inline]
fn low_mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

impl ShardingSpec {
    /// Hashed chunk id = hash(label >> preshift_bits), for hash=murmurhash3_x86_128.
    pub fn hashed(&self, label: u64) -> u64 {
        ng_shard_hash(label >> self.preshift_bits)
    }

    /// Map a label to (shard_number, minishard_number) exactly as CloudVolume's
    /// `compute_shard_location`: minishard = low `minishard_bits`; shard = next
    /// `shard_bits` above the minishard bits.
    pub fn shard_minishard(&self, label: u64) -> (u64, u64) {
        let chunkid = self.hashed(label);
        let minishard = chunkid & low_mask(self.minishard_bits);
        let shard_mask = low_mask(self.shard_bits) << self.minishard_bits;
        let shard = (chunkid & shard_mask) >> self.minishard_bits;
        (shard, minishard)
    }

    /// Shard file basename (hex, zero-padded to ceil(shard_bits/4) chars), e.g.
    /// `0` for shard_bits<=4, matching CloudVolume's `{:x}`.zfill naming.
    pub fn shard_name(&self, shard: u64) -> String {
        let width = ((self.shard_bits as usize) + 3) / 4;
        format!("{:0width$x}", shard, width = width.max(1))
    }
}

/// Write `segments` as `neuroglancer_uint64_sharded_v1` `.shard` files into
/// `out_dir`, for the multiscale-mesh layout. Each segment contributes a
/// manifest (the `.index` bytes — the keyed chunk data) and its concatenated
/// Draco fragment bytes (stored immediately BEFORE the manifest, raw, found by
/// the reader via `manifest_start - sum(fragment_sizes)`).
///
/// Encodings are RAW (both `data_encoding` and `minishard_index_encoding`):
/// spec-valid, deterministic, and — since the Draco fragment bulk is raw
/// regardless for multiscale meshes — costs only trivial manifest/index size.
///
/// File layout per shard (offsets relative to `shard_index_end = 2^m * 16`):
///   [shard index: 2^m entries x (u64le start, u64le end)]   -> minishard index ranges
///   [chunk-data region: per segment `raw fragments ++ raw manifest`]
///   [minishard-index region: per non-empty minishard, the [3,n] u64le index]
///
/// Minishard index `[3, n]` C-order (rows concatenated): row0 = delta-encoded
/// labels (ascending), row1 = start-offset deltas, row2 = manifest sizes.
pub fn write_sharded(
    out_dir: &std::path::Path,
    segments: Vec<(u64, Vec<u8>, Vec<u8>)>, // (label, manifest, fragment_data)
    spec: ShardingSpec,
) -> std::io::Result<()> {
    use std::collections::BTreeMap;

    let num_minishards: usize = 1usize << spec.minishard_bits;

    // Group by shard, then minishard. BTreeMap keeps deterministic order.
    let mut by_shard: BTreeMap<u64, BTreeMap<u64, Vec<(u64, Vec<u8>, Vec<u8>)>>> = BTreeMap::new();
    for (label, manifest, frag) in segments {
        let (shard, minishard) = spec.shard_minishard(label);
        by_shard
            .entry(shard)
            .or_default()
            .entry(minishard)
            .or_default()
            .push((label, manifest, frag));
    }

    for (shard, mut minishards) in by_shard {
        // Sort each minishard's segments by label ascending (required for the
        // unsigned delta-encoded chunk-id row).
        for segs in minishards.values_mut() {
            segs.sort_by_key(|s| s.0);
        }

        // Pass 1: write the chunk-data region; record per-minishard
        // (label, manifest_start, manifest_size). Offsets are relative to
        // shard_index_end (i.e. the data region begins at offset 0 here).
        let mut data: Vec<u8> = Vec::new();
        let mut recorded: BTreeMap<u64, Vec<(u64, u64, u64)>> = BTreeMap::new();
        for (&mshard, segs) in minishards.iter() {
            let rec = recorded.entry(mshard).or_default();
            for (label, manifest, frag) in segs {
                data.extend_from_slice(frag); // raw fragments, immediately before the manifest
                let manifest_start = data.len() as u64;
                data.extend_from_slice(manifest); // raw manifest = the keyed chunk data
                rec.push((*label, manifest_start, manifest.len() as u64));
            }
        }

        // Pass 2: append each non-empty minishard index; fill the shard index.
        let mut shard_index: Vec<(u64, u64)> = vec![(0, 0); num_minishards];
        for m in 0..num_minishards as u64 {
            let entries = match recorded.get(&m) {
                Some(e) if !e.is_empty() => e,
                _ => {
                    let cur = data.len() as u64; // empty minishard: start == end
                    shard_index[m as usize] = (cur, cur);
                    continue;
                }
            };
            let n = entries.len();
            let mut idx = Vec::with_capacity(24 * n);
            // row0: delta-encoded labels (ascending).
            let mut prev_label = 0u64;
            for (k, &(label, _, _)) in entries.iter().enumerate() {
                let delta = if k == 0 { label } else { label - prev_label };
                idx.extend_from_slice(&delta.to_le_bytes());
                prev_label = label;
            }
            // row1: start-offset deltas. row1[0] = start_0; row1[i] = start_i -
            // start_{i-1} - size_{i-1}  (all >= 0 by construction).
            for k in 0..n {
                let (_, start, _) = entries[k];
                let delta = if k == 0 {
                    start
                } else {
                    let (_, prev_start, prev_size) = entries[k - 1];
                    start - prev_start - prev_size
                };
                idx.extend_from_slice(&delta.to_le_bytes());
            }
            // row2: manifest sizes.
            for &(_, _, size) in entries.iter() {
                idx.extend_from_slice(&size.to_le_bytes());
            }
            let idx_start = data.len() as u64;
            data.extend_from_slice(&idx);
            shard_index[m as usize] = (idx_start, data.len() as u64);
        }

        // Assemble: [shard index][data region].
        let mut out: Vec<u8> = Vec::with_capacity(num_minishards * 16 + data.len());
        for (s, e) in &shard_index {
            out.extend_from_slice(&s.to_le_bytes());
            out.extend_from_slice(&e.to_le_bytes());
        }
        out.extend_from_slice(&data);
        std::fs::write(out_dir.join(format!("{}.shard", spec.shard_name(shard))), &out)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Authoritative vectors generated from `mmh3` — the exact hash CloudVolume
    /// uses for `murmurhash3_x86_128`:
    ///   uint64(mmh3.hash64(int(k).to_bytes(8,'little'), seed=0, x64arch=False)[0])
    /// If this trips, the bucketing has diverged from neuroglancer/CloudVolume.
    #[test]
    fn ng_shard_hash_matches_mmh3() {
        let cases: &[(u64, u64)] = &[
            (0x0000000000000000, 0x4772b084e028ae41),
            (0x0000000000000001, 0xe8bd67d616d4ce9a),
            (0x0000000000000002, 0xd62f9cd21b013f5a),
            (0x000000000000002a, 0xc20b94f95119f47a),
            (0x00000000000003e8, 0xfc1b462deff0cd6f),
            (0x00000000deadbeef, 0x2583af9f1fd1e05f),
            (0x0123456789abcdef, 0x708036264c109d93),
            (0x0000000000001388, 0xfcd9dd786a14f1c0),
            (0x0000000100000000, 0xbbb12d133b78fd64),
        ];
        for &(k, want) in cases {
            assert_eq!(
                ng_shard_hash(k),
                want,
                "ng_shard_hash(0x{:016x}) diverged from mmh3 reference",
                k
            );
        }
    }

    #[test]
    fn shard_minishard_decomposition() {
        // shard_bits=0, minishard_bits=6 -> shard always 0, minishard = low 6 bits
        // of the hash. key=0 hash=0x...ae41 -> 0x41 & 0x3f = 0x01.
        let spec = ShardingSpec {
            preshift_bits: 0,
            minishard_bits: 6,
            shard_bits: 0,
        };
        let (shard, minishard) = spec.shard_minishard(0);
        assert_eq!(shard, 0);
        assert_eq!(minishard, 0x4772b084e028ae41 & 0x3f);
        assert_eq!(spec.shard_name(shard), "0");

        // shard_bits=4, minishard_bits=6: minishard=low6, shard=next4.
        let spec2 = ShardingSpec {
            preshift_bits: 0,
            minishard_bits: 6,
            shard_bits: 4,
        };
        let h = 0x2583af9f1fd1e05fu64; // hash of 0xdeadbeef
        let (shard, minishard) = spec2.shard_minishard(0xdeadbeef);
        assert_eq!(minishard, h & 0x3f);
        assert_eq!(shard, (h >> 6) & 0xf);
        assert_eq!(spec2.shard_name(shard).len(), 1);
    }

    #[test]
    fn murmur_empty_is_zero() {
        // MurmurHash3_x86_128("", 0) == 0 (canonical invariant).
        assert_eq!(murmurhash3_x86_128(&[], 0), (0, 0, 0, 0));
    }

    /// Decode the emitted shard back per the neuroglancer spec and assert every
    /// segment's manifest + fragment bytes are recovered EXACTLY (the byte-lossless
    /// guarantee), and that the writer is deterministic.
    #[test]
    fn write_sharded_roundtrip() {
        use std::collections::HashMap;
        let spec = ShardingSpec {
            preshift_bits: 0,
            minishard_bits: 2, // 4 minishards
            shard_bits: 0,     // single shard "0"
        };
        let mk = |seed: u8, n: usize| -> Vec<u8> {
            (0..n).map(|i| seed.wrapping_add(i as u8)).collect()
        };
        let inputs: Vec<(u64, Vec<u8>, Vec<u8>)> = vec![
            (0, mk(10, 7), mk(100, 13)),
            (1, mk(20, 5), mk(110, 0)), // empty fragment
            (2, mk(30, 9), mk(120, 4)),
            (3, mk(40, 1), mk(130, 30)),
            (42, mk(50, 16), mk(140, 8)),
            (1000, mk(60, 3), mk(150, 19)),
            (0x0123456789abcdef, mk(70, 12), mk(160, 6)),
        ];
        let orig: HashMap<u64, (Vec<u8>, Vec<u8>)> = inputs
            .iter()
            .map(|(l, m, f)| (*l, (m.clone(), f.clone())))
            .collect();

        let base = std::env::temp_dir().join(format!("ng_sharded_rt_{}", std::process::id()));
        let d1 = base.join("a");
        let d2 = base.join("b");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&d1).unwrap();
        std::fs::create_dir_all(&d2).unwrap();
        write_sharded(&d1, inputs.clone(), spec).unwrap();
        write_sharded(&d2, inputs.clone(), spec).unwrap();

        let num_minishards = 1usize << spec.minishard_bits;
        let shard_index_end = (num_minishards as u64) * 16;
        let bytes = std::fs::read(d1.join("0.shard")).unwrap();
        assert_eq!(
            bytes,
            std::fs::read(d2.join("0.shard")).unwrap(),
            "sharded output is not deterministic"
        );

        let ru64 = |b: &[u8], off: usize| u64::from_le_bytes(b[off..off + 8].try_into().unwrap());
        let mut recovered = 0usize;
        for m in 0..num_minishards {
            let start = ru64(&bytes, m * 16);
            let end = ru64(&bytes, m * 16 + 8);
            if start == end {
                continue;
            }
            let mi = &bytes[(shard_index_end + start) as usize..(shard_index_end + end) as usize];
            let n = mi.len() / 24;
            let g = |row: usize, i: usize| ru64(mi, (row * n + i) * 8);
            let mut labels = vec![0u64; n];
            let mut acc = 0u64;
            for i in 0..n {
                acc = if i == 0 { g(0, i) } else { acc + g(0, i) };
                labels[i] = acc;
            }
            let sizes: Vec<u64> = (0..n).map(|i| g(2, i)).collect();
            let mut starts = vec![0u64; n];
            for i in 0..n {
                starts[i] = if i == 0 {
                    shard_index_end + g(1, 0)
                } else {
                    starts[i - 1] + sizes[i - 1] + g(1, i)
                };
            }
            for i in 0..n {
                let label = labels[i];
                let man = &bytes[starts[i] as usize..(starts[i] + sizes[i]) as usize];
                let (orig_man, orig_frag) = &orig[&label];
                assert_eq!(man, orig_man.as_slice(), "manifest mismatch for label {}", label);
                let fl = orig_frag.len() as u64;
                let frag = &bytes[(starts[i] - fl) as usize..starts[i] as usize];
                assert_eq!(frag, orig_frag.as_slice(), "fragment mismatch for label {}", label);
                recovered += 1;
            }
        }
        assert_eq!(recovered, inputs.len(), "did not recover every segment");
        let _ = std::fs::remove_dir_all(&base);
    }
}
