/// Binary fragment format for disk-backed tile assembly.
///
/// During streaming ingestion, each feature is clipped to each tile at each
/// zoom level, producing "fragments". These are written to append-only
/// shard files (one per rayon thread) for full write parallelism.
/// During tile encoding, fragments from all shards are read back and merged.
///
/// Fragment binary format v2 (MJF2) — little-endian, ZSTD-compressed stream:
///   [4 bytes] magic: b"MJF2"  (inside the compressed stream)
///   [4 bytes] feature_id: u32
///   [4 bytes] tile_z: u32
///   [4 bytes] tile_x: u32
///   [4 bytes] tile_y: u32
///   [4 bytes] tile_d: u32
///   [1 byte]  geom_type: u8
///   [4 bytes] n_vertices: u32
///   [n_vertices * 4 bytes] xy: pairs of f32 (flat x,y,x,y,...)
///   [n_vertices * 4 bytes] z: f32 values
///   [4 bytes] n_ring_lengths: u32
///   [n_ring_lengths * 4 bytes] ring_lengths: u32
///   [4 bytes] payload_end_magic: b"FEND"

use std::io::{self, Read, Write, BufWriter, BufReader};
use std::fs::File;
use std::path::{Path, PathBuf};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

const MAGIC: &[u8; 4] = b"MJF2";
const END_MAGIC: &[u8; 4] = b"FEND";

/// A single tile fragment — the clipped portion of one feature in one tile.
#[derive(Debug, Clone)]
pub struct Fragment3D {
    pub feature_id: u32,
    pub tile_z: u32,
    pub tile_x: u32,
    pub tile_y: u32,
    pub tile_d: u32,
    pub geom_type: u8,
    pub xy: Vec<f32>,
    pub z: Vec<f32>,
    pub ring_lengths: Vec<u32>,
}

impl Fragment3D {
    /// Estimate the in-memory byte footprint of this fragment.
    ///
    /// Used by `_next_parquet_batch` to enforce a byte budget so that
    /// batches with a few giant fragments don't blow up memory.
    pub fn estimate_bytes(&self) -> usize {
        // 5×u32 + 1×u8 + 3 Vec headers (ptr+len+cap = 24 bytes each)
        const FIXED: usize = 5 * 4 + 1 + 3 * 24;
        FIXED + self.xy.len() * 4 + self.z.len() * 4 + self.ring_lengths.len() * 4
    }
}

/// Append-only fragment writer with f32 + ZSTD compression.
///
/// Each writer owns one ZSTD-compressed file. For parallel ingestion,
/// create one writer per thread (sharded writes).
pub struct Fragment3DWriter {
    writer: Option<zstd::stream::write::Encoder<'static, BufWriter<File>>>,
    count: u64,
}

impl Fragment3DWriter {
    pub fn new(path: &Path) -> io::Result<Self> {
        let file = File::create(path)?;
        let buf = BufWriter::with_capacity(1 << 20, file); // 1 MB buffer
        let encoder = zstd::stream::write::Encoder::new(buf, 3)?; // level 3 = good speed/ratio
        Ok(Self {
            writer: Some(encoder),
            count: 0,
        })
    }

    pub fn write(&mut self, frag: &Fragment3D) -> io::Result<()> {
        let w = self.writer.as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Writer already flushed"))?;

        let n_verts = frag.z.len() as u32;

        w.write_all(MAGIC)?;
        w.write_u32::<LittleEndian>(frag.feature_id)?;
        w.write_u32::<LittleEndian>(frag.tile_z)?;
        w.write_u32::<LittleEndian>(frag.tile_x)?;
        w.write_u32::<LittleEndian>(frag.tile_y)?;
        w.write_u32::<LittleEndian>(frag.tile_d)?;
        w.write_u8(frag.geom_type)?;
        w.write_u32::<LittleEndian>(n_verts)?;

        // XY: n_verts * 2 floats (f32)
        for &v in &frag.xy {
            w.write_f32::<LittleEndian>(v)?;
        }

        // Z: n_verts floats (f32)
        for &v in &frag.z {
            w.write_f32::<LittleEndian>(v)?;
        }

        // Ring lengths
        w.write_u32::<LittleEndian>(frag.ring_lengths.len() as u32)?;
        for &rl in &frag.ring_lengths {
            w.write_u32::<LittleEndian>(rl)?;
        }

        w.write_all(END_MAGIC)?;
        self.count += 1;
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        if let Some(encoder) = self.writer.take() {
            encoder.finish()?;
        }
        Ok(())
    }

    pub fn count(&self) -> u64 {
        self.count
    }
}

/// Finalize the ZSTD frame on drop so rayon thread-local writers
/// are properly flushed when `for_each_init` drops them.
impl Drop for Fragment3DWriter {
    fn drop(&mut self) {
        if let Some(encoder) = self.writer.take() {
            let _ = encoder.finish(); // best-effort; can't propagate errors in Drop
        }
    }
}

/// Single-shard reader over one ZSTD-compressed fragment file.
struct ShardReader3D {
    reader: BufReader<zstd::stream::read::Decoder<'static, BufReader<File>>>,
}

impl ShardReader3D {
    fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let decoder = zstd::stream::read::Decoder::new(file)?;
        Ok(Self {
            reader: BufReader::with_capacity(1 << 20, decoder),
        })
    }

    fn read_next(&mut self) -> io::Result<Option<Fragment3D>> {
        let mut magic = [0u8; 4];
        match self.reader.read_exact(&mut magic) {
            Ok(()) => {}
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }
        if &magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad fragment magic"));
        }

        let feature_id = self.reader.read_u32::<LittleEndian>()?;
        let tile_z = self.reader.read_u32::<LittleEndian>()?;
        let tile_x = self.reader.read_u32::<LittleEndian>()?;
        let tile_y = self.reader.read_u32::<LittleEndian>()?;
        let tile_d = self.reader.read_u32::<LittleEndian>()?;
        let geom_type = self.reader.read_u8()?;
        let n_verts = self.reader.read_u32::<LittleEndian>()? as usize;

        // XY: read f32
        let mut xy = Vec::with_capacity(n_verts * 2);
        for _ in 0..n_verts * 2 {
            xy.push(self.reader.read_f32::<LittleEndian>()?);
        }

        // Z: read f32
        let mut z = Vec::with_capacity(n_verts);
        for _ in 0..n_verts {
            z.push(self.reader.read_f32::<LittleEndian>()?);
        }

        // Ring lengths
        let n_rl = self.reader.read_u32::<LittleEndian>()? as usize;
        let mut ring_lengths = Vec::with_capacity(n_rl);
        for _ in 0..n_rl {
            ring_lengths.push(self.reader.read_u32::<LittleEndian>()?);
        }

        // End magic
        let mut end = [0u8; 4];
        self.reader.read_exact(&mut end)?;
        if &end != END_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad fragment end magic"));
        }

        Ok(Some(Fragment3D {
            feature_id, tile_z, tile_x, tile_y, tile_d,
            geom_type, xy, z, ring_lengths,
        }))
    }
}

/// Best-effort OS prefetch hint for a shard file.
///
/// Additive, non-load-bearing readahead to overlap open()+decode. Hints are
/// best-effort and silently ignored on filesystems that don't support them.
/// On Linux, uses `posix_fadvise(SEQUENTIAL|WILLNEED)`; elsewhere, an
/// `mmap` + `madvise(WILLNEED)`. The handle is dropped immediately after the
/// hint is issued — the kernel hint persists past close.
#[cfg(target_os = "linux")]
pub(crate) fn prefetch_advise(path: &Path) {
    use std::os::unix::io::AsRawFd;
    if let Ok(f) = File::open(path) {
        let fd = f.as_raw_fd();
        // best-effort; ignore errors. (0, 0) = whole file.
        unsafe {
            libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_SEQUENTIAL);
            libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_WILLNEED);
        }
        // f drops here; hint persists in the kernel after close.
    }
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn prefetch_advise(path: &Path) {
    if let Ok(f) = File::open(path) {
        // Safety: file is opened read-only and the mapping is not aliased
        // mutably; we only issue an advisory hint and drop the mapping.
        if let Ok(m) = unsafe { memmap2::Mmap::map(&f) } {
            let _ = m.advise(memmap2::Advice::WillNeed); // best-effort
        }
    }
}

/// Parse one shard into a tile-keyed local map.
///
/// Uses the same byte loop as `ShardReader3D::read_next`. Each call opens its
/// OWN `ShardReader3D` (which is `!Sync`) so it is safe to invoke from many
/// rayon workers concurrently — no decoder is ever shared across threads.
fn parse_shard_grouped(
    path: &Path,
    fadvise: bool,
) -> io::Result<ahash::AHashMap<(u32, u32, u32, u32), Vec<Fragment3D>>> {
    use ahash::AHashMap;
    if fadvise {
        prefetch_advise(path);
    }
    let mut reader = ShardReader3D::open(path)?; // each worker owns its own decoder
    let mut local: AHashMap<(u32, u32, u32, u32), Vec<Fragment3D>> = AHashMap::new();
    while let Some(frag) = reader.read_next()? {
        let key = (frag.tile_z, frag.tile_x, frag.tile_y, frag.tile_d);
        local.entry(key).or_default().push(frag);
    }
    Ok(local)
}

/// Fragment3D reader — reads 3D fragments from one or more ZSTD-compressed shard files.
///
/// Transparently iterates through all shards in order.
pub struct Fragment3DReader {
    paths: Vec<PathBuf>,
    current_shard: Option<ShardReader3D>,
    current_index: usize,
}

impl Fragment3DReader {
    /// Open a reader over a single fragment file.
    pub fn new(path: &Path) -> io::Result<Self> {
        let shard = ShardReader3D::open(path)?;
        Ok(Self {
            paths: vec![path.to_path_buf()],
            current_shard: Some(shard),
            current_index: 0,
        })
    }

    /// Open a reader over a directory of shard files (*.mjf).
    pub fn open_dir(dir: &Path) -> io::Result<Self> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "mjf"))
            .collect();
        paths.sort(); // deterministic order

        if paths.is_empty() {
            return Ok(Self {
                paths,
                current_shard: None,
                current_index: 0,
            });
        }

        let shard = ShardReader3D::open(&paths[0])?;
        Ok(Self {
            paths,
            current_shard: Some(shard),
            current_index: 0,
        })
    }

    pub fn read_next(&mut self) -> io::Result<Option<Fragment3D>> {
        loop {
            if let Some(ref mut shard) = self.current_shard {
                if let Some(frag) = shard.read_next()? {
                    return Ok(Some(frag));
                }
                // Current shard exhausted — advance to next
                self.current_index += 1;
                if self.current_index < self.paths.len() {
                    self.current_shard = Some(ShardReader3D::open(&self.paths[self.current_index])?);
                    continue;
                } else {
                    self.current_shard = None;
                    return Ok(None);
                }
            } else {
                return Ok(None);
            }
        }
    }

    /// Read all fragments, grouped by tile key.
    pub fn read_all_grouped(&mut self) -> io::Result<ahash::AHashMap<(u32, u32, u32, u32), Vec<Fragment3D>>> {
        use ahash::AHashMap;
        let mut groups: AHashMap<(u32, u32, u32, u32), Vec<Fragment3D>> = AHashMap::new();
        while let Some(frag) = self.read_next()? {
            let key = (frag.tile_z, frag.tile_x, frag.tile_y, frag.tile_d);
            groups.entry(key).or_default().push(frag);
        }
        Ok(groups)
    }

    /// Read all fragments, grouped by feature_id.
    ///
    /// Used by `generate_neuroglancer()` which needs segment-centric grouping
    /// (one mesh per feature) rather than tile-centric grouping.
    pub fn read_all_grouped_by_feature(&mut self) -> io::Result<ahash::AHashMap<u32, Vec<Fragment3D>>> {
        use ahash::AHashMap;
        self.reset()?;
        let mut groups: AHashMap<u32, Vec<Fragment3D>> = AHashMap::new();
        while let Some(frag) = self.read_next()? {
            groups.entry(frag.feature_id).or_default().push(frag);
        }
        Ok(groups)
    }

    /// Reset reader to beginning (reopens first shard).
    pub fn reset(&mut self) -> io::Result<()> {
        self.current_index = 0;
        if !self.paths.is_empty() {
            self.current_shard = Some(ShardReader3D::open(&self.paths[0])?);
        } else {
            self.current_shard = None;
        }
        Ok(())
    }

    /// Build a reader from an explicit list of shard paths WITHOUT opening any
    /// shard up front (lazy). The serial cursor starts at the beginning; the
    /// parallel read methods read only `self.paths` and never touch the cursor.
    ///
    /// Used by the chunked parallel Parquet path to construct a reader over a
    /// slice of the shard snapshot.
    pub fn from_paths(paths: Vec<PathBuf>) -> Self {
        Self {
            paths,
            current_shard: None,
            current_index: 0,
        }
    }

    /// Parallel sibling of `read_all_grouped`.
    ///
    /// Reads ONLY the immutable `self.paths` slice — never touches
    /// `current_shard`/`current_index`. Each rayon worker opens its own
    /// `ShardReader3D` (which is `!Sync`) via `parse_shard_grouped`, builds a
    /// thread-local `AHashMap`, then bulk-merges into a shared `DashMap`
    /// (mirroring `read_group_simplify_encode` at `streaming.rs:2018-2054`).
    ///
    /// Output equals the serial `read_all_grouped` as a multiset (same key set;
    /// per key the same multiset of fragments), modulo non-deterministic order.
    ///
    /// Per-shard errors are collected and surfaced as an `Err` (the serial path
    /// propagates via `?`; this matches that contract — errors are NOT swallowed).
    ///
    /// `io_threads == 0` uses the global rayon pool; `io_threads > 0` sizes a
    /// scoped pool to that count (the measured I/O knee, not nproc).
    pub fn read_all_grouped_parallel(
        &self,
        io_threads: usize,
        fadvise: bool,
    ) -> io::Result<ahash::AHashMap<(u32, u32, u32, u32), Vec<Fragment3D>>> {
        use dashmap::DashMap;
        use std::sync::Mutex;
        if self.paths.is_empty() {
            return Ok(ahash::AHashMap::new());
        }
        let groups: DashMap<(u32, u32, u32, u32), Vec<Fragment3D>> = DashMap::new();
        let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
        let run = || {
            use rayon::prelude::*;
            self.paths.par_iter().with_min_len(1).for_each(|p| {
                match parse_shard_grouped(p, fadvise) {
                    Ok(local) => {
                        for (k, mut v) in local {
                            groups.entry(k).or_default().extend(v.drain(..));
                        }
                    }
                    Err(e) => {
                        errors
                            .lock()
                            .unwrap()
                            .push(format!("{}: {}", p.display(), e));
                    }
                }
            });
        };
        if io_threads > 0 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(io_threads)
                .build()
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            pool.install(run);
        } else {
            run(); // global pool
        }
        let errs = errors.into_inner().unwrap();
        if !errs.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("{} shard(s) failed to parse: {}", errs.len(), errs.join("; ")),
            ));
        }
        Ok(groups.into_iter().collect())
    }

    /// Parallel sibling of `read_all_grouped_by_feature`.
    ///
    /// Same par_iter + thread-local-map + `DashMap` merge pattern as
    /// `read_all_grouped_parallel`, but keyed by `feature_id` (the
    /// segment-centric grouping used by the Neuroglancer path). Reads ONLY
    /// `self.paths`; collects and surfaces per-shard errors as `Err`.
    pub fn read_all_grouped_by_feature_parallel(
        &self,
        io_threads: usize,
        fadvise: bool,
    ) -> io::Result<ahash::AHashMap<u32, Vec<Fragment3D>>> {
        use ahash::AHashMap;
        use dashmap::DashMap;
        use std::sync::Mutex;
        if self.paths.is_empty() {
            return Ok(AHashMap::new());
        }
        let groups: DashMap<u32, Vec<Fragment3D>> = DashMap::new();
        let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
        let run = || {
            use rayon::prelude::*;
            self.paths.par_iter().with_min_len(1).for_each(|p| {
                if fadvise {
                    prefetch_advise(p);
                }
                let mut reader = match ShardReader3D::open(p) {
                    Ok(r) => r,
                    Err(e) => {
                        errors
                            .lock()
                            .unwrap()
                            .push(format!("{}: {}", p.display(), e));
                        return;
                    }
                };
                let mut local: AHashMap<u32, Vec<Fragment3D>> = AHashMap::new();
                loop {
                    match reader.read_next() {
                        Ok(Some(frag)) => {
                            local.entry(frag.feature_id).or_default().push(frag);
                        }
                        Ok(None) => break,
                        Err(e) => {
                            errors
                                .lock()
                                .unwrap()
                                .push(format!("{}: {}", p.display(), e));
                            break;
                        }
                    }
                }
                for (k, mut v) in local {
                    groups.entry(k).or_default().extend(v.drain(..));
                }
            });
        };
        if io_threads > 0 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(io_threads)
                .build()
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            pool.install(run);
        } else {
            run(); // global pool
        }
        let errs = errors.into_inner().unwrap();
        if !errs.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("{} shard(s) failed to parse: {}", errs.len(), errs.join("; ")),
            ));
        }
        Ok(groups.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let dir = std::env::temp_dir().join("test_frag_roundtrip_v2");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("shard_0.mjf");

        let frag = Fragment3D {
            feature_id: 42,
            tile_z: 2,
            tile_x: 1,
            tile_y: 3,
            tile_d: 0,
            geom_type: 5,
            xy: vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6],
            z: vec![0.7f32, 0.8, 0.9],
            ring_lengths: vec![3],
        };

        {
            let mut writer = Fragment3DWriter::new(&path).unwrap();
            writer.write(&frag).unwrap();
            writer.write(&frag).unwrap();
            writer.flush().unwrap();
            assert_eq!(writer.count(), 2);
        }

        {
            let mut reader = Fragment3DReader::new(&path).unwrap();
            let f1 = reader.read_next().unwrap().unwrap();
            assert_eq!(f1.feature_id, 42);
            assert_eq!(f1.tile_z, 2);
            assert_eq!(f1.xy.len(), 6);
            assert_eq!(f1.z.len(), 3);
            assert!((f1.xy[0] - 0.1f32).abs() < 1e-6);
            assert!((f1.z[0] - 0.7f32).abs() < 1e-6);

            let f2 = reader.read_next().unwrap().unwrap();
            assert_eq!(f2.feature_id, 42);
            let f3 = reader.read_next().unwrap();
            assert!(f3.is_none());
        }

        // Test reset
        {
            let mut reader = Fragment3DReader::new(&path).unwrap();
            let _f1 = reader.read_next().unwrap().unwrap();
            reader.reset().unwrap();
            let f1_again = reader.read_next().unwrap().unwrap();
            assert_eq!(f1_again.feature_id, 42);
            assert_eq!(f1_again.tile_z, 2);
        }

        // Verify compressed file is smaller than uncompressed would be
        let file_size = std::fs::metadata(&path).unwrap().len();
        assert!(file_size < 210, "Compressed file ({file_size} bytes) should be < 210 bytes");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_multi_shard_reader() {
        let dir = std::env::temp_dir().join("test_frag_multishard_v2");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::create_dir_all(&dir);

        let frag_a = Fragment3D {
            feature_id: 1,
            tile_z: 0, tile_x: 0, tile_y: 0, tile_d: 0,
            geom_type: 5,
            xy: vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0],
            z: vec![0.0f32, 0.0, 0.0],
            ring_lengths: vec![3],
        };
        let frag_b = Fragment3D {
            feature_id: 2,
            tile_z: 1, tile_x: 0, tile_y: 0, tile_d: 0,
            geom_type: 5,
            xy: vec![0.0f32, 0.0, 0.5, 0.0, 0.0, 0.5],
            z: vec![0.0f32, 0.0, 0.0],
            ring_lengths: vec![3],
        };

        // Write to two separate shards
        {
            let mut w0 = Fragment3DWriter::new(&dir.join("shard_000.mjf")).unwrap();
            w0.write(&frag_a).unwrap();
            w0.write(&frag_a).unwrap();
            w0.flush().unwrap();
        }
        {
            let mut w1 = Fragment3DWriter::new(&dir.join("shard_001.mjf")).unwrap();
            w1.write(&frag_b).unwrap();
            w1.flush().unwrap();
        }

        // open_dir reads both shards
        {
            let mut reader = Fragment3DReader::open_dir(&dir).unwrap();
            let groups = reader.read_all_grouped().unwrap();
            assert_eq!(groups.len(), 2);
            assert_eq!(groups[&(0, 0, 0, 0)].len(), 2);
            assert_eq!(groups[&(1, 0, 0, 0)].len(), 1);
        }

        // grouped by feature
        {
            let mut reader = Fragment3DReader::open_dir(&dir).unwrap();
            let by_feat = reader.read_all_grouped_by_feature().unwrap();
            assert_eq!(by_feat.len(), 2);
            assert_eq!(by_feat[&1].len(), 2);
            assert_eq!(by_feat[&2].len(), 1);
        }

        // reset works across shards
        {
            let mut reader = Fragment3DReader::open_dir(&dir).unwrap();
            let f1 = reader.read_next().unwrap().unwrap();
            assert_eq!(f1.feature_id, 1);
            // read rest
            while reader.read_next().unwrap().is_some() {}
            // reset and read again
            reader.reset().unwrap();
            let f1_again = reader.read_next().unwrap().unwrap();
            assert_eq!(f1_again.feature_id, 1);
        }

        // Drop-based flush (no explicit flush call)
        {
            let path = dir.join("shard_drop.mjf");
            {
                let mut w = Fragment3DWriter::new(&path).unwrap();
                w.write(&frag_a).unwrap();
                // no flush — Drop should finalize ZSTD frame
            }
            let mut reader = Fragment3DReader::new(&path).unwrap();
            let f = reader.read_next().unwrap().unwrap();
            assert_eq!(f.feature_id, 1);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Canonicalise a tile-keyed group map into a comparable form: for each
    /// key, the sorted multiset of feature_ids of its fragments. Order within
    /// the parallel path is non-deterministic by design, so we sort.
    fn canonical_grouped(
        m: &ahash::AHashMap<(u32, u32, u32, u32), Vec<Fragment3D>>,
    ) -> std::collections::BTreeMap<(u32, u32, u32, u32), Vec<u32>> {
        let mut out: std::collections::BTreeMap<(u32, u32, u32, u32), Vec<u32>> =
            std::collections::BTreeMap::new();
        for (k, v) in m {
            let mut fids: Vec<u32> = v.iter().map(|f| f.feature_id).collect();
            fids.sort_unstable();
            out.insert(*k, fids);
        }
        out
    }

    #[test]
    fn test_read_all_grouped_parallel_matches_serial() {
        let dir = std::env::temp_dir().join("test_frag_parallel_v2");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::create_dir_all(&dir);

        // Helper to build a fragment with a given feature/tile.
        let mk = |feature_id: u32, tz: u32, tx: u32, ty: u32, td: u32| Fragment3D {
            feature_id,
            tile_z: tz,
            tile_x: tx,
            tile_y: ty,
            tile_d: td,
            geom_type: 5,
            xy: vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0],
            z: vec![0.0f32, 0.0, 0.0],
            ring_lengths: vec![3],
        };

        // 3 shards with overlapping AND distinct tile keys so the merge path
        // (entry().or_default().extend) is exercised across shards.
        {
            let mut w0 = Fragment3DWriter::new(&dir.join("shard_000.mjf")).unwrap();
            w0.write(&mk(1, 0, 0, 0, 0)).unwrap();
            w0.write(&mk(2, 0, 0, 0, 0)).unwrap(); // same tile as above (within shard)
            w0.write(&mk(3, 1, 0, 0, 0)).unwrap();
            w0.flush().unwrap();
        }
        {
            let mut w1 = Fragment3DWriter::new(&dir.join("shard_001.mjf")).unwrap();
            w1.write(&mk(4, 0, 0, 0, 0)).unwrap(); // shared tile (0,0,0,0) across shards
            w1.write(&mk(5, 2, 1, 1, 0)).unwrap();
            w1.flush().unwrap();
        }
        {
            let mut w2 = Fragment3DWriter::new(&dir.join("shard_002.mjf")).unwrap();
            w2.write(&mk(6, 1, 0, 0, 0)).unwrap(); // shared tile (1,0,0,0) across shards
            w2.write(&mk(7, 3, 5, 5, 1)).unwrap();
            w2.flush().unwrap();
        }

        // Serial baseline.
        let mut serial_reader = Fragment3DReader::open_dir(&dir).unwrap();
        let serial = serial_reader.read_all_grouped().unwrap();

        // Parallel (io_threads=0 => global pool; fadvise off).
        let par_reader = Fragment3DReader::open_dir(&dir).unwrap();
        let parallel = par_reader.read_all_grouped_parallel(0, false).unwrap();

        // Same key set + per-key same sorted multiset of feature_ids.
        assert_eq!(canonical_grouped(&serial), canonical_grouped(&parallel));

        // Also verify scoped-pool path (io_threads>0) and fadvise=true.
        let par_reader2 = Fragment3DReader::open_dir(&dir).unwrap();
        let parallel2 = par_reader2.read_all_grouped_parallel(4, true).unwrap();
        assert_eq!(canonical_grouped(&serial), canonical_grouped(&parallel2));

        // by_feature parallel matches serial by_feature.
        let mut serial_feat_reader = Fragment3DReader::open_dir(&dir).unwrap();
        let serial_feat = serial_feat_reader.read_all_grouped_by_feature().unwrap();
        let par_feat_reader = Fragment3DReader::open_dir(&dir).unwrap();
        let parallel_feat = par_feat_reader
            .read_all_grouped_by_feature_parallel(0, false)
            .unwrap();
        let canon_feat = |m: &ahash::AHashMap<u32, Vec<Fragment3D>>| {
            let mut out: std::collections::BTreeMap<u32, usize> =
                std::collections::BTreeMap::new();
            for (k, v) in m {
                out.insert(*k, v.len());
            }
            out
        };
        assert_eq!(canon_feat(&serial_feat), canon_feat(&parallel_feat));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_read_all_grouped_parallel_single_shard() {
        let dir = std::env::temp_dir().join("test_frag_parallel_single_v2");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::create_dir_all(&dir);

        let frag = Fragment3D {
            feature_id: 99,
            tile_z: 2,
            tile_x: 3,
            tile_y: 4,
            tile_d: 0,
            geom_type: 5,
            xy: vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6],
            z: vec![0.7f32, 0.8, 0.9],
            ring_lengths: vec![3],
        };
        {
            let mut w = Fragment3DWriter::new(&dir.join("only.mjf")).unwrap();
            w.write(&frag).unwrap();
            w.write(&frag).unwrap();
            w.flush().unwrap();
        }

        let mut serial_reader = Fragment3DReader::open_dir(&dir).unwrap();
        let serial = serial_reader.read_all_grouped().unwrap();
        let par_reader = Fragment3DReader::open_dir(&dir).unwrap();
        let parallel = par_reader.read_all_grouped_parallel(2, false).unwrap();

        assert_eq!(parallel.len(), 1);
        assert_eq!(parallel[&(2, 3, 4, 0)].len(), 2);
        assert_eq!(canonical_grouped(&serial), canonical_grouped(&parallel));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_read_all_grouped_parallel_empty_dir() {
        let dir = std::env::temp_dir().join("test_frag_parallel_empty_v2");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::create_dir_all(&dir);

        // open_dir on an empty dir => empty paths.
        let reader = Fragment3DReader::open_dir(&dir).unwrap();
        let parallel = reader.read_all_grouped_parallel(4, true).unwrap();
        assert!(parallel.is_empty());

        let parallel_feat = reader
            .read_all_grouped_by_feature_parallel(0, false)
            .unwrap();
        assert!(parallel_feat.is_empty());

        // from_paths with empty vec => empty map, no panic.
        let empty_reader = Fragment3DReader::from_paths(Vec::new());
        assert!(empty_reader
            .read_all_grouped_parallel(0, false)
            .unwrap()
            .is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_read_all_grouped_parallel_propagates_shard_error() {
        // A corrupt/truncated shard must surface as Err, never be silently
        // skipped (matches the serial read_next `?` contract; guards against
        // the parquet cursor advancing past a broken shard = silent data loss).
        let dir = std::env::temp_dir().join("test_frag_parallel_corrupt_v2");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::create_dir_all(&dir);

        let frag = Fragment3D {
            feature_id: 1,
            tile_z: 0,
            tile_x: 0,
            tile_y: 0,
            tile_d: 0,
            geom_type: 5,
            xy: vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6],
            z: vec![0.7f32, 0.8, 0.9],
            ring_lengths: vec![3],
        };
        {
            let mut w = Fragment3DWriter::new(&dir.join("a_valid.mjf")).unwrap();
            for _ in 0..3 {
                w.write(&frag).unwrap();
            }
            w.flush().unwrap();
        }
        // A second .mjf that is NOT a valid ZSTD stream: the decoder fails on
        // first read with a non-EOF error, which read_next propagates as Err
        // (a clean trailing truncation, by contrast, decodes as graceful EOF).
        std::fs::write(
            dir.join("b_corrupt.mjf"),
            b"NOT_A_VALID_ZSTD_FRAME________________________________",
        )
        .unwrap();

        let reader = Fragment3DReader::open_dir(&dir).unwrap();
        let res = reader.read_all_grouped_parallel(2, false);
        assert!(
            res.is_err(),
            "a corrupt/truncated shard must produce Err, not be silently skipped"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
