/// Binary fragment format for 2D tile assembly.
///
/// Fragment2D binary format (MF2D) — little-endian, ZSTD-compressed stream:
///   [4 bytes] magic: b"MF2D"
///   [4 bytes] feature_id: u32
///   [4 bytes] tile_z: u32
///   [4 bytes] tile_x: u32
///   [4 bytes] tile_y: u32
///   [1 byte]  geom_type: u8 (1=Point, 2=LineString, 3=Polygon)
///   [4 bytes] n_vertices: u32
///   [n_vertices * 2 * 4 bytes] xy: f32 pairs (flat x,y,x,y,...)
///   [4 bytes] n_ring_lengths: u32
///   [n_ring_lengths * 4 bytes] ring_lengths: u32
///   [4 bytes] payload_end_magic: b"2END"

use std::io::{self, Read, Write, BufWriter, BufReader};
use std::fs::File;
use std::path::{Path, PathBuf};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

const MAGIC: &[u8; 4] = b"MF2D";
const END_MAGIC: &[u8; 4] = b"2END";

/// A single 2D tile fragment — the clipped portion of one feature in one tile.
#[derive(Debug, Clone)]
pub struct Fragment2D {
    pub feature_id: u32,
    pub tile_z: u32,
    pub tile_x: u32,
    pub tile_y: u32,
    pub geom_type: u8,
    /// Flat x,y pairs in normalized [0,1]² space.
    pub xy: Vec<f32>,
    /// Ring vertex counts for Polygon; empty for Point/LineString.
    pub ring_lengths: Vec<u32>,
}

impl Fragment2D {
    /// Estimate the in-memory byte footprint of this fragment.
    pub fn estimate_bytes(&self) -> usize {
        const FIXED: usize = 4 * 4 + 1 + 2 * 24; // 4×u32 + 1×u8 + 2 Vec headers
        FIXED + self.xy.len() * 4 + self.ring_lengths.len() * 4
    }
}

/// Append-only fragment writer with f32 + ZSTD compression.
pub struct Fragment2DWriter {
    writer: Option<zstd::stream::write::Encoder<'static, BufWriter<File>>>,
    count: u64,
}

impl Fragment2DWriter {
    pub fn new(path: &Path) -> io::Result<Self> {
        let file = File::create(path)?;
        let buf = BufWriter::with_capacity(1 << 20, file);
        let encoder = zstd::stream::write::Encoder::new(buf, 3)?;
        Ok(Self {
            writer: Some(encoder),
            count: 0,
        })
    }

    pub fn write(&mut self, frag: &Fragment2D) -> io::Result<()> {
        let w = self.writer.as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Writer already flushed"))?;

        let n_verts = (frag.xy.len() / 2) as u32;

        w.write_all(MAGIC)?;
        w.write_u32::<LittleEndian>(frag.feature_id)?;
        w.write_u32::<LittleEndian>(frag.tile_z)?;
        w.write_u32::<LittleEndian>(frag.tile_x)?;
        w.write_u32::<LittleEndian>(frag.tile_y)?;
        w.write_u8(frag.geom_type)?;
        w.write_u32::<LittleEndian>(n_verts)?;

        // XY: n_verts * 2 floats (already f32)
        for &v in &frag.xy {
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

impl Drop for Fragment2DWriter {
    fn drop(&mut self) {
        if let Some(encoder) = self.writer.take() {
            let _ = encoder.finish();
        }
    }
}

/// Single-shard reader over one ZSTD-compressed fragment file.
struct ShardReader2D {
    reader: BufReader<zstd::stream::read::Decoder<'static, BufReader<File>>>,
}

impl ShardReader2D {
    fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let decoder = zstd::stream::read::Decoder::new(file)?;
        Ok(Self {
            reader: BufReader::with_capacity(1 << 20, decoder),
        })
    }

    fn read_next(&mut self) -> io::Result<Option<Fragment2D>> {
        let mut magic = [0u8; 4];
        match self.reader.read_exact(&mut magic) {
            Ok(()) => {}
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }
        if &magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad fragment2d magic"));
        }

        let feature_id = self.reader.read_u32::<LittleEndian>()?;
        let tile_z = self.reader.read_u32::<LittleEndian>()?;
        let tile_x = self.reader.read_u32::<LittleEndian>()?;
        let tile_y = self.reader.read_u32::<LittleEndian>()?;
        let geom_type = self.reader.read_u8()?;
        let n_verts = self.reader.read_u32::<LittleEndian>()? as usize;

        // XY: read f32 directly
        let mut xy = Vec::with_capacity(n_verts * 2);
        for _ in 0..n_verts * 2 {
            xy.push(self.reader.read_f32::<LittleEndian>()?);
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
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad fragment2d end magic"));
        }

        Ok(Some(Fragment2D {
            feature_id, tile_z, tile_x, tile_y,
            geom_type, xy, ring_lengths,
        }))
    }
}

/// Fragment2D reader — reads from one or more ZSTD-compressed shard files.
pub struct Fragment2DReader {
    paths: Vec<PathBuf>,
    current_shard: Option<ShardReader2D>,
    current_index: usize,
}

impl Fragment2DReader {
    pub fn new(path: &Path) -> io::Result<Self> {
        let shard = ShardReader2D::open(path)?;
        Ok(Self {
            paths: vec![path.to_path_buf()],
            current_shard: Some(shard),
            current_index: 0,
        })
    }

    /// Open a reader over a directory of shard files (*.mf2d).
    pub fn open_dir(dir: &Path) -> io::Result<Self> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "mf2d"))
            .collect();
        paths.sort();

        if paths.is_empty() {
            return Ok(Self {
                paths,
                current_shard: None,
                current_index: 0,
            });
        }

        let shard = ShardReader2D::open(&paths[0])?;
        Ok(Self {
            paths,
            current_shard: Some(shard),
            current_index: 0,
        })
    }

    pub fn read_next(&mut self) -> io::Result<Option<Fragment2D>> {
        loop {
            if let Some(ref mut shard) = self.current_shard {
                if let Some(frag) = shard.read_next()? {
                    return Ok(Some(frag));
                }
                self.current_index += 1;
                if self.current_index < self.paths.len() {
                    self.current_shard = Some(ShardReader2D::open(&self.paths[self.current_index])?);
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

    /// Read all fragments, grouped by tile key (z, x, y).
    pub fn read_all_grouped(&mut self) -> io::Result<ahash::AHashMap<(u32, u32, u32), Vec<Fragment2D>>> {
        use ahash::AHashMap;
        let mut groups: AHashMap<(u32, u32, u32), Vec<Fragment2D>> = AHashMap::new();
        while let Some(frag) = self.read_next()? {
            let key = (frag.tile_z, frag.tile_x, frag.tile_y);
            groups.entry(key).or_default().push(frag);
        }
        Ok(groups)
    }

    pub fn reset(&mut self) -> io::Result<()> {
        self.current_index = 0;
        if !self.paths.is_empty() {
            self.current_shard = Some(ShardReader2D::open(&self.paths[0])?);
        } else {
            self.current_shard = None;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let dir = std::env::temp_dir().join("test_frag2d_roundtrip");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("shard_0.mf2d");

        let frag = Fragment2D {
            feature_id: 42,
            tile_z: 2,
            tile_x: 1,
            tile_y: 3,
            geom_type: 3, // Polygon
            xy: vec![0.0f32, 0.0f32, 1.0f32, 0.0f32, 1.0f32, 1.0f32, 0.0f32, 1.0f32],
            ring_lengths: vec![4],
        };

        {
            let mut writer = Fragment2DWriter::new(&path).unwrap();
            writer.write(&frag).unwrap();
            writer.write(&frag).unwrap();
            writer.flush().unwrap();
            assert_eq!(writer.count(), 2);
        }

        {
            let mut reader = Fragment2DReader::new(&path).unwrap();
            let f1 = reader.read_next().unwrap().unwrap();
            assert_eq!(f1.feature_id, 42);
            assert_eq!(f1.tile_z, 2);
            assert_eq!(f1.xy.len(), 8);
            assert_eq!(f1.ring_lengths, vec![4]);
            assert!((f1.xy[0] - 0.0).abs() < 1e-6);

            let f2 = reader.read_next().unwrap().unwrap();
            assert_eq!(f2.feature_id, 42);
            let f3 = reader.read_next().unwrap();
            assert!(f3.is_none());
        }

        // Test reset
        {
            let mut reader = Fragment2DReader::new(&path).unwrap();
            let _f1 = reader.read_next().unwrap().unwrap();
            reader.reset().unwrap();
            let f1_again = reader.read_next().unwrap().unwrap();
            assert_eq!(f1_again.feature_id, 42);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_multi_shard_reader() {
        let dir = std::env::temp_dir().join("test_frag2d_multishard");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::create_dir_all(&dir);

        let frag_a = Fragment2D {
            feature_id: 1,
            tile_z: 0, tile_x: 0, tile_y: 0,
            geom_type: 1,
            xy: vec![0.5f32, 0.5f32],
            ring_lengths: vec![],
        };
        let frag_b = Fragment2D {
            feature_id: 2,
            tile_z: 1, tile_x: 0, tile_y: 0,
            geom_type: 2,
            xy: vec![0.0f32, 0.0f32, 1.0f32, 1.0f32],
            ring_lengths: vec![],
        };

        {
            let mut w0 = Fragment2DWriter::new(&dir.join("shard_000.mf2d")).unwrap();
            w0.write(&frag_a).unwrap();
            w0.flush().unwrap();
        }
        {
            let mut w1 = Fragment2DWriter::new(&dir.join("shard_001.mf2d")).unwrap();
            w1.write(&frag_b).unwrap();
            w1.flush().unwrap();
        }

        {
            let mut reader = Fragment2DReader::open_dir(&dir).unwrap();
            let groups = reader.read_all_grouped().unwrap();
            assert_eq!(groups.len(), 2);
            assert_eq!(groups[&(0, 0, 0)].len(), 1);
            assert_eq!(groups[&(1, 0, 0)].len(), 1);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_drop_flush() {
        let dir = std::env::temp_dir().join("test_frag2d_drop");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("shard_drop.mf2d");

        let frag = Fragment2D {
            feature_id: 7,
            tile_z: 0, tile_x: 0, tile_y: 0,
            geom_type: 1,
            xy: vec![0.3f32, 0.7f32],
            ring_lengths: vec![],
        };

        {
            let mut w = Fragment2DWriter::new(&path).unwrap();
            w.write(&frag).unwrap();
            // no flush — Drop should finalize ZSTD frame
        }
        let mut reader = Fragment2DReader::new(&path).unwrap();
        let f = reader.read_next().unwrap().unwrap();
        assert_eq!(f.feature_id, 7);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
