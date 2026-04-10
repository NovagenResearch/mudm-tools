/// MVT (Mapbox Vector Tile) protobuf decoder.
///
/// Hand-rolled protobuf decoding for the MVT spec — no prost dependency.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use crate::streaming::TagValue;

// -------------------------------------------------------------------------
// Low-level protobuf helpers
// -------------------------------------------------------------------------

/// Decode a varint from a byte slice, advancing the position.
pub fn decode_varint(data: &[u8], pos: &mut usize) -> Option<u64> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    loop {
        if *pos >= data.len() {
            return None;
        }
        let byte = data[*pos];
        *pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some(result);
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
}

/// Decode a zigzag-encoded uint32 back to i32.
#[inline]
pub fn decode_zigzag(n: u32) -> i32 {
    ((n >> 1) as i32) ^ (-((n & 1) as i32))
}

/// Wire type values.
#[derive(Debug)]
pub enum WireValue {
    Varint(u64),
    Fixed64([u8; 8]),
    LengthDelimited(Vec<u8>),
    Fixed32([u8; 4]),
}

/// Decode a single protobuf field (tag + value).
pub fn decode_field(data: &[u8], pos: &mut usize) -> Option<(u32, WireValue)> {
    let tag_val = decode_varint(data, pos)?;
    let field_number = (tag_val >> 3) as u32;
    let wire_type = (tag_val & 0x7) as u8;

    let value = match wire_type {
        0 => {
            // Varint
            let v = decode_varint(data, pos)?;
            WireValue::Varint(v)
        }
        1 => {
            // Fixed64
            if *pos + 8 > data.len() { return None; }
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&data[*pos..*pos + 8]);
            *pos += 8;
            WireValue::Fixed64(buf)
        }
        2 => {
            // Length-delimited
            let len = decode_varint(data, pos)? as usize;
            if *pos + len > data.len() { return None; }
            let bytes = data[*pos..*pos + len].to_vec();
            *pos += len;
            WireValue::LengthDelimited(bytes)
        }
        5 => {
            // Fixed32
            if *pos + 4 > data.len() { return None; }
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&data[*pos..*pos + 4]);
            *pos += 4;
            WireValue::Fixed32(buf)
        }
        _ => return None,
    };

    Some((field_number, value))
}

/// Decode a packed repeated uint32 from bytes.
fn decode_packed_u32(data: &[u8]) -> Vec<u32> {
    let mut result = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        if let Some(v) = decode_varint(data, &mut pos) {
            result.push(v as u32);
        } else {
            break;
        }
    }
    result
}

// -------------------------------------------------------------------------
// MVT decoding structs
// -------------------------------------------------------------------------

/// A decoded MVT layer.
#[derive(Debug)]
pub struct MvtTileLayer {
    pub name: String,
    pub features: Vec<MvtDecodedFeature>,
    pub extent: u32,
}

/// A decoded MVT feature.
#[derive(Debug)]
pub struct MvtDecodedFeature {
    pub id: u64,
    pub geom_type: u8,
    pub points: Vec<[i32; 2]>,    // tile-local integer coords
    pub ring_lengths: Vec<u32>,
    pub tags: Vec<(String, TagValue)>,
}

// -------------------------------------------------------------------------
// Geometry command decoding
// -------------------------------------------------------------------------

/// Decode MVT geometry commands into tile-local points + ring lengths.
pub fn decode_geometry_commands(commands: &[u32], geom_type: u8) -> (Vec<[i32; 2]>, Vec<u32>) {
    let mut points = Vec::new();
    let mut ring_lengths = Vec::new();
    let mut cx: i32 = 0;
    let mut cy: i32 = 0;
    let mut i = 0;
    let mut current_ring_start = 0usize;

    while i < commands.len() {
        let cmd_int = commands[i];
        let cmd_id = cmd_int & 0x7;
        let cmd_count = cmd_int >> 3;
        i += 1;

        match cmd_id {
            1 | 2 => {
                // MoveTo (1) or LineTo (2)
                if cmd_id == 1 && geom_type == 3 {
                    // For polygon, each MoveTo starts a new ring
                    if !points.is_empty() && points.len() > current_ring_start {
                        ring_lengths.push((points.len() - current_ring_start) as u32);
                    }
                    current_ring_start = points.len();
                }
                for _ in 0..cmd_count {
                    if i + 1 >= commands.len() { break; }
                    let dx = decode_zigzag(commands[i]);
                    let dy = decode_zigzag(commands[i + 1]);
                    i += 2;
                    cx += dx;
                    cy += dy;
                    points.push([cx, cy]);
                }
            }
            7 => {
                // ClosePath — close the current ring by repeating the first point
                if geom_type == 3 && points.len() > current_ring_start {
                    // Add the closing vertex (same as ring start)
                    let start = points[current_ring_start];
                    points.push(start);
                    ring_lengths.push((points.len() - current_ring_start) as u32);
                    current_ring_start = points.len();
                }
            }
            _ => {}
        }
    }

    // Handle any remaining ring (for non-polygon types, or if no ClosePath)
    if geom_type != 3 && !points.is_empty() {
        // For points and lines, no ring_lengths needed
    } else if geom_type == 3 && points.len() > current_ring_start {
        ring_lengths.push((points.len() - current_ring_start) as u32);
    }

    (points, ring_lengths)
}

// -------------------------------------------------------------------------
// MVT Value decoding
// -------------------------------------------------------------------------

/// Decode a Value message to a TagValue.
fn decode_mvt_value(data: &[u8]) -> TagValue {
    let mut pos = 0;
    while pos < data.len() {
        if let Some((field, value)) = decode_field(data, &mut pos) {
            match (field, value) {
                (1, WireValue::LengthDelimited(bytes)) => {
                    return TagValue::Str(String::from_utf8_lossy(&bytes).to_string());
                }
                (2, WireValue::Varint(v)) => {
                    // float_value as varint? Actually float is fixed32 (field 2)
                    return TagValue::Float(v as f64);
                }
                (2, WireValue::Fixed32(bytes)) => {
                    return TagValue::Float(f32::from_le_bytes(bytes) as f64);
                }
                (3, WireValue::Fixed64(bytes)) => {
                    return TagValue::Float(f64::from_le_bytes(bytes));
                }
                (4, WireValue::Varint(v)) => {
                    // int64 — plain signed
                    return TagValue::Int(v as i64);
                }
                (5, WireValue::Varint(v)) => {
                    // uint64 — plain unsigned
                    return TagValue::Int(v as i64);
                }
                (6, WireValue::Varint(v)) => {
                    // sint64 — zigzag decode
                    let n = ((v >> 1) as i64) ^ (-((v & 1) as i64));
                    return TagValue::Int(n);
                }
                (7, WireValue::Varint(v)) => {
                    return TagValue::Bool(v != 0);
                }
                _ => {}
            }
        } else {
            break;
        }
    }
    TagValue::Str(String::new())
}

// -------------------------------------------------------------------------
// Full tile decoder
// -------------------------------------------------------------------------

/// Decode a complete MVT tile from PBF bytes.
pub fn decode_mvt_tile(data: &[u8]) -> Vec<MvtTileLayer> {
    let mut layers = Vec::new();
    let mut pos = 0;

    while pos < data.len() {
        if let Some((field, value)) = decode_field(data, &mut pos) {
            if field == 3 {
                if let WireValue::LengthDelimited(layer_data) = value {
                    if let Some(layer) = decode_layer(&layer_data) {
                        layers.push(layer);
                    }
                }
            }
        } else {
            break;
        }
    }

    layers
}

/// Decode a single MVT Layer message.
fn decode_layer(data: &[u8]) -> Option<MvtTileLayer> {
    let mut name = String::new();
    let mut extent: u32 = 4096;
    let mut keys: Vec<String> = Vec::new();
    let mut values: Vec<TagValue> = Vec::new();
    let mut raw_features: Vec<Vec<u8>> = Vec::new();

    let mut pos = 0;
    while pos < data.len() {
        if let Some((field, value)) = decode_field(data, &mut pos) {
            match field {
                1 => {
                    if let WireValue::LengthDelimited(bytes) = value {
                        name = String::from_utf8_lossy(&bytes).to_string();
                    }
                }
                2 => {
                    if let WireValue::LengthDelimited(bytes) = value {
                        raw_features.push(bytes);
                    }
                }
                3 => {
                    if let WireValue::LengthDelimited(bytes) = value {
                        keys.push(String::from_utf8_lossy(&bytes).to_string());
                    }
                }
                4 => {
                    if let WireValue::LengthDelimited(bytes) = value {
                        values.push(decode_mvt_value(&bytes));
                    }
                }
                5 => {
                    if let WireValue::Varint(v) = value {
                        extent = v as u32;
                    }
                }
                15 => {
                    // version — ignored
                }
                _ => {}
            }
        } else {
            break;
        }
    }

    // Decode features
    let mut features = Vec::with_capacity(raw_features.len());
    for feat_data in &raw_features {
        if let Some(f) = decode_feature(feat_data, &keys, &values) {
            features.push(f);
        }
    }

    Some(MvtTileLayer { name, features, extent })
}

/// Decode a single MVT Feature message.
fn decode_feature(data: &[u8], keys: &[String], values: &[TagValue]) -> Option<MvtDecodedFeature> {
    let mut id: u64 = 0;
    let mut geom_type: u8 = 0;
    let mut tag_indices: Vec<u32> = Vec::new();
    let mut geometry_commands: Vec<u32> = Vec::new();

    let mut pos = 0;
    while pos < data.len() {
        if let Some((field, value)) = decode_field(data, &mut pos) {
            match field {
                1 => {
                    if let WireValue::Varint(v) = value {
                        id = v;
                    }
                }
                2 => {
                    if let WireValue::LengthDelimited(bytes) = value {
                        tag_indices = decode_packed_u32(&bytes);
                    }
                }
                3 => {
                    if let WireValue::Varint(v) = value {
                        geom_type = v as u8;
                    }
                }
                4 => {
                    if let WireValue::LengthDelimited(bytes) = value {
                        geometry_commands = decode_packed_u32(&bytes);
                    }
                }
                _ => {}
            }
        } else {
            break;
        }
    }

    // Decode tags
    let mut tags = Vec::new();
    let mut i = 0;
    while i + 1 < tag_indices.len() {
        let ki = tag_indices[i] as usize;
        let vi = tag_indices[i + 1] as usize;
        if ki < keys.len() && vi < values.len() {
            tags.push((keys[ki].clone(), values[vi].clone()));
        }
        i += 2;
    }

    // Decode geometry
    let (points, ring_lengths) = decode_geometry_commands(&geometry_commands, geom_type);

    Some(MvtDecodedFeature {
        id,
        geom_type,
        points,
        ring_lengths,
        tags,
    })
}

// -------------------------------------------------------------------------
// PyO3: read_pbf function
// -------------------------------------------------------------------------

/// Read PBF tiles from a directory tree, converting tile-local ints → world f64 coords.
///
/// Returns a list of Python dicts compatible with `read_parquet()` format.
#[pyfunction]
#[pyo3(signature = (path, world_bounds, zoom=None, tile_x=None, tile_y=None))]
pub fn read_pbf(
    py: Python<'_>,
    path: &str,
    world_bounds: (f64, f64, f64, f64),
    zoom: Option<u32>,
    tile_x: Option<u32>,
    tile_y: Option<u32>,
) -> PyResult<PyObject> {
    let base = std::path::Path::new(path);
    let (xmin, ymin, xmax, ymax) = world_bounds;
    let dx = if xmax != xmin { xmax - xmin } else { 1.0 };
    let dy = if ymax != ymin { ymax - ymin } else { 1.0 };

    // Collect PBF files: {path}/{z}/{x}/{y}.pbf
    let mut pbf_files: Vec<(u32, u32, u32, std::path::PathBuf)> = Vec::new();

    if !base.is_dir() {
        return Ok(PyList::empty(py).into());
    }

    for z_entry in std::fs::read_dir(base).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))? {
        let z_entry = z_entry.map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let z_name = z_entry.file_name();
        let z_str = z_name.to_string_lossy();
        let z: u32 = match z_str.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(filter_z) = zoom {
            if z != filter_z { continue; }
        }

        let z_path = z_entry.path();
        if !z_path.is_dir() { continue; }

        for x_entry in std::fs::read_dir(&z_path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))? {
            let x_entry = x_entry.map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            let x_name = x_entry.file_name();
            let x_str = x_name.to_string_lossy();
            let x: u32 = match x_str.parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(filter_x) = tile_x {
                if x != filter_x { continue; }
            }

            let x_path = x_entry.path();
            if !x_path.is_dir() { continue; }

            for y_entry in std::fs::read_dir(&x_path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))? {
                let y_entry = y_entry.map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                let y_name = y_entry.file_name();
                let y_str = y_name.to_string_lossy();
                if !y_str.ends_with(".pbf") { continue; }
                let y: u32 = match y_str.trim_end_matches(".pbf").parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if let Some(filter_y) = tile_y {
                    if y != filter_y { continue; }
                }

                pbf_files.push((z, x, y, y_entry.path()));
            }
        }
    }

    // Sort for deterministic output
    pbf_files.sort_by_key(|(z, x, y, _)| (*z, *x, *y));

    // Decode each PBF file
    let result = PyList::empty(py);

    for (z, tx, ty, pbf_path) in &pbf_files {
        let data = std::fs::read(pbf_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let layers = decode_mvt_tile(&data);

        for layer in &layers {
            let z2 = (1u64 << z) as f64;
            let ext = layer.extent as f64;

            for feat in &layer.features {
                let dict = PyDict::new(py);
                dict.set_item("zoom", *z)?;
                dict.set_item("tile_x", *tx)?;
                dict.set_item("tile_y", *ty)?;
                dict.set_item("feature_id", feat.id)?;
                dict.set_item("geom_type", feat.geom_type)?;

                // Convert tile-local ints → world coords
                let n_pts = feat.points.len();
                let mut positions: Vec<f32> = Vec::with_capacity(n_pts * 2);
                for pt in &feat.points {
                    let norm_x = (*tx as f64 + pt[0] as f64 / ext) / z2;
                    let norm_y = (*ty as f64 + pt[1] as f64 / ext) / z2;
                    let wx = xmin + norm_x * dx;
                    let wy = ymin + norm_y * dy;
                    positions.push(wx as f32);
                    positions.push(wy as f32);
                }

                // Return as numpy array
                let np = py.import("numpy")?;
                let pos_list = PyList::empty(py);
                for chunk in positions.chunks(2) {
                    let pair = PyList::empty(py);
                    pair.append(chunk[0])?;
                    pair.append(chunk[1])?;
                    pos_list.append(pair)?;
                }
                let arr = np.call_method1("array", (pos_list, "float32"))?;
                dict.set_item("positions", arr)?;

                // Ring lengths
                let rl = PyList::empty(py);
                for &r in &feat.ring_lengths {
                    rl.append(r)?;
                }
                dict.set_item("ring_lengths", rl)?;

                // Tags as dict
                let tags_dict = PyDict::new(py);
                for (k, v) in &feat.tags {
                    let vs = match v {
                        TagValue::Str(s) => s.clone(),
                        TagValue::Int(i) => i.to_string(),
                        TagValue::Float(f) => f.to_string(),
                        TagValue::Bool(b) => b.to_string(),
                    };
                    tags_dict.set_item(k.as_str(), vs.as_str())?;
                }
                dict.set_item("tags", tags_dict)?;

                result.append(dict)?;
            }
        }
    }

    Ok(result.into())
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder_mvt;

    #[test]
    fn test_decode_varint() {
        let data = [0x80, 0x01];
        let mut pos = 0;
        assert_eq!(decode_varint(&data, &mut pos), Some(128));
        assert_eq!(pos, 2);
    }

    #[test]
    fn test_decode_zigzag() {
        assert_eq!(decode_zigzag(0), 0);
        assert_eq!(decode_zigzag(1), -1);
        assert_eq!(decode_zigzag(2), 1);
        assert_eq!(decode_zigzag(3), -2);
        assert_eq!(decode_zigzag(4), 2);
    }

    #[test]
    fn test_roundtrip_point() {
        let cmds = encoder_mvt::encode_geometry_commands(1, &[0.5, 0.5], &[], 0, 0, 0, 4096);
        let feat = encoder_mvt::MvtFeature {
            id: 1,
            geom_type: 1,
            geometry_commands: cmds,
            tags: vec![("name".to_string(), TagValue::Str("test_point".to_string()))],
        };
        let tile = encoder_mvt::encode_mvt_tile(&[feat], "features", 4096);
        let layers = decode_mvt_tile(&tile);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].name, "features");
        assert_eq!(layers[0].features.len(), 1);

        let f = &layers[0].features[0];
        assert_eq!(f.id, 1);
        assert_eq!(f.geom_type, 1);
        assert_eq!(f.points.len(), 1);
        assert_eq!(f.points[0], [2048, 2048]);

        assert_eq!(f.tags.len(), 1);
        assert_eq!(f.tags[0].0, "name");
        match &f.tags[0].1 {
            TagValue::Str(s) => assert_eq!(s, "test_point"),
            _ => panic!("Expected string tag"),
        }
    }

    #[test]
    fn test_roundtrip_linestring() {
        let xy = vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0];
        let cmds = encoder_mvt::encode_geometry_commands(2, &xy, &[], 0, 0, 0, 4096);
        let feat = encoder_mvt::MvtFeature {
            id: 2,
            geom_type: 2,
            geometry_commands: cmds,
            tags: vec![],
        };
        let tile = encoder_mvt::encode_mvt_tile(&[feat], "lines", 4096);
        let layers = decode_mvt_tile(&tile);
        let f = &layers[0].features[0];
        assert_eq!(f.geom_type, 2);
        assert_eq!(f.points.len(), 3);
        assert_eq!(f.points[0], [0, 0]);
        assert_eq!(f.points[1], [2048, 2048]);
        assert_eq!(f.points[2], [4096, 0]);
    }

    #[test]
    fn test_roundtrip_polygon() {
        // Square: (0,0)→(1,0)→(1,1)→(0,1)
        let xy = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let cmds = encoder_mvt::encode_geometry_commands(3, &xy, &[4], 0, 0, 0, 4096);
        let feat = encoder_mvt::MvtFeature {
            id: 3,
            geom_type: 3,
            geometry_commands: cmds,
            tags: vec![],
        };
        let tile = encoder_mvt::encode_mvt_tile(&[feat], "polys", 4096);
        let layers = decode_mvt_tile(&tile);
        let f = &layers[0].features[0];
        assert_eq!(f.geom_type, 3);
        // Should have 4 vertices (original 3 + closing point from ClosePath)
        // After ClosePath, the ring has 4 points: (0,0),(4096,0),(4096,4096),(0,4096),(0,0) — wait,
        // the encoder removes the closing dup, then ClosePath adds it back
        assert!(f.points.len() >= 4);
        assert_eq!(f.ring_lengths.len(), 1);
        // First and last point should be the same (closing vertex)
        let first = f.points[0];
        let last = f.points[f.points.len() - 1];
        assert_eq!(first, last);
    }

    #[test]
    fn test_roundtrip_polygon_with_hole() {
        // Outer ring + inner hole
        let xy = vec![
            // outer: 4 vertices
            0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
            // hole: 4 vertices
            0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8,
        ];
        let cmds = encoder_mvt::encode_geometry_commands(3, &xy, &[4, 4], 0, 0, 0, 4096);
        let feat = encoder_mvt::MvtFeature {
            id: 4,
            geom_type: 3,
            geometry_commands: cmds,
            tags: vec![],
        };
        let tile = encoder_mvt::encode_mvt_tile(&[feat], "polys", 4096);
        let layers = decode_mvt_tile(&tile);
        let f = &layers[0].features[0];
        assert_eq!(f.ring_lengths.len(), 2);
    }

    #[test]
    fn test_roundtrip_tags_all_types() {
        let cmds = encoder_mvt::encode_geometry_commands(1, &[0.5, 0.5], &[], 0, 0, 0, 4096);
        let feat = encoder_mvt::MvtFeature {
            id: 5,
            geom_type: 1,
            geometry_commands: cmds,
            tags: vec![
                ("name".to_string(), TagValue::Str("hello".to_string())),
                ("count".to_string(), TagValue::Int(42)),
                ("ratio".to_string(), TagValue::Float(3.14)),
                ("active".to_string(), TagValue::Bool(true)),
            ],
        };
        let tile = encoder_mvt::encode_mvt_tile(&[feat], "test", 4096);
        let layers = decode_mvt_tile(&tile);
        let f = &layers[0].features[0];
        assert_eq!(f.tags.len(), 4);

        // Check each tag
        let tag_map: ahash::AHashMap<&str, &TagValue> = f.tags.iter().map(|(k, v)| (k.as_str(), v)).collect();
        match tag_map["name"] {
            TagValue::Str(s) => assert_eq!(s, "hello"),
            _ => panic!("name should be Str"),
        }
        match tag_map["count"] {
            TagValue::Int(i) => assert_eq!(*i, 42),
            _ => panic!("count should be Int"),
        }
        match tag_map["ratio"] {
            TagValue::Float(f) => assert!((*f - 3.14).abs() < 1e-10),
            _ => panic!("ratio should be Float"),
        }
        match tag_map["active"] {
            TagValue::Bool(b) => assert!(*b),
            _ => panic!("active should be Bool"),
        }
    }

    #[test]
    fn test_decode_geometry_commands_point() {
        // MoveTo(1), zigzag(10)=20, zigzag(20)=40
        let cmds = vec![
            (1 << 3) | 1, // MoveTo, count=1
            20,            // zigzag(10)
            40,            // zigzag(20)
        ];
        let (pts, rl) = decode_geometry_commands(&cmds, 1);
        assert_eq!(pts, vec![[10, 20]]);
        assert!(rl.is_empty());
    }
}
