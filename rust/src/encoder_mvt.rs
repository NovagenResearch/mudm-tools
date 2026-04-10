/// MVT (Mapbox Vector Tile) protobuf encoder.
///
/// Hand-rolled protobuf encoding for the MVT spec — no prost dependency.
/// The MVT format is small enough to encode manually using raw wire format.

use crate::streaming::TagValue;

// -------------------------------------------------------------------------
// Low-level protobuf helpers
// -------------------------------------------------------------------------

/// Encode a varint (variable-length integer) into bytes.
pub fn encode_varint(mut value: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(10);
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            break;
        } else {
            buf.push(byte | 0x80);
        }
    }
    buf
}

/// Zigzag-encode a signed i32 for protobuf sint32.
#[inline]
pub fn encode_zigzag(n: i32) -> u32 {
    ((n << 1) ^ (n >> 31)) as u32
}

/// Encode a protobuf field tag + varint value.
pub fn encode_field_varint(field: u32, value: u64) -> Vec<u8> {
    let tag = (field << 3) | 0; // wire type 0 = varint
    let mut buf = encode_varint(tag as u64);
    buf.extend_from_slice(&encode_varint(value));
    buf
}

/// Encode a protobuf field tag + length-delimited bytes.
pub fn encode_field_bytes(field: u32, data: &[u8]) -> Vec<u8> {
    let tag = (field << 3) | 2; // wire type 2 = length-delimited
    let mut buf = encode_varint(tag as u64);
    buf.extend_from_slice(&encode_varint(data.len() as u64));
    buf.extend_from_slice(data);
    buf
}

/// Encode a protobuf field tag + string value (same as bytes).
pub fn encode_field_string(field: u32, s: &str) -> Vec<u8> {
    encode_field_bytes(field, s.as_bytes())
}

/// Encode a protobuf packed repeated uint32 field.
pub fn encode_field_packed(field: u32, values: &[u32]) -> Vec<u8> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut inner = Vec::with_capacity(values.len() * 5);
    for &v in values {
        inner.extend_from_slice(&encode_varint(v as u64));
    }
    encode_field_bytes(field, &inner)
}

// -------------------------------------------------------------------------
// MVT geometry command stream
// -------------------------------------------------------------------------

/// MVT geometry command IDs.
const CMD_MOVE_TO: u32 = 1;
const CMD_LINE_TO: u32 = 2;
const CMD_CLOSE_PATH: u32 = 7;

/// Build an MVT command integer: (count << 3) | (cmd & 0x7).
#[inline]
fn command(cmd: u32, count: u32) -> u32 {
    (count << 3) | (cmd & 0x7)
}

/// Convert normalized [0,1]² coordinates to MVT command stream.
///
/// - `geom_type`: 1=Point, 2=LineString, 3=Polygon
/// - `xy`: flat [x, y, x, y, ...] in normalized [0,1]² space
/// - `ring_lengths`: vertex counts per ring (Polygon only)
/// - `zoom`, `tile_x`, `tile_y`: tile coordinates
/// - `extent`: MVT extent (usually 4096)
pub fn encode_geometry_commands(
    geom_type: u8,
    xy: &[f64],
    ring_lengths: &[u32],
    zoom: u32,
    tile_x: u32,
    tile_y: u32,
    extent: u32,
) -> Vec<u32> {
    let n_verts = xy.len() / 2;
    if n_verts == 0 {
        return Vec::new();
    }

    let z2 = (1u64 << zoom) as f64;
    let ext = extent as f64;

    // Convert normalized coords to tile-local integers
    let mut tile_pts: Vec<[i32; 2]> = Vec::with_capacity(n_verts);
    for i in 0..n_verts {
        let nx = xy[i * 2];
        let ny = xy[i * 2 + 1];
        let tx = (ext * (nx * z2 - tile_x as f64)).round() as i32;
        let ty = (ext * (ny * z2 - tile_y as f64)).round() as i32;
        tile_pts.push([tx, ty]);
    }

    let mut cmds = Vec::with_capacity(n_verts * 3 + 4);

    match geom_type {
        1 => {
            // Point(s)
            cmds.push(command(CMD_MOVE_TO, n_verts as u32));
            let mut cx = 0i32;
            let mut cy = 0i32;
            for pt in &tile_pts {
                let dx = pt[0] - cx;
                let dy = pt[1] - cy;
                cmds.push(encode_zigzag(dx));
                cmds.push(encode_zigzag(dy));
                cx = pt[0];
                cy = pt[1];
            }
        }
        2 => {
            // LineString
            if n_verts < 2 {
                return Vec::new();
            }
            cmds.push(command(CMD_MOVE_TO, 1));
            cmds.push(encode_zigzag(tile_pts[0][0]));
            cmds.push(encode_zigzag(tile_pts[0][1]));
            cmds.push(command(CMD_LINE_TO, (n_verts - 1) as u32));
            let mut cx = tile_pts[0][0];
            let mut cy = tile_pts[0][1];
            for pt in &tile_pts[1..] {
                let dx = pt[0] - cx;
                let dy = pt[1] - cy;
                cmds.push(encode_zigzag(dx));
                cmds.push(encode_zigzag(dy));
                cx = pt[0];
                cy = pt[1];
            }
        }
        3 => {
            // Polygon — one or more rings, each gets MoveTo + LineTo + ClosePath
            let mut cx = 0i32;
            let mut cy = 0i32;
            let mut offset = 0usize;

            let rings = if ring_lengths.is_empty() {
                // Treat the whole coordinate set as a single ring
                vec![n_verts as u32]
            } else {
                ring_lengths.to_vec()
            };

            for &ring_len in &rings {
                let rl = ring_len as usize;
                if rl < 3 || offset + rl > n_verts {
                    offset += rl;
                    continue;
                }

                // MoveTo first vertex
                cmds.push(command(CMD_MOVE_TO, 1));
                let first = tile_pts[offset];
                cmds.push(encode_zigzag(first[0] - cx));
                cmds.push(encode_zigzag(first[1] - cy));
                cx = first[0];
                cy = first[1];

                // LineTo remaining vertices (skip last if it duplicates first)
                let mut line_count = rl - 1;
                if line_count > 0 {
                    let last = tile_pts[offset + rl - 1];
                    if last[0] == first[0] && last[1] == first[1] {
                        line_count -= 1;
                    }
                }

                if line_count > 0 {
                    cmds.push(command(CMD_LINE_TO, line_count as u32));
                    for i in 1..=line_count {
                        let pt = tile_pts[offset + i];
                        cmds.push(encode_zigzag(pt[0] - cx));
                        cmds.push(encode_zigzag(pt[1] - cy));
                        cx = pt[0];
                        cy = pt[1];
                    }
                }

                cmds.push(command(CMD_CLOSE_PATH, 1));

                offset += rl;
            }
        }
        _ => {}
    }

    cmds
}

// -------------------------------------------------------------------------
// MVT Value encoding
// -------------------------------------------------------------------------

/// Encode an MVT Value message.
pub fn encode_mvt_value(value: &TagValue) -> Vec<u8> {
    match value {
        TagValue::Str(s) => encode_field_string(1, s),
        TagValue::Float(f) => {
            // field 3 = double_value (wire type 1 = fixed64)
            let tag = (3u32 << 3) | 1; // field 3, wire type 1
            let mut buf = encode_varint(tag as u64);
            buf.extend_from_slice(&f.to_le_bytes());
            buf
        }
        TagValue::Int(i) => {
            if *i >= 0 {
                // field 5 = uint_value (plain varint)
                encode_field_varint(5, *i as u64)
            } else {
                // field 6 = sint_value (zigzag encoded)
                let zigzag = ((*i << 1) ^ (*i >> 63)) as u64;
                encode_field_varint(6, zigzag)
            }
        }
        TagValue::Bool(b) => {
            // field 7 = bool_value (varint)
            encode_field_varint(7, if *b { 1 } else { 0 })
        }
    }
}

// -------------------------------------------------------------------------
// MVT Feature
// -------------------------------------------------------------------------

/// A feature ready for MVT encoding.
pub struct MvtFeature {
    pub id: u64,
    pub geom_type: u8, // 1=Point, 2=LineString, 3=Polygon
    pub geometry_commands: Vec<u32>,
    pub tags: Vec<(String, TagValue)>,
}

// -------------------------------------------------------------------------
// Full tile encoder
// -------------------------------------------------------------------------

/// Encode a complete MVT tile from features.
///
/// Produces a single-layer PBF tile.
pub fn encode_mvt_tile(features: &[MvtFeature], layer_name: &str, extent: u32) -> Vec<u8> {
    if features.is_empty() {
        return Vec::new();
    }

    // Build key/value pools
    let mut keys: Vec<String> = Vec::new();
    let mut key_index: ahash::AHashMap<String, u32> = ahash::AHashMap::new();
    let mut values: Vec<TagValue> = Vec::new();
    let mut value_index: ahash::AHashMap<String, u32> = ahash::AHashMap::new();

    // Encode features
    let mut encoded_features: Vec<Vec<u8>> = Vec::with_capacity(features.len());

    for feat in features {
        let mut feat_buf = Vec::new();

        // field 1 = id (uint64)
        feat_buf.extend_from_slice(&encode_field_varint(1, feat.id));

        // field 2 = tags (packed uint32)
        let mut tag_indices: Vec<u32> = Vec::new();
        for (k, v) in &feat.tags {
            let ki = *key_index.entry(k.clone()).or_insert_with(|| {
                let idx = keys.len() as u32;
                keys.push(k.clone());
                idx
            });
            // For value dedup, create a string key
            let val_key = format!("{:?}", v);
            let vi = *value_index.entry(val_key).or_insert_with(|| {
                let idx = values.len() as u32;
                values.push(v.clone());
                idx
            });
            tag_indices.push(ki);
            tag_indices.push(vi);
        }
        if !tag_indices.is_empty() {
            feat_buf.extend_from_slice(&encode_field_packed(2, &tag_indices));
        }

        // field 3 = type (GeomType enum: POINT=1, LINESTRING=2, POLYGON=3)
        feat_buf.extend_from_slice(&encode_field_varint(3, feat.geom_type as u64));

        // field 4 = geometry (packed uint32)
        if !feat.geometry_commands.is_empty() {
            feat_buf.extend_from_slice(&encode_field_packed(4, &feat.geometry_commands));
        }

        encoded_features.push(feat_buf);
    }

    // Build Layer message
    let mut layer_buf = Vec::new();

    // field 15 = version (uint32) — must be 2
    layer_buf.extend_from_slice(&encode_field_varint(15, 2));

    // field 1 = name (string)
    layer_buf.extend_from_slice(&encode_field_string(1, layer_name));

    // field 5 = extent (uint32)
    layer_buf.extend_from_slice(&encode_field_varint(5, extent as u64));

    // field 3 = keys (repeated string)
    for k in &keys {
        layer_buf.extend_from_slice(&encode_field_string(3, k));
    }

    // field 4 = values (repeated Value message)
    for v in &values {
        let val_bytes = encode_mvt_value(v);
        layer_buf.extend_from_slice(&encode_field_bytes(4, &val_bytes));
    }

    // field 2 = features (repeated Feature message)
    for fb in &encoded_features {
        layer_buf.extend_from_slice(&encode_field_bytes(2, fb));
    }

    // Build Tile message — field 3 = layers (repeated Layer message)
    encode_field_bytes(3, &layer_buf)
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_varint_small() {
        assert_eq!(encode_varint(0), vec![0]);
        assert_eq!(encode_varint(1), vec![1]);
        assert_eq!(encode_varint(127), vec![127]);
    }

    #[test]
    fn test_encode_varint_multi_byte() {
        assert_eq!(encode_varint(128), vec![0x80, 0x01]);
        assert_eq!(encode_varint(300), vec![0xAC, 0x02]);
    }

    #[test]
    fn test_encode_zigzag() {
        assert_eq!(encode_zigzag(0), 0);
        assert_eq!(encode_zigzag(-1), 1);
        assert_eq!(encode_zigzag(1), 2);
        assert_eq!(encode_zigzag(-2), 3);
        assert_eq!(encode_zigzag(2), 4);
    }

    #[test]
    fn test_geometry_commands_point() {
        // Point at normalized (0.5, 0.5) → tile 0/0/0 → tile-local (2048, 2048)
        let cmds = encode_geometry_commands(1, &[0.5, 0.5], &[], 0, 0, 0, 4096);
        // MoveTo(1), zigzag(2048), zigzag(2048)
        assert_eq!(cmds.len(), 3);
        assert_eq!(cmds[0], command(CMD_MOVE_TO, 1));
        assert_eq!(cmds[1], encode_zigzag(2048));
        assert_eq!(cmds[2], encode_zigzag(2048));
    }

    #[test]
    fn test_geometry_commands_linestring() {
        // 3-vertex line
        let xy = vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0];
        let cmds = encode_geometry_commands(2, &xy, &[], 0, 0, 0, 4096);
        // MoveTo(1) + 2 zigzag coords = 3, LineTo(2) + 2*2 zigzag coords = 5, total = 8
        // But first point is (0,0) → tile-local (0,0), cursor starts at (0,0)
        // so MoveTo delta = (0,0), LineTo for pt1=(2048,2048) and pt2=(4096,0)
        assert_eq!(cmds.len(), 8);
        assert_eq!(cmds[0], command(CMD_MOVE_TO, 1));
        assert_eq!(cmds[3], command(CMD_LINE_TO, 2));
    }

    #[test]
    fn test_geometry_commands_polygon() {
        // Square polygon: (0,0)→(1,0)→(1,1)→(0,1)
        let xy = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let cmds = encode_geometry_commands(3, &xy, &[4], 0, 0, 0, 4096);
        // MoveTo(1) + 2, LineTo(3) + 6, ClosePath(1) = 12 ints
        assert!(!cmds.is_empty());
        // Should end with ClosePath
        assert_eq!(*cmds.last().unwrap(), command(CMD_CLOSE_PATH, 1));
    }

    #[test]
    fn test_geometry_commands_polygon_closed_ring() {
        // Ring with duplicate closing vertex — should be stripped
        let xy = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        let cmds = encode_geometry_commands(3, &xy, &[5], 0, 0, 0, 4096);
        assert!(!cmds.is_empty());
        assert_eq!(*cmds.last().unwrap(), command(CMD_CLOSE_PATH, 1));
    }

    #[test]
    fn test_encode_full_tile() {
        let cmds = encode_geometry_commands(1, &[0.5, 0.5], &[], 0, 0, 0, 4096);
        let feat = MvtFeature {
            id: 1,
            geom_type: 1,
            geometry_commands: cmds,
            tags: vec![("name".to_string(), TagValue::Str("test".to_string()))],
        };
        let tile = encode_mvt_tile(&[feat], "features", 4096);
        assert!(!tile.is_empty());
        // Should start with field 3, wire type 2 (length-delimited) = tag 0x1A
        assert_eq!(tile[0], 0x1A);
    }

    #[test]
    fn test_empty_features_produces_empty_tile() {
        let tile = encode_mvt_tile(&[], "empty", 4096);
        assert!(tile.is_empty());
    }

    #[test]
    fn test_encode_mvt_value_types() {
        let s = encode_mvt_value(&TagValue::Str("hello".to_string()));
        assert!(!s.is_empty());

        let f = encode_mvt_value(&TagValue::Float(3.14));
        assert_eq!(f.len(), 1 + 8); // tag varint + 8 bytes double

        let i = encode_mvt_value(&TagValue::Int(42));
        assert!(!i.is_empty());

        let b = encode_mvt_value(&TagValue::Bool(true));
        assert!(!b.is_empty());
    }
}
