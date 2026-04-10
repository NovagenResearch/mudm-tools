/// Hand-rolled protobuf encoder for PBF3 (3D protobuf vector tiles).
///
/// Replaces encoder3d_cy.pyx. Encodes tiles directly to protobuf wire
/// format without any protobuf library dependency. Implements:
/// - Varint encoding
/// - Zigzag encoding for signed integers
/// - Delta-encoded XY geometry
/// - Dictionary-encoded tags
/// - Indexed mesh encoding (vertex dedup + float32 positions + uint32 indices)
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
use ahash::AHashMap;

// ---------------------------------------------------------------------------
// Protobuf wire format helpers
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn encode_varint(buf: &mut Vec<u8>, mut val: u64) {
    loop {
        let byte = (val & 0x7F) as u8;
        val >>= 7;
        if val == 0 {
            buf.push(byte);
            return;
        }
        buf.push(byte | 0x80);
    }
}

#[inline]
pub(crate) fn zigzag(n: i64) -> u64 {
    ((n << 1) ^ (n >> 63)) as u64
}

/// Write a protobuf field tag (field_number, wire_type).
#[inline]
pub(crate) fn write_tag(buf: &mut Vec<u8>, field: u32, wire_type: u32) {
    encode_varint(buf, ((field as u64) << 3) | wire_type as u64);
}

/// Write a length-delimited field (wire type 2).
pub(crate) fn write_bytes_field(buf: &mut Vec<u8>, field: u32, data: &[u8]) {
    write_tag(buf, field, 2);
    encode_varint(buf, data.len() as u64);
    buf.extend_from_slice(data);
}

/// Write a varint field (wire type 0).
#[inline]
pub(crate) fn write_varint_field(buf: &mut Vec<u8>, field: u32, val: u64) {
    write_tag(buf, field, 0);
    encode_varint(buf, val);
}

// ---------------------------------------------------------------------------
// Geometry encoding helpers (MVT command-based)
// ---------------------------------------------------------------------------

#[inline]
fn mvt_command(cmd: u32, count: u32) -> u32 {
    (count << 3) | (cmd & 0x7)
}

pub(crate) fn encode_point_geometry(xy: &[i64]) -> Vec<u32> {
    let n = xy.len() / 2;
    let mut result = Vec::with_capacity(1 + n * 2);
    result.push(mvt_command(1, n as u32));
    let mut x: i64 = 0;
    let mut y: i64 = 0;
    for i in 0..n {
        let dx = xy[i * 2] - x;
        let dy = xy[i * 2 + 1] - y;
        result.push(zigzag(dx) as u32);
        result.push(zigzag(dy) as u32);
        x += dx;
        y += dy;
    }
    result
}

pub(crate) fn encode_line_geometry(xy: &[i64]) -> Vec<u32> {
    let n = xy.len() / 2;
    if n < 2 {
        return vec![];
    }
    let mut result = Vec::with_capacity(4 + (n - 1) * 2);
    let mut x: i64 = 0;
    let mut y: i64 = 0;

    // MoveTo first point
    result.push(mvt_command(1, 1));
    let dx = xy[0] - x;
    let dy = xy[1] - y;
    result.push(zigzag(dx) as u32);
    result.push(zigzag(dy) as u32);
    x += dx;
    y += dy;

    // LineTo remaining
    result.push(mvt_command(2, (n - 1) as u32));
    for i in 1..n {
        let dx = xy[i * 2] - x;
        let dy = xy[i * 2 + 1] - y;
        result.push(zigzag(dx) as u32);
        result.push(zigzag(dy) as u32);
        x += dx;
        y += dy;
    }
    result
}

pub(crate) fn encode_polygon_geometry(xy: &[i64], ring_lengths: &[usize]) -> Vec<u32> {
    let mut result = Vec::new();
    let mut x: i64 = 0;
    let mut y: i64 = 0;
    let mut offset: usize = 0;

    for &ring_len in ring_lengths {
        if ring_len < 3 {
            offset += ring_len;
            continue;
        }

        // MoveTo first
        result.push(mvt_command(1, 1));
        let dx = xy[offset * 2] - x;
        let dy = xy[offset * 2 + 1] - y;
        result.push(zigzag(dx) as u32);
        result.push(zigzag(dy) as u32);
        x += dx;
        y += dy;

        // LineTo interior
        let line_count = ring_len - 2;
        if line_count > 0 {
            result.push(mvt_command(2, line_count as u32));
            for i in 1..ring_len - 1 {
                let idx = offset + i;
                let dx = xy[idx * 2] - x;
                let dy = xy[idx * 2 + 1] - y;
                result.push(zigzag(dx) as u32);
                result.push(zigzag(dy) as u32);
                x += dx;
                y += dy;
            }
        }

        // ClosePath
        result.push(mvt_command(7, 1));
        offset += ring_len;
    }
    result
}

pub(crate) fn encode_z(z_coords: &[i64]) -> Vec<i64> {
    if z_coords.is_empty() {
        return vec![];
    }
    let mut result = Vec::with_capacity(z_coords.len());
    result.push(z_coords[0]);
    for i in 1..z_coords.len() {
        result.push(z_coords[i] - z_coords[i - 1]);
    }
    result
}

// ---------------------------------------------------------------------------
// Indexed mesh builder
// ---------------------------------------------------------------------------

/// Build indexed triangle mesh from ring-based face data.
/// Returns (positions_bytes, indices_bytes).
#[pyfunction]
pub fn build_indexed_mesh(
    xy: Vec<i64>,
    z_coords: Vec<i64>,
    ring_lengths: Option<Vec<usize>>,
) -> PyResult<(Vec<u8>, Vec<u8>)> {
    let ring_lengths = ring_lengths.unwrap_or_else(|| vec![xy.len() / 2]);

    let mut vertex_map: AHashMap<(i64, i64, i64), u32> = AHashMap::new();
    let mut positions: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let mut offset: usize = 0;
    for &ring_len in &ring_lengths {
        let n_verts = if ring_len >= 4 { 3 } else { ring_len };

        if n_verts == 3 {
            let mut tri = [0u32; 3];
            for vi_off in 0..3 {
                let vi = offset + vi_off;
                let x = xy[vi * 2];
                let y = xy[vi * 2 + 1];
                let z = if vi < z_coords.len() { z_coords[vi] } else { 0 };
                let key = (x, y, z);
                let idx = match vertex_map.get(&key) {
                    Some(&idx) => idx,
                    None => {
                        let idx = vertex_map.len() as u32;
                        vertex_map.insert(key, idx);
                        positions.push(x as f32);
                        positions.push(y as f32);
                        positions.push(z as f32);
                        idx
                    }
                };
                tri[vi_off] = idx;
            }
            indices.extend_from_slice(&tri);
        }
        offset += ring_len;
    }

    // Pack to bytes
    let pos_bytes: Vec<u8> = positions.iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let idx_bytes: Vec<u8> = indices.iter()
        .flat_map(|i| i.to_le_bytes())
        .collect();

    Ok((pos_bytes, idx_bytes))
}

// ---------------------------------------------------------------------------
// Protobuf tile field numbers (matching microjson_3d_tile.proto)
// ---------------------------------------------------------------------------
// Tile: layers (field 3, repeated, length-delimited)
// Layer: version (15, uint32), name (1, string), features (2, repeated msg),
//        keys (3, repeated string), values (4, repeated msg),
//        extent (5, uint32), extent_z (7, uint32)
// Feature: id (1, uint64), tags (2, packed uint32), type (3, enum/varint),
//          geometry (4, packed uint32), geometry_z (6, packed sint32),
//          mesh_positions (7, bytes), mesh_indices (8, bytes),
//          radii (9, packed float)
// Value: string_value (1), float_value (2), double_value (3),
//        int_value (4), uint_value (5), sint_value (6), bool_value (7)

pub(crate) const TILE_LAYERS: u32 = 3;
pub(crate) const LAYER_NAME: u32 = 1;
pub(crate) const LAYER_FEATURES: u32 = 2;
pub(crate) const LAYER_KEYS: u32 = 3;
pub(crate) const LAYER_VALUES: u32 = 4;
pub(crate) const LAYER_EXTENT: u32 = 5;
pub(crate) const LAYER_EXTENT_Z: u32 = 6;
pub(crate) const LAYER_VERSION: u32 = 15;
pub(crate) const FEAT_ID: u32 = 1;
pub(crate) const FEAT_TAGS: u32 = 2;
pub(crate) const FEAT_TYPE: u32 = 3;
pub(crate) const FEAT_GEOMETRY: u32 = 4;
pub(crate) const FEAT_GEOMETRY_Z: u32 = 5;
pub(crate) const FEAT_MESH_POSITIONS: u32 = 7;
pub(crate) const FEAT_MESH_INDICES: u32 = 8;
pub(crate) const FEAT_RADII: u32 = 6;
pub(crate) const VALUE_STRING: u32 = 1;
pub(crate) const VALUE_DOUBLE: u32 = 3;
pub(crate) const VALUE_UINT: u32 = 5;
pub(crate) const VALUE_SINT: u32 = 6;
pub(crate) const VALUE_BOOL: u32 = 7;

/// Encode packed repeated uint32.
pub(crate) fn encode_packed_uint32(buf: &mut Vec<u8>, field: u32, values: &[u32]) {
    if values.is_empty() {
        return;
    }
    let mut packed = Vec::new();
    for &v in values {
        encode_varint(&mut packed, v as u64);
    }
    write_bytes_field(buf, field, &packed);
}

/// Encode packed repeated sint32 (zigzag).
pub(crate) fn encode_packed_sint32(buf: &mut Vec<u8>, field: u32, values: &[i64]) {
    if values.is_empty() {
        return;
    }
    let mut packed = Vec::new();
    for &v in values {
        encode_varint(&mut packed, zigzag(v));
    }
    write_bytes_field(buf, field, &packed);
}

/// Encode packed repeated float.
pub(crate) fn encode_packed_float(buf: &mut Vec<u8>, field: u32, values: &[f32]) {
    if values.is_empty() {
        return;
    }
    let bytes: Vec<u8> = values.iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    write_bytes_field(buf, field, &bytes);
}

/// Encode a Value message.
fn encode_value(value: &Bound<'_, pyo3::PyAny>) -> PyResult<Vec<u8>> {
    let mut buf = Vec::new();
    if value.is_instance_of::<PyBool>() {
        let b: bool = value.extract()?;
        write_varint_field(&mut buf, VALUE_BOOL, if b { 1 } else { 0 });
    } else if value.is_instance_of::<PyString>() {
        let s: String = value.extract()?;
        write_bytes_field(&mut buf, VALUE_STRING, s.as_bytes());
    } else if value.is_instance_of::<PyFloat>() {
        let d: f64 = value.extract()?;
        write_tag(&mut buf, VALUE_DOUBLE, 1); // wire type 1 = 64-bit
        buf.extend_from_slice(&d.to_le_bytes());
    } else if value.is_instance_of::<PyInt>() {
        let i: i64 = value.extract()?;
        if i < 0 {
            write_tag(&mut buf, VALUE_SINT, 0);
            encode_varint(&mut buf, zigzag(i));
        } else {
            write_varint_field(&mut buf, VALUE_UINT, i as u64);
        }
    }
    Ok(buf)
}

/// Encode a transformed tile to protobuf bytes.
///
/// Hand-rolled protobuf encoding matching microjson_3d_tile.proto.
#[pyfunction]
#[pyo3(signature = (tile_data, layer_name="default", extent=4096, extent_z=4096))]
pub fn encode_tile_3d(
    py: Python<'_>,
    tile_data: &Bound<'_, PyDict>,
    layer_name: &str,
    extent: u32,
    extent_z: u32,
) -> PyResult<Vec<u8>> {
    let features: Vec<Bound<'_, PyDict>> = tile_data
        .get_item("features")?
        .unwrap_or_else(|| PyList::empty(py).into_any())
        .extract()?;

    let mut key_indices: AHashMap<String, u32> = AHashMap::new();
    let mut value_indices: AHashMap<String, u32> = AHashMap::new(); // stringified key
    let mut keys_list: Vec<String> = Vec::new();
    let mut values_encoded: Vec<Vec<u8>> = Vec::new();

    let mut features_encoded: Vec<Vec<u8>> = Vec::new();

    for (feat_idx, feat) in features.iter().enumerate() {
        let mut feat_buf = Vec::new();

        // id
        write_varint_field(&mut feat_buf, FEAT_ID, feat_idx as u64);

        // type
        let gt: i32 = feat.get_item("type")?.map(|v| v.extract().unwrap_or(0)).unwrap_or(0);
        write_varint_field(&mut feat_buf, FEAT_TYPE, gt as u64);

        // tags (dictionary-encoded)
        let tags = feat.get_item("tags")?;
        if let Some(tags) = tags {
            if let Ok(tags_dict) = tags.downcast::<PyDict>() {
                let mut tag_indices: Vec<u32> = Vec::new();
                for (k, v) in tags_dict.iter() {
                    if v.is_none() {
                        continue;
                    }
                    let key_str: String = k.extract()?;

                    // Key index
                    let ki = match key_indices.get(&key_str) {
                        Some(&idx) => idx,
                        None => {
                            let idx = key_indices.len() as u32;
                            key_indices.insert(key_str.clone(), idx);
                            keys_list.push(key_str);
                            idx
                        }
                    };
                    tag_indices.push(ki);

                    // Value index — use repr for dedup key
                    let v_repr: String = v.repr()?.extract()?;
                    let vi = match value_indices.get(&v_repr) {
                        Some(&idx) => idx,
                        None => {
                            let idx = value_indices.len() as u32;
                            value_indices.insert(v_repr, idx);
                            values_encoded.push(encode_value(&v)?);
                            idx
                        }
                    };
                    tag_indices.push(vi);
                }
                encode_packed_uint32(&mut feat_buf, FEAT_TAGS, &tag_indices);
            }
        }

        // geometry
        let xy: Vec<i64> = feat.get_item("geometry")?
            .map(|v| v.extract().unwrap_or_default())
            .unwrap_or_default();
        let z_coords: Vec<i64> = feat.get_item("geometry_z")?
            .map(|v| v.extract().unwrap_or_default())
            .unwrap_or_default();
        let ring_lengths: Option<Vec<usize>> = feat.get_item("ring_lengths")?
            .and_then(|v| if v.is_none() { None } else { v.extract().ok() });

        if gt == 1 {
            // POINT3D
            let geom = encode_point_geometry(&xy);
            encode_packed_uint32(&mut feat_buf, FEAT_GEOMETRY, &geom);
            let z_enc = encode_z(&z_coords);
            encode_packed_sint32(&mut feat_buf, FEAT_GEOMETRY_Z, &z_enc);
        } else if gt == 2 {
            // LINESTRING3D
            let geom = encode_line_geometry(&xy);
            encode_packed_uint32(&mut feat_buf, FEAT_GEOMETRY, &geom);
            let z_enc = encode_z(&z_coords);
            encode_packed_sint32(&mut feat_buf, FEAT_GEOMETRY_Z, &z_enc);
        } else if gt == 4 || gt == 5 {
            // POLYHEDRALSURFACE, TIN — indexed mesh
            let rls = ring_lengths.unwrap_or_else(|| vec![xy.len() / 2]);
            let mut vertex_map: AHashMap<(i64, i64, i64), u32> = AHashMap::new();
            let mut positions: Vec<f32> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();
            let mut offset: usize = 0;
            for &rl in &rls {
                let nv = if rl >= 4 { 3 } else { rl };
                if nv == 3 {
                    let mut tri = [0u32; 3];
                    for vi_off in 0..3 {
                        let vi = offset + vi_off;
                        let x = xy[vi * 2];
                        let y = xy[vi * 2 + 1];
                        let z = if vi < z_coords.len() { z_coords[vi] } else { 0 };
                        let key = (x, y, z);
                        let idx = match vertex_map.get(&key) {
                            Some(&idx) => idx,
                            None => {
                                let idx = vertex_map.len() as u32;
                                vertex_map.insert(key, idx);
                                positions.push(x as f32);
                                positions.push(y as f32);
                                positions.push(z as f32);
                                idx
                            }
                        };
                        tri[vi_off] = idx;
                    }
                    indices.extend_from_slice(&tri);
                }
                offset += rl;
            }
            let pos_bytes: Vec<u8> = positions.iter().flat_map(|f| f.to_le_bytes()).collect();
            let idx_bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
            write_bytes_field(&mut feat_buf, FEAT_MESH_POSITIONS, &pos_bytes);
            write_bytes_field(&mut feat_buf, FEAT_MESH_INDICES, &idx_bytes);
        } else if gt == 3 {
            // POLYGON3D
            let rls = ring_lengths.unwrap_or_else(|| vec![xy.len() / 2]);
            let geom = encode_polygon_geometry(&xy, &rls);
            encode_packed_uint32(&mut feat_buf, FEAT_GEOMETRY, &geom);
            let z_enc = encode_z(&z_coords);
            encode_packed_sint32(&mut feat_buf, FEAT_GEOMETRY_Z, &z_enc);
        } else {
            // Default: line encoding
            let geom = encode_line_geometry(&xy);
            encode_packed_uint32(&mut feat_buf, FEAT_GEOMETRY, &geom);
            let z_enc = encode_z(&z_coords);
            encode_packed_sint32(&mut feat_buf, FEAT_GEOMETRY_Z, &z_enc);
        }

        // Per-vertex radii
        if let Some(radii) = feat.get_item("radii")? {
            if !radii.is_none() {
                let radii_vec: Vec<f32> = radii.extract()?;
                encode_packed_float(&mut feat_buf, FEAT_RADII, &radii_vec);
            }
        }

        features_encoded.push(feat_buf);
    }

    // --- Encode layer ---
    let mut layer_buf = Vec::new();

    // version
    write_varint_field(&mut layer_buf, LAYER_VERSION, 3);

    // name
    write_bytes_field(&mut layer_buf, LAYER_NAME, layer_name.as_bytes());

    // features
    for feat_buf in &features_encoded {
        write_bytes_field(&mut layer_buf, LAYER_FEATURES, feat_buf);
    }

    // keys
    for key in &keys_list {
        write_bytes_field(&mut layer_buf, LAYER_KEYS, key.as_bytes());
    }

    // values
    for val_buf in &values_encoded {
        write_bytes_field(&mut layer_buf, LAYER_VALUES, val_buf);
    }

    // extent
    write_varint_field(&mut layer_buf, LAYER_EXTENT, extent as u64);
    write_varint_field(&mut layer_buf, LAYER_EXTENT_Z, extent_z as u64);

    // --- Encode tile ---
    let mut tile_buf = Vec::new();
    write_bytes_field(&mut tile_buf, TILE_LAYERS, &layer_buf);

    Ok(tile_buf)
}
