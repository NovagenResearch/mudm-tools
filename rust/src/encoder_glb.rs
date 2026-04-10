/// Minimal glTF 2.0 binary (GLB) encoder for OGC 3D Tiles output.
///
/// Produces a single GLB file containing one or more nodes, each with
/// a mesh primitive (triangles, lines, or points). Feature properties
/// are stored in node `extras`.
///
/// No external dependencies beyond serde_json + byteorder.

use serde_json::{json, Value};
use crate::encoder_draco;
use crate::encoder_meshopt;

// glTF component types
const FLOAT: u32 = 5126;
const UNSIGNED_INT: u32 = 5125;

// Buffer view targets
const ARRAY_BUFFER: u32 = 34962;
const ELEMENT_ARRAY_BUFFER: u32 = 34963;

// Primitive modes
pub(crate) const MODE_POINTS: u32 = 0;
pub(crate) const MODE_LINES: u32 = 1;
pub(crate) const MODE_TRIANGLES: u32 = 4;

// GLB header
const GLB_MAGIC: u32 = 0x46546C67; // "glTF"
const GLB_VERSION: u32 = 2;
const JSON_CHUNK_TYPE: u32 = 0x4E4F534A; // "JSON"
const BIN_CHUNK_TYPE: u32 = 0x004E4942; // "BIN\0"

/// A feature to be encoded into GLB.
#[derive(Clone)]
pub(crate) struct GlbFeature {
    /// Flat f32 positions [x,y,z, x,y,z, ...] in world coordinates.
    pub positions: Vec<f32>,
    /// Triangle/line indices (empty for points).
    pub indices: Vec<u32>,
    /// GL primitive mode (0=POINTS, 1=LINES, 4=TRIANGLES).
    pub mode: u32,
    /// Feature metadata stored in node.extras.
    pub extras: Option<Value>,
}

/// Encode a set of features as a single GLB binary.
pub(crate) fn encode_glb(features: &[GlbFeature]) -> Vec<u8> {
    let mut bin_data: Vec<u8> = Vec::new();
    let mut buffer_views: Vec<Value> = Vec::new();
    let mut accessors: Vec<Value> = Vec::new();
    let mut meshes: Vec<Value> = Vec::new();
    let mut nodes: Vec<Value> = Vec::new();
    let mut scene_nodes: Vec<Value> = Vec::new();

    for feat in features {
        if feat.positions.is_empty() {
            continue;
        }

        let n_verts = feat.positions.len() / 3;

        // Compute min/max for position accessor
        let mut pos_min = [f32::INFINITY; 3];
        let mut pos_max = [f32::NEG_INFINITY; 3];
        for v in 0..n_verts {
            for d in 0..3 {
                let val = feat.positions[v * 3 + d];
                if val < pos_min[d] {
                    pos_min[d] = val;
                }
                if val > pos_max[d] {
                    pos_max[d] = val;
                }
            }
        }

        // --- Positions buffer view ---
        let pos_offset = bin_data.len();
        let pos_bytes: Vec<u8> = feat
            .positions
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let pos_byte_len = pos_bytes.len();
        bin_data.extend_from_slice(&pos_bytes);
        // Pad to 4 bytes
        let pad = (4 - (bin_data.len() % 4)) % 4;
        bin_data.extend(std::iter::repeat(0u8).take(pad));

        let pos_bv_idx = buffer_views.len();
        buffer_views.push(json!({
            "buffer": 0,
            "byteOffset": pos_offset,
            "byteLength": pos_byte_len,
            "target": ARRAY_BUFFER,
        }));

        let pos_acc_idx = accessors.len();
        accessors.push(json!({
            "bufferView": pos_bv_idx,
            "byteOffset": 0,
            "componentType": FLOAT,
            "count": n_verts,
            "type": "VEC3",
            "min": [pos_min[0], pos_min[1], pos_min[2]],
            "max": [pos_max[0], pos_max[1], pos_max[2]],
        }));

        // --- Indices buffer view (if any) ---
        let primitive = if !feat.indices.is_empty() {
            let idx_offset = bin_data.len();
            let idx_bytes: Vec<u8> = feat
                .indices
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect();
            let idx_byte_len = idx_bytes.len();
            bin_data.extend_from_slice(&idx_bytes);
            let pad = (4 - (bin_data.len() % 4)) % 4;
            bin_data.extend(std::iter::repeat(0u8).take(pad));

            let idx_bv_idx = buffer_views.len();
            buffer_views.push(json!({
                "buffer": 0,
                "byteOffset": idx_offset,
                "byteLength": idx_byte_len,
                "target": ELEMENT_ARRAY_BUFFER,
            }));

            let idx_acc_idx = accessors.len();
            accessors.push(json!({
                "bufferView": idx_bv_idx,
                "byteOffset": 0,
                "componentType": UNSIGNED_INT,
                "count": feat.indices.len(),
                "type": "SCALAR",
            }));

            json!({
                "attributes": {"POSITION": pos_acc_idx},
                "indices": idx_acc_idx,
                "material": 0,
                "mode": feat.mode,
            })
        } else {
            json!({
                "attributes": {"POSITION": pos_acc_idx},
                "material": 0,
                "mode": feat.mode,
            })
        };

        let mesh_idx = meshes.len();
        meshes.push(json!({"primitives": [primitive]}));

        let mut node = json!({"mesh": mesh_idx, "name": format!("feature_{}", nodes.len())});
        if let Some(extras) = &feat.extras {
            node.as_object_mut()
                .unwrap()
                .insert("extras".to_string(), extras.clone());
        }

        let node_idx = nodes.len();
        nodes.push(node);
        scene_nodes.push(json!(node_idx));
    }

    // If no features, return minimal valid GLB
    if meshes.is_empty() {
        return encode_empty_glb();
    }

    // Default material — matches Python pipeline (metallicFactor=0.1, roughnessFactor=0.8)
    let material = json!({
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.8, 0.8, 0.8, 1.0],
            "metallicFactor": 0.1,
            "roughnessFactor": 0.8,
        },
        "doubleSided": true,
    });

    // Assemble JSON descriptor
    let gltf_json = json!({
        "asset": {"version": "2.0", "generator": "mudm-rs"},
        "scene": 0,
        "scenes": [{"nodes": scene_nodes}],
        "nodes": nodes,
        "meshes": meshes,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": bin_data.len()}],
        "materials": [material],
    });

    assemble_glb(&gltf_json, &bin_data)
}

/// Encode features as GLB with KHR_draco_mesh_compression for eligible triangle meshes.
///
/// Triangle meshes with `positions.len()/3 >= min_vertices` are Draco-encoded
/// using raw f32 world-space positions (Draco handles internal quantization).
/// Non-triangle features, small meshes, and Draco failures fall back to raw encoding.
pub(crate) fn encode_glb_draco(
    features: &[GlbFeature],
    min_vertices: usize,
) -> Vec<u8> {
    let mut bin_data: Vec<u8> = Vec::new();
    let mut buffer_views: Vec<Value> = Vec::new();
    let mut accessors: Vec<Value> = Vec::new();
    let mut meshes: Vec<Value> = Vec::new();
    let mut nodes: Vec<Value> = Vec::new();
    let mut scene_nodes: Vec<Value> = Vec::new();
    let mut any_draco = false;

    for feat in features {
        if feat.positions.is_empty() {
            continue;
        }

        let n_verts = feat.positions.len() / 3;

        // Compute min/max for position accessor (needed for both paths)
        let mut pos_min = [f32::INFINITY; 3];
        let mut pos_max = [f32::NEG_INFINITY; 3];
        for v in 0..n_verts {
            for d in 0..3 {
                let val = feat.positions[v * 3 + d];
                if val < pos_min[d] { pos_min[d] = val; }
                if val > pos_max[d] { pos_max[d] = val; }
            }
        }

        // Try Draco for eligible triangle meshes
        let use_draco_for_this = feat.mode == MODE_TRIANGLES
            && !feat.indices.is_empty()
            && n_verts >= min_vertices;

        let draco_result = if use_draco_for_this {
            // Encode raw f32 world-space positions — Draco handles internal quantization.
            // The viewer will decode world-space positions directly.
            encoder_draco::encode_draco_mesh_f32(&feat.positions, &feat.indices).ok()
        } else {
            None
        };

        if let Some(draco_bytes) = draco_result {
            // --- Draco-compressed primitive ---
            any_draco = true;

            // Draco buffer view (no target — it's an opaque blob)
            let draco_offset = bin_data.len();
            let draco_byte_len = draco_bytes.len();
            bin_data.extend_from_slice(&draco_bytes);
            let pad = (4 - (bin_data.len() % 4)) % 4;
            bin_data.extend(std::iter::repeat(0u8).take(pad));

            let draco_bv_idx = buffer_views.len();
            buffer_views.push(json!({
                "buffer": 0,
                "byteOffset": draco_offset,
                "byteLength": draco_byte_len,
            }));

            // Stub position accessor — no bufferView, but has count/min/max
            let pos_acc_idx = accessors.len();
            accessors.push(json!({
                "componentType": FLOAT,
                "count": n_verts,
                "type": "VEC3",
                "min": [pos_min[0], pos_min[1], pos_min[2]],
                "max": [pos_max[0], pos_max[1], pos_max[2]],
            }));

            // Stub index accessor — no bufferView
            let idx_acc_idx = accessors.len();
            accessors.push(json!({
                "componentType": UNSIGNED_INT,
                "count": feat.indices.len(),
                "type": "SCALAR",
            }));

            let primitive = json!({
                "attributes": {"POSITION": pos_acc_idx},
                "indices": idx_acc_idx,
                "material": 0,
                "mode": MODE_TRIANGLES,
                "extensions": {
                    "KHR_draco_mesh_compression": {
                        "bufferView": draco_bv_idx,
                        "attributes": {
                            "POSITION": 0
                        }
                    }
                }
            });

            let mesh_idx = meshes.len();
            meshes.push(json!({"primitives": [primitive]}));

            let mut node = json!({"mesh": mesh_idx, "name": format!("feature_{}", nodes.len())});
            if let Some(extras) = &feat.extras {
                node.as_object_mut().unwrap().insert("extras".to_string(), extras.clone());
            }
            let node_idx = nodes.len();
            nodes.push(node);
            scene_nodes.push(json!(node_idx));
        } else {
            // --- Raw encoding (same as encode_glb) ---
            let pos_offset = bin_data.len();
            let pos_bytes: Vec<u8> = feat.positions.iter().flat_map(|f| f.to_le_bytes()).collect();
            let pos_byte_len = pos_bytes.len();
            bin_data.extend_from_slice(&pos_bytes);
            let pad = (4 - (bin_data.len() % 4)) % 4;
            bin_data.extend(std::iter::repeat(0u8).take(pad));

            let pos_bv_idx = buffer_views.len();
            buffer_views.push(json!({
                "buffer": 0,
                "byteOffset": pos_offset,
                "byteLength": pos_byte_len,
                "target": ARRAY_BUFFER,
            }));

            let pos_acc_idx = accessors.len();
            accessors.push(json!({
                "bufferView": pos_bv_idx,
                "byteOffset": 0,
                "componentType": FLOAT,
                "count": n_verts,
                "type": "VEC3",
                "min": [pos_min[0], pos_min[1], pos_min[2]],
                "max": [pos_max[0], pos_max[1], pos_max[2]],
            }));

            let primitive = if !feat.indices.is_empty() {
                let idx_offset = bin_data.len();
                let idx_bytes: Vec<u8> = feat.indices.iter().flat_map(|i| i.to_le_bytes()).collect();
                let idx_byte_len = idx_bytes.len();
                bin_data.extend_from_slice(&idx_bytes);
                let pad = (4 - (bin_data.len() % 4)) % 4;
                bin_data.extend(std::iter::repeat(0u8).take(pad));

                let idx_bv_idx = buffer_views.len();
                buffer_views.push(json!({
                    "buffer": 0,
                    "byteOffset": idx_offset,
                    "byteLength": idx_byte_len,
                    "target": ELEMENT_ARRAY_BUFFER,
                }));

                let idx_acc_idx = accessors.len();
                accessors.push(json!({
                    "bufferView": idx_bv_idx,
                    "byteOffset": 0,
                    "componentType": UNSIGNED_INT,
                    "count": feat.indices.len(),
                    "type": "SCALAR",
                }));

                json!({
                    "attributes": {"POSITION": pos_acc_idx},
                    "indices": idx_acc_idx,
                    "material": 0,
                    "mode": feat.mode,
                })
            } else {
                json!({
                    "attributes": {"POSITION": pos_acc_idx},
                    "material": 0,
                    "mode": feat.mode,
                })
            };

            let mesh_idx = meshes.len();
            meshes.push(json!({"primitives": [primitive]}));

            let mut node = json!({"mesh": mesh_idx, "name": format!("feature_{}", nodes.len())});
            if let Some(extras) = &feat.extras {
                node.as_object_mut().unwrap().insert("extras".to_string(), extras.clone());
            }
            let node_idx = nodes.len();
            nodes.push(node);
            scene_nodes.push(json!(node_idx));
        }
    }

    if meshes.is_empty() {
        return encode_empty_glb();
    }

    let material = json!({
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.8, 0.8, 0.8, 1.0],
            "metallicFactor": 0.1,
            "roughnessFactor": 0.8,
        },
        "doubleSided": true,
    });

    let mut gltf_json = json!({
        "asset": {"version": "2.0", "generator": "mudm-rs"},
        "scene": 0,
        "scenes": [{"nodes": scene_nodes}],
        "nodes": nodes,
        "meshes": meshes,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": bin_data.len()}],
        "materials": [material],
    });

    if any_draco {
        gltf_json.as_object_mut().unwrap().insert(
            "extensionsUsed".to_string(),
            json!(["KHR_draco_mesh_compression"]),
        );
        gltf_json.as_object_mut().unwrap().insert(
            "extensionsRequired".to_string(),
            json!(["KHR_draco_mesh_compression"]),
        );
    }

    assemble_glb(&gltf_json, &bin_data)
}

/// Encode features as GLB with EXT_meshopt_compression for eligible triangle meshes.
///
/// Triangle meshes with `positions.len()/3 >= min_vertices` are meshopt-encoded.
/// Compression works at the bufferView level: each compressed bufferView has an
/// `EXT_meshopt_compression` extension object while accessors remain normal.
/// Non-triangle features, small meshes, and encoding failures fall back to raw.
pub(crate) fn encode_glb_meshopt(
    features: &[GlbFeature],
    min_vertices: usize,
) -> Vec<u8> {
    let mut bin_data: Vec<u8> = Vec::new();
    let mut buffer_views: Vec<Value> = Vec::new();
    let mut accessors: Vec<Value> = Vec::new();
    let mut meshes: Vec<Value> = Vec::new();
    let mut nodes: Vec<Value> = Vec::new();
    let mut scene_nodes: Vec<Value> = Vec::new();
    let mut any_meshopt = false;

    for feat in features {
        if feat.positions.is_empty() {
            continue;
        }

        let n_verts = feat.positions.len() / 3;

        // Compute min/max for position accessor
        let mut pos_min = [f32::INFINITY; 3];
        let mut pos_max = [f32::NEG_INFINITY; 3];
        for v in 0..n_verts {
            for d in 0..3 {
                let val = feat.positions[v * 3 + d];
                if val < pos_min[d] { pos_min[d] = val; }
                if val > pos_max[d] { pos_max[d] = val; }
            }
        }

        // Try meshopt for eligible triangle meshes
        let use_meshopt_for_this = feat.mode == MODE_TRIANGLES
            && !feat.indices.is_empty()
            && n_verts >= min_vertices;

        let meshopt_result = if use_meshopt_for_this {
            encoder_meshopt::encode_meshopt_mesh(&feat.positions, &feat.indices).ok()
        } else {
            None
        };

        if let Some(encoded) = meshopt_result {
            // --- Meshopt-compressed primitive ---
            any_meshopt = true;

            // Recompute min/max from optimized positions (vertex reordering changes order)
            let mut opt_min = [f32::INFINITY; 3];
            let mut opt_max = [f32::NEG_INFINITY; 3];
            for v in 0..encoded.vertex_count {
                for d in 0..3 {
                    let val = encoded.positions[v * 3 + d];
                    if val < opt_min[d] { opt_min[d] = val; }
                    if val > opt_max[d] { opt_max[d] = val; }
                }
            }

            // --- Compressed position bufferView ---
            let pos_comp_offset = bin_data.len();
            let pos_comp_size = encoded.vertex_data.len();
            bin_data.extend_from_slice(&encoded.vertex_data);
            let pad = (4 - (bin_data.len() % 4)) % 4;
            bin_data.extend(std::iter::repeat(0u8).take(pad));

            let pos_bv_idx = buffer_views.len();
            buffer_views.push(json!({
                "buffer": 0,
                "byteOffset": pos_comp_offset,
                "byteLength": pos_comp_size,
                "byteStride": 12,
                "target": ARRAY_BUFFER,
                "extensions": {
                    "EXT_meshopt_compression": {
                        "buffer": 0,
                        "byteOffset": pos_comp_offset,
                        "byteLength": pos_comp_size,
                        "byteStride": 12,
                        "count": encoded.vertex_count,
                        "mode": "ATTRIBUTES",
                        "filter": "NONE"
                    }
                }
            }));

            let pos_acc_idx = accessors.len();
            accessors.push(json!({
                "bufferView": pos_bv_idx,
                "byteOffset": 0,
                "componentType": FLOAT,
                "count": encoded.vertex_count,
                "type": "VEC3",
                "min": [opt_min[0], opt_min[1], opt_min[2]],
                "max": [opt_max[0], opt_max[1], opt_max[2]],
            }));

            // --- Compressed index bufferView ---
            let idx_comp_offset = bin_data.len();
            let idx_comp_size = encoded.index_data.len();
            bin_data.extend_from_slice(&encoded.index_data);
            let pad = (4 - (bin_data.len() % 4)) % 4;
            bin_data.extend(std::iter::repeat(0u8).take(pad));

            let idx_bv_idx = buffer_views.len();
            // glTF forbids byteStride on ELEMENT_ARRAY_BUFFER — only inner extension has it
            buffer_views.push(json!({
                "buffer": 0,
                "byteOffset": idx_comp_offset,
                "byteLength": idx_comp_size,
                "target": ELEMENT_ARRAY_BUFFER,
                "extensions": {
                    "EXT_meshopt_compression": {
                        "buffer": 0,
                        "byteOffset": idx_comp_offset,
                        "byteLength": idx_comp_size,
                        "byteStride": 4,
                        "count": encoded.indices.len(),
                        "mode": "TRIANGLES",
                        "filter": "NONE"
                    }
                }
            }));

            let idx_acc_idx = accessors.len();
            accessors.push(json!({
                "bufferView": idx_bv_idx,
                "byteOffset": 0,
                "componentType": UNSIGNED_INT,
                "count": encoded.indices.len(),
                "type": "SCALAR",
            }));

            let primitive = json!({
                "attributes": {"POSITION": pos_acc_idx},
                "indices": idx_acc_idx,
                "material": 0,
                "mode": MODE_TRIANGLES,
            });

            let mesh_idx = meshes.len();
            meshes.push(json!({"primitives": [primitive]}));

            let mut node = json!({"mesh": mesh_idx, "name": format!("feature_{}", nodes.len())});
            if let Some(extras) = &feat.extras {
                node.as_object_mut().unwrap().insert("extras".to_string(), extras.clone());
            }
            let node_idx = nodes.len();
            nodes.push(node);
            scene_nodes.push(json!(node_idx));
        } else {
            // --- Raw encoding (same as encode_glb) ---
            let pos_offset = bin_data.len();
            let pos_bytes: Vec<u8> = feat.positions.iter().flat_map(|f| f.to_le_bytes()).collect();
            let pos_byte_len = pos_bytes.len();
            bin_data.extend_from_slice(&pos_bytes);
            let pad = (4 - (bin_data.len() % 4)) % 4;
            bin_data.extend(std::iter::repeat(0u8).take(pad));

            let pos_bv_idx = buffer_views.len();
            buffer_views.push(json!({
                "buffer": 0,
                "byteOffset": pos_offset,
                "byteLength": pos_byte_len,
                "target": ARRAY_BUFFER,
            }));

            let pos_acc_idx = accessors.len();
            accessors.push(json!({
                "bufferView": pos_bv_idx,
                "byteOffset": 0,
                "componentType": FLOAT,
                "count": n_verts,
                "type": "VEC3",
                "min": [pos_min[0], pos_min[1], pos_min[2]],
                "max": [pos_max[0], pos_max[1], pos_max[2]],
            }));

            let primitive = if !feat.indices.is_empty() {
                let idx_offset = bin_data.len();
                let idx_bytes: Vec<u8> = feat.indices.iter().flat_map(|i| i.to_le_bytes()).collect();
                let idx_byte_len = idx_bytes.len();
                bin_data.extend_from_slice(&idx_bytes);
                let pad = (4 - (bin_data.len() % 4)) % 4;
                bin_data.extend(std::iter::repeat(0u8).take(pad));

                let idx_bv_idx = buffer_views.len();
                buffer_views.push(json!({
                    "buffer": 0,
                    "byteOffset": idx_offset,
                    "byteLength": idx_byte_len,
                    "target": ELEMENT_ARRAY_BUFFER,
                }));

                let idx_acc_idx = accessors.len();
                accessors.push(json!({
                    "bufferView": idx_bv_idx,
                    "byteOffset": 0,
                    "componentType": UNSIGNED_INT,
                    "count": feat.indices.len(),
                    "type": "SCALAR",
                }));

                json!({
                    "attributes": {"POSITION": pos_acc_idx},
                    "indices": idx_acc_idx,
                    "material": 0,
                    "mode": feat.mode,
                })
            } else {
                json!({
                    "attributes": {"POSITION": pos_acc_idx},
                    "material": 0,
                    "mode": feat.mode,
                })
            };

            let mesh_idx = meshes.len();
            meshes.push(json!({"primitives": [primitive]}));

            let mut node = json!({"mesh": mesh_idx, "name": format!("feature_{}", nodes.len())});
            if let Some(extras) = &feat.extras {
                node.as_object_mut().unwrap().insert("extras".to_string(), extras.clone());
            }
            let node_idx = nodes.len();
            nodes.push(node);
            scene_nodes.push(json!(node_idx));
        }
    }

    if meshes.is_empty() {
        return encode_empty_glb();
    }

    let material = json!({
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.8, 0.8, 0.8, 1.0],
            "metallicFactor": 0.1,
            "roughnessFactor": 0.8,
        },
        "doubleSided": true,
    });

    let mut gltf_json = json!({
        "asset": {"version": "2.0", "generator": "mudm-rs"},
        "scene": 0,
        "scenes": [{"nodes": scene_nodes}],
        "nodes": nodes,
        "meshes": meshes,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": bin_data.len()}],
        "materials": [material],
    });

    if any_meshopt {
        gltf_json.as_object_mut().unwrap().insert(
            "extensionsUsed".to_string(),
            json!(["EXT_meshopt_compression"]),
        );
        gltf_json.as_object_mut().unwrap().insert(
            "extensionsRequired".to_string(),
            json!(["EXT_meshopt_compression"]),
        );
    }

    assemble_glb(&gltf_json, &bin_data)
}

fn encode_empty_glb() -> Vec<u8> {
    let gltf_json = json!({
        "asset": {"version": "2.0", "generator": "mudm-rs"},
        "scene": 0,
        "scenes": [{"nodes": []}],
    });
    assemble_glb(&gltf_json, &[])
}

/// Pack a JSON descriptor and binary buffer into a GLB byte stream.
fn assemble_glb(gltf_json: &Value, bin_data: &[u8]) -> Vec<u8> {
    let json_str = serde_json::to_string(gltf_json).unwrap();
    let json_bytes = json_str.as_bytes();
    let json_pad = (4 - (json_bytes.len() % 4)) % 4;
    let json_chunk_len = json_bytes.len() + json_pad;

    let has_bin = !bin_data.is_empty();
    let bin_pad = if has_bin {
        (4 - (bin_data.len() % 4)) % 4
    } else {
        0
    };
    let bin_chunk_len = bin_data.len() + bin_pad;

    let total_length = 12  // GLB header
        + 8 + json_chunk_len  // JSON chunk header + data
        + if has_bin { 8 + bin_chunk_len } else { 0 }; // BIN chunk

    let mut out = Vec::with_capacity(total_length);

    // GLB header (12 bytes)
    out.extend_from_slice(&GLB_MAGIC.to_le_bytes());
    out.extend_from_slice(&GLB_VERSION.to_le_bytes());
    out.extend_from_slice(&(total_length as u32).to_le_bytes());

    // JSON chunk
    out.extend_from_slice(&(json_chunk_len as u32).to_le_bytes());
    out.extend_from_slice(&JSON_CHUNK_TYPE.to_le_bytes());
    out.extend_from_slice(json_bytes);
    out.extend(std::iter::repeat(b' ').take(json_pad));

    // BIN chunk
    if has_bin {
        out.extend_from_slice(&(bin_chunk_len as u32).to_le_bytes());
        out.extend_from_slice(&BIN_CHUNK_TYPE.to_le_bytes());
        out.extend_from_slice(bin_data);
        out.extend(std::iter::repeat(0u8).take(bin_pad));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_empty() {
        let glb = encode_glb(&[]);
        // GLB header: magic + version + length
        assert!(glb.len() >= 12);
        assert_eq!(&glb[0..4], &GLB_MAGIC.to_le_bytes());
        assert_eq!(
            u32::from_le_bytes([glb[4], glb[5], glb[6], glb[7]]),
            GLB_VERSION
        );
        assert_eq!(
            u32::from_le_bytes([glb[8], glb[9], glb[10], glb[11]]),
            glb.len() as u32
        );
    }

    #[test]
    fn test_encode_triangle() {
        let feat = GlbFeature {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0],
            indices: vec![0, 1, 2],
            mode: MODE_TRIANGLES,
            extras: Some(json!({"name": "triangle"})),
        };
        let glb = encode_glb(&[feat]);

        // Verify header
        assert_eq!(&glb[0..4], &GLB_MAGIC.to_le_bytes());

        // Parse JSON chunk
        let json_len =
            u32::from_le_bytes([glb[12], glb[13], glb[14], glb[15]]) as usize;
        let json_type =
            u32::from_le_bytes([glb[16], glb[17], glb[18], glb[19]]);
        assert_eq!(json_type, JSON_CHUNK_TYPE);

        let json_str = std::str::from_utf8(&glb[20..20 + json_len])
            .unwrap()
            .trim();
        let json: Value = serde_json::from_str(json_str).unwrap();

        // Verify structure
        assert_eq!(json["asset"]["version"], "2.0");
        assert_eq!(json["nodes"][0]["extras"]["name"], "triangle");
        assert_eq!(json["meshes"].as_array().unwrap().len(), 1);
        assert_eq!(
            json["meshes"][0]["primitives"][0]["mode"],
            MODE_TRIANGLES
        );
    }

    #[test]
    fn test_encode_multiple_features() {
        let feats = vec![
            GlbFeature {
                positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0],
                indices: vec![0, 1, 2],
                mode: MODE_TRIANGLES,
                extras: Some(json!({"id": 1})),
            },
            GlbFeature {
                positions: vec![2.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                indices: vec![0, 1],
                mode: MODE_LINES,
                extras: Some(json!({"id": 2})),
            },
            GlbFeature {
                positions: vec![5.0, 5.0, 5.0],
                indices: vec![],
                mode: MODE_POINTS,
                extras: None,
            },
        ];
        let glb = encode_glb(&feats);

        // Parse JSON
        let json_len =
            u32::from_le_bytes([glb[12], glb[13], glb[14], glb[15]]) as usize;
        let json_str = std::str::from_utf8(&glb[20..20 + json_len])
            .unwrap()
            .trim();
        let json: Value = serde_json::from_str(json_str).unwrap();

        assert_eq!(json["nodes"].as_array().unwrap().len(), 3);
        assert_eq!(json["meshes"].as_array().unwrap().len(), 3);
        assert_eq!(json["scenes"][0]["nodes"].as_array().unwrap().len(), 3);
    }

    /// Helper to parse GLB JSON chunk.
    fn parse_glb_json(glb: &[u8]) -> Value {
        let json_len = u32::from_le_bytes([glb[12], glb[13], glb[14], glb[15]]) as usize;
        let json_str = std::str::from_utf8(&glb[20..20 + json_len]).unwrap().trim();
        serde_json::from_str(json_str).unwrap()
    }

    /// Build a large enough triangle mesh for Draco encoding (>= 50 verts).
    fn make_large_triangle_mesh(n_tris: usize) -> GlbFeature {
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        for i in 0..n_tris {
            let x = i as f32;
            positions.extend_from_slice(&[x, 0.0, 0.0, x + 1.0, 0.0, 0.0, x + 0.5, 1.0, 0.0]);
            let base = (i * 3) as u32;
            indices.extend_from_slice(&[base, base + 1, base + 2]);
        }
        GlbFeature {
            positions,
            indices,
            mode: MODE_TRIANGLES,
            extras: Some(json!({"name": "large_mesh"})),
        }
    }

    #[test]
    fn test_draco_glb_has_extension() {
        let feat = make_large_triangle_mesh(20); // 60 verts, well above min_vertices=50
        let glb = encode_glb_draco(&[feat], 50);

        assert_eq!(&glb[0..4], &GLB_MAGIC.to_le_bytes());
        let json = parse_glb_json(&glb);

        // Must have KHR_draco_mesh_compression in extensionsUsed
        let ext_used = json["extensionsUsed"].as_array().unwrap();
        assert!(ext_used.iter().any(|v| v == "KHR_draco_mesh_compression"));
        let ext_req = json["extensionsRequired"].as_array().unwrap();
        assert!(ext_req.iter().any(|v| v == "KHR_draco_mesh_compression"));

        // Primitive must have the extension
        let prim = &json["meshes"][0]["primitives"][0];
        assert!(prim["extensions"]["KHR_draco_mesh_compression"].is_object());
    }

    #[test]
    fn test_draco_glb_smaller_than_raw() {
        let feat = make_large_triangle_mesh(100); // 300 verts

        let raw_glb = encode_glb(&[feat.clone()]);
        let draco_glb = encode_glb_draco(&[feat], 50);

        assert!(
            draco_glb.len() < raw_glb.len(),
            "Draco GLB ({}) should be smaller than raw GLB ({})",
            draco_glb.len(), raw_glb.len()
        );
    }

    #[test]
    fn test_draco_small_mesh_stays_raw() {
        // 3 verts < min_vertices=50, should NOT have Draco extension
        let feat = GlbFeature {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0],
            indices: vec![0, 1, 2],
            mode: MODE_TRIANGLES,
            extras: None,
        };
        let glb = encode_glb_draco(&[feat], 50);

        let json = parse_glb_json(&glb);
        // No Draco extension at all
        assert!(json["extensionsUsed"].is_null());
        // Position accessor should have a bufferView (raw)
        assert!(json["accessors"][0]["bufferView"].is_number());
    }

    #[test]
    fn test_draco_mixed_modes() {
        // Large triangles (Draco) + lines (raw) + points (raw)
        let tri = make_large_triangle_mesh(20);
        let line = GlbFeature {
            positions: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            indices: vec![0, 1],
            mode: MODE_LINES,
            extras: Some(json!({"id": "line"})),
        };
        let pt = GlbFeature {
            positions: vec![5.0, 5.0, 5.0],
            indices: vec![],
            mode: MODE_POINTS,
            extras: None,
        };
        let glb = encode_glb_draco(&[tri, line, pt], 50);

        let json = parse_glb_json(&glb);
        assert_eq!(json["nodes"].as_array().unwrap().len(), 3);
        // First primitive (triangles) has Draco extension
        assert!(json["meshes"][0]["primitives"][0]["extensions"]["KHR_draco_mesh_compression"].is_object());
        // Second primitive (lines) has no Draco extension
        assert!(json["meshes"][1]["primitives"][0]["extensions"].is_null());
        // Third primitive (points) has no Draco extension
        assert!(json["meshes"][2]["primitives"][0]["extensions"].is_null());
    }

    // ----- Meshopt tests -----

    #[test]
    fn test_meshopt_glb_has_extension() {
        let feat = make_large_triangle_mesh(20); // 60 verts
        let glb = encode_glb_meshopt(&[feat], 50);

        assert_eq!(&glb[0..4], &GLB_MAGIC.to_le_bytes());
        let json = parse_glb_json(&glb);

        let ext_used = json["extensionsUsed"].as_array().unwrap();
        assert!(ext_used.iter().any(|v| v == "EXT_meshopt_compression"));
        let ext_req = json["extensionsRequired"].as_array().unwrap();
        assert!(ext_req.iter().any(|v| v == "EXT_meshopt_compression"));

        // BufferViews should have the extension
        let bvs = json["bufferViews"].as_array().unwrap();
        let has_ext = bvs.iter().any(|bv| {
            bv["extensions"]["EXT_meshopt_compression"].is_object()
        });
        assert!(has_ext, "At least one bufferView should have EXT_meshopt_compression");

        // Accessors should have bufferView (unlike Draco where they don't)
        assert!(json["accessors"][0]["bufferView"].is_number());
    }

    #[test]
    fn test_meshopt_glb_smaller_than_raw() {
        let feat = make_large_triangle_mesh(100); // 300 verts

        let raw_glb = encode_glb(&[feat.clone()]);
        let meshopt_glb = encode_glb_meshopt(&[feat], 50);

        assert!(
            meshopt_glb.len() < raw_glb.len(),
            "Meshopt GLB ({}) should be smaller than raw GLB ({})",
            meshopt_glb.len(), raw_glb.len()
        );
    }

    #[test]
    fn test_meshopt_small_mesh_stays_raw() {
        let feat = GlbFeature {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0],
            indices: vec![0, 1, 2],
            mode: MODE_TRIANGLES,
            extras: None,
        };
        let glb = encode_glb_meshopt(&[feat], 50);

        let json = parse_glb_json(&glb);
        // No meshopt extension
        assert!(json["extensionsUsed"].is_null());
        // Position accessor has a bufferView (raw)
        assert!(json["accessors"][0]["bufferView"].is_number());
        // No EXT_meshopt_compression on any bufferView
        let bvs = json["bufferViews"].as_array().unwrap();
        for bv in bvs {
            assert!(bv["extensions"].is_null());
        }
    }

    #[test]
    fn test_meshopt_mixed_modes() {
        let tri = make_large_triangle_mesh(20);
        let line = GlbFeature {
            positions: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            indices: vec![0, 1],
            mode: MODE_LINES,
            extras: Some(json!({"id": "line"})),
        };
        let pt = GlbFeature {
            positions: vec![5.0, 5.0, 5.0],
            indices: vec![],
            mode: MODE_POINTS,
            extras: None,
        };
        let glb = encode_glb_meshopt(&[tri, line, pt], 50);

        let json = parse_glb_json(&glb);
        assert_eq!(json["nodes"].as_array().unwrap().len(), 3);
        // Extension should be present at root level
        assert!(json["extensionsUsed"].as_array().unwrap().iter().any(|v| v == "EXT_meshopt_compression"));
    }
}
