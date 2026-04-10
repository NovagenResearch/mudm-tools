/// OBJ file parser — reads vertices and triangle faces.
///
/// Handles n-gon fan triangulation (faces with >3 vertices are split
/// into triangles using the first vertex as the fan pivot).
use pyo3::prelude::*;
use std::fs;
use std::io::{BufRead, BufReader};

/// Parse an OBJ file, returning (vertices, faces) as flat lists.
///
/// Returns:
///   vertices: list of [x, y, z] floats
///   faces: list of [i0, i1, i2] u32 triangle indices (0-based)
#[pyfunction]
pub fn parse_obj(path: &str) -> PyResult<(Vec<[f64; 3]>, Vec<[u32; 3]>)> {
    let file = fs::File::open(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Cannot open {}: {}", path, e)))?;
    let reader = BufReader::new(file);

    let mut vertices: Vec<[f64; 3]> = Vec::new();
    let mut faces: Vec<[u32; 3]> = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Read error: {}", e))
        })?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if line.starts_with("v ") {
            // Vertex line: v x y z
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                let x: f64 = parts[1].parse().unwrap_or(0.0);
                let y: f64 = parts[2].parse().unwrap_or(0.0);
                let z: f64 = parts[3].parse().unwrap_or(0.0);
                vertices.push([x, y, z]);
            }
        } else if line.starts_with("f ") {
            // Face line: f v1 v2 v3 ... or f v1/vt1/vn1 ...
            let parts: Vec<&str> = line.split_whitespace().skip(1).collect();
            let indices: Vec<u32> = parts
                .iter()
                .filter_map(|p| {
                    // Handle v, v/vt, v/vt/vn, v//vn formats
                    let idx_str = p.split('/').next()?;
                    let idx: i64 = idx_str.parse().ok()?;
                    // OBJ indices are 1-based; negative means relative to end
                    if idx > 0 {
                        Some((idx - 1) as u32)
                    } else if idx < 0 {
                        Some((vertices.len() as i64 + idx) as u32)
                    } else {
                        None
                    }
                })
                .collect();

            // Fan triangulation for n-gons (n >= 3)
            if indices.len() >= 3 {
                let v0 = indices[0];
                for i in 1..indices.len() - 1 {
                    faces.push([v0, indices[i], indices[i + 1]]);
                }
            }
        }
    }

    Ok((vertices, faces))
}
