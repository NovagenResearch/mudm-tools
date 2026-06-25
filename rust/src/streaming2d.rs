/// Streaming 2D tile generator — processes GeoJSON features through a quadtree.
///
/// Architecture:
///   For each feature: project → clip through quadtree → write Fragment2D to disk
///   For Parquet output: read fragments → encode to world-coord f32 LE → return to Python
///
/// # API differences from StreamingTileGenerator (streaming.rs)
///
/// **Constructor**: 2D omits `extent_z`, `base_cells`, `num_buckets` (3D-only params).
/// Both share: `min_zoom`, `max_zoom`, `buffer`, `temp_dir`.
///
/// **generate_mvt vs generate_pbf3**: 2D encodes MVT (Mapbox Vector Tiles) with XY-only
/// geometry. 3D encodes PBF3/GLB/Neuroglancer formats with Z coordinates and 3D geometry
/// types (TIN, PolyhedralSurface). Unifying signatures would require breaking changes.
///
/// **Shared utilities**: `extract_tags` is shared via `crate::streaming::extract_tags`.
///
/// **Precision**: 2D uses `extent` for quantization. 3D adds `extent_z`.
/// A future `precision` parameter could unify these.
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use rayon::prelude::*;
use ahash::AHashMap;

use crate::types2d::BBox2D;
use crate::clip2d::{self, ClipFeature2D};
use crate::fragment2d::{Fragment2D, Fragment2DWriter, Fragment2DReader};
use crate::simplify2d;
use crate::streaming::{TagValue, extract_tags};
use crate::encoder_mvt;

// Geometry type constants
const POINT: u8 = 1;
const LINESTRING: u8 = 2;
const POLYGON: u8 = 3;

/// Unique generator ID counter for temp file naming.
static GENERATOR_2D_ID: AtomicU32 = AtomicU32::new(0);

// ---------------------------------------------------------------------------
// Parquet row for 2D data
// ---------------------------------------------------------------------------

struct ParquetRow2D {
    zoom: u8,
    tile_x: u16,
    tile_y: u16,
    feature_id: u32,
    geom_type: u8,
    positions: Vec<u8>,    // raw LE float32 bytes [x,y,x,y,...]
    indices: Vec<u8>,      // raw LE uint32 bytes (line segment indices)
    ring_lengths: Vec<u32>, // ring vertex counts for polygons
}

// ---------------------------------------------------------------------------
// Parquet row collection (pure Rust, no GIL)
// ---------------------------------------------------------------------------

fn collect_parquet_rows_2d(
    tiles: &[((u32, u32, u32), Vec<Fragment2D>)],
    world_bounds: &(f64, f64, f64, f64),
    max_zoom: u32,
    simplify: bool,
) -> Vec<ParquetRow2D> {
    let (xmin, ymin, xmax, ymax) = *world_bounds;
    let dx = if xmax != xmin { xmax - xmin } else { 1.0 };
    let dy = if ymax != ymin { ymax - ymin } else { 1.0 };

    tiles.par_iter()
        .flat_map(|((tz, tx, ty), frags)| {
            let tz = *tz;
            let do_simplify = simplify && tz < max_zoom;

            let mut rows: Vec<ParquetRow2D> = Vec::new();

            for frag in frags {
                let n_verts = frag.xy.len() / 2;
                if n_verts == 0 {
                    continue;
                }

                // Optionally simplify at coarse zoom levels (simplify operates on f64)
                let (work_xy_f32, work_rl) = if do_simplify && frag.geom_type == POLYGON && !frag.ring_lengths.is_empty() {
                    let eps = simplify2d::compute_tolerance(tz, max_zoom);
                    if eps > 0.0 {
                        let xy_f64: Vec<f64> = frag.xy.iter().map(|&v| v as f64).collect();
                        let (simp, rl) = simplify2d::simplify_polygon_rings(&xy_f64, &frag.ring_lengths, eps);
                        let simp_f32: Vec<f32> = simp.iter().map(|&v| v as f32).collect();
                        (simp_f32, rl)
                    } else {
                        (frag.xy.clone(), frag.ring_lengths.clone())
                    }
                } else if do_simplify && frag.geom_type == LINESTRING {
                    let eps = simplify2d::compute_tolerance(tz, max_zoom);
                    if eps > 0.0 {
                        let xy_f64: Vec<f64> = frag.xy.iter().map(|&v| v as f64).collect();
                        let simplified = simplify2d::douglas_peucker(&xy_f64, eps);
                        let simp_f32: Vec<f32> = simplified.iter().map(|&v| v as f32).collect();
                        (simp_f32, vec![])
                    } else {
                        (frag.xy.clone(), vec![])
                    }
                } else {
                    (frag.xy.clone(), frag.ring_lengths.clone())
                };

                let work_n = work_xy_f32.len() / 2;
                if work_n == 0 {
                    continue;
                }

                // Convert to world coordinates as f32 LE bytes (x,y pairs)
                let mut pos_f32: Vec<f32> = Vec::with_capacity(work_n * 2);
                for i in 0..work_n {
                    pos_f32.push(xmin as f32 + work_xy_f32[i * 2] * dx as f32);
                    pos_f32.push(ymin as f32 + work_xy_f32[i * 2 + 1] * dy as f32);
                }

                // Build indices for LineString (line segment pairs)
                let idx_u32: Vec<u32> = if frag.geom_type == LINESTRING {
                    let mut indices = Vec::with_capacity(work_n.saturating_sub(1) * 2);
                    for i in 0..work_n.saturating_sub(1) {
                        indices.push(i as u32);
                        indices.push((i + 1) as u32);
                    }
                    indices
                } else {
                    Vec::new()
                };

                let pos_bytes: Vec<u8> = pos_f32.iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect();
                let idx_bytes: Vec<u8> = idx_u32.iter()
                    .flat_map(|i| i.to_le_bytes())
                    .collect();

                rows.push(ParquetRow2D {
                    zoom: tz as u8,
                    tile_x: *tx as u16,
                    tile_y: *ty as u16,
                    feature_id: frag.feature_id,
                    geom_type: frag.geom_type,
                    positions: pos_bytes,
                    indices: idx_bytes,
                    ring_lengths: work_rl,
                });
            }

            rows
        })
        .collect()
}

// ---------------------------------------------------------------------------
// GeoJSON parsing helpers
// ---------------------------------------------------------------------------

/// Parse a single GeoJSON geometry into ClipFeature2D(s).
fn parse_geojson_geometry(
    geom_type: &str,
    coordinates: &serde_json::Value,
    projector_xmin: f64, projector_ymin: f64,
    proj_dx: f64, proj_dy: f64,
) -> Vec<(ClipFeature2D, u8)> {
    let mut results = Vec::new();

    match geom_type {
        "Point" => {
            if let Some(coords) = coordinates.as_array() {
                if coords.len() >= 2 {
                    let x = coords[0].as_f64().unwrap_or(0.0);
                    let y = coords[1].as_f64().unwrap_or(0.0);
                    let nx = (x - projector_xmin) / proj_dx;
                    let ny = (y - projector_ymin) / proj_dy;
                    results.push((ClipFeature2D {
                        xy: vec![nx, ny],
                        ring_lengths: vec![],
                        geom_type: POINT,
                        bbox: BBox2D { min_x: nx, min_y: ny, max_x: nx, max_y: ny },
                    }, POINT));
                }
            }
        }
        "MultiPoint" => {
            if let Some(points) = coordinates.as_array() {
                for pt in points {
                    if let Some(coords) = pt.as_array() {
                        if coords.len() >= 2 {
                            let x = coords[0].as_f64().unwrap_or(0.0);
                            let y = coords[1].as_f64().unwrap_or(0.0);
                            let nx = (x - projector_xmin) / proj_dx;
                            let ny = (y - projector_ymin) / proj_dy;
                            results.push((ClipFeature2D {
                                xy: vec![nx, ny],
                                ring_lengths: vec![],
                                geom_type: POINT,
                                bbox: BBox2D { min_x: nx, min_y: ny, max_x: nx, max_y: ny },
                            }, POINT));
                        }
                    }
                }
            }
        }
        "LineString" => {
            if let Some(coords) = coordinates.as_array() {
                let mut xy = Vec::with_capacity(coords.len() * 2);
                let mut bb = BBox2D::empty();
                for pt in coords {
                    if let Some(c) = pt.as_array() {
                        let x = c[0].as_f64().unwrap_or(0.0);
                        let y = c[1].as_f64().unwrap_or(0.0);
                        let nx = (x - projector_xmin) / proj_dx;
                        let ny = (y - projector_ymin) / proj_dy;
                        xy.push(nx);
                        xy.push(ny);
                        bb.expand(nx, ny);
                    }
                }
                if xy.len() >= 4 {
                    results.push((ClipFeature2D {
                        xy, ring_lengths: vec![], geom_type: LINESTRING, bbox: bb,
                    }, LINESTRING));
                }
            }
        }
        "MultiLineString" => {
            if let Some(lines) = coordinates.as_array() {
                for line in lines {
                    let sub = parse_geojson_geometry("LineString", line, projector_xmin, projector_ymin, proj_dx, proj_dy);
                    results.extend(sub);
                }
            }
        }
        "Polygon" => {
            if let Some(rings) = coordinates.as_array() {
                let mut xy = Vec::new();
                let mut ring_lengths = Vec::new();
                let mut bb = BBox2D::empty();
                for ring in rings {
                    if let Some(coords) = ring.as_array() {
                        let start = xy.len() / 2;
                        for pt in coords {
                            if let Some(c) = pt.as_array() {
                                let x = c[0].as_f64().unwrap_or(0.0);
                                let y = c[1].as_f64().unwrap_or(0.0);
                                let nx = (x - projector_xmin) / proj_dx;
                                let ny = (y - projector_ymin) / proj_dy;
                                xy.push(nx);
                                xy.push(ny);
                                bb.expand(nx, ny);
                            }
                        }
                        let count = xy.len() / 2 - start;
                        if count >= 3 {
                            ring_lengths.push(count as u32);
                        }
                    }
                }
                if !ring_lengths.is_empty() {
                    results.push((ClipFeature2D {
                        xy, ring_lengths, geom_type: POLYGON, bbox: bb,
                    }, POLYGON));
                }
            }
        }
        "MultiPolygon" => {
            if let Some(polygons) = coordinates.as_array() {
                // Flatten all polygons into a single feature with all rings
                let mut xy = Vec::new();
                let mut ring_lengths = Vec::new();
                let mut bb = BBox2D::empty();
                for polygon in polygons {
                    if let Some(rings) = polygon.as_array() {
                        for ring in rings {
                            if let Some(coords) = ring.as_array() {
                                let start = xy.len() / 2;
                                for pt in coords {
                                    if let Some(c) = pt.as_array() {
                                        let x = c[0].as_f64().unwrap_or(0.0);
                                        let y = c[1].as_f64().unwrap_or(0.0);
                                        let nx = (x - projector_xmin) / proj_dx;
                                        let ny = (y - projector_ymin) / proj_dy;
                                        xy.push(nx);
                                        xy.push(ny);
                                        bb.expand(nx, ny);
                                    }
                                }
                                let count = xy.len() / 2 - start;
                                if count >= 3 {
                                    ring_lengths.push(count as u32);
                                }
                            }
                        }
                    }
                }
                if !ring_lengths.is_empty() {
                    results.push((ClipFeature2D {
                        xy, ring_lengths, geom_type: POLYGON, bbox: bb,
                    }, POLYGON));
                }
            }
        }
        "GeometryCollection" => {
            if let Some(geometries) = coordinates.as_array() {
                for geom in geometries {
                    if let (Some(gt), Some(c)) = (geom.get("type").and_then(|v| v.as_str()), geom.get("coordinates")) {
                        let sub = parse_geojson_geometry(gt, c, projector_xmin, projector_ymin, proj_dx, proj_dy);
                        results.extend(sub);
                    }
                }
            }
        }
        _ => {}
    }

    results
}

// ---------------------------------------------------------------------------
// StreamingTileGenerator2D — PyO3 class
// ---------------------------------------------------------------------------

#[pyclass]
pub struct StreamingTileGenerator2D {
    min_zoom: u32,
    max_zoom: u32,
    buffer: f64,
    feature_count: u32,
    tags_registry: HashMap<u32, Vec<(String, TagValue)>>,
    /// LOD gate: per-feature tag keys present only on tiles at zoom >= the
    /// recorded value (keys absent here appear at all zooms). Lets heavy
    /// per-feature tags (e.g. per-cell expression) ride the fine tiles without
    /// bloating coarse ones, where features are sub-pixel anyway.
    tag_min_zoom: HashMap<String, u32>,
    fragment_writer: Option<Fragment2DWriter>,
    frag_dir: PathBuf,
    shard_counter: std::sync::atomic::AtomicUsize,
    fragment_reader: Option<Fragment2DReader>,
    parquet_stream_active: bool,
}

#[pymethods]
impl StreamingTileGenerator2D {
    /// Create a new streaming 2D tile generator.
    ///
    /// Args:
    ///   min_zoom: Minimum zoom level (default 0).
    ///   max_zoom: Maximum zoom level (default 4).
    ///   buffer: Tile buffer in normalized space (fraction of tile size).
    #[new]
    #[pyo3(signature = (min_zoom=0, max_zoom=4, buffer=0.0, temp_dir=None))]
    fn new(
        min_zoom: u32,
        max_zoom: u32,
        buffer: f64,
        temp_dir: Option<&str>,
    ) -> PyResult<Self> {
        let gen_id = GENERATOR_2D_ID.fetch_add(1, Ordering::Relaxed);
        let base_tmp = match temp_dir {
            Some(d) => PathBuf::from(d),
            None => std::env::temp_dir(),
        };
        let frag_dir = base_tmp
            .join(format!("microjson_frags2d_{}_{}", std::process::id(), gen_id));
        std::fs::create_dir_all(&frag_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let shard_path = frag_dir.join("shard_000.mf2d");
        let writer = Fragment2DWriter::new(&shard_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        Ok(Self {
            min_zoom,
            max_zoom,
            buffer,
            feature_count: 0,
            tags_registry: HashMap::new(),
            tag_min_zoom: HashMap::new(),
            fragment_writer: Some(writer),
            frag_dir,
            shard_counter: std::sync::atomic::AtomicUsize::new(1),
            fragment_reader: None,
            parquet_stream_active: false,
        })
    }

    /// LOD gate (Path B): emit the tag `key` only on tiles at zoom >= `min_zoom`.
    /// Used to keep heavy per-feature tags (e.g. per-cell `expression`) off the
    /// coarse tiles, where features are sub-pixel. Tags without a gate appear at
    /// all zooms.
    fn set_tag_min_zoom(&mut self, key: String, min_zoom: u32) {
        self.tag_min_zoom.insert(key, min_zoom);
    }

    /// Attach extra tags to already-ingested features by matching an existing
    /// tag value (Path B): for every feature whose `id_key` tag equals a key in
    /// `attrs`, append that entry's (tag_key, value) pairs. Lets a converter
    /// join per-entity attributes (e.g. per-cell expression keyed by `cell_id`)
    /// onto the muDM tag registry AFTER `add_parquet_polygons`, so the tags flow
    /// natively into BOTH the PBF and the Parquet output (no sidecar). Returns
    /// the number of features that received at least one extra tag.
    fn attach_tags_by_id(
        &mut self,
        id_key: &str,
        attrs: HashMap<String, Vec<(String, String)>>,
    ) -> u32 {
        let mut n: u32 = 0;
        for tags in self.tags_registry.values_mut() {
            // Read the id value first; the immutable borrow ends before we push.
            let id_val = tags.iter().find_map(|(k, v)| {
                if k == id_key {
                    if let TagValue::Str(s) = v { Some(s.clone()) } else { None }
                } else {
                    None
                }
            });
            if let Some(id) = id_val {
                if let Some(extra) = attrs.get(&id) {
                    if !extra.is_empty() {
                        for (k, v) in extra {
                            tags.push((k.clone(), TagValue::Str(v.clone())));
                        }
                        n += 1;
                    }
                }
            }
        }
        n
    }

    /// Add a single 2D feature (already projected to [0,1]²).
    ///
    /// The feature dict must have keys: xy (flat [x,y,...]), geom_type (1/2/3),
    /// min_x/min_y/max_x/max_y, and optionally tags, ring_lengths.
    ///
    /// Returns the assigned feature ID.
    fn add_feature(&mut self, feat: &Bound<'_, PyDict>) -> PyResult<u32> {
        let fid = self.feature_count;
        self.feature_count += 1;

        let tags = extract_tags(feat)?;
        self.tags_registry.insert(fid, tags);

        let xy: Vec<f64> = feat.get_item("xy")?.unwrap().extract()?;
        let geom_type: u8 = feat.get_item("geom_type")?.unwrap().extract()?;
        let ring_lengths: Vec<u32> = if let Some(rl) = feat.get_item("ring_lengths")? {
            if rl.is_none() { vec![] } else { rl.extract()? }
        } else {
            vec![]
        };

        let bbox = BBox2D {
            min_x: feat.get_item("min_x")?.unwrap().extract()?,
            min_y: feat.get_item("min_y")?.unwrap().extract()?,
            max_x: feat.get_item("max_x")?.unwrap().extract()?,
            max_y: feat.get_item("max_y")?.unwrap().extract()?,
        };

        let clip_feat = ClipFeature2D { xy, ring_lengths, geom_type, bbox };

        let fragments = clip2d::quadtree_clip(&clip_feat, self.min_zoom, self.max_zoom, self.buffer);

        let writer = self.fragment_writer.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot add features after generate"))?;

        for ((tz, tx, ty), cf) in fragments {
            let frag = Fragment2D {
                feature_id: fid,
                tile_z: tz,
                tile_x: tx,
                tile_y: ty,
                geom_type: cf.geom_type,
                xy: cf.xy.iter().map(|&v| v as f32).collect(),
                ring_lengths: cf.ring_lengths,
            };
            writer.write(&frag)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        Ok(fid)
    }

    /// Add a GeoJSON string — parse, project, clip, and write fragments.
    ///
    /// Args:
    ///   json_str: GeoJSON string (Feature, FeatureCollection, or Geometry).
    ///   bounds: World bounding box (xmin, ymin, xmax, ymax).
    ///
    /// Returns list of assigned feature IDs.
    fn add_geojson(
        &mut self,
        json_str: &str,
        bounds: (f64, f64, f64, f64),
    ) -> PyResult<Vec<u32>> {
        let (xmin, ymin, xmax, ymax) = bounds;
        let proj_dx = if xmax != xmin { xmax - xmin } else { 1.0 };
        let proj_dy = if ymax != ymin { ymax - ymin } else { 1.0 };

        let parsed: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

        let features = extract_geojson_features(&parsed);

        let writer = self.fragment_writer.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot add features after generate"))?;

        let mut fids = Vec::new();

        for (geom_type, coordinates, properties) in features {
            let clip_feats = parse_geojson_geometry(
                &geom_type, &coordinates,
                xmin, ymin, proj_dx, proj_dy,
            );

            if clip_feats.is_empty() {
                continue;
            }

            let fid = self.feature_count;
            self.feature_count += 1;

            // Extract tags from properties
            let mut tag_vec: Vec<(String, TagValue)> = Vec::new();
            if let Some(obj) = properties.as_object() {
                for (k, v) in obj {
                    match v {
                        serde_json::Value::String(s) => tag_vec.push((k.clone(), TagValue::Str(s.clone()))),
                        serde_json::Value::Number(n) => {
                            if let Some(i) = n.as_i64() {
                                tag_vec.push((k.clone(), TagValue::Int(i)));
                            } else if let Some(f) = n.as_f64() {
                                tag_vec.push((k.clone(), TagValue::Float(f)));
                            }
                        }
                        serde_json::Value::Bool(b) => tag_vec.push((k.clone(), TagValue::Bool(*b))),
                        _ => {}
                    }
                }
            }
            self.tags_registry.insert(fid, tag_vec);

            for (cf, gt) in clip_feats {
                let fragments = clip2d::quadtree_clip(&cf, self.min_zoom, self.max_zoom, self.buffer);
                for ((tz, tx, ty), clipped) in fragments {
                    // Decimation for point features: at lower zooms, skip most points
                    if gt == POINT && tz < self.max_zoom {
                        let skip_factor = 1u32 << (self.max_zoom - tz);
                        if fid % skip_factor != 0 {
                            continue;
                        }
                    }
                    let frag = Fragment2D {
                        feature_id: fid,
                        tile_z: tz,
                        tile_x: tx,
                        tile_y: ty,
                        geom_type: clipped.geom_type,
                        xy: clipped.xy.iter().map(|&v| v as f32).collect(),
                        ring_lengths: clipped.ring_lengths,
                    };
                    writer.write(&frag)
                        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                }
            }

            fids.push(fid);
        }

        Ok(fids)
    }

    /// Add point features directly from a Parquet file — no JSON intermediary.
    ///
    /// Reads x/y coordinate columns and an optional property column directly
    /// from Parquet using arrow-rs. Each row becomes a Point feature.
    /// Coordinates are scaled by coord_scale (e.g., 1/um_per_px for µm→px).
    ///
    /// Args:
    ///   path: Path to the Parquet file.
    ///   x_col: Column name for x coordinates.
    ///   y_col: Column name for y coordinates.
    ///   prop_col: Optional column name for a string property (e.g., "feature_name").
    ///   prop_name: Property name in the output tags (e.g., "gene_name").
    ///   layer_type: Value for the "layer_type" tag.
    ///   bounds: World bounding box (xmin, ymin, xmax, ymax) AFTER scaling.
    ///   coord_scale: Multiply raw coordinates by this (default 1.0).
    ///
    /// Returns number of features added.
    #[pyo3(signature = (path, x_col, y_col, prop_col, prop_name, layer_type, bounds, coord_scale=1.0))]
    fn add_parquet_points(
        &mut self,
        py: Python<'_>,
        path: &str,
        x_col: &str,
        y_col: &str,
        prop_col: &str,
        prop_name: &str,
        layer_type: &str,
        bounds: (f64, f64, f64, f64),
        coord_scale: f64,
    ) -> PyResult<u32> {
        use arrow::array::{Float32Array, Float64Array, Array};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use std::fs::File;
        use std::sync::Mutex;

        let (xmin, ymin, xmax, ymax) = bounds;
        let proj_dx = if xmax != xmin { xmax - xmin } else { 1.0 };
        let proj_dy = if ymax != ymin { ymax - ymin } else { 1.0 };

        // Close the single-feature writer; we'll use per-thread shard writers
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        let min_zoom = self.min_zoom;
        let max_zoom = self.max_zoom;
        let buffer = self.buffer;
        let lt_str = layer_type.to_string();
        let prop_name_str = prop_name.to_string();
        let x_col = x_col.to_string();
        let y_col = y_col.to_string();
        let prop_col = prop_col.to_string();
        let path = path.to_string();

        // Per-thread fragment writers (same pattern as add_geojson_files)
        let n_writers = rayon::current_num_threads() + 1;
        let writers: Vec<Mutex<Fragment2DWriter>> = (0..n_writers)
            .map(|_| {
                let shard_id = self.shard_counter.fetch_add(1, Ordering::Relaxed);
                let shard_path = self.frag_dir.join(format!("shard_{:03}.mf2d", shard_id));
                Mutex::new(
                    Fragment2DWriter::new(&shard_path)
                        .expect("Failed to create fragment2d shard file")
                )
            })
            .collect();
        let writers_ref = &writers;

        // H2 (streaming_review.md): feature ids must be DETERMINISTIC. The old
        // `AtomicU32::fetch_add` inside the par_iter assigned ids in rayon
        // scheduling order — different every run — and coarse-zoom decimation
        // keys off `fid % skip_factor`, so the SET OF POINTS KEPT varied
        // run-to-run. Ids are now derived from the (stable) row position:
        // batch base + index within the batch's surviving rows.
        let base_fid = self.feature_count;

        // Read Parquet and process batches in parallel — GIL released
        let (all_tags, final_fid): (Vec<Vec<(u32, Vec<(String, TagValue)>)>>, u32) = py.allow_threads(|| {
            let file = File::open(&path).expect("Cannot open parquet file");
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)
                .expect("Parquet reader error");
            let reader = builder.build().expect("Parquet build error");

            let mut next_fid = base_fid;
            let mut batch_tags = Vec::new();

            for batch_result in reader {
                let batch = match batch_result {
                    Ok(b) => b,
                    Err(_) => continue,
                };

                let x_col_idx = batch.schema().index_of(&x_col).unwrap();
                let y_col_idx = batch.schema().index_of(&y_col).unwrap();
                let x_arr = batch.column(x_col_idx);
                let y_arr = batch.column(y_col_idx);
                let prop_col_idx = batch.schema().index_of(&prop_col).ok();
                let prop_arr = prop_col_idx.map(|i| batch.column(i).clone());

                // Collect row data for parallel processing
                let n = batch.num_rows();
                let mut rows: Vec<(f64, f64, Option<String>)> = Vec::with_capacity(n);

                for i in 0..n {
                    if x_arr.is_null(i) || y_arr.is_null(i) { continue; }

                    let raw_x = if let Some(a) = x_arr.as_any().downcast_ref::<Float32Array>() {
                        a.value(i) as f64
                    } else if let Some(a) = x_arr.as_any().downcast_ref::<Float64Array>() {
                        a.value(i)
                    } else { continue };

                    let raw_y = if let Some(a) = y_arr.as_any().downcast_ref::<Float32Array>() {
                        a.value(i) as f64
                    } else if let Some(a) = y_arr.as_any().downcast_ref::<Float64Array>() {
                        a.value(i)
                    } else { continue };

                    let prop_val = if let Some(ref arr) = prop_arr {
                        if let Some(sa) = arr.as_any().downcast_ref::<arrow::array::StringArray>() {
                            if !sa.is_null(i) { Some(sa.value(i).to_string()) } else { None }
                        } else if let Some(ba) = arr.as_any().downcast_ref::<arrow::array::BinaryArray>() {
                            if !ba.is_null(i) {
                                std::str::from_utf8(ba.value(i)).ok().map(|s| s.to_string())
                            } else { None }
                        } else if let Some(sa) = arr.as_any().downcast_ref::<arrow::array::LargeStringArray>() {
                            if !sa.is_null(i) { Some(sa.value(i).to_string()) } else { None }
                        } else { None }
                    } else { None };

                    rows.push((raw_x, raw_y, prop_val));
                }

                // Parallel clip + write for this batch. fid = batch base + row
                // index (H2): deterministic regardless of rayon scheduling.
                let batch_base = next_fid;
                let this_tags: Vec<(u32, Vec<(String, TagValue)>)> = rows.par_iter()
                    .enumerate()
                    .map(|(ri, (raw_x, raw_y, prop_val))| {
                        let x = raw_x * coord_scale;
                        let y = raw_y * coord_scale;
                        let nx = (x - xmin) / proj_dx;
                        let ny = (y - ymin) / proj_dy;

                        let fid = batch_base + ri as u32;

                        let mut tags: Vec<(String, TagValue)> = Vec::with_capacity(2);
                        tags.push(("layer_type".to_string(), TagValue::Str(lt_str.clone())));
                        if let Some(ref v) = prop_val {
                            tags.push((prop_name_str.clone(), TagValue::Str(v.clone())));
                        }

                        let cf = ClipFeature2D {
                            xy: vec![nx, ny],
                            ring_lengths: vec![],
                            geom_type: 1,
                            bbox: BBox2D { min_x: nx, min_y: ny, max_x: nx, max_y: ny },
                        };

                        let fragments = clip2d::quadtree_clip(&cf, min_zoom, max_zoom, buffer);
                        // L6 (streaming_review.md): total indexing — falls back
                        // to the last writer if this ever runs inside a pool
                        // larger than the one the writers were sized for.
                        let writer_idx = rayon::current_thread_index().unwrap_or(n_writers - 1);
                        let mut w = writers_ref
                            .get(writer_idx)
                            .unwrap_or(&writers_ref[n_writers - 1])
                            .lock()
                            .unwrap();
                        for ((tz, tx, ty), clipped) in fragments {
                            // Decimation: at lower zooms, skip most points
                            if tz < max_zoom {
                                let skip_factor = 1u32 << (max_zoom - tz);
                                if fid % skip_factor != 0 {
                                    continue;
                                }
                            }
                            let frag = Fragment2D {
                                feature_id: fid,
                                tile_z: tz, tile_x: tx, tile_y: ty,
                                geom_type: clipped.geom_type,
                                xy: clipped.xy.iter().map(|&v| v as f32).collect(),
                                ring_lengths: clipped.ring_lengths,
                            };
                            w.write(&frag).expect("Fragment write failed");
                        }

                        (fid, tags)
                    })
                    .collect();

                next_fid += rows.len() as u32;
                batch_tags.push(this_tags);
            }

            (batch_tags, next_fid)
        });

        // Flush shard writers
        for w in &writers {
            w.lock().unwrap().flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        // Collect tags into registry
        let mut total_count: u32 = 0;
        for batch in all_tags {
            for (fid, tags) in batch {
                self.tags_registry.insert(fid, tags);
                total_count += 1;
            }
        }

        self.feature_count = final_fid;
        Ok(total_count)
    }

    /// Add polygon features from a Parquet file with vertex-per-row layout.
    ///
    /// Reads id/x/y columns, groups rows by id_col to build polygon rings,
    /// projects to [0,1]², clips through quadtree, and writes fragments.
    ///
    /// Args:
    ///   path: Path to the Parquet file.
    ///   id_col: Column name for polygon identifier (e.g., "cell_id").
    ///   x_col: Column name for vertex x coordinate.
    ///   y_col: Column name for vertex y coordinate.
    ///   layer_type: Value for the "layer_type" tag.
    ///   bounds: World bounding box (xmin, ymin, xmax, ymax) AFTER scaling.
    ///   coord_scale: Multiply raw coordinates by this (default 1.0).
    ///
    /// Returns number of polygons added.
    #[pyo3(signature = (path, id_col, x_col, y_col, layer_type, bounds, coord_scale=1.0))]
    fn add_parquet_polygons(
        &mut self,
        py: Python<'_>,
        path: &str,
        id_col: &str,
        x_col: &str,
        y_col: &str,
        layer_type: &str,
        bounds: (f64, f64, f64, f64),
        coord_scale: f64,
    ) -> PyResult<u32> {
        use arrow::array::{Float32Array, Float64Array, Int32Array, Int64Array,
                           StringArray, BinaryArray, LargeStringArray, Array};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use std::fs::File;
        use std::sync::Mutex;

        let (xmin, ymin, xmax, ymax) = bounds;
        let proj_dx = if xmax != xmin { xmax - xmin } else { 1.0 };
        let proj_dy = if ymax != ymin { ymax - ymin } else { 1.0 };

        // Close the single-feature writer; we'll use per-thread shard writers
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        let min_zoom = self.min_zoom;
        let max_zoom = self.max_zoom;
        let buffer = self.buffer;
        let lt_str = layer_type.to_string();
        let id_col = id_col.to_string();
        let x_col = x_col.to_string();
        let y_col = y_col.to_string();
        let path = path.to_string();

        // Per-thread fragment writers
        let n_writers = rayon::current_num_threads() + 1;
        let writers: Vec<Mutex<Fragment2DWriter>> = (0..n_writers)
            .map(|_| {
                let shard_id = self.shard_counter.fetch_add(1, Ordering::Relaxed);
                let shard_path = self.frag_dir.join(format!("shard_{:03}.mf2d", shard_id));
                Mutex::new(
                    Fragment2DWriter::new(&shard_path)
                        .expect("Failed to create fragment2d shard file")
                )
            })
            .collect();
        let writers_ref = &writers;

        // H2 (streaming_review.md): deterministic feature ids — batch base +
        // polygon index instead of fetch_add in rayon scheduling order (ids
        // flow into the emitted fragments, so output bytes depended on
        // scheduling). Polygon order itself is already deterministic
        // (`id_order` tracks first occurrence in row order).
        let base_fid = self.feature_count;

        // Read Parquet and process batches — GIL released
        let (all_tags, final_fid): (Vec<Vec<(u32, Vec<(String, TagValue)>)>>, u32) = py.allow_threads(|| {
            let file = File::open(&path).expect("Cannot open parquet file");
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)
                .expect("Parquet reader error");
            let reader = builder.build().expect("Parquet build error");

            let mut next_fid = base_fid;
            let mut batch_tags = Vec::new();

            for batch_result in reader {
                let batch = match batch_result {
                    Ok(b) => b,
                    Err(_) => continue,
                };

                let id_col_idx = batch.schema().index_of(&id_col).unwrap();
                let x_col_idx = batch.schema().index_of(&x_col).unwrap();
                let y_col_idx = batch.schema().index_of(&y_col).unwrap();
                let id_arr = batch.column(id_col_idx);
                let x_arr = batch.column(x_col_idx);
                let y_arr = batch.column(y_col_idx);

                let n = batch.num_rows();

                // Group rows by cell_id to build polygons
                // Use an IndexMap-like approach: collect (id_string, x, y) then group
                let mut groups: ahash::AHashMap<String, Vec<(f64, f64)>> = ahash::AHashMap::new();
                let mut id_order: Vec<String> = Vec::new();

                for i in 0..n {
                    if x_arr.is_null(i) || y_arr.is_null(i) || id_arr.is_null(i) {
                        continue;
                    }

                    // Extract id as string from various types
                    let id_str = if let Some(a) = id_arr.as_any().downcast_ref::<Int32Array>() {
                        a.value(i).to_string()
                    } else if let Some(a) = id_arr.as_any().downcast_ref::<Int64Array>() {
                        a.value(i).to_string()
                    } else if let Some(a) = id_arr.as_any().downcast_ref::<StringArray>() {
                        a.value(i).to_string()
                    } else if let Some(a) = id_arr.as_any().downcast_ref::<BinaryArray>() {
                        match std::str::from_utf8(a.value(i)) {
                            Ok(s) => s.to_string(),
                            Err(_) => continue,
                        }
                    } else if let Some(a) = id_arr.as_any().downcast_ref::<LargeStringArray>() {
                        a.value(i).to_string()
                    } else {
                        continue;
                    };

                    let raw_x = if let Some(a) = x_arr.as_any().downcast_ref::<Float32Array>() {
                        a.value(i) as f64
                    } else if let Some(a) = x_arr.as_any().downcast_ref::<Float64Array>() {
                        a.value(i)
                    } else { continue };

                    let raw_y = if let Some(a) = y_arr.as_any().downcast_ref::<Float32Array>() {
                        a.value(i) as f64
                    } else if let Some(a) = y_arr.as_any().downcast_ref::<Float64Array>() {
                        a.value(i)
                    } else { continue };

                    if !groups.contains_key(&id_str) {
                        id_order.push(id_str.clone());
                    }
                    groups.entry(id_str).or_default().push((raw_x, raw_y));
                }

                // Build polygon features and clip in parallel
                let polygons: Vec<(String, Vec<(f64, f64)>)> = id_order
                    .into_iter()
                    .filter_map(|id| {
                        let verts = groups.remove(&id)?;
                        if verts.len() >= 3 {
                            Some((id, verts))
                        } else {
                            None
                        }
                    })
                    .collect();

                let batch_base = next_fid;
                let this_tags: Vec<(u32, Vec<(String, TagValue)>)> = polygons.par_iter()
                    .enumerate()
                    .map(|(pi, (cell_id, verts))| {
                        // Build normalized polygon ring
                        let mut xy = Vec::with_capacity(verts.len() * 2);
                        let mut bb = BBox2D::empty();
                        for &(raw_x, raw_y) in verts {
                            let x = raw_x * coord_scale;
                            let y = raw_y * coord_scale;
                            let nx = (x - xmin) / proj_dx;
                            let ny = (y - ymin) / proj_dy;
                            xy.push(nx);
                            xy.push(ny);
                            bb.expand(nx, ny);
                        }

                        let ring_lengths = vec![verts.len() as u32];

                        let fid = batch_base + pi as u32; // H2: deterministic

                        let mut tags: Vec<(String, TagValue)> = Vec::with_capacity(2);
                        tags.push(("layer_type".to_string(), TagValue::Str(lt_str.clone())));
                        tags.push(("cell_id".to_string(), TagValue::Str(cell_id.clone())));

                        let cf = ClipFeature2D {
                            xy,
                            ring_lengths,
                            geom_type: POLYGON,
                            bbox: bb,
                        };

                        let fragments = clip2d::quadtree_clip(&cf, min_zoom, max_zoom, buffer);
                        // L6 (streaming_review.md): total indexing — falls back
                        // to the last writer if this ever runs inside a pool
                        // larger than the one the writers were sized for.
                        let writer_idx = rayon::current_thread_index().unwrap_or(n_writers - 1);
                        let mut w = writers_ref
                            .get(writer_idx)
                            .unwrap_or(&writers_ref[n_writers - 1])
                            .lock()
                            .unwrap();
                        for ((tz, tx, ty), clipped) in fragments {
                            let frag = Fragment2D {
                                feature_id: fid,
                                tile_z: tz, tile_x: tx, tile_y: ty,
                                geom_type: clipped.geom_type,
                                xy: clipped.xy.iter().map(|&v| v as f32).collect(),
                                ring_lengths: clipped.ring_lengths,
                            };
                            w.write(&frag).expect("Fragment write failed");
                        }

                        (fid, tags)
                    })
                    .collect();

                next_fid += polygons.len() as u32;
                batch_tags.push(this_tags);
            }

            (batch_tags, next_fid)
        });

        // Flush shard writers
        for w in &writers {
            w.lock().unwrap().flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        // Collect tags into registry
        let mut total_count: u32 = 0;
        for batch in all_tags {
            for (fid, tags) in batch {
                self.tags_registry.insert(fid, tags);
                total_count += 1;
            }
        }

        self.feature_count = final_fid;
        Ok(total_count)
    }

    /// Add multiple GeoJSON files in parallel using rayon.
    ///
    /// Args:
    ///   paths: List of file paths.
    ///   bounds: World bounding box (xmin, ymin, xmax, ymax).
    ///
    /// Returns list of assigned feature IDs.
    fn add_geojson_files(
        &mut self,
        py: Python<'_>,
        paths: Vec<String>,
        bounds: (f64, f64, f64, f64),
    ) -> PyResult<Vec<u32>> {
        let (xmin, ymin, xmax, ymax) = bounds;
        let proj_dx = if xmax != xmin { xmax - xmin } else { 1.0 };
        let proj_dy = if ymax != ymin { ymax - ymin } else { 1.0 };

        // Close the single-feature writer
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        let min_zoom = self.min_zoom;
        let max_zoom = self.max_zoom;
        let buffer = self.buffer;

        // Pre-create per-thread writers
        let n_writers = rayon::current_num_threads() + 1;
        let writers: Vec<std::sync::Mutex<Fragment2DWriter>> = (0..n_writers)
            .map(|_| {
                let shard_id = self.shard_counter.fetch_add(1, Ordering::Relaxed);
                let shard_path = self.frag_dir.join(format!("shard_{:03}.mf2d", shard_id));
                std::sync::Mutex::new(
                    Fragment2DWriter::new(&shard_path)
                        .expect("Failed to create fragment2d shard file")
                )
            })
            .collect();
        let writers_ref = &writers;

        // H2 (streaming_review.md): deterministic feature ids. The old
        // `fetch_add` assigned ids in rayon scheduling order across files —
        // different every run, and the ids flow into the emitted fragments.
        // Ids are now (per-file offset + feature index within the file): a
        // parallel pre-pass counts each file's features (the JSON is parsed
        // twice — acceptable on this path), offsets are prefix-summed
        // sequentially. Every extracted feature consumes an id, including
        // those whose geometry yields no clip output (deterministic gaps).
        let counts: Vec<u32> = py.allow_threads(|| {
            paths.par_iter().map(|path| {
                let json_str = match std::fs::read_to_string(path) {
                    Ok(s) => s,
                    Err(_) => return 0u32,
                };
                let parsed: serde_json::Value = match serde_json::from_str(&json_str) {
                    Ok(v) => v,
                    Err(_) => return 0u32,
                };
                extract_geojson_features(&parsed).len() as u32
            }).collect()
        });
        let mut file_base: Vec<u32> = Vec::with_capacity(paths.len());
        let mut acc = self.feature_count;
        for c in &counts {
            file_base.push(acc);
            acc += *c;
        }
        let file_base_ref = &file_base;
        let final_fid = acc;

        // Collect tags from all files in parallel
        let all_results: Vec<Vec<(u32, Vec<(String, TagValue)>)>> = py.allow_threads(|| {
            paths.par_iter().enumerate().map(|(pi, path)| {
                let json_str = match std::fs::read_to_string(path) {
                    Ok(s) => s,
                    Err(_) => return vec![],
                };
                let parsed: serde_json::Value = match serde_json::from_str(&json_str) {
                    Ok(v) => v,
                    Err(_) => return vec![],
                };

                let features = extract_geojson_features(&parsed);
                let mut file_results = Vec::new();
                let mut local_idx: u32 = 0;

                for (geom_type, coordinates, properties) in features {
                    let fid = file_base_ref[pi] + local_idx; // H2: deterministic
                    local_idx += 1;

                    let clip_feats = parse_geojson_geometry(
                        &geom_type, &coordinates,
                        xmin, ymin, proj_dx, proj_dy,
                    );
                    if clip_feats.is_empty() { continue; }

                    let mut tag_vec: Vec<(String, TagValue)> = Vec::new();
                    if let Some(obj) = properties.as_object() {
                        for (k, v) in obj {
                            match v {
                                serde_json::Value::String(s) => tag_vec.push((k.clone(), TagValue::Str(s.clone()))),
                                serde_json::Value::Number(n) => {
                                    if let Some(i) = n.as_i64() { tag_vec.push((k.clone(), TagValue::Int(i))); }
                                    else if let Some(f) = n.as_f64() { tag_vec.push((k.clone(), TagValue::Float(f))); }
                                }
                                serde_json::Value::Bool(b) => tag_vec.push((k.clone(), TagValue::Bool(*b))),
                                _ => {}
                            }
                        }
                    }

                    // Clip and write
                    for (cf, _gt) in clip_feats {
                        let fragments = clip2d::quadtree_clip(&cf, min_zoom, max_zoom, buffer);
                        let writer_idx = rayon::current_thread_index()
                            .unwrap_or(n_writers - 1);
                        // L6: total indexing (see the parquet sites).
                        let mut w = writers_ref
                            .get(writer_idx)
                            .unwrap_or(&writers_ref[n_writers - 1])
                            .lock()
                            .unwrap();
                        for ((tz, tx, ty), clipped) in fragments {
                            let frag_out = Fragment2D {
                                feature_id: fid,
                                tile_z: tz, tile_x: tx, tile_y: ty,
                                geom_type: clipped.geom_type,
                                xy: clipped.xy.iter().map(|&v| v as f32).collect(),
                                ring_lengths: clipped.ring_lengths,
                            };
                            w.write(&frag_out).ok();
                        }
                    }

                    file_results.push((fid, tag_vec));
                }

                file_results
            }).collect()
        });

        // Flush all shard writers
        for w in &writers {
            w.lock().unwrap().flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        // Collect all fids and tags back into the registry
        let mut fids = Vec::new();
        for file_results in all_results {
            for (fid, tags) in file_results {
                self.tags_registry.insert(fid, tags);
                fids.push(fid);
            }
        }

        self.feature_count = final_fid;
        fids.sort();
        Ok(fids)
    }

    /// Number of features added so far.
    fn feature_count_val(&self) -> u32 {
        self.feature_count
    }

    // ------------------------------------------------------------------
    // Parquet data collection (in-memory)
    // ------------------------------------------------------------------

    /// Collect tile-centric Parquet data from all fragments.
    #[pyo3(signature = (world_bounds, simplify=true))]
    fn _collect_parquet_data(
        &mut self,
        py: Python<'_>,
        world_bounds: (f64, f64, f64, f64),
        simplify: bool,
    ) -> PyResult<PyObject> {
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        let mut reader = Fragment2DReader::open_dir(&self.frag_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let groups = reader.read_all_grouped()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let tiles: Vec<((u32, u32, u32), Vec<Fragment2D>)> = groups.into_iter().collect();
        let wb = world_bounds;
        let max_zoom = self.max_zoom;

        let rows = py.allow_threads(|| {
            collect_parquet_rows_2d(&tiles, &wb, max_zoom, simplify)
        });

        build_python_dict(py, &rows, &self.tags_registry)
    }

    // ------------------------------------------------------------------
    // Streaming Parquet batch API
    // ------------------------------------------------------------------

    fn _init_parquet_stream(&mut self) -> PyResult<()> {
        if self.parquet_stream_active {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Parquet stream already active"));
        }
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }
        let reader = Fragment2DReader::open_dir(&self.frag_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        self.fragment_reader = Some(reader);
        self.parquet_stream_active = true;
        Ok(())
    }

    #[pyo3(signature = (batch_size, world_bounds, max_batch_bytes=2_000_000_000, simplify=true))]
    fn _next_parquet_batch(
        &mut self,
        py: Python<'_>,
        batch_size: usize,
        world_bounds: (f64, f64, f64, f64),
        max_batch_bytes: usize,
        simplify: bool,
    ) -> PyResult<Option<PyObject>> {
        if !self.parquet_stream_active {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Parquet stream not active"));
        }

        let reader = self.fragment_reader.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Fragment reader is None")
        })?;

        let mut groups: AHashMap<(u32, u32, u32), Vec<Fragment2D>> = AHashMap::new();
        let mut count = 0usize;
        let mut batch_bytes = 0usize;
        while count < batch_size && (count == 0 || batch_bytes < max_batch_bytes) {
            match reader.read_next() {
                Ok(Some(frag)) => {
                    batch_bytes += frag.estimate_bytes();
                    let key = (frag.tile_z, frag.tile_x, frag.tile_y);
                    groups.entry(key).or_default().push(frag);
                    count += 1;
                }
                Ok(None) => break,
                Err(e) => return Err(pyo3::exceptions::PyIOError::new_err(e.to_string())),
            }
        }

        if count == 0 {
            return Ok(None);
        }

        let tiles: Vec<((u32, u32, u32), Vec<Fragment2D>)> = groups.into_iter().collect();
        let wb = world_bounds;
        let max_zoom = self.max_zoom;

        let rows = py.allow_threads(|| {
            collect_parquet_rows_2d(&tiles, &wb, max_zoom, simplify)
        });

        let dict = build_python_dict(py, &rows, &self.tags_registry)?;
        Ok(Some(dict))
    }

    fn _close_parquet_stream(&mut self) -> PyResult<()> {
        self.fragment_reader.take();
        self.parquet_stream_active = false;
        Ok(())
    }

    // ------------------------------------------------------------------
    // PBF (Mapbox Vector Tile) output
    // ------------------------------------------------------------------

    /// Generate PBF vector tiles from all fragments.
    ///
    /// Writes tiles to `{output_dir}/{z}/{x}/{y}.pbf` and a `metadata.json`.
    /// Returns the number of tiles written.
    #[pyo3(signature = (output_dir, world_bounds, extent=4096, simplify=true, layer_name="geojsonLayer"))]
    fn generate_pbf(
        &mut self,
        py: Python<'_>,
        output_dir: &str,
        world_bounds: (f64, f64, f64, f64),
        extent: u32,
        simplify: bool,
        layer_name: &str,
    ) -> PyResult<u32> {
        // Flush fragment writer
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        // Read all fragments grouped by tile
        let mut reader = Fragment2DReader::open_dir(&self.frag_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let groups = reader.read_all_grouped()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let tiles: Vec<((u32, u32, u32), Vec<Fragment2D>)> = groups.into_iter().collect();
        let max_zoom = self.max_zoom;
        let min_zoom = self.min_zoom;
        let out_dir = output_dir.to_string();
        let ln = layer_name.to_string();
        let tags_reg = &self.tags_registry;

        // Build the tag vectors for each feature outside of the parallel section
        // (tags_registry is not Send, so we need to snapshot it)
        let tags_snapshot: HashMap<u32, Vec<(String, TagValue)>> = tags_reg.clone();
        let tmz_snapshot: HashMap<String, u32> = self.tag_min_zoom.clone();

        let tile_count = py.allow_threads(|| {
            generate_pbf_tiles(
                &tiles, &out_dir, &world_bounds, max_zoom, extent,
                simplify, &ln, &tags_snapshot, &tmz_snapshot,
            )
        }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

        // Write metadata.json in TileJSON 3.0.0 format
        let meta_path = std::path::Path::new(output_dir).join("metadata.json");
        let center_x = (world_bounds.0 + world_bounds.2) / 2.0;
        let center_y = (world_bounds.1 + world_bounds.3) / 2.0;
        let meta = serde_json::json!({
            "tilejson": "3.0.0",
            "tiles": ["{z}/{x}/{y}.pbf"],
            "name": "MicroJSON Vector Tiles",
            "description": "Vector tiles generated by microjson Rust pipeline",
            "version": "1.0.0",
            "minzoom": min_zoom,
            "maxzoom": max_zoom,
            "bounds": [world_bounds.0, world_bounds.1, world_bounds.2, world_bounds.3],
            "center": [0.0, center_x, center_y],
            "vector_layers": [{
                "id": ln,
                "fields": {},
                "minzoom": min_zoom,
                "maxzoom": max_zoom,
            }],
            "tile_count": tile_count,
        });
        std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        Ok(tile_count)
    }

    /// Generate partitioned Parquet entirely in Rust — no Python round-trip.
    ///
    /// Reads fragments, transforms to world coords (parallel via Rayon),
    /// writes one Parquet file per zoom level using arrow-rs + parquet-rs.
    ///
    /// Returns total number of rows written.
    #[pyo3(signature = (output_dir, world_bounds, simplify=true, compression="zstd"))]
    fn generate_parquet_native(
        &mut self,
        py: Python<'_>,
        output_dir: &str,
        world_bounds: (f64, f64, f64, f64),
        simplify: bool,
        compression: &str,
    ) -> PyResult<u64> {
        // Flush fragment writer
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        // Read all fragments grouped by tile
        let mut reader = Fragment2DReader::open_dir(&self.frag_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let groups = reader.read_all_grouped()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let tiles: Vec<((u32, u32, u32), Vec<Fragment2D>)> = groups.into_iter().collect();
        let max_zoom = self.max_zoom;
        let tags_snapshot: HashMap<u32, Vec<(String, TagValue)>> = self.tags_registry.clone();
        let tmz_snapshot: HashMap<String, u32> = self.tag_min_zoom.clone();
        let out_dir = output_dir.to_string();
        let comp = compression.to_string();

        let total = py.allow_threads(|| {
            write_parquet_native(
                &tiles, &out_dir, &world_bounds, max_zoom,
                simplify, &tags_snapshot, &comp, &mut HashMap::new(), &tmz_snapshot,
            )
        }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

        Ok(total)
    }

    /// Bounded Parquet (G1, streaming_review.md §G): byte-chunk the shard files
    /// and write each chunk via `write_parquet_native` with a PERSISTENT
    /// per-zoom part counter, so peak resident scales with `max_batch_bytes`
    /// (one chunk's tiles) instead of the whole corpus (the unbounded
    /// `generate_parquet_native` read everything into RAM). Output ROWS are
    /// identical; only part-file numbering differs (persistent cross-chunk vs
    /// per-zoom restart) — readers use Hive/glob discovery. Truncated shards are
    /// surfaced (S1 trailer) rather than silently dropped.
    #[pyo3(signature = (output_dir, world_bounds, simplify=true, compression="zstd", max_batch_bytes=2_000_000_000))]
    fn generate_parquet_native_partitioned(
        &mut self,
        py: Python<'_>,
        output_dir: &str,
        world_bounds: (f64, f64, f64, f64),
        simplify: bool,
        compression: &str,
        max_batch_bytes: usize,
    ) -> PyResult<u64> {
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }
        let mut shard_paths: Vec<std::path::PathBuf> = std::fs::read_dir(&self.frag_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "mf2d"))
            .collect();
        shard_paths.sort(); // stable chunk boundaries

        let max_zoom = self.max_zoom;
        let tags_snapshot: HashMap<u32, Vec<(String, TagValue)>> = self.tags_registry.clone();
        let tmz_snapshot: HashMap<String, u32> = self.tag_min_zoom.clone();
        let out_dir = output_dir.to_string();
        let comp = compression.to_string();
        let budget = max_batch_bytes.max(1);

        let total = py.allow_threads(|| -> Result<u64, String> {
            // Byte-budget chunks of shard paths (mirrors 3D run_native_partitioned).
            let mut chunks: Vec<Vec<std::path::PathBuf>> = Vec::new();
            let n = shard_paths.len();
            let mut cursor = 0usize;
            while cursor < n {
                let start = cursor;
                let mut est = 0usize;
                let mut end = start;
                while end < n && (end == start || est < budget) {
                    est += std::fs::metadata(&shard_paths[end])
                        .map(|m| m.len() as usize * 3)
                        .unwrap_or(0);
                    end += 1;
                }
                chunks.push(shard_paths[start..end].to_vec());
                cursor = end;
            }
            // One chunk resident at a time; persistent per-zoom part numbering.
            let mut part_base: HashMap<u32, u32> = HashMap::new();
            let mut total = 0u64;
            for chunk in chunks {
                let mut reader = Fragment2DReader::from_paths(chunk).map_err(|e| e.to_string())?;
                let groups = reader.read_all_grouped().map_err(|e| e.to_string())?;
                let tiles: Vec<((u32, u32, u32), Vec<Fragment2D>)> = groups.into_iter().collect();
                total += write_parquet_native(
                    &tiles, &out_dir, &world_bounds, max_zoom,
                    simplify, &tags_snapshot, &comp, &mut part_base, &tmz_snapshot,
                )?;
            }
            Ok(total)
        }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

        Ok(total)
    }

    /// Generate PBF + Parquet concurrently from the same fragments.
    ///
    /// Reads fragments once, then uses rayon::join to run PBF encoding
    /// and Parquet writing in parallel. Returns (tile_count, parquet_rows).
    #[pyo3(signature = (pbf_dir, parquet_dir, world_bounds, extent=4096, simplify=true, layer_name="geojsonLayer", compression="zstd"))]
    fn generate_all(
        &mut self,
        py: Python<'_>,
        pbf_dir: &str,
        parquet_dir: &str,
        world_bounds: (f64, f64, f64, f64),
        extent: u32,
        simplify: bool,
        layer_name: &str,
        compression: &str,
    ) -> PyResult<(u32, u64)> {
        // Flush fragment writer
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        // Read all fragments once
        let mut reader = Fragment2DReader::open_dir(&self.frag_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let groups = reader.read_all_grouped()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let tiles: Vec<((u32, u32, u32), Vec<Fragment2D>)> = groups.into_iter().collect();
        let max_zoom = self.max_zoom;
        let min_zoom = self.min_zoom;
        let tags_snapshot: HashMap<u32, Vec<(String, TagValue)>> = self.tags_registry.clone();
        let tmz_snapshot: HashMap<String, u32> = self.tag_min_zoom.clone();
        let pbf_out = pbf_dir.to_string();
        let pq_out = parquet_dir.to_string();
        let ln = layer_name.to_string();
        let comp = compression.to_string();

        // Run PBF + Parquet concurrently via rayon::join (GIL released)
        let (pbf_result, pq_result) = py.allow_threads(|| {
            rayon::join(
                || generate_pbf_tiles(
                    &tiles, &pbf_out, &world_bounds, max_zoom, extent,
                    simplify, &ln, &tags_snapshot, &tmz_snapshot,
                ),
                || write_parquet_native(
                    &tiles, &pq_out, &world_bounds, max_zoom,
                    simplify, &tags_snapshot, &comp, &mut HashMap::new(), &tmz_snapshot,
                ),
            )
        });

        let tile_count = pbf_result
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
        let pq_rows = pq_result
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

        // Write PBF metadata.json
        let meta_path = std::path::Path::new(pbf_dir).join("metadata.json");
        let center_x = (world_bounds.0 + world_bounds.2) / 2.0;
        let center_y = (world_bounds.1 + world_bounds.3) / 2.0;
        let meta = serde_json::json!({
            "tilejson": "3.0.0",
            "tiles": ["{z}/{x}/{y}.pbf"],
            "name": "MicroJSON Vector Tiles",
            "description": "Vector tiles generated by microjson Rust pipeline",
            "version": "1.0.0",
            "minzoom": min_zoom,
            "maxzoom": max_zoom,
            "bounds": [world_bounds.0, world_bounds.1, world_bounds.2, world_bounds.3],
            "center": [0.0, center_x, center_y],
            "vector_layers": [{
                "id": ln,
                "fields": {},
                "minzoom": min_zoom,
                "maxzoom": max_zoom,
            }],
            "tile_count": tile_count,
        });
        std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        Ok((tile_count, pq_rows))
    }
}

// ---------------------------------------------------------------------------
// Parquet generation (pure Rust, parallel row collection + sequential write)
// ---------------------------------------------------------------------------

fn write_parquet_native(
    tiles: &[((u32, u32, u32), Vec<Fragment2D>)],
    output_dir: &str,
    world_bounds: &(f64, f64, f64, f64),
    max_zoom: u32,
    simplify: bool,
    tags: &HashMap<u32, Vec<(String, TagValue)>>,
    compression: &str,
    // G1 (streaming_review.md §G): persistent per-zoom part-number base so the
    // bounded partitioned path can call this once per shard-chunk without part
    // files colliding. The whole-corpus callers pass a fresh empty map (base 0
    // ⇒ byte-identical to before).
    part_base: &mut HashMap<u32, u32>,
    tag_min_zoom: &HashMap<String, u32>,
) -> Result<u64, String> {
    use arrow::array::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;

    if tiles.is_empty() {
        return Ok(0);
    }

    // Group tiles by zoom level first — no row collection yet
    let min_zoom = tiles.iter().map(|((z, _, _), _)| *z).min().unwrap_or(0);
    let max_z = tiles.iter().map(|((z, _, _), _)| *z).max().unwrap_or(0);

    let mut tiles_by_zoom: Vec<Vec<&((u32, u32, u32), Vec<Fragment2D>)>> =
        vec![Vec::new(); (max_z as usize) + 1];
    for entry in tiles {
        tiles_by_zoom[entry.0 .0 as usize].push(entry);
    }

    // Arrow schema matching the Python parquet_writer output
    let schema = Schema::new(vec![
        Field::new("tile_x", DataType::UInt16, false),
        Field::new("tile_y", DataType::UInt16, false),
        Field::new("feature_id", DataType::UInt32, false),
        Field::new("geom_type", DataType::UInt8, false),
        Field::new("positions", DataType::LargeBinary, false),
        Field::new("indices", DataType::LargeBinary, false),
        Field::new("ring_lengths",
            DataType::List(std::sync::Arc::new(Field::new("item", DataType::UInt32, true))),
            false),
        Field::new("tags",
            DataType::Map(
                std::sync::Arc::new(Field::new("entries", DataType::Struct(
                    vec![
                        Field::new("keys", DataType::Utf8, false),
                        Field::new("values", DataType::Utf8, true),
                    ].into()
                ), false)),
                false,
            ),
            false),
    ]);
    let schema_ref = std::sync::Arc::new(schema);

    // PY-1 (streaming_review.md §G): match level 3 (parquet-rs ZSTD(Default) is
    // level 1) — same fix as the 3D legacy path.
    let comp = match compression {
        "zstd" => Compression::ZSTD(
            parquet::basic::ZstdLevel::try_new(3)
                .map_err(|e| format!("zstd level: {}", e))?,
        ),
        "lz4" => Compression::LZ4_RAW,
        "snappy" => Compression::SNAPPY,
        _ => Compression::UNCOMPRESSED,
    };

    let props = WriterProperties::builder()
        .set_compression(comp)
        .build();

    let out_path = std::path::Path::new(output_dir);
    std::fs::create_dir_all(out_path).map_err(|e| format!("mkdir: {}", e))?;

    let total_rows = std::sync::atomic::AtomicU64::new(0);
    let total_rows_ref = &total_rows;

    // Process one zoom level at a time
    for zoom in min_zoom..=max_z {
        let zoom_tiles = &tiles_by_zoom[zoom as usize];
        if zoom_tiles.is_empty() {
            continue;
        }

        let zoom_dir = out_path.join(format!("zoom={}", zoom));
        std::fs::create_dir_all(&zoom_dir).map_err(|e| format!("mkdir: {}", e))?;

        // PY-2 / H1 (streaming_review.md §G): deterministic output — sort tiles
        // by key (the source is ahash iteration order) and use a FIXED part
        // grain instead of `current_num_threads()`. Mirrors the 3D legacy fix.
        let mut zoom_tiles: Vec<&((u32, u32, u32), Vec<Fragment2D>)> = zoom_tiles.clone();
        zoom_tiles.sort_by_key(|entry| entry.0);
        const PART_GRAIN: usize = 64;
        let n_parts = PART_GRAIN.min(zoom_tiles.len()).max(1);
        // G1: continue part numbering across chunk-calls (0 for whole-corpus).
        let part_base_z = *part_base.get(&zoom).unwrap_or(&0);
        let chunk_size = (zoom_tiles.len() + n_parts - 1) / n_parts;

        let chunks: Vec<(usize, &[&((u32, u32, u32), Vec<Fragment2D>)])> =
            zoom_tiles.chunks(chunk_size).enumerate().collect();
        chunks.par_iter().try_for_each(|(part_idx, tile_chunk)| -> Result<(), String> {
            let part_idx = *part_idx;
            // Collect rows for this chunk of tiles (Rayon parallel internally)
            let mut tiles_owned: Vec<((u32, u32, u32), Vec<Fragment2D>)> = tile_chunk
                .iter()
                .map(|&entry| entry.clone())
                .collect();
            // H1a: canonicalize within-tile fragment order. Fragment2D has no
            // frag_cmp; sort by (feature_id, geom_type, ring_lengths, xy bits)
            // — a total order over the fields, deterministic across readers.
            for (_k, frags) in tiles_owned.iter_mut() {
                frags.sort_by(|a, b| {
                    a.feature_id
                        .cmp(&b.feature_id)
                        .then_with(|| a.geom_type.cmp(&b.geom_type))
                        .then_with(|| a.ring_lengths.cmp(&b.ring_lengths))
                        .then_with(|| a.xy.len().cmp(&b.xy.len()))
                        .then_with(|| {
                            a.xy.iter()
                                .zip(b.xy.iter())
                                .find_map(|(x, y)| match x.total_cmp(y) {
                                    std::cmp::Ordering::Equal => None,
                                    o => Some(o),
                                })
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                });
            }
            let rows = collect_parquet_rows_2d(&tiles_owned, world_bounds, max_zoom, simplify);
            drop(tiles_owned);

            if rows.is_empty() { return Ok(()); }

            // muDM tile-geometry parquet (custom schema, NOT GeoParquet) — see mudm-data .mu.parquet convention
            let file_path =
                zoom_dir.join(format!("part_{:03}.mu.parquet", part_base_z as usize + part_idx));
            let file = std::fs::File::create(&file_path)
                .map_err(|e| format!("create {}: {}", file_path.display(), e))?;
            let mut pq_writer = ArrowWriter::try_new(file, schema_ref.clone(), Some(props.clone()))
                .map_err(|e| format!("ArrowWriter: {}", e))?;

            let n = rows.len();
            let tile_x: UInt16Array = rows.iter().map(|r| Some(r.tile_x)).collect();
            let tile_y: UInt16Array = rows.iter().map(|r| Some(r.tile_y)).collect();
            let feature_id: UInt32Array = rows.iter().map(|r| Some(r.feature_id)).collect();
            let geom_type: UInt8Array = rows.iter().map(|r| Some(r.geom_type)).collect();
            let positions: LargeBinaryArray = rows.iter().map(|r| Some(r.positions.as_slice())).collect();
            let indices: LargeBinaryArray = rows.iter().map(|r| Some(r.indices.as_slice())).collect();

            let mut rl_builder = ListBuilder::new(UInt32Builder::new());
            let key_builder = StringBuilder::with_capacity(n * 2, n * 32);
            let val_builder = StringBuilder::with_capacity(n * 2, n * 32);
            let mut map_builder = MapBuilder::new(None, key_builder, val_builder);

            for row in &rows {
                rl_builder.values().append_slice(&row.ring_lengths);
                rl_builder.append(true);

                if let Some(tag_vec) = tags.get(&row.feature_id) {
                    for (k, v) in tag_vec {
                        // LOD gate (Path B): skip tags whose tag_min_zoom exceeds
                        // this zoom (e.g. per-cell expression on coarse tiles).
                        if tag_min_zoom.get(k).map_or(false, |&mz| zoom < mz) {
                            continue;
                        }
                        map_builder.keys().append_value(k);
                        match v {
                            TagValue::Str(s) => map_builder.values().append_value(s),
                            TagValue::Int(i) => map_builder.values().append_value(i.to_string()),
                            TagValue::Float(f) => map_builder.values().append_value(f.to_string()),
                            TagValue::Bool(b) => map_builder.values().append_value(b.to_string()),
                        }
                    }
                }
                map_builder.append(true).map_err(|e| format!("map: {}", e))?;
            }

            let batch = arrow::record_batch::RecordBatch::try_new(
                schema_ref.clone(),
                vec![
                    std::sync::Arc::new(tile_x),
                    std::sync::Arc::new(tile_y),
                    std::sync::Arc::new(feature_id),
                    std::sync::Arc::new(geom_type),
                    std::sync::Arc::new(positions),
                    std::sync::Arc::new(indices),
                    std::sync::Arc::new(rl_builder.finish()),
                    std::sync::Arc::new(map_builder.finish()),
                ],
            ).map_err(|e| format!("RecordBatch: {}", e))?;

            pq_writer.write(&batch).map_err(|e| format!("write: {}", e))?;
            pq_writer.close().map_err(|e| format!("close: {}", e))?;

            total_rows_ref.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        })?;
        part_base.insert(zoom, part_base_z + n_parts as u32); // G1
    }

    Ok(total_rows.load(std::sync::atomic::Ordering::Relaxed))
}

// ---------------------------------------------------------------------------
// PBF generation (pure Rust, parallel)
// ---------------------------------------------------------------------------

fn generate_pbf_tiles(
    tiles: &[((u32, u32, u32), Vec<Fragment2D>)],
    output_dir: &str,
    _world_bounds: &(f64, f64, f64, f64),
    max_zoom: u32,
    extent: u32,
    simplify: bool,
    layer_name: &str,
    tags_registry: &HashMap<u32, Vec<(String, TagValue)>>,
    tag_min_zoom: &HashMap<String, u32>,
) -> Result<u32, String> {
    let tile_count = std::sync::atomic::AtomicU32::new(0);

    tiles.par_iter().try_for_each(|((tz, tx, ty), frags)| -> Result<(), String> {
        let tz = *tz;
        let tx = *tx;
        let ty = *ty;
        let do_simplify = simplify && tz < max_zoom;

        let mut mvt_features: Vec<encoder_mvt::MvtFeature> = Vec::new();

        for frag in frags {
            let n_verts = frag.xy.len() / 2;
            if n_verts == 0 {
                continue;
            }

            // Widen f32 → f64 for simplify + MVT encoding
            let xy_f64: Vec<f64> = frag.xy.iter().map(|&v| v as f64).collect();

            // Optionally simplify at coarse zoom levels
            let (work_xy, work_rl) = if do_simplify && frag.geom_type == POLYGON && !frag.ring_lengths.is_empty() {
                let eps = simplify2d::compute_tolerance(tz, max_zoom);
                if eps > 0.0 {
                    simplify2d::simplify_polygon_rings(&xy_f64, &frag.ring_lengths, eps)
                } else {
                    (xy_f64, frag.ring_lengths.clone())
                }
            } else if do_simplify && frag.geom_type == LINESTRING {
                let eps = simplify2d::compute_tolerance(tz, max_zoom);
                if eps > 0.0 {
                    let simplified = simplify2d::douglas_peucker(&xy_f64, eps);
                    (simplified, vec![])
                } else {
                    (xy_f64, vec![])
                }
            } else {
                (xy_f64, frag.ring_lengths.clone())
            };

            let work_n = work_xy.len() / 2;
            if work_n == 0 {
                continue;
            }

            // Build MVT geometry commands from normalized coords
            let cmds = encoder_mvt::encode_geometry_commands(
                frag.geom_type, &work_xy, &work_rl,
                tz, tx, ty, extent,
            );

            if cmds.is_empty() {
                continue;
            }

            // Collect tags for this feature. LOD gate (Path B): drop tags whose
            // tag_min_zoom exceeds this tile's zoom (e.g. per-cell expression on
            // coarse tiles), so heavy tags ride only the fine tiles.
            let mut tags = tags_registry.get(&frag.feature_id)
                .cloned()
                .unwrap_or_default();
            if !tag_min_zoom.is_empty() {
                tags.retain(|(k, _)| tag_min_zoom.get(k).map_or(true, |&mz| tz >= mz));
            }

            mvt_features.push(encoder_mvt::MvtFeature {
                id: frag.feature_id as u64,
                geom_type: frag.geom_type,
                geometry_commands: cmds,
                tags,
            });
        }

        if mvt_features.is_empty() {
            return Ok(());
        }

        // Encode the tile
        let tile_bytes = encoder_mvt::encode_mvt_tile(&mvt_features, layer_name, extent);

        // Write to file: {output_dir}/{z}/{x}/{y}.pbf
        let tile_dir = std::path::Path::new(output_dir)
            .join(tz.to_string())
            .join(tx.to_string());
        std::fs::create_dir_all(&tile_dir)
            .map_err(|e| format!("Failed to create dir {}: {}", tile_dir.display(), e))?;

        let tile_path = tile_dir.join(format!("{}.pbf", ty));
        std::fs::write(&tile_path, &tile_bytes)
            .map_err(|e| format!("Failed to write {}: {}", tile_path.display(), e))?;

        tile_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    })?;

    Ok(tile_count.load(std::sync::atomic::Ordering::Relaxed))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract features from a GeoJSON value (supports Feature, FeatureCollection, and bare Geometry).
fn extract_geojson_features(parsed: &serde_json::Value) -> Vec<(String, serde_json::Value, serde_json::Value)> {
    let mut features = Vec::new();

    let geojson_type = parsed.get("type").and_then(|v| v.as_str()).unwrap_or("");

    match geojson_type {
        "FeatureCollection" => {
            if let Some(feats) = parsed.get("features").and_then(|v| v.as_array()) {
                for feat in feats {
                    let props = feat.get("properties").cloned().unwrap_or(serde_json::Value::Null);
                    if let Some(geom) = feat.get("geometry") {
                        let gt = geom.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        let coords = geom.get("coordinates").cloned().unwrap_or(serde_json::Value::Null);
                        if !gt.is_empty() {
                            features.push((gt.to_string(), coords, props));
                        }
                    }
                }
            }
        }
        "Feature" => {
            let props = parsed.get("properties").cloned().unwrap_or(serde_json::Value::Null);
            if let Some(geom) = parsed.get("geometry") {
                let gt = geom.get("type").and_then(|v| v.as_str()).unwrap_or("");
                let coords = geom.get("coordinates").cloned().unwrap_or(serde_json::Value::Null);
                if !gt.is_empty() {
                    features.push((gt.to_string(), coords, props));
                }
            }
        }
        _ => {
            // Bare geometry
            let coords = parsed.get("coordinates").cloned().unwrap_or(serde_json::Value::Null);
            if !geojson_type.is_empty() && !coords.is_null() {
                features.push((geojson_type.to_string(), coords, serde_json::Value::Null));
            }
        }
    }

    features
}

/// Build the Python dict from ParquetRow2D slices.
fn build_python_dict(
    py: Python<'_>,
    rows: &[ParquetRow2D],
    tags_registry: &HashMap<u32, Vec<(String, TagValue)>>,
) -> PyResult<PyObject> {
    let n = rows.len();

    let zoom_list = PyList::empty(py);
    let tx_list = PyList::empty(py);
    let ty_list = PyList::empty(py);
    let fid_list = PyList::empty(py);
    let gt_list = PyList::empty(py);
    let pos_list = PyList::empty(py);
    let idx_list = PyList::empty(py);
    let rl_list = PyList::empty(py);
    let tags_list = PyList::empty(py);

    for row in rows {
        zoom_list.append(row.zoom)?;
        tx_list.append(row.tile_x)?;
        ty_list.append(row.tile_y)?;
        fid_list.append(row.feature_id)?;
        gt_list.append(row.geom_type)?;
        pos_list.append(pyo3::types::PyBytes::new(py, &row.positions))?;
        idx_list.append(pyo3::types::PyBytes::new(py, &row.indices))?;

        // Ring lengths as Python list
        let rl_py = PyList::empty(py);
        for &rl in &row.ring_lengths {
            rl_py.append(rl)?;
        }
        rl_list.append(rl_py)?;

        // Tags
        let tag_pairs = PyList::empty(py);
        if let Some(tags) = tags_registry.get(&row.feature_id) {
            for (k, v) in tags {
                let vs = match v {
                    TagValue::Str(s) => s.clone(),
                    TagValue::Int(i) => i.to_string(),
                    TagValue::Float(f) => f.to_string(),
                    TagValue::Bool(b) => b.to_string(),
                };
                tag_pairs.append((k.as_str(), vs.as_str()))?;
            }
        }
        tags_list.append(tag_pairs)?;
    }

    let dict = PyDict::new(py);
    dict.set_item("zoom", zoom_list)?;
    dict.set_item("tile_x", tx_list)?;
    dict.set_item("tile_y", ty_list)?;
    dict.set_item("feature_id", fid_list)?;
    dict.set_item("geom_type", gt_list)?;
    dict.set_item("positions", pos_list)?;
    dict.set_item("indices", idx_list)?;
    dict.set_item("ring_lengths", rl_list)?;
    dict.set_item("tags", tags_list)?;
    dict.set_item("row_count", n)?;

    Ok(dict.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_geojson_point() {
        let coords = serde_json::json!([10.0, 20.0]);
        let results = parse_geojson_geometry("Point", &coords, 0.0, 0.0, 100.0, 100.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, POINT);
        assert!((results[0].0.xy[0] - 0.1).abs() < 1e-10);
        assert!((results[0].0.xy[1] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_parse_geojson_linestring() {
        let coords = serde_json::json!([[0.0, 0.0], [50.0, 50.0], [100.0, 0.0]]);
        let results = parse_geojson_geometry("LineString", &coords, 0.0, 0.0, 100.0, 100.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, LINESTRING);
        assert_eq!(results[0].0.xy.len(), 6); // 3 vertices * 2
    }

    #[test]
    fn test_parse_geojson_polygon() {
        let coords = serde_json::json!([[[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]]);
        let results = parse_geojson_geometry("Polygon", &coords, 0.0, 0.0, 100.0, 100.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, POLYGON);
        assert_eq!(results[0].0.ring_lengths.len(), 1);
        assert_eq!(results[0].0.ring_lengths[0], 4);
    }

    #[test]
    fn test_extract_geojson_feature_collection() {
        let json = serde_json::json!({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                    "properties": {"name": "test"}
                }
            ]
        });
        let features = extract_geojson_features(&json);
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].0, "Point");
    }
}
