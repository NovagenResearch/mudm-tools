/// 2D clipping for quadtree tiling.
///
/// Implements Sutherland-Hodgman polygon clipping, Cohen-Sutherland line
/// clipping, and point bounds filtering. The quadtree_clip function performs
/// stack-based traversal, splitting X then Y at each level.

use crate::types2d::BBox2D;

/// Small epsilon for boundary points (matches 3D pipeline EPS).
const EPS: f64 = 1e-10;

// 2D geometry type constants
const POINT: u8 = 1;
const LINESTRING: u8 = 2;
const POLYGON: u8 = 3;

// ---------------------------------------------------------------------------
// ClipFeature2D — intermediate feature for pure-Rust quadtree clipping
// ---------------------------------------------------------------------------

/// Feature in normalized [0,1]² space with bounding box for clip acceleration.
#[derive(Debug, Clone)]
pub struct ClipFeature2D {
    /// Flat x,y pairs: [x0, y0, x1, y1, ...]
    pub xy: Vec<f64>,
    /// Ring vertex counts for Polygon (exterior + holes); empty for Point/LineString.
    pub ring_lengths: Vec<u32>,
    pub geom_type: u8,
    pub bbox: BBox2D,
}

pub fn compute_bbox_2d(xy: &[f64]) -> BBox2D {
    let mut bb = BBox2D::empty();
    let n = xy.len() / 2;
    for j in 0..n {
        bb.expand(xy[j * 2], xy[j * 2 + 1]);
    }
    bb
}

// ---------------------------------------------------------------------------
// Clip features — dispatches by geometry type
// ---------------------------------------------------------------------------

/// Clip a list of features to [k1, k2) along axis (0=X, 1=Y).
fn clip_features(features: &[ClipFeature2D], k1: f64, k2: f64, axis: u8) -> Vec<ClipFeature2D> {
    let mut result = Vec::new();
    for feat in features {
        let (a_min, a_max) = match axis {
            0 => (feat.bbox.min_x, feat.bbox.max_x),
            _ => (feat.bbox.min_y, feat.bbox.max_y),
        };

        // Trivial reject — half-open interval [k1, k2)
        if a_min >= k2 || a_max < k1 {
            continue;
        }

        // Trivial accept — fully within [k1, k2)
        if a_min >= k1 && a_max < k2 {
            result.push(feat.clone());
            continue;
        }

        // Type-specific clipping
        match feat.geom_type {
            POINT => {
                result.extend(clip_points_2d(feat, k1, k2, axis));
            }
            LINESTRING => {
                result.extend(clip_line_2d(feat, k1, k2, axis));
            }
            POLYGON => {
                if let Some(clipped) = clip_polygon_2d(feat, k1, k2, axis) {
                    result.push(clipped);
                }
            }
            _ => {}
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Point clipping — simple bounds filter
// ---------------------------------------------------------------------------

fn clip_points_2d(feat: &ClipFeature2D, k1: f64, k2: f64, axis: u8) -> Vec<ClipFeature2D> {
    let mut results = Vec::new();
    let n = feat.xy.len() / 2;

    for i in 0..n {
        let val = match axis {
            0 => feat.xy[i * 2],
            _ => feat.xy[i * 2 + 1],
        };
        if k1 <= val && val < k2 {
            let px = feat.xy[i * 2];
            let py = feat.xy[i * 2 + 1];
            results.push(ClipFeature2D {
                xy: vec![px, py],
                ring_lengths: vec![],
                geom_type: POINT,
                bbox: BBox2D {
                    min_x: px, min_y: py,
                    max_x: px, max_y: py,
                },
            });
        }
    }
    results
}

// ---------------------------------------------------------------------------
// Line clipping — Cohen-Sutherland style along one axis
// ---------------------------------------------------------------------------

fn clip_line_2d(feat: &ClipFeature2D, k1: f64, k2: f64, axis: u8) -> Vec<ClipFeature2D> {
    let xy = &feat.xy;
    let n = xy.len() / 2;

    if n < 2 {
        return vec![];
    }

    let axis_val = |idx: usize| -> f64 {
        match axis {
            0 => xy[idx * 2],
            _ => xy[idx * 2 + 1],
        }
    };

    let intersect_at = |i: usize, j: usize, k: f64| -> (f64, f64) {
        let ax = xy[i * 2]; let ay = xy[i * 2 + 1];
        let bx = xy[j * 2]; let by = xy[j * 2 + 1];
        let t = match axis {
            0 => { let d = bx - ax; if d != 0.0 { (k - ax) / d } else { 0.0 } }
            _ => { let d = by - ay; if d != 0.0 { (k - ay) / d } else { 0.0 } }
        };
        (ax + (bx - ax) * t, ay + (by - ay) * t)
    };

    let mut segments: Vec<ClipFeature2D> = Vec::new();
    let mut out_xy: Vec<f64> = Vec::new();

    let flush = |out_xy: &mut Vec<f64>, segments: &mut Vec<ClipFeature2D>| {
        if out_xy.len() >= 4 {
            // At least 2 vertices (4 f64s)
            let bbox = compute_bbox_2d(out_xy);
            segments.push(ClipFeature2D {
                xy: std::mem::take(out_xy),
                ring_lengths: vec![],
                geom_type: LINESTRING,
                bbox,
            });
        } else {
            out_xy.clear();
        }
    };

    for i in 0..n - 1 {
        let a_val = axis_val(i);
        let b_val = axis_val(i + 1);
        let a_in = k1 <= a_val && a_val < k2;

        if a_in {
            out_xy.push(xy[i * 2]);
            out_xy.push(xy[i * 2 + 1]);
        }

        let cross_k1 = (a_val < k1 && b_val > k1) || (b_val < k1 && a_val > k1);
        let cross_k2 = (a_val < k2 && b_val >= k2) || (b_val < k2 && a_val >= k2);

        let mut crossings: Vec<(f64, f64, f64, bool)> = Vec::new();
        if cross_k1 {
            let (ix, iy) = intersect_at(i, i + 1, k1);
            let d = b_val - a_val;
            let t = if d != 0.0 { (k1 - a_val) / d } else { 0.0 };
            let entering = b_val > a_val;
            crossings.push((t, ix, iy, entering));
        }
        if cross_k2 {
            let (ix, iy) = intersect_at(i, i + 1, k2);
            let d = b_val - a_val;
            let t = if d != 0.0 { (k2 - a_val) / d } else { 0.0 };
            let entering = b_val < a_val;
            crossings.push((t, ix, iy, entering));
        }

        crossings.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for (_, ix, iy, entering) in crossings {
            if entering {
                flush(&mut out_xy, &mut segments);
                out_xy.push(ix);
                out_xy.push(iy);
            } else {
                out_xy.push(ix);
                out_xy.push(iy);
                flush(&mut out_xy, &mut segments);
            }
        }
    }

    // Last point
    if n > 0 {
        let last_val = axis_val(n - 1);
        if k1 <= last_val && last_val < k2 {
            out_xy.push(xy[(n - 1) * 2]);
            out_xy.push(xy[(n - 1) * 2 + 1]);
        }
    }

    flush(&mut out_xy, &mut segments);
    segments
}

// ---------------------------------------------------------------------------
// Polygon clipping — Sutherland-Hodgman per ring
// ---------------------------------------------------------------------------

/// Clip one ring against a half-plane: keep the side where axis_val >= k (if `keep_ge`)
/// or axis_val < k (if `!keep_ge`).
fn sh_clip_ring(ring_xy: &[f64], axis: u8, k: f64, keep_ge: bool) -> Vec<f64> {
    let n = ring_xy.len() / 2;
    if n == 0 {
        return vec![];
    }

    let inside = |idx: usize| -> bool {
        let val = match axis {
            0 => ring_xy[idx * 2],
            _ => ring_xy[idx * 2 + 1],
        };
        if keep_ge { val >= k } else { val < k }
    };

    let intersect = |i: usize, j: usize| -> (f64, f64) {
        let ax = ring_xy[i * 2]; let ay = ring_xy[i * 2 + 1];
        let bx = ring_xy[j * 2]; let by = ring_xy[j * 2 + 1];
        let t = match axis {
            0 => { let d = bx - ax; if d != 0.0 { (k - ax) / d } else { 0.0 } }
            _ => { let d = by - ay; if d != 0.0 { (k - ay) / d } else { 0.0 } }
        };
        (ax + (bx - ax) * t, ay + (by - ay) * t)
    };

    let mut out = Vec::new();
    for i in 0..n {
        let j = if i == 0 { n - 1 } else { i - 1 };
        let cur_in = inside(i);
        let prev_in = inside(j);

        if cur_in {
            if !prev_in {
                // Entering: add intersection
                let (ix, iy) = intersect(j, i);
                out.push(ix);
                out.push(iy);
            }
            out.push(ring_xy[i * 2]);
            out.push(ring_xy[i * 2 + 1]);
        } else if prev_in {
            // Leaving: add intersection
            let (ix, iy) = intersect(j, i);
            out.push(ix);
            out.push(iy);
        }
    }

    out
}

/// Clip a polygon ring against the half-open interval [k1, k2) along axis.
fn sh_clip_ring_interval(ring_xy: &[f64], axis: u8, k1: f64, k2: f64) -> Vec<f64> {
    // First clip: keep >= k1
    let clipped = sh_clip_ring(ring_xy, axis, k1, true);
    if clipped.is_empty() {
        return clipped;
    }
    // Second clip: keep < k2
    sh_clip_ring(&clipped, axis, k2, false)
}

fn clip_polygon_2d(feat: &ClipFeature2D, k1: f64, k2: f64, axis: u8) -> Option<ClipFeature2D> {
    let ring_lengths = if feat.ring_lengths.is_empty() {
        vec![feat.xy.len() as u32 / 2]
    } else {
        feat.ring_lengths.clone()
    };

    let mut out_xy = Vec::new();
    let mut out_ring_lengths = Vec::new();

    let mut offset = 0usize;
    for &rl in &ring_lengths {
        let rl = rl as usize;
        let ring_start = offset * 2;
        let ring_end = (offset + rl) * 2;
        let ring_xy = &feat.xy[ring_start..ring_end];

        let clipped = sh_clip_ring_interval(ring_xy, axis, k1, k2);
        let n_clipped = clipped.len() / 2;
        if n_clipped >= 3 {
            out_xy.extend_from_slice(&clipped);
            out_ring_lengths.push(n_clipped as u32);
        }

        offset += rl;
    }

    if out_ring_lengths.is_empty() {
        return None;
    }

    let bbox = compute_bbox_2d(&out_xy);
    Some(ClipFeature2D {
        xy: out_xy,
        ring_lengths: out_ring_lengths,
        geom_type: POLYGON,
        bbox,
    })
}

// ---------------------------------------------------------------------------
// Quadtree traversal — clip one feature across all zoom levels
// ---------------------------------------------------------------------------

/// Clip a feature through the quadtree, producing (tile_key, clipped_feature) pairs.
///
/// Stack-based quadtree:
/// - Split X → Y at each level
/// - Store tile at each zoom in [min_zoom, max_zoom]
/// - Pad upper boundaries with EPS for boundary inclusion
pub fn quadtree_clip(
    feat: &ClipFeature2D,
    min_zoom: u32,
    max_zoom: u32,
    buffer: f64,
) -> Vec<((u32, u32, u32), ClipFeature2D)> {
    let mut result = Vec::new();

    // Stack: (zoom, x, y, features_at_this_node)
    let mut stack: Vec<(u32, u32, u32, Vec<ClipFeature2D>)> = vec![
        (0, 0, 0, vec![feat.clone()])
    ];

    while let Some((z, x, y, feats)) = stack.pop() {
        if feats.is_empty() {
            continue;
        }

        // Store at this level if within zoom range
        if z >= min_zoom {
            for f in &feats {
                result.push(((z, x, y), f.clone()));
            }
        }

        if z >= max_zoom {
            continue;
        }

        // Split into 4 quadrants
        let nz = z + 1;
        let n = 1u32 << nz;

        let x0 = (x * 2) as f64 / n as f64;
        let x1 = (x * 2 + 1) as f64 / n as f64;
        let x2 = (x * 2 + 2) as f64 / n as f64;
        let y0 = (y * 2) as f64 / n as f64;
        let y1 = (y * 2 + 1) as f64 / n as f64;
        let y2 = (y * 2 + 2) as f64 / n as f64;

        let buf = if buffer > 0.0 { buffer / n as f64 } else { 0.0 };

        let x2_pad = if x * 2 + 2 == n { x2 + EPS } else { x2 };
        let y2_pad = if y * 2 + 2 == n { y2 + EPS } else { y2 };

        // Clip X → two halves
        let left = clip_features(&feats, x0 - buf, x1 + buf, 0);
        let right = clip_features(&feats, x1 - buf, x2_pad + buf, 0);

        // Clip Y → four quadrants
        for (x_feats, nx) in [(left, x * 2), (right, x * 2 + 1)] {
            if x_feats.is_empty() {
                continue;
            }
            let bottom = clip_features(&x_feats, y0 - buf, y1 + buf, 1);
            let top = clip_features(&x_feats, y1 - buf, y2_pad + buf, 1);

            for (y_feats, ny) in [(bottom, y * 2), (top, y * 2 + 1)] {
                if !y_feats.is_empty() {
                    stack.push((nz, nx, ny, y_feats));
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_point_inside() {
        let feat = ClipFeature2D {
            xy: vec![0.5, 0.5],
            ring_lengths: vec![],
            geom_type: POINT,
            bbox: BBox2D { min_x: 0.5, min_y: 0.5, max_x: 0.5, max_y: 0.5 },
        };
        let result = clip_points_2d(&feat, 0.0, 1.0, 0);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_clip_point_outside() {
        let feat = ClipFeature2D {
            xy: vec![0.5, 0.5],
            ring_lengths: vec![],
            geom_type: POINT,
            bbox: BBox2D { min_x: 0.5, min_y: 0.5, max_x: 0.5, max_y: 0.5 },
        };
        let result = clip_points_2d(&feat, 0.6, 1.0, 0);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_clip_line_basic() {
        // Horizontal line from (0.0, 0.5) to (1.0, 0.5)
        let feat = ClipFeature2D {
            xy: vec![0.0, 0.5, 1.0, 0.5],
            ring_lengths: vec![],
            geom_type: LINESTRING,
            bbox: BBox2D { min_x: 0.0, min_y: 0.5, max_x: 1.0, max_y: 0.5 },
        };
        // Clip to X in [0.25, 0.75)
        let result = clip_line_2d(&feat, 0.25, 0.75, 0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].xy.len(), 4); // 2 vertices
        assert!((result[0].xy[0] - 0.25).abs() < 1e-10);
        assert!((result[0].xy[2] - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_sh_clip_ring_simple() {
        // Square: (0,0), (1,0), (1,1), (0,1)
        let ring = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        // Clip to x >= 0.5 and x < 1.5 (should cut left half)
        let clipped = sh_clip_ring_interval(&ring, 0, 0.5, 1.5);
        let n = clipped.len() / 2;
        assert!(n >= 3, "Clipped polygon should have at least 3 vertices, got {}", n);
    }

    #[test]
    fn test_clip_polygon_basic() {
        // Square: (0,0), (1,0), (1,1), (0,1) in [0,1]²
        let feat = ClipFeature2D {
            xy: vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            ring_lengths: vec![4],
            geom_type: POLYGON,
            bbox: BBox2D { min_x: 0.0, min_y: 0.0, max_x: 1.0, max_y: 1.0 },
        };
        // Clip X to [0.0, 0.5)
        let result = clip_polygon_2d(&feat, 0.0, 0.5, 0);
        assert!(result.is_some());
        let clipped = result.unwrap();
        // Should be a rectangle covering left half
        let n_verts = clipped.xy.len() / 2;
        assert!(n_verts >= 3);
        // All x coords should be in [0, 0.5]
        for i in 0..n_verts {
            assert!(clipped.xy[i * 2] <= 0.5 + 1e-10,
                    "x={} should be <= 0.5", clipped.xy[i * 2]);
        }
    }

    #[test]
    fn test_polygon_with_hole() {
        // Outer: square (0,0)-(2,0)-(2,2)-(0,2)
        // Hole: square (0.5,0.5)-(1.5,0.5)-(1.5,1.5)-(0.5,1.5)
        let feat = ClipFeature2D {
            xy: vec![
                0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0,
                0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5,
            ],
            ring_lengths: vec![4, 4],
            geom_type: POLYGON,
            bbox: BBox2D { min_x: 0.0, min_y: 0.0, max_x: 2.0, max_y: 2.0 },
        };
        // Clip X to [0.0, 1.0)
        let result = clip_polygon_2d(&feat, 0.0, 1.0, 0);
        assert!(result.is_some());
        let clipped = result.unwrap();
        // Should have 2 rings (outer clipped + hole clipped)
        assert_eq!(clipped.ring_lengths.len(), 2);
    }

    #[test]
    fn test_quadtree_clip_point() {
        let feat = ClipFeature2D {
            xy: vec![0.25, 0.75],
            ring_lengths: vec![],
            geom_type: POINT,
            bbox: BBox2D { min_x: 0.25, min_y: 0.75, max_x: 0.25, max_y: 0.75 },
        };
        let result = quadtree_clip(&feat, 0, 2, 0.0);
        assert!(!result.is_empty());
        // At zoom 0, should be in (0,0,0)
        assert!(result.iter().any(|((z, _, _), _)| *z == 0));
        // At zoom 1, should be in (0,0,1) — x in [0, 0.5), y in [0.5, 1.0)
        assert!(result.iter().any(|((z, x, y), _)| *z == 1 && *x == 0 && *y == 1));
    }

    #[test]
    fn test_quadtree_clip_polygon_spans_tiles() {
        // A square covering the entire [0,1]² space
        let feat = ClipFeature2D {
            xy: vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            ring_lengths: vec![4],
            geom_type: POLYGON,
            bbox: BBox2D { min_x: 0.0, min_y: 0.0, max_x: 1.0, max_y: 1.0 },
        };
        let result = quadtree_clip(&feat, 0, 1, 0.0);
        // zoom 0: 1 tile, zoom 1: 4 tiles
        let z0_count = result.iter().filter(|((z, _, _), _)| *z == 0).count();
        let z1_count = result.iter().filter(|((z, _, _), _)| *z == 1).count();
        assert_eq!(z0_count, 1);
        assert_eq!(z1_count, 4);
    }
}
