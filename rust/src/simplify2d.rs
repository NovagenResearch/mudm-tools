/// Douglas-Peucker line/polygon simplification for 2D.

/// Douglas-Peucker simplification on a flat [x,y,...] coordinate array.
///
/// Returns simplified coordinates as flat [x,y,...].
pub fn douglas_peucker(xy: &[f64], epsilon: f64) -> Vec<f64> {
    let n = xy.len() / 2;
    if n <= 2 {
        return xy.to_vec();
    }

    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    dp_recurse(xy, 0, n - 1, epsilon, &mut keep);

    let mut result = Vec::new();
    for i in 0..n {
        if keep[i] {
            result.push(xy[i * 2]);
            result.push(xy[i * 2 + 1]);
        }
    }
    result
}

fn dp_recurse(xy: &[f64], start: usize, end: usize, epsilon: f64, keep: &mut [bool]) {
    if end <= start + 1 {
        return;
    }

    let ax = xy[start * 2];
    let ay = xy[start * 2 + 1];
    let bx = xy[end * 2];
    let by = xy[end * 2 + 1];

    let mut max_dist = 0.0;
    let mut max_idx = start + 1;

    let dx = bx - ax;
    let dy = by - ay;
    let len_sq = dx * dx + dy * dy;

    for i in (start + 1)..end {
        let px = xy[i * 2];
        let py = xy[i * 2 + 1];

        let dist = if len_sq < 1e-30 {
            // Start and end are coincident — use distance to start
            let ex = px - ax;
            let ey = py - ay;
            (ex * ex + ey * ey).sqrt()
        } else {
            // Perpendicular distance from point to line
            ((py - ay) * dx - (px - ax) * dy).abs() / len_sq.sqrt()
        };

        if dist > max_dist {
            max_dist = dist;
            max_idx = i;
        }
    }

    if max_dist > epsilon {
        keep[max_idx] = true;
        dp_recurse(xy, start, max_idx, epsilon, keep);
        dp_recurse(xy, max_idx, end, epsilon, keep);
    }
}

/// Compute each vertex's "importance": the perpendicular deviation at which it
/// becomes a Douglas-Peucker split point. Endpoints are infinite (always kept).
/// This is the standard geojson-vt-style importance ranking; it lets a caller
/// threshold by tolerance AND enforce a minimum-vertex floor from one pass.
fn dp_importance(xy: &[f64], start: usize, end: usize, imp: &mut [f64]) {
    if end <= start + 1 {
        return;
    }
    let ax = xy[start * 2];
    let ay = xy[start * 2 + 1];
    let bx = xy[end * 2];
    let by = xy[end * 2 + 1];
    let dx = bx - ax;
    let dy = by - ay;
    let len_sq = dx * dx + dy * dy;

    let mut max_dist = 0.0;
    let mut max_idx = start + 1;
    for i in (start + 1)..end {
        let px = xy[i * 2];
        let py = xy[i * 2 + 1];
        let dist = if len_sq < 1e-30 {
            let ex = px - ax;
            let ey = py - ay;
            (ex * ex + ey * ey).sqrt()
        } else {
            ((py - ay) * dx - (px - ax) * dy).abs() / len_sq.sqrt()
        };
        if dist > max_dist {
            max_dist = dist;
            max_idx = i;
        }
    }
    imp[max_idx] = max_dist;
    dp_importance(xy, start, max_idx, imp);
    dp_importance(xy, max_idx, end, imp);
}

/// Douglas-Peucker simplification with a MINIMUM-vertex floor.
///
/// Keeps every vertex whose deviation exceeds `epsilon`, but never reduces below
/// `min_verts` vertices: if the epsilon threshold would drop below the floor, the
/// most-important remaining vertices are added back until `min_verts` are kept.
/// Unlike a plain DP + revert-to-original guard, this reduces a ring *to* the
/// floor rather than snapping back to full detail.
pub fn douglas_peucker_floor(xy: &[f64], epsilon: f64, min_verts: usize) -> Vec<f64> {
    let n = xy.len() / 2;
    let floor = min_verts.max(2);
    if n <= floor {
        return xy.to_vec();
    }
    let mut imp = vec![0.0f64; n];
    imp[0] = f64::INFINITY;
    imp[n - 1] = f64::INFINITY;
    dp_importance(xy, 0, n - 1, &mut imp);

    let mut keep: Vec<bool> = imp.iter().map(|&d| d > epsilon).collect();
    let mut kept = keep.iter().filter(|&&k| k).count();
    if kept < floor {
        // Add back the most-important not-yet-kept vertices to reach the floor.
        let mut order: Vec<usize> = (0..n).filter(|&i| !keep[i]).collect();
        order.sort_by(|&a, &b| imp[b].partial_cmp(&imp[a]).unwrap_or(std::cmp::Ordering::Equal));
        for &i in &order {
            if kept >= floor {
                break;
            }
            keep[i] = true;
            kept += 1;
        }
    }
    let mut result = Vec::with_capacity(kept * 2);
    for i in 0..n {
        if keep[i] {
            result.push(xy[i * 2]);
            result.push(xy[i * 2 + 1]);
        }
    }
    result
}

/// Simplify polygon rings using Douglas-Peucker with a minimum-vertex floor.
///
/// `min_verts` is the minimum ring vertex count (including the closing vertex);
/// a ring is reduced *to* this floor rather than reverting to full detail, and
/// rings already at/below the floor are kept as-is. Returns
/// (simplified_xy, simplified_ring_lengths).
pub fn simplify_polygon_rings(
    xy: &[f64],
    ring_lengths: &[u32],
    epsilon: f64,
    min_verts: usize,
) -> (Vec<f64>, Vec<u32>) {
    let mut out_xy = Vec::new();
    let mut out_rl = Vec::new();
    // A valid closed ring needs >= 4 vertices (3 unique + closing).
    let floor = min_verts.max(4);

    let mut offset = 0usize;
    for &rl in ring_lengths {
        let rl = rl as usize;
        let ring = &xy[offset * 2..(offset + rl) * 2];

        if rl <= floor {
            out_xy.extend_from_slice(ring);
            out_rl.push(rl as u32);
        } else {
            let simplified = douglas_peucker_floor(ring, epsilon, floor);
            let n_simplified = simplified.len() / 2;
            out_xy.extend_from_slice(&simplified);
            out_rl.push(n_simplified as u32);
        }

        offset += rl;
    }

    (out_xy, out_rl)
}

/// Compute Douglas-Peucker tolerance for a given zoom level.
///
/// At max_zoom, epsilon=0 (no simplification). At coarser levels,
/// tolerance doubles per zoom level.
pub fn compute_tolerance(zoom: u32, max_zoom: u32) -> f64 {
    if zoom >= max_zoom {
        return 0.0;
    }
    let zoom_diff = max_zoom - zoom;
    // Base tolerance: 1 pixel in normalized space at max_zoom
    // Each zoom level out doubles the tolerance
    let base = 1.0 / (1u64 << max_zoom) as f64;
    base * (1u64 << zoom_diff) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dp_straight_line() {
        // Points on a straight line — all intermediate points should be removed
        let xy = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let result = douglas_peucker(&xy, 0.01);
        assert_eq!(result.len(), 4); // 2 points (start + end)
    }

    #[test]
    fn test_dp_zigzag() {
        // Zigzag pattern — points should be kept
        let xy = vec![0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 3.0, 1.0, 4.0, 0.0];
        let result = douglas_peucker(&xy, 0.01);
        assert_eq!(result.len(), 10); // all 5 points kept
    }

    #[test]
    fn test_dp_preserves_endpoints() {
        let xy = vec![0.0, 0.0, 0.5, 0.001, 1.0, 0.0];
        let result = douglas_peucker(&xy, 0.01);
        // Middle point is within epsilon — should be removed
        assert_eq!(result.len(), 4); // 2 points
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 1.0);
        assert_eq!(result[3], 0.0);
    }

    #[test]
    fn test_dp_two_points() {
        let xy = vec![0.0, 0.0, 1.0, 1.0];
        let result = douglas_peucker(&xy, 0.01);
        assert_eq!(result, xy);
    }

    #[test]
    fn test_simplify_polygon_rings() {
        // Outer ring with a collinear point; floor 4 lets it drop to 4.
        let xy = vec![0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let rl = vec![5];
        let (simp_xy, simp_rl) = simplify_polygon_rings(&xy, &rl, 0.01, 4);
        // (0.5, 0.0) is collinear with (0,0)-(1,0), should be removed
        assert_eq!(simp_rl[0], 4);
        assert_eq!(simp_xy.len(), 8);
    }

    #[test]
    fn test_floor_reduces_not_reverts() {
        // A many-vertex ring with a collapsing epsilon must reduce TO the floor,
        // never revert to the full ring (the old guard's failure mode).
        let nv = 20usize;
        let mut xy = Vec::new();
        for i in 0..nv {
            let a = (i as f64) / (nv as f64) * std::f64::consts::TAU;
            xy.push(a.cos());
            xy.push(a.sin());
        }
        let rl = vec![nv as u32];
        let (_x, srl) = simplify_polygon_rings(&xy, &rl, 1e9, 6);
        assert_eq!(srl[0], 6, "must reduce to the floor, not revert to full");
    }

    #[test]
    fn test_dp_floor_keeps_min() {
        // Huge epsilon would collapse to 2 endpoints; floor keeps min_verts.
        let xy = vec![0.0, 0.0, 1.0, 0.001, 2.0, 0.0, 3.0, 0.001, 4.0, 0.0];
        let out = douglas_peucker_floor(&xy, 1e9, 4);
        assert_eq!(out.len() / 2, 4);
    }

    #[test]
    fn test_compute_tolerance() {
        assert_eq!(compute_tolerance(4, 4), 0.0);
        let t = compute_tolerance(3, 4);
        assert!(t > 0.0);
        let t2 = compute_tolerance(2, 4);
        assert!((t2 - t * 2.0).abs() < 1e-15);
    }
}
