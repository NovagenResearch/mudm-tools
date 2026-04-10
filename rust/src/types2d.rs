/// Core types for the 2D tiling pipeline.

/// 2D bounding box.
#[derive(Debug, Clone, Copy)]
pub struct BBox2D {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl BBox2D {
    pub fn empty() -> Self {
        Self {
            min_x: f64::INFINITY,
            min_y: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            max_y: f64::NEG_INFINITY,
        }
    }

    pub fn expand(&mut self, x: f64, y: f64) {
        if x < self.min_x { self.min_x = x; }
        if y < self.min_y { self.min_y = y; }
        if x > self.max_x { self.max_x = x; }
        if y > self.max_y { self.max_y = y; }
    }

    pub fn is_empty(&self) -> bool {
        self.min_x > self.max_x
    }
}

/// 2D geometry type enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GeomType2D {
    Point = 1,
    LineString = 2,
    Polygon = 3,
}

impl GeomType2D {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(GeomType2D::Point),
            2 => Some(GeomType2D::LineString),
            3 => Some(GeomType2D::Polygon),
            _ => None,
        }
    }
}

/// Tile key in the quadtree (zoom, x, y).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileKey2D {
    pub z: u32,
    pub x: u32,
    pub y: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox2d_empty() {
        let bb = BBox2D::empty();
        assert!(bb.is_empty());
    }

    #[test]
    fn test_bbox2d_expand() {
        let mut bb = BBox2D::empty();
        bb.expand(1.0, 2.0);
        bb.expand(3.0, 0.5);
        assert!(!bb.is_empty());
        assert_eq!(bb.min_x, 1.0);
        assert_eq!(bb.min_y, 0.5);
        assert_eq!(bb.max_x, 3.0);
        assert_eq!(bb.max_y, 2.0);
    }

    #[test]
    fn test_geomtype2d() {
        assert_eq!(GeomType2D::from_u8(1), Some(GeomType2D::Point));
        assert_eq!(GeomType2D::from_u8(2), Some(GeomType2D::LineString));
        assert_eq!(GeomType2D::from_u8(3), Some(GeomType2D::Polygon));
        assert_eq!(GeomType2D::from_u8(0), None);
        assert_eq!(GeomType2D::from_u8(4), None);
    }
}
