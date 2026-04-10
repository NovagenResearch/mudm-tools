/// Core types for the 3D tiling pipeline.

/// 3D bounding box.
#[derive(Debug, Clone, Copy)]
pub struct BBox3D {
    pub min_x: f64,
    pub min_y: f64,
    pub min_z: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub max_z: f64,
}

impl BBox3D {
    pub fn empty() -> Self {
        Self {
            min_x: f64::INFINITY,
            min_y: f64::INFINITY,
            min_z: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            max_y: f64::NEG_INFINITY,
            max_z: f64::NEG_INFINITY,
        }
    }

    pub fn expand(&mut self, x: f64, y: f64, z: f64) {
        if x < self.min_x { self.min_x = x; }
        if y < self.min_y { self.min_y = y; }
        if z < self.min_z { self.min_z = z; }
        if x > self.max_x { self.max_x = x; }
        if y > self.max_y { self.max_y = y; }
        if z > self.max_z { self.max_z = z; }
    }

    pub fn is_empty(&self) -> bool {
        self.min_x > self.max_x
    }
}

/// Geometry type enum matching protobuf GeomType.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GeomType {
    Unknown = 0,
    Point3D = 1,
    LineString3D = 2,
    Polygon3D = 3,
    PolyhedralSurface = 4,
    Tin = 5,
}

impl GeomType {
    pub fn from_int(v: i32) -> Self {
        match v {
            1 => GeomType::Point3D,
            2 => GeomType::LineString3D,
            3 => GeomType::Polygon3D,
            4 => GeomType::PolyhedralSurface,
            5 => GeomType::Tin,
            _ => GeomType::Unknown,
        }
    }
}

/// Tile key in the octree (zoom, x, y, d).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileKey {
    pub z: u32,
    pub x: u32,
    pub y: u32,
    pub d: u32,
}
