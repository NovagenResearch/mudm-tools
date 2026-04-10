/// CartesianProjector3D — normalizes coordinates from world bounds to [0, 1]³.
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
pub struct CartesianProjector3D {
    xmin: f64,
    ymin: f64,
    zmin: f64,
    dx: f64,
    dy: f64,
    dz: f64,
}

#[pymethods]
impl CartesianProjector3D {
    #[new]
    fn new(bounds6: (f64, f64, f64, f64, f64, f64)) -> Self {
        let (xmin, ymin, zmin, xmax, ymax, zmax) = bounds6;
        Self {
            xmin,
            ymin,
            zmin,
            dx: if xmax != xmin { xmax - xmin } else { 1.0 },
            dy: if ymax != ymin { ymax - ymin } else { 1.0 },
            dz: if zmax != zmin { zmax - zmin } else { 1.0 },
        }
    }

    /// Project world (x, y, z) to normalized [0, 1]³.
    fn project(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        (
            (x - self.xmin) / self.dx,
            (y - self.ymin) / self.dy,
            (z - self.zmin) / self.dz,
        )
    }

    /// Unproject normalized [0, 1]³ back to world coordinates.
    fn unproject(&self, nx: f64, ny: f64, nz: f64) -> (f64, f64, f64) {
        (
            nx * self.dx + self.xmin,
            ny * self.dy + self.ymin,
            nz * self.dz + self.zmin,
        )
    }
}

// Pure Rust methods (not exposed to Python)
impl CartesianProjector3D {
    pub fn project_rs(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        (
            (x - self.xmin) / self.dx,
            (y - self.ymin) / self.dy,
            (z - self.zmin) / self.dz,
        )
    }

    pub fn unproject_rs(&self, nx: f64, ny: f64, nz: f64) -> (f64, f64, f64) {
        (
            nx * self.dx + self.xmin,
            ny * self.dy + self.ymin,
            nz * self.dz + self.zmin,
        )
    }
}
