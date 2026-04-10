/// CartesianProjector2D — normalizes 2D coordinates from world bounds to [0, 1]².
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
pub struct CartesianProjector2D {
    xmin: f64,
    ymin: f64,
    dx: f64,
    dy: f64,
}

#[pymethods]
impl CartesianProjector2D {
    #[new]
    fn new(bounds4: (f64, f64, f64, f64)) -> Self {
        let (xmin, ymin, xmax, ymax) = bounds4;
        Self {
            xmin,
            ymin,
            dx: if xmax != xmin { xmax - xmin } else { 1.0 },
            dy: if ymax != ymin { ymax - ymin } else { 1.0 },
        }
    }

    /// Project world (x, y) to normalized [0, 1]².
    fn project(&self, x: f64, y: f64) -> (f64, f64) {
        (
            (x - self.xmin) / self.dx,
            (y - self.ymin) / self.dy,
        )
    }

    /// Unproject normalized [0, 1]² back to world coordinates.
    fn unproject(&self, nx: f64, ny: f64) -> (f64, f64) {
        (
            nx * self.dx + self.xmin,
            ny * self.dy + self.ymin,
        )
    }
}

// Pure Rust methods (not exposed to Python)
impl CartesianProjector2D {
    pub fn project_rs(&self, x: f64, y: f64) -> (f64, f64) {
        (
            (x - self.xmin) / self.dx,
            (y - self.ymin) / self.dy,
        )
    }

    pub fn unproject_rs(&self, nx: f64, ny: f64) -> (f64, f64) {
        (
            nx * self.dx + self.xmin,
            ny * self.dy + self.ymin,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_unproject() {
        let proj = CartesianProjector2D::new((10.0, 20.0, 110.0, 220.0));
        let (nx, ny) = proj.project_rs(60.0, 120.0);
        assert!((nx - 0.5).abs() < 1e-10);
        assert!((ny - 0.5).abs() < 1e-10);

        let (wx, wy) = proj.unproject_rs(nx, ny);
        assert!((wx - 60.0).abs() < 1e-10);
        assert!((wy - 120.0).abs() < 1e-10);
    }

    #[test]
    fn test_degenerate_bounds() {
        let proj = CartesianProjector2D::new((5.0, 5.0, 5.0, 5.0));
        let (nx, ny) = proj.project_rs(5.0, 5.0);
        assert_eq!(nx, 0.0);
        assert_eq!(ny, 0.0);
    }
}
