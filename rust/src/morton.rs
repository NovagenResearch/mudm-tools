/// Morton code (Z-curve) encoding for 3D coordinates.
///
/// Used by neuroglancer_multilod_draco format to sort fragment positions
/// within each LOD in Z-curve order.

/// Spread bits of a 21-bit value across a 63-bit result (every 3rd bit).
#[inline]
fn spread_bits_3d(mut v: u64) -> u64 {
    // Interleave bits for 3D Morton code (supports up to 21 bits per axis)
    v &= 0x1fffff; // 21 bits
    v = (v | (v << 32)) & 0x1f00000000ffff;
    v = (v | (v << 16)) & 0x1f0000ff0000ff;
    v = (v | (v << 8))  & 0x100f00f00f00f00f;
    v = (v | (v << 4))  & 0x10c30c30c30c30c3;
    v = (v | (v << 2))  & 0x1249249249249249;
    v
}

/// Encode a 3D coordinate as a Morton code (Z-curve).
///
/// Each axis value must be < 2^21 (about 2 million).
pub fn morton_encode_3d(x: u32, y: u32, z: u32) -> u64 {
    spread_bits_3d(x as u64)
        | (spread_bits_3d(y as u64) << 1)
        | (spread_bits_3d(z as u64) << 2)
}

/// Compare two 3D coordinates by their Morton code (Z-curve order).
pub fn morton_cmp(a: (u32, u32, u32), b: (u32, u32, u32)) -> std::cmp::Ordering {
    morton_encode_3d(a.0, a.1, a.2).cmp(&morton_encode_3d(b.0, b.1, b.2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_origin_is_zero() {
        assert_eq!(morton_encode_3d(0, 0, 0), 0);
    }

    #[test]
    fn test_x_axis() {
        // x=1 should give morton code 1 (bit 0)
        assert_eq!(morton_encode_3d(1, 0, 0), 1);
    }

    #[test]
    fn test_y_axis() {
        // y=1 should give morton code 2 (bit 1)
        assert_eq!(morton_encode_3d(0, 1, 0), 2);
    }

    #[test]
    fn test_z_axis() {
        // z=1 should give morton code 4 (bit 2)
        assert_eq!(morton_encode_3d(0, 0, 1), 4);
    }

    #[test]
    fn test_ordering() {
        // (0,0,0) < (1,0,0) < (0,1,0) < (1,1,0)
        assert!(morton_encode_3d(0, 0, 0) < morton_encode_3d(1, 0, 0));
        assert!(morton_encode_3d(1, 0, 0) < morton_encode_3d(0, 1, 0));
        assert!(morton_encode_3d(0, 1, 0) < morton_encode_3d(1, 1, 0));
    }

    #[test]
    fn test_morton_cmp() {
        use std::cmp::Ordering;
        assert_eq!(morton_cmp((0, 0, 0), (0, 0, 0)), Ordering::Equal);
        assert_eq!(morton_cmp((0, 0, 0), (1, 0, 0)), Ordering::Less);
        assert_eq!(morton_cmp((1, 0, 0), (0, 0, 0)), Ordering::Greater);
    }

    #[test]
    fn test_larger_values() {
        // Verify no panic with larger values
        let m = morton_encode_3d(1000, 2000, 500);
        assert!(m > 0);
    }
}
