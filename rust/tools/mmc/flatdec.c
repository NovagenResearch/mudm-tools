// FLAT-DELTA native decode kernels: the entire app-side decoder is
// "un-zigzag + prefix sum". Scalar (-O3) and NEON variants, raced against
// meshopt's native decoder from Python via ctypes.
//
// cc -O3 -shared -fPIC flatdec.c -o flatdec.so

#include <stdint.h>
#include <stddef.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

static inline int32_t unzz16(uint16_t z) { return (int32_t)(z >> 1) ^ -(int32_t)(z & 1); }
static inline int64_t unzz32(uint32_t z) { return (int64_t)(z >> 1) ^ -(int64_t)(z & 1); }

// positions: 3 planar streams of n u16 zigzag deltas -> interleaved u16x4
// (GPU-uploadable, KHR_mesh_quantization layout; w = 0)
void decode_pos_scalar(const uint16_t* z, size_t n, uint16_t* out) {
    for (int c = 0; c < 3; c++) {
        const uint16_t* zc = z + (size_t)c * n;
        int32_t s = 0;
        for (size_t i = 0; i < n; i++) {
            s += unzz16(zc[i]);
            out[i * 4 + c] = (uint16_t)s;
        }
    }
}

// indices: n u32 zigzag deltas -> u32
void decode_idx_scalar(const uint32_t* z, size_t n, uint32_t* out) {
    int64_t s = 0;
    for (size_t i = 0; i < n; i++) {
        s += unzz32(z[i]);
        out[i] = (uint32_t)s;
    }
}

#if defined(__aarch64__)
// NEON prefix sum over u16 lanes (wraparound arithmetic is exact for
// quantized grids: the true value always fits u16, intermediate wrap is fine).
void decode_pos_neon(const uint16_t* z, size_t n, uint16_t* out) {
    for (int c = 0; c < 3; c++) {
        const uint16_t* zc = z + (size_t)c * n;
        uint16x8_t carry = vdupq_n_u16(0);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            uint16x8_t v = vld1q_u16(zc + i);
            // un-zigzag: (v >> 1) ^ -(v & 1)  (two's-complement on u16 lanes)
            int16x8_t sign = vnegq_s16(vreinterpretq_s16_u16(vandq_u16(v, vdupq_n_u16(1))));
            v = veorq_u16(vshrq_n_u16(v, 1), vreinterpretq_u16_s16(sign));
            // intra-vector inclusive prefix sum (log steps)
            v = vaddq_u16(v, vextq_u16(vdupq_n_u16(0), v, 7));
            v = vaddq_u16(v, vextq_u16(vdupq_n_u16(0), v, 6));
            v = vaddq_u16(v, vextq_u16(vdupq_n_u16(0), v, 4));
            v = vaddq_u16(v, carry);
            // scatter into stride-4 output
            uint16_t tmp[8];
            vst1q_u16(tmp, v);
            for (int k = 0; k < 8; k++) out[(i + k) * 4 + c] = tmp[k];
            carry = vdupq_laneq_u16(v, 7);
        }
        uint16_t s = vgetq_lane_u16(carry, 7);
        for (; i < n; i++) {
            s = (uint16_t)(s + (uint16_t)unzz16(zc[i]));
            out[i * 4 + c] = s;
        }
    }
}

void decode_idx_neon(const uint32_t* z, size_t n, uint32_t* out) {
    uint32x4_t carry = vdupq_n_u32(0);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint32x4_t v = vld1q_u32(z + i);
        int32x4_t sign = vnegq_s32(vreinterpretq_s32_u32(vandq_u32(v, vdupq_n_u32(1))));
        v = veorq_u32(vshrq_n_u32(v, 1), vreinterpretq_u32_s32(sign));
        v = vaddq_u32(v, vextq_u32(vdupq_n_u32(0), v, 3));
        v = vaddq_u32(v, vextq_u32(vdupq_n_u32(0), v, 2));
        v = vaddq_u32(v, carry);
        vst1q_u32(out + i, v);
        carry = vdupq_laneq_u32(v, 3);
    }
    uint32_t s = vgetq_lane_u32(carry, 3);
    for (; i < n; i++) {
        s += (uint32_t)unzz32(z[i]);
        out[i] = s;
    }
}
#endif
