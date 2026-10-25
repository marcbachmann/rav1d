//! Distortion kernels of the mode search: a scalar lane, and AVX2 and NEON lanes with the scalar
//! lane's output.

use super::fdct_fast::Lane;

/// The sum of absolute 4×4 Hadamard coefficients of `src` − `rec` over `w`×`h` 8-bit pixels,
/// both multiples of 4, rows `stride` apart.
pub fn satd(src: &[u8], rec: &[u8], stride: usize, w: usize, h: usize) -> u64 {
    satd_in(Lane::detect(), src, rec, stride, w, h)
}

pub fn satd_in(lane: Lane, src: &[u8], rec: &[u8], stride: usize, w: usize, h: usize) -> u64 {
    assert!(w % 4 == 0 && h % 4 == 0 && w <= stride);
    if h == 0 {
        return 0;
    }
    let len = (h - 1) * stride + w;
    let (src, rec) = (&src[..len], &rec[..len]);
    match lane {
        Lane::Scalar => satd_scalar(src, rec, stride, w, h),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Lane::Simd => {
            assert!(is_x86_feature_detected!("avx2"));
            // SAFETY: AVX2 is available and the slices hold every row.
            unsafe { x86::satd(src, rec, stride, w, h) }
        }
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is part of aarch64 and the slices hold every row.
        Lane::Simd => unsafe { arm::satd(src, rec, stride, w, h) },
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        Lane::Simd => satd_scalar(src, rec, stride, w, h),
    }
}

fn satd_scalar(src: &[u8], rec: &[u8], stride: usize, w: usize, h: usize) -> u64 {
    let h4 = |v: [i32; 4]| {
        let (a, b, c, e) = (v[0] + v[1], v[0] - v[1], v[2] + v[3], v[2] - v[3]);
        [a + c, b + e, a - c, b - e]
    };
    let mut total = 0;
    for by in (0..h).step_by(4) {
        for bx in (0..w).step_by(4) {
            let mut d = [[0i32; 4]; 4];
            for (r, row) in d.iter_mut().enumerate() {
                let at = (by + r) * stride + bx;
                for c in 0..4 {
                    row[c] = src[at + c] as i32 - rec[at + c] as i32;
                }
            }
            let rows = d.map(h4);
            for c in 0..4 {
                total += h4([rows[0][c], rows[1][c], rows[2][c], rows[3][c]]).iter().map(|v| v.unsigned_abs() as u64).sum::<u64>();
            }
        }
    }
    total
}

/// The sum of squared differences of `w`×`h` 8-bit pixels, rows `sa` and `sb` apart.
pub fn sse(a: &[u8], sa: usize, b: &[u8], sb: usize, w: usize, h: usize) -> u64 {
    sse_in(Lane::detect(), a, sa, b, sb, w, h)
}

pub fn sse_in(lane: Lane, a: &[u8], sa: usize, b: &[u8], sb: usize, w: usize, h: usize) -> u64 {
    assert!(w <= sa && w <= sb && w <= 64);
    if h == 0 || w == 0 {
        return 0;
    }
    let (a, b) = (&a[..(h - 1) * sa + w], &b[..(h - 1) * sb + w]);
    match lane {
        Lane::Scalar => sse_scalar(a, sa, b, sb, w, h),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Lane::Simd => {
            assert!(is_x86_feature_detected!("avx2"));
            // SAFETY: AVX2 is available and the slices hold every row.
            unsafe { x86::sse(a, sa, b, sb, w, h) }
        }
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is part of aarch64 and the slices hold every row.
        Lane::Simd => unsafe { arm::sse(a, sa, b, sb, w, h) },
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        Lane::Simd => sse_scalar(a, sa, b, sb, w, h),
    }
}

fn sse_scalar(a: &[u8], sa: usize, b: &[u8], sb: usize, w: usize, h: usize) -> u64 {
    let mut s = 0;
    for r in 0..h {
        for (&x, &y) in a[r * sa..][..w].iter().zip(&b[r * sb..][..w]) {
            let d = x as i32 - y as i32;
            s += (d * d) as u64;
        }
    }
    s
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #![allow(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]

    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// The 4×4 Hadamards of rows `d` side by side, as absolute sums in i32 lanes: the columns
    /// between the rows, then the rows by pairwise sums and differences.
    #[inline(always)]
    unsafe fn hadamard8(d: [__m128i; 4]) -> __m128i {
        let (a, b) = (_mm_add_epi16(d[0], d[1]), _mm_sub_epi16(d[0], d[1]));
        let (c, e) = (_mm_add_epi16(d[2], d[3]), _mm_sub_epi16(d[2], d[3]));
        let v = [_mm_add_epi16(a, c), _mm_add_epi16(b, e), _mm_sub_epi16(a, c), _mm_sub_epi16(b, e)];
        let mut sum = _mm_setzero_si128();
        for k in [0, 2] {
            let s = _mm_hadd_epi16(v[k], v[k + 1]);
            let t = _mm_hsub_epi16(v[k], v[k + 1]);
            let ones = _mm_set1_epi16(1);
            sum = _mm_add_epi32(sum, _mm_madd_epi16(_mm_abs_epi16(_mm_hadd_epi16(s, t)), ones));
            sum = _mm_add_epi32(sum, _mm_madd_epi16(_mm_abs_epi16(_mm_hsub_epi16(s, t)), ones));
        }
        sum
    }

    #[inline(always)]
    unsafe fn hadamard16(d: [__m256i; 4]) -> __m256i {
        let (a, b) = (_mm256_add_epi16(d[0], d[1]), _mm256_sub_epi16(d[0], d[1]));
        let (c, e) = (_mm256_add_epi16(d[2], d[3]), _mm256_sub_epi16(d[2], d[3]));
        let v = [_mm256_add_epi16(a, c), _mm256_add_epi16(b, e), _mm256_sub_epi16(a, c), _mm256_sub_epi16(b, e)];
        let mut sum = _mm256_setzero_si256();
        for k in [0, 2] {
            let s = _mm256_hadd_epi16(v[k], v[k + 1]);
            let t = _mm256_hsub_epi16(v[k], v[k + 1]);
            let ones = _mm256_set1_epi16(1);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_abs_epi16(_mm256_hadd_epi16(s, t)), ones));
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_abs_epi16(_mm256_hsub_epi16(s, t)), ones));
        }
        sum
    }

    #[inline(always)]
    unsafe fn diff8(src: &[u8], rec: &[u8], at: usize) -> __m128i {
        let s = _mm_cvtepu8_epi16(_mm_loadl_epi64(src[at..][..8].as_ptr().cast()));
        let r = _mm_cvtepu8_epi16(_mm_loadl_epi64(rec[at..][..8].as_ptr().cast()));
        _mm_sub_epi16(s, r)
    }

    #[inline(always)]
    unsafe fn diff16(src: &[u8], rec: &[u8], at: usize) -> __m256i {
        let s = _mm256_cvtepu8_epi16(_mm_loadu_si128(src[at..][..16].as_ptr().cast()));
        let r = _mm256_cvtepu8_epi16(_mm_loadu_si128(rec[at..][..16].as_ptr().cast()));
        _mm256_sub_epi16(s, r)
    }

    #[inline(always)]
    unsafe fn load4x2(p: &[u8], at: usize, at2: Option<usize>) -> __m128i {
        let lo = u32::from_le_bytes(p[at..][..4].try_into().unwrap()) as i32;
        let hi = match at2 {
            Some(a) => u32::from_le_bytes(p[a..][..4].try_into().unwrap()) as i32,
            None => 0,
        };
        _mm_cvtepu8_epi16(_mm_set_epi32(0, 0, hi, lo))
    }

    /// Rows `at` and `at2` of 4 pixels each, side by side.
    #[inline(always)]
    unsafe fn diff4x2(src: &[u8], rec: &[u8], at: usize, at2: Option<usize>) -> __m128i {
        _mm_sub_epi16(load4x2(src, at, at2), load4x2(rec, at, at2))
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn satd(src: &[u8], rec: &[u8], stride: usize, w: usize, h: usize) -> u64 {
        let mut sum = _mm256_setzero_si256();
        let mut sum8 = _mm_setzero_si128();
        for by in (0..h).step_by(4) {
            let row = |r: usize| (by + r) * stride;
            let mut x = 0;
            while x + 16 <= w {
                let d = [diff16(src, rec, row(0) + x), diff16(src, rec, row(1) + x), diff16(src, rec, row(2) + x), diff16(src, rec, row(3) + x)];
                sum = _mm256_add_epi32(sum, hadamard16(d));
                x += 16;
            }
            if x + 8 <= w {
                let d = [diff8(src, rec, row(0) + x), diff8(src, rec, row(1) + x), diff8(src, rec, row(2) + x), diff8(src, rec, row(3) + x)];
                sum8 = _mm_add_epi32(sum8, hadamard8(d));
            }
        }
        if w % 8 == 4 {
            let x = w - 4;
            for by in (0..h).step_by(8) {
                let at = |r: usize| (by + r) * stride + x;
                let at2 = |r: usize| (by + 4 < h).then_some((by + 4 + r) * stride + x);
                let d = [diff4x2(src, rec, at(0), at2(0)), diff4x2(src, rec, at(1), at2(1)), diff4x2(src, rec, at(2), at2(2)), diff4x2(src, rec, at(3), at2(3))];
                sum8 = _mm_add_epi32(sum8, hadamard8(d));
            }
        }
        let s = _mm_add_epi32(sum8, _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256::<1>(sum)));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
        _mm_cvtsi128_si32(s) as u32 as u64
    }
    #[target_feature(enable = "avx2")]
    pub unsafe fn sse(a: &[u8], sa: usize, b: &[u8], sb: usize, w: usize, h: usize) -> u64 {
        let mut sum = _mm256_setzero_si256();
        let mut rest = 0u64;
        for r in 0..h {
            let (ra, rb) = (&a[r * sa..][..w], &b[r * sb..][..w]);
            let mut x = 0;
            while x + 16 <= w {
                let d = diff16(ra, rb, x);
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(d, d));
                x += 16;
            }
            if x + 8 <= w {
                let d = _mm256_castsi128_si256(diff8(ra, rb, x));
                let d = _mm256_and_si256(d, _mm256_set_epi64x(0, 0, -1, -1));
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(d, d));
                x += 8;
            }
            for (&p, &q) in ra[x..].iter().zip(&rb[x..]) {
                let d = p as i32 - q as i32;
                rest += (d * d) as u64;
            }
        }
        let lo = _mm256_unpacklo_epi32(sum, _mm256_setzero_si256());
        let hi = _mm256_unpackhi_epi32(sum, _mm256_setzero_si256());
        let s = _mm256_add_epi64(lo, hi);
        let s = _mm_add_epi64(_mm256_castsi256_si128(s), _mm256_extracti128_si256::<1>(s));
        let mut out = [0u64; 2];
        _mm_storeu_si128(out.as_mut_ptr().cast(), s);
        out[0] + out[1] + rest
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    #![allow(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]

    use std::arch::aarch64::*;

    #[inline(always)]
    unsafe fn hsub(x: int16x8_t, y: int16x8_t) -> int16x8_t {
        vsubq_s16(vuzp1q_s16(x, y), vuzp2q_s16(x, y))
    }

    /// `x86::hadamard8`'s twin: pairwise sums by `vpaddq`, differences by even minus odd lanes.
    #[inline(always)]
    unsafe fn hadamard8(d: [int16x8_t; 4], sum: uint32x4_t) -> uint32x4_t {
        let (a, b) = (vaddq_s16(d[0], d[1]), vsubq_s16(d[0], d[1]));
        let (c, e) = (vaddq_s16(d[2], d[3]), vsubq_s16(d[2], d[3]));
        let v = [vaddq_s16(a, c), vaddq_s16(b, e), vsubq_s16(a, c), vsubq_s16(b, e)];
        let mut sum = sum;
        for k in [0, 2] {
            let s = vpaddq_s16(v[k], v[k + 1]);
            let t = hsub(v[k], v[k + 1]);
            sum = vpadalq_u16(sum, vreinterpretq_u16_s16(vabsq_s16(vpaddq_s16(s, t))));
            sum = vpadalq_u16(sum, vreinterpretq_u16_s16(vabsq_s16(hsub(s, t))));
        }
        sum
    }

    #[inline(always)]
    unsafe fn diff8(src: &[u8], rec: &[u8], at: usize) -> int16x8_t {
        let s = vld1_u8(src[at..][..8].as_ptr());
        let r = vld1_u8(rec[at..][..8].as_ptr());
        vreinterpretq_s16_u16(vsubl_u8(s, r))
    }

    #[inline(always)]
    unsafe fn load4x2(p: &[u8], at: usize, at2: Option<usize>) -> uint8x8_t {
        let lo = u32::from_le_bytes(p[at..][..4].try_into().unwrap());
        let hi = match at2 {
            Some(a) => u32::from_le_bytes(p[a..][..4].try_into().unwrap()),
            None => 0,
        };
        vcreate_u8(lo as u64 | (hi as u64) << 32)
    }

    #[inline(always)]
    unsafe fn diff4x2(src: &[u8], rec: &[u8], at: usize, at2: Option<usize>) -> int16x8_t {
        vreinterpretq_s16_u16(vsubl_u8(load4x2(src, at, at2), load4x2(rec, at, at2)))
    }

    pub unsafe fn satd(src: &[u8], rec: &[u8], stride: usize, w: usize, h: usize) -> u64 {
        let mut sum = vdupq_n_u32(0);
        for by in (0..h).step_by(4) {
            let row = |r: usize| (by + r) * stride;
            for x in (0..w - w % 8).step_by(8) {
                let d = [diff8(src, rec, row(0) + x), diff8(src, rec, row(1) + x), diff8(src, rec, row(2) + x), diff8(src, rec, row(3) + x)];
                sum = hadamard8(d, sum);
            }
        }
        if w % 8 == 4 {
            let x = w - 4;
            for by in (0..h).step_by(8) {
                let at = |r: usize| (by + r) * stride + x;
                let at2 = |r: usize| (by + 4 < h).then_some((by + 4 + r) * stride + x);
                let d = [diff4x2(src, rec, at(0), at2(0)), diff4x2(src, rec, at(1), at2(1)), diff4x2(src, rec, at(2), at2(2)), diff4x2(src, rec, at(3), at2(3))];
                sum = hadamard8(d, sum);
            }
        }
        vaddvq_u32(sum) as u64
    }
    pub unsafe fn sse(a: &[u8], sa: usize, b: &[u8], sb: usize, w: usize, h: usize) -> u64 {
        let mut sum = vdupq_n_u64(0);
        let mut rest = 0u64;
        for r in 0..h {
            let (ra, rb) = (&a[r * sa..][..w], &b[r * sb..][..w]);
            let mut row = vdupq_n_u32(0);
            let mut x = 0;
            while x + 8 <= w {
                let d = vabdl_u8(vld1_u8(ra[x..][..8].as_ptr()), vld1_u8(rb[x..][..8].as_ptr()));
                row = vmlal_u16(row, vget_low_u16(d), vget_low_u16(d));
                row = vmlal_high_u16(row, d, d);
                x += 8;
            }
            sum = vpadalq_u32(sum, row);
            for (&p, &q) in ra[x..].iter().zip(&rb[x..]) {
                let d = p as i32 - q as i32;
                rest += (d * d) as u64;
            }
        }
        vaddvq_u64(sum) + rest
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enc::frame::tests::Rng;

    #[test]
    fn lanes_agree() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let lanes = [Lane::Scalar, Lane::detect()];
        for round in 0..20000 {
            let w = [4, 8, 12, 16, 20, 24, 28, 32, 36, 44, 48, 60, 64][round % 13];
            let h = 4 * (1 + rng.below(16) as usize);
            let stride = w + rng.below(3) as usize * 4 + rng.below(2) as usize;
            let len = (h - 1) * stride + w;
            let spread = [255, 255, 8, 0][rng.below(4) as usize];
            let src: Vec<u8> = (0..len).map(|_| if spread == 255 && rng.below(2) == 0 { 255 } else { rng.below(spread + 1) as u8 }).collect();
            let rec: Vec<u8> = (0..len).map(|i| if spread == 255 && rng.below(2) == 0 { 255 - src[i] } else { rng.below(spread + 1) as u8 }).collect();
            let want = satd_in(lanes[0], &src, &rec, stride, w, h);
            let got = satd_in(lanes[1], &src, &rec, stride, w, h);
            assert_eq!(want, got, "{:?} {w}x{h} stride {stride}", lanes[1]);
        }
    }

    #[test]
    fn sse_lanes_agree() {
        let mut rng = Rng(0x5851_f42d_4c95_7f2d);
        let lanes = [Lane::Scalar, Lane::detect()];
        for _ in 0..20000 {
            let w = 1 + rng.below(64) as usize;
            let h = 1 + rng.below(64) as usize;
            let (sa, sb) = (w + rng.below(9) as usize, w + rng.below(70) as usize);
            let extreme = rng.below(3) == 0;
            let a: Vec<u8> = (0..(h - 1) * sa + w).map(|_| if extreme { 255 } else { rng.below(256) as u8 }).collect();
            let b: Vec<u8> = (0..(h - 1) * sb + w).map(|_| if extreme { 0 } else { rng.below(256) as u8 }).collect();
            let want = sse_in(lanes[0], &a, sa, &b, sb, w, h);
            assert_eq!(want, sse_in(lanes[1], &a, sa, &b, sb, w, h), "{:?} {w}x{h}", lanes[1]);
        }
    }
}
