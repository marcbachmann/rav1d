//! The residual search's quantizer and dequantizer: a scalar lane, and AVX2 and NEON lanes with
//! the scalar lane's output. Blocks hold a multiple of 8 coefficients, each under 2^28.

use std::cmp;

use super::fdct_fast::Lane;

/// The level of each coefficient with `rounding` in 1/128 of a step, and the end of block: the
/// last nonzero level's position in `index` (the scan), or -1. Also the squared error of the
/// levels in the coefficients' scale.
#[allow(clippy::too_many_arguments)]
pub fn quantize(coefs: &[i32], levels: &mut [i32], dq: [u32; 2], qm: Option<&[u8]>, shift: u32, rounding: u32, index: &[u16]) -> (i32, u64) {
    quantize_in(Lane::detect(), coefs, levels, dq, qm, shift, rounding, index)
}

#[allow(clippy::too_many_arguments)]
pub fn quantize_in(
    lane: Lane,
    coefs: &[i32],
    levels: &mut [i32],
    dq: [u32; 2],
    qm: Option<&[u8]>,
    shift: u32,
    rounding: u32,
    index: &[u16],
) -> (i32, u64) {
    let n = coefs.len();
    assert!(n % 8 == 0 && levels.len() == n && index.len() >= n && qm.is_none_or(|m| m.len() >= n));
    match lane {
        Lane::Scalar => quantize_scalar(coefs, levels, dq, qm, shift, rounding, index),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Lane::Simd => {
            assert!(is_x86_feature_detected!("avx2"));
            // SAFETY: AVX2 is available and the lengths are checked.
            unsafe { x86::quantize(coefs, levels, dq, qm, shift, rounding, index) }
        }
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is part of aarch64 and the lengths are checked.
        Lane::Simd => unsafe { arm::quantize(coefs, levels, dq, qm, shift, rounding, index) },
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        Lane::Simd => quantize_scalar(coefs, levels, dq, qm, shift, rounding, index),
    }
}

fn quantize_scalar(coefs: &[i32], levels: &mut [i32], dq: [u32; 2], qm: Option<&[u8]>, shift: u32, rounding: u32, index: &[u16]) -> (i32, u64) {
    let mut eob = -1;
    let mut err = 0u64;
    for (rc, ((&c, l), &i)) in coefs.iter().zip(levels.iter_mut()).zip(index).enumerate() {
        let mut q = dq[(rc != 0) as usize];
        if let Some(qm) = qm {
            q = q * qm[rc] as u32 + 16 >> 5;
        }
        let a = (c.unsigned_abs() as u64) << shift;
        let v = ((a * 128 + q as u64 * rounding as u64) / (q as u64 * 128)) as i32;
        *l = if c < 0 { -v } else { v };
        if v != 0 {
            eob = cmp::max(eob, i as i32);
        }
        let d = a as i64 - q as i64 * v as i64;
        err += (d * d) as u64;
    }
    (eob, err)
}

/// The coefficients `levels` stand for, as rav1d's `decode_coefs` dequantizes them.
pub fn dequantize(levels: &[i32], cf: &mut [i16], dq: [u32; 2], qm: Option<&[u8]>, shift: u32) {
    dequantize_in(Lane::detect(), levels, cf, dq, qm, shift)
}

pub fn dequantize_in(lane: Lane, levels: &[i32], cf: &mut [i16], dq: [u32; 2], qm: Option<&[u8]>, shift: u32) {
    let n = levels.len();
    assert!(n % 8 == 0 && cf.len() == n && qm.is_none_or(|m| m.len() >= n));
    match lane {
        Lane::Scalar => dequantize_scalar(levels, cf, dq, qm, shift),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Lane::Simd => {
            assert!(is_x86_feature_detected!("avx2"));
            // SAFETY: AVX2 is available and the lengths are checked.
            unsafe { x86::dequantize(levels, cf, dq, qm, shift) }
        }
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is part of aarch64 and the lengths are checked.
        Lane::Simd => unsafe { arm::dequantize(levels, cf, dq, qm, shift) },
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        Lane::Simd => dequantize_scalar(levels, cf, dq, qm, shift),
    }
}

fn dequantize_scalar(levels: &[i32], cf: &mut [i16], dq: [u32; 2], qm: Option<&[u8]>, shift: u32) {
    for (rc, (&l, c)) in levels.iter().zip(cf.iter_mut()).enumerate() {
        if l == 0 {
            *c = 0;
            continue;
        }
        let mut q = dq[(rc != 0) as usize];
        if let Some(qm) = qm {
            q = q * qm[rc] as u32 + 16 >> 5;
        }
        let tok = l.unsigned_abs() & 0xfffff;
        let v = ((q.wrapping_mul(tok) & 0xffffff) >> shift) as i32;
        let v = cmp::min(v, 0x7fff + (l < 0) as i32);
        *c = if l < 0 { -v } else { v } as i16;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #![allow(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]

    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// The quantizer step of 8 coefficients from `rc`.
    #[inline(always)]
    unsafe fn steps(dq: [u32; 2], qm: Option<&[u8]>, rc: usize) -> __m256i {
        let mut q = _mm256_set1_epi32(dq[1] as i32);
        if rc == 0 {
            q = _mm256_blend_epi32::<1>(q, _mm256_set1_epi32(dq[0] as i32));
        }
        if let Some(qm) = qm {
            let m = _mm256_cvtepu8_epi32(_mm_loadl_epi64(qm[rc..][..8].as_ptr().cast()));
            q = _mm256_srli_epi32::<5>(_mm256_add_epi32(_mm256_mullo_epi32(q, m), _mm256_set1_epi32(16)));
        }
        q
    }

    /// (a × 128 + q × rounding) / (q × 128) of 4 lanes, exact in f64 for operands under 2^31.
    #[inline(always)]
    unsafe fn divide(a: __m128i, q: __m128i, round: __m256d) -> __m128i {
        let k128 = _mm256_set1_pd(128.0);
        let qd = _mm256_cvtepi32_pd(q);
        let num = _mm256_add_pd(_mm256_mul_pd(_mm256_cvtepi32_pd(a), k128), _mm256_mul_pd(qd, round));
        _mm256_cvttpd_epi32(_mm256_div_pd(num, _mm256_mul_pd(qd, k128)))
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn quantize(coefs: &[i32], levels: &mut [i32], dq: [u32; 2], qm: Option<&[u8]>, shift: u32, rounding: u32, index: &[u16]) -> (i32, u64) {
        let count = _mm_cvtsi32_si128(shift as i32);
        let round = _mm256_set1_pd(rounding as f64);
        let dead = _mm256_set1_epi32(128 - rounding as i32);
        let mut eob = _mm256_set1_epi32(-1);
        let mut err = _mm256_setzero_si256();
        for rc in (0..coefs.len()).step_by(8) {
            let c = _mm256_loadu_si256(coefs[rc..][..8].as_ptr().cast());
            let q = steps(dq, qm, rc);
            let a = _mm256_sll_epi32(_mm256_abs_epi32(c), count);
            // A level is 0 where 128a < q(128 - rounding): eight such coefficients skip the division.
            let live = _mm256_cmpgt_epi32(_mm256_slli_epi32::<7>(a), _mm256_sub_epi32(_mm256_mullo_epi32(q, dead), _mm256_set1_epi32(1)));
            if _mm256_testz_si256(live, live) != 0 {
                _mm256_storeu_si256(levels[rc..][..8].as_mut_ptr().cast(), _mm256_setzero_si256());
                err = _mm256_add_epi64(err, _mm256_mul_epi32(a, a));
                let odd = _mm256_srli_epi64::<32>(a);
                err = _mm256_add_epi64(err, _mm256_mul_epi32(odd, odd));
                continue;
            }
            let lo = divide(_mm256_castsi256_si128(a), _mm256_castsi256_si128(q), round);
            let hi = divide(_mm256_extracti128_si256::<1>(a), _mm256_extracti128_si256::<1>(q), round);
            let v = _mm256_set_m128i(hi, lo);
            _mm256_storeu_si256(levels[rc..][..8].as_mut_ptr().cast(), _mm256_sign_epi32(v, c));
            let zero = _mm256_cmpeq_epi32(v, _mm256_setzero_si256());
            let i = _mm256_cvtepu16_epi32(_mm_loadu_si128(index[rc..][..8].as_ptr().cast()));
            eob = _mm256_max_epi32(eob, _mm256_or_si256(i, zero));
            let d = _mm256_sub_epi32(a, _mm256_mullo_epi32(q, v));
            err = _mm256_add_epi64(err, _mm256_mul_epi32(d, d));
            let odd = _mm256_srli_epi64::<32>(d);
            err = _mm256_add_epi64(err, _mm256_mul_epi32(odd, odd));
        }
        let m = _mm_max_epi32(_mm256_castsi256_si128(eob), _mm256_extracti128_si256::<1>(eob));
        let m = _mm_max_epi32(m, _mm_shuffle_epi32::<0b01_00_11_10>(m));
        let m = _mm_max_epi32(m, _mm_shuffle_epi32::<0b10_11_00_01>(m));
        let e = _mm_add_epi64(_mm256_castsi256_si128(err), _mm256_extracti128_si256::<1>(err));
        let mut e2 = [0u64; 2];
        _mm_storeu_si128(e2.as_mut_ptr().cast(), e);
        (_mm_cvtsi128_si32(m), e2[0] + e2[1])
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn dequantize(levels: &[i32], cf: &mut [i16], dq: [u32; 2], qm: Option<&[u8]>, shift: u32) {
        let count = _mm_cvtsi32_si128(shift as i32);
        for rc in (0..levels.len()).step_by(8) {
            let l = _mm256_loadu_si256(levels[rc..][..8].as_ptr().cast());
            let q = steps(dq, qm, rc);
            let tok = _mm256_and_si256(_mm256_abs_epi32(l), _mm256_set1_epi32(0xfffff));
            let v = _mm256_srl_epi32(_mm256_and_si256(_mm256_mullo_epi32(q, tok), _mm256_set1_epi32(0xffffff)), count);
            let limit = _mm256_sub_epi32(_mm256_set1_epi32(0x7fff), _mm256_cmpgt_epi32(_mm256_setzero_si256(), l));
            let v = _mm256_sign_epi32(_mm256_min_epi32(v, limit), l);
            let packed = _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
            _mm_storeu_si128(cf[rc..][..8].as_mut_ptr().cast(), packed);
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    #![allow(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]

    use std::arch::aarch64::*;

    /// The quantizer step of 4 coefficients from `rc`.
    #[inline(always)]
    unsafe fn steps(dq: [u32; 2], qm: Option<&[u8]>, rc: usize) -> int32x4_t {
        let mut q = vdupq_n_s32(dq[1] as i32);
        if rc == 0 {
            q = vsetq_lane_s32::<0>(dq[0] as i32, q);
        }
        if let Some(qm) = qm {
            let m = &qm[rc..][..4];
            let m = vreinterpretq_s32_u32(vld1q_u32([m[0] as u32, m[1] as u32, m[2] as u32, m[3] as u32].as_ptr()));
            q = vshrq_n_s32::<5>(vaddq_s32(vmulq_s32(q, m), vdupq_n_s32(16)));
        }
        q
    }

    #[inline(always)]
    unsafe fn divide(a: int32x2_t, q: int32x2_t, round: float64x2_t) -> int32x2_t {
        let k128 = vdupq_n_f64(128.0);
        let qd = vcvtq_f64_s64(vmovl_s32(q));
        let num = vaddq_f64(vmulq_f64(vcvtq_f64_s64(vmovl_s32(a)), k128), vmulq_f64(qd, round));
        vmovn_s64(vcvtq_s64_f64(vdivq_f64(num, vmulq_f64(qd, k128))))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn quantize(coefs: &[i32], levels: &mut [i32], dq: [u32; 2], qm: Option<&[u8]>, shift: u32, rounding: u32, index: &[u16]) -> (i32, u64) {
        let count = vdupq_n_s32(shift as i32);
        let round = vdupq_n_f64(rounding as f64);
        let dead = vdupq_n_s32(128 - rounding as i32);
        let mut eob = vdupq_n_s32(-1);
        let mut err = vdupq_n_s64(0);
        for rc in (0..coefs.len()).step_by(4) {
            let c = vld1q_s32(coefs[rc..][..4].as_ptr());
            let q = steps(dq, qm, rc);
            let a = vshlq_s32(vabsq_s32(c), count);
            // A level is 0 where 128a < q(128 - rounding): four such coefficients skip the division.
            let live = vcgeq_s32(vshlq_n_s32::<7>(a), vmulq_s32(q, dead));
            if vmaxvq_u32(live) == 0 {
                vst1q_s32(levels[rc..][..4].as_mut_ptr(), vdupq_n_s32(0));
                err = vmlal_s32(err, vget_low_s32(a), vget_low_s32(a));
                err = vmlal_high_s32(err, a, a);
                continue;
            }
            let v = vcombine_s32(divide(vget_low_s32(a), vget_low_s32(q), round), divide(vget_high_s32(a), vget_high_s32(q), round));
            vst1q_s32(levels[rc..][..4].as_mut_ptr(), vbslq_s32(vcltzq_s32(c), vnegq_s32(v), v));
            let i = vreinterpretq_s32_u32(vmovl_u16(vld1_u16(index[rc..][..4].as_ptr())));
            eob = vmaxq_s32(eob, vbslq_s32(vceqzq_s32(v), vdupq_n_s32(-1), i));
            let d = vmlsq_s32(a, q, v);
            err = vmlal_s32(err, vget_low_s32(d), vget_low_s32(d));
            err = vmlal_high_s32(err, d, d);
        }
        (vmaxvq_s32(eob), vaddvq_s64(err) as u64)
    }

    pub unsafe fn dequantize(levels: &[i32], cf: &mut [i16], dq: [u32; 2], qm: Option<&[u8]>, shift: u32) {
        let count = vdupq_n_s32(-(shift as i32));
        for rc in (0..levels.len()).step_by(4) {
            let l = vld1q_s32(levels[rc..][..4].as_ptr());
            let q = vreinterpretq_u32_s32(steps(dq, qm, rc));
            let tok = vandq_u32(vreinterpretq_u32_s32(vabsq_s32(l)), vdupq_n_u32(0xfffff));
            let v = vshlq_u32(vandq_u32(vmulq_u32(q, tok), vdupq_n_u32(0xffffff)), count);
            let neg = vcltzq_s32(l);
            let limit = vsubq_s32(vdupq_n_s32(0x7fff), vreinterpretq_s32_u32(neg));
            let v = vminq_s32(vreinterpretq_s32_u32(v), limit);
            let v = vbslq_s32(neg, vnegq_s32(v), v);
            vst1_s16(cf[rc..][..4].as_mut_ptr(), vmovn_s32(v));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enc::frame::tests::Rng;

    /// Coefficients of each kind the search makes: zeros, small, near a step's rounding edge,
    /// and up to 2^20.
    fn coefs(rng: &mut Rng, n: usize, q: u32) -> Vec<i32> {
        (0..n)
            .map(|_| {
                let v = match rng.below(5) {
                    0 => 0,
                    1 => rng.below(64) as i32,
                    2 => (rng.below(40) * q + rng.below(5)) as i32 - 2 + (q as i32 * 80 / 128),
                    3 => rng.below(1 << 20) as i32,
                    _ => rng.below(4096) as i32,
                };
                if rng.below(2) == 0 { -v } else { v }
            })
            .collect()
    }

    #[test]
    fn lanes_agree() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let lanes = Lane::testable();
        for round in 0..20000 {
            let n = [16, 32, 64, 128, 256, 512, 1024][round % 7];
            let dq = [4 + rng.below(1400), 4 + rng.below(1900)];
            let qm: Option<Vec<u8>> = (rng.below(2) == 0).then(|| (0..n).map(|_| 32 + rng.below(224) as u8).collect());
            let shift = rng.below(3);
            let rounding = [48, 64, rng.below(128)][rng.below(3) as usize];
            let mut c = coefs(&mut rng, n, dq[1]);
            // Dead stretches, whole groups of 8 that quantize to 0.
            let at = rng.below(n as u32) as usize;
            c[at..(at + rng.below(64) as usize).min(n)].fill(0);
            let mut index: Vec<u16> = (0..n as u16).collect();
            for i in (1..n).rev() {
                index.swap(i, rng.below(i as u32 + 1) as usize);
            }
            let wild = (rng.below(n as u32) as usize, [i32::MIN, i32::MAX, -(1 << 20) - 3, 1 << 24][rng.below(4) as usize]);
            let mut want = None;
            for &lane in &lanes {
                let mut levels = vec![i32::MAX; n];
                let r = quantize_in(lane, &c, &mut levels, dq, qm.as_deref(), shift, rounding, &index);
                let mut cf = vec![i16::MAX; n];
                let mut odd = levels.clone();
                odd[wild.0] = wild.1;
                dequantize_in(lane, &odd, &mut cf, dq, qm.as_deref(), shift);
                let got = (r, levels, cf);
                match &want {
                    None => want = Some(got),
                    Some(w) => {
                        let at = |a: &[i32], b: &[i32]| a.iter().zip(b).position(|(x, y)| x != y);
                        assert!(w.0 == got.0, "{lane:?} n {n}: eob and error {:?} vs {:?}", w.0, got.0);
                        assert!(w.1 == got.1, "{lane:?} n {n} shift {shift} rounding {rounding}: level {:?}", at(&w.1, &got.1));
                        assert!(w.2 == got.2, "{lane:?} n {n} shift {shift}: dequantized {:?}", w.2.iter().zip(&got.2).position(|(x, y)| x != y));
                    }
                }
            }
        }
    }
}
