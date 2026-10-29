//! Eight `i16` lanes on SSE4.1, NEON and wasm `simd128`, with a scalar stand-in elsewhere: the
//! one source the Rust SIMD kernels are written in. Each backend maps an operation to one or two
//! instructions; the kernels never name a target.
//!
//! On x86 the methods are `#[inline(always)]` so that they compile into a kernel entered through
//! `#[target_feature(enable = "sse4.1")]`, which is only chosen once the CPU flags hold SSE41.
//! The ones from `load_u16` on need only SSE2, the x86_64 baseline, and inline anywhere.

/// Whether [`I16x8`] maps to vector instructions on this target.
pub const fn native() -> bool {
    cfg!(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128"),
    ))
}

#[derive(Clone, Copy)]
pub struct I16x8(imp::Inner);

impl I16x8 {
    /// Bytes `b[..4]` zero-extended into the low four lanes.
    #[inline(always)]
    pub fn from_u8x4(b: &[u8]) -> Self {
        Self::from_u8x8(&[b[0], b[1], b[2], b[3], 0, 0, 0, 0])
    }

    /// Little-endian lanes from `b[..8]` into the low four lanes.
    #[inline(always)]
    pub fn from_bytes_half(b: &[u8]) -> Self {
        let mut all = [0; 16];
        all[..8].copy_from_slice(&b[..8]);
        Self::from_bytes(&all)
    }

    /// The low four lanes into `d[..4]`.
    #[inline(always)]
    pub fn store_lo(self, d: &mut [i16]) {
        d[..4].copy_from_slice(&self.to_array()[..4]);
    }

    /// `self` with the sign of `sign`, where `self >= 0`.
    #[inline(always)]
    pub fn with_sign_of(self, sign: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        return self.sign(sign);
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            let m = sign.shr_s::<15>();
            self.xor(m).sub(m)
        }
    }

    /// `v[i]` lane `j` becomes `v[j]` lane `i`: three rounds of interleaves, 24 shuffles.
    #[inline(always)]
    pub fn transpose8x8(v: &mut [Self; 8]) {
        let a = [
            v[0].zip_lo16(v[1]),
            v[0].zip_hi16(v[1]),
            v[2].zip_lo16(v[3]),
            v[2].zip_hi16(v[3]),
            v[4].zip_lo16(v[5]),
            v[4].zip_hi16(v[5]),
            v[6].zip_lo16(v[7]),
            v[6].zip_hi16(v[7]),
        ];
        let b = [
            a[0].zip_lo32(a[2]),
            a[0].zip_hi32(a[2]),
            a[1].zip_lo32(a[3]),
            a[1].zip_hi32(a[3]),
            a[4].zip_lo32(a[6]),
            a[4].zip_hi32(a[6]),
            a[5].zip_lo32(a[7]),
            a[5].zip_hi32(a[7]),
        ];
        for i in 0..4 {
            v[2 * i] = b[i].zip_lo64(b[i + 4]);
            v[2 * i + 1] = b[i].zip_hi64(b[i + 4]);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unused_unsafe)]
mod imp {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use super::I16x8;

    pub type Inner = __m128i;

    // SAFETY (every `unsafe` here): SSE4.1, or SSE2 from `load_u16` on, holds where these are
    // compiled in (see the module docs), and loads and stores go through bounds-checked slices of 8 or 4 lanes.
    impl I16x8 {
        #[inline(always)]
        pub fn load(s: &[i16]) -> Self {
            Self(unsafe { _mm_loadu_si128(s[..8].as_ptr().cast()) })
        }

        /// Lanes `lo[..4]` then `hi[..4]`: two rows of a 4-wide block.
        #[inline(always)]
        pub fn load_halves(lo: &[i16], hi: &[i16]) -> Self {
            let (lo, hi) = (&lo[..4], &hi[..4]);
            Self(unsafe {
                _mm_unpacklo_epi64(
                    _mm_loadl_epi64(lo.as_ptr().cast()),
                    _mm_loadl_epi64(hi.as_ptr().cast()),
                )
            })
        }

        #[inline(always)]
        pub fn splat(v: i16) -> Self {
            Self(unsafe { _mm_set1_epi16(v) })
        }

        /// `v` in the low four lanes, the others unspecified: one shuffle fewer.
        #[inline(always)]
        pub fn splat_lo(v: i16) -> Self {
            Self(unsafe { _mm_shufflelo_epi16::<0>(_mm_cvtsi32_si128(v as i32)) })
        }

        /// Bytes `b[..8]` zero-extended.
        #[inline(always)]
        pub fn from_u8x8(b: &[u8]) -> Self {
            Self(unsafe { _mm_cvtepu8_epi16(_mm_loadl_epi64(b[..8].as_ptr().cast())) })
        }

        /// Little-endian lanes from `b[..16]`.
        #[inline(always)]
        pub fn from_bytes(b: &[u8]) -> Self {
            Self(unsafe { _mm_loadu_si128(b[..16].as_ptr().cast()) })
        }

        #[inline(always)]
        pub fn store(self, d: &mut [i16]) {
            unsafe { _mm_storeu_si128(d[..8].as_mut_ptr().cast(), self.0) }
        }

        /// The lanes saturated to `u8`.
        #[inline(always)]
        pub fn pack_u8(self) -> [u8; 8] {
            let mut out = [0; 8];
            unsafe { _mm_storel_epi64(out.as_mut_ptr().cast(), _mm_packus_epi16(self.0, self.0)) };
            out
        }

        #[inline(always)]
        pub fn to_array(self) -> [i16; 8] {
            let mut out = [0; 8];
            unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), self.0) };
            out
        }

        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            Self(unsafe { _mm_add_epi16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            Self(unsafe { _mm_sub_epi16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            Self(unsafe { _mm_mullo_epi16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn abs(self) -> Self {
            Self(unsafe { _mm_abs_epi16(self.0) })
        }

        #[inline(always)]
        pub fn shr_u(self, n: u32) -> Self {
            Self(unsafe { _mm_srl_epi16(self.0, _mm_cvtsi32_si128(n as i32)) })
        }

        #[inline(always)]
        pub fn shr_s<const N: i32>(self) -> Self {
            Self(unsafe { _mm_srai_epi16::<N>(self.0) })
        }

        #[inline(always)]
        pub fn sub_sat_u(self, o: Self) -> Self {
            Self(unsafe { _mm_subs_epu16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn min_u(self, o: Self) -> Self {
            Self(unsafe { _mm_min_epu16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn min(self, o: Self) -> Self {
            Self(unsafe { _mm_min_epi16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            Self(unsafe { _mm_max_epi16(self.0, o.0) })
        }

        #[inline(always)]
        pub(super) fn sign(self, o: Self) -> Self {
            Self(unsafe { _mm_sign_epi16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn load_u16(s: &[u16]) -> Self {
            Self(unsafe { _mm_loadu_si128(s[..8].as_ptr().cast()) })
        }

        /// `s[..4]` into the low four lanes.
        #[inline(always)]
        pub fn load_u16_lo(s: &[u16]) -> Self {
            Self(unsafe { _mm_loadl_epi64(s[..4].as_ptr().cast()) })
        }

        #[inline(always)]
        pub fn store_u16(self, d: &mut [u16]) {
            unsafe { _mm_storeu_si128(d[..8].as_mut_ptr().cast(), self.0) }
        }

        #[inline(always)]
        pub fn store_u16_lo(self, d: &mut [u16]) {
            unsafe { _mm_storel_epi64(d[..4].as_mut_ptr().cast(), self.0) }
        }

        #[inline(always)]
        pub fn and(self, o: Self) -> Self {
            Self(unsafe { _mm_and_si128(self.0, o.0) })
        }

        /// The high halves of the unsigned products.
        #[inline(always)]
        pub fn mulhi_u(self, o: Self) -> Self {
            Self(unsafe { _mm_mulhi_epu16(self.0, o.0) })
        }

        /// The unsigned averages rounded up.
        #[inline(always)]
        pub fn avg_u(self, o: Self) -> Self {
            Self(unsafe { _mm_avg_epu16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn shr_s_by(self, n: u32) -> Self {
            Self(unsafe { _mm_sra_epi16(self.0, _mm_cvtsi32_si128(n as i32)) })
        }

        #[inline(always)]
        pub fn eq(self, o: Self) -> Self {
            Self(unsafe { _mm_cmpeq_epi16(self.0, o.0) })
        }

        /// The first set lane of a mask, 8 if none is.
        #[inline(always)]
        pub fn first_set(self) -> u32 {
            (unsafe { _mm_movemask_epi8(self.0) } as u32 | 1 << 16).trailing_zeros() / 2
        }

        /// The first set lane of the masks `self` then `o`, 16 if none is.
        #[inline(always)]
        pub fn first_set2(self, o: Self) -> u32 {
            (unsafe { _mm_movemask_epi8(_mm_packs_epi16(self.0, o.0)) } as u32 | 1 << 16)
                .trailing_zeros()
        }

        #[inline(always)]
        pub fn add_sat(self, o: Self) -> Self {
            Self(unsafe { _mm_adds_epi16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn sub_sat(self, o: Self) -> Self {
            Self(unsafe { _mm_subs_epi16(self.0, o.0) })
        }

        /// `(a * b + 0x4000) >> 15`: a product in Q15 with rounding.
        #[inline(always)]
        pub fn mulhrs(self, o: Self) -> Self {
            Self(unsafe { _mm_mulhrs_epi16(self.0, o.0) })
        }

        /// The butterfly rotation `((a * c1 - b * c2 + 2048) >> 12, (a * c2 + b * c1 + 2048) >> 12)`
        /// in 32 bits, each result saturated to `i16`.
        #[inline(always)]
        pub fn rot(self, o: Self, c1: i16, c2: i16) -> (Self, Self) {
            let pair =
                |lo: i16, hi: i16| unsafe { _mm_set1_epi32(lo as u16 as i32 | (hi as i32) << 16) };
            let (k1, k2, rnd) = (pair(c1, c2.wrapping_neg()), pair(c2, c1), unsafe {
                _mm_set1_epi32(2048)
            });
            let (lo, hi) = unsafe {
                (
                    _mm_unpacklo_epi16(self.0, o.0),
                    _mm_unpackhi_epi16(self.0, o.0),
                )
            };
            let mul = |k| unsafe {
                _mm_packs_epi32(
                    _mm_srai_epi32::<12>(_mm_add_epi32(_mm_madd_epi16(lo, k), rnd)),
                    _mm_srai_epi32::<12>(_mm_add_epi32(_mm_madd_epi16(hi, k), rnd)),
                )
            };
            (Self(mul(k1)), Self(mul(k2)))
        }

        #[inline(always)]
        pub fn zip_lo16(self, o: Self) -> Self {
            Self(unsafe { _mm_unpacklo_epi16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn zip_hi16(self, o: Self) -> Self {
            Self(unsafe { _mm_unpackhi_epi16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn zip_lo32(self, o: Self) -> Self {
            Self(unsafe { _mm_unpacklo_epi32(self.0, o.0) })
        }

        #[inline(always)]
        pub fn zip_hi32(self, o: Self) -> Self {
            Self(unsafe { _mm_unpackhi_epi32(self.0, o.0) })
        }

        #[inline(always)]
        pub fn zip_lo64(self, o: Self) -> Self {
            Self(unsafe { _mm_unpacklo_epi64(self.0, o.0) })
        }

        #[inline(always)]
        pub fn zip_hi64(self, o: Self) -> Self {
            Self(unsafe { _mm_unpackhi_epi64(self.0, o.0) })
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unused_unsafe)]
mod imp {
    use std::arch::aarch64::*;

    use super::I16x8;

    pub type Inner = int16x8_t;

    #[inline(always)]
    fn u(v: int16x8_t) -> uint16x8_t {
        unsafe { vreinterpretq_u16_s16(v) }
    }

    #[inline(always)]
    fn s(v: uint16x8_t) -> int16x8_t {
        unsafe { vreinterpretq_s16_u16(v) }
    }

    // SAFETY (every `unsafe` here): NEON is part of aarch64, and loads and stores go through
    // bounds-checked slices of 8 or 4 lanes.
    impl I16x8 {
        #[inline(always)]
        pub fn load(s: &[i16]) -> Self {
            Self(unsafe { vld1q_s16(s[..8].as_ptr()) })
        }

        /// Lanes `lo[..4]` then `hi[..4]`: two rows of a 4-wide block.
        #[inline(always)]
        pub fn load_halves(lo: &[i16], hi: &[i16]) -> Self {
            let (lo, hi) = (&lo[..4], &hi[..4]);
            Self(unsafe { vcombine_s16(vld1_s16(lo.as_ptr()), vld1_s16(hi.as_ptr())) })
        }

        #[inline(always)]
        pub fn splat(v: i16) -> Self {
            Self(unsafe { vdupq_n_s16(v) })
        }

        /// `v` in the low four lanes, the others unspecified.
        #[inline(always)]
        pub fn splat_lo(v: i16) -> Self {
            Self::splat(v)
        }

        /// Bytes `b[..8]` zero-extended.
        #[inline(always)]
        pub fn from_u8x8(b: &[u8]) -> Self {
            Self(s(unsafe { vmovl_u8(vld1_u8(b[..8].as_ptr())) }))
        }

        /// Little-endian lanes from `b[..16]`.
        #[inline(always)]
        pub fn from_bytes(b: &[u8]) -> Self {
            Self(unsafe { vreinterpretq_s16_u8(vld1q_u8(b[..16].as_ptr())) })
        }

        #[inline(always)]
        pub fn store(self, d: &mut [i16]) {
            unsafe { vst1q_s16(d[..8].as_mut_ptr(), self.0) }
        }

        /// The lanes saturated to `u8`.
        #[inline(always)]
        pub fn pack_u8(self) -> [u8; 8] {
            let mut out = [0; 8];
            unsafe { vst1_u8(out.as_mut_ptr(), vqmovun_s16(self.0)) };
            out
        }

        #[inline(always)]
        pub fn to_array(self) -> [i16; 8] {
            let mut out = [0; 8];
            unsafe { vst1q_s16(out.as_mut_ptr(), self.0) };
            out
        }

        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            Self(unsafe { vaddq_s16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            Self(unsafe { vsubq_s16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            Self(unsafe { vmulq_s16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn abs(self) -> Self {
            Self(unsafe { vabsq_s16(self.0) })
        }

        #[inline(always)]
        pub fn shr_u(self, n: u32) -> Self {
            Self(s(unsafe { vshlq_u16(u(self.0), vdupq_n_s16(-(n as i16))) }))
        }

        #[inline(always)]
        pub fn shr_s<const N: i32>(self) -> Self {
            Self(unsafe { vshrq_n_s16::<N>(self.0) })
        }

        #[inline(always)]
        pub fn sub_sat_u(self, o: Self) -> Self {
            Self(s(unsafe { vqsubq_u16(u(self.0), u(o.0)) }))
        }

        #[inline(always)]
        pub fn min_u(self, o: Self) -> Self {
            Self(s(unsafe { vminq_u16(u(self.0), u(o.0)) }))
        }

        #[inline(always)]
        pub fn min(self, o: Self) -> Self {
            Self(unsafe { vminq_s16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            Self(unsafe { vmaxq_s16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn xor(self, o: Self) -> Self {
            Self(unsafe { veorq_s16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn load_u16(src: &[u16]) -> Self {
            Self(s(unsafe { vld1q_u16(src[..8].as_ptr()) }))
        }

        /// `s[..4]` into the low four lanes.
        #[inline(always)]
        pub fn load_u16_lo(src: &[u16]) -> Self {
            Self(s(unsafe {
                vcombine_u16(vld1_u16(src[..4].as_ptr()), vdup_n_u16(0))
            }))
        }

        #[inline(always)]
        pub fn store_u16(self, d: &mut [u16]) {
            unsafe { vst1q_u16(d[..8].as_mut_ptr(), u(self.0)) }
        }

        #[inline(always)]
        pub fn store_u16_lo(self, d: &mut [u16]) {
            unsafe { vst1_u16(d[..4].as_mut_ptr(), vget_low_u16(u(self.0))) }
        }

        #[inline(always)]
        pub fn and(self, o: Self) -> Self {
            Self(unsafe { vandq_s16(self.0, o.0) })
        }

        /// The high halves of the unsigned products.
        #[inline(always)]
        pub fn mulhi_u(self, o: Self) -> Self {
            let (a, b) = (u(self.0), u(o.0));
            let lo = unsafe { vmull_u16(vget_low_u16(a), vget_low_u16(b)) };
            let hi = unsafe { vmull_high_u16(a, b) };
            Self(s(unsafe {
                vuzp2q_u16(vreinterpretq_u16_u32(lo), vreinterpretq_u16_u32(hi))
            }))
        }

        /// The unsigned averages rounded up.
        #[inline(always)]
        pub fn avg_u(self, o: Self) -> Self {
            Self(s(unsafe { vrhaddq_u16(u(self.0), u(o.0)) }))
        }

        #[inline(always)]
        pub fn shr_s_by(self, n: u32) -> Self {
            Self(unsafe { vshlq_s16(self.0, vdupq_n_s16(-(n as i16))) })
        }

        #[inline(always)]
        pub fn eq(self, o: Self) -> Self {
            Self(s(unsafe { vceqq_s16(self.0, o.0) }))
        }

        /// The first set lane of a mask, 8 if none is.
        #[inline(always)]
        pub fn first_set(self) -> u32 {
            let bytes = unsafe { vget_lane_u64::<0>(vreinterpret_u64_u8(vmovn_u16(u(self.0)))) };
            bytes.trailing_zeros() / 8
        }

        /// The first set lane of the masks `self` then `o`, 16 if none is.
        #[inline(always)]
        pub fn first_set2(self, o: Self) -> u32 {
            let lanes = unsafe { vcombine_u8(vmovn_u16(u(self.0)), vmovn_u16(u(o.0))) };
            let nibbles = unsafe {
                vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(vreinterpretq_u16_u8(
                    lanes,
                ))))
            };
            nibbles.trailing_zeros() / 4
        }

        #[inline(always)]
        pub fn add_sat(self, o: Self) -> Self {
            Self(unsafe { vqaddq_s16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn sub_sat(self, o: Self) -> Self {
            Self(unsafe { vqsubq_s16(self.0, o.0) })
        }

        /// `(a * b + 0x4000) >> 15`: a product in Q15 with rounding.
        #[inline(always)]
        pub fn mulhrs(self, o: Self) -> Self {
            Self(unsafe { vqrdmulhq_s16(self.0, o.0) })
        }

        /// The butterfly rotation `((a * c1 - b * c2 + 2048) >> 12, (a * c2 + b * c1 + 2048) >> 12)`
        /// in 32 bits, each result saturated to `i16`.
        #[inline(always)]
        pub fn rot(self, o: Self, c1: i16, c2: i16) -> (Self, Self) {
            unsafe {
                let (a_lo, a_hi) = (vget_low_s16(self.0), vget_high_s16(self.0));
                let (b_lo, b_hi) = (vget_low_s16(o.0), vget_high_s16(o.0));
                let (c1, c2) = (vdup_n_s16(c1), vdup_n_s16(c2));
                let first = vcombine_s16(
                    vqrshrn_n_s32::<12>(vmlsl_s16(vmull_s16(a_lo, c1), b_lo, c2)),
                    vqrshrn_n_s32::<12>(vmlsl_s16(vmull_s16(a_hi, c1), b_hi, c2)),
                );
                let second = vcombine_s16(
                    vqrshrn_n_s32::<12>(vmlal_s16(vmull_s16(a_lo, c2), b_lo, c1)),
                    vqrshrn_n_s32::<12>(vmlal_s16(vmull_s16(a_hi, c2), b_hi, c1)),
                );
                (Self(first), Self(second))
            }
        }

        #[inline(always)]
        pub fn zip_lo16(self, o: Self) -> Self {
            Self(unsafe { vzip1q_s16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn zip_hi16(self, o: Self) -> Self {
            Self(unsafe { vzip2q_s16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn zip_lo32(self, o: Self) -> Self {
            Self(unsafe {
                vreinterpretq_s16_s32(vzip1q_s32(
                    vreinterpretq_s32_s16(self.0),
                    vreinterpretq_s32_s16(o.0),
                ))
            })
        }

        #[inline(always)]
        pub fn zip_hi32(self, o: Self) -> Self {
            Self(unsafe {
                vreinterpretq_s16_s32(vzip2q_s32(
                    vreinterpretq_s32_s16(self.0),
                    vreinterpretq_s32_s16(o.0),
                ))
            })
        }

        #[inline(always)]
        pub fn zip_lo64(self, o: Self) -> Self {
            Self(unsafe { vcombine_s16(vget_low_s16(self.0), vget_low_s16(o.0)) })
        }

        #[inline(always)]
        pub fn zip_hi64(self, o: Self) -> Self {
            Self(unsafe { vcombine_s16(vget_high_s16(self.0), vget_high_s16(o.0)) })
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod imp {
    use std::arch::wasm32::*;

    use super::I16x8;

    pub type Inner = v128;

    impl I16x8 {
        #[inline(always)]
        pub fn load(s: &[i16]) -> Self {
            let s = &s[..8];
            Self(i16x8(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]))
        }

        /// Lanes `lo[..4]` then `hi[..4]`: two rows of a 4-wide block.
        #[inline(always)]
        pub fn load_halves(lo: &[i16], hi: &[i16]) -> Self {
            let (lo, hi) = (&lo[..4], &hi[..4]);
            Self(i16x8(
                lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3],
            ))
        }

        #[inline(always)]
        pub fn splat(v: i16) -> Self {
            Self(i16x8_splat(v))
        }

        /// `v` in the low four lanes, the others unspecified.
        #[inline(always)]
        pub fn splat_lo(v: i16) -> Self {
            Self::splat(v)
        }

        /// Bytes `b[..8]` zero-extended.
        #[inline(always)]
        pub fn from_u8x8(b: &[u8]) -> Self {
            // SAFETY: 8 bytes are in bounds; the load takes unaligned pointers.
            Self(unsafe { u16x8_load_extend_u8x8(b[..8].as_ptr()) })
        }

        /// Little-endian lanes from `b[..16]`.
        #[inline(always)]
        pub fn from_bytes(b: &[u8]) -> Self {
            // SAFETY: 16 bytes are in bounds; the load takes unaligned pointers.
            Self(unsafe { v128_load(b[..16].as_ptr().cast()) })
        }

        #[inline(always)]
        pub fn store(self, d: &mut [i16]) {
            // SAFETY: 8 lanes are in bounds; the store takes unaligned pointers.
            unsafe { v128_store(d[..8].as_mut_ptr().cast(), self.0) }
        }

        /// The lanes saturated to `u8`.
        #[inline(always)]
        pub fn pack_u8(self) -> [u8; 8] {
            let mut out = [0; 16];
            // SAFETY: `out` holds 16 bytes.
            unsafe { v128_store(out.as_mut_ptr().cast(), u8x16_narrow_i16x8(self.0, self.0)) };
            out[..8].try_into().unwrap()
        }

        #[inline(always)]
        pub fn to_array(self) -> [i16; 8] {
            let v = self.0;
            [
                i16x8_extract_lane::<0>(v),
                i16x8_extract_lane::<1>(v),
                i16x8_extract_lane::<2>(v),
                i16x8_extract_lane::<3>(v),
                i16x8_extract_lane::<4>(v),
                i16x8_extract_lane::<5>(v),
                i16x8_extract_lane::<6>(v),
                i16x8_extract_lane::<7>(v),
            ]
        }

        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            Self(i16x8_add(self.0, o.0))
        }

        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            Self(i16x8_sub(self.0, o.0))
        }

        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            Self(i16x8_mul(self.0, o.0))
        }

        #[inline(always)]
        pub fn abs(self) -> Self {
            Self(i16x8_abs(self.0))
        }

        #[inline(always)]
        pub fn shr_u(self, n: u32) -> Self {
            Self(u16x8_shr(self.0, n))
        }

        #[inline(always)]
        pub fn shr_s<const N: i32>(self) -> Self {
            Self(i16x8_shr(self.0, N as u32))
        }

        #[inline(always)]
        pub fn sub_sat_u(self, o: Self) -> Self {
            Self(u16x8_sub_sat(self.0, o.0))
        }

        #[inline(always)]
        pub fn min_u(self, o: Self) -> Self {
            Self(u16x8_min(self.0, o.0))
        }

        #[inline(always)]
        pub fn min(self, o: Self) -> Self {
            Self(i16x8_min(self.0, o.0))
        }

        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            Self(i16x8_max(self.0, o.0))
        }

        #[inline(always)]
        pub fn xor(self, o: Self) -> Self {
            Self(v128_xor(self.0, o.0))
        }

        #[inline(always)]
        pub fn load_u16(s: &[u16]) -> Self {
            // SAFETY: 8 lanes are in bounds; the load takes unaligned pointers.
            Self(unsafe { v128_load(s[..8].as_ptr().cast()) })
        }

        /// `s[..4]` into the low four lanes.
        #[inline(always)]
        pub fn load_u16_lo(s: &[u16]) -> Self {
            // SAFETY: 4 lanes are in bounds; the load takes unaligned pointers.
            Self(unsafe { v128_load64_zero(s[..4].as_ptr().cast()) })
        }

        #[inline(always)]
        pub fn store_u16(self, d: &mut [u16]) {
            // SAFETY: 8 lanes are in bounds; the store takes unaligned pointers.
            unsafe { v128_store(d[..8].as_mut_ptr().cast(), self.0) }
        }

        #[inline(always)]
        pub fn store_u16_lo(self, d: &mut [u16]) {
            // SAFETY: 4 lanes are in bounds; the store takes unaligned pointers.
            unsafe { v128_store64_lane::<0>(self.0, d[..4].as_mut_ptr().cast()) }
        }

        #[inline(always)]
        pub fn and(self, o: Self) -> Self {
            Self(v128_and(self.0, o.0))
        }

        /// The high halves of the unsigned products.
        #[inline(always)]
        pub fn mulhi_u(self, o: Self) -> Self {
            let lo = u32x4_extmul_low_u16x8(self.0, o.0);
            let hi = u32x4_extmul_high_u16x8(self.0, o.0);
            Self(i16x8_shuffle::<1, 3, 5, 7, 9, 11, 13, 15>(lo, hi))
        }

        /// The unsigned averages rounded up.
        #[inline(always)]
        pub fn avg_u(self, o: Self) -> Self {
            Self(u16x8_avgr(self.0, o.0))
        }

        #[inline(always)]
        pub fn shr_s_by(self, n: u32) -> Self {
            Self(i16x8_shr(self.0, n))
        }

        #[inline(always)]
        pub fn eq(self, o: Self) -> Self {
            Self(i16x8_eq(self.0, o.0))
        }

        /// The first set lane of a mask, 8 if none is.
        #[inline(always)]
        pub fn first_set(self) -> u32 {
            (i16x8_bitmask(self.0) as u32 | 1 << 8).trailing_zeros()
        }

        /// The first set lane of the masks `self` then `o`, 16 if none is.
        #[inline(always)]
        pub fn first_set2(self, o: Self) -> u32 {
            (i8x16_bitmask(i8x16_narrow_i16x8(self.0, o.0)) as u32 | 1 << 16).trailing_zeros()
        }

        #[inline(always)]
        pub fn add_sat(self, o: Self) -> Self {
            Self(i16x8_add_sat(self.0, o.0))
        }

        #[inline(always)]
        pub fn sub_sat(self, o: Self) -> Self {
            Self(i16x8_sub_sat(self.0, o.0))
        }

        /// `(a * b + 0x4000) >> 15`: a product in Q15 with rounding.
        #[inline(always)]
        pub fn mulhrs(self, o: Self) -> Self {
            Self(i16x8_q15mulr_sat(self.0, o.0))
        }

        /// The butterfly rotation `((a * c1 - b * c2 + 2048) >> 12, (a * c2 + b * c1 + 2048) >> 12)`
        /// in 32 bits, each result saturated to `i16`.
        #[inline(always)]
        pub fn rot(self, o: Self, c1: i16, c2: i16) -> (Self, Self) {
            let (c1, c2, rnd) = (i16x8_splat(c1), i16x8_splat(c2), i32x4_splat(2048));
            let first = |lo: bool| {
                let (a, b) = if lo {
                    (
                        i32x4_extmul_low_i16x8(self.0, c1),
                        i32x4_extmul_low_i16x8(o.0, c2),
                    )
                } else {
                    (
                        i32x4_extmul_high_i16x8(self.0, c1),
                        i32x4_extmul_high_i16x8(o.0, c2),
                    )
                };
                i32x4_shr(i32x4_add(i32x4_sub(a, b), rnd), 12)
            };
            let second = |lo: bool| {
                let (a, b) = if lo {
                    (
                        i32x4_extmul_low_i16x8(self.0, c2),
                        i32x4_extmul_low_i16x8(o.0, c1),
                    )
                } else {
                    (
                        i32x4_extmul_high_i16x8(self.0, c2),
                        i32x4_extmul_high_i16x8(o.0, c1),
                    )
                };
                i32x4_shr(i32x4_add(i32x4_add(a, b), rnd), 12)
            };
            (
                Self(i16x8_narrow_i32x4(first(true), first(false))),
                Self(i16x8_narrow_i32x4(second(true), second(false))),
            )
        }

        #[inline(always)]
        pub fn zip_lo16(self, o: Self) -> Self {
            Self(i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(self.0, o.0))
        }

        #[inline(always)]
        pub fn zip_hi16(self, o: Self) -> Self {
            Self(i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(self.0, o.0))
        }

        #[inline(always)]
        pub fn zip_lo32(self, o: Self) -> Self {
            Self(i32x4_shuffle::<0, 4, 1, 5>(self.0, o.0))
        }

        #[inline(always)]
        pub fn zip_hi32(self, o: Self) -> Self {
            Self(i32x4_shuffle::<2, 6, 3, 7>(self.0, o.0))
        }

        #[inline(always)]
        pub fn zip_lo64(self, o: Self) -> Self {
            Self(i64x2_shuffle::<0, 2>(self.0, o.0))
        }

        #[inline(always)]
        pub fn zip_hi64(self, o: Self) -> Self {
            Self(i64x2_shuffle::<1, 3>(self.0, o.0))
        }
    }
}

#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
)))]
mod imp {
    use super::I16x8;

    pub type Inner = [i16; 8];

    impl I16x8 {
        #[inline(always)]
        fn zip(self, o: Self, f: impl Fn(i16, i16) -> i16) -> Self {
            Self(std::array::from_fn(|i| f(self.0[i], o.0[i])))
        }

        #[inline(always)]
        pub fn load(s: &[i16]) -> Self {
            Self(s[..8].try_into().unwrap())
        }

        /// Lanes `lo[..4]` then `hi[..4]`: two rows of a 4-wide block.
        #[inline(always)]
        pub fn load_halves(lo: &[i16], hi: &[i16]) -> Self {
            Self([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]])
        }

        #[inline(always)]
        pub fn splat(v: i16) -> Self {
            Self([v; 8])
        }

        /// `v` in the low four lanes, the others unspecified.
        #[inline(always)]
        pub fn splat_lo(v: i16) -> Self {
            Self::splat(v)
        }

        /// Bytes `b[..8]` zero-extended.
        #[inline(always)]
        pub fn from_u8x8(b: &[u8]) -> Self {
            Self(std::array::from_fn(|i| b[i] as i16))
        }

        /// Little-endian lanes from `b[..16]`.
        #[inline(always)]
        pub fn from_bytes(b: &[u8]) -> Self {
            Self(std::array::from_fn(|i| {
                i16::from_le_bytes([b[2 * i], b[2 * i + 1]])
            }))
        }

        #[inline(always)]
        pub fn store(self, d: &mut [i16]) {
            d[..8].copy_from_slice(&self.0);
        }

        /// The lanes saturated to `u8`.
        #[inline(always)]
        pub fn pack_u8(self) -> [u8; 8] {
            self.0.map(|v| v.clamp(0, 255) as u8)
        }

        #[inline(always)]
        pub fn to_array(self) -> [i16; 8] {
            self.0
        }

        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            self.zip(o, i16::wrapping_add)
        }

        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            self.zip(o, i16::wrapping_sub)
        }

        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            self.zip(o, i16::wrapping_mul)
        }

        #[inline(always)]
        pub fn abs(self) -> Self {
            self.zip(self, |a, _| a.wrapping_abs())
        }

        #[inline(always)]
        pub fn shr_u(self, n: u32) -> Self {
            self.zip(self, |a, _| ((a as u16) >> n) as i16)
        }

        #[inline(always)]
        pub fn shr_s<const N: i32>(self) -> Self {
            self.zip(self, |a, _| a >> N)
        }

        #[inline(always)]
        pub fn sub_sat_u(self, o: Self) -> Self {
            self.zip(o, |a, b| (a as u16).saturating_sub(b as u16) as i16)
        }

        #[inline(always)]
        pub fn min_u(self, o: Self) -> Self {
            self.zip(o, |a, b| (a as u16).min(b as u16) as i16)
        }

        #[inline(always)]
        pub fn min(self, o: Self) -> Self {
            self.zip(o, i16::min)
        }

        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            self.zip(o, i16::max)
        }

        #[inline(always)]
        pub fn xor(self, o: Self) -> Self {
            self.zip(o, |a, b| a ^ b)
        }

        #[inline(always)]
        pub fn load_u16(s: &[u16]) -> Self {
            Self(std::array::from_fn(|i| s[i] as i16))
        }

        /// `s[..4]` into the low four lanes.
        #[inline(always)]
        pub fn load_u16_lo(s: &[u16]) -> Self {
            Self(std::array::from_fn(|i| if i < 4 { s[i] as i16 } else { 0 }))
        }

        #[inline(always)]
        pub fn store_u16(self, d: &mut [u16]) {
            for (d, v) in d[..8].iter_mut().zip(self.0) {
                *d = v as u16;
            }
        }

        #[inline(always)]
        pub fn store_u16_lo(self, d: &mut [u16]) {
            for (d, v) in d[..4].iter_mut().zip(self.0) {
                *d = v as u16;
            }
        }

        #[inline(always)]
        pub fn and(self, o: Self) -> Self {
            self.zip(o, |a, b| a & b)
        }

        /// The high halves of the unsigned products.
        #[inline(always)]
        pub fn mulhi_u(self, o: Self) -> Self {
            self.zip(o, |a, b| ((a as u16 as u32 * b as u16 as u32) >> 16) as i16)
        }

        /// The unsigned averages rounded up.
        #[inline(always)]
        pub fn avg_u(self, o: Self) -> Self {
            self.zip(o, |a, b| {
                ((a as u16 as u32 + b as u16 as u32 + 1) >> 1) as i16
            })
        }

        #[inline(always)]
        pub fn shr_s_by(self, n: u32) -> Self {
            self.zip(self, |a, _| a >> n)
        }

        #[inline(always)]
        pub fn eq(self, o: Self) -> Self {
            self.zip(o, |a, b| -((a == b) as i16))
        }

        /// The first set lane of a mask, 8 if none is.
        #[inline(always)]
        pub fn first_set(self) -> u32 {
            self.0.iter().position(|&m| m != 0).unwrap_or(8) as u32
        }

        /// The first set lane of the masks `self` then `o`, 16 if none is.
        #[inline(always)]
        pub fn first_set2(self, o: Self) -> u32 {
            self.0
                .iter()
                .chain(&o.0)
                .position(|&m| m != 0)
                .unwrap_or(16) as u32
        }

        #[inline(always)]
        pub fn add_sat(self, o: Self) -> Self {
            self.zip(o, i16::saturating_add)
        }

        #[inline(always)]
        pub fn sub_sat(self, o: Self) -> Self {
            self.zip(o, i16::saturating_sub)
        }

        /// `(a * b + 0x4000) >> 15`: a product in Q15 with rounding.
        #[inline(always)]
        pub fn mulhrs(self, o: Self) -> Self {
            self.zip(o, |a, b| ((a as i32 * b as i32 + 0x4000) >> 15) as i16)
        }

        /// The butterfly rotation `((a * c1 - b * c2 + 2048) >> 12, (a * c2 + b * c1 + 2048) >> 12)`
        /// in 32 bits, each result saturated to `i16`.
        #[inline(always)]
        pub fn rot(self, o: Self, c1: i16, c2: i16) -> (Self, Self) {
            let (c1, c2) = (c1 as i32, c2 as i32);
            let sat = |v: i32| v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            (
                self.zip(o, |a, b| sat(a as i32 * c1 - b as i32 * c2 + 2048 >> 12)),
                self.zip(o, |a, b| sat(a as i32 * c2 + b as i32 * c1 + 2048 >> 12)),
            )
        }

        #[inline(always)]
        pub fn zip_lo16(self, o: Self) -> Self {
            Self(std::array::from_fn(|i| [self.0, o.0][i & 1][i >> 1]))
        }

        #[inline(always)]
        pub fn zip_hi16(self, o: Self) -> Self {
            Self(std::array::from_fn(|i| [self.0, o.0][i & 1][4 + (i >> 1)]))
        }

        #[inline(always)]
        pub fn zip_lo32(self, o: Self) -> Self {
            Self(std::array::from_fn(|i| {
                [self.0, o.0][i >> 1 & 1][(i >> 2) * 2 + (i & 1)]
            }))
        }

        #[inline(always)]
        pub fn zip_hi32(self, o: Self) -> Self {
            Self(std::array::from_fn(|i| {
                [self.0, o.0][i >> 1 & 1][4 + (i >> 2) * 2 + (i & 1)]
            }))
        }

        #[inline(always)]
        pub fn zip_lo64(self, o: Self) -> Self {
            Self(std::array::from_fn(|i| [self.0, o.0][i >> 2][i & 3]))
        }

        #[inline(always)]
        pub fn zip_hi64(self, o: Self) -> Self {
            Self(std::array::from_fn(|i| [self.0, o.0][i >> 2][4 + (i & 3)]))
        }
    }
}
