//! Eight `i16` lanes on SSE4.1, NEON and wasm `simd128`, with a scalar stand-in elsewhere: the
//! one source the Rust SIMD kernels are written in. Each backend maps an operation to one or two
//! instructions; the kernels never name a target.
//!
//! On x86 every method needs SSE4.1. They are `#[inline(always)]` so that they compile into a
//! kernel entered through `#[target_feature(enable = "sse4.1")]`, which is only chosen once the
//! CPU flags hold SSE41.

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

    // SAFETY (every `unsafe` here): SSE4.1 holds where these are compiled in (see the module
    // docs), and loads and stores go through bounds-checked slices of 8 or 4 lanes.
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
    }
}
