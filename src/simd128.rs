//! Eight `i16` lanes on SSE4.1, NEON and wasm `simd128`, with a scalar stand-in elsewhere: the
//! one source the Rust SIMD kernels are written in. Each backend maps an operation to one or two
//! instructions; the kernels never name a target. [`I32x4`] holds the sums that outgrow 16 bits.
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

/// Four `i32` lanes, for the sums that outgrow [`I16x8`]: widened from it and packed back.
#[derive(Clone, Copy)]
pub struct I32x4(imp::Inner32);

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

/// The lane operations the transforms take, for [`I16x8`] on every target and, on x86_64 with
/// AVX2, [`I16x16`]: one kernel source for both widths.
pub trait Lanes: Copy {
    fn splat(v: i16) -> Self;
    fn add_sat(self, o: Self) -> Self;
    fn sub_sat(self, o: Self) -> Self;
    /// `(a * b + 0x4000) >> 15`: a product in Q15 with rounding.
    fn mulhrs(self, o: Self) -> Self;
    /// The unsigned averages rounded up.
    fn avg_u(self, o: Self) -> Self;
    fn shr_s<const N: i32>(self) -> Self;
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn abs(self) -> Self;
    fn min(self, o: Self) -> Self;
    fn max(self, o: Self) -> Self;
    /// `a` where `m` is set, `b` elsewhere; `m`'s lanes are all ones or all zeros.
    fn select(m: Self, a: Self, b: Self) -> Self;
    fn and(self, o: Self) -> Self;
    fn eq(self, o: Self) -> Self;
    /// Whether any lane is set.
    fn any(self) -> bool;
    /// `(a * k1[0] + b * k1[1] + 2048) >> 12` and the same with `k2`, in 32 bits, each
    /// saturated to `i16`.
    fn madd2(self, o: Self, k1: [i16; 2], k2: [i16; 2]) -> (Self, Self);
    /// `(a * k[0] + b * k[1] + c * k[2] + d * k[3] + 2048) >> 12` in 32 bits, saturated to `i16`.
    fn madd4(self, b: Self, c: Self, d: Self, k: [i16; 4]) -> Self;
    /// The interleaves, within each 128-bit half.
    fn zip_lo16(self, o: Self) -> Self;
    fn zip_hi16(self, o: Self) -> Self;
    fn zip_lo32(self, o: Self) -> Self;
    fn zip_hi32(self, o: Self) -> Self;
    fn zip_lo64(self, o: Self) -> Self;
    fn zip_hi64(self, o: Self) -> Self;
}

macro_rules! forward_lanes {
    ($t:ty) => {
        impl Lanes for $t {
            #[inline(always)]
            fn splat(v: i16) -> Self {
                <$t>::splat(v)
            }
            #[inline(always)]
            fn add_sat(self, o: Self) -> Self {
                <$t>::add_sat(self, o)
            }
            #[inline(always)]
            fn sub_sat(self, o: Self) -> Self {
                <$t>::sub_sat(self, o)
            }
            #[inline(always)]
            fn mulhrs(self, o: Self) -> Self {
                <$t>::mulhrs(self, o)
            }
            #[inline(always)]
            fn avg_u(self, o: Self) -> Self {
                <$t>::avg_u(self, o)
            }
            #[inline(always)]
            fn shr_s<const N: i32>(self) -> Self {
                <$t>::shr_s::<N>(self)
            }
            #[inline(always)]
            fn add(self, o: Self) -> Self {
                <$t>::add(self, o)
            }
            #[inline(always)]
            fn any(self) -> bool {
                <$t>::any(self)
            }
            #[inline(always)]
            fn sub(self, o: Self) -> Self {
                <$t>::sub(self, o)
            }
            #[inline(always)]
            fn abs(self) -> Self {
                <$t>::abs(self)
            }
            #[inline(always)]
            fn min(self, o: Self) -> Self {
                <$t>::min(self, o)
            }
            #[inline(always)]
            fn max(self, o: Self) -> Self {
                <$t>::max(self, o)
            }
            #[inline(always)]
            fn select(m: Self, a: Self, b: Self) -> Self {
                <$t>::select(m, a, b)
            }
            #[inline(always)]
            fn and(self, o: Self) -> Self {
                <$t>::and(self, o)
            }
            #[inline(always)]
            fn eq(self, o: Self) -> Self {
                <$t>::eq(self, o)
            }
            #[inline(always)]
            fn madd2(self, o: Self, k1: [i16; 2], k2: [i16; 2]) -> (Self, Self) {
                <$t>::madd2(self, o, k1, k2)
            }
            #[inline(always)]
            fn madd4(self, b: Self, c: Self, d: Self, k: [i16; 4]) -> Self {
                <$t>::madd4(self, b, c, d, k)
            }
            #[inline(always)]
            fn zip_lo16(self, o: Self) -> Self {
                <$t>::zip_lo16(self, o)
            }
            #[inline(always)]
            fn zip_hi16(self, o: Self) -> Self {
                <$t>::zip_hi16(self, o)
            }
            #[inline(always)]
            fn zip_lo32(self, o: Self) -> Self {
                <$t>::zip_lo32(self, o)
            }
            #[inline(always)]
            fn zip_hi32(self, o: Self) -> Self {
                <$t>::zip_hi32(self, o)
            }
            #[inline(always)]
            fn zip_lo64(self, o: Self) -> Self {
                <$t>::zip_lo64(self, o)
            }
            #[inline(always)]
            fn zip_hi64(self, o: Self) -> Self {
                <$t>::zip_hi64(self, o)
            }
        }
    };
}

forward_lanes!(I16x8);
#[cfg(target_arch = "x86_64")]
forward_lanes!(I16x16);

/// `v[i]` lane `j` becomes `v[j]` lane `i` within each 128-bit half: three rounds of interleaves,
/// 24 shuffles; [`I16x16`] turns two 8x8 blocks at once.
#[inline(always)]
pub fn transpose8x8<V: Lanes>(v: &mut [V; 8]) {
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

#[cfg(target_arch = "x86_64")]
pub use wide::I16x16;

/// Sixteen `i16` lanes in one AVX2 register, entered only through a
/// `#[target_feature(enable = "avx2")]` kernel chosen once the CPU flags hold AVX2.
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)]
mod wide {
    use std::arch::x86_64::*;

    use super::I16x8;

    #[derive(Clone, Copy)]
    pub struct I16x16(__m256i);

    // SAFETY (every `unsafe` here): AVX2 holds where these are compiled in (see the type's docs),
    // and loads and stores go through bounds-checked slices of 16 lanes.
    impl I16x16 {
        #[inline(always)]
        pub fn load(s: &[i16]) -> Self {
            Self(unsafe { _mm256_loadu_si256(s[..16].as_ptr().cast()) })
        }

        #[inline(always)]
        pub fn store(self, d: &mut [i16]) {
            unsafe { _mm256_storeu_si256(d[..16].as_mut_ptr().cast(), self.0) }
        }

        #[inline(always)]
        pub fn from_halves(lo: I16x8, hi: I16x8) -> Self {
            Self(unsafe { _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo.0), hi.0) })
        }

        #[inline(always)]
        pub fn lo(self) -> I16x8 {
            I16x8(unsafe { _mm256_castsi256_si128(self.0) })
        }

        #[inline(always)]
        pub fn hi(self) -> I16x8 {
            I16x8(unsafe { _mm256_extracti128_si256::<1>(self.0) })
        }

        /// Bytes `b[..16]` zero-extended.
        #[inline(always)]
        pub fn from_u8x16(b: &[u8]) -> Self {
            Self(unsafe { _mm256_cvtepu8_epi16(_mm_loadu_si128(b[..16].as_ptr().cast())) })
        }

        /// The lanes saturated to `u8`.
        #[inline(always)]
        pub fn pack_u8(self) -> [u8; 16] {
            let mut out = [0; 16];
            unsafe {
                let v = _mm_packus_epi16(self.lo().0, self.hi().0);
                _mm_storeu_si128(out.as_mut_ptr().cast(), v);
            }
            out
        }

        #[inline(always)]
        pub fn splat(v: i16) -> Self {
            Self(unsafe { _mm256_set1_epi16(v) })
        }

        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            Self(unsafe { _mm256_add_epi16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            Self(unsafe { _mm256_sub_epi16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn abs(self) -> Self {
            Self(unsafe { _mm256_abs_epi16(self.0) })
        }

        #[inline(always)]
        pub fn min(self, o: Self) -> Self {
            Self(unsafe { _mm256_min_epi16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            Self(unsafe { _mm256_max_epi16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn select(m: Self, a: Self, b: Self) -> Self {
            Self(unsafe { _mm256_blendv_epi8(b.0, a.0, m.0) })
        }

        #[inline(always)]
        pub fn add_sat(self, o: Self) -> Self {
            Self(unsafe { _mm256_adds_epi16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn sub_sat(self, o: Self) -> Self {
            Self(unsafe { _mm256_subs_epi16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn mulhrs(self, o: Self) -> Self {
            Self(unsafe { _mm256_mulhrs_epi16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn avg_u(self, o: Self) -> Self {
            Self(unsafe { _mm256_avg_epu16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn shr_s<const N: i32>(self) -> Self {
            Self(unsafe { _mm256_srai_epi16::<N>(self.0) })
        }

        #[inline(always)]
        pub fn and(self, o: Self) -> Self {
            Self(unsafe { _mm256_and_si256(self.0, o.0) })
        }

        #[inline(always)]
        pub fn or(self, o: Self) -> Self {
            Self(unsafe { _mm256_or_si256(self.0, o.0) })
        }

        #[inline(always)]
        pub fn any(self) -> bool {
            unsafe { _mm256_movemask_epi8(self.0) != 0 }
        }

        #[inline(always)]
        pub fn eq(self, o: Self) -> Self {
            Self(unsafe { _mm256_cmpeq_epi16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn madd2(self, o: Self, k1: [i16; 2], k2: [i16; 2]) -> (Self, Self) {
            unsafe {
                let k1 = _mm256_set1_epi32(k1[0] as u16 as i32 | (k1[1] as i32) << 16);
                let k2 = _mm256_set1_epi32(k2[0] as u16 as i32 | (k2[1] as i32) << 16);
                let rnd = _mm256_set1_epi32(2048);
                let lo = _mm256_unpacklo_epi16(self.0, o.0);
                let hi = _mm256_unpackhi_epi16(self.0, o.0);
                (
                    Self(_mm256_packs_epi32(
                        _mm256_srai_epi32::<12>(_mm256_add_epi32(_mm256_madd_epi16(lo, k1), rnd)),
                        _mm256_srai_epi32::<12>(_mm256_add_epi32(_mm256_madd_epi16(hi, k1), rnd)),
                    )),
                    Self(_mm256_packs_epi32(
                        _mm256_srai_epi32::<12>(_mm256_add_epi32(_mm256_madd_epi16(lo, k2), rnd)),
                        _mm256_srai_epi32::<12>(_mm256_add_epi32(_mm256_madd_epi16(hi, k2), rnd)),
                    )),
                )
            }
        }

        #[inline(always)]
        pub fn madd4(self, b: Self, c: Self, d: Self, k: [i16; 4]) -> Self {
            unsafe {
                let k01 = _mm256_set1_epi32(k[0] as u16 as i32 | (k[1] as i32) << 16);
                let k23 = _mm256_set1_epi32(k[2] as u16 as i32 | (k[3] as i32) << 16);
                let rnd = _mm256_set1_epi32(2048);
                let lo = _mm256_add_epi32(
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(self.0, b.0), k01),
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(c.0, d.0), k23),
                );
                let hi = _mm256_add_epi32(
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(self.0, b.0), k01),
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(c.0, d.0), k23),
                );
                Self(_mm256_packs_epi32(
                    _mm256_srai_epi32::<12>(_mm256_add_epi32(lo, rnd)),
                    _mm256_srai_epi32::<12>(_mm256_add_epi32(hi, rnd)),
                ))
            }
        }

        #[inline(always)]
        pub fn zip_lo16(self, o: Self) -> Self {
            Self(unsafe { _mm256_unpacklo_epi16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn zip_hi16(self, o: Self) -> Self {
            Self(unsafe { _mm256_unpackhi_epi16(self.0, o.0) })
        }

        #[inline(always)]
        pub fn zip_lo32(self, o: Self) -> Self {
            Self(unsafe { _mm256_unpacklo_epi32(self.0, o.0) })
        }

        #[inline(always)]
        pub fn zip_hi32(self, o: Self) -> Self {
            Self(unsafe { _mm256_unpackhi_epi32(self.0, o.0) })
        }

        #[inline(always)]
        pub fn zip_lo64(self, o: Self) -> Self {
            Self(unsafe { _mm256_unpacklo_epi64(self.0, o.0) })
        }

        #[inline(always)]
        pub fn zip_hi64(self, o: Self) -> Self {
            Self(unsafe { _mm256_unpackhi_epi64(self.0, o.0) })
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

    use super::{I16x8, I32x4};

    pub type Inner = __m128i;

    pub type Inner32 = __m128i;

    // SAFETY (every `unsafe` here): SSE4.1 holds where these are compiled in (see the module
    // docs), and loads and stores go through bounds-checked slices of 4 lanes.
    impl I32x4 {
        #[inline(always)]
        pub fn load(s: &[i32]) -> Self {
            Self(unsafe { _mm_loadu_si128(s[..4].as_ptr().cast()) })
        }

        #[inline(always)]
        pub fn store(self, d: &mut [i32]) {
            unsafe { _mm_storeu_si128(d[..4].as_mut_ptr().cast(), self.0) }
        }

        #[inline(always)]
        pub fn splat(v: i32) -> Self {
            Self(unsafe { _mm_set1_epi32(v) })
        }

        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            Self(unsafe { _mm_add_epi32(self.0, o.0) })
        }

        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            Self(unsafe { _mm_sub_epi32(self.0, o.0) })
        }

        /// The low 32 bits of the products.
        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            Self(unsafe { _mm_mullo_epi32(self.0, o.0) })
        }

        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            Self(unsafe { _mm_max_epi32(self.0, o.0) })
        }

        #[inline(always)]
        pub fn shl<const N: i32>(self) -> Self {
            Self(unsafe { _mm_slli_epi32::<N>(self.0) })
        }

        #[inline(always)]
        pub fn shr_s<const N: i32>(self) -> Self {
            Self(unsafe { _mm_srai_epi32::<N>(self.0) })
        }

        #[inline(always)]
        pub fn shr_u<const N: i32>(self) -> Self {
            Self(unsafe { _mm_srli_epi32::<N>(self.0) })
        }

        /// `lo` then `hi` saturated to `i16`.
        #[inline(always)]
        pub fn pack(lo: Self, hi: Self) -> I16x8 {
            I16x8(unsafe { _mm_packs_epi32(lo.0, hi.0) })
        }

        #[inline(always)]
        pub fn min(self, o: Self) -> Self {
            Self(unsafe { _mm_min_epi32(self.0, o.0) })
        }

        #[inline(always)]
        pub fn or(self, o: Self) -> Self {
            Self(unsafe { _mm_or_si128(self.0, o.0) })
        }

        /// Whether any lane is nonzero.
        #[inline(always)]
        pub fn any(self) -> bool {
            unsafe { _mm_testz_si128(self.0, self.0) == 0 }
        }

        /// `(x + (1 << N >> 1)) >> N`.
        #[inline(always)]
        pub fn round_shr<const N: i32>(self) -> Self {
            unsafe {
                Self(_mm_srai_epi32::<N>(_mm_add_epi32(
                    self.0,
                    _mm_set1_epi32(1 << N >> 1),
                )))
            }
        }

        /// `(x * k + 2048) >> 12`, for products within 32 bits.
        #[inline(always)]
        pub fn mul_q12(self, k: i32) -> Self {
            unsafe {
                let p = _mm_mullo_epi32(self.0, _mm_set1_epi32(k));
                Self(_mm_srai_epi32::<12>(_mm_add_epi32(p, _mm_set1_epi32(2048))))
            }
        }

        /// `lo` then `hi` shifted with rounding by `N`, saturated to `i16`.
        #[inline(always)]
        pub fn pack_round_shr<const N: i32>(lo: Self, hi: Self) -> I16x8 {
            Self::pack(lo.round_shr::<N>(), hi.round_shr::<N>())
        }
    }

    impl I16x8 {
        /// The lanes of `self` and `o` as `u16`, summed into 32 bits: low four lanes, then high.
        #[inline(always)]
        pub fn addl_u(self, o: Self) -> [I32x4; 2] {
            unsafe {
                let z = _mm_setzero_si128();
                let (a, b) = (self.0, o.0);
                [
                    I32x4(_mm_add_epi32(
                        _mm_unpacklo_epi16(a, z),
                        _mm_unpacklo_epi16(b, z),
                    )),
                    I32x4(_mm_add_epi32(
                        _mm_unpackhi_epi16(a, z),
                        _mm_unpackhi_epi16(b, z),
                    )),
                ]
            }
        }

        /// The lanes as `u16` added onto `acc`'s low four, then high four.
        #[inline(always)]
        pub fn addw_u(self, acc: [I32x4; 2]) -> [I32x4; 2] {
            unsafe {
                let z = _mm_setzero_si128();
                [
                    I32x4(_mm_add_epi32(acc[0].0, _mm_unpacklo_epi16(self.0, z))),
                    I32x4(_mm_add_epi32(acc[1].0, _mm_unpackhi_epi16(self.0, z))),
                ]
            }
        }

        /// The full products, low four lanes then high four.
        #[inline(always)]
        pub fn mul_wide(self, o: Self) -> [I32x4; 2] {
            unsafe {
                let (lo, hi) = (_mm_mullo_epi16(self.0, o.0), _mm_mulhi_epi16(self.0, o.0));
                [
                    I32x4(_mm_unpacklo_epi16(lo, hi)),
                    I32x4(_mm_unpackhi_epi16(lo, hi)),
                ]
            }
        }

        #[inline(always)]
        pub fn gt(self, o: Self) -> Self {
            Self(unsafe { _mm_cmpgt_epi16(self.0, o.0) })
        }

        /// `t[lane]` for lanes below 64, 0 for those up to 255.
        #[inline(always)]
        pub fn lookup64(self, t: &[u8; 64]) -> Self {
            unsafe {
                let i = _mm_packus_epi16(self.0, self.0);
                let mut r = _mm_setzero_si128();
                for k in 0..4 {
                    let t = _mm_loadu_si128(t[16 * k..].as_ptr().cast());
                    // Indices outside this 16 saturate to the top bit, which zeroes them.
                    let ik = _mm_adds_epu8(
                        _mm_sub_epi8(i, _mm_set1_epi8(16 * k as i8)),
                        _mm_set1_epi8(0x70),
                    );
                    r = _mm_or_si128(r, _mm_shuffle_epi8(t, ik));
                }
                Self(_mm_cvtepu8_epi16(r))
            }
        }
    }

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

        #[inline(always)]
        pub fn or(self, o: Self) -> Self {
            Self(unsafe { _mm_or_si128(self.0, o.0) })
        }

        /// Whether any lane is set.
        #[inline(always)]
        pub fn any(self) -> bool {
            unsafe { _mm_movemask_epi8(self.0) != 0 }
        }

        /// The first set lane of the masks `self` then `o`, 16 if none is.
        #[inline(always)]
        pub fn first_set2(self, o: Self) -> u32 {
            (unsafe { _mm_movemask_epi8(_mm_packs_epi16(self.0, o.0)) } as u32 | 1 << 16)
                .trailing_zeros()
        }

        /// `self` where `m` is set, `o` elsewhere; `m`'s lanes are all ones or all zeros.
        #[inline(always)]
        pub fn select(m: Self, a: Self, b: Self) -> Self {
            Self(unsafe { _mm_blendv_epi8(b.0, a.0, m.0) })
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

        /// Lanes `s[..4]`, the high four zero.
        #[inline(always)]
        pub fn load_lo(s: &[i16]) -> Self {
            Self(unsafe { _mm_loadl_epi64(s[..4].as_ptr().cast()) })
        }

        /// `(v[0] * k[0] + .. + v[7] * k[7] + rnd) >> S` in 32 bits, saturated to `i16`.
        #[inline(always)]
        pub fn madd8<const S: i32>(v: [Self; 8], k: [i16; 8], rnd: i32) -> Self {
            unsafe {
                let mut lo = _mm_set1_epi32(rnd);
                let mut hi = lo;
                for i in (0..8).step_by(2) {
                    let k = _mm_set1_epi32(k[i] as u16 as i32 | (k[i + 1] as i32) << 16);
                    let (a, b) = (v[i].0, v[i + 1].0);
                    lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), k));
                    hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), k));
                }
                Self(_mm_packs_epi32(_mm_srai_epi32::<S>(lo), _mm_srai_epi32::<S>(hi)))
            }
        }

        /// `(a * k[0] + b * k[1] + c * k[2] + d * k[3] + 2048) >> 12` in 32 bits, saturated to `i16`.
        #[inline(always)]
        pub fn madd4(self, b: Self, c: Self, d: Self, k: [i16; 4]) -> Self {
            unsafe {
                let k01 = _mm_set1_epi32(k[0] as u16 as i32 | (k[1] as i32) << 16);
                let k23 = _mm_set1_epi32(k[2] as u16 as i32 | (k[3] as i32) << 16);
                let rnd = _mm_set1_epi32(2048);
                let lo = _mm_add_epi32(
                    _mm_madd_epi16(_mm_unpacklo_epi16(self.0, b.0), k01),
                    _mm_madd_epi16(_mm_unpacklo_epi16(c.0, d.0), k23),
                );
                let hi = _mm_add_epi32(
                    _mm_madd_epi16(_mm_unpackhi_epi16(self.0, b.0), k01),
                    _mm_madd_epi16(_mm_unpackhi_epi16(c.0, d.0), k23),
                );
                Self(_mm_packs_epi32(
                    _mm_srai_epi32::<12>(_mm_add_epi32(lo, rnd)),
                    _mm_srai_epi32::<12>(_mm_add_epi32(hi, rnd)),
                ))
            }
        }

        /// `(a * k1[0] + b * k1[1] + 2048) >> 12` and the same with `k2`, in 32 bits, each
        /// saturated to `i16`.
        #[inline(always)]
        pub fn madd2(self, o: Self, k1: [i16; 2], k2: [i16; 2]) -> (Self, Self) {
            // No closures: one would not inherit the kernel's target features.
            unsafe {
                let k1 = _mm_set1_epi32(k1[0] as u16 as i32 | (k1[1] as i32) << 16);
                let k2 = _mm_set1_epi32(k2[0] as u16 as i32 | (k2[1] as i32) << 16);
                let rnd = _mm_set1_epi32(2048);
                let lo = _mm_unpacklo_epi16(self.0, o.0);
                let hi = _mm_unpackhi_epi16(self.0, o.0);
                (
                    Self(_mm_packs_epi32(
                        _mm_srai_epi32::<12>(_mm_add_epi32(_mm_madd_epi16(lo, k1), rnd)),
                        _mm_srai_epi32::<12>(_mm_add_epi32(_mm_madd_epi16(hi, k1), rnd)),
                    )),
                    Self(_mm_packs_epi32(
                        _mm_srai_epi32::<12>(_mm_add_epi32(_mm_madd_epi16(lo, k2), rnd)),
                        _mm_srai_epi32::<12>(_mm_add_epi32(_mm_madd_epi16(hi, k2), rnd)),
                    )),
                )
            }
        }

        #[inline(always)]
        pub fn zip_lo16(self, o: Self) -> Self {
            Self(unsafe { _mm_unpacklo_epi16(self.0, o.0) })
        }

        /// The even lanes of `self` then `o`, and the odd ones.
        #[inline(always)]
        pub fn unzip(self, o: Self) -> (Self, Self) {
            unsafe {
                let m = _mm_set1_epi32(0xffff);
                let even = _mm_packus_epi32(_mm_and_si128(self.0, m), _mm_and_si128(o.0, m));
                let odd = _mm_packus_epi32(_mm_srli_epi32::<16>(self.0), _mm_srli_epi32::<16>(o.0));
                (Self(even), Self(odd))
            }
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

    use super::{I16x8, I32x4};

    pub type Inner = int16x8_t;

    pub type Inner32 = int32x4_t;

    // SAFETY (every `unsafe` here): NEON is part of aarch64, and loads and stores go through
    // bounds-checked slices of 4 lanes.
    impl I32x4 {
        #[inline(always)]
        pub fn load(s: &[i32]) -> Self {
            Self(unsafe { vld1q_s32(s[..4].as_ptr()) })
        }

        #[inline(always)]
        pub fn store(self, d: &mut [i32]) {
            unsafe { vst1q_s32(d[..4].as_mut_ptr(), self.0) }
        }

        #[inline(always)]
        pub fn splat(v: i32) -> Self {
            Self(unsafe { vdupq_n_s32(v) })
        }

        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            Self(unsafe { vaddq_s32(self.0, o.0) })
        }

        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            Self(unsafe { vsubq_s32(self.0, o.0) })
        }

        /// The low 32 bits of the products.
        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            Self(unsafe { vmulq_s32(self.0, o.0) })
        }

        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            Self(unsafe { vmaxq_s32(self.0, o.0) })
        }

        #[inline(always)]
        pub fn shl<const N: i32>(self) -> Self {
            Self(unsafe { vshlq_n_s32::<N>(self.0) })
        }

        #[inline(always)]
        pub fn shr_s<const N: i32>(self) -> Self {
            Self(unsafe { vshrq_n_s32::<N>(self.0) })
        }

        #[inline(always)]
        pub fn shr_u<const N: i32>(self) -> Self {
            Self(unsafe { vreinterpretq_s32_u32(vshrq_n_u32::<N>(vreinterpretq_u32_s32(self.0))) })
        }

        /// `lo` then `hi` saturated to `i16`.
        #[inline(always)]
        pub fn pack(lo: Self, hi: Self) -> I16x8 {
            I16x8(unsafe { vqmovn_high_s32(vqmovn_s32(lo.0), hi.0) })
        }

        #[inline(always)]
        pub fn min(self, o: Self) -> Self {
            Self(unsafe { vminq_s32(self.0, o.0) })
        }

        #[inline(always)]
        pub fn or(self, o: Self) -> Self {
            Self(unsafe { vorrq_s32(self.0, o.0) })
        }

        /// Whether any lane is nonzero.
        #[inline(always)]
        pub fn any(self) -> bool {
            unsafe { vmaxvq_u32(vreinterpretq_u32_s32(self.0)) != 0 }
        }

        /// `(x + (1 << N >> 1)) >> N`.
        #[inline(always)]
        pub fn round_shr<const N: i32>(self) -> Self {
            Self(unsafe { vrshrq_n_s32::<N>(self.0) })
        }

        /// `(x * k + 2048) >> 12`, for `|k| < 4096`: one rounding doubling multiply-high by
        /// `k << 19`, exact as its product is wider than 32 bits.
        #[inline(always)]
        pub fn mul_q12(self, k: i32) -> Self {
            Self(unsafe { vqrdmulhq_n_s32(self.0, k << 19) })
        }

        /// `lo` then `hi` shifted with rounding by `N`, saturated to `i16`.
        #[inline(always)]
        pub fn pack_round_shr<const N: i32>(lo: Self, hi: Self) -> I16x8 {
            I16x8(unsafe { vqrshrn_high_n_s32::<N>(vqrshrn_n_s32::<N>(lo.0), hi.0) })
        }
    }

    impl I16x8 {
        /// The lanes of `self` and `o` as `u16`, summed into 32 bits: low four lanes, then high.
        #[inline(always)]
        pub fn addl_u(self, o: Self) -> [I32x4; 2] {
            unsafe {
                let (a, b) = (u(self.0), u(o.0));
                [
                    I32x4(vreinterpretq_s32_u32(vaddl_u16(
                        vget_low_u16(a),
                        vget_low_u16(b),
                    ))),
                    I32x4(vreinterpretq_s32_u32(vaddl_high_u16(a, b))),
                ]
            }
        }

        /// The lanes as `u16` added onto `acc`'s low four, then high four.
        #[inline(always)]
        pub fn addw_u(self, acc: [I32x4; 2]) -> [I32x4; 2] {
            unsafe {
                let a = u(self.0);
                let lo = vaddw_u16(vreinterpretq_u32_s32(acc[0].0), vget_low_u16(a));
                let hi = vaddw_high_u16(vreinterpretq_u32_s32(acc[1].0), a);
                [
                    I32x4(vreinterpretq_s32_u32(lo)),
                    I32x4(vreinterpretq_s32_u32(hi)),
                ]
            }
        }

        /// The full products, low four lanes then high four.
        #[inline(always)]
        pub fn mul_wide(self, o: Self) -> [I32x4; 2] {
            unsafe {
                [
                    I32x4(vmull_s16(vget_low_s16(self.0), vget_low_s16(o.0))),
                    I32x4(vmull_high_s16(self.0, o.0)),
                ]
            }
        }

        #[inline(always)]
        pub fn gt(self, o: Self) -> Self {
            Self(s(unsafe { vcgtq_s16(self.0, o.0) }))
        }

        /// `t[lane]` for lanes below 64, 0 for those up to 255.
        #[inline(always)]
        pub fn lookup64(self, t: &[u8; 64]) -> Self {
            unsafe {
                let t = vld1q_u8_x4(t.as_ptr());
                Self(s(vmovl_u8(vqtbl4_u8(t, vqmovun_s16(self.0)))))
            }
        }
    }

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

        /// The high halves of the unsigned products. Unused on aarch64, where msac scales with
        /// a doubling multiply instead.
        #[inline(always)]
        #[allow(dead_code)]
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

        /// The first set lane of a mask among the low four, 4 if none is.
        #[inline(always)]
        pub fn first_set_lo(self) -> u32 {
            let lanes = unsafe { vgetq_lane_u64::<0>(vreinterpretq_u64_s16(self.0)) };
            lanes.trailing_zeros() / 16
        }

        /// `(2 * self * o) >> 16`, saturated.
        #[inline(always)]
        pub fn qdmulh(self, o: Self) -> Self {
            Self(unsafe { vqdmulhq_s16(self.0, o.0) })
        }

        /// The lanes as one little-endian integer.
        #[inline(always)]
        pub fn to_u128(self) -> u128 {
            let v = unsafe { vreinterpretq_u64_s16(self.0) };
            let (lo, hi) = unsafe { (vgetq_lane_u64::<0>(v), vgetq_lane_u64::<1>(v)) };
            lo as u128 | (hi as u128) << 64
        }

        #[inline(always)]
        pub fn or(self, o: Self) -> Self {
            Self(unsafe { vorrq_s16(self.0, o.0) })
        }

        /// Whether any lane is set.
        #[inline(always)]
        pub fn any(self) -> bool {
            unsafe { vmaxvq_u16(u(self.0)) != 0 }
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

        /// `a` where `m` is set, `b` elsewhere; `m`'s lanes are all ones or all zeros.
        #[inline(always)]
        pub fn select(m: Self, a: Self, b: Self) -> Self {
            Self(unsafe { vbslq_s16(u(m.0), a.0, b.0) })
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

        /// Lanes `s[..4]`, the high four zero.
        #[inline(always)]
        pub fn load_lo(s: &[i16]) -> Self {
            Self(unsafe { vcombine_s16(vld1_s16(s[..4].as_ptr()), vdup_n_s16(0)) })
        }

        /// `(v[0] * k[0] + .. + v[7] * k[7] + rnd) >> S` in 32 bits, saturated to `i16`.
        #[inline(always)]
        pub fn madd8<const S: i32>(v: [Self; 8], k: [i16; 8], rnd: i32) -> Self {
            unsafe {
                let mut lo = vdupq_n_s32(rnd);
                let mut hi = lo;
                for i in 0..8 {
                    lo = vmlal_n_s16(lo, vget_low_s16(v[i].0), k[i]);
                    hi = vmlal_high_n_s16(hi, v[i].0, k[i]);
                }
                Self(vcombine_s16(
                    vqmovn_s32(vshrq_n_s32::<S>(lo)),
                    vqmovn_s32(vshrq_n_s32::<S>(hi)),
                ))
            }
        }

        /// `(a * k[0] + b * k[1] + c * k[2] + d * k[3] + 2048) >> 12` in 32 bits, saturated to `i16`.
        #[inline(always)]
        pub fn madd4(self, b: Self, c: Self, d: Self, k: [i16; 4]) -> Self {
            unsafe {
                let (k0, k1, k2, k3) = (
                    vdup_n_s16(k[0]),
                    vdup_n_s16(k[1]),
                    vdup_n_s16(k[2]),
                    vdup_n_s16(k[3]),
                );
                let lo = vmlal_s16(
                    vmlal_s16(
                        vmlal_s16(vmull_s16(vget_low_s16(self.0), k0), vget_low_s16(b.0), k1),
                        vget_low_s16(c.0),
                        k2,
                    ),
                    vget_low_s16(d.0),
                    k3,
                );
                let hi = vmlal_s16(
                    vmlal_s16(
                        vmlal_s16(vmull_s16(vget_high_s16(self.0), k0), vget_high_s16(b.0), k1),
                        vget_high_s16(c.0),
                        k2,
                    ),
                    vget_high_s16(d.0),
                    k3,
                );
                Self(vcombine_s16(
                    vqrshrn_n_s32::<12>(lo),
                    vqrshrn_n_s32::<12>(hi),
                ))
            }
        }

        /// `(a * k1[0] + b * k1[1] + 2048) >> 12` and the same with `k2`, in 32 bits, each
        /// saturated to `i16`.
        #[inline(always)]
        pub fn madd2(self, o: Self, k1: [i16; 2], k2: [i16; 2]) -> (Self, Self) {
            unsafe {
                let (a_lo, a_hi) = (vget_low_s16(self.0), vget_high_s16(self.0));
                let (b_lo, b_hi) = (vget_low_s16(o.0), vget_high_s16(o.0));
                let (k1a, k1b) = (vdup_n_s16(k1[0]), vdup_n_s16(k1[1]));
                let (k2a, k2b) = (vdup_n_s16(k2[0]), vdup_n_s16(k2[1]));
                (
                    Self(vcombine_s16(
                        vqrshrn_n_s32::<12>(vmlal_s16(vmull_s16(a_lo, k1a), b_lo, k1b)),
                        vqrshrn_n_s32::<12>(vmlal_s16(vmull_s16(a_hi, k1a), b_hi, k1b)),
                    )),
                    Self(vcombine_s16(
                        vqrshrn_n_s32::<12>(vmlal_s16(vmull_s16(a_lo, k2a), b_lo, k2b)),
                        vqrshrn_n_s32::<12>(vmlal_s16(vmull_s16(a_hi, k2a), b_hi, k2b)),
                    )),
                )
            }
        }

        #[inline(always)]
        pub fn zip_lo16(self, o: Self) -> Self {
            Self(unsafe { vzip1q_s16(self.0, o.0) })
        }

        /// The even lanes of `self` then `o`, and the odd ones.
        #[inline(always)]
        pub fn unzip(self, o: Self) -> (Self, Self) {
            unsafe { (Self(vuzp1q_s16(self.0, o.0)), Self(vuzp2q_s16(self.0, o.0))) }
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

    use super::{I16x8, I32x4};

    pub type Inner = v128;

    pub type Inner32 = v128;

    impl I32x4 {
        #[inline(always)]
        pub fn load(s: &[i32]) -> Self {
            // SAFETY: 4 lanes are in bounds; the load takes unaligned pointers.
            Self(unsafe { v128_load(s[..4].as_ptr().cast()) })
        }

        #[inline(always)]
        pub fn store(self, d: &mut [i32]) {
            // SAFETY: 4 lanes are in bounds; the store takes unaligned pointers.
            unsafe { v128_store(d[..4].as_mut_ptr().cast(), self.0) }
        }

        #[inline(always)]
        pub fn splat(v: i32) -> Self {
            Self(i32x4_splat(v))
        }

        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            Self(i32x4_add(self.0, o.0))
        }

        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            Self(i32x4_sub(self.0, o.0))
        }

        /// The low 32 bits of the products.
        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            Self(i32x4_mul(self.0, o.0))
        }

        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            Self(i32x4_max(self.0, o.0))
        }

        #[inline(always)]
        pub fn shl<const N: i32>(self) -> Self {
            Self(i32x4_shl(self.0, N as u32))
        }

        #[inline(always)]
        pub fn shr_s<const N: i32>(self) -> Self {
            Self(i32x4_shr(self.0, N as u32))
        }

        #[inline(always)]
        pub fn shr_u<const N: i32>(self) -> Self {
            Self(u32x4_shr(self.0, N as u32))
        }

        /// `lo` then `hi` saturated to `i16`.
        #[inline(always)]
        pub fn pack(lo: Self, hi: Self) -> I16x8 {
            I16x8(i16x8_narrow_i32x4(lo.0, hi.0))
        }

        #[inline(always)]
        pub fn min(self, o: Self) -> Self {
            Self(i32x4_min(self.0, o.0))
        }

        #[inline(always)]
        pub fn or(self, o: Self) -> Self {
            Self(v128_or(self.0, o.0))
        }

        /// Whether any lane is nonzero.
        #[inline(always)]
        pub fn any(self) -> bool {
            v128_any_true(self.0)
        }

        /// `(x + (1 << N >> 1)) >> N`.
        #[inline(always)]
        pub fn round_shr<const N: i32>(self) -> Self {
            Self(i32x4_shr(
                i32x4_add(self.0, i32x4_splat(1 << N >> 1)),
                N as u32,
            ))
        }

        /// `(x * k + 2048) >> 12`, for products within 32 bits.
        #[inline(always)]
        pub fn mul_q12(self, k: i32) -> Self {
            Self(i32x4_shr(
                i32x4_add(i32x4_mul(self.0, i32x4_splat(k)), i32x4_splat(2048)),
                12,
            ))
        }

        /// `lo` then `hi` shifted with rounding by `N`, saturated to `i16`.
        #[inline(always)]
        pub fn pack_round_shr<const N: i32>(lo: Self, hi: Self) -> I16x8 {
            Self::pack(lo.round_shr::<N>(), hi.round_shr::<N>())
        }
    }

    impl I16x8 {
        /// The lanes of `self` and `o` as `u16`, summed into 32 bits: low four lanes, then high.
        #[inline(always)]
        pub fn addl_u(self, o: Self) -> [I32x4; 2] {
            let (a, b) = (self.0, o.0);
            [
                I32x4(i32x4_add(
                    u32x4_extend_low_u16x8(a),
                    u32x4_extend_low_u16x8(b),
                )),
                I32x4(i32x4_add(
                    u32x4_extend_high_u16x8(a),
                    u32x4_extend_high_u16x8(b),
                )),
            ]
        }

        /// The lanes as `u16` added onto `acc`'s low four, then high four.
        #[inline(always)]
        pub fn addw_u(self, acc: [I32x4; 2]) -> [I32x4; 2] {
            [
                I32x4(i32x4_add(acc[0].0, u32x4_extend_low_u16x8(self.0))),
                I32x4(i32x4_add(acc[1].0, u32x4_extend_high_u16x8(self.0))),
            ]
        }

        /// The full products, low four lanes then high four.
        #[inline(always)]
        pub fn mul_wide(self, o: Self) -> [I32x4; 2] {
            [
                I32x4(i32x4_extmul_low_i16x8(self.0, o.0)),
                I32x4(i32x4_extmul_high_i16x8(self.0, o.0)),
            ]
        }

        #[inline(always)]
        pub fn gt(self, o: Self) -> Self {
            Self(i16x8_gt(self.0, o.0))
        }

        /// `t[lane]` for lanes below 64, 0 for those up to 255.
        #[inline(always)]
        pub fn lookup64(self, t: &[u8; 64]) -> Self {
            let i = u8x16_narrow_i16x8(self.0, self.0);
            let mut r = u8x16_splat(0);
            for k in 0..4 {
                // SAFETY: 16 bytes are in bounds; the load takes unaligned pointers.
                let t = unsafe { v128_load(t[16 * k..].as_ptr().cast()) };
                // Indices outside this 16 wrap to 16 or more, which the swizzle zeroes.
                r = v128_or(r, i8x16_swizzle(t, u8x16_sub(i, u8x16_splat(16 * k as u8))));
            }
            Self(u16x8_extend_low_u8x16(r))
        }
    }

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

        #[inline(always)]
        pub fn or(self, o: Self) -> Self {
            Self(v128_or(self.0, o.0))
        }

        /// Whether any lane is set.
        #[inline(always)]
        pub fn any(self) -> bool {
            v128_any_true(self.0)
        }

        /// The first set lane of the masks `self` then `o`, 16 if none is.
        #[inline(always)]
        pub fn first_set2(self, o: Self) -> u32 {
            (i8x16_bitmask(i8x16_narrow_i16x8(self.0, o.0)) as u32 | 1 << 16).trailing_zeros()
        }

        /// `a` where `m` is set, `b` elsewhere; `m`'s lanes are all ones or all zeros.
        #[inline(always)]
        pub fn select(m: Self, a: Self, b: Self) -> Self {
            Self(v128_bitselect(a.0, b.0, m.0))
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

        /// Lanes `s[..4]`, the high four zero.
        #[inline(always)]
        pub fn load_lo(s: &[i16]) -> Self {
            let s = &s[..4];
            Self(i16x8(s[0], s[1], s[2], s[3], 0, 0, 0, 0))
        }

        /// `(v[0] * k[0] + .. + v[7] * k[7] + rnd) >> S` in 32 bits, saturated to `i16`.
        #[inline(always)]
        pub fn madd8<const S: i32>(v: [Self; 8], k: [i16; 8], rnd: i32) -> Self {
            let mut lo = i32x4_splat(rnd);
            let mut hi = lo;
            for i in 0..8 {
                let k = i16x8_splat(k[i]);
                lo = i32x4_add(lo, i32x4_extmul_low_i16x8(v[i].0, k));
                hi = i32x4_add(hi, i32x4_extmul_high_i16x8(v[i].0, k));
            }
            Self(i16x8_narrow_i32x4(
                i32x4_shr(lo, S as u32),
                i32x4_shr(hi, S as u32),
            ))
        }

        /// `(a * k[0] + b * k[1] + c * k[2] + d * k[3] + 2048) >> 12` in 32 bits, saturated to `i16`.
        #[inline(always)]
        pub fn madd4(self, b: Self, c: Self, d: Self, k: [i16; 4]) -> Self {
            let (k0, k1, k2, k3) = (
                i16x8_splat(k[0]),
                i16x8_splat(k[1]),
                i16x8_splat(k[2]),
                i16x8_splat(k[3]),
            );
            let rnd = i32x4_splat(2048);
            let lo = i32x4_add(
                i32x4_add(
                    i32x4_extmul_low_i16x8(self.0, k0),
                    i32x4_extmul_low_i16x8(b.0, k1),
                ),
                i32x4_add(
                    i32x4_extmul_low_i16x8(c.0, k2),
                    i32x4_extmul_low_i16x8(d.0, k3),
                ),
            );
            let hi = i32x4_add(
                i32x4_add(
                    i32x4_extmul_high_i16x8(self.0, k0),
                    i32x4_extmul_high_i16x8(b.0, k1),
                ),
                i32x4_add(
                    i32x4_extmul_high_i16x8(c.0, k2),
                    i32x4_extmul_high_i16x8(d.0, k3),
                ),
            );
            Self(i16x8_narrow_i32x4(
                i32x4_shr(i32x4_add(lo, rnd), 12),
                i32x4_shr(i32x4_add(hi, rnd), 12),
            ))
        }

        /// `(a * k1[0] + b * k1[1] + 2048) >> 12` and the same with `k2`, in 32 bits, each
        /// saturated to `i16`.
        #[inline(always)]
        pub fn madd2(self, o: Self, k1: [i16; 2], k2: [i16; 2]) -> (Self, Self) {
            let rnd = i32x4_splat(2048);
            let (k1a, k1b) = (i16x8_splat(k1[0]), i16x8_splat(k1[1]));
            let (k2a, k2b) = (i16x8_splat(k2[0]), i16x8_splat(k2[1]));
            let lo1 = i32x4_add(
                i32x4_extmul_low_i16x8(self.0, k1a),
                i32x4_extmul_low_i16x8(o.0, k1b),
            );
            let hi1 = i32x4_add(
                i32x4_extmul_high_i16x8(self.0, k1a),
                i32x4_extmul_high_i16x8(o.0, k1b),
            );
            let lo2 = i32x4_add(
                i32x4_extmul_low_i16x8(self.0, k2a),
                i32x4_extmul_low_i16x8(o.0, k2b),
            );
            let hi2 = i32x4_add(
                i32x4_extmul_high_i16x8(self.0, k2a),
                i32x4_extmul_high_i16x8(o.0, k2b),
            );
            (
                Self(i16x8_narrow_i32x4(
                    i32x4_shr(i32x4_add(lo1, rnd), 12),
                    i32x4_shr(i32x4_add(hi1, rnd), 12),
                )),
                Self(i16x8_narrow_i32x4(
                    i32x4_shr(i32x4_add(lo2, rnd), 12),
                    i32x4_shr(i32x4_add(hi2, rnd), 12),
                )),
            )
        }

        #[inline(always)]
        pub fn zip_lo16(self, o: Self) -> Self {
            Self(i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(self.0, o.0))
        }

        /// The even lanes of `self` then `o`, and the odd ones.
        #[inline(always)]
        pub fn unzip(self, o: Self) -> (Self, Self) {
            (
                Self(i16x8_shuffle::<0, 2, 4, 6, 8, 10, 12, 14>(self.0, o.0)),
                Self(i16x8_shuffle::<1, 3, 5, 7, 9, 11, 13, 15>(self.0, o.0)),
            )
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
    use super::{I16x8, I32x4};

    pub type Inner = [i16; 8];

    pub type Inner32 = [i32; 4];

    impl I32x4 {
        #[inline(always)]
        fn zip(self, o: Self, f: impl Fn(i32, i32) -> i32) -> Self {
            Self(std::array::from_fn(|i| f(self.0[i], o.0[i])))
        }

        #[inline(always)]
        pub fn load(s: &[i32]) -> Self {
            Self(s[..4].try_into().unwrap())
        }

        #[inline(always)]
        pub fn store(self, d: &mut [i32]) {
            d[..4].copy_from_slice(&self.0);
        }

        #[inline(always)]
        pub fn splat(v: i32) -> Self {
            Self([v; 4])
        }

        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            self.zip(o, i32::wrapping_add)
        }

        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            self.zip(o, i32::wrapping_sub)
        }

        /// The low 32 bits of the products.
        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            self.zip(o, i32::wrapping_mul)
        }

        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            self.zip(o, i32::max)
        }

        #[inline(always)]
        pub fn shl<const N: i32>(self) -> Self {
            Self(self.0.map(|v| v << N))
        }

        #[inline(always)]
        pub fn shr_s<const N: i32>(self) -> Self {
            Self(self.0.map(|v| v >> N))
        }

        #[inline(always)]
        pub fn shr_u<const N: i32>(self) -> Self {
            Self(self.0.map(|v| (v as u32 >> N) as i32))
        }

        /// `lo` then `hi` saturated to `i16`.
        #[inline(always)]
        pub fn pack(lo: Self, hi: Self) -> I16x8 {
            I16x8(std::array::from_fn(|i| {
                [lo.0, hi.0][i >> 2][i & 3].clamp(i16::MIN as i32, i16::MAX as i32) as i16
            }))
        }

        #[inline(always)]
        pub fn min(self, o: Self) -> Self {
            self.zip(o, i32::min)
        }

        #[inline(always)]
        pub fn or(self, o: Self) -> Self {
            self.zip(o, |a, b| a | b)
        }

        /// Whether any lane is nonzero.
        #[inline(always)]
        pub fn any(self) -> bool {
            self.0 != [0; 4]
        }

        /// `(x + (1 << N >> 1)) >> N`.
        #[inline(always)]
        pub fn round_shr<const N: i32>(self) -> Self {
            Self(self.0.map(|v| v.wrapping_add(1 << N >> 1) >> N))
        }

        /// `(x * k + 2048) >> 12`, for products within 32 bits.
        #[inline(always)]
        pub fn mul_q12(self, k: i32) -> Self {
            Self(self.0.map(|v| v.wrapping_mul(k).wrapping_add(2048) >> 12))
        }

        /// `lo` then `hi` shifted with rounding by `N`, saturated to `i16`.
        #[inline(always)]
        pub fn pack_round_shr<const N: i32>(lo: Self, hi: Self) -> I16x8 {
            Self::pack(lo.round_shr::<N>(), hi.round_shr::<N>())
        }
    }

    impl I16x8 {
        /// The lanes of `self` and `o` as `u16`, summed into 32 bits: low four lanes, then high.
        #[inline(always)]
        pub fn addl_u(self, o: Self) -> [I32x4; 2] {
            let v = |i: usize| self.0[i] as u16 as i32 + o.0[i] as u16 as i32;
            [
                I32x4(std::array::from_fn(v)),
                I32x4(std::array::from_fn(|i| v(i + 4))),
            ]
        }

        /// The lanes as `u16` added onto `acc`'s low four, then high four.
        #[inline(always)]
        pub fn addw_u(self, acc: [I32x4; 2]) -> [I32x4; 2] {
            let v = |k: usize, i: usize| acc[k].0[i].wrapping_add(self.0[4 * k + i] as u16 as i32);
            [
                I32x4(std::array::from_fn(|i| v(0, i))),
                I32x4(std::array::from_fn(|i| v(1, i))),
            ]
        }

        /// The full products, low four lanes then high four.
        #[inline(always)]
        pub fn mul_wide(self, o: Self) -> [I32x4; 2] {
            let p = |i: usize| self.0[i] as i32 * o.0[i] as i32;
            [
                I32x4(std::array::from_fn(p)),
                I32x4(std::array::from_fn(|i| p(i + 4))),
            ]
        }

        #[inline(always)]
        pub fn gt(self, o: Self) -> Self {
            self.zip(o, |a, b| -((a > b) as i16))
        }

        /// `t[lane]` for lanes below 64, 0 for those up to 255.
        #[inline(always)]
        pub fn lookup64(self, t: &[u8; 64]) -> Self {
            Self(self.0.map(|v| t.get(v as usize).map_or(0, |&x| x as i16)))
        }
    }

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

        #[inline(always)]
        pub fn or(self, o: Self) -> Self {
            Self(std::array::from_fn(|i| self.0[i] | o.0[i]))
        }

        /// Whether any lane is set.
        #[inline(always)]
        pub fn any(self) -> bool {
            self.0.iter().any(|&m| m != 0)
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

        /// `a` where `m` is set, `b` elsewhere; `m`'s lanes are all ones or all zeros.
        #[inline(always)]
        pub fn select(m: Self, a: Self, b: Self) -> Self {
            Self(std::array::from_fn(|i| {
                if m.0[i] != 0 {
                    a.0[i]
                } else {
                    b.0[i]
                }
            }))
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

        /// Lanes `s[..4]`, the high four zero.
        #[inline(always)]
        pub fn load_lo(s: &[i16]) -> Self {
            Self([s[0], s[1], s[2], s[3], 0, 0, 0, 0])
        }

        /// `(v[0] * k[0] + .. + v[7] * k[7] + rnd) >> S` in 32 bits, saturated to `i16`.
        #[inline(always)]
        pub fn madd8<const S: i32>(v: [Self; 8], k: [i16; 8], rnd: i32) -> Self {
            Self(std::array::from_fn(|l| {
                let mut s = rnd;
                for i in 0..8 {
                    s += v[i].0[l] as i32 * k[i] as i32;
                }
                (s >> S).clamp(i16::MIN as i32, i16::MAX as i32) as i16
            }))
        }

        /// `(a * k[0] + b * k[1] + c * k[2] + d * k[3] + 2048) >> 12` in 32 bits, saturated to `i16`.
        #[inline(always)]
        pub fn madd4(self, b: Self, c: Self, d: Self, k: [i16; 4]) -> Self {
            Self(std::array::from_fn(|i| {
                (self.0[i] as i32 * k[0] as i32
                    + b.0[i] as i32 * k[1] as i32
                    + c.0[i] as i32 * k[2] as i32
                    + d.0[i] as i32 * k[3] as i32
                    + 2048
                    >> 12)
                    .clamp(i16::MIN as i32, i16::MAX as i32) as i16
            }))
        }

        /// `(a * k1[0] + b * k1[1] + 2048) >> 12` and the same with `k2`, in 32 bits, each
        /// saturated to `i16`.
        #[inline(always)]
        pub fn madd2(self, o: Self, k1: [i16; 2], k2: [i16; 2]) -> (Self, Self) {
            let madd = |k: [i16; 2]| {
                self.zip(o, |a, b| {
                    (a as i32 * k[0] as i32 + b as i32 * k[1] as i32 + 2048 >> 12)
                        .clamp(i16::MIN as i32, i16::MAX as i32) as i16
                })
            };
            (madd(k1), madd(k2))
        }

        #[inline(always)]
        pub fn zip_lo16(self, o: Self) -> Self {
            Self(std::array::from_fn(|i| [self.0, o.0][i & 1][i >> 1]))
        }

        /// The even lanes of `self` then `o`, and the odd ones.
        #[inline(always)]
        pub fn unzip(self, o: Self) -> (Self, Self) {
            let at = |k: usize| [self.0, o.0][k >> 3][k & 7];
            (Self(std::array::from_fn(|i| at(2 * i))), Self(std::array::from_fn(|i| at(2 * i + 1))))
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
