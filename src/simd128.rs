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

/// Sixteen `u8` lanes, the 8 bpc loop filter's pixels as dav1d's asm holds them; the `_i`
/// operations read the lanes as `i8`. The sums that outgrow a byte widen to two [`I16x8`].
#[derive(Clone, Copy)]
pub struct U8x16(imp::Inner8);

#[cfg(target_arch = "aarch64")]
pub use imp::Lut256;

impl I16x8 {
    /// Bytes `b[..4]` zero-extended into the low four lanes.
    #[inline(always)]
    pub fn from_u8x4(b: &[u8]) -> Self {
        let v = u32::from_le_bytes(b[..4].try_into().unwrap()) as u64;
        Self::from_u8x8(&v.to_le_bytes())
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
    /// Whether any lane is nonzero.
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

/// The byte-lane operations of the 8 bpc loop filter, for [`U8x16`] on every target and, on
/// x86_64 with AVX2, [`U8x32`]: one kernel source for both widths. The `_i` ones read the lanes as
/// `i8`; the interleaves stay within each 128-bit half.
pub trait Bytes: Copy {
    /// Lanes per vector.
    const N: usize;
    /// Sixteen-bit lanes, two of which [`Bytes::widen`] the bytes into.
    type Wide: Lanes;
    fn splat(v: u8) -> Self;
    fn abd(self, o: Self) -> Self;
    fn max(self, o: Self) -> Self;
    fn add(self, o: Self) -> Self;
    fn add_sat(self, o: Self) -> Self;
    fn add_sat_i(self, o: Self) -> Self;
    fn sub_sat_i(self, o: Self) -> Self;
    fn shr_u<const S: i32>(self) -> Self;
    fn shr_i<const S: i32>(self) -> Self;
    fn le(self, o: Self) -> Self;
    fn eq(self, o: Self) -> Self;
    fn and(self, o: Self) -> Self;
    fn andnot(self, o: Self) -> Self;
    fn xor(self, o: Self) -> Self;
    fn select(m: Self, a: Self, b: Self) -> Self;
    /// Whether any lane is nonzero.
    fn any(self) -> bool;
    /// `t[lane]` for lanes below 64, 0 for the others.
    fn lookup64(self, t: &[u8; 64]) -> Self;
    fn widen(self) -> [Self::Wide; 2];
    /// [`Bytes::widen`] back, each lane rounded `>> S` and saturated.
    fn narrow_rshr<const S: i32>(lo: Self::Wide, hi: Self::Wide) -> Self;
    fn zip_lo8(self, o: Self) -> Self;
    fn zip_hi8(self, o: Self) -> Self;
    fn zip_lo16(self, o: Self) -> Self;
    fn zip_hi16(self, o: Self) -> Self;
    fn zip_lo32(self, o: Self) -> Self;
    fn zip_hi32(self, o: Self) -> Self;
    fn zip_lo64(self, o: Self) -> Self;
    fn zip_hi64(self, o: Self) -> Self;
    /// Byte `j` of `x` in lanes `4 * j..4 * j + 4`, for each of the `N / 4` low bytes.
    fn spread4(x: u64) -> Self;
    /// `v[r]` from the `N` bytes at `b[start + r * stride..]`.
    fn load_rows(v: &mut [Self], b: &[u8], start: usize, stride: isize);
    fn store_rows(v: &[Self], b: &mut [u8], start: usize, stride: isize);
    /// `N` rows of eight bytes from `b[start + r * stride..]` as eight columns of `N`.
    fn load_cols8(b: &[u8], start: usize, stride: isize) -> [Self; 8];
    fn store_cols8(c: &[Self; 8], b: &mut [u8], start: usize, stride: isize);
    /// [`Bytes::load_cols8`] of four bytes per row.
    fn load_cols4(b: &[u8], start: usize, stride: isize) -> [Self; 4];
    fn store_cols4(c: &[Self; 4], b: &mut [u8], start: usize, stride: isize);
}

macro_rules! forward_bytes {
    ($t:ty, $n:literal, $wide:ty) => {
        impl Bytes for $t {
            const N: usize = $n;
            type Wide = $wide;
            #[inline(always)]
            fn splat(v: u8) -> Self {
                <$t>::splat(v)
            }
            #[inline(always)]
            fn abd(self, o: Self) -> Self {
                <$t>::abd(self, o)
            }
            #[inline(always)]
            fn max(self, o: Self) -> Self {
                <$t>::max(self, o)
            }
            #[inline(always)]
            fn add(self, o: Self) -> Self {
                <$t>::add(self, o)
            }
            #[inline(always)]
            fn add_sat(self, o: Self) -> Self {
                <$t>::add_sat(self, o)
            }
            #[inline(always)]
            fn add_sat_i(self, o: Self) -> Self {
                <$t>::add_sat_i(self, o)
            }
            #[inline(always)]
            fn sub_sat_i(self, o: Self) -> Self {
                <$t>::sub_sat_i(self, o)
            }
            #[inline(always)]
            fn shr_u<const S: i32>(self) -> Self {
                <$t>::shr_u::<S>(self)
            }
            #[inline(always)]
            fn shr_i<const S: i32>(self) -> Self {
                <$t>::shr_i::<S>(self)
            }
            #[inline(always)]
            fn le(self, o: Self) -> Self {
                <$t>::le(self, o)
            }
            #[inline(always)]
            fn eq(self, o: Self) -> Self {
                <$t>::eq(self, o)
            }
            #[inline(always)]
            fn and(self, o: Self) -> Self {
                <$t>::and(self, o)
            }
            #[inline(always)]
            fn andnot(self, o: Self) -> Self {
                <$t>::andnot(self, o)
            }
            #[inline(always)]
            fn xor(self, o: Self) -> Self {
                <$t>::xor(self, o)
            }
            #[inline(always)]
            fn select(m: Self, a: Self, b: Self) -> Self {
                <$t>::select(m, a, b)
            }
            #[inline(always)]
            fn any(self) -> bool {
                <$t>::any(self)
            }
            #[inline(always)]
            fn lookup64(self, t: &[u8; 64]) -> Self {
                <$t>::lookup64(self, t)
            }
            #[inline(always)]
            fn widen(self) -> [$wide; 2] {
                <$t>::widen(self)
            }
            #[inline(always)]
            fn narrow_rshr<const S: i32>(lo: $wide, hi: $wide) -> Self {
                <$t>::narrow_rshr::<S>(lo, hi)
            }
            #[inline(always)]
            fn zip_lo8(self, o: Self) -> Self {
                <$t>::zip_lo8(self, o)
            }
            #[inline(always)]
            fn zip_hi8(self, o: Self) -> Self {
                <$t>::zip_hi8(self, o)
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
            #[inline(always)]
            fn spread4(x: u64) -> Self {
                <$t>::spread4(x as _)
            }
            #[inline(always)]
            fn load_rows(v: &mut [Self], b: &[u8], start: usize, stride: isize) {
                <$t>::load_rows(v, b, start, stride)
            }
            #[inline(always)]
            fn store_rows(v: &[Self], b: &mut [u8], start: usize, stride: isize) {
                <$t>::store_rows(v, b, start, stride)
            }
            #[inline(always)]
            fn load_cols8(b: &[u8], start: usize, stride: isize) -> [Self; 8] {
                <$t>::load_cols8(b, start, stride)
            }
            #[inline(always)]
            fn store_cols8(c: &[Self; 8], b: &mut [u8], start: usize, stride: isize) {
                <$t>::store_cols8(c, b, start, stride)
            }
            #[inline(always)]
            fn load_cols4(b: &[u8], start: usize, stride: isize) -> [Self; 4] {
                <$t>::load_cols4(b, start, stride)
            }
            #[inline(always)]
            fn store_cols4(c: &[Self; 4], b: &mut [u8], start: usize, stride: isize) {
                <$t>::store_cols4(c, b, start, stride)
            }
        }
    };
}

forward_bytes!(U8x16, 16, I16x8);
#[cfg(target_arch = "x86_64")]
forward_bytes!(U8x32, 32, I16x16);

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

/// The sum of `b[..N]`, `N` a power of two from 4 to 64: one `psadbw` per 16 bytes.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub fn sum_u8<const N: usize>(b: &[u8]) -> u32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let b = &b[..N];
    // SAFETY: SSE2 is the x86_64 baseline and checked with SSE4.1 on x86; the loads read `b`.
    unsafe {
        let z = _mm_setzero_si128();
        let s = match N {
            4 => _mm_cvtsi32_si128(i32::from_le_bytes(b[..4].try_into().unwrap())),
            8 => _mm_loadl_epi64(b.as_ptr().cast()),
            _ => _mm_loadu_si128(b.as_ptr().cast()),
        };
        let mut s = _mm_sad_epu8(s, z);
        for i in 1..N / 16 {
            s = _mm_add_epi32(s, _mm_sad_epu8(_mm_loadu_si128(b[16 * i..].as_ptr().cast()), z));
        }
        if N >= 16 {
            s = _mm_add_epi32(s, _mm_unpackhi_epi64(s, s));
        }
        _mm_cvtsi128_si32(s) as u32
    }
}

/// Asserts that `n` rows of `w` bytes, row `r` at `start + r * stride`, lie in `len` bytes: the
/// rows lie between the first and the last, so those two are all it checks.
#[inline(always)]
pub(crate) fn rows_fit(len: usize, start: usize, stride: isize, n: usize, w: usize) {
    // In 128 bits, where nothing here overflows.
    let fits = |s: i128| s >= 0 && s + w as i128 <= len as i128;
    let start = start as i128;
    assert!(n > 0 && fits(start) && fits(start + (n as i128 - 1) * stride as i128));
}

/// Rows of a picture as lanes, the block bounds-checked once instead of row by row.
impl U8x16 {
    /// `v[r]` from the sixteen bytes at `b[start + r * stride..]`.
    #[inline(always)]
    pub fn load_rows(v: &mut [Self], b: &[u8], start: usize, stride: isize) {
        rows_fit(b.len(), start, stride, v.len(), 16);
        let p = b.as_ptr().wrapping_add(start);
        for (r, v) in v.iter_mut().enumerate() {
            // SAFETY: Row `r` lies in `b`, checked by `rows_fit`.
            *v = unsafe { Self::load_at(p.offset(r as isize * stride)) };
        }
    }

    /// [`Self::load_rows`] of eight bytes, in the low halves.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    #[inline(always)]
    pub fn load_rows_lo(v: &mut [Self], b: &[u8], start: usize, stride: isize) {
        rows_fit(b.len(), start, stride, v.len(), 8);
        let p = b.as_ptr().wrapping_add(start);
        for (r, v) in v.iter_mut().enumerate() {
            // SAFETY: Row `r` lies in `b`, checked by `rows_fit`.
            *v = unsafe { Self::load_lo_at(p.offset(r as isize * stride)) };
        }
    }

    /// [`Self::load_rows`] back.
    #[inline(always)]
    pub fn store_rows(v: &[Self], b: &mut [u8], start: usize, stride: isize) {
        rows_fit(b.len(), start, stride, v.len(), 16);
        let p = b.as_mut_ptr().wrapping_add(start);
        for (r, v) in v.iter().enumerate() {
            // SAFETY: Row `r` lies in `b`, checked by `rows_fit`.
            unsafe { v.store_at(p.offset(r as isize * stride)) };
        }
    }

    /// The halves of `v[i]` into rows `2 * i` and `2 * i + 1` of eight bytes.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    #[inline(always)]
    pub fn store_rows_halves(v: &[Self], b: &mut [u8], start: usize, stride: isize) {
        rows_fit(b.len(), start, stride, 2 * v.len(), 8);
        let p = b.as_mut_ptr().wrapping_add(start);
        for (i, v) in v.iter().enumerate() {
            let r = 2 * i as isize;
            // SAFETY: Rows `r` and `r + 1` lie in `b`, checked by `rows_fit`.
            unsafe { v.store_halves_at(p.offset(r * stride), p.offset((r + 1) * stride)) };
        }
    }

    /// [`Self::load_rows`] of four bytes, in the low quarters.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    #[inline(always)]
    pub fn load_rows_s(v: &mut [Self], b: &[u8], start: usize, stride: isize) {
        rows_fit(b.len(), start, stride, v.len(), 4);
        let p = b.as_ptr().wrapping_add(start);
        for (r, v) in v.iter_mut().enumerate() {
            // SAFETY: Row `r` lies in `b`, checked by `rows_fit`.
            let x = unsafe { p.offset(r as isize * stride).cast::<u32>().read_unaligned() };
            *v = Self::from_u32(x);
        }
    }

    /// The quarters of `v[i]` into rows `4 * i..4 * i + 4` of four bytes.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    #[inline(always)]
    pub fn store_rows_quarters(v: &[Self], b: &mut [u8], start: usize, stride: isize) {
        rows_fit(b.len(), start, stride, 4 * v.len(), 4);
        let p = b.as_mut_ptr().wrapping_add(start);
        for (i, v) in v.iter().enumerate() {
            let x = [
                v.lane32::<0>(),
                v.lane32::<1>(),
                v.lane32::<2>(),
                v.lane32::<3>(),
            ];
            for (k, x) in x.into_iter().enumerate() {
                let r = (4 * i + k) as isize;
                // SAFETY: Row `r` lies in `b`, checked by `rows_fit`.
                unsafe { p.offset(r * stride).cast::<u32>().write_unaligned(x) };
            }
        }
    }

    /// Sixteen rows of eight bytes from `b[start + r * stride..]` as eight columns of sixteen.
    #[inline(always)]
    pub fn load_cols8(b: &[u8], start: usize, stride: isize) -> [Self; 8] {
        #[cfg(target_arch = "aarch64")]
        {
            rows_fit(b.len(), start, stride, 16, 8);
            let p = b.as_ptr().wrapping_add(start);
            let mut v = [Self::splat(0); 8];
            for (i, v) in v.iter_mut().enumerate() {
                let r = i as isize;
                // SAFETY: Rows `r` and `r + 8` lie in `b`, checked by `rows_fit`.
                *v = unsafe {
                    Self::load_halves_at(p.offset(r * stride), p.offset((r + 8) * stride))
                };
            }
            imp::transpose_8x16b(v)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut r = [Self::splat(0); 16];
            Self::load_rows_lo(&mut r, b, start, stride);
            transpose16x8(&r)
        }
    }

    /// [`Self::load_cols8`] back.
    #[inline(always)]
    pub fn store_cols8(c: &[Self; 8], b: &mut [u8], start: usize, stride: isize) {
        #[cfg(target_arch = "aarch64")]
        {
            rows_fit(b.len(), start, stride, 16, 8);
            let p = b.as_mut_ptr().wrapping_add(start);
            for (i, v) in imp::transpose_8x16b(*c).into_iter().enumerate() {
                let r = i as isize;
                // SAFETY: Rows `r` and `r + 8` lie in `b`, checked by `rows_fit`.
                unsafe { v.store_halves_at(p.offset(r * stride), p.offset((r + 8) * stride)) };
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        Self::store_rows_halves(&transpose8x16(c), b, start, stride);
    }

    /// [`Self::load_cols8`] of four bytes per row.
    #[inline(always)]
    pub fn load_cols4(b: &[u8], start: usize, stride: isize) -> [Self; 4] {
        #[cfg(target_arch = "aarch64")]
        {
            rows_fit(b.len(), start, stride, 16, 4);
            let p = b.as_ptr().wrapping_add(start);
            let mut v = [Self::splat(0); 4];
            for (j, v) in v.iter_mut().enumerate() {
                // Rows `j`, `j + 4`, `j + 8` and `j + 12`, as the asm's transpose takes them.
                let row = |l: usize| p.wrapping_offset((j + 4 * l) as isize * stride);
                // SAFETY: The four rows lie in `b`, checked by `rows_fit`.
                *v = unsafe { Self::load_lanes32_at([row(0), row(1), row(2), row(3)]) };
            }
            imp::transpose_4x16b(v)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut r = [Self::splat(0); 16];
            Self::load_rows_s(&mut r, b, start, stride);
            transpose16x4(&r)
        }
    }

    /// [`Self::load_cols4`] back.
    #[inline(always)]
    pub fn store_cols4(c: &[Self; 4], b: &mut [u8], start: usize, stride: isize) {
        #[cfg(target_arch = "aarch64")]
        {
            rows_fit(b.len(), start, stride, 16, 4);
            let p = b.as_mut_ptr().wrapping_add(start);
            for (j, v) in imp::transpose_4x16b(*c).into_iter().enumerate() {
                let row = |l: usize| p.wrapping_offset((j + 4 * l) as isize * stride);
                // SAFETY: The four rows lie in `b`, checked by `rows_fit`.
                unsafe { v.store_lanes32_at([row(0), row(1), row(2), row(3)]) };
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        Self::store_rows_quarters(&transpose4x16(c), b, start, stride);
    }

    /// Byte `j` of `x` in lanes `4 * j..4 * j + 4`.
    #[inline(always)]
    pub fn spread4(x: u32) -> Self {
        let v = Self::from_u32(x);
        let v = v.zip_lo8(v);
        v.zip_lo16(v)
    }
}

/// Sixteen rows of eight bytes, each in the low half of `r[i]`, as eight columns of sixteen:
/// pairs of rows interleaved into 16-bit elements, then [`transpose8x8`]'s rounds on those.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
pub fn transpose16x8<T: Bytes>(r: &[T; 16]) -> [T; 8] {
    let mut a = [r[0]; 8];
    for (j, a) in a.iter_mut().enumerate() {
        *a = r[2 * j].zip_lo8(r[2 * j + 1]);
    }
    // Four rows each, columns 0-3 then 4-7.
    let b = [
        a[0].zip_lo16(a[1]),
        a[0].zip_hi16(a[1]),
        a[2].zip_lo16(a[3]),
        a[2].zip_hi16(a[3]),
        a[4].zip_lo16(a[5]),
        a[4].zip_hi16(a[5]),
        a[6].zip_lo16(a[7]),
        a[6].zip_hi16(a[7]),
    ];
    // Eight rows each, two columns at a time.
    let c = [
        b[0].zip_lo32(b[2]),
        b[0].zip_hi32(b[2]),
        b[1].zip_lo32(b[3]),
        b[1].zip_hi32(b[3]),
        b[4].zip_lo32(b[6]),
        b[4].zip_hi32(b[6]),
        b[5].zip_lo32(b[7]),
        b[5].zip_hi32(b[7]),
    ];
    let mut out = a;
    for i in 0..4 {
        out[2 * i] = c[i].zip_lo64(c[i + 4]);
        out[2 * i + 1] = c[i].zip_hi64(c[i + 4]);
    }
    out
}

/// [`transpose16x8`] of sixteen rows of four bytes, each in the low quarter of `r[i]`.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
pub fn transpose16x4<T: Bytes>(r: &[T; 16]) -> [T; 4] {
    let mut a = [r[0]; 8];
    for (j, a) in a.iter_mut().enumerate() {
        *a = r[2 * j].zip_lo8(r[2 * j + 1]);
    }
    // Four rows each, the four columns.
    let b = [
        a[0].zip_lo16(a[1]),
        a[2].zip_lo16(a[3]),
        a[4].zip_lo16(a[5]),
        a[6].zip_lo16(a[7]),
    ];
    // Eight rows each, two columns at a time.
    let c = [
        b[0].zip_lo32(b[1]),
        b[0].zip_hi32(b[1]),
        b[2].zip_lo32(b[3]),
        b[2].zip_hi32(b[3]),
    ];
    [
        c[0].zip_lo64(c[2]),
        c[0].zip_hi64(c[2]),
        c[1].zip_lo64(c[3]),
        c[1].zip_hi64(c[3]),
    ]
}

/// [`transpose16x4`] back: rows `4 * i..4 * i + 4` in the quarters of `out[i]`.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
pub fn transpose4x16<T: Bytes>(c: &[T; 4]) -> [T; 4] {
    // Rows 0-7 then 8-15, two columns each.
    let a = [
        c[0].zip_lo8(c[1]),
        c[2].zip_lo8(c[3]),
        c[0].zip_hi8(c[1]),
        c[2].zip_hi8(c[3]),
    ];
    [
        a[0].zip_lo16(a[1]),
        a[0].zip_hi16(a[1]),
        a[2].zip_lo16(a[3]),
        a[2].zip_hi16(a[3]),
    ]
}

/// [`transpose16x8`] back: eight columns of sixteen bytes as sixteen rows of eight, rows `2 * i`
/// and `2 * i + 1` in the halves of `out[i]`. Three rounds, as halves need no fourth.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
pub fn transpose8x16<T: Bytes>(c: &[T; 8]) -> [T; 8] {
    // Rows 0-7 then 8-15, two columns each.
    let a = [
        c[0].zip_lo8(c[1]),
        c[2].zip_lo8(c[3]),
        c[4].zip_lo8(c[5]),
        c[6].zip_lo8(c[7]),
        c[0].zip_hi8(c[1]),
        c[2].zip_hi8(c[3]),
        c[4].zip_hi8(c[5]),
        c[6].zip_hi8(c[7]),
    ];
    let mut out = a;
    for h in 0..2 {
        let a = &a[4 * h..];
        // Rows 0-3 then 4-7 of the half, columns 0-3 then 4-7.
        let b = [
            a[0].zip_lo16(a[1]),
            a[0].zip_hi16(a[1]),
            a[2].zip_lo16(a[3]),
            a[2].zip_hi16(a[3]),
        ];
        out[4 * h] = b[0].zip_lo32(b[2]);
        out[4 * h + 1] = b[0].zip_hi32(b[2]);
        out[4 * h + 2] = b[1].zip_lo32(b[3]);
        out[4 * h + 3] = b[1].zip_hi32(b[3]);
    }
    out
}

#[cfg(target_arch = "x86_64")]
pub use wide::{I16x16, U8x32};

/// Sixteen `i16` lanes in one AVX2 register, entered only through a
/// `#[target_feature(enable = "avx2")]` kernel chosen once the CPU flags hold AVX2.
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)]
mod wide {
    use std::arch::x86_64::*;

    use super::{rows_fit, transpose16x4, transpose16x8, transpose4x16, transpose8x16, I16x8};

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

        #[inline(always)]
        pub fn load_u16(s: &[u16]) -> Self {
            Self(unsafe { _mm256_loadu_si256(s[..16].as_ptr().cast()) })
        }

        #[inline(always)]
        pub fn store_u16(self, d: &mut [u16]) {
            unsafe { _mm256_storeu_si256(d[..16].as_mut_ptr().cast(), self.0) }
        }

        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            Self(unsafe { _mm256_mullo_epi16(self.0, o.0) })
        }

        /// `t[lane & mask]` by two `vpgatherdd`s of the dwords from each lane's byte on, the
        /// lanes split by the in-lane unpacks so that the pack puts them back in order. `t`
        /// holds three bytes past `mask`.
        #[inline(always)]
        pub fn gather_u8(self, t: &[u8], mask: u16) -> Self {
            assert!(t.len() > mask as usize + 3);
            unsafe {
                let v = _mm256_and_si256(self.0, _mm256_set1_epi16(mask as i16));
                let z = _mm256_setzero_si256();
                let b = _mm256_set1_epi32(0xff);
                let p = t.as_ptr().cast();
                let g0 = _mm256_i32gather_epi32::<1>(p, _mm256_unpacklo_epi16(v, z));
                let g1 = _mm256_i32gather_epi32::<1>(p, _mm256_unpackhi_epi16(v, z));
                Self(_mm256_packus_epi32(
                    _mm256_and_si256(g0, b),
                    _mm256_and_si256(g1, b),
                ))
            }
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
            unsafe { _mm256_testz_si256(self.0, self.0) == 0 }
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

    /// Thirty-two bytes in one AVX2 register: [`super::U8x16`]'s operations on both 128-bit
    /// halves, the rows of a picture block in lanes as dav1d's AVX2 loop filter holds them.
    #[derive(Clone, Copy)]
    pub struct U8x32(__m256i);

    impl U8x32 {
        /// `lo` in the low half, `hi` in the high one.
        #[inline(always)]
        fn halves(lo: __m128i, hi: __m128i) -> Self {
            Self(unsafe { _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi) })
        }

        #[inline(always)]
        pub fn splat(v: u8) -> Self {
            Self(unsafe { _mm256_set1_epi8(v as i8) })
        }

        #[inline(always)]
        pub fn abd(self, o: Self) -> Self {
            Self(unsafe {
                _mm256_or_si256(_mm256_subs_epu8(self.0, o.0), _mm256_subs_epu8(o.0, self.0))
            })
        }

        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            Self(unsafe { _mm256_max_epu8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            Self(unsafe { _mm256_add_epi8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn add_sat(self, o: Self) -> Self {
            Self(unsafe { _mm256_adds_epu8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn add_sat_i(self, o: Self) -> Self {
            Self(unsafe { _mm256_adds_epi8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn sub_sat_i(self, o: Self) -> Self {
            Self(unsafe { _mm256_subs_epi8(self.0, o.0) })
        }

        /// There is no byte shift: a 16-bit one, with the bits from the next byte masked off.
        #[inline(always)]
        pub fn shr_u<const N: i32>(self) -> Self {
            Self(unsafe {
                _mm256_and_si256(
                    _mm256_srli_epi16::<N>(self.0),
                    _mm256_set1_epi8((0xff_u8 >> N) as i8),
                )
            })
        }

        /// The unsigned shift, its sign bit extended by `(v ^ m) - m`.
        #[inline(always)]
        pub fn shr_i<const N: i32>(self) -> Self {
            let m = Self::splat(0x80 >> N);
            Self(unsafe { _mm256_sub_epi8(self.shr_u::<N>().xor(m).0, m.0) })
        }

        #[inline(always)]
        pub fn le(self, o: Self) -> Self {
            Self(unsafe { _mm256_cmpeq_epi8(_mm256_max_epu8(self.0, o.0), o.0) })
        }

        #[inline(always)]
        pub fn eq(self, o: Self) -> Self {
            Self(unsafe { _mm256_cmpeq_epi8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn and(self, o: Self) -> Self {
            Self(unsafe { _mm256_and_si256(self.0, o.0) })
        }

        #[inline(always)]
        pub fn andnot(self, o: Self) -> Self {
            Self(unsafe { _mm256_andnot_si256(o.0, self.0) })
        }

        #[inline(always)]
        pub fn xor(self, o: Self) -> Self {
            Self(unsafe { _mm256_xor_si256(self.0, o.0) })
        }

        #[inline(always)]
        pub fn select(m: Self, a: Self, b: Self) -> Self {
            Self(unsafe { _mm256_blendv_epi8(b.0, a.0, m.0) })
        }

        #[inline(always)]
        pub fn any(self) -> bool {
            unsafe { _mm256_testz_si256(self.0, self.0) == 0 }
        }

        /// `t[lane]` for lanes below 64, 0 for the others.
        #[inline(always)]
        pub fn lookup64(self, t: &[u8; 64]) -> Self {
            unsafe {
                let mut r = _mm256_setzero_si256();
                for k in 0..4 {
                    let t =
                        _mm256_broadcastsi128_si256(_mm_loadu_si128(t[16 * k..].as_ptr().cast()));
                    // Indices outside this 16 saturate to the top bit, which zeroes them.
                    let ik = _mm256_adds_epu8(
                        _mm256_sub_epi8(self.0, _mm256_set1_epi8(16 * k as i8)),
                        _mm256_set1_epi8(0x70),
                    );
                    r = _mm256_or_si256(r, _mm256_shuffle_epi8(t, ik));
                }
                Self(r)
            }
        }

        /// The low and high eight bytes of each half, zero-extended: lane order within the
        /// halves, which [`Self::narrow_rshr`] restores.
        #[inline(always)]
        pub fn widen(self) -> [I16x16; 2] {
            unsafe {
                let z = _mm256_setzero_si256();
                [
                    I16x16(_mm256_unpacklo_epi8(self.0, z)),
                    I16x16(_mm256_unpackhi_epi8(self.0, z)),
                ]
            }
        }

        #[inline(always)]
        pub fn narrow_rshr<const N: i32>(lo: I16x16, hi: I16x16) -> Self {
            unsafe {
                let r = _mm256_set1_epi16(1 << (N - 1));
                Self(_mm256_packus_epi16(
                    _mm256_srai_epi16::<N>(_mm256_add_epi16(lo.0, r)),
                    _mm256_srai_epi16::<N>(_mm256_add_epi16(hi.0, r)),
                ))
            }
        }

        #[inline(always)]
        pub fn zip_lo8(self, o: Self) -> Self {
            Self(unsafe { _mm256_unpacklo_epi8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn zip_hi8(self, o: Self) -> Self {
            Self(unsafe { _mm256_unpackhi_epi8(self.0, o.0) })
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

        /// Byte `j` of `x` in lanes `4 * j..4 * j + 4`.
        #[inline(always)]
        pub fn spread4(x: u64) -> Self {
            unsafe {
                let v = _mm_cvtsi64_si128(x as i64);
                let v = _mm_unpacklo_epi8(v, v);
                Self::halves(_mm_unpacklo_epi16(v, v), _mm_unpackhi_epi16(v, v))
            }
        }

        /// `v[r]` from the 32 bytes at `b[start + r * stride..]`.
        #[inline(always)]
        pub fn load_rows(v: &mut [Self], b: &[u8], start: usize, stride: isize) {
            rows_fit(b.len(), start, stride, v.len(), 32);
            let p = b.as_ptr().wrapping_add(start);
            for (r, v) in v.iter_mut().enumerate() {
                // SAFETY: Row `r` lies in `b`, checked by `rows_fit`.
                *v = Self(unsafe { _mm256_loadu_si256(p.offset(r as isize * stride).cast()) });
            }
        }

        /// [`Self::load_rows`] back.
        #[inline(always)]
        pub fn store_rows(v: &[Self], b: &mut [u8], start: usize, stride: isize) {
            rows_fit(b.len(), start, stride, v.len(), 32);
            let p = b.as_mut_ptr().wrapping_add(start);
            for (r, v) in v.iter().enumerate() {
                // SAFETY: Row `r` lies in `b`, checked by `rows_fit`.
                unsafe { _mm256_storeu_si256(p.offset(r as isize * stride).cast(), v.0) };
            }
        }

        /// Thirty-two rows of eight bytes as eight columns: rows `r` and `r + 16` in the low
        /// halves of the two 128-bit halves, then [`transpose16x8`] on both at once.
        #[inline(always)]
        pub fn load_cols8(b: &[u8], start: usize, stride: isize) -> [Self; 8] {
            rows_fit(b.len(), start, stride, 32, 8);
            let p = b.as_ptr().wrapping_add(start);
            let mut r = [Self::splat(0); 16];
            for (i, r) in r.iter_mut().enumerate() {
                let i = i as isize;
                // SAFETY: Rows `i` and `i + 16` lie in `b`, checked by `rows_fit`.
                *r = unsafe {
                    Self::halves(
                        _mm_loadl_epi64(p.offset(i * stride).cast()),
                        _mm_loadl_epi64(p.offset((i + 16) * stride).cast()),
                    )
                };
            }
            transpose16x8(&r)
        }

        /// [`Self::load_cols8`] back.
        #[inline(always)]
        pub fn store_cols8(c: &[Self; 8], b: &mut [u8], start: usize, stride: isize) {
            rows_fit(b.len(), start, stride, 32, 8);
            let p = b.as_mut_ptr().wrapping_add(start);
            for (i, v) in transpose8x16(c).into_iter().enumerate() {
                let r = 2 * i as isize;
                // SAFETY: Rows `r`, `r + 1`, `r + 16` and `r + 17` lie in `b`, checked by
                // `rows_fit`.
                unsafe {
                    let (lo, hi) = (
                        _mm256_castsi256_si128(v.0),
                        _mm256_extracti128_si256::<1>(v.0),
                    );
                    _mm_storel_epi64(p.offset(r * stride).cast(), lo);
                    _mm_storeh_pd(p.offset((r + 1) * stride).cast(), _mm_castsi128_pd(lo));
                    _mm_storel_epi64(p.offset((r + 16) * stride).cast(), hi);
                    _mm_storeh_pd(p.offset((r + 17) * stride).cast(), _mm_castsi128_pd(hi));
                }
            }
        }

        /// [`Self::load_cols8`] of four bytes per row.
        #[inline(always)]
        pub fn load_cols4(b: &[u8], start: usize, stride: isize) -> [Self; 4] {
            rows_fit(b.len(), start, stride, 32, 4);
            let p = b.as_ptr().wrapping_add(start);
            let mut r = [Self::splat(0); 16];
            for (i, r) in r.iter_mut().enumerate() {
                let i = i as isize;
                // SAFETY: Rows `i` and `i + 16` lie in `b`, checked by `rows_fit`.
                *r = unsafe {
                    let lo = p.offset(i * stride).cast::<i32>().read_unaligned();
                    let hi = p.offset((i + 16) * stride).cast::<i32>().read_unaligned();
                    Self::halves(_mm_cvtsi32_si128(lo), _mm_cvtsi32_si128(hi))
                };
            }
            transpose16x4(&r)
        }

        /// [`Self::load_cols4`] back.
        #[inline(always)]
        pub fn store_cols4(c: &[Self; 4], b: &mut [u8], start: usize, stride: isize) {
            rows_fit(b.len(), start, stride, 32, 4);
            let p = b.as_mut_ptr().wrapping_add(start);
            for (i, v) in transpose4x16(c).into_iter().enumerate() {
                let x = unsafe {
                    [
                        _mm256_extract_epi32::<0>(v.0),
                        _mm256_extract_epi32::<1>(v.0),
                        _mm256_extract_epi32::<2>(v.0),
                        _mm256_extract_epi32::<3>(v.0),
                        _mm256_extract_epi32::<4>(v.0),
                        _mm256_extract_epi32::<5>(v.0),
                        _mm256_extract_epi32::<6>(v.0),
                        _mm256_extract_epi32::<7>(v.0),
                    ]
                };
                for (k, x) in x.into_iter().enumerate() {
                    let r = (4 * i + k % 4 + 16 * (k / 4)) as isize;
                    // SAFETY: Row `r` lies in `b`, checked by `rows_fit`.
                    unsafe { p.offset(r * stride).cast::<i32>().write_unaligned(x) };
                }
            }
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

    use super::{I16x8, I32x4, U8x16};

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

        /// `lo` then `hi` shifted right by `N`, saturated to `i16`.
        #[inline(always)]
        pub fn pack_shr<const N: i32>(lo: Self, hi: Self) -> I16x8 {
            Self::pack(lo.shr_s::<N>(), hi.shr_s::<N>())
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

        /// `t[lane & mask]` by one `vpgatherdd` of the dwords from each lane's byte on: AVX2
        /// only, and `t` holds three bytes past `mask`.
        #[cfg(target_arch = "x86_64")]
        #[inline(always)]
        pub fn gather_u8(self, t: &[u8], mask: u16) -> Self {
            assert!(t.len() > mask as usize + 3);
            unsafe {
                let i = _mm256_cvtepu16_epi32(_mm_and_si128(self.0, _mm_set1_epi16(mask as i16)));
                let g = _mm256_i32gather_epi32::<1>(t.as_ptr().cast(), i);
                let g = _mm256_and_si256(g, _mm256_set1_epi32(0xff));
                Self(_mm_packus_epi32(
                    _mm256_castsi256_si128(g),
                    _mm256_extracti128_si256::<1>(g),
                ))
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

        /// Bytes `b[..8]` as `i8`, sign-extended.
        #[inline(always)]
        pub fn from_i8x8(b: &[u8]) -> Self {
            Self(unsafe { _mm_cvtepi8_epi16(_mm_loadl_epi64(b[..8].as_ptr().cast())) })
        }

        /// `(b[2 * i] + b[2 * i + 1] + 1) >> 1` of `b[..16]`.
        #[inline(always)]
        pub fn avg_pairs_u8(b: &[u8]) -> Self {
            unsafe {
                let v = _mm_loadu_si128(b[..16].as_ptr().cast());
                let sum = _mm_maddubs_epi16(v, _mm_set1_epi8(1));
                Self(_mm_avg_epu16(sum, _mm_setzero_si128()))
            }
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

        /// The high byte of each lane.
        #[inline(always)]
        pub fn hi_u8(self) -> [u8; 8] {
            let mut out = [0; 8];
            unsafe {
                let hi = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1);
                _mm_storel_epi64(out.as_mut_ptr().cast(), _mm_shuffle_epi8(self.0, hi));
            }
            out
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
            self.first_set_byte() / 2
        }

        /// The byte offset of the first set lane of a mask, 16 if none is.
        #[inline(always)]
        pub fn first_set_byte(self) -> u32 {
            (unsafe { _mm_movemask_epi8(self.0) } as u32 | 1 << 16).trailing_zeros()
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

        /// The sum of the lanes as `u16`.
        #[inline(always)]
        pub fn hsum_u(self) -> u32 {
            unsafe {
                let lo = _mm_cvtepu16_epi32(self.0);
                let hi = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(self.0));
                let s = _mm_add_epi32(lo, hi);
                let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
                let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
                _mm_cvtsi128_si32(s) as u32
            }
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
                Self(_mm_packs_epi32(
                    _mm_srai_epi32::<S>(lo),
                    _mm_srai_epi32::<S>(hi),
                ))
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

    // SAFETY (every `unsafe` here): SSE4.1 holds where these are compiled in.
    impl I16x8 {
        /// The sums of adjacent lanes, of `self` then of `o`.
        #[inline(always)]
        pub fn hadd(self, o: Self) -> Self {
            Self(unsafe { _mm_hadd_epi16(self.0, o.0) })
        }

        /// Bytes `B..B + 16` of `self` then `o`: `o`'s lanes shifted in from the top.
        #[inline(always)]
        pub fn ext<const B: i32>(self, o: Self) -> Self {
            Self(unsafe { _mm_alignr_epi8::<B>(o.0, self.0) })
        }

    }

    impl I32x4 {
        /// The sums of adjacent lanes, of `self` then of `o`.
        #[inline(always)]
        pub fn hadd(self, o: Self) -> Self {
            Self(unsafe { _mm_hadd_epi32(self.0, o.0) })
        }
    }

    impl I16x8 {
        /// `self² + o²` in 32 bits, low four lanes then high four.
        #[inline(always)]
        pub fn sq2(self, o: Self) -> [I32x4; 2] {
            unsafe {
                let (lo, hi) = (_mm_unpacklo_epi16(self.0, o.0), _mm_unpackhi_epi16(self.0, o.0));
                [I32x4(_mm_madd_epi16(lo, lo)), I32x4(_mm_madd_epi16(hi, hi))]
            }
        }

        /// `acc + self² + o²`.
        #[inline(always)]
        pub fn sq2_acc(self, o: Self, acc: [I32x4; 2]) -> [I32x4; 2] {
            let s = self.sq2(o);
            [acc[0].add(s[0]), acc[1].add(s[1])]
        }

        /// `acc + self²`.
        #[inline(always)]
        pub fn sq_acc(self, acc: [I32x4; 2]) -> [I32x4; 2] {
            let s = self.mul_wide(self);
            [acc[0].add(s[0]), acc[1].add(s[1])]
        }

        /// `(v[0] * k[0] + .. + v[N - 1] * k[N - 1] + rnd) >> S` in 32 bits, saturated to `i16`.
        #[inline(always)]
        pub fn madd_n<const S: i32, const N: usize>(v: [Self; N], k: [i16; N], rnd: i32) -> Self {
            unsafe {
                let mut lo = _mm_set1_epi32(rnd);
                let mut hi = lo;
                for i in (0..N).step_by(2) {
                    let (b, kb) = if i + 1 < N {
                        (v[i + 1].0, k[i + 1])
                    } else {
                        (_mm_setzero_si128(), 0)
                    };
                    let k = _mm_set1_epi32(k[i] as u16 as i32 | (kb as i32) << 16);
                    lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(v[i].0, b), k));
                    hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_unpackhi_epi16(v[i].0, b), k));
                }
                Self(_mm_packs_epi32(_mm_srai_epi32::<S>(lo), _mm_srai_epi32::<S>(hi)))
            }
        }
    }

    pub type Inner8 = __m128i;

    // SAFETY (every `unsafe` here): SSE4.1 holds where these are compiled in (see the module
    // docs), and loads and stores go through bounds-checked slices of 16 or 8 bytes.
    impl U8x16 {
        /// # Safety
        ///
        /// The 16 bytes at `p` are readable.
        #[inline(always)]
        pub unsafe fn load_at(p: *const u8) -> Self {
            Self(unsafe { _mm_loadu_si128(p.cast()) })
        }

        /// # Safety
        ///
        /// The 8 bytes at `p` are readable.
        #[inline(always)]
        pub unsafe fn load_lo_at(p: *const u8) -> Self {
            Self(unsafe { _mm_loadl_epi64(p.cast()) })
        }

        /// # Safety
        ///
        /// The 16 bytes at `p` are writable.
        #[inline(always)]
        pub unsafe fn store_at(self, p: *mut u8) {
            unsafe { _mm_storeu_si128(p.cast(), self.0) }
        }

        /// # Safety
        ///
        /// The 8 bytes at `lo` and `hi` are writable.
        #[inline(always)]
        pub unsafe fn store_halves_at(self, lo: *mut u8, hi: *mut u8) {
            unsafe {
                _mm_storel_epi64(lo.cast(), self.0);
                _mm_storeh_pd(hi.cast(), _mm_castsi128_pd(self.0));
            }
        }

        /// Bytes `4 * L..4 * L + 4`.
        #[inline(always)]
        pub fn lane32<const L: i32>(self) -> u32 {
            unsafe { _mm_extract_epi32::<L>(self.0) as u32 }
        }

        /// `x` in the low four lanes, the others 0.
        #[inline(always)]
        pub fn from_u32(x: u32) -> Self {
            Self(unsafe { _mm_cvtsi32_si128(x as i32) })
        }

        /// `t[lane]` for lanes below 64, 0 for the others.
        #[inline(always)]
        pub fn lookup64(self, t: &[u8; 64]) -> Self {
            unsafe {
                let mut r = _mm_setzero_si128();
                for k in 0..4 {
                    let t = _mm_loadu_si128(t[16 * k..].as_ptr().cast());
                    // Indices outside this 16 saturate to the top bit, which zeroes them.
                    let ik = _mm_adds_epu8(
                        _mm_sub_epi8(self.0, _mm_set1_epi8(16 * k as i8)),
                        _mm_set1_epi8(0x70),
                    );
                    r = _mm_or_si128(r, _mm_shuffle_epi8(t, ik));
                }
                Self(r)
            }
        }

        #[inline(always)]
        pub fn splat(v: u8) -> Self {
            Self(unsafe { _mm_set1_epi8(v as i8) })
        }

        #[inline(always)]
        pub fn abd(self, o: Self) -> Self {
            Self(unsafe { _mm_or_si128(_mm_subs_epu8(self.0, o.0), _mm_subs_epu8(o.0, self.0)) })
        }

        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            Self(unsafe { _mm_max_epu8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn add_sat(self, o: Self) -> Self {
            Self(unsafe { _mm_adds_epu8(self.0, o.0) })
        }

        /// There is no byte shift: a 16-bit one, with the bits from the next byte masked off.
        #[inline(always)]
        pub fn shr_u<const N: i32>(self) -> Self {
            Self(unsafe {
                _mm_and_si128(
                    _mm_srli_epi16::<N>(self.0),
                    _mm_set1_epi8((0xff_u8 >> N) as i8),
                )
            })
        }

        #[inline(always)]
        pub fn le(self, o: Self) -> Self {
            Self(unsafe { _mm_cmpeq_epi8(_mm_max_epu8(self.0, o.0), o.0) })
        }

        #[inline(always)]
        pub fn eq(self, o: Self) -> Self {
            Self(unsafe { _mm_cmpeq_epi8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn and(self, o: Self) -> Self {
            Self(unsafe { _mm_and_si128(self.0, o.0) })
        }

        #[inline(always)]
        pub fn andnot(self, o: Self) -> Self {
            Self(unsafe { _mm_andnot_si128(o.0, self.0) })
        }

        #[inline(always)]
        pub fn xor(self, o: Self) -> Self {
            Self(unsafe { _mm_xor_si128(self.0, o.0) })
        }

        #[inline(always)]
        pub fn select(m: Self, a: Self, b: Self) -> Self {
            Self(unsafe { _mm_blendv_epi8(b.0, a.0, m.0) })
        }

        #[inline(always)]
        pub fn any(self) -> bool {
            unsafe { _mm_testz_si128(self.0, self.0) == 0 }
        }

        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            Self(unsafe { _mm_add_epi8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn add_sat_i(self, o: Self) -> Self {
            Self(unsafe { _mm_adds_epi8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn sub_sat_i(self, o: Self) -> Self {
            Self(unsafe { _mm_subs_epi8(self.0, o.0) })
        }

        /// The unsigned shift, its sign bit extended by `(v ^ m) - m`.
        #[inline(always)]
        pub fn shr_i<const N: i32>(self) -> Self {
            let m = Self::splat(0x80 >> N);
            Self(unsafe { _mm_sub_epi8(self.shr_u::<N>().xor(m).0, m.0) })
        }

        #[inline(always)]
        pub fn widen(self) -> [I16x8; 2] {
            unsafe {
                [
                    I16x8(_mm_cvtepu8_epi16(self.0)),
                    I16x8(_mm_unpackhi_epi8(self.0, _mm_setzero_si128())),
                ]
            }
        }

        #[inline(always)]
        pub fn narrow_rshr<const N: i32>(lo: I16x8, hi: I16x8) -> Self {
            unsafe {
                let r = _mm_set1_epi16(1 << (N - 1));
                Self(_mm_packus_epi16(
                    _mm_srai_epi16::<N>(_mm_add_epi16(lo.0, r)),
                    _mm_srai_epi16::<N>(_mm_add_epi16(hi.0, r)),
                ))
            }
        }

        #[inline(always)]
        pub fn zip_lo8(self, o: Self) -> Self {
            Self(unsafe { _mm_unpacklo_epi8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn zip_hi8(self, o: Self) -> Self {
            Self(unsafe { _mm_unpackhi_epi8(self.0, o.0) })
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

    use super::{I16x8, I32x4, U8x16};

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

        /// `lo` then `hi` shifted right by `N`, saturated to `i16`.
        #[inline(always)]
        pub fn pack_shr<const N: i32>(lo: Self, hi: Self) -> I16x8 {
            I16x8(unsafe { vqshrn_high_n_s32::<N>(vqshrn_n_s32::<N>(lo.0), hi.0) })
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

        /// Bytes `b[..8]` as `i8`, sign-extended.
        #[inline(always)]
        pub fn from_i8x8(b: &[u8]) -> Self {
            Self(unsafe { vmovl_s8(vld1_s8(b[..8].as_ptr().cast())) })
        }

        /// `(b[2 * i] + b[2 * i + 1] + 1) >> 1` of `b[..16]`.
        #[inline(always)]
        pub fn avg_pairs_u8(b: &[u8]) -> Self {
            Self(s(unsafe { vrshrq_n_u16::<1>(vpaddlq_u8(vld1q_u8(b[..16].as_ptr()))) }))
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

        /// The high byte of each lane.
        #[inline(always)]
        pub fn hi_u8(self) -> [u8; 8] {
            let mut out = [0; 8];
            unsafe { vst1_u8(out.as_mut_ptr(), vshrn_n_u16::<8>(u(self.0))) };
            out
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

        /// Whether any lane is nonzero.
        #[inline(always)]
        pub fn any(self) -> bool {
            unsafe { vmaxvq_u16(u(self.0)) != 0 }
        }

        /// The sum of the lanes as `u16`.
        #[inline(always)]
        pub fn hsum_u(self) -> u32 {
            unsafe { vaddlvq_u16(u(self.0)) }
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

    /// dav1d's `transpose_8x16b`: `v[i]` holding rows `i` and `i + 8` in its halves become the
    /// eight columns of sixteen, and back. 24 `trn`s where interleaves take 32.
    #[inline(always)]
    pub fn transpose_8x16b(v: [U8x16; 8]) -> [U8x16; 8] {
        let v = v.map(|v| v.0);
        unsafe {
            let h = |v: uint8x16_t| vreinterpretq_u16_u8(v);
            let w = |v: uint16x8_t| vreinterpretq_u32_u16(v);
            let (t8, t9) = (vtrn1q_u8(v[0], v[1]), vtrn2q_u8(v[0], v[1]));
            let (r1, r3) = (vtrn1q_u8(v[2], v[3]), vtrn2q_u8(v[2], v[3]));
            let (r0, r5) = (vtrn1q_u8(v[4], v[5]), vtrn2q_u8(v[4], v[5]));
            let (r2, r7) = (vtrn1q_u8(v[6], v[7]), vtrn2q_u8(v[6], v[7]));
            let (r4, r2) = (vtrn1q_u16(h(r0), h(r2)), vtrn2q_u16(h(r0), h(r2)));
            let (r6, r7) = (vtrn1q_u16(h(r5), h(r7)), vtrn2q_u16(h(r5), h(r7)));
            let (r5, t9) = (vtrn1q_u16(h(t9), h(r3)), vtrn2q_u16(h(t9), h(r3)));
            let (r3, t8) = (vtrn1q_u16(h(t8), h(r1)), vtrn2q_u16(h(t8), h(r1)));
            let c = [
                vtrn1q_u32(w(r3), w(r4)),
                vtrn1q_u32(w(r5), w(r6)),
                vtrn1q_u32(w(t8), w(r2)),
                vtrn1q_u32(w(t9), w(r7)),
                vtrn2q_u32(w(r3), w(r4)),
                vtrn2q_u32(w(r5), w(r6)),
                vtrn2q_u32(w(t8), w(r2)),
                vtrn2q_u32(w(t9), w(r7)),
            ];
            c.map(|c| U8x16(vreinterpretq_u8_u32(c)))
        }
    }

    /// dav1d's `transpose_4x16b`: `v[j]` holding rows `j`, `j + 4`, `j + 8` and `j + 12` in its
    /// quarters become the four columns of sixteen, and back.
    #[inline(always)]
    pub fn transpose_4x16b(v: [U8x16; 4]) -> [U8x16; 4] {
        let v = v.map(|v| v.0);
        unsafe {
            let h = |v: uint8x16_t| vreinterpretq_u16_u8(v);
            let (t4, t5) = (vtrn1q_u8(v[0], v[1]), vtrn2q_u8(v[0], v[1]));
            let (t6, t7) = (vtrn1q_u8(v[2], v[3]), vtrn2q_u8(v[2], v[3]));
            let c = [
                vtrn1q_u16(h(t4), h(t6)),
                vtrn1q_u16(h(t5), h(t7)),
                vtrn2q_u16(h(t4), h(t6)),
                vtrn2q_u16(h(t5), h(t7)),
            ];
            c.map(|c| U8x16(vreinterpretq_u8_u16(c)))
        }
    }

    // SAFETY (every `unsafe` here): NEON is part of aarch64.
    impl I16x8 {
        /// The sums of adjacent lanes, of `self` then of `o`.
        #[inline(always)]
        pub fn hadd(self, o: Self) -> Self {
            Self(unsafe { vpaddq_s16(self.0, o.0) })
        }

        /// Bytes `B..B + 16` of `self` then `o`: `o`'s lanes shifted in from the top.
        #[inline(always)]
        pub fn ext<const B: i32>(self, o: Self) -> Self {
            unsafe {
                let (a, b) = (vreinterpretq_s8_s16(self.0), vreinterpretq_s8_s16(o.0));
                Self(vreinterpretq_s16_s8(vextq_s8::<B>(a, b)))
            }
        }

    }

    impl I32x4 {
        /// The sums of adjacent lanes, of `self` then of `o`.
        #[inline(always)]
        pub fn hadd(self, o: Self) -> Self {
            Self(unsafe { vpaddq_s32(self.0, o.0) })
        }
    }

    impl I16x8 {
        /// `self² + o²` in 32 bits, low four lanes then high four.
        #[inline(always)]
        pub fn sq2(self, o: Self) -> [I32x4; 2] {
            let s = self.mul_wide(self);
            o.sq_acc(s)
        }

        /// `acc + self² + o²`.
        #[inline(always)]
        pub fn sq2_acc(self, o: Self, acc: [I32x4; 2]) -> [I32x4; 2] {
            o.sq_acc(self.sq_acc(acc))
        }

        /// `acc + self²`.
        #[inline(always)]
        pub fn sq_acc(self, acc: [I32x4; 2]) -> [I32x4; 2] {
            unsafe {
                [
                    I32x4(vmlal_s16(acc[0].0, vget_low_s16(self.0), vget_low_s16(self.0))),
                    I32x4(vmlal_high_s16(acc[1].0, self.0, self.0)),
                ]
            }
        }

        /// `(v[0] * k[0] + .. + v[N - 1] * k[N - 1] + rnd) >> S` in 32 bits, saturated to `i16`.
        #[inline(always)]
        pub fn madd_n<const S: i32, const N: usize>(v: [Self; N], k: [i16; N], rnd: i32) -> Self {
            unsafe {
                let mut lo = vdupq_n_s32(rnd);
                let mut hi = lo;
                for i in 0..N {
                    lo = vmlal_n_s16(lo, vget_low_s16(v[i].0), k[i]);
                    hi = vmlal_high_n_s16(hi, v[i].0, k[i]);
                }
                Self(vcombine_s16(
                    vqmovn_s32(vshrq_n_s32::<S>(lo)),
                    vqmovn_s32(vshrq_n_s32::<S>(hi)),
                ))
            }
        }
    }

    pub type Inner8 = uint8x16_t;

    /// A 256-byte table loaded for [`U8x16::lookup256`].
    #[derive(Clone, Copy)]
    pub struct Lut256([uint8x16x4_t; 4]);

    impl Lut256 {
        #[inline(always)]
        pub fn new(t: &[u8; 256]) -> Self {
            Self(std::array::from_fn(|k| unsafe {
                vld1q_u8_x4(t[64 * k..].as_ptr())
            }))
        }
    }

    #[inline(always)]
    fn s8(v: uint8x16_t) -> int8x16_t {
        unsafe { vreinterpretq_s8_u8(v) }
    }

    #[inline(always)]
    fn u8s(v: int8x16_t) -> uint8x16_t {
        unsafe { vreinterpretq_u8_s8(v) }
    }

    // SAFETY (every `unsafe` here): NEON is part of aarch64, and loads and stores go through
    // bounds-checked slices of 16 or 8 bytes.
    impl U8x16 {
        /// # Safety
        ///
        /// The 16 bytes at `p` are readable.
        #[inline(always)]
        pub unsafe fn load_at(p: *const u8) -> Self {
            Self(unsafe { vld1q_u8(p) })
        }

        /// # Safety
        ///
        /// The 8 bytes at `p` are readable.
        #[inline(always)]
        pub unsafe fn load_lo_at(p: *const u8) -> Self {
            Self(unsafe { vcombine_u8(vld1_u8(p), vdup_n_u8(0)) })
        }

        /// # Safety
        ///
        /// The 16 bytes at `p` are writable.
        #[inline(always)]
        pub unsafe fn store_at(self, p: *mut u8) {
            unsafe { vst1q_u8(p, self.0) }
        }

        /// # Safety
        ///
        /// The 8 bytes at `lo` and `hi` are writable.
        #[inline(always)]
        pub unsafe fn store_halves_at(self, lo: *mut u8, hi: *mut u8) {
            unsafe {
                vst1_u8(lo, vget_low_u8(self.0));
                vst1_u8(hi, vget_high_u8(self.0));
            }
        }

        /// Bytes `4 * L..4 * L + 4`.
        #[inline(always)]
        pub fn lane32<const L: i32>(self) -> u32 {
            unsafe { vgetq_lane_u32::<L>(vreinterpretq_u32_u8(self.0)) }
        }

        /// # Safety
        ///
        /// The 8 bytes at `lo` and at `hi` are readable.
        #[inline(always)]
        pub unsafe fn load_halves_at(lo: *const u8, hi: *const u8) -> Self {
            Self(unsafe { vcombine_u8(vld1_u8(lo), vld1_u8(hi)) })
        }

        /// # Safety
        ///
        /// The 4 bytes at each `p[l]` are readable.
        #[inline(always)]
        pub unsafe fn load_lanes32_at(p: [*const u8; 4]) -> Self {
            // SAFETY: The caller's; the reads are unaligned.
            let x = p.map(|p| unsafe { p.cast::<u32>().read_unaligned() });
            unsafe {
                let v = vdupq_n_u32(x[0]);
                let v = vsetq_lane_u32::<1>(x[1], v);
                let v = vsetq_lane_u32::<2>(x[2], v);
                Self(vreinterpretq_u8_u32(vsetq_lane_u32::<3>(x[3], v)))
            }
        }

        /// # Safety
        ///
        /// The 4 bytes at each `p[l]` are writable.
        #[inline(always)]
        pub unsafe fn store_lanes32_at(self, p: [*mut u8; 4]) {
            let x = unsafe {
                let v = vreinterpretq_u32_u8(self.0);
                [
                    vgetq_lane_u32::<0>(v),
                    vgetq_lane_u32::<1>(v),
                    vgetq_lane_u32::<2>(v),
                    vgetq_lane_u32::<3>(v),
                ]
            };
            for (p, x) in p.into_iter().zip(x) {
                // SAFETY: The caller's; the writes are unaligned.
                unsafe { p.cast::<u32>().write_unaligned(x) };
            }
        }

        /// `x` in the low four lanes, the others 0.
        #[inline(always)]
        pub fn from_u32(x: u32) -> Self {
            Self(unsafe { vreinterpretq_u8_u32(vsetq_lane_u32::<0>(x, vdupq_n_u32(0))) })
        }

        /// `t[lane]` for lanes below 64, 0 for the others.
        #[inline(always)]
        pub fn lookup64(self, t: &[u8; 64]) -> Self {
            Self(unsafe { vqtbl4q_u8(vld1q_u8_x4(t.as_ptr()), self.0) })
        }

        #[inline(always)]
        pub fn splat(v: u8) -> Self {
            Self(unsafe { vdupq_n_u8(v) })
        }

        #[inline(always)]
        pub fn abd(self, o: Self) -> Self {
            Self(unsafe { vabdq_u8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            Self(unsafe { vmaxq_u8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn add_sat(self, o: Self) -> Self {
            Self(unsafe { vqaddq_u8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn shr_u<const N: i32>(self) -> Self {
            Self(unsafe { vshrq_n_u8::<N>(self.0) })
        }

        #[inline(always)]
        pub fn le(self, o: Self) -> Self {
            Self(unsafe { vcleq_u8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn eq(self, o: Self) -> Self {
            Self(unsafe { vceqq_u8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn and(self, o: Self) -> Self {
            Self(unsafe { vandq_u8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn andnot(self, o: Self) -> Self {
            Self(unsafe { vbicq_u8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn xor(self, o: Self) -> Self {
            Self(unsafe { veorq_u8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn select(m: Self, a: Self, b: Self) -> Self {
            Self(unsafe { vbslq_u8(m.0, a.0, b.0) })
        }

        #[inline(always)]
        pub fn any(self) -> bool {
            unsafe { vmaxvq_u8(self.0) != 0 }
        }

        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            Self(unsafe { vaddq_u8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn add_sat_i(self, o: Self) -> Self {
            Self(u8s(unsafe { vqaddq_s8(s8(self.0), s8(o.0)) }))
        }

        #[inline(always)]
        pub fn sub_sat_i(self, o: Self) -> Self {
            Self(u8s(unsafe { vqsubq_s8(s8(self.0), s8(o.0)) }))
        }

        #[inline(always)]
        pub fn shr_i<const N: i32>(self) -> Self {
            Self(u8s(unsafe { vshrq_n_s8::<N>(s8(self.0)) }))
        }

        #[inline(always)]
        pub fn widen(self) -> [I16x8; 2] {
            unsafe {
                [
                    I16x8(s(vmovl_u8(vget_low_u8(self.0)))),
                    I16x8(s(vmovl_high_u8(self.0))),
                ]
            }
        }

        #[inline(always)]
        pub fn narrow_rshr<const N: i32>(lo: I16x8, hi: I16x8) -> Self {
            Self(unsafe { vqrshrun_high_n_s16::<N>(vqrshrun_n_s16::<N>(lo.0), hi.0) })
        }

        /// `lo` then `hi` saturated to `u8`.
        #[inline(always)]
        pub fn pack(lo: I16x8, hi: I16x8) -> Self {
            Self(unsafe { vqmovun_high_s16(vqmovun_s16(lo.0), hi.0) })
        }

        /// `t[lane]`: one `tbl` and three `tbx` over the table in sixteen registers.
        #[inline(always)]
        pub fn lookup256(self, t: &Lut256) -> Self {
            unsafe {
                let k = vdupq_n_u8(64);
                let i1 = vsubq_u8(self.0, k);
                let i2 = vsubq_u8(i1, k);
                let i3 = vsubq_u8(i2, k);
                let r = vqtbl4q_u8(t.0[0], self.0);
                let r = vqtbx4q_u8(r, t.0[1], i1);
                let r = vqtbx4q_u8(r, t.0[2], i2);
                Self(vqtbx4q_u8(r, t.0[3], i3))
            }
        }

        #[inline(always)]
        pub fn zip_lo8(self, o: Self) -> Self {
            Self(unsafe { vzip1q_u8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn zip_hi8(self, o: Self) -> Self {
            Self(unsafe { vzip2q_u8(self.0, o.0) })
        }

        #[inline(always)]
        pub fn zip_lo16(self, o: Self) -> Self {
            Self(unsafe {
                vreinterpretq_u8_u16(vzip1q_u16(
                    vreinterpretq_u16_u8(self.0),
                    vreinterpretq_u16_u8(o.0),
                ))
            })
        }

        #[inline(always)]
        pub fn zip_hi16(self, o: Self) -> Self {
            Self(unsafe {
                vreinterpretq_u8_u16(vzip2q_u16(
                    vreinterpretq_u16_u8(self.0),
                    vreinterpretq_u16_u8(o.0),
                ))
            })
        }

        #[inline(always)]
        pub fn zip_lo32(self, o: Self) -> Self {
            Self(unsafe {
                vreinterpretq_u8_u32(vzip1q_u32(
                    vreinterpretq_u32_u8(self.0),
                    vreinterpretq_u32_u8(o.0),
                ))
            })
        }

        #[inline(always)]
        pub fn zip_hi32(self, o: Self) -> Self {
            Self(unsafe {
                vreinterpretq_u8_u32(vzip2q_u32(
                    vreinterpretq_u32_u8(self.0),
                    vreinterpretq_u32_u8(o.0),
                ))
            })
        }

        #[inline(always)]
        pub fn zip_lo64(self, o: Self) -> Self {
            Self(unsafe { vcombine_u8(vget_low_u8(self.0), vget_low_u8(o.0)) })
        }

        #[inline(always)]
        pub fn zip_hi64(self, o: Self) -> Self {
            Self(unsafe { vcombine_u8(vget_high_u8(self.0), vget_high_u8(o.0)) })
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod imp {
    use std::arch::wasm32::*;

    use super::{I16x8, I32x4, U8x16};

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

        /// `lo` then `hi` shifted right by `N`, saturated to `i16`.
        #[inline(always)]
        pub fn pack_shr<const N: i32>(lo: Self, hi: Self) -> I16x8 {
            Self::pack(lo.shr_s::<N>(), hi.shr_s::<N>())
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

        /// Bytes `b[..8]` as `i8`, sign-extended.
        #[inline(always)]
        pub fn from_i8x8(b: &[u8]) -> Self {
            // SAFETY: 8 bytes are in bounds; the load takes unaligned pointers.
            Self(unsafe { i16x8_load_extend_i8x8(b[..8].as_ptr().cast()) })
        }

        /// `(b[2 * i] + b[2 * i + 1] + 1) >> 1` of `b[..16]`.
        #[inline(always)]
        pub fn avg_pairs_u8(b: &[u8]) -> Self {
            let v = u8x16(
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13],
                b[14], b[15],
            );
            Self(u16x8_avgr(u16x8_extadd_pairwise_u8x16(v), u16x8_splat(0)))
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

        /// The high byte of each lane.
        #[inline(always)]
        pub fn hi_u8(self) -> [u8; 8] {
            Self(u16x8_shr(self.0, 8)).pack_u8()
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

        /// Whether any lane is nonzero.
        #[inline(always)]
        pub fn any(self) -> bool {
            v128_any_true(self.0)
        }

        /// The sum of the lanes as `u16`.
        #[inline(always)]
        pub fn hsum_u(self) -> u32 {
            let s = u32x4_extadd_pairwise_u16x8(self.0);
            u32x4_extract_lane::<0>(s)
                + u32x4_extract_lane::<1>(s)
                + u32x4_extract_lane::<2>(s)
                + u32x4_extract_lane::<3>(s)
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

    pub type Inner8 = v128;

    impl U8x16 {
        /// # Safety
        ///
        /// The 16 bytes at `p` are readable.
        #[inline(always)]
        pub unsafe fn load_at(p: *const u8) -> Self {
            // SAFETY: The caller's; the load takes unaligned pointers.
            Self(unsafe { v128_load(p.cast()) })
        }

        /// # Safety
        ///
        /// The 8 bytes at `p` are readable.
        #[inline(always)]
        pub unsafe fn load_lo_at(p: *const u8) -> Self {
            // SAFETY: The caller's; the load takes unaligned pointers.
            Self(unsafe { v128_load64_zero(p.cast()) })
        }

        /// # Safety
        ///
        /// The 16 bytes at `p` are writable.
        #[inline(always)]
        pub unsafe fn store_at(self, p: *mut u8) {
            // SAFETY: The caller's; the store takes unaligned pointers.
            unsafe { v128_store(p.cast(), self.0) }
        }

        /// # Safety
        ///
        /// The 8 bytes at `lo` and `hi` are writable.
        #[inline(always)]
        pub unsafe fn store_halves_at(self, lo: *mut u8, hi: *mut u8) {
            // SAFETY: The caller's; the stores take unaligned pointers.
            unsafe {
                v128_store64_lane::<0>(self.0, lo.cast());
                v128_store64_lane::<1>(self.0, hi.cast());
            }
        }

        /// Bytes `4 * L..4 * L + 4`.
        #[inline(always)]
        pub fn lane32<const L: i32>(self) -> u32 {
            u32x4_extract_lane::<{ L as usize }>(self.0)
        }

        /// `x` in the low four lanes, the others 0.
        #[inline(always)]
        pub fn from_u32(x: u32) -> Self {
            Self(u32x4(x, 0, 0, 0))
        }

        /// `t[lane]` for lanes below 64, 0 for the others.
        #[inline(always)]
        pub fn lookup64(self, t: &[u8; 64]) -> Self {
            let mut r = u8x16_splat(0);
            for k in 0..4 {
                // SAFETY: 16 bytes are in bounds; the load takes unaligned pointers.
                let t = unsafe { v128_load(t[16 * k..].as_ptr().cast()) };
                // Indices outside this 16 wrap to 16 or more, which the swizzle zeroes.
                r = v128_or(
                    r,
                    i8x16_swizzle(t, u8x16_sub(self.0, u8x16_splat(16 * k as u8))),
                );
            }
            Self(r)
        }

        #[inline(always)]
        pub fn splat(v: u8) -> Self {
            Self(u8x16_splat(v))
        }

        #[inline(always)]
        pub fn abd(self, o: Self) -> Self {
            Self(v128_or(
                u8x16_sub_sat(self.0, o.0),
                u8x16_sub_sat(o.0, self.0),
            ))
        }

        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            Self(u8x16_max(self.0, o.0))
        }

        #[inline(always)]
        pub fn add_sat(self, o: Self) -> Self {
            Self(u8x16_add_sat(self.0, o.0))
        }

        #[inline(always)]
        pub fn shr_u<const N: i32>(self) -> Self {
            Self(u8x16_shr(self.0, N as u32))
        }

        #[inline(always)]
        pub fn le(self, o: Self) -> Self {
            Self(u8x16_le(self.0, o.0))
        }

        #[inline(always)]
        pub fn eq(self, o: Self) -> Self {
            Self(i8x16_eq(self.0, o.0))
        }

        #[inline(always)]
        pub fn and(self, o: Self) -> Self {
            Self(v128_and(self.0, o.0))
        }

        #[inline(always)]
        pub fn andnot(self, o: Self) -> Self {
            Self(v128_andnot(self.0, o.0))
        }

        #[inline(always)]
        pub fn xor(self, o: Self) -> Self {
            Self(v128_xor(self.0, o.0))
        }

        #[inline(always)]
        pub fn select(m: Self, a: Self, b: Self) -> Self {
            Self(v128_bitselect(a.0, b.0, m.0))
        }

        #[inline(always)]
        pub fn any(self) -> bool {
            v128_any_true(self.0)
        }

        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            Self(u8x16_add(self.0, o.0))
        }

        #[inline(always)]
        pub fn add_sat_i(self, o: Self) -> Self {
            Self(i8x16_add_sat(self.0, o.0))
        }

        #[inline(always)]
        pub fn sub_sat_i(self, o: Self) -> Self {
            Self(i8x16_sub_sat(self.0, o.0))
        }

        #[inline(always)]
        pub fn shr_i<const N: i32>(self) -> Self {
            Self(i8x16_shr(self.0, N as u32))
        }

        #[inline(always)]
        pub fn widen(self) -> [I16x8; 2] {
            [
                I16x8(u16x8_extend_low_u8x16(self.0)),
                I16x8(u16x8_extend_high_u8x16(self.0)),
            ]
        }

        #[inline(always)]
        pub fn narrow_rshr<const N: i32>(lo: I16x8, hi: I16x8) -> Self {
            let r = i16x8_splat(1 << (N - 1));
            Self(u8x16_narrow_i16x8(
                i16x8_shr(i16x8_add(lo.0, r), N as u32),
                i16x8_shr(i16x8_add(hi.0, r), N as u32),
            ))
        }

        #[inline(always)]
        pub fn zip_lo8(self, o: Self) -> Self {
            Self(i8x16_shuffle::<
                0,
                16,
                1,
                17,
                2,
                18,
                3,
                19,
                4,
                20,
                5,
                21,
                6,
                22,
                7,
                23,
            >(self.0, o.0))
        }

        #[inline(always)]
        pub fn zip_hi8(self, o: Self) -> Self {
            Self(i8x16_shuffle::<
                8,
                24,
                9,
                25,
                10,
                26,
                11,
                27,
                12,
                28,
                13,
                29,
                14,
                30,
                15,
                31,
            >(self.0, o.0))
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
    use super::{I16x8, I32x4, U8x16};

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

        /// `lo` then `hi` shifted right by `N`, saturated to `i16`.
        #[inline(always)]
        pub fn pack_shr<const N: i32>(lo: Self, hi: Self) -> I16x8 {
            Self::pack(lo.shr_s::<N>(), hi.shr_s::<N>())
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

        /// Bytes `b[..8]` as `i8`, sign-extended.
        #[inline(always)]
        pub fn from_i8x8(b: &[u8]) -> Self {
            Self(std::array::from_fn(|i| b[i] as i8 as i16))
        }

        /// `(b[2 * i] + b[2 * i + 1] + 1) >> 1` of `b[..16]`.
        #[inline(always)]
        pub fn avg_pairs_u8(b: &[u8]) -> Self {
            Self(std::array::from_fn(|i| {
                ((b[2 * i] as u16 + b[2 * i + 1] as u16 + 1) >> 1) as i16
            }))
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

        /// The high byte of each lane.
        #[inline(always)]
        pub fn hi_u8(self) -> [u8; 8] {
            self.0.map(|v| (v as u16 >> 8) as u8)
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

        /// Whether any lane is nonzero.
        #[inline(always)]
        pub fn any(self) -> bool {
            self.0.iter().any(|&m| m != 0)
        }

        /// The sum of the lanes as `u16`.
        #[inline(always)]
        pub fn hsum_u(self) -> u32 {
            self.0.iter().map(|&v| v as u16 as u32).sum()
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
            (
                Self(std::array::from_fn(|i| at(2 * i))),
                Self(std::array::from_fn(|i| at(2 * i + 1))),
            )
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

    pub type Inner8 = [u8; 16];

    impl U8x16 {
        #[inline(always)]
        fn zip(self, o: Self, f: impl Fn(u8, u8) -> u8) -> Self {
            Self(std::array::from_fn(|i| f(self.0[i], o.0[i])))
        }

        #[inline(always)]
        fn zip_i(self, o: Self, f: impl Fn(i8, i8) -> i8) -> Self {
            self.zip(o, |a, b| f(a as i8, b as i8) as u8)
        }

        /// The interleave of `E`-byte elements from the low (`hi == 0`) or high half.
        #[inline(always)]
        fn interleave<const E: usize>(self, o: Self, hi: usize) -> Self {
            Self(std::array::from_fn(|i| {
                let e = i / E;
                [self.0, o.0][e & 1][(hi * 8 / E + e / 2) * E + i % E]
            }))
        }

        /// # Safety
        ///
        /// The 16 bytes at `p` are readable.
        #[inline(always)]
        pub unsafe fn load_at(p: *const u8) -> Self {
            // SAFETY: The caller's.
            Self(unsafe { p.cast::<[u8; 16]>().read_unaligned() })
        }

        /// # Safety
        ///
        /// The 8 bytes at `p` are readable.
        #[inline(always)]
        pub unsafe fn load_lo_at(p: *const u8) -> Self {
            // SAFETY: The caller's.
            let lo = unsafe { p.cast::<[u8; 8]>().read_unaligned() };
            Self(std::array::from_fn(|i| if i < 8 { lo[i] } else { 0 }))
        }

        /// # Safety
        ///
        /// The 16 bytes at `p` are writable.
        #[inline(always)]
        pub unsafe fn store_at(self, p: *mut u8) {
            // SAFETY: The caller's.
            unsafe { p.cast::<[u8; 16]>().write_unaligned(self.0) }
        }

        /// # Safety
        ///
        /// The 8 bytes at `lo` and `hi` are writable.
        #[inline(always)]
        pub unsafe fn store_halves_at(self, lo: *mut u8, hi: *mut u8) {
            let a: [u8; 8] = self.0[..8].try_into().unwrap();
            let b: [u8; 8] = self.0[8..].try_into().unwrap();
            // SAFETY: The caller's.
            unsafe {
                lo.cast::<[u8; 8]>().write_unaligned(a);
                hi.cast::<[u8; 8]>().write_unaligned(b);
            }
        }

        /// Bytes `4 * L..4 * L + 4`.
        #[inline(always)]
        pub fn lane32<const L: i32>(self) -> u32 {
            let l = 4 * L as usize;
            u32::from_le_bytes(self.0[l..l + 4].try_into().unwrap())
        }

        /// `x` in the low four lanes, the others 0.
        #[inline(always)]
        pub fn from_u32(x: u32) -> Self {
            let x = x.to_le_bytes();
            Self(std::array::from_fn(|i| if i < 4 { x[i] } else { 0 }))
        }

        /// `t[lane]` for lanes below 64, 0 for the others.
        #[inline(always)]
        pub fn lookup64(self, t: &[u8; 64]) -> Self {
            Self(self.0.map(|v| t.get(v as usize).copied().unwrap_or(0)))
        }

        #[inline(always)]
        pub fn splat(v: u8) -> Self {
            Self([v; 16])
        }

        #[inline(always)]
        pub fn abd(self, o: Self) -> Self {
            self.zip(o, u8::abs_diff)
        }

        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            self.zip(o, u8::max)
        }

        #[inline(always)]
        pub fn add_sat(self, o: Self) -> Self {
            self.zip(o, u8::saturating_add)
        }

        #[inline(always)]
        pub fn shr_u<const N: i32>(self) -> Self {
            Self(self.0.map(|v| v >> N))
        }

        #[inline(always)]
        pub fn le(self, o: Self) -> Self {
            self.zip(o, |a, b| if a <= b { 0xff } else { 0 })
        }

        #[inline(always)]
        pub fn eq(self, o: Self) -> Self {
            self.zip(o, |a, b| if a == b { 0xff } else { 0 })
        }

        #[inline(always)]
        pub fn and(self, o: Self) -> Self {
            self.zip(o, |a, b| a & b)
        }

        #[inline(always)]
        pub fn andnot(self, o: Self) -> Self {
            self.zip(o, |a, b| a & !b)
        }

        #[inline(always)]
        pub fn xor(self, o: Self) -> Self {
            self.zip(o, |a, b| a ^ b)
        }

        #[inline(always)]
        pub fn select(m: Self, a: Self, b: Self) -> Self {
            Self(std::array::from_fn(|i| a.0[i] & m.0[i] | b.0[i] & !m.0[i]))
        }

        #[inline(always)]
        pub fn any(self) -> bool {
            self.0.iter().any(|&m| m != 0)
        }

        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            self.zip(o, u8::wrapping_add)
        }

        #[inline(always)]
        pub fn add_sat_i(self, o: Self) -> Self {
            self.zip_i(o, i8::saturating_add)
        }

        #[inline(always)]
        pub fn sub_sat_i(self, o: Self) -> Self {
            self.zip_i(o, i8::saturating_sub)
        }

        #[inline(always)]
        pub fn shr_i<const N: i32>(self) -> Self {
            Self(self.0.map(|v| (v as i8 >> N) as u8))
        }

        #[inline(always)]
        pub fn widen(self) -> [I16x8; 2] {
            [
                I16x8(std::array::from_fn(|i| self.0[i] as i16)),
                I16x8(std::array::from_fn(|i| self.0[8 + i] as i16)),
            ]
        }

        #[inline(always)]
        pub fn narrow_rshr<const N: i32>(lo: I16x8, hi: I16x8) -> Self {
            Self(std::array::from_fn(|i| {
                let v = [lo.0, hi.0][i >> 3][i & 7] as i32;
                ((v + (1 << (N - 1))) >> N).clamp(0, 255) as u8
            }))
        }

        #[inline(always)]
        pub fn zip_lo8(self, o: Self) -> Self {
            self.interleave::<1>(o, 0)
        }

        #[inline(always)]
        pub fn zip_hi8(self, o: Self) -> Self {
            self.interleave::<1>(o, 1)
        }

        #[inline(always)]
        pub fn zip_lo16(self, o: Self) -> Self {
            self.interleave::<2>(o, 0)
        }

        #[inline(always)]
        pub fn zip_hi16(self, o: Self) -> Self {
            self.interleave::<2>(o, 1)
        }

        #[inline(always)]
        pub fn zip_lo32(self, o: Self) -> Self {
            self.interleave::<4>(o, 0)
        }

        #[inline(always)]
        pub fn zip_hi32(self, o: Self) -> Self {
            self.interleave::<4>(o, 1)
        }

        #[inline(always)]
        pub fn zip_lo64(self, o: Self) -> Self {
            self.interleave::<8>(o, 0)
        }

        #[inline(always)]
        pub fn zip_hi64(self, o: Self) -> Self {
            self.interleave::<8>(o, 1)
        }
    }
}

/// The lane-reordering operations by value on wasm and the scalar stand-in.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
impl I16x8 {
    /// `self² + o²` in 32 bits, low four lanes then high four.
    #[inline(always)]
    pub fn sq2(self, o: Self) -> [I32x4; 2] {
        o.sq_acc(self.mul_wide(self))
    }

    /// `acc + self² + o²`.
    #[inline(always)]
    pub fn sq2_acc(self, o: Self, acc: [I32x4; 2]) -> [I32x4; 2] {
        o.sq_acc(self.sq_acc(acc))
    }

    /// `acc + self²`.
    #[inline(always)]
    pub fn sq_acc(self, acc: [I32x4; 2]) -> [I32x4; 2] {
        let s = self.mul_wide(self);
        [acc[0].add(s[0]), acc[1].add(s[1])]
    }

    /// The sums of adjacent lanes, of `self` then of `o`.
    #[inline(always)]
    pub fn hadd(self, o: Self) -> Self {
        let (a, b) = (self.to_array(), o.to_array());
        let v: [i16; 8] = std::array::from_fn(|i| {
            let s = if i < 4 { &a } else { &b };
            s[2 * (i & 3)].wrapping_add(s[2 * (i & 3) + 1])
        });
        Self::load(&v)
    }

    /// Bytes `B..B + 16` of `self` then `o`: `o`'s lanes shifted in from the top.
    #[inline(always)]
    pub fn ext<const B: i32>(self, o: Self) -> Self {
        let (a, b) = (self.to_array(), o.to_array());
        let v: [i16; 8] = std::array::from_fn(|i| {
            let k = i + B as usize / 2;
            if k < 8 {
                a[k]
            } else {
                b[k - 8]
            }
        });
        Self::load(&v)
    }

    /// `(v[0] * k[0] + .. + v[N - 1] * k[N - 1] + rnd) >> S` in 32 bits, saturated to `i16`.
    #[inline(always)]
    pub fn madd_n<const S: i32, const N: usize>(v: [Self; N], k: [i16; N], rnd: i32) -> Self {
        let v = v.map(|v| v.to_array());
        let r: [i16; 8] = std::array::from_fn(|i| {
            let s = (0..N).fold(rnd, |s, j| s.wrapping_add(v[j][i] as i32 * k[j] as i32));
            (s >> S).clamp(i16::MIN as i32, i16::MAX as i32) as i16
        });
        Self::load(&r)
    }

}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
impl I32x4 {
    /// The sums of adjacent lanes, of `self` then of `o`.
    #[inline(always)]
    pub fn hadd(self, o: Self) -> Self {
        let (mut a, mut b) = ([0; 4], [0; 4]);
        self.store(&mut a);
        o.store(&mut b);
        Self::load(&[
            a[0].wrapping_add(a[1]),
            a[2].wrapping_add(a[3]),
            b[0].wrapping_add(b[1]),
            b[2].wrapping_add(b[3]),
        ])
    }
}
