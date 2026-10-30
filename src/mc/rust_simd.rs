//! Motion compensation in Rust SIMD: the 8-tap filters, the compound averages and masks, the
//! blends. Eight `i16` lanes on SSE4.1 and NEON, sixteen on AVX2. Blocks are walked in column
//! strips of one vector; the 8-tap filters keep the strip's eight source (or first-pass) rows in
//! registers, so the two-pass filter needs no intermediate buffer. 8 bpc sums stay in `i16` where
//! dav1d's asm keeps them there (the taps sum to 64), everything else is summed in 32 bits.
//! Widths below a vector (2 and 4, and 8 on AVX2) go through a short copy or the narrower tier.

use std::mem::size_of;
use std::slice;

use super::*;
use crate::include::dav1d::picture::PicRows;

/// The lane operations the kernels take; one implementation per tier.
pub trait Vx: Copy {
    const N: usize;
    fn splat(v: i16) -> Self;
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
    fn min(self, o: Self) -> Self;
    fn max(self, o: Self) -> Self;
    fn sra(self, n: u32) -> Self;
    fn load(s: &[i16]) -> Self;
    fn store(self, d: &mut [i16]);
    /// `s[..N]` zero-extended.
    fn load_u8(s: &[u8]) -> Self;
    /// The lanes saturated to `u8` into `d[..N]`.
    fn store_u8(self, d: &mut [u8]);
    fn load_u16(s: &[u16]) -> Self;
    fn store_u16(self, d: &mut [u16]);
    /// Two rows' lanes side by side, for `maddp`.
    type P: Copy;
    fn pair(a: Self, b: Self) -> Self::P;
    /// `(sum(a_i * k[2 * i] + b_i * k[2 * i + 1]) + rnd) >> sh` over the pairs `p[i] = (a_i, b_i)`,
    /// in 32 bits, saturated to `i16`.
    fn maddp(p: &[Self::P], k: &[i16; 8], rnd: i32, sh: u32) -> Self;
    /// Whether the vertical passes keep their rows paired (`maddp`), not as rows (`maddr`).
    const PAIRS: bool = true;
    /// `(sum(r[i] * k[i]) + rnd) >> sh` in 32 bits, saturated to `i16`, over 6 or 8 rows.
    #[inline(always)]
    fn maddr(r: &[Self], k: &[i16; 8], rnd: i32, sh: u32) -> Self {
        let mut p = [Self::pair(r[0], r[1]); 4];
        for i in 1..r.len() / 2 {
            p[i] = Self::pair(r[2 * i], r[2 * i + 1]);
        }
        Self::maddp(&p[..r.len() / 2], k, rnd, sh)
    }
    /// `(a * ka + b * kb + rnd) >> sh` lane by lane in 32 bits, saturated to `i16`.
    fn madd2(a: Self, b: Self, ka: Self, kb: Self, rnd: i32, sh: u32) -> Self;
    /// `sum(s[i + k] * f[k] for k in 0..8)` for the `N` lanes `i`, from 8 bpc `s[..N + 7]`;
    /// with `T6` over `k in 1..7`.
    fn h8<const T6: bool>(s: &[u8], f: &[i8; 8]) -> Self;
    /// `min(38 + (sat_u16(|a - b| + rnd) >> sh), 64)`, `w_mask`'s weight.
    fn wmask(self, o: Self, rnd: i16, sh: u32) -> Self;
    /// The sums of adjacent lanes, `a`'s then `b`'s.
    fn hadd(a: Self, b: Self) -> Self;

    #[inline(always)]
    fn rshr(self, rnd: i16, sh: u32) -> Self {
        self.add(Self::splat(rnd)).sra(sh)
    }

    /// `(self + (1 << SH >> 1)) >> SH`, exact for every lane value.
    fn rshr_c<const SH: i32>(self) -> Self;

    /// The block rows a vector holds: 2 where its lanes are two rows' strips of `N / 2`, the
    /// upper row in the low half. Only the 8-tap filters take two-row tiers.
    const ROWS: usize = 1;
    /// Two rows' `u8` strips, zero-extended.
    fn load2_u8(_: &[u8], _: &[u8]) -> Self {
        unreachable!()
    }
    fn load2_u16(_: &[u16], _: &[u16]) -> Self {
        unreachable!()
    }
    /// `h8` of two rows, from `a[..N / 2 + 7]` and `b[..N / 2 + 7]`.
    fn h8x2<const T6: bool>(_: &[u8], _: &[u8], _: &[i8; 8]) -> Self {
        unreachable!()
    }
    /// The second row of `a` and the first of `b`.
    fn mid(_: Self, _: Self) -> Self {
        unreachable!()
    }
    /// Row `I`'s lanes as `store_u8`, `store_u16` and `store` store them.
    #[inline(always)]
    fn store_row_u8<const I: usize>(self, d: &mut [u8]) {
        self.store_u8(d)
    }
    #[inline(always)]
    fn store_row_u16<const I: usize>(self, d: &mut [u16]) {
        self.store_u16(d)
    }
    #[inline(always)]
    fn store_row<const I: usize>(self, d: &mut [i16]) {
        self.store(d)
    }

    /// Whether the 8 bpc vertical pass multiplies byte pairs (`pairb`, `maddb`).
    const BYTE_PAIRS: bool = false;
    /// The `u8` lanes of `a` and `b` as the low and high bytes of each lane.
    fn pairb(_: Self, _: Self) -> Self {
        unreachable!()
    }
    /// `sum(p[i]'s bytes times (k[2 * i], k[2 * i + 1]))`, in `i16`.
    fn maddb(_: &[Self], _: &[i16; 8]) -> Self {
        unreachable!()
    }
}

/// The pixels of a strip's row: `N / ROWS`, or `PW`.
#[inline(always)]
const fn px<V: Vx, const PW: usize>() -> usize {
    if PW == 0 {
        V::N / V::ROWS
    } else {
        PW
    }
}

/// Two vectors as one of `2 * V::N` lanes: a strip twice as wide, its rows borrowed once.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub struct D<V>(V, V);

#[cfg(target_arch = "aarch64")]
impl<V: Vx> Vx for D<V> {
    const N: usize = 2 * V::N;
    type P = (V::P, V::P);
    const PAIRS: bool = V::PAIRS;

    #[inline(always)]
    fn splat(v: i16) -> Self {
        D(V::splat(v), V::splat(v))
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        D(self.0.add(o.0), self.1.add(o.1))
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        D(self.0.sub(o.0), self.1.sub(o.1))
    }
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        D(self.0.mul(o.0), self.1.mul(o.1))
    }
    #[inline(always)]
    fn min(self, o: Self) -> Self {
        D(self.0.min(o.0), self.1.min(o.1))
    }
    #[inline(always)]
    fn max(self, o: Self) -> Self {
        D(self.0.max(o.0), self.1.max(o.1))
    }
    #[inline(always)]
    fn sra(self, n: u32) -> Self {
        D(self.0.sra(n), self.1.sra(n))
    }
    #[inline(always)]
    fn load(s: &[i16]) -> Self {
        let (a, b) = s[..Self::N].split_at(V::N);
        D(V::load(a), V::load(b))
    }
    #[inline(always)]
    fn store(self, d: &mut [i16]) {
        let (a, b) = d[..Self::N].split_at_mut(V::N);
        self.0.store(a);
        self.1.store(b);
    }
    #[inline(always)]
    fn load_u8(s: &[u8]) -> Self {
        let (a, b) = s[..Self::N].split_at(V::N);
        D(V::load_u8(a), V::load_u8(b))
    }
    #[inline(always)]
    fn store_u8(self, d: &mut [u8]) {
        let (a, b) = d[..Self::N].split_at_mut(V::N);
        self.0.store_u8(a);
        self.1.store_u8(b);
    }
    #[inline(always)]
    fn load_u16(s: &[u16]) -> Self {
        let (a, b) = s[..Self::N].split_at(V::N);
        D(V::load_u16(a), V::load_u16(b))
    }
    #[inline(always)]
    fn store_u16(self, d: &mut [u16]) {
        let (a, b) = d[..Self::N].split_at_mut(V::N);
        self.0.store_u16(a);
        self.1.store_u16(b);
    }
    #[inline(always)]
    fn pair(a: Self, b: Self) -> Self::P {
        (V::pair(a.0, b.0), V::pair(a.1, b.1))
    }
    #[inline(always)]
    fn maddp(p: &[Self::P], k: &[i16; 8], rnd: i32, sh: u32) -> Self {
        let (mut lo, mut hi) = ([p[0].0; 4], [p[0].1; 4]);
        for (i, p) in p.iter().enumerate() {
            (lo[i], hi[i]) = *p;
        }
        let n = p.len();
        D(
            V::maddp(&lo[..n], k, rnd, sh),
            V::maddp(&hi[..n], k, rnd, sh),
        )
    }
    #[inline(always)]
    fn maddr(r: &[Self], k: &[i16; 8], rnd: i32, sh: u32) -> Self {
        let (mut lo, mut hi) = ([r[0].0; 8], [r[0].1; 8]);
        for (i, r) in r.iter().enumerate() {
            (lo[i], hi[i]) = (r.0, r.1);
        }
        let n = r.len();
        D(
            V::maddr(&lo[..n], k, rnd, sh),
            V::maddr(&hi[..n], k, rnd, sh),
        )
    }
    #[inline(always)]
    fn madd2(a: Self, b: Self, ka: Self, kb: Self, rnd: i32, sh: u32) -> Self {
        D(
            V::madd2(a.0, b.0, ka.0, kb.0, rnd, sh),
            V::madd2(a.1, b.1, ka.1, kb.1, rnd, sh),
        )
    }
    #[inline(always)]
    fn h8<const T6: bool>(s: &[u8], f: &[i8; 8]) -> Self {
        let s = &s[..Self::N + 7];
        D(V::h8::<T6>(s, f), V::h8::<T6>(&s[V::N..], f))
    }
    #[inline(always)]
    fn wmask(self, o: Self, rnd: i16, sh: u32) -> Self {
        D(self.0.wmask(o.0, rnd, sh), self.1.wmask(o.1, rnd, sh))
    }
    #[inline(always)]
    fn hadd(a: Self, b: Self) -> Self {
        D(V::hadd(a.0, a.1), V::hadd(b.0, b.1))
    }
    #[inline(always)]
    fn rshr_c<const SH: i32>(self) -> Self {
        D(self.0.rshr_c::<SH>(), self.1.rshr_c::<SH>())
    }
}

/// `(v + rnd) >> sh`, as one rounding shift where 8 bpc's constants make it one.
#[inline(always)]
fn rshr8<V: Vx>(v: V, rnd: i16, sh: u32) -> V {
    match (rnd, sh) {
        (0, 0) => v,
        (2, 2) => v.rshr_c::<2>(),
        (8, 4) => v.rshr_c::<4>(),
        (16, 5) => v.rshr_c::<5>(),
        (32, 6) => v.rshr_c::<6>(),
        (34, 6) => v.add(V::splat(2)).rshr_c::<6>(),
        _ => v.rshr(rnd, sh),
    }
}

#[inline(always)]
fn cast<P, T>(s: &[P]) -> &[T] {
    assert!(size_of::<P>() == size_of::<T>());
    // SAFETY: `P` and `T` are the same primitive pixel type (`u8` or `u16`) where this is called.
    unsafe { slice::from_raw_parts(s.as_ptr().cast(), s.len()) }
}

#[inline(always)]
fn cast_mut<P, T>(s: &mut [P]) -> &mut [T] {
    assert!(size_of::<P>() == size_of::<T>());
    // SAFETY: As in `cast`.
    unsafe { slice::from_raw_parts_mut(s.as_mut_ptr().cast(), s.len()) }
}

/// `N` lanes, or with `PW` the first `PW` of them through a copy.
#[inline(always)]
fn ld_u8<V: Vx, const PW: usize>(s: &[u8]) -> V {
    if PW == 0 {
        return V::load_u8(s);
    }
    let mut t = [0; 16];
    t[..PW].copy_from_slice(&s[..PW]);
    V::load_u8(&t)
}

#[inline(always)]
fn ld_u16<V: Vx, const PW: usize>(s: &[u16]) -> V {
    if PW == 0 {
        return V::load_u16(s);
    }
    let mut t = [0; 16];
    t[..PW].copy_from_slice(&s[..PW]);
    V::load_u16(&t)
}

#[inline(always)]
fn ld_i16<V: Vx, const PW: usize>(s: &[i16]) -> V {
    if PW == 0 {
        return V::load(s);
    }
    let mut t = [0; 16];
    t[..PW].copy_from_slice(&s[..PW]);
    V::load(&t)
}

#[inline(always)]
fn st_i16<V: Vx, const PW: usize>(v: V, d: &mut [i16]) {
    if PW == 0 {
        return v.store(d);
    }
    let mut t = [0; 16];
    v.store(&mut t);
    d[..PW].copy_from_slice(&t[..PW]);
}

/// The 8 bpc first pass over `s[..n + 7]`, `n` the strip's width; with `T6` taps 1 to 6 only.
#[inline(always)]
fn h8_u8<V: Vx, const PW: usize, const T6: bool>(s: &[u8], f: &[i8; 8]) -> V {
    if PW == 0 {
        return V::h8::<T6>(s, f);
    }
    let mut t = [0; 32];
    t[..PW + 7].copy_from_slice(&s[..PW + 7]);
    V::h8::<T6>(&t, f)
}

/// The 16 bpc first pass over `s[..n + 7]`, `(sum(s[i + k] * f[k]) + rnd) >> sh`, the windows
/// paired by taps; with `T6` taps 1 to 6 only, as `k` holds them (see `taps`).
#[inline(always)]
fn h_u16<V: Vx, const PW: usize, const T6: bool>(s: &[u16], k: &[i16; 8], rnd: i32, sh: u32) -> V {
    let mut t = [0; 32];
    let s = if PW == 0 {
        s
    } else {
        t[..PW + 7].copy_from_slice(&s[..PW + 7]);
        &t[..]
    };
    let w = [
        V::load_u16(&s[0..]),
        V::load_u16(&s[1..]),
        V::load_u16(&s[2..]),
        V::load_u16(&s[3..]),
        V::load_u16(&s[4..]),
        V::load_u16(&s[5..]),
        V::load_u16(&s[6..]),
        V::load_u16(&s[7..]),
    ];
    if T6 {
        V::maddr(&w[1..7], k, rnd, sh)
    } else {
        V::maddr(&w, k, rnd, sh)
    }
}

/// `h_u16` of two rows, as a two-row tier holds them.
#[inline(always)]
fn h_u16x2<V: Vx, const T6: bool>(a: &[u16], b: &[u16], k: &[i16; 8], rnd: i32, sh: u32) -> V {
    let w = [
        V::load2_u16(&a[0..], &b[0..]),
        V::load2_u16(&a[1..], &b[1..]),
        V::load2_u16(&a[2..], &b[2..]),
        V::load2_u16(&a[3..], &b[3..]),
        V::load2_u16(&a[4..], &b[4..]),
        V::load2_u16(&a[5..], &b[5..]),
        V::load2_u16(&a[6..], &b[6..]),
        V::load2_u16(&a[7..], &b[7..]),
    ];
    if T6 {
        V::maddr(&w[1..7], k, rnd, sh)
    } else {
        V::maddr(&w, k, rnd, sh)
    }
}

/// The filter's taps as `i16`; with `T6` (`f[0]` and `f[7]` are 0) taps 1 to 6 first.
#[inline(always)]
fn taps<const T6: bool>(f: &[i8; 8]) -> [i16; 8] {
    let o = T6 as usize;
    let mut k = [0; 8];
    for i in 0..8 - 2 * o {
        k[i] = f[i + o] as i16;
    }
    k
}

/// Whether the filter has six taps (regular and smooth, and the 4-tap ones of narrow blocks).
#[inline(always)]
fn six(f: &[i8; 8]) -> bool {
    f[0] == 0 && f[7] == 0
}

/// The 8 bpc column sum `sum(v[i] * k[i])` over `v`'s rows, in `i16` as it fits.
#[inline(always)]
fn mac<V: Vx>(v: &[V], k: &[i16; 8]) -> V {
    let mut a = v[0].mul(V::splat(k[0]));
    let mut b = v[1].mul(V::splat(k[1]));
    for i in (2..v.len()).step_by(2) {
        a = a.add(v[i].mul(V::splat(k[i])));
        b = b.add(v[i + 1].mul(V::splat(k[i + 1])));
    }
    a.add(b)
}

/// `v`'s lanes as pixels into `d`: saturated at 8 bpc, clipped to `[0, max]` above.
#[inline(always)]
fn store_px<BD: BitDepth, V: Vx, const PW: usize>(v: V, d: &mut [BD::Pixel], max: i16) {
    store_px_row::<BD, V, PW, 0>(v, d, max)
}

/// `store_px` of row `I` of `v`.
#[inline(always)]
fn store_px_row<BD: BitDepth, V: Vx, const PW: usize, const I: usize>(
    v: V,
    d: &mut [BD::Pixel],
    max: i16,
) {
    if BD::BITDEPTH == 8 {
        let d = cast_mut::<_, u8>(d);
        if PW == 0 {
            return v.store_row_u8::<I>(d);
        }
        let mut t = [0; 16];
        v.store_u8(&mut t);
        d[..PW].copy_from_slice(&t[..PW]);
    } else {
        let v = v.max(V::splat(0)).min(V::splat(max));
        let d = cast_mut::<_, u16>(d);
        if PW == 0 {
            return v.store_row_u16::<I>(d);
        }
        let mut t = [0; 16];
        v.store_u16(&mut t);
        d[..PW].copy_from_slice(&t[..PW]);
    }
}

/// Where a kernel's rows go: the picture (`put`), or the compound buffer (`prep`).
pub enum Out<'a, 'b, BD: BitDepth> {
    Put(PicRows<'a, BD>),
    Prep(&'b mut [i16]),
}

impl<BD: BitDepth> Out<'_, '_, BD> {
    /// Stores `v`'s strip at `(x, y)`: clipped to pixels for `put`, as is for `prep`.
    #[inline(always)]
    fn emit<V: Vx, const PW: usize>(&mut self, v: V, x: usize, y: usize, w: usize, max: i16) {
        let n = if PW == 0 { V::N } else { PW };
        match self {
            Out::Put(d) => {
                store_px::<BD, V, PW>(v, &mut d.row_mut(y as isize, x as isize, n), max);
            }
            Out::Prep(t) => st_i16::<V, PW>(v, &mut t[y * w + x..]),
        }
    }

    fn is_prep(&self) -> bool {
        matches!(self, Out::Prep(_))
    }
}

/// Where `mc_8tap` stores its rows: the picture (`put`) or the compound buffer (`prep`).
trait Sink<BD: BitDepth> {
    const PREP: bool;
    fn emit<V: Vx, const PW: usize>(&mut self, v: V, x: usize, y: usize, max: i16);
}

struct PutRows<'a, BD: BitDepth> {
    d: PicRows<'a, BD>,
}

struct PrepRows<'b> {
    t: &'b mut [i16],
    w: usize,
}

impl<BD: BitDepth> Sink<BD> for PutRows<'_, BD> {
    const PREP: bool = false;

    #[inline(always)]
    fn emit<V: Vx, const PW: usize>(&mut self, v: V, x: usize, y: usize, max: i16) {
        let n = px::<V, PW>();
        let (x, y) = (x as isize, y as isize);
        store_px_row::<BD, V, PW, 0>(v, &mut self.d.row_mut(y, x, n), max);
        if V::ROWS == 2 {
            store_px_row::<BD, V, PW, 1>(v, &mut self.d.row_mut(y + 1, x, n), max);
        }
    }
}

impl<BD: BitDepth> Sink<BD> for PrepRows<'_> {
    const PREP: bool = true;

    #[inline(always)]
    fn emit<V: Vx, const PW: usize>(&mut self, v: V, x: usize, y: usize, _: i16) {
        let v = v.sub(V::splat(BD::PREP_BIAS));
        let i = y * self.w + x;
        if V::ROWS == 2 {
            v.store_row::<0>(&mut self.t[i..]);
            return v.store_row::<1>(&mut self.t[i + self.w..]);
        }
        st_i16::<V, PW>(v, &mut self.t[i..][..px::<V, PW>()]);
    }
}

/// Source row `y` (from the block's top) of the strip at `x`.
#[inline(always)]
fn vrow<BD: BitDepth, V: Vx, const PW: usize>(
    src: &PicRows<BD>,
    y: isize,
    x: usize,
    n: usize,
) -> V {
    let s = src.row(y, x as isize, n);
    if BD::BITDEPTH == 8 {
        ld_u8::<V, PW>(cast(&s))
    } else {
        ld_u16::<V, PW>(cast(&s))
    }
}

/// The first pass of source row `y` (from the block's top) of the strip at `x`, rounded.
#[inline(always)]
fn hrow<BD: BitDepth, V: Vx, const PW: usize, const T6: bool>(
    src: &PicRows<BD>,
    y: isize,
    x: usize,
    n: usize,
    fh: &[i8; 8],
    kh: &[i16; 8],
    hrnd: i32,
    hsh: u32,
) -> V {
    let s = src.row(y, x as isize - 3, n + 7);
    if BD::BITDEPTH == 8 {
        rshr8(h8_u8::<V, PW, T6>(cast(&s), fh), hrnd as i16, hsh)
    } else {
        h_u16::<V, PW, T6>(cast(&s), kh, hrnd, hsh)
    }
}

/// Source rows `a` and `b` of the strip at `x`, in a two-row tier.
#[inline(always)]
fn vrow2<BD: BitDepth, V: Vx>(src: &PicRows<BD>, a: isize, b: isize, x: usize, n: usize) -> V {
    let sa = src.row(a, x as isize, n);
    let sb = src.row(b, x as isize, n);
    if BD::BITDEPTH == 8 {
        V::load2_u8(cast(&sa), cast(&sb))
    } else {
        V::load2_u16(cast(&sa), cast(&sb))
    }
}

/// The first pass of source rows `a` and `b` of the strip at `x`, in a two-row tier.
#[inline(always)]
fn hrow2<BD: BitDepth, V: Vx, const T6: bool>(
    src: &PicRows<BD>,
    a: isize,
    b: isize,
    x: usize,
    n: usize,
    fh: &[i8; 8],
    kh: &[i16; 8],
    hrnd: i32,
    hsh: u32,
) -> V {
    let sa = src.row(a, x as isize - 3, n + 7);
    let sb = src.row(b, x as isize - 3, n + 7);
    if BD::BITDEPTH == 8 {
        rshr8(V::h8x2::<T6>(cast(&sa), cast(&sb), fh), hrnd as i16, hsh)
    } else {
        h_u16x2::<V, T6>(cast(&sa), cast(&sb), kh, hrnd, hsh)
    }
}

/// The source rows of a strip's vertical pass: as they are, or through the first pass (`H`).
/// A closure would not inherit the tier's target features where it is not inlined.
struct Rows<'a, BD: BitDepth, const PW: usize, const H: bool, const T6: bool> {
    src: PicRows<'a, BD>,
    y0: isize,
    x: usize,
    n: usize,
    fh: [i8; 8],
    kh: [i16; 8],
    hrnd: i32,
    hsh: u32,
}

impl<BD: BitDepth, const PW: usize, const H: bool, const T6: bool> Rows<'_, BD, PW, H, T6> {
    /// Row `i` from the first tap of output row 0.
    #[inline(always)]
    fn get<V: Vx>(&self, i: usize) -> V {
        let y = i as isize + self.y0;
        if H {
            hrow::<BD, V, PW, T6>(
                &self.src, y, self.x, self.n, &self.fh, &self.kh, self.hrnd, self.hsh,
            )
        } else {
            vrow::<BD, V, PW>(&self.src, y, self.x, self.n)
        }
    }

    /// Rows `i` and `j` in a two-row tier.
    #[inline(always)]
    fn get2<V: Vx>(&self, i: usize, j: usize) -> V {
        let (a, b) = (i as isize + self.y0, j as isize + self.y0);
        if H {
            hrow2::<BD, V, T6>(
                &self.src, a, b, self.x, self.n, &self.fh, &self.kh, self.hrnd, self.hsh,
            )
        } else {
            vrow2::<BD, V>(&self.src, a, b, self.x, self.n)
        }
    }
}

/// `put_8tap` / `prep_8tap` with at least one subpel filter.
#[inline(always)]
pub fn mc_8tap<BD: BitDepth, V: Vx, const PW: usize>(
    out: Out<BD>,
    src: Rav1dPictureDataComponentOffset,
    w: usize,
    h: usize,
    fh: Option<&[i8; 8]>,
    fv: Option<&[i8; 8]>,
    bd: BD,
) {
    match out {
        Out::Put(d) => mc_8tap_to::<BD, V, PW, _>(&mut PutRows { d }, src, w, h, fh, fv, bd),
        Out::Prep(t) => mc_8tap_to::<BD, V, PW, _>(&mut PrepRows { t, w }, src, w, h, fh, fv, bd),
    }
}

#[inline(always)]
fn mc_8tap_to<BD: BitDepth, V: Vx, const PW: usize, S: Sink<BD>>(
    s: &mut S,
    src: Rav1dPictureDataComponentOffset,
    w: usize,
    h: usize,
    fh: Option<&[i8; 8]>,
    fv: Option<&[i8; 8]>,
    bd: BD,
) {
    let (mx, my) = (fh.is_some() as isize, fv.is_some() as isize);
    // SAFETY: The 8-tap passes read rows -3..h + 3 and columns -3..w + 4 around the block, the
    // 6-tap ones less, each only along a filter it has.
    let src = unsafe {
        src.block_unchecked::<BD>(-3 * my..h as isize + 4 * my, -3 * mx..w as isize + 4 * mx)
    };
    match (fh, fv) {
        (Some(fh), None) if six(fh) => h_pass::<BD, V, PW, S, true>(s, src, w, h, fh, bd),
        (Some(fh), None) => h_pass::<BD, V, PW, S, false>(s, src, w, h, fh, bd),
        (None, Some(fv)) if six(fv) => v_pass::<BD, V, PW, S, true>(s, src, w, h, fv, bd),
        (None, Some(fv)) => v_pass::<BD, V, PW, S, false>(s, src, w, h, fv, bd),
        (Some(fh), Some(fv)) => match (six(fh), six(fv)) {
            (true, true) => hv_pass::<BD, V, PW, S, true, true>(s, src, w, h, fh, fv, bd),
            (true, false) => hv_pass::<BD, V, PW, S, true, false>(s, src, w, h, fh, fv, bd),
            (false, true) => hv_pass::<BD, V, PW, S, false, true>(s, src, w, h, fh, fv, bd),
            (false, false) => hv_pass::<BD, V, PW, S, false, false>(s, src, w, h, fh, fv, bd),
        },
        (None, None) => unreachable!(),
    }
}

#[inline(always)]
fn h_pass<BD: BitDepth, V: Vx, const PW: usize, S: Sink<BD>, const T6: bool>(
    s: &mut S,
    src: PicRows<BD>,
    w: usize,
    h: usize,
    fh: &[i8; 8],
    bd: BD,
) {
    let ib = bd.get_intermediate_bits() as u32;
    let max = bd.bitdepth_max().as_::<i32>() as i16;
    let n = px::<V, PW>();
    let hsh = 6 - ib;
    let hrnd = 1 << (hsh - 1);
    // By value, not behind the reference: stores could alias that, and keep the taps in the loop.
    let f = *fh;
    let k = taps::<T6>(&f);
    // `put`: `(sum + 32 + hrnd) >> 6`, `prep`: `(sum + hrnd) >> hsh`.
    let (rnd, sh) = if S::PREP { (hrnd, hsh) } else { (32 + hrnd, 6) };
    if V::ROWS == 2 {
        for x in (0..w).step_by(n) {
            let row = Rows::<BD, PW, true, T6> {
                src,
                y0: 0,
                x,
                n,
                fh: f,
                kh: k,
                hrnd: rnd,
                hsh: sh,
            };
            for y in (0..h).step_by(2) {
                s.emit::<V, PW>(row.get2(y, y + 1), x, y, max);
            }
        }
        return;
    }
    for y in 0..h {
        let r = src.row(y as isize, -3, w + 7);
        for x in (0..w).step_by(n) {
            let v = if BD::BITDEPTH == 8 {
                rshr8(
                    h8_u8::<V, PW, T6>(&cast::<_, u8>(&r)[x..][..n + 7], &f),
                    rnd as i16,
                    sh,
                )
            } else {
                h_u16::<V, PW, T6>(&cast::<_, u16>(&r)[x..][..n + 7], &k, rnd, sh)
            };
            s.emit::<V, PW>(v, x, y, max);
        }
    }
}

#[inline(always)]
fn v_pass<BD: BitDepth, V: Vx, const PW: usize, S: Sink<BD>, const T6: bool>(
    s: &mut S,
    src: PicRows<BD>,
    w: usize,
    h: usize,
    fv: &[i8; 8],
    bd: BD,
) {
    let ib = bd.get_intermediate_bits() as u32;
    let max = bd.bitdepth_max().as_::<i32>() as i16;
    let n = px::<V, PW>();
    let hsh = 6 - ib;
    let hrnd = 1 << (hsh - 1);
    let k = taps::<T6>(fv);
    let (rnd, sh) = if S::PREP { (hrnd, hsh) } else { (32, 6) };
    let y0 = T6 as isize - 3;
    for x in (0..w).step_by(n) {
        let row = Rows::<BD, PW, false, false> {
            src,
            y0,
            x,
            n,
            fh: *fv,
            kh: k,
            hrnd: 0,
            hsh: 0,
        };
        if BD::BITDEPTH == 8 && V::BYTE_PAIRS {
            vpairs::<BD, V, PW, S, Bytes, T6, false, false>(s, x, h, &row, &k, rnd, sh, max);
        } else if BD::BITDEPTH == 8 {
            vmac::<BD, V, PW, S, T6, false, false>(s, x, h, &row, &k, rnd as i16, sh, max);
        } else {
            v32::<BD, V, PW, S, T6, false, false>(s, x, h, &row, &k, rnd, sh, max);
        }
    }
}

#[inline(always)]
fn hv_pass<BD: BitDepth, V: Vx, const PW: usize, S: Sink<BD>, const H6: bool, const V6: bool>(
    s: &mut S,
    src: PicRows<BD>,
    w: usize,
    h: usize,
    fh: &[i8; 8],
    fv: &[i8; 8],
    bd: BD,
) {
    let ib = bd.get_intermediate_bits() as u32;
    let max = bd.bitdepth_max().as_::<i32>() as i16;
    let n = px::<V, PW>();
    let hsh = 6 - ib;
    let hrnd = 1 << (hsh - 1);
    let kh = taps::<H6>(fh);
    let kv = taps::<V6>(fv);
    let (rnd, sh) = if S::PREP {
        (32, 6)
    } else {
        (1 << (5 + ib), 6 + ib)
    };
    let y0 = V6 as isize - 3;
    for x in (0..w).step_by(n) {
        let row = Rows::<BD, PW, true, H6> {
            src,
            y0,
            x,
            n,
            fh: *fh,
            kh,
            hrnd,
            hsh,
        };
        v32::<BD, V, PW, S, V6, H6, true>(s, x, h, &row, &kv, rnd, sh, max);
    }
}

/// The vertical pass in 32 bits: `vpairs` or `vrows`, as the tier keeps rows.
#[inline(always)]
fn v32<
    BD: BitDepth,
    V: Vx,
    const PW: usize,
    S: Sink<BD>,
    const T6: bool,
    const RT6: bool,
    const H: bool,
>(
    s: &mut S,
    x: usize,
    h: usize,
    row: &Rows<BD, PW, H, RT6>,
    k: &[i16; 8],
    rnd: i32,
    sh: u32,
    max: i16,
) {
    if V::PAIRS {
        vpairs::<BD, V, PW, S, Words, T6, RT6, H>(s, x, h, row, k, rnd, sh, max)
    } else {
        vrows::<BD, V, PW, S, T6, RT6, H>(s, x, h, row, k, rnd, sh, max)
    }
}

/// The vertical pass in 32 bits over a window of rows, two rows a step, as `vmac`.
#[inline(always)]
fn vrows<
    BD: BitDepth,
    V: Vx,
    const PW: usize,
    S: Sink<BD>,
    const T6: bool,
    const RT6: bool,
    const H: bool,
>(
    s: &mut S,
    x: usize,
    h: usize,
    row: &Rows<BD, PW, H, RT6>,
    k: &[i16; 8],
    rnd: i32,
    sh: u32,
    max: i16,
) {
    let t = if T6 { 6 } else { 8 };
    let mut r = [V::splat(0); 9];
    if V::ROWS == 2 {
        let mut last = row.get2::<V>(0, 0);
        for i in 0..t / 2 - 1 {
            let a = row.get2(2 * i + 1, 2 * i + 2);
            (r[2 * i], r[2 * i + 1]) = (V::mid(last, a), a);
            last = a;
        }
        for y in (0..h).step_by(2) {
            let a = row.get2(y + t - 1, y + t);
            (r[t - 2], r[t - 1]) = (V::mid(last, a), a);
            s.emit::<V, PW>(V::maddr(&r[..t], k, rnd, sh), x, y, max);
            r = [r[2], r[3], r[4], r[5], r[6], r[7], r[7], r[7], r[7]];
            last = a;
        }
        return;
    }
    for i in 0..t - 1 {
        r[i] = row.get(i);
    }
    let mut y = 0;
    while y + 2 <= h {
        r[t - 1] = row.get(y + t - 1);
        r[t] = row.get(y + t);
        s.emit::<V, PW>(V::maddr(&r[..t], k, rnd, sh), x, y, max);
        s.emit::<V, PW>(V::maddr(&r[1..t + 1], k, rnd, sh), x, y + 1, max);
        // Rebuilt, not shifted in a loop: a loop becomes a memmove through the stack.
        r = [r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[8], r[8]];
        y += 2;
    }
    if y < h {
        r[t - 1] = row.get(y + t - 1);
        s.emit::<V, PW>(V::maddr(&r[..t], k, rnd, sh), x, y, max);
    }
}

/// The 8 bpc vertical pass in `i16`, two rows a step, over the strip's source rows `row`
/// from the first tap of output row 0.
#[inline(always)]
fn vmac<
    BD: BitDepth,
    V: Vx,
    const PW: usize,
    S: Sink<BD>,
    const T6: bool,
    const RT6: bool,
    const H: bool,
>(
    s: &mut S,
    x: usize,
    h: usize,
    row: &Rows<BD, PW, H, RT6>,
    k: &[i16; 8],
    rnd: i16,
    sh: u32,
    max: i16,
) {
    let t = if T6 { 6 } else { 8 };
    let mut r = [V::splat(0); 9];
    if V::ROWS == 2 {
        let mut last = row.get2::<V>(0, 0);
        for i in 0..t / 2 - 1 {
            let a = row.get2(2 * i + 1, 2 * i + 2);
            (r[2 * i], r[2 * i + 1]) = (V::mid(last, a), a);
            last = a;
        }
        for y in (0..h).step_by(2) {
            let a = row.get2(y + t - 1, y + t);
            (r[t - 2], r[t - 1]) = (V::mid(last, a), a);
            s.emit::<V, PW>(rshr8(mac(&r[..t], k), rnd, sh), x, y, max);
            r = [r[2], r[3], r[4], r[5], r[6], r[7], r[7], r[7], r[7]];
            last = a;
        }
        return;
    }
    for i in 0..t - 1 {
        r[i] = row.get(i);
    }
    let mut y = 0;
    while y + 2 <= h {
        r[t - 1] = row.get(y + t - 1);
        r[t] = row.get(y + t);
        let a = rshr8(mac(&r[..t], k), rnd, sh);
        let b = rshr8(mac(&r[1..t + 1], k), rnd, sh);
        s.emit::<V, PW>(a, x, y, max);
        s.emit::<V, PW>(b, x, y + 1, max);
        // Rebuilt, not shifted in a loop: a loop becomes a memmove through the stack.
        r = [r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[8], r[8]];
        y += 2;
    }
    if y < h {
        r[t - 1] = row.get(y + t - 1);
        s.emit::<V, PW>(rshr8(mac(&r[..t], k), rnd, sh), x, y, max);
    }
}

/// How `vpairs` sums rows: 32-bit products of `i16` pairs, or (8 bpc) `i16` sums of products of
/// byte pairs.
trait Acc<V: Vx> {
    type P: Copy;
    fn pair(a: V, b: V) -> Self::P;
    fn madd(p: &[Self::P], k: &[i16; 8], rnd: i32, sh: u32) -> V;
}

struct Words;

impl<V: Vx> Acc<V> for Words {
    type P = V::P;

    #[inline(always)]
    fn pair(a: V, b: V) -> V::P {
        V::pair(a, b)
    }

    #[inline(always)]
    fn madd(p: &[V::P], k: &[i16; 8], rnd: i32, sh: u32) -> V {
        V::maddp(p, k, rnd, sh)
    }
}

struct Bytes;

impl<V: Vx> Acc<V> for Bytes {
    type P = V;

    #[inline(always)]
    fn pair(a: V, b: V) -> V {
        V::pairb(a, b)
    }

    #[inline(always)]
    fn madd(p: &[V], k: &[i16; 8], rnd: i32, sh: u32) -> V {
        rshr8(V::maddb(p, k), rnd as i16, sh)
    }
}

/// The vertical pass over row pairs, two rows a step, over `row` as `vmac`. Rows are taken in
/// pairs, `e` for the even output row, `o` for the odd one; the next step keeps all pairs but
/// the first of each. A two-row tier's vectors hold rows `(j, j + 1)`: pairs of them make
/// both output rows, `mid` the vectors between those loaded.
#[inline(always)]
fn vpairs<
    BD: BitDepth,
    V: Vx,
    const PW: usize,
    S: Sink<BD>,
    K: Acc<V>,
    const T6: bool,
    const RT6: bool,
    const H: bool,
>(
    s: &mut S,
    x: usize,
    h: usize,
    row: &Rows<BD, PW, H, RT6>,
    k: &[i16; 8],
    rnd: i32,
    sh: u32,
    max: i16,
) {
    let t = if T6 { 6 } else { 8 };
    let m = t / 2 - 1;
    let z = K::pair(V::splat(0), V::splat(0));
    let (mut e, mut o) = ([z; 4], [z; 4]);
    if V::ROWS == 2 {
        let mut last = row.get2::<V>(0, 0);
        for i in 0..m {
            let a = row.get2(2 * i + 1, 2 * i + 2);
            e[i] = K::pair(V::mid(last, a), a);
            last = a;
        }
        for y in (0..h).step_by(2) {
            let a = row.get2(y + t - 1, y + t);
            e[m] = K::pair(V::mid(last, a), a);
            s.emit::<V, PW>(K::madd(&e[..m + 1], k, rnd, sh), x, y, max);
            e = [e[1], e[2], e[3], e[3]];
            last = a;
        }
        return;
    }
    let mut last = row.get(0);
    for i in 0..m {
        let a = row.get(2 * i + 1);
        let b = row.get(2 * i + 2);
        e[i] = K::pair(last, a);
        o[i] = K::pair(a, b);
        last = b;
    }
    let mut y = 0;
    while y + 2 <= h {
        let a = row.get(y + t - 1);
        let b = row.get(y + t);
        e[m] = K::pair(last, a);
        o[m] = K::pair(a, b);
        s.emit::<V, PW>(K::madd(&e[..m + 1], k, rnd, sh), x, y, max);
        s.emit::<V, PW>(K::madd(&o[..m + 1], k, rnd, sh), x, y + 1, max);
        e = [e[1], e[2], e[3], e[3]];
        o = [o[1], o[2], o[3], o[3]];
        last = b;
        y += 2;
    }
    if y < h {
        e[m] = K::pair(last, row.get(y + t - 1));
        s.emit::<V, PW>(K::madd(&e[..m + 1], k, rnd, sh), x, y, max);
    }
}

#[derive(Clone, Copy)]
pub enum Comp<'a> {
    Avg,
    WAvg(i16),
    Mask(&'a [u8]),
}

/// `avg`, `w_avg` and `mask`.
#[inline(always)]
pub fn comp<BD: BitDepth, V: Vx, const PW: usize>(
    dst: Rav1dPictureDataComponentOffset,
    t1: &[i16],
    t2: &[i16],
    w: usize,
    h: usize,
    kind: Comp,
    bd: BD,
) {
    match kind {
        Comp::Avg => comp_k::<BD, V, PW, 0>(dst, t1, t2, w, h, 0, &[], bd),
        Comp::WAvg(k) => comp_k::<BD, V, PW, 1>(dst, t1, t2, w, h, k, &[], bd),
        Comp::Mask(m) => comp_k::<BD, V, PW, 2>(dst, t1, t2, w, h, 0, m, bd),
    }
}

/// `comp` for `Comp` number `K`, walking each row in exact chunks of a vector.
#[inline(always)]
fn comp_k<BD: BitDepth, V: Vx, const PW: usize, const K: u8>(
    dst: Rav1dPictureDataComponentOffset,
    t1: &[i16],
    t2: &[i16],
    w: usize,
    h: usize,
    wt: i16,
    m: &[u8],
    bd: BD,
) {
    let ib = bd.get_intermediate_bits() as u32;
    let max = bd.bitdepth_max().as_::<i32>() as i16;
    let bias = BD::PREP_BIAS as i32;
    let n = if PW == 0 { V::N } else { PW };
    // SAFETY: The strips write the block's `h` rows of `w` pixels.
    let dst = unsafe { dst.block_unchecked::<BD>(0..h as isize, 0..w as isize) };
    let (rnd, sh) = match K {
        0 => ((1 << ib) + 2 * bias, ib + 1),
        1 => ((8 << ib) + 16 * bias, ib + 4),
        _ => ((32 << ib) + 64 * bias, ib + 6),
    };
    let (ka, kb) = if K == 0 {
        (V::splat(1), V::splat(1))
    } else {
        (V::splat(wt), V::splat(16 - wt))
    };
    let (t1, t2) = (&t1[..w * h], &t2[..w * h]);
    let m = if K == 2 { &m[..w * h] } else { &[] };
    if V::ROWS == 2 {
        // A strip is the block: two rows of the compound buffers lie side by side.
        for (y, (a, b)) in t1.chunks_exact(V::N).zip(t2.chunks_exact(V::N)).enumerate() {
            let (a, b) = (V::load(a), V::load(b));
            let v = match K {
                0 if BD::BITDEPTH == 8 => a.add(b).rshr_c::<5>(),
                0 | 1 => V::madd2(a, b, ka, kb, rnd, sh),
                _ => {
                    let m = V::load_u8(&m[y * V::N..][..V::N]);
                    V::madd2(a, b, m, V::splat(64).sub(m), rnd, sh)
                }
            };
            let y = 2 * y as isize;
            store_px_row::<BD, V, 0, 0>(v, &mut dst.row_mut(y, 0, w), max);
            store_px_row::<BD, V, 0, 1>(v, &mut dst.row_mut(y + 1, 0, w), max);
        }
        return;
    }
    for (y, (r1, r2)) in t1.chunks_exact(w).zip(t2.chunks_exact(w)).enumerate() {
        let row = &mut *dst.row_mut(y as isize, 0, w);
        let rm = if K == 2 { &m[y * w..][..w] } else { &[] };
        for (x, (d, (a, b))) in row
            .chunks_exact_mut(n)
            .zip(r1.chunks_exact(n).zip(r2.chunks_exact(n)))
            .enumerate()
        {
            let a = ld_i16::<V, PW>(a);
            let b = ld_i16::<V, PW>(b);
            let v = match K {
                0 if BD::BITDEPTH == 8 => a.add(b).rshr_c::<5>(),
                0 | 1 => V::madd2(a, b, ka, kb, rnd, sh),
                _ => {
                    let m = ld_u8::<V, PW>(&rm[x * n..][..n]);
                    V::madd2(a, b, m, V::splat(64).sub(m), rnd, sh)
                }
            };
            store_px::<BD, V, PW>(v, d, max);
        }
    }
}

/// The eight-lane operations the warp filter adds: per-pixel filters reduced across lanes.
pub trait Vw: Vx {
    fn load_i8(s: &[i8]) -> Self;
    fn srl(self, n: u32) -> Self;
    /// Lane `x` is the (wrapping) sum of `p[x]`'s lanes.
    fn hsum8(p: &[Self; 8]) -> Self;
    /// Lane `x` is `(sum(v[x] * f[x]) + rnd) >> sh` in 32 bits, saturated to `i16`.
    fn hdot8(v: &[Self; 8], f: &[Self; 8], rnd: i32, sh: u32) -> Self;
    /// `(sum(v[i] * k[i]) + rnd) >> sh` lane by lane in 32 bits, saturated to `i16`.
    fn maddv8(v: &[Self], k: &[Self; 8], rnd: i32, sh: u32) -> Self;
    fn transpose8(r: &mut [Self; 8]);
    /// Whether `warp_h8` takes its filters from `WARP_FILTER_PAIRED`.
    const PAIRED_WARP: bool = false;
    /// The 8 bpc first pass of one row: lane `x` is the (wrapping) `sum(s[x + k] * f[x][k])`,
    /// from `s[..15]`.
    #[inline(always)]
    fn warp_h8(s: &[u8], f: [&[i8; 8]; 8]) -> Self {
        let p = [
            Self::load_u8(&s[0..]).mul(Self::load_i8(f[0])),
            Self::load_u8(&s[1..]).mul(Self::load_i8(f[1])),
            Self::load_u8(&s[2..]).mul(Self::load_i8(f[2])),
            Self::load_u8(&s[3..]).mul(Self::load_i8(f[3])),
            Self::load_u8(&s[4..]).mul(Self::load_i8(f[4])),
            Self::load_u8(&s[5..]).mul(Self::load_i8(f[5])),
            Self::load_u8(&s[6..]).mul(Self::load_i8(f[6])),
            Self::load_u8(&s[7..]).mul(Self::load_i8(f[7])),
        ];
        Self::hsum8(&p)
    }
}

/// The warp filters with their taps in the order `0, 2, 1, 3, 4, 6, 5, 7`: pairs of taps two
/// apart, whose products of 8 bpc pixels sum within `i16` (`0, 1` and `2, 3` overflow).
static WARP_FILTER_PAIRED: [[i8; 8]; 193] = {
    let mut t = WARP_FILTER;
    let mut i = 0;
    while i < 193 {
        let f = WARP_FILTER[i];
        t[i] = [f[0], f[2], f[1], f[3], f[4], f[6], f[5], f[7]];
        i += 1;
    }
    t
};

/// The warp filters of the eight pixels, from position `m`, `step` apart, from `WARP_FILTER_PAIRED`
/// where `V::PAIRED_WARP`.
#[inline(always)]
fn warp_filter_refs<V: Vw>(m: i32, step: i32) -> [&'static [i8; 8]; 8] {
    let t = if V::PAIRED_WARP {
        &WARP_FILTER_PAIRED
    } else {
        &DAV1D_MC_WARP_FILTER.0
    };
    let f = |x: i32| &t[(64 + (m + x * step + 512 >> 10)) as usize];
    [f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7)]
}

#[inline(always)]
fn warp_filters<V: Vw>(m: i32, step: i32) -> [V; 8] {
    let f = |x: i32| -> V {
        V::load_i8(&DAV1D_MC_WARP_FILTER[(64 + (m + x * step + 512 >> 10)) as usize])
    };
    [f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7)]
}

/// `warp_affine_8x8` / `warp_affine_8x8t`: 15 first-pass rows with a filter per pixel, kept
/// in registers, then eight rows with a filter per column. `w` is `prep`'s row stride.
#[inline(always)]
pub fn warp<BD: BitDepth, V: Vw>(
    out: Out<BD>,
    w: usize,
    src: Rav1dPictureDataComponentOffset,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bd: BD,
) {
    let mid = warp_h::<BD, V>(src, abcd, mx, bd);
    warp_v::<BD, V>(out, w, &mid, abcd, my, bd)
}

/// The rounding of the vertical warp pass, `prep`'s or `put`'s.
#[inline(always)]
fn warp_rnd<BD: BitDepth>(prep: bool, bd: BD) -> (i32, u32) {
    let ib = bd.get_intermediate_bits() as u32;
    if prep {
        (64, 7)
    } else {
        (1 << (6 + ib), 7 + ib)
    }
}

/// The warp's 15 first-pass rows.
#[inline(always)]
fn warp_h<BD: BitDepth, V: Vw>(
    src: Rav1dPictureDataComponentOffset,
    abcd: &[i16; 4],
    mx: i32,
    bd: BD,
) -> [V; 15] {
    let ib = bd.get_intermediate_bits() as u32;
    // SAFETY: The first pass reads 15 rows of 15 pixels from (-3, -3).
    let src = unsafe { src.block_unchecked::<BD>(-3..12, -3..12) };
    let mut mid = [V::splat(0); 15];
    for y in 0..15 {
        let s = src.row(y as isize - 3, -3, 15);
        let (m, step) = (mx + y as i32 * abcd[1] as i32, abcd[0] as i32);
        mid[y] = if BD::BITDEPTH == 8 {
            let f = warp_filter_refs::<V>(m, step);
            // The sums span less than 16 bits once offset by `128 * 128` (the taps sum to 128):
            // `(sum + 4) >> 3` as an unsigned shift, the offset taken back.
            V::warp_h8(cast::<_, u8>(&s), f)
                .add(V::splat(16384 + 4))
                .srl(3)
                .sub(V::splat(2048))
        } else {
            let f = warp_filters::<V>(m, step);
            let s = cast::<_, u16>(&s);
            let v = [
                V::load_u16(&s[0..]),
                V::load_u16(&s[1..]),
                V::load_u16(&s[2..]),
                V::load_u16(&s[3..]),
                V::load_u16(&s[4..]),
                V::load_u16(&s[5..]),
                V::load_u16(&s[6..]),
                V::load_u16(&s[7..]),
            ];
            V::hdot8(&v, &f, 1 << (6 - ib), 7 - ib)
        };
    }
    mid
}

/// The warp's vertical pass over `mid`.
#[inline(always)]
fn warp_v<BD: BitDepth, V: Vw>(
    mut out: Out<BD>,
    w: usize,
    mid: &[V; 15],
    abcd: &[i16; 4],
    my: i32,
    bd: BD,
) {
    let max = bd.bitdepth_max().as_::<i32>() as i16;
    let prep = out.is_prep();
    let (rnd, sh) = warp_rnd(prep, bd);
    for y in 0..8 {
        let mut k = warp_filters::<V>(my + y as i32 * abcd[3] as i32, abcd[2] as i32);
        V::transpose8(&mut k);
        let v = V::maddv8(&mid[y..y + 8], &k, rnd, sh);
        let v = if prep {
            v.sub(V::splat(BD::PREP_BIAS))
        } else {
            v
        };
        out.emit::<V, 0>(v, 0, y, w, max);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use super::Vx;

    /// Eight lanes on SSE4.1; with `R2` two rows of four.
    #[derive(Clone, Copy)]
    pub struct S<const R2: bool = false>(__m128i);

    #[inline(always)]
    fn pair(a: i8, b: i8) -> i16 {
        (a as u8 as u16 | (b as u8 as u16) << 8) as i16
    }

    #[inline(always)]
    fn pairw(a: i16, b: i16) -> i32 {
        a as u16 as i32 | (b as i32) << 16
    }

    /// `s[..8]` and `s[7..15]`: the 15 bytes of eight outputs in two loads that stay inside
    /// them, byte 7 twice (see `sh15`).
    #[inline(always)]
    unsafe fn load15(s: &[u8]) -> __m128i {
        unsafe {
            let lo = _mm_loadl_epi64(s[..8].as_ptr().cast());
            _mm_unpacklo_epi64(lo, _mm_loadl_epi64(s[7..15].as_ptr().cast()))
        }
    }

    const SH_A: [i8; 16] = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8];

    #[inline(always)]
    fn sh(k: i8) -> [i8; 16] {
        let mut s = SH_A;
        let mut i = 0;
        while i < 16 {
            s[i] += k;
            i += 1;
        }
        s
    }

    /// `sh` over `load15`'s layout, where byte `j >= 8` sits at `j + 1`.
    #[inline(always)]
    fn sh15(k: i8) -> [i8; 16] {
        let mut s = sh(k);
        let mut i = 0;
        while i < 16 {
            s[i] += (s[i] >= 8) as i8;
            i += 1;
        }
        s
    }

    /// Pixel pairs `(i + k, i + k + 1)` times taps `(a, b)`, summed.
    #[inline(always)]
    unsafe fn tap(v: __m128i, k: i8, a: i8, b: i8) -> __m128i {
        let s = sh15(k);
        unsafe {
            _mm_maddubs_epi16(
                _mm_shuffle_epi8(v, _mm_loadu_si128(s.as_ptr().cast())),
                _mm_set1_epi16(pair(a, b)),
            )
        }
    }

    /// Pixels `(i + J, i + J + 2)` over `load15`'s layout times the tap pairs of pixel `i`
    /// in `k`, summed.
    #[inline(always)]
    unsafe fn ptap<const J: i8>(v: __m128i, k: __m128i) -> __m128i {
        let mut s = [0i8; 16];
        let mut i = 0;
        while i < 8 {
            let (a, b) = (i as i8 + J, i as i8 + J + 2);
            s[2 * i] = a + (a >= 8) as i8;
            s[2 * i + 1] = b + (b >= 8) as i8;
            i += 1;
        }
        unsafe { _mm_maddubs_epi16(_mm_shuffle_epi8(v, _mm_loadu_si128(s.as_ptr().cast())), k) }
    }

    /// `tap` on two rows in `load15`'s layout, one a lane.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn tap15(v: __m256i, k: i8, a: i8, b: i8) -> __m256i {
        let s = sh15(k);
        unsafe {
            let s = _mm256_loadu2_m128i(s.as_ptr().cast(), s.as_ptr().cast());
            _mm256_maddubs_epi16(_mm256_shuffle_epi8(v, s), _mm256_set1_epi16(pair(a, b)))
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn tap2(v: __m256i, k: i8, a: i8, b: i8) -> __m256i {
        let (lo, hi) = (sh(k), sh(k + 1));
        unsafe {
            let s = _mm256_loadu2_m128i(hi.as_ptr().cast(), lo.as_ptr().cast());
            _mm256_maddubs_epi16(_mm256_shuffle_epi8(v, s), _mm256_set1_epi16(pair(a, b)))
        }
    }

    #[inline(always)]
    unsafe fn ld64(s: &[u8]) -> __m128i {
        unsafe { _mm_loadl_epi64(s[..8].as_ptr().cast()) }
    }

    #[inline(always)]
    unsafe fn ld32(s: &[u8]) -> __m128i {
        unsafe { _mm_cvtsi32_si128(i32::from_le_bytes(s[..4].try_into().unwrap())) }
    }

    /// Tap pairs `(f[k], f[k + 1])` for the pixel pairs `(i + k, i + k + 1)` of 4 pixels in each
    /// 8-byte half of `v` (from pixel 2), summed.
    #[inline(always)]
    unsafe fn tap4(v: __m128i, k: i8, f: &[i8; 8]) -> __m128i {
        let mut s = [0i8; 16];
        let mut i = 0;
        while i < 8 {
            let j = (i & 4) as i8 * 2 + (i & 3) as i8 + k - 2;
            s[2 * i] = j;
            s[2 * i + 1] = j + 1;
            i += 1;
        }
        let k = k as usize;
        unsafe {
            _mm_maddubs_epi16(
                _mm_shuffle_epi8(v, _mm_loadu_si128(s.as_ptr().cast())),
                _mm_set1_epi16(pair(f[k], f[k + 1])),
            )
        }
    }

    // SAFETY (every `unsafe` here): SSE4.1 holds where these are compiled in, and loads and
    // stores go through bounds-checked slices.
    impl<const R2: bool> Vx for S<R2> {
        const N: usize = 8;
        const ROWS: usize = 1 + R2 as usize;
        const BYTE_PAIRS: bool = true;

        #[inline(always)]
        fn splat(v: i16) -> Self {
            Self(unsafe { _mm_set1_epi16(v) })
        }
        #[inline(always)]
        fn add(self, o: Self) -> Self {
            Self(unsafe { _mm_add_epi16(self.0, o.0) })
        }
        #[inline(always)]
        fn sub(self, o: Self) -> Self {
            Self(unsafe { _mm_sub_epi16(self.0, o.0) })
        }
        #[inline(always)]
        fn mul(self, o: Self) -> Self {
            Self(unsafe { _mm_mullo_epi16(self.0, o.0) })
        }
        #[inline(always)]
        fn min(self, o: Self) -> Self {
            Self(unsafe { _mm_min_epi16(self.0, o.0) })
        }
        #[inline(always)]
        fn max(self, o: Self) -> Self {
            Self(unsafe { _mm_max_epi16(self.0, o.0) })
        }
        #[inline(always)]
        fn sra(self, n: u32) -> Self {
            Self(unsafe { _mm_sra_epi16(self.0, _mm_cvtsi32_si128(n as i32)) })
        }
        #[inline(always)]
        fn load(s: &[i16]) -> Self {
            Self(unsafe { _mm_loadu_si128(s[..8].as_ptr().cast()) })
        }
        #[inline(always)]
        fn store(self, d: &mut [i16]) {
            unsafe { _mm_storeu_si128(d[..8].as_mut_ptr().cast(), self.0) }
        }
        #[inline(always)]
        fn load_u8(s: &[u8]) -> Self {
            Self(unsafe { _mm_cvtepu8_epi16(_mm_loadl_epi64(s[..8].as_ptr().cast())) })
        }
        #[inline(always)]
        fn store_u8(self, d: &mut [u8]) {
            unsafe {
                _mm_storel_epi64(d[..8].as_mut_ptr().cast(), _mm_packus_epi16(self.0, self.0))
            }
        }
        #[inline(always)]
        fn load_u16(s: &[u16]) -> Self {
            Self(unsafe { _mm_loadu_si128(s[..8].as_ptr().cast()) })
        }
        #[inline(always)]
        fn store_u16(self, d: &mut [u16]) {
            unsafe { _mm_storeu_si128(d[..8].as_mut_ptr().cast(), self.0) }
        }
        type P = (__m128i, __m128i);
        #[inline(always)]
        fn pair(a: Self, b: Self) -> Self::P {
            unsafe { (_mm_unpacklo_epi16(a.0, b.0), _mm_unpackhi_epi16(a.0, b.0)) }
        }
        #[inline(always)]
        fn maddp(p: &[Self::P], k: &[i16; 8], rnd: i32, sh: u32) -> Self {
            unsafe {
                let mut lo = _mm_set1_epi32(rnd);
                let mut hi = lo;
                for (i, p) in p.iter().enumerate() {
                    let k = _mm_set1_epi32(pairw(k[2 * i], k[2 * i + 1]));
                    lo = _mm_add_epi32(lo, _mm_madd_epi16(p.0, k));
                    hi = _mm_add_epi32(hi, _mm_madd_epi16(p.1, k));
                }
                let n = _mm_cvtsi32_si128(sh as i32);
                Self(_mm_packs_epi32(_mm_sra_epi32(lo, n), _mm_sra_epi32(hi, n)))
            }
        }
        #[inline(always)]
        fn madd2(a: Self, b: Self, ka: Self, kb: Self, rnd: i32, sh: u32) -> Self {
            unsafe {
                let r = _mm_set1_epi32(rnd);
                let lo =
                    _mm_madd_epi16(_mm_unpacklo_epi16(a.0, b.0), _mm_unpacklo_epi16(ka.0, kb.0));
                let hi =
                    _mm_madd_epi16(_mm_unpackhi_epi16(a.0, b.0), _mm_unpackhi_epi16(ka.0, kb.0));
                let n = _mm_cvtsi32_si128(sh as i32);
                Self(_mm_packs_epi32(
                    _mm_sra_epi32(_mm_add_epi32(lo, r), n),
                    _mm_sra_epi32(_mm_add_epi32(hi, r), n),
                ))
            }
        }
        #[inline(always)]
        fn h8<const T6: bool>(s: &[u8], f: &[i8; 8]) -> Self {
            unsafe {
                let v = load15(s);
                if T6 {
                    let a = _mm_add_epi16(tap(v, 1, f[1], f[2]), tap(v, 3, f[3], f[4]));
                    return Self(_mm_add_epi16(a, tap(v, 5, f[5], f[6])));
                }
                let a = _mm_add_epi16(tap(v, 0, f[0], f[1]), tap(v, 2, f[2], f[3]));
                let b = _mm_add_epi16(tap(v, 4, f[4], f[5]), tap(v, 6, f[6], f[7]));
                Self(_mm_add_epi16(a, b))
            }
        }
        #[inline(always)]
        fn rshr_c<const SH: i32>(self) -> Self {
            Self(unsafe { _mm_mulhrs_epi16(self.0, _mm_set1_epi16(1 << (15 - SH))) })
        }
        #[inline(always)]
        fn load2_u8(a: &[u8], b: &[u8]) -> Self {
            Self(unsafe { _mm_cvtepu8_epi16(_mm_unpacklo_epi32(ld32(a), ld32(b))) })
        }
        #[inline(always)]
        fn load2_u16(a: &[u16], b: &[u16]) -> Self {
            unsafe {
                let a = _mm_loadl_epi64(a[..4].as_ptr().cast());
                Self(_mm_unpacklo_epi64(
                    a,
                    _mm_loadl_epi64(b[..4].as_ptr().cast()),
                ))
            }
        }
        #[inline(always)]
        fn h8x2<const T6: bool>(a: &[u8], b: &[u8], f: &[i8; 8]) -> Self {
            // Four pixels a row take the 4-tap filters (taps 2 to 5).
            if T6 && f[1] == 0 && f[6] == 0 {
                unsafe {
                    let v = _mm_unpacklo_epi64(ld64(&a[2..]), ld64(&b[2..]));
                    return Self(_mm_add_epi16(tap4(v, 2, f), tap4(v, 4, f)));
                }
            }
            let (mut ta, mut tb) = ([0; 16], [0; 16]);
            ta[..11].copy_from_slice(&a[..11]);
            tb[..11].copy_from_slice(&b[..11]);
            let (ra, rb) = (S::<false>::h8::<T6>(&ta, f), S::<false>::h8::<T6>(&tb, f));
            Self(unsafe { _mm_unpacklo_epi64(ra.0, rb.0) })
        }
        #[inline(always)]
        fn mid(a: Self, b: Self) -> Self {
            Self(unsafe { _mm_alignr_epi8::<8>(b.0, a.0) })
        }
        #[inline(always)]
        fn store_row_u8<const I: usize>(self, d: &mut [u8]) {
            if !R2 {
                return self.store_u8(d);
            }
            unsafe {
                let p = _mm_packus_epi16(self.0, self.0);
                let p = if I == 0 { p } else { _mm_srli_si128::<4>(p) };
                d[..4].copy_from_slice(&_mm_cvtsi128_si32(p).to_le_bytes());
            }
        }
        #[inline(always)]
        fn store_row_u16<const I: usize>(self, d: &mut [u16]) {
            if !R2 {
                return self.store_u16(d);
            }
            unsafe {
                let v = if I == 0 {
                    self.0
                } else {
                    _mm_unpackhi_epi64(self.0, self.0)
                };
                _mm_storel_epi64(d[..4].as_mut_ptr().cast(), v)
            }
        }
        #[inline(always)]
        fn store_row<const I: usize>(self, d: &mut [i16]) {
            if !R2 {
                return self.store(d);
            }
            unsafe {
                let v = if I == 0 {
                    self.0
                } else {
                    _mm_unpackhi_epi64(self.0, self.0)
                };
                _mm_storel_epi64(d[..4].as_mut_ptr().cast(), v)
            }
        }
        #[inline(always)]
        fn pairb(a: Self, b: Self) -> Self {
            Self(unsafe { _mm_or_si128(a.0, _mm_slli_epi16::<8>(b.0)) })
        }
        #[inline(always)]
        fn maddb(p: &[Self], k: &[i16; 8]) -> Self {
            unsafe {
                let mut s = _mm_maddubs_epi16(p[0].0, _mm_set1_epi16(pair(k[0] as i8, k[1] as i8)));
                for (i, p) in p.iter().enumerate().skip(1) {
                    let k = _mm_set1_epi16(pair(k[2 * i] as i8, k[2 * i + 1] as i8));
                    s = _mm_add_epi16(s, _mm_maddubs_epi16(p.0, k));
                }
                Self(s)
            }
        }
        #[inline(always)]
        fn wmask(self, o: Self, rnd: i16, sh: u32) -> Self {
            unsafe {
                let d = _mm_sub_epi16(_mm_max_epi16(self.0, o.0), _mm_min_epi16(self.0, o.0));
                let d = _mm_adds_epu16(d, _mm_set1_epi16(rnd));
                let d = _mm_srl_epi16(d, _mm_cvtsi32_si128(sh as i32));
                Self(_mm_add_epi16(
                    _mm_min_epu16(d, _mm_set1_epi16(26)),
                    _mm_set1_epi16(38),
                ))
            }
        }
        #[inline(always)]
        fn hadd(a: Self, b: Self) -> Self {
            Self(unsafe { _mm_hadd_epi16(a.0, b.0) })
        }
    }

    // SAFETY (every `unsafe` here): as for `Vx`.
    impl super::Vw for S {
        #[inline(always)]
        fn load_i8(s: &[i8]) -> Self {
            Self(unsafe { _mm_cvtepi8_epi16(_mm_loadl_epi64(s[..8].as_ptr().cast())) })
        }
        #[inline(always)]
        fn srl(self, n: u32) -> Self {
            Self(unsafe { _mm_srl_epi16(self.0, _mm_cvtsi32_si128(n as i32)) })
        }
        #[inline(always)]
        fn hsum8(p: &[Self; 8]) -> Self {
            unsafe {
                let a = _mm_hadd_epi16(p[0].0, p[1].0);
                let b = _mm_hadd_epi16(p[2].0, p[3].0);
                let c = _mm_hadd_epi16(p[4].0, p[5].0);
                let d = _mm_hadd_epi16(p[6].0, p[7].0);
                Self(_mm_hadd_epi16(_mm_hadd_epi16(a, b), _mm_hadd_epi16(c, d)))
            }
        }
        #[inline(always)]
        fn hdot8(v: &[Self; 8], f: &[Self; 8], rnd: i32, sh: u32) -> Self {
            unsafe {
                let m = [
                    _mm_madd_epi16(v[0].0, f[0].0),
                    _mm_madd_epi16(v[1].0, f[1].0),
                    _mm_madd_epi16(v[2].0, f[2].0),
                    _mm_madd_epi16(v[3].0, f[3].0),
                    _mm_madd_epi16(v[4].0, f[4].0),
                    _mm_madd_epi16(v[5].0, f[5].0),
                    _mm_madd_epi16(v[6].0, f[6].0),
                    _mm_madd_epi16(v[7].0, f[7].0),
                ];
                let lo = _mm_hadd_epi32(_mm_hadd_epi32(m[0], m[1]), _mm_hadd_epi32(m[2], m[3]));
                let hi = _mm_hadd_epi32(_mm_hadd_epi32(m[4], m[5]), _mm_hadd_epi32(m[6], m[7]));
                let r = _mm_set1_epi32(rnd);
                let n = _mm_cvtsi32_si128(sh as i32);
                Self(_mm_packs_epi32(
                    _mm_sra_epi32(_mm_add_epi32(lo, r), n),
                    _mm_sra_epi32(_mm_add_epi32(hi, r), n),
                ))
            }
        }
        #[inline(always)]
        fn maddv8(v: &[Self], k: &[Self; 8], rnd: i32, sh: u32) -> Self {
            unsafe {
                let mut lo = _mm_set1_epi32(rnd);
                let mut hi = lo;
                for i in (0..8).step_by(2) {
                    let (a, b) = (v[i].0, v[i + 1].0);
                    let (ka, kb) = (k[i].0, k[i + 1].0);
                    lo = _mm_add_epi32(
                        lo,
                        _mm_madd_epi16(_mm_unpacklo_epi16(a, b), _mm_unpacklo_epi16(ka, kb)),
                    );
                    hi = _mm_add_epi32(
                        hi,
                        _mm_madd_epi16(_mm_unpackhi_epi16(a, b), _mm_unpackhi_epi16(ka, kb)),
                    );
                }
                let n = _mm_cvtsi32_si128(sh as i32);
                Self(_mm_packs_epi32(_mm_sra_epi32(lo, n), _mm_sra_epi32(hi, n)))
            }
        }
        #[inline(always)]
        fn transpose8(r: &mut [Self; 8]) {
            unsafe {
                let a0 = _mm_unpacklo_epi16(r[0].0, r[1].0);
                let a1 = _mm_unpackhi_epi16(r[0].0, r[1].0);
                let a2 = _mm_unpacklo_epi16(r[2].0, r[3].0);
                let a3 = _mm_unpackhi_epi16(r[2].0, r[3].0);
                let a4 = _mm_unpacklo_epi16(r[4].0, r[5].0);
                let a5 = _mm_unpackhi_epi16(r[4].0, r[5].0);
                let a6 = _mm_unpacklo_epi16(r[6].0, r[7].0);
                let a7 = _mm_unpackhi_epi16(r[6].0, r[7].0);
                let b0 = _mm_unpacklo_epi32(a0, a2);
                let b1 = _mm_unpackhi_epi32(a0, a2);
                let b2 = _mm_unpacklo_epi32(a1, a3);
                let b3 = _mm_unpackhi_epi32(a1, a3);
                let b4 = _mm_unpacklo_epi32(a4, a6);
                let b5 = _mm_unpackhi_epi32(a4, a6);
                let b6 = _mm_unpacklo_epi32(a5, a7);
                let b7 = _mm_unpackhi_epi32(a5, a7);
                r[0] = Self(_mm_unpacklo_epi64(b0, b4));
                r[1] = Self(_mm_unpackhi_epi64(b0, b4));
                r[2] = Self(_mm_unpacklo_epi64(b1, b5));
                r[3] = Self(_mm_unpackhi_epi64(b1, b5));
                r[4] = Self(_mm_unpacklo_epi64(b2, b6));
                r[5] = Self(_mm_unpackhi_epi64(b2, b6));
                r[6] = Self(_mm_unpacklo_epi64(b3, b7));
                r[7] = Self(_mm_unpackhi_epi64(b3, b7));
            }
        }
        const PAIRED_WARP: bool = true;
        #[inline(always)]
        fn warp_h8(s: &[u8], f: [&[i8; 8]; 8]) -> Self {
            unsafe {
                // The filters' tap pairs (0, 2), (1, 3), (4, 6), (5, 7) gathered by pair, so
                // pmaddubsw sums them with the pixel pairs as far apart, without phaddw.
                let l = [
                    _mm_loadl_epi64(f[0].as_ptr().cast()),
                    _mm_loadl_epi64(f[1].as_ptr().cast()),
                    _mm_loadl_epi64(f[2].as_ptr().cast()),
                    _mm_loadl_epi64(f[3].as_ptr().cast()),
                    _mm_loadl_epi64(f[4].as_ptr().cast()),
                    _mm_loadl_epi64(f[5].as_ptr().cast()),
                    _mm_loadl_epi64(f[6].as_ptr().cast()),
                    _mm_loadl_epi64(f[7].as_ptr().cast()),
                ];
                let a0 = _mm_unpacklo_epi16(l[0], l[1]);
                let a1 = _mm_unpacklo_epi16(l[2], l[3]);
                let a2 = _mm_unpacklo_epi16(l[4], l[5]);
                let a3 = _mm_unpacklo_epi16(l[6], l[7]);
                let (b0, b1) = (_mm_unpacklo_epi32(a0, a1), _mm_unpackhi_epi32(a0, a1));
                let (b2, b3) = (_mm_unpacklo_epi32(a2, a3), _mm_unpackhi_epi32(a2, a3));
                let k = [
                    _mm_unpacklo_epi64(b0, b2),
                    _mm_unpackhi_epi64(b0, b2),
                    _mm_unpacklo_epi64(b1, b3),
                    _mm_unpackhi_epi64(b1, b3),
                ];
                let v = load15(s);
                let a = _mm_add_epi16(ptap::<0>(v, k[0]), ptap::<1>(v, k[1]));
                Self(_mm_add_epi16(
                    a,
                    _mm_add_epi16(ptap::<4>(v, k[2]), ptap::<5>(v, k[3])),
                ))
            }
        }
    }

    /// The 8 bpc vertical warp pass's row pairs `(j, j + 2)`, word-interleaved, columns 0 to 3
    /// in the low lane, as the tap pairs of `WARP_FILTER_PAIRED` take them.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    pub fn warp_pairs8(mid: &[S; 15]) -> [__m256i; 13] {
        let mut r = [_mm256_setzero_si256(); 13];
        for j in 0..13 {
            let (a, b) = (mid[j].0, mid[j + 2].0);
            let lo = _mm256_castsi128_si256(_mm_unpacklo_epi16(a, b));
            r[j] = _mm256_inserti128_si256::<1>(lo, _mm_unpackhi_epi16(a, b));
        }
        r
    }

    /// Filters `a` and `b`, one a lane.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    fn q2(a: &[i8; 8], b: &[i8; 8]) -> __m256i {
        // SAFETY: Both are 8 bytes.
        unsafe {
            let lo = _mm256_castsi128_si256(_mm_loadl_epi64(a.as_ptr().cast()));
            _mm256_inserti128_si256::<1>(lo, _mm_loadl_epi64(b.as_ptr().cast()))
        }
    }

    /// One row of the 8 bpc vertical warp pass, `(sum(rows * taps) + rnd) >> sh`, from the
    /// row pairs `r[0], r[1], r[4], r[5]` and the columns' filters `f` in `WARP_FILTER_PAIRED`'s
    /// order. The filters' tap pairs are transposed as bytes, then widened into the high bytes
    /// of words (no sign extension): the sums are 256 times the dot products.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    pub fn warp_v8(r: &[__m256i], f: [&[i8; 8]; 8], rnd: i32, sh: u32) -> S {
        let ab = _mm256_unpacklo_epi16(q2(f[0], f[4]), q2(f[1], f[5]));
        let cd = _mm256_unpacklo_epi16(q2(f[2], f[6]), q2(f[3], f[7]));
        let (x, y) = (_mm256_unpacklo_epi32(ab, cd), _mm256_unpackhi_epi32(ab, cd));
        let z = _mm256_setzero_si256();
        let s = _mm256_add_epi32(
            _mm256_madd_epi16(r[0], _mm256_unpacklo_epi8(z, x)),
            _mm256_madd_epi16(r[1], _mm256_unpackhi_epi8(z, x)),
        );
        let t = _mm256_add_epi32(
            _mm256_madd_epi16(r[4], _mm256_unpacklo_epi8(z, y)),
            _mm256_madd_epi16(r[5], _mm256_unpackhi_epi8(z, y)),
        );
        let s = _mm256_add_epi32(_mm256_add_epi32(s, t), _mm256_set1_epi32(rnd << 8));
        let s = _mm256_sra_epi32(s, _mm_cvtsi32_si128(sh as i32 + 8));
        S(_mm_packs_epi32(
            _mm256_castsi256_si128(s),
            _mm256_extracti128_si256::<1>(s),
        ))
    }

    /// Sixteen lanes on AVX2; with `R2` two rows of eight.
    #[cfg(target_arch = "x86_64")]
    #[derive(Clone, Copy)]
    pub struct A<const R2: bool = false>(__m256i);

    // SAFETY (every `unsafe` here): AVX2 holds where these are compiled in, and loads and
    // stores go through bounds-checked slices.
    #[cfg(target_arch = "x86_64")]
    impl<const R2: bool> Vx for A<R2> {
        const N: usize = 16;
        const ROWS: usize = 1 + R2 as usize;
        const BYTE_PAIRS: bool = true;

        #[inline(always)]
        fn splat(v: i16) -> Self {
            Self(unsafe { _mm256_set1_epi16(v) })
        }
        #[inline(always)]
        fn add(self, o: Self) -> Self {
            Self(unsafe { _mm256_add_epi16(self.0, o.0) })
        }
        #[inline(always)]
        fn sub(self, o: Self) -> Self {
            Self(unsafe { _mm256_sub_epi16(self.0, o.0) })
        }
        #[inline(always)]
        fn mul(self, o: Self) -> Self {
            Self(unsafe { _mm256_mullo_epi16(self.0, o.0) })
        }
        #[inline(always)]
        fn min(self, o: Self) -> Self {
            Self(unsafe { _mm256_min_epi16(self.0, o.0) })
        }
        #[inline(always)]
        fn max(self, o: Self) -> Self {
            Self(unsafe { _mm256_max_epi16(self.0, o.0) })
        }
        #[inline(always)]
        fn sra(self, n: u32) -> Self {
            Self(unsafe { _mm256_sra_epi16(self.0, _mm_cvtsi32_si128(n as i32)) })
        }
        #[inline(always)]
        fn load(s: &[i16]) -> Self {
            Self(unsafe { _mm256_loadu_si256(s[..16].as_ptr().cast()) })
        }
        #[inline(always)]
        fn store(self, d: &mut [i16]) {
            unsafe { _mm256_storeu_si256(d[..16].as_mut_ptr().cast(), self.0) }
        }
        #[inline(always)]
        fn load_u8(s: &[u8]) -> Self {
            Self(unsafe { _mm256_cvtepu8_epi16(_mm_loadu_si128(s[..16].as_ptr().cast())) })
        }
        #[inline(always)]
        fn store_u8(self, d: &mut [u8]) {
            unsafe {
                let lo = _mm256_castsi256_si128(self.0);
                let hi = _mm256_extracti128_si256::<1>(self.0);
                _mm_storeu_si128(d[..16].as_mut_ptr().cast(), _mm_packus_epi16(lo, hi))
            }
        }
        #[inline(always)]
        fn load_u16(s: &[u16]) -> Self {
            Self(unsafe { _mm256_loadu_si256(s[..16].as_ptr().cast()) })
        }
        #[inline(always)]
        fn store_u16(self, d: &mut [u16]) {
            unsafe { _mm256_storeu_si256(d[..16].as_mut_ptr().cast(), self.0) }
        }
        type P = (__m256i, __m256i);
        #[inline(always)]
        fn pair(a: Self, b: Self) -> Self::P {
            unsafe {
                (
                    _mm256_unpacklo_epi16(a.0, b.0),
                    _mm256_unpackhi_epi16(a.0, b.0),
                )
            }
        }
        #[inline(always)]
        fn maddp(p: &[Self::P], k: &[i16; 8], rnd: i32, sh: u32) -> Self {
            unsafe {
                let mut lo = _mm256_set1_epi32(rnd);
                let mut hi = lo;
                for (i, p) in p.iter().enumerate() {
                    let k = _mm256_set1_epi32(pairw(k[2 * i], k[2 * i + 1]));
                    lo = _mm256_add_epi32(lo, _mm256_madd_epi16(p.0, k));
                    hi = _mm256_add_epi32(hi, _mm256_madd_epi16(p.1, k));
                }
                let n = _mm_cvtsi32_si128(sh as i32);
                Self(_mm256_packs_epi32(
                    _mm256_sra_epi32(lo, n),
                    _mm256_sra_epi32(hi, n),
                ))
            }
        }
        #[inline(always)]
        fn madd2(a: Self, b: Self, ka: Self, kb: Self, rnd: i32, sh: u32) -> Self {
            unsafe {
                let r = _mm256_set1_epi32(rnd);
                let lo = _mm256_madd_epi16(
                    _mm256_unpacklo_epi16(a.0, b.0),
                    _mm256_unpacklo_epi16(ka.0, kb.0),
                );
                let hi = _mm256_madd_epi16(
                    _mm256_unpackhi_epi16(a.0, b.0),
                    _mm256_unpackhi_epi16(ka.0, kb.0),
                );
                let n = _mm_cvtsi32_si128(sh as i32);
                Self(_mm256_packs_epi32(
                    _mm256_sra_epi32(_mm256_add_epi32(lo, r), n),
                    _mm256_sra_epi32(_mm256_add_epi32(hi, r), n),
                ))
            }
        }
        #[inline(always)]
        fn h8<const T6: bool>(s: &[u8], f: &[i8; 8]) -> Self {
            unsafe {
                // The high lane from byte 7, its shuffles one further.
                let lo = _mm_loadu_si128(s[..16].as_ptr().cast());
                let hi = _mm_loadu_si128(s[7..23].as_ptr().cast());
                let v = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi);
                if T6 {
                    let a = _mm256_add_epi16(tap2(v, 1, f[1], f[2]), tap2(v, 3, f[3], f[4]));
                    return Self(_mm256_add_epi16(a, tap2(v, 5, f[5], f[6])));
                }
                let a = _mm256_add_epi16(tap2(v, 0, f[0], f[1]), tap2(v, 2, f[2], f[3]));
                let b = _mm256_add_epi16(tap2(v, 4, f[4], f[5]), tap2(v, 6, f[6], f[7]));
                Self(_mm256_add_epi16(a, b))
            }
        }
        #[inline(always)]
        fn rshr_c<const SH: i32>(self) -> Self {
            Self(unsafe { _mm256_mulhrs_epi16(self.0, _mm256_set1_epi16(1 << (15 - SH))) })
        }
        #[inline(always)]
        fn load2_u8(a: &[u8], b: &[u8]) -> Self {
            Self(unsafe { _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(ld64(a), ld64(b))) })
        }
        #[inline(always)]
        fn load2_u16(a: &[u16], b: &[u16]) -> Self {
            Self(unsafe { _mm256_loadu2_m128i(b[..8].as_ptr().cast(), a[..8].as_ptr().cast()) })
        }
        #[inline(always)]
        fn h8x2<const T6: bool>(a: &[u8], b: &[u8], f: &[i8; 8]) -> Self {
            unsafe {
                let v = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(load15(a)), load15(b));
                if T6 {
                    let s = _mm256_add_epi16(tap15(v, 1, f[1], f[2]), tap15(v, 3, f[3], f[4]));
                    return Self(_mm256_add_epi16(s, tap15(v, 5, f[5], f[6])));
                }
                let s = _mm256_add_epi16(tap15(v, 0, f[0], f[1]), tap15(v, 2, f[2], f[3]));
                let t = _mm256_add_epi16(tap15(v, 4, f[4], f[5]), tap15(v, 6, f[6], f[7]));
                Self(_mm256_add_epi16(s, t))
            }
        }
        #[inline(always)]
        fn mid(a: Self, b: Self) -> Self {
            Self(unsafe { _mm256_permute2x128_si256::<0x21>(a.0, b.0) })
        }
        #[inline(always)]
        fn store_row_u8<const I: usize>(self, d: &mut [u8]) {
            if !R2 {
                return self.store_u8(d);
            }
            unsafe {
                let lo = _mm256_castsi256_si128(self.0);
                let p = _mm_packus_epi16(lo, _mm256_extracti128_si256::<1>(self.0));
                let p = if I == 0 { p } else { _mm_unpackhi_epi64(p, p) };
                _mm_storel_epi64(d[..8].as_mut_ptr().cast(), p)
            }
        }
        #[inline(always)]
        fn store_row_u16<const I: usize>(self, d: &mut [u16]) {
            if !R2 {
                return self.store_u16(d);
            }
            unsafe {
                let v = if I == 0 {
                    _mm256_castsi256_si128(self.0)
                } else {
                    _mm256_extracti128_si256::<1>(self.0)
                };
                _mm_storeu_si128(d[..8].as_mut_ptr().cast(), v)
            }
        }
        #[inline(always)]
        fn store_row<const I: usize>(self, d: &mut [i16]) {
            if !R2 {
                return self.store(d);
            }
            unsafe {
                let v = if I == 0 {
                    _mm256_castsi256_si128(self.0)
                } else {
                    _mm256_extracti128_si256::<1>(self.0)
                };
                _mm_storeu_si128(d[..8].as_mut_ptr().cast(), v)
            }
        }
        #[inline(always)]
        fn pairb(a: Self, b: Self) -> Self {
            Self(unsafe { _mm256_or_si256(a.0, _mm256_slli_epi16::<8>(b.0)) })
        }
        #[inline(always)]
        fn maddb(p: &[Self], k: &[i16; 8]) -> Self {
            unsafe {
                let k0 = _mm256_set1_epi16(pair(k[0] as i8, k[1] as i8));
                let mut s = _mm256_maddubs_epi16(p[0].0, k0);
                for (i, p) in p.iter().enumerate().skip(1) {
                    let k = _mm256_set1_epi16(pair(k[2 * i] as i8, k[2 * i + 1] as i8));
                    s = _mm256_add_epi16(s, _mm256_maddubs_epi16(p.0, k));
                }
                Self(s)
            }
        }
        #[inline(always)]
        fn wmask(self, o: Self, rnd: i16, sh: u32) -> Self {
            unsafe {
                let d =
                    _mm256_sub_epi16(_mm256_max_epi16(self.0, o.0), _mm256_min_epi16(self.0, o.0));
                let d = _mm256_adds_epu16(d, _mm256_set1_epi16(rnd));
                let d = _mm256_srl_epi16(d, _mm_cvtsi32_si128(sh as i32));
                Self(_mm256_add_epi16(
                    _mm256_min_epu16(d, _mm256_set1_epi16(26)),
                    _mm256_set1_epi16(38),
                ))
            }
        }
        #[inline(always)]
        fn hadd(a: Self, b: Self) -> Self {
            Self(unsafe { _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_hadd_epi16(a.0, b.0)) })
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    use super::Vx;

    /// Eight lanes on NEON; with `D` the 8 bpc first pass in `sdot`s (dotprod), where the
    /// compiler has them (`rav1d_sdot`); with `R2` two rows of four.
    #[derive(Clone, Copy)]
    pub struct S<const D: bool, const R2: bool = false>(int16x8_t);

    /// The 4-tap windows of `vqtbl1_u8` over two rows' bytes 2 to 9.
    const WIN4: [[u8; 8]; 4] = [
        [0, 1, 2, 3, 8, 9, 10, 11],
        [1, 2, 3, 4, 9, 10, 11, 12],
        [2, 3, 4, 5, 10, 11, 12, 13],
        [3, 4, 5, 6, 11, 12, 13, 14],
    ];

    /// The 4-byte windows of output pixels 0..4, 4..8 and 8..12 for `sdot`, over bytes
    /// `j >= 8` loaded at `j + 1`.
    #[cfg(rav1d_sdot)]
    const WIN: [[u8; 16]; 3] = [
        [0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6],
        [4, 5, 6, 7, 5, 6, 7, 9, 6, 7, 9, 10, 7, 9, 10, 11],
        [
            9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14, 12, 13, 14, 15,
        ],
    ];

    // SAFETY (every `unsafe` here): NEON is the aarch64 baseline, and loads and stores go
    // through bounds-checked slices.
    impl<const D: bool, const R2: bool> Vx for S<D, R2> {
        const N: usize = 8;
        const ROWS: usize = 1 + R2 as usize;

        #[inline(always)]
        fn splat(v: i16) -> Self {
            Self(unsafe { vdupq_n_s16(v) })
        }
        #[inline(always)]
        fn add(self, o: Self) -> Self {
            Self(unsafe { vaddq_s16(self.0, o.0) })
        }
        #[inline(always)]
        fn sub(self, o: Self) -> Self {
            Self(unsafe { vsubq_s16(self.0, o.0) })
        }
        #[inline(always)]
        fn mul(self, o: Self) -> Self {
            Self(unsafe { vmulq_s16(self.0, o.0) })
        }
        #[inline(always)]
        fn min(self, o: Self) -> Self {
            Self(unsafe { vminq_s16(self.0, o.0) })
        }
        #[inline(always)]
        fn max(self, o: Self) -> Self {
            Self(unsafe { vmaxq_s16(self.0, o.0) })
        }
        #[inline(always)]
        fn sra(self, n: u32) -> Self {
            Self(unsafe { vshlq_s16(self.0, vdupq_n_s16(-(n as i16))) })
        }
        #[inline(always)]
        fn load(s: &[i16]) -> Self {
            Self(unsafe { vld1q_s16(s[..8].as_ptr()) })
        }
        #[inline(always)]
        fn store(self, d: &mut [i16]) {
            unsafe { vst1q_s16(d[..8].as_mut_ptr(), self.0) }
        }
        #[inline(always)]
        fn load_u8(s: &[u8]) -> Self {
            Self(unsafe { vreinterpretq_s16_u16(vmovl_u8(vld1_u8(s[..8].as_ptr()))) })
        }
        #[inline(always)]
        fn store_u8(self, d: &mut [u8]) {
            unsafe { vst1_u8(d[..8].as_mut_ptr(), vqmovun_s16(self.0)) }
        }
        #[inline(always)]
        fn load_u16(s: &[u16]) -> Self {
            Self(unsafe { vreinterpretq_s16_u16(vld1q_u16(s[..8].as_ptr())) })
        }
        #[inline(always)]
        fn store_u16(self, d: &mut [u16]) {
            unsafe { vst1q_u16(d[..8].as_mut_ptr(), vreinterpretq_u16_s16(self.0)) }
        }
        type P = (int16x8_t, int16x8_t);
        const PAIRS: bool = false;
        #[inline(always)]
        fn pair(a: Self, b: Self) -> Self::P {
            (a.0, b.0)
        }
        #[inline(always)]
        fn maddr(r: &[Self], k: &[i16; 8], rnd: i32, sh: u32) -> Self {
            unsafe {
                let mut lo = vdupq_n_s32(rnd);
                let mut hi = lo;
                for (i, r) in r.iter().enumerate() {
                    lo = vmlal_n_s16(lo, vget_low_s16(r.0), k[i]);
                    hi = vmlal_high_n_s16(hi, r.0, k[i]);
                }
                let n = vdupq_n_s32(-(sh as i32));
                Self(vqmovn_high_s32(
                    vqmovn_s32(vshlq_s32(lo, n)),
                    vshlq_s32(hi, n),
                ))
            }
        }
        #[inline(always)]
        fn maddp(p: &[Self::P], k: &[i16; 8], rnd: i32, sh: u32) -> Self {
            unsafe {
                let mut lo = vdupq_n_s32(rnd);
                let mut hi = lo;
                for (i, p) in p.iter().enumerate() {
                    lo = vmlal_n_s16(lo, vget_low_s16(p.0), k[2 * i]);
                    hi = vmlal_high_n_s16(hi, p.0, k[2 * i]);
                    lo = vmlal_n_s16(lo, vget_low_s16(p.1), k[2 * i + 1]);
                    hi = vmlal_high_n_s16(hi, p.1, k[2 * i + 1]);
                }
                let n = vdupq_n_s32(-(sh as i32));
                Self(vqmovn_high_s32(
                    vqmovn_s32(vshlq_s32(lo, n)),
                    vshlq_s32(hi, n),
                ))
            }
        }
        #[inline(always)]
        fn madd2(a: Self, b: Self, ka: Self, kb: Self, rnd: i32, sh: u32) -> Self {
            unsafe {
                let r = vdupq_n_s32(rnd);
                let lo = vmlal_s16(
                    vmlal_s16(r, vget_low_s16(a.0), vget_low_s16(ka.0)),
                    vget_low_s16(b.0),
                    vget_low_s16(kb.0),
                );
                let hi = vmlal_high_s16(vmlal_high_s16(r, a.0, ka.0), b.0, kb.0);
                let n = vdupq_n_s32(-(sh as i32));
                Self(vqmovn_high_s32(
                    vqmovn_s32(vshlq_s32(lo, n)),
                    vshlq_s32(hi, n),
                ))
            }
        }
        #[inline(always)]
        fn h8<const T6: bool>(s: &[u8], f: &[i8; 8]) -> Self {
            unsafe {
                #[cfg(rav1d_sdot)]
                if D {
                    // `s[..8]` and `s[7..15]`, the windows' bytes `j >= 8` at `j + 1`, less 128
                    // to be signed; `128 * sum(f)` adds it back.
                    let v = vcombine_u8(vld1_u8(s[..8].as_ptr()), vld1_u8(s[7..15].as_ptr()));
                    let v = veorq_u8(v, vdupq_n_u8(0x80));
                    let p0 = vreinterpretq_s8_u8(vqtbl1q_u8(v, vld1q_u8(WIN[0].as_ptr())));
                    let p1 = vreinterpretq_s8_u8(vqtbl1q_u8(v, vld1q_u8(WIN[1].as_ptr())));
                    let p2 = vreinterpretq_s8_u8(vqtbl1q_u8(v, vld1q_u8(WIN[2].as_ptr())));
                    let f8 = vld1_s8(f.as_ptr());
                    let k = vcombine_s8(f8, f8);
                    let z = vdupq_n_s32(128 * 64);
                    let lo = vdotq_laneq_s32::<1>(vdotq_laneq_s32::<0>(z, p0, k), p1, k);
                    let hi = vdotq_laneq_s32::<1>(vdotq_laneq_s32::<0>(z, p1, k), p2, k);
                    return Self(vmovn_high_s32(vmovn_s32(lo), hi));
                }
                let v = if s.len() >= 16 {
                    vld1q_u8(s[..16].as_ptr())
                } else {
                    let t = vld1_u8(s[7..15].as_ptr());
                    vcombine_u8(vld1_u8(s[..8].as_ptr()), vext_u8::<1>(t, t))
                };
                let a = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v)));
                let b = vreinterpretq_s16_u16(vmovl_high_u8(v));
                let mut acc = vmulq_n_s16(vextq_s16::<1>(a, b), f[1] as i16);
                if !T6 {
                    acc = vmlaq_n_s16(acc, a, f[0] as i16);
                    acc = vmlaq_n_s16(acc, vextq_s16::<7>(a, b), f[7] as i16);
                }
                acc = vmlaq_n_s16(acc, vextq_s16::<2>(a, b), f[2] as i16);
                acc = vmlaq_n_s16(acc, vextq_s16::<3>(a, b), f[3] as i16);
                acc = vmlaq_n_s16(acc, vextq_s16::<4>(a, b), f[4] as i16);
                acc = vmlaq_n_s16(acc, vextq_s16::<5>(a, b), f[5] as i16);
                acc = vmlaq_n_s16(acc, vextq_s16::<6>(a, b), f[6] as i16);
                Self(acc)
            }
        }
        #[inline(always)]
        fn rshr_c<const SH: i32>(self) -> Self {
            Self(unsafe { vrshrq_n_s16::<SH>(self.0) })
        }
        #[inline(always)]
        fn load2_u8(a: &[u8], b: &[u8]) -> Self {
            let a = u32::from_le_bytes(a[..4].try_into().unwrap());
            let b = u32::from_le_bytes(b[..4].try_into().unwrap());
            unsafe {
                let v = vset_lane_u32::<1>(b, vdup_n_u32(a));
                Self(vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(v))))
            }
        }
        #[inline(always)]
        fn load2_u16(a: &[u16], b: &[u16]) -> Self {
            unsafe {
                let v = vcombine_u16(vld1_u16(a[..4].as_ptr()), vld1_u16(b[..4].as_ptr()));
                Self(vreinterpretq_s16_u16(v))
            }
        }
        #[inline(always)]
        fn h8x2<const T6: bool>(a: &[u8], b: &[u8], f: &[i8; 8]) -> Self {
            // Four pixels a row take the 4-tap filters (taps 2 to 5).
            if T6 && f[1] == 0 && f[6] == 0 {
                unsafe {
                    let q = vcombine_u8(vld1_u8(a[2..10].as_ptr()), vld1_u8(b[2..10].as_ptr()));
                    let w0 = vmovl_u8(vqtbl1_u8(q, vld1_u8(WIN4[0].as_ptr())));
                    let w1 = vmovl_u8(vqtbl1_u8(q, vld1_u8(WIN4[1].as_ptr())));
                    let w2 = vmovl_u8(vqtbl1_u8(q, vld1_u8(WIN4[2].as_ptr())));
                    let w3 = vmovl_u8(vqtbl1_u8(q, vld1_u8(WIN4[3].as_ptr())));
                    let mut acc = vmulq_n_s16(vreinterpretq_s16_u16(w0), f[2] as i16);
                    acc = vmlaq_n_s16(acc, vreinterpretq_s16_u16(w1), f[3] as i16);
                    acc = vmlaq_n_s16(acc, vreinterpretq_s16_u16(w2), f[4] as i16);
                    return Self(vmlaq_n_s16(acc, vreinterpretq_s16_u16(w3), f[5] as i16));
                }
            }
            let (mut ta, mut tb) = ([0; 16], [0; 16]);
            ta[..11].copy_from_slice(&a[..11]);
            tb[..11].copy_from_slice(&b[..11]);
            let (ra, rb) = (S::<false>::h8::<T6>(&ta, f), S::<false>::h8::<T6>(&tb, f));
            Self(unsafe { vcombine_s16(vget_low_s16(ra.0), vget_low_s16(rb.0)) })
        }
        #[inline(always)]
        fn mid(a: Self, b: Self) -> Self {
            Self(unsafe { vextq_s16::<4>(a.0, b.0) })
        }
        #[inline(always)]
        fn store_row_u8<const I: usize>(self, d: &mut [u8]) {
            if !R2 {
                return self.store_u8(d);
            }
            let v = unsafe { vreinterpret_u32_u8(vqmovun_s16(self.0)) };
            let v = unsafe {
                if I == 0 {
                    vget_lane_u32::<0>(v)
                } else {
                    vget_lane_u32::<1>(v)
                }
            };
            d[..4].copy_from_slice(&v.to_le_bytes());
        }
        #[inline(always)]
        fn store_row_u16<const I: usize>(self, d: &mut [u16]) {
            if !R2 {
                return self.store_u16(d);
            }
            unsafe {
                let v = vreinterpretq_u16_s16(self.0);
                let v = if I == 0 {
                    vget_low_u16(v)
                } else {
                    vget_high_u16(v)
                };
                vst1_u16(d[..4].as_mut_ptr(), v)
            }
        }
        #[inline(always)]
        fn store_row<const I: usize>(self, d: &mut [i16]) {
            if !R2 {
                return self.store(d);
            }
            unsafe {
                let v = if I == 0 {
                    vget_low_s16(self.0)
                } else {
                    vget_high_s16(self.0)
                };
                vst1_s16(d[..4].as_mut_ptr(), v)
            }
        }
        #[inline(always)]
        fn wmask(self, o: Self, rnd: i16, sh: u32) -> Self {
            unsafe {
                let d = vreinterpretq_u16_s16(vabdq_s16(self.0, o.0));
                let d = vqaddq_u16(d, vdupq_n_u16(rnd as u16));
                let d = vshlq_u16(d, vdupq_n_s16(-(sh as i16)));
                let d = vminq_u16(d, vdupq_n_u16(26));
                Self(vreinterpretq_s16_u16(vaddq_u16(d, vdupq_n_u16(38))))
            }
        }
        #[inline(always)]
        fn hadd(a: Self, b: Self) -> Self {
            Self(unsafe { vpaddq_s16(a.0, b.0) })
        }
    }
    // SAFETY (every `unsafe` here): as for `Vx`.
    impl<const D: bool> super::Vw for S<D> {
        #[inline(always)]
        fn load_i8(s: &[i8]) -> Self {
            Self(unsafe { vmovl_s8(vld1_s8(s[..8].as_ptr())) })
        }
        #[inline(always)]
        fn srl(self, n: u32) -> Self {
            let n = unsafe { vdupq_n_s16(-(n as i16)) };
            Self(unsafe { vreinterpretq_s16_u16(vshlq_u16(vreinterpretq_u16_s16(self.0), n)) })
        }
        #[inline(always)]
        fn hsum8(p: &[Self; 8]) -> Self {
            unsafe {
                let a = vpaddq_s16(p[0].0, p[1].0);
                let b = vpaddq_s16(p[2].0, p[3].0);
                let c = vpaddq_s16(p[4].0, p[5].0);
                let d = vpaddq_s16(p[6].0, p[7].0);
                Self(vpaddq_s16(vpaddq_s16(a, b), vpaddq_s16(c, d)))
            }
        }
        #[inline(always)]
        fn hdot8(v: &[Self; 8], f: &[Self; 8], rnd: i32, sh: u32) -> Self {
            unsafe {
                let m0 = dot(v[0].0, f[0].0);
                let m1 = dot(v[1].0, f[1].0);
                let m2 = dot(v[2].0, f[2].0);
                let m3 = dot(v[3].0, f[3].0);
                let m4 = dot(v[4].0, f[4].0);
                let m5 = dot(v[5].0, f[5].0);
                let m6 = dot(v[6].0, f[6].0);
                let m7 = dot(v[7].0, f[7].0);
                let lo = vpaddq_s32(vpaddq_s32(m0, m1), vpaddq_s32(m2, m3));
                let hi = vpaddq_s32(vpaddq_s32(m4, m5), vpaddq_s32(m6, m7));
                let r = vdupq_n_s32(rnd);
                let n = vdupq_n_s32(-(sh as i32));
                Self(vqmovn_high_s32(
                    vqmovn_s32(vshlq_s32(vaddq_s32(lo, r), n)),
                    vshlq_s32(vaddq_s32(hi, r), n),
                ))
            }
        }
        #[inline(always)]
        fn maddv8(v: &[Self], k: &[Self; 8], rnd: i32, sh: u32) -> Self {
            unsafe {
                let mut lo = vdupq_n_s32(rnd);
                let mut hi = lo;
                for i in 0..8 {
                    lo = vmlal_s16(lo, vget_low_s16(v[i].0), vget_low_s16(k[i].0));
                    hi = vmlal_high_s16(hi, v[i].0, k[i].0);
                }
                let n = vdupq_n_s32(-(sh as i32));
                Self(vqmovn_high_s32(
                    vqmovn_s32(vshlq_s32(lo, n)),
                    vshlq_s32(hi, n),
                ))
            }
        }
        #[inline(always)]
        fn transpose8(r: &mut [Self; 8]) {
            unsafe {
                let (a0, a1) = (vzip1q_s16(r[0].0, r[1].0), vzip2q_s16(r[0].0, r[1].0));
                let (a2, a3) = (vzip1q_s16(r[2].0, r[3].0), vzip2q_s16(r[2].0, r[3].0));
                let (a4, a5) = (vzip1q_s16(r[4].0, r[5].0), vzip2q_s16(r[4].0, r[5].0));
                let (a6, a7) = (vzip1q_s16(r[6].0, r[7].0), vzip2q_s16(r[6].0, r[7].0));
                let (b0, b1) = z32(a0, a2);
                let (b2, b3) = z32(a1, a3);
                let (b4, b5) = z32(a4, a6);
                let (b6, b7) = z32(a5, a7);
                let (c0, c1) = z64(b0, b4);
                let (c2, c3) = z64(b1, b5);
                let (c4, c5) = z64(b2, b6);
                let (c6, c7) = z64(b3, b7);
                *r = [
                    Self(c0),
                    Self(c1),
                    Self(c2),
                    Self(c3),
                    Self(c4),
                    Self(c5),
                    Self(c6),
                    Self(c7),
                ];
            }
        }
    }

    /// The four pair sums of `a * b`'s products.
    #[inline(always)]
    unsafe fn dot(a: int16x8_t, b: int16x8_t) -> int32x4_t {
        unsafe { vmlal_high_s16(vmull_s16(vget_low_s16(a), vget_low_s16(b)), a, b) }
    }

    #[inline(always)]
    unsafe fn z32(a: int16x8_t, b: int16x8_t) -> (int16x8_t, int16x8_t) {
        unsafe {
            let (a, b) = (vreinterpretq_s32_s16(a), vreinterpretq_s32_s16(b));
            (
                vreinterpretq_s16_s32(vzip1q_s32(a, b)),
                vreinterpretq_s16_s32(vzip2q_s32(a, b)),
            )
        }
    }

    #[inline(always)]
    unsafe fn z64(a: int16x8_t, b: int16x8_t) -> (int16x8_t, int16x8_t) {
        unsafe {
            let (a, b) = (vreinterpretq_s64_s16(a), vreinterpretq_s64_s16(b));
            (
                vreinterpretq_s16_s64(vzip1q_s64(a, b)),
                vreinterpretq_s16_s64(vzip2q_s64(a, b)),
            )
        }
    }
}

#[cfg(target_arch = "aarch64")]
type S = neon::S<false>;
#[cfg(target_arch = "aarch64")]
type A = D<neon::S<true>>;
#[cfg(target_arch = "x86_64")]
use x86::A;
/// The eight lanes of the wide tier's narrow blocks.
#[cfg(target_arch = "x86_64")]
type Narrow = x86::S;
#[cfg(target_arch = "aarch64")]
type Narrow = neon::S<true>;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use x86::S;

#[inline(always)]
fn ld_px<BD: BitDepth, V: Vx, const PW: usize>(s: &[BD::Pixel]) -> V {
    if BD::BITDEPTH == 8 {
        ld_u8::<V, PW>(cast(s))
    } else {
        ld_u16::<V, PW>(cast(s))
    }
}

/// The bilinear first pass of row `y` at `x`, `(a * (16 - f) + b * f + rnd) >> sh`.
#[inline(always)]
fn bl2<BD: BitDepth, V: Vx>(a: V, b: V, f: i16, rnd: i32, sh: u32) -> V {
    if BD::BITDEPTH == 8 {
        rshr8(
            a.mul(V::splat(16 - f)).add(b.mul(V::splat(f))),
            rnd as i16,
            sh,
        )
    } else {
        V::madd2(a, b, V::splat(16 - f), V::splat(f), rnd, sh)
    }
}

#[inline(always)]
fn bl_row<BD: BitDepth, V: Vx, const PW: usize>(
    src: &PicRows<BD>,
    y: usize,
    x: usize,
    n: usize,
    mx: i16,
    hrnd: i32,
    hsh: u32,
) -> V {
    let s = src.row(y as isize, x as isize, n + 1);
    bl2::<BD, V>(
        ld_px::<BD, V, PW>(&s),
        ld_px::<BD, V, PW>(&s[1..]),
        mx,
        hrnd,
        hsh,
    )
}

/// `put_bilin` / `prep_bilin` with a subpel position.
#[inline(always)]
pub fn bilin<BD: BitDepth, V: Vx, const PW: usize>(
    mut out: Out<BD>,
    src: Rav1dPictureDataComponentOffset,
    w: usize,
    h: usize,
    mx: usize,
    my: usize,
    bd: BD,
) {
    let ib = bd.get_intermediate_bits() as u32;
    let max = bd.bitdepth_max().as_::<i32>() as i16;
    let prep = out.is_prep();
    let bias = BD::PREP_BIAS;
    let n = if PW == 0 { V::N } else { PW };
    let (mx, my) = (mx as i16, my as i16);
    let hsh = 4 - ib;
    let hrnd = (1 << hsh) >> 1;
    let (dx, dy) = ((mx != 0) as isize, (my != 0) as isize);
    // SAFETY: Bilinear reads `h` rows of `w` pixels, one more of each along a filter it has.
    let src = unsafe { src.block_unchecked::<BD>(0..h as isize + dy, 0..w as isize + dx) };
    for x in (0..w).step_by(n) {
        if my == 0 {
            for y in 0..h {
                let v = bl_row::<BD, V, PW>(&src, y, x, n, mx, hrnd, hsh);
                let v = if prep {
                    v.sub(V::splat(bias))
                } else {
                    rshr8(v, (1 << ib) >> 1, ib)
                };
                out.emit::<V, PW>(v, x, y, w, max);
            }
        } else if mx == 0 {
            let (rnd, sh) = if prep { (hrnd, hsh) } else { (8, 4) };
            let mut a = ld_px::<BD, V, PW>(&src.row(0, x as isize, n));
            for y in 0..h {
                let b = ld_px::<BD, V, PW>(&src.row(y as isize + 1, x as isize, n));
                let v = bl2::<BD, V>(a, b, my, rnd, sh);
                let v = if prep { v.sub(V::splat(bias)) } else { v };
                out.emit::<V, PW>(v, x, y, w, max);
                a = b;
            }
        } else {
            let (rnd, sh) = if prep {
                (8, 4)
            } else {
                (1 << (3 + ib), 4 + ib)
            };
            let mut a = bl_row::<BD, V, PW>(&src, 0, x, n, mx, hrnd, hsh);
            for y in 0..h {
                let b = bl_row::<BD, V, PW>(&src, y + 1, x, n, mx, hrnd, hsh);
                let v = V::madd2(a, b, V::splat(16 - my), V::splat(my), rnd, sh);
                let v = if prep { v.sub(V::splat(bias)) } else { v };
                out.emit::<V, PW>(v, x, y, w, max);
                a = b;
            }
        }
    }
}

/// `prep` without a subpel position: `(px << ib) - bias`.
#[inline(always)]
pub fn prep_copy<BD: BitDepth, V: Vx, const PW: usize>(
    tmp: &mut [i16],
    src: Rav1dPictureDataComponentOffset,
    w: usize,
    h: usize,
    bd: BD,
) {
    let ib = bd.get_intermediate_bits() as u32;
    let n = if PW == 0 { V::N } else { PW };
    let (k, bias) = (V::splat(1 << ib), V::splat(BD::PREP_BIAS));
    // SAFETY: The copy reads the block's `h` rows of `w` pixels.
    let src = unsafe { src.block_unchecked::<BD>(0..h as isize, 0..w as isize) };
    for y in 0..h {
        let s = src.row(y as isize, 0, w);
        for x in (0..w).step_by(n) {
            let v = ld_px::<BD, V, PW>(&s[x..]).mul(k).sub(bias);
            st_i16::<V, PW>(v, &mut tmp[y * w + x..]);
        }
    }
}

#[derive(Clone, Copy)]
pub enum Blend<'a> {
    Mask(&'a [u8]),
    V,
    H,
}

/// `blend`, `blend_v` and `blend_h`: `(dst * (64 - m) + tmp * m + 32) >> 6`. `blend_v` runs
/// over the whole width with the masks past three quarters at 0, which keeps `dst`.
#[inline(always)]
pub fn blend<BD: BitDepth, V: Vx, const PW: usize>(
    dst: Rav1dPictureDataComponentOffset,
    tmp: &[BD::Pixel],
    w: usize,
    h: usize,
    kind: Blend,
) {
    // A weighted average of two pixels needs no clipping.
    let max = i16::MAX;
    let n = if PW == 0 { V::N } else { PW };
    let obmc = &DAV1D_OBMC_MASKS.0;
    let mut vm = [0u8; 32];
    if let Blend::V = kind {
        vm[..w * 3 >> 2].copy_from_slice(&obmc[w..][..w * 3 >> 2]);
    }
    let oh = if let Blend::H = kind { h * 3 >> 2 } else { h };
    // SAFETY: The blend writes `oh` rows of `w` pixels.
    let dst = unsafe { dst.block_unchecked::<BD>(0..oh as isize, 0..w as isize) };
    for y in 0..oh {
        let row = &mut *dst.row_mut(y as isize, 0, w);
        for x in (0..w).step_by(n) {
            let d = ld_px::<BD, V, PW>(&row[x..]);
            let t = ld_px::<BD, V, PW>(&tmp[y * w + x..]);
            let m = match kind {
                Blend::Mask(m) => ld_u8::<V, PW>(&m[y * w + x..]),
                Blend::V => ld_u8::<V, PW>(&vm[x..]),
                Blend::H => V::splat(obmc[h + y] as i16),
            };
            let im = V::splat(64).sub(m);
            let v = if BD::BITDEPTH == 8 {
                d.mul(im).add(t.mul(m)).rshr_c::<6>()
            } else {
                V::madd2(d, t, im, m, 32, 6)
            };
            store_px::<BD, V, PW>(v, &mut row[x..], max);
        }
    }
}

/// `v`'s lanes `from..from + d.len()` saturated to `u8` into `d`.
#[inline(always)]
fn store_u8_lanes<V: Vx>(v: V, from: usize, d: &mut [u8]) {
    let mut t = [0u8; 32];
    v.store_u8(&mut t);
    let n = d.len();
    d.copy_from_slice(&t[from..from + n]);
}

/// `w_mask`: the blend weight from the two predictions' difference, its mask stored at the
/// chroma's subsampling. Two rows a step, whose weights the subsampled masks sum.
#[inline(always)]
pub fn w_mask<BD: BitDepth, V: Vx, const PW: usize>(
    dst: Rav1dPictureDataComponentOffset,
    t1: &[i16],
    t2: &[i16],
    w: usize,
    h: usize,
    mask: &mut [u8],
    sign: u8,
    ss: (bool, bool),
    bd: BD,
) {
    let (ss_hor, ss_ver) = ss;
    let ib = bd.get_intermediate_bits() as u32;
    let max = bd.bitdepth_max().as_::<i32>() as i16;
    let rnd = (32 << ib) + 64 * BD::PREP_BIAS as i32;
    let mask_sh = bd.bitdepth() as u32 + ib - 4;
    let mask_rnd = 1 << (mask_sh - 5);
    let n = if PW == 0 { V::N } else { PW };
    let mw = w >> ss_hor as usize;
    // SAFETY: The strips write the block's `h` rows of `w` pixels.
    let dst = unsafe { dst.block_unchecked::<BD>(0..h as isize, 0..w as isize) };
    let sign = sign as i16;
    for y in (0..h).step_by(2) {
        for x in (0..w).step_by(n) {
            let mut m = [V::splat(0); 2];
            for r in 0..2 {
                let i = (y + r) * w + x;
                let a = ld_i16::<V, PW>(&t1[i..]);
                let b = ld_i16::<V, PW>(&t2[i..]);
                m[r] = a.wmask(b, mask_rnd, mask_sh);
                let v = V::madd2(a, b, m[r], V::splat(64).sub(m[r]), rnd, ib + 6);
                let d = &mut dst.row_mut((y + r) as isize, x as isize, n);
                store_px::<BD, V, PW>(v, d, max);
            }
            if !ss_hor {
                store_u8_lanes(m[0], 0, &mut mask[y * mw + x..][..n]);
                store_u8_lanes(m[1], 0, &mut mask[(y + 1) * mw + x..][..n]);
            } else if !ss_ver {
                let s = V::hadd(m[0], m[1]).add(V::splat(1 - sign)).sra(1);
                store_u8_lanes(s, 0, &mut mask[y * mw + x / 2..][..n / 2]);
                store_u8_lanes(s, V::N / 2, &mut mask[(y + 1) * mw + x / 2..][..n / 2]);
            } else {
                let s = m[0].add(m[1]);
                let s = V::hadd(s, s).add(V::splat(2 - sign)).sra(2);
                store_u8_lanes(s, 0, &mut mask[(y >> 1) * mw + x / 2..][..n / 2]);
            }
        }
    }
}

/// `mod $m` with the kernel `$k` on eight lanes (`n`, `PW` the partial width or 0) and, on
/// x86_64, sixteen (`w`), each compiled for its tier.
macro_rules! tiers {
    ($m:ident, $k:ident, ($($a:ident: $t:ty),*)) => {
        mod $m {
            use super::*;

            #[cfg_attr(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature(enable = "sse4.1")
            )]
            pub fn n<BD: BitDepth, const PW: usize>($($a: $t),*) {
                $k::<BD, S, PW>($($a),*)
            }

            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            #[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
            #[cfg_attr(target_arch = "aarch64", target_feature(enable = "dotprod"))]
            pub fn w<BD: BitDepth>($($a: $t),*) {
                $k::<BD, A, 0>($($a),*)
            }

            /// Below the wide tier's width, on its CPU: eight lanes in AVX2's VEX encoding
            /// (three operands, no register copies), or NEON with dotprod.
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            #[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
            #[cfg_attr(target_arch = "aarch64", target_feature(enable = "dotprod"))]
            pub fn nw<BD: BitDepth, const PW: usize>($($a: $t),*) {
                $k::<BD, Narrow, PW>($($a),*)
            }
        }
    };
}

tiers!(mc8, mc_8tap, (out: Out<BD>, src: Rav1dPictureDataComponentOffset, w: usize, h: usize,
    fh: Option<&[i8; 8]>, fv: Option<&[i8; 8]>, bd: BD));
/// `mod $m` with the kernel `$k` two rows a vector: 4-pixel strips on SSE4.1 (`n`) and in
/// AVX2's VEX encoding (`nw`), 8-pixel ones on AVX2 (`w`); 4-pixel strips on NEON (`n`, and
/// `nw` with dotprod).
macro_rules! tiers2 {
    ($m:ident, $k:ident, ($($a:ident: $t:ty),*)) => {
        mod $m {
            use super::*;

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[target_feature(enable = "sse4.1")]
            pub fn n<BD: BitDepth>($($a: $t),*) {
                $k::<BD, x86::S<true>, 0>($($a),*)
            }

            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx2")]
            pub fn nw<BD: BitDepth>($($a: $t),*) {
                $k::<BD, x86::S<true>, 0>($($a),*)
            }

            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx2")]
            pub fn w<BD: BitDepth>($($a: $t),*) {
                $k::<BD, A<true>, 0>($($a),*)
            }

            #[cfg(target_arch = "aarch64")]
            pub fn n<BD: BitDepth>($($a: $t),*) {
                $k::<BD, neon::S<false, true>, 0>($($a),*)
            }

            #[cfg(target_arch = "aarch64")]
            #[target_feature(enable = "dotprod")]
            pub fn nw<BD: BitDepth>($($a: $t),*) {
                $k::<BD, neon::S<true, true>, 0>($($a),*)
            }
        }
    };
}

tiers2!(mc8x2, mc_8tap, (out: Out<BD>, src: Rav1dPictureDataComponentOffset, w: usize, h: usize,
    fh: Option<&[i8; 8]>, fv: Option<&[i8; 8]>, bd: BD));
tiers2!(cmpx2, comp, (dst: Rav1dPictureDataComponentOffset, t1: &[i16], t2: &[i16], w: usize,
    h: usize, kind: Comp, bd: BD));
tiers!(cmp, comp, (dst: Rav1dPictureDataComponentOffset, t1: &[i16], t2: &[i16], w: usize,
    h: usize, kind: Comp, bd: BD));

#[cfg_attr(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature(enable = "sse4.1")
)]
fn warp_128<BD: BitDepth>(
    out: Out<BD>,
    w: usize,
    src: Rav1dPictureDataComponentOffset,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bd: BD,
) {
    warp::<BD, S>(out, w, src, abcd, mx, my, bd)
}

/// `warp_128` on an AVX2 CPU, in VEX encoding; the 8 bpc vertical pass in ymm.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn warp_vex<BD: BitDepth>(
    mut out: Out<BD>,
    w: usize,
    src: Rav1dPictureDataComponentOffset,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bd: BD,
) {
    let mid = warp_h::<BD, S>(src, abcd, mx, bd);
    if BD::BITDEPTH != 8 {
        return warp_v::<BD, S>(out, w, &mid, abcd, my, bd);
    }
    let prep = out.is_prep();
    let (rnd, sh) = warp_rnd(prep, bd);
    let r = x86::warp_pairs8(&mid);
    for y in 0..8 {
        let f = warp_filter_refs::<S>(my + y as i32 * abcd[3] as i32, abcd[2] as i32);
        let v = x86::warp_v8(&r[y..], f, rnd, sh);
        out.emit::<S, 0>(v, 0, y, w, 0);
    }
}

/// # Safety
///
/// On x86 SSE4.1 must hold, and AVX2 with `WIDE`.
#[inline(always)]
unsafe fn warp_any<BD: BitDepth, const WIDE: bool>(
    out: Out<BD>,
    w: usize,
    src: Rav1dPictureDataComponentOffset,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bd: BD,
) {
    #[cfg(target_arch = "x86_64")]
    if WIDE {
        // SAFETY: AVX2 holds with `WIDE`.
        return unsafe { warp_vex(out, w, src, abcd, mx, my, bd) };
    }
    // SAFETY: SSE4.1 holds on x86; elsewhere this is safe.
    #[allow(unused_unsafe)]
    unsafe {
        warp_128(out, w, src, abcd, mx, my, bd)
    }
}

tiers!(bl, bilin, (out: Out<BD>, src: Rav1dPictureDataComponentOffset, w: usize, h: usize,
    mx: usize, my: usize, bd: BD));
tiers!(pc, prep_copy, (tmp: &mut [i16], src: Rav1dPictureDataComponentOffset, w: usize,
    h: usize, bd: BD));
tiers!(bln, blend, (dst: Rav1dPictureDataComponentOffset, tmp: &[BD::Pixel], w: usize,
    h: usize, kind: Blend));
tiers!(wm, w_mask, (dst: Rav1dPictureDataComponentOffset, t1: &[i16], t2: &[i16], w: usize,
    h: usize, mask: &mut [u8], sign: u8, ss: (bool, bool), bd: BD));

/// Runs `$m` on the tier for width `$w`: sixteen lanes with `$wide` from 16 pixels on, eight
/// otherwise, through a copy below 8.
macro_rules! call_tier {
    ($m:ident, $wide:expr, $w:expr, ($($a:expr),*)) => {{
        // SAFETY: SSE4.1 is checked before these `fn`s are chosen on x86, and AVX2 before
        // `$wide` is; elsewhere they are safe.
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            if $wide {
                return match $w {
                    w if w >= <A>::N => $m::w::<BD>($($a),*),
                    2 => $m::nw::<BD, 2>($($a),*),
                    4 => $m::nw::<BD, 4>($($a),*),
                    _ => $m::nw::<BD, 0>($($a),*),
                };
            }
            let _ = $wide;
            match $w {
                2 => $m::n::<BD, 2>($($a),*),
                4 => $m::n::<BD, 4>($($a),*),
                _ => $m::n::<BD, 0>($($a),*),
            }
        }
    }};
}

/// `call_tier`, with `$m2`'s two-row tiers for 4-pixel blocks and on AVX2 8-pixel ones.
macro_rules! call_tier2 {
    ($m2:ident, $m:ident, $wide:expr, $w:expr, ($($a:expr),*)) => {{
        // SAFETY: As in `call_tier`.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            if $wide {
                match $w {
                    #[cfg(target_arch = "x86_64")]
                    8 => return $m2::w::<BD>($($a),*),
                    4 => return $m2::nw::<BD>($($a),*),
                    _ => {}
                }
            }
            if $w == 4 {
                return $m2::n::<BD>($($a),*);
            }
        }
        call_tier!($m, $wide, $w, ($($a),*))
    }};
}

#[inline(always)]
fn put_8tap<BD: BitDepth, const FILTER: usize, const WIDE: bool>(
    dst: Rav1dPictureDataComponentOffset,
    src: Rav1dPictureDataComponentOffset,
    w: usize,
    h: usize,
    mx: usize,
    my: usize,
    bd: BD,
) {
    let filter = Filter2d::from_repr(FILTER).unwrap();
    // SAFETY: The passes write the block's `h` rows of `w` pixels.
    let put = || Out::Put(unsafe { dst.block_unchecked::<BD>(0..h as isize, 0..w as isize) });
    if filter == Filter2d::Bilinear {
        if mx == 0 && my == 0 {
            return put_rust::<BD>(dst, src, w, h);
        }
        return call_tier!(bl, WIDE, w, (put(), src, w, h, mx, my, bd));
    }
    let (ht, vt) = filter.hv();
    let fh = get_filter(mx, w, ht);
    let fv = get_filter(my, h, vt);
    if fh.is_none() && fv.is_none() {
        return put_rust::<BD>(dst, src, w, h);
    }
    call_tier2!(mc8x2, mc8, WIDE, w, (put(), src, w, h, fh, fv, bd))
}

#[inline(always)]
fn prep_8tap<BD: BitDepth, const FILTER: usize, const WIDE: bool>(
    tmp: &mut [i16],
    src: Rav1dPictureDataComponentOffset,
    w: usize,
    h: usize,
    mx: usize,
    my: usize,
    bd: BD,
) {
    let filter = Filter2d::from_repr(FILTER).unwrap();
    if filter == Filter2d::Bilinear {
        if mx == 0 && my == 0 {
            return call_tier!(pc, WIDE, w, (tmp, src, w, h, bd));
        }
        return call_tier!(bl, WIDE, w, (Out::Prep(tmp), src, w, h, mx, my, bd));
    }
    let (ht, vt) = filter.hv();
    let fh = get_filter(mx, w, ht);
    let fv = get_filter(my, h, vt);
    if fh.is_none() && fv.is_none() {
        return call_tier!(pc, WIDE, w, (tmp, src, w, h, bd));
    }
    call_tier2!(mc8x2, mc8, WIDE, w, (Out::Prep(tmp), src, w, h, fh, fv, bd))
}

/// # Safety
///
/// Must be called by [`mc::Fn::call`], on x86 once the CPU flags hold SSE41, and with `WIDE`
/// once they hold AVX2.
#[deny(unsafe_op_in_unsafe_fn)]
pub unsafe extern "C" fn put_erased<BD: BitDepth, const FILTER: usize, const WIDE: bool>(
    _dst_ptr: *mut DynPixel,
    _dst_stride: isize,
    _src_ptr: *const DynPixel,
    _src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: FFISafeRav1dPictureDataComponentOffset,
    src: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `mc::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `mc::Fn::call`.
    let src = unsafe { FFISafe::from_with_offset(src) };
    let bd = BD::from_c(bitdepth_max);
    let (w, h, mx, my) = (w as usize, h as usize, mx as usize, my as usize);
    put_8tap::<BD, FILTER, WIDE>(dst, src, w, h, mx, my, bd)
}

/// # Safety
///
/// Must be called by [`mct::Fn::call`], on x86 once the CPU flags hold SSE41, and with `WIDE`
/// once they hold AVX2.
#[deny(unsafe_op_in_unsafe_fn)]
pub unsafe extern "C" fn prep_erased<BD: BitDepth, const FILTER: usize, const WIDE: bool>(
    tmp: *mut i16,
    _src_ptr: *const DynPixel,
    _src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `mct::Fn::call`.
    let src = unsafe { FFISafe::from_with_offset(src) };
    let (w, h, mx, my) = (w as usize, h as usize, mx as usize, my as usize);
    // SAFETY: Length sliced in `mct::Fn::call`.
    let tmp = unsafe { slice::from_raw_parts_mut(tmp, w * h) };
    let bd = BD::from_c(bitdepth_max);
    prep_8tap::<BD, FILTER, WIDE>(tmp, src, w, h, mx, my, bd)
}

#[inline(always)]
fn comp_any<BD: BitDepth, const WIDE: bool>(
    dst: Rav1dPictureDataComponentOffset,
    t1: &[i16],
    t2: &[i16],
    w: usize,
    h: usize,
    kind: Comp,
    bd: BD,
) {
    call_tier2!(cmpx2, cmp, WIDE, w, (dst, t1, t2, w, h, kind, bd))
}

/// # Safety
///
/// Must be called by [`avg::Fn::call`], on x86 once the CPU flags hold SSE41, and with `WIDE`
/// once they hold AVX2.
#[deny(unsafe_op_in_unsafe_fn)]
pub unsafe extern "C" fn avg_erased<BD: BitDepth, const WIDE: bool>(
    _dst_ptr: *mut DynPixel,
    _dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    bitdepth_max: i32,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `avg::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    let (w, h) = (w as usize, h as usize);
    let bd = BD::from_c(bitdepth_max);
    comp_any::<BD, WIDE>(dst, tmp1, tmp2, w, h, Comp::Avg, bd)
}

/// # Safety
///
/// Must be called by [`w_avg::Fn::call`], on x86 once the CPU flags hold SSE41, and with `WIDE`
/// once they hold AVX2.
#[deny(unsafe_op_in_unsafe_fn)]
pub unsafe extern "C" fn w_avg_erased<BD: BitDepth, const WIDE: bool>(
    _dst_ptr: *mut DynPixel,
    _dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    weight: i32,
    bitdepth_max: i32,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `w_avg::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    let (w, h) = (w as usize, h as usize);
    let bd = BD::from_c(bitdepth_max);
    comp_any::<BD, WIDE>(dst, tmp1, tmp2, w, h, Comp::WAvg(weight as i16), bd)
}

/// # Safety
///
/// Must be called by [`mask::Fn::call`], on x86 once the CPU flags hold SSE41, and with `WIDE`
/// once they hold AVX2.
#[deny(unsafe_op_in_unsafe_fn)]
pub unsafe extern "C" fn mask_erased<BD: BitDepth, const WIDE: bool>(
    _dst_ptr: *mut DynPixel,
    _dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: *const u8,
    bitdepth_max: i32,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `mask::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    let (w, h) = (w as usize, h as usize);
    // SAFETY: Length sliced in `mask::Fn::call`.
    let mask = unsafe { slice::from_raw_parts(mask, w * h) };
    let bd = BD::from_c(bitdepth_max);
    comp_any::<BD, WIDE>(dst, tmp1, tmp2, w, h, Comp::Mask(mask), bd)
}

/// # Safety
///
/// Must be called by [`warp8x8::Fn::call`], on x86 once the CPU flags hold SSE41, and with
/// `WIDE` once they hold AVX2.
#[deny(unsafe_op_in_unsafe_fn)]
pub unsafe extern "C" fn warp8x8_erased<BD: BitDepth, const WIDE: bool>(
    _dst_ptr: *mut DynPixel,
    _dst_stride: isize,
    _src_ptr: *const DynPixel,
    _src_stride: isize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: FFISafeRav1dPictureDataComponentOffset,
    src: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `warp_8x8::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `warp_8x8::Fn::call`.
    let src = unsafe { FFISafe::from_with_offset(src) };
    let bd = BD::from_c(bitdepth_max);
    // SAFETY: The CPU flags are checked before this `fn` is chosen; warp writes 8 rows of 8
    // pixels.
    unsafe {
        warp_any::<BD, WIDE>(
            Out::Put(dst.block_unchecked::<BD>(0..8, 0..8)),
            8,
            src,
            abcd,
            mx,
            my,
            bd,
        )
    }
}

/// # Safety
///
/// Must be called by [`warp8x8t::Fn::call`], on x86 once the CPU flags hold SSE41, and with
/// `WIDE` once they hold AVX2.
#[deny(unsafe_op_in_unsafe_fn)]
pub unsafe extern "C" fn warp8x8t_erased<BD: BitDepth, const WIDE: bool>(
    tmp: *mut i16,
    tmp_stride: usize,
    _src_ptr: *const DynPixel,
    _src_stride: isize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    tmp_len: usize,
    src: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: `warp8x8t::Fn::call` passed `tmp.len()` as `tmp_len`.
    let tmp = unsafe { slice::from_raw_parts_mut(tmp, tmp_len) };
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `warp8x8t::Fn::call`.
    let src = unsafe { FFISafe::from_with_offset(src) };
    let bd = BD::from_c(bitdepth_max);
    // SAFETY: The CPU flags are checked before this `fn` is chosen.
    unsafe { warp_any::<BD, WIDE>(Out::Prep(tmp), tmp_stride, src, abcd, mx, my, bd) }
}

/// # Safety
///
/// Must be called by [`w_mask::Fn::call`], on x86 once the CPU flags hold SSE41, and with `WIDE`
/// once they hold AVX2.
#[deny(unsafe_op_in_unsafe_fn)]
pub unsafe extern "C" fn w_mask_erased<
    BD: BitDepth,
    const SS_HOR: bool,
    const SS_VER: bool,
    const WIDE: bool,
>(
    _dst_ptr: *mut DynPixel,
    _dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    bitdepth_max: i32,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `w_mask::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    let (w, h) = (w as usize, h as usize);
    let bd = BD::from_c(bitdepth_max);
    let ss = (SS_HOR, SS_VER);
    call_tier!(
        wm,
        WIDE,
        w,
        (dst, tmp1, tmp2, w, h, mask, sign as u8, ss, bd)
    )
}

/// # Safety
///
/// Must be called by [`blend::Fn::call`], on x86 once the CPU flags hold SSE41, and with `WIDE`
/// once they hold AVX2.
#[deny(unsafe_op_in_unsafe_fn)]
pub unsafe extern "C" fn blend_erased<BD: BitDepth, const WIDE: bool>(
    _dst_ptr: *mut DynPixel,
    _dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_INTER_INTRA_BUF_LEN],
    w: i32,
    h: i32,
    mask: *const u8,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `blend::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    // SAFETY: Reverse of cast in `blend::Fn::call`.
    let tmp: &[BD::Pixel; SCRATCH_INTER_INTRA_BUF_LEN] = unsafe { &*tmp.cast() };
    let (w, h) = (w as usize, h as usize);
    // SAFETY: Length sliced in `blend::Fn::call`.
    let mask = unsafe { slice::from_raw_parts(mask, w * h) };
    call_tier!(bln, WIDE, w, (dst, tmp, w, h, Blend::Mask(mask)))
}

/// # Safety
///
/// Must be called by [`blend_dir::Fn::call`], on x86 once the CPU flags hold SSE41, and with
/// `WIDE` once they hold AVX2. `V` picks `blend_v` over `blend_h`.
#[deny(unsafe_op_in_unsafe_fn)]
pub unsafe extern "C" fn blend_dir_erased<BD: BitDepth, const V: bool, const WIDE: bool>(
    _dst_ptr: *mut DynPixel,
    _dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_LAP_LEN],
    w: i32,
    h: i32,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `blend_dir::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    // SAFETY: Reverse of cast in `blend_dir::Fn::call`.
    let tmp: &[BD::Pixel; SCRATCH_LAP_LEN] = unsafe { &*tmp.cast() };
    let (w, h) = (w as usize, h as usize);
    let kind = if V { Blend::V } else { Blend::H };
    call_tier!(bln, WIDE, w, (dst, tmp, w, h, kind))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::Align64;
    use crate::cpu::{rav1d_get_cpu_flags, rav1d_init_cpu, CpuFlags};
    use crate::include::common::bitdepth::{BitDepth16, BitDepth8};
    /// Picture buffers are multiples of 64 bytes.
    const RAV1D_PICTURE_GUARANTEED_MULTIPLE: usize = 64;

    pub struct Rng(pub u64);

    impl Rng {
        pub fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 32) as u32
        }

        pub fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % (hi - lo + 1) as u32) as i32
        }
    }

    pub const S: usize = 160;
    pub const DIMS: [usize; 7] = [2, 4, 8, 16, 32, 64, 128];
    pub const FILTERS: [Filter2d; 10] = [
        Filter2d::Regular8Tap,
        Filter2d::RegularSmooth8Tap,
        Filter2d::RegularSharp8Tap,
        Filter2d::SharpRegular8Tap,
        Filter2d::SharpSmooth8Tap,
        Filter2d::Sharp8Tap,
        Filter2d::SmoothRegular8Tap,
        Filter2d::Smooth8Tap,
        Filter2d::SmoothSharp8Tap,
        Filter2d::Bilinear,
    ];

    /// The scalar kernels, and the Rust SIMD ones on every tier the CPU has.
    pub fn dsps<BD: BitDepth>() -> (Rav1dMCDSPContext, Vec<(&'static str, Rav1dMCDSPContext)>) {
        rav1d_init_cpu();
        let flags = rav1d_get_cpu_flags();
        let scalar = Rav1dMCDSPContext::default::<BD>();
        #[allow(unused_mut)]
        let mut tiers = vec![(
            "128",
            Rav1dMCDSPContext::default::<BD>().rust_simd_tier::<BD, false>(),
        )];
        #[cfg(target_arch = "x86_64")]
        if flags.contains(CpuFlags::AVX2) {
            tiers.push((
                "avx2",
                Rav1dMCDSPContext::default::<BD>().rust_simd_tier::<BD, true>(),
            ));
        }
        #[cfg(all(target_arch = "aarch64", rav1d_sdot))]
        if flags.contains(CpuFlags::DOTPROD) {
            tiers.push((
                "dotprod",
                Rav1dMCDSPContext::default::<BD>().rust_simd_tier::<BD, true>(),
            ));
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        assert!(
            flags.contains(CpuFlags::SSE41),
            "the SIMD kernels need SSE4.1"
        );
        let _ = flags;
        (scalar, tiers)
    }

    /// `v` in a 64-byte aligned buffer, as pictures are.
    pub fn aligned<P: Copy>(v: &[P]) -> AlignedVec64<P> {
        let mut buf = AlignedVec64::<P>::new();
        buf.resize(v.len(), v[0]);
        buf.as_mut_slice().copy_from_slice(v);
        buf
    }

    pub fn pixels<BD: BitDepth>(rng: &mut Rng, n: usize, max: i32) -> Vec<BD::Pixel> {
        (0..n)
            .map(|_| rng.range(0, max).as_::<BD::Pixel>())
            .collect()
    }

    /// The block at `(x0, y0)` of a plane that ends with the last pixel the 8-tap filters
    /// read, so that a read past it panics.
    fn src_plane<BD: BitDepth>(
        rng: &mut Rng,
        w: usize,
        h: usize,
        max: i32,
    ) -> (Vec<BD::Pixel>, usize) {
        let (x0, y0) = (S - w - 4, 3);
        let len =
            ((y0 + h + 3) * S + x0 + w + 4).next_multiple_of(RAV1D_PICTURE_GUARANTEED_MULTIPLE);
        (pixels::<BD>(rng, len, max), y0 * S + x0)
    }

    fn mc_bd<BD: BitDepth>(max: i32) {
        let (scalar, tiers) = dsps::<BD>();
        let bd = BD::from_c(max);
        let mut rng = Rng(0x2545_f491_4f6c_dd1d ^ max as u64);
        for &w in &DIMS {
            for &h in &DIMS {
                for f in FILTERS {
                    for (mx, my) in [
                        (0, 0),
                        (rng.range(1, 15), 0),
                        (0, rng.range(1, 15)),
                        (rng.range(1, 15), rng.range(1, 15)),
                    ] {
                        let (sbuf, off) = src_plane::<BD>(&mut rng, w, h, max);
                        let mut sbuf = aligned(&sbuf);
                        let sp = Rav1dPictureDataComponent::wrap_buf::<BD>(sbuf.as_mut_slice(), S);
                        let src = sp.with_offset::<BD>() + off;
                        let fill = pixels::<BD>(
                            &mut rng,
                            (S * h).next_multiple_of(RAV1D_PICTURE_GUARANTEED_MULTIPLE),
                            max,
                        );
                        let mut want = aligned(&fill);
                        let dp = Rav1dPictureDataComponent::wrap_buf::<BD>(want.as_mut_slice(), S);
                        scalar.mc[f].call::<BD>(
                            dp.with_offset::<BD>() + 1usize,
                            src,
                            w as i32,
                            h as i32,
                            mx,
                            my,
                            bd,
                        );
                        let want = dp.bytes();
                        let mut tw = vec![7i16; w * h + 16];
                        scalar.mct[f].call::<BD>(&mut tw, src, w as i32, h as i32, mx, my, bd);
                        for (tier, dsp) in &tiers {
                            let mut got = aligned(&fill);
                            let dp =
                                Rav1dPictureDataComponent::wrap_buf::<BD>(got.as_mut_slice(), S);
                            dsp.mc[f].call::<BD>(
                                dp.with_offset::<BD>() + 1usize,
                                src,
                                w as i32,
                                h as i32,
                                mx,
                                my,
                                bd,
                            );
                            assert!(
                                dp.bytes() == want,
                                "put {tier} {max} {} {w}x{h} mx {mx} my {my}",
                                f as usize
                            );
                            let mut tg = vec![7i16; w * h + 16];
                            dsp.mct[f].call::<BD>(&mut tg, src, w as i32, h as i32, mx, my, bd);
                            assert!(
                                tg == tw,
                                "prep {tier} {max} {} {w}x{h} mx {mx} my {my}",
                                f as usize
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn mc_8tap_8bpc() {
        mc_bd::<BitDepth8>(255);
    }

    #[test]
    fn mc_8tap_10bpc() {
        mc_bd::<BitDepth16>(1023);
    }

    #[test]
    fn mc_8tap_12bpc() {
        mc_bd::<BitDepth16>(4095);
    }

    /// Compound inputs as `prep` leaves them, with the 8-tap overshoot.
    pub fn tmp(rng: &mut Rng, max: i32) -> Box<[i16; COMPINTER_LEN]> {
        let ib = if max == 4095 { 2 } else { 4 };
        let bias = if max == 255 { 0 } else { 8192 };
        let mut t = Box::new([0i16; COMPINTER_LEN]);
        for v in t.iter_mut() {
            *v = (rng.range(-max / 8, max + max / 8) * (1 << ib) - bias) as i16;
        }
        t
    }

    fn comp_bd<BD: BitDepth>(max: i32) {
        let (scalar, tiers) = dsps::<BD>();
        let bd = BD::from_c(max);
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ max as u64);
        let (t1, t2) = (tmp(&mut rng, max), tmp(&mut rng, max));
        for &w in &DIMS {
            for &h in &DIMS {
                let m: Vec<u8> = (0..w * h).map(|_| rng.range(0, 64) as u8).collect();
                let wt = rng.range(1, 15);
                let fill = pixels::<BD>(
                    &mut rng,
                    (S * h).next_multiple_of(RAV1D_PICTURE_GUARANTEED_MULTIPLE),
                    max,
                );
                let run = |dsp: &Rav1dMCDSPContext, k: usize| {
                    let mut buf = aligned(&fill);
                    let dp = Rav1dPictureDataComponent::wrap_buf::<BD>(buf.as_mut_slice(), S);
                    let dst = dp.with_offset::<BD>() + 3usize;
                    let (wi, hi) = (w as i32, h as i32);
                    match k {
                        0 => dsp.avg.call::<BD>(dst, &t1, &t2, wi, hi, bd),
                        1 => dsp.w_avg.call::<BD>(dst, &t1, &t2, wi, hi, wt, bd),
                        _ => dsp.mask.call::<BD>(dst, &t1, &t2, wi, hi, &m, bd),
                    }
                    dp.bytes()
                };
                let wm = |dsp: &Rav1dMCDSPContext, l: Rav1dPixelLayoutSubSampled, sign: i32| {
                    let mut buf = aligned(&fill);
                    let dp = Rav1dPictureDataComponent::wrap_buf::<BD>(buf.as_mut_slice(), S);
                    let mut m = Box::new([9u8; SEG_MASK_LEN]);
                    let dst = dp.with_offset::<BD>() + 3usize;
                    dsp.w_mask[l].call::<BD>(dst, &t1, &t2, w as i32, h as i32, &mut m, sign, bd);
                    (dp.bytes(), m)
                };
                for l in [
                    Rav1dPixelLayoutSubSampled::I420,
                    Rav1dPixelLayoutSubSampled::I422,
                    Rav1dPixelLayoutSubSampled::I444,
                ] {
                    for sign in 0..2 * (w >= 4 && h >= 4) as i32 {
                        let want = wm(&scalar, l, sign);
                        for (tier, dsp) in &tiers {
                            assert!(
                                wm(dsp, l, sign) == want,
                                "w_mask {tier} {max} {w}x{h} {sign}"
                            );
                        }
                    }
                }
                let tp = pixels::<BD>(&mut rng, SCRATCH_INTER_INTRA_BUF_LEN, max);
                let bl = |dsp: &Rav1dMCDSPContext, k: usize| {
                    let mut buf = aligned(&fill);
                    let dp = Rav1dPictureDataComponent::wrap_buf::<BD>(buf.as_mut_slice(), S);
                    let dst = dp.with_offset::<BD>() + 3usize;
                    let (wi, hi) = (w as i32, h as i32);
                    match k {
                        0 => dsp
                            .blend
                            .call::<BD>(dst, tp[..].try_into().unwrap(), wi, hi, &m),
                        1 => dsp.blend_v.call::<BD>(
                            dst,
                            tp[..SCRATCH_LAP_LEN].try_into().unwrap(),
                            wi,
                            hi,
                        ),
                        _ => dsp.blend_h.call::<BD>(
                            dst,
                            tp[..SCRATCH_LAP_LEN].try_into().unwrap(),
                            wi,
                            hi,
                        ),
                    }
                    dp.bytes()
                };
                for k in 0..3 {
                    if (k == 0 && (w > 32 || h > 32))
                        || (k == 1 && (w > 32 || w * h > SCRATCH_LAP_LEN))
                        || (k == 2 && (h > 32 || w * h > SCRATCH_LAP_LEN))
                    {
                        continue;
                    }
                    let want = bl(&scalar, k);
                    for (tier, dsp) in &tiers {
                        assert!(bl(dsp, k) == want, "blend {k} {tier} {max} {w}x{h}");
                    }
                }
                for k in 0..3 * (w >= 4 && h >= 4) as usize {
                    let want = run(&scalar, k);
                    for (tier, dsp) in &tiers {
                        assert!(run(dsp, k) == want, "comp {k} {tier} {max} {w}x{h}");
                    }
                }
            }
        }
    }

    /// ns per call of the Rust kernels, 8 and 10 bpc, on the widest tier.
    #[test]
    #[ignore]
    fn time_mc() {
        time_bd::<BitDepth8>(255);
        time_bd::<BitDepth16>(1023);
    }

    fn time_bd<BD: BitDepth>(max: i32) {
        use std::hint::black_box;
        use std::time::Instant;

        let (_, tiers) = dsps::<BD>();
        let dsps = vec![("rust", &tiers.last().unwrap().1)];
        let bd = BD::from_c(max);
        let mut rng = Rng(77);
        let mut sbuf = aligned(&pixels::<BD>(&mut rng, S * 160, max));
        let sp = Rav1dPictureDataComponent::wrap_buf::<BD>(sbuf.as_mut_slice(), S);
        let src = sp.with_offset::<BD>() + (4 * S + 8);
        // Aligned as the decoder's blocks and buffers are to their width.
        let mut dbuf = aligned(&pixels::<BD>(&mut rng, S * 160, max));
        let dp = Rav1dPictureDataComponent::wrap_buf::<BD>(dbuf.as_mut_slice(), S);
        let dst = dp.with_offset::<BD>();
        let (t1, t2) = (
            Box::new(Align64(*tmp(&mut rng, max))),
            Box::new(Align64(*tmp(&mut rng, max))),
        );
        let (t1, t2) = (&t1.0, &t2.0);
        let mut t = aligned(&[0i16; COMPINTER_LEN]);
        let t = t.as_mut_slice();
        let m: Vec<u8> = (0..COMPINTER_LEN).map(|_| rng.range(0, 64) as u8).collect();
        let m = aligned(&m);
        let m = m.as_slice();
        let mut sm = Box::new(Align64([0u8; SEG_MASK_LEN]));
        let sm = &mut sm.0;
        let tp = aligned(&pixels::<BD>(&mut rng, SCRATCH_INTER_INTRA_BUF_LEN, max));
        let abcd = [300i16, -200, 150, 250];
        let only = std::env::var("TIME_MC").unwrap_or_default();
        for k in 0..19 {
            let (k, f) = match k {
                0..13 => (k, Filter2d::Regular8Tap),
                _ => (k - 13, Filter2d::Sharp8Tap),
            };
            for &w in &[4, 8, 16, 32] {
                let (name, wi) = (
                    [
                        "put_hv",
                        "put_h",
                        "put_v",
                        "prep_hv",
                        "prep_h",
                        "prep_v",
                        "avg",
                        "w_avg",
                        "mask",
                        "w_mask420",
                        "blend_h",
                        "warp8x8",
                        "warp8x8t",
                    ][k],
                    w as i32,
                );
                let name = if f == Filter2d::Sharp8Tap {
                    format!("{name}#")
                } else {
                    name.to_string()
                };
                if (k >= 11 && w != 8) || (w == 4 && k >= 6 && k != 9) || !name.contains(&*only) {
                    continue;
                }
                let mut line = format!("{max:4} {name:10} {w:3}x{w:<3}");
                for (dn, dsp) in &dsps {
                    let n = 20000 * 8 / w;
                    let t0 = Instant::now();
                    for _ in 0..n {
                        match k {
                            0 => dsp.mc[f].call::<BD>(dst, src, wi, wi, 5, 9, bd),
                            1 => dsp.mc[f].call::<BD>(dst, src, wi, wi, 5, 0, bd),
                            2 => dsp.mc[f].call::<BD>(dst, src, wi, wi, 0, 9, bd),
                            3 => dsp.mct[f].call::<BD>(t, src, wi, wi, 5, 9, bd),
                            4 => dsp.mct[f].call::<BD>(t, src, wi, wi, 5, 0, bd),
                            5 => dsp.mct[f].call::<BD>(t, src, wi, wi, 0, 9, bd),
                            6 => dsp.avg.call::<BD>(dst, t1, t2, wi, wi, bd),
                            7 => dsp.w_avg.call::<BD>(dst, t1, t2, wi, wi, 5, bd),
                            8 => dsp.mask.call::<BD>(dst, t1, t2, wi, wi, m, bd),
                            9 => dsp.w_mask[Rav1dPixelLayoutSubSampled::I420]
                                .call::<BD>(dst, t1, t2, wi, wi, sm, 0, bd),
                            10 => dsp.blend_h.call::<BD>(
                                dst,
                                tp[..SCRATCH_LAP_LEN].try_into().unwrap(),
                                wi,
                                wi,
                            ),
                            11 => dsp.warp8x8.call::<BD>(dst, src, &abcd, 1000, -2000, bd),
                            _ => dsp.warp8x8t.call::<BD>(t, 8, src, &abcd, 1000, -2000, bd),
                        }
                        black_box(&*t);
                    }
                    line += &format!(
                        "  {dn} {:7.1} ns",
                        t0.elapsed().as_nanos() as f64 / n as f64
                    );
                }
                println!("{line}");
            }
        }
    }

    fn warp_bd<BD: BitDepth>(max: i32) {
        let (scalar, tiers) = dsps::<BD>();
        let bd = BD::from_c(max);
        let mut rng = Rng(0x1234_5678_9abc_def1 ^ max as u64);
        for _ in 0..2000 {
            let (sbuf, off) = src_plane::<BD>(&mut rng, 8, 8, max);
            let mut sbuf = aligned(&sbuf);
            let sp = Rav1dPictureDataComponent::wrap_buf::<BD>(sbuf.as_mut_slice(), S);
            let src = sp.with_offset::<BD>() + off;
            let abcd = [0; 4].map(|_| rng.range(-1000, 1000) as i16);
            let (mx, my) = (rng.range(-40000, 40000), rng.range(-40000, 40000));
            let fill = pixels::<BD>(
                &mut rng,
                (S * 8).next_multiple_of(RAV1D_PICTURE_GUARANTEED_MULTIPLE),
                max,
            );
            let run = |dsp: &Rav1dMCDSPContext| {
                let mut buf = aligned(&fill);
                let dp = Rav1dPictureDataComponent::wrap_buf::<BD>(buf.as_mut_slice(), S);
                dsp.warp8x8
                    .call::<BD>(dp.with_offset::<BD>() + 5usize, src, &abcd, mx, my, bd);
                let mut t = vec![3i16; 8 * 24];
                dsp.warp8x8t.call::<BD>(&mut t, 24, src, &abcd, mx, my, bd);
                (dp.bytes(), t)
            };
            let want = run(&scalar);
            for (tier, dsp) in &tiers {
                assert!(run(dsp) == want, "warp {tier} {max} {abcd:?} {mx} {my}");
            }
        }
    }

    #[test]
    fn warp_8bpc() {
        warp_bd::<BitDepth8>(255);
    }

    #[test]
    fn warp_10bpc() {
        warp_bd::<BitDepth16>(1023);
    }

    #[test]
    fn warp_12bpc() {
        warp_bd::<BitDepth16>(4095);
    }

    #[test]
    fn comp_8bpc() {
        comp_bd::<BitDepth8>(255);
    }

    #[test]
    fn comp_10bpc() {
        comp_bd::<BitDepth16>(1023);
    }

    #[test]
    fn comp_12bpc() {
        comp_bd::<BitDepth16>(4095);
    }
}
