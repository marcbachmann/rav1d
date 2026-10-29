#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::{c_int, c_uint};
use std::ops::{Deref, DerefMut, Range};
use std::{mem, ptr, slice};

use cfg_if::cfg_if;

use crate::c_arc::CArc;
use crate::cpu::CpuFlags;
use crate::include::common::attributes::clz;
use crate::include::common::intops::{inv_recenter, ulog2};
use crate::simd128::I16x8;

#[cfg(all(feature = "asm", target_feature = "sse2"))]
extern "C" {
    fn dav1d_msac_decode_hi_tok_sse2(s: *mut MsacAsmContext, cdf: *mut u16) -> c_uint;
    fn dav1d_msac_decode_bool_sse2(s: *mut MsacAsmContext, f: c_uint) -> c_uint;
    fn dav1d_msac_decode_bool_equi_sse2(s: *mut MsacAsmContext) -> c_uint;
    fn dav1d_msac_decode_bool_adapt_sse2(s: *mut MsacAsmContext, cdf: *mut u16) -> c_uint;
    fn dav1d_msac_decode_symbol_adapt16_sse2(
        s: &mut MsacAsmContext,
        cdf: *mut u16,
        n_symbols: usize,
        _cdf_len: usize,
    ) -> c_uint;
    fn dav1d_msac_decode_symbol_adapt8_sse2(
        s: *mut MsacAsmContext,
        cdf: *mut u16,
        n_symbols: usize,
    ) -> c_uint;
    fn dav1d_msac_decode_symbol_adapt4_sse2(
        s: *mut MsacAsmContext,
        cdf: *mut u16,
        n_symbols: usize,
    ) -> c_uint;
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
extern "C" {
    fn dav1d_msac_decode_symbol_adapt16_avx2(
        s: &mut MsacAsmContext,
        cdf: *mut u16,
        n_symbols: usize,
        _cdf_len: usize,
    ) -> c_uint;
}

#[cfg(all(feature = "asm", target_feature = "neon"))]
extern "C" {
    fn dav1d_msac_decode_hi_tok_neon(s: *mut MsacAsmContext, cdf: *mut u16) -> c_uint;
    fn dav1d_msac_decode_bool_neon(s: *mut MsacAsmContext, f: c_uint) -> c_uint;
    fn dav1d_msac_decode_bool_equi_neon(s: *mut MsacAsmContext) -> c_uint;
    fn dav1d_msac_decode_bool_adapt_neon(s: *mut MsacAsmContext, cdf: *mut u16) -> c_uint;
    fn dav1d_msac_decode_symbol_adapt16_neon(
        s: *mut MsacAsmContext,
        cdf: *mut u16,
        n_symbols: usize,
    ) -> c_uint;
    fn dav1d_msac_decode_symbol_adapt8_neon(
        s: *mut MsacAsmContext,
        cdf: *mut u16,
        n_symbols: usize,
    ) -> c_uint;
    fn dav1d_msac_decode_symbol_adapt4_neon(
        s: *mut MsacAsmContext,
        cdf: *mut u16,
        n_symbols: usize,
    ) -> c_uint;
}

pub struct Rav1dMsacDSPContext {
    symbol_adapt16: unsafe extern "C" fn(
        s: &mut MsacAsmContext,
        cdf: *mut u16,
        n_symbols: usize,
        _cdf_len: usize,
    ) -> c_uint,
}

impl Rav1dMsacDSPContext {
    pub const fn default() -> Self {
        Self {
            symbol_adapt16: rav1d_msac_decode_symbol_adapt_c,
        }
    }

    #[cfg(all(feature = "asm", any(target_arch = "x86", target_arch = "x86_64")))]
    #[inline(always)]
    const fn init_x86(mut self, flags: CpuFlags) -> Self {
        if !flags.contains(CpuFlags::SSE2) {
            return self;
        }

        self.symbol_adapt16 = dav1d_msac_decode_symbol_adapt16_sse2;

        #[cfg(target_arch = "x86_64")]
        {
            if !flags.contains(CpuFlags::AVX2) {
                return self;
            }

            self.symbol_adapt16 = dav1d_msac_decode_symbol_adapt16_avx2;
        }

        self
    }

    #[cfg(all(feature = "asm", any(target_arch = "arm", target_arch = "aarch64")))]
    #[inline(always)]
    const fn init_arm(self, _flags: CpuFlags) -> Self {
        self
    }

    #[inline(always)]
    const fn init(self, flags: CpuFlags) -> Self {
        #[cfg(feature = "asm")]
        {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                return self.init_x86(flags);
            }
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            {
                return self.init_arm(flags);
            }
        }

        #[allow(unreachable_code)] // Reachable on some #[cfg]s.
        {
            let _ = flags;
            self
        }
    }

    pub const fn new(flags: CpuFlags) -> Self {
        Self::default().init(flags)
    }
}

impl Default for Rav1dMsacDSPContext {
    fn default() -> Self {
        Self::default()
    }
}

pub type EcWin = usize;

/// # Safety
///
/// [`Self`] must be the first field of [`MsacAsmContext`] for asm layout purposes,
/// and that [`MsacAsmContext`] must be a field of [`MsacContext`].
/// And [`Self::pos`] and [`Self::end`] must be either [`ptr::null`],
/// or [`Self::pos`] must point into (or the end of) [`MsacContext::data`],
/// and [`Self::end`] must point to the end of [`MsacContext::data`],
/// where [`MsacContext::data`] is part of the [`MsacContext`]
/// containing [`MsacAsmContext`] and thus also [`Self`].
#[repr(C)]
struct MsacAsmContextBuf {
    pos: *const u8,
    end: *const u8,
}

/// SAFETY: [`MsacAsmContextBuf`] is always contained in [`MsacAsmContext::buf`],
/// which is always contained in [`MsacContext::asm`], whose [`MsacContext::data`] field
/// is what is stored in [`MsacAsmContextBuf::pos`] and [`MsacAsmContextBuf::end`].
/// Since [`MsacContext::data`] is [`Send`], [`MsacAsmContextBuf`] is also [`Send`].
unsafe impl Send for MsacAsmContextBuf {}

/// SAFETY: [`MsacAsmContextBuf`] is always contained in [`MsacAsmContext::buf`],
/// which is always contained in [`MsacContext::asm`], whose [`MsacContext::data`] field
/// is what is stored in [`MsacAsmContextBuf::pos`] and [`MsacAsmContextBuf::end`].
/// Since [`MsacContext::data`] is [`Sync`], [`MsacAsmContextBuf`] is also [`Sync`].
unsafe impl Sync for MsacAsmContextBuf {}

impl Default for MsacAsmContextBuf {
    fn default() -> Self {
        Self {
            pos: ptr::null(),
            end: ptr::null(),
        }
    }
}

impl From<&[u8]> for MsacAsmContextBuf {
    fn from(value: &[u8]) -> Self {
        let Range { start, end } = value.as_ptr_range();
        Self { pos: start, end }
    }
}

#[repr(C)]
pub struct MsacAsmContext {
    buf: MsacAsmContextBuf,
    pub dif: EcWin,
    pub rng: c_uint,
    pub cnt: c_int,
    allow_update_cdf: c_int,
    #[cfg(all(feature = "asm", target_arch = "x86_64"))]
    symbol_adapt16: unsafe extern "C" fn(
        s: &mut MsacAsmContext,
        cdf: *mut u16,
        n_symbols: usize,
        _cdf_len: usize,
    ) -> c_uint,
}

impl Default for MsacAsmContext {
    fn default() -> Self {
        Self {
            buf: Default::default(),
            dif: Default::default(),
            rng: Default::default(),
            cnt: Default::default(),
            allow_update_cdf: Default::default(),

            #[cfg(all(feature = "asm", target_arch = "x86_64"))]
            symbol_adapt16: Rav1dMsacDSPContext::default().symbol_adapt16,
        }
    }
}

impl MsacAsmContext {
    fn allow_update_cdf(&self) -> bool {
        self.allow_update_cdf != 0
    }
}

#[derive(Default)]
pub struct MsacContext {
    asm: MsacAsmContext,
    data: Option<CArc<[u8]>>,
}

impl Deref for MsacContext {
    type Target = MsacAsmContext;

    fn deref(&self) -> &Self::Target {
        &self.asm
    }
}

impl DerefMut for MsacContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.asm
    }
}

impl MsacContext {
    pub fn data(&self) -> &[u8] {
        &**self.data.as_ref().unwrap()
    }

    pub fn buf_index(&self) -> usize {
        // We safely subtract instead of unsafely use `ptr::offset_from`
        // as asm sets `buf_pos`, so we don't need to rely on its safety,
        // and because codegen is no less optimal this way.
        self.buf.pos as usize - self.data().as_ptr() as usize
    }
}

/// Return value uses `n` bits.
#[inline]
pub fn rav1d_msac_decode_bools(s: &mut MsacContext, n: u8) -> c_uint {
    let mut v = 0;
    for _ in 0..n {
        v = v << 1 | rav1d_msac_decode_bool_equi(s) as c_uint;
    }
    v
}

#[inline]
pub fn rav1d_msac_decode_uniform(s: &mut MsacContext, n: c_uint) -> c_int {
    assert!(n > 0);
    let l = ulog2(n) as u8 + 1;
    assert!(l > 1);
    let m = (1 << l) - n;
    let v = rav1d_msac_decode_bools(s, l - 1);
    (if v < m {
        v
    } else {
        (v << 1) - m + rav1d_msac_decode_bool_equi(s) as c_uint
    }) as c_int
}

const EC_PROB_SHIFT: c_uint = 6;
const EC_MIN_PROB: c_uint = 4;
const _: () = assert!(EC_MIN_PROB <= (1 << EC_PROB_SHIFT) / 16);

const EC_WIN_SIZE: usize = mem::size_of::<EcWin>() << 3;

/// [`ctx_refill_bytes`] in one load while a whole window of bytes is left: the bytes the loop
/// would take land at the same bits, and the rest of the load is masked off.
#[inline]
fn ctx_refill(s: &mut MsacContext) {
    const W: usize = mem::size_of::<EcWin>();
    let c = EC_WIN_SIZE as c_int - 24 - s.cnt;
    debug_assert!(c >= 0 && c as usize <= EC_WIN_SIZE - 8);
    let buf = &s.data()[s.buf_index()..];
    // The fallback takes values, not `s`, so the caller's state can stay in registers across it.
    let (dif, cnt, read) = match buf.get(..W) {
        Some(next) => {
            let window = !EcWin::from_be_bytes(next.try_into().unwrap());
            let c = c as u32;
            (
                s.dif | window >> (EC_WIN_SIZE as u32 - 8 - c) & !((1 << (c & 7)) - 1),
                EC_WIN_SIZE as c_int - 24 - ((c & 7) as c_int - 8),
                c as usize / 8 + 1,
            )
        }
        None => ctx_refill_bytes(buf, s.dif, c),
    };
    s.dif = dif;
    s.cnt = cnt;
    s.buf.pos = s.buf.pos.wrapping_add(read);
}

/// The new `dif` and `cnt`, and the count of bytes read from `buf`.
#[cold]
fn ctx_refill_bytes(buf: &[u8], mut dif: EcWin, mut c: c_int) -> (EcWin, c_int, usize) {
    let mut read = 0;
    loop {
        let Some(&byte) = buf.get(read) else {
            // set remaining bits to 1;
            dif |= !(!(0xff as EcWin) << c);
            break;
        };
        dif |= ((byte ^ 0xff) as EcWin) << c;
        read += 1;
        c -= 8;
        if c < 0 {
            break;
        }
    }
    (dif, (EC_WIN_SIZE as c_int) - 24 - c, read)
}

#[inline]
fn ctx_norm(s: &mut MsacContext, dif: EcWin, rng: c_uint) {
    ctx_norm_by(s, dif, rng, 15 ^ (31 ^ clz(rng)));
}

#[inline]
fn ctx_norm_by(s: &mut MsacContext, dif: EcWin, rng: c_uint, d: c_int) {
    let cnt = s.cnt;
    debug_assert!(rng <= 65535);
    s.dif = dif << d;
    s.rng = rng << d;
    s.cnt = cnt - d;
    // unsigned compare avoids redundant refills at eob
    if (cnt as u32) < (d as u32) {
        ctx_refill(s);
    }
}

#[cfg_attr(
    all(feature = "asm", any(target_feature = "sse2", target_feature = "neon")),
    allow(dead_code)
)]
fn rav1d_msac_decode_bool_equi_rust(s: &mut MsacContext) -> bool {
    let r = s.rng;
    let dif = s.dif;
    debug_assert!(dif >> (EC_WIN_SIZE - 16) < r as EcWin);
    let (bit, dif, v) = split_at(dif, r, (r >> 8 << 7) + EC_MIN_PROB);
    // Half of an `r` of at least 0x8000 is at least 0x3ffc, so the shift is 0, 1 or 2, read off
    // the top bits without a bit scan, as in the asm.
    let d = (0xbfff - v as c_int) >> 14;
    debug_assert_eq!(d, 15 ^ (31 ^ clz(v)));
    ctx_norm_by(s, dif, v, d);
    bit
}

#[cfg_attr(
    all(feature = "asm", any(target_feature = "sse2", target_feature = "neon")),
    allow(dead_code)
)]
fn rav1d_msac_decode_bool_rust(s: &mut MsacContext, f: c_uint) -> bool {
    let r = s.rng;
    let dif = s.dif;
    debug_assert!(dif >> (EC_WIN_SIZE - 16) < r as EcWin);
    let v = ((r >> 8) * (f >> EC_PROB_SHIFT) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB;
    let (bit, dif, v) = split_at(dif, r, v);
    ctx_norm(s, dif, v);
    bit
}

/// The bit whose interval holds the window when the range `r` splits at `v`, and that interval
/// (window, range); selects, as in the asm, rather than products with the bit, which put two
/// more steps on the chain from one decode to the next.
#[inline(always)]
fn split_at(dif: EcWin, r: c_uint, v: c_uint) -> (bool, EcWin, c_uint) {
    let (above, below) = dif.overflowing_sub((v as EcWin) << (EC_WIN_SIZE - 16));
    if below {
        (true, dif, v)
    } else {
        (false, above, r - v)
    }
}

pub fn rav1d_msac_decode_subexp(s: &mut MsacContext, r#ref: c_uint, n: c_uint, mut k: u8) -> c_int {
    assert!(n >> k == 8);
    let mut a = 0;
    if rav1d_msac_decode_bool_equi(s) {
        if rav1d_msac_decode_bool_equi(s) {
            k += rav1d_msac_decode_bool_equi(s) as u8 + 1;
        }
        a = 1 << k;
    }
    let v = rav1d_msac_decode_bools(s, k) + a;
    (if r#ref * 2 <= n {
        inv_recenter(r#ref, v)
    } else {
        n - 1 - inv_recenter(n - 1 - r#ref, v)
    }) as c_int
}

/// Return value is in the range `0..=n_symbols`.
///
/// `n_symbols` is in the range `0..16`, so it is really a `u4`.
fn rav1d_msac_decode_symbol_adapt_rust(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    let c = (s.dif >> (EC_WIN_SIZE - 16)) as c_uint;
    let r = s.rng >> 8;
    let mut u;
    let mut v = s.rng;
    let mut val = 0;
    assert!(n_symbols < 16);
    assert!(cdf[n_symbols as usize] <= 32);
    loop {
        u = v;
        v = r * ((cdf[val as usize] >> EC_PROB_SHIFT) as c_uint);
        v >>= 7 - EC_PROB_SHIFT;
        v += EC_MIN_PROB * ((n_symbols as c_uint) - val);
        if !(c < v) {
            break;
        }
        val += 1;
    }
    assert!(u <= s.rng);
    ctx_norm(
        s,
        s.dif.wrapping_sub((v as EcWin) << (EC_WIN_SIZE - 16)),
        u - v,
    );
    if s.allow_update_cdf() {
        let count = cdf[n_symbols as usize];
        let rate = 4 + (count >> 4) + (n_symbols > 2) as u16;
        let val = val as usize;
        for cdf in &mut cdf[..val] {
            *cdf += (1 << 15) - *cdf >> rate;
        }
        for cdf in &mut cdf[val..n_symbols as usize] {
            *cdf -= *cdf >> rate;
        }
        cdf[n_symbols as usize] = count + (count < 32) as u16;
    }
    debug_assert!(val <= n_symbols as _);
    val as u8
}

/// Whether [`I16x8`] is vector instructions that need no runtime check here.
const MSAC_SIMD: bool = cfg!(any(
    target_feature = "sse2",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
));

#[cfg_attr(
    all(
        not(rav1d_rust_simd = "msac"),
        feature = "asm",
        any(target_feature = "sse2", target_feature = "neon")
    ),
    allow(dead_code)
)]
#[inline(always)]
fn decode_symbol_adapt<const LANES: usize>(
    s: &mut MsacContext,
    cdf: &mut [u16],
    n_symbols: u8,
) -> u8 {
    if MSAC_SIMD {
        rav1d_msac_decode_symbol_adapt_simd::<LANES>(s, cdf, n_symbols)
    } else {
        rav1d_msac_decode_symbol_adapt_rust(s, cdf, n_symbols)
    }
}

/// `EC_MIN_PROB * (n_symbols - i)` for lane `i` from `[15 - n_symbols..]`, 0 from `n_symbols` on.
static EC_MIN_PROBS: [i16; 31] = {
    let mut t = [0; 31];
    let mut i = 0;
    while i < 15 {
        t[i] = (EC_MIN_PROB * (15 - i as c_uint)) as i16;
        i += 1;
    }
    t
};

/// All ones at lane `n_symbols` from `[15 - n_symbols..]`.
static COUNT_LANE: [i16; 31] = {
    let mut t = [0; 31];
    t[15] = -1;
    t
};

/// [`rav1d_msac_decode_symbol_adapt_rust`] over all of the CDF's `LANES` at once: the symbol is the
/// first lane whose bound is not above the window. The count at `cdf[n_symbols]` makes that lane's
/// bound 0, so lanes past it are never read; like the asm, the adaptation leaves them alone only
/// while they hold 0, as every CDF table's padding does.
#[inline(always)]
fn rav1d_msac_decode_symbol_adapt_simd<const LANES: usize>(
    s: &mut MsacContext,
    cdf: &mut [u16],
    n_symbols: u8,
) -> u8 {
    let n = n_symbols as usize;
    assert!(n < LANES);
    let splat = if LANES == 4 {
        I16x8::splat_lo
    } else {
        I16x8::splat
    };
    let c = splat((s.dif >> (EC_WIN_SIZE - 16)) as u16 as i16);
    #[cfg(not(target_arch = "aarch64"))]
    let r = splat((s.rng & 0xff00) as u16 as i16);
    // `rng >> 8` less 128, the doubling multiply-high's operand.
    #[cfg(target_arch = "aarch64")]
    let r = splat((s.rng & 0x7f00) as u16 as i16);
    let zero = I16x8::splat(0);
    let (mut cdfs, mut at_or_past, mut vs) = ([zero; 2], [zero; 2], [zero; 2]);
    for k in 0..LANES.div_ceil(8) {
        cdfs[k] = if LANES == 4 {
            I16x8::load_u16_lo(cdf)
        } else {
            I16x8::load_u16(&cdf[8 * k..])
        };
        // (cdf >> 6 << 7) * (rng >> 8 << 8) >> 16 == (cdf >> 6) * (rng >> 8) >> 1
        let p = cdfs[k].and(I16x8::splat(-64));
        #[cfg(not(target_arch = "aarch64"))]
        let scaled = p.add(p).mulhi_u(r);
        // As the NEON asm: with `p = (cdf >> 6) << 6`, `p * (rng >> 8) >> 7` is
        // `p + (2 * p * ((rng >> 8) - 128 << 8) >> 16)`, one multiply.
        #[cfg(target_arch = "aarch64")]
        let scaled = p.add(p.qdmulh(r));
        vs[k] = scaled.add(I16x8::load(&EC_MIN_PROBS[15 - n + 8 * k..]));
        at_or_past[k] = vs[k].sub_sat_u(c).eq(zero);
    }
    #[cfg(target_arch = "aarch64")]
    let first_lo = |m: I16x8| m.first_set_lo();
    #[cfg(not(target_arch = "aarch64"))]
    let first_lo = |m: I16x8| m.first_set();
    let val = match LANES {
        4 => first_lo(at_or_past[0]),
        8 => at_or_past[0].first_set(),
        _ => at_or_past[0].first_set2(at_or_past[1]),
    } as usize;
    // At most `n_symbols`, as its lane always is at or past; the mask only drops the check.
    let val = val & 15;
    // The symbol's bounds are `[val]` and `[val + 1]` of `rng`, then the bounds.
    let bound_pair = || {
        let mut bounds = [0; 18];
        bounds[0] = s.rng as u16 as i16;
        for k in 0..LANES.div_ceil(8) {
            vs[k].store(&mut bounds[1 + 8 * k..]);
        }
        (bounds[val] as u16, bounds[val + 1] as u16)
    };
    // As the NEON asm: the pair shifted out of the packed lanes, without a trip through memory.
    // From the 64-bit half holding lane `val`, as a variable 128-bit shift compiles to ~20
    // instructions; `u` is that half moved up a lane under the lane below it, or `rng`.
    #[cfg(target_arch = "aarch64")]
    let (u, v) = if LANES <= 8 {
        let packed = vs[0].to_u128();
        let (lo, hi) = (packed as u64, (packed >> 64) as u64);
        let rng = s.rng as u16 as u64;
        let (us, vs) = if val < 4 {
            (lo << 16 | rng, lo)
        } else {
            (hi << 16 | lo >> 48, hi)
        };
        let shift = 16 * (val & 3);
        ((us >> shift) as u16, (vs >> shift) as u16)
    } else {
        bound_pair()
    };
    #[cfg(not(target_arch = "aarch64"))]
    let (u, v) = bound_pair();
    let (u, v) = (u as c_uint, v as c_uint);
    ctx_norm(
        s,
        s.dif.wrapping_sub((v as EcWin) << (EC_WIN_SIZE - 16)),
        u - v,
    );
    if s.allow_update_cdf() {
        let count = cdf[n];
        let rate = 4 + (count >> 4) as u32 + (n_symbols > 2) as u32;
        // The count's lane falls by `count >> rate` with the others; this restores it, plus one.
        let count_step = I16x8::splat(((count >> rate) + (count < 32) as u16) as i16);
        let ones = I16x8::splat(-1);
        for k in 0..LANES.div_ceil(8) {
            // As in the asm, in signed lanes: below the symbol `(32768 - cdf) >> rate` is added,
            // from it on `((-1 - cdf) >> rate) + 1`, which is `-(cdf >> rate)`. A probability of 0
            // would wrap, and no CDF holds one.
            let step = ones.avg_u(at_or_past[k]).sub(cdfs[k]).shr_s_by(rate);
            let count_lane = I16x8::load(&COUNT_LANE[15 - n + 8 * k..]);
            let updated = cdfs[k]
                .sub(at_or_past[k])
                .add(step)
                .add(count_step.and(count_lane));
            if LANES == 4 {
                updated.store_u16_lo(cdf);
            } else {
                updated.store_u16(&mut cdf[8 * k..]);
            }
        }
    }
    val as u8
}

/// # Safety
///
/// Must be called through [`Rav1dMsacDSPContext::symbol_adapt16`]
/// in [`rav1d_msac_decode_symbol_adapt16`].
#[cfg_attr(not(all(feature = "asm", target_arch = "x86_64")), allow(dead_code))]
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn rav1d_msac_decode_symbol_adapt_c(
    s: &mut MsacAsmContext,
    cdf: *mut u16,
    n_symbols: usize,
    cdf_len: usize,
) -> c_uint {
    // SAFETY: In the `rav1d_msac_decode_symbol_adapt16` caller,
    // `&mut s.asm` is passed, so we can reverse this to get back `s`.
    // The `.sub` is safe since were are subtracting the offset of `asm` within `s`,
    // so that will stay in bounds of the `s: MsacContext` allocated object.
    let s = unsafe {
        &mut *ptr::from_mut(s)
            .sub(mem::offset_of!(MsacContext, asm))
            .cast::<MsacContext>()
    };

    // SAFETY: This is only called from [`dav1d_msac_decode_symbol_adapt16`],
    // where it comes from `cdf.len()`.
    let cdf = unsafe { slice::from_raw_parts_mut(cdf, cdf_len) };

    rav1d_msac_decode_symbol_adapt_rust(s, cdf, n_symbols as u8) as c_uint
}

#[cfg_attr(
    all(feature = "asm", any(target_feature = "sse2", target_feature = "neon")),
    allow(dead_code)
)]
fn rav1d_msac_decode_bool_adapt_rust(s: &mut MsacContext, cdf: &mut [u16; 2]) -> bool {
    let bit = rav1d_msac_decode_bool_rust(s, cdf[0] as c_uint);
    if s.allow_update_cdf() {
        let count = cdf[1];
        let rate = 4 + (count >> 4);
        // Both steps, then a select: a branch on the decoded bit mispredicts as often as it is random.
        let (up, down) = (
            cdf[0] + ((1 << 15) - cdf[0] >> rate),
            cdf[0] - (cdf[0] >> rate),
        );
        cdf[0] = if bit { up } else { down };
        cdf[1] = count + (count < 32) as u16;
    }
    bit
}

/// Return value is in the range `0..=15`.
#[cfg_attr(
    all(feature = "asm", any(target_feature = "sse2", target_feature = "neon")),
    allow(dead_code)
)]
fn rav1d_msac_decode_hi_tok_rust(s: &mut MsacContext, cdf: &mut [u16; 4]) -> u8 {
    let mut tok_br = decode_symbol_adapt::<4>(s, cdf, 3);
    let mut tok = 3 + tok_br;
    if tok_br == 3 {
        tok_br = decode_symbol_adapt::<4>(s, cdf, 3);
        tok = 6 + tok_br;
        if tok_br == 3 {
            tok_br = decode_symbol_adapt::<4>(s, cdf, 3);
            tok = 9 + tok_br;
            if tok_br == 3 {
                tok = 12 + decode_symbol_adapt::<4>(s, cdf, 3);
            }
        }
    }
    tok
}

/// [`rav1d_msac_decode_hi_tok_rust`] as one loop over its up to four symbols, with the CDF in a
/// register and the decoder state in locals, as in the asm.
#[cfg_attr(
    all(
        not(rav1d_rust_simd = "msac"),
        feature = "asm",
        any(target_feature = "sse2", target_feature = "neon")
    ),
    allow(dead_code)
)]
#[inline(always)]
fn rav1d_msac_decode_hi_tok_simd(s: &mut MsacContext, cdf: &mut [u16; 4]) -> u8 {
    let update = s.allow_update_cdf();
    let zero = I16x8::splat(0);
    let ones = I16x8::splat(-1);
    let min_probs = I16x8::load(&EC_MIN_PROBS[15 - 3..]);
    let count_lane = I16x8::load(&COUNT_LANE[15 - 3..]);
    let mut cdfv = I16x8::load_u16_lo(cdf);
    let mut count = cdf[3];
    let (mut dif, mut rng) = (s.dif, s.rng);
    let mut tok = 3;
    loop {
        let c = I16x8::splat_lo((dif >> (EC_WIN_SIZE - 16)) as u16 as i16);
        let p = cdfv.and(I16x8::splat(-64));
        #[cfg(not(target_arch = "aarch64"))]
        let scaled = p.add(p).mulhi_u(I16x8::splat_lo((rng & 0xff00) as u16 as i16));
        // The doubling multiply-high of `rng >> 8` less 128, as in the symbol decoder: one
        // multiply on the path `rng` carries from one symbol to the next.
        #[cfg(target_arch = "aarch64")]
        let scaled = p.add(p.qdmulh(I16x8::splat((rng & 0x7f00) as u16 as i16)));
        let v = scaled.add(min_probs);
        let mut bounds = [0; 10];
        bounds[0] = rng as u16 as i16;
        v.store(&mut bounds[1..]);
        let at_or_past = v.sub_sat_u(c).eq(zero);
        let val = at_or_past.first_set() as usize & 3;
        if update {
            let rate = 5 + (count >> 4) as u32;
            let count_step = I16x8::splat(((count >> rate) + (count < 32) as u16) as i16);
            let step = ones.avg_u(at_or_past).sub(cdfv).shr_s_by(rate);
            cdfv = cdfv
                .sub(at_or_past)
                .add(step)
                .add(count_step.and(count_lane));
            count += (count < 32) as u16;
        }
        let u = bounds[val] as u16 as c_uint;
        let v = bounds[val + 1] as u16 as c_uint;
        dif = dif.wrapping_sub((v as EcWin) << (EC_WIN_SIZE - 16));
        rng = u - v;
        let d = 15 ^ (31 ^ clz(rng));
        dif <<= d;
        rng <<= d;
        let cnt = s.cnt;
        s.cnt = cnt - d;
        if (cnt as u32) < (d as u32) {
            s.dif = dif;
            ctx_refill(s);
            dif = s.dif;
        }
        tok += val as u8;
        if val < 3 || tok == 15 {
            break;
        }
    }
    if update {
        cdfv.store_u16_lo(cdf);
    }
    s.dif = dif;
    s.rng = rng;
    tok
}

impl MsacContext {
    pub fn new(data: CArc<[u8]>, disable_cdf_update_flag: bool, dsp: &Rav1dMsacDSPContext) -> Self {
        let asm = MsacAsmContext {
            buf: data.as_ref().into(),
            dif: 0,
            rng: 0x8000,
            cnt: -15,
            allow_update_cdf: (!disable_cdf_update_flag).into(),
            #[cfg(all(feature = "asm", target_arch = "x86_64"))]
            symbol_adapt16: dsp.symbol_adapt16,
        };
        let mut s = Self {
            asm,
            data: Some(data),
        };
        let _ = dsp.symbol_adapt16; // Silence unused warnings.
        ctx_refill(&mut s);
        s
    }
}

/// Return value is in the range `0..=n_symbols`.
///
/// `n_symbols` is in the range `0..4`.
#[inline(always)]
pub fn rav1d_msac_decode_symbol_adapt4(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    debug_assert!(n_symbols < 4);
    let ret;
    cfg_if! {
        if #[cfg(all(not(rav1d_rust_simd = "msac"), feature = "asm", target_feature = "sse2"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_symbol_adapt_rust`].
            ret = unsafe {
                dav1d_msac_decode_symbol_adapt4_sse2(&mut s.asm, cdf.as_mut_ptr(), n_symbols as usize)
            };
        } else if #[cfg(all(not(rav1d_rust_simd = "msac"), feature = "asm", target_feature = "neon"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_symbol_adapt_rust`].
            ret = unsafe {
                dav1d_msac_decode_symbol_adapt4_neon(&mut s.asm, cdf.as_mut_ptr(), n_symbols as usize)
            };
        } else {
            ret = decode_symbol_adapt::<4>(s, cdf, n_symbols);
        }
    }
    debug_assert!(ret < 4);
    ret as u8 % 4
}

/// Return value is in the range `0..=n_symbols`.
///
/// `n_symbols` is in the range `0..8`.
#[inline(always)]
pub fn rav1d_msac_decode_symbol_adapt8(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    debug_assert!(n_symbols < 8);
    let ret;
    cfg_if! {
        if #[cfg(all(not(rav1d_rust_simd = "msac"), feature = "asm", target_feature = "sse2"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_symbol_adapt_rust`].
            ret = unsafe {
                dav1d_msac_decode_symbol_adapt8_sse2(&mut s.asm, cdf.as_mut_ptr(), n_symbols as usize)
            };
        } else if #[cfg(all(not(rav1d_rust_simd = "msac"), feature = "asm", target_feature = "neon"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_symbol_adapt_rust`].
            ret = unsafe {
                dav1d_msac_decode_symbol_adapt8_neon(&mut s.asm, cdf.as_mut_ptr(), n_symbols as usize)
            };
        } else {
            ret = decode_symbol_adapt::<8>(s, cdf, n_symbols);
        }
    }
    debug_assert!(ret < 8);
    ret as u8 % 8
}

/// Return value is in the range `0..=n_symbols`.
///
/// `n_symbols` is in the range `0..16`.
#[inline(always)]
pub fn rav1d_msac_decode_symbol_adapt16(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    debug_assert!(n_symbols < 16);
    let ret;
    cfg_if! {
        if #[cfg(all(not(rav1d_rust_simd = "msac"), feature = "asm", target_arch = "x86_64"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_symbol_adapt_rust`].
            ret = unsafe {
                (s.symbol_adapt16)(&mut s.asm, cdf.as_mut_ptr(), n_symbols as usize, cdf.len())
            };
        } else if #[cfg(all(not(rav1d_rust_simd = "msac"), feature = "asm", target_feature = "sse2"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_symbol_adapt_rust`].
            ret = unsafe {
                dav1d_msac_decode_symbol_adapt16_sse2(&mut s.asm, cdf.as_mut_ptr(), n_symbols as usize, cdf.len())
            };
        } else if #[cfg(all(not(rav1d_rust_simd = "msac"), feature = "asm", target_feature = "neon"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_symbol_adapt_rust`].
            ret = unsafe {
                dav1d_msac_decode_symbol_adapt16_neon(&mut s.asm, cdf.as_mut_ptr(), n_symbols as usize)
            };
        } else {
            ret = decode_symbol_adapt::<16>(s, cdf, n_symbols);
        }
    }
    debug_assert!(ret < 16);
    ret as u8 % 16
}

#[inline]
pub fn rav1d_msac_decode_bool_adapt(s: &mut MsacContext, cdf: &mut [u16; 2]) -> bool {
    cfg_if! {
        if #[cfg(all(not(rav1d_rust_simd = "msac"), feature = "asm", target_feature = "sse2"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_bool_adapt_rust`].
            unsafe {
                dav1d_msac_decode_bool_adapt_sse2(&mut s.asm, cdf.as_mut_ptr()) != 0
            }
        } else if #[cfg(all(not(rav1d_rust_simd = "msac"), feature = "asm", target_feature = "neon"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_bool_adapt_rust`].
            unsafe {
                dav1d_msac_decode_bool_adapt_neon(&mut s.asm, cdf.as_mut_ptr()) != 0
            }
        } else {
            rav1d_msac_decode_bool_adapt_rust(s, cdf)
        }
    }
}

#[inline]
pub fn rav1d_msac_decode_bool_equi(s: &mut MsacContext) -> bool {
    cfg_if! {
        if #[cfg(all(not(rav1d_rust_simd = "msac"), feature = "asm", target_feature = "sse2"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_bool_equi_rust`].
            unsafe {
                dav1d_msac_decode_bool_equi_sse2(&mut s.asm) != 0
            }
        } else if #[cfg(all(not(rav1d_rust_simd = "msac"), feature = "asm", target_feature = "neon"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_bool_equi_rust`].
            unsafe {
                dav1d_msac_decode_bool_equi_neon(&mut s.asm) != 0
            }
        } else {
            rav1d_msac_decode_bool_equi_rust(s)
        }
    }
}

#[inline]
pub fn rav1d_msac_decode_bool(s: &mut MsacContext, f: c_uint) -> bool {
    cfg_if! {
        if #[cfg(all(not(rav1d_rust_simd = "msac"), feature = "asm", target_feature = "sse2"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_bool_rust`].
            unsafe {
                dav1d_msac_decode_bool_sse2(&mut s.asm, f) != 0
            }
        } else if #[cfg(all(not(rav1d_rust_simd = "msac"), feature = "asm", target_feature = "neon"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_bool_rust`].
            unsafe {
                dav1d_msac_decode_bool_neon(&mut s.asm, f) != 0
            }
        } else {
            rav1d_msac_decode_bool_rust(s, f)
        }
    }
}

/// Return value is in the range `0..16`.
#[inline(always)]
pub fn rav1d_msac_decode_hi_tok(s: &mut MsacContext, cdf: &mut [u16; 4]) -> u8 {
    let ret;
    cfg_if! {
        if #[cfg(all(not(rav1d_rust_simd = "msac"), feature = "asm", target_feature = "sse2"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_hi_tok_rust`].
            ret = (unsafe {
                dav1d_msac_decode_hi_tok_sse2(&mut s.asm, cdf.as_mut_ptr())
            }) as u8;
        } else if #[cfg(all(not(rav1d_rust_simd = "msac"), feature = "asm", target_feature = "neon"))] {
            // SAFETY: `checkasm` has verified that it is equivalent to [`dav1d_msac_decode_hi_tok_rust`].
            ret = unsafe {
                dav1d_msac_decode_hi_tok_neon(&mut s.asm, cdf.as_mut_ptr())
            } as u8;
        } else {
            ret = if MSAC_SIMD {
                rav1d_msac_decode_hi_tok_simd(s, cdf)
            } else {
                rav1d_msac_decode_hi_tok_rust(s, cdf)
            };
        }
    }
    debug_assert!(ret < 16);
    ret % 16
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;
    use std::time::Instant;

    use super::*;
    use crate::c_box::CRef;
    use crate::cpu::rav1d_get_cpu_flags;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 32) as u32
        }

        fn below(&mut self, n: u32) -> u32 {
            self.next() % n
        }
    }

    fn context(data: &[u8], update: bool) -> MsacContext {
        let data = CArc::wrap(CRef::Box(data.to_vec().into_boxed_slice())).unwrap();
        MsacContext::new(
            data,
            !update,
            &Rav1dMsacDSPContext::new(rav1d_get_cpu_flags()),
        )
    }

    /// Falling probabilities within 1..32768 for `n` symbols, as in every AV1 CDF, and the
    /// adaptation count after them; the lanes past it hold 0 as in the tables, or junk that no
    /// decoder may read.
    fn random_cdf(rng: &mut Rng, n: usize, junk: bool) -> [u16; 16] {
        let mut cdf: [u16; 16] = std::array::from_fn(|_| if junk { rng.next() as u16 } else { 0 });
        let mut p: Vec<u16> = (0..n)
            .map(|_| match rng.below(4) {
                0 => 1,
                1 => 32767,
                _ => 1 + rng.below(32767) as u16,
            })
            .collect();
        p.sort_unstable_by(|a, b| b.cmp(a));
        cdf[..n].copy_from_slice(&p);
        cdf[n] = rng.below(33) as u16;
        cdf
    }

    fn state(s: &MsacContext) -> (EcWin, c_uint, c_int, usize) {
        (s.dif, s.rng, s.cnt, s.buf_index())
    }

    #[test]
    fn symbol_adapt_simd_matches_scalar() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for round in 0..3000 {
            let data: Vec<u8> = (0..rng.below(80)).map(|_| rng.next() as u8).collect();
            let update = round % 5 != 0;
            let (mut a, mut b) = (context(&data, update), context(&data, update));
            for i in 0..300 {
                let lanes = [4, 8, 16][rng.below(3) as usize];
                let n = 1 + rng.below(lanes as u32 - 1) as u8;
                let junk = rng.below(2) == 0;
                let cdf = random_cdf(&mut rng, n as usize, junk);
                let (mut ca, mut cb) = (cdf, cdf);
                let va = rav1d_msac_decode_symbol_adapt_rust(&mut a, &mut ca[..=n as usize], n);
                let vb = match lanes {
                    4 => rav1d_msac_decode_symbol_adapt_simd::<4>(&mut b, &mut cb[..4], n),
                    8 => rav1d_msac_decode_symbol_adapt_simd::<8>(&mut b, &mut cb[..8], n),
                    _ => rav1d_msac_decode_symbol_adapt_simd::<16>(&mut b, &mut cb, n),
                };
                let at = format!("round {round} symbol {i}: {lanes} lanes, n {n}, cdf {cdf:?}");
                assert_eq!(va, vb, "{at}");
                let checked = if junk { n as usize + 1 } else { 16 };
                assert_eq!(ca[..checked], cb[..checked], "{at}");
                assert_eq!(state(&a), state(&b), "{at}");
            }
        }
    }

    #[test]
    fn hi_tok_loop_matches_four_symbols() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for round in 0..3000 {
            let data: Vec<u8> = (0..rng.below(80)).map(|_| rng.next() as u8).collect();
            let update = round % 5 != 0;
            let (mut a, mut b) = (context(&data, update), context(&data, update));
            for i in 0..200 {
                let mut cdf: [u16; 4] = random_cdf(&mut rng, 3, false)[..4].try_into().unwrap();
                if rng.below(2) == 0 {
                    // Mostly the last symbol, so the token runs on to 15.
                    cdf[..3].copy_from_slice(&[32767, 32766, 32765]);
                }
                let (mut ca, mut cb) = (cdf, cdf);
                let ta = rav1d_msac_decode_hi_tok_rust(&mut a, &mut ca);
                let tb = rav1d_msac_decode_hi_tok_simd(&mut b, &mut cb);
                let at = format!("round {round} token {i}: cdf {cdf:?}");
                assert_eq!(ta, tb, "{at}");
                assert_eq!(ca, cb, "{at}");
                assert_eq!(state(&a), state(&b), "{at}");
            }
        }
    }

    #[test]
    fn refill_in_one_load_matches_bytes() {
        let mut rng = Rng(7);
        for round in 0..100_000 {
            let data: Vec<u8> = (0..rng.below(24)).map(|_| rng.next() as u8).collect();
            let mut a = context(&data, true);
            let pos = rng.below(data.len() as u32 + 1) as usize;
            a.buf.pos = a.data().as_ptr().wrapping_add(pos);
            a.cnt = rng.below(EC_WIN_SIZE as u32 - 7) as c_int - 16;
            let c = EC_WIN_SIZE as c_int - 24 - a.cnt;
            let above = (!0 as EcWin).checked_shl(c as u32 + 8).unwrap_or(0);
            a.dif = ((rng.next() as EcWin) << (EC_WIN_SIZE - 32) | rng.next() as EcWin) & above;
            let mut b = context(&data, true);
            b.buf.pos = b.data().as_ptr().wrapping_add(pos);
            (b.cnt, b.dif) = (a.cnt, a.dif);
            ctx_refill(&mut a);
            let c = EC_WIN_SIZE as c_int - 24 - b.cnt;
            let read;
            (b.dif, b.cnt, read) = ctx_refill_bytes(&data[pos..], b.dif, c);
            b.buf.pos = b.buf.pos.wrapping_add(read);
            assert_eq!(
                state(&a),
                state(&b),
                "round {round}: {pos} of {} bytes",
                data.len()
            );
        }
    }

    fn time(name: &str, symbols: usize, mut f: impl FnMut()) {
        let mut best = f64::MAX;
        for _ in 0..7 {
            let t = Instant::now();
            f();
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!("{name:28} {:6.2} ns/call", best * 1e9 / symbols as f64);
    }

    /// `cargo test --release --lib msac::tests::time -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn time_symbol_adapt() {
        let mut rng = Rng(1);
        let data: Vec<u8> = (0..1 << 20).map(|_| rng.next() as u8).collect();
        const SYMBOLS: usize = 1 << 20;
        for (lanes, n) in [(4, 3), (8, 7), (16, 15), (16, 10)] {
            let cdfs: Vec<[u16; 16]> = (0..64).map(|_| random_cdf(&mut rng, n, false)).collect();
            let n = n as u8;
            let run = |kind: usize| {
                let mut s = context(&data, true);
                let mut cdfs = cdfs.clone();
                for i in 0..SYMBOLS {
                    let cdf = &mut cdfs[i % 64];
                    black_box(match (kind, lanes) {
                        (0, _) => rav1d_msac_decode_symbol_adapt_rust(&mut s, cdf, n),
                        (1, 4) => {
                            rav1d_msac_decode_symbol_adapt_simd::<4>(&mut s, &mut cdf[..4], n)
                        }
                        (1, 8) => {
                            rav1d_msac_decode_symbol_adapt_simd::<8>(&mut s, &mut cdf[..8], n)
                        }
                        (1, _) => rav1d_msac_decode_symbol_adapt_simd::<16>(&mut s, cdf, n),
                        (_, 4) => rav1d_msac_decode_symbol_adapt4(&mut s, &mut cdf[..4], n),
                        (_, 8) => rav1d_msac_decode_symbol_adapt8(&mut s, &mut cdf[..8], n),
                        _ => rav1d_msac_decode_symbol_adapt16(&mut s, cdf, n),
                    });
                }
            };
            for (kind, name) in [
                (0, "scalar"),
                (1, "rust simd"),
                (2, "dispatched (asm if built)"),
            ] {
                time(&format!("adapt{lanes} n {n} {name}"), SYMBOLS, || run(kind));
            }
        }
        let cdfs: Vec<[u16; 4]> = (0..64)
            .map(|_| random_cdf(&mut rng, 3, false)[..4].try_into().unwrap())
            .collect();
        for (kind, name) in [
            (0, "rust 4 calls"),
            (1, "rust loop"),
            (2, "dispatched (asm if built)"),
        ] {
            time(&format!("hi_tok {name}"), SYMBOLS, || {
                let mut s = context(&data, true);
                let mut cdfs = cdfs.clone();
                for i in 0..SYMBOLS {
                    let cdf = &mut cdfs[i % 64];
                    black_box(match kind {
                        0 => rav1d_msac_decode_hi_tok_rust(&mut s, cdf),
                        1 => rav1d_msac_decode_hi_tok_simd(&mut s, cdf),
                        _ => rav1d_msac_decode_hi_tok(&mut s, cdf),
                    });
                }
            });
        }
        let probs: Vec<[u16; 2]> = (0..64)
            .map(|_| [1 + rng.below(32767) as u16, rng.below(33) as u16])
            .collect();
        for (dispatched, name) in [(false, "rust"), (true, "dispatched (asm if built)")] {
            time(&format!("bool_adapt {name}"), SYMBOLS, || {
                let mut s = context(&data, true);
                let mut cdfs = probs.clone();
                for i in 0..SYMBOLS {
                    let cdf = &mut cdfs[i % 64];
                    black_box(if dispatched {
                        rav1d_msac_decode_bool_adapt(&mut s, cdf)
                    } else {
                        rav1d_msac_decode_bool_adapt_rust(&mut s, cdf)
                    });
                }
            });
            time(&format!("bool {name}"), SYMBOLS, || {
                let mut s = context(&data, true);
                for i in 0..SYMBOLS {
                    let f = probs[i % 64][0] as c_uint;
                    black_box(if dispatched {
                        rav1d_msac_decode_bool(&mut s, f)
                    } else {
                        rav1d_msac_decode_bool_rust(&mut s, f)
                    });
                }
            });
            time(&format!("bool_equi {name}"), SYMBOLS, || {
                let mut s = context(&data, true);
                for _ in 0..SYMBOLS {
                    black_box(if dispatched {
                        rav1d_msac_decode_bool_equi(&mut s)
                    } else {
                        rav1d_msac_decode_bool_equi_rust(&mut s)
                    });
                }
            });
        }
    }
}
