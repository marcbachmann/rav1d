//! The base and base-range contexts of a transform block's coefficients, all at once as libaom
//! computes them: levels in a buffer with 4 rows of padding below each column, where a
//! neighbour past the column reads 0 as it does in rav1d before that level is decoded. A scalar
//! lane, and AVX2 and NEON lanes with its output.

use std::cmp;
use std::mem::MaybeUninit;

use strum::EnumCount;

use super::fdct_fast::Lane;
use crate::levels::{TxClass, TxfmSize};
use crate::scan::DAV1D_SCANS;
use crate::tables::{DAV1D_LO_CTX_OFFSETS, DAV1D_TXFM_DIMENSIONS};

pub const BUF: usize = 1280;

/// A transform size and class's coding order and level buffer.
pub struct Layout {
    /// Coefficient index, x, y and level buffer offset of each scan position.
    pub scan: Vec<(u16, u8, u8, u16)>,
    /// The furthest level buffer offset up to each scan position.
    reach: Vec<u16>,
    /// The base context's offset at each level buffer offset.
    offset: Vec<u8>,
    pub stride: usize,
    rows: usize,
    two_d: bool,
    transposed: bool,
}

/// rav1d's `scan` positions: coefficient index, x and y.
pub fn scan_rc(tx: TxfmSize, class: TxClass, i: usize) -> (usize, u8, u8) {
    let t_dim = &DAV1D_TXFM_DIMENSIONS[tx as usize];
    let slw = cmp::min(t_dim.lw, TxfmSize::S32x32 as u8);
    let slh = cmp::min(t_dim.lh, TxfmSize::S32x32 as u8);
    match class {
        TxClass::TwoD => {
            let rc = DAV1D_SCANS[tx as usize][i].get() as usize;
            (rc, (rc >> (slh + 2)) as u8, (rc & ((4 << slh) - 1)) as u8)
        }
        TxClass::H => {
            let x = (i & ((4 << slh) - 1)) as u8;
            let y = (i >> (slh + 2)) as u8;
            (i, x, y)
        }
        TxClass::V => {
            let x = (i & ((4 << slw) - 1)) as u8;
            let y = (i >> (slw + 2)) as u8;
            ((x as usize) << (slh + 2) | y as usize, x, y)
        }
    }
}

pub fn layout(tx: TxfmSize, class: TxClass) -> &'static Layout {
    static TABLES: std::sync::OnceLock<Vec<Layout>> = std::sync::OnceLock::new();
    let tables = TABLES.get_or_init(|| {
        (0..TxfmSize::COUNT * 3)
            .map(|k| {
                let tx = TxfmSize::from_repr(k / 3).unwrap();
                let class = [TxClass::TwoD, TxClass::H, TxClass::V][k % 3];
                let t_dim = &DAV1D_TXFM_DIMENSIONS[tx as usize];
                let slh = cmp::min(t_dim.lh, TxfmSize::S32x32 as u8);
                let n = cmp::min(t_dim.w as usize * 4, 32) * cmp::min(t_dim.h as usize * 4, 32);
                let two_d = class == TxClass::TwoD;
                let rows = 4 << slh;
                let stride = if two_d { rows + 4 } else { 20 };
                let scan: Vec<_> = (0..n)
                    .map(|i| {
                        let (rc, x, y) = scan_rc(tx, class, i);
                        (rc as u16, x, y, (x as usize * stride + y as usize) as u16)
                    })
                    .collect();
                let reach = scan.iter().scan(0, |m, s| {
                    *m = cmp::max(*m, s.3);
                    Some(*m)
                });
                let mut offset = vec![0; BUF];
                if two_d {
                    let is_rect = tx.is_rect() as usize;
                    let o = &DAV1D_LO_CTX_OFFSETS[is_rect + (tx as usize & is_rect)];
                    for &(_, x, y, off) in &scan {
                        offset[off as usize] = o[y as usize][x as usize].get();
                    }
                } else {
                    for &(_, _, y, off) in &scan {
                        offset[off as usize] = 26 + if y > 1 { 10 } else { y * 5 };
                    }
                }
                Layout { reach: reach.collect(), scan, offset, stride, rows, two_d, transposed: class == TxClass::H }
            })
            .collect()
    });
    let c = match class {
        TxClass::TwoD => 0,
        TxClass::H => 1,
        TxClass::V => 2,
    };
    &tables[tx as usize * 3 + c]
}

pub struct Scratch {
    levels: [MaybeUninit<u8>; BUF],
    base: [MaybeUninit<u8>; BUF],
    br: [MaybeUninit<u8>; BUF],
}

impl Scratch {
    pub fn new() -> Self {
        Self { levels: [MaybeUninit::uninit(); BUF], base: [MaybeUninit::uninit(); BUF], br: [MaybeUninit::uninit(); BUF] }
    }
}

fn pack(l: u32) -> u8 {
    ((cmp::min(l, 3) << 6) | cmp::min(l, 15)) as u8
}

/// The base context and the base-range context's magnitude part of every level buffer offset
/// up to the end of block's reach.
pub fn contexts<'a>(lay: &Layout, coefs: &[i32], eob: usize, s: &'a mut Scratch) -> (&'a [u8], &'a [u8]) {
    contexts_in(Lane::detect(), lay, coefs, eob, s)
}

pub fn contexts_in<'a>(lane: Lane, lay: &Layout, coefs: &[i32], eob: usize, s: &'a mut Scratch) -> (&'a [u8], &'a [u8]) {
    let reach = lay.reach[eob] as usize;
    let filled = cmp::min(BUF, reach + 2 * lay.stride + 32);
    let done = (reach | 15) + 1;
    for l in &mut s.levels[..filled] {
        l.write(0);
    }
    let cols = reach / lay.stride + 1;
    if lay.transposed {
        for rc in 0..=eob {
            s.levels[rc % lay.rows * lay.stride + rc / lay.rows].write(pack(coefs[rc].unsigned_abs()));
        }
    } else {
        assert!(coefs.len() >= cols * lay.rows);
        match lane {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Lane::Simd => {
                assert!(is_x86_feature_detected!("avx2"));
                // SAFETY: AVX2 is available; the columns fit `filled`.
                unsafe { x86::fill(coefs, lay.rows, lay.stride, cols, &mut s.levels[..filled]) }
            }
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is part of aarch64; the columns fit `filled`.
            Lane::Simd => unsafe { arm::fill(coefs, lay.rows, lay.stride, cols, &mut s.levels[..filled]) },
            _ => {
                for x in 0..cols {
                    for y in 0..lay.rows {
                        s.levels[x * lay.stride + y].write(pack(coefs[x * lay.rows + y].unsigned_abs()));
                    }
                }
            }
        }
    }
    // SAFETY: the first `filled` are written above.
    let levels = unsafe { &*(&s.levels[..filled] as *const [MaybeUninit<u8>] as *const [u8]) };
    let near = if lay.two_d { [1, lay.stride, lay.stride + 1, 2, 2 * lay.stride] } else { [1, lay.stride, 2, 3, 4] };
    match lane {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Lane::Simd => {
            // SAFETY: AVX2 is available; reads end below `filled`, writes below `done`.
            unsafe { x86::contexts(levels, &lay.offset, near, reach, &mut s.base, &mut s.br) }
        }
        #[cfg(target_arch = "aarch64")]
        // SAFETY: reads end below `filled`, writes below `done`.
        Lane::Simd => unsafe { arm::contexts(levels, &lay.offset, near, reach, &mut s.base, &mut s.br) },
        _ => {
            for off in 0..done {
                let l = |d: usize| levels[off + d] as u32;
                let hi = l(near[0]) + l(near[1]) + l(near[2]);
                let mag = hi + l(near[3]) + l(near[4]);
                s.base[off].write(lay.offset[off] + cmp::min((mag + 64) >> 7, 4) as u8);
                s.br[off].write(cmp::min(((hi & 63) + 1) >> 1, 6) as u8);
            }
        }
    }
    // SAFETY: both are written up to `done`.
    unsafe {
        (
            &*(&s.base[..done] as *const [MaybeUninit<u8>] as *const [u8]),
            &*(&s.br[..done] as *const [MaybeUninit<u8>] as *const [u8]),
        )
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #![allow(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]

    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    use std::mem::MaybeUninit;

    /// Packed levels of 4 coefficients: min(l, 3) << 6 | min(l, 15).
    #[inline(always)]
    unsafe fn pack4(c: __m128i) -> __m128i {
        let a = _mm_abs_epi32(c);
        let v = _mm_or_si128(_mm_slli_epi32::<6>(_mm_min_epu32(a, _mm_set1_epi32(3))), _mm_min_epu32(a, _mm_set1_epi32(15)));
        let v = _mm_packus_epi32(v, v);
        _mm_packus_epi16(v, v)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn fill(coefs: &[i32], rows: usize, stride: usize, cols: usize, levels: &mut [MaybeUninit<u8>]) {
        for x in 0..cols {
            let (src, dst) = (&coefs[x * rows..][..rows], &mut levels[x * stride..][..rows]);
            for y in (0..rows).step_by(4) {
                let v = pack4(_mm_loadu_si128(src[y..][..4].as_ptr().cast()));
                let b = (_mm_cvtsi128_si32(v) as u32).to_le_bytes();
                for (d, b) in dst[y..][..4].iter_mut().zip(b) {
                    d.write(b);
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn load(p: &[u8], at: usize) -> __m256i {
        _mm256_cvtepu8_epi16(_mm_loadu_si128(p[at..][..16].as_ptr().cast()))
    }

    #[inline(always)]
    unsafe fn narrow(v: __m256i) -> __m128i {
        _mm_packus_epi16(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v))
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn contexts(levels: &[u8], offset: &[u8], near: [usize; 5], reach: usize, base: &mut [MaybeUninit<u8>], br: &mut [MaybeUninit<u8>]) {
        for off in (0..=reach).step_by(16) {
            let hi = _mm256_add_epi16(_mm256_add_epi16(load(levels, off + near[0]), load(levels, off + near[1])), load(levels, off + near[2]));
            let mag = _mm256_add_epi16(hi, _mm256_add_epi16(load(levels, off + near[3]), load(levels, off + near[4])));
            let step = _mm256_min_epi16(_mm256_srli_epi16::<7>(_mm256_add_epi16(mag, _mm256_set1_epi16(64))), _mm256_set1_epi16(4));
            let b = _mm256_add_epi16(step, load(offset, off));
            let m = _mm256_and_si256(hi, _mm256_set1_epi16(63));
            let m = _mm256_min_epi16(_mm256_srli_epi16::<1>(_mm256_add_epi16(m, _mm256_set1_epi16(1))), _mm256_set1_epi16(6));
            _mm_storeu_si128(base[off..][..16].as_mut_ptr().cast(), narrow(b));
            _mm_storeu_si128(br[off..][..16].as_mut_ptr().cast(), narrow(m));
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    #![allow(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]

    use std::arch::aarch64::*;
    use std::mem::MaybeUninit;

    pub unsafe fn fill(coefs: &[i32], rows: usize, stride: usize, cols: usize, levels: &mut [MaybeUninit<u8>]) {
        for x in 0..cols {
            let (src, dst) = (&coefs[x * rows..][..rows], &mut levels[x * stride..][..rows]);
            for y in (0..rows).step_by(4) {
                let a = vreinterpretq_u32_s32(vabsq_s32(vld1q_s32(src[y..][..4].as_ptr())));
                let v = vorrq_u32(vshlq_n_u32::<6>(vminq_u32(a, vdupq_n_u32(3))), vminq_u32(a, vdupq_n_u32(15)));
                let v = vmovn_u16(vcombine_u16(vmovn_u32(v), vdup_n_u16(0)));
                let b = vget_lane_u32::<0>(vreinterpret_u32_u8(v)).to_le_bytes();
                for (d, b) in dst[y..][..4].iter_mut().zip(b) {
                    d.write(b);
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn half(levels: &[u8], offset: &[u8], near: [usize; 5], at: usize) -> (uint16x8_t, uint16x8_t) {
        let load = |d: usize| vmovl_u8(vld1_u8(levels[at + d..][..8].as_ptr()));
        let hi = vaddq_u16(vaddq_u16(load(near[0]), load(near[1])), load(near[2]));
        let mag = vaddq_u16(hi, vaddq_u16(load(near[3]), load(near[4])));
        let step = vminq_u16(vshrq_n_u16::<7>(vaddq_u16(mag, vdupq_n_u16(64))), vdupq_n_u16(4));
        let b = vaddq_u16(step, vmovl_u8(vld1_u8(offset[at..][..8].as_ptr())));
        let m = vandq_u16(hi, vdupq_n_u16(63));
        (b, vminq_u16(vshrq_n_u16::<1>(vaddq_u16(m, vdupq_n_u16(1))), vdupq_n_u16(6)))
    }

    pub unsafe fn contexts(levels: &[u8], offset: &[u8], near: [usize; 5], reach: usize, base: &mut [MaybeUninit<u8>], br: &mut [MaybeUninit<u8>]) {
        for off in (0..=reach).step_by(16) {
            let (b0, m0) = half(levels, offset, near, off);
            let (b1, m1) = half(levels, offset, near, off + 8);
            vst1q_u8(base[off..][..16].as_mut_ptr().cast(), vcombine_u8(vmovn_u16(b0), vmovn_u16(b1)));
            vst1q_u8(br[off..][..16].as_mut_ptr().cast(), vcombine_u8(vmovn_u16(m0), vmovn_u16(m1)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enc::frame::tests::Rng;

    #[test]
    fn lanes_agree() {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        for k in 0..TxfmSize::COUNT {
            let tx = TxfmSize::from_repr(k).unwrap();
            let t_dim = &DAV1D_TXFM_DIMENSIONS[tx as usize];
            for class in [TxClass::TwoD, TxClass::H, TxClass::V] {
                if class != TxClass::TwoD && t_dim.max >= TxfmSize::S32x32 as u8 {
                    continue;
                }
                let lay = layout(tx, class);
                let n = lay.scan.len();
                for _ in 0..300 {
                    let density = 1 + rng.below(8);
                    let mut coefs: Vec<i32> = (0..n)
                        .map(|_| match rng.below(64) {
                            0 => i32::MIN,
                            k if k < 8 * density => {
                                let v = if k % 3 == 0 { rng.below(1 << 20) } else { rng.below(20) } as i32;
                                if rng.below(2) == 0 { -v } else { v }
                            }
                            _ => 0,
                        })
                        .collect();
                    let eob = rng.below(n as u32) as usize;
                    for &(rc, ..) in &lay.scan[eob + 1..] {
                        coefs[rc as usize] = 0;
                    }
                    let (mut a, mut b) = (Scratch::new(), Scratch::new());
                    let want = contexts_in(Lane::Scalar, lay, &coefs, eob, &mut a);
                    let got = contexts_in(Lane::detect(), lay, &coefs, eob, &mut b);
                    assert!(want == got, "{tx:?} {class:?} eob {eob}");
                }
            }
        }
    }

    /// rav1d's `get_lo_ctx` on its unpadded buffer, levels decoded one by one.
    fn decoded_one_by_one(lay: &Layout, tx: TxfmSize, class: TxClass, coefs: &[i32], eob: usize) -> Vec<(usize, u8, u8)> {
        let t_dim = &DAV1D_TXFM_DIMENSIONS[tx as usize];
        let stride = if class == TxClass::TwoD { 4 << cmp::min(t_dim.lh, TxfmSize::S32x32 as u8) } else { 16 };
        let offsets = (class == TxClass::TwoD).then(|| {
            let is_rect = tx.is_rect() as usize;
            &DAV1D_LO_CTX_OFFSETS[is_rect + (tx as usize & is_rect)]
        });
        let mut levels = [0u8; 1088];
        let mut out = Vec::new();
        for i in (0..=eob).rev() {
            let (rc, x, y, _) = lay.scan[i];
            let at = x as usize * stride + y as usize;
            if i < eob {
                let l = |d: usize| levels[at + d] as u32;
                let mut mag = l(1) + l(stride);
                let (hi, offset) = match offsets {
                    Some(o) => {
                        mag += l(stride + 1);
                        let hi = mag;
                        mag += l(2) + l(2 * stride);
                        (hi, o[y as usize][x as usize].get())
                    }
                    None => {
                        mag += l(2);
                        let hi = mag;
                        mag += l(3) + l(4);
                        (hi, 26 + if y > 1 { 10 } else { y * 5 })
                    }
                };
                let base = offset + if mag > 512 { 4 } else { ((mag + 64) >> 7) as u8 };
                let br = cmp::min(((hi & 63) + 1) >> 1, 6) as u8;
                out.push((i, if i == 0 && offsets.is_some() { 0 } else { base }, br));
            }
            levels[at] = pack(coefs[rc as usize].unsigned_abs());
        }
        out
    }

    /// The contexts at once are those rav1d reads decoding the levels one by one into a buffer
    /// without padding.
    #[test]
    fn as_rav1d_decodes() {
        let mut rng = Rng(77);
        let mut reads = 0;
        for k in 0..TxfmSize::COUNT {
            let tx = TxfmSize::from_repr(k).unwrap();
            let t_dim = &DAV1D_TXFM_DIMENSIONS[tx as usize];
            for class in [TxClass::TwoD, TxClass::H, TxClass::V] {
                if class != TxClass::TwoD && t_dim.max >= TxfmSize::S32x32 as u8 {
                    continue;
                }
                let lay = layout(tx, class);
                let n = lay.scan.len();
                for _ in 0..200 {
                    let density = 1 + rng.below(8);
                    let mut coefs: Vec<i32> = (0..n).map(|_| if rng.below(8) < density { rng.below(20) as i32 } else { 0 }).collect();
                    let eob = rng.below(n as u32) as usize;
                    for &(rc, ..) in &lay.scan[eob + 1..] {
                        coefs[rc as usize] = 0;
                    }
                    let mut s = Scratch::new();
                    let (base, br) = contexts(lay, &coefs, eob, &mut s);
                    for (i, want_base, want_br) in decoded_one_by_one(lay, tx, class, &coefs, eob) {
                        let off = lay.scan[i].3 as usize;
                        let got_base = if i == 0 && class == TxClass::TwoD { 0 } else { base[off] };
                        assert_eq!((want_base, want_br), (got_base, br[off]), "{tx:?} {class:?} eob {eob} at {i}");
                        reads += 1;
                    }
                }
            }
        }
        assert!(reads > 500_000);
    }
}
