#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::{c_int, c_uint};
use std::{cmp, ptr};

use bitflags::bitflags;
use libc::ptrdiff_t;

use crate::align::AlignedVec64;
use crate::cpu::CpuFlags;
use crate::disjoint_mut::DisjointMut;
use crate::ffi_safe::FFISafe;
use crate::include::common::bitdepth::{AsPrimitive, BitDepth, DynPixel, LeftPixelRow2px};
use crate::include::common::intops::{apply_sign, iclip};
use crate::include::dav1d::picture::{
    FFISafeRav1dPictureDataComponentOffset, Rav1dPictureDataComponentOffset,
};
use crate::pic_or_buf::PicOrBuf;
use crate::strided::Strided as _;
use crate::tables::DAV1D_CDEF_DIRECTIONS;
use crate::with_offset::WithOffset;
use crate::wrap_fn_ptr::wrap_fn_ptr;

bitflags! {
    #[repr(transparent)]
    #[derive(Clone, Copy)]
    pub struct CdefEdgeFlags: u32 {
        const HAVE_LEFT = 1 << 0;
        const HAVE_RIGHT = 1 << 1;
        const HAVE_TOP = 1 << 2;
        const HAVE_BOTTOM = 1 << 3;
    }
}

wrap_fn_ptr!(pub unsafe extern "C" fn cdef(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    top_ptr: *const DynPixel,
    bottom_ptr: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
    _dst: FFISafeRav1dPictureDataComponentOffset,
    _top: WithOffset<*const FFISafe<DisjointMut<AlignedVec64<u8>>>>,
    _bottom: WithOffset<*const FFISafe<PicOrBuf<'_, AlignedVec64<u8>>>>,
) -> ());

pub type CdefTop<'a> = WithOffset<&'a DisjointMut<AlignedVec64<u8>>>;
pub type CdefBottom<'a> = WithOffset<PicOrBuf<'a, AlignedVec64<u8>>>;

impl cdef::Fn {
    /// CDEF operates entirely on pre-filter data.
    /// If bottom/right edges are present (according to `edges`),
    /// then the pre-filter data is located in `dst`.
    /// However, the edge pixels above `dst` may be post-filter,
    /// so in order to get access to pre-filter top pixels, use `top`.
    pub fn call<BD: BitDepth>(
        &self,
        dst: Rav1dPictureDataComponentOffset,
        left: &[LeftPixelRow2px<BD::Pixel>; 8],
        top: CdefTop,
        bottom: CdefBottom,
        pri_strength: c_int,
        sec_strength: u8,
        dir: c_int,
        damping: u8,
        edges: CdefEdgeFlags,
        bd: BD,
    ) {
        let dst_ptr = dst.as_mut_ptr::<BD>().cast();
        let stride = dst.stride();
        let left = ptr::from_ref(left).cast();
        let top_ptr = top.as_ptr::<BD>().cast();
        let bottom_ptr = bottom.wrapping_as_ptr::<BD>().cast();
        let sec_strength = sec_strength as c_int;
        let damping = damping as c_int;
        let bd = bd.into_c();

        let dst = dst.into_ffi_safe();
        let top = top.into_ffi_safe();
        let bottom = bottom.as_ref().into_ffi_safe();

        // SAFETY: Rust fallback is safe, asm is assumed to do the same.
        unsafe {
            self.get()(
                dst_ptr,
                stride,
                left,
                top_ptr,
                bottom_ptr,
                pri_strength,
                sec_strength,
                dir,
                damping,
                edges,
                bd,
                dst,
                top,
                bottom,
            )
        }
    }
}

wrap_fn_ptr!(pub unsafe extern "C" fn cdef_dir(
    dst_ptr: *const DynPixel,
    dst_stride: ptrdiff_t,
    variance: &mut c_uint,
    bitdepth_max: c_int,
    _dst: FFISafeRav1dPictureDataComponentOffset,
) -> c_int);

impl cdef_dir::Fn {
    pub fn call<BD: BitDepth>(
        &self,
        dst: Rav1dPictureDataComponentOffset,
        variance: &mut c_uint,
        bd: BD,
    ) -> c_int {
        let dst_ptr = dst.as_ptr::<BD>().cast();
        let dst_stride = dst.stride();
        let bd = bd.into_c();
        let dst = dst.into_ffi_safe();
        // SAFETY: Fallback `fn cdef_find_dir_rust` is safe; asm is supposed to do the same.
        unsafe { self.get()(dst_ptr, dst_stride, variance, bd, dst) }
    }
}

pub struct Rav1dCdefDSPContext {
    pub dir: cdef_dir::Fn,

    /// 444/luma, 422, 420
    pub fb: [cdef::Fn; 3],
}

#[inline]
pub fn constrain(diff: c_int, threshold: c_int, shift: c_int) -> c_int {
    let adiff = diff.abs();
    apply_sign(
        cmp::min(adiff, cmp::max(0, threshold - (adiff >> shift))),
        diff,
    )
}

const TMP_STRIDE: usize = 12;

#[inline]
pub fn fill(tmp: &mut [i16], w: usize, h: usize) {
    // Use a value that's a large positive number when interpreted as unsigned,
    // and a large negative number when interpreted as signed.
    for y in 0..h {
        tmp[y * TMP_STRIDE..][..w].fill(i16::MIN);
    }
}

#[expect(clippy::eq_op, reason = "easier to reason about")]
fn padding<BD: BitDepth>(
    tmp: &mut [i16; TMP_STRIDE * TMP_STRIDE],
    src: Rav1dPictureDataComponentOffset,
    left: &[LeftPixelRow2px<BD::Pixel>; 8],
    top: CdefTop,
    bottom: CdefBottom,
    w: usize,
    h: usize,
    edges: CdefEdgeFlags,
) {
    let top = top - 2_usize;
    let bottom = bottom - 2_usize;
    let stride = src.pixel_stride::<BD>();

    // Fill extended input buffer.
    let mut x_start = 2 - 2;
    let mut x_end = w + 2 + 2;
    let mut y_start = 2 - 2;
    let mut y_end = h + 2 + 2;
    if !edges.contains(CdefEdgeFlags::HAVE_TOP) {
        fill(tmp, w + 4, 2);
        y_start += 2;
    }
    if !edges.contains(CdefEdgeFlags::HAVE_BOTTOM) {
        fill(&mut tmp[(h + 2) * TMP_STRIDE..], w + 4, 2);
        y_end -= 2;
    }
    if !edges.contains(CdefEdgeFlags::HAVE_LEFT) {
        fill(&mut tmp[y_start * TMP_STRIDE..], 2, y_end - y_start);
        x_start += 2;
    }
    if !edges.contains(CdefEdgeFlags::HAVE_RIGHT) {
        fill(&mut tmp[y_start * TMP_STRIDE + w + 2..], 2, y_end - y_start);
        x_end -= 2;
    }

    for (i, y) in (y_start..2).enumerate() {
        let top = top + i as isize * stride;
        let top = top.data.slice_as::<_, BD::Pixel>((top.offset.., ..x_end));
        for x in x_start..x_end {
            tmp[x + y * TMP_STRIDE] = top[x].as_::<i16>();
        }
    }
    for y in 0..h {
        for x in x_start..2 {
            tmp[x + (y + 2) * TMP_STRIDE] = left[y][x].as_::<i16>();
        }
    }
    for y in 0..h {
        let tmp = &mut tmp[(y + 2) * TMP_STRIDE..];
        let src = src + (y as isize * stride);
        let src = &*src.slice::<BD>(x_end - 2);
        for x in 2..x_end {
            tmp[x] = src[x - 2].as_::<i16>();
        }
    }
    for (i, y) in (h + 2..y_end).enumerate() {
        let tmp = &mut tmp[y * TMP_STRIDE..];
        let bottom = bottom + i as isize * stride;
        // This is a fallback `fn`, so perf is not as important here, so an extra branch
        // here should be okay.
        let bottom = match bottom.data {
            PicOrBuf::Pic(pic) => &*pic.slice::<BD, _>((bottom.offset.., ..x_end)),
            PicOrBuf::Buf(buf) => &*buf.slice_as((bottom.offset.., ..x_end)),
        };
        for x in x_start..x_end {
            tmp[x] = bottom[x].as_::<i16>();
        }
    }
}

#[inline(never)]
fn cdef_filter_block_rust<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    left: &[LeftPixelRow2px<BD::Pixel>; 8],
    top: CdefTop,
    bottom: CdefBottom,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    w: usize,
    h: usize,
    edges: CdefEdgeFlags,
    bd: BD,
) {
    let dir = dir as usize;

    assert!((w == 4 || w == 8) && (h == 4 || h == 8));
    let mut tmp = [0; TMP_STRIDE * TMP_STRIDE]; // `12 * 12` is the maximum value of `TMP_STRIDE * (h + 4)`.

    padding::<BD>(&mut tmp, dst, left, top, bottom, w, h, edges);

    let tmp = tmp;
    let tmp_offset = 2 * TMP_STRIDE + 2;
    let tmp_index = |x: usize, offset: isize| (x + tmp_offset).wrapping_add_signed(offset);

    let dst = |y| {
        let dst = dst + (y as isize * dst.pixel_stride::<BD>());
        dst.slice_mut::<BD>(w)
    };

    if pri_strength != 0 {
        let bitdepth_min_8 = bd.bitdepth() - 8;
        let pri_tap = 4 - (pri_strength >> bitdepth_min_8 & 1);
        let pri_shift = cmp::max(0, damping - pri_strength.ilog2() as c_int);
        if sec_strength != 0 {
            let sec_shift = damping - sec_strength.ilog2() as c_int;
            for y in 0..h {
                let tmp = &tmp[y * TMP_STRIDE..];
                let dst = &mut *dst(y);
                for x in 0..w {
                    let px = dst[x].as_::<c_int>();
                    let mut sum = 0;
                    let mut max = px;
                    let mut min = px;
                    let mut pri_tap_k = pri_tap;
                    for k in 0..2 {
                        let off1 = DAV1D_CDEF_DIRECTIONS[dir + 2][k] as isize; // dir
                        let p0 = tmp[tmp_index(x, off1)] as c_int;
                        let p1 = tmp[tmp_index(x, -off1)] as c_int;
                        sum += pri_tap_k * constrain(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain(p1 - px, pri_strength, pri_shift);
                        // If `pri_tap_k == 4`, then it becomes 2, else it remains 3.
                        pri_tap_k = pri_tap_k & 3 | 2;
                        min = cmp::min(p0 as c_uint, min as c_uint) as c_int;
                        max = cmp::max(p0, max);
                        min = cmp::min(p1 as c_uint, min as c_uint) as c_int;
                        max = cmp::max(p1, max);
                        let off2 = DAV1D_CDEF_DIRECTIONS[dir + 4][k] as isize;
                        let off3 = DAV1D_CDEF_DIRECTIONS[dir + 0][k] as isize;
                        let s0 = tmp[tmp_index(x, off2)] as c_int;
                        let s1 = tmp[tmp_index(x, -off2)] as c_int;
                        let s2 = tmp[tmp_index(x, off3)] as c_int;
                        let s3 = tmp[tmp_index(x, -off3)] as c_int;
                        // `sec_tap` starts at 2 and becomes 1.
                        let sec_tap = 2 - k as c_int;
                        sum += sec_tap * constrain(s0 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain(s1 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain(s2 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain(s3 - px, sec_strength, sec_shift);
                        min = cmp::min(s0 as c_uint, min as c_uint) as c_int;
                        max = cmp::max(s0, max);
                        min = cmp::min(s1 as c_uint, min as c_uint) as c_int;
                        max = cmp::max(s1, max);
                        min = cmp::min(s2 as c_uint, min as c_uint) as c_int;
                        max = cmp::max(s2, max);
                        min = cmp::min(s3 as c_uint, min as c_uint) as c_int;
                        max = cmp::max(s3, max);
                    }
                    dst[x] = iclip(px + (sum - (sum < 0) as c_int + 8 >> 4), min, max)
                        .as_::<BD::Pixel>();
                }
            }
        } else {
            // pri_strength only
            for y in 0..h {
                let tmp = &tmp[y * TMP_STRIDE..];
                let dst = &mut *dst(y);
                for x in 0..w {
                    let px = dst[x].as_::<c_int>();
                    let mut sum = 0;
                    let mut pri_tap_k = pri_tap;
                    for k in 0..2 {
                        let off = DAV1D_CDEF_DIRECTIONS[dir + 2][k] as isize;
                        let p0 = tmp[tmp_index(x, off)] as c_int;
                        let p1 = tmp[tmp_index(x, -off)] as c_int;
                        sum += pri_tap_k * constrain(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain(p1 - px, pri_strength, pri_shift);
                        pri_tap_k = pri_tap_k & 3 | 2;
                    }
                    dst[x] = (px + (sum - (sum < 0) as c_int + 8 >> 4)).as_::<BD::Pixel>();
                }
            }
        }
    } else {
        // sec_strength only
        let sec_shift = damping - sec_strength.ilog2() as c_int;
        for y in 0..h {
            let tmp = &tmp[y * TMP_STRIDE..];
            let dst = &mut *dst(y);
            for x in 0..w {
                let px = dst[x].as_::<c_int>();
                let mut sum = 0;
                for k in 0..2 {
                    let off1 = DAV1D_CDEF_DIRECTIONS[dir + 4][k] as isize;
                    let off2 = DAV1D_CDEF_DIRECTIONS[dir + 0][k] as isize;
                    let s0 = tmp[tmp_index(x, off1)] as c_int;
                    let s1 = tmp[tmp_index(x, -off1)] as c_int;
                    let s2 = tmp[tmp_index(x, off2)] as c_int;
                    let s3 = tmp[tmp_index(x, -off2)] as c_int;
                    let sec_tap = 2 - k as c_int;
                    sum += sec_tap * constrain(s0 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain(s1 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain(s2 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain(s3 - px, sec_strength, sec_shift);
                }
                dst[x] = (px + (sum - (sum < 0) as c_int + 8 >> 4)).as_::<BD::Pixel>();
            }
        }
    };
}

/// # Safety
///
/// Must be called by [`cdef::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn cdef_filter_block_c_erased<BD: BitDepth, const W: usize, const H: usize>(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    _top_ptr: *const DynPixel,
    _bottom_ptr: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
    dst: FFISafeRav1dPictureDataComponentOffset,
    top: WithOffset<*const FFISafe<DisjointMut<AlignedVec64<u8>>>>,
    bottom: WithOffset<*const FFISafe<PicOrBuf<'_, AlignedVec64<u8>>>>,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `cdef::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };

    // SAFETY: Reverse of cast in `cdef::Fn::call`.
    let left = unsafe { &*left.cast() };

    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `cdef::Fn::call`.
    let top = unsafe { FFISafe::from_with_offset(top) };

    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `cdef::Fn::call`.
    let bottom = unsafe { FFISafe::from_with_offset(bottom) };

    let bd = BD::from_c(bitdepth_max);
    cdef_filter_block_rust(
        dst,
        left,
        top,
        bottom.map(|bot| *bot),
        pri_strength,
        sec_strength,
        dir,
        damping,
        W,
        H,
        edges,
        bd,
    )
}

/// # Safety
///
/// Must be called by [`cdef_dir::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn cdef_find_dir_c_erased<BD: BitDepth>(
    _img_ptr: *const DynPixel,
    _stride: ptrdiff_t,
    variance: &mut c_uint,
    bitdepth_max: c_int,
    img: FFISafeRav1dPictureDataComponentOffset,
) -> c_int {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `cdef_dir::Fn::call`.
    let img = unsafe { FFISafe::from_with_offset(img) };
    let bd = BD::from_c(bitdepth_max);
    cdef_find_dir_rust(img, variance, bd)
}

fn cdef_find_dir_rust<BD: BitDepth>(
    img: Rav1dPictureDataComponentOffset,
    variance: &mut c_uint,
    bd: BD,
) -> c_int {
    let bitdepth_min_8 = bd.bitdepth() - 8;
    let mut partial_sum_hv = [[0; 8]; 2];
    let mut partial_sum_diag = [[0; 15]; 2];
    let mut partial_sum_alt = [[0; 11]; 4];

    let (w, h) = (8, 8);
    for y in 0..h {
        let img = img + (y as isize * img.pixel_stride::<BD>());
        let img = &*img.slice::<BD>(w);
        for x in 0..w {
            let px = (img[x].as_::<c_int>() >> bitdepth_min_8) - 128;

            partial_sum_diag[0][y + x] += px;
            partial_sum_alt[0][y + (x >> 1)] += px;
            partial_sum_hv[0][y] += px;
            partial_sum_alt[1][3 + y - (x >> 1)] += px;
            partial_sum_diag[1][7 + y - x] += px;
            partial_sum_alt[2][3 - (y >> 1) + x] += px;
            partial_sum_hv[1][x] += px;
            partial_sum_alt[3][(y >> 1) + x] += px;
        }
    }

    let mut cost = [0; 8];
    for n in 0..8 {
        cost[2] += (partial_sum_hv[0][n] * partial_sum_hv[0][n]) as c_uint;
        cost[6] += (partial_sum_hv[1][n] * partial_sum_hv[1][n]) as c_uint;
    }
    cost[2] *= 105;
    cost[6] *= 105;

    static DIV_TABLE: [u16; 7] = [840, 420, 280, 210, 168, 140, 120];
    for n in 0..7 {
        let d = DIV_TABLE[n] as c_int;
        cost[0] += ((partial_sum_diag[0][n] * partial_sum_diag[0][n]
            + partial_sum_diag[0][14 - n] * partial_sum_diag[0][14 - n])
            * d) as c_uint;
        cost[4] += ((partial_sum_diag[1][n] * partial_sum_diag[1][n]
            + partial_sum_diag[1][14 - n] * partial_sum_diag[1][14 - n])
            * d) as c_uint;
    }
    cost[0] += (partial_sum_diag[0][7] * partial_sum_diag[0][7] * 105) as c_uint;
    cost[4] += (partial_sum_diag[1][7] * partial_sum_diag[1][7] * 105) as c_uint;

    for n in 0..4 {
        let cost_ptr = &mut cost[n * 2 + 1];
        for m in 0..5 {
            *cost_ptr += (partial_sum_alt[n][3 + m] * partial_sum_alt[n][3 + m]) as c_uint;
        }
        *cost_ptr *= 105;
        for m in 0..3 {
            let d = DIV_TABLE[2 * m + 1] as c_int;
            *cost_ptr += ((partial_sum_alt[n][m] * partial_sum_alt[n][m]
                + partial_sum_alt[n][10 - m] * partial_sum_alt[n][10 - m])
                * d) as c_uint;
        }
    }

    let mut best_dir = 0;
    let mut best_cost = cost[0];
    for n in 0..8 {
        if cost[n] > best_cost {
            best_cost = cost[n];
            best_dir = n;
        }
    }

    *variance = (best_cost - cost[best_dir ^ 4]) >> 10;
    best_dir as c_int
}

/// [`cdef_filter_block_rust`] eight `i16` lanes at a time: a row of an 8-wide block or two rows
/// of a 4-wide one.
mod rust_simd {
    use std::mem;

    use zerocopy::AsBytes;

    use super::*;
    use crate::include::common::bitdepth::BPC;
    use crate::simd128::{I16x8, I32x4};

    /// `constrain` as dav1d's asm has it: padding (`i16::MIN`) gives a difference of at least
    /// 28673 as `u16`, which every shift leaves above the threshold, so it adds nothing.
    #[inline(always)]
    fn constrain(diff: I16x8, threshold: I16x8, shift: u32) -> I16x8 {
        let adiff = diff.abs();
        threshold
            .sub_sat_u(adiff.shr_u(shift))
            .min_u(adiff)
            .with_sign_of(diff)
    }

    /// `PRI` and `SEC` say which strengths are not 0, so each of the three cases is its own loop.
    #[inline(always)]
    fn filter<BD: BitDepth, const W: usize, const PRI: bool, const SEC: bool>(
        tmp: &[i16; TMP_STRIDE * TMP_STRIDE],
        dst: Rav1dPictureDataComponentOffset,
        h: usize,
        pri_strength: c_int,
        sec_strength: c_int,
        dir: usize,
        damping: c_int,
        bd: BD,
    ) {
        let rows = if W == 8 { 1 } else { 2 };
        let pri = I16x8::splat(pri_strength as i16);
        let sec = I16x8::splat(sec_strength as i16);
        let pri_tap = 4 - (pri_strength >> (bd.bitdepth() - 8) & 1);
        let pri_taps = [
            I16x8::splat(pri_tap as i16),
            I16x8::splat((pri_tap & 3 | 2) as i16),
        ];
        let pri_shift = cmp::max(0, damping - (pri_strength | 1).ilog2() as c_int) as u32;
        let sec_shift = (damping - (sec_strength | 1).ilog2() as c_int) as u32;
        let round = I16x8::splat(8);
        for y in (0..h).step_by(rows) {
            let base = 2 * TMP_STRIDE + 2 + y * TMP_STRIDE;
            let at = |off: isize| {
                let i = base.wrapping_add_signed(off);
                if W == 8 {
                    I16x8::load(&tmp[i..])
                } else {
                    I16x8::load_halves(&tmp[i..], &tmp[i + TMP_STRIDE..])
                }
            };
            let px = at(0);
            let mut sum = I16x8::splat(0);
            let (mut min, mut max) = (px, px);
            for k in 0..2 {
                if PRI {
                    let off = DAV1D_CDEF_DIRECTIONS[dir + 2][k] as isize;
                    let (p0, p1) = (at(off), at(-off));
                    let c = constrain(p0.sub(px), pri, pri_shift).add(constrain(
                        p1.sub(px),
                        pri,
                        pri_shift,
                    ));
                    sum = sum.add(c.mul(pri_taps[k]));
                    min = min.min_u(p0).min_u(p1);
                    max = max.max(p0).max(p1);
                }
                if SEC {
                    let off2 = DAV1D_CDEF_DIRECTIONS[dir + 4][k] as isize;
                    let off3 = DAV1D_CDEF_DIRECTIONS[dir + 0][k] as isize;
                    let (s0, s1, s2, s3) = (at(off2), at(-off2), at(off3), at(-off3));
                    let mut c = constrain(s0.sub(px), sec, sec_shift)
                        .add(constrain(s1.sub(px), sec, sec_shift))
                        .add(constrain(s2.sub(px), sec, sec_shift))
                        .add(constrain(s3.sub(px), sec, sec_shift));
                    if k == 0 {
                        c = c.add(c);
                    }
                    sum = sum.add(c);
                    min = min.min_u(s0).min_u(s1).min_u(s2).min_u(s3);
                    max = max.max(s0).max(s1).max(s2).max(s3);
                }
            }
            let mut out = px.add(sum.add(sum.shr_s::<15>()).add(round).shr_s::<4>());
            if PRI && SEC {
                out = out.min(max).max(min);
            }
            let n = W * mem::size_of::<BD::Pixel>();
            let mut bytes = [0; 16];
            match BD::BPC {
                BPC::BPC8 => bytes[..8].copy_from_slice(&out.pack_u8()),
                BPC::BPC16 => bytes.copy_from_slice(out.to_array().as_bytes()),
            }
            for r in 0..rows {
                let dst = dst + ((y + r) as isize * dst.pixel_stride::<BD>());
                dst.slice_mut::<BD>(W)
                    .as_bytes_mut()
                    .copy_from_slice(&bytes[r * n..][..n]);
            }
        }
    }

    /// The block's `W` pixels at `px` into `row[..W]`.
    #[inline(always)]
    fn center<BD: BitDepth, const W: usize>(row: &mut [i16], px: &[BD::Pixel]) {
        let b = px[..W].as_bytes();
        let v = match (BD::BPC, W) {
            (BPC::BPC8, 8) => I16x8::from_u8x8(b),
            (BPC::BPC8, _) => I16x8::from_u8x4(b),
            (BPC::BPC16, 8) => I16x8::from_bytes(b),
            (BPC::BPC16, _) => I16x8::from_bytes_half(b),
        };
        if W == 8 {
            v.store(row)
        } else {
            v.store_lo(row)
        }
    }

    /// [`super::padding`] a row at a time, the block's pixels of each in one load.
    #[inline(always)]
    fn padding<BD: BitDepth, const W: usize, const H: usize>(
        tmp: &mut [i16; TMP_STRIDE * TMP_STRIDE],
        src: Rav1dPictureDataComponentOffset,
        left: &[LeftPixelRow2px<BD::Pixel>; 8],
        top: CdefTop,
        bottom: CdefBottom,
        edges: CdefEdgeFlags,
    ) {
        const NONE: i16 = i16::MIN;
        let stride = src.pixel_stride::<BD>();
        let has_left = edges.contains(CdefEdgeFlags::HAVE_LEFT);
        let has_right = edges.contains(CdefEdgeFlags::HAVE_RIGHT);
        let x_start = if has_left { 0 } else { 2 };
        let x_end = if has_right { W + 4 } else { W + 2 };
        let px = |p: BD::Pixel| p.as_::<i16>();
        // `line[0]` is column `x_start` of a padded row above or below the block.
        let outer = |row: &mut [i16], line: &[BD::Pixel]| {
            row[..2].copy_from_slice(&if has_left {
                [px(line[0]), px(line[1])]
            } else {
                [NONE; 2]
            });
            center::<BD, W>(&mut row[2..], &line[2 - x_start..]);
            let r = W + 2 - x_start;
            row[W + 2..W + 4].copy_from_slice(&if has_right {
                [px(line[r]), px(line[r + 1])]
            } else {
                [NONE; 2]
            });
        };

        if edges.contains(CdefEdgeFlags::HAVE_TOP) {
            for i in 0..2 {
                let top = top - 2_usize + i as isize * stride;
                let line = &*top
                    .data
                    .slice_as::<_, BD::Pixel>((top.offset + x_start.., ..x_end - x_start));
                outer(&mut tmp[i * TMP_STRIDE..], line);
            }
        } else {
            tmp[..2 * TMP_STRIDE].fill(NONE);
        }
        for y in 0..H {
            let row = &mut tmp[(y + 2) * TMP_STRIDE..];
            let line = &*(src + y as isize * stride).slice::<BD>(x_end - 2);
            row[..2].copy_from_slice(&if has_left { left[y].map(px) } else { [NONE; 2] });
            center::<BD, W>(&mut row[2..], line);
            row[W + 2..W + 4].copy_from_slice(&if has_right {
                [px(line[W]), px(line[W + 1])]
            } else {
                [NONE; 2]
            });
        }
        if edges.contains(CdefEdgeFlags::HAVE_BOTTOM) {
            for i in 0..2 {
                let bottom = bottom.clone() - 2_usize + i as isize * stride;
                let range = (bottom.offset + x_start.., ..x_end - x_start);
                let line = match bottom.data {
                    PicOrBuf::Pic(pic) => &*pic.slice::<BD, _>(range),
                    PicOrBuf::Buf(buf) => &*buf.slice_as(range),
                };
                outer(&mut tmp[(H + 2 + i) * TMP_STRIDE..], line);
            }
        } else {
            tmp[(H + 2) * TMP_STRIDE..(H + 4) * TMP_STRIDE].fill(NONE);
        }
    }

    #[cfg_attr(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature(enable = "sse4.1")
    )]
    fn filter_block<BD: BitDepth, const W: usize, const H: usize>(
        dst: Rav1dPictureDataComponentOffset,
        left: &[LeftPixelRow2px<BD::Pixel>; 8],
        top: CdefTop,
        bottom: CdefBottom,
        pri_strength: c_int,
        sec_strength: c_int,
        dir: c_int,
        damping: c_int,
        edges: CdefEdgeFlags,
        bd: BD,
    ) {
        let mut tmp = [0; TMP_STRIDE * TMP_STRIDE];
        padding::<BD, W, H>(&mut tmp, dst, left, top, bottom, edges);
        let dir = dir as usize;
        match (pri_strength != 0, sec_strength != 0) {
            (true, true) => filter::<BD, W, true, true>(
                &tmp,
                dst,
                H,
                pri_strength,
                sec_strength,
                dir,
                damping,
                bd,
            ),
            (true, false) => filter::<BD, W, true, false>(
                &tmp,
                dst,
                H,
                pri_strength,
                sec_strength,
                dir,
                damping,
                bd,
            ),
            _ => filter::<BD, W, false, true>(
                &tmp,
                dst,
                H,
                pri_strength,
                sec_strength,
                dir,
                damping,
                bd,
            ),
        }
    }

    /// `v` shifted up by `8 - B / 2` lanes into the 16 of `acc`, for `B` in `2..16`.
    #[inline(always)]
    fn shift_add<const B: i32>(acc: &mut [I16x8; 2], v: I16x8) {
        let z = I16x8::splat(0);
        acc[0] = acc[0].add(z.ext::<B>(v));
        acc[1] = acc[1].add(v.ext::<B>(z));
    }

    /// `v` shifted up by `n` lanes into the 16 of `acc`.
    macro_rules! shift_add {
        ($acc:expr, $v:expr, 0) => {
            $acc[0] = $acc[0].add($v)
        };
        ($acc:expr, $v:expr, $n:literal) => {
            shift_add::<{ 16 - 2 * $n }>(&mut $acc, $v)
        };
    }

    /// `Σ v[i]² * w[i]` over 16 lanes, in four `i32` lanes.
    #[inline(always)]
    fn weighted_squares(v: [I16x8; 2], w: &[i32; 16]) -> I32x4 {
        let (lo, hi) = (v[0].mul_wide(v[0]), v[1].mul_wide(v[1]));
        let w = |i: usize| I32x4::load(&w[4 * i..]);
        lo[0]
            .mul(w(0))
            .add(lo[1].mul(w(1)))
            .add(hi[0].mul(w(2)))
            .add(hi[1].mul(w(3)))
    }

    /// [`super::cdef_find_dir_rust`] on the rows of the block as `i16` lanes: each line sum
    /// is a row shifted along by its offset into 16 lanes. The diagonals and lines that run
    /// against the rows are summed in reverse order, which leaves their costs unchanged.
    #[cfg_attr(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature(enable = "sse4.1")
    )]
    fn find_dir<BD: BitDepth, const SHIFT: i32>(
        img: Rav1dPictureDataComponentOffset,
        variance: &mut c_uint,
    ) -> c_int {
        const DIAG: [i32; 16] = [
            840, 420, 280, 210, 168, 140, 120, 105, 120, 140, 168, 210, 280, 420, 840, 0,
        ];
        const ALT: [i32; 16] = [
            420, 210, 140, 105, 105, 105, 105, 105, 140, 210, 420, 0, 0, 0, 0, 0,
        ];
        let stride = img.pixel_stride::<BD>();
        let c128 = I16x8::splat(128);
        let r: [I16x8; 8] = std::array::from_fn(|y| {
            let row = &*(img + y as isize * stride).slice::<BD>(8);
            let b = row.as_bytes();
            let v = match BD::BPC {
                BPC::BPC8 => I16x8::from_u8x8(b),
                BPC::BPC16 => I16x8::from_bytes(b).shr_s::<SHIFT>(),
            };
            v.sub(c128)
        });
        let z = I16x8::splat(0);
        let mut diag = [[z; 2]; 2];
        let mut alt = [[z; 2]; 4];
        let mut hv1 = z;
        macro_rules! row {
            ($y:tt, $ry:tt, $k:tt, $rk:tt) => {{
                let v = r[$y];
                let pairs = v.hadd(z);
                hv1 = hv1.add(v);
                shift_add!(diag[0], v, $y);
                shift_add!(diag[1], v, $ry);
                shift_add!(alt[0], pairs, $y);
                shift_add!(alt[1], pairs, $ry);
                if $y % 2 == 1 {
                    let v = r[$y - 1].add(v);
                    shift_add!(alt[2], v, $rk);
                    shift_add!(alt[3], v, $k);
                }
            }};
        }
        row!(0, 7, 0, 3);
        row!(1, 6, 0, 3);
        row!(2, 5, 1, 2);
        row!(3, 4, 1, 2);
        row!(4, 3, 2, 1);
        row!(5, 2, 2, 1);
        row!(6, 1, 3, 0);
        row!(7, 0, 3, 0);
        let hv0 = r[0]
            .hadd(r[1])
            .hadd(r[2].hadd(r[3]))
            .hadd(r[4].hadd(r[5]).hadd(r[6].hadd(r[7])));
        let hv = |v: I16x8| {
            let s = v.mul_wide(v);
            s[0].add(s[1]).mul(I32x4::splat(105))
        };
        let c = [
            weighted_squares(diag[0], &DIAG),
            weighted_squares(alt[0], &ALT),
            hv(hv0),
            weighted_squares(alt[1], &ALT),
            weighted_squares(diag[1], &DIAG),
            weighted_squares(alt[2], &ALT),
            hv(hv1),
            weighted_squares(alt[3], &ALT),
        ];
        let mut cost = [0; 8];
        c[0].hadd(c[1]).hadd(c[2].hadd(c[3])).store(&mut cost[..4]);
        c[4].hadd(c[5]).hadd(c[6].hadd(c[7])).store(&mut cost[4..]);
        let cost = cost.map(|c| c as c_uint);
        let mut best_dir = 0;
        let mut best_cost = cost[0];
        for n in 1..8 {
            if cost[n] > best_cost {
                best_cost = cost[n];
                best_dir = n;
            }
        }
        *variance = (best_cost - cost[best_dir ^ 4]) >> 10;
        best_dir as c_int
    }

    /// # Safety
    ///
    /// Must be called by [`cdef_dir::Fn::call`], on x86 once the CPU flags hold SSE41.
    #[deny(unsafe_op_in_unsafe_fn)]
    #[allow(unused_unsafe)]
    pub unsafe extern "C" fn cdef_find_dir_simd_erased<BD: BitDepth>(
        _img_ptr: *const DynPixel,
        _stride: ptrdiff_t,
        variance: &mut c_uint,
        bitdepth_max: c_int,
        img: FFISafeRav1dPictureDataComponentOffset,
    ) -> c_int {
        // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `cdef_dir::Fn::call`.
        let img = unsafe { FFISafe::from_with_offset(img) };
        // SAFETY: SSE4.1 is checked before this `fn` is chosen on x86; elsewhere it is safe.
        unsafe {
            match (BD::BPC, BD::from_c(bitdepth_max).bitdepth()) {
                (BPC::BPC16, 12) => find_dir::<BD, 4>(img, variance),
                _ => find_dir::<BD, 2>(img, variance),
            }
        }
    }

    /// # Safety
    ///
    /// Must be called by [`cdef::Fn::call`], on x86 once the CPU flags hold SSE41.
    #[deny(unsafe_op_in_unsafe_fn)]
    #[allow(unused_unsafe)]
    pub unsafe extern "C" fn cdef_filter_block_simd_erased<
        BD: BitDepth,
        const W: usize,
        const H: usize,
    >(
        _dst_ptr: *mut DynPixel,
        _stride: ptrdiff_t,
        left: *const [LeftPixelRow2px<DynPixel>; 8],
        _top_ptr: *const DynPixel,
        _bottom_ptr: *const DynPixel,
        pri_strength: c_int,
        sec_strength: c_int,
        dir: c_int,
        damping: c_int,
        edges: CdefEdgeFlags,
        bitdepth_max: c_int,
        dst: FFISafeRav1dPictureDataComponentOffset,
        top: WithOffset<*const FFISafe<DisjointMut<AlignedVec64<u8>>>>,
        bottom: WithOffset<*const FFISafe<PicOrBuf<'_, AlignedVec64<u8>>>>,
    ) {
        // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `cdef::Fn::call`.
        let dst = unsafe { FFISafe::from_with_offset(dst) };
        // SAFETY: Reverse of cast in `cdef::Fn::call`.
        let left = unsafe { &*left.cast() };
        // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `cdef::Fn::call`.
        let top = unsafe { FFISafe::from_with_offset(top) };
        // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `cdef::Fn::call`.
        let bottom = unsafe { FFISafe::from_with_offset(bottom) };
        let bd = BD::from_c(bitdepth_max);
        // SAFETY: SSE4.1 is checked before this `fn` is chosen on x86; elsewhere it is safe.
        unsafe {
            filter_block::<BD, W, H>(
                dst,
                left,
                top,
                bottom.map(|bot| *bot),
                pri_strength,
                sec_strength,
                dir,
                damping,
                edges,
                bd,
            )
        }
    }
}

impl Rav1dCdefDSPContext {
    pub const fn default<BD: BitDepth>() -> Self {
        Self {
            dir: cdef_dir::Fn::new(cdef_find_dir_c_erased::<BD>),
            fb: [
                cdef::Fn::new(cdef_filter_block_c_erased::<BD, 8, 8>),
                cdef::Fn::new(cdef_filter_block_c_erased::<BD, 4, 8>),
                cdef::Fn::new(cdef_filter_block_c_erased::<BD, 4, 4>),
            ],
        }
    }


    #[inline(always)]
    const fn init<BD: BitDepth>(mut self, flags: CpuFlags) -> Self {
        use self::rust_simd::{cdef_filter_block_simd_erased, cdef_find_dir_simd_erased};

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if !flags.contains(CpuFlags::SSE41) {
            return self;
        }
        let _ = flags;
        if !crate::simd128::native() {
            return self;
        }

        self.dir = cdef_dir::Fn::new(cdef_find_dir_simd_erased::<BD>);
        self.fb = [
            cdef::Fn::new(cdef_filter_block_simd_erased::<BD, 8, 8>),
            cdef::Fn::new(cdef_filter_block_simd_erased::<BD, 4, 8>),
            cdef::Fn::new(cdef_filter_block_simd_erased::<BD, 4, 4>),
        ];
        self
    }

    pub const fn new<BD: BitDepth>(flags: CpuFlags) -> Self {
        Self::default::<BD>().init::<BD>(flags)
    }
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;
    use std::time::Instant;

    use super::*;
    use crate::cpu::{rav1d_get_cpu_flags, rav1d_init_cpu};
    use crate::include::common::bitdepth::{BitDepth16, BitDepth8};
    use crate::include::dav1d::picture::Rav1dPictureDataComponent;

    const STRIDE: usize = 64;
    const ROWS: usize = 32;
    const BX: usize = 16;
    const BY: usize = 8;
    const SIZES: [(usize, usize); 3] = [(8, 8), (4, 8), (4, 4)];

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 32) as u32
        }

        /// Mostly a smooth area with small noise, one in four a jump anywhere in range.
        fn pixel(&mut self, bits: u8) -> c_int {
            let max = (1 << bits) - 1;
            if self.next() % 4 == 0 {
                return (self.next() % (max as u32 + 1)) as c_int;
            }
            let mid = 1 << (bits - 1);
            (mid + ((self.next() % 41) as c_int - 20 << (bits - 8))).clamp(0, max)
        }
    }

    #[derive(Clone, Copy)]
    struct Case {
        size: usize,
        pri: c_int,
        sec: u8,
        dir: c_int,
        damping: u8,
        edges: CdefEdgeFlags,
    }

    /// Every edge set, direction and size with the strengths and dampings `cdef_apply` passes.
    fn cases(bits: u8) -> Vec<Case> {
        let s = bits - 8;
        let mut out = Vec::new();
        for edges in 0..16 {
            for dir in 0..8 {
                for size in 0..3 {
                    for pri in 0..16 {
                        for sec in [0, 1, 2, 4] {
                            for damping in 2..=6 {
                                if pri == 0 && sec == 0 {
                                    continue;
                                }
                                out.push(Case {
                                    size,
                                    pri: pri << s,
                                    sec: sec << s,
                                    dir,
                                    damping: damping + s,
                                    edges: CdefEdgeFlags::from_bits_truncate(edges),
                                });
                            }
                        }
                    }
                }
            }
        }
        out
    }

    struct Fixture<BD: BitDepth> {
        orig: Vec<BD::Pixel>,
        comp: Rav1dPictureDataComponent,
        _buf: AlignedVec64<BD::Pixel>,
        top: DisjointMut<AlignedVec64<u8>>,
        left: [LeftPixelRow2px<BD::Pixel>; 8],
    }

    impl<BD: BitDepth> Fixture<BD> {
        fn new(rng: &mut Rng, bits: u8) -> Self {
            let px = |rng: &mut Rng| rng.pixel(bits).as_::<BD::Pixel>();
            let orig: Vec<BD::Pixel> = (0..STRIDE * ROWS).map(|_| px(rng)).collect();
            let mut buf = AlignedVec64::<BD::Pixel>::new();
            buf.resize(STRIDE * ROWS, 0.as_::<BD::Pixel>());
            buf.as_mut_slice().copy_from_slice(&orig);
            let comp = Rav1dPictureDataComponent::wrap_buf::<BD>(buf.as_mut_slice(), STRIDE);
            let n = std::mem::size_of::<BD::Pixel>();
            let bytes: Vec<u8> = (0..2 * STRIDE)
                .flat_map(|_| (rng.pixel(bits) as u16).to_le_bytes().into_iter().take(n))
                .collect();
            let mut top = AlignedVec64::<u8>::new();
            top.resize(bytes.len(), 0);
            top.as_mut_slice().copy_from_slice(&bytes);
            let left = [[0; 2]; 8].map(|r: [u8; 2]| r.map(|_| px(rng)));
            Self {
                orig,
                comp,
                _buf: buf,
                top: DisjointMut::new(top),
                left,
            }
        }

        fn reset(&self) {
            self.comp
                .slice_mut::<BD, _>((0.., ..STRIDE * ROWS))
                .copy_from_slice(&self.orig);
        }

        fn plane(&self) -> Vec<BD::Pixel> {
            self.comp.slice::<BD, _>((0.., ..STRIDE * ROWS)).to_vec()
        }

        fn call(&self, dsp: &Rav1dCdefDSPContext, c: Case, bd: BD) {
            let h = SIZES[c.size].1;
            dsp.fb[c.size].call::<BD>(
                WithOffset {
                    data: &self.comp,
                    offset: BY * STRIDE + BX,
                },
                &self.left,
                WithOffset {
                    data: &self.top,
                    offset: BX,
                },
                WithOffset {
                    data: PicOrBuf::Pic(&self.comp),
                    offset: (BY + h) * STRIDE + BX,
                },
                c.pri,
                c.sec,
                c.dir,
                c.damping,
                c.edges,
                bd,
            );
        }
    }

    fn contexts<BD: BitDepth>() -> [(&'static str, Rav1dCdefDSPContext); 2] {
        rav1d_init_cpu();
        let flags = rav1d_get_cpu_flags();
        [
            ("scalar", Rav1dCdefDSPContext::default::<BD>()),
            ("rust simd", Rav1dCdefDSPContext::new::<BD>(flags)),
        ]
    }

    fn matches_scalar<BD: BitDepth>(bits: u8) {
        let bd = BD::from_c((1 << bits) - 1);
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let dsps = contexts::<BD>();
        for (i, c) in cases(bits).into_iter().enumerate() {
            if i % 512 == 0 {
                rng.next();
            }
            let fx = Fixture::<BD>::new(&mut rng, bits);
            fx.call(&dsps[0].1, c, bd);
            let want = fx.plane();
            for (name, dsp) in &dsps[1..] {
                fx.reset();
                fx.call(dsp, c, bd);
                assert!(
                    fx.plane() == want,
                    "{name} differs from scalar at {bits} bits, case {i}"
                );
            }
        }
    }

    #[test]
    fn rust_simd_matches_scalar_8bpc() {
        matches_scalar::<BitDepth8>(8);
    }

    #[test]
    fn rust_simd_matches_scalar_10bpc() {
        matches_scalar::<BitDepth16>(10);
    }

    #[test]
    fn rust_simd_matches_scalar_12bpc() {
        matches_scalar::<BitDepth16>(12);
    }

    /// The direction and variance of blocks of noise, flat blocks at both ends of the range,
    /// full-range noise and stripes along each direction.
    fn dir_matches_scalar<BD: BitDepth>(bits: u8) {
        let bd = BD::from_c((1 << bits) - 1);
        let max = (1 << bits) - 1;
        let mut rng = Rng(0x243f_6a88_85a3_08d3);
        let dsps = contexts::<BD>();
        for round in 0..20000 {
            let fx = Fixture::<BD>::new(&mut rng, bits);
            let kind = round % 8;
            let (dx, dy) = (rng.next() % 5, rng.next() % 5);
            for y in 0..8 {
                let row = (BY + y) * STRIDE + BX;
                let mut row = fx.comp.slice_mut::<BD, _>((row.., ..8));
                for x in 0..8 {
                    let v = match kind {
                        0 => 0,
                        1 => max,
                        2 => (rng.next() % (max as u32 + 1)) as c_int,
                        3 => [0, max][(x * dx as usize + y * dy as usize) / 3 % 2],
                        _ => continue,
                    };
                    row[x] = v.as_::<BD::Pixel>();
                }
            }
            let dst = WithOffset {
                data: &fx.comp,
                offset: BY * STRIDE + BX,
            };
            let mut want_var = 0;
            let want = dsps[0].1.dir.call::<BD>(dst, &mut want_var, bd);
            for (name, dsp) in &dsps[1..] {
                let mut var = 0;
                let dir = dsp.dir.call::<BD>(dst, &mut var, bd);
                assert_eq!(
                    (dir, var),
                    (want, want_var),
                    "{name} direction differs from scalar at {bits} bits, round {round}"
                );
            }
        }
    }

    #[test]
    fn dir_rust_simd_and_asm_match_scalar_8bpc() {
        dir_matches_scalar::<BitDepth8>(8);
    }

    #[test]
    fn dir_rust_simd_and_asm_match_scalar_10bpc() {
        dir_matches_scalar::<BitDepth16>(10);
    }

    #[test]
    fn dir_rust_simd_and_asm_match_scalar_12bpc() {
        dir_matches_scalar::<BitDepth16>(12);
    }

    /// ns per direction search, scalar, Rust and asm.
    fn time_dir<BD: BitDepth>(bits: u8) {
        let bd = BD::from_c((1 << bits) - 1);
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let fxs: Vec<_> = (0..64)
            .map(|_| Fixture::<BD>::new(&mut rng, bits))
            .collect();
        let mut line = format!("{bits:2}-bit dir:");
        for (name, dsp) in contexts::<BD>() {
            let mut best = f64::MAX;
            for _ in 0..20 {
                let start = Instant::now();
                for _ in 0..50 {
                    for fx in &fxs {
                        let dst = WithOffset {
                            data: &fx.comp,
                            offset: BY * STRIDE + BX,
                        };
                        let mut var = 0;
                        black_box(black_box(&dsp).dir.call::<BD>(dst, &mut var, bd));
                        black_box(var);
                    }
                }
                best = best.min(start.elapsed().as_secs_f64() * 1e9 / (50 * fxs.len()) as f64);
            }
            line += &format!("  {name} {best:6.1} ns");
        }
        println!("{line}");
    }

    #[test]
    #[ignore]
    fn time_dir_all() {
        time_dir::<BitDepth8>(8);
        time_dir::<BitDepth16>(10);
        time_dir::<BitDepth16>(12);
    }

    /// `cargo test --release --lib cdef::tests::time -- --ignored --nocapture`
    fn time<BD: BitDepth>(bits: u8) {
        let bd = BD::from_c((1 << bits) - 1);
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let fx = Fixture::<BD>::new(&mut rng, bits);
        let all = cases(bits);
        for size in 0..3 {
            let cases: Vec<Case> = all.iter().copied().filter(|c| c.size == size).collect();
            let mut line = format!("{bits:2}-bit {}x{}:", SIZES[size].0, SIZES[size].1);
            for (name, dsp) in contexts::<BD>() {
                let mut best = f64::MAX;
                for _ in 0..5 {
                    fx.reset();
                    let start = Instant::now();
                    for &c in &cases {
                        fx.call(black_box(&dsp), c, bd);
                    }
                    best = best.min(start.elapsed().as_secs_f64() * 1e9 / cases.len() as f64);
                }
                line += &format!("  {name} {best:6.1} ns");
            }
            let mut best = f64::MAX;
            for _ in 0..5 {
                let start = Instant::now();
                for &c in &cases {
                    let (w, h) = SIZES[c.size];
                    let mut tmp = [0; TMP_STRIDE * TMP_STRIDE];
                    padding::<BD>(
                        &mut tmp,
                        WithOffset {
                            data: &fx.comp,
                            offset: BY * STRIDE + BX,
                        },
                        &fx.left,
                        WithOffset {
                            data: &fx.top,
                            offset: BX,
                        },
                        WithOffset {
                            data: PicOrBuf::Pic(&fx.comp),
                            offset: (BY + h) * STRIDE + BX,
                        },
                        w,
                        h,
                        c.edges,
                    );
                    black_box(&tmp);
                }
                best = best.min(start.elapsed().as_secs_f64() * 1e9 / cases.len() as f64);
            }
            line += &format!("  (padding alone {best:6.1} ns)");
            println!("{line}");
        }
    }

    #[test]
    #[ignore]
    fn time_8bpc() {
        time::<BitDepth8>(8);
    }

    #[test]
    #[ignore]
    fn time_10bpc() {
        time::<BitDepth16>(10);
    }
}
