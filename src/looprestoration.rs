#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::{c_int, c_uint};
use std::ops::Add;
use std::{cmp, iter, mem, slice};

use bitflags::bitflags;
use libc::ptrdiff_t;
use to_method::To;
use zerocopy::{AsBytes, FromBytes, FromZeroes};

use crate::align::AlignedVec64;
use crate::cpu::CpuFlags;
use crate::cursor::CursorMut;
use crate::disjoint_mut::DisjointMut;
use crate::ffi_safe::FFISafe;
use crate::include::common::bitdepth::{
    AsPrimitive, BitDepth, DynPixel, LeftPixelRow, ToPrimitive, BPC,
};
use crate::include::common::intops::iclip;
use crate::include::dav1d::picture::{
    FFISafeRav1dPictureDataComponentOffset, Rav1dPictureDataComponentOffset,
};
use crate::strided::Strided as _;
use crate::tables::dav1d_sgr_x_by_x;
use crate::wrap_fn_ptr::wrap_fn_ptr;

bitflags! {
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct LrEdgeFlags: u8 {
        const LEFT = 1 << 0;
        const RIGHT = 1 << 1;
        const TOP = 1 << 2;
        const BOTTOM = 1 << 3;
    }
}

impl LrEdgeFlags {
    pub const fn select(&self, select: bool) -> Self {
        if select {
            *self
        } else {
            Self::empty()
        }
    }
}

#[derive(FromZeroes, FromBytes, AsBytes)]
#[repr(C)]
pub struct LooprestorationParamsSgr {
    pub s0: u32,
    pub s1: u32,
    pub w0: i16,
    pub w1: i16,
}

/// This [`zerocopy`]-based "`union`" has the same layout
/// as an actual `union` would, so it's safe to continue passing to asm,
/// but it's otherwise safe to use from Rust.
///
/// [`zerocopy`]: ::zerocopy
#[derive(Default)]
#[repr(C)]
#[repr(align(16))]
pub struct LooprestorationParams {
    /// [`Align16`] moved to [`Self`] because we can't `#[derive(`[`AsBytes`]`)]` on it due to generics.
    ///
    /// [`Align16`]: crate::align::Align16
    pub filter: [[i16; 8]; 2],
}

impl LooprestorationParams {
    pub fn sgr(&self) -> &LooprestorationParamsSgr {
        // These asserts ensure this is a no-op.
        const _: () = assert!(
            mem::size_of::<LooprestorationParams>() >= mem::size_of::<LooprestorationParamsSgr>()
        );
        let _: () = assert!(
            mem::align_of::<LooprestorationParams>() >= mem::align_of::<LooprestorationParamsSgr>()
        );
        FromBytes::ref_from_prefix(AsBytes::as_bytes(&self.filter)).unwrap()
    }

    pub fn sgr_mut(&mut self) -> &mut LooprestorationParamsSgr {
        // These asserts ensure this is a no-op.
        const _: () = assert!(
            mem::size_of::<LooprestorationParams>() >= mem::size_of::<LooprestorationParamsSgr>()
        );
        const _: () = assert!(
            mem::align_of::<LooprestorationParams>() >= mem::align_of::<LooprestorationParamsSgr>()
        );
        FromBytes::mut_from_prefix(AsBytes::as_bytes_mut(&mut self.filter)).unwrap()
    }
}

wrap_fn_ptr!(pub unsafe extern "C" fn loop_restoration_filter(
    dst_ptr: *mut DynPixel,
    dst_stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
    _dst: FFISafeRav1dPictureDataComponentOffset,
    _lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) -> ());

impl loop_restoration_filter::Fn {
    /// Although the spec applies restoration filters over 4x4 blocks,
    /// they can be applied to a bigger surface.
    ///
    /// * `w` is constrained by the restoration unit size (`w <= 256`).
    /// * `h` is constrained by the stripe height (`h <= 64`).
    ///
    /// The filter functions are allowed to do
    /// aligned writes past the right edge of the buffer,
    /// aligned up to the minimum loop restoration unit size
    /// (which is 32 pixels for subsampled chroma and 64 pixels for luma).
    pub fn call<BD: BitDepth>(
        &self,
        dst: Rav1dPictureDataComponentOffset,
        left: &[LeftPixelRow<BD::Pixel>],
        lpf: &DisjointMut<AlignedVec64<u8>>,
        lpf_off: isize,
        w: c_int,
        h: c_int,
        params: &LooprestorationParams,
        edges: LrEdgeFlags,
        bd: BD,
    ) {
        let dst_ptr = dst.as_mut_ptr::<BD>().cast();
        let dst_stride = dst.stride();
        let left = left[..h as usize].as_ptr().cast();
        // NOTE: The calculated pointer may point to before the beginning of
        // `lpf`, so we must use `.wrapping_offset` here. `.wrapping_offset` is
        // needed since `.offset` requires the pointer to be in bounds, which
        // `.wrapping_offset` does not, and delays that requirement to when the
        // pointer is dereferenced.
        let lpf_ptr = lpf
            .as_mut_ptr()
            .cast::<BD::Pixel>()
            .wrapping_offset(lpf_off)
            .cast();
        let bd = bd.into_c();
        let dst = dst.into_ffi_safe();
        let lpf = FFISafe::new(lpf);
        // SAFETY: Fallbacks `fn wiener_rust`, `fn sgr_{3x3,5x5,mix}_rust` are safe; asm is supposed to do the same.
        unsafe {
            self.get()(
                dst_ptr, dst_stride, left, lpf_ptr, w, h, params, edges, bd, dst, lpf,
            )
        }
    }
}

pub struct Rav1dLoopRestorationDSPContext {
    pub wiener: [loop_restoration_filter::Fn; 2],
    pub sgr: [loop_restoration_filter::Fn; 3],
}

const REST_UNIT_STRIDE: usize = 256 * 3 / 2 + 3 + 3;

// TODO Reuse p when no padding is needed (add and remove lpf pixels in p)
// TODO Chroma only requires 2 rows of padding.
#[inline(never)]
fn padding<BD: BitDepth>(
    dst: &mut [BD::Pixel; (64 + 3 + 3) * REST_UNIT_STRIDE],
    p: Rav1dPictureDataComponentOffset,
    left: &[LeftPixelRow<BD::Pixel>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    unit_w: usize,
    stripe_h: usize,
    edges: LrEdgeFlags,
) {
    let left = &left[..stripe_h];
    assert!(stripe_h > 0);
    let stride = p.pixel_stride::<BD>();

    let [have_left, have_right, have_top, have_bottom] = [
        LrEdgeFlags::LEFT,
        LrEdgeFlags::RIGHT,
        LrEdgeFlags::TOP,
        LrEdgeFlags::BOTTOM,
    ]
    .map(|lr_have| edges.contains(lr_have));
    let [have_left_3, have_right_3] = [have_left, have_right].map(|have| 3 * have as usize);

    // Copy more pixels if we don't have to pad them
    let unit_w = unit_w + have_left_3 + have_right_3;
    let dst_l = &mut dst[3 - have_left_3..];
    let p = p - have_left_3;
    let lpf_off = lpf_off - (have_left_3 as isize);
    let abs_stride = stride.unsigned_abs();

    if have_top {
        // Copy previous loop filtered rows
        let lpf_guard;
        let (above_1, above_2) = if stride < 0 {
            lpf_guard = lpf
                .slice_as::<_, BD::Pixel>(((lpf_off + stride) as usize.., ..abs_stride + unit_w));
            let above_2 = &*lpf_guard;
            let above_1 = &above_2[abs_stride..];
            (above_1, above_2)
        } else {
            lpf_guard = lpf.slice_as((lpf_off as usize.., ..abs_stride + unit_w));
            let above_1 = &*lpf_guard;
            let above_2 = &above_1[abs_stride..];
            (above_1, above_2)
        };
        BD::pixel_copy(dst_l, above_1, unit_w);
        BD::pixel_copy(&mut dst_l[REST_UNIT_STRIDE..], above_1, unit_w);
        BD::pixel_copy(&mut dst_l[2 * REST_UNIT_STRIDE..], above_2, unit_w);
    } else {
        // Pad with first row
        let p = &*p.slice::<BD>(unit_w);
        BD::pixel_copy(dst_l, p, unit_w);
        BD::pixel_copy(&mut dst_l[REST_UNIT_STRIDE..], p, unit_w);
        BD::pixel_copy(&mut dst_l[2 * REST_UNIT_STRIDE..], p, unit_w);
        if have_left {
            let left = &left[0][1..];
            BD::pixel_copy(dst_l, left, 3);
            BD::pixel_copy(&mut dst_l[REST_UNIT_STRIDE..], left, left.len());
            BD::pixel_copy(&mut dst_l[2 * REST_UNIT_STRIDE..], left, left.len());
        }
    }

    let dst_tl = &mut dst_l[3 * REST_UNIT_STRIDE..];
    if have_bottom {
        // Copy next loop filtered rows
        let offset = lpf_off + (6 + if stride < 0 { 1 } else { 0 }) * stride;
        let lpf = &*lpf.slice_as((offset as usize.., ..abs_stride + unit_w));
        let (below_1, below_2) = if stride < 0 {
            (&lpf[abs_stride..], lpf)
        } else {
            (lpf, &lpf[abs_stride..])
        };
        BD::pixel_copy(&mut dst_tl[stripe_h * REST_UNIT_STRIDE..], below_1, unit_w);
        BD::pixel_copy(
            &mut dst_tl[(stripe_h + 1) * REST_UNIT_STRIDE..],
            below_2,
            unit_w,
        );
        BD::pixel_copy(
            &mut dst_tl[(stripe_h + 2) * REST_UNIT_STRIDE..],
            below_2,
            unit_w,
        );
    } else {
        // Pad with last row
        let src = p + ((stripe_h - 1) as isize * stride);
        let src = &*src.slice::<BD>(unit_w);
        BD::pixel_copy(&mut dst_tl[stripe_h * REST_UNIT_STRIDE..], src, unit_w);
        BD::pixel_copy(
            &mut dst_tl[(stripe_h + 1) * REST_UNIT_STRIDE..],
            src,
            unit_w,
        );
        BD::pixel_copy(
            &mut dst_tl[(stripe_h + 2) * REST_UNIT_STRIDE..],
            src,
            unit_w,
        );
        if have_left {
            let left = &left[stripe_h - 1][1..];
            BD::pixel_copy(&mut dst_tl[stripe_h * REST_UNIT_STRIDE..], left, left.len());
            BD::pixel_copy(
                &mut dst_tl[(stripe_h + 1) * REST_UNIT_STRIDE..],
                left,
                left.len(),
            );
            BD::pixel_copy(
                &mut dst_tl[(stripe_h + 2) * REST_UNIT_STRIDE..],
                left,
                left.len(),
            );
        }
    }

    // Inner UNIT_WxSTRIPE_H
    let len = unit_w - have_left_3;
    for j in 0..stripe_h {
        let p = p + have_left_3 + (j as isize * stride);
        BD::pixel_copy(
            &mut dst_tl[j * REST_UNIT_STRIDE + have_left_3..],
            &p.slice::<BD>(len),
            len,
        );
    }

    if !have_right {
        // Pad 3x(STRIPE_H+6) with last column
        for j in 0..stripe_h + 6 {
            let row_last = dst_l[(unit_w - 1) + j * REST_UNIT_STRIDE];
            let pad = &mut dst_l[unit_w + j * REST_UNIT_STRIDE..];
            BD::pixel_set(pad, row_last, 3);
        }
    }

    if !have_left {
        // Pad 3x(STRIPE_H+6) with first column
        for j in 0..stripe_h + 6 {
            let offset = j * REST_UNIT_STRIDE;
            // This would be `dst_l[offset]` in C,
            // but that results in multiple mutable borrows of `dst`,
            // so we recalculate `dst_l` here.
            // `3 * (have_left == 0) as c_int` simplifies to `3 * 1` and then `3`.
            let val = dst[3 + offset];
            BD::pixel_set(&mut dst[offset..], val, 3);
        }
    } else {
        let dst = &mut dst[3 * REST_UNIT_STRIDE..];
        for j in 0..stripe_h {
            BD::pixel_copy(&mut dst[j * REST_UNIT_STRIDE..], &left[j][1..], 3);
        }
    };
}

/// Calculates the offset between `lpf` and `ptr`.
///
/// This behaves like [`offset_from`], but allows for `ptr` to point to outside
/// the allocation of `lpf`. This is necessary because `ptr` may point to before
/// the beginning of `lpf`, which violates the safety conditions of
/// [`offset_from`].
///
/// [`offset_from`]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.offset_from
fn reconstruct_lpf_offset<BD: BitDepth>(
    lpf: &DisjointMut<AlignedVec64<u8>>,
    ptr: *const BD::Pixel,
) -> isize {
    let base = lpf.as_mut_ptr().cast::<BD::Pixel>();
    (ptr as isize - base as isize) / (mem::size_of::<BD::Pixel>() as isize)
}

/// # Safety
///
/// Must be called by [`loop_restoration_filter::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn wiener_c_erased<BD: BitDepth>(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
    p: FFISafeRav1dPictureDataComponentOffset,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `loop_restoration_filter::Fn::call`.
    let p = unsafe { FFISafe::from_with_offset(p) };
    let left = left.cast();
    // SAFETY: Was passed as `FFISafe::new(_)` in `loop_restoration_filter::Fn::call`.
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_ptr = lpf_ptr.cast();
    let lpf_off = reconstruct_lpf_offset::<BD>(lpf, lpf_ptr);
    let bd = BD::from_c(bitdepth_max);
    let w = w as usize;
    let h = h as usize;
    // SAFETY: Length sliced in `loop_restoration_filter::Fn::call`.
    let left = unsafe { slice::from_raw_parts(left, h) };
    wiener_rust(p, left, lpf, lpf_off, w, h, params, edges, bd)
}

// FIXME Could split into luma and chroma specific functions,
// (since first and last tops are always 0 for chroma)
// FIXME Could implement a version that requires less temporary memory
// (should be possible to implement with only 6 rows of temp storage)
fn wiener_rust<BD: BitDepth>(
    p: Rav1dPictureDataComponentOffset,
    left: &[LeftPixelRow<BD::Pixel>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bd: BD,
) {
    // Wiener filtering is applied to a maximum stripe height of 64 + 3 pixels
    // of padding above and below
    let mut tmp = [0.into(); (64 + 3 + 3) * REST_UNIT_STRIDE];

    padding::<BD>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);

    // Values stored between horizontal and vertical filtering don't
    // fit in a u8.
    let mut hor = [0; (64 + 3 + 3) * REST_UNIT_STRIDE];

    let filter = &params.filter;
    let bitdepth = bd.bitdepth().as_::<c_int>();
    let round_bits_h = 3 + (bitdepth == 12) as c_int * 2;
    let rounding_off_h = 1 << round_bits_h - 1;
    let clip_limit = 1 << bitdepth + 1 + 7 - round_bits_h;
    for (tmp, hor) in tmp
        .chunks_exact(REST_UNIT_STRIDE)
        .zip(hor.chunks_exact_mut(REST_UNIT_STRIDE))
        .take(h + 6)
    {
        for i in 0..w {
            let mut sum = 1 << bitdepth + 6;

            if BD::BPC == BPC::BPC8 {
                sum += tmp[i + 3].to::<i32>() * 128;
            }

            for (&tmp, &filter) in iter::zip(&tmp[i..i + 7], &filter[0][..7]) {
                sum += tmp.to::<i32>() * filter as c_int;
            }

            hor[i] = iclip(sum + rounding_off_h >> round_bits_h, 0, clip_limit - 1) as u16;
        }
    }

    let round_bits_v = 11 - (bitdepth == 12) as c_int * 2;
    let rounding_off_v = 1 << round_bits_v - 1;
    let round_offset = 1 << bitdepth + (round_bits_v - 1);
    for j in 0..h {
        for i in 0..w {
            let mut sum = -round_offset;
            let z = &hor[j * REST_UNIT_STRIDE + i..(j + 7) * REST_UNIT_STRIDE];

            for k in 0..7 {
                sum += z[k * REST_UNIT_STRIDE] as c_int * filter[1][k] as c_int;
            }

            let p = p + (j as isize * p.pixel_stride::<BD>()) + i;
            *p.index_mut::<BD>() =
                iclip(sum + rounding_off_v >> round_bits_v, 0, bd.into_c()).as_();
        }
    }
}

/// Sum over a 3x3 area
///
/// The `dst` and `src` pointers are positioned 3 pixels above and 3 pixels to the
/// left of the top left corner. However, the self guided filter only needs 1
/// pixel above and one pixel to the left. As for the pixels below and to the
/// right they must be computed in the sums, but don't need to be stored.
///
/// Example for a 4x4 block:
///
/// ```text
/// x x x x x x x x x x
/// x c c c c c c c c x
/// x i s s s s s s i x
/// x i s s s s s s i x
/// x i s s s s s s i x
/// x i s s s s s s i x
/// x i s s s s s s i x
/// x i s s s s s s i x
/// x c c c c c c c c x
/// x x x x x x x x x x
/// ```
///
/// * s: Pixel summed and stored
/// * i: Pixel summed and stored (between loops)
/// * c: Pixel summed not stored
/// * x: Pixel not summed not stored
fn boxsum3<BD: BitDepth>(
    sumsq: &mut [i32; (64 + 2 + 2) * REST_UNIT_STRIDE],
    sum: &mut [BD::Coef; (64 + 2 + 2) * REST_UNIT_STRIDE],
    src: &[BD::Pixel; (64 + 3 + 3) * REST_UNIT_STRIDE],
    w: usize,
    h: usize,
) {
    // We skip the first row, as it is never used
    let src = &src[REST_UNIT_STRIDE..];

    // We skip the first and last columns, as they are never used
    for x in 1..w - 1 {
        let mut sum_v = &mut sum[x..];
        let mut sumsq_v = &mut sumsq[x..];
        let mut s = &src[x..];
        let mut a: c_int = s[0].as_();
        let mut a2 = a * a;
        let mut b: c_int = s[REST_UNIT_STRIDE].as_();
        let mut b2 = b * b;

        // We skip the first 2 rows, as they are skipped in the next loop and
        // we don't need the last 2 row as it is skipped in the next loop
        for _ in 2..h - 2 {
            s = &s[REST_UNIT_STRIDE..];
            let c: c_int = s[REST_UNIT_STRIDE].as_();
            let c2 = c * c;
            sum_v = &mut sum_v[REST_UNIT_STRIDE..];
            sumsq_v = &mut sumsq_v[REST_UNIT_STRIDE..];
            sum_v[0] = (a + b + c).as_();
            sumsq_v[0] = a2 + b2 + c2;
            a = b;
            a2 = b2;
            b = c;
            b2 = c2;
        }
    }

    // We skip the first row as it is never read
    let mut sum = &mut sum[REST_UNIT_STRIDE..];
    let mut sumsq = &mut sumsq[REST_UNIT_STRIDE..];

    // We skip the last 2 rows as it is never read
    for _ in 2..h - 2 {
        let mut a = sum[1];
        let mut a2 = sumsq[1];
        let mut b = sum[2];
        let mut b2 = sumsq[2];

        // We don't store the first column as it is never read and
        // we don't store the last 2 columns as they are never read
        for x in 2..w - 2 {
            let c = sum[x + 1];
            let c2 = sumsq[x + 1];
            sum[x] = a + b + c;
            sumsq[x] = a2 + b2 + c2;
            a = b;
            a2 = b2;
            b = c;
            b2 = c2;
        }

        sum = &mut sum[REST_UNIT_STRIDE..];
        sumsq = &mut sumsq[REST_UNIT_STRIDE..];
    }
}

/// Sum over a 5x5 area
///
/// The `dst` and `src` pointers are positioned 3 pixels above and 3 pixels to the
/// left of the top left corner. However, the self guided filter only needs 1
/// pixel above and one pixel to the left. As for the pixels below and to the
/// right they must be computed in the sums, but don't need to be stored.
///
/// Example for a 4x4 block:
///
/// ```text
/// c c c c c c c c c c
/// c c c c c c c c c c
/// i i s s s s s s i i
/// i i s s s s s s i i
/// i i s s s s s s i i
/// i i s s s s s s i i
/// i i s s s s s s i i
/// i i s s s s s s i i
/// c c c c c c c c c c
/// c c c c c c c c c c
/// ```
///
/// * s: Pixel summed and stored
/// * i: Pixel summed and stored (between loops)
/// * c: Pixel summed not stored
/// * x: Pixel not summed not stored
fn boxsum5<BD: BitDepth>(
    sumsq: &mut [i32; (64 + 2 + 2) * REST_UNIT_STRIDE],
    sum: &mut [BD::Coef; (64 + 2 + 2) * REST_UNIT_STRIDE],
    src: &[BD::Pixel; (64 + 3 + 3) * REST_UNIT_STRIDE],
    w: usize,
    h: usize,
) {
    for x in 0..w {
        let mut sum_v = &mut sum[x..];
        let mut sumsq_v = &mut sumsq[x..];
        let s = &src[x..];
        let mut a: c_int = (s[0]).as_();
        let mut a2 = a * a;
        let mut b: c_int = (s[1 * REST_UNIT_STRIDE]).as_();
        let mut b2 = b * b;
        let mut c: c_int = (s[2 * REST_UNIT_STRIDE]).as_();
        let mut c2 = c * c;
        let mut d: c_int = (s[3 * REST_UNIT_STRIDE]).as_();
        let mut d2 = d * d;

        let mut s = &src[3 * REST_UNIT_STRIDE + x..];

        // We skip the first 2 rows, as they are skipped in the next loop and
        // we don't need the last 2 row as it is skipped in the next loop
        for _ in 2..h - 2 {
            s = &s[REST_UNIT_STRIDE..];
            let e: c_int = s[0].as_();
            let e2 = e * e;
            sum_v = &mut sum_v[REST_UNIT_STRIDE..];
            sumsq_v = &mut sumsq_v[REST_UNIT_STRIDE..];
            sum_v[0] = (a + b + c + d + e).as_();
            sumsq_v[0] = a2 + b2 + c2 + d2 + e2;
            a = b;
            b = c;
            c = d;
            d = e;
            a2 = b2;
            b2 = c2;
            c2 = d2;
            d2 = e2;
        }
    }

    // We skip the first row as it is never read
    let mut sum = &mut sum[REST_UNIT_STRIDE..];
    let mut sumsq = &mut sumsq[REST_UNIT_STRIDE..];
    for _ in 2..h - 2 {
        let mut a = sum[0];
        let mut a2 = sumsq[0];
        let mut b = sum[1];
        let mut b2 = sumsq[1];
        let mut c = sum[2];
        let mut c2 = sumsq[2];
        let mut d = sum[3];
        let mut d2 = sumsq[3];

        for x in 2..w - 2 {
            let e = sum[x + 2];
            let e2 = sumsq[x + 2];
            sum[x] = a + b + c + d + e;
            sumsq[x] = a2 + b2 + c2 + d2 + e2;
            a = b;
            b = c;
            c = d;
            d = e;
            a2 = b2;
            b2 = c2;
            c2 = d2;
            d2 = e2;
        }
        sum = &mut sum[REST_UNIT_STRIDE..];
        sumsq = &mut sumsq[REST_UNIT_STRIDE..];
    }
}

#[inline(never)]
fn selfguided_filter<BD: BitDepth>(
    dst: &mut [BD::Coef; 64 * 384],
    src: &[BD::Pixel; (64 + 3 + 3) * REST_UNIT_STRIDE],
    w: usize,
    h: usize,
    n: c_int,
    s: c_uint,
    bd: BD,
) {
    let sgr_one_by_x = if n == 25 { 164 } else { 455 };

    // Selfguided filter is applied to a maximum stripe height of 64 + 3 pixels
    // of padding above and below
    let mut sumsq = [0; (64 + 2 + 2) * REST_UNIT_STRIDE];
    // By inverting `a` and `b` after the boxsums, `b` can be of `BD::Coef` instead of `i32`.
    let mut sum = [0.as_::<BD::Coef>(); (64 + 2 + 2) * REST_UNIT_STRIDE];

    let step = (n == 25) as usize + 1;
    if n == 25 {
        boxsum5::<BD>(&mut sumsq, &mut sum, src, w + 6, h + 6);
    } else {
        boxsum3::<BD>(&mut sumsq, &mut sum, src, w + 6, h + 6);
    }
    let bitdepth_min_8 = bd.bitdepth() - 8;

    let mut a = CursorMut::new(&mut sumsq) + 2 * REST_UNIT_STRIDE + 3;
    let mut b = CursorMut::new(&mut sum) + 2 * REST_UNIT_STRIDE + 3;

    let mut aa = a.clone() - REST_UNIT_STRIDE;
    let mut bb = b.clone() - REST_UNIT_STRIDE;
    for _ in (-1..h as isize + 1).step_by(step) {
        for i in -1..w as isize + 1 {
            let a = aa[i] + (1 << 2 * bitdepth_min_8 >> 1) >> 2 * bitdepth_min_8;
            let b = bb[i].as_::<c_int>() + (1 << bitdepth_min_8 >> 1) >> bitdepth_min_8;

            let p = cmp::max(a * n - b * b, 0) as c_uint;
            let z = (p * s + (1 << 19)) >> 20;
            let x = dav1d_sgr_x_by_x[cmp::min(z, 255) as usize] as c_uint;

            // This is where we invert A and B, so that B is of size coef.
            aa[i] = ((x * bb[i].as_::<c_uint>() * sgr_one_by_x + (1 << 11)) >> 12) as c_int;
            bb[i] = x.as_::<BD::Coef>();
        }
        aa += step as usize * REST_UNIT_STRIDE;
        bb += step as usize * REST_UNIT_STRIDE;
    }

    fn six_neighbors<P>(p: &CursorMut<P>, i: isize) -> c_int
    where
        P: Add<Output = P> + ToPrimitive<c_int> + Copy,
    {
        let stride = REST_UNIT_STRIDE as isize;
        (p[i - stride] + p[i + stride]).as_::<c_int>() * 6
            + (p[i - 1 - stride] + p[i - 1 + stride] + p[i + 1 - stride] + p[i + 1 + stride])
                .as_::<c_int>()
                * 5
    }

    fn eight_neighbors<P>(p: &CursorMut<P>, i: isize) -> c_int
    where
        P: Add<Output = P> + ToPrimitive<c_int> + Copy,
    {
        let stride = REST_UNIT_STRIDE as isize;
        (p[i] + p[i - 1] + p[i + 1] + p[i - stride] + p[i + stride]).as_::<c_int>() * 4
            + (p[i - 1 - stride] + p[i - 1 + stride] + p[i + 1 - stride] + p[i + 1 + stride])
                .as_::<c_int>()
                * 3
    }

    const MAX_RESTORATION_WIDTH: usize = 256 * 3 / 2;

    let mut src = &src[3 * REST_UNIT_STRIDE + 3..];
    let mut dst = dst.as_mut_slice();
    if n == 25 {
        let mut j = 0;
        while j < h - 1 {
            for i in 0..w {
                let (a, b) = (six_neighbors(&b, i as isize), six_neighbors(&a, i as isize));
                dst[i] = ((b - a * src[i].as_::<c_int>() + (1 << 8)) >> 9).as_();
            }
            dst = &mut dst[MAX_RESTORATION_WIDTH..];
            src = &src[REST_UNIT_STRIDE..];
            b += REST_UNIT_STRIDE;
            a += REST_UNIT_STRIDE;
            for i in 0..w {
                let (a, b) = (
                    b[i].as_::<c_int>() * 6 + (b[i as isize - 1] + b[i + 1]).as_::<c_int>() * 5,
                    a[i] * 6 + (a[i as isize - 1] + a[i + 1]) * 5,
                );
                dst[i] = (b - a * src[i].as_::<c_int>() + (1 << 7) >> 8).as_();
            }
            dst = &mut dst[MAX_RESTORATION_WIDTH..];
            src = &src[REST_UNIT_STRIDE..];
            b += REST_UNIT_STRIDE;
            a += REST_UNIT_STRIDE;
            j += 2;
        }
        // Last row, when number of rows is odd
        if j + 1 == h {
            for i in 0..w {
                let (a, b) = (six_neighbors(&b, i as isize), six_neighbors(&a, i as isize));
                dst[i] = (b - a * src[i].as_::<c_int>() + (1 << 8) >> 9).as_();
            }
        }
    } else {
        for _ in 0..h {
            for i in 0..w {
                let (a, b) = (
                    eight_neighbors(&b, i as isize),
                    eight_neighbors(&a, i as isize),
                );
                dst[i] = (b - a * src[i].as_::<c_int>() + (1 << 8) >> 9).as_();
            }
            dst = &mut dst[384..];
            src = &src[REST_UNIT_STRIDE..];
            b += REST_UNIT_STRIDE;
            a += REST_UNIT_STRIDE;
        }
    };
}

/// # Safety
///
/// Must be called by [`loop_restoration_filter::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn sgr_5x5_c_erased<BD: BitDepth>(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
    p: FFISafeRav1dPictureDataComponentOffset,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `loop_restoration_filter::Fn::call`.
    let p = unsafe { FFISafe::from_with_offset(p) };
    let left = left.cast();
    // SAFETY: Was passed as `FFISafe::new(_)` in `loop_restoration_filter::Fn::call`.
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_ptr = lpf_ptr.cast();
    let lpf_off = reconstruct_lpf_offset::<BD>(lpf, lpf_ptr);
    let w = w as usize;
    let h = h as usize;
    let bd = BD::from_c(bitdepth_max);
    // SAFETY: Length sliced in `loop_restoration_filter::Fn::call`.
    let left = unsafe { slice::from_raw_parts(left, h) };
    sgr_5x5_rust(p, left, lpf, lpf_off, w, h, params, edges, bd)
}

fn sgr_5x5_rust<BD: BitDepth>(
    p: Rav1dPictureDataComponentOffset,
    left: &[LeftPixelRow<BD::Pixel>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bd: BD,
) {
    // Selfguided filter is applied to a maximum stripe height of 64 + 3 pixels
    // of padding above and below
    let mut tmp = [0.as_(); (64 + 3 + 3) * REST_UNIT_STRIDE];

    // Selfguided filter outputs to a maximum stripe height of 64 and a
    // maximum restoration width of 384 (256 * 1.5)
    let mut dst = [0.as_(); 64 * 384];

    padding::<BD>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);
    let sgr = params.sgr();
    selfguided_filter(&mut dst, &mut tmp, w, h, 25, sgr.s0, bd);

    let w0 = sgr.w0 as c_int;
    for j in 0..h {
        let p = p + (j as isize * p.pixel_stride::<BD>());
        let p = &mut *p.slice_mut::<BD>(w);
        for i in 0..w {
            let v = w0 * dst[j * 384 + i].as_::<c_int>();
            p[i] = bd.iclip_pixel(p[i].as_::<c_int>() + (v + (1 << 10) >> 11));
        }
    }
}

/// # Safety
///
/// Must be called by [`loop_restoration_filter::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn sgr_3x3_c_erased<BD: BitDepth>(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
    p: FFISafeRav1dPictureDataComponentOffset,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `loop_restoration_filter::Fn::call`.
    let p = unsafe { FFISafe::from_with_offset(p) };
    let left = left.cast();
    // SAFETY: Was passed as `FFISafe::new(_)` in `loop_restoration_filter::Fn::call`.
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_ptr = lpf_ptr.cast();
    let lpf_off = reconstruct_lpf_offset::<BD>(lpf, lpf_ptr);
    let w = w as usize;
    let h = h as usize;
    let bd = BD::from_c(bitdepth_max);
    // SAFETY: Length sliced in `loop_restoration_filter::Fn::call`.
    let left = unsafe { slice::from_raw_parts(left, h) };
    sgr_3x3_rust(p, left, lpf, lpf_off, w, h, params, edges, bd)
}

fn sgr_3x3_rust<BD: BitDepth>(
    p: Rav1dPictureDataComponentOffset,
    left: &[LeftPixelRow<BD::Pixel>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bd: BD,
) {
    let mut tmp = [0.as_(); (64 + 3 + 3) * REST_UNIT_STRIDE];
    let mut dst = [0.as_(); 64 * 384];

    padding::<BD>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);
    let sgr = params.sgr();
    selfguided_filter(&mut dst, &mut tmp, w, h, 9, sgr.s1, bd);

    let w1 = sgr.w1 as c_int;
    for j in 0..h {
        let p = p + (j as isize * p.pixel_stride::<BD>());
        let p = &mut *p.slice_mut::<BD>(w);
        for i in 0..w {
            let v = w1 * dst[j * 384 + i].as_::<c_int>();
            p[i] = bd.iclip_pixel(p[i].as_::<c_int>() + (v + (1 << 10) >> 11));
        }
    }
}

/// # Safety
///
/// Must be called by [`loop_restoration_filter::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn sgr_mix_c_erased<BD: BitDepth>(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
    p: FFISafeRav1dPictureDataComponentOffset,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `loop_restoration_filter::Fn::call`.
    let p = unsafe { FFISafe::from_with_offset(p) };
    let left = left.cast();
    // SAFETY: Was passed as `FFISafe::new(_)` in `loop_restoration_filter::Fn::call`.
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_ptr = lpf_ptr.cast();
    let lpf_off = reconstruct_lpf_offset::<BD>(lpf, lpf_ptr);
    let w = w as usize;
    let h = h as usize;
    let bd = BD::from_c(bitdepth_max);
    // SAFETY: Length sliced in `loop_restoration_filter::Fn::call`.
    let left = unsafe { slice::from_raw_parts(left, h) };
    sgr_mix_rust(p, left, lpf, lpf_off, w, h, params, edges, bd)
}

fn sgr_mix_rust<BD: BitDepth>(
    p: Rav1dPictureDataComponentOffset,
    left: &[LeftPixelRow<BD::Pixel>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bd: BD,
) {
    let mut tmp = [0.as_(); (64 + 3 + 3) * REST_UNIT_STRIDE];
    let mut dst0 = [0.as_(); 64 * 384];
    let mut dst1 = [0.as_(); 64 * 384];

    padding::<BD>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);
    let sgr = params.sgr();
    selfguided_filter(&mut dst0, &mut tmp, w, h, 25, sgr.s0, bd);
    selfguided_filter(&mut dst1, &mut tmp, w, h, 9, sgr.s1, bd);

    let w0 = sgr.w0 as c_int;
    let w1 = sgr.w1 as c_int;
    for j in 0..h {
        let p = p + (j as isize * p.pixel_stride::<BD>());
        let p = &mut *p.slice_mut::<BD>(w);
        for i in 0..w {
            let v = w0 * dst0[j * 384 + i].as_::<c_int>() + w1 * dst1[j * 384 + i].as_::<c_int>();
            p[i] = bd.iclip_pixel(p[i].as_::<c_int>() + (v + (1 << 10) >> 11));
        }
    }
}

/// The 8 bpc filters on [`I16x8`].
mod rust_simd {
    use super::*;
    use crate::simd128::{I16x8, I32x4};

    const ROWS: usize = 64 + 3 + 3;
    /// A unit is at most 256 wide, the last of a row up to half as wide again.
    const MAX_W: usize = 256 * 3 / 2;

    #[inline(always)]
    fn px8(row: &[u8], i: usize) -> I16x8 {
        I16x8::from_u8x8(&row[i..])
    }

    /// One row of 7 taps across, clipped to 13 bits, 8 pixels at a time with 32-bit sums. The
    /// 128 of the centre tap is folded back into its coefficient.
    #[inline(always)]
    fn wiener_h(out: &mut [i16; MAX_W + 8], t: &[u8], w: usize, k: [i16; 8]) {
        let zero = I16x8::splat(0);
        let max = I16x8::splat((1 << 13) - 1);
        for i in (0..w).step_by(8) {
            let v = [
                px8(t, i),
                px8(t, i + 1),
                px8(t, i + 2),
                px8(t, i + 3),
                px8(t, i + 4),
                px8(t, i + 5),
                px8(t, i + 6),
                zero,
            ];
            I16x8::madd8::<3>(v, k, (1 << 14) + 4)
                .max(zero)
                .min(max)
                .store(&mut out[i..]);
        }
    }

    /// Wiener over the padded `tmp`: rows across into a ring of the 7 the taps down read.
    #[inline(always)]
    fn wiener(
        mut dst: impl FnMut(usize, &[u8]),
        tmp: &[u8; ROWS * REST_UNIT_STRIDE],
        w: usize,
        h: usize,
        filter: &[[i16; 8]; 2],
    ) {
        let mut kh = filter[0];
        kh[3] += 128;
        kh[7] = 0;
        let mut kv = filter[1];
        kv[7] = 0;
        let zero = I16x8::splat(0);
        let mut ring = [[0i16; MAX_W + 8]; 8];
        let t = |r: usize| &tmp[r * REST_UNIT_STRIDE..][..REST_UNIT_STRIDE];
        for r in 0..6 {
            wiener_h(&mut ring[r], t(r), w, kh);
        }
        for j in 0..h {
            wiener_h(&mut ring[(j + 6) % 8], t(j + 6), w, kh);
            let mut row = [0u8; MAX_W + 8];
            for i in (0..w).step_by(8) {
                let r = |k: usize| &ring[(j + k) % 8][i..];
                let v = [
                    I16x8::load(r(0)),
                    I16x8::load(r(1)),
                    I16x8::load(r(2)),
                    I16x8::load(r(3)),
                    I16x8::load(r(4)),
                    I16x8::load(r(5)),
                    I16x8::load(r(6)),
                    zero,
                ];
                let p = I16x8::madd8::<11>(v, kv, (1 << 10) - (1 << 18)).pack_u8();
                row[i..i + 8].copy_from_slice(&p);
            }
            dst(j, &row[..w]);
        }
    }

    #[cfg_attr(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature(enable = "sse4.1")
    )]
    fn wiener_128<BD: BitDepth>(
        p: Rav1dPictureDataComponentOffset,
        left: &[LeftPixelRow<BD::Pixel>],
        lpf: &DisjointMut<AlignedVec64<u8>>,
        lpf_off: isize,
        w: usize,
        h: usize,
        params: &LooprestorationParams,
        edges: LrEdgeFlags,
    ) {
        let mut tmp = [0.into(); ROWS * REST_UNIT_STRIDE];
        padding::<BD>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);
        let tmp: &[u8; ROWS * REST_UNIT_STRIDE] = tmp.as_bytes().try_into().unwrap();
        let stride = p.pixel_stride::<BD>();
        let store = |j: usize, row: &[u8]| {
            let p = p + j as isize * stride;
            p.slice_mut::<BD>(row.len()).as_bytes_mut().copy_from_slice(row);
        };
        wiener(store, tmp, w, h, &params.filter);
    }

    /// # Safety
    ///
    /// Must be called by [`loop_restoration_filter::Fn::call`] at 8 bpc, on x86 once the CPU
    /// flags hold SSE41.
    #[deny(unsafe_op_in_unsafe_fn)]
    #[allow(unused_unsafe)]
    pub unsafe extern "C" fn wiener_erased<BD: BitDepth>(
        _p_ptr: *mut DynPixel,
        _stride: ptrdiff_t,
        left: *const LeftPixelRow<DynPixel>,
        lpf_ptr: *const DynPixel,
        w: c_int,
        h: c_int,
        params: &LooprestorationParams,
        edges: LrEdgeFlags,
        _bitdepth_max: c_int,
        p: FFISafeRav1dPictureDataComponentOffset,
        lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
    ) {
        // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `loop_restoration_filter::Fn::call`.
        let p = unsafe { FFISafe::from_with_offset(p) };
        let left = left.cast();
        // SAFETY: Was passed as `FFISafe::new(_)` in `loop_restoration_filter::Fn::call`.
        let lpf = unsafe { FFISafe::get(lpf) };
        let lpf_off = reconstruct_lpf_offset::<BD>(lpf, lpf_ptr.cast());
        let (w, h) = (w as usize, h as usize);
        // SAFETY: Length sliced in `loop_restoration_filter::Fn::call`.
        let left = unsafe { slice::from_raw_parts(left, h) };
        // SAFETY: SSE4.1 is checked before this `fn` is chosen on x86; elsewhere it is safe.
        unsafe { wiener_128::<BD>(p, left, lpf, lpf_off, w, h, params, edges) }
    }

    /// Columns of the self-guided rows: the unit's `w` and one either side, in whole vectors.
    const SGR_W: usize = MAX_W + 16;

    /// One row of 16-bit sums (or `b`) and their 32-bit companions: sums of squares (or `a`).
    #[derive(Clone, Copy)]
    struct SgrRow {
        s: [i16; SGR_W],
        q: [i32; SGR_W],
    }

    const SGR_ROW: SgrRow = SgrRow {
        s: [0; SGR_W],
        q: [0; SGR_W],
    };

    #[inline(always)]
    fn ld32(q: &[i32; SGR_W], i: usize) -> [I32x4; 2] {
        [I32x4::load(&q[i..]), I32x4::load(&q[i + 4..])]
    }

    #[inline(always)]
    fn ld16(s: &[i16; SGR_W], i: usize) -> I16x8 {
        I16x8::load(&s[i..])
    }

    #[inline(always)]
    fn add32(a: [I32x4; 2], b: [I32x4; 2]) -> [I32x4; 2] {
        [a[0].add(b[0]), a[1].add(b[1])]
    }

    /// The 3- and 5-wide sums across one padded row, of pixels and of their squares, centred on
    /// columns -1..=w at index 0.
    #[inline(always)]
    fn sgr_box_h<const THREE: bool, const FIVE: bool>(
        h3: &mut SgrRow,
        h5: &mut SgrRow,
        t: &[u8],
        w: usize,
    ) {
        for i in (0..w + 2).step_by(8) {
            let (v1, v2, v3) = (px8(t, i + 1), px8(t, i + 2), px8(t, i + 3));
            let s = v1.add(v2).add(v3);
            // Squares of bytes fit 16 bits unsigned.
            let q = v3.mul(v3).addw_u(v1.mul(v1).addl_u(v2.mul(v2)));
            if THREE {
                s.store(&mut h3.s[i..]);
                q[0].store(&mut h3.q[i..]);
                q[1].store(&mut h3.q[i + 4..]);
            }
            if FIVE {
                let (v0, v4) = (px8(t, i), px8(t, i + 4));
                let s = s.add(v0).add(v4);
                let q = v4.mul(v4).addw_u(v0.mul(v0).addw_u(q));
                s.store(&mut h5.s[i..]);
                q[0].store(&mut h5.q[i..]);
                q[1].store(&mut h5.q[i + 4..]);
            }
        }
    }

    /// `dav1d_sgr_x_by_x[z]` for `z` in 0..=255 (or more): a 64-entry lookup, then the steps of
    /// the table's flat tail.
    #[inline(always)]
    fn x_by_x(z: I16x8) -> I16x8 {
        let lut: &[u8; 64] = dav1d_sgr_x_by_x.0[..64].try_into().unwrap();
        let mut x = z
            .lookup64(lut)
            .add(z.gt(I16x8::splat(63)).and(I16x8::splat(4)));
        for k in [72, 101, 169, 254] {
            x = x.add(z.gt(I16x8::splat(k)));
        }
        x
    }

    /// `a` and `b` of one row from the box sums of the rows `h` (3 or 5 of them), with `b` as
    /// `x` and `a` as `x * sum * one_by_x` the way [`selfguided_filter`] inverts them.
    #[inline(always)]
    fn sgr_ab<const N: i32>(ab: &mut SgrRow, h: &[&SgrRow], w: usize, s: u32) {
        let one_by_x = I32x4::splat(if N == 25 { 164 } else { 455 });
        let n = I32x4::splat(N);
        let s = I32x4::splat(s as i32);
        let zero = I32x4::splat(0);
        let (rnd19, rnd11) = (I32x4::splat(1 << 19), I32x4::splat(1 << 11));
        for i in (0..w + 2).step_by(8) {
            let mut sum = I16x8::load(&h[0].s[i..]);
            let mut sq = ld32(&h[0].q, i);
            for r in &h[1..] {
                sum = sum.add(I16x8::load(&r.s[i..]));
                sq = add32(sq, ld32(&r.q, i));
            }
            let sum2 = sum.mul_wide(sum);
            let mut z = [zero; 2];
            for k in 0..2 {
                let p = sq[k].mul(n).sub(sum2[k]).max(zero);
                // `p * s` stays below 2^32: the product is exact as unsigned.
                z[k] = p.mul(s).add(rnd19).shr_u::<20>();
            }
            let x = x_by_x(I32x4::pack(z[0], z[1]));
            let xs = x.mul_wide(sum);
            for k in 0..2 {
                let a = xs[k].mul(one_by_x).add(rnd11).shr_s::<12>();
                a.store(&mut ab.q[i + 4 * k..]);
            }
            x.store(&mut ab.s[i..]);
        }
    }

    /// The rows of 3x3 `a` and `b` across: `h` sums 3, `g` is `3 * h` plus the centre. The
    /// neighbours of 3x3, 4 for the cross and 3 for the corners, are `g` above and below plus
    /// `4 * h` of the row itself.
    #[inline(always)]
    fn sgr_h3(g: &mut SgrRow, h: &mut SgrRow, ab: &SgrRow, w: usize) {
        for x in (0..w).step_by(8) {
            let c = ld16(&ab.s, x + 1);
            let all = ld16(&ab.s, x).add(ld16(&ab.s, x + 2)).add(c);
            all.store(&mut h.s[x..]);
            all.mul(I16x8::splat(3)).add(c).store(&mut g.s[x..]);
            let c = ld32(&ab.q, x + 1);
            let all = add32(add32(ld32(&ab.q, x), ld32(&ab.q, x + 2)), c);
            for k in 0..2 {
                all[k].store(&mut h.q[x + 4 * k..]);
                let v = all[k].shl::<1>().add(all[k]).add(c[k]);
                v.store(&mut g.q[x + 4 * k..]);
            }
        }
    }

    /// The rows of 5x5 `a` and `b` across, weighted 6 for the centre and 5 either side: the
    /// even output rows add two of them, the odd ones take one.
    #[inline(always)]
    fn sgr_h65(f: &mut SgrRow, ab: &SgrRow, w: usize) {
        for x in (0..w).step_by(8) {
            let c = ld16(&ab.s, x + 1);
            let all = ld16(&ab.s, x).add(ld16(&ab.s, x + 2)).add(c);
            all.mul(I16x8::splat(5)).add(c).store(&mut f.s[x..]);
            let c = ld32(&ab.q, x + 1);
            let all = add32(add32(ld32(&ab.q, x), ld32(&ab.q, x + 2)), c);
            for k in 0..2 {
                let v = all[k].shl::<2>().add(all[k]).add(c[k]);
                v.store(&mut f.q[x + 4 * k..]);
            }
        }
    }

    /// `b - a * px` from the weighted neighbours `nb` of `b` and `na` of `a`, rounded off by
    /// `S` bits.
    #[inline(always)]
    fn sgr_f<const S: i32>(nb: I16x8, na: [I32x4; 2], px: I16x8) -> [I32x4; 2] {
        let bx = nb.mul_wide(px);
        let rnd = I32x4::splat(1 << (S - 1));
        [
            na[0].sub(bx[0]).add(rnd).shr_s::<S>(),
            na[1].sub(bx[1]).add(rnd).shr_s::<S>(),
        ]
    }

    /// Box sums across the padded rows `*next..=to`, into their slots of the ring.
    #[inline(always)]
    fn sgr_fill<const THREE: bool, const FIVE: bool>(
        h3: &mut [SgrRow; 5],
        h5: &mut [SgrRow; 5],
        tmp: &[u8],
        w: usize,
        next: &mut isize,
        to: isize,
    ) {
        while *next <= to {
            let y = (*next + 3) as usize;
            sgr_box_h::<THREE, FIVE>(
                &mut h3[y % 5],
                &mut h5[y % 5],
                &tmp[y * REST_UNIT_STRIDE..],
                w,
            );
            *next += 1;
        }
    }

    /// The self-guided filter over the padded `tmp`, streaming: the box sums of each row
    /// across into a ring, `a` and `b` of the rows the output needs (every other one for 5x5)
    /// summed across into the weighted rows of another, then each output row from those
    /// weighted onto its pixels.
    #[inline(always)]
    fn sgr<const THREE: bool, const FIVE: bool>(
        mut dst: impl FnMut(usize, &[u8]),
        tmp: &[u8],
        w: usize,
        h: usize,
        sgr: &LooprestorationParamsSgr,
    ) {
        let mut h3 = [SGR_ROW; 5];
        let mut h5 = [SGR_ROW; 5];
        let mut ab3 = SGR_ROW;
        let mut g3 = [SGR_ROW; 3];
        let mut h3m = [SGR_ROW; 3];
        let mut ab5 = SGR_ROW;
        let mut f5 = [SGR_ROW; 2];
        // Box sums are ready for the rows before `next`.
        let mut next = if FIVE { -3 } else { -2 };
        let slot = |y: isize| (y + 3) as usize % 5;
        let ab3_at = |y: isize| (y + 1) as usize % 3;
        let ab5_at = |y: isize| (y + 1) as usize / 2 % 2;
        if THREE {
            sgr_fill::<THREE, FIVE>(&mut h3, &mut h5, tmp, w, &mut next, 1);
            for y in -1..=0 {
                let rows = [&h3[slot(y - 1)], &h3[slot(y)], &h3[slot(y + 1)]];
                sgr_ab::<9>(&mut ab3, &rows, w, sgr.s1);
                sgr_h3(&mut g3[ab3_at(y)], &mut h3m[ab3_at(y)], &ab3, w);
            }
        }
        if FIVE {
            sgr_fill::<THREE, FIVE>(&mut h3, &mut h5, tmp, w, &mut next, 1);
            let rows = [
                &h5[slot(-3)],
                &h5[slot(-2)],
                &h5[slot(-1)],
                &h5[slot(0)],
                &h5[slot(1)],
            ];
            sgr_ab::<25>(&mut ab5, &rows, w, sgr.s0);
            sgr_h65(&mut f5[ab5_at(-1)], &ab5, w);
        }
        let (w0, w1) = (I32x4::splat(sgr.w0 as i32), I32x4::splat(sgr.w1 as i32));
        let rnd = I32x4::splat(1 << 10);
        let mut row = [0u8; MAX_W + 8];
        for j in 0..h {
            let y = j as isize;
            if THREE {
                sgr_fill::<THREE, FIVE>(&mut h3, &mut h5, tmp, w, &mut next, y + 2);
                let rows = [&h3[slot(y)], &h3[slot(y + 1)], &h3[slot(y + 2)]];
                sgr_ab::<9>(&mut ab3, &rows, w, sgr.s1);
                sgr_h3(&mut g3[ab3_at(y + 1)], &mut h3m[ab3_at(y + 1)], &ab3, w);
            }
            if FIVE && j % 2 == 0 {
                sgr_fill::<THREE, FIVE>(&mut h3, &mut h5, tmp, w, &mut next, y + 3);
                let rows = [
                    &h5[slot(y - 1)],
                    &h5[slot(y)],
                    &h5[slot(y + 1)],
                    &h5[slot(y + 2)],
                    &h5[slot(y + 3)],
                ];
                sgr_ab::<25>(&mut ab5, &rows, w, sgr.s0);
                sgr_h65(&mut f5[ab5_at(y + 1)], &ab5, w);
            }
            let src = &tmp[(y + 3) as usize * REST_UNIT_STRIDE..];
            for x in (0..w).step_by(8) {
                let px = px8(src, x + 3);
                let mut v = [I32x4::splat(0); 2];
                if FIVE {
                    let d = if j % 2 == 0 {
                        let (u, d) = (&f5[ab5_at(y - 1)], &f5[ab5_at(y + 1)]);
                        let nb = ld16(&u.s, x).add(ld16(&d.s, x));
                        sgr_f::<9>(nb, add32(ld32(&u.q, x), ld32(&d.q, x)), px)
                    } else {
                        let m = &f5[ab5_at(y)];
                        sgr_f::<8>(ld16(&m.s, x), ld32(&m.q, x), px)
                    };
                    v = [d[0].mul(w0), d[1].mul(w0)];
                }
                if THREE {
                    let (u, m, d) = (&g3[ab3_at(y - 1)], &h3m[ab3_at(y)], &g3[ab3_at(y + 1)]);
                    let nb = ld16(&u.s, x)
                        .add(ld16(&d.s, x))
                        .add(ld16(&m.s, x).mul(I16x8::splat(4)));
                    let (gu, hm, gd) = (ld32(&u.q, x), ld32(&m.q, x), ld32(&d.q, x));
                    let na = add32(add32(gu, gd), [hm[0].shl::<2>(), hm[1].shl::<2>()]);
                    let d = sgr_f::<9>(nb, na, px);
                    v = add32(v, [d[0].mul(w1), d[1].mul(w1)]);
                }
                let v = I32x4::pack(v[0].add(rnd).shr_s::<11>(), v[1].add(rnd).shr_s::<11>());
                row[x..x + 8].copy_from_slice(&px.add(v).pack_u8());
            }
            dst(j, &row[..w]);
        }
    }

    #[cfg_attr(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature(enable = "sse4.1")
    )]
    fn sgr_128<BD: BitDepth, const THREE: bool, const FIVE: bool>(
        p: Rav1dPictureDataComponentOffset,
        left: &[LeftPixelRow<BD::Pixel>],
        lpf: &DisjointMut<AlignedVec64<u8>>,
        lpf_off: isize,
        w: usize,
        h: usize,
        params: &LooprestorationParams,
        edges: LrEdgeFlags,
    ) {
        // A row more than `padding` fills, for the loads past the last one's right edge.
        let mut tmp = [0.into(); (ROWS + 1) * REST_UNIT_STRIDE];
        let padded = (&mut tmp[..ROWS * REST_UNIT_STRIDE]).try_into().unwrap();
        padding::<BD>(padded, p, left, lpf, lpf_off, w, h, edges);
        let stride = p.pixel_stride::<BD>();
        let store = |j: usize, row: &[u8]| {
            let p = p + j as isize * stride;
            p.slice_mut::<BD>(row.len())
                .as_bytes_mut()
                .copy_from_slice(row);
        };
        sgr::<THREE, FIVE>(store, tmp.as_bytes(), w, h, params.sgr());
    }

    /// # Safety
    ///
    /// Must be called by [`loop_restoration_filter::Fn::call`] at 8 bpc, on x86 once the CPU
    /// flags hold SSE41.
    #[deny(unsafe_op_in_unsafe_fn)]
    #[allow(unused_unsafe)]
    pub unsafe extern "C" fn sgr_erased<BD: BitDepth, const THREE: bool, const FIVE: bool>(
        _p_ptr: *mut DynPixel,
        _stride: ptrdiff_t,
        left: *const LeftPixelRow<DynPixel>,
        lpf_ptr: *const DynPixel,
        w: c_int,
        h: c_int,
        params: &LooprestorationParams,
        edges: LrEdgeFlags,
        _bitdepth_max: c_int,
        p: FFISafeRav1dPictureDataComponentOffset,
        lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
    ) {
        // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `loop_restoration_filter::Fn::call`.
        let p = unsafe { FFISafe::from_with_offset(p) };
        let left = left.cast();
        // SAFETY: Was passed as `FFISafe::new(_)` in `loop_restoration_filter::Fn::call`.
        let lpf = unsafe { FFISafe::get(lpf) };
        let lpf_off = reconstruct_lpf_offset::<BD>(lpf, lpf_ptr.cast());
        let (w, h) = (w as usize, h as usize);
        // SAFETY: Length sliced in `loop_restoration_filter::Fn::call`.
        let left = unsafe { slice::from_raw_parts(left, h) };
        // SAFETY: SSE4.1 is checked before this `fn` is chosen on x86; elsewhere it is safe.
        unsafe { sgr_128::<BD, THREE, FIVE>(p, left, lpf, lpf_off, w, h, params, edges) }
    }
}

impl Rav1dLoopRestorationDSPContext {
    pub const fn default<BD: BitDepth>() -> Self {
        Self {
            wiener: [loop_restoration_filter::Fn::new(wiener_c_erased::<BD>); 2],
            sgr: [
                loop_restoration_filter::Fn::new(sgr_5x5_c_erased::<BD>),
                loop_restoration_filter::Fn::new(sgr_3x3_c_erased::<BD>),
                loop_restoration_filter::Fn::new(sgr_mix_c_erased::<BD>),
            ],
        }
    }

    /// The Rust SIMD filters at 8 bpc.
    #[inline(always)]
    const fn init<BD: BitDepth>(mut self, flags: CpuFlags) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if !flags.contains(CpuFlags::SSE41) {
            return self;
        }
        let _ = flags;
        if !crate::simd128::native() || BD::BITDEPTH != 8 {
            return self;
        }
        let wiener = loop_restoration_filter::Fn::new(rust_simd::wiener_erased::<BD>);
        self.wiener = [wiener; 2];
        self.sgr = [
            loop_restoration_filter::Fn::new(rust_simd::sgr_erased::<BD, false, true>),
            loop_restoration_filter::Fn::new(rust_simd::sgr_erased::<BD, true, false>),
            loop_restoration_filter::Fn::new(rust_simd::sgr_erased::<BD, true, true>),
        ];
        self
    }

    pub const fn new<BD: BitDepth>(flags: CpuFlags) -> Self {
        Self::default::<BD>().init::<BD>(flags)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::AlignedVec64;
    use crate::cpu::{rav1d_get_cpu_flags, rav1d_init_cpu};
    use crate::include::common::bitdepth::BitDepth8;
    use crate::include::dav1d::picture::Rav1dPictureDataComponent;
    use crate::with_offset::WithOffset;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 32) as u32
        }

        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % (hi - lo + 1) as u32) as i32
        }
    }

    const STRIDE: usize = 512;
    const ROWS: usize = 72;
    /// The unit's top-left, past the 3 columns the filters read to its left.
    const AT: usize = 4 * STRIDE + 64;

    fn rust() -> Rav1dLoopRestorationDSPContext {
        rav1d_init_cpu();
        let flags = rav1d_get_cpu_flags();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        assert!(flags.contains(CpuFlags::SSE41), "the SIMD filters need SSE4.1");
        Rav1dLoopRestorationDSPContext::default::<BitDepth8>().init::<BitDepth8>(flags)
    }

    /// Pixels near a base with some outliers, the way decoded planes look.
    fn plane(rng: &mut Rng, n: usize) -> Vec<u8> {
        let base = rng.range(0, 255);
        (0..n)
            .map(|_| match rng.next() % 8 {
                0 => rng.next() as u8,
                _ => (base + rng.range(-12, 12)).clamp(0, 255) as u8,
            })
            .collect()
    }

    /// Coefficients in the ranges the bitstream codes.
    fn wiener_params(rng: &mut Rng, five: bool) -> LooprestorationParams {
        let mut params = LooprestorationParams::default();
        for (f, centre) in [(0, 0), (1, 128)] {
            let k = [
                if five { 0 } else { rng.range(-5, 10) },
                rng.range(-23, 8),
                rng.range(-17, 46),
            ]
            .map(|k| k as i16);
            let filter = &mut params.filter[f];
            filter[..3].copy_from_slice(&k);
            filter[4] = k[2];
            filter[5] = k[1];
            filter[6] = k[0];
            filter[3] = centre - (k[0] + k[1] + k[2]) * 2;
        }
        params
    }

    /// `sgr[mode]`'s parameters: a strength pair from the table that selects it, and weights
    /// in the ranges the bitstream codes.
    fn sgr_params(rng: &mut Rng, mode: usize) -> LooprestorationParams {
        use crate::tables::DAV1D_SGR_PARAMS;

        let idx = match mode {
            0 => rng.range(14, 15),
            1 => rng.range(10, 13),
            _ => rng.range(0, 9),
        };
        let [s0, s1] = DAV1D_SGR_PARAMS.0[idx as usize];
        let w0 = if s0 != 0 { rng.range(-96, 31) } else { 0 };
        let w1 = if s1 != 0 { rng.range(-32, 95) } else { 95 };
        let mut params = LooprestorationParams::default();
        *params.sgr_mut() = LooprestorationParamsSgr {
            s0: s0 as u32,
            s1: s1 as u32,
            w0: w0 as i16,
            w1: (128 - w0 - w1) as i16,
        };
        params
    }

    struct Unit {
        pic: Vec<u8>,
        lpf: Vec<u8>,
        left: Vec<LeftPixelRow<u8>>,
    }

    impl Unit {
        fn new(rng: &mut Rng) -> Self {
            let pic = plane(rng, STRIDE * ROWS);
            let lpf = plane(rng, STRIDE * 12);
            let left = plane(rng, 4 * 64).chunks_exact(4).map(|c| c.try_into().unwrap()).collect();
            Self { pic, lpf, left }
        }

        /// The plane after `f` filters a `w`x`h` unit at [`AT`].
        fn run(
            &self,
            f: &loop_restoration_filter::Fn,
            w: usize,
            h: usize,
            params: &LooprestorationParams,
            edges: LrEdgeFlags,
        ) -> Vec<u8> {
            let mut buf = AlignedVec64::<u8>::new();
            buf.resize(self.pic.len(), 0);
            buf.as_mut_slice().copy_from_slice(&self.pic);
            let mut lpf = AlignedVec64::<u8>::new();
            lpf.resize(self.lpf.len(), 0);
            lpf.as_mut_slice().copy_from_slice(&self.lpf);
            let lpf = DisjointMut::new(lpf);
            let comp = Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(buf.as_mut_slice(), STRIDE);
            let dst = WithOffset {
                data: &comp,
                offset: AT,
            };
            let bd = BitDepth8::from_c(255);
            f.call::<BitDepth8>(dst, &self.left, &lpf, 64, w as c_int, h as c_int, params, edges, bd);
            drop(comp);
            buf.as_slice().to_vec()
        }
    }

    #[test]
    fn wiener_matches_scalar() {
        let scalar = Rav1dLoopRestorationDSPContext::default::<BitDepth8>();
        let rust = rust();
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for round in 0..300 {
            let unit = Unit::new(&mut rng);
            let five = round % 2 == 1;
            let params = wiener_params(&mut rng, five);
            let w = match round % 5 {
                0 => 384,
                1 => 64,
                _ => rng.range(1, 384) as usize,
            };
            let h = match round % 3 {
                0 => 64,
                _ => rng.range(1, 64) as usize,
            };
            let edges = LrEdgeFlags::from_bits_truncate(rng.next() as u8);
            let f = five as usize;
            let want = unit.run(&scalar.wiener[f], w, h, &params, edges);
            let got = unit.run(&rust.wiener[f], w, h, &params, edges);
            assert_eq!(got, want, "{w}x{h} edges {:#x} round {round}", edges.bits());
        }
    }

    #[test]
    fn sgr_matches_scalar() {
        let scalar = Rav1dLoopRestorationDSPContext::default::<BitDepth8>();
        let rust = rust();
        let mut rng = Rng(0x6a09_e667_f3bc_c908);
        for w in 1..=384 {
            for mode in 0..3 {
                let mut unit = Unit::new(&mut rng);
                // Flat, full-range noise and 0/255 stripes: `z` at 0, mid-table and past 255.
                match rng.next() % 6 {
                    0 => unit.pic.fill(rng.next() as u8),
                    1 => unit.pic.iter_mut().for_each(|p| *p = rng.next() as u8),
                    2 => unit
                        .pic
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, p)| *p = (i & 1) as u8 * 255),
                    _ => {}
                }
                let params = sgr_params(&mut rng, mode);
                let h = match rng.next() % 3 {
                    0 => 64,
                    _ => rng.range(1, 64) as usize,
                };
                let edges = LrEdgeFlags::from_bits_truncate(rng.next() as u8);
                let want = unit.run(&scalar.sgr[mode], w, h, &params, edges);
                let got = unit.run(&rust.sgr[mode], w, h, &params, edges);
                assert_eq!(got, want, "mode {mode} {w}x{h} edges {:#x}", edges.bits());
            }
        }
    }

    /// Best of 20 batches, in ns per call.
    fn ns(n: usize, mut f: impl FnMut()) -> f64 {
        use std::time::Instant;

        let mut best = f64::MAX;
        for _ in 0..20 {
            let start = Instant::now();
            for _ in 0..n {
                f();
            }
            best = best.min(start.elapsed().as_nanos() as f64 / n as f64);
        }
        best
    }

    /// ns per call of a 64-row stripe, scalar and Rust.
    #[test]
    #[ignore]
    fn time_lr() {
        let dsps = [
            ("scalar", Rav1dLoopRestorationDSPContext::default::<BitDepth8>()),
            ("rust", rust()),
        ];
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let unit = Unit::new(&mut rng);
        let mut buf = AlignedVec64::<u8>::new();
        buf.resize(unit.pic.len(), 0);
        buf.as_mut_slice().copy_from_slice(&unit.pic);
        let mut lpf = AlignedVec64::<u8>::new();
        lpf.resize(unit.lpf.len(), 0);
        lpf.as_mut_slice().copy_from_slice(&unit.lpf);
        let lpf = DisjointMut::new(lpf);
        let comp = Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(buf.as_mut_slice(), STRIDE);
        let dst = WithOffset {
            data: &comp,
            offset: AT,
        };
        let bd = BitDepth8::from_c(255);
        let edges = LrEdgeFlags::all();
        for (name, five) in [("wiener7", false), ("wiener5", true)] {
            let params = wiener_params(&mut rng, five);
            for w in [64, 128, 256, 384] {
                let mut line = format!("{name} {w}x64:");
                for (dsp_name, dsp) in &dsps {
                    let f = &dsp.wiener[five as usize];
                    let best = ns(20, || {
                        f.call::<BitDepth8>(dst, &unit.left, &lpf, 64, w, 64, &params, edges, bd);
                    });
                    line += &format!("  {dsp_name} {best:7.0} ns");
                }
                println!("{line}");
            }
        }
        for w in [64, 384] {
            let mut tmp = [0u8; (64 + 3 + 3) * REST_UNIT_STRIDE];
            let best = ns(20, || {
                padding::<BitDepth8>(&mut tmp, dst, &unit.left, &lpf, 64, w, 64, edges);
            });
            println!("padding {w}x64:  {best:7.0} ns");
        }
        for (name, mode) in [("sgr5", 0), ("sgr3", 1), ("sgrmix", 2)] {
            let params = sgr_params(&mut rng, mode);
            for w in [64, 128, 256, 384] {
                let mut line = format!("{name} {w}x64:");
                for (dsp_name, dsp) in &dsps {
                    let f = &dsp.sgr[mode];
                    let best = ns(20, || {
                        f.call::<BitDepth8>(dst, &unit.left, &lpf, 64, w, 64, &params, edges, bd);
                    });
                    line += &format!("  {dsp_name} {best:7.0} ns");
                }
                println!("{line}");
            }
        }
    }
}
