#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::{c_int, c_uint};
use std::{cmp, slice};

use libc::ptrdiff_t;
use strum::FromRepr;
use zerocopy::{AsBytes, FromBytes};

use crate::cpu::CpuFlags;
use crate::enum_map::{enum_map, enum_map_ty, DefaultValue};
use crate::ffi_safe::FFISafe;
#[cfg(all(
    feature = "asm",
    not(any(target_arch = "riscv64", target_arch = "riscv32"))
))]
use crate::include::common::bitdepth::bd_fn;
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
use crate::include::common::bitdepth::bpc_fn;
use crate::include::common::bitdepth::{AsPrimitive, BitDepth, DynPixel, BPC};
use crate::include::common::intops::{apply_sign, iclip};
use crate::include::dav1d::headers::Rav1dPixelLayoutSubSampled;
use crate::include::dav1d::picture::{
    FFISafeRav1dPictureDataComponentOffset, Rav1dPictureDataComponentOffset,
};
use crate::internal::{SCRATCH_AC_TXTP_LEN, SCRATCH_EDGE_LEN};
use crate::levels::{
    DC_128_PRED, DC_PRED, FILTER_PRED, HOR_PRED, LEFT_DC_PRED, N_IMPL_INTRA_PRED_MODES, PAETH_PRED,
    SMOOTH_H_PRED, SMOOTH_PRED, SMOOTH_V_PRED, TOP_DC_PRED, VERT_PRED, Z1_PRED, Z2_PRED, Z3_PRED,
};
use crate::strided::Strided as _;
use crate::tables::{
    dav1d_dr_intra_derivative, dav1d_filter_intra_taps, dav1d_sm_weights, filter_fn, FLT_INCR,
};
use crate::wrap_fn_ptr::wrap_fn_ptr;

wrap_fn_ptr!(pub unsafe extern "C" fn angular_ipred(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    angle: c_int,
    max_width: c_int,
    max_height: c_int,
    bitdepth_max: c_int,
    _topleft_off: usize,
    _dst: FFISafeRav1dPictureDataComponentOffset,
) -> ());

impl angular_ipred::Fn {
    pub fn call<BD: BitDepth>(
        &self,
        dst: Rav1dPictureDataComponentOffset,
        topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
        topleft_off: usize,
        width: c_int,
        height: c_int,
        angle: c_int,
        max_width: c_int,
        max_height: c_int,
        bd: BD,
    ) {
        let dst_ptr = dst.as_mut_ptr::<BD>().cast();
        let stride = dst.stride();
        let topleft = topleft[topleft_off..].as_ptr().cast();
        let bd = bd.into_c();
        let dst = dst.into_ffi_safe();
        // SAFETY: Fallbacks are safe; asm is supposed to do the same, where the fallbacks are:
        // * `fn splat_dc`
        // * `fn ipred_{v,h}_rust`
        // * `fn ipred_paeth_rust`
        // * `fn ipred_smooth_rust`
        // * `fn ipred_smooth_{v,h}_rust`
        // * `fn ipred_z{1,2,3}_rust`
        // * `fn ipred_filter_rust`
        unsafe {
            self.get()(
                dst_ptr,
                stride,
                topleft,
                width,
                height,
                angle,
                max_width,
                max_height,
                bd,
                topleft_off,
                dst,
            )
        }
    }
}

wrap_fn_ptr!(pub unsafe extern "C" fn cfl_ac(
    ac: &mut [i16; SCRATCH_AC_TXTP_LEN],
    y_ptr: *const DynPixel,
    stride: ptrdiff_t,
    w_pad: c_int,
    h_pad: c_int,
    cw: c_int,
    ch: c_int,
    _y: FFISafeRav1dPictureDataComponentOffset,
) -> ());

impl cfl_ac::Fn {
    pub fn call<BD: BitDepth>(
        &self,
        ac: &mut [i16; SCRATCH_AC_TXTP_LEN],
        y: Rav1dPictureDataComponentOffset,
        w_pad: c_int,
        h_pad: c_int,
        cw: c_int,
        ch: c_int,
    ) {
        let y_ptr = y.as_ptr::<BD>().cast();
        let stride = y.stride();
        let y = y.into_ffi_safe();
        // SAFETY: Fallback `fn cfl_ac_rust` is safe; asm is supposed to do the same.
        unsafe { self.get()(ac, y_ptr, stride, w_pad, h_pad, cw, ch, y) }
    }
}

wrap_fn_ptr!(pub unsafe extern "C" fn cfl_pred(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    ac: &[i16; SCRATCH_AC_TXTP_LEN],
    alpha: c_int,
    bitdepth_max: c_int,
    _topleft_off: usize,
    _dst: FFISafeRav1dPictureDataComponentOffset,
) -> ());

impl cfl_pred::Fn {
    pub fn call<BD: BitDepth>(
        &self,
        dst: Rav1dPictureDataComponentOffset,
        topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
        topleft_off: usize,
        width: c_int,
        height: c_int,
        ac: &[i16; SCRATCH_AC_TXTP_LEN],
        alpha: c_int,
        bd: BD,
    ) {
        let dst_ptr = dst.as_mut_ptr::<BD>().cast();
        let stride = dst.stride();
        let topleft = topleft[topleft_off..].as_ptr().cast();
        let bd = bd.into_c();
        let dst = dst.into_ffi_safe();
        // SAFETY: Fallback `fn cfl_pred` is safe; asm is supposed to do the same.
        unsafe {
            self.get()(
                dst_ptr,
                stride,
                topleft,
                width,
                height,
                ac,
                alpha,
                bd,
                topleft_off,
                dst,
            )
        }
    }
}

wrap_fn_ptr!(pub unsafe extern "C" fn pal_pred(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    pal: *const [DynPixel; 8],
    idx: *const u8,
    w: c_int,
    h: c_int,
    _dst: FFISafeRav1dPictureDataComponentOffset,
) -> ());

impl pal_pred::Fn {
    pub fn call<BD: BitDepth>(
        &self,
        dst: Rav1dPictureDataComponentOffset,
        pal: &[BD::Pixel; 8],
        idx: &[u8],
        w: c_int,
        h: c_int,
    ) {
        // SAFETY: `DisjointMut` is unchecked for asm `fn`s,
        // but passed through as an extra arg for the fallback `fn`.
        let dst_ptr = dst.as_mut_ptr::<BD>().cast();
        let stride = dst.stride();
        let pal = pal.as_ptr().cast();
        let idx = idx[..(w * h) as usize / 2].as_ptr();
        let dst = dst.into_ffi_safe();
        // SAFETY: Fallback `fn pal_pred_rust` is safe; asm is supposed to do the same.
        unsafe { self.get()(dst_ptr, stride, pal, idx, w, h, dst) }
    }
}

pub struct Rav1dIntraPredDSPContext {
    pub intra_pred: [angular_ipred::Fn; 14],
    pub cfl_ac: enum_map_ty!(Rav1dPixelLayoutSubSampled, cfl_ac::Fn),
    pub cfl_pred: [cfl_pred::Fn; 6],
    pub pal_pred: pal_pred::Fn,
}

#[inline(never)]
fn splat_dc<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    width: c_int,
    height: c_int,
    dc: c_int,
    bd: BD,
) {
    let height = height as isize;
    let width = width as usize;
    assert!(dc <= bd.bitdepth_max().as_::<c_int>());
    let dc = dc.as_::<BD::Pixel>();
    if BD::BPC == BPC::BPC8 && width > 4 {
        for y in 0..height {
            let dst = dst + y * dst.pixel_stride::<BD>();
            let dst = &mut *dst.slice_mut::<BD>(width);
            let dst = FromBytes::mut_slice_from(AsBytes::as_bytes_mut(dst)).unwrap();
            dst.fill([dc; 8]);
        }
    } else {
        for y in 0..height {
            let dst = dst + y * dst.pixel_stride::<BD>();
            let dst = &mut *dst.slice_mut::<BD>(width);
            let dst = FromBytes::mut_slice_from(AsBytes::as_bytes_mut(dst)).unwrap();
            dst.fill([dc; 4]);
        }
    };
}

#[inline(never)]
fn cfl_pred<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    width: c_int,
    height: c_int,
    dc: c_int,
    ac: &[i16; SCRATCH_AC_TXTP_LEN],
    alpha: c_int,
    bd: BD,
) {
    let width = width as usize;
    let height = height as usize;
    let mut ac = &ac[..width * height];
    for y in 0..height {
        let dst = dst + (y as isize * dst.pixel_stride::<BD>());
        let dst = &mut *dst.slice_mut::<BD>(width);
        for x in 0..width {
            let diff = alpha * ac[x] as c_int;
            dst[x] = bd.iclip_pixel(dc + apply_sign(diff.abs() + 32 >> 6, diff));
        }
        ac = &ac[width..];
    }
}

fn dc_gen_top<BD: BitDepth>(
    topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
    offset: usize,
    width: c_int,
) -> c_uint {
    let mut dc = width as u32 >> 1;
    for i in 0..width as usize {
        dc += topleft[offset + 1 + i].as_::<c_uint>();
    }
    return dc >> width.trailing_zeros();
}

fn dc_gen_left<BD: BitDepth>(
    topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
    offset: usize,
    height: c_int,
) -> c_uint {
    let mut dc = height as u32 >> 1;
    for i in 0..height as usize {
        dc += topleft[offset - (1 + i)].as_::<c_uint>();
    }
    return dc >> height.trailing_zeros();
}

fn dc_gen<BD: BitDepth>(
    topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
    offset: usize,
    width: c_int,
    height: c_int,
) -> c_uint {
    let (multiplier_1x2, multiplier_1x4, base_shift) = match BD::BPC {
        BPC::BPC8 => (0x5556, 0x3334, 16),
        BPC::BPC16 => (0xAAAB, 0x6667, 17),
    };

    let mut dc = (width + height >> 1) as u32;
    for i in 0..width as usize {
        dc += topleft[offset + i + 1].as_::<c_uint>();
    }
    for i in 0..height as usize {
        dc += topleft[offset - (i + 1)].as_::<c_uint>();
    }
    dc >>= (width + height).trailing_zeros();

    if width != height {
        dc *= if width > height * 2 || height > width * 2 {
            multiplier_1x4
        } else {
            multiplier_1x2
        };
        dc >>= base_shift;
    }
    return dc;
}

#[derive(FromRepr)]
#[repr(u8)]
enum DcGen {
    Top,
    Left,
    TopLeft,
}

impl DcGen {
    fn call<BD: BitDepth>(
        &self,
        topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
        offset: usize,
        width: c_int,
        height: c_int,
    ) -> c_uint {
        match self {
            Self::Top => dc_gen_top::<BD>(topleft, offset, width),
            Self::Left => dc_gen_left::<BD>(topleft, offset, height),
            Self::TopLeft => dc_gen::<BD>(topleft, offset, width, height),
        }
    }
}

/// Reconstructs the reference to the topleft edge array from a pointer into the
/// array and an offset from the start of the array.
///
/// The topleft pointer passed to asm is always a pointer into a buffer of
/// length [`SCRATCH_EDGE_LEN`]. For the Rust fallbacks we also pass in the
/// offset from the front of the buffer so that we can reconstruct the original
/// array reference in order to use safe array operations within the fallbacks.
///
/// # Safety
///
/// `topleft_ptr` must be a pointer into an array of length [`SCRATCH_EDGE_LEN`]
/// and is `topleft_off` elements from the beginning of the array. This should
/// be guaranteed by the logic in `angular_ipred::call`.
unsafe fn reconstruct_topleft<'a, BD: BitDepth>(
    topleft_ptr: *const DynPixel,
    topleft_off: usize,
) -> &'a [BD::Pixel; SCRATCH_EDGE_LEN] {
    // SAFETY: Same as `# Safety` preconditions.
    unsafe {
        &*topleft_ptr
            .cast::<BD::Pixel>()
            .sub(topleft_off)
            .cast::<[BD::Pixel; SCRATCH_EDGE_LEN]>()
    }
}

/// # Safety
///
/// Must be called by [`angular_ipred::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn ipred_dc_c_erased<BD: BitDepth, const DC_GEN: u8>(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    _a: c_int,
    _max_width: c_int,
    _max_height: c_int,
    bitdepth_max: c_int,
    topleft_off: usize,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    let dc_gen = DcGen::from_repr(DC_GEN).unwrap();

    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `angular_ipred::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    // SAFETY: `fn angular_ipred::Fn::call` makes `topleft` `topleft_off` from the beginning of the array.
    let topleft = unsafe { reconstruct_topleft::<BD>(topleft, topleft_off) };
    let dc = dc_gen.call::<BD>(topleft, topleft_off, width, height) as c_int;
    let bd = BD::from_c(bitdepth_max);
    splat_dc(dst, width, height, dc, bd)
}

/// # Safety
///
/// Must be called by [`cfl_pred::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn ipred_cfl_c_erased<BD: BitDepth, const DC_GEN: u8>(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    ac: &[i16; SCRATCH_AC_TXTP_LEN],
    alpha: c_int,
    bitdepth_max: c_int,
    topleft_off: usize,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    let dc_gen = DcGen::from_repr(DC_GEN).unwrap();

    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `cfl_pred::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    // SAFETY: `fn cfl_pred::Fn::call` makes `topleft` `topleft_off` from the beginning of the array.
    let topleft = unsafe { reconstruct_topleft::<BD>(topleft, topleft_off) };
    let dc = dc_gen.call::<BD>(topleft, topleft_off, width, height) as c_int;
    let bd = BD::from_c(bitdepth_max);
    cfl_pred(dst, width, height, dc, ac, alpha, bd)
}

/// # Safety
///
/// Must be called by [`angular_ipred::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn ipred_dc_128_c_erased<BD: BitDepth>(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    _topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    _a: c_int,
    _max_width: c_int,
    _max_height: c_int,
    bitdepth_max: c_int,
    _topleft_off: usize,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `angular_ipred::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    let bd = BD::from_c(bitdepth_max);
    let dc = bd.bitdepth_max().as_::<c_int>() + 1 >> 1;
    splat_dc(dst, width, height, dc, bd)
}

/// # Safety
///
/// Must be called by [`cfl_pred::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn ipred_cfl_128_c_erased<BD: BitDepth>(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    _topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    ac: &[i16; SCRATCH_AC_TXTP_LEN],
    alpha: c_int,
    bitdepth_max: c_int,
    _topleft_off: usize,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `cfl_pred::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    let bd = BD::from_c(bitdepth_max);
    let dc = bd.bitdepth_max().as_::<c_int>() + 1 >> 1;
    cfl_pred(dst, width, height, dc, ac, alpha, bd)
}

/// The intra predictions with the block width in the type, on `u8` or `u16` pixels, over
/// [`I16x8`]. DC sums its edges into lanes and stores its rows from a constant array.
mod rust_simd {
    use std::mem::size_of;

    use zerocopy::{AsBytes, FromBytes};

    use super::*;
    use crate::simd128::{transpose8x8, I16x8, I32x4};

    pub const SMOOTH: u8 = 0;
    pub const SMOOTH_V: u8 = 1;
    pub const SMOOTH_H: u8 = 2;

    /// `GEN` value of `DC_128_PRED`, after the three [`DcGen`]s.
    pub const DC_128: u8 = 3;

    macro_rules! by_size {
        (@ $wh:expr, $w:ident $h:ident $call:expr; $(($W:literal, $H:literal))*) => {
            match $wh {
                $(($W, $H) => {
                    const $w: usize = $W;
                    const $h: usize = $H;
                    $call
                })*
                _ => unreachable!(),
            }
        };
        ($width:expr, $height:expr, |$w:ident, $h:ident| $call:expr) => {
            by_size!(@ ($width, $height), $w $h $call;
                (4, 4) (4, 8) (4, 16) (8, 4) (8, 8) (8, 16) (8, 32) (16, 4) (16, 8) (16, 16)
                (16, 32) (16, 64) (32, 8) (32, 16) (32, 32) (32, 64) (64, 16) (64, 32) (64, 64))
        };
    }

    /// `$e` with `$P` the [`Pix`] of `$BD`.
    macro_rules! with_px {
        ($BD:ty, |$P:ident| $e:expr) => {
            if <$BD>::BITDEPTH == 8 {
                type $P = u8;
                $e
            } else {
                type $P = u16;
                $e
            }
        };
    }

    /// A pixel as the kernels take it: `u8`, or `u16` above 8 bpc. Every value they compute
    /// in 16-bit lanes fits up to 12 bpc.
    pub trait Pix: AsBytes + FromBytes + Copy + Default + Into<u32> {
        fn load8(s: &[Self]) -> I16x8;
        /// `s[..4]` into the low four lanes.
        fn load4(s: &[Self]) -> I16x8;
        /// The even and the odd pixels of `s[..16]`, or with `HALF` of `s[..8]` in the low
        /// four lanes.
        fn pairs<const HALF: bool>(s: &[Self]) -> (I16x8, I16x8);
        /// Lanes in the pixel range, saturated to it at 8 bpc.
        fn px8(v: I16x8) -> [Self; 8];
        fn from_u32(v: u32) -> Self;
    }

    impl Pix for u8 {
        #[inline(always)]
        fn load8(s: &[Self]) -> I16x8 {
            I16x8::from_u8x8(s)
        }

        #[inline(always)]
        fn load4(s: &[Self]) -> I16x8 {
            I16x8::from_u8x4(s)
        }

        #[inline(always)]
        fn pairs<const HALF: bool>(s: &[Self]) -> (I16x8, I16x8) {
            let v = if HALF { I16x8::from_bytes_half(s) } else { I16x8::from_bytes(s) };
            (v.and(I16x8::splat(0xff)), v.shr_u(8))
        }

        #[inline(always)]
        fn px8(v: I16x8) -> [Self; 8] {
            v.pack_u8()
        }

        #[inline(always)]
        fn from_u32(v: u32) -> Self {
            v as u8
        }
    }

    impl Pix for u16 {
        #[inline(always)]
        fn load8(s: &[Self]) -> I16x8 {
            I16x8::load_u16(s)
        }

        #[inline(always)]
        fn load4(s: &[Self]) -> I16x8 {
            I16x8::load_u16_lo(s)
        }

        #[inline(always)]
        fn pairs<const HALF: bool>(s: &[Self]) -> (I16x8, I16x8) {
            let hi = if HALF { I16x8::splat(0) } else { I16x8::load_u16(&s[8..]) };
            I16x8::load_u16(s).unzip(hi)
        }

        #[inline(always)]
        fn px8(v: I16x8) -> [Self; 8] {
            let mut p = [0; 8];
            v.store_u16(&mut p);
            p
        }

        #[inline(always)]
        fn from_u32(v: u32) -> Self {
            v as u16
        }
    }

    #[inline(always)]
    fn px<P: Pix>(b: &[u8]) -> &[P] {
        FromBytes::slice_from(b).unwrap()
    }

    #[inline(always)]
    fn px_mut<P: Pix>(b: &mut [u8]) -> &mut [P] {
        FromBytes::mut_slice_from(b).unwrap()
    }

    /// `px[..N]` summed into the lanes, `N` a power of two from 4. Up to 64 + 64 of them fit
    /// the lanes as `u16` to 12 bpc.
    #[inline(always)]
    fn lanes<P: Pix, const N: usize>(px: &[P]) -> I16x8 {
        if N == 4 {
            return P::load4(px);
        }
        let mut v = P::load8(px);
        for i in 1..N / 8 {
            v = v.add(P::load8(&px[8 * i..]));
        }
        v
    }

    /// `half` is the DC_128 value, half the pixel range.
    #[inline(always)]
    fn dc_value<P: Pix, const W: usize, const H: usize, const GEN: u8>(
        topleft: &[P],
        off: usize,
        half: u32,
    ) -> P {
        let top = || lanes::<P, W>(&topleft[off + 1..]);
        let left = || lanes::<P, H>(&topleft[off - H..]);
        let dc = match DcGen::from_repr(GEN) {
            Some(DcGen::Top) => top().hsum_u() + (W >> 1) as u32 >> W.trailing_zeros(),
            Some(DcGen::Left) => left().hsum_u() + (H >> 1) as u32 >> H.trailing_zeros(),
            Some(DcGen::TopLeft) => {
                let sum = top().add(left()).hsum_u();
                let dc = sum + ((W + H) >> 1) as u32 >> (W + H).trailing_zeros();
                // 1/3 and 1/5 in fixed point, finer for the larger sums of `u16` pixels.
                let (third, fifth, shift) =
                    if size_of::<P>() == 1 { (0x5556, 0x3334, 16) } else { (0xaaab, 0x6667, 17) };
                match () {
                    _ if W == H => dc,
                    _ if W > 2 * H || H > 2 * W => dc * fifth >> shift,
                    _ => dc * third >> shift,
                }
            }
            None => half,
        };
        P::from_u32(dc)
    }

    #[inline(always)]
    fn dc<BD: BitDepth, P: Pix, const W: usize, const H: usize, const GEN: u8>(
        dst: Rav1dPictureDataComponentOffset,
        topleft: &[P],
        off: usize,
        half: u32,
    ) {
        let row = [dc_value::<P, W, H, GEN>(topleft, off, half); W];
        let stride = dst.pixel_stride::<BD>();
        if stride < W as isize {
            for y in 0..H {
                let dst = dst + y as isize * stride;
                px_mut::<P>(dst.slice_mut::<BD>(W).as_bytes_mut()).copy_from_slice(&row);
            }
            return;
        }
        // One borrow for the block, ending at the last row: the rows above it are exact chunks.
        let stride = stride as usize;
        let mut block = dst.slice_mut::<BD>((H - 1) * stride + W);
        let (rows, last) = px_mut::<P>(block.as_bytes_mut()).split_at_mut((H - 1) * stride);
        for r in rows.chunks_exact_mut(stride) {
            r[..W].copy_from_slice(&row);
        }
        last.copy_from_slice(&row);
    }

    #[cfg_attr(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature(enable = "sse4.1")
    )]
    fn dc_128<BD: BitDepth, const GEN: u8>(
        dst: Rav1dPictureDataComponentOffset,
        topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
        off: usize,
        width: c_int,
        height: c_int,
        half: u32,
    ) {
        with_px!(BD, |P| {
            let topleft = px::<P>(topleft.as_bytes());
            by_size!(width, height, |W, H| dc::<BD, P, W, H, GEN>(dst, topleft, off, half))
        })
    }

    /// A block's SMOOTH constants: its bottom-left and top-right pixels and the bases below.
    #[derive(Clone, Copy)]
    struct Corners {
        bottom: I16x8,
        right: I16x8,
        v_base: I16x8,
        h_base: I16x8,
        base: I32x4,
    }

    /// At 8 bpc, `v = 256 * bottom + w_ver * (top - bottom)` and `h` the same across, each in
    /// 16 bits that wrap: exact, as the true values fit in `u16`. The rounding is in the bases;
    /// for SMOOTH, `h` carries 255 so that the unsigned average floors to `(v + h + 256) >> 1`.
    /// Above, SMOOTH sums the products in 32 bits, and SMOOTH_V and SMOOTH_H take
    /// `bottom + ((top - bottom) * w + 128 >> 8)` as a rounding multiply-high by `w << 7`.
    #[inline(always)]
    fn blend<P: Pix, const MODE: u8>(
        c: Corners,
        dt: I16x8,
        wv: I16x8,
        wh: I16x8,
        dl: I16x8,
    ) -> [P; 8] {
        if size_of::<P>() == 1 {
            let v = c.v_base.add(dt.mul(wv));
            let h = c.h_base.add(wh.mul(dl));
            return P::px8(
                match MODE {
                    SMOOTH => v.avg_u(h),
                    SMOOTH_V => v,
                    _ => h,
                }
                .shr_u(8),
            );
        }
        P::px8(match MODE {
            SMOOTH => {
                let [v0, v1] = dt.mul_wide(wv);
                let [h0, h1] = dl.mul_wide(wh);
                I32x4::pack(
                    v0.add(h0).add(c.base).shr_s::<9>(),
                    v1.add(h1).add(c.base).shr_s::<9>(),
                )
            }
            SMOOTH_V => c.bottom.add(dt.mulhrs(wv)),
            _ => c.right.add(dl.mulhrs(wh)),
        })
    }

    /// `w[..8]` shifted by `k`.
    #[inline(always)]
    fn weights(w: &[u8], k: u32) -> I16x8 {
        I16x8::from_u8x8(w).mul(I16x8::splat(1 << k))
    }

    /// `block` has a stride of at least `W`.
    #[inline(always)]
    fn smooth<P: Pix, const W: usize, const MODE: u8>(
        block: &mut [P],
        stride: usize,
        topleft: &[P],
        off: usize,
        h: usize,
    ) {
        // The weights as `blend` takes them.
        let k = if size_of::<P>() > 1 && MODE != SMOOTH { 7 } else { 0 };
        let w_hor = &dav1d_sm_weights.0[W..][..W];
        let w_ver = &dav1d_sm_weights.0[h..][..h];
        let top = &topleft[off + 1..][..W];
        let left = |y: usize| topleft[off - 1 - y];
        let right: u32 = topleft[off + W].into();
        let bottom: u32 = topleft[off - h].into();
        let (v_round, h_round) = if MODE == SMOOTH { (0, 255) } else { (128, 128) };
        let c = Corners {
            bottom: I16x8::splat(bottom as i16),
            right: I16x8::splat(right as i16),
            v_base: I16x8::splat((bottom * 256 + v_round) as i16),
            h_base: I16x8::splat((right * 256 + h_round) as i16),
            base: I32x4::splat((256 * (bottom + right) + 256) as i32),
        };
        if W == 4 {
            // Two rows a vector.
            let dt = P::load8(&std::array::from_fn::<_, 8, _>(|i| top[i % 4])).sub(c.bottom);
            let wh = weights(&std::array::from_fn::<_, 8, _>(|i| w_hor[i % 4]), k);
            for (y, rows) in block.chunks_mut(2 * stride).enumerate() {
                let y = 2 * y;
                let wv = weights(&std::array::from_fn::<_, 8, _>(|i| w_ver[y + i / 4]), k);
                let l = P::load8(&std::array::from_fn::<_, 8, _>(|i| left(y + i / 4)));
                let p = blend::<P, MODE>(c, dt, wv, wh, l.sub(c.right));
                rows[..4].copy_from_slice(&p[..4]);
                rows[stride..][..4].copy_from_slice(&p[4..]);
            }
            return;
        }
        let mut dt = [I16x8::splat(0); 8];
        let mut wh = [I16x8::splat(0); 8];
        for i in 0..W / 8 {
            dt[i] = P::load8(&top[8 * i..]).sub(c.bottom);
            wh[i] = weights(&w_hor[8 * i..], k);
        }
        for (y, row) in block.chunks_mut(stride).enumerate() {
            let wv = I16x8::splat((w_ver[y] as i16) << k);
            let dl = I16x8::splat(left(y).into() as i16).sub(c.right);
            for i in 0..W / 8 {
                row[8 * i..][..8].copy_from_slice(&blend::<P, MODE>(c, dt[i], wv, wh[i], dl));
            }
        }
    }

    #[cfg_attr(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature(enable = "sse4.1")
    )]
    fn smooth_128<BD: BitDepth, const MODE: u8>(
        dst: Rav1dPictureDataComponentOffset,
        topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
        off: usize,
        w: c_int,
        h: usize,
    ) {
        let stride = dst.pixel_stride::<BD>() as usize;
        let mut block = dst.slice_mut::<BD>((h - 1) * stride + w as usize);
        with_px!(BD, |P| {
            let (b, t) = (px_mut::<P>(block.as_bytes_mut()), px::<P>(topleft.as_bytes()));
            match w {
                4 => smooth::<P, 4, MODE>(b, stride, t, off, h),
                8 => smooth::<P, 8, MODE>(b, stride, t, off, h),
                16 => smooth::<P, 16, MODE>(b, stride, t, off, h),
                32 => smooth::<P, 32, MODE>(b, stride, t, off, h),
                64 => smooth::<P, 64, MODE>(b, stride, t, off, h),
                _ => unreachable!(),
            }
        })
    }

    /// `dc + sign(alpha * ac) * ((|alpha * ac| + 32) >> 6)` clipped to the pixels, with
    /// `alpha` as `|alpha| << 9` and its sign in `neg`: the rounding multiply-high of
    /// `|ac| <= 8 << 12` divides by 64.
    #[inline(always)]
    fn cfl_px<P: Pix>(ac: I16x8, alpha: I16x8, neg: bool, dc: I16x8, max: I16x8) -> [P; 8] {
        let m = ac.abs().mulhrs(alpha).with_sign_of(ac);
        let p = if neg { dc.sub(m) } else { dc.add(m) };
        P::px8(if size_of::<P>() == 1 { p } else { p.max(I16x8::splat(0)).min(max) })
    }

    /// `block` has a stride of at least `W`.
    #[inline(always)]
    fn cfl<P: Pix, const W: usize, const H: usize, const GEN: u8>(
        block: &mut [P],
        stride: usize,
        topleft: &[P],
        off: usize,
        ac: &[i16; SCRATCH_AC_TXTP_LEN],
        alpha: c_int,
        bitdepth_max: c_int,
    ) {
        let half = (bitdepth_max + 1) as u32 >> 1;
        let dc = I16x8::splat(dc_value::<P, W, H, GEN>(topleft, off, half).into() as i16);
        let max = I16x8::splat(bitdepth_max as i16);
        let neg = alpha < 0;
        let alpha = I16x8::splat((alpha.unsigned_abs() << 9) as i16);
        let ac = &ac[..W * H];
        if W == 4 {
            for (rows, ac) in block.chunks_mut(2 * stride).zip(ac.chunks_exact(8)) {
                let p = cfl_px::<P>(I16x8::load(ac), alpha, neg, dc, max);
                rows[..4].copy_from_slice(&p[..4]);
                rows[stride..][..4].copy_from_slice(&p[4..]);
            }
            return;
        }
        for (row, ac) in block.chunks_mut(stride).zip(ac.chunks_exact(W)) {
            for i in 0..W / 8 {
                let p = cfl_px::<P>(I16x8::load(&ac[8 * i..]), alpha, neg, dc, max);
                row[8 * i..][..8].copy_from_slice(&p);
            }
        }
    }

    #[cfg_attr(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature(enable = "sse4.1")
    )]
    fn cfl_128<BD: BitDepth, const GEN: u8>(
        dst: Rav1dPictureDataComponentOffset,
        topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
        off: usize,
        width: c_int,
        height: c_int,
        ac: &[i16; SCRATCH_AC_TXTP_LEN],
        alpha: c_int,
        bitdepth_max: c_int,
    ) {
        let stride = dst.pixel_stride::<BD>() as usize;
        let mut block = dst.slice_mut::<BD>((height as usize - 1) * stride + width as usize);
        with_px!(BD, |P| {
            let (b, t) = (px_mut::<P>(block.as_bytes_mut()), px::<P>(topleft.as_bytes()));
            by_size!(width, height, |W, H| cfl::<P, W, H, GEN>(
                b,
                stride,
                t,
                off,
                ac,
                alpha,
                bitdepth_max
            ))
        })
    }

    /// Luma at chroma column `x` of one row, the pairs across summed when subsampled: the low
    /// `HALF` four lanes, or eight.
    #[inline(always)]
    fn luma_row<P: Pix, const SS_HOR: bool, const HALF: bool>(r: &[P], x: usize) -> I16x8 {
        match (SS_HOR, HALF) {
            (true, _) => {
                let (even, odd) = P::pairs::<HALF>(&r[2 * x..]);
                even.add(odd)
            }
            (false, true) => P::load4(&r[x..]),
            (false, false) => P::load8(&r[x..]),
        }
    }

    /// The subsampled sum scaled to eight times a pixel.
    #[inline(always)]
    fn luma<P: Pix, const SS_HOR: bool, const SS_VER: bool, const HALF: bool>(
        r0: &[P],
        r1: &[P],
        x: usize,
    ) -> I16x8 {
        let mut v = luma_row::<P, SS_HOR, HALF>(r0, x);
        if SS_VER {
            v = v.add(luma_row::<P, SS_HOR, HALF>(r1, x));
        }
        for _ in 0..1 + !SS_VER as u8 + !SS_HOR as u8 {
            v = v.add(v);
        }
        v
    }

    #[inline(always)]
    fn cfl_ac_rows<BD: BitDepth, P: Pix, const W: usize, const SS_HOR: bool, const SS_VER: bool>(
        ac: &mut [i16; SCRATCH_AC_TXTP_LEN],
        y_src: Rav1dPictureDataComponentOffset,
        w_pad: usize,
        h_pad: usize,
        h: usize,
    ) {
        let ac = &mut ac[..W * h];
        assert!(4 * w_pad < W && 4 * h_pad < h);
        let (wv, hv) = (W - 4 * w_pad, h - 4 * h_pad);
        let stride = y_src.pixel_stride::<BD>();
        let lw = wv << SS_HOR as u8;
        // The sum as the rows are made, in 32 bits: the vectors, then the padding.
        let mut wide = [I32x4::splat(0); 2];
        let mut pad = 0;
        for (y, row) in ac.chunks_exact_mut(W).take(hv).enumerate() {
            let src = y_src + (y as isize * stride << SS_VER as u8);
            let r0 = src.slice::<BD>(lw);
            let r1 = SS_VER.then(|| (src + stride).slice::<BD>(lw));
            let r0 = px::<P>(r0.as_bytes());
            let r1 = r1.as_deref().map_or(&[][..], |r| px::<P>(r.as_bytes()));
            let mut x = 0;
            while x + 8 <= wv {
                let v = luma::<P, SS_HOR, SS_VER, false>(r0, r1, x);
                v.store(&mut row[x..]);
                wide = v.addw_u(wide);
                x += 8;
            }
            if x < wv {
                let v = luma::<P, SS_HOR, SS_VER, true>(r0, r1, x);
                v.store_lo(&mut row[x..]);
                wide = v.addw_u(wide);
            }
            let last = row[wv - 1];
            row[wv..].fill(last);
            pad += (W - wv) as u32 * last as u32;
        }
        if hv < h {
            let last: u32 = ac[(hv - 1) * W..hv * W].iter().map(|&v| v as u32).sum();
            pad += (h - hv) as u32 * last;
        }
        for y in hv..h {
            ac.copy_within((hv - 1) * W..hv * W, y * W);
        }
        let mut lanes = [0; 4];
        wide[0].add(wide[1]).store(&mut lanes);
        let sum = lanes.iter().sum::<i32>() as u32 + pad;
        let log2sz = W.trailing_zeros() + h.trailing_zeros();
        let mean = I16x8::splat((sum + (1 << log2sz >> 1) >> log2sz) as i16);
        for c in ac.chunks_exact_mut(8) {
            I16x8::load(c).sub(mean).store(c);
        }
    }

    #[cfg_attr(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature(enable = "sse4.1")
    )]
    fn cfl_ac_128<BD: BitDepth, const SS_HOR: bool, const SS_VER: bool>(
        ac: &mut [i16; SCRATCH_AC_TXTP_LEN],
        y_src: Rav1dPictureDataComponentOffset,
        w_pad: usize,
        h_pad: usize,
        w: c_int,
        h: usize,
    ) {
        with_px!(BD, |P| match w {
            4 => cfl_ac_rows::<BD, P, 4, SS_HOR, SS_VER>(ac, y_src, w_pad, h_pad, h),
            8 => cfl_ac_rows::<BD, P, 8, SS_HOR, SS_VER>(ac, y_src, w_pad, h_pad, h),
            16 => cfl_ac_rows::<BD, P, 16, SS_HOR, SS_VER>(ac, y_src, w_pad, h_pad, h),
            32 => cfl_ac_rows::<BD, P, 32, SS_HOR, SS_VER>(ac, y_src, w_pad, h_pad, h),
            _ => unreachable!(),
        })
    }

    /// Room for a directional edge up to `max <= 127`, padded past it by [`dir_rows`].
    const EXT: usize = 128 + 2 * 64 + 16;

    /// `a + ((b - a) * frac + 32 >> 6)` for `a = e[i + STEP * x]`, `b` the next, with `frac`
    /// scaled by `1 << 9` so that the rounding multiply-high divides by 64.
    #[inline(always)]
    fn dir8<P: Pix, const STEP: usize>(e: &[P], i: usize, frac: I16x8) -> [P; 8] {
        let (a, b) = if STEP == 1 {
            (P::load8(&e[i..]), P::load8(&e[i + 1..]))
        } else {
            P::pairs::<false>(&e[i..])
        };
        P::px8(a.add(b.sub(a).mulhrs(frac)))
    }

    /// `n` rows of `W` pixels `os` apart in `out`, row `r` read from `e` at `(r + 1) * d / 64`
    /// with steps of `STEP`: the directional prediction along an edge whose last pixel is
    /// `e[max]`. The edge is padded with that pixel, which the rows past it take.
    #[inline(always)]
    fn dir_rows<P: Pix, const W: usize, const STEP: usize>(
        out: &mut [P],
        os: usize,
        n: usize,
        e: &mut [P; EXT],
        max: usize,
        d: usize,
    ) {
        let last = e[max];
        e[max + 1..].fill(last);
        for (r, row) in out.chunks_mut(os).take(n).enumerate() {
            let pos = (r + 1) * d;
            let base = pos >> 6;
            if base >= max {
                row[..W].fill(last);
                continue;
            }
            let frac = I16x8::splat(((pos & 0x3e) << 9) as i16);
            for x in (0..W).step_by(8) {
                let k = (W - x).min(8);
                row[x..x + k].copy_from_slice(&dir8::<P, STEP>(e, base + STEP * x, frac)[..k]);
            }
        }
    }

    /// [`filter_edge`] in vectors: the clipped input laid out once, padded with its end
    /// pixels, then the 5-tap kernel over 8 outputs at a time; the outputs outside
    /// `lim_from..lim_to` are the input copied. The sums wrap in 16 bits, but their true
    /// values fit in `u16`.
    #[inline(always)]
    fn filter_edge_px<P: Pix>(
        out: &mut [P],
        sz: usize,
        lim_from: usize,
        lim_to: usize,
        r#in: &[P],
        in_off: usize,
        from: isize,
        to: isize,
        strength: usize,
    ) {
        static KERNEL: [[i16; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];

        // `p[k]` is the input at `clip(k - 2, from, to - 1)`.
        let mut p = [P::default(); 128 + 4 + 8];
        let at = |i: isize| r#in[in_off.wrapping_add_signed(i)];
        let lo = (from + 2).clamp(0, (sz + 4) as isize) as usize;
        let hi = (to + 2).clamp(lo as isize, (sz + 4) as isize) as usize;
        p[..lo].fill(at(from));
        p[lo..hi].copy_from_slice(&r#in[in_off.wrapping_add_signed(lo as isize - 2)..][..hi - lo]);
        p[hi..].fill(at(to - 1));
        let k = KERNEL[strength - 1].map(I16x8::splat);
        for i in (0..sz).step_by(8) {
            let mut s = I16x8::splat(8);
            for j in 0..5 {
                s = s.add(P::load8(&p[i + j..]).mul(k[j]));
            }
            let n = (sz - i).min(8);
            out[i..i + n].copy_from_slice(&P::px8(s.shr_u(4))[..n]);
        }
        for i in (0..lim_from.min(sz)).chain(lim_to.min(sz)..sz) {
            out[i] = p[i + 2];
        }
    }

    /// [`filter_edge_px`] in the shape of [`filter_edge`], for the edge helpers.
    #[inline(always)]
    fn filter_edge_simd<BD: BitDepth>(
        out: &mut [BD::Pixel],
        sz: c_int,
        lim_from: c_int,
        lim_to: c_int,
        r#in: &[BD::Pixel; SCRATCH_EDGE_LEN],
        in_off: usize,
        from: c_int,
        to: c_int,
        strength: c_int,
    ) {
        let [sz, lim_from, lim_to, strength] = [sz, lim_from, lim_to, strength].map(|v| v.max(0) as usize);
        with_px!(BD, |P| filter_edge_px::<P>(
            px_mut(out.as_bytes_mut()),
            sz,
            lim_from,
            lim_to,
            px(r#in.as_bytes()),
            in_off,
            from as isize,
            to as isize,
            strength,
        ))
    }

    /// The edge `e` as `P`s, as long as it is.
    #[inline(always)]
    fn edge_px<P: Pix, const N: usize>(e: &mut [u8]) -> &mut [P; N] {
        px_mut::<P>(e).try_into().unwrap()
    }

    /// `dst` has a stride of at least `width`.
    #[cfg_attr(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature(enable = "sse4.1")
    )]
    fn z1_128<BD: BitDepth>(
        dst: Rav1dPictureDataComponentOffset,
        topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
        off: usize,
        width: c_int,
        height: c_int,
        angle: c_int,
        bd: BD,
    ) {
        let mut e = [0.into(); EXT];
        let (dx, max, upsample) =
            z1_edge::<BD>(&mut e, topleft, off, width, height, angle, bd, filter_edge_simd::<BD>);
        let (w, h, dx) = (width as usize, height as usize, dx as usize);
        let stride = dst.pixel_stride::<BD>() as usize;
        let mut block = dst.slice_mut::<BD>((h - 1) * stride + w);
        with_px!(BD, |P| {
            let e = edge_px::<P, EXT>(e.as_bytes_mut());
            let block = px_mut::<P>(block.as_bytes_mut());
            match (w, upsample) {
                (4, false) => dir_rows::<P, 4, 1>(block, stride, h, e, max, dx),
                (8, false) => dir_rows::<P, 8, 1>(block, stride, h, e, max, dx),
                (16, false) => dir_rows::<P, 16, 1>(block, stride, h, e, max, dx),
                (32, false) => dir_rows::<P, 32, 1>(block, stride, h, e, max, dx),
                (64, false) => dir_rows::<P, 64, 1>(block, stride, h, e, max, dx),
                (4, true) => dir_rows::<P, 4, 2>(block, stride, h, e, max, dx),
                (8, true) => dir_rows::<P, 8, 2>(block, stride, h, e, max, dx),
                _ => unreachable!(),
            }
        })
    }

    /// The length of a scratch column, a block side and the 8 lanes a step may write past it.
    const TS: usize = 64 + 8;

    /// The first `cols` of the `W` columns of `t`, `H` pixels each, as the rows of `out`, `os`
    /// apart; with 8 columns a step, up to `cols` rounded up to 8.
    #[inline(always)]
    fn transpose_cols<P: Pix, const W: usize, const H: usize>(
        t: &[[P; TS]; W],
        cols: usize,
        out: &mut [P],
        os: usize,
    ) {
        if W % 8 != 0 || H % 8 != 0 {
            for (y, row) in out.chunks_mut(os).take(H).enumerate() {
                for x in 0..cols {
                    row[x] = t[x][y];
                }
            }
            return;
        }
        for y in (0..H).step_by(8) {
            for x in (0..cols).step_by(8) {
                // A loop, not a closure: a closure would not inherit the kernel's target features.
                let mut v = [I16x8::splat(0); 8];
                for i in 0..8 {
                    v[i] = P::load8(&t[x + i][y..]);
                }
                transpose8x8(&mut v);
                for (i, v) in v.into_iter().enumerate() {
                    out[(y + i) * os + x..][..8].copy_from_slice(&P::px8(v));
                }
            }
        }
    }

    /// Z3 is Z1 along the left edge read upwards, its columns the rows of `t`.
    #[inline(always)]
    fn z3<P: Pix, const W: usize, const H: usize, const STEP: usize>(
        block: &mut [P],
        stride: usize,
        e: &mut [P; EXT],
        dy: usize,
        max: usize,
    ) {
        let mut t = [[P::default(); TS]; W];
        dir_rows::<P, H, STEP>(t.as_flattened_mut(), TS, W, e, max, dy);
        transpose_cols::<P, W, H>(&t, W, block, stride);
    }

    /// Two-tap interpolation of 8 pixels from `e[i..]`, the steps 2 apart where upsampled.
    #[inline(always)]
    fn dir8_at<P: Pix>(e: &[P], i: usize, frac: I16x8, upsampled: bool) -> [P; 8] {
        if upsampled {
            dir8::<P, 2>(e, i, frac)
        } else {
            dir8::<P, 1>(e, i, frac)
        }
    }

    /// Z2 from its edge, top-left pixel at 64. A row takes the left edge up to the column where
    /// the top edge starts. Down a column the left edge is read in a straight line with one
    /// fraction, so the left part runs per column into a scratch turned into the block's rows;
    /// the top part then runs per row over the rest of it.
    #[inline(always)]
    fn z2<P: Pix, const W: usize, const H: usize>(
        block: &mut [P],
        stride: usize,
        edge: &[P; 64 + 64 + 1],
        dx: usize,
        dy: usize,
        up_above: bool,
        up_left: bool,
    ) {
        // The top edge with `TL` at the top-left pixel and the left edge reversed, `r[PAD + j]`
        // the pixel `j` below it; the pads take the lanes before a row's or column's start.
        const TL: usize = 80;
        const PAD: usize = 16;
        let mut e = [P::default(); TL + 65 + 16];
        e[TL - 64..][..129].copy_from_slice(edge);
        let mut r = [P::default(); PAD + 128];
        for (j, r) in r[PAD..][..65].iter_mut().enumerate() {
            *r = edge[64 - j];
        }
        let (inc_x, inc_y) = (1 + up_above as usize, 1 + up_left as usize);
        let xpos = |y: usize| ((inc_x as isize) << 6) - (dx * (y + 1)) as isize;

        // Row `y` takes the left edge in columns `..split[y]`; column `x` from row `y0[x]`.
        let mut split = [0; H];
        let mut y0 = [H; W];
        let mut left_cols = 0;
        for y in 0..H {
            let base = xpos(y) >> 6;
            // `-base` over `inc_x` rounded up, as a shift.
            let up = up_above as usize;
            let s = if base >= 0 { 0 } else { ((-base) as usize + up >> up).min(W) };
            for x in left_cols..s {
                y0[x] = y;
            }
            left_cols = left_cols.max(s);
            split[y] = s;
        }

        let mut t = [[P::default(); TS]; W];
        for x in 0..left_cols {
            let c = (dy * (x + 1)) as isize;
            let frac = I16x8::splat(((-c & 0x3e) << 9) as i16);
            let j0 = PAD as isize + (-c >> 6) + inc_y as isize;
            for y in (y0[x] & !7..H).step_by(8) {
                let j = (j0 + (inc_y * y) as isize) as usize;
                t[x][y..y + 8].copy_from_slice(&dir8_at(&r, j, frac, up_left));
            }
        }
        if left_cols != 0 {
            transpose_cols::<P, W, H>(&t, left_cols, block, stride);
        }

        for (y, row) in block.chunks_mut(stride).enumerate() {
            let s = split[y];
            if s == W {
                continue;
            }
            let mut px = [P::default(); 64 + 8];
            let pos = xpos(y);
            let frac = I16x8::splat(((pos & 0x3e) << 9) as i16);
            for x in (s & !7..W).step_by(8) {
                let i = (TL as isize + (pos >> 6) + (inc_x * x) as isize) as usize;
                px[x..x + 8].copy_from_slice(&dir8_at(&e, i, frac, up_above));
            }
            row[s..W].copy_from_slice(&px[s..W]);
        }
    }

    /// `dst` has a stride of at least `width`.
    #[cfg_attr(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature(enable = "sse4.1")
    )]
    fn z2_128<BD: BitDepth>(
        dst: Rav1dPictureDataComponentOffset,
        topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
        off: usize,
        width: c_int,
        height: c_int,
        angle: c_int,
        max_width: c_int,
        max_height: c_int,
        bd: BD,
    ) {
        let mut edge = [0.into(); 64 + 64 + 1];
        let (dx, dy, up_above, up_left) = z2_edge::<BD>(
            &mut edge,
            topleft,
            off,
            width,
            height,
            angle,
            max_width,
            max_height,
            bd,
            filter_edge_simd::<BD>,
        );
        let (dx, dy) = (dx as usize, dy as usize);
        let stride = dst.pixel_stride::<BD>() as usize;
        let mut block = dst.slice_mut::<BD>((height as usize - 1) * stride + width as usize);
        with_px!(BD, |P| {
            let edge = edge_px::<P, { 64 + 64 + 1 }>(edge.as_bytes_mut());
            let block = px_mut::<P>(block.as_bytes_mut());
            by_size!(width, height, |W, H| z2::<P, W, H>(
                block, stride, edge, dx, dy, up_above, up_left
            ))
        })
    }

    /// `dst` has a stride of at least `width`.
    #[cfg_attr(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature(enable = "sse4.1")
    )]
    fn z3_128<BD: BitDepth>(
        dst: Rav1dPictureDataComponentOffset,
        topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
        off: usize,
        width: c_int,
        height: c_int,
        angle: c_int,
        bd: BD,
    ) {
        let mut left = [0.into(); 64 + 64];
        let (dy, left_off, max, upsample) =
            z3_edge::<BD>(&mut left, topleft, off, width, height, angle, bd, filter_edge_simd::<BD>);
        let stride = dst.pixel_stride::<BD>() as usize;
        let mut block = dst.slice_mut::<BD>((height as usize - 1) * stride + width as usize);
        with_px!(BD, |P| {
            let left = px::<P>(left.as_bytes());
            let mut e = [P::default(); EXT];
            for (i, e) in e[..=max].iter_mut().enumerate() {
                *e = left[left_off - i];
            }
            let (e, block) = (&mut e, px_mut::<P>(block.as_bytes_mut()));
            by_size!(width, height, |W, H| if upsample {
                z3::<P, W, H, 2>(block, stride, e, dy, max)
            } else {
                z3::<P, W, H, 1>(block, stride, e, dy, max)
            })
        })
    }

    /// # Safety
    ///
    /// Must be called by [`angular_ipred::Fn::call`], on x86 once the CPU flags hold SSE41.
    #[deny(unsafe_op_in_unsafe_fn)]
    #[allow(unused_unsafe)]
    pub unsafe extern "C" fn ipred_z1_erased<BD: BitDepth>(
        _dst_ptr: *mut DynPixel,
        _stride: ptrdiff_t,
        topleft: *const DynPixel,
        width: c_int,
        height: c_int,
        angle: c_int,
        max_width: c_int,
        max_height: c_int,
        bitdepth_max: c_int,
        topleft_off: usize,
        dst: FFISafeRav1dPictureDataComponentOffset,
    ) {
        // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `angular_ipred::Fn::call`.
        let dst = unsafe { FFISafe::from_with_offset(dst) };
        // SAFETY: `fn angular_ipred::Fn::call` makes `topleft` `topleft_off` from the beginning of the array.
        let topleft = unsafe { reconstruct_topleft::<BD>(topleft, topleft_off) };
        let bd = BD::from_c(bitdepth_max);
        if dst.pixel_stride::<BD>() < width as isize {
            return ipred_z1_rust::<BD>(
                dst, topleft, topleft_off, width, height, angle, max_width, max_height, bd,
            );
        }
        // SAFETY: SSE4.1 is checked before this `fn` is chosen on x86; elsewhere it is safe.
        unsafe { z1_128::<BD>(dst, topleft, topleft_off, width, height, angle, bd) }
    }

    /// # Safety
    ///
    /// Must be called by [`angular_ipred::Fn::call`], on x86 once the CPU flags hold SSE41.
    #[deny(unsafe_op_in_unsafe_fn)]
    #[allow(unused_unsafe)]
    pub unsafe extern "C" fn ipred_z2_erased<BD: BitDepth>(
        _dst_ptr: *mut DynPixel,
        _stride: ptrdiff_t,
        topleft: *const DynPixel,
        width: c_int,
        height: c_int,
        angle: c_int,
        max_width: c_int,
        max_height: c_int,
        bitdepth_max: c_int,
        topleft_off: usize,
        dst: FFISafeRav1dPictureDataComponentOffset,
    ) {
        // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `angular_ipred::Fn::call`.
        let dst = unsafe { FFISafe::from_with_offset(dst) };
        // SAFETY: `fn angular_ipred::Fn::call` makes `topleft` `topleft_off` from the beginning of the array.
        let topleft = unsafe { reconstruct_topleft::<BD>(topleft, topleft_off) };
        let bd = BD::from_c(bitdepth_max);
        if dst.pixel_stride::<BD>() < width as isize {
            return ipred_z2_rust::<BD>(
                dst, topleft, topleft_off, width, height, angle, max_width, max_height, bd,
            );
        }
        // SAFETY: SSE4.1 is checked before this `fn` is chosen on x86; elsewhere it is safe.
        unsafe {
            z2_128::<BD>(
                dst, topleft, topleft_off, width, height, angle, max_width, max_height, bd,
            )
        }
    }

    /// # Safety
    ///
    /// Must be called by [`angular_ipred::Fn::call`], on x86 once the CPU flags hold SSE41.
    #[deny(unsafe_op_in_unsafe_fn)]
    #[allow(unused_unsafe)]
    pub unsafe extern "C" fn ipred_z3_erased<BD: BitDepth>(
        _dst_ptr: *mut DynPixel,
        _stride: ptrdiff_t,
        topleft: *const DynPixel,
        width: c_int,
        height: c_int,
        angle: c_int,
        max_width: c_int,
        max_height: c_int,
        bitdepth_max: c_int,
        topleft_off: usize,
        dst: FFISafeRav1dPictureDataComponentOffset,
    ) {
        // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `angular_ipred::Fn::call`.
        let dst = unsafe { FFISafe::from_with_offset(dst) };
        // SAFETY: `fn angular_ipred::Fn::call` makes `topleft` `topleft_off` from the beginning of the array.
        let topleft = unsafe { reconstruct_topleft::<BD>(topleft, topleft_off) };
        let bd = BD::from_c(bitdepth_max);
        if dst.pixel_stride::<BD>() < width as isize {
            return ipred_z3_rust::<BD>(
                dst, topleft, topleft_off, width, height, angle, max_width, max_height, bd,
            );
        }
        // SAFETY: SSE4.1 is checked before this `fn` is chosen on x86; elsewhere it is safe.
        unsafe { z3_128::<BD>(dst, topleft, topleft_off, width, height, angle, bd) }
    }

    /// `GEN` is a [`DcGen`] or [`DC_128`].
    ///
    /// # Safety
    ///
    /// Must be called by [`angular_ipred::Fn::call`], on x86 once the CPU flags hold SSE41.
    #[deny(unsafe_op_in_unsafe_fn)]
    #[allow(unused_unsafe)]
    pub unsafe extern "C" fn ipred_dc_erased<BD: BitDepth, const GEN: u8>(
        _dst_ptr: *mut DynPixel,
        _stride: ptrdiff_t,
        topleft: *const DynPixel,
        width: c_int,
        height: c_int,
        _a: c_int,
        _max_width: c_int,
        _max_height: c_int,
        bitdepth_max: c_int,
        topleft_off: usize,
        dst: FFISafeRav1dPictureDataComponentOffset,
    ) {
        // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `angular_ipred::Fn::call`.
        let dst = unsafe { FFISafe::from_with_offset(dst) };
        // SAFETY: `fn angular_ipred::Fn::call` makes `topleft` `topleft_off` from the beginning of the array.
        let topleft = unsafe { reconstruct_topleft::<BD>(topleft, topleft_off) };
        let half = (bitdepth_max + 1) as u32 >> 1;
        // SAFETY: SSE4.1 is checked before this `fn` is chosen on x86; elsewhere it is safe.
        unsafe { dc_128::<BD, GEN>(dst, topleft, topleft_off, width, height, half) }
    }

    /// `MODE` is [`SMOOTH`], [`SMOOTH_V`] or [`SMOOTH_H`].
    ///
    /// # Safety
    ///
    /// Must be called by [`angular_ipred::Fn::call`], on x86 once the CPU flags hold SSE41.
    #[deny(unsafe_op_in_unsafe_fn)]
    #[allow(unused_unsafe)]
    pub unsafe extern "C" fn ipred_smooth_erased<BD: BitDepth, const MODE: u8>(
        _dst_ptr: *mut DynPixel,
        _stride: ptrdiff_t,
        topleft: *const DynPixel,
        width: c_int,
        height: c_int,
        _a: c_int,
        _max_width: c_int,
        _max_height: c_int,
        _bitdepth_max: c_int,
        topleft_off: usize,
        dst: FFISafeRav1dPictureDataComponentOffset,
    ) {
        // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `angular_ipred::Fn::call`.
        let dst = unsafe { FFISafe::from_with_offset(dst) };
        // SAFETY: `fn angular_ipred::Fn::call` makes `topleft` `topleft_off` from the beginning of the array.
        let topleft = unsafe { reconstruct_topleft::<BD>(topleft, topleft_off) };
        if dst.pixel_stride::<BD>() < width as isize {
            return match MODE {
                SMOOTH => ipred_smooth_rust::<BD>(dst, topleft, topleft_off, width, height),
                SMOOTH_V => ipred_smooth_v_rust::<BD>(dst, topleft, topleft_off, width, height),
                _ => ipred_smooth_h_rust::<BD>(dst, topleft, topleft_off, width, height),
            };
        }
        // SAFETY: SSE4.1 is checked before this `fn` is chosen on x86; elsewhere it is safe.
        unsafe { smooth_128::<BD, MODE>(dst, topleft, topleft_off, width, height as usize) }
    }

    /// `GEN` is a [`DcGen`] or [`DC_128`].
    ///
    /// # Safety
    ///
    /// Must be called by [`cfl_pred::Fn::call`], on x86 once the CPU flags hold SSE41.
    #[deny(unsafe_op_in_unsafe_fn)]
    #[allow(unused_unsafe)]
    pub unsafe extern "C" fn ipred_cfl_erased<BD: BitDepth, const GEN: u8>(
        _dst_ptr: *mut DynPixel,
        _stride: ptrdiff_t,
        topleft: *const DynPixel,
        width: c_int,
        height: c_int,
        ac: &[i16; SCRATCH_AC_TXTP_LEN],
        alpha: c_int,
        bitdepth_max: c_int,
        topleft_off: usize,
        dst: FFISafeRav1dPictureDataComponentOffset,
    ) {
        // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `cfl_pred::Fn::call`.
        let dst = unsafe { FFISafe::from_with_offset(dst) };
        // SAFETY: `fn cfl_pred::Fn::call` makes `topleft` `topleft_off` from the beginning of the array.
        let topleft = unsafe { reconstruct_topleft::<BD>(topleft, topleft_off) };
        if dst.pixel_stride::<BD>() < width as isize {
            let dc = match DcGen::from_repr(GEN) {
                Some(g) => g.call::<BD>(topleft, topleft_off, width, height) as c_int,
                None => (bitdepth_max + 1) >> 1,
            };
            return super::cfl_pred(dst, width, height, dc, ac, alpha, BD::from_c(bitdepth_max));
        }
        // SAFETY: SSE4.1 is checked before this `fn` is chosen on x86; elsewhere it is safe.
        unsafe {
            cfl_128::<BD, GEN>(dst, topleft, topleft_off, width, height, ac, alpha, bitdepth_max)
        }
    }

    /// # Safety
    ///
    /// Must be called by [`cfl_ac::Fn::call`], on x86 once the CPU flags hold SSE41.
    #[deny(unsafe_op_in_unsafe_fn)]
    #[allow(unused_unsafe)]
    pub unsafe extern "C" fn cfl_ac_erased<
        BD: BitDepth,
        const SS_HOR: bool,
        const SS_VER: bool,
    >(
        ac: &mut [i16; SCRATCH_AC_TXTP_LEN],
        _y_ptr: *const DynPixel,
        _stride: ptrdiff_t,
        w_pad: c_int,
        h_pad: c_int,
        cw: c_int,
        ch: c_int,
        y: FFISafeRav1dPictureDataComponentOffset,
    ) {
        // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `cfl_ac::Fn::call`.
        let y = unsafe { FFISafe::from_with_offset(y) };
        let [w_pad, h_pad, ch] = [w_pad, h_pad, ch].map(|v| usize::try_from(v).unwrap());
        // SAFETY: SSE4.1 is checked before this `fn` is chosen on x86; elsewhere it is safe.
        unsafe { cfl_ac_128::<BD, SS_HOR, SS_VER>(ac, y, w_pad, h_pad, cw, ch) }
    }
}

fn ipred_v_rust<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
    topleft_off: usize,
    width: c_int,
    height: c_int,
) {
    let width = width as usize;
    let height = height as usize;

    for y in 0..height {
        let dst = dst + (y as isize * dst.pixel_stride::<BD>());
        BD::pixel_copy(
            &mut *dst.slice_mut::<BD>(width),
            &topleft[topleft_off + 1..][..width],
            width,
        );
    }
}

/// # Safety
///
/// Must be called by [`angular_ipred::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn ipred_v_c_erased<BD: BitDepth>(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    _a: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    topleft_off: usize,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `angular_ipred::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    // SAFETY: `fn angular_ipred::Fn::call` makes `topleft` `topleft_off` from the beginning of the array.
    let topleft = unsafe { reconstruct_topleft::<BD>(topleft, topleft_off) };
    ipred_v_rust::<BD>(dst, topleft, topleft_off, width, height)
}

fn ipred_h_rust<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
    topleft_off: usize,
    width: c_int,
    height: c_int,
) {
    let width = width as usize;
    let height = height as usize;

    for y in 0..height {
        let dst = dst + (y as isize * dst.pixel_stride::<BD>());
        BD::pixel_set(
            &mut *dst.slice_mut::<BD>(width),
            topleft[topleft_off - (1 + y)],
            width,
        );
    }
}

/// # Safety
///
/// Must be called by [`angular_ipred::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn ipred_h_c_erased<BD: BitDepth>(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    _a: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    topleft_off: usize,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `angular_ipred::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    // SAFETY: `fn angular_ipred::Fn::call` makes `topleft` `topleft_off` from the beginning of the array.
    let topleft = unsafe { reconstruct_topleft::<BD>(topleft, topleft_off) };
    ipred_h_rust::<BD>(dst, topleft, topleft_off, width, height)
}

fn ipred_paeth_rust<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    tl: &[BD::Pixel; SCRATCH_EDGE_LEN],
    tl_off: usize,
    width: c_int,
    height: c_int,
) {
    let width = width as usize;
    let height = height as usize;

    let topleft = tl[tl_off].as_::<c_int>();
    for y in 0..height {
        let left = tl[tl_off - (y + 1)].as_::<c_int>();
        let dst = dst + (y as isize * dst.pixel_stride::<BD>());
        let dst = &mut *dst.slice_mut::<BD>(width);
        for x in 0..width {
            let top = tl[tl_off + 1 + x].as_::<c_int>();
            let base = left + top - topleft;
            let ldiff = (left - base).abs();
            let tdiff = (top - base).abs();
            let tldiff = (topleft - base).abs();

            dst[x] = (if ldiff <= tdiff && ldiff <= tldiff {
                left
            } else if tdiff <= tldiff {
                top
            } else {
                topleft
            })
            .as_::<BD::Pixel>();
        }
    }
}

/// # Safety
///
/// Must be called by [`angular_ipred::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn ipred_paeth_c_erased<BD: BitDepth>(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    tl_ptr: *const DynPixel,
    width: c_int,
    height: c_int,
    _a: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    topleft_off: usize,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `angular_ipred::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    // SAFETY: `fn angular_ipred::Fn::call` makes `topleft` `topleft_off` from the beginning of the array.
    let topleft = unsafe { reconstruct_topleft::<BD>(tl_ptr, topleft_off) };
    ipred_paeth_rust::<BD>(dst, topleft, topleft_off, width, height)
}

fn ipred_smooth_rust<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
    topleft_off: usize,
    width: c_int,
    height: c_int,
) {
    let [width, height] = [width, height].map(|it| it as usize);

    let weights_hor = &dav1d_sm_weights.0[width..][..width];
    let weights_ver = &dav1d_sm_weights.0[height..][..height];
    let right = topleft[topleft_off + width].as_::<c_int>();
    let bottom = topleft[topleft_off - height].as_::<c_int>();

    for y in 0..height {
        let dst = dst + (y as isize * dst.pixel_stride::<BD>());
        let dst = &mut *dst.slice_mut::<BD>(width);
        for x in 0..width {
            let pred = weights_ver[y] as c_int * topleft[topleft_off + 1 + x].as_::<c_int>()
                + (256 - weights_ver[y] as c_int) * bottom
                + weights_hor[x] as c_int * topleft[topleft_off - (1 + y)].as_::<c_int>()
                + (256 - weights_hor[x] as c_int) * right;
            dst[x] = (pred + 256 >> 9).as_::<BD::Pixel>();
        }
    }
}

/// # Safety
///
/// Must be called by [`angular_ipred::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn ipred_smooth_c_erased<BD: BitDepth>(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    _a: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    topleft_off: usize,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `angular_ipred::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    // SAFETY: `fn angular_ipred::Fn::call` makes `topleft` `topleft_off` from the beginning of the array.
    let topleft = unsafe { reconstruct_topleft::<BD>(topleft, topleft_off) };
    ipred_smooth_rust::<BD>(dst, topleft, topleft_off, width, height)
}

fn ipred_smooth_v_rust<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
    topleft_off: usize,
    width: c_int,
    height: c_int,
) {
    let [width, height] = [width, height].map(|it| it as usize);

    let weights_ver = &dav1d_sm_weights.0[height..][..height];
    let bottom = topleft[topleft_off - height].as_::<c_int>();

    for y in 0..height {
        let dst = dst + (y as isize * dst.pixel_stride::<BD>());
        let dst = &mut *dst.slice_mut::<BD>(width);
        for x in 0..width {
            let pred = weights_ver[y] as c_int * topleft[topleft_off + 1 + x].as_::<c_int>()
                + (256 - weights_ver[y] as c_int) * bottom;
            dst[x] = (pred + 128 >> 8).as_::<BD::Pixel>();
        }
    }
}

/// # Safety
///
/// Must be called by [`angular_ipred::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn ipred_smooth_v_c_erased<BD: BitDepth>(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    _a: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    topleft_off: usize,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `angular_ipred::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    // SAFETY: `fn angular_ipred::Fn::call` makes `topleft` `topleft_off` from the beginning of the array.
    let topleft = unsafe { reconstruct_topleft::<BD>(topleft, topleft_off) };
    ipred_smooth_v_rust::<BD>(dst, topleft, topleft_off, width, height)
}

fn ipred_smooth_h_rust<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
    topleft_off: usize,
    width: c_int,
    height: c_int,
) {
    let [width, height] = [width, height].map(|it| it as usize);

    let weights_hor = &dav1d_sm_weights.0[width..][..width];
    let right = topleft[topleft_off + width].as_::<c_int>();

    for y in 0..height {
        let dst = dst + (y as isize * dst.pixel_stride::<BD>());
        let dst = &mut *dst.slice_mut::<BD>(width);
        for x in 0..width {
            let pred = weights_hor[x] as c_int * topleft[topleft_off - (y + 1)].as_::<c_int>()
                + (256 - weights_hor[x] as c_int) * right;
            dst[x] = (pred + 128 >> 8).as_::<BD::Pixel>();
        }
    }
}

/// # Safety
///
/// Must be called by [`angular_ipred::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn ipred_smooth_h_c_erased<BD: BitDepth>(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    _a: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    topleft_off: usize,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `angular_ipred::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    // SAFETY: `fn angular_ipred::Fn::call` makes `topleft` `topleft_off` from the beginning of the array.
    let topleft = unsafe { reconstruct_topleft::<BD>(topleft, topleft_off) };
    ipred_smooth_h_rust::<BD>(dst, topleft, topleft_off, width, height)
}

#[inline(never)]
fn get_filter_strength(wh: c_int, angle: c_int, is_sm: bool) -> c_int {
    if is_sm {
        match (wh, angle) {
            (..=8, 64..) => 2,
            (..=8, 40..) => 1,
            (..=8, ..) => 0,
            (..=16, 48..) => 2,
            (..=16, 20..) => 1,
            (..=16, ..) => 0,
            (..=24, 4..) => 3,
            (..=24, ..) => 0,
            (.., _) => 3,
        }
    } else {
        match (wh, angle) {
            (..=8, 56..) => 1,
            (..=8, ..) => 0,
            (..=16, 40..) => 1,
            (..=16, ..) => 0,
            (..=24, 32..) => 3,
            (..=24, 16..) => 2,
            (..=24, 8..) => 1,
            (..=24, ..) => 0,
            (..=32, 32..) => 3,
            (..=32, 4..) => 2,
            (..=32, ..) => 1,
            (.., _) => 3,
        }
    }
}

#[inline(never)]
fn filter_edge<BD: BitDepth>(
    out: &mut [BD::Pixel],
    sz: c_int,
    lim_from: c_int,
    lim_to: c_int,
    r#in: &[BD::Pixel; SCRATCH_EDGE_LEN],
    in_off: usize,
    from: c_int,
    to: c_int,
    strength: c_int,
) {
    static KERNEL: [[u8; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];

    assert!(strength > 0);
    let mut i = 0;
    while i < cmp::min(sz, lim_from) {
        out[i as usize] = r#in[in_off + iclip(i, from, to - 1) as usize];
        i += 1;
    }
    while i < cmp::min(lim_to, sz) {
        let mut s = 0;
        for j in 0..5 {
            s += r#in[in_off.wrapping_add_signed(iclip(i - 2 + j, from, to - 1) as isize)]
                .as_::<c_int>()
                * KERNEL[(strength - 1) as usize][j as usize] as c_int;
        }
        out[i as usize] = (s + 8 >> 4).as_::<BD::Pixel>();
        i += 1;
    }
    while i < sz {
        out[i as usize] = r#in[in_off + iclip(i, from, to - 1) as usize];
        i += 1;
    }
}

#[inline]
fn get_upsample(wh: c_int, angle: c_int, is_sm: bool) -> bool {
    angle < 40 && wh <= (16 >> is_sm as u8)
}

#[inline(never)]
fn upsample_edge<BD: BitDepth>(
    out: &mut [BD::Pixel],
    hsz: c_int,
    r#in: &[BD::Pixel; SCRATCH_EDGE_LEN],
    in_off: usize,
    from: c_int,
    to: c_int,
    bd: BD,
) {
    static KERNEL: [i8; 4] = [-1, 9, 9, -1];
    for i in 0..hsz - 1 {
        out[(i * 2) as usize] = r#in[in_off + iclip(i, from, to - 1) as usize];
        let mut s = 0;
        for j in 0..4 {
            s += r#in[in_off.wrapping_add_signed(iclip(i + j - 1, from, to - 1) as isize)]
                .as_::<c_int>()
                * KERNEL[j as usize] as c_int;
        }
        out[(i * 2 + 1) as usize] =
            iclip(s + 8 >> 4, 0, bd.bitdepth_max().as_::<c_int>()).as_::<BD::Pixel>();
    }
    let i = hsz - 1;
    out[(i * 2) as usize] = r#in[in_off + iclip(i, from, to - 1) as usize];
}

/// Z1's top edge into `top`, filtered or upsampled as the angle asks: `(dx, max_base_x,
/// upsample_above)`.
#[inline(always)]
fn z1_edge<BD: BitDepth>(
    top: &mut [BD::Pixel],
    topleft_in: &[BD::Pixel; SCRATCH_EDGE_LEN],
    topleft_in_off: usize,
    width: c_int,
    height: c_int,
    mut angle: c_int,
    bd: BD,
    filter: impl FnOnce(&mut [BD::Pixel], c_int, c_int, c_int, &[BD::Pixel; SCRATCH_EDGE_LEN], usize, c_int, c_int, c_int),
) -> (c_int, usize, bool) {
    let is_sm = (angle >> 9) & 1 != 0;
    let enable_intra_edge_filter = (angle >> 10) != 0;
    angle &= 511;
    assert!(angle < 90);
    let mut dx = dav1d_dr_intra_derivative[(angle >> 1) as usize] as c_int;
    let upsample_above = if enable_intra_edge_filter {
        get_upsample(width + height, 90 - angle, is_sm)
    } else {
        false
    };
    let max_base_x = if upsample_above {
        upsample_edge::<BD>(
            top,
            width + height,
            topleft_in,
            topleft_in_off + 1,
            -1,
            width + cmp::min(width, height),
            bd,
        );
        dx <<= 1;
        2 * (width + height) - 2
    } else {
        let filter_strength = if enable_intra_edge_filter {
            get_filter_strength(width + height, 90 - angle, is_sm)
        } else {
            0
        };
        if filter_strength != 0 {
            filter(
                top,
                width + height,
                0,
                width + height,
                topleft_in,
                topleft_in_off + 1,
                -1,
                width + cmp::min(width, height),
                filter_strength,
            );
            width + height - 1
        } else {
            let n = (width + cmp::min(width, height)) as usize;
            BD::pixel_copy(&mut top[..n], &topleft_in[topleft_in_off + 1..][..n], n);
            n as c_int - 1
        }
    };
    (dx, max_base_x as usize, upsample_above)
}

fn ipred_z1_rust<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    topleft_in: &[BD::Pixel; SCRATCH_EDGE_LEN],
    topleft_in_off: usize,
    width: c_int,
    height: c_int,
    angle: c_int,
    _max_width: c_int,
    _max_height: c_int,
    bd: BD,
) {
    let mut top = [0.into(); 64 + 64];
    let (dx, max_base_x, upsample_above) =
        z1_edge::<BD>(&mut top, topleft_in, topleft_in_off, width, height, angle, bd, filter_edge::<BD>);
    let width = width as usize;
    let base_inc = 1 + upsample_above as usize;
    for y in 0..height {
        let xpos = (y + 1) * dx;
        let frac = xpos & 0x3e;

        let dst = dst + (y as isize * dst.pixel_stride::<BD>());
        let dst = &mut *dst.slice_mut::<BD>(width);
        for x in 0..width {
            let base = (xpos >> 6) as usize + base_inc * x;
            if base < max_base_x {
                let v =
                    top[base].as_::<c_int>() * (64 - frac) + top[base + 1].as_::<c_int>() * frac;
                dst[x] = (v + 32 >> 6).as_::<BD::Pixel>();
            } else {
                BD::pixel_set(&mut dst[x..], top[max_base_x], width - x);
                break;
            }
        }
    }
}

/// Z2's edge into `edge`, its top-left pixel at 64, the top filtered or upsampled as the
/// angle asks and the left the same: `(dx, dy, upsample_above, upsample_left)`.
#[inline(always)]
fn z2_edge<BD: BitDepth>(
    edge: &mut [BD::Pixel; 64 + 64 + 1],
    topleft_in: &[BD::Pixel; SCRATCH_EDGE_LEN],
    topleft_in_off: usize,
    width: c_int,
    height: c_int,
    mut angle: c_int,
    max_width: c_int,
    max_height: c_int,
    bd: BD,
    filter: impl FnOnce(&mut [BD::Pixel], c_int, c_int, c_int, &[BD::Pixel; SCRATCH_EDGE_LEN], usize, c_int, c_int, c_int) + Copy,
) -> (c_int, c_int, bool, bool) {
    let is_sm = (angle >> 9) & 1 != 0;
    let enable_intra_edge_filter = angle >> 10;
    angle &= 511;
    assert!(angle > 90 && angle < 180);
    let mut dy = dav1d_dr_intra_derivative[(angle - 90 >> 1) as usize] as c_int;
    let mut dx = dav1d_dr_intra_derivative[(180 - angle >> 1) as usize] as c_int;
    let upsample_left = if enable_intra_edge_filter != 0 {
        get_upsample(width + height, 180 - angle, is_sm)
    } else {
        false
    };
    let upsample_above = if enable_intra_edge_filter != 0 {
        get_upsample(width + height, angle - 90, is_sm)
    } else {
        false
    };
    let topleft = 64;

    if upsample_above {
        upsample_edge::<BD>(
            &mut edge[topleft..],
            width + 1,
            topleft_in,
            topleft_in_off,
            0,
            width + 1,
            bd,
        );
        dx <<= 1;
    } else {
        let filter_strength = if enable_intra_edge_filter != 0 {
            get_filter_strength(width + height, angle - 90, is_sm)
        } else {
            0
        };
        if filter_strength != 0 {
            filter(
                &mut edge[topleft + 1..],
                width,
                0,
                max_width,
                topleft_in,
                topleft_in_off + 1,
                -1,
                width,
                filter_strength,
            );
        } else {
            let width = width as usize;
            BD::pixel_copy(
                &mut edge[topleft + 1..][..width],
                &topleft_in[topleft_in_off + 1..][..width],
                width,
            );
        }
    }
    if upsample_left {
        upsample_edge::<BD>(
            &mut edge[topleft - height as usize * 2..],
            height + 1,
            topleft_in,
            topleft_in_off - height as usize,
            0,
            height + 1,
            bd,
        );
        dy <<= 1;
    } else {
        let filter_strength_0 = if enable_intra_edge_filter != 0 {
            get_filter_strength(width + height, 180 - angle, is_sm)
        } else {
            0
        };
        if filter_strength_0 != 0 {
            filter(
                &mut edge[topleft - height as usize..],
                height,
                height - max_height,
                height,
                topleft_in,
                topleft_in_off - height as usize,
                0,
                height + 1,
                filter_strength_0,
            );
        } else {
            let height = height as usize;
            BD::pixel_copy(
                &mut edge[topleft - height..][..height],
                &topleft_in[topleft_in_off - height..][..height],
                height,
            );
        }
    }
    edge[topleft] = topleft_in[topleft_in_off];

    (dx, dy, upsample_above, upsample_left)
}

fn ipred_z2_rust<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    topleft_in: &[BD::Pixel; SCRATCH_EDGE_LEN],
    topleft_in_off: usize,
    width: c_int,
    height: c_int,
    angle: c_int,
    max_width: c_int,
    max_height: c_int,
    bd: BD,
) {
    let mut edge = [0.into(); 64 + 64 + 1];
    let topleft = 64;
    let (dx, dy, upsample_above, upsample_left) = z2_edge::<BD>(
        &mut edge,
        topleft_in,
        topleft_in_off,
        width,
        height,
        angle,
        max_width,
        max_height,
        bd,
        filter_edge::<BD>,
    );
    let base_inc_x = 1 + upsample_above as usize;
    let left = topleft - (1 + upsample_left as usize);
    let width = width as usize;
    for y in 0..height {
        let xpos = (1 + (upsample_above as c_int) << 6) - (dx * (y + 1));
        let base_x = xpos >> 6;
        let frac_x = xpos & 0x3e;

        let dst = dst + (y as isize * dst.pixel_stride::<BD>());
        let dst = &mut *dst.slice_mut::<BD>(width);
        for x in 0..width {
            let ypos = (y << 6 + upsample_left as c_int) - (dy * (x + 1) as c_int);
            let base_x = base_x + (base_inc_x * x) as c_int;
            let v = if base_x >= 0 {
                edge[topleft + base_x as usize].as_::<c_int>() * (64 - frac_x)
                    + edge[topleft + base_x as usize + 1].as_::<c_int>() * frac_x
            } else {
                let base_y = ypos >> 6;
                assert!(base_y >= -(1 + upsample_left as c_int));
                let frac_y = ypos & 0x3e;
                edge[left.wrapping_add_signed(-base_y as isize)].as_::<c_int>() * (64 - frac_y)
                    + edge[left.wrapping_add_signed(-(base_y + 1) as isize)].as_::<c_int>() * frac_y
            };
            dst[x] = (v + 32 >> 6).as_::<BD::Pixel>();
        }
    }
}

/// Z3's left edge into `left`, filtered or upsampled as the angle asks, read downwards from
/// `left[left_off]`: `(dy, left_off, max_base_y, upsample_left)`.
#[inline(always)]
fn z3_edge<BD: BitDepth>(
    left: &mut [BD::Pixel],
    topleft_in: &[BD::Pixel; SCRATCH_EDGE_LEN],
    topleft_in_off: usize,
    width: c_int,
    height: c_int,
    mut angle: c_int,
    bd: BD,
    filter: impl FnOnce(&mut [BD::Pixel], c_int, c_int, c_int, &[BD::Pixel; SCRATCH_EDGE_LEN], usize, c_int, c_int, c_int),
) -> (usize, usize, usize, bool) {
    let is_sm = (angle >> 9) & 1 != 0;
    let enable_intra_edge_filter = angle >> 10;
    angle &= 511;
    assert!(angle > 180);
    let mut dy = dav1d_dr_intra_derivative[(270 - angle >> 1) as usize] as usize;
    let upsample_left = if enable_intra_edge_filter != 0 {
        get_upsample(width + height, angle - 180, is_sm)
    } else {
        false
    };
    let (left_off, max_base_y) = if upsample_left {
        upsample_edge::<BD>(
            left,
            width + height,
            topleft_in,
            topleft_in_off - (width + height) as usize,
            cmp::max(width - height, 0),
            width + height + 1,
            bd,
        );
        dy <<= 1;
        (2 * (width + height) as usize - 2, 2 * (width + height) - 2)
    } else {
        let filter_strength = if enable_intra_edge_filter != 0 {
            get_filter_strength(width + height, angle - 180, is_sm)
        } else {
            0
        };

        if filter_strength != 0 {
            filter(
                left,
                width + height,
                0,
                width + height,
                topleft_in,
                topleft_in_off - (width + height) as usize,
                cmp::max(width - height, 0),
                width + height + 1,
                filter_strength,
            );
            ((width + height - 1) as usize, width + height - 1)
        } else {
            let n = (height + cmp::min(width, height)) as usize;
            BD::pixel_copy(&mut left[..n], &topleft_in[topleft_in_off - n..][..n], n);
            (n - 1, n as c_int - 1)
        }
    };
    (dy, left_off, max_base_y as usize, upsample_left)
}

fn ipred_z3_rust<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    topleft_in: &[BD::Pixel; SCRATCH_EDGE_LEN],
    topleft_in_off: usize,
    width: c_int,
    height: c_int,
    angle: c_int,
    _max_width: c_int,
    _max_height: c_int,
    bd: BD,
) {
    let mut left = [0.into(); 64 + 64];
    let (dy, left_off, max_base_y, upsample_left) =
        z3_edge::<BD>(&mut left, topleft_in, topleft_in_off, width, height, angle, bd, filter_edge::<BD>);
    let base_inc = 1 + upsample_left as usize;
    let width = width as usize;
    let height = height as usize;
    for x in 0..width {
        let ypos = dy * (x + 1);
        let frac = (ypos & 0x3e) as i32;

        for y in 0..height {
            let base = (ypos >> 6) + base_inc * y;
            if base < max_base_y {
                let v = left[left_off - base].as_::<i32>() * (64 - frac)
                    + left[left_off - (base + 1)].as_::<i32>() * frac;
                *(dst + y as isize * dst.pixel_stride::<BD>() + x).index_mut::<BD>() =
                    (v + 32 >> 6).as_::<BD::Pixel>();
            } else {
                for y in y..height {
                    *(dst + y as isize * dst.pixel_stride::<BD>() + x).index_mut::<BD>() =
                        left[left_off - max_base_y];
                }
                break;
            }
        }
    }
}

/// # Safety
///
/// Must be called by [`angular_ipred::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn ipred_z_c_erased<BD: BitDepth, const Z: usize>(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    topleft_in: *const DynPixel,
    width: c_int,
    height: c_int,
    angle: c_int,
    max_width: c_int,
    max_height: c_int,
    bitdepth_max: c_int,
    topleft_off: usize,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `angular_ipred::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    // SAFETY: `fn angular_ipred::Fn::call` makes `topleft` `topleft_off` from the beginning of the array.
    let topleft_in = unsafe { reconstruct_topleft::<BD>(topleft_in, topleft_off) };
    let bd = BD::from_c(bitdepth_max);
    [ipred_z1_rust, ipred_z2_rust, ipred_z3_rust][Z - 1](
        dst,
        topleft_in,
        topleft_off,
        width,
        height,
        angle,
        max_width,
        max_height,
        bd,
    )
}

fn ipred_filter_rust<BD: BitDepth>(
    mut dst: Rav1dPictureDataComponentOffset,
    topleft_in: &[BD::Pixel; SCRATCH_EDGE_LEN],
    topleft_off: usize,
    width: c_int,
    height: c_int,
    filt_idx: c_int,
    _max_width: c_int,
    _max_height: c_int,
    bd: BD,
) {
    let width = width as usize / 4 * 4; // To elide bounds checks.
    let height = height as usize;
    let filt_idx = filt_idx as usize;
    let stride = dst.pixel_stride::<BD>();
    let filt_idx = filt_idx & 511;

    let filter = &dav1d_filter_intra_taps[filt_idx];
    let mut top = &topleft_in[topleft_off + 1..][..width];
    let mut top_guard;
    for y in (0..height).step_by(2) {
        let topleft_off = topleft_off - y;
        let mut topleft = topleft_in[topleft_off];
        for x in (0..width).step_by(4) {
            let p0 = topleft;
            let [p1, p2, p3, p4] = top[x..][..4].try_into().unwrap();
            let p5;
            let p6;
            if x == 0 {
                let left = &topleft_in[topleft_off - 1 - 1..][..2];
                p5 = left[1];
                p6 = left[0];
            } else {
                let left = dst + (x - 1);
                p5 = *left.index::<BD>();
                p6 = *(left + stride).index::<BD>();
            }
            let p = [p0, p1, p2, p3, p4, p5, p6].map(|p| p.as_::<i32>());
            let mut ptr = dst + x;
            let mut flt_ptr = filter.0.as_slice();

            for _yy in 0..2 {
                let ptr_slice = &mut *ptr.slice_mut::<BD>(4);
                for xx in ptr_slice {
                    let acc = filter_fn(flt_ptr, p);
                    *xx = bd.iclip_pixel(acc + 8 >> 4);
                    flt_ptr = &flt_ptr[FLT_INCR..];
                }
                ptr += stride;
            }
            topleft = p4;
        }
        dst += stride;
        top_guard = dst.slice::<BD>(width);
        top = &*top_guard;
        dst += stride;
    }
}

unsafe extern "C" fn ipred_filter_c_erased<BD: BitDepth>(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    topleft_in: *const DynPixel,
    width: c_int,
    height: c_int,
    filt_idx: c_int,
    max_width: c_int,
    max_height: c_int,
    bitdepth_max: c_int,
    topleft_off: usize,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `angular_ipred::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    // SAFETY: `fn angular_ipred::Fn::call` makes `topleft` `topleft_off` from the beginning of the array.
    let topleft = unsafe { reconstruct_topleft::<BD>(topleft_in, topleft_off) };
    let bd = BD::from_c(bitdepth_max);
    ipred_filter_rust(
        dst,
        topleft,
        topleft_off,
        width,
        height,
        filt_idx,
        max_width,
        max_height,
        bd,
    )
}

#[inline(never)]
fn cfl_ac_rust<BD: BitDepth>(
    ac: &mut [i16; SCRATCH_AC_TXTP_LEN],
    y_src: Rav1dPictureDataComponentOffset,
    w_pad: c_int,
    h_pad: c_int,
    width: usize,
    height: usize,
    is_ss_hor: bool,
    is_ss_ver: bool,
) {
    let ac = &mut ac[..width * height];
    let [w_pad, h_pad] = [w_pad, h_pad].map(|pad| usize::try_from(pad).unwrap() * 4);
    assert!(w_pad < width);
    assert!(h_pad < height);
    let [ss_hor, ss_ver] = [is_ss_hor, is_ss_ver].map(|is_ss| is_ss as u8);
    let y_pxstride = y_src.pixel_stride::<BD>();

    for y in 0..height - h_pad {
        let y_src = y_src + (y as isize * y_pxstride << ss_ver);
        let aci = y * width;
        let y_src = |i: isize| (*(y_src + i).index::<BD>()).as_::<i32>();
        for x in 0..width - w_pad {
            let sx = (x << ss_hor) as isize;
            let mut ac_sum = y_src(sx);
            if is_ss_hor {
                ac_sum += y_src(sx + 1);
            }
            if is_ss_ver {
                ac_sum += y_src(sx + y_pxstride);
                if is_ss_hor {
                    ac_sum += y_src(sx + y_pxstride + 1);
                }
            }
            ac[aci + x] = (ac_sum << 1 + !is_ss_ver as u8 + !is_ss_hor as u8) as i16;
        }
        for x in width - w_pad..width {
            ac[aci + x] = ac[aci + x - 1];
        }
    }
    for y in height - h_pad..height {
        let aci = y * width;
        let (src, dst) = ac.split_at_mut(aci);
        dst[..width].copy_from_slice(&src[src.len() - width..]);
    }

    let log2sz = width.trailing_zeros() + height.trailing_zeros();
    let mut sum = 1 << log2sz >> 1;
    for y in 0..height {
        let aci = y * width;
        for x in 0..width {
            sum += ac[aci + x] as i32;
        }
    }
    let sum = (sum >> log2sz) as i16;

    // subtract DC
    for y in 0..height {
        let aci = y * width;
        for x in 0..width {
            ac[aci + x] -= sum;
        }
    }
}

/// # Safety
///
/// Must be called by [`cfl_ac::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn cfl_ac_c_erased<BD: BitDepth, const IS_SS_HOR: bool, const IS_SS_VER: bool>(
    ac: &mut [i16; SCRATCH_AC_TXTP_LEN],
    _y_ptr: *const DynPixel,
    _stride: ptrdiff_t,
    w_pad: c_int,
    h_pad: c_int,
    cw: c_int,
    ch: c_int,
    y: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `cfl_ac::Fn::call`.
    let y = unsafe { FFISafe::from_with_offset(y) };
    let cw = cw as usize;
    let ch = ch as usize;
    cfl_ac_rust::<BD>(ac, y, w_pad, h_pad, cw, ch, IS_SS_HOR, IS_SS_VER);
}

fn pal_pred_rust<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    pal: &[BD::Pixel; 8],
    idx: &[u8],
    w: c_int,
    h: c_int,
) {
    let w = w as usize;
    let h = h as usize;
    let idx = &idx[..w * h / 2];

    let mut j = 0;
    for y in 0..h {
        let dst = dst + (y as isize * dst.pixel_stride::<BD>());
        for x in (0..w).step_by(2) {
            let i = idx[j];
            j += 1;
            assert!((i & 0x88) == 0);
            let dst = &mut *(dst + x).slice_mut::<BD>(2);
            dst[0] = pal[(i & 7) as usize];
            dst[1] = pal[(i >> 4) as usize];
        }
    }
}

/// # Safety
///
/// Must be called by [`pal_pred::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn pal_pred_c_erased<BD: BitDepth>(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    pal: *const [DynPixel; 8],
    idx: *const u8,
    w: c_int,
    h: c_int,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `FFISafe::new(dst)` in `pal_pred::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    // SAFETY: Undoing dyn cast in `pal_pred::Fn::call`.
    let pal = unsafe { &*pal.cast() };
    // SAFETY: Length sliced in `pal_pred::Fn::call`.
    let idx = unsafe { slice::from_raw_parts(idx, (w * h) as usize / 2) };
    pal_pred_rust::<BD>(dst, pal, idx, w, h)
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
mod neon {
    use to_method::To;

    use super::*;
    #[cfg(feature = "bitdepth_16")]
    use crate::include::common::bitdepth::BitDepth16;
    #[cfg(feature = "bitdepth_8")]
    use crate::include::common::bitdepth::BitDepth8;

    wrap_fn_ptr!(unsafe extern "C" fn z13_fill(
        dst: *mut DynPixel,
        stride: ptrdiff_t,
        topleft: *const DynPixel,
        width: c_int,
        height: c_int,
        dxy: c_int,
        max_base_xy: c_int,
    ) -> ());

    impl z13_fill::Fn {
        pub fn call<BD: BitDepth>(
            &self,
            dst: *mut BD::Pixel,
            stride: ptrdiff_t,
            topleft: &[BD::Pixel],
            width: c_int,
            height: c_int,
            dxy: c_int,
            max_base_xy: c_int,
        ) {
            let dst = dst.cast();
            let topleft = topleft.as_ptr().cast();
            // SAFETY: We're assuming the asm is actually correct and safe.
            unsafe { self.get()(dst, stride, topleft, width, height, dxy, max_base_xy) }
        }
    }

    wrap_fn_ptr!(unsafe extern "C" fn z2_fill(
        dst: *mut DynPixel,
        stride: ptrdiff_t,
        top: *const DynPixel,
        left: *const DynPixel,
        width: c_int,
        height: c_int,
        dx: c_int,
        dy: c_int,
    ) -> ());

    impl z2_fill::Fn {
        pub fn call<BD: BitDepth>(
            &self,
            dst: *mut BD::Pixel,
            stride: ptrdiff_t,
            top: &[BD::Pixel],
            left: &[BD::Pixel],
            width: c_int,
            height: c_int,
            dx: c_int,
            dy: c_int,
        ) {
            let dst = dst.cast();
            let top = top.as_ptr().cast();
            let left = left.as_ptr().cast();
            // SAFETY: We're assuming the asm is actually correct and safe.
            unsafe { self.get()(dst, stride, top, left, width, height, dx, dy) }
        }
    }

    wrap_fn_ptr!(unsafe extern "C" fn z1_upsample_edge(
        out: *mut DynPixel,
        hsz: c_int,
        in_0: *const DynPixel,
        end: c_int,
        _bitdepth_max: c_int,
    ) -> ());

    impl z1_upsample_edge::Fn {
        pub fn call<BD: BitDepth>(
            &self,
            out: &mut [BD::Pixel],
            hsz: c_int,
            in_0: &[BD::Pixel],
            end: c_int,
            bd: BD,
        ) {
            let out = out.as_mut_ptr().cast();
            let in_0 = in_0.as_ptr().cast();
            let bd = bd.into_c();
            // SAFETY: We're assuming the asm is actually correct and safe.
            unsafe { self.get()(out, hsz, in_0, end, bd) }
        }
    }

    wrap_fn_ptr!(unsafe extern "C" fn z1_filter_edge(
        out: *mut DynPixel,
        sz: c_int,
        in_0: *const DynPixel,
        end: c_int,
        strength: c_int,
    ) -> ());

    impl z1_filter_edge::Fn {
        pub fn call<BD: BitDepth>(
            &self,
            out: &mut [BD::Pixel],
            sz: c_int,
            in_0: &[BD::Pixel],
            end: c_int,
            strength: c_int,
        ) {
            let out = out.as_mut_ptr().cast();
            let in_0 = in_0.as_ptr().cast();
            // SAFETY: We're assuming the asm is actually correct and safe.
            unsafe { self.get()(out, sz, in_0, end, strength) }
        }
    }

    wrap_fn_ptr!(unsafe extern "C" fn z2_upsample_edge(
        out: *mut DynPixel,
        hsz: c_int,
        in_0: *const DynPixel,
        _bitdepth_max: c_int,
    ) -> ());

    impl z2_upsample_edge::Fn {
        pub fn call<BD: BitDepth>(
            &self,
            out: &mut [BD::Pixel],
            hsz: c_int,
            in_0: &[BD::Pixel],
            bd: BD,
        ) {
            let out = out.as_mut_ptr().cast();
            let in_0 = in_0.as_ptr().cast();
            let bd = bd.into_c();
            // SAFETY: We're assuming the asm is actually correct and safe.
            unsafe { self.get()(out, hsz, in_0, bd) }
        }
    }

    wrap_fn_ptr!(unsafe extern "C" fn reverse(
        dst: *mut DynPixel,
        src: *const DynPixel,
        n: c_int,
    ) -> ());

    impl reverse::Fn {
        pub fn call<BD: BitDepth>(&self, dst: &mut [BD::Pixel], src: &[BD::Pixel], n: c_int) {
            let dst = dst.as_mut_ptr().cast();
            let src = src.as_ptr().cast();
            // SAFETY: We're assuming the asm is actually correct and safe.
            unsafe { self.get()(dst, src, n) }
        }
    }

    fn rav1d_ipred_pixel_set_neon<BD: BitDepth>(out: &mut [BD::Pixel], px: BD::Pixel, n: c_int) {
        // `pixel_set` takes a `px: BD::Pixel`.
        // Since it's not behind a ptr, we can't make it a `DynPixel`
        // and call it uniformly with `bd_fn!`.

        extern "C" {
            #[cfg(feature = "bitdepth_8")]
            fn dav1d_ipred_pixel_set_8bpc_neon(
                out: *mut DynPixel,
                px: <BitDepth8 as BitDepth>::Pixel,
                n: c_int,
            );

            #[cfg(feature = "bitdepth_16")]
            fn dav1d_ipred_pixel_set_16bpc_neon(
                out: *mut DynPixel,
                px: <BitDepth16 as BitDepth>::Pixel,
                n: c_int,
            );
        }

        let out = out.as_mut_ptr().cast();
        match BD::BPC {
            BPC::BPC8 => {
                // Really a no-op cast, but it's difficult to do it properly with generics.
                let px = px.to::<u16>() as <BitDepth8 as BitDepth>::Pixel;
                // SAFETY: We're assuming the asm is actually correct and safe.
                unsafe { dav1d_ipred_pixel_set_8bpc_neon(out, px, n) }
            }
            BPC::BPC16 => {
                let px = px.into();
                // SAFETY: We're assuming the asm is actually correct and safe.
                unsafe { dav1d_ipred_pixel_set_16bpc_neon(out, px, n) }
            }
        }
    }

    fn ipred_z1_neon<BD: BitDepth>(
        dst: *mut BD::Pixel,
        stride: ptrdiff_t,
        topleft_in: &[BD::Pixel; SCRATCH_EDGE_LEN],
        topleft_off: usize,
        width: c_int,
        height: c_int,
        mut angle: c_int,
        _max_width: c_int,
        _max_height: c_int,
        bd: BD,
    ) {
        let topleft_in = &topleft_in[topleft_off..];
        let is_sm = (angle >> 9) & 1 != 0;
        let enable_intra_edge_filter = angle >> 10;
        angle &= 511;
        let mut dx = dav1d_dr_intra_derivative[(angle >> 1) as usize] as c_int;
        const TOP_OUT_SIZE: usize = 64 + 64 * (64 + 15) * 2 + 16;
        let mut top_out = [0.into(); TOP_OUT_SIZE];
        let max_base_x;
        let upsample_above = if enable_intra_edge_filter != 0 {
            get_upsample(width + height, 90 - angle, is_sm)
        } else {
            false
        };
        if upsample_above {
            bd_fn!(z1_upsample_edge::decl_fn, BD, ipred_z1_upsample_edge, neon).call(
                &mut top_out,
                width + height,
                topleft_in,
                width + cmp::min(width, height),
                bd,
            );
            max_base_x = 2 * (width + height) - 2;
            dx <<= 1;
        } else {
            let filter_strength = if enable_intra_edge_filter != 0 {
                get_filter_strength(width + height, 90 - angle, is_sm)
            } else {
                0
            };
            if filter_strength != 0 {
                bd_fn!(z1_filter_edge::decl_fn, BD, ipred_z1_filter_edge, neon).call::<BD>(
                    &mut top_out,
                    width + height,
                    topleft_in,
                    width + cmp::min(width, height),
                    filter_strength,
                );
                max_base_x = width + height - 1;
            } else {
                max_base_x = width + cmp::min(width, height) - 1;
                let len = max_base_x as usize + 1;
                top_out[..len].copy_from_slice(&topleft_in[1..][..len]);
            }
        }
        let base_inc = 1 + upsample_above as c_int;
        let pad_pixels = width + 15;
        let px = top_out[max_base_x as usize];
        rav1d_ipred_pixel_set_neon::<BD>(
            &mut top_out[max_base_x as usize + 1..],
            px,
            pad_pixels * base_inc,
        );
        if upsample_above {
            bd_fn!(z13_fill::decl_fn, BD, ipred_z1_fill2, neon)
                .call::<BD>(dst, stride, &top_out, width, height, dx, max_base_x);
        } else {
            bd_fn!(z13_fill::decl_fn, BD, ipred_z1_fill1, neon)
                .call::<BD>(dst, stride, &top_out, width, height, dx, max_base_x);
        };
    }

    fn ipred_z2_neon<BD: BitDepth>(
        dst: *mut BD::Pixel,
        stride: ptrdiff_t,
        topleft_in: &[BD::Pixel; SCRATCH_EDGE_LEN],
        topleft_off: usize,
        width: c_int,
        height: c_int,
        mut angle: c_int,
        max_width: c_int,
        max_height: c_int,
        bd: BD,
    ) {
        let topleft_in = &topleft_in[topleft_off..];
        let is_sm = (angle >> 9) & 1 != 0;
        let enable_intra_edge_filter = angle >> 10;
        angle &= 511;
        assert!(angle > 90 && angle < 180);
        let mut dy = dav1d_dr_intra_derivative[((angle - 90) >> 1) as usize] as c_int;
        let mut dx = dav1d_dr_intra_derivative[((180 - angle) >> 1) as usize] as c_int;
        let mut buf = [0.to::<BD::Pixel>(); 3 * (64 + 1)]; // NOTE: C code doesn't initialize

        // The asm can underread below the start of top[] and left[]; to avoid
        // surprising behaviour, make sure this is within the allocated stack space.
        let left_offset = 2 * (64 + 1);
        let top_offset = 1 * (64 + 1);
        let flipped_offset = 0 * (64 + 1);

        let upsample_left = if enable_intra_edge_filter != 0 {
            get_upsample(width + height, 180 - angle, is_sm)
        } else {
            false
        };
        let upsample_above = if enable_intra_edge_filter != 0 {
            get_upsample(width + height, angle - 90, is_sm)
        } else {
            false
        };

        if upsample_above {
            bd_fn!(z2_upsample_edge::decl_fn, BD, ipred_z2_upsample_edge, neon).call(
                &mut buf[top_offset..],
                width,
                topleft_in,
                bd,
            );
            dx <<= 1;
        } else {
            let filter_strength = if enable_intra_edge_filter != 0 {
                get_filter_strength(width + height, angle - 90, is_sm)
            } else {
                0
            };

            if filter_strength != 0 {
                bd_fn!(z1_filter_edge::decl_fn, BD, ipred_z1_filter_edge, neon).call::<BD>(
                    &mut buf[1 + top_offset..],
                    cmp::min(max_width, width),
                    topleft_in,
                    width,
                    filter_strength,
                );

                if max_width < width {
                    let len = (width - max_width) as usize;
                    buf[top_offset + 1 + max_width as usize..][..len]
                        .copy_from_slice(&topleft_in[1 + max_width as usize..][..len]);
                }
            } else {
                BD::pixel_copy(
                    &mut buf[1 + top_offset..],
                    &topleft_in[1..][..width as usize],
                    width as usize,
                );
            }
        }

        if upsample_left {
            buf[flipped_offset] = topleft_in[0];
            bd_fn!(reverse::decl_fn, BD, ipred_reverse, neon).call::<BD>(
                &mut buf[1 + flipped_offset..],
                topleft_in,
                height,
            );
            let (src, dst) = buf.split_at_mut(left_offset);
            bd_fn!(z2_upsample_edge::decl_fn, BD, ipred_z2_upsample_edge, neon).call(
                dst,
                height,
                &src[flipped_offset..],
                bd,
            );
            dy <<= 1;
        } else {
            let filter_strength = if enable_intra_edge_filter != 0 {
                get_filter_strength(width + height, 180 - angle, is_sm)
            } else {
                0
            };
            if filter_strength != 0 {
                buf[flipped_offset] = topleft_in[0];
                bd_fn!(reverse::decl_fn, BD, ipred_reverse, neon).call::<BD>(
                    &mut buf[1 + flipped_offset..],
                    topleft_in,
                    height,
                );
                let (src, dst) = buf.split_at_mut(1 + left_offset);
                bd_fn!(z1_filter_edge::decl_fn, BD, ipred_z1_filter_edge, neon).call::<BD>(
                    dst,
                    cmp::min(max_height, height),
                    &src[flipped_offset..],
                    height,
                    filter_strength,
                );
                if max_height < height {
                    let len = (height - max_height) as usize;
                    let (src, dst) = buf[1 + max_height as usize..].split_at_mut(left_offset);
                    dst[..len].copy_from_slice(&src[flipped_offset..][..len]);
                }
            } else {
                bd_fn!(reverse::decl_fn, BD, ipred_reverse, neon).call::<BD>(
                    &mut buf[left_offset + 1..],
                    topleft_in,
                    height,
                );
            }
        }
        buf[top_offset] = topleft_in[0];
        buf[left_offset] = topleft_in[0];

        assert!(!(upsample_above && upsample_left));

        if !upsample_above && !upsample_left {
            bd_fn!(z2_fill::decl_fn, BD, ipred_z2_fill1, neon).call::<BD>(
                dst,
                stride,
                &buf[top_offset..],
                &buf[left_offset..],
                width,
                height,
                dx,
                dy,
            );
        } else if upsample_above {
            bd_fn!(z2_fill::decl_fn, BD, ipred_z2_fill2, neon).call::<BD>(
                dst,
                stride,
                &buf[top_offset..],
                &buf[left_offset..],
                width,
                height,
                dx,
                dy,
            );
        } else {
            bd_fn!(z2_fill::decl_fn, BD, ipred_z2_fill3, neon).call::<BD>(
                dst,
                stride,
                &buf[top_offset..],
                &buf[left_offset..],
                width,
                height,
                dx,
                dy,
            );
        };
    }

    fn ipred_z3_neon<BD: BitDepth>(
        dst: *mut BD::Pixel,
        stride: ptrdiff_t,
        topleft_in: &[BD::Pixel; SCRATCH_EDGE_LEN],
        topleft_off: usize,
        width: c_int,
        height: c_int,
        mut angle: c_int,
        _max_width: c_int,
        _max_height: c_int,
        bd: BD,
    ) {
        let topleft_in = &topleft_in[topleft_off..];
        let is_sm = (angle >> 9) & 1 != 0;
        let enable_intra_edge_filter = angle >> 10;
        angle &= 511;
        assert!(angle > 180);
        let mut dy = dav1d_dr_intra_derivative[(270 - angle >> 1) as usize] as c_int;
        let mut flipped = [0.into(); 64 + 64 + 16];
        let mut left_out = [0.into(); 64 + 64 + (64 + 15) * 2];
        let max_base_y;
        let upsample_left = if enable_intra_edge_filter != 0 {
            get_upsample(width + height, angle - 180, is_sm)
        } else {
            false
        };
        if upsample_left {
            flipped[0] = topleft_in[0];
            bd_fn!(reverse::decl_fn, BD, ipred_reverse, neon).call::<BD>(
                &mut flipped[1..],
                topleft_in,
                height + cmp::max(width, height),
            );
            bd_fn!(z1_upsample_edge::decl_fn, BD, ipred_z1_upsample_edge, neon).call(
                &mut left_out,
                width + height,
                &flipped,
                height + cmp::min(width, height),
                bd,
            );
            max_base_y = 2 * (width + height) - 2;
            dy <<= 1;
        } else {
            let filter_strength = if enable_intra_edge_filter != 0 {
                get_filter_strength(width + height, angle - 180, is_sm)
            } else {
                0
            };
            if filter_strength != 0 {
                flipped[0] = topleft_in[0];
                bd_fn!(reverse::decl_fn, BD, ipred_reverse, neon).call::<BD>(
                    &mut flipped[1..],
                    topleft_in,
                    height + cmp::max(width, height),
                );
                bd_fn!(z1_filter_edge::decl_fn, BD, ipred_z1_filter_edge, neon).call::<BD>(
                    &mut left_out,
                    width + height,
                    &flipped,
                    height + cmp::min(width, height),
                    filter_strength,
                );
                max_base_y = width + height - 1;
            } else {
                bd_fn!(reverse::decl_fn, BD, ipred_reverse, neon).call::<BD>(
                    &mut left_out,
                    topleft_in,
                    height + cmp::min(width, height),
                );
                max_base_y = height + cmp::min(width, height) - 1;
            }
        }
        let base_inc = 1 + upsample_left as c_int;
        let pad_pixels = cmp::max(64 - max_base_y - 1, height + 15);
        let px = left_out[max_base_y as usize];
        rav1d_ipred_pixel_set_neon::<BD>(
            &mut left_out[max_base_y as usize + 1..],
            px,
            pad_pixels * base_inc,
        );
        if upsample_left {
            bd_fn!(z13_fill::decl_fn, BD, ipred_z3_fill2, neon)
                .call::<BD>(dst, stride, &left_out, width, height, dy, max_base_y);
        } else {
            bd_fn!(z13_fill::decl_fn, BD, ipred_z3_fill1, neon)
                .call::<BD>(dst, stride, &left_out, width, height, dy, max_base_y);
        };
    }

    /// # Safety
    ///
    /// Must be called from [`angular_ipred::Fn::call`].
    #[deny(unsafe_op_in_unsafe_fn)]
    pub unsafe extern "C" fn ipred_z_neon_erased<BD: BitDepth, const Z: usize>(
        dst: *mut DynPixel,
        stride: ptrdiff_t,
        topleft_in: *const DynPixel,
        width: c_int,
        height: c_int,
        angle: c_int,
        max_width: c_int,
        max_height: c_int,
        bitdepth_max: c_int,
        topleft_off: usize,
        _dst: FFISafeRav1dPictureDataComponentOffset,
    ) {
        let dst = dst.cast();
        // SAFETY: Reconstructed from args passed by `angular_ipred::Fn::call`.
        let topleft_in = unsafe { reconstruct_topleft::<BD>(topleft_in, topleft_off) };
        let bd = BD::from_c(bitdepth_max);
        [ipred_z1_neon, ipred_z2_neon, ipred_z3_neon][Z - 1](
            dst,
            stride,
            topleft_in,
            topleft_off,
            width,
            height,
            angle,
            max_width,
            max_height,
            bd,
        )
    }
}

impl Rav1dIntraPredDSPContext {
    pub const fn default<BD: BitDepth>() -> Self {
        Self {
            intra_pred: {
                let mut a = [DefaultValue::DEFAULT; N_IMPL_INTRA_PRED_MODES];
                a[DC_PRED as usize] =
                    angular_ipred::Fn::new(ipred_dc_c_erased::<BD, { DcGen::TopLeft as u8 }>);
                a[DC_128_PRED as usize] = angular_ipred::Fn::new(ipred_dc_128_c_erased::<BD>);
                a[TOP_DC_PRED as usize] =
                    angular_ipred::Fn::new(ipred_dc_c_erased::<BD, { DcGen::Top as u8 }>);
                a[LEFT_DC_PRED as usize] =
                    angular_ipred::Fn::new(ipred_dc_c_erased::<BD, { DcGen::Left as u8 }>);
                a[HOR_PRED as usize] = angular_ipred::Fn::new(ipred_h_c_erased::<BD>);
                a[VERT_PRED as usize] = angular_ipred::Fn::new(ipred_v_c_erased::<BD>);
                a[PAETH_PRED as usize] = angular_ipred::Fn::new(ipred_paeth_c_erased::<BD>);
                a[SMOOTH_PRED as usize] = angular_ipred::Fn::new(ipred_smooth_c_erased::<BD>);
                a[SMOOTH_V_PRED as usize] = angular_ipred::Fn::new(ipred_smooth_v_c_erased::<BD>);
                a[SMOOTH_H_PRED as usize] = angular_ipred::Fn::new(ipred_smooth_h_c_erased::<BD>);
                a[Z1_PRED as usize] = angular_ipred::Fn::new(ipred_z_c_erased::<BD, 1>);
                a[Z2_PRED as usize] = angular_ipred::Fn::new(ipred_z_c_erased::<BD, 2>);
                a[Z3_PRED as usize] = angular_ipred::Fn::new(ipred_z_c_erased::<BD, 3>);
                a[FILTER_PRED as usize] = angular_ipred::Fn::new(ipred_filter_c_erased::<BD>);
                a
            },
            cfl_ac: enum_map!(Rav1dPixelLayoutSubSampled => cfl_ac::Fn; match key {
                I420 => cfl_ac::Fn::new(cfl_ac_c_erased::<BD, true, true>),
                I422 => cfl_ac::Fn::new(cfl_ac_c_erased::<BD, true, false>),
                I444 => cfl_ac::Fn::new(cfl_ac_c_erased::<BD, false, false>),
            }),
            cfl_pred: {
                // Not all elements are initialized with fns,
                // so we default initialize first so that there is no uninitialized memory.
                // The defaults just call `unimplemented!()`,
                // which shouldn't slow down the other code paths at all.
                let mut a = [DefaultValue::DEFAULT; 6];
                a[DC_PRED as usize] =
                    cfl_pred::Fn::new(ipred_cfl_c_erased::<BD, { DcGen::TopLeft as u8 }>);
                a[DC_128_PRED as usize] = cfl_pred::Fn::new(ipred_cfl_128_c_erased::<BD>);
                a[TOP_DC_PRED as usize] =
                    cfl_pred::Fn::new(ipred_cfl_c_erased::<BD, { DcGen::Top as u8 }>);
                a[LEFT_DC_PRED as usize] =
                    cfl_pred::Fn::new(ipred_cfl_c_erased::<BD, { DcGen::Left as u8 }>);
                a
            },
            pal_pred: pal_pred::Fn::new(pal_pred_c_erased::<BD>),
        }
    }

    #[cfg(all(feature = "asm", any(target_arch = "x86", target_arch = "x86_64")))]
    #[inline(always)]
    const fn init_x86<BD: BitDepth>(mut self, flags: CpuFlags) -> Self {
        if !flags.contains(CpuFlags::SSSE3) {
            return self;
        }

        self.intra_pred[DC_PRED as usize] = bd_fn!(angular_ipred::decl_fn, BD, ipred_dc, ssse3);
        self.intra_pred[DC_128_PRED as usize] =
            bd_fn!(angular_ipred::decl_fn, BD, ipred_dc_128, ssse3);
        self.intra_pred[TOP_DC_PRED as usize] =
            bd_fn!(angular_ipred::decl_fn, BD, ipred_dc_top, ssse3);
        self.intra_pred[LEFT_DC_PRED as usize] =
            bd_fn!(angular_ipred::decl_fn, BD, ipred_dc_left, ssse3);
        self.intra_pred[HOR_PRED as usize] = bd_fn!(angular_ipred::decl_fn, BD, ipred_h, ssse3);
        self.intra_pred[VERT_PRED as usize] = bd_fn!(angular_ipred::decl_fn, BD, ipred_v, ssse3);
        self.intra_pred[PAETH_PRED as usize] =
            bd_fn!(angular_ipred::decl_fn, BD, ipred_paeth, ssse3);
        self.intra_pred[SMOOTH_PRED as usize] =
            bd_fn!(angular_ipred::decl_fn, BD, ipred_smooth, ssse3);
        self.intra_pred[SMOOTH_H_PRED as usize] =
            bd_fn!(angular_ipred::decl_fn, BD, ipred_smooth_h, ssse3);
        self.intra_pred[SMOOTH_V_PRED as usize] =
            bd_fn!(angular_ipred::decl_fn, BD, ipred_smooth_v, ssse3);
        self.intra_pred[Z1_PRED as usize] = bd_fn!(angular_ipred::decl_fn, BD, ipred_z1, ssse3);
        self.intra_pred[Z2_PRED as usize] = bd_fn!(angular_ipred::decl_fn, BD, ipred_z2, ssse3);
        self.intra_pred[Z3_PRED as usize] = bd_fn!(angular_ipred::decl_fn, BD, ipred_z3, ssse3);
        self.intra_pred[FILTER_PRED as usize] =
            bd_fn!(angular_ipred::decl_fn, BD, ipred_filter, ssse3);

        self.cfl_pred[DC_PRED as usize] = bd_fn!(cfl_pred::decl_fn, BD, ipred_cfl, ssse3);
        self.cfl_pred[DC_128_PRED as usize] = bd_fn!(cfl_pred::decl_fn, BD, ipred_cfl_128, ssse3);
        self.cfl_pred[TOP_DC_PRED as usize] = bd_fn!(cfl_pred::decl_fn, BD, ipred_cfl_top, ssse3);
        self.cfl_pred[LEFT_DC_PRED as usize] = bd_fn!(cfl_pred::decl_fn, BD, ipred_cfl_left, ssse3);

        self.cfl_ac = enum_map!(Rav1dPixelLayoutSubSampled => cfl_ac::Fn; match key {
            I420 => bd_fn!(cfl_ac::decl_fn, BD, ipred_cfl_ac_420, ssse3),
            I422 => bd_fn!(cfl_ac::decl_fn, BD, ipred_cfl_ac_422, ssse3),
            I444 => bd_fn!(cfl_ac::decl_fn, BD, ipred_cfl_ac_444, ssse3),
        });

        self.pal_pred = bd_fn!(pal_pred::decl_fn, BD, pal_pred, ssse3);

        #[cfg(target_arch = "x86_64")]
        {
            if !flags.contains(CpuFlags::AVX2) {
                return self;
            }

            self.intra_pred[DC_PRED as usize] = bd_fn!(angular_ipred::decl_fn, BD, ipred_dc, avx2);
            self.intra_pred[DC_128_PRED as usize] =
                bd_fn!(angular_ipred::decl_fn, BD, ipred_dc_128, avx2);
            self.intra_pred[TOP_DC_PRED as usize] =
                bd_fn!(angular_ipred::decl_fn, BD, ipred_dc_top, avx2);
            self.intra_pred[LEFT_DC_PRED as usize] =
                bd_fn!(angular_ipred::decl_fn, BD, ipred_dc_left, avx2);
            self.intra_pred[HOR_PRED as usize] = bd_fn!(angular_ipred::decl_fn, BD, ipred_h, avx2);
            self.intra_pred[VERT_PRED as usize] = bd_fn!(angular_ipred::decl_fn, BD, ipred_v, avx2);
            self.intra_pred[PAETH_PRED as usize] =
                bd_fn!(angular_ipred::decl_fn, BD, ipred_paeth, avx2);
            self.intra_pred[SMOOTH_PRED as usize] =
                bd_fn!(angular_ipred::decl_fn, BD, ipred_smooth, avx2);
            self.intra_pred[SMOOTH_H_PRED as usize] =
                bd_fn!(angular_ipred::decl_fn, BD, ipred_smooth_h, avx2);
            self.intra_pred[SMOOTH_V_PRED as usize] =
                bd_fn!(angular_ipred::decl_fn, BD, ipred_smooth_v, avx2);
            self.intra_pred[Z1_PRED as usize] = bd_fn!(angular_ipred::decl_fn, BD, ipred_z1, avx2);
            self.intra_pred[Z2_PRED as usize] = bd_fn!(angular_ipred::decl_fn, BD, ipred_z2, avx2);
            self.intra_pred[Z3_PRED as usize] = bd_fn!(angular_ipred::decl_fn, BD, ipred_z3, avx2);
            self.intra_pred[FILTER_PRED as usize] =
                bd_fn!(angular_ipred::decl_fn, BD, ipred_filter, avx2);

            self.cfl_pred[DC_PRED as usize] = bd_fn!(cfl_pred::decl_fn, BD, ipred_cfl, avx2);
            self.cfl_pred[DC_128_PRED as usize] =
                bd_fn!(cfl_pred::decl_fn, BD, ipred_cfl_128, avx2);
            self.cfl_pred[TOP_DC_PRED as usize] =
                bd_fn!(cfl_pred::decl_fn, BD, ipred_cfl_top, avx2);
            self.cfl_pred[LEFT_DC_PRED as usize] =
                bd_fn!(cfl_pred::decl_fn, BD, ipred_cfl_left, avx2);

            self.cfl_ac = enum_map!(Rav1dPixelLayoutSubSampled => cfl_ac::Fn; match key {
                I420 => bd_fn!(cfl_ac::decl_fn, BD, ipred_cfl_ac_420, avx2),
                I422 => bd_fn!(cfl_ac::decl_fn, BD, ipred_cfl_ac_422, avx2),
                I444 => bd_fn!(cfl_ac::decl_fn, BD, ipred_cfl_ac_444, avx2),
            });

            self.pal_pred = bd_fn!(pal_pred::decl_fn, BD, pal_pred, avx2);

            if !flags.contains(CpuFlags::AVX512ICL) {
                return self;
            }

            if let BPC::BPC8 = BD::BPC {
                self.intra_pred[DC_PRED as usize] =
                    bpc_fn!(angular_ipred::decl_fn, 8 bpc, ipred_dc, avx512icl);
                self.intra_pred[DC_128_PRED as usize] =
                    bpc_fn!(angular_ipred::decl_fn, 8 bpc, ipred_dc_128, avx512icl);
                self.intra_pred[TOP_DC_PRED as usize] =
                    bpc_fn!(angular_ipred::decl_fn, 8 bpc, ipred_dc_top, avx512icl);
                self.intra_pred[LEFT_DC_PRED as usize] =
                    bpc_fn!(angular_ipred::decl_fn, 8 bpc, ipred_dc_left, avx512icl);
                self.intra_pred[HOR_PRED as usize] =
                    bpc_fn!(angular_ipred::decl_fn, 8 bpc, ipred_h, avx512icl);
                self.intra_pred[VERT_PRED as usize] =
                    bpc_fn!(angular_ipred::decl_fn, 8 bpc, ipred_v, avx512icl);
                self.intra_pred[Z2_PRED as usize] =
                    bpc_fn!(angular_ipred::decl_fn, 8 bpc, ipred_z2, avx512icl);
            }

            self.intra_pred[PAETH_PRED as usize] =
                bd_fn!(angular_ipred::decl_fn, BD, ipred_paeth, avx512icl);
            self.intra_pred[SMOOTH_PRED as usize] =
                bd_fn!(angular_ipred::decl_fn, BD, ipred_smooth, avx512icl);
            self.intra_pred[SMOOTH_H_PRED as usize] =
                bd_fn!(angular_ipred::decl_fn, BD, ipred_smooth_h, avx512icl);
            self.intra_pred[SMOOTH_V_PRED as usize] =
                bd_fn!(angular_ipred::decl_fn, BD, ipred_smooth_v, avx512icl);
            self.intra_pred[Z1_PRED as usize] =
                bd_fn!(angular_ipred::decl_fn, BD, ipred_z1, avx512icl);
            self.intra_pred[Z2_PRED as usize] =
                bd_fn!(angular_ipred::decl_fn, BD, ipred_z2, avx512icl);
            self.intra_pred[Z3_PRED as usize] =
                bd_fn!(angular_ipred::decl_fn, BD, ipred_z3, avx512icl);
            self.intra_pred[FILTER_PRED as usize] =
                bd_fn!(angular_ipred::decl_fn, BD, ipred_filter, avx512icl);

            self.pal_pred = bd_fn!(pal_pred::decl_fn, BD, pal_pred, avx512icl);
        }

        self
    }

    #[cfg(all(feature = "asm", any(target_arch = "arm", target_arch = "aarch64")))]
    #[inline(always)]
    const fn init_arm<BD: BitDepth>(mut self, flags: CpuFlags) -> Self {
        if !flags.contains(CpuFlags::NEON) {
            return self;
        }

        self.intra_pred[DC_PRED as usize] = bd_fn!(angular_ipred::decl_fn, BD, ipred_dc, neon);
        self.intra_pred[DC_128_PRED as usize] =
            bd_fn!(angular_ipred::decl_fn, BD, ipred_dc_128, neon);
        self.intra_pred[TOP_DC_PRED as usize] =
            bd_fn!(angular_ipred::decl_fn, BD, ipred_dc_top, neon);
        self.intra_pred[LEFT_DC_PRED as usize] =
            bd_fn!(angular_ipred::decl_fn, BD, ipred_dc_left, neon);
        self.intra_pred[HOR_PRED as usize] = bd_fn!(angular_ipred::decl_fn, BD, ipred_h, neon);
        self.intra_pred[VERT_PRED as usize] = bd_fn!(angular_ipred::decl_fn, BD, ipred_v, neon);
        self.intra_pred[PAETH_PRED as usize] =
            bd_fn!(angular_ipred::decl_fn, BD, ipred_paeth, neon);
        self.intra_pred[SMOOTH_PRED as usize] =
            bd_fn!(angular_ipred::decl_fn, BD, ipred_smooth, neon);
        self.intra_pred[SMOOTH_V_PRED as usize] =
            bd_fn!(angular_ipred::decl_fn, BD, ipred_smooth_v, neon);
        self.intra_pred[SMOOTH_H_PRED as usize] =
            bd_fn!(angular_ipred::decl_fn, BD, ipred_smooth_h, neon);
        #[cfg(target_arch = "aarch64")]
        {
            use self::neon::ipred_z_neon_erased;

            self.intra_pred[Z1_PRED as usize] =
                angular_ipred::Fn::new(ipred_z_neon_erased::<BD, 1>);
            self.intra_pred[Z2_PRED as usize] =
                angular_ipred::Fn::new(ipred_z_neon_erased::<BD, 2>);
            self.intra_pred[Z3_PRED as usize] =
                angular_ipred::Fn::new(ipred_z_neon_erased::<BD, 3>);
        }
        self.intra_pred[FILTER_PRED as usize] =
            bd_fn!(angular_ipred::decl_fn, BD, ipred_filter, neon);

        self.cfl_pred[DC_PRED as usize] = bd_fn!(cfl_pred::decl_fn, BD, ipred_cfl, neon);
        self.cfl_pred[DC_128_PRED as usize] = bd_fn!(cfl_pred::decl_fn, BD, ipred_cfl_128, neon);
        self.cfl_pred[TOP_DC_PRED as usize] = bd_fn!(cfl_pred::decl_fn, BD, ipred_cfl_top, neon);
        self.cfl_pred[LEFT_DC_PRED as usize] = bd_fn!(cfl_pred::decl_fn, BD, ipred_cfl_left, neon);

        self.cfl_ac = enum_map!(Rav1dPixelLayoutSubSampled => cfl_ac::Fn; match key {
            I420 => bd_fn!(cfl_ac::decl_fn, BD, ipred_cfl_ac_420, neon),
            I422 => bd_fn!(cfl_ac::decl_fn, BD, ipred_cfl_ac_422, neon),
            I444 => bd_fn!(cfl_ac::decl_fn, BD, ipred_cfl_ac_444, neon),
        });

        self.pal_pred = bd_fn!(pal_pred::decl_fn, BD, pal_pred, neon);

        self
    }

    #[inline(always)]
    const fn init<BD: BitDepth>(self, flags: CpuFlags) -> Self {
        #[cfg(all(feature = "asm", not(rav1d_rust_simd = "ipred")))]
        {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                return self.init_x86::<BD>(flags);
            }
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            {
                return self.init_arm::<BD>(flags);
            }
        }

        #[allow(unreachable_code)] // Reachable on some #[cfg]s.
        self.init_rust_simd::<BD>(flags)
    }

    /// The Rust kernels; they stand in where there is no asm, or over it with
    /// `--cfg rav1d_rust_simd="ipred"`.
    #[inline(always)]
    const fn init_rust_simd<BD: BitDepth>(mut self, flags: CpuFlags) -> Self {
        use self::rust_simd::*;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if !flags.contains(CpuFlags::SSE41) {
            return self;
        }
        let _ = flags;
        if !crate::simd128::native() {
            return self;
        }
        let a = &mut self.intra_pred;
        a[DC_PRED as usize] = angular_ipred::Fn::new(ipred_dc_erased::<BD, { DcGen::TopLeft as u8 }>);
        a[TOP_DC_PRED as usize] = angular_ipred::Fn::new(ipred_dc_erased::<BD, { DcGen::Top as u8 }>);
        a[LEFT_DC_PRED as usize] = angular_ipred::Fn::new(ipred_dc_erased::<BD, { DcGen::Left as u8 }>);
        a[DC_128_PRED as usize] = angular_ipred::Fn::new(ipred_dc_erased::<BD, DC_128>);
        a[SMOOTH_PRED as usize] = angular_ipred::Fn::new(ipred_smooth_erased::<BD, SMOOTH>);
        a[SMOOTH_V_PRED as usize] = angular_ipred::Fn::new(ipred_smooth_erased::<BD, SMOOTH_V>);
        a[SMOOTH_H_PRED as usize] = angular_ipred::Fn::new(ipred_smooth_erased::<BD, SMOOTH_H>);
        a[Z1_PRED as usize] = angular_ipred::Fn::new(ipred_z1_erased::<BD>);
        a[Z2_PRED as usize] = angular_ipred::Fn::new(ipred_z2_erased::<BD>);
        a[Z3_PRED as usize] = angular_ipred::Fn::new(ipred_z3_erased::<BD>);
        let c = &mut self.cfl_pred;
        c[DC_PRED as usize] = cfl_pred::Fn::new(ipred_cfl_erased::<BD, { DcGen::TopLeft as u8 }>);
        c[TOP_DC_PRED as usize] = cfl_pred::Fn::new(ipred_cfl_erased::<BD, { DcGen::Top as u8 }>);
        c[LEFT_DC_PRED as usize] = cfl_pred::Fn::new(ipred_cfl_erased::<BD, { DcGen::Left as u8 }>);
        c[DC_128_PRED as usize] = cfl_pred::Fn::new(ipred_cfl_erased::<BD, DC_128>);
        self.cfl_ac = enum_map!(Rav1dPixelLayoutSubSampled => cfl_ac::Fn; match key {
            I420 => cfl_ac::Fn::new(cfl_ac_erased::<BD, true, true>),
            I422 => cfl_ac::Fn::new(cfl_ac_erased::<BD, true, false>),
            I444 => cfl_ac::Fn::new(cfl_ac_erased::<BD, false, false>),
        });
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
    use crate::include::common::bitdepth::{BitDepth16, BitDepth8};
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
    }

    const STRIDE: usize = 128;
    const ROWS: usize = 72;
    const SIZES: [(c_int, c_int); 19] = [
        (4, 4),
        (4, 8),
        (4, 16),
        (8, 4),
        (8, 8),
        (8, 16),
        (8, 32),
        (16, 4),
        (16, 8),
        (16, 16),
        (16, 32),
        (16, 64),
        (32, 8),
        (32, 16),
        (32, 32),
        (32, 64),
        (64, 16),
        (64, 32),
        (64, 64),
    ];
    const DCS: [u8; 4] = [DC_PRED, TOP_DC_PRED, LEFT_DC_PRED, DC_128_PRED];
    const SMOOTHS: [u8; 3] = [SMOOTH_PRED, SMOOTH_V_PRED, SMOOTH_H_PRED];
    const LAYOUTS: [Rav1dPixelLayoutSubSampled; 3] = [
        Rav1dPixelLayoutSubSampled::I420,
        Rav1dPixelLayoutSubSampled::I422,
        Rav1dPixelLayoutSubSampled::I444,
    ];

    #[repr(align(64))]
    struct Ac([i16; SCRATCH_AC_TXTP_LEN]);

    fn rust<BD: BitDepth>() -> Rav1dIntraPredDSPContext {
        rav1d_init_cpu();
        let flags = rav1d_get_cpu_flags();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        assert!(flags.contains(CpuFlags::SSE41), "the SIMD kernels need SSE4.1");
        Rav1dIntraPredDSPContext::default::<BD>().init_rust_simd::<BD>(flags)
    }

    fn px<BD: BitDepth>(v: u16) -> BD::Pixel {
        (v as c_int).as_::<BD::Pixel>()
    }

    fn plane<BD: BitDepth>(fill: &[u16]) -> AlignedVec64<BD::Pixel> {
        let mut buf = AlignedVec64::<BD::Pixel>::new();
        buf.resize(STRIDE * ROWS, px::<BD>(0));
        for (p, &v) in buf.as_mut_slice().iter_mut().zip(fill) {
            *p = px::<BD>(v);
        }
        buf
    }

    fn pixels<BD: BitDepth>(edge: &[u16; SCRATCH_EDGE_LEN]) -> [BD::Pixel; SCRATCH_EDGE_LEN] {
        edge.map(px::<BD>)
    }

    /// The plane's bytes after `f` predicts a `w`x`h` block at (16, 4) from `edge`, whose
    /// top-left corner is at the middle.
    fn run<BD: BitDepth>(
        f: &angular_ipred::Fn,
        edge: &[u16; SCRATCH_EDGE_LEN],
        w: c_int,
        h: c_int,
        max: c_int,
    ) -> Vec<u8> {
        run_angle::<BD>(f, edge, w, h, 0, max)
    }

    fn run_angle<BD: BitDepth>(
        f: &angular_ipred::Fn,
        edge: &[u16; SCRATCH_EDGE_LEN],
        w: c_int,
        h: c_int,
        angle: c_int,
        max: c_int,
    ) -> Vec<u8> {
        let mut buf = plane::<BD>(&[]);
        let comp = Rav1dPictureDataComponent::wrap_buf::<BD>(buf.as_mut_slice(), STRIDE);
        let dst = WithOffset {
            data: &comp,
            offset: 4 * STRIDE + 16,
        };
        let off = SCRATCH_EDGE_LEN / 2;
        f.call::<BD>(dst, &pixels::<BD>(edge), off, w, h, angle, w, h, BD::from_c(max));
        drop(comp);
        buf.as_slice().as_bytes().to_vec()
    }

    fn run_cfl<BD: BitDepth>(
        f: &cfl_pred::Fn,
        edge: &[u16; SCRATCH_EDGE_LEN],
        w: c_int,
        h: c_int,
        ac: &Ac,
        alpha: c_int,
        max: c_int,
    ) -> Vec<u8> {
        let mut buf = plane::<BD>(&[]);
        let comp = Rav1dPictureDataComponent::wrap_buf::<BD>(buf.as_mut_slice(), STRIDE);
        let dst = WithOffset {
            data: &comp,
            offset: 4 * STRIDE + 16,
        };
        let off = SCRATCH_EDGE_LEN / 2;
        f.call::<BD>(dst, &pixels::<BD>(edge), off, w, h, &ac.0, alpha, BD::from_c(max));
        drop(comp);
        buf.as_slice().as_bytes().to_vec()
    }

    /// The AC of a `cw`x`ch` chroma block whose luma is at (16, 4) of `luma`.
    fn run_ac<BD: BitDepth>(
        f: &cfl_ac::Fn,
        luma: &[u16],
        pad: (c_int, c_int),
        cw: c_int,
        ch: c_int,
    ) -> Vec<i16> {
        let mut buf = plane::<BD>(luma);
        let comp = Rav1dPictureDataComponent::wrap_buf::<BD>(buf.as_mut_slice(), STRIDE);
        let y = WithOffset {
            data: &comp,
            offset: 4 * STRIDE + 16,
        };
        let mut ac = Ac([0; SCRATCH_AC_TXTP_LEN]);
        f.call::<BD>(&mut ac.0, y, pad.0, pad.1, cw, ch);
        ac.0[..(cw * ch) as usize].to_vec()
    }

    /// Runs of near values around a base, one pixel in 8 anywhere up to `max`.
    fn edge(rng: &mut Rng, max: c_int) -> [u16; SCRATCH_EDGE_LEN] {
        let max = max as u32;
        let base = rng.next() % (max + 1);
        let spread = (max + 1) / 64 + 4;
        std::array::from_fn(|_| {
            if rng.next() % 8 == 0 {
                (rng.next() % (max + 1)) as u16
            } else {
                let d = (rng.next() % (2 * spread + 1)) as i64 - spread as i64;
                (base as i64 + d).clamp(0, max as i64) as u16
            }
        })
    }

    /// Each test at 8, 10 and 12 bpc.
    macro_rules! at_bitdepths {
        ($test:ident: $($name:ident $BD:ident $max:literal),*) => {
            $(
                #[test]
                fn $name() {
                    $test::<$BD>($max);
                }
            )*
        };
    }

    at_bitdepths!(dc_and_smooth: dc_and_smooth_match_scalar_8 BitDepth8 255,
        dc_and_smooth_match_scalar_10 BitDepth16 1023, dc_and_smooth_match_scalar_12 BitDepth16 4095);

    fn dc_and_smooth<BD: BitDepth>(max: c_int) {
        let scalar = Rav1dIntraPredDSPContext::default::<BD>();
        let rust = rust::<BD>();
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let m = max as u16;
        for round in 0..60 {
            let edge = match round {
                0 => [m; SCRATCH_EDGE_LEN],
                1 => [0; SCRATCH_EDGE_LEN],
                2 => std::array::from_fn(|i| if i % 2 == 0 { m } else { 0 }),
                _ => edge(&mut rng, max),
            };
            for (w, h) in SIZES {
                for mode in DCS.into_iter().chain(SMOOTHS) {
                    let want = run::<BD>(&scalar.intra_pred[mode as usize], &edge, w, h, max);
                    let got = run::<BD>(&rust.intra_pred[mode as usize], &edge, w, h, max);
                    assert_eq!(got, want, "{w}x{h} mode {mode} round {round}");
                }
            }
        }
    }

    /// The directional angles: the eight base angles and three steps of 3 degrees either side.
    const ANGLES: [c_int; 56] = {
        let base = [45, 67, 90, 113, 135, 157, 180, 203];
        let mut a = [0; 56];
        let mut i = 0;
        while i < 56 {
            a[i] = base[i / 7] + 3 * (i % 7) as c_int - 9;
            i += 1;
        }
        a
    };

    at_bitdepths!(directional: directional_match_scalar_8 BitDepth8 255,
        directional_match_scalar_10 BitDepth16 1023, directional_match_scalar_12 BitDepth16 4095);

    /// Every angle Z1, Z2 and Z3 take, with and without the edge filter and the smooth flag.
    fn directional<BD: BitDepth>(max: c_int) {
        let scalar = Rav1dIntraPredDSPContext::default::<BD>();
        let rust = rust::<BD>();
        let mut rng = Rng(0x5851_f42d_4c95_7f2d);
        for round in 0..6 {
            let edge = match round {
                0 => std::array::from_fn(|i| if i % 2 == 0 { max as u16 } else { 0 }),
                _ => edge(&mut rng, max),
            };
            for (w, h) in SIZES {
                for angle in ANGLES {
                    let mode = match angle {
                        ..=89 => Z1_PRED,
                        91..=179 => Z2_PRED,
                        181.. => Z3_PRED,
                        _ => continue,
                    };
                    for flags in [0, 1 << 9, 1 << 10, 3 << 9] {
                        let a = angle | flags;
                        let f = &scalar.intra_pred[mode as usize];
                        let want = run_angle::<BD>(f, &edge, w, h, a, max);
                        let got = run_angle::<BD>(&rust.intra_pred[mode as usize], &edge, w, h, a, max);
                        assert_eq!(got, want, "{w}x{h} mode {mode} angle {a:#x} round {round}");
                    }
                }
            }
        }
    }

    at_bitdepths!(cfl_pred: cfl_pred_matches_scalar_8 BitDepth8 255,
        cfl_pred_matches_scalar_10 BitDepth16 1023, cfl_pred_matches_scalar_12 BitDepth16 4095);

    /// The AC within `8 * max`, as the luma scaled to eight times a pixel gives it.
    fn cfl_pred<BD: BitDepth>(max: c_int) {
        let scalar = Rav1dIntraPredDSPContext::default::<BD>();
        let rust = rust::<BD>();
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let mut ac = Ac([0; SCRATCH_AC_TXTP_LEN]);
        let lim = 8 * max as i16;
        for round in 0..40 {
            let edge = edge(&mut rng, max);
            for a in ac.0.iter_mut() {
                *a = match round {
                    0 => lim,
                    1 => -lim,
                    _ => ((rng.next() % (2 * lim as u32 + 1)) as i32 - lim as i32) as i16,
                };
            }
            for (w, h) in SIZES.into_iter().filter(|&(w, h)| w <= 32 && h <= 32) {
                for mode in DCS {
                    let alpha = match round {
                        0 | 1 => [16, -16][rng.next() as usize % 2],
                        _ => (rng.next() % 33) as c_int - 16,
                    };
                    let f = &scalar.cfl_pred[mode as usize];
                    let want = run_cfl::<BD>(f, &edge, w, h, &ac, alpha, max);
                    let got = run_cfl::<BD>(&rust.cfl_pred[mode as usize], &edge, w, h, &ac, alpha, max);
                    assert_eq!(got, want, "{w}x{h} mode {mode} alpha {alpha} round {round}");
                }
            }
        }
    }

    at_bitdepths!(cfl_ac: cfl_ac_matches_scalar_8 BitDepth8 255,
        cfl_ac_matches_scalar_10 BitDepth16 1023, cfl_ac_matches_scalar_12 BitDepth16 4095);

    fn cfl_ac<BD: BitDepth>(max: c_int) {
        let scalar = Rav1dIntraPredDSPContext::default::<BD>();
        let rust = rust::<BD>();
        let mut rng = Rng(0x1234_5678_9abc_def1);
        for round in 0..12 {
            let luma: Vec<u16> = match round {
                0 => vec![max as u16; STRIDE * ROWS],
                _ => (0..STRIDE * ROWS).map(|i| edge(&mut rng, max)[i % 7]).collect(),
            };
            for (l, layout) in LAYOUTS.into_iter().enumerate() {
                for cw in [4, 8, 16, 32] {
                    for ch in [4, 8, 16, 32] {
                        for pad in [(0, 0), (cw / 4 - 1, 0), (0, ch / 4 - 1), (cw / 8, ch / 8)] {
                            let want = run_ac::<BD>(&scalar.cfl_ac[layout], &luma, pad, cw, ch);
                            let got = run_ac::<BD>(&rust.cfl_ac[layout], &luma, pad, cw, ch);
                            assert_eq!(got, want, "layout {l} {cw}x{ch} pad {pad:?} round {round}");
                        }
                    }
                }
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

    /// ns per call of the ported predictions, scalar, Rust and asm.
    #[test]
    #[ignore]
    fn time_ipred() {
        use std::hint::black_box;

        rav1d_init_cpu();
        let flags = rav1d_get_cpu_flags();
        #[allow(unused_mut)]
        let mut dsps = vec![
            ("scalar", Rav1dIntraPredDSPContext::default::<BitDepth8>()),
            ("rust", rust::<BitDepth8>()),
        ];
        #[cfg(all(feature = "asm", any(target_arch = "x86", target_arch = "x86_64")))]
        dsps.push((
            "asm",
            Rav1dIntraPredDSPContext::default::<BitDepth8>().init_x86::<BitDepth8>(flags),
        ));
        #[cfg(all(feature = "asm", target_arch = "aarch64"))]
        dsps.push((
            "asm",
            Rav1dIntraPredDSPContext::default::<BitDepth8>().init_arm::<BitDepth8>(flags),
        ));
        let _ = flags;
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        // The asm loads the edge aligned, as the decoder's scratch is.
        #[repr(align(64))]
        struct Edge([u8; SCRATCH_EDGE_LEN]);
        let edge = Edge(pixels::<BitDepth8>(&edge(&mut rng, 255)));
        let edge = &edge.0;
        let mut ac = Ac([0; SCRATCH_AC_TXTP_LEN]);
        for a in ac.0.iter_mut() {
            *a = (rng.next() % 257) as i16 - 128;
        }
        let luma: Vec<u16> = (0..STRIDE * ROWS).map(|_| rng.next() as u8 as u16).collect();
        let mut buf = plane::<BitDepth8>(&luma);
        let comp = Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(buf.as_mut_slice(), STRIDE);
        let at = WithOffset {
            data: &comp,
            offset: 4 * STRIDE + 64,
        };
        let off = SCRATCH_EDGE_LEN / 2;
        let bd = BitDepth8::from_c(255);
        let line = |name: &str, w: c_int, h: c_int, call: &dyn Fn(&Rav1dIntraPredDSPContext)| {
            let mut line = format!("{name} {w}x{h}:");
            for (dsp_name, dsp) in &dsps {
                let best = ns(40000 / (w * h) as usize, || call(dsp));
                line += &format!("  {dsp_name} {best:6.1} ns");
            }
            println!("{line}");
        };
        for (name, mode) in [
            ("dc", DC_PRED),
            ("smooth", SMOOTH_PRED),
            ("smooth_v", SMOOTH_V_PRED),
            ("smooth_h", SMOOTH_H_PRED),
        ] {
            for (w, h) in [(4, 4), (8, 8), (16, 8), (16, 16), (32, 32), (64, 64)] {
                line(name, w, h, &|dsp| {
                    let f = &dsp.intra_pred[mode as usize];
                    f.call::<BitDepth8>(at, black_box(edge), off, w, h, 0, w, h, bd);
                });
            }
        }
        for (name, mode, angle) in [
            ("z1", Z1_PRED, 45 | 1 << 10),
            ("z2", Z2_PRED, 135 | 1 << 10),
            ("z3", Z3_PRED, 225 | 1 << 10),
            ("z1 unfiltered", Z1_PRED, 45),
            ("z2 unfiltered", Z2_PRED, 135),
            ("z3 unfiltered", Z3_PRED, 225),
        ] {
            for (w, h) in [(4, 4), (8, 8), (16, 8), (16, 16), (32, 32), (64, 64)] {
                line(name, w, h, &|dsp| {
                    let f = &dsp.intra_pred[mode as usize];
                    f.call::<BitDepth8>(at, black_box(edge), off, w, h, angle, w, h, bd);
                });
            }
        }
        for (w, h) in [(4, 4), (8, 8), (16, 8), (16, 16), (32, 32)] {
            line("cfl", w, h, &|dsp| {
                let f = &dsp.cfl_pred[DC_PRED as usize];
                f.call::<BitDepth8>(at, black_box(edge), off, w, h, black_box(&ac.0), 7, bd);
            });
        }
        // CfL takes luma blocks up to 32x32, so 4:2:0 chroma up to 16x16.
        for (name, layout, max) in [
            ("cfl_ac 420", Rav1dPixelLayoutSubSampled::I420, 16),
            ("cfl_ac 444", Rav1dPixelLayoutSubSampled::I444, 32),
        ] {
            for (w, h) in [(4, 4), (8, 8), (16, 16), (32, 32)].into_iter().filter(|&(w, _)| w <= max) {
                line(name, w, h, &|dsp| {
                    let mut ac = Ac([0; SCRATCH_AC_TXTP_LEN]);
                    dsp.cfl_ac[layout].call::<BitDepth8>(&mut ac.0, at, 0, 0, w, h);
                    black_box(&ac);
                });
            }
        }
    }
}
