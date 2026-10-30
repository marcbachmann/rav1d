#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::{c_int, c_uint};
use std::hint::assert_unchecked;
use std::ops::{Add, Shl, Shr};
use std::{cmp, mem, ptr};

use crate::c_types::{intptr_t, ptrdiff_t};
use to_method::To;

use crate::cpu::CpuFlags;
use crate::enum_map::{enum_map, enum_map_ty, DefaultValue};
use crate::ffi_safe::FFISafe;
use crate::include::common::bitdepth::{AsPrimitive, BitDepth, DynEntry, DynPixel, DynScaling};
use crate::include::common::intops::iclip;
use crate::include::dav1d::headers::{
    Dav1dFilmGrainData, Rav1dFilmGrainData, Rav1dPixelLayoutSubSampled,
};
use crate::include::dav1d::picture::{
    FFISafeRav1dPictureDataComponentOffset, Rav1dPictureDataComponent,
    Rav1dPictureDataComponentOffset,
};
use crate::internal::GrainLut;
use crate::strided::Strided as _;
use crate::tables::DAV1D_GAUSSIAN_SEQUENCE;
use crate::wrap_fn_ptr::wrap_fn_ptr;

pub const GRAIN_WIDTH: usize = 82;
pub const GRAIN_HEIGHT: usize = 73;

pub const FG_BLOCK_SIZE: usize = 32;

const SUB_GRAIN_WIDTH: usize = 44;
const SUB_GRAIN_HEIGHT: usize = 38;

wrap_fn_ptr!(pub unsafe extern "C" fn generate_grain_y(
    buf: *mut GrainLut<DynEntry>,
    data: &Dav1dFilmGrainData,
    bitdepth_max: c_int,
) -> ());

impl generate_grain_y::Fn {
    pub fn call<BD: BitDepth>(
        &self,
        buf: &mut GrainLut<BD::Entry>,
        data: &Rav1dFilmGrainData,
        bd: BD,
    ) {
        let buf = ptr::from_mut(buf).cast();
        let data = &data.clone().into();
        let bd = bd.into_c();
        // SAFETY: Fallback `fn generate_grain_y_rust` is safe; asm is supposed to do the same.
        unsafe { self.get()(buf, data, bd) }
    }
}

wrap_fn_ptr!(pub unsafe extern "C" fn generate_grain_uv(
    buf: *mut GrainLut<DynEntry>,
    buf_y: *const GrainLut<DynEntry>,
    data: &Dav1dFilmGrainData,
    uv: intptr_t,
    bitdepth_max: c_int,
) -> ());

impl generate_grain_uv::Fn {
    pub fn call<BD: BitDepth>(
        &self,
        buf: &mut GrainLut<BD::Entry>,
        buf_y: &GrainLut<BD::Entry>,
        data: &Rav1dFilmGrainData,
        is_uv: bool,
        bd: BD,
    ) {
        let buf = ptr::from_mut(buf).cast();
        let buf_y = ptr::from_ref(buf_y).cast();
        let data = &data.clone().into();
        let uv = is_uv.into();
        let bd = bd.into_c();
        // SAFETY: Fallback `fn generate_grain_uv_rust` is safe; asm is supposed to do the same.
        unsafe { self.get()(buf, buf_y, data, uv, bd) }
    }
}

wrap_fn_ptr!(pub unsafe extern "C" fn fgy_32x32xn(
    dst_row_ptr: *mut DynPixel,
    src_row_ptr: *const DynPixel,
    stride: ptrdiff_t,
    data: &Dav1dFilmGrainData,
    pw: usize,
    scaling: *const DynScaling,
    grain_lut: *const GrainLut<DynEntry>,
    bh: c_int,
    row_num: c_int,
    bitdepth_max: c_int,
    _dst_row: FFISafeRav1dPictureDataComponentOffset,
    _src_src: FFISafeRav1dPictureDataComponentOffset,
) -> ());

impl fgy_32x32xn::Fn {
    pub fn call<BD: BitDepth>(
        &self,
        dst: &Rav1dPictureDataComponent,
        src: &Rav1dPictureDataComponent,
        data: &Rav1dFilmGrainData,
        pw: usize,
        scaling: &BD::Scaling,
        grain_lut: &GrainLut<BD::Entry>,
        bh: usize,
        row_num: usize,
        bd: BD,
    ) {
        let row_strides = (row_num * FG_BLOCK_SIZE) as isize;
        let dst_row = dst.with_offset::<BD>() + row_strides * dst.pixel_stride::<BD>();
        let src_row = src.with_offset::<BD>() + row_strides * src.pixel_stride::<BD>();
        let dst_row_ptr = dst_row.as_mut_ptr::<BD>().cast();
        let src_row_ptr = src_row.as_ptr::<BD>().cast();
        let stride = dst.stride();
        let data = &data.clone().into();
        let scaling = ptr::from_ref(scaling).cast();
        let grain_lut = ptr::from_ref(grain_lut).cast();
        let bh = bh as c_int;
        let row_num = row_num as c_int;
        let bd = bd.into_c();
        let dst_row = dst_row.into_ffi_safe();
        let src_row = src_row.into_ffi_safe();
        // SAFETY: Fallback `fn fgy_32x32xn_rust` is safe; asm is supposed to do the same.
        unsafe {
            self.get()(
                dst_row_ptr,
                src_row_ptr,
                stride,
                data,
                pw,
                scaling,
                grain_lut,
                bh,
                row_num,
                bd,
                dst_row,
                src_row,
            )
        }
    }
}

wrap_fn_ptr!(pub unsafe extern "C" fn fguv_32x32xn(
    dst_row_ptr: *mut DynPixel,
    src_row_ptr: *const DynPixel,
    stride: ptrdiff_t,
    data: &Dav1dFilmGrainData,
    pw: usize,
    scaling: *const DynScaling,
    grain_lut: *const GrainLut<DynEntry>,
    bh: c_int,
    row_num: c_int,
    luma_row_ptr: *const DynPixel,
    luma_stride: ptrdiff_t,
    uv_pl: c_int,
    is_id: c_int,
    bitdepth_max: c_int,
    _dst_row: FFISafeRav1dPictureDataComponentOffset,
    _src_row: FFISafeRav1dPictureDataComponentOffset,
    _luma_row: FFISafeRav1dPictureDataComponentOffset,
) -> ());

impl fguv_32x32xn::Fn {
    pub fn call<BD: BitDepth>(
        &self,
        layout: Rav1dPixelLayoutSubSampled,
        dst: &Rav1dPictureDataComponent,
        src: &Rav1dPictureDataComponent,
        data: &Rav1dFilmGrainData,
        pw: usize,
        scaling: &BD::Scaling,
        grain_lut: &GrainLut<BD::Entry>,
        bh: usize,
        row_num: usize,
        luma: &Rav1dPictureDataComponent,
        is_uv: bool,
        is_id: bool,
        bd: BD,
    ) {
        let ss_y = (layout == Rav1dPixelLayoutSubSampled::I420) as usize;
        let row_strides = (row_num * FG_BLOCK_SIZE) as isize;
        let dst_row = dst.with_offset::<BD>() + (row_strides * dst.pixel_stride::<BD>() >> ss_y);
        let src_row = src.with_offset::<BD>() + (row_strides * src.pixel_stride::<BD>() >> ss_y);
        let dst_row_ptr = dst_row.as_mut_ptr::<BD>().cast();
        let src_row_ptr = src_row.as_ptr::<BD>().cast();
        let stride = dst.stride();
        let data = &data.clone().into();
        let scaling = (scaling as *const BD::Scaling).cast();
        let grain_lut = (grain_lut as *const GrainLut<BD::Entry>).cast();
        let bh = bh as c_int;
        let row_num = row_num as c_int;
        let luma_row = luma.with_offset::<BD>() + (row_strides * luma.pixel_stride::<BD>());
        let luma_row_ptr = luma_row.as_ptr::<BD>().cast();
        let luma_stride = luma.stride();
        let uv_pl = is_uv as c_int;
        let is_id = is_id as c_int;
        let bd = bd.into_c();
        let dst_row = dst_row.into_ffi_safe();
        let src_row = src_row.into_ffi_safe();
        let luma_row = luma_row.into_ffi_safe();
        // SAFETY: Fallback `fn fguv_32x32xn_rust` is safe; asm is supposed to do the same.
        unsafe {
            self.get()(
                dst_row_ptr,
                src_row_ptr,
                stride,
                data,
                pw,
                scaling,
                grain_lut,
                bh,
                row_num,
                luma_row_ptr,
                luma_stride,
                uv_pl,
                is_id,
                bd,
                dst_row,
                src_row,
                luma_row,
            )
        }
    }
}

pub struct Rav1dFilmGrainDSPContext {
    pub generate_grain_y: generate_grain_y::Fn,
    pub generate_grain_uv: enum_map_ty!(Rav1dPixelLayoutSubSampled, generate_grain_uv::Fn),
    pub fgy_32x32xn: fgy_32x32xn::Fn,
    pub fguv_32x32xn: enum_map_ty!(Rav1dPixelLayoutSubSampled, fguv_32x32xn::Fn),
}

#[inline]
fn get_random_number(bits: u8, state: &mut c_uint) -> c_int {
    let r = *state;
    let bit = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
    *state = (r >> 1) | bit << 15;

    (*state >> (16 - bits) & ((1 << bits) - 1)) as c_int
}

#[inline]
fn round2<T>(x: T, shift: u8) -> T
where
    T: Add<Output = T> + From<u8> + Shl<u8, Output = T> + Shr<u8, Output = T>,
{
    (x + (T::from(1) << shift >> 1)) >> shift
}

/// For the returned `seed: [c_uint; 2]` array,
/// `seed[0]` contains the current row, and
/// `seed[1]` contains the previous row.
fn row_seed(rows: usize, row_num: usize, data: &Rav1dFilmGrainData) -> [c_uint; 2] {
    let mut seed = [0; 2];
    for (i, seed) in seed.iter_mut().enumerate().take(rows) {
        *seed = data.seed;
        *seed ^= ((((row_num - i) * 37 + 178) & 0xFF) << 8) as c_uint;
        *seed ^= (((row_num - i) * 173 + 105) & 0xFF) as c_uint;
    }
    seed
}

/// # Safety
///
/// Must be called by [`generate_grain_y::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn generate_grain_y_c_erased<BD: BitDepth>(
    buf: *mut GrainLut<DynEntry>,
    data: &Dav1dFilmGrainData,
    bitdepth_max: c_int,
) {
    // SAFETY: Casting back to the original type from the `generate_grain_y::Fn::call`.
    let buf = unsafe { &mut *buf.cast() };
    let data = &data.clone().into();
    let bd = BD::from_c(bitdepth_max);
    generate_grain_y_rust(buf, data, bd)
}

const AR_PAD: usize = 3;

fn generate_grain_y_rust<BD: BitDepth>(
    buf: &mut GrainLut<BD::Entry>,
    data: &Rav1dFilmGrainData,
    bd: BD,
) {
    let bitdepth_min_8 = bd.bitdepth() - 8;
    let mut seed = data.seed;
    let shift = 4 - bitdepth_min_8 + data.grain_scale_shift;
    let grain_ctr = 128 << bitdepth_min_8;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;

    for row in &mut buf[..GRAIN_HEIGHT] {
        row[..GRAIN_WIDTH].fill_with(|| {
            let value = get_random_number(11, &mut seed);
            round2(DAV1D_GAUSSIAN_SEQUENCE[value as usize], shift).as_::<BD::Entry>()
        });
    }

    // `ar_lag` is 2 bits; this tells the compiler it definitely is.
    // That also means `ar_lag <= ar_pad`.
    let ar_lag = data.ar_coeff_lag as usize & ((1 << 2) - 1);

    for y in 0..GRAIN_HEIGHT - AR_PAD {
        for x in 0..GRAIN_WIDTH - 2 * AR_PAD {
            let mut coeff = &data.ar_coeffs_y[..];
            let mut sum = 0;
            for (dy, buf_row) in buf[y..][AR_PAD - ar_lag..=AR_PAD].iter().enumerate() {
                for (dx, &buf_val) in buf_row[x..][AR_PAD - ar_lag..=AR_PAD + ar_lag]
                    .iter()
                    .enumerate()
                {
                    if dx == ar_lag && dy == ar_lag {
                        break;
                    }
                    sum += coeff[0] as c_int * buf_val.as_::<c_int>();
                    coeff = &coeff[1..];
                }
            }

            let buf_yx = &mut buf[y + AR_PAD][x + AR_PAD];
            let grain = (*buf_yx).as_::<c_int>() + round2(sum, data.ar_coeff_shift);
            *buf_yx = iclip(grain, grain_min, grain_max).as_::<BD::Entry>();
        }
    }
}

fn generate_grain_uv_rust<BD: BitDepth>(
    buf: &mut GrainLut<BD::Entry>,
    buf_y: &GrainLut<BD::Entry>,
    data: &Rav1dFilmGrainData,
    is_uv: bool,
    is_subx: bool,
    is_suby: bool,
    bd: BD,
) {
    let uv = is_uv as usize;

    struct IsSub {
        y: bool,
        x: bool,
    }

    impl IsSub {
        const fn chroma(&self) -> (usize, usize) {
            let h = if self.y {
                SUB_GRAIN_HEIGHT
            } else {
                GRAIN_HEIGHT
            };
            let w = if self.x { SUB_GRAIN_WIDTH } else { GRAIN_WIDTH };
            (h, w)
        }

        const fn len(&self) -> (usize, usize) {
            let (h, w) = self.chroma();
            (h - AR_PAD, w - 2 * AR_PAD)
        }

        const fn luma(&self, (y, x): (usize, usize)) -> (usize, usize) {
            (
                (y << self.y as usize) + AR_PAD,
                (x << self.x as usize) + AR_PAD,
            )
        }

        const fn buf_index(&self, (y, x): (usize, usize)) -> (usize, usize) {
            let (y, x) = self.luma((y, x));
            (y + self.y as usize, x + self.x as usize)
        }

        const fn max_buf_index(&self) -> (usize, usize) {
            let (y, x) = self.len();
            self.buf_index((y - 1, x - 1))
        }

        const fn check_buf_index<T, const Y: usize, const X: usize>(
            &self,
            _: &Option<[[T; X]; Y]>,
        ) {
            let (y, x) = self.max_buf_index();
            assert!(y < Y);
            assert!(x < X);
        }

        #[allow(dead_code)] // False positive; used in a `const`.
        const fn check_buf_index_all<T, const Y: usize, const X: usize>(buf: &Option<[[T; X]; Y]>) {
            Self { y: true, x: true }.check_buf_index(buf);
            Self { y: true, x: false }.check_buf_index(buf);
            Self { y: false, x: true }.check_buf_index(buf);
            Self { y: false, x: false }.check_buf_index(buf);
        }
    }

    let is_sub = IsSub {
        y: is_suby,
        x: is_subx,
    };

    let bitdepth_min_8 = bd.bitdepth() - 8;
    let mut seed = data.seed ^ if is_uv { 0x49d8 } else { 0xb524 };
    let shift = 4 - bitdepth_min_8 + data.grain_scale_shift;
    let grain_ctr = 128 << bitdepth_min_8;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;

    for row in &mut buf[..is_sub.chroma().0] {
        row[..is_sub.chroma().1].fill_with(|| {
            let value = get_random_number(11, &mut seed);
            round2(DAV1D_GAUSSIAN_SEQUENCE[value as usize], shift).as_::<BD::Entry>()
        });
    }

    // `ar_lag` is 2 bits; this tells the compiler it definitely is.
    // That also means `ar_lag <= ar_pad`.
    let ar_lag = data.ar_coeff_lag as usize & ((1 << 2) - 1);

    for y in 0..is_sub.len().0 {
        for x in 0..is_sub.len().1 {
            let mut coeff = &data.ar_coeffs_uv[uv][..];
            let mut sum = 0;
            for (dy, buf_row) in buf[y..][AR_PAD - ar_lag..=AR_PAD].iter().enumerate() {
                for (dx, &buf_val) in buf_row[x..][AR_PAD - ar_lag..=AR_PAD + ar_lag]
                    .iter()
                    .enumerate()
                {
                    if dx == ar_lag && dy == ar_lag {
                        let mut luma = 0;
                        let (luma_y, luma_x) = is_sub.luma((y, x));
                        const _: () = IsSub::check_buf_index_all(&None::<GrainLut<()>>);
                        // The optimizer is not smart enough to deduce this on its own.
                        // SAFETY: The above static check checks all maximum index possibilities.
                        unsafe {
                            assert_unchecked(luma_y < GRAIN_HEIGHT + 1 - 1);
                            assert_unchecked(luma_x < GRAIN_WIDTH - 1);
                        }
                        for i in 0..1 + is_sub.y as usize {
                            for j in 0..1 + is_sub.x as usize {
                                luma += buf_y[luma_y + i][luma_x + j].as_::<c_int>();
                            }
                        }
                        luma = round2(luma, is_sub.y as u8 + is_sub.x as u8);

                        sum += luma * coeff[0] as c_int;
                        break;
                    }
                    sum += coeff[0] as c_int * buf_val.as_::<c_int>();
                    coeff = &coeff[1..];
                }
            }

            let buf_yx = &mut buf[y + AR_PAD][x + AR_PAD];
            let grain = (*buf_yx).as_::<c_int>() + round2(sum, data.ar_coeff_shift);
            *buf_yx = iclip(grain, grain_min, grain_max).as_::<BD::Entry>();
        }
    }
}

/// # Safety
///
/// Must be called by [`generate_grain_uv::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
#[inline(never)]
unsafe extern "C" fn generate_grain_uv_c_erased<
    BD: BitDepth,
    const NM: usize,
    const IS_SUBX: bool,
    const IS_SUBY: bool,
>(
    buf: *mut GrainLut<DynEntry>,
    buf_y: *const GrainLut<DynEntry>,
    data: &Dav1dFilmGrainData,
    uv: intptr_t,
    bitdepth_max: c_int,
) {
    // SAFETY: Casting back to the original type from the `generate_grain_uv::Fn::call`.
    let buf = unsafe { &mut *buf.cast() };
    // SAFETY: Casting back to the original type from the `generate_grain_uv::Fn::call`.
    let buf_y = unsafe { &*buf_y.cast() };
    let data = &data.clone().into();
    let is_uv = uv != 0;
    let bd = BD::from_c(bitdepth_max);
    generate_grain_uv_rust(buf, buf_y, data, is_uv, IS_SUBX, IS_SUBY, bd)
}

/// Sample from the correct block of a grain LUT,
/// while taking into account the offsets
/// provided by the offsets cache.
#[inline]
fn sample_lut<BD: BitDepth>(
    grain_lut: &GrainLut<BD::Entry>,
    offsets: &[[c_int; 2]; 2],
    is_subx: bool,
    is_suby: bool,
    is_bx: bool,
    is_by: bool,
    x: usize,
    y: usize,
) -> i32 {
    let [subx, suby, bx, by] = [is_subx, is_suby, is_bx, is_by].map(|it| it as usize);

    let randval = offsets[bx][by] as usize;
    let offx = 3 + (2 >> subx) * (3 + (randval >> 4));
    let offy = 3 + (2 >> suby) * (3 + (randval & ((1 << 4) - 1)));
    grain_lut[offy + y + (FG_BLOCK_SIZE >> suby) * by][offx + x + (FG_BLOCK_SIZE >> subx) * bx]
        .as_::<i32>()
}

/// # Safety
///
/// Must be called by [`fgy_32x32xn::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn fgy_32x32xn_c_erased<BD: BitDepth>(
    _dst_row_ptr: *mut DynPixel,
    _src_row_ptr: *const DynPixel,
    _stride: ptrdiff_t,
    data: &Dav1dFilmGrainData,
    pw: usize,
    scaling: *const DynScaling,
    grain_lut: *const GrainLut<DynEntry>,
    bh: c_int,
    row_num: c_int,
    bitdepth_max: c_int,
    dst_row: FFISafeRav1dPictureDataComponentOffset,
    src_row: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `fgy_32x32xn::Fn::call`.
    let [dst_row, src_row] = [dst_row, src_row].map(|it| unsafe { FFISafe::from_with_offset(it) });
    let data = &data.clone().into();
    // SAFETY: Casting back to the original type from the `fn` ptr call.
    let scaling = unsafe { &*scaling.cast() };
    // SAFETY: Casting back to the original type from the `fn` ptr call.
    let grain_lut = unsafe { &*grain_lut.cast() };
    let bh = bh as usize;
    let row_num = row_num as usize;
    let bd = BD::from_c(bitdepth_max);
    fgy_32x32xn_rust(
        dst_row, src_row, data, pw, scaling, grain_lut, bh, row_num, bd,
    )
}

fn fgy_32x32xn_rust<BD: BitDepth>(
    dst_row: Rav1dPictureDataComponentOffset,
    src_row: Rav1dPictureDataComponentOffset,
    data: &Rav1dFilmGrainData,
    pw: usize,
    scaling: &BD::Scaling,
    grain_lut: &GrainLut<BD::Entry>,
    bh: usize,
    row_num: usize,
    bd: BD,
) {
    let rows = 1 + (data.overlap_flag && row_num > 0) as usize;
    let bitdepth_min_8 = bd.bitdepth() - 8;
    let grain_ctr = 128 << bitdepth_min_8;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;

    let min_value;
    let max_value;
    if data.clip_to_restricted_range {
        min_value = 16 << bitdepth_min_8;
        max_value = 235 << bitdepth_min_8;
    } else {
        min_value = 0;
        max_value = bd.bitdepth_max().as_::<c_int>();
    }

    let mut seed = row_seed(rows, row_num, data);

    assert!(dst_row.stride() % (FG_BLOCK_SIZE * mem::size_of::<BD::Pixel>()) as isize == 0);

    let mut offsets: [[c_int; 2]; 2] = [[0; 2 /* row offset */]; 2 /* col offset */];

    // process this row in FG_BLOCK_SIZE^2 blocks
    for bx in (0..pw).step_by(FG_BLOCK_SIZE) {
        let bw = cmp::min(FG_BLOCK_SIZE, pw - bx);

        if data.overlap_flag && bx != 0 {
            // shift previous offsets left
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }

        // update current offsets
        for i in 0..rows {
            offsets[0][i] = get_random_number(8, &mut seed[i]);
        }

        // x/y block offsets to compensate for overlapped regions
        let ystart = if data.overlap_flag && row_num != 0 {
            cmp::min(2, bh)
        } else {
            0
        };
        let xstart = if data.overlap_flag && bx != 0 {
            cmp::min(2, bw)
        } else {
            0
        };

        static W: [[c_int; 2]; 2] = [[27, 17], [17, 27]];

        let src_row_y = |y| {
            let row = src_row + y as isize * src_row.pixel_stride::<BD>();
            (row + bx).slice::<BD>(bw)
        };
        let dst_row_y = |y| {
            let row = dst_row + y as isize * dst_row.pixel_stride::<BD>();
            (row + bx).slice_mut::<BD>(bw)
        };

        let noise_y = |src: BD::Pixel, grain| {
            let noise = round2(
                scaling.as_ref()[src.to::<usize>()] as c_int * grain,
                data.scaling_shift,
            );
            iclip(src.as_::<c_int>() + noise, min_value, max_value).as_::<BD::Pixel>()
        };

        for y in ystart..bh {
            let src = &*src_row_y(y);
            let dst = &mut *dst_row_y(y);

            // Non-overlapped image region (straightforward)
            for x in xstart..bw {
                let grain = sample_lut::<BD>(grain_lut, &offsets, false, false, false, false, x, y);
                dst[x] = noise_y(src[x], grain);
            }

            // Special case for overlapped column
            for x in 0..xstart {
                let grain = sample_lut::<BD>(grain_lut, &offsets, false, false, false, false, x, y);
                let old = sample_lut::<BD>(grain_lut, &offsets, false, false, true, false, x, y);
                let grain = round2(old * W[x][0] + grain * W[x][1], 5);
                let grain = iclip(grain, grain_min, grain_max);
                dst[x] = noise_y(src[x], grain);
            }
        }
        for y in 0..ystart {
            let src = &*src_row_y(y);
            let dst = &mut *dst_row_y(y);

            // Special case for overlapped row (sans corner)
            for x in xstart..bw {
                let grain = sample_lut::<BD>(grain_lut, &offsets, false, false, false, false, x, y);
                let old = sample_lut::<BD>(grain_lut, &offsets, false, false, false, true, x, y);
                let grain = round2(old * W[y][0] + grain * W[y][1], 5);
                let grain = iclip(grain, grain_min, grain_max);
                dst[x] = noise_y(src[x], grain);
            }

            // Special case for doubly-overlapped corner
            for x in 0..xstart {
                // Blend the top pixel with the top left block
                let top = sample_lut::<BD>(grain_lut, &offsets, false, false, false, true, x, y);
                let old = sample_lut::<BD>(grain_lut, &offsets, false, false, true, true, x, y);
                let top = round2(old * W[x][0] + top * W[x][1], 5);
                let top = iclip(top, grain_min, grain_max);

                // Blend the current pixel with the left block
                let grain = sample_lut::<BD>(grain_lut, &offsets, false, false, false, false, x, y);
                let old = sample_lut::<BD>(grain_lut, &offsets, false, false, true, false, x, y);

                // Mix the row rows together and apply grain
                let grain = round2(old * W[x][0] + grain * W[x][1], 5);
                let grain = iclip(grain, grain_min, grain_max);
                let grain = round2(top * W[y][0] + grain * W[y][1], 5);
                let grain = iclip(grain, grain_min, grain_max);
                dst[x] = noise_y(src[x], grain);
            }
        }
    }
}

fn fguv_32x32xn_rust<BD: BitDepth>(
    dst_row: Rav1dPictureDataComponentOffset,
    src_row: Rav1dPictureDataComponentOffset,
    data: &Rav1dFilmGrainData,
    pw: usize,
    scaling: &BD::Scaling,
    grain_lut: &GrainLut<BD::Entry>,
    bh: usize,
    row_num: usize,
    luma_row: Rav1dPictureDataComponentOffset,
    is_uv: bool,
    is_id: bool,
    is_sx: bool,
    is_sy: bool,
    bd: BD,
) {
    let [uv, sx, sy] = [is_uv, is_sx, is_sy].map(|it| it as usize);

    let rows = 1 + (data.overlap_flag && row_num > 0) as usize;
    let bitdepth_min_8 = bd.bitdepth() - 8;
    let grain_ctr = 128 << bitdepth_min_8;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;

    let min_value;
    let max_value;
    if data.clip_to_restricted_range {
        min_value = 16 << bitdepth_min_8;
        max_value = (if is_id { 235 } else { 240 }) << bitdepth_min_8;
    } else {
        min_value = 0;
        max_value = bd.bitdepth_max().as_::<c_int>();
    }

    let mut seed = row_seed(rows, row_num, data);

    assert!(dst_row.stride() % (FG_BLOCK_SIZE * mem::size_of::<BD::Pixel>()) as isize == 0);

    let mut offsets: [[c_int; 2]; 2] = [[0; 2 /* row offset */]; 2 /* col offset */];

    // process this row in FG_BLOCK_SIZE^2 blocks (subsampled)
    for bx in (0..pw).step_by(FG_BLOCK_SIZE >> sx) {
        let bw = cmp::min(FG_BLOCK_SIZE >> sx, pw - bx);
        if data.overlap_flag && bx != 0 {
            // shift previous offsets left
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }

        // update current offsets
        for i in 0..rows {
            offsets[0][i] = get_random_number(8, &mut seed[i]);
        }

        // x/y block offsets to compensate for overlapped regions
        let ystart = if data.overlap_flag && row_num != 0 {
            cmp::min(2 >> sy, bh)
        } else {
            0
        };
        let xstart = if data.overlap_flag && bx != 0 {
            cmp::min(2 >> sx, bw)
        } else {
            0
        };

        static W: [[[c_int; 2]; 2 /* off */]; 2 /* sub */] = [[[27, 17], [17, 27]], [[23, 22], [0; 2]]];

        let luma_row_uv = |y| {
            let row = luma_row + (y << sy) as isize * luma_row.pixel_stride::<BD>();
            (row + (bx << sx)).slice::<BD>(bw << sx)
        };
        let src_row_uv = |y| {
            let row = src_row + y as isize * src_row.pixel_stride::<BD>();
            (row + bx).slice::<BD>(bw)
        };
        let dst_row_uv = |y| {
            let row = dst_row + y as isize * dst_row.pixel_stride::<BD>();
            (row + bx).slice_mut::<BD>(bw)
        };

        let noise_uv = |src: BD::Pixel, grain, luma: &[BD::Pixel]| {
            let mut avg = luma[0];
            if is_sx {
                avg = ((avg.as_::<c_int>() + luma[1].as_::<c_int>() + 1) >> 1).as_::<BD::Pixel>();
            }
            let mut val = avg.as_::<c_int>();
            if !data.chroma_scaling_from_luma {
                let combined = avg.as_::<c_int>() * data.uv_luma_mult[uv]
                    + src.as_::<c_int>() * data.uv_mult[uv];
                val = bd
                    .iclip_pixel((combined >> 6) + data.uv_offset[uv] * (1 << bitdepth_min_8))
                    .as_::<c_int>();
            }
            // `val` isn't out of bounds, so we can
            // eliminate extra panicking code by bit-truncating `val`.
            let noise = round2(
                scaling.as_ref()[val as usize % scaling.as_ref().len()] as c_int * grain,
                data.scaling_shift,
            );
            iclip(src.as_::<c_int>() + noise, min_value, max_value).as_::<BD::Pixel>()
        };

        for y in ystart..bh {
            let luma = &*luma_row_uv(y);
            let src = &*src_row_uv(y);
            let dst = &mut *dst_row_uv(y);

            // Non-overlapped image region (straightforward)
            for x in xstart..bw {
                let grain = sample_lut::<BD>(grain_lut, &offsets, is_sx, is_sy, false, false, x, y);
                dst[x] = noise_uv(src[x], grain, &luma[x << sx..]);
            }

            // Special case for overlapped column
            for x in 0..xstart {
                let grain = sample_lut::<BD>(grain_lut, &offsets, is_sx, is_sy, false, false, x, y);
                let old = sample_lut::<BD>(grain_lut, &offsets, is_sx, is_sy, true, false, x, y);
                let grain = round2(old * W[sx][x][0] + grain * W[sx][x][1], 5);
                let grain = iclip(grain, grain_min, grain_max);
                dst[x] = noise_uv(src[x], grain, &luma[x << sx..]);
            }
        }
        for y in 0..ystart {
            let luma = &*luma_row_uv(y);
            let src = &*src_row_uv(y);
            let dst = &mut *dst_row_uv(y);

            // Special case for overlapped row (sans corner)
            for x in xstart..bw {
                let grain = sample_lut::<BD>(grain_lut, &offsets, is_sx, is_sy, false, false, x, y);
                let old = sample_lut::<BD>(grain_lut, &offsets, is_sx, is_sy, false, true, x, y);
                let grain = round2(old * W[sy][y][0] + grain * W[sy][y][1], 5);
                let grain = iclip(grain, grain_min, grain_max);
                dst[x] = noise_uv(src[x], grain, &luma[x << sx..]);
            }

            // Special case for doubly-overlapped corner
            for x in 0..xstart {
                // Blend the top pixel with the top left block
                let top = sample_lut::<BD>(grain_lut, &offsets, is_sx, is_sy, false, true, x, y);
                let old = sample_lut::<BD>(grain_lut, &offsets, is_sx, is_sy, true, true, x, y);
                let top = round2(old * W[sx][x][0] + top * W[sx][x][1], 5);
                let top = iclip(top, grain_min, grain_max);

                // Blend the current pixel with the left block
                let grain = sample_lut::<BD>(grain_lut, &offsets, is_sx, is_sy, false, false, x, y);
                let old = sample_lut::<BD>(grain_lut, &offsets, is_sx, is_sy, true, false, x, y);

                // Mix the row rows together and apply to image
                let grain = round2(old * W[sx][x][0] + grain * W[sx][x][1], 5);
                let grain = iclip(grain, grain_min, grain_max);
                let grain = round2(top * W[sy][y][0] + grain * W[sy][y][1], 5);
                let grain = iclip(grain, grain_min, grain_max);
                dst[x] = noise_uv(src[x], grain, &luma[x << sx..]);
            }
        }
    }
}

/// # Safety
///
/// Must be called by [`fguv_32x32xn::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
#[inline(never)]
unsafe extern "C" fn fguv_32x32xn_c_erased<
    BD: BitDepth,
    const NM: usize,
    const IS_SX: bool,
    const IS_SY: bool,
>(
    _dst_row_ptr: *mut DynPixel,
    _src_row_ptr: *const DynPixel,
    _stride: ptrdiff_t,
    data: &Dav1dFilmGrainData,
    pw: usize,
    scaling: *const DynScaling,
    grain_lut: *const GrainLut<DynEntry>,
    bh: c_int,
    row_num: c_int,
    _luma_row_ptr: *const DynPixel,
    _luma_stride: ptrdiff_t,
    uv_pl: c_int,
    is_id: c_int,
    bitdepth_max: c_int,
    dst_row: FFISafeRav1dPictureDataComponentOffset,
    src_row: FFISafeRav1dPictureDataComponentOffset,
    luma_row: FFISafeRav1dPictureDataComponentOffset,
) {
    let [dst_row, src_row, luma_row] = [dst_row, src_row, luma_row].map(|row| {
        // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `fguv_32x32xn::Fn::call`.
        unsafe { FFISafe::from_with_offset(row) }
    });
    let data = &data.clone().into();
    // SAFETY: Casting back to the original type from the `fn` ptr call.
    let scaling = unsafe { &*scaling.cast() };
    // SAFETY: Casting back to the original type from the `fn` ptr call.
    let grain_lut = unsafe { &*grain_lut.cast() };
    let bh = bh as usize;
    let row_num = row_num as usize;
    let uv_pl = uv_pl as usize;
    let is_id = is_id != 0;
    let bd = BD::from_c(bitdepth_max);
    fguv_32x32xn_rust(
        dst_row,
        src_row,
        data,
        pw,
        scaling,
        grain_lut,
        bh,
        row_num,
        luma_row,
        uv_pl != 0,
        is_id,
        IS_SX,
        IS_SY,
        bd,
    )
}

/// [`fgy_32x32xn_rust`] and [`fguv_32x32xn_rust`] in `i16` lanes, a block row as four (two)
/// vectors of eight. The overlap blends run on whole vectors, with weights that leave the lanes
/// outside the overlap as they are. `noise = round2(scaling * grain, shift)` is one `mulhrs` of
/// the grain by `scaling << (15 - shift)`. The scaling bytes come one load per lane, from
/// sixteen-lane `tbl`s on NEON at 8 bpc, and from `vpgatherdd` with AVX2.
mod rust_simd {
    use zerocopy::{AsBytes, FromBytes};

    use super::*;
    #[cfg(target_arch = "x86_64")]
    use crate::simd128::I16x16;
    use crate::simd128::{I16x8, I32x4};
    #[cfg(target_arch = "aarch64")]
    use crate::simd128::{Lut256, U8x16};

    pub trait Px: AsBytes + FromBytes + Copy + Default {
        type E: Copy;
        fn load8(s: &[Self]) -> I16x8;
        /// Lanes already in the pixel range.
        fn store8(v: I16x8, d: &mut [Self]);
        fn grain8(e: &[Self::E]) -> I16x8;
        /// The rounded averages of the pairs in `s[..16]`.
        fn luma_avg(s: &[Self]) -> I16x8;
        fn e2i(e: Self::E) -> i32;
        fn i2e(v: i32) -> Self::E;
        #[cfg(target_arch = "x86_64")]
        fn load16(s: &[Self]) -> I16x16;
        #[cfg(target_arch = "x86_64")]
        fn store16(v: I16x16, d: &mut [Self]);
    }

    impl Px for u8 {
        type E = i8;

        #[inline(always)]
        fn load8(s: &[Self]) -> I16x8 {
            I16x8::from_u8x8(s)
        }

        #[inline(always)]
        fn store8(v: I16x8, d: &mut [Self]) {
            d[..8].copy_from_slice(&v.pack_u8());
        }

        #[inline(always)]
        fn grain8(e: &[i8]) -> I16x8 {
            I16x8::from_i8x8(e[..8].as_bytes())
        }

        #[inline(always)]
        fn luma_avg(s: &[Self]) -> I16x8 {
            I16x8::avg_pairs_u8(s)
        }

        #[inline(always)]
        fn e2i(e: i8) -> i32 {
            e as i32
        }

        #[inline(always)]
        fn i2e(v: i32) -> i8 {
            v as i8
        }

        #[cfg(target_arch = "x86_64")]
        #[inline(always)]
        fn load16(s: &[Self]) -> I16x16 {
            I16x16::from_u8x16(s)
        }

        #[cfg(target_arch = "x86_64")]
        #[inline(always)]
        fn store16(v: I16x16, d: &mut [Self]) {
            d[..16].copy_from_slice(&v.pack_u8());
        }
    }

    impl Px for u16 {
        type E = i16;

        #[inline(always)]
        fn load8(s: &[Self]) -> I16x8 {
            I16x8::load_u16(s)
        }

        #[inline(always)]
        fn store8(v: I16x8, d: &mut [Self]) {
            v.store_u16(d);
        }

        #[inline(always)]
        fn grain8(e: &[i16]) -> I16x8 {
            I16x8::load(e)
        }

        #[inline(always)]
        fn luma_avg(s: &[Self]) -> I16x8 {
            let (e, o) = Self::load8(s).unzip(Self::load8(&s[8..]));
            e.avg_u(o)
        }

        #[inline(always)]
        fn e2i(e: i16) -> i32 {
            e as i32
        }

        #[inline(always)]
        fn i2e(v: i32) -> i16 {
            v as i16
        }

        #[cfg(target_arch = "x86_64")]
        #[inline(always)]
        fn load16(s: &[Self]) -> I16x16 {
            I16x16::load_u16(s)
        }

        #[cfg(target_arch = "x86_64")]
        #[inline(always)]
        fn store16(v: I16x16, d: &mut [Self]) {
            v.store_u16(d);
        }
    }

    /// The scaling bytes of sixteen lanes, each in `0..t.len()`.
    pub trait Lookup {
        /// Whether [`Lookup::get16`] is there, for sixteen-lane AVX2 kernels.
        #[cfg(target_arch = "x86_64")]
        const WIDE: bool = false;

        fn get(&self, i: [I16x8; 2]) -> [I16x8; 2];

        #[cfg(target_arch = "x86_64")]
        #[inline(always)]
        fn get16(&self, _: I16x16) -> I16x16 {
            unreachable!()
        }

        /// [`Lookup::get`] of lanes clipped to `0..=max` first, `max + 1` the table's length.
        #[inline(always)]
        fn get_clipped(&self, i: [I16x8; 2], max: I16x8) -> [I16x8; 2] {
            let z = I16x8::splat(0);
            self.get([i[0].max(z).min(max), i[1].max(z).min(max)])
        }
    }

    pub struct PerLane<'a>(pub &'a [u8]);

    impl Lookup for PerLane<'_> {
        #[inline(always)]
        fn get(&self, i: [I16x8; 2]) -> [I16x8; 2] {
            let mut idx = [0; 16];
            i[0].store_u16(&mut idx[..8]);
            i[1].store_u16(&mut idx[8..]);
            let mask = self.0.len() - 1;
            let mut v = [0; 16];
            for k in 0..16 {
                v[k] = self.0[idx[k] as usize & mask];
            }
            [I16x8::from_u8x8(&v[..8]), I16x8::from_u8x8(&v[8..])]
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub struct Tbl(pub Lut256);

    #[cfg(target_arch = "aarch64")]
    impl Lookup for Tbl {
        #[inline(always)]
        fn get(&self, i: [I16x8; 2]) -> [I16x8; 2] {
            U8x16::pack(i[0], i[1]).lookup256(&self.0).widen()
        }

        /// The pack to bytes saturates to the table.
        #[inline(always)]
        fn get_clipped(&self, i: [I16x8; 2], _: I16x8) -> [I16x8; 2] {
            self.get(i)
        }
    }

    /// The table with three bytes of padding.
    #[cfg(target_arch = "x86_64")]
    pub struct Gather<'a>(pub &'a [u8], pub u16);

    #[cfg(target_arch = "x86_64")]
    impl Lookup for Gather<'_> {
        #[inline(always)]
        fn get(&self, i: [I16x8; 2]) -> [I16x8; 2] {
            [i[0].gather_u8(self.0, self.1), i[1].gather_u8(self.0, self.1)]
        }

        #[cfg(target_arch = "x86_64")]
        const WIDE: bool = true;

        #[inline(always)]
        fn get16(&self, i: I16x16) -> I16x16 {
            i.gather_u8(self.0, self.1)
        }
    }

    #[inline(always)]
    fn px<P: Px>(b: &[u8]) -> &[P] {
        FromBytes::slice_from(b).unwrap()
    }

    #[inline(always)]
    fn px_mut<P: Px>(b: &mut [u8]) -> &mut [P] {
        FromBytes::mut_slice_from(b).unwrap()
    }

    /// `clip(round2(old * wo + cur * wc, 5))` to the grain range.
    #[inline(always)]
    fn blend(old: I16x8, cur: I16x8, wo: I16x8, wc: I16x8, lim: (I16x8, I16x8)) -> I16x8 {
        let [a0, a1] = old.mul_wide(wo);
        let [b0, b1] = cur.mul_wide(wc);
        I32x4::pack_round_shr::<5>(a0.add(b0), a1.add(b1))
            .max(lim.0)
            .min(lim.1)
    }

    /// The lane weights of the left overlap: `W[sx][x]` for its columns, `(0, 32)` elsewhere.
    #[inline(always)]
    fn h_weights(sx: bool) -> (I16x8, I16x8) {
        let (mut wo, mut wc) = ([0; 8], [32; 8]);
        if sx {
            (wo[0], wc[0]) = (23, 22);
        } else {
            (wo[0], wc[0], wo[1], wc[1]) = (27, 17, 17, 27);
        }
        (I16x8::load(&wo), I16x8::load(&wc))
    }

    /// Where the grain of block `(bx, by)` (0 current, 1 left or above) starts in the LUT.
    #[inline(always)]
    fn lut_at(
        offsets: &[[c_int; 2]; 2],
        sx: bool,
        sy: bool,
        bx: usize,
        by: usize,
    ) -> (usize, usize) {
        let r = offsets[bx][by] as usize;
        let offx = 3 + (2 >> sx as usize) * (3 + (r >> 4));
        let offy = 3 + (2 >> sy as usize) * (3 + (r & 15));
        (
            offy + (FG_BLOCK_SIZE >> sy as usize) * by,
            offx + (FG_BLOCK_SIZE >> sx as usize) * bx,
        )
    }

    /// The LUT positions of a block's grain and of the grain it blends with: its own, the
    /// left block's, the top's and the top left's, as `(row, column)`. The columns are clamped
    /// to where `NV` vectors fit in a row, which they already are, so that loads need no checks.
    #[inline(always)]
    fn lut_pos<const NV: usize>(offsets: &[[c_int; 2]; 2], sx: bool, sy: bool) -> [(usize, usize); 4] {
        [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(bx, by)| {
            let (oy, ox) = lut_at(offsets, sx, sy, bx, by);
            // The left blocks give one vector.
            (oy, ox.min(GRAIN_WIDTH - 8 * if bx == 0 { NV } else { 1 }))
        })
    }

    #[inline(always)]
    fn grain_at<P: Px, const NV: usize>(row: &[P::E; GRAIN_WIDTH], ox: usize) -> [I16x8; NV] {
        let row = &row[..ox + 8 * NV];
        let mut g = [I16x8::splat(0); NV];
        for c in 0..NV {
            g[c] = P::grain8(&row[ox + 8 * c..]);
        }
        g
    }

    /// Row `y`'s `NV` grain vectors, blended as the overlaps ask: with the left block for
    /// `XOVER`, with the top for `TOP` by the weights `vw`.
    #[inline(always)]
    fn grain_row<P: Px, const NV: usize, const XOVER: bool, const TOP: bool>(
        lut: &GrainLut<P::E>,
        pos: &[(usize, usize); 4],
        y: usize,
        vw: (I16x8, I16x8),
        hw: (I16x8, I16x8),
        lim: (I16x8, I16x8),
    ) -> [I16x8; NV] {
        let at = |k: usize| (&lut[pos[k].0 + y], pos[k].1);
        let (row, ox) = at(0);
        let mut g = grain_at::<P, NV>(row, ox);
        if XOVER {
            let (row, ox) = at(1);
            g[0] = blend(grain_at::<P, 1>(row, ox)[0], g[0], hw.0, hw.1, lim);
        }
        if TOP {
            let (row, ox) = at(2);
            let mut t = grain_at::<P, NV>(row, ox);
            if XOVER {
                let (row, ox) = at(3);
                t[0] = blend(grain_at::<P, 1>(row, ox)[0], t[0], hw.0, hw.1, lim);
            }
            for c in 0..NV {
                g[c] = blend(t[c], g[c], vw.0, vw.1, lim);
            }
        }
        g
    }

    /// The pixel bounds and `1 << (15 - scaling_shift)`.
    #[derive(Clone, Copy)]
    struct Apply {
        min: I16x8,
        max: I16x8,
        mul: I16x8,
        #[cfg(target_arch = "x86_64")]
        wide: [I16x16; 3],
    }

    impl Apply {
        #[inline(always)]
        fn new(min: I16x8, max: I16x8, scaling_shift: u8) -> Self {
            let mul = I16x8::splat(1 << (15 - scaling_shift));
            Self {
                min,
                max,
                mul,
                #[cfg(target_arch = "x86_64")]
                wide: [min, max, mul].map(|v| I16x16::from_halves(v, v)),
            }
        }
    }

    /// [`add_noise`] in sixteen lanes.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn add_noise16(src: I16x16, scale: I16x16, grain: I16x16, a: Apply) -> I16x16 {
        let [min, max, mul] = a.wide;
        src.add(grain.mulhrs(scale.mul(mul))).max(min).min(max)
    }

    #[inline(always)]
    fn add_noise(src: I16x8, scale: I16x8, grain: I16x8, a: Apply) -> I16x8 {
        src.add(grain.mulhrs(scale.mul(a.mul)))
            .max(a.min)
            .min(a.max)
    }

    #[inline(always)]
    fn apply_y<P: Px, L: Lookup>(src: &[P], dst: &mut [P], g: &[I16x8; 4], lut: &L, a: Apply) {
        #[cfg(target_arch = "x86_64")]
        if L::WIDE {
            for h in 0..2 {
                let s = P::load16(&src[16 * h..]);
                let g = I16x16::from_halves(g[2 * h], g[2 * h + 1]);
                P::store16(add_noise16(s, lut.get16(s), g, a), &mut dst[16 * h..]);
            }
            return;
        }
        for h in 0..2 {
            let s = [P::load8(&src[16 * h..]), P::load8(&src[16 * h + 8..])];
            let sc = lut.get(s);
            for k in 0..2 {
                P::store8(
                    add_noise(s[k], sc[k], g[2 * h + k], a),
                    &mut dst[16 * h + 8 * k..],
                );
            }
        }
    }

    /// Runs `$e` on the row's `n` pixels (`nl` of luma) in place when the block is `full` wide,
    /// else through copies. A macro, as closures would not take the kernel's target features.
    macro_rules! with_row {
        ($src:expr, $luma:expr, $dst:expr, ($n:expr, $nl:expr, $full:expr), |$s:ident, $l:ident, $d:ident| $e:expr) => {{
            let (src, luma, dst, n, nl) = ($src, $luma, $dst, $n, $nl);
            if n == $full {
                let ($s, $l, $d) = (src, luma, dst);
                $e
            } else {
                const N: usize = FG_BLOCK_SIZE;
                let (mut s, mut l, mut d) = ([P::default(); N], [P::default(); N], [P::default(); N]);
                s[..n].copy_from_slice(&src[..n]);
                l[..nl].copy_from_slice(&luma[..nl]);
                {
                    let ($s, $l, $d) = (&s[..], &l[..], &mut d[..]);
                    $e
                }
                dst[..n].copy_from_slice(&d[..n]);
            }
        }};
    }

    #[inline(always)]
    fn bounds(
        data: &Rav1dFilmGrainData,
        bitdepth_max: c_int,
        max_uv: c_int,
    ) -> (I16x8, I16x8, (I16x8, I16x8)) {
        let bitdepth_min_8 = (bitdepth_max + 1).ilog2() - 8;
        let (min, max) = if data.clip_to_restricted_range {
            (16 << bitdepth_min_8, max_uv << bitdepth_min_8)
        } else {
            (0, bitdepth_max)
        };
        let ctr = 128 << bitdepth_min_8;
        (
            I16x8::splat(min as i16),
            I16x8::splat(max as i16),
            (I16x8::splat(-ctr as i16), I16x8::splat((ctr - 1) as i16)),
        )
    }

    #[inline(always)]
    fn fgy<BD: BitDepth, P: Px, L: Lookup>(
        dst_row: Rav1dPictureDataComponentOffset,
        src_row: Rav1dPictureDataComponentOffset,
        data: &Rav1dFilmGrainData,
        pw: usize,
        lookup: &L,
        lut: &GrainLut<P::E>,
        bh: usize,
        row_num: usize,
        bitdepth_max: c_int,
    ) {
        let rows = 1 + (data.overlap_flag && row_num > 0) as usize;
        let (min, max, lim) = bounds(data, bitdepth_max, 235);
        let a = Apply::new(min, max, data.scaling_shift);
        let hw = h_weights(false);
        let mut seed = row_seed(rows, row_num, data);
        let mut offsets = [[0; 2]; 2];
        for bx in (0..pw).step_by(FG_BLOCK_SIZE) {
            let bw = cmp::min(FG_BLOCK_SIZE, pw - bx);
            if data.overlap_flag && bx != 0 {
                for i in 0..rows {
                    offsets[1][i] = offsets[0][i];
                }
            }
            for i in 0..rows {
                offsets[0][i] = get_random_number(8, &mut seed[i]);
            }
            let ystart = if data.overlap_flag && row_num != 0 { cmp::min(2, bh) } else { 0 };
            let xover = data.overlap_flag && bx != 0;
            let pos = lut_pos::<4>(&offsets, false, false);
            let k = Fgy { src_row, dst_row, bx, bw, lut, pos, hw, lim, lookup, a };
            for y in 0..ystart {
                const W: [[i16; 2]; 2] = [[27, 17], [17, 27]];
                let vw = (I16x8::splat(W[y][0]), I16x8::splat(W[y][1]));
                if xover {
                    fgy_row::<BD, P, L, true, true>(&k, y, vw);
                } else {
                    fgy_row::<BD, P, L, false, true>(&k, y, vw);
                }
            }
            let vw = hw;
            if xover {
                for y in ystart..bh {
                    fgy_row::<BD, P, L, true, false>(&k, y, vw);
                }
            } else {
                for y in ystart..bh {
                    fgy_row::<BD, P, L, false, false>(&k, y, vw);
                }
            }
        }
    }

    /// What every row of a luma block takes.
    struct Fgy<'a, P: Px, L> {
        src_row: Rav1dPictureDataComponentOffset<'a>,
        dst_row: Rav1dPictureDataComponentOffset<'a>,
        bx: usize,
        bw: usize,
        lut: &'a GrainLut<P::E>,
        pos: [(usize, usize); 4],
        hw: (I16x8, I16x8),
        lim: (I16x8, I16x8),
        lookup: &'a L,
        a: Apply,
    }

    #[inline(always)]
    fn fgy_row<BD: BitDepth, P: Px, L: Lookup, const XOVER: bool, const TOP: bool>(
        k: &Fgy<P, L>,
        y: usize,
        vw: (I16x8, I16x8),
    ) {
        let g = grain_row::<P, 4, XOVER, TOP>(k.lut, &k.pos, y, vw, k.hw, k.lim);
        let (src_row, dst_row, bx, bw) = (k.src_row, k.dst_row, k.bx, k.bw);
        let off = y as isize;
        let src = (src_row + off * src_row.pixel_stride::<BD>() + bx).slice::<BD>(bw);
        let mut dst = (dst_row + off * dst_row.pixel_stride::<BD>() + bx).slice_mut::<BD>(bw);
        let (src, dst) = (px::<P>(src.as_bytes()), px_mut::<P>(dst.as_bytes_mut()));
        let (lookup, a) = (k.lookup, k.a);
        with_row!(src, &[][..], dst, (bw, 0, FG_BLOCK_SIZE), |s, _l, d| {
            apply_y(s, d, &g, lookup, a)
        });
    }

    /// The chroma constants: `uv_luma_mult`, `uv_mult` and `uv_offset << (bitdepth - 8)`.
    /// `(luma * luma_mult + src * mult >> 6) + offset` is within `i16` up to 12 bits.
    #[derive(Clone, Copy)]
    struct Uv {
        csfl: bool,
        luma_mult: I16x8,
        mult: I16x8,
        offset: I16x8,
        pixel_max: I16x8,
    }

    #[inline(always)]
    fn apply_uv<P: Px, L: Lookup, const SX: bool, const NV: usize>(
        src: &[P],
        luma: &[P],
        dst: &mut [P],
        g: &[I16x8; NV],
        lookup: &L,
        a: Apply,
        uv: Uv,
    ) {
        for h in 0..NV / 2 {
            let mut s = [I16x8::splat(0); 2];
            let mut idx = [I16x8::splat(0); 2];
            for k in 0..2 {
                let c = 2 * h + k;
                let avg = if SX {
                    P::luma_avg(&luma[16 * c..])
                } else {
                    P::load8(&luma[8 * c..])
                };
                s[k] = P::load8(&src[8 * c..]);
                idx[k] = if uv.csfl {
                    avg
                } else {
                    let [l0, l1] = avg.mul_wide(uv.luma_mult);
                    let [s0, s1] = s[k].mul_wide(uv.mult);
                    I32x4::pack_shr::<6>(l0.add(s0), l1.add(s1)).add(uv.offset)
                };
            }
            #[cfg(target_arch = "x86_64")]
            if L::WIDE {
                let mut i = I16x16::from_halves(idx[0], idx[1]);
                if !uv.csfl {
                    let max = I16x16::from_halves(uv.pixel_max, uv.pixel_max);
                    i = i.max(I16x16::splat(0)).min(max);
                }
                let s = I16x16::from_halves(s[0], s[1]);
                let g = I16x16::from_halves(g[2 * h], g[2 * h + 1]);
                P::store16(add_noise16(s, lookup.get16(i), g, a), &mut dst[16 * h..]);
                continue;
            }
            let sc = if uv.csfl {
                lookup.get(idx)
            } else {
                lookup.get_clipped(idx, uv.pixel_max)
            };
            for k in 0..2 {
                let c = 2 * h + k;
                P::store8(add_noise(s[k], sc[k], g[c], a), &mut dst[8 * c..]);
            }
        }
    }

    #[inline(always)]
    fn fguv<BD: BitDepth, P: Px, L: Lookup, const SX: bool, const SY: bool, const NV: usize>(
        dst_row: Rav1dPictureDataComponentOffset,
        src_row: Rav1dPictureDataComponentOffset,
        data: &Rav1dFilmGrainData,
        pw: usize,
        lookup: &L,
        lut: &GrainLut<P::E>,
        bh: usize,
        row_num: usize,
        luma_row: Rav1dPictureDataComponentOffset,
        is_uv: bool,
        is_id: bool,
        bitdepth_max: c_int,
    ) {
        let (sx, sy) = (SX as usize, SY as usize);
        let bw_full = FG_BLOCK_SIZE >> sx;
        let rows = 1 + (data.overlap_flag && row_num > 0) as usize;
        let (min, max, lim) = bounds(data, bitdepth_max, if is_id { 235 } else { 240 });
        let a = Apply::new(min, max, data.scaling_shift);
        let pl = is_uv as usize;
        let bitdepth_min_8 = (bitdepth_max + 1).ilog2() - 8;
        let uv = Uv {
            csfl: data.chroma_scaling_from_luma,
            luma_mult: I16x8::splat(data.uv_luma_mult[pl] as i16),
            mult: I16x8::splat(data.uv_mult[pl] as i16),
            offset: I16x8::splat((data.uv_offset[pl] << bitdepth_min_8) as i16),
            pixel_max: I16x8::splat(bitdepth_max as i16),
        };
        let hw = h_weights(SX);
        let mut seed = row_seed(rows, row_num, data);
        let mut offsets = [[0; 2]; 2];
        for bx in (0..pw).step_by(bw_full) {
            let bw = cmp::min(bw_full, pw - bx);
            if data.overlap_flag && bx != 0 {
                for i in 0..rows {
                    offsets[1][i] = offsets[0][i];
                }
            }
            for i in 0..rows {
                offsets[0][i] = get_random_number(8, &mut seed[i]);
            }
            let ystart = if data.overlap_flag && row_num != 0 {
                cmp::min(2 >> sy, bh)
            } else {
                0
            };
            let xover = data.overlap_flag && bx != 0;
            let pos = lut_pos::<NV>(&offsets, SX, SY);
            let k = Fguv { src_row, dst_row, luma_row, bx, bw, lut, pos, hw, lim, lookup, a, uv };
            for y in 0..ystart {
                const W: [[[i16; 2]; 2]; 2] = [[[27, 17], [17, 27]], [[23, 22], [0; 2]]];
                let vw = (I16x8::splat(W[sy][y][0]), I16x8::splat(W[sy][y][1]));
                if xover {
                    fguv_row::<BD, P, L, SX, SY, NV, true, true>(&k, y, vw);
                } else {
                    fguv_row::<BD, P, L, SX, SY, NV, false, true>(&k, y, vw);
                }
            }
            if xover {
                for y in ystart..bh {
                    fguv_row::<BD, P, L, SX, SY, NV, true, false>(&k, y, hw);
                }
            } else {
                for y in ystart..bh {
                    fguv_row::<BD, P, L, SX, SY, NV, false, false>(&k, y, hw);
                }
            }
        }
    }

    /// What every row of a chroma block takes.
    struct Fguv<'a, P: Px, L> {
        src_row: Rav1dPictureDataComponentOffset<'a>,
        dst_row: Rav1dPictureDataComponentOffset<'a>,
        luma_row: Rav1dPictureDataComponentOffset<'a>,
        bx: usize,
        bw: usize,
        lut: &'a GrainLut<P::E>,
        pos: [(usize, usize); 4],
        hw: (I16x8, I16x8),
        lim: (I16x8, I16x8),
        lookup: &'a L,
        a: Apply,
        uv: Uv,
    }

    #[inline(always)]
    fn fguv_row<
        BD: BitDepth,
        P: Px,
        L: Lookup,
        const SX: bool,
        const SY: bool,
        const NV: usize,
        const XOVER: bool,
        const TOP: bool,
    >(
        k: &Fguv<P, L>,
        y: usize,
        vw: (I16x8, I16x8),
    ) {
        let (sx, sy) = (SX as usize, SY as usize);
        let g = grain_row::<P, NV, XOVER, TOP>(k.lut, &k.pos, y, vw, k.hw, k.lim);
        let (src_row, dst_row, luma_row, bx, bw) = (k.src_row, k.dst_row, k.luma_row, k.bx, k.bw);
        let off = y as isize;
        let luma = (luma_row + ((off << sy) * luma_row.pixel_stride::<BD>()) + (bx << sx))
            .slice::<BD>(bw << sx);
        let src = (src_row + off * src_row.pixel_stride::<BD>() + bx).slice::<BD>(bw);
        let mut dst = (dst_row + off * dst_row.pixel_stride::<BD>() + bx).slice_mut::<BD>(bw);
        let (luma, src) = (px::<P>(luma.as_bytes()), px::<P>(src.as_bytes()));
        let dst = px_mut::<P>(dst.as_bytes_mut());
        let (lookup, a, uv) = (k.lookup, k.a, k.uv);
        with_row!(src, luma, dst, (bw, bw << sx, FG_BLOCK_SIZE >> sx), |s, l, d| {
            apply_uv::<P, L, SX, NV>(s, l, d, &g, lookup, a, uv)
        });
    }

    /// The 8 bpc AVX2 kernels in the asm's layout: 32 pixels in one register (two rows of
    /// sixteen when chroma is subsampled horizontally), grain blended in bytes, and words
    /// unpacked within the 128-bit halves only, so that nothing crosses them. The scaling
    /// bytes come from dword gathers at the even words and two bytes lower at the odd ones,
    /// and `pmaddubsw` multiplies them by the grain without masking the bytes above them.
    #[cfg(target_arch = "x86_64")]
    pub mod avx2_8 {
        use std::arch::x86_64::*;

        use super::*;
        use crate::include::common::bitdepth::BitDepth8;

        /// The scaling table with four bytes either side, for the gathers, aligned so that no
        /// gathered dword spans two cache lines: clipped pixels gather the last entry in
        /// every lane, which read in place from an aligned table makes eight split loads.
        pub struct Scaling(crate::align::Align64<[u8; 264]>);

        impl Scaling {
            pub fn new(t: &[u8]) -> Self {
                let mut p = [0; 264];
                p[4..260].copy_from_slice(&t[..256]);
                Self(crate::align::Align64(p))
            }
        }

        #[derive(Clone, Copy)]
        struct K {
            mul: __m256i,
            min: __m256i,
            max: __m256i,
            round5: __m256i,
        }

        // SAFETY (every `unsafe` here): these are only inlined into the `avx2` functions below,
        // and loads and stores go through slices of the length they access.
        impl K {
            #[inline(always)]
            fn new(data: &Rav1dFilmGrainData, max_restricted: u8) -> Self {
                let (min, max) = if data.clip_to_restricted_range { (16, max_restricted) } else { (0, 255) };
                unsafe {
                    Self {
                        mul: _mm256_set1_epi16(1 << (15 - data.scaling_shift)),
                        min: _mm256_set1_epi8(min as i8),
                        max: _mm256_set1_epi8(max as i8),
                        round5: _mm256_set1_epi16(1 << 10),
                    }
                }
            }
        }

        #[inline(always)]
        fn load32<T: AsBytes>(s: &[T]) -> __m256i {
            unsafe { _mm256_loadu_si256(s.as_bytes()[..32].as_ptr().cast()) }
        }

        #[inline(always)]
        fn load16x2<T: AsBytes>(a: &[T], b: &[T]) -> __m256i {
            let (a, b) = (&a.as_bytes()[..16], &b.as_bytes()[..16]);
            unsafe {
                _mm256_inserti128_si256::<1>(
                    _mm256_castsi128_si256(_mm_loadu_si128(a.as_ptr().cast())),
                    _mm_loadu_si128(b.as_ptr().cast()),
                )
            }
        }

        #[inline(always)]
        fn load4x2(a: &[i8], b: &[i8]) -> __m256i {
            let d = |s: &[i8]| i32::from_le_bytes(s[..4].as_bytes().try_into().unwrap());
            unsafe { _mm256_setr_epi32(d(a), 0, 0, 0, d(b), 0, 0, 0) }
        }

        #[inline(always)]
        fn store32(v: __m256i, d: &mut [u8]) {
            unsafe { _mm256_storeu_si256(d[..32].as_mut_ptr().cast(), v) }
        }

        #[inline(always)]
        fn store16x2(v: __m256i, a: &mut [u8], b: &mut [u8]) {
            let (a, b) = (&mut a[..16], &mut b[..16]);
            unsafe {
                _mm_storeu_si128(a.as_mut_ptr().cast(), _mm256_castsi256_si128(v));
                _mm_storeu_si128(b.as_mut_ptr().cast(), _mm256_extracti128_si256::<1>(v));
            }
        }

        #[inline(always)]
        fn pairs(lo: u8, hi: u8) -> __m256i {
            unsafe { _mm256_set1_epi16(i16::from_le_bytes([lo, hi])) }
        }

        /// Words whose low byte is `t[v]`, for words `v` in `0..=255`.
        #[inline(always)]
        fn scaling(t: &Scaling, v: __m256i) -> __m256i {
            unsafe {
                let p = t.0 .0.as_ptr().add(4);
                let even = _mm256_blend_epi16::<0xaa>(v, _mm256_setzero_si256());
                let a = _mm256_i32gather_epi32::<1>(p.cast(), even);
                let b = _mm256_i32gather_epi32::<1>(p.sub(2).cast(), _mm256_srli_epi32::<16>(v));
                _mm256_blend_epi16::<0xaa>(a, b)
            }
        }

        /// The pixels with words `lo` and `hi` plus the noise of the grain bytes `g` scaled by
        /// `sc`, clipped.
        #[inline(always)]
        fn add_noise(lo: __m256i, hi: __m256i, sc: [__m256i; 2], g: __m256i, k: K) -> __m256i {
            unsafe {
                let z = _mm256_setzero_si256();
                let n0 = _mm256_mulhrs_epi16(_mm256_maddubs_epi16(sc[0], _mm256_unpacklo_epi8(g, z)), k.mul);
                let n1 = _mm256_mulhrs_epi16(_mm256_maddubs_epi16(sc[1], _mm256_unpackhi_epi8(g, z)), k.mul);
                let p = _mm256_packus_epi16(_mm256_add_epi16(lo, n0), _mm256_add_epi16(hi, n1));
                _mm256_min_epu8(_mm256_max_epu8(p, k.min), k.max)
            }
        }

        /// The 32 pixels `s` with the noise of the grain `g` and of the scaling of `idx`.
        #[inline(always)]
        fn noise_px(s: __m256i, idx: __m256i, g: __m256i, t: &Scaling, k: K) -> __m256i {
            unsafe {
                let z = _mm256_setzero_si256();
                let sc = [
                    scaling(t, _mm256_unpacklo_epi8(idx, z)),
                    scaling(t, _mm256_unpackhi_epi8(idx, z)),
                ];
                add_noise(_mm256_unpacklo_epi8(s, z), _mm256_unpackhi_epi8(s, z), sc, g, k)
            }
        }

        /// The rounded averages of the byte pairs of `lo` and `hi`.
        #[inline(always)]
        fn avg_pairs(lo: __m256i, hi: __m256i) -> __m256i {
            unsafe {
                let one = _mm256_set1_epi8(1);
                let z = _mm256_setzero_si256();
                _mm256_packus_epi16(
                    _mm256_avg_epu16(_mm256_maddubs_epi16(lo, one), z),
                    _mm256_avg_epu16(_mm256_maddubs_epi16(hi, one), z),
                )
            }
        }

        /// `round2(a * w.0 + b * w.1, 5)` of the byte pairs `w`, saturated to the grain range.
        #[inline(always)]
        fn blend(a: __m256i, b: __m256i, w: __m256i, k: K) -> __m256i {
            unsafe {
                let lo = _mm256_mulhrs_epi16(_mm256_maddubs_epi16(w, _mm256_unpacklo_epi8(a, b)), k.round5);
                let hi = _mm256_mulhrs_epi16(_mm256_maddubs_epi16(w, _mm256_unpackhi_epi8(a, b)), k.round5);
                _mm256_packs_epi16(lo, hi)
            }
        }

        /// `g` with its first dword (of each half for two rows) blended with the left grain `l`
        /// by `w`, whose weights past the overlap keep `g`.
        #[inline(always)]
        fn hblend(l: __m256i, g: __m256i, w: __m256i, k: K, two: bool) -> __m256i {
            let x = blend(l, g, w, k);
            unsafe {
                if two {
                    _mm256_blend_epi32::<0x11>(g, x)
                } else {
                    _mm256_blend_epi32::<0x01>(g, x)
                }
            }
        }

        /// Blocks per picture row borrow: rows are borrowed once for this many blocks.
        pub(super) const CHUNK: usize = 16;

        pub(super) const LUT_LEN: usize = GRAIN_WIDTH * (GRAIN_HEIGHT + 1);

        /// Where the grain of a chunk's blocks starts in the flattened LUT (as [`lut_at`]: own,
        /// left, top, top left), and whether its first block blends with the left; the others
        /// do when the strip overlaps.
        pub(super) struct Chunk {
            pub(super) pos: [[usize; 4]; CHUNK],
            pub(super) xover0: bool,
        }

        /// The offsets of the blocks along the strip.
        pub(super) struct Offsets {
            seed: [c_uint; 2],
            rows: usize,
            cur: [[c_int; 2]; 2],
            overlap: bool,
        }

        impl Offsets {
            #[inline(always)]
            pub(super) fn new(data: &Rav1dFilmGrainData, row_num: usize) -> Self {
                let rows = 1 + (data.overlap_flag && row_num > 0) as usize;
                Self { seed: row_seed(rows, row_num, data), rows, cur: [[0; 2]; 2], overlap: data.overlap_flag }
            }

            /// The positions of the `n` blocks from block `b0` on.
            #[inline(always)]
            pub(super) fn chunk(&mut self, b0: usize, n: usize, sx: bool, sy: bool) -> Chunk {
                let mut c = Chunk { pos: [[0; 4]; CHUNK], xover0: self.overlap && b0 != 0 };
                let w = FG_BLOCK_SIZE >> sx as usize;
                for i in 0..n {
                    if self.overlap && b0 + i != 0 {
                        self.cur[1] = self.cur[0];
                    }
                    for r in 0..self.rows {
                        self.cur[0][r] = get_random_number(8, &mut self.seed[r]);
                    }
                    for (k, (bx, by)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                        let (oy, ox) = lut_at(&self.cur, sx, sy, bx, by);
                        // Already within the row.
                        c.pos[i][k] = oy * GRAIN_WIDTH + ox.min(GRAIN_WIDTH - if bx == 0 { w } else { 4 });
                    }
                }
                c
            }
        }

        struct Strip<'a> {
            lut: &'a GrainLut<i8>,
            wh: __m256i,
            t: &'a Scaling,
            k: K,
            uv: Option<(__m256i, __m256i)>,
        }

        impl Strip<'_> {
            /// `W` grain bytes from `i` (and from the next row in the upper half).
            #[inline(always)]
            fn lut_get<const W: usize>(&self, i: usize) -> __m256i {
                // No grain starts later, and the bound spares the loads their checks.
                let i = i.min(LUT_LEN - GRAIN_WIDTH - 32);
                let lut = self.lut.as_flattened();
                if W == 32 {
                    load32(&lut[i..])
                } else {
                    load16x2(&lut[i..], &lut[i + GRAIN_WIDTH..])
                }
            }

            #[inline(always)]
            fn lut_left(&self, i: usize, dy: usize) -> __m256i {
                let i = i.min(LUT_LEN - GRAIN_WIDTH - 4);
                let lut = self.lut.as_flattened();
                load4x2(&lut[i..], &lut[i + dy..])
            }

            /// The grain of the block at `p` in row `y` (and `y + 1` for sixteen wide blocks),
            /// blended as the overlaps ask, with the top by `wv` for `TOP`.
            #[inline(always)]
            fn grain<const W: usize, const TOP: bool>(
                &self,
                p: &[usize; 4],
                xover: bool,
                y: usize,
                wv: __m256i,
            ) -> __m256i {
                let (two, dy) = (W == 16, (W == 16) as usize * GRAIN_WIDTH);
                let y = y * GRAIN_WIDTH;
                let mut g = self.lut_get::<W>(p[0] + y);
                if xover {
                    g = hblend(self.lut_left(p[1] + y, dy), g, self.wh, self.k, two);
                }
                if TOP {
                    let mut t = self.lut_get::<W>(p[2] + y);
                    if xover {
                        t = hblend(self.lut_left(p[3] + y, dy), t, self.wh, self.k, two);
                    }
                    g = blend(t, g, wv, self.k);
                }
                g
            }

            /// The scaling indices of the chroma `s` over the luma `l`.
            #[inline(always)]
            fn uv_idx(&self, l: __m256i, s: __m256i) -> __m256i {
                let Some((mults, offset)) = self.uv else { return l };
                unsafe {
                    let lo = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(l, s), mults);
                    let hi = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(l, s), mults);
                    _mm256_packus_epi16(
                        _mm256_add_epi16(_mm256_srai_epi16::<6>(lo), offset),
                        _mm256_add_epi16(_mm256_srai_epi16::<6>(hi), offset),
                    )
                }
            }

            #[inline(always)]
            fn px32<const UV: bool>(&self, s: __m256i, l: __m256i, g: __m256i) -> __m256i {
                let idx = if UV { self.uv_idx(l, s) } else { s };
                noise_px(s, idx, g, self.t, self.k)
            }

            /// Block `i` of a row of 32 wide blocks through copies: the last, narrower one.
            #[inline(always)]
            fn part32<const UV: bool, const TOP: bool>(
                &self,
                p: &[usize; 4],
                xover: bool,
                src: &[u8],
                luma: &[u8],
                dst: &mut [u8],
                y: usize,
                wv: __m256i,
            ) {
                let n = src.len();
                let (mut sb, mut lb) = ([0; 32], [0; 32]);
                sb[..n].copy_from_slice(src);
                if UV {
                    lb[..n].copy_from_slice(&luma[..n]);
                }
                let g = self.grain::<32, TOP>(p, xover, y, wv);
                let s = load32(&sb);
                let l = if UV { load32(&lb) } else { s };
                store32(self.px32::<UV>(s, l, g), &mut sb);
                dst.copy_from_slice(&sb[..n]);
            }
        }

        /// One row of 32 wide blocks: luma (`luma` is `src`) or chroma over unsubsampled luma.
        #[target_feature(enable = "avx2")]
        #[inline(never)]
        fn row32<const UV: bool, const TOP: bool, const XOVER: bool>(
            s: &Strip,
            c: &Chunk,
            src: &[u8],
            luma: &[u8],
            dst: &mut [u8],
            y: usize,
            wv: __m256i,
        ) {
            let (mut sc, mut lc, mut dc) = (src.chunks_exact(32), luma.chunks_exact(32), dst.chunks_exact_mut(32));
            let mut i = 0;
            for ((s32, l32), d32) in (&mut sc).zip(&mut lc).zip(&mut dc) {
                let xover = if i == 0 { c.xover0 } else { XOVER };
                let g = s.grain::<32, TOP>(&c.pos[i % CHUNK], xover, y, wv);
                let sv = load32(s32);
                let lv = if UV { load32(l32) } else { sv };
                store32(s.px32::<UV>(sv, lv, g), d32);
                i += 1;
            }
            let (sr, lr, dr) = (sc.remainder(), lc.remainder(), dc.into_remainder());
            if !sr.is_empty() {
                let xover = if i == 0 { c.xover0 } else { XOVER };
                s.part32::<UV, TOP>(&c.pos[i % CHUNK], xover, sr, lr, dr, y, wv);
            }
        }

        /// Two chroma rows of sixteen wide blocks over luma averaged in pairs.
        #[target_feature(enable = "avx2")]
        #[inline(never)]
        fn pair16<const TOP: bool, const XOVER: bool>(
            s: &Strip,
            c: &Chunk,
            src: [&[u8]; 2],
            luma: [&[u8]; 2],
            dst: [&mut [u8]; 2],
            y: usize,
            wv: __m256i,
        ) {
            let [sa, sb] = src;
            let [la, lb] = luma;
            let [da, db] = dst;
            let n = sa.len();
            let it = sa.chunks_exact(16).zip(sb.chunks_exact(16)).zip(la.chunks_exact(32).zip(lb.chunks_exact(32)));
            let mut i = 0;
            for (((sa, sb), (la, lb)), (da, db)) in it.zip(da.chunks_exact_mut(16).zip(db.chunks_exact_mut(16))) {
                let xover = if i == 0 { c.xover0 } else { XOVER };
                let g = s.grain::<16, TOP>(&c.pos[i % CHUNK], xover, y, wv);
                let l = avg_pairs(load16x2(&la[..16], &lb[..16]), load16x2(&la[16..], &lb[16..]));
                let sv = load16x2(sa, sb);
                store16x2(noise_px(sv, s.uv_idx(l, sv), g, s.t, s.k), da, db);
                i += 1;
            }
            let x = 16 * i;
            if x < n {
                let w = n - x;
                let (mut sv, mut l) = ([0; 32], [[0; 32]; 2]);
                sv[..w].copy_from_slice(&sa[x..]);
                sv[16..][..w].copy_from_slice(&sb[x..]);
                l[0][..2 * w].copy_from_slice(&la[2 * x..][..2 * w]);
                l[1][..2 * w].copy_from_slice(&lb[2 * x..][..2 * w]);
                let xover = if i == 0 { c.xover0 } else { XOVER };
                let g = s.grain::<16, TOP>(&c.pos[i % CHUNK], xover, y, wv);
                let lv = avg_pairs(load16x2(&l[0][..16], &l[1][..16]), load16x2(&l[0][16..], &l[1][16..]));
                let v = load32(&sv);
                store32(noise_px(v, s.uv_idx(lv, v), g, s.t, s.k), &mut sv);
                da[x..].copy_from_slice(&sv[..w]);
                db[x..].copy_from_slice(&sv[16..][..w]);
            }
        }

        type Row<'a> = Rav1dPictureDataComponentOffset<'a>;

        /// Row `y` of `row` from column `x`.
        #[inline(always)]
        fn at(row: Row, y: usize, x: usize) -> Row {
            row + y as isize * row.pixel_stride::<BitDepth8>() + x
        }

        /// The rows of 32 wide blocks of a strip, luma or (`UV`) unsubsampled chroma.
        #[inline(always)]
        fn strip32<const UV: bool, const XOVER: bool>(
            s: &Strip,
            dst_row: Row,
            src_row: Row,
            luma_row: Row,
            pw: usize,
            bh: usize,
            ystart: usize,
            wv: [__m256i; 2],
            o: &mut Offsets,
        ) {
            let nb = pw.div_ceil(FG_BLOCK_SIZE);
            for b0 in (0..nb).step_by(CHUNK) {
                let c = o.chunk(b0, cmp::min(CHUNK, nb - b0), false, false);
                let x0 = b0 * FG_BLOCK_SIZE;
                let w = cmp::min(CHUNK * FG_BLOCK_SIZE, pw - x0);
                for y in 0..bh {
                    let src = at(src_row, y, x0).slice::<BitDepth8>(w);
                    let luma = if UV { Some(at(luma_row, y, x0).slice::<BitDepth8>(w)) } else { None };
                    let luma = luma.as_deref().unwrap_or(&src);
                    let mut dst = at(dst_row, y, x0).slice_mut::<BitDepth8>(w);
                    // SAFETY: AVX2 is checked before the callers are chosen.
                    unsafe {
                        if y < ystart {
                            row32::<UV, true, XOVER>(s, &c, &src, luma, &mut dst, y, wv[y]);
                        } else {
                            row32::<UV, false, XOVER>(s, &c, &src, luma, &mut dst, y, s.wh);
                        }
                    }
                }
            }
        }

        #[target_feature(enable = "avx2")]
        pub fn fgy(
            dst_row: Row,
            src_row: Row,
            data: &Rav1dFilmGrainData,
            pw: usize,
            t: &Scaling,
            lut: &GrainLut<i8>,
            bh: usize,
            row_num: usize,
        ) {
            let s = Strip {
                lut,
                wh: _mm256_broadcastq_epi64(_mm_setr_epi8(27, 17, 17, 27, 0, 32, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0)),
                t,
                k: K::new(data, 235),
                uv: None,
            };
            let wv = [pairs(27, 17), pairs(17, 27)];
            let ystart = if data.overlap_flag && row_num != 0 { cmp::min(2, bh) } else { 0 };
            let mut o = Offsets::new(data, row_num);
            if data.overlap_flag {
                strip32::<false, true>(&s, dst_row, src_row, src_row, pw, bh, ystart, wv, &mut o);
            } else {
                strip32::<false, false>(&s, dst_row, src_row, src_row, pw, bh, ystart, wv, &mut o);
            }
        }

        #[target_feature(enable = "avx2")]
        pub fn fguv<const SX: bool, const SY: bool>(
            dst_row: Row,
            src_row: Row,
            data: &Rav1dFilmGrainData,
            pw: usize,
            t: &Scaling,
            lut: &GrainLut<i8>,
            bh: usize,
            row_num: usize,
            luma_row: Row,
            is_uv: bool,
            is_id: bool,
        ) {
            let sy = SY as usize;
            let pl = is_uv as usize;
            let uv = if data.chroma_scaling_from_luma {
                None
            } else {
                Some((
                    pairs(data.uv_luma_mult[pl] as u8, data.uv_mult[pl] as u8),
                    _mm256_set1_epi16(data.uv_offset[pl] as i16),
                ))
            };
            let wh = _mm256_broadcastq_epi64(if SX {
                _mm_setr_epi8(23, 22, 0, 32, 0, 32, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0)
            } else {
                _mm_setr_epi8(27, 17, 17, 27, 0, 32, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0)
            });
            let s = Strip { lut, wh, t, k: K::new(data, if is_id { 235 } else { 240 }), uv };
            let ystart = if data.overlap_flag && row_num != 0 { cmp::min(2 >> sy, bh) } else { 0 };
            let mut o = Offsets::new(data, row_num);
            if !SX {
                let wv = [pairs(27, 17), pairs(17, 27)];
                if data.overlap_flag {
                    strip32::<true, true>(&s, dst_row, src_row, luma_row, pw, bh, ystart, wv, &mut o);
                } else {
                    strip32::<true, false>(&s, dst_row, src_row, luma_row, pw, bh, ystart, wv, &mut o);
                }
                return;
            }
            // The top weights of the first two rows, as a pair in one register.
            let (a, b) = if SY { (pairs(23, 22), pairs(0, 32)) } else { (pairs(27, 17), pairs(17, 27)) };
            let wv = _mm256_blend_epi32::<0xf0>(a, b);
            let nb = pw.div_ceil(16);
            for b0 in (0..nb).step_by(CHUNK) {
                let c = o.chunk(b0, cmp::min(CHUNK, nb - b0), SX, SY);
                let x0 = b0 * 16;
                let w = cmp::min(CHUNK * 16, pw - x0);
                let mut spare = [0; CHUNK * 16];
                for y in (0..bh).step_by(2) {
                    let two = y + 1 < bh;
                    let y1 = y + two as usize;
                    let la = at(luma_row, y << sy, x0 << 1).slice::<BitDepth8>(w << 1);
                    let lb = at(luma_row, y1 << sy, x0 << 1).slice::<BitDepth8>(w << 1);
                    let sa = at(src_row, y, x0).slice::<BitDepth8>(w);
                    let sb = at(src_row, y1, x0).slice::<BitDepth8>(w);
                    let mut da = at(dst_row, y, x0).slice_mut::<BitDepth8>(w);
                    let mut db;
                    let db = if two {
                        db = at(dst_row, y1, x0).slice_mut::<BitDepth8>(w);
                        &mut *db
                    } else {
                        &mut spare[..w]
                    };
                    let (src, luma, dst) = ([&*sa, &*sb], [&*la, &*lb], [&mut *da, db]);
                    match (y == 0 && ystart > 0, data.overlap_flag) {
                        (true, true) => pair16::<true, true>(&s, &c, src, luma, dst, y, wv),
                        (true, false) => pair16::<true, false>(&s, &c, src, luma, dst, y, wv),
                        (false, true) => pair16::<false, true>(&s, &c, src, luma, dst, y, wh),
                        (false, false) => pair16::<false, false>(&s, &c, src, luma, dst, y, wh),
                    }
                }
            }
        }
    }

    /// The 10 and 12 bpc AVX2 kernels: sixteen pixels per register, two for a row of a 32
    /// wide block or for two rows of a sixteen wide one. The scaling bytes come from dword
    /// gathers at the even and odd words, and `pmaddubsw` scales them past the byte above.
    #[cfg(target_arch = "x86_64")]
    pub mod avx2_16 {
        use std::arch::x86_64::*;

        use super::avx2_8::{Chunk, Offsets, CHUNK, LUT_LEN};
        use super::*;
        use crate::include::common::bitdepth::BitDepth16;

        #[derive(Clone, Copy)]
        struct K {
            /// `1 << 14 - scaling_shift` in the low bytes.
            mul: __m256i,
            min: __m256i,
            max: __m256i,
            gmin: __m256i,
            gmax: __m256i,
            /// `bitdepth_max` in the dwords' low words, and in all words.
            idx_mask: __m256i,
            pixel_max: __m256i,
            round5: __m256i,
        }

        // SAFETY (every `unsafe` here): these are only inlined into the `avx2` functions below,
        // loads and stores go through slices of the length they access, and gathers read
        // `t[i..i + 4]` for `i` masked to `bitdepth_max`, which `fgy` and `fguv` check.
        impl K {
            #[inline(always)]
            fn new(data: &Rav1dFilmGrainData, max_restricted: i32, bitdepth_max: i32) -> Self {
                let b8 = (bitdepth_max + 1).ilog2() - 8;
                let (min, max) = if data.clip_to_restricted_range {
                    (16 << b8, max_restricted << b8)
                } else {
                    (0, bitdepth_max)
                };
                let ctr = 128 << b8;
                unsafe {
                    Self {
                        mul: _mm256_set1_epi16(1 << (14 - data.scaling_shift)),
                        min: _mm256_set1_epi16(min as i16),
                        max: _mm256_set1_epi16(max as i16),
                        gmin: _mm256_set1_epi16(-ctr as i16),
                        gmax: _mm256_set1_epi16((ctr - 1) as i16),
                        idx_mask: _mm256_set1_epi32(bitdepth_max),
                        pixel_max: _mm256_set1_epi16(bitdepth_max as i16),
                        round5: _mm256_set1_epi32(16),
                    }
                }
            }
        }

        #[inline(always)]
        fn load<T: AsBytes>(s: &[T]) -> __m256i {
            unsafe { _mm256_loadu_si256(s.as_bytes()[..32].as_ptr().cast()) }
        }

        #[inline(always)]
        fn store(v: __m256i, d: &mut [u16]) {
            unsafe { _mm256_storeu_si256(d.as_bytes_mut()[..32].as_mut_ptr().cast(), v) }
        }

        #[inline(always)]
        fn dword_pairs(lo: i32, hi: i32) -> __m256i {
            unsafe { _mm256_set1_epi32((lo & 0xffff) | hi << 16) }
        }

        /// Words whose low byte is `t[v & bitdepth_max]`.
        #[inline(always)]
        fn scaling(t: &[u8], v: __m256i, k: K) -> __m256i {
            unsafe {
                let p = t.as_ptr().cast();
                let even = _mm256_and_si256(v, k.idx_mask);
                let odd = _mm256_and_si256(_mm256_srli_epi32::<16>(v), k.idx_mask);
                let a = _mm256_i32gather_epi32::<1>(p, even);
                let b = _mm256_i32gather_epi32::<1>(p, odd);
                _mm256_blend_epi16::<0xaa>(a, _mm256_slli_epi32::<16>(b))
            }
        }

        /// The pixels `s` plus the noise of the grain `g` scaled by `t` at `idx`, clipped.
        #[inline(always)]
        fn noise_px(s: __m256i, idx: __m256i, g: __m256i, t: &[u8], k: K) -> __m256i {
            unsafe {
                let m = _mm256_maddubs_epi16(scaling(t, idx, k), k.mul);
                let n = _mm256_mulhrs_epi16(g, _mm256_add_epi16(m, m));
                _mm256_min_epi16(_mm256_max_epi16(_mm256_add_epi16(s, n), k.min), k.max)
            }
        }

        /// `round2(a * w.0 + b * w.1, 5)` of the word pairs `w`, clipped to the grain range.
        #[inline(always)]
        fn blend(a: __m256i, b: __m256i, w: __m256i, k: K) -> __m256i {
            unsafe {
                let lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), w);
                let hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), w);
                let lo = _mm256_srai_epi32::<5>(_mm256_add_epi32(lo, k.round5));
                let hi = _mm256_srai_epi32::<5>(_mm256_add_epi32(hi, k.round5));
                _mm256_min_epi16(_mm256_max_epi16(_mm256_packs_epi32(lo, hi), k.gmin), k.gmax)
            }
        }

        /// `g` with its first two words blended with the left grain `l` by `w`, whose weights
        /// for the second keep `g` when only one overlaps.
        #[inline(always)]
        fn hblend(l: i32, g: __m256i, w: __m128i, k: K) -> __m256i {
            unsafe {
                let g0 = _mm256_castsi256_si128(g);
                let x = _mm_madd_epi16(_mm_unpacklo_epi16(_mm_cvtsi32_si128(l), g0), w);
                let x = _mm_srai_epi32::<5>(_mm_add_epi32(x, _mm256_castsi256_si128(k.round5)));
                let x = _mm_packs_epi32(x, x);
                let x = _mm_min_epi16(_mm_max_epi16(x, _mm256_castsi256_si128(k.gmin)), _mm256_castsi256_si128(k.gmax));
                _mm256_blend_epi32::<0x01>(g, _mm256_castsi128_si256(x))
            }
        }

        struct Strip<'a> {
            lut: &'a GrainLut<i16>,
            wh: __m128i,
            t: &'a [u8],
            k: K,
            uv: Option<(__m256i, __m256i)>,
        }

        impl Strip<'_> {
            #[inline(always)]
            fn lut_get(&self, i: usize) -> __m256i {
                // No grain reaches the end, and the bound spares the load its checks.
                load(&self.lut.as_flattened()[i.min(LUT_LEN - 16)..])
            }

            #[inline(always)]
            fn lut_left(&self, i: usize) -> i32 {
                let l = &self.lut.as_flattened()[i.min(LUT_LEN - 2)..][..2];
                i32::from_le_bytes(l.as_bytes().try_into().unwrap())
            }

            /// The grain of the block at `p` in row `y`, as two vectors: its two halves for
            /// 32 wide blocks (`W`), rows `y` and `y + 1` for sixteen wide ones, blended as the
            /// overlaps ask, with the top by `wv` (per row) for `TOP`.
            #[inline(always)]
            fn grain<const W: usize, const TOP: bool>(
                &self,
                p: &[usize; 4],
                xover: bool,
                y: usize,
                wv: [__m256i; 2],
            ) -> [__m256i; 2] {
                let rows = W == 16;
                let y = y * GRAIN_WIDTH;
                let d = if rows { GRAIN_WIDTH } else { 16 };
                let mut g = [self.lut_get(p[0] + y), self.lut_get(p[0] + y + d)];
                let mut t = if TOP { [self.lut_get(p[2] + y), self.lut_get(p[2] + y + d)] } else { g };
                if xover {
                    g[0] = hblend(self.lut_left(p[1] + y), g[0], self.wh, self.k);
                    if TOP {
                        t[0] = hblend(self.lut_left(p[3] + y), t[0], self.wh, self.k);
                    }
                    if rows {
                        g[1] = hblend(self.lut_left(p[1] + y + d), g[1], self.wh, self.k);
                        if TOP {
                            t[1] = hblend(self.lut_left(p[3] + y + d), t[1], self.wh, self.k);
                        }
                    }
                }
                if TOP {
                    g[0] = blend(t[0], g[0], wv[0], self.k);
                    g[1] = blend(t[1], g[1], wv[1], self.k);
                }
                g
            }

            /// The scaling indices of the chroma `s` over the luma `l`.
            #[inline(always)]
            fn uv_idx(&self, l: __m256i, s: __m256i) -> __m256i {
                let Some((mults, offset)) = self.uv else { return l };
                unsafe {
                    let lo = _mm256_srai_epi32::<6>(_mm256_madd_epi16(_mm256_unpacklo_epi16(l, s), mults));
                    let hi = _mm256_srai_epi32::<6>(_mm256_madd_epi16(_mm256_unpackhi_epi16(l, s), mults));
                    let v = _mm256_add_epi16(_mm256_packs_epi32(lo, hi), offset);
                    _mm256_min_epi16(_mm256_max_epi16(v, _mm256_setzero_si256()), self.k.pixel_max)
                }
            }

            #[inline(always)]
            fn px<const UV: bool>(&self, s: __m256i, l: __m256i, g: __m256i) -> __m256i {
                let idx = if UV { self.uv_idx(l, s) } else { s };
                noise_px(s, idx, g, self.t, self.k)
            }

            /// One block row of 32 pixels in `s`, `l` and `d`.
            #[inline(always)]
            fn block32<const UV: bool>(&self, s: &[u16], l: &[u16], d: &mut [u16], g: [__m256i; 2]) {
                for h in 0..2 {
                    let sv = load(&s[16 * h..]);
                    let lv = if UV { load(&l[16 * h..]) } else { sv };
                    store(self.px::<UV>(sv, lv, g[h]), &mut d[16 * h..]);
                }
            }
        }

        /// One row of 32 wide blocks: luma (`luma` is `src`) or chroma over unsubsampled luma.
        #[target_feature(enable = "avx2")]
        #[inline(never)]
        fn row32<const UV: bool, const TOP: bool, const XOVER: bool>(
            s: &Strip,
            c: &Chunk,
            src: &[u16],
            luma: &[u16],
            dst: &mut [u16],
            y: usize,
            wv: __m256i,
        ) {
            let (mut sc, mut lc, mut dc) = (src.chunks_exact(32), luma.chunks_exact(32), dst.chunks_exact_mut(32));
            let mut i = 0;
            for ((s32, l32), d32) in (&mut sc).zip(&mut lc).zip(&mut dc) {
                let xover = if i == 0 { c.xover0 } else { XOVER };
                let g = s.grain::<32, TOP>(&c.pos[i % CHUNK], xover, y, [wv; 2]);
                s.block32::<UV>(s32, l32, d32, g);
                i += 1;
            }
            let (sr, lr, dr) = (sc.remainder(), lc.remainder(), dc.into_remainder());
            if !sr.is_empty() {
                let n = sr.len();
                let (mut sb, mut lb) = ([0; 32], [0; 32]);
                sb[..n].copy_from_slice(sr);
                if UV {
                    lb[..n].copy_from_slice(&lr[..n]);
                }
                let xover = if i == 0 { c.xover0 } else { XOVER };
                let g = s.grain::<32, TOP>(&c.pos[i % CHUNK], xover, y, [wv; 2]);
                let mut db = [0; 32];
                s.block32::<UV>(&sb, &lb, &mut db, g);
                dr.copy_from_slice(&db[..n]);
            }
        }

        /// The rounded averages of the pairs of the 32 luma pixels at `l`.
        #[inline(always)]
        fn avg_pairs(l: &[u16]) -> __m256i {
            unsafe {
                let one = _mm256_set1_epi16(1);
                let a = _mm256_madd_epi16(load(l), one);
                let b = _mm256_madd_epi16(load(&l[16..]), one);
                let v = _mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi32(a, b));
                _mm256_avg_epu16(v, _mm256_setzero_si256())
            }
        }

        /// Two chroma rows of sixteen wide blocks over luma averaged in pairs.
        #[target_feature(enable = "avx2")]
        #[inline(never)]
        fn pair16<const TOP: bool, const XOVER: bool>(
            s: &Strip,
            c: &Chunk,
            src: [&[u16]; 2],
            luma: [&[u16]; 2],
            dst: [&mut [u16]; 2],
            y: usize,
            wv: [__m256i; 2],
        ) {
            let [sa, sb] = src;
            let [la, lb] = luma;
            let [da, db] = dst;
            let n = sa.len();
            let it = sa.chunks_exact(16).zip(sb.chunks_exact(16)).zip(la.chunks_exact(32).zip(lb.chunks_exact(32)));
            let mut i = 0;
            for (((sa, sb), (la, lb)), (da, db)) in it.zip(da.chunks_exact_mut(16).zip(db.chunks_exact_mut(16))) {
                let xover = if i == 0 { c.xover0 } else { XOVER };
                let g = s.grain::<16, TOP>(&c.pos[i % CHUNK], xover, y, wv);
                let (va, vb) = (load(sa), load(sb));
                store(s.px::<true>(va, avg_pairs(la), g[0]), da);
                store(s.px::<true>(vb, avg_pairs(lb), g[1]), db);
                i += 1;
            }
            let x = 16 * i;
            if x < n {
                let w = n - x;
                let (mut sv, mut l) = ([[0; 16]; 2], [[0; 32]; 2]);
                sv[0][..w].copy_from_slice(&sa[x..]);
                sv[1][..w].copy_from_slice(&sb[x..]);
                l[0][..2 * w].copy_from_slice(&la[2 * x..][..2 * w]);
                l[1][..2 * w].copy_from_slice(&lb[2 * x..][..2 * w]);
                let xover = if i == 0 { c.xover0 } else { XOVER };
                let g = s.grain::<16, TOP>(&c.pos[i % CHUNK], xover, y, wv);
                for r in 0..2 {
                    let v = load(&sv[r]);
                    let out = s.px::<true>(v, avg_pairs(&l[r]), g[r]);
                    store(out, &mut sv[r]);
                }
                da[x..].copy_from_slice(&sv[0][..w]);
                db[x..].copy_from_slice(&sv[1][..w]);
            }
        }

        type Row<'a> = Rav1dPictureDataComponentOffset<'a>;

        #[inline(always)]
        fn at(row: Row, y: usize, x: usize) -> Row {
            row + y as isize * row.pixel_stride::<BitDepth16>() + x
        }

        #[inline(always)]
        fn strip32<const UV: bool, const XOVER: bool>(
            s: &Strip,
            dst_row: Row,
            src_row: Row,
            luma_row: Row,
            pw: usize,
            bh: usize,
            ystart: usize,
            wv: [__m256i; 2],
            o: &mut Offsets,
        ) {
            let nb = pw.div_ceil(FG_BLOCK_SIZE);
            for b0 in (0..nb).step_by(CHUNK) {
                let c = o.chunk(b0, cmp::min(CHUNK, nb - b0), false, false);
                let x0 = b0 * FG_BLOCK_SIZE;
                let w = cmp::min(CHUNK * FG_BLOCK_SIZE, pw - x0);
                for y in 0..bh {
                    let src = at(src_row, y, x0).slice::<BitDepth16>(w);
                    let luma = if UV { Some(at(luma_row, y, x0).slice::<BitDepth16>(w)) } else { None };
                    let luma = luma.as_deref().unwrap_or(&src);
                    let mut dst = at(dst_row, y, x0).slice_mut::<BitDepth16>(w);
                    // SAFETY: AVX2 is checked before the callers are chosen.
                    unsafe {
                        if y < ystart {
                            row32::<UV, true, XOVER>(s, &c, &src, luma, &mut dst, y, wv[y]);
                        } else {
                            row32::<UV, false, XOVER>(s, &c, &src, luma, &mut dst, y, wv[0]);
                        }
                    }
                }
            }
        }

        /// The entries of `scaling` up to `bitdepth_max` copied four bytes into an aligned
        /// table with room for the gathers past them, as [`avx2_8::Scaling`].
        macro_rules! with_table {
            ($scaling:expr, $bitdepth_max:expr, |$t:ident| $e:expr) => {{
                let scaling: &[u8] = $scaling;
                let n = $bitdepth_max as usize + 1;
                if n <= 1024 {
                    let mut p = crate::align::Align64([0; 1024 + 8]);
                    p.0[4..][..n].copy_from_slice(&scaling[..n]);
                    let $t = &p.0[4..];
                    $e
                } else {
                    let mut p = crate::align::Align64([0; 4096 + 8]);
                    p.0[4..][..n].copy_from_slice(&scaling[..n]);
                    let $t = &p.0[4..];
                    $e
                }
            }};
        }

        #[target_feature(enable = "avx2")]
        pub fn fgy(
            dst_row: Row,
            src_row: Row,
            data: &Rav1dFilmGrainData,
            pw: usize,
            scaling: &[u8],
            lut: &GrainLut<i16>,
            bh: usize,
            row_num: usize,
            bitdepth_max: c_int,
        ) {
            with_table!(scaling, bitdepth_max, |t| {
                let s = Strip {
                    lut,
                    wh: _mm_setr_epi16(27, 17, 17, 27, 0, 32, 0, 32),
                    t,
                    k: K::new(data, 235, bitdepth_max),
                    uv: None,
                };
                let wv = [dword_pairs(27, 17), dword_pairs(17, 27)];
                let ystart = if data.overlap_flag && row_num != 0 { cmp::min(2, bh) } else { 0 };
                let mut o = Offsets::new(data, row_num);
                if data.overlap_flag {
                    strip32::<false, true>(&s, dst_row, src_row, src_row, pw, bh, ystart, wv, &mut o);
                } else {
                    strip32::<false, false>(&s, dst_row, src_row, src_row, pw, bh, ystart, wv, &mut o);
                }
            })
        }

        #[target_feature(enable = "avx2")]
        pub fn fguv<const SX: bool, const SY: bool>(
            dst_row: Row,
            src_row: Row,
            data: &Rav1dFilmGrainData,
            pw: usize,
            scaling: &[u8],
            lut: &GrainLut<i16>,
            bh: usize,
            row_num: usize,
            luma_row: Row,
            is_uv: bool,
            is_id: bool,
            bitdepth_max: c_int,
        ) {
            let sy = SY as usize;
            let pl = is_uv as usize;
            let b8 = (bitdepth_max + 1).ilog2() - 8;
            let uv = if data.chroma_scaling_from_luma {
                None
            } else {
                Some((
                    dword_pairs(data.uv_luma_mult[pl], data.uv_mult[pl]),
                    _mm256_set1_epi16((data.uv_offset[pl] << b8) as i16),
                ))
            };
            let wh = if SX {
                _mm_setr_epi16(23, 22, 0, 32, 0, 32, 0, 32)
            } else {
                _mm_setr_epi16(27, 17, 17, 27, 0, 32, 0, 32)
            };
            with_table!(scaling, bitdepth_max, |t| {
                let s = Strip { lut, wh, t, k: K::new(data, if is_id { 235 } else { 240 }, bitdepth_max), uv };
                let ystart = if data.overlap_flag && row_num != 0 { cmp::min(2 >> sy, bh) } else { 0 };
                let mut o = Offsets::new(data, row_num);
                if !SX {
                    let wv = [dword_pairs(27, 17), dword_pairs(17, 27)];
                    if data.overlap_flag {
                        strip32::<true, true>(&s, dst_row, src_row, luma_row, pw, bh, ystart, wv, &mut o);
                    } else {
                        strip32::<true, false>(&s, dst_row, src_row, luma_row, pw, bh, ystart, wv, &mut o);
                    }
                    return;
                }
                let wv = if SY {
                    [dword_pairs(23, 22), dword_pairs(0, 32)]
                } else {
                    [dword_pairs(27, 17), dword_pairs(17, 27)]
                };
                let nb = pw.div_ceil(16);
                for b0 in (0..nb).step_by(CHUNK) {
                    let c = o.chunk(b0, cmp::min(CHUNK, nb - b0), SX, SY);
                    let x0 = b0 * 16;
                    let w = cmp::min(CHUNK * 16, pw - x0);
                    let mut spare = [0; CHUNK * 16];
                    for y in (0..bh).step_by(2) {
                        let two = y + 1 < bh;
                        let y1 = y + two as usize;
                        let la = at(luma_row, y << sy, x0 << 1).slice::<BitDepth16>(w << 1);
                        let lb = at(luma_row, y1 << sy, x0 << 1).slice::<BitDepth16>(w << 1);
                        let sa = at(src_row, y, x0).slice::<BitDepth16>(w);
                        let sb = at(src_row, y1, x0).slice::<BitDepth16>(w);
                        let mut da = at(dst_row, y, x0).slice_mut::<BitDepth16>(w);
                        let mut db;
                        let db = if two {
                            db = at(dst_row, y1, x0).slice_mut::<BitDepth16>(w);
                            &mut *db
                        } else {
                            &mut spare[..w]
                        };
                        let (src, luma, dst) = ([&*sa, &*sb], [&*la, &*lb], [&mut *da, db]);
                        match (y == 0 && ystart > 0, data.overlap_flag) {
                            (true, true) => pair16::<true, true>(&s, &c, src, luma, dst, y, wv),
                            (true, false) => pair16::<true, false>(&s, &c, src, luma, dst, y, wv),
                            (false, true) => pair16::<false, true>(&s, &c, src, luma, dst, y, wv),
                            (false, false) => pair16::<false, false>(&s, &c, src, luma, dst, y, wv),
                        }
                    }
                }
            })
        }
    }

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

    /// `t` copied with three bytes after it for [`Gather`].
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn padded<const N: usize>(t: &[u8]) -> [u8; N] {
        let mut p = [0; N];
        p[..t.len()].copy_from_slice(t);
        p
    }

    /// Calls `$f::<BD, P, _>` with the lookup of this tier as the `lookup` argument.
    macro_rules! with_lookup {
        ($BD:ty, $P:ty, $avx2:expr, $scaling:expr, |$l:ident| $e:expr) => {{
            let scaling: &[u8] = $scaling;
            #[cfg(target_arch = "x86_64")]
            if $avx2 {
                if <$BD>::BITDEPTH == 8 {
                    let t = padded::<{ 256 + 4 }>(scaling);
                    let $l = &Gather(&t, 255);
                    return $e;
                }
                let t = padded::<{ 4096 + 4 }>(scaling);
                let $l = &Gather(&t, 4095);
                return $e;
            }
            #[cfg(target_arch = "aarch64")]
            if <$BD>::BITDEPTH == 8 {
                let $l = &Tbl(Lut256::new(scaling.try_into().unwrap()));
                return $e;
            }
            let _ = $avx2;
            let $l = &PerLane(scaling);
            $e
        }};
    }

    #[cfg_attr(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature(enable = "sse4.1")
    )]
    fn fgy_128<BD: BitDepth, P: Px>(
        dst_row: Rav1dPictureDataComponentOffset,
        src_row: Rav1dPictureDataComponentOffset,
        data: &Rav1dFilmGrainData,
        pw: usize,
        scaling: &[u8],
        lut: &GrainLut<P::E>,
        bh: usize,
        row_num: usize,
        bitdepth_max: c_int,
    ) {
        with_lookup!(BD, P, false, scaling, |l| fgy::<BD, P, _>(
            dst_row,
            src_row,
            data,
            pw,
            l,
            lut,
            bh,
            row_num,
            bitdepth_max
        ))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    fn fgy_avx2<BD: BitDepth, P: Px>(
        dst_row: Rav1dPictureDataComponentOffset,
        src_row: Rav1dPictureDataComponentOffset,
        data: &Rav1dFilmGrainData,
        pw: usize,
        scaling: &[u8],
        lut: &GrainLut<P::E>,
        bh: usize,
        row_num: usize,
        bitdepth_max: c_int,
    ) {
        with_lookup!(BD, P, true, scaling, |l| fgy::<BD, P, _>(
            dst_row,
            src_row,
            data,
            pw,
            l,
            lut,
            bh,
            row_num,
            bitdepth_max
        ))
    }

    /// # Safety
    ///
    /// Must be called by [`fgy_32x32xn::Fn::call`], on x86 once the CPU flags hold SSE4.1, and
    /// AVX2 for `AVX2`.
    #[allow(unused_unsafe)] // Safe calls on aarch64.
    pub unsafe extern "C" fn fgy_32x32xn_simd_erased<BD: BitDepth, const AVX2: bool>(
        _dst_row_ptr: *mut DynPixel,
        _src_row_ptr: *const DynPixel,
        _stride: ptrdiff_t,
        data: &Dav1dFilmGrainData,
        pw: usize,
        scaling: *const DynScaling,
        grain_lut: *const GrainLut<DynEntry>,
        bh: c_int,
        row_num: c_int,
        bitdepth_max: c_int,
        dst_row: FFISafeRav1dPictureDataComponentOffset,
        src_row: FFISafeRav1dPictureDataComponentOffset,
    ) {
        // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `fgy_32x32xn::Fn::call`.
        let [dst_row, src_row] =
            [dst_row, src_row].map(|it| unsafe { FFISafe::from_with_offset(it) });
        let data = &data.clone().into();
        // SAFETY: Casting back to the original type from the `fn` ptr call.
        let scaling: &BD::Scaling = unsafe { &*scaling.cast() };
        let (bh, row_num) = (bh as usize, row_num as usize);
        #[cfg(target_arch = "x86_64")]
        if AVX2 && BD::BITDEPTH == 8 {
            // SAFETY: The 8 bpc LUT, of `i8`.
            let lut = unsafe { &*grain_lut.cast::<GrainLut<i8>>() };
            let t = avx2_8::Scaling::new(scaling.as_ref());
            // SAFETY: AVX2 is checked before `AVX2` is chosen.
            return unsafe { avx2_8::fgy(dst_row, src_row, data, pw, &t, lut, bh, row_num) };
        }
        #[cfg(target_arch = "x86_64")]
        if AVX2 {
            // SAFETY: The high bit depth LUT, of `i16`.
            let lut = unsafe { &*grain_lut.cast::<GrainLut<i16>>() };
            // SAFETY: AVX2 is checked before `AVX2` is chosen.
            return unsafe {
                avx2_16::fgy(dst_row, src_row, data, pw, scaling.as_ref(), lut, bh, row_num, bitdepth_max)
            };
        }
        with_px!(BD, |P| {
            // SAFETY: Casting back to the original type from the `fn` ptr call, `P::E` being `BD::Entry`.
            let lut: &GrainLut<<P as Px>::E> = unsafe { &*grain_lut.cast() };
            let scaling = scaling.as_ref();
            // SAFETY: SSE4.1 is checked before this `fn` is chosen on x86, and AVX2 before
            // `AVX2` is; elsewhere it is safe.
            unsafe {
                #[cfg(target_arch = "x86_64")]
                if AVX2 {
                    return fgy_avx2::<BD, P>(
                        dst_row,
                        src_row,
                        data,
                        pw,
                        scaling,
                        lut,
                        bh,
                        row_num,
                        bitdepth_max,
                    );
                }
                fgy_128::<BD, P>(
                    dst_row,
                    src_row,
                    data,
                    pw,
                    scaling,
                    lut,
                    bh,
                    row_num,
                    bitdepth_max,
                )
            }
        })
    }

    macro_rules! fguv_tier {
        ($(#[$m:meta])* $name:ident, $avx2:expr) => {
            $(#[$m])*
            fn $name<BD: BitDepth, P: Px, const SX: bool, const SY: bool>(
                dst_row: Rav1dPictureDataComponentOffset,
                src_row: Rav1dPictureDataComponentOffset,
                data: &Rav1dFilmGrainData,
                pw: usize,
                scaling: &[u8],
                lut: &GrainLut<P::E>,
                bh: usize,
                row_num: usize,
                luma_row: Rav1dPictureDataComponentOffset,
                is_uv: bool,
                is_id: bool,
                bitdepth_max: c_int,
            ) {
                with_lookup!(BD, P, $avx2, scaling, |l| if SX {
                    fguv::<BD, P, _, true, SY, 2>(
                        dst_row, src_row, data, pw, l, lut, bh, row_num, luma_row, is_uv, is_id,
                        bitdepth_max,
                    )
                } else {
                    fguv::<BD, P, _, false, SY, 4>(
                        dst_row, src_row, data, pw, l, lut, bh, row_num, luma_row, is_uv, is_id,
                        bitdepth_max,
                    )
                })
            }
        };
    }

    fguv_tier!(
        #[cfg_attr(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature(enable = "sse4.1")
        )]
        fguv_128,
        false
    );

    #[cfg(target_arch = "x86_64")]
    fguv_tier!(
        #[target_feature(enable = "avx2")]
        fguv_avx2,
        true
    );

    /// # Safety
    ///
    /// Must be called by [`fguv_32x32xn::Fn::call`], on x86 once the CPU flags hold SSE4.1, and
    /// AVX2 for `AVX2`.
    #[allow(unused_unsafe)] // Safe calls on aarch64.
    pub unsafe extern "C" fn fguv_32x32xn_simd_erased<
        BD: BitDepth,
        const NM: usize,
        const IS_SX: bool,
        const IS_SY: bool,
        const AVX2: bool,
    >(
        _dst_row_ptr: *mut DynPixel,
        _src_row_ptr: *const DynPixel,
        _stride: ptrdiff_t,
        data: &Dav1dFilmGrainData,
        pw: usize,
        scaling: *const DynScaling,
        grain_lut: *const GrainLut<DynEntry>,
        bh: c_int,
        row_num: c_int,
        _luma_row_ptr: *const DynPixel,
        _luma_stride: ptrdiff_t,
        uv_pl: c_int,
        is_id: c_int,
        bitdepth_max: c_int,
        dst_row: FFISafeRav1dPictureDataComponentOffset,
        src_row: FFISafeRav1dPictureDataComponentOffset,
        luma_row: FFISafeRav1dPictureDataComponentOffset,
    ) {
        let [dst_row, src_row, luma_row] = [dst_row, src_row, luma_row].map(|row| {
            // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `fguv_32x32xn::Fn::call`.
            unsafe { FFISafe::from_with_offset(row) }
        });
        let data = &data.clone().into();
        // SAFETY: Casting back to the original type from the `fn` ptr call.
        let scaling: &BD::Scaling = unsafe { &*scaling.cast() };
        let (bh, row_num) = (bh as usize, row_num as usize);
        let (is_uv, is_id) = (uv_pl != 0, is_id != 0);
        #[cfg(target_arch = "x86_64")]
        if AVX2 && BD::BITDEPTH == 8 {
            // SAFETY: The 8 bpc LUT, of `i8`.
            let lut = unsafe { &*grain_lut.cast::<GrainLut<i8>>() };
            let t = avx2_8::Scaling::new(scaling.as_ref());
            // SAFETY: AVX2 is checked before `AVX2` is chosen.
            return unsafe {
                avx2_8::fguv::<IS_SX, IS_SY>(
                    dst_row, src_row, data, pw, &t, lut, bh, row_num, luma_row, is_uv, is_id,
                )
            };
        }
        #[cfg(target_arch = "x86_64")]
        if AVX2 {
            // SAFETY: The high bit depth LUT, of `i16`.
            let lut = unsafe { &*grain_lut.cast::<GrainLut<i16>>() };
            // SAFETY: AVX2 is checked before `AVX2` is chosen.
            return unsafe {
                avx2_16::fguv::<IS_SX, IS_SY>(
                    dst_row, src_row, data, pw, scaling.as_ref(), lut, bh, row_num, luma_row,
                    is_uv, is_id, bitdepth_max,
                )
            };
        }
        with_px!(BD, |P| {
            // SAFETY: Casting back to the original type from the `fn` ptr call, `P::E` being `BD::Entry`.
            let lut: &GrainLut<<P as Px>::E> = unsafe { &*grain_lut.cast() };
            let scaling = scaling.as_ref();
            // SAFETY: SSE4.1 is checked before this `fn` is chosen on x86, and AVX2 before
            // `AVX2` is; elsewhere it is safe.
            unsafe {
                #[cfg(target_arch = "x86_64")]
                if AVX2 {
                    return fguv_avx2::<BD, P, IS_SX, IS_SY>(
                        dst_row,
                        src_row,
                        data,
                        pw,
                        scaling,
                        lut,
                        bh,
                        row_num,
                        luma_row,
                        is_uv,
                        is_id,
                        bitdepth_max,
                    );
                }
                fguv_128::<BD, P, IS_SX, IS_SY>(
                    dst_row,
                    src_row,
                    data,
                    pw,
                    scaling,
                    lut,
                    bh,
                    row_num,
                    luma_row,
                    is_uv,
                    is_id,
                    bitdepth_max,
                )
            }
        })
    }

    /// Eight steps of the grain LFSR from the 16-bit state `r`: the 11-bit number each step
    /// draws, and the state after. The bits fed in are `r[i] ^ r[i + 1] ^ r[i + 3] ^ r[i + 12]`,
    /// where `r[16 + j]` is the `j`th bit fed in: for the last four, the first four.
    #[inline(always)]
    fn random8(r: c_uint) -> ([usize; 8], c_uint) {
        let t = r ^ r >> 1 ^ r >> 3 ^ r >> 12;
        let fed = (t ^ t << 4) & 0xff;
        let c = r | fed << 16;
        let mut v = [0; 8];
        for k in 0..8 {
            v[k] = (c >> (k + 6) & 0x7ff) as usize;
        }
        (v, c >> 8)
    }

    /// The `(h, w)` Gaussian numbers from `seed`, scaled down by `shift`.
    #[inline(always)]
    fn gaussian<P: Px>(buf: &mut GrainLut<P::E>, mut seed: c_uint, shift: u8, (h, w): (usize, usize)) {
        let g = |v: usize| P::i2e(round2(DAV1D_GAUSSIAN_SEQUENCE[v] as i32, shift));
        for row in &mut buf[..h] {
            let row = &mut row[..w];
            let mut x = 0;
            while x + 8 <= w {
                let (v, next) = random8(seed);
                seed = next;
                for k in 0..8 {
                    row[x + k] = g(v[k]);
                }
                x += 8;
            }
            for e in &mut row[x..] {
                *e = g(get_random_number(11, &mut seed) as usize);
            }
        }
    }

    /// The auto-regression over the `(h, w)` grain: per row, the taps of the rows above (and
    /// the luma's) summed eight lanes at a time, then the `LAG` taps on the row in turn.
    #[inline(always)]
    fn auto_regress<P: Px, const LAG: usize>(
        buf: &mut GrainLut<P::E>,
        luma: Option<(&GrainLut<P::E>, bool, bool)>,
        coeffs: &[i8],
        (h, w): (usize, usize),
        shift: u8,
        (min, max): (i32, i32),
    ) {
        const PAD: usize = AR_PAD;
        let n = w - 2 * PAD;
        let row_taps = LAG * (2 * LAG + 1);
        let cur = &coeffs[row_taps..][..LAG];
        for y in 0..h - PAD {
            let mut acc = [0; GRAIN_WIDTH];
            for x0 in (0..n).step_by(8) {
                let x = cmp::min(x0, n - 8);
                let mut s = [I32x4::splat(0); 2];
                for dy in 0..LAG {
                    let row = &buf[y + PAD - LAG + dy][x + PAD - LAG..];
                    for dx in 0..2 * LAG + 1 {
                        let k = I16x8::splat(coeffs[dy * (2 * LAG + 1) + dx] as i16);
                        let [p0, p1] = P::grain8(&row[dx..]).mul_wide(k);
                        s = [s[0].add(p0), s[1].add(p1)];
                    }
                }
                s[0].store(&mut acc[x..]);
                s[1].store(&mut acc[x + 4..]);
            }
            if let Some((luma, sx, sy)) = luma {
                let (sx, sy) = (sx as usize, sy as usize);
                let luma_c = coeffs[row_taps + LAG] as i32;
                for x in 0..n {
                    let (ly, lx) = ((y << sy) + PAD, (x << sx) + PAD);
                    let mut l = 0;
                    for i in 0..1 + sy {
                        for j in 0..1 + sx {
                            l += P::e2i(luma[ly + i][lx + j]);
                        }
                    }
                    acc[x] += round2(l, (sx + sy) as u8) * luma_c;
                }
            }
            let row = &mut buf[y + PAD];
            // The last three grains in registers: an array went through the stack.
            let k = |i: usize| if i < LAG { cur[LAG - 1 - i] as i32 } else { 0 };
            let (k1, k2, k3) = (k(0), k(1), k(2));
            let (mut g1, mut g2, mut g3) = (
                P::e2i(row[PAD - 1]),
                P::e2i(row[PAD - 2]),
                P::e2i(row[PAD - 3]),
            );
            // All but the newest grain's product are summed a pixel ahead, off the chain.
            let rnd = 1 << shift >> 1;
            // `grain + (sum >> shift)` as `(grain << shift) + sum >> shift`.
            let base = |x: usize, row: &[P::E]| (P::e2i(row[x + PAD]) << shift) + acc[x] + rnd;
            let mut part = base(0, row) + k3 * g3 + k2 * g2;
            for x in 0..n {
                let g = ((part + k1 * g1) >> shift).max(min).min(max);
                row[x + PAD] = P::i2e(g);
                part = base(x + 1, row) + k3 * g2 + k2 * g1;
                (g3, g2, g1) = (g2, g1, g);
            }
            let _ = g3;
        }
    }

    #[cfg_attr(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature(enable = "sse4.1")
    )]
    fn generate_grain<P: Px>(
        buf: &mut GrainLut<P::E>,
        luma: Option<(&GrainLut<P::E>, bool, bool)>,
        data: &Rav1dFilmGrainData,
        is_uv: bool,
        bitdepth_max: c_int,
    ) {
        let bitdepth_min_8 = ((bitdepth_max + 1).ilog2() - 8) as u8;
        let shift = 4 - bitdepth_min_8 + data.grain_scale_shift;
        let ctr = 128 << bitdepth_min_8;
        let (seed, coeffs, dims) = match luma {
            None => (data.seed, &data.ar_coeffs_y[..], (GRAIN_HEIGHT, GRAIN_WIDTH)),
            Some((_, sx, sy)) => (
                data.seed ^ if is_uv { 0x49d8 } else { 0xb524 },
                &data.ar_coeffs_uv[is_uv as usize][..],
                (
                    if sy { SUB_GRAIN_HEIGHT } else { GRAIN_HEIGHT },
                    if sx { SUB_GRAIN_WIDTH } else { GRAIN_WIDTH },
                ),
            ),
        };
        gaussian::<P>(buf, seed, shift, dims);
        let lim = (-ctr, ctr - 1);
        let ar = data.ar_coeff_shift;
        match data.ar_coeff_lag & 3 {
            0 if luma.is_none() => {}
            0 => auto_regress::<P, 0>(buf, luma, coeffs, dims, ar, lim),
            1 => auto_regress::<P, 1>(buf, luma, coeffs, dims, ar, lim),
            2 => auto_regress::<P, 2>(buf, luma, coeffs, dims, ar, lim),
            _ => auto_regress::<P, 3>(buf, luma, coeffs, dims, ar, lim),
        }
    }

    /// # Safety
    ///
    /// Must be called by [`generate_grain_y::Fn::call`], on x86 once the CPU flags hold SSE4.1.
    #[allow(unused_unsafe)] // Safe calls on aarch64.
    pub unsafe extern "C" fn generate_grain_y_simd_erased<BD: BitDepth>(
        buf: *mut GrainLut<DynEntry>,
        data: &Dav1dFilmGrainData,
        bitdepth_max: c_int,
    ) {
        let data = &data.clone().into();
        with_px!(BD, |P| {
            // SAFETY: Casting back to the original type from `generate_grain_y::Fn::call`.
            let buf: &mut GrainLut<<P as Px>::E> = unsafe { &mut *buf.cast() };
            // SAFETY: SSE4.1 is checked before this `fn` is chosen on x86; elsewhere it is safe.
            unsafe { generate_grain::<P>(buf, None, data, false, bitdepth_max) }
        })
    }

    /// # Safety
    ///
    /// Must be called by [`generate_grain_uv::Fn::call`], on x86 once the CPU flags hold SSE4.1.
    #[allow(unused_unsafe)] // Safe calls on aarch64.
    pub unsafe extern "C" fn generate_grain_uv_simd_erased<
        BD: BitDepth,
        const NM: usize,
        const IS_SUBX: bool,
        const IS_SUBY: bool,
    >(
        buf: *mut GrainLut<DynEntry>,
        buf_y: *const GrainLut<DynEntry>,
        data: &Dav1dFilmGrainData,
        uv: intptr_t,
        bitdepth_max: c_int,
    ) {
        let data = &data.clone().into();
        with_px!(BD, |P| {
            // SAFETY: Casting back to the original type from `generate_grain_uv::Fn::call`.
            let buf: &mut GrainLut<<P as Px>::E> = unsafe { &mut *buf.cast() };
            // SAFETY: Casting back to the original type from `generate_grain_uv::Fn::call`.
            let buf_y: &GrainLut<<P as Px>::E> = unsafe { &*buf_y.cast() };
            let luma = Some((buf_y, IS_SUBX, IS_SUBY));
            // SAFETY: SSE4.1 is checked before this `fn` is chosen on x86; elsewhere it is safe.
            unsafe { generate_grain::<P>(buf, luma, data, uv != 0, bitdepth_max) }
        })
    }
}


impl Rav1dFilmGrainDSPContext {
    pub const fn default<BD: BitDepth>() -> Self {
        Self {
            generate_grain_y: generate_grain_y::Fn::new(generate_grain_y_c_erased::<BD>),
            generate_grain_uv: enum_map!(Rav1dPixelLayoutSubSampled => generate_grain_uv::Fn; match key {
                I420 => generate_grain_uv::Fn::new(generate_grain_uv_c_erased::<BD, 420, true, true>),
                I422 => generate_grain_uv::Fn::new(generate_grain_uv_c_erased::<BD, 422, true, false>),
                I444 => generate_grain_uv::Fn::new(generate_grain_uv_c_erased::<BD, 444, false, false>),
            }),
            fgy_32x32xn: fgy_32x32xn::Fn::new(fgy_32x32xn_c_erased::<BD>),
            fguv_32x32xn: enum_map!(Rav1dPixelLayoutSubSampled => fguv_32x32xn::Fn; match key {
                I420 => fguv_32x32xn::Fn::new(fguv_32x32xn_c_erased::<BD, 420, true, true>),
                I422 => fguv_32x32xn::Fn::new(fguv_32x32xn_c_erased::<BD, 422, true, false>),
                I444 => fguv_32x32xn::Fn::new(fguv_32x32xn_c_erased::<BD, 444, false, false>),
            }),
        }
    }

    /// The Rust SIMD kernels: eight lanes, with the scaling looked up by AVX2 gathers where
    /// they are fast.
    #[inline(always)]
    const fn init<BD: BitDepth>(mut self, flags: CpuFlags) -> Self {
        use self::rust_simd::*;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if !flags.contains(CpuFlags::SSE41) {
            return self;
        }
        #[cfg(target_arch = "x86_64")]
        let gather = flags.contains(CpuFlags::AVX2) && !flags.contains(CpuFlags::SLOW_GATHER);
        #[cfg(not(target_arch = "x86_64"))]
        let gather = false;
        let _ = flags;
        if !crate::simd128::native() {
            return self;
        }
        self.generate_grain_y = generate_grain_y::Fn::new(generate_grain_y_simd_erased::<BD>);
        self.generate_grain_uv = enum_map!(Rav1dPixelLayoutSubSampled => generate_grain_uv::Fn; match key {
            I420 => generate_grain_uv::Fn::new(generate_grain_uv_simd_erased::<BD, 420, true, true>),
            I422 => generate_grain_uv::Fn::new(generate_grain_uv_simd_erased::<BD, 422, true, false>),
            I444 => generate_grain_uv::Fn::new(generate_grain_uv_simd_erased::<BD, 444, false, false>),
        });
        if gather {
            self.fgy_32x32xn = fgy_32x32xn::Fn::new(fgy_32x32xn_simd_erased::<BD, true>);
            self.fguv_32x32xn = enum_map!(Rav1dPixelLayoutSubSampled => fguv_32x32xn::Fn; match key {
                I420 => fguv_32x32xn::Fn::new(fguv_32x32xn_simd_erased::<BD, 420, true, true, true>),
                I422 => fguv_32x32xn::Fn::new(fguv_32x32xn_simd_erased::<BD, 422, true, false, true>),
                I444 => fguv_32x32xn::Fn::new(fguv_32x32xn_simd_erased::<BD, 444, false, false, true>),
            });
        } else {
            self.fgy_32x32xn = fgy_32x32xn::Fn::new(fgy_32x32xn_simd_erased::<BD, false>);
            self.fguv_32x32xn = enum_map!(Rav1dPixelLayoutSubSampled => fguv_32x32xn::Fn; match key {
                I420 => fguv_32x32xn::Fn::new(fguv_32x32xn_simd_erased::<BD, 420, true, true, false>),
                I422 => fguv_32x32xn::Fn::new(fguv_32x32xn_simd_erased::<BD, 422, true, false, false>),
                I444 => fguv_32x32xn::Fn::new(fguv_32x32xn_simd_erased::<BD, 444, false, false, false>),
            });
        }
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
    use crate::align::{Align64, AlignedVec64, ArrayDefault};
    use crate::cpu::{rav1d_get_cpu_flags, rav1d_init_cpu};
    use crate::include::common::bitdepth::{BitDepth16, BitDepth8};

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 32) as u32
        }

        fn range(&mut self, lo: c_int, hi: c_int) -> c_int {
            lo + (self.next() % (hi - lo + 1) as u32) as c_int
        }

        fn bit(&mut self) -> bool {
            self.next() & 1 != 0
        }
    }

    const STRIDE: usize = 192;
    const ROWS: usize = 64;

    fn contexts<BD: BitDepth>() -> Vec<(&'static str, Rav1dFilmGrainDSPContext)> {
        rav1d_init_cpu();
        let flags = rav1d_get_cpu_flags();
        let new = || Rav1dFilmGrainDSPContext::default::<BD>();
        #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
        let mut c = vec![
            ("scalar", new()),
            ("rust simd", new().init::<BD>(flags)),
        ];
        #[cfg(target_arch = "x86_64")]
        if flags.contains(CpuFlags::AVX2) {
            c.push((
                "rust sse4.1",
                new().init::<BD>(flags.difference(CpuFlags::AVX2)),
            ));
        }
        c
    }

    struct Plane<BD: BitDepth> {
        orig: Vec<BD::Pixel>,
        comp: Rav1dPictureDataComponent,
        _buf: AlignedVec64<BD::Pixel>,
    }

    impl<BD: BitDepth> Plane<BD> {
        fn new(rng: &mut Rng, max: c_int) -> Self {
            let orig: Vec<BD::Pixel> = (0..STRIDE * ROWS)
                .map(|_| rng.range(0, max).as_::<BD::Pixel>())
                .collect();
            let mut buf = AlignedVec64::<BD::Pixel>::new();
            buf.resize(STRIDE * ROWS, 0.as_::<BD::Pixel>());
            buf.as_mut_slice().copy_from_slice(&orig);
            let comp = Rav1dPictureDataComponent::wrap_buf::<BD>(buf.as_mut_slice(), STRIDE);
            Self {
                orig,
                comp,
                _buf: buf,
            }
        }

        fn reset(&self) {
            self.comp
                .slice_mut::<BD, _>((0.., ..STRIDE * ROWS))
                .copy_from_slice(&self.orig);
        }

        fn pixels(&self) -> Vec<BD::Pixel> {
            self.comp.slice::<BD, _>((0.., ..STRIDE * ROWS)).to_vec()
        }
    }

    struct Case<BD: BitDepth> {
        data: Rav1dFilmGrainData,
        scaling: BD::Scaling,
        lut: Align64<GrainLut<BD::Entry>>,
        pw: usize,
        bh: usize,
        row_num: usize,
        is_uv: bool,
        is_id: bool,
    }

    fn case<BD: BitDepth>(rng: &mut Rng, bits: u8, (sx, sy): (bool, bool)) -> Case<BD> {
        let ctr = 128 << (bits - 8);
        let mut data = Rav1dFilmGrainData {
            seed: rng.next() & 0xffff,
            ..Default::default()
        };
        data.scaling_shift = rng.range(8, 11) as u8;
        data.overlap_flag = rng.bit();
        data.clip_to_restricted_range = rng.bit();
        data.chroma_scaling_from_luma = rng.next() % 4 == 0;
        for pl in 0..2 {
            data.uv_mult[pl] = rng.range(-128, 127);
            data.uv_luma_mult[pl] = rng.range(-128, 127);
            data.uv_offset[pl] = rng.range(-256, 255);
        }
        let mut scaling: BD::Scaling = ArrayDefault::default();
        scaling
            .as_mut()
            .iter_mut()
            .for_each(|s| *s = rng.next() as u8);
        let lut = Align64(std::array::from_fn(|_| {
            std::array::from_fn(|_| rng.range(-ctr, ctr - 1).as_::<BD::Entry>())
        }));
        let full = rng.next() % 2 == 0;
        let pw = if full {
            STRIDE >> sx as usize
        } else {
            rng.range(1, (STRIDE >> sx as usize) as c_int) as usize
        };
        let bh = if full { 32 } else { rng.range(1, 32) as usize };
        Case {
            data,
            scaling,
            lut,
            pw,
            bh: bh + sy as usize >> sy as usize,
            row_num: rng.range(0, 1) as usize,
            is_uv: rng.bit(),
            is_id: rng.bit(),
        }
    }

    const LAYOUTS: [(Option<Rav1dPixelLayoutSubSampled>, (bool, bool)); 4] = [
        (None, (false, false)),
        (Some(Rav1dPixelLayoutSubSampled::I420), (true, true)),
        (Some(Rav1dPixelLayoutSubSampled::I422), (true, false)),
        (Some(Rav1dPixelLayoutSubSampled::I444), (false, false)),
    ];

    fn call<BD: BitDepth>(
        dsp: &Rav1dFilmGrainDSPContext,
        layout: Option<Rav1dPixelLayoutSubSampled>,
        c: &Case<BD>,
        dst: &Plane<BD>,
        src: &Plane<BD>,
        luma: &Plane<BD>,
        bd: BD,
    ) {
        match layout {
            None => dsp.fgy_32x32xn.call(
                &dst.comp, &src.comp, &c.data, c.pw, &c.scaling, &c.lut.0, c.bh, c.row_num, bd,
            ),
            Some(l) => dsp.fguv_32x32xn[l].call(
                l, &dst.comp, &src.comp, &c.data, c.pw, &c.scaling, &c.lut.0, c.bh, c.row_num,
                &luma.comp, c.is_uv, c.is_id, bd,
            ),
        }
    }

    fn matches_scalar<BD: BitDepth>(bits: u8) {
        let max = (1 << bits) - 1;
        let bd = BD::from_c(max);
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ bits as u64);
        let dsps = contexts::<BD>();
        for (layout, sub) in LAYOUTS {
            for i in 0..400 {
                let c = case::<BD>(&mut rng, bits, sub);
                let (dst, src, luma) = (
                    Plane::<BD>::new(&mut rng, max),
                    Plane::<BD>::new(&mut rng, max),
                    Plane::<BD>::new(&mut rng, max),
                );
                call(&dsps[0].1, layout, &c, &dst, &src, &luma, bd);
                let want = dst.pixels();
                for (name, dsp) in &dsps[1..] {
                    dst.reset();
                    call(dsp, layout, &c, &dst, &src, &luma, bd);
                    let same = dst.pixels() == want;
                    assert!(
                        same,
                        "{name} differs from scalar at {bits} bits, {sub:?}, case {i}: pw {} bh {} row {} overlap {} clip {} csfl {}",
                        c.pw, c.bh, c.row_num, c.data.overlap_flag, c.data.clip_to_restricted_range, c.data.chroma_scaling_from_luma,
                    );
                }
            }
        }
    }

    #[test]
    fn rust_simd_and_asm_match_scalar_8bpc() {
        matches_scalar::<BitDepth8>(8);
    }

    #[test]
    fn rust_simd_and_asm_match_scalar_10bpc() {
        matches_scalar::<BitDepth16>(10);
    }

    #[test]
    fn rust_simd_and_asm_match_scalar_12bpc() {
        matches_scalar::<BitDepth16>(12);
    }

    fn grain_matches_scalar<BD: BitDepth>(bits: u8) {
        let bd = BD::from_c((1 << bits) - 1);
        let mut rng = Rng(0x0123_4567_89ab_cdef ^ bits as u64);
        let dsps = contexts::<BD>();
        // The asm takes aligned LUTs.
        let zero = || -> Box<Align64<GrainLut<BD::Entry>>> {
            Box::new(Align64(std::array::from_fn(|_| std::array::from_fn(|_| 0.as_::<BD::Entry>()))))
        };
        let dump = |g: &GrainLut<BD::Entry>| -> Vec<c_int> {
            g.iter().flatten().map(|&e| e.as_::<c_int>()).collect()
        };
        for i in 0..512 {
            let mut data = Rav1dFilmGrainData {
                seed: rng.next() & 0xffff,
                ..Default::default()
            };
            data.ar_coeff_lag = (i % 4) as c_int;
            data.grain_scale_shift = (i / 4 % 4) as u8;
            data.ar_coeff_shift = rng.range(6, 9) as u8;
            data.ar_coeffs_y.iter_mut().for_each(|c| *c = rng.next() as i8);
            data.ar_coeffs_uv.iter_mut().flatten().for_each(|c| *c = rng.next() as i8);
            let mut want_y = zero();
            dsps[0].1.generate_grain_y.call(&mut want_y.0, &data, bd);
            let layout = [
                Rav1dPixelLayoutSubSampled::I420,
                Rav1dPixelLayoutSubSampled::I422,
                Rav1dPixelLayoutSubSampled::I444,
            ][i % 3];
            let is_uv = rng.bit();
            let mut want_uv = zero();
            dsps[0].1.generate_grain_uv[layout].call(&mut want_uv.0, &want_y.0, &data, is_uv, bd);
            for (name, dsp) in &dsps[1..] {
                let mut y = zero();
                dsp.generate_grain_y.call(&mut y.0, &data, bd);
                let same = dump(&y.0) == dump(&want_y.0);
                assert!(same, "{name} grain y differs at {bits} bits, case {i}");
                let mut uv = zero();
                dsp.generate_grain_uv[layout].call(&mut uv.0, &want_y.0, &data, is_uv, bd);
                let same = dump(&uv.0) == dump(&want_uv.0);
                assert!(
                    same,
                    "{name} grain uv differs at {bits} bits, case {i}, lag {}",
                    data.ar_coeff_lag
                );
            }
        }
    }

    #[test]
    fn grain_lut_matches_scalar_8bpc() {
        grain_matches_scalar::<BitDepth8>(8);
    }

    #[test]
    fn grain_lut_matches_scalar_10bpc() {
        grain_matches_scalar::<BitDepth16>(10);
    }

    #[test]
    fn grain_lut_matches_scalar_12bpc() {
        grain_matches_scalar::<BitDepth16>(12);
    }

    /// `cargo test --release --lib filmgrain::tests::time -- --ignored --nocapture`: one full
    /// 32-row strip of `STRIDE` pixels with overlaps.
    fn time<BD: BitDepth>(bits: u8) {
        let max = (1 << bits) - 1;
        let bd = BD::from_c(max);
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for (layout, sub) in LAYOUTS {
            let mut c = case::<BD>(&mut rng, bits, sub);
            c.pw = STRIDE >> sub.0 as usize;
            c.bh = 32 >> sub.1 as usize;
            c.row_num = 1;
            c.data.overlap_flag = true;
            c.data.chroma_scaling_from_luma = false;
            let (dst, src, luma) = (
                Plane::<BD>::new(&mut rng, max),
                Plane::<BD>::new(&mut rng, max),
                Plane::<BD>::new(&mut rng, max),
            );
            let mut line = format!(
                "{bits:2}-bit {:5}:",
                layout.map_or("y".into(), |l| format!("uv{}", l as u8))
            );
            // `FG_BENCH=<context>,<layout index>,<calls>`: only these calls, for perf.
            if let Ok(only) = std::env::var("FG_BENCH") {
                let f: Vec<&str> = only.split(',').collect();
                if f[1].parse::<usize>().unwrap() == LAYOUTS.iter().position(|l| l.0 == layout).unwrap() {
                    let dsp = contexts::<BD>().into_iter().find(|c| c.0 == f[0]).unwrap().1;
                    for _ in 0..f[2].parse::<usize>().unwrap() {
                        call(black_box(&dsp), layout, &c, &dst, &src, &luma, bd);
                    }
                }
                continue;
            }
            for (name, dsp) in contexts::<BD>() {
                let mut best = f64::MAX;
                for _ in 0..7 {
                    let start = Instant::now();
                    for _ in 0..200 {
                        call(black_box(&dsp), layout, &c, &dst, &src, &luma, bd);
                    }
                    best = best.min(start.elapsed().as_secs_f64() * 1e9 / 200.);
                }
                line += &format!("  {name} {best:8.0} ns");
            }
            println!("{line}");
        }
        if std::env::var("FG_BENCH").is_ok() {
            return;
        }
        let mut data = Rav1dFilmGrainData {
            seed: 0x1234,
            ..Default::default()
        };
        data.ar_coeff_lag = 3;
        data.ar_coeff_shift = 7;
        data.ar_coeffs_y.iter_mut().for_each(|c| *c = rng.range(-20, 20) as i8);
        data.ar_coeffs_uv.iter_mut().flatten().for_each(|c| *c = rng.range(-20, 20) as i8);
        let mut line = format!("{bits:2}-bit grain y + uv420, lag 3:");
        for (name, dsp) in contexts::<BD>() {
            let mut y: Box<Align64<GrainLut<BD::Entry>>> =
                Box::new(Align64(std::array::from_fn(|_| std::array::from_fn(|_| 0.as_::<BD::Entry>()))));
            let mut uv = Box::new(Align64(y.0));
            let mut best = f64::MAX;
            for _ in 0..7 {
                let start = Instant::now();
                for _ in 0..20 {
                    dsp.generate_grain_y.call(black_box(&mut y.0), &data, bd);
                    dsp.generate_grain_uv[Rav1dPixelLayoutSubSampled::I420].call(black_box(&mut uv.0), &y.0, &data, false, bd);
                }
                best = best.min(start.elapsed().as_secs_f64() * 1e9 / 20.);
            }
            line += &format!("  {name} {best:8.0} ns");
        }
        println!("{line}");
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
