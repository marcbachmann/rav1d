use std::num::NonZeroUsize;
use std::{cmp, slice};

use strum::EnumCount;

use crate::cpu::CpuFlags;
use crate::enum_map::DefaultValue;
use crate::ffi_safe::FFISafe;
use crate::include::common::bitdepth::{AsPrimitive, BitDepth, DynCoef, DynPixel};
use crate::include::common::intops::iclip;
use crate::include::dav1d::picture::{
    FFISafeRav1dPictureDataComponentOffset, Rav1dPictureDataComponentOffset,
};
use crate::itx_1d::rav1d_inv_wht4_1d_c;
use crate::itx_1d::{
    rav1d_inv_adst16_1d_c, rav1d_inv_adst4_1d_c, rav1d_inv_adst8_1d_c, rav1d_inv_dct16_1d_c,
    rav1d_inv_dct32_1d_c, rav1d_inv_dct4_1d_c, rav1d_inv_dct64_1d_c, rav1d_inv_dct8_1d_c,
    rav1d_inv_flipadst16_1d_c, rav1d_inv_flipadst4_1d_c, rav1d_inv_flipadst8_1d_c,
    rav1d_inv_identity16_1d_c, rav1d_inv_identity32_1d_c, rav1d_inv_identity4_1d_c,
    rav1d_inv_identity8_1d_c,
};
use crate::itx_simd;
use crate::levels::{
    TxfmSize, TxfmType, ADST_ADST, ADST_DCT, ADST_FLIPADST, DCT_ADST, DCT_DCT, DCT_FLIPADST,
    FLIPADST_ADST, FLIPADST_DCT, FLIPADST_FLIPADST, H_ADST, H_DCT, H_FLIPADST, IDTX,
    N_TX_TYPES_PLUS_LL, V_ADST, V_DCT, V_FLIPADST, WHT_WHT,
};
use crate::strided::Strided as _;
use crate::wrap_fn_ptr::wrap_fn_ptr;

pub type Itx1dFn = fn(c: &mut [i32], stride: NonZeroUsize, min: i32, max: i32);

#[inline(never)]
fn inv_txfm_add<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    coeff: &mut [BD::Coef],
    eob: i32,
    w: usize,
    h: usize,
    shift: u8,
    first_1d_fn: Itx1dFn,
    second_1d_fn: Itx1dFn,
    has_dc_only: bool,
    bd: BD,
) {
    let bitdepth_max = bd.bitdepth_max().as_::<i32>();

    assert!(w >= 4 && w <= 64);
    assert!(h >= 4 && h <= 64);
    assert!(eob >= 0);

    let is_rect2 = w * 2 == h || h * 2 == w;
    let rnd = 1 << shift >> 1;

    if eob < has_dc_only as i32 {
        let mut dc = coeff[0].as_::<i32>();
        coeff[0] = 0.as_();
        if is_rect2 {
            dc = dc * 181 + 128 >> 8;
        }
        dc = dc * 181 + 128 >> 8;
        dc = dc + rnd >> shift;
        dc = dc * 181 + 128 + 2048 >> 12;
        for y in 0..h {
            let dst = dst + (y as isize * dst.pixel_stride::<BD>());
            let dst = &mut *dst.slice_mut::<BD>(w);
            for x in 0..w {
                dst[x] = bd.iclip_pixel(dst[x].as_::<i32>() + dc);
            }
        }
        return;
    }

    let sh = cmp::min(h, 32);
    let sw = cmp::min(w, 32);

    let coeff = &mut coeff[..sh * sw];

    let row_clip_min;
    let col_clip_min;
    if BD::BITDEPTH == 8 {
        row_clip_min = i16::MIN as i32;
        col_clip_min = i16::MIN as i32;
    } else {
        row_clip_min = (!bitdepth_max) << 7;
        col_clip_min = (!bitdepth_max) << 5;
    }
    let row_clip_max = !row_clip_min;
    let col_clip_max = !col_clip_min;

    let mut tmp = [0; 64 * 64];
    let mut c = &mut tmp[..];
    for y in 0..sh {
        if is_rect2 {
            for x in 0..sw {
                c[x] = coeff[y + x * sh].as_::<i32>() * 181 + 128 >> 8;
            }
        } else {
            for x in 0..sw {
                c[x] = coeff[y + x * sh].as_();
            }
        }
        first_1d_fn(c, 1.try_into().unwrap(), row_clip_min, row_clip_max);
        c = &mut c[w..];
    }

    coeff.fill(0.into());
    for i in 0..w * sh {
        tmp[i] = iclip(tmp[i] + rnd >> shift, col_clip_min, col_clip_max);
    }

    for x in 0..w {
        second_1d_fn(
            &mut tmp[x..],
            w.try_into().unwrap(),
            col_clip_min,
            col_clip_max,
        );
    }

    for y in 0..h {
        let dst = dst + (y as isize * dst.pixel_stride::<BD>());
        let dst = &mut *dst.slice_mut::<BD>(w);
        for x in 0..w {
            dst[x] = bd.iclip_pixel(dst[x].as_::<i32>() + (tmp[y * w + x] + 8 >> 4));
        }
    }
}

fn inv_txfm_add_rust<const W: usize, const H: usize, const TYPE: TxfmType, BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    coeff: &mut [BD::Coef],
    eob: i32,
    bd: BD,
) {
    let shift = match (W, H) {
        (4, 4) => 0,
        (4, 8) => 0,
        (4, 16) => 1,
        (8, 4) => 0,
        (8, 8) => 1,
        (8, 16) => 1,
        (8, 32) => 2,
        (16, 4) => 1,
        (16, 8) => 1,
        (16, 16) => 2,
        (16, 32) => 1,
        (16, 64) => 2,
        (32, 8) => 2,
        (32, 16) => 1,
        (32, 32) => 2,
        (32, 64) => 1,
        (64, 16) => 2,
        (64, 32) => 1,
        (64, 64) => 2,
        _ => unreachable!(),
    };
    let has_dc_only = TYPE == DCT_DCT;

    enum Type {
        Identity,
        Dct,
        Adst,
        FlipAdst,
    }
    use Type::*;
    // For some reason, this is flipped.
    let (second, first) = match TYPE {
        IDTX => (Identity, Identity),
        DCT_DCT => (Dct, Dct),
        ADST_DCT => (Adst, Dct),
        FLIPADST_DCT => (FlipAdst, Dct),
        H_DCT => (Identity, Dct),
        DCT_ADST => (Dct, Adst),
        ADST_ADST => (Adst, Adst),
        FLIPADST_ADST => (FlipAdst, Adst),
        DCT_FLIPADST => (Dct, FlipAdst),
        ADST_FLIPADST => (Adst, FlipAdst),
        FLIPADST_FLIPADST => (FlipAdst, FlipAdst),
        V_DCT => (Dct, Identity),
        H_ADST => (Identity, Adst),
        H_FLIPADST => (Identity, FlipAdst),
        V_ADST => (Adst, Identity),
        V_FLIPADST => (FlipAdst, Identity),

        WHT_WHT if (W, H) == (4, 4) => return inv_txfm_add_wht_wht_4x4_rust(dst, coeff, bd),

        _ => unreachable!(),
    };

    fn resolve_1d_fn(r#type: Type, n: usize) -> Itx1dFn {
        match (r#type, n) {
            (Identity, 4) => rav1d_inv_identity4_1d_c,
            (Identity, 8) => rav1d_inv_identity8_1d_c,
            (Identity, 16) => rav1d_inv_identity16_1d_c,
            (Identity, 32) => rav1d_inv_identity32_1d_c,
            (Dct, 4) => rav1d_inv_dct4_1d_c,
            (Dct, 8) => rav1d_inv_dct8_1d_c,
            (Dct, 16) => rav1d_inv_dct16_1d_c,
            (Dct, 32) => rav1d_inv_dct32_1d_c,
            (Dct, 64) => rav1d_inv_dct64_1d_c,
            (Adst, 4) => rav1d_inv_adst4_1d_c,
            (Adst, 8) => rav1d_inv_adst8_1d_c,
            (Adst, 16) => rav1d_inv_adst16_1d_c,
            (FlipAdst, 4) => rav1d_inv_flipadst4_1d_c,
            (FlipAdst, 8) => rav1d_inv_flipadst8_1d_c,
            (FlipAdst, 16) => rav1d_inv_flipadst16_1d_c,
            _ => unreachable!(),
        }
    }

    let first_1d_fn = resolve_1d_fn(first, W);
    let second_1d_fn = resolve_1d_fn(second, H);

    inv_txfm_add(
        dst,
        coeff,
        eob,
        W,
        H,
        shift,
        first_1d_fn,
        second_1d_fn,
        has_dc_only,
        bd,
    )
}

/// # Safety
///
/// Must be called by [`itxfm::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn inv_txfm_add_c_erased<
    const W: usize,
    const H: usize,
    const TYPE: TxfmType,
    BD: BitDepth,
>(
    _dst_ptr: *mut DynPixel,
    _stride: isize,
    coeff: *mut DynCoef,
    eob: i32,
    bitdepth_max: i32,
    coeff_len: u16,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `itxfm::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    // SAFETY: `fn itxfm::Fn::call` passes `coeff.len()` as `coeff_len`.
    let coeff = unsafe { slice::from_raw_parts_mut(coeff.cast(), coeff_len.into()) };
    let bd = BD::from_c(bitdepth_max);
    inv_txfm_add_rust::<W, H, TYPE, BD>(dst, coeff, eob, bd)
}

/// # Safety
///
/// Must be called by [`itxfm::Fn::call`] at 8 or 10 bits, on x86 once the CPU flags hold SSE41.
#[deny(unsafe_op_in_unsafe_fn)]
#[allow(unused_unsafe)]
unsafe extern "C" fn inv_txfm_add_simd_erased<
    const W: usize,
    const H: usize,
    const TYPE: TxfmType,
    BD: BitDepth,
    const G: usize,
    const WIDE: bool,
>(
    _dst_ptr: *mut DynPixel,
    _stride: isize,
    coeff: *mut DynCoef,
    eob: i32,
    _bitdepth_max: i32,
    coeff_len: u16,
    dst: FFISafeRav1dPictureDataComponentOffset,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `itxfm::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    // A debugging aid: `RAV1D_ITX_SIMD_SIZES` (a `TxfmSize` bit mask) leaves the other sizes scalar.
    #[cfg(rav1d_itx_bisect)]
    {
        static MASK: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        let mask = *MASK.get_or_init(|| {
            std::env::var("RAV1D_ITX_SIMD_SIZES")
                .ok()
                .and_then(|v| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok())
                .unwrap_or(u32::MAX)
        });
        if mask >> TxfmSize::from_wh(W, H) as u32 & 1 == 0 {
            let bd = BD::from_c(_bitdepth_max);
            // SAFETY: `fn itxfm::Fn::call` passes `coeff.len()` as `coeff_len`.
            let coeff = unsafe { slice::from_raw_parts_mut(coeff.cast(), coeff_len.into()) };
            return inv_txfm_add_rust::<W, H, TYPE, BD>(dst, coeff, eob, bd);
        }
    }
    if BD::BITDEPTH == 16 {
        // SAFETY: `fn itxfm::Fn::call` passes `coeff.len()` as `coeff_len`, and `BD::Coef` is
        // `i32` at 16 bpc.
        let coeff = unsafe { slice::from_raw_parts_mut(coeff.cast::<i32>(), coeff_len.into()) };
        // SAFETY: SSE4.1 is checked before this `fn` is chosen on x86; elsewhere it is safe.
        return unsafe {
            itx_simd::add10::<BD, W, H, TYPE, G>(dst, coeff, eob, _bitdepth_max as i16)
        };
    }
    // SAFETY: `fn itxfm::Fn::call` passes `coeff.len()` as `coeff_len`, and `BD::Coef` is `i16`
    // at 8 bpc.
    let coeff = unsafe { slice::from_raw_parts_mut(coeff.cast::<i16>(), coeff_len.into()) };
    // SAFETY: SSE4.1 is checked before this `fn` is chosen on x86, and AVX2 before `WIDE` is;
    // elsewhere it is safe.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        if WIDE {
            return itx_simd::add_avx2::<BD, W, H, TYPE, G>(dst, coeff, eob);
        }
        match (W, H) {
            (4, 4)
            | (4, 8)
            | (8, 4)
            | (4, 16)
            | (16, 4)
            | (8, 8)
            | (16, 16)
            | (8, 16)
            | (16, 8)
            | (32, 32)
            | (8, 32)
            | (32, 8)
            | (16, 32)
            | (32, 16)
            | (16, 64)
            | (64, 16)
            | (32, 64)
            | (64, 32)
            | (64, 64) => itx_simd::add::<BD, W, H, TYPE, G>(dst, coeff, eob),
            _ => unreachable!(),
        }
    }
}

wrap_fn_ptr!(unsafe extern "C" fn itxfm(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: i32,
    bitdepth_max: i32,
    _coeff_len: u16,
    _dst: FFISafeRav1dPictureDataComponentOffset,
) -> ());

impl itxfm::Fn {
    pub fn call<BD: BitDepth>(
        &self,
        dst: Rav1dPictureDataComponentOffset,
        coeff: &mut [BD::Coef],
        eob: i32,
        bd: BD,
    ) {
        let dst_ptr = dst.as_mut_ptr::<BD>().cast();
        let dst_stride = dst.stride();
        let coeff_len = coeff.len() as u16;
        let coeff = coeff.as_mut_ptr().cast();
        let bd = bd.into_c();
        let dst = dst.into_ffi_safe();
        // SAFETY: Fallback `fn inv_txfm_add_rust` is safe; asm is supposed to do the same.
        unsafe { self.get()(dst_ptr, dst_stride, coeff, eob, bd, coeff_len, dst) }
    }
}

pub struct Rav1dInvTxfmDSPContext {
    pub itxfm_add: [[itxfm::Fn; N_TX_TYPES_PLUS_LL]; TxfmSize::COUNT],
}

fn inv_txfm_add_wht_wht_4x4_rust<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    coeff: &mut [BD::Coef],
    bd: BD,
) {
    const H: usize = 4;
    const W: usize = 4;

    let coeff = &mut coeff[..W * H];

    let mut tmp = [0; W * H];
    let mut c = &mut tmp[..];
    for y in 0..H {
        for x in 0..W {
            c[x] = coeff[y + x * H].as_::<i32>() >> 2;
        }
        rav1d_inv_wht4_1d_c(c, 1.try_into().unwrap());
        c = &mut c[W..];
    }
    coeff.fill(0.into());

    for x in 0..W {
        rav1d_inv_wht4_1d_c(&mut tmp[x..], H.try_into().unwrap());
    }

    for y in 0..H {
        let dst = dst + (y as isize * dst.pixel_stride::<BD>());
        let dst = &mut *dst.slice_mut::<BD>(W);
        for x in 0..W {
            dst[x] = bd.iclip_pixel(dst[x].as_::<i32>() + tmp[y * W + x]);
        }
    }
}

impl Rav1dInvTxfmDSPContext {
    const fn assign<const W: usize, const H: usize, BD: BitDepth>(mut self) -> Self {
        let tx = TxfmSize::from_wh(W, H) as usize;

        macro_rules! assign {
            ($type:expr) => {{
                self.itxfm_add[tx][$type as usize] =
                    itxfm::Fn::new(inv_txfm_add_c_erased::<W, H, $type, BD>);
            }};
        }

        let max_wh = if W > H { W } else { H };

        let assign84 = W * H <= 8 * 16;
        let assign16 = assign84 || (W == 16 && H == 16);
        let assign32 = assign16 || max_wh == 32;
        let assign64 = assign32 || max_wh == 64;

        if assign84 {
            assign!(H_FLIPADST);
            assign!(V_FLIPADST);
            assign!(H_ADST);
            assign!(V_ADST);
        }
        if assign16 {
            assign!(DCT_ADST);
            assign!(ADST_DCT);
            assign!(ADST_ADST);
            assign!(ADST_FLIPADST);
            assign!(FLIPADST_ADST);
            assign!(DCT_FLIPADST);
            assign!(FLIPADST_DCT);
            assign!(FLIPADST_FLIPADST);
            assign!(H_DCT);
            assign!(V_DCT);
        }
        if assign32 {
            assign!(IDTX);
        }
        if assign64 {
            assign!(DCT_DCT);
        }

        if W == 4 && H == 4 {
            assign!(WHT_WHT);
        }

        self
    }

    pub const fn default<BD: BitDepth>() -> Self {
        let mut c = Self {
            itxfm_add: [[itxfm::Fn::DEFAULT; N_TX_TYPES_PLUS_LL]; TxfmSize::COUNT],
        };

        c = c.assign::<4, 4, BD>();
        c = c.assign::<4, 8, BD>();
        c = c.assign::<4, 16, BD>();
        c = c.assign::<8, 4, BD>();
        c = c.assign::<8, 8, BD>();
        c = c.assign::<8, 16, BD>();
        c = c.assign::<8, 32, BD>();
        c = c.assign::<16, 4, BD>();
        c = c.assign::<16, 8, BD>();
        c = c.assign::<16, 16, BD>();
        c = c.assign::<16, 32, BD>();
        c = c.assign::<16, 64, BD>();
        c = c.assign::<32, 8, BD>();
        c = c.assign::<32, 16, BD>();
        c = c.assign::<32, 32, BD>();
        c = c.assign::<32, 64, BD>();
        c = c.assign::<64, 16, BD>();
        c = c.assign::<64, 32, BD>();
        c = c.assign::<64, 64, BD>();

        c
    }

    /// The sizes with Rust SIMD kernels, at 8 and 10 bits; the rest keep the scalar `fn`s. At 12
    /// bits the column pass outgrows `i16`.
    #[inline(always)]
    const fn init<BD: BitDepth>(mut self, flags: CpuFlags, bpc: u8) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if !flags.contains(CpuFlags::SSE41) {
            return self;
        }
        #[cfg(target_arch = "x86_64")]
        let avx2 = flags.contains(CpuFlags::AVX2);
        #[cfg(not(target_arch = "x86_64"))]
        let avx2 = false;
        let _ = flags;
        if !crate::simd128::native() || bpc > 10 {
            return self;
        }

        // Sixteen lanes on AVX2 where a side reaches 16.
        macro_rules! assign {
            ($w:literal, $h:literal, $($type:ident),*) => {{
                let tx = TxfmSize::from_wh($w, $h) as usize;
                let wide = avx2 && BD::BITDEPTH == 8 && ($w >= 16 || $h >= 16);
                $(
                    self.itxfm_add[tx][$type as usize] = if wide {
                        itxfm::Fn::new(inv_txfm_add_simd_erased::<$w, $h, $type, BD, { ($w + 7) / 8 }, true>)
                    } else {
                        itxfm::Fn::new(inv_txfm_add_simd_erased::<$w, $h, $type, BD, { ($w + 7) / 8 }, false>)
                    };
                )*
            }};
        }

        assign!(
            8,
            8,
            DCT_DCT,
            ADST_DCT,
            DCT_ADST,
            ADST_ADST,
            FLIPADST_DCT,
            DCT_FLIPADST,
            FLIPADST_FLIPADST,
            ADST_FLIPADST,
            FLIPADST_ADST,
            IDTX,
            V_DCT,
            H_DCT,
            V_ADST,
            H_ADST,
            V_FLIPADST,
            H_FLIPADST
        );
        assign!(
            16,
            16,
            DCT_DCT,
            ADST_DCT,
            DCT_ADST,
            ADST_ADST,
            FLIPADST_DCT,
            DCT_FLIPADST,
            FLIPADST_FLIPADST,
            ADST_FLIPADST,
            FLIPADST_ADST,
            IDTX,
            V_DCT,
            H_DCT
        );
        assign!(
            8,
            16,
            DCT_DCT,
            ADST_DCT,
            DCT_ADST,
            ADST_ADST,
            FLIPADST_DCT,
            DCT_FLIPADST,
            FLIPADST_FLIPADST,
            ADST_FLIPADST,
            FLIPADST_ADST,
            IDTX,
            V_DCT,
            H_DCT,
            V_ADST,
            H_ADST,
            V_FLIPADST,
            H_FLIPADST
        );
        assign!(
            16,
            8,
            DCT_DCT,
            ADST_DCT,
            DCT_ADST,
            ADST_ADST,
            FLIPADST_DCT,
            DCT_FLIPADST,
            FLIPADST_FLIPADST,
            ADST_FLIPADST,
            FLIPADST_ADST,
            IDTX,
            V_DCT,
            H_DCT,
            V_ADST,
            H_ADST,
            V_FLIPADST,
            H_FLIPADST
        );
        assign!(32, 32, DCT_DCT, IDTX);
        assign!(8, 32, DCT_DCT, IDTX);
        assign!(32, 8, DCT_DCT, IDTX);
        assign!(16, 32, DCT_DCT, IDTX);
        assign!(32, 16, DCT_DCT, IDTX);
        assign!(16, 64, DCT_DCT);
        assign!(64, 16, DCT_DCT);
        assign!(32, 64, DCT_DCT);
        assign!(64, 32, DCT_DCT);
        assign!(64, 64, DCT_DCT);
        assign!(
            4,
            4,
            DCT_DCT,
            ADST_DCT,
            DCT_ADST,
            ADST_ADST,
            FLIPADST_DCT,
            DCT_FLIPADST,
            FLIPADST_FLIPADST,
            ADST_FLIPADST,
            FLIPADST_ADST,
            IDTX,
            V_DCT,
            H_DCT,
            V_ADST,
            H_ADST,
            V_FLIPADST,
            H_FLIPADST
        );
        assign!(
            4,
            8,
            DCT_DCT,
            ADST_DCT,
            DCT_ADST,
            ADST_ADST,
            FLIPADST_DCT,
            DCT_FLIPADST,
            FLIPADST_FLIPADST,
            ADST_FLIPADST,
            FLIPADST_ADST,
            IDTX,
            V_DCT,
            H_DCT,
            V_ADST,
            H_ADST,
            V_FLIPADST,
            H_FLIPADST
        );
        assign!(
            8,
            4,
            DCT_DCT,
            ADST_DCT,
            DCT_ADST,
            ADST_ADST,
            FLIPADST_DCT,
            DCT_FLIPADST,
            FLIPADST_FLIPADST,
            ADST_FLIPADST,
            FLIPADST_ADST,
            IDTX,
            V_DCT,
            H_DCT,
            V_ADST,
            H_ADST,
            V_FLIPADST,
            H_FLIPADST
        );
        assign!(
            4,
            16,
            DCT_DCT,
            ADST_DCT,
            DCT_ADST,
            ADST_ADST,
            FLIPADST_DCT,
            DCT_FLIPADST,
            FLIPADST_FLIPADST,
            ADST_FLIPADST,
            FLIPADST_ADST,
            IDTX,
            V_DCT,
            H_DCT,
            V_ADST,
            H_ADST,
            V_FLIPADST,
            H_FLIPADST
        );
        assign!(
            16,
            4,
            DCT_DCT,
            ADST_DCT,
            DCT_ADST,
            ADST_ADST,
            FLIPADST_DCT,
            DCT_FLIPADST,
            FLIPADST_FLIPADST,
            ADST_FLIPADST,
            FLIPADST_ADST,
            IDTX,
            V_DCT,
            H_DCT,
            V_ADST,
            H_ADST,
            V_FLIPADST,
            H_FLIPADST
        );
        self
    }

    pub const fn new<BD: BitDepth>(flags: CpuFlags, bpc: u8) -> Self {
        Self::default::<BD>().init::<BD>(flags, bpc)
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
    /// A 16-wide block's rows are stored aligned by the asm, as the decoder lays them out.
    const BX: usize = 16;

    /// The block at (16, 4) of a random plane after `dsp`'s transform of `coeff`, and the
    /// coefficients it leaves.
    fn run<BD: BitDepth>(
        dsp: &Rav1dInvTxfmDSPContext,
        tx: TxfmSize,
        t: TxfmType,
        orig: &[i32],
        coeff: &[i32],
        eob: i32,
        bitdepth_max: i32,
    ) -> (Vec<i32>, Vec<i32>) {
        let mut buf = AlignedVec64::<BD::Pixel>::new();
        buf.resize(orig.len(), 0.into());
        for (b, &o) in buf.as_mut_slice().iter_mut().zip(orig) {
            *b = o.as_::<BD::Pixel>();
        }
        let comp = Rav1dPictureDataComponent::wrap_buf::<BD>(buf.as_mut_slice(), STRIDE);
        // The asm loads the coefficients aligned, as the decoder's buffers are.
        let mut c = AlignedVec64::<BD::Coef>::new();
        c.resize(coeff.len(), 0i16.into());
        for (c, &o) in c.as_mut_slice().iter_mut().zip(coeff) {
            *c = o.as_::<BD::Coef>();
        }
        dsp.itxfm_add[tx as usize][t as usize].call::<BD>(
            WithOffset {
                data: &comp,
                offset: 4 * STRIDE + BX,
            },
            c.as_mut_slice(),
            eob,
            BD::from_c(bitdepth_max),
        );
        drop(comp);
        (
            buf.as_slice().iter().map(|&p| p.into()).collect(),
            c.as_slice().iter().map(|&c| c.into()).collect(),
        )
    }

    /// The Rust SIMD kernels at every width the CPU has: 128-bit, and on x86_64 at 8 bpc also
    /// AVX2.
    fn simd_contexts<BD: BitDepth>(bpc: u8) -> Vec<(&'static str, Rav1dInvTxfmDSPContext)> {
        rav1d_init_cpu();
        let flags = rav1d_get_cpu_flags();
        let ctx = |flags| Rav1dInvTxfmDSPContext::default::<BD>().init::<BD>(flags, bpc);
        #[allow(unused_mut)]
        let mut out = vec![("rust simd", ctx(flags))];
        #[cfg(target_arch = "x86_64")]
        if flags.contains(CpuFlags::AVX2) && BD::BITDEPTH == 8 {
            out.push((
                "rust 128",
                ctx(flags.difference(CpuFlags::AVX2 | CpuFlags::AVX512ICL)),
            ));
        }
        out
    }

    /// Every type of a size on random coefficients and pixels, against the scalar transform, at
    /// 8 and 10 bits; `eob` 0 takes the DC path. Across the whole coefficient range the C's
    /// 32-bit intermediates and the asm's saturating 16-bit ones part, and the conformance md5 is
    /// the oracle there: the C is held to the 8-bit coefficient range, and at 10 bits, where
    /// there is asm, four times that range (twice from 32 points on, where both part from what
    /// this does in the sums of 32) matches the C or the asm.
    fn simd_matches_scalar(tx: TxfmSize, w: usize, h: usize) {
        let scalar = || Rav1dInvTxfmDSPContext::default::<BitDepth16>();
        matches::<BitDepth8>(tx, w, h, 255, &[&Rav1dInvTxfmDSPContext::default::<BitDepth8>()], 1);
        matches::<BitDepth16>(tx, w, h, 1023, &[&scalar()], 1);
    }

    fn matches<BD: BitDepth>(
        tx: TxfmSize,
        w: usize,
        h: usize,
        bitdepth_max: i32,
        oracles: &[&Rav1dInvTxfmDSPContext],
        scale: i32,
    ) {
        rav1d_init_cpu();
        let scalar = Rav1dInvTxfmDSPContext::default::<BD>();
        let simds = simd_contexts::<BD>(if bitdepth_max == 255 { 8 } else { 10 });
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for t in 0..N_TX_TYPES_PLUS_LL as TxfmType - 1 {
            if *scalar.itxfm_add[tx as usize][t as usize].get() as usize
                == *itxfm::Fn::DEFAULT.get() as usize
            {
                continue;
            }
            for round in 0..200 {
                let eob = if round < 2 {
                    round
                } else {
                    (rng.next() % (w.min(32) * h.min(32)) as u32) as i32
                };
                // A side of 64 has 32 coefficients.
                let (cw, ch) = (w.min(32), h.min(32));
                let mut coeff = vec![0i32; cw * ch];
                let n = if eob == 0 { 1 } else { eob as usize + 1 };
                let range = [512, 4096][round as usize % 2] * scale;
                // A 32x32 block with `eob < 136` keeps its coefficients in the top-left 16x16.
                let (nw, nh) = if w == 32 && h == 32 && eob < 136 {
                    (16, 16)
                } else {
                    (cw, ch)
                };
                let coef = |rng: &mut Rng| (rng.next() % (2 * range + 1) as u32) as i32 - range;
                for _ in 0..n {
                    if rng.next() % 2 == 0 {
                        let (x, y) = (rng.next() as usize % nw, rng.next() as usize % nh);
                        coeff[x * ch + y] = coef(&mut rng);
                    }
                }
                coeff[0] = coef(&mut rng);
                let orig: Vec<i32> = (0..STRIDE * ROWS)
                    .map(|_| (rng.next() % (bitdepth_max as u32 + 1)) as i32)
                    .collect();
                let wants: Vec<_> = oracles
                    .iter()
                    .map(|o| run::<BD>(o, tx, t, &orig, &coeff, eob, bitdepth_max))
                    .collect();
                for (name, simd) in &simds {
                    let got = run::<BD>(simd, tx, t, &orig, &coeff, eob, bitdepth_max);
                    if !wants.contains(&got) {
                        let want = &wants[0];
                        let at = (0..want.0.len()).find(|&i| got.0[i] != want.0[i]);
                        panic!(
                            "{w}x{h} type {t} eob {eob} {name} {bitdepth_max} x{scale}: pixel \
                             {at:?} got {:?} want {:?}",
                            at.map(|i| got.0[i]),
                            at.map(|i| want.0[i]),
                        );
                    }
                }
            }
        }
    }

    /// 12 bits keep the scalar transforms.
    #[test]
    fn simd_not_at_12_bits() {
        rav1d_init_cpu();
        let scalar = Rav1dInvTxfmDSPContext::default::<BitDepth16>();
        let simd = &simd_contexts::<BitDepth16>(12)[0].1;
        for (a, b) in scalar
            .itxfm_add
            .iter()
            .flatten()
            .zip(simd.itxfm_add.iter().flatten())
        {
            assert_eq!(*a.get() as usize, *b.get() as usize);
        }
    }

    /// 4x16's identity rows halve at once: `x + 0.414x` overflows `i16` from about 23170 on, and
    /// `x == -1` is the one lane where the overflow-free average needs its fix-up.
    #[test]
    fn identity4_rows_round_without_overflow() {
        rav1d_init_cpu();
        let scalar = Rav1dInvTxfmDSPContext::default::<BitDepth8>();
        // Column 0: rows 0 and 1 overflow and their column DCT cancels to an in-range pixel near
        // row 3; column 1: a lone -1, which must stay 0.
        let mut coeff = vec![0; 64];
        coeff[0] = 30000;
        coeff[1] = -30000;
        coeff[16 + 2] = -1;
        let orig = vec![128; STRIDE * ROWS];
        let want = run::<BitDepth8>(&scalar, TxfmSize::R4x16, V_DCT, &orig, &coeff, 16, 255);
        for (name, simd) in &simd_contexts::<BitDepth8>(8) {
            let got = run::<BitDepth8>(simd, TxfmSize::R4x16, V_DCT, &orig, &coeff, 16, 255);
            assert_eq!(got, want, "{name}");
        }
    }

    #[test]
    fn simd_8x8_matches_scalar() {
        simd_matches_scalar(TxfmSize::S8x8, 8, 8);
    }

    #[test]
    fn simd_16x16_matches_scalar() {
        simd_matches_scalar(TxfmSize::S16x16, 16, 16);
    }

    #[test]
    fn simd_8x16_matches_scalar() {
        simd_matches_scalar(TxfmSize::R8x16, 8, 16);
    }

    #[test]
    fn simd_16x8_matches_scalar() {
        simd_matches_scalar(TxfmSize::R16x8, 16, 8);
    }

    #[test]
    fn simd_32x32_matches_scalar() {
        simd_matches_scalar(TxfmSize::S32x32, 32, 32);
    }

    #[test]
    fn simd_8x32_matches_scalar() {
        simd_matches_scalar(TxfmSize::R8x32, 8, 32);
    }

    #[test]
    fn simd_32x8_matches_scalar() {
        simd_matches_scalar(TxfmSize::R32x8, 32, 8);
    }

    #[test]
    fn simd_16x32_matches_scalar() {
        simd_matches_scalar(TxfmSize::R16x32, 16, 32);
    }

    #[test]
    fn simd_32x16_matches_scalar() {
        simd_matches_scalar(TxfmSize::R32x16, 32, 16);
    }

    #[test]
    fn simd_16x64_matches_scalar() {
        simd_matches_scalar(TxfmSize::R16x64, 16, 64);
    }

    #[test]
    fn simd_64x16_matches_scalar() {
        simd_matches_scalar(TxfmSize::R64x16, 64, 16);
    }

    #[test]
    fn simd_32x64_matches_scalar() {
        simd_matches_scalar(TxfmSize::R32x64, 32, 64);
    }

    #[test]
    fn simd_64x32_matches_scalar() {
        simd_matches_scalar(TxfmSize::R64x32, 64, 32);
    }

    #[test]
    fn simd_64x64_matches_scalar() {
        simd_matches_scalar(TxfmSize::S64x64, 64, 64);
    }

    #[test]
    fn simd_4x4_matches_scalar() {
        simd_matches_scalar(TxfmSize::S4x4, 4, 4);
    }

    #[test]
    fn simd_4x8_matches_scalar() {
        simd_matches_scalar(TxfmSize::R4x8, 4, 8);
    }

    #[test]
    fn simd_8x4_matches_scalar() {
        simd_matches_scalar(TxfmSize::R8x4, 8, 4);
    }

    #[test]
    fn simd_4x16_matches_scalar() {
        simd_matches_scalar(TxfmSize::R4x16, 4, 16);
    }

    #[test]
    fn simd_16x4_matches_scalar() {
        simd_matches_scalar(TxfmSize::R16x4, 16, 4);
    }

    /// ns per call of a size's DCT_DCT and ADST_ADST with a full block, scalar and Rust SIMD, at 8
    /// and 10 bits.
    #[test]
    #[ignore]
    fn time_itx() {
        time_itx_at::<BitDepth8>(255);
        time_itx_at::<BitDepth16>(1023);
    }

    fn time_itx_at<BD: BitDepth>(bitdepth_max: i32) {
        use std::hint::black_box;
        use std::time::Instant;

        let bpc = if bitdepth_max == 255 { 8 } else { 10 };
        let mut dsps = vec![("scalar", Rav1dInvTxfmDSPContext::default::<BD>())];
        dsps.extend(simd_contexts::<BD>(bpc));
        let scale = (bitdepth_max + 1) / 256;
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let mut buf = AlignedVec64::<BD::Pixel>::new();
        buf.resize(STRIDE * ROWS, 0.into());
        for b in buf.as_mut_slice() {
            *b = ((rng.next() % (bitdepth_max as u32 + 1)) as i32).as_::<BD::Pixel>();
        }
        let comp = Rav1dPictureDataComponent::wrap_buf::<BD>(buf.as_mut_slice(), STRIDE);
        println!("{bitdepth_max}:");
        for (tx, w, h) in [
            (TxfmSize::S4x4, 4, 4),
            (TxfmSize::R8x4, 8, 4),
            (TxfmSize::R4x16, 4, 16),
            (TxfmSize::S8x8, 8, 8),
            (TxfmSize::S16x16, 16, 16),
            (TxfmSize::R16x8, 16, 8),
            (TxfmSize::R16x32, 16, 32),
            (TxfmSize::S32x32, 32, 32),
            (TxfmSize::S64x64, 64, 64),
        ] {
            let (cw, ch) = (w.min(32), h.min(32));
            for (t, eob) in [
                (DCT_DCT, cw * ch - 1),
                (ADST_ADST, cw * ch - 1),
                (DCT_DCT, 100),
            ] {
                let unassigned = *dsps[0].1.itxfm_add[tx as usize][t as usize].get() as usize
                    == *itxfm::Fn::DEFAULT.get() as usize;
                if unassigned || (eob == 100 && (w, h) != (32, 32)) {
                    continue;
                }
                let coeff: Vec<BD::Coef> = (0..cw * ch)
                    .map(|i| {
                        if eob < 136 && (i / ch >= 16 || i % ch >= 16) {
                            0
                        } else {
                            (rng.next() % 1024) as i32 - 512
                        }
                    })
                    .map(|c| (c * scale).as_::<BD::Coef>())
                    .collect();
                let eob = eob as i32;
                let mut line = format!("{w}x{h} type {t:2}:");
                for (name, dsp) in &dsps {
                    let f = &dsp.itxfm_add[tx as usize][t as usize];
                    // The asm loads the coefficients aligned, as the decoder's buffers are.
                    let mut c = AlignedVec64::<BD::Coef>::new();
                    c.resize(coeff.len(), 0i16.into());
                    let c = c.as_mut_slice();
                    let mut best = f64::MAX;
                    for _ in 0..20 {
                        let n = 16000 / (w * h);
                        let start = Instant::now();
                        for _ in 0..n {
                            c.copy_from_slice(&coeff);
                            f.call::<BD>(
                                WithOffset {
                                    data: &comp,
                                    offset: 4 * STRIDE + BX,
                                },
                                black_box(&mut *c),
                                eob,
                                BD::from_c(bitdepth_max),
                            );
                        }
                        best = best.min(start.elapsed().as_nanos() as f64 / n as f64);
                    }
                    line += &format!("  {name} {best:6.1} ns");
                }
                println!("{line}");
            }
        }
    }
}
