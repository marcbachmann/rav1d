#![deny(unsafe_op_in_unsafe_fn)]

use std::cmp;
use std::ffi::c_int;

use libc::ptrdiff_t;
use strum::FromRepr;

use crate::align::{Align16, AlignedVec2};
use crate::cpu::CpuFlags;
use crate::disjoint_mut::{DisjointMut, DisjointMutGuard};
use crate::ffi_safe::FFISafe;
use crate::include::common::bitdepth::{AsPrimitive, BitDepth, DynPixel};
use crate::include::common::intops::iclip;
use crate::include::dav1d::picture::{
    FFISafeRav1dPictureDataComponentOffset, Rav1dPictureDataComponentInner,
    Rav1dPictureDataComponentOffset,
};
use crate::internal::Rav1dFrameData;
use crate::lf_mask::Av1FilterLUT;
use crate::strided::Strided as _;
use crate::with_offset::WithOffset;
use crate::wrap_fn_ptr::wrap_fn_ptr;

wrap_fn_ptr!(pub unsafe extern "C" fn loopfilter_sb(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: ptrdiff_t,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    _dst: FFISafeRav1dPictureDataComponentOffset,
    _lvl: WithOffset<*const FFISafe<DisjointMut<AlignedVec2<u8>>>>,
) -> ());

impl loopfilter_sb::Fn {
    pub fn call<BD: BitDepth>(
        &self,
        f: &Rav1dFrameData,
        dst: Rav1dPictureDataComponentOffset,
        mask: &[u32; 3],
        lvl: WithOffset<&DisjointMut<AlignedVec2<u8>>>,
        w: usize,
    ) {
        let dst_ptr = dst.as_mut_ptr::<BD>().cast();
        let stride = dst.stride();
        assert!(lvl.offset <= lvl.data.len());
        // SAFETY: `lvl.offset` is in bounds, just checked above.
        let lvl_ptr = unsafe { lvl.data.as_mut_ptr().add(lvl.offset) };
        let lvl_ptr = lvl_ptr.cast::<[u8; 4]>();
        let b4_stride = f.b4_stride;
        let lut = &f.lf.lim_lut;
        let w = w as c_int;
        let bd = f.bitdepth_max;
        let dst = dst.into_ffi_safe();
        let lvl = lvl.into_ffi_safe();
        // SAFETY: Fallback `fn loop_filter_sb128_rust` is safe; asm is supposed to do the same.
        unsafe {
            self.get()(
                dst_ptr, stride, mask, lvl_ptr, b4_stride, lut, w, bd, dst, lvl,
            )
        }
    }

    const fn default<BD: BitDepth, const HV: usize, const YUV: usize>() -> Self {
        Self::new(loop_filter_sb128_c_erased::<BD, { HV }, { YUV }>)
    }
}

pub struct LoopFilterHVDSPContext {
    pub h: loopfilter_sb::Fn,
    pub v: loopfilter_sb::Fn,
}

pub struct LoopFilterYUVDSPContext {
    pub y: LoopFilterHVDSPContext,
    pub uv: LoopFilterHVDSPContext,
}

pub struct Rav1dLoopFilterDSPContext {
    pub loop_filter_sb: LoopFilterYUVDSPContext,
}

#[inline(never)]
fn loop_filter<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    e: u8,
    i: u8,
    h: u8,
    stridea: ptrdiff_t,
    strideb: ptrdiff_t,
    wd: c_int,
    bd: BD,
) {
    let bitdepth_min_8 = bd.bitdepth() - 8;
    let [f, e, i, h] = [1, e, i, h].map(|n| (n as i32) << bitdepth_min_8);

    for idx in 0..4 {
        let dst = dst + (idx * stridea);
        let dst = |stride_index: isize| (dst + (strideb * stride_index)).index_mut::<BD>();

        let get_dst = |stride_index| (*dst(stride_index)).as_::<i32>();
        let set_dst = |stride_index, pixel: i32| {
            *dst(stride_index) = pixel.as_::<BD::Pixel>();
        };
        let set_dst_clipped = |stride_index, pixel: i32| {
            *dst(stride_index) = bd.iclip_pixel(pixel);
        };

        let mut p6 = 0;
        let mut p5 = 0;
        let mut p4 = 0;
        let mut p3 = 0;
        let mut p2 = 0;
        let p1 = get_dst(-2);
        let p0 = get_dst(-1);
        let q0 = get_dst(0);
        let q1 = get_dst(1);
        let mut q2 = 0;
        let mut q3 = 0;
        let mut q4 = 0;
        let mut q5 = 0;
        let mut q6 = 0;
        let mut flat8out = false;
        let mut flat8in = false;

        let mut fm = (p1 - p0).abs() <= i
            && (q1 - q0).abs() <= i
            && (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1) <= e;

        if wd > 4 {
            p2 = get_dst(-3);
            q2 = get_dst(2);

            fm &= (p2 - p1).abs() <= i && (q2 - q1).abs() <= i;

            if wd > 6 {
                p3 = get_dst(-4);
                q3 = get_dst(3);

                fm &= (p3 - p2).abs() <= i && (q3 - q2).abs() <= i;
            }
        }
        if !fm {
            continue;
        }

        if wd >= 16 {
            p6 = get_dst(-7);
            p5 = get_dst(-6);
            p4 = get_dst(-5);
            q4 = get_dst(4);
            q5 = get_dst(5);
            q6 = get_dst(6);

            flat8out = (p6 - p0).abs() <= f
                && (p5 - p0).abs() <= f
                && (p4 - p0).abs() <= f
                && (q4 - q0).abs() <= f
                && (q5 - q0).abs() <= f
                && (q6 - q0).abs() <= f;
        }

        if wd >= 6 {
            flat8in = (p2 - p0).abs() <= f
                && (p1 - p0).abs() <= f
                && (q1 - q0).abs() <= f
                && (q2 - q0).abs() <= f;
        }

        if wd >= 8 {
            flat8in &= (p3 - p0).abs() <= f && (q3 - q0).abs() <= f;
        }

        if wd >= 16 && flat8out && flat8in {
            set_dst(
                -6,
                p6 + p6 + p6 + p6 + p6 + p6 * 2 + p5 * 2 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + 8 >> 4,
            );
            set_dst(
                -5,
                p6 + p6 + p6 + p6 + p6 + p5 * 2 + p4 * 2 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + 8 >> 4,
            );
            set_dst(
                -4,
                p6 + p6 + p6 + p6 + p5 + p4 * 2 + p3 * 2 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + 8 >> 4,
            );
            set_dst(
                -3,
                p6 + p6 + p6 + p5 + p4 + p3 * 2 + p2 * 2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + 8 >> 4,
            );
            set_dst(
                -2,
                p6 + p6 + p5 + p4 + p3 + p2 * 2 + p1 * 2 + p0 * 2 + q0 + q1 + q2 + q3 + q4 + 8 >> 4,
            );
            set_dst(
                -1,
                p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + q2 + q3 + q4 + q5 + 8 >> 4,
            );
            set_dst(
                0,
                p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + q3 + q4 + q5 + q6 + 8 >> 4,
            );
            set_dst(
                1,
                p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 * 2 + q2 * 2 + q3 + q4 + q5 + q6 + q6 + 8 >> 4,
            );
            set_dst(
                2,
                p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 * 2 + q3 * 2 + q4 + q5 + q6 + q6 + q6 + 8 >> 4,
            );
            set_dst(
                3,
                p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 * 2 + q4 * 2 + q5 + q6 + q6 + q6 + q6 + 8 >> 4,
            );
            set_dst(
                4,
                p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 * 2 + q5 * 2 + q6 + q6 + q6 + q6 + q6 + 8 >> 4,
            );
            set_dst(
                5,
                p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 * 2 + q6 * 2 + q6 + q6 + q6 + q6 + q6 + 8 >> 4,
            );
        } else if wd >= 8 && flat8in {
            set_dst(-3, p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0 + 4 >> 3);
            set_dst(-2, p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4 >> 3);
            set_dst(-1, p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4 >> 3);
            set_dst(0, p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4 >> 3);
            set_dst(1, p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3 + 4 >> 3);
            set_dst(2, p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3 + 4 >> 3);
        } else if wd == 6 && flat8in {
            set_dst(-2, p2 + 2 * p2 + 2 * p1 + 2 * p0 + q0 + 4 >> 3);
            set_dst(-1, p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4 >> 3);
            set_dst(0, p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4 >> 3);
            set_dst(1, p0 + 2 * q0 + 2 * q1 + 2 * q2 + q2 + 4 >> 3);
        } else {
            let hev = (p1 - p0).abs() > h || (q1 - q0).abs() > h;

            fn iclip_diff(v: c_int, bitdepth_min_8: u8) -> i32 {
                iclip(
                    v,
                    -128 * (1 << bitdepth_min_8),
                    128 * (1 << bitdepth_min_8) - 1,
                )
            }

            if hev {
                let f = iclip_diff(p1 - q1, bitdepth_min_8);
                let f = iclip_diff(3 * (q0 - p0) + f, bitdepth_min_8);

                let f1 = cmp::min(f + 4, (128 << bitdepth_min_8) - 1) >> 3;
                let f2 = cmp::min(f + 3, (128 << bitdepth_min_8) - 1) >> 3;

                set_dst_clipped(-1, p0 + f2);
                set_dst_clipped(0, q0 - f1);
            } else {
                let f = iclip_diff(3 * (q0 - p0), bitdepth_min_8);

                let f1 = cmp::min(f + 4, (128 << bitdepth_min_8) - 1) >> 3;
                let f2 = cmp::min(f + 3, (128 << bitdepth_min_8) - 1) >> 3;

                set_dst_clipped(-1, p0 + f2);
                set_dst_clipped(0, q0 - f1);

                let f = (f1 + 1) >> 1;
                set_dst_clipped(-2, p1 + f);
                set_dst_clipped(1, q1 - f);
            }
        }
    }
}

/// [`loop_filter_sb128_rust`] sixteen lines of an edge at a time at 8 bpc, in byte lanes as
/// dav1d's asm, and eight at a time at 10 bits (sixteen on AVX2): four (two) 4-line
/// segments, each with its own levels and filter length. Every lane takes the narrow filter and
/// the flat filters the group's longest length needs, and masks per lane pick its result, as the
/// C's branches would. Along a vertical edge the lines are rows, turned into lanes by transposes;
/// along a horizontal one they are columns, loaded as they lie.
mod rust_simd {
    use zerocopy::{AsBytes, FromBytes};

    use super::*;
    use crate::simd128::{rows_fit, transpose8x8, Bytes, I16x8, Lanes, U8x16};
    #[cfg(target_arch = "x86_64")]
    use crate::simd128::{I16x16, U8x32};

    /// A lane type with its pixel loads and stores.
    pub trait Px: Lanes {
        /// Lines per vector.
        const N: usize;
        /// Bytes `b[..N]` zero-extended.
        fn load_px(b: &[u8]) -> Self;
        /// The low `N` lanes of `b` zero-extended.
        fn from_bytes(b: U8x16) -> Self;
        /// [`Px::from_bytes`] of a mask: all ones where `b` is.
        fn from_mask(b: U8x16) -> Self;
        /// The lanes saturated into `d[..N]`.
        fn store_px(self, d: &mut [u8]);
        /// Eight bytes of `a`, and on sixteen lanes eight of `b` in the high half: a row window
        /// of two lines eight apart.
        fn load_rows(a: &[u8], b: &[u8]) -> Self;
        fn store_rows(self, a: &mut [u8], b: &mut [u8]);
        /// `s[..N]`: sixteen-bit pixels, or a lane's thresholds.
        fn load_i16(s: &[i16]) -> Self;
        fn store_i16(self, d: &mut [i16]);
        /// [`Px::load_rows`] for sixteen-bit pixels.
        fn load_rows16(a: &[i16], b: &[i16]) -> Self;
        fn store_rows16(self, a: &mut [i16], b: &mut [i16]);
        /// The lanes shifted right by `n`, zeros in.
        fn shr_var(self, n: u32) -> Self;
        /// `N` rows of eight pixels from `b[start + r * stride..]` as eight columns of `N`.
        fn load_cols8(b: &[i16], start: usize, stride: isize) -> [Self; 8];
        fn store_cols8(c: &[Self; 8], b: &mut [i16], start: usize, stride: isize);
        /// [`Px::load_cols8`] of four pixels per row.
        fn load_cols4(b: &[i16], start: usize, stride: isize) -> [Self; 4];
        /// [`Px::store_cols8`] of four columns.
        fn store_cols4(c: &[Self; 4], b: &mut [i16], start: usize, stride: isize);
    }

    /// Row `r` of `w` pixels from `start`, the rows checked by [`rows_fit`] for `w`.
    #[inline(always)]
    fn row16(b: &[i16], start: usize, stride: isize, r: usize, w: usize) -> &[i16] {
        // SAFETY: The caller checked the rows with `rows_fit`.
        unsafe { std::slice::from_raw_parts(b.as_ptr().add(start).offset(r as isize * stride), w) }
    }

    #[inline(always)]
    fn row16_mut(b: &mut [i16], start: usize, stride: isize, r: usize, w: usize) -> &mut [i16] {
        // SAFETY: The caller checked the rows with `rows_fit`.
        unsafe {
            std::slice::from_raw_parts_mut(b.as_mut_ptr().add(start).offset(r as isize * stride), w)
        }
    }

    /// The rows of four columns `c` holds, two rows per vector (per 128-bit half).
    #[inline(always)]
    fn transpose4<V: Lanes>(c: &[V; 4]) -> [V; 4] {
        let (a0, a1) = (c[0].zip_lo16(c[1]), c[0].zip_hi16(c[1]));
        let (b0, b1) = (c[2].zip_lo16(c[3]), c[2].zip_hi16(c[3]));
        [a0.zip_lo32(b0), a0.zip_hi32(b0), a1.zip_lo32(b1), a1.zip_hi32(b1)]
    }

    /// Rows `r` and `r + 1` from the halves of `v`.
    #[inline(always)]
    fn store_pair(v: I16x8, b: &mut [i16], start: usize, stride: isize, r: usize) {
        let (lo, hi) = (r as isize * stride, (r + 1) as isize * stride);
        // SAFETY: The caller checked the rows with `rows_fit`.
        let (lo, hi) = unsafe {
            let p = b.as_mut_ptr().add(start);
            (
                std::slice::from_raw_parts_mut(p.offset(lo), 4),
                std::slice::from_raw_parts_mut(p.offset(hi), 4),
            )
        };
        v.store_halves(lo, hi);
    }

    /// The four columns of rows `k`, `k + 4`, .. in the 64-bit halves of `c[k]`.
    #[inline(always)]
    fn cols4<V: Lanes>(c: [V; 4]) -> [V; 4] {
        let (a0, a1) = (c[0].zip_lo16(c[1]), c[0].zip_hi16(c[1]));
        let (b0, b1) = (c[2].zip_lo16(c[3]), c[2].zip_hi16(c[3]));
        let (o0, o1) = (a0.zip_lo32(b0), a0.zip_hi32(b0));
        let (o2, o3) = (a1.zip_lo32(b1), a1.zip_hi32(b1));
        [o0.zip_lo64(o2), o0.zip_hi64(o2), o1.zip_lo64(o3), o1.zip_hi64(o3)]
    }

    impl Px for I16x8 {
        const N: usize = 8;
        #[inline(always)]
        fn load_px(b: &[u8]) -> Self {
            I16x8::from_u8x8(b)
        }
        #[inline(always)]
        fn from_bytes(b: U8x16) -> Self {
            b.widen()[0]
        }
        #[inline(always)]
        fn from_mask(b: U8x16) -> Self {
            b.widen()[0].eq(Self::splat(0xff))
        }
        #[inline(always)]
        fn store_px(self, d: &mut [u8]) {
            d[..8].copy_from_slice(&self.pack_u8());
        }
        #[inline(always)]
        fn load_rows(a: &[u8], _: &[u8]) -> Self {
            I16x8::from_u8x8(a)
        }
        #[inline(always)]
        fn store_rows(self, a: &mut [u8], _: &mut [u8]) {
            a[..8].copy_from_slice(&self.pack_u8());
        }
        #[inline(always)]
        fn load_i16(s: &[i16]) -> Self {
            I16x8::load(s)
        }
        #[inline(always)]
        fn store_i16(self, d: &mut [i16]) {
            self.store(d)
        }
        #[inline(always)]
        fn load_rows16(a: &[i16], _: &[i16]) -> Self {
            I16x8::load(a)
        }
        #[inline(always)]
        fn store_rows16(self, a: &mut [i16], _: &mut [i16]) {
            self.store(a)
        }
        #[inline(always)]
        fn shr_var(self, n: u32) -> Self {
            self.shr_u(n)
        }
        #[inline(always)]
        fn load_cols8(b: &[i16], start: usize, stride: isize) -> [Self; 8] {
            rows_fit(b.len(), start, stride, 8, 8);
            let mut v = [I16x8::splat(0); 8];
            for (r, v) in v.iter_mut().enumerate() {
                *v = I16x8::load(row16(b, start, stride, r, 8));
            }
            transpose8x8(&mut v);
            v
        }
        #[inline(always)]
        fn store_cols8(c: &[Self; 8], b: &mut [i16], start: usize, stride: isize) {
            rows_fit(b.len(), start, stride, 8, 8);
            let mut v = *c;
            transpose8x8(&mut v);
            for (r, v) in v.iter().enumerate() {
                v.store(row16_mut(b, start, stride, r, 8));
            }
        }
        #[inline(always)]
        fn load_cols4(b: &[i16], start: usize, stride: isize) -> [Self; 4] {
            rows_fit(b.len(), start, stride, 8, 4);
            let row = |r| row16(b, start, stride, r, 4);
            let mut c = [I16x8::splat(0); 4];
            for (k, c) in c.iter_mut().enumerate() {
                *c = I16x8::load_halves(row(k), row(k + 4));
            }
            cols4(c)
        }
        #[inline(always)]
        fn store_cols4(c: &[Self; 4], b: &mut [i16], start: usize, stride: isize) {
            rows_fit(b.len(), start, stride, 8, 4);
            for (i, v) in transpose4(c).into_iter().enumerate() {
                store_pair(v, b, start, stride, 2 * i);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    impl Px for I16x16 {
        const N: usize = 16;
        #[inline(always)]
        fn load_px(b: &[u8]) -> Self {
            I16x16::from_u8x16(b)
        }
        #[inline(always)]
        fn from_bytes(b: U8x16) -> Self {
            I16x16::widen_u8(b)
        }
        #[inline(always)]
        fn from_mask(b: U8x16) -> Self {
            I16x16::widen_i8(b)
        }
        #[inline(always)]
        fn store_px(self, d: &mut [u8]) {
            d[..16].copy_from_slice(&self.pack_u8());
        }
        #[inline(always)]
        fn load_rows(a: &[u8], b: &[u8]) -> Self {
            I16x16::from_halves(I16x8::from_u8x8(a), I16x8::from_u8x8(b))
        }
        #[inline(always)]
        fn store_rows(self, a: &mut [u8], b: &mut [u8]) {
            let px = self.pack_u8();
            a[..8].copy_from_slice(&px[..8]);
            b[..8].copy_from_slice(&px[8..]);
        }
        #[inline(always)]
        fn load_i16(s: &[i16]) -> Self {
            I16x16::load(s)
        }
        #[inline(always)]
        fn store_i16(self, d: &mut [i16]) {
            self.store(d)
        }
        #[inline(always)]
        fn load_rows16(a: &[i16], b: &[i16]) -> Self {
            I16x16::from_halves(I16x8::load(a), I16x8::load(b))
        }
        #[inline(always)]
        fn store_rows16(self, a: &mut [i16], b: &mut [i16]) {
            self.lo().store(a);
            self.hi().store(b);
        }
        #[inline(always)]
        fn shr_var(self, n: u32) -> Self {
            self.shr_u(n)
        }
        /// Rows `r` and `r + 8` in the halves of a vector, then [`transpose8x8`] on both.
        #[inline(always)]
        fn load_cols8(b: &[i16], start: usize, stride: isize) -> [Self; 8] {
            rows_fit(b.len(), start, stride, 16, 8);
            let mut v = [I16x16::splat(0); 8];
            for (r, v) in v.iter_mut().enumerate() {
                *v = I16x16::from_halves(
                    I16x8::load(row16(b, start, stride, r, 8)),
                    I16x8::load(row16(b, start, stride, r + 8, 8)),
                );
            }
            transpose8x8(&mut v);
            v
        }
        #[inline(always)]
        fn store_cols8(c: &[Self; 8], b: &mut [i16], start: usize, stride: isize) {
            rows_fit(b.len(), start, stride, 16, 8);
            let mut v = *c;
            transpose8x8(&mut v);
            for (r, v) in v.iter().enumerate() {
                v.lo().store(row16_mut(b, start, stride, r, 8));
                v.hi().store(row16_mut(b, start, stride, r + 8, 8));
            }
        }
        #[inline(always)]
        fn load_cols4(b: &[i16], start: usize, stride: isize) -> [Self; 4] {
            rows_fit(b.len(), start, stride, 16, 4);
            let row = |r| row16(b, start, stride, r, 4);
            let mut c = [I16x16::splat(0); 4];
            for (k, c) in c.iter_mut().enumerate() {
                *c = I16x16::from_halves(
                    I16x8::load_halves(row(k), row(k + 4)),
                    I16x8::load_halves(row(k + 8), row(k + 12)),
                );
            }
            cols4(c)
        }
        #[inline(always)]
        fn store_cols4(c: &[Self; 4], b: &mut [i16], start: usize, stride: isize) {
            rows_fit(b.len(), start, stride, 16, 4);
            for (i, v) in transpose4(c).into_iter().enumerate() {
                store_pair(v.lo(), b, start, stride, 2 * i);
                store_pair(v.hi(), b, start, stride, 2 * i + 8);
            }
        }
    }

    /// A picture row's sixteen-bit pixels as lanes.
    #[inline(always)]
    fn px16(b: &[u8]) -> &[i16] {
        FromBytes::slice_from(b).unwrap()
    }

    #[inline(always)]
    fn px16_mut(b: &mut [u8]) -> &mut [i16] {
        FromBytes::mut_slice_from(b).unwrap()
    }

    /// A group's thresholds and filter-length masks, per lane.
    #[derive(Clone, Copy)]
    struct Params<V> {
        e: V,
        i: V,
        h: V,
        active: V,
        /// Filter length above 4 (6, 8 or 16), and 16. A group whose longest filter is 8 or 16 is
        /// luma, where no length is 6, so above 4 is also above 6.
        gt4: V,
        wd16: V,
        /// The flat test's limit, the narrow filter's clip range and the pixel maximum, scaled to
        /// the bit depth.
        flat: V,
        lo: V,
        hi: V,
        max: V,
    }

    #[inline(always)]
    fn le<V: Lanes>(a: V, b: V) -> V {
        a.max(b).eq(b)
    }

    #[inline(always)]
    fn absdiff<V: Lanes>(a: V, b: V) -> V {
        a.sub(b).abs()
    }

    #[inline(always)]
    fn rshr<V: Lanes, const N: i32>(v: V) -> V {
        v.add(V::splat(1 << (N - 1))).shr_s::<N>()
    }

    /// The sums the flat filters keep: lanes, or a pair of 16-bit halves of byte lanes.
    trait Sum: Copy {
        fn plus(self, o: Self) -> Self;
        fn minus(self, o: Self) -> Self;
    }

    impl<V: Lanes> Sum for V {
        #[inline(always)]
        fn plus(self, o: Self) -> Self {
            self.add(o)
        }
        #[inline(always)]
        fn minus(self, o: Self) -> Self {
            self.sub(o)
        }
    }

    /// [`Bytes::widen`]ed lanes.
    #[derive(Clone, Copy)]
    struct Wide<H>([H; 2]);

    impl<H: Lanes> Sum for Wide<H> {
        #[inline(always)]
        fn plus(self, o: Self) -> Self {
            Self([self.0[0].add(o.0[0]), self.0[1].add(o.0[1])])
        }
        #[inline(always)]
        fn minus(self, o: Self) -> Self {
            Self([self.0[0].sub(o.0[0]), self.0[1].sub(o.0[1])])
        }
    }

    /// Where a flat filter's output `k` goes, as soon as it is summed: one at a time keeps few
    /// sums live.
    trait Sink<S> {
        fn put(&mut self, k: usize, s: S);
    }

    /// `px[k]` rounded from the sum where `m` is set, `>> N`.
    struct Select<'a, V, const N: i32> {
        px: &'a mut [V; 14],
        m: V,
    }

    impl<V: Lanes, const N: i32> Sink<V> for Select<'_, V, N> {
        #[inline(always)]
        fn put(&mut self, k: usize, s: V) {
            self.px[k] = V::select(self.m, rshr::<V, N>(s), self.px[k]);
        }
    }

    /// [`Select`] of byte lanes from their [`Wide`] sums.
    struct SelectBytes<'a, B, const N: i32> {
        px: &'a mut [B; 14],
        m: B,
    }

    impl<B: Bytes, const N: i32> Sink<Wide<B::Wide>> for SelectBytes<'_, B, N> {
        #[inline(always)]
        fn put(&mut self, k: usize, s: Wide<B::Wide>) {
            let v = B::narrow_rshr::<N>(s.0[0], s.0[1]);
            self.px[k] = B::select(self.m, v, self.px[k]);
        }
    }

    /// Generated from `loop_filter`'s flat16 branch: one running sum, a term in and out per output,
    /// each sum handed to `out` before [`rshr`]`::<4>` as soon as it is complete.
    #[inline(always)]
    fn flat16<S: Sum>(px: &[S; 14], out: &mut impl Sink<S>) {
        let mut s = px[0]
            .plus(px[0])
            .plus(px[0])
            .plus(px[0])
            .plus(px[0])
            .plus(px[0])
            .plus(px[0])
            .plus(px[1])
            .plus(px[1])
            .plus(px[2])
            .plus(px[2])
            .plus(px[3])
            .plus(px[4])
            .plus(px[5])
            .plus(px[6])
            .plus(px[7]);
        out.put(1, s);
        s = s.minus(px[0]).minus(px[0]).plus(px[3]).plus(px[8]);
        out.put(2, s);
        s = s.minus(px[0]).minus(px[1]).plus(px[4]).plus(px[9]);
        out.put(3, s);
        s = s.minus(px[0]).minus(px[2]).plus(px[5]).plus(px[10]);
        out.put(4, s);
        s = s.minus(px[0]).minus(px[3]).plus(px[6]).plus(px[11]);
        out.put(5, s);
        s = s.minus(px[0]).minus(px[4]).plus(px[7]).plus(px[12]);
        out.put(6, s);
        s = s.minus(px[0]).minus(px[5]).plus(px[8]).plus(px[13]);
        out.put(7, s);
        s = s.minus(px[1]).minus(px[6]).plus(px[9]).plus(px[13]);
        out.put(8, s);
        s = s.minus(px[2]).minus(px[7]).plus(px[10]).plus(px[13]);
        out.put(9, s);
        s = s.minus(px[3]).minus(px[8]).plus(px[11]).plus(px[13]);
        out.put(10, s);
        s = s.minus(px[4]).minus(px[9]).plus(px[12]).plus(px[13]);
        out.put(11, s);
        s = s.minus(px[5]).minus(px[10]).plus(px[13]).plus(px[13]);
        out.put(12, s);
    }

    /// Generated from `loop_filter`'s flat8 branch: one running sum, a term in and out per output,
    /// each sum handed to `out` before [`rshr`]`::<3>` as soon as it is complete.
    #[inline(always)]
    fn flat8<S: Sum>(px: &[S; 14], out: &mut impl Sink<S>) {
        let mut s = px[3]
            .plus(px[3])
            .plus(px[3])
            .plus(px[4])
            .plus(px[4])
            .plus(px[5])
            .plus(px[6])
            .plus(px[7]);
        out.put(4, s);
        s = s.minus(px[3]).minus(px[4]).plus(px[5]).plus(px[8]);
        out.put(5, s);
        s = s.minus(px[3]).minus(px[5]).plus(px[6]).plus(px[9]);
        out.put(6, s);
        s = s.minus(px[3]).minus(px[6]).plus(px[7]).plus(px[10]);
        out.put(7, s);
        s = s.minus(px[4]).minus(px[7]).plus(px[8]).plus(px[10]);
        out.put(8, s);
        s = s.minus(px[5]).minus(px[8]).plus(px[9]).plus(px[10]);
        out.put(9, s);
    }

    /// Generated from `loop_filter`'s flat6 branch: one running sum, a term in and out per output,
    /// each sum handed to `out` before [`rshr`]`::<3>` as soon as it is complete.
    #[inline(always)]
    fn flat6<S: Sum>(px: &[S; 14], out: &mut impl Sink<S>) {
        let mut s = px[4]
            .plus(px[4])
            .plus(px[4])
            .plus(px[5])
            .plus(px[5])
            .plus(px[6])
            .plus(px[6])
            .plus(px[7]);
        out.put(5, s);
        s = s.minus(px[4]).minus(px[4]).plus(px[7]).plus(px[8]);
        out.put(6, s);
        s = s.minus(px[4]).minus(px[5]).plus(px[8]).plus(px[9]);
        out.put(7, s);
        s = s.minus(px[5]).minus(px[6]).plus(px[9]).plus(px[9]);
        out.put(8, s);
    }

    /// `px[7 + k]` is the pixel at offset `k` across the edge, `-7..=6`; `MAX` is the group's
    /// longest filter, and lanes of a shorter one ignore what it adds.
    #[inline(always)]
    fn filter<V: Lanes, const MAX: usize>(px: &mut [V; 14], p: Params<V>) -> usize {
        let orig = *px;
        let [p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6] = orig;
        let (zero, one) = (V::splat(0), V::splat(1));
        let (dp1, dq1) = (absdiff(p1, p0), absdiff(q1, q0));
        let inner = dp1.max(dq1);
        let mut m = inner;
        if MAX > 4 {
            let mut c = absdiff(p2, p1).max(absdiff(q2, q1));
            if MAX > 6 {
                c = c.max(absdiff(p3, p2)).max(absdiff(q3, q2));
            }
            m = m.max(c.and(p.gt4));
        }
        let d = absdiff(p0, q0);
        let fm = le(m, p.i)
            .and(le(d.add(d).add(absdiff(p1, q1).shr_s::<1>()), p.e))
            .and(p.active);
        // As the asm, each step only where some lane takes it.
        if !fm.any() {
            return 0;
        }

        let nhev = le(inner, p.h);
        let hev = nhev.eq(zero);
        let (lo, hi) = (p.lo, p.hi);
        let f = p1.sub(q1).max(lo).min(hi).and(hev);
        let d = q0.sub(p0);
        let f = d.add(d).add(d).add(f).max(lo).min(hi);
        let f1 = f.add(V::splat(4)).min(hi).shr_s::<3>();
        let f2 = f.add(V::splat(3)).min(hi).shr_s::<3>();
        let f3 = f1.add(one).shr_s::<1>().and(nhev);
        px[5] = V::select(fm, p1.add(f3), p1);
        px[6] = V::select(fm, p0.add(f2), p0);
        px[7] = V::select(fm, q0.sub(f1), q0);
        px[8] = V::select(fm, q1.sub(f3), q1);

        if MAX < 6 {
            return 2;
        }
        let mut flat = absdiff(p2, p0).max(dp1).max(dq1).max(absdiff(q2, q0));
        if MAX > 6 {
            flat = flat.max(absdiff(p3, p0)).max(absdiff(q3, q0));
        }
        let mut m8 = le(flat, p.flat).and(fm).and(p.gt4);
        if !m8.any() {
            return 2;
        }
        if MAX == 6 {
            flat6(&orig, &mut Select::<V, 3> { px, m: m8 });
            return 2;
        }
        let mut changed = 3;
        if MAX == 16 && m8.and(p.wd16).any() {
            let flat_out = absdiff(p6, p0)
                .max(absdiff(p5, p0))
                .max(absdiff(p4, p0))
                .max(absdiff(q4, q0))
                .max(absdiff(q5, q0))
                .max(absdiff(q6, q0));
            let m16 = le(flat_out, p.flat).and(m8).and(p.wd16);
            m8 = m8.and(m16.eq(zero));
            if m16.any() {
                flat16(&orig, &mut Select::<V, 4> { px, m: m16 });
                changed = 6;
            }
            if !m8.any() {
                return changed;
            }
        }
        flat8(&orig, &mut Select::<V, 3> { px, m: m8 });
        changed
    }

    /// [`Params`] at 8 bpc, on byte lanes.
    #[derive(Clone, Copy)]
    struct Params8<B> {
        e: B,
        i: B,
        h: B,
        active: B,
        /// Filter length above 4, and 16. A group whose longest filter is 8 or 16 is luma, where
        /// no length is 6, so above 4 is also above 6.
        gt4: B,
        wd16: B,
    }

    /// [`filter`] at 8 bpc as dav1d's asm: masks and the narrow filter on bytes, the latter
    /// signed after `^ 0x80` with saturating adds, which clip as `iclip_diff` and the pixel clip
    /// do; only the flat filters' sums widen to 16 bits. Returns how many pixels on each side
    /// of the edge it may have changed, so that, as in the asm, only those are stored: 0, 2
    /// (narrow and flat6), 3 (flat8) or 6 (flat16).
    #[inline(always)]
    fn filter8<B: Bytes, const MAX: usize>(px: &mut [B; 14], p: Params8<B>) -> usize {
        let orig = *px;
        let [p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6] = orig;
        let one = B::splat(1);
        let (dp1, dq1) = (p1.abd(p0), q1.abd(q0));
        let inner = dp1.max(dq1);
        let mut m = inner;
        if MAX > 4 {
            let mut c = p2.abd(p1).max(q2.abd(q1));
            if MAX > 6 {
                c = c.max(p3.abd(p2)).max(q3.abd(q2));
            }
            m = m.max(c.and(p.gt4));
        }
        let d = p0.abd(q0);
        let fm = m
            .le(p.i)
            .and(d.add_sat(d).add_sat(p1.abd(q1).shr_u::<1>()).le(p.e))
            .and(p.active);
        if !fm.any() {
            return 0;
        }

        // A sum of the same sign saturated step by step clips as the sum would: once past a
        // bound, every further step stays past it.
        let nhev = inner.le(p.h);
        let x80 = B::splat(0x80);
        let (sp1, sp0, sq0, sq1) = (p1.xor(x80), p0.xor(x80), q0.xor(x80), q1.xor(x80));
        let f = sp1.sub_sat_i(sq1).andnot(nhev);
        let d = sq0.sub_sat_i(sp0);
        let f = f.add_sat_i(d).add_sat_i(d).add_sat_i(d);
        let f1 = f.add_sat_i(B::splat(4)).shr_i::<3>();
        let f2 = f.add_sat_i(B::splat(3)).shr_i::<3>();
        let f3 = f1.add(one).shr_i::<1>().and(nhev);
        px[5] = B::select(fm, sp1.add_sat_i(f3).xor(x80), p1);
        px[6] = B::select(fm, sp0.add_sat_i(f2).xor(x80), p0);
        px[7] = B::select(fm, sq0.sub_sat_i(f1).xor(x80), q0);
        px[8] = B::select(fm, sq1.sub_sat_i(f3).xor(x80), q1);

        if MAX < 6 {
            return 2;
        }
        let mut flat = p2.abd(p0).max(dp1).max(dq1).max(q2.abd(q0));
        if MAX > 6 {
            flat = flat.max(p3.abd(p0)).max(q3.abd(q0));
        }
        let mut m8 = flat.le(one).and(fm).and(p.gt4);
        if !m8.any() {
            return 2;
        }
        let mut wide = [Wide([B::Wide::splat(0); 2]); 14];
        for (w, v) in wide.iter_mut().zip(orig) {
            *w = Wide(v.widen());
        }
        if MAX == 6 {
            flat6(&wide, &mut SelectBytes::<_, 3> { px, m: m8 });
            return 2;
        }
        let mut changed = 3;
        if MAX == 16 && m8.and(p.wd16).any() {
            let flat_out = p6
                .abd(p0)
                .max(p5.abd(p0))
                .max(p4.abd(p0))
                .max(q4.abd(q0))
                .max(q5.abd(q0))
                .max(q6.abd(q0));
            let m16 = flat_out.le(one).and(m8).and(p.wd16);
            m8 = m8.andnot(m16);
            if m16.any() {
                flat16(&wide, &mut SelectBytes::<_, 4> { px, m: m16 });
                changed = 6;
            }
            if !m8.any() {
                return changed;
            }
        }
        flat8(&wide, &mut SelectBytes::<_, 3> { px, m: m8 });
        changed
    }

    /// The pixels on each side of the edge the filter of length `max` reads.
    const fn reach(max: usize) -> usize {
        match max {
            4 => 2,
            6 => 3,
            8 => 4,
            _ => 7,
        }
    }

    /// Eight pixels from `a`, and on sixteen lanes eight from `b` in the high half.
    #[inline(always)]
    fn load_rows<BD: BitDepth, V: Px>(
        a: Rav1dPictureDataComponentOffset,
        b: Rav1dPictureDataComponentOffset,
    ) -> V {
        let a = a.slice::<BD>(8);
        let a = a.as_bytes();
        match (V::N, BD::BITDEPTH) {
            (8, 8) => V::load_rows(a, &[]),
            (8, _) => V::load_rows16(px16(a), &[]),
            (_, 8) => V::load_rows(a, b.slice::<BD>(8).as_bytes()),
            _ => V::load_rows16(px16(a), px16(b.slice::<BD>(8).as_bytes())),
        }
    }

    /// [`load_rows`] back, sixteen-bit pixels clipped to `max`.
    #[inline(always)]
    fn store_rows<BD: BitDepth, V: Px>(
        v: V,
        a: Rav1dPictureDataComponentOffset,
        b: Rav1dPictureDataComponentOffset,
        max: V,
    ) {
        let mut a = a.slice_mut::<BD>(8);
        let a = a.as_bytes_mut();
        if BD::BITDEPTH == 8 {
            return if V::N == 8 {
                v.store_rows(a, &mut [])
            } else {
                v.store_rows(a, b.slice_mut::<BD>(8).as_bytes_mut())
            };
        }
        let v = v.max(V::splat(0)).min(max);
        if V::N == 8 {
            v.store_rows16(px16_mut(a), &mut [])
        } else {
            v.store_rows16(px16_mut(a), px16_mut(b.slice_mut::<BD>(8).as_bytes_mut()))
        }
    }

    /// The narrow filter's outputs `px[5..9]` clipped to the pixels' range; the flat filters'
    /// stay in it.
    #[inline(always)]
    fn clip_narrow<V: Lanes>(px: &mut [V; 14], max: V) {
        for v in &mut px[5..9] {
            *v = v.max(V::splat(0)).min(max);
        }
    }

    /// [`group_v`] on sixteen-bit pixels of the block from `start`, the rows `stride` apart.
    #[inline(always)]
    fn group_v16<V: Px, const MAX: usize>(
        b: &mut [i16],
        start: usize,
        stride: isize,
        p: Params<V>,
    ) {
        let r = reach(MAX);
        rows_fit(b.len(), start, stride, 2 * r, V::N);
        let mut px = [V::splat(0); 14];
        for k in 0..2 * r {
            px[7 - r + k] = V::load_i16(row16(b, start, stride, k, V::N));
        }
        let changed = filter::<V, MAX>(&mut px, p);
        if changed == 0 {
            return;
        }
        clip_narrow(&mut px, p.max);
        let w = match changed {
            2 => 2,
            3 => 3,
            _ if MAX == 16 => 6,
            _ => 3,
        };
        for k in r - w..r + w {
            px[7 - r + k].store_i16(row16_mut(b, start, stride, k, V::N));
        }
    }

    /// [`group_h`] on sixteen-bit pixels: `V::N` rows of the block from `start`, 8 (16) pixels
    /// from 4 (8) left of the edge, stored back only as wide as the filter changed them.
    #[inline(always)]
    fn group_h16<V: Px, const MAX: usize>(
        b: &mut [i16],
        start: usize,
        stride: isize,
        p: Params<V>,
    ) {
        let mut px = [V::splat(0); 14];
        if MAX == 4 {
            // As the asm, only the four columns the filter reads, from `start`.
            px[5..9].copy_from_slice(&V::load_cols4(b, start, stride));
            if filter::<V, MAX>(&mut px, p) != 0 {
                clip_narrow(&mut px, p.max);
                V::store_cols4(&[px[5], px[6], px[7], px[8]], b, start, stride);
            }
            return;
        }
        let (n, x0) = if MAX == 16 { (2, -8) } else { (1, -4) };
        let mut win = [[V::splat(0); 8]; 2];
        for (w, win) in win.iter_mut().enumerate().take(n) {
            *win = V::load_cols8(b, start + 8 * w, stride);
        }
        for k in -7..7isize {
            let c = (k - x0) as usize;
            if c < 8 * n {
                px[(7 + k) as usize] = win[c / 8][c % 8];
            }
        }
        let changed = filter::<V, MAX>(&mut px, p);
        if changed == 0 {
            return;
        }
        clip_narrow(&mut px, p.max);
        let at = |x: isize| start + (x - x0) as usize;
        if changed == 2 {
            return V::store_cols4(&[px[5], px[6], px[7], px[8]], b, at(-2), stride);
        }
        if MAX != 16 || changed < 6 {
            let c = [px[3], px[4], px[5], px[6], px[7], px[8], px[9], px[10]];
            return V::store_cols8(&c, b, at(-4), stride);
        }
        for k in -7..7isize {
            let c = (k - x0) as usize;
            win[c / 8][c % 8] = px[(7 + k) as usize];
        }
        for (w, win) in win.iter().enumerate() {
            V::store_cols8(win, b, start + 8 * w, stride);
        }
    }

    /// One group along a horizontal edge: lanes are the columns from `dst`.
    #[inline(always)]
    fn group_v<BD: BitDepth, V: Px, const MAX: usize>(
        dst: Rav1dPictureDataComponentOffset,
        p: Params<V>,
    ) {
        let stride = dst.pixel_stride::<BD>();
        let r = reach(MAX) as isize;
        let mut px = [V::splat(0); 14];
        for k in -r..r {
            let row = (dst + k * stride).slice::<BD>(V::N);
            px[(7 + k) as usize] = if BD::BITDEPTH == 8 {
                V::load_px(row.as_bytes())
            } else {
                V::load_i16(px16(row.as_bytes()))
            };
        }
        let store = |px: &[V; 14], w: isize| {
            for k in -w..w {
                let mut row = (dst + k * stride).slice_mut::<BD>(V::N);
                let v = px[(7 + k) as usize];
                if BD::BITDEPTH == 8 {
                    v.store_px(row.as_bytes_mut());
                } else {
                    v.max(V::splat(0))
                        .min(p.max)
                        .store_i16(px16_mut(row.as_bytes_mut()));
                }
            }
        };
        // As the asm, only the pixels the filter may have changed.
        match filter::<V, MAX>(&mut px, p) {
            0 => {}
            2 => store(&px, 2),
            3 => store(&px, 3),
            _ if MAX == 16 => store(&px, 6),
            _ => {}
        }
    }

    /// One group along a vertical edge: lanes are the rows from `dst`, turned into lanes in 8x8
    /// blocks of the windows from 4 (or 8) pixels left of the edge.
    #[inline(always)]
    fn group_h<BD: BitDepth, V: Px, const MAX: usize>(
        dst: Rav1dPictureDataComponentOffset,
        p: Params<V>,
    ) {
        let stride = dst.pixel_stride::<BD>();
        let (n, x0) = if MAX == 16 { (2, -8) } else { (1, -4) };
        let mut win = [[V::splat(0); 8]; 2];
        for (w, win) in win.iter_mut().enumerate().take(n) {
            let x = x0 + 8 * w as isize;
            for (r, v) in win.iter_mut().enumerate() {
                let a = dst + (r as isize * stride + x);
                let b = dst + ((r + 8) as isize * stride + x);
                *v = load_rows::<BD, V>(a, b);
            }
            transpose8x8(win);
        }
        // `px[7 + k]` is window column `k - x0`; -8 and 7 lie outside every filter's reach.
        let mut px = [V::splat(0); 14];
        for k in -7..7isize {
            let c = (k - x0) as usize;
            if c < 8 * n {
                px[(7 + k) as usize] = win[c / 8][c % 8];
            }
        }
        let store = |win: &mut [V; 8], x: isize| {
            transpose8x8(win);
            for (r, &v) in win.iter().enumerate() {
                let a = dst + (r as isize * stride + x);
                let b = dst + ((r + 8) as isize * stride + x);
                store_rows::<BD, V>(v, a, b, p.max);
            }
        };
        let changed = filter::<V, MAX>(&mut px, p);
        if changed == 0 {
            return;
        }
        if MAX != 16 || changed < 6 {
            // The eight columns from 4 left of the edge hold every pixel changed.
            let mut c = [px[3], px[4], px[5], px[6], px[7], px[8], px[9], px[10]];
            return store(&mut c, -4);
        }
        for k in -7..7isize {
            let c = (k - x0) as usize;
            win[c / 8][c % 8] = px[(7 + k) as usize];
        }
        for (w, win) in win.iter_mut().enumerate() {
            store(win, x0 + 8 * w as isize);
        }
    }

    /// One guard over `n` rows of `w` pixels from `dst`, the stride, and where the first row
    /// starts in the guard: the rows' bytes for [`U8x16::load_rows`].
    #[inline(always)]
    fn rows_mut<'a, BD: BitDepth>(
        dst: Rav1dPictureDataComponentOffset<'a>,
        n: usize,
        w: usize,
    ) -> (
        DisjointMutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>,
        isize,
        usize,
    ) {
        let stride = dst.pixel_stride::<BD>();
        let last = (n as isize - 1) * stride;
        let lo = last.min(0);
        let rows = (dst + lo).slice_mut::<BD>(last.unsigned_abs() + w);
        (rows, stride, lo.unsigned_abs())
    }

    /// Where a group's block starts, as pixels before its first one: along a horizontal edge
    /// `reach` rows above, along a vertical one the first column the filter reads. Its rows and
    /// pixels per row.
    const fn block(h: bool, max: usize, n: usize) -> (usize, usize, usize) {
        match (h, max) {
            (false, _) => (reach(max), 2 * reach(max), n),
            (true, 4) => (2, n, 4),
            (true, 16) => (8, n, 16),
            _ => (4, n, 8),
        }
    }

    /// [`group_v`] at 8 bpc: `B::N` columns as bytes, the block's rows `stride` apart in `rows`
    /// from `start`.
    #[inline(always)]
    fn group_v8<B: Bytes, const MAX: usize>(
        rows: &mut [u8],
        start: usize,
        stride: isize,
        p: Params8<B>,
    ) {
        let r = reach(MAX);
        let mut px = [B::splat(0); 14];
        B::load_rows(&mut px[7 - r..7 + r], rows, start, stride);
        let w = filter8::<B, MAX>(&mut px, p);
        let start = |w: usize| start.wrapping_add_signed((r - w) as isize * stride);
        match w {
            0 => {}
            2 => B::store_rows(&px[5..9], rows, start(2), stride),
            3 => B::store_rows(&px[4..10], rows, start(3), stride),
            _ if MAX == 16 => B::store_rows(&px[1..13], rows, start(6), stride),
            _ => {}
        }
    }

    /// [`group_h`] at 8 bpc: `B::N` rows of 4, 8 or 16 bytes turned into columns.
    #[inline(always)]
    fn group_h8<B: Bytes, const MAX: usize>(
        rows: &mut [u8],
        start: usize,
        stride: isize,
        p: Params8<B>,
    ) {
        let mut px = [B::splat(0); 14];
        if MAX == 4 {
            // As the asm, only the four columns the filter reads.
            px[5..9].copy_from_slice(&B::load_cols4(rows, start, stride));
            if filter8::<B, MAX>(&mut px, p) != 0 {
                let c = [px[5], px[6], px[7], px[8]];
                B::store_cols4(&c, rows, start, stride);
            }
            return;
        }
        let (n, x0) = if MAX == 16 { (2, -8) } else { (1, -4) };
        let mut win = [[B::splat(0); 8]; 2];
        for (w, win) in win.iter_mut().enumerate().take(n) {
            *win = B::load_cols8(rows, start + 8 * w, stride);
        }
        for k in -7..7isize {
            let c = (k - x0) as usize;
            if c < 8 * n {
                px[(7 + k) as usize] = win[c / 8][c % 8];
            }
        }
        let changed = filter8::<B, MAX>(&mut px, p);
        // The columns from `x` on, as offsets into `rows`.
        let at = |x: isize| start + (x - x0) as usize;
        match changed {
            0 => {}
            2 => {
                B::store_cols4(&[px[5], px[6], px[7], px[8]], rows, at(-2), stride);
            }
            3 => {
                let c = [px[3], px[4], px[5], px[6], px[7], px[8], px[9], px[10]];
                B::store_cols8(&c, rows, at(-4), stride);
            }
            _ if MAX == 16 => {
                for k in -7..7isize {
                    let c = (k - x0) as usize;
                    win[c / 8][c % 8] = px[(7 + k) as usize];
                }
                for (w, win) in win.iter().enumerate() {
                    B::store_cols8(win, rows, start + 8 * w, stride);
                }
            }
            _ => {}
        }
    }

    /// A group at 8 bpc whose first `n` segments lie in the picture: all four in place under
    /// one guard, fewer through a copy of those segments' part of the block, so that no lane
    /// reads past the picture.
    #[inline(always)]
    fn group8<BD: BitDepth, B: Bytes, const MAX: usize>(
        dst: Rav1dPictureDataComponentOffset,
        hv: super::HV,
        n: usize,
        p: Params8<B>,
    ) {
        let h = matches!(hv, super::HV::H);
        let stride = dst.pixel_stride::<BD>();
        let (before, rows, w) = block(h, MAX, B::N);
        let first = if h {
            dst - before
        } else {
            dst - before as isize * stride
        };
        if n == B::N / 4 {
            let (mut b, stride, start) = rows_mut::<BD>(first, rows, w);
            return run8::<B, MAX>(h, b.as_bytes_mut(), start, stride, p);
        }
        // Along a vertical edge the segments are rows of at most 16 bytes, along a horizontal
        // one columns, of rows `B::N` apart.
        let (lines, len, s) = if h {
            (4 * n, w, 16)
        } else {
            (rows, 4 * n, B::N)
        };
        let mut copy = [0; 16 * 32];
        for r in 0..lines {
            let src = (first + r as isize * stride).slice::<BD>(len);
            copy[s * r..][..len].copy_from_slice(src.as_bytes());
        }
        run8::<B, MAX>(h, &mut copy, 0, s as isize, p);
        for r in 0..lines {
            let mut d = (first + r as isize * stride).slice_mut::<BD>(len);
            d.as_bytes_mut().copy_from_slice(&copy[s * r..][..len]);
        }
    }

    #[inline(always)]
    fn run8<B: Bytes, const MAX: usize>(
        h: bool,
        b: &mut [u8],
        start: usize,
        stride: isize,
        p: Params8<B>,
    ) {
        match h {
            true => group_h8::<B, MAX>(b, start, stride, p),
            false => group_v8::<B, MAX>(b, start, stride, p),
        }
    }

    /// [`Bytes::spread4`] of the first `n` levels from `lvl[start]`, `stride` apart, the others 0.
    /// Across rows one byte at a time: the bytes between belong to blocks other threads may be
    /// writing.
    #[inline(always)]
    fn levels8<B: Bytes>(
        lvl: &DisjointMut<AlignedVec2<u8>>,
        start: usize,
        stride: usize,
        n: usize,
    ) -> B {
        let mut l = [0; 8];
        if stride == 4 {
            // Along a horizontal edge the levels are one row's entries, whose other bytes are
            // written together with them: one borrow.
            let row = &*lvl.index(start..start + 4 * n - 3);
            for (j, l) in l.iter_mut().enumerate().take(n) {
                *l = row[4 * j];
            }
        } else {
            for (j, l) in l.iter_mut().enumerate().take(n) {
                *l = *lvl.index(start + j * stride);
            }
        }
        B::spread4(u64::from_le_bytes(l))
    }

    /// A segment's thresholds and filter length.
    #[derive(Clone, Copy)]
    struct Seg {
        e: u8,
        i: u8,
        h: u8,
        wd: usize,
    }

    /// A group: its first pixel, the vmask bits from its first segment on, and its levels.
    #[derive(Clone, Copy)]
    struct At<'a> {
        dst: Rav1dPictureDataComponentOffset<'a>,
        hv: super::HV,
        yuv: super::YUV,
        vmask: [u32; 3],
        lvl: WithOffset<&'a DisjointMut<AlignedVec2<u8>>>,
        /// From a segment's levels to the next one's, and to those across the edge.
        b4_stridea: usize,
        b4_strideb: usize,
    }

    impl At<'_> {
        /// The `vmask` bits of the segments filtered at all, of those longer than 4, and of 16.
        #[inline(always)]
        fn masks(&self) -> [u32; 3] {
            let [m0, m1, m2] = self.vmask;
            match self.yuv {
                super::YUV::Y => [m0 | m1 | m2, m1 | m2, m2],
                super::YUV::UV => [m0 | m1, m1, 0],
            }
        }

        /// Segment `j`'s thresholds and length, as [`loop_filter_sb128_rust`] reads them, if it
        /// is filtered.
        #[inline(always)]
        fn seg(&self, j: usize, lut: &Align16<Av1FilterLUT>) -> Option<Seg> {
            let [vm, gt4, wd16] = self.masks().map(|m| m >> j & 1 != 0);
            if !vm {
                return None;
            }
            let at = self.lvl + 4 * self.b4_stridea * j;
            let l = match *at.data.index(at.offset) {
                0 => {
                    let at = at - 4 * self.b4_strideb;
                    *at.data.index(at.offset)
                }
                l => l,
            };
            if l == 0 {
                return None;
            }
            let wd = match (self.yuv, gt4, wd16) {
                (_, _, true) => 16,
                (super::YUV::Y, true, _) => 8,
                (super::YUV::UV, true, _) => 6,
                _ => 4,
            };
            let (e, i, h) = (lut.0.e[l as usize], lut.0.i[l as usize], l >> 4);
            Some(Seg { e, i, h, wd })
        }
    }

    /// A lane type filtering a group of `N / 4` segments.
    trait Groups {
        const N: usize;
        fn group<BD: BitDepth>(at: At, lut: &Align16<Av1FilterLUT>, bd: BD);

        /// A group of which only the first `n` segments lie in the picture.
        #[inline(always)]
        fn partial<BD: BitDepth>(at: At, n: usize, lut: &Align16<Av1FilterLUT>, bd: BD) {
            scalar_group(at, n, lut, bd)
        }
    }

    /// The first `n` segments of a group through the scalar filter.
    #[inline(always)]
    fn scalar_group<BD: BitDepth>(at: At, n: usize, lut: &Align16<Av1FilterLUT>, bd: BD) {
        let stride = at.dst.pixel_stride::<BD>();
        let (stridea, strideb) = match at.hv {
            super::HV::H => (stride, 1),
            super::HV::V => (1, stride),
        };
        for j in 0..n {
            if let Some(Seg { e, i, h, wd }) = at.seg(j, lut) {
                let dst = at.dst + (4 * j) as isize * stridea;
                loop_filter(dst, e, i, h, stridea, strideb, wd as c_int, bd);
            }
        }
    }

    /// The thresholds as the asm computes them at 16 bpc, in sixteen-bit lanes from the levels
    /// and the sharpness, scaled to the bit depth (10 bits at most).
    #[inline(always)]
    fn group16<BD: BitDepth, V: Px>(at: At, lut: &Align16<Av1FilterLUT>, bd: BD) {
        let Some(([l, active, gt4, wd16], max)) = levels_masks::<U8x16>(&at, V::N / 4) else {
            return;
        };
        let s = bd.bitdepth() - 8;
        debug_assert!(s == 0 || s == 2);
        let scaled = |v: V| {
            if s == 0 {
                return v;
            }
            let v = v.add(v);
            v.add(v)
        };
        let l = V::from_bytes(l);
        let [sharp0, sharp1] = lut.0.sharp;
        let i = l
            .shr_var(sharp0 as u32)
            .min(V::splat(sharp1 as i16))
            .max(V::splat(1));
        let e = l.add(V::splat(2));
        let e = e.add(e).add(i);
        let p = Params {
            e: scaled(e),
            i: scaled(i),
            h: scaled(l.shr_s::<4>()),
            active: V::from_mask(active),
            gt4: V::from_mask(gt4),
            wd16: V::from_mask(wd16),
            flat: V::splat(1 << s),
            lo: V::splat(-128 << s),
            hi: V::splat((128 << s) - 1),
            max: V::splat(bd.bitdepth_max().as_::<i16>()),
        };
        let dst = at.dst;
        if BD::BITDEPTH != 8 {
            let h = matches!(at.hv, super::HV::H);
            let stride = dst.pixel_stride::<BD>();
            // As `block`, but along a vertical edge 8 (16) pixels from 4 (8) left of it, or the
            // four the narrow filter reads.
            let (before, rows, w) = match (h, max) {
                (false, _) => (reach(max), 2 * reach(max), V::N),
                (true, 4) => (2, V::N, 4),
                (true, 16) => (8, V::N, 16),
                _ => (4, V::N, 8),
            };
            let first = if h {
                dst - before
            } else {
                dst - before as isize * stride
            };
            let (mut g, stride, start) = rows_mut::<BD>(first, rows, w);
            let b = px16_mut(g.as_bytes_mut());
            return match (h, max) {
                (true, 4) => group_h16::<V, 4>(b, start, stride, p),
                (true, 6) => group_h16::<V, 6>(b, start, stride, p),
                (true, 8) => group_h16::<V, 8>(b, start, stride, p),
                (true, _) => group_h16::<V, 16>(b, start, stride, p),
                (false, 4) => group_v16::<V, 4>(b, start, stride, p),
                (false, 6) => group_v16::<V, 6>(b, start, stride, p),
                (false, 8) => group_v16::<V, 8>(b, start, stride, p),
                (false, _) => group_v16::<V, 16>(b, start, stride, p),
            };
        }
        match (at.hv, max) {
            (super::HV::H, 4) => group_h::<BD, V, 4>(dst, p),
            (super::HV::H, 6) => group_h::<BD, V, 6>(dst, p),
            (super::HV::H, 8) => group_h::<BD, V, 8>(dst, p),
            (super::HV::H, _) => group_h::<BD, V, 16>(dst, p),
            (super::HV::V, 4) => group_v::<BD, V, 4>(dst, p),
            (super::HV::V, 6) => group_v::<BD, V, 6>(dst, p),
            (super::HV::V, 8) => group_v::<BD, V, 8>(dst, p),
            (super::HV::V, _) => group_v::<BD, V, 16>(dst, p),
        }
    }

    impl Groups for I16x8 {
        const N: usize = 8;
        #[inline(always)]
        fn group<BD: BitDepth>(at: At, lut: &Align16<Av1FilterLUT>, bd: BD) {
            group16::<BD, Self>(at, lut, bd)
        }
    }

    #[cfg(target_arch = "x86_64")]
    impl Groups for I16x16 {
        const N: usize = 16;
        #[inline(always)]
        fn group<BD: BitDepth>(at: At, lut: &Align16<Av1FilterLUT>, bd: BD) {
            group16::<BD, Self>(at, lut, bd)
        }
    }

    /// The low `B::N / 4` bits of `m` as masks, each over four lanes.
    #[inline(always)]
    fn seg_masks<B: Bytes>(m: u32) -> B {
        let bits = B::spread4(0x8040_2010_0804_0201);
        B::splat(m as u8).and(bits).eq(bits)
    }

    /// 8 bpc only. The thresholds as the asm takes them: all four segments' levels at once, a
    /// level of 0 replaced by the one across the edge, and E and I looked up per lane. A segment
    /// whose level is 0 stays inactive, but the longest filter is the longest any segment's
    /// bits ask for.
    macro_rules! groups_bytes {
        ($t:ty) => {
            impl Groups for $t {
                const N: usize = <$t as Bytes>::N;
                #[inline(always)]
                fn group<BD: BitDepth>(at: At, lut: &Align16<Av1FilterLUT>, _: BD) {
                    group8_levels::<BD, $t>(at, <$t as Bytes>::N / 4, lut)
                }

                #[inline(always)]
                fn partial<BD: BitDepth>(at: At, n: usize, lut: &Align16<Av1FilterLUT>, _: BD) {
                    group8_levels::<BD, $t>(at, n, lut)
                }
            }
        };
    }

    groups_bytes!(U8x16);
    #[cfg(target_arch = "x86_64")]
    groups_bytes!(U8x32);

    /// The levels of the first `n` segments in byte lanes, a level of 0 replaced by the one
    /// across the edge, with [`Params8`]'s masks and the group's longest filter; none if no
    /// segment is filtered.
    #[inline(always)]
    fn levels_masks<B: Bytes>(at: &At, n: usize) -> Option<([B; 4], usize)> {
        let [vm, gt4, wd16] = at.masks().map(|m| m & ((1 << n) - 1));
        if vm == 0 {
            return None;
        }
        let (sa, sb) = (4 * at.b4_stridea, 4 * at.b4_strideb);
        let zero = B::splat(0);
        let filtered = seg_masks::<B>(vm);
        let l = levels8::<B>(at.lvl.data, at.lvl.offset, sa, n);
        // A level of 0 takes the one across the edge, read only if some segment needs it.
        let l = if l.le(zero).and(filtered).any() {
            B::select(l.le(zero), levels8(at.lvl.data, at.lvl.offset - sb, sa, n), l)
        } else {
            l
        };
        let max = match (wd16 != 0, gt4 != 0, at.yuv) {
            (true, _, _) => 16,
            (_, true, super::YUV::Y) => 8,
            (_, true, super::YUV::UV) => 6,
            _ => 4,
        };
        let active = filtered.andnot(l.le(zero));
        Some(([l, active, seg_masks(gt4), seg_masks(wd16)], max))
    }

    /// The thresholds of the first `n` segments as the asm takes them, in byte lanes, and the
    /// group's longest filter; none if no segment is filtered.
    #[inline(always)]
    fn thresholds8<B: Bytes>(
        at: &At,
        n: usize,
        lut: &Align16<Av1FilterLUT>,
    ) -> Option<(Params8<B>, usize)> {
        let ([l, active, gt4, wd16], max) = levels_masks::<B>(at, n)?;
        let (e, i) = l.limits(&lut.0.e, &lut.0.i, lut.0.sharp);
        let p = Params8 {
            e,
            i,
            h: l.shr_u::<4>(),
            active,
            gt4,
            wd16,
        };
        Some((p, max))
    }

    /// [`Groups::group`] at 8 bpc for the first `n` segments.
    #[inline(always)]
    fn group8_levels<BD: BitDepth, B: Bytes>(at: At, n: usize, lut: &Align16<Av1FilterLUT>) {
        debug_assert_eq!(BD::BITDEPTH, 8);
        let Some((p, max)) = thresholds8::<B>(&at, n, lut) else {
            return;
        };
        let dst = at.dst;
        match max {
            4 => group8::<BD, B, 4>(dst, at.hv, n, p),
            6 => group8::<BD, B, 6>(dst, at.hv, n, p),
            8 => group8::<BD, B, 8>(dst, at.hv, n, p),
            _ => group8::<BD, B, 16>(dst, at.hv, n, p),
        }
    }

    /// [`loop_filter_sb128_rust`] a group of `G::N / 4` segments at a time; a group running past
    /// the `wh` segments inside the picture takes [`Groups::partial`], so no lane reads beyond it.
    #[inline(always)]
    fn filter_sb<BD: BitDepth, G: Groups, const HV: usize, const YUV: usize>(
        dst: Rav1dPictureDataComponentOffset,
        vmask: &[u32; 3],
        lvl: WithOffset<&DisjointMut<AlignedVec2<u8>>>,
        b4_stride: usize,
        lut: &Align16<Av1FilterLUT>,
        wh: c_int,
        bd: BD,
    ) {
        let hv = super::HV::from_repr(HV).unwrap();
        let yuv = super::YUV::from_repr(YUV).unwrap();
        let stridea = match hv {
            super::HV::H => dst.pixel_stride::<BD>(),
            super::HV::V => 1,
        };
        let (b4_stridea, b4_strideb) = match hv {
            super::HV::H => (b4_stride, 1),
            super::HV::V => (1, b4_stride),
        };
        let mut at = At {
            dst,
            hv,
            yuv,
            vmask: *vmask,
            lvl,
            b4_stridea,
            b4_strideb,
        };
        let n = G::N / 4;
        let mut s0 = 0;
        while s0 < 32 && at.masks()[0] != 0 {
            if at.masks()[0] & ((1 << n) - 1) != 0 {
                let inside = (wh as usize).saturating_sub(s0);
                if inside >= n {
                    G::group(at, lut, bd);
                } else if inside > 0 && (at.masks()[0] & ((1 << n) - 1)) >> inside == 0 {
                    G::partial(at, inside, lut, bd);
                } else {
                    scalar_group(at, n, lut, bd);
                }
            }
            s0 += n;
            at.vmask = at.vmask.map(|m| m >> n);
            at.dst += (4 * n) as isize * stridea;
            at.lvl += 4 * n * b4_stridea;
        }
    }

    /// Bytes at 8 bpc, sixteen-bit lanes above.
    #[cfg_attr(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature(enable = "sse4.1")
    )]
    fn filter_sb_128<BD: BitDepth, const HV: usize, const YUV: usize>(
        dst: Rav1dPictureDataComponentOffset,
        vmask: &[u32; 3],
        lvl: WithOffset<&DisjointMut<AlignedVec2<u8>>>,
        b4_stride: usize,
        lut: &Align16<Av1FilterLUT>,
        wh: c_int,
        bd: BD,
    ) {
        if BD::BITDEPTH == 8 {
            return filter_sb::<BD, U8x16, { HV }, { YUV }>(
                dst, vmask, lvl, b4_stride, lut, wh, bd,
            );
        }
        filter_sb::<BD, I16x8, { HV }, { YUV }>(dst, vmask, lvl, b4_stride, lut, wh, bd)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    fn filter_sb_avx2<BD: BitDepth, const HV: usize, const YUV: usize>(
        dst: Rav1dPictureDataComponentOffset,
        vmask: &[u32; 3],
        lvl: WithOffset<&DisjointMut<AlignedVec2<u8>>>,
        b4_stride: usize,
        lut: &Align16<Av1FilterLUT>,
        wh: c_int,
        bd: BD,
    ) {
        if BD::BITDEPTH == 8 {
            return filter_sb::<BD, U8x32, { HV }, { YUV }>(
                dst, vmask, lvl, b4_stride, lut, wh, bd,
            );
        }
        filter_sb::<BD, I16x16, { HV }, { YUV }>(dst, vmask, lvl, b4_stride, lut, wh, bd)
    }

    /// # Safety
    ///
    /// Must be called by [`loopfilter_sb::Fn::call`] at 8 bpc, on x86 once the CPU flags hold
    /// SSE41, and with `WIDE` once they hold AVX2.
    #[deny(unsafe_op_in_unsafe_fn)]
    #[allow(unused_unsafe)]
    pub unsafe extern "C" fn loop_filter_sb_simd_erased<
        BD: BitDepth,
        const HV: usize,
        const YUV: usize,
        const WIDE: bool,
    >(
        _dst_ptr: *mut DynPixel,
        _stride: ptrdiff_t,
        vmask: &[u32; 3],
        _lvl_ptr: *const [u8; 4],
        b4_stride: isize,
        lut: &Align16<Av1FilterLUT>,
        wh: c_int,
        bitdepth_max: c_int,
        dst: FFISafeRav1dPictureDataComponentOffset,
        lvl: WithOffset<*const FFISafe<DisjointMut<AlignedVec2<u8>>>>,
    ) {
        // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `loopfilter_sb::Fn::call`.
        let dst = unsafe { FFISafe::from_with_offset(dst) };
        // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `loopfilter_sb::Fn::call`.
        let lvl = unsafe { FFISafe::from_with_offset(lvl) };
        let b4_stride = b4_stride as usize;
        let bd = BD::from_c(bitdepth_max);
        // Above 10 bits the flat filters' sums leave `i16`.
        if bitdepth_max > 1023 {
            return loop_filter_sb128_rust::<BD, { HV }, { YUV }>(
                dst, vmask, lvl, b4_stride, lut, wh, bd,
            );
        }
        // SAFETY: SSE4.1 is checked before this `fn` is chosen on x86, and AVX2 before `WIDE` is;
        // elsewhere it is safe.
        unsafe {
            #[cfg(target_arch = "x86_64")]
            if WIDE {
                return filter_sb_avx2::<BD, { HV }, { YUV }>(
                    dst, vmask, lvl, b4_stride, lut, wh, bd,
                );
            }
            filter_sb_128::<BD, { HV }, { YUV }>(dst, vmask, lvl, b4_stride, lut, wh, bd)
        }
    }
}

#[derive(Clone, Copy, FromRepr)]
enum HV {
    H,
    V,
}

#[derive(Clone, Copy, FromRepr)]
enum YUV {
    Y,
    UV,
}

fn loop_filter_sb128_rust<BD: BitDepth, const HV: usize, const YUV: usize>(
    mut dst: Rav1dPictureDataComponentOffset,
    vmask: &[u32; 3],
    mut lvl: WithOffset<&DisjointMut<AlignedVec2<u8>>>,
    b4_stride: usize,
    lut: &Align16<Av1FilterLUT>,
    _wh: c_int,
    bd: BD,
) {
    let hv = HV::from_repr(HV).unwrap();
    let yuv = YUV::from_repr(YUV).unwrap();

    let stride = dst.pixel_stride::<BD>();
    let (stridea, strideb) = match hv {
        HV::H => (stride, 1),
        HV::V => (1, stride),
    };
    let (b4_stridea, b4_strideb) = match hv {
        HV::H => (b4_stride, 1),
        HV::V => (1, b4_stride),
    };

    let vm = match yuv {
        YUV::Y => vmask[0] | vmask[1] | vmask[2],
        YUV::UV => vmask[0] | vmask[1],
    };
    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        'block: {
            if vm & xy == 0 {
                break 'block;
            }
            let l = *lvl.data.index(lvl.offset);
            let l = if l != 0 {
                l
            } else {
                let lvl = lvl - 4 * b4_strideb;
                *lvl.data.index(lvl.offset)
            };
            if l == 0 {
                break 'block;
            }
            let h = l >> 4;
            let e = lut.0.e[l as usize];
            let i = lut.0.i[l as usize];
            let idx = match yuv {
                YUV::Y => {
                    let idx = if vmask[2] & xy != 0 {
                        2
                    } else {
                        (vmask[1] & xy != 0) as c_int
                    };
                    4 << idx
                }
                YUV::UV => {
                    let idx = (vmask[1] & xy != 0) as c_int;
                    4 + 2 * idx
                }
            };
            loop_filter(dst, e, i, h, stridea, strideb, idx, bd);
        }
        xy <<= 1;
        dst += 4 * stridea;
        lvl += 4 * b4_stridea;
    }
}

/// # Safety
///
/// Must be called by [`loopfilter_sb::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn loop_filter_sb128_c_erased<BD: BitDepth, const HV: usize, const YUV: usize>(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    vmask: &[u32; 3],
    _lvl_ptr: *const [u8; 4],
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    wh: c_int,
    bitdepth_max: c_int,
    dst: FFISafeRav1dPictureDataComponentOffset,
    lvl: WithOffset<*const FFISafe<DisjointMut<AlignedVec2<u8>>>>,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `loopfilter_sb::Fn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `loopfilter_sb::Fn::call`.
    let lvl = unsafe { FFISafe::from_with_offset(lvl) };
    let b4_stride = b4_stride as usize;
    let bd = BD::from_c(bitdepth_max);
    loop_filter_sb128_rust::<BD, { HV }, { YUV }>(dst, vmask, lvl, b4_stride, lut, wh, bd)
}

impl Rav1dLoopFilterDSPContext {
    pub const fn default<BD: BitDepth>() -> Self {
        use HV::*;
        use YUV::*;
        Self {
            loop_filter_sb: LoopFilterYUVDSPContext {
                y: LoopFilterHVDSPContext {
                    h: loopfilter_sb::Fn::default::<BD, { H as _ }, { Y as _ }>(),
                    v: loopfilter_sb::Fn::default::<BD, { V as _ }, { Y as _ }>(),
                },
                uv: LoopFilterHVDSPContext {
                    h: loopfilter_sb::Fn::default::<BD, { H as _ }, { UV as _ }>(),
                    v: loopfilter_sb::Fn::default::<BD, { V as _ }, { UV as _ }>(),
                },
            },
        }
    }

    /// The Rust SIMD filters, at 10 bits on sixteen lanes once the CPU flags hold AVX2.
    #[inline(always)]
    const fn init<BD: BitDepth>(mut self, flags: CpuFlags) -> Self {
        use HV::*;
        use YUV::*;

        use self::rust_simd::loop_filter_sb_simd_erased as f;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if !flags.contains(CpuFlags::SSE41) {
            return self;
        }
        #[cfg(target_arch = "x86_64")]
        let avx2 = flags.contains(CpuFlags::AVX2);
        #[cfg(not(target_arch = "x86_64"))]
        let avx2 = false;
        let _ = flags;
        if !crate::simd128::native() {
            return self;
        }
        let lf = &mut self.loop_filter_sb;
        if avx2 {
            lf.y.h = loopfilter_sb::Fn::new(f::<BD, { H as _ }, { Y as _ }, true>);
            lf.y.v = loopfilter_sb::Fn::new(f::<BD, { V as _ }, { Y as _ }, true>);
            lf.uv.h = loopfilter_sb::Fn::new(f::<BD, { H as _ }, { UV as _ }, true>);
            lf.uv.v = loopfilter_sb::Fn::new(f::<BD, { V as _ }, { UV as _ }, true>);
        } else {
            lf.y.h = loopfilter_sb::Fn::new(f::<BD, { H as _ }, { Y as _ }, false>);
            lf.y.v = loopfilter_sb::Fn::new(f::<BD, { V as _ }, { Y as _ }, false>);
            lf.uv.h = loopfilter_sb::Fn::new(f::<BD, { H as _ }, { UV as _ }, false>);
            lf.uv.v = loopfilter_sb::Fn::new(f::<BD, { V as _ }, { UV as _ }, false>);
        }
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
    use crate::lf_mask::rav1d_calc_eih;

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
    const ROWS: usize = 160;
    const B4_STRIDE: usize = 40;

    /// The plane after `f` filters one edge: vertical at column 32 from row 8, or horizontal at
    /// row 32 from column 0, `wh` segments long.
    fn run<BD: BitDepth>(
        f: &loopfilter_sb::Fn,
        hv: HV,
        orig: &[BD::Pixel],
        mask: &[u32; 3],
        lvl: &DisjointMut<AlignedVec2<u8>>,
        lut: &Align16<Av1FilterLUT>,
        wh: usize,
        bitdepth_max: c_int,
    ) -> Vec<BD::Pixel> {
        let mut buf = AlignedVec64::<BD::Pixel>::new();
        buf.resize(orig.len(), 0.into());
        buf.as_mut_slice().copy_from_slice(orig);
        let comp = Rav1dPictureDataComponent::wrap_buf::<BD>(buf.as_mut_slice(), STRIDE);
        let offset = match hv {
            HV::H => 8 * STRIDE + 32,
            HV::V => 32 * STRIDE,
        };
        let dst = WithOffset {
            data: &comp,
            offset,
        };
        let lvl = WithOffset {
            data: lvl,
            offset: 8 * B4_STRIDE,
        };
        let dst_ptr = dst.as_mut_ptr::<BD>().cast();
        let lvl_ptr = lvl.data.as_mut_ptr().wrapping_add(lvl.offset).cast();
        // SAFETY: The arguments are what `loopfilter_sb::Fn::call` passes.
        unsafe {
            f.get()(
                dst_ptr,
                dst.stride(),
                mask,
                lvl_ptr,
                B4_STRIDE as isize,
                lut,
                wh as c_int,
                bitdepth_max,
                dst.into_ffi_safe(),
                lvl.into_ffi_safe(),
            )
        };
        drop(comp);
        buf.as_slice().to_vec()
    }

    /// Mostly smooth areas a step apart at the edges, so the flat filters fire, with noise; the
    /// values and steps scaled to the bit depth.
    fn plane<BD: BitDepth>(rng: &mut Rng, bitdepth_max: c_int) -> Vec<BD::Pixel> {
        let scale = (bitdepth_max + 1) / 256;
        let base = [rng.next() % 256, rng.next() % 256];
        let noise = [1, 2, 4, 16][rng.next() as usize % 4] * scale as u32;
        (0..STRIDE * ROWS)
            .map(|i| {
                let (y, x) = (i / STRIDE, i % STRIDE);
                let side = ((x / 4 + y / 4) & 1) as usize;
                let v = base[side] as i32 * scale + (rng.next() % noise) as i32 - noise as i32 / 2;
                v.clamp(0, bitdepth_max).as_::<BD::Pixel>()
            })
            .collect()
    }

    fn contexts<BD: BitDepth>() -> Vec<(&'static str, Rav1dLoopFilterDSPContext)> {
        rav1d_init_cpu();
        let flags = rav1d_get_cpu_flags();
        let ctx = |flags| Rav1dLoopFilterDSPContext::default::<BD>().init::<BD>(flags);
        #[allow(unused_mut)]
        let mut out = vec![("rust simd", ctx(flags))];
        #[cfg(target_arch = "x86_64")]
        if flags.contains(CpuFlags::AVX2) {
            out.push((
                "rust 128",
                ctx(flags.difference(CpuFlags::AVX2 | CpuFlags::AVX512ICL)),
            ));
        }
        out
    }

    #[test]
    fn simd_matches_scalar() {
        matches_scalar::<BitDepth8>(255, 3000);
    }

    /// 12 bits takes the scalar filter, which this checks is wired in.
    #[test]
    fn simd_matches_scalar_16bpc() {
        matches_scalar::<BitDepth16>(1023, 3000);
        matches_scalar::<BitDepth16>(4095, 300);
    }

    /// ns per call of a whole edge (32 segments) of each direction, plane and longest filter,
    /// on fresh planes, at 8 and 10 bits.
    #[test]
    #[ignore]
    fn time_lf() {
        time_lf_at::<BitDepth8>(255);
        time_lf_at::<BitDepth16>(1023);
    }

    fn time_lf_at<BD: BitDepth>(bitdepth_max: c_int) {
        use std::time::Instant;

        const N: usize = 32;
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let mut lut = Align16(Av1FilterLUT::default());
        rav1d_calc_eih(&mut lut.0, 0);
        let mut lvl = AlignedVec2::<u8>::new();
        lvl.resize(4 * B4_STRIDE * 48, 0);
        for l in lvl.as_mut_slice() {
            *l = 8 + (rng.next() % 40) as u8;
        }
        let lvl = DisjointMut::new(lvl);
        let origs: Vec<_> = (0..N).map(|_| plane::<BD>(&mut rng, bitdepth_max)).collect();
        let mut bufs: Vec<_> = (0..N)
            .map(|_| {
                let mut b = AlignedVec64::<BD::Pixel>::new();
                b.resize(STRIDE * ROWS, 0.into());
                b
            })
            .collect();
        let dsps = contexts::<BD>();
        for (hv, name) in [(HV::H, "h"), (HV::V, "v")] {
            for (yuv, wds) in [(YUV::Y, &[4, 8, 16][..]), (YUV::UV, &[4, 6][..])] {
                for &wd in wds {
                    let on = |b: bool| if b { u32::MAX } else { 0 };
                    let mask = [u32::MAX, on(wd > 4), on(wd == 16)];
                    let mut line = format!("{bitdepth_max} {name} {} wd {wd:2}:", yuv as usize);
                    // `TIME_LF` picks the lines holding it, timed `TIME_LF_N` (20) times.
                    if !std::env::var("TIME_LF").map_or(true, |f| line.contains(&f)) {
                        continue;
                    }
                    for (dsp_name, dsp) in &dsps {
                        let c = match yuv {
                            YUV::Y => &dsp.loop_filter_sb.y,
                            YUV::UV => &dsp.loop_filter_sb.uv,
                        };
                        let f = match hv {
                            HV::H => c.h,
                            HV::V => c.v,
                        };
                        let mut best = f64::MAX;
                        let n: usize = std::env::var("TIME_LF_N").map_or(20, |n| n.parse().unwrap());
                        for _ in 0..n {
                            for (b, o) in bufs.iter_mut().zip(&origs) {
                                b.as_mut_slice().copy_from_slice(o);
                            }
                            let start = Instant::now();
                            for b in &mut bufs {
                                let comp = Rav1dPictureDataComponent::wrap_buf::<BD>(
                                    b.as_mut_slice(),
                                    STRIDE,
                                );
                                let offset = match hv {
                                    HV::H => 8 * STRIDE + 32,
                                    HV::V => 32 * STRIDE,
                                };
                                let dst = WithOffset { data: &comp, offset };
                                let l = WithOffset { data: &lvl, offset: 8 * B4_STRIDE };
                                let dst_ptr = dst.as_mut_ptr::<BD>().cast();
                                let lvl_ptr = l.data.as_mut_ptr().wrapping_add(l.offset).cast();
                                // SAFETY: The arguments are what `loopfilter_sb::Fn::call` passes.
                                unsafe {
                                    f.get()(
                                        dst_ptr,
                                        dst.stride(),
                                        &mask,
                                        lvl_ptr,
                                        B4_STRIDE as isize,
                                        &lut,
                                        32,
                                        bitdepth_max,
                                        dst.into_ffi_safe(),
                                        l.into_ffi_safe(),
                                    )
                                };
                            }
                            best = best.min(start.elapsed().as_nanos() as f64 / N as f64);
                        }
                        line += &format!("  {dsp_name} {best:7.1} ns");
                    }
                    println!("{line}");
                }
            }
        }
    }

    fn matches_scalar<BD: BitDepth>(bitdepth_max: c_int, rounds: usize) {
        let scalar = Rav1dLoopFilterDSPContext::default::<BD>();
        let simds = contexts::<BD>();
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for round in 0..rounds {
            let mut lut = Align16(Av1FilterLUT::default());
            rav1d_calc_eih(&mut lut.0, (rng.next() % 8) as u8);
            let mut lvl = AlignedVec2::<u8>::new();
            lvl.resize(4 * B4_STRIDE * 48, 0);
            for l in lvl.as_mut_slice() {
                *l = if rng.next() % 5 == 0 {
                    0
                } else {
                    (rng.next() % 64) as u8
                };
            }
            let lvl = DisjointMut::new(lvl);
            let wh = 1 + rng.next() as usize % 32;
            let valid = if wh == 32 { u32::MAX } else { (1 << wh) - 1 };
            let mask = [rng.next() & valid, rng.next() & valid, rng.next() & valid];
            let orig = plane::<BD>(&mut rng, bitdepth_max);
            for (hv, name) in [(HV::H, "h"), (HV::V, "v")] {
                for yuv in [YUV::Y, YUV::UV] {
                    let pick = |c: &Rav1dLoopFilterDSPContext| {
                        let c = match yuv {
                            YUV::Y => &c.loop_filter_sb.y,
                            YUV::UV => &c.loop_filter_sb.uv,
                        };
                        match hv {
                            HV::H => c.h,
                            HV::V => c.v,
                        }
                    };
                    let want = run::<BD>(
                        &pick(&scalar),
                        hv,
                        &orig,
                        &mask,
                        &lvl,
                        &lut,
                        wh,
                        bitdepth_max,
                    );
                    for (simd_name, simd) in &simds {
                        let got =
                            run::<BD>(&pick(simd), hv, &orig, &mask, &lvl, &lut, wh, bitdepth_max);
                        let first = got.iter().zip(&want).position(|(a, b)| a != b);
                        assert!(
                            first.is_none(),
                            "{bitdepth_max} round {round} {name} {} {simd_name}: first difference at row {}, column {}",
                            yuv as usize,
                            first.unwrap() / STRIDE,
                            first.unwrap() % STRIDE,
                        );
                    }
                }
            }
        }
    }
}
