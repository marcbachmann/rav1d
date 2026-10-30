#![deny(unsafe_op_in_unsafe_fn)]

use std::cmp;
use std::ffi::c_int;

use libc::ptrdiff_t;
use strum::FromRepr;

use crate::align::{Align16, AlignedVec2};
use crate::cpu::CpuFlags;
use crate::disjoint_mut::DisjointMut;
use crate::ffi_safe::FFISafe;
#[cfg(all(
    feature = "asm",
    not(any(target_arch = "riscv64", target_arch = "riscv32"))
))]
use crate::include::common::bitdepth::bd_fn;
use crate::include::common::bitdepth::{AsPrimitive, BitDepth, DynPixel};
use crate::include::common::intops::iclip;
use crate::include::dav1d::picture::{
    FFISafeRav1dPictureDataComponentOffset, Rav1dPictureDataComponentOffset,
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
/// dav1d's asm (on AVX2 in sixteen-bit lanes), and eight at a time at 10 bits: four (two) 4-line
/// segments, each with its own levels and filter length. Every lane takes the narrow filter and
/// the flat filters the group's longest length needs, and masks per lane pick its result, as the
/// C's branches would. Along a vertical edge the lines are rows, turned into lanes by transposes;
/// along a horizontal one they are columns, loaded as they lie.
mod rust_simd {
    use zerocopy::{AsBytes, FromBytes};

    use super::*;
    #[cfg(target_arch = "x86_64")]
    use crate::simd128::I16x16;
    use crate::simd128::{transpose16x8, transpose8x16, transpose8x8, I16x8, Lanes, U8x16};

    /// A lane type with its pixel loads and stores.
    pub trait Px: Lanes {
        /// Lines per vector.
        const N: usize;
        /// Bytes `b[..N]` zero-extended.
        fn load_px(b: &[u8]) -> Self;
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
    }

    impl Px for I16x8 {
        const N: usize = 8;
        #[inline(always)]
        fn load_px(b: &[u8]) -> Self {
            I16x8::from_u8x8(b)
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
    }

    #[cfg(target_arch = "x86_64")]
    impl Px for I16x16 {
        const N: usize = 16;
        #[inline(always)]
        fn load_px(b: &[u8]) -> Self {
            I16x16::from_u8x16(b)
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
        /// Filter length above 4 (6, 8 or 16), above 6, and 16.
        gt4: V,
        gt6: V,
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

    /// Generated from `loop_filter`'s flat16 branch: one running sum, a term in and out per output;
    /// `out` holds the sums before [`rshr`]`::<4>`.
    #[inline(always)]
    fn flat16<V: Lanes>(px: &[V; 14], out: &mut [V; 14]) {
        let mut s = px[0]
            .add(px[0])
            .add(px[0])
            .add(px[0])
            .add(px[0])
            .add(px[0])
            .add(px[0])
            .add(px[1])
            .add(px[1])
            .add(px[2])
            .add(px[2])
            .add(px[3])
            .add(px[4])
            .add(px[5])
            .add(px[6])
            .add(px[7]);
        out[1] = s;
        s = s.sub(px[0]).sub(px[0]).add(px[3]).add(px[8]);
        out[2] = s;
        s = s.sub(px[0]).sub(px[1]).add(px[4]).add(px[9]);
        out[3] = s;
        s = s.sub(px[0]).sub(px[2]).add(px[5]).add(px[10]);
        out[4] = s;
        s = s.sub(px[0]).sub(px[3]).add(px[6]).add(px[11]);
        out[5] = s;
        s = s.sub(px[0]).sub(px[4]).add(px[7]).add(px[12]);
        out[6] = s;
        s = s.sub(px[0]).sub(px[5]).add(px[8]).add(px[13]);
        out[7] = s;
        s = s.sub(px[1]).sub(px[6]).add(px[9]).add(px[13]);
        out[8] = s;
        s = s.sub(px[2]).sub(px[7]).add(px[10]).add(px[13]);
        out[9] = s;
        s = s.sub(px[3]).sub(px[8]).add(px[11]).add(px[13]);
        out[10] = s;
        s = s.sub(px[4]).sub(px[9]).add(px[12]).add(px[13]);
        out[11] = s;
        s = s.sub(px[5]).sub(px[10]).add(px[13]).add(px[13]);
        out[12] = s;
    }

    /// Generated from `loop_filter`'s flat8 branch: one running sum, a term in and out per output;
    /// `out` holds the sums before [`rshr`]`::<3>`.
    #[inline(always)]
    fn flat8<V: Lanes>(px: &[V; 14], out: &mut [V; 14]) {
        let mut s = px[3]
            .add(px[3])
            .add(px[3])
            .add(px[4])
            .add(px[4])
            .add(px[5])
            .add(px[6])
            .add(px[7]);
        out[4] = s;
        s = s.sub(px[3]).sub(px[4]).add(px[5]).add(px[8]);
        out[5] = s;
        s = s.sub(px[3]).sub(px[5]).add(px[6]).add(px[9]);
        out[6] = s;
        s = s.sub(px[3]).sub(px[6]).add(px[7]).add(px[10]);
        out[7] = s;
        s = s.sub(px[4]).sub(px[7]).add(px[8]).add(px[10]);
        out[8] = s;
        s = s.sub(px[5]).sub(px[8]).add(px[9]).add(px[10]);
        out[9] = s;
    }

    /// Generated from `loop_filter`'s flat6 branch: one running sum, a term in and out per output;
    /// `out` holds the sums before [`rshr`]`::<3>`.
    #[inline(always)]
    fn flat6<V: Lanes>(px: &[V; 14], out: &mut [V; 14]) {
        let mut s = px[4]
            .add(px[4])
            .add(px[4])
            .add(px[5])
            .add(px[5])
            .add(px[6])
            .add(px[6])
            .add(px[7]);
        out[5] = s;
        s = s.sub(px[4]).sub(px[4]).add(px[7]).add(px[8]);
        out[6] = s;
        s = s.sub(px[4]).sub(px[5]).add(px[8]).add(px[9]);
        out[7] = s;
        s = s.sub(px[5]).sub(px[6]).add(px[9]).add(px[9]);
        out[8] = s;
    }

    /// `px[7 + k]` is the pixel at offset `k` across the edge, `-7..=6`; `MAX` is the group's
    /// longest filter, and lanes of a shorter one ignore what it adds.
    #[inline(always)]
    fn filter<V: Lanes, const MAX: usize>(px: &mut [V; 14], p: Params<V>) {
        let orig = *px;
        let [p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6] = orig;
        let (zero, ones, one) = (V::splat(0), V::splat(-1), V::splat(1));
        let mut fm = le(absdiff(p1, p0), p.i)
            .and(le(absdiff(q1, q0), p.i))
            .and(le(
                absdiff(p0, q0)
                    .add(absdiff(p0, q0))
                    .add(absdiff(p1, q1).shr_s::<1>()),
                p.e,
            ))
            .and(p.active);
        if MAX > 4 {
            let c = le(absdiff(p2, p1), p.i).and(le(absdiff(q2, q1), p.i));
            fm = fm.and(V::select(p.gt4, c, ones));
        }
        if MAX > 6 {
            let c = le(absdiff(p3, p2), p.i).and(le(absdiff(q3, q2), p.i));
            fm = fm.and(V::select(p.gt6, c, ones));
        }
        // As the asm, each step only where some lane takes it.
        if !fm.any() {
            return;
        }

        let hev = le(absdiff(p1, p0), p.h)
            .and(le(absdiff(q1, q0), p.h))
            .eq(zero);
        let (lo, hi) = (p.lo, p.hi);
        let f = p1.sub(q1).max(lo).min(hi).and(hev);
        let d = q0.sub(p0);
        let f = d.add(d).add(d).add(f).max(lo).min(hi);
        let f1 = f.add(V::splat(4)).min(hi).shr_s::<3>();
        let f2 = f.add(V::splat(3)).min(hi).shr_s::<3>();
        let f3 = f1.add(one).shr_s::<1>().and(hev.eq(zero));
        px[5] = V::select(fm, p1.add(f3), p1);
        px[6] = V::select(fm, p0.add(f2), p0);
        px[7] = V::select(fm, q0.sub(f1), q0);
        px[8] = V::select(fm, q1.sub(f3), q1);

        if MAX < 6 {
            return;
        }
        let mut flat_in = le(absdiff(p2, p0), p.flat)
            .and(le(absdiff(p1, p0), p.flat))
            .and(le(absdiff(q1, q0), p.flat))
            .and(le(absdiff(q2, q0), p.flat));
        if MAX > 6 {
            let c = le(absdiff(p3, p0), p.flat).and(le(absdiff(q3, q0), p.flat));
            flat_in = flat_in.and(V::select(p.gt6, c, ones));
        }
        let flat_in = flat_in.and(fm);
        let mut out = orig;
        if MAX == 6 {
            let m = flat_in.and(p.gt4);
            if !m.any() {
                return;
            }
            flat6(&orig, &mut out);
            for k in 5..=8 {
                px[k] = V::select(m, rshr::<V, 3>(out[k]), px[k]);
            }
            return;
        }
        let m = flat_in.and(p.gt6);
        if !m.any() {
            return;
        }
        flat8(&orig, &mut out);
        for k in 4..=9 {
            px[k] = V::select(m, rshr::<V, 3>(out[k]), px[k]);
        }
        if MAX == 16 && flat_in.and(p.wd16).any() {
            let flat_out = le(absdiff(p6, p0), p.flat)
                .and(le(absdiff(p5, p0), p.flat))
                .and(le(absdiff(p4, p0), p.flat))
                .and(le(absdiff(q4, q0), p.flat))
                .and(le(absdiff(q5, q0), p.flat))
                .and(le(absdiff(q6, q0), p.flat));
            let m = flat_in.and(flat_out).and(p.wd16);
            flat16(&orig, &mut out);
            for k in 1..=12 {
                px[k] = V::select(m, rshr::<V, 4>(out[k]), px[k]);
            }
        }
    }

    /// [`Params`] at 8 bpc, on byte lanes.
    #[derive(Clone, Copy)]
    struct Params8 {
        e: U8x16,
        i: U8x16,
        h: U8x16,
        active: U8x16,
        /// Filter length above 4, and 16. A group whose longest filter is 8 or 16 is luma, where
        /// no length is 6, so above 4 is also above 6.
        gt4: U8x16,
        wd16: U8x16,
    }

    /// [`filter`] at 8 bpc as dav1d's asm: masks and the narrow filter on bytes, the latter
    /// signed after `^ 0x80` with saturating adds, which clip as `iclip_diff` and the pixel clip
    /// do; only the flat filters' sums widen to 16 bits.
    #[inline(always)]
    fn filter8<const MAX: usize>(px: &mut [U8x16; 14], p: Params8) {
        let orig = *px;
        let [p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6] = orig;
        let one = U8x16::splat(1);
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
            return;
        }

        // A sum of the same sign saturated step by step clips as the sum would: once past a
        // bound, every further step stays past it.
        let nhev = inner.le(p.h);
        let x80 = U8x16::splat(0x80);
        let (sp1, sp0, sq0, sq1) = (p1.xor(x80), p0.xor(x80), q0.xor(x80), q1.xor(x80));
        let f = sp1.sub_sat_i(sq1).andnot(nhev);
        let d = sq0.sub_sat_i(sp0);
        let f = f.add_sat_i(d).add_sat_i(d).add_sat_i(d);
        let f1 = f.add_sat_i(U8x16::splat(4)).shr_i::<3>();
        let f2 = f.add_sat_i(U8x16::splat(3)).shr_i::<3>();
        let f3 = f1.add(one).shr_i::<1>().and(nhev);
        px[5] = U8x16::select(fm, sp1.add_sat_i(f3).xor(x80), p1);
        px[6] = U8x16::select(fm, sp0.add_sat_i(f2).xor(x80), p0);
        px[7] = U8x16::select(fm, sq0.sub_sat_i(f1).xor(x80), q0);
        px[8] = U8x16::select(fm, sq1.sub_sat_i(f3).xor(x80), q1);

        if MAX < 6 {
            return;
        }
        let mut flat = p2.abd(p0).max(dp1).max(dq1).max(q2.abd(q0));
        if MAX > 6 {
            flat = flat.max(p3.abd(p0)).max(q3.abd(q0));
        }
        let mut m8 = flat.le(one).and(fm).and(p.gt4);
        if !m8.any() {
            return;
        }
        let (mut lo, mut hi) = ([I16x8::splat(0); 14], [I16x8::splat(0); 14]);
        for k in 0..14 {
            [lo[k], hi[k]] = orig[k].widen();
        }
        let (mut out_lo, mut out_hi) = (lo, hi);
        if MAX == 6 {
            flat6(&lo, &mut out_lo);
            flat6(&hi, &mut out_hi);
            for k in 5..=8 {
                let v = U8x16::narrow_rshr::<3>(out_lo[k], out_hi[k]);
                px[k] = U8x16::select(m8, v, px[k]);
            }
            return;
        }
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
                flat16(&lo, &mut out_lo);
                flat16(&hi, &mut out_hi);
                for k in 1..=12 {
                    let v = U8x16::narrow_rshr::<4>(out_lo[k], out_hi[k]);
                    px[k] = U8x16::select(m16, v, px[k]);
                }
            }
            if !m8.any() {
                return;
            }
        }
        flat8(&lo, &mut out_lo);
        flat8(&hi, &mut out_hi);
        for k in 4..=9 {
            let v = U8x16::narrow_rshr::<3>(out_lo[k], out_hi[k]);
            px[k] = U8x16::select(m8, v, px[k]);
        }
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
        filter::<V, MAX>(&mut px, p);
        let w = if MAX == 16 { 6 } else { r.min(3) };
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
        filter::<V, MAX>(&mut px, p);
        for k in -7..7isize {
            let c = (k - x0) as usize;
            if c < 8 * n {
                win[c / 8][c % 8] = px[(7 + k) as usize];
            }
        }
        for (w, win) in win.iter_mut().enumerate().take(n) {
            let x = x0 + 8 * w as isize;
            transpose8x8(win);
            for (r, &v) in win.iter().enumerate() {
                let a = dst + (r as isize * stride + x);
                let b = dst + ((r + 8) as isize * stride + x);
                store_rows::<BD, V>(v, a, b, p.max);
            }
        }
    }

    /// [`group_v`] at 8 bpc: sixteen columns as bytes.
    #[inline(always)]
    fn group_v8<BD: BitDepth, const MAX: usize>(dst: Rav1dPictureDataComponentOffset, p: Params8) {
        let stride = dst.pixel_stride::<BD>();
        let r = reach(MAX) as isize;
        let mut px = [U8x16::splat(0); 14];
        for k in -r..r {
            let row = (dst + k * stride).slice::<BD>(16);
            px[(7 + k) as usize] = U8x16::load(row.as_bytes());
        }
        filter8::<MAX>(&mut px, p);
        let w = if MAX == 16 { 6 } else { r.min(3) };
        for k in -w..w {
            let mut row = (dst + k * stride).slice_mut::<BD>(16);
            px[(7 + k) as usize].store(row.as_bytes_mut());
        }
    }

    /// [`group_h`] at 8 bpc: sixteen rows of 8 (or 16) bytes turned into columns.
    #[inline(always)]
    fn group_h8<BD: BitDepth, const MAX: usize>(dst: Rav1dPictureDataComponentOffset, p: Params8) {
        let stride = dst.pixel_stride::<BD>();
        let (n, x0) = if MAX == 16 { (2, -8) } else { (1, -4) };
        let mut win = [[U8x16::splat(0); 8]; 2];
        for (w, win) in win.iter_mut().enumerate().take(n) {
            let x = x0 + 8 * w as isize;
            let mut rows = [U8x16::splat(0); 16];
            for (r, v) in rows.iter_mut().enumerate() {
                let row = (dst + (r as isize * stride + x)).slice::<BD>(8);
                *v = U8x16::load_lo(row.as_bytes());
            }
            *win = transpose16x8(&rows);
        }
        let mut px = [U8x16::splat(0); 14];
        for k in -7..7isize {
            let c = (k - x0) as usize;
            if c < 8 * n {
                px[(7 + k) as usize] = win[c / 8][c % 8];
            }
        }
        filter8::<MAX>(&mut px, p);
        for k in -7..7isize {
            let c = (k - x0) as usize;
            if c < 8 * n {
                win[c / 8][c % 8] = px[(7 + k) as usize];
            }
        }
        for (w, win) in win.iter().enumerate().take(n) {
            let x = x0 + 8 * w as isize;
            for (i, v) in transpose8x16(win).into_iter().enumerate() {
                let mut a = (dst + ((2 * i) as isize * stride + x)).slice_mut::<BD>(8);
                let mut b = (dst + ((2 * i + 1) as isize * stride + x)).slice_mut::<BD>(8);
                v.store_halves(a.as_bytes_mut(), b.as_bytes_mut());
            }
        }
    }

    /// A segment's thresholds and filter length.
    #[derive(Clone, Copy)]
    struct Seg {
        e: u8,
        i: u8,
        h: u8,
        wd: usize,
    }

    /// A lane type filtering a group of `N / 4` segments, `max` the longest filter among them.
    trait Groups {
        const N: usize;
        fn group<BD: BitDepth>(
            dst: Rav1dPictureDataComponentOffset,
            hv: super::HV,
            segs: &[Option<Seg>; 4],
            max: usize,
            bd: BD,
        );
    }

    #[inline(always)]
    fn group16<BD: BitDepth, V: Px>(
        dst: Rav1dPictureDataComponentOffset,
        hv: super::HV,
        segs: &[Option<Seg>; 4],
        max: usize,
        bd: BD,
    ) {
        let s = bd.bitdepth() - 8;
        let mut lanes = [[0i16; 16]; 7];
        for (j, seg) in segs.iter().enumerate().take(V::N / 4) {
            let Some(Seg { e, i, h, wd }) = *seg else {
                continue;
            };
            let flags = [
                (e as i16) << s,
                (i as i16) << s,
                (h as i16) << s,
                -1,
                -((wd > 4) as i16),
                -((wd > 6) as i16),
                -((wd == 16) as i16),
            ];
            for (lane, flag) in lanes.iter_mut().zip(flags) {
                lane[4 * j..4 * j + 4].fill(flag);
            }
        }
        let [e, i, h, active, gt4, gt6, wd16] = lanes.map(|l| V::load_i16(&l));
        let p = Params {
            e,
            i,
            h,
            active,
            gt4,
            gt6,
            wd16,
            flat: V::splat(1 << s),
            lo: V::splat(-128 << s),
            hi: V::splat((128 << s) - 1),
            max: V::splat(bd.bitdepth_max().as_::<i16>()),
        };
        match (hv, max) {
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
        fn group<BD: BitDepth>(
            dst: Rav1dPictureDataComponentOffset,
            hv: super::HV,
            segs: &[Option<Seg>; 4],
            max: usize,
            bd: BD,
        ) {
            group16::<BD, Self>(dst, hv, segs, max, bd)
        }
    }

    #[cfg(target_arch = "x86_64")]
    impl Groups for I16x16 {
        const N: usize = 16;
        #[inline(always)]
        fn group<BD: BitDepth>(
            dst: Rav1dPictureDataComponentOffset,
            hv: super::HV,
            segs: &[Option<Seg>; 4],
            max: usize,
            bd: BD,
        ) {
            group16::<BD, Self>(dst, hv, segs, max, bd)
        }
    }

    /// 8 bpc only.
    impl Groups for U8x16 {
        const N: usize = 16;
        #[inline(always)]
        fn group<BD: BitDepth>(
            dst: Rav1dPictureDataComponentOffset,
            hv: super::HV,
            segs: &[Option<Seg>; 4],
            max: usize,
            _: BD,
        ) {
            debug_assert_eq!(BD::BITDEPTH, 8);
            let mut lanes = [[0u8; 16]; 6];
            for (j, seg) in segs.iter().enumerate() {
                let Some(Seg { e, i, h, wd }) = *seg else {
                    continue;
                };
                let flags = [e, i, h, !0, 0u8.wrapping_sub((wd > 4) as u8), 0u8.wrapping_sub((wd == 16) as u8)];
                for (lane, flag) in lanes.iter_mut().zip(flags) {
                    lane[4 * j..4 * j + 4].fill(flag);
                }
            }
            let [e, i, h, active, gt4, wd16] = lanes.map(|l| U8x16::load(&l));
            let p = Params8 {
                e,
                i,
                h,
                active,
                gt4,
                wd16,
            };
            match (hv, max) {
                (super::HV::H, 4) => group_h8::<BD, 4>(dst, p),
                (super::HV::H, 6) => group_h8::<BD, 6>(dst, p),
                (super::HV::H, 8) => group_h8::<BD, 8>(dst, p),
                (super::HV::H, _) => group_h8::<BD, 16>(dst, p),
                (super::HV::V, 4) => group_v8::<BD, 4>(dst, p),
                (super::HV::V, 6) => group_v8::<BD, 6>(dst, p),
                (super::HV::V, 8) => group_v8::<BD, 8>(dst, p),
                (super::HV::V, _) => group_v8::<BD, 16>(dst, p),
            }
        }
    }

    /// [`loop_filter_sb128_rust`] a group of `G::N / 4` segments at a time; a group running past
    /// the `wh` segments inside the picture takes the scalar filter, so no lane reads beyond it.
    #[inline(always)]
    fn filter_sb<BD: BitDepth, G: Groups, const HV: usize, const YUV: usize>(
        mut dst: Rav1dPictureDataComponentOffset,
        vmask: &[u32; 3],
        mut lvl: WithOffset<&DisjointMut<AlignedVec2<u8>>>,
        b4_stride: usize,
        lut: &Align16<Av1FilterLUT>,
        wh: c_int,
        bd: BD,
    ) {
        let hv = super::HV::from_repr(HV).unwrap();
        let yuv = super::YUV::from_repr(YUV).unwrap();
        let stride = dst.pixel_stride::<BD>();
        let (stridea, strideb) = match hv {
            super::HV::H => (stride, 1),
            super::HV::V => (1, stride),
        };
        let (b4_stridea, b4_strideb) = match hv {
            super::HV::H => (b4_stride, 1),
            super::HV::V => (1, b4_stride),
        };
        let vm = match yuv {
            super::YUV::Y => vmask[0] | vmask[1] | vmask[2],
            super::YUV::UV => vmask[0] | vmask[1],
        };
        let n = G::N / 4;
        let mut s0 = 0;
        while s0 < 32 && vm >> s0 != 0 {
            let mut segs = [None; 4];
            let mut max = 0;
            for (j, seg) in segs.iter_mut().enumerate().take(n) {
                let xy = 1u32 << (s0 + j);
                if vm & xy == 0 {
                    continue;
                }
                let at = lvl + 4 * b4_stridea * j;
                let l = match *at.data.index(at.offset) {
                    0 => {
                        let at = at - 4 * b4_strideb;
                        *at.data.index(at.offset)
                    }
                    l => l,
                };
                if l == 0 {
                    continue;
                }
                let idx = match yuv {
                    super::YUV::Y => {
                        if vmask[2] & xy != 0 {
                            2
                        } else {
                            (vmask[1] & xy != 0) as usize
                        }
                    }
                    super::YUV::UV => (vmask[1] & xy != 0) as usize,
                };
                let wd = match yuv {
                    super::YUV::Y => 4 << idx,
                    super::YUV::UV => 4 + 2 * idx,
                };
                let (e, i, h) = (lut.0.e[l as usize], lut.0.i[l as usize], l >> 4);
                *seg = Some(Seg { e, i, h, wd });
                max = max.max(wd);
            }
            if max != 0 && s0 + n > wh as usize {
                for (j, seg) in segs.into_iter().enumerate() {
                    if let Some(Seg { e, i, h, wd }) = seg {
                        let dst = dst + (4 * j) as isize * stridea;
                        loop_filter(dst, e, i, h, stridea, strideb, wd as c_int, bd);
                    }
                }
            } else if max != 0 {
                G::group(dst, hv, &segs, max, bd);
            }
            s0 += n;
            dst += (4 * n) as isize * stridea;
            lvl += 4 * n * b4_stridea;
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
            return filter_sb::<BD, U8x16, { HV }, { YUV }>(dst, vmask, lvl, b4_stride, lut, wh, bd);
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

    #[cfg(all(feature = "asm", any(target_arch = "x86", target_arch = "x86_64")))]
    #[inline(always)]
    const fn init_x86<BD: BitDepth>(mut self, flags: CpuFlags) -> Self {
        if !flags.contains(CpuFlags::SSSE3) {
            return self;
        }

        self.loop_filter_sb.y.h = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_h_sb_y, ssse3);
        self.loop_filter_sb.y.v = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_v_sb_y, ssse3);
        self.loop_filter_sb.uv.h = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_h_sb_uv, ssse3);
        self.loop_filter_sb.uv.v = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_v_sb_uv, ssse3);

        #[cfg(target_arch = "x86_64")]
        {
            if !flags.contains(CpuFlags::AVX2) {
                return self;
            }

            self.loop_filter_sb.y.h = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_h_sb_y, avx2);
            self.loop_filter_sb.y.v = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_v_sb_y, avx2);
            self.loop_filter_sb.uv.h = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_h_sb_uv, avx2);
            self.loop_filter_sb.uv.v = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_v_sb_uv, avx2);

            if !flags.contains(CpuFlags::AVX512ICL) {
                return self;
            }

            self.loop_filter_sb.y.v = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_v_sb_y, avx512icl);
            self.loop_filter_sb.uv.v = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_v_sb_uv, avx512icl);

            if !flags.contains(CpuFlags::SLOW_GATHER) {
                self.loop_filter_sb.y.h = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_h_sb_y, avx512icl);
                self.loop_filter_sb.uv.h =
                    bd_fn!(loopfilter_sb::decl_fn, BD, lpf_h_sb_uv, avx512icl);
            }
        }

        self
    }

    #[cfg(all(feature = "asm", any(target_arch = "arm", target_arch = "aarch64")))]
    #[inline(always)]
    const fn init_arm<BD: BitDepth>(mut self, flags: CpuFlags) -> Self {
        if !flags.contains(CpuFlags::NEON) {
            return self;
        }

        self.loop_filter_sb.y.h = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_h_sb_y, neon);
        self.loop_filter_sb.y.v = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_v_sb_y, neon);
        self.loop_filter_sb.uv.h = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_h_sb_uv, neon);
        self.loop_filter_sb.uv.v = bd_fn!(loopfilter_sb::decl_fn, BD, lpf_v_sb_uv, neon);

        self
    }

    /// The Rust SIMD filters at 8 bpc, on sixteen lanes once the CPU flags hold AVX2.
    #[inline(always)]
    const fn init_rust_simd<BD: BitDepth>(mut self, flags: CpuFlags) -> Self {
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

    /// The Rust SIMD filters stand in where there is no asm, or over it with
    /// `--cfg rav1d_rust_simd="loopfilter"`.
    #[inline(always)]
    const fn init<BD: BitDepth>(self, flags: CpuFlags) -> Self {
        #[cfg(all(feature = "asm", not(rav1d_rust_simd = "loopfilter")))]
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
        let ctx = |flags| Rav1dLoopFilterDSPContext::default::<BD>().init_rust_simd::<BD>(flags);
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
                    let want = run::<BD>(&pick(&scalar), hv, &orig, &mask, &lvl, &lut, wh, bitdepth_max);
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
