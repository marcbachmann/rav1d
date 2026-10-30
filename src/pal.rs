#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::c_int;
use std::slice;

use crate::cpu::CpuFlags;
use crate::wrap_fn_ptr::wrap_fn_ptr;

wrap_fn_ptr!(pub unsafe extern "C" fn pal_idx_finish(
    dst: *mut u8,
    src: *const u8,
    bw: c_int,
    bh: c_int,
    w: c_int,
    h: c_int,
) -> ());

impl pal_idx_finish::Fn {
    /// If `dst` is [`None`], `tmp` is used as `dst`.
    /// This is why `tmp` must be `&mut`, too.
    /// `tmp` is always used as `src`.
    pub fn call(
        &self,
        dst: Option<&mut [u8]>,
        tmp: &mut [u8],
        bw: usize,
        bh: usize,
        w: usize,
        h: usize,
    ) {
        let dst = dst.map(|dst| &mut dst[..(bw / 2) * bh]);
        let tmp = &mut tmp[..bw * bh];
        // SAFETY: Note that `dst` and `src` may be the same.
        // This is safe because they are raw ptrs for now,
        // and in the fallback `fn pal_idx_finish_rust`, this is checked for
        // before creating `&mut`s from them.
        let dst = dst.unwrap_or(tmp).as_mut_ptr();
        let src = tmp.as_ptr();
        let [bw, bh, w, h] = [bw, bh, w, h].map(|it| it as c_int);
        // SAFETY: Fallback `fn pal_idx_finish_rust` is safe; asm is supposed to do the same.
        unsafe { self.get()(dst, src, bw, bh, w, h) }
    }
}

pub struct Rav1dPalDSPContext {
    pub pal_idx_finish: pal_idx_finish::Fn,
}

enum PalIdx<'a> {
    Idx { dst: &'a mut [u8], src: &'a [u8] },
    Tmp(&'a mut [u8]),
}

/// # Safety
///
/// Must be called by [`pal_idx_finish::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn pal_idx_finish_c(
    dst: *mut u8,
    src: *const u8,
    bw: c_int,
    bh: c_int,
    w: c_int,
    h: c_int,
) {
    let [bw, bh, w, h] = [bw, bh, w, h].map(|it| it as usize);

    assert!(bw >= 4 && bw <= 64 && bw.is_power_of_two());
    assert!(bh >= 4 && bh <= 64 && bh.is_power_of_two());
    assert!(w >= 4 && w <= bw && (w & 3) == 0);
    assert!(h >= 4 && h <= bh && (h & 3) == 0);

    let idx = if src == dst {
        // SAFETY: `src` length sliced in `pal_idx_finish::Fn::call` and `src == dst`.
        let tmp = unsafe { slice::from_raw_parts_mut(dst, bw * bh) };
        PalIdx::Tmp(tmp)
    } else {
        // SAFETY: `src` length sliced in `pal_idx_finish::Fn::call`.
        let src = unsafe { slice::from_raw_parts(src, bw * bh) };
        // SAFETY: `dst` length sliced in `pal_idx_finish::Fn::call`.
        let dst = unsafe { slice::from_raw_parts_mut(dst, (bw / 2) * bh) };
        PalIdx::Idx { dst, src }
    };

    pal_idx_finish_rust(idx, bw, bh, w, h)
}

/// Fill invisible edges and pack to 4-bit (2 pixels per byte).
fn pal_idx_finish_rust(idx: PalIdx, bw: usize, bh: usize, w: usize, h: usize) {
    let dst_w = w / 2;
    let dst_bw = bw / 2;

    let dst = match idx {
        PalIdx::Tmp(tmp) => {
            for y in 0..h {
                let src = y * bw;
                let dst = y * dst_bw;
                for x in 0..dst_w {
                    let src = &tmp[src + 2 * x..][..2];
                    tmp[dst + x] = src[0] | (src[1] << 4)
                }
                if dst_w < dst_bw {
                    let src = tmp[src + w];
                    tmp[dst..][dst_w..dst_bw].fill(0x11 * src);
                }
            }

            &mut tmp[..dst_bw * bh]
        }
        PalIdx::Idx { dst, src } => {
            for y in 0..h {
                let src = &src[y * bw..];
                let dst = &mut dst[y * dst_bw..];
                for x in 0..dst_w {
                    dst[x] = src[2 * x] | (src[2 * x + 1] << 4)
                }
                if dst_w < dst_bw {
                    dst[dst_w..dst_bw].fill(0x11 * src[w]);
                }
            }

            dst
        }
    };

    if h < bh {
        let (last_row, dst) = dst[(h - 1) * dst_bw..].split_at_mut(dst_bw);

        for row in dst.chunks_exact_mut(dst_bw) {
            row.copy_from_slice(last_row);
        }
    }
}

/// [`pal_idx_finish_rust`] sixteen indices at a time: `lo | hi << 4` of each pair from their
/// `u16` lanes.
mod rust_simd {
    use super::*;
    use crate::simd128::I16x8;

    #[inline(always)]
    fn pack_row(src: &[u8], dst: &mut [u8], w: usize, dst_bw: usize) {
        let mut x = 0;
        while x + 16 <= w {
            let v = I16x8::from_bytes(&src[x..]);
            let v = v.and(I16x8::splat(0xff)).or(v.shr_u(4).and(I16x8::splat(0xf0)));
            dst[x / 2..][..8].copy_from_slice(&v.pack_u8());
            x += 16;
        }
        for x in x / 2..w / 2 {
            dst[x] = src[2 * x] | src[2 * x + 1] << 4;
        }
        if w / 2 < dst_bw {
            dst[w / 2..dst_bw].fill(0x11 * src[w]);
        }
    }

    #[cfg_attr(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature(enable = "sse4.1")
    )]
    fn finish(idx: PalIdx, bw: usize, bh: usize, w: usize, h: usize) {
        let dst_bw = bw / 2;
        let dst = match idx {
            PalIdx::Tmp(tmp) => {
                let mut row = [0; 64];
                for y in 0..h {
                    row[..bw].copy_from_slice(&tmp[y * bw..][..bw]);
                    pack_row(&row, &mut tmp[y * dst_bw..][..dst_bw], w, dst_bw);
                }
                &mut tmp[..dst_bw * bh]
            }
            PalIdx::Idx { dst, src } => {
                for y in 0..h {
                    pack_row(&src[y * bw..][..bw], &mut dst[y * dst_bw..][..dst_bw], w, dst_bw);
                }
                dst
            }
        };
        if h < bh {
            let (last_row, dst) = dst[(h - 1) * dst_bw..].split_at_mut(dst_bw);
            for row in dst.chunks_exact_mut(dst_bw) {
                row.copy_from_slice(last_row);
            }
        }
    }

    /// # Safety
    ///
    /// Must be called by [`pal_idx_finish::Fn::call`], on x86 once the CPU flags hold SSE4.1.
    #[allow(unused_unsafe)] // A safe call on aarch64.
    pub unsafe extern "C" fn pal_idx_finish_simd(
        dst: *mut u8,
        src: *const u8,
        bw: c_int,
        bh: c_int,
        w: c_int,
        h: c_int,
    ) {
        let [bw, bh, w, h] = [bw, bh, w, h].map(|it| it as usize);
        assert!(bw >= 4 && bw <= 64 && bw.is_power_of_two());
        assert!(bh >= 4 && bh <= 64 && bh.is_power_of_two());
        assert!(w >= 4 && w <= bw && (w & 3) == 0);
        assert!(h >= 4 && h <= bh && (h & 3) == 0);
        let idx = if src == dst {
            // SAFETY: `src` length sliced in `pal_idx_finish::Fn::call` and `src == dst`.
            PalIdx::Tmp(unsafe { slice::from_raw_parts_mut(dst, bw * bh) })
        } else {
            PalIdx::Idx {
                // SAFETY: `src` length sliced in `pal_idx_finish::Fn::call`.
                src: unsafe { slice::from_raw_parts(src, bw * bh) },
                // SAFETY: `dst` length sliced in `pal_idx_finish::Fn::call`.
                dst: unsafe { slice::from_raw_parts_mut(dst, (bw / 2) * bh) },
            }
        };
        // SAFETY: SSE4.1 is checked before this `fn` is chosen on x86; elsewhere it is safe.
        unsafe { finish(idx, bw, bh, w, h) }
    }
}

impl Rav1dPalDSPContext {
    pub const fn default() -> Self {
        Self {
            pal_idx_finish: pal_idx_finish::Fn::new(pal_idx_finish_c),
        }
    }

    #[inline(always)]
    const fn init(mut self, flags: CpuFlags) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if !flags.contains(CpuFlags::SSE41) {
            return self;
        }
        let _ = flags;
        if !crate::simd128::native() {
            return self;
        }
        self.pal_idx_finish = pal_idx_finish::Fn::new(rust_simd::pal_idx_finish_simd);
        self
    }

    pub const fn new(flags: CpuFlags) -> Self {
        Self::default().init(flags)
    }
}

impl Default for Rav1dPalDSPContext {
    fn default() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;
    use std::time::Instant;

    use super::*;
    use crate::align::AlignedVec64;
    use crate::cpu::{rav1d_get_cpu_flags, rav1d_init_cpu};

    fn contexts() -> Vec<(&'static str, Rav1dPalDSPContext)> {
        rav1d_init_cpu();
        let flags = rav1d_get_cpu_flags();
        vec![
            ("scalar", Rav1dPalDSPContext::default()),
            ("rust simd", Rav1dPalDSPContext::default().init(flags)),
        ]
    }

    fn aligned(b: &[u8]) -> AlignedVec64<u8> {
        let mut v = AlignedVec64::new();
        v.resize(b.len(), 0);
        v.as_mut_slice().copy_from_slice(b);
        v
    }

    /// Every size, visible size and both the in-place and separate forms, on random indices.
    #[test]
    fn pal_idx_finish_matches_scalar() {
        let mut r = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            (r >> 32) as u8
        };
        let dsps = contexts();
        for bw in [4, 8, 16, 32, 64] {
            for bh in [4, 8, 16, 32, 64] {
                for w in (4..=bw).step_by(4) {
                    for h in (4..=bh).step_by(4) {
                        let mut src: Vec<u8> = (0..bw * bh).map(|_| next() & 7).collect();
                        // The decoder repeats the last visible index to the right, which the
                        // x86 asm relies on.
                        for row in src.chunks_exact_mut(bw) {
                            let last = row[w - 1];
                            row[w..].fill(last);
                        }
                        for in_place in [false, true] {
                            // The asm takes aligned buffers, as the decoder's are.
                            let run = |dsp: &Rav1dPalDSPContext| {
                                let mut tmp = aligned(&src);
                                let mut dst = aligned(&vec![0xaa; bw / 2 * bh]);
                                let (tmp, dst) = (tmp.as_mut_slice(), dst.as_mut_slice());
                                if in_place {
                                    dsp.pal_idx_finish.call(None, tmp, bw, bh, w, h);
                                    tmp[..bw / 2 * bh].to_vec()
                                } else {
                                    dsp.pal_idx_finish.call(Some(dst), tmp, bw, bh, w, h);
                                    dst.to_vec()
                                }
                            };
                            let want = run(&dsps[0].1);
                            for (name, dsp) in &dsps[1..] {
                                assert!(
                                    run(dsp) == want,
                                    "{name} differs: {bw}x{bh} visible {w}x{h} in place {in_place}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// `cargo test --release --lib pal::tests::time -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn time() {
        let src: Vec<u8> = (0..64 * 64).map(|i| (i * 7 % 8) as u8).collect();
        for (name, dsp) in contexts() {
            let mut line = format!("{name:10}");
            for bw in [8, 16, 32, 64] {
                let mut tmp = aligned(&src);
                let mut dst = aligned(&vec![0; bw / 2 * bw]);
                let mut best = f64::MAX;
                for _ in 0..7 {
                    let start = Instant::now();
                    for _ in 0..1000 {
                        dsp.pal_idx_finish.call(
                            Some(dst.as_mut_slice()),
                            black_box(tmp.as_mut_slice()),
                            bw,
                            bw,
                            bw,
                            bw,
                        );
                    }
                    best = best.min(start.elapsed().as_secs_f64() * 1e9 / 1000.);
                }
                line += &format!("  {bw}x{bw} {best:6.1} ns");
            }
            println!("{line}");
        }
    }
}
