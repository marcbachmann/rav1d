// Copyright (c) 2019-2023, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! rav1e's NEON lane and 2D driver (`src/asm/aarch64/transform/forward.rs`), the twin of
//! `x86.rs`: the same coded rows, `i32` stores and 64-bit loads of 4-wide rows.

#![allow(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]

use std::arch::aarch64::*;
use std::mem::MaybeUninit;

use super::{array, array_mut, assume_init_mut, ColumnBuf, Config, Txfm1d};

#[derive(Clone, Copy)]
#[repr(transparent)]
struct I32X8(int32x4x2_t);

impl I32X8 {
    #[inline(always)]
    fn vec(self) -> int32x4x2_t {
        self.0
    }

    #[inline(always)]
    fn raw(a: int32x4x2_t) -> I32X8 {
        Self(a)
    }

    #[inline(always)]
    fn new(a: int32x4_t, b: int32x4_t) -> I32X8 {
        Self(int32x4x2_t(a, b))
    }
}

type TxfmFunc = unsafe fn(&mut [I32X8]);

impl_1d_tx!(allow(unused_attributes), unsafe);

impl TxOperations for I32X8 {
    #[inline]
    unsafe fn zero() -> Self {
        let zero = vdupq_n_s32(0);
        I32X8::new(zero, zero)
    }

    #[inline]
    unsafe fn tx_mul<const SHIFT: i32>(self, mul: i32) -> Self {
        I32X8::new(
            vrshrq_n_s32::<SHIFT>(vmulq_n_s32(self.vec().0, mul)),
            vrshrq_n_s32::<SHIFT>(vmulq_n_s32(self.vec().1, mul)),
        )
    }

    #[inline]
    unsafe fn rshift1(self) -> Self {
        I32X8::new(
            vhsubq_s32(self.vec().0, vreinterpretq_s32_u32(vcltzq_s32(self.vec().0))),
            vhsubq_s32(self.vec().1, vreinterpretq_s32_u32(vcltzq_s32(self.vec().1))),
        )
    }

    #[inline]
    unsafe fn add(self, b: Self) -> Self {
        I32X8::new(vaddq_s32(self.vec().0, b.vec().0), vaddq_s32(self.vec().1, b.vec().1))
    }

    #[inline]
    unsafe fn sub(self, b: Self) -> Self {
        I32X8::new(vsubq_s32(self.vec().0, b.vec().0), vsubq_s32(self.vec().1, b.vec().1))
    }

    #[inline]
    unsafe fn add_avg(self, b: Self) -> Self {
        I32X8::new(vhaddq_s32(self.vec().0, b.vec().0), vhaddq_s32(self.vec().1, b.vec().1))
    }

    #[inline]
    unsafe fn sub_avg(self, b: Self) -> Self {
        I32X8::new(vhsubq_s32(self.vec().0, b.vec().0), vhsubq_s32(self.vec().1, b.vec().1))
    }
}

#[inline]
unsafe fn vtrnq_s64_to_s32(a: int32x4_t, b: int32x4_t) -> int32x4x2_t {
    let a = vreinterpretq_s64_s32(a);
    let b = vreinterpretq_s64_s32(b);
    int32x4x2_t(
        vreinterpretq_s32_s64(vtrn1q_s64(a, b)),
        vreinterpretq_s32_s64(vtrn2q_s64(a, b)),
    )
}

#[inline]
unsafe fn transpose_8x8(input: &[I32X8; 8], into: &mut [MaybeUninit<I32X8>; 8]) {
    let stage1 = (
        vtrnq_s32(input[0].vec().0, input[1].vec().0),
        vtrnq_s32(input[2].vec().0, input[3].vec().0),
        vtrnq_s32(input[4].vec().0, input[5].vec().0),
        vtrnq_s32(input[6].vec().0, input[7].vec().0),
        vtrnq_s32(input[0].vec().1, input[1].vec().1),
        vtrnq_s32(input[2].vec().1, input[3].vec().1),
        vtrnq_s32(input[4].vec().1, input[5].vec().1),
        vtrnq_s32(input[6].vec().1, input[7].vec().1),
    );
    let stage2 = (
        vtrnq_s64_to_s32(stage1.0 .0, stage1.1 .0),
        vtrnq_s64_to_s32(stage1.0 .1, stage1.1 .1),
        vtrnq_s64_to_s32(stage1.2 .0, stage1.3 .0),
        vtrnq_s64_to_s32(stage1.2 .1, stage1.3 .1),
        vtrnq_s64_to_s32(stage1.4 .0, stage1.5 .0),
        vtrnq_s64_to_s32(stage1.4 .1, stage1.5 .1),
        vtrnq_s64_to_s32(stage1.6 .0, stage1.7 .0),
        vtrnq_s64_to_s32(stage1.6 .1, stage1.7 .1),
    );
    into[0].write(I32X8::new(stage2.0 .0, stage2.2 .0));
    into[1].write(I32X8::new(stage2.1 .0, stage2.3 .0));
    into[2].write(I32X8::new(stage2.0 .1, stage2.2 .1));
    into[3].write(I32X8::new(stage2.1 .1, stage2.3 .1));
    into[4].write(I32X8::new(stage2.4 .0, stage2.6 .0));
    into[5].write(I32X8::new(stage2.5 .0, stage2.7 .0));
    into[6].write(I32X8::new(stage2.4 .1, stage2.6 .1));
    into[7].write(I32X8::new(stage2.5 .1, stage2.7 .1));
}

#[inline]
unsafe fn transpose_8x4(input: &[I32X8; 8], into: &mut [MaybeUninit<I32X8>; 4]) {
    let stage1 = (
        vtrnq_s32(input[0].vec().0, input[1].vec().0),
        vtrnq_s32(input[2].vec().0, input[3].vec().0),
        vtrnq_s32(input[4].vec().0, input[5].vec().0),
        vtrnq_s32(input[6].vec().0, input[7].vec().0),
    );
    let stage2 = (
        vtrnq_s64_to_s32(stage1.0 .0, stage1.1 .0),
        vtrnq_s64_to_s32(stage1.0 .1, stage1.1 .1),
        vtrnq_s64_to_s32(stage1.2 .0, stage1.3 .0),
        vtrnq_s64_to_s32(stage1.2 .1, stage1.3 .1),
    );
    into[0].write(I32X8::new(stage2.0 .0, stage2.2 .0));
    into[1].write(I32X8::new(stage2.1 .0, stage2.3 .0));
    into[2].write(I32X8::new(stage2.0 .1, stage2.2 .1));
    into[3].write(I32X8::new(stage2.1 .1, stage2.3 .1));
}

#[inline]
unsafe fn transpose_4x8(input: &[I32X8; 4], into: &mut [MaybeUninit<I32X8>; 8]) {
    let stage1 = (
        vtrnq_s32(input[0].vec().0, input[1].vec().0),
        vtrnq_s32(input[2].vec().0, input[3].vec().0),
        vtrnq_s32(input[0].vec().1, input[1].vec().1),
        vtrnq_s32(input[2].vec().1, input[3].vec().1),
    );
    let stage2 = (
        vtrnq_s64_to_s32(stage1.0 .0, stage1.1 .0),
        vtrnq_s64_to_s32(stage1.0 .1, stage1.1 .1),
        vtrnq_s64_to_s32(stage1.2 .0, stage1.3 .0),
        vtrnq_s64_to_s32(stage1.2 .1, stage1.3 .1),
    );
    into[0].write(I32X8::raw(stage2.0));
    into[1].write(I32X8::raw(stage2.1));
    into[2].write(I32X8::new(stage2.0 .1, stage2.0 .0));
    into[3].write(I32X8::new(stage2.1 .1, stage2.1 .0));
    into[4].write(I32X8::raw(stage2.2));
    into[5].write(I32X8::raw(stage2.3));
    into[6].write(I32X8::new(stage2.2 .1, stage2.2 .0));
    into[7].write(I32X8::new(stage2.3 .1, stage2.3 .0));
}

#[inline]
unsafe fn transpose_4x4(input: &[I32X8; 4], into: &mut [MaybeUninit<I32X8>; 4]) {
    let stage1 = (
        vtrnq_s32(input[0].vec().0, input[1].vec().0),
        vtrnq_s32(input[2].vec().0, input[3].vec().0),
    );
    let stage2 = (
        vtrnq_s64_to_s32(stage1.0 .0, stage1.1 .0),
        vtrnq_s64_to_s32(stage1.0 .1, stage1.1 .1),
    );
    into[0].write(I32X8::raw(stage2.0));
    into[1].write(I32X8::raw(stage2.1));
    into[2].write(I32X8::new(stage2.0 .1, stage2.0 .0));
    into[3].write(I32X8::new(stage2.1 .1, stage2.1 .0));
}

#[inline]
unsafe fn round_shift(arr: &mut [I32X8], bit: i8) {
    if bit > 0 {
        let shift = vdupq_n_s32(-(bit as i32));
        for v in arr {
            *v = I32X8::new(vrshlq_s32(v.vec().0, shift), vrshlq_s32(v.vec().1, shift));
        }
    }
}

/// # Safety
///
/// NEON is available.
pub(super) unsafe fn forward(residual: &[i16], stride: usize, coefs: &mut [i32], cfg: &Config) {
    let mut buf = ColumnBuf::new();
    forward_rows(residual, stride, coefs, cfg, &mut buf, false)
}

/// `forward` with the column pass in `buf`, computed unless `have` says the caller's last call
/// on the same residual left it there with the same vertical transform.
///
/// # Safety
///
/// NEON is available, and `buf` holds this residual's column pass when `have`.
pub(super) unsafe fn forward_rows(residual: &[i16], stride: usize, coefs: &mut [i32], cfg: &Config, buf: &mut ColumnBuf, have: bool) {
    let &Config { w, h, .. } = cfg;
    let (cw, ch) = (w.min(32), h.min(32));
    // SAFETY: `I32X8` is 8 `i32`s, 32 bytes aligned to 16, and the buffer is aligned to 32.
    let buf = &mut *(&mut buf.0 as *mut MaybeUninit<[[i32; 8]; 64 * 32 / 8]> as *mut [MaybeUninit<I32X8>; 64 * 32 / 8]);
    if !have {
        let col_fn = get_func(cfg.col);
        let shift = vdupq_n_s32(cfg.shift[0] as i32);
        for cg in (0..w).step_by(8) {
            let mut col = [MaybeUninit::<I32X8>::uninit(); 64];
            for r in 0..h {
                let y = if cfg.ud_flip { h - 1 - r } else { r };
                let src = &residual[y * stride + cg..];
                let (lo, hi) = if w == 4 {
                    (vmovl_s16(vld1_s16(src[..4].as_ptr())), vdupq_n_s32(0))
                } else {
                    let v = vld1q_s16(src[..8].as_ptr());
                    (vmovl_s16(vget_low_s16(v)), vmovl_high_s16(v))
                };
                col[r].write(I32X8::new(vshlq_s32(lo, shift), vshlq_s32(hi, shift)));
            }
            let col = assume_init_mut(&mut col[..h]);
            col_fn(col);
            round_shift(col, -cfg.shift[1]);
            match (h == 4, w == 4) {
                (false, false) => {
                    for rg in (0..ch).step_by(8) {
                        transpose_8x8(array(&col[rg..]), array_mut(&mut buf[rg / 8 * w + cg..]));
                    }
                }
                (false, true) => {
                    for rg in (0..ch).step_by(8) {
                        transpose_8x4(array(&col[rg..]), array_mut(&mut buf[rg / 8 * w..]));
                    }
                }
                (true, false) => transpose_4x8(array(col), array_mut(&mut buf[cg..])),
                (true, true) => transpose_4x4(array(col), array_mut(&mut buf[..])),
            }
        }
    }
    let row_fn = get_func(cfg.row);
    for rg in (0..ch).step_by(8) {
        let mut row = [MaybeUninit::<I32X8>::uninit(); 64];
        row[..w].copy_from_slice(&buf[rg / 8 * w..][..w]);
        let row = assume_init_mut(&mut row[..w]);
        if cfg.lr_flip {
            row.reverse();
        }
        row_fn(row);
        round_shift(row, -cfg.shift[2]);
        for c in 0..cw {
            let dst = &mut coefs[c * ch + rg..];
            if ch == 4 {
                vst1q_s32(dst[..4].as_mut_ptr(), row[c].vec().0);
            } else {
                vst1q_s32(dst[..8].as_mut_ptr(), row[c].vec().0);
                vst1q_s32(dst[4..8].as_mut_ptr(), row[c].vec().1);
            }
        }
    }
}
