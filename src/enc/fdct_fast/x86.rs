// Copyright (c) 2019-2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! rav1e's AVX2 lane and 2D driver (`src/asm/x86/transform/forward.rs`): eight columns,
//! then eight rows, at a time. Stores `i32` coefficients of the coded rows only; 4-wide
//! rows are read with 64-bit loads.

#![allow(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::mem::MaybeUninit;

use super::{array, array_mut, assume_init_mut, ColumnBuf, Config, Txfm1d};

#[derive(Copy, Clone)]
struct I32X8 {
    data: __m256i,
}

impl I32X8 {
    #[inline(always)]
    fn vec(self) -> __m256i {
        self.data
    }

    #[inline(always)]
    fn new(data: __m256i) -> I32X8 {
        I32X8 { data }
    }
}

type TxfmFunc = unsafe fn(&mut [I32X8]);

impl_1d_tx!(target_feature(enable = "avx2"), unsafe);

impl TxOperations for I32X8 {
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn zero() -> Self {
        I32X8::new(_mm256_setzero_si256())
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn tx_mul<const SHIFT: i32>(self, mul: i32) -> Self {
        I32X8::new(_mm256_srav_epi32(
            _mm256_add_epi32(
                _mm256_mullo_epi32(self.vec(), _mm256_set1_epi32(mul)),
                _mm256_set1_epi32(1 << SHIFT >> 1),
            ),
            _mm256_set1_epi32(SHIFT),
        ))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn rshift1(self) -> Self {
        I32X8::new(_mm256_srai_epi32(
            _mm256_sub_epi32(self.vec(), _mm256_cmpgt_epi32(_mm256_setzero_si256(), self.vec())),
            1,
        ))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn add(self, b: Self) -> Self {
        I32X8::new(_mm256_add_epi32(self.vec(), b.vec()))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn sub(self, b: Self) -> Self {
        I32X8::new(_mm256_sub_epi32(self.vec(), b.vec()))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn add_avg(self, b: Self) -> Self {
        I32X8::new(_mm256_srai_epi32(_mm256_add_epi32(self.vec(), b.vec()), 1))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn sub_avg(self, b: Self) -> Self {
        I32X8::new(_mm256_srai_epi32(_mm256_sub_epi32(self.vec(), b.vec()), 1))
    }
}

#[target_feature(enable = "avx2")]
unsafe fn transpose_8x8(input: &[I32X8; 8], into: &mut [MaybeUninit<I32X8>; 8]) {
    let stage1 = (
        _mm256_unpacklo_epi32(input[0].vec(), input[1].vec()),
        _mm256_unpackhi_epi32(input[0].vec(), input[1].vec()),
        _mm256_unpacklo_epi32(input[2].vec(), input[3].vec()),
        _mm256_unpackhi_epi32(input[2].vec(), input[3].vec()),
        _mm256_unpacklo_epi32(input[4].vec(), input[5].vec()),
        _mm256_unpackhi_epi32(input[4].vec(), input[5].vec()),
        _mm256_unpacklo_epi32(input[6].vec(), input[7].vec()),
        _mm256_unpackhi_epi32(input[6].vec(), input[7].vec()),
    );

    let stage2 = (
        _mm256_unpacklo_epi64(stage1.0, stage1.2),
        _mm256_unpackhi_epi64(stage1.0, stage1.2),
        _mm256_unpacklo_epi64(stage1.1, stage1.3),
        _mm256_unpackhi_epi64(stage1.1, stage1.3),
        _mm256_unpacklo_epi64(stage1.4, stage1.6),
        _mm256_unpackhi_epi64(stage1.4, stage1.6),
        _mm256_unpacklo_epi64(stage1.5, stage1.7),
        _mm256_unpackhi_epi64(stage1.5, stage1.7),
    );

    const LO: i32 = 2 << 4;
    const HI: i32 = (3 << 4) | 1;
    into[0].write(I32X8::new(_mm256_permute2x128_si256::<LO>(stage2.0, stage2.4)));
    into[1].write(I32X8::new(_mm256_permute2x128_si256::<LO>(stage2.1, stage2.5)));
    into[2].write(I32X8::new(_mm256_permute2x128_si256::<LO>(stage2.2, stage2.6)));
    into[3].write(I32X8::new(_mm256_permute2x128_si256::<LO>(stage2.3, stage2.7)));
    into[4].write(I32X8::new(_mm256_permute2x128_si256::<HI>(stage2.0, stage2.4)));
    into[5].write(I32X8::new(_mm256_permute2x128_si256::<HI>(stage2.1, stage2.5)));
    into[6].write(I32X8::new(_mm256_permute2x128_si256::<HI>(stage2.2, stage2.6)));
    into[7].write(I32X8::new(_mm256_permute2x128_si256::<HI>(stage2.3, stage2.7)));
}

#[target_feature(enable = "avx2")]
unsafe fn transpose_8x4(input: &[I32X8; 8], into: &mut [MaybeUninit<I32X8>; 4]) {
    let stage1 = (
        _mm256_unpacklo_epi32(input[0].vec(), input[1].vec()),
        _mm256_unpackhi_epi32(input[0].vec(), input[1].vec()),
        _mm256_unpacklo_epi32(input[2].vec(), input[3].vec()),
        _mm256_unpackhi_epi32(input[2].vec(), input[3].vec()),
        _mm256_unpacklo_epi32(input[4].vec(), input[5].vec()),
        _mm256_unpackhi_epi32(input[4].vec(), input[5].vec()),
        _mm256_unpacklo_epi32(input[6].vec(), input[7].vec()),
        _mm256_unpackhi_epi32(input[6].vec(), input[7].vec()),
    );

    let stage2 = (
        _mm256_unpacklo_epi64(stage1.0, stage1.2),
        _mm256_unpackhi_epi64(stage1.0, stage1.2),
        _mm256_unpacklo_epi64(stage1.1, stage1.3),
        _mm256_unpackhi_epi64(stage1.1, stage1.3),
        _mm256_unpacklo_epi64(stage1.4, stage1.6),
        _mm256_unpackhi_epi64(stage1.4, stage1.6),
        _mm256_unpacklo_epi64(stage1.5, stage1.7),
        _mm256_unpackhi_epi64(stage1.5, stage1.7),
    );

    const LO: i32 = 2 << 4;
    into[0].write(I32X8::new(_mm256_permute2x128_si256::<LO>(stage2.0, stage2.4)));
    into[1].write(I32X8::new(_mm256_permute2x128_si256::<LO>(stage2.1, stage2.5)));
    into[2].write(I32X8::new(_mm256_permute2x128_si256::<LO>(stage2.2, stage2.6)));
    into[3].write(I32X8::new(_mm256_permute2x128_si256::<LO>(stage2.3, stage2.7)));
}

#[target_feature(enable = "avx2")]
unsafe fn transpose_4x8(input: &[I32X8; 4], into: &mut [MaybeUninit<I32X8>; 8]) {
    let stage1 = (
        _mm256_unpacklo_epi32(input[0].vec(), input[1].vec()),
        _mm256_unpackhi_epi32(input[0].vec(), input[1].vec()),
        _mm256_unpacklo_epi32(input[2].vec(), input[3].vec()),
        _mm256_unpackhi_epi32(input[2].vec(), input[3].vec()),
    );

    let stage2 = (
        _mm256_unpacklo_epi64(stage1.0, stage1.2),
        _mm256_unpackhi_epi64(stage1.0, stage1.2),
        _mm256_unpacklo_epi64(stage1.1, stage1.3),
        _mm256_unpackhi_epi64(stage1.1, stage1.3),
    );

    into[0].write(I32X8::new(stage2.0));
    into[1].write(I32X8::new(stage2.1));
    into[2].write(I32X8::new(stage2.2));
    into[3].write(I32X8::new(stage2.3));
    into[4].write(I32X8::new(_mm256_castsi128_si256(_mm256_extracti128_si256::<1>(stage2.0))));
    into[5].write(I32X8::new(_mm256_castsi128_si256(_mm256_extracti128_si256::<1>(stage2.1))));
    into[6].write(I32X8::new(_mm256_castsi128_si256(_mm256_extracti128_si256::<1>(stage2.2))));
    into[7].write(I32X8::new(_mm256_castsi128_si256(_mm256_extracti128_si256::<1>(stage2.3))));
}

#[target_feature(enable = "avx2")]
unsafe fn transpose_4x4(input: &[I32X8; 4], into: &mut [MaybeUninit<I32X8>; 4]) {
    let stage1 = (
        _mm256_unpacklo_epi32(input[0].vec(), input[1].vec()),
        _mm256_unpackhi_epi32(input[0].vec(), input[1].vec()),
        _mm256_unpacklo_epi32(input[2].vec(), input[3].vec()),
        _mm256_unpackhi_epi32(input[2].vec(), input[3].vec()),
    );

    into[0].write(I32X8::new(_mm256_unpacklo_epi64(stage1.0, stage1.2)));
    into[1].write(I32X8::new(_mm256_unpackhi_epi64(stage1.0, stage1.2)));
    into[2].write(I32X8::new(_mm256_unpacklo_epi64(stage1.1, stage1.3)));
    into[3].write(I32X8::new(_mm256_unpackhi_epi64(stage1.1, stage1.3)));
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn round_shift(arr: &mut [I32X8], bit: i8) {
    if bit > 0 {
        let round = _mm256_set1_epi32(1 << (bit - 1));
        let bit = _mm_cvtsi32_si128(bit as i32);
        for v in arr {
            *v = I32X8::new(_mm256_sra_epi32(_mm256_add_epi32(v.vec(), round), bit));
        }
    }
}

/// # Safety
///
/// AVX2 is available.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn forward(residual: &[i16], stride: usize, coefs: &mut [i32], cfg: &Config) {
    let mut buf = ColumnBuf::new();
    forward_rows(residual, stride, coefs, cfg, &mut buf, false)
}

/// `forward` with the column pass in `buf`, computed unless `have` says the caller's last call
/// on the same residual left it there with the same vertical transform.
///
/// # Safety
///
/// AVX2 is available, and `buf` holds this residual's column pass when `have`.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn forward_rows(residual: &[i16], stride: usize, coefs: &mut [i32], cfg: &Config, buf: &mut ColumnBuf, have: bool) {
    let &Config { w, h, .. } = cfg;
    let (cw, ch) = (w.min(32), h.min(32));
    // SAFETY: `I32X8` is 8 `i32`s, 32 bytes, and the buffer is aligned to 32.
    let buf = &mut *(&mut buf.0 as *mut MaybeUninit<[[i32; 8]; 64 * 32 / 8]> as *mut [MaybeUninit<I32X8>; 64 * 32 / 8]);
    if !have {
        let col_fn = get_func(cfg.col);
        let shift = _mm_cvtsi32_si128(cfg.shift[0] as i32);
        for cg in (0..w).step_by(8) {
            let mut col = [MaybeUninit::<I32X8>::uninit(); 64];
            for r in 0..h {
                let y = if cfg.ud_flip { h - 1 - r } else { r };
                let src = &residual[y * stride + cg..];
                let v = if w == 4 {
                    _mm_loadl_epi64(src[..4].as_ptr() as *const __m128i)
                } else {
                    _mm_loadu_si128(src[..8].as_ptr() as *const __m128i)
                };
                col[r].write(I32X8::new(_mm256_sll_epi32(_mm256_cvtepi16_epi32(v), shift)));
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
                _mm_storeu_si128(dst[..4].as_mut_ptr() as *mut __m128i, _mm256_castsi256_si128(row[c].vec()));
            } else {
                _mm256_storeu_si256(dst[..8].as_mut_ptr() as *mut __m256i, row[c].vec());
            }
        }
    }
}
