// Copyright (c) 2018-2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! rav1e's `i32` lane and 2D driver (`src/transform/forward.rs`), storing only the coded
//! rows and columns, column major.

use super::{Config, Txfm1d};

type TxfmFunc = fn(&mut [i32]);

impl_1d_tx!();

impl TxOperations for i32 {
    fn zero() -> Self {
        0
    }

    fn tx_mul<const SHIFT: i32>(self, mul: i32) -> Self {
        ((self * mul) + (1 << SHIFT >> 1)) >> SHIFT
    }

    fn rshift1(self) -> Self {
        (self + i32::from(self < 0)) >> 1
    }

    fn add(self, b: Self) -> Self {
        self + b
    }

    fn sub(self, b: Self) -> Self {
        self - b
    }

    fn add_avg(self, b: Self) -> Self {
        (self + b) >> 1
    }

    fn sub_avg(self, b: Self) -> Self {
        (self - b) >> 1
    }
}

fn round_shift(arr: &mut [i32], bit: i8) {
    if bit > 0 {
        for v in arr {
            *v = (*v + (1 << (bit - 1))) >> bit;
        }
    }
}

pub(super) fn forward(residual: &[i16], stride: usize, coefs: &mut [i32], cfg: &Config) {
    let &Config { w, h, .. } = cfg;
    let (cw, ch) = (w.min(32), h.min(32));
    let col_fn = get_func(cfg.col);
    let row_fn = get_func(cfg.row);
    let mut buf = [0; 64 * 32];
    let mut col = [0; 64];
    for c in 0..w {
        for r in 0..h {
            let y = if cfg.ud_flip { h - 1 - r } else { r };
            col[r] = (residual[y * stride + c] as i32) << cfg.shift[0];
        }
        col_fn(&mut col[..h]);
        round_shift(&mut col[..h], -cfg.shift[1]);
        let x = if cfg.lr_flip { w - 1 - c } else { c };
        for r in 0..ch {
            buf[r * w + x] = col[r];
        }
    }
    for r in 0..ch {
        let row = &mut buf[r * w..][..w];
        row_fn(row);
        round_shift(row, -cfg.shift[2]);
        for c in 0..cw {
            coefs[c * ch + r] = row[c];
        }
    }
}
