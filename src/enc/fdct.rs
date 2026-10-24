//! Forward transforms, ported from libaom v3.15.1's `av1_fwd_txfm1d.c`,
//! `av1_fwd_txfm2d.c`, `av1_txfm.c` and `hybrid_fwd_txfm.c`
//! (Copyright (c) 2016, Alliance for Open Media, BSD-2). Same integer
//! arithmetic as libaom's C code, so the same coefficients.

use strum::EnumCount;

use crate::levels::{
    TxClass, TxfmSize, TxfmType, ADST_ADST, ADST_DCT, ADST_FLIPADST, DCT_ADST, DCT_DCT,
    DCT_FLIPADST, FLIPADST_ADST, FLIPADST_DCT, FLIPADST_FLIPADST, H_ADST, H_DCT, H_FLIPADST, IDTX,
    V_ADST, V_DCT, V_FLIPADST, WHT_WHT,
};
use crate::scan::DAV1D_SCANS;
use crate::tables::{DAV1D_TXFM_DIMENSIONS, DAV1D_TX_TYPE_CLASS};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Dct,
    Adst,
    FlipAdst,
    Identity,
}

/// (vertical, horizontal) 1D transforms of a 2D type.
fn kinds(ty: TxfmType) -> (Kind, Kind) {
    use Kind::*;
    match ty {
        DCT_DCT => (Dct, Dct),
        ADST_DCT => (Adst, Dct),
        DCT_ADST => (Dct, Adst),
        ADST_ADST => (Adst, Adst),
        FLIPADST_DCT => (FlipAdst, Dct),
        DCT_FLIPADST => (Dct, FlipAdst),
        FLIPADST_FLIPADST => (FlipAdst, FlipAdst),
        ADST_FLIPADST => (Adst, FlipAdst),
        FLIPADST_ADST => (FlipAdst, Adst),
        IDTX => (Identity, Identity),
        V_DCT => (Dct, Identity),
        H_DCT => (Identity, Dct),
        V_ADST => (Adst, Identity),
        H_ADST => (Identity, Adst),
        V_FLIPADST => (FlipAdst, Identity),
        H_FLIPADST => (Identity, FlipAdst),
        _ => panic!("no 2D transform type {ty}"),
    }
}

type Txfm1d = fn(input: &[i32], output: &mut [i32], cos_bit: i8);

fn txfm_1d(kind: Kind, n: usize) -> Option<Txfm1d> {
    use Kind::*;
    Some(match (kind, n) {
        (Dct, 4) => fdct4,
        (Dct, 8) => fdct8,
        (Dct, 16) => fdct16,
        (Dct, 32) => fdct32,
        (Dct, 64) => fdct64,
        (Adst | FlipAdst, 4) => fadst4,
        (Adst | FlipAdst, 8) => fadst8,
        (Adst | FlipAdst, 16) => fadst16,
        (Identity, 4) => fidentity4,
        (Identity, 8) => fidentity8,
        (Identity, 16) => fidentity16,
        (Identity, 32) => fidentity32,
        _ => return None,
    })
}

/// Whether libaom has a forward transform of this type and size. The
/// bitstream allows fewer: all 16 types up to 16×8 and 8×16, 12 at 16×16,
/// `DCT_DCT` and `IDTX` up to 32, only `DCT_DCT` at 64, `WHT_WHT` at 4×4.
pub fn supported(size: TxfmSize, ty: TxfmType) -> bool {
    let (w, h) = dimensions(size);
    if ty == WHT_WHT {
        return size == TxfmSize::S4x4;
    }
    if ty > H_FLIPADST {
        return false;
    }
    let (v, hz) = kinds(ty);
    txfm_1d(v, h).is_some() && txfm_1d(hz, w).is_some()
}

pub fn dimensions(size: TxfmSize) -> (usize, usize) {
    let t = &DAV1D_TXFM_DIMENSIONS[size as usize];
    (4 << t.lw, 4 << t.lh)
}

/// Coefficients kept for a size: 64-point sides keep their first 32.
pub fn coded_dimensions(size: TxfmSize) -> (usize, usize) {
    let (w, h) = dimensions(size);
    (w.min(32), h.min(32))
}

const FWD_SHIFT: [[i8; 3]; TxfmSize::COUNT] = [
    [2, 0, 0],
    [2, -1, 0],
    [2, -2, 0],
    [2, -4, 0],
    [0, -2, -2],
    [2, -1, 0],
    [2, -1, 0],
    [2, -2, 0],
    [2, -2, 0],
    [2, -4, 0],
    [2, -4, 0],
    [0, -2, -2],
    [2, -4, -2],
    [2, -1, 0],
    [2, -1, 0],
    [2, -2, 0],
    [2, -2, 0],
    [0, -2, 0],
    [2, -4, 0],
];

const FWD_COS_BIT_COL: [[i8; 5]; 5] = [
    [13, 13, 13, 0, 0],
    [13, 13, 13, 12, 0],
    [13, 13, 13, 12, 13],
    [0, 13, 13, 12, 13],
    [0, 0, 13, 12, 13],
];

const FWD_COS_BIT_ROW: [[i8; 5]; 5] = [
    [13, 13, 12, 0, 0],
    [13, 13, 13, 12, 0],
    [13, 13, 12, 13, 12],
    [0, 12, 13, 12, 11],
    [0, 0, 12, 11, 10],
];

const NEW_SQRT2: i64 = 5793;
const NEW_SQRT2_BITS: u32 = 12;

/// Transforms the `w`×`h` residual at `residual[y * stride + x]` into
/// `coefs[x * ch + y]` for `x < cw`, `y < ch` (`coded_dimensions`): column
/// major, the layout rav1d's inverse transforms and scans read.
pub fn forward(residual: &[i16], stride: usize, coefs: &mut [i32], size: TxfmSize, ty: TxfmType) {
    if ty == WHT_WHT {
        assert_eq!(size, TxfmSize::S4x4);
        return fwht4x4(residual, stride, coefs);
    }
    let (w, h) = dimensions(size);
    let (cw, ch) = coded_dimensions(size);
    let coefs = &mut coefs[..cw * ch];
    let (v, hz) = kinds(ty);
    let col_fn = txfm_1d(v, h).expect("column transform");
    let row_fn = txfm_1d(hz, w).expect("row transform");
    let ud_flip = v == Kind::FlipAdst;
    let lr_flip = hz == Kind::FlipAdst;
    let shift = FWD_SHIFT[size as usize];
    let (wi, hi) = (
        w.trailing_zeros() as usize - 2,
        h.trailing_zeros() as usize - 2,
    );
    let cos_bit_col = FWD_COS_BIT_COL[wi][hi];
    let cos_bit_row = FWD_COS_BIT_ROW[wi][hi];
    let rect2 = w == 2 * h || h == 2 * w;

    let mut buf = [0; 64 * 64];
    let mut col_in = [0; 64];
    let mut col_out = [0; 64];
    for c in 0..w {
        for r in 0..h {
            let y = if ud_flip { h - 1 - r } else { r };
            col_in[r] = residual[y * stride + c] as i32;
        }
        round_shift_array(&mut col_in[..h], -shift[0]);
        col_fn(&col_in[..h], &mut col_out[..h], cos_bit_col);
        round_shift_array(&mut col_out[..h], -shift[1]);
        let x = if lr_flip { w - 1 - c } else { c };
        for r in 0..h {
            buf[r * w + x] = col_out[r];
        }
    }

    let mut row_out = [0; 64];
    for r in 0..ch {
        row_fn(&buf[r * w..][..w], &mut row_out[..w], cos_bit_row);
        round_shift_array(&mut row_out[..w], -shift[2]);
        for c in 0..cw {
            let v = row_out[c];
            coefs[c * ch + r] = if rect2 {
                round_shift(v as i64 * NEW_SQRT2, NEW_SQRT2_BITS)
            } else {
                v
            };
        }
    }
}

/// The lossless Walsh-Hadamard transform, scaled by 4 as libaom codes it.
fn fwht4x4(residual: &[i16], stride: usize, coefs: &mut [i32]) {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let [a, b, c, d] = wht4(std::array::from_fn(|k| residual[k * stride + i] as i32));
        tmp[i * 4..][..4].copy_from_slice(&[a, c, d, b]);
    }
    for i in 0..4 {
        let [a, b, c, d] = wht4(std::array::from_fn(|k| tmp[k * 4 + i]));
        for (k, v) in [a, c, d, b].into_iter().enumerate() {
            coefs[k * 4 + i] = v * 4;
        }
    }
}

fn wht4([mut a1, mut b1, mut c1, mut d1]: [i32; 4]) -> [i32; 4] {
    a1 += b1;
    d1 -= c1;
    let e1 = (a1 - d1) >> 1;
    b1 = e1 - b1;
    c1 = e1 - c1;
    a1 -= c1;
    d1 += b1;
    [a1, b1, c1, d1]
}

fn round_shift(value: i64, bit: u32) -> i32 {
    ((value + (1 << (bit - 1))) >> bit) as i32
}

fn round_shift_array(arr: &mut [i32], bit: i8) {
    if bit > 0 {
        for v in arr {
            *v = round_shift(*v as i64, bit as u32);
        }
    } else if bit < 0 {
        for v in arr {
            *v = ((*v as i64) << -bit).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
    }
}

fn half_btf(w0: i32, in0: i32, w1: i32, in1: i32, bit: i8) -> i32 {
    let sum = w0.wrapping_mul(in0) as i64 + w1.wrapping_mul(in1) as i64;
    ((sum + (1 << (bit - 1))) >> bit) as i32
}

fn cospi_arr(cos_bit: i8) -> &'static [i32; 64] {
    &COSPI[cos_bit as usize - 10]
}

fn sinpi_arr(cos_bit: i8) -> &'static [i32; 5] {
    &SINPI[cos_bit as usize - 10]
}

fn fadst4(input: &[i32], output: &mut [i32], cos_bit: i8) {
    let sinpi = sinpi_arr(cos_bit);
    let [mut x0, mut x1, mut x2, mut x3] = [input[0], input[1], input[2], input[3]];
    if x0 | x1 | x2 | x3 == 0 {
        output[..4].fill(0);
        return;
    }
    let s0 = sinpi[1] * x0;
    let s1 = sinpi[4] * x0;
    let s2 = sinpi[2] * x1;
    let s3 = sinpi[1] * x1;
    let s4 = sinpi[3] * x2;
    let s5 = sinpi[4] * x3;
    let s6 = sinpi[2] * x3;
    let s7 = x0 + x1 - x3;
    x0 = s0 + s2;
    x1 = sinpi[3] * s7;
    x2 = s1 - s3;
    x3 = s4;
    x0 += s5;
    x2 += s6;
    let bit = cos_bit as u32;
    output[0] = round_shift((x0 + x3) as i64, bit);
    output[1] = round_shift(x1 as i64, bit);
    output[2] = round_shift((x2 - x3) as i64, bit);
    output[3] = round_shift((x2 - x0 + x3) as i64, bit);
}

fn fidentity4(input: &[i32], output: &mut [i32], _cos_bit: i8) {
    for (o, &i) in output[..4].iter_mut().zip(input) {
        *o = round_shift(i as i64 * NEW_SQRT2, NEW_SQRT2_BITS);
    }
}

fn fidentity8(input: &[i32], output: &mut [i32], _cos_bit: i8) {
    for (o, &i) in output[..8].iter_mut().zip(input) {
        *o = i * 2;
    }
}

fn fidentity16(input: &[i32], output: &mut [i32], _cos_bit: i8) {
    for (o, &i) in output[..16].iter_mut().zip(input) {
        *o = round_shift(i as i64 * 2 * NEW_SQRT2, NEW_SQRT2_BITS);
    }
}

fn fidentity32(input: &[i32], output: &mut [i32], _cos_bit: i8) {
    for (o, &i) in output[..32].iter_mut().zip(input) {
        *o = i * 4;
    }
}

/// Where the decoder stores the `i`th coefficient of its scan order.
pub fn scan_rc(size: TxfmSize, ty: TxfmType, i: usize) -> usize {
    let (cw, ch) = coded_dimensions(size);
    match DAV1D_TX_TYPE_CLASS[ty as usize] {
        TxClass::TwoD => DAV1D_SCANS[size as usize][i].get() as usize,
        TxClass::H => i,
        TxClass::V => (i % cw) * ch + i / cw,
    }
}

/// The `eob` rav1d's inverse transforms take: the scan position of the last
/// nonzero coefficient, `None` when all are zero.
pub fn last_nonzero(coefs: &[i32], size: TxfmSize, ty: TxfmType) -> Option<u16> {
    let (cw, ch) = coded_dimensions(size);
    (0..cw * ch)
        .rev()
        .find(|&i| coefs[scan_rc(size, ty, i)] != 0)
        .map(|i| i as u16)
}

const COSPI: [[i32; 64]; 4] = [
    [
        1024, 1024, 1023, 1021, 1019, 1016, 1013, 1009, 1004, 999, 993, 987, 980, 972, 964, 955,
        946, 936, 926, 915, 903, 891, 878, 865, 851, 837, 822, 807, 792, 775, 759, 742, 724, 706,
        688, 669, 650, 630, 610, 590, 569, 548, 526, 505, 483, 460, 438, 415, 392, 369, 345, 321,
        297, 273, 249, 224, 200, 175, 150, 125, 100, 75, 50, 25,
    ],
    [
        2048, 2047, 2046, 2042, 2038, 2033, 2026, 2018, 2009, 1998, 1987, 1974, 1960, 1945, 1928,
        1911, 1892, 1872, 1851, 1829, 1806, 1782, 1757, 1730, 1703, 1674, 1645, 1615, 1583, 1551,
        1517, 1483, 1448, 1412, 1375, 1338, 1299, 1260, 1220, 1179, 1138, 1096, 1053, 1009, 965,
        921, 876, 830, 784, 737, 690, 642, 595, 546, 498, 449, 400, 350, 301, 251, 201, 151, 100,
        50,
    ],
    [
        4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948, 3920, 3889, 3857,
        3822, 3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102,
        3035, 2967, 2896, 2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931,
        1842, 1751, 1660, 1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700, 601, 501, 401,
        301, 201, 101,
    ],
    [
        8192, 8190, 8182, 8170, 8153, 8130, 8103, 8071, 8035, 7993, 7946, 7895, 7839, 7779, 7713,
        7643, 7568, 7489, 7405, 7317, 7225, 7128, 7027, 6921, 6811, 6698, 6580, 6458, 6333, 6203,
        6070, 5933, 5793, 5649, 5501, 5351, 5197, 5040, 4880, 4717, 4551, 4383, 4212, 4038, 3862,
        3683, 3503, 3320, 3135, 2948, 2760, 2570, 2378, 2185, 1990, 1795, 1598, 1401, 1202, 1003,
        803, 603, 402, 201,
    ],
];
const SINPI: [[i32; 5]; 4] = [
    [0, 330, 621, 836, 951],
    [0, 660, 1241, 1672, 1901],
    [0, 1321, 2482, 3344, 3803],
    [0, 2642, 4964, 6689, 7606],
];

fn fdct4(input: &[i32], output: &mut [i32], cos_bit: i8) {
    let cospi = cospi_arr(cos_bit);
    let mut step = [0; 4];
    output[0] = input[0] + input[3];
    output[1] = input[1] + input[2];
    output[2] = -input[2] + input[1];
    output[3] = -input[3] + input[0];
    step[0] = half_btf(cospi[32], output[0], cospi[32], output[1], cos_bit);
    step[1] = half_btf(-cospi[32], output[1], cospi[32], output[0], cos_bit);
    step[2] = half_btf(cospi[48], output[2], cospi[16], output[3], cos_bit);
    step[3] = half_btf(cospi[48], output[3], -cospi[16], output[2], cos_bit);
    output[0] = step[0];
    output[1] = step[2];
    output[2] = step[1];
    output[3] = step[3];
}

fn fdct8(input: &[i32], output: &mut [i32], cos_bit: i8) {
    let cospi = cospi_arr(cos_bit);
    let mut step = [0; 8];
    output[0] = input[0] + input[7];
    output[1] = input[1] + input[6];
    output[2] = input[2] + input[5];
    output[3] = input[3] + input[4];
    output[4] = -input[4] + input[3];
    output[5] = -input[5] + input[2];
    output[6] = -input[6] + input[1];
    output[7] = -input[7] + input[0];
    step[0] = output[0] + output[3];
    step[1] = output[1] + output[2];
    step[2] = -output[2] + output[1];
    step[3] = -output[3] + output[0];
    step[4] = output[4];
    step[5] = half_btf(-cospi[32], output[5], cospi[32], output[6], cos_bit);
    step[6] = half_btf(cospi[32], output[6], cospi[32], output[5], cos_bit);
    step[7] = output[7];
    output[0] = half_btf(cospi[32], step[0], cospi[32], step[1], cos_bit);
    output[1] = half_btf(-cospi[32], step[1], cospi[32], step[0], cos_bit);
    output[2] = half_btf(cospi[48], step[2], cospi[16], step[3], cos_bit);
    output[3] = half_btf(cospi[48], step[3], -cospi[16], step[2], cos_bit);
    output[4] = step[4] + step[5];
    output[5] = -step[5] + step[4];
    output[6] = -step[6] + step[7];
    output[7] = step[7] + step[6];
    step[0] = output[0];
    step[1] = output[1];
    step[2] = output[2];
    step[3] = output[3];
    step[4] = half_btf(cospi[56], output[4], cospi[8], output[7], cos_bit);
    step[5] = half_btf(cospi[24], output[5], cospi[40], output[6], cos_bit);
    step[6] = half_btf(cospi[24], output[6], -cospi[40], output[5], cos_bit);
    step[7] = half_btf(cospi[56], output[7], -cospi[8], output[4], cos_bit);
    output[0] = step[0];
    output[1] = step[4];
    output[2] = step[2];
    output[3] = step[6];
    output[4] = step[1];
    output[5] = step[5];
    output[6] = step[3];
    output[7] = step[7];
}

fn fdct16(input: &[i32], output: &mut [i32], cos_bit: i8) {
    let cospi = cospi_arr(cos_bit);
    let mut step = [0; 16];
    output[0] = input[0] + input[15];
    output[1] = input[1] + input[14];
    output[2] = input[2] + input[13];
    output[3] = input[3] + input[12];
    output[4] = input[4] + input[11];
    output[5] = input[5] + input[10];
    output[6] = input[6] + input[9];
    output[7] = input[7] + input[8];
    output[8] = -input[8] + input[7];
    output[9] = -input[9] + input[6];
    output[10] = -input[10] + input[5];
    output[11] = -input[11] + input[4];
    output[12] = -input[12] + input[3];
    output[13] = -input[13] + input[2];
    output[14] = -input[14] + input[1];
    output[15] = -input[15] + input[0];
    step[0] = output[0] + output[7];
    step[1] = output[1] + output[6];
    step[2] = output[2] + output[5];
    step[3] = output[3] + output[4];
    step[4] = -output[4] + output[3];
    step[5] = -output[5] + output[2];
    step[6] = -output[6] + output[1];
    step[7] = -output[7] + output[0];
    step[8] = output[8];
    step[9] = output[9];
    step[10] = half_btf(-cospi[32], output[10], cospi[32], output[13], cos_bit);
    step[11] = half_btf(-cospi[32], output[11], cospi[32], output[12], cos_bit);
    step[12] = half_btf(cospi[32], output[12], cospi[32], output[11], cos_bit);
    step[13] = half_btf(cospi[32], output[13], cospi[32], output[10], cos_bit);
    step[14] = output[14];
    step[15] = output[15];
    output[0] = step[0] + step[3];
    output[1] = step[1] + step[2];
    output[2] = -step[2] + step[1];
    output[3] = -step[3] + step[0];
    output[4] = step[4];
    output[5] = half_btf(-cospi[32], step[5], cospi[32], step[6], cos_bit);
    output[6] = half_btf(cospi[32], step[6], cospi[32], step[5], cos_bit);
    output[7] = step[7];
    output[8] = step[8] + step[11];
    output[9] = step[9] + step[10];
    output[10] = -step[10] + step[9];
    output[11] = -step[11] + step[8];
    output[12] = -step[12] + step[15];
    output[13] = -step[13] + step[14];
    output[14] = step[14] + step[13];
    output[15] = step[15] + step[12];
    step[0] = half_btf(cospi[32], output[0], cospi[32], output[1], cos_bit);
    step[1] = half_btf(-cospi[32], output[1], cospi[32], output[0], cos_bit);
    step[2] = half_btf(cospi[48], output[2], cospi[16], output[3], cos_bit);
    step[3] = half_btf(cospi[48], output[3], -cospi[16], output[2], cos_bit);
    step[4] = output[4] + output[5];
    step[5] = -output[5] + output[4];
    step[6] = -output[6] + output[7];
    step[7] = output[7] + output[6];
    step[8] = output[8];
    step[9] = half_btf(-cospi[16], output[9], cospi[48], output[14], cos_bit);
    step[10] = half_btf(-cospi[48], output[10], -cospi[16], output[13], cos_bit);
    step[11] = output[11];
    step[12] = output[12];
    step[13] = half_btf(cospi[48], output[13], -cospi[16], output[10], cos_bit);
    step[14] = half_btf(cospi[16], output[14], cospi[48], output[9], cos_bit);
    step[15] = output[15];
    output[0] = step[0];
    output[1] = step[1];
    output[2] = step[2];
    output[3] = step[3];
    output[4] = half_btf(cospi[56], step[4], cospi[8], step[7], cos_bit);
    output[5] = half_btf(cospi[24], step[5], cospi[40], step[6], cos_bit);
    output[6] = half_btf(cospi[24], step[6], -cospi[40], step[5], cos_bit);
    output[7] = half_btf(cospi[56], step[7], -cospi[8], step[4], cos_bit);
    output[8] = step[8] + step[9];
    output[9] = -step[9] + step[8];
    output[10] = -step[10] + step[11];
    output[11] = step[11] + step[10];
    output[12] = step[12] + step[13];
    output[13] = -step[13] + step[12];
    output[14] = -step[14] + step[15];
    output[15] = step[15] + step[14];
    step[0] = output[0];
    step[1] = output[1];
    step[2] = output[2];
    step[3] = output[3];
    step[4] = output[4];
    step[5] = output[5];
    step[6] = output[6];
    step[7] = output[7];
    step[8] = half_btf(cospi[60], output[8], cospi[4], output[15], cos_bit);
    step[9] = half_btf(cospi[28], output[9], cospi[36], output[14], cos_bit);
    step[10] = half_btf(cospi[44], output[10], cospi[20], output[13], cos_bit);
    step[11] = half_btf(cospi[12], output[11], cospi[52], output[12], cos_bit);
    step[12] = half_btf(cospi[12], output[12], -cospi[52], output[11], cos_bit);
    step[13] = half_btf(cospi[44], output[13], -cospi[20], output[10], cos_bit);
    step[14] = half_btf(cospi[28], output[14], -cospi[36], output[9], cos_bit);
    step[15] = half_btf(cospi[60], output[15], -cospi[4], output[8], cos_bit);
    output[0] = step[0];
    output[1] = step[8];
    output[2] = step[4];
    output[3] = step[12];
    output[4] = step[2];
    output[5] = step[10];
    output[6] = step[6];
    output[7] = step[14];
    output[8] = step[1];
    output[9] = step[9];
    output[10] = step[5];
    output[11] = step[13];
    output[12] = step[3];
    output[13] = step[11];
    output[14] = step[7];
    output[15] = step[15];
}

fn fdct32(input: &[i32], output: &mut [i32], cos_bit: i8) {
    let cospi = cospi_arr(cos_bit);
    let mut step = [0; 32];
    output[0] = input[0] + input[31];
    output[1] = input[1] + input[30];
    output[2] = input[2] + input[29];
    output[3] = input[3] + input[28];
    output[4] = input[4] + input[27];
    output[5] = input[5] + input[26];
    output[6] = input[6] + input[25];
    output[7] = input[7] + input[24];
    output[8] = input[8] + input[23];
    output[9] = input[9] + input[22];
    output[10] = input[10] + input[21];
    output[11] = input[11] + input[20];
    output[12] = input[12] + input[19];
    output[13] = input[13] + input[18];
    output[14] = input[14] + input[17];
    output[15] = input[15] + input[16];
    output[16] = -input[16] + input[15];
    output[17] = -input[17] + input[14];
    output[18] = -input[18] + input[13];
    output[19] = -input[19] + input[12];
    output[20] = -input[20] + input[11];
    output[21] = -input[21] + input[10];
    output[22] = -input[22] + input[9];
    output[23] = -input[23] + input[8];
    output[24] = -input[24] + input[7];
    output[25] = -input[25] + input[6];
    output[26] = -input[26] + input[5];
    output[27] = -input[27] + input[4];
    output[28] = -input[28] + input[3];
    output[29] = -input[29] + input[2];
    output[30] = -input[30] + input[1];
    output[31] = -input[31] + input[0];
    step[0] = output[0] + output[15];
    step[1] = output[1] + output[14];
    step[2] = output[2] + output[13];
    step[3] = output[3] + output[12];
    step[4] = output[4] + output[11];
    step[5] = output[5] + output[10];
    step[6] = output[6] + output[9];
    step[7] = output[7] + output[8];
    step[8] = -output[8] + output[7];
    step[9] = -output[9] + output[6];
    step[10] = -output[10] + output[5];
    step[11] = -output[11] + output[4];
    step[12] = -output[12] + output[3];
    step[13] = -output[13] + output[2];
    step[14] = -output[14] + output[1];
    step[15] = -output[15] + output[0];
    step[16] = output[16];
    step[17] = output[17];
    step[18] = output[18];
    step[19] = output[19];
    step[20] = half_btf(-cospi[32], output[20], cospi[32], output[27], cos_bit);
    step[21] = half_btf(-cospi[32], output[21], cospi[32], output[26], cos_bit);
    step[22] = half_btf(-cospi[32], output[22], cospi[32], output[25], cos_bit);
    step[23] = half_btf(-cospi[32], output[23], cospi[32], output[24], cos_bit);
    step[24] = half_btf(cospi[32], output[24], cospi[32], output[23], cos_bit);
    step[25] = half_btf(cospi[32], output[25], cospi[32], output[22], cos_bit);
    step[26] = half_btf(cospi[32], output[26], cospi[32], output[21], cos_bit);
    step[27] = half_btf(cospi[32], output[27], cospi[32], output[20], cos_bit);
    step[28] = output[28];
    step[29] = output[29];
    step[30] = output[30];
    step[31] = output[31];
    output[0] = step[0] + step[7];
    output[1] = step[1] + step[6];
    output[2] = step[2] + step[5];
    output[3] = step[3] + step[4];
    output[4] = -step[4] + step[3];
    output[5] = -step[5] + step[2];
    output[6] = -step[6] + step[1];
    output[7] = -step[7] + step[0];
    output[8] = step[8];
    output[9] = step[9];
    output[10] = half_btf(-cospi[32], step[10], cospi[32], step[13], cos_bit);
    output[11] = half_btf(-cospi[32], step[11], cospi[32], step[12], cos_bit);
    output[12] = half_btf(cospi[32], step[12], cospi[32], step[11], cos_bit);
    output[13] = half_btf(cospi[32], step[13], cospi[32], step[10], cos_bit);
    output[14] = step[14];
    output[15] = step[15];
    output[16] = step[16] + step[23];
    output[17] = step[17] + step[22];
    output[18] = step[18] + step[21];
    output[19] = step[19] + step[20];
    output[20] = -step[20] + step[19];
    output[21] = -step[21] + step[18];
    output[22] = -step[22] + step[17];
    output[23] = -step[23] + step[16];
    output[24] = -step[24] + step[31];
    output[25] = -step[25] + step[30];
    output[26] = -step[26] + step[29];
    output[27] = -step[27] + step[28];
    output[28] = step[28] + step[27];
    output[29] = step[29] + step[26];
    output[30] = step[30] + step[25];
    output[31] = step[31] + step[24];
    step[0] = output[0] + output[3];
    step[1] = output[1] + output[2];
    step[2] = -output[2] + output[1];
    step[3] = -output[3] + output[0];
    step[4] = output[4];
    step[5] = half_btf(-cospi[32], output[5], cospi[32], output[6], cos_bit);
    step[6] = half_btf(cospi[32], output[6], cospi[32], output[5], cos_bit);
    step[7] = output[7];
    step[8] = output[8] + output[11];
    step[9] = output[9] + output[10];
    step[10] = -output[10] + output[9];
    step[11] = -output[11] + output[8];
    step[12] = -output[12] + output[15];
    step[13] = -output[13] + output[14];
    step[14] = output[14] + output[13];
    step[15] = output[15] + output[12];
    step[16] = output[16];
    step[17] = output[17];
    step[18] = half_btf(-cospi[16], output[18], cospi[48], output[29], cos_bit);
    step[19] = half_btf(-cospi[16], output[19], cospi[48], output[28], cos_bit);
    step[20] = half_btf(-cospi[48], output[20], -cospi[16], output[27], cos_bit);
    step[21] = half_btf(-cospi[48], output[21], -cospi[16], output[26], cos_bit);
    step[22] = output[22];
    step[23] = output[23];
    step[24] = output[24];
    step[25] = output[25];
    step[26] = half_btf(cospi[48], output[26], -cospi[16], output[21], cos_bit);
    step[27] = half_btf(cospi[48], output[27], -cospi[16], output[20], cos_bit);
    step[28] = half_btf(cospi[16], output[28], cospi[48], output[19], cos_bit);
    step[29] = half_btf(cospi[16], output[29], cospi[48], output[18], cos_bit);
    step[30] = output[30];
    step[31] = output[31];
    output[0] = half_btf(cospi[32], step[0], cospi[32], step[1], cos_bit);
    output[1] = half_btf(-cospi[32], step[1], cospi[32], step[0], cos_bit);
    output[2] = half_btf(cospi[48], step[2], cospi[16], step[3], cos_bit);
    output[3] = half_btf(cospi[48], step[3], -cospi[16], step[2], cos_bit);
    output[4] = step[4] + step[5];
    output[5] = -step[5] + step[4];
    output[6] = -step[6] + step[7];
    output[7] = step[7] + step[6];
    output[8] = step[8];
    output[9] = half_btf(-cospi[16], step[9], cospi[48], step[14], cos_bit);
    output[10] = half_btf(-cospi[48], step[10], -cospi[16], step[13], cos_bit);
    output[11] = step[11];
    output[12] = step[12];
    output[13] = half_btf(cospi[48], step[13], -cospi[16], step[10], cos_bit);
    output[14] = half_btf(cospi[16], step[14], cospi[48], step[9], cos_bit);
    output[15] = step[15];
    output[16] = step[16] + step[19];
    output[17] = step[17] + step[18];
    output[18] = -step[18] + step[17];
    output[19] = -step[19] + step[16];
    output[20] = -step[20] + step[23];
    output[21] = -step[21] + step[22];
    output[22] = step[22] + step[21];
    output[23] = step[23] + step[20];
    output[24] = step[24] + step[27];
    output[25] = step[25] + step[26];
    output[26] = -step[26] + step[25];
    output[27] = -step[27] + step[24];
    output[28] = -step[28] + step[31];
    output[29] = -step[29] + step[30];
    output[30] = step[30] + step[29];
    output[31] = step[31] + step[28];
    step[0] = output[0];
    step[1] = output[1];
    step[2] = output[2];
    step[3] = output[3];
    step[4] = half_btf(cospi[56], output[4], cospi[8], output[7], cos_bit);
    step[5] = half_btf(cospi[24], output[5], cospi[40], output[6], cos_bit);
    step[6] = half_btf(cospi[24], output[6], -cospi[40], output[5], cos_bit);
    step[7] = half_btf(cospi[56], output[7], -cospi[8], output[4], cos_bit);
    step[8] = output[8] + output[9];
    step[9] = -output[9] + output[8];
    step[10] = -output[10] + output[11];
    step[11] = output[11] + output[10];
    step[12] = output[12] + output[13];
    step[13] = -output[13] + output[12];
    step[14] = -output[14] + output[15];
    step[15] = output[15] + output[14];
    step[16] = output[16];
    step[17] = half_btf(-cospi[8], output[17], cospi[56], output[30], cos_bit);
    step[18] = half_btf(-cospi[56], output[18], -cospi[8], output[29], cos_bit);
    step[19] = output[19];
    step[20] = output[20];
    step[21] = half_btf(-cospi[40], output[21], cospi[24], output[26], cos_bit);
    step[22] = half_btf(-cospi[24], output[22], -cospi[40], output[25], cos_bit);
    step[23] = output[23];
    step[24] = output[24];
    step[25] = half_btf(cospi[24], output[25], -cospi[40], output[22], cos_bit);
    step[26] = half_btf(cospi[40], output[26], cospi[24], output[21], cos_bit);
    step[27] = output[27];
    step[28] = output[28];
    step[29] = half_btf(cospi[56], output[29], -cospi[8], output[18], cos_bit);
    step[30] = half_btf(cospi[8], output[30], cospi[56], output[17], cos_bit);
    step[31] = output[31];
    output[0] = step[0];
    output[1] = step[1];
    output[2] = step[2];
    output[3] = step[3];
    output[4] = step[4];
    output[5] = step[5];
    output[6] = step[6];
    output[7] = step[7];
    output[8] = half_btf(cospi[60], step[8], cospi[4], step[15], cos_bit);
    output[9] = half_btf(cospi[28], step[9], cospi[36], step[14], cos_bit);
    output[10] = half_btf(cospi[44], step[10], cospi[20], step[13], cos_bit);
    output[11] = half_btf(cospi[12], step[11], cospi[52], step[12], cos_bit);
    output[12] = half_btf(cospi[12], step[12], -cospi[52], step[11], cos_bit);
    output[13] = half_btf(cospi[44], step[13], -cospi[20], step[10], cos_bit);
    output[14] = half_btf(cospi[28], step[14], -cospi[36], step[9], cos_bit);
    output[15] = half_btf(cospi[60], step[15], -cospi[4], step[8], cos_bit);
    output[16] = step[16] + step[17];
    output[17] = -step[17] + step[16];
    output[18] = -step[18] + step[19];
    output[19] = step[19] + step[18];
    output[20] = step[20] + step[21];
    output[21] = -step[21] + step[20];
    output[22] = -step[22] + step[23];
    output[23] = step[23] + step[22];
    output[24] = step[24] + step[25];
    output[25] = -step[25] + step[24];
    output[26] = -step[26] + step[27];
    output[27] = step[27] + step[26];
    output[28] = step[28] + step[29];
    output[29] = -step[29] + step[28];
    output[30] = -step[30] + step[31];
    output[31] = step[31] + step[30];
    step[0] = output[0];
    step[1] = output[1];
    step[2] = output[2];
    step[3] = output[3];
    step[4] = output[4];
    step[5] = output[5];
    step[6] = output[6];
    step[7] = output[7];
    step[8] = output[8];
    step[9] = output[9];
    step[10] = output[10];
    step[11] = output[11];
    step[12] = output[12];
    step[13] = output[13];
    step[14] = output[14];
    step[15] = output[15];
    step[16] = half_btf(cospi[62], output[16], cospi[2], output[31], cos_bit);
    step[17] = half_btf(cospi[30], output[17], cospi[34], output[30], cos_bit);
    step[18] = half_btf(cospi[46], output[18], cospi[18], output[29], cos_bit);
    step[19] = half_btf(cospi[14], output[19], cospi[50], output[28], cos_bit);
    step[20] = half_btf(cospi[54], output[20], cospi[10], output[27], cos_bit);
    step[21] = half_btf(cospi[22], output[21], cospi[42], output[26], cos_bit);
    step[22] = half_btf(cospi[38], output[22], cospi[26], output[25], cos_bit);
    step[23] = half_btf(cospi[6], output[23], cospi[58], output[24], cos_bit);
    step[24] = half_btf(cospi[6], output[24], -cospi[58], output[23], cos_bit);
    step[25] = half_btf(cospi[38], output[25], -cospi[26], output[22], cos_bit);
    step[26] = half_btf(cospi[22], output[26], -cospi[42], output[21], cos_bit);
    step[27] = half_btf(cospi[54], output[27], -cospi[10], output[20], cos_bit);
    step[28] = half_btf(cospi[14], output[28], -cospi[50], output[19], cos_bit);
    step[29] = half_btf(cospi[46], output[29], -cospi[18], output[18], cos_bit);
    step[30] = half_btf(cospi[30], output[30], -cospi[34], output[17], cos_bit);
    step[31] = half_btf(cospi[62], output[31], -cospi[2], output[16], cos_bit);
    output[0] = step[0];
    output[1] = step[16];
    output[2] = step[8];
    output[3] = step[24];
    output[4] = step[4];
    output[5] = step[20];
    output[6] = step[12];
    output[7] = step[28];
    output[8] = step[2];
    output[9] = step[18];
    output[10] = step[10];
    output[11] = step[26];
    output[12] = step[6];
    output[13] = step[22];
    output[14] = step[14];
    output[15] = step[30];
    output[16] = step[1];
    output[17] = step[17];
    output[18] = step[9];
    output[19] = step[25];
    output[20] = step[5];
    output[21] = step[21];
    output[22] = step[13];
    output[23] = step[29];
    output[24] = step[3];
    output[25] = step[19];
    output[26] = step[11];
    output[27] = step[27];
    output[28] = step[7];
    output[29] = step[23];
    output[30] = step[15];
    output[31] = step[31];
}

fn fdct64(input: &[i32], output: &mut [i32], cos_bit: i8) {
    let cospi = cospi_arr(cos_bit);
    let mut step = [0; 64];
    output[0] = input[0] + input[63];
    output[1] = input[1] + input[62];
    output[2] = input[2] + input[61];
    output[3] = input[3] + input[60];
    output[4] = input[4] + input[59];
    output[5] = input[5] + input[58];
    output[6] = input[6] + input[57];
    output[7] = input[7] + input[56];
    output[8] = input[8] + input[55];
    output[9] = input[9] + input[54];
    output[10] = input[10] + input[53];
    output[11] = input[11] + input[52];
    output[12] = input[12] + input[51];
    output[13] = input[13] + input[50];
    output[14] = input[14] + input[49];
    output[15] = input[15] + input[48];
    output[16] = input[16] + input[47];
    output[17] = input[17] + input[46];
    output[18] = input[18] + input[45];
    output[19] = input[19] + input[44];
    output[20] = input[20] + input[43];
    output[21] = input[21] + input[42];
    output[22] = input[22] + input[41];
    output[23] = input[23] + input[40];
    output[24] = input[24] + input[39];
    output[25] = input[25] + input[38];
    output[26] = input[26] + input[37];
    output[27] = input[27] + input[36];
    output[28] = input[28] + input[35];
    output[29] = input[29] + input[34];
    output[30] = input[30] + input[33];
    output[31] = input[31] + input[32];
    output[32] = -input[32] + input[31];
    output[33] = -input[33] + input[30];
    output[34] = -input[34] + input[29];
    output[35] = -input[35] + input[28];
    output[36] = -input[36] + input[27];
    output[37] = -input[37] + input[26];
    output[38] = -input[38] + input[25];
    output[39] = -input[39] + input[24];
    output[40] = -input[40] + input[23];
    output[41] = -input[41] + input[22];
    output[42] = -input[42] + input[21];
    output[43] = -input[43] + input[20];
    output[44] = -input[44] + input[19];
    output[45] = -input[45] + input[18];
    output[46] = -input[46] + input[17];
    output[47] = -input[47] + input[16];
    output[48] = -input[48] + input[15];
    output[49] = -input[49] + input[14];
    output[50] = -input[50] + input[13];
    output[51] = -input[51] + input[12];
    output[52] = -input[52] + input[11];
    output[53] = -input[53] + input[10];
    output[54] = -input[54] + input[9];
    output[55] = -input[55] + input[8];
    output[56] = -input[56] + input[7];
    output[57] = -input[57] + input[6];
    output[58] = -input[58] + input[5];
    output[59] = -input[59] + input[4];
    output[60] = -input[60] + input[3];
    output[61] = -input[61] + input[2];
    output[62] = -input[62] + input[1];
    output[63] = -input[63] + input[0];
    step[0] = output[0] + output[31];
    step[1] = output[1] + output[30];
    step[2] = output[2] + output[29];
    step[3] = output[3] + output[28];
    step[4] = output[4] + output[27];
    step[5] = output[5] + output[26];
    step[6] = output[6] + output[25];
    step[7] = output[7] + output[24];
    step[8] = output[8] + output[23];
    step[9] = output[9] + output[22];
    step[10] = output[10] + output[21];
    step[11] = output[11] + output[20];
    step[12] = output[12] + output[19];
    step[13] = output[13] + output[18];
    step[14] = output[14] + output[17];
    step[15] = output[15] + output[16];
    step[16] = -output[16] + output[15];
    step[17] = -output[17] + output[14];
    step[18] = -output[18] + output[13];
    step[19] = -output[19] + output[12];
    step[20] = -output[20] + output[11];
    step[21] = -output[21] + output[10];
    step[22] = -output[22] + output[9];
    step[23] = -output[23] + output[8];
    step[24] = -output[24] + output[7];
    step[25] = -output[25] + output[6];
    step[26] = -output[26] + output[5];
    step[27] = -output[27] + output[4];
    step[28] = -output[28] + output[3];
    step[29] = -output[29] + output[2];
    step[30] = -output[30] + output[1];
    step[31] = -output[31] + output[0];
    step[32] = output[32];
    step[33] = output[33];
    step[34] = output[34];
    step[35] = output[35];
    step[36] = output[36];
    step[37] = output[37];
    step[38] = output[38];
    step[39] = output[39];
    step[40] = half_btf(-cospi[32], output[40], cospi[32], output[55], cos_bit);
    step[41] = half_btf(-cospi[32], output[41], cospi[32], output[54], cos_bit);
    step[42] = half_btf(-cospi[32], output[42], cospi[32], output[53], cos_bit);
    step[43] = half_btf(-cospi[32], output[43], cospi[32], output[52], cos_bit);
    step[44] = half_btf(-cospi[32], output[44], cospi[32], output[51], cos_bit);
    step[45] = half_btf(-cospi[32], output[45], cospi[32], output[50], cos_bit);
    step[46] = half_btf(-cospi[32], output[46], cospi[32], output[49], cos_bit);
    step[47] = half_btf(-cospi[32], output[47], cospi[32], output[48], cos_bit);
    step[48] = half_btf(cospi[32], output[48], cospi[32], output[47], cos_bit);
    step[49] = half_btf(cospi[32], output[49], cospi[32], output[46], cos_bit);
    step[50] = half_btf(cospi[32], output[50], cospi[32], output[45], cos_bit);
    step[51] = half_btf(cospi[32], output[51], cospi[32], output[44], cos_bit);
    step[52] = half_btf(cospi[32], output[52], cospi[32], output[43], cos_bit);
    step[53] = half_btf(cospi[32], output[53], cospi[32], output[42], cos_bit);
    step[54] = half_btf(cospi[32], output[54], cospi[32], output[41], cos_bit);
    step[55] = half_btf(cospi[32], output[55], cospi[32], output[40], cos_bit);
    step[56] = output[56];
    step[57] = output[57];
    step[58] = output[58];
    step[59] = output[59];
    step[60] = output[60];
    step[61] = output[61];
    step[62] = output[62];
    step[63] = output[63];
    output[0] = step[0] + step[15];
    output[1] = step[1] + step[14];
    output[2] = step[2] + step[13];
    output[3] = step[3] + step[12];
    output[4] = step[4] + step[11];
    output[5] = step[5] + step[10];
    output[6] = step[6] + step[9];
    output[7] = step[7] + step[8];
    output[8] = -step[8] + step[7];
    output[9] = -step[9] + step[6];
    output[10] = -step[10] + step[5];
    output[11] = -step[11] + step[4];
    output[12] = -step[12] + step[3];
    output[13] = -step[13] + step[2];
    output[14] = -step[14] + step[1];
    output[15] = -step[15] + step[0];
    output[16] = step[16];
    output[17] = step[17];
    output[18] = step[18];
    output[19] = step[19];
    output[20] = half_btf(-cospi[32], step[20], cospi[32], step[27], cos_bit);
    output[21] = half_btf(-cospi[32], step[21], cospi[32], step[26], cos_bit);
    output[22] = half_btf(-cospi[32], step[22], cospi[32], step[25], cos_bit);
    output[23] = half_btf(-cospi[32], step[23], cospi[32], step[24], cos_bit);
    output[24] = half_btf(cospi[32], step[24], cospi[32], step[23], cos_bit);
    output[25] = half_btf(cospi[32], step[25], cospi[32], step[22], cos_bit);
    output[26] = half_btf(cospi[32], step[26], cospi[32], step[21], cos_bit);
    output[27] = half_btf(cospi[32], step[27], cospi[32], step[20], cos_bit);
    output[28] = step[28];
    output[29] = step[29];
    output[30] = step[30];
    output[31] = step[31];
    output[32] = step[32] + step[47];
    output[33] = step[33] + step[46];
    output[34] = step[34] + step[45];
    output[35] = step[35] + step[44];
    output[36] = step[36] + step[43];
    output[37] = step[37] + step[42];
    output[38] = step[38] + step[41];
    output[39] = step[39] + step[40];
    output[40] = -step[40] + step[39];
    output[41] = -step[41] + step[38];
    output[42] = -step[42] + step[37];
    output[43] = -step[43] + step[36];
    output[44] = -step[44] + step[35];
    output[45] = -step[45] + step[34];
    output[46] = -step[46] + step[33];
    output[47] = -step[47] + step[32];
    output[48] = -step[48] + step[63];
    output[49] = -step[49] + step[62];
    output[50] = -step[50] + step[61];
    output[51] = -step[51] + step[60];
    output[52] = -step[52] + step[59];
    output[53] = -step[53] + step[58];
    output[54] = -step[54] + step[57];
    output[55] = -step[55] + step[56];
    output[56] = step[56] + step[55];
    output[57] = step[57] + step[54];
    output[58] = step[58] + step[53];
    output[59] = step[59] + step[52];
    output[60] = step[60] + step[51];
    output[61] = step[61] + step[50];
    output[62] = step[62] + step[49];
    output[63] = step[63] + step[48];
    step[0] = output[0] + output[7];
    step[1] = output[1] + output[6];
    step[2] = output[2] + output[5];
    step[3] = output[3] + output[4];
    step[4] = -output[4] + output[3];
    step[5] = -output[5] + output[2];
    step[6] = -output[6] + output[1];
    step[7] = -output[7] + output[0];
    step[8] = output[8];
    step[9] = output[9];
    step[10] = half_btf(-cospi[32], output[10], cospi[32], output[13], cos_bit);
    step[11] = half_btf(-cospi[32], output[11], cospi[32], output[12], cos_bit);
    step[12] = half_btf(cospi[32], output[12], cospi[32], output[11], cos_bit);
    step[13] = half_btf(cospi[32], output[13], cospi[32], output[10], cos_bit);
    step[14] = output[14];
    step[15] = output[15];
    step[16] = output[16] + output[23];
    step[17] = output[17] + output[22];
    step[18] = output[18] + output[21];
    step[19] = output[19] + output[20];
    step[20] = -output[20] + output[19];
    step[21] = -output[21] + output[18];
    step[22] = -output[22] + output[17];
    step[23] = -output[23] + output[16];
    step[24] = -output[24] + output[31];
    step[25] = -output[25] + output[30];
    step[26] = -output[26] + output[29];
    step[27] = -output[27] + output[28];
    step[28] = output[28] + output[27];
    step[29] = output[29] + output[26];
    step[30] = output[30] + output[25];
    step[31] = output[31] + output[24];
    step[32] = output[32];
    step[33] = output[33];
    step[34] = output[34];
    step[35] = output[35];
    step[36] = half_btf(-cospi[16], output[36], cospi[48], output[59], cos_bit);
    step[37] = half_btf(-cospi[16], output[37], cospi[48], output[58], cos_bit);
    step[38] = half_btf(-cospi[16], output[38], cospi[48], output[57], cos_bit);
    step[39] = half_btf(-cospi[16], output[39], cospi[48], output[56], cos_bit);
    step[40] = half_btf(-cospi[48], output[40], -cospi[16], output[55], cos_bit);
    step[41] = half_btf(-cospi[48], output[41], -cospi[16], output[54], cos_bit);
    step[42] = half_btf(-cospi[48], output[42], -cospi[16], output[53], cos_bit);
    step[43] = half_btf(-cospi[48], output[43], -cospi[16], output[52], cos_bit);
    step[44] = output[44];
    step[45] = output[45];
    step[46] = output[46];
    step[47] = output[47];
    step[48] = output[48];
    step[49] = output[49];
    step[50] = output[50];
    step[51] = output[51];
    step[52] = half_btf(cospi[48], output[52], -cospi[16], output[43], cos_bit);
    step[53] = half_btf(cospi[48], output[53], -cospi[16], output[42], cos_bit);
    step[54] = half_btf(cospi[48], output[54], -cospi[16], output[41], cos_bit);
    step[55] = half_btf(cospi[48], output[55], -cospi[16], output[40], cos_bit);
    step[56] = half_btf(cospi[16], output[56], cospi[48], output[39], cos_bit);
    step[57] = half_btf(cospi[16], output[57], cospi[48], output[38], cos_bit);
    step[58] = half_btf(cospi[16], output[58], cospi[48], output[37], cos_bit);
    step[59] = half_btf(cospi[16], output[59], cospi[48], output[36], cos_bit);
    step[60] = output[60];
    step[61] = output[61];
    step[62] = output[62];
    step[63] = output[63];
    output[0] = step[0] + step[3];
    output[1] = step[1] + step[2];
    output[2] = -step[2] + step[1];
    output[3] = -step[3] + step[0];
    output[4] = step[4];
    output[5] = half_btf(-cospi[32], step[5], cospi[32], step[6], cos_bit);
    output[6] = half_btf(cospi[32], step[6], cospi[32], step[5], cos_bit);
    output[7] = step[7];
    output[8] = step[8] + step[11];
    output[9] = step[9] + step[10];
    output[10] = -step[10] + step[9];
    output[11] = -step[11] + step[8];
    output[12] = -step[12] + step[15];
    output[13] = -step[13] + step[14];
    output[14] = step[14] + step[13];
    output[15] = step[15] + step[12];
    output[16] = step[16];
    output[17] = step[17];
    output[18] = half_btf(-cospi[16], step[18], cospi[48], step[29], cos_bit);
    output[19] = half_btf(-cospi[16], step[19], cospi[48], step[28], cos_bit);
    output[20] = half_btf(-cospi[48], step[20], -cospi[16], step[27], cos_bit);
    output[21] = half_btf(-cospi[48], step[21], -cospi[16], step[26], cos_bit);
    output[22] = step[22];
    output[23] = step[23];
    output[24] = step[24];
    output[25] = step[25];
    output[26] = half_btf(cospi[48], step[26], -cospi[16], step[21], cos_bit);
    output[27] = half_btf(cospi[48], step[27], -cospi[16], step[20], cos_bit);
    output[28] = half_btf(cospi[16], step[28], cospi[48], step[19], cos_bit);
    output[29] = half_btf(cospi[16], step[29], cospi[48], step[18], cos_bit);
    output[30] = step[30];
    output[31] = step[31];
    output[32] = step[32] + step[39];
    output[33] = step[33] + step[38];
    output[34] = step[34] + step[37];
    output[35] = step[35] + step[36];
    output[36] = -step[36] + step[35];
    output[37] = -step[37] + step[34];
    output[38] = -step[38] + step[33];
    output[39] = -step[39] + step[32];
    output[40] = -step[40] + step[47];
    output[41] = -step[41] + step[46];
    output[42] = -step[42] + step[45];
    output[43] = -step[43] + step[44];
    output[44] = step[44] + step[43];
    output[45] = step[45] + step[42];
    output[46] = step[46] + step[41];
    output[47] = step[47] + step[40];
    output[48] = step[48] + step[55];
    output[49] = step[49] + step[54];
    output[50] = step[50] + step[53];
    output[51] = step[51] + step[52];
    output[52] = -step[52] + step[51];
    output[53] = -step[53] + step[50];
    output[54] = -step[54] + step[49];
    output[55] = -step[55] + step[48];
    output[56] = -step[56] + step[63];
    output[57] = -step[57] + step[62];
    output[58] = -step[58] + step[61];
    output[59] = -step[59] + step[60];
    output[60] = step[60] + step[59];
    output[61] = step[61] + step[58];
    output[62] = step[62] + step[57];
    output[63] = step[63] + step[56];
    step[0] = half_btf(cospi[32], output[0], cospi[32], output[1], cos_bit);
    step[1] = half_btf(-cospi[32], output[1], cospi[32], output[0], cos_bit);
    step[2] = half_btf(cospi[48], output[2], cospi[16], output[3], cos_bit);
    step[3] = half_btf(cospi[48], output[3], -cospi[16], output[2], cos_bit);
    step[4] = output[4] + output[5];
    step[5] = -output[5] + output[4];
    step[6] = -output[6] + output[7];
    step[7] = output[7] + output[6];
    step[8] = output[8];
    step[9] = half_btf(-cospi[16], output[9], cospi[48], output[14], cos_bit);
    step[10] = half_btf(-cospi[48], output[10], -cospi[16], output[13], cos_bit);
    step[11] = output[11];
    step[12] = output[12];
    step[13] = half_btf(cospi[48], output[13], -cospi[16], output[10], cos_bit);
    step[14] = half_btf(cospi[16], output[14], cospi[48], output[9], cos_bit);
    step[15] = output[15];
    step[16] = output[16] + output[19];
    step[17] = output[17] + output[18];
    step[18] = -output[18] + output[17];
    step[19] = -output[19] + output[16];
    step[20] = -output[20] + output[23];
    step[21] = -output[21] + output[22];
    step[22] = output[22] + output[21];
    step[23] = output[23] + output[20];
    step[24] = output[24] + output[27];
    step[25] = output[25] + output[26];
    step[26] = -output[26] + output[25];
    step[27] = -output[27] + output[24];
    step[28] = -output[28] + output[31];
    step[29] = -output[29] + output[30];
    step[30] = output[30] + output[29];
    step[31] = output[31] + output[28];
    step[32] = output[32];
    step[33] = output[33];
    step[34] = half_btf(-cospi[8], output[34], cospi[56], output[61], cos_bit);
    step[35] = half_btf(-cospi[8], output[35], cospi[56], output[60], cos_bit);
    step[36] = half_btf(-cospi[56], output[36], -cospi[8], output[59], cos_bit);
    step[37] = half_btf(-cospi[56], output[37], -cospi[8], output[58], cos_bit);
    step[38] = output[38];
    step[39] = output[39];
    step[40] = output[40];
    step[41] = output[41];
    step[42] = half_btf(-cospi[40], output[42], cospi[24], output[53], cos_bit);
    step[43] = half_btf(-cospi[40], output[43], cospi[24], output[52], cos_bit);
    step[44] = half_btf(-cospi[24], output[44], -cospi[40], output[51], cos_bit);
    step[45] = half_btf(-cospi[24], output[45], -cospi[40], output[50], cos_bit);
    step[46] = output[46];
    step[47] = output[47];
    step[48] = output[48];
    step[49] = output[49];
    step[50] = half_btf(cospi[24], output[50], -cospi[40], output[45], cos_bit);
    step[51] = half_btf(cospi[24], output[51], -cospi[40], output[44], cos_bit);
    step[52] = half_btf(cospi[40], output[52], cospi[24], output[43], cos_bit);
    step[53] = half_btf(cospi[40], output[53], cospi[24], output[42], cos_bit);
    step[54] = output[54];
    step[55] = output[55];
    step[56] = output[56];
    step[57] = output[57];
    step[58] = half_btf(cospi[56], output[58], -cospi[8], output[37], cos_bit);
    step[59] = half_btf(cospi[56], output[59], -cospi[8], output[36], cos_bit);
    step[60] = half_btf(cospi[8], output[60], cospi[56], output[35], cos_bit);
    step[61] = half_btf(cospi[8], output[61], cospi[56], output[34], cos_bit);
    step[62] = output[62];
    step[63] = output[63];
    output[0] = step[0];
    output[1] = step[1];
    output[2] = step[2];
    output[3] = step[3];
    output[4] = half_btf(cospi[56], step[4], cospi[8], step[7], cos_bit);
    output[5] = half_btf(cospi[24], step[5], cospi[40], step[6], cos_bit);
    output[6] = half_btf(cospi[24], step[6], -cospi[40], step[5], cos_bit);
    output[7] = half_btf(cospi[56], step[7], -cospi[8], step[4], cos_bit);
    output[8] = step[8] + step[9];
    output[9] = -step[9] + step[8];
    output[10] = -step[10] + step[11];
    output[11] = step[11] + step[10];
    output[12] = step[12] + step[13];
    output[13] = -step[13] + step[12];
    output[14] = -step[14] + step[15];
    output[15] = step[15] + step[14];
    output[16] = step[16];
    output[17] = half_btf(-cospi[8], step[17], cospi[56], step[30], cos_bit);
    output[18] = half_btf(-cospi[56], step[18], -cospi[8], step[29], cos_bit);
    output[19] = step[19];
    output[20] = step[20];
    output[21] = half_btf(-cospi[40], step[21], cospi[24], step[26], cos_bit);
    output[22] = half_btf(-cospi[24], step[22], -cospi[40], step[25], cos_bit);
    output[23] = step[23];
    output[24] = step[24];
    output[25] = half_btf(cospi[24], step[25], -cospi[40], step[22], cos_bit);
    output[26] = half_btf(cospi[40], step[26], cospi[24], step[21], cos_bit);
    output[27] = step[27];
    output[28] = step[28];
    output[29] = half_btf(cospi[56], step[29], -cospi[8], step[18], cos_bit);
    output[30] = half_btf(cospi[8], step[30], cospi[56], step[17], cos_bit);
    output[31] = step[31];
    output[32] = step[32] + step[35];
    output[33] = step[33] + step[34];
    output[34] = -step[34] + step[33];
    output[35] = -step[35] + step[32];
    output[36] = -step[36] + step[39];
    output[37] = -step[37] + step[38];
    output[38] = step[38] + step[37];
    output[39] = step[39] + step[36];
    output[40] = step[40] + step[43];
    output[41] = step[41] + step[42];
    output[42] = -step[42] + step[41];
    output[43] = -step[43] + step[40];
    output[44] = -step[44] + step[47];
    output[45] = -step[45] + step[46];
    output[46] = step[46] + step[45];
    output[47] = step[47] + step[44];
    output[48] = step[48] + step[51];
    output[49] = step[49] + step[50];
    output[50] = -step[50] + step[49];
    output[51] = -step[51] + step[48];
    output[52] = -step[52] + step[55];
    output[53] = -step[53] + step[54];
    output[54] = step[54] + step[53];
    output[55] = step[55] + step[52];
    output[56] = step[56] + step[59];
    output[57] = step[57] + step[58];
    output[58] = -step[58] + step[57];
    output[59] = -step[59] + step[56];
    output[60] = -step[60] + step[63];
    output[61] = -step[61] + step[62];
    output[62] = step[62] + step[61];
    output[63] = step[63] + step[60];
    step[0] = output[0];
    step[1] = output[1];
    step[2] = output[2];
    step[3] = output[3];
    step[4] = output[4];
    step[5] = output[5];
    step[6] = output[6];
    step[7] = output[7];
    step[8] = half_btf(cospi[60], output[8], cospi[4], output[15], cos_bit);
    step[9] = half_btf(cospi[28], output[9], cospi[36], output[14], cos_bit);
    step[10] = half_btf(cospi[44], output[10], cospi[20], output[13], cos_bit);
    step[11] = half_btf(cospi[12], output[11], cospi[52], output[12], cos_bit);
    step[12] = half_btf(cospi[12], output[12], -cospi[52], output[11], cos_bit);
    step[13] = half_btf(cospi[44], output[13], -cospi[20], output[10], cos_bit);
    step[14] = half_btf(cospi[28], output[14], -cospi[36], output[9], cos_bit);
    step[15] = half_btf(cospi[60], output[15], -cospi[4], output[8], cos_bit);
    step[16] = output[16] + output[17];
    step[17] = -output[17] + output[16];
    step[18] = -output[18] + output[19];
    step[19] = output[19] + output[18];
    step[20] = output[20] + output[21];
    step[21] = -output[21] + output[20];
    step[22] = -output[22] + output[23];
    step[23] = output[23] + output[22];
    step[24] = output[24] + output[25];
    step[25] = -output[25] + output[24];
    step[26] = -output[26] + output[27];
    step[27] = output[27] + output[26];
    step[28] = output[28] + output[29];
    step[29] = -output[29] + output[28];
    step[30] = -output[30] + output[31];
    step[31] = output[31] + output[30];
    step[32] = output[32];
    step[33] = half_btf(-cospi[4], output[33], cospi[60], output[62], cos_bit);
    step[34] = half_btf(-cospi[60], output[34], -cospi[4], output[61], cos_bit);
    step[35] = output[35];
    step[36] = output[36];
    step[37] = half_btf(-cospi[36], output[37], cospi[28], output[58], cos_bit);
    step[38] = half_btf(-cospi[28], output[38], -cospi[36], output[57], cos_bit);
    step[39] = output[39];
    step[40] = output[40];
    step[41] = half_btf(-cospi[20], output[41], cospi[44], output[54], cos_bit);
    step[42] = half_btf(-cospi[44], output[42], -cospi[20], output[53], cos_bit);
    step[43] = output[43];
    step[44] = output[44];
    step[45] = half_btf(-cospi[52], output[45], cospi[12], output[50], cos_bit);
    step[46] = half_btf(-cospi[12], output[46], -cospi[52], output[49], cos_bit);
    step[47] = output[47];
    step[48] = output[48];
    step[49] = half_btf(cospi[12], output[49], -cospi[52], output[46], cos_bit);
    step[50] = half_btf(cospi[52], output[50], cospi[12], output[45], cos_bit);
    step[51] = output[51];
    step[52] = output[52];
    step[53] = half_btf(cospi[44], output[53], -cospi[20], output[42], cos_bit);
    step[54] = half_btf(cospi[20], output[54], cospi[44], output[41], cos_bit);
    step[55] = output[55];
    step[56] = output[56];
    step[57] = half_btf(cospi[28], output[57], -cospi[36], output[38], cos_bit);
    step[58] = half_btf(cospi[36], output[58], cospi[28], output[37], cos_bit);
    step[59] = output[59];
    step[60] = output[60];
    step[61] = half_btf(cospi[60], output[61], -cospi[4], output[34], cos_bit);
    step[62] = half_btf(cospi[4], output[62], cospi[60], output[33], cos_bit);
    step[63] = output[63];
    output[0] = step[0];
    output[1] = step[1];
    output[2] = step[2];
    output[3] = step[3];
    output[4] = step[4];
    output[5] = step[5];
    output[6] = step[6];
    output[7] = step[7];
    output[8] = step[8];
    output[9] = step[9];
    output[10] = step[10];
    output[11] = step[11];
    output[12] = step[12];
    output[13] = step[13];
    output[14] = step[14];
    output[15] = step[15];
    output[16] = half_btf(cospi[62], step[16], cospi[2], step[31], cos_bit);
    output[17] = half_btf(cospi[30], step[17], cospi[34], step[30], cos_bit);
    output[18] = half_btf(cospi[46], step[18], cospi[18], step[29], cos_bit);
    output[19] = half_btf(cospi[14], step[19], cospi[50], step[28], cos_bit);
    output[20] = half_btf(cospi[54], step[20], cospi[10], step[27], cos_bit);
    output[21] = half_btf(cospi[22], step[21], cospi[42], step[26], cos_bit);
    output[22] = half_btf(cospi[38], step[22], cospi[26], step[25], cos_bit);
    output[23] = half_btf(cospi[6], step[23], cospi[58], step[24], cos_bit);
    output[24] = half_btf(cospi[6], step[24], -cospi[58], step[23], cos_bit);
    output[25] = half_btf(cospi[38], step[25], -cospi[26], step[22], cos_bit);
    output[26] = half_btf(cospi[22], step[26], -cospi[42], step[21], cos_bit);
    output[27] = half_btf(cospi[54], step[27], -cospi[10], step[20], cos_bit);
    output[28] = half_btf(cospi[14], step[28], -cospi[50], step[19], cos_bit);
    output[29] = half_btf(cospi[46], step[29], -cospi[18], step[18], cos_bit);
    output[30] = half_btf(cospi[30], step[30], -cospi[34], step[17], cos_bit);
    output[31] = half_btf(cospi[62], step[31], -cospi[2], step[16], cos_bit);
    output[32] = step[32] + step[33];
    output[33] = -step[33] + step[32];
    output[34] = -step[34] + step[35];
    output[35] = step[35] + step[34];
    output[36] = step[36] + step[37];
    output[37] = -step[37] + step[36];
    output[38] = -step[38] + step[39];
    output[39] = step[39] + step[38];
    output[40] = step[40] + step[41];
    output[41] = -step[41] + step[40];
    output[42] = -step[42] + step[43];
    output[43] = step[43] + step[42];
    output[44] = step[44] + step[45];
    output[45] = -step[45] + step[44];
    output[46] = -step[46] + step[47];
    output[47] = step[47] + step[46];
    output[48] = step[48] + step[49];
    output[49] = -step[49] + step[48];
    output[50] = -step[50] + step[51];
    output[51] = step[51] + step[50];
    output[52] = step[52] + step[53];
    output[53] = -step[53] + step[52];
    output[54] = -step[54] + step[55];
    output[55] = step[55] + step[54];
    output[56] = step[56] + step[57];
    output[57] = -step[57] + step[56];
    output[58] = -step[58] + step[59];
    output[59] = step[59] + step[58];
    output[60] = step[60] + step[61];
    output[61] = -step[61] + step[60];
    output[62] = -step[62] + step[63];
    output[63] = step[63] + step[62];
    step[0] = output[0];
    step[1] = output[1];
    step[2] = output[2];
    step[3] = output[3];
    step[4] = output[4];
    step[5] = output[5];
    step[6] = output[6];
    step[7] = output[7];
    step[8] = output[8];
    step[9] = output[9];
    step[10] = output[10];
    step[11] = output[11];
    step[12] = output[12];
    step[13] = output[13];
    step[14] = output[14];
    step[15] = output[15];
    step[16] = output[16];
    step[17] = output[17];
    step[18] = output[18];
    step[19] = output[19];
    step[20] = output[20];
    step[21] = output[21];
    step[22] = output[22];
    step[23] = output[23];
    step[24] = output[24];
    step[25] = output[25];
    step[26] = output[26];
    step[27] = output[27];
    step[28] = output[28];
    step[29] = output[29];
    step[30] = output[30];
    step[31] = output[31];
    step[32] = half_btf(cospi[63], output[32], cospi[1], output[63], cos_bit);
    step[33] = half_btf(cospi[31], output[33], cospi[33], output[62], cos_bit);
    step[34] = half_btf(cospi[47], output[34], cospi[17], output[61], cos_bit);
    step[35] = half_btf(cospi[15], output[35], cospi[49], output[60], cos_bit);
    step[36] = half_btf(cospi[55], output[36], cospi[9], output[59], cos_bit);
    step[37] = half_btf(cospi[23], output[37], cospi[41], output[58], cos_bit);
    step[38] = half_btf(cospi[39], output[38], cospi[25], output[57], cos_bit);
    step[39] = half_btf(cospi[7], output[39], cospi[57], output[56], cos_bit);
    step[40] = half_btf(cospi[59], output[40], cospi[5], output[55], cos_bit);
    step[41] = half_btf(cospi[27], output[41], cospi[37], output[54], cos_bit);
    step[42] = half_btf(cospi[43], output[42], cospi[21], output[53], cos_bit);
    step[43] = half_btf(cospi[11], output[43], cospi[53], output[52], cos_bit);
    step[44] = half_btf(cospi[51], output[44], cospi[13], output[51], cos_bit);
    step[45] = half_btf(cospi[19], output[45], cospi[45], output[50], cos_bit);
    step[46] = half_btf(cospi[35], output[46], cospi[29], output[49], cos_bit);
    step[47] = half_btf(cospi[3], output[47], cospi[61], output[48], cos_bit);
    step[48] = half_btf(cospi[3], output[48], -cospi[61], output[47], cos_bit);
    step[49] = half_btf(cospi[35], output[49], -cospi[29], output[46], cos_bit);
    step[50] = half_btf(cospi[19], output[50], -cospi[45], output[45], cos_bit);
    step[51] = half_btf(cospi[51], output[51], -cospi[13], output[44], cos_bit);
    step[52] = half_btf(cospi[11], output[52], -cospi[53], output[43], cos_bit);
    step[53] = half_btf(cospi[43], output[53], -cospi[21], output[42], cos_bit);
    step[54] = half_btf(cospi[27], output[54], -cospi[37], output[41], cos_bit);
    step[55] = half_btf(cospi[59], output[55], -cospi[5], output[40], cos_bit);
    step[56] = half_btf(cospi[7], output[56], -cospi[57], output[39], cos_bit);
    step[57] = half_btf(cospi[39], output[57], -cospi[25], output[38], cos_bit);
    step[58] = half_btf(cospi[23], output[58], -cospi[41], output[37], cos_bit);
    step[59] = half_btf(cospi[55], output[59], -cospi[9], output[36], cos_bit);
    step[60] = half_btf(cospi[15], output[60], -cospi[49], output[35], cos_bit);
    step[61] = half_btf(cospi[47], output[61], -cospi[17], output[34], cos_bit);
    step[62] = half_btf(cospi[31], output[62], -cospi[33], output[33], cos_bit);
    step[63] = half_btf(cospi[63], output[63], -cospi[1], output[32], cos_bit);
    output[0] = step[0];
    output[1] = step[32];
    output[2] = step[16];
    output[3] = step[48];
    output[4] = step[8];
    output[5] = step[40];
    output[6] = step[24];
    output[7] = step[56];
    output[8] = step[4];
    output[9] = step[36];
    output[10] = step[20];
    output[11] = step[52];
    output[12] = step[12];
    output[13] = step[44];
    output[14] = step[28];
    output[15] = step[60];
    output[16] = step[2];
    output[17] = step[34];
    output[18] = step[18];
    output[19] = step[50];
    output[20] = step[10];
    output[21] = step[42];
    output[22] = step[26];
    output[23] = step[58];
    output[24] = step[6];
    output[25] = step[38];
    output[26] = step[22];
    output[27] = step[54];
    output[28] = step[14];
    output[29] = step[46];
    output[30] = step[30];
    output[31] = step[62];
    output[32] = step[1];
    output[33] = step[33];
    output[34] = step[17];
    output[35] = step[49];
    output[36] = step[9];
    output[37] = step[41];
    output[38] = step[25];
    output[39] = step[57];
    output[40] = step[5];
    output[41] = step[37];
    output[42] = step[21];
    output[43] = step[53];
    output[44] = step[13];
    output[45] = step[45];
    output[46] = step[29];
    output[47] = step[61];
    output[48] = step[3];
    output[49] = step[35];
    output[50] = step[19];
    output[51] = step[51];
    output[52] = step[11];
    output[53] = step[43];
    output[54] = step[27];
    output[55] = step[59];
    output[56] = step[7];
    output[57] = step[39];
    output[58] = step[23];
    output[59] = step[55];
    output[60] = step[15];
    output[61] = step[47];
    output[62] = step[31];
    output[63] = step[63];
}

fn fadst8(input: &[i32], output: &mut [i32], cos_bit: i8) {
    let cospi = cospi_arr(cos_bit);
    let mut step = [0; 8];
    output[0] = input[0];
    output[1] = -input[7];
    output[2] = -input[3];
    output[3] = input[4];
    output[4] = -input[1];
    output[5] = input[6];
    output[6] = input[2];
    output[7] = -input[5];
    step[0] = output[0];
    step[1] = output[1];
    step[2] = half_btf(cospi[32], output[2], cospi[32], output[3], cos_bit);
    step[3] = half_btf(cospi[32], output[2], -cospi[32], output[3], cos_bit);
    step[4] = output[4];
    step[5] = output[5];
    step[6] = half_btf(cospi[32], output[6], cospi[32], output[7], cos_bit);
    step[7] = half_btf(cospi[32], output[6], -cospi[32], output[7], cos_bit);
    output[0] = step[0] + step[2];
    output[1] = step[1] + step[3];
    output[2] = step[0] - step[2];
    output[3] = step[1] - step[3];
    output[4] = step[4] + step[6];
    output[5] = step[5] + step[7];
    output[6] = step[4] - step[6];
    output[7] = step[5] - step[7];
    step[0] = output[0];
    step[1] = output[1];
    step[2] = output[2];
    step[3] = output[3];
    step[4] = half_btf(cospi[16], output[4], cospi[48], output[5], cos_bit);
    step[5] = half_btf(cospi[48], output[4], -cospi[16], output[5], cos_bit);
    step[6] = half_btf(-cospi[48], output[6], cospi[16], output[7], cos_bit);
    step[7] = half_btf(cospi[16], output[6], cospi[48], output[7], cos_bit);
    output[0] = step[0] + step[4];
    output[1] = step[1] + step[5];
    output[2] = step[2] + step[6];
    output[3] = step[3] + step[7];
    output[4] = step[0] - step[4];
    output[5] = step[1] - step[5];
    output[6] = step[2] - step[6];
    output[7] = step[3] - step[7];
    step[0] = half_btf(cospi[4], output[0], cospi[60], output[1], cos_bit);
    step[1] = half_btf(cospi[60], output[0], -cospi[4], output[1], cos_bit);
    step[2] = half_btf(cospi[20], output[2], cospi[44], output[3], cos_bit);
    step[3] = half_btf(cospi[44], output[2], -cospi[20], output[3], cos_bit);
    step[4] = half_btf(cospi[36], output[4], cospi[28], output[5], cos_bit);
    step[5] = half_btf(cospi[28], output[4], -cospi[36], output[5], cos_bit);
    step[6] = half_btf(cospi[52], output[6], cospi[12], output[7], cos_bit);
    step[7] = half_btf(cospi[12], output[6], -cospi[52], output[7], cos_bit);
    output[0] = step[1];
    output[1] = step[6];
    output[2] = step[3];
    output[3] = step[4];
    output[4] = step[5];
    output[5] = step[2];
    output[6] = step[7];
    output[7] = step[0];
}

fn fadst16(input: &[i32], output: &mut [i32], cos_bit: i8) {
    let cospi = cospi_arr(cos_bit);
    let mut step = [0; 16];
    output[0] = input[0];
    output[1] = -input[15];
    output[2] = -input[7];
    output[3] = input[8];
    output[4] = -input[3];
    output[5] = input[12];
    output[6] = input[4];
    output[7] = -input[11];
    output[8] = -input[1];
    output[9] = input[14];
    output[10] = input[6];
    output[11] = -input[9];
    output[12] = input[2];
    output[13] = -input[13];
    output[14] = -input[5];
    output[15] = input[10];
    step[0] = output[0];
    step[1] = output[1];
    step[2] = half_btf(cospi[32], output[2], cospi[32], output[3], cos_bit);
    step[3] = half_btf(cospi[32], output[2], -cospi[32], output[3], cos_bit);
    step[4] = output[4];
    step[5] = output[5];
    step[6] = half_btf(cospi[32], output[6], cospi[32], output[7], cos_bit);
    step[7] = half_btf(cospi[32], output[6], -cospi[32], output[7], cos_bit);
    step[8] = output[8];
    step[9] = output[9];
    step[10] = half_btf(cospi[32], output[10], cospi[32], output[11], cos_bit);
    step[11] = half_btf(cospi[32], output[10], -cospi[32], output[11], cos_bit);
    step[12] = output[12];
    step[13] = output[13];
    step[14] = half_btf(cospi[32], output[14], cospi[32], output[15], cos_bit);
    step[15] = half_btf(cospi[32], output[14], -cospi[32], output[15], cos_bit);
    output[0] = step[0] + step[2];
    output[1] = step[1] + step[3];
    output[2] = step[0] - step[2];
    output[3] = step[1] - step[3];
    output[4] = step[4] + step[6];
    output[5] = step[5] + step[7];
    output[6] = step[4] - step[6];
    output[7] = step[5] - step[7];
    output[8] = step[8] + step[10];
    output[9] = step[9] + step[11];
    output[10] = step[8] - step[10];
    output[11] = step[9] - step[11];
    output[12] = step[12] + step[14];
    output[13] = step[13] + step[15];
    output[14] = step[12] - step[14];
    output[15] = step[13] - step[15];
    step[0] = output[0];
    step[1] = output[1];
    step[2] = output[2];
    step[3] = output[3];
    step[4] = half_btf(cospi[16], output[4], cospi[48], output[5], cos_bit);
    step[5] = half_btf(cospi[48], output[4], -cospi[16], output[5], cos_bit);
    step[6] = half_btf(-cospi[48], output[6], cospi[16], output[7], cos_bit);
    step[7] = half_btf(cospi[16], output[6], cospi[48], output[7], cos_bit);
    step[8] = output[8];
    step[9] = output[9];
    step[10] = output[10];
    step[11] = output[11];
    step[12] = half_btf(cospi[16], output[12], cospi[48], output[13], cos_bit);
    step[13] = half_btf(cospi[48], output[12], -cospi[16], output[13], cos_bit);
    step[14] = half_btf(-cospi[48], output[14], cospi[16], output[15], cos_bit);
    step[15] = half_btf(cospi[16], output[14], cospi[48], output[15], cos_bit);
    output[0] = step[0] + step[4];
    output[1] = step[1] + step[5];
    output[2] = step[2] + step[6];
    output[3] = step[3] + step[7];
    output[4] = step[0] - step[4];
    output[5] = step[1] - step[5];
    output[6] = step[2] - step[6];
    output[7] = step[3] - step[7];
    output[8] = step[8] + step[12];
    output[9] = step[9] + step[13];
    output[10] = step[10] + step[14];
    output[11] = step[11] + step[15];
    output[12] = step[8] - step[12];
    output[13] = step[9] - step[13];
    output[14] = step[10] - step[14];
    output[15] = step[11] - step[15];
    step[0] = output[0];
    step[1] = output[1];
    step[2] = output[2];
    step[3] = output[3];
    step[4] = output[4];
    step[5] = output[5];
    step[6] = output[6];
    step[7] = output[7];
    step[8] = half_btf(cospi[8], output[8], cospi[56], output[9], cos_bit);
    step[9] = half_btf(cospi[56], output[8], -cospi[8], output[9], cos_bit);
    step[10] = half_btf(cospi[40], output[10], cospi[24], output[11], cos_bit);
    step[11] = half_btf(cospi[24], output[10], -cospi[40], output[11], cos_bit);
    step[12] = half_btf(-cospi[56], output[12], cospi[8], output[13], cos_bit);
    step[13] = half_btf(cospi[8], output[12], cospi[56], output[13], cos_bit);
    step[14] = half_btf(-cospi[24], output[14], cospi[40], output[15], cos_bit);
    step[15] = half_btf(cospi[40], output[14], cospi[24], output[15], cos_bit);
    output[0] = step[0] + step[8];
    output[1] = step[1] + step[9];
    output[2] = step[2] + step[10];
    output[3] = step[3] + step[11];
    output[4] = step[4] + step[12];
    output[5] = step[5] + step[13];
    output[6] = step[6] + step[14];
    output[7] = step[7] + step[15];
    output[8] = step[0] - step[8];
    output[9] = step[1] - step[9];
    output[10] = step[2] - step[10];
    output[11] = step[3] - step[11];
    output[12] = step[4] - step[12];
    output[13] = step[5] - step[13];
    output[14] = step[6] - step[14];
    output[15] = step[7] - step[15];
    step[0] = half_btf(cospi[2], output[0], cospi[62], output[1], cos_bit);
    step[1] = half_btf(cospi[62], output[0], -cospi[2], output[1], cos_bit);
    step[2] = half_btf(cospi[10], output[2], cospi[54], output[3], cos_bit);
    step[3] = half_btf(cospi[54], output[2], -cospi[10], output[3], cos_bit);
    step[4] = half_btf(cospi[18], output[4], cospi[46], output[5], cos_bit);
    step[5] = half_btf(cospi[46], output[4], -cospi[18], output[5], cos_bit);
    step[6] = half_btf(cospi[26], output[6], cospi[38], output[7], cos_bit);
    step[7] = half_btf(cospi[38], output[6], -cospi[26], output[7], cos_bit);
    step[8] = half_btf(cospi[34], output[8], cospi[30], output[9], cos_bit);
    step[9] = half_btf(cospi[30], output[8], -cospi[34], output[9], cos_bit);
    step[10] = half_btf(cospi[42], output[10], cospi[22], output[11], cos_bit);
    step[11] = half_btf(cospi[22], output[10], -cospi[42], output[11], cos_bit);
    step[12] = half_btf(cospi[50], output[12], cospi[14], output[13], cos_bit);
    step[13] = half_btf(cospi[14], output[12], -cospi[50], output[13], cos_bit);
    step[14] = half_btf(cospi[58], output[14], cospi[6], output[15], cos_bit);
    step[15] = half_btf(cospi[6], output[14], -cospi[58], output[15], cos_bit);
    output[0] = step[1];
    output[1] = step[14];
    output[2] = step[3];
    output[3] = step[12];
    output[4] = step[5];
    output[5] = step[10];
    output[6] = step[7];
    output[7] = step[8];
    output[8] = step[9];
    output[9] = step[6];
    output[10] = step[11];
    output[11] = step[4];
    output[12] = step[13];
    output[13] = step[2];
    output[14] = step[15];
    output[15] = step[0];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::Align64;
    use crate::cpu::CpuFlags;
    use crate::include::common::bitdepth::{BitDepth, BitDepth8};
    use crate::include::dav1d::picture::Rav1dPictureDataComponent;
    use crate::itx::Rav1dInvTxfmDSPContext;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 32) as u32
        }

        fn residual(&mut self, max: i32) -> i16 {
            (self.next() % (2 * max as u32 + 1)) as i16 - max as i16
        }
    }

    fn sizes() -> impl Iterator<Item = TxfmSize> {
        (0..TxfmSize::COUNT).map(|i| TxfmSize::from_repr(i).unwrap())
    }

    fn decodable(size: TxfmSize, ty: TxfmType) -> bool {
        let (w, h) = dimensions(size);
        match ty {
            WHT_WHT => size == TxfmSize::S4x4,
            DCT_DCT => true,
            IDTX => w.max(h) <= 32,
            H_ADST | V_ADST | H_FLIPADST | V_FLIPADST => w * h <= 128,
            _ => w.max(h) <= 16,
        }
    }

    fn blocks(size: TxfmSize, seed: u64, random: usize, max: i32) -> Vec<Vec<i16>> {
        let (w, h) = dimensions(size);
        let mut rng = Rng(seed);
        let mut blocks = vec![
            vec![max as i16; w * h],
            vec![-max as i16; w * h],
            (0..w * h)
                .map(|i| if (i % w + i / w) % 2 == 0 { max } else { -max } as i16)
                .collect(),
        ];
        blocks.extend((0..random).map(|_| (0..w * h).map(|_| rng.residual(max)).collect()));
        blocks
    }

    fn reconstruct(
        itx: &Rav1dInvTxfmDSPContext,
        size: TxfmSize,
        ty: TxfmType,
        coefs: &[i32],
        eob: i32,
    ) -> Vec<i32> {
        let (w, h) = dimensions(size);
        let (cw, ch) = coded_dimensions(size);
        let mut cf = Align64([0i16; 32 * 32]);
        for (c, &v) in cf.0.iter_mut().zip(&coefs[..cw * ch]) {
            *c = i16::try_from(v).expect("coefficient beyond 16 bits");
        }
        let mut px = Align64([128u8; 64 * 64]);
        {
            let pic = Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(&mut px.0, w);
            itx.itxfm_add[size as usize][ty as usize].call::<BitDepth8>(
                pic.with_offset::<BitDepth8>(),
                &mut cf.0[..cw * ch],
                eob,
                BitDepth8::new(()),
            );
        }
        px.0[..w * h].iter().map(|&p| p as i32 - 128).collect()
    }

    fn eob(coefs: &[i32], size: TxfmSize, ty: TxfmType) -> i32 {
        last_nonzero(coefs, size, ty).map_or(0, |e| e as i32)
    }

    /// 64-point sides drop the upper half of the frequencies, so their blocks
    /// are first projected onto the coded half by a round trip.
    fn max_error(itx: &Rav1dInvTxfmDSPContext, size: TxfmSize, ty: TxfmType) -> i32 {
        let (w, h) = dimensions(size);
        let band_limited = w.max(h) == 64;
        let mut coefs = [0; 32 * 32];
        let mut worst = 0;
        let seed = 0x9e37_79b9_7f4a_7c15 ^ ((size as u64) << 8 | ty as u64);
        let max = if band_limited { 64 } else { 127 };
        for mut block in blocks(size, seed, 64, max)
            .into_iter()
            .chain((!band_limited).then(|| vec![-128; w * h]))
        {
            if band_limited {
                forward(&block, w, &mut coefs, size, ty);
                let recon = reconstruct(itx, size, ty, &coefs, eob(&coefs, size, ty));
                let (lo, hi) = (recon.iter().min().unwrap(), recon.iter().max().unwrap());
                assert!(
                    *lo > -128 && *hi < 127,
                    "clipped {size:?} {lo} {hi} {:?}",
                    &block[..8]
                );
                block = recon.iter().map(|&r| r as i16).collect();
            }
            forward(&block, w, &mut coefs, size, ty);
            let recon = reconstruct(itx, size, ty, &coefs, eob(&coefs, size, ty));
            for (&r, &b) in recon.iter().zip(&block[..w * h]) {
                worst = worst.max((r - b as i32).abs());
            }
        }
        worst
    }

    const ROUND_TRIP_BOUND: [i32; TxfmSize::COUNT] = [
        0, // 4x4
        1, // 8x8
        1, // 16x16
        2, // 32x32
        3, // 64x64
        1, // 4x8
        1, // 8x4
        1, // 8x16
        1, // 16x8
        2, // 16x32
        2, // 32x16
        4, // 32x64
        4, // 64x32
        1, // 4x16
        1, // 16x4
        1, // 8x32
        1, // 32x8
        2, // 16x64
        2, // 64x16
    ];

    /// Every kernel level of this CPU, since each has its own `eob` shortcuts.
    fn cpu_levels() -> Vec<CpuFlags> {
        let detected = CpuFlags::run_time_detect();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let levels = vec![
            detected,
            detected - CpuFlags::AVX512ICL,
            detected & (CpuFlags::SSE2 | CpuFlags::SSSE3 | CpuFlags::SSE41),
            CpuFlags::empty(),
        ];
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let levels = vec![detected, CpuFlags::empty()];
        levels
    }

    #[test]
    fn round_trip_through_rav1d() {
        let mut report = String::new();
        let mut failed = false;
        for flags in cpu_levels() {
            let itx = Rav1dInvTxfmDSPContext::new::<BitDepth8>(flags, 8);
            report += &format!("cpu flags {:#x}\n", flags.bits());
            for size in sizes() {
                for ty in (0..=WHT_WHT).filter(|&ty| decodable(size, ty)) {
                    assert!(supported(size, ty), "{size:?} {ty}");
                    let worst = max_error(&itx, size, ty);
                    let bound = if ty == WHT_WHT {
                        0
                    } else {
                        ROUND_TRIP_BOUND[size as usize]
                    };
                    report += &format!("{size:?} {ty}: {worst}\n");
                    failed |= worst > bound;
                }
            }
        }
        println!("{report}");
        assert!(!failed, "max error beyond the bound:\n{report}");
    }

    /// FNV-1a over the coefficients of 10 blocks per size and type, from
    /// libaom v3.15.1's C functions (`av1_fwd_txfm2d_*_c`, `av1_fwht4x4_c`) on
    /// the same inputs; the harness is in `doc/encoder-fdct-handoff.md`.
    const LIBAOM_DIGESTS: &[(usize, TxfmType, u64)] = &[
        (0, 0, 0x7edc02226aad7110),
        (0, 1, 0xce71e0626e514cc0),
        (0, 2, 0xeea384c5f72ba1f7),
        (0, 3, 0x2917544b980388bb),
        (0, 4, 0x0ac93e1ace9d8284),
        (0, 5, 0x5082b4d07065dab9),
        (0, 6, 0xd2c11877f5fb79ba),
        (0, 7, 0x5bb095d664d3d594),
        (0, 8, 0xd6549ed5d5843a29),
        (0, 9, 0x967b1eb996a9f98c),
        (0, 10, 0xd687aa26557bd3fe),
        (0, 11, 0x4f136bc0e6b9a11c),
        (0, 12, 0xf46cb7079afaab83),
        (0, 13, 0x8ed78913412aff43),
        (0, 14, 0x0b5806ebd624d1b0),
        (0, 15, 0x18de137e2ed082cc),
        (0, 16, 0x35c8bd6aa8b63ab0),
        (1, 0, 0x3e3c9d97bba86023),
        (1, 1, 0x7e84fe57516143df),
        (1, 2, 0xf3d90bd93d49b516),
        (1, 3, 0x9d7e4fa7b4e99f94),
        (1, 4, 0x0c7f0aac980e1a80),
        (1, 5, 0x8f26d8869185fa1f),
        (1, 6, 0x2f1c711f1dda02f2),
        (1, 7, 0x4e6b560ef9a0ef11),
        (1, 8, 0xa470a97b51517029),
        (1, 9, 0xfcaadc11aa5da25c),
        (1, 10, 0x3dfa9b11c8da0919),
        (1, 11, 0xb1b648986fb2bcb8),
        (1, 12, 0x27d0388dd9f1394b),
        (1, 13, 0x572093edc4c4496b),
        (1, 14, 0xf0fb80192b8115de),
        (1, 15, 0x52af97e27f373571),
        (2, 0, 0x2d6f9f5559e94a53),
        (2, 1, 0x6df1b3218dbdc3cf),
        (2, 2, 0xe0900f6d5e0e0660),
        (2, 3, 0x21d350a69304c88d),
        (2, 4, 0xbbedfd527c1a0557),
        (2, 5, 0xc92a93fe77b25d0d),
        (2, 6, 0x54a46197b3adcd44),
        (2, 7, 0x1d2df190d6b27f7f),
        (2, 8, 0x86fed3005c4b81c2),
        (2, 9, 0x67f0fe619a04f06d),
        (2, 10, 0xafa2bd63666f06d6),
        (2, 11, 0x02ff8f76902e9c7c),
        (2, 12, 0x14814515d6fd2f16),
        (2, 13, 0xb41a969ccc96dffc),
        (2, 14, 0xd8655f6fcb30a46c),
        (2, 15, 0xd82e590e147ce67b),
        (3, 0, 0xd526788a76509669),
        (3, 9, 0x35f1bcbe8dd702dd),
        (3, 10, 0xa95fcaae79626dd9),
        (3, 11, 0x97681afda5c72f0f),
        (4, 0, 0x1efd1ea5ae7a416c),
        (5, 0, 0x2802f414b7193c9b),
        (5, 1, 0x95b31c87fe73c8e3),
        (5, 2, 0xc8b53c86d4beb891),
        (5, 3, 0xe9d7328406cc9d6e),
        (5, 4, 0x7f4e16569f1991f3),
        (5, 5, 0xa11effce8ddc52e6),
        (5, 6, 0x4fece65d0e3e3c65),
        (5, 7, 0x22c3511262b9f031),
        (5, 8, 0xfd13cb6fea67d6e2),
        (5, 9, 0x96ec1c2e23742f15),
        (5, 10, 0xf17bb0f5bf3ddd87),
        (5, 11, 0x61f9a02011d0ada2),
        (5, 12, 0x99902c0b61d78d9f),
        (5, 13, 0x6958ad5cb28d6724),
        (5, 14, 0x7605c6140947dfb2),
        (5, 15, 0x8d57bac71d5c180b),
        (6, 0, 0xf460ba54cd4e061c),
        (6, 1, 0x95324d7f3dae564c),
        (6, 2, 0x55b3b09713a9770c),
        (6, 3, 0x212b4ab8a96093a1),
        (6, 4, 0x829fa0987a947ed8),
        (6, 5, 0xa9ed3c93f05fccf7),
        (6, 6, 0xa731f05760e64970),
        (6, 7, 0xdc8cf3d6fefb5133),
        (6, 8, 0xa4d3f0fa0b62f3c0),
        (6, 9, 0x3ba68606800394a6),
        (6, 10, 0x4f623ff35fe6d965),
        (6, 11, 0x2c22b312183b1908),
        (6, 12, 0x709363e052bc0df2),
        (6, 13, 0x7ac7035baa89afc8),
        (6, 14, 0x53548cb55a133803),
        (6, 15, 0xe7964923eb5487b0),
        (7, 0, 0x8909224a60682c14),
        (7, 1, 0xff5248d21ed60b2f),
        (7, 2, 0xda57208b0ec8874c),
        (7, 3, 0x60652da6951dddbc),
        (7, 4, 0xb6ed3eadd915d4f2),
        (7, 5, 0x1afc2faa4ca539f5),
        (7, 6, 0x4edd6a545dbad54d),
        (7, 7, 0xaa7dacc9d3411ac9),
        (7, 8, 0xe3e8eaa2a4c216ac),
        (7, 9, 0x13f0af5e822c776b),
        (7, 10, 0x196ccf8df58be00c),
        (7, 11, 0xc63d0591f2c82c02),
        (7, 12, 0x159022330924538e),
        (7, 13, 0x9cbf0b3f26da66a2),
        (7, 14, 0xa5974e55099789e6),
        (7, 15, 0x3045c1076fc7aa59),
        (8, 0, 0xa37c916492ac01f4),
        (8, 1, 0x4bbc933120ecab51),
        (8, 2, 0xa19d3078e576b0f9),
        (8, 3, 0xc746d48201a5bec1),
        (8, 4, 0x2dccff2a3eb22911),
        (8, 5, 0xf588896d18dba360),
        (8, 6, 0x3710967fcff2a833),
        (8, 7, 0xf0c9b15b61f151cf),
        (8, 8, 0x16163dae1c4b7ff5),
        (8, 9, 0xa664e41b20ddf0fb),
        (8, 10, 0xac326a04335d7a0e),
        (8, 11, 0xfc6396057356d1f8),
        (8, 12, 0xdc0fd4887a5a75db),
        (8, 13, 0xa81a30b36b320a52),
        (8, 14, 0xbb1427845f5159ad),
        (8, 15, 0x293188e7b9e80fe1),
        (9, 0, 0x18926cd36b782421),
        (9, 2, 0xe8afe55cc90985d8),
        (9, 5, 0xf79753c3d46b5b98),
        (9, 9, 0x9e1a5af53a7956f3),
        (9, 10, 0xf14ad41038e343f7),
        (9, 11, 0x36bffc0996a68c52),
        (9, 13, 0x3495257d38953559),
        (9, 15, 0x6819e2d67ea6ef53),
        (10, 0, 0xcdc0fb33e36b6b2c),
        (10, 1, 0xa4686e145478df62),
        (10, 4, 0xa44e2df2e99fe962),
        (10, 9, 0x5ec44667bc921212),
        (10, 10, 0xd80b1f5d469b6cc2),
        (10, 11, 0x67116079ec1c82f4),
        (10, 12, 0xdee7ca1dcefb14d6),
        (10, 14, 0xe0bc2d651f557ffd),
        (11, 0, 0x05c13d49e146ae07),
        (11, 10, 0xa5551283487a7156),
        (12, 0, 0x137dcb0e52690d2a),
        (12, 11, 0x8614f7334e4a2e90),
        (13, 0, 0x619e0b35761c085d),
        (13, 1, 0x9ffa942ae7ac9ebe),
        (13, 2, 0x74adffe46a768bad),
        (13, 3, 0x0b0b0b904b2a3430),
        (13, 4, 0x6b6eb58c5f0c01c3),
        (13, 5, 0x9e2aeef6e83c1821),
        (13, 6, 0x91d247a2a3b53bc6),
        (13, 7, 0xb94a193beda8ca7d),
        (13, 8, 0x55fc2bdfe05d9579),
        (13, 9, 0xef55a027c1d8e22c),
        (13, 10, 0xd9ac0ab0bbbc1321),
        (13, 11, 0x120f2e68bb34ad08),
        (13, 12, 0xf800059dc1a66b91),
        (13, 13, 0x961b58ff844d1da7),
        (13, 14, 0xb06a5f91bc3e2431),
        (13, 15, 0x8753ec10755f8185),
        (14, 0, 0x6699222cbde04374),
        (14, 1, 0xa061f7f385e16b54),
        (14, 2, 0xbc24536b0218d7d2),
        (14, 3, 0x5edf33a3f52f3979),
        (14, 4, 0x28ede38a96be0406),
        (14, 5, 0xcd749aaa36429c88),
        (14, 6, 0x92fd2c896bee327c),
        (14, 7, 0x395c9aaa10dc7dae),
        (14, 8, 0xa433558fb419f91a),
        (14, 9, 0xd5fa02cdd7004132),
        (14, 10, 0x50c9aaf309ec62b4),
        (14, 11, 0x7c89f07511e82326),
        (14, 12, 0xc8c69dfd9cfade08),
        (14, 13, 0x7473a3739ed8b289),
        (14, 14, 0x3bbe6fe23042e6e1),
        (14, 15, 0xd8e5631cb1c44327),
        (15, 0, 0x038a4616bb5faca4),
        (15, 2, 0x6980943f32a67193),
        (15, 5, 0x514411de87bf389b),
        (15, 9, 0x2d8e29783dee7490),
        (15, 10, 0x723d31339182c02d),
        (15, 11, 0xc55900402fde319c),
        (15, 13, 0xaa0f653f10b66055),
        (15, 15, 0xeef014ab2af9a770),
        (16, 0, 0x01c77dfc11605818),
        (16, 1, 0x472833f0669f4dfe),
        (16, 4, 0x8392222b7d2f72dd),
        (16, 9, 0xcbe6d5bbfcddd597),
        (16, 10, 0xaa7804d8d7b635e7),
        (16, 11, 0x9e4eafc8bad739a3),
        (16, 12, 0x8c819633e4ce6e70),
        (16, 14, 0x23f8b5475eb5f24f),
        (17, 0, 0x911dd250e30a14c5),
        (17, 2, 0xb45bf504641f268a),
        (17, 5, 0x346ca5c4fd405be5),
        (17, 10, 0x880c5f7f2db7c109),
        (18, 0, 0x2a52e732a0da5624),
        (18, 1, 0x84f3c82672411186),
        (18, 4, 0x0d9bce81e8128711),
        (18, 11, 0x1fb84cc4aa9a748a),
    ];

    fn digest(size: TxfmSize, ty: TxfmType) -> u64 {
        let (w, h) = dimensions(size);
        let (cw, ch) = coded_dimensions(size);
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ ((size as u64) << 8 | ty as u64));
        let mut coefs = [0; 32 * 32];
        let mut d = 0xcbf2_9ce4_8422_2325u64;
        for b in 0..10 {
            let block: Vec<i16> = (0..w * h)
                .map(|_| match b {
                    0 => 255,
                    1 => -255,
                    _ => rng.residual(255),
                })
                .collect();
            forward(&block, w, &mut coefs, size, ty);
            for &c in &coefs[..cw * ch] {
                for byte in c.to_le_bytes() {
                    d = (d ^ byte as u64).wrapping_mul(0x100_0000_01b3);
                }
            }
        }
        d
    }

    #[test]
    fn matches_libaom() {
        let mut expected = LIBAOM_DIGESTS.iter();
        let mut wrong = Vec::new();
        for size in sizes() {
            for ty in (0..=WHT_WHT).filter(|&ty| supported(size, ty)) {
                let &(s, t, d) = expected.next().expect("a digest per supported type");
                assert_eq!((s, t), (size as usize, ty));
                if digest(size, ty) != d {
                    wrong.push((size, ty));
                }
            }
        }
        assert!(expected.next().is_none());
        assert!(wrong.is_empty(), "differ from libaom: {wrong:?}");
    }

    /// Coefficients cut after scan position `k`: the kernels' `eob` shortcuts
    /// must give what the full transform gives.
    #[test]
    fn sparse_blocks_take_the_shortcuts() {
        for flags in cpu_levels() {
            let itx = Rav1dInvTxfmDSPContext::new::<BitDepth8>(flags, 8);
            for size in sizes() {
                let (w, _) = dimensions(size);
                let (cw, ch) = coded_dimensions(size);
                let n = cw * ch;
                for ty in (0..WHT_WHT).filter(|&ty| decodable(size, ty)) {
                    let seed = 0x2545_f491_4f6c_dd1d ^ ((size as u64) << 8 | ty as u64);
                    let block = &blocks(size, seed, 1, 127)[3];
                    let mut full = [0; 32 * 32];
                    forward(block, w, &mut full, size, ty);
                    for k in [0, 1, 2, 5, 9, 15, 35, 63, 99, 135, 255, 299, 511, n - 1] {
                        if k >= n {
                            continue;
                        }
                        let mut coefs = full;
                        for i in k + 1..n {
                            coefs[scan_rc(size, ty, i)] = 0;
                        }
                        let fast = reconstruct(&itx, size, ty, &coefs, eob(&coefs, size, ty));
                        let slow = reconstruct(&itx, size, ty, &coefs, n as i32 - 1);
                        assert!(
                            fast == slow,
                            "{size:?} {ty} k {k} flags {:#x}",
                            flags.bits()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn last_nonzero_follows_scan() {
        let mut coefs = [0; 32 * 32];
        assert_eq!(last_nonzero(&coefs, TxfmSize::S8x8, DCT_DCT), None);
        coefs[0] = 1;
        assert_eq!(last_nonzero(&coefs, TxfmSize::S8x8, DCT_DCT), Some(0));
        coefs[8] = 1;
        assert_eq!(last_nonzero(&coefs, TxfmSize::S8x8, DCT_DCT), Some(1));
        coefs[1] = 1;
        assert_eq!(last_nonzero(&coefs, TxfmSize::S8x8, DCT_DCT), Some(2));
        assert_eq!(last_nonzero(&coefs, TxfmSize::S8x8, H_DCT), Some(8));
        assert_eq!(last_nonzero(&coefs, TxfmSize::S8x8, V_DCT), Some(8));
        coefs[8 * 3 + 1] = 1;
        assert_eq!(last_nonzero(&coefs, TxfmSize::R16x8, V_DCT), Some(16 + 3));
    }
}
