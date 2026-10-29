//! The inverse transforms eight `i16` lanes at a time for 8 bpc, laid out as dav1d's asm has
//! them: the coefficients are stored column by column, so the first pass takes each column as one
//! load and runs the row transform with the rows in the lanes, a transpose turns the rows into
//! registers, and the second pass adds its rows to the picture. Butterflies saturate to `i16`
//! where the C clips, rotations run in 32 bits, and the rounding shifts are Q15 multiplies, so
//! the adst's negated outputs cost nothing: their multiplier is negated instead.

use zerocopy::AsBytes;

use crate::include::common::bitdepth::BitDepth;
use crate::include::dav1d::picture::Rav1dPictureDataComponentOffset;
use crate::levels::{
    TxfmType, ADST_ADST, ADST_DCT, ADST_FLIPADST, DCT_ADST, DCT_DCT, DCT_FLIPADST, FLIPADST_ADST,
    FLIPADST_DCT, FLIPADST_FLIPADST, H_ADST, H_DCT, H_FLIPADST, IDTX, V_ADST, V_DCT, V_FLIPADST,
};
use crate::simd128::I16x8;
use crate::strided::Strided as _;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tx1d {
    Dct,
    Adst,
    FlipAdst,
    Identity,
}
use Tx1d::*;

/// The row and the column transform of a 2-D type; the C calls them (first, second).
const fn kinds(t: TxfmType) -> (Tx1d, Tx1d) {
    match t {
        IDTX => (Identity, Identity),
        DCT_DCT => (Dct, Dct),
        ADST_DCT => (Dct, Adst),
        FLIPADST_DCT => (Dct, FlipAdst),
        H_DCT => (Dct, Identity),
        DCT_ADST => (Adst, Dct),
        ADST_ADST => (Adst, Adst),
        FLIPADST_ADST => (Adst, FlipAdst),
        DCT_FLIPADST => (FlipAdst, Dct),
        ADST_FLIPADST => (FlipAdst, Adst),
        FLIPADST_FLIPADST => (FlipAdst, FlipAdst),
        V_DCT => (Identity, Dct),
        H_ADST => (Adst, Identity),
        H_FLIPADST => (FlipAdst, Identity),
        V_ADST => (Identity, Adst),
        V_FLIPADST => (Identity, FlipAdst),
        _ => unreachable!(),
    }
}

/// The outputs a transform leaves negated: the adst's odd ones, so the flipped adst's even ones.
const fn negated(t: Tx1d) -> u32 {
    match t {
        Adst => 0xaaaa,
        FlipAdst => 0x5555,
        _ => 0,
    }
}

/// The rounding shift of a pass between the two transforms (the rows) or into the picture (the
/// columns, always 4).
const fn row_shift(w: usize, h: usize) -> u32 {
    match (w, h) {
        (4, 4) | (4, 8) | (8, 4) => 0,
        (8, 8)
        | (4, 16)
        | (8, 16)
        | (16, 4)
        | (16, 8)
        | (16, 32)
        | (32, 16)
        | (32, 64)
        | (64, 32) => 1,
        _ => 2,
    }
}

/// The Q15 multiplier of a pass's rounding shift after transform `t` of `n` points; the identity
/// transforms scale in their own way, and 32768 means nothing is left to do.
const fn round_mul(t: Tx1d, n: usize, shift: u32) -> i32 {
    match (t, n) {
        (Identity, 8) => 1 << (16 - shift),
        (Identity, 16) if shift < 4 => 32768,
        _ => 1 << (15 - shift),
    }
}

#[inline(always)]
fn round<const N: usize>(v: &mut [I16x8; N], k: i32, negated: u32) {
    if k == 32768 {
        return;
    }
    let (pos, neg) = (I16x8::splat(k as i16), I16x8::splat(-k as i16));
    for i in 0..N {
        v[i] = v[i].mulhrs(if negated >> i & 1 != 0 { neg } else { pos });
    }
}

#[inline(always)]
fn dct8(v: &mut [I16x8; 8]) {
    let (t1, t0) = v[0].rot(v[4], 2896, 2896);
    let (t2, t3) = v[2].rot(v[6], 1567, 3784);
    let d0 = t0.add_sat(t3);
    let d1 = t1.add_sat(t2);
    let d2 = t1.sub_sat(t2);
    let d3 = t0.sub_sat(t3);

    let (t4a, t7a) = v[1].rot(v[7], 799, 4017);
    let (t5a, t6a) = v[5].rot(v[3], 3406, 2276);
    let t4 = t4a.add_sat(t5a);
    let t5a = t4a.sub_sat(t5a);
    let t7 = t7a.add_sat(t6a);
    let t6a = t7a.sub_sat(t6a);
    let (t5, t6) = t6a.rot(t5a, 2896, 2896);

    v[0] = d0.add_sat(t7);
    v[1] = d1.add_sat(t6);
    v[2] = d2.add_sat(t5);
    v[3] = d3.add_sat(t4);
    v[4] = d3.sub_sat(t4);
    v[5] = d2.sub_sat(t5);
    v[6] = d1.sub_sat(t6);
    v[7] = d0.sub_sat(t7);
}

#[inline(always)]
fn dct16(v: &mut [I16x8; 16]) {
    let mut e = [v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14]];
    dct8(&mut e);

    let (t8a, t15a) = v[1].rot(v[15], 401, 4076);
    let (t9a, t14a) = v[9].rot(v[7], 3166, 2598);
    let (t10a, t13a) = v[5].rot(v[11], 1931, 3612);
    let (t11a, t12a) = v[13].rot(v[3], 3920, 1189);

    let t8 = t8a.add_sat(t9a);
    let t9 = t8a.sub_sat(t9a);
    let t10 = t11a.sub_sat(t10a);
    let t11 = t11a.add_sat(t10a);
    let t12 = t12a.add_sat(t13a);
    let t13 = t12a.sub_sat(t13a);
    let t14 = t15a.sub_sat(t14a);
    let t15 = t15a.add_sat(t14a);

    let (t9a, t14a) = t14.rot(t9, 1567, 3784);
    let (t10a, t13a) = t10.madd2(t13, [-1567, -3784], [-3784, 1567]);

    let t8a = t8.add_sat(t11);
    let t9 = t9a.add_sat(t10a);
    let t10 = t9a.sub_sat(t10a);
    let t11a = t8.sub_sat(t11);
    let t12a = t15.sub_sat(t12);
    let t13 = t14a.sub_sat(t13a);
    let t14 = t14a.add_sat(t13a);
    let t15a = t15.add_sat(t12);

    let (t10a, t13a) = t13.rot(t10, 2896, 2896);
    let (t11, t12) = t12a.rot(t11a, 2896, 2896);

    let o = [t15a, t14, t13a, t12, t11, t10a, t9, t8a];
    for i in 0..8 {
        v[i] = e[i].add_sat(o[i]);
        v[15 - i] = e[i].sub_sat(o[i]);
    }
}

/// Outputs 1, 3, 5 and 7 come out negated. The last stage's `* 2896 >> 12` runs in 32 bits on
/// the row pass and in 16 on the column pass, where the asm has the inputs small enough.
#[inline(always)]
fn adst8<const COLS: bool>(v: &mut [I16x8; 8]) {
    let (t1a, t0a) = v[7].rot(v[0], 401, 4076);
    let (t3a, t2a) = v[5].rot(v[2], 1931, 3612);
    let (t5a, t4a) = v[3].rot(v[4], 3166, 2598);
    let (t7a, t6a) = v[1].rot(v[6], 3920, 1189);

    let t0 = t0a.add_sat(t4a);
    let t1 = t1a.add_sat(t5a);
    let t2 = t2a.add_sat(t6a);
    let t3 = t3a.add_sat(t7a);
    let t4 = t0a.sub_sat(t4a);
    let t5 = t1a.sub_sat(t5a);
    let t6 = t2a.sub_sat(t6a);
    let t7 = t3a.sub_sat(t7a);

    let (t5a, t4a) = t4.rot(t5, 1567, 3784);
    let (t6a, t7a) = t7.rot(t6, 3784, 1567);

    v[0] = t0.add_sat(t2);
    v[7] = t1.add_sat(t3);
    let t2 = t0.sub_sat(t2);
    let t3 = t1.sub_sat(t3);
    v[1] = t4a.add_sat(t6a);
    v[6] = t5a.add_sat(t7a);
    let t6 = t4a.sub_sat(t6a);
    let t7 = t5a.sub_sat(t7a);

    (v[4], v[3]) = half_rot::<COLS>(t2, t3);
    (v[5], v[2]) = half_rot::<COLS>(t6, t7);
}

/// `((a - b) * 2896 >> 12, (a + b) * 2896 >> 12)`: the adst's last stage.
#[inline(always)]
fn half_rot<const COLS: bool>(a: I16x8, b: I16x8) -> (I16x8, I16x8) {
    if COLS {
        let k = I16x8::splat(2896 * 8);
        (a.sub_sat(b).mulhrs(k), a.add_sat(b).mulhrs(k))
    } else {
        a.rot(b, 2896, 2896)
    }
}

/// The odd outputs come out negated, as in [`adst8`].
#[inline(always)]
fn adst16<const COLS: bool>(v: &mut [I16x8; 16]) {
    let (t1, t0) = v[15].rot(v[0], 201, 4091);
    let (t3, t2) = v[13].rot(v[2], 995, 3973);
    let (t5, t4) = v[11].rot(v[4], 1751, 3703);
    let (t7, t6) = v[9].rot(v[6], 2440, 3290);
    let (t9, t8) = v[7].rot(v[8], 3035, 2751);
    let (t11, t10) = v[5].rot(v[10], 3513, 2106);
    let (t13, t12) = v[3].rot(v[12], 3857, 1380);
    let (t15, t14) = v[1].rot(v[14], 4052, 601);

    let t0a = t0.add_sat(t8);
    let t1a = t1.add_sat(t9);
    let t2a = t2.add_sat(t10);
    let t3a = t3.add_sat(t11);
    let t4a = t4.add_sat(t12);
    let t5a = t5.add_sat(t13);
    let t6a = t6.add_sat(t14);
    let t7a = t7.add_sat(t15);
    let t8a = t0.sub_sat(t8);
    let t9a = t1.sub_sat(t9);
    let t10a = t2.sub_sat(t10);
    let t11a = t3.sub_sat(t11);
    let t12a = t4.sub_sat(t12);
    let t13a = t5.sub_sat(t13);
    let t14a = t6.sub_sat(t14);
    let t15a = t7.sub_sat(t15);

    let (t9, t8) = t8a.rot(t9a, 799, 4017);
    let (t11, t10) = t10a.rot(t11a, 3406, 2276);
    let (t12, t13) = t13a.rot(t12a, 4017, 799);
    let (t14, t15) = t15a.rot(t14a, 2276, 3406);

    let t0 = t0a.add_sat(t4a);
    let t1 = t1a.add_sat(t5a);
    let t2 = t2a.add_sat(t6a);
    let t3 = t3a.add_sat(t7a);
    let t4 = t0a.sub_sat(t4a);
    let t5 = t1a.sub_sat(t5a);
    let t6 = t2a.sub_sat(t6a);
    let t7 = t3a.sub_sat(t7a);
    let t8a = t8.add_sat(t12);
    let t9a = t9.add_sat(t13);
    let t10a = t10.add_sat(t14);
    let t11a = t11.add_sat(t15);
    let t12a = t8.sub_sat(t12);
    let t13a = t9.sub_sat(t13);
    let t14a = t10.sub_sat(t14);
    let t15a = t11.sub_sat(t15);

    let (t5a, t4a) = t4.rot(t5, 1567, 3784);
    let (t6a, t7a) = t7.rot(t6, 3784, 1567);
    let (t13, t12) = t12a.rot(t13a, 1567, 3784);
    let (t14, t15) = t15a.rot(t14a, 3784, 1567);

    v[0] = t0.add_sat(t2);
    v[15] = t1.add_sat(t3);
    let t2a = t0.sub_sat(t2);
    let t3a = t1.sub_sat(t3);
    v[3] = t4a.add_sat(t6a);
    v[12] = t5a.add_sat(t7a);
    let t6 = t4a.sub_sat(t6a);
    let t7 = t5a.sub_sat(t7a);
    v[1] = t8a.add_sat(t10a);
    v[14] = t9a.add_sat(t11a);
    let t10 = t8a.sub_sat(t10a);
    let t11 = t9a.sub_sat(t11a);
    v[2] = t12.add_sat(t14);
    v[13] = t13.add_sat(t15);
    let t14a = t12.sub_sat(t14);
    let t15a = t13.sub_sat(t15);

    (v[8], v[7]) = half_rot::<COLS>(t2a, t3a);
    (v[11], v[4]) = half_rot::<COLS>(t6, t7);
    (v[9], v[6]) = half_rot::<COLS>(t10, t11);
    (v[10], v[5]) = half_rot::<COLS>(t14a, t15a);
}

/// `2 * x + (x * 1697 + 1024 >> 11)`, with the pass's rounding shift folded in before the
/// picture: by 2 as an average of `x` and the half term (the same sign, so the unsigned average
/// is exact), by 1 as `x + (term + 1 >> 1)`.
#[inline(always)]
fn identity16(v: &mut [I16x8; 16], shift: u32) {
    let k = I16x8::splat(1697 * 16);
    for x in v {
        let term = x.mulhrs(k);
        *x = match shift {
            1 => x.add_sat(term.mulhrs(I16x8::splat(16384))),
            2 => x.avg_u(term.shr_s::<1>()),
            _ => x.add_sat(*x).add_sat(term),
        };
    }
}

#[inline(always)]
fn tx8<const COLS: bool>(v: &mut [I16x8; 8], t: Tx1d) {
    match t {
        Dct => dct8(v),
        Adst => adst8::<COLS>(v),
        FlipAdst => {
            adst8::<COLS>(v);
            v.reverse();
        }
        Identity => {}
    }
}

#[inline(always)]
fn tx16<const COLS: bool>(v: &mut [I16x8; 16], t: Tx1d, shift: u32) {
    match t {
        Dct => dct16(v),
        Adst => adst16::<COLS>(v),
        FlipAdst => {
            adst16::<COLS>(v);
            v.reverse();
        }
        Identity => identity16(v, shift),
    }
}

#[inline(always)]
fn tx<const N: usize, const COLS: bool>(v: &mut [I16x8; N], t: Tx1d, shift: u32) {
    match N {
        8 => tx8::<COLS>((&mut v[..8]).try_into().unwrap(), t),
        16 => tx16::<COLS>((&mut v[..16]).try_into().unwrap(), t, shift),
        _ => unreachable!(),
    }
}

/// `v[y]` added to the block's row `y` from column `x0`, saturated to the pixel range.
#[inline(always)]
fn add_rows<BD: BitDepth, const H: usize>(
    dst: Rav1dPictureDataComponentOffset,
    x0: usize,
    v: &[I16x8; H],
) {
    for y in 0..H {
        let row = dst + (y as isize * dst.pixel_stride::<BD>());
        let px = &mut *row.slice_mut::<BD>(x0 + 8);
        let px = &mut px.as_bytes_mut()[x0..];
        let out = I16x8::from_u8x8(px).add(v[y]);
        px.copy_from_slice(&out.pack_u8());
    }
}

/// `eob == 0` of a DCT: one value over the block.
#[inline(always)]
fn add_dc<BD: BitDepth, const W: usize, const H: usize>(
    dst: Rav1dPictureDataComponentOffset,
    coeff: &mut [i16],
) {
    let shift = row_shift(W, H);
    let mut dc = coeff[0] as i32;
    coeff[0] = 0;
    if W * 2 == H || H * 2 == W {
        dc = dc * 181 + 128 >> 8;
    }
    dc = dc * 181 + 128 >> 8;
    dc = dc + (1 << shift >> 1) >> shift;
    dc = dc * 181 + 128 + 2048 >> 12;
    let dc = I16x8::splat(dc as i16);
    for y in 0..H {
        let row = dst + (y as isize * dst.pixel_stride::<BD>());
        let px = &mut *row.slice_mut::<BD>(W);
        for px in px.as_bytes_mut().chunks_exact_mut(8) {
            let out = I16x8::from_u8x8(px).add(dc);
            px.copy_from_slice(&out.pack_u8());
        }
    }
}

/// A `W`x`H` block of 8 and 16: the rows eight at a time into `tmp`, the columns eight at a time
/// out of it; 8x8 turns around in the registers.
#[inline(always)]
fn add_block<BD: BitDepth, const W: usize, const H: usize, const TYPE: TxfmType>(
    dst: Rav1dPictureDataComponentOffset,
    coeff: &mut [i16],
    eob: i32,
) {
    let (rows, cols) = kinds(TYPE);
    if TYPE == DCT_DCT && eob == 0 {
        return add_dc::<BD, W, H>(dst, coeff);
    }
    let shift = row_shift(W, H);
    let coeff = &mut coeff[..W * H];
    if W == 8 && H == 8 {
        let mut v: [I16x8; 8] = std::array::from_fn(|x| I16x8::load(&coeff[x * 8..]));
        coeff.fill(0);
        tx8::<false>(&mut v, rows);
        round(&mut v, round_mul(rows, 8, shift), negated(rows));
        I16x8::transpose8x8(&mut v);
        tx8::<true>(&mut v, cols);
        round(&mut v, round_mul(cols, 8, 4), negated(cols));
        return add_rows::<BD, 8>(dst, 0, &v);
    }
    let mut tmp = [0i16; 256];
    let tmp = &mut tmp[..W * H];
    let rect = if W * 2 == H || H * 2 == W {
        Some(I16x8::splat(2896 * 8))
    } else {
        None
    };
    for y0 in (0..H).step_by(8) {
        let mut v: [I16x8; W] = std::array::from_fn(|x| {
            let v = I16x8::load(&coeff[x * H + y0..]);
            match rect {
                Some(k) => v.mulhrs(k),
                None => v,
            }
        });
        tx::<W, false>(&mut v, rows, shift);
        round(&mut v, round_mul(rows, W, shift), negated(rows));
        for x0 in (0..W).step_by(8) {
            let block: &mut [I16x8; 8] = (&mut v[x0..x0 + 8]).try_into().unwrap();
            I16x8::transpose8x8(block);
            for y in 0..8 {
                block[y].store(&mut tmp[(y0 + y) * W + x0..]);
            }
        }
    }
    coeff.fill(0);
    for x0 in (0..W).step_by(8) {
        let mut v: [I16x8; H] = std::array::from_fn(|y| I16x8::load(&tmp[y * W + x0..]));
        tx::<H, true>(&mut v, cols, 4);
        round(&mut v, round_mul(cols, H, 4), negated(cols));
        add_rows::<BD, H>(dst, x0, &v);
    }
}

#[cfg_attr(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature(enable = "sse4.1")
)]
pub fn add<BD: BitDepth, const W: usize, const H: usize, const TYPE: TxfmType>(
    dst: Rav1dPictureDataComponentOffset,
    coeff: &mut [i16],
    eob: i32,
) {
    add_block::<BD, W, H, TYPE>(dst, coeff, eob)
}
