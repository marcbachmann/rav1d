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

/// `log2` of the factor the identity transform of `n` points leaves out, folded into the rounding.
const fn scale_log2(t: Tx1d, n: usize) -> u32 {
    match (t, n) {
        (Identity, 8) => 1,
        _ => 0,
    }
}

/// The Q15 multiplier of a rounding shift by `shift` after a transform scaled by `2^scale_log2`;
/// 32768 is the identity and skipped.
const fn round_mul(shift: u32, scale_log2: u32) -> i32 {
    1 << (15 - shift + scale_log2)
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

    if COLS {
        let k = I16x8::splat(2896 * 8);
        v[3] = t2.add_sat(t3).mulhrs(k);
        v[4] = t2.sub_sat(t3).mulhrs(k);
        v[2] = t6.add_sat(t7).mulhrs(k);
        v[5] = t6.sub_sat(t7).mulhrs(k);
    } else {
        (v[4], v[3]) = t2.rot(t3, 2896, 2896);
        (v[5], v[2]) = t6.rot(t7, 2896, 2896);
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

/// `v[y]` added to the block's row `y`, saturated to the pixel range.
#[inline(always)]
fn add_rows<BD: BitDepth, const H: usize>(dst: Rav1dPictureDataComponentOffset, v: &[I16x8; H]) {
    for y in 0..H {
        let row = dst + (y as isize * dst.pixel_stride::<BD>());
        let px = &mut *row.slice_mut::<BD>(8);
        let px = px.as_bytes_mut();
        let out = I16x8::from_u8x8(px).add(v[y]);
        px.copy_from_slice(&out.pack_u8());
    }
}

/// `eob == 0` of a DCT: one value over the block.
#[inline(always)]
fn add_dc<BD: BitDepth, const W: usize, const H: usize>(
    dst: Rav1dPictureDataComponentOffset,
    coeff: &mut [i16],
    shift: u32,
) {
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

#[cfg_attr(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature(enable = "sse4.1")
)]
pub fn add_8x8<BD: BitDepth, const TYPE: TxfmType>(
    dst: Rav1dPictureDataComponentOffset,
    coeff: &mut [i16],
    eob: i32,
) {
    let (rows, cols) = kinds(TYPE);
    if TYPE == DCT_DCT && eob == 0 {
        return add_dc::<BD, 8, 8>(dst, coeff, 1);
    }
    let coeff = &mut coeff[..64];
    let mut v: [I16x8; 8] = std::array::from_fn(|x| I16x8::load(&coeff[x * 8..]));
    coeff.fill(0);
    tx8::<false>(&mut v, rows);
    round(&mut v, round_mul(1, scale_log2(rows, 8)), negated(rows));
    I16x8::transpose8x8(&mut v);
    tx8::<true>(&mut v, cols);
    round(&mut v, round_mul(4, scale_log2(cols, 8)), negated(cols));
    add_rows::<BD, 8>(dst, &v);
}
