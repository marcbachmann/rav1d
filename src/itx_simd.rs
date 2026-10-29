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
        (Identity, 32) => 1 << (17 - shift),
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

/// `(x * c1 + 2048 >> 12, x * c2 + 2048 >> 12)`: a rotation whose other input is zero.
#[inline(always)]
fn rot1(x: I16x8, c1: i16, c2: i16) -> (I16x8, I16x8) {
    (
        x.mulhrs(I16x8::splat(c1 * 8)),
        x.mulhrs(I16x8::splat(c2 * 8)),
    )
}

/// `HALF`: the upper half of the inputs is zero (the C's `tx64`), so the first rotations are
/// single products.
#[inline(always)]
fn dct8<const HALF: bool>(v: &mut [I16x8; 8]) {
    let (t1, t0) = if HALF {
        let t = v[0].mulhrs(I16x8::splat(2896 * 8));
        (t, t)
    } else {
        v[0].rot(v[4], 2896, 2896)
    };
    let (t2, t3) = if HALF {
        rot1(v[2], 1567, 3784)
    } else {
        v[2].rot(v[6], 1567, 3784)
    };
    let d0 = t0.add_sat(t3);
    let d1 = t1.add_sat(t2);
    let d2 = t1.sub_sat(t2);
    let d3 = t0.sub_sat(t3);

    let (t4a, t7a) = if HALF {
        rot1(v[1], 799, 4017)
    } else {
        v[1].rot(v[7], 799, 4017)
    };
    let (t5a, t6a) = if HALF {
        rot1(v[3], -2276, 3406)
    } else {
        v[5].rot(v[3], 3406, 2276)
    };
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
fn dct16<const HALF: bool>(v: &mut [I16x8; 16]) {
    let mut e = [v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14]];
    dct8::<HALF>(&mut e);

    let ((t8a, t15a), (t9a, t14a), (t10a, t13a), (t11a, t12a)) = if HALF {
        (
            rot1(v[1], 401, 4076),
            rot1(v[7], -2598, 3166),
            rot1(v[5], 1931, 3612),
            rot1(v[3], -1189, 3920),
        )
    } else {
        (
            v[1].rot(v[15], 401, 4076),
            v[9].rot(v[7], 3166, 2598),
            v[5].rot(v[11], 1931, 3612),
            v[13].rot(v[3], 3920, 1189),
        )
    };

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

#[inline(always)]
fn dct32<const HALF: bool>(v: &mut [I16x8; 32]) {
    let mut e: [I16x8; 16] = std::array::from_fn(|i| v[2 * i]);
    dct16::<HALF>(&mut e);

    let (t16a, t31a, t17a, t30a, t18a, t29a, t19a, t28a);
    let (t20a, t27a, t21a, t26a, t22a, t25a, t23a, t24a);
    if HALF {
        (t16a, t31a) = rot1(v[1], 201, 4091);
        (t17a, t30a) = rot1(v[15], -2751, 3035);
        (t18a, t29a) = rot1(v[9], 1751, 3703);
        (t19a, t28a) = rot1(v[7], -1380, 3857);
        (t20a, t27a) = rot1(v[5], 995, 3973);
        (t21a, t26a) = rot1(v[11], -2106, 3513);
        (t22a, t25a) = rot1(v[13], 2440, 3290);
        (t23a, t24a) = rot1(v[3], -601, 4052);
    } else {
        (t16a, t31a) = v[1].rot(v[31], 201, 4091);
        (t17a, t30a) = v[17].rot(v[15], 3035, 2751);
        (t18a, t29a) = v[9].rot(v[23], 1751, 3703);
        (t19a, t28a) = v[25].rot(v[7], 3857, 1380);
        (t20a, t27a) = v[5].rot(v[27], 995, 3973);
        (t21a, t26a) = v[21].rot(v[11], 3513, 2106);
        (t22a, t25a) = v[13].rot(v[19], 2440, 3290);
        (t23a, t24a) = v[29].rot(v[3], 4052, 601);
    }

    let t16 = t16a.add_sat(t17a);
    let t17 = t16a.sub_sat(t17a);
    let t18 = t19a.sub_sat(t18a);
    let t19 = t19a.add_sat(t18a);
    let t20 = t20a.add_sat(t21a);
    let t21 = t20a.sub_sat(t21a);
    let t22 = t23a.sub_sat(t22a);
    let t23 = t23a.add_sat(t22a);
    let t24 = t24a.add_sat(t25a);
    let t25 = t24a.sub_sat(t25a);
    let t26 = t27a.sub_sat(t26a);
    let t27 = t27a.add_sat(t26a);
    let t28 = t28a.add_sat(t29a);
    let t29 = t28a.sub_sat(t29a);
    let t30 = t31a.sub_sat(t30a);
    let t31 = t31a.add_sat(t30a);

    let (t17a, t30a) = t30.rot(t17, 799, 4017);
    let (t18a, t29a) = t18.madd2(t29, [-799, -4017], [-4017, 799]);
    let (t21a, t26a) = t26.rot(t21, 3406, 2276);
    let (t22a, t25a) = t22.madd2(t25, [-3406, -2276], [-2276, 3406]);

    let t16a = t16.add_sat(t19);
    let t17 = t17a.add_sat(t18a);
    let t18 = t17a.sub_sat(t18a);
    let t19a = t16.sub_sat(t19);
    let t20a = t23.sub_sat(t20);
    let t21 = t22a.sub_sat(t21a);
    let t22 = t22a.add_sat(t21a);
    let t23a = t23.add_sat(t20);
    let t24a = t24.add_sat(t27);
    let t25 = t25a.add_sat(t26a);
    let t26 = t25a.sub_sat(t26a);
    let t27a = t24.sub_sat(t27);
    let t28a = t31.sub_sat(t28);
    let t29 = t30a.sub_sat(t29a);
    let t30 = t30a.add_sat(t29a);
    let t31a = t31.add_sat(t28);

    let (t18a, t29a) = t29.rot(t18, 1567, 3784);
    let (t19, t28) = t28a.rot(t19a, 1567, 3784);
    let (t20, t27) = t20a.madd2(t27a, [-1567, -3784], [-3784, 1567]);
    let (t21a, t26a) = t21.madd2(t26, [-1567, -3784], [-3784, 1567]);

    let t16 = t16a.add_sat(t23a);
    let t17a = t17.add_sat(t22);
    let t18 = t18a.add_sat(t21a);
    let t19a = t19.add_sat(t20);
    let t20a = t19.sub_sat(t20);
    let t21 = t18a.sub_sat(t21a);
    let t22a = t17.sub_sat(t22);
    let t23 = t16a.sub_sat(t23a);
    let t24 = t31a.sub_sat(t24a);
    let t25a = t30.sub_sat(t25);
    let t26 = t29a.sub_sat(t26a);
    let t27a = t28.sub_sat(t27);
    let t28a = t28.add_sat(t27);
    let t29 = t29a.add_sat(t26a);
    let t30a = t30.add_sat(t25);
    let t31 = t31a.add_sat(t24a);

    let (t20, t27) = t27a.rot(t20a, 2896, 2896);
    let (t21a, t26a) = t26.rot(t21, 2896, 2896);
    let (t22, t25) = t25a.rot(t22a, 2896, 2896);
    let (t23a, t24a) = t24.rot(t23, 2896, 2896);

    let o = [
        t31, t30a, t29, t28a, t27, t26a, t25, t24a, t23a, t22, t21a, t20, t19a, t18, t17a, t16,
    ];
    for i in 0..16 {
        v[i] = e[i].add_sat(o[i]);
        v[31 - i] = e[i].sub_sat(o[i]);
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
        Dct => dct8::<false>(v),
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
        Dct => dct16::<false>(v),
        Adst => adst16::<COLS>(v),
        FlipAdst => {
            adst16::<COLS>(v);
            v.reverse();
        }
        Identity => identity16(v, shift),
    }
}

/// Only the DCT and the identity reach 32 points; the identity's factor of 4 is a shift the
/// rounding absorbs.
#[inline(always)]
fn tx32(v: &mut [I16x8; 32], t: Tx1d, half: bool) {
    match (t, half) {
        (Dct, true) => dct32::<true>(v),
        (Dct, false) => dct32::<false>(v),
        _ => {}
    }
}

/// `half`: the upper half of the inputs is zero.
#[inline(always)]
fn tx<const N: usize, const COLS: bool>(v: &mut [I16x8; N], t: Tx1d, shift: u32, half: bool) {
    match N {
        8 => tx8::<COLS>((&mut v[..8]).try_into().unwrap(), t),
        16 => tx16::<COLS>((&mut v[..16]).try_into().unwrap(), t, shift),
        32 => tx32((&mut v[..32]).try_into().unwrap(), t, half),
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

/// A `W`x`H` block of 8, 16 and 32: the rows eight at a time into `tmp`, the columns eight at a
/// time out of it; 8x8 turns around in the registers. A 32x32 block with `eob < 136` has every
/// coefficient in its top-left 16x16 (the asm's tier), so half the rows and inputs are skipped.
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
    let half = W == 32 && eob < 136;
    let (w, h) = if half { (16, 16) } else { (W, H) };
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
    let mut tmp = [0i16; 1024];
    let tmp = &mut tmp[..W * H];
    let rect = if W * 2 == H || H * 2 == W {
        Some(I16x8::splat(2896 * 8))
    } else {
        None
    };
    let zero = I16x8::splat(0);
    for y0 in (0..h).step_by(8) {
        let mut v: [I16x8; W] = std::array::from_fn(|x| {
            if x >= w {
                return zero;
            }
            let c = &mut coeff[x * H + y0..];
            let v = I16x8::load(c);
            zero.store(c);
            match rect {
                Some(k) => v.mulhrs(k),
                None => v,
            }
        });
        tx::<W, false>(&mut v, rows, shift, half);
        round(&mut v, round_mul(rows, W, shift), negated(rows));
        for x0 in (0..W).step_by(8) {
            let block: &mut [I16x8; 8] = (&mut v[x0..x0 + 8]).try_into().unwrap();
            I16x8::transpose8x8(block);
            for y in 0..8 {
                block[y].store(&mut tmp[(y0 + y) * W + x0..]);
            }
        }
    }
    for x0 in (0..W).step_by(8) {
        let mut v: [I16x8; H] = std::array::from_fn(|y| {
            if y >= h {
                zero
            } else {
                I16x8::load(&tmp[y * W + x0..])
            }
        });
        tx::<H, true>(&mut v, cols, 4, half);
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
