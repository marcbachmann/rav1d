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
use crate::simd128::{transpose8x8, I16x8, Lanes};
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

/// The outputs a transform of `n` points leaves negated: the adst's odd ones from 8 points on,
/// so the flipped adst's even ones.
const fn negated(t: Tx1d, n: usize) -> u32 {
    match t {
        Adst if n > 4 => 0xaaaa,
        FlipAdst if n > 4 => 0x5555,
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
/// transforms scale in their own way, 32768 means nothing is left to do and 65536 a doubling.
const fn round_mul(t: Tx1d, n: usize, shift: u32) -> i32 {
    match (t, n) {
        (Identity, 4) if shift == 1 => 32768,
        (Identity, 8) => 1 << (16 - shift),
        (Identity, 16) if shift < 4 => 32768,
        (Identity, 32) => 1 << (17 - shift),
        _ => 1 << (15 - shift),
    }
}

#[inline(always)]
fn round<V: Lanes, const N: usize>(v: &mut [V; N], k: i32, negated: u32) {
    if k == 65536 {
        for x in v {
            *x = x.add_sat(*x);
        }
        return;
    }
    if k == 32768 {
        let zero = V::splat(0);
        for i in 0..N {
            if negated >> i & 1 != 0 {
                v[i] = zero.sub_sat(v[i]);
            }
        }
        return;
    }
    let (pos, neg) = (V::splat(k as i16), V::splat(-k as i16));
    for i in 0..N {
        v[i] = v[i].mulhrs(if negated >> i & 1 != 0 { neg } else { pos });
    }
}

#[inline(always)]
fn dct4<V: Lanes>(v: &mut [V; 4]) {
    let (t1, t0) = v[0].rot(v[2], 2896, 2896);
    let (t2, t3) = v[1].rot(v[3], 1567, 3784);
    v[0] = t0.add_sat(t3);
    v[1] = t1.add_sat(t2);
    v[2] = t1.sub_sat(t2);
    v[3] = t0.sub_sat(t3);
}

/// Four-term sums in 32 bits, as the C has them; nothing comes out negated.
#[inline(always)]
fn adst4<V: Lanes>(v: &mut [V; 4]) {
    let [a, b, c, d] = *v;
    v[0] = a.madd4(b, c, d, [1321, 3344, 3803, 2482]);
    v[1] = a.madd4(b, c, d, [2482, 3344, -1321, -3803]);
    v[2] = a.madd4(b, c, d, [3344, 0, -3344, 3344]);
    v[3] = a.madd4(b, c, d, [3803, -3344, 2482, -1321]);
}

/// `x + (x * 1697 + 2048 >> 12)`; with the pass's rounding shift by 1 folded in as the asm does
/// it, an average that cannot overflow (`x` and the term share a sign, except at `x == -1`, whose
/// result is 0).
#[inline(always)]
fn identity4<V: Lanes>(v: &mut [V; 4], shift: u32) {
    let k = V::splat(1697 * 8);
    let (minus_one, zero) = (V::splat(-1), V::splat(0));
    for x in v {
        let term = x.mulhrs(k);
        *x = if shift == 1 {
            x.avg_u(term).and(x.eq(minus_one).eq(zero))
        } else {
            x.add_sat(term)
        };
    }
}

#[inline(always)]
fn tx4<V: Lanes>(v: &mut [V; 4], t: Tx1d, shift: u32) {
    match t {
        Dct => dct4(v),
        Adst => adst4(v),
        FlipAdst => {
            adst4(v);
            v.reverse();
        }
        Identity => identity4(v, shift),
    }
}

/// `(x * c1 + 2048 >> 12, x * c2 + 2048 >> 12)`: a rotation whose other input is zero.
#[inline(always)]
fn rot1<V: Lanes>(x: V, c1: i16, c2: i16) -> (V, V) {
    (x.mulhrs(V::splat(c1 * 8)), x.mulhrs(V::splat(c2 * 8)))
}

/// `HALF`: the upper half of the inputs is zero (the C's `tx64`), so the first rotations are
/// single products.
#[inline(always)]
fn dct8<V: Lanes, const HALF: bool>(v: &mut [V; 8]) {
    let (t1, t0) = if HALF {
        let t = v[0].mulhrs(V::splat(2896 * 8));
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
fn dct16<V: Lanes, const HALF: bool>(v: &mut [V; 16]) {
    let mut e = [v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14]];
    dct8::<_, HALF>(&mut e);

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
fn dct32<V: Lanes, const HALF: bool>(v: &mut [V; 32]) {
    let mut e: [V; 16] = std::array::from_fn(|i| v[2 * i]);
    dct16::<_, HALF>(&mut e);

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

/// A 64-point DCT sees 32 coefficients, so its upper inputs are always zero.
#[inline(always)]
fn dct64<V: Lanes>(v: &mut [V; 64]) {
    let mut e: [V; 32] = std::array::from_fn(|i| v[2 * i]);
    dct32::<_, true>(&mut e);

    let (t32a, t63a) = rot1(v[1], 101, 4095);
    let (t33a, t62a) = rot1(v[31], -2824, 2967);
    let (t34a, t61a) = rot1(v[17], 1660, 3745);
    let (t35a, t60a) = rot1(v[15], -1474, 3822);
    let (t36a, t59a) = rot1(v[9], 897, 3996);
    let (t37a, t58a) = rot1(v[23], -2191, 3461);
    let (t38a, t57a) = rot1(v[25], 2359, 3349);
    let (t39a, t56a) = rot1(v[7], -700, 4036);
    let (t40a, t55a) = rot1(v[5], 501, 4065);
    let (t41a, t54a) = rot1(v[27], -2520, 3229);
    let (t42a, t53a) = rot1(v[21], 2019, 3564);
    let (t43a, t52a) = rot1(v[11], -1092, 3948);
    let (t44a, t51a) = rot1(v[13], 1285, 3889);
    let (t45a, t50a) = rot1(v[19], -1842, 3659);
    let (t46a, t49a) = rot1(v[29], 2675, 3102);
    let (t47a, t48a) = rot1(v[3], -301, 4085);

    let t32 = t32a.add_sat(t33a);
    let t33 = t32a.sub_sat(t33a);
    let t34 = t35a.sub_sat(t34a);
    let t35 = t35a.add_sat(t34a);
    let t36 = t36a.add_sat(t37a);
    let t37 = t36a.sub_sat(t37a);
    let t38 = t39a.sub_sat(t38a);
    let t39 = t39a.add_sat(t38a);
    let t40 = t40a.add_sat(t41a);
    let t41 = t40a.sub_sat(t41a);
    let t42 = t43a.sub_sat(t42a);
    let t43 = t43a.add_sat(t42a);
    let t44 = t44a.add_sat(t45a);
    let t45 = t44a.sub_sat(t45a);
    let t46 = t47a.sub_sat(t46a);
    let t47 = t47a.add_sat(t46a);
    let t48 = t48a.add_sat(t49a);
    let t49 = t48a.sub_sat(t49a);
    let t50 = t51a.sub_sat(t50a);
    let t51 = t51a.add_sat(t50a);
    let t52 = t52a.add_sat(t53a);
    let t53 = t52a.sub_sat(t53a);
    let t54 = t55a.sub_sat(t54a);
    let t55 = t55a.add_sat(t54a);
    let t56 = t56a.add_sat(t57a);
    let t57 = t56a.sub_sat(t57a);
    let t58 = t59a.sub_sat(t58a);
    let t59 = t59a.add_sat(t58a);
    let t60 = t60a.add_sat(t61a);
    let t61 = t60a.sub_sat(t61a);
    let t62 = t63a.sub_sat(t62a);
    let t63 = t63a.add_sat(t62a);

    let (t33a, t62a) = t62.rot(t33, 401, 4076);
    let (t34a, t61a) = t34.madd2(t61, [-401, -4076], [-4076, 401]);
    let (t37a, t58a) = t58.rot(t37, 3166, 2598);
    let (t38a, t57a) = t38.madd2(t57, [-3166, -2598], [-2598, 3166]);
    let (t41a, t54a) = t54.rot(t41, 1931, 3612);
    let (t42a, t53a) = t42.madd2(t53, [-1931, -3612], [-3612, 1931]);
    let (t45a, t50a) = t50.rot(t45, 3920, 1189);
    let (t46a, t49a) = t46.madd2(t49, [-3920, -1189], [-1189, 3920]);

    let t32a = t32.add_sat(t35);
    let t33 = t33a.add_sat(t34a);
    let t34 = t33a.sub_sat(t34a);
    let t35a = t32.sub_sat(t35);
    let t36a = t39.sub_sat(t36);
    let t37 = t38a.sub_sat(t37a);
    let t38 = t38a.add_sat(t37a);
    let t39a = t39.add_sat(t36);
    let t40a = t40.add_sat(t43);
    let t41 = t41a.add_sat(t42a);
    let t42 = t41a.sub_sat(t42a);
    let t43a = t40.sub_sat(t43);
    let t44a = t47.sub_sat(t44);
    let t45 = t46a.sub_sat(t45a);
    let t46 = t46a.add_sat(t45a);
    let t47a = t47.add_sat(t44);
    let t48a = t48.add_sat(t51);
    let t49 = t49a.add_sat(t50a);
    let t50 = t49a.sub_sat(t50a);
    let t51a = t48.sub_sat(t51);
    let t52a = t55.sub_sat(t52);
    let t53 = t54a.sub_sat(t53a);
    let t54 = t54a.add_sat(t53a);
    let t55a = t55.add_sat(t52);
    let t56a = t56.add_sat(t59);
    let t57 = t57a.add_sat(t58a);
    let t58 = t57a.sub_sat(t58a);
    let t59a = t56.sub_sat(t59);
    let t60a = t63.sub_sat(t60);
    let t61 = t62a.sub_sat(t61a);
    let t62 = t62a.add_sat(t61a);
    let t63a = t63.add_sat(t60);

    let (t34a, t61a) = t61.rot(t34, 799, 4017);
    let (t35, t60) = t60a.rot(t35a, 799, 4017);
    let (t36, t59) = t36a.madd2(t59a, [-799, -4017], [-4017, 799]);
    let (t37a, t58a) = t37.madd2(t58, [-799, -4017], [-4017, 799]);
    let (t42a, t53a) = t53.rot(t42, 3406, 2276);
    let (t43, t52) = t52a.rot(t43a, 3406, 2276);
    let (t44, t51) = t44a.madd2(t51a, [-3406, -2276], [-2276, 3406]);
    let (t45a, t50a) = t45.madd2(t50, [-3406, -2276], [-2276, 3406]);

    let t32 = t32a.add_sat(t39a);
    let t33a = t33.add_sat(t38);
    let t34 = t34a.add_sat(t37a);
    let t35a = t35.add_sat(t36);
    let t36a = t35.sub_sat(t36);
    let t37 = t34a.sub_sat(t37a);
    let t38a = t33.sub_sat(t38);
    let t39 = t32a.sub_sat(t39a);
    let t40 = t47a.sub_sat(t40a);
    let t41a = t46.sub_sat(t41);
    let t42 = t45a.sub_sat(t42a);
    let t43a = t44.sub_sat(t43);
    let t44a = t44.add_sat(t43);
    let t45 = t45a.add_sat(t42a);
    let t46a = t46.add_sat(t41);
    let t47 = t47a.add_sat(t40a);
    let t48 = t48a.add_sat(t55a);
    let t49a = t49.add_sat(t54);
    let t50 = t50a.add_sat(t53a);
    let t51a = t51.add_sat(t52);
    let t52a = t51.sub_sat(t52);
    let t53 = t50a.sub_sat(t53a);
    let t54a = t49.sub_sat(t54);
    let t55 = t48a.sub_sat(t55a);
    let t56 = t63a.sub_sat(t56a);
    let t57a = t62.sub_sat(t57);
    let t58 = t61a.sub_sat(t58a);
    let t59a = t60.sub_sat(t59);
    let t60a = t60.add_sat(t59);
    let t61 = t61a.add_sat(t58a);
    let t62a = t62.add_sat(t57);
    let t63 = t63a.add_sat(t56a);

    let (t36, t59) = t59a.rot(t36a, 1567, 3784);
    let (t37a, t58a) = t58.rot(t37, 1567, 3784);
    let (t38, t57) = t57a.rot(t38a, 1567, 3784);
    let (t39a, t56a) = t56.rot(t39, 1567, 3784);
    let (t40a, t55a) = t40.madd2(t55, [-1567, -3784], [-3784, 1567]);
    let (t41, t54) = t41a.madd2(t54a, [-1567, -3784], [-3784, 1567]);
    let (t42a, t53a) = t42.madd2(t53, [-1567, -3784], [-3784, 1567]);
    let (t43, t52) = t43a.madd2(t52a, [-1567, -3784], [-3784, 1567]);

    let t32a = t32.add_sat(t47);
    let t33 = t33a.add_sat(t46a);
    let t34a = t34.add_sat(t45);
    let t35 = t35a.add_sat(t44a);
    let t36a = t36.add_sat(t43);
    let t37 = t37a.add_sat(t42a);
    let t38a = t38.add_sat(t41);
    let t39 = t39a.add_sat(t40a);
    let t40 = t39a.sub_sat(t40a);
    let t41a = t38.sub_sat(t41);
    let t42 = t37a.sub_sat(t42a);
    let t43a = t36.sub_sat(t43);
    let t44 = t35a.sub_sat(t44a);
    let t45a = t34.sub_sat(t45);
    let t46 = t33a.sub_sat(t46a);
    let t47a = t32.sub_sat(t47);
    let t48a = t63.sub_sat(t48);
    let t49 = t62a.sub_sat(t49a);
    let t50a = t61.sub_sat(t50);
    let t51 = t60a.sub_sat(t51a);
    let t52a = t59.sub_sat(t52);
    let t53 = t58a.sub_sat(t53a);
    let t54a = t57.sub_sat(t54);
    let t55 = t56a.sub_sat(t55a);
    let t56 = t56a.add_sat(t55a);
    let t57a = t57.add_sat(t54);
    let t58 = t58a.add_sat(t53a);
    let t59a = t59.add_sat(t52);
    let t60 = t60a.add_sat(t51a);
    let t61a = t61.add_sat(t50);
    let t62 = t62a.add_sat(t49a);
    let t63a = t63.add_sat(t48);

    let (t40a, t55a) = t55.rot(t40, 2896, 2896);
    let (t41, t54) = t54a.rot(t41a, 2896, 2896);
    let (t42a, t53a) = t53.rot(t42, 2896, 2896);
    let (t43, t52) = t52a.rot(t43a, 2896, 2896);
    let (t44a, t51a) = t51.rot(t44, 2896, 2896);
    let (t45, t50) = t50a.rot(t45a, 2896, 2896);
    let (t46a, t49a) = t49.rot(t46, 2896, 2896);
    let (t47, t48) = t48a.rot(t47a, 2896, 2896);

    let o = [
        t63a, t62, t61a, t60, t59a, t58, t57a, t56, t55a, t54, t53a, t52, t51a, t50, t49a, t48,
        t47, t46a, t45, t44a, t43, t42a, t41, t40a, t39, t38a, t37, t36a, t35, t34a, t33, t32a,
    ];
    for i in 0..32 {
        v[i] = e[i].add_sat(o[i]);
        v[63 - i] = e[i].sub_sat(o[i]);
    }
}

/// Outputs 1, 3, 5 and 7 come out negated. The last stage's `* 2896 >> 12` runs in 32 bits on
/// the row pass and in 16 on the column pass, where the asm has the inputs small enough.
#[inline(always)]
fn adst8<V: Lanes, const COLS: bool>(v: &mut [V; 8]) {
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

    (v[4], v[3]) = half_rot::<_, COLS>(t2, t3);
    (v[5], v[2]) = half_rot::<_, COLS>(t6, t7);
}

/// `((a - b) * 2896 >> 12, (a + b) * 2896 >> 12)`: the adst's last stage.
#[inline(always)]
fn half_rot<V: Lanes, const COLS: bool>(a: V, b: V) -> (V, V) {
    if COLS {
        let k = V::splat(2896 * 8);
        (a.sub_sat(b).mulhrs(k), a.add_sat(b).mulhrs(k))
    } else {
        a.rot(b, 2896, 2896)
    }
}

/// The odd outputs come out negated, as in [`adst8`].
#[inline(always)]
fn adst16<V: Lanes, const COLS: bool>(v: &mut [V; 16]) {
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

    (v[8], v[7]) = half_rot::<_, COLS>(t2a, t3a);
    (v[11], v[4]) = half_rot::<_, COLS>(t6, t7);
    (v[9], v[6]) = half_rot::<_, COLS>(t10, t11);
    (v[10], v[5]) = half_rot::<_, COLS>(t14a, t15a);
}

/// `2 * x + (x * 1697 + 1024 >> 11)`, with the pass's rounding shift folded in before the
/// picture: by 2 as an average of `x` and the half term (the same sign, so the unsigned average
/// is exact), by 1 as `x + (term + 1 >> 1)`.
#[inline(always)]
fn identity16<V: Lanes>(v: &mut [V; 16], shift: u32) {
    let k = V::splat(1697 * 16);
    for x in v {
        let term = x.mulhrs(k);
        *x = match shift {
            1 => x.add_sat(term.mulhrs(V::splat(16384))),
            2 => x.avg_u(term.shr_s1()),
            _ => x.add_sat(*x).add_sat(term),
        };
    }
}

#[inline(always)]
fn tx8<V: Lanes, const COLS: bool>(v: &mut [V; 8], t: Tx1d) {
    match t {
        Dct => dct8::<_, false>(v),
        Adst => adst8::<_, COLS>(v),
        FlipAdst => {
            adst8::<_, COLS>(v);
            v.reverse();
        }
        Identity => {}
    }
}

#[inline(always)]
fn tx16<V: Lanes, const COLS: bool>(v: &mut [V; 16], t: Tx1d, shift: u32) {
    match t {
        Dct => dct16::<_, false>(v),
        Adst => adst16::<_, COLS>(v),
        FlipAdst => {
            adst16::<_, COLS>(v);
            v.reverse();
        }
        Identity => identity16(v, shift),
    }
}

/// Only the DCT and the identity reach 32 points; the identity's factor of 4 is a shift the
/// rounding absorbs.
#[inline(always)]
fn tx32<V: Lanes>(v: &mut [V; 32], t: Tx1d, half: bool) {
    match (t, half) {
        (Dct, true) => dct32::<_, true>(v),
        (Dct, false) => dct32::<_, false>(v),
        _ => {}
    }
}

/// `half`: the upper half of the inputs is zero.
#[inline(always)]
fn tx<V: Lanes, const N: usize, const COLS: bool>(v: &mut [V; N], t: Tx1d, shift: u32, half: bool) {
    match N {
        4 => tx4((&mut v[..4]).try_into().unwrap(), t, shift),
        8 => tx8::<_, COLS>((&mut v[..8]).try_into().unwrap(), t),
        16 => tx16::<_, COLS>((&mut v[..16]).try_into().unwrap(), t, shift),
        32 => tx32((&mut v[..32]).try_into().unwrap(), t, half),
        64 => dct64((&mut v[..64]).try_into().unwrap()),
        _ => unreachable!(),
    }
}

/// `v[y]` added to the block's row `y` from column `x0`, saturated to the pixel range; `W` lanes
/// of 4 or 8.
#[inline(always)]
fn add_rows<BD: BitDepth, const W: usize, const H: usize>(
    dst: Rav1dPictureDataComponentOffset,
    x0: usize,
    v: &[I16x8; H],
) {
    let n = if W == 4 { 4 } else { 8 };
    for y in 0..H {
        let row = dst + (y as isize * dst.pixel_stride::<BD>());
        let px = &mut *row.slice_mut::<BD>(x0 + n);
        let px = &mut px.as_bytes_mut()[x0..];
        let out = if n == 4 {
            I16x8::from_u8x4(px)
        } else {
            I16x8::from_u8x8(px)
        }
        .add(v[y]);
        px.copy_from_slice(&out.pack_u8()[..n]);
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
        let px = px.as_bytes_mut();
        if W == 4 {
            let out = I16x8::from_u8x4(px).add(dc);
            px.copy_from_slice(&out.pack_u8()[..4]);
            continue;
        }
        for px in px.chunks_exact_mut(8) {
            let out = I16x8::from_u8x8(px).add(dc);
            px.copy_from_slice(&out.pack_u8());
        }
    }
}

/// A `W`x`H` block of 4 to 64 in `G = W / 8` (rounded up) column groups: the rows eight at a
/// time, each group's 8x8 turned around into its column vectors, then each group's columns into
/// the picture. The column vectors are an array the compiler keeps in registers where they fit,
/// with no scratch in between. A side of 4 fills half the lanes; a side of 64 has 32
/// coefficients. A 32x32 block with `eob < 136` has every coefficient in its top-left 16x16 (the
/// asm's tier), so half the rows and inputs are skipped.
#[inline(always)]
fn add_block<
    BD: BitDepth,
    const W: usize,
    const H: usize,
    const TYPE: TxfmType,
    const G: usize,
    const WIDE: bool,
>(
    dst: Rav1dPictureDataComponentOffset,
    coeff: &mut [i16],
    eob: i32,
) {
    let (rows, cols) = kinds(TYPE);
    if TYPE == DCT_DCT && eob == 0 {
        return add_dc::<BD, W, H>(dst, coeff);
    }
    let shift = row_shift(W, H);
    let (cw, ch) = (W.min(32), H.min(32));
    let coeff = &mut coeff[..cw * ch];
    let half = W == 32 && H == 32 && eob < 136;
    let (w, h) = if half { (16, 16) } else { (cw, ch) };
    let rect = W * 2 == H || H * 2 == W;
    let zero = I16x8::splat(0);
    let mut groups = [[zero; H]; G];
    let mut y0 = 0;
    #[cfg(target_arch = "x86_64")]
    if WIDE && ch >= 16 {
        while y0 < h {
            wide::rows::<W, H, G>(coeff, ch, y0, w, rect, rows, shift, half, &mut groups);
            y0 += 16;
        }
    }
    while y0 < h {
        let mut v = [zero; W];
        for x in 0..w {
            let c = &mut coeff[x * ch + y0..];
            v[x] = if H == 4 {
                let v = I16x8::load_lo(c);
                zero.store_lo(c);
                v
            } else {
                let v = I16x8::load(c);
                zero.store(c);
                v
            };
            if rect {
                v[x] = v[x].mulhrs(I16x8::splat(2896 * 8));
            }
        }
        tx::<_, W, false>(&mut v, rows, shift, half);
        round(&mut v, round_mul(rows, W, shift), negated(rows, W));
        for (g, group) in groups.iter_mut().enumerate() {
            let mut block = [zero; 8];
            for i in 0..8.min(W) {
                block[i] = v[g * 8 + i];
            }
            transpose8x8(&mut block);
            for y in 0..8.min(H) {
                group[y0 + y] = block[y];
            }
        }
        y0 += 8;
    }
    let mut g = 0;
    #[cfg(target_arch = "x86_64")]
    if WIDE && W >= 16 {
        while g + 1 < G {
            wide::cols::<BD, H>(dst, g * 8, &groups[g], &groups[g + 1], cols, half);
            g += 2;
        }
    }
    while g < G {
        let group = &mut groups[g];
        tx::<_, H, true>(group, cols, 4, half);
        round(group, round_mul(cols, H, 4), negated(cols, H));
        add_rows::<BD, W, H>(dst, g * 8, group);
        g += 1;
    }
}

/// The passes sixteen lanes at a time: rows where a column holds 16 coefficients, columns where
/// the block is 16 wide. A 16-lane 8x8 turn is two 8x8 turns, one per 128-bit half, so the row
/// pass hands each column group rows `y0..y0 + 8` from the low halves and the next eight from
/// the high ones, and the column pass joins two groups' rows.
#[cfg(target_arch = "x86_64")]
mod wide {
    use super::*;
    use crate::simd128::I16x16;

    #[inline(always)]
    pub fn rows<const W: usize, const H: usize, const G: usize>(
        coeff: &mut [i16],
        ch: usize,
        y0: usize,
        w: usize,
        rect: bool,
        rows: Tx1d,
        shift: u32,
        half: bool,
        groups: &mut [[I16x8; H]; G],
    ) {
        let zero = I16x16::splat(0);
        let mut v = [zero; W];
        for x in 0..w {
            let c = &mut coeff[x * ch + y0..];
            v[x] = I16x16::load(c);
            zero.store(c);
            if rect {
                v[x] = v[x].mulhrs(I16x16::splat(2896 * 8));
            }
        }
        tx::<_, W, false>(&mut v, rows, shift, half);
        round(&mut v, round_mul(rows, W, shift), negated(rows, W));
        for (g, group) in groups.iter_mut().enumerate() {
            let mut block = [zero; 8];
            for i in 0..8.min(W) {
                block[i] = v[g * 8 + i];
            }
            transpose8x8(&mut block);
            for y in 0..8 {
                group[y0 + y] = block[y].lo();
                group[y0 + 8 + y] = block[y].hi();
            }
        }
    }

    #[inline(always)]
    pub fn cols<BD: BitDepth, const H: usize>(
        dst: Rav1dPictureDataComponentOffset,
        x0: usize,
        a: &[I16x8; H],
        b: &[I16x8; H],
        cols: Tx1d,
        half: bool,
    ) {
        let mut v = [I16x16::splat(0); H];
        for y in 0..H {
            v[y] = I16x16::from_halves(a[y], b[y]);
        }
        tx::<_, H, true>(&mut v, cols, 4, half);
        round(&mut v, round_mul(cols, H, 4), negated(cols, H));
        for y in 0..H {
            let row = dst + (y as isize * dst.pixel_stride::<BD>());
            let px = &mut *row.slice_mut::<BD>(x0 + 16);
            let px = &mut px.as_bytes_mut()[x0..];
            let out = I16x16::from_u8x16(px).add(v[y]);
            px.copy_from_slice(&out.pack_u8());
        }
    }
}

#[cfg_attr(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature(enable = "sse4.1")
)]
pub fn add<BD: BitDepth, const W: usize, const H: usize, const TYPE: TxfmType, const G: usize>(
    dst: Rav1dPictureDataComponentOffset,
    coeff: &mut [i16],
    eob: i32,
) {
    add_block::<BD, W, H, TYPE, G, false>(dst, coeff, eob)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn add_avx2<
    BD: BitDepth,
    const W: usize,
    const H: usize,
    const TYPE: TxfmType,
    const G: usize,
>(
    dst: Rav1dPictureDataComponentOffset,
    coeff: &mut [i16],
    eob: i32,
) {
    add_block::<BD, W, H, TYPE, G, true>(dst, coeff, eob)
}
