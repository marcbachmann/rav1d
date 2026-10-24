//! Forward transforms for the encoder's search: rav1e's Daala-style lifting kernels
//! (rusty-av1-toolkit, BSD-2, the rav1e contributors) in a scalar lane, an AVX2 lane and a
//! NEON lane with the scalar lane's output. Not libaom's arithmetic, so not `fdct.rs`'s
//! coefficients, but its layout and scale.

use std::mem::MaybeUninit;

use strum::EnumCount;

use super::fdct::{self, dimensions, kinds, Kind};
use crate::cpu::{rav1d_get_cpu_flags, CpuFlags};
use crate::levels::{TxfmSize, TxfmType, WHT_WHT};

#[macro_use]
mod kernels;
#[cfg(all(test, unix))]
mod bounds;
#[cfg(target_arch = "aarch64")]
mod arm;
mod scalar;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Txfm1d {
    Dct4,
    Dct8,
    Dct16,
    Dct32,
    Dct64,
    Adst4,
    Adst8,
    Adst16,
    Identity,
}

impl Txfm1d {
    fn new(kind: Kind, n: usize) -> Self {
        use Txfm1d::*;
        match (kind, n) {
            (Kind::Dct, 4) => Dct4,
            (Kind::Dct, 8) => Dct8,
            (Kind::Dct, 16) => Dct16,
            (Kind::Dct, 32) => Dct32,
            (Kind::Dct, 64) => Dct64,
            (Kind::Adst | Kind::FlipAdst, 4) => Adst4,
            (Kind::Adst | Kind::FlipAdst, 8) => Adst8,
            (Kind::Adst | Kind::FlipAdst, 16) => Adst16,
            (Kind::Identity, 4..=32) => Identity,
            _ => panic!("no {kind:?} transform of {n} points"),
        }
    }
}

/// rav1e's 8-bit shifts: left before the columns, rounding right (negated) after the columns
/// and after the rows. Its 1D transforms are orthonormal, so these give libaom's scale.
const SHIFT: [[i8; 3]; TxfmSize::COUNT] = [
    [3, 0, 0],
    [4, -1, 0],
    [4, -1, 0],
    [4, -2, 0],
    [4, -1, -2],
    [4, -1, 0],
    [4, -1, 0],
    [4, -1, 0],
    [4, -1, 0],
    [4, -2, 0],
    [4, -2, 0],
    [4, -1, -2],
    [4, -1, -2],
    [4, -1, 0],
    [4, -1, 0],
    [4, -1, 0],
    [4, -1, 0],
    [4, -2, 0],
    [4, -2, 0],
];

struct Config {
    w: usize,
    h: usize,
    col: Txfm1d,
    row: Txfm1d,
    ud_flip: bool,
    lr_flip: bool,
    shift: [i8; 3],
}

impl Config {
    fn new(size: TxfmSize, ty: TxfmType) -> Self {
        let (w, h) = dimensions(size);
        let (v, hz) = kinds(ty);
        Self {
            w,
            h,
            col: Txfm1d::new(v, h),
            row: Txfm1d::new(hz, w),
            ud_flip: v == Kind::FlipAdst,
            lr_flip: hz == Kind::FlipAdst,
            shift: SHIFT[size as usize],
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lane {
    Scalar,
    /// AVX2 on x86, NEON on aarch64: the scalar lane's coefficients.
    Simd,
}

impl Lane {
    pub fn detect() -> Self {
        let flags = rav1d_get_cpu_flags();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let simd = flags.contains(CpuFlags::AVX2);
        #[cfg(target_arch = "aarch64")]
        let simd = flags.contains(CpuFlags::NEON);
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        let simd = {
            let _ = flags;
            false
        };
        if simd {
            Lane::Simd
        } else {
            Lane::Scalar
        }
    }
}

/// `fdct::forward`'s contract (`coefs[x * ch + y]`, `coded_dimensions`, `WHT_WHT` on its
/// `fwht4x4`) on the lane rav1d's CPU flags allow.
pub fn forward(residual: &[i16], stride: usize, coefs: &mut [i32], size: TxfmSize, ty: TxfmType) {
    forward_in(Lane::detect(), residual, stride, coefs, size, ty)
}

pub fn forward_in(
    lane: Lane,
    residual: &[i16],
    stride: usize,
    coefs: &mut [i32],
    size: TxfmSize,
    ty: TxfmType,
) {
    if ty == WHT_WHT {
        return fdct::forward(residual, stride, coefs, size, ty);
    }
    run(lane, residual, stride, coefs, &Config::new(size, ty))
}

fn run(lane: Lane, residual: &[i16], stride: usize, coefs: &mut [i32], cfg: &Config) {
    let coefs = &mut coefs[..cfg.w.min(32) * cfg.h.min(32)];
    assert!(stride >= cfg.w && residual.len() >= (cfg.h - 1) * stride + cfg.w);
    match lane {
        Lane::Scalar => scalar::forward(residual, stride, coefs, cfg),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Lane::Simd => {
            assert!(is_x86_feature_detected!("avx2"));
            // SAFETY: AVX2 is available.
            unsafe { x86::forward(residual, stride, coefs, cfg) }
        }
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is part of aarch64.
        Lane::Simd => unsafe { arm::forward(residual, stride, coefs, cfg) },
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        Lane::Simd => scalar::forward(residual, stride, coefs, cfg),
    }
}

fn array<const N: usize, T>(s: &[T]) -> &[T; N] {
    s[..N].try_into().unwrap()
}

fn array_mut<const N: usize, T>(s: &mut [T]) -> &mut [T; N] {
    (&mut s[..N]).try_into().unwrap()
}

/// # Safety
///
/// Every element of `s` is initialized.
#[cfg_attr(
    not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")),
    allow(dead_code)
)]
unsafe fn assume_init_mut<T>(s: &mut [MaybeUninit<T>]) -> &mut [T] {
    // SAFETY: `MaybeUninit<T>` has `T`'s layout, and the caller says all are initialized.
    unsafe { &mut *(s as *mut [MaybeUninit<T>] as *mut [T]) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enc::fdct::tests::{
        blocks, cpu_levels, decodable, max_error, sizes, Rng, ROUND_TRIP_BOUND,
    };
    use crate::enc::fdct::coded_dimensions;
    use crate::include::common::bitdepth::BitDepth8;
    use crate::itx::Rav1dInvTxfmDSPContext;

    fn simd_available() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        return is_x86_feature_detected!("avx2");
        #[cfg(target_arch = "aarch64")]
        return true;
        #[allow(unreachable_code)]
        false
    }

    fn lanes() -> Vec<Lane> {
        let mut lanes = vec![Lane::Scalar];
        if simd_available() {
            lanes.push(Lane::Simd);
        }
        lanes
    }

    fn pairs() -> impl Iterator<Item = (TxfmSize, TxfmType)> {
        sizes().flat_map(|size| {
            (0..WHT_WHT)
                .filter(move |&ty| decodable(size, ty))
                .map(move |ty| (size, ty))
        })
    }

    /// Blocks inside a wider picture, so a stride beyond the width is read right.
    fn assert_lanes_agree(size: TxfmSize, ty: TxfmType, block: &[i16]) {
        let (w, h) = dimensions(size);
        let stride = w + 8;
        let mut wide = vec![0x5a5a; stride * h + 8];
        for r in 0..h {
            wide[r * stride..][..w].copy_from_slice(&block[r * w..][..w]);
        }
        let n = cw_ch(size);
        let mut scalar = vec![i32::MIN; n];
        let mut simd = vec![i32::MAX; n];
        forward_in(Lane::Scalar, block, w, &mut scalar, size, ty);
        forward_in(Lane::Simd, &wide, stride, &mut simd, size, ty);
        assert!(scalar == simd, "{size:?} {ty}: lanes differ");
    }

    fn cw_ch(size: TxfmSize) -> usize {
        let (cw, ch) = coded_dimensions(size);
        cw * ch
    }

    #[test]
    fn simd_matches_scalar() {
        if !simd_available() {
            println!("no SIMD lane on this CPU");
            return;
        }
        for (size, ty) in pairs() {
            let seed = 0x6a09_e667_f3bc_c908 ^ ((size as u64) << 8 | ty as u64);
            for block in blocks(size, seed, 32, 255) {
                assert_lanes_agree(size, ty, &block);
            }
        }
    }

    /// For each coded coefficient, the ±255 block that drives it furthest, from the signs
    /// of the transform's impulse responses: in opt-dev the scalar lane's overflow checks
    /// fail on any intermediate beyond `i32`, where the SIMD lanes would wrap or saturate.
    #[test]
    fn extreme_blocks_stay_in_range() {
        for (size, ty) in pairs() {
            let (w, h) = dimensions(size);
            let n = cw_ch(size);
            let mut responses = vec![0i32; w * h * n];
            let mut impulse = vec![0i16; w * h];
            let mut coefs = vec![0; n];
            for i in 0..w * h {
                impulse[i] = 255;
                forward_in(Lane::Scalar, &impulse, w, &mut coefs, size, ty);
                impulse[i] = 0;
                for k in 0..n {
                    responses[k * w * h + i] = coefs[k];
                }
            }
            for k in 0..n {
                let block: Vec<i16> = responses[k * w * h..][..w * h]
                    .iter()
                    .map(|&r| if r < 0 { -255 } else { 255 })
                    .collect();
                forward_in(Lane::Scalar, &block, w, &mut coefs, size, ty);
                let negated: Vec<i16> = block.iter().map(|&v| -v).collect();
                forward_in(Lane::Scalar, &negated, w, &mut coefs, size, ty);
                if simd_available() {
                    assert_lanes_agree(size, ty, &block);
                    assert_lanes_agree(size, ty, &negated);
                }
            }
        }
    }

    #[test]
    fn round_trip_through_rav1d() {
        let mut report = String::new();
        let mut failed = false;
        for lane in lanes() {
            for flags in cpu_levels() {
                let itx = Rav1dInvTxfmDSPContext::new::<BitDepth8>(flags, 8);
                report += &format!("{lane:?} cpu flags {:#x}\n", flags.bits());
                let fwd = |r: &[i16], s: usize, c: &mut [i32], size: TxfmSize, ty: TxfmType| {
                    forward_in(lane, r, s, c, size, ty)
                };
                for size in sizes() {
                    for ty in (0..=WHT_WHT).filter(|&ty| decodable(size, ty)) {
                        // rav1d has no Rust inverse WHT where its NEON one is compiled in.
                        if cfg!(target_arch = "aarch64") && ty == WHT_WHT && flags.is_empty() {
                            continue;
                        }
                        let worst = max_error(&itx, size, ty, &fwd);
                        let bound = if ty == WHT_WHT {
                            0
                        } else {
                            ROUND_TRIP_BOUND[size as usize]
                        };
                        if worst > bound {
                            report += &format!("{size:?} {ty}: {worst} > {bound}\n");
                            failed = true;
                        }
                    }
                }
            }
        }
        println!("{report}");
        assert!(!failed, "max error beyond the bound:\n{report}");
    }

    /// The quantizer and rav1d's dequantizer assume libaom's scale: the RMS of the
    /// coefficients of random blocks per size and type, against `fdct.rs`'s.
    #[test]
    fn same_scale_as_libaom() {
        let mut report = String::new();
        let mut worst = 0f64;
        for size in sizes() {
            let n = cw_ch(size);
            let (mut lo, mut hi) = (f64::MAX, 0f64);
            let (mut fast_all, mut aom_all) = (0f64, 0f64);
            for ty in (0..WHT_WHT).filter(|&ty| decodable(size, ty)) {
                let mut rng = Rng(0xbb67_ae85_84ca_a73b ^ ((size as u64) << 8 | ty as u64));
                let (w, h) = dimensions(size);
                let (mut fast_sq, mut aom_sq) = (0f64, 0f64);
                let mut a = vec![0; n];
                let mut b = vec![0; n];
                for _ in 0..64 {
                    let block: Vec<i16> = (0..w * h).map(|_| rng.residual(255)).collect();
                    forward_in(Lane::Scalar, &block, w, &mut a, size, ty);
                    fdct::forward(&block, w, &mut b, size, ty);
                    fast_sq += a.iter().map(|&v| (v as f64).powi(2)).sum::<f64>();
                    aom_sq += b.iter().map(|&v| (v as f64).powi(2)).sum::<f64>();
                }
                let ratio = (fast_sq / aom_sq).sqrt();
                lo = lo.min(ratio);
                hi = hi.max(ratio);
                worst = worst.max((ratio - 1.0).abs());
                fast_all += fast_sq;
                aom_all += aom_sq;
            }
            report += &format!(
                "{size:?}: {:.4} (types {lo:.4}..{hi:.4})\n",
                (fast_all / aom_all).sqrt()
            );
        }
        println!("{report}");
        assert!(worst < 0.01, "scale off by {worst:.4}:\n{report}");
    }

    #[test]
    #[ignore]
    fn bench_forward() {
        use std::hint::black_box;
        use std::time::Instant;
        for size in sizes() {
            let (w, h) = dimensions(size);
            let mut rng = Rng(size as u64 + 1);
            let block: Vec<i16> = (0..w * h).map(|_| rng.residual(64)).collect();
            let mut coefs = vec![0; cw_ch(size)];
            let iters = 2_000_000 / (w * h);
            let time = |f: &mut dyn FnMut(&mut [i32])| {
                let mut coefs = coefs.clone();
                let t = Instant::now();
                for _ in 0..iters {
                    f(black_box(&mut coefs));
                }
                t.elapsed().as_nanos() as f64 / iters as f64
            };
            let aom = time(&mut |c| fdct::forward(black_box(&block), w, c, size, 0));
            let scalar = time(&mut |c| forward_in(Lane::Scalar, black_box(&block), w, c, size, 0));
            let simd = time(&mut |c| forward_in(*lanes().last().unwrap(), black_box(&block), w, c, size, 0));
            coefs[0] = 0;
            println!("{size:?}: libaom port {aom:8.0} ns, scalar {scalar:8.0} ns, simd {simd:8.0} ns");
        }
    }
}
