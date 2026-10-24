//! Bounds tests for `fdct_fast`: exact-size slices between guard pages, an affine bound
//! on every scalar intermediate (overflow is the only way the lanes can disagree), heavier
//! lane comparisons and a layout check against `fdct.rs`.

use std::cell::Cell;

use super::{dimensions, forward_in, Config, Lane, Txfm1d};
use crate::enc::fdct::{self, coded_dimensions};
use crate::enc::fdct::tests::{decodable, sizes, Rng};
use crate::levels::{TxfmSize, TxfmType, WHT_WHT};

fn pairs() -> Vec<(TxfmSize, TxfmType)> {
    sizes()
        .flat_map(|size| (0..WHT_WHT).filter(move |&ty| decodable(size, ty)).map(move |ty| (size, ty)))
        .collect()
}

fn simd_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    return is_x86_feature_detected!("avx2");
    #[cfg(target_arch = "aarch64")]
    return true;
    #[allow(unreachable_code)]
    false
}

struct Guarded {
    base: *mut u8,
    total: usize,
    page: usize,
    pages: usize,
}

impl Guarded {
    /// `[guard][data pages][guard]`.
    fn new(bytes: usize) -> Self {
        unsafe {
            let page = libc::sysconf(libc::_SC_PAGESIZE) as usize;
            let pages = bytes.div_ceil(page).max(1);
            let total = (pages + 2) * page;
            let base = libc::mmap(
                std::ptr::null_mut(),
                total,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            );
            assert!(base != libc::MAP_FAILED);
            let base = base as *mut u8;
            assert_eq!(libc::mprotect(base as *mut _, page, libc::PROT_NONE), 0);
            assert_eq!(libc::mprotect(base.add((pages + 1) * page) as *mut _, page, libc::PROT_NONE), 0);
            Self { base, total, page, pages }
        }
    }

    /// `n` values ending at the trailing guard page.
    fn tail<T>(&mut self, n: usize) -> &mut [T] {
        unsafe {
            let end = self.base.add((self.pages + 1) * self.page);
            std::slice::from_raw_parts_mut((end as *mut T).sub(n), n)
        }
    }

    /// `n` values starting right after the leading guard page.
    fn head<T>(&mut self, n: usize) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.base.add(self.page) as *mut T, n) }
    }
}

impl Drop for Guarded {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.base as *mut _, self.total) };
    }
}

/// Exactly `pw * ph` residuals and `cw * ch` coefficients, as `rd_residual` passes them, with
/// an unmapped page right after (and, in the second layout, right before) each slice. A read
/// or write outside them is a SIGSEGV/SIGBUS, not a silent pass.
#[test]
fn exact_slices_between_guard_pages() {
    let lanes: Vec<Lane> = if simd_available() { vec![Lane::Scalar, Lane::Simd] } else { vec![Lane::Scalar] };
    let mut checked = 0;
    for (size, ty) in pairs() {
        let (w, h) = dimensions(size);
        let (cw, ch) = coded_dimensions(size);
        let mut rng = Rng(0x3c6e_f372_fe94_f82b ^ ((size as u64) << 8 | ty as u64));
        for rep in 0..8 {
            let block: Vec<i16> = (0..w * h).map(|_| rng.residual(255)).collect();
            let mut want = vec![0i32; cw * ch];
            forward_in(Lane::Scalar, &block, w, &mut want, size, ty);
            for &lane in &lanes {
                for at_tail in [true, false] {
                    let mut rg = Guarded::new(w * h * 2);
                    let mut cg = Guarded::new(cw * ch * 4);
                    let residual: &mut [i16] = if at_tail { rg.tail(w * h) } else { rg.head(w * h) };
                    residual.copy_from_slice(&block);
                    let coefs: &mut [i32] = if at_tail { cg.tail(cw * ch) } else { cg.head(cw * ch) };
                    coefs.fill(0x5a5a_5a5a);
                    forward_in(lane, residual, w, coefs, size, ty);
                    assert!(coefs == &want[..], "{size:?} {ty} {lane:?} tail={at_tail} rep={rep}: differs");
                    checked += 1;
                }
            }
        }
    }
    println!("guard-page runs: {checked}");
}

thread_local!(static PEAK: Cell<f64> = const { Cell::new(0.0) });

fn note(v: f64) {
    PEAK.with(|p| p.set(p.get().max(v)));
}

/// A value as an affine form over the block's inputs: `|v - Σ c_i x_i| <= err`, `|x_i| <= 1`.
#[derive(Clone, Copy)]
struct Af {
    c: [f64; 64],
    err: f64,
}

impl Af {
    fn bound(&self) -> f64 {
        self.c.iter().map(|v| v.abs()).sum::<f64>() + self.err
    }

    fn map2(self, b: Self, f: impl Fn(f64, f64) -> f64) -> Self {
        let mut c = [0.0; 64];
        for i in 0..64 {
            c[i] = f(self.c[i], b.c[i]);
        }
        Af { c, err: self.err + b.err }
    }

    fn scale(mut self, k: f64, round_err: f64) -> Self {
        for v in &mut self.c {
            *v *= k;
        }
        self.err = self.err * k.abs() + round_err;
        self
    }
}

mod affine {
    use super::{note, Af, Txfm1d};

    type TxfmFunc = fn(&mut [Af]);

    impl_1d_tx!();

    impl TxOperations for Af {
        fn zero() -> Self {
            Af { c: [0.0; 64], err: 0.0 }
        }

        fn tx_mul<const SHIFT: i32>(self, mul: i32) -> Self {
            let product = self.bound() * (mul as f64).abs();
            note(product);
            note(product + (1i64 << SHIFT >> 1) as f64);
            let r = self.scale(mul as f64 / (1i64 << SHIFT) as f64, 0.5);
            note(r.bound());
            r
        }

        fn rshift1(self) -> Self {
            note(self.bound() + 1.0);
            self.scale(0.5, 0.5)
        }

        fn add(self, b: Self) -> Self {
            let r = self.map2(b, |x, y| x + y);
            note(r.bound());
            r
        }

        fn sub(self, b: Self) -> Self {
            let r = self.map2(b, |x, y| x - y);
            note(r.bound());
            r
        }

        fn add_avg(self, b: Self) -> Self {
            self.add(b).scale(0.5, 0.5)
        }

        fn sub_avg(self, b: Self) -> Self {
            self.sub(b).scale(0.5, 0.5)
        }
    }

    pub(super) fn func(t: Txfm1d) -> TxfmFunc {
        get_func(t)
    }
}

fn round_shift_af(arr: &mut [Af], bit: i8) {
    if bit > 0 {
        for v in arr.iter_mut() {
            note(v.bound() + (1i64 << (bit - 1)) as f64);
            *v = v.scale(1.0 / (1i64 << bit) as f64, 0.5);
        }
    }
}

fn symbols(n: usize, bound: impl Fn(usize) -> f64) -> Vec<Af> {
    (0..n)
        .map(|i| {
            let mut c = [0.0; 64];
            c[i] = bound(i);
            Af { c, err: 0.0 }
        })
        .collect()
}

/// The largest magnitude any scalar `i32` intermediate (products before the rounding shift
/// included) can reach on a residual in ±255. Columns are independent, so the row pass's
/// inputs range over a box and a fresh form per column is sound.
fn peak(size: TxfmSize, ty: TxfmType) -> f64 {
    let cfg = Config::new(size, ty);
    let (w, h) = (cfg.w, cfg.h);
    let ch = h.min(32);
    PEAK.with(|p| p.set(0.0));
    let x = (255i64 << cfg.shift[0]) as f64;
    note(x);
    let mut col = symbols(h, |_| x);
    affine::func(cfg.col)(&mut col);
    round_shift_af(&mut col, -cfg.shift[1]);
    for r in 0..ch {
        let b = col[r].bound();
        let mut row = symbols(w, |_| b);
        affine::func(cfg.row)(&mut row);
        round_shift_af(&mut row, -cfg.shift[2]);
    }
    PEAK.with(|p| p.get())
}

/// Below this, no scalar intermediate overflows, so the SIMD lanes' wrapping
/// (`mullo`, `add`) and non-wrapping (`vhadd`, `vrshr`) forms equal the scalar's.
#[test]
fn intermediates_fit_i32() {
    let mut report = String::new();
    let mut worst = (0.0f64, None);
    for (size, ty) in pairs() {
        let p = peak(size, ty);
        if p > worst.0 {
            worst = (p, Some((size, ty)));
        }
        report += &format!("{size:?} {ty}: {p:.0} ({:.2} bits)\n", p.log2());
    }
    println!("{report}worst {:?}: {:.0} = 2^{:.3}", worst.1, worst.0, worst.0.log2());
    assert!(worst.0 < i32::MAX as f64, "an intermediate can overflow i32");
}

fn assert_lanes_agree(size: TxfmSize, ty: TxfmType, block: &[i16], pad: usize, junk: i16) {
    let (w, h) = dimensions(size);
    let n = {
        let (cw, ch) = coded_dimensions(size);
        cw * ch
    };
    let stride = w + pad;
    let mut wide = vec![junk; stride * h];
    wide.truncate((h - 1) * stride + w);
    for r in 0..h {
        wide[r * stride..][..w].copy_from_slice(&block[r * w..][..w]);
    }
    let mut scalar = vec![i32::MIN; n];
    let mut simd = vec![i32::MAX; n];
    forward_in(Lane::Scalar, block, w, &mut scalar, size, ty);
    forward_in(Lane::Simd, &wide, stride, &mut simd, size, ty);
    if scalar != simd {
        let i = scalar.iter().zip(&simd).position(|(a, b)| a != b).unwrap();
        panic!("{size:?} {ty} pad {pad}: lanes differ at {i}: {} vs {}", scalar[i], simd[i]);
    }
}

/// Many more blocks than `simd_matches_scalar`: uniform, ±255 signs only, sparse spikes,
/// rows/columns constant, and strides with odd padding filled with junk.
#[test]
fn simd_matches_scalar_heavy() {
    if !simd_available() {
        println!("no SIMD lane");
        return;
    }
    let mut runs = 0usize;
    for (size, ty) in pairs() {
        let (w, h) = dimensions(size);
        let mut rng = Rng(0xa54f_f53a_5f1d_36f1 ^ ((size as u64) << 8 | ty as u64));
        let reps = (300_000 / (w * h)).clamp(64, 4096);
        for k in 0..reps {
            let block: Vec<i16> = match k % 5 {
                0 => (0..w * h).map(|_| rng.residual(255)).collect(),
                1 => (0..w * h).map(|_| if rng.next() & 1 == 0 { 255 } else { -255 }).collect(),
                2 => (0..w * h).map(|_| if rng.next() % 16 == 0 { rng.residual(255) } else { 0 }).collect(),
                3 => {
                    let line: Vec<i16> = (0..w).map(|_| if rng.next() & 1 == 0 { 255 } else { -255 }).collect();
                    (0..w * h).map(|i| line[i % w]).collect()
                }
                _ => {
                    let line: Vec<i16> = (0..h).map(|_| if rng.next() & 1 == 0 { 255 } else { -255 }).collect();
                    (0..w * h).map(|i| line[i / w]).collect()
                }
            };
            let pad = [0, 1, 3, 8, 17][k % 5];
            assert_lanes_agree(size, ty, &block, pad, i16::MIN);
            runs += 1;
        }
    }
    println!("lane comparisons: {runs}");
}

/// `coefs[x * ch + y]` and the flips: on random blocks the fast coefficients must point the
/// same way as `fdct.rs`'s (normalized correlation near 1). A transposed layout, a swapped
/// flip or a wrong 64-point half drops it far below.
#[test]
fn layout_matches_libaom() {
    let mut report = String::new();
    let mut worst = (1.0f64, None);
    for (size, ty) in pairs() {
        let (w, h) = dimensions(size);
        let (cw, ch) = coded_dimensions(size);
        let mut rng = Rng(0x510e_527f_ade6_82d1 ^ ((size as u64) << 8 | ty as u64));
        let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
        let mut a = vec![0; cw * ch];
        let mut b = vec![0; cw * ch];
        for _ in 0..32 {
            let block: Vec<i16> = (0..w * h).map(|_| rng.residual(255)).collect();
            forward_in(Lane::Scalar, &block, w, &mut a, size, ty);
            fdct::forward(&block, w, &mut b, size, ty);
            for (&x, &y) in a.iter().zip(&b) {
                ab += x as f64 * y as f64;
                aa += (x as f64).powi(2);
                bb += (y as f64).powi(2);
            }
        }
        let r = ab / (aa * bb).sqrt();
        if r < worst.0 {
            worst = (r, Some((size, ty)));
        }
        report += &format!("{size:?} {ty}: {r:.5}\n");
    }
    println!("{report}worst {:?}: {:.5}", worst.1, worst.0);
    assert!(worst.0 > 0.999, "layout differs from fdct.rs");
}
