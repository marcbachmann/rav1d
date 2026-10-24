//! Rate and distortion: a sink that counts what the writer would spend, and libaom's λ.

use crate::dequant_tables::DAV1D_DQ_TBL;
use crate::enc::ec::{adapt, Writer};

/// Where symbols go: the range writer, or a counter of their cost.
pub trait Sink {
    /// `n` is the number of symbols minus one.
    fn symbol(&mut self, s: u8, cdf: &mut [u16], n: u8);
    fn bool_adapt(&mut self, bit: bool, cdf: &mut [u16; 2]);
    fn bool(&mut self, bit: bool, f: u32);
    fn bool_equi(&mut self, bit: bool);

    fn bools(&mut self, v: u32, n: u8) {
        for i in (0..n).rev() {
            self.bool_equi(v >> i & 1 != 0);
        }
    }

    fn hi_tok(&mut self, tok: u8, cdf: &mut [u16; 4]) {
        let mut rest = tok - 3;
        for _ in 0..4 {
            let br = rest.min(3);
            self.symbol(br, cdf, 3);
            if br < 3 {
                return;
            }
            rest -= 3;
        }
    }

    fn golomb(&mut self, v: u32) {
        let x = v + 1;
        let len = 31 - x.leading_zeros();
        for _ in 0..len {
            self.bool_equi(false);
        }
        self.bool_equi(true);
        for i in (0..len).rev() {
            self.bool_equi(x >> i & 1 != 0);
        }
    }
}

impl Sink for Writer {
    fn symbol(&mut self, s: u8, cdf: &mut [u16], n: u8) {
        Writer::symbol(self, s, cdf, n)
    }

    fn bool_adapt(&mut self, bit: bool, cdf: &mut [u16; 2]) {
        Writer::bool_adapt(self, bit, cdf)
    }

    fn bool(&mut self, bit: bool, f: u32) {
        Writer::bool(self, bit, f)
    }

    fn bool_equi(&mut self, bit: bool) {
        Writer::bool_equi(self, bit)
    }

    fn hi_tok(&mut self, tok: u8, cdf: &mut [u16; 4]) {
        Writer::hi_tok(self, tok, cdf)
    }
}

/// Bits in 1/512, as libaom counts them; the CDFs are read, never adapted.
#[derive(Default, Clone, Copy)]
pub struct Counter {
    pub bits: u32,
}

/// Cost in 1/512 bits of a probability `p`/32768.
#[inline]
pub fn cost(p: u32) -> u32 {
    static TABLE: std::sync::OnceLock<[u16; 1025]> = std::sync::OnceLock::new();
    let t = TABLE.get_or_init(|| {
        std::array::from_fn(|i| (-(i.max(1) as f64 / 1024.0).log2() * 512.0).round() as u16)
    });
    t[(p.clamp(1, 32768) + 16 >> 5) as usize] as u32
}

impl Sink for Counter {
    fn symbol(&mut self, s: u8, cdf: &mut [u16], n: u8) {
        let (s, n) = (s as usize, n as usize);
        let fl = if s > 0 { cdf[s - 1] as u32 } else { 32768 };
        let fh = if s < n { cdf[s] as u32 } else { 0 };
        self.bits += cost(fl - fh);
    }

    fn bool_adapt(&mut self, bit: bool, cdf: &mut [u16; 2]) {
        self.bool(bit, cdf[0] as u32);
    }

    fn bool(&mut self, bit: bool, f: u32) {
        self.bits += cost(if bit { f } else { 32768 - f });
    }

    fn bool_equi(&mut self, _: bool) {
        self.bits += 512;
    }

    fn bools(&mut self, _: u32, n: u8) {
        self.bits += 512 * n as u32;
    }
}

/// A counter that also adapts the CDFs it reads, for exact costs of a sequence.
pub struct Adapting(pub Counter);

impl Sink for Adapting {
    fn symbol(&mut self, s: u8, cdf: &mut [u16], n: u8) {
        self.0.symbol(s, cdf, n);
        adapt(s, cdf, n);
    }

    fn bool_adapt(&mut self, bit: bool, cdf: &mut [u16; 2]) {
        self.0.bool(bit, cdf[0] as u32);
        let mut c = [cdf[0], cdf[1]];
        Writer::default().bool_adapt(bit, &mut c);
        *cdf = c;
    }

    fn bool(&mut self, bit: bool, f: u32) {
        self.0.bool(bit, f);
    }

    fn bool_equi(&mut self, bit: bool) {
        self.0.bool_equi(bit);
    }
}

/// λ in squared error per bit, from libaom's key-frame `rdmult` for an 8-bit `qidx`, with the
/// tune-IQ weight when `iq`.
pub fn lambda(qidx: u8, iq: bool) -> f64 {
    let q = DAV1D_DQ_TBL[0][qidx as usize][0] as f64;
    let mut rdmult = (q * q * (3.3 + 0.0015 * q)).floor();
    if iq {
        let weight = (((255 - qidx as i32) * 3) / 4).clamp(0, 72) + 128;
        rdmult = (rdmult * weight as f64 / 128.0).floor();
    }
    rdmult.max(1.0) / 2048.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn costs_follow_probability() {
        assert_eq!(cost(32768), 0);
        assert_eq!(cost(16384), 512);
        assert_eq!(cost(8192), 1024);
        let mut c = Counter::default();
        c.bools(5, 3);
        c.bool_equi(true);
        assert_eq!(c.bits, 2048);
    }

    #[test]
    fn lambda_grows_with_q() {
        assert!(lambda(20, false) < lambda(100, false));
        assert!(lambda(100, false) < lambda(100, true));
        assert!((lambda(255, true) - lambda(255, false)).abs() < 1e-9);
    }
}
