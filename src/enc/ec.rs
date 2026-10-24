//! The range encoder that `msac.rs` decodes, ported from libaom's `entenc.c`
//! (BSD-2, Alliance for Open Media). CDFs use rav1d's layout: `n` probabilities
//! of "the symbol is greater than i" in Q15, then the adaptation counter.

const EC_PROB_SHIFT: u32 = 6;
const EC_MIN_PROB: u32 = 4;

pub struct Writer {
    buf: Vec<u8>,
    low: u64,
    rng: u32,
    cnt: i32,
}

impl Default for Writer {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl Writer {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buf: Vec::with_capacity(capacity),
            low: 0,
            rng: 0x8000,
            cnt: -9,
        }
    }

    fn carry(&mut self) {
        for byte in self.buf.iter_mut().rev() {
            let (sum, overflow) = byte.overflowing_add(1);
            *byte = sum;
            if !overflow {
                return;
            }
        }
        unreachable!("carry out of the first byte");
    }

    fn normalize(&mut self, mut low: u64, rng: u32) {
        debug_assert!(rng <= 0xffff);
        let d = rng.leading_zeros() as i32 - 16;
        let mut c = self.cnt;
        let mut s = c + d;
        if s >= 40 {
            let ready = (s >> 3) + 1;
            c += 24 - (ready << 3);
            let mut output = low >> c;
            low &= (1 << c) - 1;
            let top = 1u64 << (ready << 3);
            if output & top != 0 {
                self.carry();
            }
            output &= top - 1;
            for i in (0..ready).rev() {
                self.buf.push((output >> (i << 3)) as u8);
            }
            s = c + d - 24;
        }
        self.low = low << d;
        self.rng = rng << d;
        self.cnt = s;
    }

    fn scale(&self, f: u32) -> u32 {
        (self.rng >> 8) * (f >> EC_PROB_SHIFT) >> (7 - EC_PROB_SHIFT)
    }

    fn encode(&mut self, fl: u32, fh: u32, s: u32, n: u32) {
        let r = self.rng;
        let v = self.scale(fh) + EC_MIN_PROB * (n - s);
        if fl < 0x8000 {
            let u = self.scale(fl) + EC_MIN_PROB * (n - s + 1);
            self.normalize(self.low + (r - u) as u64, u - v);
        } else {
            self.normalize(self.low, r - v);
        }
    }

    /// `n` is the number of symbols minus one, as `msac.rs` takes it.
    pub fn symbol(&mut self, s: u8, cdf: &mut [u16], n: u8) {
        self.symbol_fixed(s, cdf, n);
        adapt(s, cdf, n);
    }

    pub fn symbol_fixed(&mut self, s: u8, cdf: &[u16], n: u8) {
        debug_assert!(s <= n && n < 16);
        let (s, n) = (s as usize, n as usize);
        let fl = if s > 0 { cdf[s - 1] as u32 } else { 0x8000 };
        let fh = if s < n { cdf[s] as u32 } else { 0 };
        self.encode(fl, fh, s as u32, n as u32);
    }

    pub fn bool(&mut self, bit: bool, f: u32) {
        let r = self.rng;
        let v = self.scale(f) + EC_MIN_PROB;
        if bit {
            self.normalize(self.low + (r - v) as u64, v);
        } else {
            self.normalize(self.low, r - v);
        }
    }

    pub fn bool_adapt(&mut self, bit: bool, cdf: &mut [u16; 2]) {
        self.bool(bit, cdf[0] as u32);
        let count = cdf[1];
        let rate = 4 + (count >> 4);
        if bit {
            cdf[0] += (1 << 15) - cdf[0] >> rate;
        } else {
            cdf[0] -= cdf[0] >> rate;
        }
        cdf[1] = count + (count < 32) as u16;
    }

    pub fn bool_equi(&mut self, bit: bool) {
        self.bool(bit, 1 << 14);
    }

    /// `v` in `n` bits, most significant first.
    pub fn bools(&mut self, v: u32, n: u8) {
        for i in (0..n).rev() {
            self.bool_equi(v >> i & 1 != 0);
        }
    }

    pub fn uniform(&mut self, v: u32, n: u32) {
        debug_assert!(v < n);
        let l = (31 - n.leading_zeros()) as u8 + 1;
        let m = (1 << l) - n;
        if v < m {
            self.bools(v, l - 1);
        } else {
            let w = v + m;
            self.bools(w >> 1, l - 1);
            self.bool_equi(w & 1 != 0);
        }
    }

    pub fn subexp(&mut self, v: u32, r#ref: u32, n: u32, k: u8) {
        debug_assert!(n >> k == 8 && v < n);
        let v = if r#ref * 2 <= n {
            recenter(r#ref, v)
        } else {
            recenter(n - 1 - r#ref, n - 1 - v)
        };
        if v < 1 << k {
            self.bool_equi(false);
            self.bools(v, k);
        } else if v < 2 << k {
            self.bools(0b10, 2);
            self.bools(v - (1 << k), k);
        } else if v < 4 << k {
            self.bools(0b110, 3);
            self.bools(v - (2 << k), k + 1);
        } else {
            self.bools(0b111, 3);
            self.bools(v - (4 << k), k + 2);
        }
    }

    /// A coefficient token from 3 to 15, as `rav1d_msac_decode_hi_tok` reads it.
    pub fn hi_tok(&mut self, tok: u8, cdf: &mut [u16; 4]) {
        debug_assert!((3..16).contains(&tok));
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

    /// Bits written so far, rounded up, as libaom's `od_ec_enc_tell`.
    pub fn tell(&self) -> u32 {
        (self.cnt + 10) as u32 + self.buf.len() as u32 * 8
    }

    pub fn finish(mut self) -> Vec<u8> {
        let mut c = self.cnt;
        let m = 0x3fff;
        let mut e = ((self.low + m) & !m) | (m + 1);
        let mut s = c + 10;
        let mut n = (1u64 << (c + 16)) - 1;
        while s > 0 {
            let v = (e >> (c + 16)) as u16;
            if v & 0x100 != 0 {
                self.carry();
            }
            self.buf.push(v as u8);
            e &= n;
            s -= 8;
            c -= 8;
            n >>= 8;
        }
        self.buf
    }
}

pub fn adapt(s: u8, cdf: &mut [u16], n: u8) {
    let n = n as usize;
    let count = cdf[n];
    let rate = 4 + (count >> 4) + (n > 2) as u16;
    let s = s as usize;
    for p in &mut cdf[..s] {
        *p += (1 << 15) - *p >> rate;
    }
    for p in &mut cdf[s..n] {
        *p -= *p >> rate;
    }
    cdf[n] = count + (count < 32) as u16;
}

fn recenter(r: u32, v: u32) -> u32 {
    if v > r << 1 {
        v
    } else if v >= r {
        (v - r) << 1
    } else {
        ((r - v) << 1) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c_arc::CArc;
    use crate::c_box::CRef;
    use crate::cpu::rav1d_get_cpu_flags;
    use crate::msac::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 32) as u32
        }

        fn below(&mut self, n: u32) -> u32 {
            self.next() % n
        }
    }

    #[derive(Clone)]
    enum Op {
        Symbol { s: u8, n: u8, cdf: [u16; 16] },
        Fixed { s: u8, n: u8, cdf: [u16; 16] },
        BoolAdapt { bit: bool, cdf: [u16; 2] },
        Bool { bit: bool, f: u32 },
        Equi(bool),
        Bools { v: u32, n: u8 },
        Uniform { v: u32, n: u32 },
        Subexp { v: u32, r#ref: u32, n: u32, k: u8 },
        HiTok { tok: u8, cdf: [u16; 4] },
    }

    /// Probabilities stay within 1..32768 as in every AV1 CDF: the decoder's
    /// vector adaptation works in signed 16 bits and wraps on a 0.
    fn random_cdf(rng: &mut Rng, n: u8, skewed: bool) -> [u16; 16] {
        let mut cdf = [0u16; 16];
        for p in &mut cdf[..n as usize] {
            *p = if skewed {
                [1, 2, 31, 32_767, 32_736][rng.below(5) as usize]
            } else {
                1 + rng.below(32_767) as u16
            };
        }
        cdf[..n as usize].sort_unstable_by(|a, b| b.cmp(a));
        cdf[n as usize] = rng.below(33) as u16;
        cdf
    }

    fn random_ops(rng: &mut Rng, len: usize, skewed: bool, kinds: &[u32]) -> Vec<Op> {
        (0..len)
            .map(|_| match kinds[rng.below(kinds.len() as u32) as usize] {
                0 => {
                    let n = 1 + rng.below(15) as u8;
                    let s = rng.below(n as u32 + 1) as u8;
                    Op::Symbol { s, n, cdf: random_cdf(rng, n, skewed) }
                }
                1 => {
                    let n = 1 + rng.below(15) as u8;
                    let s = rng.below(n as u32 + 1) as u8;
                    Op::Fixed { s, n, cdf: random_cdf(rng, n, skewed) }
                }
                2 => {
                    let p = random_cdf(rng, 1, skewed)[0];
                    Op::BoolAdapt { bit: rng.below(2) == 1, cdf: [p, rng.below(33) as u16] }
                }
                3 => Op::Bool { bit: rng.below(2) == 1, f: 1 + rng.below(32_767) },
                4 => Op::Equi(rng.below(2) == 1),
                5 => {
                    let n = rng.below(25) as u8;
                    Op::Bools { v: rng.next() & ((1u64 << n) - 1) as u32, n }
                }
                6 => {
                    let n = 2 + rng.below(1000);
                    Op::Uniform { v: rng.below(n), n }
                }
                7 => {
                    let k = rng.below(6) as u8;
                    let n = 8 << k;
                    Op::Subexp { v: rng.below(n), r#ref: rng.below(n), n, k }
                }
                _ => {
                    let mut cdf = [0u16; 4];
                    cdf[..4].copy_from_slice(&random_cdf(rng, 3, skewed)[..4]);
                    Op::HiTok { tok: 3 + rng.below(13) as u8, cdf }
                }
            })
            .collect()
    }

    fn encode(ops: &[Op]) -> Vec<u8> {
        let mut w = Writer::default();
        let mut ops = ops.to_vec();
        for op in &mut ops {
            match op {
                Op::Symbol { s, n, cdf } => w.symbol(*s, cdf, *n),
                Op::Fixed { s, n, cdf } => w.symbol_fixed(*s, cdf, *n),
                Op::BoolAdapt { bit, cdf } => w.bool_adapt(*bit, cdf),
                Op::Bool { bit, f } => w.bool(*bit, *f),
                Op::Equi(bit) => w.bool_equi(*bit),
                Op::Bools { v, n } => w.bools(*v, *n),
                Op::Uniform { v, n } => w.uniform(*v, *n),
                Op::Subexp { v, r#ref, n, k } => w.subexp(*v, *r#ref, *n, *k),
                Op::HiTok { tok, cdf } => w.hi_tok(*tok, cdf),
            }
        }
        w.finish()
    }

    fn decoder(data: Vec<u8>) -> MsacContext {
        let dsp = Rav1dMsacDSPContext::new(rav1d_get_cpu_flags());
        let data = CArc::wrap(CRef::Box(data.into_boxed_slice())).unwrap();
        MsacContext::new(data, false, &dsp)
    }

    /// The decoder's assembly loads CDFs with aligned vector loads.
    #[repr(C, align(32))]
    struct Aligned<T>(T);

    fn decode_symbol(d: &mut MsacContext, cdf: &mut [u16; 16], n: u8) -> u8 {
        let mut a = Aligned(*cdf);
        let s = match n {
            0..=3 => rav1d_msac_decode_symbol_adapt4(d, &mut a.0, n),
            4..=7 => rav1d_msac_decode_symbol_adapt8(d, &mut a.0, n),
            _ => rav1d_msac_decode_symbol_adapt16(d, &mut a.0, n),
        };
        *cdf = a.0;
        s
    }

    fn decode_bool_adapt(d: &mut MsacContext, cdf: &mut [u16; 2]) -> bool {
        let mut a = Aligned(*cdf);
        let bit = rav1d_msac_decode_bool_adapt(d, &mut a.0);
        *cdf = a.0;
        bit
    }

    fn decode_hi_tok(d: &mut MsacContext, cdf: &mut [u16; 4]) -> u8 {
        let mut a = Aligned(*cdf);
        let tok = rav1d_msac_decode_hi_tok(d, &mut a.0);
        *cdf = a.0;
        tok
    }

    fn check(ops: &[Op]) {
        let mut d = decoder(encode(ops));
        for (i, op) in ops.iter().enumerate() {
            match op.clone() {
                Op::Symbol { s, n, mut cdf } => {
                    let mut ours = cdf;
                    adapt(s, &mut ours, n);
                    assert_eq!(decode_symbol(&mut d, &mut cdf, n), s, "op {i}");
                    assert_eq!(cdf[..=n as usize], ours[..=n as usize], "op {i}: adaptation");
                }
                Op::Fixed { s, n, mut cdf } => {
                    assert_eq!(decode_symbol(&mut d, &mut cdf, n), s, "op {i}")
                }
                Op::BoolAdapt { bit, mut cdf } => {
                    let mut ours = cdf;
                    Writer::default().bool_adapt(bit, &mut ours);
                    assert_eq!(decode_bool_adapt(&mut d, &mut cdf), bit, "op {i}");
                    assert_eq!(cdf, ours, "op {i}: adaptation");
                }
                Op::Bool { bit, f } => assert_eq!(rav1d_msac_decode_bool(&mut d, f), bit, "op {i}"),
                Op::Equi(bit) => assert_eq!(rav1d_msac_decode_bool_equi(&mut d), bit, "op {i}"),
                Op::Bools { v, n } => assert_eq!(rav1d_msac_decode_bools(&mut d, n), v, "op {i}"),
                Op::Uniform { v, n } => {
                    assert_eq!(rav1d_msac_decode_uniform(&mut d, n), v as i32, "op {i}")
                }
                Op::Subexp { v, r#ref, n, k } => {
                    assert_eq!(rav1d_msac_decode_subexp(&mut d, r#ref, n, k), v as i32, "op {i}")
                }
                Op::HiTok { tok, mut cdf } => {
                    assert_eq!(decode_hi_tok(&mut d, &mut cdf), tok, "op {i}")
                }
            }
        }
    }

    fn rounds(kinds: &[u32]) {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for round in 0..2000 {
            let len = [0, 1, 2, 7, 64, 1000][round % 6];
            let ops = random_ops(&mut rng, len, round % 3 == 0, kinds);
            check(&ops);
        }
    }

    #[test]
    fn symbols() {
        rounds(&[0, 1]);
    }

    #[test]
    fn adaptive_bools() {
        rounds(&[2]);
    }

    #[test]
    fn raw_bits() {
        rounds(&[3, 4, 5, 6, 7]);
    }

    #[test]
    fn coefficient_tokens() {
        rounds(&[8]);
    }

    #[test]
    fn everything_mixed() {
        rounds(&[0, 1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn adapting_one_cdf_over_a_long_run() {
        let mut rng = Rng(7);
        let mut cdf = random_cdf(&mut rng, 12, false);
        cdf[12] = 0;
        let symbols: Vec<u8> = (0..100_000).map(|_| (rng.below(13) * rng.below(2)) as u8).collect();
        let mut w = Writer::default();
        let mut ours = cdf;
        for &s in &symbols {
            w.symbol(s, &mut ours, 12);
        }
        let bits = w.tell();
        let data = w.finish();
        assert!(data.len() as u32 * 8 <= bits + 8);
        let mut d = decoder(data);
        for (i, &s) in symbols.iter().enumerate() {
            assert_eq!(decode_symbol(&mut d, &mut cdf, 12), s, "symbol {i}");
        }
        assert_eq!(cdf[..=12], ours[..=12]);
    }
}
