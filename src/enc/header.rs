//! Sequence and frame headers, OBUs and the `av1C` record, as `obu.rs` reads them.

#[derive(Default)]
pub struct BitWriter {
    buf: Vec<u8>,
    acc: u64,
    n: u32,
}

impl BitWriter {
    pub fn put(&mut self, v: u32, n: u32) {
        debug_assert!(n <= 32 && (n == 32 || v >> n == 0));
        self.acc = self.acc << n | v as u64;
        self.n += n;
        while self.n >= 8 {
            self.n -= 8;
            self.buf.push((self.acc >> self.n) as u8);
        }
    }

    pub fn bit(&mut self, b: bool) {
        self.put(b as u32, 1);
    }

    pub fn signed(&mut self, v: i32, n: u32) {
        self.put(v as u32 & ((1 << n) - 1), n);
    }

    pub fn align(&mut self) {
        if self.n > 0 {
            self.put(0, 8 - self.n);
        }
    }

    pub fn trailing(&mut self) {
        self.bit(true);
        self.align();
    }

    pub fn finish(mut self) -> Vec<u8> {
        self.align();
        self.buf
    }
}

pub const OBU_SEQUENCE_HEADER: u8 = 1;
pub const OBU_FRAME: u8 = 6;

pub fn obu(kind: u8, payload: &[u8], out: &mut Vec<u8>) {
    out.push(kind << 3 | 2);
    let mut n = payload.len();
    loop {
        let byte = (n & 0x7f) as u8;
        n >>= 7;
        if n == 0 {
            out.push(byte);
            break;
        }
        out.push(byte | 0x80);
    }
    out.extend_from_slice(payload);
}

#[derive(Clone, Copy)]
pub struct Colour {
    pub primaries: u8,
    pub transfer: u8,
    pub matrix: u8,
}

#[derive(Clone)]
pub struct Sequence {
    pub width: u32,
    pub height: u32,
    pub mono: bool,
    pub colour: Option<Colour>,
    pub full_range: bool,
    pub filter_intra: bool,
    pub edge_filter: bool,
    pub cdef: bool,
    pub restoration: bool,
}

impl Sequence {
    pub fn profile(&self) -> u8 {
        if self.mono {
            0
        } else {
            1
        }
    }

    /// The lowest level whose picture size limits hold, or 31 (none).
    pub fn level(&self) -> u8 {
        const LEVELS: [(u8, u32, u32, u32); 6] = [
            (0, 147_456, 2048, 1152),
            (1, 278_784, 2816, 1584),
            (4, 665_856, 4352, 2448),
            (5, 1_065_024, 5504, 3096),
            (8, 2_359_296, 6144, 3456),
            (12, 8_912_896, 8192, 4352),
        ];
        let area = self.width * self.height;
        LEVELS
            .iter()
            .chain(&[(16, 35_651_584, 16384, 8704)])
            .find(|&&(_, size, w, h)| area <= size && self.width <= w && self.height <= h)
            .map_or(31, |l| l.0)
    }

    pub fn write(&self) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.put(self.profile() as u32, 3);
        w.bit(true);
        w.bit(true);
        w.put(self.level() as u32, 5);
        let wbits = 32 - (self.width - 1).max(1).leading_zeros();
        let hbits = 32 - (self.height - 1).max(1).leading_zeros();
        w.put(wbits - 1, 4);
        w.put(hbits - 1, 4);
        w.put(self.width - 1, wbits);
        w.put(self.height - 1, hbits);
        w.bit(false);
        w.bit(self.filter_intra);
        w.bit(self.edge_filter);
        w.bit(false);
        w.bit(self.cdef);
        w.bit(self.restoration);
        w.bit(false);
        if self.profile() == 0 {
            w.bit(self.mono);
        }
        w.bit(self.colour.is_some());
        if let Some(c) = self.colour {
            w.put(c.primaries as u32, 8);
            w.put(c.transfer as u32, 8);
            w.put(c.matrix as u32, 8);
        }
        let srgb_identity = self
            .colour
            .is_some_and(|c| (c.primaries, c.transfer, c.matrix) == (1, 13, 0));
        if self.mono || !srgb_identity {
            w.bit(self.full_range);
        }
        if !self.mono {
            w.bit(false);
        }
        w.bit(false);
        w.trailing();
        w.finish()
    }

    /// The `av1C` box payload: the record and the sequence header OBU.
    pub fn config(&self) -> Vec<u8> {
        let sampling = if self.mono { 0b1_1_1_00 } else { 0 };
        let mut out = vec![0x81, self.profile() << 5 | self.level(), sampling, 0];
        obu(OBU_SEQUENCE_HEADER, &self.write(), &mut out);
        out
    }
}

#[derive(Clone, Copy, Default)]
pub struct Quant {
    pub base: u8,
    pub y_dc: i8,
    pub u_dc: i8,
    pub u_ac: i8,
    pub qm: Option<[u8; 3]>,
}

impl Quant {
    pub fn lossless(&self) -> bool {
        self.base == 0 && self.y_dc == 0 && self.u_dc == 0 && self.u_ac == 0
    }
}

#[derive(Clone, Copy, Default)]
pub struct Cdef {
    pub damping: u8,
    pub bits: u8,
    pub y: [u8; 8],
    pub uv: [u8; 8],
}

#[derive(Clone, Copy, Default)]
pub struct Restoration {
    /// Coded `lr_type` per plane: 0 none, 1 switchable, 2 Wiener, 3 self-guided.
    pub types: [u8; 3],
    /// Unit size 64 << shift.
    pub shift: u8,
}

#[derive(Clone, Copy)]
pub struct Tiling {
    pub log2_cols: u8,
    pub log2_rows: u8,
    pub min_log2_cols: u8,
    pub max_log2_cols: u8,
    pub min_log2_rows: u8,
    pub max_log2_rows: u8,
    /// Superblocks per tile, across and down.
    pub tile_w: u32,
    pub tile_h: u32,
    pub sbw: u32,
    pub sbh: u32,
}

fn tile_log2(sz: u32, tgt: u32) -> u8 {
    let mut k = 0;
    while sz << k < tgt {
        k += 1;
    }
    k
}

impl Tiling {
    /// The fewest uniform tiles the level limits allow.
    pub fn new(width: u32, height: u32) -> Self {
        let sbw = width.div_ceil(64);
        let sbh = height.div_ceil(64);
        let min_log2_cols = tile_log2(64, sbw);
        let max_log2_cols = tile_log2(1, sbw.min(64));
        let max_log2_rows = tile_log2(1, sbh.min(64));
        let min_log2_tiles = tile_log2(4096 * 2304 >> 12, sbw * sbh).max(min_log2_cols);
        let log2_cols = min_log2_cols;
        let min_log2_rows = min_log2_tiles.saturating_sub(log2_cols);
        let log2_rows = min_log2_rows;
        Self {
            log2_cols,
            log2_rows,
            min_log2_cols,
            max_log2_cols,
            min_log2_rows,
            max_log2_rows,
            tile_w: 1 + (sbw - 1 >> log2_cols),
            tile_h: 1 + (sbh - 1 >> log2_rows),
            sbw,
            sbh,
        }
    }

    pub fn cols(&self) -> u32 {
        self.sbw.div_ceil(self.tile_w)
    }

    pub fn rows(&self) -> u32 {
        self.sbh.div_ceil(self.tile_h)
    }

    fn write(&self, w: &mut BitWriter) {
        w.bit(true);
        for i in self.min_log2_cols..self.max_log2_cols {
            w.bit(i < self.log2_cols);
            if i == self.log2_cols {
                break;
            }
        }
        for i in self.min_log2_rows..self.max_log2_rows {
            w.bit(i < self.log2_rows);
            if i == self.log2_rows {
                break;
            }
        }
    }
}

#[derive(Clone)]
pub struct Frame {
    pub screen_content: bool,
    pub quant: Quant,
    pub delta_q_log2: Option<u8>,
    /// Loop filter levels: luma vertical, luma horizontal, u, v.
    pub lf: [u8; 4],
    pub sharpness: u8,
    pub cdef: Option<Cdef>,
    pub lr: Option<Restoration>,
    pub tx_select: bool,
    pub reduced_tx_set: bool,
    pub tiling: Tiling,
}

impl Frame {
    pub fn write(&self, seq: &Sequence, tile_size_bytes: u8, w: &mut BitWriter) {
        w.bit(false);
        w.bit(self.screen_content);
        if self.screen_content {
            w.bit(false);
        }
        w.bit(false);
        if self.screen_content {
            w.bit(false);
        }
        self.tiling.write(w);
        if self.tiling.log2_cols + self.tiling.log2_rows > 0 {
            w.put(0, (self.tiling.log2_cols + self.tiling.log2_rows) as u32);
            w.put(tile_size_bytes as u32 - 1, 2);
        }
        let q = &self.quant;
        w.put(q.base as u32, 8);
        let delta = |w: &mut BitWriter, d: i8| {
            w.bit(d != 0);
            if d != 0 {
                w.signed(d as i32, 7);
            }
        };
        delta(w, q.y_dc);
        if !seq.mono {
            delta(w, q.u_dc);
            delta(w, q.u_ac);
        }
        w.bit(q.qm.is_some());
        if let Some(qm) = q.qm {
            w.put(qm[0] as u32, 4);
            w.put(qm[1] as u32, 4);
        }
        w.bit(false);
        if q.base != 0 {
            w.bit(self.delta_q_log2.is_some());
            if let Some(log2) = self.delta_q_log2 {
                w.put(log2 as u32, 2);
                w.bit(false);
            }
        }
        let lossless = q.lossless();
        if !lossless {
            w.put(self.lf[0] as u32, 6);
            w.put(self.lf[1] as u32, 6);
            if !seq.mono && (self.lf[0] != 0 || self.lf[1] != 0) {
                w.put(self.lf[2] as u32, 6);
                w.put(self.lf[3] as u32, 6);
            }
            w.put(self.sharpness as u32, 3);
            w.bit(false);
            if seq.cdef {
                let c = self.cdef.unwrap_or(Cdef {
                    damping: 3,
                    ..Default::default()
                });
                w.put(c.damping as u32 - 3, 2);
                w.put(c.bits as u32, 2);
                for i in 0..1 << c.bits {
                    w.put(c.y[i] as u32, 6);
                    if !seq.mono {
                        w.put(c.uv[i] as u32, 6);
                    }
                }
            }
        }
        if !lossless && seq.restoration {
            let lr = self.lr.unwrap_or_default();
            w.put(lr.types[0] as u32, 2);
            if !seq.mono {
                w.put(lr.types[1] as u32, 2);
                w.put(lr.types[2] as u32, 2);
            }
            if lr.types != [0; 3] {
                w.bit(lr.shift > 0);
                if lr.shift > 0 {
                    w.bit(lr.shift > 1);
                }
            }
        }
        if !lossless {
            w.bit(self.tx_select);
        }
        w.bit(self.reduced_tx_set);
    }
}
