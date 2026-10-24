//! An AV1 still-image encoder, see `doc/encoder.md`.

pub mod ec;
pub mod fdct;
pub mod fdct_fast;
pub mod frame;
pub mod header;
pub mod palette;
pub mod rd;

use std::cmp;

use crate::dequant_tables::DAV1D_DQ_TBL;
use frame::{Coded, Encoder, Rd};
use header::{Cdef, Colour, Frame, Quant, Sequence, Tiling};

/// The headers of a still picture at `qidx`, monochrome or 4:4:4 YCbCr in BT.601 full range,
/// with libaom's deblocking and CDEF strengths for its key frames at that qindex.
pub fn headers(width: u32, height: u32, mono: bool, qidx: u8) -> (Sequence, Frame) {
    let seq = Sequence {
        width,
        height,
        mono,
        colour: (!mono).then_some(Colour { primaries: 1, transfer: 13, matrix: 6 }),
        full_range: true,
        filter_intra: true,
        edge_filter: true,
        cdef: qidx != 0,
        restoration: false,
    };
    let q = DAV1D_DQ_TBL[0][qidx as usize][1] as i32;
    let level = ((q * 17563 - 421574 + (1 << 17)) >> 18).clamp(0, 63) as u8;
    let frame = Frame {
        screen_content: false,
        quant: Quant { base: qidx, ..Default::default() },
        delta_q_log2: None,
        lf: [level; 4],
        sharpness: 0,
        cdef: (qidx != 0).then(|| cdef_from_q(qidx)),
        lr: None,
        tx_select: qidx != 0,
        reduced_tx_set: false,
        tiling: Tiling::new(width, height),
    };
    (seq, frame)
}

/// libaom's `av1_pick_cdef_from_qp` for an intra frame of natural content: one strength.
fn cdef_from_q(qidx: u8) -> Cdef {
    let q = DAV1D_DQ_TBL[0][qidx as usize][1] as f32;
    let fit = |a: f32, b: f32, c: f32, max: i32| ((q * q * a + q * b + c).round() as i32).clamp(0, max) as u8;
    let y = fit(0.0000033731974, 0.008070594, 0.0187634, 15) * 4 + fit(0.0000029167343, 0.0027798624, 0.0079405, 3);
    let uv = fit(-0.0000130790995, 0.012892405, -0.00748388, 15) * 4 + fit(0.0000032651783, 0.00035520183, 0.00228092, 3);
    let mut c = Cdef { damping: 3 + (qidx >> 6), bits: 0, ..Default::default() };
    c.y[0] = y;
    c.uv[0] = uv;
    c
}

#[derive(Clone, Default)]
pub struct Options {
    /// 0 is lossless.
    pub qidx: u8,
    /// Luma modes of the SATD ranking that go on to rate and distortion, all of them when 0.
    pub keep_modes: usize,
    /// Chroma modes of the SATD ranking that go on to rate and distortion, all of them when 0.
    pub keep_chroma: usize,
    /// Screen content tools: palettes.
    pub screen: bool,
    /// H and V partitions: 0 never, 1 always, 2 where Split beat None.
    pub rect: u8,
    /// Transform types are searched for the chosen depth only.
    pub late_txtp: bool,
    /// See `Encoder::levels`: 0 is 64×64, 4 the 4×4 blocks of a split 8×8.
    pub levels: std::ops::Range<u8>,
    /// See `Encoder::prune`; 0 tries every partition.
    pub prune: f64,
    /// libaom's Variance Boost: lower qindexes for flat superblocks, as its tune IQ codes them.
    pub variance_boost: bool,
    /// See `Encoder::rdoq`.
    pub rdoq: usize,
    /// libaom's tune-IQ rate weighting: λ per 16×16 block by its activity over the picture's.
    pub ssim: bool,
    /// libaom's tune-IQ quantizer matrices: luma levels for all-intra, chroma levels for 4:4:4.
    pub quant_matrices: bool,
    /// libaom's tune-IQ chroma AC qindex for 4:4:4, half the base qindex higher, at most 24.
    pub chroma_delta_q: bool,
    /// libaom's tune-IQ weight on λ, up to 72/128 more at low qindexes.
    pub iq_rdmult: bool,
    /// See `Encoder::skip_prediction`; libaom predicts skips at every speed.
    pub skip_prediction: bool,
}

impl Options {
    /// The search for an effort of 0 (fastest) to 9, as libaom's `cpu-used` of 9 − effort.
    /// Measured on 256 px crops at qidx 120 against the full search (`doc/encoder-progress.md`).
    pub fn effort(qidx: u8, effort: u8, screen: bool) -> Self {
        let (keep_modes, keep_chroma, rect, levels, prune) = match effort {
            8.. => (0, 0, 1, 0..5, 0.0),
            6..=7 => (4, 0, 1, 0..5, 0.0),
            4..=5 => (4, 4, 2, 0..4, 0.0),
            2..=3 => (3, 3, 0, 0..4, 0.0),
            _ => (2, 2, 0, 0..4, 1.0),
        };
        let rdoq = if effort >= 8 { 8 } else { 0 };
        Self { qidx, keep_modes, keep_chroma, screen, rect, late_txtp: false, levels, prune, variance_boost: false, rdoq, ssim: false, quant_matrices: false, chroma_delta_q: false, iq_rdmult: false, skip_prediction: true }
    }
}

impl Options {
    /// Everything libaom's tune IQ changes that the encoder has.
    pub fn tune_iq(self) -> Self {
        Self { variance_boost: true, ssim: true, quant_matrices: true, chroma_delta_q: true, iq_rdmult: true, ..self }
    }
}

/// libaom's `aom_get_qmlevel_allintra`, tune IQ's range 2 to 10.
fn luma_qm_level(qidx: u8) -> u8 {
    match qidx {
        ..=40 => 10,
        ..=100 => 9,
        ..=160 => 8,
        ..=200 => 7,
        ..=220 => 6,
        ..=240 => 5,
        _ => 4,
    }
}

/// libaom's `aom_get_qmlevel_444_chroma`, tune IQ's range 2 to 10.
fn chroma_qm_level(qidx: u8) -> u8 {
    match qidx {
        ..=12 => 10,
        ..=24 => 9,
        ..=32 => 8,
        ..=36 => 7,
        ..=44 => 6,
        ..=48 => 5,
        ..=56 => 4,
        ..=88 => 3,
        _ => 2,
    }
}

/// A still picture of 8-bit planes: one plane is monochrome, three are 4:4:4 YCbCr, BT.601 full
/// range, sRGB.
pub fn still(width: u32, height: u32, planes: &[&[u8]], options: &Options) -> Coded {
    still_with(width, height, planes, options).finish()
}

/// `still` up to the last row pushed, the encoder kept for its statistics.
pub fn still_with(width: u32, height: u32, planes: &[&[u8]], options: &Options) -> Encoder<Rd> {
    let (seq, mut frame) = headers(width, height, planes.len() == 1, options.qidx);
    frame.screen_content = options.screen;
    let lossy = options.qidx != 0;
    if lossy && options.chroma_delta_q && planes.len() == 3 {
        frame.quant.u_ac = cmp::min(options.qidx / 2, 24) as i8;
    }
    if lossy && options.quant_matrices {
        let u = (options.qidx as i32 + frame.quant.u_ac as i32).clamp(0, 255) as u8;
        frame.quant.qm = Some([luma_qm_level(options.qidx), chroma_qm_level(u), chroma_qm_level(u)]);
    }
    let boost = options.variance_boost && options.qidx != 0;
    if boost {
        frame.delta_q_log2 = Some(match options.qidx {
            160.. => 3,
            120.. => 2,
            80.. => 1,
            _ => 0,
        });
    }
    let mut e = Encoder::new(seq, frame, Rd);
    e.iq = options.iq_rdmult;
    e.skip_prediction = options.skip_prediction;
    e.keep_modes = options.keep_modes;
    e.keep_chroma = options.keep_chroma;
    e.rect = options.rect;
    e.late_txtp = options.late_txtp;
    e.levels = options.levels.clone();
    e.variance_boost = boost;
    e.rdoq = options.rdoq;
    e.prune = options.prune;
    if options.ssim && options.qidx != 0 {
        e.ssim_reference = Some(frame::ssim_reference(width as usize, height as usize, planes[0]));
    }
    let w = width as usize;
    for y in 0..height as usize {
        let rows: Vec<&[u8]> = planes.iter().map(|p| &p[y * w..][..w]).collect();
        e.push(&rows);
    }
    e
}
