//! An AV1 still-image encoder, see `doc/encoder.md`.

pub mod contexts;
pub mod dist;
pub mod ec;
pub mod fdct;
pub mod fdct_fast;
pub mod filters;
pub mod frame;
pub mod header;
pub mod palette;
pub mod partition_cnn;
pub mod quant;
pub mod rd;

use std::cmp;
use std::sync::{Arc, Condvar, Mutex};

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
    /// H and V partitions: 0 never, 1 always, 2 where Split beat None, 3 also the one along the
    /// direction of None's luma mode where None did (libaom's `prune_rect_part_using_none_pred_mode`).
    pub rect: u8,
    /// Transform types are searched for the chosen depth only.
    pub late_txtp: bool,
    /// libaom's `use_reduced_intra_txset`: under 16×16 the four DCT and ADST types and the one
    /// of IDTX, V_DCT, H_DCT the luma mode suits; to effort 7 where the search is not full.
    pub reduced_txtp: bool,
    /// libaom's transform-domain distortion: transform types are ranked by their coefficients'
    /// error, and only the winner is reconstructed.
    pub tx_domain_dist: bool,
    /// libaom's `prune_chroma_modes_using_luma_winner`: chroma tries DC, SMOOTH, CfL and the luma
    /// winner's mode.
    pub uv_from_luma: bool,
    /// libaom's `disable_smooth_intra` and `prune_filter_intra_level` 1: no SMOOTH_V or SMOOTH_H,
    /// and filter intra only as DC or the filter mode of the best mode so far.
    pub prune_intra: bool,
    /// Luma modes are ranked by their DCT's quantization error and rate instead of SATD.
    pub rank_rd: bool,
    /// With `rank_rd`, only this many of the SATD ranking get the DCT estimate; all when 0. Six
    /// to effort 5 above qindex 64 (on 384 px crops at qindex 140: -16.8% time for -0.065 dB at
    /// equal bytes; at qindex 32 it cost -0.18 dB). Blocks over the picture edge rank all.
    pub rank_pre: usize,
    /// Transform depths the search tries below the largest transform of a block wholly inside
    /// the picture, at most 2; libaom's `intra_tx_size_search_init_depth` 1 tries one. Blocks
    /// over the edge try 2, which a 1×9000 line needs.
    pub tx_depths: u8,
    /// See `Encoder::levels`: 0 is 64×64, 4 the 4×4 blocks of a split 8×8.
    pub levels: std::ops::Range<u8>,
    /// See `Encoder::prune`; 0 tries every partition.
    pub prune: f64,
    /// libaom's Variance Boost: lower qindexes for flat superblocks, as its tune IQ codes them.
    pub variance_boost: bool,
    /// See `Encoder::cnn_prune`; to effort 3 2 for natural content and 1 for screen content, as
    /// libaom from cpu-used 5. At effort 4 it fails AVIF gates (a HEIC 44% larger, two flat
    /// halves -1.4 dB), so the shortcuts that cost gates start below it.
    pub cnn_prune: u8,
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
    /// libaom's tune-IQ filters: no CDEF up to qindex 32 (`CDEF_ADAPTIVE`), deblocking sharpness
    /// 7 up to qindex 112 and 1 up to 160 (adaptive sharpness).
    pub iq_filters: bool,
    /// Deblocking and CDEF strengths searched on the decoded picture (`filters::search`), as
    /// libaom's full-image picks at cpu-used 5 and below; else they come from the qindex.
    pub filter_search: bool,
    /// At least 2^this many tile columns where the picture is 4 superblocks wide or more, each
    /// coded on its own thread by `still_on`. Two cost photos over 512×512 about 0.5% more bytes
    /// at 0.02-0.05 dB more; screen content and smaller pictures stay one tile, where a tile's
    /// own CDFs cost a fixed 100-300 B (graph paper 431 -> 704 B, a 300×200 ramp -1.5 dB).
    pub tile_cols_log2: u8,
}

impl Options {
    /// The search for an effort of 0 (fastest) to 9, as libaom's `cpu-used` of 9 − effort.
    /// Measured on 256 px crops at qidx 120 against the full search (`doc/encoder-progress.md`);
    /// from effort 4 qindexes to 64 and pictures to 512×512 keep 4 luma modes ranked by
    /// `luma_estimate`, every chroma mode, and search transform types at every depth. To effort 5
    /// one transform depth is tried, as libaom at cpu-used 5 (`intra_tx_size_search_init_depth`).
    pub fn effort(qidx: u8, effort: u8, screen: bool, pixels: u64) -> Self {
        let (keep_modes, keep_chroma, rect, levels, prune) = match effort {
            8.. => (0, 0, 1, 0..5, 0.0),
            6..=7 => (4, 0, 1, 0..5, 0.0),
            4..=5 => (4, 4, 2, 0..4, 1.0),
            2..=3 => (3, 3, 0, 0..4, 2.0),
            _ => (2, 2, 0, 0..4, 4.0),
        };
        let levels = if effort <= 3 && qidx > 100 { levels.start..levels.end.min(3) } else { levels };
        // Text and graphics need the H or V partition along None's direction; photos gain 0.03 dB for 30% more time.
        let rect = if screen && rect == 2 { 3 } else { rect };
        let fine = effort >= 4 && (qidx <= 64 || pixels <= 512 * 512);
        let (keep_modes, keep_chroma) = if fine { (4, 0) } else { (keep_modes, keep_chroma) };
        let rdoq = if effort >= 8 { 8 } else { 0 };
        Self { qidx, keep_modes, keep_chroma, screen, rect, late_txtp: effort <= 5 && !fine, reduced_txtp: effort <= 5 || effort <= 7 && !fine, tx_domain_dist: effort <= 7, uv_from_luma: effort <= 5, prune_intra: effort <= 7 && !fine, rank_rd: fine, rank_pre: if effort <= 3 || effort <= 5 && qidx > 64 { 6 } else { 0 }, tx_depths: if effort <= 5 { 1 } else { 2 }, levels, prune, variance_boost: false, cnn_prune: if effort > 3 { 0 } else if screen { 1 } else { 2 }, rdoq, ssim: false, quant_matrices: false, chroma_delta_q: false, iq_rdmult: false, skip_prediction: true, iq_filters: false, filter_search: effort >= 4, tile_cols_log2: (!screen && pixels > 512 * 512) as u8 }
    }
}

impl Options {
    /// Everything libaom's tune IQ changes that the encoder has.
    pub fn tune_iq(self) -> Self {
        Self { variance_boost: true, ssim: true, quant_matrices: true, chroma_delta_q: true, iq_rdmult: true, iq_filters: true, ..self }
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
    finish(still_rows(width, height, planes, options, false, None), planes, options)
}

/// `still` with the tile columns after the first coded on threads of `spawner`, as many as it
/// takes; the output is the same.
pub fn still_on(width: u32, height: u32, planes: Arc<Vec<Vec<u8>>>, options: &Options, spawner: Option<&crate::Spawner>) -> Coded {
    let rows: Vec<&[u8]> = planes.iter().map(|p| p.as_slice()).collect();
    finish(still_rows(width, height, &rows, options, false, spawner.map(|s| (s, &planes))), &rows, options)
}

fn finish(mut e: Encoder<Rd>, planes: &[&[u8]], options: &Options) -> Coded {
    if options.filter_search && options.qidx != 0 {
        e.flush();
        filters::search(&mut e, planes);
    }
    e.finish()
}

/// `still` up to the last row pushed, the encoder kept for its statistics.
pub fn still_with(width: u32, height: u32, planes: &[&[u8]], options: &Options) -> Encoder<Rd> {
    still_rows(width, height, planes, options, false, None)
}

/// `still_with`, timing each stage of the search into `Encoder::stage_ns`.
pub fn still_timed(width: u32, height: u32, planes: &[&[u8]], options: &Options) -> Encoder<Rd> {
    still_rows(width, height, planes, options, true, None)
}

/// A tile column handed to a spawner: not started, coded by whoever started it, or done.
enum Column {
    Waiting,
    Coding,
    Done(std::thread::Result<Encoder<Rd>>),
}

pub(crate) fn still_rows(
    width: u32,
    height: u32,
    planes: &[&[u8]],
    options: &Options,
    timing: bool,
    workers: Option<(&crate::Spawner, &Arc<Vec<Vec<u8>>>)>,
) -> Encoder<Rd> {
    let mut e = encoder(width, height, planes, options, timing);
    let cols = e.frame.tiling.cols();
    let push = move |mut e: Encoder<Rd>, planes: &[&[u8]]| {
        let w = width as usize;
        for y in 0..height as usize {
            let rows: Vec<&[u8]> = planes.iter().map(|p| &p[y * w..][..w]).collect();
            e.push(&rows);
        }
        e.flush();
        e
    };
    let Some((spawner, shared)) = workers.filter(|_| cols > 1) else {
        return push(e, planes);
    };
    let code = move |col: u32, planes: &[&[u8]], options: &Options| {
        let mut e = encoder(width, height, planes, options, timing);
        e.own = 1 << col;
        push(e, planes)
    };
    let slots: Vec<_> = (1..cols).map(|_| Arc::new((Mutex::new(Column::Waiting), Condvar::new()))).collect();
    for (col, slot) in (1..).zip(&slots) {
        let (slot, shared, options) = (slot.clone(), shared.clone(), options.clone());
        let taken = spawner(Box::new(move || {
            if !matches!(std::mem::replace(&mut *slot.0.lock().unwrap(), Column::Coding), Column::Waiting) {
                return;
            }
            let rows: Vec<&[u8]> = shared.iter().map(|p| p.as_slice()).collect();
            let coded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| code(col, &rows, &options)));
            *slot.0.lock().unwrap() = Column::Done(coded);
            slot.1.notify_all();
        }));
        if !taken {
            break;
        }
    }
    e.own = 1;
    let mut e = push(e, planes);
    // Columns no worker has started are coded here; started ones are waited for.
    for (col, slot) in (1..).zip(&slots) {
        let mut state = slot.0.lock().unwrap();
        let worker = loop {
            match std::mem::replace(&mut *state, Column::Coding) {
                Column::Waiting => {
                    drop(state);
                    break code(col, planes, options);
                }
                Column::Coding => state = slot.1.wait(state).unwrap(),
                Column::Done(coded) => break coded.unwrap_or_else(|p| std::panic::resume_unwind(p)),
            }
        };
        e.merge(worker);
    }
    e
}

pub(crate) fn encoder(width: u32, height: u32, planes: &[&[u8]], options: &Options, timing: bool) -> Encoder<Rd> {
    let (seq, mut frame) = headers(width, height, planes.len() == 1, options.qidx);
    frame.tiling = Tiling::with_cols(width, height, if width >= 4 * 64 { options.tile_cols_log2 } else { 0 });
    frame.screen_content = options.screen;
    let lossy = options.qidx != 0;
    if lossy && options.iq_filters {
        frame.sharpness = match options.qidx {
            ..=112 => 7,
            ..=160 => 1,
            _ => 0,
        };
        if options.qidx <= 32 {
            frame.cdef = Some(Cdef { damping: 3, ..Default::default() });
        }
    }
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
    e.timing = timing;
    e.iq = options.iq_rdmult;
    e.skip_prediction = options.skip_prediction;
    e.keep_modes = options.keep_modes;
    e.keep_chroma = options.keep_chroma;
    e.rect = options.rect;
    e.late_txtp = options.late_txtp;
    e.reduced_txtp = options.reduced_txtp;
    e.tx_domain_dist = options.tx_domain_dist;
    e.uv_from_luma = options.uv_from_luma;
    e.prune_intra = options.prune_intra;
    e.rank_rd = options.rank_rd;
    e.rank_pre = options.rank_pre;
    e.tx_depths = options.tx_depths;
    e.levels = options.levels.clone();
    e.variance_boost = boost;
    e.cnn_prune = options.cnn_prune;
    e.rdoq = options.rdoq;
    e.prune = options.prune;
    if options.ssim && options.qidx != 0 {
        e.ssim_reference = Some(frame::ssim_reference(width as usize, height as usize, planes[0]));
    }
    e
}
