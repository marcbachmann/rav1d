//! One frame coded a superblock row at a time. The source and the reconstruction are held for
//! one superblock row only; the row above a superblock row is kept as rav1d keeps it, in an edge
//! buffer, and handed to `rav1d_prepare_intra_edges` the same way.
//!
//! When the decider searches, each superblock is searched first: blocks are coded dry, against
//! snapshots of the reconstruction and the contexts, with costs counted from the CDFs as they
//! stood when the superblock began. The plan that wins is then coded for real by the same code.

use std::cmp;
use std::ops::Range;
use std::sync::Once;

use strum::EnumCount;

use crate::align::AlignedVec64;
use crate::cdf::{rav1d_cdf_thread_copy, rav1d_cdf_thread_init_static, CdfContext};
use crate::cpu::{rav1d_get_cpu_flags, rav1d_init_cpu};
use crate::dequant_tables::DAV1D_DQ_TBL;
use crate::disjoint_mut::DisjointMut;
use crate::enc::contexts::{self, scan_rc};
use crate::enc::dist;
use crate::enc::ec::Writer;
use crate::enc::fdct_fast;
use crate::enc::palette;
use crate::enc::partition_cnn::{Cnn, Prune};
use crate::enc::quant;
use crate::enc::header::{obu, BitWriter, Frame, Sequence, OBU_FRAME, OBU_SEQUENCE_HEADER};
use crate::enc::rd::{lambda, CoefCosts, Counter, Sink};
use crate::enum_map::DefaultValue;
use crate::env::{gather_left_partition_prob, gather_top_partition_prob};
use crate::include::common::bitdepth::{BitDepth, BitDepth8};
use crate::include::dav1d::headers::{Rav1dPixelLayout, Rav1dPixelLayoutSubSampled};
use crate::include::dav1d::picture::Rav1dPictureDataComponent;
use crate::internal::SCRATCH_AC_TXTP_LEN;
use crate::intra_edge::{EdgeFlags, EdgeIndex, IntraEdges};
use crate::ipred::Rav1dIntraPredDSPContext;
use crate::ipred_prepare::rav1d_prepare_intra_edges;
use crate::itx::Rav1dInvTxfmDSPContext;
use crate::levels::{
    BlockLevel, BlockPartition, BlockSize, TxClass, TxfmSize, TxfmType, CFL_PRED, DCT_DCT,
    DC_PRED, FILTER_PRED, HOR_DOWN_PRED, HOR_PRED, HOR_UP_PRED, IDTX, N_INTRA_PRED_MODES,
    N_UV_INTRA_PRED_MODES, SMOOTH_H_PRED, SMOOTH_PRED, SMOOTH_V_PRED, VERT_LEFT_PRED, VERT_PRED,
    VERT_RIGHT_PRED, WHT_WHT,
};
use crate::qm::DAV1D_QM_TBL;
use crate::recon::{get_dc_sign_ctx, get_skip_ctx};
use crate::tables::{
    TxfmInfo, CFL_ALLOWED_MASK, DAV1D_AL_PART_CTX, DAV1D_BLOCK_SIZES, DAV1D_FILTER_MODE_TO_Y_MODE,
    DAV1D_INTRA_MODE_CONTEXT, DAV1D_MAX_TXFM_SIZE_FOR_BS,
    DAV1D_PARTITION_TYPE_COUNT, DAV1D_TXFM_DIMENSIONS, DAV1D_TXTP_FROM_UVMODE,
    DAV1D_TX_TYPES_PER_SET, DAV1D_TX_TYPE_CLASS,
};

type BD = BitDepth8;

const SB: usize = 64;

pub(crate) struct Dsp {
    pub ipred: Rav1dIntraPredDSPContext,
    pub itx: Rav1dInvTxfmDSPContext,
}

impl Dsp {
    pub fn new() -> Self {
        static INIT: Once = Once::new();
        INIT.call_once(rav1d_init_cpu);
        let flags = rav1d_get_cpu_flags();
        Self {
            ipred: Rav1dIntraPredDSPContext::new::<BD>(flags),
            itx: Rav1dInvTxfmDSPContext::new::<BD>(flags, 8),
        }
    }
}

/// A plane of `rows` rows in a 64-byte aligned buffer, seen as a rav1d picture component.
struct Plane {
    _buf: AlignedVec64<u8>,
    comp: Rav1dPictureDataComponent,
    stride: usize,
}

// SAFETY: `comp` points only into `_buf`, which the plane owns and moves with it.
unsafe impl Send for Plane {}

impl Plane {
    fn new(stride: usize, rows: usize) -> Self {
        let mut buf = AlignedVec64::new();
        buf.resize(stride * rows, 0);
        let comp = Rav1dPictureDataComponent::wrap_buf::<BD>(buf.as_mut_slice(), stride);
        Self {
            _buf: buf,
            comp,
            stride,
        }
    }

    fn row(&self, y: usize, x: usize, n: usize) -> Vec<u8> {
        self.comp.slice::<BD, _>((y * self.stride + x.., ..n)).to_vec()
    }

    fn read(&self, x: usize, y: usize, w: usize, h: usize, out: &mut Vec<u8>) {
        let at = out.len();
        out.resize(at + w * h, 0);
        let len = (h - 1) * self.stride + w;
        copy_rect(&mut out[at..], w, &self.comp.slice::<BD, _>((y * self.stride + x.., ..len)), self.stride, w, h);
    }

    fn write(&self, x: usize, y: usize, w: usize, h: usize, data: &[u8]) {
        let len = (h - 1) * self.stride + w;
        copy_rect(&mut self.comp.slice_mut::<BD, _>((y * self.stride + x.., ..len)), self.stride, data, w, w, h);
    }
}

/// Above or left context, per 4×4 unit, as rav1d's `BlockContext` holds it for intra frames.
struct Ctx {
    mode: Vec<u8>,
    uvmode: Vec<u8>,
    skip: Vec<u8>,
    tx_intra: Vec<i8>,
    lcoef: Vec<u8>,
    ccoef: [Vec<u8>; 2],
    pal_sz: Vec<u8>,
    pal_sz_uv: Vec<u8>,
    /// Palette colours per plane where `pal_sz` (`pal_sz_uv` for U and V) is not 0.
    pal: Vec<[[u8; 8]; 3]>,
    partition: Vec<u8>,
}

impl Ctx {
    fn new(n4: usize) -> Self {
        let n = n4 + 32;
        Self {
            mode: vec![DC_PRED; n],
            uvmode: vec![DC_PRED; n],
            skip: vec![0; n],
            tx_intra: vec![-1; n],
            lcoef: vec![0x40; n],
            ccoef: [vec![0x40; n], vec![0x40; n]],
            pal_sz: vec![0; n],
            pal_sz_uv: vec![0; n],
            pal: vec![[[0; 8]; 3]; n],
            partition: vec![0; n / 2 + 16],
        }
    }

    fn reset(&mut self) {
        self.mode.fill(DC_PRED);
        self.uvmode.fill(DC_PRED);
        self.skip.fill(0);
        self.tx_intra.fill(-1);
        self.lcoef.fill(0x40);
        self.ccoef.iter_mut().for_each(|c| c.fill(0x40));
        self.pal_sz.fill(0);
        self.pal_sz_uv.fill(0);
        self.partition.fill(0);
    }

    /// The units in `r`, at most 16 (and their 8×8 partition units), to be put back by `restore`.
    fn save(&self, r: Range<usize>) -> Saved {
        let p = r.start / 2..r.end.div_ceil(2);
        let mut s = Saved { n: r.len(), np: p.len(), ..Saved::default() };
        let a = r.start;
        s.mode.copy_from_slice(&self.mode[a..][..16]);
        s.uvmode.copy_from_slice(&self.uvmode[a..][..16]);
        s.skip.copy_from_slice(&self.skip[a..][..16]);
        s.tx_intra.copy_from_slice(&self.tx_intra[a..][..16]);
        s.lcoef.copy_from_slice(&self.lcoef[a..][..16]);
        s.ccoef[0].copy_from_slice(&self.ccoef[0][a..][..16]);
        s.ccoef[1].copy_from_slice(&self.ccoef[1][a..][..16]);
        s.pal_sz.copy_from_slice(&self.pal_sz[a..][..16]);
        s.pal_sz_uv.copy_from_slice(&self.pal_sz_uv[a..][..16]);
        s.pal[..s.n].copy_from_slice(&self.pal[r]);
        s.partition[..s.np].copy_from_slice(&self.partition[p]);
        s
    }

    /// The coefficient contexts of the units in `r`, at most 16.
    fn save_coef(&self, r: Range<usize>) -> Saved {
        let mut s = Saved { n: r.len(), ..Saved::default() };
        s.lcoef.copy_from_slice(&self.lcoef[r.start..][..16]);
        s.ccoef[0].copy_from_slice(&self.ccoef[0][r.start..][..16]);
        s.ccoef[1].copy_from_slice(&self.ccoef[1][r.start..][..16]);
        s
    }

    fn restore(&mut self, start: usize, s: &Saved) {
        let n = s.n;
        put(&mut self.mode[start..], &s.mode, n);
        put(&mut self.uvmode[start..], &s.uvmode, n);
        put(&mut self.skip[start..], &s.skip, n);
        put(&mut self.tx_intra[start..], &s.tx_intra, n);
        put(&mut self.pal_sz[start..], &s.pal_sz, n);
        put(&mut self.pal_sz_uv[start..], &s.pal_sz_uv, n);
        self.pal[start..start + n].copy_from_slice(&s.pal[..n]);
        self.restore_coef(start, s);
        let p = start / 2;
        self.partition[p..p + s.np].copy_from_slice(&s.partition[..s.np]);
    }

    fn restore_coef(&mut self, start: usize, s: &Saved) {
        let n = s.n;
        put(&mut self.lcoef[start..], &s.lcoef, n);
        put(&mut self.ccoef[0][start..], &s.ccoef[0], n);
        put(&mut self.ccoef[1][start..], &s.ccoef[1], n);
    }
}

/// The first `n` of 16 saved units, as fixed-size copies for the block sides 1 to 16.
#[inline(always)]
fn put<T: Copy>(dst: &mut [T], src: &[T; 16], n: usize) {
    match n {
        1 => dst[..1].copy_from_slice(&src[..1]),
        2 => dst[..2].copy_from_slice(&src[..2]),
        4 => dst[..4].copy_from_slice(&src[..4]),
        8 => dst[..8].copy_from_slice(&src[..8]),
        16 => dst[..16].copy_from_slice(src),
        _ => dst[..n].copy_from_slice(&src[..n]),
    }
}

/// Up to 16 units of a `Ctx`, copied out without allocating.
#[derive(Clone, Default)]
struct Saved {
    n: usize,
    np: usize,
    mode: [u8; 16],
    uvmode: [u8; 16],
    skip: [u8; 16],
    tx_intra: [i8; 16],
    lcoef: [u8; 16],
    ccoef: [[u8; 16]; 2],
    pal_sz: [u8; 16],
    pal_sz_uv: [u8; 16],
    pal: [[[u8; 8]; 3]; 16],
    partition: [u8; 9],
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Intra {
    pub y_mode: u8,
    /// The angle delta, or the filter-intra mode when `y_mode` is `FILTER_PRED`.
    pub y_angle: i8,
    pub uv_mode: u8,
    pub uv_angle: i8,
    pub cfl: [i8; 2],
    /// Palette sizes for luma and chroma, 0 for none; then the colours of each plane, sorted
    /// for luma and U, V alongside U.
    pub pal_sz: [u8; 2],
    pub pal: [[u8; 8]; 3],
}

/// The quantized coefficients of one transform block, in rav1d's coefficient layout.
#[derive(Clone)]
pub struct Coefs {
    pub txtp: TxfmType,
    pub levels: Vec<i32>,
}

/// Which halves of a block the frame holds when its partition is coded.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Halves {
    /// Both: any partition of the level.
    Both,
    /// Only the top half: `Split` or `H`.
    Top,
    /// Only the left half: `Split` or `V`.
    Left,
}

/// What the syntax allows a block to choose.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub angles: bool,
    pub cfl: bool,
    pub filter: bool,
    pub chroma: bool,
    pub palette: bool,
    pub max_depth: u8,
}

/// The decisions of a frame, asked in coding order. An encoder whose decider `searches` makes
/// them itself and asks nothing.
pub trait Decide {
    fn searches(&self) -> bool {
        false
    }

    fn partition(&mut self, bl: BlockLevel, x: i32, y: i32, halves: Halves) -> BlockPartition;

    /// The superblock's qidx, when the frame codes delta q.
    fn qidx(&mut self, _x: i32, _y: i32, last: u8) -> u8 {
        last
    }

    /// Modes and the transform depth below the largest transform of the block.
    fn mode(&mut self, bs: BlockSize, x: i32, y: i32, limits: &Limits) -> (Intra, u8);

    /// A transform type from `set` (ignored when `set` is empty and the type is fixed) and its
    /// levels, or `None` to have them chosen by rate and distortion.
    fn residual(&mut self, _plane: usize, _tx: TxfmSize, _set: &[TxfmType], _levels: &mut [i32]) -> Option<TxfmType> {
        None
    }
}

/// Every superblock split down to one block level, DC prediction, no residual.
pub struct Skip(pub BlockLevel);

impl Decide for Skip {
    fn partition(&mut self, bl: BlockLevel, _: i32, _: i32, halves: Halves) -> BlockPartition {
        match ((bl as u8) < self.0 as u8, halves) {
            (true, _) => BlockPartition::Split,
            (false, Halves::Both) => BlockPartition::None,
            (false, Halves::Top) => BlockPartition::H,
            (false, Halves::Left) => BlockPartition::V,
        }
    }

    fn mode(&mut self, _: BlockSize, _: i32, _: i32, _: &Limits) -> (Intra, u8) {
        (Intra::default(), 0)
    }

    fn residual(&mut self, _: usize, _: TxfmSize, set: &[TxfmType], levels: &mut [i32]) -> Option<TxfmType> {
        levels.fill(0);
        Some(set.first().copied().unwrap_or(DCT_DCT))
    }
}

/// Decisions made by rate and distortion.
pub struct Rd;

impl Decide for Rd {
    fn searches(&self) -> bool {
        true
    }

    fn partition(&mut self, _: BlockLevel, _: i32, _: i32, _: Halves) -> BlockPartition {
        unreachable!()
    }

    fn mode(&mut self, _: BlockSize, _: i32, _: i32, _: &Limits) -> (Intra, u8) {
        unreachable!()
    }
}

#[derive(Clone, Copy, Debug)]
enum Node {
    Part(BlockPartition),
    /// Modes, transform depth, and the index of the block's residuals in `Encoder::chosen`.
    Block(Intra, u8, usize),
}

pub struct Coded {
    /// The sequence header and frame OBUs.
    pub data: Vec<u8>,
    /// The `av1C` payload.
    pub config: Vec<u8>,
    pub recon: Option<Vec<Vec<u8>>>,
}

struct Tile {
    w: Writer,
    cdf: CdfContext,
    col_start: i32,
    col_end: i32,
    row_start: i32,
    row_end: i32,
    last_qidx: u8,
}

/// The state of a block's area: its pixels in every plane and its contexts.
struct Snap {
    x: i32,
    y: i32,
    w4: usize,
    h4: usize,
    pixels: Vec<u8>,
    above: Saved,
    left: Saved,
}

/// A block's palette map and the bits of its indexes, kept while the block is searched: both
/// depend only on the source and the colours.
struct PalKept {
    key: (bool, u8, i32, i32, i32, i32, [[u8; 8]; 2]),
    map: Vec<u8>,
    bits: Option<u32>,
}

/// A transform block decided while predicting, before its symbols are written.
#[derive(Clone)]
struct TxDone {
    plane: usize,
    x4: i32,
    y4: i32,
    tx: TxfmSize,
    coefs: Coefs,
    eob: i32,
    /// The coefficient bits the residual search counted, in the contexts `coef_bits` sees.
    bits: Option<u32>,
}

pub struct Encoder<D: Decide> {
    pub seq: Sequence,
    pub frame: Frame,
    pub decide: D,
    /// λ is libaom's with the tune-IQ weight.
    pub iq: bool,
    dsp: Dsp,
    planes: usize,
    /// Frame size in 4×4 units, rounded up to 8×8 as the decoder rounds it.
    bw: i32,
    bh: i32,
    src: Vec<Plane>,
    rec: Vec<Plane>,
    scratch: Plane,
    /// `scratch` holds the reconstruction of the levels `rd_residual` chose.
    recon_ready: bool,
    /// Residuals to take instead of searching, last first: a block the search already decided.
    replay: Option<Vec<TxDone>>,
    /// The residuals of the planned blocks of the superblock.
    chosen: Vec<Vec<TxDone>>,
    /// Emptied level and transform block lists for the next trials.
    spare: (Vec<Vec<i32>>, Vec<Vec<TxDone>>),
    /// The bits of the levels `rd_residual` chose.
    residual_bits: Option<u32>,
    /// Their distortion.
    residual_dist: u64,
    /// A cost the planes being predicted must stay under, their decided transform blocks'
    /// distortion and bits so far, and whether they stopped there.
    limit: Option<f64>,
    spent: (u64, u32),
    stopped: bool,
    edge: DisjointMut<AlignedVec64<u8>>,
    edge_stride: usize,
    above: Ctx,
    left: Ctx,
    rows: usize,
    sby: usize,
    tiles: Vec<Tile>,
    /// The tile columns this encoder codes, a bit each; the others' tiles stay empty for `merge`.
    pub own: u64,
    coded: Vec<Vec<u8>>,
    /// The superblock's qidx and its delta from the tile's last one, until coded.
    sb_qidx: u8,
    sb_delta: Option<i32>,
    dq: [[u32; 2]; 3],
    qm: [[Option<&'static [u8]>; 3]; TxfmSize::COUNT],
    lambda: f64,
    /// Searching: blocks are counted, not written, and their modes come from `trial`.
    dry: bool,
    /// While searching modes, only `DCT_DCT` is tried where the type is coded.
    dct_only: bool,
    /// Prediction alone, no residual: ranks luma modes by SATD before rate and distortion.
    predict_only: bool,
    /// Luma modes kept from the SATD ranking for rate and distortion, all when 0.
    pub keep_modes: usize,
    /// Chroma modes kept from the SATD ranking for rate and distortion, all when 0.
    pub keep_chroma: usize,
    /// Whether the search tries H and V partitions: 0 never, 1 always, 2 where Split beat None.
    pub rect: u8,
    /// Transform depths are compared with `DCT_DCT` only; the chosen one searches every type.
    pub late_txtp: bool,
    /// See `Options::reduced_txtp`.
    pub reduced_txtp: bool,
    /// See `Options::tx_domain_dist`.
    pub tx_domain_dist: bool,
    /// See `Options::uv_from_luma`.
    pub uv_from_luma: bool,
    /// See `Options::prune_intra`.
    pub prune_intra: bool,
    /// See `Options::rank_rd`.
    pub rank_rd: bool,
    /// See `Options::tx_depths`.
    pub tx_depths: u8,
    /// See `Options::rank_pre`.
    pub rank_pre: usize,
    /// Block levels the search codes unsplit: 64×64 down to 8×8, and 4×4 when `Bl8x8` splits.
    pub levels: Range<u8>,
    /// A block whose unsplit cost, distortion and rate, is under `prune` bits' worth per 4×4 unit
    /// is not split.
    pub prune: f64,
    /// Superblock qindexes from libaom's Variance Boost; needs `Frame::delta_q_log2`.
    pub variance_boost: bool,
    /// libaom's `intra_cnn_based_part_prune_level`: 0 off, 1 prunes Split and, where only Split
    /// is predicted, H and V; 2 also None (`partition_cnn`).
    pub cnn_prune: u8,
    /// The CNN's outputs for the superblock being searched, when it lies wholly in the picture.
    cnn: Option<Box<Cnn>>,
    /// 16 × 16 × the variance of each whole 4×4 luma area of the superblock, -1 past the picture.
    var4: [i64; 256],
    /// The last source luma row of the superblock row above, for the CNN.
    src_above: Vec<u8>,
    /// Levels the residual search tries one step nearer zero, from the last coefficient back,
    /// keeping each that lowers the cost.
    pub rdoq: usize,
    /// The geometric mean of `ssim_factor` over the frame: λ is scaled per 16×16 block by its
    /// factor over this, libaom's tune-IQ rate weighting.
    pub ssim_reference: Option<f64>,
    /// libaom's `predict_skip_txfm`: a luma block whose residual is small at its largest
    /// transform is coded without residual, and no smaller transform is searched.
    pub skip_prediction: bool,
    /// Whether the last luma residual searched was predicted to skip.
    skip_predicted: bool,
    /// See `Options::uv_outward`.
    pub uv_outward: bool,
    /// See `Options::rect_ratio`.
    pub rect_ratio: f64,
    /// See `Options::rect_adaptive`.
    pub rect_adaptive: bool,
    /// See `Options::split8_adaptive`.
    pub split8_adaptive: bool,
    /// Per activity class of the block (`activity_bucket`): 16×16 blocks met, searched below
    /// 16×16 (8×8 splits and the rect partitions after them), the cost that search gained over
    /// None where a partition won, the cost of every decision tried, and the wins that gained
    /// over 2% of their block's cost.
    pub split_seen: [(u32, u32, f64, f64, u32); 4],
    /// Per partition level and activity class of the block under `rect_ratio`: rect partitions
    /// met and tried, the cost their wins saved over the best square partition, the cost of every
    /// decision tried, and the wins that saved over 2% of the block's cost.
    pub rect_seen: [[(u32, u32, f64, f64, u32); 4]; 5],
    /// See `Options::fast_cdef`.
    pub fast_cdef: bool,
    /// See `Options::graphic_guard`.
    pub graphic_guard: bool,
    /// The superblock's 16×16 blocks' factors over `ssim_reference`, and its λ before them.
    ssim: [f64; 16],
    sb_lambda: f64,

    /// The colour indexes of the last block predicted with a palette, luma and chroma, a row of
    /// the block's width at a time.
    pal_map: [Vec<u8>; 2],
    pal_kept: Vec<PalKept>,
    bufs: Option<Box<Bufs>>,
    /// Which part of the search runs, for `stats`.
    stage: usize,
    /// Residual searches and the coefficients they transformed, per stage: luma ranking, luma
    /// depths, chroma, CfL, the block's final dry code, coding.
    pub stats: [[u64; 2]; 6],
    /// Nanoseconds per stage of `stats` when `timing`.
    pub stage_ns: [u64; 10],
    /// Nanoseconds in `pick_block` by the block's larger side: 4, 8, 16, 32, 64.
    pub size_ns: [u64; 5],
    pub timing: bool,
    /// The search's rate estimate of the coded superblocks, in 1/512 bits.
    pub estimated: u64,
    /// The search's squared error of the coded superblocks, before the in-loop filters.
    pub estimated_sse: u64,
    lv: Option<Box<[i32; 32 * 32]>>,
    counter: Counter,
    cost_cdf: Box<CdfContext>,
    coef_costs: Box<CoefCosts>,
    trial: Option<(Intra, u8)>,
    plan: Vec<Node>,
    next: usize,
    /// The whole reconstruction, kept only when asked for (tests).
    pub recon: Option<Vec<Vec<u8>>>,
}

/// libaom's tune-IQ activity of a 16×16 luma block of `cols`×`rows` 8×8 blocks
/// (`av1_set_mb_ssim_rdmult_scaling`): its mean per-pixel 8×8 variance through a fitted curve.
fn ssim_factor(px: &dyn Fn(usize, usize) -> u32, cols: usize, rows: usize) -> f64 {
    let mut var = 0.0;
    for b in 0..cols * rows {
        let (bx, by) = ((b % cols) * 8, (b / cols) * 8);
        let (mut sum, mut sse) = (0u32, 0u32);
        for r in 0..8 {
            for c in 0..8 {
                let p = px(bx + c, by + r);
                sum += p;
                sse += p * p;
            }
        }
        var += ((sse - ((sum as u64 * sum as u64) >> 6) as u32 + 32) >> 6) as f64;
    }
    67.035434 * (1.0 - (-0.0021489 * var / (cols * rows) as f64).exp()) + 17.492222
}

/// The geometric mean of `ssim_factor` over a picture's luma, the reference its blocks' rate
/// weights are taken against.
pub fn ssim_reference(width: usize, height: usize, luma: &[u8]) -> f64 {
    let (w8, h8) = (width.div_ceil(8), height.div_ceil(8));
    let (cols, rows) = (w8.div_ceil(2), h8.div_ceil(2));
    let mut log = 0.0;
    for r in 0..rows {
        for c in 0..cols {
            let (x0, y0) = (c * 16, r * 16);
            let px = |x: usize, y: usize| luma[cmp::min(y0 + y, height - 1) * width + cmp::min(x0 + x, width - 1)] as u32;
            log += ssim_factor(&px, cmp::min(2, w8 - 2 * c), cmp::min(2, h8 - 2 * r)).ln();
        }
    }
    (log / (rows * cols) as f64).exp()
}

impl<D: Decide> Encoder<D> {
    pub fn new(seq: Sequence, frame: Frame, decide: D) -> Self {
        let planes = if seq.mono { 1 } else { 3 };
        let bw = ((seq.width as i32 + 7) >> 3) << 1;
        let bh = ((seq.height as i32 + 7) >> 3) << 1;
        let stride = (bw as usize * 4).next_multiple_of(SB) + SB;
        let src = (0..planes).map(|_| Plane::new(stride, SB)).collect();
        let rec = (0..planes).map(|_| Plane::new(stride, SB)).collect();
        let mut edge = AlignedVec64::new();
        edge.resize(stride * planes, 0);
        let q = frame.quant;
        let mut qm = [[None; 3]; TxfmSize::COUNT];
        if let Some(levels) = q.qm {
            for (i, qm) in qm.iter_mut().enumerate() {
                *qm = [
                    DAV1D_QM_TBL[levels[0] as usize][0][i],
                    DAV1D_QM_TBL[levels[1] as usize][1][i],
                    DAV1D_QM_TBL[levels[1] as usize][1][i],
                ];
            }
        }
        let cost_cdf = Box::new(rav1d_cdf_thread_copy(&rav1d_cdf_thread_init_static(q.base)));
        let mut e = Self {
            iq: false,
            planes,
            bw,
            bh,
            src,
            rec,
            scratch: Plane::new(SB, SB),
            recon_ready: false,
            replay: None,
            chosen: Vec::new(),
            spare: (Vec::new(), Vec::new()),
            residual_bits: None,
            residual_dist: 0,
            limit: None,
            spent: (0, 0),
            stopped: false,
            edge: DisjointMut::new(edge),
            edge_stride: stride,
            above: Ctx::new(bw as usize),
            left: Ctx::new(16),
            rows: 0,
            sby: 0,
            tiles: Vec::new(),
            own: u64::MAX,
            coded: Vec::new(),
            sb_qidx: q.base,
            sb_delta: None,
            dq: [[0; 2]; 3],
            qm,
            lambda: 1.0,
            dry: false,
            dct_only: false,
            predict_only: false,
            keep_modes: 0,
            keep_chroma: 0,
            rect: 1,
            late_txtp: false,
            reduced_txtp: false,
            tx_domain_dist: false,
            uv_from_luma: false,
            prune_intra: false,
            rank_rd: false,
            tx_depths: 2,
            rank_pre: 0,
            levels: 0..5,
            prune: 0.0,
            variance_boost: false,
            cnn_prune: 0,
            cnn: None,
            var4: [-1; 256],
            src_above: Vec::new(),
            rdoq: 0,
            ssim_reference: None,
            skip_prediction: false,
            skip_predicted: false,
            uv_outward: false,
            rect_ratio: 0.0,
            rect_adaptive: false,
            split8_adaptive: false,
            split_seen: [(0, 0, 0.0, 0.0, 0); 4],
            rect_seen: [[(0, 0, 0.0, 0.0, 0); 4]; 5],
            fast_cdef: false,
            graphic_guard: false,
            ssim: [1.0; 16],
            sb_lambda: 1.0,

            pal_map: [Vec::new(), Vec::new()],
            pal_kept: Vec::new(),
            bufs: Some(Box::new(Bufs {
                src: [0; 64 * 64],
                pred: [0; 64 * 64],
                residual: [0; 64 * 64],
                coefs: [0; 32 * 32],
                trial: [0; 32 * 32],
                cf: Coef([0; 32 * 32]),
            })),
            lv: Some(Box::new([0; 32 * 32])),
            stage: 5,
            stats: [[0; 2]; 6],
            stage_ns: [0; 10],
            size_ns: [0; 5],
            timing: false,
            estimated: 0,
            estimated_sse: 0,
            counter: Counter::default(),
            coef_costs: CoefCosts::new(&cost_cdf.coef),
            cost_cdf,
            trial: None,
            plan: Vec::new(),
            next: 0,
            recon: None,
            seq,
            frame,
            decide,
            dsp: Dsp::new(),
        };
        e.set_qidx(q.base);
        e
    }

    /// libaom's Variance Boost for a still picture at its default strength: superblocks whose
    /// 8×8 luma variances are low at the fifth octile get a lower qindex.
    fn boost_qidx(&self, x: i32) -> u8 {
        let base = self.frame.quant.base as i32;
        let src = &self.src[0];
        let x0 = x as usize * 4;
        let mut variances = [0u32; 64];
        for (i, v) in variances.iter_mut().enumerate() {
            let (bx, by) = (x0 + (i % 8) * 8, (i / 8) * 8);
            let (mut sum, mut sse) = (0i64, 0u64);
            for r in 0..8 {
                let row = src.comp.slice::<BD, _>(((by + r) * src.stride + bx.., ..8));
                for &p in row.iter() {
                    sum += p as i64;
                    sse += (p as u64) * (p as u64);
                }
            }
            *v = ((sse as u32).wrapping_sub(((sum * sum) >> 6) as u32)) / 64;
        }
        variances.sort_unstable();
        let variance = ((variances[31] + 2 * variances[39] + variances[47] + 2) / 4).max(1);
        let strength = 3.0;
        let ratio = (0.15 * strength * (10.0 - (variance as f64).log2()) + 1.0).clamp(1.0, 8.0);
        let q = |qidx: i32| DAV1D_DQ_TBL[0][qidx as usize][1] as f64 / 4.0;
        let target_q = q(base) / ratio;
        let target = (1..255).find(|&i| q(i) >= target_q).unwrap_or(255);
        let boost = (((base as f64 + 544.0) * (base - target) as f64 / 1279.0).round() as i32).min(80);
        cmp::max(base - boost, 1) as u8
    }

    fn ssim_scales(&mut self, x: i32, reference: f64) {
        let src = &self.src[0];
        let y = self.sby as i32 * 16;
        for (i, s) in self.ssim.iter_mut().enumerate() {
            let (bx, by) = (x + (i as i32 % 4) * 4, y + (i as i32 / 4) * 4);
            if bx >= self.bw || by >= self.bh {
                continue;
            }
            let cols = cmp::min(2, (self.bw - bx) / 2) as usize;
            let rows = cmp::min(2, (self.bh - by) / 2) as usize;
            let (x0, y0) = (bx as usize * 4, (by - y) as usize * 4);
            let mut block = [0u8; 256];
            for r in 0..rows * 8 {
                let row = src.comp.slice::<BD, _>(((y0 + r) * src.stride + x0.., ..cols * 8));
                block[r * 16..][..cols * 8].copy_from_slice(&row);
            }
            *s = ssim_factor(&|c, r| block[r * 16 + c] as u32, cols, rows) / reference;
        }
    }

    /// λ for a block: the superblock's, scaled by the geometric mean of its 16×16 blocks' factors.
    fn block_lambda(&mut self, x: i32, y: i32, w4: usize, h4: usize) {
        if self.ssim_reference.is_none() {
            return;
        }
        let (cols, rows) = ((self.bw + 3) >> 2, (self.bh + 3) >> 2);
        let (c0, r0) = (x >> 2, y >> 2);
        let (c1, r1) = (cmp::min(cols, c0 + (w4 as i32 + 3) / 4), cmp::min(rows, r0 + (h4 as i32 + 3) / 4));
        let (mut log, mut n) = (0.0, 0.0);
        for r in r0..r1 {
            for c in c0..c1 {
                log += self.ssim[((r & 3) * 4 + (c & 3)) as usize].ln();
                n += 1.0;
            }
        }
        self.lambda = self.sb_lambda * (log / n).exp();
    }

    fn set_qidx(&mut self, qidx: u8) {
        let q = self.frame.quant;
        let clip = |d: i8| (qidx as i32 + d as i32).clamp(0, 255) as usize;
        let tbl = &DAV1D_DQ_TBL[0];
        let u = [tbl[clip(q.u_dc)][0] as u32, tbl[clip(q.u_ac)][1] as u32];
        self.dq = [[tbl[clip(q.y_dc)][0] as u32, tbl[qidx as usize][1] as u32], u, u];
        self.lambda = lambda(qidx, self.iq);
    }

    /// One row of each plane, `width` pixels.
    pub fn push(&mut self, row: &[&[u8]]) {
        let w = self.seq.width as usize;
        let y = self.rows % SB;
        for (p, plane) in self.src.iter().enumerate() {
            let mut dst = plane.comp.slice_mut::<BD, _>((y * plane.stride.., ..plane.stride));
            dst[..w].copy_from_slice(&row[p][..w]);
            let last = dst[w - 1];
            dst[w..].fill(last);
        }
        self.rows += 1;
        if self.rows % SB == 0 {
            self.code_sb_row();
        }
    }

    pub fn finish(mut self) -> Coded {
        self.flush();
        Coded {
            data: self.bytes(),
            config: self.seq.config(),
            recon: self.recon,
        }
    }

    /// Codes the last superblock row, its rows past the picture repeating its last one.
    pub fn flush(&mut self) {
        assert_eq!(self.rows, self.seq.height as usize);
        if self.sby as u32 == self.frame.tiling.sbh {
            return;
        }
        let y = self.rows % SB;
        for plane in &self.src {
            let last = plane.row(y - 1, 0, plane.stride);
            for r in y..SB {
                plane
                    .comp
                    .slice_mut::<BD, _>((r * plane.stride.., ..plane.stride))
                    .copy_from_slice(&last);
            }
        }
        self.code_sb_row();
    }

    /// The sequence header and frame OBUs of the tiles coded so far.
    pub fn bytes(&self) -> Vec<u8> {
        let mut tile_data = Vec::new();
        let n = self.coded.len();
        let size_bytes = self
            .coded
            .iter()
            .take(n - 1)
            .map(|t| (t.len() as u64).max(1))
            .max()
            .map_or(1, |m| (64 - (m - 1).leading_zeros()).div_ceil(8).max(1)) as u8;
        let mut w = BitWriter::default();
        self.frame.write(&self.seq, size_bytes, &mut w);
        w.align();
        if n > 1 {
            w.bit(false);
        }
        let mut payload = w.finish();
        for (i, t) in self.coded.iter().enumerate() {
            if i + 1 < n {
                let s = t.len() as u64 - 1;
                for b in 0..size_bytes {
                    tile_data.push((s >> (8 * b)) as u8);
                }
            }
            tile_data.extend_from_slice(t);
        }
        payload.extend_from_slice(&tile_data);
        let mut data = Vec::new();
        obu(OBU_SEQUENCE_HEADER, &self.seq.write(), &mut data);
        obu(OBU_FRAME, &payload, &mut data);
        data
    }

    fn code_sb_row(&mut self) {
        let t = self.frame.tiling;
        let sby = self.sby as u32;
        if sby % t.tile_h == 0 {
            self.finish_tiles();
            let row_start = (sby * 16) as i32;
            let row_end = cmp::min(((sby + t.tile_h) * 16) as i32, self.bh);
            let base = self.frame.quant.base;
            for col in 0..t.cols() {
                let col_start = (col * t.tile_w * 16) as i32;
                let col_end = cmp::min(((col + 1) * t.tile_w * 16) as i32, self.bw);
                self.tiles.push(Tile {
                    w: Writer::default(),
                    cdf: rav1d_cdf_thread_copy(&rav1d_cdf_thread_init_static(base)),
                    col_start,
                    col_end,
                    row_start,
                    row_end,
                    last_qidx: base,
                });
            }
            self.above.reset();
        }
        let y = (sby * 16) as i32;
        for ti in 0..self.tiles.len() {
            if self.own >> ti & 1 == 0 {
                continue;
            }
            self.left.reset();
            let (start, end) = (self.tiles[ti].col_start, self.tiles[ti].col_end);
            for x in (start..end).step_by(16) {
                let last = self.tiles[ti].last_qidx;
                self.sb_delta = None;
                let mut qidx = last;
                let mut target = None;
                if let Some(log2) = self.frame.delta_q_log2 {
                    let delta = if self.variance_boost {
                        let want = self.boost_qidx(x);
                        target = Some(want);
                        delta_steps(1 << log2, last, want)
                    } else {
                        let want = if self.decide.searches() { last } else { self.decide.qidx(x, y, last) };
                        (want as i32 - last as i32) >> log2
                    };
                    qidx = (last as i32 + (delta << log2)).clamp(1, 255) as u8;
                    self.sb_delta = Some(delta);
                }
                self.sb_qidx = qidx;
                self.set_qidx(qidx);
                if let Some(target) = target {
                    self.lambda = lambda(target, self.iq);
                }
                self.sb_lambda = self.lambda;
                if let Some(reference) = self.ssim_reference {
                    self.ssim_scales(x, reference);
                }
                if self.decide.searches() {
                    *self.cost_cdf = self.tiles[ti].cdf.clone();
                    self.coef_costs.fill(&self.cost_cdf.coef);
                    self.search_sb(ti, x, y);
                }
                self.code_sb(ti, BlockLevel::Bl64x64, x, y, EdgeIndex::root());
                assert_eq!(self.next, self.plan.len());
            }
        }
        if let Some(recon) = &mut self.recon {
            let (w, h) = (self.seq.width as usize, self.seq.height as usize);
            for (p, plane) in self.rec.iter().enumerate() {
                for r in 0..cmp::min(SB, h - self.sby * SB) {
                    recon[p].extend_from_slice(&plane.row(r, 0, w));
                }
            }
        }
        if self.cnn_prune != 0 {
            self.src_above = self.src[0].row(SB - 1, 0, self.src[0].stride);
        }
        for (p, plane) in self.rec.iter().enumerate() {
            let last = plane.row(SB - 1, 0, plane.stride);
            self.edge
                .mut_slice_as::<_, u8>((p * self.edge_stride.., ..plane.stride))
                .copy_from_slice(&last);
        }
        self.sby += 1;
        if self.sby == self.frame.tiling.sbh as usize {
            self.finish_tiles();
        }
    }

    fn finish_tiles(&mut self) {
        for (ti, tile) in self.tiles.drain(..).enumerate() {
            self.coded.push(if self.own >> ti & 1 != 0 { tile.w.finish() } else { Vec::new() });
        }
    }

    /// Takes the tiles, estimates and reconstruction of the tile columns `other` coded.
    pub fn merge(&mut self, other: Encoder<D>) {
        let t = self.frame.tiling;
        for (i, data) in other.coded.into_iter().enumerate() {
            if other.own >> (i as u32 % t.cols()) & 1 != 0 {
                self.coded[i] = data;
            }
        }
        self.estimated += other.estimated;
        self.estimated_sse += other.estimated_sse;
        for (a, b) in self.stats.iter_mut().zip(other.stats) {
            a[0] += b[0];
            a[1] += b[1];
        }
        for (a, b) in self.stage_ns.iter_mut().zip(other.stage_ns) {
            *a += b;
        }
        for (a, b) in self.size_ns.iter_mut().zip(other.size_ns) {
            *a += b;
        }
        if let (Some(mine), Some(theirs)) = (&mut self.recon, other.recon) {
            let w = self.seq.width as usize;
            let px = |c: u32| cmp::min(w, (c * t.tile_w * 64) as usize);
            for c in (0..t.cols()).filter(|c| other.own >> c & 1 != 0) {
                for (a, b) in mine.iter_mut().zip(&theirs) {
                    for (ra, rb) in a.chunks_exact_mut(w).zip(b.chunks_exact(w)) {
                        ra[px(c)..px(c + 1)].copy_from_slice(&rb[px(c)..px(c + 1)]);
                    }
                }
            }
        }
    }

    fn partition_ctx(&self, bl: BlockLevel, x: i32, y: i32) -> usize {
        let has_bl = |v: u8| (v >> (4 - bl as u8)) & 1;
        (has_bl(self.above.partition[(x >> 1) as usize]) + 2 * has_bl(self.left.partition[((y & 15) >> 1) as usize])) as usize
    }

    fn halves(&self, bl: BlockLevel, x: i32, y: i32) -> Option<Halves> {
        let hsz = 16 >> bl as u8;
        match (self.bw > x + hsz, self.bh > y + hsz) {
            (true, true) => Some(Halves::Both),
            (true, false) => Some(Halves::Top),
            (false, true) => Some(Halves::Left),
            (false, false) => None,
        }
    }

    fn write_partition<S: Sink>(s: &mut S, cdf: &mut CdfContext, bl: BlockLevel, ctx: usize, halves: Halves, bp: BlockPartition) {
        let pc = &mut cdf.m.partition[bl as usize][ctx];
        match halves {
            Halves::Both => {
                let n = DAV1D_PARTITION_TYPE_COUNT[bl as usize];
                assert!((bp as u8) <= n);
                s.symbol(bp as u8, pc, n);
            }
            Halves::Top => {
                assert!(matches!(bp, BlockPartition::Split | BlockPartition::H));
                s.bool(bp == BlockPartition::Split, gather_top_partition_prob(pc, bl));
            }
            Halves::Left => {
                assert!(matches!(bp, BlockPartition::Split | BlockPartition::V));
                s.bool(bp == BlockPartition::Split, gather_left_partition_prob(pc, bl));
            }
        }
    }

    fn next_node(&mut self) -> Node {
        self.next += 1;
        self.plan[self.next - 1]
    }

    fn code_sb(&mut self, ti: usize, bl: BlockLevel, x: i32, y: i32, edge_index: EdgeIndex) {
        let edges = &IntraEdges::DEFAULT;
        let Some(halves) = self.halves(bl, x, y) else {
            let next = bl.decrease().unwrap();
            return self.code_sb(ti, next, x, y, edges.branch(false, edge_index).split[0]);
        };
        let ctx = self.partition_ctx(bl, x, y);
        let bp = if self.decide.searches() {
            match self.next_node() {
                Node::Part(bp) => bp,
                n => panic!("plan has {n:?} where a partition is due"),
            }
        } else {
            self.decide.partition(bl, x, y, halves)
        };
        let tile = &mut self.tiles[ti];
        Self::write_partition(&mut tile.w, &mut tile.cdf, bl, ctx, halves, bp);
        self.code_partition(
            ti,
            bl,
            bp,
            x,
            y,
            edge_index,
            halves,
            &mut |e, ti, bs, x, y, f| {
                let txs = e.code_b(ti, bs, x, y, f);
                e.recycle(txs);
            },
            &mut |e, ti, bl, x, y, edge_index| e.code_sb(ti, bl, x, y, edge_index),
        );
    }

    /// Runs `block` on every block of `bp`, recursing through splits (searching them when dry),
    /// then sets the partition contexts, as `decode_sb` does.
    #[allow(clippy::too_many_arguments)]
    fn code_partition(
        &mut self,
        ti: usize,
        bl: BlockLevel,
        bp: BlockPartition,
        x: i32,
        y: i32,
        edge_index: EdgeIndex,
        halves: Halves,
        block: &mut dyn FnMut(&mut Self, usize, BlockSize, i32, i32, EdgeFlags),
        split: &mut dyn FnMut(&mut Self, usize, BlockLevel, i32, i32, EdgeIndex),
    ) {
        let hsz = 16 >> bl as u8;
        let (have_h_split, have_v_split) = (halves != Halves::Left, halves != Halves::Top);
        let edges = &IntraEdges::DEFAULT;
        let b = DAV1D_BLOCK_SIZES[bl as usize][bp as usize];
        let node = edges.node(false, edge_index);
        let (nh, nv, no) = (node.h, node.v, node.o);
        use BlockPartition::*;
        match bp {
            None => block(self, ti, b[0], x, y, no),
            H => {
                block(self, ti, b[0], x, y, nh[0]);
                if have_v_split {
                    block(self, ti, b[0], x, y + hsz, nh[1]);
                }
            }
            V => {
                block(self, ti, b[0], x, y, nv[0]);
                if have_h_split {
                    block(self, ti, b[0], x + hsz, y, nv[1]);
                }
            }
            Split if bl == BlockLevel::Bl8x8 => {
                let tip = edges.tip(false, edge_index).split;
                block(self, ti, BlockSize::Bs4x4, x, y, EdgeFlags::ALL_TR_AND_BL);
                block(self, ti, BlockSize::Bs4x4, x + 1, y, tip[0]);
                block(self, ti, BlockSize::Bs4x4, x, y + 1, tip[1]);
                block(self, ti, BlockSize::Bs4x4, x + 1, y + 1, tip[2]);
            }
            Split => {
                let next = bl.decrease().unwrap();
                let edge = edges.branch(false, edge_index).split;
                let mut sub = |e: &mut Self, x: i32, y: i32, i: usize| split(e, ti, next, x, y, edge[i]);
                sub(self, x, y, 0);
                if have_h_split {
                    sub(self, x + hsz, y, 1);
                }
                if have_v_split {
                    sub(self, x, y + hsz, 2);
                    if have_h_split {
                        sub(self, x + hsz, y + hsz, 3);
                    }
                }
            }
            TopSplit => {
                block(self, ti, b[0], x, y, EdgeFlags::ALL_TR_AND_BL);
                block(self, ti, b[0], x + hsz, y, nv[1]);
                block(self, ti, b[1], x, y + hsz, nh[1]);
            }
            BottomSplit => {
                block(self, ti, b[0], x, y, nh[0]);
                block(self, ti, b[1], x, y + hsz, nv[0]);
                block(self, ti, b[1], x + hsz, y + hsz, EdgeFlags::empty());
            }
            LeftSplit => {
                block(self, ti, b[0], x, y, EdgeFlags::ALL_TR_AND_BL);
                block(self, ti, b[0], x, y + hsz, nh[1]);
                block(self, ti, b[1], x + hsz, y, nv[1]);
            }
            RightSplit => {
                block(self, ti, b[0], x, y, nv[0]);
                block(self, ti, b[1], x + hsz, y, nh[0]);
                block(self, ti, b[1], x + hsz, y + hsz, EdgeFlags::empty());
            }
            H4 => {
                let branch = edges.branch(false, edge_index);
                let (h4, h0, h1) = (branch.h4, branch.node.h[0], branch.node.h[1]);
                block(self, ti, b[0], x, y, h0);
                block(self, ti, b[0], x, y + (hsz >> 1), h4);
                block(self, ti, b[0], x, y + hsz, EdgeFlags::ALL_LEFT_HAS_BOTTOM);
                if y + (hsz * 3 >> 1) < self.bh {
                    block(self, ti, b[0], x, y + (hsz * 3 >> 1), h1);
                }
            }
            V4 => {
                let branch = edges.branch(false, edge_index);
                let (v4, v0, v1) = (branch.v4, branch.node.v[0], branch.node.v[1]);
                block(self, ti, b[0], x, y, v0);
                block(self, ti, b[0], x + (hsz >> 1), y, v4);
                block(self, ti, b[0], x + hsz, y, EdgeFlags::ALL_TOP_HAS_RIGHT);
                if x + (hsz * 3 >> 1) < self.bw {
                    block(self, ti, b[0], x + (hsz * 3 >> 1), y, v1);
                }
            }
        }
        if bp != Split || bl == BlockLevel::Bl8x8 {
            let n = hsz as usize;
            let (bx8, by8) = ((x >> 1) as usize, ((y & 15) >> 1) as usize);
            self.above.partition[bx8..bx8 + n].fill(DAV1D_AL_PART_CTX[0][bl as usize][bp as usize]);
            self.left.partition[by8..by8 + n].fill(DAV1D_AL_PART_CTX[1][bl as usize][bp as usize]);
        }
    }

    fn limits(&self, bs: BlockSize) -> (Limits, TxfmSize) {
        let b_dim = bs.dimensions();
        let (bw4, bh4) = (b_dim[0] as i32, b_dim[1] as i32);
        let lossless = self.frame.quant.lossless();
        let max_tx = if lossless { TxfmSize::S4x4 } else { DAV1D_MAX_TXFM_SIZE_FOR_BS[bs as usize][0] };
        let max_dim = &DAV1D_TXFM_DIMENSIONS[max_tx as usize];
        let limits = Limits {
            angles: b_dim[2] + b_dim[3] >= 2,
            cfl: if lossless { bw4 == 1 && bh4 == 1 } else { CFL_ALLOWED_MASK & (1 << bs as u8) != 0 },
            filter: cmp::max(b_dim[2], b_dim[3]) <= 3 && self.seq.filter_intra,
            chroma: self.planes > 1,
            palette: self.frame.screen_content && cmp::max(bw4, bh4) <= 16 && bw4 + bh4 >= 4,
            max_depth: if self.frame.tx_select && !lossless { cmp::min(max_dim.max, 2) } else { 0 },
        };
        (limits, max_tx)
    }

    fn tx_sizes(&self, bs: BlockSize, depth: u8) -> (TxfmSize, TxfmSize) {
        if self.frame.quant.lossless() {
            return (TxfmSize::S4x4, TxfmSize::S4x4);
        }
        let mut tx = DAV1D_MAX_TXFM_SIZE_FOR_BS[bs as usize][0];
        for _ in 0..depth {
            tx = DAV1D_TXFM_DIMENSIONS[tx as usize].sub;
        }
        (tx, DAV1D_MAX_TXFM_SIZE_FOR_BS[bs as usize][3])
    }

    /// Codes one block: predicts and reconstructs it, then writes it, or counts it when dry.
    fn code_b(&mut self, ti: usize, bs: BlockSize, x: i32, y: i32, edge_flags: EdgeFlags) -> Vec<TxDone> {
        let b_dim = bs.dimensions();
        self.block_lambda(x, y, b_dim[0] as usize, b_dim[1] as usize);
        let (limits, max_tx) = self.limits(bs);
        let (intra, depth) = if self.dry {
            self.trial.unwrap()
        } else if self.decide.searches() {
            match self.next_node() {
                Node::Block(intra, depth, k) => {
                    let mut txs = std::mem::take(&mut self.chosen[k]);
                    txs.reverse();
                    self.replay = Some(txs);
                    (intra, depth)
                }
                n => panic!("plan has {n:?} where a block is due"),
            }
        } else {
            self.decide.mode(bs, x, y, &limits)
        };
        assert!(depth <= limits.max_depth);
        let txs = self.predict_block(ti, bs, x, y, edge_flags, &intra, depth, 0..self.planes);
        debug_assert!(self.replay.as_ref().is_none_or(|r| r.is_empty()));
        if let Some(r) = self.replay.take() {
            self.recycle(r);
        }
        let skip = txs.iter().all(|t| t.eob < 0);
        let b = BlockWrite {
            bs,
            x,
            y,
            skip,
            intra,
            depth,
            max_tx,
            limits,
            delta: (x & 15 == 0 && y & 15 == 0 && (bs != BlockSize::Bs64x64 || !skip))
                .then_some(self.sb_delta)
                .flatten(),
            tx_select: self.frame.tx_select,
            lossless: self.frame.quant.lossless(),
            qidx: self.frame.quant.base,
            reduced_tx_set: self.frame.reduced_tx_set,
            bw: self.bw,
            bh: self.bh,
        };
        if self.dry {
            b.write(&mut self.counter, &mut self.cost_cdf, &mut self.above, &mut self.left, &txs, &self.pal_map);
        } else {
            let tile = &mut self.tiles[ti];
            b.write(&mut tile.w, &mut tile.cdf, &mut self.above, &mut self.left, &txs, &self.pal_map);
            if b.delta.is_some() {
                tile.last_qidx = self.sb_qidx;
            }
        }
        txs
    }

    /// Keeps the lists and level buffers of transform blocks no longer needed for later trials.
    fn recycle(&mut self, mut txs: Vec<TxDone>) {
        for t in txs.drain(..) {
            if t.coefs.levels.capacity() > 0 && self.spare.0.len() < 128 {
                self.spare.0.push(t.coefs.levels);
            }
        }
        if self.spare.1.len() < 16 {
            self.spare.1.push(txs);
        }
    }

    /// Predicts and reconstructs the given planes of a block, choosing every transform block's
    /// residual. The coefficient contexts are left as they were.
    #[allow(clippy::too_many_arguments)]
    fn predict_block(
        &mut self,
        ti: usize,
        bs: BlockSize,
        x: i32,
        y: i32,
        edge_flags: EdgeFlags,
        intra: &Intra,
        depth: u8,
        planes: Range<usize>,
    ) -> Vec<TxDone> {
        let b_dim = bs.dimensions();
        let (bw4, bh4) = (b_dim[0] as i32, b_dim[1] as i32);
        let w4 = cmp::min(bw4, self.bw - x);
        let h4 = cmp::min(bh4, self.bh - y);
        let (tx, uvtx) = self.tx_sizes(bs, depth);
        let (ax, ly) = (x as usize, (y & 15) as usize);
        let saved_a = self.above.save_coef(ax..ax + bw4 as usize);
        let saved_l = self.left.save_coef(ly..ly + bh4 as usize);
        let edge_filter = (self.seq.edge_filter as i32) << 10;
        let mut txs = self.spare.1.pop().unwrap_or_default();
        if planes.contains(&0) {
            let flags = self.sm_flag(ax, ly) | edge_filter;
            let pal = intra.pal_sz[0] > 0;
            if pal {
                self.predict_palette(false, x, y, w4, h4, bw4, bh4, intra);
            }
            self.predict_plane(ti, bs, 0, x, y, w4, h4, tx, intra, edge_flags, flags, !pal, &mut txs);
        }
        if planes.contains(&1) {
            if intra.pal_sz[1] > 0 {
                self.predict_palette(true, x, y, w4, h4, bw4, bh4, intra);
            }
            if intra.uv_mode == CFL_PRED {
                self.predict_cfl(ti, x, y, w4, h4, bw4, bh4, tx, uvtx, intra);
            }
            let flags = self.sm_uv_flag(ax, ly) | edge_filter;
            for p in 1..3 {
                let predict = !(intra.uv_mode == CFL_PRED && intra.cfl[p - 1] != 0) && intra.pal_sz[1] == 0;
                self.predict_plane(ti, bs, p, x, y, w4, h4, uvtx, intra, edge_flags, flags, predict, &mut txs);
                if self.stopped {
                    break;
                }
            }
        }
        self.above.restore_coef(ax, &saved_a);
        self.left.restore_coef(ly, &saved_l);
        txs
    }

    /// Fills a block with its palette colours, each pixel the nearest to the source, and keeps
    /// the indexes; past the visible `w4`×`h4` they repeat the last column and row, as
    /// `read_pal_indices` fills them.
    #[allow(clippy::too_many_arguments)]
    fn predict_palette(&mut self, chroma: bool, x: i32, y: i32, w4: i32, h4: i32, bw4: i32, bh4: i32, intra: &Intra) {
        let (w, h, bw, bh) = (w4 as usize * 4, h4 as usize * 4, bw4 as usize * 4, bh4 as usize * 4);
        let (x0, wy) = (x as usize * 4, (y as usize * 4) % SB);
        let n = intra.pal_sz[chroma as usize] as usize;
        let mut map = std::mem::take(&mut self.pal_map[chroma as usize]);
        map.clear();
        let key = pal_key(chroma, x, y, bw4, bh4, intra);
        if let Some(kept) = self.pal_kept.iter().find(|k| k.key == key) {
            map.extend_from_slice(&kept.map);
        } else {
            map.resize(bw * bh, 0);
            self.palette_map(chroma, x0, wy, w, h, bw, bh, n, intra, &mut map);
            self.pal_kept.push(PalKept { key, map: map.clone(), bits: None });
        }
        let planes = if chroma { 1..3 } else { 0..1 };
        for p in planes {
            let rec = &self.rec[p];
            for r in 0..bh {
                let mut dst = rec.comp.slice_mut::<BD, _>(((wy + r) * rec.stride + x0.., ..bw));
                for c in 0..bw {
                    dst[c] = intra.pal[p][map[r * bw + c] as usize];
                }
            }
        }
        self.pal_map[chroma as usize] = map;
    }

    /// The bits of a block's palette indexes in `pal_map`, counted once per palette.
    fn palette_bits(&mut self, chroma: bool, x: i32, y: i32, bw4: i32, bh4: i32, vw: usize, vh: usize, intra: &Intra) -> u32 {
        let key = pal_key(chroma, x, y, bw4, bh4, intra);
        let at = self.pal_kept.iter().position(|k| k.key == key).expect("the palette was predicted");
        if let Some(bits) = self.pal_kept[at].bits {
            return bits;
        }
        let n = intra.pal_sz[chroma as usize];
        let mut c = Counter::default();
        palette::write_indices(&mut c, &mut self.cost_cdf.m.color_map[chroma as usize][n as usize - 2], &self.pal_kept[at].map, bw4 as usize * 4, vw, vh, n);
        self.pal_kept[at].bits = Some(c.bits);
        c.bits
    }

    #[allow(clippy::too_many_arguments)]
    fn palette_map(&self, chroma: bool, x0: usize, wy: usize, w: usize, h: usize, bw: usize, bh: usize, n: usize, intra: &Intra, map: &mut [u8]) {
        for r in 0..h {
            let at = (wy + r) * self.src[0].stride + x0;
            if chroma {
                let u = self.src[1].comp.slice::<BD, _>((at.., ..w));
                let v = self.src[2].comp.slice::<BD, _>((at.., ..w));
                for c in 0..w {
                    map[r * bw + c] = palette::nearest2(&intra.pal[1][..n], &intra.pal[2][..n], (u[c], v[c]));
                }
            } else {
                let px = self.src[0].comp.slice::<BD, _>((at.., ..w));
                for c in 0..w {
                    map[r * bw + c] = palette::nearest(&intra.pal[0][..n], px[c]);
                }
            }
            let last = map[r * bw + w - 1];
            map[r * bw + w..][..bw - w].fill(last);
        }
        for r in h..bh {
            map.copy_within((h - 1) * bw..h * bw, r * bw);
        }
    }

    fn sm_flag(&self, ax: usize, ly: usize) -> i32 {
        smooth(self.above.mode[ax]) | smooth(self.left.mode[ly])
    }

    fn sm_uv_flag(&self, ax: usize, ly: usize) -> i32 {
        smooth(self.above.uvmode[ax]) | smooth(self.left.uvmode[ly])
    }

    fn top_edge(&self, plane: usize, py: i32) -> Option<(&DisjointMut<AlignedVec64<u8>>, isize)> {
        (py * 4 % SB as i32 == 0 && py > 0).then_some((&self.edge, (plane * self.edge_stride) as isize))
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_cfl(&mut self, ti: usize, x: i32, y: i32, w4: i32, h4: i32, bw4: i32, bh4: i32, tx: TxfmSize, uvtx: TxfmSize, intra: &Intra) {
        let ac = self.cfl_ac(x, y, w4, h4, bw4, bh4, tx);
        for pl in 0..2 {
            if intra.cfl[pl] == 0 {
                continue;
            }
            self.predict_cfl_plane(ti, x, y, uvtx, &ac, pl, intra.cfl[pl]);
        }
    }

    /// The luma AC a block's CfL predictions scale.
    #[allow(clippy::too_many_arguments)]
    fn cfl_ac(&self, x: i32, y: i32, w4: i32, h4: i32, bw4: i32, bh4: i32, tx: TxfmSize) -> Ac {
        let t_dim = &DAV1D_TXFM_DIMENSIONS[tx as usize];
        let wy = (y as usize * 4) % SB;
        let luma = &self.rec[0];
        let y_src = luma.comp.with_offset::<BD>() + (wy * luma.stride + x as usize * 4);
        let furthest_r = (w4 + t_dim.w as i32 - 1) & !(t_dim.w as i32 - 1);
        let furthest_b = (h4 + t_dim.h as i32 - 1) & !(t_dim.h as i32 - 1);
        let mut ac = Ac([0; SCRATCH_AC_TXTP_LEN]);
        self.dsp.ipred.cfl_ac[Rav1dPixelLayoutSubSampled::I444].call::<BD>(
            &mut ac.0,
            y_src,
            bw4 - furthest_r,
            bh4 - furthest_b,
            bw4 * 4,
            bh4 * 4,
        );
        ac
    }

    /// One chroma plane predicted from its DC and `alpha` times `ac`; 0 is the DC prediction.
    #[allow(clippy::too_many_arguments)]
    fn predict_cfl_plane(&mut self, ti: usize, x: i32, y: i32, uvtx: TxfmSize, ac: &Ac, pl: usize, alpha: i8) {
        let uv_dim = &DAV1D_TXFM_DIMENSIONS[uvtx as usize];
        let tile = &self.tiles[ti];
        let wy = (y as usize * 4) % SB;
        let rec = &self.rec[1 + pl];
        let dst = rec.comp.with_offset::<BD>() + (wy * rec.stride + x as usize * 4);
        let mut topleft = EdgePx([0; EDGE_LEN]);
        let mut angle = 0;
        let m = rav1d_prepare_intra_edges(
            x,
            x > tile.col_start,
            y,
            y > tile.row_start,
            tile.col_end,
            tile.row_end,
            EdgeFlags::empty(),
            dst,
            self.top_edge(1 + pl, y),
            DC_PRED,
            &mut angle,
            uv_dim.w as i32,
            uv_dim.h as i32,
            0,
            &mut topleft.0,
            128,
            BD::new(()),
        );
        self.dsp.ipred.cfl_pred[m as usize].call::<BD>(
            dst,
            &topleft.0,
            128,
            uv_dim.w as i32 * 4,
            uv_dim.h as i32 * 4,
            &ac.0,
            alpha as i32,
            BD::new(()),
        );
    }

    /// Predicts every transform block of one plane, chooses its residual and reconstructs it,
    /// updating the coefficient contexts as the decoder will.
    #[allow(clippy::too_many_arguments)]
    fn predict_plane(
        &mut self,
        ti: usize,
        bs: BlockSize,
        plane: usize,
        bx: i32,
        by: i32,
        w4: i32,
        h4: i32,
        tx: TxfmSize,
        intra: &Intra,
        intra_edge_flags: EdgeFlags,
        intra_flags: i32,
        predict: bool,
        out: &mut Vec<TxDone>,
    ) {
        let lossless = self.frame.quant.lossless();
        let t_dim = &DAV1D_TXFM_DIMENSIONS[tx as usize];
        let (tw, th) = (t_dim.w as i32, t_dim.h as i32);
        let tile = &self.tiles[ti];
        let (col_start, col_end, row_start, row_end) = (tile.col_start, tile.col_end, tile.row_start, tile.row_end);
        let sb_has_tr = intra_edge_flags.contains(EdgeFlags::I444_TOP_HAS_RIGHT);
        let sb_has_bl = intra_edge_flags.contains(EdgeFlags::I444_LEFT_HAS_BOTTOM);
        let mode = if plane == 0 {
            intra.y_mode
        } else if intra.uv_mode == CFL_PRED {
            DC_PRED
        } else {
            intra.uv_mode
        };
        let angle0 = if plane == 0 { intra.y_angle } else { intra.uv_angle } as i32;
        let coded_txtp = !lossless && self.frame.quant.base != 0 && plane == 0 && t_dim.max < TxfmSize::S32x32 as u8;
        let set: &[TxfmType] = if !coded_txtp {
            &[]
        } else if self.frame.reduced_tx_set || t_dim.min == TxfmSize::S16x16 as u8 {
            &DAV1D_TX_TYPES_PER_SET[..5]
        } else {
            &DAV1D_TX_TYPES_PER_SET[5..12]
        };
        let fixed_txtp = if lossless {
            WHT_WHT
        } else if plane == 0 || t_dim.max >= TxfmSize::S32x32 as u8 {
            DCT_DCT
        } else {
            DAV1D_TXTP_FROM_UVMODE[intra.uv_mode as usize]
        };
        let mut y = 0;
        while y < h4 {
            let mut x = 0;
            while x < w4 {
                let (px, py) = (bx + x, by + y);
                let wy = (py as usize * 4) % SB;
                if predict {
                    let rec = &self.rec[plane];
                    let dst = rec.comp.with_offset::<BD>() + (wy * rec.stride + px as usize * 4);
                    let edge = EdgeFlags::I444_TOP_HAS_RIGHT.select(!((y > 0 || !sb_has_tr) && x + tw >= w4))
                        | EdgeFlags::I444_LEFT_HAS_BOTTOM.select(!(x > 0 || (!sb_has_bl && y + th >= h4)));
                    let mut topleft = EdgePx([0; EDGE_LEN]);
                    let mut angle = angle0;
                    let m = rav1d_prepare_intra_edges(
                        px,
                        px > col_start,
                        py,
                        py > row_start,
                        col_end,
                        row_end,
                        edge,
                        dst,
                        self.top_edge(plane, py),
                        mode,
                        &mut angle,
                        tw,
                        th,
                        self.seq.edge_filter as u8,
                        &mut topleft.0,
                        128,
                        BD::new(()),
                    );
                    self.dsp.ipred.intra_pred[m as usize].call::<BD>(
                        dst,
                        &topleft.0,
                        128,
                        tw * 4,
                        th * 4,
                        angle | intra_flags,
                        4 * self.bw - 4 * px,
                        4 * self.bh - 4 * py,
                        BD::new(()),
                    );
                }
                let (pw, ph) = (tw as usize * 4, th as usize * 4);
                let n = cmp::min(pw, 32) * cmp::min(ph, 32);
                let mut lv = self.lv.take().unwrap();
                let levels = &mut lv[..n];
                self.recon_ready = false;
                self.residual_bits = None;
                self.residual_dist = 0;
                let mut known_eob = None;
                let mut replayed = None;
                let decided = if self.predict_only {
                    known_eob = Some(-1);
                    Some(fixed_txtp)
                } else if let Some(replay) = self.replay.as_mut() {
                    let t = replay.pop().expect("a replayed residual per transform block");
                    debug_assert!(t.plane == plane && t.tx == tx && (t.x4, t.y4) == (px, py));
                    if t.eob >= 0 {
                        levels.copy_from_slice(&t.coefs.levels);
                    }
                    self.residual_bits = t.bits;
                    known_eob = Some(t.eob);
                    let txtp = t.coefs.txtp;
                    replayed = Some(t.coefs.levels);
                    Some(txtp)
                } else if self.dry || self.decide.searches() {
                    None
                } else {
                    levels.fill(0);
                    self.decide.residual(plane, tx, set, levels)
                };
                let txtp = match decided {
                    Some(t) if self.predict_only => t,
                    Some(t) if set.is_empty() => {
                        let _ = t;
                        fixed_txtp
                    }
                    Some(t) => {
                        assert!(set.contains(&t));
                        t
                    }
                    None => {
                        let all = if set.is_empty() { std::slice::from_ref(&fixed_txtp) } else { set };
                        let mut reduced = [DCT_DCT; 7];
                        let cands = if self.dct_only && !set.is_empty() {
                            &all[1..2]
                        } else if self.reduced_txtp && set.len() == 7 {
                            let y_mode = if intra.y_mode == FILTER_PRED { DAV1D_FILTER_MODE_TO_Y_MODE[intra.y_angle as usize] } else { intra.y_mode };
                            let used = REDUCED_INTRA_TXTP[y_mode as usize];
                            let mut k = 0;
                            for &t in set.iter().filter(|&&t| used >> t as u8 & 1 != 0) {
                                reduced[k] = t;
                                k += 1;
                            }
                            &reduced[..k]
                        } else {
                            all
                        };
                        let (txtp, eob) = self.rd_residual(bs, plane, px, py, tx, cands, intra, levels);
                        known_eob = Some(eob);
                        txtp
                    }
                };
                let eob = known_eob.unwrap_or_else(|| eob_of(tx, txtp, levels));
                if eob >= 0 && self.recon_ready {
                    let rec = &self.rec[plane];
                    for r in 0..ph {
                        let row = self.scratch.comp.slice::<BD, _>((r * self.scratch.stride.., ..pw));
                        rec.comp.slice_mut::<BD, _>(((wy + r) * rec.stride + px as usize * 4.., ..pw)).copy_from_slice(&row);
                    }
                } else if eob >= 0 {
                    let mut b = self.bufs.take().unwrap();
                    self.dequant(plane, tx, txtp, levels, &mut b.cf.0[..n], n);
                    let rec = &self.rec[plane];
                    let dst = rec.comp.with_offset::<BD>() + (wy * rec.stride + px as usize * 4);
                    self.dsp.itx.itxfm_add[tx as usize][txtp as usize].call::<BD>(dst, &mut b.cf.0[..n], eob, BD::new(()));
                    self.bufs = Some(b);
                }
                let v = res_ctx(levels, eob);
                let levels = match replayed {
                    _ if eob < 0 => Vec::new(),
                    Some(kept) => kept,
                    None => {
                        let mut v = self.spare.0.pop().unwrap_or_default();
                        v.clear();
                        v.extend_from_slice(levels);
                        v
                    }
                };
                self.lv = Some(lv);
                let (a, l) = coef_ctx(&mut self.above, &mut self.left, plane);
                let cw = cmp::min(tw, self.bw - px) as usize;
                let ch = cmp::min(th, self.bh - py) as usize;
                a[px as usize..][..cw].fill(v);
                l[(py & 15) as usize..][..ch].fill(v);
                out.push(TxDone { plane, x4: px, y4: py, tx, coefs: Coefs { txtp, levels }, eob, bits: self.residual_bits });
                if let Some(limit) = self.limit {
                    self.spent.0 += self.residual_dist;
                    self.spent.1 += self.residual_bits.unwrap_or(0);
                    if self.spent.0 as f64 + self.lambda * self.spent.1 as f64 / 512.0 >= limit {
                        self.stopped = true;
                        return;
                    }
                }
                x += tw;
            }
            y += th;
        }
    }

    /// A luma block's cost at depth 0 by its DCT alone, the prediction in place: the quantized
    /// coefficients' error and their counted rate, for ranking modes.
    fn luma_estimate(&mut self, bs: BlockSize, x: i32, y: i32, intra: &Intra) -> f64 {
        let (tx, _) = self.tx_sizes(bs, 0);
        let t_dim = &DAV1D_TXFM_DIMENSIONS[tx as usize];
        let (pw, ph) = (t_dim.w as usize * 4, t_dim.h as usize * 4);
        let n = cmp::min(pw, 32) * cmp::min(ph, 32);
        let (x0, wy) = (x as usize * 4, (y as usize * 4) % SB);
        let mut b = self.bufs.take().unwrap();
        let Bufs { residual, coefs, trial, .. } = &mut *b;
        let (src, rec) = (&self.src[0], &self.rec[0]);
        let (at, len) = (wy * src.stride + x0, (ph - 1) * src.stride + pw);
        dist::residual(residual, &src.comp.slice::<BD, _>((at.., ..len)), src.stride, &rec.comp.slice::<BD, _>((at.., ..len)), rec.stride, pw, ph);
        fdct_fast::forward(&residual[..pw * ph], pw, &mut coefs[..n], tx, DCT_DCT);
        let qm = self.qm[tx as usize][0];
        let shift = cmp::max(0, t_dim.ctx as i32 - 2) as u32;
        let (eob, err, _) = quant::quantize(&coefs[..n], &mut trial[..n], self.dq[0], qm, shift, 48, scan_index(tx, TxClass::TwoD));
        let (ax, ly) = (x as usize, (y & 15) as usize);
        let (mut a, mut l) = ([0x40u8; 48], [0x40u8; 48]);
        a[..16].copy_from_slice(&self.above.lcoef[ax..][..16]);
        l[..16].copy_from_slice(&self.left.lcoef[ly..][..16]);
        let mut c = Counter::default();
        let (qidx, reduced) = (self.frame.quant.base, self.frame.reduced_tx_set);
        write_coefs(&mut c, &mut self.cost_cdf, Some(&self.coef_costs), &mut a, &mut l, 0, 0, 0, tx, bs, DCT_DCT, &trial[..n], eob, intra, false, qidx, reduced, i32::MAX, i32::MAX);
        self.bufs = Some(b);
        err as f64 / 64.0 + self.lambda * c.bits as f64 / 512.0
    }

    /// Whether a luma block is drawn rather than photographed: flat 4×4 areas (variance under
    /// 0.5) over a quarter of it beside a hard edge (a 4×4 area of variance over 16), or smooth
    /// throughout (every 4×4 area under 16), as gradients are.
    fn graphic_block(&self, x: i32, y: i32, n4: usize) -> bool {
        let (x0, y0) = ((x & 15) as usize, (y & 15) as usize);
        let (mut flat, mut most, mut all) = (0, 0i64, 0);
        for row in self.var4[y0 * 16..].chunks(16).take(n4) {
            for &v in row[x0..x0 + n4].iter().filter(|&&v| v >= 0) {
                flat += (v < 128) as usize;
                most = most.max(v);
                all += 1;
            }
        }
        all > 0 && (most < 4096 || flat * 4 >= all)
    }

    fn var4_of(&self, x: i32, y: i32) -> [i64; 256] {
        let mut out = [-1; 256];
        let w = cmp::min(64, (self.seq.width as usize).saturating_sub(x as usize * 4)) & !3;
        let h = cmp::min(64, (self.seq.height as usize).saturating_sub(y as usize * 4)) & !3;
        if w == 0 || h == 0 {
            return out;
        }
        let px = self.source(0, x, y, w, h);
        for by in (0..h).step_by(4) {
            for bx in (0..w).step_by(4) {
                let (mut s, mut q) = (0i64, 0i64);
                for row in px[by * w + bx..].chunks(w).take(4) {
                    for &p in &row[..4] {
                        s += p as i64;
                        q += p as i64 * p as i64;
                    }
                }
                out[by / 4 * 16 + bx / 4] = 16 * q - s * s;
            }
        }
        out
    }

    /// Which of four activity classes a 16×16 luma block at `(x, y)` (in 4-pixel units) belongs
    /// to, by its variance: under 4, 32, 256, and above.
    fn activity_bucket(&self, x: i32, y: i32, n4: usize) -> usize {
        let x0 = x as usize * 4;
        let w = cmp::min(n4 * 4, (self.seq.width as usize).saturating_sub(x0));
        let h = cmp::min(n4 * 4, (self.seq.height as usize).saturating_sub(y as usize * 4));
        if w == 0 || h == 0 {
            return 0;
        }
        let px = self.source(0, x, y, w, h);
        let n = (w * h) as u64;
        let (sum, sse) = px.iter().fold((0u64, 0u64), |(s, q), &p| (s + p as u64, q + (p as u64) * (p as u64)));
        let var = (sse * n - sum * sum) / (n * n);
        match var {
            ..4 => 0,
            ..32 => 1,
            ..256 => 2,
            _ => 3,
        }
    }

    /// Chooses a transform type from `cands`, its levels and their end of block by rate and
    /// distortion; the prediction is in place in the reconstruction.
    #[allow(clippy::too_many_arguments)]
    fn rd_residual(
        &mut self,
        bs: BlockSize,
        plane: usize,
        px: i32,
        py: i32,
        tx: TxfmSize,
        cands: &[TxfmType],
        intra: &Intra,
        levels: &mut [i32],
    ) -> (TxfmType, i32) {
        let t_dim = &DAV1D_TXFM_DIMENSIONS[tx as usize];
        let (pw, ph) = (t_dim.w as usize * 4, t_dim.h as usize * 4);
        let wy = (py as usize * 4) % SB;
        let x0 = px as usize * 4;
        let vw = cmp::min(pw, (self.seq.width as usize).saturating_sub(x0));
        let vh = cmp::min(ph, (self.seq.height as usize).saturating_sub(py as usize * 4));
        let n = levels.len();
        self.stats[self.stage][0] += 1;
        self.stats[self.stage][1] += (n * cands.len()) as u64;
        let mut b = self.bufs.take().unwrap();
        let Bufs { src, pred, residual, coefs, trial, cf } = &mut *b;
        let stride = self.src[plane].stride;
        let (at, len) = (wy * stride + x0, (ph - 1) * stride + pw);
        copy_rect(src, pw, &self.src[plane].comp.slice::<BD, _>((at.., ..len)), stride, pw, ph);
        copy_rect(pred, pw, &self.rec[plane].comp.slice::<BD, _>((at.., ..len)), stride, pw, ph);
        dist::residual(residual, src, pw, pred, pw, pw, ph);
        let block = bs.dimensions();
        self.skip_predicted = plane == 0
            && self.skip_prediction
            && !self.frame.quant.lossless()
            && (t_dim.w, t_dim.h) == (block[0], block[1])
            && predicts_skip(&residual[..pw * ph], pw, ph, vw, vh, self.dq[0]);
        if self.skip_predicted {
            levels.fill(0);
            self.bufs = Some(b);
            return (cands[0], -1);
        }
        let sse = |a: &[u8], b: &[u8], stride: usize| dist::sse(a, pw, b, stride, vw, vh);
        let (a0, l0) = {
            let (a, l) = coef_ctx(&mut self.above, &mut self.left, plane);
            let mut a0 = [0x40u8; 48];
            let mut l0 = [0x40u8; 48];
            a0[..16].copy_from_slice(&a[px as usize..][..16]);
            l0[..16].copy_from_slice(&l[(py & 15) as usize..][..16]);
            (a0, l0)
        };
        let lossless = self.frame.quant.lossless();
        let rate = |e: &mut Self, txtp: TxfmType, lv: &[i32], eob: i32| -> u32 {
            let (mut a, mut l) = (a0, l0);
            let mut c = Counter::default();
            write_coefs(&mut c, &mut e.cost_cdf, Some(&e.coef_costs), &mut a, &mut l, plane, 0, 0, tx, bs, txtp, lv, eob, intra, lossless, e.frame.quant.base, e.frame.reduced_tx_set, i32::MAX, i32::MAX);
            c.bits
        };
        let zero_rate = rate(self, cands[0], &[], -1);
        self.residual_bits = Some(zero_rate);
        let zero_dist = if lossless { 0 } else { sse(&src[..], &pred[..], pw) };
        self.residual_dist = zero_dist;
        let zero = if lossless { f64::MAX } else { zero_dist as f64 + self.lambda * zero_rate as f64 / 512.0 };
        let mut best = (zero, cands[0], -1);
        levels.fill(0);
        let shift = cmp::max(0, t_dim.ctx as i32 - 2) as u32;
        if self.tx_domain_dist && cands.len() > 1 && self.rdoq == 0 && !lossless && (vw, vh) == (pw, ph) {
            let mut winner: Option<(f64, TxfmType, i32, u32, usize)> = None;
            let mut columns = fdct_fast::Columns::new();
            for &txtp in cands {
                columns.forward(&residual[..pw * ph], pw, &mut coefs[..n], tx, txtp);
                let qm = if txtp < IDTX { self.qm[tx as usize][plane] } else { None };
                let (eob, err, last) = quant::quantize(&coefs[..n], &mut trial[..n], self.dq[plane], qm, shift, 48, scan_index(tx, DAV1D_TX_TYPE_CLASS[txtp as usize]));
                if eob < 0 {
                    continue;
                }
                let bits = rate(self, txtp, &trial[..n], eob);
                let cost = err as f64 / 64.0 + self.lambda * bits as f64 / 512.0;
                if winner.is_none_or(|w| cost < w.0) {
                    winner = Some((cost, txtp, eob, bits, last));
                    levels.copy_from_slice(&trial[..n]);
                }
            }
            if let Some((_, txtp, eob, bits, last)) = winner {
                self.dequant(plane, tx, txtp, levels, &mut cf.0[..n], last);
                self.scratch.write(0, 0, pw, ph, &pred[..pw * ph]);
                let dst = self.scratch.comp.with_offset::<BD>();
                self.dsp.itx.itxfm_add[tx as usize][txtp as usize].call::<BD>(dst, &mut cf.0[..n], eob, BD::new(()));
                let dist = sse(&src[..], &self.scratch.comp.slice::<BD, _>((0.., ..SB * SB)), SB);
                if dist as f64 + self.lambda * bits as f64 / 512.0 < zero {
                    self.recon_ready = true;
                    self.residual_bits = Some(bits);
                    self.residual_dist = dist;
                    self.bufs = Some(b);
                    return (txtp, eob);
                }
                levels.fill(0);
            }
            self.bufs = Some(b);
            return (cands[0], -1);
        }
        let mut last = None;
        let mut columns = fdct_fast::Columns::new();
        for &txtp in cands {
            columns.forward(&residual[..pw * ph], pw, &mut coefs[..n], tx, txtp);
            let qm = if txtp < IDTX { self.qm[tx as usize][plane] } else { None };
            let index = scan_index(tx, DAV1D_TX_TYPE_CLASS[txtp as usize]);
            let (eob, _, used) = quant::quantize(&coefs[..n], &mut trial[..n], self.dq[plane], qm, shift, if lossless { 64 } else { 48 }, index);
            if eob < 0 {
                continue;
            }
            let mut eob = eob;
            let mut bits = rate(self, txtp, &trial[..n], eob);
            let reconstruct = |e: &mut Self, lv: &[i32], eob: i32, cf: &mut Coef| -> u64 {
                e.dequant(plane, tx, txtp, lv, &mut cf.0[..n], used);
                e.scratch.write(0, 0, pw, ph, &pred[..pw * ph]);
                let dst = e.scratch.comp.with_offset::<BD>();
                e.dsp.itx.itxfm_add[tx as usize][txtp as usize].call::<BD>(dst, &mut cf.0[..n], eob, BD::new(()));
                sse(&src[..], &e.scratch.comp.slice::<BD, _>((0.., ..SB * SB)), SB)
            };
            let mut dist = reconstruct(self, &trial[..n], eob, cf);
            last = Some(txtp);
            if self.rdoq > 0 && !self.dct_only && !lossless {
                let qm = if txtp < IDTX { self.qm[tx as usize][plane] } else { None };
                let dqp = self.dq[plane];
                let step = |rc: usize| {
                    let q = dqp[(rc != 0) as usize];
                    qm.map_or(q, |m| q * m[rc] as u32 + 16 >> 5)
                };
                let err = |rc: usize, l: i32| {
                    let r = ((step(rc).wrapping_mul(l.unsigned_abs()) & 0xffffff) >> shift) as f64;
                    let d = coefs[rc].abs() as f64 - r;
                    d * d
                };
                let coef_dist: f64 = (0..n).map(|rc| err(rc, trial[rc])).sum();
                if coef_dist > 0.0 {
                    let k = dist as f64 / coef_dist;
                    let class = DAV1D_TX_TYPE_CLASS[txtp as usize];
                    let mut tried = 0;
                    let mut i = eob;
                    let mut changed = false;
                    while i >= 0 && tried < self.rdoq {
                        let rc = scan_rc(tx, class, i as usize).0;
                        let l = trial[rc];
                        if l != 0 {
                            tried += 1;
                            let lower = l - l.signum();
                            trial[rc] = lower;
                            let new_eob = if lower == 0 && i == eob { eob_of(tx, txtp, &trial[..n]) } else { eob };
                            let new_bits = if new_eob < 0 { u32::MAX } else { rate(self, txtp, &trial[..n], new_eob) };
                            let delta = k * (err(rc, lower) - err(rc, l)) + self.lambda * (new_bits as f64 - bits as f64) / 512.0;
                            if new_eob >= 0 && delta < 0.0 {
                                bits = new_bits;
                                eob = new_eob;
                                changed = true;
                            } else {
                                trial[rc] = l;
                            }
                        }
                        i -= 1;
                    }
                    if changed {
                        dist = reconstruct(self, &trial[..n], eob, cf);
                    }
                }
            }
            let cost = dist as f64 + self.lambda * bits as f64 / 512.0;
            if cost < best.0 {
                best = (cost, txtp, eob);
                levels.copy_from_slice(&trial[..n]);
                self.residual_bits = Some(bits);
                self.residual_dist = dist;
            }
        }
        self.recon_ready = best.0 < zero && last == Some(best.1);
        self.bufs = Some(b);
        (best.1, best.2)
    }

    /// `levels` beyond the `last` in raster order are 0 and leave 0 in `cf`.
    fn dequant(&self, plane: usize, tx: TxfmSize, txtp: TxfmType, levels: &[i32], cf: &mut [i16], last: usize) {
        let t_dim = &DAV1D_TXFM_DIMENSIONS[tx as usize];
        let shift = cmp::max(0, t_dim.ctx as i32 - 2) as u32;
        let qm = if txtp < IDTX { self.qm[tx as usize][plane] } else { None };
        let m = last.next_multiple_of(8).min(levels.len());
        quant::dequantize(&levels[..m], &mut cf[..m], self.dq[plane], qm, shift);
        cf[m..].fill(0);
    }

    /// The squared error of the reconstruction in a block's visible area, in the given planes.
    fn sse(&self, x: i32, y: i32, w4: usize, h4: usize, planes: Range<usize>) -> u64 {
        let x0 = x as usize * 4;
        let w = cmp::min(w4 * 4, (self.seq.width as usize).saturating_sub(x0));
        let h = cmp::min(h4 * 4, (self.seq.height as usize).saturating_sub(y as usize * 4));
        let wy = (y as usize * 4) % SB;
        if w == 0 || h == 0 {
            return 0;
        }
        let mut total = 0;
        for p in planes {
            let (src, rec) = (&self.src[p], &self.rec[p]);
            let a = src.comp.slice::<BD, _>((wy * src.stride + x0.., ..(h - 1) * src.stride + w));
            let b = rec.comp.slice::<BD, _>((wy * rec.stride + x0.., ..(h - 1) * rec.stride + w));
            total += dist::sse(&a, src.stride, &b, rec.stride, w, h);
        }
        total
    }

    /// The source pixels of a plane in `w`×`h` at a block's corner.
    fn source(&self, plane: usize, x: i32, y: i32, w: usize, h: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(w * h);
        self.src[plane].read(x as usize * 4, (y as usize * 4) % SB, w, h, &mut out);
        out
    }

    /// The sum of absolute 4×4 Hadamard coefficients of the luma residual in a block's visible
    /// area, rounded out to whole 4×4 blocks.
    fn satd(&self, x: i32, y: i32, w4: usize, h4: usize) -> u64 {
        self.satd_plane(0, x, y, w4, h4)
    }

    fn satd_plane(&self, plane: usize, x: i32, y: i32, w4: usize, h4: usize) -> u64 {
        let x0 = x as usize * 4;
        let w = cmp::min(w4 * 4, (self.seq.width as usize).saturating_sub(x0)).next_multiple_of(4);
        let h = cmp::min(h4 * 4, (self.seq.height as usize).saturating_sub(y as usize * 4)).next_multiple_of(4);
        if w == 0 || h == 0 {
            return 0;
        }
        let (src, rec) = (&self.src[plane], &self.rec[plane]);
        let at = (y as usize * 4) % SB * src.stride + x0;
        let len = (h - 1) * src.stride + w;
        dist::satd(&src.comp.slice::<BD, _>((at.., ..len)), &rec.comp.slice::<BD, _>((at.., ..len)), src.stride, w, h)
    }

    fn snap(&self, x: i32, y: i32, w4: usize, h4: usize) -> Snap {
        let mut pixels = Vec::new();
        let wy = (y as usize * 4) % SB;
        for p in &self.rec {
            p.read(x as usize * 4, wy, w4 * 4, h4 * 4, &mut pixels);
        }
        let (ax, ly) = (x as usize, (y & 15) as usize);
        Snap {
            x,
            y,
            w4,
            h4,
            pixels,
            above: self.above.save(ax..ax + w4),
            left: self.left.save(ly..ly + h4),
        }
    }

    fn restore(&mut self, s: &Snap) {
        let wy = (s.y as usize * 4) % SB;
        let n = s.w4 * s.h4 * 16;
        for (i, p) in self.rec.iter().enumerate() {
            p.write(s.x as usize * 4, wy, s.w4 * 4, s.h4 * 4, &s.pixels[i * n..][..n]);
        }
        self.restore_contexts(s);
    }

    /// `restore` for a trial within the block: it predicts each transform block from pixels
    /// outside the block or ones it coded itself, as the edge flags allow, and overwrites the
    /// planes it measures, so their earlier pixels need no restoring.
    fn restore_contexts(&mut self, s: &Snap) {
        self.above.restore(s.x as usize, &s.above);
        self.left.restore((s.y & 15) as usize, &s.left);
    }

    fn lap(&mut self, clock: &mut Option<std::time::Instant>, stage: usize) {
        if let Some(t) = clock {
            let now = std::time::Instant::now();
            self.stage_ns[stage] += (now - *t).as_nanos() as u64;
            *t = now;
        }
    }

    fn rd_cost(&self, dist: u64, bits: u32) -> f64 {
        dist as f64 + self.lambda * bits as f64 / 512.0
    }

    /// Searches the superblock and leaves its plan to be coded.
    fn search_sb(&mut self, ti: usize, x: i32, y: i32) {
        self.plan.clear();
        for txs in std::mem::take(&mut self.chosen) {
            self.recycle(txs);
        }
        self.next = 0;
        let s = self.snap(x, y, 16, 16);
        self.dry = true;
        self.counter = Counter::default();
        self.cnn = (self.cnn_prune != 0 && x + 16 <= self.bw && y + 16 <= self.bh).then(|| Box::new(self.cnn_of(x)));
        if self.graphic_guard {
            self.var4 = self.var4_of(x, y);
        }
        self.pick_partition(ti, BlockLevel::Bl64x64, x, y, EdgeIndex::root());
        self.estimated += self.counter.bits as u64;
        self.estimated_sse += self.sse(x, y, 16, 16, 0..self.planes);
        self.dry = false;
        self.restore(&s);
    }

    /// The CNN of the superblock at column `x` of the current row: its luma from one row above
    /// and one column left, the source repeated past the picture as libaom extends it.
    fn cnn_of(&self, x: i32) -> Cnn {
        let src = &self.src[0];
        let x0 = x as usize * 4;
        let mut px = [0u8; 65 * 65];
        for (r, row) in px.chunks_exact_mut(65).enumerate() {
            let line = match r {
                0 if self.sby == 0 => src.row(0, 0, src.stride),
                0 => self.src_above.clone(),
                _ => src.row(r - 1, 0, src.stride),
            };
            row[0] = line[x0.saturating_sub(1)];
            row[1..].copy_from_slice(&line[x0..x0 + 64]);
        }
        Cnn::new(&px, self.sb_qidx)
    }

    /// Finds the best partition of a block and leaves it applied; its nodes go on the plan and
    /// its bits on the counter.
    fn pick_partition(&mut self, ti: usize, bl: BlockLevel, x: i32, y: i32, edge_index: EdgeIndex) {
        let edges = &IntraEdges::DEFAULT;
        let Some(halves) = self.halves(bl, x, y) else {
            return self.pick_partition(ti, bl.decrease().unwrap(), x, y, edges.branch(false, edge_index).split[0]);
        };
        let n4 = 32 >> bl as u8;
        self.block_lambda(x, y, n4, n4);
        let lambda = self.lambda;
        let start = self.snap(x, y, n4, n4);
        let plan_len = self.plan.len();
        let base = self.counter.bits;
        let ctx = self.partition_ctx(bl, x, y);
        use BlockPartition::*;
        let cands: &[BlockPartition] = match (halves, self.rect) {
            (Halves::Both, 0) => &[None, Split],
            (Halves::Both, _) => &[None, Split, H, V],
            (Halves::Top, _) => &[Split, H],
            (Halves::Left, _) => &[Split, V],
        };
        let level = bl as u8 - BlockLevel::Bl64x64 as u8;
        let split_allowed = level + 1 < self.levels.end;
        let none_allowed = !(level < self.levels.start && split_allowed);
        let cnn = match &self.cnn {
            Some(cnn) if halves == Halves::Both && level < 4 && x + n4 as i32 <= self.bw && y + n4 as i32 <= self.bh => {
                let (bx, by) = ((x & 15) as usize / n4, (y & 15) as usize / n4);
                cnn.prune(level as usize, bx, by, self.cnn_prune, self.seq.width, self.seq.height)
            }
            _ => Prune::default(),
        };
        let mut graphic = Option::None;
        let mut is_graphic = |e: &Self| *graphic.get_or_insert_with(|| e.graphic_block(x, y, n4));
        let cnn = if self.graphic_guard && (cnn.none || cnn.split || cnn.rect) && is_graphic(self) { Prune::default() } else { cnn };
        // The best candidate's pixels and contexts are snapped only when a later candidate is
        // about to overwrite them; a best left in place needs neither the snap nor the restore.
        let mut best: Option<(f64, u32, Option<Snap>, Vec<Node>)> = Option::None;
        let mut dirty = false;
        let (mut none_cost, mut split_cost) = (f64::INFINITY, f64::INFINITY);
        let mut rect_under = false;
        let split8 = self.split8_adaptive && level == 2 && halves == Halves::Both && none_allowed && split_allowed;
        // The block's own activity picks the counters it learns from: a flat region teaches
        // nothing about the edges and textures beside it.
        let bucket = if split8 || (self.rect_adaptive || self.graphic_guard) && self.rect_ratio > 0.0 { self.activity_bucket(x, y, n4) } else { 0 };
        let prune8 = split8 && {
            let seen = &mut self.split_seen[bucket];
            seen.0 += 1;
            let (met, tried, gained, total, big) = *seen;
            tried >= 32 && gained <= total * 0.05 && big * 4 <= tried && met % 16 != 0
        };
        for &bp in cands {
            if halves == Halves::Both && cands.len() > 1 {
                // The CNN may prune None only where Split is searched: with Split pruned as
                // well, by the CNN or the learnt prune, nothing would be left to code.
                let split_pruned = cnn.split && none_allowed || prune8;
                let skip = match bp {
                    Split => !split_allowed || split_pruned,
                    None => !none_allowed || cnn.none && split_allowed && !split_pruned,
                    _ => !none_allowed || cnn.rect,
                };
                if skip {
                    continue;
                }
            }
            if matches!(bp, H | V) && halves == Halves::Both && self.rect >= 2 {
                if let Some([Node::Part(None), Node::Block(intra, ..), ..]) = best.as_ref().map(|b| b.3.as_slice()) {
                    let along = match bp {
                        V => matches!(intra.y_mode, VERT_PRED | VERT_LEFT_PRED | VERT_RIGHT_PRED),
                        _ => matches!(intra.y_mode, HOR_PRED | HOR_DOWN_PRED | HOR_UP_PRED),
                    };
                    if !(self.rect == 3 && along) {
                        continue;
                    }
                }
            }
            if matches!(bp, H | V) && halves == Halves::Both && self.rect >= 1 {
                // A drawn block takes the full search's learnt prune at its ratio instead.
                let (ratio, adaptive) = if self.graphic_guard && !self.rect_adaptive && is_graphic(self) { (0.9, true) } else { (self.rect_ratio, self.rect_adaptive) };
                if split_cost < ratio * none_cost {
                    if !adaptive {
                        continue;
                    }
                    let seen = &mut self.rect_seen[level as usize][bucket];
                    if !rect_under {
                        seen.0 += 1;
                    }
                    let (met, tried, gained, total, big) = *seen;
                    if tried >= 32 && gained <= total * 0.05 && big * 4 <= tried && met % 16 != 0 {
                        continue;
                    }
                    if !rect_under {
                        rect_under = true;
                        seen.1 += 1;
                    }
                }
            }
            if bp != None && self.prune > 0.0 && best.as_ref().is_some_and(|b| b.0 < self.prune * self.lambda * (n4 * n4) as f64) {
                break;
            }
            if dirty {
                if let Some(b) = best.as_mut().filter(|b| b.2.is_none()) {
                    b.2 = Some(self.snap(x, y, n4, n4));
                }
                self.restore(&start);
            }
            dirty = true;
            self.plan.truncate(plan_len);
            self.counter.bits = base;
            self.plan.push(Node::Part(bp));
            Self::write_partition(&mut self.counter, &mut self.cost_cdf, bl, ctx, halves, bp);
            // A candidate's cost only grows block by block, so one that reaches the best so far
            // stops: its decided blocks' distortion and bits at this block's λ.
            let limit = best.as_ref().map_or(f64::INFINITY, |b| b.0);
            let done = std::cell::Cell::new((0u64, false));
            let over = |e: &Self, x: i32, y: i32, w4: usize, h4: usize| {
                let dist = done.get().0 + e.sse(x, y, w4, h4, 0..e.planes);
                let over = dist as f64 + lambda * (e.counter.bits - base) as f64 / 512.0 >= limit;
                done.set((dist, over));
            };
            self.code_partition(
                ti,
                bl,
                bp,
                x,
                y,
                edge_index,
                halves,
                &mut |e, ti, bs, x, y, f| {
                    if !done.get().1 {
                        e.pick_block(ti, bs, x, y, f);
                        let d = bs.dimensions();
                        over(e, x, y, d[0] as usize, d[1] as usize);
                    }
                },
                &mut |e, ti, bl, x, y, edge_index| {
                    if !done.get().1 {
                        e.pick_partition(ti, bl, x, y, edge_index);
                        let n = 32 >> bl as u8;
                        over(e, x, y, n, n);
                    }
                },
            );
            if done.get().1 {
                self.lambda = lambda;
                continue;
            }
            let bits = self.counter.bits - base;
            self.block_lambda(x, y, n4, n4);
            let cost = self.rd_cost(self.sse(x, y, n4, n4, 0..self.planes), bits);
            match bp {
                None => none_cost = cost,
                Split => split_cost = cost,
                _ => {}
            }
            if best.as_ref().is_none_or(|b| cost < b.0) {
                best = Some((cost, bits, Option::None, self.plan[plan_len..].to_vec()));
            }
        }
        let (cost, bits, snap, nodes) = best.unwrap();
        if split8 && !prune8 && none_cost.is_finite() {
            let seen = &mut self.split_seen[bucket];
            seen.1 += 1;
            seen.3 += cost;
            if !matches!(nodes[0], Node::Part(None)) {
                seen.2 += none_cost - cost;
                if none_cost - cost > 0.02 * none_cost {
                    seen.4 += 1;
                    // A big win after the prune engaged: the content changed, so learn afresh.
                    if seen.1 > 32 && seen.4 * 4 <= seen.1 {
                        *seen = (seen.0, 0, 0.0, 0.0, 0);
                    }
                }
            }
        }
        if rect_under {
            let seen = &mut self.rect_seen[level as usize][bucket];
            seen.3 += cost;
            if matches!(nodes[0], Node::Part(H | V)) {
                let gain = none_cost.min(split_cost) - cost;
                seen.2 += gain;
                if gain > 0.02 * cost {
                    seen.4 += 1;
                }
            }
        }
        if let Some(snap) = snap {
            self.restore(&snap);
        }
        self.counter.bits = base + bits;
        self.plan.truncate(plan_len);
        self.plan.extend(nodes);
    }

    /// Finds the best modes of a block and leaves them applied; its node goes on the plan and its
    /// bits on the counter.
    fn pick_block(&mut self, ti: usize, bs: BlockSize, x: i32, y: i32, edge_flags: EdgeFlags) {
        let started = self.timing.then(std::time::Instant::now);
        self.pick_block_timed(ti, bs, x, y, edge_flags);
        if let Some(t) = started {
            let d = bs.dimensions();
            let side = cmp::max(d[0], d[1]) as u32;
            self.size_ns[side.trailing_zeros() as usize] += t.elapsed().as_nanos() as u64;
        }
    }

    fn pick_block_timed(&mut self, ti: usize, bs: BlockSize, x: i32, y: i32, edge_flags: EdgeFlags) {
        self.pal_kept.clear();
        let b_dim = bs.dimensions();
        let (w4, h4) = (b_dim[0] as usize, b_dim[1] as usize);
        self.block_lambda(x, y, w4, h4);
        let start = self.snap(x, y, w4, h4);
        let (limits, max_tx) = self.limits(bs);
        let (ax, ly) = (x as usize, (y & 15) as usize);
        let lossless = self.frame.quant.lossless();
        let tx_select = self.frame.tx_select && !lossless;
        let (vw, vh) = ((cmp::min(w4 as i32, self.bw - x) * 4) as usize, (cmp::min(h4 as i32, self.bh - y) * 4) as usize);
        let mode_bits = |e: &mut Self, intra: &Intra, depth: u8| -> u32 {
            let mut c = Counter::default();
            write_luma_modes(&mut c, &mut e.cost_cdf, &e.above, &e.left, ax, ly, intra, &limits);
            if limits.palette {
                write_pal_y(&mut c, &mut e.cost_cdf, &e.above, &e.left, ax, ly, bs, intra);
            }
            write_filter(&mut c, &mut e.cost_cdf, bs, intra, &limits);
            if intra.pal_sz[0] > 0 {
                c.bits += e.palette_bits(false, x, y, w4 as i32, h4 as i32, vw, vh, intra);
            }
            write_depth(&mut c, &mut e.cost_cdf, &e.above, &e.left, ax, ly, depth, max_tx, tx_select);
            c.bits
        };

        let luma = |e: &mut Self, intra: &Intra, depth: u8, limit: Option<f64>| -> (f64, Vec<TxDone>) {
            e.restore_contexts(&start);
            (e.limit, e.spent) = (limit, (0, 0));
            let txs = e.predict_block(ti, bs, x, y, edge_flags, intra, depth, 0..1);
            e.limit = None;
            if std::mem::take(&mut e.stopped) {
                return (f64::INFINITY, txs);
            }
            let bits = mode_bits(e, intra, depth) + coef_bits(e, &txs, bs, intra);
            let dist = e.sse(x, y, w4, h4, 0..1);
            (e.rd_cost(dist, bits), txs)
        };

        let keep = self.keep_modes;
        let sad_lambda = self.lambda.sqrt();
        let inside = (x as u32 + w4 as u32) * 4 <= self.seq.width && (y as u32 + h4 as u32) * 4 <= self.seq.height;
        let rank_rd = self.rank_rd && !self.frame.quant.lossless();
        let pre = if rank_rd && inside { self.rank_pre } else { 0 };
        let score = |e: &mut Self, intra: &Intra, estimate: bool| -> f64 {
            if keep == 0 {
                let (cost, txs) = luma(e, intra, 0, None);
                e.recycle(txs);
                return cost;
            }
            let mut clock = e.timing.then(std::time::Instant::now);
            e.restore_contexts(&start);
            e.lap(&mut clock, 6);
            e.predict_only = true;
            let txs = e.predict_block(ti, bs, x, y, edge_flags, intra, 0, 0..1);
            e.recycle(txs);
            e.predict_only = false;
            e.lap(&mut clock, 7);
            let bits = mode_bits(e, intra, 0);
            e.lap(&mut clock, 8);
            if estimate {
                let v = e.luma_estimate(bs, x, y, intra) + e.lambda * bits as f64 / 512.0;
                e.lap(&mut clock, 9);
                return v;
            }
            e.satd(x, y, w4, h4) as f64 + sad_lambda * bits as f64 / 512.0
        };

        self.stage = 0;
        let mut clock = self.timing.then(std::time::Instant::now);
        self.dct_only = true;
        let prune_intra = self.prune_intra;
        let mut scored: Vec<(f64, Intra)> = (0..N_INTRA_PRED_MODES as u8)
            .filter(|&m| !(prune_intra && matches!(m, SMOOTH_V_PRED | SMOOTH_H_PRED)))
            .map(|m| {
                let c = Intra { y_mode: m, ..Default::default() };
                (score(self, &c, rank_rd && pre == 0), c)
            })
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        if limits.angles {
            let dirs: Vec<Intra> = scored
                .iter()
                .filter(|s| (VERT_PRED..=VERT_LEFT_PRED).contains(&s.1.y_mode))
                .take(2)
                .map(|s| s.1)
                .collect();
            for d in dirs {
                for a in [-3, -2, -1, 1, 2, 3] {
                    let c = Intra { y_angle: a, ..d };
                    scored.push((score(self, &c, rank_rd && pre == 0), c));
                }
            }
        }
        if limits.filter {
            let best_so_far = scored[0].1.y_mode as usize;
            for f in 0..5 {
                if prune_intra && FILTER_FROM_BEST[best_so_far] >> f & 1 == 0 {
                    continue;
                }
                let c = Intra { y_mode: FILTER_PRED, y_angle: f, ..Default::default() };
                scored.push((score(self, &c, rank_rd && pre == 0), c));
            }
        }
        if limits.palette {
            for pal in palette::candidates_y(&self.source(0, x, y, vw, vh)) {
                let mut c = Intra { y_mode: DC_PRED, ..Default::default() };
                c.pal_sz[0] = pal.len() as u8;
                c.pal[0][..pal.len()].copy_from_slice(&pal);
                scored.push((score(self, &c, rank_rd && pre == 0), c));
            }
        }
        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        if pre > 0 {
            scored.truncate(pre);
            for s in scored.iter_mut() {
                s.0 = score(self, &s.1, true);
            }
            scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        }
        self.lap(&mut clock, 0);
        // The kept modes' depth-0 codes are the depth search's where that too is DCT-only.
        let mut coded: Vec<(Intra, f64, Vec<TxDone>, bool)> = Vec::new();
        if keep > 0 {
            scored.truncate(keep);
            for s in scored.iter_mut() {
                let (cost, txs) = luma(self, &s.1, 0, None);
                s.0 = cost;
                if self.late_txtp {
                    coded.push((s.1, cost, txs, self.skip_predicted));
                } else {
                    self.recycle(txs);
                }
            }
            scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        }
        self.lap(&mut clock, 3);
        self.dct_only = self.late_txtp;
        self.stage = 1;
        let depths = if inside { self.tx_depths } else { 2 };
        let mut best_luma = (f64::MAX, scored[0].1, 0u8);
        let mut luma_txs = Vec::new();
        for &(_, c) in scored.iter().take(2) {
            for depth in 0..=cmp::min(limits.max_depth, depths) {
                let done = (depth == 0).then(|| coded.iter().position(|d| d.0 == c)).flatten();
                let (cost, txs) = match done {
                    Some(i) => {
                        let (_, cost, txs, skip) = coded.swap_remove(i);
                        self.skip_predicted = skip;
                        (cost, txs)
                    }
                    None => luma(self, &c, depth, Some(best_luma.0)),
                };
                if cost < best_luma.0 {
                    best_luma = (cost, c, depth);
                    let old = std::mem::replace(&mut luma_txs, txs);
                    self.recycle(old);
                } else {
                    self.recycle(txs);
                }
                if self.skip_predicted {
                    break;
                }
            }
        }
        for (_, _, txs, _) in coded {
            self.recycle(txs);
        }
        self.dct_only = false;
        let (_, mut intra, depth) = best_luma;
        let mut chroma_txs = Vec::new();
        if self.late_txtp {
            self.restore(&start);
            let txs = self.predict_block(ti, bs, x, y, edge_flags, &intra, depth, 0..1);
            let old = std::mem::replace(&mut luma_txs, txs);
            self.recycle(old);
        } else if limits.chroma {
            self.restore(&start);
            luma_txs.reverse();
            self.replay = Some(luma_txs);
            luma_txs = self.predict_block(ti, bs, x, y, edge_flags, &intra, depth, 0..1);
            if let Some(r) = self.replay.take() {
                self.recycle(r);
            }
        }
        self.lap(&mut clock, 1);

        if limits.chroma {
            let luma_done = self.snap(x, y, w4, h4);
            self.stage = 2;
            let chroma = |e: &mut Self, intra: &Intra, limit: f64| -> (f64, Vec<TxDone>) {
                e.restore_contexts(&luma_done);
                (e.limit, e.spent) = (Some(limit), (0, 0));
                let txs = e.predict_block(ti, bs, x, y, edge_flags, intra, depth, 1..3);
                e.limit = None;
                if std::mem::take(&mut e.stopped) {
                    return (f64::INFINITY, txs);
                }
                let mut c = Counter::default();
                write_chroma_modes(&mut c, &mut e.cost_cdf, intra, &limits);
                if limits.palette {
                    write_pal_uv(&mut c, &mut e.cost_cdf, &e.above, &e.left, ax, ly, bs, intra);
                }
                if intra.pal_sz[1] > 0 {
                    c.bits += e.palette_bits(true, x, y, w4 as i32, h4 as i32, vw, vh, intra);
                }
                let bits = c.bits + coef_bits(e, &txs, bs, intra);
                let dist = e.sse(x, y, w4, h4, 1..3);
                (e.rd_cost(dist, bits), txs)
            };
            let mut best = (f64::MAX, intra);
            let mut modes: Vec<(f64, Intra)> = (0..N_UV_INTRA_PRED_MODES as u8 - 1)
                .map(|m| {
                    let mut c = Intra { uv_mode: m, uv_angle: 0, cfl: [0; 2], ..intra };
                    c.pal_sz[1] = 0;
                    (0.0, c)
                })
                .collect();
            let keep_chroma = self.keep_chroma;
            if self.uv_from_luma {
                let y_mode = if intra.y_mode == FILTER_PRED { DC_PRED } else { intra.y_mode };
                modes.retain(|m| UV_FROM_LUMA[y_mode as usize] >> m.1.uv_mode & 1 != 0);
            } else if keep_chroma > 0 {
                for m in modes.iter_mut() {
                    self.restore_contexts(&luma_done);
                    self.predict_only = true;
                    let txs = self.predict_block(ti, bs, x, y, edge_flags, &m.1, depth, 1..3);
                    self.recycle(txs);
                    self.predict_only = false;
                    let mut c = Counter::default();
                    write_chroma_modes(&mut c, &mut self.cost_cdf, &m.1, &limits);
                    let satd = self.satd_plane(1, x, y, w4, h4) + self.satd_plane(2, x, y, w4, h4);
                    m.0 = satd as f64 + sad_lambda * c.bits as f64 / 512.0;
                }
                modes.sort_by(|a, b| a.0.total_cmp(&b.0));
                modes.truncate(keep_chroma);
            }
            for &(_, c) in &modes {
                let (cost, txs) = chroma(self, &c, best.0);
                if cost < best.0 {
                    best = (cost, c);
                    let old = std::mem::replace(&mut chroma_txs, txs);
                    self.recycle(old);
                } else {
                    self.recycle(txs);
                }
            }
            if limits.angles && (VERT_PRED..=VERT_LEFT_PRED).contains(&best.1.uv_mode) {
                let d = best.1;
                let outward = self.uv_outward;
                let deltas: &[[i8; 3]] = if outward { &[[-1, -2, -3], [1, 2, 3]] } else { &[[-3, -2, -1], [1, 2, 3]] };
                for run in deltas {
                    for &a in run {
                        let c = Intra { uv_angle: a, ..d };
                        let (cost, txs) = chroma(self, &c, best.0);
                        let better = cost < best.0;
                        if better {
                            best = (cost, c);
                            let old = std::mem::replace(&mut chroma_txs, txs);
                            self.recycle(old);
                        } else {
                            self.recycle(txs);
                        }
                        if outward && !better {
                            break;
                        }
                    }
                }
            }
            if limits.palette {
                let (u, v) = (self.source(1, x, y, vw, vh), self.source(2, x, y, vw, vh));
                for (pu, pv) in palette::candidates_uv(&u, &v) {
                    let mut c = Intra { uv_mode: DC_PRED, uv_angle: 0, cfl: [0; 2], ..intra };
                    c.pal_sz[1] = pu.len() as u8;
                    c.pal[1][..pu.len()].copy_from_slice(&pu);
                    c.pal[2][..pv.len()].copy_from_slice(&pv);
                    let (cost, txs) = chroma(self, &c, best.0);
                    if cost < best.0 {
                        best = (cost, c);
                        let old = std::mem::replace(&mut chroma_txs, txs);
                        self.recycle(old);
                    } else {
                        self.recycle(txs);
                    }
                }
            }
            if limits.cfl {
                let alphas = self.cfl_alphas(ti, bs, x, y, &intra, depth, &luma_done);
                if alphas != [0, 0] {
                    let c = Intra { uv_mode: CFL_PRED, uv_angle: 0, cfl: alphas, ..intra };
                    let (cost, txs) = chroma(self, &c, best.0);
                    if cost < best.0 {
                        best = (cost, c);
                        let old = std::mem::replace(&mut chroma_txs, txs);
                        self.recycle(old);
                    } else {
                        self.recycle(txs);
                    }
                }
            }
            intra = best.1;
        }

        self.lap(&mut clock, 2);
        self.restore(&start);
        self.trial = Some((intra, depth));
        self.stage = 4;
        luma_txs.append(&mut chroma_txs);
        self.recycle(chroma_txs);
        luma_txs.reverse();
        self.replay = Some(luma_txs);
        let txs = self.code_b(ti, bs, x, y, edge_flags);
        self.lap(&mut clock, 4);
        self.stage = 5;
        self.chosen.push(txs);
        self.plan.push(Node::Block(intra, depth, self.chosen.len() - 1));
    }

    /// CfL alphas that best predict each chroma plane from the reconstructed luma.
    #[allow(clippy::too_many_arguments)]
    fn cfl_alphas(&mut self, ti: usize, bs: BlockSize, x: i32, y: i32, intra: &Intra, depth: u8, luma_done: &Snap) -> [i8; 2] {
        let b_dim = bs.dimensions();
        let (w4, h4) = (b_dim[0] as usize, b_dim[1] as usize);
        let (bw4, bh4) = (b_dim[0] as i32, b_dim[1] as i32);
        let cw4 = cmp::min(bw4, self.bw - x);
        let ch4 = cmp::min(bh4, self.bh - y);
        let (tx, uvtx) = self.tx_sizes(bs, depth);
        let mut best = [(u64::MAX, 0i8); 2];
        if self.uv_outward {
            self.restore_contexts(luma_done);
            let ac = self.cfl_ac(x, y, cw4, ch4, bw4, bh4, tx);
            for p in 0..2 {
                let err = |e: &mut Self, a: i8| {
                    e.predict_cfl_plane(ti, x, y, uvtx, &ac, p, a);
                    e.sse(x, y, w4, h4, 1 + p..2 + p)
                };
                best[p] = (err(self, 0), 0);
                for dir in [1i8, -1] {
                    for i in 1..=16 {
                        let a = dir * i;
                        let d = err(self, a);
                        if d >= best[p].0 {
                            break;
                        }
                        best[p] = (d, a);
                    }
                }
            }
            self.restore(luma_done);
            return [best[0].1, best[1].1];
        }
        for a in -16i8..=16 {
            if a == 0 {
                continue;
            }
            self.restore_contexts(luma_done);
            let c = Intra { uv_mode: CFL_PRED, cfl: [a, a], ..*intra };
            self.predict_cfl(ti, x, y, cw4, ch4, bw4, bh4, tx, uvtx, &c);
            for p in 0..2 {
                let d = self.sse(x, y, w4, h4, 1 + p..2 + p);
                if d < best[p].0 {
                    best[p] = (d, a);
                }
            }
        }
        self.restore(luma_done);
        [best[0].1, best[1].1]
    }
}

/// The bits of a block's coefficients in the order the decoder reads them; the contexts are
/// left as they were.
fn coef_bits<D: Decide>(e: &mut Encoder<D>, txs: &[TxDone], bs: BlockSize, intra: &Intra) -> u32 {
    if let Some(bits) = txs.iter().map(|t| t.bits).sum::<Option<u32>>() {
        return bits;
    }
    let Some(first) = txs.first() else { return 0 };
    let b_dim = bs.dimensions();
    let (ax, ly) = (first.x4 as usize, (first.y4 & 15) as usize);
    let saved_a = e.above.save_coef(ax..ax + b_dim[0] as usize);
    let saved_l = e.left.save_coef(ly..ly + b_dim[1] as usize);
    let mut c = Counter::default();
    let lossless = e.frame.quant.lossless();
    for t in txs {
        let (a, l) = coef_ctx(&mut e.above, &mut e.left, t.plane);
        write_coefs(
            &mut c,
            &mut e.cost_cdf,
            Some(&e.coef_costs),
            a,
            l,
            t.plane,
            t.x4 as usize,
            t.y4 as usize,
            t.tx,
            bs,
            t.coefs.txtp,
            &t.coefs.levels,
            t.eob,
            intra,
            lossless,
            e.frame.quant.base,
            e.frame.reduced_tx_set,
            e.bw,
            e.bh,
        );
    }
    e.above.restore_coef(ax, &saved_a);
    e.left.restore_coef(ly, &saved_l);
    c.bits
}

fn smooth(m: u8) -> i32 {
    if matches!(m, SMOOTH_PRED | SMOOTH_V_PRED | SMOOTH_H_PRED) {
        512
    } else {
        0
    }
}

/// The coded delta q, in steps of `res`, that brings the last qidx nearest `want`, as libaom's
/// `av1_adjust_q_from_delta_q_res` rounds it.
fn delta_steps(res: i32, last: u8, want: u8) -> i32 {
    let want = (want as i32).clamp(res, 256 - res);
    let diff = want - last as i32;
    let steps = (diff.abs() + res / 4) / res;
    if diff >= 0 {
        steps
    } else {
        -steps
    }
}

/// libaom's `predict_skip_txfm` at level 1 for an 8-bit luma residual of `pw`×`ph`, `vw`×`vh`
/// of it visible: a mean square error under (dc_q / 8)² / 8 and DCT coefficients, in tiles of up
/// to 16×16, all under a block-size multiple of the quantizer.
fn predicts_skip(residual: &[i16], pw: usize, ph: usize, vw: usize, vh: usize, dq: [u32; 2]) -> bool {
    let mut se = 0u64;
    for r in 0..vh {
        // A row's squares of differences under 256 fit u32, which vectorizes.
        se += residual[r * pw..][..vw].iter().map(|&d| (d as i32 * d as i32) as u32).sum::<u32>() as u64;
    }
    let n = (dq[0] >> 3) as u64;
    if se / (pw * ph) as u64 > n * n / 8 {
        return false;
    }
    use TxfmSize::*;
    let (thresh, tx) = match (pw, ph) {
        (4, 4) => (64, S4x4),
        (4, 8) => (64, R4x8),
        (8, 4) => (64, R8x4),
        (8, 8) => (70, S8x8),
        (8, 16) => (60, R8x16),
        (16, 8) => (60, R16x8),
        (4, 16) => (64, R4x16),
        (16, 4) => (64, R16x4),
        (8, 32) | (32, 8) => (70, S8x8),
        _ => (68, S16x16),
    };
    let t = &DAV1D_TXFM_DIMENSIONS[tx as usize];
    let (tw, th) = (t.w as usize * 4, t.h as usize * 4);
    let mut tile = [0i16; 256];
    let mut coefs = [0i32; 256];
    for y in (0..ph).step_by(th) {
        for x in (0..pw).step_by(tw) {
            for r in 0..th {
                tile[r * tw..][..tw].copy_from_slice(&residual[(y + r) * pw + x..][..tw]);
            }
            fdct_fast::forward(&tile[..tw * th], tw, &mut coefs[..tw * th], tx, DCT_DCT);
            if coefs[0].unsigned_abs() << 7 >= thresh * dq[0] || coefs[1..tw * th].iter().any(|c| c.unsigned_abs() << 7 >= thresh * dq[1]) {
                return false;
            }
        }
    }
    true
}

/// A `w`×`h` rectangle copied between strided buffers, with the width known to the compiler for
/// each block width so that a row is a few vector moves, not a `memmove` call.
fn copy_rect(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize) {
    #[inline(always)]
    fn rows<const W: usize>(dst: &mut [u8], ds: usize, src: &[u8], ss: usize, h: usize) {
        for r in 0..h {
            dst[r * ds..][..W].copy_from_slice(&src[r * ss..][..W]);
        }
    }
    match w {
        4 => rows::<4>(dst, dst_stride, src, src_stride, h),
        8 => rows::<8>(dst, dst_stride, src, src_stride, h),
        16 => rows::<16>(dst, dst_stride, src, src_stride, h),
        32 => rows::<32>(dst, dst_stride, src, src_stride, h),
        64 => rows::<64>(dst, dst_stride, src, src_stride, h),
        _ => {
            for r in 0..h {
                dst[r * dst_stride..][..w].copy_from_slice(&src[r * src_stride..][..w]);
            }
        }
    }
}

/// The context a transform block leaves for its neighbours: the sum of its levels, capped, and
/// the sign of its DC.
fn res_ctx(levels: &[i32], eob: i32) -> u8 {
    if eob < 0 {
        return 0x40;
    }
    // At most 1024 levels under 2^20: the sum cannot wrap, and a plain sum vectorizes.
    let cul = levels.iter().fold(0u32, |a, l| a.wrapping_add(l.unsigned_abs() & 0xfffff));
    let sign = match levels[0] {
        0 => 0x40,
        d if d < 0 => 0,
        _ => 0x80,
    };
    (cmp::min(cul, 63) | sign) as u8
}

/// Everything a block writes, in the order `decode_b` reads it.
struct BlockWrite {
    bs: BlockSize,
    x: i32,
    y: i32,
    skip: bool,
    intra: Intra,
    depth: u8,
    max_tx: TxfmSize,
    limits: Limits,
    delta: Option<i32>,
    tx_select: bool,
    lossless: bool,
    qidx: u8,
    reduced_tx_set: bool,
    bw: i32,
    bh: i32,
}

impl BlockWrite {
    #[allow(clippy::too_many_arguments)]
    fn write<S: Sink>(&self, s: &mut S, cdf: &mut CdfContext, above: &mut Ctx, left: &mut Ctx, txs: &[TxDone], maps: &[Vec<u8>; 2]) {
        let (ax, ly) = (self.x as usize, (self.y & 15) as usize);
        let b_dim = self.bs.dimensions();
        let (bw4, bh4) = (b_dim[0] as usize, b_dim[1] as usize);
        let sctx = (above.skip[ax] + left.skip[ly]) as usize;
        s.bool_adapt(self.skip, &mut cdf.m.skip[sctx]);
        if let Some(delta) = self.delta {
            let abs = delta.unsigned_abs();
            s.symbol(cmp::min(abs, 3) as u8, &mut cdf.m.delta_q.0, 3);
            if abs >= 3 {
                let n_bits = 31 - (abs - 1).leading_zeros();
                s.bools(n_bits - 1, 3);
                s.bools(abs - 1 - (1 << n_bits), n_bits as u8);
            }
            if abs != 0 {
                s.bool_equi(delta < 0);
            }
        }
        write_luma_modes(s, cdf, above, left, ax, ly, &self.intra, &self.limits);
        if self.limits.chroma {
            write_chroma_modes(s, cdf, &self.intra, &self.limits);
        }
        let y_mode = if self.intra.y_mode == FILTER_PRED { DC_PRED } else { self.intra.y_mode };
        if self.limits.palette {
            write_pal_y(s, cdf, above, left, ax, ly, self.bs, &self.intra);
            if self.limits.chroma {
                write_pal_uv(s, cdf, above, left, ax, ly, self.bs, &self.intra);
            }
        }
        write_filter(s, cdf, self.bs, &self.intra, &self.limits);
        let (w, h) = ((cmp::min(bw4 as i32, self.bw - self.x) * 4) as usize, (cmp::min(bh4 as i32, self.bh - self.y) * 4) as usize);
        for pl in 0..2 {
            if self.intra.pal_sz[pl] > 0 {
                let n = self.intra.pal_sz[pl];
                palette::write_indices(s, &mut cdf.m.color_map[pl][n as usize - 2], &maps[pl], bw4 * 4, w, h, n);
            }
        }
        write_depth(s, cdf, above, left, ax, ly, self.depth, self.max_tx, self.tx_select && !self.lossless);
        if !self.skip {
            for t in txs {
                let (a, l) = coef_ctx(above, left, t.plane);
                write_coefs(
                    s,
                    cdf,
                    None,
                    a,
                    l,
                    t.plane,
                    t.x4 as usize,
                    t.y4 as usize,
                    t.tx,
                    self.bs,
                    t.coefs.txtp,
                    &t.coefs.levels,
                    t.eob,
                    &self.intra,
                    self.lossless,
                    self.qidx,
                    self.reduced_tx_set,
                    self.bw,
                    self.bh,
                );
            }
        } else {
            for t in txs {
                let t_dim = &DAV1D_TXFM_DIMENSIONS[t.tx as usize];
                let (a, l) = coef_ctx(above, left, t.plane);
                a[t.x4 as usize..][..t_dim.w as usize].fill(0x40);
                l[(t.y4 & 15) as usize..][..t_dim.h as usize].fill(0x40);
            }
        }
        let mut tx = self.max_tx;
        for _ in 0..self.depth {
            tx = DAV1D_TXFM_DIMENSIONS[tx as usize].sub;
        }
        let t_dim = &DAV1D_TXFM_DIMENSIONS[tx as usize];
        above.tx_intra[ax..ax + bw4].fill(t_dim.lw as i8);
        left.tx_intra[ly..ly + bh4].fill(t_dim.lh as i8);
        above.mode[ax..ax + bw4].fill(y_mode);
        left.mode[ly..ly + bh4].fill(y_mode);
        above.pal_sz[ax..ax + bw4].fill(self.intra.pal_sz[0]);
        left.pal_sz[ly..ly + bh4].fill(self.intra.pal_sz[0]);
        if self.intra.pal_sz[0] > 0 {
            above.pal[ax..ax + bw4].iter_mut().for_each(|p| p[0] = self.intra.pal[0]);
            left.pal[ly..ly + bh4].iter_mut().for_each(|p| p[0] = self.intra.pal[0]);
        }
        above.skip[ax..ax + bw4].fill(self.skip as u8);
        left.skip[ly..ly + bh4].fill(self.skip as u8);
        if self.limits.chroma {
            above.uvmode[ax..ax + bw4].fill(self.intra.uv_mode);
            left.uvmode[ly..ly + bh4].fill(self.intra.uv_mode);
            above.pal_sz_uv[ax..ax + bw4].fill(self.intra.pal_sz[1]);
            left.pal_sz_uv[ly..ly + bh4].fill(self.intra.pal_sz[1]);
            if self.intra.pal_sz[1] > 0 {
                let [_, u, v] = self.intra.pal;
                above.pal[ax..ax + bw4].iter_mut().for_each(|p| (p[1], p[2]) = (u, v));
                left.pal[ly..ly + bh4].iter_mut().for_each(|p| (p[1], p[2]) = (u, v));
            }
        }
    }
}

/// The luma mode and its angle delta.
#[allow(clippy::too_many_arguments)]
fn write_luma_modes<S: Sink>(s: &mut S, cdf: &mut CdfContext, above: &Ctx, left: &Ctx, ax: usize, ly: usize, intra: &Intra, limits: &Limits) {
    let y_mode = if intra.y_mode == FILTER_PRED { DC_PRED } else { intra.y_mode };
    let ymode_cdf = &mut cdf.kfym[DAV1D_INTRA_MODE_CONTEXT[above.mode[ax] as usize] as usize]
        [DAV1D_INTRA_MODE_CONTEXT[left.mode[ly] as usize] as usize];
    s.symbol(y_mode, ymode_cdf, N_INTRA_PRED_MODES as u8 - 1);
    if limits.angles && (VERT_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
        let acdf = &mut cdf.m.angle_delta[(y_mode - VERT_PRED) as usize];
        s.symbol((intra.y_angle + 3) as u8, acdf, 6);
    }
}

/// The luma palette flag and, with a palette, its size and colours.
#[allow(clippy::too_many_arguments)]
fn write_pal_y<S: Sink>(s: &mut S, cdf: &mut CdfContext, above: &Ctx, left: &Ctx, ax: usize, ly: usize, bs: BlockSize, intra: &Intra) {
    if !matches!(intra.y_mode, DC_PRED | FILTER_PRED) {
        return;
    }
    let b_dim = bs.dimensions();
    let sz_ctx = (b_dim[2] + b_dim[3] - 2) as usize;
    let pal_ctx = (above.pal_sz[ax] > 0) as usize + (left.pal_sz[ly] > 0) as usize;
    let n = intra.pal_sz[0] as usize;
    s.bool_adapt(n > 0, &mut cdf.m.pal_y[sz_ctx][pal_ctx]);
    if n > 0 {
        s.symbol(n as u8 - 2, &mut cdf.m.pal_sz[0][sz_ctx], 6);
        let a_n = if ly != 0 { above.pal_sz[ax] as usize } else { 0 };
        let (cache, c_n) = palette::cache(&above.pal[ax][0][..a_n], &left.pal[ly][0][..left.pal_sz[ly] as usize]);
        palette::write_colors(s, &intra.pal[0][..n], &cache[..c_n], false);
    }
}

/// The chroma palette flag and, with a palette, its size and U and V colours.
#[allow(clippy::too_many_arguments)]
fn write_pal_uv<S: Sink>(s: &mut S, cdf: &mut CdfContext, above: &Ctx, left: &Ctx, ax: usize, ly: usize, bs: BlockSize, intra: &Intra) {
    if intra.uv_mode != DC_PRED {
        return;
    }
    let b_dim = bs.dimensions();
    let sz_ctx = (b_dim[2] + b_dim[3] - 2) as usize;
    let n = intra.pal_sz[1] as usize;
    s.bool_adapt(n > 0, &mut cdf.m.pal_uv[(intra.pal_sz[0] > 0) as usize]);
    if n > 0 {
        s.symbol(n as u8 - 2, &mut cdf.m.pal_sz[1][sz_ctx], 6);
        let a_n = if ly != 0 { above.pal_sz_uv[ax] as usize } else { 0 };
        let (cache, c_n) = palette::cache(&above.pal[ax][1][..a_n], &left.pal[ly][1][..left.pal_sz_uv[ly] as usize]);
        palette::write_colors(s, &intra.pal[1][..n], &cache[..c_n], true);
        palette::write_v(s, &intra.pal[2][..n]);
    }
}

fn write_filter<S: Sink>(s: &mut S, cdf: &mut CdfContext, bs: BlockSize, intra: &Intra, limits: &Limits) {
    let y_mode = if intra.y_mode == FILTER_PRED { DC_PRED } else { intra.y_mode };
    if y_mode == DC_PRED && intra.pal_sz[0] == 0 && limits.filter {
        let filter = (intra.y_mode == FILTER_PRED).then_some(intra.y_angle as u8);
        s.bool_adapt(filter.is_some(), &mut cdf.m.use_filter_intra[bs as usize]);
        if let Some(f) = filter {
            s.symbol(f, &mut cdf.m.filter_intra.0, 4);
        }
    }
}

/// The transform depth, when the frame selects it.
#[allow(clippy::too_many_arguments)]
fn write_depth<S: Sink>(s: &mut S, cdf: &mut CdfContext, above: &Ctx, left: &Ctx, ax: usize, ly: usize, depth: u8, max_tx: TxfmSize, tx_select: bool) {
    let max_dim = &DAV1D_TXFM_DIMENSIONS[max_tx as usize];
    if tx_select && max_dim.max > 0 {
        let tctx = (left.tx_intra[ly] as i32 >= max_dim.lh as i32) as usize + (above.tx_intra[ax] as i32 >= max_dim.lw as i32) as usize;
        let c = &mut cdf.m.txsz[(max_dim.max - 1) as usize][tctx];
        s.symbol(depth, c, cmp::min(max_dim.max, 2));
    }
}

fn write_chroma_modes<S: Sink>(s: &mut S, cdf: &mut CdfContext, intra: &Intra, limits: &Limits) {
    let y_mode = if intra.y_mode == FILTER_PRED { DC_PRED } else { intra.y_mode };
    let c = &mut cdf.m.uv_mode[limits.cfl as usize][y_mode as usize];
    s.symbol(intra.uv_mode, c, N_UV_INTRA_PRED_MODES as u8 - 1 - !limits.cfl as u8);
    if intra.uv_mode == CFL_PRED {
        let sign = intra.cfl.map(|a| if a == 0 { 0u8 } else if a < 0 { 1 } else { 2 });
        s.symbol(sign[0] * 3 + sign[1] - 1, &mut cdf.m.cfl_sign.0, 7);
        for i in 0..2 {
            if sign[i] != 0 {
                let ctx = (sign[i] == 2) as usize * 3 + sign[1 - i] as usize;
                s.symbol(intra.cfl[i].unsigned_abs() - 1, &mut cdf.m.cfl_alpha[ctx], 15);
            }
        }
    } else if limits.angles && (VERT_PRED..=VERT_LEFT_PRED).contains(&intra.uv_mode) {
        let acdf = &mut cdf.m.angle_delta[(intra.uv_mode - VERT_PRED) as usize];
        s.symbol((intra.uv_angle + 3) as u8, acdf, 6);
    }
}

#[repr(C, align(64))]
struct Ac([i16; SCRATCH_AC_TXTP_LEN]);

const EDGE_LEN: usize = 257;

/// libaom's `av1_derived_filter_intra_mode_used_flag`: FILTER_DC and the filter mode of the best
/// mode so far.
const FILTER_FROM_BEST: [u8; N_INTRA_PRED_MODES] = [0x01, 0x03, 0x05, 0x01, 0x01, 0x01, 0x09, 0x01, 0x01, 0x01, 0x01, 0x01, 0x11];

/// libaom's `av1_derived_chroma_intra_mode_used_flag` without CfL: DC, SMOOTH and the luma
/// winner's own mode.
const UV_FROM_LUMA: [u16; N_INTRA_PRED_MODES] =
    [0x201, 0x203, 0x205, 0x209, 0x211, 0x221, 0x241, 0x281, 0x301, 0x201, 0x601, 0xa01, 0x1201];

/// libaom's `av1_reduced_intra_tx_used_flag`: transform types by luma mode, bit per type.
const REDUCED_INTRA_TXTP: [u16; N_INTRA_PRED_MODES] =
    [0x80f, 0x40f, 0x80f, 0x20f, 0x80f, 0x40f, 0x80f, 0x80f, 0x40f, 0x80f, 0x40f, 0x80f, 0xc0e];

#[repr(C, align(64))]
struct EdgePx([u8; EDGE_LEN]);

#[repr(C, align(64))]
struct Coef([i16; 32 * 32]);

/// The working arrays of the residual search, allocated once.
struct Bufs {
    src: [u8; 64 * 64],
    pred: [u8; 64 * 64],
    residual: [i16; 64 * 64],
    coefs: [i32; 32 * 32],
    trial: [i32; 32 * 32],
    cf: Coef,
}

fn coef_ctx<'a>(above: &'a mut Ctx, left: &'a mut Ctx, plane: usize) -> (&'a mut [u8], &'a mut [u8]) {
    if plane == 0 {
        (&mut above.lcoef, &mut left.lcoef)
    } else {
        (&mut above.ccoef[plane - 1], &mut left.ccoef[plane - 1])
    }
}

/// Scan index by coefficient index, the inverse of `scan_rc`, per transform size and class.
fn scan_index(tx: TxfmSize, class: TxClass) -> &'static [u16] {
    static TABLES: std::sync::OnceLock<Vec<Vec<u16>>> = std::sync::OnceLock::new();
    let tables = TABLES.get_or_init(|| {
        (0..TxfmSize::COUNT * 3)
            .map(|k| {
                let tx = TxfmSize::from_repr(k / 3).unwrap();
                let class = [TxClass::TwoD, TxClass::H, TxClass::V][k % 3];
                let t_dim = &DAV1D_TXFM_DIMENSIONS[tx as usize];
                let n = cmp::min(t_dim.w as usize * 4, 32) * cmp::min(t_dim.h as usize * 4, 32);
                let mut inv = vec![0u16; n];
                for i in 0..n {
                    inv[scan_rc(tx, class, i).0] = i as u16;
                }
                inv
            })
            .collect()
    });
    let c = match class {
        TxClass::TwoD => 0,
        TxClass::H => 1,
        TxClass::V => 2,
    };
    &tables[tx as usize * 3 + c]
}

fn pal_key(chroma: bool, x: i32, y: i32, bw4: i32, bh4: i32, intra: &Intra) -> (bool, u8, i32, i32, i32, i32, [[u8; 8]; 2]) {
    let colours = if chroma { [intra.pal[1], intra.pal[2]] } else { [intra.pal[0], [0; 8]] };
    (chroma, intra.pal_sz[chroma as usize], x, y, bw4, bh4, colours)
}

fn eob_of(tx: TxfmSize, txtp: TxfmType, levels: &[i32]) -> i32 {
    let index = scan_index(tx, DAV1D_TX_TYPE_CLASS[txtp as usize]);
    levels.iter().zip(index).filter(|(&l, _)| l != 0).map(|(_, &i)| i as i32).max().unwrap_or(-1)
}

fn base_tok<S: Sink>(w: &mut S, costs: Option<&CoefCosts>, cdf: &mut CdfContext, t: usize, chroma: usize, ctx: usize, s: u8) {
    match costs {
        Some(c) => w.add(c.base[t][chroma][ctx][s as usize] as u32),
        None => w.symbol(s, &mut cdf.coef.base_tok[t][chroma][ctx], 3),
    }
}

fn br_tok<S: Sink>(w: &mut S, costs: Option<&CoefCosts>, cdf: &mut CdfContext, hi: usize, chroma: usize, ctx: usize, tok: u8) {
    match costs {
        Some(c) => w.add(c.br[hi][chroma][ctx][tok as usize]),
        None => w.hi_tok(tok, &mut cdf.coef.br_tok[hi][chroma][ctx]),
    }
}

/// Codes a transform block's coefficients; with `costs` a counter takes the per-coefficient
/// symbols from them instead of the CDFs.
#[allow(clippy::too_many_arguments)]
fn write_coefs<S: Sink>(
    w: &mut S,
    cdf: &mut CdfContext,
    costs: Option<&CoefCosts>,
    a: &mut [u8],
    l: &mut [u8],
    plane: usize,
    x4: usize,
    y4: usize,
    tx: TxfmSize,
    bs: BlockSize,
    txtp: TxfmType,
    coefs: &[i32],
    eob: i32,
    intra: &Intra,
    lossless: bool,
    qidx: u8,
    reduced_tx_set: bool,
    bw: i32,
    bh: i32,
) {
    let t_dim: &TxfmInfo = &DAV1D_TXFM_DIMENSIONS[tx as usize];
    let chroma = (plane != 0) as usize;
    let ly = y4 & 15;
    let sctx = get_skip_ctx(
        t_dim,
        bs,
        &a[x4..x4 + t_dim.w as usize],
        &l[ly..ly + t_dim.h as usize],
        plane != 0,
        Rav1dPixelLayout::I444,
    );
    let all_skip = eob < 0;
    w.bool_adapt(all_skip, &mut cdf.coef.skip[t_dim.ctx as usize][sctx.get() as usize]);
    let set_ctx = |a: &mut [u8], l: &mut [u8], v: u8| {
        let n = cmp::min(t_dim.w as i64, bw as i64 - x4 as i64) as usize;
        let m = cmp::min(t_dim.h as i64, bh as i64 - y4 as i64) as usize;
        a[x4..x4 + n].fill(v);
        l[ly..ly + m].fill(v);
    };
    if all_skip {
        set_ctx(a, l, 0x40);
        return;
    }
    if !lossless && qidx != 0 && plane == 0 && t_dim.max < TxfmSize::S32x32 as u8 {
        let y_mode = if intra.y_mode == FILTER_PRED {
            DAV1D_FILTER_MODE_TO_Y_MODE[intra.y_angle as usize]
        } else {
            intra.y_mode
        } as usize;
        if reduced_tx_set || t_dim.min == TxfmSize::S16x16 as u8 {
            let idx = DAV1D_TX_TYPES_PER_SET[..5].iter().position(|&t| t == txtp).unwrap();
            w.symbol(idx as u8, &mut cdf.m.txtp_intra2[t_dim.min as usize][y_mode], 4);
        } else {
            let idx = DAV1D_TX_TYPES_PER_SET[5..12].iter().position(|&t| t == txtp).unwrap();
            w.symbol(idx as u8, &mut cdf.m.txtp_intra1[t_dim.min as usize][y_mode], 6);
        }
    }
    let slw = cmp::min(t_dim.lw, TxfmSize::S32x32 as u8);
    let slh = cmp::min(t_dim.lh, TxfmSize::S32x32 as u8);
    let tx2dszctx = slw + slh;
    let class = DAV1D_TX_TYPE_CLASS[txtp as usize];
    let is_1d = (class != TxClass::TwoD) as usize;
    let eob = eob as u32;
    let bin = if eob == 0 { 0 } else { 32 - eob.leading_zeros() } as u8;
    let n = 4 + tx2dszctx;
    match tx2dszctx {
        0 => w.symbol(bin, &mut cdf.coef.eob_bin_16[chroma][is_1d], n),
        1 => w.symbol(bin, &mut cdf.coef.eob_bin_32[chroma][is_1d], n),
        2 => w.symbol(bin, &mut cdf.coef.eob_bin_64[chroma][is_1d], n),
        3 => w.symbol(bin, &mut cdf.coef.eob_bin_128[chroma][is_1d], n),
        4 => w.symbol(bin, &mut cdf.coef.eob_bin_256[chroma][is_1d], n),
        5 => w.symbol(bin, &mut cdf.coef.eob_bin_512[chroma], n),
        _ => w.symbol(bin, &mut cdf.coef.eob_bin_1024[chroma], n),
    }
    if bin > 1 {
        let hi = eob >> (bin - 2) & 1 != 0;
        w.bool_adapt(hi, &mut cdf.coef.eob_hi_bit[t_dim.ctx as usize][chroma][bin as usize]);
        w.bools(eob & ((1 << (bin - 2)) - 1), bin - 2);
    }

    let lay = contexts::layout(tx, class);
    let scan = &lay.scan[..];
    let level_at = |i: usize| coefs[scan[i].0 as usize].unsigned_abs();
    let hi_ctx = cmp::min(t_dim.ctx, 3) as usize;
    let mut scratch = contexts::Scratch::new();
    let (base_ctx, br_mag) = contexts::contexts(lay, coefs, eob as usize, &mut scratch);
    // Counted, the AC signs and Golomb tails cost the same wherever they fall, so they are
    // added in this pass; the AC levels' sum is kept for the contexts.
    let (mut tail_sum, mut tail_bits) = (0u32, 0u32);
    let mut tail = |lv: u32| {
        tail_sum = tail_sum.wrapping_add(lv);
        if lv != 0 {
            tail_bits += 512;
            if lv >= 15 {
                tail_bits += 512 * (2 * (31 - (lv - 14).leading_zeros()) + 1);
            }
        }
    };
    let raster = costs.filter(|_| lay.by_column());
    let mut ac = (0, 0);
    let dc_level;
    if eob > 0 {
        let (_, x, y, _) = scan[eob as usize];
        let lv = level_at(eob as usize);
        if raster.is_none() {
            tail(lv);
        }
        let ctx = 1 + (eob > (2 << tx2dszctx)) as usize + (eob > (4 << tx2dszctx)) as usize;
        w.symbol((cmp::min(lv, 3) - 1) as u8, &mut cdf.coef.eob_base_tok[t_dim.ctx as usize][chroma][ctx], 2);
        if lv >= 3 {
            let far = if class == TxClass::TwoD { (x | y) > 1 } else { y != 0 };
            let ctx = if far { 14 } else { 7 };
            br_tok(w, costs, cdf, hi_ctx, chroma, ctx, cmp::min(lv, 15) as u8);
        }
        let mut serial = eob as usize;
        if let Some(c) = raster {
            let t = t_dim.ctx as usize;
            ac = contexts::ac_bits(lay, coefs, eob as usize, base_ctx, br_mag, &c.base[t][chroma], &c.br[hi_ctx][chroma]);
            serial = 1;
        }
        for i in (1..serial).rev() {
            let (rc, x, mut y, off) = scan[i];
            let off = off as usize;
            let lv = coefs[rc as usize].unsigned_abs();
            tail(lv);
            base_tok(w, costs, cdf, t_dim.ctx as usize, chroma, base_ctx[off] as usize, cmp::min(lv, 3) as u8);
            if class == TxClass::TwoD {
                y |= x;
            }
            if lv >= 3 {
                let ctx = if y > (class == TxClass::TwoD) as u8 { 14 } else { 7 } + br_mag[off] as usize;
                br_tok(w, costs, cdf, hi_ctx, chroma, ctx, cmp::min(lv, 15) as u8);
            }
        }
        let lo = if class == TxClass::TwoD { 0 } else { base_ctx[0] as usize };
        let lv = level_at(0);
        base_tok(w, costs, cdf, t_dim.ctx as usize, chroma, lo, cmp::min(lv, 3) as u8);
        if lv >= 3 {
            br_tok(w, costs, cdf, hi_ctx, chroma, br_mag[0] as usize, cmp::min(lv, 15) as u8);
        }
        dc_level = lv;
    } else {
        let lv = level_at(0);
        w.symbol((cmp::min(lv, 3) - 1) as u8, &mut cdf.coef.eob_base_tok[t_dim.ctx as usize][chroma][0], 2);
        if lv >= 3 {
            br_tok(w, costs, cdf, hi_ctx, chroma, 0, cmp::min(lv, 15) as u8);
        }
        dc_level = lv;
    }
    let mut cul_level = 0u32;
    let dc_sign_level;
    let dc = coefs[0];
    if dc_level == 0 {
        dc_sign_level = 0x40;
    } else {
        let ctx = get_dc_sign_ctx(tx, &a[x4..], &l[ly..]) as usize;
        w.bool_adapt(dc < 0, &mut cdf.coef.dc_sign[chroma][ctx]);
        if dc_level >= 15 {
            w.golomb(dc_level - 15);
        }
        dc_sign_level = if dc < 0 { 0 } else { 0x80 };
        cul_level = dc_level;
    }
    if costs.is_some() {
        w.add(tail_bits.wrapping_add(ac.0));
        cul_level = cul_level.wrapping_add(tail_sum).wrapping_add(ac.1);
    } else {
        for &(rc, ..) in &scan[1..=eob as usize] {
            let v = coefs[rc as usize];
            if v == 0 {
                continue;
            }
            w.bool_equi(v < 0);
            let lv = v.unsigned_abs();
            if lv >= 15 {
                w.golomb(lv - 15);
            }
            cul_level = cul_level.wrapping_add(lv);
        }
    }
    set_ctx(a, l, (cmp::min(cul_level, 63) | dc_sign_level) as u8);
}
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::enc::header::{Colour, Quant, Tiling};
    use crate::{Decoder, PlanarImageComponent, Settings};

    pub struct Rng(pub u64);

    impl Rng {
        pub fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 32) as u32
        }

        pub fn below(&mut self, n: u32) -> u32 {
            self.next() % n
        }
    }

    /// Every decision at random, within what the syntax allows.
    struct Random {
        rng: Rng,
        density: u32,
    }

    impl Decide for Random {
        fn partition(&mut self, bl: BlockLevel, _: i32, _: i32, halves: Halves) -> BlockPartition {
            let split = self.rng.below(3) == 0 && bl != BlockLevel::Bl8x8 || self.rng.below(5) == 0;
            match halves {
                Halves::Both => {
                    let n = DAV1D_PARTITION_TYPE_COUNT[bl as usize] as u32 + 1;
                    BlockPartition::from_repr(self.rng.below(n) as usize).unwrap()
                }
                Halves::Top if split => BlockPartition::Split,
                Halves::Top => BlockPartition::H,
                Halves::Left if split => BlockPartition::Split,
                Halves::Left => BlockPartition::V,
            }
        }

        fn qidx(&mut self, _: i32, _: i32, last: u8) -> u8 {
            (last as i32 + self.rng.below(121) as i32 - 60).clamp(0, 255) as u8
        }

        fn mode(&mut self, _: BlockSize, _: i32, _: i32, l: &Limits) -> (Intra, u8) {
            let r = &mut self.rng;
            let mut m = Intra { y_mode: r.below(13) as u8, ..Default::default() };
            if l.angles && (VERT_PRED..=VERT_LEFT_PRED).contains(&m.y_mode) {
                m.y_angle = r.below(7) as i8 - 3;
            }
            if m.y_mode == DC_PRED && l.filter && r.below(2) == 0 {
                m.y_mode = FILTER_PRED;
                m.y_angle = r.below(5) as i8;
            }
            if l.palette && m.y_mode == DC_PRED && r.below(2) == 0 {
                let n = 2 + r.below(7) as usize;
                let mut pal = random_colors(r, n);
                pal.dedup();
                while pal.len() < 2 {
                    pal = random_colors(r, n);
                    pal.dedup();
                }
                m.pal_sz[0] = pal.len() as u8;
                m.pal[0][..pal.len()].copy_from_slice(&pal);
            }
            if l.chroma {
                m.uv_mode = r.below(13 + l.cfl as u32) as u8;
                if m.uv_mode == CFL_PRED {
                    loop {
                        m.cfl = [r.below(33) as i8 - 16, r.below(33) as i8 - 16];
                        if m.cfl != [0, 0] {
                            break;
                        }
                    }
                } else if l.angles && (VERT_PRED..=VERT_LEFT_PRED).contains(&m.uv_mode) {
                    m.uv_angle = r.below(7) as i8 - 3;
                }
                if l.palette && m.uv_mode == DC_PRED && r.below(2) == 0 {
                    let n = 2 + r.below(7) as usize;
                    m.pal_sz[1] = n as u8;
                    m.pal[1][..n].copy_from_slice(&random_colors(r, n));
                    for v in &mut m.pal[2][..n] {
                        *v = [0, 255, 7, 128, 129][r.below(5) as usize] ^ (r.below(4) as u8 * 50);
                    }
                }
            }
            (m, r.below(l.max_depth as u32 + 1) as u8)
        }

        fn residual(&mut self, _: usize, _: TxfmSize, set: &[TxfmType], levels: &mut [i32]) -> Option<TxfmType> {
            let txtp = if set.is_empty() { DCT_DCT } else { set[self.rng.below(set.len() as u32) as usize] };
            let density = [0, 1, 4, self.density][self.rng.below(4) as usize];
            for l in levels.iter_mut() {
                *l = if self.rng.below(100) < density {
                    let m = [2, 4, 16, 3000][self.rng.below(4) as usize];
                    let v = 1 + self.rng.below(m) as i32;
                    if self.rng.next() & 1 == 0 { v } else { -v }
                } else {
                    0
                };
            }
            Some(txtp)
        }
    }

    /// Sorted colours from a small set, so neighbouring palettes share some.
    fn random_colors(r: &mut Rng, n: usize) -> Vec<u8> {
        const SET: [u8; 12] = [0, 1, 37, 74, 127, 128, 191, 200, 223, 239, 254, 255];
        let mut c: Vec<u8> = (0..n).map(|_| SET[r.below(12) as usize] ^ (r.below(3) as u8 * 2)).collect();
        c.sort_unstable();
        c
    }

    pub fn sequence(width: u32, height: u32, mono: bool) -> Sequence {
        Sequence {
            width,
            height,
            mono,
            colour: (!mono).then_some(Colour { primaries: 1, transfer: 13, matrix: 6 }),
            full_range: true,
            filter_intra: true,
            edge_filter: true,
            cdef: false,
            restoration: false,
        }
    }

    pub fn frame(seq: &Sequence, base: u8) -> Frame {
        Frame {
            screen_content: false,
            quant: Quant { base, ..Default::default() },
            delta_q_log2: None,
            lf: [0; 4],
            sharpness: 0,
            cdef: None,
            lr: None,
            tx_select: false,
            reduced_tx_set: false,
            tiling: Tiling::new(seq.width, seq.height),
        }
    }

    pub fn image(w: usize, h: usize, planes: usize, seed: u64) -> Vec<Vec<u8>> {
        let mut rng = Rng(seed);
        (0..planes)
            .map(|p| {
                (0..w * h)
                    .map(|i| ((i % w) * 3 + (i / w) * 2 + p * 50) as u8 ^ (rng.next() % 16) as u8)
                    .collect()
            })
            .collect()
    }

    /// The planes rav1d decodes, with or without its in-loop filters.
    pub fn decode_with(data: Vec<u8>, w: usize, h: usize, planes: usize, filters: bool) -> Vec<Vec<u8>> {
        let mut s = Settings::new();
        if !filters {
            s.set_inloop_filters(crate::InloopFilterType::empty());
        }
        s.set_n_threads(1);
        s.set_max_frame_delay(1);
        s.set_strict_std_compliance(true);
        let mut d = Decoder::with_settings(&s).unwrap();
        d.send_data(data.into_boxed_slice(), None, None, None).unwrap();
        let pic = d.get_picture().or_else(|_| d.get_picture()).expect("a picture");
        assert_eq!((pic.width() as usize, pic.height() as usize), (w, h));
        let comps = [PlanarImageComponent::Y, PlanarImageComponent::U, PlanarImageComponent::V];
        comps[..planes]
            .iter()
            .map(|&c| {
                let (plane, stride) = (pic.plane(c), pic.stride(c) as usize);
                (0..h).flat_map(|y| plane[y * stride..][..w].to_vec()).collect()
            })
            .collect()
    }

    pub fn decode(data: Vec<u8>, w: usize, h: usize, planes: usize) -> Vec<Vec<u8>> {
        decode_with(data, w, h, planes, false)
    }

    pub fn round_trip<D: Decide>(img: &[Vec<u8>], w: usize, h: usize, frame: Frame, decide: D) -> Coded {
        round_trip_with(img, w, h, sequence(w as u32, h as u32, img.len() == 1), frame, decide)
    }

    /// Codes `img`, decodes it with rav1d's in-loop filters off and checks the planes equal the
    /// encoder's reconstruction.
    pub fn round_trip_with<D: Decide>(img: &[Vec<u8>], w: usize, h: usize, seq: Sequence, frame: Frame, decide: D) -> Coded {
        let planes = img.len();
        let mut e = Encoder::new(seq.clone(), frame, decide);
        e.recon = Some(vec![Vec::new(); planes]);
        for y in 0..h {
            let rows: Vec<&[u8]> = img.iter().map(|p| &p[y * w..][..w]).collect();
            e.push(&rows);
        }
        let coded = e.finish();
        assert_eq!(coded.config[..2], [0x81, seq.profile() << 5 | seq.level()]);
        let decoded = decode(coded.data.clone(), w, h, planes);
        let recon = coded.recon.as_ref().unwrap();
        for p in 0..planes {
            if let Some(i) = (0..w * h).find(|&i| decoded[p][i] != recon[p][i]) {
                panic!(
                    "{w}x{h} plane {p}: first difference at ({}, {}): decoded {} encoder {}",
                    i % w,
                    i / w,
                    decoded[p][i],
                    recon[p][i]
                );
            }
        }
        coded
    }

    fn random(w: usize, h: usize, mono: bool, seed: u64, f: impl Fn(&mut Frame)) {
        let planes = if mono { 1 } else { 3 };
        let seq = sequence(w as u32, h as u32, mono);
        let mut fr = frame(&seq, 1 + (seed * 37 % 250) as u8);
        f(&mut fr);
        let img = image(w, h, planes, seed);
        round_trip(&img, w, h, fr, Random { rng: Rng(seed * 7919 + 1), density: 30 });
    }

    #[test]
    fn coef_costs_count_as_the_cdfs() {
        let mut rng = Rng(7);
        for qidx in [20u8, 100, 200] {
            let mut cdf = rav1d_cdf_thread_copy(&rav1d_cdf_thread_init_static(qidx));
            let costs = CoefCosts::new(&cdf.coef);
            for _ in 0..3000 {
                let tx = TxfmSize::from_repr(rng.below(TxfmSize::COUNT as u32) as usize).unwrap();
                let t_dim = &DAV1D_TXFM_DIMENSIONS[tx as usize];
                let types: &[TxfmType] = if t_dim.max >= TxfmSize::S32x32 as u8 {
                    &[DCT_DCT]
                } else if t_dim.min == TxfmSize::S16x16 as u8 {
                    &[DCT_DCT, IDTX]
                } else {
                    &[DCT_DCT, IDTX, 10, 11]
                };
                let txtp = types[rng.below(types.len() as u32) as usize];
                let n = cmp::min(t_dim.w as usize * 4, 32) * cmp::min(t_dim.h as usize * 4, 32);
                let dense = rng.below(4) + 1;
                let coefs: Vec<i32> = (0..n)
                    .map(|_| match rng.below(16 / dense) {
                        0 => {
                            let v = if rng.below(8) == 0 { rng.below(60) } else { rng.below(5) } as i32;
                            if rng.below(2) == 0 { -v } else { v }
                        }
                        _ => 0,
                    })
                    .collect();
                let eob = eob_of(tx, txtp, &coefs);
                let plane = if t_dim.max > TxfmSize::S32x32 as u8 { 0 } else { rng.below(3) as usize };
                let intra = Intra { y_mode: rng.below(13) as u8, ..Default::default() };
                let count = |costs: Option<&CoefCosts>, cdf: &mut CdfContext| {
                    let (mut a, mut l) = ([0x40u8; 48], [0x40u8; 48]);
                    let mut c = Counter::default();
                    write_coefs(&mut c, cdf, costs, &mut a, &mut l, plane, 0, 0, tx, BlockSize::Bs64x64, txtp, &coefs, eob, &intra, false, qidx, false, i32::MAX, i32::MAX);
                    c.bits
                };
                assert_eq!(count(Some(&costs), &mut cdf), count(None, &mut cdf), "{tx:?} {txtp} eob {eob}");
            }
        }
    }

    #[test]
    fn prediction_only() {
        for (w, h) in [(64, 64), (1, 1), (100, 37), (130, 70), (8, 200)] {
            for (mono, bl) in [(false, BlockLevel::Bl16x16), (true, BlockLevel::Bl32x32), (false, BlockLevel::Bl8x8)] {
                let planes = if mono { 1 } else { 3 };
                let seq = sequence(w as u32, h as u32, mono);
                round_trip(&image(w, h, planes, 1), w, h, frame(&seq, 100), Skip(bl));
            }
        }
    }

    #[test]
    fn random_decisions() {
        let sizes = [(64, 64), (100, 37), (130, 70), (3, 5), (257, 129), (16, 300)];
        for (i, (w, h)) in sizes.into_iter().enumerate() {
            for seed in 0..6 {
                let s = (i * 10 + seed) as u64 + 1;
                random(w, h, seed % 3 == 2, s, |f| {
                    f.tx_select = seed % 2 == 0;
                    f.reduced_tx_set = seed % 4 == 1;
                    f.screen_content = seed % 3 != 1;
                });
            }
        }
    }

    #[test]
    fn random_delta_q() {
        for seed in 1..8 {
            random(200, 150, seed % 3 == 0, seed, |f| {
                f.delta_q_log2 = Some((seed % 4) as u8);
                f.tx_select = true;
            });
        }
    }

    #[test]
    fn random_quantizer_matrices() {
        for seed in 1..6 {
            random(96, 80, seed == 3, seed, |f| {
                f.quant.qm = Some([seed as u8 * 3 % 16, seed as u8 * 5 % 16, 0]);
                f.quant.y_dc = seed as i8 * 7 - 20;
                f.quant.u_dc = -(seed as i8) * 3;
                f.quant.u_ac = seed as i8 * 4;
                f.tx_select = true;
            });
        }
    }

    #[test]
    fn random_lossless() {
        for seed in 1..4 {
            random(70, 45, seed == 2, seed, |f| f.quant.base = 0);
        }
    }

    /// Two tile columns at effort 4 with tune IQ decode as the encoder reconstructed them.
    #[test]
    fn tiles_decode_as_reconstructed() {
        for (w, h, screen) in [(320, 128, true), (320, 100, false), (500, 70, true)] {
            for img in [graphic(w, h), photo(w, h, 3, 3)] {
                let rows: Vec<&[u8]> = img.iter().map(|p| p.as_slice()).collect();
                let options = crate::enc::Options { tile_cols_log2: 1, ..crate::enc::Options::effort(100, 4, screen, (w * h) as u64).tune_iq() };
                let mut e = crate::enc::encoder(w as u32, h as u32, &rows, &options, false);
                assert!(e.frame.tiling.cols() == 2);
                e.recon = Some(vec![Vec::new(); 3]);
                for y in 0..h {
                    let r: Vec<&[u8]> = img.iter().map(|p| &p[y * w..][..w]).collect();
                    e.push(&r);
                }
                let c = e.finish();
                let decoded = decode(c.data.clone(), w, h, 3);
                let recon = c.recon.as_ref().unwrap();
                for p in 0..3 {
                    if let Some(i) = (0..w * h).find(|&i| decoded[p][i] != recon[p][i]) {
                        panic!("{w}x{h} screen {screen} plane {p}: first difference at ({}, {})", i % w, i / w);
                    }
                }
            }
        }
    }

    /// Tile columns coded on other threads, or refused by them, give the bytes of one thread.
    #[test]
    fn tile_threads() {
        let (w, h) = (700, 150);
        let img = photo(w, h, 3, 7);
        let rows: Vec<&[u8]> = img.iter().map(|p| p.as_slice()).collect();
        let options = crate::enc::Options { tile_cols_log2: 1, ..crate::enc::Options::effort(120, 4, false, (w * h) as u64).tune_iq() };
        let alone = crate::enc::still(w as u32, h as u32, &rows, &options);
        let threads: crate::Spawner = std::sync::Arc::new(|f| {
            std::thread::spawn(f);
            true
        });
        let refusing: crate::Spawner = std::sync::Arc::new(|_| false);
        let dropping: crate::Spawner = std::sync::Arc::new(|_| true);
        let late = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let keeping: crate::Spawner = {
            let late = late.clone();
            std::sync::Arc::new(move |f| {
                late.lock().unwrap().push(f);
                false
            })
        };
        let panicking: crate::Spawner = std::sync::Arc::new(|f| {
            std::thread::spawn(f);
            panic!("a spawner that fails after taking work")
        });
        let shared = std::sync::Arc::new(img.clone());
        for spawner in [&threads, &refusing, &dropping, &keeping] {
            let coded = crate::enc::still_on(w as u32, h as u32, shared.clone(), &options, Some(spawner));
            assert!(coded.data == alone.data);
        }
        for f in late.lock().unwrap().drain(..) {
            f();
        }
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| crate::enc::still_on(w as u32, h as u32, shared.clone(), &options, Some(&panicking))));
        assert!(failed.is_err());
        let one = crate::enc::still(w as u32, h as u32, &rows, &crate::enc::Options { tile_cols_log2: 0, ..options });
        assert!(one.data != alone.data);
    }

    #[test]
    fn random_two_tile_columns() {
        random(4200, 72, false, 3, |f| f.tx_select = true);
    }

    fn psnr(a: &[Vec<u8>], b: &[Vec<u8>]) -> f64 {
        let (mut se, mut n) = (0u64, 0usize);
        for (p, q) in a.iter().zip(b) {
            se += p.iter().zip(q).map(|(&x, &y)| ((x as i32 - y as i32).pow(2)) as u64).sum::<u64>();
            n += p.len();
        }
        10.0 * (255.0f64 * 255.0 * n as f64 / se.max(1) as f64).log10()
    }

    /// A smooth gradient with edges and a little noise, closer to a photo than `image`.
    fn photo(w: usize, h: usize, planes: usize, seed: u64) -> Vec<Vec<u8>> {
        let mut rng = Rng(seed);
        (0..planes)
            .map(|p| {
                (0..w * h)
                    .map(|i| {
                        let (x, y) = ((i % w) as f64, (i / w) as f64);
                        let v = 128.0 + 60.0 * ((x * 0.05 + p as f64).sin() * (y * 0.03).cos())
                            + if (x - 40.0).abs() < 10.0 || (y - 70.0).abs() < 3.0 { 50.0 } else { 0.0 };
                        (v as i32 + (rng.next() % 5) as i32 - 2).clamp(0, 255) as u8
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn rd_search() {
        let cases = [
            (64, 64, false, 100, true, 234, 43.45),
            (130, 70, false, 60, true, 604, 44.24),
            (100, 37, true, 160, false, 59, 40.33),
            (70, 45, false, 0, false, usize::MAX, 0.0),
        ];
        for (w, h, mono, base, tx_select, max_bytes, min_db) in cases {
            let planes = if mono { 1 } else { 3 };
            let seq = sequence(w as u32, h as u32, mono);
            let mut fr = frame(&seq, base);
            fr.tx_select = tx_select;
            let img = photo(w, h, planes, 5);
            let coded = round_trip(&img, w, h, fr, Rd);
            let db = psnr(&img, coded.recon.as_ref().unwrap());
            let bpp = coded.data.len() as f64 * 8.0 / (w * h) as f64;
            eprintln!("rd {w}x{h} mono {mono} q {base}: {db:.4} dB {} bytes {bpp:.3} bpp", coded.data.len());
            if base == 0 {
                assert!(coded.recon.as_ref().unwrap() == &img, "lossless changed pixels");
            }
            assert!(coded.data.len() <= max_bytes && db >= min_db, "{} bytes {db} dB", coded.data.len());
        }
    }

    /// Lossless keeps residuals whose distortion costs less than their bits.
    #[test]
    fn lossless_keeps_small_residuals() {
        let (w, h) = (40, 24);
        let mut img = vec![vec![128u8; w * h]; 3];
        for (i, p) in img.iter_mut().enumerate() {
            p[17 + i * 3] = 129;
            p[w * 9 + 30] = 127;
        }
        let seq = sequence(w as u32, h as u32, false);
        let coded = round_trip(&img, w, h, frame(&seq, 0), Rd);
        assert!(coded.recon.as_ref().unwrap() == &img);
    }

    /// The search beats fixed DC prediction with chosen residuals at the same quantizer.
    #[test]
    fn rd_beats_dc() {
        struct Dc;
        impl Decide for Dc {
            fn partition(&mut self, bl: BlockLevel, x: i32, y: i32, h: Halves) -> BlockPartition {
                Skip(BlockLevel::Bl16x16).partition(bl, x, y, h)
            }
            fn mode(&mut self, _: BlockSize, _: i32, _: i32, _: &Limits) -> (Intra, u8) {
                (Intra::default(), 0)
            }
        }
        let (w, h) = (128, 96);
        let img = photo(w, h, 3, 9);
        let seq = sequence(w as u32, h as u32, false);
        let cost = |c: &Coded| {
            let db = psnr(&img, c.recon.as_ref().unwrap());
            (db, c.data.len())
        };
        let dc = cost(&round_trip(&img, w, h, frame(&seq, 120), Dc));
        let rd = cost(&round_trip(&img, w, h, frame(&seq, 120), Rd));
        eprintln!("dc {dc:?} rd {rd:?}");
        assert!(rd.0 >= dc.0 - 0.3 && rd.1 < dc.1, "dc {dc:?} rd {rd:?}");
    }

    /// Deblocking and CDEF at libaom's strengths for the qindex lift the decoded picture above the
    /// reconstruction, and cost no bytes in the tiles.
    #[test]
    fn filters_from_q() {
        let (w, h) = (192, 128);
        for (mono, qidx) in [(false, 160), (true, 200), (false, 60)] {
            let planes = if mono { 1 } else { 3 };
            let img = photo(w, h, planes, 3);
            let (seq, fr) = crate::enc::headers(w as u32, h as u32, mono, qidx);
            assert!(fr.lf[0] > 0 && fr.cdef.is_some_and(|c| c.y[0] > 0), "{:?}", fr.lf);
            let mut unfiltered_fr = fr.clone();
            unfiltered_fr.lf = [0; 4];
            unfiltered_fr.cdef = None;
            let plain = round_trip_with(&img, w, h, seq.clone(), unfiltered_fr, Rd);
            let coded = round_trip_with(&img, w, h, seq, fr, Rd);
            let unfiltered = psnr(&img, coded.recon.as_ref().unwrap());
            let filtered = psnr(&img, &decode_with(coded.data.clone(), w, h, planes, true));
            eprintln!("filters q {qidx} mono {mono}: {unfiltered:.3} -> {filtered:.3} dB, {} -> {} bytes", plain.data.len(), coded.data.len());
            assert!(filtered > unfiltered + 0.05);
            assert!(coded.data.len() <= plain.data.len() + 8);
        }
    }

    /// Ranking luma modes by SATD first and keeping 4 for rate and distortion loses little.
    #[test]
    fn satd_ranking() {
        let (w, h) = (192, 128);
        let img = photo(w, h, 3, 7);
        let (seq, fr) = crate::enc::headers(w as u32, h as u32, false, 120);
        let run = |keep: usize| {
            let mut e = Encoder::new(seq.clone(), fr.clone(), Rd);
            e.keep_modes = keep;
            e.recon = Some(vec![Vec::new(); 3]);
            let t = std::time::Instant::now();
            for y in 0..h {
                let rows: Vec<&[u8]> = img.iter().map(|p| &p[y * w..][..w]).collect();
                e.push(&rows);
            }
            let c = e.finish();
            let db = psnr(&img, &decode_with(c.data.clone(), w, h, 3, true));
            assert!(c.recon.as_ref().unwrap() == &decode(c.data.clone(), w, h, 3));
            (c.data.len(), db, t.elapsed().as_millis())
        };
        let full = run(0);
        let kept = run(4);
        eprintln!("satd ranking: all modes {full:?}, 4 kept {kept:?}");
        assert!(kept.0 as f64 <= full.0 as f64 * 1.04 && kept.1 >= full.1 - 0.15, "{full:?} {kept:?}");
    }

    /// A picture of flat shapes and one-pixel lines in a few colours.
    fn graphic(w: usize, h: usize) -> Vec<Vec<u8>> {
        let colours = [[235u8, 128, 128], [40, 160, 110], [120, 90, 200], [200, 30, 150], [16, 128, 128]];
        let mut planes = vec![vec![0u8; w * h]; 3];
        for y in 0..h {
            for x in 0..w {
                let k = if (x / 3 + y / 7) % 11 == 0 || y % 17 == 0 {
                    4
                } else if (x as i32 - 60).pow(2) + (y as i32 - 40).pow(2) < 900 {
                    1
                } else if x > 100 && y > 20 && y < 90 {
                    2
                } else if (x + y) % 37 < 3 {
                    3
                } else {
                    0
                };
                for p in 0..3 {
                    planes[p][y * w + x] = colours[k][p];
                }
            }
        }
        planes
    }

    /// On flat colours, palettes code fewer bytes at a higher PSNR than the other intra tools.
    #[test]
    fn palette_on_graphics() {
        let (w, h) = (160, 112);
        let img = graphic(w, h);
        let (seq, fr) = crate::enc::headers(w as u32, h as u32, false, 100);
        let mut screen = fr.clone();
        screen.screen_content = true;
        let plain = round_trip_with(&img, w, h, seq.clone(), fr, Rd);
        let pal = round_trip_with(&img, w, h, seq, screen, Rd);
        let db = |c: &Coded| psnr(&img, &decode_with(c.data.clone(), w, h, 3, true));
        let (a, b) = ((plain.data.len(), db(&plain)), (pal.data.len(), db(&pal)));
        eprintln!("graphic: without palette {a:?}, with {b:?}");
        assert!(b.0 < a.0 && b.1 > a.1, "{a:?} {b:?}");
    }

    /// Every 16×16 block takes the same palettes; the luma colours shrink the delta width after
    /// 223 (254 − 223 needs 5 bits where the first deltas need 7) and V is delta coded.
    #[test]
    fn palette_colour_widths() {
        struct Fixed;
        impl Decide for Fixed {
            fn partition(&mut self, bl: BlockLevel, x: i32, y: i32, h: Halves) -> BlockPartition {
                Skip(BlockLevel::Bl16x16).partition(bl, x, y, h)
            }
            fn mode(&mut self, _: BlockSize, x: i32, _: i32, l: &Limits) -> (Intra, u8) {
                let mut m = Intra::default();
                if l.palette {
                    let y = [[0, 100, 223, 240], [3, 60, 191, 250]][(x / 4 % 2) as usize];
                    m.pal_sz = [4, 3];
                    m.pal[0][..4].copy_from_slice(&y);
                    m.pal[1][..3].copy_from_slice(&[10, 127, 200]);
                    m.pal[2][..3].copy_from_slice(&[128, 140, 120]);
                }
                (m, 0)
            }
            fn residual(&mut self, _: usize, _: TxfmSize, set: &[TxfmType], levels: &mut [i32]) -> Option<TxfmType> {
                levels.fill(0);
                Some(set.first().copied().unwrap_or(DCT_DCT))
            }
        }
        let (w, h) = (72, 40);
        let img = graphic(w, h);
        let seq = sequence(w as u32, h as u32, false);
        let mut fr = frame(&seq, 90);
        fr.screen_content = true;
        round_trip(&img, w, h, fr, Fixed);
    }

    pub fn ppm(path: &std::path::Path) -> (usize, usize, Vec<u8>) {
        let data = std::fs::read(path).unwrap();
        let mut fields = Vec::new();
        let mut at = 0;
        while fields.len() < 4 {
            while data[at].is_ascii_whitespace() {
                at += 1;
            }
            let start = at;
            while !data[at].is_ascii_whitespace() {
                at += 1;
            }
            fields.push(String::from_utf8_lossy(&data[start..at]).to_string());
        }
        let (w, h): (usize, usize) = (fields[1].parse().unwrap(), fields[2].parse().unwrap());
        (w, h, data[at + 1..][..w * h * 3].to_vec())
    }

    /// `ENC_CORPUS=<dir of P6 PPMs> [ENC_QIDX=..] [ENC_KEEP=..] [ENC_CROP=<side>] cargo test --release --features
    /// encoder --lib enc::frame::tests::corpus -- --ignored --nocapture`: time, bytes and PSNR
    /// after the in-loop filters, per picture and in total.
    #[test]
    #[ignore]
    fn corpus() {
        let dir = std::env::var("ENC_CORPUS").unwrap();
        let qidx: u8 = std::env::var("ENC_QIDX").map_or(120, |v| v.parse().unwrap());
        let options = crate::enc::Options {
            qidx,
            keep_modes: std::env::var("ENC_KEEP").map_or(0, |v| v.parse().unwrap()),
            keep_chroma: std::env::var("ENC_KEEP_UV").map_or(0, |v| v.parse().unwrap()),
            screen: std::env::var("ENC_SCREEN").map_or(true, |v| v != "0"),
            rect: std::env::var("ENC_RECT").map_or(1, |v| v.parse().unwrap()),
            late_txtp: std::env::var("ENC_LATE").is_ok_and(|v| v == "1"),
            reduced_txtp: std::env::var("ENC_REDUCED_TXTP").is_ok_and(|v| v == "1"),
            tx_domain_dist: std::env::var("ENC_TX_DOMAIN").is_ok_and(|v| v == "1"),
            uv_from_luma: std::env::var("ENC_UV_FROM_LUMA").is_ok_and(|v| v == "1"),
            prune_intra: std::env::var("ENC_PRUNE_INTRA").is_ok_and(|v| v == "1"),
            rank_rd: std::env::var("ENC_RANK_RD").is_ok_and(|v| v == "1"),
            tx_depths: std::env::var("ENC_TX_DEPTHS").map_or(2, |v| v.parse().unwrap()),
            rank_pre: std::env::var("ENC_RANK_PRE").map_or(0, |v| v.parse().unwrap()),
            levels: std::env::var("ENC_LEVELS").map_or(0..5, |v| {
                let (a, b) = v.split_once("..").unwrap();
                a.parse().unwrap()..b.parse().unwrap()
            }),
            prune: std::env::var("ENC_PRUNE").map_or(0.0, |v| v.parse().unwrap()),
            variance_boost: std::env::var("ENC_BOOST").is_ok_and(|v| v == "1"),
            cnn_prune: std::env::var("ENC_CNN_PRUNE").map_or(0, |v| v.parse().unwrap()),
            rdoq: std::env::var("ENC_RDOQ").map_or(0, |v| v.parse().unwrap()),
            ssim: std::env::var("ENC_SSIM").is_ok_and(|v| v == "1"),
            quant_matrices: std::env::var("ENC_QM").is_ok_and(|v| v == "1"),
            chroma_delta_q: std::env::var("ENC_CHROMA_DQ").is_ok_and(|v| v == "1"),
            iq_rdmult: std::env::var("ENC_IQ_RDMULT").is_ok_and(|v| v == "1"),
            skip_prediction: std::env::var("ENC_SKIP_PRED").map_or(true, |v| v != "0"),
            iq_filters: std::env::var("ENC_IQ_FILTERS").is_ok_and(|v| v == "1"),
            filter_search: std::env::var("ENC_FILTER_SEARCH").is_ok_and(|v| v == "1"),
            uv_outward: std::env::var("ENC_UV_OUTWARD").is_ok_and(|v| v == "1"),
            rect_ratio: std::env::var("ENC_RECT_RATIO").map_or(0.0, |v| v.parse().unwrap()),
            rect_adaptive: std::env::var("ENC_RECT_ADAPTIVE").map_or(true, |v| v != "0"),
            split8_adaptive: std::env::var("ENC_SPLIT8").is_ok_and(|v| v == "1"),
            graphic_guard: false,
            fast_cdef: std::env::var("ENC_FAST_CDEF").is_ok_and(|v| v == "1"),
            tile_cols_log2: std::env::var("ENC_TILES").map_or(1, |v| v.parse().unwrap()),
        };
        let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
        files.sort();
        let (mut ms, mut bytes, mut db) = (0, 0, 0.0);
        for path in &files {
            let (w0, h0, rgb) = ppm(path);
            let crop: usize = std::env::var("ENC_CROP").map_or(usize::MAX, |v| v.parse().unwrap());
            let (w, h) = (w0.min(crop), h0.min(crop));
            let rgb: Vec<u8> = (0..h).flat_map(|y| rgb[y * w0 * 3..][..w * 3].to_vec()).collect();
            let mut planes = vec![vec![0u8; w * h]; 3];
            for (i, px) in rgb.chunks_exact(3).enumerate() {
                let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
                planes[0][i] = ((19595 * r + 38470 * g + 7471 * b + 32768) >> 16) as u8;
                planes[1][i] = ((-11059 * r - 21709 * g + 32768 * b + (128 << 16) + 32768) >> 16).clamp(0, 255) as u8;
                planes[2][i] = ((32768 * r - 27439 * g - 5329 * b + (128 << 16) + 32768) >> 16).clamp(0, 255) as u8;
            }
            let rows: Vec<&[u8]> = planes.iter().map(|p| p.as_slice()).collect();
            // `ENC_EFFORT` takes the product's options: effort, tune IQ and `ENC_FAST=1` for the fast search.
            let options = match std::env::var("ENC_EFFORT") {
                Ok(effort) => {
                    let mut o = crate::enc::Options::effort(qidx, effort.parse().unwrap(), options.screen, (w * h) as u64).tune_iq();
                    if let Ok(t) = std::env::var("ENC_TILES") {
                        o.tile_cols_log2 = t.parse().unwrap();
                    }
                    if std::env::var("ENC_FAST").is_ok_and(|v| v == "1") { o.fast() } else { o }
                }
                Err(_) => options.clone(),
            };
            let t = std::time::Instant::now();
            let threads: crate::Spawner = std::sync::Arc::new(|f| {
                std::thread::spawn(f);
                true
            });
            let shared = std::sync::Arc::new(planes.clone());
            let workers = std::env::var("ENC_THREADS").is_ok_and(|v| v == "1").then_some((&threads, &shared));
            let mut e = crate::enc::still_rows(w as u32, h as u32, &rows, &options, std::env::var("ENC_STATS").is_ok(), workers);
            let searched = t.elapsed().as_millis();
            let stats = e.stats;
            let rect_seen = e.rect_seen;
            let split_seen = e.split_seen;
            let ns = e.stage_ns.map(|n| n / 1_000_000);
            let sizes = e.size_ns.map(|n| n / 1_000_000);
            let estimated = e.estimated as f64 / 4096.0;
            let estimated_db = 10.0 * (65025.0 * (w * h * 3) as f64 / e.estimated_sse.max(1) as f64).log10();
            if options.filter_search {
                e.flush();
                crate::enc::filters::search(&mut e, &rows);
            }
            let c = e.finish();
            let t = t.elapsed().as_millis();
            if std::env::var("ENC_STATS").is_ok() {
                let px = (w * h) as f64;
                let per = stats.map(|[n, c]| format!("{:.2}/{:.1}", n as f64 * 64.0 / px, c as f64 / px));
                println!("  estimated {estimated:.0} B {estimated_db:.2} dB unfiltered");
                println!("  searches per 8x8 / coefficients per pixel: rank {} depth {} chroma {} final {} code {}", per[0], per[1], per[2], per[4], per[5]);
                println!("  ms: rank {} depth {} chroma {} final {}, coded {searched}, with filters {t}; kept modes {}; scoring restore {} predict {} mode bits {} estimate {}", ns[0], ns[1], ns[2], ns[4], ns[3], ns[6], ns[7], ns[8], ns[9]);
                for (l, side) in [64, 32, 16].into_iter().enumerate() {
                    println!("  rect under the ratio at {side}, met/tried/gain%/big by activity: {}", rect_seen[l].iter().map(|s| format!("{}/{}/{:.3}/{}", s.0, s.1, 100.0 * s.2 / s.3.max(1.0), s.4)).collect::<Vec<_>>().join(" "));
                }
                println!("  16x16 splits met/tried/gain%/big by activity: {}", split_seen.iter().map(|s| format!("{}/{}/{:.3}/{}", s.0, s.1, 100.0 * s.2 / s.3.max(1.0), s.4)).collect::<Vec<_>>().join(" "));
                println!("  ms by block side: 4 {} 8 {} 16 {} 32 {} 64 {}", sizes[0], sizes[1], sizes[2], sizes[3], sizes[4]);
            }
            let d = psnr(&planes, &decode_with(c.data.clone(), w, h, 3, true));
            println!("{:20} {t:6} ms {:7} B {d:6.2} dB", path.file_stem().unwrap().to_string_lossy(), c.data.len());
            if std::env::var("ENC_UNFILTERED").is_ok() {
                println!("  before the in-loop filters {:.2} dB", psnr(&planes, &decode_with(c.data.clone(), w, h, 3, false)));
            }
            ms += t;
            bytes += c.data.len();
            db += d;
        }
        println!("total {ms} ms {bytes} B mean {:.3} dB", db / files.len() as f64);
    }

    /// Variance Boost lowers the qindex of flat superblocks: a picture half photo, half gradient
    /// decodes to the reconstruction, with the gradient's superblocks coded finer.
    #[test]
    fn variance_boost() {
        let (w, h) = (256, 128);
        let mut img = photo(w, h, 3, 4);
        for p in img.iter_mut() {
            for y in 0..h {
                for x in w / 2..w {
                    p[y * w + x] = (60 + x / 4 + y / 8) as u8;
                }
            }
        }
        let (seq, mut fr) = crate::enc::headers(w as u32, h as u32, false, 160);
        let plain = round_trip_with(&img, w, h, seq.clone(), fr.clone(), Rd);
        fr.delta_q_log2 = Some(3);
        let mut e = Encoder::new(seq, fr, Rd);
        e.variance_boost = true;
        e.recon = Some(vec![Vec::new(); 3]);
        for y in 0..h {
            let rows: Vec<&[u8]> = img.iter().map(|p| &p[y * w..][..w]).collect();
            e.push(&rows);
        }
        let boosted = e.finish();
        assert!(boosted.recon.as_ref().unwrap() == &decode(boosted.data.clone(), w, h, 3));
        let db = |c: &Coded, half: usize| {
            let d = decode_with(c.data.clone(), w, h, 3, true);
            let pick = |v: &Vec<Vec<u8>>| v.iter().map(|p| (0..h).flat_map(|y| p[y * w + half * w / 2..][..w / 2].to_vec()).collect()).collect::<Vec<Vec<u8>>>();
            psnr(&pick(&img), &pick(&d))
        };
        eprintln!("variance boost: plain {} B {:.2}/{:.2} dB, boosted {} B {:.2}/{:.2} dB", plain.data.len(), db(&plain, 0), db(&plain, 1), boosted.data.len(), db(&boosted, 0), db(&boosted, 1));
        assert!(db(&boosted, 1) > db(&plain, 1) + 0.5);
    }

    /// The CNN prunes partitions of the superblocks wholly in the picture, here all but the right
    /// column and the bottom row; the stream decodes to the reconstruction.
    #[test]
    fn cnn_prune() {
        let (w, h) = (150, 140);
        let img = photo(w, h, 3, 6);
        let (seq, fr) = crate::enc::headers(w as u32, h as u32, false, 100);
        let full = round_trip_with(&img, w, h, seq.clone(), fr.clone(), Rd);
        for level in [1, 2] {
            let mut e = Encoder::new(seq.clone(), fr.clone(), Rd);
            e.cnn_prune = level;
            e.recon = Some(vec![Vec::new(); 3]);
            for y in 0..h {
                let rows: Vec<&[u8]> = img.iter().map(|p| &p[y * w..][..w]).collect();
                e.push(&rows);
            }
            let c = e.finish();
            let recon = c.recon.as_ref().unwrap();
            assert!(recon == &decode(c.data.clone(), w, h, 3));
            eprintln!("cnn prune {level}: {} B {:.2} dB, full search {} B {:.2} dB", c.data.len(), psnr(&img, recon), full.data.len(), psnr(&img, full.recon.as_ref().unwrap()));
        }
    }

    /// A photo beside a flat half: weighted against the picture's mean activity, the photo's
    /// blocks get a larger λ and fewer bits, and the stream still decodes to the reconstruction.
    #[test]
    fn ssim_weighting() {
        let (w, h) = (256, 128);
        let mut img = vec![vec![200u8; w * h]; 3];
        for (i, p) in img.iter_mut().enumerate() {
            for y in 0..h {
                for x in 0..w / 2 {
                    p[y * w + x] = (100 + (((x * 37 + y * 91) ^ (x * y + i)) & 63)) as u8;
                }
            }
        }
        assert!((ssim_reference(8, 8, &[7; 64]) - 17.492222).abs() < 1e-9);
        let (seq, fr) = crate::enc::headers(w as u32, h as u32, false, 120);
        let plain = round_trip_with(&img, w, h, seq.clone(), fr.clone(), Rd);
        let mut e = Encoder::new(seq, fr, Rd);
        e.ssim_reference = Some(ssim_reference(w, h, &img[0]));
        e.recon = Some(vec![Vec::new(); 3]);
        for y in 0..h {
            let rows: Vec<&[u8]> = img.iter().map(|p| &p[y * w..][..w]).collect();
            e.push(&rows);
        }
        let weighted = e.finish();
        assert!(weighted.recon.as_ref().unwrap() == &decode(weighted.data.clone(), w, h, 3));
        eprintln!("ssim weighting: plain {} B, weighted {} B", plain.data.len(), weighted.data.len());
        assert!(weighted.data.len() * 10 < plain.data.len() * 9);
    }

    /// Quantizer matrices, a coarser chroma AC and the tune-IQ λ decode to a picture
    /// close to the source, in fewer bytes than without them.
    #[test]
    fn tune_iq() {
        let (w, h) = (200, 136);
        let img = photo(w, h, 3, 9);
        let planes: Vec<&[u8]> = img.iter().map(|p| p.as_slice()).collect();
        let plain = crate::enc::Options::effort(150, 4, false, u64::MAX);
        let code = |o: &crate::enc::Options| {
            let c = crate::enc::still(w as u32, h as u32, &planes, o);
            let db = psnr(&img, &decode_with(c.data.clone(), w, h, 3, true));
            (c.data.len(), db)
        };
        let iq = crate::enc::Options { quant_matrices: true, chroma_delta_q: true, iq_rdmult: true, ..plain.clone() };
        let (a, b) = (code(&plain), code(&iq));
        eprintln!("tune iq: plain {} B {:.2} dB, tune iq {} B {:.2} dB", a.0, a.1, b.0, b.1);
        assert!(b.0 < a.0 && b.1 > a.1 - 3.0);
    }

    /// Gradient squares 20 px wide, their edges off the 8×8 grid: deblocking and CDEF from the
    /// qindex blur the edges, the searched strengths keep them.
    #[test]
    fn filter_search() {
        let (w, h) = (160, 120);
        let img: Vec<Vec<u8>> = (0..3)
            .map(|p| (0..w * h).map(|i| {
                let (x, y) = (i % w, i / w);
                let base = (x + 2 * y + 60 * p) as u8;
                if (x / 20 + y / 20) % 2 == 0 { base } else { base.wrapping_add(90) }
            }).collect())
            .collect();
        let planes: Vec<&[u8]> = img.iter().map(|p| p.as_slice()).collect();
        let code = |search: bool| {
            let o = crate::enc::Options { filter_search: search, ..crate::enc::Options::effort(144, 4, false, u64::MAX) };
            let c = crate::enc::still(w as u32, h as u32, &planes, &o);
            (c.data.len(), psnr(&img, &decode_with(c.data.clone(), w, h, 3, true)))
        };
        let (from_q, searched) = (code(false), code(true));
        eprintln!("filter search: from q {from_q:?}, searched {searched:?}");
        assert!(searched.0.abs_diff(from_q.0) <= 2 && searched.1 > from_q.1 + 0.5);
    }

    /// libaom's limits: a 16×16 DC of 128 per level against 68/128 of the DC step, and a mean
    /// square error against (dc_q / 8)² / 8, 703 at a DC step of 600.
    #[test]
    fn skip_prediction() {
        let flat = |r: i16| vec![r; 64 * 64];
        let checker = |r: i16| (0..64 * 64).map(|i| if (i / 64 + i % 64) % 2 == 0 { r } else { -r }).collect::<Vec<i16>>();
        let skip = |res: &[i16], dq| predicts_skip(res, 64, 64, 64, 64, dq);
        assert_eq!((skip(&flat(2), [600, 600]), skip(&flat(3), [600, 600])), (true, false));
        assert_eq!((skip(&checker(26), [600, 1 << 20]), skip(&checker(27), [600, 1 << 20])), (true, false));
        assert!(!skip(&checker(2), [600, 20]));
    }
}
