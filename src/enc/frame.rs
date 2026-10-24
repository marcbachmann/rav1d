//! One frame coded a superblock row at a time. The source and the reconstruction are held for
//! one superblock row only; the row above a superblock row is kept as rav1d keeps it, in an edge
//! buffer, and handed to `rav1d_prepare_intra_edges` the same way.

use std::cmp;
use std::sync::Once;

use strum::EnumCount;

use crate::align::AlignedVec64;
use crate::cdf::{rav1d_cdf_thread_copy, rav1d_cdf_thread_init_static, CdfContext};
use crate::cpu::{rav1d_get_cpu_flags, rav1d_init_cpu};
use crate::dequant_tables::DAV1D_DQ_TBL;
use crate::disjoint_mut::DisjointMut;
use crate::enc::ec::Writer;
use crate::enc::header::{obu, BitWriter, Frame, Sequence, OBU_FRAME, OBU_SEQUENCE_HEADER};
use crate::enum_map::DefaultValue;
use crate::env::{gather_left_partition_prob, gather_top_partition_prob};
use crate::include::common::bitdepth::{BitDepth, BitDepth8};
use crate::include::dav1d::headers::Rav1dPixelLayoutSubSampled;
use crate::include::dav1d::picture::Rav1dPictureDataComponent;
use crate::internal::SCRATCH_AC_TXTP_LEN;
use crate::intra_edge::{EdgeFlags, EdgeIndex, IntraEdges};
use crate::ipred::Rav1dIntraPredDSPContext;
use crate::ipred_prepare::rav1d_prepare_intra_edges;
use crate::itx::Rav1dInvTxfmDSPContext;
use crate::levels::{
    BlockLevel, BlockPartition, BlockSize, TxClass, TxfmSize, TxfmType, CFL_PRED, DCT_DCT,
    DC_PRED, FILTER_PRED, IDTX, N_INTRA_PRED_MODES, N_UV_INTRA_PRED_MODES, SMOOTH_H_PRED,
    SMOOTH_PRED, SMOOTH_V_PRED, VERT_LEFT_PRED, VERT_PRED, WHT_WHT,
};
use crate::qm::DAV1D_QM_TBL;
use crate::recon::{get_dc_sign_ctx, get_lo_ctx, get_skip_ctx};
use crate::scan::DAV1D_SCANS;
use crate::tables::{
    TxfmInfo, CFL_ALLOWED_MASK, DAV1D_AL_PART_CTX, DAV1D_BLOCK_SIZES, DAV1D_FILTER_MODE_TO_Y_MODE,
    DAV1D_INTRA_MODE_CONTEXT, DAV1D_LO_CTX_OFFSETS, DAV1D_MAX_TXFM_SIZE_FOR_BS,
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
        self.partition.fill(0);
    }
}

#[derive(Clone, Copy, Default)]
pub struct Intra {
    pub y_mode: u8,
    pub y_angle: i8,
    pub uv_mode: u8,
    pub uv_angle: i8,
    pub cfl: [i8; 2],
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
    pub max_depth: u8,
}

/// The decisions of a frame, asked in coding order.
pub trait Decide {
    fn partition(&mut self, bl: BlockLevel, x: i32, y: i32, halves: Halves) -> BlockPartition;

    /// The superblock's qidx, when the frame codes delta q.
    fn qidx(&mut self, _x: i32, _y: i32, last: u8) -> u8 {
        last
    }

    /// Modes and the transform depth below the largest transform of the block.
    fn mode(&mut self, bs: BlockSize, x: i32, y: i32, limits: &Limits) -> (Intra, u8);

    fn txtp(&mut self, _tx: TxfmSize, set: &[TxfmType]) -> TxfmType {
        set[1]
    }

    /// Levels for one transform block; `src` and `pred` are its pixels, row by row.
    fn levels(&mut self, plane: usize, tx: TxfmSize, txtp: TxfmType, src: &[u8], pred: &[u8], dq: [u32; 2], levels: &mut [i32]);
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

    fn levels(&mut self, _: usize, _: TxfmSize, _: TxfmType, _: &[u8], _: &[u8], _: [u32; 2], levels: &mut [i32]) {
        levels.fill(0);
    }
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

pub struct Encoder<D: Decide> {
    pub seq: Sequence,
    pub frame: Frame,
    pub decide: D,
    dsp: Dsp,
    planes: usize,
    /// Frame size in 4×4 units, rounded up to 8×8 as the decoder rounds it.
    bw: i32,
    bh: i32,
    src: Vec<Plane>,
    rec: Vec<Plane>,
    edge: DisjointMut<AlignedVec64<u8>>,
    edge_stride: usize,
    above: Ctx,
    left: Ctx,
    rows: usize,
    sby: usize,
    tiles: Vec<Tile>,
    coded: Vec<Vec<u8>>,
    /// The superblock's qidx and its delta from the tile's last one, until coded.
    sb_qidx: u8,
    sb_delta: Option<i32>,
    dq: [[u32; 2]; 3],
    qm: [[Option<&'static [u8]>; 3]; TxfmSize::COUNT],
    /// The whole reconstruction, kept only when asked for (tests).
    pub recon: Option<Vec<Vec<u8>>>,
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
        let mut e = Self {
            planes,
            bw,
            bh,
            src,
            rec,
            edge: DisjointMut::new(edge),
            edge_stride: stride,
            above: Ctx::new(bw as usize),
            left: Ctx::new(16),
            rows: 0,
            sby: 0,
            tiles: Vec::new(),
            coded: Vec::new(),
            sb_qidx: q.base,
            sb_delta: None,
            dq: [[0; 2]; 3],
            qm,
            recon: None,
            seq,
            frame,
            decide,
            dsp: Dsp::new(),
        };
        e.dq = e.dq_for(q.base);
        e
    }

    fn dq_for(&self, qidx: u8) -> [[u32; 2]; 3] {
        let q = self.frame.quant;
        let clip = |d: i8| (qidx as i32 + d as i32).clamp(0, 255) as usize;
        let tbl = &DAV1D_DQ_TBL[0];
        let u = [tbl[clip(q.u_dc)][0] as u32, tbl[clip(q.u_ac)][1] as u32];
        [[tbl[clip(q.y_dc)][0] as u32, tbl[qidx as usize][1] as u32], u, u]
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
        assert_eq!(self.rows, self.seq.height as usize);
        if self.rows % SB != 0 {
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
        Coded {
            data,
            config: self.seq.config(),
            recon: self.recon,
        }
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
            self.left.reset();
            let (start, end) = (self.tiles[ti].col_start, self.tiles[ti].col_end);
            for x in (start..end).step_by(16) {
                let last = self.tiles[ti].last_qidx;
                self.sb_qidx = last;
                self.sb_delta = None;
                if let Some(log2) = self.frame.delta_q_log2 {
                    let want = self.decide.qidx(x, y, last);
                    let delta = (want as i32 - last as i32) >> log2;
                    self.sb_qidx = (last as i32 + (delta << log2)).clamp(1, 255) as u8;
                    self.sb_delta = Some(delta);
                }
                self.dq = self.dq_for(self.sb_qidx);
                self.code_sb(ti, BlockLevel::Bl64x64, x, y, EdgeIndex::root());
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
        for tile in self.tiles.drain(..) {
            self.coded.push(tile.w.finish());
        }
    }

    fn code_sb(&mut self, ti: usize, bl: BlockLevel, x: i32, y: i32, edge_index: EdgeIndex) {
        let hsz = 16 >> bl as u8;
        let have_h_split = self.bw > x + hsz;
        let have_v_split = self.bh > y + hsz;
        let edges = &IntraEdges::DEFAULT;
        if !have_h_split && !have_v_split {
            let next = bl.decrease().unwrap();
            return self.code_sb(ti, next, x, y, edges.branch(false, edge_index).split[0]);
        }
        let bx8 = (x >> 1) as usize;
        let by8 = ((y & 15) >> 1) as usize;
        let has_bl = |v: u8| (v >> (4 - bl as u8)) & 1;
        let ctx = (has_bl(self.above.partition[bx8]) + 2 * has_bl(self.left.partition[by8])) as usize;
        let halves = match (have_h_split, have_v_split) {
            (true, true) => Halves::Both,
            (true, false) => Halves::Top,
            _ => Halves::Left,
        };
        let bp = self.decide.partition(bl, x, y, halves);
        {
            let tile = &mut self.tiles[ti];
            let pc = &mut tile.cdf.m.partition[bl as usize][ctx];
            match halves {
                Halves::Both => {
                    let n = DAV1D_PARTITION_TYPE_COUNT[bl as usize];
                    assert!((bp as u8) <= n);
                    tile.w.symbol(bp as u8, pc, n);
                }
                Halves::Top => {
                    assert!(matches!(bp, BlockPartition::Split | BlockPartition::H));
                    tile.w.bool(bp == BlockPartition::Split, gather_top_partition_prob(pc, bl));
                }
                Halves::Left => {
                    assert!(matches!(bp, BlockPartition::Split | BlockPartition::V));
                    tile.w.bool(bp == BlockPartition::Split, gather_left_partition_prob(pc, bl));
                }
            }
        }
        let b = DAV1D_BLOCK_SIZES[bl as usize][bp as usize];
        let node = edges.node(false, edge_index);
        let (nh, nv, no) = (node.h, node.v, node.o);
        use BlockPartition::*;
        match bp {
            None => self.code_b(ti, b[0], x, y, no),
            H => {
                self.code_b(ti, b[0], x, y, nh[0]);
                if have_v_split {
                    self.code_b(ti, b[0], x, y + hsz, nh[1]);
                }
            }
            V => {
                self.code_b(ti, b[0], x, y, nv[0]);
                if have_h_split {
                    self.code_b(ti, b[0], x + hsz, y, nv[1]);
                }
            }
            Split if bl == BlockLevel::Bl8x8 => {
                let tip = edges.tip(false, edge_index).split;
                self.code_b(ti, BlockSize::Bs4x4, x, y, EdgeFlags::ALL_TR_AND_BL);
                self.code_b(ti, BlockSize::Bs4x4, x + 1, y, tip[0]);
                self.code_b(ti, BlockSize::Bs4x4, x, y + 1, tip[1]);
                self.code_b(ti, BlockSize::Bs4x4, x + 1, y + 1, tip[2]);
            }
            Split => {
                let next = bl.decrease().unwrap();
                let split = edges.branch(false, edge_index).split;
                self.code_sb(ti, next, x, y, split[0]);
                if have_h_split {
                    self.code_sb(ti, next, x + hsz, y, split[1]);
                }
                if have_v_split {
                    self.code_sb(ti, next, x, y + hsz, split[2]);
                    if have_h_split {
                        self.code_sb(ti, next, x + hsz, y + hsz, split[3]);
                    }
                }
            }
            TopSplit => {
                self.code_b(ti, b[0], x, y, EdgeFlags::ALL_TR_AND_BL);
                self.code_b(ti, b[0], x + hsz, y, nv[1]);
                self.code_b(ti, b[1], x, y + hsz, nh[1]);
            }
            BottomSplit => {
                self.code_b(ti, b[0], x, y, nh[0]);
                self.code_b(ti, b[1], x, y + hsz, nv[0]);
                self.code_b(ti, b[1], x + hsz, y + hsz, EdgeFlags::empty());
            }
            LeftSplit => {
                self.code_b(ti, b[0], x, y, EdgeFlags::ALL_TR_AND_BL);
                self.code_b(ti, b[0], x, y + hsz, nh[1]);
                self.code_b(ti, b[1], x + hsz, y, nv[1]);
            }
            RightSplit => {
                self.code_b(ti, b[0], x, y, nv[0]);
                self.code_b(ti, b[1], x + hsz, y, nh[0]);
                self.code_b(ti, b[1], x + hsz, y + hsz, EdgeFlags::empty());
            }
            H4 => {
                let branch = edges.branch(false, edge_index);
                let (h4, h0, h1) = (branch.h4, branch.node.h[0], branch.node.h[1]);
                self.code_b(ti, b[0], x, y, h0);
                self.code_b(ti, b[0], x, y + (hsz >> 1), h4);
                self.code_b(ti, b[0], x, y + hsz, EdgeFlags::ALL_LEFT_HAS_BOTTOM);
                if y + (hsz * 3 >> 1) < self.bh {
                    self.code_b(ti, b[0], x, y + (hsz * 3 >> 1), h1);
                }
            }
            V4 => {
                let branch = edges.branch(false, edge_index);
                let (v4, v0, v1) = (branch.v4, branch.node.v[0], branch.node.v[1]);
                self.code_b(ti, b[0], x, y, v0);
                self.code_b(ti, b[0], x + (hsz >> 1), y, v4);
                self.code_b(ti, b[0], x + hsz, y, EdgeFlags::ALL_TOP_HAS_RIGHT);
                if x + (hsz * 3 >> 1) < self.bw {
                    self.code_b(ti, b[0], x + (hsz * 3 >> 1), y, v1);
                }
            }
        }
        if bp != Split || bl == BlockLevel::Bl8x8 {
            let n = hsz as usize;
            self.above.partition[bx8..bx8 + n].fill(DAV1D_AL_PART_CTX[0][bl as usize][bp as usize]);
            self.left.partition[by8..by8 + n].fill(DAV1D_AL_PART_CTX[1][bl as usize][bp as usize]);
        }
    }

    fn code_b(&mut self, ti: usize, bs: BlockSize, x: i32, y: i32, edge_flags: EdgeFlags) {
        let b_dim = bs.dimensions();
        let (bw4, bh4) = (b_dim[0] as i32, b_dim[1] as i32);
        let w4 = cmp::min(bw4, self.bw - x);
        let h4 = cmp::min(bh4, self.bh - y);
        let lossless = self.frame.quant.lossless();
        let max_tx = if lossless { TxfmSize::S4x4 } else { DAV1D_MAX_TXFM_SIZE_FOR_BS[bs as usize][0] };
        let max_dim = &DAV1D_TXFM_DIMENSIONS[max_tx as usize];
        let limits = Limits {
            angles: b_dim[2] + b_dim[3] >= 2,
            cfl: if lossless { bw4 == 1 && bh4 == 1 } else { CFL_ALLOWED_MASK & (1 << bs as u8) != 0 },
            filter: cmp::max(b_dim[2], b_dim[3]) <= 3 && self.seq.filter_intra,
            chroma: self.planes > 1,
            max_depth: if self.frame.tx_select && !lossless { cmp::min(max_dim.max, 2) } else { 0 },
        };
        let (intra, depth) = self.decide.mode(bs, x, y, &limits);
        assert!(depth <= limits.max_depth);
        let mut tx = max_tx;
        for _ in 0..depth {
            tx = DAV1D_TXFM_DIMENSIONS[tx as usize].sub;
        }
        let uvtx = if lossless { TxfmSize::S4x4 } else { DAV1D_MAX_TXFM_SIZE_FOR_BS[bs as usize][3] };
        let (ax, ly) = (x as usize, (y & 15) as usize);
        let edge_filter = (self.seq.edge_filter as i32) << 10;

        let mut coefs = Vec::new();
        let y_flags = self.sm_flag(ax, ly) | edge_filter;
        self.predict_and_choose(ti, 0, x, y, w4, h4, tx, &intra, edge_flags, y_flags, true, lossless, &mut coefs);
        if self.planes > 1 {
            if intra.uv_mode == CFL_PRED {
                self.predict_cfl(ti, x, y, w4, h4, bw4, bh4, tx, uvtx, &intra);
            }
            let uv_flags = self.sm_uv_flag(ax, ly) | edge_filter;
            for p in 1..3 {
                let predict = !(intra.uv_mode == CFL_PRED && intra.cfl[p - 1] != 0);
                self.predict_and_choose(ti, p, x, y, w4, h4, uvtx, &intra, edge_flags, uv_flags, predict, lossless, &mut coefs);
            }
        }
        let skip = coefs.iter().all(|c: &(usize, i32, i32, Coefs, i32)| c.4 < 0);

        let tile = &mut self.tiles[ti];
        let sctx = (self.above.skip[ax] + self.left.skip[ly]) as usize;
        tile.w.bool_adapt(skip, &mut tile.cdf.m.skip[sctx]);
        if x & 15 == 0 && y & 15 == 0 {
            match self.sb_delta {
                Some(delta) if bs != BlockSize::Bs64x64 || !skip => {
                    let abs = delta.unsigned_abs();
                    tile.w.symbol(cmp::min(abs, 3) as u8, &mut tile.cdf.m.delta_q.0, 3);
                    if abs >= 3 {
                        let n_bits = 31 - (abs - 1).leading_zeros();
                        tile.w.bools(n_bits - 1, 3);
                        tile.w.bools(abs - 1 - (1 << n_bits), n_bits as u8);
                    }
                    if abs != 0 {
                        tile.w.bool_equi(delta < 0);
                    }
                    tile.last_qidx = self.sb_qidx;
                }
                _ => {}
            }
        }
        let (y_mode, filter) = if intra.y_mode == FILTER_PRED {
            (DC_PRED, Some(intra.y_angle as u8))
        } else {
            (intra.y_mode, None)
        };
        let ymode_cdf = &mut tile.cdf.kfym[DAV1D_INTRA_MODE_CONTEXT[self.above.mode[ax] as usize] as usize]
            [DAV1D_INTRA_MODE_CONTEXT[self.left.mode[ly] as usize] as usize];
        tile.w.symbol(y_mode, ymode_cdf, N_INTRA_PRED_MODES as u8 - 1);
        if limits.angles && (VERT_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
            let acdf = &mut tile.cdf.m.angle_delta[(y_mode - VERT_PRED) as usize];
            tile.w.symbol((intra.y_angle + 3) as u8, acdf, 6);
        }
        if limits.chroma {
            let cdf = &mut tile.cdf.m.uv_mode[limits.cfl as usize][y_mode as usize];
            tile.w.symbol(intra.uv_mode, cdf, N_UV_INTRA_PRED_MODES as u8 - 1 - !limits.cfl as u8);
            if intra.uv_mode == CFL_PRED {
                let sign = intra.cfl.map(|a| if a == 0 { 0u8 } else if a < 0 { 1 } else { 2 });
                tile.w.symbol(sign[0] * 3 + sign[1] - 1, &mut tile.cdf.m.cfl_sign.0, 7);
                for i in 0..2 {
                    if sign[i] != 0 {
                        let ctx = (sign[i] == 2) as usize * 3 + sign[1 - i] as usize;
                        tile.w.symbol(intra.cfl[i].unsigned_abs() - 1, &mut tile.cdf.m.cfl_alpha[ctx], 15);
                    }
                }
            } else if limits.angles && (VERT_PRED..=VERT_LEFT_PRED).contains(&intra.uv_mode) {
                let acdf = &mut tile.cdf.m.angle_delta[(intra.uv_mode - VERT_PRED) as usize];
                tile.w.symbol((intra.uv_angle + 3) as u8, acdf, 6);
            }
        }
        if self.frame.screen_content && cmp::max(bw4, bh4) <= 16 && bw4 + bh4 >= 4 {
            let sz_ctx = (b_dim[2] + b_dim[3] - 2) as usize;
            if y_mode == DC_PRED {
                let pal_ctx = (self.above.pal_sz[ax] > 0) as usize + (self.left.pal_sz[ly] > 0) as usize;
                tile.w.bool_adapt(false, &mut tile.cdf.m.pal_y[sz_ctx][pal_ctx]);
            }
            if limits.chroma && intra.uv_mode == DC_PRED {
                tile.w.bool_adapt(false, &mut tile.cdf.m.pal_uv[0]);
            }
        }
        if y_mode == DC_PRED && limits.filter {
            tile.w.bool_adapt(filter.is_some(), &mut tile.cdf.m.use_filter_intra[bs as usize]);
            if let Some(f) = filter {
                tile.w.symbol(f, &mut tile.cdf.m.filter_intra.0, 4);
            }
        }
        if self.frame.tx_select && !lossless && max_dim.max > 0 {
            let tctx = (self.left.tx_intra[ly] as i32 >= max_dim.lh as i32) as usize
                + (self.above.tx_intra[ax] as i32 >= max_dim.lw as i32) as usize;
            let cdf = &mut tile.cdf.m.txsz[(max_dim.max - 1) as usize][tctx];
            tile.w.symbol(depth, cdf, cmp::min(max_dim.max, 2));
        }
        if !skip {
            for (plane, x4, y4, c, eob) in &coefs {
                let t = if *plane == 0 { tx } else { uvtx };
                write_coefs(
                    &mut tile.w,
                    &mut tile.cdf,
                    &mut self.above,
                    &mut self.left,
                    *plane,
                    *x4 as usize,
                    *y4 as usize,
                    t,
                    bs,
                    c,
                    *eob,
                    &intra,
                    lossless,
                    self.frame.quant.base,
                    self.frame.reduced_tx_set,
                    self.bw,
                    self.bh,
                );
            }
        } else {
            for (plane, x4, y4, _, _) in &coefs {
                let t_dim = &DAV1D_TXFM_DIMENSIONS[if *plane == 0 { tx } else { uvtx } as usize];
                let (a, l) = coef_ctx(&mut self.above, &mut self.left, *plane);
                a[*x4 as usize..][..t_dim.w as usize].fill(0x40);
                l[(*y4 & 15) as usize..][..t_dim.h as usize].fill(0x40);
            }
        }
        let t_dim = &DAV1D_TXFM_DIMENSIONS[tx as usize];
        let (n, m) = (bw4 as usize, bh4 as usize);
        self.above.tx_intra[ax..ax + n].fill(t_dim.lw as i8);
        self.left.tx_intra[ly..ly + m].fill(t_dim.lh as i8);
        self.above.mode[ax..ax + n].fill(y_mode);
        self.left.mode[ly..ly + m].fill(y_mode);
        self.above.pal_sz[ax..ax + n].fill(0);
        self.left.pal_sz[ly..ly + m].fill(0);
        self.above.skip[ax..ax + n].fill(skip as u8);
        self.left.skip[ly..ly + m].fill(skip as u8);
        if limits.chroma {
            self.above.uvmode[ax..ax + n].fill(intra.uv_mode);
            self.left.uvmode[ly..ly + m].fill(intra.uv_mode);
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
        let t_dim = &DAV1D_TXFM_DIMENSIONS[tx as usize];
        let uv_dim = &DAV1D_TXFM_DIMENSIONS[uvtx as usize];
        let tile = &self.tiles[ti];
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
        for pl in 0..2 {
            if intra.cfl[pl] == 0 {
                continue;
            }
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
                intra.cfl[pl] as i32,
                BD::new(()),
            );
        }
    }

    /// Predicts every transform block of one plane, chooses its residual and reconstructs it.
    /// Pushes `(plane, x4, y4, coefs, eob)` per transform block, in coding order.
    #[allow(clippy::too_many_arguments)]
    fn predict_and_choose(
        &mut self,
        ti: usize,
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
        lossless: bool,
        out: &mut Vec<(usize, i32, i32, Coefs, i32)>,
    ) {
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
        let set = if self.frame.reduced_tx_set || t_dim.min == TxfmSize::S16x16 as u8 {
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
                let rec = &self.rec[plane];
                let wy = (py as usize * 4) % SB;
                let dst = rec.comp.with_offset::<BD>() + (wy * rec.stride + px as usize * 4);
                if predict {
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
                let mut src = vec![0u8; pw * ph];
                let mut pred = vec![0u8; pw * ph];
                for r in 0..ph {
                    let o = (wy + r) * rec.stride + px as usize * 4;
                    pred[r * pw..][..pw].copy_from_slice(&rec.comp.slice::<BD, _>((o.., ..pw)));
                    let s = &self.src[plane];
                    src[r * pw..][..pw].copy_from_slice(&s.comp.slice::<BD, _>((o.., ..pw)));
                }
                let txtp = if coded_txtp { self.decide.txtp(tx, set) } else { fixed_txtp };
                assert!(!coded_txtp || set.contains(&txtp));
                let n = cmp::min(pw, 32) * cmp::min(ph, 32);
                let mut levels = vec![0i32; n];
                self.decide.levels(plane, tx, txtp, &src, &pred, self.dq[plane], &mut levels);
                let eob = eob_of(tx, txtp, &levels);
                if eob >= 0 {
                    let mut cf = Coef([0; 32 * 32]);
                    self.dequant(plane, tx, txtp, &levels, &mut cf.0[..n]);
                    self.dsp.itx.itxfm_add[tx as usize][txtp as usize].call::<BD>(dst, &mut cf.0[..n], eob, BD::new(()));
                }
                out.push((plane, px, py, Coefs { txtp, levels }, eob));
                x += tw;
            }
            y += th;
        }
    }

    fn dequant(&self, plane: usize, tx: TxfmSize, txtp: TxfmType, levels: &[i32], cf: &mut [i16]) {
        let t_dim = &DAV1D_TXFM_DIMENSIONS[tx as usize];
        let shift = cmp::max(0, t_dim.ctx as i32 - 2);
        let qm = if txtp < IDTX { self.qm[tx as usize][plane] } else { None };
        for (rc, (&l, c)) in levels.iter().zip(cf.iter_mut()).enumerate() {
            if l == 0 {
                *c = 0;
                continue;
            }
            let mut dq = self.dq[plane][(rc != 0) as usize];
            if let Some(qm) = qm {
                dq = dq * qm[rc] as u32 + 16 >> 5;
            }
            let tok = l.unsigned_abs() & 0xfffff;
            let v = ((dq.wrapping_mul(tok) & 0xffffff) >> shift) as i32;
            let v = cmp::min(v, 0x7fff + (l < 0) as i32);
            *c = if l < 0 { -v } else { v } as i16;
        }
    }
}

fn smooth(m: u8) -> i32 {
    if matches!(m, SMOOTH_PRED | SMOOTH_V_PRED | SMOOTH_H_PRED) {
        512
    } else {
        0
    }
}

#[repr(C, align(64))]
struct Ac([i16; SCRATCH_AC_TXTP_LEN]);

const EDGE_LEN: usize = 257;

#[repr(C, align(64))]
struct EdgePx([u8; EDGE_LEN]);

#[repr(C, align(64))]
struct Coef([i16; 32 * 32]);

fn coef_ctx<'a>(above: &'a mut Ctx, left: &'a mut Ctx, plane: usize) -> (&'a mut [u8], &'a mut [u8]) {
    if plane == 0 {
        (&mut above.lcoef, &mut left.lcoef)
    } else {
        (&mut above.ccoef[plane - 1], &mut left.ccoef[plane - 1])
    }
}

/// Coefficient positions in coding order: scan index to rav1d's coefficient index.
fn scan_rc(tx: TxfmSize, class: TxClass, i: usize) -> (usize, u8, u8) {
    let t_dim = &DAV1D_TXFM_DIMENSIONS[tx as usize];
    let slw = cmp::min(t_dim.lw, TxfmSize::S32x32 as u8);
    let slh = cmp::min(t_dim.lh, TxfmSize::S32x32 as u8);
    match class {
        TxClass::TwoD => {
            let rc = DAV1D_SCANS[tx as usize][i].get() as usize;
            (rc, (rc >> (slh + 2)) as u8, (rc & ((4 << slh) - 1)) as u8)
        }
        TxClass::H => {
            let x = (i & ((4 << slh) - 1)) as u8;
            let y = (i >> (slh + 2)) as u8;
            (i, x, y)
        }
        TxClass::V => {
            let x = (i & ((4 << slw) - 1)) as u8;
            let y = (i >> (slw + 2)) as u8;
            ((x as usize) << (slh + 2) | y as usize, x, y)
        }
    }
}

fn eob_of(tx: TxfmSize, txtp: TxfmType, levels: &[i32]) -> i32 {
    let class = DAV1D_TX_TYPE_CLASS[txtp as usize];
    (0..levels.len())
        .rev()
        .find(|&i| levels[scan_rc(tx, class, i).0] != 0)
        .map_or(-1, |i| i as i32)
}

fn golomb(w: &mut Writer, v: u32) {
    let x = v + 1;
    let len = 31 - x.leading_zeros();
    for _ in 0..len {
        w.bool_equi(false);
    }
    w.bool_equi(true);
    for i in (0..len).rev() {
        w.bool_equi(x >> i & 1 != 0);
    }
}

#[allow(clippy::too_many_arguments)]
fn write_coefs(
    w: &mut Writer,
    cdf: &mut CdfContext,
    above: &mut Ctx,
    left: &mut Ctx,
    plane: usize,
    x4: usize,
    y4: usize,
    tx: TxfmSize,
    bs: BlockSize,
    c: &Coefs,
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
    let (a, l) = coef_ctx(above, left, plane);
    let sctx = get_skip_ctx(
        t_dim,
        bs,
        &a[x4..x4 + t_dim.w as usize],
        &l[ly..ly + t_dim.h as usize],
        plane != 0,
        crate::include::dav1d::headers::Rav1dPixelLayout::I444,
    );
    let all_skip = eob < 0;
    w.bool_adapt(all_skip, &mut cdf.coef.skip[t_dim.ctx as usize][sctx.get() as usize]);
    let set_ctx = |a: &mut [u8], l: &mut [u8], v: u8| {
        let n = cmp::min(t_dim.w as i32, bw - x4 as i32) as usize;
        let m = cmp::min(t_dim.h as i32, bh - y4 as i32) as usize;
        a[x4..x4 + n].fill(v);
        l[ly..ly + m].fill(v);
    };
    if all_skip {
        set_ctx(a, l, 0x40);
        return;
    }
    let txtp = c.txtp;
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

    let level_at = |i: usize| c.levels[scan_rc(tx, class, i).0].unsigned_abs();
    let pack = |l: u32| ((cmp::min(l, 3) << 6) | cmp::min(l, 15)) as u8;
    let hi_ctx = cmp::min(t_dim.ctx, 3) as usize;
    let mut levels = [0u8; 1088];
    let stride: usize = if class == TxClass::TwoD { 4 << slh } else { 16 };
    let offsets = if class == TxClass::TwoD {
        let is_rect = tx.is_rect() as usize;
        Some(&DAV1D_LO_CTX_OFFSETS[is_rect + (tx as usize & is_rect)])
    } else {
        None
    };
    let level_off = |rc: usize, x: u8, y: u8| {
        if class == TxClass::TwoD {
            rc
        } else {
            x as usize * stride + y as usize
        }
    };
    let dc_level;
    if eob > 0 {
        let (rc, x, y) = scan_rc(tx, class, eob as usize);
        let l = level_at(eob as usize);
        let ctx = 1 + (eob > (2 << tx2dszctx)) as usize + (eob > (4 << tx2dszctx)) as usize;
        w.symbol((cmp::min(l, 3) - 1) as u8, &mut cdf.coef.eob_base_tok[t_dim.ctx as usize][chroma][ctx], 2);
        if l >= 3 {
            let far = if class == TxClass::TwoD { (x | y) > 1 } else { y != 0 };
            let ctx = if far { 14 } else { 7 };
            w.hi_tok(cmp::min(l, 15) as u8, &mut cdf.coef.br_tok[hi_ctx][chroma][ctx]);
        }
        levels[level_off(rc, x, y)] = pack(l);
        for i in (1..eob as usize).rev() {
            let (rc, x, mut y) = scan_rc(tx, class, i);
            let off = level_off(rc, x, y);
            let mut mag = 0;
            let lo = get_lo_ctx(&levels[off..], class, &mut mag, offsets, x, y, stride as u8);
            let l = level_at(i);
            w.symbol(cmp::min(l, 3) as u8, &mut cdf.coef.base_tok[t_dim.ctx as usize][chroma][lo.get()], 3);
            if class == TxClass::TwoD {
                y |= x;
            }
            if l >= 3 {
                let mag = mag as u8 & 63;
                let ctx = if y > (class == TxClass::TwoD) as u8 { 14 } else { 7 }
                    + if mag > 12 { 6 } else { (mag + 1) >> 1 } as usize;
                w.hi_tok(cmp::min(l, 15) as u8, &mut cdf.coef.br_tok[hi_ctx][chroma][ctx]);
            }
            levels[off] = pack(l);
        }
        let mut mag = 0;
        let lo = if class == TxClass::TwoD {
            0
        } else {
            get_lo_ctx(&levels, class, &mut mag, offsets, 0, 0, stride as u8).get()
        };
        let l = level_at(0);
        w.symbol(cmp::min(l, 3) as u8, &mut cdf.coef.base_tok[t_dim.ctx as usize][chroma][lo], 3);
        if l >= 3 {
            if class == TxClass::TwoD {
                mag = levels[1] as u32 + levels[stride] as u32 + levels[stride + 1] as u32;
            }
            let mag = mag as u8 & 63;
            let ctx = if mag > 12 { 6 } else { (mag + 1) >> 1 } as usize;
            w.hi_tok(cmp::min(l, 15) as u8, &mut cdf.coef.br_tok[hi_ctx][chroma][ctx]);
        }
        dc_level = l;
    } else {
        let l = level_at(0);
        w.symbol((cmp::min(l, 3) - 1) as u8, &mut cdf.coef.eob_base_tok[t_dim.ctx as usize][chroma][0], 2);
        if l >= 3 {
            w.hi_tok(cmp::min(l, 15) as u8, &mut cdf.coef.br_tok[hi_ctx][chroma][0]);
        }
        dc_level = l;
    }
    let mut cul_level = 0u32;
    let dc_sign_level;
    let dc = c.levels[0];
    if dc_level == 0 {
        dc_sign_level = 0x40;
    } else {
        let ctx = get_dc_sign_ctx(tx, &a[x4..], &l[ly..]) as usize;
        w.bool_adapt(dc < 0, &mut cdf.coef.dc_sign[chroma][ctx]);
        if dc_level >= 15 {
            golomb(w, dc_level - 15);
        }
        dc_sign_level = if dc < 0 { 0 } else { 0x80 };
        cul_level = dc_level;
    }
    for i in 1..=eob as usize {
        let v = c.levels[scan_rc(tx, class, i).0];
        if v == 0 {
            continue;
        }
        w.bool_equi(v < 0);
        let l = v.unsigned_abs();
        if l >= 15 {
            golomb(w, l - 15);
        }
        cul_level = cul_level.wrapping_add(l);
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
            }
            (m, r.below(l.max_depth as u32 + 1) as u8)
        }

        fn txtp(&mut self, _: TxfmSize, set: &[TxfmType]) -> TxfmType {
            set[self.rng.below(set.len() as u32) as usize]
        }

        fn levels(&mut self, _: usize, _: TxfmSize, _: TxfmType, _: &[u8], _: &[u8], _: [u32; 2], levels: &mut [i32]) {
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
        }
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

    pub fn decode(data: Vec<u8>, w: usize, h: usize, planes: usize) -> Vec<Vec<u8>> {
        let mut s = Settings::new();
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

    /// Codes `img`, decodes it with rav1d and checks the planes equal the encoder's reconstruction.
    pub fn round_trip<D: Decide>(img: &[Vec<u8>], w: usize, h: usize, frame: Frame, decide: D) -> Coded {
        let planes = img.len();
        let seq = sequence(w as u32, h as u32, planes == 1);
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

    #[test]
    fn random_two_tile_columns() {
        random(4200, 72, false, 3, |f| f.tx_select = true);
    }
}
