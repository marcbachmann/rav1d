//! libaom's all-intra CNN partition pruning, ported from libaom v3.15.1 (Copyright (c) 2019,
//! Alliance for Open Media, BSD 2-Clause): `intra_mode_cnn_partition` in
//! av1/encoder/partition_strategy.c, `av1_cnn_predict_c` in av1/encoder/cnn.c for the layers the
//! model uses (valid 2-D convolutions with ReLU), `av1_nn_predict_c` in av1/encoder/ml.c, and the
//! model of av1/encoder/partition_cnn_weights.h. Every sum runs in f32 in libaom's C order, so
//! the outputs are bit-identical to its C code (`scripts/cnn-partition-compare.sh`).

mod weights;

use weights::*;

use crate::dequant_tables::DAV1D_DQ_TBL;

const SIDE: usize = 65;

/// The CNN's outputs for one 64×64 superblock: 20 channels at 1×1, 4 at 2×2, 20 at 4×4 and 20
/// at 8×8, channel by channel, and the normalised log of the DC quantizer.
pub struct Cnn {
    b0: [f32; 20],
    b1: [f32; 4 * 4],
    b2: [f32; 20 * 16],
    b3: [f32; 20 * 64],
    log_q: f32,
}

/// What the model prunes for one block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Prune {
    pub none: bool,
    pub split: bool,
    pub rect: bool,
}

fn relu(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}

/// `av1_cnn_convolve_no_maxpool_padding_valid_c` with ReLU, a square input of `side` channel by
/// channel; the output channels run side by side, each summed in libaom's order.
fn conv<const OUT: usize>(input: &[f32], side: usize, k: usize, stride: usize, kernel: &[f32], bias: &[f32; OUT], out: &mut [f32]) {
    let in_ch = input.len() / (side * side);
    let out_side = (side - k + stride) / stride;
    for u in 0..out_side {
        for v in 0..out_side {
            let mut acc = *bias;
            for c in 0..in_ch {
                for ii in 0..k {
                    let row = &input[c * side * side + (u * stride + ii) * side + v * stride..][..k];
                    for (jj, &x) in row.iter().enumerate() {
                        let w = &kernel[((ii * k + jj) * in_ch + c) * OUT..][..OUT];
                        for (a, &w) in acc.iter_mut().zip(w) {
                            *a += w * x;
                        }
                    }
                }
            }
            for (i, a) in acc.into_iter().enumerate() {
                out[i * out_side * out_side + u * out_side + v] = relu(a);
            }
        }
    }
}

/// `av1_nn_predict_c` of a model with two hidden layers and one output, its precision reduced.
fn dense(features: &[f32], layers: [(&[f32], &[f32]); 3]) -> f32 {
    let mut input = [0f32; 41];
    input[..features.len()].copy_from_slice(features);
    let mut n = features.len();
    for (kernel, bias) in &layers[..2] {
        let mut out = [0f32; 41];
        for (node, &b) in bias.iter().enumerate() {
            let mut val = b;
            for i in 0..n {
                val += kernel[node * n + i] * input[i];
            }
            out[node] = relu(val);
        }
        input = out;
        n = bias.len();
    }
    let (kernel, bias) = layers[2];
    let mut val = bias[0];
    for i in 0..n {
        val += kernel[i] * input[i];
    }
    ((val * 512.0) as f64 + 0.5) as i32 as f32 * (1.0 / 512.0) as f32
}

impl Cnn {
    /// The 65×65 luma pixels from one row above and one column left of the superblock, as
    /// libaom reads its border-extended source, at the superblock's `qidx`.
    pub fn new(pixels: &[u8; SIDE * SIDE], qidx: u8) -> Self {
        let input: Vec<f32> = pixels.iter().map(|&p| p as f32 / 255.0).collect();
        let mut l0 = [0f32; 20 * 16 * 16];
        let mut s = Self { b0: [0.0; 20], b1: [0.0; 16], b2: [0.0; 320], b3: [0.0; 1280], log_q: 0.0 };
        conv::<20>(&input, SIDE, 5, 4, &CNN_LAYER_0_KERNEL, &CNN_LAYER_0_BIAS, &mut l0);
        conv::<20>(&l0, 16, 2, 2, &CNN_LAYER_1_KERNEL, &CNN_LAYER_1_BIAS, &mut s.b3);
        conv::<20>(&s.b3, 8, 2, 2, &CNN_LAYER_2_KERNEL, &CNN_LAYER_2_BIAS, &mut s.b2);
        conv::<4>(&s.b2, 4, 2, 2, &CNN_LAYER_3_KERNEL, &CNN_LAYER_3_BIAS, &mut s.b1);
        conv::<20>(&s.b1, 2, 2, 2, &CNN_LAYER_4_KERNEL, &CNN_LAYER_4_BIAS, &mut s.b0);
        let dc_q = DAV1D_DQ_TBL[0][qidx as usize][0] as i32;
        s.log_q = (((dc_q * dc_q) as f32 / 256.0).ln_1p() - MEAN[0]) / STD[0];
        s
    }

    /// The outputs in libaom's `cnn_buffer` order, and the normalised log of the quantizer.
    pub fn outputs(&self) -> (Vec<f32>, f32) {
        ([&self.b0[..], &self.b1, &self.b2, &self.b3].concat(), self.log_q)
    }

    /// The model's logit for the block at `level` (0 is 64×64, 3 is 8×8) and position `bx`, `by`
    /// in blocks of that level within the superblock.
    pub fn logit(&self, level: usize, bx: usize, by: usize) -> f32 {
        let mut f = [0f32; 41];
        let mut n = 0;
        let mut put = |v: f32| {
            f[n] = v;
            n += 1;
        };
        match level {
            0 => {
                self.b0.iter().for_each(|&v| put(v));
                (0..4).for_each(|lin| (0..4).for_each(|ch| put(self.b1[lin + ch * 4])));
            }
            1 => {
                self.b0.iter().for_each(|&v| put(v));
                (0..4).for_each(|ch| put(self.b1[by * 2 + bx + ch * 4]));
            }
            2 => {
                (0..4).for_each(|ch| put(self.b1[by / 2 * 2 + bx / 2 + ch * 4]));
                (0..20).for_each(|ch| put(self.b2[by * 4 + bx + ch * 16]));
            }
            _ => {
                (0..20).for_each(|ch| put(self.b2[by / 2 * 4 + bx / 2 + ch * 16]));
                (0..20).for_each(|ch| put(self.b3[by * 8 + bx + ch * 64]));
            }
        }
        put(self.log_q);
        let layers: [(&[f32], &[f32]); 3] = match level {
            0 => [(&BRANCH_0_DNN_LAYER_0_KERNEL, &BRANCH_0_DNN_LAYER_0_BIAS), (&BRANCH_0_DNN_LAYER_1_KERNEL, &BRANCH_0_DNN_LAYER_1_BIAS), (&BRANCH_0_LOGITS_KERNEL, &BRANCH_0_LOGITS_BIAS)],
            1 => [(&BRANCH_1_DNN_LAYER_0_KERNEL, &BRANCH_1_DNN_LAYER_0_BIAS), (&BRANCH_1_DNN_LAYER_1_KERNEL, &BRANCH_1_DNN_LAYER_1_BIAS), (&BRANCH_1_LOGITS_KERNEL, &BRANCH_1_LOGITS_BIAS)],
            2 => [(&BRANCH_2_DNN_LAYER_0_KERNEL, &BRANCH_2_DNN_LAYER_0_BIAS), (&BRANCH_2_DNN_LAYER_1_KERNEL, &BRANCH_2_DNN_LAYER_1_BIAS), (&BRANCH_2_LOGITS_KERNEL, &BRANCH_2_LOGITS_BIAS)],
            _ => [(&BRANCH_3_DNN_LAYER_0_KERNEL, &BRANCH_3_DNN_LAYER_0_BIAS), (&BRANCH_3_DNN_LAYER_1_KERNEL, &BRANCH_3_DNN_LAYER_1_BIAS), (&BRANCH_3_LOGITS_KERNEL, &BRANCH_3_LOGITS_BIAS)],
        };
        dense(&f[..n], layers)
    }

    /// libaom's decision for the block from its logit, for a frame of `width`×`height`: above the
    /// split threshold only Split is searched (None still is at prune level 1, for screen
    /// content), below the no-split threshold Split is not.
    pub fn prune(&self, level: usize, bx: usize, by: usize, prune_level: u8, width: u32, height: u32) -> Prune {
        decide(self.logit(level, bx, by), level, prune_level, width, height)
    }
}

pub fn decide(logit: f32, level: usize, prune_level: u8, width: u32, height: u32) -> Prune {
    let (split, no_split) = match width.min(height) {
        720.. => (&SPLIT_THRESH_HDRES, &NO_SPLIT_THRESH_HDRES),
        480.. => (&SPLIT_THRESH_MIDRES, &NO_SPLIT_THRESH_MIDRES),
        _ => (&SPLIT_THRESH_LOWRES, &NO_SPLIT_THRESH_LOWRES),
    };
    let split_only = logit > split[level + 1];
    Prune { none: split_only && prune_level != 1, split: logit < no_split[level + 1], rect: split_only }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enc::frame::tests::{ppm, Rng};
    use std::io::{Read, Write};

    const OUTPUTS: usize = 20 + 16 + 320 + 1280;

    /// The level and position of libaom's `quad_tree_idx`.
    fn quad(q: usize) -> (usize, usize, usize) {
        if q == 0 {
            return (0, 0, 0);
        }
        let (l, x, y) = quad((q - 1) / 4);
        let c = (q - 1) % 4;
        (l + 1, x * 2 + (c & 1), y * 2 + (c >> 1))
    }

    fn blocks() -> Vec<(u8, [u8; SIDE * SIDE])> {
        let mut r = Rng(0x9e3779b97f4a7c15);
        let mut out = Vec::new();
        for i in 0..2400 {
            let mut px = [0u8; SIDE * SIDE];
            let (gx, gy, base, noise) = (r.below(9) as i32 - 4, r.below(9) as i32 - 4, r.below(256) as i32, 1 + r.below([256, 64, 8, 1][i % 4]) as i32);
            for (j, p) in px.iter_mut().enumerate() {
                let (x, y) = ((j % SIDE) as i32, (j / SIDE) as i32);
                *p = (base + (gx * x + gy * y) / 4 + r.below(noise as u32) as i32 - noise / 2).clamp(0, 255) as u8;
            }
            for _ in 0..r.below(6) {
                let (x0, y0, w, h, v) = (r.below(65) as usize, r.below(65) as usize, 1 + r.below(40) as usize, 1 + r.below(40) as usize, r.below(256) as u8);
                for y in y0..(y0 + h).min(SIDE) {
                    px[y * SIDE + x0..y * SIDE + (x0 + w).min(SIDE)].fill(v);
                }
            }
            out.push((r.below(256) as u8, px));
        }
        for q in 0..=255 {
            out.push((q, out[0].1));
        }
        if let Ok(dir) = std::env::var("ENC_CORPUS") {
            let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
            files.sort();
            for path in files {
                let (w, h, rgb) = ppm(&path);
                let luma: Vec<u8> = rgb
                    .chunks_exact(3)
                    .map(|p| ((19595 * p[0] as i32 + 38470 * p[1] as i32 + 7471 * p[2] as i32 + 32768) >> 16) as u8)
                    .collect();
                for y0 in (0..h.saturating_sub(63)).step_by(32) {
                    for x0 in (0..w.saturating_sub(63)).step_by(32) {
                        let mut px = [0u8; SIDE * SIDE];
                        for (j, p) in px.iter_mut().enumerate() {
                            let (x, y) = ((x0 + j % SIDE).saturating_sub(1), (y0 + j / SIDE).saturating_sub(1));
                            *p = luma[y.min(h - 1) * w + x.min(w - 1)];
                        }
                        out.push(([32, 140, 255, r.below(256) as u8][(x0 / 32 + y0 / 32) % 4], px));
                    }
                }
            }
        }
        out
    }

    /// `CNN_REF=<libaom's reference> [ENC_CORPUS=<dir of P6 PPMs>]`, run by
    /// `scripts/cnn-partition-compare.sh`: every output, logit and decision against libaom's C.
    #[test]
    #[ignore]
    fn matches_libaom() {
        let blocks = blocks();
        let mut child = std::process::Command::new(std::env::var("CNN_REF").unwrap())
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let input: Vec<u8> = blocks.iter().flat_map(|(q, px)| std::iter::once(*q).chain(px.iter().copied())).collect();
        let writer = std::thread::spawn(move || stdin.write_all(&input).unwrap());
        let mut data = Vec::new();
        child.stdout.take().unwrap().read_to_end(&mut data).unwrap();
        writer.join().unwrap();
        assert!(child.wait().unwrap().success());
        let record = (OUTPUTS + 1 + 85) * 4 + 85 * 3;
        assert_eq!(data.len(), blocks.len() * record);
        let (mut differ, mut max, mut decisions, mut pruned) = (0, 0f32, 0, [0; 3]);
        for ((qidx, px), rec) in blocks.iter().zip(data.chunks_exact(record)) {
            let reference: Vec<f32> = rec[..(OUTPUTS + 86) * 4].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            let cnn = Cnn::new(px, *qidx);
            let (outputs, log_q) = cnn.outputs();
            let mut ours: Vec<f32> = outputs;
            ours.push(log_q);
            ours.extend((0..85).map(|q| {
                let (l, x, y) = quad(q);
                cnn.logit(l, x, y)
            }));
            for (a, b) in ours.iter().zip(&reference) {
                if a.to_bits() != b.to_bits() {
                    differ += 1;
                    max = max.max((a - b).abs());
                }
            }
            for q in 0..85 {
                let (l, x, y) = quad(q);
                for (r, side) in [384, 480, 720].into_iter().enumerate() {
                    let flags = rec[(OUTPUTS + 86) * 4 + q * 3 + r];
                    let p = cnn.prune(l, x, y, 2, side, side);
                    decisions += (p.rect as u8 | (p.split as u8) << 1 != flags) as usize;
                    pruned[r] += (flags != 0) as usize;
                }
            }
        }
        println!("{} blocks, {} values: {differ} differ, max {max}; {} decisions differ; pruned (low/mid/hd res) {pruned:?} of {}", blocks.len(), blocks.len() * (OUTPUTS + 86), decisions, blocks.len() * 85);
        assert_eq!((differ, decisions), (0, 0));
    }
}
