//! An AV1 still-image encoder, see `doc/encoder.md`.

pub mod ec;
pub mod fdct;
pub mod frame;
pub mod header;
pub mod rd;

use frame::{Coded, Encoder, Rd};
use header::{Colour, Frame, Quant, Sequence, Tiling};

/// A still picture of 8-bit planes coded at `qidx` (0 is lossless): one plane is monochrome,
/// three are 4:4:4 YCbCr, BT.601 full range, sRGB.
pub fn still(width: u32, height: u32, planes: &[&[u8]], qidx: u8) -> Coded {
    let mono = planes.len() == 1;
    let seq = Sequence {
        width,
        height,
        mono,
        colour: (!mono).then_some(Colour { primaries: 1, transfer: 13, matrix: 6 }),
        full_range: true,
        filter_intra: true,
        edge_filter: true,
        cdef: false,
        restoration: false,
    };
    let frame = Frame {
        screen_content: false,
        quant: Quant { base: qidx, ..Default::default() },
        delta_q_log2: None,
        lf: [0; 4],
        sharpness: 0,
        cdef: None,
        lr: None,
        tx_select: qidx != 0,
        reduced_tx_set: false,
        tiling: Tiling::new(width, height),
    };
    let mut e = Encoder::new(seq, frame, Rd);
    let w = width as usize;
    for y in 0..height as usize {
        let rows: Vec<&[u8]> = planes.iter().map(|p| &p[y * w..][..w]).collect();
        e.push(&rows);
    }
    e.finish()
}
