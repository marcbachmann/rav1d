//! Deblocking and CDEF strengths searched on the decoded picture, as libaom's full-image picks
//! search them at its default speeds: the frame header is written last, so the strengths can be
//! chosen once every tile is coded.

use crate::enc::frame::{Encoder, Rd};
use crate::enc::header::Cdef;
use crate::{Decoder, InloopFilterType, PlanarImageComponent, Settings};

/// The squared error of each plane of the decoded frame against `src`.
fn plane_errors(data: Vec<u8>, src: &[&[u8]], w: usize, h: usize) -> [u64; 3] {
    let mut s = Settings::new();
    s.set_inloop_filters(InloopFilterType::DEBLOCK | InloopFilterType::CDEF);
    s.set_n_threads(1);
    s.set_max_frame_delay(1);
    let mut d = Decoder::with_settings(&s).expect("a decoder");
    d.send_data(data.into_boxed_slice(), None, None, None).expect("its own stream");
    let pic = d.get_picture().or_else(|_| d.get_picture()).expect("a picture");
    let comps = [PlanarImageComponent::Y, PlanarImageComponent::U, PlanarImageComponent::V];
    let mut out = [0; 3];
    for (p, &c) in comps[..src.len()].iter().enumerate() {
        let (plane, stride) = (pic.plane(c), pic.stride(c) as usize);
        for y in 0..h {
            for (&a, &b) in plane[y * stride..][..w].iter().zip(&src[p][y * w..][..w]) {
                let d = a as i64 - b as i64;
                out[p] += (d * d) as u64;
            }
        }
    }
    out
}

/// Searches each plane's deblocking level from the header's, in steps of 8 down to 1, then the
/// CDEF primary and secondary strengths of luma and chroma, keeping what lowers the error. Each
/// plane is filtered on its own, so one decode tries a level for every plane.
pub fn search(e: &mut Encoder<Rd>, src: &[&[u8]]) {
    let (w, h) = (e.seq.width as usize, e.seq.height as usize);
    let errors = |e: &Encoder<Rd>| plane_errors(e.bytes(), src, w, h);
    let mut best = errors(e);
    for step in [8, 4, 2, 1] {
        for dir in [-1, 1] {
            let lf = e.frame.lf;
            let level = |l: u8| (l as i32 + dir * step).clamp(1, 63) as u8;
            e.frame.lf = [level(lf[0]), level(lf[0]), level(lf[2]), level(lf[3])];
            let err = errors(e);
            for (p, l) in [(0, [0, 1]), (1, [2, 2]), (2, [3, 3])] {
                if p < src.len() && err[p] < best[p] {
                    best[p] = err[p];
                } else {
                    e.frame.lf[l[0]] = lf[l[0]];
                    e.frame.lf[l[1]] = lf[l[1]];
                }
            }
        }
    }
    let lf = e.frame.lf;
    e.frame.lf = [0; 4];
    let off = errors(e);
    if off.iter().sum::<u64>() < best.iter().sum::<u64>() {
        best = off;
    } else {
        e.frame.lf = lf;
    }
    if !e.seq.cdef {
        return;
    }
    let mut cdef = e.frame.cdef.unwrap_or(Cdef { damping: 3, ..Default::default() });
    for primary in [true, false] {
        let values: &[u8] = if primary { &[0, 1, 2, 3, 4, 6, 8, 10, 12, 15] } else { &[0, 1, 2, 3] };
        let set = |s: u8, v: u8| if primary { v << 2 | s & 3 } else { s & !3 | v };
        for &v in values {
            let cand = Cdef { y: [set(cdef.y[0], v); 8], uv: [set(cdef.uv[0], v); 8], ..cdef };
            e.frame.cdef = Some(cand);
            let err = errors(e);
            if err[0] < best[0] {
                cdef.y[0] = cand.y[0];
                best[0] = err[0];
            }
            if err[1] + err[2] < best[1] + best[2] {
                cdef.uv[0] = cand.uv[0];
                (best[1], best[2]) = (err[1], err[2]);
            }
        }
    }
    e.frame.cdef = Some(cdef);
}
