//! Deblocking and CDEF strengths searched on the decoded picture, as libaom's full-image picks
//! search them at its default speeds: the frame header is written last, so the strengths can be
//! chosen once every tile is coded.

use crate::enc::dist;
use crate::enc::frame::{Encoder, Rd};
use crate::enc::header::{Cdef, Frame};
use crate::include::dav1d::headers::Rav1dFrameHeader;
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

/// The header a decoder reads from `fr`'s filter fields, written into `hdr`.
fn put_filters(fr: &Frame, mono: bool, cdef: bool, hdr: &mut Rav1dFrameHeader) {
    hdr.loopfilter.level_y = [fr.lf[0], fr.lf[1]];
    let chroma = !mono && fr.lf[..2] != [0, 0];
    hdr.loopfilter.level_u = if chroma { fr.lf[2] } else { 0 };
    hdr.loopfilter.level_v = if chroma { fr.lf[3] } else { 0 };
    if cdef {
        let c = fr.cdef.unwrap_or(Cdef { damping: 3, ..Default::default() });
        hdr.cdef.damping = c.damping;
        hdr.cdef.n_bits = c.bits;
        for i in 0..8 {
            let coded = i < 1 << c.bits;
            hdr.cdef.y_strength[i] = if coded { c.y[i] } else { 0 };
            hdr.cdef.uv_strength[i] = if coded && !mono { c.uv[i] } else { 0 };
        }
    }
}

/// Searches the strengths on one decode of the frame, filtered again for each candidate; a frame
/// the decoder cannot try so (a loop filter level of 0 to start from) is decoded per candidate.
pub fn search(e: &mut Encoder<Rd>, src: &[&[u8]]) {
    let (w, h) = (e.seq.width as usize, e.seq.height as usize);
    let (mono, cdef) = (e.seq.mono, e.seq.cdef);
    let mut s = Settings::new();
    s.set_inloop_filters(InloopFilterType::empty());
    s.set_n_threads(1);
    s.set_max_frame_delay(1);
    let mut d = Decoder::with_settings(&s).expect("a decoder");
    let mut measure = |p: usize, px: &[u8], stride: usize, pw: usize, ph: usize| {
        assert!((pw, ph) == (w, h));
        let mut sum = 0;
        for y in (0..h).step_by(64) {
            for x in (0..w).step_by(64) {
                let (tw, th) = ((w - x).min(64), (h - y).min(64));
                sum += dist::sse(&px[y * stride + x..], stride, &src[p][y * w + x..], w, tw, th);
            }
        }
        sum
    };
    // The decoder keeps the loop filter's masks only under a level; the tiles do not depend on it.
    let lf = e.frame.lf;
    if lf[..2] == [0, 0] {
        e.frame.lf = [1; 4];
    }
    let data = e.bytes().into_boxed_slice();
    e.frame.lf = lf;
    let mut tried = false;
    let _ = d.filter_trials(data, &mut measure, &mut |trial| {
        tried = true;
        run(e, src.len(), &mut |e| trial(&|hdr| put_filters(&e.frame, mono, cdef, hdr)));
    });
    if !tried {
        run(e, src.len(), &mut |e| plane_errors(e.bytes(), src, w, h));
    }
}

/// Searches each plane's deblocking level from the header's, in steps of 8 down to 1, then the
/// CDEF primary and secondary strengths of luma and chroma, keeping what lowers the error. Each
/// plane is filtered on its own, so one decode tries a level for every plane.
fn run(e: &mut Encoder<Rd>, planes: usize, errors: &mut dyn FnMut(&Encoder<Rd>) -> [u64; 3]) {
    let mut best = errors(e);
    for step in [8, 4, 2, 1] {
        for dir in [-1, 1] {
            let lf = e.frame.lf;
            let level = |l: u8| (l as i32 + dir * step).clamp(1, 63) as u8;
            e.frame.lf = [level(lf[0]), level(lf[0]), level(lf[2]), level(lf[3])];
            let err = errors(e);
            for (p, l) in [(0, [0, 1]), (1, [2, 2]), (2, [3, 3])] {
                if p < planes && err[p] < best[p] {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enc::frame::tests::Rng;
    use crate::enc::{still_with, Options};

    /// A trial's errors are those of a decode of the stream under the same filter header.
    #[test]
    fn trials_as_decodes() {
        let mut rng = Rng(0xf17e5);
        for (w, h, planes, qidx) in [(96, 72, 3, 60), (130, 66, 1, 140), (200, 140, 3, 200), (64, 64, 3, 20), (33, 17, 3, 100)] {
            let px: Vec<Vec<u8>> = (0..planes)
                .map(|p| (0..w * h).map(|i| ((i % w) * 3 + (i / w) * 2 + p * 40 + rng.below(40) as usize) as u8).collect())
                .collect();
            let src: Vec<&[u8]> = px.iter().map(|p| &p[..]).collect();
            let mut e = still_with(w as u32, h as u32, &src, &Options::effort(qidx, 4, false, (w * h) as u64));
            e.flush();
            let (mono, cdef) = (e.seq.mono, e.seq.cdef);
            let base = e.frame.cdef.unwrap_or(Cdef { damping: 3, ..Default::default() });
            let cands: Vec<([u8; 4], Cdef)> = (0..10)
                .map(|k| {
                    let lf = if k == 1 { [0; 4] } else { [0; 4].map(|_| rng.below(64) as u8) };
                    let mut c = Cdef { damping: 3 + rng.below(4) as u8, ..base };
                    for i in 0..8 {
                        (c.y[i], c.uv[i]) = (rng.below(64) as u8, rng.below(64) as u8);
                    }
                    (lf, c)
                })
                .collect();
            let lf = e.frame.lf;
            e.frame.lf = [1; 4];
            let data = e.bytes().into_boxed_slice();
            e.frame.lf = lf;
            let mut s = Settings::new();
            s.set_inloop_filters(InloopFilterType::empty());
            s.set_n_threads(1);
            s.set_max_frame_delay(1);
            let mut d = Decoder::with_settings(&s).unwrap();
            let mut measure = |p: usize, px: &[u8], stride: usize, _: usize, _: usize| {
                let at = (0..h).flat_map(|y| (0..w).step_by(64).map(move |x| (y, x)));
                at.map(|(y, x)| dist::sse(&px[y * stride + x..], stride, &src[p][y * w + x..], w, (w - x).min(64), 1)).sum()
            };
            let mut runs = 0;
            d.filter_trials(data, &mut measure, &mut |trial| {
                for &(lf, c) in &cands {
                    e.frame.lf = lf;
                    e.frame.cdef = Some(c);
                    let got = trial(&|hdr| put_filters(&e.frame, mono, cdef, hdr));
                    assert_eq!(got, plane_errors(e.bytes(), &src, w, h), "{w}x{h} q{qidx} lf {lf:?} cdef {c:?}");
                    runs += 1;
                }
            })
            .unwrap();
            assert_eq!(runs, cands.len());
        }
    }
}
