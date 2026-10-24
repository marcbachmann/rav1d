//! Palette blocks: candidate palettes, colour index maps and their syntax, as `read_pal_plane`,
//! `read_pal_uv` and `read_pal_indices` read them.

use std::cmp;

use crate::decode::order_palette;
use crate::enc::rd::Sink;

/// The colours above and left that a block's palette may reuse, sorted and without repeats, as
/// the decoder merges them.
pub fn cache(above: &[u8], left: &[u8]) -> ([u8; 16], usize) {
    let mut out = [0u8; 16];
    let mut n = 0;
    let (mut a, mut l) = (above, left);
    let mut push = |v: u8, out: &mut [u8; 16]| {
        if n == 0 || out[n - 1] != v {
            out[n] = v;
            n += 1;
        }
    };
    while !l.is_empty() && !a.is_empty() {
        if l[0] < a[0] {
            push(l[0], &mut out);
            l = &l[1..];
        } else {
            if a[0] == l[0] {
                l = &l[1..];
            }
            push(a[0], &mut out);
            a = &a[1..];
        }
    }
    for &v in l.iter().chain(a) {
        push(v, &mut out);
    }
    (out, n)
}

fn bits_for(v: u32) -> u8 {
    (32 - v.leading_zeros()) as u8
}

/// The colours of a luma or U palette, sorted: which cache entries it reuses, then the rest as
/// deltas from the one before.
pub fn write_colors<S: Sink>(s: &mut S, pal: &[u8], cache: &[u8], pl: bool) {
    let not_pl = !pl as u32;
    let mut used = 0;
    let mut reused = [false; 8];
    for &c in cache {
        if used == pal.len() {
            break;
        }
        let hit = pal.iter().position(|&p| p == c).filter(|&i| !reused[i]);
        s.bool_equi(hit.is_some());
        if let Some(i) = hit {
            reused[i] = true;
            used += 1;
        }
    }
    let new: Vec<u32> = pal.iter().zip(reused).filter(|(_, r)| !r).map(|(&c, _)| c as u32).collect();
    let Some(&first) = new.first() else { return };
    s.bools(first, 8);
    if new.len() == 1 {
        return;
    }
    let max_delta = new.windows(2).map(|w| w[1] - w[0] - not_pl).max().unwrap();
    let min_bits = 8 - 3;
    let mut bits = cmp::max(bits_for(max_delta), min_bits);
    s.bools((bits - min_bits) as u32, 2);
    let mut prev = first;
    for &c in &new[1..] {
        s.bools(c - prev - not_pl, bits);
        prev = c;
        if prev + not_pl >= 255 {
            break;
        }
        bits = cmp::min(bits, bits_for(255 - prev - not_pl));
    }
}

/// The V colours of a chroma palette, in the order of the U colours: raw, or as signed deltas
/// when that is shorter.
pub fn write_v<S: Sink>(s: &mut S, v: &[u8]) {
    let deltas: Vec<i32> = v
        .windows(2)
        .map(|w| {
            let d = w[1] as i32 - w[0] as i32;
            if d > 128 {
                d - 256
            } else if d < -128 {
                d + 256
            } else {
                d
            }
        })
        .collect();
    let min_bits = 8 - 4;
    let bits = cmp::max(bits_for(deltas.iter().map(|d| d.unsigned_abs()).max().unwrap_or(0)), min_bits);
    let delta_bits = 2 + 8 + deltas.iter().map(|&d| bits as usize + (d != 0) as usize).sum::<usize>();
    if bits <= min_bits + 3 && delta_bits < 8 * v.len() {
        s.bool_equi(true);
        s.bools((bits - min_bits) as u32, 2);
        s.bools(v[0] as u32, 8);
        for &d in &deltas {
            s.bools(d.unsigned_abs(), bits);
            if d != 0 {
                s.bool_equi(d < 0);
            }
        }
    } else {
        s.bool_equi(false);
        for &c in v {
            s.bools(c as u32, 8);
        }
    }
}

/// A block's colour indexes, `w`×`h` visible of a map `stride` wide, in wavefront order.
pub fn write_indices<S: Sink>(s: &mut S, cdf: &mut [[u16; 8]; 5], map: &[u8], stride: usize, w: usize, h: usize, n: u8) {
    s.uniform(map[0] as u32, n as u32);
    let mut order = [[0u8; 8]; 64];
    let mut ctx = [0u8; 64];
    for i in 1..w + h - 1 {
        let first = cmp::min(i, w - 1);
        let last = (i + 1).saturating_sub(h);
        order_palette(map, stride, i, first, last, &mut order, &mut ctx);
        for (m, j) in (last..=first).rev().enumerate() {
            let idx = map[(i - j) * stride + j];
            let sym = order[m].iter().position(|&o| o == idx).unwrap();
            s.symbol(sym as u8, &mut cdf[ctx[m] as usize], n - 1);
        }
    }
}

/// The index of the nearest colour, the first of equals.
pub fn nearest(pal: &[u8], v: u8) -> u8 {
    let mut best = (u32::MAX, 0);
    for (i, &p) in pal.iter().enumerate() {
        let d = (p as i32 - v as i32).unsigned_abs();
        if d < best.0 {
            best = (d, i as u8);
        }
    }
    best.1
}

pub fn nearest2(u: &[u8], v: &[u8], (pu, pv): (u8, u8)) -> u8 {
    let mut best = (u32::MAX, 0);
    for i in 0..u.len() {
        let du = u[i] as i32 - pu as i32;
        let dv = v[i] as i32 - pv as i32;
        let d = (du * du + dv * dv) as u32;
        if d < best.0 {
            best = (d, i as u8);
        }
    }
    best.1
}

/// Luma palettes worth trying for these pixels: their colours when there are 2 to 8, else
/// k-means palettes of 8 and 4 colours when there are no more than 64.
pub fn candidates_y(px: &[u8]) -> Vec<Vec<u8>> {
    let mut hist = [0u32; 256];
    for &p in px {
        hist[p as usize] += 1;
    }
    let colors: Vec<u8> = (0..=255u8).filter(|&c| hist[c as usize] > 0).collect();
    match colors.len() {
        0 | 1 => vec![],
        2..=8 => vec![colors],
        9..=64 => [8, 4].iter().map(|&k| kmeans1(&hist, &colors, k)).collect(),
        _ => vec![],
    }
}

fn kmeans1(hist: &[u32; 256], colors: &[u8], k: usize) -> Vec<u8> {
    let mut by_count = colors.to_vec();
    by_count.sort_by_key(|&c| cmp::Reverse(hist[c as usize]));
    let mut centres: Vec<f64> = by_count[..k].iter().map(|&c| c as f64).collect();
    for _ in 0..8 {
        let mut sum = vec![(0f64, 0f64); k];
        for &c in colors {
            let i = (0..k).min_by(|&a, &b| (centres[a] - c as f64).abs().total_cmp(&(centres[b] - c as f64).abs())).unwrap();
            let w = hist[c as usize] as f64;
            sum[i].0 += w * c as f64;
            sum[i].1 += w;
        }
        for (c, s) in centres.iter_mut().zip(&sum) {
            if s.1 > 0.0 {
                *c = s.0 / s.1;
            }
        }
    }
    let mut out: Vec<u8> = centres.iter().map(|c| c.round() as u8).collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Chroma palettes (U colours sorted, V alongside) worth trying for these pixel pairs. A block
/// with more than 64 pairs has none, which is known after the 65th distinct pair.
pub fn candidates_uv(u: &[u8], v: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut seen = [0u64; 1024];
    let mut keys = Vec::with_capacity(65);
    for (&a, &b) in u.iter().zip(v) {
        let k = (a as usize) << 8 | b as usize;
        if seen[k >> 6] & 1 << (k & 63) == 0 {
            seen[k >> 6] |= 1 << (k & 63);
            keys.push(k as u16);
            if keys.len() > 64 {
                return vec![];
            }
        }
    }
    keys.sort_unstable();
    let mut counts: Vec<((u8, u8), u32)> = keys.iter().map(|&k| (((k >> 8) as u8, k as u8), 0)).collect();
    for (&a, &b) in u.iter().zip(v) {
        let k = (a as u16) << 8 | b as u16;
        counts[keys.binary_search(&k).unwrap()].1 += 1;
    }
    let split = |c: &[(u8, u8)]| (c.iter().map(|p| p.0).collect(), c.iter().map(|p| p.1).collect());
    match counts.len() {
        0 | 1 => vec![],
        2..=8 => vec![split(&counts.iter().map(|c| c.0).collect::<Vec<_>>())],
        _ => [8, 4].iter().map(|&k| split(&kmeans2(&counts, k))).collect(),
    }
}

fn kmeans2(counts: &[((u8, u8), u32)], k: usize) -> Vec<(u8, u8)> {
    let mut by_count = counts.to_vec();
    by_count.sort_by_key(|c| cmp::Reverse(c.1));
    let mut centres: Vec<(f64, f64)> = by_count[..k].iter().map(|c| (c.0 .0 as f64, c.0 .1 as f64)).collect();
    let dist = |a: (f64, f64), b: (u8, u8)| (a.0 - b.0 as f64).powi(2) + (a.1 - b.1 as f64).powi(2);
    for _ in 0..8 {
        let mut sum = vec![(0f64, 0f64, 0f64); k];
        for &(p, n) in counts {
            let i = (0..k).min_by(|&a, &b| dist(centres[a], p).total_cmp(&dist(centres[b], p))).unwrap();
            let w = n as f64;
            sum[i].0 += w * p.0 as f64;
            sum[i].1 += w * p.1 as f64;
            sum[i].2 += w;
        }
        for (c, s) in centres.iter_mut().zip(&sum) {
            if s.2 > 0.0 {
                *c = (s.0 / s.2, s.1 / s.2);
            }
        }
    }
    let mut out: Vec<(u8, u8)> = centres.iter().map(|c| (c.0.round() as u8, c.1.round() as u8)).collect();
    out.sort_unstable();
    out.dedup();
    out
}
