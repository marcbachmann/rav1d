//! `ivfhash <file.ivf>...`: decodes each file on one thread through the Rust API and prints an FNV-1a hash
//! of every visible plane row of every picture, its picture count and its name. CI compares the lines of a
//! native build with those of a wasm32-wasip1 build run in wasmtime.

use rav1d::{Decoder, Picture, PixelLayout, PlanarImageComponent, Rav1dError, Settings};

fn ivf_frames(data: &[u8]) -> Vec<&[u8]> {
    let mut frames = Vec::new();
    if data.len() < 32 || &data[..4] != b"DKIF" {
        return frames;
    }
    let mut at = u16::from_le_bytes([data[6], data[7]]) as usize;
    while at + 12 <= data.len() {
        let size = u32::from_le_bytes(data[at..at + 4].try_into().unwrap()) as usize;
        at += 12;
        if size == 0 || at + size > data.len() {
            break;
        }
        frames.push(&data[at..at + size]);
        at += size;
    }
    frames
}

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h = (*h ^ b as u64).wrapping_mul(0x100000001b3);
    }
}

fn hash_picture(h: &mut u64, p: &Picture) {
    let (w, ht) = (p.width() as usize, p.height() as usize);
    fnv(h, &(w as u32).to_le_bytes());
    fnv(h, &(ht as u32).to_le_bytes());
    let sample = if p.bit_depth() > 8 { 2 } else { 1 };
    for (i, c) in [
        PlanarImageComponent::Y,
        PlanarImageComponent::U,
        PlanarImageComponent::V,
    ]
    .into_iter()
    .enumerate()
    {
        let (sx, sy) = match p.pixel_layout() {
            PixelLayout::I400 if i > 0 => continue,
            PixelLayout::I420 if i > 0 => (1, 1),
            PixelLayout::I422 if i > 0 => (1, 0),
            _ => (0, 0),
        };
        let (bytes, rows, stride, data) = (
            (w + sx >> sx) * sample,
            ht + sy >> sy,
            p.stride(c) as usize,
            p.plane_ptr(c),
        );
        for y in 0..rows {
            fnv(h, unsafe {
                std::slice::from_raw_parts(data.add(stride * y), bytes)
            });
        }
    }
}

fn drain(decoder: &mut Decoder, h: &mut u64, n: &mut usize) {
    loop {
        match decoder.get_picture() {
            Ok(p) => {
                hash_picture(h, &p);
                *n += 1;
            }
            Err(Rav1dError::TryAgain) => break,
            Err(e) => {
                fnv(h, format!("{e:?}").as_bytes());
                break;
            }
        }
    }
}

fn main() {
    for file in std::env::args().skip(1) {
        let data = std::fs::read(&file).unwrap();
        let mut settings = Settings::new();
        settings.set_n_threads(1);
        settings.set_max_frame_delay(1);
        let mut decoder = Decoder::with_settings(&settings).unwrap();
        let (mut h, mut n) = (0xcbf29ce484222325u64, 0);
        for frame in ivf_frames(&data) {
            let mut sent = decoder.send_data(frame.into(), None, None, None);
            loop {
                drain(&mut decoder, &mut h, &mut n);
                match sent {
                    Err(Rav1dError::TryAgain) => sent = decoder.send_pending_data(),
                    Err(e) => {
                        fnv(&mut h, format!("{e:?}").as_bytes());
                        break;
                    }
                    Ok(()) => break,
                }
            }
        }
        drain(&mut decoder, &mut h, &mut n);
        println!("{h:016x} {n} {}", file.rsplit('/').next().unwrap());
    }
}
