//! Pictures decoded with every new allocation filled with one byte and then another: a picture that
//! changes read memory nothing wrote.

use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::c_int;
use std::hash::{DefaultHasher, Hasher};
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU8, Ordering};

use rav1d::include::dav1d::data::Dav1dData;
use rav1d::include::dav1d::dav1d::{Dav1dContext, Dav1dSettings};
use rav1d::include::dav1d::headers::{DAV1D_PIXEL_LAYOUT_I400, DAV1D_PIXEL_LAYOUT_I420, DAV1D_PIXEL_LAYOUT_I444};
use rav1d::include::dav1d::picture::Dav1dPicture;
use rav1d::{
    dav1d_close, dav1d_data_create, dav1d_data_unref, dav1d_decode_rows, dav1d_default_settings,
    dav1d_get_picture, dav1d_open, dav1d_picture_unref, dav1d_send_data, dav1d_set_decode_by_rows,
};

static FILL: AtomicU8 = AtomicU8::new(0);

struct Filling;

unsafe impl GlobalAlloc for Filling {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            p.write_bytes(FILL.load(Ordering::Relaxed), layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        System.dealloc(p, layout)
    }

    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let q = System.realloc(p, layout, size);
        if !q.is_null() && size > layout.size() {
            q.add(layout.size()).write_bytes(FILL.load(Ordering::Relaxed), size - layout.size());
        }
        q
    }
}

#[global_allocator]
static GLOBAL: Filling = Filling;

#[cfg(target_os = "macos")]
const EAGAIN: i32 = -libc::EAGAIN;
#[cfg(not(target_os = "macos"))]
const EAGAIN: i32 = -11;

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

fn digest(p: &Dav1dPicture) -> u64 {
    let mut h = DefaultHasher::new();
    let (w, hgt) = (p.p.w as usize, p.p.h as usize);
    let sample = if p.p.bpc > 8 { 2 } else { 1 };
    for i in 0..3 {
        let (sx, sy) = match p.p.layout {
            DAV1D_PIXEL_LAYOUT_I400 if i > 0 => continue,
            DAV1D_PIXEL_LAYOUT_I420 if i > 0 => (1, 1),
            DAV1D_PIXEL_LAYOUT_I444 => (0, 0),
            _ if i > 0 => (1, 0),
            _ => (0, 0),
        };
        let Some(data) = p.data[i] else { continue };
        let stride = p.stride[i.min(1)];
        for y in 0..(hgt + sy) >> sy {
            let row = unsafe {
                std::slice::from_raw_parts(
                    (data.as_ptr() as *const u8).offset(stride * y as isize),
                    ((w + sx) >> sx) * sample,
                )
            };
            h.write(row);
        }
    }
    h.finish()
}

fn decode(file: &Path, threads: c_int, by_rows: bool) -> Vec<Result<u64, i32>> {
    let data = std::fs::read(file).unwrap();
    let mut out = Vec::new();
    unsafe {
        let mut settings = MaybeUninit::<Dav1dSettings>::uninit();
        dav1d_default_settings(NonNull::new(settings.as_mut_ptr()).unwrap());
        let mut settings = settings.assume_init();
        settings.n_threads = threads;
        settings.max_frame_delay = 1;
        let mut ctx: Option<Dav1dContext> = None;
        assert_eq!(dav1d_open(NonNull::new(&mut ctx), NonNull::new(&mut settings)).0, 0);
        let ctx = ctx.unwrap();
        dav1d_set_decode_by_rows(ctx, by_rows);
        let drain = |out: &mut Vec<Result<u64, i32>>| loop {
            let mut n = 0;
            while dav1d_decode_rows(ctx, &mut n).0 == 0 {}
            let mut p = Dav1dPicture::default();
            match dav1d_get_picture(Some(ctx), NonNull::new(&mut p)).0 {
                0 => {
                    out.push(Ok(digest(&p)));
                    dav1d_picture_unref(NonNull::new(&mut p));
                }
                EAGAIN => {
                    let mut n = 0;
                    if dav1d_decode_rows(ctx, &mut n).0 != 0 {
                        break;
                    }
                }
                e => out.push(Err(e)),
            }
        };
        for frame in ivf_frames(&data) {
            let mut d = Dav1dData::default();
            let buf = dav1d_data_create(NonNull::new(&mut d), frame.len());
            std::ptr::copy_nonoverlapping(frame.as_ptr(), buf, frame.len());
            for _ in 0..1000 {
                let res = dav1d_send_data(Some(ctx), NonNull::new(&mut d)).0;
                if res != 0 && res != EAGAIN {
                    out.push(Err(res));
                    break;
                }
                drain(&mut out);
                if d.sz == 0 {
                    break;
                }
            }
            if d.sz > 0 {
                dav1d_data_unref(NonNull::new(&mut d));
            }
        }
        drain(&mut out);
        dav1d_close(NonNull::new(&mut Some(ctx)));
    }
    out
}

fn streams() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "ivf") {
                out.push(p);
            }
        }
    }
    let mut out = Vec::new();
    for bits in ["8-bit", "10-bit", "12-bit", "multi-bit", "oss-fuzz"] {
        walk(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/dav1d-test-data").join(bits), &mut out);
    }
    out.sort();
    out
}

/// One test, as the fill is the process's: one thread whole and by rows, and four threads. Six
/// minutes over the test data: `cargo test --test uninit -- --ignored`.
#[test]
#[ignore]
fn reads_no_memory_nothing_wrote() {
    let filter = std::env::var("ROWS_FILTER").unwrap_or_default();
    let mut differ = Vec::new();
    let mut pictures = 0;
    for file in streams().iter().filter(|f| f.to_string_lossy().contains(&filter)) {
        for (threads, by_rows) in [(1, false), (1, true), (4, false)] {
            FILL.store(0, Ordering::Relaxed);
            let zero = decode(file, threads, by_rows);
            FILL.store(0xff, Ordering::Relaxed);
            let ones = decode(file, threads, by_rows);
            let upto = |d: &Vec<Result<u64, i32>>| d.iter().position(|o| o.is_err()).unwrap_or(d.len());
            let n = upto(&zero).min(upto(&ones));
            if zero[..n] != ones[..n] {
                differ.push(format!("{} threads {threads} by rows {by_rows}", file.display()));
            }
            pictures += n;
        }
    }
    eprintln!("{pictures} pictures, {} differ: {differ:#?}", differ.len());
    assert!(differ.is_empty());
}
