//! Frames decoded by rows against the same frames decoded whole, over the test data's IVF streams.

use std::ffi::{c_int, c_void};
use std::hash::{DefaultHasher, Hasher};
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};

use rav1d::include::dav1d::data::Dav1dData;
use rav1d::include::dav1d::dav1d::{Dav1dContext, Dav1dSettings};
use rav1d::include::dav1d::headers::{DAV1D_PIXEL_LAYOUT_I400, DAV1D_PIXEL_LAYOUT_I420, DAV1D_PIXEL_LAYOUT_I444};
use rav1d::include::dav1d::picture::Dav1dPicture;
use rav1d::send_sync_non_null::SendSyncNonNull;
use rav1d::{
    dav1d_close, dav1d_data_create, dav1d_data_unref, dav1d_data_wrap, dav1d_decode_rows,
    dav1d_default_settings, dav1d_get_picture, dav1d_open, dav1d_picture_unref, dav1d_rows_picture,
    dav1d_send_data, dav1d_set_decode_by_rows,
};

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

fn open(apply_grain: bool, by_rows: bool) -> Dav1dContext {
    unsafe {
        let mut settings = MaybeUninit::<Dav1dSettings>::uninit();
        dav1d_default_settings(NonNull::new(settings.as_mut_ptr()).unwrap());
        let mut settings = settings.assume_init();
        settings.n_threads = 1;
        settings.max_frame_delay = 1;
        settings.apply_grain = apply_grain as c_int;
        let mut ctx: Option<Dav1dContext> = None;
        assert_eq!(dav1d_open(NonNull::new(&mut ctx), NonNull::new(&mut settings)).0, 0);
        let ctx = ctx.unwrap();
        dav1d_set_decode_by_rows(ctx, by_rows);
        ctx
    }
}

/// A plane's visible rows: bytes a row, rows, stride, data.
fn plane(p: &Dav1dPicture, i: usize) -> Option<(usize, usize, isize, *const u8)> {
    let (w, h) = (p.p.w as usize, p.p.h as usize);
    let (ss_hor, ss_ver) = match p.p.layout {
        DAV1D_PIXEL_LAYOUT_I400 if i > 0 => return None,
        DAV1D_PIXEL_LAYOUT_I420 if i > 0 => (1, 1),
        DAV1D_PIXEL_LAYOUT_I444 => (0, 0),
        _ if i > 0 => (1, 0),
        _ => (0, 0),
    };
    let sample = if p.p.bpc > 8 { 2 } else { 1 };
    Some((
        (w + ss_hor >> ss_hor) * sample,
        h + ss_ver >> ss_ver,
        p.stride[i.min(1)],
        p.data[i]?.as_ptr() as *const u8,
    ))
}

fn row<'a>((bytes, _, stride, data): (usize, usize, isize, *const u8), y: usize) -> &'a [u8] {
    unsafe { std::slice::from_raw_parts(data.offset(stride * y as isize), bytes) }
}

fn shift(p: &Dav1dPicture, i: usize) -> u32 {
    (i > 0 && p.p.layout == DAV1D_PIXEL_LAYOUT_I420) as u32
}

fn digest(p: &Dav1dPicture) -> u64 {
    let mut h = DefaultHasher::new();
    h.write_i32(p.p.w);
    h.write_i32(p.p.h);
    for i in 0..3 {
        if let Some(pl) = plane(p, i) {
            for y in 0..pl.1 {
                h.write(row(pl, y));
            }
        }
    }
    h.finish()
}

/// What a decode gave: a digest a picture or the error, in order, and how many frames went by rows.
#[derive(Debug, Default, PartialEq)]
struct Decoded {
    out: Vec<Result<u64, i32>>,
    by_rows: usize,
    /// Overwrite each row once it is final, and stop at the first picture: its digest is then of
    /// the rows as they were when final.
    poison: bool,
}

/// The frame being decoded by rows: its picture, and its rows copied as each became final.
struct Rows {
    picture: Dav1dPicture,
    copied: Vec<Vec<Vec<u8>>>,
    done: usize,
}

impl Rows {
    unsafe fn copy_to(&mut self, rows: usize) {
        for i in 0..3 {
            if let Some(pl) = plane(&self.picture, i) {
                let s = shift(&self.picture, i);
                let (from, to) = ((self.done + s as usize) >> s, (rows + s as usize) >> s);
                for y in from..to.min(pl.1) {
                    self.copied[i].push(row(pl, y).to_vec());
                }
            }
        }
        self.done = rows;
    }

    unsafe fn poison(&self) {
        for i in 0..3 {
            if let Some(pl) = plane(&self.picture, i) {
                for y in 0..self.copied[i].len() {
                    std::ptr::write_bytes(pl.3.offset(pl.2 * y as isize) as *mut u8, 0xa5, pl.0);
                }
            }
        }
    }

    fn digest(&self) -> u64 {
        let mut h = DefaultHasher::new();
        h.write_i32(self.picture.p.w);
        h.write_i32(self.picture.p.h);
        for plane in &self.copied {
            for row in plane {
                h.write(row);
            }
        }
        h.finish()
    }
}

/// Decodes the frame set up by rows to its end, checking that the rows it called final stay so.
unsafe fn by_rows(ctx: Dav1dContext, decoded: &mut Decoded) -> bool {
    let mut picture = Dav1dPicture::default();
    if dav1d_rows_picture(ctx, NonNull::new(&mut picture).unwrap()).0 != 0 {
        return false;
    }
    let mut rows = Rows {
        picture,
        copied: vec![Vec::new(); 3],
        done: 0,
    };
    loop {
        let mut n: c_int = -1;
        let res = dav1d_decode_rows(ctx, &mut n).0;
        if res != 0 {
            assert_ne!(res, EAGAIN, "rows of a frame left undecoded");
            decoded.out.push(Err(res));
            break;
        }
        assert!(n as usize >= rows.done && n <= rows.picture.p.h, "rows {n} after {}", rows.done);
        rows.copy_to(n as usize);
        if decoded.poison {
            rows.poison();
            if n == rows.picture.p.h {
                decoded.out.push(Ok(rows.digest()));
                decoded.by_rows += 1;
                break;
            }
            continue;
        }
        if n == rows.picture.p.h {
            assert_eq!(dav1d_decode_rows(ctx, &mut n).0, EAGAIN, "rows past the frame's end");
            for i in 0..3 {
                if let Some(pl) = plane(&rows.picture, i) {
                    assert_eq!(rows.copied[i].len(), pl.1);
                    for (y, copy) in rows.copied[i].iter().enumerate() {
                        assert!(row(pl, y) == copy.as_slice(), "plane {i} row {y} changed after it was final");
                    }
                }
            }
            decoded.by_rows += 1;
            break;
        }
    }
    dav1d_picture_unref(NonNull::new(&mut rows.picture));
    true
}

unsafe fn drain(ctx: Dav1dContext, rows: bool, decoded: &mut Decoded) {
    loop {
        let stepped = rows && by_rows(ctx, decoded);
        let mut p = Dav1dPicture::default();
        match dav1d_get_picture(Some(ctx), NonNull::new(&mut p)).0 {
            0 => {
                if !(decoded.poison && stepped) {
                    decoded.out.push(Ok(digest(&p)));
                }
                dav1d_picture_unref(NonNull::new(&mut p));
            }
            // get_picture may have set the next frame up by rows
            EAGAIN if stepped || rows && by_rows(ctx, decoded) => {}
            EAGAIN => break,
            e => decoded.out.push(Err(e)),
        }
    }
}

fn decode(file: &Path, apply_grain: bool, rows: bool, poison: bool) -> Decoded {
    let data = std::fs::read(file).unwrap();
    let mut decoded = Decoded {
        poison,
        ..Default::default()
    };
    let ctx = open(apply_grain, rows);
    unsafe {
        for frame in ivf_frames(&data) {
            if poison && !decoded.out.is_empty() {
                break;
            }
            let mut d = Dav1dData::default();
            let buf = dav1d_data_create(NonNull::new(&mut d), frame.len());
            std::ptr::copy_nonoverlapping(frame.as_ptr(), buf, frame.len());
            for _ in 0..1000 {
                let res = dav1d_send_data(Some(ctx), NonNull::new(&mut d)).0;
                if res != 0 && res != EAGAIN {
                    decoded.out.push(Err(res));
                    break;
                }
                drain(ctx, rows, &mut decoded);
                if d.sz == 0 {
                    break;
                }
            }
            if d.sz > 0 {
                dav1d_data_unref(NonNull::new(&mut d));
            }
        }
        drain(ctx, rows, &mut decoded);
        dav1d_close(NonNull::new(&mut Some(ctx)));
    }
    decoded
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

#[test]
fn decodes_by_rows_as_whole() {
    let files = streams();
    assert!(files.len() > 700, "{} streams", files.len());
    let filter = std::env::var("ROWS_FILTER").unwrap_or_default();
    let (mut pictures, mut frames_by_rows, mut errors) = (0, 0, 0);
    for file in files.iter().filter(|f| f.to_string_lossy().contains(&filter)) {
        for apply_grain in [true, false] {
            let whole = decode(file, apply_grain, false, false);
            let rows = decode(file, apply_grain, true, false);
            assert_eq!(whole.by_rows, 0);
            // What a decoder does with data after an error is its own; up to the first it is the same.
            let upto = |d: &Decoded| d.out.iter().position(|o| o.is_err()).map_or(d.out.len(), |i| i + 1);
            assert_eq!(
                whole.out[..upto(&whole)],
                rows.out[..upto(&rows)],
                "{} (grain applied: {apply_grain})",
                file.display()
            );
            pictures += whole.out.iter().filter(|o| o.is_ok()).count();
            errors += whole.out.iter().filter(|o| o.is_err()).count();
            frames_by_rows += rows.by_rows;
        }
    }
    eprintln!("{pictures} pictures, {errors} errors, {frames_by_rows} frames by rows");
    assert!(frames_by_rows * 10 > pictures * 8, "{frames_by_rows} of {pictures} by rows");
}

/// A frame left part decoded goes with the decoder, and so does the data it was decoded from.
#[test]
fn closes_with_a_frame_part_decoded() {
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/dav1d-test-data/8-bit/intra/av1-1-b8-02-allintra.ivf");
    let data = std::fs::read(file).unwrap();
    let frame = ivf_frames(&data)[0];
    unsafe extern "C" fn released(_: *const u8, cookie: Option<SendSyncNonNull<c_void>>) {
        cookie.unwrap().cast::<AtomicBool>().as_ref().store(true, Ordering::Relaxed);
    }
    let freed = AtomicBool::new(false);
    let ctx = open(true, true);
    unsafe {
        let mut d = Dav1dData::default();
        let cookie = SendSyncNonNull::from_ref(&freed).cast::<c_void>();
        let wrapped = dav1d_data_wrap(
            NonNull::new(&mut d),
            NonNull::new(frame.as_ptr() as *mut u8),
            frame.len(),
            Some(released),
            Some(cookie),
        );
        assert_eq!(wrapped.0, 0);
        assert_eq!(dav1d_send_data(Some(ctx), NonNull::new(&mut d)).0, 0);
        let mut n = 0;
        assert_eq!(dav1d_decode_rows(ctx, &mut n).0, 0);
        assert!(n > 0, "{n} rows final after the first superblock row");
        let mut p = Dav1dPicture::default();
        assert_eq!(dav1d_get_picture(Some(ctx), NonNull::new(&mut p)).0, EAGAIN, "a picture before its rows are");
        dav1d_close(NonNull::new(&mut Some(ctx)));
    }
    assert!(freed.load(Ordering::Relaxed), "the data outlived the decoder");
}

/// No later row of a frame reads a row once it is final: overwritten then, the picture is the same.
#[test]
fn reads_no_row_once_final() {
    let filter = std::env::var("ROWS_FILTER").unwrap_or_default();
    let (mut firsts, mut by_rows, mut differ) = (0, 0, Vec::new());
    for file in streams().iter().filter(|f| f.to_string_lossy().contains(&filter)) {
        let whole = decode(file, false, false, false);
        let poisoned = decode(file, false, true, true);
        let Some(first) = whole.out.first() else { continue };
        if poisoned.out.first().is_some_and(|o| o.is_err()) || first.is_err() {
            continue;
        }
        if poisoned.out.first() != Some(first) {
            differ.push(file.display().to_string());
        }
        firsts += 1;
        by_rows += poisoned.by_rows;
    }
    eprintln!("{firsts} first pictures, {by_rows} by rows, {} differ", differ.len());
    assert!(differ.is_empty(), "{differ:#?}");
    assert!(by_rows * 10 > firsts * 8, "{by_rows} of {firsts} by rows");
}

/// A tile group after the frame fails the data at the `dav1d_get_picture` after the picture's, by
/// rows as when the frame is decoded whole.
#[test]
fn fails_a_tile_group_after_the_frame_as_whole() {
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/dav1d-test-data/8-bit/intra/av1-1-b8-02-allintra.ivf");
    let data = std::fs::read(file).unwrap();
    let mut frame = ivf_frames(&data)[0].to_vec();
    frame.extend_from_slice(&[0x22, 0x01, 0x00]);
    let calls = |by_rows: bool| unsafe {
        let ctx = open(true, by_rows);
        let mut d = Dav1dData::default();
        let buf = dav1d_data_create(NonNull::new(&mut d), frame.len());
        std::ptr::copy_nonoverlapping(frame.as_ptr(), buf, frame.len());
        let mut results = vec![dav1d_send_data(Some(ctx), NonNull::new(&mut d)).0];
        let mut rows = 0;
        while dav1d_decode_rows(ctx, &mut rows).0 == 0 {}
        for _ in 0..3 {
            let mut p = Dav1dPicture::default();
            results.push(dav1d_get_picture(Some(ctx), NonNull::new(&mut p)).0);
            dav1d_picture_unref(NonNull::new(&mut p));
        }
        if d.sz > 0 {
            dav1d_data_unref(NonNull::new(&mut d));
        }
        dav1d_close(NonNull::new(&mut Some(ctx)));
        (results, rows)
    };
    let (whole, _) = calls(false);
    assert!(whole[0] == 0 && whole[1] == 0 && whole[2] < 0 && whole[2] != EAGAIN, "{whole:?}");
    let (by_rows, rows) = calls(true);
    assert_eq!(by_rows, whole);
    assert!(rows > 0);
}
