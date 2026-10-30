//! The scenarios of `decode_rows.rs` through the Rust API alone, and the parts of it that replace
//! the C API's: data read in place, picture allocators, workers on the caller's threads and
//! sequence headers parsed alone.

use std::hash::{DefaultHasher, Hasher};
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use rav1d::{
    parse_sequence_header, Decoder, Picture, PictureAllocator, PictureBuffer, PictureParameters,
    PixelLayout, PlanarImageComponent, PlaneLayout, Rav1dError, Settings, Spawner, PICTURE_ALIGNMENT,
};

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

/// `data` with every 97th byte of the last quarter of each frame flipped, where its last tiles are.
fn damaged(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let base = data.as_ptr() as usize;
    for frame in ivf_frames(data) {
        let start = frame.as_ptr() as usize - base;
        for i in (start + frame.len() * 3 / 4..start + frame.len()).step_by(97) {
            out[i] ^= 0x5a;
        }
    }
    out
}

#[derive(Clone, Default)]
struct Open {
    apply_grain: bool,
    by_rows: bool,
    threads: u32,
    rows_spawner: Option<Spawner>,
    spawner: Option<Spawner>,
    allocator: Option<Arc<dyn PictureAllocator>>,
}

impl Open {
    fn new(apply_grain: bool, by_rows: bool) -> Self {
        Open {
            apply_grain,
            by_rows,
            threads: 1,
            ..Default::default()
        }
    }

    fn decoder(&self) -> Decoder {
        let mut settings = Settings::new();
        settings.set_n_threads(self.threads);
        settings.set_max_frame_delay(1);
        settings.set_apply_grain(self.apply_grain);
        if let Some(allocator) = &self.allocator {
            settings.set_allocator(allocator.clone());
        }
        let mut decoder = match &self.spawner {
            Some(spawner) => Decoder::with_spawner(&settings, spawner),
            None => Decoder::with_settings(&settings),
        }
        .unwrap();
        decoder.set_decode_by_rows(self.by_rows);
        decoder.set_rows_spawner(self.rows_spawner.clone());
        decoder
    }
}

const COMPONENTS: [PlanarImageComponent; 3] = [
    PlanarImageComponent::Y,
    PlanarImageComponent::U,
    PlanarImageComponent::V,
];

/// A plane's visible rows: bytes a row, rows, stride, data.
fn plane(p: &Picture, i: usize) -> Option<(usize, usize, usize, *const u8)> {
    let (w, h) = (p.width() as usize, p.height() as usize);
    let (ss_hor, ss_ver) = match p.pixel_layout() {
        PixelLayout::I400 if i > 0 => return None,
        PixelLayout::I420 if i > 0 => (1, 1),
        PixelLayout::I444 => (0, 0),
        _ if i > 0 => (1, 0),
        _ => (0, 0),
    };
    let sample = if p.bit_depth() > 8 { 2 } else { 1 };
    let data = p.plane_ptr(COMPONENTS[i]);
    assert!(!data.is_null());
    Some((
        (w + ss_hor >> ss_hor) * sample,
        h + ss_ver >> ss_ver,
        p.stride(COMPONENTS[i]) as usize,
        data,
    ))
}

fn row<'a>((bytes, _, stride, data): (usize, usize, usize, *const u8), y: usize) -> &'a [u8] {
    unsafe { std::slice::from_raw_parts(data.add(stride * y), bytes) }
}

fn shift(p: &Picture, i: usize) -> u32 {
    (i > 0 && p.pixel_layout() == PixelLayout::I420) as u32
}

/// The digest of a whole picture, read through [`Picture::plane`].
fn digest(p: &Picture) -> u64 {
    let mut h = DefaultHasher::new();
    h.write_i32(p.width() as i32);
    h.write_i32(p.height() as i32);
    for i in 0..3 {
        if let Some((bytes, rows, stride, data)) = plane(p, i) {
            let whole = p.plane(COMPONENTS[i]);
            assert_eq!(whole.as_ptr(), data);
            for y in 0..rows {
                h.write(&whole[y * stride..][..bytes]);
            }
        }
    }
    h.finish()
}

/// What a decode gave: a digest a picture or the error, in order, and how many frames went by rows.
#[derive(Debug, Default, PartialEq)]
struct Decoded {
    out: Vec<Result<u64, Rav1dError>>,
    by_rows: usize,
    /// Overwrite each row once it is final, and stop at the first picture: its digest is then of
    /// the rows as they were when final.
    poison: bool,
}

/// The frame being decoded by rows: its picture, and its rows copied as each became final.
struct Rows {
    picture: Picture,
    copied: Vec<Vec<Vec<u8>>>,
    done: usize,
}

impl Rows {
    fn copy_to(&mut self, rows: usize) {
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
                    std::ptr::write_bytes(pl.3.add(pl.2 * y) as *mut u8, 0xa5, pl.0);
                }
            }
        }
    }

    fn digest(&self) -> u64 {
        let mut h = DefaultHasher::new();
        h.write_i32(self.picture.width() as i32);
        h.write_i32(self.picture.height() as i32);
        for plane in &self.copied {
            for row in plane {
                h.write(row);
            }
        }
        h.finish()
    }
}

/// Decodes the frame set up by rows to its end, checking that the rows it called final stay so.
fn by_rows(decoder: &mut Decoder, decoded: &mut Decoded) -> bool {
    let Ok(picture) = decoder.rows_picture() else {
        return false;
    };
    let height = picture.height() as usize;
    let mut rows = Rows {
        picture,
        copied: vec![Vec::new(); 3],
        done: 0,
    };
    loop {
        let n = match decoder.decode_rows() {
            Ok(n) => n as usize,
            Err(e) => {
                assert_ne!(e, Rav1dError::TryAgain, "rows of a frame left undecoded");
                decoded.out.push(Err(e));
                break;
            }
        };
        assert!(n >= rows.done && n <= height, "rows {n} after {}", rows.done);
        rows.copy_to(n);
        if decoded.poison {
            unsafe { rows.poison() };
            if n == height {
                decoded.out.push(Ok(rows.digest()));
                decoded.by_rows += 1;
                break;
            }
            continue;
        }
        if n == height {
            assert_eq!(decoder.decode_rows(), Err(Rav1dError::TryAgain), "rows past the frame's end");
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
    true
}

fn drain(decoder: &mut Decoder, rows: bool, decoded: &mut Decoded) {
    loop {
        let stepped = rows && by_rows(decoder, decoded);
        match decoder.get_picture() {
            Ok(p) => {
                if !(decoded.poison && stepped) {
                    decoded.out.push(Ok(digest(&p)));
                }
            }
            // get_picture may have set the next frame up by rows
            Err(Rav1dError::TryAgain) if stepped || rows && by_rows(decoder, decoded) => {}
            Err(Rav1dError::TryAgain) => break,
            Err(e) => decoded.out.push(Err(e)),
        }
    }
}

fn decode(file: &Path, open: &Open, poison: bool) -> Decoded {
    decode_data(&std::fs::read(file).unwrap(), open, poison)
}

fn decode_data(data: &[u8], open: &Open, poison: bool) -> Decoded {
    let mut decoded = Decoded {
        poison,
        ..Default::default()
    };
    let mut decoder = open.decoder();
    for frame in ivf_frames(data) {
        if poison && !decoded.out.is_empty() {
            break;
        }
        let mut sent = decoder.send_data(frame.into(), None, None, None);
        for _ in 0..1000 {
            match sent {
                Ok(()) => {}
                Err(Rav1dError::TryAgain) => {}
                Err(e) => {
                    decoded.out.push(Err(e));
                    break;
                }
            }
            drain(&mut decoder, open.by_rows, &mut decoded);
            if sent.is_ok() {
                break;
            }
            sent = decoder.send_pending_data();
        }
    }
    drain(&mut decoder, open.by_rows, &mut decoded);
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

fn filter() -> String {
    std::env::var("ROWS_FILTER").unwrap_or_default()
}

fn intra() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/dav1d-test-data/8-bit/intra/av1-1-b8-02-allintra.ivf")
}

/// Up to and with the first error: what a decoder does with data after one is its own.
fn upto(d: &Decoded) -> &[Result<u64, Rav1dError>] {
    &d.out[..d.out.iter().position(|o| o.is_err()).map_or(d.out.len(), |i| i + 1)]
}

#[test]
fn decodes_by_rows_as_whole() {
    let files = streams();
    assert!(files.len() > 700, "{} streams", files.len());
    let filter = filter();
    let (mut pictures, mut frames_by_rows, mut errors) = (0, 0, 0);
    for file in files.iter().filter(|f| f.to_string_lossy().contains(&filter)) {
        for apply_grain in [true, false] {
            let whole = decode(file, &Open::new(apply_grain, false), false);
            let rows = decode(file, &Open::new(apply_grain, true), false);
            assert_eq!(whole.by_rows, 0);
            assert_eq!(upto(&whole), upto(&rows), "{} (grain applied: {apply_grain})", file.display());
            pictures += whole.out.iter().filter(|o| o.is_ok()).count();
            errors += whole.out.iter().filter(|o| o.is_err()).count();
            frames_by_rows += rows.by_rows;
        }
    }
    eprintln!("{pictures} pictures, {errors} errors, {frames_by_rows} frames by rows");
    assert!(frames_by_rows * 10 > pictures * 8, "{frames_by_rows} of {pictures} by rows");
}

/// Bytes read in place that count, through a shared [`Arc`], how many hold them.
struct Shared(Arc<Vec<u8>>);

impl AsRef<[u8]> for Shared {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// A frame left part decoded goes with the decoder, and so does the data it was decoded from.
#[test]
fn closes_with_a_frame_part_decoded() {
    let data = std::fs::read(intra()).unwrap();
    let frame = Arc::new(ivf_frames(&data)[0].to_vec());
    let mut decoder = Open::new(true, true).decoder();
    decoder.send_data_owned(Shared(frame.clone()), None, None, None).unwrap();
    let n = decoder.decode_rows().unwrap();
    assert!(n > 0, "{n} rows final after the first superblock row");
    assert_eq!(decoder.get_picture().err(), Some(Rav1dError::TryAgain), "a picture before its rows are");
    assert!(Arc::strong_count(&frame) > 1, "the data went before the decoder");
    drop(decoder);
    assert!(Arc::try_unwrap(frame).is_ok(), "the data outlived the decoder");
}

/// Data read in place is the decoder's until it is dropped, whether it decoded it whole, kept some
/// of it to send again, or failed it; the picture is that of the same data copied.
#[test]
fn drops_data_read_in_place_with_the_decoder() {
    let data = std::fs::read(intra()).unwrap();
    let frames = ivf_frames(&data);
    let copied = decode_data(&data, &Open::new(true, false), false);
    let mut out = Vec::new();
    let mut decoder = Open::new(true, false).decoder();
    let shared: Vec<_> = frames.iter().map(|f| Arc::new(f.to_vec())).collect();
    for frame in &shared {
        let mut sent = decoder.send_data_owned(Shared(frame.clone()), None, None, None);
        while sent == Err(Rav1dError::TryAgain) {
            out.extend(decoder.get_picture().ok().map(|p| digest(&p)));
            sent = decoder.send_pending_data();
        }
        sent.unwrap();
        out.extend(decoder.get_picture().ok().map(|p| digest(&p)));
    }
    while let Ok(p) = decoder.get_picture() {
        out.push(digest(&p));
    }
    assert_eq!(copied.out, out.into_iter().map(Ok).collect::<Vec<_>>());
    drop(decoder);
    for frame in shared {
        assert!(Arc::try_unwrap(frame).is_ok(), "the data outlived the decoder");
    }

    // Frames sent while the first one's picture is not taken: the decoder holds the next, and
    // the one after waits as pending data.
    let shared: Vec<_> = frames[..3].iter().map(|f| Arc::new(f.to_vec())).collect();
    let mut decoder = Open::new(true, false).decoder();
    let sent: Vec<_> = shared
        .iter()
        .map(|f| decoder.send_data_owned(Shared(f.clone()), None, None, None))
        .take_while(|s| s.is_ok())
        .collect();
    assert_eq!(sent.len(), 2, "{sent:?}");
    assert!(shared.iter().skip(1).all(|f| Arc::strong_count(f) > 1));
    drop(decoder);
    for frame in shared {
        assert!(Arc::try_unwrap(frame).is_ok(), "the held or pending data outlived the decoder");
    }

    let broken = Arc::new(vec![0x12, 0x00, 0x0a, 0x0a, 0xff, 0xff, 0xff, 0xff]);
    let mut decoder = Open::new(true, false).decoder();
    assert!(decoder.send_data_owned(Shared(broken.clone()), None, None, None).is_err());
    drop(decoder);
    assert!(Arc::try_unwrap(broken).is_ok(), "the failed data outlived the decoder");
}

/// A tile group after the frame fails the data at the `get_picture` after the picture's, by rows
/// as when the frame is decoded whole.
#[test]
fn fails_a_tile_group_after_the_frame_as_whole() {
    let data = std::fs::read(intra()).unwrap();
    let mut frame = ivf_frames(&data)[0].to_vec();
    frame.extend_from_slice(&[0x22, 0x01, 0x00]);
    let calls = |by_rows: bool| {
        let mut decoder = Open::new(true, by_rows).decoder();
        let mut results = vec![decoder.send_data(frame.clone().into(), None, None, None)];
        let mut rows = 0;
        while let Ok(n) = decoder.decode_rows() {
            rows = n;
        }
        for _ in 0..3 {
            results.push(decoder.get_picture().map(|_| ()));
        }
        (results, rows)
    };
    let (whole, _) = calls(false);
    assert!(
        whole[0].is_ok() && whole[1].is_ok() && whole[2].is_err() && whole[2] != Err(Rav1dError::TryAgain),
        "{whole:?}"
    );
    let (by_rows, rows) = calls(true);
    assert_eq!(by_rows, whole);
    assert!(rows > 0);
}

/// One thread keeps a superblock row of the frame's line buffers where tile threads keep them all.
#[test]
fn decodes_on_one_thread_as_on_four() {
    let filter = filter();
    let (mut pictures, mut differ) = (0, Vec::new());
    for file in streams().iter().filter(|f| f.to_string_lossy().contains(&filter)) {
        let one = decode(file, &Open::new(true, false), false);
        let four = decode(
            file,
            &Open {
                threads: 4,
                ..Open::new(true, false)
            },
            false,
        );
        let upto = |d: &Decoded| d.out.iter().position(|o| o.is_err());
        let n = match (upto(&one), upto(&four)) {
            (None, None) if one.out.len() != four.out.len() => usize::MAX,
            (a, b) => a.unwrap_or(one.out.len()).min(b.unwrap_or(four.out.len())),
        };
        if n == usize::MAX || one.out[..n] != four.out[..n] {
            differ.push(file.display().to_string());
            continue;
        }
        pictures += n;
    }
    eprintln!("{pictures} pictures, {} streams differ", differ.len());
    assert!(differ.is_empty(), "{differ:#?}");
    assert!(!filter.is_empty() || pictures > 10_000, "{pictures} pictures");
}

/// A spawner that counts what it is offered and takes, running the work on threads of its own
/// except every third offer, which it refuses.
fn spawning(asked: &Arc<AtomicUsize>, taken: &Arc<AtomicUsize>, inline: bool) -> Spawner {
    let (a, t) = (Arc::clone(asked), Arc::clone(taken));
    Arc::new(move |work| {
        if inline {
            a.fetch_add(1, Ordering::Relaxed);
            t.fetch_add(1, Ordering::Relaxed);
            work();
            return true;
        }
        if a.fetch_add(1, Ordering::Relaxed) % 3 == 2 {
            return false;
        }
        t.fetch_add(1, Ordering::Relaxed);
        std::thread::spawn(work);
        true
    })
}

/// The tile columns of a row decoded by rows on spawned threads, some taken and some not, or all of
/// them run as offered so that the calling thread decodes none, come out as decoded whole, and so
/// do the streams with columns damaged, up to their first error.
#[test]
fn decodes_tile_columns_on_spawned_threads_as_whole() {
    let filter = filter();
    let (asked, taken) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let (mut pictures, mut differ, mut damaged_errors) = (0, Vec::new(), 0);
    for (i, file) in streams().iter().filter(|f| f.to_string_lossy().contains(&filter)).enumerate() {
        let offered = asked.load(Ordering::Relaxed);
        let rows = Open {
            rows_spawner: Some(spawning(&asked, &taken, i % 2 == 0)),
            ..Open::new(true, true)
        };
        let whole = decode(file, &Open::new(true, false), false);
        let by_rows = decode(file, &rows, false);
        if upto(&whole) != upto(&by_rows) {
            differ.push(file.display().to_string());
            continue;
        }
        pictures += whole.out.iter().filter(|o| o.is_ok()).count();
        if asked.load(Ordering::Relaxed) == offered {
            continue;
        }
        let data = damaged(&std::fs::read(file).unwrap());
        let whole = decode_data(&data, &Open::new(true, false), false);
        let by_rows = decode_data(&data, &rows, false);
        if upto(&whole) != upto(&by_rows) {
            differ.push(format!("{} damaged", file.display()));
        }
        damaged_errors += whole.out.iter().filter(|o| o.is_err()).count();
    }
    let (asked, taken) = (asked.load(Ordering::Relaxed), taken.load(Ordering::Relaxed));
    eprintln!(
        "{pictures} pictures, {asked} columns offered, {taken} taken, {damaged_errors} errors damaged, {} streams differ",
        differ.len()
    );
    assert!(differ.is_empty(), "{differ:#?}");
    assert!(filter.len() > 0 || damaged_errors > 0, "no damaged stream failed");
    assert!(taken > 0 && asked > taken, "{asked} offered, {taken} taken");
}

/// Workers on the caller's threads, some refused, decode what the decoder's own threads do.
#[test]
fn decodes_with_workers_on_spawned_threads_as_on_its_own() {
    let filter = filter();
    let (asked, taken) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let (mut pictures, mut differ) = (0, Vec::new());
    let files = streams();
    for file in files.iter().filter(|f| f.to_string_lossy().contains(&filter)).step_by(8) {
        let own = decode(file, &Open::new(true, false), false);
        let spawned = Open {
            threads: 4,
            spawner: Some(spawning(&asked, &taken, false)),
            ..Open::new(true, false)
        };
        let spawned = decode(file, &spawned, false);
        if upto(&own) != upto(&spawned) {
            differ.push(file.display().to_string());
            continue;
        }
        pictures += own.out.iter().filter(|o| o.is_ok()).count();
    }
    let (asked, taken) = (asked.load(Ordering::Relaxed), taken.load(Ordering::Relaxed));
    eprintln!("{pictures} pictures, {asked} workers offered, {taken} taken, {} streams differ", differ.len());
    assert!(differ.is_empty(), "{differ:#?}");
    assert!(taken > 0 && asked > taken, "{asked} offered, {taken} taken");
}

/// No later row of a frame reads a row once it is final: overwritten then, the picture is the same.
#[test]
fn reads_no_row_once_final() {
    let filter = filter();
    let (mut firsts, mut by_rows, mut differ) = (0, 0, Vec::new());
    for file in streams().iter().filter(|f| f.to_string_lossy().contains(&filter)) {
        let whole = decode(file, &Open::new(false, false), false);
        let poisoned = decode(file, &Open::new(false, true), true);
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

#[test]
#[should_panic(expected = "plane_ptr")]
fn refuses_a_slice_of_a_picture_decoded_by_rows() {
    let data = std::fs::read(intra()).unwrap();
    let mut decoder = Open::new(true, true).decoder();
    decoder.send_data(ivf_frames(&data)[0].into(), None, None, None).unwrap();
    let picture = decoder.rows_picture().unwrap();
    picture.plane(PlanarImageComponent::Y);
}

/// Memory from the heap, aligned by where the planes start in it.
struct Aligned(Vec<u8>);

impl AsMut<[u8]> for Aligned {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

/// Allocates as rav1d's own allocator, or as `broken` breaks that, counting what it allocates and
/// what goes.
struct Counting {
    allocated: AtomicUsize,
    released: Arc<AtomicUsize>,
    broken: Option<fn(&PictureParameters, &mut PlaneLayout)>,
}

struct Released(Aligned, Arc<AtomicUsize>);

impl AsMut<[u8]> for Released {
    fn as_mut(&mut self) -> &mut [u8] {
        self.0.as_mut()
    }
}

impl Drop for Released {
    fn drop(&mut self) {
        self.1.fetch_add(1, Ordering::Relaxed);
    }
}

impl Counting {
    fn new(broken: Option<fn(&PictureParameters, &mut PlaneLayout)>) -> Arc<Self> {
        Arc::new(Counting {
            allocated: AtomicUsize::new(0),
            released: Arc::new(AtomicUsize::new(0)),
            broken,
        })
    }
}

impl PictureAllocator for Counting {
    fn alloc(&self, picture: &PictureParameters) -> Result<PictureBuffer, Rav1dError> {
        let mut layout = picture.plane_layout().ok_or(Rav1dError::OutOfMemory)?;
        let mut memory = vec![0x5a; layout.len + PICTURE_ALIGNMENT];
        let skip = memory.as_ptr().align_offset(PICTURE_ALIGNMENT);
        layout.offsets = layout.offsets.map(|o| o + skip);
        layout.len += skip;
        if let Some(broken) = self.broken {
            broken(picture, &mut layout);
        }
        memory.truncate(layout.len);
        self.allocated.fetch_add(1, Ordering::Relaxed);
        Ok(PictureBuffer {
            memory: Box::new(Released(Aligned(memory), self.released.clone())),
            layout,
        })
    }
}

/// Pictures from an allocator of the caller's, whole and by rows, are those of rav1d's own, and
/// every one goes back to it by the time the decoder and its pictures are gone.
#[test]
fn decodes_into_pictures_of_its_allocator() {
    let filter = filter();
    let counting = Counting::new(None);
    let (mut pictures, mut differ) = (0, Vec::new());
    let files = streams();
    for file in files.iter().filter(|f| f.to_string_lossy().contains(&filter)).step_by(4) {
        let own = decode(file, &Open::new(true, false), false);
        for by_rows in [false, true] {
            let theirs = Open {
                allocator: Some(counting.clone()),
                ..Open::new(true, by_rows)
            };
            if upto(&own) != upto(&decode(file, &theirs, false)) {
                differ.push(format!("{} (by rows: {by_rows})", file.display()));
            }
        }
        pictures += own.out.iter().filter(|o| o.is_ok()).count();
    }
    let (allocated, released) = (counting.allocated.load(Ordering::Relaxed), counting.released.load(Ordering::Relaxed));
    eprintln!("{pictures} pictures, {allocated} allocated, {} streams differ", differ.len());
    assert!(differ.is_empty(), "{differ:#?}");
    assert!(allocated > 0);
    assert_eq!(allocated, released);
}

/// A buffer that breaks what [`PictureAllocator`] asks fails the frame, and goes back.
#[test]
fn fails_a_frame_its_allocator_breaks() {
    let data = std::fs::read(intra()).unwrap();
    let breaks: [(&str, fn(&PictureParameters, &mut PlaneLayout)); 6] = [
        ("misaligned", |_, l| l.offsets[0] += 1),
        ("narrow", |p, l| l.strides[0] = p.min_stride(0) - PICTURE_ALIGNMENT),
        ("odd stride", |_, l| l.strides[1] += 1),
        ("overlapping", |_, l| l.offsets[2] = l.offsets[1]),
        ("short", |_, l| l.len -= 1),
        ("past the end", |_, l| l.offsets[2] += PICTURE_ALIGNMENT),
    ];
    for (what, broken) in breaks {
        let counting = Counting::new(Some(broken));
        for by_rows in [false, true] {
            let open = Open {
                allocator: Some(counting.clone()),
                ..Open::new(true, by_rows)
            };
            let decoded = decode_data(&data, &open, false);
            assert_eq!(decoded.out.first(), Some(&Err(Rav1dError::InvalidArgument)), "{what}");
        }
        let released = counting.released.load(Ordering::Relaxed);
        assert_eq!(counting.allocated.load(Ordering::Relaxed), released, "{what}");
        assert!(released > 0);
    }

    struct Panicking;
    impl PictureAllocator for Panicking {
        fn alloc(&self, _: &PictureParameters) -> Result<PictureBuffer, Rav1dError> {
            panic!("no memory for a picture");
        }
    }
    let decoded = decode_data(&data, &Open { allocator: Some(Arc::new(Panicking)), ..Open::new(true, false) }, false);
    assert!(decoded.out.first().is_some_and(|o| o.is_err()), "{:?}", decoded.out);
}

#[test]
fn parses_a_sequence_header_alone() {
    let mut checked = 0;
    for file in streams().iter().step_by(16) {
        let data = std::fs::read(file).unwrap();
        let Some(&frame) = ivf_frames(&data).first() else { continue };
        let mut c = MaybeUninit::<rav1d::include::dav1d::headers::Dav1dSequenceHeader>::zeroed();
        let res = unsafe {
            rav1d::dav1d_parse_sequence_header(NonNull::new(c.as_mut_ptr()), NonNull::new(frame.as_ptr() as *mut u8), frame.len())
        };
        let rust = parse_sequence_header(frame);
        assert_eq!(res.0 == 0, rust.is_ok(), "{}", file.display());
        let Ok(seq) = rust else { continue };
        let c = unsafe { c.assume_init() };
        assert_eq!((seq.max_width, seq.max_height), (c.max_width, c.max_height));
        assert_eq!(seq.layout as u32, c.layout as u32);
        assert_eq!(seq.hbd, c.hbd);
        assert_eq!(seq.film_grain_present, c.film_grain_present);
        assert_eq!(seq.reduced_still_picture_header, c.reduced_still_picture_header);
        assert_eq!(seq.num_operating_points, c.num_operating_points);
        assert_eq!(seq.operating_points[0].idc, c.operating_points[0].idc);
        assert_eq!(seq.screen_content_tools as u32, c.screen_content_tools as u32);
        assert_eq!(seq.order_hint_n_bits, c.order_hint_n_bits);
        checked += 1;
    }
    assert!(checked > 20, "{checked} sequence headers");
    assert_eq!(parse_sequence_header(&[]).err(), Some(Rav1dError::InvalidArgument));
    assert_eq!(parse_sequence_header(&[0x32, 0x00]).err(), Some(Rav1dError::NoEntity));
}

#[test]
fn gives_each_picture_the_sequence_header_it_was_decoded_under() {
    let mut checked = 0;
    for file in streams().iter().step_by(8) {
        let data = std::fs::read(file).unwrap();
        let mut decoder = Decoder::with_settings(&Settings::new()).unwrap();
        let mut last = None;
        for frame in ivf_frames(&data) {
            if let Ok(seq) = parse_sequence_header(frame) {
                last = Some(seq);
            }
            let mut sent = decoder.send_data(frame.into(), None, None, None);
            while matches!(sent, Err(Rav1dError::TryAgain)) {
                while let Ok(p) = decoder.get_picture() {
                    check_sequence_header(&p, last.as_ref(), file, &mut checked);
                }
                sent = decoder.send_pending_data();
            }
            if sent.is_err() {
                break;
            }
            while let Ok(p) = decoder.get_picture() {
                check_sequence_header(&p, last.as_ref(), file, &mut checked);
            }
        }
    }
    assert!(checked > 100, "{checked} pictures");
}

fn check_sequence_header(p: &Picture, parsed: Option<&rav1d::SequenceHeader>, file: &Path, checked: &mut usize) {
    let seq = p.sequence_header().expect("a decoded picture has a sequence header");
    let parsed = parsed.expect("a picture before any sequence header");
    let f = file.display();
    assert_eq!((seq.max_width, seq.max_height, seq.hbd), (parsed.max_width, parsed.max_height, parsed.hbd), "{f}");
    assert_eq!(
        (seq.pri as u32, seq.trc as u32, seq.mtrx as u32, seq.color_range as u32, seq.layout as u32),
        (parsed.pri as u32, parsed.trc as u32, parsed.mtrx as u32, parsed.color_range as u32, parsed.layout as u32),
        "{f}"
    );
    assert_eq!(p.matrix_coefficients(), seq.mtrx.into(), "{f}");
    *checked += 1;
}

/// Data read in place goes by the time a decoder with threads of its own is dropped, also with a
/// frame left part decoded in it.
#[test]
fn drops_data_read_in_place_with_a_decoder_of_threads() {
    let data = std::fs::read(intra()).unwrap();
    let frames = ivf_frames(&data);
    for threads in [2, 4, 8] {
        for run in 0..50 {
            let mut decoder = Open { threads, ..Open::new(true, false) }.decoder();
            let mut sent = Vec::new();
            for (i, frame) in frames.iter().take(3).enumerate() {
                let bytes = match i == 2 && run % 2 == 1 {
                    true => frame[..frame.len() / 2].to_vec(),
                    false => frame.to_vec(),
                };
                let owner = Arc::new(bytes);
                sent.push(Arc::clone(&owner));
                if decoder.send_data_owned(Shared(owner), None, None, None).is_err() {
                    break;
                }
            }
            drop(decoder);
            for owner in &sent {
                assert_eq!(Arc::strong_count(owner), 1, "{threads} threads, run {run}");
            }
        }
    }
}

/// A rows spawner that unwinds after it took a column: the column, run once the decoder is gone,
/// finds nothing left to decode, as `decode_rows` unwinding takes or waits for every column.
#[test]
fn waits_for_a_column_its_spawner_took_before_it_unwound() {
    let threads = Arc::new(std::sync::Mutex::new(Vec::new()));
    let t = Arc::clone(&threads);
    let spawner: Spawner = Arc::new(move |work| {
        t.lock().unwrap().push(std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            work();
        }));
        panic!("the spawner unwinds after it took a column");
    });
    let mut unwound = 0;
    for file in streams() {
        let mut decoder = Open { rows_spawner: Some(spawner.clone()), ..Open::new(true, true) }.decoder();
        let data = std::fs::read(&file).unwrap();
        'frames: for frame in ivf_frames(&data) {
            if decoder.send_data(frame.into(), None, None, None).is_err() || decoder.rows_picture().is_err() {
                break;
            }
            loop {
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| decoder.decode_rows())) {
                    Err(_) => {
                        unwound += 1;
                        break 'frames;
                    }
                    Ok(Ok(rows)) if rows < decoder.rows_picture().map_or(0, |p| p.height()) => {}
                    Ok(_) => break,
                }
            }
        }
        drop(decoder);
        for thread in threads.lock().unwrap().drain(..) {
            assert!(thread.join().is_ok(), "a column ran on a decoder that is gone: {}", file.display());
        }
        if unwound >= 3 {
            break;
        }
    }
    assert!(unwound > 0, "no stream had a column to hand out");
}

/// Memory whose `as_mut` or `Drop` panics fails its frame or is leaked, and never aborts.
#[test]
fn survives_picture_memory_that_panics() {
    struct Memory(Aligned, bool);
    impl AsMut<[u8]> for Memory {
        fn as_mut(&mut self) -> &mut [u8] {
            assert!(!self.1, "no bytes for a picture");
            self.0.as_mut()
        }
    }
    impl Drop for Memory {
        fn drop(&mut self) {
            if !self.1 {
                panic!("the picture's memory panics as it goes");
            }
        }
    }
    struct Allocator(bool);
    impl PictureAllocator for Allocator {
        fn alloc(&self, picture: &PictureParameters) -> Result<PictureBuffer, Rav1dError> {
            let mut layout = picture.plane_layout().ok_or(Rav1dError::OutOfMemory)?;
            let memory = vec![0; layout.len + PICTURE_ALIGNMENT];
            let skip = memory.as_ptr().align_offset(PICTURE_ALIGNMENT);
            layout.offsets = layout.offsets.map(|o| o + skip);
            layout.len += skip;
            Ok(PictureBuffer { memory: Box::new(Memory(Aligned(memory), self.0)), layout })
        }
    }
    let data = std::fs::read(intra()).unwrap();
    let in_as_mut = decode_data(&data, &Open { allocator: Some(Arc::new(Allocator(true))), ..Open::new(true, false) }, false);
    assert_eq!(in_as_mut.out.first(), Some(&Err(Rav1dError::OutOfMemory)));
    let in_drop = decode_data(&data, &Open { allocator: Some(Arc::new(Allocator(false))), ..Open::new(true, false) }, false);
    assert!(in_drop.out.iter().filter(|o| o.is_ok()).count() > 0, "{:?}", in_drop.out);
}
