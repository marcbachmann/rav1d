//! The dav1d-compatible C API, for the CLI and its tests (feature `capi`).

use std::ffi::{c_char, c_int, c_uint, c_void};
use std::ptr::NonNull;
use std::{mem, ptr, slice};

use to_method::To as _;

use super::*;
use crate::c_arc::RawArc;
use crate::c_box::{CBox, FnFree};
use crate::decode::rav1d_decode_rows;
use crate::include::common::validate::validate_input;
use crate::include::dav1d::common::{Dav1dDataProps, Rav1dDataProps};
use crate::include::dav1d::data::Dav1dData;
use crate::include::dav1d::dav1d::{Dav1dContext, Dav1dEventFlags, Dav1dSettings};
use crate::include::dav1d::headers::Dav1dSequenceHeader;
use crate::include::dav1d::picture::Dav1dPicture;
use crate::obu::rav1d_parse_sequence_header;

#[no_mangle]
#[cold]
pub extern "C" fn dav1d_version() -> *const c_char {
    DAV1D_VERSION.as_ptr()
}

/// Get the `dav1d` library C API version.
///
/// Return a value in the format `0x00XXYYZZ`, where `XX` is the major version,
/// `YY` the minor version, and `ZZ` the patch version.
#[no_mangle]
#[cold]
pub extern "C" fn dav1d_version_api() -> c_uint {
    u32::from_be_bytes([
        0,
        DAV1D_API_VERSION_MAJOR,
        DAV1D_API_VERSION_MINOR,
        DAV1D_API_VERSION_PATCH,
    ])
}

/// # Safety
///
/// * `s` must be valid to [`ptr::write`] to.
///   The former contents of `s` are not [`drop`]ped and it may be uninitialized.
#[no_mangle]
#[cold]
pub unsafe extern "C" fn dav1d_default_settings(s: NonNull<Dav1dSettings>) {
    let settings = Rav1dSettings::default().into();
    // SAFETY: `s` is safe to `ptr::write` to.
    unsafe { s.as_ptr().write(settings) };
}

/// # Safety
///
/// * `s`, if [`NonNull`], must valid to [`ptr::read`] from.
#[no_mangle]
#[cold]
pub unsafe extern "C" fn dav1d_get_frame_delay(s: Option<NonNull<Dav1dSettings>>) -> Dav1dResult {
    (|| {
        let s = validate_input!(s.ok_or(Rav1dError::InvalidArgument))?;
        // SAFETY: `s` is safe to `ptr::read`.
        let s = unsafe { s.as_ptr().read() };
        let s = s.try_into()?;
        Ok(rav1d_get_frame_delay(&s) as c_uint)
    })()
    .into()
}

/// # Safety
///
/// * `c_out`, if [`NonNull`], is valid to [`ptr::write`] to.
/// * `s`, if [`NonNull`], is valid to [`ptr::read`] from.
#[no_mangle]
#[cold]
pub unsafe extern "C" fn dav1d_open(
    c_out: Option<NonNull<Option<Dav1dContext>>>,
    s: Option<NonNull<Dav1dSettings>>,
) -> Dav1dResult {
    // SAFETY: as this function's.
    unsafe { open(c_out, s, None) }
}

/// [`dav1d_open`] with the workers on threads of the caller's `spawner` instead of the
/// decoder's own: of `n_threads - 1` workers, those it takes when asked, while the calling thread
/// decodes as a worker too until its frame is done. Frames are decoded one at a time
/// (`max_frame_delay` 1); with no worker taken it decodes alone. [`dav1d_close`] returns once
/// every worker has returned and let go of the context, with it the data it was sent.
///
/// # Safety
///
/// As [`dav1d_open`].
#[cold]
pub unsafe fn dav1d_open_with_spawner(
    c_out: Option<NonNull<Option<Dav1dContext>>>,
    s: Option<NonNull<Dav1dSettings>>,
    spawner: &Spawner,
) -> Dav1dResult {
    // SAFETY: as this function's.
    unsafe { open(c_out, s, Some(spawner)) }
}

/// # Safety
///
/// As [`dav1d_open`].
unsafe fn open(
    c_out: Option<NonNull<Option<Dav1dContext>>>,
    s: Option<NonNull<Dav1dSettings>>,
    spawner: Option<&Spawner>,
) -> Dav1dResult {
    (|| {
        let mut c_out = validate_input!(c_out.ok_or(Rav1dError::InvalidArgument))?;
        let s = validate_input!(s.ok_or(Rav1dError::InvalidArgument))?;
        // SAFETY: `c_out` is safe to write to.
        let c_out = unsafe { c_out.as_mut() };
        // SAFETY: `s` is safe to read from.
        let s = unsafe { s.as_ptr().read() };
        let s = s.try_into()?;
        let c = rav1d_open(&s, spawner).inspect_err(|_| {
            *c_out = None;
        })?;
        *c_out = Some(RawArc::from_arc(c));
        Ok(())
    })()
    .into()
}

/// # Safety
///
/// * `out`, if [`NonNull`], is valid to [`ptr::write`] to.
/// * `ptr`, if [`NonNull`], is the start of a `&[u8]` slice of length `sz`.
#[no_mangle]
pub unsafe extern "C" fn dav1d_parse_sequence_header(
    out: Option<NonNull<Dav1dSequenceHeader>>,
    ptr: Option<NonNull<u8>>,
    sz: usize,
) -> Dav1dResult {
    (|| {
        let out = validate_input!(out.ok_or(Rav1dError::InvalidArgument))?;
        let ptr = validate_input!(ptr.ok_or(Rav1dError::InvalidArgument))?;
        validate_input!((sz > 0 && sz <= usize::MAX / 2, Rav1dError::InvalidArgument))?;
        // SAFETY: `ptr` is the start of a `&[u8]` slice of length `sz`.
        let data = unsafe { slice::from_raw_parts(ptr.as_ptr(), sz) };
        let seq_hdr = rav1d_parse_sequence_header(data)?.dav1d;
        // SAFETY: `out` is safe to write to.
        unsafe { out.as_ptr().write(seq_hdr) };
        Ok(())
    })()
    .into()
}

/// # Safety
///
/// * `c`, if [`NonNull`], must be from [`dav1d_open`] and not be passed to [`dav1d_close`] yet.
/// * `r#in`, if [`NonNull`], must be valid to [`ptr::read`] from and [`ptr::write`] to.
#[no_mangle]
pub unsafe extern "C" fn dav1d_send_data(
    c: Option<Dav1dContext>,
    r#in: Option<NonNull<Dav1dData>>,
) -> Dav1dResult {
    (|| {
        let c = validate_input!(c.ok_or(Rav1dError::InvalidArgument))?;
        let r#in = validate_input!(r#in.ok_or(Rav1dError::InvalidArgument))?;
        // SAFETY: `c` is from `dav1d_open` and thus from `RawArc::from_arc`.
        // It has not yet been passed to `dav1d_close` and thus not to `RawArc::into_arc` yet.
        let c = unsafe { c.as_ref() };
        // SAFETY: `r#in` is safe to read from.
        let in_c = unsafe { r#in.as_ptr().read() };
        let mut in_rust = in_c.into();
        let result = rav1d_send_data(c, &mut in_rust);
        let in_c = in_rust.into();
        // SAFETY: `r#in` is safe to write to.
        unsafe { r#in.as_ptr().write(in_c) };
        result
    })()
    .into()
}

/// # Safety
///
/// * `c`, if [`NonNull`], must be from [`dav1d_open`] and not be passed to [`dav1d_close`] yet.
/// * `out`, if [`NonNull`], must be valid to [`ptr::write`] to.
#[no_mangle]
pub unsafe extern "C" fn dav1d_get_picture(
    c: Option<Dav1dContext>,
    out: Option<NonNull<Dav1dPicture>>,
) -> Dav1dResult {
    (|| {
        let c = validate_input!(c.ok_or(Rav1dError::InvalidArgument))?;
        let out = validate_input!(out.ok_or(Rav1dError::InvalidArgument))?;
        // SAFETY: `c` is from `dav1d_open` and thus from `RawArc::from_arc`.
        // It has not yet been passed to `dav1d_close` and thus not to `RawArc::into_arc` yet.
        let c = unsafe { c.as_ref() };
        let mut out_rust = Default::default(); // TODO(kkysen) Temporary until we return it directly.
        let result = rav1d_get_picture(c, &mut out_rust);
        let out_c = out_rust.into();
        // SAFETY: `out` is safe to write to.
        unsafe { out.as_ptr().write(out_c) };
        result
    })()
    .into()
}

/// Decodes frames that are decoded on the calling thread alone (one frame at a time, no worker
/// taken) a superblock row at a time: [`dav1d_send_data`] or [`dav1d_get_picture`], whichever
/// parses such a frame, only sets it up, each [`dav1d_decode_rows`] decodes one more row of
/// superblocks, and [`dav1d_get_picture`] gives the picture once all are, `EAGAIN` before. A frame
/// with film grain to apply, one copying blocks from rows above (intra block copy), or one not
/// output, is decoded whole.
///
/// # Safety
///
/// * `c` must be from [`dav1d_open`] and not be passed to [`dav1d_close`] yet.
pub unsafe fn dav1d_set_decode_by_rows(c: Dav1dContext, by_rows: bool) {
    // SAFETY: `c` is from `dav1d_open` and thus from `RawArc::from_arc`.
    let c = unsafe { c.as_ref() };
    c.state.try_lock().unwrap().by_rows = by_rows;
}

/// Decodes the tile columns of each superblock row decoded by rows on the threads `spawner` hands
/// out as well as on the calling thread, which [`dav1d_decode_rows`] returns once all are decoded;
/// `None` decodes them on the calling thread alone.
///
/// # Safety
///
/// * `c` must be from [`dav1d_open`] and not be passed to [`dav1d_close`] yet.
pub unsafe fn dav1d_set_rows_spawner(c: Dav1dContext, spawner: Option<Spawner>) {
    // SAFETY: `c` is from `dav1d_open` and thus from `RawArc::from_arc`.
    let c = unsafe { c.as_ref() };
    c.state.try_lock().unwrap().rows_spawner = spawner;
}

/// Decodes the next superblock row of the frame [`dav1d_send_data`] set up to be decoded by rows,
/// and writes how many rows from the top of its picture no later row changes or reads, its height
/// once the frame is done. `EAGAIN` where no frame is being decoded by rows; on an error the frame is
/// gone, as where [`dav1d_send_data`] fails to decode it.
///
/// # Safety
///
/// * `c` must be from [`dav1d_open`] and not be passed to [`dav1d_close`] yet.
pub unsafe fn dav1d_decode_rows(c: Dav1dContext, rows: &mut c_int) -> Dav1dResult {
    // SAFETY: `c` is from `dav1d_open` and thus from `RawArc::from_arc`.
    let c = unsafe { c.as_ref() };
    let state = &mut *c.state.try_lock().unwrap();
    rav1d_decode_rows(c, state).map(|n| *rows = n).into()
}

/// A reference to the picture of the frame being decoded by rows, to read its final rows from
/// while it is; released with [`dav1d_picture_unref`]. `EAGAIN` where no frame is.
///
/// # Safety
///
/// * `c` must be from [`dav1d_open`] and not be passed to [`dav1d_close`] yet.
/// * `out` must be valid to [`ptr::write`] to.
pub unsafe fn dav1d_rows_picture(c: Dav1dContext, out: NonNull<Dav1dPicture>) -> Dav1dResult {
    // SAFETY: `c` is from `dav1d_open` and thus from `RawArc::from_arc`.
    let c = unsafe { c.as_ref() };
    let state = &mut *c.state.try_lock().unwrap();
    if state.rows.is_none() {
        return Err::<(), _>(Rav1dError::TryAgain).into();
    }
    let picture: Dav1dPicture = state.out.p.clone().into();
    // SAFETY: `out` is safe to write to.
    unsafe { out.as_ptr().write(picture) };
    Ok(()).into()
}

/// # Safety
///
/// * `c`, if [`NonNull`], must be from [`dav1d_open`] and not be passed to [`dav1d_close`] yet.
/// * `out`, if [`NonNull`], must be valid to [`ptr::write`] to.
/// * `r#in`, if [`NonNull`], must be valid to [`ptr::read`] from.
#[no_mangle]
pub unsafe extern "C" fn dav1d_apply_grain(
    c: Option<Dav1dContext>,
    out: Option<NonNull<Dav1dPicture>>,
    r#in: Option<NonNull<Dav1dPicture>>,
) -> Dav1dResult {
    (|| {
        let c = validate_input!(c.ok_or(Rav1dError::InvalidArgument))?;
        let out = validate_input!(out.ok_or(Rav1dError::InvalidArgument))?;
        let r#in = validate_input!(r#in.ok_or(Rav1dError::InvalidArgument))?;
        // SAFETY: `c` is from `dav1d_open` and thus from `RawArc::from_arc`.
        // It has not yet been passed to `dav1d_close` and thus not to `RawArc::into_arc` yet.
        let c = unsafe { c.as_ref() };
        // SAFETY: `r#in` is safe to read from.
        let in_c = unsafe { r#in.as_ptr().read() };
        // Don't `.update_rav1d()` [`Rav1dSequenceHeader`] because it's meant to be read-only.
        // Don't `.update_rav1d()` [`Rav1dFrameHeader`] because it's meant to be read-only.
        // Don't `.update_rav1d()` [`Rav1dITUTT35`] because we never read it.
        let mut out_rust = Default::default(); // TODO(kkysen) Temporary until we return it directly.
        let in_rust = in_c.into();
        let result = rav1d_apply_grain(c, &mut out_rust, &in_rust);
        let out_c = out_rust.into();
        // SAFETY: `out` is safe to write to.
        unsafe { out.as_ptr().write(out_c) };
        result
    })()
    .into()
}

/// # Safety
///
/// * `c` must be from [`dav1d_open`] and not be passed to [`dav1d_close`] yet.
#[no_mangle]
pub unsafe extern "C" fn dav1d_flush(c: Dav1dContext) {
    // SAFETY: `c` is from `dav1d_open` and thus from `RawArc::from_arc`.
    // It has not yet been passed to `dav1d_close` and thus not to `RawArc::into_arc` yet.
    let c = unsafe { c.as_ref() };
    rav1d_flush(c)
}

/// # Safety
///
/// * `c_out`, if [`NonNull`], must be safe to [`ptr::read`] from and [`ptr::write`] to.
///   The `Dav1dContext` pointed to by `c_out` must be from [`dav1d_open`].
#[no_mangle]
#[cold]
pub unsafe extern "C" fn dav1d_close(c_out: Option<NonNull<Option<Dav1dContext>>>) {
    let Ok(mut c_out) = validate_input!(c_out.ok_or(())) else {
        return;
    };
    // SAFETY: `c_out` is safe to read from and write to.
    let c_out = unsafe { c_out.as_mut() };
    mem::take(c_out).map(|c| {
        // SAFETY: `c` is from `dav1d_open` and thus from `RawArc::from_arc`.
        let c = unsafe { c.into_arc() };
        rav1d_close(&c);
    });
}

/// # Safety
///
/// * `c`, if [`NonNull`], must be from [`dav1d_open`] and not be passed to [`dav1d_close`] yet.
/// * `flags`, if [`NonNull`], must be valid to [`ptr::write`] to.
#[no_mangle]
pub unsafe extern "C" fn dav1d_get_event_flags(
    c: Option<Dav1dContext>,
    flags: Option<NonNull<Dav1dEventFlags>>,
) -> Dav1dResult {
    (|| {
        let c = validate_input!(c.ok_or(Rav1dError::InvalidArgument))?;
        let flags = validate_input!(flags.ok_or(Rav1dError::InvalidArgument))?;
        // SAFETY: `c` is from `dav1d_open` and thus from `RawArc::from_arc`.
        // It has not yet been passed to `dav1d_close` and thus not to `RawArc::into_arc` yet.
        let c = unsafe { c.as_ref() };
        let state = &mut *c.state.try_lock().unwrap();
        let flags_rust = mem::take(&mut state.event_flags);
        let flags_c = flags_rust.into();
        // SAFETY: `flags` is safe to write to.
        unsafe { flags.as_ptr().write(flags_c) };
        Ok(())
    })()
    .into()
}

/// # Safety
///
/// * `c`, if [`NonNull`], must be from [`dav1d_open`] and not be passed to [`dav1d_close`] yet.
/// * `out`, if [`NonNull`], is valid to [`ptr::write`] to.
#[no_mangle]
pub unsafe extern "C" fn dav1d_get_decode_error_data_props(
    c: Option<Dav1dContext>,
    out: Option<NonNull<Dav1dDataProps>>,
) -> Dav1dResult {
    (|| {
        let c = validate_input!(c.ok_or(Rav1dError::InvalidArgument))?;
        let out = validate_input!(out.ok_or(Rav1dError::InvalidArgument))?;
        // SAFETY: `c` is from `dav1d_open` and thus from `RawArc::from_arc`.
        // It has not yet been passed to `dav1d_close` and thus not to `RawArc::into_arc` yet.
        let c = unsafe { c.as_ref() };
        let state = &mut *c.state.try_lock().unwrap();
        let props_rust = mem::take(&mut state.cached_error_props);
        let props_c = props_rust.into();
        // SAFETY: `out` is safety to write to.
        unsafe { out.as_ptr().write(props_c) };
        Ok(())
    })()
    .into()
}

/// # Safety
///
/// * `p`, if [`NonNull`], must be valid to [`ptr::read`] from and [`ptr::write`] to.
#[no_mangle]
pub unsafe extern "C" fn dav1d_picture_unref(p: Option<NonNull<Dav1dPicture>>) {
    let Ok(p) = validate_input!(p.ok_or(())) else {
        return;
    };
    // SAFETY: `p` is safe to read from.
    let p_c = unsafe { p.as_ptr().read() };
    let mut p_rust = p_c.to::<Rav1dPicture>();
    let _ = mem::take(&mut p_rust);
    let p_c = p_rust.into();
    // SAFETY: `p` is safe to write to.
    unsafe { p.as_ptr().write(p_c) };
}

/// # Safety
///
/// * `buf`, if [`NonNull`], is valid to [`ptr::write`] to.
///   After this call, `buf.data` will be an allocated slice of length `sz`.
#[no_mangle]
pub unsafe extern "C" fn dav1d_data_create(buf: Option<NonNull<Dav1dData>>, sz: usize) -> *mut u8 {
    || -> Rav1dResult<*mut u8> {
        let buf = validate_input!(buf.ok_or(Rav1dError::InvalidArgument))?;
        validate_input!((sz <= usize::MAX / 2, Rav1dError::InvalidArgument))?;
        let data = Rav1dData::create(sz)?;
        let data = data.to::<Dav1dData>();
        let ptr = data
            .data
            .map(|ptr| ptr.as_ptr())
            .unwrap_or_else(ptr::null_mut);
        // SAFETY: `buf` is safe to write to.
        unsafe { buf.as_ptr().write(data) };
        Ok(ptr)
    }()
    .unwrap_or_else(|_| ptr::null_mut())
}

/// # Safety
///
/// * `buf`, if [`NonNull`], is valid to [`ptr::write`] to.
/// * `ptr`, if [`NonNull`], is the start of a `&[u8]` slice of length `sz`.
/// * `ptr`'s slice must be valid to dereference until `free_callback` is called on it, which must deallocate it.
#[no_mangle]
pub unsafe extern "C" fn dav1d_data_wrap(
    buf: Option<NonNull<Dav1dData>>,
    ptr: Option<NonNull<u8>>,
    sz: usize,
    free_callback: Option<FnFree>,
    user_data: Option<SendSyncNonNull<c_void>>,
) -> Dav1dResult {
    || -> Rav1dResult {
        let buf = validate_input!(buf.ok_or(Rav1dError::InvalidArgument))?;
        let ptr = validate_input!(ptr.ok_or(Rav1dError::InvalidArgument))?;
        let free = validate_input!(free_callback.ok_or(Rav1dError::InvalidArgument))?;
        validate_input!((sz <= usize::MAX / 2, Rav1dError::InvalidArgument))?;
        // SAFETY: `ptr` is the start of a `&[u8]` slice of length `sz`.
        let data = unsafe { slice::from_raw_parts(ptr.as_ptr(), sz) };
        // SAFETY: `ptr`, and thus `data`, is valid to dereference until `free_callback` is called on it, which deallocates it.
        let data = unsafe { CBox::new(data.into(), free, user_data) };
        let data = Rav1dData::wrap(CRef::C(data))?;
        let data_c = data.into();
        // SAFETY: `buf` is safe to write to.
        unsafe { buf.as_ptr().write(data_c) };
        Ok(())
    }()
    .into()
}

/// # Safety
///
/// * `buf`, if [`NonNull`], is valid to [`ptr::read`] from and [`ptr::write`] to.
/// * `user_data`, if [`NonNull`], is valid to dereference until `free_callback` is called on it, which must deallocate it.
#[no_mangle]
pub unsafe extern "C" fn dav1d_data_wrap_user_data(
    buf: Option<NonNull<Dav1dData>>,
    user_data: Option<NonNull<u8>>,
    free_callback: Option<FnFree>,
    cookie: Option<SendSyncNonNull<c_void>>,
) -> Dav1dResult {
    || -> Rav1dResult {
        let buf = validate_input!(buf.ok_or(Rav1dError::InvalidArgument))?;
        // Note that `dav1d` doesn't do this check, but they do for the similar [`dav1d_data_wrap`].
        let user_data = validate_input!(user_data.ok_or(Rav1dError::InvalidArgument))?;
        let free = validate_input!(free_callback.ok_or(Rav1dError::InvalidArgument))?;
        // SAFETY: `buf` is safe to read from.
        let data_c = unsafe { buf.as_ptr().read() };
        let mut data = data_c.to::<Rav1dData>();
        // SAFETY: `user_data` is valid to dereference until `free_callback` is called on it, which deallocates it.
        let user_data = unsafe { CBox::new(user_data, free, cookie) };
        data.wrap_user_data(CRef::C(user_data))?;
        let data_c = data.into();
        // SAFETY: `buf` is safe to write to.
        unsafe { buf.as_ptr().write(data_c) };
        Ok(())
    }()
    .into()
}

/// # Safety
///
/// * `buf`, if [`NonNull`], is safe to [`ptr::read`] from and [`ptr::write`] from.
#[no_mangle]
pub unsafe extern "C" fn dav1d_data_unref(buf: Option<NonNull<Dav1dData>>) {
    let buf = validate_input!(buf.ok_or(()));
    let Ok(mut buf) = buf else { return };
    // SAFETY: `buf` is safe to read from and write to.
    let buf = unsafe { buf.as_mut() };
    let _ = mem::take(buf).to::<Rav1dData>();
}

/// # Safety
///
/// * `props`, if [`NonNull`], is safe to [`ptr::read`] from and [`ptr::write`] from.
#[no_mangle]
pub unsafe extern "C" fn dav1d_data_props_unref(props: Option<NonNull<Dav1dDataProps>>) {
    let props = validate_input!(props.ok_or(()));
    let Ok(mut props) = props else { return };
    // SAFETY: `props` is safe to read from and write to.
    let props = unsafe { props.as_mut() };
    let _ = mem::take(props).to::<Rav1dDataProps>();
}
