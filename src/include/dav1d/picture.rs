#![deny(unsafe_op_in_unsafe_fn)]

use std::ffi::{c_int, c_void};
use std::hint::assert_unchecked;
use std::marker::PhantomData;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::Arc;
use std::{array, mem};

use crate::c_types::{ptrdiff_t, uintptr_t};
use to_method::To as _;
use zerocopy::{AsBytes, FromBytes, FromZeroes};

use crate::c_arc::RawArc;
use crate::disjoint_mut::{
    AsMutPtr, DisjointImmutGuard, DisjointMut, DisjointMutGuard, SliceBounds,
};
use crate::error::{Dav1dResult, Rav1dError, Rav1dResult};
use crate::ffi_safe::FFISafe;
use crate::include::common::bitdepth::BitDepth;
use crate::include::common::validate::validate_input;
use crate::include::dav1d::common::{Dav1dDataProps, Rav1dDataProps};
use crate::include::dav1d::headers::{
    DRav1d, Dav1dFrameHeader, Dav1dITUTT35, Dav1dPixelLayout, Dav1dSequenceHeader,
    Rav1dContentLightLevel, Rav1dFrameHeader, Rav1dITUTT35, Rav1dMasteringDisplay,
    Rav1dPixelLayout, Rav1dSequenceHeader,
};
use crate::pixels::Pixels;
use crate::send_sync_non_null::SendSyncNonNull;
use crate::strided::Strided;
use crate::with_offset::WithOffset;

// Number of bytes to align AND pad picture memory buffers by, so that SIMD
// implementations can over-read by a few bytes, and use aligned read/write
// instructions.
pub const RAV1D_PICTURE_ALIGNMENT: usize = 64;
pub const DAV1D_PICTURE_ALIGNMENT: usize = RAV1D_PICTURE_ALIGNMENT;

#[derive(Default)]
#[repr(C)]
pub struct Dav1dPictureParameters {
    pub w: c_int,
    pub h: c_int,
    pub layout: Dav1dPixelLayout,
    pub bpc: c_int,
}

// TODO(kkysen) Eventually the [`impl Default`] might not be needed.
#[derive(Clone, Default)]
#[repr(C)]
pub struct Rav1dPictureParameters {
    pub w: c_int,
    pub h: c_int,
    pub layout: Rav1dPixelLayout,
    pub bpc: u8,
}

impl From<Dav1dPictureParameters> for Rav1dPictureParameters {
    fn from(value: Dav1dPictureParameters) -> Self {
        let Dav1dPictureParameters { w, h, layout, bpc } = value;
        Self {
            w,
            h,
            layout: layout.try_into().unwrap(),
            bpc: bpc.try_into().unwrap(),
        }
    }
}

impl From<Rav1dPictureParameters> for Dav1dPictureParameters {
    fn from(value: Rav1dPictureParameters) -> Self {
        let Rav1dPictureParameters { w, h, layout, bpc } = value;
        Self {
            w,
            h,
            layout: layout.into(),
            bpc: bpc.into(),
        }
    }
}

#[derive(Default)]
#[repr(C)]
pub struct Dav1dPicture {
    pub seq_hdr: Option<NonNull<Dav1dSequenceHeader>>,
    pub frame_hdr: Option<NonNull<Dav1dFrameHeader>>,
    pub data: [Option<NonNull<c_void>>; 3],
    pub stride: [ptrdiff_t; 2],
    pub p: Dav1dPictureParameters,
    pub m: Dav1dDataProps,
    pub content_light: Option<NonNull<Rav1dContentLightLevel>>,
    pub mastering_display: Option<NonNull<Rav1dMasteringDisplay>>,
    pub itut_t35: Option<NonNull<Dav1dITUTT35>>,
    pub n_itut_t35: usize,
    pub reserved: [uintptr_t; 4],
    pub frame_hdr_ref: Option<RawArc<DRav1d<Rav1dFrameHeader, Dav1dFrameHeader>>>, // opaque, so we can change this
    pub seq_hdr_ref: Option<RawArc<DRav1d<Rav1dSequenceHeader, Dav1dSequenceHeader>>>, // opaque, so we can change this
    pub content_light_ref: Option<RawArc<Rav1dContentLightLevel>>, // opaque, so we can change this
    pub mastering_display_ref: Option<RawArc<Rav1dMasteringDisplay>>, // opaque, so we can change this
    pub itut_t35_ref: Option<RawArc<DRav1d<Box<[Rav1dITUTT35]>, Box<[Dav1dITUTT35]>>>>, // opaque, so we can change this
    pub reserved_ref: [uintptr_t; 4],
    pub r#ref: Option<RawArc<Rav1dPictureData>>, // opaque, so we can change this
    pub allocator_data: Option<SendSyncNonNull<c_void>>,
}

#[derive(Clone, FromZeroes, FromBytes, AsBytes)]
#[repr(C, align(64))]
pub struct AlignedPixelChunk([u8; RAV1D_PICTURE_ALIGNMENT]);

const _: () = assert!(mem::align_of::<AlignedPixelChunk>() == RAV1D_PICTURE_ALIGNMENT);
const _: () = assert!(mem::size_of::<AlignedPixelChunk>() == RAV1D_PICTURE_ALIGNMENT);

/// The guaranteed length multiple of [`Rav1dPictureDataComponentInner`]s.
/// This is checked and [`assume`]d.
const RAV1D_PICTURE_GUARANTEED_MULTIPLE: usize = 64;

/// Actual [`Rav1dPictureData`]'s components should be multiples of this,
/// as this is guaranteed by [`Rav1dPicAllocator::alloc_picture_callback`],
/// though wrapped buffers may only be [`RAV1D_PICTURE_GUARANTEED_MULTIPLE`].
const RAV1D_PICTURE_MULTIPLE: usize = 64 * 64;

pub struct Rav1dPictureDataComponentInner {
    /// A ptr to the start of this slice of [`BitDepth::Pixel`]s*,
    /// even if [`Self::stride`] is negative.
    ///
    /// It is aligned to [`RAV1D_PICTURE_ALIGNMENT`].
    ptr: NonNull<u8>,

    /// The length of [`Self::ptr`] in [`u8`] bytes.
    ///
    /// It is a multiple of [`RAV1D_PICTURE_GUARANTEED_MULTIPLE`].
    len: usize,

    /// The stride of [`Self::ptr`] in [`u8`] bytes.
    stride: isize,
}

impl Rav1dPictureDataComponentInner {
    /// `len` and `stride` are in terms of [`u8`] bytes.
    ///
    /// # Safety
    ///
    /// `ptr`, `len`, and `stride` must follow the requirements of [`Dav1dPicAllocator::alloc_picture_callback`].
    unsafe fn new(ptr: Option<NonNull<u8>>, len: usize, stride: isize) -> Self {
        let ptr = match ptr {
            None => {
                return Self {
                    // Ensure it is aligned enough.
                    ptr: NonNull::<AlignedPixelChunk>::dangling().cast(),
                    len: 0,
                    stride,
                };
            }
            Some(ptr) => ptr,
        };

        assert!(len != 0); // If `len` was 0, `ptr` should've been `None`.
        assert!(ptr.cast::<AlignedPixelChunk>().is_aligned());

        let ptr = if stride < 0 {
            let ptr = ptr.as_ptr();
            // SAFETY: According to `Dav1dPicAllocator::alloc_picture_callback`,
            // if the `stride` is negative, this is how we get the start of the data.
            // `.offset(-stride)` puts us at one element past the end of the slice,
            // and `.sub(len)` puts us back at the start of the slice.
            let ptr = unsafe { ptr.offset(-stride).sub(len) };
            NonNull::new(ptr).unwrap()
        } else {
            ptr
        };
        // Guaranteed by `Dav1dPicAllocator::alloc_picture_callback`.
        assert!(len % RAV1D_PICTURE_MULTIPLE == 0);
        Self { ptr, len, stride }
    }

    /// # Safety
    ///
    /// As opposed to [`Self::new`], this is safe because `buf` is a `&mut` and thus unique,
    /// so it is sound to further subdivide it into disjoint `&mut`s.
    pub fn wrap_buf<BD: BitDepth>(buf: &mut [BD::Pixel], stride: usize) -> Self {
        let buf = AsBytes::as_bytes_mut(buf);
        let ptr = NonNull::new(buf.as_mut_ptr()).unwrap();
        assert!(ptr.cast::<AlignedPixelChunk>().is_aligned());
        let len = buf.len();
        assert!(len % RAV1D_PICTURE_GUARANTEED_MULTIPLE == 0);
        let stride = (stride * mem::size_of::<BD::Pixel>()) as isize;
        Self { ptr, len, stride }
    }
}

// SAFETY: The pixels are memory an allocator handed out for the picture, which it takes back on
// any thread; they are only reached through the `DisjointMut` around this, whose callers borrow
// disjoint ranges (checked only with debug assertions).
unsafe impl Send for Rav1dPictureDataComponentInner {}
// SAFETY: As for `Send`: a `&Self` reads only `ptr`, `len` and `stride`.
unsafe impl Sync for Rav1dPictureDataComponentInner {}

// SAFETY: We only store the raw pointer, so we never materialize a `&mut`.
unsafe impl AsMutPtr for Rav1dPictureDataComponentInner {
    type Target = u8;

    #[inline] // Inline so callers can see the assume.
    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut Self::Target {
        // SAFETY: Safe to dereference by unsafe preconditions.
        // Since we don't store any `&mut`s, just a raw ptr, we can have a `&Self`.
        let this = unsafe { &*ptr };

        // Assume this so that the compiler knows `ptr` is aligned.
        // Normally we'd store this as a slice so the compiler would know,
        // but since it's a ptr due to `DisjointMut`, we explicitly assume it here.
        // SAFETY: We already checked this in `Self::new`.
        unsafe { assert_unchecked(this.ptr.cast::<AlignedPixelChunk>().is_aligned()) };

        this.ptr.as_ptr()
    }

    #[inline] // Inline so callers can see the assume.
    fn len(&self) -> usize {
        // SAFETY: We already checked this in `Self::new`.
        unsafe { assert_unchecked(self.len % RAV1D_PICTURE_GUARANTEED_MULTIPLE == 0) };
        self.len
    }
}

pub struct Rav1dPictureDataComponent(DisjointMut<Rav1dPictureDataComponentInner>);

impl Pixels for Rav1dPictureDataComponent {
    fn byte_len(&self) -> usize {
        self.0.len()
    }

    fn as_byte_mut_ptr(&self) -> *mut u8 {
        self.0.as_mut_ptr()
    }
}

impl Strided for Rav1dPictureDataComponent {
    fn stride(&self) -> isize {
        // SAFETY: We're only accessing the `stride` fields, not `ptr`.
        unsafe { (*self.0.inner()).stride }
    }
}

impl Rav1dPictureDataComponent {
    pub fn wrap_buf<BD: BitDepth>(buf: &mut [BD::Pixel], stride: usize) -> Self {
        Self(DisjointMut::new(
            Rav1dPictureDataComponentInner::wrap_buf::<BD>(buf, stride),
        ))
    }

    pub fn pixel_offset<BD: BitDepth>(&self) -> usize {
        let stride = self.stride();
        if stride >= 0 {
            return 0;
        }
        BD::pxstride(self.byte_len() - (-stride) as usize)
    }

    pub fn with_offset<BD: BitDepth>(&self) -> Rav1dPictureDataComponentOffset<'_> {
        Rav1dPictureDataComponentOffset {
            data: self,
            offset: self.pixel_offset::<BD>(),
        }
    }

    /// Strided ptr to [`u8`] bytes.
    fn as_strided_byte_mut_ptr(&self) -> *mut u8 {
        let ptr = self.0.as_mut_ptr();
        let stride = self.stride();
        if stride < 0 {
            // SAFETY: This puts `ptr` one element past the end of the slice of pixels.
            let ptr = unsafe { ptr.add(self.byte_len()) };
            // SAFETY: `stride` is negative and `-stride < len`, so this should stay in bounds.
            let ptr = unsafe { ptr.offset(stride) };
            ptr
        } else {
            ptr
        }
    }

    /// Strided ptr to [`BitDepth::Pixel`]s.
    pub fn as_strided_mut_ptr<BD: BitDepth>(&self) -> *mut BD::Pixel {
        // SAFETY: Transmutation is safe because we verify this with `zerocopy` in `Self::slice`.
        self.as_strided_byte_mut_ptr().cast()
    }

    /// Strided ptr to [`BitDepth::Pixel`]s.
    pub fn as_strided_ptr<BD: BitDepth>(&self) -> *const BD::Pixel {
        self.as_strided_mut_ptr::<BD>().cast_const()
    }

    pub(crate) fn as_dav1d(&self) -> Option<NonNull<c_void>> {
        if self.byte_len() == 0 {
            None
        } else {
            NonNull::new(self.as_strided_byte_mut_ptr().cast())
        }
    }

    /// The component's bytes, for [`Self::put_bytes`].
    pub fn bytes(&self) -> Vec<u8> {
        self.0.index(..).to_vec()
    }

    pub fn byte_slice(&self) -> DisjointImmutGuard<'_, Rav1dPictureDataComponentInner, [u8]> {
        self.0.index(..)
    }

    pub fn put_bytes(&self, bytes: &[u8]) {
        self.0.index_mut(..).copy_from_slice(bytes);
    }

    pub fn copy_from(&self, src: &Self) {
        let dst = &mut *self.0.index_mut(..);
        let src = &*src.0.index(..);
        dst.clone_from_slice(src);
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn index<'a, BD: BitDepth>(
        &'a self,
        index: usize,
    ) -> DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, BD::Pixel> {
        self.0.element_as(index)
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn index_mut<'a, BD: BitDepth>(
        &'a self,
        index: usize,
    ) -> DisjointMutGuard<'a, Rav1dPictureDataComponentInner, BD::Pixel> {
        self.0.mut_element_as(index)
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn slice<'a, BD, I>(
        &'a self,
        index: I,
    ) -> DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>
    where
        BD: BitDepth,
        I: SliceBounds,
    {
        self.0.slice_as(index)
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn slice_mut<'a, BD, I>(
        &'a self,
        index: I,
    ) -> DisjointMutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>
    where
        BD: BitDepth,
        I: SliceBounds,
    {
        self.0.mut_slice_as(index)
    }
}

pub type Rav1dPictureDataComponentOffset<'a> = WithOffset<&'a Rav1dPictureDataComponent>;
pub type FFISafeRav1dPictureDataComponentOffset<'a> =
    WithOffset<*const FFISafe<'a, Rav1dPictureDataComponent>>;

impl<'a> Rav1dPictureDataComponentOffset<'a> {
    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn index<BD: BitDepth>(
        &self,
    ) -> DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, BD::Pixel> {
        self.data.index::<BD>(self.offset)
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn index_mut<BD: BitDepth>(
        &self,
    ) -> DisjointMutGuard<'a, Rav1dPictureDataComponentInner, BD::Pixel> {
        self.data.index_mut::<BD>(self.offset)
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn slice<BD: BitDepth>(
        &self,
        len: usize,
    ) -> DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]> {
        self.data.slice::<BD, _>((self.offset.., ..len))
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn slice_mut<BD: BitDepth>(
        &self,
        len: usize,
    ) -> DisjointMutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]> {
        self.data.slice_mut::<BD, _>((self.offset.., ..len))
    }

    /// The block of `rows` and `cols` around this offset, in pixels, for [`PicBlock`]'s rows.
    /// Panics unless the footprint lies in the component or is empty.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn block<BD: BitDepth>(self, rows: Range<isize>, cols: Range<isize>) -> PicBlock<'a, BD> {
        self.block_of(rows, cols)
    }

    /// [`Self::block`] whose rows skip their release check against the footprint's pixels.
    ///
    /// # Safety
    ///
    /// Every row borrowed from the block lies in its footprint: `y` in `rows`, `x..x + n` in
    /// `cols`. Debug builds check this on every row.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub unsafe fn block_unchecked<BD: BitDepth>(
        self,
        rows: Range<isize>,
        cols: Range<isize>,
    ) -> PicRows<'a, BD> {
        self.block_of(rows, cols)
    }

    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    fn block_of<BD: BitDepth, const CHECKED: bool>(
        self,
        rows: Range<isize>,
        cols: Range<isize>,
    ) -> PicBlock<'a, BD, CHECKED> {
        #[inline(never)]
        #[cfg_attr(debug_assertions, track_caller)]
        fn out_of_bounds(rows: Range<isize>, cols: Range<isize>, at: usize, len: usize) -> ! {
            panic!("block rows {rows:?} columns {cols:?} at pixel offset {at} out of range for pixel length {len}");
        }

        /// Where row `y`'s columns start; `offset` wraps below 0 as `WithOffset`'s arithmetic does.
        fn row_start(offset: usize, stride: isize, y: isize, x: isize) -> Option<isize> {
            (offset as isize)
                .checked_add(y.checked_mul(stride)?)?
                .checked_add(x)
        }

        let stride = self.pixel_stride::<BD>();
        let len = self.data.pixel_len::<BD>();
        // Row starts are linear in `y`, so the first and the last row bound all the others.
        // `(lo, hi)`: the pixels from the first row's start to the last row's end.
        let range = if rows.is_empty() || cols.is_empty() {
            Some((self.offset as isize, self.offset as isize))
        } else {
            match (
                row_start(self.offset, stride, rows.start, cols.start),
                row_start(self.offset, stride, rows.end - 1, cols.start),
                cols.end.checked_sub(cols.start),
            ) {
                (Some(a), Some(b), Some(w)) => a
                    .max(b)
                    .checked_add(w)
                    .filter(|&end| a.min(b) >= 0 && end as usize <= len)
                    .map(|end| (a.min(b), end)),
                _ => None,
            }
        };
        let Some((lo, hi)) = range else {
            out_of_bounds(rows, cols, self.offset, len);
        };
        #[cfg(debug_assertions)]
        let _ = (lo, hi);
        PicBlock {
            #[cfg(debug_assertions)]
            at: self,
            #[cfg(not(debug_assertions))]
            origin: self.data.as_mut_ptr::<BD>().wrapping_add(self.offset),
            #[cfg(not(debug_assertions))]
            lo: lo.wrapping_sub(self.offset as isize),
            #[cfg(not(debug_assertions))]
            span: (hi - lo) as usize,
            stride,
            rows: (rows.start, rows.end),
            cols: (cols.start, cols.end),
            phantom: PhantomData,
        }
    }
}

/// The rows of a block in a picture component, its whole footprint checked once by
/// [`Rav1dPictureDataComponentOffset::block`] to lie in the component's allocation.
///
/// Release builds then check a row with one compare, against the pixels from the footprint's
/// first row to its last, so a row never leaves the allocation; without `CHECKED`
/// ([`Rav1dPictureDataComponentOffset::block_unchecked`]) not even that. Like
/// [`DisjointMut::index_mut`] in release, the rest relies on correct use, which only debug
/// builds check:
/// - the `n` pixels at `(x, y)` lie in the footprint: `y` in its rows, `x..x + n` in its
///   columns. Kernels declare the footprint from the block size they loop over.
/// - a borrowed row overlaps no mutable borrow of the same pixels while held.
///
/// With `debug_assertions` each row is checked against the footprint and borrowed on its own
/// through the [`DisjointMut`], so its tracker sees one row at a time, never the whole block.
#[derive(Clone, Copy)]
pub struct PicBlock<'a, BD: BitDepth, const CHECKED: bool = true> {
    #[cfg(debug_assertions)]
    at: Rav1dPictureDataComponentOffset<'a>,
    #[cfg(not(debug_assertions))]
    /// Pixel `(0, 0)` of the block; only the footprint is known to be in bounds.
    origin: *mut BD::Pixel,
    /// The footprint's pixels in the allocation, `origin + lo..origin + lo + span`.
    #[cfg(not(debug_assertions))]
    lo: isize,
    #[cfg(not(debug_assertions))]
    span: usize,
    stride: isize,
    rows: (isize, isize),
    cols: (isize, isize),
    phantom: PhantomData<(&'a Rav1dPictureDataComponent, BD)>,
}

/// A [`PicBlock`] whose rows the kernel keeps in the footprint (checked in debug builds only).
pub type PicRows<'a, BD> = PicBlock<'a, BD, false>;

#[cfg(not(debug_assertions))]
#[inline(never)]
#[cold]
fn row_out_of_bounds() -> ! {
    panic!("row out of its block's pixels");
}

#[cfg(debug_assertions)]
pub type PicRow<'a, BD> =
    DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, [<BD as BitDepth>::Pixel]>;
#[cfg(not(debug_assertions))]
pub type PicRow<'a, BD> = &'a [<BD as BitDepth>::Pixel];

#[cfg(debug_assertions)]
pub type PicRowMut<'a, BD> =
    DisjointMutGuard<'a, Rav1dPictureDataComponentInner, [<BD as BitDepth>::Pixel]>;
#[cfg(not(debug_assertions))]
pub type PicRowMut<'a, BD> = &'a mut [<BD as BitDepth>::Pixel];

impl<'a, BD: BitDepth, const CHECKED: bool> PicBlock<'a, BD, CHECKED> {
    /// The offset of `n` pixels at `(x, y)` from pixel `(0, 0)`.
    #[inline(always)]
    #[cfg_attr(debug_assertions, track_caller)]
    fn at(&self, y: isize, x: isize, n: usize) -> isize {
        let ((top, bottom), (left, right)) = (self.rows, self.cols);
        let inside = y >= top && y < bottom && x >= left && x <= right && n <= (right - x) as usize;
        debug_assert!(
            inside,
            "{n} pixels at ({x}, {y}) outside block rows {top}..{bottom} columns {left}..{right}"
        );
        let i = y.wrapping_mul(self.stride).wrapping_add(x);
        #[cfg(not(debug_assertions))]
        if CHECKED
            && (i.wrapping_sub(self.lo) as usize > self.span.wrapping_sub(n) || n > self.span)
        {
            row_out_of_bounds();
        }
        i
    }

    /// `n` pixels of row `y` from column `x`.
    #[inline(always)]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn row(&self, y: isize, x: isize, n: usize) -> PicRow<'a, BD> {
        let i = self.at(y, x, n);
        #[cfg(debug_assertions)]
        return (self.at + i).slice::<BD>(n);
        // SAFETY: `block` checked that the footprint's pixels lie in the allocation, and `at`
        // that `i`'s `n` pixels lie in the footprint's, or `block_unchecked`'s caller that
        // they lie in the footprint.
        #[cfg(not(debug_assertions))]
        unsafe {
            std::slice::from_raw_parts(self.origin.wrapping_offset(i), n)
        }
    }

    /// `n` pixels of row `y` from column `x`, to write.
    #[inline(always)]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn row_mut(&self, y: isize, x: isize, n: usize) -> PicRowMut<'a, BD> {
        let i = self.at(y, x, n);
        #[cfg(debug_assertions)]
        return (self.at + i).slice_mut::<BD>(n);
        // SAFETY: As in `row`; no other borrow of these pixels is held (`DisjointMut`'s rule).
        #[cfg(not(debug_assertions))]
        unsafe {
            std::slice::from_raw_parts_mut(self.origin.wrapping_offset(i), n)
        }
    }
}

pub struct Rav1dPictureData {
    pub data: [Rav1dPictureDataComponent; 3],
    pub(crate) allocator_data: Option<SendSyncNonNull<c_void>>,
    pub(crate) allocator: Rav1dPicAllocator,
}

impl Drop for Rav1dPictureData {
    fn drop(&mut self) {
        let Self {
            data,
            allocator_data,
            allocator,
        } = self;
        allocator.dealloc_picture_data(data, *allocator_data);
    }
}

// TODO(kkysen) Eventually the [`impl Default`] might not be needed.
// It's needed currently for a [`mem::take`] that simulates a move,
// but once everything is Rusty, we may not need to clear the `dst` anymore.
// This also applies to the `#[derive(Default)]`
// on [`Rav1dPictureParameters`] and [`Rav1dPixelLayout`].
#[derive(Clone, Default)]
#[repr(C)]
pub struct Rav1dPicture {
    pub seq_hdr: Option<Arc<DRav1d<Rav1dSequenceHeader, Dav1dSequenceHeader>>>,
    pub frame_hdr: Option<Arc<DRav1d<Rav1dFrameHeader, Dav1dFrameHeader>>>,
    pub data: Option<Arc<Rav1dPictureData>>,
    pub stride: [ptrdiff_t; 2],
    pub p: Rav1dPictureParameters,
    pub m: Rav1dDataProps,
    pub content_light: Option<Arc<Rav1dContentLightLevel>>,
    pub mastering_display: Option<Arc<Rav1dMasteringDisplay>>,
    pub itut_t35: Arc<DRav1d<Box<[Rav1dITUTT35]>, Box<[Dav1dITUTT35]>>>,
}

impl From<Dav1dPicture> for Rav1dPicture {
    fn from(value: Dav1dPicture) -> Self {
        let Dav1dPicture {
            seq_hdr: _,
            frame_hdr: _,
            data: _,
            stride,
            p,
            m,
            content_light: _,
            mastering_display: _,
            itut_t35: _,
            n_itut_t35: _,
            reserved: _,
            frame_hdr_ref,
            seq_hdr_ref,
            content_light_ref,
            mastering_display_ref,
            itut_t35_ref,
            reserved_ref: _,
            r#ref: data_ref,
            allocator_data: _,
        } = value;
        Self {
            // We don't `.update_rav1d()` [`Rav1dSequenceHeader`] because it's meant to be read-only.
            seq_hdr: seq_hdr_ref.map(|raw| {
                // SAFETY: `raw` came from [`RawArc::from_arc`].
                unsafe { raw.into_arc() }
            }),
            // We don't `.update_rav1d()` [`Rav1dFrameHeader`] because it's meant to be read-only.
            frame_hdr: frame_hdr_ref.map(|raw| {
                // SAFETY: `raw` came from [`RawArc::from_arc`].
                unsafe { raw.into_arc() }
            }),
            data: data_ref.map(|raw| {
                // SAFETY: `raw` came from [`RawArc::from_arc`].
                unsafe { raw.into_arc() }
            }),
            stride,
            p: p.into(),
            m: m.into(),
            content_light: content_light_ref.map(|raw| {
                // SAFETY: `raw` came from [`RawArc::from_arc`].
                unsafe { raw.into_arc() }
            }),
            mastering_display: mastering_display_ref.map(|raw| {
                // Safety: `raw` came from [`RawArc::from_arc`].
                unsafe { raw.into_arc() }
            }),
            // We don't `.update_rav1d` [`Rav1dITUTT35`] because never read it.
            itut_t35: itut_t35_ref
                .map(|raw| {
                    // SAFETY: `raw` came from [`RawArc::from_arc`].
                    unsafe { raw.into_arc() }
                })
                .unwrap_or_default(),
        }
    }
}

impl From<Rav1dPicture> for Dav1dPicture {
    fn from(value: Rav1dPicture) -> Self {
        let Rav1dPicture {
            seq_hdr,
            frame_hdr,
            data,
            stride,
            p,
            m,
            content_light,
            mastering_display,
            itut_t35,
        } = value;
        Self {
            // [`DRav1d::from_rav1d`] is called right after [`parse_seq_hdr`].
            seq_hdr: seq_hdr.as_ref().map(|arc| (&arc.as_ref().dav1d).into()),
            // [`DRav1d::from_rav1d`] is called in [`parse_frame_hdr`].
            frame_hdr: frame_hdr.as_ref().map(|arc| (&arc.as_ref().dav1d).into()),
            data: data
                .as_ref()
                .map(|arc| arc.data.each_ref().map(|data| data.as_dav1d()))
                .unwrap_or_default(),
            stride,
            p: p.into(),
            m: m.into(),
            content_light: content_light.as_ref().map(|arc| arc.as_ref().into()),
            mastering_display: mastering_display.as_ref().map(|arc| arc.as_ref().into()),
            // [`DRav1d::from_rav1d`] is called in [`rav1d_parse_obus`].
            itut_t35: Some(NonNull::new(itut_t35.dav1d.as_ptr().cast_mut()).unwrap()),
            n_itut_t35: itut_t35.len(),
            reserved: Default::default(),
            frame_hdr_ref: frame_hdr.map(RawArc::from_arc),
            seq_hdr_ref: seq_hdr.map(RawArc::from_arc),
            content_light_ref: content_light.map(RawArc::from_arc),
            mastering_display_ref: mastering_display.map(RawArc::from_arc),
            itut_t35_ref: Some(itut_t35).map(RawArc::from_arc),
            reserved_ref: Default::default(),
            // Order flipped so that the borrow comes before the move.
            allocator_data: data.as_ref().and_then(|arc| arc.allocator_data),
            r#ref: data.map(RawArc::from_arc),
        }
    }
}

impl Rav1dPicture {
    pub fn lf_offsets<BD: BitDepth>(&self, y: c_int) -> [Rav1dPictureDataComponentOffset<'_>; 3] {
        // Init loopfilter offsets. Point the chroma offsets in 4:0:0 to the luma
        // plane here to avoid having additional in-loop branches in various places.
        // We never use those values, so it doesn't really matter what they point
        // at, as long as the offsets are valid.
        let layout = self.p.layout;
        let has_chroma = layout != Rav1dPixelLayout::I400;
        let data = &self.data.as_ref().unwrap().data;
        array::from_fn(|i| {
            let data = &data[has_chroma as usize * i];
            let ss_ver = layout == Rav1dPixelLayout::I420 && i != 0;
            data.with_offset::<BD>() + (y as isize * data.pixel_stride::<BD>() >> ss_ver as u8)
        })
    }
}

#[derive(Clone)]
#[repr(C)]
pub struct Dav1dPicAllocator {
    /// Custom data to pass to the allocator callbacks.
    ///
    /// # Safety
    ///
    /// All accesses to [`Self::cookie`] must be thread-safe
    /// (i.e. [`Self::cookie`] must be [`Send`]` + `[`Sync`]).
    ///
    /// If used from Rust, [`Self::cookie`] is a [`SendSyncNonNull`],
    /// whose constructors ensure this [`Send`]` + `[`Sync`] safety.
    pub cookie: Option<SendSyncNonNull<c_void>>,

    /// Allocate the picture buffer based on the [`Dav1dPictureParameters`].
    ///
    /// [`data`]`[0]`, [`data`]`[1]` and [`data`]`[2]`
    /// must be [`DAV1D_PICTURE_ALIGNMENT`]-byte aligned
    /// and with a pixel width/height multiple of 128 pixels.
    /// Any allocated memory area should also be padded by [`DAV1D_PICTURE_ALIGNMENT`] bytes.
    /// [`data`]`[1]` and [`data`]`[2]` must share the same [`stride`]`[1]`.
    ///
    /// # Safety
    ///
    /// See [`Self::cookie`]'s safety requirements.
    ///
    /// ### Additional `rav1d` requirement:
    ///
    /// The allocated data must be initialized.
    /// If newly (e.x. not reused) allocated data is zero initialized using OS APIs,
    /// it is possible for this to not be slower than an uninitialized allocation.
    /// For example, see `dav1d_default_picture_alloc` and `MemPool::pop_init`.
    ///
    /// If the allocated data is not initialized,
    /// it is possible there will be reads of uninitialized data.
    /// `rav1d` should not read this data before writing to it first,
    /// but it does not guarantee that it does so.
    /// Instead, initializing the allocated data guarantees all uses of it will be sound.
    ///
    /// # Args
    ///
    /// * `pic`: The picture to allocate the buffer for.
    ///     The callback needs to fill the picture
    ///     [`data`]`[0]`, [`data`]`[1]`, [`data`]`[2]`,
    ///     [`stride`]`[0]`, and [`stride`]`[1]`.
    ///     The allocator can fill the pic [`allocator_data`] pointer
    ///     with a custom pointer that will be passed to
    ///     [`release_picture_callback`].
    ///
    ///     The only fields of `pic` that will be already set are:
    ///     * [`Dav1dPicture::p`]
    ///     * [`Dav1dPicture::seq_hdr`]
    ///     * [`Dav1dPicture::frame_hdr`]
    ///
    ///     This is not a change from the original `DAV1D_API`,
    ///     just a clarification of it.
    ///
    /// * `cookie`: Custom pointer passed to all calls.
    ///
    /// *Note*: No fields other than [`data`], [`stride`] and [`allocator_data`]
    /// must be filled by this callback.
    ///
    /// # Return
    ///
    /// 0 on success. A negative `DAV1D_ERR` value on error.
    /// <!--- TODO(kkysen) Translate `DAV1D_ERR` -->
    ///
    /// [`data`]: Dav1dPicture::data
    /// [`stride`]: Dav1dPicture::data
    /// [`allocator_data`]: Dav1dPicture::allocator_data
    /// [`release_picture_callback`]: Self::release_picture_callback
    pub alloc_picture_callback: Option<
        unsafe extern "C" fn(
            pic: *mut Dav1dPicture,
            cookie: Option<SendSyncNonNull<c_void>>,
        ) -> Dav1dResult,
    >,

    /// Release the picture buffer.
    ///
    /// # Safety
    ///
    /// If frame threading is used, accesses to `cookie` must be thread-safe.
    ///
    /// If frame threading is used, this function may be called by the main thread
    /// (the thread which calls [`dav1d_get_picture`]),
    /// or any of the frame threads and thus must be thread-safe.
    /// If frame threading is not used, this function will only be called on the main thread.
    ///
    /// # Args
    ///
    /// * `pic`: The picture that was filled by [`alloc_picture_callback`].
    ///
    ///     The only fields of `pic` that will be set are
    ///     the ones allocated by [`Self::alloc_picture_callback`]:
    ///     * [`Dav1dPicture::data`]
    ///     * [`Dav1dPicture::allocator_data`]
    ///
    ///     NOTE: This is a slight change from the original `DAV1D_API`, which was underspecified.
    ///     However, all known uses of this API follow this already:
    ///     * `libdav1d`: [`dav1d_default_picture_release`](https://code.videolan.org/videolan/dav1d/-/blob/16ed8e8b99f2fcfffe016e929d3626e15267ad3e/src/picture.c#L85-87)
    ///     * `dav1d`: [`picture_release`](https://code.videolan.org/videolan/dav1d/-/blob/16ed8e8b99f2fcfffe016e929d3626e15267ad3e/tools/dav1d.c#L180-182)
    ///     * `dav1dplay`: [`placebo_release_pic`](https://code.videolan.org/videolan/dav1d/-/blob/16ed8e8b99f2fcfffe016e929d3626e15267ad3e/examples/dp_renderer_placebo.c#L375-383)
    ///     * `libplacebo`: [`pl_release_dav1dpicture`](https://github.com/haasn/libplacebo/blob/34e019bfedaa5a64f268d8f9263db352c0a8f67f/src/include/libplacebo/utils/dav1d_internal.h#L594-L607)
    ///     * `ffmpeg`: [`libdav1d_picture_release`](https://github.com/FFmpeg/FFmpeg/blob/00b288da73f45acb78b74bcc40f73c7ba1fff7cb/libavcodec/libdav1d.c#L124-L129)
    ///
    ///     Making this API safe without this slight tightening of the API
    ///     [is very difficult](https://github.com/memorysafety/rav1d/pull/685#discussion_r1458171639).
    ///
    /// * `cookie`: Custom pointer passed to all calls.
    ///
    /// [`dav1d_get_picture`]: crate::lib::dav1d_get_picture
    /// [`alloc_picture_callback`]: Self::alloc_picture_callback
    pub release_picture_callback: Option<
        unsafe extern "C" fn(pic: *mut Dav1dPicture, cookie: Option<SendSyncNonNull<c_void>>) -> (),
    >,
}

#[derive(Clone)]
#[repr(C)]
pub struct Rav1dPicAllocator {
    /// See [`Dav1dPicAllocator::cookie`].
    ///
    /// # Safety
    ///
    /// If [`Self::is_default`]`()`, then this cookie is a reference to
    /// [`Rav1dContext::picture_pool`], a `&Arc<MemPool<u8>`.
    /// Thus, its lifetime is that of `&c.picture_pool`,
    /// so the lifetime of the `&`[`Rav1dContext`].
    /// This is used from `dav1d_default_picture_alloc`
    /// ([`Self::default`]`().alloc_picture_callback`),
    /// which is called from [`Self::alloc_picture_data`],
    /// which is called further up on the call stack with a `&`[`Rav1dContext`].
    /// Thus, the lifetime will always be valid where used.
    ///
    /// Note that this is an `&Arc<MemPool<u8>` turned into a raw pointer,
    /// not an [`Arc::into_raw`] of that [`Arc`].
    /// This is because storing the [`Arc`] would require C to
    /// free data owned by a [`Dav1dPicAllocator`] potentially,
    /// which it may not do, as there are no current APIs for doing so.
    ///
    /// [`Rav1dContext::picture_pool`]: crate::internal::Rav1dContext::picture_pool
    /// [`Rav1dContext`]: crate::internal::Rav1dContext
    pub cookie: Option<SendSyncNonNull<c_void>>,

    /// See [`Dav1dPicAllocator::alloc_picture_callback`].
    ///
    /// # Safety
    ///
    /// `pic` is passed as a `&mut`.
    ///
    /// If frame threading is used, accesses to [`Self::cookie`] must be thread-safe,
    /// i.e. [`Self::cookie`] must be [`Send`]` + `[`Sync`].
    pub alloc_picture_callback: unsafe extern "C" fn(
        pic: *mut Dav1dPicture,
        cookie: Option<SendSyncNonNull<c_void>>,
    ) -> Dav1dResult,

    /// See [`Dav1dPicAllocator::release_picture_callback`].
    ///
    /// # Safety
    ///
    /// `pic` is passed as a `&mut`.
    ///
    /// If frame threading is used, accesses to [`Self::cookie`] must be thread-safe,
    /// i.e. [`Self::cookie`] must be [`Send`]` + `[`Sync`].
    pub release_picture_callback:
        unsafe extern "C" fn(pic: *mut Dav1dPicture, cookie: Option<SendSyncNonNull<c_void>>) -> (),
}

impl TryFrom<Dav1dPicAllocator> for Rav1dPicAllocator {
    type Error = Rav1dError;

    fn try_from(value: Dav1dPicAllocator) -> Result<Self, Self::Error> {
        let Dav1dPicAllocator {
            cookie,
            alloc_picture_callback,
            release_picture_callback,
        } = value;
        Ok(Self {
            cookie,
            alloc_picture_callback: validate_input!(
                alloc_picture_callback.ok_or(Rav1dError::InvalidArgument)
            )?,
            release_picture_callback: validate_input!(
                release_picture_callback.ok_or(Rav1dError::InvalidArgument)
            )?,
        })
    }
}

impl From<Rav1dPicAllocator> for Dav1dPicAllocator {
    fn from(value: Rav1dPicAllocator) -> Self {
        let Rav1dPicAllocator {
            cookie,
            alloc_picture_callback,
            release_picture_callback,
        } = value;
        Self {
            cookie,
            alloc_picture_callback: Some(alloc_picture_callback),
            release_picture_callback: Some(release_picture_callback),
        }
    }
}

impl Rav1dPicAllocator {
    pub fn alloc_picture_data(
        &self,
        w: c_int,
        h: c_int,
        seq_hdr: Arc<DRav1d<Rav1dSequenceHeader, Dav1dSequenceHeader>>,
        frame_hdr: Option<Arc<DRav1d<Rav1dFrameHeader, Dav1dFrameHeader>>>,
    ) -> Rav1dResult<Rav1dPicture> {
        let pic = Rav1dPicture {
            p: Rav1dPictureParameters {
                w,
                h,
                layout: seq_hdr.layout,
                bpc: 8 + 2 * seq_hdr.hbd,
            },
            seq_hdr: Some(seq_hdr),
            frame_hdr,
            ..Default::default()
        };
        let mut pic_c = pic.to::<Dav1dPicture>();
        // SAFETY: `pic_c` is a valid `Dav1dPicture` with `data`, `stride`, `allocator_data` unset.
        let result = unsafe { (self.alloc_picture_callback)(&mut pic_c, self.cookie) };
        result.try_to::<Rav1dResult>().unwrap()?;
        // `data`, `stride`, and `allocator_data` are the only fields set by the allocator.
        // Of those, only `data` and `allocator_data` are read through `r#ref`,
        // so we need to read those directly first and allocate the `Arc`.
        let data = pic_c.data;
        let allocator_data = pic_c.allocator_data;
        let mut pic = pic_c.to::<Rav1dPicture>();
        let len = pic.p.pic_len(pic.stride);
        // TODO fallible allocation
        pic.data = Some(Arc::new(Rav1dPictureData {
            data: array::from_fn(|i| {
                let ptr = data[i].map(|ptr| ptr.cast::<u8>());
                let len = len[(i != 0) as usize];
                let stride = pic.stride[(i != 0) as usize];
                // SAFETY: These args come from `Self::alloc_picture_callback`.
                let component = unsafe { Rav1dPictureDataComponentInner::new(ptr, len, stride) };
                Rav1dPictureDataComponent(DisjointMut::new(component))
            }),
            allocator_data,
            allocator: self.clone(),
        }));
        Ok(pic)
    }

    pub fn dealloc_picture_data(
        &self,
        data: &mut [Rav1dPictureDataComponent; 3],
        allocator_data: Option<SendSyncNonNull<c_void>>,
    ) {
        let data = data.each_mut().map(|data| data.as_dav1d());
        let mut pic_c = Dav1dPicture {
            data,
            allocator_data,
            ..Default::default()
        };
        // SAFETY: `pic_c` contains the same `data` and `allocator_data`
        // that `Self::alloc_picture_data` set, which now get deallocated here.
        unsafe {
            (self.release_picture_callback)(&mut pic_c, self.cookie);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::AlignedVec64;
    use crate::include::common::bitdepth::{BitDepth16, BitDepth8};

    const S: usize = 64;
    const H: usize = 6;

    /// `H` rows of `S` pixels, each pixel its index.
    fn plane<BD: BitDepth>() -> AlignedVec64<BD::Pixel> {
        let mut buf = AlignedVec64::<BD::Pixel>::new();
        buf.resize(S * H, 0.into());
        for (i, p) in buf.as_mut_slice().iter_mut().enumerate() {
            *p = (i as u8).into();
        }
        buf
    }

    fn rows_match<BD: BitDepth>(
        b: &PicBlock<BD>,
        buf: &[BD::Pixel],
        at: usize,
        rows: Range<isize>,
        cols: Range<isize>,
    ) {
        for y in rows {
            let n = (cols.end - cols.start) as usize;
            let i = at as isize + y * S as isize + cols.start;
            assert!(*b.row(y, cols.start, n) == buf[i as usize..][..n]);
        }
    }

    fn edges<BD: BitDepth>() {
        let mut buf = plane::<BD>();
        let want = buf.as_slice().to_vec();
        let pic = Rav1dPictureDataComponent::wrap_buf::<BD>(buf.as_mut_slice(), S);
        let at = 3 * S + 3;
        let all = (-3..H as isize - 3, -3..S as isize - 3);
        let b = (pic.with_offset::<BD>() + at).block::<BD>(all.0.clone(), all.1.clone());
        rows_match(&b, &want, at, all.0, all.1);
        let last = (pic.with_offset::<BD>() + (S * H - 1)).block::<BD>(-1..1, -1..1);
        assert!(last.row(0, 0, 1)[0] == want[S * H - 1]);
        let first = pic.with_offset::<BD>().block::<BD>(0..1, 0..1);
        assert!(first.row(0, 0, 1)[0] == want[0]);
    }

    #[test]
    fn block_at_the_edges() {
        edges::<BitDepth8>();
        edges::<BitDepth16>();
    }

    #[test]
    fn block_empty() {
        let mut buf = plane::<BitDepth8>();
        let pic = Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(buf.as_mut_slice(), S);
        pic.with_offset::<BitDepth8>()
            .block::<BitDepth8>(-9..-9, 0..4);
    }

    fn refused(rows: Range<isize>, cols: Range<isize>) {
        let mut buf = plane::<BitDepth16>();
        let pic = Rav1dPictureDataComponent::wrap_buf::<BitDepth16>(buf.as_mut_slice(), S);
        (pic.with_offset::<BitDepth16>() + (3 * S + 3)).block::<BitDepth16>(rows, cols);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn block_one_past_the_end() {
        refused(-3..H as isize - 3, -3..S as isize - 2);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn block_one_row_below() {
        refused(-3..H as isize - 2, 0..1);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn block_one_before_the_start() {
        refused(-3..0, -4..0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn block_one_row_above() {
        refused(-4..0, 0..1);
    }

    /// The component with its rows bottom up, as an allocator with a negative stride hands out.
    fn bottom_up(buf: &mut [u8]) -> Rav1dPictureDataComponent {
        Rav1dPictureDataComponent(DisjointMut::new(Rav1dPictureDataComponentInner {
            ptr: NonNull::new(buf.as_mut_ptr()).unwrap(),
            len: buf.len(),
            stride: -(S as isize),
        }))
    }

    #[test]
    fn block_negative_stride() {
        let mut buf = plane::<BitDepth8>();
        let want = buf.as_slice().to_vec();
        let pic = bottom_up(buf.as_mut_slice());
        let b = pic
            .with_offset::<BitDepth8>()
            .block::<BitDepth8>(0..H as isize, 0..S as isize);
        for y in 0..H {
            assert_eq!(&*b.row(y as isize, 0, S), &want[(H - 1 - y) * S..][..S]);
        }
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn block_negative_stride_one_row_below() {
        let mut buf = plane::<BitDepth8>();
        let pic = bottom_up(buf.as_mut_slice());
        pic.with_offset::<BitDepth8>()
            .block::<BitDepth8>(0..H as isize + 1, 0..1);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn block_negative_stride_one_row_above() {
        let mut buf = plane::<BitDepth8>();
        let pic = bottom_up(buf.as_mut_slice());
        pic.with_offset::<BitDepth8>()
            .block::<BitDepth8>(-1..1, 0..1);
    }

    /// Rows are borrowed one at a time: another borrow may hold the row between them.
    #[test]
    fn block_rows_borrowed_alone() {
        let mut buf = plane::<BitDepth8>();
        let pic = Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(buf.as_mut_slice(), S);
        let b = pic
            .with_offset::<BitDepth8>()
            .block::<BitDepth8>(0..3, 0..8);
        let _middle = pic
            .with_offset::<BitDepth8>()
            .block::<BitDepth8>(1..2, 0..8)
            .row_mut(1, 0, 8);
        let (_a, _c) = (b.row(0, 0, 8), b.row_mut(2, 0, 8));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "overlapping DisjointMut")]
    fn block_rows_overlap_tracked() {
        let mut buf = plane::<BitDepth8>();
        let pic = Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(buf.as_mut_slice(), S);
        let b = pic
            .with_offset::<BitDepth8>()
            .block::<BitDepth8>(0..2, 0..8);
        let _w = b.row_mut(1, 0, 8);
        let _r = b.row(1, 4, 4);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "outside block")]
    fn block_row_outside_the_footprint() {
        let mut buf = plane::<BitDepth8>();
        let pic = Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(buf.as_mut_slice(), S);
        let b = pic
            .with_offset::<BitDepth8>()
            .block::<BitDepth8>(0..2, 0..8);
        let _ = b.row(0, 1, 8);
    }

    #[cfg(not(debug_assertions))]
    #[test]
    #[should_panic(expected = "row out of its block's pixels")]
    fn block_row_past_its_pixels_refused_in_release() {
        let mut buf = plane::<BitDepth8>();
        let pic = Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(buf.as_mut_slice(), S);
        let b = pic
            .with_offset::<BitDepth8>()
            .block::<BitDepth8>(0..1, 0..8);
        let _ = b.row(H as isize, 0, 8);
    }

    #[cfg(not(debug_assertions))]
    #[test]
    #[should_panic(expected = "row out of its block's pixels")]
    fn block_row_before_its_pixels_refused_in_release() {
        let mut buf = plane::<BitDepth8>();
        let pic = Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(buf.as_mut_slice(), S);
        let b = (pic.with_offset::<BitDepth8>() + S).block::<BitDepth8>(0..1, 0..8);
        let _ = b.row(-1, 0, 8);
    }
}
