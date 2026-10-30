# An AV1 still-image encoder in rav1d

Status: design, nothing built yet. Branch `encoder` of marcbachmann/rav1d, from `helpers` (bc62eb90).

## Goal

A Rust AV1 encoder for still images. It replaces libaom 3.15.1 as the AVIF encoder of
image-processing (`src/engine/avif/aom.rs`). It must:

- code at least the quality libaom codes, at the same size, on every image the AVIF gates
  cover, photos and graphics alike;
- be much faster than libaom at the default effort, on one core first, then with helpers;
- **keep memory low. This is a hard requirement, not a nice-to-have.** Today a libaom grid tile
  costs 48 MB plus 48 bytes a pixel, and a picture coded as one frame costs about 47 bytes a
  pixel: a 4000×3000 screenshot is about 600 MB. The encoder streams: what it holds grows with
  the width of the image and the size of the output, not with its area (see [Memory](#memory));
- build without cmake, libclang, llvm-tools, nasm or a C library: rav1d is Rust only.

Being bit-exact with libaom is **not** a goal. The output only has to be a conformant AV1
stream that meets the quality gates (see [Quality](#quality)). Without bit-exactness,
every search can take its own shortcuts, and that is where the speed comes from.

## Non-goals (for now)

- Inter frames, animation (`avis`), motion search, rate control, lookahead.
- Intra block copy: image-processing codes with it off (on text it was 70% smaller and 2.9 dB
  lower than the reference).
- Film grain, superres, 4:2:0/4:2:2 input, 10/12-bit. The data structures stay generic over
  `BitDepth` like the rest of rav1d, so high bit depth can come later without a rewrite.

## What has to be coded

This is what image-processing asks of libaom today, and what the encoder takes over:

| | Colour | Alpha |
|---|---|---|
| Layout | 8-bit 4:4:4 (profile 1), BT.601 matrix, full range, sRGB transfer (13) | 8-bit monochrome (profile 0) |
| Quality | libavif's tune-IQ table, Q 0–100 → cq-level 63–0 (`iq_cq_level`) | `((100 - Q) * 63 + 50) / 100` (`cq_level`) |
| Tuning | tune IQ | tune PSNR |
| Speed | cpu-used `9 - effort`, effort 0–9, default 4 (cpu-used 5) | same |
| Screen content | decided once for the whole image by `aom::Screen`, then forced on each grid tile | same |
| Q100 | cq-level 0. Whether libaom codes that lossless under tune IQ (delta q may lift blocks off qindex 0) is checked in M0. The git history has no lossless AVIF path that was dropped | same |
| Headers | still picture, reduced still-picture header, one key frame | same |

Grid tiles are image-processing's business and stay there: photos are cut into grid tiles of at
most 1400 px, rounded up to whole 8×8 blocks, and pictures (graphics) are coded as one frame.
The encoder codes one frame per call. A whole-frame picture can be large, e.g. a 4000×3000
screenshot, so the encoder must also be fast and lean on big frames.

## Why inside the rav1d fork

An encoder has to reconstruct every block exactly as the decoder will, so it needs the
decoder's whole prediction and reconstruction path. rav1d already has all of it, with dav1d's
assembly (SSE/AVX2/AVX-512, NEON):

| rav1d module | What the encoder uses it for |
|---|---|
| `ipred.rs` (`Rav1dIntraPredDSPContext`): 14 intra predictors incl. directional with edge filter/upsampling, smooth, Paeth, filter-intra, `cfl_ac`/`cfl_pred`, `pal_pred` | prediction during search and for the reconstruction |
| `ipred_prepare.rs` (`rav1d_prepare_intra_edges`), `intra_edge.rs` | edge availability and edge pixels, exactly as the decoder sees them |
| `itx.rs`, `itx_1d.rs` (`Rav1dInvTxfmDSPContext`, all sizes and types incl. WHT) | reconstruction after quantization |
| `loopfilter.rs`, `cdef.rs`, `looprestoration.rs` (DSP contexts) | evaluating filter candidates in the filter searches |
| `cdf.rs` (default CDFs per qindex, `rav1d_cdf_thread_init_static`) | starting CDFs for the writer and the rate model |
| `scan.rs`, `tables.rs`, `levels.rs`, `dequant_tables.rs`, `qm.rs`, `pal.rs` | scans, context tables, block/transform geometry, dequantizers, quantization matrices |
| `msac.rs`, `decode.rs`, `recon.rs`, `obu.rs` | not reused for encoding; used in tests to decode what we write and check it |

These modules are `mod`, not `pub mod`, in `lib.rs`. An encoder in a separate crate would need
all of them made public, which is a large API surface for the fork to carry. So the encoder lives
in the crate as `src/enc/`, behind a cargo feature `encoder`, and uses them as `crate::…`.

Not reusable as they are: the filter walkers (`lf_apply.rs`, `cdef_apply.rs`, `lr_apply.rs`) and
`recon.rs` are tied to the decoder's `Rav1dFrameData`/`Rav1dTaskContext`. The encoder gets its own
walkers that call the same DSP functions. The DSP calls take a
`Rav1dPictureDataComponentOffset`, so the encoder's reconstruction frame is a rav1d picture
buffer. Making that work is the first thing to check in M1.

## What we write

| Part | libaom reference (BSD-2) | Notes |
|---|---|---|
| Range encoder (`od_ec_enc`) and CDF adaptation | `aom_dsp/entenc.c`, `aom_dsp/bitwriter.h` | the mirror of `msac.rs`; CDFs adapt as in `msac.rs` |
| Headers: sequence, frame, OBUs, `av1C` | `av1/encoder/bitstream.c` | reduced still-picture header |
| Block and mode syntax: partition, modes, angle deltas, CfL alphas, palette, filter-intra, tx size/type, skip, delta q, CDEF index, LR units | `bitstream.c` | the order and contexts are the decoder's (`decode.rs`); our writer is checked against it |
| Coefficient coding (eob, base, base range, Golomb, dc sign, contexts) | `av1/encoder/encodetxb.c` | the mirror of rav1d's `decode_coefs` |
| Forward transforms, all sizes and types | `av1/encoder/av1_fwd_txfm2d.c`, `av1_fwd_txfm1d.c` | encoder-only; AVX2 + NEON |
| Quantization, QM, coefficient optimization (trellis) | `av1_quantize.c`, `txb_rdopt.c` | encoder-only |
| Rate model: costs from the current CDFs | `av1/encoder/rd.c` (`av1_fill_mode_rates`, `av1_fill_coeff_costs`) | refreshed per superblock row |
| Partition search | `partition_search.c`, `partition_strategy.c` (+ its ML pruning weights) | 64×64 superblocks first |
| Intra mode search incl. angle deltas, filter-intra, CfL | `intra_mode_search.c`, `rdopt.c` | estimate with SATD first, full RD on the best few |
| Transform size and type search | `tx_search.c` | pruned by speed |
| Palette search | `av1/encoder/palette.c` | k-means colours, the palette cache, colour-index map order as the decoder reads it |
| Screen content decision | `av1_set_screen_content_options` | already ported in image-processing (`aom::Screen`); it moves here |
| Deblocking, CDEF, loop restoration searches | `picklpf.c`, `pickcdef.c`, `pickrst.c` | own walkers over rav1d's DSP |
| Tune-IQ tools | see below | |

### Sources

| Source | Licence | Use |
|---|---|---|
| libaom **v3.15.1**, `aomedia.googlesource.com/aom` at the release tag (what `shiguredo_aom` builds) | BSD-2 + AOM Patent Licence 1.0 | the reference we port from and measure against; notices kept in ported files |
| github.com/darkskygit/libaom | same | **not used**: the mirror stopped syncing on 2020-07-10 (newest tag v2.0.0). It has no all-intra usage, no tune IQ, and three years of encoder work missing |
| rav1e | BSD-2 | can be read and ported from: range encoder, forward transforms with Rust SIMD, RDO structure |
| github.com/Remade-With-Rust/rusty-av1-toolkit | BSD-2 | a rav1e fork plus a dav1d port. Measured 2026-10-24 at 824300c: byte-identical to rav1e by default; its quality mode shrinks graphics 9–36% but stays 1–13 dB behind libaom, with no palette or screen-content tools. It offers nothing to port that rav1e does not |
| SVT-AV1 | BSD-3-Clause-Clear + AOM patent licence | can be read |
| github.com/imazen/zenav1-aom | AGPL-3.0 or paid commercial licence | **not read, not copied, not a dependency**, unless the commercial licence is bought or AGPL is accepted for image-processing. Its README is a data point only: an all-intra libaom port that is byte-exact took about 1,500 commits and is still about 2× slower than libaom. We avoid that cost by not being exact |
| github.com/kornelski/avif-decode | BSD-3 | not for the encoder: it is a decoder (avif-parse + rav1d + yuv crate) that converts a whole image to RGB in memory, without grids, `irot`/`imir`/`clap` or row streaming, so it does not replace image-processing's AVIF reader either |

### Tune IQ

Colour is coded with libaom's tune IQ, and most of libaom's photo quality at a given size comes
from it. M0 writes down exactly which settings `AOM_TUNE_IQ` turns on in 3.15.1
(`av1_cx_iface.c`, `encoder.c`, `speed_features.c`) with their values. What is known so far:

- quantization matrices (`enable-qm` with the IQ qm-level formulas);
- per-block delta q from block variance (variance boost, `deltaq-mode 6`);
- chroma delta q;
- SSIM-based rate weighting per 16×16 block (`av1_set_mb_ssim_rdmult_scaling`; image-processing
  already models it in `aom::Balance` so that grid tiles weigh as the whole frame does);
- sharpness and adaptive CDEF settings.

Alpha is coded with tune PSNR, i.e. none of these.

## Architecture

```
src/enc/
  mod.rs         Encoder, Config, the public API
  frame.rs       source planes, reconstruction frame (a rav1d picture), mode info
  analyse.rs     screen content estimate, activity map (variance boost, SSIM weights)
  search/        partition.rs, intra.rs, palette.rs, cfl.rs, tx.rs, rd.rs (rate + distortion)
  quant.rs       quantize, QM, trellis
  fdct/          forward transforms; x86 (AVX2) and arm (NEON) kernels with the same output
  filter/        deblock.rs, cdef.rs, lr.rs: searches and their walkers
  write/         ec.rs (range encoder), header.rs, block.rs, coefs.rs, obu.rs
  tokens.rs      compact store of each block's decisions and coefficients until the frame is written
```

### Flow of one frame: a stream of superblock rows

The encoder never holds the frame. It works on a window of superblock rows that moves down the
image as rows arrive:

1. **Rows in.** The caller pushes rows as they arrive. Image-processing never waits on a slow
   client, so the encoder starts on the first 64 rows while the rest is still uploading.
2. **Search per superblock row**, in raster order (or as a wavefront, see Threads): partition,
   then modes and transforms per block, with the reconstruction written into the window as each
   block is decided, so the next block predicts from exactly what the decoder will have. AV1
   intra prediction reads unfiltered pixels, so the search never waits for a filter.
   Decisions and quantized coefficients go into `tokens`, compactly (see Memory).
3. **Filter statistics, one row behind.** Once the row below is reconstructed, the window holds
   everything the filters of a row touch. For that row the encoder runs:
   - deblocking at the row's level (a per-superblock level through `delta_lf`, or one level from
     q; M4 measures both);
   - for each 64×64, the SSE against the source of every CDEF strength candidate. These are
     numbers, about 1 KB a 64×64;
   - the Wiener and self-guided statistics of each loop restoration unit, measured after the
     64×64's best CDEF strength.

   Then the row's source and reconstruction leave the window.
4. **Frame decisions from statistics.** At the end: the CDEF strength set (up to 8 pairs) and
   each 64×64's index, from the stored SSEs; each LR unit's filter, from its statistics.
5. **Write.** The frame header now has everything (q, delta q, filter levels, CDEF strength set,
   LR types). The tile data is written from `tokens` with adapting CDFs. Output: the OBUs and
   the `av1C` record.

Writing at the end, rather than during the search, is needed because CDEF indexes and LR
coefficients are coded inside the tile data but are chosen per frame. It also keeps the search
free to try things without touching the writer's state.

The cost of streaming is that loop restoration is measured after a 64×64's own best CDEF
strength, which may not end up in the frame's set of 8. libaom searches the whole filtered frame
instead. M4 measures this against a whole-frame search (a test mode that holds the frame), and
only accepts streaming if it holds the gates.

**No whole-image pre-pass.** libaom decides screen content per frame before coding, so today
image-processing runs `aom::Screen` over the whole image first and keeps every row aside (on
disk, or in memory when the temp dir is missing or full) until the grid tiles are coded. Here,
`allow_screen_content_tools` is a header bit that only allows palette per block: the encoder
turns it on, and each superblock decides from its own 16×16 counts whether to search palette.
It is the same estimate, applied locally. On photos the per-block palette flag costs a few bits
until its CDF adapts. The SSIM rate weighting of tune IQ is the other whole-image input today
(the frame's mean activity, `aom::Balance`). It becomes a fixed reference, calibrated on the
corpus, or a running mean. M5 measures which one holds the gates. Without a pre-pass,
image-processing no longer keeps the rows aside.

### Rate and distortion

- Rates come from the CDFs as they adapt. Cost tables are rebuilt from the live CDFs once per
  superblock row, as libaom does with its cost-update frequency. In a wavefront, a row uses
  the CDFs as they stood after the row above.
- Distortion is SSE on the reconstruction (QM-weighted for tune IQ), with a SATD/Hadamard
  estimate to rank candidates before a full transform, quantization and reconstruction.
- Lagrangian λ comes from qindex as in libaom's `rd.c`, scaled per 16×16 by the tune-IQ SSIM
  weight.

### Speed

The speed comes from three things:

1. **Searching less, by estimate.** Rank intra modes by SATD of the prediction residual and do
   a full RD pass only on the best few. Prune partitions early: by variance and the cost of the
   parent block, and with libaom's partition-pruning NN weights, which are BSD data tables. Try
   transform types by estimate, and skip search work on flat blocks. Effort 0–9 sets how
   many candidates survive each stage. Every pruning rule is kept only if it holds the gates.
2. **Fast kernels.** Prediction and inverse transforms use dav1d's assembly through rav1d. The
   encoder's own kernels — forward transforms, SATD, SSE/variance, quantization, CDEF/LR
   statistics — get AVX2 versions with NEON twins giving the same output, plus scalar fallbacks.
3. **No waste.** No per-frame allocation of whole-frame mode-info grids at 4×4. One arena per
   frame. No copy of the input: the source planes are the caller's rows.

### Memory

The encoder holds three things, and none of them grows with the image's area:

| What | Size |
|---|---|
| The window: source, reconstruction and the deblocked/CDEF copies of about 2 superblock rows plus filter borders (~144 rows) | width × 144 × 3 planes × 4 copies: **1.7 KB a column**, 7 MB for a 4000 px wide image |
| Above contexts: one row of 4×4 mode info, partition, skip, tx size, coefficient and palette contexts | a few hundred bytes a column |
| `tokens`: each block's decisions and its coefficients up to eob, as variable-length bytes (most levels are 0–2) | **in proportion to the output**: about 2–3 times the coded size |
| Filter statistics: CDEF SSE per 64×64 and candidate, LR statistics per unit | ~0.3 bytes a pixel |
| Fixed: CDFs, cost tables, scratch for one 64×64 search | under 4 MB |

For a 4000×3000 screenshot coded as one frame, that is about 15 MB plus 2–3 times its output size,
against about 600 MB with libaom today. For a 1400 px grid tile, it is about 7 MB plus tokens,
against 48 MB plus 94 MB. A lossless Q100 holds no more than a lossy one beyond its larger
output: its tokens are bigger, its window is the same.

The weak spot is the tokens at high quality on noise. Every coefficient is non-zero there, so
the tokens come close to 1 byte per coefficient, about 3 bytes a pixel. If that is too much, the
fallback is to fix the CDEF strength set and LR types from the first superblock rows. With those
fixed in advance, the tile data can be written straight away as each row is decided, without
storing tokens.

The encoder reports the most it will hold before it starts (`Encoder::memory(config)`, from the
width, the quality and the effort), so image-processing reserves it from the job's budget as it
does for libaom today (`frame_memory`, `context`). Every bench reports peak memory next to time,
and a change that raises the peak needs a reason.

### Threads

Two uniform tile columns for photos over 512×512, each coded by its own encoder: `still_on`
hands the columns after the first to the caller's spawner (the `Spawner` of
`dav1d_open_with_spawner`), so image-processing's thread pool stays in charge, and the caller
codes every column no worker started. The tiles are merged before the filter search, and the
bytes do not depend on the threads. On the full-size corpus at qindex 140 two threads take
3311 → 1947 ms, a second encoder about 2 MB; the second column costs photos +0.5% bytes at
+0.02 to +0.05 dB. Screen content and smaller pictures stay one tile: each tile learns its
CDFs afresh, a fixed 100–300 B that grew graph paper 60% and failed three AVIF gates.

A wavefront over superblock rows was the first plan: no tile edges, but each row's search
would take the CDFs of the superblock above-right (−0.02 to −0.05 dB at equal bytes, measured
on branch `wavefront`), and the writer, the delta-q chain and the row handoffs would need a
symbol recorder. Tiles cost about as much and decode on rav1d's tile threads too.

### API sketch

```rust
pub struct Config {
    pub width: u32,
    pub height: u32,
    pub layout: Layout,            // I444 or I400
    pub quality: Quality,          // cq-level 0-63; 0 = lossless
    pub effort: u8,                // 0-9
    pub tune: Tune,                // Iq or Psnr
    pub screen: Option<bool>,      // None: decide from the image
    pub ssim_reference: Option<f64>, // the whole image's activity mean, for a grid tile
    pub colour: Colour,            // primaries, transfer, matrix, range for the headers
}

impl Encoder {
    pub fn memory(config: &Config) -> usize;
    pub fn new(config: Config, helpers: Option<&Runner>) -> Result<Encoder>;
    pub fn push(&mut self, rows: Rows<'_>) -> Result<()>;   // planes, any number of rows
    pub fn finish(self) -> Result<Coded>;                   // OBUs + av1C
}
```

## Quality

The judge is the gate image-processing already applies (`test/engine-avif.test.js`): against the
same lossless render, **at most 0.49 dB below and at most 5% larger** than today's output, with the
documented exceptions (grey noise Q30 +11.8%, photos beside large flat areas +5.2–7.5%).

Measured on:

- the corpus: the AVIF gate inputs, plus the photo and graphics set from the 2026-10-24 rav1e
  comparison (photos, chart, logo, diagram, screenshot, text page, graph paper, grey noise), with
  and without alpha;
- Q 30, 50, 75, 90 and 100, at efforts 0, 4 and 9;
- metrics: PSNR as the gates compute it, bytes, and SSIMULACRA2. libaom's tune IQ trades PSNR
  for perceived quality, and an encoder that only maximises PSNR could pass the gates while
  looking worse. SSIMULACRA2 at equal size must not fall below libaom's either;
- references: libaom 3.15.1 as configured in `aom.rs`, recorded once as numbers (dB, bytes,
  SSIMULACRA2, time, peak), not files.

Q100 is held to what libaom does with cq-level 0 today, which M0 measures: whether it reconstructs
the YUV planes exactly, and how big and how memory-hungry it is.

## Validation

Invariants that every test and fuzz run checks:

1. **The decoder agrees.** rav1d decodes every stream we write (with `strict_std_compliance`) to
   exactly the reconstruction the encoder made, compared by CRC per plane. A debug mode builds
   the fully filtered frame for this. Any mismatch is a writer or reconstruction bug, caught on
   the first block where it happens.
2. **A second decoder agrees.** libaom's decoder, on `worker`, decodes the same streams to the
   same pixels.
3. **Writer round trips.** Random symbols with random CDFs, and random coefficient blocks of
   every size and type, are written and read back through `msac.rs` and the coefficient decoder.
4. **Searches against brute force.** On small blocks, each pruned search is compared with an
   exhaustive one. A pruning rule may lose some RD cost but must never crash or mis-signal.
5. **Fuzz.** Random sizes (1×1 to large, odd sides), contents, configs: no panic, every stream
   decodes, invariant 1 holds.
6. **Kernels.** Every SIMD kernel against its scalar version, on x86 and arm.

Benchmarks run through the queue on `worker`, pinned, at 1, 2 and 3 CPUs (production pods have
2–3 CPUs shared with Node), for both time and peak memory.

## Milestones

Each one ends with something measured. The branch is committed after every built and measured
step, with a running handoff file.

| | What | Done when |
|---|---|---|
| M0 | Harness: corpus, libaom reference numbers (dB, bytes, SSIMULACRA2, time, **peak**), metric scripts. Exact tune-IQ and speed settings of 3.15.1 at cpu-used 0–9 written down. What libaom does at cq-level 0 (lossless or not, size, peak). Ranking of libaom's tools on graphics (palette, filter-intra, intra edge, CfL, smooth, IDTX, 64-point transforms, QM), each turned off in turn | reference table committed; the ranking says which tools carry the graphics advantage |
| M1 | Writer and streaming skeleton: range encoder, headers, OBUs, `av1C`, the row window and `tokens`; every block DC_PRED at a fixed partition and transform; forward transform, quantization, coefficient coding; reconstruction in the window | invariants 1–3 hold on the corpus; peak memory independent of the height |
| M2 | Lossless (qindex 0, WHT 4×4) | Q100 compared with M0's libaom numbers |
| M3 | Full intra: all modes and angle deltas, filter-intra, CfL, partition RD search, transform size and type search, rate model from CDFs, trellis | photos within the gate at Q30–90, with no filters yet (libaom with filters off as the reference for this step) |
| M4 | Filter searches from per-row statistics: deblocking, CDEF, loop restoration; a whole-frame test mode to compare against | photos within the gate against libaom as configured today; streaming loses at most 0.1 dB against the whole-frame mode |
| M5 | Tune IQ: QM, variance-boost delta q, chroma delta q, SSIM rate weighting with a fixed or running reference; tune PSNR for alpha | photos within the gate including SSIMULACRA2; alpha within the gate |
| M6 | Screen content per superblock (the `aom::Screen` counts, applied locally), palette search with the palette cache, screen-content transform choices | every graphics image within the gate, with no whole-image pre-pass |
| M7 | Speed: pruning per effort, AVX2 + NEON kernels, profile-driven | at default effort, single core, the targets below; every effort within the gate |
| M8 | Threads: tile columns on the caller's helpers | 2 and 3 CPU numbers; output identical to single-threaded |
| M9 | image-processing: `aom.rs` replaced, `shiguredo_aom` and its cmake/libclang/llvm-tools build removed, `frame_memory`/`context` replaced by `Encoder::memory`, the rows no longer kept aside for the screen-content pre-pass; then the grid tile size revisited, since memory no longer forces small tiles and every tile edge costs 100–700 bytes | full Rust + JS suite green on worker |

M1 and M2 make the bitstream correct; M3–M6 bring the quality; M7–M8 bring the speed.
Speed work before M6 is limited to keeping the search from becoming unusably slow.

### Speed targets (proposed)

At default effort 4, against libaom at cpu-used 5 as configured today, on one core:

- **at least 3× faster** on photos and on graphics, at gate quality;
- **peak memory independent of the height**: about 15 MB plus 2–3 times the output for a
  4000 px wide frame, and never above a tenth of libaom's.

For scale: the 2026-10-18 measurements put a 6000×4000 photo at Q50 at 10.2 s and 493 MB with
libaom, and at 5.4 s and 265 MB with rav1e at speed 7, which matched libaom's quality on photos
but not on graphics.

## Risks

- **Graphics quality.** libaom's lead there rests on palette, the screen-content transform
  choices and a thorough RD search; M0's ranking says how much of each we need. If matching it
  costs too much time, a separate effort level for screen content is a fallback.
- **Tune IQ.** Part of libaom's photo quality is in many small settings. M0 lists them; M5 may
  find that some only matter for SSIMULACRA2 and not for the PSNR gate. Whether to keep those
  is a decision (see below).
- **Coupling to rav1d internals.** The encoder calls private DSP and table APIs, and upstream
  rav1d may change them. The fork pins its upstream base; rebases are deliberate.
- **Reconstruction mismatches** between search and write are the classic encoder bug. Invariant
  1 is checked on every test, so they fail loudly and early.

## Open decisions

1. **Location.** In the rav1d fork as `src/enc/` behind feature `encoder` (proposed), or a
   separate crate with rav1d's DSP made public?
2. **Targets.** Is 3× the speed at effort 4 the right bar? Memory: is a peak set by the width
   plus 2–3 times the output enough, or must the tokens go too, by fixing the filter parameters
   in advance, at some cost in size?
3. **Metric.** PSNR gate only, or also "SSIMULACRA2 not below libaom at equal size"?
4. **Scope.** 8-bit 4:4:4 + monochrome only until everything else is done?
5. **Effort.** Keep effort 0–9 with the same meaning (4 = default), with our own search depths?
