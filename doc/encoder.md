# An AV1 still-image encoder in rav1d

Status: design, nothing built yet. Branch `encoder` of marcbachmann/rav1d, from `helpers` (bc62eb90).

## Goal

A Rust AV1 encoder for still images. It replaces libaom 3.15.1 as the AVIF encoder of
image-processing (`src/engine/avif/aom.rs`). It must:

- code at least the quality libaom codes, at the same size, on every image the AVIF gates
  cover, photos and graphics alike;
- be much faster than libaom at the default effort, on one core first, then with helpers;
- hold far less memory than libaom: today one grid tile costs 48 MB plus 48 bytes a pixel;
- build without cmake, libclang, llvm-tools or a C library. Only nasm stays, which rav1d's
  assembly already needs.

Being bit-exact with libaom is **not** a goal. The output only has to be a conformant AV1
stream that meets the quality gates (see [Quality](#quality)). Without bit-exactness,
every search can take its own shortcuts, and that is where the speed comes from.

## Non-goals (for now)

- Inter frames, animation (`avis`), motion search, rate control, lookahead.
- Intra block copy: image-processing codes with it off (on text it was 70% smaller and 2.9 dB
  lower than the reference).
- Film grain, superres, 4:2:0/4:2:2 input, 10/12-bit. The data structures stay generic over
  `BitDepth` like the rest of rav1d, so high bit depth can come later without a rewrite.
- AV1 tiles inside a frame. Threads come from a wavefront instead (see [Threads](#threads)).

## What has to be coded

This is what image-processing asks of libaom today, and what the encoder takes over:

| | Colour | Alpha |
|---|---|---|
| Layout | 8-bit 4:4:4 (profile 1), BT.601 matrix, full range, sRGB transfer (13) | 8-bit monochrome (profile 0) |
| Quality | libavif's tune-IQ table, Q 0–100 → cq-level 63–0 (`iq_cq_level`) | `((100 - Q) * 63 + 50) / 100` (`cq_level`) |
| Tuning | tune IQ | tune PSNR |
| Speed | cpu-used `9 - effort`, effort 0–9, default 4 (cpu-used 5) | same |
| Screen content | decided once for the whole image by `aom::Screen`, then forced on each grid tile | same |
| Q100 | cq-level 0 = qindex 0 = coded lossless (WHT, no filters) | same |
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
| SVT-AV1 | BSD-3-Clause-Clear + AOM patent licence | can be read |
| github.com/imazen/zenav1-aom | AGPL-3.0 or paid commercial licence | **not read, not copied, not a dependency**, unless Marc buys the commercial licence or accepts AGPL for image-processing. Its README is a data point only: an all-intra libaom port that is byte-exact took about 1,500 commits and is still about 2× slower than libaom. We avoid that cost by not being exact |
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

### Flow of one frame

1. **Rows in.** The caller pushes rows as they arrive. Image-processing never waits on a slow
   client, so the encoder can start before the image is complete. Each 64-row band is analysed
   (screen-content counts, 16×16 activity) as soon as it is complete.
2. **Search per superblock**, in raster order (or as a wavefront, see Threads): partition, then
   modes and transforms per block, with the reconstruction written into the frame as each block
   is decided, so the next block predicts from exactly what the decoder will have. Decisions and
   quantized coefficients go into `tokens`, compactly: only up to each block's eob.
3. **Filter searches** once the frame is reconstructed: deblocking levels on the whole frame,
   then CDEF strengths per 64×64, then loop restoration per unit. Each search measures its
   candidates locally against the source; the filtered frame is only built where a later search
   needs it. libaom's `SKIP_POSTPROC_FILTERING` already skips the final filtered output, and so
   do we.
4. **Write.** The frame header now has everything (q, delta q, filter levels, CDEF strength set,
   LR types). The tile data is written from `tokens` with adapting CDFs. Output: the OBUs and
   the `av1C` record.

Writing at the end, rather than during the search, is needed because CDEF indexes and LR
coefficients are coded inside the tile data but are chosen per frame. It also keeps the search
free to try things without touching the writer's state.

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

Target for 8-bit 4:4:4: under 10 bytes a pixel of the frame plus under 4 MB fixed, against
libaom's 48 MB plus 48 bytes a pixel. The budget: source 3 B/px, reconstruction 3 B/px, mode
info under 1 B/px, tokens (coefficients up to eob, typically well under 1 B/px at mid quality),
and the filter search buffers per 64×64. The encoder reports what it will hold before it starts
(`Encoder::memory(config)`), so image-processing can reserve it from the job's budget the way it
does for libaom today (`frame_memory`, `context`).

### Threads

Single core comes first. Threads come once that is fast.

- **Wavefront search.** Superblock rows search in parallel, each row two superblocks behind the
  one above, so the above-right neighbours are always reconstructed. This needs no AV1 tiles,
  so no bytes are lost to tile edges, and it works on one big frame, which is the case grid
  tiles do not cover.
- **Serial writing.** Writing the tokens is a small part of the time and stays on one thread.
- Workers run on the caller's helpers through the same spawner API as `helpers`
  (`dav1d_open_with_spawner`), so image-processing's thread pool and budget stay in charge.

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

Q100 must reconstruct the YUV planes exactly.

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
| M0 | Harness: corpus, libaom reference numbers, metric scripts. Exact tune-IQ and speed settings of 3.15.1 at cpu-used 0–9 written down. Ranking of libaom's tools on graphics (palette, filter-intra, intra edge, CfL, smooth, IDTX, 64-point transforms, QM), each turned off in turn | reference table committed; the ranking says which tools carry the graphics advantage |
| M1 | Writer and frame skeleton: range encoder, headers, OBUs, `av1C`; every block DC_PRED at a fixed partition and transform; forward transform, quantization, coefficient coding; reconstruction in a rav1d picture | invariants 1–3 hold on the corpus; image-processing can show the output |
| M2 | Lossless (qindex 0, WHT 4×4) | Q100 exact on the corpus; size compared with libaom's lossless |
| M3 | Full intra: all modes and angle deltas, filter-intra, CfL, partition RD search, transform size and type search, rate model from CDFs, trellis | photos within the gate at Q30–90, with no filters yet (libaom with filters off as the reference for this step) |
| M4 | Filter searches: deblocking, CDEF, loop restoration | photos within the gate against libaom as configured today |
| M5 | Tune IQ: QM, variance-boost delta q, chroma delta q, SSIM rate weighting; tune PSNR for alpha | photos within the gate including SSIMULACRA2; alpha within the gate |
| M6 | Screen content: the estimate moves in from `aom::Screen`, palette search with the palette cache, screen-content transform choices | every graphics image within the gate |
| M7 | Speed: pruning per effort, AVX2 + NEON kernels, profile-driven | at default effort, single core, the targets below; every effort within the gate |
| M8 | Threads: wavefront on the caller's helpers | 2 and 3 CPU numbers; output identical to single-threaded |
| M9 | image-processing: `aom.rs` replaced, `shiguredo_aom` and its cmake/libclang/llvm-tools build removed, `frame_memory`/`context` replaced by `Encoder::memory` | full Rust + JS suite green on worker |

M1 and M2 make the bitstream correct; M3–M6 bring the quality; M7–M8 bring the speed.
Speed work before M6 is limited to keeping the search from becoming unusably slow.

### Speed targets (proposed)

At default effort 4, against libaom at cpu-used 5 as configured today, on one core:

- **at least 3× faster** on photos and on graphics, at gate quality;
- **peak memory at most a quarter** of libaom's for the same frame.

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

## Decisions for Marc

1. **Location.** In the rav1d fork as `src/enc/` behind feature `encoder` (proposed), or a
   separate crate with rav1d's DSP made public?
2. **Targets.** Are 3× the speed and a quarter of the memory at effort 4 the right bar?
3. **Metric.** PSNR gate only, or also "SSIMULACRA2 not below libaom at equal size"?
4. **Scope.** 8-bit 4:4:4 + monochrome only until everything else is done?
5. **Effort.** Keep effort 0–9 with the same meaning (4 = default), with our own search depths?
