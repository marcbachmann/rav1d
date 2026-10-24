# Encoder progress

The running handoff for `doc/encoder.md`: where the work stands, how to build and test it,
and one paragraph per step, newest first.

## State

- Branch `encoder`. Built: range encoder, headers, every intra tool with palettes, libaom's
  forward transforms, the RD search with RDOQ, tune IQ (Variance Boost, SSIM λ, quantizer
  matrices, chroma delta q), deblocking and CDEF strengths searched on the decoded picture.
- Not yet: restoration, threads, SIMD for the search, alpha and lossless parity checks.
- image-processing branch `integrate-high-quality-avif` codes AVIF colour with it
  (`src/engine/avif/own.rs`, calibration `own::compare` with `AVIF_CORPUS` of PPMs). AVIF gates
  188/195 at rav1d a12c8720: misses at libaom's bytes on the strip pattern, Q90 fine texture,
  antialiased text Q80; noise is 63% larger at its quality.
- Open decisions in `doc/encoder.md` are taken at their proposed defaults (Marc started the
  implementation on 2026-10-24 without answering them): in the fork, 3× at effort 4, peak by
  width plus tokens, PSNR gate plus SSIMULACRA2, 8-bit 4:4:4 and monochrome first, effort 0–9
  as today.
- Porting reference: libaom v3.15.1 at `~/Development/livingdocsIO/libaom-v3.15.1`.

## Build and test

On `worker` through the queue, in the `ip-build:trixie` image, with the stable toolchain the
image ships (`RUSTUP_TOOLCHAIN=stable`; the repo pins a nightly):

```sh
D=/root/.cache/ld-image-processing-deps
docker run --rm --cgroup-parent="$COMMANDQUEUE_SLICE" -v "$PWD":/work -v $D:/deps -w /work \
  -e CARGO_TARGET_DIR=/deps/target-rav1d-enc -e CARGO_HOME=/deps/cargo-home-rav1d-enc \
  -e RUSTUP_TOOLCHAIN=stable ip-build:trixie \
  cargo test -p rav1d --profile opt-dev --features encoder --lib enc::
```

An incremental build takes about 17 s.

## Log

**2026-10-24, rate counted once.** Coefficients are written through a cached scan table with the
lo context inline (ee598570), and the chosen transform blocks' bits come from the residual
search instead of a second count in `coef_bits` (same bytes; an assert held them equal over the
unit tests and the corpus at qindex 140, 32 and 200 with RDOQ). Pinned on 13 crops: qindex 140
3940 ms, qindex 32 9907 ms (from 5978 and 21231 at the start of the speed work).

**2026-10-24, search speed.** Pinned on 13 crops of 384 px: qindex 140 5978 → 5577 ms and
qindex 32 21231 → 17521 ms with the same or near bytes and dB: symbol costs from a table built at
compile time (same bytes), libaom's reduced intra transform-type set where the search is not
full (qindex 140 −1.8%, 0.2% fewer bytes; at 32 it lost 0.034 dB, so off there), coefficient
costs from tables filled per superblock as `av1_fill_coeff_costs` (same bytes, −4.6% and
−13%), transform types ranked by their coefficients' error with only the winner reconstructed
(−0.3% and −3.4%, ±0.006 dB). The quantizer's u64 division is the hottest line in a profile,
yet three replacements lost: a 32-bit division +6.5%, reciprocal tables rebuilt per superblock
+4.5%, reciprocal tables cached per qindex +7.5% (16 KB per transform size crowd L1). At qindex
32 a final code takes about 190 residual searches per 8×8: ranking 60, depths 46, chroma 68.
Then: split pruning on the unsplit block's whole cost (distortion alone kept dense palette
blocks whole and a chart grew 2226 → 3752 B), prune 1 at effort 4-5 (−8% at qindex 140, same
dB); the residual search's reconstruction copied instead of computed again (same bytes, −3.4%
and −6.3%). Now 4954 ms and 15740 ms, AVIF gates 190/195. A 16-wide AVX2 SATD gained 0.6%:
most SATD calls are 4 and 8 wide and stay scalar.

**2026-10-24, the gate misses on synthetic patterns.** The strip pattern of the AVIF gates
(sawtooth R, diagonal G, XOR B) loses 1.9 dB at libaom's bytes at effort 4 and gains 0.1 at
effort 9. Settings one at a time on a 1024×256 crop at Q50 (−3.20 dB): all luma or chroma
modes kept change nothing; filters off −3.76; H and V partitions at every block +2.6 dB, 4×4
blocks +1.3. The gain is at 16×16 and 8×8 (at 64 and 32 nothing), and comes from 8×4/4×8: at
effort 4 an 8×8 never splits, so the "Split beat None" rule never tries H and V there. Trying
them costs 2× time on photos for nothing (+0.03 dB always, −0.03 at 8×8 alone); a gate on
None's bits per pixel recovers the strip at 0.5 bit but costs antialiased text 3.7 dB. That is
no costing bug: the search's rate estimate is 2.5% under the coded bytes and its distortion is
the decoded picture's within 0.2 dB, with H/V or without; the extra blocks move the search
along λ, and the tune-IQ λ weight (×1.56 at qindex 133) makes that a poor trade on text.
Without the weight: text +1.5 dB at libaom's bytes, photos at Q90 −0.1 to −0.3, mean +0.24 →
+0.30 over 28 codes, bytes at a qindex up to 18% higher; kept, as libaom's tune IQ has it.
Antialiased text (grey levels 26 apart) is not monotonic in qindex: 49147 B at 48, 56810 at 54,
49321 at 60, 54919 at 64, with or without quantizer matrices, IQ filters, Variance Boost, SSIM λ
or skip prediction, and in the search's own estimate; without palettes the curve is smooth and
2.6× larger, and searching transform types at every depth fixes 54 but not 64. The quantizer
steps resonate with the level grid; the gate's Q80 lands on 54.
RDOQ does not bring grey noise near libaom's bytes (53 KB against 32 KB at qindex 144, the
same at 2, 8 and 64 coefficients); at equal bytes it is +0.2 to +1.4 dB ahead.

**2026-10-24, speed.** On 13 crops of 384 px at qindex 140, effort 4: 8535 → 5690 ms. The
chroma palette search keeps a bitmap of colour pairs and stops past 64 (857fbe94, same bytes);
effort ≤5 compares depths with DCT only and searches transform types at the chosen depth
(b982f643, 96623 B at 42.738 dB); context snapshots are a fixed struct and the end of block
comes from a cached inverse scan (a12c8720, same bytes). A 32-bit quantizer division was 6.5%
slower on a pinned A/B (6001 against 6394 ms, three rounds) and was dropped. Profile now:
`predict_plane` 24%, `write_coefs` 12%, `rd_residual` 5.5%, the scalar SATD 5%.

**2026-10-24, filters searched on the decoded picture.** The frame header is written last, so
`filters::search` decodes the coded tiles with rav1d under candidate headers: deblocking levels
per plane by steps of 8, 4, 2, 1 and off, then CDEF primary and secondary strengths for luma
and chroma, about 20 decodes. Mean at libaom's bytes +1.01 → +1.10 dB; a HEIC pattern 45.26 →
49.71 dB. Tune IQ's filter rules (no CDEF to qindex 32, sharpness 7 to 112 and 1 to 160) come
first. libaom's `predict_skip_txfm` (level 1) skips the residual search where the prediction
error is under its DC and AC limits: −4% time, same quality.

**2026-10-24, tune IQ complete for the quantizer.** `Options::tune_iq` turns on what libaom's
tune IQ changes and the encoder has: Variance Boost, the SSIM rate weighting
(`av1_set_mb_ssim_rdmult_scaling`: λ per block by the geometric mean of its 16×16 activity
factors over the picture's), quantizer matrices (all-intra luma levels, 4:4:4 chroma levels),
chroma AC delta q of min(base/2, 24) and the IQ rdmult weight. Against libaom at equal bytes
(PSNR, 10 pictures, Q30-Q90): photos −0.08 to +0.65 dB (were −1.08 to +0.95 with Variance Boost
alone; flat-area photos and dense photos needed 10-20% more bytes), graphics +1 to +8 dB except
a logo at Q80 (−0.63); mean +1.03 dB over 40 codes. The qindex meeting libaom's bytes is now
uniform over photos: about libaom's below 60, 6% finer at 148, 6% finer at 196. Still to port:
sharpness 7 with adaptive sharpness, CDEF_ADAPTIVE and the deblock and CDEF searches.

**2026-10-24, comparison with palettes.** At libaom's bytes (PSNR, 4:4:4, decoded with
filters): photos +0.11 to +1.85 dB, a gradient +3.0 to +6.0, a UI mock +4.2 to +5.5, a hard-edged
logo +4.2 to +4.4 (was −5 to −11 without palettes); mean +2.41 dB over 14 codes. Several graphics
points are extrapolated past the three qindexes measured. libaom is tuned for IQ, not PSNR, so
the JS gates still decide. Time 18–85 s against libaom's 0.1–1.6 s: speed is next.

**2026-10-24, palettes.** `palette.rs` writes what `read_pal_plane`, `read_pal_uv` and
`read_pal_indices` read: cache reuse flags, delta-coded sorted colours (raw or signed deltas for
V), and wavefront index maps with `order_palette`'s contexts. The search tries a block's own
colours when it has 2 to 8, else k-means palettes of 8 and 4 when it has up to 64, for luma as a
mode and for chroma (U, V pairs) beside the chroma modes. On a synthetic graphic: 3368 bytes at
43.8 dB without, 1336 at 70.6 dB with. The random-decision fuzzer draws palettes from a small
colour set so caches are reused.

**2026-10-24, first comparison with libaom** (image-processing `own::compare`, before
palettes): at libaom's bytes the own encoder is +0.11 to +1.83 dB on photos, a hard-edged logo
-5 to -11 dB, and 60-100x slower. Allocation-free residual search halved the time.

**2026-10-24, RD search.** Each superblock is searched dry: blocks are coded into a `Counter`
on a copy of the CDFs from the superblock's start, against snapshots of the reconstruction and
contexts, then the winning plan is coded by the same code into the range writer. Partitions
None/Split/H/V, the 13 luma modes, angle deltas of the two best directions, filter-intra and
depths; chroma modes, angles and CfL with alphas fitted per plane. Residuals try each coded
transform type through fdct, a deadzone quantizer (48/128, 64 lossless), counted rate and pixel
distortion. 473 bytes at 42.59 dB where DC prediction takes 929 at 41.83. Faults: worst
candidate, rate ignored in residuals, bottom-edge contexts, all caught. Next: compare with
libaom at equal bytes, then filters.

**2026-10-24, frame coder and forward transforms.** `Encoder<D: Decide>` codes a frame a
superblock row at a time with every partition, mode, angle, CfL, filter-intra, depth, transform
type and delta q a decider asks for; random decisions, delta q, quantizer matrices, lossless and
two tile columns decode in rav1d to the encoder's reconstruction. `fdct.rs` is libaom's forward
transforms, exact to its C code on 194 digests (handoff in `encoder-fdct-handoff.md`).

**2026-10-24, range encoder.** `Writer` ports libaom's `od_ec_enc` to rav1d's CDF layout and
adapts CDFs as `msac.rs` does. Random sequences of every symbol kind round-trip through
`MsacContext` with the x86 assembly, and the adapted CDFs match lane for lane. Faults in the
interval, the adaptation rate and the last byte each fail the tests. Two harness traps came
from the decoder's assembly, not the encoder: it loads CDFs with aligned vector loads (an
unaligned array crashes with SIGSEGV), and its adaptation works in signed 16 bits, so a
probability lane of 0, which no AV1 CDF holds, wraps. The tests keep CDFs aligned and lanes in
1..32767. Next: headers and OBUs, then the block writer for DC_PRED blocks.
