# Encoder progress

The running handoff for `doc/encoder.md`: where the work stands, how to build and test it,
and one paragraph per step, newest first.

## State

- Branch `encoder`, M3 in progress. Built: range encoder (`ec.rs`), sequence and frame headers
  (`header.rs`), the frame coder for every intra tool but palette (`frame.rs`), libaom's forward
  transforms (`fdct.rs`), rate counting and λ (`rd.rs`), the RD search, and `enc::still`.
- Not yet: deblocking, CDEF, restoration, quantizer matrices and delta q chosen by the search,
  trellis quantization, palette, threads, speed work.
- image-processing branch `integrate-high-quality-avif` holds `src/engine/avif/own.rs` with a
  calibration test against libaom (`own::compare`, `AVIF_CORPUS` of PPMs).
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
