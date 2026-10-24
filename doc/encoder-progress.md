# Encoder progress

The running handoff for `doc/encoder.md`: where the work stands, how to build and test it,
and one paragraph per step, newest first.

## State

- Branch `encoder`, M1 in progress. Built: `src/enc/ec.rs` (range encoder).
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

**2026-10-24, range encoder.** `Writer` ports libaom's `od_ec_enc` to rav1d's CDF layout and
adapts CDFs as `msac.rs` does. Random sequences of every symbol kind round-trip through
`MsacContext` with the x86 assembly, and the adapted CDFs match lane for lane. Faults in the
interval, the adaptation rate and the last byte each fail the tests. Two harness traps came
from the decoder's assembly, not the encoder: it loads CDFs with aligned vector loads (an
unaligned array crashes with SIGSEGV), and its adaptation works in signed 16 bits, so a
probability lane of 0, which no AV1 CDF holds, wraps. The tests keep CDFs aligned and lanes in
1..32767. Next: headers and OBUs, then the block writer for DC_PRED blocks.
