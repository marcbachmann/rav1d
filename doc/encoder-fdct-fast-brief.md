# Brief: fast forward transforms from rav1e (rusty-av1-toolkit)

Worktree `~/Development/livingdocsIO/rav1d-fdct-fast`, branch `enc-fdct-fast` (from `encoder`
d135c7e6). Work only here. Commit after each built and measured step (`git commit
--no-gpg-sign`, explicit paths, no attribution lines, never push), and keep a running
handoff in `doc/encoder-fdct-fast-handoff.md`.

## Goal

The encoder (`src/enc/`, feature `encoder`) spends ~27% of its time in `src/enc/fdct.rs`, a
scalar, bit-exact port of libaom's C forward transforms. Replace its use in the encoder's search
with a fast forward transform built from rav1e's: the 1D kernels in
`rusty_av1e/src/transform/forward_shared.rs` (generic over a `TxOperations` lane trait) with the
scalar `i32` lane (`rusty_av1e/src/transform/forward.rs`), the AVX2 `I32X8` lane and 2D driver
(`rusty_av1e/src/asm/x86/transform/forward.rs`) and the NEON twin
(`rusty_av1e/src/asm/aarch64/transform/forward.rs`). A clone of the toolkit (BSD-2, rav1e
contributors; keep their copyright headers on ported files) is at
`/private/tmp/claude-501/-Users-marcbachmann-Development-livingdocsIO-image-processing/6e626144-8f6c-4a5d-89c8-905d6891d283/scratchpad/rusty-av1`.

These are not libaom's transforms (Daala-style lifting with `add_avg`/`rshift1`), so the output
will not match `fdct.rs`; that is fine, forward transforms are not normative. What must hold:

1. **Same API and layout** as `fdct::forward(residual: &[i16], stride, coefs: &mut [i32], size,
   ty)`: `coefs[x * ch + y]`, sides capped at 32 (`coded_dimensions`), every size/type pair
   `fdct::supported` lists for the bitstream (the ones `DAV1D_TX_TYPES_PER_SET` and the
   64-point DCT allow; WHT stays on the existing `fwht4x4`). Same scale: the existing quantizer
   and rav1d's dequantizer and inverse must reconstruct correctly. rav1e has its own shift and
   rectangular (`NewSqrt2`) handling; check the scale against `fdct.rs` per size (ratio of
   coefficient magnitudes on random blocks ≈ 1), not just the round trip.
2. **Round trip through rav1d's inverse** within the bounds `fdct.rs`'s test
   `round_trip_through_rav1d` uses (0–4 by size), for every size/type, at every CPU level
   (`CpuFlags::run_time_detect()`; see the trap in `doc/encoder-fdct-handoff.md`).
3. **AVX2 and NEON give exactly the scalar lane's output** (a test over every size/type on
   random blocks). The NEON path is compiled on aarch64 only; it must exist and be tested on
   the Mac (`cargo test` on aarch64 runs it), even if timings come from worker. Running cargo
   on this laptop is allowed only for that aarch64 test, small and short; everything else goes
   through the queue on `worker`.
4. Keep `fdct.rs` (it is the libaom reference and its tests stay); add the new code as
   `src/enc/fdct_fast.rs` (+ arch modules as you see fit) and switch the encoder's call sites
   in `src/enc/frame.rs` (`fdct::forward` in `rd_residual`) to it.

## Deciding measurements

- Encoder tests: `cargo test -p rav1d --profile opt-dev --features encoder --lib enc::` all
  green (the round-trip tests decode the encoder's output and compare with its reconstruction).
- Quality and speed on the corpus bench, same pin, old vs new tree:
  `scripts/enc-ab.sh <tree A> <tree B> ENC_KEEP=4 ENC_RECT=2 ENC_LEVELS=0..4` with
  `ENC_CORPUS` (the PPMs at `.../scratchpad/avif-corpus`) and `ENC_CROP=256`. Report total ms,
  bytes and mean dB per tree. The base for comparison is the `encoder` branch tree at
  `~/Development/livingdocsIO/rav1d` (or a detached worktree of d135c7e6). Bytes/dB will differ
  slightly (different transform); a bytes rise > 1% or a dB drop > 0.05 dB at equal qidx is a
  quality regression to report, not hide.

## How to build and run (worker, through the queue MCP tools)

Use `mcp__queue__submit` (or the mcp-proxy queue tools), `hosts: ["worker"]`, `context`
listing the worktree (e.g. `"~/Development/livingdocsIO/rav1d-fdct-fast:rav1d-fast"`) and, for
the A/B, the base tree and the corpus as further contexts (they land as siblings under the
host's tree root). Run cargo inside docker:

```
docker run --rm --cgroup-parent="$COMMANDQUEUE_SLICE" --ulimit core=0 -v $PWD/..:/trees \
  -v /root/.cache/ld-image-processing-deps:/deps -w /trees/rav1d-fast \
  -e CARGO_TARGET_DIR=/deps/target-rav1d-fast -e CARGO_HOME=/deps/cargo-home-rav1d-enc \
  -e RUSTUP_TOOLCHAIN=stable ip-build:trixie bash -c '<cargo command>'
```

Benchmarks: `pin: true, cpus: 2`, one measurement per job, and keep each job under ~5 minutes.
Read job outputs with `mcp__queue__log` (the submit answer is only the last 10 lines). Never
put big temporary data in /tmp. Do not touch `~/Development/livingdocsIO/rav1d` (the main
`encoder` worktree another session edits) or image-processing.

## Report

When done (or at a token cap: write the handoff and stop): the commits, the test results with
job tickets, the A/B table, the scale check per size, and anything that did not hold.
