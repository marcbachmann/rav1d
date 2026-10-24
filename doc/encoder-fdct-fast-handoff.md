# Fast forward transforms: handoff

Branch `enc-fdct-fast` (from `encoder` d135c7e6). Brief: `doc/encoder-fdct-fast-brief.md`.

## What it is

`src/enc/fdct_fast.rs` (+ `fdct_fast/`): rav1e's forward transforms from rusty-av1-toolkit
(BSD-2, rav1e contributors; headers kept on the ported files).

- `kernels.rs`: `forward_shared.rs`'s `impl_1d_tx!` macro (Daala lifting kernels generic over
  `TxOperations`), verbatim except `get_func` (own `Txfm1d`) and no WHT kernel.
- `scalar.rs`: the `i32` lane and 2D driver (`transform/forward.rs`).
- `x86.rs` (AVX2) and `arm.rs` (NEON): the `I32X8` lanes, transposes and 2D drivers of
  `asm/{x86,aarch64}/transform/forward.rs`. Changed: `i32` stores in `coefs[x * ch + y]`,
  only the coded rows (< 32) are transposed and row-transformed, 4-wide rows are read with
  64-bit loads (rav1e reads 16 bytes past a 4-wide block), height-4 stores write 4 lanes
  (rav1e writes 8 and spills into the next column).
- `forward(residual, stride, coefs, size, ty)`: `fdct::forward`'s contract; `WHT_WHT` goes to
  `fdct::forward`. The lane comes from `rav1d_get_cpu_flags()` (AVX2 / NEON), so it is the
  SIMD lane only after `rav1d_init_cpu` (the encoder's `Dsp::new` runs it). `forward_in(lane,
  ..)` picks one explicitly.
- Shifts: rav1e's 8-bit table (`[3,0,0]` at 4×4, `[4,-1,0]` up to 16, `[4,-2,0]` with a
  32-point side, `[4,-1,-2]` with 32×64/64×32/64×64, `[4,-2,0]` 16×64/64×16). Its 1D transforms
  are orthonormal, so these give libaom's scale, rectangular sizes included (no NewSqrt2 step
  needed).
- `rd_residual` (`frame.rs`) calls `fdct_fast::forward`. `fdct.rs` stays (reference + tests);
  its `Kind`/`kinds` and test helpers became `pub(super)`/`pub(crate)`; `max_error` takes the
  forward function.

## Tests (`fdct_fast::tests`)

- `simd_matches_scalar`: every decodable size/type, 35 blocks ±255 (constants, checkerboard,
  32 random), SIMD reads from a wider picture (stride w+8): identical to the scalar lane.
- `extreme_blocks_stay_in_range`: per coded coefficient the ±255 sign pattern of its impulse
  response (and its negation); opt-dev overflow checks make the scalar lane panic on any
  intermediate beyond `i32`; SIMD must equal scalar on them.
- `round_trip_through_rav1d`: fdct's harness (`max_error`, same bounds 0–4) for both lanes ×
  every rav1d kernel level. On aarch64 the Rust inverse has no WHT (only the NEON one is
  compiled), so WHT at empty flags is skipped there.
- `same_scale_as_libaom`: RMS of fast / fdct coefficients on 64 random blocks per size and
  type, asserted within 1%. Measured 0.9998–1.0003 for every size.
- `bench_forward` (ignored, release).

`frame::tests::rd_search` byte gates raised (232→234, 589→604, 56→59; dB gates unchanged, all
three cases higher in dB: 43.501→43.509, 44.293→44.308, 40.38→40.87).

## Verified

| Job / where | What | Result |
|---|---|---|
| Mac aarch64 | `enc::fdct_fast` (NEON) | 4 passed |
| `mvlv6tr5etlgv02n` | `enc::` on worker (AVX2), then 3 faults | 29 passed, 0 failed; faults below |
| `mvluyqtmenks493v` | `bench_forward` release, worker | table below |
| `mvlunyrar1bckv5o`, `mvlv3jlj74tiefmp` | pinned A/B, 2 CPUs | table below |
| `mvlus3r4vwt6lu4i`, `mvluv7600rq3dqfp` | qidx sweep 90–150, BD-rate | below |

Faults: AVX2 `lr_flip` dropped → simd_matches_scalar, extreme, round trip; 4×4 shift 3→4 →
scale, round trip; scalar `ud_flip` dropped → lanes, extreme, round trip.

Per call, DCT_DCT, release, worker (ns): libaom port / fast scalar / AVX2
4×4 147/66/32, 8×8 272/226/51, 16×16 1092/777/154, 32×32 5432/3187/667, 64×64
19316/11666/2392, 16×32 2615/1552/329, 64×32 12938/7237/1480.

A/B, corpus crop 256, `ENC_KEEP=4 ENC_RECT=2 ENC_LEVELS=0..4`, qidx 120, same pin (clean
rounds; the host was loaded in some rounds of `mvlv3jlj74tiefmp`, those dropped):

| tree | ms | bytes | mean dB |
|---|---|---|---|
| base d135c7e6 | 3532–3542 | 30171 | 49.564 |
| fast (AVX2) | 2542–2558 | 30106 | 49.499 |
| fast, scalar lane forced | 3093–3095 | 30106 | 49.499 |

−27.8% time. The SIMD and scalar lanes give the same bytes end to end.

Quality: at qidx 120 the mean dB drops 0.065 (> 0.05) at −0.2% bytes, nearly all from
graphic-logo (−0.37 dB, −13 B). Across qidx 90/105/120/135/150 the mean dB difference is
−0.02/+0.05/−0.07/+0.09/0.00 at bytes +0.3/+0.1/−0.2/0.0/−0.2%: noise of the RD path, not a
trend. BD-rate (PSNR, cubic, 5 points) fast vs base: gradient +1.0%, graphic-logo +0.1%,
graphic-ui +0.9%, photo-detail −0.1%, photo-orient +0.1%, photo-wiediger +0.3%, photo-xeon
+0.1%; mean +0.35%.

Tried and dropped: 1 or 2 extra precision bits (first shift +E, last shift −E). BD-rate
mean +0.27% / +0.17% (same noise band), and +1 already overflows `i32` on the extreme
blocks, +2 already on random ±255 blocks at 32×64 and 64×32.

## Open

- Base A/B tree was a detached worktree `~/Development/livingdocsIO/rav1d-fdct-base` at d135c7e6
  (remove with `git worktree remove` from the rav1d repo).
- 10/12-bit: only rav1e's 8-bit shift row is used; the encoder is 8-bit.
