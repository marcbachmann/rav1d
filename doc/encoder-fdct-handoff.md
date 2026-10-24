# Forward transforms: handoff

Branch `enc-fdct` (from `encoder` 8d19d9ef). Code: `src/enc/fdct.rs`, registered in
`src/enc/mod.rs`. Code commits `ced13160` (port + round trip + libaom digests) and `cd830dfd`
(`scan_rc`, sparse-block eob test); this file is the commit after them.

## What it is

A scalar port of libaom v3.15.1's C forward transforms (`av1_fwd_txfm1d.c`,
`av1_fwd_txfm2d.c`, `av1_txfm.c`, `av1_fwht4x4_c` from `hybrid_fwd_txfm.c`), same integer
arithmetic. The DCT 4–64 and ADST 8/16 bodies were translated mechanically from the C (each
`bf0`/`bf1` stage rewritten onto the buffer it points to, range checks dropped, as they only
assert); ADST 4, the identities, the 2D driver, the shift and cos_bit tables and WHT by hand.

- `forward(residual, stride, coefs, size, ty)`: every size and every type libaom has
  (`supported`), `WHT_WHT` at 4×4. 193 size/type pairs plus WHT.
- `coefs[x * ch + y]`, `x < cw`, `y < ch`, `(cw, ch) = coded_dimensions(size)` = sides capped
  at 32: column major, `x` the horizontal frequency. This is libaom's output layout after its
  64-point repacking, and what rav1d's `itxfm_add` and `DAV1D_SCANS` read. Rows past 32 are
  not transformed at all (libaom computes then drops them; the output is the same).
- `scan_rc(size, ty, i)`: where the decoder stores the `i`th coefficient of its scan
  (2D class: `DAV1D_SCANS`; H class: `i`; V class: row-major index turned column major).
- `last_nonzero(coefs, size, ty)`: the `eob` rav1d's inverse transforms take.

## What the encoder must know

- **`eob` is the scan position of the last nonzero coefficient** (0-based), as `decode_coefs`
  returns it and `itxfm_add[..].call(dst, cf, eob, bd)` takes it. libaom's `eob` is that plus
  one. All-zero blocks are skipped by the decoder (`eob < 0`), never passed to the transform.
  A larger `eob` than needed is always safe; a smaller one skips coefficients in the asm.
- **The inverse transform zeroes the coefficient buffer** (Rust fallback and asm). Copy first
  if the coefficients are still needed (token writing, RD) after reconstruction.
- 8-bit coefficients enter the decoder as `i16`; the tests keep the buffer `Align64` as the
  decoder's own buffers are, and its length exactly `cw * ch`.
- WHT coefficients carry libaom's `UNIT_QUANT_FACTOR` (×4); rav1d's inverse WHT shifts by 2.
  Lossless dequantizer is 4, so the coded level is `coef / 4`.
- Allowed by the bitstream (what rav1d's table fills): all 16 types up to 8×16/16×8/4×16/16×4,
  12 at 16×16 (no H/V ADST/FLIPADST), `DCT_DCT` + `IDTX` when the larger side is 32, only
  `DCT_DCT` when it is 64. `forward` also computes libaom's extra pairs (e.g. 32×32 `V_DCT`),
  which no decoder accepts.
- Test trap: `rav1d_get_cpu_flags()` returns only the compile-time flags (SSE2) unless
  `rav1d_init_cpu` ran, so a test using it never reaches the SSSE3/AVX2 kernels. The tests
  here use `CpuFlags::run_time_detect()`. `ec.rs`'s msac test uses `rav1d_get_cpu_flags()`,
  so it ran the SSE2 msac (which is the only x86 msac level, so it is still covered).

## Verified

All on `worker`, `ip-build:trixie`, stable 1.99.1, `--profile opt-dev --features encoder`.

| Job | What | Result |
|---|---|---|
| `mvloj0lja2pt9q6u` | `enc::fdct` tests at `cd830dfd` (first block), then the fault runs | 4 passed, 0 failed |
| `mvlod0uua3l0xn7o` | libaom C reference digests (harness below) | exit 0, 194 digests |
| `mvlog4hkk883gune` | 3 tests at `ced13160`, kernel levels 0xf, 0xf, 0x7, 0x0 printed | exit 0 |

Tests in `src/enc/fdct.rs`:

1. `matches_libaom`: FNV-1a over the coefficients of 10 blocks per size/type (all +255, all
   −255, 8 random in −255..255) equals the digest from libaom's own C functions for every one
   of the 194 pairs. **Bit-exact to libaom's C code.**
2. `round_trip_through_rav1d`: 64 random blocks (−127..127) plus ±127 constants, a ±127
   checkerboard and a −128 constant, forward, then rav1d's `itxfm_add` onto 128, at every
   kernel level of the CPU (worker: AVX2, SSSE3/SSE4.1, Rust fallback). 64-point sizes drop
   half the spectrum, so their blocks (−64..64) are first projected onto the coded half by one
   round trip, then must survive a second. Max |reconstruction − residual|, asserted per size
   and kernel level (all levels pass the same bounds):

   | size | bound | size | bound | size | bound |
   |---|---|---|---|---|---|
   | 4×4 | 0 | 4×8, 8×4 | 1 | 16×32, 32×16 | 2 |
   | 8×8 | 1 | 8×16, 16×8 | 1 | 32×64, 64×32 | 4 |
   | 16×16 | 1 | 4×16, 16×4 | 1 | 16×64, 64×16 | 2 |
   | 32×32 | 2 | 8×32, 32×8 | 1 | 64×64 | 3 |

   Many types reach 0 (identity types mostly); WHT 4×4 is exact (bound 0).
3. `sparse_blocks_take_the_shortcuts`: coefficients cut after scan positions 0, 1, 2, 5, 9,
   15, 35, 63, 99, 135, 255, 299, 511 and the last; reconstruction with
   `last_nonzero` as `eob` equals the one with the full `eob`, every size/type/kernel level.
4. `last_nonzero_follows_scan`: hand-checked positions for 2D, H and V classes.

## Faults injected (job `mvloj0lja2pt9q6u`)

| Fault | Caught by |
|---|---|
| FLIPADST without the up-down flip | `matches_libaom`, `round_trip_through_rav1d` |
| column round shift (`shift[1]`) dropped | `matches_libaom`, `round_trip_through_rav1d` |
| output transposed (`r * cw + c`) | `matches_libaom`, `round_trip_through_rav1d` |
| rectangular √2 scaling dropped | `matches_libaom`, `round_trip_through_rav1d` |
| `eob` one short | `sparse_blocks_take_the_shortcuts`, `last_nonzero_follows_scan` |
| V-class scan read as H class | `last_nonzero_follows_scan` only |
| one fdct8 constant `cospi[8]` → `cospi[9]` | `matches_libaom` only |

Not covered: the round trip is not sensitive enough for a constant off by 0.5%, only the
libaom digests are. The H/V-class scan order rests on reading `decode_coefs` plus the unit
test; the sparse test truncates with the same `scan_rc` it checks, so it cannot see a wrong
order. A bitstream decoded by rav1d will settle it once the coefficient writer exists.

Not done: 10/12-bit (the code is bit-depth independent, as libaom's; only 8-bit is tested),
speed (scalar, zero-fills a 16 KB buffer per call), SIMD.

## The libaom harness

The job ships the libaom clone (`libaom-v3.15.1:libaom-v3.15.1`) and runs, in `ip-build:trixie`:
`apt-get install cmake`, `cmake -S /aom -B build -DAOM_TARGET_CPU=generic -DENABLE_TESTS=0
-DENABLE_DOCS=0 -DENABLE_EXAMPLES=0 -DENABLE_TOOLS=0` (only for the config headers), then
`gcc -O1 -ffunction-sections -fdata-sections -I/aom -Ibuild harness.c
av1/encoder/av1_fwd_txfm1d.c av1/encoder/av1_fwd_txfm2d.c av1/common/av1_txfm.c
av1/encoder/hybrid_fwd_txfm.c -Wl,--gc-sections` (`--unresolved-symbols=ignore-all` instead
builds a binary that fails to load). `harness.c`:

```c
#include <stdint.h>
#include <stdio.h>

#include "config/av1_rtcd.h"

typedef void (*Fwd)(const int16_t *, int32_t *, int, TX_TYPE, int);

static const Fwd fwd[19] = {
  av1_fwd_txfm2d_4x4_c,   av1_fwd_txfm2d_8x8_c,   av1_fwd_txfm2d_16x16_c,
  av1_fwd_txfm2d_32x32_c, av1_fwd_txfm2d_64x64_c, av1_fwd_txfm2d_4x8_c,
  av1_fwd_txfm2d_8x4_c,   av1_fwd_txfm2d_8x16_c,  av1_fwd_txfm2d_16x8_c,
  av1_fwd_txfm2d_16x32_c, av1_fwd_txfm2d_32x16_c, av1_fwd_txfm2d_32x64_c,
  av1_fwd_txfm2d_64x32_c, av1_fwd_txfm2d_4x16_c,  av1_fwd_txfm2d_16x4_c,
  av1_fwd_txfm2d_8x32_c,  av1_fwd_txfm2d_32x8_c,  av1_fwd_txfm2d_16x64_c,
  av1_fwd_txfm2d_64x16_c,
};
static const int W[19] = { 4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64 };
static const int H[19] = { 4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16 };
/* 0 DCT, 1 ADST, 2 FLIPADST, 3 identity, as libaom's vtx_tab / htx_tab */
static const int vtx[16] = { 0, 1, 0, 1, 2, 0, 2, 1, 2, 3, 0, 3, 1, 3, 2, 3 };
static const int htx[16] = { 0, 0, 1, 1, 0, 2, 2, 2, 1, 3, 3, 0, 3, 1, 3, 2 };

static int has_1d(int kind, int n) {
  if (kind == 0) return 1;
  if (kind == 3) return n <= 32;
  return n <= 16;
}

static uint64_t s;
static uint32_t next(void) {
  s ^= s << 13;
  s ^= s >> 7;
  s ^= s << 17;
  return (uint32_t)(s >> 32);
}

int main(void) {
  static int16_t in[64 * 64];
  static int32_t out[64 * 64];
  for (int size = 0; size < 19; size++) {
    const int w = W[size], h = H[size];
    const int cw = w < 32 ? w : 32, ch = h < 32 ? h : 32;
    for (int ty = 0; ty < 17; ty++) {
      if (ty == 16 ? size != 0 : !(has_1d(vtx[ty], h) && has_1d(htx[ty], w)))
        continue;
      s = 0x9e3779b97f4a7c15ULL ^ ((uint64_t)size << 8 | (uint64_t)ty);
      uint64_t d = 0xcbf29ce484222325ULL;
      for (int b = 0; b < 10; b++) {
        for (int i = 0; i < w * h; i++)
          in[i] = b == 0 ? 255 : b == 1 ? -255 : (int16_t)(next() % 511) - 255;
        if (ty == 16)
          av1_fwht4x4_c(in, out, w);
        else
          fwd[size](in, out, w, (TX_TYPE)ty, 8);
        for (int i = 0; i < cw * ch; i++) {
          const uint32_t v = (uint32_t)out[i];
          for (int k = 0; k < 4; k++) {
            d ^= (v >> (8 * k)) & 0xff;
            d *= 0x100000001b3ULL;
          }
        }
      }
      printf("        (%d, %d, 0x%016llx),\n", size, ty, (unsigned long long)d);
    }
  }
  return 0;
}
```
