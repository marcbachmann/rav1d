#!/bin/bash
# Compares src/enc/partition_cnn.rs with libaom's C: builds libaom's cnn.c, ml.c and the intra
# CNN model with scripts/cnn-ref/cnn-ref.c (a generic, C-only build, no FMA contraction), then
# runs enc::partition_cnn::tests::matches_libaom, which feeds random blocks and, with ENC_CORPUS,
# the corpus' superblocks to both and requires every output bit-identical.
# Usage: scripts/cnn-partition-compare.sh <libaom v3.15.1 source>
set -eu
aom=$(cd "$1" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
out=${CARGO_TARGET_DIR:-target}/cnn-ref
mkdir -p "$out"
{
    echo '#include <stdint.h>'
    echo '#define QINDEX_RANGE 256'
    sed -n '/^static const int16_t dc_qlookup_QTX\[/,/^};/p' "$aom/av1/common/quant_common.c"
} > "$out/dc_qlookup.h"
gcc -O2 -ffp-contract=off -I "$here/cnn-ref/stub" -I "$out" -I "$aom" -o "$out/cnn-ref" \
    "$here/cnn-ref/cnn-ref.c" "$aom/av1/encoder/cnn.c" "$aom/av1/encoder/ml.c" "$aom/aom_mem/aom_mem.c" "$aom/aom_util/aom_thread.c" -lm
CNN_REF="$out/cnn-ref" cargo test -p rav1d --release --features encoder --lib enc::partition_cnn::tests::matches_libaom -- --ignored --nocapture
