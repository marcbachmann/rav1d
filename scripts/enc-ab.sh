#!/bin/bash
# A/B of two encoder trees on one pin: builds the corpus bench of each (release, own target
# dirs under /deps) and runs them alternately $ROUNDS times with the same configuration.
# Usage: enc-ab.sh <tree A> <tree B> [VAR=value ...]
set -eu
a=$1 b=$2
shift 2
for t in "$a" "$b"; do
    (cd "$t" && CARGO_TARGET_DIR=/deps/target-ab-$(basename "$t") cargo test -p rav1d --release --features encoder --lib --no-run 2>&1 | tail -1)
done
for i in $(seq ${ROUNDS:-3}); do
    for t in "$a" "$b"; do
        echo "$(basename "$t") $(cd "$t" && env "$@" CARGO_TARGET_DIR=/deps/target-ab-$(basename "$t") cargo test -p rav1d --release --features encoder --lib enc::frame::tests::corpus -- --ignored --nocapture 2>&1 | grep '^total')"
    done
done
