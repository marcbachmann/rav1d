#!/bin/bash
# Builds the encoder's tests in release once, then runs the corpus bench for each
# configuration given as an argument: a space-separated list of VAR=value.
set -eu
cargo test -p rav1d --release --features encoder --lib --no-run 2>&1 | tail -1
for config in "$@"; do
    echo "== $config"
    env $config cargo test -p rav1d --release --features encoder --lib enc::frame::tests::corpus -- --ignored --nocapture 2>&1 | grep -E " ms |^total|panicked|searches"
done
