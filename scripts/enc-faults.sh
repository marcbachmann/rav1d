#!/bin/bash
# Runs the encoder tests on the unmodified tree, then once per fault in enc-faults.py,
# and reports which tests each fault failed. The tree is restored and compared after each.
set -u
run() {
    cargo test -p rav1d --profile opt-dev --features encoder --lib enc:: -- --nocapture > "$1" 2>&1
}
run /work/base.log
base=$?
grep -E "^rd |^dc |test result" /work/base.log
if [ $base -ne 0 ]; then
    grep -E "panicked|FAILED" /work/base.log
    echo "base failed"
    exit 1
fi
for f in $(python3 scripts/enc-faults.py list); do
    python3 scripts/enc-faults.py apply "$f" || exit 2
    run /work/fault.log
    e=$?
    python3 scripts/enc-faults.py restore "$f" || exit 3
    caught=$(grep -E "^test .* FAILED" /work/fault.log | sed 's/ \.\.\. FAILED//;s/^test //' | tr '\n' ' ')
    built=$(grep -c "^error" /work/fault.log)
    echo "fault $f: exit $e build-errors $built caught: ${caught:-BLIND}"
done
