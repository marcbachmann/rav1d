import os
import shutil
import sys

PATH = "src/enc/frame.rs"
SAVED = "/tmp/enc-fault-orig.rs"

FAULTS = {
    "pick-worst": ("if best.as_ref().is_none_or(|b| cost < b.0) {", "if best.as_ref().is_none_or(|b| cost > b.0) {"),
    "no-rate": ("let cost = sse(&src, &rec) as f64 + self.lambda * bits as f64 / 512.0;", "let cost = sse(&src, &rec) as f64 + 0.0 * bits as f64;"),
    "ctx-ly": ("let m = cmp::min(t_dim.h as i64, bh as i64 - y4 as i64) as usize;", "let m = cmp::min(t_dim.h as i64, bh as i64 - ly as i64) as usize;"),
    "no-lossless-guard": ("let zero = if lossless { f64::MAX } else {", "let zero = if false { f64::MAX } else {"),
}

cmd, name = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None
if cmd == "list":
    print("\n".join(FAULTS))
elif cmd == "apply":
    src = open(PATH).read()
    shutil.copyfile(PATH, SAVED)
    a, b = FAULTS[name]
    assert src.count(a) == 1, (name, src.count(a))
    os.unlink(PATH)
    open(PATH, "w").write(src.replace(a, b))
else:
    os.unlink(PATH)
    shutil.copyfile(SAVED, PATH)
    a, _ = FAULTS[name]
    assert open(PATH).read().count(a) == 1
