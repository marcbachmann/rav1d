import os
import shutil
import sys

SAVED = "/tmp/enc-fault-orig.rs"

FAULTS = {
    "cache-above-at-sb-top": ("src/enc/frame.rs", "let a_n = if ly != 0 { above.pal_sz[ax] as usize } else { 0 };", "let a_n = above.pal_sz[ax] as usize;"),
    "colors-bits-shrink": ("src/enc/palette.rs", "bits = cmp::min(bits, bits_for(255 - prev - not_pl));", "bits = cmp::min(bits, bits_for(255 - prev));"),
    "v-delta-sign": ("src/enc/palette.rs", "s.bool_equi(d < 0);", "s.bool_equi(d > 0);"),
    "index-order": ("src/enc/palette.rs", "let sym = order[m].iter().position(|&o| o == idx).unwrap();", "let sym = order[m].iter().position(|&o| o == idx).unwrap().min(n as usize - 2);"),
    "pal-uv-ctx": ("src/enc/frame.rs", "s.bool_adapt(n > 0, &mut cdf.m.pal_uv[(intra.pal_sz[0] > 0) as usize]);", "s.bool_adapt(n > 0, &mut cdf.m.pal_uv[0]);"),
    "map-fill-last-row": ("src/enc/frame.rs", "map.copy_within((h - 1) * bw..h * bw, r * bw);", "map.copy_within(0..bw, r * bw);"),
}

cmd, name = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None
if cmd == "list":
    only = os.environ.get("ENC_FAULTS")
    print("\n".join(f for f in FAULTS if not only or f in only.split(",")))
elif cmd == "apply":
    path, a, b = FAULTS[name]
    src = open(path).read()
    shutil.copyfile(path, SAVED)
    assert src.count(a) == 1, (name, src.count(a))
    os.unlink(path)
    open(path, "w").write(src.replace(a, b))
else:
    path, a, _ = FAULTS[name]
    os.unlink(path)
    shutil.copyfile(SAVED, path)
    assert open(path).read().count(a) == 1
