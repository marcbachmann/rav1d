/// `rav1d_sdot` where the aarch64 `sdot` intrinsics build: stable from Rust 1.98, behind
/// `stdarch_neon_dotprod` (then `rav1d_sdot_feature`) on an older nightly.
fn sdot_cfg() {
    println!("cargo:rustc-check-cfg=cfg(rav1d_sdot, rav1d_sdot_feature)");
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let Ok(out) = std::process::Command::new(rustc).arg("--version").output() else {
        return;
    };
    let version = String::from_utf8_lossy(&out.stdout);
    let minor = version
        .split_whitespace()
        .nth(1)
        .and_then(|v| v.split('.').nth(1))
        .and_then(|m| m.parse::<u32>().ok())
        .unwrap_or(0);
    if minor >= 98 {
        println!("cargo:rustc-cfg=rav1d_sdot");
    } else if version.contains("nightly") || version.contains("-dev") {
        println!("cargo:rustc-cfg=rav1d_sdot");
        println!("cargo:rustc-cfg=rav1d_sdot_feature");
    }
}

fn main() {
    sdot_cfg();
}
