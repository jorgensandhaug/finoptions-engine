fn main() {
    let version = std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map_or_else(|| "unknown".to_owned(), |value| value.trim().to_owned());
    println!("cargo:rustc-env=RUSTC_VERSION={version}");
    println!("cargo:rerun-if-changed=build.rs");
}
