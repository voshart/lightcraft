//! Workspace tooling: `cargo xtask <command>`.
//!
//! Pure Rust (std + serde_json; flate2/brotli for the web bundle; sha2 to check corpus downloads). External tools (`cargo`, `curl`, `tar`) are
//! invoked through `std::process::Command`.

mod assets;
mod bench;
mod ico;
mod layers;
mod parity;
mod stats;
mod version;
mod web;

use std::path::PathBuf;
use std::process::{Command, ExitCode};

const USAGE: &str = "\
usage: cargo xtask <command>

commands:
  assets          every image/icon/font/media file is attributed in assets/ATTRIBUTION.md; no Adobe assets
  bench [FILE] [--strict] [--threshold PCT]
                  run the render benchmark, append to target/bench/history.jsonl, compare CPU time with
                  the previous run (default input: corpus/raw/arw-sony-a7m3-compressed.arw)
  ico <out.ico> <in.png>...
                  pack square PNGs (<= 256 px) into a Windows .ico (see packaging/icons.sh)
  layers          enforce the crate dependency layering (plan/architecture.md §3)
  parity [--write]
                  check docs/parity.md (every cmd:/ctl: id and path it cites exists) and print the
                  Lightroom parity summary; --write refreshes the summary table in the document
  wasm            cargo check --target wasm32-unknown-unknown for the wasm-safe crates (+ the web app)
  web [--serve [port]] [--dev]
                  build the browser app (apps/lightcraft-web) into <target>/web/;
                  --serve serves it on http://127.0.0.1:<port> (default 8080)
  ci              fmt --check, clippy -D warnings, test, parity refs, layers, assets, wasm (stops at first failure)
  corpus [--download]
                  show where test corpora live; --download fetches the CC0 raw samples (raw.pixls.us) and the
                  public face gallery (Wikimedia Commons) into corpus/, each checked against its pinned SHA-256
  stats [--exact] count tests and lines per crate (--exact: ask the test harness via `-- --list`)
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rest: Vec<&str> = args.iter().skip(1).map(String::as_str).collect();
    let result = match args.first().map(String::as_str) {
        Some("ico") => ico::run(&rest),
        Some("version") => version::run(&root(), &rest),
        Some("layers") => cmd_layers(),
        Some("assets") => assets::run(&root()),
        Some("bench") => bench::run(&root(), &rest),
        Some("parity") => parity::run(&root(), rest.contains(&"--write")),
        Some("wasm") => cmd_wasm(),
        Some("web") => web::run(&rest),
        Some("ci") => cmd_ci(),
        Some("corpus") => cmd_corpus(rest.contains(&"--download")),
        Some("stats") => stats::run(&root(), rest.contains(&"--exact")),
        Some("-h" | "--help" | "help") | None => {
            print!("{USAGE}");
            Ok(())
        }
        Some(other) => Err(format!("unknown command `{other}`\n\n{USAGE}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Workspace root (parent of the xtask crate).
pub fn root() -> PathBuf {
    // `cargo run` sets it at run time; prefer that over the compile-time value, which goes stale
    // when the checkout moves and the cached xtask binary isn't rebuilt
    let dir = std::env::var_os("CARGO_MANIFEST_DIR").map(PathBuf::from).filter(|d| d.join("Cargo.toml").is_file());
    let dir = dir.unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    dir.parent().expect("xtask has a parent dir").to_path_buf()
}

pub fn cargo() -> Command {
    let mut c = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    c.current_dir(root());
    c
}

pub fn run(mut cmd: Command, what: &str) -> Result<(), String> {
    eprintln!("$ {what}");
    let status = cmd.status().map_err(|e| format!("{what}: failed to spawn: {e}"))?;
    if status.success() { Ok(()) } else { Err(format!("{what}: exited with {status}")) }
}

pub fn metadata() -> Result<serde_json::Value, String> {
    let out = cargo().args(["metadata", "--format-version", "1", "--no-deps"]).output().map_err(|e| format!("cargo metadata: {e}"))?;
    if !out.status.success() {
        return Err(format!("cargo metadata failed:\n{}", String::from_utf8_lossy(&out.stderr)));
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("cargo metadata: bad JSON: {e}"))
}

fn cmd_layers() -> Result<(), String> {
    let crates = layers::from_metadata(&metadata()?)?;
    println!("Dependency layering (plan/architecture.md §3)\n");
    println!("{:<28} {:<14} workspace deps", "crate", "layer");
    for c in &crates {
        let ws: Vec<String> = c
            .deps
            .iter()
            .filter(|d| d.workspace)
            .map(|d| {
                let k = match d.kind {
                    layers::DepKind::Normal => "",
                    layers::DepKind::Dev => " (dev)",
                    layers::DepKind::Build => " (build)",
                };
                format!("{}{k}", layers::short_name(&d.name))
            })
            .collect();
        println!("{:<28} {:<14} {}", c.name, layers::describe(layers::classify(&c.name)), ws.join(", "));
    }
    let violations = layers::check(&crates);
    println!();
    if violations.is_empty() {
        println!("OK: {} crates, no layering violations.", crates.len());
        Ok(())
    } else {
        println!("{} violation(s):", violations.len());
        for v in &violations {
            println!("  - {v}");
        }
        Err(format!("{} layering violation(s)", violations.len()))
    }
}

/// Workspace packages that must build for wasm32: all L0–L5 crates plus
/// the egui shell and the web app.
fn wasm_set() -> Result<Vec<String>, String> {
    let crates = layers::from_metadata(&metadata()?)?;
    Ok(crates
        .into_iter()
        .filter(|c| match layers::classify(&c.name) {
            Some(layers::Class::Layer(l)) => l <= 5,
            Some(layers::Class::Standalone) => true,
            _ => false,
        })
        .map(|c| c.name)
        .chain(std::iter::once("lightcraft-web".to_string()))
        .collect())
}

fn cmd_wasm() -> Result<(), String> {
    let set = wasm_set()?;
    let mut results = Vec::new();
    for pkg in &set {
        let mut c = cargo();
        c.args(["check", "--target", "wasm32-unknown-unknown", "-p", pkg]);
        let ok = run(c, &format!("cargo check --target wasm32-unknown-unknown -p {pkg}")).is_ok();
        results.push((pkg.clone(), ok));
    }
    println!("\nwasm32-unknown-unknown check:");
    for (p, ok) in &results {
        println!("  {:<6} {p}", if *ok { "ok" } else { "FAIL" });
    }
    let failed = results.iter().filter(|r| !r.1).count();
    if failed == 0 { Ok(()) } else { Err(format!("{failed} crate(s) failed the wasm check")) }
}

fn cmd_ci() -> Result<(), String> {
    type Step = (&'static str, Box<dyn Fn() -> Result<(), String>>);
    let steps: Vec<Step> = vec![
        (
            "fmt",
            Box::new(|| {
                let mut c = cargo();
                c.args(["fmt", "--all", "--", "--check"]);
                run(c, "cargo fmt --all -- --check")
            }),
        ),
        (
            "clippy",
            Box::new(|| {
                let mut c = cargo();
                c.args(["clippy", "--workspace", "--all-targets", "--", "-D", "warnings"]);
                run(c, "cargo clippy --workspace --all-targets -- -D warnings")
            }),
        ),
        (
            "test",
            Box::new(|| {
                let mut c = cargo();
                c.args(["test", "--workspace"]);
                run(c, "cargo test --workspace")
            }),
        ),
        ("parity", Box::new(|| parity::run(&root(), false))),
        ("layers", Box::new(cmd_layers)),
        ("assets", Box::new(|| assets::run(&root()))),
        ("wasm", Box::new(cmd_wasm)),
    ];
    let mut done = Vec::new();
    for (name, f) in &steps {
        eprintln!("\n=== ci: {name} ===");
        if let Err(e) = f() {
            println!("\nCI summary:");
            for d in &done {
                println!("  ok    {d}");
            }
            println!("  FAIL  {name}: {e}");
            for (n, _) in steps.iter().skip(done.len() + 1) {
                println!("  skip  {n}");
            }
            return Err(format!("ci failed at `{name}`"));
        }
        done.push(*name);
    }
    println!("\nCI summary: all {} steps passed ({})", done.len(), done.join(", "));
    Ok(())
}

/// CC0 raw samples from raw.pixls.us (each verified CC0 on the site): name, address, SHA-256 of the file as downloaded.
/// One per format / compression variant we decode or deliberately report as unsupported (preview only). Extend freely
/// (CC0 only): download the file once and pin its SHA-256 here.
const RAW_SAMPLES: &[(&str, &str, &str)] = &[
    (
        "arw-sony-a7m3-compressed.arw",
        "https://raw.pixls.us/getfile.php/2414/nice/Sony%20-%20ILCE-7M3%20-%2014bit%2014bit%20compressed%20%283:2%29.ARW",
        "250784580ea527442c09004417bb0eead484f2bf3ee8f9121a776ac65bb50d0f",
    ),
    (
        "arw-sony-a7m3-uncompressed.arw",
        "https://raw.pixls.us/getfile.php/2418/nice/Sony%20-%20ILCE-7M3%20-%2014bit%2014bit%20uncompressed%20%283:2%29.ARW",
        "ece80551abf64949dbe826985a80f1d9b265401ee4ea6b989e1829549616fdcd",
    ),
    (
        "arw-sony-a7m4-14bit.arw",
        "https://raw.pixls.us/getfile.php/6936/nice/Sony%20-%20ILCE-7M4%20-%2014bit%20%283:2%29.ARW",
        "639a6d4db881f1359e3ea7e1137b314e59417d24e6d4d862b7202111a2c823b2",
    ),
    // lossless compressed (Compression 7): L is 2×2 CFA cells per LJ92 sample; M and S are subsampled
    (
        "arw-sony-a7m4-lossless-l.arw",
        "https://raw.pixls.us/data/Sony/ILCE-7M4/ILCE-7M4_DSC06674_FullFrame-LossLess-Compressed-Large.ARW",
        "851b43c2116c4139104a5036f83ac3b6a148789b2142214dd7192c13972b25b6",
    ),
    (
        "arw-sony-a7m4-lossless-m.arw",
        "https://raw.pixls.us/data/Sony/ILCE-7M4/ILCE-7M4_DSC06675_FullFrame-LossLess-Compressed-Medium.ARW",
        "d453005714327addd75bcb99c1c6223173dc92f2a59542d82e6761ef0e0e7571",
    ),
    (
        "arw-sony-a7m4-lossless-s.arw",
        "https://raw.pixls.us/data/Sony/ILCE-7M4/ILCE-7M4_DSC06676_FullFrame-LossLess-Compressed-Small.ARW",
        "cbbd0930c7d8706dff84c68a2004454266e6fd0d8354f5f76a106b5d776e0223",
    ),
    // pre-2017 bodies: white balance only in the enciphered maker note, black level only in the SR2SubIFD (#148)
    (
        "arw-sony-rx100m3.arw",
        "https://raw.pixls.us/data/Sony/DSC-RX100M3/DSC00734.ARW",
        "cb81392e3a8810231ce341fb45de3e4d1b9e2f87bc47f3c314d084041ccfd138",
    ),
    (
        "arw-sony-rx100.arw",
        "https://raw.pixls.us/data/Sony/DSC-RX100/DSC00838.ARW",
        "579a485b5126a25cbd55cbd5dadfa7d09cf021c99cc7d4869f9e56e3f759390b",
    ),
    (
        "arw-sony-a7rm2-12bit-uncompressed.arw",
        "https://raw.pixls.us/data/Sony/ILCE-7RM2/12-bit-uncompressed.ARW",
        "71e0888396ef52c7e6a990b4e72ebb4d8413a1fc1c35bada449870539cef6e39",
    ),
    // CR2 colour-filter layouts differ by model (issue #85): CR2CFAPattern 3 (GBRG) and 1 (RGGB) samples
    (
        "cr2-canon-40d.cr2",
        "https://raw.pixls.us/data/Canon/EOS%2040D/_MG_0153.CR2",
        "775c806358fedddec7622113a5e399a3330bd5a65e35e37ae1108d8bf58067be",
    ),
    (
        "cr2-canon-550d.cr2",
        "https://raw.pixls.us/data/Canon/EOS%20550D/IMG_4047.CR2",
        "f390e0ba566f0be9af2d8e7f955b815996630c0c3cb13fcfb95d98baec1890f6",
    ),
    (
        "cr2-canon-5d2.cr2",
        "https://raw.pixls.us/data/Canon/EOS%205D%20Mark%20II/08.canon.raw.cr2",
        "12b3ad6aed9fbcd19ba3bfffcd0769aa5a041d610205eee840bf4b2c420dffae",
    ),
    (
        "cr2-canon-5dsr.cr2",
        "https://raw.pixls.us/data/Canon/EOS%205DS%20R/_DSR2002.CR2",
        "01b94b0586419d64894ccdf2e7b9f1defeedc7b51dfcd306c622fe784181a9b8",
    ),
    (
        "cr2-canon-6d.cr2",
        "https://raw.pixls.us/data/Canon/EOS%206D/EOS_6D_RAW.CR2",
        "360842779d08fe4805b4a2fe4979f06fdc6da982c85c3acb181f25bcf4c9f536",
    ),
    (
        "cr2-canon-7d.cr2",
        "https://raw.pixls.us/data/Canon/EOS%207D/RAW_CANON_EOS_7D-raw.CR2",
        "b5e47c5fcf7332ac03e0134926f17a338a42e68c1fd7f83e16f45f4b767544e8",
    ),
    (
        "cr2-canon-5d3-sraw2.cr2",
        "https://raw.pixls.us/getfile.php/773/nice/Canon%20-%20EOS%205D%20Mark%20III%20-%20sRAW2%20%28sRAW%29.CR2",
        "a79065e78a6c5bdbc8726e786ade02d54a7d496cc3cb4504d804c924a3d999da",
    ),
    (
        "cr2-canon-5d3.cr2",
        "https://raw.pixls.us/getfile.php/771/nice/Canon%20-%20EOS%205D%20Mark%20III.CR2",
        "ec069b178b9383d80c72f8cdc1b79bd54ca451e7c20793b109d8ad73bef6bdf6",
    ),
    (
        "cr2-canon-80d.cr2",
        "https://raw.pixls.us/getfile.php/1294/nice/Canon%20-%20EOS%2080D%20-%20RAW%20%283:2%29.CR2",
        "8f28465cee09844ffad95dbd3ec91923717a329c5ec4f3f405f5b4fce83aaea3",
    ),
    (
        "cr3-canon-m50-craw.cr3",
        "https://raw.pixls.us/getfile.php/2663/nice/Canon%20-%20EOS%20M50%20-%20CRAW%20%283:2%29.CR3",
        "15384b775867ec4c42b11882837f1e368cedc0561832ffab271221e6bb80be4c",
    ),
    (
        "dng-adobe-canon-5d3-linear-lj92.dng",
        "https://raw.pixls.us/getfile.php/1032/nice/Adobe%20DNG%20Converter%20-%20Canon%20EOS%205D%20Mark%20III%20-%20Lossless%20JPEG%20compression%2C%20rgb%20%283:2%29.DNG",
        "0608c9f277307b745530e61b589c465540611df22f9a1fe2d01237c5b2b4c13b",
    ),
    (
        "dng-adobe-canon-5d3-lj92.dng",
        "https://raw.pixls.us/getfile.php/1024/nice/Adobe%20DNG%20Converter%20-%20Canon%20EOS%205D%20Mark%20III%20-%2016bit%2016bit%20Lossless%20JPEG%20compression%20%283:2%29.DNG",
        "3118116735d4f6dd01fd9d6a80ee9aac52a2c2debdc6dcd112bd5e3059044a55",
    ),
    (
        "dng-adobe-canon-5d3-lossy.dng",
        "https://raw.pixls.us/getfile.php/1023/nice/Adobe%20DNG%20Converter%20-%20Canon%20EOS%205D%20Mark%20III%20-%20Lossy%20JPEG%20compression%20%283:2%29.DNG",
        "159326856c29073e845c3c5a9ecf98c6474f43ca15798a88ad5e2baecd0664b7",
    ),
    (
        "dng-canon-5d3-14bit-small.dng",
        "https://raw.pixls.us/getfile.php/2204/nice/Canon%20-%20EOS%205D%20Mark%20III%20-%2014bit%2014bit%20%282.3471882640587%29.dng",
        "1d77dcc6cdb839b10c1948f0b4471984ab48cf15e973fb77520584cee72b603e",
    ),
    (
        "dng-canon-5d3-16bit-169.dng",
        "https://raw.pixls.us/getfile.php/2649/nice/Canon%20-%20EOS%205D%20Mark%20III%20-%2016bit%20%2816:9%29.dng",
        "bc29ac8a37c800346749b9a5a1adc2722357f7f0e49df05a7104454881604b92",
    ),
    (
        "dng-canon-5d3-16bit.dng",
        "https://raw.pixls.us/getfile.php/885/nice/Canon%20-%20EOS%205D%20Mark%20III%20-%2016bit%2016bit%20RAW.dng",
        "6275226bfa2d86ba479130999d045ea0b8da6be4b011130c9eb91cd6a0eb2233",
    ),
    (
        "dng-google-pixel2xl.dng",
        "https://raw.pixls.us/getfile.php/2206/nice/Google%20-%20Pixel%202%20XL%20-%2016bit%20%284:3%29.dng",
        "5541093a369f0487fef92bcd1f5fef1be6e4e3cec2aa4d52ad38d2211f3309d7",
    ),
    (
        "dng-ricoh-gr3.dng",
        "https://raw.pixls.us/getfile.php/3115/nice/Ricoh%20-%20GR%20III%20-%2014bit%20%283:2%29.DNG",
        "05513ee72f7cac4165534c9f64995412084f414a033c80c473c79fd94154292e",
    ),
    (
        "nef-nikon-d5100-lossless.nef",
        "https://raw.pixls.us/getfile.php/1597/nice/Nikon%20-%20D5100%20-%2014bit%2014bit%20compressed%20%28Lossless%29%20%283:2%29.nef",
        "3ce328830942381648be26b372a54b317fdb27b63c1aa9c2ee1fce076a2e1b4f",
    ),
    (
        "nef-nikon-d5100-uncompressed.nef",
        "https://raw.pixls.us/getfile.php/1598/nice/Nikon%20-%20D5100%20-%2014bit%2014bit%20uncompressed%20%283:2%29.nef",
        "2fb2f1b87a3b89de5ba9e56527f21a912ca624f2fa2a594cef108ee8d7180412",
    ),
    (
        "nef-nikon-d7000-lossy12.nef",
        "https://raw.pixls.us/getfile.php/961/nice/Nikon%20-%20D7000%20-%2012bit%2012bit%20compressed%20%28Lossy%20%28type%202%29%29%20%283:2%29.NEF",
        "986db2bf9ed08cf095bb94cfbaf02f0e89d123edad28b63c05133e7815749486",
    ),
    (
        "nrw-nikon-b700-uncompressed.nrw",
        "https://raw.pixls.us/getfile.php/1621/nice/Nikon%20-%20COOLPIX%20B700%20-%2012bit%2012bit%20uncompressed%20%284:3%29.NRW",
        "a175d5892304e79282add07eacf4bdc498ca3f050af48cf989c080355b4735ec",
    ),
    (
        "orf-olympus-e1.orf",
        "https://raw.pixls.us/getfile.php/1800/nice/Olympus%20-%20E-1%20-%2016bit%20%284:3%29.ORF",
        "042286653fbae5b085bef4a4e626145385ea6e82e817f166b4904abcef64457c",
    ),
    (
        "orf-olympus-e400.orf",
        "https://raw.pixls.us/getfile.php/2151/nice/Olympus%20-%20E-400%20-%2016bit%20%284:3%29.ORF",
        "1354e0ac98ea90820227650031d8215637a2aaabd15de995c5a1e00b1986261b",
    ),
    (
        "orf-olympus-em1.orf",
        "https://raw.pixls.us/getfile.php/1051/nice/Olympus%20-%20E-M1%20-%2016bit%20%284:3%29.orf",
        "19ba17e778ae802d4039f0e88bb1cd68825780788b8058c479067e32b9a63eda",
    ),
    (
        "orf-olympus-em10iii.orf",
        "https://raw.pixls.us/getfile.php/1787/nice/Olympus%20-%20E-M10%20Mark%20III%20-%2016bit%20%284:3%29.ORF",
        "11d90cbb564ad2c660c7a8d5d81d99c2c018aa27fdec45ed6c461fa5d24f942b",
    ),
    (
        "orf-olympus-xz2.orf",
        "https://raw.pixls.us/getfile.php/1432/nice/Olympus%20-%20XZ-2%20-%2012bit%20%284:3%29.orf",
        "675fefe7c9e99281783b987889d723b5f46692f5c20e19fed5e422c7c6d2636b",
    ),
    (
        "pef-pentax-k10d.pef",
        "https://raw.pixls.us/getfile.php/2239/nice/Pentax%20-%20K10D%20-%2012bit%2012bit%20compressed%20%283:2%29.PEF",
        "e35ae4154a468be3154f5f462e884ba5941f010d3e8f23d347fbec14809f44d3",
    ),
    (
        "pef-pentax-k3.pef",
        "https://raw.pixls.us/getfile.php/1075/nice/Pentax%20-%20K-3%20-%2014bit%20%283:2%29.PEF",
        "abfb3907b53734b5f353e8de0c6173470c7aad07f9cecc10f723d222a49dec9a",
    ),
    (
        "pef-pentax-k5iis.pef",
        "https://raw.pixls.us/getfile.php/1198/nice/Pentax%20-%20K-5%20II%20s%20-%2014bit%20%283:2%29.PEF",
        "e579e7360c35e512f9c3a5c0eb9519575b87b489c6f86be82590f7706a7ed17b",
    ),
    (
        "raf-fuji-xa2-12bit-bayer.raf",
        "https://raw.pixls.us/getfile.php/2883/nice/Fujifilm%20-%20X-A2%20-%2012bit%2012bit%20uncompressed%20%283:2%29.RAF",
        "58ad687b23138c3c49ebeb6c1f80daf13185349be364b947e2bf27e39a462d2f",
    ),
    (
        "raf-fuji-xa5-14bit-bayer.raf",
        "https://raw.pixls.us/getfile.php/2526/nice/Fujifilm%20-%20X-A5%20-%2014bit%2014bit%20uncompressed%20%283:2%29.RAF",
        "46daaf8c2b07f40bd01f737181d048a497bb0a87615a1fde65809fdd5de86bf1",
    ),
    (
        "raf-fuji-xe1-12bit.raf",
        "https://raw.pixls.us/getfile.php/3098/nice/Fujifilm%20-%20X-E1%20-%2012bit%2012bit%20uncompressed%20%283:2%29.RAF",
        "0fa01fa1a674bcfa40f659225b4e66c5bedf15d2f78407ca1966dc8f38cd0e9f",
    ),
    (
        "raf-fuji-xt20-14bit.raf",
        "https://raw.pixls.us/getfile.php/1177/nice/Fujifilm%20-%20X-T20%20-%2014bit%2014bit%20uncompressed%20%283:2%29.RAF",
        "afe552f4a2794c7aa2376dd826165354b243dcf6f2296b01505dded8c8e4be76",
    ),
    (
        "raf-fuji-xt20-compressed.raf",
        "https://raw.pixls.us/getfile.php/1178/nice/Fujifilm%20-%20X-T20%20-%2014bit%2014bit%20compressed%20%283:2%29.RAF",
        "a23045101b2912e64f154100d646c228c0d16585d21e95e0d1a006e537f7784c",
    ),
    (
        "rw2-panasonic-g9-b.rw2",
        "https://raw.pixls.us/getfile.php/2348/nice/Panasonic%20-%20DC-G9%20-%204:3.RW2",
        "16282357ce0143ff7d36737f82074662db7aa0790ecd309196f406baf0f586a7",
    ),
    (
        "rw2-panasonic-g9.rw2",
        "https://raw.pixls.us/getfile.php/2585/nice/Panasonic%20-%20DC-G9%20-%204:3.RW2",
        "7a56babeb19f9bde3eaa532a4c8d1b11637a0f550963a1182ec845722c65583d",
    ),
    (
        "rw2-panasonic-gh5.rw2",
        "https://raw.pixls.us/getfile.php/1517/nice/Panasonic%20-%20DC-GH5%20-%204:3.RW2",
        "dae2281df393d8c3c1b8c513d00985f1834ae0e69ce18fbd571370c25d95d22b",
    ),
    (
        "rw2-panasonic-gh5s.rw2",
        "https://raw.pixls.us/getfile.php/2603/nice/Panasonic%20-%20DC-GH5S%20-%204:3.RW2",
        "b617151876e62b492641cef14b8c98ee3797e5014d24c6af1c780715a501fa53",
    ),
    (
        "rw2-panasonic-gx80.rw2",
        "https://raw.pixls.us/getfile.php/1569/nice/Panasonic%20-%20DMC-GX80%20-%204:3.RW2",
        "6e9a419c70c2124630912b2934027c635d3871e3d22dd91fed0655d58cfb5382",
    ),
];

/// The public face gallery (`corpus/public-faces/`); its header says what it is.
const PUBLIC_FACES: &str = include_str!("../corpus/public-faces.tsv");
/// Sent with every download: Wikimedia asks tools to say who they are.
const USER_AGENT: &str = "LightCraft-xtask/0.1 (test corpus download; https://github.com/storytold/lightcraft)";

/// One file of a downloadable corpus, pinned by its SHA-256.
struct Pinned<'a> {
    name: &'a str,
    url: &'a str,
    sha256: &'a str,
    /// Licence, author and source page, kept beside the files in SOURCES.tsv.
    credit: Option<[&'a str; 3]>,
}

fn raw_samples() -> Vec<Pinned<'static>> {
    RAW_SAMPLES.iter().map(|&(name, url, sha256)| Pinned { name, url, sha256, credit: None }).collect()
}

/// The gallery list: one `file, url, sha256, bytes, licence, author, source page` line per image; `#` lines are notes.
fn public_faces() -> Result<Vec<Pinned<'static>>, String> {
    PUBLIC_FACES
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .enumerate()
        .map(|(i, l)| {
            let c: Vec<&str> = l.split('\t').collect();
            match c[..] {
                [name, url, sha256, _bytes, licence, author, page] => Ok(Pinned { name, url, sha256, credit: Some([licence, author, page]) }),
                _ => Err(format!("xtask/corpus/public-faces.tsv: entry {} has {} columns, not 7", i + 1, c.len())),
            }
        })
        .collect()
}

fn sha256_file(path: &std::path::Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (mut h, mut buf) = (Sha256::new(), vec![0u8; 1 << 20]);
    loop {
        let n = f.read(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
        match buf.get(..n) {
            Some(chunk) if n > 0 => h.update(chunk),
            _ => break,
        }
    }
    Ok(format!("{:x}", h.finalize()))
}

/// Make sure `dir` holds `p`: a file already there with the pinned SHA-256 is kept, anything else is downloaded (to a
/// `.part` file, checked, then moved into place). `Ok(true)` when it was downloaded now.
fn fetch_pinned(dir: &std::path::Path, p: &Pinned) -> Result<bool, String> {
    let out = dir.join(p.name);
    if out.is_file() {
        if sha256_file(&out)? == p.sha256 {
            return Ok(false);
        }
        eprintln!("{}: not the pinned file (its SHA-256 differs), downloading it again", p.name);
    }
    let part = dir.join(format!("{}.part", p.name));
    let mut curl = Command::new("curl");
    curl.args(["-fsSL", "--retry", "2", "-A", USER_AGENT, "-o"]).arg(&part).arg(p.url);
    let checked = run(curl, &format!("curl {}", p.url)).and_then(|()| {
        let got = sha256_file(&part)?;
        if got == p.sha256 { Ok(()) } else { Err(format!("{}: the download's SHA-256 is {got}, not the pinned {} ({})", p.name, p.sha256, p.url)) }
    });
    match checked {
        Ok(()) => std::fs::rename(&part, &out).map(|()| true).map_err(|e| format!("{}: {e}", out.display())),
        Err(e) => {
            let _ = std::fs::remove_file(&part);
            Err(e)
        }
    }
}

fn cmd_corpus(download: bool) -> Result<(), String> {
    let corpus = root().join("corpus");
    println!(
        "Test corpora live under {} (git-ignored, never committed). `--download` fetches them from their sources
and checks every file against the SHA-256 pinned in xtask; tests that use a corpus skip cleanly when it is absent.

  corpus/raw/           raw.pixls.us samples (CC0), one per format and compression variant: lightcraft-raw decodes
                        every file, and the AI Denoise tests use the Bayer ones
  corpus/public-faces/  public-domain / CC0 photographs, paintings and busts from Wikimedia Commons, with and without
                        faces, credited in its SOURCES.tsv (the list: xtask/corpus/public-faces.tsv)
  corpus/images/        CC0 / public-domain JPEG/PNG/TIFF/HEIC samples (optional, not downloaded)

Only public files go in corpus/: anything else stays outside the repository, where git, GitHub and the tools
working in the repository cannot pick it up.
",
        corpus.display()
    );
    if !download {
        return Ok(());
    }
    let sets = [("raw", raw_samples()), ("public-faces", public_faces()?)];
    let mut failed = Vec::new();
    for (sub, set) in &sets {
        let dir = corpus.join(sub);
        std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        let (mut fetched, mut kept) = (0, 0);
        for p in set {
            match fetch_pinned(&dir, p) {
                Ok(true) => fetched += 1,
                Ok(false) => kept += 1,
                Err(e) => failed.push(e),
            }
        }
        let credits: Vec<String> = set
            .iter()
            .filter_map(|p| p.credit.map(|[licence, author, page]| format!("{}\t{licence}\t{author}\t{page}\t{}", p.name, p.url)))
            .collect();
        if !credits.is_empty() {
            let sources = dir.join("SOURCES.tsv");
            std::fs::write(&sources, format!("file\tlicence\tauthor\tsource page\turl\n{}\n", credits.join("\n")))
                .map_err(|e| format!("{}: {e}", sources.display()))?;
        }
        println!("corpus/{sub}/: {fetched} downloaded, {kept} already there and checked ({} in the list)", set.len());
    }
    if failed.is_empty() {
        return Ok(());
    }
    for e in &failed {
        eprintln!("  - {e}");
    }
    Err(format!("{} file(s) could not be downloaded or did not match their SHA-256 (listed above); run it again to retry", failed.len()))
}

#[cfg(test)]
mod corpus_tests {
    use super::*;

    #[test]
    fn every_download_is_pinned_and_named_once() {
        let faces = public_faces().unwrap();
        assert_eq!(faces.len(), 45, "the gallery list parses");
        let mut names = std::collections::HashSet::new();
        for p in raw_samples().iter().chain(&faces) {
            assert!(p.sha256.len() == 64 && p.sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)), "{}", p.name);
            assert!(p.url.starts_with("https://"), "{}", p.name);
            assert!(!p.name.is_empty() && !p.name.starts_with('.') && !p.name.contains(['/', '\\']), "{}", p.name);
            assert!(names.insert(p.name), "{} is listed twice", p.name);
        }
        for p in &faces {
            let [licence, _, page] = p.credit.unwrap();
            assert!(licence == "CC0" || licence == "Public domain", "{}: {licence}", p.name);
            assert!(page.starts_with("https://commons.wikimedia.org/wiki/File:"), "{}", p.name);
        }
    }
}
