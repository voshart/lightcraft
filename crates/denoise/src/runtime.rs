//! Running an installed denoise model on the CPU with `tract`, behind the `tract` feature. tract is Rust, but its
//! build assembles hand-written speed kernels (and on Linux aarch64 may compile optional C ones); a build without
//! the feature has none of that.
//!
//! A model is a stranger's file, so loading and running it are guarded: tract's errors become ours, a panic inside
//! tract (a large library reading hostile graphs) is caught and reported as an error, the answer must have the
//! size the manifest promised, and a model that produces nothing useful fails the [self-test](TractRunner::self_test)
//! before it is ever used on a photo.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;
use tract_onnx::prelude::*;

use crate::manifest::{DenoiserManifest, Domain, Gain};
use crate::run::{Error, TileRunner};

type Plan = TypedRunnableModel;

/// A loaded model that takes `[1, 4, tile, tile]` and gives `[1, 3, 2·tile, 2·tile]`.
#[derive(Clone)]
pub struct TractRunner {
    plan: Arc<Plan>,
    manifest: DenoiserManifest,
    load_ms: f64,
}

/// What `self_test` found.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SelfTest {
    pub ok: bool,
    /// Milliseconds to load and optimise the model.
    pub load_ms: f64,
    /// Median milliseconds to run one tile.
    pub tile_ms: f64,
    /// Megapixels of picture one tile covers, for estimating a whole photo's time.
    pub tile_megapixels: f64,
    /// The model's output scale relative to its input on the test picture (what `Gain::MatchMean` takes out).
    pub scale: f64,
    /// Each check: what was checked and whether it held.
    pub checks: Vec<(String, bool)>,
}

fn guarded<T>(what: &str, f: impl FnOnce() -> Result<T, Error>) -> Result<T, Error> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(_) => Err(Error::Runtime(format!("the runtime gave up on this model while {what}"))),
    }
}

impl TractRunner {
    /// Load the `.onnx` at `path` as described by `manifest`.
    pub fn load(path: &Path, manifest: &DenoiserManifest) -> Result<TractRunner, Error> {
        let Domain::BayerToRgb = manifest.domain;
        let t = manifest.tile as usize;
        let started = Instant::now();
        let plan = guarded("loading it", || {
            tract_onnx::onnx()
                .model_for_path(path)
                .and_then(|m| m.with_input_fact(0, f32::fact([1, 4, t, t]).into()))
                .and_then(|m| m.into_optimized())
                .and_then(|m| m.into_runnable())
                .map_err(|e| Error::Load(format!("{e:#}")))
        })?;
        Ok(TractRunner { plan, manifest: manifest.clone(), load_ms: started.elapsed().as_secs_f64() * 1000.0 })
    }

    pub fn manifest(&self) -> &DenoiserManifest {
        &self.manifest
    }

    /// Check the model does something sensible before it is ever used on a photo: the right number of finite
    /// numbers, the same tile twice gives the same answer, its scale is the one the manifest says, and noise on a
    /// smooth picture goes down.
    pub fn self_test(&self) -> SelfTest {
        let t = self.manifest.tile as usize;
        let mut checks: Vec<(String, bool)> = Vec::new();
        let (input, base) = test_tile(t);
        let (mut times, mut scale) = (Vec::new(), 0.0f64);
        let mut run = || {
            let started = Instant::now();
            let r = self.run(&input);
            times.push(started.elapsed().as_secs_f64() * 1000.0);
            r
        };
        let (a, b) = (run(), run());
        match (&a, &b) {
            (Ok(a), Ok(b)) => {
                checks.push((format!("gives {} finite numbers per tile", a.len()), true));
                checks.push((
                    "gives the same answer for the same tile".into(),
                    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-4 * x.abs().max(y.abs()).max(1e-9)),
                ));
                let side = 2 * t;
                let mean_in = base.iter().map(|&v| f64::from(v)).sum::<f64>() / base.len().max(1) as f64;
                let mean_out = a.iter().map(|&v| f64::from(v)).sum::<f64>() / a.len().max(1) as f64;
                scale = if mean_in > 0.0 { mean_out / mean_in } else { 0.0 };
                if let Gain::MatchMean { nominal, max_deviation } = self.manifest.gain {
                    let ratio = scale / f64::from(nominal);
                    let tolerance = f64::from(max_deviation).max(0.05) * 4.0;
                    checks
                        .push((format!("its scale is the {nominal:.3e} the description says (found {scale:.3e})"), (ratio - 1.0).abs() <= tolerance));
                }
                // noise left after bringing the answer back to the input's scale, against the noise that went in
                let g = if scale > 0.0 { 1.0 / scale } else { 1.0 };
                let (mut noise_in, mut noise_out, mut n_in, mut n_out) = (0.0f64, 0.0f64, 0usize, 0usize);
                for (i, (&v, &s)) in input.iter().zip(base.iter().cycle()).enumerate() {
                    // centre of each plane only: edges are the model's guess
                    let (x, y) = ((i % (t * t)) % t, (i % (t * t)) / t);
                    if x >= t / 4 && x < 3 * t / 4 && y >= t / 4 && y < 3 * t / 4 {
                        noise_in += f64::from(v - s).powi(2);
                        n_in += 1;
                    }
                }
                for c in 0..3 {
                    for y in side / 4..3 * side / 4 {
                        for x in side / 4..3 * side / 4 {
                            if let (Some(&o), Some(&s)) = (a.get(c * side * side + y * side + x), base.get((y / 2) * t + x / 2)) {
                                noise_out += (f64::from(o) * g - f64::from(s)).powi(2);
                                n_out += 1;
                            }
                        }
                    }
                }
                let (rms_in, rms_out) = ((noise_in / n_in.max(1) as f64).sqrt(), (noise_out / n_out.max(1) as f64).sqrt());
                checks.push((format!("takes noise down (from {rms_in:.4} to {rms_out:.4})"), rms_out < 0.8 * rms_in));
            }
            (Err(e), _) | (_, Err(e)) => checks.push((format!("runs: {e}"), false)),
        }
        times.sort_by(|x, y| x.total_cmp(y));
        SelfTest {
            ok: !checks.is_empty() && checks.iter().all(|(_, ok)| *ok),
            load_ms: self.load_ms,
            tile_ms: times.get(times.len() / 2).copied().unwrap_or(0.0),
            tile_megapixels: (2 * t * 2 * t) as f64 / 1e6,
            scale,
            checks,
        }
    }
}

impl TileRunner for TractRunner {
    fn run(&self, input: &[f32]) -> Result<Vec<f32>, Error> {
        let t = self.manifest.tile as usize;
        if input.len() != 4 * t * t {
            return Err(Error::Input("the tile is not the size the model wants".into()));
        }
        let out = guarded("running it", || {
            let tensor: Tensor = tract_ndarray::Array4::from_shape_vec((1, 4, t, t), input.to_vec()).map_err(|e| Error::Input(e.to_string()))?.into();
            let result = self.plan.run(tvec!(tensor.into())).map_err(|e| Error::Runtime(format!("{e:#}")))?;
            let first = result.first().ok_or_else(|| Error::Model("no output".into()))?;
            let view = first.to_plain_array_view::<f32>().map_err(|e| Error::Runtime(format!("{e:#}")))?;
            Ok(view.iter().copied().collect::<Vec<f32>>())
        })?;
        let want = 3 * (2 * t) * (2 * t);
        if out.len() != want {
            return Err(Error::Model(format!("{} numbers per tile, not the {want} its description says", out.len())));
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(Error::Model("numbers that are not finite".into()));
        }
        Ok(out)
    }
}

/// A deterministic test tile: four planes of a smooth gradient with noise on top, and the gradient alone for one
/// plane (`tile × tile`).
fn test_tile(t: usize) -> (Vec<f32>, Vec<f32>) {
    let mut seed = 0x2545_f491u32;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed >> 8) as f32 / (1u32 << 24) as f32
    };
    let base: Vec<f32> = (0..t * t).map(|i| 0.12 + 0.3 * (((i % t) + (i / t)) as f32 / (2 * t) as f32)).collect();
    let input = (0..4 * t * t)
        .map(|i| {
            // sum of four uniforms: close to Gaussian, deviation 0.0577 × 2 before scaling
            let n = next() + next() + next() + next() - 2.0;
            base.get(i % (t * t)).copied().unwrap_or(0.0) + 0.05 * n
        })
        .collect();
    (input, base)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lightcraft_faces::manifest::Licence;
    use lightcraft_faces::synthetic::{len_field, value_info_bytes};

    fn varint(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return;
            }
            out.push(b | 0x80);
        }
    }

    fn varint_field(n: u64, v: u64, out: &mut Vec<u8>) {
        varint(n << 3, out);
        varint(v, out);
    }

    fn node(op: &str, inputs: &[&str], output: &str, ints: &[(&str, i64)]) -> Vec<u8> {
        let mut n = Vec::new();
        for i in inputs {
            len_field(1, i.as_bytes(), &mut n);
        }
        len_field(2, output.as_bytes(), &mut n);
        len_field(4, op.as_bytes(), &mut n);
        for (name, value) in ints {
            let mut a = Vec::new();
            len_field(1, name.as_bytes(), &mut a);
            varint_field(3, *value as u64, &mut a);
            varint_field(20, 2, &mut a);
            len_field(5, &a, &mut n);
        }
        n
    }

    fn tensor(name: &str, dims: &[u64], values: &[f32]) -> Vec<u8> {
        let mut t = Vec::new();
        for d in dims {
            varint_field(1, *d, &mut t);
        }
        varint_field(2, 1, &mut t);
        len_field(8, name.as_bytes(), &mut t);
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        len_field(9, &raw, &mut t);
        t
    }

    /// A tiny working denoiser-shaped network: `[1, 4, t, t]` → a 1 × 1 convolution to 12 channels (each output
    /// channel is `scale` times one input plane, so the scale is known) → depth-to-space → `[1, 3, 2t, 2t]`.
    fn tiny_model(t: u64, scale: f32) -> Vec<u8> {
        tiny_model_with(t, scale, 12)
    }

    /// The same with another number of convolution channels; only 12 can be arranged into the three colours.
    fn tiny_model_with(t: u64, scale: f32, channels: u64) -> Vec<u8> {
        let mut graph = Vec::new();
        len_field(1, &node("Conv", &["data", "w"], if channels == 12 { "c" } else { "rgb" }, &[("group", 1)]), &mut graph);
        if channels == 12 {
            len_field(1, &node("DepthToSpace", &["c"], "rgb", &[("blocksize", 2)]), &mut graph);
        }
        // output channel (dy·2 + dx)·3 + colour reads input plane [R, G1, B] of the tile
        let mut w = vec![0f32; channels as usize * 4];
        for oc in 0..channels as usize {
            let colour = oc % 3;
            let plane = [0, 1, 3][colour];
            w[oc * 4 + plane] = scale;
        }
        len_field(5, &tensor("w", &[channels, 4, 1, 1], &w), &mut graph);
        len_field(11, &value_info_bytes("data", 1, &[Ok(1), Ok(4), Ok(t), Ok(t)]), &mut graph);
        len_field(12, &value_info_bytes("rgb", 1, &[Ok(1), Ok(3), Ok(2 * t), Ok(2 * t)]), &mut graph);
        let mut ops = Vec::new();
        len_field(1, b"", &mut ops);
        varint_field(2, 13, &mut ops);
        let mut model = Vec::new();
        varint_field(1, 8, &mut model);
        len_field(7, &graph, &mut model);
        len_field(8, &ops, &mut model);
        model
    }

    fn manifest(tile: u32, nominal: f32) -> DenoiserManifest {
        DenoiserManifest {
            id: "tiny".into(),
            name: "Tiny".into(),
            version: "1".into(),
            licence: Licence::default(),
            source: None,
            sha256: None,
            size_bytes: None,
            provenance: String::new(),
            domain: Domain::BayerToRgb,
            tile,
            overlap: 16,
            gain: Gain::MatchMean { nominal, max_deviation: 0.05 },
        }
    }

    fn temp_model(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("lc-denoise-runtime-{name}-{}.onnx", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn a_working_model_loads_runs_and_has_the_scale_it_says() {
        let p = temp_model("ok", &tiny_model(64, 1000.0));
        let r = TractRunner::load(&p, &manifest(64, 1000.0)).unwrap();
        let input: Vec<f32> = (0..4 * 64 * 64).map(|i| (i % 97) as f32 / 97.0).collect();
        let out = r.run(&input).unwrap();
        assert_eq!(out.len(), 3 * 128 * 128);
        // output (colour c, pixel (2y+dy, 2x+dx)) = scale × input plane [R, G1, B][c] at (y, x)
        for (c, plane) in [0usize, 1, 3].into_iter().enumerate() {
            let (y, x) = (5usize, 9usize);
            assert!((out[c * 128 * 128 + (2 * y + 1) * 128 + 2 * x] - 1000.0 * input[plane * 64 * 64 + y * 64 + x]).abs() < 1e-2);
        }
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn the_self_test_checks_scale_and_noise() {
        // a model that only multiplies cannot take noise down: it must fail that check but pass the others
        let p = temp_model("selftest", &tiny_model(64, 1000.0));
        let t = TractRunner::load(&p, &manifest(64, 1000.0)).unwrap().self_test();
        assert!(!t.ok, "{t:?}");
        assert!(t.checks.iter().any(|(c, ok)| c.contains("same answer") && *ok), "{t:?}");
        assert!(t.checks.iter().any(|(c, ok)| c.contains("scale") && *ok), "{t:?}");
        assert!(t.checks.iter().any(|(c, ok)| c.contains("noise") && !*ok), "{t:?}");
        // and one that claims the wrong scale fails that check
        let q = TractRunner::load(&p, &manifest(64, 5.0)).unwrap().self_test();
        assert!(q.checks.iter().any(|(c, ok)| c.contains("scale") && !*ok), "{q:?}");
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn wrong_sizes_and_broken_files_are_errors() {
        let p = temp_model("sizes", &tiny_model(64, 1.0));
        let r = TractRunner::load(&p, &manifest(64, 1.0)).unwrap();
        assert!(matches!(r.run(&[0.0; 10]), Err(Error::Input(_))));
        // a fully convolutional model works at whatever tile size the manifest asks for
        let wide = TractRunner::load(&p, &manifest(128, 1.0)).unwrap();
        assert_eq!(wide.run(&vec![0.5; 4 * 128 * 128]).unwrap().len(), 3 * 256 * 256);
        // a model that gives another number of values than the manifest promises is an error
        let q = temp_model("channels", &tiny_model_with(64, 1.0, 8));
        let odd = TractRunner::load(&q, &manifest(64, 1.0)).unwrap();
        assert!(matches!(odd.run(&vec![0.5; 4 * 64 * 64]), Err(Error::Model(_))));
        assert!(!odd.self_test().ok);
        let _ = std::fs::remove_file(q);
        for (name, bytes) in [("junk", b"nonsense".to_vec()), ("empty", Vec::new())] {
            let q = temp_model(name, &bytes);
            assert!(TractRunner::load(&q, &manifest(64, 1.0)).is_err(), "{name}");
            let _ = std::fs::remove_file(q);
        }
        assert!(TractRunner::load(&std::env::temp_dir().join("does-not-exist.onnx"), &manifest(64, 1.0)).is_err());
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn truncated_and_corrupted_models_never_panic() {
        let good = tiny_model(64, 1.0);
        for n in (0..good.len()).step_by(5) {
            let q = temp_model("trunc", &good[..n]);
            let _ = TractRunner::load(&q, &manifest(64, 1.0));
            let _ = std::fs::remove_file(q);
        }
        for at in (0..good.len()).step_by(7) {
            let mut bad = good.clone();
            if let Some(b) = bad.get_mut(at) {
                *b = b.wrapping_add(0x55);
            }
            let q = temp_model("flip", &bad);
            if let Ok(r) = TractRunner::load(&q, &manifest(64, 1.0)) {
                let _ = r.run(&vec![0.5; 4 * 64 * 64]);
            }
            let _ = std::fs::remove_file(q);
        }
    }

    /// Opt-in: `LC_DENOISE_MODEL=<model_bayer.onnx> cargo test -p lightcraft-denoise --features tract --release
    /// -- --ignored --nocapture` self-tests a real model (RawNIND's Bayer model: tile 512, scale about 1e6).
    #[test]
    #[ignore = "needs a real model file: set LC_DENOISE_MODEL"]
    fn a_real_model_passes_the_self_test() {
        let Some(path) = std::env::var_os("LC_DENOISE_MODEL") else { return };
        let m = manifest(512, 1.0e6);
        let r = TractRunner::load(Path::new(&path), &m).unwrap();
        let t = r.self_test();
        println!("load {:.0} ms | {:.0} ms per tile | scale {:.4e} | {:?}", t.load_ms, t.tile_ms, t.scale, t.checks);
        assert!(t.ok);
    }
}
