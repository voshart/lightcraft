//! AI denoise, the session's side (`docs/denoise.md`): which photos have a denoised picture, making the missing ones
//! in the background, and keeping the cache within its size.
//!
//! The denoised picture of a raw photo is *cached data*, never a file in the library: it is made from the untouched
//! original by the model the user chose, kept in `<library>/denoise/<key>.lcdn`, and thrown away whenever space is
//! needed (it is made again when it is next wanted). Its name is a hash of the model, the algorithm and the photo's
//! content, so a changed file or another model simply does not find it. The Denoise *amount* is an ordinary develop
//! setting (`enhance.denoise`): a render mixes the plain and the denoised picture by it ([`crate::media`]).
//!
//! What runs where: the session thread only decides (which photos, which model) and keeps an index of what is ready
//! ([`crate::media::DenoiseIndex`], memory only). One background thread at a time decodes a photo, runs the model on
//! its mosaic tile by tile and writes the product ([`make_product`]); an export that finds its photo's product missing
//! makes it itself, on its own thread, so it never comes out without the denoise it was asked for.
//!
//! Failure is a message, never a panic: a file that is not a Bayer raw, a model that does not load, a full disk, a
//! panic inside the runtime and a cancelled job all end as an [`Outcome`] the pump records.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};
use std::time::{Duration, SystemTime};

use lightcraft_catalog::{Photo, PhotoId, Source};
use lightcraft_denoise::bayer::Layout;
use lightcraft_denoise::manifest::DenoiserManifest;
use lightcraft_denoise::run::{Control, Params, TileRunner, denoise_bayer};
use lightcraft_denoise::{Error as RunError, product};
use lightcraft_develop::DevelopSettings;
use lightcraft_preview::Hasher128;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::face_download::Downloads;
use crate::media::{DenoiseSpec, MakeProduct, PairLoader};
use crate::merge::ByteReader;
use crate::{Selection, Session};

/// Extension of a product file.
pub const EXT: &str = "lcdn";
/// Where the products of a library are kept, inside the library's folder (safe to delete, like `thumbs/`).
pub const DIR: &str = "denoise";
/// What the cache may take up unless the user says otherwise (GB). A product is about as large as the raw file's
/// decoded samples, compressed: ~100 MB for 24 megapixels.
pub const DEFAULT_CACHE_GB: u32 = 20;
/// Samples at or above this (white = 1) are clipped: the model leaves them as the camera recorded them.
const CLIP: f32 = 0.99;
/// Most photos waiting in the queue.
const MAX_QUEUE: usize = 5000;
/// Most selected photos the automatic queue looks at in one pump.
const AUTO_LOOK: usize = 64;
/// How long a look at the model settings is trusted.
const SETTINGS_TTL: Duration = Duration::from_secs(2);
/// Stack of the threads that run the model (the runtime recurses through its graph).
const STACK: usize = 16 << 20;

// ---------------------------------------------------------------------------------------------------------------------
// the model

/// A loaded denoise model: what runs tiles, and a self-check.
pub(crate) trait Model: Send + Sync {
    fn runner(&self) -> &dyn TileRunner;
    /// Check it does something sensible: the test's result, or why not.
    fn self_test(&self) -> Result<Value, String>;
}

/// Loads the model at a path as its manifest describes it.
pub(crate) type Loader = Arc<dyn Fn(&Path, &DenoiserManifest) -> Result<Arc<dyn Model>, String> + Send + Sync>;

#[cfg(feature = "denoise")]
struct Tract(lightcraft_denoise::runtime::TractRunner);

#[cfg(feature = "denoise")]
impl Model for Tract {
    fn runner(&self) -> &dyn TileRunner {
        &self.0
    }

    fn self_test(&self) -> Result<Value, String> {
        let t = self.0.self_test();
        if !t.ok {
            let failed: Vec<&str> = t.checks.iter().filter(|(_, ok)| !*ok).map(|(c, _)| c.as_str()).collect();
            return Err(format!("the model failed its self-test: it does not {}", failed.join(", and does not ")));
        }
        serde_json::to_value(&t).map_err(|e| e.to_string())
    }
}

/// The loader of this build: tract with the `denoise` feature, else one that says why nothing can run.
pub(crate) fn default_loader() -> Loader {
    #[cfg(feature = "denoise")]
    {
        Arc::new(|path, manifest| {
            lightcraft_denoise::runtime::TractRunner::load(path, manifest).map(|r| Arc::new(Tract(r)) as Arc<dyn Model>).map_err(|e| e.to_string())
        })
    }
    #[cfg(not(feature = "denoise"))]
    {
        Arc::new(|_, _| Err("this build cannot run denoise models".to_string()))
    }
}

/// A model that is loaded on first use and kept, shared by the background job and the exports.
pub(crate) type ModelSlot = Arc<Mutex<Option<Arc<dyn Model>>>>;

// ---------------------------------------------------------------------------------------------------------------------
// settings and installed models

/// `settings.json` in the models folder.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub(crate) struct Settings {
    /// The installed model in use; `None`: denoise is off.
    pub model: Option<String>,
    /// Photos with a Denoise amount above 0 that are being looked at (active, selected) get their picture made without
    /// being asked (`None`: yes).
    pub auto: Option<bool>,
    /// Cache size limit (GB; `None`: [`DEFAULT_CACHE_GB`]).
    pub cache_gb: Option<u32>,
    /// Most tiles run at once (`None`: by how hard the pump says to work).
    pub threads: Option<u32>,
}

impl Settings {
    pub fn auto(&self) -> bool {
        self.auto.unwrap_or(true)
    }

    pub fn cache_bytes(&self) -> u64 {
        u64::from(self.cache_gb.unwrap_or(DEFAULT_CACHE_GB).clamp(1, 10_000)) << 30
    }
}

pub(crate) fn read_capped(path: &Path, max: usize) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut buf = Vec::new();
    std::fs::File::open(path).ok()?.take(max as u64 + 1).read_to_end(&mut buf).ok()?;
    (buf.len() <= max).then_some(buf)
}

pub(crate) fn read_settings(dir: &Path) -> Settings {
    read_capped(&dir.join("settings.json"), 4096).and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

/// Write a file next to its final name and rename it into place.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let part = path.with_extension("part");
    std::fs::write(&part, bytes).map_err(|e| format!("could not write {}: {e}", path.display()))?;
    std::fs::rename(&part, path).map_err(|e| {
        let _ = std::fs::remove_file(&part);
        format!("could not write {}: {e}", path.display())
    })
}

pub(crate) fn write_settings(dir: &Path, st: &Settings) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("could not create the denoise models folder: {e}"))?;
    write_atomic(&dir.join("settings.json"), &serde_json::to_vec_pretty(st).map_err(|e| e.to_string())?)
}

/// An installed model folder: `<id>/model.onnx`, `denoise-model.json`, `installed.json`.
pub(crate) struct Installed {
    pub manifest: DenoiserManifest,
    pub onnx: PathBuf,
    /// What was recorded when it was installed (terms accepted, hash of the file, self-test).
    pub accepted: Value,
}

impl Installed {
    /// Names the model file the cache is made with: a hash of the file as installed, so another file with the same id
    /// never finds this one's products.
    pub fn fingerprint(&self) -> String {
        let sha = self.accepted.get("sha256").and_then(Value::as_str).or(self.manifest.sha256.as_deref()).unwrap_or("");
        let len = std::fs::metadata(&self.onnx).map(|m| m.len()).unwrap_or(0);
        format!("{}@{}:{len}", self.manifest.id, sha)
    }
}

/// The models in the folder. A folder that is not a valid model (wrong name, no model file, a manifest that does not
/// validate or does not match its folder) is ignored, never trusted.
pub(crate) fn installed_models(dir: &Path) -> Vec<Installed> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut out = Vec::new();
    for e in entries.flatten() {
        let p = e.path();
        let Some(name) = p.file_name().and_then(|n| n.to_str()).map(str::to_string) else { continue };
        let onnx = p.join("model.onnx");
        if !lightcraft_faces::manifest::valid_id(&name) || !p.is_dir() || !onnx.is_file() {
            continue;
        }
        let Some(m) = read_capped(&p.join("denoise-model.json"), lightcraft_denoise::manifest::MAX_MANIFEST_BYTES)
            .and_then(|b| lightcraft_denoise::manifest::parse(&b).ok())
        else {
            continue;
        };
        if m.id != name {
            continue;
        }
        let accepted = read_capped(&p.join("installed.json"), 8192).and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
        out.push(Installed { manifest: m, onnx, accepted });
    }
    out.sort_by(|a, b| a.manifest.id.cmp(&b.manifest.id));
    out
}

/// The model in use.
pub(crate) struct Active {
    pub id: String,
    pub manifest: DenoiserManifest,
    pub onnx: PathBuf,
    pub fingerprint: String,
    pub model: ModelSlot,
}

// ---------------------------------------------------------------------------------------------------------------------
// products

/// The photos whose files can be denoised at all: raw files developed from their sensor data and read from disk.
pub(crate) fn eligible(p: &Photo) -> bool {
    p.develops_raw() && matches!(p.source, Source::File { .. }) && !p.deleted
}

/// A product's name: what it was made from and with.
pub(crate) fn product_key(fingerprint: &str, p: &Photo) -> String {
    let mut h = Hasher128::new();
    h.str("lcdn").u64(u64::from(product::ALGORITHM)).str(fingerprint).str(&crate::media::content_key(p));
    h.finish().to_string()
}

pub(crate) fn product_path(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("{key}.{EXT}"))
}

/// The `.lcdn` files in `dir`: path, size, last change.
fn products_in(dir: &Path) -> Vec<(PathBuf, u64, SystemTime)> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    entries
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == EXT))
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            m.is_file().then(|| (e.path(), m.len(), m.modified().unwrap_or(SystemTime::UNIX_EPOCH)))
        })
        .collect()
}

/// Make sure only one thread writes a product at a time: a second one waits, then finds it made.
fn making() -> &'static (Mutex<HashSet<PathBuf>>, Condvar) {
    static MAKING: OnceLock<(Mutex<HashSet<PathBuf>>, Condvar)> = OnceLock::new();
    MAKING.get_or_init(|| (Mutex::new(HashSet::new()), Condvar::new()))
}

struct Slot(PathBuf);

impl Slot {
    fn enter(path: &Path) -> Slot {
        let (set, cv) = making();
        let mut g = set.lock().unwrap_or_else(PoisonError::into_inner);
        while g.contains(path) {
            g = cv.wait(g).unwrap_or_else(PoisonError::into_inner);
        }
        g.insert(path.to_path_buf());
        Slot(path.to_path_buf())
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        let (set, cv) = making();
        set.lock().unwrap_or_else(PoisonError::into_inner).remove(&self.0);
        cv.notify_all();
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// making a product

/// Everything a thread needs to make one product, detached from the session.
#[derive(Clone)]
pub(crate) struct JobSpec {
    /// The original file.
    pub path: String,
    pub read: Option<ByteReader>,
    pub product: PathBuf,
    pub key: String,
    pub manifest: DenoiserManifest,
    pub onnx: PathBuf,
    pub model: ModelSlot,
    pub loader: Loader,
    /// Tiles run at once.
    pub parallel: usize,
}

/// Why a product was not made.
#[derive(Debug, PartialEq)]
pub(crate) enum MakeError {
    /// Denoise does not apply to this file (not a Bayer raw): nothing is wrong.
    Unsupported(String),
    Failed(String),
    Cancelled,
}

impl From<RunError> for MakeError {
    fn from(e: RunError) -> Self {
        match e {
            RunError::Cancelled => MakeError::Cancelled,
            other => MakeError::Failed(other.to_string()),
        }
    }
}

/// Progress and cancel of a running job.
#[derive(Default)]
pub(crate) struct Progress {
    done: AtomicUsize,
    total: AtomicUsize,
    cancel: AtomicBool,
}

impl Progress {
    /// Ask the job to stop at the next tile.
    pub(crate) fn cancel_now(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// The loaded model, loading it first when it is not (kept for the next photo).
fn loaded(spec: &JobSpec) -> Result<Arc<dyn Model>, MakeError> {
    if let Some(m) = spec.model.lock().unwrap_or_else(PoisonError::into_inner).clone() {
        return Ok(m);
    }
    let m = (spec.loader)(&spec.onnx, &spec.manifest).map_err(|e| MakeError::Failed(format!("the denoise model could not be loaded: {e}")))?;
    *spec.model.lock().unwrap_or_else(PoisonError::into_inner) = Some(m.clone());
    Ok(m)
}

/// The cell `[(0,0), (1,0), (0,1), (1,1)]` colours of a Bayer mosaic and so its layout.
fn layout_of(cfa: &lightcraft_raw::Cfa) -> Option<Layout> {
    Layout::from_cell([cfa.color_at(0, 0), cfa.color_at(1, 0), cfa.color_at(0, 1), cfa.color_at(1, 1)])
}

/// Make the product of `spec` unless it is there: decode the photo, run the model over its mosaic, write the result.
/// `Ok(true)` when it was made now, `Ok(false)` when it already existed.
pub(crate) fn make_product(spec: &JobSpec, progress: Option<&Progress>) -> Result<bool, MakeError> {
    let _slot = Slot::enter(&spec.product);
    if product::is_current(&spec.product, &spec.key) {
        return Ok(false);
    }
    let cancelled = || progress.is_some_and(|p| p.cancel.load(Ordering::Relaxed));
    if cancelled() {
        return Err(MakeError::Cancelled);
    }
    let bytes = match &spec.read {
        Some(r) => r(&spec.path).map_err(MakeError::Failed)?,
        None => std::fs::read(&spec.path).map_err(|e| MakeError::Failed(format!("{}: {e}", spec.path)))?,
    };
    // the decode and the normalised mosaic wait their turn at the process-wide memory gate, as every decode does
    let (mosaic, layout) = {
        let _held = crate::memory::work_gate().acquire(bytes.len().saturating_mul(4));
        if lightcraft_raw::probe(&bytes).is_none() {
            return Err(MakeError::Unsupported("not a raw file".into()));
        }
        let raw = match lightcraft_raw::decode(&bytes) {
            Ok(r) => r,
            Err(lightcraft_raw::RawError::Unsupported(why)) => return Err(MakeError::Unsupported(why)),
            Err(e) => return Err(MakeError::Failed(e.to_string())),
        };
        drop(bytes);
        if raw.cpp != 1 || !raw.cfa.as_ref().is_some_and(lightcraft_raw::Cfa::is_bayer) {
            return Err(MakeError::Unsupported(
                "only Bayer raw files can be denoised (this one has another sensor layout or is already demosaiced)".into(),
            ));
        }
        let n = raw.normalized().map_err(|e| MakeError::Failed(e.to_string()))?;
        drop(raw);
        let layout =
            n.cfa.as_ref().and_then(layout_of).ok_or_else(|| MakeError::Unsupported("the colour filter layout is not a Bayer pattern".into()))?;
        (n, layout)
    };
    if cancelled() {
        return Err(MakeError::Cancelled);
    }
    let model = loaded(spec)?;
    let params = Params {
        tile: spec.manifest.tile as usize,
        overlap: spec.manifest.overlap as usize,
        gain: spec.manifest.gain,
        clip: Some(CLIP),
        parallel: spec.parallel.max(1),
    };
    let report = |done: usize, total: usize| {
        if let Some(p) = progress {
            p.total.store(total, Ordering::Relaxed);
            p.done.store(done, Ordering::Relaxed);
        }
    };
    let ctl = Control { cancel: progress.map(|p| &p.cancel), progress: Some(&report) };
    // the tiles run on threads of their own, so a busy window's renders keep theirs
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(params.parallel)
        .stack_size(STACK)
        .thread_name(|i| format!("denoise-{i}"))
        .build()
        .map_err(|e| MakeError::Failed(format!("could not start the worker threads: {e}")))?;
    let rgb = pool.install(|| denoise_bayer(&mosaic.data, mosaic.width, mosaic.height, layout, model.runner(), &params, &ctl))?;
    drop(mosaic);
    if cancelled() {
        return Err(MakeError::Cancelled);
    }
    product::write(&spec.product, &rgb, &spec.key).map_err(|e| MakeError::Failed(format!("could not save the denoised picture: {e}")))?;
    Ok(true)
}

// ---------------------------------------------------------------------------------------------------------------------
// the session's state

/// How hard the background work may go (the app picks it from what the user is doing, like the face scan).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pace {
    /// Start nothing new.
    Pause,
    /// Two tiles at once.
    Light,
    /// Half the machine.
    Normal,
    /// Most of the machine: the user is watching the progress.
    Full,
}

impl Pace {
    /// `pause`, `light`, `normal` or `full`; anything else is `normal`.
    pub fn parse(name: Option<&str>) -> Pace {
        match name {
            Some("pause") => Pace::Pause,
            Some("light") => Pace::Light,
            Some("full") => Pace::Full,
            _ => Pace::Normal,
        }
    }

    /// Tiles to run at once on a machine with `cores` threads (`cap`: the user's own limit).
    pub fn parallel(self, cores: usize, cap: Option<u32>) -> usize {
        let n = match self {
            Pace::Pause | Pace::Light => 2,
            Pace::Normal => cores / 2,
            Pace::Full => cores * 4 / 5,
        }
        .clamp(1, 16);
        cap.map_or(n, |c| n.min(c.max(1) as usize))
    }
}

/// How a photo stands, for the interface.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "camelCase")]
pub enum PhotoState {
    /// Not a photo denoise applies to (not a raw, offline, a demo).
    NotApplicable,
    /// No model is chosen: nothing can be made.
    NoModel,
    /// Its picture is made: the Amount slider works.
    Ready,
    Queued {
        /// Photos ahead of it (0 = next).
        ahead: usize,
    },
    Running {
        done: usize,
        total: usize,
    },
    /// Making it failed (`why`), or denoise does not apply to this file (`unsupported`).
    Failed {
        why: String,
        unsupported: bool,
    },
    /// Nothing has asked for it yet.
    Idle,
}

#[derive(Clone)]
struct Failure {
    key: String,
    why: String,
    unsupported: bool,
}

enum Outcome {
    Done,
    Failed(String, bool),
    Cancelled,
}

struct Running {
    photo: PhotoId,
    key: String,
    product: PathBuf,
    progress: Arc<Progress>,
    outcome: Arc<Mutex<Option<Outcome>>>,
    started: web_time::Instant,
}

pub(crate) struct State {
    /// Where the models are kept (`None` where there is no file system).
    pub models_dir: Option<PathBuf>,
    /// Models being downloaded at the user's request.
    pub downloads: Downloads,
    /// How a model is loaded (tests put another in).
    pub loader: Loader,
    pub(crate) active: Option<Active>,
    active_seen: Option<web_time::Instant>,
    /// `settings.json` as last read (with the model choice).
    pub(crate) settings: Settings,
    /// The index must be made again (the library, the model or the cache changed).
    dirty: bool,
    /// The catalog revision and time of the last index, to look again when photos changed.
    reindexed: Option<(u64, web_time::Instant)>,
    queue: VecDeque<PhotoId>,
    running: Option<Running>,
    failed: HashMap<PhotoId, Failure>,
    /// Counts every change of what is ready, so the interface knows to draw again.
    pub generation: u64,
    /// Photos made since the session started.
    made: u64,
}

impl Default for State {
    fn default() -> Self {
        State {
            models_dir: None,
            downloads: Downloads::default(),
            loader: default_loader(),
            active: None,
            active_seen: None,
            settings: Settings::default(),
            dirty: true,
            reindexed: None,
            queue: VecDeque::new(),
            running: None,
            failed: HashMap::new(),
            generation: 0,
            made: 0,
        }
    }
}

impl Drop for State {
    /// Quitting stops the job instead of leaving it to finish a photo nobody will see.
    fn drop(&mut self) {
        if let Some(r) = &self.running {
            r.progress.cancel_now();
        }
    }
}

impl State {
    /// The library was replaced: nothing carries over (photo ids are per library).
    pub fn library_changed(&mut self) {
        if let Some(r) = self.running.take() {
            r.progress.cancel_now();
        }
        self.queue.clear();
        self.failed.clear();
        self.dirty = true;
        self.generation += 1;
    }

    /// The model choice or the cache changed on disk: look again at the next pump.
    pub fn touch(&mut self) {
        self.active_seen = None;
        self.dirty = true;
    }

    pub fn queued(&self) -> usize {
        self.queue.len()
    }
}

impl Session {
    /// The folder the products of the open library are kept in; `None` without a library on disk.
    pub(crate) fn denoise_products_dir(&self) -> Option<PathBuf> {
        self.library.as_ref().filter(|l| l.on_disk).map(|l| l.dir.join(DIR))
    }

    /// Read which model is in use (at most every couple of seconds): the manifest and file of the model the settings
    /// name, if it is installed.
    pub(crate) fn denoise_refresh_active(&mut self, force: bool) {
        let fresh = self.denoise.active_seen.is_some_and(|t| t.elapsed() < SETTINGS_TTL);
        if fresh && !force {
            return;
        }
        self.denoise.active_seen = Some(web_time::Instant::now());
        self.denoise.settings = self.denoise.models_dir.as_deref().map(read_settings).unwrap_or_default();
        let chosen = self.denoise.models_dir.as_deref().and_then(|dir| {
            let id = self.denoise.settings.model.clone()?;
            installed_models(dir).into_iter().find(|i| i.manifest.id == id)
        });
        let found = chosen.map(|i| (i.fingerprint(), i));
        let same = match (&self.denoise.active, &found) {
            (None, None) => true,
            (Some(a), Some((fp, _))) => a.fingerprint == *fp,
            _ => false,
        };
        if same {
            return;
        }
        // another model: what was made with the old one is not this one's, and what failed may work now
        self.denoise.active = found.map(|(fingerprint, i)| Active {
            id: i.manifest.id.clone(),
            manifest: i.manifest,
            onnx: i.onnx,
            fingerprint,
            model: Arc::new(Mutex::new(None)),
        });
        if let Some(r) = self.denoise.running.take() {
            r.progress.cancel_now();
        }
        self.denoise.queue.clear();
        self.denoise.failed.clear();
        self.denoise.dirty = true;
    }

    fn denoise_spec(&self, p: &Photo) -> Option<(DenoiseSpec, String)> {
        let (dir, active) = (self.denoise_products_dir()?, self.denoise.active.as_ref()?);
        let key = product_key(&active.fingerprint, p);
        Some((DenoiseSpec { product: product_path(&dir, &key), key: key.clone(), make: None }, key))
    }

    /// Make the index of ready photos from the products on disk (a directory listing and a hash per photo: no file is
    /// opened). Pictures that appeared or vanished drop the decoded sources they affect.
    pub(crate) fn denoise_reindex(&mut self) {
        self.denoise.dirty = false;
        self.denoise.reindexed = Some((self.catalog.revision, web_time::Instant::now()));
        let mut now: HashMap<PhotoId, DenoiseSpec> = HashMap::new();
        if let (Some(dir), Some(active)) = (self.denoise_products_dir(), self.denoise.active.as_ref()) {
            let names: HashSet<String> =
                products_in(&dir).into_iter().filter_map(|(p, _, _)| p.file_stem().and_then(|s| s.to_str()).map(str::to_string)).collect();
            if !names.is_empty() {
                for p in self.catalog.photos().filter(|p| eligible(p)) {
                    let key = product_key(&active.fingerprint, p);
                    if names.contains(&key) {
                        now.insert(p.id, DenoiseSpec { product: product_path(&dir, &key), key, make: None });
                    }
                }
            }
        }
        let mut changed = false;
        for id in self.media.denoise.ids() {
            if !now.contains_key(&id) {
                self.media.denoise.remove(id);
                self.media.forget(id);
                changed = true;
            }
        }
        for (id, spec) in now {
            if self.media.denoise.spec(id).is_none_or(|old| old.key != spec.key) {
                self.media.denoise.set(id, spec);
                self.media.forget(id);
                changed = true;
            }
        }
        if changed {
            self.denoise.generation += 1;
        }
    }

    /// A photo whose file changed (its content key is not the one its index entry was made for) has no picture.
    fn denoise_check_photo(&mut self, id: PhotoId) {
        let Some(p) = self.catalog.photo(id).cloned() else { return };
        let Some((spec, _)) = self.denoise_spec(&p) else { return };
        match self.media.denoise.spec(id) {
            Some(have) if have.key != spec.key => {
                self.media.denoise.remove(id);
                self.media.forget(id);
                self.denoise.generation += 1;
            }
            None if spec.product.is_file() && product::is_current(&spec.product, &spec.key) => {
                // made by an export, or by another session
                self.media.denoise.set(id, spec);
                self.media.forget(id);
                self.denoise.generation += 1;
            }
            _ => {}
        }
    }

    /// How photo `id` stands.
    pub fn denoise_photo_state(&self, id: PhotoId) -> PhotoState {
        let Some(p) = self.catalog.photo(id) else { return PhotoState::NotApplicable };
        if !eligible(p) {
            return PhotoState::NotApplicable;
        }
        if self.denoise.active.is_none() {
            return PhotoState::NoModel;
        }
        if self.media.denoise.spec(id).is_some() {
            return PhotoState::Ready;
        }
        if let Some(r) = self.denoise.running.as_ref().filter(|r| r.photo == id) {
            return PhotoState::Running { done: r.progress.done.load(Ordering::Relaxed), total: r.progress.total.load(Ordering::Relaxed) };
        }
        if let Some(ahead) = self.denoise.queue.iter().position(|q| *q == id) {
            return PhotoState::Queued { ahead: ahead + usize::from(self.denoise.running.is_some()) };
        }
        let key = self.denoise_spec(p).map(|s| s.1);
        match self.denoise.failed.get(&id) {
            Some(f) if Some(&f.key) == key.as_ref() => PhotoState::Failed { why: f.why.clone(), unsupported: f.unsupported },
            _ => PhotoState::Idle,
        }
    }

    /// Ask for photos' pictures to be made (the ones that are not made, queued or known to be impossible). With `retry`
    /// a photo that failed before is tried again. A photo that is not raw is skipped. Returns how many were queued.
    pub(crate) fn denoise_enqueue(&mut self, ids: &[PhotoId], front: bool, retry: bool) -> usize {
        self.denoise_refresh_active(false);
        if self.denoise.active.is_none() || self.denoise_products_dir().is_none() {
            return 0;
        }
        let mut added = 0;
        for id in ids {
            let Some(p) = self.catalog.photo(*id).cloned() else { continue };
            if !eligible(&p) || self.media.denoise.spec(*id).is_some() || self.denoise.queue.contains(id) {
                continue;
            }
            if self.denoise.running.as_ref().is_some_and(|r| r.photo == *id) || self.denoise.queue.len() >= MAX_QUEUE {
                continue;
            }
            let Some((_, key)) = self.denoise_spec(&p) else { continue };
            if retry {
                self.denoise.failed.remove(id);
            } else if self.denoise.failed.get(id).is_some_and(|f| f.key == key) {
                continue;
            }
            if front {
                self.denoise.queue.push_front(*id);
            } else {
                self.denoise.queue.push_back(*id);
            }
            added += 1;
        }
        added
    }

    /// Stop the job that is running, and with `all` forget everything that waits.
    pub(crate) fn denoise_cancel(&mut self, photo: Option<PhotoId>, all: bool) -> usize {
        let mut n = 0;
        if let Some(r) = &self.denoise.running
            && photo.is_none_or(|p| p == r.photo)
        {
            r.progress.cancel_now();
            n += 1;
        }
        let before = self.denoise.queue.len();
        match photo {
            Some(p) => self.denoise.queue.retain(|q| *q != p),
            None if all => self.denoise.queue.clear(),
            None => {}
        }
        n + before - self.denoise.queue.len()
    }

    /// The photos the automatic queue looks at: the one open and the ones selected, those with a Denoise amount.
    fn denoise_wanted(&self) -> Vec<PhotoId> {
        let Selection { active, ids, .. } = &self.selection;
        active
            .iter()
            .chain(ids.iter().take(AUTO_LOOK))
            .copied()
            .filter(|id| self.catalog.photo(*id).is_some_and(|p| p.develop.denoise_amount() > 0.0))
            .collect()
    }

    fn denoise_job_spec(&self, p: &Photo, parallel: usize) -> Option<JobSpec> {
        let Source::File { path } = &p.source else { return None };
        let (spec, _) = self.denoise_spec(p)?;
        let active = self.denoise.active.as_ref()?;
        Some(JobSpec {
            path: path.clone(),
            read: self.media.file_bytes.clone(),
            product: spec.product,
            key: spec.key,
            manifest: active.manifest.clone(),
            onnx: active.onnx.clone(),
            model: active.model.clone(),
            loader: self.denoise.loader.clone(),
            parallel,
        })
    }

    /// Start the next queued photo on its own thread.
    fn denoise_start_next(&mut self, parallel: usize) {
        while let Some(id) = self.denoise.queue.pop_front() {
            let Some(p) = self.catalog.photo(id).cloned() else { continue };
            if !eligible(&p) || self.media.denoise.spec(id).is_some() {
                continue;
            }
            let Some(spec) = self.denoise_job_spec(&p, parallel) else { continue };
            // another session or an export may have made it since it was queued
            if product::is_current(&spec.product, &spec.key) {
                self.denoise_check_photo(id);
                continue;
            }
            let progress = Arc::new(Progress::default());
            let outcome: Arc<Mutex<Option<Outcome>>> = Arc::new(Mutex::new(None));
            let (key, product) = (spec.key.clone(), spec.product.clone());
            let (worker_progress, worker_outcome) = (progress.clone(), outcome.clone());
            let started = std::thread::Builder::new().name("denoise".into()).stack_size(STACK).spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| make_product(&spec, Some(&worker_progress))));
                let o = match result {
                    Ok(Ok(_)) => Outcome::Done,
                    Ok(Err(MakeError::Cancelled)) => Outcome::Cancelled,
                    Ok(Err(MakeError::Unsupported(why))) => Outcome::Failed(why, true),
                    Ok(Err(MakeError::Failed(why))) => Outcome::Failed(why, false),
                    Err(_) => Outcome::Failed("the denoiser stopped unexpectedly".into(), false),
                };
                *worker_outcome.lock().unwrap_or_else(PoisonError::into_inner) = Some(o);
            });
            match started {
                Ok(_) => {
                    self.denoise.running = Some(Running { photo: id, key, product, progress, outcome, started: web_time::Instant::now() });
                }
                Err(e) => {
                    self.denoise.failed.insert(id, Failure { key, why: format!("could not start the denoiser: {e}"), unsupported: false });
                    continue;
                }
            }
            return;
        }
    }

    /// Take in the running job's result.
    fn denoise_poll(&mut self) {
        let Some(r) = &self.denoise.running else { return };
        let Some(outcome) = r.outcome.lock().unwrap_or_else(PoisonError::into_inner).take() else { return };
        let Some(r) = self.denoise.running.take() else { return };
        match outcome {
            Outcome::Done => {
                self.denoise.made += 1;
                self.denoise_check_photo(r.photo);
                self.denoise_evict(Some(&r.product));
            }
            Outcome::Failed(why, unsupported) => {
                self.denoise.failed.insert(r.photo, Failure { key: r.key, why, unsupported });
            }
            Outcome::Cancelled => {}
        }
        // the picture's state changed: the interface draws it again
        self.denoise.generation += 1;
    }

    /// Keep the cache within its limit: the pictures nobody asks for go first, the oldest first; `keep` (the one just
    /// made) never goes. The index forgets what was deleted.
    pub(crate) fn denoise_evict(&mut self, keep: Option<&Path>) {
        let Some(models) = self.denoise.models_dir.clone() else { return };
        self.denoise_evict_to(read_settings(&models).cache_bytes(), keep);
    }

    /// [`Self::denoise_evict`] for a limit of `limit` bytes.
    pub(crate) fn denoise_evict_to(&mut self, limit: u64, keep: Option<&Path>) {
        let Some(dir) = self.denoise_products_dir() else { return };
        let mut files = products_in(&dir);
        let mut total: u64 = files.iter().map(|f| f.1).sum();
        if total <= limit {
            return;
        }
        // the pictures photos use now (a Denoise amount above 0) are the last to go
        let wanted: HashSet<PathBuf> = self
            .media
            .denoise
            .ids()
            .into_iter()
            .filter(|id| self.catalog.photo(*id).is_some_and(|p| p.develop.denoise_amount() > 0.0))
            .filter_map(|id| self.media.denoise.spec(id).map(|s| s.product.clone()))
            .collect();
        files.sort_by_key(|(path, _, modified)| (wanted.contains(path), *modified));
        let mut gone: HashSet<PathBuf> = HashSet::new();
        for (path, size, _) in files {
            if total <= limit {
                break;
            }
            if keep == Some(path.as_path()) {
                continue;
            }
            if std::fs::remove_file(&path).is_ok() {
                total = total.saturating_sub(size);
                gone.insert(path);
            }
        }
        if gone.is_empty() {
            return;
        }
        for id in self.media.denoise.ids() {
            if self.media.denoise.spec(id).is_some_and(|s| gone.contains(&s.product)) {
                self.media.denoise.remove(id);
                self.media.forget(id);
            }
        }
        self.denoise.generation += 1;
    }

    /// Delete every product of this library (they are made again when wanted).
    pub(crate) fn denoise_clear(&mut self) -> (usize, u64) {
        self.denoise_cancel(None, true);
        let (mut files, mut bytes) = (0, 0);
        if let Some(dir) = self.denoise_products_dir() {
            for (path, size, _) in products_in(&dir) {
                if std::fs::remove_file(&path).is_ok() {
                    files += 1;
                    bytes += size;
                }
            }
        }
        self.denoise.failed.clear();
        self.denoise.touch();
        (files, bytes)
    }

    /// Per frame: take in what finished, look at what the user is looking at, and keep one job going. Cheap when
    /// nothing is happening (no file is opened unless a photo with a Denoise amount lacks its picture).
    pub(crate) fn denoise_pump(&mut self, pace: Pace) {
        crate::cmd::denoise::finish_downloads(self);
        self.denoise_refresh_active(false);
        // photos that changed (a replaced file has another key) must not keep the picture of what they were
        let stale = !self.media.denoise.is_empty()
            && self.denoise.reindexed.is_none_or(|(rev, at)| rev != self.catalog.revision && at.elapsed() > SETTINGS_TTL);
        if self.denoise.dirty || stale {
            self.denoise_reindex();
        }
        self.denoise_poll();
        if self.denoise.active.is_none() || self.denoise_products_dir().is_none() {
            return;
        }
        let (cap, auto) = (self.denoise.settings.threads, self.denoise.settings.auto());
        let wanted = self.denoise_wanted();
        for id in &wanted {
            self.denoise_check_photo(*id);
        }
        if auto {
            // the photo being edited goes first
            let (open, others): (Vec<PhotoId>, Vec<PhotoId>) = wanted.into_iter().partition(|id| self.selection.active == Some(*id));
            self.denoise_enqueue(&others, false, false);
            self.denoise_enqueue(&open, true, false);
        }
        if self.denoise.running.is_none() && pace != Pace::Pause {
            let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
            self.denoise_start_next(pace.parallel(cores, cap));
        }
    }

    /// The statistics `denoise.status` reports besides the queue (reads the cache folder).
    pub(crate) fn denoise_cache_usage(&self) -> (usize, u64) {
        match self.denoise_products_dir() {
            Some(dir) => {
                let files = products_in(&dir);
                (files.len(), files.iter().map(|f| f.1).sum())
            }
            None => (0, 0),
        }
    }

    /// Seconds the running job has been going, and its progress.
    pub(crate) fn denoise_running_json(&self) -> Value {
        match &self.denoise.running {
            Some(r) => json!({
                "photo": r.photo.0,
                "done": r.progress.done.load(Ordering::Relaxed),
                "total": r.progress.total.load(Ordering::Relaxed),
                "seconds": r.started.elapsed().as_secs_f64(),
            }),
            None => Value::Null,
        }
    }

    pub(crate) fn denoise_made(&self) -> u64 {
        self.denoise.made
    }

    /// What an export of photo `id` with `settings` needs for its Denoise amount: its denoised picture, made now when the
    /// queue has not got to it. `Ok(None)` when denoise does not apply (amount 0, not a Bayer raw), `Err` when it should
    /// have and cannot (no model chosen, nowhere to keep the picture, the model failing).
    pub(crate) fn denoise_for_export(&mut self, id: PhotoId, settings: &DevelopSettings) -> Result<Option<(DenoiseSpec, PairLoader)>, String> {
        if settings.denoise_amount() <= 0.0 {
            return Ok(None);
        }
        let Some(p) = self.catalog.photo(id).cloned() else { return Ok(None) };
        let Some(loader) = self.media.denoise.loader.clone() else { return Ok(None) };
        if !eligible(&p) {
            return Ok(None);
        }
        self.denoise_refresh_active(false);
        let name = &p.file_name;
        if self.denoise.active.is_none() {
            return Err(format!("{name}: Denoise is on for this photo but no denoise model is chosen; choose one in Settings, or set Denoise to 0"));
        }
        if self.denoise_products_dir().is_none() {
            return Err(format!("{name}: Denoise needs a library on disk to keep its pictures in"));
        }
        self.denoise_check_photo(id);
        let Some(job) = self.denoise_job_spec(&p, Pace::Full.parallel(std::thread::available_parallelism().map_or(1, |n| n.get()), None)) else {
            return Ok(None);
        };
        let label = name.clone();
        let spec = job.clone();
        let make: MakeProduct = Arc::new(move || match make_product(&spec, None) {
            Ok(_) => Ok(true),
            Err(MakeError::Unsupported(_)) => Ok(false),
            Err(MakeError::Failed(why)) => Err(format!("{label}: could not denoise: {why}")),
            Err(MakeError::Cancelled) => Err(format!("{label}: denoise was cancelled")),
        });
        Ok(Some((DenoiseSpec { product: job.product, key: job.key, make: Some(make) }, loader)))
    }
}
