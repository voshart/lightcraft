//! AI denoise in the interface: the line under the Detail panel's Denoise slider (what the picture is doing), the
//! Settings ▸ Denoise tab (the model and the cache) and the per-frame pump that keeps the engine's background work going.
//!
//! The Amount slider is an ordinary develop control (`enhance.denoise`); everything here is about the cleaned picture the
//! slider mixes in. It is cached data made in the background for the photos being looked at, and never a file in the
//! library. A model is fetched only when the user presses Download and accepts its terms (the same dialog as a face model).

use egui::RichText;
use lightcraft_catalog::PhotoId;
use lightcraft_engine::denoise::PhotoState;
use serde_json::{Value, json};

use super::faces::{licence_line, mb, open_page, sentence, window_pace};
use super::settings::{check, heading, hint};
use crate::LightcraftApp;
use crate::theme::Tokens;
use crate::widgets::register;

/// How long a read of the model list is reused (it is a few small files, but not for every frame).
const REFRESH_SECS: f64 = 1.5;
/// Cache limits offered (GB).
const CACHE_LIMITS: [u32; 6] = [5, 10, 20, 50, 100, 250];

/// What the interface keeps between frames.
#[derive(Default)]
pub struct Ui {
    next_pump: f64,
    generation: u64,
    active: bool,
    /// (tiles done, tiles in all) of the photo being made.
    running: Option<(u64, u64)>,
    queued: u64,
    /// Bumped when a model is installed, removed or chosen, so Settings re-reads the list at once.
    pub epoch: u64,
    /// Models the user pressed Download for that have not been installed yet.
    dl_watch: Vec<String>,
    list: Option<(u64, f64, Value)>,
}

/// Called every frame: keeps the engine's denoise work going and asks for a redraw when a photo's picture appears (the
/// loupe and the thumbnails then render from it) or while one is being made (the progress).
pub fn pump(app: &mut LightcraftApp, ctx: &egui::Context) {
    watch_downloads(app, ctx);
    let now = ctx.input(|i| i.time);
    if now < app.caches.denoise.next_pump {
        return;
    }
    let watching = matches!(&app.ui.dialog, Some(crate::state::Dialog::Settings { tab }) if tab == "denoise");
    let pace = window_pace(app, ctx, now, watching);
    let Ok(v) = app.session.execute("denoise.pump", &json!({"pace": pace})) else { return };
    let c = &mut app.caches.denoise;
    let generation = v["generation"].as_u64().unwrap_or(0);
    if generation != c.generation {
        c.generation = generation;
        ctx.request_repaint();
    }
    c.active = v["active"] == true;
    c.queued = v["queued"].as_u64().unwrap_or(0);
    c.running = v["running"].is_object().then(|| (v["running"]["done"].as_u64().unwrap_or(0), v["running"]["total"].as_u64().unwrap_or(0)));
    let busy = c.running.is_some() || c.queued > 0;
    c.next_pump = now
        + if busy {
            0.1
        } else if c.active {
            0.5
        } else {
            1.0
        };
    // frames are drawn only when something asks for one: while a photo is being made, ask again for its progress
    if busy {
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }
}

/// The engine's model list, re-read at most every [`REFRESH_SECS`], or at once after an action changed it.
fn models(app: &mut LightcraftApp, ctx: &egui::Context) -> Value {
    let now = ctx.input(|i| i.time);
    let epoch = app.caches.denoise.epoch;
    if let Some((e, at, v)) = &app.caches.denoise.list
        && *e == epoch
        && now - at < REFRESH_SECS
    {
        return v.clone();
    }
    let v = app.run("denoise.models.list", json!({})).unwrap_or(Value::Null);
    app.caches.denoise.list = Some((epoch, now, v.clone()));
    v
}

/// Follows the downloads the user started: keeps redrawing while one runs (the progress bar), and says so when a
/// model has been installed and is in use (the engine does all of that by itself).
fn watch_downloads(app: &mut LightcraftApp, ctx: &egui::Context) {
    if app.caches.denoise.dl_watch.is_empty() {
        return;
    }
    let Ok(v) = app.session.execute("denoise.models.downloads", &json!({})) else {
        app.caches.denoise.dl_watch.clear();
        return;
    };
    let rows: Vec<Value> = v["downloads"].as_array().cloned().unwrap_or_default();
    let (mut keep, mut installed) = (Vec::new(), Vec::new());
    for id in std::mem::take(&mut app.caches.denoise.dl_watch) {
        match rows.iter().find(|r| r["id"] == id.as_str()).and_then(|r| r["state"].as_str()) {
            Some("running" | "done") => keep.push(id),
            Some("installed") => installed.push(id),
            _ => {}
        }
    }
    app.caches.denoise.dl_watch = keep;
    for id in installed {
        let _ = app.session.execute("denoise.models.downloadCancel", &json!({"id": id}));
        app.caches.denoise.epoch += 1;
        let text = "The denoise model is installed and in use: the Denoise slider under Detail now works";
        app.ui.status = text.into();
        app.toast(ctx, text);
    }
    ctx.request_repaint_after(std::time::Duration::from_millis(120));
}

/// The line under the Denoise slider for the photo `id`, whose Amount is `amount`.
pub fn detail_status(app: &mut LightcraftApp, ui: &mut egui::Ui, id: PhotoId, amount: f64) {
    let t = Tokens::get(ui.ctx());
    let state = app.session.denoise_photo_state(id);
    let pad = egui::Margin { left: 24, right: 22, top: 0, bottom: 6 };
    let note = |ui: &mut egui::Ui, text: &str, color: egui::Color32| {
        ui.add(egui::Label::new(RichText::new(crate::i18n::tr(text)).size(11.0).color(color)).wrap());
    };
    match state {
        PhotoState::NotApplicable | PhotoState::Ready => {}
        PhotoState::NoModel => {
            egui::Frame::NONE.inner_margin(pad).show(ui, |ui| {
                note(ui, "AI Denoise needs a model: a one-time download, in Settings.", t.text_dim);
                ui.add_space(2.0);
                let r = ui.small_button(crate::i18n::tr("Set up AI Denoise…"));
                register(ui.ctx(), "denoise:setup", r.rect);
                if r.clicked() {
                    let _ = app.run("app.settings", json!({"tab": "denoise"}));
                }
            });
        }
        PhotoState::Idle => {
            if amount > 0.0 && app.session.denoise_auto() {
                // the pump has not met this photo yet: let it come round soon instead of waiting out its pause
                let now = ui.ctx().input(|i| i.time);
                app.caches.denoise.next_pump = app.caches.denoise.next_pump.min(now + 0.05);
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(60));
                egui::Frame::NONE.inner_margin(pad).show(ui, |ui| note(ui, "Waiting to start…", t.text_dim));
            } else if amount > 0.0 {
                egui::Frame::NONE.inner_margin(pad).show(ui, |ui| {
                    note(ui, "Its denoised picture is not made yet, and pictures are not made on their own (Settings, AI Denoise).", t.text_dim);
                    ui.add_space(2.0);
                    let r = ui.small_button(crate::i18n::tr("Make it now"));
                    register(ui.ctx(), "denoise:make", r.rect);
                    if r.clicked() {
                        let _ = app.run("denoise.queue", json!({"ids": [id.0]}));
                    }
                });
            }
        }
        PhotoState::Queued { ahead } => {
            egui::Frame::NONE.inner_margin(pad).show(ui, |ui| {
                let text = if ahead == 0 { "Next up for AI Denoise…".to_string() } else { format!("In line for AI Denoise ({ahead} ahead)…") };
                note(ui, &text, t.text_dim);
            });
        }
        PhotoState::Running { done, total } => {
            egui::Frame::NONE.inner_margin(pad).show(ui, |ui| {
                let frac = if total > 0 { (done as f32 / total as f32).clamp(0.0, 1.0) } else { 0.0 };
                let text =
                    if total > 0 { format!("Making the denoised picture… {}%", (frac * 100.0).round() as u32) } else { "Starting…".to_string() };
                let bar = ui.add(egui::ProgressBar::new(frac).desired_width(ui.available_width().min(240.0)).text(RichText::new(text).size(11.0)));
                register(ui.ctx(), "denoise:progress", bar.rect);
            });
        }
        PhotoState::Failed { why, unsupported } => {
            egui::Frame::NONE.inner_margin(pad).show(ui, |ui| {
                let color = if unsupported { t.text_dim } else { t.caution };
                note(ui, &sentence(&why), color);
                if !unsupported {
                    let r = ui.small_button(crate::i18n::tr("Try again"));
                    register(ui.ctx(), "denoise:retry", r.rect);
                    if r.clicked() {
                        let _ = app.run("denoise.queue", json!({"ids": [id.0], "retry": true}));
                    }
                }
            });
        }
    }
}

/// The Settings ▸ Denoise tab.
pub fn settings_tab(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens) {
    let list = models(app, ui.ctx());
    heading(ui, t, "AI Denoise");
    if list["dir"].is_null() {
        hint(ui, t, "This build has nowhere to keep denoise models, so AI Denoise is not available. (The desktop app has.)");
        return;
    }
    hint(
        ui,
        t,
        "Cleans noise out of raw photos with an AI model, before the picture is built: the Denoise slider under Detail sets how much of it you see. \
         Your files are never changed and no extra files are made in your library: the cleaned picture is cached data that is made again when needed.",
    );
    let runtime = list["runtime"].as_bool() == Some(true);
    if !runtime {
        hint(ui, t, "This build cannot run denoise models.");
    }
    let all: Vec<Value> = list["models"].as_array().cloned().unwrap_or_default();
    let downloads = app
        .session
        .execute("denoise.models.downloads", &json!({}))
        .map(|v| v["downloads"].as_array().cloned().unwrap_or_default())
        .unwrap_or_default();
    heading(ui, t, "Models");
    for m in &all {
        let dl = downloads.iter().find(|d| d["id"] == m["id"]);
        model_row(app, ui, t, m, dl, runtime);
    }
    if all.iter().all(|m| m["selected"] != true) && all.iter().any(|m| m["installed"] == true) {
        hint(ui, t, "AI Denoise is off: press Use on a model.");
    }
    hint(ui, t, "Works on raw files from cameras with a standard (Bayer) sensor. Other files keep their normal noise reduction.");
    work_section(app, ui, t, &list);
}

/// What the cache holds, its limit, and what is being made.
fn work_section(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens, list: &Value) {
    let status = app.session.execute("denoise.status", &json!({})).unwrap_or(Value::Null);
    if status["enabled"] != true {
        return;
    }
    heading(ui, t, "Photos");
    let mut auto = list["auto"].as_bool().unwrap_or(true);
    if check(ui, "settings.denoiseAuto", &mut auto, "Make the denoised picture of the photos I am looking at") {
        let _ = app.run("denoise.settings", json!({"auto": auto}));
        app.caches.denoise.epoch += 1;
    }
    hint(ui, t, "Only photos whose Denoise amount is above 0 are made. Exports always make theirs if it is missing.");
    let (running, queued) = (app.caches.denoise.running, app.caches.denoise.queued);
    match running {
        Some((done, total)) => {
            let frac = if total > 0 { (done as f32 / total as f32).clamp(0.0, 1.0) } else { 0.0 };
            let name = status["running"]["photo"]
                .as_u64()
                .and_then(|p| app.session.catalog.photo(PhotoId(p)).map(|p| p.file_name.clone()))
                .unwrap_or_default();
            let text =
                format!("Denoising {name}: {done} of {total} tiles{}", if queued > 0 { format!(" · {queued} waiting") } else { String::new() });
            let bar = ui.add(egui::ProgressBar::new(frac).desired_width(380.0).text(RichText::new(text).size(11.5)));
            register(ui.ctx(), "denoise:settingsProgress", bar.rect);
        }
        None if queued > 0 => hint(ui, t, &count(queued, "photo waiting", "photos waiting")),
        None => {
            let ready = status["ready"].as_u64().unwrap_or(0);
            hint(ui, t, &count(ready, "photo has its denoised picture.", "photos have their denoised picture."))
        }
    }
    ui.horizontal(|ui| {
        let r = ui.button("Denoise all photos with an Amount");
        register(ui.ctx(), "denoise:queueAll", r.rect);
        if r.clicked() {
            let _ = app.run("denoise.queue", json!({"scope": "withAmount"}));
        }
        if running.is_some() || queued > 0 {
            let r = ui.button("Stop");
            register(ui.ctx(), "denoise:stop", r.rect);
            if r.clicked() {
                let _ = app.run("denoise.cancel", json!({}));
            }
        }
    });
    heading(ui, t, "Cache");
    let (files, bytes) = (status["cache"]["files"].as_u64().unwrap_or(0), status["cache"]["bytes"].as_u64().unwrap_or(0));
    hint(
        ui,
        t,
        &format!(
            "{} · {}. Kept in the library's “denoise” folder; the oldest ones that no photo uses go first when the limit is reached.",
            count(files, "cached picture", "cached pictures"),
            mb(Some(bytes))
        ),
    );
    ui.horizontal(|ui| {
        ui.label(RichText::new(crate::i18n::tr("Limit")).color(t.text_label));
        let current = list["cacheGb"].as_u64().unwrap_or(20) as u32;
        let mut choice = current;
        egui::ComboBox::from_id_salt("denoise-cache-limit").selected_text(format!("{current} GB")).show_ui(ui, |ui| {
            for gb in CACHE_LIMITS {
                ui.selectable_value(&mut choice, gb, format!("{gb} GB"));
            }
        });
        if choice != current {
            let _ = app.run("denoise.settings", json!({"cacheGb": choice}));
            app.caches.denoise.epoch += 1;
        }
        let r = ui.button("Clear cache");
        register(ui.ctx(), "denoise:clear", r.rect);
        if r.clicked() {
            let _ = app.run("denoise.clear", json!({}));
        }
    });
}

fn model_row(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens, m: &Value, dl: Option<&Value>, can_run: bool) {
    let id = m["id"].as_str().unwrap_or("").to_string();
    let (installed, selected) = (m["installed"].as_bool() == Some(true), m["selected"].as_bool() == Some(true));
    let dl_state = if installed { None } else { dl.and_then(|d| d["state"].as_str()) };
    let host = if can_run { m["downloadHost"].as_str().unwrap_or("").to_string() } else { String::new() };
    egui::Frame::NONE.inner_margin(egui::Margin::symmetric(0, 3)).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(m["name"].as_str().unwrap_or("")).color(t.text));
                    let size = mb(m["sizeBytes"].as_u64());
                    if !size.is_empty() {
                        ui.label(RichText::new(size).color(t.text_dim));
                    }
                    if selected {
                        ui.label(RichText::new("in use").color(t.accent));
                    } else if installed {
                        ui.label(RichText::new("installed").color(t.text_dim));
                    }
                });
                ui.label(RichText::new(licence_line(m)).font(t.font(11.5)).color(if m["licence"]["commercial"] == "yes" {
                    t.text_dim
                } else {
                    t.caution
                }));
                match m["accepted"]["selfTest"].as_object() {
                    Some(test) if test.get("ok").and_then(Value::as_bool) == Some(true) => {
                        ui.label(RichText::new("Tested and works on this computer").font(t.font(11.5)).color(t.text_dim));
                    }
                    Some(_) => {
                        ui.label(RichText::new("Failed its last test").font(t.font(11.5)).color(t.caution));
                    }
                    None => {}
                }
                match (dl_state, dl) {
                    (Some("running"), Some(d)) => {
                        let (got, total) = (d["bytes"].as_u64().unwrap_or(0), d["total"].as_u64().unwrap_or(0));
                        let frac = if total > 0 { (got as f32 / total as f32).clamp(0.0, 1.0) } else { 0.0 };
                        let text = if total > 0 && got >= total {
                            "Checking…".to_string()
                        } else {
                            format!("{} of {} from {host}", mb(Some(got)), mb(Some(total)))
                        };
                        let bar = ui.add(egui::ProgressBar::new(frac).desired_width(260.0).text(RichText::new(text).font(t.font(11.5))));
                        register(ui.ctx(), format!("denoise:progress:{id}"), bar.rect);
                    }
                    (Some("done"), _) => {
                        ui.label(RichText::new("Downloaded and checked. Installing…").font(t.font(11.5)).color(t.text_dim));
                    }
                    (Some("failed"), Some(d)) => {
                        let why = sentence(d["error"].as_str().unwrap_or("The download failed"));
                        ui.add(egui::Label::new(RichText::new(why).font(t.font(11.5)).color(t.caution)).wrap());
                    }
                    _ => {}
                }
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if installed {
                    let r = ui.button("Remove");
                    register(ui.ctx(), format!("denoise:remove:{id}"), r.rect);
                    if r.clicked() {
                        let _ = app.run("denoise.models.remove", json!({"id": id}));
                        app.caches.denoise.epoch += 1;
                    }
                    if selected {
                        let r = ui.button("Turn off");
                        register(ui.ctx(), format!("denoise:off:{id}"), r.rect);
                        if r.clicked() {
                            let _ = app.run("denoise.models.select", json!({"id": null}));
                            app.caches.denoise.epoch += 1;
                        }
                    } else {
                        let r = ui.button("Use");
                        register(ui.ctx(), format!("denoise:use:{id}"), r.rect);
                        if r.clicked() {
                            let _ = app.run("denoise.models.select", json!({"id": id}));
                            app.caches.denoise.epoch += 1;
                        }
                    }
                } else {
                    match dl_state {
                        Some("running") => {
                            let r = ui.button("Cancel");
                            register(ui.ctx(), format!("denoise:cancelDownload:{id}"), r.rect);
                            if r.clicked() {
                                let _ = app.run("denoise.models.downloadCancel", json!({"id": id}));
                                app.caches.denoise.dl_watch.retain(|w| w != &id);
                            }
                        }
                        // the engine installs it within a moment; nothing to press
                        Some("done") => {}
                        _ => {
                            if let Some(url) = m["source"].as_str() {
                                let r = ui
                                    .button("Open page")
                                    .on_hover_text("Opens the model's own page in your browser, to read about it or get the file yourself.");
                                register(ui.ctx(), format!("denoise:get:{id}"), r.rect);
                                if r.clicked() {
                                    open_page(app, url);
                                }
                            }
                            if !host.is_empty() {
                                let label = if dl_state == Some("failed") { "Try again" } else { "Download" };
                                let tip = format!(
                                    "Shows the model's terms, then downloads {} from {host}, installs it and starts using it. Nothing else is sent.",
                                    mb(m["sizeBytes"].as_u64())
                                );
                                let r = ui.button(label).on_hover_text(tip);
                                register(ui.ctx(), format!("denoise:download:{id}"), r.rect);
                                if r.clicked() {
                                    open_download_dialog(app, m);
                                }
                            }
                        }
                    }
                }
            });
        });
    });
}

/// Pressing Download: the model's terms first, in the same dialog as a face model. Accepting them starts the download;
/// nothing is fetched before.
fn open_download_dialog(app: &mut LightcraftApp, m: &Value) {
    let info = json!({
        "kind": "known",
        "domain": "denoise",
        "fileName": m["id"],
        "sizeBytes": m["sizeBytes"],
        "model": m,
        "alreadyInstalled": false,
        "download": m["id"],
        "host": m["downloadHost"],
        "assumptions": [],
        "reason": null,
    });
    app.ui.dialog = Some(crate::state::Dialog::FaceModel { path: String::new(), info, accepted: false });
}

/// The dialog's OK for a denoise model: start the download and watch it.
pub fn install(app: &mut LightcraftApp, info: &Value, accepted: bool) -> Result<Value, String> {
    if !accepted {
        return Err("Tick the box to accept the model's terms first".into());
    }
    app.caches.denoise.epoch += 1;
    let id = info["download"].as_str().ok_or("There is no download for this model")?;
    let r = app.run("denoise.models.download", json!({"id": id, "acknowledged": true}));
    if r.is_ok() {
        app.caches.denoise.dl_watch.push(id.to_string());
        // the progress is in Settings: stay there
        app.ui.dialog = Some(crate::state::Dialog::Settings { tab: "denoise".into() });
    }
    r
}

/// "1 photo" / "3 photos": the count with the right noun.
fn count(n: u64, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}
