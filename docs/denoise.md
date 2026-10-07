# AI Denoise

Edit ▸ Detail ▸ **Denoise** is an ordinary slider (`enhance.denoise`, 0–100, the *Amount*). It mixes a cleaner version
of a raw photo into the picture before any other edit touches it, so exposure, white balance, curves and masks see
the cleaned data, as with Lightroom's AI Denoise.

The point of this page: **it is a non-destructive adjustment, not an operation that writes files.** Nothing is added to
your library or next to your originals. Your raw file is never changed and no DNG is made.

## How it works without creating files

A neural denoiser is slow (about ten seconds a photo here), so its result cannot be computed for every slider tick.
LightCraft separates the two jobs:

- **The Amount is a develop setting.** It lives in the photo's edits like any slider: history, copy and paste, presets,
  virtual copies, undo, export all treat it as one more number. Changing it is instant.
- **The cleaned picture is a cache.** It is a pure function of (the raw file, the model, the algorithm version), so it is
  made in the background, kept in `<library>/denoise/<key>.lcdn`, and made again whenever it is missing. It is not part of
  the catalog, is never backed up as data, and deleting the folder (Settings ▸ AI Denoise ▸ **Clear cache**) loses
  nothing but time. Its key is a hash of the file's content, the model and the algorithm, so replacing the file, switching
  model or fixing the algorithm never shows a stale picture.
- **The render mixes the two.** The pipeline develops the raw twice from one decode (once as usual, once from the model's
  output) through the same colour finish, then blends them by the Amount. Only the blend is redone when you drag.

This is the strategy darktable uses for its raw denoise, and the reason there are no duplicate DNGs: the expensive
step is cached by content, the cheap step is live.

What the user sees:

| Situation | Under the slider |
|---|---|
| Raw photo, no model installed | "AI Denoise needs a model" and a **Set up AI Denoise…** button |
| Amount > 0, picture not made yet | "In line…", then a progress bar; the loupe and thumbnails show the normal picture until it is ready, then switch to the mix |
| Picture made | nothing (the slider just works) |
| Not a Bayer raw (X-Trans, a JPEG, a demosaiced DNG) | a short note; the file keeps its normal noise reduction |
| The model failed on this file | the reason, and **Try again** |
| Pictures not made on their own (Settings) | "Make it now" |

**Exports always get it.** An export of a photo with an Amount makes the picture itself when the queue has not, and
**fails with a clear message** (never silently exports the unclean picture) if it cannot: no model, a failing model,
nowhere to keep it. The CLI, MCP and control channel go through the same path.

## Setting a model up

LightCraft ships **no** denoise model. It follows the same rule as face recognition models ([faces.md](faces.md)): a
model is fetched only when you press **Download** and accept its terms, and only from the address pinned in the code,
checked against its recorded size and SHA-256.

**Settings ▸ AI Denoise ▸ Download** on *RawNIND UtNet2 (Bayer)* (31 MB), or **Use** / add your own model file (below).
Installing runs a self-test (it must give finite numbers, give the same answer twice, bring noise down on a synthetic
tile and have the scale its description says), then chooses the model.

### The model, and its licence

*RawNIND UtNet2* is a U-Net trained on RawNIND (real photographs of the same scene taken noisy and clean, CC BY 4.0 / CC0;
paper arXiv 2501.08924), packaged as ONNX for darktable (`darktable-ai`, release-5.6.0). **Its weights are published under
GPL-3.0.** LightCraft is MIT OR Apache-2.0 and does not bundle, link or copy them: the file is downloaded to the user's
computer at their request after showing the terms, and read by LightCraft's own tract-based runner. Whether that is
acceptable for distribution or for releases that offer the button is a **maintainer decision**; the alternatives are in
*What is next*. Nothing here reads or copies code from darktable (GPL): only the published ONNX file is used, through
LightCraft's own tiling, packing and blending code (`crates/denoise`).

### Bring your own model

Put the `.onnx` file anywhere and a `denoise-model.json` next to it (or use a file LightCraft knows by hash). Adding a
model shows its terms and needs the same "I accept" box.

```json
{
  "id": "my-denoiser",
  "name": "My denoiser",
  "version": "1",
  "licence": { "name": "MIT", "commercial": "yes", "url": "https://opensource.org/license/mit" },
  "source": "https://example.org/where-it-came-from",
  "sha256": "…64 lowercase hex digits (optional)…",
  "provenance": "what it was trained on, or \"undisclosed\"",
  "domain": "bayerToRgb",
  "tile": 512,
  "overlap": 64,
  "gain": { "kind": "none" }
}
```

`domain: bayerToRgb` is the one contract today: the model takes the sensor's mosaic, black subtracted and normalised,
**not white balanced**, packed as four planes `[R, G1, G2, B]` of each 2 × 2 cell (always as RGGB), as `[1, 4, tile, tile]`,
and gives camera RGB at the mosaic's resolution, `[1, 3, 2·tile, 2·tile]`: it denoises *and demosaics*. `gain` says
how its output scale is brought back to the input's (`none`, or `matchMean` with `nominal` and `maxDeviation`, for a model
whose output has a scale of its own, as RawNIND's is: about a million times the input). Every field is validated
(sizes, overlap, id, https-only addresses) before a manifest is used.

## What it does to a picture

The mosaic is cut into 512-cell tiles that overlap by 64 cells, run through the model on the CPU (tract, pure Rust), and
the tiles are blended across their overlaps with smoothstep weights; clipped highlights are kept as they were (a model
must not invent detail in blown areas). The result is stored as f16 planar strips, byte-shuffled and deflated, with a CRC
per strip, and read back in the window and at the binned size the pipeline's preview needs, so the loupe at any zoom and a
full-size export line up with the plain picture to the pixel.

Measured with the real model on five CC0 raws from `corpus/raw` (one Pentax K-3, one Nikon D5100, one Canon 6D, one Sony
a7 III, one Pixel 2 XL DNG), 32 cores, release build, `cargo test --features denoise --test denoise_real -- --ignored`:

| | |
|---|---|
| Time to make a picture | 8–15 s for 12–24 MP at full pace, in the background |
| Fine-detail roughness at Amount 100 (mean absolute second difference of the luma, 1600 px preview) | 4–26 % lower than the plain picture, by file |
| Cache size per photo | 41–68 MB |
| Memory | about 1.9 GB peak working set for the whole test process (one file each, a 24 MP Sony and a 12 MP Pixel: making the picture, two renders and an export) |
| Fuji X-Trans | reported as "not supported", cleanly, in 0.3 s |

Those are *measurements of whether it works* — a lower roughness is not proof it looks better. There is no side-by-side
comparison against Lightroom's AI Denoise yet (see *Render fidelity* in the roadmap).

## Making pictures

- **Automatically**, for the photo open in the loupe and the ones selected, when their Amount is above 0 (Settings ▸ AI
  Denoise ▸ "Make the denoised picture of the photos I am looking at", on by default). The open photo goes first.
- **On request**: **Denoise all photos with an Amount**, or `denoise.queue` with a list of ids or `scope`.
- **Pace**: the queue follows what the person is doing, like face scanning: paused for a moment after each drag or keystroke,
  light while the pointer is moving, normal when the window is idle or behind another app and full when Settings ▸ AI Denoise
  is open. `denoise.pump {pace}` sets it for headless use.
- One picture at a time, each using several worker threads; the memory gate that raw decodes share applies while the raw
  is decoded. A failure is remembered per file and model, not retried until asked.

### Cache limits

20 GB by default (1–10,000 GB in Settings); what no open photo uses goes first, oldest first. The folder is just
`<key>.lcdn` files: it is safe to delete while LightCraft is closed or with **Clear cache**.

## Commands

All of it is `denoise.*` commands (UI, CLI, control channel and MCP use the same ids): `denoise.models.list`, `.install`,
`.download`, `.downloads`, `.downloadCancel`, `.test`, `.remove`, `.select`; `denoise.settings`, `denoise.queue`,
`denoise.cancel`, `denoise.status`, `denoise.pump`, `denoise.clear`. A build without tract (`--no-default-features`,
and the web build) has no runner: the commands say so and the slider stays out of the way.

## What is not done

- **Bayer raws only.** X-Trans (Fujifilm), Foveon, already demosaiced DNGs and non-raw photos keep their normal noise
  reduction. A *linear* (demosaiced RGB) model would cover those and DNGs from phones; the contract does not have it yet.
- **CPU only.** tract is single-threaded per tile; the pipeline parallelises across tiles. A GPU path would turn ten
  seconds into one or two.
- **No quality comparison** against Lightroom, and no tuning of the blend for dark or clipped areas beyond keeping clipped
  highlights.
- **The weights are GPL-3.0** (above), and RawNIND was trained on a limited set of sensors.

## What is next

1. A maintainer decision on the model: keep the opt-in GPL download, or **train our own on the RawNIND data** (CC BY 4.0 /
   CC0, the paper describes the recipe) and publish MIT/Apache weights; then the button can be a plain one.
2. A linear-RGB contract for X-Trans and phone DNGs.
3. A GPU path.
4. A side-by-side fidelity suite (shared with the render-fidelity work).
