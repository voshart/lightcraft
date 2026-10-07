# AI Denoise

Edit ▸ Detail ▸ **Denoise** is an ordinary slider (`enhance.denoise`, 0–100, the *Amount*). It mixes a cleaner version
of a raw photo into the picture before any other edit touches it, so exposure, white balance, curves and masks see
the cleaned data, as with Lightroom's AI Denoise.

The point of this page: **it is a non-destructive adjustment, not an operation that writes files.** Nothing is added to
your library or next to your originals. Your raw file is never changed and no DNG is made.

## How it works without creating files

A neural denoiser is slow (several seconds a photo even on a fast machine), so its result cannot be computed for every slider tick.
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

The mosaic is cut into 512-cell tiles that overlap by 64 cells, run through the model on the graphics card when there is
one that can run it (our own wgpu kernels, pure Rust) and otherwise on the CPU (tract, pure Rust), and the tiles are
blended across their overlaps with smoothstep weights; clipped highlights are kept as they were (a model
must not invent detail in blown areas). The result is stored as f16 planar strips, byte-shuffled and deflated, with a CRC
per strip, and read back in the window and at the binned size the pipeline's preview needs, so the loupe at any zoom and a
full-size export line up with the plain picture to the pixel.

Measured with the real model on five CC0 raws from `corpus/raw` (one Pentax K-3, one Nikon D5100, one Canon 6D, one Sony
a7 III, one Pixel 2 XL DNG), Ryzen 9 7945HX (16 cores / 32 threads), release build, `cargo test --release --features denoise --test denoise_real -- --ignored`
(`LC_DENOISE_GPU=0` for the processor alone). The first table is the processor; the graphics card follows it:

| | |
|---|---|
| Time to make a picture, start to finish, on the processor | 5–9 s for 12–24 MP at full pace (a 24 MP Sony ARW 9 s, a 12 MP Pixel DNG 5 s), in the background |
| Fine-detail roughness at Amount 100 (mean absolute second difference of the luma, 1600 px preview) | 4–26 % lower than the plain picture, by file |
| Cache size per photo | 41–68 MB |
| Memory | about 1.9 GB peak working set (release build) for the whole test process (one file each, a 24 MP Sony and a 12 MP Pixel: making the picture, two renders and an export) |
| Fuji X-Trans | reported as "not supported", cleanly, in 0.3 s |

Those are *measurements of whether it works* — a lower roughness is not proof it looks better. There is no side-by-side
comparison against Lightroom's AI Denoise yet (see *Render fidelity* in the roadmap).

### On a graphics card

The same network runs as WGSL compute kernels (`crates/gpu/src/nn.rs`, `wgsl/nn_conv.wgsl`, `wgsl/nn_pool.wgsl`): each
convolution is a tiled matrix product (the 3 × 3 taps and the input channels are the rows of the weight matrix) with the
leaky ReLU, the join of a skip connection, the transposed convolution and the last depth-to-space folded into the same
kernels, and activations kept in as few buffers as the network's lifetimes allow. It is pure Rust and runs through wgpu on
DX12, Vulkan or Metal; the device is a separate one from the interactive renders (they share the switches and the crash
sentinel: `LIGHTCRAFT_GPU=0`, `LIGHTCRAFT_GPU_BACKEND`, the GPU rendering preference).

Measured on an NVIDIA GeForce RTX 4090 Laptop GPU (the main adapter tried; a second, small one is below), release build, the real model:

| | |
|---|---|
| One 512-cell tile on the card | 12–15 ms with two tiles in flight (DX12; Vulkan 13–16 ms). One CPU core takes 1.4 s; the whole 16-core CPU at its best takes 6.7 s for a 24 MP photo's 35 tiles, the card about 0.45 s |
| The card's answer against tract's | within 1.6 × 10⁻⁶ of the largest value in the tile |
| A picture, start to finish | about 1.0–1.8 s for 12–24 MP (a 3–4 s outlier or two when the machine was busy with other work); the first photo of a session also pays about 1 s to set the card up and 1.4 s for the check against the CPU |
| Where a 24 MP photo's second goes | reading and decoding 0.15–0.25 s, packing + the model + blending 0.55 s, writing the picture 0.3–0.7 s (the numbers move with what else the machine is doing) |
| Setting up | about 1 s: the three convolution kernels build in about 0.2 s each on DX12 |
| Video memory | the runner's own count is 164 MB per tile in flight (two at once) plus the 31 MB of weights |

Rules it follows, so a graphics card never makes things worse:

- **It has to agree with the processor.** The first time a model is used, the card and the CPU runner both run a test tile
  and the card is only used if the answers match within 10⁻³ of the largest value. A card that disagrees, runs out of
  memory, or cannot build the kernels is not used, and Settings says why (`denoise.status` → `device`).
- **A tile it fails is run on the CPU**, and a device that errors or is lost is not used again in this session, so a
  photo is always finished.
- **Only networks it knows**: convolutions (1 × 1 and 3 × 3), 2 × 2 transposed convolutions, leaky ReLU, 2 × 2 max-pool, skip
  joins and a final depth-to-space, with every channel count a multiple of 4 and a tile up to 1024 cells. Another model
  runs on the CPU.
- **Software adapters are skipped** (the CPU is faster than a software rasteriser).
- Settings ▸ AI Denoise ▸ Speed has **Use the graphics card when it can run the model** (default on; `denoise.settings {gpu}`),
  and under it the adapter and its time per tile. Four tiles at once are enough to keep a card busy, so the pace setting
  lends the processor fewer threads while a card does the work.

To try another card (more than one GPU, or a laptop with an integrated one next to a discrete one): the adapter is picked
as the high-performance one, or by name with `LIGHTCRAFT_GPU_ADAPTER=<part of its name>` (e.g. `=radeon`, `=intel`), and
the API with `LIGHTCRAFT_GPU_BACKEND=dx12|vulkan|metal`. `LC_DENOISE_MODEL=<model_bayer.onnx> cargo test --release -p lightcraft-gpu --lib real_model_on_the_gpu_matches_tract -- --ignored --nocapture`
checks the card against tract on the real network and prints the adapter, its set-up time and its time per tile;
`time_a_shader_build` (same crate, with `LC_SHADER` and `LC_BLOCKS`) times a kernel build. The kernels are checked on
synthetic networks of several sizes against a plain-loop reference interpreter on every test run that has an adapter.

DX12 and Vulkan gave the same speed here, but DX12's shader compiler took 53 s over one convolution kernel until wgpu's
workgroup-memory zeroing was switched off for these pipelines (it is written out as one store per element); it now takes
0.2 s. Another vendor's compiler may have other surprises: a first look at the set-up time in the test above is worth it.

### On the processor: why it takes that long, and what a smaller machine gets

The model is a U-Net run over 512 × 512-cell tiles (1 MP each) that overlap by 64 cells, so a 24 MP photo is 35 tiles and a
12 MP photo about 15. One tile costs about 1.4 s on one core of the test machine (an AMD Ryzen 9 7945HX, 16 cores / 32 threads, a high-end laptop chip), so the whole
photo is roughly 50 s of CPU work, and the question is how many cores share it. The model stage alone, on the 24 MP Sony
(release build, `LC_DENOISE_PARALLEL=n`):

| Tiles run at once | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|
| Seconds | 50 | 15 | 9.8 | 7.4 | 6.7 |

It stops improving past about 16 because the test machine has 16 physical cores (the other 16 threads are their
hyper-thread siblings) and because 35 tiles split unevenly into waves. Reading and decoding the raw and writing the
picture add about 2 s, which makes the 9 s above. The same run in the repository's default *dev* build is about 40 %
slower (10.6 s at 16), so a debug-ish build is not what to time.

What that means elsewhere (estimates from the table, not measurements on those machines):

- **While you are working in the window** the queue runs two tiles at once (pace `light`), so a 24 MP photo takes
  about 30 s. Nothing blocks: the loupe keeps showing the normal picture, with a progress bar under the slider.
- **A typical 8-thread laptop** at its `normal` pace (4 tiles at once) would be in the 15–30 s range per 24 MP photo,
  depending on how fast its cores are. A slow or old CPU could take a minute or more.
- **It is paid once per photo and model.** The picture is cached, so moving the slider, reopening the photo and the
  second export are instant; only changing the file or the model repeats it.
- **Still slow compared with what people expect from a slider.** Waiting seconds for a photo's first clean picture is
  the cost of running a neural network on a CPU; a graphics card (above) is the fix, and a smaller or quantised model
  would be another. The only model offered here is the 31 MB RawNIND one.

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
- **The graphics card is tried on two adapters, both on Windows with DX12.** An NVIDIA RTX 4090 Laptop GPU (also on
  Vulkan) and the small AMD Radeon 610M (2 compute units) built into the same laptop's CPU. Both give tract's answer to
  1.6 × 10⁻⁶ of the largest value. Intel and Apple GPUs, Vulkan or Metal on other drivers, and macOS and Linux generally
  are untested; the check against the CPU and the per-tile fallback are what keep an untested card from doing harm, not
  evidence that it works or is fast.
- **A small card is slower than a big processor, and nothing switches away from it.** The Radeon 610M takes 485–494 ms
  a tile: about 3× faster than one CPU core (1.46 s) but about 2.5× slower than this machine's whole 16-core CPU
  (about 0.19 s a tile at its best), so a 24 MP photo would take about 17 s on it against about 7 s on the processor.
  The runner asks for the high-performance adapter (here the NVIDIA one), but on a machine whose only GPU is an integrated
  one the card is used while the checkbox is on, even where the processor would be faster. The set-up already runs a
  tile on both, so comparing them is free; it is not done yet.
- **tract stays the processor path** and the reference. The existing pure-Rust GPU ONNX runtime, wonnx, is archived and has
  no transposed convolution or depth-to-space (per the operator table on its repository page, October 2026), so the runner is ours:
  `lightcraft_gpu::nn`, driven by a plain description of the network (`lightcraft_denoise::net`) read from the ONNX file.
- **No quality comparison** against Lightroom, and no tuning of the blend for dark or clipped areas beyond keeping clipped
  highlights.
- **The weights are GPL-3.0** (above), and RawNIND was trained on a limited set of sensors.

## What is next

1. A maintainer decision on the model: keep the opt-in GPL download, or **train our own on the RawNIND data** (CC BY 4.0 /
   CC0, the paper describes the recipe) and publish MIT/Apache weights; then the button can be a plain one.
2. A linear-RGB contract for X-Trans and phone DNGs.
3. **The graphics card on more hardware, and faster.** The model does about 93 GFLOP per tile (tract's cost model,
   convolutions only), 3.2 TFLOP for a 24 MP photo, so the card's 12–15 ms per tile is about 7 TFLOP/s. The kernels are
   plain 32-bit shaders, so there is probably headroom (16-bit weights and activations, cooperative-matrix instructions
   where the adapter has them), but that is a guess: the card's peak was not measured. More worth doing first: try AMD, Intel and
   Apple adapters and an integrated GPU (see *On a graphics card* for how), and overlap one photo's decode and write
   with the next photo's tiles, since on the card the rest of the pipeline is now more than half of a photo's time.
4. A side-by-side fidelity suite (shared with the render-fidelity work).
