//! AI denoise for LightCraft, as a step between the sensor and the demosaic that the rest of the develop
//! pipeline never sees: the raw file stays untouched, the denoised picture is cached data that can always be
//! made again, and the amount is an ordinary develop setting.
//!
//! - [`manifest`]: what a denoise model takes and gives, and who may use it (untrusted JSON, validated).
//! - [`known`] and [`archive`]: the models LightCraft can fetch, and taking the model file out of the `.zip` it comes in.
//! - [`bayer`]: Bayer layouts and the packing of a mosaic into the four planes a model takes.
//! - [`tiles`] and [`run`]: cutting a picture into overlapping tiles for a fixed-size model, bringing each tile's
//!   output scale back to the input's, keeping clipped highlights as they were, and blending the tiles together.
//!   The model itself is a [`run::TileRunner`], so all of this is tested without one.
//! - [`product`]: the cache file the denoised picture is kept in (half floats, compressed in strips) and the
//!   blend with the plain demosaic that the Amount slider controls.
//! - [`runtime`] (feature `tract`): running an `.onnx` model on the CPU with tract.
//!
//! Everything here treats model files, manifests and cache files as hostile input: sizes are capped, numbers must
//! be finite, and a malformed file is an error, never a panic.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod archive;
pub mod bayer;
pub mod known;
pub mod manifest;
pub mod product;
pub mod run;
#[cfg(feature = "tract")]
pub mod runtime;
pub mod tiles;

pub use manifest::{DenoiserManifest, Domain, Gain, ManifestError};
pub use run::{Control, Error, Params, TileRunner, denoise_bayer};
