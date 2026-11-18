//! The Craft apps' model-free raw denoiser: block-matching collaborative filtering on the variance-stabilised Bayer
//! mosaic, with the photo's noise measured from the photo itself, an edge-directed demosaic, and an optional light
//! non-local-means pass after it. It started as LightCraft's built-in AI Denoise and is shared so that every app that
//! denoises a raw gets the same picture from the same mosaic.
//!
//! * [`raw::denoise_mosaic`]: a whole normalised mosaic (black subtracted, white at 1, not white balanced) to camera RGB,
//!   with a [`Method`] (`"builtin"` or `"builtin-nlm"`). This is what an app calls at its raw stage.
//! * [`run::denoise_bayer`] with a [`run::TileRunner`]: the tiling, highlight protection and blending around any tile
//!   denoiser (the built-in one, [`builtin::Builtin`], or a model an app loads itself).
//! * [`image::denoise_image`]: the same filter on a picture that is no longer a mosaic (one to four channels, any colour
//!   model, any bit depth as floats).
//!
//! No I/O, no `unsafe`, no panics on hostile input, and it builds for `wasm32-unknown-unknown`.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod bayer;
pub mod bm3d;
pub mod builtin;
pub mod image;
pub mod method;
pub mod nlm;
pub mod nlm_post;
pub mod raw;
pub mod run;
pub mod tiles;

pub use bayer::Layout;
pub use image::{ImageOptions, denoise_image, measure_image_noise};
pub use method::Method;
pub use raw::denoise_mosaic;
pub use run::{CameraRgb, Control, Error, Gain, Params, TileRunner, denoise_bayer};
