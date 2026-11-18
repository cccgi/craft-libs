# craft-raw-denoise

The Craft apps' model-free raw denoiser. It started as LightCraft's built-in AI Denoise (`lightcraft-denoise-core`)
and lives here so that every app that denoises a raw gets the same picture from the same mosaic. No weights, no
download, no I/O, no `unsafe`, no panics on hostile input; it builds for `wasm32-unknown-unknown`.

```toml
craft-raw-denoise = { git = "https://github.com/storytold/craft-libs", rev = "<commit>" }
```

## A raw photo

```rust
use craft_raw_denoise::{Control, Layout, Method, denoise_mosaic};

// mosaic: the sensor samples with black subtracted and white at 1, not white balanced, width × height
let layout = Layout::from_cell([0, 1, 1, 2]).ok_or("not a Bayer cell")?; // RGGB at the mosaic's origin
let rgb = denoise_mosaic(&mosaic, width, height, layout, Method::Builtin, 4, &Control::default())?;
// rgb: camera RGB at the mosaic's resolution (denoised and demosaiced): white balance and colour come next
```

- The photo's noise is measured once from its mosaic (`builtin::measure_noise`: a Poisson-Gaussian line, variance =
  `a · signal + b`, from the quietest blocks at each brightness), so the ISO need not be known.
- The mosaic is cut into 256-cell tiles overlapping by 16; each tile is variance-stabilised with that noise, cleaned
  by block matching (`bm3d`, from Dabov et al. 2007), stabilised back and demosaiced (edge-directed green,
  colour-difference red and blue). Samples at or above 0.99 keep their recorded value, so clipped highlights stay
  clipped. Tiles are blended in a fixed order: the picture does not depend on the thread count.
- `Method::BuiltinNlm` (`"builtin-nlm"`) adds a light non-local-means pass after each tile (`nlm_post`, from Buades,
  Coll and Morel 2005), patches compared on stabilised brightness and the weights applied to all three colours.

Apps that run their own models or a GPU use `run::denoise_bayer` with their own `run::TileRunner` and
`builtin::Builtin::run_with`, which keeps the stabilise / inverse / demosaic steps shared with the processor path.

## A developed picture

`image::denoise_image` runs the same filter on a picture that is no longer a mosaic: one to four channels as floats
(grey, RGB, CMYK, Lab; any bit depth), its noise measured from the picture itself
(`builtin::estimate_noise_planes`), strength 0 to 1. Nothing is demosaiced.

## Ids and versions

`Method::id()` is `"builtin"` or `"builtin-nlm"`; `Method::version()` changes whenever the pictures change, so a
cache keyed by it is made again.
