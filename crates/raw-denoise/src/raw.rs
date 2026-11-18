//! A whole raw photo through the built-in denoiser on the processor, as one call: the photo's noise is measured once
//! from its mosaic, the mosaic is cut into the built-in denoiser's tiles, cleaned and demosaiced, and clipped highlights
//! are kept as the camera recorded them. Two apps that call this on the same mosaic get the same picture, bit for bit.

use crate::bayer::Layout;
use crate::builtin::{OVERLAP, TILE, measure_noise};
use crate::method::Method;
use crate::run::{CameraRgb, Control, Error, Gain, Params, denoise_bayer};

/// Samples at or above this (white = 1) are clipped: the denoiser leaves them as the camera recorded them.
pub const CLIP: f32 = 0.99;

/// The tiling the built-in denoiser runs with: its tile and overlap, no scale of its own, highlights at [`CLIP`] kept,
/// `parallel` tiles at once (the picture does not depend on it).
pub fn params(parallel: usize) -> Params {
    Params { tile: TILE as usize, overlap: OVERLAP as usize, gain: Gain::None, clip: Some(CLIP), parallel: parallel.max(1) }
}

/// Denoise and demosaic the normalised Bayer mosaic `mosaic` (`width × height` samples, black subtracted, white at 1,
/// not white balanced, laid out as `layout`) with `method`. Gives camera RGB at the mosaic's resolution.
///
/// `parallel` is how many tiles run at once (1 to 4 are used); the picture is the same for any value.
pub fn denoise_mosaic(
    mosaic: &[f32],
    width: usize,
    height: usize,
    layout: Layout,
    method: Method,
    parallel: usize,
    ctl: &Control<'_>,
) -> Result<CameraRgb, Error> {
    if width.checked_mul(height) != Some(mosaic.len()) {
        return Err(Error::Input("the mosaic does not have width × height samples".into()));
    }
    let runner = method.runner(measure_noise(mosaic, width, height));
    denoise_bayer(mosaic, width, height, layout, &runner, &params(parallel), ctl)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin::Builtin;
    use crate::run::TileRunner;

    /// Deterministic noise.
    fn mosaic(w: usize, h: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut uniform = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 40) as f32 + 0.5) / (1u64 << 24) as f32
        };
        (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let v = 0.1 + 0.5 * x as f32 / w as f32 + if (x / 40 + y / 30) % 2 == 0 { 0.2 } else { 0.0 };
                let (u, r) = (uniform(), uniform());
                (v + (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * r).cos() * (0.01 * v + 1e-4).sqrt()).max(0.0)
            })
            .collect()
    }

    /// The one call is exactly the steps an app runs itself (measure once, the method's runner, the built-in tiling),
    /// bit for bit, for both methods and any number of tiles at once.
    #[test]
    fn the_one_call_is_the_steps_an_app_runs_bit_for_bit() {
        let (w, h) = (280, 260);
        let m = mosaic(w, h, 3);
        let layout = Layout { dx: 1, dy: 0 };
        for method in Method::ALL {
            let noise = measure_noise(&m, w, h);
            assert!(noise.is_some());
            let runner: Builtin = method.runner(noise);
            let want = denoise_bayer(&m, w, h, layout, &runner as &dyn TileRunner, &params(2), &Control::default()).unwrap();
            for parallel in [0, 3] {
                let got = denoise_mosaic(&m, w, h, layout, method, parallel, &Control::default()).unwrap();
                assert_eq!((got.width, got.height), (w, h));
                assert!(got.data.iter().flatten().zip(want.data.iter().flatten()).all(|(a, b)| a.to_bits() == b.to_bits()), "{method:?}");
            }
        }
    }

    /// The two methods give different pictures, and hostile mosaics are errors, not panics.
    #[test]
    fn the_methods_differ_and_hostile_mosaics_are_errors() {
        let (w, h) = (200, 180);
        let m = mosaic(w, h, 5);
        let layout = Layout { dx: 0, dy: 0 };
        let a = denoise_mosaic(&m, w, h, layout, Method::Builtin, 2, &Control::default()).unwrap();
        let b = denoise_mosaic(&m, w, h, layout, Method::BuiltinNlm, 2, &Control::default()).unwrap();
        assert!(a.data != b.data);
        assert!(denoise_mosaic(&m, w, h + 1, layout, Method::Builtin, 2, &Control::default()).is_err());
        assert!(denoise_mosaic(&[], 0, 0, layout, Method::Builtin, 2, &Control::default()).is_err());
        assert!(denoise_mosaic(&[], usize::MAX, 2, layout, Method::Builtin, 2, &Control::default()).is_err());
        let mut nan = m.clone();
        nan[17] = f32::NAN;
        assert!(denoise_mosaic(&nan, w, h, layout, Method::Builtin, 2, &Control::default()).is_err());
    }
}
