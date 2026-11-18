//! The built-in denoiser on a picture that is no longer a mosaic: a developed photo, a layer, a scan.
//!
//! The picture's channels (one to four: grey, RGB, CMYK, Lab, … as numbers, in any bit depth turned into floats) are
//! put into the four planes the filter works on: grey as `[v, v, v, v]`, three channels as `[c0, c1, c1, c2]` (the
//! middle one, green in RGB, takes the place of a Bayer cell's two greens), four as they are. Its noise is measured from
//! the picture itself ([`estimate_noise_planes`]), each tile is variance-stabilised with it, cleaned by the same filter
//! as a raw ([`Builtin::filter_on_cpu`]), stabilised back, optionally given the light non-local-means pass, and the
//! tiles are blended. Nothing is demosaiced. The noise of a developed picture is not quite the sensor's shape
//! (variance = `a · signal + b`), but the fitted line still follows how its noise grows with brightness.

use rayon::prelude::*;

use crate::bayer::reflect;
use crate::builtin::{Builtin, NoiseProbe, OVERLAP, TILE, estimate_noise_planes, noise_model};
use crate::method::Method;
use crate::nlm::Stabiliser;
use crate::run::{Control, Error, MAX_PIXELS};
use crate::tiles::{starts, weight};

/// Side of the tiles a picture is cut into, in pixels: the planes are the size of a raw tile's.
pub const IMAGE_TILE: usize = TILE as usize;
/// Pixels neighbouring tiles share: as many pixels as a raw tile's overlap covers (twice its cells), more than the
/// filter's reach.
pub const IMAGE_OVERLAP: usize = 2 * OVERLAP as usize;
/// The smallest variance the noise measurement reports: nothing could be measured.
const NOTHING_MEASURED: (f32, f32) = (0.0, 1e-10);

/// How a picture is cleaned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageOptions {
    pub method: Method,
    /// How much of the cleaning is kept, 0 (the picture as it was) to 1 (all of it).
    pub strength: f32,
    /// The picture's noise as `(a, b)`, variance = `a · value + b`; `None` measures it from the picture.
    pub noise: Option<(f32, f32)>,
}

impl Default for ImageOptions {
    fn default() -> Self {
        ImageOptions { method: Method::Builtin, strength: 1.0, noise: None }
    }
}

fn allocate(n: usize) -> Result<Vec<f32>, Error> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| Error::Input("not enough memory for denoise".into()))?;
    v.resize(n, 0.0);
    Ok(v)
}

/// The noise of an interleaved picture of `channels` channels, as `(a, b)` (variance = `a · value + b`), or `None`
/// when nothing could be measured (too small, or nothing but near-white).
pub fn measure_image_noise(data: &[f32], width: usize, height: usize, channels: usize) -> Option<(f32, f32)> {
    let n = width.checked_mul(height)?;
    if !(1..=4).contains(&channels) || n.checked_mul(channels) != Some(data.len()) || n > MAX_PIXELS {
        return None;
    }
    let planes: Vec<Vec<f32>> =
        (0..channels).map(|c| data.iter().skip(c).step_by(channels).map(|v| if v.is_finite() { *v } else { 0.0 }).collect()).collect();
    let refs: Vec<&[f32]> = planes.iter().map(Vec::as_slice).collect();
    let noise = estimate_noise_planes(&refs, width, height, &NoiseProbe::default());
    (noise != NOTHING_MEASURED).then_some(noise)
}

/// The four filter planes of the `t × t` tile at `(x0, y0)` of `planes` (`channels` planes of `width × height`); the
/// picture's edge is mirrored.
fn pack(planes: &[Vec<f32>], width: usize, height: usize, x0: usize, y0: usize, t: usize) -> Vec<Vec<f32>> {
    let map: [usize; 4] = match planes.len() {
        1 => [0, 0, 0, 0],
        2 => [0, 1, 1, 0],
        3 => [0, 1, 1, 2],
        _ => [0, 1, 2, 3],
    };
    map.iter()
        .map(|&c| {
            let plane = planes.get(c).map_or(&[][..], Vec::as_slice);
            let mut out = Vec::with_capacity(t * t);
            for j in 0..t {
                let y = reflect(y0 as isize + j as isize, height);
                for i in 0..t {
                    let x = reflect(x0 as isize + i as isize, width);
                    out.push(plane.get(y * width + x).copied().unwrap_or(0.0));
                }
            }
            out
        })
        .collect()
}

/// The noise deviation of the brightness plane (half the sum of the four planes) when each channel has unit noise and
/// fills the planes as [`pack`] does: grey fills all four (2), two channels two each (√2), three channels one, two and
/// one (√6 / 2), four channels one each (1).
fn brightness_deviation(channels: usize) -> f32 {
    match channels {
        1 => 2.0,
        2 => std::f32::consts::SQRT_2,
        3 => 6f32.sqrt() / 2.0,
        _ => 1.0,
    }
}

/// The picture's channels back from the four cleaned planes.
fn unpack(cleaned: &[Vec<f32>], channels: usize) -> Vec<Vec<f32>> {
    let p = |i: usize| cleaned.get(i).map_or(&[][..], Vec::as_slice);
    let mean = |ids: &[usize]| -> Vec<f32> {
        let n = p(ids[0]).len();
        (0..n).map(|k| ids.iter().map(|&i| p(i).get(k).copied().unwrap_or(0.0)).sum::<f32>() / ids.len() as f32).collect()
    };
    match channels {
        1 => vec![mean(&[0, 1, 2, 3])],
        2 => vec![mean(&[0, 3]), mean(&[1, 2])],
        3 => vec![p(0).to_vec(), mean(&[1, 2]), p(3).to_vec()],
        _ => (0..4).map(|i| p(i).to_vec()).collect(),
    }
}

/// The light non-local-means pass over a tile's channels, three at a time (a grey or a lone fourth channel as three).
fn nlm_pass(channels: &mut [Vec<f32>], t: usize, st: Stabiliser, strength: f32, b: &Builtin) -> Result<(), Error> {
    let Some(p) = &b.post else { return Ok(()) };
    let n = t * t;
    let groups: Vec<[usize; 3]> = match channels.len() {
        1 => vec![[0, 0, 0]],
        2 => vec![[0, 1, 1]],
        3 => vec![[0, 1, 2]],
        _ => vec![[0, 1, 2], [3, 3, 3]],
    };
    for g in groups {
        let mut rgb = Vec::with_capacity(3 * n);
        for &c in &g {
            rgb.extend_from_slice(channels.get(c).map_or(&[][..], Vec::as_slice));
        }
        let out = crate::nlm_post::apply(&rgb, t, st, strength, p)?;
        for (k, &c) in g.iter().enumerate() {
            if let (Some(dst), Some(src)) = (channels.get_mut(c), out.get(k * n..(k + 1) * n)) {
                dst.copy_from_slice(src);
            }
        }
    }
    Ok(())
}

/// Clean one `t × t` tile: its channels, cleaned.
fn clean_tile(planes: &[Vec<f32>], size: (usize, usize), at: (usize, usize), t: usize, b: &Builtin) -> Result<Vec<Vec<f32>>, Error> {
    let packed = pack(planes, size.0, size.1, at.0, at.1, t);
    let (a, bb) = match b.noise {
        Some((a, bb)) if a.is_finite() && bb.is_finite() => (a, bb),
        _ => noise_model(&packed.iter().map(Vec::as_slice).collect::<Vec<_>>(), t),
    };
    let st = Stabiliser::new(a, bb);
    // The filter expects four planes of independent unit noise, whose brightness (half their sum) has unit noise too.
    // A channel that fills two or four planes adds its noise coherently, so the planes are scaled by the brightness's
    // own deviation (and back after).
    let k = brightness_deviation(planes.len());
    let stabilised: Vec<Vec<f32>> = packed.iter().map(|p| p.iter().map(|&v| st.forward(v) / k).collect()).collect();
    let noise_sd = (a * 0.18 + bb).max(0.0).sqrt();
    let cleaned: Vec<Vec<f32>> = b.filter_on_cpu(&stabilised, t, noise_sd).iter().map(|p| p.iter().map(|&v| st.inverse(v * k)).collect()).collect();
    if cleaned.len() != 4 || cleaned.iter().any(|p| p.len() != t * t) {
        return Err(Error::Runtime("the filter returned planes of the wrong size".into()));
    }
    let mut channels = unpack(&cleaned, planes.len());
    nlm_pass(&mut channels, t, st, b.tuning.strength(noise_sd), b)?;
    if channels.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::Model("numbers that are not finite".into()));
    }
    Ok(channels)
}

/// Denoise `data`, an interleaved picture of `width × height` pixels and `channels` channels (1 to 4; leave alpha out),
/// values as floats (white = 1 for integer depths; float pictures may go beyond). Returns the cleaned picture in the
/// same layout. Tiles run in parallel and are blended in a fixed order, so the result does not depend on timing.
pub fn denoise_image(data: &[f32], width: usize, height: usize, channels: usize, opts: &ImageOptions, ctl: &Control<'_>) -> Result<Vec<f32>, Error> {
    let pixels = width
        .checked_mul(height)
        .filter(|&n| n > 0 && n <= MAX_PIXELS)
        .ok_or_else(|| Error::Input("the picture has no size, or is too large".into()))?;
    if !(1..=4).contains(&channels) {
        return Err(Error::Input("a picture to denoise has one to four channels".into()));
    }
    if pixels.checked_mul(channels) != Some(data.len()) {
        return Err(Error::Input("the picture does not have width × height × channels values".into()));
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(Error::Input("the picture has values that are not finite".into()));
    }
    if !(opts.strength.is_finite() && (0.0..=1.0).contains(&opts.strength)) {
        return Err(Error::Input("the strength must be a number from 0 to 1".into()));
    }
    let noise = match opts.noise {
        Some(n) => Some(n),
        None => measure_image_noise(data, width, height, channels),
    };
    let runner = opts.method.runner(noise);
    let mut planes: Vec<Vec<f32>> = Vec::with_capacity(channels);
    for c in 0..channels {
        let mut p = allocate(pixels)?;
        for (dst, src) in p.iter_mut().zip(data.iter().skip(c).step_by(channels)) {
            *dst = *src;
        }
        planes.push(p);
    }
    let t = IMAGE_TILE;
    let (sx, sy) = (starts(width, t, IMAGE_OVERLAP), starts(height, t, IMAGE_OVERLAP));
    let (cols, rows) = (sx.len(), sy.len());
    if cols == 0 || rows == 0 {
        return Err(Error::Input("the picture needs too many tiles".into()));
    }
    let tiles: Vec<(usize, usize, [bool; 4])> = sy
        .iter()
        .enumerate()
        .flat_map(|(j, &y0)| sx.iter().enumerate().map(move |(i, &x0)| (x0, y0, [i > 0, i + 1 < cols, j > 0, j + 1 < rows])))
        .collect();
    let total = tiles.len();
    let mut acc: Vec<Vec<f32>> = Vec::with_capacity(channels);
    for _ in 0..channels {
        acc.push(allocate(pixels)?);
    }
    let mut wsum = allocate(pixels)?;
    let group = rayon::current_num_threads().clamp(1, 16);
    let mut done = 0;
    for chunk in tiles.chunks(group) {
        if ctl.cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed)) {
            return Err(Error::Cancelled);
        }
        let outs: Vec<Result<Vec<Vec<f32>>, Error>> =
            chunk.par_iter().map(|&(x0, y0, _)| clean_tile(&planes, (width, height), (x0, y0), t, &runner)).collect();
        for (&(x0, y0, edges), out) in chunk.iter().zip(outs) {
            let out = out?;
            let wx: Vec<f32> = (0..t).map(|i| weight(i, t, IMAGE_OVERLAP, edges[0], edges[1])).collect();
            let wy: Vec<f32> = (0..t).map(|j| weight(j, t, IMAGE_OVERLAP, edges[2], edges[3])).collect();
            for (j, &wj) in wy.iter().enumerate() {
                let y = y0 + j;
                if y >= height {
                    break;
                }
                for (i, &wi) in wx.iter().enumerate() {
                    let x = x0 + i;
                    if x >= width {
                        break;
                    }
                    let w = wi * wj;
                    let at = y * width + x;
                    if let Some(s) = wsum.get_mut(at) {
                        *s += w;
                    }
                    for (a, o) in acc.iter_mut().zip(&out) {
                        if let (Some(a), Some(&v)) = (a.get_mut(at), o.get(j * t + i)) {
                            *a += w * v;
                        }
                    }
                }
            }
            done += 1;
            if let Some(f) = ctl.progress {
                f(done, total);
            }
        }
    }
    let mut result = allocate(pixels.saturating_mul(channels))?;
    for (k, px) in result.chunks_exact_mut(channels).enumerate() {
        let w = wsum.get(k).copied().unwrap_or(0.0);
        for (c, v) in px.iter_mut().enumerate() {
            let original = planes.get(c).and_then(|p| p.get(k)).copied().unwrap_or(0.0);
            let cleaned = if w > 0.0 { acc.get(c).and_then(|p| p.get(k)).copied().unwrap_or(0.0) / w } else { original };
            *v = original + opts.strength * (cleaned - original);
        }
    }
    if result.iter().any(|v| !v.is_finite()) {
        return Err(Error::Model("numbers that are not finite after tile blending".into()));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic normal numbers.
    struct Rng(u64);
    impl Rng {
        fn uniform(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 + 0.5) / (1u64 << 24) as f32
        }
        fn normal(&mut self) -> f32 {
            let (u, v) = (self.uniform(), self.uniform());
            (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
        }
    }

    /// A picture with a ramp and a step edge, `channels` channels, and its noisy copy.
    fn scene(w: usize, h: usize, channels: usize, sd: f32, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut rng = Rng(seed);
        let mut clean = Vec::with_capacity(w * h * channels);
        for y in 0..h {
            for x in 0..w {
                for c in 0..channels {
                    let step = if x > w / 2 { 0.3 } else { 0.0 };
                    clean.push(0.15 + 0.3 * y as f32 / h as f32 + step + 0.05 * c as f32);
                }
            }
        }
        let noisy = clean.iter().map(|v| v + sd * rng.normal()).collect();
        (clean, noisy)
    }

    fn rmse(a: &[f32], b: &[f32]) -> f32 {
        (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / a.len().max(1) as f32).sqrt()
    }

    /// Grey, RGB and four-channel pictures over more than one tile come out much closer to the clean picture, with
    /// both methods, and the step edge survives.
    #[test]
    fn noisy_pictures_of_one_to_four_channels_come_out_much_cleaner() {
        let (w, h) = (300, 270);
        for channels in [1, 3, 4] {
            let (clean, noisy) = scene(w, h, channels, 0.04, 11 + channels as u64);
            for method in Method::ALL {
                let out = denoise_image(&noisy, w, h, channels, &ImageOptions { method, ..Default::default() }, &Control::default()).unwrap();
                assert_eq!(out.len(), noisy.len());
                let (before, after) = (rmse(&noisy, &clean), rmse(&out, &clean));
                assert!(after < 0.5 * before, "{channels} channels, {method:?}: {before} -> {after}");
                // the step at the middle column stays a step
                let row = h / 2;
                let at = |x: usize| out[(row * w + x) * channels];
                assert!(at(w / 2 + 4) - at(w / 2 - 4) > 0.2, "{channels} {method:?}");
            }
        }
    }

    /// Strength 0 gives the picture back exactly; the result does not depend on how many threads run.
    #[test]
    fn strength_zero_is_the_input_and_threads_do_not_change_the_picture() {
        let (w, h) = (300, 260);
        let (_, noisy) = scene(w, h, 3, 0.03, 4);
        let none = denoise_image(&noisy, w, h, 3, &ImageOptions { strength: 0.0, ..Default::default() }, &Control::default()).unwrap();
        assert!(none.iter().zip(&noisy).all(|(a, b)| a.to_bits() == b.to_bits()));
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| denoise_image(&noisy, w, h, 3, &ImageOptions::default(), &Control::default()).unwrap())
        };
        assert_eq!(run(1), run(3));
    }

    /// A picture smaller than a tile, a flat picture and an all-white picture give finite results of the right size.
    #[test]
    fn small_flat_and_white_pictures_work() {
        for (w, h) in [(1, 1), (5, 3), (40, 30)] {
            let out = denoise_image(&vec![0.4; w * h * 3], w, h, 3, &ImageOptions::default(), &Control::default()).unwrap();
            assert_eq!(out.len(), w * h * 3);
            assert!(out.iter().all(|v| (v - 0.4).abs() < 1e-3), "{w}x{h}");
        }
        let white = denoise_image(&vec![1.0; 64 * 64], 64, 64, 1, &ImageOptions::default(), &Control::default()).unwrap();
        assert!(white.iter().all(|v| v.is_finite()));
    }

    /// Hostile sizes, channel counts, values and strengths are errors, never panics; a cancel stops it.
    #[test]
    fn hostile_input_is_an_error() {
        let ok = ImageOptions::default();
        let c = Control::default();
        assert!(denoise_image(&[], 0, 0, 3, &ok, &c).is_err());
        assert!(denoise_image(&[0.5; 12], 2, 2, 0, &ok, &c).is_err());
        assert!(denoise_image(&[0.5; 20], 2, 2, 5, &ok, &c).is_err());
        assert!(denoise_image(&[0.5; 11], 2, 2, 3, &ok, &c).is_err());
        assert!(denoise_image(&[0.5; 12], usize::MAX, 2, 3, &ok, &c).is_err());
        assert!(denoise_image(&[f32::NAN; 12], 2, 2, 3, &ok, &c).is_err());
        for strength in [-0.1, 1.5, f32::NAN] {
            assert!(denoise_image(&[0.5; 12], 2, 2, 3, &ImageOptions { strength, ..ok }, &c).is_err());
        }
        let wild = ImageOptions { noise: Some((f32::NAN, f32::INFINITY)), ..ok };
        assert!(denoise_image(&[0.5; 300], 10, 10, 3, &wild, &c).unwrap().iter().all(|v| v.is_finite()));
        let stop = std::sync::atomic::AtomicBool::new(true);
        let cancel = Control { cancel: Some(&stop), ..Default::default() };
        assert!(matches!(denoise_image(&[0.5; 300], 10, 10, 3, &ok, &cancel), Err(Error::Cancelled)));
        assert_eq!(measure_image_noise(&[0.5; 12], 2, 2, 3), None);
        assert_eq!(measure_image_noise(&[0.5; 12], 2, 2, 7), None);
    }
}
