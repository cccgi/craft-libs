//! An optional light non-local-means pass over the built-in denoiser's picture ("Non-local means (after built-in)").
//!
//! It is written from the published description of non-local means (Buades, Coll and Morel, 2005): each pixel becomes
//! the weighted average of the pixels around it, where a neighbour's weight falls off with how much the small patch
//! around it differs from the patch around the pixel itself. Only what the paper says is used here.
//!
//! * The patches are compared on brightness alone (the stabilised red, green and blue combined into one plane with unit
//!   noise), and the weights found there average all three colour planes, so colour is cleaned from the brightness's
//!   matches and cannot drift apart from it.
//! * The photo's noise comes from the built-in denoiser's own measurement ([`crate::builtin::estimate_noise`]): the
//!   picture is variance-stabilised with it, so one threshold fits every brightness, and how hard the pass works follows
//!   the noise the same way the built-in filter's does ([`crate::nlm::Tuning::strength`]): little at low ISO, more above.
//! * The pass runs on what is left after the built-in denoiser, which is far below the photo's own noise, so the
//!   threshold is a fraction of it ([`Params::residual`]); it is meant to be light.
//! * Cost is bounded. The search radius and patch radius are capped ([`MAX_SEARCH`], [`MAX_PATCH`]), the patch distance of
//!   every offset is a box average (a cost per pixel that does not grow with the patch), and it runs on one tile at a time
//!   with [`working_bytes`] of working memory, so a 24 MP photo costs the same memory as a small one.

use crate::nlm::Stabiliser;
use crate::run::Error;

/// The largest search radius, in pixels (a window of up to 11 × 11 around each pixel).
pub const MAX_SEARCH: usize = 5;
/// The largest patch radius: a patch is `2 · radius + 1` pixels on a side, 3 to 7.
pub const MAX_PATCH: usize = 3;
/// Smallest and largest side of a picture this works on (a built-in tile is 512).
pub const MIN_SIDE: usize = 8;
pub const MAX_SIDE: usize = 4096;
/// Beyond this the weight is below 0.0004 and skipped.
const CUT: f32 = 8.0;

/// What the pass may be tuned by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Search radius in pixels (1 to [`MAX_SEARCH`]; more is capped).
    pub search: usize,
    /// Patch radius in pixels (1 to [`MAX_PATCH`]; more is capped), so patches of 3 to 7 pixels on a side.
    pub patch: usize,
    /// The filter strength `h` as a multiple of the noise left after the built-in denoiser: larger blurs more.
    pub h: f32,
    /// The noise left after the built-in denoiser, as a fraction of the photo's own (unit) noise.
    pub residual: f32,
}

impl Default for Params {
    fn default() -> Self {
        Params { search: 4, patch: 2, h: 0.8, residual: 0.3 }
    }
}

impl Params {
    /// How far a result pixel can be changed by pixels beyond it: what a tile's overlap must cover.
    pub fn reach(&self) -> usize {
        self.search.clamp(1, MAX_SEARCH) + self.patch.clamp(1, MAX_PATCH)
    }

    fn check(&self) -> Result<(), Error> {
        if !(self.h.is_finite() && self.h > 0.0 && self.residual.is_finite() && self.residual > 0.0) {
            return Err(Error::Input("the non-local-means strength must be a positive number".into()));
        }
        Ok(())
    }
}

/// The working memory of the pass on a picture of `side × side` pixels (three colour planes in, the same out, and nine
/// planes of working space), in bytes; it depends on the tile, never on the size of the photo.
pub fn working_bytes(side: usize) -> usize {
    side.saturating_mul(side).saturating_mul(15 * std::mem::size_of::<f32>())
}

/// The mean of each pixel's `2r + 1` square (cells beyond the edge repeat the edge), through `tmp`.
fn box_mean(src: &[f32], dst: &mut [f32], tmp: &mut [f32], s: usize, r: usize) {
    let (si, ri) = (s as isize, r as isize);
    let inv = 1.0 / ((2 * r + 1) * (2 * r + 1)) as f32;
    let at = |row: &[f32], i: isize| row.get(i.clamp(0, si - 1) as usize).copied().unwrap_or(0.0);
    for (row, out) in src.chunks_exact(s).zip(tmp.chunks_exact_mut(s)) {
        let mut sum: f32 = (-ri..=ri).map(|k| at(row, k)).sum();
        for (x, o) in out.iter_mut().enumerate() {
            *o = sum;
            let x = x as isize;
            sum += at(row, x + ri + 1) - at(row, x - ri);
        }
    }
    let rows = |y: isize| {
        let y = y.clamp(0, si - 1) as usize;
        tmp.get(y * s..(y + 1) * s).unwrap_or(&[])
    };
    let mut col = vec![0f32; s];
    for k in -ri..=ri {
        for (c, v) in col.iter_mut().zip(rows(k)) {
            *c += v;
        }
    }
    for (y, out) in dst.chunks_exact_mut(s).enumerate() {
        for (o, c) in out.iter_mut().zip(&col) {
            *o = c * inv;
        }
        let y = y as isize;
        let (add, sub) = (rows(y + ri + 1), rows(y - ri));
        for ((c, a), b) in col.iter_mut().zip(add).zip(sub) {
            *c += a - b;
        }
    }
}

/// Clean `rgb` (three planes `[R, G, B]` of `side × side`, linear camera values as the built-in denoiser gives them).
///
/// `stabiliser` is the photo's noise model, `strength` is 0 for a lightly noisy photo up to 1 for a heavily noisy one
/// ([`crate::nlm::Tuning::strength`]). Anything that is not three finite planes of a sensible size is an error.
pub fn apply(rgb: &[f32], side: usize, stabiliser: Stabiliser, strength: f32, p: &Params) -> Result<Vec<f32>, Error> {
    p.check()?;
    if !(MIN_SIDE..=MAX_SIDE).contains(&side) {
        return Err(Error::Input("the picture for the non-local-means pass has no usable size".into()));
    }
    let n = side * side;
    if rgb.len() != 3 * n {
        return Err(Error::Input("the non-local-means pass needs three square planes".into()));
    }
    if !strength.is_finite() || rgb.iter().any(|v| !v.is_finite()) {
        return Err(Error::Input("the non-local-means pass was given values that are not finite".into()));
    }
    let (search, patch) = (p.search.clamp(1, MAX_SEARCH) as isize, p.patch.clamp(1, MAX_PATCH));
    let (r, g, b) = (rgb.get(..n).unwrap_or(&[]), rgb.get(n..2 * n).unwrap_or(&[]), rgb.get(2 * n..).unwrap_or(&[]));
    let luma: Vec<f32> = r
        .iter()
        .zip(g)
        .zip(b)
        .map(|((&r, &g), &b)| (stabiliser.forward(r) + 2.0 * stabiliser.forward(g) + stabiliser.forward(b)) / 6f32.sqrt())
        .collect();
    let h = p.h * p.residual * (0.25 + 0.75 * strength.clamp(0.0, 1.0));
    let inv_h2 = 1.0 / (h * h).max(1e-12);
    // two patches of the same signal differ by twice the noise's variance
    let floor = 2.0 * p.residual * p.residual;

    let mut acc = vec![0f32; 3 * n];
    let (mut wsum, mut wmax) = (vec![0f32; n], vec![0f32; n]);
    let (mut d2, mut dist, mut tmp) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    let si = side as isize;
    for dy in -search..=search {
        for dx in -search..=search {
            if dx == 0 && dy == 0 {
                continue;
            }
            for (y, out) in d2.chunks_exact_mut(side).enumerate() {
                let ny = (y as isize + dy).clamp(0, si - 1) as usize;
                let (row, mine) = (luma.get(ny * side..(ny + 1) * side).unwrap_or(&[]), luma.get(y * side..(y + 1) * side).unwrap_or(&[]));
                for (x, (o, m)) in out.iter_mut().zip(mine).enumerate() {
                    let nx = (x as isize + dx).clamp(0, si - 1) as usize;
                    let d = m - row.get(nx).copied().unwrap_or(0.0);
                    *o = d * d;
                }
            }
            box_mean(&d2, &mut dist, &mut tmp, side, patch);
            for y in 0..side {
                let ny = (y as isize + dy).clamp(0, si - 1) as usize;
                for x in 0..side {
                    let i = y * side + x;
                    let e = (dist.get(i).copied().unwrap_or(0.0) - floor).max(0.0) * inv_h2;
                    if e >= CUT {
                        continue;
                    }
                    let w = (-e).exp();
                    let j = ny * side + (x as isize + dx).clamp(0, si - 1) as usize;
                    for c in 0..3 {
                        if let (Some(a), Some(v)) = (acc.get_mut(c * n + i), rgb.get(c * n + j)) {
                            *a += w * v;
                        }
                    }
                    if let (Some(s), Some(m)) = (wsum.get_mut(i), wmax.get_mut(i)) {
                        *s += w;
                        *m = m.max(w);
                    }
                }
            }
        }
    }
    // the pixel itself counts as much as its best match
    let mut out = vec![0f32; 3 * n];
    for i in 0..n {
        let (s, m) = (wsum.get(i).copied().unwrap_or(0.0), wmax.get(i).copied().unwrap_or(0.0));
        let m = if m > 0.0 { m } else { 1.0 };
        for c in 0..3 {
            if let (Some(o), Some(a), Some(v)) = (out.get_mut(c * n + i), acc.get(c * n + i), rgb.get(c * n + i)) {
                *o = (a + m * v) / (s + m);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic standard normal numbers (Box-Muller).
    struct Noise(u64);

    impl Noise {
        fn normal(&mut self) -> f32 {
            let mut u = || {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((self.0 >> 40) as f32 + 0.5) / (1u64 << 24) as f32
            };
            let (a, b) = (u(), u());
            (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
        }
    }

    /// The noise the stabiliser sees as unit: variance `b`, no signal-dependent part.
    const B: f32 = 1e-4;

    fn stabiliser() -> Stabiliser {
        Stabiliser::new(0.0, B)
    }

    /// Three planes of `s × s`: a left half at `lo` and a right half at `hi` (a vertical step), plus noise of `sigma`
    /// deviation (in the stabiliser's units: `sigma · sqrt(B)`).
    fn step_scene(s: usize, lo: f32, hi: f32, sigma: f32, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut rng = Noise(seed);
        let truth: Vec<f32> = (0..3 * s * s).map(|i| if (i % s) < s / 2 { lo } else { hi }).collect();
        let noisy = truth.iter().map(|v| v + sigma * B.sqrt() * rng.normal()).collect();
        (truth, noisy)
    }

    fn std_dev(v: &[f32], mean: f32) -> f32 {
        (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
    }

    /// The first plane's values in a window of columns `x0..x1` and rows `y0..y1`.
    fn window(v: &[f32], s: usize, x0: usize, x1: usize, y0: usize, y1: usize) -> Vec<f32> {
        (y0..y1).flat_map(|y| (x0..x1).map(move |x| v[y * s + x])).collect()
    }

    #[test]
    fn a_noisy_flat_patch_loses_noise_and_a_sharp_edge_keeps_its_contrast() {
        let s = 64;
        let (truth, noisy) = step_scene(s, 0.3, 0.6, 0.3, 7);
        let out = apply(&noisy, s, stabiliser(), 1.0, &Params::default()).unwrap();
        // flat areas, away from the edge and the picture's border
        let before = std_dev(&window(&noisy, s, 6, 24, 6, 58), 0.3);
        let after = std_dev(&window(&out, s, 6, 24, 6, 58), 0.3);
        assert!(after < 0.75 * before, "noise {before:e} -> {after:e}");
        // the step: the jump between the two columns beside it, against the true jump
        let edge = |v: &[f32]| {
            let (l, r) = (window(v, s, s / 2 - 1, s / 2, 8, 56), window(v, s, s / 2, s / 2 + 1, 8, 56));
            r.iter().sum::<f32>() / r.len() as f32 - l.iter().sum::<f32>() / l.len() as f32
        };
        let (want, got) = (edge(&truth), edge(&out));
        assert!(got > 0.85 * want, "edge contrast {want} -> {got}");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn a_larger_strength_cleans_more_and_a_lightly_noisy_photo_is_touched_less() {
        let s = 48;
        let (_, noisy) = step_scene(s, 0.4, 0.4, 0.3, 3);
        let flat = |v: &[f32]| std_dev(&window(v, s, 6, 42, 6, 42), 0.4);
        let strong = apply(&noisy, s, stabiliser(), 1.0, &Params::default()).unwrap();
        let weak = apply(&noisy, s, stabiliser(), 0.0, &Params::default()).unwrap();
        assert!(flat(&strong) < flat(&weak) && flat(&weak) < flat(&noisy), "{} {} {}", flat(&noisy), flat(&weak), flat(&strong));
    }

    #[test]
    fn hostile_input_is_an_error_and_never_a_panic() {
        let ok = vec![0.3f32; 3 * 16 * 16];
        let d = Params::default();
        assert!(apply(&ok, 16, stabiliser(), 0.5, &d).is_ok());
        let mut nan = ok.clone();
        nan[5] = f32::NAN;
        assert!(apply(&nan, 16, stabiliser(), 0.5, &d).is_err());
        let mut inf = ok.clone();
        inf[0] = f32::INFINITY;
        assert!(apply(&inf, 16, stabiliser(), 0.5, &d).is_err());
        assert!(apply(&[], 0, stabiliser(), 0.5, &d).is_err(), "zero size");
        assert!(apply(&ok, 0, stabiliser(), 0.5, &d).is_err());
        assert!(apply(&ok, 17, stabiliser(), 0.5, &d).is_err(), "wrong length");
        assert!(apply(&ok, usize::MAX, stabiliser(), 0.5, &d).is_err(), "huge side");
        assert!(apply(&ok, 16, stabiliser(), f32::NAN, &d).is_err());
        for bad in [f32::NAN, 0.0, -1.0, f32::INFINITY] {
            assert!(apply(&ok, 16, stabiliser(), 0.5, &Params { h: bad, ..d }).is_err(), "h {bad}");
            assert!(apply(&ok, 16, stabiliser(), 0.5, &Params { residual: bad, ..d }).is_err(), "residual {bad}");
        }
        // a stabiliser told nonsense still gives a finite picture
        assert!(apply(&ok, 16, Stabiliser::new(f32::NAN, f32::NAN), 0.5, &d).unwrap().iter().all(|v| v.is_finite()));
        // absurd radii are capped, not obeyed
        let wide = Params { search: usize::MAX, patch: usize::MAX, ..d };
        assert!(apply(&ok, 16, stabiliser(), 0.5, &wide).is_ok());
        assert_eq!(wide.reach(), MAX_SEARCH + MAX_PATCH);
    }

    /// The cost does not follow the photo: a tile of the built-in denoiser needs a fixed, small amount of working
    /// memory, the reach fits inside the overlap two tiles share, and the cap on the search bounds the offsets tried.
    #[test]
    fn the_memory_is_bounded_by_the_tile_and_the_reach_fits_the_overlap() {
        let tile = 2 * crate::builtin::TILE as usize;
        assert!(working_bytes(tile) <= 16 * 1024 * 1024, "{} bytes", working_bytes(tile));
        // a 24 MP photo is cut into tiles by `denoise_bayer`, so what it needs at once is the tile's, not its own
        let photo = 6000usize * 4000;
        assert!(working_bytes(tile) * 4 < photo * 4 * 3, "four tiles at once cost less than one copy of the photo");
        assert_eq!(working_bytes(usize::MAX), usize::MAX, "no overflow");
        assert!(Params::default().reach() <= 2 * crate::builtin::OVERLAP as usize);
        assert!((MAX_SEARCH + MAX_PATCH) <= 2 * crate::builtin::OVERLAP as usize);
    }

    #[test]
    fn the_box_mean_of_a_constant_is_that_constant_and_of_a_spike_is_spread() {
        let s = 12;
        let (mut dst, mut tmp) = (vec![0f32; s * s], vec![0f32; s * s]);
        box_mean(&vec![2.0; s * s], &mut dst, &mut tmp, s, 2);
        assert!(dst.iter().all(|v| (v - 2.0).abs() < 1e-5));
        let mut spike = vec![0f32; s * s];
        spike[6 * s + 6] = 25.0;
        box_mean(&spike, &mut dst, &mut tmp, s, 2);
        assert!((dst[6 * s + 6] - 1.0).abs() < 1e-5 && (dst[4 * s + 4] - 1.0).abs() < 1e-5 && dst[3 * s + 3].abs() < 1e-5);
    }
}
