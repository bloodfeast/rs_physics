//! From per-band losses to what a voice can apply: one gain and one low-pass.
//!
//! # Why a fit
//!
//! The propagation laws answer per frequency band: air, a barrier and foliage each take a
//! number of decibels out of each band. A voice does not carry four bands; it carries a
//! gain and a one-pole low-pass. So the four losses are fitted to the response that voice
//! can make.
//!
//! A one-pole low-pass with a broadband gain `g` has the **power** response
//! `T(f) = g / (1 + (f / fc)^2)`. That is nonlinear in `fc` in decibels, but its reciprocal
//! is a straight line in `f^2`: `1 / T = a + b f^2`, with `g = 1 / a` and
//! `fc = sqrt(a / b)`. Least squares on that line over the bands is a 2x2 closed form, so
//! the fit has no iteration and no starting guess. (Fitting the cutoff alone cannot express
//! a barrier, which takes about 5 dB even at 125 Hz: that part is broadband and belongs in
//! the gain.)
//!
//! Frequencies enter as `(f / FIT_REFERENCE_HZ)^2` so the sums stay well scaled in f32 on
//! the GPU, which evaluates the same fit.
//!
//! # Examples
//!
//! ```
//! use rs_physics::acoustics::band::{fit_lowpass, NO_LOWPASS_HZ};
//!
//! let bands = [125.0, 500.0, 2_000.0, 8_000.0];
//! // No loss anywhere: unity gain and no filter.
//! let clear = fit_lowpass(&bands, &[0.0; 4]);
//! assert!((clear.gain - 1.0).abs() < 1e-12);
//! assert_eq!(clear.cutoff_hz, NO_LOWPASS_HZ);
//! // A loss that climbs with frequency is a low-pass.
//! let dull = fit_lowpass(&bands, &[0.1, 0.5, 3.0, 12.0]);
//! assert!(dull.cutoff_hz < 8_000.0 && dull.cutoff_hz > 1_000.0);
//! ```

/// The frequency the fit's abscissa is normalised by, in hertz. It only scales the sums
/// (the fitted gain and cutoff do not depend on it); 1 kHz keeps the four octave-spaced
/// bands' squares between about 0.016 and 64.
pub const FIT_REFERENCE_HZ: f64 = 1_000.0;

/// The cutoff reported when there is no low-pass to speak of, in hertz: the top of
/// hearing. A fitted cutoff above it is reported as it.
pub const NO_LOWPASS_HZ: f64 = 20_000.0;

/// The result of [`fit_lowpass`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LowPass {
    /// Broadband **amplitude** gain, in `0..=1`: `sqrt(g)` of the power fit, because the
    /// voice multiplies samples (pressure), not power.
    pub gain: f64,
    /// Cutoff of the one-pole low-pass, in hertz, in `(0, NO_LOWPASS_HZ]`.
    pub cutoff_hz: f64,
}

/// Fit per-band losses to a broadband gain and a one-pole low-pass.
///
/// Closed-form least squares of `1 / T_b = a + b x_b` with `T_b = 10^(-L_b / 10)` and
/// `x_b = (f_b / FIT_REFERENCE_HZ)^2`. Then:
///
/// * `a` is held at 1 or above: a passive path never transmits more than it received, so
///   a fit whose intercept would imply `g > 1` is read as `g = 1`;
/// * the power gain is `g = 1 / a` and the amplitude gain `sqrt(g)`;
/// * the cutoff is `FIT_REFERENCE_HZ x sqrt(a / b)` when `b > 0`, capped at
///   [`NO_LOWPASS_HZ`]; with `b <= 0` the loss does not climb with frequency and there is
///   no low-pass. The cap makes the cutoff continuous as `b` falls through zero.
///
/// # Arguments
///
/// * `bands_hz` - the four band centres, in hertz; at least two must differ.
/// * `loss_db` - the loss in each band, in decibels (non-negative for a passive path).
///
/// # Returns
///
/// The [`LowPass`].
///
/// # Examples
///
/// ```
/// use rs_physics::acoustics::band::fit_lowpass;
///
/// // A flat 6 dB loss is all gain and no filter.
/// let flat = fit_lowpass(&[125.0, 500.0, 2_000.0, 8_000.0], &[6.0; 4]);
/// assert!((flat.gain - 10f64.powf(-6.0 / 20.0)).abs() < 1e-12);
/// assert_eq!(flat.cutoff_hz, 20_000.0);
/// ```
pub fn fit_lowpass(bands_hz: &[f64; 4], loss_db: &[f64; 4]) -> LowPass {
    let x = bands_hz.map(|f| (f / FIT_REFERENCE_HZ) * (f / FIT_REFERENCE_HZ));
    let y = loss_db.map(|l| 10f64.powf(l / 10.0));
    let (x_mean, y_mean) = (x.iter().sum::<f64>() / 4.0, y.iter().sum::<f64>() / 4.0);
    let sxx: f64 = x.iter().map(|xi| (xi - x_mean) * (xi - x_mean)).sum();
    let b = x
        .iter()
        .zip(&y)
        .map(|(xi, yi)| (xi - x_mean) * yi)
        .sum::<f64>()
        / sxx;
    let a = (y_mean - b * x_mean).max(1.0);
    let cutoff_hz = if b > 0.0 {
        (FIT_REFERENCE_HZ * (a / b).sqrt()).min(NO_LOWPASS_HZ)
    } else {
        NO_LOWPASS_HZ
    };
    LowPass {
        gain: (1.0 / a).sqrt(),
        cutoff_hz,
    }
}

/// Piecewise-linear lookup of a cutoff ceiling against distance.
///
/// The curve is `(metres, hertz)` points in ascending distance. Before the first point the
/// first value holds, after the last the last value holds, and between two points the
/// cutoff is interpolated linearly in both. An empty curve imposes nothing.
///
/// # Arguments
///
/// * `curve` - `(distance_m, cutoff_hz)` points, ascending in distance.
/// * `distance_m` - where to read it, in metres.
///
/// # Returns
///
/// The ceiling in hertz, or `None` for an empty curve.
///
/// # Examples
///
/// ```
/// use rs_physics::acoustics::band::cutoff_ceiling;
///
/// let ladder = [(0.0, 20_000.0), (45.0, 1_400.0), (130.0, 480.0)];
/// assert_eq!(cutoff_ceiling(&ladder, 45.0), Some(1_400.0));
/// assert_eq!(cutoff_ceiling(&ladder, 500.0), Some(480.0));
/// assert_eq!(cutoff_ceiling(&[], 10.0), None);
/// ```
pub fn cutoff_ceiling(curve: &[(f64, f64)], distance_m: f64) -> Option<f64> {
    let (first, last) = (curve.first()?, curve.last()?);
    if distance_m <= first.0 {
        return Some(first.1);
    }
    if distance_m >= last.0 {
        return Some(last.1);
    }
    for pair in curve.windows(2) {
        let ((m0, h0), (m1, h1)) = (pair[0], pair[1]);
        if distance_m <= m1 {
            let span = m1 - m0;
            let t = if span > 0.0 {
                (distance_m - m0) / span
            } else {
                1.0
            };
            return Some(h0 + (h1 - h0) * t);
        }
    }
    Some(last.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BANDS: [f64; 4] = [125.0, 500.0, 2_000.0, 8_000.0];

    /// A loss that *is* a one-pole low-pass times a gain is fitted back exactly: the fit is
    /// linear in `1 / T`, so an exact model gives a zero residual.
    #[test]
    fn an_exact_one_pole_is_recovered() {
        for (g, fc) in [(1.0, 3_000.0), (0.25, 900.0), (0.5, 12_000.0)] {
            let loss = BANDS.map(|f| -10.0 * (g / (1.0 + (f / fc) * (f / fc))).log10());
            let fit = fit_lowpass(&BANDS, &loss);
            assert!(
                (fit.gain - g.sqrt()).abs() < 1e-9,
                "gain {} for {g}",
                fit.gain
            );
            assert!(
                (fit.cutoff_hz - fc).abs() < 1e-6 * fc,
                "cutoff {} for {fc}",
                fit.cutoff_hz
            );
        }
    }

    /// The cap keeps the cutoff continuous as the slope falls through zero.
    #[test]
    fn the_cutoff_is_continuous_through_a_flat_loss() {
        let tiny = fit_lowpass(&BANDS, &[3.0, 3.0, 3.0, 3.0 + 1e-9]);
        let flat = fit_lowpass(&BANDS, &[3.0; 4]);
        assert_eq!(tiny.cutoff_hz, NO_LOWPASS_HZ);
        assert_eq!(flat.cutoff_hz, NO_LOWPASS_HZ);
        assert!((tiny.gain - flat.gain).abs() < 1e-9);
    }

    /// A passive path never gains: an intercept under one is held at one.
    #[test]
    fn a_fit_never_amplifies() {
        // Steeply convex in f^2, which drags the intercept under one.
        let fit = fit_lowpass(&BANDS, &[0.0, 0.0, 0.0, 40.0]);
        assert!(fit.gain <= 1.0);
    }
}
