//! What a hot thing looks like: colour from temperature, by Planck's law.
//!
//! Everything that glows because it is hot — a flame, an ember, molten metal, a
//! filament — emits a spectrum fixed entirely by its temperature. That is why fire
//! reads as fire: the white-yellow-orange-red sequence is not an artistic choice, it
//! is the Planckian locus, and an effect that walks it correctly looks right without
//! anyone tuning a gradient.
//!
//! # The approximation, stated plainly
//!
//! A colorimetrically exact answer means integrating Planck's law against the CIE
//! colour-matching functions and converting through XYZ. This module instead
//! evaluates Planck's law directly at three representative wavelengths — 600 nm,
//! 550 nm and 450 nm — and normalises.
//!
//! That is a real physical calculation rather than a curve fit: the spectral radiance
//! is the actual Planck function with the actual constants. What it gives up is the
//! eye's response curve, so the result is *qualitatively* right at every temperature
//! and not a colour-managed match. For rendering fire that is the correct trade —
//! the sequence and the relative weights are what carry the read, and the cost is
//! three exponentials instead of a numerical integration per particle.
//!
//! # Range
//!
//! Meaningful from roughly 800 K (a dull red ember) to 12,000 K (blue-white). Below
//! 800 K a body emits essentially nothing visible, which the function reports
//! honestly by returning near-black rather than clamping to red.

use crate::thermodynamics::constants::{K_B, PLANCK, WIEN_DISPLACEMENT};
use crate::utils::PhysicsError;

/// Speed of light in vacuum, m/s.
const SPEED_OF_LIGHT: f64 = 2.997_924_58e8;

/// Wavelengths sampled, in metres. Chosen near the peak sensitivity of each of the
/// eye's three cone types, which is what makes three samples enough to place a colour
/// on the Planckian locus.
const RED_NM: f64 = 600e-9;
const GREEN_NM: f64 = 550e-9;
const BLUE_NM: f64 = 450e-9;

/// Below this a body's visible output is negligible however you measure it.
const VISIBLE_FLOOR_K: f64 = 700.0;

/// Spectral radiance of a blackbody at a wavelength, by Planck's law.
///
/// ```text
///   B(lambda, T) = 2hc^2 / lambda^5 / (exp(hc / (lambda kT)) - 1)
/// ```
///
/// Returned in SI units, W/(m^2 sr m). Callers generally want the *ratios* between
/// wavelengths rather than the absolute value, which is what [`blackbody_rgb`]
/// takes.
pub fn spectral_radiance(wavelength: f64, temperature: f64) -> Result<f64, PhysicsError> {
    if wavelength <= 0.0 {
        return Err(PhysicsError::InvalidDistance);
    }
    if temperature <= 0.0 {
        return Err(PhysicsError::CalculationError(
            "temperature must be above absolute zero".to_string(),
        ));
    }

    let hc = PLANCK * SPEED_OF_LIGHT;
    let exponent = hc / (wavelength * K_B * temperature);

    // Guard the exponential: at low temperature and short wavelength this overflows,
    // and the physically correct answer there is simply "no light".
    if exponent > 700.0 {
        return Ok(0.0);
    }

    let numerator = 2.0 * hc * SPEED_OF_LIGHT / wavelength.powi(5);
    let denominator = exponent.exp() - 1.0;
    if denominator <= 0.0 {
        return Ok(0.0);
    }

    Ok(numerator / denominator)
}

/// Wavelength of peak emission, by Wien's displacement law. Metres.
///
/// A useful sanity check on any colour derived here: at 1,500 K the peak is deep in
/// the infrared and only the tail is visible, which is exactly why a wood fire looks
/// orange rather than white.
pub fn peak_wavelength(temperature: f64) -> Result<f64, PhysicsError> {
    if temperature <= 0.0 {
        return Err(PhysicsError::CalculationError(
            "temperature must be above absolute zero".to_string(),
        ));
    }
    Ok(WIEN_DISPLACEMENT / temperature)
}

/// Linear RGB of a blackbody at a given temperature, normalised so the brightest
/// channel is 1.0.
///
/// Normalised rather than absolute because emissive power spans many orders of
/// magnitude across the range a fire covers — a renderer wants the *hue*, and gets
/// brightness from its own falloff. Returns near-black below the visible floor.
///
/// # Example
///
/// ```
/// use rs_physics::thermodynamics::blackbody_rgb;
///
/// let cool = blackbody_rgb(1000.0).unwrap();   // dull red ember
/// let hot  = blackbody_rgb(6500.0).unwrap();   // daylight white
///
/// // A cooler body is proportionally redder.
/// assert!(cool[0] > cool[2]);
/// assert!(hot[2] > cool[2]);
/// ```
pub fn blackbody_rgb(temperature: f64) -> Result<[f64; 3], PhysicsError> {
    if temperature <= 0.0 {
        return Err(PhysicsError::CalculationError(
            "temperature must be above absolute zero".to_string(),
        ));
    }
    if temperature < VISIBLE_FLOOR_K {
        return Ok([0.0, 0.0, 0.0]);
    }

    let r = spectral_radiance(RED_NM, temperature)?;
    let g = spectral_radiance(GREEN_NM, temperature)?;
    let b = spectral_radiance(BLUE_NM, temperature)?;

    let peak = r.max(g).max(b);
    if peak <= 0.0 {
        return Ok([0.0, 0.0, 0.0]);
    }

    Ok([r / peak, g / peak, b / peak])
}

/// Colour of the blue reaction zone at the base of a hydrocarbon flame.
///
/// **Not a blackbody colour, and it cannot be derived from one.** Everything else in
/// this module answers "what colour is a hot thing", where the emission comes from
/// temperature alone. The blue base of a flame is a different mechanism entirely:
/// excited CH and C₂ radicals in the primary reaction zone dropping to lower states
/// and emitting at fixed molecular wavelengths — the CH band near 431 nm and the C₂
/// Swan bands at 473, 516 and 563 nm. It is chemiluminescence, not thermal radiation,
/// so Planck's law has nothing to say about it.
///
/// This is why a flame's colour is not a single gradient. Where fuel meets plenty of
/// air, combustion is complete, little soot forms, and what you see is this blue.
/// Further up, the mixture is fuel-rich, soot particles form, and *those* glow as
/// blackbodies — the familiar orange. A bunsen burner shows both at once, and so does
/// burning gel: blue at the base, orange in the body, black smoke above.
///
/// Returned normalised to the brightest channel, matching [`blackbody_rgb`], so the
/// two can be mixed directly by how much soot has formed.
pub fn swan_band_rgb() -> [f64; 3] {
    // Weighted toward the C₂ green-blue bands, with the CH violet band pulling the
    // blue channel to full. The green content is what stops it reading as a flat
    // cartoon blue — a real reaction zone is closer to cyan than to sky.
    [0.18, 0.52, 1.0]
}

/// How much of a flame's light is soot glowing thermally, rather than radicals
/// emitting in the reaction zone.
///
/// Zero at the base, where combustion is clean and the light is chemiluminescent; one
/// higher up, where the mixture has gone fuel-rich and soot dominates. `along` is the
/// fraction of the way up the visible flame.
///
/// The curve is deliberately fast. The blue zone of a diffusion flame is *thin* — a
/// few centimetres on a burner, a fraction of the height on a pool fire — and a flame
/// that fades gently from blue to orange over its whole length reads as a gas jet
/// rather than as burning liquid.
pub fn soot_fraction(along: f64) -> f64 {
    let t = along.clamp(0.0, 1.0);
    // Rises steeply and saturates: 50% soot by a fifth of the way up.
    (t / (t + 0.2)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The blue base is bluer than *any* blackbody a flame can reach. If this stops
    /// holding, the reaction zone has been folded back into the thermal model and the
    /// distinction the two functions exist to draw has been lost.
    #[test]
    fn the_reaction_zone_is_bluer_than_any_flame_temperature() {
        let swan = swan_band_rgb();
        assert!(swan[2] > swan[0], "the reaction zone should be blue-dominant");

        // A sooting flame tops out well under 2000 K, and is orange all the way.
        for t in [1200.0, 1500.0, 1900.0] {
            let thermal = blackbody_rgb(t).unwrap();
            assert!(
                swan[2] / swan[0].max(1e-12) > thermal[2] / thermal[0].max(1e-12),
                "blackbody at {t} K came out bluer than the reaction zone"
            );
        }
    }

    /// Soot takes over quickly, which is what keeps the blue confined to the base.
    #[test]
    fn soot_takes_over_low_in_the_flame() {
        assert!(soot_fraction(0.0) < 0.01, "the very base should be clean");
        assert!(soot_fraction(0.2) > 0.4, "soot should dominate early");
        assert!(soot_fraction(1.0) > 0.8, "the tip should be sooty");

        // Monotonic: soot never un-forms on the way up.
        let mut previous = -1.0;
        for step in 0..=20 {
            let f = soot_fraction(step as f64 / 20.0);
            assert!(f >= previous, "soot fraction fell at {step}");
            previous = f;
        }
    }

    /// The sequence everyone recognises: cool things are red, hot things are white,

    /// very hot things are blue-white. If this inverts, so does every fire drawn
    /// with it.
    #[test]
    fn colour_walks_the_planckian_locus_from_red_to_blue() {
        // Compared as a blue-to-red *ratio* rather than by the blue channel alone.
        // Output is normalised to the brightest channel, so once a body is hot enough
        // for blue to be brightest it pins at 1.0 and stops being able to rise — the
        // thing that keeps changing above that point is red falling away.
        let blueness = |t: f64| {
            let rgb = blackbody_rgb(t).unwrap();
            rgb[2] / rgb[0].max(1e-12)
        };

        let ember = blueness(900.0);
        let flame = blueness(1600.0);
        let daylight = blueness(6500.0);
        let arc = blueness(11000.0);

        assert!(ember < flame, "ember bluer than flame: {ember} vs {flame}");
        assert!(flame < daylight, "flame bluer than daylight: {flame} vs {daylight}");
        assert!(daylight < arc, "daylight bluer than an arc: {daylight} vs {arc}");

        let ember_rgb = blackbody_rgb(900.0).unwrap();
        assert!(ember_rgb[0] > ember_rgb[1], "an ember should be red-dominant");

    }

    #[test]
    fn output_is_normalised_to_the_brightest_channel() {
        for t in [800.0, 1500.0, 3000.0, 6500.0, 10000.0] {
            let rgb = blackbody_rgb(t).unwrap();
            let peak = rgb[0].max(rgb[1]).max(rgb[2]);
            assert!(
                (peak - 1.0).abs() < 1e-9,
                "at {t} K the brightest channel was {peak}"
            );
            assert!(rgb.iter().all(|c| (0.0..=1.0).contains(c)));
        }
    }

    /// Below the visible floor the honest answer is "no light", not "clamped red".
    #[test]
    fn a_cold_body_emits_nothing_visible() {
        assert_eq!(blackbody_rgb(300.0).unwrap(), [0.0, 0.0, 0.0]);
    }

    /// Wien's law is a cheap independent check that the Planck evaluation is not
    /// nonsense: the sun's peak should land in visible green.
    #[test]
    fn wien_puts_the_solar_peak_in_the_visible_band() {
        let peak = peak_wavelength(5772.0).unwrap();
        assert!(
            (450e-9..600e-9).contains(&peak),
            "solar peak at {} nm",
            peak * 1e9
        );
    }

    /// And a flame's peak is in the infrared, which is why fire looks orange rather
    /// than white despite being extremely hot by everyday standards.
    #[test]
    fn a_flame_peaks_in_the_infrared() {
        let peak = peak_wavelength(1500.0).unwrap();
        assert!(peak > 700e-9, "flame peak at {} nm is visible", peak * 1e9);
    }

    #[test]
    fn radiance_rises_with_temperature_at_every_wavelength() {
        for &lambda in &[RED_NM, GREEN_NM, BLUE_NM] {
            let cool = spectral_radiance(lambda, 1000.0).unwrap();
            let hot = spectral_radiance(lambda, 3000.0).unwrap();
            assert!(hot > cool, "radiance fell with temperature at {lambda} m");
        }
    }

    #[test]
    fn bad_inputs_are_rejected_rather_than_producing_nan() {
        assert!(blackbody_rgb(0.0).is_err());
        assert!(blackbody_rgb(-5.0).is_err());
        assert!(peak_wavelength(0.0).is_err());
        assert!(spectral_radiance(0.0, 1000.0).is_err());
        assert!(spectral_radiance(500e-9, 0.0).is_err());
    }
}
