//! # Acoustics
//!
//! Sound as a physical quantity: how fast it travels, what the air takes out of it on the
//! way, what a surface does to it, and where a listener hears it from.
//!
//! ## Why this exists
//!
//! Games author distance. The usual shape is two or three hand-made variants of a sound —
//! a "near" one, a "far" one — with a chosen low-pass and a chosen delay, tuned until it
//! feels right. That works, and it is a lookup table standing in for a calculation: the
//! numbers describe one designer's idea of one distance on one map, and nothing about
//! them responds to the world. A shot fired in a canyon and the same shot fired on open
//! ground come out identical, because the table has no idea where either happened.
//!
//! Every quantity involved is computable and has been for decades. Air absorption is
//! standardised (ISO 9613-1) and depends on frequency, temperature and humidity. A
//! reflection's strength follows from the acoustic impedance of what it bounced off, and
//! impedance follows from density and stiffness — which a [`crate::materials::Material`]
//! already carries. A wall between source and listener attenuates by an amount that
//! follows from the extra distance sound must travel around it. Direction is geometry.
//!
//! So this module computes them, and the caller supplies the world.
//!
//! ## What it does not do
//!
//! It does not touch samples. There is no filter, no convolution, no mixer here: it
//! answers *how much, how late, how dull, from where*, and whatever owns the audio
//! applies that. Keeping the physics separate from the DSP means the physics can be tested
//! against published figures rather than against how something sounds.
//!
//! ## The pieces
//!
//! * [`Air`] — the medium. Speed of sound, and absorption per metre at a frequency.
//! * [`spreading_gain`], [`delay`] — the two things distance does regardless of medium.
//! * [`doppler_ratio`] — motion.
//! * [`surfaces`] — what a material does to sound that hits it, and what a barrier does
//!   to sound that has to get around it.
//! * [`spatial`] — where a listener hears it from, and what their head does to it.
//!
//! ## Example
//!
//! ```
//! use rs_physics::acoustics::{Air, spreading_gain, delay};
//!
//! let air = Air::standard();
//! // A rifle shot a hundred metres away.
//! let gain = spreading_gain(100.0, 1.0);
//! let late = delay(100.0, &air);
//! assert!(late > 0.28 && late < 0.30);          // about a third of a second
//!
//! // The air takes far more out of the crack than the thump.
//! let high = air.absorption_db_per_m(8_000.0);
//! let low = air.absorption_db_per_m(250.0);
//! assert!(high > low * 10.0);
//! ```

pub mod band;
pub mod spatial;
pub mod surfaces;

pub use spatial::{Ears, Heard};
pub use surfaces::{
    barrier_insertion_db, fresnel_radius, impedance, lit_zone_limit, occludes,
    reflection_coefficient,
};

/// Reference distance for [`spreading_gain`], in metres.
///
/// A sound's level is quoted *at* some distance; one metre is the convention. Without a
/// reference the inverse-square law has no scale and a source at half a metre is twice as
/// loud as it can possibly be.
pub const REFERENCE_M: f64 = 1.0;

pub use crate::atmosphere::Air;

/// Reference pressure for the absorption model, in pascals. One standard atmosphere.
pub const REFERENCE_PRESSURE: f64 = crate::atmosphere::STANDARD_PRESSURE;
/// Reference temperature for the absorption model, in kelvin. 20 °C.
pub const REFERENCE_TEMPERATURE: f64 = crate::atmosphere::REFERENCE_TEMPERATURE;

/// The acoustics of the medium.
///
/// [`Air`] itself lives in [`crate::atmosphere`], because what air *is* — its temperature,
/// its humidity, its pressure, and therefore its density and its viscosity — is not an
/// acoustics question. It used to be declared here, and the consequence was that the crate
/// held two irreconcilable descriptions of air: this one, which knew about the weather, and
/// `Fluid::air()`, whose density was frozen at 1.225 kg/m³ whatever the weather was. A
/// caller could hold both at once, in the same frame, and nothing would object.
///
/// What stays here is the physics that is genuinely acoustic: how fast sound crosses the
/// medium, and how much of it the medium eats on the way.
impl Air {
    /// Speed of sound, in metres per second.
    ///
    /// From the ideal-gas relation `c = sqrt(γ R T / M)`, which for air reduces to
    /// `20.05 * sqrt(T)`. Humidity raises it slightly — water is lighter than the nitrogen
    /// it displaces — by about 0.3 m/s at full saturation and room temperature, which is
    /// under a tenth of a per cent and is included because it costs one term.
    ///
    /// There is no guard on the temperature here any more. [`Air::new`] will not build a
    /// state at or below 0 K, so the square root cannot see a negative and the `max(1.0)`
    /// that used to stand in this line is gone. That is the whole argument for making the
    /// fields private: the check moved from three use sites to one constructor, and the
    /// use sites became total.
    pub fn speed_of_sound(&self) -> f64 {
        let dry = 20.05 * self.temperature().sqrt();
        dry * (1.0 + 0.0016 * self.water_vapour_percent())
    }

    /// Mole fraction of water vapour as a **percentage**, which is the unit ISO 9613-1
    /// writes its absorption formula in.
    ///
    /// The crate's public form is [`Air::water_vapour_mole_fraction`], a dimensionless
    /// fraction, per the SI convention everything else follows. This is the one place the
    /// factor of 100 exists, and it sits beside the two formulas that need it rather than
    /// in a public signature where a caller could pick the wrong one.
    #[inline]
    fn water_vapour_percent(&self) -> f64 {
        100.0 * self.water_vapour_mole_fraction()
    }

    /// Atmospheric absorption at a frequency, in decibels per metre.
    ///
    /// # Where the dullness of distance comes from
    ///
    /// This is the term that makes a far-off sound *dark* rather than merely quiet, and it
    /// is the one a hand-tuned low-pass is standing in for. It is not a smooth roll-off:
    /// air absorbs through two relaxation mechanisms, oxygen and nitrogen, each with its
    /// own resonance whose frequency moves with humidity. That is why the shape of a
    /// distant sound changes with the weather and not just its brightness.
    ///
    /// ISO 9613-1. Returns dB per metre; multiply by path length.
    pub fn absorption_db_per_m(&self, frequency_hz: f64) -> f64 {
        if frequency_hz <= 0.0 {
            return 0.0;
        }
        let f = frequency_hz;
        let t = self.temperature();
        let p_ratio = self.pressure() / REFERENCE_PRESSURE;
        let t_ratio = t / REFERENCE_TEMPERATURE;
        let h = self.water_vapour_percent();

        // Relaxation frequency of oxygen. Water vapour is the catalyst, so this climbs
        // steeply with humidity — which is why damp air absorbs *less* at speech
        // frequencies than dry air does.
        let f_ro = p_ratio * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h));

        // And of nitrogen, which relaxes far lower and carries the low end.
        let f_rn = p_ratio
            * t_ratio.powf(-0.5)
            * (9.0 + 280.0 * h * (-4.170 * (t_ratio.powf(-1.0 / 3.0) - 1.0)).exp());

        let f2 = f * f;
        let classical = 1.84e-11 * p_ratio.recip() * t_ratio.sqrt();
        let oxygen = 0.01275 * (-2239.1 / t).exp() / (f_ro + f2 / f_ro);
        let nitrogen = 0.1068 * (-3352.0 / t).exp() / (f_rn + f2 / f_rn);

        // Nepers per metre out of the standard form, then to decibels.
        8.686 * f2 * (classical + t_ratio.powf(-2.5) * (oxygen + nitrogen))
    }

    /// Absorption over a path, as a linear amplitude factor in 0..=1.
    ///
    /// The form a mixer wants: multiply the sound by this. Decibels are for reading.
    pub fn absorption_gain(&self, frequency_hz: f64, metres: f64) -> f64 {
        let db = self.absorption_db_per_m(frequency_hz) * metres.max(0.0);
        10f64.powf(-db / 20.0)
    }
}

/// Amplitude falling off with distance, relative to [`REFERENCE_M`].
///
/// Inverse *distance*, not inverse square: intensity falls as `1/r²` and pressure — which
/// is what a sample is — falls as `1/r`. Squaring it here is the single most common way to
/// make a game sound wrong, because everything vanishes twice as fast as it should.
///
/// `directivity` is the source's gain in the listener's direction, 1.0 for a point source
/// that radiates evenly. A muzzle is not one of those.
pub fn spreading_gain(metres: f64, directivity: f64) -> f64 {
    let r = metres.max(REFERENCE_M);
    directivity.max(0.0) * REFERENCE_M / r
}

/// How long sound takes to arrive, in seconds.
///
/// The reason a distant explosion is seen before it is heard, and — applied per
/// reflection — the reason a canyon sounds like a canyon.
pub fn delay(metres: f64, air: &Air) -> f64 {
    metres.max(0.0) / air.speed_of_sound()
}

/// Frequency ratio from relative motion, along the source-to-listener line.
///
/// Positive velocities are *toward* the other party. Returns the factor to multiply the
/// source frequency by; 1.0 is no shift.
///
/// Clamped below the speed of sound: at or past it the classical expression diverges, and
/// a shell that outruns its own report is a different phenomenon than a Doppler shift.
pub fn doppler_ratio(source_toward: f64, listener_toward: f64, air: &Air) -> f64 {
    let c = air.speed_of_sound();
    let closing = source_toward.clamp(-0.95 * c, 0.95 * c);
    (c + listener_toward) / (c - closing)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Published figure: 343 m/s at 20 °C.
    #[test]
    fn the_speed_of_sound_matches_the_textbook() {
        let c = Air::standard().speed_of_sound();
        assert!(
            (c - 343.0).abs() < 2.0,
            "speed of sound came out {c:.1} m/s at 20 C, against a textbook 343",
        );
    }

    /// And it rises with temperature, which is why sound carries differently in winter.
    #[test]
    fn sound_travels_faster_in_warm_air() {
        assert!(Air::standard().speed_of_sound() > Air::winter().speed_of_sound());
    }

    /// **The whole reason distance sounds dull.** Absorption climbs steeply with
    /// frequency — roughly as its square — so the top of a sound is gone long before the
    /// bottom of it is.
    #[test]
    fn air_eats_the_top_of_a_sound_first() {
        let air = Air::standard();
        let low = air.absorption_db_per_m(250.0);
        let mid = air.absorption_db_per_m(2_000.0);
        let high = air.absorption_db_per_m(8_000.0);

        assert!(low < mid && mid < high, "absorption is not rising with frequency");
        assert!(
            high > low * 20.0,
            "8 kHz is absorbed at {high:.4} dB/m against 250 Hz at {low:.4} - not nearly \
             the separation that makes a far shot a thump",
        );
    }

    /// Against ISO 9613-1's own published table: at 20 C and 70% humidity, 1 kHz absorbs
    /// about 5 dB per kilometre and 4 kHz about 24.
    #[test]
    fn absorption_matches_the_standards_table() {
        let air = Air::new(293.15, 0.70, REFERENCE_PRESSURE).unwrap();
        let at_1k = air.absorption_db_per_m(1_000.0) * 1_000.0;
        let at_4k = air.absorption_db_per_m(4_000.0) * 1_000.0;

        assert!(
            (at_1k - 5.0).abs() < 2.0,
            "1 kHz came out {at_1k:.1} dB/km against a published ~5",
        );
        assert!(
            (at_4k - 24.0).abs() < 8.0,
            "4 kHz came out {at_4k:.1} dB/km against a published ~24",
        );
    }

    /// Damp air is *kinder* to mid frequencies than dry air, which is the opposite of
    /// what most people expect and is why it is worth having the real model.
    #[test]
    fn dry_air_absorbs_more_than_damp_air_at_speech_frequencies() {
        let base = Air::standard();
        let dry = base.with_humidity(0.05).unwrap();
        let damp = base.with_humidity(0.80).unwrap();
        assert!(
            dry.absorption_db_per_m(2_000.0) > damp.absorption_db_per_m(2_000.0),
            "dry air did not absorb more than damp air at 2 kHz",
        );
    }

    /// Pressure, not intensity. Halving the distance doubles the amplitude.
    #[test]
    fn amplitude_falls_as_one_over_distance() {
        let near = spreading_gain(10.0, 1.0);
        let far = spreading_gain(20.0, 1.0);
        assert!(
            (near / far - 2.0).abs() < 1e-9,
            "doubling the distance changed amplitude by {:.3}x, not 2x - this is the \
             inverse-square mistake",
            near / far,
        );
    }

    /// Inside the reference distance nothing gets louder than the reference.
    #[test]
    fn a_source_on_top_of_you_does_not_divide_by_zero() {
        assert!(spreading_gain(0.0, 1.0).is_finite());
        assert_eq!(spreading_gain(0.0, 1.0), spreading_gain(REFERENCE_M, 1.0));
    }

    /// A hundred metres is about a third of a second, which is long enough to see the
    /// flash first.
    #[test]
    fn sound_arrives_late() {
        let late = delay(100.0, &Air::standard());
        assert!((late - 0.2915).abs() < 0.01, "100 m took {late:.3} s");
    }

    /// Approaching raises the pitch, receding lowers it, and standing still does nothing.
    #[test]
    fn doppler_runs_the_right_way() {
        let air = Air::standard();
        assert!(doppler_ratio(30.0, 0.0, &air) > 1.0, "approaching did not raise pitch");
        assert!(doppler_ratio(-30.0, 0.0, &air) < 1.0, "receding did not lower pitch");
        assert!((doppler_ratio(0.0, 0.0, &air) - 1.0).abs() < 1e-12);
    }

    /// And it does not blow up at the speed of sound.
    #[test]
    fn doppler_is_finite_at_mach_one() {
        let air = Air::standard();
        let c = air.speed_of_sound();
        assert!(doppler_ratio(c, 0.0, &air).is_finite());
        assert!(doppler_ratio(c * 4.0, 0.0, &air).is_finite());
    }
}
