//! Where a listener hears something from.
//!
//! # Two ears, two cues, and neither of them is a pan knob
//!
//! Stereo positioning in games is usually a gain difference: the sound is a bit louder on
//! the side it came from. That is one of the two cues a head actually uses, and it is the
//! weaker one at the frequencies most game sounds live at.
//!
//! The stronger cue is **time**. Sound reaches the near ear before the far one, by up to
//! about 0.7 ms, and the brain resolves that difference to a couple of degrees. It is also
//! the cue that survives everything: it does not care how loud the sound is, it works at
//! low frequencies where the head casts no shadow at all, and it is what makes a source
//! feel *placed* rather than merely panned.
//!
//! So this returns both, computed from geometry and the size of a head:
//!
//! * **Interaural time difference** — from the extra distance around the skull. Woodworth's
//!   relation, which is the path length of a wave diffracting round a sphere.
//! * **Interaural level difference** — the head shadow, which is real only when the head is
//!   large compared with the wavelength. Below roughly 700 Hz a head is acoustically
//!   invisible and the level difference genuinely is nearly zero; a pan knob that keeps
//!   panning down there is inventing a cue that does not exist.
//!
//! # It does not synthesise anything
//!
//! No HRTF, no filtering, no binaural rendering. This says *how much later and how much
//! quieter at each ear*, and whatever owns the audio decides what to do with it. A
//! stereo mixer can use the level difference and ignore the timing; something better can
//! use both.

use crate::acoustics::Air;

/// Radius of an average adult head, in metres.
///
/// The one anthropometric constant in here. Everything else is geometry.
pub const HEAD_RADIUS_M: f64 = 0.0875;

/// Frequency below which a head casts no useful shadow, in hertz.
///
/// Where the wavelength is roughly the head's circumference. Under it, sound diffracts
/// round cleanly and both ears hear the same level — which is why timing is the only
/// low-frequency cue and why a distant explosion is hard to place by loudness alone.
pub const SHADOW_ONSET_HZ: f64 = 700.0;

/// A listener: where they are and which way they face.
///
/// `forward` and `right` must be unit vectors and perpendicular. They are supplied rather
/// than derived from an angle so a caller with a camera basis can hand it over directly
/// and nothing has to agree about which way zero points.
#[derive(Debug, Clone, Copy)]
pub struct Ears {
    pub position: [f64; 3],
    pub forward: [f64; 3],
    pub right: [f64; 3],
}

/// What a listener hears of one source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Heard {
    /// Straight-line distance, in metres.
    pub distance: f64,
    /// Where it is, relative to facing: −π to π, positive to the right, 0 straight ahead.
    pub azimuth: f64,
    /// Above or below the ear plane, −π/2 to π/2.
    pub elevation: f64,
    /// Seconds by which the far ear lags the near one. Always positive; `azimuth` says
    /// which ear is which.
    pub interaural_delay: f64,
    /// Linear amplitude at the left ear, from the head shadow alone, 0 to 1.
    pub left_gain: f64,
    /// And at the right.
    pub right_gain: f64,
}

impl Ears {
    /// A listener at the origin facing down −Z, which is the usual convention for a
    /// camera looking into the scene.
    pub fn at(position: [f64; 3]) -> Ears {
        Ears {
            position,
            forward: [0.0, 0.0, -1.0],
            right: [1.0, 0.0, 0.0],
        }
    }

    /// Resolve a source position into what the two ears receive.
    pub fn hear(&self, source: [f64; 3], air: &Air, frequency_hz: f64) -> Heard {
        let to = [
            source[0] - self.position[0],
            source[1] - self.position[1],
            source[2] - self.position[2],
        ];
        let distance = (to[0] * to[0] + to[1] * to[1] + to[2] * to[2]).sqrt();
        if distance < 1e-9 {
            return Heard {
                distance: 0.0,
                azimuth: 0.0,
                elevation: 0.0,
                interaural_delay: 0.0,
                left_gain: 1.0,
                right_gain: 1.0,
            };
        }

        let unit = [to[0] / distance, to[1] / distance, to[2] / distance];
        let ahead = dot(unit, self.forward);
        let beside = dot(unit, self.right);
        // Up is right × forward, which keeps the basis self-consistent rather than
        // assuming Y. The other order points at the floor - worth stating, because the
        // first version had it and "overhead" came out as directly below.
        let up = cross(self.right, self.forward);
        let above = dot(unit, up);

        let azimuth = beside.atan2(ahead);
        let elevation = above.clamp(-1.0, 1.0).asin();

        // **Woodworth.** The extra distance to the far ear is the straight run across the
        // head plus the arc the wave bends around it, which for an angle θ off centre is
        // `r(θ + sin θ)`. It peaks at the sides and vanishes dead ahead and behind, which
        // is also why front and back are famously hard to tell apart by timing alone.
        let lateral = beside.clamp(-1.0, 1.0).asin().abs();
        let interaural_delay =
            HEAD_RADIUS_M * (lateral + lateral.sin()) / air.speed_of_sound();

        // **The shadow, and only where there is one.**
        //
        // A head shadows sound the way any obstacle does: strongly when it is large
        // compared with the wavelength, not at all when it is small. That is a high-pass
        // in shape, not a ramp - `(f/f0)² / (1 + (f/f0)²)` - and the difference matters at
        // the bottom end. A linear ramp still panned an 80 Hz source by sixteen per cent,
        // which is a cue a real head does not provide and an ear cannot use; this leaves
        // it at about one per cent and lets the timing carry it, which is what actually
        // happens.
        let ratio = frequency_hz.max(0.0) / SHADOW_ONSET_HZ;
        let shadow_strength = ratio * ratio / (1.0 + ratio * ratio);
        // At most about 20 dB at the far ear, at high frequency, directly to one side.
        let far_ear = 1.0 - 0.9 * shadow_strength * beside.abs();
        let (left_gain, right_gain) = if beside >= 0.0 {
            (far_ear.max(0.0), 1.0)
        } else {
            (1.0, far_ear.max(0.0))
        };

        Heard {
            distance,
            azimuth,
            elevation,
            interaural_delay,
            left_gain,
            right_gain,
        }
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ears() -> Ears {
        Ears::at([0.0, 0.0, 0.0])
    }

    /// Straight ahead is centred, and both ears get it at once.
    #[test]
    fn a_source_in_front_is_centred() {
        let h = ears().hear([0.0, 0.0, -10.0], &Air::standard(), 1_000.0);
        assert!(h.azimuth.abs() < 1e-9, "azimuth {} was not zero", h.azimuth);
        assert!(h.interaural_delay < 1e-9, "a centred source arrived at the ears late");
        assert!((h.left_gain - h.right_gain).abs() < 1e-9, "a centred source was panned");
    }

    /// To the right is to the right, in both cues.
    #[test]
    fn a_source_to_the_side_leads_in_the_near_ear() {
        let h = ears().hear([10.0, 0.0, 0.0], &Air::standard(), 4_000.0);
        assert!(h.azimuth > 1.5, "a source dead right had azimuth {}", h.azimuth);
        assert!(h.right_gain > h.left_gain, "the near ear was not the louder one");
        assert!(h.interaural_delay > 0.0);
    }

    /// **The published figure.** Maximum interaural delay for an adult head is about
    /// 0.65 ms, and it happens at the sides.
    #[test]
    fn the_interaural_delay_peaks_where_it_should_and_at_the_right_size() {
        let air = Air::standard();
        let side = ears().hear([10.0, 0.0, 0.0], &air, 1_000.0).interaural_delay;
        let front = ears().hear([0.0, 0.0, -10.0], &air, 1_000.0).interaural_delay;

        assert!(
            (0.0005..0.0009).contains(&side),
            "maximum interaural delay came out {:.4} ms against a published ~0.65",
            side * 1000.0,
        );
        assert!(side > front, "the delay did not peak at the side");
    }

    /// **Bass cannot be placed by loudness, and the model knows it.** A head is
    /// acoustically invisible below a few hundred hertz, so the level difference there is
    /// nearly nothing — while the timing cue is undiminished.
    #[test]
    fn a_head_casts_no_shadow_at_low_frequencies() {
        let air = Air::standard();
        let low = ears().hear([10.0, 0.0, 0.0], &air, 80.0);
        let high = ears().hear([10.0, 0.0, 0.0], &air, 8_000.0);

        let low_difference = low.right_gain - low.left_gain;
        let high_difference = high.right_gain - high.left_gain;
        assert!(
            low_difference < 0.15,
            "an 80 Hz source was panned by {low_difference:.2} - a head does not do that",
        );
        assert!(
            high_difference > low_difference * 3.0,
            "8 kHz was shadowed by {high_difference:.2} against 80 Hz at \
             {low_difference:.2} - not the frequency dependence a head has",
        );
        // And timing does not care about frequency at all.
        assert!((low.interaural_delay - high.interaural_delay).abs() < 1e-12);
    }

    /// Elevation comes out of the basis rather than assuming which axis is up.
    #[test]
    fn overhead_is_overhead() {
        let h = ears().hear([0.0, 10.0, 0.0], &Air::standard(), 1_000.0);
        assert!(
            (h.elevation - std::f64::consts::FRAC_PI_2).abs() < 1e-6,
            "a source directly overhead had elevation {}",
            h.elevation,
        );
    }

    /// A source in the listener's own head does not divide by zero.
    #[test]
    fn a_source_at_the_listener_is_finite() {
        let h = ears().hear([0.0, 0.0, 0.0], &Air::standard(), 1_000.0);
        assert_eq!(h.distance, 0.0);
        assert!(h.left_gain.is_finite() && h.right_gain.is_finite());
    }

    /// Turning the listener moves the world, not the other way round.
    #[test]
    fn facing_the_source_centres_it() {
        let air = Air::standard();
        let facing_right = Ears {
            position: [0.0, 0.0, 0.0],
            forward: [1.0, 0.0, 0.0],
            right: [0.0, 0.0, 1.0],
        };
        let h = facing_right.hear([10.0, 0.0, 0.0], &air, 2_000.0);
        assert!(
            h.azimuth.abs() < 1e-9,
            "a source the listener is facing had azimuth {}",
            h.azimuth,
        );
    }
}
