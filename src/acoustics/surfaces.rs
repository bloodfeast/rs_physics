//! What a surface does to sound: reflect it, absorb it, or stand in its way.
//!
//! # Reflections are not a chosen number
//!
//! The usual approach gives each surface a "reflectivity" between zero and one, picked by
//! ear. But how much sound bounces off something follows from a property the material
//! already has: its **acoustic impedance**, the product of its density and the speed of
//! sound within it — and the speed of sound within a solid follows from its stiffness and
//! its density. So a [`Material`] that knows its density and Young's modulus already knows
//! how loud its echo is, and asking it is both less work and impossible to get
//! inconsistent with the rest of the physics.
//!
//! The consequence is one nobody has to author: concrete rings and rubber does not, and
//! the reason is that concrete's impedance is enormously mismatched to air's while
//! rubber's is merely large. That falls out of two numbers that were in the material for
//! entirely unrelated reasons.
//!
//! # And a barrier is a path length
//!
//! Sound does not stop at a wall, it goes round it, arriving late and weakened by an
//! amount that depends on how much further it had to travel measured in wavelengths.
//! That is Maekawa's relation, it has been checked against measurement since 1968, and it
//! explains without any extra rule why a wall muffles a distant rumble far less than a
//! nearby crack: the rumble's wavelength is longer than the wall.

use crate::materials::Material;

/// Characteristic acoustic impedance of air at 20 °C, in rayl (Pa·s/m).
///
/// Density times speed of sound: `1.204 × 343`. Everything reflects off everything else
/// in proportion to how far it is from this number.
pub const AIR_IMPEDANCE: f64 = 413.0;

/// Characteristic acoustic impedance of a material, in rayl.
///
/// `Z = ρc`, where the speed of sound in a solid is `sqrt(E/ρ)` — so `Z = sqrt(Eρ)`, and a
/// material needs to know nothing it did not already know.
///
/// This is the *bulk* longitudinal impedance, which is the right quantity for a thick
/// slab of something. A thin panel that flexes is a different problem and this will
/// overstate it.
pub fn impedance(material: &Material) -> f64 {
    (material.youngs_modulus.max(0.0) * material.density.max(0.0)).sqrt()
}

/// Fraction of *pressure* reflected at an air-to-material boundary, 0 to 1.
///
/// From the impedance mismatch: `R = (Z₂ − Z₁) / (Z₂ + Z₁)`. Returned as a magnitude,
/// because the sign is a phase inversion and nothing here is doing phase.
///
/// At normal incidence. Glancing angles reflect more, which matters for a ricochet off a
/// distant slope and is left to the caller who knows the geometry.
pub fn reflection_coefficient(material: &Material) -> f64 {
    let z = impedance(material);
    ((z - AIR_IMPEDANCE) / (z + AIR_IMPEDANCE)).abs().clamp(0.0, 1.0)
}

/// Fraction of *energy* absorbed by a surface, 0 to 1.
///
/// The complement of the reflected energy, and the number room acoustics is usually
/// written in terms of. A material with a huge impedance absorbs almost nothing — which
/// is why a concrete room is unbearable and a sofa fixes it.
pub fn absorption_coefficient(material: &Material) -> f64 {
    let r = reflection_coefficient(material);
    (1.0 - r * r).clamp(0.0, 1.0)
}

/// Loss from having to diffract around an obstacle, in decibels.
///
/// # Maekawa
///
/// `path_difference` is how much further sound must travel over or around the barrier
/// than it would in a straight line, in metres. From that and the wavelength comes the
/// Fresnel number `N = 2δ/λ`, and the attenuation follows.
///
/// The behaviour worth knowing: attenuation grows with `N`, so a *short* wavelength is
/// blocked far more effectively than a long one by the same obstacle. A ridge between you
/// and a firefight removes the crack and leaves the thump, which is exactly what a ridge
/// does in life and is not a rule anybody has to write down separately.
///
/// Returns 0 for a clear line of sight, and is capped: real barriers stop attenuating
/// somewhere around 20-25 dB because sound arrives by other routes, and a model that
/// returned 60 would silence things the player can plainly see.
pub fn barrier_insertion_db(path_difference_m: f64, wavelength_m: f64) -> f64 {
    if path_difference_m <= 0.0 || wavelength_m <= 0.0 {
        return 0.0;
    }
    let n = 2.0 * path_difference_m / wavelength_m;
    let arg = (2.0 * std::f64::consts::PI * n).sqrt();
    // `tanh` saturates, so this is the whole curve: it rises steeply for small N and then
    // flattens, which is the measured shape.
    let db = 5.0 + 20.0 * (arg / arg.tanh()).log10();
    db.clamp(0.0, 24.0)
}

/// Wavelength in metres, for a frequency in a medium of a given sound speed.
pub fn wavelength(frequency_hz: f64, speed_of_sound: f64) -> f64 {
    if frequency_hz <= 0.0 {
        return f64::INFINITY;
    }
    speed_of_sound / frequency_hz
}

/// Excess attenuation from propagating *through* standing vegetation, in dB per metre.
///
/// # The term that makes a treeline audible
///
/// Air absorption is small at the scale of a field — a few dB over a hundred metres —
/// and a barrier only applies when something is squarely in the way. Neither explains the
/// most familiar outdoor effect there is: a sound that has come through woodland is
/// duller than the same sound across open ground at the same distance.
///
/// That is scattering off leaves and stems, and ISO 9613-2 tabulates it. The values here
/// are its dense-foliage figures for a path of 20 m or more, running from 0.02 dB/m in
/// the bass to 0.12 dB/m at 8 kHz. The ratio is the point: six to one across the
/// spectrum, so a hundred metres of scrub costs 2 dB of bass and 12 dB of treble, which
/// is a filter and not a fader.
///
/// Interpolated on a log-frequency axis, because that is how the table is spaced and how
/// hearing is.
pub fn foliage_absorption_db_per_m(frequency_hz: f64) -> f64 {
    // ISO 9613-2 Table 5, dense foliage, 20 m <= path <= 200 m.
    const OCTAVES: [(f64, f64); 8] = [
        (63.0, 0.02),
        (125.0, 0.03),
        (250.0, 0.04),
        (500.0, 0.05),
        (1_000.0, 0.06),
        (2_000.0, 0.08),
        (4_000.0, 0.09),
        (8_000.0, 0.12),
    ];

    let last = OCTAVES[OCTAVES.len() - 1];
    if frequency_hz <= OCTAVES[0].0 {
        return OCTAVES[0].1;
    }
    if frequency_hz >= last.0 {
        return last.1;
    }
    for pair in OCTAVES.windows(2) {
        let (f0, a0) = pair[0];
        let (f1, a1) = pair[1];
        if frequency_hz <= f1 {
            let t = (frequency_hz / f0).ln() / (f1 / f0).ln();
            return a0 + (a1 - a0) * t;
        }
    }
    last.1
}

/// Loss from a path through vegetation, in decibels.
///
/// `metres` is the distance spent *inside* cover, not the total distance — a source in
/// the open beyond a treeline is attenuated by the treeline's depth and nothing more.
///
/// Capped at the standard's own limit. ISO 9613-2 stops crediting foliage past 200 m of
/// path, because beyond that the sound is arriving over the canopy rather than through
/// it, and a model that kept integrating would silence a forest.
pub fn foliage_attenuation_db(metres: f64, frequency_hz: f64) -> f64 {
    const MAX_CREDITED_M: f64 = 200.0;
    if metres <= 0.0 {
        return 0.0;
    }
    foliage_absorption_db_per_m(frequency_hz) * metres.min(MAX_CREDITED_M)
}

/// A loss in decibels as a linear amplitude multiplier.
///
/// Here rather than at every call site because every term in this module is quoted in dB
/// and every mixer wants a gain, and the sign convention is exactly the kind of thing
/// that gets inverted once and then hidden behind a compensating constant.
pub fn gain_from_db_loss(db: f64) -> f64 {
    if db <= 0.0 {
        return 1.0;
    }
    10f64.powf(-db / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acoustics::Air;


    /// **Foliage is a filter, not a fader.** ISO 9613-2's dense-foliage figures run six
    /// to one across the spectrum, which is why a treeline takes the crack out of a
    /// distant shot and leaves the thump — the same character as a ridge, from a
    /// completely different mechanism.
    #[test]
    fn foliage_takes_treble_and_leaves_bass() {
        let through = 100.0;
        let treble = foliage_attenuation_db(through, 4_000.0);
        let bass = foliage_attenuation_db(through, 125.0);

        assert!(
            (8.0..11.0).contains(&treble),
            "100 m of dense cover took {treble:.1} dB of 4 kHz against the standard's 9",
        );
        assert!(
            treble > bass * 2.5,
            "treble lost {treble:.1} dB and bass {bass:.1} - not the ratio the table has",
        );
    }

    /// The published table, at the frequencies it is published at.
    #[test]
    fn foliage_matches_the_tabulated_octaves() {
        for (hz, expected) in [
            (63.0, 0.02),
            (500.0, 0.05),
            (1_000.0, 0.06),
            (4_000.0, 0.09),
            (8_000.0, 0.12),
        ] {
            let got = foliage_absorption_db_per_m(hz);
            assert!(
                (got - expected).abs() < 1e-9,
                "{hz} Hz gave {got:.4} dB/m against ISO 9613-2's {expected}",
            );
        }
        // And between them it interpolates rather than stepping.
        let between = foliage_absorption_db_per_m(1_400.0);
        assert!(
            between > 0.06 && between < 0.08,
            "1.4 kHz gave {between:.4}, outside the octaves it sits between",
        );
    }

    /// Off the ends of the table it holds rather than running away.
    #[test]
    fn foliage_is_bounded_outside_the_table() {
        assert_eq!(foliage_absorption_db_per_m(1.0), 0.02);
        assert_eq!(foliage_absorption_db_per_m(40_000.0), 0.12);
        // And a path longer than the standard credits stops accumulating: past a couple
        // of hundred metres sound arrives over the canopy, not through it.
        let credited = foliage_attenuation_db(200.0, 4_000.0);
        assert_eq!(foliage_attenuation_db(10_000.0, 4_000.0), credited);
        assert_eq!(foliage_attenuation_db(0.0, 4_000.0), 0.0);
    }

    /// Decibels convert the way decibels convert, in the direction losses go.
    #[test]
    fn a_loss_in_decibels_is_a_gain_below_one() {
        assert!((gain_from_db_loss(0.0) - 1.0).abs() < 1e-12);
        assert!((gain_from_db_loss(6.0206) - 0.5).abs() < 1e-4, "6 dB should halve it");
        assert!((gain_from_db_loss(20.0) - 0.1).abs() < 1e-9);
        // A negative loss is a gain, and this is a loss function: it does not amplify.
        assert_eq!(gain_from_db_loss(-10.0), 1.0);
    }

    /// **The two obstruction terms are different animals.** A barrier saturates - past a
    /// point the sound arrives by another route - while foliage keeps integrating along
    /// the path. Which one dominates depends on the map, and both are needed.
    #[test]
    fn a_barrier_saturates_and_foliage_accumulates() {
        let c = Air::standard().speed_of_sound();
        let lambda = wavelength(4_000.0, c);
        assert_eq!(
            barrier_insertion_db(50.0, lambda),
            barrier_insertion_db(500.0, lambda),
            "a barrier should have stopped attenuating well before this",
        );
        assert!(
            foliage_attenuation_db(150.0, 4_000.0) > foliage_attenuation_db(50.0, 4_000.0),
            "three times the cover was not more attenuation",
        );
    }
    /// Concrete rings; rubber does not. Neither was told to.
    #[test]
    fn hard_surfaces_reflect_and_soft_ones_do_not() {
        let concrete = reflection_coefficient(&Material::concrete());
        let rubber = reflection_coefficient(&Material::rubber());
        let steel = reflection_coefficient(&Material::steel());

        assert!(
            concrete > 0.99 && steel > 0.99,
            "concrete {concrete:.4} and steel {steel:.4} should be near-perfect \
             reflectors - their impedance is thousands of times air's",
        );
        assert!(
            rubber < concrete,
            "rubber ({rubber:.4}) reflected as hard as concrete ({concrete:.4})",
        );
    }

    /// Impedance is derived, so a denser and stiffer material is a louder echo without
    /// anybody choosing that.
    #[test]
    fn impedance_follows_density_and_stiffness() {
        assert!(impedance(&Material::steel()) > impedance(&Material::wood()));
        assert!(impedance(&Material::wood()) > impedance(&Material::dry_vegetation()));
        assert!(
            impedance(&Material::steel()) > AIR_IMPEDANCE * 1_000.0,
            "steel's impedance should dwarf air's",
        );
    }

    /// Absorbed plus reflected is all of it.
    #[test]
    fn energy_is_conserved_at_a_boundary() {
        for m in [Material::concrete(), Material::wood(), Material::rubber(), Material::ice()] {
            let r = reflection_coefficient(&m);
            let a = absorption_coefficient(&m);
            assert!(
                (r * r + a - 1.0).abs() < 1e-9,
                "reflected energy {:.4} plus absorbed {a:.4} is not one",
                r * r,
            );
        }
    }

    /// **A ridge takes the crack and leaves the thump.** The same obstacle attenuates a
    /// short wavelength far more than a long one, which is the whole character of hearing
    /// a fight from behind cover.
    #[test]
    fn a_barrier_blocks_treble_far_better_than_bass() {
        let c = Air::standard().speed_of_sound();
        // Two metres further round the obstacle than through it.
        let detour = 2.0;

        let treble = barrier_insertion_db(detour, wavelength(6_000.0, c));
        let bass = barrier_insertion_db(detour, wavelength(120.0, c));

        assert!(
            treble > bass + 6.0,
            "a barrier removed {treble:.1} dB of 6 kHz and {bass:.1} dB of 120 Hz - not \
             enough separation to sound like cover",
        );
    }

    /// Line of sight costs nothing, and the model never silences something outright.
    #[test]
    fn a_barrier_is_bounded_at_both_ends() {
        let c = Air::standard().speed_of_sound();
        assert_eq!(barrier_insertion_db(0.0, wavelength(1_000.0, c)), 0.0);
        assert!(barrier_insertion_db(500.0, wavelength(16_000.0, c)) <= 24.0);
    }

    /// More detour is more loss, monotonically — a bigger hill hides more.
    #[test]
    fn deeper_shadow_is_quieter() {
        let c = Air::standard().speed_of_sound();
        let lambda = wavelength(2_000.0, c);
        let shallow = barrier_insertion_db(0.2, lambda);
        let deep = barrier_insertion_db(3.0, lambda);
        assert!(deep > shallow, "a deeper acoustic shadow was not quieter");
    }
}
