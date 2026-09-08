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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acoustics::Air;

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
