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
/// # Signed, and continuous at grazing (Kurze-Anderson)
///
/// The path difference is **signed**: positive when the obstacle's edge stands above the
/// straight line (the listener is in its shadow), negative when the line clears the edge
/// by that much (the lit zone). With `N = 2 x path_difference / wavelength`:
///
/// | Zone | Insertion loss, dB |
/// |---|---|
/// | `N > 0` (shadow) | `5 + 20 log10( sqrt(2 pi N) / tanh(sqrt(2 pi N)) )` |
/// | `-N0 < N <= 0` (lit, inside the first Fresnel zone) | `5 + 20 log10( sqrt(2 pi \|N\|) / tan(sqrt(2 pi \|N\|)) )` |
/// | `N <= -N0` | 0 |
///
/// Both branches give [`BARRIER_GRAZING_DB`] at `N = 0`, so a source that walks over a
/// crest fades through it instead of stepping 5 dB the instant the line is cut. That step
/// was here until 2026-09-25 (the function returned 0 for any `path_difference <= 0` and 5
/// just past it) and was an audible click. `N0` is where the lit-zone branch reaches 0 dB;
/// it is computed by [`lit_zone_limit`], never typed.
///
/// Capped at [`BARRIER_CAP_DB`]: real barriers stop attenuating somewhere around 20-25 dB
/// because sound arrives by other routes, and a model that returned 60 would silence
/// things the player can plainly see.
///
/// # Arguments
///
/// * `path_difference_m` - the signed excess path `|SQ| + |QL| - |SL|` over the edge `Q`,
///   in metres: positive in the shadow, negative in the lit zone.
/// * `wavelength_m` - the wavelength, in metres (see [`wavelength`]).
///
/// # Returns
///
/// The insertion loss in decibels, in `0..=BARRIER_CAP_DB`. A non-positive or non-finite
/// wavelength returns 0.
///
/// # Examples
///
/// ```
/// use rs_physics::acoustics::surfaces::{
///     barrier_insertion_db, lit_zone_limit, wavelength, BARRIER_GRAZING_DB,
/// };
///
/// let lambda = wavelength(4_000.0, 343.0);
/// // Grazing is 5 dB from both sides, so a crest fades rather than clicks.
/// assert!((barrier_insertion_db(0.0, lambda) - BARRIER_GRAZING_DB).abs() < 1e-12);
/// assert!((barrier_insertion_db(1e-9, lambda) - BARRIER_GRAZING_DB).abs() < 1e-3);
/// assert!((barrier_insertion_db(-1e-9, lambda) - BARRIER_GRAZING_DB).abs() < 1e-3);
/// // Clear by more than the lit zone, nothing is lost.
/// let clear = -lit_zone_limit() * lambda / 2.0;
/// assert_eq!(barrier_insertion_db(clear * 1.01, lambda), 0.0);
/// // Deep in the shadow, a ridge takes a good deal.
/// assert!(barrier_insertion_db(1.0, lambda) > 15.0);
/// ```
pub fn barrier_insertion_db(path_difference_m: f64, wavelength_m: f64) -> f64 {
    if !(wavelength_m > 0.0) || !wavelength_m.is_finite() || path_difference_m.is_nan() {
        return 0.0;
    }
    let n = 2.0 * path_difference_m / wavelength_m;
    if n == 0.0 {
        return BARRIER_GRAZING_DB;
    }
    let x = (2.0 * std::f64::consts::PI * n.abs()).sqrt();
    let ratio = if n > 0.0 {
        // `tanh` saturates, so this is the whole shadow curve: it rises steeply for small
        // N and then flattens, which is the measured shape.
        x / x.tanh()
    } else {
        // `x / tan x` falls from 1 at grazing through zero at pi/2; past the root of the
        // loss (x0 < pi/2) the edge is outside the first Fresnel zone and costs nothing.
        if x >= std::f64::consts::FRAC_PI_2 {
            return 0.0;
        }
        x / x.tan()
    };
    let db = BARRIER_GRAZING_DB + 20.0 * ratio.log10();
    db.clamp(0.0, BARRIER_CAP_DB)
}

/// Insertion loss at grazing incidence (`N = 0`), in decibels: Maekawa's and
/// Kurze-Anderson's value for an edge exactly on the line of sight.
pub const BARRIER_GRAZING_DB: f64 = 5.0;

/// The most a single barrier takes, in decibels.
///
/// Measured barriers stop gaining somewhere between 20 and 25 dB because sound reaches the
/// far side by other routes (over the top of the ground effect, off anything nearby). This
/// sits inside that range; the GPU acoustics use the same figure.
pub const BARRIER_CAP_DB: f64 = 24.0;

/// `N0`, the Fresnel number at which the lit-zone branch of [`barrier_insertion_db`]
/// reaches 0 dB: the root of `5 + 20 log10(x / tan x) = 0` with `N0 = x^2 / (2 pi)`.
///
/// Computed, not typed. It comes out at about 0.19, and a path that clears its edge by
/// more than `N0 x wavelength / 2` loses nothing to it.
///
/// # Returns
///
/// `N0`, dimensionless, in `(0, 0.25)`.
///
/// # Examples
///
/// ```
/// use rs_physics::acoustics::surfaces::{barrier_insertion_db, lit_zone_limit};
///
/// let n0 = lit_zone_limit();
/// assert!(n0 > 0.18 && n0 < 0.20);
/// // At `-N0` the lit zone ends: the loss is zero to rounding.
/// let lambda = 1.0;
/// assert!(barrier_insertion_db(-n0 * lambda / 2.0, lambda) < 1e-9);
/// ```
pub fn lit_zone_limit() -> f64 {
    static N0: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *N0.get_or_init(|| {
        // x / tan x falls monotonically from 1 (x -> 0) to 0 (x = pi/2), so bisection on
        // it against the ratio that makes the loss zero converges to the one root.
        let target = 10f64.powf(-BARRIER_GRAZING_DB / 20.0);
        let (mut lo, mut hi) = (1e-6f64, std::f64::consts::FRAC_PI_2);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if mid / mid.tan() > target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let x0 = 0.5 * (lo + hi);
        x0 * x0 / (2.0 * std::f64::consts::PI)
    })
}

/// Radius of the first Fresnel zone at a point on a path, in metres.
///
/// `r1 = sqrt(wavelength x d1 x d2 / (d1 + d2))`, where the point splits the path into
/// `d1` and `d2`. An obstacle narrower than this does not shadow the wavelength: the wave
/// passes round both sides of it (see [`occludes`]).
///
/// # Arguments
///
/// * `wavelength_m` - the wavelength, in metres.
/// * `d1_m` - distance from the source to the point, in metres.
/// * `d2_m` - distance from the point to the listener, in metres.
///
/// # Returns
///
/// The radius in metres; 0 when either distance is non-positive or the wavelength is.
///
/// # Examples
///
/// ```
/// use rs_physics::acoustics::surfaces::{fresnel_radius, wavelength};
///
/// // The midpoint of a 17.4 m path at 4 kHz: the scene rule's threshold, about 0.61 m.
/// let r1 = fresnel_radius(wavelength(4_000.0, 343.2), 8.7, 8.7);
/// assert!((r1 - 0.611).abs() < 0.002);
/// ```
pub fn fresnel_radius(wavelength_m: f64, d1_m: f64, d2_m: f64) -> f64 {
    if !(wavelength_m > 0.0) || !(d1_m > 0.0) || !(d2_m > 0.0) {
        return 0.0;
    }
    (wavelength_m * d1_m * d2_m / (d1_m + d2_m)).sqrt()
}

/// Whether an obstacle of a lateral half-width counts as an occluder at a wavelength.
///
/// **The Fresnel rule.** An obstacle whose half-width is under the first Fresnel radius at
/// the point where the path crosses it does not shadow that band: sound goes round both
/// sides. So a soldier does not muffle a gunshot behind him, and a tank does. This is the
/// one rule for what occludes; no list of kinds stands beside it.
///
/// # Arguments
///
/// * `half_width_m` - the obstacle's lateral half-width across the path, in metres.
/// * `wavelength_m` - the wavelength, in metres.
/// * `d1_m`, `d2_m` - the path's lengths either side of the crossing point, in metres.
///
/// # Returns
///
/// `true` when `half_width_m >= fresnel_radius(wavelength_m, d1_m, d2_m)`.
///
/// # Examples
///
/// ```
/// use rs_physics::acoustics::surfaces::{occludes, wavelength};
///
/// let lambda = wavelength(4_000.0, 343.2);
/// // Midway along a 17.4 m path: an infantryman (0.5 m) is dropped, a tank (1.25 m) is kept.
/// assert!(!occludes(0.5, lambda, 8.7, 8.7));
/// assert!(occludes(1.25, lambda, 8.7, 8.7));
/// ```
pub fn occludes(half_width_m: f64, wavelength_m: f64, d1_m: f64, d2_m: f64) -> bool {
    half_width_m >= fresnel_radius(wavelength_m, d1_m, d2_m)
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
        let lambda = wavelength(1_000.0, c);
        // A clear line of sight, by more than the lit zone, costs nothing. (Grazing costs
        // 5 dB since the law became signed; see `the_barrier_law_is_continuous_at_grazing`.)
        assert_eq!(barrier_insertion_db(-lit_zone_limit() * lambda, lambda), 0.0);
        assert_eq!(barrier_insertion_db(-10.0, lambda), 0.0);
        assert!(barrier_insertion_db(500.0, wavelength(16_000.0, c)) <= BARRIER_CAP_DB);
    }

    /// **L6, the law half.** Both branches meet at 5 dB at grazing, the lit branch reaches
    /// 0 at `-N0` and stays there, and the whole curve is monotone in the signed excess -
    /// so a source walking over a crest fades through it instead of clicking.
    #[test]
    fn the_barrier_law_is_continuous_at_grazing() {
        let lambda = 0.0858;
        let n0 = lit_zone_limit();
        let at = |n: f64| barrier_insertion_db(n * lambda / 2.0, lambda);
        assert_eq!(at(0.0), BARRIER_GRAZING_DB);
        for eps in [1e-3, 1e-6, 1e-9] {
            assert!((at(eps) - BARRIER_GRAZING_DB).abs() < 20.0 * eps.sqrt(), "shadow side at {eps}");
            assert!((at(-eps) - BARRIER_GRAZING_DB).abs() < 20.0 * eps.sqrt(), "lit side at {eps}");
        }
        assert!(at(-n0).abs() < 1e-9, "the lit zone does not end at -N0: {}", at(-n0));
        assert_eq!(at(-n0 * 1.001), 0.0);
        let mut last = 0.0;
        for k in 0..=4000 {
            let n = -0.3 + k as f64 * 0.001;
            let db = at(n);
            assert!(db >= last - 1e-12, "not monotone at N = {n}: {db} after {last}");
            last = db;
        }
    }

    /// The Fresnel rule's threshold at the design's path: a 17.4 m path at 4 kHz has a
    /// first-zone radius of 0.611 m at its midpoint, which keeps a Siege (1.25 m) and
    /// drops a Worker (0.50 m).
    #[test]
    fn the_fresnel_rule_keeps_vehicles_and_drops_infantry() {
        let lambda = wavelength(4_000.0, Air::standard().speed_of_sound());
        let r1 = fresnel_radius(lambda, 8.7, 8.7);
        assert!((r1 - 0.611).abs() < 0.003, "r1 came out {r1}");
        assert!(occludes(1.25, lambda, 8.7, 8.7));
        assert!(!occludes(0.50, lambda, 8.7, 8.7));
        // Degenerate paths have no zone.
        assert_eq!(fresnel_radius(lambda, 0.0, 10.0), 0.0);
        assert_eq!(fresnel_radius(0.0, 5.0, 10.0), 0.0);
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
