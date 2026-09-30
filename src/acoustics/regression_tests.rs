//! Regression tests from the 2026-09-29 acoustics review
//! (`docs/reviews/2026-09-29-acoustics.md`).
//!
//! Each test checks the module against a published figure or an independent formula,
//! never against its own output: ISO 9613-2's absorption table, Cramer's (1993) fit for
//! the speed of sound, Kinsler's tabulated bulk impedances, and the limits of the
//! Kurze–Anderson form of Maekawa's curve. The first three pass on the code as reviewed
//! and are here because the module's own tests allow ±40%. The rest failed on it; the
//! incompressible-material test guards the edge case of the impedance fix itself. The
//! `#[ignore]`d ones pin limitations the review documents but does not fix; run them with
//! `cargo test --lib acoustics -- --ignored` to see the current behaviour.

use crate::acoustics::surfaces::{
    absorption_coefficient, barrier_insertion_db, gain_from_db_loss, impedance,
    reflection_coefficient, wavelength,
};
use crate::acoustics::{doppler_ratio, Air, REFERENCE_PRESSURE};
use crate::materials::Material;

/// Every value in ISO 9613-2 Table 2 (dB/km, octave bands 63 Hz to 8 kHz), at the exact
/// midband frequencies the standard computes them at. The table is rounded to 0.1 dB/km
/// and to three significant figures, which sets the tolerance; 15 C, 80%, 1 kHz sits on
/// the rounding boundary (4.15 against a published 4.1).
#[test]
fn review_absorption_matches_every_row_of_iso_9613_2_table_2() {
    const TABLE: [((f64, f64), [f64; 8]); 6] = [
        ((10.0, 0.70), [0.1, 0.4, 1.0, 1.9, 3.7, 9.7, 32.8, 117.0]),
        ((20.0, 0.70), [0.1, 0.3, 1.1, 2.8, 5.0, 9.0, 22.9, 76.6]),
        ((30.0, 0.70), [0.1, 0.3, 1.0, 3.1, 7.4, 12.7, 23.1, 59.3]),
        ((15.0, 0.20), [0.3, 0.6, 1.2, 2.7, 8.2, 28.2, 88.8, 202.0]),
        ((15.0, 0.50), [0.1, 0.5, 1.2, 2.2, 4.2, 10.8, 36.2, 129.0]),
        ((15.0, 0.80), [0.1, 0.3, 1.1, 2.4, 4.1, 8.3, 23.7, 82.8]),
    ];
    for ((celsius, humidity), row) in TABLE {
        let air = Air::from_celsius(celsius, humidity, REFERENCE_PRESSURE).unwrap();
        for (band, published) in row.iter().enumerate() {
            let exact_midband = 1_000.0 * 10f64.powf((3.0 * band as f64 - 12.0) / 10.0);
            let got = air.absorption_db_per_m(exact_midband) * 1_000.0;
            let tolerance = (0.005 * published).max(0.06);
            assert!(
                (got - published).abs() <= tolerance,
                "{celsius} C, {humidity} RH, {exact_midband:.0} Hz: {got:.3} dB/km against \
                 the table's {published}",
            );
        }
    }
}

/// The table is all at one atmosphere. ISO 9613-1 scales both relaxation frequencies and
/// the classical term with pressure, and so does the code: these are an independent
/// implementation of ISO 9613-1 eqs. (3)-(5) and Annex B at 80 kPa, 10 C and 30% RH.
#[test]
fn review_absorption_follows_iso_9613_1_off_standard_pressure() {
    let air = Air::from_celsius(10.0, 0.30, 80_000.0).unwrap();
    for (hz, expected) in [(1_000.0, 0.006_240_957_540_911_371), (4_000.0, 0.073_468_193_178_785_31)] {
        let got = air.absorption_db_per_m(hz);
        assert!(
            ((got - expected) / expected).abs() < 1e-9,
            "{hz} Hz at 80 kPa gave {got} dB/m against ISO 9613-1's {expected}",
        );
    }
}

/// Humidity raises the speed of sound by about 1.25 m/s at saturation at 20 C and 2.3 m/s
/// at 30 C, per Cramer (1993), not the 0.3 m/s the doc used to quote.
#[test]
fn review_humidity_raises_the_speed_of_sound_by_cramers_amount() {
    for (celsius, cramer) in [(20.0, 1.25), (30.0, 2.31)] {
        let dry = Air::from_celsius(celsius, 0.0, REFERENCE_PRESSURE).unwrap().speed_of_sound();
        let wet = Air::from_celsius(celsius, 1.0, REFERENCE_PRESSURE).unwrap().speed_of_sound();
        assert!(
            (wet - dry - cramer).abs() < 0.1,
            "saturation at {celsius} C raised c by {:.3} m/s against Cramer's {cramer}",
            wet - dry,
        );
    }
}

/// **ACU-3.** A thick slab carries a compression wave at the P-wave speed, not the thin-rod
/// speed `sqrt(E/ρ)`. Kinsler et al. (Fundamentals of Acoustics, 4th ed., appendix table
/// of solids) give the bulk impedance of steel as 47.0 and aluminium as 17.0 MPa·s/m; the
/// rod formula gave 39.6 and 13.7.
#[test]
fn review_impedance_is_the_bulk_p_wave_impedance() {
    for (name, material, kinsler) in [
        ("steel", Material::steel(), 47.0e6),
        ("aluminium", Material::aluminum(), 17.0e6),
    ] {
        let z = impedance(&material);
        assert!(
            ((z - kinsler) / kinsler).abs() < 0.05,
            "{name} impedance {z:.3e} rayl against Kinsler's bulk {kinsler:.3e}",
        );
        let rod = (material.youngs_modulus * material.density).sqrt();
        assert!(z > rod * 1.1, "{name} came out at the thin-rod impedance");
    }
}

/// An incompressible material (Poisson's ratio exactly 0.5) has an infinite impedance and
/// reflects everything, rather than producing `inf / inf`.
#[test]
fn review_an_incompressible_material_reflects_everything() {
    let mut rubber = Material::rubber();
    rubber.poisson_ratio = 0.5;
    assert_eq!(reflection_coefficient(&rubber), 1.0);
    assert_eq!(absorption_coefficient(&rubber), 0.0);
}

/// The module doc's claim, pinned: every solid preset reflects more than 99.7% of the
/// incident pressure, a few hundredths of a decibel per bounce, so none of them is soft.
#[test]
fn review_every_solid_preset_reflects_more_than_99_7_percent() {
    for (name, material) in [
        ("steel", Material::steel()),
        ("aluminum", Material::aluminum()),
        ("rubber", Material::rubber()),
        ("polyurethane", Material::polyurethane()),
        ("rope", Material::rope()),
        ("wood", Material::wood()),
        ("dry_vegetation", Material::dry_vegetation()),
        ("copper", Material::copper()),
        ("titanium", Material::titanium()),
        ("concrete", Material::concrete()),
        ("glass", Material::glass()),
        ("brass", Material::brass()),
        ("ice", Material::ice()),
        ("stainless_steel", Material::stainless_steel()),
    ] {
        let r = reflection_coefficient(&material);
        assert!(r > 0.997, "{name} reflected {r:.5} of the pressure");
        assert!(-20.0 * r.log10() < 0.03, "{name} lost more than 0.03 dB per bounce");
    }
}

/// **ACU-4.** A listener receding faster than sound used to get a negative frequency ratio
/// (-0.16 at 400 m/s), and exactly at `c`, zero. It is clamped like the source now: small,
/// positive, and never rising as the listener recedes faster.
#[test]
fn review_doppler_is_never_negative_for_a_receding_listener() {
    let air = Air::standard();
    let c = air.speed_of_sound();
    let mut previous = doppler_ratio(0.0, 0.0, &air);
    for listener in [-0.5 * c, -0.95 * c, -c, -400.0, -10.0 * c] {
        let ratio = doppler_ratio(0.0, listener, &air);
        assert!(
            ratio.is_finite() && ratio > 0.0,
            "a listener receding at {listener:.0} m/s got a ratio of {ratio}",
        );
        assert!(ratio <= previous + 1e-12, "receding faster raised the pitch");
        previous = ratio;
    }
}

/// **ACU-5.** `wavelength` returns infinity at 0 Hz, and `barrier_insertion_db` of that was
/// `0/0`. A NaN that reaches a recursive filter stays there for good, so every function
/// that turns a band into a loss or a gain returns a number for any input.
#[test]
fn review_non_finite_inputs_never_become_nan() {
    let c = Air::standard().speed_of_sound();
    let at_dc = barrier_insertion_db(1.0, wavelength(0.0, c));
    assert!(
        (at_dc - 5.0).abs() < 1e-9,
        "a barrier at 0 Hz gave {at_dc} dB, not the N -> 0 limit of 5",
    );
    assert_eq!(barrier_insertion_db(f64::NAN, 1.0), 0.0);
    assert_eq!(barrier_insertion_db(1.0, f64::NAN), 0.0);
    assert_eq!(gain_from_db_loss(f64::NAN), 1.0);
    assert_eq!(Air::standard().absorption_db_per_m(f64::NAN), 0.0);
    assert_eq!(Air::standard().absorption_gain(f64::NAN, 100.0), 1.0);
}

/// **ACU-2, fixed** (the signed, continuous barrier law, merged from the branch that had
/// already made it; the review saw the older one-sided form). Maekawa's curve is continuous through the shadow boundary: in the
/// Kurze–Anderson form the lit side is `5 + 20 log10(√(2π|N|) / tan √(2π|N|))`, which
/// is 5 dB at `N → 0⁻` and reaches 0 near `N = −0.19`. The one-sided form returned 0 on the
/// lit side, so a source dropping behind a ridge stepped by 5 dB at every frequency at once.
#[test]
fn review_barrier_is_continuous_across_the_shadow_boundary() {
    let c = Air::standard().speed_of_sound();
    for hz in [125.0, 1_000.0, 8_000.0] {
        let lambda = wavelength(hz, c);
        let lit = barrier_insertion_db(-1e-6, lambda);
        let shadowed = barrier_insertion_db(1e-6, lambda);
        assert!(
            (shadowed - lit).abs() < 0.5,
            "{hz} Hz stepped from {lit:.2} dB to {shadowed:.2} dB across the shadow boundary",
        );
    }
}

/// **ACU-1, open.** ISO 9613-2 (§7.3.1) classes ground covered by grass or other
/// vegetation as porous (G = 1) and paving, concrete and ice as hard (G = 0). Here the
/// grass-and-scrub preset absorbs well under 1% at normal incidence, like concrete.
/// Measured absorption for grass-covered ground is tenths, not thousandths.
#[test]
#[ignore = "known limitation ACU-1: no porous absorption; vegetation reflects like a hard surface"]
fn review_vegetation_ground_is_acoustically_soft() {
    let vegetation = absorption_coefficient(&Material::dry_vegetation());
    let concrete = absorption_coefficient(&Material::concrete());
    assert!(concrete < 0.05, "concrete absorbed {concrete:.4}");
    assert!(
        vegetation > 0.1,
        "grass and scrub absorbed {vegetation:.4} of the incident energy",
    );
}
