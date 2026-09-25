//! **L6. The physics of occlusion.** Not the GPU against the CPU (they could share a wrong
//! law) but the law against the wave equation's own answer.
//!
//! A column of half-width `w` stands at the midpoint of a 17.4 m path (the default
//! camera's focus-to-edge distance) and the wave is 4 kHz. The field behind it comes from
//! the Fresnel-Kirchhoff diffraction integral over the plane of the column, less the part
//! the column blocks (Babinet). In the Fresnel approximation, which holds here because the
//! column is under 0.2 of the path's half-length wide, the integral over the column's
//! height is the free field's own and the lateral one is
//!
//! `U / U0 = 1 - (1 / (1 + i)) integral from -v0 to v0 of exp(i pi v^2 / 2) dv`,
//! `v0 = w sqrt(2 / (lambda d_eff))`, `d_eff = d1 d2 / (d1 + d2)`.
//!
//! It is integrated numerically here (composite Gauss-Legendre in f64), not taken from a
//! table of Fresnel integrals.

use rs_physics::acoustics::surfaces::{barrier_insertion_db, fresnel_radius, occludes, wavelength};
use rs_physics::acoustics::Air;

/// Five-point Gauss-Legendre nodes and weights on [-1, 1].
const GL: [(f64, f64); 5] = [
    (0.0, 0.568_888_888_888_888_9),
    (-0.538_469_310_105_683_1, 0.478_628_670_499_366_5),
    (0.538_469_310_105_683_1, 0.478_628_670_499_366_5),
    (-0.906_179_845_938_664, 0.236_926_885_056_189_1),
    (0.906_179_845_938_664, 0.236_926_885_056_189_1),
];

/// `integral_0^v exp(i pi t^2 / 2) dt`, as (re, im), over panels of 1/256.
fn fresnel(v: f64) -> (f64, f64) {
    let panels = ((v * 256.0).ceil() as usize).max(1);
    let h = v / panels as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for p in 0..panels {
        let mid = (p as f64 + 0.5) * h;
        for (x, w) in GL {
            let t = mid + 0.5 * h * x;
            let phase = std::f64::consts::PI * t * t / 2.0;
            re += 0.5 * h * w * phase.cos();
            im += 0.5 * h * w * phase.sin();
        }
    }
    (re, im)
}

/// Insertion loss of a column of half-width `w` at the midpoint of a path, in dB.
fn column_loss_db(w: f64, lambda: f64, d1: f64, d2: f64) -> f64 {
    let d_eff = d1 * d2 / (d1 + d2);
    let v0 = w * (2.0 / (lambda * d_eff)).sqrt();
    let (c, s) = fresnel(v0);
    // The blocked strip is 2 (C + iS); divided by (1 + i): ((2C + 2S) + i(2S - 2C)) / 2.
    let (br, bi) = (c + s, s - c);
    let (ur, ui) = (1.0 - br, -bi);
    -20.0 * (ur * ur + ui * ui).sqrt().log10()
}

#[test]
fn l6_a_column_occludes_as_the_diffraction_integral_says() {
    // The quadrature against the one closed-form value there is: the full integral's limit.
    let (c, s) = fresnel(60.0);
    assert!(
        (c - 0.5).abs() < 0.006 && (s - 0.5).abs() < 0.006,
        "C, S at 60: {c}, {s}"
    );

    let c_air = Air::standard().speed_of_sound();
    let lambda = wavelength(4_000.0, c_air);
    let (d1, d2) = (8.7, 8.7);
    let r1 = fresnel_radius(lambda, d1, d2);

    // Monotone: a wider column never lets more through.
    let mut last = -1.0;
    let mut w = 0.0;
    while w <= 3.0 {
        let db = column_loss_db(w, lambda, d1, d2);
        assert!(
            db >= last - 1e-9,
            "the loss fell from {last} to {db} at w = {w}"
        );
        last = db;
        w += 0.005;
    }
    assert!(
        column_loss_db(0.0, lambda, d1, d2).abs() < 1e-12,
        "no column, no loss"
    );

    // What the drop rule accepts: the loss of the widest column it drops.
    let at_r1 = column_loss_db(r1, lambda, d1, d2);
    let at_half = column_loss_db(0.5 * r1, lambda, d1, d2);
    let at_soldier = column_loss_db(0.25, lambda, d1, d2);
    let at_tank = column_loss_db(1.25, lambda, d1, d2);
    // For context: what a semi-infinite screen at the same grazing geometry would charge.
    let grazing = barrier_insertion_db(0.0, lambda);
    println!("L6: 4 kHz, a 17.4 m path, r1 = {r1:.3} m");
    println!(
        "  column of half-width r1     : {at_r1:.2} dB  (the error the Fresnel drop rule accepts)"
    );
    println!("  column of half-width r1 / 2 : {at_half:.2} dB");
    println!(
        "  a soldier (0.25 m)          : {at_soldier:.2} dB, dropped: {}",
        !occludes(0.25, lambda, d1, d2)
    );
    println!(
        "  a Siege (1.25 m)            : {at_tank:.2} dB, kept: {}",
        occludes(1.25, lambda, d1, d2)
    );
    println!("  a screen's edge at grazing  : {grazing:.2} dB");
    assert!(!occludes(0.25, lambda, d1, d2) && occludes(1.25, lambda, d1, d2));
    // The rule's premise: what it drops costs less than what it keeps.
    assert!(at_soldier < at_r1 && at_r1 < at_tank);
}
