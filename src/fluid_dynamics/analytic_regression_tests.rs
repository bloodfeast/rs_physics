//! Review regression tests for the analytic fluid code and the thin-film law
//! (2026-09-30). Findings `FLD-n` (fluid_dynamics.rs, validation.rs) and `FILM-n`
//! (thin_film.rs).
//!
//! Every assertion here is against an independent oracle -- a textbook formula, a
//! published value, a limiting case, a maximum principle or a brute-force integral --
//! and never against the code's own output. Fixed findings keep the test that failed on
//! the base commit; open ones are `#[ignore = "known defect <ID>: ..."]` and fail when
//! run with `--ignored`.

use crate::atmosphere::Air;
use crate::fluid_dynamics::*;
use std::f64::consts::PI;

const G: f64 = 9.81;

/// `ρg/3μ` computed from the fluid's own fields, independently of `FilmFlow`.
fn nusselt_mobility(fluid: &Fluid) -> f64 {
    fluid.density * G / (3.0 * fluid.viscosity)
}

// ============================================================================ FLD-1
// NaN and infinity were accepted everywhere: `validate_positive` tested `value <= 0.0`,
// which is false for NaN, so `Fluid::new(NaN, ..)` succeeded and every helper that
// returns `Result` handed back `Ok(NaN)` or `Ok(inf)`.

#[test]
fn review_validators_reject_nan() {
    assert!(validate_positive(f64::NAN, "x").is_err(), "NaN is not positive");
    assert!(validate_non_negative(f64::NAN, "x").is_err(), "NaN is not non-negative");
    // Unchanged behaviour on ordinary values.
    assert!(validate_positive(1e-300, "x").is_ok());
    assert!(validate_non_negative(0.0, "x").is_ok());
    assert!(validate_non_negative(-0.0, "x").is_ok());
}

#[test]
fn review_fluid_new_rejects_non_finite_properties() {
    for (rho, mu) in [
        (f64::NAN, 1e-3),
        (1000.0, f64::NAN),
        (f64::INFINITY, 1e-3),
        (1000.0, f64::INFINITY),
    ] {
        assert!(Fluid::new(rho, mu).is_err(), "Fluid::new({rho}, {mu}) was accepted");
    }
    assert!(Fluid::new(998.0, 1e-3).is_ok());
}

#[test]
fn review_analytic_helpers_never_return_ok_nan_or_infinity() {
    let water = Fluid::water();
    for bad in [f64::NAN, f64::INFINITY] {
        assert!(calculate_drag_force(&water, bad, 1.0, 0.5).is_err());
        assert!(calculate_drag_force(&water, 1.0, bad, 0.5).is_err());
        assert!(calculate_drag_force(&water, 1.0, 1.0, bad).is_err());
        assert!(calculate_reynolds_number(&water, bad, 0.1).is_err());
        assert!(calculate_reynolds_number(&water, 1.0, bad).is_err());
        assert!(calculate_buoyant_force(&water, bad, G).is_err());
        assert!(calculate_buoyant_force(&water, 1.0, bad).is_err());
        assert!(calculate_pressure_drop(&water, bad, 0.05, 2.0, 0.02).is_err());
        assert!(calculate_pressure_drop(&water, 10.0, bad, 2.0, 0.02).is_err());
        assert!(calculate_pressure_drop(&water, 10.0, 0.05, bad, 0.02).is_err());
        assert!(calculate_pressure_drop(&water, 10.0, 0.05, 2.0, bad).is_err());
        assert!(thermal_buoyancy(bad, 290.0, G).is_err());
        assert!(thermal_buoyancy(600.0, bad, G).is_err());
        assert!(thermal_buoyancy(600.0, 290.0, bad).is_err());
    }

    // `Fluid`'s fields are public, so a struct literal walks around `Fluid::new`. The
    // helpers return `Result`, so they must say so rather than return `Ok(NaN)`.
    let poisoned = Fluid { density: f64::NAN, viscosity: 1e-3 };
    assert!(calculate_drag_force(&poisoned, 1.0, 1.0, 0.5).is_err());
    assert!(calculate_buoyant_force(&poisoned, 1.0, G).is_err());
    assert!(calculate_reynolds_number(&poisoned, 1.0, 0.1).is_err());
    assert!(calculate_pressure_drop(&poisoned, 10.0, 0.05, 2.0, 0.02).is_err());
    // Zero viscosity made the Reynolds number `Ok(inf)`.
    let inviscid = Fluid { density: 1000.0, viscosity: 0.0 };
    assert!(calculate_reynolds_number(&inviscid, 1.0, 0.1).is_err());
    // Buoyancy and drag do not use viscosity, and still accept an inviscid literal.
    assert!(calculate_buoyant_force(&inviscid, 1.0, G).is_ok());
    assert!(calculate_drag_force(&inviscid, 1.0, 1.0, 0.5).is_ok());
}

// ============================================================================ FLD-2

/// Limiting case: drag `½ρv²C_dA`, Reynolds number `ρvL/μ` and buoyancy `ρVg` all go to
/// zero continuously as their argument does. A body at rest in still water -- the most
/// common state a river solver hands these functions -- used to get an `Err`, while
/// `1e-150` m/s got `Ok(≈0)`. Zero is now exactly `Ok(0.0)`; negative, NaN and ±∞ are
/// still refused.
#[test]
fn review_drag_reynolds_and_buoyancy_vanish_continuously_at_rest() {
    let water = Fluid::water();
    let tiny = calculate_drag_force(&water, 1e-150, 1.0, 0.47).unwrap();
    assert!(tiny < 1e-290, "the limit from above is zero");
    assert!(calculate_reynolds_number(&water, 1e-150, 0.1).unwrap() < 1e-140);
    assert!(calculate_buoyant_force(&water, 1e-150, G).unwrap() < 1e-140);

    for zero in [0.0, -0.0] {
        for got in [
            calculate_drag_force(&water, zero, 1.0, 0.47),
            calculate_reynolds_number(&water, zero, 0.1),
            calculate_buoyant_force(&water, zero, G),
        ] {
            let got = got.expect("zero is a state, not an error");
            assert_eq!(got.to_bits(), 0.0f64.to_bits(), "exactly +0.0 for {zero:?}, got {got:?}");
        }
    }
    for bad in [-1e-300, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(calculate_drag_force(&water, bad, 1.0, 0.47).is_err(), "drag at {bad}");
        assert!(calculate_reynolds_number(&water, bad, 0.1).is_err(), "Re at {bad}");
        assert!(calculate_buoyant_force(&water, bad, G).is_err(), "buoyancy at {bad}");
    }
    // Zero speed does not excuse the other arguments.
    assert!(calculate_drag_force(&water, 0.0, -1.0, 0.47).is_err());
    assert!(calculate_reynolds_number(&water, 0.0, f64::NAN).is_err());
    assert!(calculate_buoyant_force(&Fluid { density: f64::NAN, viscosity: 1e-3 }, 0.0, G).is_err());
}

// ============================================================================ FLD-4

/// Glycerol at 20 °C is 1.412 Pa·s: Segur & Oberstar, Ind. Eng. Chem. 43 (1951) 2117,
/// and the CRC Handbook. Cheng's correlation (Ind. Eng. Chem. Res. 47, 2008),
/// `μ = 12.1·exp((T − 1233)·T / (9900 + 70T))` Pa·s with T in °C, is evaluated here as a
/// second oracle and gives 1.414. The constant used to be 1.5 (+6.2%), which pure
/// glycerol reaches near 19.4 °C.
#[test]
fn review_glycerin_viscosity_matches_published_value_at_20c() {
    let t = 20.0f64;
    let cheng = 12.1 * ((t - 1233.0) * t / (9900.0 + 70.0 * t)).exp();
    let mu = Fluid::glycerin().viscosity;
    for (source, published) in [("Segur & Oberstar / CRC", 1.412), ("Cheng 2008", cheng)] {
        assert!(
            (mu - published).abs() / published < 0.005,
            "glycerin viscosity {mu} Pa·s against {source}'s {published:.4} at 20 °C"
        );
    }
    assert!((Fluid::glycerin().density - 1261.0).abs() < 1.0, "density unchanged");
}

// ============================================================================ FILM-1
// The advertised stable step omitted the levelling (diffusion) half of the flux law.

/// Blood film with a 1% cosine ripple on level ground, stepped at a quarter of the
/// grid's own `max_step`, as the module's tests and docs do.
///
/// Oracles: (1) the **maximum principle** -- `∂h/∂t = ∇·(D(h)∇h)` on level ground can
/// never raise its maximum or lower its minimum; (2) the linearised decay rate of the
/// mode, `exp(-k²·D·t)` with `D = ρgh³/3μ`.
///
/// On the base commit `max_step` here was ~61 s against a diffusive limit of 0.09 s. By
/// the second step the film spanned 1.766–2.240 mm against a starting 1.980–2.020, and
/// over 200 steps it settled into a checkerboard of roughly ±15% that never decayed.
#[test]
fn review_level_film_levels_at_a_quarter_of_max_step() {
    let fluid = Fluid::blood();
    let flow = FilmFlow::new(&fluid, G).unwrap();
    let (w, h, dx) = (32usize, 32usize, 0.05);
    let (h0, ripple) = (0.002, 0.01);
    let k = 2.0 * PI / (w as f64 * dx);
    let mut grid = FilmGrid::new(w, h, dx).unwrap();
    for y in 0..h {
        for x in 0..w {
            // A cosine on cell centres is a no-flux eigenmode of the walled domain; the
            // 1e-6 checkerboard is the seed any real film carries.
            let mode = (k * (x as f64 + 0.5) * dx).cos();
            let seed = if (x + y) % 2 == 0 { 1e-6 } else { -1e-6 };
            grid.set_thickness(x, y, h0 * (1.0 + ripple * mode + seed)).unwrap();
        }
    }
    let extremes = |g: &FilmGrid| {
        let t = g.thickness();
        (
            t.iter().cloned().fold(f64::MAX, f64::min),
            t.iter().cloned().fold(f64::MIN, f64::max),
        )
    };
    let (min0, max0) = extremes(&grid);
    let mut elapsed = 0.0;
    for step in 0..200 {
        let dt = grid.max_step(&flow) * 0.25;
        assert!(dt.is_finite() && dt > 0.0);
        grid.step(&flow, dt);
        elapsed += dt;
        let (lo, hi) = extremes(&grid);
        assert!(
            hi <= max0 * (1.0 + 1e-12) && lo >= min0 * (1.0 - 1e-12),
            "maximum principle violated at step {step}: [{lo:e}, {hi:e}] left [{min0:e}, {max0:e}]"
        );
    }
    let (lo, hi) = extremes(&grid);
    let measured = ((hi - lo) / (max0 - min0)).ln();
    let expected = -k * k * nusselt_mobility(&fluid) * h0.powi(3) * elapsed;
    assert!(
        (measured / expected - 1.0).abs() < 0.1,
        "ripple decay exponent {measured:.4} against the diffusion oracle {expected:.4}"
    );
}

/// The per-face limit against the two textbook bounds, computed here from `ρ`, `g`, `μ`
/// and the angle: the kinematic-wave CFL `dx/c`, with `c = dq/dh` of the inclined film
/// `q = ρg sinθ cos³θ h³/3μ` (so `c = ρg sinθ cos³θ h²/μ`), and forward Euler's
/// `dx²/(2·d·D)` with `d = 2` and the documented bound `D = ρgh³/3μ`. The step must also
/// never exceed the limit set by the true `dq/d(slope) = cos⁴θ·ρgh³/3μ`. On base the
/// gentle-slope cases returned the wave limit alone -- 307.7 s against 0.92 s at a grade
/// of 1e-4, 0.5 mm, 2 cm cells, and up to 3300× over across these cases.
#[test]
fn review_film_max_step_is_the_smaller_of_the_wave_and_diffusion_limits() {
    let fluid = Fluid::blood();
    let flow = FilmFlow::new(&fluid, G).unwrap();
    let m = nusselt_mobility(&fluid);
    for &slope in &[1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0] {
        let theta = f64::atan(slope);
        let (sin, cos) = (theta.sin(), theta.cos());
        for &depth in &[5e-4, 2e-3, 5e-3] {
            for &dx in &[0.02, 0.05, 0.2] {
                let c = fluid.density * G * sin * cos.powi(3) * depth * depth / fluid.viscosity;
                let wave = dx / c;
                let diffusion = dx * dx / (4.0 * m * depth.powi(3));
                let oracle = wave.min(diffusion);
                let got = flow.max_step(slope, depth, dx);
                assert!(
                    (got / oracle - 1.0).abs() < 1e-12,
                    "slope {slope}, h {depth}, dx {dx}: max_step {got:e}, oracle {oracle:e}"
                );
                let true_diffusion = dx * dx / (4.0 * m * cos.powi(4) * depth.powi(3));
                assert!(got <= wave.min(true_diffusion) * (1.0 + 1e-12), "never above the limit");
            }
        }
    }
    // The documented escape for a fresh pool on level ground: the steepest free-surface
    // slope is its edge, depth/dx, and the answer must be the diffusive limit.
    let (depth, dx) = (0.004, 0.05);
    let pool = flow.max_step(depth / dx, depth, dx);
    let limit = dx * dx / (4.0 * m * depth.powi(3));
    assert!((pool / limit - 1.0).abs() < 1e-12, "{pool:e} against {limit:e}");
    // Still infinite where nothing moves, so a settled grid can be detected.
    assert_eq!(flow.max_step(0.0, 0.002, 0.1), f64::INFINITY);
    assert_eq!(flow.max_step(0.1, 0.0, 0.1), f64::INFINITY);
    let yielding = flow.with_yield_stress(BLOOD_YIELD_STRESS).unwrap();
    let held = yielding.arrest_thickness(0.1) * 0.5;
    assert_eq!(yielding.max_step(0.1, held, 0.1), f64::INFINITY);
}

/// Huppert's planar viscous gravity current (J. Fluid Mech. 121, 1982): a fixed volume
/// `A` per unit width spreading on level ground has the similarity solution
/// `h ∝ (ξ_N² − ξ²)^{1/3}`, so its second moment obeys `σ⁵ = K A³ M₂^{5/2} (t + t₀)`
/// with `K = ρg/3μ` and `M₂ = ∫ξ²F/∫F`. Driven by the grid's own `max_step`, as a caller
/// would. On base the rate was within 8% but 24 273 interior maxima grew on the way --
/// impossible for a levelling flow -- which is the assertion that failed.
#[test]
fn review_pool_spreading_follows_huppert_one_fifth_law_with_max_step() {
    let fluid = Fluid::blood();
    let flow = FilmFlow::new(&fluid, G).unwrap();
    let k = nusselt_mobility(&fluid);
    let (n, dx) = (200usize, 0.005);
    let mut grid = FilmGrid::new(n, 1, dx).unwrap();
    let (h0, cells0) = (0.004, 10usize);
    for x in 0..cells0 {
        grid.set_thickness(x, 0, h0).unwrap(); // x = 0 is a wall: the symmetry plane
    }
    let a = h0 * cells0 as f64 * dx;

    // M₂ of the similarity profile F = (3/10)^{1/3}(ξ_N² − ξ²)^{1/3}, by quadrature.
    let xi_n = 1.4112; // Huppert's η_N for a fixed planar volume
    let (mut i0, mut i2) = (0.0, 0.0);
    for j in 0..20_000 {
        let xi = (j as f64 + 0.5) / 20_000.0 * xi_n;
        let f = (0.3 * (xi_n * xi_n - xi * xi)).cbrt();
        i0 += f;
        i2 += xi * xi * f;
    }
    let predicted_rate = k * a.powi(3) * (i2 / i0).powf(2.5);

    let sigma = |g: &FilmGrid| {
        let t = g.thickness();
        let m0: f64 = t.iter().sum();
        let m2: f64 = t
            .iter()
            .enumerate()
            .map(|(i, h)| ((i as f64 + 0.5) * dx).powi(2) * h)
            .sum();
        (m2 / m0).sqrt()
    };
    let (mut t, mut grown_peaks, mut samples) = (0.0, 0usize, Vec::new());
    let mut before = grid.thickness().to_vec();
    for target in [0.2, 0.3] {
        while (grid.thickness().iter().rposition(|&h| h > 1e-7).unwrap() as f64) * dx < target {
            let dt = grid.max_step(&flow) * 0.25;
            grid.step(&flow, dt);
            t += dt;
            let now = grid.thickness();
            for i in 1..n - 1 {
                if now[i] > before[i] && now[i] > now[i - 1] && now[i] > now[i + 1] {
                    grown_peaks += 1;
                }
            }
            before.copy_from_slice(now);
        }
        samples.push((t, sigma(&grid)));
    }
    assert_eq!(grown_peaks, 0, "a levelling flow grew {grown_peaks} interior maxima");
    let rate = (samples[1].1.powi(5) - samples[0].1.powi(5)) / (samples[1].0 - samples[0].0);
    assert!(
        (rate / predicted_rate - 1.0).abs() < 0.1,
        "dσ⁵/dt = {rate:e} against Huppert's {predicted_rate:e}"
    );
}

// ============================================================================ FILM-2
// The film law used `tanθ` and vertical depth in a formula written for `sinθ` and normal
// thickness.

/// Nusselt's inclined film, `ρg sinθ h_n³ / 3μ`, in the grid's own variables: vertical
/// depth `H = h_n / cosθ` and gradient `tanθ`. The oracle is built from the angle.
fn inclined_nusselt_flux(fluid: &Fluid, grade: f64, depth: f64) -> f64 {
    let theta = grade.atan();
    fluid.density * G * theta.sin() * (depth * theta.cos()).powi(3) / (3.0 * fluid.viscosity)
}

/// A uniform film on an incline, depth measured vertically as `FilmGrid` stores it
/// (volume per horizontal area): normal thickness `H cosθ`, and the Nusselt flux
/// `q = ρg sinθ (H cosθ)³ / 3μ` crosses any section. The base code evaluated
/// `ρg tanθ H³ / 3μ`, `1/cos⁴θ` too large: +2% at a grade of 0.1, +8.2% at 0.2, +19% at
/// 0.3, 4× at 45° and unbounded at vertical -- and this test failed at grade 0.2.
#[test]
fn review_film_flux_on_a_steep_grade_matches_the_inclined_nusselt_film() {
    let fluid = Fluid::blood();
    let flow = FilmFlow::new(&fluid, G).unwrap();
    let depth = 0.002;
    let grades = [0.01f64, 0.1, 0.2, 0.3, 1.0, 3f64.sqrt(), 5.0, -0.3];
    let mut batch = [0.0; 8];
    flow.flux_batch(&grades, &[depth; 8], &mut batch).unwrap();
    for (i, &grade) in grades.iter().enumerate() {
        let exact = inclined_nusselt_flux(&fluid, grade, depth);
        let got = flow.flux(grade, depth);
        assert!(
            (got / exact - 1.0).abs() < 1e-12,
            "grade {grade}: flux {got:e} against the inclined Nusselt film {exact:e} ({:+.2}%)",
            (got / exact - 1.0) * 100.0
        );
        assert_eq!(batch[i], got, "flux_batch and flux must stay the same law");
        assert_eq!(flow.flux_on_bed(grade, grade, depth), got, "uniform film: bed = drive");
        // Depth-averaged horizontal speed is q/H.
        assert!((flow.velocity(grade, depth) / (exact / depth) - 1.0).abs() < 1e-12);
    }
    // Towards vertical the liquid over a unit of horizontal area has nowhere to be.
    assert_eq!(flow.flux(f64::INFINITY, depth), 0.0);
    assert!(flow.flux(1e6, depth) < 1e-15 * flow.flux(1.0, depth));
    // Level ground is unchanged bit for bit: the pre-FILM-2 formula, m·s·h³.
    for s in [1e-4, -0.02, 0.07] {
        let old = flow.mobility() * s * (depth * depth * depth);
        assert_eq!(flow.flux_on_bed(0.0, s, depth).to_bits(), old.to_bits());
    }
}

/// `wave_speed` is `dq/dh` at fixed slope, checked by a central difference of the
/// inclined Bingham film written in normal coordinates, `q = (ρg sinθ h_n³/3μ)(1 −
/// 1.5X + 0.5X³)`, `X = τ_y/(ρg sinθ h_n)`, `h_n = H cosθ` -- none of the code's
/// `1/(1 + s²)` algebra. On base the wave speed was the derivative of the tan law.
#[test]
fn review_wave_speed_is_the_derivative_of_the_inclined_flux() {
    let fluid = Fluid::blood();
    let tau_y = 0.05;
    let exact = |grade: f64, depth: f64, yield_stress: f64| {
        let theta = grade.atan();
        let h_n = depth * theta.cos();
        let driving = fluid.density * G * theta.sin();
        let x = yield_stress / (driving * h_n);
        if x >= 1.0 {
            return 0.0;
        }
        driving * h_n.powi(3) / (3.0 * fluid.viscosity) * (1.0 - 1.5 * x + 0.5 * x.powi(3))
    };
    for (yield_stress, flow) in [
        (0.0, FilmFlow::new(&fluid, G).unwrap()),
        (tau_y, FilmFlow::new(&fluid, G).unwrap().with_yield_stress(tau_y).unwrap()),
    ] {
        for &grade in &[0.1, 0.4, 1.0, 2.0] {
            for &depth in &[3e-4, 1e-3, 4e-3] {
                let d = depth * 1e-5;
                let oracle = (exact(grade, depth + d, yield_stress)
                    - exact(grade, depth - d, yield_stress))
                    / (2.0 * d);
                let got = flow.wave_speed(grade, depth);
                assert!(
                    (got / oracle - 1.0).abs() < 1e-6,
                    "τ_y {yield_stress}, grade {grade}, h {depth}: c {got:e}, dq/dh {oracle:e}"
                );
            }
        }
    }
}

/// The grid's face flux is `FilmFlow::flux_on_bed`, bit for bit, in both directions: one
/// small step of a two-cell grid on a 0.3 grade moves exactly `q·dt/dx` of depth.
#[test]
fn review_grid_face_flux_is_flux_on_bed() {
    let fluid = Fluid::blood();
    let (dx, grade, dt) = (0.1, 0.3, 1e-4);
    let (upper, lower) = (0.003, 0.001);
    for flow in [
        FilmFlow::new(&fluid, G).unwrap(),
        FilmFlow::new(&fluid, G).unwrap().with_yield_stress(0.05).unwrap(),
    ] {
        for (w, h) in [(2usize, 1usize), (1, 2)] {
            let mut grid = FilmGrid::new(w, h, dx).unwrap();
            grid.set_ground_from(&[0.0, -grade * dx]).unwrap();
            grid.set_thickness(0, 0, upper).unwrap();
            let (x1, y1) = if w == 2 { (1, 0) } else { (0, 1) };
            grid.set_thickness(x1, y1, lower).unwrap();

            let inv_dx = 1.0 / dx;
            let fall = 0.0 - (-grade * dx);
            let q = flow.flux_on_bed(fall * inv_dx, (fall + (upper - lower)) * inv_dx, upper);
            assert!(q > 0.0 && q * dt * inv_dx < upper, "the limiter must stay out of it");
            grid.step(&flow, dt);
            let scale = dt * inv_dx;
            assert_eq!(grid.thickness_at(0, 0).unwrap(), upper + (-q) * scale, "{w}x{h}");
            assert_eq!(grid.thickness_at(x1, y1).unwrap(), lower + q * scale, "{w}x{h}");
        }
    }
}

// ============================================================ checked and found correct

/// The Bingham plug factor `1 − 1.5X + 0.5X³` and the arrest thickness, on an incline,
/// against a brute-force integral of the Bingham velocity profile across the *normal*
/// thickness `h_n = H cosθ`: `μ du/dη = max(ρ g sinθ (h_n − η) − τ_y, 0)`. The arrest
/// depth is where the wall stress `ρ g sinθ h_n` reaches `τ_y`, converted to vertical
/// depth. (Until FILM-2 this ran at one gentle grade with `sinθ ≈ tanθ`.)
#[test]
fn review_bingham_plug_factor_matches_the_integrated_velocity_profile() {
    let fluid = Fluid::blood();
    let tau_y = 0.05; // ten times blood's, so X spans the whole range cheaply
    let flow = FilmFlow::new(&fluid, G).unwrap().with_yield_stress(tau_y).unwrap();
    for &grade in &[0.1f64, 0.5, 1.0, 3.0] {
        let theta = grade.atan();
        let driving = fluid.density * G * theta.sin();
        let arrest = tau_y / (driving * theta.cos());
        let got_arrest = flow.arrest_thickness(grade);
        assert!((got_arrest / arrest - 1.0).abs() < 1e-12, "grade {grade}: {got_arrest:e} vs {arrest:e}");

        for &factor in &[0.9, 1.05, 1.2, 2.0, 20.0] {
            let depth = arrest * factor;
            let h_n = depth * theta.cos();
            let n = 200_000;
            let d_eta = h_n / n as f64;
            let (mut u, mut q) = (0.0, 0.0);
            for i in 0..n {
                let eta = (i as f64 + 0.5) * d_eta;
                let du = (driving * (h_n - eta) - tau_y).max(0.0) / fluid.viscosity * d_eta;
                q += (u + 0.5 * du) * d_eta;
                u += du;
            }
            let got = flow.flux(grade, depth);
            if q == 0.0 {
                assert_eq!(got, 0.0, "grade {grade}: held below the arrest depth, h = {depth:e}");
            } else {
                assert!(
                    (got / q - 1.0).abs() < 1e-6,
                    "grade {grade}, h = {factor}·arrest: flux {got:e}, integral {q:e}"
                );
            }
        }
    }
}

/// Huppert (Nature 300, 1982): a fixed planar volume `A` per unit width on an incline
/// runs as a kinematic wave with front `ξ_N³ = (9A²g sinθ/4ν)·t` along the slope. Its
/// horizontal position is `x_N = ξ_N cosθ`, so
/// `x_N³ = (27/4)·A²·(ρg/3μ)·sinθ·cos³θ·t` -- the `t^{1/3}` law in the grid's
/// coordinates. On the pre-FILM-2 law the rate was `1/cos⁴θ` = 8% fast at this grade.
#[test]
fn review_slope_current_follows_huppert_one_third_law() {
    let fluid = Fluid::blood();
    let flow = FilmFlow::new(&fluid, G).unwrap();
    let theta = f64::atan(0.2);
    let k = nusselt_mobility(&fluid) * theta.sin() * theta.cos().powi(3); // ρg sinθ cos³θ / 3μ
    let (n, dx, s) = (400usize, 0.005, 0.2);
    let mut grid = FilmGrid::new(n, 1, dx).unwrap();
    let bed: Vec<f64> = (0..n).map(|i| -s * i as f64 * dx).collect();
    grid.set_ground_from(&bed).unwrap();
    let (h0, cells0) = (0.002, 10usize);
    for x in 0..cells0 {
        grid.set_thickness(x, 0, h0).unwrap();
    }
    let a = h0 * cells0 as f64 * dx;
    let front_rate = 27.0 / 4.0 * a * a * k;
    let front = |g: &FilmGrid| (g.thickness().iter().rposition(|&h| h > 1e-9).unwrap() as f64) * dx;
    let mut t = 0.0;
    let mut samples = Vec::new();
    for target in [0.9, 1.5] {
        while front(&grid) < target {
            let dt = grid.max_step(&flow) * 0.25;
            grid.step(&flow, dt);
            t += dt;
        }
        samples.push((t, front(&grid)));
    }
    let rate = (samples[1].1.powi(3) - samples[0].1.powi(3)) / (samples[1].0 - samples[0].0);
    assert!(
        (rate / front_rate - 1.0).abs() < 0.03,
        "d(x_N³)/dt = {rate:e} against Huppert's {front_rate:e}"
    );
}

/// Stokes: with `C_d = 24/Re` on the frontal area, `½ρv²C_dA = 6πμrv`; and the helpers
/// compose to the Stokes terminal velocity `2(ρ_s − ρ_f)g r²/9μ` for a 1 mm glass bead
/// in glycerin, where Re ≈ 0.01 and the law applies.
#[test]
fn review_drag_and_buoyancy_reproduce_stokes_law() {
    let glycerin = Fluid::glycerin();
    let (r, rho_s) = (0.5e-3, 2500.0);
    let area = PI * r * r;
    let volume = 4.0 / 3.0 * PI * r.powi(3);
    let v_t = 2.0 * (rho_s - glycerin.density) * G * r * r / (9.0 * glycerin.viscosity);
    let re = calculate_reynolds_number(&glycerin, v_t, 2.0 * r).unwrap();
    assert!(re < 0.1, "Stokes regime, Re = {re}");
    let drag = calculate_drag_force(&glycerin, v_t, area, 24.0 / re).unwrap();
    let stokes = 6.0 * PI * glycerin.viscosity * r * v_t;
    assert!((drag / stokes - 1.0).abs() < 1e-12);
    let buoyancy = calculate_buoyant_force(&glycerin, volume, G).unwrap();
    let weight = rho_s * volume * G;
    assert!(((drag + buoyancy) / weight - 1.0).abs() < 1e-12, "forces balance at v_t");
}

/// Darcy–Weisbach with the laminar friction factor `f = 64/Re` is Hagen–Poiseuille,
/// `ΔP = 32μLv/D²`.
#[test]
fn review_laminar_pressure_drop_is_hagen_poiseuille() {
    let oil = Fluid::oil();
    let (length, diameter, v) = (2.0, 0.01, 0.1);
    let re = calculate_reynolds_number(&oil, v, diameter).unwrap();
    assert!(re < 2300.0);
    let dp = calculate_pressure_drop(&oil, length, diameter, v, 64.0 / re).unwrap();
    let poiseuille = 32.0 * oil.viscosity * length * v / (diameter * diameter);
    assert!((dp / poiseuille - 1.0).abs() < 1e-12);
}

/// Preset properties against published tables at the stated temperatures. Water:
/// IAPWS-95 / IAPWS-2008 at 20 °C, 998.21 kg/m³ and 1.0016 mPa·s. Seawater: ITTC 2011
/// (S = 35 g/kg, 20 °C), 1024.76 kg/m³ and 1.0764 mPa·s. Blood: the Casson fit
/// `(√(τ_y/γ̇) + √μ_c)²` at the documented 300 s⁻¹. Air: ICAO sea level, 1.2250 kg/m³
/// and 1.7894e-5 Pa·s.
#[test]
fn review_fluid_presets_match_published_tables() {
    let within = |got: f64, table: f64, tol: f64| (got / table - 1.0).abs() < tol;
    let water = Fluid::water();
    assert!(within(water.density, 998.21, 5e-4) && within(water.viscosity, 1.0016e-3, 5e-3));
    let sea = Fluid::seawater();
    assert!(within(sea.density, 1024.76, 5e-4) && within(sea.viscosity, 1.0764e-3, 5e-3));
    let casson = ((BLOOD_YIELD_STRESS / 300.0).sqrt() + BLOOD_CASSON_VISCOSITY.sqrt()).powi(2);
    assert!(within(Fluid::blood().viscosity, casson, 1e-3));
    let air = Fluid::from_air(&Air::sea_level());
    assert!(within(air.density, 1.2250, 1e-4) && within(air.viscosity, 1.7894e-5, 1e-3));
}

/// Puddle depth `2κ⁻¹ sin(θ/2)` (de Gennes, Brochard-Wyart & Quéré, *Capillarity and
/// Wetting Phenomena*, §2.4): water at 90° stands `√2` capillary lengths deep, 3.86 mm.
#[test]
fn review_puddle_depth_matches_de_gennes() {
    let (gamma, rho) = (0.0728, 998.2);
    let capillary_length = (gamma / (rho * G)).sqrt();
    let h = puddle_depth(gamma, rho, G, PI / 2.0).unwrap();
    assert!((h / (2f64.sqrt() * capillary_length) - 1.0).abs() < 1e-12);
    assert!((h - 3.86e-3).abs() < 0.01e-3, "h = {h}");
}
