//! Review regression tests for the analytic fluid code and the thin-film law
//! (2026-09-30). Findings `FLD-n` (fluid_dynamics.rs, validation.rs) and `FILM-n`
//! (thin_film.rs).
//!
//! Every assertion here is against an independent oracle — a textbook formula, a
//! published value, a limiting case, a maximum principle or a brute-force integral —
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

// ============================================================================ FLD-2 (open)

/// Limiting case: drag `½ρv²C_dA`, Reynolds number `ρvL/μ` and buoyancy `ρVg` all go to
/// zero continuously as their argument does. A body at rest in still water — the most
/// common state a river solver hands these functions — currently gets an `Err`, while
/// `1e-300` m/s gets `Ok(≈0)`.
#[test]
#[ignore = "known defect FLD-2: drag, Reynolds number and buoyancy return Err at zero speed / zero displaced volume instead of 0"]
fn review_drag_reynolds_and_buoyancy_vanish_continuously_at_rest() {
    let water = Fluid::water();
    let tiny = calculate_drag_force(&water, 1e-150, 1.0, 0.47).unwrap();
    assert!(tiny < 1e-290, "the limit from above is zero");
    assert_eq!(calculate_drag_force(&water, 0.0, 1.0, 0.47).ok(), Some(0.0));
    assert_eq!(calculate_reynolds_number(&water, 0.0, 0.1).ok(), Some(0.0));
    assert_eq!(calculate_buoyant_force(&water, 0.0, G).ok(), Some(0.0));
}

// ============================================================================ FLD-4 (open)

/// Glycerol at 20 °C is 1.412 Pa·s: Segur & Oberstar, Ind. Eng. Chem. 43 (1951) 2117,
/// and the CRC Handbook; Cheng's correlation (Ind. Eng. Chem. Res. 47, 2008) gives
/// 1.414. `Fluid::glycerin()` documents "at 20°C" and carries 1.5 (+6.2%), a figure
/// pure glycerol reaches only at about 18.5 °C; any water content lowers it further.
#[test]
#[ignore = "known defect FLD-4: Fluid::glycerin() viscosity 1.5 Pa·s is 6.2% above the published 1.412 Pa·s at its stated 20 °C"]
fn review_glycerin_viscosity_matches_published_value_at_20c() {
    let published = 1.412;
    let mu = Fluid::glycerin().viscosity;
    assert!(
        (mu - published).abs() / published < 0.02,
        "glycerin viscosity {mu} Pa·s against the published {published} at 20 °C"
    );
}

// ============================================================================ FILM-1
// The advertised stable step omitted the levelling (diffusion) half of the flux law.

/// Blood film with a 1% cosine ripple on level ground, stepped at a quarter of the
/// grid's own `max_step`, as the module's tests and docs do.
///
/// Oracles: (1) the **maximum principle** — `∂h/∂t = ∇·(D(h)∇h)` on level ground can
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

/// The per-face limit against the two textbook bounds, computed here from `ρ`, `g`, `μ`:
/// the kinematic-wave CFL `dx/c` with `c = dq/dh = 3·(ρg/3μ)·s·h²`, and forward Euler's
/// `dx²/(2·d·D)` with `d = 2`, `D = ρgh³/3μ`. On base the gentle-slope cases returned the
/// wave limit alone — 307.7 s against 0.92 s at a grade of 1e-4, 0.5 mm, 2 cm cells, and
/// up to 3300× over across these cases.
#[test]
fn review_film_max_step_is_the_smaller_of_the_wave_and_diffusion_limits() {
    let fluid = Fluid::blood();
    let flow = FilmFlow::new(&fluid, G).unwrap();
    let m = nusselt_mobility(&fluid);
    for &slope in &[1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0] {
        for &depth in &[5e-4, 2e-3, 5e-3] {
            for &dx in &[0.02, 0.05, 0.2] {
                let wave = dx / (3.0 * m * slope * depth * depth);
                let diffusion = dx * dx / (4.0 * m * depth.powi(3));
                let oracle = wave.min(diffusion);
                let got = flow.max_step(slope, depth, dx);
                assert!(
                    (got / oracle - 1.0).abs() < 1e-12,
                    "slope {slope}, h {depth}, dx {dx}: max_step {got:e}, oracle {oracle:e}"
                );
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
/// would. On base the rate was within 8% but 24 273 interior maxima grew on the way —
/// impossible for a levelling flow — which is the assertion that failed.
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

// ============================================================================ FILM-2 (open)

/// A uniform film on an incline, depth measured vertically as `FilmGrid` stores it
/// (volume per horizontal area): normal thickness `H cosθ`, and the Nusselt flux
/// `q = ρg sinθ (H cosθ)³ / 3μ` crosses any section. The code evaluates
/// `ρg tanθ H³ / 3μ`, which is `1/cos⁴θ` too large: +2% at a grade of 0.1, +8% at 0.2,
/// **+19% at 0.3**, 4× at 45° and unbounded at vertical. The doc calls the tan/sin
/// difference "far below the error in knowing the depth".
#[test]
#[ignore = "known defect FILM-2: tan-slope film flux overstates the inclined Nusselt flux by 1/cos^4(theta): +19% at grade 0.3, 4x at 45 deg"]
fn review_film_flux_on_a_steep_grade_matches_the_inclined_nusselt_film() {
    let fluid = Fluid::blood();
    let flow = FilmFlow::new(&fluid, G).unwrap();
    let depth = 0.002;
    for &grade in &[0.1f64, 0.2, 0.3] {
        let theta = grade.atan();
        let exact = fluid.density * G * theta.sin() * (depth * theta.cos()).powi(3)
            / (3.0 * fluid.viscosity);
        let got = flow.flux(grade, depth);
        assert!(
            (got / exact - 1.0).abs() < 0.05,
            "grade {grade}: flux {got:e} against the inclined Nusselt film {exact:e} ({:+.1}%)",
            (got / exact - 1.0) * 100.0
        );
    }
}

// ============================================================ checked and found correct

/// The Bingham plug factor `1 − 1.5X + 0.5X³` against a brute-force integral of the
/// Bingham velocity profile `μ du/dz = max(ρgs(h − z) − τ_y, 0)` across the depth.
#[test]
fn review_bingham_plug_factor_matches_the_integrated_velocity_profile() {
    let fluid = Fluid::blood();
    let tau_y = 0.05; // ten times blood's, so X spans the whole range cheaply
    let flow = FilmFlow::new(&fluid, G).unwrap().with_yield_stress(tau_y).unwrap();
    let slope = 0.1;
    let rho_g_s = fluid.density * G * slope;
    for &depth in &[5.0e-5, 5.5e-5, 6.0e-5, 1.0e-4, 1.0e-3] {
        let n = 20_000;
        let dz = depth / n as f64;
        let (mut u, mut q) = (0.0, 0.0);
        for i in 0..n {
            let z = (i as f64 + 0.5) * dz;
            let du = (rho_g_s * (depth - z) - tau_y).max(0.0) / fluid.viscosity * dz;
            q += (u + 0.5 * du) * dz;
            u += du;
        }
        let got = flow.flux(slope, depth);
        if q == 0.0 {
            assert_eq!(got, 0.0, "below the arrest thickness at h = {depth}");
        } else {
            assert!((got / q - 1.0).abs() < 1e-6, "h {depth}: flux {got:e}, integral {q:e}");
        }
    }
}

/// Huppert (Nature 300, 1982): a fixed planar volume on an incline runs as a kinematic
/// wave, `x_N³ = (27/4)·A²·(ρg/3μ)·s·t`, the `t^{1/3}` law. Front and centroid
/// (`0.6 x_N`) rates both.
#[test]
fn review_slope_current_follows_huppert_one_third_law() {
    let fluid = Fluid::blood();
    let flow = FilmFlow::new(&fluid, G).unwrap();
    let k = nusselt_mobility(&fluid);
    let (n, dx, s) = (400usize, 0.005, 0.2);
    let mut grid = FilmGrid::new(n, 1, dx).unwrap();
    let bed: Vec<f64> = (0..n).map(|i| -s * i as f64 * dx).collect();
    grid.set_ground_from(&bed).unwrap();
    let (h0, cells0) = (0.002, 10usize);
    for x in 0..cells0 {
        grid.set_thickness(x, 0, h0).unwrap();
    }
    let a = h0 * cells0 as f64 * dx;
    let front_rate = 27.0 / 4.0 * a * a * k * s;
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
