//! Regression tests for the Eulerian grid solvers, `FluidGrid` and `FluidGrid3D`.
//!
//! Each test checks the solver against an oracle that does not come from the solver:
//! an analytic decay rate, a conservation law, the definition of a boundary
//! condition, a many-iteration reference solve, or a sum written out here. Findings
//! are numbered `GRID-n`; all thirteen are fixed. A defect found later and left open
//! should be pinned by an `#[ignore = "known defect GRID-n: ..."]` test, run with
//! `cargo test --lib --features fluid_simulation -- --ignored grid_regression`.
//!
//! ## The grid's unit of length
//!
//! Both grids measure length in *domain widths*: the cell size is `h = 1 / width` on
//! every axis, so velocities are in widths per second (`advect` moves a quantity
//! `dt * width * v` cells) and viscosity and diffusion are in widths² per second.
//! The outermost ring of cells is a ghost layer that `set_boundaries` overwrites; the
//! fluid occupies cells `1..n-1` on each axis, so it is `(n - 2) * h` long there, with
//! walls half a cell outside the first and last fluid cells. A cell with index `i`
//! is centred `(i - 0.5) * h` from the wall. The grids refuse sources on the ghost
//! layer, so the helpers below write fluid cells only.

use super::{FluidGrid, FluidGrid3D, PressureSolver, SolverConfig, SolverType, WallCondition};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Distance of cell `i`'s centre from the low wall, in widths.
fn from_wall(i: usize, h: f64) -> f64 {
    (i as f64 - 0.5) * h
}

/// Length of the fluid region along an axis of `n` cells, in widths.
fn fluid_length(n: usize, h: f64) -> f64 {
    (n - 2) as f64 * h
}

/// Per-step amplitude factor of the lowest cosine mode along an axis of `n` cells
/// under implicit-Euler diffusion with coefficient `nu`. This is the exact answer of
/// the discrete scheme: the mode is an eigenvector of the 5- or 7-point Laplacian with
/// the ghost-copy (Neumann) boundary, with eigenvalue `(2 - 2cos(pi/(n-2))) / h^2`.
fn discrete_step_factor(lambda: f64, nu: f64, dt: f64) -> f64 {
    1.0 / (1.0 + dt * nu * lambda)
}

fn lowest_mode_eigenvalue(n: usize, h: f64) -> f64 {
    (2.0 - 2.0 * (PI / (n - 2) as f64).cos()) / (h * h)
}

fn set_velocity_2d(g: &mut FluidGrid, f: impl Fn(usize, usize) -> (f64, f64)) {
    for j in 1..g.get_height() - 1 {
        for i in 1..g.get_width() - 1 {
            let (u0, v0) = g.get_velocity(i, j).unwrap();
            let (u, v) = f(i, j);
            g.add_velocity(i, j, u - u0, v - v0).unwrap();
        }
    }
}

fn set_density_2d(g: &mut FluidGrid, f: impl Fn(usize, usize) -> f64) {
    for j in 1..g.get_height() - 1 {
        for i in 1..g.get_width() - 1 {
            let d0 = g.get_density(i, j).unwrap();
            g.add_density(i, j, f(i, j) - d0).unwrap();
        }
    }
}

fn set_velocity_3d(g: &mut FluidGrid3D, f: impl Fn(usize, usize, usize) -> (f64, f64, f64)) {
    for k in 1..g.get_depth() - 1 {
        for j in 1..g.get_height() - 1 {
            for i in 1..g.get_width() - 1 {
                let (u0, v0, w0) = g.get_velocity(i, j, k).unwrap();
                let (u, v, w) = f(i, j, k);
                g.add_velocity(i, j, k, u - u0, v - v0, w - w0).unwrap();
            }
        }
    }
}

fn set_density_3d(g: &mut FluidGrid3D, f: impl Fn(usize, usize, usize) -> f64) {
    for k in 1..g.get_depth() - 1 {
        for j in 1..g.get_height() - 1 {
            for i in 1..g.get_width() - 1 {
                let d0 = g.get_density(i, j, k).unwrap();
                g.add_density(i, j, k, f(i, j, k) - d0).unwrap();
            }
        }
    }
}

/// Sum of |v|² over the fluid cells only.
fn fluid_speed_sq_2d(g: &FluidGrid) -> f64 {
    let mut s = 0.0;
    for j in 1..g.get_height() - 1 {
        for i in 1..g.get_width() - 1 {
            let (u, v) = g.get_velocity(i, j).unwrap();
            s += u * u + v * v;
        }
    }
    s
}

fn fluid_mass_2d(g: &FluidGrid) -> f64 {
    let mut s = 0.0;
    for j in 1..g.get_height() - 1 {
        for i in 1..g.get_width() - 1 {
            s += g.get_density(i, j).unwrap();
        }
    }
    s
}

fn fluid_mass_3d(g: &FluidGrid3D) -> f64 {
    let mut s = 0.0;
    for k in 1..g.get_depth() - 1 {
        for j in 1..g.get_height() - 1 {
            for i in 1..g.get_width() - 1 {
                s += g.get_density(i, j, k).unwrap();
            }
        }
    }
    s
}

/// The scheme's own central-difference gradient of `phi`, sampled at a cell.
/// A projection must remove a field like this entirely, up to the known O(h²)
/// mismatch between the wide Laplacian `D·G` and the compact one the pressure
/// solve inverts.
fn discrete_gradient_2d(w: usize, h: usize) -> impl Fn(usize, usize) -> (f64, f64) {
    let dx = 1.0 / w as f64;
    let (lx, ly) = (fluid_length(w, dx), fluid_length(h, dx));
    let phi = move |i: f64, j: f64| {
        ((PI * (i - 0.5) * dx / lx).cos()) * ((PI * (j - 0.5) * dx / ly).cos())
    };
    move |i, j| {
        let (x, y) = (i as f64, j as f64);
        (
            (phi(x + 1.0, y) - phi(x - 1.0, y)) / (2.0 * dx),
            (phi(x, y + 1.0) - phi(x, y - 1.0)) / (2.0 * dx),
        )
    }
}

/// Fraction of `(1 - r)^4` that two converged projections leave of a discrete
/// gradient of the lowest (1,1) cosine mode, where `r` is the ratio of the wide to
/// the compact Laplacian's eigenvalue for that mode. This is what the collocated
/// scheme can promise; anything much larger is a projection that does not project.
fn collocated_residual_2d(w: usize, h: usize) -> f64 {
    let (tx, ty) = (PI / (w - 2) as f64, PI / (h - 2) as f64);
    let wide = tx.sin().powi(2) + ty.sin().powi(2);
    let compact = 4.0 * (tx / 2.0).sin().powi(2) + 4.0 * (ty / 2.0).sin().powi(2);
    (1.0 - wide / compact).powi(4)
}

/// Taylor-Green cell `psi = sin(pi x/L) sin(pi y/L)` on a square grid of `n` cells.
/// With the ghost-cell walls this is an exact eigenmode of the discrete Stokes
/// operator, discretely divergence-free, and it decays as `exp(-nu k^2 t)`.
fn taylor_green_2d(n: usize, amp: f64) -> impl Fn(usize, usize) -> (f64, f64) {
    let dx = 1.0 / n as f64;
    let l = fluid_length(n, dx);
    move |i, j| {
        let (x, y) = (from_wall(i, dx), from_wall(j, dx));
        (
            amp * (PI / l) * (PI * x / l).sin() * (PI * y / l).cos(),
            -amp * (PI / l) * (PI * x / l).cos() * (PI * y / l).sin(),
        )
    }
}

// ---------------------------------------------------------------------------
// GRID-1: 3D viscosity and diffusion scaled with the cell count, not the cell size
// ---------------------------------------------------------------------------

/// A cosine of density along one axis decays at `exp(-kappa k^2 t)`, whatever the
/// resolution or the axis. The base scaled the implicit coefficient by
/// `width * height * depth` instead of `1 / h^2 = width^2`, so it diffused N times
/// too fast on an N³ grid: 9.9× at 10³, 17.9× at 18³, 33.9× at 34³.
#[test]
fn review_3d_diffusion_matches_the_analytic_rate_at_any_resolution_and_axis() {
    let (kappa, dt, steps) = (1e-3, 0.01, 20);
    // (width, height, depth, axis the mode varies along)
    for &(w, h, d, axis) in &[(10, 10, 10, 0usize), (18, 18, 18, 0), (18, 10, 14, 1), (18, 10, 14, 2)] {
        let dx = 1.0 / w as f64;
        let n_axis = [w, h, d][axis];
        let l = fluid_length(n_axis, dx);
        let mode = move |i: usize, j: usize, k: usize| (PI * from_wall([i, j, k][axis], dx) / l).cos();

        let mut g = FluidGrid3D::with_solver(w, h, d, kappa, 0.0, dt, SolverConfig::new(50)).unwrap();
        set_density_3d(&mut g, |i, j, k| 1.0 + mode(i, j, k));
        for _ in 0..steps {
            g.step();
        }

        let (mut num, mut den) = (0.0, 0.0);
        for k in 1..d - 1 {
            for j in 1..h - 1 {
                for i in 1..w - 1 {
                    let c = mode(i, j, k);
                    num += (g.get_density(i, j, k).unwrap() - 1.0) * c;
                    den += c * c;
                }
            }
        }
        let amplitude = num / den;
        let exact = discrete_step_factor(lowest_mode_eigenvalue(n_axis, dx), kappa, dt).powi(steps);
        let analytic = (-kappa * (PI / l).powi(2) * dt * steps as f64).exp();
        // The discrete eigenvalue is the continuous k² times 1 - theta²/12, theta = pi/(n-2).
        assert!(
            (exact.ln() / analytic.ln() - 1.0).abs() < 0.02,
            "oracle self-check: {exact} vs {analytic}"
        );
        assert!(
            (amplitude - exact).abs() < 1e-7,
            "{w}x{h}x{d}, axis {axis}: amplitude {amplitude:.6}, analytic {analytic:.6} (exact scheme {exact:.6})"
        );
    }
}

/// The same defect in the velocity solve: a Taylor-Green cell (z-uniform) decays at
/// `exp(-nu k^2 t)`. Base: 0.95626 after 0.1 s at 18³ against 0.99750.
#[test]
fn review_3d_viscosity_matches_taylor_green_decay() {
    let (nu, dt, steps, n) = (1e-3, 0.01, 10, 18);
    let dx = 1.0 / n as f64;
    let l = fluid_length(n, dx);
    let tg = taylor_green_2d(n, 1e-6);

    let mut g = FluidGrid3D::with_solver(n, n, n, 0.0, nu, dt, SolverConfig::new(20)).unwrap();
    set_velocity_3d(&mut g, |i, j, _| {
        let (u, v) = tg(i, j);
        (u, v, 0.0)
    });
    for _ in 0..steps {
        g.step();
    }

    let (mut num, mut den) = (0.0, 0.0);
    for k in 1..n - 1 {
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let (u, v) = tg(i, j);
                let (a, b, _) = g.get_velocity(i, j, k).unwrap();
                num += a * u + b * v;
                den += u * u + v * v;
            }
        }
    }
    let amplitude = num / den;
    let exact = discrete_step_factor(2.0 * lowest_mode_eigenvalue(n, dx), nu, dt).powi(steps);
    let analytic = (-nu * 2.0 * (PI / l).powi(2) * dt * steps as f64).exp();
    assert!(
        (amplitude - exact).abs() < 1e-6,
        "amplitude {amplitude:.6}, analytic {analytic:.6} (exact scheme {exact:.6})"
    );
}

// ---------------------------------------------------------------------------
// GRID-2: 2D diffusion on a non-square grid used width * height for 1 / h²
// ---------------------------------------------------------------------------

/// Only square grids had the right coefficient. Base, after 1 s: a 34×18 grid decays
/// at 0.53× (= 18/34) the analytic rate and a 34×66 grid at 1.94× (= 66/34).
#[test]
fn review_2d_diffusion_rate_does_not_depend_on_the_aspect_ratio() {
    let (kappa, dt, steps) = (1e-3, 0.01, 100);
    for &(w, h, axis) in &[(34, 34, 0usize), (34, 18, 0), (34, 18, 1), (18, 34, 0), (34, 66, 0)] {
        let dx = 1.0 / w as f64;
        let n_axis = [w, h][axis];
        let l = fluid_length(n_axis, dx);
        let mode = move |i: usize, j: usize| (PI * from_wall([i, j][axis], dx) / l).cos();

        let mut g = FluidGrid::with_solver(w, h, kappa, 0.0, dt, SolverConfig::new(50)).unwrap();
        set_density_2d(&mut g, |i, j| 1.0 + mode(i, j));
        for _ in 0..steps {
            g.step();
        }

        let (mut num, mut den) = (0.0, 0.0);
        for j in 1..h - 1 {
            for i in 1..w - 1 {
                let c = mode(i, j);
                num += (g.get_density(i, j).unwrap() - 1.0) * c;
                den += c * c;
            }
        }
        let amplitude = num / den;
        let exact = discrete_step_factor(lowest_mode_eigenvalue(n_axis, dx), kappa, dt).powi(steps);
        let analytic = (-kappa * (PI / l).powi(2) * dt * steps as f64).exp();
        assert!(
            (amplitude - exact).abs() < 1e-7,
            "{w}x{h}, axis {axis}: amplitude {amplitude:.6}, analytic {analytic:.6} (exact scheme {exact:.6})"
        );
    }
}

// ---------------------------------------------------------------------------
// GRID-3: 2D projection scaled the y pressure gradient by height, not 1 / h
// ---------------------------------------------------------------------------

/// A projection removes a pure gradient field. On a square grid, two converged
/// projections leave 3.4e-11 of it (the collocated scheme's O(h^4) floor). On the
/// base, a 34×18 or 18×34 grid left 8.9e-2: the divergence used `h = 1/width` and
/// the y-gradient `1/height`, so the solve and the correction disagreed by H/W.
#[test]
fn review_2d_projection_removes_a_gradient_field_on_non_square_grids() {
    // This tests the discretisation, so the pressure is solved to rounding.
    let exact_pressure = SolverConfig::default().with_pressure_tolerance(1e-12, 10_000);
    for &(w, h) in &[(34usize, 34usize), (34, 18), (18, 34)] {
        let mut g = FluidGrid::with_solver(w, h, 0.0, 0.0, 1e-9, exact_pressure).unwrap();
        set_velocity_2d(&mut g, discrete_gradient_2d(w, h));
        let before = fluid_speed_sq_2d(&g);
        g.step();
        let left = fluid_speed_sq_2d(&g) / before;
        let floor = collocated_residual_2d(w, h);
        assert!(
            left < 10.0 * floor + 1e-12,
            "{w}x{h}: projection left {left:.3e} of a gradient field; the scheme's floor is {floor:.3e}"
        );
    }
}

// ---------------------------------------------------------------------------
// GRID-4: total mass and kinetic energy counted the ghost ring
// ---------------------------------------------------------------------------

/// Implicit diffusion with the ghost-copy boundary conserves the mass in the fluid
/// cells exactly once converged, so the reported total must not move. The base
/// summed the ghost ring as well, which holds copies of the wall-adjacent cells:
/// 1.0 injected on a 10×10 grid reported 1.5625 (= 100/64) once spread out, and on
/// an 8³ grid 2.37 (= 512/216).
#[test]
fn review_total_mass_is_conserved_by_pure_diffusion() {
    let mut g = FluidGrid::with_solver(10, 10, 0.1, 0.0, 0.1, SolverConfig::new(100)).unwrap();
    g.add_density(5, 5, 1.0).unwrap();
    for _ in 0..100 {
        g.step();
    }
    assert!((fluid_mass_2d(&g) - 1.0).abs() < 1e-6, "oracle self-check: fluid mass {}", fluid_mass_2d(&g));
    assert!((g.get_total_mass() - 1.0).abs() < 1e-6, "2D reported {} for 1.0 injected", g.get_total_mass());

    let mut g = FluidGrid3D::with_solver(8, 8, 8, 0.1, 0.0, 0.1, SolverConfig::new(100)).unwrap();
    g.add_density(4, 4, 4, 1.0).unwrap();
    for _ in 0..100 {
        g.step();
    }
    assert!((fluid_mass_3d(&g) - 1.0).abs() < 1e-6, "oracle self-check: fluid mass {}", fluid_mass_3d(&g));
    assert!((g.get_total_mass() - 1.0).abs() < 1e-6, "3D reported {} for 1.0 injected", g.get_total_mass());
}

/// Kinetic energy is `sum of 1/2 d |v|^2` over the fluid. The base also summed the
/// ghost ring, whose normal velocity is the negated neighbour and tangential the
/// copied one, so every wall-adjacent cell was counted twice.
#[test]
fn review_kinetic_energy_counts_each_fluid_cell_once() {
    let n = 18;
    let mut g = FluidGrid::with_solver(n, n, 0.0, 0.0, 1e-6, SolverConfig::new(20)).unwrap();
    set_density_2d(&mut g, |_, _| 1.0);
    set_velocity_2d(&mut g, taylor_green_2d(n, 0.1));
    g.step();

    let mut reference = 0.0;
    for j in 1..n - 1 {
        for i in 1..n - 1 {
            let (u, v) = g.get_velocity(i, j).unwrap();
            reference += 0.5 * g.get_density(i, j).unwrap() * (u * u + v * v);
        }
    }
    let reported = g.get_kinetic_energy();
    assert!(
        (reported - reference).abs() < 1e-12 * reference.max(1.0),
        "reported {reported:.6e}, fluid cells hold {reference:.6e}"
    );

    let mut g = FluidGrid3D::with_solver(n, n, n, 0.0, 0.0, 1e-6, SolverConfig::new(20)).unwrap();
    set_density_3d(&mut g, |_, _, _| 1.0);
    let tg = taylor_green_2d(n, 0.1);
    set_velocity_3d(&mut g, |i, j, _| {
        let (u, v) = tg(i, j);
        (u, v, 0.0)
    });
    g.step();
    let mut reference = 0.0;
    for k in 1..n - 1 {
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let (u, v, w) = g.get_velocity(i, j, k).unwrap();
                reference += 0.5 * g.get_density(i, j, k).unwrap() * (u * u + v * v + w * w);
            }
        }
    }
    let reported = g.get_kinetic_energy();
    assert!(
        (reported - reference).abs() < 1e-12 * reference.max(1.0),
        "3D reported {reported:.6e}, fluid cells hold {reference:.6e}"
    );
}

// ---------------------------------------------------------------------------
// GRID-5: a grid one cell thick was accepted and then panicked in `step`
// ---------------------------------------------------------------------------

/// `new` accepted any non-zero size, and `step` on a grid one cell thick indexed
/// `width - 2`, which underflows: an out-of-bounds panic in library code. Either
/// the constructor refuses it or `step` survives it.
#[test]
fn review_one_cell_thick_grids_do_not_panic() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    for &(w, h) in &[(1usize, 8usize), (8, 1)] {
        if let Ok(mut g) = FluidGrid::new(w, h, 0.1, 0.1, 0.1) {
            let stepped = catch_unwind(AssertUnwindSafe(|| g.step()));
            assert!(stepped.is_ok(), "{w}x{h}: step panicked");
        }
    }
    for &(w, h, d) in &[(1usize, 8usize, 8usize), (8, 1, 8), (8, 8, 1)] {
        if let Ok(mut g) = FluidGrid3D::new(w, h, d, 0.1, 0.1, 0.1) {
            let stepped = catch_unwind(AssertUnwindSafe(|| g.step()));
            assert!(stepped.is_ok(), "{w}x{h}x{d}: step panicked");
        }
    }
}

// ---------------------------------------------------------------------------
// GRID-6: NaN and infinite parameters were accepted
// ---------------------------------------------------------------------------

/// `validate_positive(NaN)` and `validate_non_negative(NaN)` pass, and infinity is
/// positive, so the constructors and setters accepted all of them. One step with
/// `dt = inf` turns a grid holding one unit of still dye entirely to NaN (`inf * 0`
/// in the back-trace). `SolverConfig::new(0)` was also accepted and silently
/// disabled both the projection and the diffusion, where `set_solver_iterations`
/// already clamps to 1.
#[test]
fn review_non_finite_parameters_are_rejected() {
    for bad in [f64::NAN, f64::INFINITY] {
        assert!(FluidGrid::new(8, 8, 0.0, 0.0, bad).is_err(), "2D new accepted dt = {bad}");
        assert!(FluidGrid::new(8, 8, 0.0, bad, 0.1).is_err(), "2D new accepted viscosity = {bad}");
        assert!(FluidGrid::new(8, 8, bad, 0.0, 0.1).is_err(), "2D new accepted diffusion = {bad}");
        assert!(FluidGrid3D::new(8, 8, 8, 0.0, 0.0, bad).is_err(), "3D new accepted dt = {bad}");
        assert!(FluidGrid3D::new(8, 8, 8, 0.0, bad, 0.1).is_err(), "3D new accepted viscosity = {bad}");
        assert!(FluidGrid3D::new(8, 8, 8, bad, 0.0, 0.1).is_err(), "3D new accepted diffusion = {bad}");

        let mut g = FluidGrid::new(8, 8, 0.0, 0.0, 0.1).unwrap();
        assert!(g.set_dt(bad).is_err(), "2D set_dt accepted {bad}");
        assert!(g.set_viscosity(bad).is_err(), "2D set_viscosity accepted {bad}");
        assert!(g.set_diffusion(bad).is_err(), "2D set_diffusion accepted {bad}");
        assert_eq!((g.get_dt(), g.get_viscosity(), g.get_diffusion()), (0.1, 0.0, 0.0));

        let mut g = FluidGrid3D::new(8, 8, 8, 0.0, 0.0, 0.1).unwrap();
        assert!(g.set_dt(bad).is_err(), "3D set_dt accepted {bad}");
        assert!(g.set_viscosity(bad).is_err(), "3D set_viscosity accepted {bad}");
        assert!(g.set_diffusion(bad).is_err(), "3D set_diffusion accepted {bad}");
        assert_eq!((g.get_dt(), g.get_viscosity(), g.get_diffusion()), (0.1, 0.0, 0.0));
    }

    // Zero iterations: refused, or clamped to the setter's documented minimum of 1.
    if let Ok(g) = FluidGrid::with_solver(8, 8, 0.0, 0.0, 0.1, SolverConfig::new(0)) {
        assert!(g.get_solver_iterations() >= 1, "2D with_solver kept 0 iterations");
    }
    if let Ok(g) = FluidGrid3D::with_solver(8, 8, 8, 0.0, 0.0, 0.1, SolverConfig::new(0)) {
        assert!(g.get_solver_iterations() >= 1, "3D with_solver kept 0 iterations");
    }
    let mut g = FluidGrid::new(8, 8, 0.0, 0.0, 0.1).unwrap();
    g.set_solver_config(SolverConfig::new(0)).unwrap();
    assert!(g.get_solver_iterations() >= 1, "2D set_solver_config kept 0 iterations");
    let mut g = FluidGrid3D::new(8, 8, 8, 0.0, 0.0, 0.1).unwrap();
    g.set_solver_config(SolverConfig::new(0)).unwrap();
    assert!(g.get_solver_iterations() >= 1, "3D set_solver_config kept 0 iterations");
}

// ---------------------------------------------------------------------------
// GRID-7: NaN and infinite sources were accepted
// ---------------------------------------------------------------------------

/// One NaN velocity at the centre of a 16×16 grid made all 256 cells non-finite
/// within 5 steps: the pressure solve reaches every cell. A source is the caller's
/// input and is refused, as `SphFluid::spawn` refuses a non-finite particle.
#[test]
fn review_non_finite_sources_are_rejected() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut g = FluidGrid::new(16, 16, 0.0, 0.0, 0.1).unwrap();
        assert!(g.add_density(8, 8, bad).is_err(), "2D add_density accepted {bad}");
        assert!(g.add_velocity(8, 8, bad, 0.0).is_err(), "2D add_velocity accepted vx = {bad}");
        assert!(g.add_velocity(8, 8, 0.0, bad).is_err(), "2D add_velocity accepted vy = {bad}");
        for _ in 0..5 {
            g.step();
        }
        assert!(g.validate_state().is_ok(), "2D grid was poisoned by a refused source");

        let mut g = FluidGrid3D::new(8, 8, 8, 0.0, 0.0, 0.1).unwrap();
        assert!(g.add_density(4, 4, 4, bad).is_err(), "3D add_density accepted {bad}");
        assert!(g.add_velocity(4, 4, 4, bad, 0.0, 0.0).is_err(), "3D add_velocity accepted vx = {bad}");
        assert!(g.add_velocity(4, 4, 4, 0.0, 0.0, bad).is_err(), "3D add_velocity accepted vz = {bad}");
        g.step();
        assert!(g.validate_state().is_ok(), "3D grid was poisoned by a refused source");
    }
}

// ---------------------------------------------------------------------------
// GRID-8: `SolverConfig::solver_type` and `relaxation` were ignored
// ---------------------------------------------------------------------------

/// Pressure solved by relaxation, so the configured solver type decides it.
fn relaxation_pressure(config: SolverConfig) -> SolverConfig {
    config.with_pressure_solver(PressureSolver::Relaxation)
}

/// `lin_solve` read only `iterations`: SOR(1.9), Jacobi and Gauss-Seidel gave
/// bit-identical output (each left 0.3401 of a gradient field after 30 sweeps).
/// Oracle, for the model Poisson problem at this size: SOR near its optimal factor
/// contracts the error by about ω − 1 = 0.9 per sweep, Gauss-Seidel by about
/// 0.995 for the smoothest mode, and Jacobi more slowly than Gauss-Seidel (Young).
#[test]
fn review_solver_type_selects_the_solver() {
    let n = 34;
    let left_after = |config: SolverConfig| {
        let mut g = FluidGrid::with_solver(n, n, 0.0, 0.0, 1e-9, relaxation_pressure(config)).unwrap();
        set_velocity_2d(&mut g, discrete_gradient_2d(n, n));
        let before = fluid_speed_sq_2d(&g);
        g.step();
        fluid_speed_sq_2d(&g) / before
    };
    let gs = left_after(SolverConfig::gauss_seidel(30));
    let sor = left_after(SolverConfig::sor(30, 1.9));
    let jacobi = left_after(SolverConfig::jacobi(30));
    assert!(sor < 0.1 * gs, "SOR left {sor:.4e}, Gauss-Seidel {gs:.4e}");
    assert!(jacobi > gs, "Jacobi left {jacobi:.4e}, Gauss-Seidel {gs:.4e}");
}

/// Jacobi's contraction of one Fourier mode is known exactly. The diffusion system is
/// `(1 + 4a)·x − a·Σ neighbours = x0`; a cosine mode `φ` along x (uniform in y) has
/// `Σ neighbours = (4 − μ)·φ` with `μ = 2 − 2cos(π/(n−2))`. Jacobi maps the error in
/// that mode to `ρ·error` with `ρ = a(4 − μ)/(1 + 4a)`, starting from the guess
/// `x = x0`, so after `m` sweeps the amplitude is `A/(1 + aμ) + ρᵐ·A·aμ/(1 + aμ)`.
/// Gauss-Seidel mixes modes and contracts faster, so matching this is Jacobi.
#[test]
fn review_jacobi_contracts_a_mode_at_its_exact_rate() {
    let (n, h, dt, kappa, sweeps) = (34usize, 10usize, 0.1, 0.05, 3usize);
    let dx = 1.0 / n as f64;
    let l = fluid_length(n, dx);
    let mode = move |i: usize| (PI * from_wall(i, dx) / l).cos();
    // a = dt·κ/h² ≈ 5.8: a slow system, where the solvers differ.
    let a = dt * kappa * (n * n) as f64;
    let mu = 2.0 - 2.0 * (PI / (n - 2) as f64).cos();

    let mut g = FluidGrid::with_solver(n, h, 0.0, 0.0, dt, SolverConfig::jacobi(sweeps)).unwrap();
    set_density_2d(&mut g, |i, _| 1.0 + mode(i));
    // One step without diffusion fills the boundary ring from the fluid cells, so the
    // next solve starts from a field that is exactly the mode, ring included.
    g.step();
    g.set_diffusion(kappa).unwrap();
    g.step();

    let (mut num, mut den) = (0.0, 0.0);
    for j in 1..h - 1 {
        for i in 1..n - 1 {
            num += (g.get_density(i, j).unwrap() - 1.0) * mode(i);
            den += mode(i) * mode(i);
        }
    }
    let amplitude = num / den;
    let rho = a * (4.0 - mu) / (1.0 + 4.0 * a);
    let exact = 1.0 / (1.0 + a * mu) + rho.powi(sweeps as i32) * a * mu / (1.0 + a * mu);
    assert!(
        (amplitude - exact).abs() < 1e-12,
        "Jacobi, {sweeps} sweeps: amplitude {amplitude:.15}, exact {exact:.15}"
    );

    // Gauss-Seidel on the same problem is closer to the converged 1/(1 + aμ).
    let mut g = FluidGrid::with_solver(n, h, 0.0, 0.0, dt, SolverConfig::gauss_seidel(sweeps)).unwrap();
    set_density_2d(&mut g, |i, _| 1.0 + mode(i));
    g.step();
    g.set_diffusion(kappa).unwrap();
    g.step();
    let (mut num, mut den) = (0.0, 0.0);
    for j in 1..h - 1 {
        for i in 1..n - 1 {
            num += (g.get_density(i, j).unwrap() - 1.0) * mode(i);
            den += mode(i) * mode(i);
        }
    }
    let converged = 1.0 / (1.0 + a * mu);
    assert!(
        (num / den - converged).abs() < (exact - converged).abs(),
        "Gauss-Seidel should beat Jacobi: {:.6} vs Jacobi {exact:.6}, converged {converged:.6}",
        num / den
    );
}

/// SOR with ω = 1 is Gauss-Seidel by definition, in 2D and 3D.
#[test]
fn review_sor_with_unit_relaxation_is_gauss_seidel() {
    let run_2d = |config: SolverConfig| {
        let mut g = FluidGrid::with_solver(20, 14, 0.02, 0.01, 0.05, relaxation_pressure(config)).unwrap();
        set_velocity_2d(&mut g, taylor_green_2d(20, 0.3));
        set_density_2d(&mut g, |i, j| (i * j) as f64 * 0.01);
        for _ in 0..5 {
            g.step();
        }
        (0..20).flat_map(|i| (0..14).map(move |j| (i, j))).map(|(i, j)| g.get_velocity(i, j).unwrap().0 + g.get_density(i, j).unwrap()).collect::<Vec<_>>()
    };
    let (gs, sor) = (run_2d(SolverConfig::gauss_seidel(6)), run_2d(SolverConfig::sor(6, 1.0)));
    for (a, b) in gs.iter().zip(&sor) {
        assert!((a - b).abs() <= 1e-12 * a.abs().max(1.0), "2D: {a} vs {b}");
    }

    let run_3d = |config: SolverConfig| {
        let mut g = FluidGrid3D::with_solver(10, 9, 8, 0.02, 0.01, 0.05, relaxation_pressure(config)).unwrap();
        set_velocity_3d(&mut g, |i, j, k| ((j * k) as f64 * 0.01, (i + k) as f64 * 0.01, 0.0));
        for _ in 0..3 {
            g.step();
        }
        let (u, v, w) = g.get_velocity(4, 4, 4).unwrap();
        u + v + w
    };
    let (gs, sor) = (run_3d(SolverConfig::gauss_seidel(6)), run_3d(SolverConfig::sor(6, 1.0)));
    assert!((gs - sor).abs() <= 1e-12 * gs.abs().max(1.0), "3D: {gs} vs {sor}");
}

/// `SolverConfig::sor` asserted its factor, a panic in library code. It now builds
/// any config, and every grid refuses a relaxation outside (0, 2) or a pressure
/// tolerance outside (0, 1), keeping its old configuration.
#[test]
fn review_invalid_solver_configs_are_rejected_not_panicking() {
    for bad in [0.0, 2.0, -0.5, 2.5, f64::NAN, f64::INFINITY] {
        let config = SolverConfig::sor(4, bad);
        assert!(FluidGrid::with_solver(8, 8, 0.0, 0.0, 0.1, config).is_err(), "2D accepted relaxation {bad}");
        assert!(FluidGrid3D::with_solver(8, 8, 8, 0.0, 0.0, 0.1, config).is_err(), "3D accepted relaxation {bad}");

        let mut g = FluidGrid::new(8, 8, 0.0, 0.0, 0.1).unwrap();
        assert!(g.set_solver_config(config).is_err(), "2D set_solver_config accepted relaxation {bad}");
        assert_eq!(g.get_solver_config().solver_type, SolverType::GaussSeidel, "2D config changed on error");
        let mut g = FluidGrid3D::new(8, 8, 8, 0.0, 0.0, 0.1).unwrap();
        assert!(g.set_solver_config(config).is_err(), "3D set_solver_config accepted relaxation {bad}");
        assert_eq!(g.get_solver_config().solver_type, SolverType::GaussSeidel, "3D config changed on error");
    }
    for bad in [0.0, 1.0, -1e-6, f64::NAN, f64::INFINITY] {
        let config = SolverConfig::default().with_pressure_tolerance(bad, 50);
        assert!(FluidGrid::with_solver(8, 8, 0.0, 0.0, 0.1, config).is_err(), "2D accepted tolerance {bad}");
        assert!(FluidGrid3D::with_solver(8, 8, 8, 0.0, 0.0, 0.1, config).is_err(), "3D accepted tolerance {bad}");
    }
    let mut g = FluidGrid::new(8, 8, 0.0, 0.0, 0.1).unwrap();
    assert!(g.set_solver_config(SolverConfig::sor(4, 1.5)).is_ok());
    assert_eq!(g.get_solver_config().relaxation, 1.5);
}

// ---------------------------------------------------------------------------
// GRID-9: the default pressure solve was far from converged
// ---------------------------------------------------------------------------

/// `step` promises a divergence-free velocity. With 4 Gauss-Seidel sweeps from zero,
/// one step at 128² left 99.06% of a smooth gradient field's energy (the
/// `high_quality` preset, 95.4%). The default is now a warm-started MIC(0)
/// conjugate gradient to a relative residual of 1e-4.
#[test]
fn review_default_projection_removes_most_of_a_smooth_divergence() {
    let n = 130;
    let mut g = FluidGrid::new(n, n, 0.0, 0.0, 1e-9).unwrap();
    set_velocity_2d(&mut g, discrete_gradient_2d(n, n));
    let before = fluid_speed_sq_2d(&g);
    g.step();
    let left = fluid_speed_sq_2d(&g) / before;
    assert!(left < 0.01, "one default step left {:.2}% of a gradient field's energy", 100.0 * left);
    // The collocated scheme's floor for this mode, which only a converged solve reaches.
    assert!(left < 10.0 * collocated_residual_2d(n, n) + 1e-12, "left {left:.3e}");
}

/// Same, 3D: one default step at 34³ removes a gradient field down to the scheme's
/// floor (the base's 4 sweeps left most of it).
#[test]
fn review_3d_default_projection_removes_most_of_a_smooth_divergence() {
    let n = 34;
    let dx = 1.0 / n as f64;
    let l = fluid_length(n, dx);
    let phi = move |i: f64, j: f64, k: f64| {
        (PI * (i - 0.5) * dx / l).cos() * (PI * (j - 0.5) * dx / l).cos() * (PI * (k - 0.5) * dx / l).cos()
    };
    let field = move |i: usize, j: usize, k: usize| {
        let (x, y, z) = (i as f64, j as f64, k as f64);
        (
            (phi(x + 1.0, y, z) - phi(x - 1.0, y, z)) / (2.0 * dx),
            (phi(x, y + 1.0, z) - phi(x, y - 1.0, z)) / (2.0 * dx),
            (phi(x, y, z + 1.0) - phi(x, y, z - 1.0)) / (2.0 * dx),
        )
    };
    let energy = |g: &FluidGrid3D| {
        let mut s = 0.0;
        for k in 1..n - 1 {
            for j in 1..n - 1 {
                for i in 1..n - 1 {
                    let (u, v, w) = g.get_velocity(i, j, k).unwrap();
                    s += u * u + v * v + w * w;
                }
            }
        }
        s
    };
    let mut g = FluidGrid3D::new(n, n, n, 0.0, 0.0, 1e-9).unwrap();
    set_velocity_3d(&mut g, field);
    let before = energy(&g);
    g.step();
    let left = energy(&g) / before;
    // (1 - r)^4 for the (1,1,1) mode: the wide and compact Laplacians' ratio.
    let t = PI / (n - 2) as f64;
    let floor = (1.0 - (t.sin().powi(2)) / (4.0 * (t / 2.0).sin().powi(2))).powi(4);
    assert!(left < 10.0 * floor + 1e-12, "one default 3D step left {left:.3e}; floor {floor:.3e}");
}

/// Independent reference: the conjugate gradient and 20 000 Gauss-Seidel sweeps
/// solve the same pressure system, so they must project a messy divergent field to
/// the same velocity (to the CG tolerance), on a non-square 2D grid and in 3D.
#[test]
fn review_conjugate_gradient_matches_a_many_sweep_reference() {
    let (w, h) = (26usize, 18usize);
    let field = |i: usize, j: usize| {
        let (x, y) = (i as f64, j as f64);
        ((0.37 * x).sin() * (0.23 * y).cos() + 0.1 * ((i * 7 + j * 3) % 5) as f64, (0.19 * x * y).cos() * 0.5)
    };
    let project = |config: SolverConfig| {
        let mut g = FluidGrid::with_solver(w, h, 0.0, 0.0, 1e-9, config).unwrap();
        set_velocity_2d(&mut g, field);
        g.step();
        (0..w).flat_map(|i| (0..h).map(move |j| (i, j))).map(|(i, j)| g.get_velocity(i, j).unwrap()).collect::<Vec<_>>()
    };
    let reference = project(relaxation_pressure(SolverConfig::gauss_seidel(20_000)));
    let scale = reference.iter().map(|(u, v)| u.abs().max(v.abs())).fold(0.0, f64::max);
    for (tolerance, agree) in [(1e-10, 1e-8), (1e-4, 2e-3)] {
        let cg = project(SolverConfig::default().with_pressure_tolerance(tolerance, 10_000));
        let worst = reference.iter().zip(&cg).map(|(a, b)| (a.0 - b.0).abs().max((a.1 - b.1).abs())).fold(0.0, f64::max);
        assert!(worst < agree * scale, "2D, tolerance {tolerance:e}: max difference {worst:.3e} of {scale:.3e}");
    }

    let n = 10;
    let project_3d = |config: SolverConfig| {
        let mut g = FluidGrid3D::with_solver(n, n + 2, n - 1, 0.0, 0.0, 1e-9, config).unwrap();
        set_velocity_3d(&mut g, |i, j, k| {
            let (x, y, z) = (i as f64, j as f64, k as f64);
            ((0.4 * y).sin() + 0.3 * x, (0.3 * z * x).cos(), 0.2 * ((i + 2 * j + k) % 3) as f64)
        });
        g.step();
        let mut out = vec![];
        for i in 0..n { for j in 0..n + 2 { for k in 0..n - 1 { out.push(g.get_velocity(i, j, k).unwrap()); } } }
        out
    };
    let reference = project_3d(relaxation_pressure(SolverConfig::gauss_seidel(5_000)));
    let cg = project_3d(SolverConfig::default().with_pressure_tolerance(1e-10, 10_000));
    let scale = reference.iter().map(|v| v.0.abs().max(v.1.abs()).max(v.2.abs())).fold(0.0, f64::max);
    let worst = reference.iter().zip(&cg).map(|(a, b)| (a.0 - b.0).abs().max((a.1 - b.1).abs()).max((a.2 - b.2).abs())).fold(0.0, f64::max);
    assert!(worst < 1e-8 * scale, "3D: max difference {worst:.3e} of {scale:.3e}");
}

/// The pressure is kept between steps, one field per projection: once a forced
/// flow settles, a step needs well under the cold start's iterations, and never
/// more than the cap.
#[test]
fn review_pressure_warm_start_and_cap() {
    let n = 66;
    let mut g = FluidGrid::new(n, n, 1e-5, 1e-5, 1.0 / 60.0).unwrap();
    let mut counts = vec![];
    for _ in 0..30 {
        for i in n / 4..3 * n / 4 {
            g.add_velocity(i, n / 2, 0.0, 0.05).unwrap();
        }
        g.step();
        counts.push(g.get_last_pressure_iterations());
    }
    let (cold, settled) = (counts[0], *counts.last().unwrap());
    assert!(cold > 0 && 2 * settled < cold, "iterations per step: cold {cold}, settled {settled} ({counts:?})");

    let capped = SolverConfig::default().with_pressure_tolerance(1e-12, 3);
    let mut g = FluidGrid::with_solver(n, n, 0.0, 0.0, 1e-9, capped).unwrap();
    set_velocity_2d(&mut g, discrete_gradient_2d(n, n));
    g.step();
    assert_eq!(g.get_last_pressure_iterations(), 6, "two projections, three iterations each");
}

// ---------------------------------------------------------------------------
// GRID-10: walls were documented as no-slip and are free-slip; no-slip is opt-in
// ---------------------------------------------------------------------------

/// Under `WallCondition::NoSlip` the tangential velocity is zero at every wall: the
/// ghost cell holds the negated neighbour, so their mean (the value at the wall,
/// half-way between) vanishes. The normal component vanishes there under either
/// condition. Checked on every wall of a 2D and a 3D grid after a step.
#[test]
fn review_walls_are_no_slip_when_asked() {
    let no_slip = SolverConfig::default().with_wall_condition(WallCondition::NoSlip);
    let n = 18;
    let mut g = FluidGrid::with_solver(n, n, 0.0, 1e-3, 0.01, no_slip).unwrap();
    set_velocity_2d(&mut g, taylor_green_2d(n, 1e-2));
    g.step();
    for t in 1..n - 1 {
        // (ghost, fluid) pairs across the left, right, bottom and top walls.
        for (ghost, fluid) in [((0, t), (1, t)), ((n - 1, t), (n - 2, t)), ((t, 0), (t, 1)), ((t, n - 1), (t, n - 2))] {
            let (gu, gv) = g.get_velocity(ghost.0, ghost.1).unwrap();
            let (fu, fv) = g.get_velocity(fluid.0, fluid.1).unwrap();
            assert!((gu + fu).abs() <= 1e-15 && (gv + fv).abs() <= 1e-15, "wall at {ghost:?}: u {gu:e}/{fu:e}, v {gv:e}/{fv:e}");
        }
    }

    let mut g = FluidGrid3D::with_solver(10, 10, 10, 0.0, 1e-3, 0.01, no_slip).unwrap();
    set_velocity_3d(&mut g, |i, j, k| ((j * k) as f64 * 1e-3, (i * k) as f64 * 1e-3, (i * j) as f64 * 1e-3));
    g.step();
    for a in 1..9 {
        for b in 1..9 {
            for (ghost, fluid) in [
                ((0, a, b), (1, a, b)), ((9, a, b), (8, a, b)),
                ((a, 0, b), (a, 1, b)), ((a, 9, b), (a, 8, b)),
                ((a, b, 0), (a, b, 1)), ((a, b, 9), (a, b, 8)),
            ] {
                let gv = g.get_velocity(ghost.0, ghost.1, ghost.2).unwrap();
                let fv = g.get_velocity(fluid.0, fluid.1, fluid.2).unwrap();
                assert!(
                    (gv.0 + fv.0).abs() <= 1e-15 && (gv.1 + fv.1).abs() <= 1e-15 && (gv.2 + fv.2).abs() <= 1e-15,
                    "3D wall at {ghost:?}: {gv:?} / {fv:?}"
                );
            }
        }
    }
}

/// No-slip as physics, not just bookkeeping: in a tall closed channel (8 widths of
/// fluid per width here, 12), a shear flow `v = sin(2πx/L)` along the walls is an exact
/// Dirichlet eigenmode of the no-slip diffusion, so away from the ends it decays by
/// `1/(1 + a(2 − 2cos(2π/(n−2))))` per step. Free-slip walls give a different
/// answer, which the second assertion checks so the first cannot pass by accident.
#[test]
fn review_no_slip_shear_mode_decays_at_the_dirichlet_rate() {
    let (w, dt, nu, steps) = (18usize, 0.01, 3e-3, 20);
    let h = 12 * (w - 2) + 2;
    let dx = 1.0 / w as f64;
    let l = fluid_length(w, dx);
    let mode = move |i: usize| (2.0 * PI * from_wall(i, dx) / l).sin();
    let middle = (h / 2 - 8)..(h / 2 + 8);
    let amplitude_after = |wall: WallCondition| {
        let config = SolverConfig::gauss_seidel(40).with_pressure_tolerance(1e-10, 2000).with_wall_condition(wall);
        let mut g = FluidGrid::with_solver(w, h, 0.0, nu, dt, config).unwrap();
        set_velocity_2d(&mut g, |i, _| (0.0, 1e-6 * mode(i)));
        for _ in 0..steps {
            g.step();
        }
        let (mut num, mut den) = (0.0, 0.0);
        for j in middle.clone() {
            for i in 1..w - 1 {
                num += g.get_velocity(i, j).unwrap().1 * mode(i);
                den += 1e-6 * mode(i) * mode(i);
            }
        }
        num / den
    };
    let a = dt * nu * (w * w) as f64;
    let exact = (1.0 / (1.0 + a * (2.0 - 2.0 * (2.0 * PI / (w - 2) as f64).cos()))).powi(steps);
    let no_slip = amplitude_after(WallCondition::NoSlip);
    assert!((no_slip - exact).abs() < 1e-6, "no-slip amplitude {no_slip:.8}, Dirichlet mode {exact:.8}");
    let free_slip = amplitude_after(WallCondition::FreeSlip);
    assert!((free_slip - exact).abs() > 1e-3, "free-slip amplitude {free_slip:.8} should not match {exact:.8}");
}

/// The default is free-slip, as the docs now say: the ghost cell copies the
/// tangential velocity (so the wall sees the fluid's) and negates the normal one.
/// `review_2d_viscosity_matches_taylor_green_decay` confirms the physics: that cell
/// decays at exactly the free-slip analytic rate.
#[test]
fn review_walls_are_free_slip_by_default() {
    assert_eq!(SolverConfig::default().wall, WallCondition::FreeSlip);
    let n = 18;
    let mut g = FluidGrid::new(n, n, 0.0, 1e-3, 0.01).unwrap();
    set_velocity_2d(&mut g, taylor_green_2d(n, 1e-2));
    g.step();
    for t in 1..n - 1 {
        let ((gu, gv), (fu, fv)) = (g.get_velocity(0, t).unwrap(), g.get_velocity(1, t).unwrap());
        assert_eq!((gu, gv), (-fu, fv), "left wall at y = {t}");
        let ((gu, gv), (fu, fv)) = (g.get_velocity(t, 0).unwrap(), g.get_velocity(t, 1).unwrap());
        assert_eq!((gu, gv), (fu, -fv), "bottom wall at x = {t}");
    }
    assert!(g.get_velocity(0, n / 3).unwrap().1.abs() > 1e-4, "the tangential flow at the wall is not zero");

    let mut g = FluidGrid3D::new(10, 10, 10, 0.0, 1e-3, 0.01).unwrap();
    set_velocity_3d(&mut g, |i, j, k| ((j * k) as f64 * 1e-3, (i * k) as f64 * 1e-3, (i * j) as f64 * 1e-3));
    g.step();
    for a in 1..9 {
        for b in 1..9 {
            let (gv, fv) = (g.get_velocity(0, a, b).unwrap(), g.get_velocity(1, a, b).unwrap());
            assert_eq!(gv, (-fv.0, fv.1, fv.2), "3D left wall at ({a}, {b})");
            let (gv, fv) = (g.get_velocity(a, b, 9).unwrap(), g.get_velocity(a, b, 8).unwrap());
            assert_eq!(gv, (fv.0, fv.1, -fv.2), "3D back wall at ({a}, {b})");
        }
    }
}

// ---------------------------------------------------------------------------
// GRID-11: writes to the boundary ring returned Ok and were discarded
// ---------------------------------------------------------------------------

/// `add_density` and `add_velocity` took the boundary ring (index 0 or n − 1 on any
/// axis), returned `Ok`, and the next step overwrote it: all of a unit of dye added
/// at (0, 5) was gone. Every ring cell is now refused, and a fluid cell, including
/// one against the wall, keeps what it is given.
#[test]
fn review_sources_on_the_boundary_ring_are_refused() {
    let (w, h) = (10usize, 7usize);
    let mut g = FluidGrid::new(w, h, 0.0, 0.0, 0.1).unwrap();
    for x in 0..w {
        for y in 0..h {
            let ring = x == 0 || y == 0 || x == w - 1 || y == h - 1;
            assert_eq!(g.add_density(x, y, 1.0).is_err(), ring, "add_density({x}, {y})");
            assert_eq!(g.add_velocity(x, y, 0.0, 0.0).is_err(), ring, "add_velocity({x}, {y})");
        }
    }
    // Every fluid cell got 1.0 above; nothing moves, nothing diffuses, nothing is lost.
    g.step();
    let fluid_cells = ((w - 2) * (h - 2)) as f64;
    assert!((g.get_total_mass() - fluid_cells).abs() < 1e-12, "kept {} of {fluid_cells}", g.get_total_mass());
    assert_eq!(g.get_density(1, 3).unwrap(), 1.0, "a cell against the wall keeps its dye");

    let (w, h, d) = (6usize, 5usize, 4usize);
    let mut g = FluidGrid3D::new(w, h, d, 0.0, 0.0, 0.1).unwrap();
    for x in 0..w {
        for y in 0..h {
            for z in 0..d {
                let ring = x == 0 || y == 0 || z == 0 || x == w - 1 || y == h - 1 || z == d - 1;
                assert_eq!(g.add_density(x, y, z, 1.0).is_err(), ring, "3D add_density({x}, {y}, {z})");
                assert_eq!(g.add_velocity(x, y, z, 0.0, 0.0, 0.0).is_err(), ring, "3D add_velocity({x}, {y}, {z})");
            }
        }
    }
    g.step();
    let fluid_cells = ((w - 2) * (h - 2) * (d - 2)) as f64;
    assert!((g.get_total_mass() - fluid_cells).abs() < 1e-12, "3D kept {} of {fluid_cells}", g.get_total_mass());
}

// ---------------------------------------------------------------------------
// Checked and found correct (these pass on the base; they guard what holds up)
// ---------------------------------------------------------------------------

/// Viscosity in 2D on a square grid: a Taylor-Green cell decays at `exp(-nu k^2 t)`
/// and the answer converges with resolution (|error| 4e-5, 1e-5 at 18², 34²).
#[test]
fn review_2d_viscosity_matches_taylor_green_decay() {
    let (nu, dt, steps) = (1e-3, 0.01, 50);
    for &n in &[18usize, 34] {
        let dx = 1.0 / n as f64;
        let l = fluid_length(n, dx);
        let tg = taylor_green_2d(n, 1e-6);
        let mut g = FluidGrid::with_solver(n, n, 0.0, nu, dt, SolverConfig::new(20)).unwrap();
        set_velocity_2d(&mut g, &tg);
        for _ in 0..steps {
            g.step();
        }
        let (mut num, mut den) = (0.0, 0.0);
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let (u, v) = tg(i, j);
                let (a, b) = g.get_velocity(i, j).unwrap();
                num += a * u + b * v;
                den += u * u + v * v;
            }
        }
        let amplitude = num / den;
        let exact = discrete_step_factor(2.0 * lowest_mode_eigenvalue(n, dx), nu, dt).powi(steps);
        let analytic = (-nu * 2.0 * (PI / l).powi(2) * dt * steps as f64).exp();
        assert!((amplitude - exact).abs() < 1e-6, "{n}²: amplitude {amplitude:.6}, exact {exact:.6}");
        assert!((amplitude - analytic).abs() < 1e-4, "{n}²: amplitude {amplitude:.6}, analytic {analytic:.6}");
    }
}

/// The Gauss-Seidel sweep over a 5-point stencil is symmetric under x <-> y on a
/// square grid (a cell's west and south neighbours are always already updated, in
/// either loop order), so a mirror-symmetric setup stays mirror-symmetric to rounding.
#[test]
fn review_2d_is_symmetric_under_axis_swap() {
    let n = 34;
    let blob = move |i: usize, j: usize| {
        let (x, y) = (i as f64 / n as f64 - 0.3, j as f64 / n as f64 - 0.3);
        (-(x * x + y * y) / 0.01).exp()
    };
    let mut g = FluidGrid::new(n, n, 1e-4, 1e-4, 0.01).unwrap();
    set_velocity_2d(&mut g, |i, j| (blob(i, j), blob(i, j)));
    set_density_2d(&mut g, blob);
    for _ in 0..30 {
        g.step();
    }
    for j in 0..n {
        for i in 0..n {
            let (u, _) = g.get_velocity(i, j).unwrap();
            let (_, v) = g.get_velocity(j, i).unwrap();
            assert!((u - v).abs() < 1e-12, "u({i},{j}) = {u}, v({j},{i}) = {v}");
            let (a, b) = (g.get_density(i, j).unwrap(), g.get_density(j, i).unwrap());
            assert!((a - b).abs() < 1e-12, "density ({i},{j}) = {a}, ({j},{i}) = {b}");
        }
    }
}

/// Semi-Lagrangian advection is unconditionally stable: at a game timestep far past
/// the CFL limit (a step moves the flow ~30 cells) the state stays finite and the
/// fluid's kinetic energy does not grow.
#[test]
fn review_large_timestep_stays_bounded() {
    let n = 34;
    let mut g = FluidGrid::new(n, n, 0.0, 0.0, 1.0 / 30.0).unwrap();
    set_velocity_2d(&mut g, taylor_green_2d(n, 10.0));
    let mut energy = fluid_speed_sq_2d(&g);
    for _ in 0..200 {
        g.step();
        assert!(g.validate_state().is_ok());
        let now = fluid_speed_sq_2d(&g);
        assert!(now <= energy * (1.0 + 1e-9), "energy grew from {energy} to {now}");
        energy = now;
    }
}
