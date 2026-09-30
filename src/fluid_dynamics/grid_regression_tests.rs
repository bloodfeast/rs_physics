//! Regression tests for the Eulerian grid solvers, `FluidGrid` and `FluidGrid3D`.
//!
//! Each test checks the solver against an oracle that does not come from the solver:
//! an analytic decay rate, a conservation law, the definition of a boundary
//! condition, or a reference sum written out here. Findings are numbered `GRID-n`.
//! Defects still open are pinned by `#[ignore = "known defect GRID-n: ..."]` tests;
//! run them with `cargo test --lib --features fluid_simulation -- --ignored grid_regression`.
//!
//! ## The grid's unit of length
//!
//! Both grids measure length in *domain widths*: the cell size is `h = 1 / width` on
//! every axis, so velocities are in widths per second (`advect` moves a quantity
//! `dt * width * v` cells) and viscosity and diffusion are in widths² per second.
//! The outermost ring of cells is a ghost layer that `set_boundaries` overwrites; the
//! fluid occupies cells `1..n-1` on each axis, so it is `(n - 2) * h` long there, with
//! walls half a cell outside the first and last fluid cells. A cell with index `i`
//! is centred `(i - 0.5) * h` from the wall.

use super::{FluidGrid, FluidGrid3D, SolverConfig};
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
    for j in 0..g.get_height() {
        for i in 0..g.get_width() {
            let (u0, v0) = g.get_velocity(i, j).unwrap();
            let (u, v) = f(i, j);
            g.add_velocity(i, j, u - u0, v - v0).unwrap();
        }
    }
}

fn set_density_2d(g: &mut FluidGrid, f: impl Fn(usize, usize) -> f64) {
    for j in 0..g.get_height() {
        for i in 0..g.get_width() {
            let d0 = g.get_density(i, j).unwrap();
            g.add_density(i, j, f(i, j) - d0).unwrap();
        }
    }
}

fn set_velocity_3d(g: &mut FluidGrid3D, f: impl Fn(usize, usize, usize) -> (f64, f64, f64)) {
    for k in 0..g.get_depth() {
        for j in 0..g.get_height() {
            for i in 0..g.get_width() {
                let (u0, v0, w0) = g.get_velocity(i, j, k).unwrap();
                let (u, v, w) = f(i, j, k);
                g.add_velocity(i, j, k, u - u0, v - v0, w - w0).unwrap();
            }
        }
    }
}

fn set_density_3d(g: &mut FluidGrid3D, f: impl Fn(usize, usize, usize) -> f64) {
    for k in 0..g.get_depth() {
        for j in 0..g.get_height() {
            for i in 0..g.get_width() {
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

/// The scheme's own central-difference gradient of `phi`, sampled at every cell.
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
    for &(w, h) in &[(34usize, 34usize), (34, 18), (18, 34)] {
        let mut g = FluidGrid::with_solver(w, h, 0.0, 0.0, 1e-9, SolverConfig::new(6000)).unwrap();
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
    g.set_solver_config(SolverConfig::new(0));
    assert!(g.get_solver_iterations() >= 1, "2D set_solver_config kept 0 iterations");
    let mut g = FluidGrid3D::new(8, 8, 8, 0.0, 0.0, 0.1).unwrap();
    g.set_solver_config(SolverConfig::new(0));
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
// Open defects
// ---------------------------------------------------------------------------

/// GRID-8. `SolverConfig` documents three solvers and a relaxation factor, and
/// `lin_solve` reads only `iterations`: SOR(1.9), Jacobi and Gauss-Seidel produce
/// bit-identical output (all leave 0.3401 of a gradient field at 30 iterations).
/// Oracle: for the model Poisson problem at this size, SOR with omega = 1.9 contracts
/// the error by ~0.9 per sweep against GS's ~0.995 for the smoothest mode (Young),
/// and Jacobi converges more slowly than GS.
#[test]
#[ignore = "known defect GRID-8: SolverConfig::solver_type and relaxation are ignored; every config runs Gauss-Seidel"]
fn review_solver_type_selects_the_solver() {
    let n = 34;
    let left_after = |config: SolverConfig| {
        let mut g = FluidGrid::with_solver(n, n, 0.0, 0.0, 1e-9, config).unwrap();
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

/// GRID-9. `step` documents that projection "ensures incompressibility". With the
/// default config (4 Gauss-Seidel sweeps from a zero pressure every call) one step at
/// 128² leaves 99.06% of the energy of a smooth gradient field in place; the
/// `high_quality` preset (20) leaves 95.4%. A converged solve leaves ~1e-10.
#[test]
#[ignore = "known defect GRID-9: default pressure solve is far from converged; a step removes ~1% of a smooth divergence at 128^2"]
fn review_default_projection_removes_most_of_a_smooth_divergence() {
    let n = 130;
    let mut g = FluidGrid::new(n, n, 0.0, 0.0, 1e-9).unwrap();
    set_velocity_2d(&mut g, discrete_gradient_2d(n, n));
    let before = fluid_speed_sq_2d(&g);
    g.step();
    let left = fluid_speed_sq_2d(&g) / before;
    assert!(left < 0.01, "one default step left {:.2}% of a gradient field's energy", 100.0 * left);
}

/// GRID-10. `BoundaryType` and `set_boundaries` document the walls as no-slip, which
/// makes the tangential velocity zero at the wall: the ghost value is the negated
/// neighbour, so their average is 0. The code copies the tangential component
/// instead, which is free-slip: the wall value equals the fluid value (3.1e-3 here,
/// not 0). The Taylor-Green test above confirms it: the cell decays at exactly the
/// free-slip rate.
#[test]
#[ignore = "known defect GRID-10: walls are free-slip, documented as no-slip"]
fn review_walls_are_no_slip_as_documented() {
    let n = 18;
    let mut g = FluidGrid::with_solver(n, n, 0.0, 1e-3, 0.01, SolverConfig::new(50)).unwrap();
    set_velocity_2d(&mut g, taylor_green_2d(n, 1e-3));
    g.step();
    let (_, ghost) = g.get_velocity(0, n / 3).unwrap();
    let (_, fluid) = g.get_velocity(1, n / 3).unwrap();
    let at_wall = 0.5 * (ghost + fluid);
    assert!(
        at_wall.abs() < 1e-3 * fluid.abs(),
        "tangential velocity at the left wall is {at_wall:.3e}; the adjacent fluid moves at {fluid:.3e}"
    );
}

/// GRID-11. `add_density` and `add_velocity` accept the ghost ring (index 0 and
/// n - 1 on any axis) and return `Ok`, and the next `step` overwrites it: all of a
/// unit of dye added at (0, 5) is gone after one step. An inlet at the domain edge,
/// the natural place for a river's source, silently does nothing.
#[test]
#[ignore = "known defect GRID-11: sources written to the ghost ring return Ok and are discarded by the next step"]
fn review_density_written_to_the_boundary_ring_is_not_lost() {
    let mut g = FluidGrid::new(10, 10, 0.0, 0.0, 0.1).unwrap();
    if g.add_density(0, 5, 1.0).is_ok() {
        g.step();
        let kept = fluid_mass_2d(&g);
        assert!((kept - 1.0).abs() < 1e-9, "accepted 1.0 on the boundary ring; {kept} reached the fluid");
    }
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
