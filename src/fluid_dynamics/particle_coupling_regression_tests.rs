//! Regression tests for particle-fluid coupling, from the 2026-09-29 correctness
//! and performance review: see `docs/reviews/2026-09-29-correctness-performance.md`.
//!
//! Tests marked `#[ignore]` with a "known defect" reason reproduce findings
//! deliberately left for follow-up work.

use super::*;
use crate::fluid_dynamics::{PressureSolver, SolverConfig, SolverType};

fn grid_velocity_sum_2d(g: &FluidGrid) -> (f64, f64) {
    let mut s = (0.0, 0.0);
    for i in 0..g.get_width() {
        for j in 0..g.get_height() {
            let v = g.get_velocity(i, j).unwrap();
            s.0 += v.0;
            s.1 += v.1;
        }
    }
    s
}

fn grid_velocity_sum_3d(g: &FluidGrid3D) -> (f64, f64, f64) {
    let mut s = (0.0, 0.0, 0.0);
    for i in 0..g.get_width() {
        for j in 0..g.get_height() {
            for k in 0..g.get_depth() {
                let v = g.get_velocity(i, j, k).unwrap();
                s.0 += v.0;
                s.1 += v.1;
                s.2 += v.2;
            }
        }
    }
    s
}

/// No projection, so a uniform field advects as itself and only advection is being
/// compared. (A converged projection would remove a uniform flow in a closed box
/// entirely; one relaxation sweep, the least a grid runs, barely touches it.)
fn advection_only() -> SolverConfig {
    SolverConfig {
        iterations: 0,
        relaxation: 1.0,
        solver_type: SolverType::GaussSeidel,
        pressure_solver: PressureSolver::Relaxation,
        ..SolverConfig::default()
    }
}

/// A 0.1 mm sand grain, 2650 kg/m^3: (radius, mass).
fn sand_grain() -> (f64, f64) {
    let r = 1e-4;
    (r, 4.0 / 3.0 * std::f64::consts::PI * r * r * r * 2650.0)
}

/// Newton's third law: the fluid's reaction to a force F held for the grid step
/// depends on F and the step, not on the mass of the particle that felt it. The
/// function divides by `particle_mass`, so the same 1 N pushes the fluid 1000x
/// harder when it acts on a 1 g particle than on a 1 kg one.
#[test]
#[ignore = "known defect, not fixed in this change: see docs/reviews/2026-09-29-correctness-performance.md"]
fn reaction_impulse_is_independent_of_particle_mass() {
    let mut heavy = FluidGrid::new(20, 20, 0.0, 0.0, 0.016).unwrap();
    let mut light = FluidGrid::new(20, 20, 0.0, 0.0, 0.016).unwrap();
    apply_particle_force_to_grid_2d(&mut heavy, (10.3, 10.6), (1.0, 0.0), 1.0);
    apply_particle_force_to_grid_2d(&mut light, (10.3, 10.6), (1.0, 0.0), 0.001);
    let h = grid_velocity_sum_2d(&heavy).0;
    let l = grid_velocity_sum_2d(&light).0;
    assert!(
        (h - l).abs() <= 1e-12 * h.abs(),
        "same 1 N for the same 0.016 s: fluid kick {h:.3e} from a 1 kg particle, \
         {l:.3e} from a 1 g particle"
    );
}

/// The particle integrated its drag with the caller's `dt` while the reaction on
/// the grid used `grid.get_dt()`, so sub-stepping particles (dt < grid dt) handed
/// the fluid a multiple of the impulse the particle lost: 4.00x at four substeps.
/// Asserted under the module's own "sum of dv on the grid = -dv of the particle"
/// convention.
#[test]
fn two_way_coupling_uses_one_timestep() {
    let grid_dt = 0.016;
    let dt = grid_dt / 4.0;
    let mut grid = FluidGrid::new(20, 20, 0.0, 0.0, grid_dt).unwrap();
    let mut p = FluidParticle2D::new(10.25, 10.5, 0.05, 2.0);
    p.vx = 1.0; // moving through still fluid: drag decelerates it
    let v0 = p.vx;
    p.update(&mut grid, 1000.0, 0.001, dt, true);
    let particle_dv = p.vx - v0;
    let fluid_dv = grid_velocity_sum_2d(&grid).0;
    assert!(particle_dv < 0.0, "drag did not decelerate the particle");
    assert!(
        (fluid_dv + particle_dv).abs() <= 1e-9 * particle_dv.abs(),
        "particle dv {particle_dv:.4e}, fluid sum dv {fluid_dv:.4e} (ratio {:.2})",
        -fluid_dv / particle_dv
    );
}

/// Same, 3D.
#[test]
fn two_way_coupling_uses_one_timestep_3d() {
    let grid_dt = 0.016;
    let dt = grid_dt / 4.0;
    let mut grid = FluidGrid3D::new(10, 10, 10, 0.0, 0.0, grid_dt).unwrap();
    let mut p = FluidParticle3D::new(5.25, 5.5, 5.5, 0.05, 2.0);
    p.vx = 1.0;
    let v0 = p.vx;
    p.update(&mut grid, 1000.0, 0.001, dt, true);
    let particle_dv = p.vx - v0;
    let fluid_dv = grid_velocity_sum_3d(&grid).0;
    assert!(particle_dv < 0.0, "drag did not decelerate the particle");
    assert!(
        (fluid_dv + particle_dv).abs() <= 1e-9 * particle_dv.abs(),
        "particle dv {particle_dv:.4e}, fluid sum dv {fluid_dv:.4e} (ratio {:.2})",
        -fluid_dv / particle_dv
    );
}

/// Explicit Euler on a drag force whose relaxation time is shorter than dt
/// overshoots and diverges. A sand grain released at rest in water flowing at
/// 1 m/s, stepped at 60 Hz, used to go 18.2, -2943, 8.6e7, ... -inf in eight steps.
/// Its velocity must approach the flow's and never overshoot it.
#[test]
fn drag_relaxation_is_stable_for_light_particles() {
    let mut grid = FluidGrid::new(20, 20, 0.0, 0.0, 0.016).unwrap();
    // Fluid cells only: the grid refuses its boundary ring.
    for i in 1..19 {
        for j in 1..19 {
            grid.add_velocity(i, j, 1.0, 0.0).unwrap();
        }
    }
    let (r, m) = sand_grain();
    let mut p = FluidParticle2D::new(10.0, 10.0, r, m);
    let mut history = Vec::new();
    for _ in 0..8 {
        // Keep it sampling the same uniform flow.
        p.x = 10.0;
        p.y = 10.0;
        p.update(&mut grid, 1000.0, 1.0e-3, 0.016, false);
        history.push(p.vx);
    }
    assert!(
        history.windows(2).all(|w| w[1] >= w[0])
            && history
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0 + 1e-9),
        "grain velocity history {history:?} — should rise monotonically toward 1.0"
    );
}

/// Same, 3D.
#[test]
fn drag_relaxation_is_stable_for_light_particles_3d() {
    let mut grid = FluidGrid3D::new(10, 10, 10, 0.0, 0.0, 0.016).unwrap();
    for i in 1..9 {
        for j in 1..9 {
            for k in 1..9 {
                grid.add_velocity(i, j, k, 0.0, 0.0, 1.0).unwrap();
            }
        }
    }
    let (r, m) = sand_grain();
    let mut p = FluidParticle3D::new(5.0, 5.0, 5.0, r, m);
    let mut history = Vec::new();
    for _ in 0..8 {
        p.x = 5.0;
        p.y = 5.0;
        p.z = 5.0;
        p.update(&mut grid, 1000.0, 1.0e-3, 0.016, false);
        history.push(p.vz);
    }
    assert!(
        history.windows(2).all(|w| w[1] >= w[0])
            && history
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0 + 1e-9),
        "grain velocity history {history:?} — should rise monotonically toward 1.0"
    );
}

/// Units: the grid's own advection moves a quantity `dt * width * v` cells per
/// step (Stam's unit-square convention, `dt0 = dt * width` in `advect`), while
/// `FluidParticle2D::update` moved the particle `dt * v` cells. A tracer already
/// moving with the flow lagged the dye it sat in by a factor of `width`: 64x here.
#[test]
fn tracer_moves_with_the_grid_advection() {
    let (n, dt, u) = (64usize, 0.01, 0.5);
    let make = || {
        let mut g = FluidGrid::new(n, n, 0.0, 0.0, dt).unwrap();
        g.set_solver_config(advection_only()).unwrap();
        // Fluid cells only: the grid refuses its boundary ring, which `step` overwrote.
        for i in 1..n - 1 {
            for j in 1..n - 1 {
                g.add_velocity(i, j, u, 0.0).unwrap();
            }
        }
        g
    };

    let mut dye = make();
    dye.add_density(20, 32, 1.0).unwrap();
    dye.step();
    let (mut m0, mut m1) = (0.0, 0.0);
    for i in 0..n {
        for j in 0..n {
            let d = dye.get_density(i, j).unwrap();
            m0 += d;
            m1 += d * i as f64;
        }
    }
    let dye_shift = m1 / m0 - 20.0;

    let mut g = make();
    let mut p = FluidParticle2D::new(20.0, 32.0, 0.01, 1.0);
    p.vx = u; // already moving with the flow: zero drag
    p.update(&mut g, 1000.0, 0.001, dt, false);
    let tracer_shift = p.x - 20.0;

    assert!(
        (tracer_shift - dye_shift).abs() < 0.05 * dye_shift.abs(),
        "one step of the same flow: dye moved {dye_shift:.4} cells, tracer moved \
         {tracer_shift:.4} cells ({:.1}x)",
        dye_shift / tracer_shift
    );
}

/// Same, 3D.
#[test]
fn tracer_moves_with_the_grid_advection_3d() {
    let (n, dt, u) = (16usize, 0.01, 0.5);
    let make = || {
        let mut g = FluidGrid3D::new(n, n, n, 0.0, 0.0, dt).unwrap();
        g.set_solver_config(advection_only()).unwrap();
        for i in 1..n - 1 {
            for j in 1..n - 1 {
                for k in 1..n - 1 {
                    g.add_velocity(i, j, k, u, 0.0, 0.0).unwrap();
                }
            }
        }
        g
    };

    let mut dye = make();
    dye.add_density(5, 8, 8, 1.0).unwrap();
    dye.step();
    let (mut m0, mut m1) = (0.0, 0.0);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let d = dye.get_density(i, j, k).unwrap();
                m0 += d;
                m1 += d * i as f64;
            }
        }
    }
    let dye_shift = m1 / m0 - 5.0;

    let mut g = make();
    let mut p = FluidParticle3D::new(5.0, 8.0, 8.0, 0.01, 1.0);
    p.vx = u;
    p.update(&mut g, 1000.0, 0.001, dt, false);
    let tracer_shift = p.x - 5.0;

    assert!(
        (tracer_shift - dye_shift).abs() < 0.05 * dye_shift.abs(),
        "one step of the same flow: dye moved {dye_shift:.4} cells, tracer moved \
         {tracer_shift:.4} cells ({:.1}x)",
        dye_shift / tracer_shift
    );
}
