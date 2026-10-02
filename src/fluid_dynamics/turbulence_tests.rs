//! Tests for the grids' turbulence options: MacCormack advection and vorticity
//! confinement ([`AdvectionScheme`], [`VorticityConfinement`]).
//!
//! The oracles are the inviscid Taylor-Green cell, which the Euler equations hold
//! steady, so everything it loses is the scheme's numerical dissipation, and the
//! definition of a projection.

use super::{AdvectionScheme, FluidGrid, FluidGrid3D, SolverConfig, VorticityConfinement};

/// A Taylor-Green array of `m x m` cells, stream function
/// `psi = amp L / (m pi) sin(m pi x / L) sin(m pi y / L)` over the fluid region of
/// length `L`, so the peak speed is `amp`. The stream function vanishes on the walls,
/// so nothing crosses them, and the flow is an exact steady solution of the inviscid
/// equations.
fn taylor_green(n: usize, m: f64, amp: f64) -> impl Fn(usize, usize) -> (f64, f64) {
    let h = 1.0 / n as f64;
    let length = (n - 2) as f64 * h;
    let k = m * std::f64::consts::PI / length;
    move |i, j| {
        let (x, y) = ((i as f64 - 0.5) * h, (j as f64 - 0.5) * h);
        (amp * (k * x).sin() * (k * y).cos(), -amp * (k * x).cos() * (k * y).sin())
    }
}

fn set_velocity_2d(g: &mut FluidGrid, f: impl Fn(usize, usize) -> (f64, f64)) {
    let (w, h) = (g.get_width(), g.get_height());
    for i in 1..w - 1 {
        for j in 1..h - 1 {
            let (vx, vy) = f(i, j);
            let (ox, oy) = g.get_velocity(i, j).unwrap();
            g.add_velocity(i, j, vx - ox, vy - oy).unwrap();
        }
    }
}

/// The curl `dv/dx - du/dy` at every fluid cell not next to a wall, by central
/// differences in widths, and the central divergence there.
fn curl_and_divergence_2d(g: &FluidGrid) -> (Vec<f64>, Vec<f64>) {
    let (w, h) = (g.get_width(), g.get_height());
    let inv_2h = 0.5 * w as f64;
    let v = |i: usize, j: usize| g.get_velocity(i, j).unwrap();
    let (mut curl, mut div) = (Vec::new(), Vec::new());
    for i in 2..w - 2 {
        for j in 2..h - 2 {
            curl.push(((v(i + 1, j).1 - v(i - 1, j).1) - (v(i, j + 1).0 - v(i, j - 1).0)) * inv_2h);
            div.push(((v(i + 1, j).0 - v(i - 1, j).0) + (v(i, j + 1).1 - v(i, j - 1).1)) * inv_2h);
        }
    }
    (curl, div)
}

fn peak(values: &[f64]) -> f64 {
    values.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

fn sum_sq(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum()
}

/// A 2D Taylor-Green run with no viscosity: `(peak vorticity, enstrophy, kinetic
/// energy, divergence)` before and after `steps`.
struct Run2d {
    peak: [f64; 2],
    enstrophy: [f64; 2],
    energy: [f64; 2],
    divergence: Vec<f64>,
}

fn run_2d(config: SolverConfig, n: usize, steps: usize) -> Run2d {
    // 0.3 widths/s at 1/60 s on 64 cells is a Courant number of 0.32 at the peak: the
    // fractional offsets where linear interpolation is most diffusive.
    let mut g = FluidGrid::with_solver(n, n, 0.0, 0.0, 1.0 / 60.0, config).unwrap();
    set_velocity_2d(&mut g, taylor_green(n, 2.0, 0.3));
    let (curl0, _) = curl_and_divergence_2d(&g);
    let energy0 = g_energy(&g);
    for _ in 0..steps {
        g.step();
        assert!(g.validate_state().is_ok());
    }
    let (curl, divergence) = curl_and_divergence_2d(&g);
    Run2d {
        peak: [peak(&curl0), peak(&curl)],
        enstrophy: [sum_sq(&curl0), sum_sq(&curl)],
        energy: [energy0, g_energy(&g)],
        divergence,
    }
}

fn g_energy(g: &FluidGrid) -> f64 {
    let (w, h) = (g.get_width(), g.get_height());
    let mut e = 0.0;
    for i in 1..w - 1 {
        for j in 1..h - 1 {
            let (u, v) = g.get_velocity(i, j).unwrap();
            e += 0.5 * (u * u + v * v);
        }
    }
    e
}

/// MacCormack keeps more of an inviscid vortex than first-order advection. Measured
/// 2026-10-02 at 64², 120 steps (2 s): the first-order scheme keeps 0.55 of the
/// peak vorticity and MacCormack 0.95.
#[test]
fn maccormack_keeps_more_of_an_inviscid_vortex_2d() {
    let first = run_2d(SolverConfig::default(), 64, 120);
    let second = run_2d(SolverConfig::default().with_advection(AdvectionScheme::MacCormack), 64, 120);
    let kept_first = first.peak[1] / first.peak[0];
    let kept_second = second.peak[1] / second.peak[0];
    println!("2D peak vorticity kept: first order {kept_first:.3}, MacCormack {kept_second:.3}");
    assert!(kept_second > kept_first, "MacCormack {kept_second} vs first order {kept_first}");
    // The steady solution loses nothing; MacCormack loses less than half of what the
    // first-order scheme loses.
    assert!(1.0 - kept_second < 0.5 * (1.0 - kept_first), "{kept_second} vs {kept_first}");
    // The clamp keeps it bounded: it may not invent rotation.
    assert!(kept_second <= 1.0 + 1e-9);
}

/// The same in 3D, on a Taylor-Green cell uniform in z.
#[test]
fn maccormack_keeps_more_of_an_inviscid_vortex_3d() {
    let run = |config: SolverConfig| {
        let n = 24;
        let mut g = FluidGrid3D::with_solver(n, n, 6, 0.0, 0.0, 1.0 / 60.0, config).unwrap();
        // 0.6 widths/s on 24 cells at 1/60 s: a Courant number of 0.24.
        let f = taylor_green(n, 1.0, 0.6);
        for i in 1..n - 1 {
            for j in 1..n - 1 {
                for k in 1..5 {
                    let (u, v) = f(i, j);
                    g.add_velocity(i, j, k, u, v, 0.0).unwrap();
                }
            }
        }
        let peak_curl = |g: &FluidGrid3D| {
            let inv_2h = 0.5 * n as f64;
            let mut m = 0.0f64;
            for i in 2..n - 2 {
                for j in 2..n - 2 {
                    let dv = g.get_velocity(i + 1, j, 3).unwrap().1 - g.get_velocity(i - 1, j, 3).unwrap().1;
                    let du = g.get_velocity(i, j + 1, 3).unwrap().0 - g.get_velocity(i, j - 1, 3).unwrap().0;
                    m = m.max(((dv - du) * inv_2h).abs());
                }
            }
            m
        };
        let before = peak_curl(&g);
        for _ in 0..60 {
            g.step();
        }
        assert!(g.validate_state().is_ok());
        peak_curl(&g) / before
    };
    let kept_first = run(SolverConfig::default());
    let kept_second = run(SolverConfig::default().with_advection(AdvectionScheme::MacCormack));
    println!("3D peak vorticity kept: first order {kept_first:.3}, MacCormack {kept_second:.3}");
    assert!(kept_second > kept_first, "MacCormack {kept_second} vs first order {kept_first}");
    assert!(kept_second <= 1.0 + 1e-9);
}

/// Confinement puts rotation back: a decaying vortex keeps more enstrophy with it on.
/// And the force goes in before the projection, so the step still ends projected: the
/// total divergence is zero (the walls' ghost cells make the sum telescope) and the
/// cell-by-cell divergence is at the same collocated floor as without confinement.
#[test]
fn confinement_raises_enstrophy_and_stays_projected() {
    // A converged pressure solve, so the divergence left is the collocated scheme's.
    let base = SolverConfig::default().with_pressure_tolerance(1e-10, 2000);
    let off = run_2d(base, 64, 120);
    let on = run_2d(base.with_vorticity_confinement(VorticityConfinement::MatchNumericalDissipation), 64, 120);

    let kept_off = off.enstrophy[1] / off.enstrophy[0];
    let kept_on = on.enstrophy[1] / on.enstrophy[0];
    let energy_on = on.energy[1] / on.energy[0];
    println!("enstrophy kept: off {kept_off:.3}, confined {kept_on:.3}; energy kept confined {energy_on:.3}");
    assert!(kept_on > kept_off, "confined {kept_on} vs off {kept_off}");
    // It puts back what the advection removed, not more than the flow had.
    assert!(energy_on <= 1.0 + 1e-9, "confinement grew the energy to {energy_on}");

    let total: f64 = on.divergence.iter().sum();
    let scale = peak(&on.divergence).max(1e-300) * on.divergence.len() as f64;
    assert!(total.abs() <= 1e-9 * scale, "total divergence {total}");
    let rms = |d: &[f64]| (sum_sq(d) / d.len() as f64).sqrt();
    let (div_on, div_off) = (rms(&on.divergence), rms(&off.divergence));
    println!("rms divergence: off {div_off:e}, confined {div_on:e}");
    let curl_scale = (on.enstrophy[1] / on.divergence.len() as f64).sqrt();
    assert!(div_on <= 1e-2 * curl_scale, "confined divergence {div_on} against curl {curl_scale}");
}

/// The same force in 3D: enstrophy up, energy not, on a z-uniform Taylor-Green cell.
#[test]
fn confinement_raises_enstrophy_3d() {
    let run = |config: SolverConfig| {
        let n = 20;
        let mut g = FluidGrid3D::with_solver(n, n, 6, 0.0, 0.0, 1.0 / 60.0, config).unwrap();
        let f = taylor_green(n, 1.0, 0.5);
        for i in 1..n - 1 {
            for j in 1..n - 1 {
                for k in 1..5 {
                    let (u, v) = f(i, j);
                    g.add_velocity(i, j, k, u, v, 0.0).unwrap();
                }
            }
        }
        let measure = |g: &FluidGrid3D| {
            let inv_2h = 0.5 * n as f64;
            let (mut enstrophy, mut energy) = (0.0, 0.0);
            for i in 2..n - 2 {
                for j in 2..n - 2 {
                    let dv = g.get_velocity(i + 1, j, 3).unwrap().1 - g.get_velocity(i - 1, j, 3).unwrap().1;
                    let du = g.get_velocity(i, j + 1, 3).unwrap().0 - g.get_velocity(i, j - 1, 3).unwrap().0;
                    enstrophy += ((dv - du) * inv_2h).powi(2);
                    let (u, v, w) = g.get_velocity(i, j, 3).unwrap();
                    energy += 0.5 * (u * u + v * v + w * w);
                }
            }
            (enstrophy, energy)
        };
        let (z0, e0) = measure(&g);
        for _ in 0..60 {
            g.step();
        }
        assert!(g.validate_state().is_ok());
        let (z1, e1) = measure(&g);
        (z1 / z0, e1 / e0)
    };
    let (z_off, _) = run(SolverConfig::default());
    let (z_on, e_on) =
        run(SolverConfig::default().with_vorticity_confinement(VorticityConfinement::MatchNumericalDissipation));
    println!("3D enstrophy kept: off {z_off:.3}, confined {z_on:.3}; energy kept confined {e_on:.3}");
    assert!(z_on > z_off, "confined {z_on} vs off {z_off}");
    assert!(e_on <= 1.0 + 1e-9, "confinement grew the energy to {e_on}");
}

/// The derived strength's offset: zero where a step moves the flow a whole number of
/// cells (semi-Lagrangian advection is then an exact shift and dissipates nothing),
/// largest at half a cell, and the same for either direction of travel.
#[test]
fn the_confinement_offset_is_the_fractional_cell_displacement() {
    use super::fluid_simulation::cell_offset;
    assert_eq!(cell_offset(0.0), 0.0);
    assert_eq!(cell_offset(3.0), 0.0);
    assert_eq!(cell_offset(0.5), 0.5);
    assert_eq!(cell_offset(-2.25), 0.25);
    assert_eq!(cell_offset(2.25), 0.25);
}

/// Both options together stay stable and bounded on a forced plume for a long run.
#[test]
fn both_options_stay_bounded_on_a_forced_plume() {
    let config = SolverConfig::default()
        .with_advection(AdvectionScheme::MacCormack)
        .with_vorticity_confinement(VorticityConfinement::MatchNumericalDissipation);
    let n = 48;
    let mut g = FluidGrid::with_solver(n, n, 0.0, 0.0, 1.0 / 60.0, config).unwrap();
    for _ in 0..600 {
        for i in n / 3..2 * n / 3 {
            g.add_density(i, 3, 1.0).unwrap();
            g.add_velocity(i, 3, 0.0, 0.05).unwrap();
        }
        g.step();
    }
    assert!(g.validate_state().is_ok());
    assert!(g.get_average_velocity() < 1.0, "{}", g.get_average_velocity());
}
