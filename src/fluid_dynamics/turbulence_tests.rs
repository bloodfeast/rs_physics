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
        (
            amp * (k * x).sin() * (k * y).cos(),
            -amp * (k * x).cos() * (k * y).sin(),
        )
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

/// The curl `dv/dx - du/dy` at every fluid cell not next to a wall, and the central
/// divergence at every fluid cell, by central differences in widths.
fn curl_and_divergence_2d(g: &FluidGrid) -> (Vec<f64>, Vec<f64>) {
    let (w, h) = (g.get_width(), g.get_height());
    let inv_2h = 0.5 * w as f64;
    let v = |i: usize, j: usize| g.get_velocity(i, j).unwrap();
    let (mut curl, mut div) = (Vec::new(), Vec::new());
    for i in 1..w - 1 {
        for j in 1..h - 1 {
            if i > 1 && j > 1 && i < w - 2 && j < h - 2 {
                curl.push(
                    ((v(i + 1, j).1 - v(i - 1, j).1) - (v(i, j + 1).0 - v(i, j - 1).0)) * inv_2h,
                );
            }
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

fn run_2d(config: SolverConfig, n: usize, cells: f64, steps: usize) -> Run2d {
    // 0.3 widths/s at 1/60 s on 64 cells is a Courant number of 0.32 at the peak: the
    // fractional offsets where linear interpolation is most diffusive.
    let mut g = FluidGrid::with_solver(n, n, 0.0, 0.0, 1.0 / 60.0, config).unwrap();
    set_velocity_2d(&mut g, taylor_green(n, cells, 0.3));
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

/// MacCormack keeps more of an inviscid vortex than first-order advection.
///
/// A 4 x 4 Taylor-Green array (eddies 15 cells across) at 64^2, 120 steps (2 s).
/// Measured 2026-10-02: the first-order scheme keeps 0.68 of the peak vorticity and
/// 0.36 of the enstrophy; MacCormack keeps 0.82 and 0.80. On a 2 x 2 array the gap
/// is 0.73 against 0.94 in enstrophy, and on 8 x 8 (eddies of 7 cells) 0.07 against
/// 0.46. The peak is the noisier of the two figures (a single cell), so the margin is
/// asserted on the enstrophy.
#[test]
fn maccormack_keeps_more_of_an_inviscid_vortex_2d() {
    let first = run_2d(SolverConfig::default(), 64, 4.0, 120);
    let second = run_2d(
        SolverConfig::default().with_advection(AdvectionScheme::MacCormack),
        64,
        4.0,
        120,
    );
    let kept_first = first.peak[1] / first.peak[0];
    let kept_second = second.peak[1] / second.peak[0];
    let z_first = first.enstrophy[1] / first.enstrophy[0];
    let z_second = second.enstrophy[1] / second.enstrophy[0];
    println!(
        "2D peak vorticity kept: first order {kept_first:.3}, MacCormack {kept_second:.3}; enstrophy {z_first:.3}, {z_second:.3}"
    );
    assert!(
        kept_second > kept_first,
        "MacCormack {kept_second} vs first order {kept_first}"
    );
    // The steady solution loses nothing; MacCormack loses less than half of what the
    // first-order scheme loses.
    assert!(
        1.0 - z_second < 0.5 * (1.0 - z_first),
        "{z_second} vs {z_first}"
    );
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
                    let dv = g.get_velocity(i + 1, j, 3).unwrap().1
                        - g.get_velocity(i - 1, j, 3).unwrap().1;
                    let du = g.get_velocity(i, j + 1, 3).unwrap().0
                        - g.get_velocity(i, j - 1, 3).unwrap().0;
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
    assert!(
        kept_second > kept_first,
        "MacCormack {kept_second} vs first order {kept_first}"
    );
    assert!(kept_second <= 1.0 + 1e-9);
}

/// Confinement puts rotation back: a decaying vortex keeps more enstrophy with it on,
/// and never more energy than it started with, because each step's confinement is
/// capped at the kinetic energy that step's advection removed.
///
/// Uncapped, matched to the numerical dissipation at the grid scale, it added energy to
/// a vortex this smooth (1.39 times the start after 2 s on this 2 x 2 array), because a
/// structure many cells across is confined more than it dissipates. Capped, measured
/// 2026-10-02: enstrophy 0.980 and energy 0.936 of the start after 2 s, against 0.732
/// and 0.731 without confinement.
///
/// The force goes in before the projection, so the total divergence after the step is
/// still zero (the walls' ghost cells make the sum telescope). Cell by cell it is not at
/// the unconfined floor: the grids are collocated, so the projection cannot remove a
/// divergence at the grid scale (the wide Laplacian `D G` and the compact one the
/// pressure solve inverts differ there), and confinement feeds exactly that scale.
/// Measured with a converged pressure solve: rms central divergence 5.3e-4 without
/// confinement and 1.7e-3 with it capped (0.118 uncapped), against an rms vorticity of
/// about 3.
#[test]
fn confinement_raises_enstrophy_and_stays_projected() {
    // A converged pressure solve, so the divergence left is the collocated scheme's.
    let base = SolverConfig::default().with_pressure_tolerance(1e-10, 2000);
    let off = run_2d(base, 64, 2.0, 120);
    let on = run_2d(
        base.with_vorticity_confinement(VorticityConfinement::MatchNumericalDissipation),
        64,
        2.0,
        120,
    );

    let kept_off = off.enstrophy[1] / off.enstrophy[0];
    let kept_on = on.enstrophy[1] / on.enstrophy[0];
    let energy_on = on.energy[1] / on.energy[0];
    println!("enstrophy kept: off {kept_off:.3}, confined {kept_on:.3}; energy kept confined {energy_on:.3}");
    assert!(kept_on > kept_off, "confined {kept_on} vs off {kept_off}");
    assert!(
        energy_on <= 1.0 + 1e-9,
        "confinement grew the energy to {energy_on}"
    );

    let total: f64 = on.divergence.iter().sum();
    let scale = peak(&on.divergence).max(1e-300) * on.divergence.len() as f64;
    assert!(total.abs() <= 1e-9 * scale, "total divergence {total}");
    let rms = |d: &[f64]| (sum_sq(d) / d.len() as f64).sqrt();
    let (div_on, div_off) = (rms(&on.divergence), rms(&off.divergence));
    println!("rms divergence: off {div_off:e}, confined {div_on:e}");
    let curl_scale = (on.enstrophy[1] / on.divergence.len() as f64).sqrt();
    assert!(
        div_off <= 1e-3 * curl_scale,
        "unconfined divergence {div_off} against curl {curl_scale}"
    );
    assert!(
        div_on <= 0.1 * curl_scale,
        "confined divergence {div_on} against curl {curl_scale}"
    );
}

/// The same force in 3D raises enstrophy on a z-uniform Taylor-Green cell, and the cap
/// keeps the energy at or below where it started.
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
                    let dv = g.get_velocity(i + 1, j, 3).unwrap().1
                        - g.get_velocity(i - 1, j, 3).unwrap().1;
                    let du = g.get_velocity(i, j + 1, 3).unwrap().0
                        - g.get_velocity(i, j - 1, 3).unwrap().0;
                    enstrophy += ((dv - du) * inv_2h).powi(2);
                }
            }
            // The energy over every fluid cell: the cap bounds the whole box's.
            for i in 1..n - 1 {
                for j in 1..n - 1 {
                    for k in 1..5 {
                        let (u, v, w) = g.get_velocity(i, j, k).unwrap();
                        energy += 0.5 * (u * u + v * v + w * w);
                    }
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
    let (z_on, e_on) = run(SolverConfig::default()
        .with_vorticity_confinement(VorticityConfinement::MatchNumericalDissipation));
    println!(
        "3D enstrophy kept: off {z_off:.3}, confined {z_on:.3}; energy kept confined {e_on:.3}"
    );
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
    assert!(
        g.get_average_velocity() < 1.0,
        "{}",
        g.get_average_velocity()
    );
}

/// What each combination keeps of the 2 x 2 and 4 x 4 inviscid Taylor-Green arrays at
/// 64^2 over 2 s, printed for the record; the assertions are in the tests above.
#[test]
#[ignore = "a table of figures, run by hand"]
fn table_of_what_each_option_keeps() {
    let configs = [
        ("first order", SolverConfig::default()),
        (
            "MacCormack",
            SolverConfig::default().with_advection(AdvectionScheme::MacCormack),
        ),
        (
            "confinement",
            SolverConfig::default()
                .with_vorticity_confinement(VorticityConfinement::MatchNumericalDissipation),
        ),
        (
            "both",
            SolverConfig::default()
                .with_advection(AdvectionScheme::MacCormack)
                .with_vorticity_confinement(VorticityConfinement::MatchNumericalDissipation),
        ),
    ];
    for cells in [2.0, 4.0] {
        for (name, config) in configs {
            let r = run_2d(config, 64, cells, 120);
            println!(
                "{cells} x {cells} {name:<12}: peak {:.3} enstrophy {:.3} energy {:.3}",
                r.peak[1] / r.peak[0],
                r.enstrophy[1] / r.enstrophy[0],
                r.energy[1] / r.energy[0]
            );
        }
    }
}

/// The cap's scale: the largest fraction of the confinement velocity whose added energy,
/// `s along + s^2 spread`, stays within the loss.
#[test]
fn the_cap_adds_exactly_the_loss_or_all_of_the_force() {
    use super::fluid_simulation::confinement_scale;
    // Nothing lost, nothing added.
    assert_eq!(confinement_scale(0.0, 1.0, 1.0), 0.0);
    assert_eq!(confinement_scale(-1.0, 1.0, 1.0), 0.0);
    // More loss than the whole force adds: all of it.
    assert_eq!(confinement_scale(10.0, 1.0, 1.0), 1.0);
    // A force that takes energy out is never cut.
    assert_eq!(confinement_scale(0.1, -5.0, 1.0), 1.0);
    // Otherwise the energy added is the loss, to rounding.
    for (loss, along, spread) in [
        (0.5, 1.0, 1.0),
        (1e-6, 0.3, 2.0),
        (0.2, 0.0, 4.0),
        (0.3, 1.0, 0.0),
    ] {
        let s = confinement_scale(loss, along, spread);
        let added = s * along + s * s * spread;
        assert!(
            (added - loss).abs() <= 1e-12 * loss.max(1.0),
            "{loss} {along} {spread}: {s} adds {added}"
        );
    }
}

/// MacCormack and capped confinement together: MacCormack removes so little that the
/// cap leaves confinement almost nothing to put back. Measured 2026-10-02 on the 4 x 4
/// array at 64^2 over 2 s: enstrophy 0.804 of the start with both, 0.800 with MacCormack
/// alone; on the 2 x 2 array, 0.941 either way.
#[test]
fn maccormack_with_capped_confinement_keeps_at_least_maccormack() {
    let alone = run_2d(
        SolverConfig::default().with_advection(AdvectionScheme::MacCormack),
        64,
        4.0,
        120,
    );
    let both = run_2d(
        SolverConfig::default()
            .with_advection(AdvectionScheme::MacCormack)
            .with_vorticity_confinement(VorticityConfinement::MatchNumericalDissipation),
        64,
        4.0,
        120,
    );
    let (z_alone, z_both) = (
        alone.enstrophy[1] / alone.enstrophy[0],
        both.enstrophy[1] / both.enstrophy[0],
    );
    let e_both = both.energy[1] / both.energy[0];
    println!("MacCormack enstrophy {z_alone:.3}; with capped confinement {z_both:.3}, energy {e_both:.3}");
    assert!(z_both >= z_alone - 1e-9, "{z_both} vs {z_alone}");
    assert!(e_both <= 1.0 + 1e-9, "{e_both}");
}
