//! What does a grid fluid step cost?
//!
//! Sizes span a small effect to a river reach. Every iteration injects a line of
//! dye and upward velocity and then steps, the way a game drives a smoke source:
//! the pressure solve's cost depends on how much new divergence each step brings,
//! and a grid left to decay would flatter a warm-started solver.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rs_physics::fluid_dynamics::{
    AdvectionScheme, FluidGrid, FluidGrid3D, PressureSolver, SolverConfig, VorticityConfinement,
};

fn source_2d(grid: &mut FluidGrid, n: usize) {
    for i in n / 4..3 * n / 4 {
        grid.add_density(i, n / 2, 1.0).unwrap();
        grid.add_velocity(i, n / 2, 0.0, 0.05).unwrap();
    }
}

fn source_3d(grid: &mut FluidGrid3D, n: usize) {
    for i in n / 4..3 * n / 4 {
        grid.add_density(i, n / 2, n / 2, 1.0).unwrap();
        grid.add_velocity(i, n / 2, n / 2, 0.0, 0.05, 0.0).unwrap();
    }
}

fn step_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fluid_grid/step_2d");
    group.sample_size(20);
    for &n in &[64usize, 128, 256] {
        let mut grid = FluidGrid::with_solver(n, n, 1e-5, 1e-5, 1.0 / 60.0, SolverConfig::default()).unwrap();
        // Settle past the cold start, so the samples time the steady forced state.
        for _ in 0..30 {
            source_2d(&mut grid, n);
            grid.step();
        }
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                source_2d(&mut grid, n);
                grid.step();
            });
        });
    }
    group.finish();
}

fn step_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fluid_grid/step_3d");
    group.sample_size(10);
    for &n in &[32usize, 64] {
        let mut grid =
            FluidGrid3D::with_solver(n, n, n, 1e-5, 1e-5, 1.0 / 60.0, SolverConfig::default()).unwrap();
        for _ in 0..10 {
            source_3d(&mut grid, n);
            grid.step();
        }
        group.throughput(Throughput::Elements((n * n * n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                source_3d(&mut grid, n);
                grid.step();
            });
        });
    }
    group.finish();
}

/// What the pressure solve's accuracy costs. One step of the same forced grid at each
/// conjugate-gradient tolerance (a relative residual), beside the old four-sweep
/// relaxation. On one cold step at 130², a tolerance of 0.3 leaves about 1e-4 of a
/// smooth divergence's energy in 12 iterations; 1e-2 leaves 2e-9 (54); the default 1e-4
/// leaves 2e-13 (91); 1e-8 is converged to rounding.
fn pressure_tolerance(c: &mut Criterion) {
    let configs = [
        ("cg_tol_0.3", SolverConfig::default().with_pressure_tolerance(0.3, 200)),
        ("cg_tol_1e-2", SolverConfig::default().with_pressure_tolerance(1e-2, 200)),
        ("cg_tol_1e-4_default", SolverConfig::default()),
        ("cg_tol_1e-8", SolverConfig::default().with_pressure_tolerance(1e-8, 400)),
        ("relaxation_4_sweeps_old", SolverConfig::default().with_pressure_solver(PressureSolver::Relaxation)),
    ];
    let mut group = c.benchmark_group("fluid_grid/pressure_tolerance_2d");
    group.sample_size(20);
    for &n in &[128usize, 256] {
        for (name, config) in &configs {
            let mut grid = FluidGrid::with_solver(n, n, 1e-5, 1e-5, 1.0 / 60.0, *config).unwrap();
            for _ in 0..30 {
                source_2d(&mut grid, n);
                grid.step();
            }
            group.bench_with_input(BenchmarkId::new(*name, n), &n, |b, _| {
                b.iter(|| {
                    source_2d(&mut grid, n);
                    grid.step();
                });
            });
        }
    }
    group.finish();

    let mut group = c.benchmark_group("fluid_grid/pressure_tolerance_3d");
    group.sample_size(10);
    let n = 64;
    for (name, config) in &configs {
        let mut grid = FluidGrid3D::with_solver(n, n, n, 1e-5, 1e-5, 1.0 / 60.0, *config).unwrap();
        for _ in 0..10 {
            source_3d(&mut grid, n);
            grid.step();
        }
        group.bench_with_input(BenchmarkId::new(*name, n), &n, |b, _| {
            b.iter(|| {
                source_3d(&mut grid, n);
                grid.step();
            });
        });
    }
    group.finish();
}

/// The turbulence options, each on and off, on the same forced source at every size, in
/// one run so the figures share the machine's state. The pressure iterations a step
/// ran are printed beside each, from the last settling step, so a change in the
/// projection's cost shows apart from the option's own passes.
fn options(c: &mut Criterion) {
    let configs = [
        ("off", SolverConfig::default()),
        ("maccormack", SolverConfig::default().with_advection(AdvectionScheme::MacCormack)),
        (
            "confinement",
            SolverConfig::default().with_vorticity_confinement(VorticityConfinement::MatchNumericalDissipation),
        ),
        (
            "both",
            SolverConfig::default()
                .with_advection(AdvectionScheme::MacCormack)
                .with_vorticity_confinement(VorticityConfinement::MatchNumericalDissipation),
        ),
    ];
    let mut group = c.benchmark_group("fluid_grid/options_2d");
    group.sample_size(20);
    for &n in &[64usize, 128, 256] {
        for (name, config) in &configs {
            let mut grid = FluidGrid::with_solver(n, n, 1e-5, 1e-5, 1.0 / 60.0, *config).unwrap();
            for _ in 0..30 {
                source_2d(&mut grid, n);
                grid.step();
            }
            eprintln!("options_2d {name}/{n}: {} pressure iterations a step", grid.get_last_pressure_iterations());
            group.bench_with_input(BenchmarkId::new(*name, n), &n, |b, _| {
                b.iter(|| {
                    source_2d(&mut grid, n);
                    grid.step();
                });
            });
        }
    }
    group.finish();

    let mut group = c.benchmark_group("fluid_grid/options_3d");
    group.sample_size(10);
    for &n in &[32usize, 64] {
        for (name, config) in &configs {
            let mut grid = FluidGrid3D::with_solver(n, n, n, 1e-5, 1e-5, 1.0 / 60.0, *config).unwrap();
            for _ in 0..10 {
                source_3d(&mut grid, n);
                grid.step();
            }
            eprintln!("options_3d {name}/{n}: {} pressure iterations a step", grid.get_last_pressure_iterations());
            group.bench_with_input(BenchmarkId::new(*name, n), &n, |b, _| {
                b.iter(|| {
                    source_3d(&mut grid, n);
                    grid.step();
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, step_2d, step_3d, pressure_tolerance, options);
criterion_main!(benches);
