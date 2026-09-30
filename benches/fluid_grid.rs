//! What does a grid fluid step cost?
//!
//! Sizes span a small effect to a river reach. Every iteration injects a line of
//! dye and upward velocity and then steps, the way a game drives a smoke source:
//! the pressure solve's cost depends on how much new divergence each step brings,
//! and a grid left to decay would flatter a warm-started solver.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rs_physics::fluid_dynamics::{FluidGrid, FluidGrid3D, SolverConfig};

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

criterion_group!(benches, step_2d, step_3d);
criterion_main!(benches);
