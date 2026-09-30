//! What does a grid fluid step cost?
//!
//! `FluidGrid::step` and `FluidGrid3D::step` do a fixed amount of work per call —
//! `iterations` Gauss-Seidel sweeps for each of the diffusion and pressure solves,
//! plus the advection passes — so the cost does not depend on what is in the grid,
//! and one grid can be stepped repeatedly. Sizes span a small effect to a river reach.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rs_physics::fluid_dynamics::{FluidGrid, FluidGrid3D, SolverConfig};

fn seeded_2d(n: usize, config: SolverConfig) -> FluidGrid {
    let mut grid = FluidGrid::with_solver(n, n, 1e-5, 1e-5, 1.0 / 60.0, config).unwrap();
    for i in n / 4..3 * n / 4 {
        grid.add_density(i, n / 2, 1.0).unwrap();
        grid.add_velocity(i, n / 2, 0.0, 0.5).unwrap();
    }
    grid
}

fn seeded_3d(n: usize, config: SolverConfig) -> FluidGrid3D {
    let mut grid = FluidGrid3D::with_solver(n, n, n, 1e-5, 1e-5, 1.0 / 60.0, config).unwrap();
    for i in n / 4..3 * n / 4 {
        grid.add_density(i, n / 2, n / 2, 1.0).unwrap();
        grid.add_velocity(i, n / 2, n / 2, 0.0, 0.5, 0.0).unwrap();
    }
    grid
}

fn step_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fluid_grid/step_2d");
    group.sample_size(20);
    for &n in &[64usize, 128, 256] {
        let mut grid = seeded_2d(n, SolverConfig::default());
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| grid.step());
        });
    }
    group.finish();
}

fn step_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fluid_grid/step_3d");
    group.sample_size(10);
    for &n in &[32usize, 64] {
        let mut grid = seeded_3d(n, SolverConfig::default());
        group.throughput(Throughput::Elements((n * n * n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| grid.step());
        });
    }
    group.finish();
}

criterion_group!(benches, step_2d, step_3d);
criterion_main!(benches);
