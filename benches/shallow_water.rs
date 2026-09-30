//! What does a river cost?
//!
//! The shallow-water solver is O(cells) per substep, and a substep does the same work
//! whatever the water is doing, so the number that matters for a game is how many cells
//! fit in a frame. This measures one 60 Hz frame of a river in steady flow down a
//! sloping plane that is wet from edge to edge -- the worst case, since dry cells are
//! cheaper -- on 1 m cells, where a 2 m deep river takes one substep per frame.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rs_physics::fluid_dynamics::{Boundary, Edge, ShallowWater, Threading};

/// From a village stream to a valley.
const SIDES: [usize; 4] = [64, 128, 256, 512];

fn river(n: usize) -> ShallowWater {
    let (slope, q, manning) = (0.002, 3.0, 0.03);
    let bed = (0..n * n).map(|k| slope * (n - k % n) as f64).collect();
    let mut water = ShallowWater::new(n, n, 1.0, bed).unwrap().with_manning(manning).unwrap();
    water.set_boundary(Edge::MinX, 0..n, Boundary::Inflow { discharge: q * n as f64 }).unwrap();
    water.set_boundary(Edge::MaxX, 0..n, Boundary::Open).unwrap();
    // Start at Manning's normal depth, so every sample times the same steady river.
    let depth: f64 = (q * manning / slope.sqrt()).powf(0.6);
    for j in 0..n {
        for i in 0..n {
            water.add_water(i, j, depth).unwrap();
            water.set_velocity(i, j, [q / depth, 0.0]).unwrap();
        }
    }
    water
}

fn frame(c: &mut Criterion) {
    let mut group = c.benchmark_group("shallow_water/frame_60hz");
    for &n in &SIDES {
        let mut water = river(n);
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| water.step(std::hint::black_box(1.0 / 60.0)).unwrap());
        });
    }
    group.finish();
}

/// The way a game runs a river: presentation-side, on one worker thread, at 30 Hz. At
/// these depths one substep covers a 30 Hz frame too, so this is one substep on one
/// thread.
fn frame_30hz_one_thread(c: &mut Criterion) {
    let mut group = c.benchmark_group("shallow_water/frame_30hz_one_thread");
    for &n in &SIDES {
        let mut water = river(n).with_threading(Threading::Serial);
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| water.step(std::hint::black_box(1.0 / 30.0)).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, frame, frame_30hz_one_thread);
criterion_main!(benches);
