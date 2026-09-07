//! What does a splash cost?
//!
//! SPH is O(n) in particles but with a large constant: every particle walks 27 grid
//! cells and evaluates three kernels per neighbour. The number that matters for a
//! game is how many particles fit in a frame budget, so that is what this measures.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rs_physics::fluid_dynamics::{SphFluid, SphParams};

/// Populations spanning one wound to a massacre.
const SCALES: [usize; 4] = [64, 256, 1_024, 4_096];

fn filled(count: usize) -> SphFluid {
    let params = SphParams::blood();
    let spacing = params.smoothing_radius * 0.5;
    let mut fluid = SphFluid::new(params, count).unwrap();

    // A packed cube, which is the worst case: every particle has a full complement
    // of neighbours. A dispersed splash is cheaper.
    let side = (count as f64).cbrt().ceil() as usize;
    'outer: for x in 0..side {
        for y in 0..side {
            for z in 0..side {
                if !fluid.spawn(
                    [
                        x as f64 * spacing,
                        1.0 + y as f64 * spacing,
                        z as f64 * spacing,
                    ],
                    [0.0; 3],
                ) {
                    break 'outer;
                }
            }
        }
    }
    fluid
}

fn step(c: &mut Criterion) {
    let mut group = c.benchmark_group("sph/step");

    for &n in &SCALES {
        let mut fluid = filled(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                fluid.step(std::hint::black_box(1.0 / 240.0), 9.81, |_, _| 0.0);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, step);
criterion_main!(benches);
