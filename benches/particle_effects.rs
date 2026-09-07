//! Where does a GPU backend start paying for itself?
//!
//! `ParticleEffects::integrate` is pure data-parallel arithmetic over flat `f32`
//! arrays — the shape that ports directly to a CUDA kernel. But a kernel launch
//! plus a round trip costs tens of microseconds regardless of how little work it
//! does, so below some population the CPU wins outright and dispatching to the GPU
//! is a pessimization wearing an optimization's clothes.
//!
//! This bench exists to find that crossover with a number instead of an intuition.
//! Run it before wiring a GPU backend, and again after, on the same machine.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rs_physics::particles::{Burst, EffectRng, ParticleClass, ParticleEffects};

/// Populations spanning "a few sparks from one hit" to "a debris field".
const SCALES: [usize; 5] = [1_000, 10_000, 100_000, 500_000, 1_000_000];

fn filled_pool(count: usize) -> ParticleEffects {
    let mut fx = ParticleEffects::with_capacity(count);
    fx.set_class(
        0,
        ParticleClass {
            gravity: 26.0,
            drag: 1.4,
            restitution: 0.32,
        },
    );
    fx.set_class(
        1,
        ParticleClass {
            gravity: 1.6,
            drag: 3.4,
            restitution: 0.0,
        },
    );

    let mut rng = EffectRng::new(0xC0FFEE);
    // Two classes interleaved, because a single-class pool would let the class
    // lookup fold away entirely and flatter the result.
    for class in [0u8, 1u8] {
        fx.emit(
            &Burst {
                origin: [0.0, 40.0, 0.0],
                class,
                count: (count / 2) as u32,
                speed: 6.0..15.0,
                // Long enough that nothing retires mid-measurement, so every
                // iteration integrates the same population.
                lifetime: 10_000.0..10_001.0,
                size: 0.7..1.3,
                lift: 0.35,
            },
            &mut rng,
        );
    }
    fx
}

fn integrate(c: &mut Criterion) {
    let mut group = c.benchmark_group("particle_effects/integrate");

    for &n in &SCALES {
        let mut fx = filled_pool(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| fx.integrate(std::hint::black_box(1.0 / 60.0)));
        });
    }

    group.finish();
}

/// Ground collision is the host-side pass a GPU backend does *not* take over, so
/// its cost is what remains on the CPU either way. Measured separately for that
/// reason.
fn collide(c: &mut Criterion) {
    let mut group = c.benchmark_group("particle_effects/collide_ground");

    for &n in &[10_000usize, 100_000, 1_000_000] {
        let mut fx = filled_pool(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| fx.collide_ground(|x, z| (x * 0.01 + z * 0.01).sin() * 2.0));
        });
    }

    group.finish();
}

criterion_group!(benches, integrate, collide);
criterion_main!(benches);
