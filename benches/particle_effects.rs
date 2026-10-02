//! Where does a GPU backend start paying for itself?
//!
//! `ParticleEffects::integrate` is pure data-parallel arithmetic over flat `f32`
//! arrays -- the shape that ports directly to a CUDA kernel. But a kernel launch
//! plus a round trip costs tens of microseconds regardless of how little work it
//! does, so below some population the CPU wins outright and dispatching to the GPU
//! is a pessimization wearing an optimization's clothes.
//!
//! This bench exists to find that crossover with a number instead of an intuition.
//! Run it before wiring a GPU backend, and again after, on the same machine.
//!
//! # The population is live
//!
//! Until 2026-09-28 this bench emitted one population that never retired and
//! integrated it for as long as criterion liked, tens of thousands of steps for the
//! small pools. Drag took every horizontal velocity into the subnormal range, where
//! each multiply costs a microcode assist, so the bench read 21 to 25 ns a particle
//! against a real 1.3 to 1.5 ns, and larger pools (fewer steps) read *cheaper* a
//! particle. Any crossover computed from it put the GPU win about 15 times too early.
//!
//! Now the pool is what a game runs: sparks (0.2 to 0.5 s) and dust (20 to 40 s),
//! half each, retired as they expire and re-emitted every step at the rate that holds
//! the population, after 45 s of warm-up so the dust has its steady age spread. Only
//! `integrate` is timed; the emission between steps is not. `integrate` now also
//! zeroes a velocity too small to move its particle, so an aged pool no longer
//! degrades (see `examples/r2_bench.rs` for the aged and fresh controls, alternated in
//! one process).

use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rs_physics::particles::{Burst, EffectRng, ParticleClass, ParticleEffects, SwirlField, TurbulenceDrive};

/// Populations spanning "a few sparks from one hit" to "a debris field".
const SCALES: [usize; 5] = [1_000, 10_000, 100_000, 500_000, 1_000_000];

const DT: f32 = 1.0 / 60.0;
const SPARK_LIFE: core::ops::Range<f32> = 0.2..0.5;
const DUST_LIFE: core::ops::Range<f32> = 20.0..40.0;

/// A pool of `count` particles in steady state, re-emitting what retires.
struct Live {
    fx: ParticleEffects,
    rng: EffectRng,
    owed: [f32; 2],
    rate: [f32; 2],
}

impl Live {
    fn new(count: usize) -> Live {
        Live::in_air(count, None)
    }

    /// The same pool, with both classes on the air (`swirl` 1, the physical coupling)
    /// when `air` is given, warmed up in it so the population has the spread the
    /// swirl gives it. The field is advanced at 10 Hz during the warm-up.
    fn in_air(count: usize, mut air: Option<&mut SwirlField>) -> Live {
        let swirl = if air.is_some() { 1.0 } else { 0.0 };
        let mut fx = ParticleEffects::with_capacity(count + count / 4);
        fx.set_class(0, ParticleClass { gravity: 26.0, drag: 1.4, restitution: 0.32, swirl });
        fx.set_class(1, ParticleClass { gravity: 1.6, drag: 3.4, restitution: 0.0, swirl });
        // Two classes interleaved, because a single-class pool would let the class
        // lookup fold away entirely and flatter the result. Steady state: population
        // is the emission rate times the mean lifetime.
        let half = count as f32 / 2.0;
        let rate = [half / 0.35, half / 30.0];
        let mut live = Live { fx, rng: EffectRng::new(0xC0FFEE), owed: [0.0; 2], rate };
        for step in 0..(45.0 / DT) as usize {
            live.feed();
            match air.as_deref_mut() {
                Some(field) => {
                    if step % 6 == 0 {
                        field.advance(6.0 * DT);
                    }
                    live.fx.integrate_in_air(DT, field.velocity());
                }
                None => live.fx.integrate(DT),
            }
        }
        live
    }

    fn feed(&mut self) {
        for c in 0..2 {
            self.owed[c] += self.rate[c] * DT;
            let whole = self.owed[c].floor();
            self.owed[c] -= whole;
            self.fx.emit(
                &Burst {
                    origin: [0.0, 40.0, 0.0],
                    class: c as u8,
                    count: whole as u32,
                    speed: 6.0..15.0,
                    lifetime: if c == 0 { SPARK_LIFE } else { DUST_LIFE },
                    size: 0.7..1.3,
                    lift: 0.35,
                },
                &mut self.rng,
            );
        }
    }
}

fn integrate(c: &mut Criterion) {
    let mut group = c.benchmark_group("particle_effects/integrate");

    for &n in &SCALES {
        let mut live = Live::new(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter_custom(|iters| {
                let mut spent = Duration::ZERO;
                for _ in 0..iters {
                    live.feed();
                    let started = Instant::now();
                    live.fx.integrate(std::hint::black_box(DT));
                    spent += started.elapsed();
                }
                spent
            });
        });
    }

    group.finish();
}

/// Ground collision is the host-side pass a GPU backend does *not* take over, so
/// its cost is what remains on the CPU either way. Measured separately for that
/// reason, on the same live pool.
fn collide(c: &mut Criterion) {
    let mut group = c.benchmark_group("particle_effects/collide_ground");

    for &n in &[10_000usize, 100_000, 1_000_000] {
        let mut live = Live::new(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| live.fx.collide_ground(|x, z| (x * 0.01 + z * 0.01).sin() * 2.0));
        });
    }

    group.finish();
}

/// The swirl the field-on case integrates through: 32^3 cells of 2 m over the 64 m
/// around the emitter, driven by a 3 m/s plume 12 m wide (octaves of 4 and 8 m).
fn swirl_field() -> SwirlField {
    let drive = TurbulenceDrive::new(3.0, 12.0).unwrap();
    SwirlField::new([-32.0, 8.0, -32.0], [32, 32, 32], 2.0, drive, 0x5EED).unwrap()
}

/// The gate for the swirl: `integrate_in_air` with both classes on the air against
/// `integrate` on the same live population, adjacent in one run at every scale. The
/// field-on cost a particle must stay under twice the field-off cost.
fn air(c: &mut Criterion) {
    let mut group = c.benchmark_group("particle_effects/air");
    for &n in &SCALES {
        let mut off = Live::new(n);
        let mut field = swirl_field();
        let mut on = Live::in_air(n, Some(&mut field));
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("off", n), &n, |b, _| {
            b.iter_custom(|iters| {
                let mut spent = Duration::ZERO;
                for _ in 0..iters {
                    off.feed();
                    let started = Instant::now();
                    off.fx.integrate(std::hint::black_box(DT));
                    spent += started.elapsed();
                }
                spent
            });
        });
        group.bench_with_input(BenchmarkId::new("on", n), &n, |b, _| {
            b.iter_custom(|iters| {
                let mut spent = Duration::ZERO;
                for _ in 0..iters {
                    on.feed();
                    let started = Instant::now();
                    on.fx.integrate_in_air(std::hint::black_box(DT), field.velocity());
                    spent += started.elapsed();
                }
                spent
            });
        });
    }
    group.finish();

    // What a swirl update costs, on whichever thread calls it.
    let mut group = c.benchmark_group("particle_effects/swirl_advance");
    group.sample_size(20);
    for &n in &[32usize, 64] {
        let drive = TurbulenceDrive::new(3.0, 40.0).unwrap();
        let mut field = SwirlField::new([0.0; 3], [n, n, n], 2.0, drive, 1).unwrap();
        eprintln!("swirl {n}^3: {} bytes", field.bytes());
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| field.advance(std::hint::black_box(0.1)));
        });
    }
    group.finish();
}

criterion_group!(benches, integrate, collide, air);
criterion_main!(benches);
