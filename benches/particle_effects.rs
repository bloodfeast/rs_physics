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
        Live::in_air(count, None, 1)
    }

    /// The same pool, with both classes on the air (`swirl` 1, the physical coupling)
    /// when `air` is given, warmed up in it so the population has the spread the swirl
    /// gives it. The field is advanced every `period` frames and the particles refresh
    /// their samples over that period.
    fn in_air(count: usize, mut air: Option<&mut SwirlField>, period: usize) -> Live {
        let mut fx = ParticleEffects::with_capacity(count + count / 4);
        fx.set_class(0, ParticleClass { gravity: 26.0, drag: 1.4, restitution: 0.32 });
        fx.set_class(1, ParticleClass { gravity: 1.6, drag: 3.4, restitution: 0.0 });
        if air.is_some() {
            fx.set_swirl(0, 1.0);
            fx.set_swirl(1, 1.0);
        }
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
                    if step % period == 0 {
                        field.advance(period as f32 * DT);
                    }
                    live.fx.integrate_in_air(DT, field.velocity(), period as f32 * DT);
                }
                None => live.fx.integrate(DT),
            }
        }
        live
    }

    fn feed(&mut self) {
        let Live { fx, rng, owed, rate } = self;
        Self::emit_owed(owed, *rate, rng, |burst, rng| fx.emit(burst, rng));
    }

    /// This pool's emission for one frame, into a GPU pool instead of its own.
    #[cfg(feature = "gpu")]
    fn feed_into(&mut self, pool: &mut rs_physics::gpu::GpuParticlePool) {
        Self::emit_owed(&mut self.owed, self.rate, &mut self.rng, |burst, rng| pool.emit(burst, rng));
    }

    fn emit_owed(owed: &mut [f32; 2], rate: [f32; 2], rng: &mut EffectRng, mut emit: impl FnMut(&Burst, &mut EffectRng)) {
        for c in 0..2 {
            owed[c] += rate[c] * DT;
            let whole = owed[c].floor();
            owed[c] -= whole;
            emit(
                &Burst {
                    origin: [0.0, 40.0, 0.0],
                    class: c as u8,
                    count: whole as u32,
                    speed: 6.0..15.0,
                    lifetime: if c == 0 { SPARK_LIFE } else { DUST_LIFE },
                    size: 0.7..1.3,
                    lift: 0.35,
                },
                rng,
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
/// `integrate` on the same live population, adjacent in one run at every scale, with
/// the field updated at 10 Hz (every 6 frames) and at 20 Hz (every 3) and each
/// particle refreshing its sample once a field period. The field's own update is
/// outside the timed span (it is reported by `swirl_advance`). The field-on cost a
/// particle must stay under twice the field-off cost.
fn air(c: &mut Criterion) {
    let mut group = c.benchmark_group("particle_effects/air");
    for &n in &SCALES {
        let mut off = Live::new(n);
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
        for (name, period) in [("on_10hz", 6usize), ("on_20hz", 3)] {
            let mut field = swirl_field();
            let mut on = Live::in_air(n, Some(&mut field), period);
            let mut frame = 0usize;
            group.bench_with_input(BenchmarkId::new(name, n), &n, |b, _| {
                b.iter_custom(|iters| {
                    let mut spent = Duration::ZERO;
                    for _ in 0..iters {
                        on.feed();
                        if frame % period == 0 {
                            field.advance(period as f32 * DT);
                        }
                        frame += 1;
                        let started = Instant::now();
                        on.fx.integrate_in_air(std::hint::black_box(DT), field.velocity(), period as f32 * DT);
                        spent += started.elapsed();
                    }
                    spent
                });
            });
        }
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

/// The plume gate: `integrate_in_air` reading a `PlumeField` whose worker steps at
/// 20 Hz on its own thread beside the bench, against `integrate` on the same live
/// population in the same run, each particle refreshing its sample once a worker
/// period. The timed span is the frame thread's whole cost: taking the newest frame
/// and integrating through it.
#[cfg(feature = "fluid_simulation")]
fn air_with_plume_worker(c: &mut Criterion) {
    use rs_physics::fluid_dynamics::{PlumeField, PlumeRegion, PlumeSource};
    let region = PlumeRegion { origin: [-32.0, 8.0, -32.0], cells: [32, 32, 32], cell_size: 2.0 };
    let source = PlumeSource {
        position: [0.0, 9.0, 0.0],
        drive: TurbulenceDrive::new(3.0, 12.0).unwrap(),
        smoke_rate: 1.0,
    };
    let mut group = c.benchmark_group("particle_effects/air_with_plume_worker");
    for &n in &[100_000usize, 1_000_000] {
        let mut off = Live::new(n);
        let mut field = swirl_field();
        let mut on = Live::in_air(n, Some(&mut field), 3);
        let (plume, mut air) = PlumeField::new(region, source, [2.0, 0.0, 0.5], 0x5EED).unwrap();
        let worker = plume.spawn(20.0).unwrap();
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
                    let frame = air.latest();
                    on.fx.integrate_in_air(std::hint::black_box(DT), frame.velocity(), frame.interval());
                    spent += started.elapsed();
                }
                spent
            });
        });
        let frame = air.latest();
        eprintln!(
            "plume worker at step {}, last step cost {:.2} ms",
            frame.step(),
            frame.step_cost().as_secs_f64() * 1e3
        );
        drop(worker);
    }
    group.finish();
}

/// The resident GPU pool (`gpu` feature, where an adapter exists, on a context of its
/// own from `GpuContext::with_features`): the same live population emitted into a
/// `GpuParticlePool` with both classes on an `rgba16float` swirl updated every 3 frames
/// (20 Hz) and uploaded the frame it changes. Two figures a population, both per frame:
///
/// - `device`: the GPU time of the emit and integrate passes, from timestamp queries
///   around them (what the policy's GPU cost model calls the dispatch);
/// - `host`: the frame thread's cost of the GPU path, staging the emission plus the
///   encode and its one buffer write, plus the field upload on the frames it happens.
///
/// Set `WGPU_VALIDATION_INDIRECT_CALL=0` to measure as the engine's release build runs
/// (wgpu 30 otherwise adds a validation dispatch to every indirect dispatch).
#[cfg(feature = "gpu")]
fn gpu_pool(c: &mut Criterion) {
    use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig, PoolTimestamps};

    let Some(gpu) = GpuContext::with_features(wgpu::Features::TIMESTAMP_QUERY) else {
        eprintln!("gpu_pool: no adapter, skipped");
        return;
    };
    if !gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        eprintln!("gpu_pool: no timestamp queries, skipped");
        return;
    }
    // Timestamps a chunk of frames at a time: four a frame (emit and integrate).
    const CHUNK: usize = 256;
    let queries = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        ty: wgpu::QueryType::Timestamp,
        count: (4 * CHUNK) as u32,
    });
    let bytes = (4 * CHUNK * 8) as u64;
    let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let period = gpu.queue.get_timestamp_period() as f64;

    let mut group = c.benchmark_group("particle_effects/gpu_pool");
    for &n in &[10_000usize, 100_000, 1_000_000] {
        // The live pool's emission, into the GPU pool instead.
        let mut feed = Live::new(1);
        let half = n as f32 / 2.0;
        feed.rate = [half / 0.35, half / 30.0];
        let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new((n + n / 4) as u32)).unwrap();
        for (c, class) in [feed.fx.class(0), feed.fx.class(1)].into_iter().enumerate() {
            pool.set_class(c as u8, class);
            pool.set_swirl(c as u8, 1.0);
        }
        let mut field = swirl_field();
        let mut frame = 0usize;
        let mut run = |frames: usize, timed: bool, pool: &mut GpuParticlePool, feed: &mut Live| -> (Duration, Duration) {
            let mut host = Duration::ZERO;
            let mut device_ns = 0.0f64;
            let mut done = 0;
            while done < frames {
                let chunk = (frames - done).min(CHUNK);
                for f in 0..chunk {
                    let started = Instant::now();
                    feed.feed_into(pool);
                    if frame % 3 == 0 {
                        field.advance(3.0 * DT);
                        pool.upload_field(field.velocity());
                    }
                    frame += 1;
                    let mut encoder = gpu.device.create_command_encoder(&Default::default());
                    let base = (4 * f) as u32;
                    pool.encode_timed(
                        &mut encoder,
                        DT,
                        timed.then_some(PoolTimestamps { query_set: &queries, field: None, emit: Some(base), integrate: Some(base + 2) }),
                    );
                    host += started.elapsed();
                    gpu.queue.submit([encoder.finish()]);
                }
                if timed {
                    let mut encoder = gpu.device.create_command_encoder(&Default::default());
                    encoder.resolve_query_set(&queries, 0..(4 * chunk) as u32, &resolve, 0);
                    encoder.copy_buffer_to_buffer(&resolve, 0, &read, 0, (4 * chunk * 8) as u64);
                    gpu.queue.submit([encoder.finish()]);
                    let slice = read.slice(..);
                    slice.map_async(wgpu::MapMode::Read, |_| {});
                    gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                    {
                        let view = slice.get_mapped_range().unwrap();
                        let ticks: &[u64] = bytemuck::cast_slice(&view);
                        for f in 0..chunk {
                            for pass in [0, 2] {
                                device_ns += ticks[4 * f + pass + 1].wrapping_sub(ticks[4 * f + pass]) as f64 * period;
                            }
                        }
                    }
                    read.unmap();
                } else {
                    gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                }
                done += chunk;
            }
            (host, Duration::from_nanos(device_ns as u64))
        };
        // The 45 s warm-up of the CPU pools, so the dust has its age spread.
        run((45.0 / DT) as usize, false, &mut pool, &mut feed);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("device", n), &n, |b, _| {
            b.iter_custom(|iters| run(iters as usize, true, &mut pool, &mut feed).1);
        });
        group.bench_with_input(BenchmarkId::new("host", n), &n, |b, _| {
            b.iter_custom(|iters| run(iters as usize, false, &mut pool, &mut feed).0);
        });
    }
    group.finish();
}

#[cfg(all(feature = "fluid_simulation", feature = "gpu"))]
criterion_group!(benches, integrate, collide, air, air_with_plume_worker, gpu_pool);
#[cfg(all(feature = "fluid_simulation", not(feature = "gpu")))]
criterion_group!(benches, integrate, collide, air, air_with_plume_worker);
#[cfg(all(not(feature = "fluid_simulation"), feature = "gpu"))]
criterion_group!(benches, integrate, collide, air, gpu_pool);
#[cfg(all(not(feature = "fluid_simulation"), not(feature = "gpu")))]
criterion_group!(benches, integrate, collide, air);
criterion_main!(benches);
