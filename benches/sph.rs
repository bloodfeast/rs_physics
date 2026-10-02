//! What does a splash cost?
//!
//! SPH is O(n) in particles but with a large constant: every particle walks 27 grid
//! cells and evaluates three kernels per neighbour. The number that matters for a
//! game is how many particles fit in a frame budget, so that is what this measures.
//!
//! Since 2026-09-28 the step runs in parallel on the calling rayon pool, here the
//! global one, so these figures are whole-machine figures. The per-thread-count and
//! per-phase numbers come from `examples/r2_bench.rs` and `examples/r2_phases.rs`,
//! which alternate 1, 4 and 8 threads inside one process.
//!
//! `sph/solids` prices `step_with_solids` in the same run: solids out of every
//! particle's reach (which must cost what `step` costs, the binning being the whole
//! price) and a shin wading through a resting pool, each beside its plain-step twin, and
//! then prints per-phase medians with the fraction of particles that ran the contact test.

use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rs_physics::fluid_dynamics::{SphFluid, SphParams, SphSolids};

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
            // Zero gravity. Every iteration steps the same fluid, and under gravity
            // the cube hit the floor within ~100 steps, so almost every sample
            // timed a puddle rather than the packed cube this is meant to measure.
            // At rest spacing and without gravity it stays packed.
            b.iter(|| {
                fluid.step(std::hint::black_box(1.0 / 240.0), 0.0, |_, _| 0.0);
            });
        });
    }

    group.finish();
}

/// A squad's worth of limbs and four crates, three metres from the liquid: present, and
/// out of every particle's reach. What this costs over `step` is the binning alone.
fn far_solids() -> SphSolids {
    let mut solids = SphSolids::with_capacity(24, 4);
    for k in 0..24 {
        let x = 3.0 + 0.3 * (k % 6) as f64;
        let z = 0.3 * (k / 6) as f64;
        solids.push_capsule(
            [x, 0.1, z],
            [x, 0.5, z],
            0.06,
            [1.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        );
    }
    for k in 0..4 {
        let x = -3.0 - 0.8 * k as f64;
        solids.push_box([x, 0.25, 0.0], [0.25; 3], 0.3 * k as f64, [0.0; 3]);
    }
    solids
}

/// A resting pool four particles deep, settled on level ground.
fn pool(count: usize) -> SphFluid {
    let params = SphParams::blood();
    let spacing = params.smoothing_radius * 0.5;
    let mut fluid = SphFluid::new(params, count).unwrap();
    let side = ((count / 4) as f64).sqrt().ceil() as usize;
    'outer: for x in 0..side {
        for z in 0..side {
            for y in 0..4 {
                let p = [
                    x as f64 * spacing,
                    0.5 * spacing + y as f64 * spacing,
                    z as f64 * spacing,
                ];
                if !fluid.spawn(p, [0.0; 3]) {
                    break 'outer;
                }
            }
        }
    }
    for _ in 0..240 {
        fluid.step(1.0 / 240.0, 9.81, |_, _| 0.0);
    }
    fluid
}

/// A shin wading back and forth across the pool at 1 m/s, at substep `s`.
fn wading_shin(fluid: &SphFluid, s: u64, solids: &mut SphSolids) {
    let spacing = fluid.params().smoothing_radius * 0.5;
    let side = ((fluid.capacity() / 4) as f64).sqrt().ceil() * spacing;
    // A triangle wave from 0.1 m before the pool to 0.1 m past it.
    let span = side + 0.2;
    let travel = (s as f64 / 240.0) % (2.0 * span);
    let (x, vx) = if travel < span {
        (travel - 0.1, 1.0)
    } else {
        (2.0 * span - travel - 0.1, -1.0)
    };
    solids.clear();
    solids.push_capsule(
        [x, -0.05, 0.5 * side],
        [x, 0.4, 0.5 * side],
        0.06,
        [vx, 0.0, 0.0],
        [vx, 0.0, 0.0],
    );
}

/// Solids, in the same run as `step`: off and (a) out of reach on the packed cube, then a
/// resting pool off and (b) with a shin wading through it. Each scene twice, the rounds
/// interleaved, so heat drift shows as a gap between a scene's two rounds rather than as
/// a difference between scenes. Then the phase timers, scene after scene a step at a
/// time, with the fraction of particles that ran the contact test.
fn solids(c: &mut Criterion) {
    let mut group = c.benchmark_group("sph/solids");
    let far = far_solids();
    for &n in &[1_024usize, 4_096] {
        group.throughput(Throughput::Elements(n as u64));
        let cube = filled(n);
        let resting = pool(n);
        for round in 0..2 {
            let mut off = cube.clone();
            group.bench_function(BenchmarkId::new(format!("cube_off_r{round}"), n), |b| {
                b.iter(|| off.step(std::hint::black_box(1.0 / 240.0), 0.0, |_, _| 0.0));
            });
            let mut out_of_reach = cube.clone();
            group.bench_function(BenchmarkId::new(format!("cube_far_r{round}"), n), |b| {
                b.iter(|| {
                    out_of_reach.step_with_solids(
                        std::hint::black_box(1.0 / 240.0),
                        0.0,
                        |_, _| 0.0,
                        &far,
                    )
                });
            });
            let mut still = resting.clone();
            group.bench_function(BenchmarkId::new(format!("pool_off_r{round}"), n), |b| {
                b.iter(|| still.step(std::hint::black_box(1.0 / 240.0), 9.81, |_, _| 0.0));
            });
            let mut waded = resting.clone();
            let mut shin = SphSolids::with_capacity(1, 0);
            let mut s = 0u64;
            group.bench_function(BenchmarkId::new(format!("pool_wade_r{round}"), n), |b| {
                b.iter(|| {
                    wading_shin(&waded, s, &mut shin);
                    s += 1;
                    waded.step_with_solids(
                        std::hint::black_box(1.0 / 240.0),
                        9.81,
                        |_, _| 0.0,
                        &shin,
                    );
                });
            });
        }
    }
    group.finish();

    // Phase medians, the four scenes stepped in turn so they share the machine's state.
    const STEPS: usize = 400;
    for &n in &[1_024usize, 4_096] {
        let mut scenes = [filled(n), filled(n), pool(n), pool(n)];
        let names = ["cube_off", "cube_far", "pool_off", "pool_wade"];
        let mut phases = vec![[[0.0f64; 6]; STEPS]; 4];
        let mut tested = [0.0f64; 4];
        let mut contacts = [0.0f64; 4];
        let mut shin = SphSolids::with_capacity(1, 0);
        for s in 0..STEPS {
            for (k, fluid) in scenes.iter_mut().enumerate() {
                let dt = 1.0 / 240.0;
                match k {
                    0 => fluid.step(dt, 0.0, |_, _| 0.0),
                    1 => fluid.step_with_solids(dt, 0.0, |_, _| 0.0, &far),
                    2 => fluid.step(dt, 9.81, |_, _| 0.0),
                    _ => {
                        wading_shin(fluid, s as u64, &mut shin);
                        fluid.step_with_solids(dt, 9.81, |_, _| 0.0, &shin);
                    }
                }
                let t = fluid.phase_times();
                let st = fluid.solid_stats();
                let us = |d: Duration| d.as_secs_f64() * 1e6;
                phases[k][s] = [
                    us(t.grid),
                    us(st.binning),
                    us(t.density),
                    us(t.forces),
                    us(t.integrate),
                    us(t.total()),
                ];
                tested[k] += st.ray_tested as f64 / fluid.len() as f64 / STEPS as f64;
                contacts[k] += st.contacts as f64 / STEPS as f64;
            }
        }
        for k in 0..4 {
            let mut med = [0.0f64; 6];
            for (c, m) in med.iter_mut().enumerate() {
                let mut col: Vec<f64> = phases[k].iter().map(|p| p[c]).collect();
                col.sort_by(|a, b| a.partial_cmp(b).unwrap());
                *m = col[STEPS / 2];
            }
            println!(
                "sph/solids phases n={n:>5} {:<9} us: grid {:7.1} (binning {:5.1}) density {:7.1} \
                 forces {:7.1} integrate {:7.1} total {:7.1}; ray-tested {:5.2}% of particles, \
                 {:6.1} contacts a step",
                names[k],
                med[0],
                med[1],
                med[2],
                med[3],
                med[4],
                med[5],
                tested[k] * 100.0,
                contacts[k],
            );
        }
    }
}

criterion_group!(benches, step, solids);
criterion_main!(benches);
