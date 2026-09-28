//! In-process measurement for the particle drag flush and the parallel SPH step.
//!
//! Criterion was the wrong tool for both numbers this answers. The machine drifts about
//! 2.5x over an hour of heat, so two figures taken minutes apart are not comparable; and the
//! old `particle_effects` bench integrated one never-ending population, which after tens of
//! thousands of steps measured subnormal arithmetic rather than particles. Here every
//! variant is alternated inside one process, round by round, and each figure is a median.
//!
//! ```text
//! cargo build --release --example r2_bench --features "particles fluid_simulation"
//! r2_bench [rounds]
//! ```
//!
//! Particles: ns a particle for one `integrate`, in three populations:
//! - `live`: sparks (0.2 to 0.5 s) and dust (20 to 40 s) retired and re-emitted at a steady
//!   population, which is what a game pool is.
//! - `aged`: the old bench's population (nothing retires), after 30,000 steps of drag. This
//!   is where drag used to leave velocities in the subnormal range.
//! - `fresh`: the same never-retiring population on its first steps, the control.
//!
//! SPH: microseconds a `step` for a packed cube of blood at 1/240 s, at 1, 4 and 8 threads.

use std::time::Instant;

use rs_physics::fluid_dynamics::{SphFluid, SphParams};
use rs_physics::particles::{Burst, EffectRng, ParticleClass, ParticleEffects};

const DT: f32 = 1.0 / 60.0;

fn classes(fx: &mut ParticleEffects) {
    fx.set_class(0, ParticleClass { gravity: 26.0, drag: 1.4, restitution: 0.32 });
    fx.set_class(1, ParticleClass { gravity: 1.6, drag: 3.4, restitution: 0.0 });
}

fn burst(class: u8, count: u32, lifetime: core::ops::Range<f32>) -> Burst {
    Burst {
        origin: [0.0, 40.0, 0.0],
        class,
        count,
        speed: 6.0..15.0,
        lifetime,
        size: 0.7..1.3,
        lift: 0.35,
    }
}

/// A pool that never retires, as the old criterion bench built it.
fn immortal(n: usize) -> ParticleEffects {
    let mut fx = ParticleEffects::with_capacity(n);
    classes(&mut fx);
    let mut rng = EffectRng::new(0xC0FFEE);
    for class in [0u8, 1] {
        fx.emit(&burst(class, (n / 2) as u32, 10_000.0..10_001.0), &mut rng);
    }
    fx
}

/// Half sparks, half dust, re-emitted each step at the rate that holds the population.
struct Live {
    fx: ParticleEffects,
    rng: EffectRng,
    owed: [f32; 2],
    rate: [f32; 2],
}

const SPARK_LIFE: core::ops::Range<f32> = 0.2..0.5;
const DUST_LIFE: core::ops::Range<f32> = 20.0..40.0;

impl Live {
    fn new(n: usize) -> Live {
        let mut fx = ParticleEffects::with_capacity(n + n / 4);
        classes(&mut fx);
        let half = n as f32 / 2.0;
        // Steady state: population = rate * mean lifetime.
        let rate = [half / 0.35, half / 30.0];
        let mut live = Live { fx, rng: EffectRng::new(0xFACE), owed: [0.0; 2], rate };
        // Run long enough for the dust to reach its steady age distribution.
        for _ in 0..(45.0 / DT) as usize {
            live.feed();
            live.fx.integrate(DT);
        }
        live
    }

    fn feed(&mut self) {
        for c in 0..2 {
            self.owed[c] += self.rate[c] * DT;
            let whole = self.owed[c].floor();
            self.owed[c] -= whole;
            let life = if c == 0 { SPARK_LIFE } else { DUST_LIFE };
            self.fx.emit(&burst(c as u8, whole as u32, life), &mut self.rng);
        }
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = v.len() / 2;
    if v.len() % 2 == 0 { (v[m - 1] + v[m]) / 2.0 } else { v[m] }
}

/// ns a particle for one integrate, median over `reps` calls.
fn time_integrate(fx: &mut ParticleEffects, reps: usize, mut between: impl FnMut(&mut ParticleEffects)) -> f64 {
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        between(fx);
        let n = fx.len().max(1);
        let t = Instant::now();
        fx.integrate(std::hint::black_box(DT));
        samples.push(t.elapsed().as_nanos() as f64 / n as f64);
    }
    median(&mut samples)
}

fn packed_cube(n: usize) -> SphFluid {
    let params = SphParams::blood();
    let spacing = params.smoothing_radius * 0.5;
    let mut fluid = SphFluid::new(params, n).unwrap();
    let side = (n as f64).cbrt().ceil() as usize;
    'outer: for x in 0..side {
        for y in 0..side {
            for z in 0..side {
                let p = [x as f64 * spacing, 1.0 + y as f64 * spacing, z as f64 * spacing];
                if !fluid.spawn(p, [0.0; 3]) {
                    break 'outer;
                }
            }
        }
    }
    fluid
}

/// Median microseconds a step over `steps` steps from a fresh copy of `scene`.
fn time_sph(scene: &SphFluid, steps: usize) -> f64 {
    let mut fluid = scene.clone();
    fluid.step(1.0 / 240.0, 9.81, |_, _| 0.0);
    let mut samples = Vec::with_capacity(steps);
    for _ in 0..steps {
        let t = Instant::now();
        fluid.step(std::hint::black_box(1.0 / 240.0), 9.81, |_, _| 0.0);
        samples.push(t.elapsed().as_nanos() as f64 / 1e3);
    }
    median(&mut samples)
}

fn main() {
    let rounds: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(4);
    let pools: Vec<(usize, rayon::ThreadPool)> = [1usize, 4, 8]
        .iter()
        .map(|&t| (t, rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap()))
        .collect();
    let sph_sizes = [1_024usize, 4_096, 8_192, 16_384];
    let scenes: Vec<SphFluid> = sph_sizes.iter().map(|&n| packed_cube(n)).collect();

    let fx_sizes = [16_384usize, 100_000];
    // Built once and aged once: 30,000 steps is 500 s of drag, far past the subnormal range.
    let mut aged: Vec<ParticleEffects> = fx_sizes.iter().map(|&n| immortal(n)).collect();
    for fx in &mut aged {
        for _ in 0..30_000 {
            fx.integrate(DT);
        }
    }
    let mut live: Vec<Live> = fx_sizes.iter().map(|&n| Live::new(n)).collect();

    let mut table: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
    for round in 0..rounds {
        eprintln!("round {round}");
        for (k, &n) in fx_sizes.iter().enumerate() {
            let mut fresh = immortal(n);
            let f = time_integrate(&mut fresh, 20, |_| {});
            let a = time_integrate(&mut aged[k], 200, |_| {});
            let l = {
                let lv = &mut live[k];
                let mut samples = Vec::with_capacity(200);
                for _ in 0..200 {
                    lv.feed();
                    let len = lv.fx.len().max(1);
                    let t = Instant::now();
                    lv.fx.integrate(std::hint::black_box(DT));
                    samples.push(t.elapsed().as_nanos() as f64 / len as f64);
                }
                median(&mut samples)
            };
            table.entry(format!("particles fresh n={n:>6} ns/particle")).or_default().push(f);
            table.entry(format!("particles aged  n={n:>6} ns/particle")).or_default().push(a);
            table.entry(format!("particles live  n={n:>6} ns/particle")).or_default().push(l);
        }
        for (s, &n) in scenes.iter().zip(&sph_sizes) {
            for (t, pool) in &pools {
                let us = pool.install(|| time_sph(s, 40));
                table.entry(format!("sph n={n:>5} threads={t} us/step")).or_default().push(us);
            }
        }
    }
    for (name, mut v) in table {
        let rounds: Vec<String> = v.iter().map(|x| format!("{x:.2}")).collect();
        println!("{name:<40} median {:>10.2}   rounds [{}]", median(&mut v), rounds.join(", "));
    }
}
