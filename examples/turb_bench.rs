//! In-process, interleaved measurements for the turbulence package (TURB).
//!
//! The machine drifts with heat over minutes, so every on/off pair here is alternated
//! round by round inside one process and each figure is a median of rounds.
//!
//! ```text
//! cargo run --release --example turb_bench --features "particles fluid_simulation" -- [rounds]
//! ```
//!
//! - `particles`: ns a particle for `integrate` (field off) and `integrate_in_air` (both
//!   classes on a 32^3 swirl), on the live sparks-and-dust pool of
//!   `benches/particle_effects.rs`, at 10k, 100k and 1M; the fetch alone on the same
//!   positions; and `integrate_in_air` reading a `PlumeField` whose worker steps at
//!   20 Hz beside it.
//! - `grid`: ms a `FluidGrid3D` step at 32^3 and 64^3 with each turbulence option, on
//!   the forced source of `benches/fluid_grid.rs`.

use std::time::{Duration, Instant};

use rs_physics::fluid_dynamics::{
    AdvectionScheme, FluidGrid3D, PlumeField, PlumeRegion, PlumeSource, SolverConfig,
    VorticityConfinement,
};
use rs_physics::particles::{
    Burst, EffectRng, ParticleClass, ParticleEffects, SwirlField, TurbulenceDrive, VelocityGrid,
};

const DT: f32 = 1.0 / 60.0;

/// The live pool of `benches/particle_effects.rs`: sparks and dust, half each.
struct Live {
    fx: ParticleEffects,
    rng: EffectRng,
    owed: [f32; 2],
    rate: [f32; 2],
}

impl Live {
    /// The pool, on the air (both classes, `swirl` 1) when `air` is given, the field
    /// updated every `period` frames and the samples refreshed over that period.
    fn new(count: usize, air: Option<&mut SwirlField>, period: usize) -> Live {
        let mut fx = ParticleEffects::with_capacity(count + count / 4);
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
        if air.is_some() {
            fx.set_swirl(0, 1.0);
            fx.set_swirl(1, 1.0);
        }
        let half = count as f32 / 2.0;
        let rate = [half / 0.35, half / 30.0];
        let mut live = Live {
            fx,
            rng: EffectRng::new(0xC0FFEE),
            owed: [0.0; 2],
            rate,
        };
        let mut air = air;
        for step in 0..(45.0 / DT) as usize {
            live.feed();
            match air.as_deref_mut() {
                Some(field) => {
                    if step % period == 0 {
                        field.advance(period as f32 * DT);
                    }
                    live.fx
                        .integrate_in_air(DT, field.velocity(), period as f32 * DT);
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
                    lifetime: if c == 0 { 0.2..0.5 } else { 20.0..40.0 },
                    size: 0.7..1.3,
                    lift: 0.35,
                },
                &mut self.rng,
            );
        }
    }
}

fn swirl_field() -> SwirlField {
    let drive = TurbulenceDrive::new(3.0, 12.0).unwrap();
    SwirlField::new([-32.0, 8.0, -32.0], [32, 32, 32], 2.0, drive, 0x5EED).unwrap()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Steps per round, so a round of the small pools is long enough to time.
fn steps_for(n: usize) -> usize {
    (2_000_000 / n).clamp(4, 400)
}

fn particles(rounds: usize) {
    println!(
        "particles: ns a particle, median of {rounds} interleaved rounds; the field updates \
         every 6 frames (10 Hz) or 3 (20 Hz), outside the timed span"
    );
    for &n in &[10_000usize, 100_000, 1_000_000] {
        let mut off = Live::new(n, None, 1);
        let periods = [6usize, 3];
        let mut fields: Vec<SwirlField> = periods.iter().map(|_| swirl_field()).collect();
        let mut ons: Vec<Live> = periods
            .iter()
            .zip(fields.iter_mut())
            .map(|(&p, f)| Live::new(n, Some(f), p))
            .collect();
        let steps = steps_for(n);
        let mut t_off = Vec::new();
        let mut t_on = vec![Vec::new(); periods.len()];
        for _ in 0..rounds {
            let mut spent = Duration::ZERO;
            let mut count = 0usize;
            for _ in 0..steps {
                off.feed();
                count += off.fx.len();
                let started = Instant::now();
                off.fx.integrate(std::hint::black_box(DT));
                spent += started.elapsed();
            }
            t_off.push(spent.as_nanos() as f64 / count as f64);

            for (k, &period) in periods.iter().enumerate() {
                let (mut spent, mut count) = (Duration::ZERO, 0usize);
                for step in 0..steps {
                    ons[k].feed();
                    if step % period == 0 {
                        fields[k].advance(period as f32 * DT);
                    }
                    count += ons[k].fx.len();
                    let started = Instant::now();
                    ons[k].fx.integrate_in_air(
                        std::hint::black_box(DT),
                        fields[k].velocity(),
                        period as f32 * DT,
                    );
                    spent += started.elapsed();
                }
                t_on[k].push(spent.as_nanos() as f64 / count as f64);
            }
        }
        let a = median(t_off);
        let (b10, b20) = (median(t_on[0].clone()), median(t_on[1].clone()));
        println!(
            "  {n:>9}: off {a:6.2}  on 10 Hz {b10:6.2} ({:4.2}x)  on 20 Hz {b20:6.2} ({:4.2}x)",
            b10 / a,
            b20 / a
        );
    }
}

/// The 1b gate: `integrate_in_air` reading a plume whose worker steps beside it.
fn particles_with_worker(rounds: usize) {
    println!(
        "particles beside a stepping plume worker (32^3 of 2 m, at 10 and 20 Hz): ns a particle"
    );
    let region = PlumeRegion {
        origin: [-32.0, 8.0, -32.0],
        cells: [32, 32, 32],
        cell_size: 2.0,
    };
    let source = PlumeSource {
        position: [0.0, 9.0, 0.0],
        drive: TurbulenceDrive::new(3.0, 12.0).unwrap(),
        smoke_rate: 1.0,
    };
    for (&n, rate) in [100_000usize, 1_000_000]
        .iter()
        .flat_map(|n| [(n, 10.0f32), (n, 20.0)])
    {
        let mut off = Live::new(n, None, 1);
        let mut warm = swirl_field();
        let mut on = Live::new(n, Some(&mut warm), 3);
        let (plume, mut air) = PlumeField::new(region, source, [2.0, 0.0, 0.5], 0x5EED).unwrap();
        let worker = plume.spawn(rate).unwrap();
        let steps = steps_for(n);
        let (mut t_off, mut t_on) = (Vec::new(), Vec::new());
        let mut costs = Vec::new();
        for _ in 0..rounds {
            let (mut spent, mut count) = (Duration::ZERO, 0usize);
            for _ in 0..steps {
                off.feed();
                count += off.fx.len();
                let started = Instant::now();
                off.fx.integrate(std::hint::black_box(DT));
                spent += started.elapsed();
            }
            t_off.push(spent.as_nanos() as f64 / count as f64);

            let (mut spent, mut count) = (Duration::ZERO, 0usize);
            for _ in 0..steps {
                on.feed();
                count += on.fx.len();
                let started = Instant::now();
                // The frame thread's whole cost: take the newest frame, then integrate.
                let frame = air.latest();
                on.fx.integrate_in_air(
                    std::hint::black_box(DT),
                    frame.velocity(),
                    frame.interval(),
                );
                spent += started.elapsed();
            }
            t_on.push(spent.as_nanos() as f64 / count as f64);
            costs.push(air.latest().step_cost().as_secs_f64() * 1e3);
        }
        let last_step = air.latest().step();
        drop(worker);
        let (a, b) = (median(t_off), median(t_on));
        println!(
            "  {n:>9} at {rate} Hz: off {a:6.2}  on {b:6.2}  ratio {:5.2}   worker reached step {last_step}, step cost median {:.2} ms",
            b / a,
            median(costs)
        );
    }
}

fn source_3d(grid: &mut FluidGrid3D, n: usize) {
    for i in n / 4..3 * n / 4 {
        grid.add_density(i, n / 2, n / 2, 1.0).unwrap();
        grid.add_velocity(i, n / 2, n / 2, 0.0, 0.05, 0.0).unwrap();
    }
}

fn grid(rounds: usize) {
    println!(
        "FluidGrid3D step: ms, median of {rounds} interleaved rounds (and pressure iterations)"
    );
    let configs = [
        ("off", SolverConfig::default()),
        (
            "maccormack",
            SolverConfig::default().with_advection(AdvectionScheme::MacCormack),
        ),
        (
            "confinement",
            SolverConfig::default()
                .with_vorticity_confinement(VorticityConfinement::MatchNumericalDissipation),
        ),
        (
            "both",
            SolverConfig::default()
                .with_advection(AdvectionScheme::MacCormack)
                .with_vorticity_confinement(VorticityConfinement::MatchNumericalDissipation),
        ),
    ];
    for &n in &[32usize, 64] {
        let mut grids: Vec<FluidGrid3D> = configs
            .iter()
            .map(|(_, c)| {
                let mut g = FluidGrid3D::with_solver(n, n, n, 1e-5, 1e-5, 1.0 / 60.0, *c).unwrap();
                for _ in 0..10 {
                    source_3d(&mut g, n);
                    g.step();
                }
                g
            })
            .collect();
        let steps = if n == 32 { 10 } else { 2 };
        let mut times = vec![Vec::new(); configs.len()];
        let mut iterations = vec![0usize; configs.len()];
        for _ in 0..rounds {
            for (c, g) in grids.iter_mut().enumerate() {
                let started = Instant::now();
                for _ in 0..steps {
                    source_3d(g, n);
                    g.step();
                    iterations[c] += g.get_last_pressure_iterations();
                }
                times[c].push(started.elapsed().as_secs_f64() * 1e3 / steps as f64);
            }
        }
        let off = median(times[0].clone());
        print!("  {n}^3:");
        for (c, (name, _)) in configs.iter().enumerate() {
            let t = median(times[c].clone());
            print!(
                "  {name} {t:.2} ({:+.0}%, {:.1} it)",
                100.0 * (t / off - 1.0),
                iterations[c] as f64 / (rounds * steps) as f64
            );
        }
        println!();
    }
}

/// A `PlumeField` step on this thread (sources, grid step with both options, swirl,
/// writing the frame) at 32^3 and 64^3 regions of 2 m cells.
fn plume(rounds: usize) {
    println!(
        "PlumeField::step_now: ms a step, median of {rounds}, and the fraction of a 20 Hz period"
    );
    for &n in &[32usize, 64] {
        let region = PlumeRegion {
            origin: [-(n as f32), 0.0, -(n as f32)],
            cells: [n, n, n],
            cell_size: 2.0,
        };
        let source = PlumeSource {
            position: [0.0, 1.0, 0.0],
            drive: TurbulenceDrive::new(3.0, 12.0).unwrap(),
            smoke_rate: 1.0,
        };
        let (mut plume, mut air) = PlumeField::new(region, source, [2.0, 0.0, 0.5], 1).unwrap();
        for _ in 0..5 {
            plume.step_now(0.05);
        }
        let mut costs = Vec::new();
        for _ in 0..rounds {
            plume.step_now(0.05);
            costs.push(air.latest().step_cost().as_secs_f64() * 1e3);
        }
        let m = median(costs);
        println!(
            "  {n}^3: {m:.2} ms, {:.0}% of 50 ms; {} bytes",
            100.0 * m / 50.0,
            plume.bytes()
        );
    }
}

/// What the diffusion sweeps cost a plume that has no viscosity: the plume's grid
/// (both options, inviscid) at 4 sweeps and at 1, interleaved. The sweeps run even at a
/// coefficient of zero, so the difference over three is one sweep of every field.
fn sweeps(rounds: usize) {
    println!("inviscid FluidGrid3D, both options: ms a step at 4 diffusion sweeps and at 1");
    for &n in &[34usize, 66] {
        let config = SolverConfig::default()
            .with_advection(AdvectionScheme::MacCormack)
            .with_vorticity_confinement(VorticityConfinement::MatchNumericalDissipation);
        let mut grids: Vec<FluidGrid3D> = [4usize, 1]
            .iter()
            .map(|&it| {
                let mut c = config;
                c.iterations = it;
                let mut g = FluidGrid3D::with_solver(n, n, n, 0.0, 0.0, 0.05, c).unwrap();
                for _ in 0..5 {
                    source_3d(&mut g, n);
                    g.step();
                }
                g
            })
            .collect();
        let mut times = vec![Vec::new(); 2];
        for _ in 0..rounds {
            for (c, g) in grids.iter_mut().enumerate() {
                let started = Instant::now();
                source_3d(g, n);
                g.step();
                times[c].push(started.elapsed().as_secs_f64() * 1e3);
            }
        }
        let (four, one) = (median(times[0].clone()), median(times[1].clone()));
        println!(
            "  {n}^3: 4 sweeps {four:.2} ms, 1 sweep {one:.2} ms; the sweeps at a zero coefficient are {:.0}% of the step",
            100.0 * (four - one) * 4.0 / 3.0 / four
        );
    }
}

/// Where the field-on time goes, at 100k: the update loop with only new particles
/// sampled (a refresh period far longer than the run), and the fetch alone.
fn probe(rounds: usize) {
    let n = 100_000;
    let mut field = swirl_field();
    let mut on = Live::new(n, Some(&mut field), 6);
    let mut off = Live::new(n, None, 1);
    let steps = steps_for(n);
    let (mut t_off, mut t_fresh, mut t_fetch) = (Vec::new(), Vec::new(), Vec::new());
    let mut sink = 0.0f32;
    for _ in 0..rounds {
        let (mut spent, mut count) = (Duration::ZERO, 0usize);
        for _ in 0..steps {
            off.feed();
            count += off.fx.len();
            let started = Instant::now();
            off.fx.integrate(std::hint::black_box(DT));
            spent += started.elapsed();
        }
        t_off.push(spent.as_nanos() as f64 / count as f64);
        let (mut spent, mut count) = (Duration::ZERO, 0usize);
        for _ in 0..steps {
            on.feed();
            count += on.fx.len();
            let started = Instant::now();
            on.fx
                .integrate_in_air(std::hint::black_box(DT), field.velocity(), 1e9);
            spent += started.elapsed();
        }
        t_fresh.push(spent.as_nanos() as f64 / count as f64);
        let (xs, ys, zs) = on.fx.positions_soa();
        let grid: &VelocityGrid = field.velocity();
        let started = Instant::now();
        for _ in 0..steps {
            for i in 0..xs.len() {
                let u = grid.sample([xs[i], ys[i], zs[i]]);
                sink += u[0] + u[1] + u[2];
            }
        }
        t_fetch.push(started.elapsed().as_nanos() as f64 / (steps * xs.len()) as f64);
    }
    println!(
        "probe 100k: off {:.2}  on with new particles only {:.2}  fetch alone {:.2} (sink {})",
        median(t_off),
        median(t_fresh),
        median(t_fetch),
        sink.is_finite()
    );
}

fn main() {
    let rounds: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(9);
    let which = std::env::args().nth(2).unwrap_or_else(|| "all".into());
    if which == "all" || which == "particles" {
        particles(rounds);
    }
    if which == "all" || which == "worker" {
        particles_with_worker(rounds);
    }
    if which == "all" || which == "grid" {
        grid(rounds);
    }
    if which == "all" || which == "plume" {
        plume(rounds);
    }
    if which == "probe" {
        probe(rounds);
    }
    if which == "sweeps" {
        sweeps(rounds);
    }
}
