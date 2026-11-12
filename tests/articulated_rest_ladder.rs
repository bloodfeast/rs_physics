//! The rest ladder: one set of scenes, from a single body up to a heap of rigs, each run
//! for the same length and read with the same instruments, so that what keeps a jointed
//! body from coming to rest can be ranked on one scale.
//!
//! **This is a diagnostic and not a law.** Every test here is `#[ignore]`d, prints a table
//! and asserts nothing: a scene that does not rest is the finding, not a failure. The
//! laws about rest live in `articulated_laws.rs`, and the fixtures below are copies of the
//! ones there (an integration test cannot import another), kept identical on purpose so
//! the two files talk about the same rig.
//!
//! Each scene is run twice. With sleeping on, what is read is when the solver itself put
//! the bodies to sleep. With sleeping off, the settling test is replayed here from the
//! public state -- translation plus swept turn since the window began, against two per
//! cent of the body's reach -- because the residual is positional and a kinetic threshold
//! would measure the wrong quantity.
//!
//! Run one scene with
//! `cargo test --release --test articulated_rest_ladder -- --ignored --nocapture <name>`.
//! Set `REST_LADDER_CSV` to a directory to get the per-step record of every run.

use rs_physics::articulated::{Body, Field, Joint, Skeleton};
use rs_physics::models::Quaternion;
use std::fmt::Write as _;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const DT: f64 = 1.0 / 60.0;
const GRAVITY: f64 = 9.80665;
const G: V = (0.0, -GRAVITY, 0.0);
const STEPS: usize = 2000;
/// The settling test's own two numbers, restated: see `sleep.rs`.
const STILL_FRACTION: f64 = 0.02;
const BONES: [&str; 17] = [
    "pelvis", "spine1", "spine2", "spine3", "spine4", "armA1", "armA2", "armA3", "armB1", "armB2",
    "armB3", "legA1", "legA2", "legA3", "legB1", "legB2", "legB3",
];

type V = (f64, f64, f64);

fn add(a: V, b: V) -> V {
    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
}
fn sub(a: V, b: V) -> V {
    (a.0 - b.0, a.1 - b.1, a.2 - b.2)
}
fn scale(a: V, k: f64) -> V {
    (a.0 * k, a.1 * k, a.2 * k)
}
fn dot(a: V, b: V) -> f64 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}
fn cross(a: V, b: V) -> V {
    (a.1 * b.2 - a.2 * b.1, a.2 * b.0 - a.0 * b.2, a.0 * b.1 - a.1 * b.0)
}
fn length(a: V) -> f64 {
    dot(a, a).sqrt()
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

/// What the bodies are standing on.
#[derive(Clone, Copy)]
enum Floor<'a> {
    Plane(V, f64),
    Field(Field<'a>),
}

impl Floor<'_> {
    const FLAT: Floor<'static> = Floor::Plane((0.0, 1.0, 0.0), 0.0);

    /// How far above the floor the point `p` is, along the floor's own normal there.
    fn clearance(&self, p: V) -> f64 {
        match self {
            Floor::Plane(normal, distance) => dot(*normal, p) - distance,
            Floor::Field(field) => {
                let (normal, distance) = field.plane_at(p.0, p.2);
                dot(normal, p) - distance
            }
        }
    }
}

/// Every dial the public interface has on rest, written out so that a run states all of
/// them rather than inheriting some.
#[derive(Clone, Copy, Debug)]
struct Dials {
    passes: usize,
    friction: f64,
    rolling: f64,
    damping: f64,
    self_collision: bool,
    sleeping: bool,
    /// `Skeleton::set_contact_patch`: a crossed pair held by the two ends of its patch.
    patch: bool,
    /// `Skeleton::set_stick_memory`: pairs remember where they stuck, bodies where their rolling stuck.
    stick: bool,
}

impl Dials {
    /// The crate's defaults at eight passes.
    const PLAIN: Dials = Dials {
        passes: 8,
        friction: 0.5,
        rolling: 0.25,
        damping: 0.0,
        self_collision: true,
        sleeping: true,
        patch: true,
        stick: false,
    };

    fn awake(self) -> Dials {
        Dials { sleeping: false, ..self }
    }

    fn apply(self, s: &mut Skeleton) {
        s.set_friction(self.friction);
        s.set_rolling_resistance(self.rolling);
        s.set_joint_damping(self.damping);
        s.set_self_collision(self.self_collision);
        s.set_sleeping(self.sleeping);
        s.set_contact_patch(self.patch);
        s.set_stick_memory(self.stick);
    }
}

/// One step's reading of the whole world.
#[derive(Clone, Copy, Default)]
struct Sample {
    energy: f64,
    kinetic: f64,
    point_speed: f64,
    ratio: f64,
    unready: usize,
    awake: usize,
    contacts: usize,
    load: f64,
    penetration: f64,
    joint_gap: f64,
    hinge_error: f64,
}

/// What one unit -- a lone body or one rig -- did over a run.
#[derive(Clone, Debug, Default)]
struct Unit {
    /// The step at which the solver had every body of the unit asleep, with sleeping on.
    slept_at: Option<usize>,
    /// The first step at which the replayed settling test had every body ready.
    ready_at: Option<usize>,
    /// The share of the tail's steps on which every body was ready.
    tail_ready: f64,
    /// Bodies not ready at the last step.
    unready_end: usize,
    /// Centre of mass after the first step, at the start of the tail, and at the end.
    com: [V; 3],
    /// How many times each body's settling window was reset during the tail.
    resets: Vec<u32>,
}

/// What a whole run did.
#[derive(Clone, Debug, Default)]
struct Outcome {
    units: Vec<Unit>,
    all_asleep_at: Option<usize>,
    /// Median kinetic energy over the second, third and last quarter of the run, in joules.
    kinetic: [f64; 3],
    /// The period the tail's kinetic energy repeats with, in steps, and how well.
    period: Option<(usize, f64)>,
    unready_end: usize,
    tail_point_speed: f64,
    tail_ratio: f64,
    tail_contacts: f64,
    tail_load: f64,
    tail_penetration: f64,
    tail_joint_gap: f64,
    tail_hinge_error: f64,
    /// Window resets over the tail by what alone was over the limit: the turn, the
    /// translation, both, or only the two together.
    causes: [u32; 4],
}

/// The settling test of `Skeleton::settle`, replayed from outside, and everything else a
/// step is read for.
struct Recorder {
    units: Vec<Range<usize>>,
    from: Vec<V>,
    turn: Vec<Quaternion>,
    still: Vec<u32>,
    ratios: Vec<f64>,
    samples: Vec<Sample>,
    out: Vec<Unit>,
    ready_steps: Vec<usize>,
    all_asleep_at: Option<usize>,
    causes: [u32; 4],
    tail_from: usize,
    weight: f64,
    csv: Option<String>,
}

fn window(reach: f64) -> u32 {
    let seconds = (2.0 * reach.max(1e-6) / GRAVITY).sqrt();
    ((seconds / DT).ceil() as u32).clamp(1, 600)
}

fn centre_of_mass(s: &Skeleton, bodies: Range<usize>) -> V {
    let mut total = (0.0, 0.0, 0.0);
    let mut mass = 0.0;
    for i in bodies {
        let body = s.body(i);
        if body.inv_mass > 0.0 {
            total = add(total, scale(body.position, 1.0 / body.inv_mass));
            mass += 1.0 / body.inv_mass;
        }
    }
    scale(total, 1.0 / mass.max(1e-12))
}

impl Recorder {
    fn new(s: &Skeleton, units: Vec<Range<usize>>, steps: usize) -> Recorder {
        let n = s.len();
        let mut weight = 0.0;
        for i in 0..n {
            let body = s.body(i);
            if body.inv_mass > 0.0 {
                weight += GRAVITY / body.inv_mass;
            }
        }
        let out = units
            .iter()
            .map(|u| Unit { resets: vec![0; u.len()], ..Unit::default() })
            .collect();
        Recorder {
            ready_steps: vec![0; units.len()],
            units,
            from: (0..n).map(|i| s.position(i)).collect(),
            turn: (0..n).map(|i| s.orientation(i)).collect(),
            still: vec![0; n],
            ratios: vec![0.0; n],
            samples: Vec::with_capacity(steps),
            out,
            all_asleep_at: None,
            causes: [0; 4],
            tail_from: steps - steps / 4,
            weight,
            csv: std::env::var("REST_LADDER_CSV").ok().map(|_| {
                String::from(
                    "step,energy,kinetic,point_speed,ratio,unready,awake,contacts,load,\
                     penetration,joint_gap,hinge_error,ratios\n",
                )
            }),
        }
    }

    /// Read the world after a step. `step` counts from one.
    fn observe(&mut self, s: &Skeleton, floor: Floor<'_>, step: usize) {
        let in_tail = step > self.tail_from;
        let mut sample = Sample { awake: s.awake_count(), contacts: s.contact_count(), ..Sample::default() };
        let mut load = 0.0;
        for i in 0..s.len() {
            let body = s.body(i);
            if body.inv_mass <= 0.0 {
                self.ratios[i] = 0.0;
                continue;
            }
            let mass = 1.0 / body.inv_mass;
            let w = body.angular_velocity;
            let mut kinetic = 0.5 * mass * dot(body.velocity, body.velocity);
            for (component, inverse) in [(w.0, body.inv_inertia.0), (w.1, body.inv_inertia.1), (w.2, body.inv_inertia.2)] {
                if inverse > 0.0 {
                    kinetic += 0.5 * component * component / inverse;
                }
            }
            sample.kinetic += kinetic;
            sample.energy += kinetic + mass * GRAVITY * body.position.1;
            load += s.normal_load(i);

            // The lowest point of the body's surface: where it would be resting, and so the
            // point whose speed says whether it is sliding or rolling there.
            let half = body.orientation.rotate_point((0.0, body.half_length, 0.0));
            let mut lowest = (f64::INFINITY, (0.0, 0.0, 0.0));
            for end in [1.0, -1.0] {
                let tip = add(body.position, scale(half, end));
                let gap = floor.clearance(tip) - body.radius;
                sample.penetration = sample.penetration.max(-gap);
                if gap < lowest.0 {
                    lowest = (gap, sub(scale(half, end), (0.0, body.radius, 0.0)));
                }
            }
            let at_point = add(body.velocity, cross(w, lowest.1));
            sample.point_speed = sample.point_speed.max(length(at_point));

            // The settling test, as `Skeleton::settle` runs it.
            let reach = (body.radius + body.half_length).max(1e-3);
            let limit = STILL_FRACTION * reach;
            let moved = length(sub(body.position, self.from[i]));
            let delta = body.orientation.multiply(&self.turn[i].conjugate());
            let sign = if delta.w < 0.0 { -2.0 } else { 2.0 };
            let turned = scale((delta.x, delta.y, delta.z), sign);
            let axis = body.orientation.rotate_point((0.0, 1.0, 0.0));
            let along = dot(turned, axis);
            let across = length(sub(turned, scale(axis, along)));
            let swept = across * reach + along.abs() * body.radius;
            self.ratios[i] = (moved + swept) / limit;
            sample.ratio = sample.ratio.max(self.ratios[i]);
            if moved + swept > limit {
                self.still[i] = 0;
                self.from[i] = body.position;
                self.turn[i] = body.orientation;
                if in_tail {
                    let cause = match (swept > limit, moved > limit) {
                        (true, false) => 0,
                        (false, true) => 1,
                        (true, true) => 2,
                        (false, false) => 3,
                    };
                    self.causes[cause] += 1;
                    for (u, unit) in self.units.iter().enumerate() {
                        if unit.contains(&i) {
                            self.out[u].resets[i - unit.start] += 1;
                        }
                    }
                }
            } else {
                self.still[i] += 1;
            }
            if self.still[i] < window(reach) {
                sample.unready += 1;
            }
        }
        sample.load = load / self.weight.max(1e-12);

        for joint in s.joints() {
            let (a, b, anchor_a, anchor_b, axes) = match *joint {
                Joint::Ball { a, b, anchor_a, anchor_b, .. } => (a, b, anchor_a, anchor_b, None),
                Joint::Hinge { a, b, anchor_a, anchor_b, axis_a, axis_b, .. } => {
                    (a, b, anchor_a, anchor_b, Some((axis_a, axis_b)))
                }
            };
            let (qa, qb) = (s.orientation(a), s.orientation(b));
            let pa = add(s.position(a), qa.rotate_point(anchor_a));
            let pb = add(s.position(b), qb.rotate_point(anchor_b));
            sample.joint_gap = sample.joint_gap.max(length(sub(pa, pb)));
            if let Some((axis_a, axis_b)) = axes {
                let cosine = dot(qa.rotate_point(axis_a), qb.rotate_point(axis_b));
                sample.hinge_error = sample.hinge_error.max(cosine.clamp(-1.0, 1.0).acos());
            }
        }

        for (u, unit) in self.units.iter().enumerate() {
            let out = &mut self.out[u];
            if out.slept_at.is_none() && unit.clone().all(|i| !s.is_awake(i)) {
                out.slept_at = Some(step);
            }
            let reach = |i: usize| {
                let body = s.body(i);
                (body.radius + body.half_length).max(1e-3)
            };
            let unready = unit.clone().filter(|&i| self.still[i] < window(reach(i))).count();
            if unready == 0 {
                out.ready_at.get_or_insert(step);
                if in_tail {
                    self.ready_steps[u] += 1;
                }
            }
            out.unready_end = unready;
            if step == 1 {
                out.com[0] = centre_of_mass(s, unit.clone());
            }
            if step == self.tail_from {
                out.com[1] = centre_of_mass(s, unit.clone());
            }
            out.com[2] = centre_of_mass(s, unit.clone());
        }
        if self.all_asleep_at.is_none() && sample.awake == 0 {
            self.all_asleep_at = Some(step);
        }
        if let Some(csv) = self.csv.as_mut() {
            let _ = write!(
                csv,
                "{step},{:.9e},{:.6e},{:.6e},{:.4},{},{},{},{:.5},{:.6e},{:.6e},{:.6e},",
                sample.energy,
                sample.kinetic,
                sample.point_speed,
                sample.ratio,
                sample.unready,
                sample.awake,
                sample.contacts,
                sample.load,
                sample.penetration,
                sample.joint_gap,
                sample.hinge_error,
            );
            for ratio in &self.ratios {
                let _ = write!(csv, "{ratio:.2} ");
            }
            csv.push('\n');
        }
        self.samples.push(sample);
    }

    fn finish(mut self, label: &str) -> Outcome {
        let steps = self.samples.len();
        let quarter = steps / 4;
        let tail = &self.samples[self.tail_from.min(steps)..];
        let kinetic_over = |range: Range<usize>| {
            median(&self.samples[range].iter().map(|s| s.kinetic).collect::<Vec<_>>())
        };
        for (u, unit) in self.out.iter_mut().enumerate() {
            unit.tail_ready = self.ready_steps[u] as f64 / tail.len().max(1) as f64;
        }
        if let (Some(csv), Ok(dir)) = (self.csv.take(), std::env::var("REST_LADDER_CSV")) {
            let name: String = label
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '.' { c } else { '_' })
                .collect();
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::fs::write(format!("{dir}/{name}.csv"), csv);
        }
        let worst = |f: fn(&Sample) -> f64| tail.iter().map(f).fold(0.0_f64, f64::max);
        Outcome {
            all_asleep_at: self.all_asleep_at,
            kinetic: [
                kinetic_over(quarter..2 * quarter),
                kinetic_over(2 * quarter..3 * quarter),
                kinetic_over(3 * quarter..steps),
            ],
            period: period(&tail.iter().map(|s| s.kinetic).collect::<Vec<_>>()),
            unready_end: self.samples.last().map_or(0, |s| s.unready),
            tail_point_speed: worst(|s| s.point_speed),
            tail_ratio: worst(|s| s.ratio),
            tail_contacts: median(&tail.iter().map(|s| s.contacts as f64).collect::<Vec<_>>()),
            tail_load: median(&tail.iter().map(|s| s.load).collect::<Vec<_>>()),
            tail_penetration: worst(|s| s.penetration),
            tail_joint_gap: worst(|s| s.joint_gap),
            tail_hinge_error: worst(|s| s.hinge_error),
            causes: self.causes,
            units: self.out,
        }
    }
}

/// The lag at which a series repeats, if it does: the first peak of the autocorrelation of
/// its step-to-step differences that is above 0.8. Differences, so that a slow decay is
/// not read as a period.
fn period(series: &[f64]) -> Option<(usize, f64)> {
    if series.len() < 64 {
        return None;
    }
    let diffs: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();
    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let x: Vec<f64> = diffs.iter().map(|d| d - mean).collect();
    if x.iter().map(|v| v * v).sum::<f64>() <= 1e-30 {
        return None;
    }
    let correlation = |lag: usize| {
        let (head, shifted) = (&x[..x.len() - lag], &x[lag..]);
        let both: f64 = head.iter().zip(shifted).map(|(a, b)| a * b).sum();
        let norm = (head.iter().map(|v| v * v).sum::<f64>() * shifted.iter().map(|v| v * v).sum::<f64>()).sqrt();
        if norm > 0.0 { both / norm } else { 0.0 }
    };
    let longest = x.len() / 3;
    let r: Vec<f64> = (0..=longest + 1).map(|lag| if lag == 0 { 1.0 } else { correlation(lag) }).collect();
    (2..=longest).find(|&lag| r[lag] > 0.8 && r[lag] > r[lag - 1] && r[lag] >= r[lag + 1]).map(|lag| (lag, r[lag]))
}

/// Build a world, step it `steps` times and read every step.
fn run(
    label: &str,
    floor: Floor<'_>,
    dials: Dials,
    steps: usize,
    build: &(dyn Fn(&mut Skeleton) -> Vec<Range<usize>> + Sync),
) -> Outcome {
    let mut s = Skeleton::new();
    let field = match floor {
        Floor::Plane(normal, distance) => {
            s.set_ground(normal, distance);
            None
        }
        Floor::Field(field) => {
            s.set_ground((0.0, 1.0, 0.0), 0.0);
            Some(field)
        }
    };
    dials.apply(&mut s);
    let units = build(&mut s);
    let mut recorder = Recorder::new(&s, units, steps);
    for step in 1..=steps {
        s.step_over(DT, G, dials.passes, field);
        recorder.observe(&s, floor, step);
    }
    recorder.finish(label)
}

/// `count` jobs over the machine's threads, results in job order. The scenes are
/// independent and deterministic, so how many threads ran them changes nothing but the
/// wait.
fn parallel<T: Send>(count: usize, job: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<T>>> = Mutex::new((0..count).map(|_| None).collect());
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 12);
    std::thread::scope(|scope| {
        for _ in 0..threads.min(count.max(1)) {
            scope.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= count {
                    break;
                }
                let value = job(k);
                results.lock().expect("a job panicked")[k] = Some(value);
            });
        }
    });
    results
        .into_inner()
        .expect("a job panicked")
        .into_iter()
        .map(|v| v.expect("every job ran"))
        .collect()
}

// -- reading a set of runs ----------------------------------------------------------

/// How a run with sleeping off ended, by the settling test first and the energy second.
fn residual(outcome: &Outcome) -> &'static str {
    if outcome.units.iter().all(|u| u.tail_ready >= 1.0) {
        "still"
    } else if outcome.kinetic[2] < 0.5 * outcome.kinetic[1] && outcome.kinetic[1] < 0.5 * outcome.kinetic[0] {
        "decaying"
    } else if outcome.period.is_some() {
        "cycling"
    } else {
        "plateau"
    }
}

/// One row of a scene's table: the same worlds with sleeping on and with it off.
fn report(name: &str, asleep: &[Outcome], awake: &[Outcome]) -> String {
    let mut text = String::new();
    let units: Vec<&Unit> = asleep.iter().flat_map(|o| o.units.iter()).collect();
    let slept: Vec<f64> = units.iter().filter_map(|u| u.slept_at.map(|s| s as f64)).collect();
    let by_240 = slept.iter().filter(|&&s| s <= 240.0).count();
    let worlds = asleep.iter().filter(|o| o.all_asleep_at.is_some()).count();
    let _ = writeln!(
        text,
        "{name}\n  sleeping on : {} of {} units asleep (median step {:.0}, {} by step 240); {} of {} worlds wholly asleep",
        slept.len(),
        units.len(),
        median(&slept),
        by_240,
        worlds,
        asleep.len(),
    );
    if awake.is_empty() {
        return text;
    }
    let free: Vec<&Unit> = awake.iter().flat_map(|o| o.units.iter()).collect();
    let ready_end = free.iter().filter(|u| u.unready_end == 0).count();
    let ready_ever: Vec<f64> = free.iter().filter_map(|u| u.ready_at.map(|s| s as f64)).collect();
    let steady = free.iter().filter(|u| u.tail_ready >= 1.0).count();
    let mut classes = [0usize; 4];
    for outcome in awake {
        let k = ["still", "decaying", "cycling", "plateau"]
            .iter()
            .position(|c| *c == residual(outcome))
            .expect("a known class");
        classes[k] += 1;
    }
    let periods: Vec<f64> = awake.iter().filter_map(|o| o.period.map(|p| p.0 as f64)).collect();
    let over = |f: fn(&Outcome) -> f64| awake.iter().map(f).collect::<Vec<_>>();
    let worst = |f: fn(&Outcome) -> f64| awake.iter().map(f).fold(0.0_f64, f64::max);
    let mut causes = [0u32; 4];
    for outcome in awake {
        for (total, one) in causes.iter_mut().zip(outcome.causes) {
            *total += one;
        }
    }
    let _ = writeln!(
        text,
        "  sleeping off: {} of {} units ready at the end, {} ready through the whole tail, {} ever ready (median first step {:.0})",
        ready_end,
        free.len(),
        steady,
        ready_ever.len(),
        median(&ready_ever),
    );
    let _ = writeln!(
        text,
        "    residual   : still {} / decaying {} / cycling {} / plateau {} of {} worlds; {} periodic (median period {:.0} steps)",
        classes[0],
        classes[1],
        classes[2],
        classes[3],
        awake.len(),
        periods.len(),
        median(&periods),
    );
    let _ = writeln!(
        text,
        "    kinetic J  : median by quarter {:.3e} -> {:.3e} -> {:.3e}; unready bodies at the end, median {:.0} max {:.0}",
        median(&over(|o| o.kinetic[0])),
        median(&over(|o| o.kinetic[1])),
        median(&over(|o| o.kinetic[2])),
        median(&over(|o| o.unready_end as f64)),
        worst(|o| o.unready_end as f64),
    );
    let _ = writeln!(
        text,
        "    tail       : motion/limit median {:.2} max {:.2}; point speed max {:.4} m/s; contacts median {:.0}; load/weight median {:.3}",
        median(&over(|o| o.tail_ratio)),
        worst(|o| o.tail_ratio),
        worst(|o| o.tail_point_speed),
        median(&over(|o| o.tail_contacts)),
        median(&over(|o| o.tail_load)),
    );
    let _ = writeln!(
        text,
        "    errors     : ground penetration max {:.2} mm; joint gap max {:.2} mm; hinge axis max {:.4} rad",
        1e3 * worst(|o| o.tail_penetration),
        1e3 * worst(|o| o.tail_joint_gap),
        worst(|o| o.tail_hinge_error),
    );
    let _ = writeln!(
        text,
        "    resets     : turn only {} / move only {} / both {} / sum only {}",
        causes[0], causes[1], causes[2], causes[3],
    );
    if free.iter().all(|u| u.resets.len() == BONES.len()) {
        let mut by_bone = [0u32; 17];
        let mut rigs_with = [0u32; 17];
        for unit in &free {
            for (k, &count) in unit.resets.iter().enumerate() {
                by_bone[k] += count;
                rigs_with[k] += u32::from(count > 0);
            }
        }
        let mut order: Vec<usize> = (0..17).collect();
        order.sort_by_key(|&k| std::cmp::Reverse(by_bone[k]));
        let _ = write!(text, "    restless   :");
        for &k in order.iter().take(6).filter(|&&k| by_bone[k] > 0) {
            let _ = write!(text, " {} {} resets in {} rigs;", BONES[k], by_bone[k], rigs_with[k]);
        }
        text.push('\n');
    }
    text
}

fn emit(scene: &str, text: &str) {
    println!("{text}");
    if let Ok(dir) = std::env::var("REST_LADDER_CSV") {
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(format!("{dir}/{scene}.txt"), text);
    }
}

// -- fixtures ----------------------------------------------------------------------

/// What is dropped.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Kit {
    /// The seventeen-bone rig of `articulated_laws.rs`, bone for bone and joint for joint,
    /// spawned exactly as that file spawns it: a pelvis, a spine of four, and four limbs of
    /// three -- shoulders and hips as balls, elbows and knees as hinges with a range. Its
    /// spawn is not a pose: each upper arm starts in the same place as the thigh below it
    /// and 0.475 m from the shoulder it is jointed to, so the first step pulls two limbs
    /// through each other.
    Rig,
    /// The same bodies and joints, spawned with every joint closed and nothing overlapping:
    /// arms raised from the shoulders, legs hanging from the hips.
    Untangled,
    /// The pelvis and the spine alone: five bodies on four free balls.
    Spine,
    /// The pelvis and one leg: a ball and two hinges.
    Leg,
    /// Two limb segments on one hinge.
    Pair,
}

/// Adds `kit` to the skeleton about the origin, its length up +Y.
fn build(into: &mut Skeleton, kit: Kit) {
    const REFUSED: &str = "the fixture could not build its own rig";
    let limb_at = |y: f64, z: f64, raised: bool| {
        let mut body = Body::capsule(4.0, 0.06, 0.25, (0.0, y, z));
        if raised {
            body.orientation = Quaternion::from_axis_angle((1.0, 0.0, 0.0), std::f64::consts::PI);
        }
        body
    };
    let hinge = |a: usize, b: usize| Joint::Hinge {
        a,
        b,
        anchor_a: (0.0, -0.125, 0.0),
        anchor_b: (0.0, 0.125, 0.0),
        axis_a: (1.0, 0.0, 0.0),
        axis_b: (1.0, 0.0, 0.0),
        min: -0.1,
        max: 2.2,
    };
    if kit == Kit::Pair {
        let a = into.add_body(limb_at(0.0, 0.0, false));
        let b = into.add_body(limb_at(-0.25, 0.0, false));
        assert!(into.add_joint(hinge(a, b)), "{REFUSED}");
        return;
    }
    let base = into.len();
    let pelvis = into.add_body(Body::capsule(8.0, 0.1, 0.16, (0.0, 0.0, 0.0)));
    if kit != Kit::Leg {
        let mut up = pelvis;
        for i in 0..4 {
            let link = into.add_body(Body::capsule(6.0, 0.08, 0.2, (0.0, 0.2 + 0.2 * i as f64, 0.0)));
            assert!(into.add_joint(Joint::free_ball(up, link, (0.0, 0.1, 0.0), (0.0, -0.1, 0.0))), "{REFUSED}");
            up = link;
        }
    }
    let limbs = match kit {
        Kit::Rig | Kit::Untangled => 0..4,
        Kit::Leg => 2..3,
        Kit::Spine | Kit::Pair => 0..0,
    };
    for limb in limbs {
        let (root, side) = if limb < 2 { (base + 3, 1.0) } else { (pelvis, -1.0) };
        let side_z = if limb % 2 == 0 { 0.15 } else { -0.15 };
        let mut previous = root;
        for segment in 0..3 {
            let along = 0.25 * segment as f64;
            let body = into.add_body(match (kit, limb < 2) {
                (Kit::Rig, _) => limb_at(side * along, side_z, false),
                (_, true) => limb_at(0.725 + along, side_z, true),
                (_, false) => limb_at(-0.125 - along, side_z, false),
            });
            let joint = if segment == 0 {
                Joint::free_ball(previous, body, (0.0, 0.0, side_z), (0.0, 0.125, 0.0))
            } else {
                hinge(previous, body)
            };
            assert!(into.add_joint(joint), "{REFUSED}");
            previous = body;
        }
    }
}

/// `kit`, turned by `turn` about its origin and set down over `at` with its lowest surface
/// `clearance` above the ground beneath that surface. Returns its bodies.
fn dropped(
    into: &mut Skeleton,
    kit: Kit,
    at: (f64, f64),
    turn: Quaternion,
    clearance: f64,
    field: Option<Field<'_>>,
) -> Range<usize> {
    let start = into.len();
    build(into, kit);
    let bodies = start..into.len();
    let mut lowest = f64::INFINITY;
    for i in bodies.clone() {
        let mut body = into.body(i);
        let p = turn.rotate_point(body.position);
        body.position = (p.0 + at.0, p.1, p.2 + at.1);
        body.orientation = turn.multiply(&body.orientation);
        let half = body.orientation.rotate_point((0.0, body.half_length, 0.0));
        for end in [1.0, -1.0] {
            let x = body.position.0 + end * half.0;
            let z = body.position.2 + end * half.2;
            let ground = field.map_or(0.0, |f| f.height_at(x, z));
            lowest = lowest.min(body.position.1 + end * half.1 - body.radius - ground);
        }
        into.set_body(i, body);
    }
    for i in bodies.clone() {
        let mut body = into.body(i);
        body.position.1 += clearance - lowest;
        into.set_body(i, body);
    }
    bodies
}

/// The turn the `k`th rig of the landing-pose ensemble lands with.
fn landing_turn(k: usize) -> Quaternion {
    let n = k as f64;
    let height = 2.0 * (n * 0.618_033_988_75 + 0.1).fract() - 1.0;
    let around = std::f64::consts::TAU * (n * 0.754_877_666_25 + 0.3).fract();
    let angle = std::f64::consts::TAU * (n * 0.569_840_290_99 + 0.7).fract();
    let flat = (1.0 - height * height).max(0.0).sqrt();
    Quaternion::from_axis_angle((flat * around.cos(), height, flat * around.sin()), angle)
}

fn clearance_of(draw: usize) -> f64 {
    0.55 * (1.0 + draw as f64 * 1e-12)
}

/// A limb-sized body of either shape: `facets` of zero is a capsule.
fn limb(facets: u32, position: V) -> Body {
    Body::prism(4.0, 0.06, 0.25, facets, position)
}

fn shape_name(facets: u32) -> &'static str {
    if facets == 0 { "capsule" } else { "prism" }
}

// -- (a) one body on flat ground ------------------------------------------------------

/// One capsule and one square prism, each alone on the plane: laid down and dropped, and
/// dropped on a tilt so that it lands on an end and falls over. And the smallest scene
/// with a contact between two bodies that is a single point: one body dropped across
/// another that is lying on the ground, so that it ends leaning on it.
///
/// The plane treats every shape as the capsule that encloses it, so a prism's flats only
/// come into the second half.
#[test]
#[ignore = "diagnostic ladder: prints a table and asserts nothing"]
fn a_one_body_on_flat_ground() {
    let mut text = String::from("(a) one body on flat ground\n");
    for facets in [0, 4] {
        for (pose, tilt, height) in [("laid", 0.0, 0.36), ("tilted", 0.6, 0.5)] {
            let build = move |s: &mut Skeleton| {
                let mut body = limb(facets, (0.0, height, 0.0));
                body.orientation = Quaternion::from_axis_angle((0.0, 0.0, 1.0), tilt - std::f64::consts::FRAC_PI_2);
                let i = s.add_body(body);
                std::iter::once(i..i + 1).collect()
            };
            let name = format!("a {} {pose}", shape_name(facets));
            let asleep = run(&name, Floor::FLAT, Dials::PLAIN, STEPS, &build);
            let awake = run(&format!("{name} awake"), Floor::FLAT, Dials::PLAIN.awake(), STEPS, &build);
            text += &report(&name, &[asleep], &[awake]);
        }
        for (pose, lean) in [("crossed at the end", None), ("leaning at 15 deg", Some(15.0_f64)), ("leaning at 22 deg", Some(22.0))] {
            let build = move |s: &mut Skeleton| two_bodies(s, facets, lean);
            let name = format!("a2 {} {pose}", shape_name(facets));
            let asleep = run(&name, Floor::FLAT, Dials::PLAIN, STEPS, &build);
            let awake = run(&format!("{name} awake"), Floor::FLAT, Dials::PLAIN.awake(), STEPS, &build);
            text += &report(&name, &[asleep], &[awake]);
        }
    }
    emit("a", &text);
}

/// The two-body scenes of (a): a limb dropped across another at its end, or, with `lean`,
/// leaning on a heavy one at that many degrees. The same scene as the leaning law in
/// `articulated_laws.rs`.
fn two_bodies(s: &mut Skeleton, facets: u32, lean: Option<f64>) -> Vec<Range<usize>> {
    // The body underneath lies along z. Leant on, it is ten times the weight of a limb, so
    // that it is the leaning body that is being asked to hold and not the one under it
    // being asked not to roll away.
    let mut under = Body::prism(if lean.is_some() { 40.0 } else { 4.0 }, 0.06, 0.25, facets, (0.0, 0.06, 0.0));
    under.orientation = Quaternion::from_axis_angle((1.0, 0.0, 0.0), std::f64::consts::FRAC_PI_2);
    let a = s.add_body(under);
    let over = match lean {
        None => {
            let mut over = limb(facets, (0.11, 0.20, 0.0));
            over.orientation = Quaternion::from_axis_angle((0.0, 0.0, 1.0), -std::f64::consts::FRAC_PI_2);
            over
        }
        Some(degrees) => {
            // Half a metre long, one end on the plane and its side against the body
            // underneath, a millimetre clear of both: statically it holds at either angle
            // on a coefficient of 0.5, its weight falling between the two.
            let angle = degrees.to_radians();
            let foot = (-(0.12 + 0.001) / angle.sin(), 0.06 + 0.001, 0.0);
            let along = (angle.cos(), angle.sin(), 0.0);
            let mut over = Body::prism(4.0, 0.06, 0.5, facets, add(foot, scale(along, 0.25)));
            over.orientation = Quaternion::from_axis_angle((0.0, 0.0, 1.0), angle - std::f64::consts::FRAC_PI_2);
            over
        }
    };
    let b = s.add_body(over);
    vec![a..a + 1, b..b + 1]
}

// -- (b) one body on a tilted plane -----------------------------------------------------

/// One body laid on a plane tilted by `degrees`, its length either down the slope (it can
/// only slide) or across it (it can roll), stepped with sleeping as `dials` says. Returns
/// how far it travelled downhill between half a second in and the end, how far in the two
/// seconds after that half second, the downhill speed at the half second, and the
/// recorder's reading.
fn on_a_slope(label: &str, facets: u32, degrees: f64, across: bool, dials: Dials) -> (f64, f64, f64, Outcome) {
    let angle = degrees.to_radians();
    let up = (-angle.sin(), angle.cos(), 0.0);
    let downhill = (-angle.cos(), -angle.sin(), 0.0);
    let floor = Floor::Plane(up, 0.0);
    let mut s = Skeleton::new();
    s.set_ground(up, 0.0);
    dials.apply(&mut s);
    let mut body = limb(facets, scale(up, 0.06 + 0.002));
    body.orientation = if across {
        Quaternion::from_axis_angle((1.0, 0.0, 0.0), std::f64::consts::FRAC_PI_2)
    } else {
        Quaternion::from_axis_angle((0.0, 0.0, 1.0), angle - std::f64::consts::FRAC_PI_2)
    };
    let i = s.add_body(body);
    let mut recorder = Recorder::new(&s, std::iter::once(i..i + 1).collect(), STEPS);
    let (mut settled, mut speed, mut early) = ((0.0, 0.0, 0.0), 0.0, 0.0);
    for step in 1..=STEPS {
        s.step(DT, G, dials.passes);
        recorder.observe(&s, floor, step);
        if step == 30 {
            settled = s.position(i);
            speed = dot(s.velocity(i), downhill);
        }
        if step == 150 {
            early = dot(sub(s.position(i), settled), downhill);
        }
    }
    let travel = dot(sub(s.position(i), settled), downhill);
    (travel, early, speed, recorder.finish(label))
}

/// Coulomb's angle, exactly: a body laid down a slope holds if and only if the tangent of
/// the slope is below the coefficient, and when it lets go it accelerates at
/// `g (sin - mu cos)`. And the rolling half: laid across the slope a round body rolls
/// unless the rolling resistance, as a fraction of the radius, exceeds that same tangent.
#[test]
#[ignore = "diagnostic ladder: prints a table and asserts nothing"]
fn b_one_body_on_a_tilted_plane() {
    let mut text = String::from("(b) one body on a tilted plane\n");
    for facets in [0, 4] {
        for (degrees, friction) in [(20.0_f64, 0.5), (35.0, 0.5), (35.0, 0.8)] {
            let sliding = Dials { friction, rolling: 0.0, ..Dials::PLAIN };
            let name = format!("b {} down the slope, {degrees} deg, mu {friction}", shape_name(facets));
            let (travel, early, speed, asleep) = on_a_slope(&name, facets, degrees, false, sliding);
            let (free_travel, free_early, free_speed, awake) =
                on_a_slope(&format!("{name} awake"), facets, degrees, false, sliding.awake());
            let angle = degrees.to_radians();
            let holds = angle.tan() < friction;
            let pull = GRAVITY * (angle.sin() - friction * angle.cos());
            let predicted = if holds { 0.0 } else { free_speed * 2.0 + 0.5 * pull * 4.0 };
            let verdict = if holds { free_travel.abs() < 1e-8 && travel.abs() < 1e-8 } else { free_early > 0.5 * predicted };
            text += &report(&name, &[asleep], &[awake]);
            let _ = writeln!(
                text,
                "    coulomb    : predicted {}; travelled {travel:.3e} m asleep-able, {free_travel:.3e} m awake; in the \
                 2 s after settling {early:.4} / {free_early:.4} m against {predicted:.4} predicted (speed at settling \
                 {speed:.4} / {free_speed:.4} m/s) -> {}",
                if holds { "holds" } else { "slides" },
                if verdict { "as predicted" } else { "NOT as predicted" },
            );
        }
        for (degrees, rolling, passes) in [(0.0_f64, 0.25, 8), (5.0, 0.25, 8), (10.0, 0.25, 8), (10.0, 0.25, 32), (10.0, 0.25, 128), (20.0, 0.25, 8), (20.0, 0.5, 8), (20.0, 0.5, 128)] {
            let dials = Dials { rolling, passes, ..Dials::PLAIN };
            let name = format!("b2 {} across the slope, {degrees} deg, mu 0.5, rolling {rolling}, {passes} passes", shape_name(facets));
            let (travel, _, _, asleep) = on_a_slope(&name, facets, degrees, true, dials);
            let (free_travel, _, _, awake) = on_a_slope(&format!("{name} awake"), facets, degrees, true, dials.awake());
            let holds = degrees.to_radians().tan() < rolling;
            text += &report(&name, &[asleep], &[awake]);
            let _ = writeln!(
                text,
                "    rolling    : predicted {}; travelled {travel:.3e} m asleep-able, {free_travel:.3e} m awake",
                if holds { "holds" } else { "rolls" },
            );
        }
    }
    emit("b", &text);
}

// -- (a3) one chain on flat ground ------------------------------------------------------

/// The rungs between a lone body and a rig: two limb segments on a hinge, a pelvis with
/// one leg, and a pelvis with the spine, each dropped at eight of the landing poses.
/// Bodies on joints and the plane, with self-collision on and off.
#[test]
#[ignore = "diagnostic ladder: prints a table and asserts nothing"]
fn a3_one_chain_on_flat_ground() {
    let mut text = String::from("(a3) one jointed chain on flat ground, 8 landing poses x 2 draws\n");
    for kit in [Kit::Pair, Kit::Leg, Kit::Spine] {
        for (which, dials) in [("self-collision on", Dials::PLAIN), ("self-collision off", NO_SELF)] {
            let name = format!("a3 {kit:?}, {which}");
            let (asleep, awake) = lone(&name, kit, 8, Floor::FLAT, dials);
            text += &report(&name, &asleep, &awake);
        }
    }
    emit("a3", &text);
}

/// (a3) with self-collision off under each dial of [`dial_sweep`]: what is left when
/// there is nothing but joints and the plane.
#[test]
#[ignore = "diagnostic ladder: prints a table and asserts nothing; a minute or two"]
fn a3_one_chain_under_each_dial() {
    let mut text = String::from("(a3) one jointed chain on flat ground under each dial, self-collision off\n");
    for kit in [Kit::Pair, Kit::Leg, Kit::Spine] {
        for (dial, dials) in dial_sweep(NO_SELF) {
            let name = format!("a3-dial {kit:?}, {dial}");
            let (asleep, awake) = lone(&name, kit, 8, Floor::FLAT, dials);
            text += &report(&name, &asleep, &awake);
        }
    }
    emit("a3_dials", &text);
}

// -- (c) one rig on flat ground -----------------------------------------------------------

const POSES: usize = 20;
const POSE_DRAWS: usize = 2;

/// `kit` at each of the first `poses` landing poses and every draw, each alone in its own
/// world, with sleeping on and then off.
fn lone(name: &str, kit: Kit, poses: usize, floor: Floor<'_>, dials: Dials) -> (Vec<Outcome>, Vec<Outcome>) {
    let field = match floor {
        Floor::Field(field) => Some(field),
        Floor::Plane(..) => None,
    };
    let both = parallel(2 * poses * POSE_DRAWS, |job| {
        let (k, sleeping) = (job / 2, job % 2 == 0);
        let (pose, draw) = (k % poses, k / poses);
        let build = |s: &mut Skeleton| vec![dropped(s, kit, (0.0, 0.0), landing_turn(pose), clearance_of(draw), field)];
        let dials = if sleeping { dials } else { dials.awake() };
        let label = format!("{name} pose {pose} draw {draw}{}", if sleeping { "" } else { " awake" });
        run(&label, floor, dials, STEPS, &build)
    });
    halves(both)
}

/// Splits runs that alternate sleeping on and off into the two sets.
fn halves(both: Vec<Outcome>) -> (Vec<Outcome>, Vec<Outcome>) {
    let mut both = both.into_iter();
    let (mut asleep, mut awake) = (Vec::new(), Vec::new());
    while let (Some(a), Some(b)) = (both.next(), both.next()) {
        asleep.push(a);
        awake.push(b);
    }
    (asleep, awake)
}

/// The named dials the public interface offers against the defect, one changed at a time
/// from `p`.
fn dial_sweep(p: Dials) -> Vec<(&'static str, Dials)> {
    vec![
        ("no dial changed", p),
        ("16 passes", Dials { passes: 16, ..p }),
        ("32 passes", Dials { passes: 32, ..p }),
        ("128 passes", Dials { passes: 128, ..p }),
        ("joint damping 0.1", Dials { damping: 0.1, ..p }),
        ("joint damping 0.3", Dials { damping: 0.3, ..p }),
        ("rolling resistance 0", Dials { rolling: 0.0, ..p }),
        ("friction 0.3", Dials { friction: 0.3, ..p }),
        ("friction 0.8", Dials { friction: 0.8, ..p }),
    ]
}

const NO_SELF: Dials = Dials { self_collision: false, ..Dials::PLAIN };

/// One rig dropped at each of the twenty landing poses, two draws, self-collision on and
/// off, as the laws spawn it and untangled.
#[test]
#[ignore = "diagnostic ladder: prints a table and asserts nothing; about a minute"]
fn c_one_rig_on_flat_ground() {
    let mut text = String::from("(c) one rig on flat ground, 20 landing poses x 2 draws\n");
    for kit in [Kit::Rig, Kit::Untangled] {
        for (which, dials) in [("self-collision on", Dials::PLAIN), ("self-collision off", NO_SELF)] {
            let name = format!("c {kit:?}, {which}");
            let (asleep, awake) = lone(&name, kit, POSES, Floor::FLAT, dials);
            text += &report(&name, &asleep, &awake);
            let _ = write!(text, "    by pose (step asleep | unready at the end, awake):");
            for (k, (a, b)) in asleep.iter().zip(&awake).enumerate() {
                if k % POSES == 0 {
                    let _ = write!(text, "\n      draw {}:", k / POSES);
                }
                let slept = a.units[0].slept_at.map_or("never".to_string(), |s| s.to_string());
                let _ = write!(text, " {slept}|{}", b.unready_end);
            }
            text.push('\n');
        }
    }
    emit("c", &text);
}

/// (c) again under each dial of [`dial_sweep`], with self-collision on and off.
#[test]
#[ignore = "diagnostic ladder: prints a table and asserts nothing; several minutes"]
fn c_one_rig_under_each_dial() {
    let mut text = String::from("(c) one rig on flat ground under each dial, 20 landing poses x 2 draws\n");
    for kit in [Kit::Rig, Kit::Untangled] {
        for (which, base) in [("self-collision on", Dials::PLAIN), ("self-collision off", NO_SELF)] {
            for (dial, dials) in dial_sweep(base) {
                let name = format!("c-dial {kit:?}, {which}, {dial}");
                let (asleep, awake) = lone(&name, kit, POSES, Floor::FLAT, dials);
                text += &report(&name, &asleep, &awake);
            }
        }
    }
    emit("c_dials", &text);
}

// -- (d) one rig on a slope -----------------------------------------------------------------

const SLOPE_SIDE: usize = 257;
const SLOPE_CELL: f64 = 1.0;
const SLOPE_ORIGIN: (f64, f64) = (-128.0, -128.0);

/// A height field that is one plane, falling towards +x at `degrees`. Bilinear cells of a
/// linear height are that plane exactly, so there is no crease anywhere in it.
fn planar_slope(degrees: f64) -> Vec<f64> {
    let fall = degrees.to_radians().tan();
    let mut heights = vec![0.0; SLOPE_SIDE * SLOPE_SIDE];
    for row in 0..SLOPE_SIDE {
        for column in 0..SLOPE_SIDE {
            let x = SLOPE_ORIGIN.0 + column as f64 * SLOPE_CELL;
            heights[row * SLOPE_SIDE + column] = -fall * x;
        }
    }
    heights
}

/// How far each rig went downhill in all, and over the last quarter of the run, as metres
/// along the slope travelled by its centre of mass.
fn downhill(outcomes: &[Outcome], degrees: f64) -> (Vec<f64>, Vec<f64>) {
    let angle = degrees.to_radians();
    let along = (angle.cos(), -angle.sin(), 0.0);
    let total = outcomes.iter().map(|o| dot(sub(o.units[0].com[2], o.units[0].com[0]), along)).collect();
    let tail = outcomes.iter().map(|o| dot(sub(o.units[0].com[2], o.units[0].com[1]), along)).collect();
    (total, tail)
}

/// One rig dropped on a planar height field at ten and twenty degrees, friction 0.5. A rig
/// has no single Coulomb angle -- its bones can roll where a block could only slide -- so
/// what is read is whether it rests, and if not whether it goes as a whole or creeps.
#[test]
#[ignore = "diagnostic ladder: prints a table and asserts nothing; about a minute"]
fn d_one_rig_on_a_slope() {
    let mut text = String::from("(d) one rig on a planar slope, 20 landing poses x 2 draws, mu 0.5\n");
    for degrees in [10.0, 20.0] {
        let heights = planar_slope(degrees);
        let field = Field::new(&heights, SLOPE_SIDE, SLOPE_SIDE, SLOPE_CELL, SLOPE_ORIGIN).expect("a valid grid");
        for kit in [Kit::Rig, Kit::Untangled] {
            for (which, dials) in [("self-collision on", Dials::PLAIN), ("self-collision off", NO_SELF)] {
                let name = format!("d {degrees} deg, {kit:?}, {which}");
                let (asleep, awake) = lone(&name, kit, POSES, Floor::Field(field), dials);
                text += &report(&name, &asleep, &awake);
                for (mode, outcomes) in [("sleeping on ", &asleep), ("sleeping off", &awake)] {
                    let (total, tail) = downhill(outcomes, degrees);
                    let far = total.iter().filter(|&&t| t > 1.0).count();
                    let creeping = tail.iter().filter(|&&t| t.abs() > 1e-3).count();
                    let tail_seconds = (STEPS / 4) as f64 * DT;
                    let _ = writeln!(
                        text,
                        "    downhill, {mode}: total median {:.3} m, max {:.3} m, {far} of {} went over a metre; over the last \
                         quarter median {:.2e} m, max {:.3} m ({:.4} m/s), {creeping} of {} moved over a millimetre",
                        median(&total),
                        total.iter().copied().fold(0.0_f64, f64::max),
                        total.len(),
                        median(&tail),
                        tail.iter().copied().fold(0.0_f64, f64::max),
                        tail.iter().copied().fold(0.0_f64, f64::max) / tail_seconds,
                        tail.len(),
                    );
                }
            }
        }
    }
    emit("d", &text);
}

// -- (e) heaps -------------------------------------------------------------------------------

const HEAP_DRAWS: usize = 8;

/// How the rigs of a heap are arranged between worlds.
#[derive(Clone, Copy, PartialEq)]
enum Arrangement {
    /// One world, the 0.7 m grid of the heap law: the rigs land on one another.
    Together,
    /// The same rigs at the same places and poses, each in a world of its own, so that no
    /// contact between two rigs exists and nothing else differs.
    Split,
    /// One world, the rigs 2.5 m apart: no contact between rigs, but one skeleton, one
    /// colouring and one broad phase.
    Apart,
}

/// What a heap is made of and how it is laid out.
#[derive(Clone, Copy)]
struct Heap {
    kit: Kit,
    rigs: usize,
    /// At the landing poses, or upright as the heap law drops them.
    turned: bool,
    arrangement: Arrangement,
}

/// A heap at every draw, with sleeping on and then off. A split heap returns one world a
/// rig.
fn heaps(name: &str, heap: Heap, dials: Dials) -> (Vec<Outcome>, Vec<Outcome>) {
    let Heap { kit, rigs, turned, arrangement } = heap;
    let side = (rigs as f64).sqrt().ceil() as usize;
    let turn = move |k: usize| if turned { landing_turn(k) } else { Quaternion::identity() };
    let place = move |k: usize, pitch: f64| ((k % side) as f64 * pitch, (k / side) as f64 * pitch);
    let worlds = if arrangement == Arrangement::Split { rigs } else { 1 };
    let all = parallel(2 * HEAP_DRAWS * worlds, |job| {
        let (sleeping, rest) = (job % 2 == 0, job / 2);
        let (draw, world) = (rest / worlds, rest % worlds);
        let one = |s: &mut Skeleton, k: usize, pitch: f64| dropped(s, kit, place(k, pitch), turn(k), clearance_of(draw), None);
        let build = |s: &mut Skeleton| -> Vec<Range<usize>> {
            match arrangement {
                Arrangement::Split => vec![one(s, world, 0.7)],
                Arrangement::Together => (0..rigs).map(|k| one(s, k, 0.7)).collect(),
                Arrangement::Apart => (0..rigs).map(|k| one(s, k, 2.5)).collect(),
            }
        };
        let dials = if sleeping { dials } else { dials.awake() };
        let label = format!("{name} draw {draw} world {world}{}", if sleeping { "" } else { " awake" });
        run(&label, Floor::FLAT, dials, STEPS, &build)
    });
    halves(all)
}

fn heap_name(heap: Heap) -> String {
    format!("{} {:?} {}", heap.rigs, heap.kit, if heap.turned { "at the landing poses" } else { "upright" })
}

/// Heaps of two, four and eight rigs on the grid of the heap law, against the two
/// controls that take the contacts between rigs away.
#[test]
#[ignore = "diagnostic ladder: prints a table and asserts nothing; about a minute"]
fn e_heaps_of_rigs() {
    let mut text = format!("(e) heaps of rigs on a 0.7 m grid, {HEAP_DRAWS} draws\n");
    for kit in [Kit::Rig, Kit::Untangled] {
        for turned in [false, true] {
            for rigs in [2, 4, 8] {
                for (which, arrangement) in [("together", Arrangement::Together), ("split into own worlds", Arrangement::Split), ("apart in one world", Arrangement::Apart)] {
                    let heap = Heap { kit, rigs, turned, arrangement };
                    let name = format!("e {}, {which}", heap_name(heap));
                    let (asleep, awake) = heaps(&name, heap, Dials::PLAIN);
                    text += &report(&name, &asleep, &awake);
                }
            }
        }
    }
    emit("e", &text);
}

/// (e), together and at the landing poses, under each dial of [`dial_sweep`] and with
/// self-collision off.
#[test]
#[ignore = "diagnostic ladder: prints a table and asserts nothing; several minutes"]
fn e_heaps_under_each_dial() {
    heaps_under_each_dial(Kit::Rig, "e_dials");
}

/// The same for rigs spawned untangled.
#[test]
#[ignore = "diagnostic ladder: prints a table and asserts nothing; several minutes"]
fn e_untangled_heaps_under_each_dial() {
    heaps_under_each_dial(Kit::Untangled, "e_dials_untangled");
}

fn heaps_under_each_dial(kit: Kit, scene: &str) {
    let mut text = format!("(e) heaps of rigs together under each dial, {HEAP_DRAWS} draws\n");
    let mut sweep = dial_sweep(Dials::PLAIN);
    sweep.push(("self-collision off", NO_SELF));
    for (dial, dials) in &sweep {
        for rigs in [2, 4, 8] {
            let heap = Heap { kit, rigs, turned: true, arrangement: Arrangement::Together };
            let name = format!("e-dial {}, {dial}", heap_name(heap));
            let (asleep, awake) = heaps(&name, heap, *dials);
            text += &report(&name, &asleep, &awake);
        }
    }
    emit(scene, &text);
}

// -- (p) the pair patch, off and on ---------------------------------------------------------

/// The rungs the pair patch is judged on, with `Skeleton::set_contact_patch` off and then on
/// in the same build: the two-body scenes of (a), the rolling-hold scene of (b), the lone
/// untangled rig of (c) and the untangled heap of eight of (e), at the landing poses.
#[test]
#[ignore = "diagnostic ladder: prints a table and asserts nothing; a few minutes"]
fn p_the_pair_patch_off_and_on() {
    let mut text = String::from("(p) the pair patch, off and on\n");
    for patch in [false, true] {
        let dials = Dials { patch, ..Dials::PLAIN };
        let which = if patch { "patch on" } else { "patch off" };
        for (pose, lean) in [("crossed at the end", None), ("leaning at 15 deg", Some(15.0_f64)), ("leaning at 22 deg", Some(22.0))] {
            let build = move |s: &mut Skeleton| two_bodies(s, 0, lean);
            let name = format!("p a2 capsule {pose}, {which}");
            let asleep = run(&name, Floor::FLAT, dials, STEPS, &build);
            let awake = run(&format!("{name} awake"), Floor::FLAT, dials.awake(), STEPS, &build);
            text += &report(&name, &[asleep], &[awake]);
        }
        let rolling = Dials { rolling: 0.25, ..dials };
        let name = format!("p b2 capsule across the slope, 10 deg, rolling 0.25, {which}");
        let (travel, _, _, asleep) = on_a_slope(&name, 0, 10.0, true, rolling);
        let (free_travel, _, _, awake) = on_a_slope(&format!("{name} awake"), 0, 10.0, true, rolling.awake());
        text += &report(&name, &[asleep], &[awake]);
        let _ = writeln!(text, "    rolling    : predicted holds; travelled {travel:.3e} m asleep-able, {free_travel:.3e} m awake");

        let name = format!("p c Untangled, self-collision on, {which}");
        let (asleep, awake) = lone(&name, Kit::Untangled, POSES, Floor::FLAT, dials);
        text += &report(&name, &asleep, &awake);

        let heap = Heap { kit: Kit::Untangled, rigs: 8, turned: true, arrangement: Arrangement::Together };
        let name = format!("p e {}, together, {which}", heap_name(heap));
        let (asleep, awake) = heaps(&name, heap, dials);
        text += &report(&name, &asleep, &awake);
    }
    emit("p", &text);
}

// -- (s) the stick memory, off and on -------------------------------------------------------

/// The rungs the stick memory is judged on, with `Skeleton::set_stick_memory` off and then on
/// in the same build and the pair patch off: the two leaning scenes of (a), the rolling-hold
/// scene of (b) at 5, 10 and 20 degrees, the lone untangled rig of (c) and the untangled heap
/// of eight of (e), at the landing poses. In the order the stop rule reads them.
#[test]
#[ignore = "diagnostic ladder: prints a table and asserts nothing; a few minutes"]
fn s_the_stick_memory_off_and_on() {
    let mut text = String::from("(s) the stick memory, off and on\n");
    for stick in [false, true] {
        let dials = Dials { patch: false, stick, ..Dials::PLAIN };
        let which = if stick { "memory on" } else { "memory off" };
        for degrees in [15.0_f64, 22.0] {
            let build = move |s: &mut Skeleton| two_bodies(s, 0, Some(degrees));
            let name = format!("s a2 capsule leaning at {degrees} deg, {which}");
            let asleep = run(&name, Floor::FLAT, dials, STEPS, &build);
            let awake = run(&format!("{name} awake"), Floor::FLAT, dials.awake(), STEPS, &build);
            text += &report(&name, &[asleep], &[awake]);
        }
        for degrees in [5.0, 10.0, 20.0] {
            let name = format!("s b2 capsule across the slope, {degrees} deg, rolling 0.25, {which}");
            let (travel, _, _, asleep) = on_a_slope(&name, 0, degrees, true, dials);
            let (free_travel, _, _, awake) = on_a_slope(&format!("{name} awake"), 0, degrees, true, dials.awake());
            text += &report(&name, &[asleep], &[awake]);
            let _ = writeln!(text, "    rolling    : travelled {travel:.3e} m asleep-able, {free_travel:.3e} m awake");
        }

        let name = format!("s c Untangled, self-collision on, {which}");
        let (asleep, awake) = lone(&name, Kit::Untangled, POSES, Floor::FLAT, dials);
        text += &report(&name, &asleep, &awake);

        let heap = Heap { kit: Kit::Untangled, rigs: 8, turned: true, arrangement: Arrangement::Together };
        let name = format!("s e {}, together, {which}", heap_name(heap));
        let (asleep, awake) = heaps(&name, heap, dials);
        text += &report(&name, &asleep, &awake);
    }
    emit("s", &text);
}
