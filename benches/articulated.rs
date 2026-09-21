//! What an articulated skeleton costs, and **which half of the step it is in**.
//!
//! The workload is a ragdoll pile: a humanoid is seventeen segments, and a crowd scene
//! is several hundred of them at once. One skeleton gives the shape of a step; the pile
//! gives the number that decides anything.
//!
//! # The pair that matters
//!
//! `step` is two things bolted together: **sweeps** over every body (predict under
//! gravity, integrate the spin, read the velocities back out) and **the solve**, a few
//! passes over the coloured joint sets. They want opposite optimisations -- sweeps are
//! contiguous and vectorisable, the solve is pairwise and random-access -- so knowing
//! which one the time is in is the whole point of measuring.
//!
//! `iterations/1` and `iterations/8` isolate it without instrumenting anything: the
//! sweeps run once either way, so the difference is seven solve passes and the solve's
//! cost per pass falls out of it. Anything left over is the sweeps.
//!
//! That is the number to look at before splitting the arrays into components for SIMD.
//! Component arrays are what AVX needs -- a `Vec<(f64, f64, f64)>` is stride three and
//! will not vectorise -- but it is a second restructure, and it only pays for the half of
//! the step the sweeps are.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rs_physics::articulated::{Body, Joint, Skeleton};

const DT: f64 = 1.0 / 60.0;
const G: (f64, f64, f64) = (0.0, -9.80665, 0.0);

/// Bodies in the rig this bench builds: a pelvis, four up the spine to the head, and four
/// limbs of three.
///
/// **Seventeen, where an animation rig usually has nineteen.** A rig carries a shoulder
/// bone each side for the mesh to be skinned to; it is not a segment with its own mass
/// and it moves with the chest, so a physics body count folds the two in. A rendering
/// one does not.
const BONES: usize = 17;

/// Bodies in the crowd. Six hundred is the order of magnitude a battle scene reaches,
/// and the point at which a serial solver stops being an option.
const PILE: usize = 600;

/// Steps into the drop at which [`a_heap_arriving`] is timed: far enough in that the rigs
/// have reached the ground and are folding against each other, and well short of the
/// point where anything has stopped.
const ARRIVING: usize = 120;

/// Steps before [`a_heap_at_thirty_seconds`] is timed.
///
/// From the fixture's measured settling curve rather than picked. Median body speed after
/// 30, 120, 300, 600, 900, 1200, 1800, 2400 and 3600 steps is 2.74, 1.64, 0.68, 0.27,
/// 0.22, 0.26, 0.29, 0.24 and 0.14 m/s: it falls by a factor of ten over the first ten
/// seconds and then creeps down. Thirty seconds is past the knee and short of the point
/// where the run costs more than the bench.
const SETTLED: usize = 1800;

/// Capsules in [`a_heap_that_has_settled`], and the height of each stack.
///
/// Three high because that is what settles: measured, stacks one, two and three deep all
/// reach the point where nothing is awake, and five deep does not. Ten thousand of them
/// so the count matches the other heaps here.
const STACKS: usize = 3333;
const STACK_HIGH: usize = 3;

/// `count` rigs dropped onto the plane in a grid close enough that they land on one
/// another. Nothing is pinned, so the heap has somewhere to put its energy.
fn dropped(count: usize) -> Skeleton {
    let mut s = Skeleton::new();
    s.set_ground((0.0, 1.0, 0.0), 0.0);
    let side = (count as f64).sqrt().ceil() as usize;
    for i in 0..count {
        corpse(
            &mut s,
            (i % side) as f64 * 0.7,
            (i / side) as f64 * 0.7,
            0.6,
        );
    }
    s
}

/// One ragdoll: a pinned pelvis, a spine up to a head, and four limbs -- shoulders and
/// hips as balls, elbows and knees as hinges with a range.
///
/// Pinned at the pelvis, which is the usual way to drive one of these: something outside
/// the solver owns where the body *is* -- a ballistic integrator that knows about the
/// ground, an animation, a vehicle seat -- and the skeleton hangs off it.
fn ragdoll(into: &mut Skeleton, x: f64, z: f64) -> usize {
    rig(into, x, z, 1.0, true)
}

/// The same rig with **nothing holding it up**, dropped from `y`.
///
/// The difference is not cosmetic and it is why there are two of these. A pinned pelvis
/// with four limbs hanging off it is an array of undamped pendulums: there is no damping
/// in a position solver, and limbs that never reach the ground never meet Coulomb either,
/// so nothing can take their energy away. Measured on the pinned fixture over thirty
/// seconds, the median body was moving at 3.4 m/s at the start and 3.6 m/s at the end --
/// it is a heap permanently arriving, and no amount of skipping settled work can show up
/// in it. Dropped rigs land on the plane, which is what dissipates.
fn corpse(into: &mut Skeleton, x: f64, z: f64, y: f64) -> usize {
    rig(into, x, z, y, false)
}

fn rig(into: &mut Skeleton, x: f64, z: f64, y: f64, anchored: bool) -> usize {
    let base = into.len();
    let pelvis = if anchored {
        into.add_body(Body::pinned((x, y, z)))
    } else {
        into.add_body(Body::capsule(8.0, 0.1, 0.16, (x, y, z)))
    };

    // Spine, neck, head.
    let mut up = pelvis;
    for i in 0..4 {
        let link = into.add_body(Body::capsule(
            6.0,
            0.08,
            0.2,
            (x, y + 0.2 + 0.2 * i as f64, z),
        ));
        into.add_joint(Joint::Ball {
            a: up,
            b: link,
            anchor_a: (0.0, 0.1, 0.0),
            anchor_b: (0.0, -0.1, 0.0),
        });
        up = link;
    }

    // Four limbs of three, hung off the chest and the pelvis.
    for limb in 0..4 {
        let (root, side) = if limb < 2 { (base + 3, 1.0) } else { (pelvis, -1.0) };
        let side_z = z + if limb % 2 == 0 { 0.15 } else { -0.15 };
        let mut previous = root;
        for segment in 0..3 {
            let body = into.add_body(Body::capsule(
                4.0,
                0.06,
                0.25,
                (x, y + side * 0.25 * segment as f64, side_z),
            ));
            // Shoulders and hips turn every way; elbows and knees do not.
            let joint = if segment == 0 {
                Joint::Ball {
                    a: previous,
                    b: body,
                    anchor_a: (0.0, 0.0, side_z - z),
                    anchor_b: (0.0, 0.125, 0.0),
                }
            } else {
                Joint::Hinge {
                    a: previous,
                    b: body,
                    anchor_a: (0.0, -0.125, 0.0),
                    anchor_b: (0.0, 0.125, 0.0),
                    axis_a: (1.0, 0.0, 0.0),
                    axis_b: (1.0, 0.0, 0.0),
                    min: -0.1,
                    max: 2.2,
                }
            };
            into.add_joint(joint);
            previous = body;
        }
    }
    into.len() - base
}

fn one_skeleton(c: &mut Criterion) {
    let mut group = c.benchmark_group("articulated/one");
    let mut s = Skeleton::new();
    let bones = ragdoll(&mut s, 0.0, 0.0);
    assert_eq!(bones, BONES, "the bench's rig is not the rig it documents");

    for iterations in [1usize, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("iterations", iterations),
            &iterations,
            |b, &iterations| {
                b.iter(|| {
                    s.step(black_box(DT), black_box(G), black_box(iterations));
                });
            },
        );
    }
    group.finish();
}

/// The pile, which is the number that decides anything.
///
/// One `Skeleton` holding all six hundred rather than six hundred of them, because that
/// is what lets the colouring find parallelism *across* skeletons as well as within one.
/// Six hundred separate `Skeleton`s would be six hundred serial solves of sixteen joints
/// each, which leaves a thread pool idle.
fn the_pile(c: &mut Criterion) {
    let mut group = c.benchmark_group("articulated/pile");
    group.sample_size(20);

    let mut s = Skeleton::new();
    for i in 0..PILE {
        ragdoll(&mut s, i as f64 * 0.8, 0.0);
    }
    let colours = s.colours().len();
    println!(
        "  pile: {} bodies, {} joints, {colours} colours",
        s.len(),
        s.joints().len(),
    );

    for iterations in [1usize, 8] {
        group.bench_with_input(
            BenchmarkId::new("iterations", iterations),
            &iterations,
            |b, &iterations| {
                b.iter(|| {
                    s.step(black_box(DT), black_box(G), black_box(iterations));
                });
            },
        );
    }
    group.finish();
}

/// **The same six hundred, packed close enough to touch, and never coming to rest.**
///
/// The rigs are pinned at the pelvis with four limbs hanging clear of the ground, so this
/// measures a solver working flat out on a set that will still be working flat out in
/// half an hour. That is worth measuring -- a solver has to be fast while things are
/// happening -- but it is not a heap, and it cannot be read as one. See
/// [`a_heap_arriving`] and [`a_heap_settled`] for rigs that actually land.
fn the_pendulums(c: &mut Criterion) {
    let mut group = c.benchmark_group("articulated/pendulums");
    group.sample_size(10);

    let mut s = Skeleton::new();
    s.set_ground((0.0, 1.0, 0.0), 0.0);
    let side = (PILE as f64).sqrt().ceil() as usize;
    for i in 0..PILE {
        ragdoll(
            &mut s,
            (i % side) as f64 * 0.35,
            (i / side) as f64 * 0.35,
        );
    }
    for _ in 0..30 {
        s.step(DT, G, 8);
    }
    println!(
        "  pendulums: {} bodies, {} joints, {} contacts",
        s.len(),
        s.joints().len(),
        s.contact_count(),
    );

    for iterations in [1usize, 8] {
        group.bench_with_input(
            BenchmarkId::new("iterations", iterations),
            &iterations,
            |b, &iterations| {
                b.iter(|| {
                    s.step(black_box(DT), black_box(G), black_box(iterations));
                });
            },
        );
    }
    group.finish();
}

/// Six hundred rigs **in the middle of hitting the ground and each other**.
///
/// The moving half of the pair, and the one the frame budget is actually about: bodies
/// arriving, contacts appearing and vanishing, the solve doing real work. Timed at the
/// point the heap is loudest rather than after it has quietened, so the number does not
/// depend on how long the fixture happened to be left running.
fn a_heap_arriving(c: &mut Criterion) {
    let mut group = c.benchmark_group("articulated/arriving");
    group.sample_size(10);

    let mut s = dropped(PILE);
    for _ in 0..ARRIVING {
        s.step(DT, G, 8);
    }
    println!(
        "  arriving: {} bodies, {} joints, {} contacts, {} of {} awake",
        s.len(),
        s.joints().len(),
        s.contact_count(),
        s.awake_count(),
        s.len(),
    );

    for iterations in [1usize, 8] {
        group.bench_with_input(
            BenchmarkId::new("iterations", iterations),
            &iterations,
            |b, &iterations| {
                b.iter(|| {
                    s.step(black_box(DT), black_box(G), black_box(iterations));
                });
            },
        );
    }
    group.finish();
}

/// **The same heap after thirty seconds**, which is as close to rest as a rig of this
/// solver's gets -- and not close enough for anything to fall asleep.
///
/// Named for what it measures rather than for what it was meant to. Every body is still
/// awake here, so the sleeping column is not a win, it is **the cost of asking**: the
/// settling test over ten thousand bodies and the live constraint lists, measured at five
/// to ten per cent. It is kept for exactly that, as the control on the feature's
/// overhead. What stops the heap settling is in the module header and is a defect in the
/// solve, not in this fixture; [`a_heap_that_has_settled`] is the one that shows what
/// sleeping is worth once a heap does come to rest.
fn a_heap_at_thirty_seconds(c: &mut Criterion) {
    let mut group = c.benchmark_group("articulated/thirty_seconds");
    group.sample_size(10);

    for sleeping in [false, true] {
        let mut s = dropped(PILE);
        s.set_sleeping(sleeping);
        for _ in 0..SETTLED {
            s.step(DT, G, 8);
        }
        println!(
            "  thirty seconds, sleeping {sleeping}: {} of {} awake, {} contacts",
            s.awake_count(),
            s.len(),
            s.contact_count(),
        );
        group.bench_with_input(BenchmarkId::new("sleeping", sleeping), &sleeping, |b, _| {
            b.iter(|| {
                s.step(black_box(DT), black_box(G), black_box(8));
            });
        });
    }
    group.finish();
}

/// **Ten thousand bodies that have actually come to rest**, which is the case sleeping
/// exists for and the only one that demonstrates it.
///
/// Stacks of three capsules, spread far enough apart to be their own islands. They reach
/// the point where nothing is awake at about step 700, and from there a step is a word
/// test per sixty-four bodies and nothing else. The sleeping-off column is the same heap
/// solved in full, for ever, which is what every settled workload cost before.
fn a_heap_that_has_settled(c: &mut Criterion) {
    let mut group = c.benchmark_group("articulated/has_settled");
    group.sample_size(10);

    for sleeping in [false, true] {
        let mut s = Skeleton::new();
        s.set_ground((0.0, 1.0, 0.0), 0.0);
        let side = (STACKS as f64).sqrt().ceil() as usize;
        for i in 0..STACKS {
            let (x, z) = ((i % side) as f64 * 1.5, (i / side) as f64 * 1.5);
            for k in 0..STACK_HIGH {
                let mut body =
                    Body::capsule(4.0, 0.1, 0.5, (x, 0.11 + 0.21 * k as f64, z));
                body.orientation = rs_physics::models::Quaternion::from_axis_angle(
                    (0.0, 0.0, 1.0),
                    -std::f64::consts::FRAC_PI_2,
                );
                s.add_body(body);
            }
        }
        s.set_sleeping(sleeping);
        for _ in 0..900 {
            s.step(DT, G, 8);
        }
        println!(
            "  has settled, sleeping {sleeping}: {} of {} awake",
            s.awake_count(),
            s.len(),
        );
        group.bench_with_input(BenchmarkId::new("sleeping", sleeping), &sleeping, |b, _| {
            b.iter(|| {
                s.step(black_box(DT), black_box(G), black_box(8));
            });
        });
    }
    group.finish();
}

/// **A rig arriving in a skeleton that already holds six hundred.**
///
/// The case no bench here could see, because every other one builds the skeleton once and
/// then steps it. A caller that spawns rigs as a scene fills does this on most frames, and
/// before the colouring was made incremental it paid a full pass over every joint in the
/// skeleton -- with a heap allocation per body -- on each of them.
fn joining(c: &mut Criterion) {
    let mut group = c.benchmark_group("articulated/joining");
    group.sample_size(20);

    let mut s = dropped(PILE);
    for _ in 0..30 {
        s.step(DT, G, 8);
    }

    let mut at = 0.0;
    group.bench_function("rig_then_step", |b| {
        b.iter(|| {
            at += 0.8;
            corpse(&mut s, black_box(at), 60.0, 1.0);
            s.step(black_box(DT), black_box(G), black_box(8));
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    one_skeleton,
    the_pile,
    the_pendulums,
    a_heap_arriving,
    a_heap_at_thirty_seconds,
    a_heap_that_has_settled,
    joining
);
criterion_main!(benches);
