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

/// One ragdoll: a pinned pelvis, a spine up to a head, and four limbs -- shoulders and
/// hips as balls, elbows and knees as hinges with a range.
///
/// Pinned at the pelvis, which is the usual way to drive one of these: something outside
/// the solver owns where the body *is* -- a ballistic integrator that knows about the
/// ground, an animation, a vehicle seat -- and the skeleton hangs off it.
fn ragdoll(into: &mut Skeleton, x: f64) -> usize {
    let base = into.len();
    let pelvis = into.add_body(Body::pinned((x, 1.0, 0.0)));

    // Spine, neck, head.
    let mut up = pelvis;
    for i in 0..4 {
        let link = into.add_body(Body::capsule(
            6.0,
            0.08,
            0.2,
            (x, 1.2 + 0.2 * i as f64, 0.0),
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
        let z = if limb % 2 == 0 { 0.15 } else { -0.15 };
        let mut previous = root;
        for segment in 0..3 {
            let body = into.add_body(Body::capsule(
                4.0,
                0.06,
                0.25,
                (x, 1.0 + side * 0.25 * segment as f64, z),
            ));
            // Shoulders and hips turn every way; elbows and knees do not.
            let joint = if segment == 0 {
                Joint::Ball {
                    a: previous,
                    b: body,
                    anchor_a: (0.0, 0.0, z),
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
    let bones = ragdoll(&mut s, 0.0);
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
        ragdoll(&mut s, i as f64 * 0.8);
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

criterion_group!(benches, one_skeleton, the_pile);
criterion_main!(benches);
