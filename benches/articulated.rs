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
//!
//! # Every sample starts from the same scene, and it used not to
//!
//! A `Skeleton` is mutated by `step`, and criterion times a routine by running it many
//! times over. So a fixture that has not settled is a **different scene on every sample**:
//! the first is timed at step 1 and the last some tens of thousands of steps later, and
//! what is being averaged is a trajectory rather than a workload. Two runs of the same
//! binary then disagree by however much the scene changed, and two runs of *different*
//! binaries disagree by that plus whatever the change did.
//!
//! Measured, that is not a rounding error. Three runs of one unmodified binary on
//! `one/8` gave 38.5, 59.6 and 60.5 us -- a spread of 57 per cent with nothing changing
//! at all -- and `pile/8` swung twofold on a single build while the commit it was being
//! compared against held steady to seven per cent, because the two builds' costs scale
//! differently with a contact count that was drifting between 4,200 and 6,600 under the
//! benchmark. A regression was reported off that and did not exist.
//!
//! So every fixture that has not come to rest is timed through [`steady`], which clones
//! the scene before each sample and times one step from the identical state. What that
//! costs is in [`steady`]'s own documentation, because it is not free and pretending it is
//! would be the same mistake one level down.
//!
//! # And every fixture says how many contacts it has
//!
//! `pile` printed its bodies, its joints and its colours and no contact count, and the
//! absence was read as a zero for most of this module's life -- it printed that line
//! before contacts existed here at all and nobody updated it when they arrived. It has
//! between four and seven thousand, they are the rigs' own limbs touching, and a change
//! that costs anything per contact shows up in it. A fixture that does not say what it
//! holds will eventually be quoted for something it is not, so they all say it now.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
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

/// How fast the joint-only fixture's bodies turn, in radians a second.
///
/// Enough that the joints carry a real load -- a limb at a quarter of a metre from the
/// axis sees a few times gravity at this rate -- and not so much that a rig tears through
/// its own hinge limits before the fixture has settled into its measurement.
const TUMBLE: f64 = 6.0;

/// How far apart the tumbling rigs stand. Wide enough that a spinning limb cannot reach
/// its neighbour: at the 0.8 m the other fixtures use, this one grows contacts and stops
/// being a control.
const TUMBLING_APART: f64 = 2.5;

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

/// Rows of capsules down the ploughed field.
///
/// **A sheet and not a lane.** It was a lane, and that was the broad phase's doing rather
/// than a taste: the grid's cell was twice the largest reach in the set, so a roller wide
/// enough to cover a broad field coarsened the cell for every small body in it, and the
/// step cost more in candidate pairs than in everything else together. A lane kept the
/// roller within twice a field body's reach, which was what the grid was sized for. The
/// grid now keeps a body that size out of itself and tests it directly, so the fixture no
/// longer has to be the shape the broad phase wanted. See [`roller`].
const FIELD_LONG: usize = 800;

/// Capsules across the field, side by side in each row.
///
/// Six, which with [`FIELD_ACROSS`] is a field four and a half metres wide: wide enough
/// that the grid's cell is coarse across the field as well as along it, which is where the
/// cost of sizing it for the roller actually lived. The roller spans the whole of it --
/// see [`roller`] -- so every body in the field is driven over and the retirement curve
/// below is the whole field rather than a strip out of the middle of it.
const FIELD_WIDE: usize = 6;

/// How far apart the field's capsules stand across the field.
///
/// A field capsule lies across the lane and is 0.7 m from end to end, so this is the same
/// five centimetres of clearance [`FIELD_ALONG`] leaves down it.
const FIELD_ACROSS: f64 = 0.75;

/// How far apart the field's capsules stand down the lane.
///
/// A capsule of radius 0.1 and segment 0.5 reaches 0.1 across its own axis, so this is
/// five centimetres of clearance: near enough that the field is a surface rather than
/// scattered bodies, and not touching at the moment it is built, since a spawn overlap is
/// a different thing to be measuring.
const FIELD_ALONG: f64 = 0.25;

/// Steps of settling before the roller arrives. The field is laid a centimetre above the
/// plane and has only to fall that far and stop.
const FIELD_SETTLE: usize = 120;

/// How fast the roller is driven, in metres a second.
///
/// A sixth of a metre a step, which is less than the roller's own diameter: any faster
/// and it would step over a body between one narrow phase and the next, and the fixture
/// would be measuring tunnelling rather than ploughing.
const ROLLER_SPEED: f64 = 10.0;

/// Steps of the drive at which the fixture is timed.
///
/// The first is thirty rather than zero so that the roller is already inside the field and
/// the four figures are four points on one curve; at zero it is still outside, and a step
/// that is only the settled field would be a different fixture. The last leaves a quarter
/// of the lane in front of it.
const PLOUGHED: [usize; 4] = [30, 330, 630, 930];

/// **The load at which this bench's caller decides a body has been crushed**, in newtons.
///
/// Twenty times a field body's own weight. It is here and not in the crate on purpose:
/// `rs_physics` reports the load and holds no threshold, no material strength and nothing
/// resembling one, because what load breaks a body depends entirely on what the bodies
/// are taken to be. A bench is a caller, so a bench may have an opinion.
const CRUSHED: f64 = 20.0 * 8.0 * 9.80665;

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
/// it is a heap permanently arriving. Dropped rigs land on the plane, which is what
/// dissipates.
///
/// **The part of a pinned rig that does stop is its legs**, which hang straight down off
/// the anchor with nothing swinging them, and they are their own island because the anchor
/// does not join one. Six of a rig's sixteen movable bodies leave the step inside half a
/// second; the spine and arms go on for ever. So skipping settled work does show up here
/// after all -- see [`joints_only`] -- just not in the median body.
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

/// **Time one step from the same state every sample**, by cloning the scene first.
///
/// The scene a bench builds is the scene every sample sees. Without this, sample `n` is
/// timed on the state sample `n - 1` left behind, which for anything that has not settled
/// means the measurement is an average over a trajectory and is not reproducible between
/// runs. See the module header for what that cost the session this was written in.
///
/// # What it costs, which is nothing, and that was measured rather than assumed
///
/// The obvious objection is that copying a few megabytes before each batch evicts the
/// caches the step is about to read, so the step is timed cold and the clone is being
/// measured after all. It is not. [`joints_only`] is the control, because it is the one
/// fixture that does *not* drift -- no contacts, so nothing about it changes under
/// stepping -- which makes in situ and cloned the same scene and the only difference the
/// clone. Same binary, ten thousand two hundred bodies:
///
/// ```text
///   joints_only/8   in situ   2.657 ms      cloned   2.652 .. 2.753 ms
/// ```
///
/// So the clone is free of the measurement, and every level shift this change produced
/// elsewhere is the **scene** rather than the harness. That matters for reading the
/// numbers: cloned figures are not comparable with anything quoted before this existed,
/// not because the harness got slower but because the old ones were averages over a
/// trajectory and these are a named state.
///
/// What it buys is the spread. Same binary, run to run:
///
/// ```text
///                   in situ                      cloned
///   one/8           38.5 .. 60.5 us   (57%)      108.5 .. 110.9 us   (2%)
///   pile/8          3.90 .. 7.86 ms  (100%)      7.39 .. 8.19 ms    (11%)
/// ```
///
/// The levels moved because the named state is a denser moment than the drifted ones the
/// old samples wandered into -- `pile` at step 30 carries 8,400 contacts, more than any
/// moment the drift was caught at -- and the contacts are most of what those two fixtures
/// cost.
///
/// A settled scene does not need this -- it stays settled, so its samples are already the
/// same scene -- and paying a ten-thousand-body clone to time a step that costs nothing
/// would be measuring the clone. [`a_heap_that_has_settled`] is left in place.
fn steady(b: &mut criterion::Bencher<'_>, scene: &Skeleton, mut one_step: impl FnMut(&mut Skeleton)) {
    b.iter_batched_ref(
        || scene.clone(),
        |s| one_step(s),
        BatchSize::NumIterations(batch_of(scene.len()) as u64),
    );
}

/// How many clones of a scene of this many bodies may be alive while one batch is timed.
///
/// Criterion times a *batch* of iterations together, so a batch larger than one amortises
/// the per-sample timer and loop overhead over it. That matters: at one clone per
/// iteration a seventeen-body rig's step measured 108 us against the 60 it costs, because
/// the overhead of starting and stopping a measurement is a real fraction of forty
/// microseconds. It is nothing against three milliseconds.
///
/// What bounds the batch is memory, so that is what sets it. A clone is about
/// [`BYTES_A_BODY`] per body and a batch has to fit somewhere without paging or
/// evicting more than the step itself streams; sixteen megabytes is the budget. A
/// seventeen-body rig therefore gets the cap and a ten-thousand-body heap gets one clone
/// at a time, which is the right answer at both ends for the same reason.
fn batch_of(bodies: usize) -> usize {
    const BUDGET: usize = 16 << 20;
    /// A body's share of a cloned skeleton, near enough: ten parallel arrays of a vector,
    /// a quaternion or a scalar each, plus its share of the contact and colour buffers.
    /// An estimate, and it only has to be the right order -- it chooses a batch size, not
    /// a result.
    const BYTES_A_BODY: usize = 320;
    (BUDGET / (bodies.max(1) * BYTES_A_BODY)).clamp(1, 64)
}

fn one_skeleton(c: &mut Criterion) {
    let mut group = c.benchmark_group("articulated/one");
    let mut s = Skeleton::new();
    let bones = ragdoll(&mut s, 0.0, 0.0);
    assert_eq!(bones, BONES, "the bench's rig is not the rig it documents");
    // Far enough in that the rig hangs off its pinned pelvis rather than sitting in the
    // pose it was authored in, and the contacts below are the ones it actually carries.
    for _ in 0..30 {
        s.step(DT, G, 8);
    }
    // **This rig is not joint-only either.** Its limb segments are authored overlapping,
    // self-collision is on by default, and the contacts that makes are solved on every
    // pass like any others. See the module header.
    println!(
        "  one: {} bodies, {} joints, {} contacts",
        s.len(),
        s.joints().len(),
        s.contact_count(),
    );

    for iterations in [1usize, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("iterations", iterations),
            &iterations,
            |b, &iterations| {
                steady(b, &s, |s| {
                    s.step(black_box(DT), black_box(G), black_box(iterations));
                });
            },
        );
    }
    group.finish();
}

/// **The one fixture here that really is only joints**, which is what `pile` was believed
/// to be for most of this module's life.
///
/// Six hundred rigs with self-collision off and no ground: nine thousand six hundred joints,
/// no contacts of any kind, and no contact list to build or colour. It is a real workload --
/// a crowd of corpses driven by an animation that owns where they are, a solver asked for
/// the articulation and nothing else -- and it is the measurement that says which half of a
/// step a change lands in, because anything that costs per contact costs exactly nothing
/// here.
///
/// **Six thousand of the ten thousand two hundred are awake by the time it is timed**, and
/// that is not a flaw in the fixture, it is the pinned pelvis doing what a pinned body does
/// to islands. An anchor does not join a component, so each rig is not one island but three
/// -- the spine with the arms hanging off it, and a leg each -- and the two legs hang
/// straight down from the anchor and come to rest inside the warm-up while the spine and
/// arms are still swinging. Six of a rig's sixteen movable bodies then leave the step.
/// Before `Skeleton::settle` stopped reading "jointed to something that is not ready" as
/// "jointed to something that is still moving", none of them could: everything hung off an
/// anchor was disqualified for ever, and this fixture measured 2.67 to 2.88 ms where it now
/// measures 2.05 to 2.19.
fn joints_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("articulated/joints_only");
    group.sample_size(20);

    let mut s = Skeleton::new();
    s.set_self_collision(false);
    // **Tumbling, and nothing is pinned.**
    //
    // A control has to hold everything else still while one thing changes, and two
    // earlier versions of this fixture could not. Pinned, a rig's legs hang straight down
    // with nothing to swing them and six of its sixteen movable bodies leave the step
    // within half a second, so the scene ran four tenths asleep and its number moved when
    // the *sleeping* changed rather than the joint solve -- which duly happened, and a
    // sleeping fix was briefly read as a gain in the joint path. Forcing sleeping off
    // fixed the reading and left a control that disables the feature it is meant to be
    // neutral about.
    //
    // Falling free fixes both and needs neither. A tumbling rig is awake because it is
    // moving, not because it was told to be, and **its joints are loaded by its own
    // rotation**: in uniform gravity every body falls at the same rate and a joint has
    // nothing to hold, but spin puts every limb under centripetal acceleration and the
    // joints carry it. Measured, the cost still scales with the iteration count -- 1.80 ms
    // at one pass against 4.11 at eight -- which is what says the joints are doing work
    // rather than early-returning on a satisfied constraint.
    //
    // The spacing is what keeps it honest: at the usual 0.8 m a spinning limb reaches its
    // neighbour and the fixture quietly grows a hundred and eighty contacts. Far enough
    // apart, the assert below holds and this measures nine thousand six hundred joints and
    // nothing else.
    for i in 0..PILE {
        corpse(&mut s, (i % 25) as f64 * TUMBLING_APART, (i / 25) as f64 * TUMBLING_APART, 1.0);
    }
    for i in 0..s.len() {
        let n = i as f64;
        s.set_angular_velocity(
            i,
            (
                TUMBLE * (n * 0.7).sin(),
                TUMBLE * (n * 1.1).cos(),
                TUMBLE * (n * 0.3).sin(),
            ),
        );
    }
    for _ in 0..30 {
        s.step(DT, G, 8);
    }
    assert_eq!(
        s.contact_count(),
        0,
        "the joint-only fixture found contacts, so it is not measuring what it says",
    );
    println!(
        "  joints only: {} bodies, {} joints, {} contacts, {} colours, {} of {} awake, \
         nothing pinned",
        s.len(),
        s.joints().len(),
        s.contact_count(),
        s.colours().len(),
        s.awake_count(),
        s.len(),
    );

    for iterations in [1usize, 8] {
        group.bench_with_input(
            BenchmarkId::new("iterations", iterations),
            &iterations,
            |b, &iterations| {
                steady(b, &s, |s| {
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
///
/// **It is not a joints-only fixture and it never was, whatever its name suggests.** The
/// rigs are far enough apart not to touch each other, but each one's limb segments are
/// authored overlapping and self-collision is on, so it carries between four and seven
/// thousand contacts and they are solved on every pass. The count printed below is there
/// because its absence was read as a zero for most of this module's life, and a change
/// that costs anything per contact was then judged against a workload nobody realised had
/// any. [`joints_only`] is the fixture this was mistaken for.
///
/// The name is kept because the solver's module header quotes `pile/8` throughout and a
/// rename would silently break every one of those references. What it holds is stated
/// instead.
fn the_pile(c: &mut Criterion) {
    let mut group = c.benchmark_group("articulated/pile");
    group.sample_size(20);

    let mut s = Skeleton::new();
    for i in 0..PILE {
        ragdoll(&mut s, i as f64 * 0.8, 0.0);
    }
    // A defined moment rather than step zero, so the contact count printed below is the
    // one every sample is timed on. Before [`steady`], samples were drawn from wherever
    // the scene had drifted to and the count ranged from 4,200 to 6,600 within one run.
    for _ in 0..30 {
        s.step(DT, G, 8);
    }
    println!(
        "  pile: {} bodies, {} joints, {} contacts, {} colours",
        s.len(),
        s.joints().len(),
        s.contact_count(),
        s.colours().len(),
    );

    for iterations in [1usize, 8] {
        group.bench_with_input(
            BenchmarkId::new("iterations", iterations),
            &iterations,
            |b, &iterations| {
                steady(b, &s, |s| {
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
                steady(b, &s, |s| {
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
                steady(b, &s, |s| {
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
            steady(b, &s, |s| {
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
///
/// **This is the one fixture that steps in place**, and it is the one that may: a settled
/// scene stays settled, so every sample is already the same scene and [`steady`] would buy
/// nothing. It would cost a great deal -- a ten-thousand-body clone before each sample, to
/// time a step whose whole claim is that it costs nothing measurable -- and the
/// measurement would be of the clone. The sleeping-off column does keep moving, slowly,
/// and is read as the control on the feature's overhead rather than as a number in its own
/// right.
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
    println!(
        "  joining: {} bodies, {} joints, {} contacts before the rig arrives",
        s.len(),
        s.joints().len(),
        s.contact_count(),
    );

    // **This one needed the clone most of all.** Stepping in place, it added a rig per
    // sample and never took one away, so the skeleton grew without bound *during* the
    // measurement: the last sample was timed on a scene tens of thousands of bodies larger
    // than the first, and the answer depended on how many samples criterion chose to take.
    // What it is meant to measure is one arrival into a six-hundred-rig scene, which is
    // what it measures now.
    group.bench_function("rig_then_step", |b| {
        steady(b, &s, |s| {
            corpse(s, black_box(480.0), 60.0, 1.0);
            s.step(black_box(DT), black_box(G), black_box(8));
        });
    });
    group.finish();
}


/// **A settled field with a heavy body driven through it, retiring what it crushes.**
///
/// The fixture for `Skeleton::retire`, and the number it exists to show is the one that
/// *falls*: destruction here is subtraction, so a field driven through has fewer bodies
/// behind the roller than in front of it and the step gets cheaper as it goes. Every
/// other fixture in this file measures a scene that costs what it costs; this one is the
/// only place the shape of that claim can be seen.
///
/// **Sleeping is off, and that is what makes the number mean what it says.** With it on, a
/// settled field is already out of the step and what a step costs is the awake
/// neighbourhood around the roller, which travels with the roller and stays about the same
/// size -- so the cost would be flat and would say nothing about how many bodies are left.
/// With it off, every live body is solved every step, so the step cost is a measurement of
/// the live body count and the fall is the thing being claimed.
///
/// The rule for what counts as crushed is [`CRUSHED`], and it is **here rather than in the
/// crate** on purpose: what load breaks a body is a material judgement and `rs_physics`
/// holds no threshold anywhere.
///
/// # What it shows
///
/// Medians of four runs, since one run of anything on this machine says nothing:
///
/// ```text
///   steps driven        30      330      630      930
///   bodies still live  784      585      387      189
///   a step            3.24 ms  2.12 ms  1.71 ms  0.89 ms
///   per live body     4.13 us  3.62 us  4.42 us  4.70 us
/// ```
///
/// **The cost falls by a factor of 3.6 while the drive is going on**, and the bottom row
/// is why: the cost per live body is flat to within the spread, so what the step is paying
/// for is the bodies that are left and there are fewer of them every step. That is the
/// whole claim of retirement being subtraction, and it is the one number in this file that
/// goes down as a fixture runs.
fn ploughing(c: &mut Criterion) {
    let mut group = c.benchmark_group("articulated/ploughing");
    group.sample_size(20);

    let mut s = field();
    for _ in 0..FIELD_SETTLE {
        s.step(DT, G, 8);
    }
    println!(
        "  ploughing: {} bodies, {} candidate pairs, {} contacts in the settled field",
        s.len(),
        s.candidate_pairs(),
        s.contact_count(),
    );
    let roller = s.add_body(roller());

    let mut driven = 0usize;
    for upto in PLOUGHED {
        while driven < upto {
            plough(&mut s, roller);
            driven += 1;
        }
        let live = (0..s.len()).filter(|&i| !s.is_retired(i)).count();
        println!(
            "  ploughing after {upto} steps: {live} of {} bodies live, {} candidate pairs, \
             {} contacts, roller at x {:.1}",
            s.len(),
            s.candidate_pairs(),
            s.contact_count(),
            s.position(roller).0,
        );
        group.bench_with_input(BenchmarkId::new("driven", upto), &upto, |b, _| {
            steady(b, &s, |s| plough(s, black_box(roller)));
        });
    }
    group.finish();
}

/// One step of the drive: hold the roller's speed, step, and retire whatever the step
/// says has been crushed.
///
/// The scan is a pass over every body, which is what a caller doing this pays and so
/// belongs inside the measurement. It is also why retiring has to be cheap to *ask* about:
/// `Skeleton::normal_load` is a load from an array and `Skeleton::is_retired` a bit.
fn plough(s: &mut Skeleton, roller: usize) {
    s.set_velocity(roller, (ROLLER_SPEED, s.velocity(roller).1, 0.0));
    s.step(DT, G, 8);
    for i in 0..s.len() {
        if i != roller && !s.is_retired(i) && s.normal_load(i) > CRUSHED {
            s.retire(i);
        }
    }
}

/// A field of capsules lying flat on the plane, each across the line of the drive and near
/// enough to its neighbours to be a surface rather than scattered bodies.
fn field() -> Skeleton {
    let mut s = Skeleton::new();
    s.set_ground((0.0, 1.0, 0.0), 0.0);
    s.set_sleeping(false);
    for column in 0..FIELD_LONG {
        for row in 0..FIELD_WIDE {
            // Centred on the line the roller is driven down, so the roller meets the
            // whole width of the field at once.
            let across = (row as f64 - (FIELD_WIDE - 1) as f64 * 0.5) * FIELD_ACROSS;
            let mut body = Body::capsule(
                8.0,
                0.1,
                0.5,
                (column as f64 * FIELD_ALONG, 0.11, across),
            );
            // Lying flat with its axis across the drive, so the roller meets each one
            // side on.
            body.orientation = rs_physics::models::Quaternion::from_axis_angle(
                (1.0, 0.0, 0.0),
                std::f64::consts::FRAC_PI_2,
            );
            s.add_body(body);
        }
    }
    s
}

/// The heavy body that is driven through it: a dense roller lying across the field, long
/// enough to span the whole width of it.
///
/// **Six times a field body's reach**, which used to be a thing this fixture could not
/// have. The grid's cell was twice the largest reach in the set, so a roller this long
/// coarsened the cell for every small body in it, and the fixture was cut down to a lane
/// and a roller half this size to keep the broad phase honest. It is now kept out of the
/// grid and tested against it directly, so the roller may be the size the field wants.
/// See [`FIELD_LONG`].
///
/// Its mass is the same steel it always was -- two and a half thousand kilogrammes a
/// metre of length -- so the load it puts on a body under it, which is what [`CRUSHED`]
/// is measured against, has not moved with the geometry.
fn roller() -> Body {
    let mut body = Body::capsule(10_000.0, 0.25, 4.0, (-1.0, 0.25, 0.0));
    body.orientation = rs_physics::models::Quaternion::from_axis_angle(
        (1.0, 0.0, 0.0),
        std::f64::consts::FRAC_PI_2,
    );
    body.velocity = (ROLLER_SPEED, 0.0, 0.0);
    body
}

criterion_group!(
    benches,
    one_skeleton,
    joints_only,
    the_pile,
    the_pendulums,
    a_heap_arriving,
    a_heap_at_thirty_seconds,
    a_heap_that_has_settled,
    joining,
    ploughing
);
criterion_main!(benches);
