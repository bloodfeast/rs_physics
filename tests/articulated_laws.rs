//! The laws the articulated solver obeys, stated against the public API only.
//!
//! # What this file is for
//!
//! The unit tests inside the module check that each piece does what it was written to do.
//! This file checks the things that must remain true **however it is written**: that the
//! answer does not depend on how many threads computed it, that a coefficient means what
//! it says, that energy is not created, that a number does not move because somebody made
//! the solver faster.
//!
//! It exists because optimisation is where physics quietly stops being physics. A change
//! that makes a step twice as fast and shifts the friction angle by five degrees will pass
//! every timing check ever written and fail nothing else. So the properties here are
//! deliberately stated as laws with units and derivations rather than as recorded outputs:
//! each one either follows from the physics or from an invariant the implementation
//! promises, and a failure means the promise was broken rather than that a golden value
//! needs updating.
//!
//! Nothing here reaches inside the module. If a law cannot be expressed through the public
//! API then the public API is missing something, which is worth knowing on its own.

use rs_physics::articulated::{Body, Joint, Skeleton};

const DT: f64 = 1.0 / 60.0;
const G: (f64, f64, f64) = (0.0, -9.80665, 0.0);

// -- determinism ------------------------------------------------------------------

/// **The same simulation twice gives bit-identical answers.**
///
/// Not approximately equal: identical. Floating point addition is not associative, so any
/// accumulation whose order depends on how work happened to be distributed will drift, and
/// the drift compounds step after step. A solver that is only *usually* reproducible
/// cannot be debugged, cannot be regression-tested, and cannot be trusted to have got the
/// same answer as the run somebody reported.
#[test]
fn the_same_simulation_twice_is_bit_identical() {
    let first = settled_heap_state(300);
    let second = settled_heap_state(300);
    assert_eq!(
        first.len(),
        second.len(),
        "the two runs did not even produce the same number of bodies",
    );
    for (i, (a, b)) in first.iter().zip(second.iter()).enumerate() {
        assert_eq!(
            a, b,
            "body {i} ended at {a:?} on the first run and {b:?} on the second; something \
             in the step depends on more than its inputs",
        );
    }
}

/// **And the answer does not depend on how many threads computed it.**
///
/// This is the law that parallel work breaks first and most quietly. A correction applied
/// in a different order, a reduction combined in a different sequence, a work-stealing
/// split that lands differently under load: each gives an answer that is just as valid
/// numerically and is not the same answer.
///
/// It is also the sharpest test there is for unsafe parallel writes. Scattering into
/// shared arrays is sound only if the indices really are disjoint; if they are not, the
/// result depends on which thread arrived first, and that shows up here as a divergence
/// long before it shows up as a crash.
///
/// **Which is why the heap has to be big enough to actually go parallel, and why the law
/// now asserts that it did.** It used to run forty bodies, whose widest pass is around two
/// hundred constraints against a pass floor of five hundred and twelve: every one of the
/// three thread counts took the serial path, so this compared single-threaded output with
/// single-threaded output and could not have failed. A fixture can drift under a floor
/// without anybody noticing, and no care taken in the test can see it, because whether the
/// parallel path ran is not visible from outside -- so `Skeleton::solved_in_parallel`
/// exists and is checked here.
#[test]
fn the_answer_does_not_depend_on_how_many_threads_ran_it() {
    let counts = [1usize, 2, 8];
    let mut answers = Vec::new();
    for threads in counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a thread pool");
        answers.push(pool.install(|| settled_heap_state(300)));
    }

    for (n, other) in answers.iter().enumerate().skip(1) {
        for (i, (a, b)) in answers[0].iter().zip(other.iter()).enumerate() {
            assert_eq!(
                a, b,
                "on one thread body {i} ended at {a:?}, on {} threads it ended at {b:?}; \
                 the result depends on the schedule",
                counts[n],
            );
        }
    }
}

// -- the coefficients mean what they say ------------------------------------------

/// **Friction holds a slope, and lets go past `atan(mu)`.**
///
/// Only travel *down* the slope counts, because only sliding is Coulomb's business. A
/// capsule is a cylinder and will also roll, which is a different law with a different
/// coefficient; the fixture lays the body along the fall line so rolling carries it
/// sideways and sliding carries it down, then measures the component that answers the
/// question being asked.
///
/// # Held to the nanometre, at every angle inside the limit
///
/// Anywhere below `atan(mu)` the body is held still to within a nanometre over a minute,
/// which is arithmetic and not motion. Measured over the whole grid this test walks --
/// four coefficients, five fractions of the limit, four iteration counts, out to
/// sixty-four seconds -- the worst downhill travel is 1.5e-9 m, and what remains is
/// slightly *uphill* at about a micrometre a minute, so it is a residual and not a slide.
///
/// It was not always so, and the bound is written this tight deliberately so that
/// slackening it is a decision somebody has to make. A previous version crept downhill at
/// a steady 19 mm a second at three quarters of the limit with `mu` of 0.2 -- 0.019 m in
/// one second, 0.077 in four, 0.306 in sixteen, 1.226 in sixty-four, linear in time and
/// identical at four iterations and at thirty-two, so a genuine steady state rather than
/// a convergence shortfall. Two things caused it and both are worth knowing, because
/// either one coming back would look like this again:
///
/// * Coulomb's limit was carried across the solver's passes as a running total of
///   *magnitude* rather than as a cone on the resultant. The friction direction genuinely
///   reverses between passes, and charging both directions against one total spent the
///   coefficient to produce no net impulse: measured, the whole budget was consumed
///   while the resultant was only three quarters of it.
/// * The friction impulse was applied at the contact point with its full lever arm, so it
///   tipped the body forward over the contact. The normal constraints repair that tipping
///   but they can only push, so a resting body ratcheted itself clear of the plane, its
///   contacts stopped reporting any depth, and friction stopped acting while the slide it
///   was holding was still there.
///
/// The second is the one that makes a *patch* different from a point: two ends of a
/// capsule on a plane can shift load between them and cancel the tipping couple between
/// themselves, and a single point cannot. Which is why a capsule stood on one end still
/// topples and a capsule lying across the slide direction still rolls -- both have a patch
/// of no reach in the direction that matters.
#[test]
fn friction_holds_a_slope_and_lets_go_past_coulombs_angle() {
    // Ten nanometres, which is not a tolerance chosen to pass: the worst travel measured
    // anywhere on this grid is 1.5e-9 m over sixty-four seconds, and a body that has moved
    // a nanometre in a minute has not moved. It is an order of magnitude of headroom over
    // arithmetic and seven orders below the creep this replaced.
    const STILL: f64 = 1e-8;

    for mu in [0.2_f64, 0.3, 0.5, 0.8] {
        let limit = mu.atan().to_degrees();
        for iterations in [4usize, 8, 16, 32] {
            // Comfortably inside the limit, where it is held.
            let inside = slid_down((mu * 0.25).atan().to_degrees(), mu, iterations, 4.0);
            assert!(
                inside < STILL,
                "with mu {mu} at a quarter of the limit the body should not move at all, \
                 and at {iterations} iterations it moved {inside:e} m",
            );

            // And near the limit, which is where the old version gave way: four seconds
            // at the rate it used to creep would be 0.077 m.
            let near = slid_down((mu * 0.9).atan().to_degrees(), mu, iterations, 4.0);
            assert!(
                near < STILL,
                "with mu {mu} at nine tenths of the limit the body moved {near:e} m in \
                 four seconds at {iterations} iterations; friction is holding it at all \
                 the shallower angles and letting go here, which is the rotational creep \
                 coming back",
            );

            // And past the limit it genuinely runs.
            let outside = slid_down(limit + 6.0, mu, iterations, 4.0);
            assert!(
                outside > 0.5,
                "with mu {mu} the slope should let go above {limit:.1} degrees, but at \
                 {:.1} degrees and {iterations} iterations it travelled only {outside:.3} m",
                limit + 6.0,
            );
        }
    }
}

/// **And it is still held a minute later.** Separately from the angles, the hold must not
/// decay with time: a slow leak at the contact is invisible in four seconds and obvious in
/// a minute, and it is exactly what the rotational creep looked like before it was fixed.
///
/// Stated as travel that does not grow with the time it is given rather than as a rate,
/// because a rate divides by a number that is nearly zero. Sixteen times the duration must
/// not give sixteen times the travel; it must give the same nothing.
#[test]
fn a_held_slope_stays_held_however_long_it_is_watched() {
    const STILL: f64 = 1e-8;
    let mu = 0.2_f64;
    let angle = (mu * 0.75).atan().to_degrees();
    for seconds in [4.0, 16.0, 64.0] {
        let travelled = slid_down(angle, mu, 8, seconds);
        assert!(
            travelled < STILL,
            "at three quarters of the limit the body travelled {travelled:e} m in \
             {seconds} seconds; at four seconds it travelled {:e}, so this is a leak that \
             grows with time rather than arithmetic that does not",
            slid_down(angle, mu, 8, 4.0),
        );
    }
}

/// **Coulomb friction does not depend on mass.** The friction force rises with the normal
/// force, and so does the weight driving the slide, so the two cancel exactly and a heavy
/// body and a light one let go at the same angle.
///
/// A solver that gets this wrong has the coefficient entangled with something it should
/// not be -- an inertia, an impulse that was not divided by the right mass -- and the
/// symptom in a pile is that heavy bodies behave like they are on ice.
#[test]
fn the_friction_angle_does_not_depend_on_mass() {
    let mu = 0.5_f64;
    let limit = mu.atan().to_degrees();
    for mass in [0.5, 4.0, 400.0] {
        let inside = slid_down_with_mass((mu * 0.25).atan().to_degrees(), mu, 8, 4.0, mass);
        let outside = slid_down_with_mass(limit + 6.0, mu, 8, 4.0, mass);
        assert!(
            inside < 1e-8,
            "a {mass} kg body should be held below {limit:.1} degrees; it moved \
             {inside:e} m",
        );
        assert!(
            outside > 0.5,
            "a {mass} kg body should slide above {limit:.1} degrees; it travelled \
             {outside:.3} m",
        );
    }
}

/// **A resting capsule sits exactly its own radius above the ground, at any size and any
/// mass.** The contact condition is geometric: the axis of a capsule touching a plane is
/// one radius from it. Nothing about the body's weight or the solver's effort enters into
/// where that is.
///
/// Mass is the interesting half. A solver with too few iterations lets a heavy body sink
/// further than a light one, because the constraint is only approximately satisfied and
/// the error grows with the load, so this doubles as a convergence check that states its
/// tolerance in metres rather than in iterations.
#[test]
fn a_resting_capsule_sits_one_radius_above_the_ground() {
    for radius in [0.02, 0.1, 0.5] {
        for mass in [0.5, 4.0, 400.0] {
            let mut s = Skeleton::new();
            s.set_ground((0.0, 1.0, 0.0), 0.0);
            let body = s.add_body(lying(Body::capsule(mass, radius, radius * 5.0, (0.0, 1.0, 0.0))));
            for _ in 0..600 {
                s.step(DT, G, 8);
            }
            let y = s.position(body).1;
            // The tolerance is a length the physics supplies rather than one picked to
            // pass: `g * dt^2` is how far gravity pulls a body in a single step, which is
            // the finest grain at which a contact discovered once per step can hold
            // anything. Resting within that is exact; outside it means the contact is
            // losing rather than merely discretising.
            let grain = 9.80665 * DT * DT;
            assert!(
                (y - radius).abs() < grain,
                "a {mass} kg capsule of radius {radius} should rest at {radius}; it rests \
                 at {y:.6}, which is {:.3} mm out against a {:.3} mm step grain",
                (y - radius).abs() * 1000.0,
                grain * 1000.0,
            );
        }
    }
}

// -- energy -----------------------------------------------------------------------

/// **A closed system never ends with more energy than it started with.**
///
/// Gravity does work, contacts and friction take energy out, and nothing in a skeleton
/// puts any in. So total mechanical energy is non-increasing, and a solver that violates
/// it is feeding itself -- the failure mode that ends with a limb at an absurd distance,
/// and the one that is invisible until it is catastrophic.
///
/// Stated against the starting energy rather than step by step, because a position-based
/// solver legitimately exchanges energy between its position and velocity views within a
/// step; what it may not do is come out ahead.
#[test]
fn a_closed_system_never_gains_energy() {
    let mut s = Skeleton::new();
    let root = s.add_body(Body::pinned((0.0, 3.0, 0.0)));
    let mut previous = root;
    for i in 0..8 {
        // Half a link below the anchor above it, so the chain starts where its own joints
        // put it. Built a link too low instead, the first solve hauls the whole chain up
        // and the work it does reads as energy from nowhere -- which is the constraint
        // doing its job, and would have been blamed on the solver.
        let link = s.add_body(Body::capsule(
            3.0,
            0.05,
            0.35,
            (0.0, 3.0 - 0.2 - 0.4 * i as f64, 0.0),
        ));
        assert!(s.add_joint(Joint::free_ball(
    previous,
    link,
    if i == 0 { (0.0, 0.0, 0.0) } else { (0.0, -0.2, 0.0) },
    (0.0, 0.2, 0.0),
)));
        previous = link;
    }
    // Shoved once, so there is real energy to account for rather than a chain at rest.
    s.set_velocity(previous, (6.0, 0.0, 2.0));

    let start = energy(&s);
    let mut worst = start;
    for _ in 0..1800 {
        s.step(DT, G, 8);
        worst = worst.max(energy(&s));
    }

    // A tenth of a percent of headroom for the exchange a position-based step makes
    // within itself. Measured, a chain built consistently holds its energy to six
    // significant figures once it has come to rest, so this is loose rather than tight.
    assert!(
        worst <= start * 1.001 + 1e-6,
        "the chain reached {worst:.4} J having started with {start:.4} J; the solver is \
         putting energy in",
    );
}

/// **An overlap that was already there is a correction, not a launch.**
///
/// A position-based solver reads velocity back out of how far a body moved, so lifting one
/// out of an overlap it began the step inside is indistinguishable from it having
/// travelled that far under its own power unless the two are separated deliberately. The
/// visible symptom is a heap that detonates on the frame it is created.
#[test]
fn a_body_spawned_inside_the_ground_is_not_launched_out_of_it() {
    for depth in [0.05, 0.2, 1.0] {
        let mut s = Skeleton::new();
        s.set_ground((0.0, 1.0, 0.0), 0.0);
        let r = 0.1;
        let body = s.add_body(lying(Body::capsule(4.0, r, 0.5, (0.0, -depth, 0.0))));

        let mut fastest: f64 = 0.0;
        for _ in 0..300 {
            s.step(DT, G, 8);
            fastest = fastest.max(speed(s.velocity(body)));
        }
        // Falling freely for the time it takes to recover even the deepest of these would
        // not reach 3 m/s, so anything above that is energy the solver invented.
        assert!(
            fastest < 3.0,
            "a body spawned {depth} m inside the ground left at {fastest:.2} m/s",
        );
        let y = s.position(body).1;
        assert!(
            (y - r).abs() < 1e-3,
            "and it should still end up resting at {r}; it is at {y:.5}",
        );
    }
}

/// **A settled pile wanders; it does not drift.** A known limitation, measured rather
/// than hidden, and the bound is the one the solver currently meets.
///
/// # What is measured and why it is a ratio
///
/// Watch a settled pile for thirty-two times as long and a body that is *drifting* has
/// gone thirty-two times as far, while a body that is only jostling against its
/// neighbours has not. No absolute bound tells those apart -- loose enough for the
/// jostling and it admits a slow drift, tight enough to exclude the drift and the
/// jostling fails it. So this asks how the distance grows with the window: 15 steps
/// against 480.
///
/// The **surface** and not the centre, because a capsule turning on the spot has moved
/// against whatever it rests on exactly as much as one that slid, and in this pile the
/// turning is the larger half. So the centre's travel, plus the angle turned times the
/// reach, except about the capsule's own long axis where it is times the radius: a
/// surface of revolution spinning about its own axis has not gone anywhere, and charging
/// that at the reach makes a resting pile look far worse than it is.
///
/// # A ratio alone cannot say it, and asking for one was a mistake
///
/// The first version of this asked only that the ratio be sub-linear, and that punishes
/// an improvement: a change that took the fifteen-step drift down by three and a half and
/// the four-hundred-and-eighty-step drift down by a factor of one and a half made the
/// *ratio* worse and failed, for reducing the jostle faster than the drift. Taken to the
/// end, a pile that has stopped almost dead has a short window of nearly nothing and a
/// ratio of nearly anything.
///
/// So there are two ways to pass and a pile needs either: **sub-linear growth, or a long
/// window it has barely moved in.** The second bound is not a tolerance either. A body
/// that is entitled to go to sleep is one whose surface moves less than
/// `STILL_FRACTION` of its own reach over a settling window; a pile of such bodies
/// jostling in place, each window independent of the last, covers `STILL_FRACTION *
/// sqrt(windows)` over many of them, where one that is really travelling covers
/// `STILL_FRACTION * windows`. The long window here is forty settling windows, so the
/// random walk allows about 0.126 of a reach and the drift allows 0.8. Below the first,
/// the pile is doing no worse than a heap of bodies sitting exactly on the threshold and
/// going nowhere, and that is what "wanders but does not drift" means.
///
/// # Where the solver is
///
/// Measured at pile sizes twenty, forty and sixty, before the ground patch became one
/// constraint: ratios **23.6, 18.7 and 18.4**, mean 20.3, with the long window at 0.275
/// of a reach -- above the random walk, and the pile really was still going. Since the
/// patch, ratio 39.9 and long window 0.094: the growth is *more* linear and the pile has
/// moved a third as far, which is the case the first predicate got wrong.
///
/// Since the ground patch was given a friction anchor, the long window over eight draws is
/// **0.007, 0.103 and 0.126** of a reach at twenty, forty and sixty against 0.075, 0.141
/// and 0.217 before it, and the straightness of that travel -- see below -- falls from
/// 0.88, 0.80 and 0.72 to 0.37, 0.46 and 0.53. The forty and sixty medians are below the
/// whole eight-draw spread they used to sit in, so that is the solver and not the draw.
///
/// The cause of what is left is understood and is the same one, one level up: the contacts
/// *between* bodies have no anchor, and they carry what remains. Measured before the
/// anchor landed, with friction between bodies switched off and only the ground's left, a
/// settled pile of twenty and of forty stop outright. Two capsules lying against each
/// other are also a patch sampled twice exactly as a capsule on the plane was, and
/// `capsule_contact` still emits the two ends as two independent contacts: two capsules
/// stacked drift at 2.3 mm a second where a lone capsule on the ground drifts at nothing
/// measurable.
///
/// # How much of the ratio is the solver and how much is the draw
///
/// **Not as much as it looks, and this is a warning to whoever reads a number out of
/// here.** A settled pile is chaotic: perturbing each body's starting height by a
/// relative seven parts in a million million -- far below anything the solver could be
/// said to resolve -- and running the same measurement eight times gives, on this
/// implementation, ratios of 15.0 to 40.3 on a pile of sixty and 16.3 to 31.3 on a pile
/// of forty. The bound of 28 sits inside both spreads, so which side of it a given
/// arrangement falls is the arrangement and not the solve. The long window is the steadier
/// half of the pair -- 0.123 to 0.171 of a reach over the same eight draws at forty
/// bodies, a quarter of the spread the ratio has, because the short window is a small
/// number in a denominator.
///
/// What that means in practice: a single run of this law moving from 18 to 33 is not
/// evidence of anything, and a change to the solver has to be judged on several draws of
/// the median long window rather than on one ratio.
///
/// **A predicate that does not have that problem** is the straightness of the travel: walk
/// the same four hundred and eighty steps as thirty-two windows of fifteen, and compare
/// each body's net displacement with the sum of the thirty-two window displacements it
/// walked. A body that is drifting has a ratio of one; a body wandering over `n`
/// independent windows has `1 / sqrt(n)`, which is 0.177 here. Both halves come off the
/// same trajectory and the denominator is a sum of thirty-two terms, so there is nothing
/// small under the line. It was not adopted here because it did not pass: measured on this
/// pile it was **0.905**, and on a settled rig 0.99, which is the honest statement that
/// both of them really were drifting and that the ratio predicate cannot see it. It is now
/// what `a_settled_rig_stays_where_it_settled` asserts, because the rig has stopped; this
/// pile is at 0.37 to 0.53 and is on its way but not there.
#[test]
fn a_settled_pile_wanders_but_does_not_drift() {
    // The fraction of its own reach a body may move over a settling window and still be
    // called still. Written out here rather than reached for, because this is the
    // solver's own threshold seen from outside it: it is what `Skeleton::step` uses to
    // decide that a body may sleep, and if the two ever disagree this law is measuring
    // something the solver is not.
    const STILL_FRACTION: f64 = 0.02;

    for count in [20usize, 40, 60] {
        let mut s = heap(count);
        for _ in 0..1500 {
            s.step(DT, G, 8);
        }
        let long = median_drift(&mut s, 480);

        // What a pile of bodies sitting exactly on the sleeping threshold and going
        // nowhere would cover over the long window: each settling window independent of
        // the last, so the square root and not the count.
        let windows: f64 = 480.0 / 15.0;
        let wandering = STILL_FRACTION * windows.sqrt();
        // And what a pile every body of which slid the whole time would cover: the count
        // rather than its square root, because a drift adds up and a wander does not.
        let sliding = STILL_FRACTION * windows;

        // The guard sits between the two, near enough to `sliding` to survive the
        // chaos and far enough below it to catch a pile that has started to travel.
        //
        // **Why this is one absolute bound and not the ratio it used to be.** A settling
        // pile is chaotic: perturbing each body's starting height by a *relative* 7e-12 --
        // a change of one unit in the last place -- moves the answer across a wide range,
        // because tiny differences in who touches whom first compound. Measured over
        // eight such seeds on the commit this bound was set from:
        //
        // ```text
        //   pile   long-window drift        short/long ratio
        //     20   0.000 .. 0.095           4.7 .. infinite
        //     40   0.134 .. 0.202          14.4 .. 32.3
        //     60   0.196 .. 0.258          20.0 .. 31.1
        // ```
        //
        // The ratio this law used to assert had a bound of 28, and the spread above
        // straddles it at two of the three sizes -- so the law passed on the seed it
        // happened to be written against and would have failed on a neighbouring one,
        // while reporting a regression that had not happened. It also goes infinite
        // whenever the short window reads exactly zero. The drift alone is steadier by a
        // factor of two and has no singularity, so it is what the law weighs.
        //
        // Staggering the normal and friction solves moved those figures without moving
        // this bound, and the shape of the change is the same one the rig shows: the tail
        // comes in. Over eight draws a relative 1e-12 apart, long-window drift before and
        // after:
        //
        // ```text
        //     20   0.000..0.137   ->   0.000..0.008
        //     40   0.063..0.135   ->   0.037..0.105
        //     60   0.051..0.193   ->   0.107..0.203
        // ```
        //
        // The pile does still drift, and this bound does not pretend otherwise -- see the
        // header for the ratchet that causes it. This is a regression guard on a known
        // limitation, not a claim that a pile is still.
        let guard = 0.7 * sliding;
        assert!(
            long < guard,
            "a pile of {count} covered {long:.4} of a reach over four hundred and eighty \
             steps, past the {guard:.4} this law allows. A pile jostling on the sleeping \
             threshold and going nowhere would cover {wandering:.4}; one whose every body \
             slid the whole time would cover {sliding:.4}. This is nearer the second.",
        );
    }
}

/// How far the middle body of a pile's surface moves over `steps`, as a fraction of its
/// own reach. The median rather than the mean, because one body rolling off the top of a
/// pile is not what this is asking about.
fn median_drift(s: &mut Skeleton, steps: usize) -> f64 {
    let before: Vec<_> = (0..s.len())
        .map(|i| (s.position(i), s.orientation(i)))
        .collect();
    for _ in 0..steps {
        s.step(DT, G, 8);
    }
    let mut ratios: Vec<f64> = (0..s.len())
        .map(|i| {
            let (was, now) = (before[i], (s.position(i), s.orientation(i)));
            let body = s.body(i);
            let travel = speed((now.0 .0 - was.0 .0, now.0 .1 - was.0 .1, now.0 .2 - was.0 .2));

            // The turn since, as an axis-angle vector, split into the part about the
            // capsule's own long axis and the rest. See the law above for why the two are
            // charged at different radii.
            let delta = now.1.multiply(&was.1.conjugate());
            let sign = if delta.w < 0.0 { -1.0 } else { 1.0 };
            let turn = (
                2.0 * sign * delta.x,
                2.0 * sign * delta.y,
                2.0 * sign * delta.z,
            );
            let axis = was.1.rotate_point((0.0, 1.0, 0.0));
            let along = turn.0 * axis.0 + turn.1 * axis.1 + turn.2 * axis.2;
            let across = speed((
                turn.0 - along * axis.0,
                turn.1 - along * axis.1,
                turn.2 - along * axis.2,
            ));

            let reach = body.radius + body.half_length;
            (travel + across * reach + along.abs() * body.radius) / reach
        })
        .collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).expect("no body is at a NaN"));
    ratios[ratios.len() / 2]
}

/// **A heap settles, stays where it settled, and stays out of the ground.**
///
/// The three failures this catches are different: bodies still moving means the solver is
/// feeding them, a heap that keeps spreading means nothing resists rolling, and a body
/// below the plane means the contact is losing to something.
#[test]
fn a_heap_settles_and_stays_where_it_settled() {
    let mut s = heap(40);
    for _ in 0..900 {
        s.step(DT, G, 8);
    }
    let footprint_then = footprint(&s);
    let spread_then = spread(&s);

    for i in 0..s.len() {
        assert!(
            s.position(i).1 > -1e-3,
            "body {i} is at y {:.4}, below the ground",
            s.position(i).1,
        );
        assert!(
            speed(s.velocity(i)) < 0.25,
            "body {i} is still moving at {:.3} m/s after fifteen seconds",
            speed(s.velocity(i)),
        );
    }

    // And it is still there a further fifteen seconds later, which is the part that
    // separates a heap at rest from a heap drifting slowly enough to look like one.
    //
    // **Measured on the whole pile, not on its furthest body.** A pile of forty capsules
    // is chaotic: which body ends up outermost, and what it is balanced on when the clock
    // starts, swing the furthest radius by hundreds of millimetres between two runs that
    // differ only in the iteration count. Measured across pile sizes from twenty to sixty,
    // the furthest body's fifteen-second drift scatters over -0.02 m to 0.18 m with no
    // trend, so a tight bound on it tests the draw rather than the solver.
    //
    // The mean radius does not scatter: a settled pile holds it to a few centimetres at
    // every size, and a pile that is genuinely rolling apart -- the same fixture with
    // rolling resistance turned off -- moves it by 0.7 m to 2.3 m. Two orders of
    // magnitude of daylight, so this is the looser-looking number that actually
    // discriminates.
    for _ in 0..900 {
        s.step(DT, G, 8);
    }
    let spread_now = spread(&s);
    assert!(
        spread_now - spread_then < 0.1,
        "the pile's mean radius went from {spread_then:.3} m to {spread_now:.3} m while it \
         was supposed to be at rest; a settled pile holds it to about 0.05 m and one that \
         is rolling apart moves it by a metre",
    );
    let footprint_now = footprint(&s);
    assert!(
        footprint_now - footprint_then < 0.5,
        "and no body should have left: the furthest went from {footprint_then:.3} m to \
         {footprint_now:.3} m",
    );
}

/// **A jointed rig dropped on the plane comes to rest, if it does not collide with
/// itself.**
///
/// The whole of what `set_self_collision` is for, and the reason it exists rather than
/// being a hard-coded choice. A contact between two bodies of one skeleton closes a loop
/// with the joints: the joints hold the pair in a small overlap, the contact pushes them
/// apart, the joints put them back, and the two corrections are applied one after the
/// other. Rigid displacements about different points do not compose back to where they
/// started, and since the configuration repeats every step so does the leftover, which
/// integrates into a straight walk. Measured on this rig: net travel is 0.99 of the path
/// walked getting there, at twenty to forty millimetres a second, for ever.
///
/// With the loop removed the rig lands, stops, and leaves the simulation. The bound is
/// generous on purpose -- a rig with limbs to fold takes a few seconds of settling and
/// the exact count is chaotic -- because what this law claims is that it happens at all,
/// which is the difference between sixty nanoseconds a step and milliseconds.
///
/// A rig that may touch itself is the harder case and used to be a separate question
/// entirely; it is now a law of its own, immediately below.
///
/// The bound here is generous for the same reason it always was, but the numbers behind it
/// have moved again: over six draws this rig sleeps in all of them, between steps 80 and
/// 117, where before the normal and friction solves were staggered it took between 127 and
/// 342, and before the ground anchor four of six slept and took between 198 and 1520.
#[test]
fn a_rig_that_does_not_touch_itself_comes_to_rest() {
    let mut s = Skeleton::new();
    s.set_ground((0.0, 1.0, 0.0), 0.0);
    s.set_self_collision(false);
    let bones = rig(&mut s, 1.0);
    assert!(!s.self_collision(), "the switch did not take");

    let mut slept = None;
    for step in 1..=3000 {
        s.step(DT, G, 8);
        if s.awake_count() == 0 {
            slept = Some(step);
            break;
        }
    }
    let slept = slept.unwrap_or_else(|| {
        panic!(
            "a rig of {bones} bones that cannot touch itself was still being solved after \
             fifty seconds; {} of {bones} bodies awake, the median one moving at {:.4} m/s",
            s.awake_count(),
            {
                let mut v: Vec<f64> = (0..s.len()).map(|i| speed(s.velocity(i))).collect();
                v.sort_by(|a, b| a.partial_cmp(b).expect("no body is at a NaN"));
                v[v.len() / 2]
            },
        )
    });
    assert!(
        slept > 30,
        "it slept at step {slept}, which is inside the time it takes to fall from a metre \
         -- something is calling a rig still while it is still in the air",
    );
    for i in 0..s.len() {
        assert!(
            s.position(i).1 > -1e-3,
            "bone {i} came to rest at y {:.4}, below the ground",
            s.position(i).1,
        );
    }
}

/// **And so does a rig that may touch itself.**
///
/// The same claim as the law above, for the case it deliberately did not make for most of
/// this module's life. With self-collision on, a contact between two of a skeleton's own
/// bones closes a loop with the joints, and every defect in the solver's module header
/// lives in that loop: first the rig walked, then it sat on a limit cycle, and last a
/// single bone propped on another buzzed on alternate steps because its contact existed
/// only every other one. A rig that never stops being solved costs milliseconds a step for
/// ever, where one that has stopped costs a bitset scan.
///
/// # Why this is stated over draws
///
/// A settling rig is chaotic. Eight starts a relative 1e-12 apart -- one unit in the last
/// place -- fall into visibly different heaps, and a law run on one of them reports which
/// heap it happened to pick. The same reason `a_settled_rig_stays_where_it_settled` weighs
/// eight draws, and here the law is the stronger one: **every** draw has to come to rest,
/// because a rig that stops on seven starts in eight and runs for ever on the eighth has
/// not stopped.
///
/// # The bound
///
/// Generous on purpose, like the law above and for the same reason: what is claimed is that
/// it happens at all. Measured over thirty-two draws, the rig sleeps in all of them between
/// steps 112 and 296, where before the narrow phase kept a loaded pair's contact alive it
/// slept in nine of sixteen, between steps 312 and 1312. Two thousand steps is thirty-three
/// seconds and six times the worst draw.
#[test]
fn a_rig_that_touches_itself_comes_to_rest() {
    const DRAWS: usize = 8;
    const CAP: usize = 2000;

    let mut slept = Vec::new();
    let mut bones = 0;
    for draw in 0..DRAWS {
        let mut s = Skeleton::new();
        s.set_ground((0.0, 1.0, 0.0), 0.0);
        bones = rig(&mut s, 1.0 * (1.0 + draw as f64 * 1e-12));
        assert!(s.self_collision(), "this law is about a rig that may touch itself");

        let mut when = None;
        for step in 1..=CAP {
            s.step(DT, G, 8);
            if s.awake_count() == 0 {
                when = Some(step);
                break;
            }
        }
        let when = when.unwrap_or_else(|| {
            panic!(
                "draw {draw} of a {bones}-bone rig that may touch itself was still being \
                 solved after {CAP} steps; {} of {bones} bodies awake, the median one \
                 moving at {:.4} m/s. Draws so far slept at {slept:?}",
                s.awake_count(),
                {
                    let mut v: Vec<f64> = (0..s.len()).map(|i| speed(s.velocity(i))).collect();
                    v.sort_by(|a, b| a.partial_cmp(b).expect("no body is at a NaN"));
                    v[v.len() / 2]
                },
            )
        });
        assert!(
            when > 30,
            "draw {draw} slept at step {when}, which is inside the time it takes to fall \
             from a metre -- something is calling a rig still while it is still in the air",
        );
        for i in 0..s.len() {
            assert!(
                s.position(i).1 > -1e-3,
                "bone {i} of draw {draw} came to rest at y {:.4}, below the ground",
                s.position(i).1,
            );
        }
        slept.push(when);
    }
    assert_eq!(slept.len(), DRAWS, "every draw has to have come to rest");
}

/// **A settled rig stays where it settled, even where it touches itself.**
///
/// This is the law the ground patch's friction anchor exists for, and it is stated the way
/// the pile law states its own bound rather than as a recorded number.
///
/// # What was wrong, in one sentence
///
/// A contact between two bones of one skeleton closes a loop with the joints, the loop's
/// corrections do not commute, and the leftover is the same small screw every step; the
/// ground's friction turns that shake into travel, the way a crawling thing gets along.
/// Measured on this rig before the anchor: 0.239 of a body's reach over four hundred and
/// eighty steps with a straightness of 0.72, which is a walk and not a jostle.
///
/// # The bound, and why it is this one
///
/// A body is entitled to be called still while it stays within [`STILL_FRACTION`] of its
/// own reach of where its settling window opened. Over `n` such windows, one jostling in
/// place covers `STILL_FRACTION * sqrt(n)` -- a random walk -- and one that is travelling
/// covers `STILL_FRACTION * n`. This watches thirty-two windows, so the jostling allowance
/// is 0.113 of a reach and a travelling rig would cover 0.64. The law asks for the first.
///
/// **And it asks for straightness as well**, because that is the statistic that tells a
/// ratchet from a wander and the reason the drift alone is not enough: net travel over the
/// sum of the thirty-two window travels is one for a body being carried and about
/// `1/sqrt(32)` -- 0.18 -- for one being jostled. Measured now: 0.021 of a reach and a
/// straightness of 0.036, so both bounds have an order of margin. They are written where
/// the physics puts them rather than next to the measurement, so that a change which halves
/// the margin passes and one that brings the walk back does not.
///
/// # The outlier is bounded by its straightness, not only by its count
///
/// The arm below that counts the draws past the jostling allowance was written when one
/// draw in twelve bolted six reaches, and counting them was all that could be asked for.
/// Since the normal and friction solves were staggered -- see the solver's module header --
/// no draw is *carried* any more, and that is a sharper thing to assert than a rate, because
/// it does not depend on how many draws happen to be run. Measured over twenty-four draws a
/// relative 1e-12 apart: the worst straightness is 0.31 and the worst drift 0.19 of a reach,
/// against 0.94 and 6.37 before. So every draw now has to look jostled, and the count arm
/// stays as the weaker guard behind it.
#[test]
fn a_settled_rig_stays_where_it_settled() {
    const STILL_FRACTION: f64 = 0.02;
    const WINDOWS: usize = 32;
    const PER: usize = 15;
    // Several starts, a relative 1e-12 apart, which is one unit in the last place. A
    // settling rig is chaotic and one draw says nothing: the first version of this law
    // ran a single start, passed, and was quoted as evidence that a rig no longer
    // travels. Measured over twelve draws it stops dead on eleven and bolts on one, so
    // what a single draw reported was which draw it happened to pick.
    const DRAWS: usize = 8;

    let mut drifts = Vec::new();
    let mut straights = Vec::new();
    let mut bones = 0;
    for draw in 0..DRAWS {
        let mut s = Skeleton::new();
        s.set_ground((0.0, 1.0, 0.0), 0.0);
        // Self-collision left on, which is the whole point: with it off this rig goes to
        // sleep and there is nothing left to measure.
        bones = rig(&mut s, 1.0 * (1.0 + draw as f64 * 1e-12));
        assert!(s.self_collision(), "this law is about a rig that may touch itself");
        s.set_sleeping(false);
        for _ in 0..1500 {
            s.step(DT, G, 8);
        }

        let pose = |s: &Skeleton| -> Vec<((f64, f64, f64), rs_physics::models::Quaternion)> {
            (0..s.len())
                .map(|i| (s.position(i), s.orientation(i)))
                .collect()
        };
        let start = pose(&s);
        let mut last = start.clone();
        let mut path = vec![0.0f64; s.len()];
        for _ in 0..WINDOWS {
            for _ in 0..PER {
                s.step(DT, G, 8);
            }
            let now = pose(&s);
            for i in 0..s.len() {
                path[i] += surface_travel(&s, i, last[i], now[i]);
            }
            last = now;
        }
        let now = pose(&s);

        let mut bone_drift: Vec<f64> = Vec::new();
        let mut bone_straight: Vec<f64> = Vec::new();
        for i in 0..s.len() {
            let net = surface_travel(&s, i, start[i], now[i]);
            bone_drift.push(net);
            // **A bone that has not moved has no direction**, and asking one for a
            // straightness is dividing arithmetic noise by itself: it answers one, which
            // is the number this law reads as "carried". The guard used to be `> 0.0`,
            // which only caught a bone that was still to the last bit, and it was enough
            // while every draw still had a jostle in it. It stopped being enough the step
            // a rig started freezing outright -- measured, four draws of eight came back
            // with a drift of 0.0000 of a reach and a straightness of 1.0000, which is a
            // rig sitting perfectly still failing a law about rigs that travel.
            //
            // The line is the law's own `STILL_FRACTION`: a body is entitled to be called
            // still while it stays inside that fraction of its own reach over a settling
            // window, so a bone that has not covered even one window's worth over all
            // thirty-two of them has not gone anywhere. This makes the law stricter rather
            // than looser -- it removes a false positive and leaves every real travel
            // untouched, because a carried bone's path is an order above this.
            bone_straight.push(if path[i] > STILL_FRACTION {
                net / path[i]
            } else {
                0.0
            });
        }
        bone_drift.sort_by(|a, b| a.partial_cmp(b).expect("no bone is at a NaN"));
        bone_straight.sort_by(|a, b| a.partial_cmp(b).expect("no bone is at a NaN"));
        drifts.push(bone_drift[bone_drift.len() / 2]);
        straights.push(bone_straight[bone_straight.len() / 2]);
    }

    let jostling = STILL_FRACTION * (WINDOWS as f64).sqrt();
    let travelling = STILL_FRACTION * WINDOWS as f64;

    // The typical draw is what the fix is claimed to have achieved, so the median draw is
    // what the law weighs.
    let mut ordered = drifts.clone();
    ordered.sort_by(|a, b| a.partial_cmp(b).expect("no draw is at a NaN"));
    let typical = ordered[ordered.len() / 2];
    assert!(
        typical < jostling,
        "the median draw's median bone covered {typical:.4} of its own reach over \
         {WINDOWS} windows, across {DRAWS} draws of a {bones}-bone rig. One jostling on \
         the sleeping threshold would cover {jostling:.4} and one carried the whole way \
         {travelling:.4}: this is the second. Drifts were {drifts:.4?}.",
    );

    // And the outliers are bounded rather than ignored. Measured over twelve draws on the
    // commit this bound was set from, eleven had a drift under 0.011 of a reach and one
    // bolted at 6.368 with a straightness of 0.936 -- a rig that found somewhere to go.
    // That was a real defect and it is recorded here rather than hidden behind a median:
    // what this arm asserts is that it stays an outlier. A quarter of the draws going the
    // same way would be the fix having stopped working.
    let bolted = drifts.iter().filter(|d| **d >= jostling).count();
    assert!(
        bolted * 4 <= DRAWS,
        "{bolted} of {DRAWS} draws of a {bones}-bone rig travelled past the {jostling:.4} \
         a jostling body covers, where the recorded rate is one in twenty-four. Drifts \
         were {drifts:.4?}, straightnesses {straights:.4?}.",
    );

    // Straightness separates a rig that wandered from one that was carried, and **no draw**
    // may look like the second -- not merely the typical one. The line is the law's own:
    // a body being carried measures one and a jostled one about `1/sqrt(32)`, so a half is
    // the midpoint between them and the same number the median arm used to be the only
    // user of. See the header for the twenty-four draws behind it.
    let mut ordered = straights.clone();
    ordered.sort_by(|a, b| a.partial_cmp(b).expect("no draw is at a NaN"));
    let worst = ordered[ordered.len() - 1];
    assert!(
        worst < 0.5,
        "a draw's median bone made {worst:.4} of the path it walked into net travel. A \
         body being carried measures one and a jostled one about 0.18, so this is a rig \
         with somewhere to go rather than one sitting still. Straightnesses were \
         {straights:.4?}, drifts {drifts:.4?}.",
    );
}

/// **A resting contact is perfectly inelastic, so a settled rig does not sit on a limit
/// cycle.**
///
/// # The quantity, and why it is this one
///
/// Gravity gives every body `g dt` of speed in a step and the contact it is resting on
/// takes the same `g dt` back out; a body that is genuinely at rest ends the step with
/// neither. Whatever speed a settled body *is* left carrying is what the contact failed to
/// take back, measured in the only unit the question has -- one step of gravity. So the
/// law is stated as a fraction of `g dt` rather than in metres a second, and it does not
/// move if the step or the gravity does.
///
/// # What it caught
///
/// Before the velocity pass -- see the solver's module header -- this rig's median bone
/// held 0.335, 0.342, 0.388, 0.357, 0.347 and 0.335 of `g dt` on six draws a relative
/// 1e-12 apart. **The same third on every draw**, which is an attractor rather than a
/// spread: the solve was feeding one side of a cycle and Coulomb was taking from the
/// other, and nothing in the module was entitled to say what the relative velocity of two
/// resting surfaces ought to be. A quarter of a step of gravity is below every one of
/// those six and above the typical draw now, which is what a regression guard on that
/// defect has to be.
///
/// # And why the median draw
///
/// A settling rig is chaotic, and one draw says nothing -- the same reason
/// [`a_settled_rig_stays_where_it_settled`] weighs its median. Measured over sixteen draws
/// the residual runs 0.038 to 0.198 with two outliers at 0.425 and 0.578, so the attractor
/// is gone but something in its band is still reachable; that is recorded in the module
/// header as one of the things the pass did not close, and this arm is deliberately not
/// the one that would fail for it.
#[test]
fn a_settled_rig_does_not_sit_on_a_limit_cycle() {
    // What one step of gravity is worth as a speed, which is the unit the question has.
    let step_of_gravity = speed(G) * DT;
    const DRAWS: usize = 8;
    const SETTLE: usize = 1500;
    const WATCH: usize = 60;

    let mut residual = Vec::new();
    let mut bones = 0;
    for draw in 0..DRAWS {
        let mut s = Skeleton::new();
        s.set_ground((0.0, 1.0, 0.0), 0.0);
        // Self-collision on, which is the case the cycle lived in: with it off this rig
        // goes to sleep and there is nothing to measure.
        bones = rig(&mut s, 1.0 * (1.0 + draw as f64 * 1e-12));
        assert!(s.self_collision(), "this law is about a rig that may touch itself");
        // Sleeping off, or a rig that settles is removed from the step and reads zero --
        // which would pass the law by not answering it.
        s.set_sleeping(false);
        for _ in 0..SETTLE {
            s.step(DT, G, 8);
        }
        let mut watched = Vec::new();
        for _ in 0..WATCH {
            s.step(DT, G, 8);
            let mut bone: Vec<f64> = (0..s.len()).map(|i| speed(s.velocity(i))).collect();
            bone.sort_by(|a, b| a.partial_cmp(b).expect("no bone is at a NaN"));
            watched.push(bone[bone.len() / 2]);
        }
        watched.sort_by(|a, b| a.partial_cmp(b).expect("no step is at a NaN"));
        residual.push(watched[watched.len() / 2] / step_of_gravity);
    }

    let mut ordered = residual.clone();
    ordered.sort_by(|a, b| a.partial_cmp(b).expect("no draw is at a NaN"));
    let typical = ordered[ordered.len() / 2];
    assert!(
        typical < 0.25,
        "the median draw of a settled {bones}-bone rig holds {typical:.3} of one step of          gravity in its median bone. A body a resting contact has finished with holds          none of it, and the cycle this law was written against held a third of it on          every draw. Residuals were {residual:.3?}.",
    );
}

/// How far body `i`'s surface moved between two poses, as a fraction of its own reach: the
/// centre's travel, plus the turn charged at the reach except about the body's own long
/// axis, where a surface of revolution has not gone anywhere and it is charged at the
/// radius.
fn surface_travel(
    s: &Skeleton,
    i: usize,
    was: ((f64, f64, f64), rs_physics::models::Quaternion),
    now: ((f64, f64, f64), rs_physics::models::Quaternion),
) -> f64 {
    let body = s.body(i);
    let travel = speed((now.0 .0 - was.0 .0, now.0 .1 - was.0 .1, now.0 .2 - was.0 .2));
    let delta = now.1.multiply(&was.1.conjugate());
    let sign = if delta.w < 0.0 { -1.0 } else { 1.0 };
    let turn = (
        2.0 * sign * delta.x,
        2.0 * sign * delta.y,
        2.0 * sign * delta.z,
    );
    let axis = was.1.rotate_point((0.0, 1.0, 0.0));
    let along = turn.0 * axis.0 + turn.1 * axis.1 + turn.2 * axis.2;
    let across = speed((
        turn.0 - along * axis.0,
        turn.1 - along * axis.1,
        turn.2 - along * axis.2,
    ));
    let reach = body.radius + body.half_length;
    (travel + across * reach + along.abs() * body.radius) / reach
}

/// **A skeleton left to itself does not move its own centre of mass.** Joints and the
/// contacts between a skeleton's own bones are internal: they are equal and opposite pairs
/// and they cannot carry the whole of it anywhere. With no gravity and no ground there is
/// nothing else acting, so whatever the solver does inside the rig -- pulling violated
/// joints back together, pushing overlapping bones apart, doing both in a loop for ever --
/// the mass-weighted mean of the positions must be exactly where it started.
///
/// # Why this is worth a law of its own
///
/// It is the statement that separates two very different failures, and the diagnosis of
/// the drift in [`Skeleton::step`]'s header rests on it. A rig resting on the plane walks,
/// and there are only two ways it could: the internal corrections could be pushing it, in
/// which case this law fails and momentum is being manufactured; or they could be shaking
/// it while the ground's friction converts the shake into travel, in which case this law
/// holds and the fault is at the contact. Measured, this holds to under a nanometre over
/// two thousand steps, which is what says the travel leaves through the ground rather than
/// out of the joints.
///
/// The rig is started **wrong** on purpose -- every other bone displaced by fifty
/// millimetres, which is most of a bone -- so that the solver has real work to do rather
/// than confirming that a rig already at rest stays there. The test asserts that the work
/// happened as well as that the centre did not move, because a law that both sides pass by
/// doing nothing is not a law.
#[test]
fn a_skeleton_left_to_itself_does_not_move_its_own_centre_of_mass() {
    let mut s = Skeleton::new();
    // No ground and no gravity: nothing outside the rig is acting on it at all.
    s.set_sleeping(false);
    let bones = rig(&mut s, 0.0);
    for i in (0..s.len()).step_by(2) {
        let mut body = s.body(i);
        body.position.0 += 0.05;
        s.set_body(i, body);
    }

    let mass: Vec<f64> = (0..s.len()).map(|i| 1.0 / s.body(i).inv_mass).collect();
    let total: f64 = mass.iter().sum();
    let centre = |s: &Skeleton| {
        let mut c = (0.0, 0.0, 0.0);
        for i in 0..s.len() {
            let p = s.position(i);
            c = (
                c.0 + p.0 * mass[i] / total,
                c.1 + p.1 * mass[i] / total,
                c.2 + p.2 * mass[i] / total,
            );
        }
        c
    };
    let started = centre(&s);
    let placed: Vec<(f64, f64, f64)> = (0..s.len()).map(|i| s.position(i)).collect();

    for _ in 0..2000 {
        s.step(DT, (0.0, 0.0, 0.0), 8);
    }

    let mut worked = 0.0f64;
    for i in 0..s.len() {
        let now = s.position(i);
        worked = worked.max(speed((
            now.0 - placed[i].0,
            now.1 - placed[i].1,
            now.2 - placed[i].2,
        )));
    }
    assert!(
        worked > 0.01,
        "no bone of the {bones} moved as much as ten millimetres, so the rig was never \
         put wrong enough for this to be a test of anything",
    );

    let ended = centre(&s);
    let went = speed((ended.0 - started.0, ended.1 - started.1, ended.2 - started.2));
    // The bound is arithmetic rather than physics: the corrections are equal and opposite
    // before they are divided by mass, so the only thing left is the last bits of the
    // division, and over two thousand steps of a seventeen-bone rig that is nanometres.
    assert!(
        went < 1e-9,
        "the rig's centre of mass moved {went:.3e} m with nothing acting on it, while its \
         bones moved {worked:.3} m: the joints and the contacts between its own bones are \
         carrying it somewhere, which no internal force can do",
    );
}

/// **Asking for more quality never makes the answer worse.**
///
/// `iterations` is the one dial a caller turns, and every other law in this file runs the
/// rig at eight. That is why this defect survived: a settled seventeen-bone rig held
/// **one to two kilojoules** at twelve, sixteen, twenty-four and thirty-two passes while
/// eight and sixty-four were quiet, and nothing was watching. Traced, five draws of ten
/// ended up airborne -- every bone off the ground, bodies at ten to fifteen metres a
/// second, on three to eleven contacts between its own limbs -- and stayed there for
/// twelve thousand steps without decaying. A rig that never touches the ground is not
/// taking that energy from the ground.
///
/// # The property, and why it is stated as one bound rather than as a trend
///
/// The tempting statement is monotonicity -- that each count is at least as good as the
/// one below. It is the wrong one: a settling rig is chaotic, so two counts differ by
/// which configuration the rig happens to land in as well as by how well it was solved,
/// and a monotone assertion would fail on noise while a real doubling of the residual
/// slipped past. The right statement is that **the bound does not depend on the count**:
/// whatever a caller asks for, a settled rig has to be as still as
/// `a_settled_rig_does_not_sit_on_a_limit_cycle` requires it to be at eight, measured in
/// the same unit and against the same number. More passes may then be better or level; it
/// may not be a different regime.
///
/// The unit is one step of gravity, which is what a resting contact has finished handing a
/// body and therefore the largest speed a settled one has any business holding. The
/// quarter is the same quarter the eight-pass law uses, and it is not a recorded output:
/// the cycle these laws were written against sat at a third of `g dt` on every draw, so a
/// quarter is inside that and an order above the arithmetic.
#[test]
fn asking_for_more_passes_does_not_make_a_settled_rig_worse() {
    let step_of_gravity = speed(G) * DT;
    const DRAWS: usize = 6;
    const SETTLE: usize = 1200;
    const WATCH: usize = 60;
    // Eight is what every other law runs, and the four above it are where the defect
    // lived. Sixty-four is left out only because it costs eight times eight to say
    // something the four already say.
    const COUNTS: [usize; 5] = [8, 12, 16, 24, 32];

    let mut residual = Vec::new();
    let mut bones = 0;
    for iterations in COUNTS {
        let mut draws = Vec::new();
        for draw in 0..DRAWS {
            let mut s = Skeleton::new();
            s.set_ground((0.0, 1.0, 0.0), 0.0);
            bones = rig(&mut s, 1.0 * (1.0 + draw as f64 * 1e-12));
            assert!(s.self_collision(), "this law is about a rig that may touch itself");
            // Sleeping off, or a rig that settles is removed from the step and reads zero
            // -- which would pass the law by not answering it.
            s.set_sleeping(false);
            for _ in 0..SETTLE {
                s.step(DT, G, iterations);
            }
            let mut watched = Vec::new();
            for _ in 0..WATCH {
                s.step(DT, G, iterations);
                let mut bone: Vec<f64> = (0..s.len()).map(|i| speed(s.velocity(i))).collect();
                bone.sort_by(|a, b| a.partial_cmp(b).expect("no bone is at a NaN"));
                watched.push(bone[bone.len() / 2]);
            }
            watched.sort_by(|a, b| a.partial_cmp(b).expect("no step is at a NaN"));
            draws.push(watched[watched.len() / 2] / step_of_gravity);
        }
        draws.sort_by(|a, b| a.partial_cmp(b).expect("no draw is at a NaN"));
        residual.push((iterations, draws[draws.len() / 2], draws[draws.len() - 1]));
    }

    for &(iterations, typical, _) in residual.iter() {
        assert!(
            typical < 0.25,
            "at {iterations} passes the median draw of a settled {bones}-bone rig holds {typical:.3} of one step of gravity in its median bone, where eight passes is held to a quarter. Asking for more quality moved the answer into a different regime. Residuals, as (passes, median, worst): {residual:.3?}",
        );
    }

    // And the outlier arm, because the defect this law was written against was not in the
    // median: at sixteen and thirty-two passes the median draw was as still as it ever is
    // and one draw of ten held a kilojoule. A rig off the ground reads tens of steps of
    // gravity, so this is orders clear of anything a settled rig does and will only catch
    // a scene that has found a way to power itself.
    for &(iterations, _, worst) in residual.iter() {
        assert!(
            worst < 4.0,
            "at {iterations} passes one draw of a settled {bones}-bone rig holds {worst:.3} of one step of gravity in its median bone. That is a rig with an energy source rather than a rig settling. Residuals, as (passes, median, worst): {residual:.3?}",
        );
    }
}

/// **A hinge keeps its axis pointing the way it was handed.**
///
/// # Why this is a law and not a detail
///
/// A hinge holds two things: that the bodies' axes are the *same* axis, and that the swing
/// about it stays inside the range. The second is only meaningful if the first is -- the
/// angle is measured from a reference carried in each body's own frame, so a child that has
/// been turned end for end about the joint reports its swing from a zero that is half a turn
/// out and with the opposite sign. `min` and `max` are then enforced on a number that is not
/// the angle anybody asked about, and the limit drives the limb to the wrong end of its
/// range and holds it there.
///
/// # The bound, and where it comes from
///
/// The alignment correction turns the two axes onto one another about `a x b`, and that
/// cross product vanishes in two places: at no angle at all, and at a half turn. **Both are
/// fixed points and only one of them is a hinge.** The watershed between their basins is a
/// quarter turn, so the property a working hinge has is that it never reaches one -- the
/// cosine between the two axes stays positive. That is a statement about the geometry of
/// the constraint rather than a tolerance, which is why it is the bound here; the margin in
/// practice is far larger, and the message reports what was actually seen so that a change
/// which eats the margin is visible before it fails.
///
/// Measured on this rig, turned on its side so that it lands on its limbs: the worst cosine
/// any hinge reaches is **0.796**, and it is at step 1, while the joints are still hauling
/// the rig together out of the pose its bodies were authored in. Once it has landed, the
/// worst over the remaining twelve hundred steps is 0.9988. Before the correction's axis was
/// written the right way round, all eight hinges were inverted -- cosine -1.0000 -- within
/// five steps of the drop, and stayed there, and the limbs jammed 50 mm inside one another
/// for as long as they were watched.
#[test]
fn a_hinge_does_not_turn_itself_inside_out() {
    let mut s = Skeleton::new();
    s.set_ground((0.0, 1.0, 0.0), 0.0);
    let bones = rig(&mut s, 1.0);
    // Turned onto its side so that it lands on its limbs rather than on its feet, which is
    // what drives the hinges hard enough to be worth watching.
    for i in 0..s.len() {
        let mut body = s.body(i);
        let turn =
            rs_physics::models::Quaternion::from_axis_angle((0.0, 0.0, 1.0), 1.2);
        let p = turn.rotate_point((body.position.0, body.position.1 - 1.0, body.position.2));
        body.position = (p.0, p.1 + 1.0, p.2);
        body.orientation = turn.multiply(&body.orientation).normalized();
        s.set_body(i, body);
    }

    // Every hinge in the rig, as the pair of bodies it links and the local axis in each.
    let hinges: Vec<Hinge> = s
        .joints()
        .iter()
        .filter_map(|joint| match *joint {
            Joint::Hinge {
                a,
                b,
                axis_a,
                axis_b,
                ..
            } => Some((a, b, axis_a, axis_b)),
            Joint::Ball { .. } => None,
        })
        .collect();
    assert!(
        hinges.len() >= 8,
        "a {bones}-bone rig should carry the elbows and knees this law is about, and has \
         {} hinges",
        hinges.len(),
    );

    let mut worst = 1.0_f64;
    let mut worst_at = (0usize, 0usize, 0usize);
    for step in 1..=1200 {
        s.step(DT, G, 8);
        for &(a, b, axis_a, axis_b) in hinges.iter() {
            let wa = s.orientation(a).rotate_point(axis_a);
            let wb = s.orientation(b).rotate_point(axis_b);
            let cosine = (wa.0 * wb.0 + wa.1 * wb.1 + wa.2 * wb.2)
                / (speed(wa) * speed(wb)).max(1e-12);
            if cosine < worst {
                worst = cosine;
                worst_at = (step, a, b);
            }
        }
    }

    assert!(
        worst > 0.0,
        "the hinge between bodies {} and {} reached a cosine of {worst:.4} at step {} -- \
         past the quarter turn that separates a hinge from a hinge turned inside out, which \
         is the other place its own alignment correction is satisfied",
        worst_at.1,
        worst_at.2,
        worst_at.0,
    );
}

// -- fixtures ---------------------------------------------------------------------

/// A seventeen-bone rig with nothing holding it up: a pelvis, a spine of four up to a
/// head, and four limbs of three -- shoulders and hips as balls, elbows and knees as
/// hinges with a range -- dropped from `y`.
fn rig(into: &mut Skeleton, y: f64) -> usize {
    let base = into.len();
    let pelvis = into.add_body(Body::capsule(8.0, 0.1, 0.16, (0.0, y, 0.0)));
    let mut up = pelvis;
    for i in 0..4 {
        let link = into.add_body(Body::capsule(6.0, 0.08, 0.2, (0.0, y + 0.2 + 0.2 * i as f64, 0.0)));
        into.add_joint(Joint::free_ball(up, link, (0.0, 0.1, 0.0), (0.0, -0.1, 0.0)));
        up = link;
    }
    for limb in 0..4 {
        let (root, side) = if limb < 2 { (base + 3, 1.0) } else { (pelvis, -1.0) };
        let side_z = if limb % 2 == 0 { 0.15 } else { -0.15 };
        let mut previous = root;
        for segment in 0..3 {
            let body = into.add_body(Body::capsule(
                4.0,
                0.06,
                0.25,
                (0.0, y + side * 0.25 * segment as f64, side_z),
            ));
            // Shoulders and hips turn every way; elbows and knees do not.
            let joint = if segment == 0 {
                Joint::free_ball(previous, body, (0.0, 0.0, side_z), (0.0, 0.125, 0.0))
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

/// One hinge as `a_hinge_does_not_turn_itself_inside_out` reads it: the two bodies it
/// links and the axis each of them carries in its own frame.
type Hinge = (usize, usize, (f64, f64, f64), (f64, f64, f64));

/// A capsule turned to lie along x. A body's length runs down its own +Y, so this is the
/// quarter turn that puts it on its side.
fn lying(body: Body) -> Body {
    tilted(body, 0.0)
}

/// The same, then tilted by `tilt` radians about z, so it lies along a slope of that angle.
fn tilted(mut body: Body, tilt: f64) -> Body {
    body.orientation = rs_physics::models::Quaternion::from_axis_angle(
        (0.0, 0.0, 1.0),
        tilt - std::f64::consts::FRAC_PI_2,
    );
    body
}

fn speed(v: (f64, f64, f64)) -> f64 {
    (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt()
}

/// Total mechanical energy: kinetic, rotational and gravitational.
///
/// The datum is the ground, not some point far below it. A distant datum makes every
/// potential term huge and nearly constant, and then a real change in kinetic energy is
/// lost in the last digits of a large sum -- the test still passes, having measured
/// nothing.
fn energy(s: &Skeleton) -> f64 {
    const DATUM: f64 = 0.0;
    let mut total = 0.0;
    for i in 0..s.len() {
        let body = s.body(i);
        if body.inv_mass <= 0.0 {
            continue;
        }
        let mass = 1.0 / body.inv_mass;
        let v = speed(body.velocity);
        total += 0.5 * mass * v * v;
        total += mass * 9.80665 * (body.position.1 - DATUM);

        // Rotational energy needs the inertia back out of its reciprocal, skipping any
        // axis that was pinned.
        let w = body.angular_velocity;
        let inv = body.inv_inertia;
        for (component, inverse) in [(w.0, inv.0), (w.1, inv.1), (w.2, inv.2)] {
            if inverse > 0.0 {
                total += 0.5 * component * component / inverse;
            }
        }
    }
    total
}

/// How far the pile sits from the axis it was dropped on, on average. The stable measure
/// of a pile's size: see [`a_heap_settles_and_stays_where_it_settled`] for why the
/// furthest body is not one.
fn spread(s: &Skeleton) -> f64 {
    let mut total = 0.0;
    for i in 0..s.len() {
        let p = s.position(i);
        total += (p.0 * p.0 + p.2 * p.2).sqrt();
    }
    total / s.len().max(1) as f64
}

/// How far the furthest body is from the axis the heap was dropped on.
fn footprint(s: &Skeleton) -> f64 {
    let mut worst: f64 = 0.0;
    for i in 0..s.len() {
        let p = s.position(i);
        worst = worst.max((p.0 * p.0 + p.2 * p.2).sqrt());
    }
    worst
}

/// How far a capsule laid on a slope of `degrees` travels in `seconds`, after half a
/// second to settle onto the surface.
fn slid_down(degrees: f64, friction: f64, iterations: usize, seconds: f64) -> f64 {
    slid_down_with_mass(degrees, friction, iterations, seconds, 4.0)
}

fn slid_down_with_mass(
    degrees: f64,
    friction: f64,
    iterations: usize,
    seconds: f64,
    mass: f64,
) -> f64 {
    const R: f64 = 0.1;
    let angle = degrees.to_radians();
    let mut s = Skeleton::new();
    s.set_friction(friction);
    // Rolling would confuse a test about sliding, and the two are separate laws.
    s.set_rolling_resistance(0.0);
    let up = (-angle.sin(), angle.cos(), 0.0);
    s.set_ground(up, 0.0);

    // Placed on the surface rather than dropped, so what is measured is the slope and the
    // friction rather than the energy of an impact.
    let start = (up.0 * (R + 0.002), up.1 * (R + 0.002), 0.0);
    let body = s.add_body(tilted(Body::capsule(mass, R, 0.5, start), angle));

    for _ in 0..30 {
        s.step(DT, G, iterations);
    }
    let settled = s.position(body);
    for _ in 0..(seconds * 60.0) as usize {
        s.step(DT, G, iterations);
    }
    let now = s.position(body);
    let moved = (now.0 - settled.0, now.1 - settled.1, now.2 - settled.2);

    // Downhill is the part of gravity that lies in the plane, which for a slope tilted by
    // `angle` about z works out as -(cos, sin, 0). Projecting onto it discards the
    // sideways travel that rolling produces.
    let downhill = (-angle.cos(), -angle.sin(), 0.0);
    (moved.0 * downhill.0 + moved.1 * downhill.1 + moved.2 * downhill.2).max(0.0)
}

/// A ground plane and `count` capsules dropped onto it in a loose heap, at assorted angles
/// so that crossings and line contacts both occur.
fn heap(count: usize) -> Skeleton {
    let mut s = Skeleton::new();
    s.set_ground((0.0, 1.0, 0.0), 0.0);
    for i in 0..count {
        let n = i as f64;
        let mut body = Body::capsule(
            4.0,
            0.08,
            0.4,
            (
                0.30 * (n * 1.7).sin(),
                0.2 + 0.11 * n,
                0.30 * (n * 2.3).cos(),
            ),
        );
        let axis = (1.0, 0.3 * (n * 0.9).sin(), 0.7 * (n * 1.3).cos());
        let norm = speed(axis);
        body.orientation = rs_physics::models::Quaternion::from_axis_angle(
            (axis.0 / norm, axis.1 / norm, axis.2 / norm),
            0.4 * n,
        );
        s.add_body(body);
    }
    s
}

/// A heap stepped `steps` times, reduced to the exact state of every body. Bit patterns
/// rather than floats, so that a comparison is a comparison and not a tolerance.
fn settled_heap_state(steps: usize) -> Vec<(u64, u64, u64, u64, u64, u64)> {
    let mut s = crowded_heap();
    let mut ever_parallel = false;
    for _ in 0..steps {
        s.step(DT, G, 8);
        ever_parallel |= s.solved_in_parallel();
    }
    assert!(
        ever_parallel || rayon::current_num_threads() == 1,
        "not one of the {steps} steps divided its work across lanes, so whatever this run \
         is being compared against, both sides of the comparison ran on one thread",
    );
    (0..s.len())
        .map(|i| {
            let p = s.position(i);
            let v = s.velocity(i);
            (
                p.0.to_bits(),
                p.1.to_bits(),
                p.2.to_bits(),
                v.0.to_bits(),
                v.1.to_bits(),
                v.2.to_bits(),
            )
        })
        .collect()
}



/// A heap in which bodies are **retired while it is running**, reduced to the exact state
/// of every body and to the step each one was taken out at.
///
/// The rule that decides is the caller's, which is the point: the crate holds no
/// threshold, so the figure is here. It is a little over forty times a body's own weight,
/// which on this fixture takes out eleven of the forty -- enough that the retirements
/// change the answer, and not so many that what is being compared is two empty scenes.
///
/// **The schedule is part of the state.** Which body was retired at which step is decided
/// by a number the solve produced, so a run that diverged by a bit anywhere would retire a
/// different body a step earlier and the two runs would not even be comparable. Returning
/// it alongside the positions is what makes this a sharper determinism test than the one
/// without retirements rather than a weaker one.
/// The side of the square [`crowded_heap`] drops bodies over, and how many it stacks on
/// each square of it.
///
/// Sized so that a pass clears the crate's internal floor on going parallel by a wide
/// margin for the whole run rather than for a step or two at the start -- which is the
/// difference between a law that exercises the parallel path and one that merely touched
/// it. The number is checked rather than trusted: both readers below assert that the steps
/// they timed actually divided their work, which is the whole reason
/// `Skeleton::solved_in_parallel` is public.
const CROWD_SIDE: usize = 18;
const CROWD_DEEP: usize = 4;

/// **A heap that is wide rather than tall**, for the two laws about determinism.
///
/// [`heap`] drops its bodies down a single narrow column, which suits the laws that use it
/// -- but a column is a queue: the bodies meet the floor a few at a time, so however many
/// are in it, the number touching *at once* stays around one per body and a pass of it
/// never reaches the size at which the solve is divided across lanes. Measured, a column of
/// two thousand crossed the floor on its first step and on none of the two hundred and
/// ninety-nine after it.
///
/// So this drops them over a square instead, a few deep, and they collapse into a broad
/// heap that keeps thousands of contacts live for the whole run. That matters here and
/// nowhere else: these two laws are the ones asserting that the answer does not depend on
/// how many threads computed it, and they are therefore the guard on the raw-pointer
/// scatter in the crate's `scatter` module. A guard that runs the serial path is not a
/// guard at all -- which is what they did, on a heap of forty, for as long as they have
/// existed.
fn crowded_heap() -> Skeleton {
    let mut s = Skeleton::new();
    s.set_ground((0.0, 1.0, 0.0), 0.0);
    for high in 0..CROWD_DEEP {
        for row in 0..CROWD_SIDE {
            for column in 0..CROWD_SIDE {
                let n = (high * CROWD_SIDE * CROWD_SIDE + row * CROWD_SIDE + column) as f64;
                let mut body = Body::capsule(
                    4.0,
                    0.08,
                    0.4,
                    (
                        column as f64 * 0.55 + 0.02 * (n * 1.7).sin(),
                        0.3 + 0.55 * high as f64,
                        row as f64 * 0.55 + 0.02 * (n * 2.3).cos(),
                    ),
                );
                // Turned every which way, so the heap collapses into something disorderly
                // rather than into a lattice that would settle without ever touching.
                let axis = (1.0, 0.3 * (n * 0.9).sin(), 0.7 * (n * 1.3).cos());
                let norm = speed(axis);
                body.orientation = rs_physics::models::Quaternion::from_axis_angle(
                    (axis.0 / norm, axis.1 / norm, axis.2 / norm),
                    0.4 * n,
                );
                s.add_body(body);
            }
        }
    }
    s
}

fn ploughed_heap_state(steps: usize) -> Vec<(u64, u64, u64, u64, u64, u64, usize)> {
    /// Newtons. See above: the caller's judgement, stated in the caller.
    const CRUSHED: f64 = 1600.0;

    let mut s = crowded_heap();
    let mut ever_parallel = false;
    let mut retired_at = vec![usize::MAX; s.len()];
    for step in 0..steps {
        s.step(DT, G, 8);
        ever_parallel |= s.solved_in_parallel();
        for i in 0..s.len() {
            if !s.is_retired(i) && s.normal_load(i) > CRUSHED {
                assert!(s.retire(i), "a live body refused to be retired");
                retired_at[i] = step;
            }
        }
    }
    assert!(
        ever_parallel || rayon::current_num_threads() == 1,
        "not one of the {steps} steps divided its work across lanes, so both sides of \
         whatever this run is compared against ran on one thread",
    );
    (0..s.len())
        .map(|i| {
            let p = s.position(i);
            let v = s.velocity(i);
            (
                p.0.to_bits(),
                p.1.to_bits(),
                p.2.to_bits(),
                v.0.to_bits(),
                v.1.to_bits(),
                v.2.to_bits(),
                retired_at[i],
            )
        })
        .collect()
}

/// **Retiring bodies while the simulation runs costs neither determinism.**
///
/// Both determinism laws again over a run that takes bodies out of the solve as it goes.
/// Retirement is the one operation here that changes the *structure* of the step --
/// joints leave, the whole joint set is coloured again, the broad phase's contents change
/// -- and any of that going through a hash, or a set whose order depends on how the work
/// was split, would show up as two runs disagreeing.
///
/// It is a stronger statement than the two laws above rather than a repeat of them,
/// because the retirement schedule is itself derived from what the solve measured: a
/// single bit of divergence anywhere changes which body is taken out and when, and the
/// two runs then diverge structurally rather than numerically. See
/// [`ploughed_heap_state`].
#[test]
fn retiring_bodies_mid_run_is_still_bit_identical_and_still_schedule_free() {
    let first = ploughed_heap_state(300);
    let second = ploughed_heap_state(300);
    let retired = first.iter().filter(|b| b.6 != usize::MAX).count();
    assert!(
        retired > 0 && retired < first.len(),
        "the fixture retired {retired} of {}, so it is not testing what it claims to",
        first.len(),
    );
    for (i, (a, b)) in first.iter().zip(second.iter()).enumerate() {
        assert_eq!(
            a, b,
            "body {i} ended at {a:?} on the first run and {b:?} on the second; a step \
             that retires bodies depends on more than its inputs",
        );
    }

    let counts = [1usize, 2, 8];
    for threads in counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a thread pool");
        let answer = pool.install(|| ploughed_heap_state(300));
        for (i, (a, b)) in first.iter().zip(answer.iter()).enumerate() {
            assert_eq!(
                a, b,
                "body {i} ended at {a:?} by default and at {b:?} on {threads} threads; \
                 which body is retired, or when, depends on the schedule",
            );
        }
    }
}
