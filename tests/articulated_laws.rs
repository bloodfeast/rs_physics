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
    let first = settled_heap_state(40, 300);
    let second = settled_heap_state(40, 300);
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
#[test]
fn the_answer_does_not_depend_on_how_many_threads_ran_it() {
    let counts = [1usize, 2, 8];
    let mut answers = Vec::new();
    for threads in counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a thread pool");
        answers.push(pool.install(|| settled_heap_state(40, 300)));
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
        assert!(s.add_joint(Joint::Ball {
            a: previous,
            b: link,
            anchor_a: if i == 0 { (0.0, 0.0, 0.0) } else { (0.0, -0.2, 0.0) },
            anchor_b: (0.0, 0.2, 0.0),
        }));
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

/// **A settled pile wanders; it does not drift.**
///
/// The sharpest statement of the difference, and the reason it is stated as a *ratio*
/// rather than as a distance: watch a settled pile for thirty-two times as long and a body
/// that is drifting has gone thirty-two times as far, while a body that is only jostling
/// against its neighbours has not. No absolute bound can tell those apart -- pick one
/// loose enough for the jostling and it admits a slow drift, pick one tight enough to
/// exclude the drift and the jostling fails it.
///
/// Measured as the median over the pile of how far a body's **surface** moved as a
/// fraction of its own reach, over a fifteen-step window and a four-hundred-and-eighty-step
/// one. A body travelling at a steady speed gives exactly 32.
///
/// The surface and not the centre, because a capsule turning on the spot has moved against
/// whatever it is resting on exactly as much as one that slid, and measuring the centre
/// alone misses it -- in this pile the turning is the larger half. So the centre's travel,
/// plus the angle turned times the reach, except about the capsule's own long axis where it
/// is times the radius: a surface of revolution spinning about its own axis has not gone
/// anywhere, and charging that at the reach makes a resting pile look far worse than it is.
///
/// What this catches is friction that forgives, in any of its forms: a version measuring
/// each step's slip from the start of that step rather than from where the contact stuck
/// scored 17.5, 17.7 and 19.3 at pile sizes twenty, forty and sixty -- which is to say,
/// linear, which is to say every body in the pile was still going. With the slip carried
/// across steps the same three are 7.3, 4.7 and 8.0.
///
/// Three sizes averaged rather than one measured, because which body ends up where in a
/// forty-capsule pile is chaotic and a single draw of it measures the draw. The bound sits
/// between the two means -- 18.1 forgiving, 6.7 holding -- with a little over half again
/// either way.
#[test]
fn a_settled_pile_wanders_but_does_not_drift() {
    let mut total = 0.0;
    let sizes = [20usize, 40, 60];
    for count in sizes {
        let mut s = heap(count);
        for _ in 0..1500 {
            s.step(DT, G, 8);
        }
        let short = median_drift(&mut s, 15);
        let long = median_drift(&mut s, 480);
        assert!(
            short > 0.0,
            "a pile of {count} stopped dead, so this test has measured nothing",
        );
        let ratio = long / short;
        assert!(
            ratio < 14.0,
            "a pile of {count} moved {ratio:.1} times as far in thirty-two times the \
             window ({short:.5} then {long:.5}); at 32 every body is simply still going, \
             and a pile that has settled scores about 7",
        );
        total += ratio;
    }
    let mean = total / sizes.len() as f64;
    assert!(
        mean < 12.0,
        "averaged over pile sizes {sizes:?} the drift grew {mean:.1} times for a \
         thirty-two times longer window; a settled pile scores about 7 and one whose \
         friction forgives its own residual every step scores about 18",
    );
}

/// How far the middle body of a pile's surface moves over `steps`, as a fraction of its own
/// reach. The median rather than the mean, because one body rolling off the top of a pile
/// is not what this is asking about.
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

// -- fixtures ---------------------------------------------------------------------

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
fn settled_heap_state(count: usize, steps: usize) -> Vec<(u64, u64, u64, u64, u64, u64)> {
    let mut s = heap(count);
    for _ in 0..steps {
        s.step(DT, G, 8);
    }
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
