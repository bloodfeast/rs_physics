//! What the articulated solver has to do, stated as the things a skeleton cannot be
//! allowed to do: come apart, fold the wrong way, or gain energy.

use super::*;

const G: (f64, f64, f64) = (0.0, -9.80665, 0.0);
const DT: f64 = 1.0 / 60.0;

/// A ball joint holds its two anchors together. This is the whole claim of the module:
/// the arm does not come off.
#[test]
fn a_ball_joint_keeps_its_two_anchors_in_one_place() {
    let mut s = Skeleton::new();
    let root = s.add_body(Body::pinned((0.0, 2.0, 0.0)));
    let limb = s.add_body(Body::capsule(5.0, 0.05, 0.4, (0.0, 1.6, 0.0)));
    assert!(s.add_joint(Joint::Ball {
        a: root,
        b: limb,
        anchor_a: (0.0, 0.0, 0.0),
        anchor_b: (0.0, 0.2, 0.0),
    }));

    for _ in 0..600 {
        s.step(DT, G, 8);
    }

    let a = s.position(root);
    let b = add(s.position(limb), s.orientation(limb).rotate_point((0.0, 0.2, 0.0)));
    let gap = length(sub(b, a));
    assert!(
        gap < 0.01,
        "ten seconds of hanging opened the joint by {gap:.4} m",
    );
}

/// A pinned body is pinned. Without this the whole thing is a cloud of parts: a skeleton
/// is anchored to something the solver does not own, and if the anchor drifts the corpse
/// walks away from its own body.
#[test]
fn a_pinned_body_does_not_move() {
    let mut s = Skeleton::new();
    let root = s.add_body(Body::pinned((1.0, 2.0, 3.0)));
    let limb = s.add_body(Body::capsule(50.0, 0.1, 0.8, (1.0, 1.2, 3.0)));
    s.add_joint(Joint::Ball {
        a: root,
        b: limb,
        anchor_a: (0.0, 0.0, 0.0),
        anchor_b: (0.0, 0.4, 0.0),
    });

    for _ in 0..300 {
        s.step(DT, G, 8);
    }

    assert_eq!(
        s.position(root),
        (1.0, 2.0, 3.0),
        "a fifty-kilo limb hanging off it dragged the anchor",
    );
}

/// **A knee does not bend backwards.** The range of motion is the reason this module
/// exists rather than a rope with cones bolted on: the limit is a constraint the solver
/// enforces, not a clamp applied to its answer afterwards.
#[test]
fn a_hinge_stays_inside_its_range() {
    let (min, max) = (-0.2_f64, 2.0_f64);
    let mut s = Skeleton::new();
    let thigh = s.add_body(Body::pinned((0.0, 2.0, 0.0)));
    let shin = s.add_body(Body::capsule(4.0, 0.06, 0.4, (0.0, 1.6, 0.0)));
    assert!(s.add_joint(Joint::Hinge {
        a: thigh,
        b: shin,
        anchor_a: (0.0, 0.0, 0.0),
        anchor_b: (0.0, 0.2, 0.0),
        axis_a: (1.0, 0.0, 0.0),
        axis_b: (1.0, 0.0, 0.0),
        min,
        max,
    }));

    // Shoved hard, the way a shell shoves a corpse, and from both directions.
    for step in 0..900 {
        if step % 150 == 0 {
            let sign = if (step / 150) % 2 == 0 { 1.0 } else { -1.0 };
            s.set_angular_velocity(shin, (14.0 * sign, 0.0, 0.0));
        }
        s.step(DT, G, 8);
    }

    let axis = normalized(s.orientation(thigh).rotate_point((1.0, 0.0, 0.0))).expect("an axis");
    let angle = hinge_angle(
        s.orientation(thigh),
        s.orientation(shin),
        axis,
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    );
    let slack = 0.25;
    assert!(
        angle >= min - slack && angle <= max + slack,
        "the hinge reached {angle:.3} rad, outside [{min}, {max}] by more than the \
         solver's own slack",
    );
}

/// A chain settles instead of running away. XPBD reads its velocities back out of the
/// positions it corrected, so a constraint that fights itself shows up as a body that
/// gains speed every step -- and a position-based solver that feeds itself does not drift,
/// it diverges, reaching absurd magnitudes within seconds.
#[test]
fn a_chain_settles_rather_than_gaining_energy() {
    let mut s = chain(4);

    for _ in 0..1800 {
        s.step(DT, G, 8);
    }

    for i in 0..s.len() {
        let v = length(s.velocity(i));
        let w = length(s.angular_velocity(i));
        assert!(
            v.is_finite() && w.is_finite(),
            "body {i} went non-finite: v {v}, w {w}",
        );
        assert!(
            v < 12.0 && w < 40.0,
            "body {i} is still moving after thirty seconds of hanging: v {v:.2} m/s, \
             w {w:.2} rad/s -- the solver is feeding it",
        );
    }
}

/// **The colouring is what makes a step parallel**, so it has to actually separate the
/// joints: no two in a colour may name the same body, or two threads write one body and
/// the answer depends on which got there first.
///
/// Since the solve applies each correction from the thread that computed it, this is not
/// a property of the answer any more -- it is the precondition of the `unsafe` in
/// [`super::scatter`], and it is asserted here directly rather than inferred from a
/// simulation looking right.
///
/// On a **branching** graph, not a chain: a chain colours in two whatever the algorithm
/// does, so it cannot tell a working greedy colouring from a broken one. This is a trunk
/// with limbs hanging off it at every joint, which is the shape a skeleton actually is
/// and where a body reaches degree four.
#[test]
fn no_two_joints_in_a_colour_share_a_body() {
    let mut s = chain(12);
    // A limb off every link, so bodies reach degree four and the greedy colouring has to
    // work for its answer.
    for link in 1..12 {
        let mut previous = link;
        for segment in 0..2 {
            let limb = s.add_body(Body::capsule(
                2.0,
                0.04,
                0.3,
                (0.4 + 0.3 * segment as f64, 3.0 - 0.4 * link as f64, 0.0),
            ));
            s.add_joint(Joint::Ball {
                a: previous,
                b: limb,
                anchor_a: (0.0, -0.2, 0.0),
                anchor_b: (0.0, 0.15, 0.0),
            });
            previous = limb;
        }
    }

    let joints: Vec<Joint> = s.joints().to_vec();
    let colours: Vec<Vec<usize>> = s.colours().to_vec();

    let total: usize = colours.iter().map(|c| c.len()).sum();
    assert_eq!(total, joints.len(), "a joint was dropped or counted twice");

    for (n, colour) in colours.iter().enumerate() {
        let mut seen = Vec::new();
        for &k in colour {
            let (a, b) = joints[k].bodies();
            for body in [a, b] {
                assert!(
                    !seen.contains(&body),
                    "colour {n} has two joints on body {body}; solving it in parallel \
                     would be a data race",
                );
                seen.push(body);
            }
        }
    }
}

/// A chain of twelve colours in two. If greedy colouring ever starts producing a colour
/// per joint the solve is serial again and the parallelism is gone without a test failing
/// anywhere else.
#[test]
fn a_chain_colours_in_two() {
    let mut s = chain(12);
    assert_eq!(
        s.colours().len(),
        2,
        "a simple chain should alternate between two colours; {:?}",
        s.colours().iter().map(|c| c.len()).collect::<Vec<_>>(),
    );
}

/// A joint naming a body that is not there is refused rather than panicking. A solver
/// that runs every frame on data a caller assembled is the wrong place to unwind.
#[test]
fn a_joint_to_nowhere_is_refused() {
    let mut s = Skeleton::new();
    let only = s.add_body(Body::pinned((0.0, 0.0, 0.0)));
    assert!(!s.add_joint(Joint::Ball {
        a: only,
        b: 7,
        anchor_a: (0.0, 0.0, 0.0),
        anchor_b: (0.0, 0.0, 0.0),
    }));
    assert!(
        !s.add_joint(Joint::Ball {
            a: only,
            b: only,
            anchor_a: (0.0, 0.0, 0.0),
            anchor_b: (0.0, 0.0, 0.0),
        }),
        "a body jointed to itself has no solution and a solver should not be asked for one",
    );
    assert!(s.joints().is_empty());
}

/// The size of a `Body`, stated so a layout change is a decision rather than a drift.
#[test]
fn a_body_is_the_size_it_looks() {
    let size = std::mem::size_of::<Body>();
    assert_eq!(
        size, 152,
        "a Body is {size} bytes; the layout moved and the arithmetic in the module header \
         moved with it",
    );
}

// -- contacts ---------------------------------------------------------------------

/// **A body lands on the ground instead of passing through it**, and the height it
/// lands at is not a tolerance to pick: a capsule touching a plane has its axis exactly
/// its own radius above it.
#[test]
fn a_body_comes_to_rest_on_the_ground_rather_than_through_it() {
    let mut s = Skeleton::new();
    floor(&mut s, 0.0);
    let r = 0.1;
    let body = s.add_body(lying(Body::capsule(4.0, r, 0.5, (0.0, 2.0, 0.0)), 0.0));

    for _ in 0..360 {
        s.step(DT, G, 8);
    }

    let y = s.position(body).1;
    assert!(
        (y - r).abs() < 0.01,
        "a capsule of radius {r} resting on a plane at zero should sit at {r}; it is at \
         {y:.4}",
    );
}

/// **And a body lands on another body**, which is the claim the pair contacts make and
/// the ground plane cannot stand in for. Two capsules resting against each other have
/// their axes the sum of their radii apart.
#[test]
fn a_body_comes_to_rest_on_another_body() {
    let mut s = Skeleton::new();
    floor(&mut s, 0.0);
    let (lower_r, upper_r) = (0.12, 0.08);
    s.add_body(lying(
        Body::pinned((0.0, lower_r, 0.0)).shaped(lower_r, 0.6),
        0.0,
    ));
    let upper = s.add_body(lying(Body::capsule(4.0, upper_r, 0.5, (0.0, 1.5, 0.0)), 0.0));

    for _ in 0..600 {
        s.step(DT, G, 8);
    }

    let y = s.position(upper).1;
    let expected = lower_r + lower_r + upper_r;
    assert!(
        (y - expected).abs() < 0.02,
        "a capsule of radius {upper_r} resting on one of radius {lower_r} whose axis is \
         at {lower_r} should sit at {expected}; it is at {y:.4}",
    );
}

/// **An overlapping spawn is separated without being launched.** A position-based solver
/// reads velocity back out of how far things moved, so an overlap removed in a single
/// step is indistinguishable from speed: this is the difference between a heap dropped in
/// as a heap and a heap that detonates on its first frame.
#[test]
fn an_overlapping_spawn_is_separated_without_being_launched() {
    let mut s = Skeleton::new();
    floor(&mut s, 0.0);
    let r = 0.1;
    // Buried most of its own depth in the ground, which is what a careless spawn does.
    let body = s.add_body(lying(Body::capsule(4.0, r, 0.5, (0.0, -0.08, 0.0)), 0.0));

    let mut fastest: f64 = 0.0;
    for _ in 0..120 {
        s.step(DT, G, 8);
        fastest = fastest.max(length(s.velocity(body)));
    }

    assert!(
        fastest < 3.0,
        "a body spawned 0.18 m inside the ground left at {fastest:.2} m/s; recovering an \
         overlap is a correction and must not read back as speed",
    );
    let y = s.position(body).1;
    assert!(
        (y - r).abs() < 0.01,
        "and it should still end up resting at {r}; it is at {y:.4}",
    );
}

/// **Friction holds a body on a slope, and the angle it holds to is the one Coulomb
/// predicts.** A body stays put while `tan(angle) < mu` and slides once it does not, so
/// a single coefficient decides both cases and the test brackets the angle rather than
/// asserting a distance somebody measured once.
///
/// It also pins the thing that was wrong twice. Spending the whole friction limit on
/// every solver pass made this hold past forty degrees; dividing the limit between the
/// passes made it slide at five. Carrying the impulse across the step gets the angle
/// right and makes the answer the same at four iterations and at thirty-two, which is
/// what lets `friction` be a material property instead of a number for one setting.
#[test]
fn a_body_holds_on_a_slope_until_the_angle_beats_the_friction() {
    let mu = 0.5_f64;
    let predicted = mu.atan().to_degrees();
    for iterations in [4usize, 8, 32] {
        let held = slid_down(25.0_f64.to_radians(), mu, iterations);
        let slipped = slid_down(30.0_f64.to_radians(), mu, iterations);
        assert!(
            held < 0.05,
            "at {iterations} iterations, twenty-five degrees is inside the {predicted:.1} \
             degrees a friction of {mu} allows, and the body should hold; it travelled \
             {held:.3} m",
        );
        assert!(
            slipped > 1.0,
            "at {iterations} iterations, thirty degrees is outside {predicted:.1} degrees \
             and the body should run; it travelled {slipped:.3} m against {held:.3} m on \
             the shallower slope",
        );
    }
}

/// Two bones either side of a joint overlap by construction -- they share an anchor --
/// so a contact between them would be the joint and the contact pulling against each
/// other for as long as the body exists.
#[test]
fn bones_that_share_a_joint_do_not_collide() {
    let mut s = Skeleton::new();
    let upper = s.add_body(Body::pinned((0.0, 2.0, 0.0)).shaped(0.08, 0.4));
    let lower = s.add_body(Body::capsule(4.0, 0.08, 0.4, (0.0, 1.6, 0.0)));
    assert!(s.add_joint(Joint::Ball {
        a: upper,
        b: lower,
        anchor_a: (0.0, -0.2, 0.0),
        anchor_b: (0.0, 0.2, 0.0),
    }));

    s.step(DT, G, 8);

    assert_eq!(
        s.contact_count(),
        0,
        "the two halves of a limb overlap at the elbow and must not be pushed apart",
    );
}

/// Two capsules lying along each other touch along a **line**, and the narrow phase has
/// to say so: one point from the middle of that line leaves the pair free to rock about
/// it, and a pile of parallel limbs never comes to rest.
#[test]
fn parallel_capsules_meet_along_a_line_and_crossed_ones_at_a_point() {
    let mut s = Skeleton::new();
    s.add_body(lying(Body::pinned((0.0, 0.0, 0.0)).shaped(0.1, 0.5), 0.0));
    s.add_body(lying(Body::capsule(4.0, 0.1, 0.5, (0.0, 0.19, 0.0)), 0.0));
    s.step(DT, G, 1);
    assert_eq!(
        s.contact_count(),
        2,
        "one capsule lying along another touches it at both ends of the overlap",
    );

    let mut s = Skeleton::new();
    s.add_body(lying(Body::pinned((0.0, 0.0, 0.0)).shaped(0.1, 0.5), 0.0));
    // The same capsule turned across the lower one rather than along it.
    let mut across = Body::capsule(4.0, 0.1, 0.5, (0.0, 0.19, 0.0));
    across.orientation = Quaternion::from_axis_angle((1.0, 0.0, 0.0), std::f64::consts::FRAC_PI_2);
    s.add_body(across);
    s.step(DT, G, 1);
    assert_eq!(
        s.contact_count(),
        1,
        "two capsules crossing touch at one point, and inventing a second would invent a \
         torque with it",
    );
}

/// The same invariant the joint colouring has, for the set that is rebuilt every step:
/// no two contacts in a colour may name the same body, or solving the colour in parallel
/// is two threads writing one body.
#[test]
fn no_two_contacts_in_a_colour_share_a_body() {
    let mut s = pile(40);
    for _ in 0..90 {
        s.step(DT, G, 4);
    }
    assert!(s.contact_count() > 0, "the pile never touched itself");

    let mut placed = 0;
    for (n, colour) in s.contact_colours.iter().enumerate() {
        let mut seen = Vec::new();
        for &k in colour {
            let contact = s.contacts[k];
            for body in [contact.a, contact.b] {
                assert!(
                    !seen.contains(&body),
                    "colour {n} has two contacts on body {body}; solving it in parallel \
                     would be a data race",
                );
                seen.push(body);
            }
            placed += 1;
        }
    }
    assert_eq!(
        placed + s.contact_overflow.len(),
        s.contact_count(),
        "a contact was dropped between generation and colouring",
    );
}

/// **A pile settles, and stays a pile.** Forty bodies dropped on each other end up
/// stacked, still, and above the ground rather than inside it -- and *together*, which
/// is the part that took a second law.
///
/// Coulomb friction alone does not hold a heap of capsules: the contact point of a
/// rolling body is instantaneously still, so there is no sliding for friction to resist
/// and the pile converts its sliding into rolling. Measured without rolling resistance,
/// bodies were still leaving at half a metre a second after thirty seconds and the heap
/// had spread to twenty metres. With it the spread stops at about a metre and a quarter
/// and stays there, which is what a heap of forty bodies of this size covers.
#[test]
fn a_pile_settles_into_a_heap_rather_than_rolling_away() {
    let mut s = pile(40);
    for _ in 0..900 {
        s.step(DT, G, 8);
    }

    let mut moving = 0;
    let mut footprint: f64 = 0.0;
    let mut energy = 0.0;
    for i in 0..s.len() {
        let p = s.position(i);
        assert!(
            p.0.is_finite() && p.1.is_finite() && p.2.is_finite(),
            "body {i} went non-finite: {p:?}",
        );
        assert!(
            p.1 > -0.02,
            "body {i} is at y {:.3}, which is inside the ground",
            p.1,
        );
        let speed = length(s.velocity(i));
        energy += 0.5 * 4.0 * speed * speed;
        if speed > 0.25 {
            moving += 1;
        }
        footprint = footprint.max(length((p.0, 0.0, p.2)));
    }

    assert_eq!(
        moving, 0,
        "bodies are still moving after fifteen seconds; a pile that does not settle is a \
         solver feeding itself",
    );
    assert!(
        energy < 1.0,
        "the pile still holds {energy:.2} J after fifteen seconds",
    );
    assert!(
        footprint < 3.0,
        "the heap has spread to {footprint:.2} m; capsules are rolling out of it",
    );
}

/// The contrast that names the cause: turn rolling resistance off and the same pile is
/// still coming apart. Kept because the failure is silent -- a pile with no rolling
/// resistance settles onto the ground perfectly happily and *then* drifts, so nothing
/// about a single body catches it.
#[test]
fn without_rolling_resistance_the_same_pile_comes_apart() {
    let mut s = pile(40);
    s.set_rolling_resistance(0.0);
    for _ in 0..900 {
        s.step(DT, G, 8);
    }

    let mut footprint: f64 = 0.0;
    for i in 0..s.len() {
        let p = s.position(i);
        footprint = footprint.max(length((p.0, 0.0, p.2)));
    }
    assert!(
        footprint > 4.0,
        "with nothing resisting the roll the heap should be spreading; it is {footprint:.2} m \
         across, so either the test no longer isolates rolling or something else is now \
         holding the pile together",
    );
}

/// **The broad phase must not lose a pair.** A grid that drops one does not fail loudly
/// -- bodies pass through each other now and again, and a pile leaks -- so it is checked
/// against the quadratic answer it exists to avoid, on a set arranged to be awkward for
/// it: a dense heap, nothing axis-aligned, and one body several times the size of the
/// rest, which is what decides the cell.
#[test]
fn the_grid_finds_every_pair_the_quadratic_search_would() {
    let mut s = pile(60);
    // Part way through the collapse, where the heap is dense and nothing is aligned.
    for _ in 0..240 {
        s.step(DT, G, 8);
    }
    s.add_body(lying(Body::capsule(40.0, 0.4, 2.0, (0.0, 0.4, 0.0)), 0.3));
    s.step(DT, G, 8);
    s.rebuild_jointed();

    let mut got = Vec::new();
    let mut reached = Vec::new();
    let mut grid = super::broadphase::Grid::default();
    grid.rebuild(&s.position, &s.radius, &s.half_length);
    // Everything sweeping and nothing already swept, which is the case the grid has to be
    // exhaustive in. The sleeping sweep is a restriction of this one; that it loses
    // nothing is what `a_body_dropped_on_a_sleeping_stack_lands_on_top_of_it` checks.
    let mut everything = super::sleep::BitSet::default();
    everything.resize(s.len(), true);
    let mut nothing = super::sleep::BitSet::default();
    nothing.resize(s.len(), false);
    grid.pairs(
        &s.position,
        &s.inv_mass,
        s.jointed(),
        &everything,
        &nothing,
        &mut got,
        &mut reached,
    );
    got.sort_unstable();

    let mut expected = Vec::new();
    for a in 0..s.len() {
        for b in (a + 1)..s.len() {
            if s.radius[a] <= 0.0 || s.radius[b] <= 0.0 {
                continue;
            }
            if s.inv_mass[a] <= 0.0 && s.inv_mass[b] <= 0.0 {
                continue;
            }
            if s.is_jointed(a, b) {
                continue;
            }
            // The same near-enough test the grid applies once a pair is in hand.
            let apart = sub(s.position[a], s.position[b]);
            let allowed =
                s.radius[a] + s.half_length[a] + s.radius[b] + s.half_length[b];
            if dot(apart, apart) > allowed * allowed {
                continue;
            }
            expected.push((a, b));
        }
    }
    expected.sort_unstable();

    assert!(
        expected.len() > 50,
        "the fixture produced only {} pairs to compare",
        expected.len(),
    );
    assert_eq!(
        got, expected,
        "the grid produced {} pairs where testing every one gives {}",
        got.len(),
        expected.len(),
    );
}

/// The geometry on its own, at the two cases the segment solve is easy to get wrong:
/// segments that cross without meeting, and segments that are parallel, where the system
/// is singular and any point along the overlap is equally close.
#[test]
fn the_closest_points_are_closest() {
    use super::contacts::closest_points_on_segments;

    // A horizontal segment along x, and one along z passing a metre above it.
    let (a, b) = closest_points_on_segments(
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, -1.0),
        (0.0, 1.0, 1.0),
    );
    assert!(length(sub(a, (0.0, 0.0, 0.0))) < 1e-9, "got {a:?}");
    assert!(length(sub(b, (0.0, 1.0, 0.0))) < 1e-9, "got {b:?}");

    // Parallel, offset, and overlapping: the answer is a metre apart wherever it is taken.
    let (a, b) = closest_points_on_segments(
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (-0.5, 1.0, 0.0),
        (1.5, 1.0, 0.0),
    );
    assert!((length(sub(b, a)) - 1.0).abs() < 1e-9, "{a:?} to {b:?}");

    // Past each other entirely: the answer is the near endpoints.
    let (a, b) = closest_points_on_segments(
        (-2.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    );
    assert!((length(sub(b, a)) - 2.0).abs() < 1e-9, "{a:?} to {b:?}");
}

/// **The same simulation twice gives the same bits.**
///
/// A colour is now solved *and applied* from whichever thread the pool handed each
/// constraint to, so the order corrections reach the body arrays in is not the program's
/// to choose any more. Floating-point addition is not associative, so if two constraints
/// in one colour could touch one body the answer would depend on that order -- and it
/// would depend on it *slightly*, which is the failure that never shows up as a crash. A
/// heap would settle differently on a busy machine than on an idle one and every other
/// test here would still pass.
///
/// So this compares raw bit patterns rather than anything with a tolerance: one run is
/// either the same double as the other or it is not. It also asserts that the run really
/// went through the thread pool, because a workload that stayed under
/// [`PARALLEL_FLOOR`] would pass this without testing anything.
#[test]
fn the_same_crowd_twice_lands_on_the_same_bits() {
    let run = || {
        let mut s = crowd(24);
        let mut widest = 0;
        for _ in 0..40 {
            s.step(DT, G, 4);
            widest = widest.max(
                s.contact_colours
                    .iter()
                    .map(|colour| colour.len())
                    .max()
                    .unwrap_or(0),
            );
        }
        let mut bits = Vec::with_capacity(s.len() * 13);
        for i in 0..s.len() {
            let p = s.position(i);
            let q = s.orientation(i);
            let v = s.velocity(i);
            let w = s.angular_velocity(i);
            for value in [
                p.0, p.1, p.2, q.w, q.x, q.y, q.z, v.0, v.1, v.2, w.0, w.1, w.2,
            ] {
                bits.push(value.to_bits());
            }
        }
        (bits, widest, s.awake_count())
    };

    let (first, widest, first_awake) = run();
    let (second, _, second_awake) = run();

    assert_eq!(
        first_awake, second_awake,
        "the two runs put a different number of bodies to sleep, so the settling test is \
         reading something other than the simulation's own state",
    );

    assert!(
        widest >= PARALLEL_FLOOR,
        "the biggest contact colour reached {widest}, under the {PARALLEL_FLOOR} at which \
         a colour goes to the thread pool -- this ran entirely on one thread and proves \
         nothing about the parallel apply",
    );
    let differing = first
        .iter()
        .zip(second.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing, 0,
        "{differing} of {} doubles came out different on the second run of an identical \
         simulation; the solve is depending on the order the thread pool chose",
        first.len(),
    );
}

/// A slab of capsules packed closer together than their own length, on the ground, at
/// assorted angles.
///
/// Dense on purpose: the contact colouring has to produce sets of hundreds before any of
/// them crosses [`PARALLEL_FLOOR`], and a loose pile never does.
fn crowd(side: usize) -> Skeleton {
    let mut s = Skeleton::new();
    floor(&mut s, 0.0);
    for i in 0..side {
        for j in 0..side {
            let n = (i * side + j) as f64;
            let position = (
                0.18 * i as f64,
                0.12 + 0.03 * (n * 0.7).sin(),
                0.18 * j as f64,
            );
            let mut body = Body::capsule(4.0, 0.08, 0.4, position);
            body.orientation = Quaternion::from_axis_angle(
                normalized((1.0, 0.3 * (n * 0.9).sin(), 0.7 * (n * 1.3).cos())).expect("an axis"),
                0.4 * n,
            );
            s.add_body(body);
        }
    }
    s
}

// -- sleeping ---------------------------------------------------------------------

/// **A pile that has arrived stops costing anything.** The whole claim of [`super::sleep`]:
/// a heap spends nearly all of its life settled, and a settled heap that is still being
/// solved is the largest single piece of wasted work in the step.
#[test]
fn a_settled_stack_leaves_the_simulation() {
    let mut s = stack(5);
    let mut slept = None;
    for step in 1..=900 {
        s.step(DT, G, 8);
        if s.awake_count() == 0 {
            slept = Some(step);
            break;
        }
    }
    let slept = slept.unwrap_or_else(|| {
        panic!(
            "a stack of five dropped on the ground was still being solved after fifteen \
             seconds; {} of 5 bodies awake",
            s.awake_count()
        )
    });
    assert!(
        slept > 30,
        "it went to sleep after {slept} steps, which is less than the time it takes to \
         fall and stop moving -- something is calling a body still while it is still \
         arriving",
    );
}

/// **The test the whole mechanism has to pass: a sleeping pile wakes when something
/// lands on it, and the bodies underneath wake too.**
///
/// Waking only what was touched is the failure that looks right and is not: the body that
/// was hit starts moving and drives straight through the ones below it, because they are
/// no longer being solved. The island is the unit for exactly this reason.
#[test]
fn a_sleeping_stack_wakes_all_the_way_down_when_something_lands_on_it() {
    let mut s = stack(5);
    for _ in 0..900 {
        s.step(DT, G, 8);
        if s.awake_count() == 0 {
            break;
        }
    }
    assert_eq!(
        s.awake_count(),
        0,
        "the stack never settled, so this test cannot say anything about waking it",
    );
    let bottom = s.position(0);

    let falling = s.add_body(lying(Body::capsule(4.0, 0.1, 0.5, (0.0, 3.0, 0.0)), 0.0));
    // Still well clear of the stack: nothing should have been disturbed yet.
    for _ in 0..20 {
        s.step(DT, G, 8);
    }
    assert!(
        s.position(falling).1 > 2.0,
        "the fixture's body has already reached the stack, so the next assertion is vacuous",
    );
    assert_eq!(
        s.awake_count(),
        1,
        "only the falling body should be awake while it is still in the air",
    );

    for _ in 0..120 {
        s.step(DT, G, 8);
    }
    for i in 0..5 {
        assert!(
            s.is_awake(i),
            "body {i} of the stack is still asleep after something landed on top of it; \
             the bottom of a stack is as disturbed as the top",
        );
    }
    let moved = length(sub(s.position(0), bottom));
    assert!(
        moved > 0.0,
        "the bottom body did not move at all under the impact, so nothing was actually \
         transmitted through the stack",
    );
}

/// A sleeping body is not merely slow, it is **exactly** where it was left. Anything less
/// and a settled heap creeps for free, which is the artefact sleeping exists to remove.
#[test]
fn a_sleeping_body_does_not_move_at_all() {
    let mut s = stack(1);
    for _ in 0..900 {
        s.step(DT, G, 8);
        if s.awake_count() == 0 {
            break;
        }
    }
    assert_eq!(s.awake_count(), 0, "the body never settled");
    let (where_it_is, how_it_lies) = (s.position(0), s.orientation(0));
    for _ in 0..600 {
        s.step(DT, G, 8);
    }
    assert_eq!(
        s.position(0),
        where_it_is,
        "ten seconds asleep moved it; a sleeping body must be bit-identical, not close",
    );
    assert_eq!(s.orientation(0), how_it_lies, "and it must not have turned");
    assert_eq!(s.velocity(0), (0.0, 0.0, 0.0), "nor read back as moving");
}

/// **A body dropped on a sleeping pile lands on it rather than through it.**
///
/// The broad phase only sweeps outward from the awake set, so a sleeping body is a
/// collider that never looks around itself. If that restriction lost a pair, this is
/// where it would show: the newcomer would find nothing under it.
#[test]
fn a_body_dropped_on_a_sleeping_stack_lands_on_top_of_it() {
    let mut s = stack(3);
    for _ in 0..900 {
        s.step(DT, G, 8);
        if s.awake_count() == 0 {
            break;
        }
    }
    assert_eq!(s.awake_count(), 0, "the stack never settled");
    let top = s.position(2).1;

    let falling = s.add_body(lying(Body::capsule(4.0, 0.1, 0.5, (0.0, 3.0, 0.0)), 0.0));
    for _ in 0..300 {
        s.step(DT, G, 8);
    }
    let landed = s.position(falling).1;
    assert!(
        landed > top,
        "the body finished at {landed:.3} and the stack's top body is at {top:.3}: it \
         went through a pile that was asleep",
    );
}

/// Turning sleeping off has to give back the solver that was there before it, or the
/// feature is not a feature but a change of physics.
#[test]
fn sleeping_does_not_move_where_a_pile_ends_up() {
    let settle = |sleeping: bool| {
        let mut s = pile(40);
        s.set_sleeping(sleeping);
        for _ in 0..900 {
            s.step(DT, G, 8);
        }
        let mut height = 0.0;
        let mut footprint: f64 = 0.0;
        for i in 0..s.len() {
            height += s.position(i).1;
            footprint = footprint.max(length((s.position(i).0, 0.0, s.position(i).2)));
        }
        (height / s.len() as f64, footprint)
    };
    let (asleep_height, asleep_spread) = settle(true);
    let (awake_height, awake_spread) = settle(false);
    // Aggregates rather than positions: a pile of forty is chaotic, and two runs that
    // differ by one body's sleep step diverge in detail while staying the same heap.
    assert!(
        (asleep_height - awake_height).abs() < 0.05,
        "the pile settles at {asleep_height:.3} m with sleeping and {awake_height:.3} m \
         without it",
    );
    assert!(
        (asleep_spread - awake_spread).abs() < 0.5,
        "the pile spreads to {asleep_spread:.3} m with sleeping and {awake_spread:.3} m \
         without it",
    );
}

/// Colouring a joint as it arrives has to land on the assignment a from-scratch greedy
/// pass would have produced, or the parallelism quietly gets worse as a rig is built.
///
/// The invariant is not subtle -- greedy takes the joints in the order they were added
/// either way -- but it is the thing that would break silently if the colour bits ever
/// stopped being per-body, so it is worth stating.
#[test]
fn colouring_joints_as_they_arrive_matches_colouring_them_all_at_once() {
    let mut s = Skeleton::new();
    for i in 0..40 {
        rig(&mut s, i as f64 * 0.4);
    }

    let mut taken: Vec<Vec<usize>> = vec![Vec::new(); s.len()];
    let mut expected: Vec<Vec<usize>> = Vec::new();
    for (index, joint) in s.joints().iter().enumerate() {
        let (a, b) = joint.bodies();
        let mut colour = 0;
        while taken[a].contains(&colour) || taken[b].contains(&colour) {
            colour += 1;
        }
        taken[a].push(colour);
        taken[b].push(colour);
        if colour >= expected.len() {
            expected.resize_with(colour + 1, Vec::new);
        }
        expected[colour].push(index);
    }

    assert_eq!(
        s.colours().len(),
        expected.len(),
        "incremental colouring used {} colours where a full pass uses {}; the colour \
         count is what decides how parallel the solve can be",
        s.colours().len(),
        expected.len(),
    );
    assert_eq!(s.colours(), expected.as_slice());
}

/// A stack of `count` capsules lying flat on the ground, each resting on the one below.
/// The smallest arrangement where "the bodies underneath" means anything.
fn stack(count: usize) -> Skeleton {
    let mut s = Skeleton::new();
    floor(&mut s, 0.0);
    for i in 0..count {
        s.add_body(lying(
            Body::capsule(4.0, 0.1, 0.5, (0.0, 0.11 + 0.21 * i as f64, 0.0)),
            0.0,
        ));
    }
    s
}

/// A pinned root with two chains of three hung off it, which is enough joint degree for
/// greedy colouring to have to make a choice.
fn rig(into: &mut Skeleton, x: f64) {
    let root = into.add_body(Body::pinned((x, 2.0, 0.0)));
    for side in [-1.0f64, 1.0] {
        let mut previous = root;
        for i in 0..3 {
            let body = into.add_body(Body::capsule(
                3.0,
                0.05,
                0.3,
                (x + side * 0.1, 1.7 - 0.35 * i as f64, 0.0),
            ));
            into.add_joint(Joint::Ball {
                a: previous,
                b: body,
                anchor_a: (0.0, -0.15, 0.0),
                anchor_b: (0.0, 0.15, 0.0),
            });
            previous = body;
        }
    }
}

/// A pinned root with `links` capsules hanging off it in a line.
fn chain(links: usize) -> Skeleton {
    let mut s = Skeleton::new();
    let root = s.add_body(Body::pinned((0.0, 3.0, 0.0)));
    let mut previous = root;
    for i in 0..links {
        let y = 3.0 - 0.4 * (i as f64 + 1.0);
        let link = s.add_body(Body::capsule(3.0, 0.05, 0.35, (0.0, y, 0.0)));
        s.add_joint(Joint::Ball {
            a: previous,
            b: link,
            anchor_a: if i == 0 { (0.0, 0.0, 0.0) } else { (0.0, -0.2, 0.0) },
            anchor_b: (0.0, 0.2, 0.0),
        });
        previous = link;
    }
    s
}

/// The ground plane, tilted by `tilt` radians about z so a body laid on it is on a
/// slope of that angle.
fn floor(s: &mut Skeleton, tilt: f64) {
    s.set_ground((-tilt.sin(), tilt.cos(), 0.0), 0.0);
}

/// The same body, turned to lie along x and then tilted by `tilt` about z. A body's
/// length is down its own +Y, so this is the quarter turn that puts it on its side.
fn lying(mut body: Body, tilt: f64) -> Body {
    body.orientation =
        Quaternion::from_axis_angle((0.0, 0.0, 1.0), tilt - std::f64::consts::FRAC_PI_2);
    body
}

/// How far a capsule laid on a slope of `angle` travels down it in two seconds, after
/// half a second to settle onto the surface.
fn slid_down(angle: f64, friction: f64, iterations: usize) -> f64 {
    const R: f64 = 0.1;
    let mut s = Skeleton::new();
    s.set_friction(friction);
    floor(&mut s, angle);

    // Placed on the surface rather than dropped, so what is measured is the slope and
    // the friction rather than the energy of an impact.
    let up = (-angle.sin(), angle.cos(), 0.0);
    let start = scale(up, R + 0.002);
    let body = s.add_body(lying(Body::capsule(4.0, R, 0.5, start), angle));

    for _ in 0..30 {
        s.step(DT, G, iterations);
    }
    let settled = s.position(body);
    for _ in 0..120 {
        s.step(DT, G, iterations);
    }
    length(sub(s.position(body), settled))
}

/// A floor and `count` capsules dropped onto it in a loose heap, at assorted angles so
/// the pile has to resolve crossings and line contacts both.
fn pile(count: usize) -> Skeleton {
    let mut s = Skeleton::new();
    floor(&mut s, 0.0);
    for i in 0..count {
        let n = i as f64;
        // Spread across a small patch and stacked up, so they fall onto each other.
        let position = (
            0.30 * (n * 1.7).sin(),
            0.2 + 0.11 * n,
            0.30 * (n * 2.3).cos(),
        );
        let mut body = Body::capsule(4.0, 0.08, 0.4, position);
        body.orientation = Quaternion::from_axis_angle(
            normalized((1.0, 0.3 * (n * 0.9).sin(), 0.7 * (n * 1.3).cos())).expect("an axis"),
            0.4 * n,
        );
        s.add_body(body);
    }
    s
}
