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

/// **A pair that was carrying load keeps its contact through the step it is solved exactly
/// together.**
///
/// The positional solve removes the whole overlap, so a resting pair ends a step touching
/// to within microns; the narrow phase's own test is `distance >= r_a + r_b`, so the next
/// step finds it clear and gives it nothing at all. That is the step the joints get to pull
/// the pair together unopposed, and the module header has the period-two cycle it closes.
///
/// The gap here is the one measured on the bone that carried that cycle: thirty-two
/// microns, a hundredth of what one step of gravity can close.
#[test]
fn a_loaded_pair_keeps_its_contact_when_it_is_solved_exactly_together() {
    let position = [(0.0, 0.0, 0.0), (0.0, 0.2 + 3.19e-5, 0.0)];
    let orientation = [
        // Along x, and across it: a crossed pair, which is the shape a rig's self-contacts
        // have.
        Quaternion::from_axis_angle((0.0, 0.0, 1.0), -std::f64::consts::FRAC_PI_2),
        Quaternion::from_axis_angle((1.0, 0.0, 0.0), std::f64::consts::FRAC_PI_2),
    ];
    let radius = [0.1, 0.1];
    let half_length = [0.5, 0.5];

    let gone = capsule_contact(0, 1, &position, &orientation, &radius, &half_length, false);
    assert_eq!(
        gone.iter().flatten().count(),
        0,
        "a pair clear by thirty-two microns is clear, and a narrow phase that has not been \
         told otherwise says so",
    );

    let kept = capsule_contact(0, 1, &position, &orientation, &radius, &half_length, true);
    let kept: Vec<Contact> = kept.iter().flatten().copied().collect();
    assert_eq!(
        kept.len(),
        1,
        "a pair that carried load last step keeps one contact, at the point it was touching",
    );
    let contact = kept[0];
    let surface_a = add(position[0], rotate(orientation[0], contact.local_a));
    let surface_b = add(position[1], rotate(orientation[1], contact.local_b));
    let depth = dot(sub(surface_a, surface_b), contact.normal);
    assert!(
        (depth + 3.19e-5).abs() < 1e-9,
        "the kept contact reports a depth of {depth:e}, where the pair is clear by 3.19e-5 \
         -- a revived contact has to carry the gap it actually has, or it is a push rather \
         than a constraint waiting to be needed",
    );
    assert!(
        depth < 0.0,
        "a contact at a negative depth does nothing: `solve_contact_normal` returns on it \
         and every other half returns on the normal impulse it did not spend. That is what \
         makes keeping one free while the gap is open",
    );
}

/// **And a patch with one end loaded and the other clear is not given a second constraint.**
///
/// Two near-parallel capsules resting at a slight relative tilt touch at one end of their
/// overlap and are clear at the other, and the loaded end already holds them apart -- the
/// couple about it *is* the tilt. Reviving the clear end adds a constraint to a pair that
/// had one, which is not what a lost contact needs: measured, a settled stack of three
/// shears 0.354 m sideways under it and leans 1.6 degrees, so a body dropped on the stack
/// misses and lands beside it. A revived contact is for the pair that has lost *every*
/// constraint.
#[test]
fn a_patch_with_one_end_loaded_is_not_given_a_second() {
    // A tilt of a milliradian puts one end of the upper capsule half a millimetre into the
    // lower one and the other half a millimetre clear of it.
    let position = [(0.0, 0.0, 0.0), (0.0, 0.1999, 0.0)];
    let orientation = [
        Quaternion::from_axis_angle((0.0, 0.0, 1.0), -std::f64::consts::FRAC_PI_2),
        Quaternion::from_axis_angle((0.0, 0.0, 1.0), -std::f64::consts::FRAC_PI_2 + 1e-3),
    ];
    let radius = [0.1, 0.1];
    let half_length = [0.5, 0.5];

    let plain = capsule_contact(0, 1, &position, &orientation, &radius, &half_length, false);
    assert_eq!(
        plain.iter().flatten().count(),
        1,
        "one end of the overlap is loaded and the other is clear",
    );
    let revived = capsule_contact(0, 1, &position, &orientation, &radius, &half_length, true);
    assert_eq!(
        revived.iter().flatten().count(),
        1,
        "the pair still has a constraint, so nothing is revived",
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

// -- self collision ----------------------------------------------------------------

/// A root with two one-capsule arms at a right angle off it, whose anchors put the two
/// arms' axes 0.113 m apart while their radii sum to 0.12: the joints hold them inside
/// one another and no solve can take them out.
fn folded(into: &mut Skeleton, x: f64) {
    let root = into.add_body(Body::capsule(8.0, 0.1, 0.16, (x, 0.5, 0.0)));
    for (dx, dz) in [(1.0, 0.0), (0.0, 1.0)] {
        let mut arm = Body::capsule(4.0, 0.06, 0.25, (x + dx * 0.205, 0.5, dz * 0.205));
        arm.orientation = Quaternion::from_axis_angle((0.0, 1.0, 0.0), -dz * std::f64::consts::FRAC_PI_2)
            .multiply(&Quaternion::from_axis_angle((0.0, 0.0, 1.0), -std::f64::consts::FRAC_PI_2));
        let arm = into.add_body(arm);
        into.add_joint(Joint::Ball {
            a: root,
            b: arm,
            anchor_a: (dx * 0.08, 0.0, dz * 0.08),
            anchor_b: (0.0, 0.125, 0.0),
        });
    }
}

/// **Turning self-collision off takes away a skeleton's contacts with itself, and takes
/// away nothing else.**
///
/// The failure this catches is the one the implementation invites: labelling by
/// joint-connected component and then testing the labels too loosely, so that two rigs
/// standing in each other stop colliding as well and a crowd falls through itself. The
/// third case is the guard -- the two rigs are built overlapping on purpose.
#[test]
fn self_collision_off_only_takes_away_a_skeleton_s_contacts_with_itself() {
    let mut touching = Skeleton::new();
    folded(&mut touching, 0.0);
    touching.step(DT, G, 8);
    assert!(
        touching.contact_count() > 0,
        "the fixture is meant to hold its two arms inside one another and found no contact",
    );

    let mut apart = Skeleton::new();
    folded(&mut apart, 0.0);
    apart.set_self_collision(false);
    apart.step(DT, G, 8);
    assert_eq!(
        apart.contact_count(),
        0,
        "a skeleton that may not touch itself still reported contacts with itself",
    );

    // Two of them, overlapping each other rather than themselves. The bodies of one are
    // inside the bodies of the other, and those contacts must survive.
    let mut crowd = Skeleton::new();
    folded(&mut crowd, 0.0);
    folded(&mut crowd, 0.05);
    crowd.set_self_collision(false);
    crowd.step(DT, G, 8);
    assert!(
        crowd.contact_count() > 0,
        "two separate skeletons standing inside one another stopped colliding; the \
         component labels are matching across skeletons",
    );
}

/// A body with no joints belongs to no skeleton, so it collides with everything however
/// the switch is set. Without this the labelling could give every loose body the same
/// label and a pile would pass through itself.
#[test]
fn loose_bodies_collide_whatever_self_collision_says() {
    for collide in [true, false] {
        let mut s = stack(3);
        s.set_self_collision(collide);
        // Long enough for the stack to have closed the gap it is built with and be
        // resting on itself.
        for _ in 0..30 {
            s.step(DT, G, 8);
        }
        assert!(
            s.contact_count() > 0,
            "a stack of three unjointed capsules found no contacts with self collision \
             {collide}",
        );
    }
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

    // Whether each body was ever woken, rather than whether it is awake at some chosen
    // later moment. The moment is the wrong thing to ask about: a stack that is solved
    // well enough wakes, rearranges and goes back to sleep inside two seconds, and an
    // assertion about a fixed step count then fails for the solver getting better. What
    // is being claimed is that the disturbance reached the bottom, and that is a claim
    // about the whole interval.
    let mut woke = [false; 5];
    for _ in 0..120 {
        s.step(DT, G, 8);
        for (i, ever) in woke.iter_mut().enumerate() {
            *ever |= s.is_awake(i);
        }
    }
    for (i, ever) in woke.iter().enumerate() {
        assert!(
            ever,
            "body {i} of the stack was never woken after something landed on top of it; \
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

/// **A limb hanging at rest from a pinned anchor goes to sleep.**
///
/// A pinned body is never awake, so it is never *ready* either -- and the edge scan in
/// [`Skeleton::settle`] used to read "not ready" as "still moving" and disqualify whatever
/// was jointed to it. Every skeleton hung off an anchor was then held awake for ever by the
/// stillest thing in the simulation. This limb is built exactly at its own equilibrium, so
/// the assertion below that nothing is moving is not a tolerance: every velocity is bitwise
/// zero and it stayed awake for twelve thousand steps anyway.
///
/// It sleeps on the first step it is allowed to: the settling window for a body of this
/// reach is twelve steps, which is the time it would take to fall its own radius.
#[test]
fn a_limb_hanging_from_an_anchor_goes_to_sleep() {
    let mut s = Skeleton::new();
    let root = s.add_body(Body::pinned((0.0, 2.0, 0.0)));
    let mut previous = root;
    let mut anchor_a = (0.0, 0.0, 0.0);
    for i in 0..3 {
        // Each capsule hangs with its top anchor exactly on the one above it, so the whole
        // limb starts in the pose the joints already agree on.
        let body = s.add_body(Body::capsule(
            4.0,
            0.06,
            0.25,
            (0.0, 2.0 - 0.125 - 0.25 * i as f64, 0.0),
        ));
        s.add_joint(Joint::Ball {
            a: previous,
            b: body,
            anchor_a,
            anchor_b: (0.0, 0.125, 0.0),
        });
        anchor_a = (0.0, -0.125, 0.0);
        previous = body;
    }

    let mut slept = None;
    for step in 1..=600 {
        s.step(DT, G, 8);
        if s.awake_count() == 0 {
            slept = Some(step);
            break;
        }
    }
    let awake = s.awake_count();
    let fastest = (0..s.len())
        .map(|i| length(s.velocity(i)) + length(s.angular_velocity(i)))
        .fold(0.0f64, f64::max);
    assert!(
        slept.is_some(),
        "{awake} of {} bodies were still being solved after ten seconds, with the fastest \
         of them moving at {fastest:.3e}: a limb at rest on a pinned anchor is as settled \
         as anything in this module gets",
        s.len(),
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

/// **The third correction kind changes the speed and not the place.**
///
/// This is the invariant [`scatter::Bodies::apply_velocity`] exists for and the one its
/// `unsafe` rests on being about: it writes `prev_position` and `prev_orientation` and
/// nothing else, so the body is exactly where the positional solve left it and the
/// velocity the step reads out of it is not. If it ever moved `position`, the velocity
/// pass would be a positional correction in disguise -- and
/// `a_body_dropped_on_a_sleeping_stack_lands_on_top_of_it` is the law that would catch it
/// a long way downstream, which is why it is asserted directly here.
///
/// It asserts the arithmetic too, not merely that something moved: the impulse is chosen
/// so the answer is a round number, and the read-back is done exactly as
/// [`Skeleton::read_velocities`] does it.
#[test]
fn a_velocity_correction_changes_the_speed_and_not_the_place() {
    const DT: f64 = 1.0 / 60.0;
    let mass = 4.0;
    let pose = Pose {
        position: (1.0, 2.0, 3.0),
        orientation: Quaternion::from_axis_angle((0.0, 0.0, 1.0), 0.7),
        inv_mass: 1.0 / mass,
        inv_inertia: (1.0 / 0.5, 1.0 / 0.5, 1.0 / 0.5),
        world_inv_inertia: SymMat3::of(
            Quaternion::from_axis_angle((0.0, 0.0, 1.0), 0.7),
            (1.0 / 0.5, 1.0 / 0.5, 1.0 / 0.5),
        ),
    };

    let mut position = vec![pose.position];
    let mut orientation = vec![pose.orientation];
    // Where it came from: a body travelling at 1 m/s along x and not turning.
    let mut prev_position = vec![sub(pose.position, (DT, 0.0, 0.0))];
    let mut prev_orientation = vec![pose.orientation];

    // An impulse of `mass * 0.5` along -x, at the centre, is a velocity change of half a
    // metre a second backwards and no turn at all: see [`Charge::Still`], where the
    // impulse is at the positional scale and the two `dt`s have cancelled.
    let mut correction = Correction::none();
    correction.body = 0;
    accumulate(
        &mut correction,
        &pose,
        (0.0, 0.0, 0.0),
        (-0.5 * mass * DT, 0.0, 0.0),
        Charge::Still,
    );

    {
        let bodies = Bodies::of(
            &mut position,
            &mut orientation,
            &mut prev_position,
            &mut prev_orientation,
        );
        // SAFETY: one body, one correction, one thread. The disjointness the parallel case
        // needs is trivially satisfied, which is the point of testing the invariant here
        // rather than through a colour.
        unsafe { bodies.apply_velocity([correction, Correction::none()]) };
    }

    assert_eq!(
        position[0], pose.position,
        "a velocity correction moved the body; it may only move where the body came from",
    );
    assert_eq!(
        orientation[0], pose.orientation,
        "a velocity correction turned the body; it may only turn where the body came from",
    );

    let read = scale(sub(position[0], prev_position[0]), 1.0 / DT);
    assert!(
        (read.0 - 0.5).abs() < 1e-12 && read.1.abs() < 1e-12 && read.2.abs() < 1e-12,
        "a body at 1 m/s given half a metre a second backwards reads back at {read:?}, \
         where it should read exactly (0.5, 0, 0)",
    );

    // And a couple at an arm turns it without moving it, by the same rule.
    let mut correction = Correction::none();
    correction.body = 0;
    accumulate(
        &mut correction,
        &pose,
        (0.0, 0.25, 0.0),
        (0.0, 0.0, 0.2 * DT),
        Charge::Still,
    );
    let before = (position[0], orientation[0]);
    {
        let bodies = Bodies::of(
            &mut position,
            &mut orientation,
            &mut prev_position,
            &mut prev_orientation,
        );
        // SAFETY: as above.
        unsafe { bodies.apply_velocity([correction, Correction::none()]) };
    }
    assert_eq!(
        (position[0], orientation[0]),
        before,
        "an angular velocity correction moved or turned the body",
    );
    let spun = turned_since(orientation[0], prev_orientation[0]);
    assert!(
        length(spun) > 1e-9,
        "an angular velocity correction left the body reading as not turning at all",
    );
}


// -- what a body is carrying ------------------------------------------------------

/// **The load accessor is calibrated, and this is the calibration.** A capsule lying on
/// the plane and carrying nothing but itself reads its own weight -- not something
/// proportional to it, the number itself.
///
/// That is the derivation in [`Skeleton::normal_load`] run backwards. The body sags
/// `g dt^2` into the plane in a step, the plane drives exactly that back out, so the
/// impulse is `m g dt^2` in the solver's `correction = impulse * inv_mass` convention;
/// dividing by the step twice -- once to reach momentum, once to reach a mean force --
/// leaves `m g`. If the accessor ever reports a raw impulse, or divides once, or sums
/// `normal` instead of `driven`, this number moves and nothing else here would notice.
///
/// **And a body that has gone to sleep reads zero**, which is not a defect but is worth
/// stating where a caller will find it: a step that does not solve a body drives nothing
/// into it, so what this reports is load *arriving*, and a caller whose rule has to see a
/// static load has to keep the bodies awake to see it.
#[test]
fn a_body_resting_on_the_ground_reads_its_own_weight() {
    const MASS: f64 = 4.0;
    let weight = MASS * length(G);

    // Eight draws a relative 1e-12 apart, because one draw of anything that has been
    // through the solve says nothing about the next.
    for draw in 0..8 {
        let mut s = Skeleton::new();
        floor(&mut s, 0.0);
        s.set_sleeping(false);
        let body = s.add_body(lying(
            Body::capsule(MASS, 0.1, 0.5, (0.0, 0.11 * (1.0 + draw as f64 * 1e-12), 0.0)),
            0.0,
        ));
        assert_eq!(
            s.normal_load(body),
            0.0,
            "a body that has never been stepped is carrying something",
        );
        for _ in 0..300 {
            s.step(DT, G, 8);
        }
        let load = s.normal_load(body);
        assert!(
            (load - weight).abs() < 0.01 * weight,
            "draw {draw}: a capsule lying on the plane reads {load:.4} N where its own \
             weight is {weight:.4} N",
        );
    }

    // The same body left to fall asleep. Nothing is being solved, so nothing is being
    // driven into it, and the load it reports is zero rather than stale.
    let mut s = stack(1);
    for _ in 0..900 {
        s.step(DT, G, 8);
    }
    assert_eq!(s.awake_count(), 0, "the fixture did not settle");
    assert_eq!(
        s.normal_load(0),
        0.0,
        "a sleeping body is reporting a load out of a step that never ran",
    );
}

/// **A body with weight on it reads the weight**, and the figure is the one the statics
/// gives rather than merely a larger number.
///
/// A column of three identical capsules: the bottom one is pressed by the plane carrying
/// all three and by the contact above carrying two, so it sums `(m + 2 * 2m) g = 5 m g`.
/// The middle one sums `(2m + m) g = 3 m g` and the top one its own `m g`. That the three
/// come out 5 : 3 : 1 is what says the accessor is adding up normal forces and not
/// something that merely grows with a pile.
///
/// The bottom is also five times what the same capsule reads lying by itself, which is
/// the "rises sharply" half of the claim, stated as a ratio the statics fixes rather than
/// as a number that happens to be big.
///
/// **Over a window rather than at a step**, because what the statics fixes is the mean.
/// A column left awake jostles -- the module header has the whole account of why -- and
/// the instantaneous normal force on the bottom body swings with it: sampled at single
/// steps 900, 1200 and 1500 the same fixture reads 183, 195 and 105 N. Averaged over six
/// hundred steps it reads 4.995, 2.995 and 0.998 of a body's weight on all eight draws,
/// which is the statics to a part in a thousand.
#[test]
fn a_body_under_a_pile_reads_what_the_pile_weighs() {
    const MASS: f64 = 4.0;
    const WINDOW: usize = 600;
    let weight = MASS * length(G);

    for draw in 0..8 {
        let nudge = 1.0 + draw as f64 * 1e-12;
        let mut s = Skeleton::new();
        floor(&mut s, 0.0);
        s.set_sleeping(false);
        for k in 0..3 {
            s.add_body(lying(
                Body::capsule(MASS, 0.1, 0.5, (0.0, (0.11 + 0.21 * k as f64) * nudge, 0.0)),
                0.0,
            ));
        }
        for _ in 0..600 {
            s.step(DT, G, 8);
        }
        let mut total = [0.0f64; 3];
        for _ in 0..WINDOW {
            s.step(DT, G, 8);
            for (i, sum) in total.iter_mut().enumerate() {
                *sum += s.normal_load(i);
            }
        }
        for (i, share) in [5.0, 3.0, 1.0].into_iter().enumerate() {
            let want = share * weight;
            let load = total[i] / WINDOW as f64;
            assert!(
                (load - want).abs() < 0.01 * want,
                "draw {draw}: body {i} of a column of three carries a mean {load:.2} N \
                 where the statics puts {share} of a body's weight, {want:.2} N, on it",
            );
        }
    }
}

/// **The number rises the way an impact does.** A body four times as heavy dropped on to
/// a resting capsule from a metre: the peak load is a hundred times what the capsule was
/// reading before it arrived, and comfortably past the arriving body's own weight, which
/// is the lower bound the mechanics gives -- a body still moving when it lands is being
/// stopped as well as held, and being stopped is the larger half.
#[test]
fn a_body_landed_on_reads_far_more_than_it_was() {
    const MASS: f64 = 4.0;
    const HEAVY: f64 = 16.0;
    let mut s = Skeleton::new();
    floor(&mut s, 0.0);
    s.set_sleeping(false);
    let low = s.add_body(lying(Body::capsule(MASS, 0.1, 0.5, (0.0, 0.11, 0.0)), 0.0));
    for _ in 0..300 {
        s.step(DT, G, 8);
    }
    let resting = s.normal_load(low);

    s.add_body(lying(Body::capsule(HEAVY, 0.1, 0.5, (0.0, 1.2, 0.0)), 0.0));
    let mut peak: f64 = 0.0;
    for _ in 0..120 {
        s.step(DT, G, 8);
        peak = peak.max(s.normal_load(low));
    }
    assert!(
        peak > 10.0 * resting && peak > HEAVY * length(G),
        "a body four times its mass landed on it from a metre and the load went from \
         {resting:.1} N to a peak of {peak:.1} N, which is neither ten times what it was \
         nor past the {:.1} N the arriving body weighs",
        HEAVY * length(G),
    );
}

/// **A badly placed body is not a crushed one**, and this is the test that says the
/// accessor sums the right one of [`Spent`]'s two totals.
///
/// Two capsules spawned deeply inside one another are separated by an enormous normal
/// impulse -- a third of a metre of overlap removed in a single step -- and *nothing is
/// pressing on either of them*. `Spent::normal` counts the whole of that push and would
/// report a spawn as the hardest crush in the scene; `Spent::driven` counts only the
/// overlap the step itself drove, which for an inherited one is zero. See
/// [`solve_contact_normal`], where the split is taken.
#[test]
fn an_overlapping_spawn_is_not_a_crushed_body() {
    let mut s = Skeleton::new();
    s.set_sleeping(false);
    let a = s.add_body(Body::capsule(4.0, 0.2, 0.4, (0.0, 0.0, 0.0)));
    let b = s.add_body(Body::capsule(4.0, 0.2, 0.4, (0.05, 0.0, 0.0)));
    let overlap = 0.4 - 0.05;

    // No gravity and no plane, so the only thing either body can be carrying is the other.
    let opened = {
        let before = length(sub(s.position(b), s.position(a)));
        s.step(DT, (0.0, 0.0, 0.0), 8);
        length(sub(s.position(b), s.position(a))) - before
    };
    assert!(
        opened > 0.5 * overlap,
        "the fixture did not actually separate: one step opened {opened:.4} m of an \
         overlap of {overlap:.4} m, so there was no large normal impulse to be confused \
         by",
    );
    for i in [a, b] {
        assert_eq!(
            s.normal_load(i),
            0.0,
            "recovering from a spawn overlap read as a load on body {i}",
        );
    }

    for _ in 0..30 {
        s.step(DT, (0.0, 0.0, 0.0), 8);
        for i in [a, b] {
            assert_eq!(s.normal_load(i), 0.0, "body {i} is still reading a load");
        }
    }
}

// -- retirement -------------------------------------------------------------------

/// **Retiring a bone in the middle of a limb lets the far part come away**, and leaves the
/// near part hanging.
///
/// This is the whole claim of taking the joints with the body: a smashed bone that still
/// anchors its neighbours is wrong, so the two joints that named it go and the far half is
/// free -- while every joint that did not name it is untouched, so the near half is still
/// a limb and still on the root.
///
/// "Unaffected" is stated as *still jointed and still hanging* rather than as bit-equality
/// with an uncut limb, which would be the wrong claim: the near part was carrying the far
/// part's weight, so taking the far part away changes what it is holding, and it should
/// move. What may not change is that it is attached.
#[test]
fn retiring_a_bone_in_a_limb_lets_the_far_part_come_away() {
    const LINKS: usize = 5;
    const CUT: usize = 3;
    // Half the segment at each end, which is where [`limb`] puts the anchors.
    const ANCHOR: f64 = 0.175;

    let mut s = limb(LINKS);
    for _ in 0..120 {
        s.step(DT, G, 8);
    }
    let fell_from = s.position(CUT + 1).1;

    assert_eq!(s.joints().len(), LINKS);
    assert!(s.retire(CUT), "the bone refused to be retired");
    assert_eq!(
        s.joints().len(),
        LINKS - 2,
        "retiring a bone in the middle of a limb left one of its two joints behind",
    );
    assert!(!s.retire(CUT), "a body was retired twice");

    for _ in 0..120 {
        s.step(DT, G, 8);
    }

    // The near part is still a limb: every surviving joint's two anchors are still one
    // point, including the one between the root and the first bone.
    for joint in s.joints() {
        let (a, b) = joint.bodies();
        let Joint::Ball {
            anchor_a, anchor_b, ..
        } = *joint
        else {
            unreachable!("the fixture is ball joints");
        };
        let at_a = add(s.position(a), s.orientation(a).rotate_point(anchor_a));
        let at_b = add(s.position(b), s.orientation(b).rotate_point(anchor_b));
        let gap = length(sub(at_a, at_b));
        assert!(
            gap < 0.01,
            "the joint between bodies {a} and {b} opened by {gap:.4} m; retiring a bone \
             elsewhere disturbed a joint that never named it",
        );
    }
    // And it is still hanging off the root rather than falling with the far part: the
    // furthest near bone cannot be below the root by more than the chain that holds it.
    let reach = 2.0 * ANCHOR * CUT as f64;
    let hangs = s.position(0).1 - s.position(CUT - 1).1;
    assert!(
        hangs < reach + 0.01,
        "the near part hangs {hangs:.3} m below the root where its chain is {reach:.3} m \
         long; it came off",
    );

    // The far part: nothing holds it up any more, so it falls freely. Two seconds is
    // `0.5 g t^2 = 0.33 m` from rest, and it started with whatever the limb had given it,
    // so the bound is one-sided.
    let fell = fell_from - s.position(CUT + 1).1;
    assert!(
        fell > 0.3,
        "the far part of the limb fell {fell:.3} m in two seconds, where a body released \
         from rest falls 0.33 m; the retired bone is still holding it up",
    );
    // It fell as a piece, not as loose parts: the joint between the two far bones is one
    // of the ones checked above, and this is the distance that would have grown if it had
    // been dropped instead.
    let apart = length(sub(s.position(CUT + 1), s.position(LINKS)));
    assert!(
        apart < 2.0 * ANCHOR * (LINKS - CUT) as f64 + 0.01,
        "the far part came apart as well as coming away: its ends are {apart:.3} m apart",
    );
}

/// **A retired body takes no contacts**: something dropped where it was passes through.
///
/// Not "is solved and produces no correction" -- it is never offered to the narrow phase
/// at all, because [`Skeleton::retire`] takes its radius away and the broad phase's grid
/// holds only bodies with one. The visible consequence is the one a caller cares about:
/// the place it occupied is empty.
#[test]
fn a_retired_body_takes_no_contacts() {
    let mut s = stack(1);
    for _ in 0..300 {
        s.step(DT, G, 8);
    }
    let resting = s.position(0).1;
    assert!(s.retire(0));

    // Dropped straight on to where it was lying.
    let over = s.add_body(lying(
        Body::capsule(4.0, 0.1, 0.5, (0.0, resting + 0.8, 0.0)),
        0.0,
    ));
    for _ in 0..600 {
        s.step(DT, G, 8);
        assert!(
            !s.contacts.iter().any(|c| c.a == 0 || c.b == 0),
            "the narrow phase produced a contact on a retired body",
        );
        assert!(
            !s.ground_contacts.iter().any(|g| g.body == 0),
            "the plane produced a contact on a retired body",
        );
    }

    let landed = s.position(over).1;
    assert!(
        (landed - resting).abs() < 1e-3,
        "the dropped body came to rest at {landed:.4} m where the plane is at \
         {resting:.4} m; it landed on the body that was retired",
    );
}

/// **A retired body never wakes and never appears in a colour.**
///
/// Never wakes is what makes it free: every sweep in the step walks the awake set, so a
/// body that cannot be in it costs nothing per step rather than costing a test. Never
/// coloured is the other half -- a colour is a set of constraints solved in parallel, and
/// a retired body has no constraints of any kind to be in one.
///
/// Stated over a scene that is actively trying to wake it: the retired body is in the
/// middle of a stack, and bodies keep landing on what is left.
#[test]
fn a_retired_body_never_wakes_and_never_appears_in_a_colour() {
    let mut s = stack(3);
    for _ in 0..300 {
        s.step(DT, G, 8);
    }
    assert!(s.retire(1), "the middle of the stack refused to be retired");
    assert!(s.is_retired(1) && !s.is_retired(0) && !s.is_retired(2));

    // **And the caller cannot undo it.** Retirement is permanent by definition, so every
    // door back in is shut: writing the body, pushing it, waking it by hand, or anchoring
    // a new joint to it. Without this a retired body could be given a shape and a mass
    // again and would rejoin the solve with no joints and no history.
    let was = s.body(1);
    s.set_body(1, Body::capsule(4.0, 0.1, 0.5, (0.0, 5.0, 0.0)));
    s.set_velocity(1, (9.0, 9.0, 9.0));
    s.set_angular_velocity(1, (9.0, 9.0, 9.0));
    s.wake(1);
    s.wake_all();
    assert_eq!(s.body(1), was, "a retired body was written back into the solve");
    assert!(!s.is_awake(1), "a retired body was woken by hand");
    assert!(
        !s.add_joint(Joint::Ball {
            a: 0,
            b: 1,
            anchor_a: (0.0, 0.0, 0.0),
            anchor_b: (0.0, 0.0, 0.0),
        }),
        "a joint was anchored to a retired body",
    );

    for step in 0..900 {
        if step % 150 == 0 {
            s.add_body(lying(
                Body::capsule(4.0, 0.1, 0.5, (0.0, 1.4, 0.0)),
                0.0,
            ));
        }
        s.step(DT, G, 8);

        assert!(!s.is_awake(1), "a retired body woke up at step {step}");
        assert!(!s.ready.get(1), "a retired body was found ready to sleep");
        for (colour, set) in s.colours().iter().enumerate() {
            for &k in set {
                let (a, b) = s.joints()[k].bodies();
                assert!(
                    a != 1 && b != 1,
                    "joint colour {colour} names a retired body",
                );
            }
        }
        for (colour, set) in s.contact_colours.iter().enumerate() {
            for &k in set {
                let c = s.contacts[k];
                assert!(
                    c.a != 1 && c.b != 1,
                    "contact colour {colour} names a retired body",
                );
            }
        }
        for &k in s.ground_colours.iter() {
            assert!(
                s.ground_contacts[k].body != 1,
                "the ground colour names a retired body",
            );
        }
    }
}

/// **Whatever was resting on it wakes.** A body that has stopped has stopped *because* of
/// what is under it; take that away without telling it and it stays frozen in mid-air for
/// as long as nothing else touches it.
///
/// The fixture is a stack that has gone all the way to sleep, so there is no other
/// disturbance to confuse it with: the whole skeleton is out of the step, the bottom body
/// is retired, and the two above it have to be back in.
#[test]
fn retiring_a_body_wakes_what_was_resting_on_it() {
    let mut s = stack(3);
    let mut slept = None;
    for step in 1..=3000 {
        s.step(DT, G, 8);
        if s.awake_count() == 0 {
            slept = Some(step);
            break;
        }
    }
    assert!(slept.is_some(), "the stack never settled, so there is nothing to wake");
    let was = [s.position(1).1, s.position(2).1];

    assert!(s.retire(0), "the bottom of the stack refused to be retired");
    assert!(
        s.is_awake(1) && s.is_awake(2),
        "the bodies resting on the retired one are still asleep: {} of {} awake",
        s.awake_count(),
        s.len(),
    );

    for _ in 0..300 {
        s.step(DT, G, 8);
    }
    for (n, before) in was.iter().enumerate() {
        let i = n + 1;
        assert!(
            s.position(i).1 < before - 0.15,
            "body {i} was resting on the retired one at {before:.3} m and is still at \
             {:.3} m; it never found out the support had gone",
            s.position(i).1,
        );
    }
}

/// A limb of `links` capsules hung off a pinned root by ball joints, in space rather than
/// over a plane, so that what a retirement does is the only thing happening to it.
fn limb(links: usize) -> Skeleton {
    let mut s = Skeleton::new();
    // No self-collision: the question is what the joints do, and a limb that folds onto
    // itself would answer it with contacts.
    s.set_self_collision(false);
    s.set_sleeping(false);
    let mut previous = s.add_body(Body::pinned((0.0, 4.0, 0.0)));
    for i in 0..links {
        let body = s.add_body(Body::capsule(
            3.0,
            0.05,
            0.3,
            (0.0, 3.7 - 0.35 * i as f64, 0.0),
        ));
        s.add_joint(Joint::Ball {
            a: previous,
            b: body,
            anchor_a: (0.0, -0.175, 0.0),
            anchor_b: (0.0, 0.175, 0.0),
        });
        previous = body;
    }
    s
}

/// **Retiring a body leaves the colouring in the state it would have been built in**, and
/// a joint added afterwards is still coloured in constant time against a truth.
///
/// This is the invariant the incremental path rests on: `add_joint` decides a colour from
/// two bitmasks alone, which is only right if those bitmasks describe the assignment
/// `colours` actually holds. A removal that cleared the sets without clearing the bits,
/// or replayed them in a different order, would leave the two disagreeing -- and the
/// damage would be a quietly worse colour count rather than a wrong answer, which is the
/// kind of thing no simulation test catches.
///
/// Stated against the same from-scratch greedy pass
/// `colouring_joints_as_they_arrive_matches_colouring_them_all_at_once` uses, run over
/// the joints that survived, and with more joints added after the retirement so that the
/// incremental path has to work from what the replay left behind.
#[test]
fn colouring_after_a_retirement_matches_colouring_what_is_left() {
    let mut s = Skeleton::new();
    for i in 0..12 {
        rig(&mut s, i as f64 * 0.4);
    }
    // A rig is a pinned root with two chains of three, so this is a root and the middle
    // of one of its limbs -- the two shapes whose removal frees a different number of
    // colours.
    assert!(s.retire(0));
    assert!(s.retire(9));

    // And then more joints, which have to be coloured against what the replay left.
    for i in 0..6 {
        let a = 14 + i * 3;
        let b = 15 + i * 3;
        assert!(s.add_joint(Joint::Ball {
            a,
            b,
            anchor_a: (0.0, -0.15, 0.0),
            anchor_b: (0.0, 0.15, 0.0),
        }));
    }

    let mut taken: Vec<Vec<usize>> = vec![Vec::new(); s.len()];
    let mut expected: Vec<Vec<usize>> = Vec::new();
    for (index, joint) in s.joints().iter().enumerate() {
        let (a, b) = joint.bodies();
        assert!(
            !s.is_retired(a) && !s.is_retired(b),
            "joint {index} still names a retired body",
        );
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
        s.colours(),
        expected.as_slice(),
        "after a retirement the colouring is not the one a full greedy pass over the \
         surviving joints would have produced",
    );
}
