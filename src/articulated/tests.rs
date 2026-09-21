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

    let a = s.bodies[root].position;
    let rb = s.bodies[limb].orientation.rotate_point((0.0, 0.2, 0.0));
    let b = add(s.bodies[limb].position, rb);
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
        s.bodies[root].position,
        (1.0, 2.0, 3.0),
        "a ten-kilo limb hanging off it dragged the anchor",
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
            s.bodies[shin].angular_velocity = (14.0 * sign, 0.0, 0.0);
        }
        s.step(DT, G, 8);
    }

    let angle = hinge_angle(&s, thigh, shin, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0));
    let slack = 0.25;
    assert!(
        angle >= min - slack && angle <= max + slack,
        "the hinge reached {angle:.3} rad, outside [{min}, {max}] by more than the \
         solver's own slack",
    );
}

/// A chain settles instead of running away. XPBD reads its velocities back out of the
/// positions it corrected, so a constraint that fights itself shows up as a body that
/// gains speed every step -- which is the failure mode that ends with a limb at 1e12 m,
/// and this crate's caller has met that one already.
#[test]
fn a_chain_settles_rather_than_gaining_energy() {
    let mut s = Skeleton::new();
    let root = s.add_body(Body::pinned((0.0, 3.0, 0.0)));
    let mut previous = root;
    for i in 0..4 {
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

    for _ in 0..1800 {
        s.step(DT, G, 8);
    }

    for (i, body) in s.bodies.iter().enumerate() {
        let v = length(body.velocity);
        let w = length(body.angular_velocity);
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

/// The angle a `Hinge` is measuring, recovered the same way the solver measures it, so
/// the test is reading the quantity the constraint is about rather than one near it.
fn hinge_angle(
    s: &Skeleton,
    a: usize,
    b: usize,
    axis_a: (f64, f64, f64),
    axis_b: (f64, f64, f64),
) -> f64 {
    let axis = normalized(s.bodies[a].orientation.rotate_point(axis_a)).expect("an axis");
    let reference = perpendicular(axis);
    let in_a = s.bodies[a].orientation.rotate_point(reference);
    let in_b = s
        .bodies[b]
        .orientation
        .rotate_point(rotate_into(reference, axis_b, axis_a));
    dot(cross(in_a, in_b), axis).atan2(dot(in_b, in_a))
}

/// The size of a `Body`, stated so a layout change is a decision rather than a drift.
#[test]
fn a_body_is_the_size_it_looks() {
    let size = std::mem::size_of::<Body>();
    assert_eq!(
        size, 136,
        "a Body is {size} bytes; the layout moved and the cache arithmetic in the module \
         header moved with it",
    );
}
