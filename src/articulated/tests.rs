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
        &[s.orientation(thigh), s.orientation(shin)],
        0,
        1,
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
/// gains speed every step -- which is the failure mode that ends with a limb at 1e12 m,
/// and this crate's caller has met that one already.
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
#[test]
fn no_two_joints_in_a_colour_share_a_body() {
    let mut s = chain(12);
    // And a branch, so the graph is a skeleton rather than a line.
    let shoulder = s.add_body(Body::capsule(2.0, 0.04, 0.3, (0.4, 2.6, 0.0)));
    s.add_joint(Joint::Ball {
        a: 1,
        b: shoulder,
        anchor_a: (0.0, -0.2, 0.0),
        anchor_b: (0.0, 0.15, 0.0),
    });

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
        size, 136,
        "a Body is {size} bytes; the layout moved and the arithmetic in the module header \
         moved with it",
    );
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
