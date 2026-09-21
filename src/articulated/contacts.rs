//! Capsule contacts: which pairs are touching, where on each surface, and the constraint
//! that stops them overlapping.
//!
//! # Closed form, and the precedent for it is already in this crate
//!
//! Two capsules are two segments carrying a radius, so the entire query is the closest
//! pair of points between two line segments: a clamped least-squares solve, a few dozen
//! flops, no iteration and no tolerance to tune. It is exact.
//!
//! [`crate::interactions::gjk_collision_detection_ex`] is the general convex answer and
//! the right one when the shapes are arbitrary. It is the wrong one here, and that module
//! says so itself -- it carries a hand-written sphere-versus-sphere fast path that skips
//! GJK entirely, for two reasons stated in its own comments: the general path costs more
//! than the closed form, and a sphere drives the simplex degenerate. **A capsule is that
//! same fast path one dimension up**: a sphere swept along a segment, closed form for the
//! same reason and degenerate for the same reason. Following the precedent rather than
//! widening [`crate::models::Shape3D`] is the cheaper and the more accurate answer at once.
//!
//! # Generated once a step, solved every iteration
//!
//! [`Contact`] holds its two surface points in *body* frames and its normal in world
//! space. That split is deliberate:
//!
//! * The points are re-derived from each body's current transform on every solver pass,
//!   so a pass corrects the overlap that is there now rather than the one measured before
//!   any correction had been applied. Re-deriving is two quaternion rotations.
//! * The normal is frozen for the step. Recomputing it mid-solve rotates the contact
//!   frame under the correction, and two bodies then chase each other around a curved
//!   surface without the overlap ever closing.
//!
//! # Friction, because a pile without it is a pile of ball bearings
//!
//! Static friction is positional here, like everything else: the contact points are
//! compared with where they were when the step began, and the tangential part of that
//! drift is removed up to the Coulomb limit of the normal correction. A body resting on
//! another then stays where it was put instead of sliding out from under the load.

use super::*;

/// One touching pair, resolved to a point on each surface.
#[derive(Clone, Copy, Debug)]
pub(super) struct Contact {
    pub a: usize,
    pub b: usize,
    /// The point on `a`'s surface, written in `a`'s own frame, and likewise for `b`. See
    /// the module header for why these are body-frame rather than world.
    pub local_a: (f64, f64, f64),
    pub local_b: (f64, f64, f64),
    /// Unit vector from `a` towards `b`, fixed for the step.
    pub normal: (f64, f64, f64),
}

/// The closest pair of points between segment `p1..q1` and segment `p2..q2`.
///
/// The unclamped solution is a two-by-two linear system; clamping each parameter to its
/// own segment and then re-solving the other is what makes it right for segments rather
/// than for infinite lines. Parallel segments leave the system singular, which the
/// `denom` test catches: every point along the overlap is equally close, and taking
/// `s = 0` picks one of them.
pub(super) fn closest_points_on_segments(
    p1: (f64, f64, f64),
    q1: (f64, f64, f64),
    p2: (f64, f64, f64),
    q2: (f64, f64, f64),
) -> ((f64, f64, f64), (f64, f64, f64)) {
    const TINY: f64 = 1e-12;
    let d1 = sub(q1, p1);
    let d2 = sub(q2, p2);
    let r = sub(p1, p2);
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);

    // Either segment may degenerate to a point, which is a sphere's case and not an
    // error: a body is allowed to be a ball.
    if a <= TINY && e <= TINY {
        return (p1, p2);
    }
    let (s, t) = if a <= TINY {
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = dot(d1, r);
        if e <= TINY {
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = dot(d1, d2);
            let denom = a * e - b * b;
            let mut s = if denom > TINY {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut t = (b * s + f) / e;
            // Clamping t moves the closest point off the second segment's interior, so s
            // is solved again against the clamped t rather than left where it was.
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
            (s, t)
        }
    };
    (add(p1, scale(d1, s)), add(p2, scale(d2, t)))
}

/// The two ends of a body's segment, in world space. A body's length lies along its own
/// +Y, which is the convention [`Body::capsule`] documents.
#[inline]
pub(super) fn segment(
    position: (f64, f64, f64),
    orientation: Quaternion,
    half_length: f64,
) -> ((f64, f64, f64), (f64, f64, f64)) {
    let axis = rotate(orientation, (0.0, half_length, 0.0));
    (sub(position, axis), add(position, axis))
}

/// How far from parallel two axes may be and still be treated as a line contact. The
/// sine of the angle between them, so this is a little under three degrees.
const PARALLEL_SINE: f64 = 0.05;

/// The contact or contacts between two capsules, or nothing if they are clear.
///
/// **Two, when the capsules are near enough parallel.** Two cylinders lying against each
/// other touch along a *line*, and a single point taken from the middle of it leaves them
/// free to rotate about that point: a pile of parallel limbs then rocks forever instead
/// of resting, however many solver iterations it is given. Where the overlap along the
/// axes is real, a contact is emitted at each end of it, and the pair is held flat for
/// the same reason a table needs more than one leg.
pub(super) fn capsule_contact(
    a: usize,
    b: usize,
    position: &[(f64, f64, f64)],
    orientation: &[Quaternion],
    radius: &[f64],
    half_length: &[f64],
) -> [Option<Contact>; 2] {
    let none = [None, None];
    let (pa, qa) = segment(position[a], orientation[a], half_length[a]);
    let (pb, qb) = segment(position[b], orientation[b], half_length[b]);
    let (ca, cb) = closest_points_on_segments(pa, qa, pb, qb);

    let between = sub(cb, ca);
    let reach = radius[a] + radius[b];
    if length(between) >= reach {
        return none;
    }

    // Axes through one another leave no direction to separate along. Falling back to the
    // line between the two centres picks the direction the bodies are actually offset in;
    // when even that is zero the pair is exactly coincident and any axis will serve.
    let normal = normalized(between)
        .or_else(|| normalized(sub(position[b], position[a])))
        .unwrap_or((0.0, 1.0, 0.0));

    // The line case, when both are real segments and the axes nearly agree.
    let axis_a = rotate(orientation[a], (0.0, 1.0, 0.0));
    let axis_b = rotate(orientation[b], (0.0, 1.0, 0.0));
    if half_length[a] > 1e-9
        && half_length[b] > 1e-9
        && length(cross(axis_a, axis_b)) < PARALLEL_SINE
    {
        // Where b's ends fall along a's axis, and how much of a they overlap.
        let along = |p: (f64, f64, f64)| dot(sub(p, position[a]), axis_a);
        let (first, second) = (along(pb), along(qb));
        let low = first.min(second).max(-half_length[a]);
        let high = first.max(second).min(half_length[a]);
        if high - low > 1e-6 {
            let mut out = none;
            for (slot, s) in [low, high].into_iter().enumerate() {
                let point_a = add(position[a], scale(axis_a, s));
                let t = dot(sub(point_a, position[b]), axis_b)
                    .clamp(-half_length[b], half_length[b]);
                let point_b = add(position[b], scale(axis_b, t));
                out[slot] = touching(
                    a,
                    b,
                    point_a,
                    point_b,
                    normal,
                    position,
                    orientation,
                    radius,
                );
            }
            if out[0].is_some() || out[1].is_some() {
                return out;
            }
        }
    }

    [
        touching(a, b, ca, cb, normal, position, orientation, radius),
        None,
    ]
}

/// One contact from a pair of points on the two axes, or `None` if the capsules are clear
/// of one another there. The surface points are carried into each body's own frame so
/// that a later pass can ask where they have moved to.
#[allow(clippy::too_many_arguments)]
fn touching(
    a: usize,
    b: usize,
    axis_point_a: (f64, f64, f64),
    axis_point_b: (f64, f64, f64),
    normal: (f64, f64, f64),
    position: &[(f64, f64, f64)],
    orientation: &[Quaternion],
    radius: &[f64],
) -> Option<Contact> {
    let surface_a = add(axis_point_a, scale(normal, radius[a]));
    let surface_b = sub(axis_point_b, scale(normal, radius[b]));
    if dot(sub(surface_a, surface_b), normal) <= 0.0 {
        return None;
    }
    Some(Contact {
        a,
        b,
        local_a: rotate_inv(orientation[a], sub(surface_a, position[a])),
        local_b: rotate_inv(orientation[b], sub(surface_b, position[b])),
        normal,
    })
}

/// Everything one contact wants done, as two corrections. Reads only; the caller applies.
///
/// The same shape as [`super::solve_joint`], and for the same reason: a colour's worth of
/// them is computed in parallel and applied afterwards.
pub(super) fn solve_contact(
    contact: Contact,
    first: &Gathered,
    second: &Gathered,
    friction: f64,
    rolling_resistance: f64,
    spent: (f64, f64, f64),
) -> ([Correction; 2], (f64, f64, f64)) {
    let mut out = [Correction::none(); 2];
    let (mut normal_impulse, mut tangential_impulse, mut rolling_impulse) = spent;
    let Contact {
        a,
        b,
        local_a,
        local_b,
        normal,
    } = contact;
    out[0].body = a;
    out[1].body = b;

    let ra = rotate(first.now.orientation, local_a);
    let rb = rotate(second.now.orientation, local_b);
    let surface_a = add(first.now.position, ra);
    let surface_b = add(second.now.position, rb);

    // Positive when the surfaces have passed through one another along the normal. See
    // the module header for why the normal is the one frozen at generation.
    let depth = dot(sub(surface_a, surface_b), normal);
    if depth <= 0.0 {
        return (out, spent);
    }

    // How far the two surface points have moved relative to one another since the step
    // began. Measured against the stored previous transforms, which the solve does not
    // change, so every pass sees the whole of it rather than the part left over.
    let was_a = add(first.prev_position, rotate(first.prev_orientation, local_a));
    let was_b = add(second.prev_position, rotate(second.prev_orientation, local_b));
    let slid = sub(sub(surface_a, was_a), sub(surface_b, was_b));

    // The normal part of that is how much of this overlap the step itself made, and it is
    // the only part allowed to read back as velocity. See `Correction::free_translation`.
    let driven = dot(slid, normal).clamp(0.0, depth);
    let inherited = depth - driven;

    let wa = generalised_inverse_mass(&first.now, ra, normal);
    let wb = generalised_inverse_mass(&second.now, rb, normal);
    let total = wa + wb;
    if total <= 1e-12 {
        return (out, spent);
    }
    normal_impulse += depth / total;

    for (share, free) in [(driven, false), (inherited, true)] {
        if share <= 0.0 {
            continue;
        }
        let push = scale(normal, share / total);
        accumulate(&mut out[0], &first.now, ra, scale(push, -1.0), free);
        accumulate(&mut out[1], &second.now, rb, push, free);
    }

    // The rolling half of the same law, on the same carried budget. The arm is the
    // smaller radius: the tighter body is the one that rolls.
    rolling_impulse += resist_rolling(
        &mut out,
        first,
        Some(second),
        normal,
        rolling_resistance * first.radius.min(second.radius),
        normal_impulse,
        rolling_impulse,
    );

    if friction <= 0.0 {
        return (out, (normal_impulse, tangential_impulse, rolling_impulse));
    }

    let tangential = sub(slid, scale(normal, dot(slid, normal)));
    let Some(direction) = normalized(tangential) else {
        return (out, (normal_impulse, tangential_impulse, rolling_impulse));
    };
    let ta = generalised_inverse_mass(&first.now, ra, direction);
    let tb = generalised_inverse_mass(&second.now, rb, direction);
    let total = ta + tb;
    if total <= 1e-12 {
        return (out, (normal_impulse, tangential_impulse, rolling_impulse));
    }

    // Coulomb, over the step rather than over this pass: the tangential impulse already
    // spent plus whatever is added now must stay under the coefficient times the normal
    // impulse spent. Within that, the whole slide is removed and the contact holds
    // static; at the limit, what is left over is the pair sliding.
    let wanted = length(tangential) / total;
    let allowed = (friction * normal_impulse - tangential_impulse).max(0.0);
    let spend = wanted.min(allowed);
    if spend <= 0.0 {
        return (out, (normal_impulse, tangential_impulse, rolling_impulse));
    }
    tangential_impulse += spend;
    let grip = scale(direction, spend);
    accumulate(&mut out[0], &first.now, ra, scale(grip, -1.0), false);
    accumulate(&mut out[1], &second.now, rb, grip, false);
    (out, (normal_impulse, tangential_impulse, rolling_impulse))
}

/// One body resting against the ground plane.
///
/// A separate kind from [`Contact`] because the ground is not a body: it has no mass, no
/// inertia and no transform to carry a contact point in, so half of a two-body solve
/// would be multiplications by zero.
#[derive(Clone, Copy, Debug)]
pub(super) struct GroundContact {
    pub body: usize,
    /// The point on the body's surface, in the body's own frame. Body-frame for the same
    /// reason [`Contact`]'s are.
    pub local: (f64, f64, f64),
}

/// Where a capsule meets the plane `dot(normal, p) = distance`, appended to `out`.
///
/// **Up to two, one per end**, and that is the whole reason the ground is a plane rather
/// than a very fat pinned capsule. A capsule lying across a cylinder touches it at a
/// single point however large the cylinder is, and a single point cannot hold a body
/// flat: it is a pencil balanced on its side, and the solver spends its iterations
/// rocking it instead of resting it. Against a plane a lying capsule gets a contact at
/// each end and settles in a handful of passes. A capsule stood on one end still gets
/// one contact, and still topples, which is correct.
pub(super) fn ground_contacts(
    body: usize,
    position: (f64, f64, f64),
    orientation: Quaternion,
    radius: f64,
    half_length: f64,
    normal: (f64, f64, f64),
    distance: f64,
    out: &mut Vec<GroundContact>,
) {
    if radius <= 0.0 {
        return;
    }
    let (low, high) = segment(position, orientation, half_length);
    let ends: [(f64, f64, f64); 2] = [low, high];
    
    for (index, end) in ends.into_iter().enumerate() {
        // A sphere's two ends are the same point, so it gets one contact rather than two
        // of the same one.
        if index == 1 && half_length <= 1e-9 {
            break;
        }
        let surface = sub(end, scale(normal, radius));
        if dot(normal, surface) >= distance {
            continue;
        }
        out.push(GroundContact {
            body,
            local: rotate_inv(orientation, sub(surface, position)),
        });
    }
}

/// What one ground contact wants done. Reads only; the caller applies.
#[allow(clippy::too_many_arguments)]
pub(super) fn solve_ground(
    contact: GroundContact,
    body: &Gathered,
    friction: f64,
    rolling_resistance: f64,
    normal: (f64, f64, f64),
    distance: f64,
    spent: (f64, f64, f64),
) -> (Correction, (f64, f64, f64)) {
    let mut out = Correction::none();
    let (mut normal_impulse, mut tangential_impulse, mut rolling_impulse) = spent;
    let GroundContact { body: index, local } = contact;
    out.body = index;

    let r = rotate(body.now.orientation, local);
    let surface = add(body.now.position, r);
    let depth = distance - dot(normal, surface);
    if depth <= 0.0 {
        return (out, spent);
    }

    let was = add(body.prev_position, rotate(body.prev_orientation, local));
    let slid = sub(surface, was);

    // How much of the overlap this step drove into the plane, which is the only part
    // allowed to read back as velocity. See `Correction::free_translation`.
    let driven = (-dot(slid, normal)).clamp(0.0, depth);
    let inherited = depth - driven;

    let w = generalised_inverse_mass(&body.now, r, normal);
    if w <= 1e-12 {
        return (out, spent);
    }
    normal_impulse += depth / w;

    for (share, free) in [(driven, false), (inherited, true)] {
        if share <= 0.0 {
            continue;
        }
        accumulate(&mut out, &body.now, r, scale(normal, share / w), free);
    }

    // The ground does not turn, so the whole of the resistance lands on the body.
    let mut pair = [out, Correction::none()];
    rolling_impulse += resist_rolling(
        &mut pair,
        body,
        None,
        normal,
        rolling_resistance * body.radius,
        normal_impulse,
        rolling_impulse,
    );
    out = pair[0];

    if friction <= 0.0 {
        return (out, (normal_impulse, tangential_impulse, rolling_impulse));
    }
    let tangential = sub(slid, scale(normal, dot(slid, normal)));
    let Some(direction) = normalized(tangential) else {
        return (out, (normal_impulse, tangential_impulse, rolling_impulse));
    };
    let tw = generalised_inverse_mass(&body.now, r, direction);
    if tw <= 1e-12 {
        return (out, (normal_impulse, tangential_impulse, rolling_impulse));
    }
    // Coulomb over the step; see the pair version for why it is carried rather than
    // re-spent each pass.
    let wanted = length(tangential) / tw;
    let allowed = (friction * normal_impulse - tangential_impulse).max(0.0);
    let spend = wanted.min(allowed);
    if spend <= 0.0 {
        return (out, (normal_impulse, tangential_impulse, rolling_impulse));
    }
    tangential_impulse += spend;
    accumulate(&mut out, &body.now, r, scale(direction, -spend), false);
    (out, (normal_impulse, tangential_impulse, rolling_impulse))
}
