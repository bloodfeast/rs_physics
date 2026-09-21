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
//!
//! Two things about that are easy to get wrong and were: what the Coulomb limit bounds
//! (the resultant over the step, not the distance it walked -- see [`Spent`]), and where
//! the impulse's couple goes when the contact is a patch rather than a point (see
//! [`patch_arm`]). Each of them on its own is enough to make a body resting below the
//! Coulomb angle creep downhill for ever.

use super::*;

/// What one contact has already spent this step, carried across the solver's passes.
///
/// # Why the tangential half is a vector and the other two are not
///
/// Coulomb's limit is a budget for the whole step rather than for each pass -- see the
/// module header on [`super`] for the two ways of dividing it that do not work. But a
/// *budget* is the wrong word for the tangential half, and using it literally is a
/// defect: the limit is a **cone**, `|P_t| <= friction * P_n`, and what has to stay
/// inside it is the resultant friction impulse over the step, not the distance the
/// resultant walked getting there.
///
/// The difference is not academic, because the direction genuinely reverses between
/// passes. The normal correction acts at a lever arm from the centre of mass, so it turns
/// the body as well as lifting it, and turning it drags the contact points tangentially;
/// the next pass then sees a drift pointing the other way and pushes back. Charging both
/// pushes against one running total spends budget to produce no net impulse at all, and a
/// body near the limit runs out of friction while it is still being asked to hold.
///
/// So the tangential half accumulates as a vector and is projected back onto the cone
/// when it leaves it. A reversal then returns budget rather than burning it, and the only
/// thing bounded is the quantity the law actually bounds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Spent {
    /// Normal impulse applied so far this step, always positive.
    pub normal: f64,
    /// Friction impulse applied so far this step, as a world-space vector: the impulse
    /// the *first* body received. For a pair the second received its negative; against
    /// the ground there is no second, because the ground does not move.
    pub tangential: (f64, f64, f64),
    /// Rolling-resistance angular impulse applied so far this step, as a world-space
    /// vector, and on the first body for the same reason `tangential` is. A cone for the
    /// same reason too: a roll reverses between passes exactly as a slide does.
    pub rolling: (f64, f64, f64),
}

/// The impulse to add this pass, given what the contact wants and what the cone allows,
/// and the new running total.
///
/// `wanted` is the impulse that would cancel the whole remaining drift, and `limit` is the
/// coefficient times the normal impulse spent so far. The running total plus `wanted` is
/// projected back onto the cone if it has left it, and what is applied is the difference
/// -- which is `wanted` exactly whenever the cone is not binding, and points back towards
/// the cone rather than nowhere when it is.
#[inline]
pub(super) fn cone(
    already: (f64, f64, f64),
    wanted: (f64, f64, f64),
    limit: f64,
) -> ((f64, f64, f64), (f64, f64, f64)) {
    let mut total = add(already, wanted);
    let size = length(total);
    if size > limit {
        total = if limit > 0.0 && size > 0.0 {
            scale(total, limit / size)
        } else {
            (0.0, 0.0, 0.0)
        };
    }
    (sub(total, already), total)
}

/// **The arm a friction impulse turns a body about, once the contact patch has taken the
/// part of the couple it can carry.**
///
/// A friction impulse acts on the surface, a distance `dot(r, normal)` from the centre of
/// mass along the normal, so it carries a couple of that times the impulse which tips the
/// body forward over the contact. The couple is real, and for a body touching at a single
/// point it is the whole story: the body tips, and it should.
///
/// A body resting on a *patch* is different, and the difference is not a detail. The
/// normal load is free to move within the patch, and it moves exactly as far as it must to
/// cancel the tipping couple: a patch of half-width `reach` in the direction of the slide
/// can supply a counter-couple of `reach` times the normal impulse it carries, while the
/// largest friction impulse Coulomb allows is `friction` times that same normal impulse.
/// So the couple survives only in the ratio `1 - reach / (friction * offset)`, and where
/// the patch reaches far enough it does not survive at all: the ends of the patch together
/// resist the slide in pure translation, which is what a real contact patch does. The
/// ratio is geometry and the coefficient, with nothing in it to tune, and the normal
/// impulse cancels out of it entirely.
///
/// Leaving the whole couple in is what makes a resting body creep downhill for ever. The
/// tipping that friction induces is repaired by the normal constraints, and those can only
/// push: a body ratchets itself clear of the surface over the solver's passes, its
/// contacts stop reporting any depth, and friction stops acting while the slide it was
/// holding is still there. Measured, that is a steady creep at every angle inside the
/// Coulomb limit, at every iteration count.
///
/// The geometry decides rather than a threshold. A capsule stood on one end has a patch
/// that reaches nowhere, so it still topples; a capsule lying down and sliding *across*
/// its own axis has a patch that reaches nowhere in that direction, so it still rolls.
#[inline]
fn patch_arm(
    r: (f64, f64, f64),
    normal: (f64, f64, f64),
    reach: f64,
    friction: f64,
) -> (f64, f64, f64) {
    let offset = dot(r, normal);
    let couple = friction * offset.abs();
    let survives = if couple > 1e-12 {
        (1.0 - reach / couple).clamp(0.0, 1.0)
    } else {
        1.0
    };
    add(
        sub(r, scale(normal, offset)),
        scale(normal, offset * survives),
    )
}

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
    /// The vector from one end of this pair's contact patch to the other, or zero where
    /// they touch at a point. Both contacts of a line contact carry the same one. See
    /// [`patch_arm`] for what a patch does that a point cannot.
    pub span: (f64, f64, f64),
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
            // The two ends are the two ends of one *patch*, and the friction solve needs
            // to know how far it reaches: see [`patch_arm`].
            let span = scale(axis_a, high - low);
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
                    span,
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

    // Crossed capsules touch at a point, which reaches nowhere.
    [
        touching(
            a,
            b,
            ca,
            cb,
            normal,
            (0.0, 0.0, 0.0),
            position,
            orientation,
            radius,
        ),
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
    span: (f64, f64, f64),
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
        span,
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
    spent: Spent,
) -> ([Correction; 2], Spent) {
    let mut out = [Correction::none(); 2];
    let mut spent = spent;
    let Contact {
        a,
        b,
        local_a,
        local_b,
        normal,
        span,
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
    spent.normal += depth / total;

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
    spent.rolling = resist_rolling(
        &mut out,
        first,
        Some(second),
        normal,
        rolling_resistance * first.radius.min(second.radius),
        spent.normal,
        spent.rolling,
    );

    if friction <= 0.0 {
        return (out, spent);
    }

    let tangential = sub(slid, scale(normal, dot(slid, normal)));
    let Some(direction) = normalized(tangential) else {
        return (out, spent);
    };
    // Each body feels only the part of the friction couple its share of the patch cannot
    // cancel; see [`patch_arm`], which is where the argument is. The reach is the pair's,
    // the offset is each body's own radius.
    let reach = 0.5 * dot(span, direction).abs();
    let arm_a = patch_arm(ra, normal, reach, friction);
    let arm_b = patch_arm(rb, normal, reach, friction);
    let ta = generalised_inverse_mass(&first.now, arm_a, direction);
    let tb = generalised_inverse_mass(&second.now, arm_b, direction);
    let total = ta + tb;
    if total <= 1e-12 {
        return (out, spent);
    }

    // Coulomb, over the step rather than over this pass and as a cone rather than a
    // running total: see [`Spent`]. The impulse that would cancel the whole remaining
    // slide is added to what the step has already applied, the resultant is projected
    // back into the cone if it has left it, and the difference is what goes on now.
    let wanted = scale(direction, -length(tangential) / total);
    let (grip, total_grip) = cone(spent.tangential, wanted, friction * spent.normal);
    if grip == (0.0, 0.0, 0.0) {
        return (out, spent);
    }
    spent.tangential = total_grip;
    accumulate(&mut out[0], &first.now, arm_a, grip, false);
    accumulate(&mut out[1], &second.now, arm_b, scale(grip, -1.0), false);
    (out, spent)
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
    // The vector between the two ends of this body's contact patch with the plane, or
    // zero where it touches at one point. See [`patch_arm`] for what it is for.
    span: (f64, f64, f64),
    spent: Spent,
) -> (Correction, Spent) {
    let mut out = Correction::none();
    let mut spent = spent;
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
    spent.normal += depth / w;

    for (share, free) in [(driven, false), (inherited, true)] {
        if share <= 0.0 {
            continue;
        }
        accumulate(&mut out, &body.now, r, scale(normal, share / w), free);
    }

    // The ground does not turn, so the whole of the resistance lands on the body.
    let mut pair = [out, Correction::none()];
    spent.rolling = resist_rolling(
        &mut pair,
        body,
        None,
        normal,
        rolling_resistance * body.radius,
        spent.normal,
        spent.rolling,
    );
    out = pair[0];

    if friction <= 0.0 {
        return (out, spent);
    }
    let tangential = sub(slid, scale(normal, dot(slid, normal)));
    let Some(direction) = normalized(tangential) else {
        return (out, spent);
    };
    // The couple this impulse leaves on the body once the patch has carried what it can:
    // see [`patch_arm`], which is where the argument is.
    let arm = patch_arm(r, normal, 0.5 * dot(span, direction).abs(), friction);
    let tw = generalised_inverse_mass(&body.now, arm, direction);
    if tw <= 1e-12 {
        return (out, spent);
    }
    // Coulomb over the step, and as a cone; see the pair version and [`Spent`].
    let wanted = scale(direction, -length(tangential) / tw);
    let (grip, total_grip) = cone(spent.tangential, wanted, friction * spent.normal);
    if grip == (0.0, 0.0, 0.0) {
        return (out, spent);
    }
    spent.tangential = total_grip;
    accumulate(&mut out, &body.now, arm, grip, false);
    (out, spent)
}
