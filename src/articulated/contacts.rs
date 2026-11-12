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
//!
//! **And it is solved in a sub-pass of its own**, after every normal correction in the pass
//! rather than alongside its own, which is why each of the two constraints here is two
//! functions: [`solve_contact_normal`] and [`solve_contact_friction`] for a pair,
//! [`solve_ground_normal`] and [`solve_ground_friction`] for the plane. The module header
//! on [`super`] has the argument and the numbers; the short of it is that friction applied
//! next to its own normal is undone by everybody else's before the pass is out.

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
    /// **Normal impulse the bodies were actually handed as momentum**, as opposed to moved
    /// apart for free.
    ///
    /// # It is not a share of `normal`, and it used to say it was
    ///
    /// The positional solve contributes the part of `normal` that the step itself drove, so
    /// over that pass alone it is a share. The **velocity pass then adds to it** -- a
    /// positive `lambda` in [`solve_contact_velocity_normal`] and
    /// [`solve_ground_velocity_normal`] is genuine extra normal impulse -- and does not add
    /// that to `normal`, deliberately: `normal` and `tangential` are the positional solve's
    /// record, which is what [`super::Skeleton::anchor_ground`] reads to decide whether a
    /// patch stuck. So `driven` can and does exceed `normal`, measured at 1.14 times it on a
    /// settled stack.
    ///
    /// That makes it the right quantity for the one thing that reads it from outside --
    /// [`super::Skeleton::normal_load`], which wants the momentum actually delivered and
    /// would under-report a load without the velocity pass's share -- and the **wrong**
    /// quantity for anything that wants a fraction. Nothing may divide by it or by `normal`
    /// expecting a ratio in `[0, 1]`.
    ///
    /// This matters beyond bookkeeping because `driven` is the leading candidate for what a
    /// friction cone should be charged against (see the note above [`Anchor`] on the burial
    /// defect). A cone built on it is building on an impulse, not on a proportion, and its
    /// own doc said otherwise until this was measured.
    pub driven: f64,
    /// Friction impulse applied so far this step, as a world-space vector: the impulse
    /// the *first* body received. For a pair the second received its negative; against
    /// the ground there is no second, because the ground does not move.
    pub tangential: (f64, f64, f64),
    /// Rolling-resistance angular impulse applied so far this step, as a world-space
    /// vector, and on the first body for the same reason `tangential` is. A cone for the
    /// same reason too: a roll reverses between passes exactly as a slide does.
    pub rolling: (f64, f64, f64),
}

/// What one ground patch has spent this step, and **where its load is standing**.
///
/// The second half is here because the ground's two halves are solved in separate
/// sub-passes and the load point belongs to the first of them. A patch's normal load is
/// free to move between its two ends, and [`solve_patch`] is what decides where it ends
/// up; friction and the rolling couple then act *there*, because that is where the normal
/// force is. Recomputing it in the tangential sub-pass would not work and would not be
/// right: by then the normal sub-pass has closed the overlap, so the patch solve sees no
/// depth at either end and answers that nothing is loaded.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Patch {
    pub spent: Spent,
    /// Where the normal load stands, in the body's own frame, as of the last sub-pass
    /// that found any. Body-frame so that it follows the body through the corrections
    /// applied between the two sub-passes.
    pub local_load: (f64, f64, f64),
    /// **The twist the stick memory has taken back this step**, as an angular impulse about the
    /// plane's normal, carried across the passes like the other totals. Its own budget rather
    /// than a share of rolling's: see [`RollMemory`].
    pub twist: f64,
}

/// **Where a ground patch stuck, and the Coulomb budget it stuck under.**
///
/// Friction is otherwise asked to undo the slide since the start of *this* step, so a slip
/// a step fails to remove is forgiven by the next one, which measures from the new
/// position. Over a rig that is being shaken from the inside that forgiveness is the whole
/// drift: each step banks a fraction of a millimetre and nothing ever asks for it back.
/// An anchor is the memory that asks for it back -- the pose the patch was in when it
/// stuck, held while the contact stays inside its cone.
///
/// # `hold`, and why an offset carries its own authority
///
/// The memory cannot be a displacement alone. A body resting on a stack banks a few tenths
/// of a millimetre under its own weight; when something lands on it, the normal impulse for
/// that step is tens of times larger, and Coulomb's limit with it. Redeeming a resting
/// step's slip at an impact's authority is a sideways kick that has nothing to do with the
/// physics -- measured, it knocks a settled stack of three over.
///
/// So an anchor remembers the displacement *and what it was banked at*: `hold` is the
/// smallest Coulomb budget -- `friction` times the normal impulse over a step -- seen since
/// the anchor was set. The stored offset may be redeemed only up to `hold`, whatever the
/// contact could afford today; this step's own slide is answered at today's budget as it
/// always was. The two bounds are different quantities and both are the physics: you may
/// not undo with a hammer what was written down with a feather.
///
/// The minimum rather than the latest, because an offset accumulates over many steps and
/// the parts of it were banked under whatever load was there at the time; the weakest of
/// those is the only bound that is true of all of them. It also fails safe in the direction
/// that matters: a patch being unloaded has its authority fall towards zero and its memory
/// with it, so a body about to be lifted is not held down by what it remembers.
///
/// **What would make this wrong.** A caller whose loads swing by orders of magnitude while
/// a contact genuinely stays stuck -- a body at the bottom of a pile that is being built --
/// gets an anchor pinned to the lightest moment, which is conservative but weaker than the
/// truth, and the residual it leaves is the ordinary forgiven slip. And a contact that
/// slips without the cone reporting it -- which would mean the cone is wrong -- would keep
/// an anchor it has no right to, and the body would be dragged back towards a place it has
/// genuinely left.
#[derive(Clone, Copy, Debug)]
pub(super) struct Anchor {
    pub position: (f64, f64, f64),
    pub orientation: Quaternion,
    /// The smallest Coulomb budget seen while this anchor has been live, as an impulse.
    pub hold: f64,
    /// **The piece of the body that is stuck**, in body coordinates: the point of its
    /// surface that was against the plane when it stuck.
    ///
    /// An anchor has to name a material point rather than a place under the body, because
    /// a body that *rolls* is not sliding and friction has no business resisting it. Track
    /// where the body's surface is over time and rolling shows up as the stuck point
    /// rising off the plane, which is exactly when the memory stops being about the same
    /// piece of ground and is dropped. Anchoring the load point instead -- the place the
    /// patch stands, which stays under the body as it rolls -- charges a rolling capsule
    /// for the whole of its roll: measured, a pile with rolling resistance switched off
    /// spread to 1.9 m instead of the 4 m it spreads to with nothing holding it, which is
    /// the anchor quietly doing rolling resistance's job.
    pub local: (f64, f64, f64),
}

/// **Where two bodies stuck to one another, and the budgets they stuck under**: the pair's
/// half of the stick memory. See [`super::Skeleton::set_stick_memory`] for what it is for
/// and the module header on [`super`] for what was measured.
///
/// One per touching pair rather than one per contact, because a memory per contact remembers
/// a pair's relative rotation out of two moments and was measured to be the shape that fails
/// (see the module header on anchors between bodies). It travels with the pair's first
/// contact, and the list of them is sorted by the pair like [`super::Skeleton::persist_contacts`]'s.
///
/// Both halves are relative and name material points, so a rigid motion the two bodies make
/// together is not a slip and is not asked back.
///
/// # How what it remembers is bounded
///
/// What it may hand back in one step is bounded by the weakest budget it was banked under
/// (`hold`, `turn_hold`, as [`Anchor::hold`] argues) and by one step of gravity's sag,
/// `anchor_reach`, as a distance at the contact and as that distance over the radius as an
/// angle. Both bounds are applied to the memory itself when it is kept at the end of a step,
/// by moving the remembered pose towards the present one until what is left fits: so the
/// most a step can be asked to redeem is what it starts with. That is a different thing from
/// dropping a memory that has outgrown a step, which the header records as worse.
///
/// # Redeemed as a free correction
///
/// Unlike the ground's [`Anchor`], whose stored slip reads back as velocity, what this hands
/// back is charged [`super::Charge::Free`]: it moves the bodies and moves where they came
/// from with them. That is the one variant of the anchors between bodies that stopped the
/// pumping at its source -- a pair that redeems real motion as if it were residue would
/// otherwise hand it back as momentum, and two bodies can then pump one another. It is also
/// what makes the memory spend itself: the stored slip is measured from where the bodies
/// came from, which a free correction moves, so the passes of one step converge on it the
/// way they converge on any other constraint.
#[derive(Clone, Copy, Debug)]
pub(super) struct PairMemory {
    pub a: u32,
    pub b: u32,
    /// The material point on each body that stuck, in its own frame.
    pub local_a: (f64, f64, f64),
    pub local_b: (f64, f64, f64),
    /// `a`'s point minus `b`'s, in `a`'s frame, as they stood when they stuck.
    pub offset: (f64, f64, f64),
    /// The smallest Coulomb budget seen while the slide memory has been live, as an impulse.
    /// Zero when the pair's friction is not remembered this step.
    pub hold: f64,
    /// `conj(q_b) * q_a` when they stuck: the pair's relative orientation.
    pub turn: Quaternion,
    /// The smallest rolling budget, `arm * N`, seen while the rolling memory has been live,
    /// as an angular impulse. Zero when the pair's rolling is not remembered this step.
    pub turn_hold: f64,
}

/// **Where a body stood against the ground when its rolling stuck, and the budget it stuck
/// under**: the rolling half of the ground's stick memory. The sliding half is [`Anchor`].
///
/// What it asks back is the turn since it stuck, in two parts: about axes in the plane, on
/// the rolling cone, and about the normal, on a budget of its own (see [`untwist`]). The
/// first is charged as velocity, like [`Anchor`]'s stored slip and for its measured reason:
/// the plane does not move, so there is no loop to pump, and charged free the old error is
/// never damped out of the body. Measured on a capsule laid across a slope, 64 s after half
/// a second to land, free against velocity: 5 degrees -1e-18 m against 6e-18 m, both still;
/// 10 degrees 4.7e-4 m against 2.6e-6 m, both arriving in the first second and constant from
/// then to the end, where without the memory it creeps 7.3 mm. (Both 10 degree figures were
/// read while the step's own twist went unresisted, at one phase of the cycle that left;
/// with [`untwist`] answering it, charged as velocity, 10 degrees is 2.2e-11 m.) The stored twist is charged
/// free: charged as velocity the leaning pair goes from 0.07 of still to 4.5 and the 10 degree
/// capsule rolls away. The step's own twist is charged as velocity; see [`untwist`].
#[derive(Clone, Copy, Debug)]
pub(super) struct RollMemory {
    pub orientation: Quaternion,
    /// The smallest rolling budget, `arm * N`, seen while it has been live.
    pub hold: f64,
}

/// What a pair's stick memory asks back at the start of this step: how far `a`'s remembered
/// point has slid across `b`'s, flattened onto the contact plane, and how far the pair has
/// rolled about axes in that plane. Both measured from where the bodies came from, which is
/// what the free correction that redeems them moves.
#[inline]
pub(super) fn pair_memory_debt(
    memory: &PairMemory,
    first: &Gathered,
    second: &Gathered,
    normal: (f64, f64, f64),
) -> ((f64, f64, f64), (f64, f64, f64)) {
    let flat = |v: (f64, f64, f64)| sub(v, scale(normal, dot(v, normal)));
    let slid = if memory.hold > 0.0 {
        let was_a = add(first.prev_position, rotate(first.prev_orientation, memory.local_a));
        let was_b = add(second.prev_position, rotate(second.prev_orientation, memory.local_b));
        flat(sub(sub(was_a, was_b), rotate(first.prev_orientation, memory.offset)))
    } else {
        (0.0, 0.0, 0.0)
    };
    let rolled = if memory.turn_hold > 0.0 {
        let held = second.prev_orientation.multiply(&memory.turn);
        flat(turned_since(first.prev_orientation, held))
    } else {
        (0.0, 0.0, 0.0)
    };
    (slid, rolled)
}

/// **The twist a ground patch's stick memory asks back**: the turn about the plane's normal
/// since the body's rolling stuck, taken back on a budget of its own, as a free correction.
///
/// A body lying on the plane is held against sliding at the point its load stands and
/// against rolling about axes in the plane, and nothing holds it against turning about the
/// normal through that point. Friction acting at one point cannot: the load point is where
/// friction acts, so a turn about it costs friction nothing. A real patch resists that turn
/// with friction spread along it, and the most it can resist is `friction * N` times the
/// mean lever of a uniform load over the patch, which for a patch of half-length `h` is
/// `h / 2`. A patch that reaches nowhere -- a sphere, an upright capsule -- resists nothing,
/// which is right.
///
/// Measured on a capsule laid across a 5 degree slope with the rolling memory on: without
/// this it turns on the spot at 8.2e-4 rad/s and its centre creeps 8.5e-9 m a step; with it
/// the travel over 64 s is 2e-17 m. Charging it on the rolling cone instead was measured
/// and is wrong: the cone is on the resultant, so a twist shrinks the share left for the
/// roll, and at 10 degrees the capsule rolled away.
///
/// # The step's own twist first, as motion
///
/// The stored twist is only what earlier steps left. The turn the body makes about the
/// normal *this* step is answered first, on the same budget, and charged
/// [`Charge::Moving`], so friction takes it out of the body's spin. Without it nothing
/// resisted a step's own twist and the free correction turned the body back without
/// touching its spin: measured on a capsule laid across a 10 degree slope, an exact 15-step
/// limit cycle with the body spinning at 0.16 to 0.45 rad/s about the normal and its centre
/// swinging by 3e-4 m, held in place and never dissipated. The law is
/// [`crate::forces::friction::twist_friction`].
#[allow(clippy::too_many_arguments, reason = "private solver helper; a struct for these arguments would only be built to be taken apart")]
fn untwist(
    out: Correction,
    body: &Gathered,
    normal: (f64, f64, f64),
    span: (f64, f64, f64),
    friction: f64,
    normal_impulse: f64,
    taken: &mut f64,
    roll: RollMemory,
) -> Correction {
    use crate::forces::friction::{twist_friction, twist_friction_limit};
    let mut out = out;
    let limit = twist_friction_limit(friction, normal_impulse, 0.5 * length(span));
    let ease = dot(normal, body.now.world_inv_inertia.apply(normal));
    if limit <= 0.0 || ease <= 1e-12 || !body.now.movable() {
        return out;
    }
    // Including what this contact's correction is already turning it by, which is how the
    // slide's own impulse, at a load point off the centre, shows up here.
    let pending = out.rotation.multiply(&body.now.orientation);
    let turned = dot(turned_since(pending, body.prev_orientation), normal);
    let (applied, total) = twist_friction(turned, ease, *taken, limit);
    *taken = total;
    if applied != 0.0 {
        let turn = Quaternion::from_axis_angle(normal, applied * ease);
        out.rotation = renormalized(turn.multiply(&out.rotation));
    }
    let owed = dot(turned_since(body.prev_orientation, roll.orientation), normal);
    let (applied, total) = twist_friction(owed, ease, *taken, limit);
    *taken = total;
    if applied != 0.0 {
        let turn = Quaternion::from_axis_angle(normal, applied * ease);
        out.free_rotation = renormalized(turn.multiply(&out.free_rotation));
    }
    out
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
/// **Coulomb's cone is charged against the wrong quantity, and this is what is known.**
///
/// Not fixed. Recorded here because three fixes were built and measured and all three are
/// worse than the defect, and the next attempt should start from that rather than from the
/// beginning.
///
/// The limit passed to [`cone`] is `friction * spent.normal`, and `spent.normal` counts the
/// whole normal impulse -- including the share spent *unburying* a body that was placed
/// inside something. Friction then scales with how carelessly a body was put down. One
/// step, a capsule sliding at 4 m/s with nothing resting on it, friction 0.5:
///
/// ```text
///   buried 0.00 m   4.0000 -> 3.9183 m/s     the honest weight-driven figure
///   buried 0.01 m   4.0000 -> 3.6183
///   buried 0.05 m   4.0000 -> 2.4183
///   buried 0.20 m   4.0000 -> 0.6883         five sixths of the momentum, to a contact
///                                            force that does not exist
///   buried 1.00 m   4.0000 -> 0.6883         saturated: the cone no longer binds at all
/// ```
///
/// This module already made the argument one accessor over: `Skeleton::normal_load` reads
/// `Spent::driven` and not `Spent::normal`, precisely because "`normal` would make a badly
/// placed body the most crushed thing in the scene". The same objection applies to the
/// cone and was not carried across. [`resist_rolling`]'s limit has it too.
///
/// # The three that did not work, and what each one proves
///
/// The instrument for all of them is the drop family: a capsule dropped on a settled stack
/// of three from twenty-eight heights a tenth of a metre apart, counting how many pass
/// through. It is the right instrument because a single height is one sample of a chaotic
/// family -- `a_body_dropped_on_a_sleeping_stack_lands_on_top_of_it` is that single sample,
/// and it passes and fails for reasons that have nothing to do with the change under test.
/// **Today's baseline is 9 of 28 through**, which is its own open defect: a first contact
/// deeper than a capsule's radius drives two parallel capsules into each other and they
/// never separate.
///
/// * **`driven` alone.** Kills the defect outright -- 3.9183 m/s at *every* burial depth,
///   which is friction becoming depth-independent, which is the physics. But a *revived*
///   contact has its `driven` forced to zero on purpose, its overlap being the solver's own
///   repair work rather than earned momentum, so a settled pair ends up with no friction at
///   all. A body under a pile read 171.5 N where it carries 196.1.
/// * **`driven` plus the inherited overlap capped at `|g| dt^2`**, the sag one step of
///   gravity makes. Right in shape and wrong in magnitude: the residual of a *loaded*
///   contact is larger than one body's sag in proportion to what it carries, so the cap
///   starves exactly the contacts at the bottom of a pile. 194.15 N against 196.13, a one
///   per cent shortfall that was systematic rather than noise -- and it double-counts
///   unless netted against `driven`, since a buried body sags too.
/// * **The causal rule: all of the inherited overlap when `Contact::revived`, none of it
///   otherwise.** This is the one that looked right -- it is the same question `driven`
///   already asks, the residual scales with load because it is the residual, and it passed
///   the burial test and the pile-weight test together. It takes the drop family from 9 of
///   28 through to **28 of 28**. The reason is the plane: a ground contact is never
///   revived, because the plane does not go away and the contact is rebuilt from the body
///   every step, so under this rule the ground keeps only `driven` -- and a body that has
///   settled is not moving, so its `driven` decays towards nothing and ground friction goes
///   with it. The stack slides out from under whatever lands on it.
///
/// # What the next attempt needs
///
/// A ground contact needs the same distinction the pair contacts get, and it cannot borrow
/// `revived` to get it. The question is "was this contact bearing load before this step",
/// and for the plane the crate already keeps something close to the answer in
/// `Skeleton::ground_stuck` and in [`Anchor`]. Whatever carries it, the test of a candidate
/// is all four of these at once, because each of the three above passed some and not
/// others: the burial table, `a_body_under_a_pile_reads_what_the_pile_weighs`, the drop
/// family at no worse than 9 of 28, and `friction_holds_a_slope_and_lets_go_past_coulombs_angle`.
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
pub(super) fn patch_arm(
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
    /// **This constraint came back for a pair that is not overlapping**, because the pair
    /// carried load last step. See [`capsule_contact`], and [`solve_contact_normal`] for
    /// the one thing it changes.
    pub revived: bool,
    /// **Which pair of features this contact came from**, for a pair of prisms, and zero
    /// for everything else. Handed back to the next step so the pair keeps resting on what
    /// it was resting on: see [`super::prism::Feature`].
    ///
    /// It travels here because this is where a contact's identity already lives, and it
    /// fits in the padding after `revived` rather than growing the type -- which this
    /// module's header records the cost of the last time it happened.
    pub feature: u32,
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
///
/// **And one when the pair is clear, if `alive` says it was carrying load last step.** A
/// contact that is only allowed to exist while the surfaces already overlap vanishes the
/// step a resting pair is solved exactly together, and that is a defect rather than an
/// economy: see the module header on [`super`]. The constraint that comes back is inert
/// while the gap is open -- [`solve_contact_normal`] returns on `depth <= 0` and every
/// other half returns on `spent.normal <= 0` -- so it changes nothing at all unless
/// something closes the gap *during* the step, which is exactly the case it is for.
///
/// **What it may not do is hand the bodies momentum**, and that is [`Contact::revived`].
/// See [`solve_contact_normal`]: without it a rig flies.
///
/// # There is no distance bound on it, and that was measured rather than assumed
///
/// The obvious guard is to revive only a pair that is clear by less than the furthest a
/// step can close it -- `|g| dt^2`, which [`Anchor`] already derives as `anchor_reach`.
/// It was built and it is **worse than either extreme**. Settled kinetic energy of a rig at
/// eight passes, twenty-four draws a relative 1e-12 apart, as median / worst / draws over
/// one joule:
///
/// ```text
///   no revival at all        0.42 /   1.37 /  1 of 24
///   bounded at half          0.39 /   1.67 /  7 of 24
///   bounded at one           7.23 / 104.31 / 14 of 24
///   bounded at two           2.62 / 220.02 / 13 of 24
///   no bound                 0.015/   0.037/  0 of 24
/// ```
///
/// A bound is not a weaker revival, it is an **intermittent** one: the constraint appears
/// and disappears as the gap crosses it, and the module header records three separate
/// occasions on which an intermittent constraint is what a rig walks on. Reviving always,
/// or never, is stable; reviving sometimes is not. So the rule stays "it carried load last
/// step", which is a fact about the pair rather than a distance.
///
/// # And a crossed pair gets a patch too, when `patch` says so
///
/// Every other pair -- crossed, or meeting end to end -- touches at a point, and that is
/// ninety-nine contacts in a hundred in a heap. With `patch` on, an overlapping one is
/// given the patch its overlap actually has instead: see [`pair_patch`].
#[allow(
    clippy::too_many_arguments,
    reason = "two bodies' shapes, whether the pair is remembered and which contact model is on, read in the narrow phase's hot loop; a struct would only be built to be taken apart"
)]
pub(super) fn capsule_contact(
    a: usize,
    b: usize,
    position: &[(f64, f64, f64)],
    orientation: &[Quaternion],
    radius: &[f64],
    half_length: &[f64],
    alive: bool,
    patch: bool,
) -> [Option<Contact>; 2] {
    let none = [None, None];
    let (pa, qa) = segment(position[a], orientation[a], half_length[a]);
    let (pb, qb) = segment(position[b], orientation[b], half_length[b]);
    let (ca, cb) = closest_points_on_segments(pa, qa, pb, qb);

    let between = sub(cb, ca);
    let reach = radius[a] + radius[b];
    if length(between) >= reach && !alive {
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
            // The two ends are the two ends of one *patch*, and the friction solve needs
            // to know how far it reaches: see [`patch_arm`].
            let span = scale(axis_a, high - low);
            let end = |s: f64, revive: bool| {
                let point_a = add(position[a], scale(axis_a, s));
                let t = dot(sub(point_a, position[b]), axis_b)
                    .clamp(-half_length[b], half_length[b]);
                let point_b = add(position[b], scale(axis_b, t));
                touching(
                    a,
                    b,
                    point_a,
                    point_b,
                    normal,
                    span,
                    position,
                    orientation,
                    radius,
                    revive,
                )
            };
            // **Asked for as it stands first, and revived only if that leaves the pair
            // with nothing at all.** A patch with one end loaded and the other clear is a
            // pair resting at a slight relative tilt, and it is *already* fully
            // constrained: the loaded end holds the pair apart and the couple about it is
            // what the tilt is. Reviving the clear end of such a patch adds a second
            // constraint to a pair that had one, which is not what a lost contact needs
            // and is measurably wrong -- a settled stack of three shears 0.354 m sideways
            // and leans 1.6 degrees under it, so a body dropped on the stack misses and
            // lands beside it. What a revived contact is for is the pair that has lost
            // *every* constraint, and for a patch that means both of its ends.
            let out = [end(low, false), end(high, false)];
            if out[0].is_some() || out[1].is_some() {
                return out;
            }
            if alive {
                return [end(low, true), end(high, true)];
            }
        }
    }

    // Crossed capsules overlapping by `depth` share a patch, and with `patch` on they are
    // held by its two ends. A pair kept alive while clear has no overlap to derive one from
    // and keeps its one point.
    let depth = reach - length(between);
    if patch && depth > 0.0 {
        let along = |i: usize, c: (f64, f64, f64)| {
            let axis = rotate(orientation[i], (0.0, 1.0, 0.0));
            let s = dot(sub(c, position[i]), axis);
            // On the cylinder's side the closest point is strictly inside the segment; at
            // either end it is on a cap, which is a sphere.
            (axis, s.abs() < half_length[i] - 1e-9)
        };
        let shapes = [along(a, ca), along(b, cb)];
        if let Some(PairPatch { offset, span }) = pair_patch(
            normal,
            depth,
            [radius[a], radius[b]],
            [shapes[0].0, shapes[1].0],
            [shapes[0].1, shapes[1].1],
        ) {
            let end = |sign: f64| {
                let shift = scale(offset, sign);
                touching(
                    a,
                    b,
                    add(ca, shift),
                    add(cb, shift),
                    normal,
                    span,
                    position,
                    orientation,
                    radius,
                    alive,
                )
            };
            return [end(-1.0), end(1.0)];
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
            alive,
        ),
        None,
    ]
}

/// **The patch two curved surfaces share when they overlap by `depth`**: the offset from the
/// closest point to one end of it, and the span from one end to the other, or `None` when
/// there is no overlap to have a patch.
///
/// # Why a crossed pair has a patch at all
///
/// As rigid shapes two crossed capsules touch at one point, and a point has no moment arm:
/// the pair turns about it freely, and a heap whose contacts are nearly all such points
/// rocks and creeps instead of resting. But the solver never holds a pair at zero overlap.
/// Every contact it builds is built from the predicted positions, overlapping by the
/// step's own sag at least, and over that overlap the two surfaces coincide on a region,
/// not at a point. That region is the patch, and its size is geometry.
///
/// # The derivation
///
/// Near the contact, a surface of curvature `k` along a tangent direction `u` falls away
/// from its tangent plane by `k u^2 / 2`. Two surfaces pressed `depth` into each other
/// still overlap wherever the two fall-aways together are less than `depth`:
///
/// ```text
///   (k_a(u) + k_b(u)) u^2 / 2 < depth
/// ```
///
/// A capsule's curvature is `1 / r` across its cylinder and on its caps in every
/// direction, and zero along its cylinder. So in every direction `k_a + k_b` is at most
/// `1 / r_a + 1 / r_b`, which is the two-sphere case, and the patch reaches at least
///
/// ```text
///   rho = sqrt(2 depth r),   r = r_a r_b / (r_a + r_b)
/// ```
///
/// from the closest point whichever way it is measured -- the reduced radius, the same `r`
/// Hertz's contact radius uses. The overlap has no coefficient in it, nothing to tune, and
/// it grows with load because a heavier contact sags deeper.
///
/// # Which way the two ends go
///
/// Two points hold a turn about the axis across them and not the turn about the line
/// through them, so one direction has to be chosen, and the geometry chooses it: the
/// patch is an ellipse, and the ends go along its **long axis**, where the true overlap
/// reaches furthest past `rho` and the two points are surest to be inside it. The long
/// axis is the direction the surfaces fall away from each other slowest, the smallest
/// eigenvector of the summed curvature `M = sum_i c_i` in the tangent plane, with `c_i =
/// p_i p_i^T / r_i` for a cylinder's side (`p_i` the tangent direction across its axis)
/// and `I / r_i` for a cap. Two crossed limbs put the ends along the thinner limb, which is
/// the one a seesaw over the thicker would turn; two spheres have no long axis and any
/// direction is as good.
///
/// Both ends are carried at the closest point's depth: the solve holds the pair on the
/// plane the two surfaces share across the patch, as the ground's patch is held on the
/// ground, so that tipping it one way presses one end and lifts the other.
///
/// # What it does not reach
///
/// A near-parallel pair already has the line patch [`capsule_contact`] gives it, a pair of
/// prisms has the face patch [`super::prism::prism_contact`] clips, and a pair kept alive
/// while clear has no overlap to derive one from. All three are unchanged.
pub(super) fn pair_patch(
    normal: (f64, f64, f64),
    depth: f64,
    radius: [f64; 2],
    axis: [(f64, f64, f64); 2],
    on_side: [bool; 2],
) -> Option<PairPatch> {
    let sum = radius[0] + radius[1];
    if depth <= 0.0 || radius[0] <= 0.0 || radius[1] <= 0.0 {
        return None;
    }
    let rho = (2.0 * depth * radius[0] * radius[1] / sum).sqrt();

    // A tangent basis. Any will do: the eigenvector found in it is a direction in the
    // plane, not a coordinate.
    let seed = if normal.0.abs() < 0.9 { (1.0, 0.0, 0.0) } else { (0.0, 1.0, 0.0) };
    let first = normalized(cross(normal, seed))?;
    let second = cross(normal, first);

    // The summed curvature in that basis, as the symmetric 2x2 [[m00, m01], [m01, m11]].
    let (mut m00, mut m01, mut m11) = (0.0, 0.0, 0.0);
    for i in 0..2 {
        let k = 1.0 / radius[i];
        match normalized(cross(normal, axis[i])).filter(|_| on_side[i]) {
            // Curved across the axis only.
            Some(across) => {
                let (x, y) = (dot(across, first), dot(across, second));
                m00 += k * x * x;
                m01 += k * x * y;
                m11 += k * y * y;
            }
            // A cap, or an axis along the normal: curved every way.
            None => {
                m00 += k;
                m11 += k;
            }
        }
    }
    // The eigenvector of the larger eigenvalue lies at half the angle of (m00 - m11, 2
    // m01); the long axis is the one at right angles to it.
    let half = 0.5 * (2.0 * m01).atan2(m00 - m11);
    let (sin, cos) = half.sin_cos();
    let long = add(scale(first, -sin), scale(second, cos));
    let offset = scale(long, rho);
    Some(PairPatch {
        offset,
        span: scale(offset, 2.0),
    })
}

/// What [`pair_patch`] answers: where one end of the patch is from the closest point (the
/// other is the same distance the other way), and the vector from one end to the other,
/// which is what [`Contact::span`] carries.
#[derive(Clone, Copy, Debug)]
pub(super) struct PairPatch {
    pub offset: (f64, f64, f64),
    pub span: (f64, f64, f64),
}

/// One contact from a pair of points on the two axes, or `None` if the capsules are clear
/// of one another there. The surface points are carried into each body's own frame so
/// that a later pass can ask where they have moved to.
#[allow(clippy::too_many_arguments, reason = "private solver helper; a struct for these arguments would only be built to be taken apart")]
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
    alive: bool,
) -> Option<Contact> {
    let surface_a = add(axis_point_a, scale(normal, radius[a]));
    let surface_b = sub(axis_point_b, scale(normal, radius[b]));
    // A pair that was carrying load last step keeps its constraint even where this end of
    // the patch is clear. The depth it is solved at is re-derived from the bodies' current
    // transforms on every pass, so a negative one here is a constraint that does nothing
    // until the step itself closes the gap. See [`capsule_contact`].
    let depth = dot(sub(surface_a, surface_b), normal);
    if depth <= 0.0 && !alive {
        return None;
    }
    Some(Contact {
        a,
        b,
        local_a: rotate_inv(orientation[a], sub(surface_a, position[a])),
        local_b: rotate_inv(orientation[b], sub(surface_b, position[b])),
        normal,
        span,
        // **The pair is not overlapping and this constraint exists only because it was
        // loaded last step.** Whatever overlap the step then finds here was made by the
        // step, not driven into by the bodies: see [`solve_contact_normal`].
        revived: depth <= 0.0,
        // A capsule has no features to remember: its normal is the direction between two
        // closest points on two segments, which moves smoothly and never jumps.
        feature: 0,
    })
}

/// **Where the two surfaces are now, and how far they have slid past one another since
/// the step began.** Both halves of a pair contact's solve start here.
///
/// Re-derived from the bodies' current transforms every time it is asked for, which is
/// the whole point of a position-based pass: what is corrected is the overlap there is
/// now rather than the one measured before anything had been applied. The slide is
/// measured against the transforms the step *began* with, which the solve never writes
/// to, so every pass sees the whole of it rather than the part left over.
struct Surfaces {
    ra: (f64, f64, f64),
    rb: (f64, f64, f64),
    /// Positive when the surfaces have passed through one another along the normal. See
    /// the module header for why the normal is the one frozen at generation.
    depth: f64,
    /// How far `a`'s surface point has moved relative to `b`'s since the step began.
    slid: (f64, f64, f64),
}

#[inline]
fn surfaces(contact: &Contact, first: &Gathered, second: &Gathered) -> Surfaces {
    let ra = rotate(first.now.orientation, contact.local_a);
    let rb = rotate(second.now.orientation, contact.local_b);
    let surface_a = add(first.now.position, ra);
    let surface_b = add(second.now.position, rb);
    let was_a = add(
        first.prev_position,
        rotate(first.prev_orientation, contact.local_a),
    );
    let was_b = add(
        second.prev_position,
        rotate(second.prev_orientation, contact.local_b),
    );
    Surfaces {
        ra,
        rb,
        depth: dot(sub(surface_a, surface_b), contact.normal),
        slid: sub(sub(surface_a, was_a), sub(surface_b, was_b)),
    }
}

/// **The normal half of one pair contact**: stop the surfaces overlapping, and record what
/// that cost, which is the budget the tangential half is then allowed to spend.
///
/// Reads only; the caller applies. The same shape as [`super::solve_joint`], and for the
/// same reason: a colour's worth of them is computed in parallel and applied afterwards.
pub(super) fn solve_contact_normal(
    contact: Contact,
    first: &Gathered,
    second: &Gathered,
    spent: Spent,
) -> ([Correction; 2], Spent) {
    let mut out = [Correction::none(); 2];
    let mut spent = spent;
    let Contact { a, b, normal, .. } = contact;
    out[0].body = a;
    out[1].body = b;

    let Surfaces {
        ra,
        rb,
        depth,
        slid,
    } = surfaces(&contact, first, second);
    if depth <= 0.0 {
        return (out, spent);
    }

    // The normal part of the slide is how much of this overlap the step itself made, and
    // it is the only part allowed to read back as velocity. See
    // `Correction::free_translation`.
    //
    // **A revived contact may turn none of it into velocity, and that is what makes
    // reviving one safe.** The pair was not overlapping when the narrow phase looked, so
    // the bodies did not drive into anything: whatever overlap this constraint finds was
    // made by the step's own corrections -- the joints pulling a resting pair back
    // together, which is the case revival exists for -- and charging the solver's own
    // repair work as momentum the bodies never earned is how a rig ends up flying. The
    // slide cannot tell the two apart, because the slide includes every correction applied
    // since the step began; the narrow phase can, and it has already said so.
    //
    // It does not close the channel by which "stopped" propagates up a stack, which is
    // what the module header warns any bound on the read-back velocity about. A body that
    // arrives with momentum drives a *genuine* overlap, so its contact is not revived and
    // charges as it always did. Only a pair that was already resting together gets this.
    let driven = if contact.revived && spent.normal > 0.0 {
        0.0
    } else {
        dot(slid, normal).clamp(0.0, depth)
    };
    let inherited = depth - driven;

    let wa = generalised_inverse_mass(&first.now, ra, normal);
    let wb = generalised_inverse_mass(&second.now, rb, normal);
    let total = wa + wb;
    if total <= 1e-12 {
        return (out, spent);
    }
    spent.normal += depth / total;
    spent.driven += driven / total;

    for (share, charge) in [(driven, Charge::Moving), (inherited, Charge::Free)] {
        if share <= 0.0 {
            continue;
        }
        let push = scale(normal, share / total);
        accumulate(&mut out[0], &first.now, ra, scale(push, -1.0), charge);
        accumulate(&mut out[1], &second.now, rb, push, charge);
    }
    (out, spent)
}

/// **The tangential half of the same contact**: friction and rolling resistance, applied
/// against the normal impulse the half above has already agreed on.
///
/// See the module header on [`super`] for why the two halves are separate sub-passes over
/// the same colours rather than one call.
///
/// # What loads a contact, and why it is not the depth
///
/// This asks `spent.normal > 0` rather than re-testing the overlap, and the difference is
/// the whole reason the split works. By the time this runs, the normal sub-pass has just
/// removed the overlap it found -- so a depth test here reports *every loaded contact as
/// unloaded*, and friction would never act at all. The physical statement is the other
/// one anyway: friction exists wherever a normal force was carried, and the normal force
/// this contact carried over this step is exactly `spent.normal`. A contact that carried
/// nothing is untouched and costs one comparison.
///
/// # The stick memory, when the pair has one
///
/// `memory` is the pair's [`PairMemory`], handed only to its first contact. What it asks
/// back is answered after this step's own slide and roll, on the same cones, so this step's
/// motion is answered first at today's authority and the memory gets what is left. It is
/// charged [`Charge::Free`]; see [`PairMemory`] for why.
pub(super) fn solve_contact_friction(
    contact: Contact,
    first: &Gathered,
    second: &Gathered,
    friction: f64,
    rolling_resistance: f64,
    spent: Spent,
    memory: Option<PairMemory>,
) -> ([Correction; 2], Spent) {
    let mut out = [Correction::none(); 2];
    let mut spent = spent;
    let Contact {
        a, b, normal, span, ..
    } = contact;
    out[0].body = a;
    out[1].body = b;
    if spent.normal <= 0.0 {
        return (out, spent);
    }

    let Surfaces { ra, rb, slid, .. } = surfaces(&contact, first, second);
    let (owed_slide, owed_roll) = match memory {
        Some(memory) => pair_memory_debt(&memory, first, second, normal),
        None => ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    };

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
        owed_roll,
    );

    if friction <= 0.0 {
        return (out, spent);
    }

    // This step's slide, then what the memory is owed, at the memory's own points. Each
    // body feels only the part of the friction couple its share of the patch cannot
    // cancel; see [`patch_arm`], which is where the argument is. The reach is the pair's,
    // the offset is each body's own radius.
    let tangential = sub(slid, scale(normal, dot(slid, normal)));
    let owed_at = memory.map(|memory| {
        (
            rotate(first.now.orientation, memory.local_a),
            rotate(second.now.orientation, memory.local_b),
        )
    });
    for (drift, (ra, rb), charge) in [
        (tangential, (ra, rb), Charge::Moving),
        (owed_slide, owed_at.unwrap_or((ra, rb)), Charge::Free),
    ] {
        let Some(direction) = normalized(drift) else {
            continue;
        };
        let reach = 0.5 * dot(span, direction).abs();
        let arm_a = patch_arm(ra, normal, reach, friction);
        let arm_b = patch_arm(rb, normal, reach, friction);
        let ta = generalised_inverse_mass(&first.now, arm_a, direction);
        let tb = generalised_inverse_mass(&second.now, arm_b, direction);
        let total = ta + tb;
        if total <= 1e-12 {
            continue;
        }

        // Coulomb, over the step rather than over this pass and as a cone rather than a
        // running total: see [`Spent`]. The impulse that would cancel the whole remaining
        // slide is added to what the step has already applied, the resultant is projected
        // back into the cone if it has left it, and the difference is what goes on now.
        let wanted = scale(direction, -length(drift) / total);
        let (grip, total_grip) = cone(spent.tangential, wanted, friction * spent.normal);
        if grip == (0.0, 0.0, 0.0) {
            continue;
        }
        spent.tangential = total_grip;
        accumulate(&mut out[0], &first.now, arm_a, grip, charge);
        accumulate(&mut out[1], &second.now, arm_b, scale(grip, -1.0), charge);
    }
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
    /// **The whole of the body's contact with the plane, as the two ends of it**, in the
    /// body's own frame. Body-frame for the same reason [`Contact`]'s are.
    ///
    /// Always two, and neither degenerate case needs asking about. A sphere's two ends
    /// are the same point; a capsule stood upright has one end clear of the plane. In
    /// both, [`solve_ground`] finds that one end carries the whole load and the other
    /// carries none, which is the answer -- and it finds it the same way it finds the
    /// split for a capsule lying flat, so there is nothing here that tests how many
    /// contacts there are.
    pub local: [(f64, f64, f64); 2],
    /// **The plane this contact was made against**, as `dot(normal, p) = distance`.
    ///
    /// Carried per contact rather than read from the skeleton because the ground is not
    /// necessarily one plane: against a height field every body gets the plane tangent to
    /// the surface underneath it, and two bodies a metre apart on a slope get two
    /// different ones. A skeleton whose ground *is* a plane simply stores the same pair in
    /// every contact, which costs it thirty-two bytes on a structure that is built fresh
    /// each step and never outlives it.
    ///
    /// It is also the plane the rest of the step has to keep agreeing with. The normal
    /// solve, the friction budget and the anchor test are three separate readings of "is
    /// this body still against the ground", and if any of them read a different plane from
    /// the one the contact was built against, they disagree about where the ground is --
    /// which is a body that sticks to a surface it has left. Storing it once on the
    /// contact is what makes that impossible to get wrong rather than merely unlikely.
    pub normal: (f64, f64, f64),
    pub distance: f64,
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
#[allow(
    clippy::too_many_arguments,
    reason = "one body's pose, shape and the plane, read in the solver's hot loop; a struct would only be built to be taken apart"
)]
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
    let surface = |end: (f64, f64, f64)| sub(end, scale(normal, radius));
    let (first, second) = (surface(low), surface(high));
    // One contact for the body, carrying both ends, as soon as either of them is under
    // the plane. The end that is not under it needs no case of its own: it comes out of
    // the solve carrying no load.
    if dot(normal, first) >= distance && dot(normal, second) >= distance {
        return;
    }
    let inverse = |p: (f64, f64, f64)| rotate_inv(orientation, sub(p, position));
    out.push(GroundContact {
        body,
        local: [inverse(first), inverse(second)],
        normal,
        distance,
    });
}

/// What one ground contact wants done. Reads only; the caller applies.
/// **The whole of one body's contact with the plane, solved as one thing.**
///
/// # Why this is one constraint and was two
///
/// A capsule lying on the plane touches it along a line, and the two ends of that line
/// used to be two constraints, solved one stage after the other with a barrier between
/// them. They were never independent: the Coulomb budget had to be pooled across them by
/// hand and was the one running impulse indexed by body rather than by constraint;
/// [`patch_arm`] needed the span between them to work out where the load could move to;
/// and `ground_span` existed for no other reason than to let one end know about the
/// other. Three arguments from three directions that the patch was one thing pretending
/// to be two.
///
/// It also **skated**. Solving one end and then the other tips the body about the
/// across-patch axis and then tips it back, and rolling resistance billed that transient
/// as a roll while friction hauled the centre after the contact point it displaced.
/// Measured, a lone capsule on level ground travelled 10 to 40 mm a second for as long as
/// it was watched, turning by nothing at all, and a sphere -- which has no second sample
/// -- did not move at all. Nothing downstream fixes that: four corrections were tried and
/// every one of them was consistent only where the contacts already agreed.
///
/// # The two rows, and why it is not two solves
///
/// Both ends must end up out of the plane, and one rigid displacement has to do it. The
/// normal displacement an impulse `l_i` along the normal at `r_i` produces at `r_j` is
/// `l_i * K_ji`, where
///
/// ```text
///   K_ji = 1/m + (r_j x n) . I^-1 (r_i x n)
/// ```
///
/// -- the same generalised inverse mass the rest of this module uses, with two different
/// arms. So the pair of depths is `K l = d`, a two-by-two solve rather than two
/// one-by-one ones, and it is the off-diagonal `K_01` that carries what each end does to
/// the other.
///
/// **The normal may push and not pull**, so `l` is also required to be non-negative,
/// which makes this a two-variable complementarity problem rather than a linear solve.
/// Two variables is small enough to enumerate: try both ends loaded, and if that wants a
/// negative impulse anywhere, try each end alone and keep the one whose impulse lifts the
/// other end clear. That is the whole of it, and the degenerate cases fall out of it --
/// a sphere's two ends coincide, which makes `K` singular and sends it to the one-end
/// branch; an upright capsule's second end is above the plane, which gives it a
/// non-positive depth and no load.
#[allow(clippy::too_many_arguments, reason = "kept: stale, the fn takes fewer than 8 arguments so the lint does not fire; dropping it is outside an attribute-only pass")]
pub(super) fn solve_ground_normal(
    contact: GroundContact,
    body: &Gathered,
    normal: (f64, f64, f64),
    distance: f64,
    patch: Patch,
) -> (Correction, Patch) {
    let mut out = Correction::none();
    let mut patch = patch;
    let spent = &mut patch.spent;
    let GroundContact { body: index, local, .. } = contact;
    out.body = index;

    let arm = [
        rotate(body.now.orientation, local[0]),
        rotate(body.now.orientation, local[1]),
    ];
    let at = [add(body.now.position, arm[0]), add(body.now.position, arm[1])];
    let depth = [
        distance - dot(normal, at[0]),
        distance - dot(normal, at[1]),
    ];
    if depth[0] <= 0.0 && depth[1] <= 0.0 {
        return (out, patch);
    }

    let cross_n = [cross(arm[0], normal), cross(arm[1], normal)];
    let turned = [
        body.now.world_inv_inertia.apply(cross_n[0]),
        body.now.world_inv_inertia.apply(cross_n[1]),
    ];
    let k00 = body.now.inv_mass + dot(cross_n[0], turned[0]);
    let k11 = body.now.inv_mass + dot(cross_n[1], turned[1]);
    let k01 = body.now.inv_mass + dot(cross_n[0], turned[1]);
    if k00 <= 1e-12 && k11 <= 1e-12 {
        return (out, patch);
    }

    let share = solve_patch(depth, k00, k11, k01);
    let total = share[0] + share[1];
    if total <= 0.0 {
        return (out, patch);
    }

    // Where the load ends up standing, which is where the tangential impulse and the
    // rolling couple act. For one loaded end that is the end; for two it is between
    // them, weighted as they carry -- and moving it is what a real patch does instead of
    // tipping, which is why [`patch_arm`] no longer has to stand in for it.
    let load = scale(
        add(scale(arm[0], share[0]), scale(arm[1], share[1])),
        1.0 / total,
    );
    patch.local_load = rotate_inv(body.now.orientation, load);

    let was = |local: (f64, f64, f64)| add(body.prev_position, rotate(body.prev_orientation, local));
    for end in 0..2 {
        if share[end] <= 0.0 {
            continue;
        }
        // How much of this end's overlap the step itself drove into the plane, which is
        // the only part allowed to read back as velocity. See
        // `Correction::free_translation`.
        let slid = sub(at[end], was(local[end]));
        let driven = (-dot(slid, normal)).clamp(0.0, depth[end]);
        let part = if depth[end] > 0.0 { driven / depth[end] } else { 0.0 };
        spent.normal += share[end];
        spent.driven += share[end] * part;
        for (fraction, charge) in [(part, Charge::Moving), (1.0 - part, Charge::Free)] {
            if fraction <= 0.0 {
                continue;
            }
            accumulate(
                &mut out,
                &body.now,
                arm[end],
                scale(normal, share[end] * fraction),
                charge,
            );
        }
    }

    (out, patch)
}

/// **The tangential half of the same ground patch**: friction, the anchor's stored slip,
/// and rolling resistance, against the normal impulse the half above agreed on.
///
/// See [`solve_contact_friction`] for why the gate is the normal impulse this step spent
/// rather than the overlap that is left, and the module header on [`super`] for why the
/// two halves are separate sub-passes.
#[allow(clippy::too_many_arguments, reason = "private solver helper; a struct for these arguments would only be built to be taken apart")]
pub(super) fn solve_ground_friction(
    contact: GroundContact,
    body: &Gathered,
    friction: f64,
    rolling_resistance: f64,
    normal: (f64, f64, f64),
    distance: f64,
    patch: Patch,
    anchor: Option<Anchor>,
    anchor_reach: f64,
    roll: Option<RollMemory>,
) -> (Correction, Patch) {
    let mut patch = patch;
    if patch.spent.normal <= 0.0 {
        let mut out = Correction::none();
        out.body = contact.body;
        return (out, patch);
    }
    let out = ground_friction(
        contact,
        body,
        friction,
        rolling_resistance,
        normal,
        distance,
        patch.local_load,
        &mut patch.spent,
        &mut patch.twist,
        anchor,
        anchor_reach,
        roll,
    );
    (out, patch)
}

/// The body of the above, with the running totals borrowed rather than moved through, so
/// that its half-dozen early returns have one thing to return.
#[allow(clippy::too_many_arguments, reason = "private solver helper; a struct for these arguments would only be built to be taken apart")]
fn ground_friction(
    contact: GroundContact,
    body: &Gathered,
    friction: f64,
    rolling_resistance: f64,
    normal: (f64, f64, f64),
    distance: f64,
    local_load: (f64, f64, f64),
    spent: &mut Spent,
    twist: &mut f64,
    anchor: Option<Anchor>,
    anchor_reach: f64,
    roll: Option<RollMemory>,
) -> Correction {
    let mut out = Correction::none();
    let GroundContact { body: index, local, .. } = contact;
    out.body = index;

    let arm = [
        rotate(body.now.orientation, local[0]),
        rotate(body.now.orientation, local[1]),
    ];
    // Where the normal sub-pass put the load, carried through whatever has moved the body
    // since. See [`Patch`].
    let load = rotate(body.now.orientation, local_load);

    // What the rolling memory is owed: the turn since the body's rolling stuck, about axes
    // in the plane, measured from where the body came from. See [`RollMemory`].
    let owed_roll = match roll {
        Some(roll) => {
            let turned = turned_since(body.prev_orientation, roll.orientation);
            sub(turned, scale(normal, dot(turned, normal)))
        }
        None => (0.0, 0.0, 0.0),
    };
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
        owed_roll,
    );
    out = pair[0];

    if friction <= 0.0 {
        return out;
    }
    let mut out = 'slide: {
        // The patch slides as one, so the drift that friction answers is the drift of the
        // point the load stands at, not of either end on its own.
        let here = add(body.now.position, load);
        let before = add(
            body.prev_position,
            rotate(body.prev_orientation, local_load),
        );
        let flat = |v: (f64, f64, f64)| sub(v, scale(normal, dot(v, normal)));
        let slid = flat(sub(here, before));
        // What earlier steps failed to take off, if this patch has been stuck since -- the
        // travel of the piece of surface that is stuck, from where it stuck to where the step
        // began. The anchor is dropped here as well as in the maintenance if that piece has
        // risen off the plane, because then the body has rolled and the memory is about
        // somewhere else. See [`Anchor`].
        let anchor = anchor.filter(|anchor| {
            let at = add(body.now.position, rotate(body.now.orientation, anchor.local));
            distance - dot(normal, at) > 0.0
        });
        let stored = match anchor {
            Some(anchor) => {
                let stuck = add(anchor.position, rotate(anchor.orientation, anchor.local));
                let then = add(body.prev_position, rotate(body.prev_orientation, anchor.local));
                flat(sub(then, stuck))
            }
            None => (0.0, 0.0, 0.0),
        };
        // The direction the whole of it lies in, which is what the patch's reach and the
        // couple it can carry are measured along.
        let Some(heading) = normalized(add(slid, stored)) else {
            break 'slide out;
        };
        // The stored part is redeemable only up to the budget it was banked at, turned into a
        // distance by the same inverse mass the impulse will be divided by. This step's own
        // slide keeps today's authority, which is the cone below.
        let tangential = match anchor {
            None => slid,
            Some(anchor) => {
                let span = 0.5 * dot(sub(arm[1], arm[0]), heading).abs();
                let lever = patch_arm(load, normal, span, friction);
                let mobility = generalised_inverse_mass(&body.now, lever, heading);
                let most = (anchor.hold * mobility).min(anchor_reach);
                let size = length(stored);
                if size > most {
                    add(slid, scale(stored, most / size))
                } else {
                    add(slid, stored)
                }
            }
        };
        let Some(direction) = normalized(tangential) else {
            break 'slide out;
        };
        // The load stands where it stands, but the couple the impulse leaves still has to be
        // carried, and how much of it the patch can absorb is what [`patch_arm`] answers. The
        // reach is the patch's own, half of it, along the direction being resisted -- which is
        // the one thing the two ends are still needed for once the normal solve has decided
        // how they share the load.
        let reach = 0.5 * dot(sub(arm[1], arm[0]), direction).abs();
        let couple = patch_arm(load, normal, reach, friction);
        let tw = generalised_inverse_mass(&body.now, couple, direction);
        if tw <= 1e-12 {
            break 'slide out;
        }
        // Coulomb over the step, and as a cone; see the pair version and [`Spent`].
        let wanted = scale(direction, -length(tangential) / tw);
        let (grip, total_grip) = cone(spent.tangential, wanted, friction * spent.normal);
        if grip == (0.0, 0.0, 0.0) {
            break 'slide out;
        }
        spent.tangential = total_grip;
        // **The whole of it reads back as velocity, the stored part included**, and that was
        // measured rather than assumed. Taking off slip an earlier step left behind is the
        // correction of an error rather than something the body did, so the argument for
        // `Correction::free_translation` -- which the normal solve a few lines above makes for
        // exactly this reason -- looks like it should apply to the stored share. It does not
        // pay: charging that share as free leaves the body undamped by it, and over eight draws
        // that takes a stack of five from settling every time to settling in five, and a
        // settled pile of forty from 0.103 of a reach to 0.189. Damping the old error as well
        // is what those settle on.
        accumulate(&mut out, &body.now, couple, grip, Charge::Moving);
        out
    };
    // **The twist last**, so that what it answers includes the turn the slide's own impulse
    // makes: the load point is off the centre, so holding the body against sliding turns
    // it about the normal. Answered before the slide, that turn was left in the body and
    // the next step's free untwist took it back without its spin, which also moved the
    // slide anchor's stuck point and so came back as velocity: measured on a capsule laid
    // across a 10 degree slope, a two-step cycle at 2e-3 to 7e-3 rad/s and 3e-3 m/s.
    if let Some(roll) = roll {
        out = untwist(out, body, normal, sub(arm[1], arm[0]), friction, spent.normal, twist, roll);
    }
    out
}

/// **The velocity pass's normal half for one pair contact: a resting contact is perfectly
/// inelastic.**
///
/// # What this is for
///
/// The positional solve pushes two overlapping surfaces apart, and a position-based step
/// reads that push back as speed -- which is what stops a falling body, and is also what
/// hands a resting one momentum it did not earn. Nothing in this module said what the
/// relative normal velocity of a contact should be *after* the step, so the answer was
/// whatever the corrections happened to leave. This says it: zero.
///
/// Restitution is the coefficient that would make it something else, and it is zero here
/// because these are rigid capsules resting on one another rather than a coefficient
/// chosen to make a scene look right. A caller who wants a bouncy contact wants
/// `-e * closing` as the target instead of `0`, measured before the solve rather than
/// after it, and this is the only place in the module where that number could act.
///
/// # The one bound, and it is not a tuning
///
/// **A contact cannot pull.** Cancelling a separating velocity means taking normal impulse
/// back out of the step, and the most that can be taken out is what the positional solve
/// put in -- `spent.normal`, which is the normal impulse this contact carried. Take more
/// and the pair would be stuck together, which is the one thing a unilateral constraint
/// may never do. Nothing else bounds it: the constraint is otherwise as hard as the
/// positional one is.
///
/// Reads only; the caller applies. Writes no running total, because the pass runs once a
/// step and there is nothing for a second one to carry.
#[allow(clippy::too_many_arguments, reason = "private solver helper; a struct for these arguments would only be built to be taken apart")]
pub(super) fn solve_contact_velocity_normal(
    contact: Contact,
    first: &Gathered,
    second: &Gathered,
    spent: Spent,
    dt: f64,
    a_moves: ((f64, f64, f64), (f64, f64, f64)),
    b_moves: ((f64, f64, f64), (f64, f64, f64)),
    share: f64,
) -> ([Correction; 2], Spent) {
    let mut out = [Correction::none(); 2];
    let mut spent = spent;
    let Contact { a, b, normal, .. } = contact;
    out[0].body = a;
    out[1].body = b;
    // Friction's gate, for friction's reason: the normal sub-pass has just closed the
    // overlap, so a depth test here reports every loaded contact as unloaded. What says a
    // contact is resting on something is the normal impulse it carried.
    if spent.normal <= 0.0 || share <= 0.0 {
        return (out, spent);
    }

    let ra = rotate(first.now.orientation, contact.local_a);
    let rb = rotate(second.now.orientation, contact.local_b);
    let ((va, wa), (vb, wb)) = (a_moves, b_moves);
    // The rate the overlap is growing at, which is the derivative of `Surfaces::depth`:
    // positive is the two still driving into each other, negative is separating.
    let closing = dot(sub(add(va, cross(wa, ra)), add(vb, cross(wb, rb))), normal);

    let ka = generalised_inverse_mass(&first.now, ra, normal);
    let kb = generalised_inverse_mass(&second.now, rb, normal);
    let total = ka + kb;
    if total <= 1e-12 {
        return (out, spent);
    }
    // At the positional scale, like everything [`accumulate`] is handed: the impulse that
    // leaves the surfaces neither closing nor opening, then the no-pull bound.
    let lambda = (share * closing * dt / total).max(-spent.driven);
    if lambda == 0.0 {
        return (out, spent);
    }
    // **What is taken back comes off the total, so a second sweep cannot take it again.**
    // The pass runs [`super::VELOCITY_SWEEPS`] times, and a bound re-read from the
    // positional figure each sweep would let N of them hand back N times what the contact
    // ever gave. A positive `lambda` is normal impulse this sweep *added* as velocity, so it
    // raises `driven` by the same accounting.
    //
    // **`normal` is not raised with it**, which is why `driven` is not a share of `normal`
    // and its own doc no longer claims to be. It is not an oversight: `normal` is the
    // positional solve's record and `anchor_ground` reads it to decide whether a patch
    // stuck, so a velocity-pass addition does not belong in it. The consequence to keep in
    // mind is that the Coulomb budget here is charged against a `spent.normal` that is
    // missing whatever this sibling half just applied.
    spent.driven = (spent.driven + lambda).max(0.0);
    let push = scale(normal, lambda);
    accumulate(&mut out[0], &first.now, ra, scale(push, -1.0), Charge::Still);
    accumulate(&mut out[1], &second.now, rb, push, Charge::Still);
    (out, spent)
}

/// **The velocity pass's tangential half for one pair contact**: Coulomb and rolling
/// resistance on the velocity the positional solve left, out of what is left of the step's
/// own cone.
///
/// # Why it spends the same budget rather than a fresh one
///
/// Coulomb's limit is a budget for the **step** -- that is the module header's first law
/// about friction, and the velocity pass is part of the step. So this does not get a
/// second `friction * spent.normal` to spend: it adds to the same resultant the positional
/// passes accumulated in [`Spent`] and is clipped by the same cone.
///
/// That is not conservatism, it is the only answer that keeps the coefficient meaning what
/// it says. Give the velocity pass a budget of its own and a contact may spend
/// `friction * N` twice in one step, which is a coefficient of `2 * friction`: a slope
/// that should let go at Coulomb's angle would hold to twice the tangent of it. Sharing
/// the cone also sorts the two cases by itself, with nothing to decide. A **sliding**
/// contact has spent its cone in the positional passes, so there is nothing here for it
/// and it slides exactly as it did. A **sticking** one has spent only what it took to hold
/// still, and what is left is what this may use to take the last of the movement out.
///
/// # And why rolling resistance is not in it, where the positional half has them together
///
/// Rolling resistance is the module's model of a **deformed contact patch**: a soft body
/// flattens, the normal load moves ahead of the contact point, and the offset is a torque
/// against the roll. It exists because Coulomb cannot see a roll at all -- the contact
/// point of a rolling body is instantaneously still, so there is no relative surface
/// velocity there for a velocity-level law to act on. That is the whole reason the module
/// has it, and it is the reason it has no velocity-level half: the quantity this pass acts
/// on is the relative velocity of two surfaces, and a roll is not one.
///
/// What the angular version would act on instead is the relative *spin* of the two bodies,
/// and between two bones of one skeleton that is mostly the joints doing their job.
/// Measured, resisting it takes the seventeen-bone rig from no draw of twenty-four
/// travelling past the jostling allowance to six of them, the worst at 4.4 of a reach --
/// the articulation being braked at the contacts and the rig rocking on what is left.
#[allow(
    clippy::too_many_arguments,
    reason = "the two gathered bodies and the pair's coefficients, read in the solver's hot loop; a struct would only be built to be taken apart"
)]
pub(super) fn solve_contact_velocity_friction(
    contact: Contact,
    first: &Gathered,
    second: &Gathered,
    friction: f64,
    spent: Spent,
    dt: f64,
    a_moves: ((f64, f64, f64), (f64, f64, f64)),
    b_moves: ((f64, f64, f64), (f64, f64, f64)),
    share: f64,
) -> ([Correction; 2], Spent) {
    let mut out = [Correction::none(); 2];
    let mut spent = spent;
    let Contact {
        a, b, normal, span, ..
    } = contact;
    out[0].body = a;
    out[1].body = b;
    if spent.normal <= 0.0 || share <= 0.0 || friction <= 0.0 {
        return (out, spent);
    }

    let ra = rotate(first.now.orientation, contact.local_a);
    let rb = rotate(second.now.orientation, contact.local_b);
    let ((va, wa), (vb, wb)) = (a_moves, b_moves);
    let slip = {
        let relative = sub(add(va, cross(wa, ra)), add(vb, cross(wb, rb)));
        sub(relative, scale(normal, dot(relative, normal)))
    };
    let Some(direction) = normalized(slip) else {
        return (out, spent);
    };
    // The patch carries what of the friction couple it can, exactly as it does one level
    // up. See [`patch_arm`].
    let reach = 0.5 * dot(span, direction).abs();
    let arm_a = patch_arm(ra, normal, reach, friction);
    let arm_b = patch_arm(rb, normal, reach, friction);
    let ta = generalised_inverse_mass(&first.now, arm_a, direction);
    let tb = generalised_inverse_mass(&second.now, arm_b, direction);
    let total = ta + tb;
    if total <= 1e-12 {
        return (out, spent);
    }
    // At the positional scale: the impulse that stops the slip is the one that would undo
    // the distance the slip is about to cover, which is `slip * dt`.
    let wanted = scale(direction, -share * length(slip) * dt / total);
    let (grip, total_grip) = cone(spent.tangential, wanted, friction * spent.normal);
    if grip == (0.0, 0.0, 0.0) {
        return (out, spent);
    }
    // **One cone for the whole step, the velocity sweeps included.** The resultant is
    // carried the way the positional passes carry it, so running the pass N times spends
    // the coefficient once. Re-reading the positional resultant each sweep would be a
    // coefficient of N times `friction`, which is the defect this module rejects a fresh
    // budget for one level up.
    spent.tangential = total_grip;
    accumulate(&mut out[0], &first.now, arm_a, grip, Charge::Still);
    accumulate(
        &mut out[1],
        &second.now,
        arm_b,
        scale(grip, -1.0),
        Charge::Still,
    );
    (out, spent)
}

/// **The velocity pass's normal half against the plane.** Everything
/// [`solve_contact_velocity_normal`] says, with the second body left out because the
/// ground has no velocity to take.
///
/// The arm is where the patch's load stands, which the normal sub-pass worked out and
/// [`Patch`] carried: the plane pushes where the load is, so that is where taking some of
/// the push back has to act.
#[allow(clippy::too_many_arguments, reason = "private solver helper; a struct for these arguments would only be built to be taken apart")]
pub(super) fn solve_ground_velocity_normal(
    contact: GroundContact,
    body: &Gathered,
    normal: (f64, f64, f64),
    patch: Patch,
    dt: f64,
    v: (f64, f64, f64),
    w: (f64, f64, f64),
    share: f64,
) -> (Correction, Patch) {
    let mut out = Correction::none();
    let mut patch = patch;
    out.body = contact.body;
    if patch.spent.normal <= 0.0 || share <= 0.0 {
        return (out, patch);
    }

    let load = rotate(body.now.orientation, patch.local_load);
    // The plane's normal points out of it, so a body leaving the plane has a positive
    // component along it -- the opposite sign to the pair version, where the normal runs
    // from one body into the other.
    let leaving = dot(add(v, cross(w, load)), normal);
    let k = generalised_inverse_mass(&body.now, load, normal);
    if k <= 1e-12 {
        return (out, patch);
    }
    let lambda = (-share * leaving * dt / k).max(-patch.spent.driven);
    if lambda == 0.0 {
        return (out, patch);
    }
    // Off the total, so a second sweep cannot take it again; see the pair version. **Only
    // `driven` moves**: `normal` and `tangential` are what [`Skeleton::anchor_ground`]
    // decides a patch's stickiness from once the step is over, and they are the positional
    // solve's record of it rather than a running budget.
    patch.spent.driven = (patch.spent.driven + lambda).max(0.0);
    accumulate(
        &mut out,
        &body.now,
        load,
        scale(normal, lambda),
        Charge::Still,
    );
    (out, patch)
}

/// How much of the normal load each end of a patch carries: `K l = d` subject to `l >= 0`.
///
/// Enumerated rather than iterated, because with two variables there are only three
/// candidate active sets and the right one is the first that is feasible. See
/// [`solve_ground`] for what `K` is and why the non-negativity is not optional.
fn solve_patch(depth: [f64; 2], k00: f64, k11: f64, k01: f64) -> [f64; 2] {
    // Both ends loaded. Singular where the two ends are the same point -- a sphere --
    // which is exactly when there is only one end to load.
    let det = k00 * k11 - k01 * k01;
    if det > 1e-18 && depth[0] > 0.0 && depth[1] > 0.0 {
        let first = (k11 * depth[0] - k01 * depth[1]) / det;
        let second = (k00 * depth[1] - k01 * depth[0]) / det;
        if first >= 0.0 && second >= 0.0 {
            return [first, second];
        }
    }
    // One end loaded, and it has to lift the other clear rather than leave it under the
    // plane -- which is the complementarity condition, and the reason this cannot just
    // take whichever end is deeper.
    if depth[0] > 0.0 && k00 > 1e-12 {
        let first = depth[0] / k00;
        if first * k01 >= depth[1] {
            return [first, 0.0];
        }
    }
    if depth[1] > 0.0 && k11 > 1e-12 {
        let second = depth[1] / k11;
        if second * k01 >= depth[0] {
            return [0.0, second];
        }
    }
    // Neither end alone answers for the other and both together want a pull. Take the
    // deeper end on its own: it is the row that must be satisfied, and the pass after
    // this one sees what it left.
    if depth[0] >= depth[1] && depth[0] > 0.0 && k00 > 1e-12 {
        return [depth[0] / k00, 0.0];
    }
    if depth[1] > 0.0 && k11 > 1e-12 {
        return [0.0, depth[1] / k11];
    }
    [0.0, 0.0]
}
