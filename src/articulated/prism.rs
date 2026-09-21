//! Contacts between bodies that have flats on them.
//!
//! # Why a shape with faces at all
//!
//! A capsule is a swept sphere: curved everywhere, so it touches anything at a point or
//! along a line, and a point contact has no moment arm. Tip a capsule about the point it
//! rests on and nothing resists to first order -- it simply rolls. Measured on a heap of
//! twenty seventeen-bone rigs, three hundred and two of three hundred and five touching
//! pairs get a single point, and the pile rocks on them for ever: rotation is sixty-two
//! per cent of the motion that keeps those bodies from being counted as still.
//!
//! [`super::contacts::capsule_contact`] already knows this and already has the answer for
//! one case -- two capsules within three degrees of parallel touch along a line and are
//! given a contact at each end of it. A heap is not parallel: the median touching pair is
//! near fifty degrees, so that patch reaches one per cent of them.
//!
//! A body with flats resting on a face touches over a *polygon*. Tip it and one edge lifts
//! while another presses, which is a restoring torque out of the geometry rather than out
//! of a coefficient. It is why a box on a table does not rock and a can does.
//!
//! # Why a prism, and why separating axes
//!
//! A regular prism is the convex hull with the most structure to exploit. Its faces are
//! `n` side normals perpendicular to its own axis plus two caps along it, all known
//! analytically, so there is no support search to run and no hull to store. Its inertia is
//! closed form. And with a fixed vertex count the whole test is a **fixed set of axes**,
//! which terminates by construction -- where GJK terminates on a tolerance, and a solver
//! whose behaviour is pinned by a bit-identity law should prefer the one that cannot
//! iterate a different number of times on a different machine.
//!
//! The axis set is the two prism axes, the two sets of `n` side normals, and the cross
//! products pairing each prism's axis against the other's rim edges: `4n + 3` axes, which
//! is thirty-five for a pair of octagons. **The `n^2` rim-against-rim pairs are left out**,
//! and that is an approximation with a shape: it can under-report the penetration of two
//! prisms meeting corner-of-one-end to corner-of-the-other-end, which is the rarest of the
//! contact cases and never the resting one. If a pile is ever seen to sink at the ends of
//! two crossed limbs, this is the paragraph that is wrong.
//!
//! Projection is by support function rather than by projecting every vertex: the extent of
//! a prism along `d` is `h |d . axis|` plus the largest `d . v` over the `n` cross-section
//! directions, so an axis costs two runs of `n` dot products instead of two runs of `2n`.
//!
//! # What the shape is worth, and the one thing that stops it being used
//!
//! **The geometry does what it was bought for.** Crossed prisms get two contact points at
//! all forty-five poses swept over fifteen crossing angles and three rolls, where the same
//! arrangement in capsules is a single point with no arm at all. On a heap of twenty
//! seventeen-bone rigs, against the same rigs built from capsules:
//!
//! ```text
//!   rigs   shape     asleep at   contacts   a step
//!      1   capsule      254          2      122 us
//!      1   prism      never          6      154
//!      4   capsule    never         34      621
//!      4   prism      never         27      731
//!     20   capsule    never        308     2624
//!     20   prism      never        190     3131
//! ```
//!
//! Thirty-eight per cent fewer contacts for twenty per cent more time a step, which is far
//! cheaper than a separating-axis test against a segment-to-segment one has any right to
//! be -- the support function is why.
//!
//! **And the pile still does not settle, and a lone rig that settled now does not.** That
//! regression is the finding. Measured on one rig after it has collapsed, how far the
//! contact normal of a pair moves between one step and the next:
//!
//! ```text
//!   capsule     0 of 1198 pair-steps moved it more than six degrees   worst agreement 1.0000
//!   prism     107 of 2668 (4.0 per cent)                             worst agreement 0.7934
//! ```
//!
//! An agreement of 0.79 is a **thirty-seven degree jump in the direction the pair is being
//! pushed apart, between one step and the next**. That is the separating axis changing its
//! mind: the minimum-penetration axis switches from one face normal to the neighbouring one
//! as the bodies shift, and nothing here remembers what it said last time. A capsule's
//! normal cannot do this -- it is the direction between two closest points on two segments,
//! and that moves smoothly -- which is why the column above reads zero.
//!
//! So the shape trades a contact with no moment arm for a contact whose *direction* is
//! intermittent, and this module's header records four separate occasions on which an
//! intermittent constraint is exactly what a rig walks on. It is the worse of the two
//! problems and it is the reason prisms are not yet the shape to build a pile out of.
//!
//! # The persistent manifold, and what it is and is not worth
//!
//! Built: [`touch_keeping`], with the axis a pair was last separated along kept in
//! `Skeleton::held_axis` in the first body's own frame, so it follows the body rather than
//! staying put while the body turns under it. A new axis has to beat it by more than one
//! step's sag -- `anchor_reach`, `|g| dt^2` -- which is the scale below which a difference
//! in overlap is indistinguishable from the residual the solve leaves anyway.
//!
//! **It does what it was built to do.** The same one-rig measurement, before and after:
//!
//! ```text
//!   capsule               0 of 1198 pair-steps (0.0 per cent)   worst 1.0000
//!   prism, nothing held 107 of 2668 (4.0 per cent)              worst 0.7934
//!   prism, axis held     11 of 3471 (0.3 per cent)              worst 0.9476
//! ```
//!
//! Thirteen times fewer flips and the worst jump halved, from thirty-seven degrees to
//! eighteen. And the regression it was for is gone: a lone rig of prisms that would not
//! settle at all now settles.
//!
//! **And a pile still does not settle.** Four rigs and twenty do not come to rest with
//! prisms any more than with capsules, though a heap of twenty carries two hundred and two
//! contacts against three hundred and eight.
//!
//! The margin was swept -- one, four, eight and sixteen times the sag -- and a lone rig
//! settled at 1921, 1322, 92 and 623 steps. That is not a lever, it is one draw of a
//! chaotic system going four different ways, which is exactly the trap [`super`]'s header
//! records being caught by elsewhere. So the margin stays at the derived quantity and is
//! not tuned. Piles never settled at any of them.
//!
//! # Remembering the feature rather than the direction
//!
//! What is remembered is now a [`Feature`] -- which face, or which pair of edges -- and not
//! a direction. A direction has to be stored in some frame, and whichever body's frame it
//! is stored in, the *other* body turning moves the face it was resting on out from under
//! it. A feature has no frame: the axis is rebuilt from both orientations every step.
//!
//! **Two bugs came out of doing it, and both were worth more than the change.**
//!
//! The first: the feature was being stored in the same loop that builds the *revival* list,
//! which is gated on a pair having carried normal impulse. Those two lists look alike and
//! are for opposite things -- reviving a constraint is about what a pair *did*, steadying a
//! normal is about what a pair *is* -- so four fifths of a heap's contacts were choosing an
//! axis afresh every step. Measured: forty-four pairs remembered out of two hundred and
//! eleven contacts.
//!
//! The second, and the larger: a prism pair that came momentarily clear fell back to its
//! *bounding capsule's* contact. So a resting pair alternated between a face normal while
//! it overlapped and a between-the-axes direction while it did not -- a bigger swing than
//! the one the manifold exists to stop -- and the revived contact carried no feature, so
//! the pair forgot what it was resting on every time it let go. A revived prism pair now
//! keeps the feature it was resting on and is solved along it.
//!
//! ```text
//!                                   remembered / contacts   flips      worst
//!   nothing held (twenty rigs)                    0 / 308   4.0 %      0.7934
//!   direction held, load-gated                   44 / 212   4.2 %      0.0085
//!   feature held, every contact                 207 / 357   0.75 %     0.0603
//! ```
//!
//! **And no pile settles, at any of them.** A lone rig settles erratically -- 254 steps as
//! capsules, and 1921, 1389, 1005 or never as prisms across these variants -- which is one
//! draw of a chaotic system and not a measurement of anything. What can be said is the flip
//! rate, which is what the manifold was built to move, and it moved by five and a half
//! times.
//!
//! So the shape is **not yet a net win** and `facets` stays opt-in. What it has bought is a
//! contact with a real moment arm and a direction that mostly holds still; what it has not
//! bought is a pile that sleeps. The worst single swing is still eighty-six degrees
//! somewhere in a heap of twenty, and until that is understood rather than averaged away,
//! a caller wanting piles to settle should use capsules and read [`super`]'s header for
//! where that stands.
//!
//! `facets` is opt-in throughout, so a caller who does not ask for flats is not affected:
//! every capsule path is the one it always was.

use super::*;

/// The most flats a prism may have here.
///
/// Not a limit on the shape so much as on the arithmetic: the axis set grows as `4n + 3`
/// and the vertex loops as `2n`, so this is where a contact stops being cheap enough to
/// have thousands of. Eight is the shape this exists for and sixteen is already round.
pub(super) const MOST_FACETS: u32 = 16;

/// Where two prisms touch: the direction to separate them along, how deep they are, and up
/// to two points to apply that at.
///
/// **Two points and not one**, which is the whole reason this module exists. One point is
/// what a capsule gives and what a pile rocks on; two give the pair a moment arm about the
/// axis between them. Two rather than the four a face-against-face clip would produce
/// because [`super::contacts::Contact`] comes in pairs already and the deepest two carry
/// nearly all of the load -- a third point inside the hull of the other two adds no arm.
pub(super) struct Touch {
    pub normal: (f64, f64, f64),
    pub points: [Option<((f64, f64, f64), f64)>; 2],
    /// **Which feature of the two bodies the normal came from**, so the pair can be handed
    /// it back next step. See [`Feature`].
    pub feature: u32,
}

/// Which of the two prisms' features a separating axis was built from.
///
/// **A feature and not a direction, and the difference is the whole of why this is better
/// than what it replaced.** A direction remembered across a step has to be stored in some
/// frame, and whichever body's frame it is stored in, the *other* body turning moves the
/// face it was resting on out from under it. A feature index has no frame: the axis is
/// rebuilt from both orientations every step, so it follows both bodies exactly and there
/// is nothing to drift.
///
/// Packed into a `u32` because it travels on every [`super::contacts::Contact`] and that
/// type is on the hot stream -- the module header records what growing it cost the last
/// time. The high bits are the kind and the low bits are which face or rim edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Feature {
    /// Neither: a pair with nothing remembered yet.
    None,
    /// The cap of one prism or the other.
    CapA,
    CapB,
    /// A side face, by its index in the cross-section.
    FaceA(usize),
    FaceB(usize),
    /// The two axes crossed, which is the edge-against-edge case for the side edges.
    Axes,
    /// One prism's axis crossed with the other's rim edge.
    AxisARimB(usize),
    RimAAxisB(usize),
}

impl Feature {
    const KIND: u32 = 32;

    pub(super) fn code(self) -> u32 {
        match self {
            Feature::None => 0,
            Feature::CapA => Feature::KIND,
            Feature::CapB => 2 * Feature::KIND,
            Feature::FaceA(k) => 3 * Feature::KIND + k as u32,
            Feature::FaceB(k) => 4 * Feature::KIND + k as u32,
            Feature::Axes => 5 * Feature::KIND,
            Feature::AxisARimB(k) => 6 * Feature::KIND + k as u32,
            Feature::RimAAxisB(k) => 7 * Feature::KIND + k as u32,
        }
    }

    pub(super) fn of(code: u32) -> Feature {
        let k = (code % Feature::KIND) as usize;
        match code / Feature::KIND {
            1 => Feature::CapA,
            2 => Feature::CapB,
            3 => Feature::FaceA(k),
            4 => Feature::FaceB(k),
            5 => Feature::Axes,
            6 => Feature::AxisARimB(k),
            7 => Feature::RimAAxisB(k),
            _ => Feature::None,
        }
    }

    /// The axis this feature names, built from where the two prisms are **now**.
    fn axis(self, a: &Shape, b: &Shape) -> Option<(f64, f64, f64)> {
        let rim = |shape: &Shape, k: usize| {
            sub(shape.corners[(k + 1) % shape.facets], shape.corners[k])
        };
        let face = |shape: &Shape, k: usize| {
            add(shape.corners[k], shape.corners[(k + 1) % shape.facets])
        };
        let within = |shape: &Shape, k: usize| k < shape.facets;
        match self {
            Feature::None => None,
            Feature::CapA => Some(a.axis),
            Feature::CapB => Some(b.axis),
            Feature::FaceA(k) if within(a, k) => Some(face(a, k)),
            Feature::FaceB(k) if within(b, k) => Some(face(b, k)),
            Feature::Axes => Some(cross(a.axis, b.axis)),
            Feature::AxisARimB(k) if within(b, k) => Some(cross(a.axis, rim(b, k))),
            Feature::RimAAxisB(k) if within(a, k) => Some(cross(rim(a, k), b.axis)),
            // A face that no longer exists, because the caller changed the shape under the
            // pair. Nothing to hold on to, and the ordinary search answers.
            _ => None,
        }
    }
}

/// One prism's geometry in world space, built once per test.
pub(super) struct Shape {
    pub at: (f64, f64, f64),
    pub axis: (f64, f64, f64),
    pub half_length: f64,
    /// The `n` directions from the axis out to the cross-section's corners, scaled to the
    /// circumradius, in world space.
    pub corners: [(f64, f64, f64); MOST_FACETS as usize],
    pub facets: usize,
}

impl Shape {
    /// The prism as it stands, with its cross-section written out in world space.
    pub(super) fn of(
        position: (f64, f64, f64),
        orientation: Quaternion,
        radius: f64,
        half_length: f64,
        facets: u32,
    ) -> Shape {
        let facets = (facets as usize).min(MOST_FACETS as usize);
        let axis = rotate(orientation, (0.0, 1.0, 0.0));
        let mut corners = [(0.0, 0.0, 0.0); MOST_FACETS as usize];
        for (k, slot) in corners.iter_mut().enumerate().take(facets) {
            // The cross-section is authored in the body's own XZ plane and turned with it,
            // so a prism's flats follow its bone exactly as its length does.
            let turn = std::f64::consts::TAU * k as f64 / facets as f64;
            *slot = rotate(orientation, (radius * turn.cos(), 0.0, radius * turn.sin()));
        }
        Shape {
            at: position,
            axis,
            half_length,
            corners,
            facets,
        }
    }

    /// How far the prism reaches from its centre along `d`, for a unit `d`.
    ///
    /// The cap term and the cross-section term are independent because the axis and the
    /// cross-section are perpendicular by construction, so the support is the sum of the
    /// two rather than a search over the corners of both.
    fn reach_along(&self, d: (f64, f64, f64)) -> f64 {
        let mut widest = f64::NEG_INFINITY;
        for corner in self.corners.iter().take(self.facets) {
            let out = dot(*corner, d);
            if out > widest {
                widest = out;
            }
        }
        self.half_length * dot(self.axis, d).abs() + widest
    }

    /// The `2n` corners of the prism, in a fixed order.
    fn vertices(&self, out: &mut [(f64, f64, f64); 2 * MOST_FACETS as usize]) -> usize {
        let cap = scale(self.axis, self.half_length);
        for k in 0..self.facets {
            let corner = self.corners[k];
            out[2 * k] = add(add(self.at, cap), corner);
            out[2 * k + 1] = add(sub(self.at, cap), corner);
        }
        2 * self.facets
    }

    /// The prism's `n + 2` outward face normals and how far each face stands from the
    /// centre, in a fixed order: the two caps, then the side faces.
    fn planes(&self, out: &mut [((f64, f64, f64), f64); MOST_FACETS as usize + 2]) -> usize {
        out[0] = (self.axis, self.half_length);
        out[1] = (scale(self.axis, -1.0), self.half_length);
        let mut count = 2;
        for k in 0..self.facets {
            let next = self.corners[(k + 1) % self.facets];
            if let Some(normal) = normalized(add(self.corners[k], next)) {
                out[count] = (normal, dot(self.corners[k], normal));
                count += 1;
            }
        }
        count
    }

    /// The face most nearly facing `d`, as its own vertices in world space.
    ///
    /// A cap when the axis is what faces that way, and a side rectangle otherwise. This is
    /// the *incident* face when `d` points back along the contact normal, and the
    /// *reference* face when it points along it.
    fn face_towards(
        &self,
        d: (f64, f64, f64),
        out: &mut [(f64, f64, f64); MOST_FACETS as usize],
    ) -> usize {
        let along = dot(self.axis, d);
        let mut widest = along.abs();
        let mut which = None;
        for k in 0..self.facets {
            let next = self.corners[(k + 1) % self.facets];
            let Some(normal) = normalized(add(self.corners[k], next)) else {
                continue;
            };
            let facing = dot(normal, d);
            if facing > widest {
                widest = facing;
                which = Some(k);
            }
        }
        match which {
            // A side face: the two corners it runs between, at each end of the body.
            Some(k) => {
                let next = self.corners[(k + 1) % self.facets];
                let cap = scale(self.axis, self.half_length);
                out[0] = add(add(self.at, cap), self.corners[k]);
                out[1] = add(add(self.at, cap), next);
                out[2] = add(sub(self.at, cap), next);
                out[3] = add(sub(self.at, cap), self.corners[k]);
                4
            }
            // A cap: the whole cross-section at whichever end faces that way.
            None => {
                let cap = scale(self.axis, if along >= 0.0 { self.half_length } else { -self.half_length });
                for k in 0..self.facets {
                    out[k] = add(add(self.at, cap), self.corners[k]);
                }
                self.facets
            }
        }
    }

    /// How far inside the prism a world point is, or a negative number if it is outside.
    ///
    /// The faces are the `n` side planes plus the two caps, and a convex body's inside is
    /// the intersection of its half-spaces, so the depth is the smallest of them.
    fn depth_of(&self, point: (f64, f64, f64)) -> f64 {
        let from = sub(point, self.at);
        // The caps first: cheapest, and the one that rejects most points.
        let along = dot(from, self.axis);
        let mut least = self.half_length - along.abs();
        if least <= 0.0 {
            return least;
        }
        for k in 0..self.facets {
            // A side face's outward normal is its own corner direction turned by half a
            // step, which is the bisector of the two corners the face runs between.
            let next = self.corners[(k + 1) % self.facets];
            let Some(normal) = normalized(add(self.corners[k], next)) else {
                continue;
            };
            let inradius = dot(self.corners[k], normal);
            let out = inradius - dot(from, normal);
            if out < least {
                least = out;
            }
            if least <= 0.0 {
                return least;
            }
        }
        least
    }
}

/// **Where two prisms touch, by separating axes.**
///
/// `None` when they are clear of one another. The normal points from `a` to `b`, so `a` is
/// pushed against it and `b` along it, which is the convention
/// [`super::contacts::capsule_contact`] uses.
///
/// `held` is the axis this pair was separated along last step, in world space, or `None`
/// for a pair that is new or was clear. **It is kept unless another axis beats it by more
/// than `decisive`**, and that is what makes the direction of a resting contact steady: see
/// [`touch_keeping`].
pub(super) fn touch(a: &Shape, b: &Shape) -> Option<Touch> {
    touch_keeping(a, b, Feature::None, 0.0)
}

/// The same test, told what it said last time.
///
/// # Why a pair has to be remembered at all
///
/// A separating-axis test answers "which direction are these two least overlapped along",
/// and near a face-to-edge transition two axes are *equally* good. Which one wins then
/// depends on the last bit of two nearly equal numbers, so as the bodies shift by a
/// micron the answer switches from one face normal to its neighbour and back. Measured on
/// one rig collapsed on the ground, with nothing remembered: a hundred and seven of two
/// thousand six hundred and sixty-eight pair-steps moved the contact normal by more than
/// six degrees, the worst by thirty-seven. A capsule's normal cannot do this -- it is the
/// direction between two closest points on two segments, which moves smoothly -- and its
/// column of the same measurement reads zero of eleven hundred and ninety-eight.
///
/// A constraint whose *direction* changes every few steps is an intermittent constraint,
/// and [`super`]'s header records four separate occasions on which one of those is what a
/// rig walks on. It is a worse defect than the one flats were bought to fix, and it is why
/// the shape needs this to be usable at all.
///
/// # What makes a new axis decisive
///
/// `decisive` is how much better a new axis has to be before the pair changes its mind,
/// as a depth. The caller passes the crate's own `anchor_reach` -- `|g| dt^2`, the distance
/// one step of gravity drives a resting body into whatever it stands on -- because that is
/// the scale below which a difference in overlap is indistinguishable from the residual
/// the solve leaves anyway. Two axes that differ by less than one step's sag are not
/// meaningfully different, and switching between them is noise being amplified into a
/// constraint.
///
/// **The held axis is not free**: if the pair is genuinely separated along it, they are
/// separated, and the whole test ends there exactly as it would for any other axis.
pub(super) fn touch_keeping(
    a: &Shape,
    b: &Shape,
    held: Feature,
    decisive: f64,
) -> Option<Touch> {
    let between = sub(b.at, a.at);

    // The least-penetrating axis is the one to separate along: any deeper one is a
    // direction the pair is not actually trying to escape in.
    let mut best = f64::INFINITY;
    let mut normal = (0.0, 0.0, 0.0);
    let mut chosen = Feature::None;
    let overlap_along = |axis: (f64, f64, f64)| {
        a.reach_along(axis) + b.reach_along(axis) - dot(between, axis).abs()
    };
    let mut consider = |axis: (f64, f64, f64), what: Feature| -> bool {
        let Some(axis) = normalized(axis) else {
            return true;
        };
        let overlap = overlap_along(axis);
        if overlap <= 0.0 {
            return false;
        }
        if overlap < best {
            best = overlap;
            chosen = what;
            // Pointing from `a` towards `b`, whichever way the axis was built.
            normal = if dot(between, axis) < 0.0 {
                scale(axis, -1.0)
            } else {
                axis
            };
        }
        true
    };

    if !consider(a.axis, Feature::CapA) || !consider(b.axis, Feature::CapB) {
        return None;
    }
    for k in 0..a.facets {
        let face = add(a.corners[k], a.corners[(k + 1) % a.facets]);
        if !consider(face, Feature::FaceA(k)) {
            return None;
        }
    }
    for k in 0..b.facets {
        let face = add(b.corners[k], b.corners[(k + 1) % b.facets]);
        if !consider(face, Feature::FaceB(k)) {
            return None;
        }
    }
    // Edge against edge, minus the rim-against-rim pairs: see the module header for what
    // that leaves out and why it is the case nobody rests in.
    if !consider(cross(a.axis, b.axis), Feature::Axes) {
        return None;
    }
    for k in 0..b.facets {
        let rim = sub(b.corners[(k + 1) % b.facets], b.corners[k]);
        if !consider(cross(a.axis, rim), Feature::AxisARimB(k)) {
            return None;
        }
    }
    for k in 0..a.facets {
        let rim = sub(a.corners[(k + 1) % a.facets], a.corners[k]);
        if !consider(cross(rim, b.axis), Feature::RimAAxisB(k)) {
            return None;
        }
    }
    if best == f64::INFINITY {
        return None;
    }

    // **And the feature this pair was resting on keeps its job unless it has been clearly
    // beaten.** Rebuilt from where the two bodies are now, so it follows both of them.
    if let Some(axis) = held.axis(a, b).and_then(normalized) {
        let overlap = overlap_along(axis);
        if overlap <= 0.0 {
            // Separated along the feature it was resting on, which is separated.
            return None;
        }
        if overlap - best <= decisive {
            best = overlap;
            chosen = held;
            normal = if dot(between, axis) < 0.0 {
                scale(axis, -1.0)
            } else {
                axis
            };
        }
    }

    Some(Touch {
        normal,
        points: deepest_two(a, b, normal),
        feature: chosen.code(),
    })
}

/// How far outside a prism a point may be and still count as touching it.
///
/// **Not a tolerance on the geometry, a tolerance on the arithmetic.** Two prisms of the
/// same length lying flush have their contacting corners *exactly* on each other's cap
/// planes, so the honest answer to "how far inside is this corner" is zero, and in
/// floating point it is zero give or take a bit. Without this the commonest resting
/// arrangement there is -- two limbs side by side -- produces no contact points at all,
/// which is how this was found.
const FLUSH: f64 = 1e-9;

/// The most points a clipped face can have: a prism's widest face is its cap, with
/// `MOST_FACETS` corners, and each of the other prism's `MOST_FACETS + 2` planes can add
/// one more as it cuts a corner off.
const CLIPPED: usize = 2 * MOST_FACETS as usize + 2;

/// The contact region, as the other prism's facing face clipped to this one's.
///
/// **A vertex-based manifold cannot see the case a pile is mostly made of.** Two crossed
/// limbs touch along one's *edge* lying across the other's *face*, and that region is in
/// the interior of an edge -- never at a corner of either body. Asking which corners are
/// inside the other hull finds nothing at all there, which is how this was found: seven
/// arrangements out of eight worked and the commonest one in a heap did not.
///
/// So the incident face is clipped against the reference prism's other half-spaces, in the
/// usual way, and what survives is the region the two actually share. Depth is measured
/// from the reference plane along the contact normal, which is the quantity the solve
/// needs and the only one it can use.
fn clipped(
    reference: &Shape,
    incident: &Shape,
    normal: (f64, f64, f64),
    out: &mut [(f64, f64, f64); CLIPPED],
) -> usize {
    let mut face = [(0.0, 0.0, 0.0); MOST_FACETS as usize];
    let mut count = incident.face_towards(scale(normal, -1.0), &mut face);
    for (k, point) in face.iter().take(count).enumerate() {
        out[k] = *point;
    }

    let mut planes = [((0.0, 0.0, 0.0), 0.0); MOST_FACETS as usize + 2];
    let walls = reference.planes(&mut planes);
    let mut next = [(0.0, 0.0, 0.0); CLIPPED];
    for &(wall, stands) in planes.iter().take(walls) {
        // The face the pair is resting on is not a wall to be clipped against: it is the
        // plane depth is measured from, and clipping to it would throw the contact away.
        if dot(wall, normal) > 0.999 {
            continue;
        }
        let mut kept = 0usize;
        for k in 0..count {
            let here = out[k];
            let there = out[(k + 1) % count];
            let inside = |p: (f64, f64, f64)| stands - dot(sub(p, reference.at), wall);
            let (da, db) = (inside(here), inside(there));
            if da >= -FLUSH && kept < CLIPPED {
                next[kept] = here;
                kept += 1;
            }
            if (da >= -FLUSH) != (db >= -FLUSH) && (da - db).abs() > 0.0 && kept < CLIPPED {
                let t = da / (da - db);
                next[kept] = add(here, scale(sub(there, here), t));
                kept += 1;
            }
        }
        out[..kept].copy_from_slice(&next[..kept]);
        count = kept;
        if count == 0 {
            return 0;
        }
    }
    count
}

/// The two points to solve the contact at, and how deep each of them is.
///
/// **Containment and depth are two different measurements**, and conflating them is the
/// mistake above. Whether a corner belongs to this contact is a question about the *hull*:
/// a corner on the far side of a prism is behind its contact plane by two inradii and has
/// nothing to do with the contact. How deep it is, once it belongs, is a question about
/// the *normal*: the distance behind the supporting plane along the direction the pair is
/// being separated in, which is the number the solve needs and the only one it can use.
///
/// Both prisms are searched rather than only the incident one, because a face resting on
/// an edge has its deep points on one body and a face resting on a face has them on both,
/// and telling those apart costs more than looking at twice as many corners.
fn deepest_two(
    a: &Shape,
    b: &Shape,
    normal: (f64, f64, f64),
) -> [Option<((f64, f64, f64), f64)>; 2] {
    let mut best: [Option<((f64, f64, f64), f64)>; 2] = [None, None];
    let mut offer = |point: (f64, f64, f64), depth: f64| {
        if depth <= 0.0 {
            return;
        }
        match best {
            [None, _] => best[0] = Some((point, depth)),
            [Some((_, first)), None] => {
                if depth > first {
                    best = [Some((point, depth)), best[0]];
                } else {
                    best[1] = Some((point, depth));
                }
            }
            [Some((_, first)), Some((_, second))] => {
                if depth > first {
                    best = [Some((point, depth)), best[0]];
                } else if depth > second {
                    best[1] = Some((point, depth));
                }
            }
        }
    };

    // The region the two share, taken from each side. `a` as the reference finds a face
    // of `b` lying on `a`; `b` as the reference finds the other way round; and an edge
    // across a face is only seen from one of them, so both are asked.
    let front_a = a.reach_along(normal);
    let back = scale(normal, -1.0);
    let front_b = b.reach_along(back);

    let mut region = [(0.0, 0.0, 0.0); CLIPPED];
    let count = clipped(a, b, normal, &mut region);
    for point in region.iter().take(count) {
        offer(*point, front_a - dot(sub(*point, a.at), normal));
    }
    let count = clipped(b, a, back, &mut region);
    for point in region.iter().take(count) {
        offer(*point, front_b - dot(sub(*point, b.at), back));
    }
    best
}

/// **The contact or contacts between two prisms**, in the form the solve takes.
///
/// The same shape [`super::contacts::capsule_contact`] returns, and for the same reasons:
/// the surface points are carried in each body's own frame so a later pass can ask where
/// they have moved to, and a pair that carried load last step keeps its constraint even
/// where the surfaces have come apart -- a stack that has settled perfectly overlaps by
/// nothing, and a constraint that only exists while it overlaps vanishes at exactly the
/// moment it is doing its job.
///
/// The `span` both contacts carry is the arm between them, which is what tells the
/// friction solve how far this patch reaches. For a prism that is not a guess: it is the
/// distance between the two clipped points, which is a real length on a real face.
#[allow(clippy::too_many_arguments)]
pub(super) fn prism_contact(
    a: usize,
    b: usize,
    position: &[(f64, f64, f64)],
    orientation: &[Quaternion],
    radius: &[f64],
    half_length: &[f64],
    facets: &[u32],
    alive: bool,
    // The feature this pair was resting on last step, or zero for a pair with nothing
    // remembered. See [`Feature`].
    held: u32,
    decisive: f64,
) -> [Option<super::contacts::Contact>; 2] {
    let first = Shape::of(position[a], orientation[a], radius[a], half_length[a], facets[a]);
    let second = Shape::of(position[b], orientation[b], radius[b], half_length[b], facets[b]);

    let Some(hit) = touch_keeping(&first, &second, Feature::of(held), decisive) else {
        // Clear of each other. A pair that was carrying load last step still wants its
        // constraint back: a stack that has settled perfectly overlaps by nothing, and a
        // constraint that only exists while it overlaps vanishes at exactly the moment it
        // is doing its job.
        if !alive {
            return [None, None];
        }
        // **Along the feature it was resting on, if it has one.** Handing a prism pair its
        // bounding capsule's normal here was measured and is a defect: a resting pair then
        // alternates between a face normal while it overlaps and a between-the-axes
        // direction while it does not, which is a bigger swing than the one the whole
        // manifold exists to stop. It also leaves the revived contact with no feature, so
        // the pair forgets what it was resting on every time it comes momentarily clear --
        // measured on a heap of twenty, only forty-four of two hundred and twelve contacts
        // carried a feature at all.
        if let Some(axis) = Feature::of(held)
            .axis(&first, &second)
            .and_then(normalized)
        {
            let normal = if dot(sub(second.at, first.at), axis) < 0.0 {
                scale(axis, -1.0)
            } else {
                axis
            };
            let surface_a = add(first.at, scale(normal, first.reach_along(normal)));
            let surface_b = sub(second.at, scale(normal, second.reach_along(scale(normal, -1.0))));
            return [
                Some(super::contacts::Contact {
                    a,
                    b,
                    local_a: rotate_inv(orientation[a], sub(surface_a, position[a])),
                    local_b: rotate_inv(orientation[b], sub(surface_b, position[b])),
                    normal,
                    span: (0.0, 0.0, 0.0),
                    revived: true,
                    feature: held,
                }),
                None,
            ];
        }
        // Nothing remembered, so there is nothing to be consistent with and the bounding
        // capsules answer where the pair is closest.
        return super::contacts::capsule_contact(
            a,
            b,
            position,
            orientation,
            radius,
            half_length,
            true,
        );
    };

    let span = match (hit.points[0], hit.points[1]) {
        (Some((one, _)), Some((two, _))) => sub(two, one),
        _ => (0.0, 0.0, 0.0),
    };
    let mut out = [None, None];
    for (slot, point) in out.iter_mut().zip(hit.points) {
        let Some((at, depth)) = point else { continue };
        // The point is the shared one, so each body's surface point is it: the depth is
        // carried by how far apart they are along the normal, exactly as the capsule path
        // carries it.
        let surface_a = add(at, scale(hit.normal, 0.5 * depth));
        let surface_b = sub(at, scale(hit.normal, 0.5 * depth));
        *slot = Some(super::contacts::Contact {
            a,
            b,
            local_a: rotate_inv(orientation[a], sub(surface_a, position[a])),
            local_b: rotate_inv(orientation[b], sub(surface_b, position[b])),
            normal: hit.normal,
            span,
            revived: false,
            feature: hit.feature,
        });
    }
    out
}

#[cfg(test)]
mod tests;
