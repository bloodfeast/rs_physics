//! Articulated rigid bodies: a shared body array, joints that reference it by index, and
//! one solver that sees all of them at once.
//!
//! # Why this exists next to [`crate::constraints`]
//!
//! The constraint types already in this crate -- [`crate::constraints::Joint3D`],
//! [`crate::constraints::Hinge3D`], [`crate::constraints::Fixed3D`] -- each **own their
//! two bodies by value**:
//!
//! ```ignore
//! pub struct Hinge3D { pub object1: ObjectIn3D, pub object2: ObjectIn3D, .. }
//! ```
//!
//! That is the right shape for one constraint between two things, and the wrong shape for
//! a *skeleton*. A forearm is the second body of the elbow and the first body of the
//! wrist: with ownership by value there are two copies of it, and a correction applied
//! through the elbow is invisible to the wrist until somebody copies it across. Iterating
//! such a set does not converge on the joint set, it oscillates between two pictures of
//! the same limb.
//!
//! So this module keeps **one** set of bodies and gives joints indices into it.
//!
//! # Structure of arrays, and the reason it is not a style preference
//!
//! The bodies are eight parallel arrays rather than a `Vec<Body>`. Two things need that,
//! and neither is cache-line arithmetic:
//!
//! * **SIMD on the streaming passes.** Predicting and reading velocities back are pure
//!   sweeps over every body. As arrays they vectorise and parallelise by chunk; as a
//!   `Vec<Body>` each lane would be a gather out of a 152-byte struct.
//! * **The GPU, if it is ever asked for.** A warp reading `bodies[tid].position` out of an
//!   interleaved struct wastes most of every memory transaction -- coalescing wants the
//!   field contiguous. Converting later would mean rewriting whatever had been built on
//!   top, which is why it is done before contacts rather than after.
//!
//! [`Body`] still exists as the thing you hand to [`Skeleton::add_body`] and get back from
//! [`Skeleton::body`]. It is a *view*, assembled on demand; the storage is the arrays.
//!
//! # Colouring, because the layout was only half the problem
//!
//! The first cut of this solver was Gauss-Seidel -- each joint reading the corrections the
//! last one made and writing immediately -- which is **inherently serial whatever the
//! layout is**: two joints sharing a body cannot run at once. Structure of arrays alone
//! would have parallelised the sweeps and left the solve exactly as serial as it was, and
//! the solve is where the time goes once contacts arrive.
//!
//! So the joints are partitioned into **colours**, where no two joints in a colour touch
//! the same body. Every colour runs fully parallel; within a colour it is still
//! Gauss-Seidel, so convergence is not traded away. A limb colours in two or three, a
//! whole skeleton in four or five.
//!
//! The alternative is Jacobi -- accumulate every correction and apply them at the end --
//! which parallelises without colouring and converges slower, so it needs more iterations
//! to hold a knee. Colouring keeps the iteration count.
//!
//! A colour is solved **and applied** on the threads that solved it. That is what
//! disjointness was always for, and doing anything else gives it away: an earlier version
//! computed a colour's corrections into a buffer and applied them from one thread
//! afterwards, and on a heap of ten thousand bodies the buffer cost more than the
//! arithmetic did. See [`scatter`], which holds the measurement and the safety argument.
//!
//! # Going parallel is not free, and below a size it is a loss
//!
//! Measured, one skeleton of seventeen bodies against a pile of ten thousand:
//!
//! ```text
//!   one skeleton   207 us a step    12.2 us per body
//!   the pile       2.43 ms a step    0.24 us per body
//! ```
//!
//! Fifty times worse per body on the small one, for the same arithmetic. Handing a
//! seventeen-element sweep to a thread pool costs more in scheduling than the sweep costs
//! to run, and a solver is usually called on one skeleton at a time.
//!
//! So each sweep and each colour goes parallel only above [`PARALLEL_FLOOR`], and runs on
//! the calling thread below it.
//!
//! **And a fork is dearer than it looks.** Measured on a twenty-four core machine, one
//! `par_iter` over a few thousand items with an empty body costs 26 to 72 us before any
//! work happens -- the pool has that many threads to wake. A heap colours its contacts in
//! about twenty sets, so eight passes over them is a hundred and sixty forks a step, and
//! the scheduling is a real fraction of the solve. The answers are to ask the pool for
//! fewer, larger things: the three predict sweeps are one sweep, the narrow phase and the
//! broad phase are one fork each over fixed-size chunks, and a colour no longer has a
//! second traversal to apply what it computed. What is left is one fork per colour per
//! pass, which is the floor this structure has.
//!
//! # Contacts, and the two laws that turned out to be needed
//!
//! [`contacts`] adds capsule-versus-capsule and capsule-versus-ground constraints to the
//! same solve: found once a step from the predicted positions, coloured the same way the
//! joints are, and solved in the same passes. Three things there were not obvious, and
//! each of them was a measurement rather than a guess.
//!
//! * **A contact may turn into velocity only the overlap it made this step.** A
//!   position-based solver derives velocity from how far a body moved, so lifting a body
//!   out of an overlap it was already in reads back as speed: bodies spawned inside one
//!   another leave at metres a second, and a heap dropped in as a heap detonates on its
//!   first frame. Separating the two at the source costs a second pair of fields on
//!   [`Correction`] and needs no rate limit, no clamp and no tuning.
//! * **Coulomb's limit is a budget for the step, not for each pass.** Spending it per
//!   pass multiplies friction by the iteration count -- a slope that should have let go
//!   at twenty-seven degrees held past forty -- and dividing it between the passes fails
//!   the other way, because the first pass removes nearly all the overlap and leaves the
//!   rest almost no normal impulse to be a fraction of. Carrying the totals across the
//!   step gets the angle right and makes it the same at four iterations and at
//!   thirty-two. What is carried is a **cone on the resultant**, not a running total of
//!   magnitude: the friction direction reverses between passes, and charging both
//!   directions against one total spends the coefficient to produce no net impulse. See
//!   [`contacts::Spent`].
//! * **A contact patch is not a point, and the difference is a couple.** Friction acts at
//!   the surface, below the centre of mass, so it tips a body forward over its contact.
//!   For a body touching at one point that is the whole story and it should tip. A body
//!   resting on a patch moves its normal load within the patch instead and does not tip,
//!   and modelling it as a point makes a resting body ratchet itself clear of the plane
//!   over the passes until its contacts report no depth and friction stops acting. See
//!   [`contacts::patch_arm`].
//! * **Friction does not resist rolling, so a heap of capsules rolls apart.** The contact
//!   point of a rolling body is instantaneously still, so there is nothing for Coulomb to
//!   act on. Measured, a pile of forty settled onto the ground perfectly happily and then
//!   spread to twenty metres over thirty seconds. Rolling resistance -- the same law on
//!   the same budget, one dimension over -- holds it at about a metre and a quarter.
//!
//! # Allocation
//!
//! [`Skeleton::step`] allocates nothing once it is warm. The predicted state, the colour
//! sets, the contact buffers and the per-chunk buffers the broad and narrow phases fill
//! all live in the struct and are reused. Joint colouring happens when the joint set
//! changes rather than per step; contact colouring has to happen every step, because the
//! contacts do.

use rayon::prelude::*;

use crate::models::Quaternion;

mod broadphase;
mod contacts;
mod scatter;

use broadphase::{Grid, Jointed};
use contacts::{
    capsule_contact, ground_contacts, solve_contact, solve_ground, Contact, GroundContact, Spent,
};
use scatter::Bodies;

/// Below this many items, a sweep or a colour runs on the calling thread.
///
/// See the module header for the measurement. A thread pool has a fixed cost per split --
/// a task, a queue, a join -- and a few dozen elements of arithmetic does not repay it.
///
/// Raising it to a thousand was tried, on the reasoning that a fork costs tens of
/// microseconds and a colour of three hundred contacts is worth twenty. On a loaded
/// machine it looked like a win and on an idle one it was a loss, which is the answer:
/// the fork is dear because the pool's threads are asleep, and on an idle machine they
/// are not. Left where it was, since that is the case the solver is meant for.
const PARALLEL_FLOOR: usize = 256;

/// Candidate pairs per chunk of the narrow phase. See [`Skeleton::build_contacts`].
const NARROW_CHUNK: usize = 1024;

/// Coulomb friction between two bodies, unless a caller says otherwise.
///
/// Not tuned against how a pile looks: it is the measured static coefficient for cloth on
/// cloth, which is what is actually in contact when two clothed bodies rest on each other,
/// and it sits in the same band as skin on most dry surfaces. A pile of rubber wants
/// more and a pile of ice wants far less, which is what [`Skeleton::set_friction`] is for.
const DEFAULT_FRICTION: f64 = 0.5;

/// Rolling resistance between a body and whatever it is resting on.
///
/// **A capsule is perfectly round and a limb is not**, and that difference has to be paid
/// for somewhere. Coulomb friction does not resist rolling at all -- the contact point of
/// a rolling body is instantaneously still, so there is no sliding for friction to act on
/// -- so a heap of ideal capsules converts its sliding into rolling and then rolls apart
/// for ever. Measured on a pile of forty dropped together, bodies were still leaving the
/// heap at half a metre a second after thirty seconds, and had reached twenty metres out.
///
/// Rolling resistance is the real effect the round shape threw away: a deformable body
/// flattens against what it rests on, the support moves ahead of the contact point, and
/// the offset is a torque against the roll. The coefficient is the offset as a fraction
/// of the radius, and a quarter is the band measured for soft bodies on soft ground --
/// far above a steel wheel on rail, which is thousandths.
const DEFAULT_ROLLING_RESISTANCE: f64 = 0.25;

/// A rigid body, as a value. The storage is [`Skeleton`]'s arrays; this is what crosses
/// the API in either direction.
///
/// Deliberately not [`crate::models::PhysicalObject3D`], which carries a `Vec<Force>` per
/// body and an Euler-angle orientation. A skeleton is hundreds of these: a heap
/// allocation each is a cost nothing here needs, and Euler angles gimbal-lock exactly
/// where a shoulder lives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Body {
    pub position: (f64, f64, f64),
    pub orientation: Quaternion,
    pub velocity: (f64, f64, f64),
    pub angular_velocity: (f64, f64, f64),
    /// Reciprocal mass. **Zero pins the body**, which is how a skeleton is anchored to
    /// something the solver does not own -- a corpse's root, a hand still on a blade.
    pub inv_mass: f64,
    /// Reciprocal of the inertia tensor's diagonal, in the body's own frame.
    ///
    /// Diagonal because every shape a limb is made of -- a capsule, a box, a sphere --
    /// has its principal axes along its own, so the off-diagonal terms are zero in the
    /// frame the body is authored in. Storing three numbers instead of nine is not an
    /// approximation here; it is the same tensor written where it is diagonal.
    pub inv_inertia: (f64, f64, f64),
    /// The capsule this body collides as: a segment of `2 * half_length` down its own
    /// +Y, with `radius` around it.
    ///
    /// **Radius zero means no extent and no contacts.** That is what a body used purely
    /// as a joint anchor wants, and it is the default, so a caller who has not thought
    /// about collision does not silently get it.
    pub radius: f64,
    pub half_length: f64,
}

impl Body {
    /// A body at rest at `position`, with the inertia of a solid capsule of this `mass`,
    /// `radius` and segment `length`, its long axis along local **+Y**.
    ///
    /// The axis is +Y because that is the convention skeletal formats put a bone's own
    /// length down, so a segment authored in one arrives pointing the right way.
    pub fn capsule(mass: f64, radius: f64, length: f64, position: (f64, f64, f64)) -> Self {
        // A capsule's inertia, taken as the cylinder it mostly is: `m r^2 / 2` about the
        // long axis and `m (3 r^2 + L^2) / 12` across it. The hemispherical caps move
        // both terms by a few percent and are not worth the algebra for a corpse.
        let along = 0.5 * mass * radius * radius;
        let across = mass * (3.0 * radius * radius + length * length) / 12.0;
        let inv = |i: f64| if i > 0.0 { 1.0 / i } else { 0.0 };
        Body {
            position,
            orientation: Quaternion::identity(),
            velocity: (0.0, 0.0, 0.0),
            angular_velocity: (0.0, 0.0, 0.0),
            inv_mass: if mass > 0.0 { 1.0 / mass } else { 0.0 },
            inv_inertia: (inv(across), inv(along), inv(across)),
            radius,
            half_length: 0.5 * length,
        }
    }

    /// The same body given a capsule to collide as.
    ///
    /// Pairs with [`Body::pinned`], which has no extent of its own: a pinned body handed
    /// a shape is an immovable collider -- a floor, a wall, a vehicle that everything
    /// else piles against -- and the solver moves the rest of the world around it.
    pub fn shaped(mut self, radius: f64, length: f64) -> Self {
        self.radius = radius.max(0.0);
        self.half_length = 0.5 * length.max(0.0);
        self
    }

    /// The same body, pinned: infinite mass and infinite inertia, so the solver moves
    /// everything else around it.
    pub fn pinned(position: (f64, f64, f64)) -> Self {
        Body {
            position,
            orientation: Quaternion::identity(),
            velocity: (0.0, 0.0, 0.0),
            angular_velocity: (0.0, 0.0, 0.0),
            inv_mass: 0.0,
            inv_inertia: (0.0, 0.0, 0.0),
            radius: 0.0,
            half_length: 0.0,
        }
    }
}

/// What holds two bodies together, by index into the [`Skeleton`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Joint {
    /// **A point shared by two bodies**, each anchor given in its own body's frame. The
    /// shoulder and the hip: three degrees of rotational freedom, none of translation.
    Ball {
        a: usize,
        b: usize,
        anchor_a: (f64, f64, f64),
        anchor_b: (f64, f64, f64),
    },
    /// **A point shared, plus an axis shared, plus a range on the angle about it.** The
    /// elbow and the knee: one degree of freedom, and it does not go backwards.
    ///
    /// `axis_a` and `axis_b` are the hinge axis written in each body's own frame, and
    /// `min`/`max` bound the angle from `b`'s rest orientation about it, in radians.
    Hinge {
        a: usize,
        b: usize,
        anchor_a: (f64, f64, f64),
        anchor_b: (f64, f64, f64),
        axis_a: (f64, f64, f64),
        axis_b: (f64, f64, f64),
        min: f64,
        max: f64,
    },
}

impl Joint {
    fn bodies(&self) -> (usize, usize) {
        match *self {
            Joint::Ball { a, b, .. } => (a, b),
            Joint::Hinge { a, b, .. } => (a, b),
        }
    }
}

// -- small vector helpers, local because this module is the only user -------------

#[inline]
fn add(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
}

#[inline]
fn sub(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 - b.0, a.1 - b.1, a.2 - b.2)
}

#[inline]
fn scale(a: (f64, f64, f64), k: f64) -> (f64, f64, f64) {
    (a.0 * k, a.1 * k, a.2 * k)
}

#[inline]
fn dot(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

#[inline]
fn cross(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    )
}

#[inline]
fn length(a: (f64, f64, f64)) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
fn normalized(a: (f64, f64, f64)) -> Option<(f64, f64, f64)> {
    let n = length(a);
    if n > 1e-12 {
        Some(scale(a, 1.0 / n))
    } else {
        None
    }
}

/// `q v q*` for a **unit** quaternion, as the cross-product form rather than as two
/// quaternion multiplies.
///
/// [`Quaternion::rotate_point`] is the general one: it normalises its receiver, takes a
/// true inverse and multiplies twice, which is a square root and eight divisions before
/// any rotating happens. That is the right answer for a quaternion of unknown length and
/// the wrong one here, where every orientation is unit by construction --
/// [`Skeleton::add_body`] and [`Skeleton::set_body`] normalise on the way in and every
/// write inside the solver ends in `normalized`.
///
/// Measured on a heap of ten thousand bodies, a solver pass calls this about thirty times
/// per contact, and swapping the general form for this one took the contact pass from
/// 11.6 ms to 4.1 ms. It is the same rotation to the last bit the general form would give
/// a unit quaternion; it is not an approximation.
#[inline]
fn rotate(q: Quaternion, v: (f64, f64, f64)) -> (f64, f64, f64) {
    let u = (q.x, q.y, q.z);
    let t = scale(cross(u, v), 2.0);
    add(add(v, scale(t, q.w)), cross(u, t))
}

/// The same rotation backwards, `q* v q`. The conjugate is the inverse for a unit
/// quaternion, so this costs nothing the forward one does not.
#[inline]
fn rotate_inv(q: Quaternion, v: (f64, f64, f64)) -> (f64, f64, f64) {
    rotate(
        Quaternion {
            w: q.w,
            x: -q.x,
            y: -q.y,
            z: -q.z,
        },
        v,
    )
}

/// A quaternion made unit, for the case this module is always in: one that is already
/// nearly unit.
///
/// [`Quaternion::normalized`] divides all four components by the magnitude -- a square
/// root and **four divisions**, and a division is a dozen-odd cycles that do not pipeline
/// with each other. Multiplying by the reciprocal once is the same answer for a tenth of
/// the latency.
///
/// And the near-unit case skips the square root as well. Every write inside the solver is
/// a product of two unit quaternions, so its magnitude is one to within a few ulp before
/// this is called, and one step of Newton's method from a guess of one is
/// `(3 - m) / 2`. Its error is `(3/8)(m - 1)^2`, which over the band it is allowed here
/// -- a part in a billion -- is four parts in `10^19`, under an ulp of the value itself.
/// It is not an approximation at this distance from unit; it is the same double.
///
/// Measured on the heap, the solver renormalises about seventy-five thousand times a pass
/// and the sweeps ten thousand more, and this took a solve pass from 2.14 ms to 1.83 ms.
#[inline]
fn renormalized(q: Quaternion) -> Quaternion {
    let m = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    let k = if (m - 1.0).abs() < 1e-9 {
        0.5 * (3.0 - m)
    } else if m > 1e-20 {
        1.0 / m.sqrt()
    } else {
        return Quaternion::identity();
    };
    Quaternion {
        w: q.w * k,
        x: q.x * k,
        y: q.y * k,
        z: q.z * k,
    }
}

/// A body's inverse inertia written out in world space, as the symmetric matrix it is.
///
/// `I^-1` is diagonal in the body's own frame, so `R I^-1 R^T` in world space, and the
/// obvious way to apply it is to rotate the vector into the body, scale by three numbers
/// and rotate back. That is right, and it is the wrong shape when a body's tensor is
/// wanted five or six times over: a contact asks for it twice for the generalised inverse
/// masses, twice for the impulse folds, once for the rolling resistance and once more for
/// friction, and each of those was a pair of quaternion rotations.
///
/// Built flat it is one rotation matrix and three outer products, and every use after that
/// is nine multiplies. Only six of the nine entries are stored because `R I^-1 R^T` is
/// symmetric for any `R`, which is not an approximation but the shape of the thing.
#[derive(Clone, Copy, Debug)]
pub(super) struct SymMat3 {
    xx: f64,
    yy: f64,
    zz: f64,
    xy: f64,
    xz: f64,
    yz: f64,
}

impl SymMat3 {
    /// `R diag(inv_inertia) R^T`, for the unit quaternion `q`.
    #[inline]
    fn of(q: Quaternion, inv_inertia: (f64, f64, f64)) -> Self {
        let (w, x, y, z) = (q.w, q.x, q.y, q.z);
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, xz, yz) = (x * y, x * z, y * z);
        let (wx, wy, wz) = (w * x, w * y, w * z);
        // The columns of R, which are the body's own axes written in world space.
        let c0 = (1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy));
        let c1 = (2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx));
        let c2 = (2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy));
        let (ia, ib, ic) = inv_inertia;
        SymMat3 {
            xx: ia * c0.0 * c0.0 + ib * c1.0 * c1.0 + ic * c2.0 * c2.0,
            yy: ia * c0.1 * c0.1 + ib * c1.1 * c1.1 + ic * c2.1 * c2.1,
            zz: ia * c0.2 * c0.2 + ib * c1.2 * c1.2 + ic * c2.2 * c2.2,
            xy: ia * c0.0 * c0.1 + ib * c1.0 * c1.1 + ic * c2.0 * c2.1,
            xz: ia * c0.0 * c0.2 + ib * c1.0 * c1.2 + ic * c2.0 * c2.2,
            yz: ia * c0.1 * c0.2 + ib * c1.1 * c1.2 + ic * c2.1 * c2.2,
        }
    }

    #[inline]
    fn apply(&self, v: (f64, f64, f64)) -> (f64, f64, f64) {
        (
            self.xx * v.0 + self.xy * v.1 + self.xz * v.2,
            self.xy * v.0 + self.yy * v.1 + self.yz * v.2,
            self.xz * v.0 + self.yz * v.1 + self.zz * v.2,
        )
    }
}

/// Where a body is and how hard it is to move, gathered once for a constraint.
///
/// The solve used to index eight parallel arrays a dozen times over per constraint --
/// `position[a]`, `orientation[a]`, `inv_mass[a]` and so on, each a bounds-checked load
/// from a different cache line. Gathering the body once and passing it down makes the
/// scattered half of the access pattern two reads per body instead of a dozen, and it is
/// what lets [`SymMat3`] be built once rather than implied six times.
#[derive(Clone, Copy, Debug)]
pub(super) struct Pose {
    pub position: (f64, f64, f64),
    pub orientation: Quaternion,
    pub inv_mass: f64,
    pub inv_inertia: (f64, f64, f64),
    pub world_inv_inertia: SymMat3,
}

impl Pose {
    /// Whether the solver can move this body at all. A pinned one has neither mass nor
    /// inertia to give.
    #[inline]
    fn movable(&self) -> bool {
        self.inv_mass > 0.0 || self.inv_inertia != (0.0, 0.0, 0.0)
    }
}

/// A [`Pose`] plus what a contact needs and a joint does not: where the body was when the
/// step began, and how fat it is.
#[derive(Clone, Copy, Debug)]
pub(super) struct Gathered {
    pub now: Pose,
    pub prev_position: (f64, f64, f64),
    pub prev_orientation: Quaternion,
    pub radius: f64,
}

/// One body's share of one constraint's correction: where to move it and how to turn it.
///
/// **It never leaves the thread that made it.** A constraint computes its pair of these
/// and writes them into the bodies itself, so this is a local that lives in registers for
/// a few dozen instructions and is gone. That is worth saying because it used to be the
/// opposite: a colour's worth of them went into a buffer for a serial half to read back,
/// and at a hundred and twenty bytes each -- the free fields doubled it -- a pass over a
/// heap of ten thousand bodies moved ten megabytes through that buffer twice. Splitting
/// the type into a lean one for joints and a fat one for contacts would have halved a
/// cost that did not need to exist. See [`scatter`].
#[derive(Clone, Copy, Debug)]
struct Correction {
    body: usize,
    translation: (f64, f64, f64),
    /// The quaternion *delta* to left-multiply, already weighted. Identity when the body
    /// is only being moved.
    rotation: Quaternion,
    /// **The part of the same correction that must not read back as velocity.**
    ///
    /// A position-based solver derives velocity from how far a body moved, so a body
    /// lifted out of an overlap it was already in reads as having travelled under its own
    /// power: spawn two bodies inside each other and they leave at several metres a
    /// second. But a body that drove into a surface *during this step* must read as
    /// having been stopped, or nothing ever collides.
    ///
    /// Both are position corrections and no amount of clamping tells them apart, so they
    /// are separated at the source: a contact may turn into velocity only the overlap it
    /// made this step, and whatever was already there is carried here. The applying half
    /// moves the previous position by exactly this much as well, which leaves the
    /// difference the velocity is read from untouched.
    free_translation: (f64, f64, f64),
    free_rotation: Quaternion,
}

impl Correction {
    fn none() -> Self {
        Correction {
            body: usize::MAX,
            translation: (0.0, 0.0, 0.0),
            rotation: Quaternion::identity(),
            free_translation: (0.0, 0.0, 0.0),
            free_rotation: Quaternion::identity(),
        }
    }
}

/// A set of bodies and the joints between them, solved together.
///
/// See the module header for why the bodies are arrays, and why the joints are coloured.
#[derive(Clone, Debug)]
pub struct Skeleton {
    position: Vec<(f64, f64, f64)>,
    orientation: Vec<Quaternion>,
    velocity: Vec<(f64, f64, f64)>,
    angular_velocity: Vec<(f64, f64, f64)>,
    inv_mass: Vec<f64>,
    inv_inertia: Vec<(f64, f64, f64)>,
    radius: Vec<f64>,
    half_length: Vec<f64>,

    joints: Vec<Joint>,
    /// Joint indices grouped so that no two joints in a group share a body. Rebuilt when
    /// the joint set changes, not per step. See the module header.
    colours: Vec<Vec<usize>>,
    coloured: bool,

    prev_position: Vec<(f64, f64, f64)>,
    prev_orientation: Vec<Quaternion>,

    /// Which bodies are directly jointed to each body, as one run per body, so contact
    /// generation can skip them. Rebuilt with the colouring. See [`Jointed`].
    jointed_start: Vec<u32>,
    jointed_to: Vec<u32>,
    /// Candidate pairs from the broad phase, and the contacts that survived the narrow
    /// one. Both are cleared and refilled per step rather than reallocated.
    pairs: Vec<(usize, usize)>,
    contacts: Vec<Contact>,
    /// Where each chunk of the narrow phase puts its contacts before they are
    /// concatenated. See [`Skeleton::build_contacts`].
    contact_scratch: Vec<Vec<Contact>>,
    contact_colours: Vec<Vec<usize>>,
    /// Contacts on a body that has already used every colour the bitmask can hold. See
    /// [`Skeleton::colour_contacts`]; solved serially, and in practice empty.
    contact_overflow: Vec<usize>,
    /// One word per body: bit `c` set means this body already has a contact in colour
    /// `c`. A bitmask rather than a set per body because this is rebuilt every step.
    colour_bits: Vec<u64>,
    /// The plane everything rests on, as a unit normal and the distance along it, or
    /// `None` for a skeleton that hangs in space. See [`Skeleton::set_ground`].
    ground: Option<((f64, f64, f64), f64)>,
    ground_contacts: Vec<GroundContact>,
    /// Ground contacts split so a colour can be solved in parallel. A body has at most
    /// one contact per end, so two sets are always enough and no colouring pass is
    /// needed: the first contact found for a body goes in one, the second in the other.
    ground_colours: [Vec<usize>; 2],
    /// One per body: the vector from one end of its contact with the plane to the other,
    /// or zero where it touches at a point. Rebuilt with the ground contacts. See
    /// [`contacts::patch_arm`].
    ground_span: Vec<(f64, f64, f64)>,
    friction: f64,
    rolling_resistance: f64,
    /// What each contact has already spent this step. See [`Spent`], which is also where
    /// the tangential half's shape is argued.
    ///
    /// **`contact_impulse` is indexed by contact and `ground_impulse` by body**, and the
    /// difference is deliberate: see [`Skeleton::build_contacts`] on why the two ends of
    /// one capsule on the plane share one budget, and [`Skeleton::solve_ground_colour`]
    /// on why indexing it by body is still sound under the parallel scatter.
    ///
    /// **Coulomb's limit is a budget for the whole step, not for each solver pass**, and
    /// it has to be carried across the passes or the coefficient stops meaning anything.
    /// Spending the full limit every pass multiplies the friction by the iteration count:
    /// measured, a slope that should have let go at twenty-seven degrees still held at
    /// forty. Dividing the limit between the passes instead fails the other way, because
    /// the first pass removes nearly all the overlap and the later ones have almost no
    /// normal impulse left to be a fraction of -- at thirty-two iterations that version
    /// slid at five degrees.
    ///
    /// Totals have neither problem: the tangential impulse over the step is held under
    /// `friction` times the normal impulse over the step, which is the law itself, and
    /// the answer stops depending on the quality dial.
    contact_impulse: Vec<Spent>,
    ground_impulse: Vec<Spent>,

    /// The broad phase. See [`broadphase`] for why it is a grid.
    grid: Grid,
}

impl Default for Skeleton {
    /// Empty, and with [`DEFAULT_FRICTION`] between its bodies. Written out rather than
    /// derived because a derived one would start at zero friction, and a pile with no
    /// friction slides flat without anything reporting an error.
    fn default() -> Self {
        Skeleton {
            position: Vec::new(),
            orientation: Vec::new(),
            velocity: Vec::new(),
            angular_velocity: Vec::new(),
            inv_mass: Vec::new(),
            inv_inertia: Vec::new(),
            radius: Vec::new(),
            half_length: Vec::new(),
            joints: Vec::new(),
            colours: Vec::new(),
            coloured: false,
            prev_position: Vec::new(),
            prev_orientation: Vec::new(),
            jointed_start: Vec::new(),
            jointed_to: Vec::new(),
            pairs: Vec::new(),
            contacts: Vec::new(),
            contact_scratch: Vec::new(),
            contact_colours: Vec::new(),
            contact_overflow: Vec::new(),
            colour_bits: Vec::new(),
            ground: None,
            ground_contacts: Vec::new(),
            ground_colours: [Vec::new(), Vec::new()],
            ground_span: Vec::new(),
            friction: DEFAULT_FRICTION,
            rolling_resistance: DEFAULT_ROLLING_RESISTANCE,
            contact_impulse: Vec::new(),
            ground_impulse: Vec::new(),
            grid: Grid::default(),
        }
    }
}

impl Skeleton {
    pub fn new() -> Self {
        Skeleton::default()
    }

    /// The Coulomb coefficient between every pair of bodies. Zero turns friction off and
    /// leaves only the non-penetration constraint.
    pub fn set_friction(&mut self, friction: f64) {
        self.friction = friction.max(0.0);
    }

    /// Rolling resistance between bodies, as a fraction of the contact radius. Zero lets
    /// a capsule roll like the ideal cylinder it is; see [`DEFAULT_ROLLING_RESISTANCE`]
    /// for why that is not what a pile wants.
    pub fn set_rolling_resistance(&mut self, resistance: f64) {
        self.rolling_resistance = resistance.max(0.0);
    }

    /// **The ground**: the plane `dot(normal, p) = distance`, which every shaped body
    /// rests on. `None` removes it.
    ///
    /// A plane rather than a wide pinned body, because the two are not equivalent where
    /// it matters. A capsule lying across a cylinder touches it at one point however fat
    /// the cylinder is, and one point cannot hold a body flat -- it rocks, and the solve
    /// spends its iterations on that instead of on the pile. Against a plane the same
    /// capsule gets a contact at each end. Terrain that is not flat is a caller's
    /// problem for now; this is the half of it every pile needs.
    pub fn set_ground(&mut self, normal: (f64, f64, f64), distance: f64) {
        self.ground = normalized(normal).map(|n| (n, distance));
    }

    /// Removes the ground plane.
    pub fn clear_ground(&mut self) {
        self.ground = None;
    }

    /// How many contacts the last [`Skeleton::step`] found. The number a broad phase is
    /// judged against, and the one that says whether a pile is resting or interpenetrating.
    pub fn contact_count(&self) -> usize {
        self.contacts.len()
    }

    pub fn len(&self) -> usize {
        self.position.len()
    }

    pub fn is_empty(&self) -> bool {
        self.position.is_empty()
    }

    /// Adds a body and returns its index, which is what joints are written against.
    /// Orientations are normalised on the way in, and that is load-bearing rather than
    /// tidy: the solver's rotation is the unit-quaternion form (see [`rotate`]), which is
    /// only the right answer for a quaternion of length one. This and
    /// [`Skeleton::set_body`] are the only places one can arrive from outside; everything
    /// the solver itself writes is already unit.
    pub fn add_body(&mut self, mut body: Body) -> usize {
        body.orientation = body.orientation.normalized();
        self.position.push(body.position);
        self.orientation.push(body.orientation);
        self.velocity.push(body.velocity);
        self.angular_velocity.push(body.angular_velocity);
        self.inv_mass.push(body.inv_mass);
        self.inv_inertia.push(body.inv_inertia);
        self.radius.push(body.radius);
        self.half_length.push(body.half_length);
        self.prev_position.push(body.position);
        self.prev_orientation.push(body.orientation);
        self.colour_bits.push(0);
        self.position.len() - 1
    }

    /// One body, gathered out of the arrays.
    pub fn body(&self, i: usize) -> Body {
        Body {
            position: self.position[i],
            orientation: self.orientation[i],
            velocity: self.velocity[i],
            angular_velocity: self.angular_velocity[i],
            inv_mass: self.inv_mass[i],
            inv_inertia: self.inv_inertia[i],
            radius: self.radius[i],
            half_length: self.half_length[i],
        }
    }

    /// Writes one body back. The whole body, because a caller that has one has usually
    /// changed more than one field of it.
    pub fn set_body(&mut self, i: usize, body: Body) {
        self.position[i] = body.position;
        // Unit on the way in; see [`Skeleton::add_body`].
        self.orientation[i] = body.orientation.normalized();
        self.velocity[i] = body.velocity;
        self.angular_velocity[i] = body.angular_velocity;
        self.inv_mass[i] = body.inv_mass;
        self.inv_inertia[i] = body.inv_inertia;
        self.radius[i] = body.radius;
        self.half_length[i] = body.half_length;
    }

    pub fn position(&self, i: usize) -> (f64, f64, f64) {
        self.position[i]
    }

    pub fn orientation(&self, i: usize) -> Quaternion {
        self.orientation[i]
    }

    pub fn velocity(&self, i: usize) -> (f64, f64, f64) {
        self.velocity[i]
    }

    pub fn angular_velocity(&self, i: usize) -> (f64, f64, f64) {
        self.angular_velocity[i]
    }

    pub fn set_angular_velocity(&mut self, i: usize, w: (f64, f64, f64)) {
        self.angular_velocity[i] = w;
    }

    pub fn set_velocity(&mut self, i: usize, v: (f64, f64, f64)) {
        self.velocity[i] = v;
    }

    /// Adds a joint. Returns `false` and adds nothing if it names a body that does not
    /// exist, or joints a body to itself -- an out-of-range index is a caller's bug and
    /// panicking in a solver that runs per frame is worse than refusing.
    pub fn add_joint(&mut self, joint: Joint) -> bool {
        let (a, b) = joint.bodies();
        let n = self.position.len();
        if a >= n || b >= n || a == b {
            return false;
        }
        self.joints.push(joint);
        self.coloured = false;
        true
    }

    pub fn joints(&self) -> &[Joint] {
        &self.joints
    }

    /// How the joints were partitioned. Exposed because the colour count is the thing
    /// that decides how parallel a step can be, and a caller tuning a rig wants to see it.
    pub fn colours(&mut self) -> &[Vec<usize>] {
        self.recolour();
        &self.colours
    }

    /// **Greedy colouring**: each joint takes the lowest colour no joint already coloured
    /// on either of its bodies is using.
    ///
    /// Greedy rather than optimal because optimal colouring is NP-hard and the gain would
    /// be at most a colour or two on a graph where every vertex has degree three or four.
    /// A skeleton lands on four or five either way.
    fn recolour(&mut self) {
        if self.coloured {
            return;
        }
        for set in self.colours.iter_mut() {
            set.clear();
        }
        // Which colours are already taken on each body. Indexed by body, holding the
        // highest colour seen plus a bitmask of the low ones, would be faster; a small
        // vec per body is clearer and this runs when the rig changes, not per step.
        let mut taken: Vec<Vec<usize>> = vec![Vec::new(); self.position.len()];
        for (index, joint) in self.joints.iter().enumerate() {
            let (a, b) = joint.bodies();
            let mut colour = 0;
            while taken[a].contains(&colour) || taken[b].contains(&colour) {
                colour += 1;
            }
            taken[a].push(colour);
            taken[b].push(colour);
            if colour >= self.colours.len() {
                self.colours.resize_with(colour + 1, Vec::new);
            }
            self.colours[colour].push(index);
        }
        self.colours.retain(|set| !set.is_empty());

        // The pairs contact generation must not produce, as a run per body. See
        // [`Jointed`] for why it is that shape and not a sorted list of pairs.
        let bodies = self.position.len();
        self.jointed_start.clear();
        self.jointed_start.resize(bodies + 1, 0);
        for joint in self.joints.iter() {
            let (a, b) = joint.bodies();
            self.jointed_start[a] += 1;
            self.jointed_start[b] += 1;
        }
        let mut running = 0u32;
        for slot in self.jointed_start.iter_mut() {
            let count = *slot;
            *slot = running;
            running += count;
        }
        self.jointed_to.clear();
        self.jointed_to.resize(running as usize, 0);
        // A cursor per body, allocated here rather than kept on the struct because this
        // runs when the rig changes and not per step.
        let mut cursor = self.jointed_start.clone();
        for joint in self.joints.iter() {
            let (a, b) = joint.bodies();
            for (from, to) in [(a, b), (b, a)] {
                self.jointed_to[cursor[from] as usize] = to as u32;
                cursor[from] += 1;
            }
        }

        self.coloured = true;
    }

    /// Which bodies each body is jointed to. See [`Jointed`].
    fn jointed(&self) -> Jointed<'_> {
        Jointed {
            start: &self.jointed_start,
            to: &self.jointed_to,
        }
    }

    /// Whether a joint holds these two bodies together, which is the pair contact
    /// generation must not produce.
    pub fn is_jointed(&mut self, a: usize, b: usize) -> bool {
        self.recolour();
        self.jointed().holds(a, b)
    }

    /// Candidate pairs for the narrow phase, from the broad phase.
    fn find_pairs(&mut self) {
        let mut pairs = std::mem::take(&mut self.pairs);
        pairs.clear();
        let mut grid = std::mem::take(&mut self.grid);
        grid.rebuild(&self.position, &self.radius, &self.half_length);
        grid.pairs(&self.position, &self.inv_mass, self.jointed(), &mut pairs);
        self.grid = grid;
        self.pairs = pairs;
    }

    /// The narrow phase: which candidates are actually touching, and where.
    fn build_contacts(&mut self) {
        let mut contacts = std::mem::take(&mut self.contacts);
        contacts.clear();
        let position = &self.position;
        let orientation = &self.orientation;
        let radius = &self.radius;
        let half_length = &self.half_length;
        let test = |&(a, b): &(usize, usize)| {
            capsule_contact(a, b, position, orientation, radius, half_length)
                .into_iter()
                .flatten()
        };
        // Chunked into buffers this struct owns, rather than `par_extend` over a
        // flat-mapping parallel iterator. The number of contacts a pair yields is not
        // known in advance, so `par_extend` cannot write in place: it builds a tree of
        // collections and folds them together, and measured on a heap of ten thousand
        // bodies that cost 3.7 ms against 0.5 for the same tests run this way. Chunked by
        // a fixed count rather than by the thread count, so the contact list is the same
        // list in the same order on every machine.
        if self.pairs.len() >= PARALLEL_FLOOR {
            let mut scratch = std::mem::take(&mut self.contact_scratch);
            let chunks = self.pairs.len().div_ceil(NARROW_CHUNK);
            if scratch.len() < chunks {
                scratch.resize_with(chunks, Vec::new);
            }
            scratch[..chunks]
                .par_iter_mut()
                .zip(self.pairs.par_chunks(NARROW_CHUNK))
                .for_each(|(into, chunk)| {
                    into.clear();
                    into.extend(chunk.iter().flat_map(test));
                });
            for filled in scratch[..chunks].iter() {
                contacts.extend_from_slice(filled);
            }
            self.contact_scratch = scratch;
        } else {
            contacts.extend(self.pairs.iter().flat_map(test));
        }
        self.contacts = contacts;
        self.contact_impulse.clear();
        self.contact_impulse
            .resize(self.contacts.len(), Spent::default());

        self.ground_contacts.clear();
        self.ground_colours[0].clear();
        self.ground_colours[1].clear();
        self.ground_span.clear();
        self.ground_span.resize(self.position.len(), (0.0, 0.0, 0.0));
        let Some((normal, distance)) = self.ground else {
            return;
        };
        for i in 0..self.position.len() {
            if self.inv_mass[i] <= 0.0 {
                continue;
            }
            let before = self.ground_contacts.len();
            ground_contacts(
                i,
                self.position[i],
                self.orientation[i],
                self.radius[i],
                self.half_length[i],
                normal,
                distance,
                &mut self.ground_contacts,
            );
            for (nth, index) in (before..self.ground_contacts.len()).enumerate() {
                self.ground_colours[nth.min(1)].push(index);
            }
            // How far this body's contact with the plane reaches, as the vector from one
            // end of it to the other, or zero where it touches at a point. See
            // [`contacts::patch_arm`] for what a patch does that a point cannot.
            self.ground_span[i] = match self.ground_contacts[before..] {
                [first, second] => {
                    let a = add(self.position[i], rotate(self.orientation[i], first.local));
                    let b = add(self.position[i], rotate(self.orientation[i], second.local));
                    sub(b, a)
                }
                _ => (0.0, 0.0, 0.0),
            };
        }

        // **One budget per body, not one per end.** A capsule lying on the plane touches
        // it along a line and gets a contact at each end of that line, but the two are
        // samples of a single contact *patch*: they express the same tangential
        // constraint -- a rigid body's contact line cannot slide at one end and stay put
        // at the other -- and Coulomb's limit belongs to the patch, `friction` times the
        // whole normal load it carries.
        //
        // Giving each end its own limit out of its own share of the load is what a point
        // contact would want, and it is wrong here in a way that shows: as the body tips
        // the load moves between the ends, so one end's cone shrinks while it is still
        // being asked to hold, and the pair settles into equal and opposite impulses that
        // cancel and leave the resultant short. Pooling the budget lets whichever end is
        // loaded supply the grip, which is what the patch does.
        //
        // **This is why the two ends must stay in different colours, and that is now a
        // memory-safety requirement as well as a physical one.** The loop above puts a
        // body's first ground contact in set zero and its second in set one, and the two
        // sets are solved one after the other, so the pooled entry is read and written by
        // one thread at a time: within a set, each body appears at most once. That is the
        // same disjointness the body arrays need under [`scatter`], checked by the same
        // `scatter::disjoint` call, which walks the set's bodies rather than its contacts
        // -- so it covers this without extension. Coupling the ends is deliberate: set
        // one must see what set zero spent. Putting them in one parallel set would both
        // race the budget and defeat the pooling.
        self.ground_impulse.clear();
        self.ground_impulse
            .resize(self.position.len(), Spent::default());
    }

    /// **Greedy colouring again, but every step**, because the contact set is new every
    /// step where the joint set is not.
    ///
    /// The taken-colour test is a bit in a word per body instead of the joint version's
    /// vector per body: same algorithm, no allocation, and the whole pass is a handful of
    /// instructions per contact. Sixty-four colours is the price -- a body touched by
    /// more than sixty-four others at once sends its extra contacts to
    /// `contact_overflow`, which is solved serially. Reaching that means a body buried
    /// under sixty-four neighbours, and a pile that deep has worse problems than a
    /// serial tail.
    fn colour_contacts(&mut self) {
        for set in self.contact_colours.iter_mut() {
            set.clear();
        }
        self.contact_overflow.clear();
        for bits in self.colour_bits.iter_mut() {
            *bits = 0;
        }

        for (index, contact) in self.contacts.iter().enumerate() {
            let (a, b) = (contact.a, contact.b);
            let taken = self.colour_bits[a] | self.colour_bits[b];
            if taken == u64::MAX {
                self.contact_overflow.push(index);
                continue;
            }
            let colour = taken.trailing_ones() as usize;
            let bit = 1u64 << colour;
            self.colour_bits[a] |= bit;
            self.colour_bits[b] |= bit;
            if colour >= self.contact_colours.len() {
                self.contact_colours.resize_with(colour + 1, Vec::new);
            }
            self.contact_colours[colour].push(index);
        }
    }

    /// **One step.** Predict under `gravity`, run `iterations` passes over the coloured
    /// joint sets, then read the velocities back out of what moved.
    ///
    /// Allocates nothing. `iterations` is the quality dial: four is enough for a corpse,
    /// and the cost is linear in it.
    pub fn step(&mut self, dt: f64, gravity: (f64, f64, f64), iterations: usize) {
        if dt <= 0.0 || self.position.is_empty() {
            return;
        }
        self.recolour();

        self.prev_position.copy_from_slice(&self.position);
        self.prev_orientation.copy_from_slice(&self.orientation);

        // **One sweep, not three.** Falling, travelling and spinning were a pass each,
        // which is three reads of every array and three handings of the same ten thousand
        // bodies to the thread pool. Nothing in them crosses bodies, so they fuse: the
        // arrays are read once and the pool is asked once. Measured on the heap, a fork
        // and a join cost 30 to 70 us on their own, and the whole predict is under a
        // millisecond -- the scheduling was a real fraction of it.
        let g = gravity;
        let wide = self.position.len() >= PARALLEL_FLOOR;

        let predict = |((((p, v), q), &inv_m), w): (
            (((&mut (f64, f64, f64), &mut (f64, f64, f64)), &mut Quaternion), &f64),
            &(f64, f64, f64),
        )| {
            if inv_m > 0.0 {
                *v = add(*v, scale(g, dt));
            }
            *p = add(*p, scale(*v, dt));
            *q = integrate_spin(*q, *w, dt);
        };

        if wide {
            self.position
                .par_iter_mut()
                .zip(self.velocity.par_iter_mut())
                .zip(self.orientation.par_iter_mut())
                .zip(self.inv_mass.par_iter())
                .zip(self.angular_velocity.par_iter())
                .for_each(predict);
        } else {
            self.position
                .iter_mut()
                .zip(self.velocity.iter_mut())
                .zip(self.orientation.iter_mut())
                .zip(self.inv_mass.iter())
                .zip(self.angular_velocity.iter())
                .for_each(predict);
        }

        // Contacts are found once, from the predicted positions, and then solved on every
        // iteration. Regenerating them per iteration would double the narrow phase for a
        // set that barely changes across a step, and would let a pair appear and vanish
        // between passes so that nothing ever converged.
        self.find_pairs();
        self.build_contacts();
        self.colour_contacts();

        for _ in 0..iterations.max(1) {
            for colour in 0..self.colours.len() {
                self.solve_colour(colour);
            }
            for colour in 0..self.contact_colours.len() {
                if !self.contact_colours[colour].is_empty() {
                    self.solve_contact_colour(colour);
                }
            }
            self.solve_contact_overflow();
            for colour in 0..2 {
                if !self.ground_colours[colour].is_empty() {
                    self.solve_ground_colour(colour);
                }
            }
        }

        // And one sweep to read both velocities back, for the same reason the predict is
        // one.
        let inv_dt = 1.0 / dt;
        let read_back = |((((v, w), p), prev_p), (q, prev_q)): (
            (
                ((&mut (f64, f64, f64), &mut (f64, f64, f64)), &(f64, f64, f64)),
                &(f64, f64, f64),
            ),
            (&Quaternion, &Quaternion),
        )| {
            *v = scale(sub(*p, *prev_p), inv_dt);
            // The rotation that happened, as an axis-angle, divided by the step. The
            // `w < 0` flip keeps the short way round: a quaternion and its negation are
            // the same orientation, and without the check a body can read as spinning
            // almost a full turn when it barely moved.
            let delta = q.multiply(&prev_q.conjugate());
            let sign = if delta.w < 0.0 { -1.0 } else { 1.0 };
            *w = scale((delta.x, delta.y, delta.z), 2.0 * inv_dt * sign);
        };

        if wide {
            self.velocity
                .par_iter_mut()
                .zip(self.angular_velocity.par_iter_mut())
                .zip(self.position.par_iter())
                .zip(self.prev_position.par_iter())
                .zip(self.orientation.par_iter().zip(self.prev_orientation.par_iter()))
                .for_each(read_back);
        } else {
            self.velocity
                .iter_mut()
                .zip(self.angular_velocity.iter_mut())
                .zip(self.position.iter())
                .zip(self.prev_position.iter())
                .zip(self.orientation.iter().zip(self.prev_orientation.iter()))
                .for_each(read_back);
        }
    }

    /// The four body arrays every correction writes, as the disjoint-scatter view a
    /// colour is applied through. See [`scatter`] for why, and for the safety argument.
    #[inline]
    fn writable(&mut self) -> Bodies {
        Bodies::of(
            &mut self.position,
            &mut self.orientation,
            &mut self.prev_position,
            &mut self.prev_orientation,
        )
    }

    /// One colour: every joint in it gathers its two bodies, works out its two
    /// corrections and writes them straight back. No two joints in a colour name the same
    /// body, so the threads never meet and the order cannot change the answer.
    fn solve_colour(&mut self, colour: usize) {
        #[cfg(debug_assertions)]
        scatter::disjoint(
            self.position.len(),
            "joint",
            self.colours[colour]
                .iter()
                .flat_map(|&k| {
                    let (a, b) = self.joints[k].bodies();
                    [a, b]
                }),
        );

        let bodies = self.writable();
        let joints = &self.joints;
        let inv_mass = &self.inv_mass;
        let inv_inertia = &self.inv_inertia;
        let set = &self.colours[colour];

        // SAFETY: every body this closure reads or writes is named by joint `k` and by no
        // other joint in the colour, so no two threads address one element of any array.
        // That partition is established by `Skeleton::recolour`, which gives each joint
        // the lowest colour neither of its bodies already holds; a change there that let
        // two joints on one body share a colour is what would make this a data race, and
        // is why the `disjoint` check above runs in debug builds. See [`scatter`] for the
        // argument in full.
        let solve = |&k: &usize| unsafe {
            let joint = joints[k];
            let (a, b) = joint.bodies();
            let first = bodies.pose(a, inv_mass, inv_inertia);
            let second = bodies.pose(b, inv_mass, inv_inertia);
            bodies.apply(solve_joint(joint, &first, &second));
        };
        if set.len() >= PARALLEL_FLOOR {
            set.par_iter().for_each(solve);
        } else {
            set.iter().for_each(solve);
        }
    }

    /// One colour of contacts, the same way [`Skeleton::solve_colour`] does one colour of
    /// joints.
    fn solve_contact_colour(&mut self, colour: usize) {
        #[cfg(debug_assertions)]
        scatter::disjoint(
            self.position.len(),
            "contact",
            self.contact_colours[colour]
                .iter()
                .flat_map(|&k| [self.contacts[k].a, self.contacts[k].b]),
        );

        let bodies = self.writable();
        let impulse = scatter::Cells::of(&mut self.contact_impulse);
        let contacts = &self.contacts;
        let inv_mass = &self.inv_mass;
        let inv_inertia = &self.inv_inertia;
        let friction = self.friction;
        let rolling = self.rolling_resistance;
        let radius = &self.radius;
        let set = &self.contact_colours[colour];

        // SAFETY: every body this closure reads or writes is named by contact `k` and by
        // no other contact in the colour, so no two threads address one element of any
        // body array; and the running impulse is indexed by the contact, which is unique
        // to this closure call by construction. That partition is established by
        // `Skeleton::colour_contacts`, which gives each contact a colour neither of its
        // bodies has a bit set for and sends a contact it cannot place to
        // `contact_overflow` -- solved one at a time, never here. A change there that
        // placed a contact in a colour one of its bodies already used is what would make
        // this a data race, and is why the `disjoint` check above runs in debug builds.
        // See [`scatter`] for the argument in full.
        let solve = |&k: &usize| unsafe {
            let contact = contacts[k];
            let first = bodies.gather(contact.a, inv_mass, inv_inertia, radius);
            let second = bodies.gather(contact.b, inv_mass, inv_inertia, radius);
            let (corrections, totals) =
                solve_contact(contact, &first, &second, friction, rolling, impulse.get(k));
            impulse.set(k, totals);
            bodies.apply(corrections);
        };
        if set.len() >= PARALLEL_FLOOR {
            set.par_iter().for_each(solve);
        } else {
            set.iter().for_each(solve);
        }
    }

    /// The contacts colouring could not place, solved one at a time. Each reads the
    /// positions the one before it wrote, which is what makes it safe without a colour --
    /// and slow, which is why it is a tail and not the main path.
    fn solve_contact_overflow(&mut self) {
        if self.contact_overflow.is_empty() {
            return;
        }
        let bodies = self.writable();
        let impulse = scatter::Cells::of(&mut self.contact_impulse);
        let contacts = &self.contacts;
        let inv_mass = &self.inv_mass;
        let inv_inertia = &self.inv_inertia;
        let friction = self.friction;
        let rolling = self.rolling_resistance;
        let radius = &self.radius;

        // SAFETY: one at a time on this thread, so nothing is shared at all.
        for &k in self.contact_overflow.iter() {
            unsafe {
                let contact = contacts[k];
                let first = bodies.gather(contact.a, inv_mass, inv_inertia, radius);
                let second = bodies.gather(contact.b, inv_mass, inv_inertia, radius);
                let (corrections, totals) =
                    solve_contact(contact, &first, &second, friction, rolling, impulse.get(k));
                impulse.set(k, totals);
                bodies.apply(corrections);
            }
        }
    }

    /// One set of ground contacts, the same way a colour of pair contacts is done.
    fn solve_ground_colour(&mut self, colour: usize) {
        let Some((normal, distance)) = self.ground else {
            return;
        };
        #[cfg(debug_assertions)]
        scatter::disjoint(
            self.position.len(),
            "ground",
            self.ground_colours[colour]
                .iter()
                .map(|&k| self.ground_contacts[k].body),
        );

        let bodies = self.writable();
        let impulse = scatter::Cells::of(&mut self.ground_impulse);
        let contacts = &self.ground_contacts;
        let inv_mass = &self.inv_mass;
        let inv_inertia = &self.inv_inertia;
        let friction = self.friction;
        let rolling = self.rolling_resistance;
        let radius = &self.radius;
        let span = &self.ground_span;
        let set = &self.ground_colours[colour];

        // SAFETY: the one body this closure reads or writes is named by ground contact
        // `k` and by no other contact in the set, so no two threads address one element
        // of any body array. That partition is established by `Skeleton::build_contacts`,
        // which emits at most one contact per end of a capsule and puts a body's first in
        // set zero and its second in set one. A change to `contacts::ground_contacts`
        // that emitted a third contact for a body is what would make this a data race --
        // it would land in set one alongside the second -- and is why the `disjoint`
        // check above runs in debug builds. See [`scatter`] for the argument in full.
        //
        // **The running impulse here is indexed by the body, not by the contact**, which
        // is the one place that departs from the rule [`scatter`] states, so it needs the
        // invariant said out loud: the two ends of a capsule pool one budget, so
        // `impulse` is addressed at `contact.body`. That is sound for exactly the reason
        // the body arrays are, and under exactly the same check -- the `disjoint` call
        // above walks this set's *bodies*, so it is already asserting that no two entries
        // of this map touch one slot of `ground_impulse` either. `ground_span` is read
        // only, and indexed by the same body.
        let solve = |&k: &usize| unsafe {
            let contact = contacts[k];
            let body = bodies.gather(contact.body, inv_mass, inv_inertia, radius);
            let (correction, totals) = solve_ground(
                contact,
                &body,
                friction,
                rolling,
                normal,
                distance,
                span[contact.body],
                impulse.get(contact.body),
            );
            impulse.set(contact.body, totals);
            bodies.apply([correction, Correction::none()]);
        };
        if set.len() >= PARALLEL_FLOOR {
            set.par_iter().for_each(solve);
        } else {
            set.iter().for_each(solve);
        }
    }
}

/// `q` advanced by angular velocity `w` over `dt`, renormalised. The small-angle
/// integrator every position-based solver uses: exact enough over a frame and far cheaper
/// than an exponential map.
#[inline]
fn integrate_spin(q: Quaternion, w: (f64, f64, f64), dt: f64) -> Quaternion {
    let spin = Quaternion {
        w: 0.0,
        x: w.0,
        y: w.1,
        z: w.2,
    }
    .multiply(&q);
    renormalized(Quaternion {
        w: q.w + 0.5 * dt * spin.w,
        x: q.x + 0.5 * dt * spin.x,
        y: q.y + 0.5 * dt * spin.y,
        z: q.z + 0.5 * dt * spin.z,
    })
}

/// Everything one joint wants done, as two corrections. Reads only; the caller applies.
fn solve_joint(joint: Joint, first: &Pose, second: &Pose) -> [Correction; 2] {
    let mut out = [Correction::none(); 2];
    let (a, b) = joint.bodies();
    out[0].body = a;
    out[1].body = b;

    let (anchor_a, anchor_b) = match joint {
        Joint::Ball {
            anchor_a, anchor_b, ..
        } => (anchor_a, anchor_b),
        Joint::Hinge {
            anchor_a, anchor_b, ..
        } => (anchor_a, anchor_b),
    };

    // -- the positional half: the two anchors are one point ----------------------
    let ra = rotate(first.orientation, anchor_a);
    let rb = rotate(second.orientation, anchor_b);
    let error = sub(add(second.position, rb), add(first.position, ra));
    if let Some(n) = normalized(error) {
        let c = length(error);
        let wa = generalised_inverse_mass(first, ra, n);
        let wb = generalised_inverse_mass(second, rb, n);
        let total = wa + wb;
        if total > 1e-12 {
            let impulse = scale(n, c / total);
            accumulate(&mut out[0], first, ra, impulse, false);
            accumulate(&mut out[1], second, rb, scale(impulse, -1.0), false);
        }
    }

    // -- and, for a hinge, the axis and the range on it --------------------------
    if let Joint::Hinge {
        axis_a,
        axis_b,
        min,
        max,
        ..
    } = joint
    {
        let world_a = rotate(first.orientation, axis_a);
        let world_b = rotate(second.orientation, axis_b);
        let misaligned = cross(world_b, world_a);
        if let Some(n) = normalized(misaligned) {
            let angle = length(misaligned).clamp(-1.0, 1.0).asin();
            share_turn(&mut out, first, second, n, angle);
        }

        if let Some(axis) = normalized(world_a) {
            let angle = hinge_angle(
                first.orientation,
                second.orientation,
                axis,
                axis_a,
                axis_b,
            );
            let excess = if angle < min {
                angle - min
            } else if angle > max {
                angle - max
            } else {
                0.0
            };
            if excess != 0.0 {
                share_turn(&mut out, first, second, axis, excess);
            }
        }
    }

    out
}

/// The angle between two bodies about a hinge, measured from a reference perpendicular to
/// the axis in each -- so it is the swing, with the twist the axis constraint removes left
/// out of it.
fn hinge_angle(
    a: Quaternion,
    b: Quaternion,
    axis: (f64, f64, f64),
    axis_a: (f64, f64, f64),
    axis_b: (f64, f64, f64),
) -> f64 {
    let reference = perpendicular(axis);
    let in_a = rotate(a, reference);
    let in_b = rotate(b, rotate_into(reference, axis_b, axis_a));
    dot(cross(in_a, in_b), axis).atan2(dot(in_b, in_a))
}

/// `w = inv_m + (r x n) . I^-1 (r x n)`: how much a unit impulse along `n` applied at `r`
/// actually moves this body. The denominator of every positional correction.
#[inline]
fn generalised_inverse_mass(body: &Pose, r: (f64, f64, f64), n: (f64, f64, f64)) -> f64 {
    let rn = cross(r, n);
    body.inv_mass + dot(rn, body.world_inv_inertia.apply(rn))
}

/// Fold one impulse at `r` into a body's correction.
fn accumulate(
    into: &mut Correction,
    body: &Pose,
    r: (f64, f64, f64),
    impulse: (f64, f64, f64),
    free: bool,
) {
    if !body.movable() {
        return;
    }
    let move_by = scale(impulse, body.inv_mass);
    if free {
        into.free_translation = add(into.free_translation, move_by);
    } else {
        into.translation = add(into.translation, move_by);
    }

    // The orientation update is `q + (1/2) dw q`, and `dw q` factors, so the *delta* to
    // left-multiply is `1 + (1/2) dw` -- independent of the orientation it will be
    // applied to, which is exactly what lets this be computed now and applied later.
    let dw = body.world_inv_inertia.apply(cross(r, impulse));
    let delta = renormalized(Quaternion {
        w: 1.0,
        x: 0.5 * dw.0,
        y: 0.5 * dw.1,
        z: 0.5 * dw.2,
    });
    // Composed onto whatever this body has already been asked to do by this joint.
    if free {
        into.free_rotation = renormalized(delta.multiply(&into.free_rotation));
    } else {
        into.rotation = renormalized(delta.multiply(&into.rotation));
    }
}

/// Turn two bodies apart about a world axis, split by their inertias, into their
/// corrections.
fn share_turn(
    out: &mut [Correction; 2],
    a: &Pose,
    b: &Pose,
    axis: (f64, f64, f64),
    angle: f64,
) {
    if angle.abs() < 1e-9 {
        return;
    }
    let ia = dot(axis, a.world_inv_inertia.apply(axis));
    let ib = dot(axis, b.world_inv_inertia.apply(axis));
    let total = ia + ib;
    if total <= 1e-12 {
        return;
    }
    if a.inv_inertia != (0.0, 0.0, 0.0) {
        let turn = Quaternion::from_axis_angle(axis, angle * ia / total);
        out[0].rotation = renormalized(turn.multiply(&out[0].rotation));
    }
    if b.inv_inertia != (0.0, 0.0, 0.0) {
        let turn = Quaternion::from_axis_angle(axis, -angle * ib / total);
        out[1].rotation = renormalized(turn.multiply(&out[1].rotation));
    }
}

/// How far a body has turned since the step began, as an axis-angle vector.
///
/// The `w < 0` flip keeps the short way round, for the same reason the velocity writeback
/// does: a quaternion and its negation are the same orientation.
#[inline]
fn turned_since(now: Quaternion, before: Quaternion) -> (f64, f64, f64) {
    let delta = now.multiply(&before.conjugate());
    let sign = if delta.w < 0.0 { -1.0 } else { 1.0 };
    scale((delta.x, delta.y, delta.z), 2.0 * sign)
}

/// Rolling resistance at a contact, as the angular half of Coulomb.
///
/// The resisting torque a real contact patch applies is the coefficient times the normal
/// force times the radius, so over a step the resisting angular impulse is bounded by the
/// coefficient times the radius times the normal impulse -- carried across the passes
/// exactly like the tangential one, and for the same reason, including that what is
/// bounded is the resultant rather than the distance it walked. A pass asks for exactly
/// the roll that has happened and no more, so resistance never turns into a push; what it
/// may do, once the cone has clipped an earlier pass, is give back some of what that pass
/// over-applied.
///
/// Only rotation about axes *in* the contact plane is resisted. Rotation about the normal
/// is a body spinning on the spot, which is a different effect with a different arm.
/// `b` is `None` when the other side of the contact is the ground, which does not turn
/// and takes none of the correction.
#[allow(clippy::too_many_arguments)]
fn resist_rolling(
    out: &mut [Correction; 2],
    a: &Gathered,
    b: Option<&Gathered>,
    normal: (f64, f64, f64),
    arm: f64,
    normal_impulse: f64,
    spent: (f64, f64, f64),
) -> (f64, f64, f64) {
    let mut relative = turned_since(a.now.orientation, a.prev_orientation);
    if let Some(b) = b {
        relative = sub(relative, turned_since(b.now.orientation, b.prev_orientation));
    }
    let rolled = sub(relative, scale(normal, dot(relative, normal)));
    let Some(axis) = normalized(rolled) else {
        return spent;
    };
    let ia = dot(axis, a.now.world_inv_inertia.apply(axis));
    let ib = match b {
        Some(b) => dot(axis, b.now.world_inv_inertia.apply(axis)),
        None => 0.0,
    };
    let total = ia + ib;
    if total <= 1e-12 {
        return spent;
    }

    // A cone on the resultant, not a running total of what has been spent: a roll that
    // reverses between passes has to give its budget back, or the reversals eat the
    // coefficient. Exactly the argument [`contacts::Spent`] makes for the tangential
    // half, one dimension over -- and this is the dimension where reversals are most
    // likely, because the friction impulse that turns a body is applied at an arm and the
    // normal impulse that untilts it is applied at another.
    let wanted = scale(axis, -length(rolled) / total);
    let (delta, total_impulse) = contacts::cone(spent, wanted, arm * normal_impulse);
    let size = length(delta);
    let Some(along) = normalized(delta) else {
        return spent;
    };

    // Split by inertia, the same way a hinge's range is, and opposing the roll. The
    // angular impulse turns each body by its own inverse inertia along the axis it acts
    // on, which is the axis of the correction rather than of the roll once anything has
    // been carried over from an earlier pass.
    let ia = dot(along, a.now.world_inv_inertia.apply(along));
    let ib = match b {
        Some(b) => dot(along, b.now.world_inv_inertia.apply(along)),
        None => 0.0,
    };
    if ia > 0.0 {
        let turn = Quaternion::from_axis_angle(along, size * ia);
        out[0].rotation = renormalized(turn.multiply(&out[0].rotation));
    }
    if ib > 0.0 {
        let turn = Quaternion::from_axis_angle(along, -size * ib);
        out[1].rotation = renormalized(turn.multiply(&out[1].rotation));
    }
    total_impulse
}

/// Any unit vector at right angles to `axis`. Which one does not matter -- it is only ever
/// a shared reference for measuring an angle, and both bodies measure from the same one.
fn perpendicular(axis: (f64, f64, f64)) -> (f64, f64, f64) {
    let candidate = if axis.0.abs() < 0.9 {
        (1.0, 0.0, 0.0)
    } else {
        (0.0, 1.0, 0.0)
    };
    normalized(cross(axis, candidate)).unwrap_or((0.0, 1.0, 0.0))
}

/// Carry a vector given about `from` over to the frame where the hinge axis is `to`, so
/// both bodies measure their angle from the same reference.
fn rotate_into(
    v: (f64, f64, f64),
    from: (f64, f64, f64),
    to: (f64, f64, f64),
) -> (f64, f64, f64) {
    let (Some(f), Some(t)) = (normalized(from), normalized(to)) else {
        return v;
    };
    let Some(axis) = normalized(cross(f, t)) else {
        return v;
    };
    let angle = dot(f, t).clamp(-1.0, 1.0).acos();
    rotate(Quaternion::from_axis_angle(axis, angle), v)
}

#[cfg(test)]
mod tests;
