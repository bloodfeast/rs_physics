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
//! the calling thread below it. The floor is not tuned to a machine -- it is the size at
//! which a rayon split has anything to amortise over, and being wrong about it by a factor
//! of two costs a few percent either way.
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
//!   thirty-two.
//! * **Friction does not resist rolling, so a heap of capsules rolls apart.** The contact
//!   point of a rolling body is instantaneously still, so there is nothing for Coulomb to
//!   act on. Measured, a pile of forty settled onto the ground perfectly happily and then
//!   spread to twenty metres over thirty seconds. Rolling resistance -- the same law on
//!   the same budget, one dimension over -- holds it at about a metre and a quarter.
//!
//! # Allocation
//!
//! [`Skeleton::step`] allocates nothing. The predicted state, the colour sets, the
//! contact buffers and the answer buffers each colour's parallel half fills all live in
//! the struct and are reused. Joint colouring happens when the joint set changes rather
//! than per step; contact colouring has to happen every step, because the contacts do.

use rayon::prelude::*;

use crate::models::Quaternion;

mod broadphase;
mod contacts;

use broadphase::Grid;
use contacts::{
    capsule_contact, ground_contacts, solve_contact, solve_ground, Contact, GroundContact,
};

/// Below this many items, a sweep or a colour runs on the calling thread.
///
/// See the module header for the measurement. A thread pool has a fixed cost per split --
/// a task, a queue, a join -- and a few dozen elements of arithmetic does not repay it.
const PARALLEL_FLOOR: usize = 256;

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

/// `I^-1 v` in world space for a body whose inverse inertia is diagonal in its own frame:
/// rotate into the body, scale, rotate back.
#[inline]
fn apply_inv_inertia(
    orientation: Quaternion,
    inv_inertia: (f64, f64, f64),
    v: (f64, f64, f64),
) -> (f64, f64, f64) {
    let local = orientation.inverse().rotate_point(v);
    orientation.rotate_point((
        local.0 * inv_inertia.0,
        local.1 * inv_inertia.1,
        local.2 * inv_inertia.2,
    ))
}

/// One body's share of one joint's correction: where to move it and how to turn it.
///
/// Produced in parallel and applied afterwards. Within a colour no two joints name the
/// same body, so the order corrections are applied in cannot change the answer.
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

    /// Which bodies are directly jointed, sorted, so contact generation can skip them.
    /// Rebuilt with the colouring. See [`Skeleton::build_contacts`].
    jointed: Vec<(usize, usize)>,
    /// Candidate pairs from the broad phase, and the contacts that survived the narrow
    /// one. Both are cleared and refilled per step rather than reallocated.
    pairs: Vec<(usize, usize)>,
    contacts: Vec<Contact>,
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
    friction: f64,
    rolling_resistance: f64,
    /// Running totals of the normal, tangential and rolling impulse each contact has
    /// applied so
    /// far this step, and the same for each ground contact.
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
    contact_impulse: Vec<(f64, f64, f64)>,
    ground_impulse: Vec<(f64, f64, f64)>,

    /// Where a colour's parallel half puts its answers before the serial half applies
    /// them. Held on the struct rather than collected fresh, because a colour is solved
    /// once per pass per step: a `collect` here is an allocation every few microseconds
    /// for the life of the program.
    joint_answers: Vec<(usize, [Correction; 2])>,
    contact_answers: Vec<(usize, [Correction; 2], (f64, f64, f64))>,
    ground_answers: Vec<(usize, Correction, (f64, f64, f64))>,
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
            jointed: Vec::new(),
            pairs: Vec::new(),
            contacts: Vec::new(),
            contact_colours: Vec::new(),
            contact_overflow: Vec::new(),
            colour_bits: Vec::new(),
            ground: None,
            ground_contacts: Vec::new(),
            ground_colours: [Vec::new(), Vec::new()],
            friction: DEFAULT_FRICTION,
            rolling_resistance: DEFAULT_ROLLING_RESISTANCE,
            contact_impulse: Vec::new(),
            ground_impulse: Vec::new(),
            joint_answers: Vec::new(),
            contact_answers: Vec::new(),
            ground_answers: Vec::new(),
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
    pub fn add_body(&mut self, body: Body) -> usize {
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
        self.orientation[i] = body.orientation;
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

        // The pairs contact generation must not produce. Two bones either side of an
        // elbow share an anchor point, so their capsules overlap by construction and a
        // contact between them would be the joint and the contact fighting each other
        // forever. Sorted so the test during generation is a binary search.
        self.jointed.clear();
        for joint in self.joints.iter() {
            let (a, b) = joint.bodies();
            self.jointed.push((a.min(b), a.max(b)));
        }
        self.jointed.sort_unstable();
        self.jointed.dedup();

        self.coloured = true;
    }

    /// Candidate pairs for the narrow phase, from the broad phase.
    fn find_pairs(&mut self) {
        self.pairs.clear();
        let mut grid = std::mem::take(&mut self.grid);
        grid.rebuild(&self.position, &self.radius, &self.half_length);
        grid.pairs(
            &self.position,
            &self.radius,
            &self.half_length,
            &self.inv_mass,
            &self.jointed,
            &mut self.pairs,
        );
        self.grid = grid;
    }

    /// The narrow phase: which candidates are actually touching, and where.
    fn build_contacts(&mut self) {
        self.contacts.clear();
        let position = &self.position;
        let orientation = &self.orientation;
        let radius = &self.radius;
        let half_length = &self.half_length;
        let test = |&(a, b): &(usize, usize)| {
            capsule_contact(a, b, position, orientation, radius, half_length)
                .into_iter()
                .flatten()
        };
        if self.pairs.len() >= PARALLEL_FLOOR {
            self.contacts
                .par_extend(self.pairs.par_iter().flat_map_iter(test));
        } else {
            self.contacts.extend(self.pairs.iter().flat_map(test));
        }
        self.contact_impulse.clear();
        self.contact_impulse
            .resize(self.contacts.len(), (0.0, 0.0, 0.0));

        self.ground_contacts.clear();
        self.ground_colours[0].clear();
        self.ground_colours[1].clear();
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
        }
        self.ground_impulse.clear();
        self.ground_impulse
            .resize(self.ground_contacts.len(), (0.0, 0.0, 0.0));
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

        // Three sweeps, each over two arrays, each independently parallel. This is the
        // shape the structure of arrays is for: no gather, and rayon can chunk it.
        let g = gravity;
        let wide = self.position.len() >= PARALLEL_FLOOR;

        let fall = |(v, &inv_m): (&mut (f64, f64, f64), &f64)| {
            if inv_m > 0.0 {
                *v = add(*v, scale(g, dt));
            }
        };
        let travel = |(p, v): (&mut (f64, f64, f64), &(f64, f64, f64))| {
            *p = add(*p, scale(*v, dt));
        };
        let spin = |(q, w): (&mut Quaternion, &(f64, f64, f64))| {
            *q = integrate_spin(*q, *w, dt);
        };

        if wide {
            self.velocity
                .par_iter_mut()
                .zip(self.inv_mass.par_iter())
                .for_each(fall);
            self.position
                .par_iter_mut()
                .zip(self.velocity.par_iter())
                .for_each(travel);
            self.orientation
                .par_iter_mut()
                .zip(self.angular_velocity.par_iter())
                .for_each(spin);
        } else {
            self.velocity.iter_mut().zip(self.inv_mass.iter()).for_each(fall);
            self.position.iter_mut().zip(self.velocity.iter()).for_each(travel);
            self.orientation
                .iter_mut()
                .zip(self.angular_velocity.iter())
                .for_each(spin);
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

        let inv_dt = 1.0 / dt;
        let moved = |((v, p), prev): ((&mut (f64, f64, f64), &(f64, f64, f64)), &(f64, f64, f64))| {
            *v = scale(sub(*p, *prev), inv_dt);
        };
        let turned = |((w, q), prev): ((&mut (f64, f64, f64), &Quaternion), &Quaternion)| {
            // The rotation that happened, as an axis-angle, divided by the step. The
            // `w < 0` flip keeps the short way round: a quaternion and its negation are
            // the same orientation, and without the check a body can read as spinning
            // almost a full turn when it barely moved.
            let delta = q.multiply(&prev.inverse());
            let sign = if delta.w < 0.0 { -1.0 } else { 1.0 };
            *w = scale((delta.x, delta.y, delta.z), 2.0 * inv_dt * sign);
        };

        if wide {
            self.velocity
                .par_iter_mut()
                .zip(self.position.par_iter())
                .zip(self.prev_position.par_iter())
                .for_each(moved);
            self.angular_velocity
                .par_iter_mut()
                .zip(self.orientation.par_iter())
                .zip(self.prev_orientation.par_iter())
                .for_each(turned);
        } else {
            self.velocity
                .iter_mut()
                .zip(self.position.iter())
                .zip(self.prev_position.iter())
                .for_each(moved);
            self.angular_velocity
                .iter_mut()
                .zip(self.orientation.iter())
                .zip(self.prev_orientation.iter())
                .for_each(turned);
        }
    }

    /// One colour: every joint in it computes its two corrections **in parallel**, then
    /// they are applied. No two joints in a colour name the same body, so applying them
    /// in any order gives the same answer -- which is what makes the parallel half safe
    /// without any unsafe.
    fn solve_colour(&mut self, colour: usize) {
        let joints = &self.joints;
        let position = &self.position;
        let orientation = &self.orientation;
        let inv_mass = &self.inv_mass;
        let inv_inertia = &self.inv_inertia;
        let set = &self.colours[colour];

        // Borrowed apart so the parallel closure only sees the read-only arrays.
        let solve = |&k: &usize| {
            (
                k,
                solve_joint(joints[k], position, orientation, inv_mass, inv_inertia),
            )
        };
        // Taken out so the parallel half can borrow the arrays and the serial half can
        // borrow `self`; put back below, so nothing here allocates after the first step.
        let mut answers = std::mem::take(&mut self.joint_answers);
        if set.len() >= PARALLEL_FLOOR {
            set.par_iter().map(solve).collect_into_vec(&mut answers);
        } else {
            answers.clear();
            answers.extend(set.iter().map(solve));
        }
        // Applied straight out of the colour's own answers. An earlier version scattered
        // them into an array the size of the whole joint set and read them back in the
        // next loop, which for a heap of six hundred bodies was ten megabytes of
        // pointless traffic every solver pass.
        for &(_, corrections) in answers.iter() {
            self.apply(corrections);
        }
        self.joint_answers = answers;
    }

    /// One colour of contacts, the same way [`Skeleton::solve_colour`] does one colour of
    /// joints: compute in parallel, then apply.
    fn solve_contact_colour(&mut self, colour: usize) {
        let contacts = &self.contacts;
        let position = &self.position;
        let orientation = &self.orientation;
        let prev_position = &self.prev_position;
        let prev_orientation = &self.prev_orientation;
        let inv_mass = &self.inv_mass;
        let inv_inertia = &self.inv_inertia;
        let friction = self.friction;
        let rolling = self.rolling_resistance;
        let radius = &self.radius;
        let spent = &self.contact_impulse;
        let set = &self.contact_colours[colour];

        let solve = |&k: &usize| {
            let (corrections, totals) = solve_contact(
                contacts[k],
                position,
                orientation,
                prev_position,
                prev_orientation,
                inv_mass,
                inv_inertia,
                radius,
                friction,
                rolling,
                spent[k],
            );
            (k, corrections, totals)
        };
        let mut answers = std::mem::take(&mut self.contact_answers);
        if set.len() >= PARALLEL_FLOOR {
            set.par_iter().map(solve).collect_into_vec(&mut answers);
        } else {
            answers.clear();
            answers.extend(set.iter().map(solve));
        }
        for &(k, corrections, totals) in answers.iter() {
            self.contact_impulse[k] = totals;
            self.apply(corrections);
        }
        self.contact_answers = answers;
    }

    /// The contacts colouring could not place, solved one at a time. Each reads the
    /// positions the one before it wrote, which is what makes it safe without a colour --
    /// and slow, which is why it is a tail and not the main path.
    fn solve_contact_overflow(&mut self) {
        if self.contact_overflow.is_empty() {
            return;
        }
        for index in 0..self.contact_overflow.len() {
            let k = self.contact_overflow[index];
            let (corrections, totals) = solve_contact(
                self.contacts[k],
                &self.position,
                &self.orientation,
                &self.prev_position,
                &self.prev_orientation,
                &self.inv_mass,
                &self.inv_inertia,
                &self.radius,
                self.friction,
                self.rolling_resistance,
                self.contact_impulse[k],
            );
            self.contact_impulse[k] = totals;
            self.apply(corrections);
        }
    }

    /// One set of ground contacts, the same way a colour of pair contacts is done.
    fn solve_ground_colour(&mut self, colour: usize) {
        let Some((normal, distance)) = self.ground else {
            return;
        };
        let contacts = &self.ground_contacts;
        let position = &self.position;
        let orientation = &self.orientation;
        let prev_position = &self.prev_position;
        let prev_orientation = &self.prev_orientation;
        let inv_mass = &self.inv_mass;
        let inv_inertia = &self.inv_inertia;
        let friction = self.friction;
        let rolling = self.rolling_resistance;
        let radius = &self.radius;
        let spent = &self.ground_impulse;
        let set = &self.ground_colours[colour];

        let solve = |&k: &usize| {
            let (correction, totals) = solve_ground(
                contacts[k],
                position,
                orientation,
                prev_position,
                prev_orientation,
                inv_mass,
                inv_inertia,
                radius,
                friction,
                rolling,
                normal,
                distance,
                spent[k],
            );
            (k, correction, totals)
        };
        let mut answers = std::mem::take(&mut self.ground_answers);
        if set.len() >= PARALLEL_FLOOR {
            set.par_iter().map(solve).collect_into_vec(&mut answers);
        } else {
            answers.clear();
            answers.extend(set.iter().map(solve));
        }
        for &(k, correction, totals) in answers.iter() {
            self.ground_impulse[k] = totals;
            self.apply([correction, Correction::none()]);
        }
        self.ground_answers = answers;
    }

    /// Move and turn the bodies one constraint asked for.
    #[inline]
    fn apply(&mut self, corrections: [Correction; 2]) {
        for correction in corrections {
            if correction.body == usize::MAX {
                continue;
            }
            let i = correction.body;
            let moved = add(correction.translation, correction.free_translation);
            self.position[i] = add(self.position[i], moved);
            let turn = correction.free_rotation.multiply(&correction.rotation);
            if !turn.is_near_identity(1e-12) {
                self.orientation[i] = turn.multiply(&self.orientation[i]).normalized();
            }

            // The free part moves where the body came from as well, so the velocity read
            // back at the end of the step does not see it at all. See
            // [`Correction::free_translation`].
            if correction.free_translation != (0.0, 0.0, 0.0) {
                self.prev_position[i] =
                    add(self.prev_position[i], correction.free_translation);
            }
            if !correction.free_rotation.is_near_identity(1e-12) {
                self.prev_orientation[i] = correction
                    .free_rotation
                    .multiply(&self.prev_orientation[i])
                    .normalized();
            }
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
    Quaternion {
        w: q.w + 0.5 * dt * spin.w,
        x: q.x + 0.5 * dt * spin.x,
        y: q.y + 0.5 * dt * spin.y,
        z: q.z + 0.5 * dt * spin.z,
    }
    .normalized()
}

/// Everything one joint wants done, as two corrections. Reads only; the caller applies.
fn solve_joint(
    joint: Joint,
    position: &[(f64, f64, f64)],
    orientation: &[Quaternion],
    inv_mass: &[f64],
    inv_inertia: &[(f64, f64, f64)],
) -> [Correction; 2] {
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
    let ra = orientation[a].rotate_point(anchor_a);
    let rb = orientation[b].rotate_point(anchor_b);
    let error = sub(add(position[b], rb), add(position[a], ra));
    if let Some(n) = normalized(error) {
        let c = length(error);
        let wa = generalised_inverse_mass(orientation[a], inv_mass[a], inv_inertia[a], ra, n);
        let wb = generalised_inverse_mass(orientation[b], inv_mass[b], inv_inertia[b], rb, n);
        let total = wa + wb;
        if total > 1e-12 {
            let impulse = scale(n, c / total);
            accumulate(
                &mut out[0],
                orientation[a],
                inv_mass[a],
                inv_inertia[a],
                ra,
                impulse,
                false,
            );
            accumulate(
                &mut out[1],
                orientation[b],
                inv_mass[b],
                inv_inertia[b],
                rb,
                scale(impulse, -1.0),
                false,
            );
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
        let world_a = orientation[a].rotate_point(axis_a);
        let world_b = orientation[b].rotate_point(axis_b);
        if let Some(n) = normalized(cross(world_b, world_a)) {
            let angle = length(cross(world_b, world_a)).clamp(-1.0, 1.0).asin();
            share_turn(
                &mut out,
                orientation,
                inv_inertia,
                a,
                b,
                n,
                angle,
            );
        }

        if let Some(axis) = normalized(world_a) {
            let angle = hinge_angle(orientation, a, b, axis, axis_a, axis_b);
            let excess = if angle < min {
                angle - min
            } else if angle > max {
                angle - max
            } else {
                0.0
            };
            if excess != 0.0 {
                share_turn(
                    &mut out,
                    orientation,
                    inv_inertia,
                    a,
                    b,
                    axis,
                    excess,
                );
            }
        }
    }

    out
}

/// The angle between two bodies about a hinge, measured from a reference perpendicular to
/// the axis in each -- so it is the swing, with the twist the axis constraint removes left
/// out of it.
fn hinge_angle(
    orientation: &[Quaternion],
    a: usize,
    b: usize,
    axis: (f64, f64, f64),
    axis_a: (f64, f64, f64),
    axis_b: (f64, f64, f64),
) -> f64 {
    let reference = perpendicular(axis);
    let in_a = orientation[a].rotate_point(reference);
    let in_b = orientation[b].rotate_point(rotate_into(reference, axis_b, axis_a));
    dot(cross(in_a, in_b), axis).atan2(dot(in_b, in_a))
}

/// `w = inv_m + (r x n) . I^-1 (r x n)`: how much a unit impulse along `n` applied at `r`
/// actually moves this body. The denominator of every positional correction.
#[inline]
fn generalised_inverse_mass(
    orientation: Quaternion,
    inv_mass: f64,
    inv_inertia: (f64, f64, f64),
    r: (f64, f64, f64),
    n: (f64, f64, f64),
) -> f64 {
    let rn = cross(r, n);
    inv_mass + dot(rn, apply_inv_inertia(orientation, inv_inertia, rn))
}

/// Fold one impulse at `r` into a body's correction.
fn accumulate(
    into: &mut Correction,
    orientation: Quaternion,
    inv_mass: f64,
    inv_inertia: (f64, f64, f64),
    r: (f64, f64, f64),
    impulse: (f64, f64, f64),
    free: bool,
) {
    if inv_mass <= 0.0 && inv_inertia == (0.0, 0.0, 0.0) {
        return;
    }
    let move_by = scale(impulse, inv_mass);
    if free {
        into.free_translation = add(into.free_translation, move_by);
    } else {
        into.translation = add(into.translation, move_by);
    }

    // The orientation update is `q + (1/2) dw q`, and `dw q` factors, so the *delta* to
    // left-multiply is `1 + (1/2) dw` -- independent of the orientation it will be
    // applied to, which is exactly what lets this be computed now and applied later.
    let dw = apply_inv_inertia(orientation, inv_inertia, cross(r, impulse));
    let delta = Quaternion {
        w: 1.0,
        x: 0.5 * dw.0,
        y: 0.5 * dw.1,
        z: 0.5 * dw.2,
    }
    .normalized();
    // Composed onto whatever this body has already been asked to do by this joint.
    if free {
        into.free_rotation = delta.multiply(&into.free_rotation).normalized();
    } else {
        into.rotation = delta.multiply(&into.rotation).normalized();
    }
}

/// Turn two bodies apart about a world axis, split by their inertias, into their
/// corrections.
fn share_turn(
    out: &mut [Correction; 2],
    orientation: &[Quaternion],
    inv_inertia: &[(f64, f64, f64)],
    a: usize,
    b: usize,
    axis: (f64, f64, f64),
    angle: f64,
) {
    if angle.abs() < 1e-9 {
        return;
    }
    let ia = dot(axis, apply_inv_inertia(orientation[a], inv_inertia[a], axis));
    let ib = dot(axis, apply_inv_inertia(orientation[b], inv_inertia[b], axis));
    let total = ia + ib;
    if total <= 1e-12 {
        return;
    }
    if inv_inertia[a] != (0.0, 0.0, 0.0) {
        let turn = Quaternion::from_axis_angle(axis, angle * ia / total);
        out[0].rotation = turn.multiply(&out[0].rotation).normalized();
    }
    if inv_inertia[b] != (0.0, 0.0, 0.0) {
        let turn = Quaternion::from_axis_angle(axis, -angle * ib / total);
        out[1].rotation = turn.multiply(&out[1].rotation).normalized();
    }
}

/// How far a body has turned since the step began, as an axis-angle vector.
///
/// The `w < 0` flip keeps the short way round, for the same reason the velocity writeback
/// does: a quaternion and its negation are the same orientation.
#[inline]
fn turned_since(now: Quaternion, before: Quaternion) -> (f64, f64, f64) {
    let delta = now.multiply(&before.inverse());
    let sign = if delta.w < 0.0 { -1.0 } else { 1.0 };
    scale((delta.x, delta.y, delta.z), 2.0 * sign)
}

/// Rolling resistance at a contact, as the angular half of Coulomb.
///
/// The resisting torque a real contact patch applies is the coefficient times the normal
/// force times the radius, so over a step the resisting angular impulse is bounded by the
/// coefficient times the radius times the normal impulse -- carried across the passes
/// exactly like the tangential one, and for the same reason. Clamped so it can at most
/// stop the rolling that happened, never reverse it into rolling the other way.
///
/// Only rotation about axes *in* the contact plane is resisted. Rotation about the normal
/// is a body spinning on the spot, which is a different effect with a different arm.
/// `b` is `None` when the other side of the contact is the ground, which does not turn
/// and takes none of the correction.
#[allow(clippy::too_many_arguments)]
fn resist_rolling(
    out: &mut [Correction; 2],
    orientation: &[Quaternion],
    prev_orientation: &[Quaternion],
    inv_inertia: &[(f64, f64, f64)],
    a: usize,
    b: Option<usize>,
    normal: (f64, f64, f64),
    arm: f64,
    normal_impulse: f64,
    spent: f64,
) -> f64 {
    let allowed = (arm * normal_impulse - spent).max(0.0);
    if allowed <= 0.0 {
        return 0.0;
    }
    let mut relative = turned_since(orientation[a], prev_orientation[a]);
    if let Some(b) = b {
        relative = sub(relative, turned_since(orientation[b], prev_orientation[b]));
    }
    let rolled = sub(relative, scale(normal, dot(relative, normal)));
    let Some(axis) = normalized(rolled) else {
        return 0.0;
    };
    let ia = dot(axis, apply_inv_inertia(orientation[a], inv_inertia[a], axis));
    let ib = match b {
        Some(b) => dot(axis, apply_inv_inertia(orientation[b], inv_inertia[b], axis)),
        None => 0.0,
    };
    let total = ia + ib;
    if total <= 1e-12 {
        return 0.0;
    }
    let spend = (length(rolled) / total).min(allowed);
    if spend <= 0.0 {
        return 0.0;
    }
    // Split by inertia, the same way a hinge's range is, and opposing the roll.
    if ia > 0.0 {
        let turn = Quaternion::from_axis_angle(axis, -spend * ia);
        out[0].rotation = turn.multiply(&out[0].rotation).normalized();
    }
    if ib > 0.0 {
        let turn = Quaternion::from_axis_angle(axis, spend * ib);
        out[1].rotation = turn.multiply(&out[1].rotation).normalized();
    }
    spend
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
    Quaternion::from_axis_angle(axis, angle).rotate_point(v)
}

#[cfg(test)]
mod tests;
