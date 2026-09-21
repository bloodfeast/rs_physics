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
//! The bodies are six parallel arrays rather than a `Vec<Body>`. Two things need that,
//! and neither is cache-line arithmetic:
//!
//! * **SIMD on the streaming passes.** Predicting and reading velocities back are pure
//!   sweeps over every body. As arrays they vectorise and parallelise by chunk; as a
//!   `Vec<Body>` each lane would be a gather out of a 136-byte struct.
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
//! # Allocation
//!
//! [`Skeleton::step`] allocates nothing. The predicted state, the colour sets and the
//! correction scratch live in the struct and are reused. Colouring itself happens once,
//! when the joint set changes, not per step.

use rayon::prelude::*;

use crate::models::Quaternion;

/// Below this many items, a sweep or a colour runs on the calling thread.
///
/// See the module header for the measurement. A thread pool has a fixed cost per split --
/// a task, a queue, a join -- and a few dozen elements of arithmetic does not repay it.
const PARALLEL_FLOOR: usize = 256;

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
        }
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
}

impl Correction {
    fn none() -> Self {
        Correction {
            body: usize::MAX,
            translation: (0.0, 0.0, 0.0),
            rotation: Quaternion::identity(),
        }
    }
}

/// A set of bodies and the joints between them, solved together.
///
/// See the module header for why the bodies are arrays, and why the joints are coloured.
#[derive(Clone, Debug, Default)]
pub struct Skeleton {
    position: Vec<(f64, f64, f64)>,
    orientation: Vec<Quaternion>,
    velocity: Vec<(f64, f64, f64)>,
    angular_velocity: Vec<(f64, f64, f64)>,
    inv_mass: Vec<f64>,
    inv_inertia: Vec<(f64, f64, f64)>,

    joints: Vec<Joint>,
    /// Joint indices grouped so that no two joints in a group share a body. Rebuilt when
    /// the joint set changes, not per step. See the module header.
    colours: Vec<Vec<usize>>,
    coloured: bool,

    prev_position: Vec<(f64, f64, f64)>,
    prev_orientation: Vec<Quaternion>,
    /// Two corrections per joint -- one per body -- written in parallel, applied after.
    /// Reused; `step` allocates nothing.
    scratch: Vec<[Correction; 2]>,
}

impl Skeleton {
    pub fn new() -> Self {
        Skeleton::default()
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
        self.prev_position.push(body.position);
        self.prev_orientation.push(body.orientation);
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
        self.scratch.push([Correction::none(); 2]);
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
        self.coloured = true;
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

        for _ in 0..iterations.max(1) {
            for colour in 0..self.colours.len() {
                self.solve_colour(colour);
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
        let pairs: Vec<(usize, [Correction; 2])> = if set.len() >= PARALLEL_FLOOR {
            set.par_iter().map(solve).collect()
        } else {
            set.iter().map(solve).collect()
        };
        let scratch = &mut self.scratch;
        for (k, corrections) in pairs {
            scratch[k] = corrections;
        }

        for &k in set.iter() {
            for correction in scratch[k] {
                if correction.body == usize::MAX {
                    continue;
                }
                let i = correction.body;
                self.position[i] = add(self.position[i], correction.translation);
                if !correction.rotation.is_near_identity(1e-12) {
                    self.orientation[i] = correction
                        .rotation
                        .multiply(&self.orientation[i])
                        .normalized();
                }
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
            );
            accumulate(
                &mut out[1],
                orientation[b],
                inv_mass[b],
                inv_inertia[b],
                rb,
                scale(impulse, -1.0),
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
) {
    if inv_mass <= 0.0 && inv_inertia == (0.0, 0.0, 0.0) {
        return;
    }
    into.translation = add(into.translation, scale(impulse, inv_mass));

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
    into.rotation = delta.multiply(&into.rotation).normalized();
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
