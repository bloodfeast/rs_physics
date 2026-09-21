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
//! the same limb -- and copying state in and out per constraint per iteration is both the
//! slow way and the wrong one.
//!
//! So this module keeps **one** array of bodies and gives joints indices into it. Every
//! constraint reads and writes the same memory, which is what lets a chain converge.
//!
//! # What it is solved with, and why not impulses
//!
//! Extended Position Based Dynamics. The same family as
//! [`crate::constraints::RopeChain3D`], which is already the solver this crate's users
//! reach for, so a reader who knows the rope knows this: predict, correct positions and
//! orientations directly, then read the velocities back out of what moved.
//!
//! XPBD is chosen over sequential impulses for the reason that matters at scale: it is
//! stable at low iteration counts. A ragdoll pile does not need to be accurate, it needs
//! to not explode when the budget says four iterations rather than forty.
//!
//! # Allocation
//!
//! [`Skeleton::step`] allocates nothing. The predicted state and the per-body
//! accumulators live in the struct and are reused; `step` is safe to call every frame on
//! hundreds of skeletons. That is the price of entry this crate asks of anything new, and
//! it is asserted by `step_allocates_nothing_after_the_first`.

use crate::models::Quaternion;

/// A rigid body in a [`Skeleton`], in world space.
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
    /// The axis is +Y because that is where a bone's length lives in every rig this
    /// crate's callers export -- see `ridgeline`'s ragdoll, which reads a bone's own
    /// direction off its child offset and falls back to local +Y for a leaf.
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

/// What holds two bodies together, by index into [`Skeleton::bodies`].
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

/// `I^-1 v` in world space, for a body whose inverse inertia is diagonal in its own
/// frame: rotate into the body, scale, rotate back.
#[inline]
fn apply_inv_inertia(body: &Body, v: (f64, f64, f64)) -> (f64, f64, f64) {
    let local = body.orientation.inverse().rotate_point(v);
    let scaled = (
        local.0 * body.inv_inertia.0,
        local.1 * body.inv_inertia.1,
        local.2 * body.inv_inertia.2,
    );
    body.orientation.rotate_point(scaled)
}

/// A set of bodies and the joints between them, solved together.
///
/// See the module header for why the bodies live here rather than inside the joints.
#[derive(Clone, Debug, Default)]
pub struct Skeleton {
    pub bodies: Vec<Body>,
    joints: Vec<Joint>,
    /// Position at the top of the step, so velocities can be read back out of what the
    /// solver moved. Reused; see the module header on allocation.
    prev_position: Vec<(f64, f64, f64)>,
    prev_orientation: Vec<Quaternion>,
}

impl Skeleton {
    pub fn new() -> Self {
        Skeleton::default()
    }

    /// Adds a body and returns its index, which is what joints are written against.
    pub fn add_body(&mut self, body: Body) -> usize {
        self.bodies.push(body);
        self.prev_position.push(body.position);
        self.prev_orientation.push(body.orientation);
        self.bodies.len() - 1
    }

    /// Adds a joint. Returns `false` and adds nothing if it names a body that does not
    /// exist -- an out-of-range index is a caller's bug and panicking in a solver that
    /// runs per frame is worse than refusing.
    pub fn add_joint(&mut self, joint: Joint) -> bool {
        let (a, b) = joint.bodies();
        let n = self.bodies.len();
        if a >= n || b >= n || a == b {
            return false;
        }
        self.joints.push(joint);
        true
    }

    pub fn joints(&self) -> &[Joint] {
        &self.joints
    }

    /// **One step.** Predict under `gravity`, run `iterations` passes over the joints,
    /// then read the velocities back out of what moved.
    ///
    /// Allocates nothing. `iterations` is the quality dial: four is enough for a corpse,
    /// and the cost is linear in it.
    pub fn step(&mut self, dt: f64, gravity: (f64, f64, f64), iterations: usize) {
        if dt <= 0.0 || self.bodies.is_empty() {
            return;
        }

        for (i, body) in self.bodies.iter_mut().enumerate() {
            self.prev_position[i] = body.position;
            self.prev_orientation[i] = body.orientation;
            if body.inv_mass > 0.0 {
                body.velocity = add(body.velocity, scale(gravity, dt));
            }
            body.position = add(body.position, scale(body.velocity, dt));

            // q' = q + (dt/2) * omega_quat * q, renormalised. The small-angle integrator
            // every position-based solver uses; exact enough over a frame and far cheaper
            // than an exponential map.
            let w = body.angular_velocity;
            let spin = Quaternion {
                w: 0.0,
                x: w.0,
                y: w.1,
                z: w.2,
            }
            .multiply(&body.orientation);
            let q = body.orientation;
            body.orientation = Quaternion {
                w: q.w + 0.5 * dt * spin.w,
                x: q.x + 0.5 * dt * spin.x,
                y: q.y + 0.5 * dt * spin.y,
                z: q.z + 0.5 * dt * spin.z,
            }
            .normalized();
        }

        for _ in 0..iterations.max(1) {
            for k in 0..self.joints.len() {
                let joint = self.joints[k];
                match joint {
                    Joint::Ball {
                        a,
                        b,
                        anchor_a,
                        anchor_b,
                    } => self.solve_point(a, b, anchor_a, anchor_b),
                    Joint::Hinge {
                        a,
                        b,
                        anchor_a,
                        anchor_b,
                        axis_a,
                        axis_b,
                        min,
                        max,
                    } => {
                        self.solve_point(a, b, anchor_a, anchor_b);
                        self.solve_hinge_axis(a, b, axis_a, axis_b);
                        self.solve_hinge_limit(a, b, axis_a, axis_b, min, max);
                    }
                }
            }
        }

        let inv_dt = 1.0 / dt;
        for (i, body) in self.bodies.iter_mut().enumerate() {
            body.velocity = scale(sub(body.position, self.prev_position[i]), inv_dt);

            // The rotation that happened, as an axis-angle, divided by the step. The
            // `w < 0` flip keeps the short way round: a quaternion and its negation are
            // the same orientation, and without the check a body can read as spinning
            // almost a full turn when it barely moved.
            let delta = body.orientation.multiply(&self.prev_orientation[i].inverse());
            let sign = if delta.w < 0.0 { -1.0 } else { 1.0 };
            body.angular_velocity = scale((delta.x, delta.y, delta.z), 2.0 * inv_dt * sign);
        }
    }

    /// The positional half of every joint: two anchors, given in their own bodies'
    /// frames, are the same point in the world.
    fn solve_point(
        &mut self,
        a: usize,
        b: usize,
        anchor_a: (f64, f64, f64),
        anchor_b: (f64, f64, f64),
    ) {
        let (ra, rb) = (
            self.bodies[a].orientation.rotate_point(anchor_a),
            self.bodies[b].orientation.rotate_point(anchor_b),
        );
        let world_a = add(self.bodies[a].position, ra);
        let world_b = add(self.bodies[b].position, rb);
        let error = sub(world_b, world_a);
        let Some(n) = normalized(error) else { return };
        let c = length(error);

        let wa = self.generalised_inverse_mass(a, ra, n);
        let wb = self.generalised_inverse_mass(b, rb, n);
        let total = wa + wb;
        if total <= 1e-12 {
            return;
        }
        let impulse = scale(n, c / total);
        self.apply_correction(a, ra, impulse, 1.0);
        self.apply_correction(b, rb, impulse, -1.0);
    }

    /// The hinge's axis: `b`'s axis is brought onto `a`'s. Purely angular -- it moves no
    /// position, which is what leaves `solve_point` in charge of where the joint is.
    fn solve_hinge_axis(
        &mut self,
        a: usize,
        b: usize,
        axis_a: (f64, f64, f64),
        axis_b: (f64, f64, f64),
    ) {
        let world_a = self.bodies[a].orientation.rotate_point(axis_a);
        let world_b = self.bodies[b].orientation.rotate_point(axis_b);
        let error = cross(world_b, world_a);
        let Some(n) = normalized(error) else { return };
        let angle = length(error).clamp(-1.0, 1.0).asin();
        self.apply_angular(a, b, n, angle);
    }

    /// And the range of motion about it. Nothing is done while the angle is inside
    /// `[min, max]`; outside, the excess is taken back.
    fn solve_hinge_limit(
        &mut self,
        a: usize,
        b: usize,
        axis_a: (f64, f64, f64),
        axis_b: (f64, f64, f64),
        min: f64,
        max: f64,
    ) {
        let Some(axis) = normalized(self.bodies[a].orientation.rotate_point(axis_a)) else {
            return;
        };
        // The angle between the two bodies about the hinge, measured from a reference
        // that is perpendicular to the axis in each -- so it is the swing, with the twist
        // the axis constraint has already removed left out of it.
        let reference = perpendicular(axis);
        let in_a = self.bodies[a].orientation.rotate_point(reference);
        let in_b = self
            .bodies[b]
            .orientation
            .rotate_point(rotate_into(reference, axis_b, axis_a));
        let x = dot(in_b, in_a);
        let y = dot(cross(in_a, in_b), axis);
        let angle = y.atan2(x);

        let excess = if angle < min {
            angle - min
        } else if angle > max {
            angle - max
        } else {
            return;
        };
        self.apply_angular(a, b, axis, excess);
    }

    /// `w = inv_m + (r x n) . I^-1 (r x n)`: how much a unit impulse along `n` applied at
    /// `r` actually moves this body. The denominator of every correction below.
    fn generalised_inverse_mass(
        &self,
        i: usize,
        r: (f64, f64, f64),
        n: (f64, f64, f64),
    ) -> f64 {
        let body = &self.bodies[i];
        let rn = cross(r, n);
        body.inv_mass + dot(rn, apply_inv_inertia(body, rn))
    }

    fn apply_correction(
        &mut self,
        i: usize,
        r: (f64, f64, f64),
        impulse: (f64, f64, f64),
        sign: f64,
    ) {
        let body = &mut self.bodies[i];
        if body.inv_mass <= 0.0 && body.inv_inertia == (0.0, 0.0, 0.0) {
            return;
        }
        let p = scale(impulse, sign);
        body.position = add(body.position, scale(p, body.inv_mass));

        let dw = apply_inv_inertia(body, cross(r, p));
        let spin = Quaternion {
            w: 0.0,
            x: dw.0,
            y: dw.1,
            z: dw.2,
        }
        .multiply(&body.orientation);
        let q = body.orientation;
        body.orientation = Quaternion {
            w: q.w + 0.5 * spin.w,
            x: q.x + 0.5 * spin.x,
            y: q.y + 0.5 * spin.y,
            z: q.z + 0.5 * spin.z,
        }
        .normalized();
    }

    /// Rotate `a` and `b` apart about `axis` by `angle`, split by their inertias.
    fn apply_angular(&mut self, a: usize, b: usize, axis: (f64, f64, f64), angle: f64) {
        if angle.abs() < 1e-9 {
            return;
        }
        let ia = dot(axis, apply_inv_inertia(&self.bodies[a], axis));
        let ib = dot(axis, apply_inv_inertia(&self.bodies[b], axis));
        let total = ia + ib;
        if total <= 1e-12 {
            return;
        }
        turn(&mut self.bodies[a], axis, angle * ia / total);
        turn(&mut self.bodies[b], axis, -angle * ib / total);
    }
}

/// Turn one body about a world axis, leaving its position alone.
fn turn(body: &mut Body, axis: (f64, f64, f64), angle: f64) {
    if body.inv_inertia == (0.0, 0.0, 0.0) {
        return;
    }
    body.orientation = Quaternion::from_axis_angle(axis, angle)
        .multiply(&body.orientation)
        .normalized();
}

/// Any unit vector at right angles to `axis`. Which one does not matter -- it is only
/// ever used as a shared reference for measuring an angle, and both bodies measure from
/// the same one.
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
    let axis = cross(f, t);
    let Some(axis) = normalized(axis) else { return v };
    let angle = dot(f, t).clamp(-1.0, 1.0).acos();
    Quaternion::from_axis_angle(axis, angle).rotate_point(v)
}

#[cfg(test)]
mod tests;
