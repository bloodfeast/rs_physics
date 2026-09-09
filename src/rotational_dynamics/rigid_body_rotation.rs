//! **Euler's equation for a rigid body**, integrated with the gyroscopic term present.
//!
//! # Why this type exists rather than a second method on [`AngularState3D`]
//!
//! The rotation of a rigid body is
//!
//! ```text
//!     I ω̇ + ω × (I ω) = τ        so        ω̇ = I⁻¹ (τ − ω × Iω)
//! ```
//!
//! and the `ω × Iω` term is not a refinement. It is why a thrown object wobbles instead
//! of spinning cleanly, why a spacecraft precesses, and why a body spun about its
//! *intermediate* principal axis periodically turns end over end with nothing acting on
//! it (the Dzhanibekov effect, asserted in
//! [`tests::intermediate_axis_spin_flips`]). Drop it and **a free body with zero torque
//! never changes its angular velocity at all** — no wobble, no precession, no tumble.
//!
//! [`AngularState3D::apply_torque`] drops it. That is not a bug that can be fixed in
//! place, and the reason is worth being precise about, because "just add the term"
//! is the obvious move and it is wrong:
//!
//! > **`apply_torque` is an accumulator, not a step.** Its contract is "add the effect
//! > of this torque", and a caller with three torques calls it three times. The
//! > gyroscopic term belongs to *advancing time once*, not to each torque, so adding it
//! > inside `apply_torque` would apply it three times in that frame and once in the next
//! > — a silent, timestep- and call-count-dependent error that is far harder to find
//! > than the missing term was.
//!
//! So the coupling cannot be optional and it cannot live on the accumulator. Here it is
//! structural: [`RigidBodyRotation`] **owns** `ω`, torque goes into an accumulator that
//! only [`RigidBodyRotation::step`] may consume, and `step` is the only thing in the
//! crate that moves `ω` forward in time. There is no path through this type that
//! integrates without the gyroscopic term, and none that applies it twice.
//!
//! # What is cached, and why the constructor is fallible
//!
//! [`InertiaTensor::apply_inverse_to_torque`] recomputes the full 3×3 inverse on every
//! call. The inverse is a property of the *body*, not of the call, so it is computed
//! once here — in [`RigidBodyRotation::new`], which is also where the tensor is checked
//! for being a real mass distribution. Positive definiteness is what makes that inverse
//! finite; the triangle inequality on the principal moments is what bounds the substep
//! rule below. Both are enforced at construction, so **every later path may assume
//! them** rather than re-check.
//!
//! # Frames
//!
//! `ω`, `τ` and `I` are in the **body frame** — the only frame in which `I` is constant,
//! which the derivation of Euler's equation depends on. World-frame accessors and
//! setters exist and say so in their names; there is no method whose frame is implied.
//! Getting that wrong does not fail loudly, it rotates the body at the right rate about
//! the wrong axis, so no method here takes an unnamed frame.

use crate::models::Quaternion;
use crate::utils::PhysicsError;
use crate::utils::vector3::{add, cross_product, dot_product, magnitude, scale, sub, Vec3};

use super::inertia::InertiaTensor;

/// **The largest angle one substep may turn through, in radians.**
///
/// Two things need `|ω|·h` small and this bounds both. The orientation is integrated as
/// part of the same Runge-Kutta state, and a Runge-Kutta step of a rotation is only
/// accurate while the angle covered is well below a radian; and the gyroscopic term
/// evolves on the timescale `1/|ω|`, so the same quantity measures how far `ω` itself
/// moves within a step.
///
/// A quarter radian is where the measured drift in the conserved quantities sits at the
/// level [`tests::free_rotation_conserves_angular_momentum_but_not_speed`] asserts. It is a step size,
/// not a tuning knob: halving it buys accuracy at linear cost and changes no behaviour.
pub const MAX_STEP_RADIANS: f64 = 0.25;

/// **The largest fractional change in `ω` one substep may make.**
///
/// [`MAX_STEP_RADIANS`] alone does not bound a step in which an applied torque is large
/// and `ω` is small — the angle turned is tiny while `ω` changes by a factor. This is
/// the second criterion, and [`RigidBodyRotation::substeps_for`] takes whichever of the
/// two asks for more substeps.
pub const MAX_RELATIVE_CHANGE_PER_SUBSTEP: f64 = 0.25;

/// **Ceiling on substeps in one [`RigidBodyRotation::step`] call.**
///
/// Work per call has to be bounded — a `dt` of ten seconds after a stall, or an `ω` of
/// ten thousand rad/s out of a bad impulse, would otherwise ask for an unbounded loop.
/// Unlike a clamp that hides the fact, `step` **returns the substep count it used**, so
/// a caller can compare it against this constant and learn that the frame was
/// under-resolved. Reaching it means the accuracy criteria above were not met.
pub const MAX_SUBSTEPS: u32 = 32;

/// A rigid body's rotational state and the mass properties that govern it.
///
/// Owns the angular velocity, so the only way to advance it is [`step`], which
/// integrates the full Euler equation. See the [module docs](self) for why that
/// ownership is the point rather than an implementation detail.
///
/// # Units
///
/// SI throughout: `ω` in rad/s, `τ` in N·m, `I` in kg·m², `dt` in seconds, energy in J,
/// angular momentum in kg·m²/s.
///
/// # Invariants
///
/// Established by [`new`] and preserved by every method, none of which can be
/// side-stepped because all fields are private:
///
/// - `inertia` is finite, positive definite, and satisfies the triangle inequality.
/// - `inverse` is `inertia.inverse()`, computed once.
/// - `velocity` and `torque` are finite.
/// - `orientation` is a unit quaternion.
///
/// # Example
///
/// ```
/// use rs_physics::rotational_dynamics::{RigidBodyRotation, inertia_3d};
///
/// // A brick: three distinct principal moments, so it can tumble.
/// let mut body = RigidBodyRotation::new(inertia_3d::solid_cuboid(1.0, 0.2, 0.1, 0.05))?;
/// body.set_angular_velocity_body((0.05, 8.0, 0.0))?;  // mostly about the middle axis
///
/// let start = body.angular_velocity_body().1;
/// for _ in 0..600 {
///     body.step(1.0 / 240.0)?;
/// }
/// // With no torque at all, the middle-axis spin has moved. That is the gyroscopic term.
/// assert!((body.angular_velocity_body().1 - start).abs() > 1.0);
/// # Ok::<(), rs_physics::utils::PhysicsError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidBodyRotation {
    /// Body-frame inertia tensor, kg·m². Finite, positive definite, physically realizable.
    inertia: InertiaTensor,
    /// `inertia.inverse()`, computed once in [`RigidBodyRotation::new`].
    inverse: InertiaTensor,
    /// Body-frame angular velocity, rad/s. Always finite.
    velocity: Vec3,
    /// Body-frame torque accumulated since the last [`RigidBodyRotation::step`], N·m.
    /// Always finite. Consumed and zeroed by `step`.
    torque: Vec3,
    /// Body-to-world rotation. Always unit length.
    orientation: Quaternion,
}

impl RigidBodyRotation {
    /// Build a body at rest, at the identity orientation, from its body-frame inertia
    /// tensor.
    ///
    /// # Arguments
    ///
    /// * `inertia` - The body-frame inertia tensor in kg·m². Must describe a real mass
    ///   distribution; see [`InertiaTensor::is_positive_definite`] and
    ///   [`InertiaTensor::satisfies_triangle_inequality`], which are the two conditions
    ///   checked here and are separately queryable if you need to know which failed.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidInertiaTensor`] if the tensor is not finite, is singular
    /// or indefinite (so `I⁻¹` does not exist — `inertia_3d::thin_rod_center` returns
    /// such a tensor deliberately, as an idealisation with a zero moment), or has
    /// principal moments no mass distribution could produce.
    pub fn new(inertia: InertiaTensor) -> Result<Self, PhysicsError> {
        if !inertia.is_positive_definite() || !inertia.satisfies_triangle_inequality() {
            return Err(PhysicsError::InvalidInertiaTensor);
        }
        let inverse = inertia.inverse();
        if !inverse.is_finite() {
            return Err(PhysicsError::InvalidInertiaTensor);
        }
        Ok(Self {
            inertia,
            inverse,
            velocity: (0.0, 0.0, 0.0),
            torque: (0.0, 0.0, 0.0),
            orientation: Quaternion::identity(),
        })
    }

    //--------------------------------------------------------------------------
    // Reading the state
    //--------------------------------------------------------------------------

    /// The body-frame inertia tensor, kg·m².
    #[inline]
    pub fn inertia(&self) -> &InertiaTensor {
        &self.inertia
    }

    /// The body-frame inverse inertia tensor. Computed once at construction; this is a
    /// borrow, not a recomputation.
    #[inline]
    pub fn inverse_inertia(&self) -> &InertiaTensor {
        &self.inverse
    }

    /// Body-frame angular velocity, rad/s.
    #[inline]
    pub fn angular_velocity_body(&self) -> Vec3 {
        self.velocity
    }

    /// World-frame angular velocity, rad/s.
    #[inline]
    pub fn angular_velocity_world(&self) -> Vec3 {
        self.orientation.rotate_point(self.velocity)
    }

    /// `|ω|` in rad/s. A rotation preserves length, so this is the same in either frame.
    ///
    /// **Not conserved under free rotation** — see [`angular_momentum_world`]. If you
    /// are clamping a spin rate, clamp what you draw with, and know that a value clamped
    /// on the way into [`step`] can come out above the bound.
    ///
    /// [`angular_momentum_world`]: RigidBodyRotation::angular_momentum_world
    /// [`step`]: RigidBodyRotation::step
    #[inline]
    pub fn speed(&self) -> f64 {
        magnitude(self.velocity)
    }

    /// Body-to-world orientation. Always a unit quaternion.
    #[inline]
    pub fn orientation(&self) -> Quaternion {
        self.orientation
    }

    /// Torque accumulated since the last [`step`], body frame, N·m.
    ///
    /// [`step`]: RigidBodyRotation::step
    #[inline]
    pub fn pending_torque_body(&self) -> Vec3 {
        self.torque
    }

    /// Angular momentum `L = Iω` in the body frame, kg·m²/s.
    ///
    /// This vector is *not* constant under free rotation — it is the world-frame one
    /// that is fixed — but its magnitude is.
    #[inline]
    pub fn angular_momentum_body(&self) -> Vec3 {
        self.inertia.multiply_vector(self.velocity)
    }

    /// Angular momentum in the world frame, kg·m²/s.
    ///
    /// **The conserved vector.** With no torque applied this is constant in both
    /// direction and magnitude, which is the strongest single check on this integrator:
    /// it holds only if the gyroscopic term and the orientation update are both right
    /// and are consistent with each other.
    #[inline]
    pub fn angular_momentum_world(&self) -> Vec3 {
        self.orientation.rotate_point(self.angular_momentum_body())
    }

    /// Rotational kinetic energy `½ ω·Iω`, joules. Conserved under free rotation.
    #[inline]
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * dot_product(self.velocity, self.angular_momentum_body())
    }

    //--------------------------------------------------------------------------
    // Setting the state
    //--------------------------------------------------------------------------

    /// Set the body-frame angular velocity, rad/s.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidVelocity`] if any component is not finite. Rejected here,
    /// at the edge, so that nothing downstream has to guard: a `NaN` that reaches `ω`
    /// poisons the orientation on the next step and never washes out.
    pub fn set_angular_velocity_body(&mut self, velocity: Vec3) -> Result<(), PhysicsError> {
        if !is_finite(velocity) {
            return Err(PhysicsError::InvalidVelocity);
        }
        self.velocity = velocity;
        Ok(())
    }

    /// Set the angular velocity from a world-frame vector, rad/s. Converted into the
    /// body frame using the current orientation.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidVelocity`] if any component is not finite.
    pub fn set_angular_velocity_world(&mut self, velocity: Vec3) -> Result<(), PhysicsError> {
        if !is_finite(velocity) {
            return Err(PhysicsError::InvalidVelocity);
        }
        self.velocity = self.orientation.inverse().rotate_point(velocity);
        Ok(())
    }

    /// Set the body-to-world orientation. Normalised on the way in, so the unit-length
    /// invariant holds regardless of what the caller had.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidAngle`] if the quaternion is not finite or is too close to
    /// zero to have a direction.
    pub fn set_orientation(&mut self, orientation: Quaternion) -> Result<(), PhysicsError> {
        if !orientation.w.is_finite()
            || !orientation.x.is_finite()
            || !orientation.y.is_finite()
            || !orientation.z.is_finite()
            || orientation.magnitude() < 1e-10
        {
            return Err(PhysicsError::InvalidAngle);
        }
        self.orientation = orientation.normalized();
        Ok(())
    }

    //--------------------------------------------------------------------------
    // Applying torques and impulses
    //--------------------------------------------------------------------------

    /// **Accumulate** a body-frame torque, N·m. It acts on the next [`step`] and is
    /// cleared by it.
    ///
    /// Accumulating rather than integrating is what keeps the gyroscopic term correct:
    /// several torques in a frame sum into one `τ`, and `step` applies
    /// `ω̇ = I⁻¹(τ − ω × Iω)` once.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidForce`] if any component is not finite. The accumulator is
    /// left unchanged in that case, so one bad torque does not destroy the good ones
    /// already added this frame.
    ///
    /// [`step`]: RigidBodyRotation::step
    pub fn apply_torque_body(&mut self, torque: Vec3) -> Result<(), PhysicsError> {
        if !is_finite(torque) {
            return Err(PhysicsError::InvalidForce);
        }
        self.torque = add(self.torque, torque);
        Ok(())
    }

    /// **Accumulate** a world-frame torque, N·m, rotating it into the body frame with
    /// the current orientation.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidForce`] if any component is not finite.
    pub fn apply_torque_world(&mut self, torque: Vec3) -> Result<(), PhysicsError> {
        if !is_finite(torque) {
            return Err(PhysicsError::InvalidForce);
        }
        let body = self.orientation.inverse().rotate_point(torque);
        self.torque = add(self.torque, body);
        Ok(())
    }

    /// Discard any torque accumulated but not yet stepped.
    #[inline]
    pub fn clear_torque(&mut self) {
        self.torque = (0.0, 0.0, 0.0);
    }

    /// Apply an angular impulse `J` in the body frame, kg·m²/s: `Δω = I⁻¹ J`,
    /// immediately.
    ///
    /// Unlike a torque this is *not* accumulated, and that difference is physical rather
    /// than a convention. An impulse is the limit of a large torque over a vanishing
    /// time, and over a vanishing time the gyroscopic term contributes nothing, so
    /// `Δω = I⁻¹J` is exact and there is nothing for a step to add.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidForce`] if any component is not finite.
    pub fn apply_angular_impulse_body(&mut self, impulse: Vec3) -> Result<(), PhysicsError> {
        if !is_finite(impulse) {
            return Err(PhysicsError::InvalidForce);
        }
        self.velocity = add(self.velocity, self.inverse.multiply_vector(impulse));
        Ok(())
    }

    /// Apply an angular impulse in the world frame, kg·m²/s. See
    /// [`apply_angular_impulse_body`].
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidForce`] if any component is not finite.
    ///
    /// [`apply_angular_impulse_body`]: RigidBodyRotation::apply_angular_impulse_body
    pub fn apply_angular_impulse_world(&mut self, impulse: Vec3) -> Result<(), PhysicsError> {
        if !is_finite(impulse) {
            return Err(PhysicsError::InvalidForce);
        }
        let body = self.orientation.inverse().rotate_point(impulse);
        self.velocity = add(self.velocity, self.inverse.multiply_vector(body));
        Ok(())
    }

    /// Apply the angular impulse an off-centre linear impulse produces: `Δω = I⁻¹(r × J)`.
    ///
    /// # Arguments
    ///
    /// * `lever` - Body-frame vector from the centre of mass to where the impulse acted, m.
    /// * `impulse` - Body-frame linear impulse, kg·m/s.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidForce`] if any component of either argument is not finite.
    pub fn apply_offset_impulse_body(
        &mut self,
        lever: Vec3,
        impulse: Vec3,
    ) -> Result<(), PhysicsError> {
        if !is_finite(lever) || !is_finite(impulse) {
            return Err(PhysicsError::InvalidForce);
        }
        self.apply_angular_impulse_body(cross_product(lever, impulse))
    }

    //--------------------------------------------------------------------------
    // Advancing time
    //--------------------------------------------------------------------------

    /// Substeps the accuracy criteria ask for, before the cap. Fractional and possibly
    /// enormous; [`substeps_for`] clamps it and [`step`] refuses when it exceeds the cap.
    ///
    /// [`substeps_for`]: RigidBodyRotation::substeps_for
    /// [`step`]: RigidBodyRotation::step
    fn required_substeps(&self, dt: f64) -> f64 {
        if !dt.is_finite() || dt <= 0.0 {
            return 1.0;
        }
        let speed = magnitude(self.velocity);

        // How far the orientation would turn over the whole step.
        let by_angle = speed * dt / MAX_STEP_RADIANS;

        // How far ω itself would move over the whole step, relative to its own size.
        //
        // Bounded rather than evaluated: `|ω̇| ≤ |ω|² + |I⁻¹τ|`, where the first term is
        // the triangle-inequality bound on the gyroscopic part that `new` guarantees.
        // That costs one matrix-vector product instead of the two products and a cross
        // an actual derivative evaluation would take, it never under-estimates, and it
        // makes the substep count a function of two magnitudes — no branches, nothing
        // per-substep. Note that with `τ = 0` it reduces to exactly `by_angle` whenever
        // `|ω|·dt ≤ 1`, so a free body pays nothing for this criterion at all.
        let torque_alpha = magnitude(self.inverse.multiply_vector(self.torque));
        let change = (speed * speed + torque_alpha) * dt;
        let scale = speed.max(change);
        let by_change = if scale > 0.0 {
            change / (MAX_RELATIVE_CHANGE_PER_SUBSTEP * scale)
        } else {
            0.0
        };

        let wanted = by_angle.max(by_change).ceil();
        if !wanted.is_finite() || wanted < 1.0 {
            1.0
        } else {
            wanted
        }
    }

    /// How many substeps a [`step`] of `dt` would take, clamped to [`MAX_SUBSTEPS`].
    /// Exposed so a caller can price a frame.
    ///
    /// [`step`]: RigidBodyRotation::step
    pub fn substeps_for(&self, dt: f64) -> u32 {
        let wanted = self.required_substeps(dt);
        if wanted >= MAX_SUBSTEPS as f64 {
            MAX_SUBSTEPS
        } else {
            wanted as u32
        }
    }

    /// **The longest `dt` this body can currently be stepped through**, in seconds.
    ///
    /// `MAX_SUBSTEPS × MAX_STEP_RADIANS / |ω|` — the cap on substeps and the cap on the
    /// angle each may turn multiply into a cap on the angle one call may cover, which is
    /// 8 radians. Beyond that [`step`] returns [`PhysicsError::InvalidTime`] rather than
    /// integrating something it cannot resolve: a fourth-order Runge-Kutta step across
    /// twenty radians of rotation does not produce a large rotation, it produces a `NaN`,
    /// and there is no useful answer to hand back.
    ///
    /// Infinite for a body at rest. The relative-change criterion can ask for more
    /// substeps than the angle criterion when a large torque acts on a slow body, so
    /// this is an upper bound rather than an exact threshold; `step`'s error is the
    /// authority.
    ///
    /// [`step`]: RigidBodyRotation::step
    #[inline]
    pub fn max_resolvable_dt(&self) -> f64 {
        let speed = magnitude(self.velocity);
        if speed <= 0.0 {
            f64::INFINITY
        } else {
            MAX_SUBSTEPS as f64 * MAX_STEP_RADIANS / speed
        }
    }

    /// **Advance the rotation by `dt` seconds**, integrating
    /// `ω̇ = I⁻¹(τ − ω × Iω)` together with `q̇ = ½ q ⊗ ω`, and clear the accumulated
    /// torque.
    ///
    /// This is the only method that moves `ω` forward in time, which is what makes the
    /// gyroscopic term non-optional; see the [module docs](self).
    ///
    /// # Scheme, and what bounds its error
    ///
    /// Classical fourth-order Runge-Kutta on the joint state `(q, ω)`, substepped so
    /// that no substep turns more than [`MAX_STEP_RADIANS`] or changes `ω` by more than
    /// [`MAX_RELATIVE_CHANGE_PER_SUBSTEP`] of itself, capped at [`MAX_SUBSTEPS`].
    ///
    /// The first criterion is a genuine bound and not a guess, and it rests on the
    /// tensor check in [`new`]: in principal axes `ω̇ᵢ = (Iⱼ − I_k)/Iᵢ · ωⱼ ω_k`, and the
    /// triangle inequality `Iⱼ ≤ I_k + Iᵢ` — which every real mass distribution obeys
    /// and [`new`] rejects a tensor for violating — makes `|Iⱼ − I_k| ≤ Iᵢ`, so the
    /// coefficient is at most 1 and `|ω̇| ≲ |ω|²`. The relative change in `ω` across a
    /// substep is therefore `≲ |ω|·h`, the same quantity the angle criterion bounds.
    ///
    /// RK4 is fourth order but not symplectic, so `|Iω|` and the energy drift slowly
    /// rather than exactly. [`tests::free_rotation_decays_rather_than_gains_over_a_long_run`] pins that
    /// drift over a minute of simulated time, which is the test that would catch a
    /// change to the scheme trading accuracy away.
    ///
    /// # Returns
    ///
    /// The number of substeps taken, so a caller can price the frame it just paid for.
    ///
    /// # Errors
    ///
    /// - [`PhysicsError::InvalidTime`] if `dt` is not finite, is negative, or is longer
    ///   than [`max_resolvable_dt`] — a step covering more than
    ///   `MAX_SUBSTEPS × MAX_STEP_RADIANS` = 8 radians of rotation cannot be integrated
    ///   accurately at any bounded cost, and handing back a plausible-looking wrong
    ///   answer would be worse than saying so. The state is left untouched; split the
    ///   step.
    /// - [`PhysicsError::CalculationError`] if the integrated state came out non-finite
    ///   anyway, which given the constructor's checks and the finiteness guards on every
    ///   setter requires magnitudes large enough to overflow `f64`. The previous state is
    ///   restored before returning, so the body remains usable and the invariant that
    ///   `ω` is finite is never broken.
    ///
    /// [`new`]: RigidBodyRotation::new
    /// [`max_resolvable_dt`]: RigidBodyRotation::max_resolvable_dt
    pub fn step(&mut self, dt: f64) -> Result<u32, PhysicsError> {
        if !dt.is_finite() || dt < 0.0 {
            return Err(PhysicsError::InvalidTime);
        }
        if dt == 0.0 {
            return Ok(1);
        }
        // Computed once. `substeps_for` would recompute it, and this is a per-body
        // per-frame path.
        let wanted = self.required_substeps(dt);
        if wanted > MAX_SUBSTEPS as f64 {
            return Err(PhysicsError::InvalidTime);
        }
        let substeps = wanted as u32;
        if self.integrate(dt, substeps) {
            Ok(substeps)
        } else {
            Err(PhysicsError::CalculationError(
                "rotational integration produced a non-finite state; angular velocity or torque magnitude overflowed".to_string(),
            ))
        }
    }

    /// **Step a contiguous batch of bodies.** Same physics as [`step`], arranged for the
    /// caller that has hundreds of them.
    ///
    /// The reason this exists rather than leaving callers to write the loop: `dt` is
    /// validated once for the whole slice rather than once per body, there is no
    /// `Result` constructed per body, and the bodies are walked in memory order. A body
    /// is 176 bytes of plain `f64`, so a `Vec<RigidBodyRotation>` streams.
    ///
    /// Be honest about what this is not. **It is not faster than the loop it replaces**:
    /// `benches/rigid_body_rotation.rs` measures 62.5 µs against 61.2 µs for 520 bodies,
    /// a wash. What it buys is contract — one validation, one error, a batch-level skip
    /// count — not throughput. It is also **not** cross-body SIMD; that would need a
    /// structure-of-arrays layout, and at 1.5% of a 240 Hz frame for the workload that
    /// motivated it, nothing has asked for one. What this is, is a shape that does not
    /// stand in the way of one later.
    ///
    /// # Returns
    ///
    /// The number of bodies that were **skipped** because `dt` exceeded their
    /// [`max_resolvable_dt`] or the integration came out non-finite. Skipped bodies are
    /// left exactly as they were — no `NaN` enters the batch — and their accumulated
    /// torque is *not* cleared, so re-stepping them with a shorter `dt` does the right
    /// thing. Zero is the normal answer; anything else means some body is spinning fast
    /// enough that the frame could not resolve it.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidTime`] if `dt` is not finite or is negative. Checked once,
    /// before the loop, and no body is touched.
    ///
    /// [`step`]: RigidBodyRotation::step
    /// [`max_resolvable_dt`]: RigidBodyRotation::max_resolvable_dt
    pub fn step_many(bodies: &mut [RigidBodyRotation], dt: f64) -> Result<usize, PhysicsError> {
        if !dt.is_finite() || dt < 0.0 {
            return Err(PhysicsError::InvalidTime);
        }
        if dt == 0.0 {
            return Ok(0);
        }
        let mut skipped = 0usize;
        for body in bodies.iter_mut() {
            let wanted = body.required_substeps(dt);
            if wanted > MAX_SUBSTEPS as f64 {
                skipped += 1;
                continue;
            }
            if !body.integrate(dt, wanted as u32) {
                skipped += 1;
            }
        }
        Ok(skipped)
    }

    /// The integration itself, shared by [`step`] and [`step_many`]. Returns `false` and
    /// leaves the body untouched if the result was not finite.
    ///
    /// No allocation, no `Result`, and the only branch is the finiteness check at the
    /// end — the substep loop is a fixed count decided before it starts.
    ///
    /// [`step`]: RigidBodyRotation::step
    /// [`step_many`]: RigidBodyRotation::step_many
    #[inline]
    fn integrate(&mut self, dt: f64, substeps: u32) -> bool {
        let substeps = substeps.max(1);
        let h = dt / substeps as f64;

        let mut q = self.orientation;
        let mut w = self.velocity;
        for _ in 0..substeps {
            let (nq, nw) = self.rk4(q, w, h);
            q = nq;
            w = nw;
        }

        let magnitude_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
        if !is_finite(w) || !magnitude_sq.is_finite() || magnitude_sq < 1e-20 {
            return false;
        }

        self.velocity = w;
        // Composing a quaternion a few hundred times a second walks it off the unit
        // sphere, and a non-unit quaternion does not fail loudly — it scales whatever it
        // is applied to. Renormalised once per call rather than trusted to stay unit.
        let inv = 1.0 / magnitude_sq.sqrt();
        self.orientation = Quaternion {
            w: q.w * inv,
            x: q.x * inv,
            y: q.y * inv,
            z: q.z * inv,
        };
        self.torque = (0.0, 0.0, 0.0);
        true
    }

    //--------------------------------------------------------------------------
    // The integrator itself
    //--------------------------------------------------------------------------

    /// `ω̇ = I⁻¹(τ − ω × Iω)`. The gyroscopic term is the `ω × Iω`.
    #[inline]
    fn velocity_derivative(&self, w: Vec3) -> Vec3 {
        let momentum = self.inertia.multiply_vector(w);
        let gyroscopic = cross_product(w, momentum);
        self.inverse.multiply_vector(sub(self.torque, gyroscopic))
    }

    /// `q̇ = ½ q ⊗ (0, ω)`.
    ///
    /// **On the right**, because `ω` is expressed in axes `q` has already rotated. The
    /// world-frame form is `½ (0, ω) ⊗ q`, and swapping the two turns the body at the
    /// right rate about the wrong axis without anything failing.
    #[inline]
    fn orientation_derivative(q: Quaternion, w: Vec3) -> Quaternion {
        let omega = Quaternion { w: 0.0, x: w.0, y: w.1, z: w.2 };
        let d = q.multiply(&omega);
        Quaternion { w: 0.5 * d.w, x: 0.5 * d.x, y: 0.5 * d.y, z: 0.5 * d.z }
    }

    /// One classical RK4 substep on the joint state `(q, ω)`.
    #[inline]
    fn rk4(&self, q: Quaternion, w: Vec3, h: f64) -> (Quaternion, Vec3) {
        let (dq1, dw1) = (Self::orientation_derivative(q, w), self.velocity_derivative(w));

        let q2 = q_axpy(q, dq1, 0.5 * h);
        let w2 = add(w, scale(dw1, 0.5 * h));
        let (dq2, dw2) = (Self::orientation_derivative(q2, w2), self.velocity_derivative(w2));

        let q3 = q_axpy(q, dq2, 0.5 * h);
        let w3 = add(w, scale(dw2, 0.5 * h));
        let (dq3, dw3) = (Self::orientation_derivative(q3, w3), self.velocity_derivative(w3));

        let q4 = q_axpy(q, dq3, h);
        let w4 = add(w, scale(dw3, h));
        let (dq4, dw4) = (Self::orientation_derivative(q4, w4), self.velocity_derivative(w4));

        let sixth = h / 6.0;
        let q_next = Quaternion {
            w: q.w + sixth * (dq1.w + 2.0 * dq2.w + 2.0 * dq3.w + dq4.w),
            x: q.x + sixth * (dq1.x + 2.0 * dq2.x + 2.0 * dq3.x + dq4.x),
            y: q.y + sixth * (dq1.y + 2.0 * dq2.y + 2.0 * dq3.y + dq4.y),
            z: q.z + sixth * (dq1.z + 2.0 * dq2.z + 2.0 * dq3.z + dq4.z),
        };
        let w_next = (
            w.0 + sixth * (dw1.0 + 2.0 * dw2.0 + 2.0 * dw3.0 + dw4.0),
            w.1 + sixth * (dw1.1 + 2.0 * dw2.1 + 2.0 * dw3.1 + dw4.1),
            w.2 + sixth * (dw1.2 + 2.0 * dw2.2 + 2.0 * dw3.2 + dw4.2),
        );
        (q_next, w_next)
    }
}

/// `q + s·d`, componentwise. Not a rotation composition — an arithmetic step in the
/// four-dimensional space the Runge-Kutta stages live in.
#[inline]
fn q_axpy(q: Quaternion, d: Quaternion, s: f64) -> Quaternion {
    Quaternion { w: q.w + d.w * s, x: q.x + d.x * s, y: q.y + d.y * s, z: q.z + d.z * s }
}

#[inline]
fn is_finite(v: Vec3) -> bool {
    v.0.is_finite() && v.1.is_finite() && v.2.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rotational_dynamics::{inertia_3d, AngularState3D};

    /// A brick with three distinct principal moments — the precondition for any of the
    /// free-rotation behaviour below. Dimensions chosen only to keep the moments well
    /// separated; the moments themselves come from `inertia_3d::solid_cuboid`, which is
    /// the crate's single source for the box formula.
    fn brick() -> RigidBodyRotation {
        RigidBodyRotation::new(inertia_3d::solid_cuboid(1.28, 0.09, 0.26, 0.07)).unwrap()
    }

    /// Index of the intermediate principal moment. Derived from the tensor rather than
    /// assumed, so the tests keep testing what they say if `brick` is ever edited.
    fn intermediate_axis(body: &RigidBodyRotation) -> usize {
        let (a, b, c) = body.inertia().diagonal;
        let mut order = [(a, 0usize), (b, 1), (c, 2)];
        order.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        order[1].1
    }

    fn component(v: Vec3, i: usize) -> f64 {
        match i {
            0 => v.0,
            1 => v.1,
            _ => v.2,
        }
    }

    fn set_component(v: &mut Vec3, i: usize, value: f64) {
        match i {
            0 => v.0 = value,
            1 => v.1 = value,
            _ => v.2 = value,
        }
    }

    // ==================== The defect this type exists to fix ====================

    /// **A free body with zero torque must change its angular velocity.**
    ///
    /// The whole claim, in its smallest form. `AngularState3D::apply_torque` provably
    /// fails this — the companion assertion below shows it failing — and it is the
    /// reason a caller reaching for the obviously-named function got a body that spun
    /// about one axis forever.
    #[test]
    fn free_body_with_no_torque_changes_its_angular_velocity() {
        let mut body = brick();
        // Not aligned with any principal axis, which is the condition for ω × Iω ≠ 0.
        body.set_angular_velocity_body((4.0, 0.7, 11.0)).unwrap();
        let start = body.angular_velocity_body();

        for _ in 0..120 {
            body.step(1.0 / 120.0).unwrap();
        }

        let moved = magnitude(sub(body.angular_velocity_body(), start));
        assert!(
            moved > 0.1,
            "ω moved by only {moved} rad/s in a second of free rotation — the gyroscopic \
             term is missing"
        );
    }

    /// The same setup through the old accumulator, pinned as *not* doing this. Kept so
    /// that the difference between the two is a fact the test suite states rather than
    /// something a reader has to take on trust.
    #[test]
    #[allow(deprecated)]
    fn angular_state_3d_apply_torque_is_an_accumulator_and_does_not_tumble() {
        let inertia = inertia_3d::solid_cuboid(1.28, 0.09, 0.26, 0.07);
        let mut state = AngularState3D::new((4.0, 0.7, 11.0));
        let start = state.velocity;

        for _ in 0..120 {
            state.apply_torque((0.0, 0.0, 0.0), 1.0 / 120.0, &inertia);
        }

        assert_eq!(
            state.velocity, start,
            "AngularState3D::apply_torque changed ω with zero torque; if this now moves, \
             the accumulator has grown a step and the two APIs have converged"
        );
    }

    // ==================== Conservation ====================

    /// **`|Iω|` is conserved under free rotation, and `|ω|` is not.**
    ///
    /// Both halves matter. The first is the invariant; the second is why clamping `|ω|`
    /// against a bound does not hold it — a body clamped exactly to a limit on the way
    /// in comes back out above it as it travels round its polhode.
    #[test]
    fn free_rotation_conserves_angular_momentum_but_not_speed() {
        let mut body = brick();
        body.set_angular_velocity_body((4.0, 0.7, 11.0)).unwrap();

        let momentum_0 = magnitude(body.angular_momentum_body());
        let speed_0 = body.speed();
        let mut speed_max: f64 = speed_0;
        let mut speed_min: f64 = speed_0;

        for _ in 0..480 {
            body.step(1.0 / 120.0).unwrap();
            speed_max = speed_max.max(body.speed());
            speed_min = speed_min.min(body.speed());
        }

        let drift = (magnitude(body.angular_momentum_body()) - momentum_0).abs() / momentum_0;
        assert!(drift < 1e-9, "|Iω| drifted by {drift} over four seconds");

        let spread = (speed_max - speed_min) / speed_0;
        assert!(
            spread > 0.01,
            "|ω| varied by only {spread} — it should breathe visibly while |Iω| does not, \
             and a test that saw both constant would be testing a body that cannot tumble"
        );
    }

    /// Energy, the independent second conserved quantity. Momentum alone can be held by
    /// an integrator that is still on the wrong trajectory; the two together pin it to
    /// the right polhode.
    #[test]
    fn free_rotation_conserves_energy() {
        let mut body = brick();
        body.set_angular_velocity_body((4.0, 0.7, 11.0)).unwrap();
        let energy_0 = body.kinetic_energy();

        for _ in 0..480 {
            body.step(1.0 / 120.0).unwrap();
        }

        let drift = (body.kinetic_energy() - energy_0).abs() / energy_0;
        assert!(drift < 1e-9, "energy drifted by {drift} over four seconds");
    }

    /// **The world-frame angular momentum *vector* is fixed**, not just its length.
    ///
    /// The strongest single assertion available here, because it fails if either half of
    /// the integration is wrong: a bad gyroscopic term moves `Iω` in the body frame, and
    /// a quaternion composed on the wrong side rotates it in the world frame at exactly
    /// the right rate about the wrong axis — the failure that looks like "the tumble is
    /// a bit off" and gets debugged for an afternoon.
    #[test]
    fn world_frame_angular_momentum_is_fixed_under_free_rotation() {
        let mut body = brick();
        body.set_orientation(Quaternion::from_axis_angle((1.0, 2.0, 3.0), 0.7))
            .unwrap();
        body.set_angular_velocity_body((4.0, 0.7, 11.0)).unwrap();

        let l0 = body.angular_momentum_world();
        for _ in 0..1200 {
            body.step(1.0 / 120.0).unwrap();
        }
        let l1 = body.angular_momentum_world();

        let error = magnitude(sub(l1, l0)) / magnitude(l0);
        assert!(
            error < 1e-6,
            "world-frame L moved by {error} of its magnitude over ten seconds: {l0:?} -> {l1:?}"
        );
    }

    /// The long run, and **the direction the error goes in**.
    ///
    /// RK4 is fourth order but not symplectic, so the conserved quantities drift
    /// secularly rather than exactly. Over a minute at 120 Hz — 7200 steps — the drift
    /// measures −1.8×10⁻⁸ in `|Iω|` and −4.0×10⁻⁸ in energy on this body.
    ///
    /// Both assertions matter and the *sign* one matters more. A bounded magnitude says
    /// the scheme is fourth order; the sign says the error is dissipative, and a tumble
    /// that imperceptibly slows is a different thing entirely from one that
    /// imperceptibly speeds up. An integrator that gains angular momentum produces
    /// debris that accelerates over its flight — the failure that is hardest to
    /// attribute later, because it looks like the launch impulse being too big rather
    /// than the integrator being wrong. Explicit Euler on this equation has exactly that
    /// sign, which is why it is not the scheme here.
    ///
    /// The magnitude bound is deliberately fifty times looser than the measurement, so
    /// it holds the shape of the guarantee without pinning a digit a compiler version
    /// could move. The sign has no slack because it should not have any.
    #[test]
    fn free_rotation_decays_rather_than_gains_over_a_long_run() {
        let mut body = brick();
        body.set_angular_velocity_body((4.0, 0.7, 11.0)).unwrap();
        let momentum_0 = magnitude(body.angular_momentum_body());
        let energy_0 = body.kinetic_energy();

        for _ in 0..(120 * 60) {
            body.step(1.0 / 120.0).unwrap();
        }

        let momentum_drift = (magnitude(body.angular_momentum_body()) - momentum_0) / momentum_0;
        let energy_drift = (body.kinetic_energy() - energy_0) / energy_0;

        assert!(
            momentum_drift <= 0.0,
            "|Iω| *gained* {momentum_drift} over a minute: the integrator is pumping energy \
             into a free body and its debris will accelerate as it flies"
        );
        assert!(
            energy_drift <= 0.0,
            "rotational energy *gained* {energy_drift} over a minute"
        );
        assert!(
            momentum_drift.abs() < 1e-6,
            "|Iω| drifted by {momentum_drift} over a minute"
        );
        assert!(
            energy_drift.abs() < 1e-6,
            "energy drifted by {energy_drift} over a minute"
        );
    }

    // ==================== The physical predictions ====================

    /// **The Dzhanibekov flip.** A body spun about its *intermediate* principal axis is
    /// unstable and turns end over end on its own, with nothing acting on it.
    ///
    /// The sharpest test in the subject: the behaviour is produced by the `ω × Iω` term
    /// alone, so it fails for any integrator missing the coupling, and it is a genuine
    /// physical prediction rather than a restatement of the code.
    #[test]
    fn intermediate_axis_spin_flips() {
        let mut body = brick();
        let middle = intermediate_axis(&body);

        let mut w = (0.0, 0.0, 0.0);
        set_component(&mut w, middle, 10.0);
        // A nudge: exactly on the axis is an equilibrium and the instability has nothing
        // to grow from. Any real body is always slightly off it.
        set_component(&mut w, (middle + 1) % 3, 0.02);
        body.set_angular_velocity_body(w).unwrap();

        let mut most_reversed: f64 = 1.0;
        for _ in 0..(240 * 6) {
            body.step(1.0 / 240.0).unwrap();
            most_reversed =
                most_reversed.min(component(body.angular_velocity_body(), middle) / 10.0);
        }

        assert!(
            most_reversed < -0.9,
            "the middle axis only reached {most_reversed} of its starting rate; it should \
             swing to about -1, and if it does not the gyroscopic term is gone"
        );
    }

    /// The other half of the claim: the largest and smallest principal axes are stable.
    /// Without this the flip test alone would pass for an integrator that simply made
    /// everything unstable.
    #[test]
    fn extreme_axis_spins_are_steady() {
        let (a, b, c) = brick().inertia().diagonal;
        let mut order = [(a, 0usize), (b, 1), (c, 2)];
        order.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());

        for axis in [order[0].1, order[2].1] {
            let mut body = brick();
            let mut w = (0.0, 0.0, 0.0);
            set_component(&mut w, axis, 10.0);
            set_component(&mut w, (axis + 1) % 3, 0.02);
            body.set_angular_velocity_body(w).unwrap();

            let mut lowest: f64 = 1.0;
            for _ in 0..(240 * 6) {
                body.step(1.0 / 240.0).unwrap();
                lowest = lowest.min(component(body.angular_velocity_body(), axis) / 10.0);
            }
            assert!(
                lowest > 0.99,
                "spin about principal axis {axis} fell to {lowest} of its starting rate; \
                 only the intermediate axis is unstable"
            );
        }
    }

    /// **A cube does not tumble.** Three equal moments make `ω × Iω = ω × Iω` vanish
    /// identically, so the body keeps a fixed axis exactly.
    ///
    /// Worth pinning as a degenerate case in both directions: it is the one shape for
    /// which the old accumulator was accidentally right, and a test suite built only on
    /// cubes would never have caught the missing term.
    #[test]
    fn a_cube_does_not_tumble() {
        let mut body =
            RigidBodyRotation::new(inertia_3d::solid_cuboid(2.0, 0.3, 0.3, 0.3)).unwrap();
        let start = (1.0, 2.0, 3.0);
        body.set_angular_velocity_body(start).unwrap();

        for _ in 0..1200 {
            body.step(1.0 / 120.0).unwrap();
        }

        let moved = magnitude(sub(body.angular_velocity_body(), start));
        assert!(
            moved < 1e-12,
            "an isotropic body's ω moved by {moved} rad/s; ω × Iω must vanish identically"
        );
    }

    /// A sphere, reached through a different constructor, for the same reason.
    #[test]
    fn a_sphere_does_not_tumble() {
        let mut body = RigidBodyRotation::new(inertia_3d::solid_sphere(3.0, 0.4)).unwrap();
        let start = (0.0, 5.0, 5.0);
        body.set_angular_velocity_body(start).unwrap();
        for _ in 0..600 {
            body.step(1.0 / 120.0).unwrap();
        }
        assert!(magnitude(sub(body.angular_velocity_body(), start)) < 1e-12);
    }

    // ==================== Torque ====================

    /// With a torque about a symmetry axis of a symmetric body the gyroscopic term stays
    /// zero, so `ω` is exactly `I⁻¹τ t` and the closed form is available to check
    /// against. This is the case in which the new integrator and the old accumulator
    /// must agree, and it is what says the extra term did not break plain torque.
    #[test]
    fn torque_about_a_symmetry_axis_matches_the_closed_form() {
        let inertia = inertia_3d::solid_cylinder(2.0, 0.3, 1.0);
        let mut body = RigidBodyRotation::new(inertia).unwrap();
        let izz = inertia.diagonal.2;

        // One second of 5 N·m about z, applied as a torque each frame.
        let dt = 1.0 / 100.0;
        for _ in 0..100 {
            body.apply_torque_body((0.0, 0.0, 5.0)).unwrap();
            body.step(dt).unwrap();
        }

        let expected = 5.0 * 1.0 / izz;
        let w = body.angular_velocity_body();
        assert!(
            (w.2 - expected).abs() < 1e-9 * expected,
            "ω_z came out {} against the closed form {expected}",
            w.2
        );
        assert!(w.0.abs() < 1e-12 && w.1.abs() < 1e-12, "torque about z leaked into x/y: {w:?}");
    }

    /// The accumulator is consumed by the step. Two torques in one frame sum; the next
    /// frame starts empty. This is the property that makes it impossible to apply the
    /// gyroscopic term once per torque instead of once per step.
    #[test]
    fn torque_accumulates_within_a_frame_and_clears_after_it() {
        let mut body = brick();
        body.apply_torque_body((1.0, 0.0, 0.0)).unwrap();
        body.apply_torque_body((2.0, 0.0, 0.0)).unwrap();
        assert_eq!(body.pending_torque_body(), (3.0, 0.0, 0.0));

        body.step(1.0 / 120.0).unwrap();
        assert_eq!(body.pending_torque_body(), (0.0, 0.0, 0.0));
    }

    /// A world-frame torque on a rotated body must reach the same physical result as the
    /// body-frame torque it equals. Catches an inverted rotation in either converter.
    #[test]
    fn world_and_body_torque_agree_once_the_frame_is_accounted_for() {
        let orientation = Quaternion::from_axis_angle((0.3, -0.7, 0.2), 1.1);
        let world_torque = (2.0, -1.0, 0.5);

        let mut by_world = brick();
        by_world.set_orientation(orientation).unwrap();
        by_world.apply_torque_world(world_torque).unwrap();

        let mut by_body = brick();
        by_body.set_orientation(orientation).unwrap();
        by_body
            .apply_torque_body(orientation.inverse().rotate_point(world_torque))
            .unwrap();

        let a = by_world.pending_torque_body();
        let b = by_body.pending_torque_body();
        assert!(magnitude(sub(a, b)) < 1e-12, "{a:?} vs {b:?}");
    }

    /// An impulse is instantaneous, so `Δω = I⁻¹J` exactly and it is not accumulated.
    ///
    /// The tensor is `(4, 6, 8)` rather than the `(2, 4, 8)` the older
    /// `AngularState3D` tests use, because `2 + 4 < 8` violates the triangle inequality
    /// and no body has those moments. `AngularState3D` accepts it; this type does not.
    #[test]
    fn angular_impulse_is_immediate_and_exact() {
        let mut body =
            RigidBodyRotation::new(InertiaTensor::diagonal_only(4.0, 6.0, 8.0)).unwrap();
        body.apply_angular_impulse_body((4.0, 6.0, 8.0)).unwrap();
        let w = body.angular_velocity_body();
        assert!((w.0 - 1.0).abs() < 1e-12 && (w.1 - 1.0).abs() < 1e-12 && (w.2 - 1.0).abs() < 1e-12);
        assert_eq!(body.pending_torque_body(), (0.0, 0.0, 0.0));
    }

    /// `Δω = I⁻¹(r × J)` for an off-centre hit.
    #[test]
    fn offset_impulse_produces_the_expected_spin() {
        let mut body = RigidBodyRotation::new(InertiaTensor::uniform(2.0)).unwrap();
        body.apply_offset_impulse_body((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
            .unwrap();
        // r × J = (0,0,1); I⁻¹ scales by 1/2.
        let w = body.angular_velocity_body();
        assert!(w.0.abs() < 1e-12 && w.1.abs() < 1e-12 && (w.2 - 0.5).abs() < 1e-12);
    }

    // ==================== Frames and orientation ====================

    /// A body spinning about a world axis it is already aligned with keeps that axis in
    /// world coordinates while its own frame turns underneath it.
    #[test]
    fn orientation_advances_at_the_rate_omega_says() {
        let mut body = RigidBodyRotation::new(InertiaTensor::uniform(1.0)).unwrap();
        body.set_angular_velocity_body((0.0, 0.0, 1.0)).unwrap();

        // One second at 1 rad/s about z.
        for _ in 0..1000 {
            body.step(1.0 / 1000.0).unwrap();
        }

        let turned = body.orientation().rotate_point((1.0, 0.0, 0.0));
        let expected = (1.0_f64.cos(), 1.0_f64.sin(), 0.0);
        assert!(
            magnitude(sub(turned, expected)) < 1e-9,
            "x turned to {turned:?}, expected {expected:?}"
        );
    }

    /// The orientation stays a unit quaternion across a long, fast run — the invariant
    /// that keeps it from silently scaling whatever it is applied to.
    #[test]
    fn orientation_stays_unit_length() {
        let mut body = brick();
        body.set_angular_velocity_body((30.0, 5.0, 80.0)).unwrap();
        for _ in 0..6000 {
            body.step(1.0 / 120.0).unwrap();
        }
        assert!((body.orientation().magnitude() - 1.0).abs() < 1e-12);
    }

    /// Body and world angular velocity are the same vector in two frames.
    #[test]
    fn body_and_world_angular_velocity_round_trip() {
        let mut body = brick();
        body.set_orientation(Quaternion::from_axis_angle((1.0, 1.0, 0.0), 2.0))
            .unwrap();
        let world = (3.0, -1.0, 4.0);
        body.set_angular_velocity_world(world).unwrap();
        let back = body.angular_velocity_world();
        assert!(magnitude(sub(back, world)) < 1e-12, "{back:?} vs {world:?}");
    }

    // ==================== Validity at the edges ====================

    /// A tensor with a zero moment cannot be integrated, and the constructor is where
    /// that is refused. `thin_rod_center` returns exactly such a tensor, deliberately, so
    /// this is a case a real caller can reach.
    #[test]
    fn a_singular_tensor_is_refused() {
        let rod = inertia_3d::thin_rod_center(1.0, 2.0);
        assert!(!rod.is_positive_definite());
        assert_eq!(
            RigidBodyRotation::new(rod).unwrap_err(),
            PhysicsError::InvalidInertiaTensor
        );
    }

    /// And so is a tensor no mass distribution could produce.
    #[test]
    fn a_tensor_violating_the_triangle_inequality_is_refused() {
        // 1 + 1 < 5: physically impossible, and the substep bound in `step` would have
        // no basis for it.
        let bogus = InertiaTensor::diagonal_only(1.0, 1.0, 5.0);
        assert!(bogus.is_positive_definite());
        assert!(!bogus.satisfies_triangle_inequality());
        assert_eq!(
            RigidBodyRotation::new(bogus).unwrap_err(),
            PhysicsError::InvalidInertiaTensor
        );
    }

    #[test]
    fn non_finite_tensors_are_refused() {
        for t in [
            InertiaTensor::diagonal_only(f64::NAN, 1.0, 1.0),
            InertiaTensor::diagonal_only(f64::INFINITY, 1.0, 1.0),
            InertiaTensor::diagonal_only(-1.0, 1.0, 1.0),
            InertiaTensor::diagonal_only(0.0, 0.0, 0.0),
        ] {
            assert_eq!(
                RigidBodyRotation::new(t).unwrap_err(),
                PhysicsError::InvalidInertiaTensor,
                "accepted {t:?}"
            );
        }
    }

    /// Every real box, cylinder and sphere the crate can build is accepted. Guards
    /// against a validity check tight enough to reject the crate's own constructors.
    #[test]
    fn the_crates_own_shape_tensors_are_accepted() {
        let shapes = [
            inertia_3d::solid_sphere(1.0, 0.5),
            inertia_3d::hollow_sphere(1.0, 0.5),
            inertia_3d::solid_cylinder(1.0, 0.2, 1.0),
            inertia_3d::solid_cylinder(1.0, 2.0, 0.05),
            inertia_3d::solid_cuboid(1.0, 0.1, 0.1, 0.1),
            inertia_3d::solid_cuboid(1.0, 3.0, 0.02, 0.02),
            inertia_3d::solid_cuboid(1.0, 0.02, 3.0, 0.9),
        ];
        for s in shapes {
            assert!(RigidBodyRotation::new(s).is_ok(), "refused {s:?}");
        }
    }

    /// An off-diagonal tensor — a body whose own axes are not its principal axes — is
    /// accepted, and the parallel-axis theorem is the usual way to get one.
    #[test]
    fn an_off_diagonal_tensor_integrates() {
        let shifted = inertia_3d::parallel_axis(
            inertia_3d::solid_cuboid(1.0, 0.2, 0.1, 0.05),
            1.0,
            (0.3, 0.2, 0.1),
        );
        let mut body = RigidBodyRotation::new(shifted).unwrap();
        body.set_angular_velocity_body((2.0, 3.0, 1.0)).unwrap();
        let momentum_0 = magnitude(body.angular_momentum_body());
        for _ in 0..1200 {
            body.step(1.0 / 120.0).unwrap();
        }
        let drift = (magnitude(body.angular_momentum_body()) - momentum_0).abs() / momentum_0;
        assert!(drift < 1e-8, "|Iω| drifted {drift} for a non-diagonal tensor");
    }

    #[test]
    fn bad_timesteps_are_reported_not_absorbed() {
        let mut body = brick();
        body.set_angular_velocity_body((1.0, 2.0, 3.0)).unwrap();

        assert_eq!(body.step(f64::NAN).unwrap_err(), PhysicsError::InvalidTime);
        assert_eq!(body.step(f64::INFINITY).unwrap_err(), PhysicsError::InvalidTime);
        assert_eq!(body.step(-0.1).unwrap_err(), PhysicsError::InvalidTime);
        // Zero is not an error, and does nothing.
        assert_eq!(body.step(0.0).unwrap(), 1);
        assert_eq!(body.angular_velocity_body(), (1.0, 2.0, 3.0));
    }

    #[test]
    fn non_finite_inputs_are_refused_at_the_edge() {
        let mut body = brick();
        assert_eq!(
            body.set_angular_velocity_body((f64::NAN, 0.0, 0.0)).unwrap_err(),
            PhysicsError::InvalidVelocity
        );
        assert_eq!(
            body.apply_torque_body((0.0, f64::INFINITY, 0.0)).unwrap_err(),
            PhysicsError::InvalidForce
        );
        assert_eq!(
            body.apply_angular_impulse_body((0.0, 0.0, f64::NAN)).unwrap_err(),
            PhysicsError::InvalidForce
        );
        // None of the rejected values landed.
        assert_eq!(body.angular_velocity_body(), (0.0, 0.0, 0.0));
        assert_eq!(body.pending_torque_body(), (0.0, 0.0, 0.0));
    }

    /// A rejected torque must not discard the good ones already accumulated this frame.
    #[test]
    fn a_rejected_torque_leaves_the_accumulator_intact() {
        let mut body = brick();
        body.apply_torque_body((1.0, 2.0, 3.0)).unwrap();
        assert!(body.apply_torque_body((f64::NAN, 0.0, 0.0)).is_err());
        assert_eq!(body.pending_torque_body(), (1.0, 2.0, 3.0));
    }

    // ==================== Substepping ====================

    /// A slow body costs one substep; a fast one costs more; and the count never exceeds
    /// the cap however absurd the input.
    #[test]
    fn substep_count_tracks_the_step_size_and_is_capped() {
        let mut body = brick();
        body.set_angular_velocity_body((0.0, 1.0, 0.0)).unwrap();
        assert_eq!(body.substeps_for(1.0 / 120.0), 1);

        body.set_angular_velocity_body((0.0, 200.0, 0.0)).unwrap();
        assert!(body.substeps_for(1.0 / 120.0) > 1);

        body.set_angular_velocity_body((1e6, 1e6, 1e6)).unwrap();
        assert_eq!(body.substeps_for(1.0), MAX_SUBSTEPS);
        assert_eq!(body.substeps_for(1e9), MAX_SUBSTEPS);
    }

    /// **A step too large to resolve is refused, not approximated.**
    ///
    /// Past `MAX_SUBSTEPS × MAX_STEP_RADIANS` of rotation there is no accurate answer at
    /// bounded cost, and an RK4 step across twenty radians does not produce a large
    /// rotation — it produces a `NaN`. The state must come back untouched so the caller
    /// can split the step and try again.
    #[test]
    fn a_step_too_large_to_resolve_is_refused_and_changes_nothing() {
        let mut body = brick();
        let fast = (300.0, 40.0, 700.0);
        body.set_angular_velocity_body(fast).unwrap();
        body.apply_torque_body((1.0, 0.0, 0.0)).unwrap();

        let limit = body.max_resolvable_dt();
        assert!(limit < 1.0, "a body at {} rad/s should not resolve a one-second step", body.speed());

        assert_eq!(body.step(1.0).unwrap_err(), PhysicsError::InvalidTime);
        assert_eq!(body.angular_velocity_body(), fast, "a refused step moved ω");
        assert_eq!(
            body.pending_torque_body(),
            (1.0, 0.0, 0.0),
            "a refused step consumed the torque, so re-stepping would lose it"
        );

        // Just inside the limit it goes through, and stays finite.
        let taken = body.step(limit * 0.9).unwrap();
        assert!(taken <= MAX_SUBSTEPS);
        assert!(is_finite(body.angular_velocity_body()));
        assert!((body.orientation().magnitude() - 1.0).abs() < 1e-9);
    }

    // ==================== Batch stepping ====================

    /// `step_many` must be the same physics as `step`, bit for bit. If it ever is not,
    /// the batch path has quietly become a second implementation.
    #[test]
    fn step_many_agrees_with_step_exactly() {
        let mut batch: Vec<RigidBodyRotation> = Vec::new();
        let seeds = [
            (4.0, 0.7, 11.0),
            (0.0, 0.0, 3.0),
            (-2.0, 9.0, 0.5),
            (0.01, 0.0, 0.0),
        ];
        for w in seeds {
            let mut b = brick();
            b.set_angular_velocity_body(w).unwrap();
            b.apply_torque_body((0.1, -0.2, 0.05)).unwrap();
            batch.push(b);
        }
        let mut singles = batch.clone();

        let dt = 1.0 / 120.0;
        for _ in 0..600 {
            assert_eq!(RigidBodyRotation::step_many(&mut batch, dt).unwrap(), 0);
            for b in singles.iter_mut() {
                b.step(dt).unwrap();
            }
            for b in batch.iter_mut() {
                b.apply_torque_body((0.1, -0.2, 0.05)).unwrap();
            }
            for b in singles.iter_mut() {
                b.apply_torque_body((0.1, -0.2, 0.05)).unwrap();
            }
        }

        for (a, b) in batch.iter().zip(singles.iter()) {
            assert_eq!(a.angular_velocity_body(), b.angular_velocity_body());
            assert_eq!(a.orientation(), b.orientation());
        }
    }

    /// One unresolvable body in a batch must not damage the rest, and must be counted.
    #[test]
    fn step_many_skips_what_it_cannot_resolve_and_says_how_many() {
        let mut bodies = vec![brick(), brick(), brick()];
        bodies[0].set_angular_velocity_body((1.0, 2.0, 3.0)).unwrap();
        bodies[1].set_angular_velocity_body((5000.0, 0.0, 9000.0)).unwrap();
        bodies[2].set_angular_velocity_body((0.5, 0.0, 0.0)).unwrap();
        let untouched = bodies[1];

        let skipped = RigidBodyRotation::step_many(&mut bodies, 1.0 / 120.0).unwrap();
        assert_eq!(skipped, 1);
        assert_eq!(bodies[1], untouched, "the skipped body was modified");
        assert!(is_finite(bodies[0].angular_velocity_body()));
        assert!(is_finite(bodies[2].angular_velocity_body()));
        assert_ne!(bodies[0].orientation(), Quaternion::identity());
    }

    #[test]
    fn step_many_validates_dt_once_and_touches_nothing_when_it_is_bad() {
        let mut bodies = vec![brick(), brick()];
        bodies[0].set_angular_velocity_body((1.0, 2.0, 3.0)).unwrap();
        let before = bodies.clone();
        assert_eq!(
            RigidBodyRotation::step_many(&mut bodies, f64::NAN).unwrap_err(),
            PhysicsError::InvalidTime
        );
        assert_eq!(bodies, before);
        assert_eq!(RigidBodyRotation::step_many(&mut bodies, 0.0).unwrap(), 0);
        assert_eq!(bodies, before);
        assert_eq!(RigidBodyRotation::step_many(&mut [], 0.01).unwrap(), 0);
    }

    /// Halving the timestep must not change the answer beyond the integrator's own
    /// error. The direct statement that the substepping is a refinement and not a
    /// behaviour switch.
    #[test]
    fn the_trajectory_does_not_depend_on_the_frame_rate() {
        let mut coarse = brick();
        let mut fine = brick();
        coarse.set_angular_velocity_body((4.0, 0.7, 11.0)).unwrap();
        fine.set_angular_velocity_body((4.0, 0.7, 11.0)).unwrap();

        for _ in 0..240 {
            coarse.step(1.0 / 240.0).unwrap();
        }
        for _ in 0..2400 {
            fine.step(1.0 / 2400.0).unwrap();
        }

        let gap = magnitude(sub(coarse.angular_velocity_body(), fine.angular_velocity_body()));
        assert!(
            gap / fine.speed() < 1e-6,
            "a tenfold change in dt moved ω by {gap} rad/s after one second"
        );
    }
}
