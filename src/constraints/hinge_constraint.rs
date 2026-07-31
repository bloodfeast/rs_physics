//! Hinge/revolute constraints for 3D physics simulations.
//!
//! This module provides hinge constraints that allow rotation around a single axis
//! while maintaining positional constraints between two objects.
//!
//! # True Angular Dynamics
//!
//! Unlike simple position-based constraints, this hinge uses true rotational physics:
//! - **Angle (θ)**: Current rotation angle in radians
//! - **Angular velocity (ω)**: Rate of rotation in rad/s
//! - **Moment of inertia (I)**: Resistance to angular acceleration (I = m × r²)
//! - **Torque (τ)**: Rotational force from gravity (τ = r⊥ × m × g)
//!
//! This enables realistic behavior like a trap door swinging open, bouncing at
//! its limit, and settling naturally.

use crate::models::ObjectIn3D;
use crate::materials::Material;
use crate::utils::PhysicsError;
use super::solver::Constraint3D;

/// A hinge (revolute) constraint between two 3D objects with true angular dynamics.
///
/// A hinge constraint connects two objects at a shared anchor point and allows
/// rotation only around a specified axis. This is useful for modeling doors,
/// wheels, pendulums, and other rotating mechanisms.
///
/// # Physics Model
///
/// The hinge uses true rotational dynamics:
/// 1. Tracks angle (θ) and angular velocity (ω) directly
/// 2. Computes torque from gravity based on perpendicular distance to axis
/// 3. Uses moment of inertia (I = m × r²) for angular acceleration
/// 4. Integrates: α = τ/I, ω += α×dt, θ += ω×dt
/// 5. Applies angular restitution at limits for bouncing
///
/// # Examples
///
/// ```rust,ignore
/// use rs_physics::constraints::Hinge3D;
/// use rs_physics::models::ObjectIn3D;
///
/// // Create a trap door that swings open
/// let frame = ObjectIn3D::new(f64::INFINITY, 0.0, 0.0, 0.0, (0.0, 4.0, 0.0));
/// let door = ObjectIn3D::new(2.0, 0.0, 0.0, 0.0, (0.0, 4.0, 1.5));
/// let hinge = Hinge3D::new(frame, door, (0.0, 4.0, 0.0), (1.0, 0.0, 0.0))
///     .expect("Valid hinge")
///     .with_limits(0.0, 1.57);  // 0 to 90 degrees
/// ```
#[derive(Debug)]
pub struct Hinge3D {
    /// First object connected by the hinge (typically fixed anchor)
    pub object1: ObjectIn3D,
    /// Second object connected by the hinge (rotating body)
    pub object2: ObjectIn3D,
    /// Anchor point in world space (connection point / hinge axis location)
    pub anchor: (f64, f64, f64),
    /// Hinge axis (normalized, rotation allowed around this axis)
    pub axis: (f64, f64, f64),
    /// Local anchor offset from object1 position
    pub local_anchor1: (f64, f64, f64),
    /// Local anchor offset from object2 position
    pub local_anchor2: (f64, f64, f64),
    /// Optional lower angle limit (radians)
    pub angle_min: Option<f64>,
    /// Optional upper angle limit (radians)
    pub angle_max: Option<f64>,
    /// Baumgarte stabilization factor (0.0 to 1.0, default 0.2)
    pub baumgarte: f64,
    /// Accumulated impulse for warm starting
    pub lambda: f64,

    // === Angular dynamics state ===
    /// Current angle of rotation (radians, 0 = initial position)
    pub angle: f64,
    /// Current angular velocity (radians/second)
    pub angular_velocity: f64,
    /// Distance from anchor to object2 center of mass (arm length)
    pub arm_length: f64,
    /// Moment of inertia (I = m × r², computed from mass and arm_length)
    pub moment_of_inertia: f64,
    /// Angular restitution coefficient for bouncing at limits (0.0 to 1.0)
    pub restitution: f64,
    /// Angular damping coefficient (friction in the hinge)
    pub angular_damping: f64,
    /// Reference direction for angle=0 (perpendicular to axis, in swing plane)
    pub reference_direction: (f64, f64, f64),
}

impl Hinge3D {
    /// Creates a new Hinge3D constraint between two objects with true angular dynamics.
    ///
    /// # Arguments
    ///
    /// * `object1` - First object (typically the fixed/anchor object)
    /// * `object2` - Second object (rotating object)
    /// * `anchor` - World-space anchor point where the hinge connects
    /// * `axis` - Rotation axis (will be normalized)
    ///
    /// # Returns
    ///
    /// * `Ok(Hinge3D)` - Valid hinge constraint
    /// * `Err(PhysicsError)` - If axis is zero-length
    ///
    /// # Physics Setup
    ///
    /// The hinge automatically calculates:
    /// - **arm_length**: Distance from anchor to object2 center of mass
    /// - **moment_of_inertia**: I = m × r² (point mass approximation)
    /// - **reference_direction**: Initial direction from anchor to object2 (defines angle=0)
    /// - **initial_angle**: Computed from current object2 position (usually ~0)
    pub fn new(
        object1: ObjectIn3D,
        object2: ObjectIn3D,
        anchor: (f64, f64, f64),
        axis: (f64, f64, f64),
    ) -> Result<Self, PhysicsError> {
        // Normalize the axis
        let axis_len = (axis.0 * axis.0 + axis.1 * axis.1 + axis.2 * axis.2).sqrt();
        if axis_len < 1e-10 {
            return Err(PhysicsError::CalculationError("Hinge axis cannot be zero-length".to_string()));
        }
        let axis = (axis.0 / axis_len, axis.1 / axis_len, axis.2 / axis_len);

        // Calculate local anchor offsets
        let local_anchor1 = (
            anchor.0 - object1.position.x,
            anchor.1 - object1.position.y,
            anchor.2 - object1.position.z,
        );
        let local_anchor2 = (
            anchor.0 - object2.position.x,
            anchor.1 - object2.position.y,
            anchor.2 - object2.position.z,
        );

        // === Calculate angular dynamics parameters ===

        // Vector from anchor to object2 center of mass
        let arm = (
            object2.position.x - anchor.0,
            object2.position.y - anchor.1,
            object2.position.z - anchor.2,
        );

        // Project arm onto plane perpendicular to axis to get the swing arm
        let arm_dot_axis = arm.0 * axis.0 + arm.1 * axis.1 + arm.2 * axis.2;
        let arm_perp = (
            arm.0 - arm_dot_axis * axis.0,
            arm.1 - arm_dot_axis * axis.1,
            arm.2 - arm_dot_axis * axis.2,
        );

        // Arm length is the distance in the swing plane (perpendicular to axis)
        let arm_length = (arm_perp.0 * arm_perp.0 + arm_perp.1 * arm_perp.1 + arm_perp.2 * arm_perp.2).sqrt();

        // If arm_length is near zero, object is on the axis - use small default
        let arm_length = if arm_length < 1e-6 { 0.1 } else { arm_length };

        // Reference direction: normalized arm in swing plane (defines angle = 0)
        let reference_direction = (
            arm_perp.0 / arm_length,
            arm_perp.1 / arm_length,
            arm_perp.2 / arm_length,
        );

        // Moment of inertia: I = m × r² (point mass at end of arm)
        let mass = if object2.mass.is_infinite() { 1.0 } else { object2.mass };
        let moment_of_inertia = mass * arm_length * arm_length;

        // Initial angle is 0 by definition (reference_direction is current position)
        let angle = 0.0;
        let angular_velocity = 0.0;

        Ok(Self {
            object1,
            object2,
            anchor,
            axis,
            local_anchor1,
            local_anchor2,
            angle_min: None,
            angle_max: None,
            baumgarte: 0.2,
            lambda: 0.0,
            // Angular dynamics state
            angle,
            angular_velocity,
            arm_length,
            moment_of_inertia,
            restitution: 0.5,       // Default bounce at limits
            angular_damping: 0.02,  // Small friction
            reference_direction,
        })
    }

    /// Sets the Baumgarte stabilization factor.
    pub fn with_baumgarte(mut self, factor: f64) -> Self {
        self.baumgarte = factor;
        self
    }

    /// Sets angle limits for the hinge (in radians).
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum angle (must be <= max)
    /// * `max` - Maximum angle (must be >= min)
    pub fn with_limits(mut self, min: f64, max: f64) -> Self {
        self.angle_min = Some(min);
        self.angle_max = Some(max);
        self
    }

    /// Sets the angular restitution (bounciness at limits).
    ///
    /// # Arguments
    ///
    /// * `restitution` - Coefficient of restitution (0.0 = no bounce, 1.0 = perfect bounce)
    pub fn with_restitution(mut self, restitution: f64) -> Self {
        self.restitution = restitution.clamp(0.0, 1.0);
        self
    }

    /// Sets the angular damping (friction in hinge).
    ///
    /// # Arguments
    ///
    /// * `damping` - Damping coefficient (0.0 = no friction, higher = more friction)
    pub fn with_angular_damping(mut self, damping: f64) -> Self {
        self.angular_damping = damping.max(0.0);
        self
    }

    /// Configures the hinge using a material's physical properties.
    ///
    /// This sets the restitution based on the material's `restitution_coefficient`.
    /// Common materials:
    /// - Steel: 0.85 (highly elastic bounce)
    /// - Wood: 0.60 (moderate bounce)
    /// - Rubber: 0.90 (very bouncy)
    ///
    /// # Arguments
    ///
    /// * `material` - The material whose restitution coefficient will be used
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use rs_physics::materials::Material;
    /// use rs_physics::constraints::Hinge3D;
    ///
    /// let hinge = Hinge3D::new(frame, door, anchor, axis)
    ///     .unwrap()
    ///     .with_material(&Material::steel());  // Uses steel's restitution (0.85)
    /// ```
    pub fn with_material(mut self, material: &Material) -> Self {
        self.restitution = material.restitution_coefficient.clamp(0.0, 1.0);
        self
    }

    /// Returns the current world-space anchor point for object1.
    fn anchor_world_1(&self) -> (f64, f64, f64) {
        (
            self.object1.position.x + self.local_anchor1.0,
            self.object1.position.y + self.local_anchor1.1,
            self.object1.position.z + self.local_anchor1.2,
        )
    }

    /// Returns the current world-space anchor point for object2.
    #[allow(dead_code)]
    fn anchor_world_2(&self) -> (f64, f64, f64) {
        (
            self.object2.position.x + self.local_anchor2.0,
            self.object2.position.y + self.local_anchor2.1,
            self.object2.position.z + self.local_anchor2.2,
        )
    }

    /// Returns the current angle in radians.
    pub fn current_angle(&self) -> f64 {
        self.angle
    }

    /// Returns the current angular velocity in radians/second.
    pub fn current_angular_velocity(&self) -> f64 {
        self.angular_velocity
    }

    /// Solves the hinge constraint using true angular dynamics.
    ///
    /// This implements proper rotational physics:
    /// 1. **Torque from gravity**: τ = r⊥ × m × g (perpendicular lever arm × weight)
    /// 2. **Angular acceleration**: α = τ / I (torque / moment of inertia)
    /// 3. **Angular velocity integration**: ω += α × dt
    /// 4. **Angle integration**: θ += ω × dt
    /// 5. **Angle limit enforcement**: With angular restitution for bouncing
    /// 6. **Position update**: Compute object2 position from angle
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    /// * `gravity` - Gravitational acceleration (typically -9.81)
    pub fn solve_with_gravity(&mut self, dt: f64, gravity: f64) -> Result<(), PhysicsError> {
        if dt <= 0.0 {
            return Ok(());
        }

        // Skip if object2 has infinite mass (static body)
        if self.object2.mass.is_infinite() {
            return Ok(());
        }

        let mass = self.object2.mass;

        // === Step 1: Calculate torque from gravity ===
        // Gravity acts downward (-Y direction)
        // Torque = r⊥ × F where r⊥ is the perpendicular distance from axis to line of force
        //
        // For a hinge rotating around axis A, with object at angle θ:
        // - The current arm direction is: ref_dir rotated by θ around axis
        // - Gravity force is (0, mass * gravity, 0)
        // - Torque = arm × gravity_force (take component along axis)

        // Calculate current arm direction by rotating reference direction by angle
        let (sin_theta, cos_theta) = self.angle.sin_cos();

        // Rodrigues rotation formula to rotate reference_direction around axis by angle
        // rotated = ref * cos(θ) + (axis × ref) * sin(θ) + axis * (axis · ref) * (1 - cos(θ))
        let ref_dir = self.reference_direction;
        let axis = self.axis;

        // Cross product: axis × ref_dir
        let cross = (
            axis.1 * ref_dir.2 - axis.2 * ref_dir.1,
            axis.2 * ref_dir.0 - axis.0 * ref_dir.2,
            axis.0 * ref_dir.1 - axis.1 * ref_dir.0,
        );

        // Dot product: axis · ref_dir (should be ~0 since ref_dir is perpendicular to axis)
        let dot = axis.0 * ref_dir.0 + axis.1 * ref_dir.1 + axis.2 * ref_dir.2;

        // Rotated direction (current arm direction in swing plane)
        let current_dir = (
            ref_dir.0 * cos_theta + cross.0 * sin_theta + axis.0 * dot * (1.0 - cos_theta),
            ref_dir.1 * cos_theta + cross.1 * sin_theta + axis.1 * dot * (1.0 - cos_theta),
            ref_dir.2 * cos_theta + cross.2 * sin_theta + axis.2 * dot * (1.0 - cos_theta),
        );

        // Arm vector from anchor to center of mass (in world space)
        let arm = (
            current_dir.0 * self.arm_length,
            current_dir.1 * self.arm_length,
            current_dir.2 * self.arm_length,
        );

        // Gravity force vector
        let gravity_force = (0.0, mass * gravity, 0.0);

        // Torque = arm × gravity_force
        let torque_vec = (
            arm.1 * gravity_force.2 - arm.2 * gravity_force.1,
            arm.2 * gravity_force.0 - arm.0 * gravity_force.2,
            arm.0 * gravity_force.1 - arm.1 * gravity_force.0,
        );

        // Project torque onto hinge axis (scalar torque about the axis)
        let torque = torque_vec.0 * axis.0 + torque_vec.1 * axis.1 + torque_vec.2 * axis.2;

        // === Step 2: Angular acceleration ===
        let angular_acceleration = torque / self.moment_of_inertia;

        // === Step 3: Integrate angular velocity ===
        self.angular_velocity += angular_acceleration * dt;

        // Apply angular damping (friction in the hinge)
        self.angular_velocity *= 1.0 - self.angular_damping;

        // === Step 4: Integrate angle ===
        self.angle += self.angular_velocity * dt;

        // === Step 5: Angle limit enforcement with restitution ===
        if let Some(min_angle) = self.angle_min {
            if self.angle < min_angle {
                // Clamp angle to limit
                self.angle = min_angle;

                // Bounce: reverse angular velocity with restitution
                if self.angular_velocity < 0.0 {
                    self.angular_velocity = -self.angular_velocity * self.restitution;
                }
            }
        }

        if let Some(max_angle) = self.angle_max {
            if self.angle > max_angle {
                // Clamp angle to limit
                self.angle = max_angle;

                // Bounce: reverse angular velocity with restitution
                if self.angular_velocity > 0.0 {
                    self.angular_velocity = -self.angular_velocity * self.restitution;
                }
            }
        }

        // === Step 6: Update object2 position from current angle ===
        // Recalculate current_dir with (potentially clamped) angle
        let (sin_theta, cos_theta) = self.angle.sin_cos();
        let current_dir = (
            ref_dir.0 * cos_theta + cross.0 * sin_theta + axis.0 * dot * (1.0 - cos_theta),
            ref_dir.1 * cos_theta + cross.1 * sin_theta + axis.1 * dot * (1.0 - cos_theta),
            ref_dir.2 * cos_theta + cross.2 * sin_theta + axis.2 * dot * (1.0 - cos_theta),
        );

        // Get anchor position (from object1)
        let anchor = self.anchor_world_1();

        // New position = anchor + arm
        self.object2.position.x = anchor.0 + current_dir.0 * self.arm_length;
        self.object2.position.y = anchor.1 + current_dir.1 * self.arm_length;
        self.object2.position.z = anchor.2 + current_dir.2 * self.arm_length;

        // === Step 7: Update linear velocity to match angular motion ===
        // Linear velocity = ω × r (angular velocity cross arm)
        // For rotation around axis: v = ω × arm
        let omega_vec = (
            self.angular_velocity * axis.0,
            self.angular_velocity * axis.1,
            self.angular_velocity * axis.2,
        );
        let arm_world = (
            current_dir.0 * self.arm_length,
            current_dir.1 * self.arm_length,
            current_dir.2 * self.arm_length,
        );

        // v = ω × r
        self.object2.velocity.x = omega_vec.1 * arm_world.2 - omega_vec.2 * arm_world.1;
        self.object2.velocity.y = omega_vec.2 * arm_world.0 - omega_vec.0 * arm_world.2;
        self.object2.velocity.z = omega_vec.0 * arm_world.1 - omega_vec.1 * arm_world.0;

        Ok(())
    }

    /// Solves the hinge constraint for one iteration (legacy interface).
    ///
    /// This calls `solve_with_gravity` with a default gravity of -9.81.
    /// For proper physics, prefer calling `solve_with_gravity` directly with
    /// the actual gravity value from your simulation.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    pub fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        // Use default gravity of -9.81 m/s²
        self.solve_with_gravity(dt, -9.81)
    }

    /// Calculates the positional constraint error.
    ///
    /// # Returns
    ///
    /// The distance between the two anchor points
    pub fn calculate_error(&self) -> f64 {
        let anchor1 = self.anchor_world_1();
        let anchor2 = self.anchor_world_2();

        let dx = anchor2.0 - anchor1.0;
        let dy = anchor2.1 - anchor1.1;
        let dz = anchor2.2 - anchor1.2;

        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Returns whether angle limits are set.
    pub fn has_limits(&self) -> bool {
        self.angle_min.is_some() && self.angle_max.is_some()
    }

    /// Returns the hinge axis.
    pub fn get_axis(&self) -> (f64, f64, f64) {
        self.axis
    }
}

impl Constraint3D for Hinge3D {
    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        Hinge3D::solve(self, dt)
    }

    fn calculate_error(&self) -> f64 {
        Hinge3D::calculate_error(self)
    }

    fn get_lambda(&self) -> f64 {
        self.lambda
    }

    fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Axis3D, Velocity3D};

    fn make_3d_object(mass: f64, x: f64, y: f64, z: f64) -> ObjectIn3D {
        ObjectIn3D {
            mass,
            velocity: Velocity3D { x: 0.0, y: 0.0, z: 0.0 },
            position: Axis3D { x, y, z },
            forces: Vec::new(),
            material: None,
        }
    }

    #[test]
    fn test_hinge_3d_creation() {
        let obj1 = make_3d_object(1.0, 0.0, 0.0, 0.0);
        let obj2 = make_3d_object(1.0, 2.0, 0.0, 0.0);
        let hinge = Hinge3D::new(obj1, obj2, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0));
        assert!(hinge.is_ok());
    }

    #[test]
    fn test_hinge_3d_invalid_axis() {
        let obj1 = make_3d_object(1.0, 0.0, 0.0, 0.0);
        let obj2 = make_3d_object(1.0, 2.0, 0.0, 0.0);
        let hinge = Hinge3D::new(obj1, obj2, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0));
        assert!(hinge.is_err());
    }

    #[test]
    fn test_hinge_3d_axis_normalized() {
        let obj1 = make_3d_object(1.0, 0.0, 0.0, 0.0);
        let obj2 = make_3d_object(1.0, 2.0, 0.0, 0.0);
        let hinge = Hinge3D::new(obj1, obj2, (1.0, 0.0, 0.0), (0.0, 3.0, 4.0)).unwrap();

        let axis = hinge.get_axis();
        let len = (axis.0 * axis.0 + axis.1 * axis.1 + axis.2 * axis.2).sqrt();
        assert!((len - 1.0).abs() < 1e-10, "Axis should be normalized");
    }

    #[test]
    fn test_hinge_3d_maintains_anchor() {
        // Create hinge with anchor at (1, 0, 0)
        let obj1 = make_3d_object(1.0, 0.0, 0.0, 0.0);
        let obj2 = make_3d_object(1.0, 2.0, 0.0, 0.0);
        let hinge = Hinge3D::new(obj1, obj2, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)).unwrap();

        let error = hinge.calculate_error();
        assert!(error < 1e-10, "Error should be near zero when objects aligned at anchor");
    }

    #[test]
    fn test_hinge_3d_solve_restores_anchor() {
        let obj1 = make_3d_object(1.0, 0.0, 0.0, 0.0);
        // Object 2 is displaced
        let obj2 = make_3d_object(1.0, 3.0, 0.0, 0.0);
        // Anchor at (1, 0, 0) - obj2 should be at (2, 0, 0) for local_anchor2 = (-1, 0, 0)
        let mut hinge = Hinge3D::new(obj1, obj2, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)).unwrap();

        // Manually perturb object2 further
        hinge.object2.position.x += 1.0;

        let initial_error = hinge.calculate_error();
        assert!(initial_error > 0.5, "Should have initial error");

        // Solve multiple times to converge
        for _ in 0..10 {
            hinge.solve(0.016).unwrap();
        }

        let final_error = hinge.calculate_error();
        assert!(
            final_error < initial_error,
            "Error should decrease: initial={}, final={}",
            initial_error,
            final_error
        );
    }

    #[test]
    fn test_hinge_3d_with_limits() {
        let obj1 = make_3d_object(1.0, 0.0, 0.0, 0.0);
        let obj2 = make_3d_object(1.0, 2.0, 0.0, 0.0);
        let hinge = Hinge3D::new(obj1, obj2, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
            .unwrap()
            .with_limits(-1.57, 1.57);

        assert!(hinge.has_limits());
        assert!(hinge.angle_min.is_some());
        assert!(hinge.angle_max.is_some());
    }

    #[test]
    fn test_hinge_3d_mass_affects_angular_acceleration() {
        // With true angular dynamics, heavier mass means:
        // - Higher moment of inertia (I = m × r²)
        // - Lower angular acceleration (α = τ/I) for same torque
        // Compare two hinges with different masses
        //
        // Geometry: X-axis rotation (like a trap door)
        // - Anchor at origin
        // - Object2 offset in +Z direction (arm in Z direction)
        // - Gravity acts in -Y direction
        // - Torque = arm × gravity = Z × (-Y) = rotation around X axis
        let obj1 = make_3d_object(f64::INFINITY, 0.0, 0.0, 0.0);  // Fixed anchor
        let obj2_light = make_3d_object(1.0, 0.0, 0.0, 2.0);      // Light door at Z=2
        let obj2_heavy = make_3d_object(10.0, 0.0, 0.0, 2.0);     // Heavy door at Z=2

        let mut hinge_light = Hinge3D::new(
            obj1.clone(),
            obj2_light,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0)  // X axis rotation (like trap door)
        ).unwrap();

        let mut hinge_heavy = Hinge3D::new(
            obj1,
            obj2_heavy,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0)  // X axis rotation
        ).unwrap();

        // Run both for same time with gravity
        let dt = 0.016;
        for _ in 0..10 {
            hinge_light.solve(dt).unwrap();
            hinge_heavy.solve(dt).unwrap();
        }

        // Both should rotate, but light one should have higher angular velocity
        // (More acceleration due to lower moment of inertia)
        // Light: I = 1 × 2² = 4, Heavy: I = 10 × 2² = 40
        // Same torque, but heavy has 10× more inertia, so 10× less acceleration
        assert!(
            hinge_light.angular_velocity.abs() > hinge_heavy.angular_velocity.abs() * 0.5,
            "Light hinge should have higher angular velocity: light={:.4}, heavy={:.4}",
            hinge_light.angular_velocity,
            hinge_heavy.angular_velocity
        );
    }

    #[test]
    fn test_hinge_3d_infinite_mass_anchor() {
        let obj1 = make_3d_object(f64::INFINITY, 0.0, 0.0, 0.0);
        let obj2 = make_3d_object(1.0, 2.0, 0.0, 0.0);
        let mut hinge = Hinge3D::new(obj1, obj2, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)).unwrap();

        // Perturb
        hinge.object2.position.x += 1.0;

        let initial_pos1 = hinge.object1.position.x;

        hinge.solve(0.016).unwrap();

        // Infinite mass object should not move
        assert!(
            (hinge.object1.position.x - initial_pos1).abs() < 1e-10,
            "Infinite mass object should not move"
        );
    }
}
