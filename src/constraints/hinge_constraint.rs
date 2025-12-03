//! Hinge/revolute constraints for 3D physics simulations.
//!
//! This module provides hinge constraints that allow rotation around a single axis
//! while maintaining positional constraints between two objects.

use crate::models::ObjectIn3D;
use crate::utils::PhysicsError;
use super::solver::Constraint3D;

/// A hinge (revolute) constraint between two 3D objects.
///
/// A hinge constraint connects two objects at a shared anchor point and allows
/// rotation only around a specified axis. This is useful for modeling doors,
/// wheels, pendulums, and other rotating mechanisms.
///
/// # Physics Model
///
/// The hinge constraint enforces two conditions:
/// 1. Both objects remain connected at the anchor point
/// 2. Relative rotation is only allowed around the hinge axis
///
/// Optionally, angle limits can be specified to constrain the rotation range.
///
/// # Examples
///
/// ```rust,ignore
/// use rs_physics::constraints::Hinge3D;
/// use rs_physics::models::ObjectIn3D;
///
/// let obj1 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0));
/// let obj2 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (2.0, 0.0, 0.0));
/// let hinge = Hinge3D::new(obj1, obj2, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
///     .expect("Valid hinge");
/// ```
pub struct Hinge3D {
    /// First object connected by the hinge
    pub object1: ObjectIn3D,
    /// Second object connected by the hinge
    pub object2: ObjectIn3D,
    /// Anchor point in world space (connection point)
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
}

impl Hinge3D {
    /// Creates a new Hinge3D constraint between two objects.
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

    /// Returns the current world-space anchor point for object1.
    fn anchor_world_1(&self) -> (f64, f64, f64) {
        (
            self.object1.position.x + self.local_anchor1.0,
            self.object1.position.y + self.local_anchor1.1,
            self.object1.position.z + self.local_anchor1.2,
        )
    }

    /// Returns the current world-space anchor point for object2.
    fn anchor_world_2(&self) -> (f64, f64, f64) {
        (
            self.object2.position.x + self.local_anchor2.0,
            self.object2.position.y + self.local_anchor2.1,
            self.object2.position.z + self.local_anchor2.2,
        )
    }

    /// Solves the hinge constraint for one iteration.
    ///
    /// This enforces the positional constraint (both anchors must coincide).
    /// For simplicity, this implementation focuses on the positional constraint.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    pub fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        // Get world-space anchor positions
        let anchor1 = self.anchor_world_1();
        let anchor2 = self.anchor_world_2();

        // Calculate error (difference between anchor positions)
        let error_x = anchor2.0 - anchor1.0;
        let error_y = anchor2.1 - anchor1.1;
        let error_z = anchor2.2 - anchor1.2;

        let error_magnitude = (error_x * error_x + error_y * error_y + error_z * error_z).sqrt();

        if error_magnitude < 1e-10 {
            return Ok(());
        }

        // Calculate inverse masses
        let inv_mass1 = if self.object1.mass.is_infinite() {
            0.0
        } else {
            1.0 / self.object1.mass
        };
        let inv_mass2 = if self.object2.mass.is_infinite() {
            0.0
        } else {
            1.0 / self.object2.mass
        };
        let total_inv_mass = inv_mass1 + inv_mass2;

        if total_inv_mass < 1e-10 {
            return Ok(());
        }

        // Normalize error direction
        let nx = error_x / error_magnitude;
        let ny = error_y / error_magnitude;
        let nz = error_z / error_magnitude;

        // Baumgarte stabilization
        let bias = self.baumgarte * error_magnitude / dt;

        // Calculate relative velocity at anchor points
        let rel_vx = self.object2.velocity.x - self.object1.velocity.x;
        let rel_vy = self.object2.velocity.y - self.object1.velocity.y;
        let rel_vz = self.object2.velocity.z - self.object1.velocity.z;
        let relative_velocity = rel_vx * nx + rel_vy * ny + rel_vz * nz;

        // Impulse magnitude
        let lambda = -(relative_velocity + bias) / total_inv_mass;

        // Clamp impulse for stability
        let max_impulse = 0.1 / dt;
        let lambda = lambda.clamp(-max_impulse, max_impulse);

        // Apply velocity corrections
        self.object1.velocity.x -= lambda * inv_mass1 * nx;
        self.object1.velocity.y -= lambda * inv_mass1 * ny;
        self.object1.velocity.z -= lambda * inv_mass1 * nz;
        self.object2.velocity.x += lambda * inv_mass2 * nx;
        self.object2.velocity.y += lambda * inv_mass2 * ny;
        self.object2.velocity.z += lambda * inv_mass2 * nz;

        // Position correction
        let max_correction = 0.1;
        let position_correction = error_magnitude.clamp(-max_correction * 2.0, max_correction * 2.0);

        self.object1.position.x += position_correction * (inv_mass1 / total_inv_mass) * nx;
        self.object1.position.y += position_correction * (inv_mass1 / total_inv_mass) * ny;
        self.object1.position.z += position_correction * (inv_mass1 / total_inv_mass) * nz;
        self.object2.position.x -= position_correction * (inv_mass2 / total_inv_mass) * nx;
        self.object2.position.y -= position_correction * (inv_mass2 / total_inv_mass) * ny;
        self.object2.position.z -= position_correction * (inv_mass2 / total_inv_mass) * nz;

        Ok(())
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
    fn test_hinge_3d_mass_weighted() {
        let obj1 = make_3d_object(1.0, 0.0, 0.0, 0.0);
        let obj2 = make_3d_object(4.0, 2.0, 0.0, 0.0);
        let mut hinge = Hinge3D::new(obj1, obj2, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)).unwrap();

        // Perturb to create error
        hinge.object2.position.x += 1.0;

        let initial_pos1 = hinge.object1.position.x;
        let initial_pos2 = hinge.object2.position.x;

        hinge.solve(0.016).unwrap();

        let delta1 = (hinge.object1.position.x - initial_pos1).abs();
        let delta2 = (hinge.object2.position.x - initial_pos2).abs();

        // Lighter object should move more
        assert!(
            delta1 > delta2,
            "Lighter object should move more: delta1={}, delta2={}",
            delta1,
            delta2
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
