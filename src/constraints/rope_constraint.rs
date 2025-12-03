//! Rope/cable constraints for physics simulations.
//!
//! This module provides rope constraints that only resist stretching.
//! Unlike joint constraints that maintain exact distances, ropes allow
//! objects to be closer than the max length (slack) but not further.

use crate::utils::PhysicsError;
use crate::models::{ObjectIn2D, ObjectIn3D};
use super::solver::{Constraint2D, Constraint3D};

/// A rope constraint between two 2D objects.
///
/// Ropes only apply corrections when stretched beyond their maximum length.
/// When the distance is less than max_length, no force is applied (rope is slack).
///
/// # Examples
///
/// ```rust,ignore
/// use rs_physics::constraints::Rope2D;
/// use rs_physics::models::ObjectIn2D;
///
/// let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Default::default());
/// let obj2 = ObjectIn2D::with_shape(1.0, (5.0, 0.0), Default::default());
/// let rope = Rope2D::new(obj1, obj2, 5.0).expect("Valid rope");
/// ```
pub struct Rope2D {
    /// First object connected by the rope
    pub object1: ObjectIn2D,
    /// Second object connected by the rope
    pub object2: ObjectIn2D,
    /// Maximum length before constraint activates (meters)
    pub max_length: f64,
    /// Baumgarte stabilization factor (0.0 to 1.0, default 0.2)
    pub baumgarte: f64,
    /// Accumulated impulse for warm starting
    pub lambda: f64,
}

impl Rope2D {
    /// Creates a new Rope2D constraint between two objects.
    ///
    /// # Arguments
    ///
    /// * `object1` - First object connected by the rope
    /// * `object2` - Second object connected by the rope
    /// * `max_length` - Maximum length before constraint activates (must be > 0)
    ///
    /// # Returns
    ///
    /// * `Ok(Rope2D)` - Valid rope constraint
    /// * `Err(PhysicsError)` - If max_length is not positive
    pub fn new(object1: ObjectIn2D, object2: ObjectIn2D, max_length: f64) -> Result<Self, PhysicsError> {
        if max_length <= 0.0 {
            return Err(PhysicsError::InvalidDistance);
        }
        Ok(Self {
            object1,
            object2,
            max_length,
            baumgarte: 0.2,
            lambda: 0.0,
        })
    }

    /// Sets the Baumgarte stabilization factor.
    pub fn with_baumgarte(mut self, factor: f64) -> Self {
        self.baumgarte = factor;
        self
    }

    /// Returns true if the rope is taut (stretched to or beyond max_length).
    pub fn is_taut(&self) -> bool {
        let dx = self.object2.position.x - self.object1.position.x;
        let dy = self.object2.position.y - self.object1.position.y;
        let current_length = (dx * dx + dy * dy).sqrt();
        current_length >= self.max_length
    }

    /// Solves the rope constraint for one iteration.
    ///
    /// Only applies corrections if the rope is taut.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    pub fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        let dx = self.object2.position.x - self.object1.position.x;
        let dy = self.object2.position.y - self.object1.position.y;
        let current_length = (dx * dx + dy * dy).sqrt();

        // Only apply constraint when rope is taut (stretched beyond max_length)
        if current_length <= self.max_length {
            return Ok(());
        }

        if current_length < 1e-10 {
            return Ok(());
        }

        let error = current_length - self.max_length;

        // Calculate inverse masses for mass-weighted corrections
        let inv_mass1 = if self.object1.mass.is_infinite() { 0.0 } else { 1.0 / self.object1.mass };
        let inv_mass2 = if self.object2.mass.is_infinite() { 0.0 } else { 1.0 / self.object2.mass };
        let total_inv_mass = inv_mass1 + inv_mass2;

        if total_inv_mass < 1e-10 {
            return Ok(());
        }

        // Normalize direction
        let nx = dx / current_length;
        let ny = dy / current_length;

        // Baumgarte stabilization
        let bias = self.baumgarte * error / dt;

        // Calculate relative velocity along constraint direction
        let rel_vx = self.object2.velocity.x - self.object1.velocity.x;
        let rel_vy = self.object2.velocity.y - self.object1.velocity.y;
        let relative_velocity = rel_vx * nx + rel_vy * ny;

        // Impulse magnitude (only allow pulling, not pushing)
        let lambda = -(relative_velocity + bias) / total_inv_mass;
        let lambda = lambda.max(0.0); // Rope can only pull, not push

        // Clamp impulse for stability
        let max_impulse = 0.1 / dt;
        let lambda = lambda.min(max_impulse);

        // Apply velocity corrections (mass-weighted)
        self.object1.velocity.x -= lambda * inv_mass1 * nx;
        self.object1.velocity.y -= lambda * inv_mass1 * ny;
        self.object2.velocity.x += lambda * inv_mass2 * nx;
        self.object2.velocity.y += lambda * inv_mass2 * ny;

        // Position correction (mass-weighted)
        let max_correction = 0.1;
        let position_correction = error.min(max_correction * 2.0);

        self.object1.position.x += position_correction * (inv_mass1 / total_inv_mass) * nx;
        self.object1.position.y += position_correction * (inv_mass1 / total_inv_mass) * ny;
        self.object2.position.x -= position_correction * (inv_mass2 / total_inv_mass) * nx;
        self.object2.position.y -= position_correction * (inv_mass2 / total_inv_mass) * ny;

        Ok(())
    }

    /// Calculates the current constraint error.
    ///
    /// # Returns
    ///
    /// 0 if rope is slack, positive value if rope is taut
    pub fn calculate_error(&self) -> f64 {
        let dx = self.object2.position.x - self.object1.position.x;
        let dy = self.object2.position.y - self.object1.position.y;
        let current_length = (dx * dx + dy * dy).sqrt();
        (current_length - self.max_length).max(0.0)
    }
}

impl Constraint2D for Rope2D {
    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        Rope2D::solve(self, dt)
    }

    fn calculate_error(&self) -> f64 {
        Rope2D::calculate_error(self)
    }

    fn get_lambda(&self) -> f64 {
        self.lambda
    }

    fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda;
    }
}

/// A rope constraint between two 3D objects.
///
/// Ropes only apply corrections when stretched beyond their maximum length.
/// When the distance is less than max_length, no force is applied (rope is slack).
///
/// # Examples
///
/// ```rust,ignore
/// use rs_physics::constraints::Rope3D;
/// use rs_physics::models::ObjectIn3D;
///
/// let mut obj1 = ObjectIn3D::default();
/// obj1.position = Axis3D { x: 0.0, y: 0.0, z: 0.0 };
/// let mut obj2 = ObjectIn3D::default();
/// obj2.position = Axis3D { x: 5.0, y: 0.0, z: 0.0 };
/// let rope = Rope3D::new(obj1, obj2, 5.0).expect("Valid rope");
/// ```
pub struct Rope3D {
    /// First object connected by the rope
    pub object1: ObjectIn3D,
    /// Second object connected by the rope
    pub object2: ObjectIn3D,
    /// Maximum length before constraint activates (meters)
    pub max_length: f64,
    /// Baumgarte stabilization factor (0.0 to 1.0, default 0.2)
    pub baumgarte: f64,
    /// Accumulated impulse for warm starting
    pub lambda: f64,
}

impl Rope3D {
    /// Creates a new Rope3D constraint between two objects.
    ///
    /// # Arguments
    ///
    /// * `object1` - First object connected by the rope
    /// * `object2` - Second object connected by the rope
    /// * `max_length` - Maximum length before constraint activates (must be > 0)
    ///
    /// # Returns
    ///
    /// * `Ok(Rope3D)` - Valid rope constraint
    /// * `Err(PhysicsError)` - If max_length is not positive
    pub fn new(object1: ObjectIn3D, object2: ObjectIn3D, max_length: f64) -> Result<Self, PhysicsError> {
        if max_length <= 0.0 {
            return Err(PhysicsError::InvalidDistance);
        }
        Ok(Self {
            object1,
            object2,
            max_length,
            baumgarte: 0.2,
            lambda: 0.0,
        })
    }

    /// Sets the Baumgarte stabilization factor.
    pub fn with_baumgarte(mut self, factor: f64) -> Self {
        self.baumgarte = factor;
        self
    }

    /// Returns true if the rope is taut (stretched to or beyond max_length).
    pub fn is_taut(&self) -> bool {
        let dx = self.object2.position.x - self.object1.position.x;
        let dy = self.object2.position.y - self.object1.position.y;
        let dz = self.object2.position.z - self.object1.position.z;
        let current_length = (dx * dx + dy * dy + dz * dz).sqrt();
        current_length >= self.max_length
    }

    /// Solves the rope constraint for one iteration.
    ///
    /// Only applies corrections if the rope is taut.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    pub fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        let dx = self.object2.position.x - self.object1.position.x;
        let dy = self.object2.position.y - self.object1.position.y;
        let dz = self.object2.position.z - self.object1.position.z;
        let current_length = (dx * dx + dy * dy + dz * dz).sqrt();

        // Only apply constraint when rope is taut (stretched beyond max_length)
        if current_length <= self.max_length {
            return Ok(());
        }

        if current_length < 1e-10 {
            return Ok(());
        }

        let error = current_length - self.max_length;

        // Calculate inverse masses for mass-weighted corrections
        let inv_mass1 = if self.object1.mass.is_infinite() { 0.0 } else { 1.0 / self.object1.mass };
        let inv_mass2 = if self.object2.mass.is_infinite() { 0.0 } else { 1.0 / self.object2.mass };
        let total_inv_mass = inv_mass1 + inv_mass2;

        if total_inv_mass < 1e-10 {
            return Ok(());
        }

        // Normalize direction
        let nx = dx / current_length;
        let ny = dy / current_length;
        let nz = dz / current_length;

        // Baumgarte stabilization
        let bias = self.baumgarte * error / dt;

        // Calculate relative velocity along constraint direction
        let rel_vx = self.object2.velocity.x - self.object1.velocity.x;
        let rel_vy = self.object2.velocity.y - self.object1.velocity.y;
        let rel_vz = self.object2.velocity.z - self.object1.velocity.z;
        let relative_velocity = rel_vx * nx + rel_vy * ny + rel_vz * nz;

        // Impulse magnitude (only allow pulling, not pushing)
        let lambda = -(relative_velocity + bias) / total_inv_mass;
        let lambda = lambda.max(0.0); // Rope can only pull, not push

        // Clamp impulse for stability
        let max_impulse = 0.1 / dt;
        let lambda = lambda.min(max_impulse);

        // Apply velocity corrections (mass-weighted)
        self.object1.velocity.x -= lambda * inv_mass1 * nx;
        self.object1.velocity.y -= lambda * inv_mass1 * ny;
        self.object1.velocity.z -= lambda * inv_mass1 * nz;
        self.object2.velocity.x += lambda * inv_mass2 * nx;
        self.object2.velocity.y += lambda * inv_mass2 * ny;
        self.object2.velocity.z += lambda * inv_mass2 * nz;

        // Position correction (mass-weighted)
        let max_correction = 0.1;
        let position_correction = error.min(max_correction * 2.0);

        self.object1.position.x += position_correction * (inv_mass1 / total_inv_mass) * nx;
        self.object1.position.y += position_correction * (inv_mass1 / total_inv_mass) * ny;
        self.object1.position.z += position_correction * (inv_mass1 / total_inv_mass) * nz;
        self.object2.position.x -= position_correction * (inv_mass2 / total_inv_mass) * nx;
        self.object2.position.y -= position_correction * (inv_mass2 / total_inv_mass) * ny;
        self.object2.position.z -= position_correction * (inv_mass2 / total_inv_mass) * nz;

        Ok(())
    }

    /// Calculates the current constraint error.
    ///
    /// # Returns
    ///
    /// 0 if rope is slack, positive value if rope is taut
    pub fn calculate_error(&self) -> f64 {
        let dx = self.object2.position.x - self.object1.position.x;
        let dy = self.object2.position.y - self.object1.position.y;
        let dz = self.object2.position.z - self.object1.position.z;
        let current_length = (dx * dx + dy * dy + dz * dz).sqrt();
        (current_length - self.max_length).max(0.0)
    }
}

impl Constraint3D for Rope3D {
    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        Rope3D::solve(self, dt)
    }

    fn calculate_error(&self) -> f64 {
        Rope3D::calculate_error(self)
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
    use crate::models::Shape2DCollider;

    // ============================================================================
    // 2D Rope Tests
    // ============================================================================

    #[test]
    fn test_rope_2d_creation() {
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (3.0, 0.0), Shape2DCollider::Circle(1.0));
        let rope = Rope2D::new(obj1, obj2, 5.0);
        assert!(rope.is_ok());
    }

    #[test]
    fn test_rope_2d_invalid_length() {
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (3.0, 0.0), Shape2DCollider::Circle(1.0));
        let rope = Rope2D::new(obj1, obj2, 0.0);
        assert!(rope.is_err());

        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (3.0, 0.0), Shape2DCollider::Circle(1.0));
        let rope = Rope2D::new(obj1, obj2, -1.0);
        assert!(rope.is_err());
    }

    #[test]
    fn test_rope_2d_slack_no_correction() {
        // Rope is slack (objects closer than max_length)
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (3.0, 0.0), Shape2DCollider::Circle(1.0));
        let mut rope = Rope2D::new(obj1, obj2, 5.0).unwrap();

        let initial_pos1 = rope.object1.position.x;
        let initial_pos2 = rope.object2.position.x;

        rope.solve(0.1).unwrap();

        // No correction should be applied when slack
        assert!((rope.object1.position.x - initial_pos1).abs() < 1e-10);
        assert!((rope.object2.position.x - initial_pos2).abs() < 1e-10);
        assert!(!rope.is_taut());
    }

    #[test]
    fn test_rope_2d_taut_correction() {
        // Rope is taut (objects further than max_length)
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (7.0, 0.0), Shape2DCollider::Circle(1.0));
        let mut rope = Rope2D::new(obj1, obj2, 5.0).unwrap();

        assert!(rope.is_taut());
        let initial_error = rope.calculate_error();

        rope.solve(0.1).unwrap();

        let final_error = rope.calculate_error();
        assert!(final_error < initial_error, "Error should decrease after solving");
    }

    #[test]
    fn test_rope_2d_at_max_length() {
        // Rope exactly at max_length (should be considered taut)
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (5.0, 0.0), Shape2DCollider::Circle(1.0));
        let rope = Rope2D::new(obj1, obj2, 5.0).unwrap();

        assert!(rope.is_taut());
        assert!(rope.calculate_error() < 1e-10, "Error should be near zero at exact max_length");
    }

    #[test]
    fn test_rope_2d_diagonal() {
        // Rope stretched diagonally
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (6.0, 8.0), Shape2DCollider::Circle(1.0)); // Distance = 10
        let mut rope = Rope2D::new(obj1, obj2, 5.0).unwrap();

        assert!(rope.is_taut());
        let initial_error = rope.calculate_error();
        assert!((initial_error - 5.0).abs() < 1e-6, "Error should be 5.0 (10 - 5)");

        rope.solve(0.1).unwrap();

        let final_error = rope.calculate_error();
        assert!(final_error < initial_error, "Error should decrease after solving");
    }

    // ============================================================================
    // 3D Rope Tests
    // ============================================================================

    fn make_3d_object_at(x: f64, y: f64, z: f64) -> ObjectIn3D {
        use crate::models::{Axis3D, Velocity3D};
        ObjectIn3D {
            mass: 1.0,
            velocity: Velocity3D { x: 0.0, y: 0.0, z: 0.0 },
            position: Axis3D { x, y, z },
            forces: Vec::new(),
        }
    }

    #[test]
    fn test_rope_3d_creation() {
        let obj1 = make_3d_object_at(0.0, 0.0, 0.0);
        let obj2 = make_3d_object_at(3.0, 0.0, 0.0);
        let rope = Rope3D::new(obj1, obj2, 5.0);
        assert!(rope.is_ok());
    }

    #[test]
    fn test_rope_3d_slack_no_correction() {
        let obj1 = make_3d_object_at(0.0, 0.0, 0.0);
        let obj2 = make_3d_object_at(3.0, 0.0, 0.0);
        let mut rope = Rope3D::new(obj1, obj2, 5.0).unwrap();

        let initial_pos1 = rope.object1.position.x;
        let initial_pos2 = rope.object2.position.x;

        rope.solve(0.1).unwrap();

        // No correction should be applied when slack
        assert!((rope.object1.position.x - initial_pos1).abs() < 1e-10);
        assert!((rope.object2.position.x - initial_pos2).abs() < 1e-10);
        assert!(!rope.is_taut());
    }

    #[test]
    fn test_rope_3d_taut_correction() {
        let obj1 = make_3d_object_at(0.0, 0.0, 0.0);
        let obj2 = make_3d_object_at(7.0, 0.0, 0.0);
        let mut rope = Rope3D::new(obj1, obj2, 5.0).unwrap();

        assert!(rope.is_taut());
        let initial_error = rope.calculate_error();

        rope.solve(0.1).unwrap();

        let final_error = rope.calculate_error();
        assert!(final_error < initial_error, "Error should decrease after solving");
    }

    #[test]
    fn test_rope_3d_diagonal_xyz() {
        // Rope stretched diagonally in 3D space
        let obj1 = make_3d_object_at(0.0, 0.0, 0.0);
        let obj2 = make_3d_object_at(6.0, 6.0, 3.0); // Distance = 9
        let mut rope = Rope3D::new(obj1, obj2, 5.0).unwrap();

        assert!(rope.is_taut());
        let initial_error = rope.calculate_error();
        assert!(initial_error > 3.0, "Error should be > 3.0");

        rope.solve(0.1).unwrap();

        let final_error = rope.calculate_error();
        assert!(final_error < initial_error, "Error should decrease after solving");
    }
}
