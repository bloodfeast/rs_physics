//! Fixed/weld constraints for physics simulations.
//!
//! This module provides fixed constraints that maintain a constant offset
//! between two objects, effectively "welding" them together.

use crate::models::{ObjectIn2D, ObjectIn3D};
use crate::utils::PhysicsError;
use super::solver::{Constraint2D, Constraint3D};

/// A fixed (weld) constraint between two 2D objects.
///
/// Maintains a constant relative offset between two objects by applying
/// position and velocity corrections each timestep.
///
/// # Examples
///
/// ```rust,ignore
/// use rs_physics::constraints::Fixed2D;
/// use rs_physics::models::ObjectIn2D;
///
/// let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Default::default());
/// let obj2 = ObjectIn2D::with_shape(1.0, (2.0, 1.0), Default::default());
/// let fixed = Fixed2D::new(obj1, obj2).expect("Valid constraint");
/// ```
pub struct Fixed2D {
    /// First object connected by the constraint
    pub object1: ObjectIn2D,
    /// Second object connected by the constraint
    pub object2: ObjectIn2D,
    /// Target X offset (object2.x - object1.x)
    pub target_offset_x: f64,
    /// Target Y offset (object2.y - object1.y)
    pub target_offset_y: f64,
    /// Baumgarte stabilization factor (0.0 to 1.0, default 0.2)
    pub baumgarte: f64,
    /// Accumulated impulse for warm starting
    pub lambda: f64,
}

impl Fixed2D {
    /// Creates a new Fixed2D constraint between two objects.
    ///
    /// The current offset between the objects becomes the target offset
    /// that will be maintained.
    ///
    /// # Arguments
    ///
    /// * `object1` - First object (anchor)
    /// * `object2` - Second object (attached)
    ///
    /// # Returns
    ///
    /// * `Ok(Fixed2D)` - Valid fixed constraint
    /// * `Err(PhysicsError)` - If objects have invalid configuration
    pub fn new(object1: ObjectIn2D, object2: ObjectIn2D) -> Result<Self, PhysicsError> {
        let target_offset_x = object2.position.x - object1.position.x;
        let target_offset_y = object2.position.y - object1.position.y;

        Ok(Self {
            object1,
            object2,
            target_offset_x,
            target_offset_y,
            baumgarte: 0.2,
            lambda: 0.0,
        })
    }

    /// Creates a new Fixed2D constraint with a specific offset.
    ///
    /// # Arguments
    ///
    /// * `object1` - First object (anchor)
    /// * `object2` - Second object (attached)
    /// * `offset` - Target offset (x, y) from object1 to object2
    pub fn with_offset(
        object1: ObjectIn2D,
        object2: ObjectIn2D,
        offset: (f64, f64),
    ) -> Result<Self, PhysicsError> {
        Ok(Self {
            object1,
            object2,
            target_offset_x: offset.0,
            target_offset_y: offset.1,
            baumgarte: 0.2,
            lambda: 0.0,
        })
    }

    /// Sets the Baumgarte stabilization factor.
    pub fn with_baumgarte(mut self, factor: f64) -> Self {
        self.baumgarte = factor;
        self
    }

    /// Solves the fixed constraint for one iteration.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    pub fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        // Calculate current offset
        let current_offset_x = self.object2.position.x - self.object1.position.x;
        let current_offset_y = self.object2.position.y - self.object1.position.y;

        // Calculate error (how far from target offset)
        let error_x = current_offset_x - self.target_offset_x;
        let error_y = current_offset_y - self.target_offset_y;

        let error_magnitude = (error_x * error_x + error_y * error_y).sqrt();

        if error_magnitude < 1e-10 {
            // Already at target offset
            return Ok(());
        }

        // Calculate inverse masses for mass-weighted corrections
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
            // Both objects have infinite mass, can't correct
            return Ok(());
        }

        // Normalize error direction
        let nx = error_x / error_magnitude;
        let ny = error_y / error_magnitude;

        // Baumgarte stabilization: bias velocity toward zero error
        let bias = self.baumgarte * error_magnitude / dt;

        // Calculate relative velocity along error direction
        let rel_vx = self.object2.velocity.x - self.object1.velocity.x;
        let rel_vy = self.object2.velocity.y - self.object1.velocity.y;
        let relative_velocity = rel_vx * nx + rel_vy * ny;

        // Impulse magnitude
        let lambda = -(relative_velocity + bias) / total_inv_mass;

        // Clamp impulse for stability
        let max_impulse = 0.1 / dt;
        let lambda = lambda.clamp(-max_impulse, max_impulse);

        // Apply velocity corrections (mass-weighted)
        self.object1.velocity.x -= lambda * inv_mass1 * nx;
        self.object1.velocity.y -= lambda * inv_mass1 * ny;
        self.object2.velocity.x += lambda * inv_mass2 * nx;
        self.object2.velocity.y += lambda * inv_mass2 * ny;

        // Position correction (mass-weighted)
        let max_correction = 0.1;
        let position_correction = error_magnitude.clamp(-max_correction * 2.0, max_correction * 2.0);

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
    /// The distance between current offset and target offset
    pub fn calculate_error(&self) -> f64 {
        let current_offset_x = self.object2.position.x - self.object1.position.x;
        let current_offset_y = self.object2.position.y - self.object1.position.y;

        let error_x = current_offset_x - self.target_offset_x;
        let error_y = current_offset_y - self.target_offset_y;

        (error_x * error_x + error_y * error_y).sqrt()
    }
}

impl Constraint2D for Fixed2D {
    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        Fixed2D::solve(self, dt)
    }

    fn calculate_error(&self) -> f64 {
        Fixed2D::calculate_error(self)
    }

    fn get_lambda(&self) -> f64 {
        self.lambda
    }

    fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda;
    }
}

/// A fixed (weld) constraint between two 3D objects.
///
/// Maintains a constant relative offset between two objects by applying
/// position and velocity corrections each timestep.
///
/// # Examples
///
/// ```rust,ignore
/// use rs_physics::constraints::Fixed3D;
/// use rs_physics::models::ObjectIn3D;
///
/// let obj1 = ObjectIn3D::with_shape(1.0, (0.0, 0.0, 0.0), Default::default());
/// let obj2 = ObjectIn3D::with_shape(1.0, (2.0, 1.0, 0.5), Default::default());
/// let fixed = Fixed3D::new(obj1, obj2).expect("Valid constraint");
/// ```
pub struct Fixed3D {
    /// First object connected by the constraint
    pub object1: ObjectIn3D,
    /// Second object connected by the constraint
    pub object2: ObjectIn3D,
    /// Target X offset (object2.x - object1.x)
    pub target_offset_x: f64,
    /// Target Y offset (object2.y - object1.y)
    pub target_offset_y: f64,
    /// Target Z offset (object2.z - object1.z)
    pub target_offset_z: f64,
    /// Baumgarte stabilization factor (0.0 to 1.0, default 0.2)
    pub baumgarte: f64,
    /// Accumulated impulse for warm starting
    pub lambda: f64,
}

impl Fixed3D {
    /// Creates a new Fixed3D constraint between two objects.
    ///
    /// The current offset between the objects becomes the target offset
    /// that will be maintained.
    ///
    /// # Arguments
    ///
    /// * `object1` - First object (anchor)
    /// * `object2` - Second object (attached)
    ///
    /// # Returns
    ///
    /// * `Ok(Fixed3D)` - Valid fixed constraint
    /// * `Err(PhysicsError)` - If objects have invalid configuration
    pub fn new(object1: ObjectIn3D, object2: ObjectIn3D) -> Result<Self, PhysicsError> {
        let target_offset_x = object2.position.x - object1.position.x;
        let target_offset_y = object2.position.y - object1.position.y;
        let target_offset_z = object2.position.z - object1.position.z;

        Ok(Self {
            object1,
            object2,
            target_offset_x,
            target_offset_y,
            target_offset_z,
            baumgarte: 0.2,
            lambda: 0.0,
        })
    }

    /// Creates a new Fixed3D constraint with a specific offset.
    ///
    /// # Arguments
    ///
    /// * `object1` - First object (anchor)
    /// * `object2` - Second object (attached)
    /// * `offset` - Target offset (x, y, z) from object1 to object2
    pub fn with_offset(
        object1: ObjectIn3D,
        object2: ObjectIn3D,
        offset: (f64, f64, f64),
    ) -> Result<Self, PhysicsError> {
        Ok(Self {
            object1,
            object2,
            target_offset_x: offset.0,
            target_offset_y: offset.1,
            target_offset_z: offset.2,
            baumgarte: 0.2,
            lambda: 0.0,
        })
    }

    /// Sets the Baumgarte stabilization factor.
    pub fn with_baumgarte(mut self, factor: f64) -> Self {
        self.baumgarte = factor;
        self
    }

    /// Solves the fixed constraint for one iteration.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    pub fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        // Calculate current offset
        let current_offset_x = self.object2.position.x - self.object1.position.x;
        let current_offset_y = self.object2.position.y - self.object1.position.y;
        let current_offset_z = self.object2.position.z - self.object1.position.z;

        // Calculate error (how far from target offset)
        let error_x = current_offset_x - self.target_offset_x;
        let error_y = current_offset_y - self.target_offset_y;
        let error_z = current_offset_z - self.target_offset_z;

        let error_magnitude = (error_x * error_x + error_y * error_y + error_z * error_z).sqrt();

        if error_magnitude < 1e-10 {
            // Already at target offset
            return Ok(());
        }

        // Calculate inverse masses for mass-weighted corrections
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
            // Both objects have infinite mass, can't correct
            return Ok(());
        }

        // Normalize error direction
        let nx = error_x / error_magnitude;
        let ny = error_y / error_magnitude;
        let nz = error_z / error_magnitude;

        // Baumgarte stabilization: bias velocity toward zero error
        let bias = self.baumgarte * error_magnitude / dt;

        // Calculate relative velocity along error direction
        let rel_vx = self.object2.velocity.x - self.object1.velocity.x;
        let rel_vy = self.object2.velocity.y - self.object1.velocity.y;
        let rel_vz = self.object2.velocity.z - self.object1.velocity.z;
        let relative_velocity = rel_vx * nx + rel_vy * ny + rel_vz * nz;

        // Impulse magnitude
        let lambda = -(relative_velocity + bias) / total_inv_mass;

        // Clamp impulse for stability
        let max_impulse = 0.1 / dt;
        let lambda = lambda.clamp(-max_impulse, max_impulse);

        // Apply velocity corrections (mass-weighted)
        self.object1.velocity.x -= lambda * inv_mass1 * nx;
        self.object1.velocity.y -= lambda * inv_mass1 * ny;
        self.object1.velocity.z -= lambda * inv_mass1 * nz;
        self.object2.velocity.x += lambda * inv_mass2 * nx;
        self.object2.velocity.y += lambda * inv_mass2 * ny;
        self.object2.velocity.z += lambda * inv_mass2 * nz;

        // Position correction (mass-weighted)
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

    /// Calculates the current constraint error.
    ///
    /// # Returns
    ///
    /// The distance between current offset and target offset
    pub fn calculate_error(&self) -> f64 {
        let current_offset_x = self.object2.position.x - self.object1.position.x;
        let current_offset_y = self.object2.position.y - self.object1.position.y;
        let current_offset_z = self.object2.position.z - self.object1.position.z;

        let error_x = current_offset_x - self.target_offset_x;
        let error_y = current_offset_y - self.target_offset_y;
        let error_z = current_offset_z - self.target_offset_z;

        (error_x * error_x + error_y * error_y + error_z * error_z).sqrt()
    }
}

impl Constraint3D for Fixed3D {
    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        Fixed3D::solve(self, dt)
    }

    fn calculate_error(&self) -> f64 {
        Fixed3D::calculate_error(self)
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
    // Fixed2D Tests
    // ============================================================================

    #[test]
    fn test_fixed_2d_creation() {
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (3.0, 4.0), Shape2DCollider::Circle(1.0));
        let fixed = Fixed2D::new(obj1, obj2);
        assert!(fixed.is_ok());
        let fixed = fixed.unwrap();
        assert!((fixed.target_offset_x - 3.0).abs() < 1e-10);
        assert!((fixed.target_offset_y - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_2d_with_offset() {
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (3.0, 4.0), Shape2DCollider::Circle(1.0));
        let fixed = Fixed2D::with_offset(obj1, obj2, (5.0, 5.0));
        assert!(fixed.is_ok());
        let fixed = fixed.unwrap();
        assert!((fixed.target_offset_x - 5.0).abs() < 1e-10);
        assert!((fixed.target_offset_y - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_2d_maintains_offset() {
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (3.0, 4.0), Shape2DCollider::Circle(1.0));
        let fixed = Fixed2D::new(obj1, obj2).unwrap();

        let error = fixed.calculate_error();
        assert!(error < 1e-10, "Error should be near zero at creation");
    }

    #[test]
    fn test_fixed_2d_solve_restores_offset() {
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        // Object 2 is displaced from its target position
        let obj2 = ObjectIn2D::with_shape(1.0, (4.0, 5.0), Shape2DCollider::Circle(1.0));
        // But we want a 3,4 offset
        let mut fixed = Fixed2D::with_offset(obj1, obj2, (3.0, 4.0)).unwrap();

        let initial_error = fixed.calculate_error();
        assert!(initial_error > 1.0, "Should have significant initial error");

        // Solve multiple times to converge
        for _ in 0..10 {
            fixed.solve(0.016).unwrap();
        }

        let final_error = fixed.calculate_error();
        assert!(
            final_error < initial_error,
            "Error should decrease: initial={}, final={}",
            initial_error,
            final_error
        );
    }

    #[test]
    fn test_fixed_2d_mass_weighted() {
        // With unequal masses, lighter object should move more
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(4.0, (4.0, 0.0), Shape2DCollider::Circle(1.0));
        let mut fixed = Fixed2D::with_offset(obj1, obj2, (3.0, 0.0)).unwrap();

        let initial_pos1 = fixed.object1.position.x;
        let initial_pos2 = fixed.object2.position.x;

        fixed.solve(0.016).unwrap();

        let delta1 = (fixed.object1.position.x - initial_pos1).abs();
        let delta2 = (fixed.object2.position.x - initial_pos2).abs();

        // Lighter object should move more
        assert!(
            delta1 > delta2,
            "Lighter object should move more: delta1={}, delta2={}",
            delta1,
            delta2
        );
    }

    #[test]
    fn test_fixed_2d_zero_offset() {
        // Objects at same position with zero offset
        let obj1 = ObjectIn2D::with_shape(1.0, (5.0, 5.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (5.0, 5.0), Shape2DCollider::Circle(1.0));
        let mut fixed = Fixed2D::new(obj1, obj2).unwrap();

        assert!(fixed.calculate_error() < 1e-10);
        fixed.solve(0.016).unwrap();
        assert!(fixed.calculate_error() < 1e-10);
    }

    // ============================================================================
    // Fixed3D Tests
    // ============================================================================

    fn make_3d_object(mass: f64, x: f64, y: f64, z: f64) -> ObjectIn3D {
        use crate::models::{Axis3D, Velocity3D};
        ObjectIn3D {
            mass,
            velocity: Velocity3D { x: 0.0, y: 0.0, z: 0.0 },
            position: Axis3D { x, y, z },
            forces: Vec::new(),
        }
    }

    #[test]
    fn test_fixed_3d_creation() {
        let obj1 = make_3d_object(1.0, 0.0, 0.0, 0.0);
        let obj2 = make_3d_object(1.0, 2.0, 3.0, 4.0);
        let fixed = Fixed3D::new(obj1, obj2);
        assert!(fixed.is_ok());
        let fixed = fixed.unwrap();
        assert!((fixed.target_offset_x - 2.0).abs() < 1e-10);
        assert!((fixed.target_offset_y - 3.0).abs() < 1e-10);
        assert!((fixed.target_offset_z - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_3d_with_offset() {
        let obj1 = make_3d_object(1.0, 0.0, 0.0, 0.0);
        let obj2 = make_3d_object(1.0, 2.0, 3.0, 4.0);
        let fixed = Fixed3D::with_offset(obj1, obj2, (1.0, 1.0, 1.0));
        assert!(fixed.is_ok());
        let fixed = fixed.unwrap();
        assert!((fixed.target_offset_x - 1.0).abs() < 1e-10);
        assert!((fixed.target_offset_y - 1.0).abs() < 1e-10);
        assert!((fixed.target_offset_z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_3d_maintains_offset() {
        let obj1 = make_3d_object(1.0, 0.0, 0.0, 0.0);
        let obj2 = make_3d_object(1.0, 2.0, 3.0, 4.0);
        let fixed = Fixed3D::new(obj1, obj2).unwrap();

        let error = fixed.calculate_error();
        assert!(error < 1e-10, "Error should be near zero at creation");
    }

    #[test]
    fn test_fixed_3d_solve_restores_offset() {
        let obj1 = make_3d_object(1.0, 0.0, 0.0, 0.0);
        // Object 2 is displaced from its target position
        let obj2 = make_3d_object(1.0, 3.0, 4.0, 5.0);
        // But we want a 2,3,4 offset
        let mut fixed = Fixed3D::with_offset(obj1, obj2, (2.0, 3.0, 4.0)).unwrap();

        let initial_error = fixed.calculate_error();
        assert!(initial_error > 1.0, "Should have significant initial error");

        // Solve multiple times to converge
        for _ in 0..10 {
            fixed.solve(0.016).unwrap();
        }

        let final_error = fixed.calculate_error();
        assert!(
            final_error < initial_error,
            "Error should decrease: initial={}, final={}",
            initial_error,
            final_error
        );
    }

    #[test]
    fn test_fixed_3d_mass_weighted() {
        // With unequal masses, lighter object should move more
        let obj1 = make_3d_object(1.0, 0.0, 0.0, 0.0);
        let obj2 = make_3d_object(4.0, 4.0, 0.0, 0.0);
        let mut fixed = Fixed3D::with_offset(obj1, obj2, (3.0, 0.0, 0.0)).unwrap();

        let initial_pos1 = fixed.object1.position.x;
        let initial_pos2 = fixed.object2.position.x;

        fixed.solve(0.016).unwrap();

        let delta1 = (fixed.object1.position.x - initial_pos1).abs();
        let delta2 = (fixed.object2.position.x - initial_pos2).abs();

        // Lighter object should move more
        assert!(
            delta1 > delta2,
            "Lighter object should move more: delta1={}, delta2={}",
            delta1,
            delta2
        );
    }

    #[test]
    fn test_fixed_3d_diagonal() {
        // Test with diagonal offset
        let obj1 = make_3d_object(1.0, 0.0, 0.0, 0.0);
        let obj2 = make_3d_object(1.0, 1.0, 1.0, 1.0);
        let mut fixed = Fixed3D::new(obj1, obj2).unwrap();

        // Initially at target offset
        assert!(fixed.calculate_error() < 1e-10);

        // Perturb object2
        fixed.object2.position.x += 0.5;
        fixed.object2.position.y += 0.5;
        fixed.object2.position.z += 0.5;

        let initial_error = fixed.calculate_error();
        assert!(initial_error > 0.5, "Should have error after perturbation");

        // Solve
        for _ in 0..10 {
            fixed.solve(0.016).unwrap();
        }

        let final_error = fixed.calculate_error();
        assert!(
            final_error < initial_error,
            "Error should decrease after solving"
        );
    }
}
