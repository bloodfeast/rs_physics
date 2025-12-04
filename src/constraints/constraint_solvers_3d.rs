//! 3D constraint solvers for physics simulations.
//!
//! This module provides constraint types for 3D physics objects.

use crate::utils::PhysicsError;
use crate::models::ObjectIn3D;
use super::solver::Constraint3D;

// Import for tests
#[cfg(test)]
use crate::models::{Axis3D, Velocity3D};

/// A rigid joint constraint between two 3D objects.
///
/// Maintains a fixed distance between two objects by applying position
/// and velocity corrections each timestep.
///
/// # Examples
///
/// ```rust,ignore
/// use rs_physics::constraints::Joint3D;
/// use rs_physics::models::ObjectIn3D;
///
/// let mut obj1 = ObjectIn3D::default();
/// obj1.position = Axis3D { x: 0.0, y: 0.0, z: 0.0 };
/// let mut obj2 = ObjectIn3D::default();
/// obj2.position = Axis3D { x: 5.0, y: 0.0, z: 0.0 };
/// let joint = Joint3D::new(obj1, obj2, 5.0).expect("Valid joint");
/// ```
pub struct Joint3D {
    /// First object connected by the joint
    pub object1: ObjectIn3D,
    /// Second object connected by the joint
    pub object2: ObjectIn3D,
    /// Target distance to maintain between objects (meters)
    pub constraint_distance: f64,
    /// Baumgarte stabilization factor (0.0 to 1.0, default 0.2)
    pub baumgarte: f64,
    /// Accumulated impulse for warm starting
    pub lambda: f64,
}

impl Joint3D {
    /// Creates a new Joint3D constraint between two objects.
    ///
    /// # Arguments
    ///
    /// * `object1` - First object in the joint
    /// * `object2` - Second object in the joint
    /// * `constraint_distance` - Target distance to maintain (must be >= 0)
    ///
    /// # Returns
    ///
    /// * `Ok(Joint3D)` - Valid joint constraint
    /// * `Err(PhysicsError)` - If constraint_distance is negative
    pub fn new(object1: ObjectIn3D, object2: ObjectIn3D, constraint_distance: f64) -> Result<Self, PhysicsError> {
        if constraint_distance < 0.0 {
            return Err(PhysicsError::InvalidDistance);
        }
        Ok(Self {
            object1,
            object2,
            constraint_distance,
            baumgarte: 0.2,
            lambda: 0.0,
        })
    }

    /// Sets the Baumgarte stabilization factor.
    pub fn with_baumgarte(mut self, factor: f64) -> Self {
        self.baumgarte = factor;
        self
    }

    /// Solves the joint constraint for one iteration.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    pub fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        let dx = self.object2.position.x - self.object1.position.x;
        let dy = self.object2.position.y - self.object1.position.y;
        let dz = self.object2.position.z - self.object1.position.z;
        let current_distance = (dx * dx + dy * dy + dz * dz).sqrt();

        if current_distance < 1e-10 {
            // Objects are at the same position, can't determine direction
            return Ok(());
        }

        let error = current_distance - self.constraint_distance;

        // Calculate inverse masses for mass-weighted corrections
        let inv_mass1 = if self.object1.mass.is_infinite() { 0.0 } else { 1.0 / self.object1.mass };
        let inv_mass2 = if self.object2.mass.is_infinite() { 0.0 } else { 1.0 / self.object2.mass };
        let total_inv_mass = inv_mass1 + inv_mass2;

        if total_inv_mass < 1e-10 {
            // Both objects have infinite mass, can't correct
            return Ok(());
        }

        // Normalize direction
        let nx = dx / current_distance;
        let ny = dy / current_distance;
        let nz = dz / current_distance;

        // Baumgarte stabilization: bias velocity toward zero error
        let bias = self.baumgarte * error / dt;

        // Calculate relative velocity along constraint direction
        let rel_vx = self.object2.velocity.x - self.object1.velocity.x;
        let rel_vy = self.object2.velocity.y - self.object1.velocity.y;
        let rel_vz = self.object2.velocity.z - self.object1.velocity.z;
        let relative_velocity = rel_vx * nx + rel_vy * ny + rel_vz * nz;

        // Impulse magnitude
        let lambda = -(relative_velocity + bias) / total_inv_mass;

        // Clamp impulse for stability
        let max_impulse = 0.1 / dt;
        let clamped_lambda = lambda.clamp(-max_impulse, max_impulse);

        // Track accumulated impulse for warm starting
        self.lambda += clamped_lambda;

        // Apply velocity corrections (mass-weighted)
        self.object1.velocity.x -= clamped_lambda * inv_mass1 * nx;
        self.object1.velocity.y -= clamped_lambda * inv_mass1 * ny;
        self.object1.velocity.z -= clamped_lambda * inv_mass1 * nz;
        self.object2.velocity.x += clamped_lambda * inv_mass2 * nx;
        self.object2.velocity.y += clamped_lambda * inv_mass2 * ny;
        self.object2.velocity.z += clamped_lambda * inv_mass2 * nz;

        // Position correction (mass-weighted)
        let max_correction = 0.1;
        let position_correction = error.clamp(-max_correction * 2.0, max_correction * 2.0);

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
    /// The distance between current and target constraint distance
    pub fn calculate_error(&self) -> f64 {
        let dx = self.object2.position.x - self.object1.position.x;
        let dy = self.object2.position.y - self.object1.position.y;
        let dz = self.object2.position.z - self.object1.position.z;
        let current_distance = (dx * dx + dy * dy + dz * dz).sqrt();
        (current_distance - self.constraint_distance).abs()
    }
}

impl Constraint3D for Joint3D {
    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        Joint3D::solve(self, dt)
    }

    fn calculate_error(&self) -> f64 {
        Joint3D::calculate_error(self)
    }

    fn get_lambda(&self) -> f64 {
        self.lambda
    }

    fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda;
    }
}

/// A spring constraint between two 3D objects.
///
/// Models an elastic connection with configurable spring constant and damping.
/// Implements Hooke's law with viscous damping.
///
/// # Physics Model
///
/// The spring force is calculated as:
/// ```text
/// F_spring = k × (current_length - rest_length)
/// F_damping = c × relative_velocity · direction
/// ```
///
/// # Examples
///
/// ```rust,ignore
/// use rs_physics::constraints::Spring3D;
/// use rs_physics::models::ObjectIn3D;
///
/// let mut obj1 = ObjectIn3D::default();
/// obj1.position = Axis3D { x: 0.0, y: 0.0, z: 0.0 };
/// let mut obj2 = ObjectIn3D::default();
/// obj2.position = Axis3D { x: 3.0, y: 0.0, z: 0.0 };
/// let spring = Spring3D::new(obj1, obj2, 100.0, 2.0, 0.5).expect("Valid spring");
/// ```
pub struct Spring3D {
    /// First object connected to the spring
    pub object1: ObjectIn3D,
    /// Second object connected to the spring
    pub object2: ObjectIn3D,
    /// Spring stiffness constant (N/m)
    pub spring_constant: f64,
    /// Natural/rest length of the spring (m)
    pub rest_length: f64,
    /// Viscous damping coefficient (N·s/m)
    pub damping_factor: f64,
    /// Accumulated impulse for warm starting
    pub lambda: f64,
}

impl Spring3D {
    /// Creates a new Spring3D constraint between two objects.
    ///
    /// # Arguments
    ///
    /// * `object1` - First object connected to the spring
    /// * `object2` - Second object connected to the spring
    /// * `spring_constant` - Spring stiffness (N/m, must be > 0)
    /// * `rest_length` - Natural length of the spring (m, must be >= 0)
    /// * `damping_factor` - Viscous damping coefficient (N·s/m, must be >= 0)
    pub fn new(
        object1: ObjectIn3D,
        object2: ObjectIn3D,
        spring_constant: f64,
        rest_length: f64,
        damping_factor: f64,
    ) -> Result<Self, PhysicsError> {
        if spring_constant <= 0.0 {
            return Err(PhysicsError::InvalidCoefficient);
        }
        if rest_length < 0.0 {
            return Err(PhysicsError::InvalidDistance);
        }
        if damping_factor < 0.0 {
            return Err(PhysicsError::InvalidCoefficient);
        }
        Ok(Self {
            object1,
            object2,
            spring_constant,
            rest_length,
            damping_factor,
            lambda: 0.0,
        })
    }

    /// Solves the spring constraint for one timestep.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    pub fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        let dx = self.object2.position.x - self.object1.position.x;
        let dy = self.object2.position.y - self.object1.position.y;
        let dz = self.object2.position.z - self.object1.position.z;
        let current_length = (dx * dx + dy * dy + dz * dz).sqrt();

        if current_length < 1e-10 {
            return Ok(());
        }

        // Normalize direction
        let nx = dx / current_length;
        let ny = dy / current_length;
        let nz = dz / current_length;

        let stretch = current_length - self.rest_length;

        // Calculate spring force
        let spring_force = self.spring_constant * stretch;

        // Calculate relative velocity along spring axis
        let rel_vx = self.object2.velocity.x - self.object1.velocity.x;
        let rel_vy = self.object2.velocity.y - self.object1.velocity.y;
        let rel_vz = self.object2.velocity.z - self.object1.velocity.z;
        let relative_velocity_along_spring = rel_vx * nx + rel_vy * ny + rel_vz * nz;

        // Calculate damping force
        let damping_force = self.damping_factor * relative_velocity_along_spring;

        // Total force magnitude
        let total_force = spring_force + damping_force;

        // Apply forces (F = ma, so a = F/m)
        let accel1 = total_force / self.object1.mass;
        let accel2 = total_force / self.object2.mass;

        self.object1.velocity.x += accel1 * nx * dt;
        self.object1.velocity.y += accel1 * ny * dt;
        self.object1.velocity.z += accel1 * nz * dt;
        self.object2.velocity.x -= accel2 * nx * dt;
        self.object2.velocity.y -= accel2 * ny * dt;
        self.object2.velocity.z -= accel2 * nz * dt;

        // Update positions
        self.object1.position.x += self.object1.velocity.x * dt;
        self.object1.position.y += self.object1.velocity.y * dt;
        self.object1.position.z += self.object1.velocity.z * dt;
        self.object2.position.x += self.object2.velocity.x * dt;
        self.object2.position.y += self.object2.velocity.y * dt;
        self.object2.position.z += self.object2.velocity.z * dt;

        Ok(())
    }

    /// Calculates the current constraint error.
    ///
    /// # Returns
    ///
    /// The difference between current and rest length
    pub fn calculate_error(&self) -> f64 {
        let dx = self.object2.position.x - self.object1.position.x;
        let dy = self.object2.position.y - self.object1.position.y;
        let dz = self.object2.position.z - self.object1.position.z;
        let current_length = (dx * dx + dy * dy + dz * dz).sqrt();
        (current_length - self.rest_length).abs()
    }

    /// Calculates the critical damping coefficient for this spring.
    pub fn critical_damping(&self) -> f64 {
        let m_reduced = (self.object1.mass * self.object2.mass)
            / (self.object1.mass + self.object2.mass);
        2.0 * (self.spring_constant * m_reduced).sqrt()
    }

    /// Returns the damping ratio (ζ) for this spring.
    pub fn damping_ratio(&self) -> f64 {
        self.damping_factor / self.critical_damping()
    }
}

impl Constraint3D for Spring3D {
    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        Spring3D::solve(self, dt)
    }

    fn calculate_error(&self) -> f64 {
        Spring3D::calculate_error(self)
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

    fn make_object_at(x: f64, y: f64, z: f64) -> ObjectIn3D {
        ObjectIn3D {
            mass: 1.0,
            velocity: Velocity3D { x: 0.0, y: 0.0, z: 0.0 },
            position: Axis3D { x, y, z },
            forces: Vec::new(),
            material: None,
        }
    }

    #[test]
    fn test_joint_3d_creation() {
        let obj1 = make_object_at(0.0, 0.0, 0.0);
        let obj2 = make_object_at(5.0, 0.0, 0.0);
        let joint = Joint3D::new(obj1, obj2, 5.0);
        assert!(joint.is_ok());
    }

    #[test]
    fn test_joint_3d_invalid_distance() {
        let obj1 = make_object_at(0.0, 0.0, 0.0);
        let obj2 = make_object_at(5.0, 0.0, 0.0);
        let joint = Joint3D::new(obj1, obj2, -1.0);
        assert!(joint.is_err());
    }

    #[test]
    fn test_joint_3d_solve() {
        let obj1 = make_object_at(0.0, 0.0, 0.0);
        let obj2 = make_object_at(6.0, 0.0, 0.0);
        let mut joint = Joint3D::new(obj1, obj2, 5.0).unwrap();

        let initial_error = joint.calculate_error();
        joint.solve(0.1).unwrap();
        let final_error = joint.calculate_error();

        assert!(final_error < initial_error, "Error should decrease after solving");
    }

    #[test]
    fn test_joint_3d_diagonal() {
        let obj1 = make_object_at(0.0, 0.0, 0.0);
        let obj2 = make_object_at(3.0, 4.0, 0.0); // Distance = 5
        let mut joint = Joint3D::new(obj1, obj2, 5.0).unwrap();

        let error = joint.calculate_error();
        assert!(error < 1e-10, "Error should be near zero when at constraint distance");

        joint.solve(0.1).unwrap();
    }

    #[test]
    fn test_joint_3d_diagonal_xyz() {
        let obj1 = make_object_at(0.0, 0.0, 0.0);
        let obj2 = make_object_at(2.0, 2.0, 1.0); // Distance = 3
        let mut joint = Joint3D::new(obj1, obj2, 3.0).unwrap();

        let error = joint.calculate_error();
        assert!(error < 1e-10, "Error should be near zero when at constraint distance");
    }

    #[test]
    fn test_spring_3d_creation() {
        let obj1 = make_object_at(0.0, 0.0, 0.0);
        let obj2 = make_object_at(3.0, 0.0, 0.0);
        let spring = Spring3D::new(obj1, obj2, 100.0, 2.0, 0.5);
        assert!(spring.is_ok());
    }

    #[test]
    fn test_spring_3d_invalid_constant() {
        let obj1 = make_object_at(0.0, 0.0, 0.0);
        let obj2 = make_object_at(3.0, 0.0, 0.0);
        let spring = Spring3D::new(obj1, obj2, -100.0, 2.0, 0.5);
        assert!(spring.is_err());
    }

    #[test]
    fn test_spring_3d_solve() {
        let obj1 = make_object_at(0.0, 0.0, 0.0);
        let obj2 = make_object_at(5.0, 0.0, 0.0);
        let mut spring = Spring3D::new(obj1, obj2, 100.0, 3.0, 0.5).unwrap();

        spring.solve(0.01).unwrap();

        // Objects should move towards each other
        assert!(spring.object1.position.x > 0.0, "Object 1 should move right");
        assert!(spring.object2.position.x < 5.0, "Object 2 should move left");
    }

    #[test]
    fn test_spring_3d_solve_z_direction() {
        let obj1 = make_object_at(0.0, 0.0, 0.0);
        let obj2 = make_object_at(0.0, 0.0, 5.0);
        let mut spring = Spring3D::new(obj1, obj2, 100.0, 3.0, 0.5).unwrap();

        spring.solve(0.01).unwrap();

        // Objects should move towards each other along z-axis
        assert!(spring.object1.position.z > 0.0, "Object 1 should move forward");
        assert!(spring.object2.position.z < 5.0, "Object 2 should move backward");
    }

    #[test]
    fn test_spring_3d_critical_damping() {
        let obj1 = ObjectIn3D {
            mass: 2.0,
            velocity: Velocity3D { x: 0.0, y: 0.0, z: 0.0 },
            position: Axis3D { x: 0.0, y: 0.0, z: 0.0 },
            forces: Vec::new(),
            material: None,
        };
        let obj2 = ObjectIn3D {
            mass: 2.0,
            velocity: Velocity3D { x: 0.0, y: 0.0, z: 0.0 },
            position: Axis3D { x: 3.0, y: 0.0, z: 0.0 },
            forces: Vec::new(),
            material: None,
        };
        let spring = Spring3D::new(obj1, obj2, 100.0, 2.0, 0.0).unwrap();

        let critical = spring.critical_damping();
        // Reduced mass = 2*2/(2+2) = 1
        // Critical damping = 2 * sqrt(100 * 1) = 20
        assert!((critical - 20.0).abs() < 1e-6);
    }
}
