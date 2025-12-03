//! 2D constraint solvers for physics simulations.
//!
//! This module provides constraint types for 2D physics objects.

use crate::utils::PhysicsError;
use crate::models::ObjectIn2D;
use super::solver::Constraint2D;

/// A rigid joint constraint between two 2D objects.
///
/// Maintains a fixed distance between two objects by applying position
/// and velocity corrections each timestep.
///
/// # Examples
///
/// ```rust,ignore
/// use rs_physics::constraints::Joint2D;
/// use rs_physics::models::ObjectIn2D;
///
/// let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Default::default());
/// let obj2 = ObjectIn2D::with_shape(1.0, (5.0, 0.0), Default::default());
/// let joint = Joint2D::new(obj1, obj2, 5.0).expect("Valid joint");
/// ```
pub struct Joint2D {
    /// First object connected by the joint
    pub object1: ObjectIn2D,
    /// Second object connected by the joint
    pub object2: ObjectIn2D,
    /// Target distance to maintain between objects (meters)
    pub constraint_distance: f64,
    /// Baumgarte stabilization factor (0.0 to 1.0, default 0.2)
    pub baumgarte: f64,
    /// Accumulated impulse for warm starting
    pub lambda: f64,
}

impl Joint2D {
    /// Creates a new Joint2D constraint between two objects.
    ///
    /// # Arguments
    ///
    /// * `object1` - First object in the joint
    /// * `object2` - Second object in the joint
    /// * `constraint_distance` - Target distance to maintain (must be >= 0)
    ///
    /// # Returns
    ///
    /// * `Ok(Joint2D)` - Valid joint constraint
    /// * `Err(PhysicsError)` - If constraint_distance is negative
    pub fn new(object1: ObjectIn2D, object2: ObjectIn2D, constraint_distance: f64) -> Result<Self, PhysicsError> {
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
        let current_distance = (dx * dx + dy * dy).sqrt();

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

        // Baumgarte stabilization: bias velocity toward zero error
        let bias = self.baumgarte * error / dt;

        // Calculate relative velocity along constraint direction
        let rel_vx = self.object2.velocity.x - self.object1.velocity.x;
        let rel_vy = self.object2.velocity.y - self.object1.velocity.y;
        let relative_velocity = rel_vx * nx + rel_vy * ny;

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
        self.object2.velocity.x += clamped_lambda * inv_mass2 * nx;
        self.object2.velocity.y += clamped_lambda * inv_mass2 * ny;

        // Position correction (mass-weighted)
        let max_correction = 0.1;
        let position_correction = error.clamp(-max_correction * 2.0, max_correction * 2.0);

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
    /// The distance between current and target constraint distance
    pub fn calculate_error(&self) -> f64 {
        let dx = self.object2.position.x - self.object1.position.x;
        let dy = self.object2.position.y - self.object1.position.y;
        let current_distance = (dx * dx + dy * dy).sqrt();
        (current_distance - self.constraint_distance).abs()
    }
}

impl Constraint2D for Joint2D {
    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        Joint2D::solve(self, dt)
    }

    fn calculate_error(&self) -> f64 {
        Joint2D::calculate_error(self)
    }

    fn get_lambda(&self) -> f64 {
        self.lambda
    }

    fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda;
    }
}

/// A spring constraint between two 2D objects.
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
/// use rs_physics::constraints::Spring2D;
/// use rs_physics::models::ObjectIn2D;
///
/// let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Default::default());
/// let obj2 = ObjectIn2D::with_shape(1.0, (3.0, 0.0), Default::default());
/// let spring = Spring2D::new(obj1, obj2, 100.0, 2.0, 0.5).expect("Valid spring");
/// ```
pub struct Spring2D {
    /// First object connected to the spring
    pub object1: ObjectIn2D,
    /// Second object connected to the spring
    pub object2: ObjectIn2D,
    /// Spring stiffness constant (N/m)
    pub spring_constant: f64,
    /// Natural/rest length of the spring (m)
    pub rest_length: f64,
    /// Viscous damping coefficient (N·s/m)
    pub damping_factor: f64,
    /// Accumulated impulse for warm starting
    pub lambda: f64,
}

impl Spring2D {
    /// Creates a new Spring2D constraint between two objects.
    ///
    /// # Arguments
    ///
    /// * `object1` - First object connected to the spring
    /// * `object2` - Second object connected to the spring
    /// * `spring_constant` - Spring stiffness (N/m, must be > 0)
    /// * `rest_length` - Natural length of the spring (m, must be >= 0)
    /// * `damping_factor` - Viscous damping coefficient (N·s/m, must be >= 0)
    pub fn new(
        object1: ObjectIn2D,
        object2: ObjectIn2D,
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
        let current_length = (dx * dx + dy * dy).sqrt();

        if current_length < 1e-10 {
            return Ok(());
        }

        // Normalize direction
        let nx = dx / current_length;
        let ny = dy / current_length;

        let stretch = current_length - self.rest_length;

        // Calculate spring force
        let spring_force = self.spring_constant * stretch;

        // Calculate relative velocity along spring axis
        let rel_vx = self.object2.velocity.x - self.object1.velocity.x;
        let rel_vy = self.object2.velocity.y - self.object1.velocity.y;
        let relative_velocity_along_spring = rel_vx * nx + rel_vy * ny;

        // Calculate damping force
        let damping_force = self.damping_factor * relative_velocity_along_spring;

        // Total force magnitude
        let total_force = spring_force + damping_force;

        // Apply forces (F = ma, so a = F/m)
        let accel1 = total_force / self.object1.mass;
        let accel2 = total_force / self.object2.mass;

        self.object1.velocity.x += accel1 * nx * dt;
        self.object1.velocity.y += accel1 * ny * dt;
        self.object2.velocity.x -= accel2 * nx * dt;
        self.object2.velocity.y -= accel2 * ny * dt;

        // Update positions
        self.object1.position.x += self.object1.velocity.x * dt;
        self.object1.position.y += self.object1.velocity.y * dt;
        self.object2.position.x += self.object2.velocity.x * dt;
        self.object2.position.y += self.object2.velocity.y * dt;

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
        let current_length = (dx * dx + dy * dy).sqrt();
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

impl Constraint2D for Spring2D {
    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        Spring2D::solve(self, dt)
    }

    fn calculate_error(&self) -> f64 {
        Spring2D::calculate_error(self)
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

    #[test]
    fn test_joint_2d_creation() {
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (5.0, 0.0), Shape2DCollider::Circle(1.0));
        let joint = Joint2D::new(obj1, obj2, 5.0);
        assert!(joint.is_ok());
    }

    #[test]
    fn test_joint_2d_invalid_distance() {
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (5.0, 0.0), Shape2DCollider::Circle(1.0));
        let joint = Joint2D::new(obj1, obj2, -1.0);
        assert!(joint.is_err());
    }

    #[test]
    fn test_joint_2d_solve() {
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (6.0, 0.0), Shape2DCollider::Circle(1.0));
        let mut joint = Joint2D::new(obj1, obj2, 5.0).unwrap();

        let initial_error = joint.calculate_error();
        joint.solve(0.1).unwrap();
        let final_error = joint.calculate_error();

        assert!(final_error < initial_error, "Error should decrease after solving");
    }

    #[test]
    fn test_joint_2d_diagonal() {
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (3.0, 4.0), Shape2DCollider::Circle(1.0)); // Distance = 5
        let mut joint = Joint2D::new(obj1, obj2, 5.0).unwrap();

        let error = joint.calculate_error();
        assert!(error < 1e-10, "Error should be near zero when at constraint distance");

        joint.solve(0.1).unwrap();
        // Positions should remain similar since we're at constraint distance
    }

    #[test]
    fn test_spring_2d_creation() {
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (3.0, 0.0), Shape2DCollider::Circle(1.0));
        let spring = Spring2D::new(obj1, obj2, 100.0, 2.0, 0.5);
        assert!(spring.is_ok());
    }

    #[test]
    fn test_spring_2d_invalid_constant() {
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (3.0, 0.0), Shape2DCollider::Circle(1.0));
        let spring = Spring2D::new(obj1, obj2, -100.0, 2.0, 0.5);
        assert!(spring.is_err());
    }

    #[test]
    fn test_spring_2d_solve() {
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (5.0, 0.0), Shape2DCollider::Circle(1.0));
        let mut spring = Spring2D::new(obj1, obj2, 100.0, 3.0, 0.5).unwrap();

        spring.solve(0.01).unwrap();

        // Objects should move towards each other
        assert!(spring.object1.position.x > 0.0, "Object 1 should move right");
        assert!(spring.object2.position.x < 5.0, "Object 2 should move left");
    }

    #[test]
    fn test_spring_2d_critical_damping() {
        let obj1 = ObjectIn2D::with_shape(2.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(2.0, (3.0, 0.0), Shape2DCollider::Circle(1.0));
        let spring = Spring2D::new(obj1, obj2, 100.0, 2.0, 0.0).unwrap();

        let critical = spring.critical_damping();
        // Reduced mass = 2*2/(2+2) = 1
        // Critical damping = 2 * sqrt(100 * 1) = 20
        assert!((critical - 20.0).abs() < 1e-6);
    }
}
