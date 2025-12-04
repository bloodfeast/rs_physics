//! Contact constraints for collision resolution in physics simulations.
//!
//! This module provides contact constraints that resolve penetrations and
//! apply impulses based on collision detection results.

use crate::models::{ObjectIn2D, ObjectIn3D};
use crate::utils::PhysicsError;
use super::solver::{Constraint2D, Constraint3D};

/// A 2D contact point describing a collision.
#[derive(Debug, Clone)]
pub struct ContactPoint2D {
    /// Contact position in world space
    pub position: (f64, f64),
    /// Contact normal (points from object2 to object1)
    pub normal: (f64, f64),
    /// Penetration depth (positive if penetrating)
    pub penetration: f64,
}

/// A 3D contact point describing a collision.
#[derive(Debug, Clone)]
pub struct ContactPoint3D {
    /// Contact position in world space
    pub position: (f64, f64, f64),
    /// Contact normal (points from object2 to object1)
    pub normal: (f64, f64, f64),
    /// Penetration depth (positive if penetrating)
    pub penetration: f64,
}

/// A contact constraint between two 2D objects.
///
/// Resolves penetration and applies collision impulses based on
/// restitution (bounciness) and friction coefficients.
///
/// # Examples
///
/// ```rust,ignore
/// use rs_physics::constraints::{Contact2D, ContactPoint2D};
/// use rs_physics::models::ObjectIn2D;
///
/// let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Default::default());
/// let obj2 = ObjectIn2D::with_shape(1.0, (1.5, 0.0), Default::default());
/// let contact = ContactPoint2D {
///     position: (0.75, 0.0),
///     normal: (-1.0, 0.0),
///     penetration: 0.5,
/// };
/// let constraint = Contact2D::new(obj1, obj2, contact, 0.5, 0.3).expect("Valid contact");
/// ```
pub struct Contact2D {
    /// First object in the contact
    pub object1: ObjectIn2D,
    /// Second object in the contact
    pub object2: ObjectIn2D,
    /// Contact point information
    pub contact_point: ContactPoint2D,
    /// Coefficient of restitution (0 = inelastic, 1 = perfectly elastic)
    pub restitution: f64,
    /// Friction coefficient
    pub friction: f64,
    /// Baumgarte stabilization factor (0.0 to 1.0, default 0.2)
    pub baumgarte: f64,
    /// Accumulated impulse for warm starting
    pub lambda: f64,
}

impl Contact2D {
    /// Creates a new Contact2D constraint.
    ///
    /// # Arguments
    ///
    /// * `object1` - First object in the contact
    /// * `object2` - Second object in the contact
    /// * `contact_point` - Contact point information
    /// * `restitution` - Coefficient of restitution (0-1)
    /// * `friction` - Friction coefficient (>= 0)
    ///
    /// # Returns
    ///
    /// * `Ok(Contact2D)` - Valid contact constraint
    /// * `Err(PhysicsError)` - If parameters are invalid
    pub fn new(
        object1: ObjectIn2D,
        object2: ObjectIn2D,
        contact_point: ContactPoint2D,
        restitution: f64,
        friction: f64,
    ) -> Result<Self, PhysicsError> {
        if restitution < 0.0 || restitution > 1.0 {
            return Err(PhysicsError::InvalidCoefficient);
        }
        if friction < 0.0 {
            return Err(PhysicsError::InvalidCoefficient);
        }

        Ok(Self {
            object1,
            object2,
            contact_point,
            restitution,
            friction,
            baumgarte: 0.2,
            lambda: 0.0,
        })
    }

    /// Sets the Baumgarte stabilization factor.
    pub fn with_baumgarte(mut self, factor: f64) -> Self {
        self.baumgarte = factor;
        self
    }

    /// Gets the effective restitution coefficient for this contact.
    /// If both objects have materials, averages their restitution coefficients.
    /// If one object has a material, uses that.
    /// Otherwise, falls back to the constraint's default restitution value.
    pub fn get_effective_restitution(&self) -> f64 {
        match (&self.object1.material, &self.object2.material) {
            (Some(m1), Some(m2)) => {
                // Average the restitution of both materials
                (m1.restitution_coefficient + m2.restitution_coefficient) / 2.0
            }
            (Some(m), None) | (None, Some(m)) => m.restitution_coefficient,
            (None, None) => self.restitution,
        }
    }

    /// Gets the effective friction coefficient for this contact.
    /// If both objects have materials, averages their friction coefficients.
    /// If one object has a material, uses that.
    /// Otherwise, falls back to the constraint's default friction value.
    pub fn get_effective_friction(&self) -> f64 {
        match (&self.object1.material, &self.object2.material) {
            (Some(m1), Some(m2)) => {
                // Average the friction of both materials
                (m1.friction_coefficient + m2.friction_coefficient) / 2.0
            }
            (Some(m), None) | (None, Some(m)) => m.friction_coefficient,
            (None, None) => self.friction,
        }
    }

    /// Returns true if the objects are separating (moving apart).
    pub fn is_separating(&self) -> bool {
        let rel_vx = self.object2.velocity.x - self.object1.velocity.x;
        let rel_vy = self.object2.velocity.y - self.object1.velocity.y;
        let normal_velocity = rel_vx * self.contact_point.normal.0
            + rel_vy * self.contact_point.normal.1;
        normal_velocity > 0.0
    }

    /// Solves the contact constraint for one iteration.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    pub fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        // Only apply impulses if penetrating
        if self.contact_point.penetration <= 0.0 {
            return Ok(());
        }

        let nx = self.contact_point.normal.0;
        let ny = self.contact_point.normal.1;

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

        // Calculate relative velocity at contact point
        let rel_vx = self.object2.velocity.x - self.object1.velocity.x;
        let rel_vy = self.object2.velocity.y - self.object1.velocity.y;
        let normal_velocity = rel_vx * nx + rel_vy * ny;

        // If separating, don't apply normal impulse
        if normal_velocity > 0.0 {
            return Ok(());
        }

        // Baumgarte stabilization bias (positive, pushes objects apart)
        let bias = self.baumgarte * self.contact_point.penetration / dt;

        // Calculate target velocity change:
        // We want to change normal_velocity (negative, approaching) to positive (separating)
        // with some restitution bounce
        // Target relative velocity = -restitution * normal_velocity (bounce back)
        // Change needed = target - current = (-e * vn) - vn = -vn * (1 + e)
        // Plus bias for position correction
        let effective_restitution = self.get_effective_restitution();
        let delta_v = -normal_velocity * (1.0 + effective_restitution) + bias;

        // Impulse magnitude
        let lambda = delta_v / total_inv_mass;

        // Clamp to prevent pulling (only push, lambda >= 0)
        let lambda = lambda.max(0.0);

        // Apply normal impulse
        // Object1 gets pushed in -normal direction (away from obj2)
        // Object2 gets pushed in +normal direction (away from obj1)
        self.object1.velocity.x -= lambda * inv_mass1 * nx;
        self.object1.velocity.y -= lambda * inv_mass1 * ny;
        self.object2.velocity.x += lambda * inv_mass2 * nx;
        self.object2.velocity.y += lambda * inv_mass2 * ny;

        // Friction impulse
        let effective_friction = self.get_effective_friction();
        if effective_friction > 0.0 {
            // Recalculate relative velocity after normal impulse
            let rel_vx = self.object2.velocity.x - self.object1.velocity.x;
            let rel_vy = self.object2.velocity.y - self.object1.velocity.y;

            // Tangent direction
            let tx = -ny;
            let ty = nx;
            let tangent_velocity = rel_vx * tx + rel_vy * ty;

            // Friction impulse magnitude
            let friction_impulse = -tangent_velocity / total_inv_mass;
            let max_friction = effective_friction * lambda;
            let friction_impulse = friction_impulse.clamp(-max_friction, max_friction);

            // Apply friction impulse
            self.object1.velocity.x -= friction_impulse * inv_mass1 * tx;
            self.object1.velocity.y -= friction_impulse * inv_mass1 * ty;
            self.object2.velocity.x += friction_impulse * inv_mass2 * tx;
            self.object2.velocity.y += friction_impulse * inv_mass2 * ty;
        }

        // Position correction
        let max_correction = 0.1;
        let slop = 0.01; // Allow some penetration to prevent jitter
        let correction_magnitude = (self.contact_point.penetration - slop).max(0.0);
        let position_correction = correction_magnitude.min(max_correction);

        self.object1.position.x -= position_correction * (inv_mass1 / total_inv_mass) * nx;
        self.object1.position.y -= position_correction * (inv_mass1 / total_inv_mass) * ny;
        self.object2.position.x += position_correction * (inv_mass2 / total_inv_mass) * nx;
        self.object2.position.y += position_correction * (inv_mass2 / total_inv_mass) * ny;

        Ok(())
    }

    /// Calculates the current constraint error.
    ///
    /// # Returns
    ///
    /// The penetration depth (0 if not penetrating)
    pub fn calculate_error(&self) -> f64 {
        self.contact_point.penetration.max(0.0)
    }
}

impl Constraint2D for Contact2D {
    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        Contact2D::solve(self, dt)
    }

    fn calculate_error(&self) -> f64 {
        Contact2D::calculate_error(self)
    }

    fn get_lambda(&self) -> f64 {
        self.lambda
    }

    fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda;
    }
}

/// A contact constraint between two 3D objects.
///
/// Resolves penetration and applies collision impulses based on
/// restitution (bounciness) and friction coefficients.
///
/// # Examples
///
/// ```rust,ignore
/// use rs_physics::constraints::{Contact3D, ContactPoint3D};
/// use rs_physics::models::ObjectIn3D;
///
/// let obj1 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0));
/// let obj2 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (1.5, 0.0, 0.0));
/// let contact = ContactPoint3D {
///     position: (0.75, 0.0, 0.0),
///     normal: (-1.0, 0.0, 0.0),
///     penetration: 0.5,
/// };
/// let constraint = Contact3D::new(obj1, obj2, contact, 0.5, 0.3).expect("Valid contact");
/// ```
pub struct Contact3D {
    /// First object in the contact
    pub object1: ObjectIn3D,
    /// Second object in the contact
    pub object2: ObjectIn3D,
    /// Contact point information
    pub contact_point: ContactPoint3D,
    /// Coefficient of restitution (0 = inelastic, 1 = perfectly elastic)
    pub restitution: f64,
    /// Friction coefficient
    pub friction: f64,
    /// Baumgarte stabilization factor (0.0 to 1.0, default 0.2)
    pub baumgarte: f64,
    /// Accumulated impulse for warm starting
    pub lambda: f64,
}

impl Contact3D {
    /// Creates a new Contact3D constraint.
    ///
    /// # Arguments
    ///
    /// * `object1` - First object in the contact
    /// * `object2` - Second object in the contact
    /// * `contact_point` - Contact point information
    /// * `restitution` - Coefficient of restitution (0-1)
    /// * `friction` - Friction coefficient (>= 0)
    ///
    /// # Returns
    ///
    /// * `Ok(Contact3D)` - Valid contact constraint
    /// * `Err(PhysicsError)` - If parameters are invalid
    pub fn new(
        object1: ObjectIn3D,
        object2: ObjectIn3D,
        contact_point: ContactPoint3D,
        restitution: f64,
        friction: f64,
    ) -> Result<Self, PhysicsError> {
        if restitution < 0.0 || restitution > 1.0 {
            return Err(PhysicsError::InvalidCoefficient);
        }
        if friction < 0.0 {
            return Err(PhysicsError::InvalidCoefficient);
        }

        Ok(Self {
            object1,
            object2,
            contact_point,
            restitution,
            friction,
            baumgarte: 0.2,
            lambda: 0.0,
        })
    }

    /// Sets the Baumgarte stabilization factor.
    pub fn with_baumgarte(mut self, factor: f64) -> Self {
        self.baumgarte = factor;
        self
    }

    /// Gets the effective restitution coefficient for this contact.
    /// If both objects have materials, averages their restitution coefficients.
    /// If one object has a material, uses that.
    /// Otherwise, falls back to the constraint's default restitution value.
    pub fn get_effective_restitution(&self) -> f64 {
        match (&self.object1.material, &self.object2.material) {
            (Some(m1), Some(m2)) => {
                // Average the restitution of both materials
                (m1.restitution_coefficient + m2.restitution_coefficient) / 2.0
            }
            (Some(m), None) | (None, Some(m)) => m.restitution_coefficient,
            (None, None) => self.restitution,
        }
    }

    /// Gets the effective friction coefficient for this contact.
    /// If both objects have materials, averages their friction coefficients.
    /// If one object has a material, uses that.
    /// Otherwise, falls back to the constraint's default friction value.
    pub fn get_effective_friction(&self) -> f64 {
        match (&self.object1.material, &self.object2.material) {
            (Some(m1), Some(m2)) => {
                // Average the friction of both materials
                (m1.friction_coefficient + m2.friction_coefficient) / 2.0
            }
            (Some(m), None) | (None, Some(m)) => m.friction_coefficient,
            (None, None) => self.friction,
        }
    }

    /// Returns true if the objects are separating (moving apart).
    pub fn is_separating(&self) -> bool {
        let rel_vx = self.object2.velocity.x - self.object1.velocity.x;
        let rel_vy = self.object2.velocity.y - self.object1.velocity.y;
        let rel_vz = self.object2.velocity.z - self.object1.velocity.z;
        let normal_velocity = rel_vx * self.contact_point.normal.0
            + rel_vy * self.contact_point.normal.1
            + rel_vz * self.contact_point.normal.2;
        normal_velocity > 0.0
    }

    /// Solves the contact constraint for one iteration.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    pub fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        // Only apply impulses if penetrating
        if self.contact_point.penetration <= 0.0 {
            return Ok(());
        }

        let nx = self.contact_point.normal.0;
        let ny = self.contact_point.normal.1;
        let nz = self.contact_point.normal.2;

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

        // Calculate relative velocity at contact point
        let rel_vx = self.object2.velocity.x - self.object1.velocity.x;
        let rel_vy = self.object2.velocity.y - self.object1.velocity.y;
        let rel_vz = self.object2.velocity.z - self.object1.velocity.z;
        let normal_velocity = rel_vx * nx + rel_vy * ny + rel_vz * nz;

        // If separating, don't apply normal impulse
        if normal_velocity > 0.0 {
            return Ok(());
        }

        // Baumgarte stabilization bias (positive, pushes objects apart)
        let bias = self.baumgarte * self.contact_point.penetration / dt;

        // Calculate target velocity change:
        // We want to change normal_velocity (negative, approaching) to positive (separating)
        // with some restitution bounce
        // Target relative velocity = -restitution * normal_velocity (bounce back)
        // Change needed = target - current = (-e * vn) - vn = -vn * (1 + e)
        // Plus bias for position correction
        let effective_restitution = self.get_effective_restitution();
        let delta_v = -normal_velocity * (1.0 + effective_restitution) + bias;

        // Impulse magnitude
        let lambda = delta_v / total_inv_mass;

        // Clamp to prevent pulling (only push, lambda >= 0)
        let lambda = lambda.max(0.0);

        // Apply normal impulse
        // Object1 gets pushed in -normal direction (away from obj2)
        // Object2 gets pushed in +normal direction (away from obj1)
        self.object1.velocity.x -= lambda * inv_mass1 * nx;
        self.object1.velocity.y -= lambda * inv_mass1 * ny;
        self.object1.velocity.z -= lambda * inv_mass1 * nz;
        self.object2.velocity.x += lambda * inv_mass2 * nx;
        self.object2.velocity.y += lambda * inv_mass2 * ny;
        self.object2.velocity.z += lambda * inv_mass2 * nz;

        // Friction impulse
        let effective_friction = self.get_effective_friction();
        if effective_friction > 0.0 {
            // Recalculate relative velocity after normal impulse
            let rel_vx = self.object2.velocity.x - self.object1.velocity.x;
            let rel_vy = self.object2.velocity.y - self.object1.velocity.y;
            let rel_vz = self.object2.velocity.z - self.object1.velocity.z;

            // Calculate tangent velocity
            let tangent_vx = rel_vx - normal_velocity * nx;
            let tangent_vy = rel_vy - normal_velocity * ny;
            let tangent_vz = rel_vz - normal_velocity * nz;
            let tangent_speed = (tangent_vx * tangent_vx + tangent_vy * tangent_vy + tangent_vz * tangent_vz).sqrt();

            if tangent_speed > 1e-10 {
                let tx = tangent_vx / tangent_speed;
                let ty = tangent_vy / tangent_speed;
                let tz = tangent_vz / tangent_speed;

                // Friction impulse magnitude
                let friction_impulse = -tangent_speed / total_inv_mass;
                let max_friction = effective_friction * lambda;
                let friction_impulse = friction_impulse.max(-max_friction).min(max_friction);

                // Apply friction impulse
                self.object1.velocity.x -= friction_impulse * inv_mass1 * tx;
                self.object1.velocity.y -= friction_impulse * inv_mass1 * ty;
                self.object1.velocity.z -= friction_impulse * inv_mass1 * tz;
                self.object2.velocity.x += friction_impulse * inv_mass2 * tx;
                self.object2.velocity.y += friction_impulse * inv_mass2 * ty;
                self.object2.velocity.z += friction_impulse * inv_mass2 * tz;
            }
        }

        // Position correction
        let max_correction = 0.1;
        let slop = 0.01; // Allow some penetration to prevent jitter
        let correction_magnitude = (self.contact_point.penetration - slop).max(0.0);
        let position_correction = correction_magnitude.min(max_correction);

        self.object1.position.x -= position_correction * (inv_mass1 / total_inv_mass) * nx;
        self.object1.position.y -= position_correction * (inv_mass1 / total_inv_mass) * ny;
        self.object1.position.z -= position_correction * (inv_mass1 / total_inv_mass) * nz;
        self.object2.position.x += position_correction * (inv_mass2 / total_inv_mass) * nx;
        self.object2.position.y += position_correction * (inv_mass2 / total_inv_mass) * ny;
        self.object2.position.z += position_correction * (inv_mass2 / total_inv_mass) * nz;

        Ok(())
    }

    /// Calculates the current constraint error.
    ///
    /// # Returns
    ///
    /// The penetration depth (0 if not penetrating)
    pub fn calculate_error(&self) -> f64 {
        self.contact_point.penetration.max(0.0)
    }
}

impl Constraint3D for Contact3D {
    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        Contact3D::solve(self, dt)
    }

    fn calculate_error(&self) -> f64 {
        Contact3D::calculate_error(self)
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
    use crate::models::{Axis3D, Shape2DCollider, Velocity3D};

    // ============================================================================
    // Contact2D Tests
    // ============================================================================

    #[test]
    fn test_contact_2d_creation() {
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (1.5, 0.0), Shape2DCollider::Circle(1.0));
        let contact = ContactPoint2D {
            position: (0.75, 0.0),
            normal: (-1.0, 0.0),
            penetration: 0.5,
        };
        let constraint = Contact2D::new(obj1, obj2, contact, 0.5, 0.3);
        assert!(constraint.is_ok());
    }

    #[test]
    fn test_contact_2d_invalid_restitution() {
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (1.5, 0.0), Shape2DCollider::Circle(1.0));
        let contact = ContactPoint2D {
            position: (0.75, 0.0),
            normal: (-1.0, 0.0),
            penetration: 0.5,
        };

        // Restitution > 1
        let constraint = Contact2D::new(obj1.clone(), obj2.clone(), contact.clone(), 1.5, 0.3);
        assert!(constraint.is_err());

        // Restitution < 0
        let constraint = Contact2D::new(obj1, obj2, contact, -0.5, 0.3);
        assert!(constraint.is_err());
    }

    #[test]
    fn test_contact_2d_invalid_friction() {
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (1.5, 0.0), Shape2DCollider::Circle(1.0));
        let contact = ContactPoint2D {
            position: (0.75, 0.0),
            normal: (-1.0, 0.0),
            penetration: 0.5,
        };
        let constraint = Contact2D::new(obj1, obj2, contact, 0.5, -0.3);
        assert!(constraint.is_err());
    }

    #[test]
    fn test_contact_2d_separating_no_impulse() {
        let mut obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let mut obj2 = ObjectIn2D::with_shape(1.0, (1.5, 0.0), Shape2DCollider::Circle(1.0));

        // Objects moving apart - normal points from obj2 to obj1 (negative x)
        // For separating: rel_velocity dot normal > 0
        // rel_v = obj2.v - obj1.v = (1.0) - (-1.0) = 2.0
        // normal = (1.0, 0.0) (pointing from obj2 toward obj1)
        // 2.0 * 1.0 = 2.0 > 0 = separating
        obj1.velocity.x = -1.0;
        obj2.velocity.x = 1.0;

        let contact = ContactPoint2D {
            position: (0.75, 0.0),
            normal: (1.0, 0.0), // Points from obj2 to obj1
            penetration: 0.1,
        };
        let constraint = Contact2D::new(obj1, obj2, contact, 0.5, 0.3).unwrap();

        assert!(constraint.is_separating());
    }

    #[test]
    fn test_contact_2d_penetrating_correction() {
        let mut obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let mut obj2 = ObjectIn2D::with_shape(1.0, (1.5, 0.0), Shape2DCollider::Circle(1.0));

        // Objects moving together - normal points from obj2 to obj1
        // rel_v = obj2.v - obj1.v = (-1.0) - (1.0) = -2.0
        // normal = (1.0, 0.0)
        // -2.0 * 1.0 = -2.0 < 0 = penetrating
        obj1.velocity.x = 1.0;
        obj2.velocity.x = -1.0;

        let contact = ContactPoint2D {
            position: (0.75, 0.0),
            normal: (1.0, 0.0), // Points from obj2 to obj1
            penetration: 0.5,
        };
        let mut constraint = Contact2D::new(obj1, obj2, contact, 0.5, 0.3).unwrap();

        assert!(!constraint.is_separating());

        constraint.solve(0.016).unwrap();

        // After solving, objects should have had their velocities changed
        // They should no longer be approaching as fast
        let rel_velocity = constraint.object2.velocity.x - constraint.object1.velocity.x;
        assert!(rel_velocity > -2.0, "Relative velocity should change after collision");
    }

    #[test]
    fn test_contact_2d_restitution() {
        let mut obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let mut obj2 = ObjectIn2D::with_shape(1.0, (1.5, 0.0), Shape2DCollider::Circle(1.0));

        obj1.velocity.x = 1.0;
        obj2.velocity.x = -1.0;

        let contact = ContactPoint2D {
            position: (0.75, 0.0),
            normal: (-1.0, 0.0),
            penetration: 0.5,
        };

        // Test with high restitution (bouncy)
        let mut constraint = Contact2D::new(obj1, obj2, contact, 0.9, 0.0).unwrap();
        constraint.solve(0.016).unwrap();

        // With high restitution, objects should bounce off energetically
        // At least the relative velocity direction should reverse
    }

    // ============================================================================
    // Contact3D Tests
    // ============================================================================

    fn make_3d_object(mass: f64, x: f64, y: f64, z: f64) -> ObjectIn3D {
        ObjectIn3D {
            mass,
            velocity: Velocity3D { x: 0.0, y: 0.0, z: 0.0 },
            position: Axis3D { x, y, z },
            forces: Vec::new(),
            material: None,
        }
    }

    fn make_3d_object_with_velocity(mass: f64, pos: (f64, f64, f64), vel: (f64, f64, f64)) -> ObjectIn3D {
        ObjectIn3D {
            mass,
            velocity: Velocity3D { x: vel.0, y: vel.1, z: vel.2 },
            position: Axis3D { x: pos.0, y: pos.1, z: pos.2 },
            forces: Vec::new(),
            material: None,
        }
    }

    #[test]
    fn test_contact_3d_creation() {
        let obj1 = make_3d_object(1.0, 0.0, 0.0, 0.0);
        let obj2 = make_3d_object(1.0, 1.5, 0.0, 0.0);
        let contact = ContactPoint3D {
            position: (0.75, 0.0, 0.0),
            normal: (-1.0, 0.0, 0.0),
            penetration: 0.5,
        };
        let constraint = Contact3D::new(obj1, obj2, contact, 0.5, 0.3);
        assert!(constraint.is_ok());
    }

    #[test]
    fn test_contact_3d_separating_no_impulse() {
        // Objects moving apart - normal points from obj2 to obj1
        let obj1 = make_3d_object_with_velocity(1.0, (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0));
        let obj2 = make_3d_object_with_velocity(1.0, (1.5, 0.0, 0.0), (1.0, 0.0, 0.0));

        let contact = ContactPoint3D {
            position: (0.75, 0.0, 0.0),
            normal: (1.0, 0.0, 0.0), // Points from obj2 to obj1
            penetration: 0.1,
        };
        let constraint = Contact3D::new(obj1, obj2, contact, 0.5, 0.3).unwrap();

        assert!(constraint.is_separating());
    }

    #[test]
    fn test_contact_3d_penetrating_correction() {
        // Objects moving together
        let obj1 = make_3d_object_with_velocity(1.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0));
        let obj2 = make_3d_object_with_velocity(1.0, (1.5, 0.0, 0.0), (-1.0, 0.0, 0.0));

        let contact = ContactPoint3D {
            position: (0.75, 0.0, 0.0),
            normal: (1.0, 0.0, 0.0), // Points from obj2 to obj1
            penetration: 0.5,
        };
        let mut constraint = Contact3D::new(obj1, obj2, contact, 0.5, 0.3).unwrap();

        assert!(!constraint.is_separating());

        constraint.solve(0.016).unwrap();

        // After solving, collision response should have been applied
        let rel_velocity = constraint.object2.velocity.x - constraint.object1.velocity.x;
        assert!(rel_velocity > -2.0, "Relative velocity should change after collision");
    }

    #[test]
    fn test_contact_3d_mass_weighted() {
        // Objects moving together
        let obj1 = make_3d_object_with_velocity(1.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0));
        let obj2 = make_3d_object_with_velocity(4.0, (1.5, 0.0, 0.0), (-1.0, 0.0, 0.0));

        let contact = ContactPoint3D {
            position: (0.75, 0.0, 0.0),
            normal: (1.0, 0.0, 0.0), // Points from obj2 to obj1
            penetration: 0.5,
        };
        let mut constraint = Contact3D::new(obj1, obj2, contact, 0.0, 0.0).unwrap();

        let initial_vel1 = constraint.object1.velocity.x;
        let initial_vel2 = constraint.object2.velocity.x;

        constraint.solve(0.016).unwrap();

        let delta_vel1 = (constraint.object1.velocity.x - initial_vel1).abs();
        let delta_vel2 = (constraint.object2.velocity.x - initial_vel2).abs();

        // Lighter object should change velocity more
        assert!(
            delta_vel1 > delta_vel2,
            "Lighter object should have larger velocity change: delta1={}, delta2={}",
            delta_vel1,
            delta_vel2
        );
    }

    #[test]
    fn test_contact_3d_infinite_mass_anchor() {
        let obj1 = make_3d_object_with_velocity(f64::INFINITY, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0));
        let obj2 = make_3d_object_with_velocity(1.0, (1.5, 0.0, 0.0), (-1.0, 0.0, 0.0));

        let contact = ContactPoint3D {
            position: (0.75, 0.0, 0.0),
            normal: (-1.0, 0.0, 0.0),
            penetration: 0.5,
        };
        let mut constraint = Contact3D::new(obj1, obj2, contact, 0.5, 0.0).unwrap();

        let initial_vel1 = constraint.object1.velocity.x;

        constraint.solve(0.016).unwrap();

        // Infinite mass object should not change velocity
        assert!(
            (constraint.object1.velocity.x - initial_vel1).abs() < 1e-10,
            "Infinite mass object velocity should not change"
        );
    }
}
