//! Legacy rotational dynamics module
//!
//! This module provides backwards-compatible 1D rotational dynamics.
//! For new code, consider using `AngularState2D` and `InertiaScalar` directly.

use crate::utils::PhysicsError;
use super::inertia::{InertiaScalar, Shape2D};

/// Shape types for moment of inertia calculations
///
/// Note: For more shapes, see `Shape2D` in the inertia module
pub enum ObjectShape {
    SolidSphere,
    HollowSphere,
    SolidCylinder,
    Rod,
}

/// A rotational object with mass, radius, and angular properties
///
/// For new code, consider using `AngularState2D` with `InertiaScalar` instead.
pub struct RotationalObject {
    pub mass: f64,
    pub radius: f64,
    pub angular_velocity: f64,
    pub moment_of_inertia: f64,
}

impl RotationalObject {
    /// Creates a new `RotationalObject` with the given mass and radius.
    ///
    /// The moment of inertia is calculated assuming a solid disk (I = 0.5 * m * r²).
    ///
    /// # Arguments
    /// * `mass` - The mass of the object in kilograms.
    /// * `radius` - The radius of the object in meters.
    ///
    /// # Returns
    /// Returns a `Result` containing the new `RotationalObject` if successful,
    /// or a `PhysicsError` if the input parameters are invalid.
    ///
    /// # Errors
    /// Returns an error if:
    /// * The mass is less than or equal to zero.
    /// * The radius is less than or equal to zero.
    ///
    /// # Examples
    /// ```
    /// use rs_physics::rotational_dynamics::RotationalObject;
    ///
    /// let obj = RotationalObject::new(1.0, 0.5).unwrap();
    /// assert_eq!(obj.mass, 1.0);
    /// assert_eq!(obj.radius, 0.5);
    /// ```
    pub fn new(mass: f64, radius: f64) -> Result<Self, PhysicsError> {
        if mass <= 0.0 {
            return Err(PhysicsError::InvalidMass);
        }
        if radius <= 0.0 {
            return Err(PhysicsError::InvalidArea);
        }
        // Use the new Shape2D for calculation
        let inertia = Shape2D::Disk(radius).moment_of_inertia(mass);
        Ok(Self {
            mass,
            radius,
            angular_velocity: 0.0,
            moment_of_inertia: inertia.value(),
        })
    }

    /// Get the moment of inertia as an InertiaScalar for use with new APIs
    pub fn inertia(&self) -> InertiaScalar {
        InertiaScalar::from(self.moment_of_inertia)
    }
}

/// Calculates the angular momentum of a rotational object.
///
/// Angular momentum L = I * ω
///
/// # Arguments
/// * `obj` - A reference to the `RotationalObject`.
///
/// # Returns
/// Returns the angular momentum in kg·m²/s.
///
/// # Examples
/// ```
/// use rs_physics::rotational_dynamics::{RotationalObject, calculate_angular_momentum};
///
/// let obj = RotationalObject::new(2.0, 0.5).unwrap();
/// let angular_momentum = calculate_angular_momentum(&obj);
/// ```
pub fn calculate_angular_momentum(obj: &RotationalObject) -> f64 {
    obj.moment_of_inertia * obj.angular_velocity
}

/// Calculates the rotational kinetic energy of a rotational object.
///
/// Rotational KE = (1/2) * I * ω²
///
/// # Arguments
/// * `obj` - A reference to the `RotationalObject`.
///
/// # Returns
/// Returns the rotational kinetic energy in joules (J).
///
/// # Examples
/// ```
/// use rs_physics::rotational_dynamics::{RotationalObject, calculate_rotational_kinetic_energy};
///
/// let obj = RotationalObject::new(2.0, 0.5).unwrap();
/// let kinetic_energy = calculate_rotational_kinetic_energy(&obj);
/// ```
pub fn calculate_rotational_kinetic_energy(obj: &RotationalObject) -> f64 {
    0.5 * obj.moment_of_inertia * obj.angular_velocity * obj.angular_velocity
}

/// Applies a torque to a rotational object for a given time period.
///
/// The angular acceleration α = τ / I, and Δω = α * Δt
///
/// # Arguments
/// * `obj` - A mutable reference to the `RotationalObject`.
/// * `torque` - The applied torque in newton-meters (N·m).
/// * `time` - The duration for which the torque is applied, in seconds.
///
/// # Returns
/// Returns `Ok(())` if the torque was successfully applied, or a `PhysicsError` if there was an error.
///
/// # Errors
/// Returns an error if:
/// * The time is less than or equal to zero.
///
/// # Examples
/// ```
/// use rs_physics::rotational_dynamics::{RotationalObject, apply_torque};
///
/// let mut obj = RotationalObject::new(2.0, 0.5).unwrap();
/// apply_torque(&mut obj, 10.0, 2.0).unwrap();
/// ```
pub fn apply_torque(obj: &mut RotationalObject, torque: f64, time: f64) -> Result<(), PhysicsError> {
    if time <= 0.0 {
        return Err(PhysicsError::InvalidTime);
    }
    let angular_acceleration = torque / obj.moment_of_inertia;
    obj.angular_velocity += angular_acceleration * time;
    Ok(())
}

/// Calculates the moment of inertia for various object shapes.
///
/// # Arguments
/// * `shape` - The shape of the object, specified as an `ObjectShape`.
/// * `mass` - The mass of the object in kilograms.
/// * `dimension` - The characteristic dimension of the object in meters
///   (e.g., radius for spheres, length for rods).
///
/// # Returns
/// Returns a `Result` containing the calculated moment of inertia in kg·m² if successful,
/// or a `PhysicsError` if the input parameters are invalid.
///
/// # Errors
/// Returns an error if:
/// * The mass is less than or equal to zero.
/// * The dimension is less than or equal to zero.
///
/// # Examples
/// ```
/// use rs_physics::rotational_dynamics::{ObjectShape, calculate_moment_of_inertia};
///
/// let moment = calculate_moment_of_inertia(&ObjectShape::SolidSphere, 1.0, 0.5).unwrap();
/// ```
pub fn calculate_moment_of_inertia(shape: &ObjectShape, mass: f64, dimension: f64) -> Result<f64, PhysicsError> {
    if mass <= 0.0 {
        return Err(PhysicsError::InvalidMass);
    }
    if dimension <= 0.0 {
        return Err(PhysicsError::InvalidDimension);
    }

    // Map to new Shape2D where applicable
    let result = match shape {
        ObjectShape::SolidSphere => {
            // I = (2/5) * m * r² for solid sphere
            0.4 * mass * dimension * dimension
        },
        ObjectShape::HollowSphere => {
            // I = (2/3) * m * r² for hollow sphere
            (2.0 / 3.0) * mass * dimension * dimension
        },
        ObjectShape::SolidCylinder => {
            // I = (1/2) * m * r² for solid cylinder (about axis)
            Shape2D::Disk(dimension).moment_of_inertia(mass).value()
        },
        ObjectShape::Rod => {
            // I = (1/12) * m * L² for rod about center
            Shape2D::Rod(dimension).moment_of_inertia(mass).value()
        },
    };

    Ok(result)
}
