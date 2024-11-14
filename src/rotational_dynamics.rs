// src/rotational_dynamics.rs

use crate::errors::PhysicsError;

pub enum ObjectShape {
    SolidSphere,
    HollowSphere,
    SolidCylinder,
    Rod,
}

pub struct RotationalObject {
    pub mass: f64,
    pub radius: f64,
    pub angular_velocity: f64,
    pub moment_of_inertia: f64,
}

impl RotationalObject {

    /// Creates a new `RotationalObject` with the given mass and radius.
    /// # Arguments
    /// * `mass` - The mass of the object in kilograms.
    /// * `radius` - The radius of the object in meters.
    ///
    /// # Return
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
        let moment_of_inertia = 0.5 * mass * radius * radius; // For a solid disk
        Ok(Self {
            mass,
            radius,
            angular_velocity: 0.0,
            moment_of_inertia,
        })
    }
}

/// Calculates the angular momentum of a rotational object.
/// # Arguments
/// * `obj` - A reference to the `RotationalObject`.
///
/// # Return
/// Returns the angular momentum in kg⋅m²/s.
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
/// # Arguments
/// * `obj` - A reference to the `RotationalObject`.
///
/// # Return
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
/// # Arguments
/// * `obj` - A mutable reference to the `RotationalObject`.
/// * `torque` - The applied torque in newton-meters (N⋅m).
/// * `time` - The duration for which the torque is applied, in seconds.
///
/// # Return
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
/// # Arguments
/// * `shape` - The shape of the object, specified as an `ObjectShape`.
/// * `mass` - The mass of the object in kilograms.
/// * `dimension` - The characteristic dimension of the object in meters (e.g., radius for spheres, length for rods).
///
/// # Return
/// Returns a `Result` containing the calculated moment of inertia in kg⋅m² if successful,
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
    match shape {
        ObjectShape::SolidSphere => Ok(0.4 * mass * dimension * dimension),
        ObjectShape::HollowSphere => Ok((2.0 / 3.0) * mass * dimension * dimension),
        ObjectShape::SolidCylinder => Ok(0.5 * mass * dimension * dimension),
        ObjectShape::Rod => Ok((1.0 / 12.0) * mass * dimension * dimension),
        // If we get this far, then we must have forgotten something along the way
        #[allow(unreachable_patterns)]
        _ => Err(PhysicsError::UnsupportedShape),
    }
}