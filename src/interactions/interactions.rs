// src/interactions.rs

use crate::utils::PhysicsConstants;
use crate::forces::Force;
use crate::models::Object;
use crate::physics;


impl Object {
    /// Creates a new `Object` with the given mass, velocity, and position.
    ///
    /// # Arguments
    /// * `mass` - The mass of the object in kilograms.
    /// * `velocity` - The velocity of the object in meters per second.
    /// * `position` - The position of the object in meters.
    ///
    /// # Returns
    /// A new `Object` if the mass is positive, otherwise an error.
    ///
    /// # Errors
    /// Returns an error if the mass is not positive.
    ///
    /// # Example
    /// ```
    /// use rs_physics::models::Object;
    ///
    /// let obj = Object::new(1.0, 2.0, 3.0);
    /// assert!(obj.is_ok());
    ///
    /// let error_obj = Object::new(-1.0, 2.0, 3.0);
    /// assert!(error_obj.is_err());
    /// ```
    pub fn new(mass: f64, velocity: f64, position: f64) -> Result<Self, &'static str> {
        if mass <= 0.0 {
            return Err("Mass must be positive");
        }
        Ok(Object {
            mass,
            velocity,
            position,
            forces: Vec::new(),
        })
    }

    pub fn add_force(&mut self, force: Force) {
        self.forces.retain(|f| {
            match force {
                Force::Gravity(_) => !matches!(f, Force::Gravity(_)),
                Force::Drag { .. } => !matches!(f, Force::Drag { .. }),
                Force::Spring { .. } => !matches!(f, Force::Spring { .. }),
                Force::Constant(_) => !matches!(f, Force::Constant(_)),
                Force::Thrust { .. } => !matches!(f, Force::Thrust { .. }),
            }
        });
        self.forces.push(force);
    }

    pub fn clear_forces(&mut self) {
        self.forces.clear();
    }
}

/// Simulates an elastic collision between two objects.
///
/// # Arguments
/// * `constants` - The physics constants to use for the simulation.
/// * `obj1` - A mutable reference to the first object.
/// * `obj2` - A mutable reference to the second object.
/// * `angle` - The angle of the collision in radians.
/// * `duration` - The duration of the collision in seconds.
///
/// # Returns
/// `Ok(())` if the collision was successfully simulated.
///
/// # Example
/// ```
/// use rs_physics::interactions::elastic_collision;
/// use rs_physics::models::Object;
/// let mut obj1 = Object::new(1.0, 2.0, 0.0).unwrap();
/// let mut obj2 = Object::new(1.0, -1.0, 1.0).unwrap();
///
/// elastic_collision(&rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS, &mut obj1, &mut obj2, 0.0, 1.0, 0.45, 1.0).unwrap();
///
/// assert_eq!(obj1.velocity, -0.724375);
/// assert_eq!(obj2.velocity, 0.8975);
/// ```
///
pub fn elastic_collision(
    constants: &PhysicsConstants,
    obj1: &mut Object,
    obj2: &mut Object,
    angle: f64,
    duration: f64,
    drag_coefficient: f64,
    cross_sectional_area: f64
) -> Result<(), &'static str> {
    if duration <= 0.0 {
        return Err("Collision duration must be positive");
    }
    if drag_coefficient < 0.0 {
        return Err("Drag coefficient must be non-negative");
    }
    if cross_sectional_area < 0.0 {
        return Err("Cross-sectional area must be non-negative");
    }

    let m1 = obj1.mass;
    let m2 = obj2.mass;
    let v1 = obj1.velocity;
    let v2 = obj2.velocity;

    // Pre-compute common terms
    let total_mass = m1 + m2;
    let mass_diff = m1 - m2;

    // Calculate velocities after collision (ignoring external forces)
    let v1_final = (mass_diff * v1 + 2.0 * m2 * v2) / total_mass;
    let v2_final = (2.0 * m1 * v1 - mass_diff * v2) / total_mass;

    // Apply gravity effect
    let gravity_effect = constants.gravity * duration * angle.sin();

    // Simplified air resistance calculation
    let air_resistance_factor = 0.5 * constants.air_density * drag_coefficient * cross_sectional_area * duration;

    // Calculate final velocities considering gravity and air resistance
    obj1.velocity = v1_final - gravity_effect - air_resistance_factor * v1_final.abs() * v1_final / m1;
    obj2.velocity = v2_final - gravity_effect - air_resistance_factor * v2_final.abs() * v2_final / m2;

    Ok(())
}

/// Calculates the gravitational force between two objects.
///
/// # Arguments
/// * `constants` - The physics constants to use for the calculation.
/// * `obj1` - A reference to the first object.
/// * `obj2` - A reference to the second object.
///
/// # Returns
/// The gravitational force between the two objects in Newtons.
///
/// # Errors
/// Returns an error if the objects are at the same position.
///
/// # Example
/// ```
/// use rs_physics::interactions::gravitational_force;
/// use rs_physics::models::Object;
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(Some(6.67430e-11), None, None, None, None);
/// let obj1 = Object::new(5.97e24, 0.0, 0.0).unwrap(); // Earth
/// let obj2 = Object::new(7.34e22, 0.0, 3.84e8).unwrap(); // Moon
///
/// let force = gravitational_force(&constants, &obj1, &obj2).unwrap();
/// assert!((force - 1.98e20).abs() < 1e18); // Approximate force between Earth and Moon
/// ```
///
pub fn gravitational_force(constants: &PhysicsConstants, obj1: &Object, obj2: &Object) -> Result<f64, &'static str> {
    let distance = (obj2.position - obj1.position).abs();
    if distance == 0.0 {
        return Err("Objects cannot be at the same position");
    }
    let force = constants.gravity * obj1.mass * obj2.mass / (distance * distance);
    Ok(force)
}

/// Applies a force to an object for a given time and updates its velocity and position.
///
/// # Arguments
/// * `constants` - The physics constants to use for the calculation.
/// * `obj` - A mutable reference to the object to which the force is applied.
/// * `force` - The magnitude of the force in Newtons.
/// * `time` - The duration for which the force is applied in seconds.
///
/// # Returns
/// `Ok(())` if the force was successfully applied and the object's state updated.
///
/// # Errors
/// Returns an error if the time is negative.
///
/// # Example
/// ```
/// use rs_physics::interactions::apply_force;
/// use rs_physics::models::Object;
/// let mut obj = Object::new(1.0, 0.0, 0.0).unwrap();
///
/// apply_force(&rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS, &mut obj, 1.0, 1.0).unwrap();
///
/// assert_eq!(obj.velocity, 1.0);
/// assert_eq!(obj.position, 0.5);
/// ```
///
pub fn apply_force(constants: &PhysicsConstants, obj: &mut Object, force: f64, time: f64) -> Result<(), &'static str> {
    if time < 0.0 {
        return Err("Time cannot be negative");
    }
    let acceleration = physics::calculate_acceleration(constants, force, obj.mass);
    let velocity_change = physics::calculate_velocity(constants, 0.0, acceleration, time);
    let avg_velocity = obj.velocity + velocity_change / 2.0;
    obj.velocity += velocity_change;
    obj.position += avg_velocity * time;
    Ok(())
}