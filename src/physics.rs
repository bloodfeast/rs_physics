// src/physics.rs

pub use crate::constants_config::PhysicsConstants;

/// Creates a new set of physics constants.
/// # Arguments
/// * `gravity` - The acceleration due to gravity.
/// * `air_density` - The density of air.
/// * `speed_of_sound` - The speed of sound in air.
/// * `atmospheric_pressure` - The atmospheric pressure.
///
/// # Returns
/// A new set of physics constants.
///
/// # Example
/// ```
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(Some(9.81), Some(1.225), Some(343.0), Some(101325.0));
/// ```
pub fn create_constants(
    gravity: Option<f64>,
    air_density: Option<f64>,
    speed_of_sound: Option<f64>,
    atmospheric_pressure: Option<f64>
) -> PhysicsConstants {
    PhysicsConstants::new(gravity, air_density, speed_of_sound, atmospheric_pressure)
}

/// Calculates the terminal velocity of an object.
/// # Arguments
/// * `constants` - The set of physics constants to use.
/// * `mass` - The mass of the object.
/// * `drag_coefficient` - The drag coefficient of the object.
/// * `cross_sectional_area` - The cross-sectional area of the object.
///
/// # Returns
/// The terminal velocity of the object.
///
/// # Errors
/// Returns an error if the mass, drag coefficient, or cross-sectional area are not positive.
///
/// # Example
/// ```
/// use rs_physics::physics::calculate_terminal_velocity;
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(None, None, None, None);
///
/// let terminal_velocity = calculate_terminal_velocity(&constants, 1.0, 0.5, 1.0);
///
/// assert_eq!(terminal_velocity, Ok(5.658773213843641));
/// ```
///
pub fn calculate_terminal_velocity(constants: &PhysicsConstants, mass: f64, drag_coefficient: f64, cross_sectional_area: f64) -> Result<f64, &'static str> {
    constants.calculate_terminal_velocity(mass, drag_coefficient, cross_sectional_area)
}

/// Calculates the air resistance on an object.
/// # Arguments
/// * `constants` - The set of physics constants to use.
/// * `velocity` - The velocity of the object.
/// * `drag_coefficient` - The drag coefficient of the object.
/// * `cross_sectional_area` - The cross-sectional area of the object.
///
/// # Returns
/// The air resistance on the object.
///
/// # Errors
/// Returns an error if the drag coefficient or cross-sectional area are negative.
///
/// # Example
/// ```
/// use rs_physics::physics::calculate_air_resistance;
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(None, None, None, None);
/// let air_resistance = calculate_air_resistance(&constants, 1.0, 0.5, 1.0);
/// assert_eq!(air_resistance, Ok(0.30625));
/// ```
///
pub fn calculate_air_resistance(constants: &PhysicsConstants, velocity: f64, drag_coefficient: f64, cross_sectional_area: f64) -> Result<f64, &'static str> {
    constants.calculate_air_resistance(velocity, drag_coefficient, cross_sectional_area)
}

/// Calculates the acceleration of an object.
/// # Arguments
/// * `constants` - The set of physics constants to use.
/// * `force` - The force acting on the object.
/// * `mass` - The mass of the object.
///
/// # Returns
/// The acceleration of the object.
///
/// # Errors
/// Returns an error if the mass is zero.
///
/// # Example
/// ```
/// use rs_physics::physics::calculate_acceleration;
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(None, None, None, None);
/// let acceleration = calculate_acceleration(&constants, 10.0, 2.0);
/// assert_eq!(acceleration, Ok(5.0));
/// ```
///
pub fn calculate_acceleration(constants: &PhysicsConstants, force: f64, mass: f64) -> Result<f64, &'static str> {
    constants.calculate_acceleration(force, mass)
}

/// Calculates the deceleration of an object.
/// # Arguments
/// * `constants` - The set of physics constants to use.
/// * `force` - The force acting on the object.
/// * `mass` - The mass of the object.
///
/// # Returns
/// The deceleration of the object.
///
/// # Errors
/// Returns an error if the mass is zero.
///
/// # Example
/// ```
/// use rs_physics::physics::calculate_deceleration;
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(None, None, None, None);
/// let deceleration = calculate_deceleration(&constants, 10.0, 2.0);
/// assert_eq!(deceleration, Ok(-5.0));
/// ```
///
pub fn calculate_deceleration(constants: &PhysicsConstants, force: f64, mass: f64) -> Result<f64, &'static str> {
    constants.calculate_deceleration(force, mass)
}

/// Calculates the force acting on an object.
/// # Arguments
/// * `constants` - The set of physics constants to use.
/// * `mass` - The mass of the object.
/// * `acceleration` - The acceleration of the object.
///
/// # Returns
/// The force acting on the object.
///
/// # Errors
/// Returns an error if the mass is negative.
///
/// # Example
/// ```
/// use rs_physics::physics::calculate_force;
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(None, None, None, None);
/// let force = calculate_force(&constants, 2.0, 5.0);
/// assert_eq!(force, Ok(10.0));
/// ```
///
pub fn calculate_force(constants: &PhysicsConstants, mass: f64, acceleration: f64) -> Result<f64, &'static str> {
    constants.calculate_force(mass, acceleration)
}

/// Calculates the momentum of an object.
/// # Arguments
/// * `constants` - The set of physics constants to use.
/// * `mass` - The mass of the object.
/// * `velocity` - The velocity of the object.
///
/// # Returns
/// The momentum of the object.
///
/// # Errors
/// Returns an error if the mass is negative.
///
/// # Example
/// ```
/// use rs_physics::physics::calculate_momentum;
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(None, None, None, None);
/// let momentum = calculate_momentum(&constants, 2.0, 5.0);
/// assert_eq!(momentum, Ok(10.0));
/// ```
///
pub fn calculate_momentum(constants: &PhysicsConstants, mass: f64, velocity: f64) -> Result<f64, &'static str> {
    constants.calculate_momentum(mass, velocity)
}

/// Calculates the final velocity of an object under constant acceleration.
/// # Arguments
/// * `constants` - The set of physics constants to use.
/// * `initial_velocity` - The initial velocity of the object.
/// * `acceleration` - The acceleration of the object.
/// * `time` - The duration of acceleration.
///
/// # Returns
/// The final velocity of the object.
///
/// # Errors
/// Returns an error if the time is negative.
///
/// # Example
/// ```
/// use rs_physics::physics::calculate_velocity;
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(None, None, None, None);
/// let final_velocity = calculate_velocity(&constants, 10.0, 2.0, 5.0);
/// assert_eq!(final_velocity, Ok(20.0));
/// ```
///
pub fn calculate_velocity(constants: &PhysicsConstants, initial_velocity: f64, acceleration: f64, time: f64) -> Result<f64, &'static str> {
    constants.calculate_velocity(initial_velocity, acceleration, time)
}

/// Calculates the average velocity of an object.
/// # Arguments
/// * `constants` - The set of physics constants to use.
/// * `displacement` - The total displacement of the object.
/// * `time` - The time taken for the displacement.
///
/// # Returns
/// The average velocity of the object.
///
/// # Errors
/// Returns an error if the time is zero.
///
/// # Example
/// ```
/// use rs_physics::physics::calculate_average_velocity;
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(None, None, None, None);
/// let avg_velocity = calculate_average_velocity(&constants, 100.0, 10.0);
/// assert_eq!(avg_velocity, Ok(10.0));
/// ```
///
pub fn calculate_average_velocity(constants: &PhysicsConstants, displacement: f64, time: f64) -> Result<f64, &'static str> {
    constants.calculate_average_velocity(displacement, time)
}

/// Calculates the kinetic energy of an object.
/// # Arguments
/// * `constants` - The set of physics constants to use.
/// * `mass` - The mass of the object.
/// * `velocity` - The velocity of the object.
///
/// # Returns
/// The kinetic energy of the object.
///
/// # Errors
/// Returns an error if the mass is negative.
///
/// # Example
/// ```
/// use rs_physics::physics::calculate_kinetic_energy;
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(None, None, None, None);
/// let kinetic_energy = calculate_kinetic_energy(&constants, 2.0, 3.0);
/// assert_eq!(kinetic_energy, Ok(9.0));
/// ```
///
pub fn calculate_kinetic_energy(constants: &PhysicsConstants, mass: f64, velocity: f64) -> Result<f64, &'static str> {
    constants.calculate_kinetic_energy(mass, velocity)
}

/// Calculates the gravitational potential energy of an object.
/// # Arguments
/// * `constants` - The set of physics constants to use.
/// * `mass` - The mass of the object.
/// * `height` - The height of the object above a reference point.
///
/// # Returns
/// The gravitational potential energy of the object.
///
/// # Errors
/// Returns an error if the mass is negative.
///
/// # Example
/// ```
/// use rs_physics::physics::calculate_potential_energy;
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(Some(9.8), None, None, None);
/// let potential_energy = calculate_potential_energy(&constants, 2.0, 5.0);
/// assert_eq!(potential_energy, Ok(98.0));
/// ```
///
pub fn calculate_potential_energy(constants: &PhysicsConstants, mass: f64, height: f64) -> Result<f64, &'static str> {
    constants.calculate_potential_energy(mass, height)
}

/// Calculates the work done on an object.
/// # Arguments
/// * `constants` - The set of physics constants to use.
/// * `force` - The force applied to the object.
/// * `displacement` - The displacement of the object.
///
/// # Returns
/// The work done on the object.
///
/// # Example
/// ```
/// use rs_physics::physics::calculate_work;
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(None, None, None, None);
/// let work = calculate_work(&constants, 10.0, 5.0);
/// assert_eq!(work, Ok(50.0));
/// ```
///
pub fn calculate_work(constants: &PhysicsConstants, force: f64, displacement: f64) -> Result<f64, &'static str> {
    constants.calculate_work(force, displacement)
}


/// Calculates the power of a system.
/// # Arguments
/// * `constants` - The set of physics constants to use.
/// * `work` - The work done by the system.
/// * `time` - The time taken to do the work.
///
/// # Returns
/// The power of the system.
///
/// # Errors
/// Returns an error if the time is zero.
///
/// # Example
/// ```
/// use rs_physics::physics::calculate_power;
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(None, None, None, None);
/// let power = calculate_power(&constants, 100.0, 10.0);
/// assert_eq!(power, Ok(10.0));
/// ```
///
pub fn calculate_power(constants: &PhysicsConstants, work: f64, time: f64) -> Result<f64, &'static str> {
    constants.calculate_power(work, time)
}

/// Calculates the impulse applied to an object.
/// # Arguments
/// * `constants` - The set of physics constants to use.
/// * `force` - The force applied to the object.
/// * `time` - The duration of the force application.
///
/// # Returns
/// The impulse applied to the object.
///
/// # Errors
/// Returns an error if the time is negative.
///
/// # Example
/// ```
/// use rs_physics::physics::calculate_impulse;
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(None, None, None, None);
/// let impulse = calculate_impulse(&constants, 10.0, 0.5);
/// assert_eq!(impulse, Ok(5.0));
/// ```
///
pub fn calculate_impulse(constants: &PhysicsConstants, force: f64, time: f64) -> Result<f64, &'static str> {
    constants.calculate_impulse(force, time)
}

/// Calculates the coefficient of restitution for a collision.
/// # Arguments
/// * `constants` - The set of physics constants to use.
/// * `velocity_before` - The velocity before the collision.
/// * `velocity_after` - The velocity after the collision.
///
/// # Returns
/// The coefficient of restitution.
///
/// # Errors
/// Returns an error if the velocity before collision is zero.
///
/// # Example
/// ```
/// use rs_physics::physics::calculate_coefficient_of_restitution;
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(None, None, None, None);
/// let cor = calculate_coefficient_of_restitution(&constants, -5.0, 3.0);
/// assert_eq!(cor, Ok(0.6));
/// ```
///
pub fn calculate_coefficient_of_restitution(constants: &PhysicsConstants, velocity_before: f64, velocity_after: f64) -> Result<f64, &'static str> {
    constants.calculate_coefficient_of_restitution(velocity_before, velocity_after)
}


/// Calculates the time of flight for a projectile.
/// # Arguments
/// * `constants` - The set of physics constants to use.
/// * `initial_velocity` - The initial velocity of the projectile.
/// * `angle` - The launch angle in radians.
///
/// # Returns
/// The time of flight for the projectile.
///
/// # Errors
/// Returns an error if the initial velocity is negative or if the angle is not between 0 and π/2.
///
/// # Example
/// ```
/// use std::f64::consts::PI;
/// use rs_physics::physics::calculate_projectile_time_of_flight;
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(Some(9.8), None, None, None);
/// let time = calculate_projectile_time_of_flight(&constants, 10.0, PI/4.0);
/// assert_eq!(time.unwrap(), 1.4430750636460152);
/// ```
///
pub fn calculate_projectile_time_of_flight(constants: &PhysicsConstants, initial_velocity: f64, angle: f64) -> Result<f64, &'static str> {
    constants.calculate_projectile_time_of_flight(initial_velocity, angle)
}

/// Calculates the maximum height reached by a projectile.
/// # Arguments
/// * `constants` - The set of physics constants to use.
/// * `initial_velocity` - The initial velocity of the projectile.
/// * `angle` - The launch angle in radians.
///
/// # Returns
/// The maximum height reached by the projectile.
///
/// # Errors
/// Returns an error if the initial velocity is negative or if the angle is not between 0 and π/2.
///
/// # Example
/// ```
/// use std::f64::consts::PI;
/// use rs_physics::physics::calculate_projectile_max_height;
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(Some(9.8), None, None, None);
/// let height = calculate_projectile_max_height(&constants, 10.0, PI/4.0);
/// assert_eq!(height.unwrap(), 2.5510204081632657);
/// ```
///
pub fn calculate_projectile_max_height(constants: &PhysicsConstants, initial_velocity: f64, angle: f64) -> Result<f64, &'static str> {
    constants.calculate_projectile_max_height(initial_velocity, angle)
}

/// Calculates the centripetal force on an object in circular motion.
/// # Arguments
/// * `constants` - The set of physics constants to use.
/// * `mass` - The mass of the object.
/// * `velocity` - The velocity of the object.
/// * `radius` - The radius of the circular path.
///
/// # Returns
/// The centripetal force on the object.
///
/// # Errors
/// Returns an error if the mass is negative or if the radius is zero.
///
/// # Example
/// ```
/// use rs_physics::physics::calculate_centripetal_force;
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(None, None, None, None);
/// let force = calculate_centripetal_force(&constants, 1.0, 5.0, 2.0);
/// assert_eq!(force, Ok(12.5));
/// ```
///
pub fn calculate_centripetal_force(constants: &PhysicsConstants, mass: f64, velocity: f64, radius: f64) -> Result<f64, &'static str> {
    constants.calculate_centripetal_force(mass, velocity, radius)
}

/// Calculates the torque applied to an object.
/// # Arguments
/// * `constants` - The set of physics constants to use.
/// * `force` - The force applied.
/// * `lever_arm` - The distance from the pivot point to the point where the force is applied.
/// * `angle` - The angle between the force vector and the lever arm, in radians.
///
/// # Returns
/// The torque applied to the object.
///
/// # Errors
/// Returns an error if the lever arm is negative.
///
/// # Example
/// ```
/// use std::f64::consts::PI;
/// use rs_physics::physics::calculate_torque;
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(None, None, None, None);
/// let torque = calculate_torque(&constants, 10.0, 2.0, PI/2.0);
/// assert_eq!(torque, Ok(20.0));
/// ```
///
pub fn calculate_torque(constants: &PhysicsConstants, force: f64, lever_arm: f64, angle: f64) -> Result<f64, &'static str> {
    constants.calculate_torque(force, lever_arm, angle)
}

/// Calculates the angular velocity of an object in circular motion.
/// # Arguments
/// * `constants` - The set of physics constants to use.
/// * `linear_velocity` - The linear velocity of the object.
/// * `radius` - The radius of the circular path.
///
/// # Returns
/// The angular velocity of the object.
///
/// # Errors
/// Returns an error if the radius is zero.
///
/// # Example
/// ```
/// use rs_physics::physics::calculate_angular_velocity;
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(None, None, None, None);
/// let angular_velocity = calculate_angular_velocity(&constants, 10.0, 2.0);
/// assert_eq!(angular_velocity, Ok(5.0));
/// ```
///
pub fn calculate_angular_velocity(constants: &PhysicsConstants, linear_velocity: f64, radius: f64) -> Result<f64, &'static str> {
    constants.calculate_angular_velocity(linear_velocity, radius)
}
