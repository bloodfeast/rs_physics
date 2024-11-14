// src/physics.rs

use std::f64::consts::PI;
use log::{error, warn};
pub use crate::constants_config::PhysicsConstants;
use crate::errors::PhysicsError;

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

/// Defaulting to 0.0 to prevent a panic in the case of a completely unexpected error.
/// We may want to actually panic here (since it would indicate that the calculation has failed) but for now let's leave it up to the caller to handle.
fn warn_about_unexpected_calculation_error() -> f64 {
    warn!("Unexpected calculation error.\nDefaulting to 0.0 to prevent panic\n - This behavior may be changed in the future.");
    0.0
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
///
/// let terminal_velocity = calculate_terminal_velocity(&rs_physics::DEFAULT_PHYSICS_CONSTANTS, 1.0, 0.5, 1.0);
///
/// assert_eq!(terminal_velocity, 5.658773213843641);
/// ```
///
pub fn calculate_terminal_velocity(constants: &PhysicsConstants, mass: f64, drag_coefficient: f64, cross_sectional_area: f64) -> f64 {
    match constants.calculate_terminal_velocity(mass, drag_coefficient, cross_sectional_area) {
        Ok(velocity) => velocity,
        Err(e) => {
            error!("Error calculating terminal velocity: {}", e);
            match e {
                PhysicsError::InvalidMass => {
                    warn!("Using absolute value of mass");
                    constants.calculate_terminal_velocity(mass.abs(), drag_coefficient, cross_sectional_area).unwrap_or(0.0)
                },
                PhysicsError::InvalidCoefficient => {
                    warn!("Using absolute value of drag coefficient");
                    constants.calculate_terminal_velocity(mass, drag_coefficient.abs(), cross_sectional_area).unwrap_or(0.0)
                },
                PhysicsError::InvalidArea => {
                    warn!("Using absolute value of cross-sectional area");
                    constants.calculate_terminal_velocity(mass, drag_coefficient, cross_sectional_area.abs()).unwrap_or(0.0)
                },
                _ => warn_about_unexpected_calculation_error(),
            }
        }
    }
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
///
/// let air_resistance = calculate_air_resistance(&rs_physics::DEFAULT_PHYSICS_CONSTANTS, 1.0, 0.5, 1.0);
/// assert_eq!(air_resistance, 0.30625);
/// ```
///
pub fn calculate_air_resistance(constants: &PhysicsConstants, velocity: f64, drag_coefficient: f64, cross_sectional_area: f64) -> f64 {
    match constants.calculate_air_resistance(velocity, drag_coefficient, cross_sectional_area) {
        Ok(resistance) => resistance,
        Err(e) => {
            error!("Error calculating air resistance: {}", e);
            match e {
                PhysicsError::InvalidCoefficient => {
                    warn!("Using absolute value of drag coefficient");
                    constants.calculate_air_resistance(velocity, drag_coefficient.abs(), cross_sectional_area).unwrap_or(0.0)
                },
                PhysicsError::InvalidArea => {
                    warn!("Using absolute value of cross-sectional area");
                    constants.calculate_air_resistance(velocity, drag_coefficient, cross_sectional_area.abs()).unwrap_or(0.0)
                },
                _ => warn_about_unexpected_calculation_error(),
            }
        }
    }
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
///
/// let acceleration = calculate_acceleration(&rs_physics::DEFAULT_PHYSICS_CONSTANTS, 10.0, 2.0);
/// assert_eq!(acceleration, 5.0);
/// ```
///
pub fn calculate_acceleration(constants: &PhysicsConstants, force: f64, mass: f64) -> f64 {
    match constants.calculate_acceleration(force, mass) {
        Ok(acceleration) => acceleration,
        Err(e) => {
            error!("Error calculating acceleration: {}", e);
            match e {
                PhysicsError::InvalidMass => {
                    warn!("Using absolute value of mass");
                    constants.calculate_acceleration(force, mass.abs()).unwrap_or(0.0)
                },
                PhysicsError::DivisionByZero => {
                    error!("Mass cannot be zero");
                    0.0
                },
                _ => warn_about_unexpected_calculation_error(),
            }
        }
    }
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
///
/// let deceleration = calculate_deceleration(&rs_physics::DEFAULT_PHYSICS_CONSTANTS, 10.0, 2.0);
/// assert_eq!(deceleration, -5.0);
/// ```
///
pub fn calculate_deceleration(constants: &PhysicsConstants, force: f64, mass: f64) -> f64 {
    -calculate_acceleration(constants, force, mass)
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
///
/// let force = calculate_force(&rs_physics::DEFAULT_PHYSICS_CONSTANTS, 2.0, 5.0);
/// assert_eq!(force, 10.0);
/// ```
///
pub fn calculate_force(constants: &PhysicsConstants, mass: f64, acceleration: f64) -> f64 {
    match constants.calculate_force(mass, acceleration) {
        Ok(force) => force,
        Err(e) => {
            error!("Error calculating force: {}", e);
            match e {
                PhysicsError::InvalidMass => {
                    warn!("Using absolute value of mass");
                    constants.calculate_force(mass.abs(), acceleration).unwrap_or(0.0)
                },
                _ => warn_about_unexpected_calculation_error(),
            }
        }
    }
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
///
/// let momentum = calculate_momentum(&rs_physics::DEFAULT_PHYSICS_CONSTANTS, 2.0, 5.0);
/// assert_eq!(momentum, 10.0);
/// ```
///
pub fn calculate_momentum(constants: &PhysicsConstants, mass: f64, velocity: f64) -> f64 {
    match constants.calculate_momentum(mass, velocity) {
        Ok(momentum) => momentum,
        Err(e) => {
            error!("Error calculating momentum: {}", e);
            match e {
                PhysicsError::InvalidMass => {
                    warn!("Using absolute value of mass");
                    constants.calculate_momentum(mass.abs(), velocity).unwrap_or(0.0)
                },
                _ => warn_about_unexpected_calculation_error(),
            }
        }
    }
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
///
/// let final_velocity = calculate_velocity(&rs_physics::DEFAULT_PHYSICS_CONSTANTS, 10.0, 2.0, 5.0);
/// assert_eq!(final_velocity, 20.0);
/// ```
///
pub fn calculate_velocity(constants: &PhysicsConstants, initial_velocity: f64, acceleration: f64, time: f64) -> f64 {
    match constants.calculate_velocity(initial_velocity, acceleration, time) {
        Ok(velocity) => velocity,
        Err(e) => {
            error!("Error calculating velocity: {}", e);
            match e {
                PhysicsError::InvalidTime => {
                    warn!("Using absolute value of time");
                    constants.calculate_velocity(initial_velocity, acceleration, time.abs()).unwrap_or(initial_velocity)
                },
                _ => initial_velocity,
            }
        }
    }
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
///
/// let avg_velocity = calculate_average_velocity(&rs_physics::DEFAULT_PHYSICS_CONSTANTS, 100.0, 10.0);
/// assert_eq!(avg_velocity, 10.0);
/// ```
///
pub fn calculate_average_velocity(constants: &PhysicsConstants, displacement: f64, time: f64) -> f64 {
    constants.calculate_average_velocity(displacement, time).unwrap_or_else(|e| {
        error!("Error calculating average velocity: {}", e);
        match e {
            PhysicsError::DivisionByZero => {
                error!("Time cannot be zero");
                0.0
            },
            _ => warn_about_unexpected_calculation_error(),
        }
    })
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
///
/// let kinetic_energy = calculate_kinetic_energy(&rs_physics::DEFAULT_PHYSICS_CONSTANTS, 2.0, 3.0);
/// assert_eq!(kinetic_energy, 9.0);
/// ```
///
pub fn calculate_kinetic_energy(constants: &PhysicsConstants, mass: f64, velocity: f64) -> f64 {
    match constants.calculate_kinetic_energy(mass, velocity) {
        Ok(energy) => energy,
        Err(e) => {
            error!("Error calculating kinetic energy: {}", e);
            match e {
                PhysicsError::InvalidMass => {
                    warn!("Using absolute value of mass");
                    constants.calculate_kinetic_energy(mass.abs(), velocity).unwrap_or(0.0)
                },
                _ => warn_about_unexpected_calculation_error(),
            }
        }
    }
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
/// assert_eq!(potential_energy, 98.0);
/// ```
///
pub fn calculate_potential_energy(constants: &PhysicsConstants, mass: f64, height: f64) -> f64 {
    match constants.calculate_potential_energy(mass, height) {
        Ok(energy) => energy,
        Err(e) => {
            error!("Error calculating potential energy: {}", e);
            match e {
                PhysicsError::InvalidMass => {
                    warn!("Using absolute value of mass");
                    constants.calculate_potential_energy(mass.abs(), height).unwrap_or(0.0)
                },
                _ => warn_about_unexpected_calculation_error(),
            }
        }
    }
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
///
/// let work = calculate_work(&rs_physics::DEFAULT_PHYSICS_CONSTANTS, 10.0, 5.0);
/// assert_eq!(work, 50.0);
/// ```
///
pub fn calculate_work(constants: &PhysicsConstants, force: f64, displacement: f64) -> f64 {
    constants.calculate_work(force, displacement).unwrap_or_else(|e| {
        error!("Error calculating work: {}", e);
        0.0
    })
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
///
/// let power = calculate_power(&rs_physics::DEFAULT_PHYSICS_CONSTANTS, 100.0, 10.0);
/// assert_eq!(power, 10.0);
/// ```
///
pub fn calculate_power(constants: &PhysicsConstants, work: f64, time: f64) -> f64 {
    constants.calculate_power(work, time).unwrap_or_else(|e| {
        error!("Error calculating power: {}", e);
        match e {
            PhysicsError::DivisionByZero => {
                error!("Time cannot be zero");
                0.0
            },
            _ => warn_about_unexpected_calculation_error(),
        }
    })
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
///
/// let impulse = calculate_impulse(&rs_physics::DEFAULT_PHYSICS_CONSTANTS, 10.0, 0.5);
/// assert_eq!(impulse, 5.0);
/// ```
///
pub fn calculate_impulse(constants: &PhysicsConstants, force: f64, time: f64) -> f64 {
    match constants.calculate_impulse(force, time) {
        Ok(impulse) => impulse,
        Err(e) => {
            error!("Error calculating impulse: {}", e);
            match e {
                PhysicsError::InvalidTime => {
                    warn!("Using absolute value of time");
                    constants.calculate_impulse(force, time.abs()).unwrap_or(0.0)
                },
                _ => warn_about_unexpected_calculation_error(),
            }
        }
    }
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
///
/// let cor = calculate_coefficient_of_restitution(&rs_physics::DEFAULT_PHYSICS_CONSTANTS, -5.0, 3.0);
/// assert_eq!(cor, 0.6);
/// ```
///
pub fn calculate_coefficient_of_restitution(constants: &PhysicsConstants, velocity_before: f64, velocity_after: f64) -> f64 {
    constants.calculate_coefficient_of_restitution(velocity_before, velocity_after).unwrap_or_else(|e| {
        error!("Error calculating coefficient of restitution: {}", e);
        match e {
            PhysicsError::DivisionByZero => {
                error!("Velocity before collision cannot be zero");
                0.0
            },
            _ => warn_about_unexpected_calculation_error(),
        }
    })
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
/// assert_eq!(time, 1.4430750636460152);
/// ```
///
pub fn calculate_projectile_time_of_flight(constants: &PhysicsConstants, initial_velocity: f64, angle: f64) -> f64 {
    match constants.calculate_projectile_time_of_flight(initial_velocity, angle) {
        Ok(time) => time,
        Err(e) => {
            error!("Error calculating projectile time of flight: {}", e);
            match e {
                PhysicsError::InvalidVelocity => {
                    warn!("Invalid initial_velocity provided ({:?}), defaulting to the absolute value of initial_velocity", initial_velocity);
                    constants
                        .calculate_projectile_time_of_flight(initial_velocity.abs(), angle)
                        .unwrap_or(0.0)
                },
                PhysicsError::InvalidAngle => {
                    warn!("Recovering from InvalidAngle by clamping angle to range [0, π/2].\n - The angle provided was: {:?}", angle);
                    let clamped_angle = angle.clamp(0.0, PI / 2.0);
                    constants
                        .calculate_projectile_time_of_flight(initial_velocity, clamped_angle)
                        .unwrap_or(0.0)
                },
                _ => warn_about_unexpected_calculation_error(),
            }
        }
    }
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
/// assert_eq!(height, 2.5510204081632657);
/// ```
///
pub fn calculate_projectile_max_height(constants: &PhysicsConstants, initial_velocity: f64, angle: f64) -> f64 {
    match constants.calculate_projectile_max_height(initial_velocity, angle) {
        Ok(height) => height,
        Err(e) => {
            error!("Error calculating projectile max height: {}", e);
            match e {
                PhysicsError::InvalidVelocity => {
                    warn!("Invalid initial_velocity provided ({:?}), defaulting to the absolute value of initial_velocity", initial_velocity);
                    constants
                        .calculate_projectile_max_height(initial_velocity.abs(), angle)
                        .unwrap_or(0.0)
                },
                PhysicsError::InvalidAngle => {
                    warn!("Recovering from InvalidAngle by clamping angle to range [0, π/2].\n - The angle provided was: {:?}", angle);
                    let clamped_angle = angle.clamp(0.0, PI / 2.0);
                    constants
                        .calculate_projectile_max_height(initial_velocity, clamped_angle)
                        .unwrap_or(0.0)
                },
                _ => warn_about_unexpected_calculation_error(),
            }
        }
    }
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
///
/// let force = calculate_centripetal_force(&rs_physics::DEFAULT_PHYSICS_CONSTANTS, 1.0, 5.0, 2.0);
/// assert_eq!(force, 12.5);
/// ```
///
pub fn calculate_centripetal_force(constants: &PhysicsConstants, mass: f64, velocity: f64, radius: f64) -> f64 {
    match constants.calculate_centripetal_force(mass, velocity, radius) {
        Ok(force) => force,
        Err(e) => {
            error!("Error calculating centripetal force: {}", e);
            match e {
                PhysicsError::InvalidMass => {
                    warn!("Invalid mass provided ({:?}), defaulting to the absolute value of mass", mass);
                    constants
                        .calculate_centripetal_force(mass.abs(), velocity, radius)
                        .unwrap_or(0.0)
                },
                PhysicsError::DivisionByZero => {
                    error!("Radius cannot be zero");
                    0.0
                },
                _ => warn_about_unexpected_calculation_error(),
            }
        }
    }
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
///
/// let torque = calculate_torque(&rs_physics::DEFAULT_PHYSICS_CONSTANTS, 10.0, 2.0, PI/2.0);
/// assert_eq!(torque, 20.0);
/// ```
///
pub fn calculate_torque(constants: &PhysicsConstants, force: f64, lever_arm: f64, angle: f64) -> f64 {
    match constants.calculate_torque(force, lever_arm, angle) {
        Ok(torque) => torque,
        Err(e) => {
            error!("Error calculating torque: {}", e);
            match e {
                PhysicsError::InvalidDistance => {
                    warn!("Using absolute value of lever arm");
                    constants
                        .calculate_torque(force, lever_arm.abs(), angle)
                        .unwrap_or(0.0)
                },
                PhysicsError::InvalidAngle => {
                    warn!("Recovering from InvalidAngle by clamping angle to range [0, π].\n - The angle provided was: {:?}", angle);
                    let clamped_angle = angle.clamp(0.0, PI);
                    constants
                        .calculate_torque(force, lever_arm, clamped_angle)
                        .unwrap_or(0.0)
                },
                _ => warn_about_unexpected_calculation_error(),
            }
        }
    }
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
///
/// let angular_velocity = calculate_angular_velocity(&rs_physics::DEFAULT_PHYSICS_CONSTANTS, 10.0, 2.0);
/// assert_eq!(angular_velocity, 5.0);
/// ```
///
pub fn calculate_angular_velocity(constants: &PhysicsConstants, linear_velocity: f64, radius: f64) -> f64 {
    if linear_velocity == 0.0 {
        warn!("Linear velocity is zero; angular velocity will also be zero.");
        return 0.0;
    }
    constants.calculate_angular_velocity(linear_velocity, radius).unwrap_or_else(|e| {
        error!("Error calculating angular velocity: {}", e);
        match e {
            PhysicsError::InvalidRadius => {
                warn!("Recovering from InvalidRadius, using the absolute value of radius");
                constants
                    .calculate_angular_velocity(linear_velocity, radius.abs())
                    .unwrap_or(0.0)
            },
            PhysicsError::DivisionByZero => {
                error!("Radius cannot be zero");
                0.0
            },
            _ => warn_about_unexpected_calculation_error(),
        }
    })
}
