// src/lib.rs

use crate::constants_config::PhysicsConstants;

mod errors;
mod constants_config;
pub mod apis;
pub mod physics;
pub mod interactions;
pub mod forces;
#[cfg(feature = "rotational_dynamics")]
pub mod rotational_dynamics;
#[cfg(feature = "fluid_dynamics")]
pub mod fluid_dynamics;
#[cfg(feature = "thermodynamics")]
pub mod thermodynamics;

#[cfg(feature = "constraints")]
pub mod constraint_solvers;
#[cfg(feature = "fluid_simulation")]
pub mod fluid_simulation;
#[cfg(feature = "materials")]
pub mod materials;

pub const DEFAULT_PHYSICS_CONSTANTS: PhysicsConstants = PhysicsConstants {
    gravity: 9.80665,
    air_density: 1.225,
    speed_of_sound: 343.0,
    atmospheric_pressure: 101_325.0,
};

/// Asserts that two floating point numbers are approximately equal.
///
/// # Arguments
///
/// * `a` - The first floating point number.
/// * `b` - The second floating point number.
/// * `epsilon` - The maximum difference between `a` and `b` for them to be considered equal.
/// * `optional_message` - An optional message to display if the assertion fails.
///
pub fn assert_float_eq(a: f64, b: f64, epsilon: f64, optional_message: Option<&str>) {
    match optional_message {
        Some(message) => assert!((a - b).abs() < epsilon, "a: {:?},\nb: {:?},\nepsilon: {:?},\n message: {:?}", a, b, epsilon, message),
        None => assert!((a - b).abs() < epsilon, "Expected {} to be approximately equal to {} (epsilon: {})", a, b, epsilon),
    }
}

// -------------------------- //
// local unit tests by module //
// -------------------------- //

#[cfg(test)]
mod physics_tests;
#[cfg(test)]
mod interactions_tests;
#[cfg(test)]
mod forces_tests;
#[cfg(feature = "rotational_dynamics")]
#[cfg(test)]
mod rotational_dynamics_tests;
#[cfg(feature = "fluid_dynamics")]
#[cfg(test)]
mod fluid_dynamics_tests;
#[cfg(feature = "thermodynamics")]
#[cfg(test)]
mod thermodynamics_tests;
#[cfg(feature = "constraints")]
#[cfg(test)]
mod constraint_solvers_tests;
#[cfg(feature = "fluid_simulation")]
#[cfg(test)]
mod fluid_simulation_tests;
#[cfg(feature = "materials")]
#[cfg(test)]
mod materials_tests;
