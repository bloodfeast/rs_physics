// src/lib.rs

mod errors;
pub mod apis;
mod constants_config;
pub mod physics;
pub mod interactions;
pub mod forces;
pub mod rotational_dynamics;
pub mod fluid_dynamics;
pub mod thermodynamics;

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
#[cfg(test)]
mod rotational_dynamics_tests;
#[cfg(test)]
mod fluid_dynamics_tests;
#[cfg(test)]
mod thermodynamics_tests;
