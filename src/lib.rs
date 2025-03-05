#![feature(avx512_target_feature)]
#![feature(stdarch_x86_avx512)]
pub mod utils;
pub mod apis;
pub mod physics;
pub mod interactions;
pub mod forces;
pub mod constraints;
pub mod rotational_dynamics;
pub mod fluid_dynamics;
pub mod thermodynamics;
pub mod materials;
pub mod models;
pub mod particles;

/// ### General helper function
/// - Asserts that two floating point numbers are approximately equal.
///
/// ### Arguments
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

