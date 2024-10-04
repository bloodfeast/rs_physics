// src/physics_tests.rs

use std::f64::consts::PI;
use crate::constants_config::PhysicsConstants;
use crate::{assert_float_eq, physics};

#[test]
fn test_create_constants() {
    let custom_constants = physics::create_constants(Some(9.8), Some(1.2), Some(340.0), Some(101000.0));
    assert_float_eq(custom_constants.gravity, 9.8, 1e-6, None);
    assert_float_eq(custom_constants.air_density, 1.2, 1e-6, None);
    assert_float_eq(custom_constants.speed_of_sound, 340.0, 1e-6, None);
    assert_float_eq(custom_constants.atmospheric_pressure, 101000.0, 1e-6, None);

    let partial_constants = physics::create_constants(Some(9.8), None, None, Some(101000.0));
    assert_float_eq(partial_constants.gravity, 9.8, 1e-6, None);
    assert_float_eq(partial_constants.air_density, PhysicsConstants::default().air_density, 1e-6, None);
    assert_float_eq(partial_constants.speed_of_sound, PhysicsConstants::default().speed_of_sound, 1e-6, None);
    assert_float_eq(partial_constants.atmospheric_pressure, 101000.0, 1e-6, None);
}

#[test]
fn test_terminal_velocity() {
    let constants = PhysicsConstants::default();
    let mass = 70.0; // kg
    let drag_coefficient = 0.7;
    let cross_sectional_area = 0.5; // m^2
    let terminal_velocity = physics::calculate_terminal_velocity(&constants, mass, drag_coefficient, cross_sectional_area).unwrap();
    let expected = ((2.0 * mass * constants.gravity) / (constants.air_density * drag_coefficient * cross_sectional_area)).sqrt();
    assert_float_eq(terminal_velocity, expected, 1e-6, None);
}

#[test]
fn test_terminal_velocity_invalid_inputs() {
    let constants = PhysicsConstants::default();

    // Test with zero mass
    assert!(physics::calculate_terminal_velocity(&constants, 0.0, 0.7, 0.5).is_err());

    // Test with negative drag coefficient
    assert!(physics::calculate_terminal_velocity(&constants, 70.0, -0.7, 0.5).is_err());

    // Test with zero cross-sectional area
    assert!(physics::calculate_terminal_velocity(&constants, 70.0, 0.7, 0.0).is_err());
}

#[test]
fn test_air_resistance() {
    let constants = PhysicsConstants::default();
    let velocity = 10.0; // m/s
    let drag_coefficient = 0.7;
    let cross_sectional_area = 0.5; // m^2
    let air_resistance = physics::calculate_air_resistance(&constants, velocity, drag_coefficient, cross_sectional_area).unwrap();
    let expected = 0.5 * constants.air_density * velocity * velocity * drag_coefficient * cross_sectional_area;
    assert_float_eq(air_resistance, expected, 1e-6, None);
}

#[test]
fn test_air_resistance_invalid_inputs() {
    let constants = PhysicsConstants::default();

    // Test with negative drag coefficient
    assert!(physics::calculate_air_resistance(&constants, 10.0, -0.7, 0.5).is_err());

    // Test with negative cross-sectional area
    assert!(physics::calculate_air_resistance(&constants, 10.0, 0.7, -0.5).is_err());
}

#[test]
fn test_acceleration() {
    let constants = PhysicsConstants::default();
    let force = 100.0; // N
    let mass = 10.0; // kg
    let acceleration = physics::calculate_acceleration(&constants, force, mass).unwrap();
    assert_float_eq(acceleration, 10.0, 1e-6, None);
}

#[test]
fn test_acceleration_invalid_inputs() {
    let constants = PhysicsConstants::default();

    // Test with zero mass
    assert!(physics::calculate_acceleration(&constants, 100.0, 0.0).is_err());

    // Test with negative mass
    let result = physics::calculate_acceleration(&constants, 100.0, -10.0);
    assert!(result.is_ok());
    assert!(result.unwrap().is_sign_negative());
}

#[test]
fn test_deceleration() {
    let constants = PhysicsConstants::default();
    let force = 100.0; // N
    let mass = 10.0; // kg
    let deceleration = physics::calculate_deceleration(&constants, force, mass).unwrap();
    assert_float_eq(deceleration, -10.0, 1e-6, None);
}

#[test]
fn test_deceleration_invalid_inputs() {
    let constants = PhysicsConstants::default();

    // Test with zero mass
    assert!(physics::calculate_deceleration(&constants, 100.0, 0.0).is_err());

    // Test with negative mass
    let result = physics::calculate_deceleration(&constants, 100.0, -10.0);
    assert!(result.is_ok());
    assert!(result.unwrap().is_sign_positive());
}

#[test]
fn test_force() {
    let constants = PhysicsConstants::default();
    let mass = 10.0; // kg
    let acceleration = 5.0; // m/s^2
    let force = physics::calculate_force(&constants, mass, acceleration).unwrap();
    assert_float_eq(force, 50.0, 1e-6, None);
}

#[test]
fn test_force_invalid_inputs() {
    let constants = PhysicsConstants::default();

    // Test with negative mass
    assert!(physics::calculate_force(&constants, -10.0, 5.0).is_err());
}

#[test]
fn test_momentum() {
    let constants = PhysicsConstants::default();
    let mass = 5.0; // kg
    let velocity = 10.0; // m/s
    let momentum = physics::calculate_momentum(&constants, mass, velocity).unwrap();
    assert_float_eq(momentum, 50.0, 1e-6, None);
}

#[test]
fn test_momentum_invalid_inputs() {
    let constants = PhysicsConstants::default();

    // Test with negative mass
    assert!(physics::calculate_momentum(&constants, -5.0, 10.0).is_err());
}

#[test]
fn test_velocity() {
    let constants = PhysicsConstants::default();
    let initial_velocity = 5.0; // m/s
    let acceleration = 2.0; // m/s^2
    let time = 3.0; // s
    let velocity = physics::calculate_velocity(&constants, initial_velocity, acceleration, time).unwrap();
    assert_float_eq(velocity, 11.0, 1e-6, None);
}

#[test]
fn test_velocity_invalid_inputs() {
    let constants = PhysicsConstants::default();

    // Test with negative time
    assert!(physics::calculate_velocity(&constants, 5.0, 2.0, -3.0).is_err());
}

#[test]
fn test_average_velocity() {
    let constants = PhysicsConstants::default();
    let displacement = 100.0; // m
    let time = 10.0; // s
    let avg_velocity = physics::calculate_average_velocity(&constants, displacement, time).unwrap();
    assert_float_eq(avg_velocity, 10.0, 1e-6, None);
}

#[test]
fn test_average_velocity_invalid_inputs() {
    let constants = PhysicsConstants::default();

    // Test with zero time
    assert!(physics::calculate_average_velocity(&constants, 100.0, 0.0).is_err());
}

#[test]
fn test_kinetic_energy() {
    let constants = PhysicsConstants::default();
    let mass = 2.0; // kg
    let velocity = 3.0; // m/s
    let kinetic_energy = physics::calculate_kinetic_energy(&constants, mass, velocity).unwrap();
    assert_float_eq(kinetic_energy, 9.0, 1e-6, None);
}

#[test]
fn test_kinetic_energy_invalid_inputs() {
    let constants = PhysicsConstants::default();

    // Test with negative mass
    assert!(physics::calculate_kinetic_energy(&constants, -2.0, 3.0).is_err());
}

#[test]
fn test_potential_energy() {
    let constants = PhysicsConstants::default();
    let mass = 5.0; // kg
    let height = 10.0; // m
    let potential_energy = physics::calculate_potential_energy(&constants, mass, height).unwrap();
    assert_float_eq(potential_energy, 5.0 * 10.0 * constants.gravity, 1e-6, None);
}

#[test]
fn test_potential_energy_invalid_inputs() {
    let constants = PhysicsConstants::default();

    // Test with negative mass
    assert!(physics::calculate_potential_energy(&constants, -5.0, 10.0).is_err());
}

#[test]
fn test_work() {
    let constants = PhysicsConstants::default();
    let force = 20.0; // N
    let displacement = 5.0; // m
    let work = physics::calculate_work(&constants, force, displacement).unwrap();
    assert_float_eq(work, 100.0, 1e-6, None);
}

#[test]
fn test_work_invalid_inputs() {
    let constants = PhysicsConstants::default();

    // Work calculation doesn't have invalid inputs in our current implementation
    let work = physics::calculate_work(&constants, -20.0, -5.0);
    assert!(work.is_ok());
}

#[test]
fn test_power() {
    let constants = PhysicsConstants::default();
    let work = 1000.0; // J
    let time = 10.0; // s
    let power = physics::calculate_power(&constants, work, time).unwrap();
    assert_float_eq(power, 100.0, 1e-6, None);
}

#[test]
fn test_power_invalid_inputs() {
    let constants = PhysicsConstants::default();

    // Test with zero time
    assert!(physics::calculate_power(&constants, 1000.0, 0.0).is_err());
}

#[test]
fn test_impulse() {
    let constants = PhysicsConstants::default();
    let force = 50.0; // N
    let time = 0.1; // s
    let impulse = physics::calculate_impulse(&constants, force, time).unwrap();
    assert_float_eq(impulse, 5.0, 1e-6, None);
}

#[test]
fn test_impulse_invalid_inputs() {
    let constants = PhysicsConstants::default();

    // Test with negative time
    assert!(physics::calculate_impulse(&constants, 50.0, -0.1).is_err());
}

#[test]
fn test_coefficient_of_restitution() {
    let constants = PhysicsConstants::default();
    let velocity_before = -5.0; // m/s
    let velocity_after = 3.0; // m/s
    let cor = physics::calculate_coefficient_of_restitution(&constants, velocity_before, velocity_after).unwrap();
    assert_float_eq(cor, 0.6, 1e-6, None);
}

#[test]
fn test_coefficient_of_restitution_invalid_inputs() {
    let constants = PhysicsConstants::default();

    // Test with zero velocity before collision
    assert!(physics::calculate_coefficient_of_restitution(&constants, 0.0, 3.0).is_err());
}

#[test]
fn test_projectile_time_of_flight() {
    let constants = PhysicsConstants::default();
    let initial_velocity = 50.0; // m/s
    let angle = PI / 4.0; // 45 degrees in radians
    let time = physics::calculate_projectile_time_of_flight(&constants, initial_velocity, angle).unwrap();
    let expected_time = 2.0 * initial_velocity * (PI / 4.0).sin() / constants.gravity;
    assert_float_eq(time, expected_time, 1e-6, None);
}

#[test]
fn test_projectile_max_height() {
    let constants = PhysicsConstants::default();
    let initial_velocity = 50.0; // m/s
    let angle = PI / 4.0; // 45 degrees in radians
    let height = physics::calculate_projectile_max_height(&constants, initial_velocity, angle).unwrap();
    let expected_height = (initial_velocity * (PI / 4.0).sin()).powi(2) / (2.0 * constants.gravity);
    assert_float_eq(height, expected_height, 1e-6, None);
}

#[test]
fn test_projectile_motion_invalid_inputs() {
    let constants = PhysicsConstants::default();

    // Test with negative initial velocity
    assert!(physics::calculate_projectile_time_of_flight(&constants, -50.0, PI / 4.0).is_err());
    assert!(physics::calculate_projectile_max_height(&constants, -50.0, PI / 4.0).is_err());

    // Test with angle outside [0, pi/2] range
    assert!(physics::calculate_projectile_time_of_flight(&constants, 50.0, 3.0 * PI / 4.0).is_err());
    assert!(physics::calculate_projectile_max_height(&constants, 50.0, 3.0 * PI / 4.0).is_err());
}

#[test]
fn test_centripetal_force() {
    let constants = PhysicsConstants::default();
    let mass = 2.0; // kg
    let velocity = 5.0; // m/s
    let radius = 3.0; // m
    let force = physics::calculate_centripetal_force(&constants, mass, velocity, radius).unwrap();
    assert_float_eq(force, (2.0 * 5.0 * 5.0) / 3.0, 1e-6, None);
}

#[test]
fn test_centripetal_force_invalid_inputs() {
    let constants = PhysicsConstants::default();

    // Test with negative mass
    assert!(physics::calculate_centripetal_force(&constants, -2.0, 5.0, 3.0).is_err());

    // Test with zero radius
    assert!(physics::calculate_centripetal_force(&constants, 2.0, 5.0, 0.0).is_err());
}

#[test]
fn test_torque() {
    let constants = PhysicsConstants::default();
    let force = 10.0; // N
    let lever_arm = 2.0; // m
    let angle = PI / 2.0; // 90 degrees in radians
    let torque = physics::calculate_torque(&constants, force, lever_arm, angle).unwrap();
    assert_float_eq(torque, 20.0, 1e-6, None);
}

#[test]
fn test_torque_invalid_inputs() {
    let constants = PhysicsConstants::default();

    // Test with negative lever arm
    assert!(physics::calculate_torque(&constants, 10.0, -2.0, PI / 2.0).is_err());
}

#[test]
fn test_angular_velocity() {
    let constants = PhysicsConstants::default();
    let linear_velocity = 10.0; // m/s
    let radius = 2.0; // m
    let angular_velocity = physics::calculate_angular_velocity(&constants, linear_velocity, radius).unwrap();
    assert_float_eq(angular_velocity, 5.0, 1e-6, None);
}

#[test]
fn test_angular_velocity_invalid_inputs() {
    let constants = PhysicsConstants::default();

    // Test with zero radius
    assert!(physics::calculate_angular_velocity(&constants, 10.0, 0.0).is_err());
}
