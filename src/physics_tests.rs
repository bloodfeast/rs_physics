use std::f64::consts::PI;
use std::cell::RefCell;
use log::{Record, Level, Metadata, LevelFilter};
use crate::physics::*;
use crate::constants_config::PhysicsConstants;

// Helper function to assert floating point equality (didn't want to pass a `None` message to assert_float_eq)
fn assert_float_eq(a: f64, b: f64, epsilon: f64) {
    assert!((a - b).abs() < epsilon, "Expected {} to be approximately equal to {} (epsilon: {})", a, b, epsilon);
}

thread_local! {
    static LOG_MESSAGES: RefCell<Vec<String>> = RefCell::new(Vec::new());
}

struct ThreadLocalLogger;

impl log::Log for ThreadLocalLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Warn
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            let message = format!("{}: {}", record.level(), record.args());
            LOG_MESSAGES.with(|messages| messages.borrow_mut().push(message));
        }
    }

    fn flush(&self) {}
}

fn setup_test_logger() {
    let logger = Box::new(ThreadLocalLogger);
    log::set_boxed_logger(logger).ok(); // It's okay if this fails (logger already set)
    log::set_max_level(LevelFilter::Warn);
}

fn clear_log_messages() {
    LOG_MESSAGES.with(|messages| messages.borrow_mut().clear());
}

fn get_log_messages() -> Vec<String> {
    LOG_MESSAGES.with(|messages| messages.borrow().clone())
}

#[test]
fn test_create_constants() {
    let custom_constants = create_constants(Some(9.8), Some(1.2), Some(340.0), Some(101000.0));
    assert_float_eq(custom_constants.gravity, 9.8, 1e-6);
    assert_float_eq(custom_constants.air_density, 1.2, 1e-6);
    assert_float_eq(custom_constants.speed_of_sound, 340.0, 1e-6);
    assert_float_eq(custom_constants.atmospheric_pressure, 101000.0, 1e-6);

    let partial_constants = create_constants(Some(9.8), None, None, Some(101000.0));
    assert_float_eq(partial_constants.gravity, 9.8, 1e-6);
    assert_float_eq(partial_constants.air_density, PhysicsConstants::default().air_density, 1e-6);
    assert_float_eq(partial_constants.speed_of_sound, PhysicsConstants::default().speed_of_sound, 1e-6);
    assert_float_eq(partial_constants.atmospheric_pressure, 101000.0, 1e-6);
}

#[test]
fn test_terminal_velocity() {
    let constants = create_constants(None, None, None, None);
    let mass = 70.0; // kg
    let drag_coefficient = 0.7;
    let cross_sectional_area = 0.5; // m^2
    let terminal_velocity = calculate_terminal_velocity(&constants, mass, drag_coefficient, cross_sectional_area);
    let expected = ((2.0 * mass * constants.gravity) / (constants.air_density * drag_coefficient * cross_sectional_area)).sqrt();
    assert_float_eq(terminal_velocity, expected, 1e-6);
}

#[test]
fn test_terminal_velocity_invalid_inputs() {
    let constants = create_constants(None, None, None, None);
    setup_test_logger();
    clear_log_messages();

    // Test with negative mass
    let velocity = calculate_terminal_velocity(&constants, -70.0, 0.7, 0.5);
    assert!(velocity > 0.0);

    // Test with negative drag coefficient
    let velocity = calculate_terminal_velocity(&constants, 70.0, -0.7, 0.5);
    assert!(velocity > 0.0);

    // Test with negative cross-sectional area
    let velocity = calculate_terminal_velocity(&constants, 70.0, 0.7, -0.5);
    assert!(velocity > 0.0);

    // Check log messages
    let messages = get_log_messages();
    assert!(messages.iter().any(|m| m.contains("Using absolute value of mass")));
    assert!(messages.iter().any(|m| m.contains("Using absolute value of drag coefficient")));
    assert!(messages.iter().any(|m| m.contains("Using absolute value of cross-sectional area")));
}

#[test]
fn test_air_resistance() {
    let constants = create_constants(None, None, None, None);
    let velocity = 10.0; // m/s
    let drag_coefficient = 0.7;
    let cross_sectional_area = 0.5; // m^2
    let air_resistance = calculate_air_resistance(&constants, velocity, drag_coefficient, cross_sectional_area);
    let expected = 0.5 * constants.air_density * velocity * velocity * drag_coefficient * cross_sectional_area;
    assert_float_eq(air_resistance, expected, 1e-6);
}

#[test]
fn test_air_resistance_invalid_inputs() {
    let constants = create_constants(None, None, None, None);
    setup_test_logger();
    clear_log_messages();

    // Test with negative drag coefficient
    let resistance = calculate_air_resistance(&constants, 10.0, -0.7, 0.5);
    assert!(resistance > 0.0);

    // Test with negative cross-sectional area
    let resistance = calculate_air_resistance(&constants, 10.0, 0.7, -0.5);
    assert!(resistance > 0.0);

    // Check log messages
    let messages = get_log_messages();
    assert!(messages.iter().any(|m| m.contains("Using absolute value of drag coefficient")));
    assert!(messages.iter().any(|m| m.contains("Using absolute value of cross-sectional area")));
}

#[test]
fn test_acceleration() {
    let constants = create_constants(None, None, None, None);
    let force = 100.0; // N
    let mass = 10.0; // kg
    let acceleration = calculate_acceleration(&constants, force, mass);
    assert_float_eq(acceleration, 10.0, 1e-6);
}

#[test]
fn test_acceleration_invalid_inputs() {
    let constants = create_constants(None, None, None, None);

    // Test with zero mass
    let acceleration = calculate_acceleration(&constants, 100.0, 0.0);
    assert_float_eq(acceleration, 0.0, 1e-6);

    // Test with negative mass
    let acceleration = calculate_acceleration(&constants, 100.0, -10.0);
    assert_float_eq(acceleration, 10.0, 1e-6); // Should use absolute value of mass
}

#[test]
fn test_kinetic_energy() {
    let constants = create_constants(None, None, None, None);
    let mass = 2.0; // kg
    let velocity = 3.0; // m/s
    let kinetic_energy = calculate_kinetic_energy(&constants, mass, velocity);
    assert_float_eq(kinetic_energy, 9.0, 1e-6);
}

#[test]
fn test_kinetic_energy_invalid_inputs() {
    let constants = create_constants(None, None, None, None);

    // Test with negative mass
    let energy = calculate_kinetic_energy(&constants, -2.0, 3.0);
    assert_float_eq(energy, 9.0, 1e-6); // Should use absolute value of mass
}

#[test]
fn test_potential_energy() {
    let constants = create_constants(Some(9.8), None, None, None);
    let mass = 2.0; // kg
    let height = 5.0; // m
    let potential_energy = calculate_potential_energy(&constants, mass, height);
    assert_float_eq(potential_energy, 98.0, 1e-6);
}

#[test]
fn test_potential_energy_invalid_inputs() {
    let constants = create_constants(Some(9.8), None, None, None);

    // Test with negative mass
    let energy = calculate_potential_energy(&constants, -2.0, 5.0);
    assert_float_eq(energy, 98.0, 1e-6); // Should use absolute value of mass
}

#[test]
fn test_work() {
    let constants = create_constants(None, None, None, None);
    let force = 10.0; // N
    let displacement = 5.0; // m
    let work = calculate_work(&constants, force, displacement);
    assert_float_eq(work, 50.0, 1e-6);
}

#[test]
fn test_power() {
    let constants = create_constants(None, None, None, None);
    let work = 1000.0; // J
    let time = 10.0; // s
    let power = calculate_power(&constants, work, time);
    assert_float_eq(power, 100.0, 1e-6);
}

#[test]
fn test_power_invalid_inputs() {
    let constants = create_constants(None, None, None, None);

    // Test with zero time
    let power = calculate_power(&constants, 1000.0, 0.0);
    assert_float_eq(power, 0.0, 1e-6);
}

#[test]
fn test_impulse() {
    let constants = create_constants(None, None, None, None);
    let force = 50.0; // N
    let time = 0.1; // s
    let impulse = calculate_impulse(&constants, force, time);
    assert_float_eq(impulse, 5.0, 1e-6);
}

#[test]
fn test_impulse_invalid_inputs() {
    let constants = create_constants(None, None, None, None);

    // Test with negative time
    let impulse = calculate_impulse(&constants, 50.0, -0.1);
    assert_float_eq(impulse, 5.0, 1e-6); // Should use absolute value of time
}

#[test]
fn test_coefficient_of_restitution() {
    let constants = create_constants(None, None, None, None);
    let velocity_before = -5.0; // m/s
    let velocity_after = 3.0; // m/s
    let cor = calculate_coefficient_of_restitution(&constants, velocity_before, velocity_after);
    assert_float_eq(cor, 0.6, 1e-6);
}

#[test]
fn test_coefficient_of_restitution_invalid_inputs() {
    let constants = create_constants(None, None, None, None);

    // Test with zero velocity before collision
    let cor = calculate_coefficient_of_restitution(&constants, 0.0, 3.0);
    assert_float_eq(cor, 0.0, 1e-6);
}

#[test]
fn test_projectile_motion() {
    let constants = create_constants(Some(9.8), None, None, None);
    let initial_velocity = 50.0; // m/s
    let angle = PI / 4.0; // 45 degrees in radians

    let time = calculate_projectile_time_of_flight(&constants, initial_velocity, angle);
    let expected_time = 2.0 * initial_velocity * (PI / 4.0).sin() / constants.gravity;
    assert_float_eq(time, expected_time, 1e-6);

    let height = calculate_projectile_max_height(&constants, initial_velocity, angle);
    let expected_height = (initial_velocity * (PI / 4.0).sin()).powi(2) / (2.0 * constants.gravity);
    assert_float_eq(height, expected_height, 1e-6);
}

#[test]
fn test_projectile_motion_invalid_inputs() {
    let constants = create_constants(Some(9.8), None, None, None);

    // Test with negative initial velocity
    let time = calculate_projectile_time_of_flight(&constants, -50.0, PI / 4.0);
    assert!(time > 0.0);

    let height = calculate_projectile_max_height(&constants, -50.0, PI / 4.0);
    assert!(height > 0.0);

    // Test with angle outside [0, pi/2] range
    let time = calculate_projectile_time_of_flight(&constants, 50.0, 3.0 * PI / 4.0);
    assert!(time > 0.0);

    let height = calculate_projectile_max_height(&constants, 50.0, 3.0 * PI / 4.0);
    assert!(height > 0.0);
}

#[test]
fn test_centripetal_force() {
    let constants = create_constants(None, None, None, None);
    let mass = 2.0; // kg
    let velocity = 5.0; // m/s
    let radius = 3.0; // m
    let force = calculate_centripetal_force(&constants, mass, velocity, radius);
    assert_float_eq(force, (2.0 * 5.0 * 5.0) / 3.0, 1e-6);
}

#[test]
fn test_centripetal_force_invalid_inputs() {
    let constants = create_constants(None, None, None, None);

    // Test with negative mass
    let force = calculate_centripetal_force(&constants, -2.0, 5.0, 3.0);
    assert_float_eq(force, (2.0 * 5.0 * 5.0) / 3.0, 1e-6);

    // Test with zero radius
    let force = calculate_centripetal_force(&constants, 2.0, 5.0, 0.0);
    assert_float_eq(force, 0.0, 1e-6);
}

#[test]
fn test_torque() {
    let constants = create_constants(None, None, None, None);
    let force = 10.0; // N
    let lever_arm = 2.0; // m
    let angle = PI / 2.0; // 90 degrees in radians
    let torque = calculate_torque(&constants, force, lever_arm, angle);
    assert_float_eq(torque, 20.0, 1e-6);
}

#[test]
fn test_torque_invalid_inputs() {
    let constants = create_constants(None, None, None, None);

    // Test with negative lever arm
    let torque = calculate_torque(&constants, 10.0, -2.0, PI / 2.0);
    assert_float_eq(torque, 20.0, 1e-6);
}

#[test]
fn test_angular_velocity() {
    let constants = create_constants(None, None, None, None);
    let linear_velocity = 10.0; // m/s
    let radius = 2.0; // m
    let angular_velocity = calculate_angular_velocity(&constants, linear_velocity, radius);
    assert_float_eq(angular_velocity, 5.0, 1e-6);
}

#[test]
fn test_angular_velocity_invalid_inputs() {
    let constants = create_constants(None, None, None, None);

    // Test with zero radius
    let angular_velocity = calculate_angular_velocity(&constants, 10.0, 0.0);
    assert_float_eq(angular_velocity, 0.0, 1e-6);
}