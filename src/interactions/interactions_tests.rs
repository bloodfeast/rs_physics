// src/interactions_tests.rs

use std::f64::consts::PI;
use crate::assert_float_eq;
use crate::utils::PhysicsConstants;
use crate::interactions::{elastic_collision, gravitational_force, apply_force};
use crate::models::Object;

#[test]
fn test_object_creation() {
    let obj = Object::new(1.0, 2.0, 3.0).unwrap();
    assert_float_eq(obj.mass, 1.0, 1e-6, None);
    assert_float_eq(obj.velocity, 2.0, 1e-6, None);
    assert_float_eq(obj.position, 3.0, 1e-6, None);

    assert!(Object::new(-1.0, 2.0, 3.0).is_err());
}

#[test]
fn test_elastic_collision() {
    let constants = PhysicsConstants::default();
    let mut obj1 = Object::new(1.0, 1.0, 0.0).unwrap();
    let mut obj2 = Object::new(1.0, -1.0, 1.0).unwrap();
    let drag_coefficient = 0.47;
    let cross_sectional_area = 1.0;

    // Horizontal collision (angle = 0)
    elastic_collision(&constants, &mut obj1, &mut obj2, 0.0, 0.001, drag_coefficient, cross_sectional_area).unwrap();

    // For a horizontal collision with equal masses, objects should exchange velocities
    // We'll use a slightly larger epsilon to account for small numerical effects and air resistance
    assert_float_eq(obj1.velocity, -1.0, 1e-2, Some("Horizontal collision, object 1"));
    assert_float_eq(obj2.velocity, 1.0, 1e-2, Some("Horizontal collision, object 2"));

    // Reset objects
    obj1 = Object::new(1.0, 1.0, 0.0).unwrap();
    obj2 = Object::new(1.0, -1.0, 1.0).unwrap();

    // Collision at 45 degrees upward
    elastic_collision(&constants, &mut obj1, &mut obj2, PI / 4.0, 0.1, drag_coefficient, cross_sectional_area).unwrap();

    // Calculate expected velocities
    let gravity_effect = constants.gravity * 0.1 * (PI / 4.0).sin();
    let air_resistance = |v: f64| -> f64 {
        let air_resistance_force = constants.calculate_air_resistance(v.abs(), drag_coefficient, cross_sectional_area).unwrap();
        let deceleration = air_resistance_force / 1.0; // mass is 1.0
        -deceleration * 0.1 * v.signum()
    };

    let expected_v1 = -1.0 - gravity_effect + air_resistance(-1.0);
    let expected_v2 = 1.0 - gravity_effect + air_resistance(1.0);

    assert_float_eq(obj1.velocity, expected_v1, 1e-3, Some("45 degree collision, object 1"));
    assert_float_eq(obj2.velocity, expected_v2, 1e-3, Some("45 degree collision, object 2"));
}

#[test]
fn test_elastic_collision_with_different_masses() {
    let constants = PhysicsConstants::default();
    let mut obj1 = Object::new(1.0, 2.0, 0.0).unwrap();
    let mut obj2 = Object::new(2.0, -1.0, 1.0).unwrap();
    let drag_coefficient = 0.47;
    let cross_sectional_area = 1.0;

    elastic_collision(&constants, &mut obj1, &mut obj2, 0.0, 0.001, drag_coefficient, cross_sectional_area).unwrap();

    // Calculate expected velocities
    let m1 = 1.0;
    let m2 = 2.0;
    let v1 = 2.0;
    let v2 = -1.0;
    let expected_v1 = ((m1 - m2) * v1 + 2.0 * m2 * v2) / (m1 + m2);
    let expected_v2 = ((m2 - m1) * v2 + 2.0 * m1 * v1) / (m1 + m2);

    // Account for minor air resistance effects
    assert_float_eq(obj1.velocity, expected_v1, 1e-2, Some("Different masses, object 1"));
    assert_float_eq(obj2.velocity, expected_v2, 1e-2, Some("Different masses, object 2"));
}

#[test]
fn test_elastic_collision_with_long_duration() {
    let constants = PhysicsConstants::default();
    let mut obj1 = Object::new(1.0, 1.0, 0.0).unwrap();
    let mut obj2 = Object::new(1.0, -1.0, 1.0).unwrap();
    let drag_coefficient = 0.47;
    let cross_sectional_area = 1.0;

    elastic_collision(&constants, &mut obj1, &mut obj2, PI / 4.0, 1.0, drag_coefficient, cross_sectional_area).unwrap();

    // Calculate expected velocities
    let m1 = obj1.mass;
    let m2 = obj2.mass;
    let v1 = 1.0; // Initial velocity of obj1
    let v2 = -1.0; // Initial velocity of obj2

    // Calculate velocities after collision (ignoring external forces)
    let v1_final = ((m1 - m2) * v1 + 2.0 * m2 * v2) / (m1 + m2);
    let v2_final = ((m2 - m1) * v2 + 2.0 * m1 * v1) / (m1 + m2);

    // Apply gravity effect
    let gravity_effect = constants.gravity * 1.0 * (PI / 4.0).sin();

    // Apply air resistance
    let air_resistance = |v: f64, m: f64| -> f64 {
        let air_resistance_force = constants.calculate_air_resistance(v.abs(), drag_coefficient, cross_sectional_area).unwrap();
        let deceleration = air_resistance_force / m;
        -deceleration * 1.0 * v.signum()
    };

    // Calculate final velocities considering gravity and air resistance
    let expected_v1 = v1_final - gravity_effect + air_resistance(v1_final, m1);
    let expected_v2 = v2_final - gravity_effect + air_resistance(v2_final, m2);

    assert_float_eq(obj1.velocity, expected_v1, 1e-3, Some("Long duration collision, object 1"));
    assert_float_eq(obj2.velocity, expected_v2, 1e-3, Some("Long duration collision, object 2"));
}

#[test]
fn test_gravitational_force() {
    let constants = PhysicsConstants::default();
    let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();
    let obj2 = Object::new(1.0, 0.0, 1.0).unwrap();

    let force = gravitational_force(&constants, &obj1, &obj2).unwrap();
    assert_float_eq(force, constants.gravity, 1e-6, None);

    let obj3 = Object::new(1.0, 0.0, 0.0).unwrap();
    assert!(gravitational_force(&constants, &obj1, &obj3).is_err());
}

#[test]
fn test_apply_force() {
    let constants = PhysicsConstants::default();
    let mut obj = Object::new(1.0, 0.0, 0.0).unwrap();

    apply_force(&constants, &mut obj, 1.0, 1.0).unwrap();

    assert_float_eq(obj.velocity, 1.0, 1e-6, None);
    assert_float_eq(obj.position, 0.5, 1e-6, None);
}

#[test]
fn test_object_creation_failure() {
    assert!(Object::new(-1.0, 0.0, 0.0).is_err(), "Should fail with negative mass");
    assert!(Object::new(0.0, 0.0, 0.0).is_err(), "Should fail with zero mass");
}

#[test]
fn test_gravitational_force_failure() {
    let constants = PhysicsConstants::default();
    let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();
    let obj2 = Object::new(1.0, 0.0, 0.0).unwrap();

    assert!(gravitational_force(&constants, &obj1, &obj2).is_err(), "Should fail when objects are at the same position");
}

#[test]
fn test_apply_force_failure() {
    let constants = PhysicsConstants::default();
    let mut obj = Object::new(1.0, 0.0, 0.0).unwrap();

    assert!(apply_force(&constants, &mut obj, 1.0, -1.0).is_err(), "Should fail with negative time");
}

#[test]
fn test_elastic_collision_invalid_inputs() {
    let constants = PhysicsConstants::default();
    let mut obj1 = Object::new(1.0, 1.0, 0.0).unwrap();
    let mut obj2 = Object::new(1.0, -1.0, 1.0).unwrap();
    let drag_coefficient = 0.47;
    let cross_sectional_area = 1.0;

    // Test with zero duration
    assert!(elastic_collision(&constants, &mut obj1, &mut obj2, 0.0, 0.0, drag_coefficient, cross_sectional_area).is_err());

    // Test with negative duration
    assert!(elastic_collision(&constants, &mut obj1, &mut obj2, 0.0, -1.0, drag_coefficient, cross_sectional_area).is_err());

    // Test with negative drag coefficient
    assert!(elastic_collision(&constants, &mut obj1, &mut obj2, 0.0, 1.0, -0.5, cross_sectional_area).is_err());

    // Test with negative cross-sectional area
    assert!(elastic_collision(&constants, &mut obj1, &mut obj2, 0.0, 1.0, drag_coefficient, -1.0).is_err());
}