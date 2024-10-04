// src/interactions_tests.rs

use std::f64::consts::PI;
use crate::assert_float_eq;
use crate::constants_config::PhysicsConstants;
use crate::interactions::{Object, elastic_collision, gravitational_force, apply_force};

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

    // Horizontal collision (angle = 0)
    elastic_collision(&constants, &mut obj1, &mut obj2, 0.0, 0.001).unwrap();

    // For a horizontal collision with equal masses, objects should exchange velocities
    // We'll use a slightly larger epsilon to account for small numerical effects
    assert_float_eq(obj1.velocity, -1.0, 1e-3, Some("Horizontal collision, object 1"));
    assert_float_eq(obj2.velocity, 1.0, 1e-3, Some("Horizontal collision, object 2"));

    // Print values for investigation
    println!("Horizontal collision, object 1 velocity: {}", obj1.velocity);
    println!("Horizontal collision, object 2 velocity: {}", obj2.velocity);

    // Reset objects
    obj1 = Object::new(1.0, 1.0, 0.0).unwrap();
    obj2 = Object::new(1.0, -1.0, 1.0).unwrap();

    // Collision at 45 degrees upward
    elastic_collision(&constants, &mut obj1, &mut obj2, PI / 4.0, 0.1).unwrap();

    // Calculate expected velocities
    let gravity_effect = constants.gravity * 0.1 * (PI / 4.0).sin();
    let air_resistance = |v: f64| -> f64 {
        let drag_coefficient = 0.47;
        let cross_sectional_area = 1.0;
        let drag_force = 0.5 * constants.air_density * v.powi(2) * drag_coefficient * cross_sectional_area;
        -drag_force * 0.1
    };

    let expected_v1 = -1.0 - gravity_effect + air_resistance(-1.0);
    let expected_v2 = 1.0 - gravity_effect + air_resistance(1.0);

    assert_float_eq(obj1.velocity, expected_v1, 1e-3, Some("45 degree collision, object 1"));
    assert_float_eq(obj2.velocity, expected_v2, 1e-3, Some("45 degree collision, object 2"));

    println!("45 degree collision, object 1 velocity: {}", obj1.velocity);
    println!("45 degree collision, object 2 velocity: {}", obj2.velocity);
    println!("Expected velocity for object 1: {}", expected_v1);
    println!("Expected velocity for object 2: {}", expected_v2);
}

#[test]
fn test_elastic_collision_with_different_masses() {
    let constants = PhysicsConstants::default();
    let mut obj1 = Object::new(1.0, 2.0, 0.0).unwrap();
    let mut obj2 = Object::new(2.0, -1.0, 1.0).unwrap();

    elastic_collision(&constants, &mut obj1, &mut obj2, 0.0, 0.001).unwrap();

    // Calculate expected velocities
    let m1 = 1.0;
    let m2 = 2.0;
    let v1 = 2.0;
    let v2 = -1.0;
    let expected_v1 = ((m1 - m2) * v1 + 2.0 * m2 * v2) / (m1 + m2);
    let expected_v2 = ((m2 - m1) * v2 + 2.0 * m1 * v1) / (m1 + m2);

    assert_float_eq(obj1.velocity, expected_v1, 1e-2, Some("Different masses, object 1"));
    assert_float_eq(obj2.velocity, expected_v2, 1e-2, Some("Different masses, object 2"));

    // Print actual and expected values for verification
    println!("Object 1: Expected velocity: {}, Actual velocity: {}", expected_v1, obj1.velocity);
    println!("Object 2: Expected velocity: {}, Actual velocity: {}", expected_v2, obj2.velocity);
}

#[test]
fn test_elastic_collision_with_long_duration() {
    let constants = PhysicsConstants::default();
    let mut obj1 = Object::new(1.0, 1.0, 0.0).unwrap();
    let mut obj2 = Object::new(1.0, -1.0, 1.0).unwrap();

    elastic_collision(&constants, &mut obj1, &mut obj2, PI / 4.0, 1.0).unwrap();

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

    // Apply air resistance (simplified model)
    let air_resistance = |v: f64, m: f64| -> f64 {
        let drag_coefficient = 0.47;
        let cross_sectional_area = 1.0;
        let drag_force = 0.5 * constants.air_density * v.powi(2) * drag_coefficient * cross_sectional_area;
        let deceleration = drag_force / m;
        -deceleration * 1.0
    };

    // Calculate final velocities considering gravity and air resistance
    let expected_v1 = v1_final - gravity_effect + air_resistance(v1_final, m1);
    let expected_v2 = v2_final - gravity_effect + air_resistance(v2_final, m2);

    println!("Long duration collision, object 1 velocity: {}", obj1.velocity);
    println!("Long duration collision, object 2 velocity: {}", obj2.velocity);
    println!("Expected velocity for object 1: {}", expected_v1);
    println!("Expected velocity for object 2: {}", expected_v2);

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
fn test_elastic_collision_failure() {
    let constants = PhysicsConstants::default();
    let mut obj1 = Object::new(1.0, 1.0, 0.0).unwrap();
    let mut obj2 = Object::new(1.0, -1.0, 1.0).unwrap();

    assert!(elastic_collision(&constants, &mut obj1, &mut obj2, 0.0, 0.0).is_err(), "Should fail with zero duration");
    assert!(elastic_collision(&constants, &mut obj1, &mut obj2, 0.0, -1.0).is_err(), "Should fail with negative duration");
}