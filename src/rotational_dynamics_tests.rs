// src/rotational_dynamics_tests.rs

use crate::assert_float_eq;
use crate::rotational_dynamics::{apply_torque, calculate_angular_momentum, calculate_moment_of_inertia, calculate_rotational_kinetic_energy, ObjectShape, RotationalObject};

#[test]
fn test_rotational_object_creation() {
    let obj = RotationalObject::new(1.0, 2.0);
    assert!(obj.is_ok());
    let obj = obj.unwrap();
    assert_float_eq(obj.mass, 1.0, 1e-6, Some("Mass should be 1.0"));
    assert_float_eq(obj.radius, 2.0, 1e-6, Some("Radius should be 2.0"));
    assert_float_eq(obj.angular_velocity, 0.0, 1e-6, Some("Initial angular velocity should be 0.0"));
    assert_float_eq(obj.moment_of_inertia, 2.0, 1e-6, Some("Moment of inertia should be 2.0"));
}

#[test]
fn test_rotational_object_creation_invalid_inputs() {
    assert!(RotationalObject::new(-1.0, 2.0).is_err(), "Should fail with negative mass");
    assert!(RotationalObject::new(0.0, 2.0).is_err(), "Should fail with zero mass");
    assert!(RotationalObject::new(1.0, -2.0).is_err(), "Should fail with negative radius");
    assert!(RotationalObject::new(1.0, 0.0).is_err(), "Should fail with zero radius");
}

#[test]
fn test_calculate_angular_momentum() {
    let obj = RotationalObject {
        mass: 1.0,
        radius: 2.0,
        angular_velocity: 3.0,
        moment_of_inertia: 4.0,
    };
    let angular_momentum = calculate_angular_momentum(&obj);
    assert_float_eq(angular_momentum, 12.0, 1e-6, Some("Angular momentum should be 12.0"));
}

#[test]
fn test_calculate_rotational_kinetic_energy() {
    let obj = RotationalObject {
        mass: 1.0,
        radius: 2.0,
        angular_velocity: 3.0,
        moment_of_inertia: 4.0,
    };
    let kinetic_energy = calculate_rotational_kinetic_energy(&obj);
    assert_float_eq(kinetic_energy, 18.0, 1e-6, Some("Rotational kinetic energy should be 18.0"));
}

#[test]
fn test_apply_torque() {
    let mut obj = RotationalObject {
        mass: 1.0,
        radius: 2.0,
        angular_velocity: 3.0,
        moment_of_inertia: 4.0,
    };
    let result = apply_torque(&mut obj, 8.0, 2.0);
    assert!(result.is_ok(), "Apply torque should succeed");
    assert_float_eq(obj.angular_velocity, 7.0, 1e-6, Some("Angular velocity should be 7.0 after applying torque"));
}

#[test]
fn test_apply_torque_invalid_time() {
    let mut obj = RotationalObject::new(1.0, 2.0).unwrap();
    assert!(apply_torque(&mut obj, 8.0, 0.0).is_err(), "Should fail with zero time");
    assert!(apply_torque(&mut obj, 8.0, -1.0).is_err(), "Should fail with negative time");
}

#[test]
fn test_calculate_moment_of_inertia() {
    let mass = 2.0;
    let dimension = 3.0;

    let solid_sphere = calculate_moment_of_inertia(&ObjectShape::SolidSphere, mass, dimension);
    assert_float_eq(solid_sphere.unwrap(), 7.2, 1e-6, Some("Solid sphere moment of inertia"));

    let hollow_sphere = calculate_moment_of_inertia(&ObjectShape::HollowSphere, mass, dimension);
    assert_float_eq(hollow_sphere.unwrap(), 12.0, 1e-6, Some("Hollow sphere moment of inertia"));

    let solid_cylinder = calculate_moment_of_inertia(&ObjectShape::SolidCylinder, mass, dimension);
    assert_float_eq(solid_cylinder.unwrap(), 9.0, 1e-6, Some("Solid cylinder moment of inertia"));

    let rod = calculate_moment_of_inertia(&ObjectShape::Rod, mass, dimension);
    assert_float_eq(rod.unwrap(), 1.5, 1e-6, Some("Rod moment of inertia"));
}

#[test]
fn test_calculate_moment_of_inertia_invalid_inputs() {
    assert!(calculate_moment_of_inertia(&ObjectShape::SolidSphere, -1.0, 1.0).is_err(), "Should fail with negative mass");
    assert!(calculate_moment_of_inertia(&ObjectShape::SolidSphere, 0.0, 1.0).is_err(), "Should fail with zero mass");
    assert!(calculate_moment_of_inertia(&ObjectShape::SolidSphere, 1.0, -1.0).is_err(), "Should fail with negative dimension");
    assert!(calculate_moment_of_inertia(&ObjectShape::SolidSphere, 1.0, 0.0).is_err(), "Should fail with zero dimension");
}