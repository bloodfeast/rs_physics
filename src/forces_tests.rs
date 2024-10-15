// src/forces_tests.rs

use crate::constants_config::PhysicsConstants;
use crate::forces::{Force, PhysicsSystem};
use crate::interactions::Object;
use crate::assert_float_eq;

#[test]
fn test_gravity_force() {
    let constants = PhysicsConstants::default();
    let gravity = Force::Gravity(constants.gravity);
    let mass = 10.0;
    let velocity = 5.0;

    let force = gravity.apply(mass, velocity);
    assert_float_eq(force, mass * constants.gravity, 1e-6, Some("Gravity force calculation"));
}

#[test]
fn test_drag_force() {
    let drag = Force::Drag { coefficient: 0.5, area: 2.0 };
    let mass = 10.0;
    let velocity = 5.0;

    let force = drag.apply(mass, velocity);
    let expected_force = -0.5 * 0.5 * 2.0 * velocity.abs() * velocity;
    assert_float_eq(force, expected_force, 1e-6, Some("Drag force calculation"));
}

#[test]
fn test_spring_force() {
    let spring = Force::Spring { k: 10.0, x: 0.5 };
    let mass = 10.0;
    let velocity = 5.0;

    let force = spring.apply(mass, velocity);
    assert_float_eq(force, -5.0, 1e-6, Some("Spring force calculation"));
}

#[test]
fn test_constant_force() {
    let constant = Force::Constant(20.0);
    let mass = 10.0;
    let velocity = 5.0;

    let force = constant.apply(mass, velocity);
    assert_float_eq(force, 20.0, 1e-6, Some("Constant force calculation"));
}

#[test]
fn test_physics_system_creation() {
    let constants = PhysicsConstants::default();
    let system = PhysicsSystem::new(constants);

    assert_eq!(system.objects.len(), 0);
    assert_float_eq(system.constants.gravity, constants.gravity, 1e-6, Some("PhysicsSystem creation"));
}

#[test]
fn test_add_object_to_system() {
    let constants = PhysicsConstants::default();
    let mut system = PhysicsSystem::new(constants);
    let object = Object::new(10.0, 5.0, 0.0).unwrap();

    system.add_object(object);

    assert_eq!(system.objects.len(), 1);
    assert_float_eq(system.objects[0].mass, 10.0, 1e-6, Some("Object mass in system"));
    assert_float_eq(system.objects[0].velocity, 5.0, 1e-6, Some("Object velocity in system"));
}

#[test]
fn test_update_system() {
    let constants = PhysicsConstants::default();
    let mut system = PhysicsSystem::new(constants);
    let mut object = Object::new(10.0, 5.0, 0.0).unwrap();
    object.add_force(Force::Constant(20.0));
    system.add_object(object);

    system.update(1.0);

    // Calculate expected values
    let acceleration = 20.0 / 10.0; // Force / mass
    let expected_velocity = 5.0 + acceleration * 1.0;
    let expected_position = 0.0 + 5.0 * 1.0 + 0.5 * acceleration * 1.0 * 1.0;

    assert_float_eq(system.objects[0].velocity, expected_velocity, 1e-6, Some("Updated velocity"));
    assert_float_eq(system.objects[0].position, expected_position, 1e-6, Some("Updated position"));
}

#[test]
fn test_apply_gravity() {
    let constants = PhysicsConstants::default();
    let mut system = PhysicsSystem::new(constants);
    let object = Object::new(10.0, 5.0, 0.0).unwrap();
    system.add_object(object);

    system.apply_gravity();

    assert_eq!(system.objects[0].forces.len(), 1);
    match system.objects[0].forces[0] {
        Force::Gravity(g) => assert_float_eq(g, constants.gravity, 1e-6, Some("Applied gravity")),
        _ => panic!("Expected gravity force"),
    }
}

#[test]
fn test_apply_drag() {
    let constants = PhysicsConstants::default();
    let mut system = PhysicsSystem::new(constants);
    let object = Object::new(10.0, 5.0, 0.0).unwrap();
    system.add_object(object);

    system.apply_drag(0.5, 2.0);

    assert_eq!(system.objects[0].forces.len(), 1);
    match system.objects[0].forces[0] {
        Force::Drag { coefficient, area } => {
            assert_float_eq(coefficient, 0.5, 1e-6, Some("Applied drag coefficient"));
            assert_float_eq(area, 2.0, 1e-6, Some("Applied drag area"));
        },
        _ => panic!("Expected drag force"),
    }
}

#[test]
fn test_apply_spring() {
    let constants = PhysicsConstants::default();
    let mut system = PhysicsSystem::new(constants);
    let object = Object::new(10.0, 5.0, 0.0).unwrap();
    system.add_object(object);

    system.apply_spring(10.0, 0.5);

    assert_eq!(system.objects[0].forces.len(), 1);
    match system.objects[0].forces[0] {
        Force::Spring { k, x } => {
            assert_float_eq(k, 10.0, 1e-6, Some("Applied spring constant"));
            assert_float_eq(x, 0.5, 1e-6, Some("Applied spring displacement"));
        },
        _ => panic!("Expected spring force"),
    }
}

#[test]
fn test_clear_forces() {
    let constants = PhysicsConstants::default();
    let mut system = PhysicsSystem::new(constants);
    let mut object = Object::new(10.0, 5.0, 0.0).unwrap();
    object.add_force(Force::Constant(20.0));
    system.add_object(object);

    system.clear_forces();

    assert_eq!(system.objects[0].forces.len(), 0);
}

#[test]
fn test_multiple_forces() {
    let constants = PhysicsConstants::default();
    let mut system = PhysicsSystem::new(constants);
    let mut object = Object::new(10.0, 5.0, 0.0).unwrap();
    object.add_force(Force::Constant(20.0));
    object.add_force(Force::Gravity(constants.gravity));
    system.add_object(object);

    system.update(1.0);

    let total_force = 20.0 + 10.0 * constants.gravity;
    let acceleration = total_force / 10.0;
    let expected_velocity = 5.0 + acceleration * 1.0;
    let expected_position = 0.0 + 5.0 * 1.0 + 0.5 * acceleration * 1.0 * 1.0;

    assert_float_eq(system.objects[0].velocity, expected_velocity, 1e-6, Some("Updated velocity with multiple forces"));
    assert_float_eq(system.objects[0].position, expected_position, 1e-6, Some("Updated position with multiple forces"));
}