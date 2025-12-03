// src/constraint_solvers_tests.rs

use crate::assert_float_eq;
use crate::models::Object;
use crate::constraints::constraint_solvers::{Joint, Spring, ConstraintSolver, IterativeConstraintSolver};
use crate::utils::PhysicsError;

#[test]
fn test_joint_creation() {
    let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();
    let obj2 = Object::new(1.0, 0.0, 5.0).unwrap();
    let joint = Joint::new(obj1, obj2, 5.0).unwrap();
    assert_float_eq(joint.constraint_distance, 5.0, 1e-6, Some("Joint constraint distance"));
}

#[test]
fn test_joint_solve() {
    let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();
    let obj2 = Object::new(1.0, 0.0, 6.0).unwrap();
    let mut joint = Joint::new(obj1, obj2, 5.0).unwrap();

    let initial_error = joint.calculate_error();
    joint.solve(0.1).unwrap();
    let final_error = joint.calculate_error();

    // Error should decrease after solving
    assert!(final_error < initial_error, "Error should decrease after solving");

    // With equal masses, both objects should move symmetrically
    // Object 1 should move right (positive direction)
    assert!(joint.object1.position > 0.0, "Object 1 should move right");
    // Object 2 should move left (negative direction from 6.0)
    assert!(joint.object2.position < 6.0, "Object 2 should move left");
}

#[test]
fn test_joint_calculate_error() {
    let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();
    let obj2 = Object::new(1.0, 0.0, 6.0).unwrap();
    let joint = Joint::new(obj1, obj2, 5.0).unwrap();

    let error = joint.calculate_error();
    assert_float_eq(error, 1.0, 1e-6, Some("Joint error calculation"));
}

#[test]
fn test_spring_creation() {
    let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();
    let obj2 = Object::new(1.0, 0.0, 5.0).unwrap();
    let spring = Spring {
        object1: obj1,
        object2: obj2,
        spring_constant: 10.0,
        rest_length: 4.0,
        damping_factor: 0.5,
    };
    assert_float_eq(spring.spring_constant, 10.0, 1e-6, Some("Spring constant"));
    assert_float_eq(spring.rest_length, 4.0, 1e-6, Some("Spring rest length"));
    assert_float_eq(spring.damping_factor, 0.5, 1e-6, Some("Spring damping factor"));
}

#[test]
fn test_spring_solve() {
    let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();
    let obj2 = Object::new(1.0, 0.0, 5.0).unwrap();
    let mut spring = Spring {
        object1: obj1,
        object2: obj2,
        spring_constant: 10.0,
        rest_length: 4.0,
        damping_factor: 0.5,
    };

    spring.solve(0.1).unwrap();

    // The exact values will depend on the implementation details,
    // but we can check that the objects have moved in the expected direction
    assert!(spring.object1.position > 0.0, "Object 1 should move right");
    assert!(spring.object2.position < 5.0, "Object 2 should move left");
}

#[test]
fn test_spring_calculate_error() {
    let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();
    let obj2 = Object::new(1.0, 0.0, 5.0).unwrap();
    let spring = Spring {
        object1: obj1,
        object2: obj2,
        spring_constant: 10.0,
        rest_length: 4.0,
        damping_factor: 0.5,
    };

    let error = spring.calculate_error();
    assert_float_eq(error, 1.0, 1e-6, Some("Spring error calculation"));
}

#[test]
fn test_iterative_constraint_solver() {
    let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();
    let obj2 = Object::new(1.0, 0.0, 6.0).unwrap();
    let joint = Box::new(Joint::new(obj1, obj2, 5.0).unwrap());

    let mut solver = IterativeConstraintSolver::new(10, 1e-6);
    solver.add_constraint(joint);

    let result = solver.solve(0.1);
    assert!(result.is_ok(), "Solver should complete without errors");
}

#[test]
fn test_iterative_constraint_solver_max_iterations() {
    let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();
    let obj2 = Object::new(1.0, 0.0, 10.0).unwrap();  // Increased initial distance
    let joint = Box::new(Joint::new(obj1, obj2, 5.0).unwrap());

    let mut solver = IterativeConstraintSolver::new(5, 1e-8);  // Reduced max iterations, increased precision
    solver.add_constraint(joint);

    let result = solver.solve(0.1);
    assert!(result.is_err(), "Solver should fail to converge with too few iterations");
    if let Err(PhysicsError::CalculationError(msg)) = result {
        assert!(msg.contains("did not converge"), "Error message should indicate failure to converge");
    } else {
        panic!("Expected CalculationError");
    }
}

#[test]
fn test_iterative_constraint_solver_convergence() {
    let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();
    let obj2 = Object::new(1.0, 0.0, 6.0).unwrap();
    let joint = Box::new(Joint::new(obj1, obj2, 5.0).unwrap());

    let mut solver = IterativeConstraintSolver::new(50, 1e-6);  // Increased max iterations
    solver.add_constraint(joint);

    let result = solver.solve(0.1);
    assert!(result.is_ok(), "Solver should converge with sufficient iterations");
}

// ============================================================================
// Phase 1.1: Mass-Weighted Corrections Tests
// ============================================================================

#[test]
fn test_joint_mass_weighted_correction_equal_masses() {
    // With equal masses, both objects should move equally
    let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();
    let obj2 = Object::new(1.0, 0.0, 6.0).unwrap();
    let mut joint = Joint::new(obj1, obj2, 5.0).unwrap();

    let initial_pos1 = joint.object1.position;
    let initial_pos2 = joint.object2.position;

    joint.solve(0.1).unwrap();

    let delta1 = (joint.object1.position - initial_pos1).abs();
    let delta2 = (joint.object2.position - initial_pos2).abs();

    // With equal masses, deltas should be equal (or very close)
    assert!((delta1 - delta2).abs() < 1e-6,
        "Equal masses should result in equal corrections: delta1={}, delta2={}", delta1, delta2);
}

#[test]
fn test_joint_mass_weighted_correction_unequal_masses() {
    // With unequal masses, lighter object should move more
    let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();  // 1 kg
    let obj2 = Object::new(3.0, 0.0, 6.0).unwrap();  // 3 kg
    let mut joint = Joint::new(obj1, obj2, 5.0).unwrap();

    let initial_pos1 = joint.object1.position;
    let initial_pos2 = joint.object2.position;

    joint.solve(0.1).unwrap();

    let delta1 = (joint.object1.position - initial_pos1).abs();
    let delta2 = (joint.object2.position - initial_pos2).abs();

    // Lighter object (obj1) should move more than heavier object (obj2)
    // With masses 1:3, obj1 should move 3x as much as obj2
    // Ratio should be approximately 3:1
    let ratio = delta1 / delta2;
    assert!((ratio - 3.0).abs() < 0.1,
        "Mass ratio 1:3 should result in ~3:1 movement ratio, got {}", ratio);
}

#[test]
fn test_joint_infinite_mass_anchor() {
    // An object with infinite mass should not move at all
    let obj1 = Object::new(f64::INFINITY, 0.0, 0.0).unwrap();  // Static anchor
    let obj2 = Object::new(1.0, 0.0, 6.0).unwrap();  // 1 kg
    let mut joint = Joint::new(obj1, obj2, 5.0).unwrap();

    let initial_pos1 = joint.object1.position;

    joint.solve(0.1).unwrap();

    // Infinite mass object should not move
    assert_float_eq(joint.object1.position, initial_pos1, 1e-10,
        Some("Infinite mass object should not move"));

    // All correction should be applied to the finite mass object
    assert!(joint.object2.position < 6.0,
        "Finite mass object should move toward constraint");
}

#[test]
fn test_spring_mass_weighted_forces() {
    // With unequal masses, lighter object should accelerate more
    let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();  // 1 kg
    let obj2 = Object::new(4.0, 0.0, 5.0).unwrap();  // 4 kg
    let mut spring = Spring::new(obj1, obj2, 100.0, 3.0, 0.0).unwrap();  // No damping for cleaner test

    let initial_vel1 = spring.object1.velocity;
    let initial_vel2 = spring.object2.velocity;

    spring.solve(0.01).unwrap();

    let delta_vel1 = (spring.object1.velocity - initial_vel1).abs();
    let delta_vel2 = (spring.object2.velocity - initial_vel2).abs();

    // Lighter object should gain more velocity (F=ma means lighter objects accelerate more)
    assert!(delta_vel1 > delta_vel2,
        "Lighter object should accelerate more: delta_vel1={}, delta_vel2={}", delta_vel1, delta_vel2);

    // The ratio of accelerations should be inverse of mass ratio (4:1)
    let ratio = delta_vel1 / delta_vel2;
    assert!((ratio - 4.0).abs() < 0.5,
        "Mass ratio 1:4 should result in ~4:1 acceleration ratio, got {}", ratio);
}

// ============================================================================
// Phase 1.2: Baumgarte Stabilization Tests
// ============================================================================

#[test]
fn test_joint_new_with_baumgarte() {
    let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();
    let obj2 = Object::new(1.0, 0.0, 5.0).unwrap();

    // Test with_baumgarte builder method
    let joint = Joint::new(obj1, obj2, 5.0).unwrap().with_baumgarte(0.3);
    assert_float_eq(joint.baumgarte, 0.3, 1e-6, Some("Baumgarte factor should be set"));
}

#[test]
fn test_joint_default_baumgarte() {
    let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();
    let obj2 = Object::new(1.0, 0.0, 5.0).unwrap();

    let joint = Joint::new(obj1, obj2, 5.0).unwrap();
    // Default Baumgarte factor should be 0.2
    assert_float_eq(joint.baumgarte, 0.2, 1e-6, Some("Default Baumgarte should be 0.2"));
}

#[test]
fn test_baumgarte_prevents_drift() {
    // Test that Baumgarte stabilization helps reduce positional error over time
    let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();
    let obj2 = Object::new(1.0, 0.0, 6.0).unwrap();  // 1m error

    let mut joint_with_baumgarte = Joint::new(obj1.clone(), obj2.clone(), 5.0).unwrap()
        .with_baumgarte(0.2);
    let mut joint_without_baumgarte = Joint::new(obj1, obj2, 5.0).unwrap()
        .with_baumgarte(0.0);

    // Run multiple iterations
    for _ in 0..10 {
        joint_with_baumgarte.solve(0.016).unwrap();
        joint_without_baumgarte.solve(0.016).unwrap();
    }

    let error_with = joint_with_baumgarte.calculate_error();
    let error_without = joint_without_baumgarte.calculate_error();

    // Baumgarte should help reduce error faster (or at least as well)
    assert!(error_with <= error_without + 1e-6,
        "Baumgarte should help reduce drift: with={}, without={}", error_with, error_without);
}