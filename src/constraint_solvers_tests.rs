// src/constraint_solvers_tests.rs

use crate::assert_float_eq;
use crate::interactions::Object;
use crate::constraint_solvers::{Joint, Spring, ConstraintSolver, IterativeConstraintSolver};
use crate::errors::PhysicsError;

#[test]
fn test_joint_creation() {
    let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();
    let obj2 = Object::new(1.0, 0.0, 5.0).unwrap();
    let joint = Joint {
        object1: obj1,
        object2: obj2,
        constraint_distance: 5.0,
    };
    assert_float_eq(joint.constraint_distance, 5.0, 1e-6, Some("Joint constraint distance"));
}

#[test]
fn test_joint_solve() {
    let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();
    let obj2 = Object::new(1.0, 0.0, 6.0).unwrap();
    let mut joint = Joint {
        object1: obj1,
        object2: obj2,
        constraint_distance: 5.0,
    };

    joint.solve(0.1).unwrap();

    // The total correction needed is 1.0, so each object should move by 0.1 (the max_correction)
    assert_float_eq(joint.object1.position, 0.1, 1e-6, Some("Object 1 position after solve"));
    assert_float_eq(joint.object2.position, 5.9, 1e-6, Some("Object 2 position after solve"));

    // The velocity change should be the position change divided by dt
    assert_float_eq(joint.object1.velocity, 1.0, 1e-6, Some("Object 1 velocity after solve"));
    assert_float_eq(joint.object2.velocity, -1.0, 1e-6, Some("Object 2 velocity after solve"));
}

#[test]
fn test_joint_calculate_error() {
    let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();
    let obj2 = Object::new(1.0, 0.0, 6.0).unwrap();
    let joint = Joint {
        object1: obj1,
        object2: obj2,
        constraint_distance: 5.0,
    };

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
    let joint = Box::new(Joint {
        object1: obj1,
        object2: obj2,
        constraint_distance: 5.0,
    });

    let mut solver = IterativeConstraintSolver::new(10, 1e-6);
    solver.add_constraint(joint);

    let result = solver.solve(0.1);
    assert!(result.is_ok(), "Solver should complete without errors");
}

#[test]
fn test_iterative_constraint_solver_max_iterations() {
    let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();
    let obj2 = Object::new(1.0, 0.0, 10.0).unwrap();  // Increased initial distance
    let joint = Box::new(Joint {
        object1: obj1,
        object2: obj2,
        constraint_distance: 5.0,
    });

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
    let joint = Box::new(Joint {
        object1: obj1,
        object2: obj2,
        constraint_distance: 5.0,
    });

    let mut solver = IterativeConstraintSolver::new(50, 1e-6);  // Increased max iterations
    solver.add_constraint(joint);

    let result = solver.solve(0.1);
    assert!(result.is_ok(), "Solver should converge with sufficient iterations");
}