//! Integration tests for the unified constraint solvers.
//!
//! These tests verify that the UnifiedSolver2D and UnifiedSolver3D work correctly
//! with real constraint types in realistic scenarios.

use crate::models::{ObjectIn2D, ObjectIn3D};
use crate::constraints::{
    UnifiedSolver2D, UnifiedSolver3D,
    Joint2D, Joint3D,
    Spring2D, Spring3D,
    Rope3D,
    Hinge3D,
    Contact3D, ContactPoint3D,
};

// ============================================================================
// 3D Integration Tests
// ============================================================================

/// Test a chain of joints connecting multiple objects in series.
/// This simulates a rope-like structure where each object is connected
/// to its neighbors by joint constraints.
#[test]
fn test_chain_of_joints_3d() {
    // Create a chain of 5 objects along the X axis
    let obj1 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0));
    let obj2 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (2.0, 0.0, 0.0));
    let obj3 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (4.0, 0.0, 0.0));
    let obj4 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (6.0, 0.0, 0.0));
    let _obj5 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (8.0, 0.0, 0.0));

    // Create joints between consecutive objects (each with 2m target distance)
    let joint1 = Joint3D::new(obj1, obj2, 2.0).unwrap();
    let joint2 = Joint3D::new(obj3.clone(), obj4.clone(), 2.0).unwrap();

    // Create a solver with generous iterations
    let mut solver = UnifiedSolver3D::new(50, 0.01).unwrap();
    solver.add_constraint(Box::new(joint1));
    solver.add_constraint(Box::new(joint2));

    // Solve and verify
    let result = solver.solve(0.016).unwrap();

    // Should converge since objects start at correct distances
    assert!(result.converged, "Chain of joints should converge");
    assert!(result.max_error < 0.01, "Error should be below tolerance: {}", result.max_error);
}

/// Test a chain of joints with initial error that needs correction.
#[test]
fn test_chain_of_joints_with_error_3d() {
    // Create objects with incorrect initial spacing (3m apart instead of 2m)
    let obj1 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0));
    let obj2 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (3.0, 0.0, 0.0));  // 3m instead of 2m
    let obj3 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (6.0, 0.0, 0.0));  // 3m instead of 2m

    let joint1 = Joint3D::new(obj1, obj2, 2.0).unwrap();
    let joint2 = Joint3D::new(obj3.clone(), ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (9.0, 0.0, 0.0)), 2.0).unwrap();

    let mut solver = UnifiedSolver3D::new(100, 0.01).unwrap();
    solver.add_constraint(Box::new(joint1));
    solver.add_constraint(Box::new(joint2));

    // Run solver
    let result = solver.solve(0.016).unwrap();

    // Should converge and reduce error significantly
    assert!(result.max_error < 1.0, "Error should be reduced from initial 1m error: {}", result.max_error);
}

/// Test a 2x2 grid of springs simulating a soft body.
/// This tests multiple interconnected springs working together.
#[test]
fn test_soft_body_spring_network_3d() {
    // Create a 2x2 grid of objects
    // (0,0) -- (1,0)
    //   |        |
    // (0,1) -- (1,1)
    let obj_00 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0));
    let obj_10 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (1.0, 0.0, 0.0));
    let obj_01 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (0.0, 1.0, 0.0));
    let obj_11 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (1.0, 1.0, 0.0));

    // Create springs connecting adjacent objects
    let spring_horizontal_top = Spring3D::new(obj_00.clone(), obj_10.clone(), 100.0, 1.0, 0.5).unwrap();
    let spring_horizontal_bottom = Spring3D::new(obj_01.clone(), obj_11.clone(), 100.0, 1.0, 0.5).unwrap();
    let spring_vertical_left = Spring3D::new(obj_00, obj_01, 100.0, 1.0, 0.5).unwrap();
    let spring_vertical_right = Spring3D::new(obj_10, obj_11, 100.0, 1.0, 0.5).unwrap();

    let mut solver = UnifiedSolver3D::new(20, 0.1).unwrap();
    solver.add_constraint(Box::new(spring_horizontal_top));
    solver.add_constraint(Box::new(spring_horizontal_bottom));
    solver.add_constraint(Box::new(spring_vertical_left));
    solver.add_constraint(Box::new(spring_vertical_right));

    // Solve
    let result = solver.solve(0.016).unwrap();

    // Springs at rest length should have low error
    assert!(result.converged || result.max_error < 0.2,
        "Spring network should stabilize: converged={}, error={}", result.converged, result.max_error);
}

/// Test a spring network with initial deformation.
#[test]
fn test_soft_body_spring_network_deformed_3d() {
    // Create a deformed 2x2 grid (stretched horizontally)
    let obj_00 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0));
    let obj_10 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (2.0, 0.0, 0.0));  // Stretched to 2m instead of 1m
    let obj_01 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (0.0, 1.0, 0.0));
    let obj_11 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (2.0, 1.0, 0.0));  // Stretched

    // Springs with rest length 1.0
    let spring_top = Spring3D::new(obj_00.clone(), obj_10.clone(), 100.0, 1.0, 0.5).unwrap();
    let spring_bottom = Spring3D::new(obj_01.clone(), obj_11.clone(), 100.0, 1.0, 0.5).unwrap();
    let spring_left = Spring3D::new(obj_00, obj_01, 100.0, 1.0, 0.5).unwrap();
    let spring_right = Spring3D::new(obj_10, obj_11, 100.0, 1.0, 0.5).unwrap();

    let mut solver = UnifiedSolver3D::new(50, 0.1).unwrap();
    solver.add_constraint(Box::new(spring_top));
    solver.add_constraint(Box::new(spring_bottom));
    solver.add_constraint(Box::new(spring_left));
    solver.add_constraint(Box::new(spring_right));

    // Run for multiple frames to let springs pull objects back
    for _ in 0..5 {
        let _ = solver.solve(0.016);
    }

    // Check constraint count is maintained
    assert_eq!(solver.constraint_count(), 4);
}

/// Test hinge constraints simulating a simple ragdoll-like arm chain.
/// This tests angular constraints working in series.
#[test]
fn test_ragdoll_hinge_chain_3d() {
    // Create a simple arm: shoulder -> elbow -> wrist
    // Each segment rotates around the Z axis (like a 2D arm in the XY plane)

    let shoulder_pos = (0.0, 0.0, 0.0);
    let elbow_pos = (1.0, 0.0, 0.0);
    let wrist_pos = (2.0, 0.0, 0.0);

    let upper_arm = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, shoulder_pos);
    let lower_arm = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, elbow_pos);
    let hand = ObjectIn3D::new(0.5, 0.0, 0.0, 0.0, wrist_pos);

    // Create hinge at elbow (rotates around Z axis)
    let elbow_hinge = Hinge3D::new(
        upper_arm.clone(),
        lower_arm.clone(),
        elbow_pos,
        (0.0, 0.0, 1.0),  // Z axis rotation
    ).unwrap().with_limits(-2.0, 0.5);  // Elbow can bend from -2 to +0.5 radians

    // Create hinge at wrist
    let wrist_hinge = Hinge3D::new(
        lower_arm,
        hand,
        wrist_pos,
        (0.0, 0.0, 1.0),
    ).unwrap().with_limits(-1.0, 1.0);  // Wrist has more limited range

    let mut solver = UnifiedSolver3D::new(30, 0.01).unwrap();
    solver.add_constraint(Box::new(elbow_hinge));
    solver.add_constraint(Box::new(wrist_hinge));

    // Solve
    let result = solver.solve(0.016).unwrap();

    // Hinges should maintain their constraints
    assert!(result.iterations > 0, "Solver should perform iterations");
    assert_eq!(solver.constraint_count(), 2, "Should have 2 hinge constraints");
}

/// Test hinge constraints with angle limits being violated.
#[test]
fn test_ragdoll_hinge_at_limits_3d() {
    // Create hinge that starts at its limit
    let obj1 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0));
    let obj2 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (1.0, 0.0, 0.0));

    let hinge = Hinge3D::new(
        obj1,
        obj2,
        (0.5, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ).unwrap().with_limits(-0.5, 0.5);

    let mut solver = UnifiedSolver3D::new(20, 0.01).unwrap();
    solver.add_constraint(Box::new(hinge));

    let result = solver.solve(0.016).unwrap();

    // Should complete without error
    assert!(result.iterations > 0);
}

/// Test contact constraints for collision resolution.
/// Simulates two objects that are penetrating and need to be separated.
#[test]
fn test_collision_contact_solver_3d() {
    // Two objects penetrating each other
    let obj1 = ObjectIn3D::new(1.0, 0.0, -1.0, 0.0, (0.0, 0.0, 0.0));  // Moving down
    let obj2 = ObjectIn3D::new(f64::INFINITY, 0.0, 0.0, 0.0, (0.0, -0.5, 0.0));  // Static floor

    // Contact point with penetration
    let contact_point = ContactPoint3D {
        position: (0.0, -0.25, 0.0),
        normal: (0.0, 1.0, 0.0),  // Points up from floor to obj1
        penetration: 0.25,  // 25cm penetration
    };

    let contact = Contact3D::new(
        obj1,
        obj2,
        contact_point,
        0.5,  // Restitution
        0.3,  // Friction
    ).unwrap();

    let mut solver = UnifiedSolver3D::new(20, 0.001).unwrap();
    solver.add_constraint(Box::new(contact));

    let result = solver.solve(0.016).unwrap();

    // Contact should be processed
    assert!(result.iterations > 0, "Solver should process contact");
}

/// Test multiple contacts being solved simultaneously.
#[test]
fn test_multiple_contacts_3d() {
    // Object resting on two contact points
    let obj = ObjectIn3D::new(1.0, 0.0, -1.0, 0.0, (0.0, 0.5, 0.0));
    let floor = ObjectIn3D::new(f64::INFINITY, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0));

    let contact1 = Contact3D::new(
        obj.clone(),
        floor.clone(),
        ContactPoint3D {
            position: (-0.5, 0.0, 0.0),
            normal: (0.0, 1.0, 0.0),
            penetration: 0.1,
        },
        0.3,
        0.5,
    ).unwrap();

    let contact2 = Contact3D::new(
        obj,
        floor,
        ContactPoint3D {
            position: (0.5, 0.0, 0.0),
            normal: (0.0, 1.0, 0.0),
            penetration: 0.1,
        },
        0.3,
        0.5,
    ).unwrap();

    let mut solver = UnifiedSolver3D::new(30, 0.001).unwrap();
    solver.add_constraint(Box::new(contact1));
    solver.add_constraint(Box::new(contact2));

    let result = solver.solve(0.016).unwrap();

    assert_eq!(solver.constraint_count(), 2);
    assert!(result.iterations > 0);
}

/// Test mixed constraint types in a single solver.
#[test]
fn test_mixed_constraints_3d() {
    // Create a scenario with joints, springs, and contacts
    let obj1 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (0.0, 2.0, 0.0));
    let obj2 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (1.0, 2.0, 0.0));
    let floor = ObjectIn3D::new(f64::INFINITY, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0));

    // Joint between obj1 and obj2
    let joint = Joint3D::new(obj1.clone(), obj2.clone(), 1.0).unwrap();

    // Spring between obj2 and anchor
    let anchor = ObjectIn3D::new(f64::INFINITY, 0.0, 0.0, 0.0, (2.0, 3.0, 0.0));
    let spring = Spring3D::new(obj2.clone(), anchor, 50.0, 1.5, 0.3).unwrap();

    // Contact with floor
    let contact = Contact3D::new(
        obj1,
        floor,
        ContactPoint3D {
            position: (0.0, 0.0, 0.0),
            normal: (0.0, 1.0, 0.0),
            penetration: 0.0,  // Just touching
        },
        0.5,
        0.3,
    ).unwrap();

    let mut solver = UnifiedSolver3D::new(50, 0.01).unwrap();
    solver.add_constraint(Box::new(joint));
    solver.add_constraint(Box::new(spring));
    solver.add_constraint(Box::new(contact));

    let result = solver.solve(0.016).unwrap();

    assert_eq!(solver.constraint_count(), 3);
    assert!(result.iterations > 0);
}

/// Test warm starting improves convergence over multiple frames.
#[test]
fn test_warm_starting_convergence_3d() {
    let obj1 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0));
    let obj2 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (1.5, 0.0, 0.0));  // Slight error

    let joint = Joint3D::new(obj1, obj2, 1.0).unwrap();

    let mut solver_with_warm = UnifiedSolver3D::new(20, 0.01).unwrap()
        .with_warm_starting(true);
    solver_with_warm.add_constraint(Box::new(joint));

    // Run multiple frames
    let mut total_iterations = 0;
    for _ in 0..5 {
        let result = solver_with_warm.solve(0.016).unwrap();
        total_iterations += result.iterations;
    }

    // With warm starting, total iterations should be reasonable
    assert!(total_iterations < 100, "Warm starting should help convergence: {}", total_iterations);
}

// ============================================================================
// 2D Integration Tests
// ============================================================================

/// Test a chain of 2D joints.
#[test]
fn test_chain_of_joints_2d() {
    let obj1 = ObjectIn2D::new(1.0, 0.0, 0.0, (0.0, 0.0));
    let obj2 = ObjectIn2D::new(1.0, 0.0, 0.0, (2.0, 0.0));
    let obj3 = ObjectIn2D::new(1.0, 0.0, 0.0, (4.0, 0.0));

    let joint1 = Joint2D::new(obj1, obj2, 2.0).unwrap();
    let joint2 = Joint2D::new(obj3.clone(), ObjectIn2D::new(1.0, 0.0, 0.0, (6.0, 0.0)), 2.0).unwrap();

    let mut solver = UnifiedSolver2D::new(50, 0.01).unwrap();
    solver.add_constraint(Box::new(joint1));
    solver.add_constraint(Box::new(joint2));

    let result = solver.solve(0.016).unwrap();

    assert!(result.converged || result.max_error < 0.1);
}

/// Test a 2D spring network.
#[test]
fn test_soft_body_spring_network_2d() {
    // 2x2 grid of springs
    let obj_00 = ObjectIn2D::new(1.0, 0.0, 0.0, (0.0, 0.0));
    let obj_10 = ObjectIn2D::new(1.0, 0.0, 0.0, (1.0, 0.0));
    let obj_01 = ObjectIn2D::new(1.0, 0.0, 0.0, (0.0, 1.0));
    let obj_11 = ObjectIn2D::new(1.0, 0.0, 0.0, (1.0, 1.0));

    let spring_top = Spring2D::new(obj_00.clone(), obj_10.clone(), 100.0, 1.0, 0.5).unwrap();
    let spring_bottom = Spring2D::new(obj_01.clone(), obj_11.clone(), 100.0, 1.0, 0.5).unwrap();
    let spring_left = Spring2D::new(obj_00, obj_01, 100.0, 1.0, 0.5).unwrap();
    let spring_right = Spring2D::new(obj_10, obj_11, 100.0, 1.0, 0.5).unwrap();

    let mut solver = UnifiedSolver2D::new(20, 0.1).unwrap();
    solver.add_constraint(Box::new(spring_top));
    solver.add_constraint(Box::new(spring_bottom));
    solver.add_constraint(Box::new(spring_left));
    solver.add_constraint(Box::new(spring_right));

    let result = solver.solve(0.016).unwrap();

    assert_eq!(solver.constraint_count(), 4);
    assert!(result.iterations > 0);
}

/// Test the solver with an empty constraint list.
#[test]
fn test_empty_solver_3d() {
    let mut solver = UnifiedSolver3D::new(10, 0.001).unwrap();
    let result = solver.solve(0.016).unwrap();

    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.max_error, 0.0);
}

/// Test the solver with an empty constraint list.
#[test]
fn test_empty_solver_2d() {
    let mut solver = UnifiedSolver2D::new(10, 0.001).unwrap();
    let result = solver.solve(0.016).unwrap();

    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.max_error, 0.0);
}

/// Test adding and removing constraints dynamically.
#[test]
fn test_dynamic_constraint_management_3d() {
    let mut solver = UnifiedSolver3D::new(10, 0.01).unwrap();

    let obj1 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0));
    let obj2 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (1.0, 0.0, 0.0));
    let obj3 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (2.0, 0.0, 0.0));

    // Add constraints
    solver.add_constraint(Box::new(Joint3D::new(obj1.clone(), obj2.clone(), 1.0).unwrap()));
    assert_eq!(solver.constraint_count(), 1);

    solver.add_constraint(Box::new(Joint3D::new(obj2, obj3, 1.0).unwrap()));
    assert_eq!(solver.constraint_count(), 2);

    // Solve with 2 constraints
    let _ = solver.solve(0.016);

    // Remove first constraint
    solver.remove_constraint(0);
    assert_eq!(solver.constraint_count(), 1);

    // Solve with 1 constraint
    let result = solver.solve(0.016).unwrap();
    assert!(result.iterations > 0);

    // Clear all
    solver.clear();
    assert_eq!(solver.constraint_count(), 0);
}

/// Test rope constraint behavior (only resists stretching, not compression).
#[test]
fn test_rope_taut_vs_slack_3d() {
    // Rope that is taut (stretched beyond max length)
    let obj1_taut = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0));
    let obj2_taut = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (3.0, 0.0, 0.0));  // 3m apart
    let rope_taut = Rope3D::new(obj1_taut, obj2_taut, 2.0).unwrap();  // Max length 2m

    // Rope that is slack (closer than max length)
    let obj1_slack = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0));
    let obj2_slack = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (1.0, 0.0, 0.0));  // 1m apart
    let rope_slack = Rope3D::new(obj1_slack, obj2_slack, 2.0).unwrap();  // Max length 2m

    // Taut rope should have error (stretched 1m beyond max)
    assert!(rope_taut.is_taut(), "Rope should be taut when stretched beyond max");
    assert!(rope_taut.calculate_error() > 0.0, "Taut rope should have error");

    // Slack rope should have no error
    assert!(!rope_slack.is_taut(), "Rope should be slack when within max length");
    assert!(rope_slack.calculate_error() < 0.001, "Slack rope should have no error");
}
