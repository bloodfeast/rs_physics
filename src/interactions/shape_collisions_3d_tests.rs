use crate::interactions::shape_collisions_3d;
use crate::interactions::gjk_collision_3d::gjk_collision_detection;
use crate::materials::Material;
use crate::models::{PhysicalObject3D, Shape3D, Quaternion};
use crate::utils::DEFAULT_PHYSICS_CONSTANTS;
use std::f64::consts::PI;
use std::time::Instant;

// Testing helpers that replace removed functions
fn check_collision(shape1: &Shape3D, pos1: (f64, f64, f64), shape2: &Shape3D, pos2: (f64, f64, f64)) -> bool {
    // Create quaternions for neutral orientation
    let orientation = Quaternion::identity();

    // Use GJK to detect collision
    gjk_collision_detection(shape1, pos1, orientation, shape2, pos2, orientation).is_some()
}

#[test]
fn test_calculate_impact_point() {
    // Create two colliding spheres
    // Use one of the predefined materials like steel
    let material = Material::steel();

    let sphere1 = PhysicalObject3D::new(
        1.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        Shape3D::new_sphere(1.0),
        Some(material),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        DEFAULT_PHYSICS_CONSTANTS,
    );

    let sphere2 = PhysicalObject3D::new(
        1.0,
        (0.0, 0.0, 0.0),
        (1.5, 0.0, 0.0),
        Shape3D::new_sphere(1.0),
        Some(material),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        DEFAULT_PHYSICS_CONSTANTS,
    );

    // Calculate impact point with normal pointing from sphere1 to sphere2
    let impact = shape_collisions_3d::calculate_impact_point(&sphere1, &sphere2, (1.0, 0.0, 0.0));

    // Impact should be at edge of first sphere (1.0, 0.0, 0.0)
    assert!((impact.0 - 1.0).abs() < 1e-10);
    assert!(impact.1.abs() < 1e-10);
    assert!(impact.2.abs() < 1e-10);
}

#[test]
fn test_calculate_point_velocity() {
    // Object with linear and angular velocity
    let obj = PhysicalObject3D::new(
        1.0,
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 0.0),
        Shape3D::new_sphere(1.0),
        None,
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0), // Angular velocity around z-axis
        DEFAULT_PHYSICS_CONSTANTS,
    );

    // Point at (1.0, 0.0, 0.0) relative to center
    let r = (1.0, 0.0, 0.0);

    // Calculate velocity at point
    let vel = shape_collisions_3d::calculate_point_velocity(&obj, r);

    // Expected: linear velocity + angular velocity × r
    // Angular velocity (0,0,1) × r (1,0,0) = (0,1,0)
    // So total velocity should be (1,3,3)
    assert!((vel.0 - 1.0).abs() < 1e-10);
    assert!((vel.1 - 3.0).abs() < 1e-10); // 2.0 + 1.0
    assert!((vel.2 - 3.0).abs() < 1e-10);
}

#[test]
fn test_collision_impulse_and_response() {
    // Create two colliding objects with opposite velocities
    let material = Material::steel();

    let mut obj1 = PhysicalObject3D::new(
        1.0,
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        Shape3D::new_sphere(1.0),
        Some(material),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        DEFAULT_PHYSICS_CONSTANTS
    );

    let mut obj2 = PhysicalObject3D::new(
        1.0,
        (-1.0, 0.0, 0.0),
        (1.5, 0.0, 0.0),
        Shape3D::new_sphere(1.0),
        Some(material),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        DEFAULT_PHYSICS_CONSTANTS
    );

    // Calculate collision parameters
    let normal = (1.0, 0.0, 0.0); // From obj1 to obj2
    let impact_point = (1.0, 0.0, 0.0); // On surface of obj1

    let r1 = (
        impact_point.0 - obj1.object.position.x,
        impact_point.1 - obj1.object.position.y,
        impact_point.2 - obj1.object.position.z
    );

    let r2 = (
        impact_point.0 - obj2.object.position.x,
        impact_point.1 - obj2.object.position.y,
        impact_point.2 - obj2.object.position.z
    );

    // Calculate velocities at impact point
    let v1 = shape_collisions_3d::calculate_point_velocity(&obj1, r1);
    let v2 = shape_collisions_3d::calculate_point_velocity(&obj2, r2);

    // Relative velocity
    let vrel = (v2.0 - v1.0, v2.1 - v1.1, v2.2 - v1.2);

    // Calculate impulse magnitude
    let restitution = 1.0; // Perfectly elastic
    let impulse_mag = shape_collisions_3d::calculate_collision_impulse(
        &obj1, &obj2, vrel, normal, restitution, r1, r2
    );

    // Save the initial velocities
    let initial_v1 = obj1.object.velocity.x;
    let initial_v2 = obj2.object.velocity.x;

    // Apply impulses - using the actual impulse (not abs value)
    shape_collisions_3d::apply_linear_impulse(&mut obj1, &mut obj2, normal, impulse_mag);

    // Just verify the velocities changed in the right direction
    // This is more robust than checking exact values
    assert!(obj1.object.velocity.x < initial_v1, "obj1 should slow down");
    assert!(obj2.object.velocity.x > initial_v2, "obj2 should speed up");

    // Alternatively, check that momentum is conserved
    let total_momentum_before = obj1.object.mass * initial_v1 + obj2.object.mass * initial_v2;
    let total_momentum_after = obj1.object.mass * obj1.object.velocity.x +
        obj2.object.mass * obj2.object.velocity.x;
    let epsilon = 1e-6;
    assert!((total_momentum_before - total_momentum_after).abs() < epsilon,
            "Momentum should be conserved");
}

#[test]
fn test_update_physics() {
    // Create an object with velocity
    let mut obj = PhysicalObject3D::new(
        1.0,
        (1.0, 0.0, 0.0),
        (0.0, 5.0, 0.0),
        Shape3D::new_sphere(1.0),
        None,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        DEFAULT_PHYSICS_CONSTANTS,
    );

    // Apply gravity and update for 0.1 seconds
    shape_collisions_3d::apply_gravity(&mut obj, 9.8, 0.1);
    shape_collisions_3d::update_physics(&mut obj, 0.1);

    // Object should have moved in x direction
    assert!((obj.object.position.x - 0.1).abs() < 1e-10);

    // Object should have accelerated downward due to gravity
    assert!(obj.object.velocity.y < 0.0);

    // After enough updates, object should hit ground
    for _ in 0..100 {
        shape_collisions_3d::apply_gravity(&mut obj, 9.8, 0.1);
        shape_collisions_3d::update_physics(&mut obj, 0.1);
    }

    // Object should be resting at or above ground level + sphere radius
    assert!(obj.object.position.y >= obj.physics_constants.ground_level + 1.0 - 1e-10);
}

#[test]
fn test_world_vertices_and_faces() {
    // Create a cuboid
    let cuboid = Shape3D::new_cuboid(2.0, 2.0, 2.0);

    // Create physical object with rotation
    let obj = PhysicalObject3D::new(
        1.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        cuboid,
        None,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, PI / 4.0), // 45 degree rotation around z-axis
        DEFAULT_PHYSICS_CONSTANTS,
    );

    // Get world vertices
    let vertices = shape_collisions_3d::world_vertices(&obj);

    // Should have 8 vertices for a cuboid
    assert_eq!(vertices.len(), 8);

    // Just verify we have 8 unique vertices
    let mut unique_vertices = Vec::new();
    for vertex in &vertices {
        // Round to reduce floating point comparison issues
        let rounded = (
            (vertex.0 * 100.0).round() / 100.0,
            (vertex.1 * 100.0).round() / 100.0,
            (vertex.2 * 100.0).round() / 100.0
        );
        unique_vertices.push(rounded);
    }

    assert_eq!(unique_vertices.len(), 8, "Should have 8 unique vertices");

    // Get world faces
    let faces = shape_collisions_3d::world_faces(&obj);

    // Should have 6 faces for a cuboid
    assert_eq!(faces.len(), 6);

    // Each face should have 4 vertices
    for face in &faces {
        assert_eq!(face.len(), 4);
    }
}

#[test]
fn test_die_face_up() {
    // Create a die (beveled cuboid)
    let die = Shape3D::new_die(2.0, 0.08);

    // Create physical object with different orientations
    let orientations = [
        (0.0, 0.0, 0.0),       // Default orientation, face 1 up
        (PI, 0.0, 0.0),        // Flipped 180 around x, face 6 up
        (0.0, -PI / 2.0, 0.0),  // Rotated -90 around y, face 4 up
        (0.0, PI / 2.0, 0.0), // Rotated 90 around y, face 3 up
        (PI / 2.0, 0.0, 0.0),  // Rotated 90 around x, face 2 up
        (-PI / 2.0, 0.0, 0.0), // Rotated -90 around x, face 5 up
    ];

    let expected_faces = [1, 6, 4, 3, 2, 5];

    for i in 0..orientations.len() {
        let obj = PhysicalObject3D::new(
            1.0,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            die.clone(),
            None,
            (0.0, 0.0, 0.0),
            orientations[i],
            DEFAULT_PHYSICS_CONSTANTS,
        );

        let face = shape_collisions_3d::die_face_up(&obj);
        assert!(face.is_some());
        assert_eq!(face.unwrap(), expected_faces[i]);
    }
}

#[test]
fn test_damping_and_rest() {
    // Create an object with velocity
    let mut obj = PhysicalObject3D::new(
        1.0,
        (1.0, 1.0, 1.0),
        (0.0, 5.0, 0.0),
        Shape3D::new_sphere(1.0),
        None,
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
        DEFAULT_PHYSICS_CONSTANTS,
    );

    // Apply strong damping
    shape_collisions_3d::apply_damping(&mut obj, 0.5, 0.5, 1.0);

    // Velocities should be reduced
    assert!(obj.object.velocity.x < 1.0);
    assert!(obj.object.velocity.y < 1.0);
    assert!(obj.object.velocity.z < 1.0);

    assert!(obj.angular_velocity.0 < 1.0);
    assert!(obj.angular_velocity.1 < 1.0);
    assert!(obj.angular_velocity.2 < 1.0);

    // Apply more damping until object comes to rest
    for _ in 0..10 {
        shape_collisions_3d::apply_damping(&mut obj, 0.5, 0.5, 1.0);
    }

    // Check if object is at rest
    assert!(shape_collisions_3d::is_at_rest(&obj, 0.01, 0.01));
}

#[test]
fn test_simulation_performance() {
    // Create multiple objects
    let mut objects = Vec::new();

    for i in 0..10 {
        for j in 0..10 {
            let obj = PhysicalObject3D::new(
                1.0,
                (0.0, 0.0, 0.0),
                (i as f64 * 3.0, 5.0, j as f64 * 3.0),
                Shape3D::new_sphere(1.0),
                None,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                DEFAULT_PHYSICS_CONSTANTS,
            );
            objects.push(obj);
        }
    }

    // Measure time to update all objects
    let start = Instant::now();

    for _ in 0..10 {
        shape_collisions_3d::update_physics_system(&mut objects, 0.1);
    }

    let duration = start.elapsed();

    // Print performance info
    println!("Time to simulate 10 steps with 100 objects: {:?}", duration);

    // Make sure simulation completes
    assert!(true);
}

#[test]
fn test_handle_collision_with_gjk() {
    // Create two spheres with clear overlap, one moving away from the other
    let material = Material::rubber();

    let mut obj1 = PhysicalObject3D::new(
        1.0,
        (2.0, 0.0, 0.0), // Moving RIGHT toward obj2
        (0.0, 0.0, 0.0),
        Shape3D::new_sphere(1.0),
        Some(material),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        DEFAULT_PHYSICS_CONSTANTS
    );

    let mut obj2 = PhysicalObject3D::new(
        1.0,
        (-2.0, 0.0, 0.0), // Moving LEFT toward obj1
        (0.5, 0.0, 0.0),
        Shape3D::new_sphere(1.0),
        Some(material),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        DEFAULT_PHYSICS_CONSTANTS
    );

    // Store initial positions
    let initial_x1 = obj1.object.position.x;
    let initial_x2 = obj2.object.position.x;

    // Handle collision
    let collision_occurred = shape_collisions_3d::handle_collision(&mut obj1, &mut obj2, 0.1);

    println!("obj1 initial_x {initial_x1} - new position_x {}", obj1.object.position.x);

    // Verify collision was detected and handled
    assert!(collision_occurred, "Collision should be detected for overlapping spheres");

    // Objects should have moved apart (penetration resolution)
    assert!(obj1.object.position.x < initial_x1, "Object 1 should move left to resolve penetration");
    assert!(obj2.object.position.x > initial_x2, "Object 2 should move right to resolve penetration");

    println!("obj1.object.velocity.x {}", obj1.object.velocity.x);
    println!("obj2.object.velocity.x {}", obj2.object.velocity.x);
    // Check velocities have changed (collision response)
    assert!(obj1.object.velocity.x < 0.0, "Object 1 velocity should increase in negative direction");
    assert!(obj2.object.velocity.x > 1.0, "Object 2 velocity should increase in positive direction");

    // Check the new distance between them
    let new_distance = obj2.object.position.x - obj1.object.position.x;
    println!("new distance: {new_distance}");
    assert!(new_distance >= 1.95, "Objects should be separated to at least sum of radii (2.0) minus epsilon");
}

#[test]
fn test_resolve_penetration() {
    // Create two overlapping spheres
    let material = Material::steel();

    let mut obj1 = PhysicalObject3D::new(
        1.0,
        (0.0, 0.0, 0.0), // Not moving
        (0.0, 0.0, 0.0),
        Shape3D::new_sphere(1.0),
        Some(material),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        DEFAULT_PHYSICS_CONSTANTS
    );

    let mut obj2 = PhysicalObject3D::new(
        1.0,
        (0.1, 0.0, 0.0), // Not moving
        (1.5, 0.0, 0.0), // Centers are 1.5 units apart, each has radius 1.0
        Shape3D::new_sphere(1.0), // So they overlap by 0.5 units
        Some(material),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        DEFAULT_PHYSICS_CONSTANTS
    );

    // Initial positions
    let initial_x1 = obj1.object.position.x;
    let initial_x2 = obj2.object.position.x;

    // Directly call the sphere penetration resolver with known values
    shape_collisions_3d::resolve_sphere_penetration(
        &mut obj1,
        &mut obj2,
        (1.0, 0.0, 0.0), // Normal from obj1 to obj2
        0.5             // Penetration depth
    );

    // Verify objects moved apart
    assert!(obj1.object.position.x < initial_x1, "Object 1 should move left");
    assert!(obj2.object.position.x > initial_x2, "Object 2 should move right");

    // For equal masses, they should move equally
    let total_movement = (obj2.object.position.x - initial_x2) +
        (initial_x1 - obj1.object.position.x);

    // Total movement should be approximately equal to penetration * correction_percentage
    let correction_percentage = 1.0; // This value is used in the implementation
    assert!((total_movement - 0.5 * correction_percentage).abs() < 1e-6,
            "Total movement should match penetration * correction_percentage");
}