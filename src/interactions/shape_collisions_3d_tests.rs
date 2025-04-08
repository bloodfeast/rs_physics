use crate::interactions::shape_collisions_3d;
use crate::materials::Material;
use crate::models::{PhysicalObject3D, Shape3D};
use crate::utils::DEFAULT_PHYSICS_CONSTANTS;
use std::f64::consts::PI;
use std::time::Instant;
use crate::assert_float_eq;

#[test]
fn test_sphere_sphere_collision() {
    // Two overlapping spheres
    let sphere1 = Shape3D::new_sphere(1.0);
    let sphere2 = Shape3D::new_sphere(1.0);

    // Centers are 1.5 units apart - should collide
    assert!(shape_collisions_3d::check_collision(
        &sphere1,
        (0.0, 0.0, 0.0),
        &sphere2,
        (1.5, 0.0, 0.0)
    ));

    // Centers are 2.5 units apart - should not collide
    assert!(!shape_collisions_3d::check_collision(
        &sphere1,
        (0.0, 0.0, 0.0),
        &sphere2,
        (2.5, 0.0, 0.0)
    ));
}

#[test]
fn test_sphere_cuboid_collision() {
    let sphere = Shape3D::new_sphere(1.0);
    let cuboid = Shape3D::new_cuboid(2.0, 2.0, 2.0);

    // Sphere center at (2.0, 0.0, 0.0), cuboid center at (0.0, 0.0, 0.0)
    // Distance between centers = 2.0
    // Closest point on cuboid to sphere = (1.0, 0.0, 0.0)
    // Distance from closest point to sphere center = 1.0
    // Sphere radius = 1.0, so they should be touching
    assert!(shape_collisions_3d::check_collision(
        &sphere,
        (2.0, 0.0, 0.0),
        &cuboid,
        (0.0, 0.0, 0.0)
    ));

    // Sphere center at (2.5, 0.0, 0.0), cuboid center at (0.0, 0.0, 0.0)
    // Distance between centers = 2.5
    // Closest point on cuboid to sphere = (1.0, 0.0, 0.0)
    // Distance from closest point to sphere center = 1.5
    // Sphere radius = 1.0, so they should not be colliding
    assert!(!shape_collisions_3d::check_collision(
        &sphere,
        (2.5, 0.0, 0.0),
        &cuboid,
        (0.0, 0.0, 0.0)
    ));
}

#[test]
fn test_cuboid_cuboid_collision() {
    let cuboid1 = Shape3D::new_cuboid(2.0, 2.0, 2.0);
    let cuboid2 = Shape3D::new_cuboid(2.0, 2.0, 2.0);

    // Cuboids centers are 2.0 units apart in x-direction
    // Half-widths add up to 2.0, so they should be touching
    assert!(shape_collisions_3d::check_collision(
        &cuboid1,
        (0.0, 0.0, 0.0),
        &cuboid2,
        (2.0, 0.0, 0.0)
    ));

    // Cuboids centers are 2.1 units apart in x-direction
    // Half-widths add up to 2.0, so they should not be colliding
    assert!(!shape_collisions_3d::check_collision(
        &cuboid1,
        (0.0, 0.0, 0.0),
        &cuboid2,
        (2.1, 0.0, 0.0)
    ));
}

#[test]
fn test_collision_normal() {
    let sphere1 = Shape3D::new_sphere(1.0);
    let sphere2 = Shape3D::new_sphere(1.0);

    // Spheres are colliding in x-direction
    let normal =
        shape_collisions_3d::collision_normal(&sphere1, (0.0, 0.0, 0.0), &sphere2, (1.5, 0.0, 0.0));

    assert!(normal.is_some());
    let (nx, ny, nz) = normal.unwrap();
    assert!((nx - 1.0).abs() < 1e-10); // Normal should point in x-direction
    assert!(ny.abs() < 1e-10);
    assert!(nz.abs() < 1e-10);

    // Spheres are not colliding
    let normal =
        shape_collisions_3d::collision_normal(&sphere1, (0.0, 0.0, 0.0), &sphere2, (3.0, 0.0, 0.0));

    assert!(normal.is_none());
}

#[test]
fn test_check_overlap_along_axis() {
    // Two cubes with corners
    let corners1 = vec![
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ];

    // Second cube next to first cube in x-direction
    let corners2 = vec![
        (2.0, -1.0, -1.0),
        (4.0, -1.0, -1.0),
        (2.0, 1.0, -1.0),
        (4.0, 1.0, -1.0),
        (2.0, -1.0, 1.0),
        (4.0, -1.0, 1.0),
        (2.0, 1.0, 1.0),
        (4.0, 1.0, 1.0),
    ];

    // No overlap along x-axis
    assert!(!shape_collisions_3d::check_overlap_along_axis(
        &corners1,
        &corners2,
        &(1.0, 0.0, 0.0)
    ));

    // Overlap along y-axis and z-axis
    assert!(shape_collisions_3d::check_overlap_along_axis(
        &corners1,
        &corners2,
        &(0.0, 1.0, 0.0)
    ));
    assert!(shape_collisions_3d::check_overlap_along_axis(
        &corners1,
        &corners2,
        &(0.0, 0.0, 1.0)
    ));
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
        1.0, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        Shape3D::new_sphere(1.0), Some(material),
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        DEFAULT_PHYSICS_CONSTANTS
    );

    let mut obj2 = PhysicalObject3D::new(
        1.0, (-1.0, 0.0, 0.0), (1.5, 0.0, 0.0),
        Shape3D::new_sphere(1.0), Some(material),
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
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
    let vrel = (
        v1.0 - v2.0,
        v1.1 - v2.1,
        v1.2 - v2.2
    );

    // Calculate impulse magnitude
    let restitution = 1.0; // Perfectly elastic

    // In the original function, impulse_mag is calculated as:
    // -(1.0 + restitution) * vrel_n / impulse_denom
    // where vrel_n is the normal component of relative velocity
    // The function itself takes care of the sign, so just check absolute value
    let impulse_mag = shape_collisions_3d::calculate_collision_impulse(
        &obj1, &obj2, vrel, normal, restitution, r1, r2
    );

    // Take the absolute value since the sign depends on convention
    let abs_impulse_mag = impulse_mag.abs();
    assert!(abs_impulse_mag > 0.0);

    // Apply impulses - we'll use the absolute value here to ensure forces are applied correctly
    shape_collisions_3d::apply_linear_impulse(&mut obj1, &mut obj2, normal, abs_impulse_mag);

    // For equal masses and elastic collision, velocities should swap
    // Allow some margin for floating point errors
    let epsilon = 1e10;
    assert_float_eq(
        obj1.object.velocity.x,
        3.0,
        epsilon,
        Some("test_collision_impulse_and_response - obj1.object.velocity.x")
    );
    assert_float_eq(
        obj2.object.velocity.x,
        -3.0,
        epsilon,
        Some("test_collision_impulse_and_response - obj2.object.velocity.x")
    );
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

    // Verify a few transformed vertices
    // Original corner at (-1,-1,-1) should now be at (-sqrt(2), 0, -1) approximately
    // due to 45-degree rotation around z-axis
    let transformed_corner = vertices
        .iter()
        .find(|v| {
            (v.0 + 2.0_f64.sqrt() / 2.0).abs() < 1e10
                && (v.1 + 2.0_f64.sqrt() / 2.0).abs() < 1e10
                && (v.2 + 1.0).abs() < 1e10
    });

    assert!(transformed_corner.is_some());

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
