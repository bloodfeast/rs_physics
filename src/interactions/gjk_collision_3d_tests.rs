use crate::interactions::gjk_collision_3d::{epa_contact_points, get_support_point_for_shape, gjk_collision_detection, handle_line_case, handle_tetrahedron_case, handle_triangle_case};
use crate::interactions::{dot_product, shape_collisions_3d};
use crate::materials::Material;
use crate::models::{PhysicalObject3D, Quaternion, Shape3D, Simplex, SupportPoint, ToCoordinates};
use crate::utils::DEFAULT_PHYSICS_CONSTANTS;

#[test]
fn test_gjk_sphere_sphere_collision() {
    // Create two overlapping spheres
    let shape1 = Shape3D::new_sphere(1.0);
    let shape2 = Shape3D::new_sphere(1.0);

    // Position them to overlap
    let pos1 = (0.0, 0.0, 0.0);
    let pos2 = (1.5, 0.0, 0.0); // Distance 1.5, combined radii 2.0

    // Test with identity orientation
    let orientation = Quaternion::identity();

    // Run GJK collision detection
    let result = gjk_collision_detection(
        &shape1, pos1, orientation,
        &shape2, pos2, orientation
    );

    assert!(result.is_some(), "GJK should detect collision between overlapping spheres");
}

#[test]
fn test_gjk_sphere_sphere_no_collision() {
    // Create two non-overlapping spheres
    let shape1 = Shape3D::new_sphere(1.0);
    let shape2 = Shape3D::new_sphere(1.0);

    // Position them far apart
    let pos1 = (0.0, 0.0, 0.0);
    let pos2 = (3.0, 0.0, 0.0); // Distance 3.0, combined radii 2.0

    // Test with identity orientation
    let orientation = Quaternion::identity();

    // Run GJK collision detection
    let result = gjk_collision_detection(
        &shape1, pos1, orientation,
        &shape2, pos2, orientation
    );

    assert!(result.is_none(), "GJK should not detect collision between non-overlapping spheres");
}

#[test]
fn test_epa_sphere_sphere_contact_info() {
    // Create two overlapping spheres
    let shape1 = Shape3D::new_sphere(1.0);
    let shape2 = Shape3D::new_sphere(1.0);

    // Position them to overlap
    let pos1 = (0.0, 0.0, 0.0);
    let pos2 = (1.5, 0.0, 0.0); // Distance 1.5, combined radii 2.0

    // Test with identity orientation
    let orientation = Quaternion::identity();

    // First run GJK to get the simplex
    let simplex = gjk_collision_detection(
        &shape1, pos1, orientation,
        &shape2, pos2, orientation
    ).expect("GJK should detect collision");

    // Then run EPA to get contact info
    let contact = epa_contact_points(
        &shape1, pos1, orientation,
        &shape2, pos2, orientation,
        &simplex
    );

    assert!(contact.is_some(), "EPA should generate contact information for colliding spheres");

    // Verify the contact information
    if let Some(info) = contact {
        // Normal should point from shape2 to shape1 (negative x direction)
        assert!(info.normal.0 < 0.0, "Contact normal should point from shape2 to shape1");

        // Penetration depth should be approximately (2.0 - 1.5) = 0.5
        assert!((info.penetration - 0.5).abs() < 0.01,
                "Penetration depth should be approximately 0.5, got {}", info.penetration);

        // Contact points should be on the surfaces of the spheres
        // For sphere1, it should be at (1.0, 0.0, 0.0)
        // For sphere2, it should be at (0.5, 0.0, 0.0)
        assert!((info.point1.0 - 1.0).abs() < 0.01,
                "Contact point on sphere1 should be at x ≈ 1.0, got {}", info.point1.0);
        assert!((info.point2.0 - 0.5).abs() < 0.01,
                "Contact point on sphere2 should be at x ≈ 0.5, got {}", info.point2.0);
    }
}

#[test]
fn test_get_support_point() {
    // Test support point calculation for a sphere
    let shape = Shape3D::new_sphere(1.0);
    let position = (0.0, 0.0, 0.0);
    let orientation = Quaternion::identity();

    // Direction along positive x-axis
    let direction = (1.0, 0.0, 0.0);

    let support = get_support_point_for_shape(&shape, position, orientation, direction);

    // Support point should be at (1.0, 0.0, 0.0)
    assert!((support.0 - 1.0).abs() < 0.001, "Support point x should be 1.0");
    assert!(support.1.abs() < 0.001, "Support point y should be 0.0");
    assert!(support.2.abs() < 0.001, "Support point z should be 0.0");

    // Direction along negative z-axis
    let direction = (0.0, 0.0, -1.0);

    let support = get_support_point_for_shape(&shape, position, orientation, direction);

    // Support point should be at (0.0, 0.0, -1.0)
    assert!(support.0.abs() < 0.001, "Support point x should be 0.0");
    assert!(support.1.abs() < 0.001, "Support point y should be 0.0");
    assert!((support.2 + 1.0).abs() < 0.001, "Support point z should be -1.0");
}

#[test]
fn test_do_simplex_line_case() {
    // Create a simplex with a line segment
    let mut simplex = Simplex::new();

    // Add two support points to form a line
    let point_a = SupportPoint {
        point: (1.0, 0.0, 0.0),
        point_a: (2.0, 0.0, 0.0),
        point_b: (1.0, 0.0, 0.0)
    };

    let point_b = SupportPoint {
        point: (-1.0, 0.0, 0.0),
        point_a: (0.0, 0.0, 0.0),
        point_b: (1.0, 0.0, 0.0)
    };

    simplex.add(point_a);
    simplex.add(point_b);

    // Direction to be updated by do_simplex
    let mut direction = (0.0, 1.0, 0.0);

    // Test the line case
    let result = handle_line_case(&mut simplex, &mut direction);

    // Line segment shouldn't contain the origin
    assert!(!result, "Line case should not report containing the origin");

    // Direction should be updated to point towards the origin
    assert_ne!(direction.1, 0.0, "Direction should be updated");
}

#[test]
fn test_do_simplex_triangle_case() {
    // Create a simplex with a triangle
    let mut simplex = Simplex::new();

    // Add three support points to form a triangle
    let point_a = SupportPoint {
        point: (1.0, 0.0, 0.0),
        point_a: (2.0, 0.0, 0.0),
        point_b: (1.0, 0.0, 0.0)
    };

    let point_b = SupportPoint {
        point: (0.0, 1.0, 0.0),
        point_a: (0.0, 2.0, 0.0),
        point_b: (0.0, 1.0, 0.0)
    };

    let point_c = SupportPoint {
        point: (0.0, 0.0, 1.0),
        point_a: (0.0, 0.0, 2.0),
        point_b: (0.0, 0.0, 1.0)
    };

    simplex.add(point_c);
    simplex.add(point_b);
    simplex.add(point_a);

    // Direction to be updated by do_simplex
    let mut direction = (0.0, 0.0, 0.0);

    // Test the triangle case
    let result = handle_triangle_case(&mut simplex, &mut direction);

    // Triangle shouldn't contain the origin in 3D
    assert!(!result, "Triangle case should not report containing the origin in 3D");

    // Direction should be updated to point towards the origin
    assert!(direction.0 != 0.0 || direction.1 != 0.0 || direction.2 != 0.0,
            "Direction should be updated");
}

#[test]
fn test_do_simplex_tetrahedron_case() {
    // Create a simplex with a tetrahedron containing the origin
    let mut simplex = Simplex::new();

    // Add four support points to form a tetrahedron containing the origin
    let point_a = SupportPoint {
        point: (1.0, 1.0, 1.0),
        point_a: (2.0, 1.0, 1.0),
        point_b: (1.0, 0.0, 0.0)
    };

    let point_b = SupportPoint {
        point: (-1.0, 1.0, 1.0),
        point_a: (0.0, 1.0, 1.0),
        point_b: (1.0, 0.0, 0.0)
    };

    let point_c = SupportPoint {
        point: (1.0, -1.0, 1.0),
        point_a: (2.0, 0.0, 1.0),
        point_b: (1.0, 1.0, 0.0)
    };

    let point_d = SupportPoint {
        point: (1.0, 1.0, -1.0),
        point_a: (2.0, 1.0, 0.0),
        point_b: (1.0, 0.0, 1.0)
    };

    simplex.add(point_d);
    simplex.add(point_c);
    simplex.add(point_b);
    simplex.add(point_a);

    // Direction doesn't matter as it won't be used if tetrahedron contains origin
    let mut direction = (0.0, 0.0, 0.0);

    // Test the tetrahedron case
    let result = handle_tetrahedron_case(&mut simplex, &mut direction);

    // This tetrahedron should contain the origin
    assert!(result, "Tetrahedron case should report containing the origin");
}

#[test]
fn test_integration_with_sphere_collisions_3d() {
    // Create two overlapping spheres
    let material = Material::steel();

    let mut obj1 = PhysicalObject3D::new(
        1.0,
        (0.0, 0.0, 0.0), // No velocity
        (0.0, 0.0, 0.0), // At origin
        Shape3D::new_sphere(1.0),
        Some(material),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        DEFAULT_PHYSICS_CONSTANTS
    );

    let mut obj2 = PhysicalObject3D::new(
        1.0,
        (0.0, 0.0, 0.0), // No velocity initially
        (1.5, 0.0, 0.0),  // Positioned to overlap with obj1
        Shape3D::new_sphere(1.0),
        Some(material),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        DEFAULT_PHYSICS_CONSTANTS
    );

    // Now give them relative velocity toward each other to satisfy vrel_n < 0
    obj1.object.velocity.x = 0.5; // Moving right
    obj2.object.velocity.x = -0.5; // Moving left

    // Test direct GJK collision
    let orientation1 = Quaternion::from_euler(
        obj1.orientation.roll,
        obj1.orientation.pitch,
        obj1.orientation.yaw
    );

    let orientation2 = Quaternion::from_euler(
        obj2.orientation.roll,
        obj2.orientation.pitch,
        obj2.orientation.yaw
    );

    let gjk_result = gjk_collision_detection(
        &obj1.shape, obj1.object.position.to_coord(), orientation1,
        &obj2.shape, obj2.object.position.to_coord(), orientation2
    );

    assert!(gjk_result.is_some(), "GJK should detect collision between overlapping spheres");

    // Now test the full handle_collision function
    let initial_x1 = obj1.object.position.x;
    let initial_x2 = obj2.object.position.x;

    let collision_occurred = shape_collisions_3d::handle_collision(&mut obj1, &mut obj2, 0.1);

    assert!(collision_occurred, "Collision should be detected by handle_collision");

    // Check that the objects move apart appropriately
    assert!(obj1.object.position.x < initial_x1, "Object 1 should move left to resolve penetration");
    assert!(obj2.object.position.x > initial_x2, "Object 2 should move right to resolve penetration");
}

#[test]
fn test_relative_velocity_calculation() {
    // Create two objects with specific velocities
    let material = Material::steel();

    let obj1 = PhysicalObject3D::new(
        1.0,
        (1.0, 0.0, 0.0), // Moving right
        (0.0, 0.0, 0.0), // At origin
        Shape3D::new_sphere(1.0),
        Some(material),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        DEFAULT_PHYSICS_CONSTANTS
    );

    let obj2 = PhysicalObject3D::new(
        1.0,
        (-1.0, 0.0, 0.0), // Moving left
        (1.5, 0.0, 0.0),  // To the right of obj1
        Shape3D::new_sphere(1.0),
        Some(material),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        DEFAULT_PHYSICS_CONSTANTS
    );

    // Normal from obj1 to obj2 (positive x direction)
    let normal = (1.0, 0.0, 0.0);

    // Calculate points on the surfaces where the objects would contact
    let r1 = (normal.0 * 1.0, normal.1 * 1.0, normal.2 * 1.0); // (1,0,0)
    let r2 = (-normal.0 * 1.0, -normal.1 * 1.0, -normal.2 * 1.0); // (-1,0,0)

    // Calculate velocities at the contact points
    let v1 = shape_collisions_3d::calculate_point_velocity(&obj1, r1);
    let v2 = shape_collisions_3d::calculate_point_velocity(&obj2, r2);

    // Calculate relative velocity
    let vrel = (v2.0 - v1.0, v2.1 - v1.1, v2.2 - v1.2);

    // Calculate normal component of relative velocity
    let vrel_n = dot_product(vrel, normal);

    // With these velocities, vrel_n should be negative (objects approaching each other)
    assert!(vrel_n < 0.0, "Relative velocity along normal should be negative for objects moving toward each other, got {}", vrel_n);

    // The relative velocity should be (1 - (-1)) = 2, but might be affected by the r vectors
    println!("Velocity at contact point 1: ({}, {}, {})", v1.0, v1.1, v1.2);
    println!("Velocity at contact point 2: ({}, {}, {})", v2.0, v2.1, v2.2);
    println!("Relative velocity: ({}, {}, {})", vrel.0, vrel.1, vrel.2);
    println!("vrel_n: {}", vrel_n);
}