use crate::assert_float_eq;
use crate::models::{ObjectIn3D};
use crate::interactions::{
    elastic_collision_3d,
    gravitational_force_3d,
    apply_force_3d,
    cross_product,
    dot_product,
    vector_magnitude,
    normalize_vector,
    rotate_around_x,
    rotate_around_y,
    rotate_around_z,
    spheres_colliding,
    sphere_collision_normal
};
use crate::utils::{PhysicsConstants, DEFAULT_PHYSICS_CONSTANTS};
use std::f64::consts::PI;

#[test]
fn test_elastic_collision_3d() {
    let constants = DEFAULT_PHYSICS_CONSTANTS;

    // Create two objects moving along x-axis in opposite directions
    let mut obj1 = ObjectIn3D::new(1.0, 2.0, 0.0, 0.0, (0.0, 0.0, 0.0));
    let mut obj2 = ObjectIn3D::new(1.0, -1.0, 0.0, 0.0, (5.0, 0.0, 0.0));

    // Collision along x-axis (head-on)
    let normal = (1.0, 0.0, 0.0);
    let result = elastic_collision_3d(&constants, &mut obj1, &mut obj2, normal, 0.001, 0.0, 0.0);

    assert!(result.is_ok());
    assert!((obj1.velocity.x - (-1.0)).abs() < 0.01, "Object 1 should have velocity close to -1.0");
    assert!((obj2.velocity.x - 2.0).abs() < 0.01, "Object 2 should have velocity close to 2.0");
}

#[test]
fn test_elastic_collision_3d_angle() {
    let constants = DEFAULT_PHYSICS_CONSTANTS;

    // Create two objects with different masses
    let mut obj1 = ObjectIn3D::new(1.0, 1.0, 1.0, 0.0, (0.0, 0.0, 0.0));
    let mut obj2 = ObjectIn3D::new(2.0, -1.0, 0.0, 0.0, (1.0, 1.0, 0.0));

    // Collision at 45 degrees (along x=y direction)
    let normal = (std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2, 0.0);  // Normalized (1,1,0)
    let result = elastic_collision_3d(&constants, &mut obj1, &mut obj2, normal, 0.001, 0.0, 0.0);

    assert!(result.is_ok());
    // After collision, momentum should be conserved
    let momentum_before_x = 1.0 * 1.0 + 2.0 * (-1.0);
    let momentum_before_y = 1.0 * 1.0 + 2.0 * 0.0;

    let momentum_after_x = 1.0 * obj1.velocity.x + 2.0 * obj2.velocity.x;
    let momentum_after_y = 1.0 * obj1.velocity.y + 2.0 * obj2.velocity.y;

    assert!((momentum_before_x - momentum_after_x).abs() < 0.001, "x-momentum should be conserved");
    assert!((momentum_before_y - momentum_after_y).abs() < 0.05, "y-momentum should be conserved");
}

#[test]
fn test_elastic_collision_3d_invalid_inputs() {
    let constants = DEFAULT_PHYSICS_CONSTANTS;
    let mut obj1 = ObjectIn3D::new(1.0, 1.0, 0.0, 0.0, (0.0, 0.0, 0.0));
    let mut obj2 = ObjectIn3D::new(1.0, -1.0, 0.0, 0.0, (1.0, 0.0, 0.0));

    // Test with negative mass
    let mut obj_neg_mass = ObjectIn3D::new(-1.0, 1.0, 0.0, 0.0, (0.0, 0.0, 0.0));
    let result = elastic_collision_3d(&constants, &mut obj_neg_mass, &mut obj2, (1.0, 0.0, 0.0), 0.1, 0.5, 1.0);
    assert!(result.is_err(), "Should fail with negative mass");

    // Test with zero duration
    let result = elastic_collision_3d(&constants, &mut obj1, &mut obj2, (1.0, 0.0, 0.0), 0.0, 0.5, 1.0);
    assert!(result.is_err(), "Should fail with zero duration");

    // Test with negative duration
    let result = elastic_collision_3d(&constants, &mut obj1, &mut obj2, (1.0, 0.0, 0.0), -1.0, 0.5, 1.0);
    assert!(result.is_err(), "Should fail with negative duration");

    // Test with negative drag coefficient
    let result = elastic_collision_3d(&constants, &mut obj1, &mut obj2, (1.0, 0.0, 0.0), 0.1, -0.5, 1.0);
    assert!(result.is_err(), "Should fail with negative drag coefficient");

    // Test with negative cross-sectional area
    let result = elastic_collision_3d(&constants, &mut obj1, &mut obj2, (1.0, 0.0, 0.0), 0.1, 0.5, -1.0);
    assert!(result.is_err(), "Should fail with negative cross-sectional area");

    // Test with zero normal vector
    let result = elastic_collision_3d(&constants, &mut obj1, &mut obj2, (0.0, 0.0, 0.0), 0.1, 0.5, 1.0);
    assert!(result.is_err(), "Should fail with zero normal vector");
}

#[test]
fn test_gravitational_force_3d() {
    // Define objects with known masses and positions
    let constants = PhysicsConstants::new(Some(6.67430e-11), None, None, None, None);
    let earth = ObjectIn3D::new(5.972e24, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0));
    let moon = ObjectIn3D::new(7.342e22, 0.0, 0.0, 0.0, (3.844e8, 0.0, 0.0));

    // Calculate gravitational force
    let force = gravitational_force_3d(&constants, &earth, &moon).unwrap();

    // Expected force magnitude using Newton's law of universal gravitation
    let expected_magnitude = 6.67430e-11 * 5.972e24 * 7.342e22 / (3.844e8 * 3.844e8);

    // Force should be along x-axis (from Earth to Moon)
    assert!((force.0 - expected_magnitude).abs() / expected_magnitude < 0.01, "Force magnitude should match expected value");
    assert!(force.1.abs() < 1e-10, "y-component should be close to zero");
    assert!(force.2.abs() < 1e-10, "z-component should be close to zero");

    // Test that objects at the same position return an error
    let obj1 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0));
    let obj2 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0));
    let result = gravitational_force_3d(&constants, &obj1, &obj2);
    assert!(result.is_err(), "Should fail when objects are at the same position");
}

#[test]
fn test_apply_force_3d() {
    let constants = DEFAULT_PHYSICS_CONSTANTS;
    let mut obj = ObjectIn3D::new(2.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0));

    // Apply a force of (10, 5, 0) N for 2 seconds
    let force = (10.0, 5.0, 0.0);
    let time = 2.0;
    let result = apply_force_3d(&constants, &mut obj, force, time);

    assert!(result.is_ok());

    // Calculate expected velocity changes: a = F/m, v = a*t
    let expected_vx = 10.0 / 2.0 * 2.0; // 10 N / 2 kg * 2 s = 10 m/s
    let expected_vy = 5.0 / 2.0 * 2.0;  // 5 N / 2 kg * 2 s = 5 m/s
    let expected_vz = 0.0;

    // Calculate expected position changes: x = v0*t + 0.5*a*t^2
    let expected_x = 0.0 + 0.5 * (10.0 / 2.0) * 4.0; // 0 + 0.5 * 5 * 4 = 10
    let expected_y = 0.0 + 0.5 * (5.0 / 2.0) * 4.0 - constants.gravity;  // 0 + 0.5 * 2.5 * 4 = 5
    let expected_z = 0.0;

    assert_float_eq(obj.velocity.x, expected_vx, 1e-6, Some("x velocity incorrect"));
    assert_float_eq(obj.velocity.y, expected_vy, 1e-6, Some("y velocity incorrect"));
    assert_float_eq(obj.velocity.z, expected_vz, 1e-6, Some("z velocity incorrect"));

    assert_float_eq(obj.position.x, expected_x, 1e-6, Some("x position incorrect"));
    assert_float_eq(obj.position.y, expected_y, 1e-6, Some("y position incorrect"));
    assert_float_eq(obj.position.z, expected_z, 1e-6, Some("z position incorrect"));

    // Test negative time
    let mut obj2 = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0));
    let result = apply_force_3d(&constants, &mut obj2, (1.0, 0.0, 0.0), -1.0);
    assert!(result.is_err(), "Should fail with negative time");
}

#[test]
fn test_cross_product() {
    // Test standard cases
    let v1 = (1.0, 0.0, 0.0);
    let v2 = (0.0, 1.0, 0.0);
    let result = cross_product(v1, v2);
    assert_eq!(result, (0.0, 0.0, 1.0), "x cross y should equal z");

    let v1 = (0.0, 1.0, 0.0);
    let v2 = (0.0, 0.0, 1.0);
    let result = cross_product(v1, v2);
    assert_eq!(result, (1.0, 0.0, 0.0), "y cross z should equal x");

    let v1 = (0.0, 0.0, 1.0);
    let v2 = (1.0, 0.0, 0.0);
    let result = cross_product(v1, v2);
    assert_eq!(result, (0.0, 1.0, 0.0), "z cross x should equal y");

    // Test anti-commutativity: a × b = -(b × a)
    let a = (3.0, -2.0, 5.0);
    let b = (-1.0, 4.0, 2.0);
    let a_cross_b = cross_product(a, b);
    let b_cross_a = cross_product(b, a);
    assert!((a_cross_b.0 + b_cross_a.0).abs() < 1e-10);
    assert!((a_cross_b.1 + b_cross_a.1).abs() < 1e-10);
    assert!((a_cross_b.2 + b_cross_a.2).abs() < 1e-10);
}

#[test]
fn test_dot_product() {
    // Test orthogonal vectors
    let v1 = (1.0, 0.0, 0.0);
    let v2 = (0.0, 1.0, 0.0);
    let result = dot_product(v1, v2);
    assert_eq!(result, 0.0, "Dot product of orthogonal vectors should be 0");

    // Test parallel vectors
    let v1 = (2.0, 0.0, 0.0);
    let v2 = (3.0, 0.0, 0.0);
    let result = dot_product(v1, v2);
    assert_eq!(result, 6.0, "Dot product of parallel vectors should be product of magnitudes");

    // Test arbitrary vectors
    let v1 = (1.0, 2.0, 3.0);
    let v2 = (4.0, -5.0, 6.0);
    let result = dot_product(v1, v2);
    let expected = 1.0 * 4.0 + 2.0 * (-5.0) + 3.0 * 6.0;
    assert_eq!(result, expected);
}

#[test]
fn test_vector_magnitude() {
    // Test zero vector
    let v = (0.0, 0.0, 0.0);
    assert_eq!(vector_magnitude(v), 0.0);

    // Test unit vectors
    let v = (1.0, 0.0, 0.0);
    assert_eq!(vector_magnitude(v), 1.0);

    // Test pythagorean triple
    let v = (3.0, 4.0, 0.0);
    assert_eq!(vector_magnitude(v), 5.0);

    // Test 3D case
    let v = (1.0, 2.0, 2.0);
    assert_eq!(vector_magnitude(v), 3.0);
}

#[test]
fn test_normalize_vector() {
    // Test unit vector (should remain the same)
    let v = (1.0, 0.0, 0.0);
    let result = normalize_vector(v).unwrap();
    assert!((result.0 - 1.0).abs() < 1e-10);
    assert!((result.1 - 0.0).abs() < 1e-10);
    assert!((result.2 - 0.0).abs() < 1e-10);

    // Test arbitrary vector
    let v = (3.0, 4.0, 0.0);  // magnitude = 5
    let result = normalize_vector(v).unwrap();
    assert!((result.0 - 0.6).abs() < 1e-10);
    assert!((result.1 - 0.8).abs() < 1e-10);
    assert!((result.2 - 0.0).abs() < 1e-10);

    // Test zero vector (should return error)
    let v = (0.0, 0.0, 0.0);
    let result = normalize_vector(v);
    assert!(result.is_err());

    // Verify result is a unit vector
    let v = (123.45, -67.89, 42.0);
    let result = normalize_vector(v).unwrap();
    let magnitude = (result.0 * result.0 + result.1 * result.1 + result.2 * result.2).sqrt();
    assert!((magnitude - 1.0).abs() < 1e-10, "Result should be a unit vector");
}

#[test]
fn test_rotation_around_axes() {
    // Test rotation around x-axis
    let point = (0.0, 1.0, 0.0);
    let rotated = rotate_around_x(point, PI / 2.0); // 90 degrees
    assert!((rotated.0 - 0.0).abs() < 1e-10);
    assert!((rotated.1 - 0.0).abs() < 1e-10);
    assert!((rotated.2 - 1.0).abs() < 1e-10);

    // Test rotation around y-axis
    let point = (1.0, 0.0, 0.0);
    let rotated = rotate_around_y(point, PI / 2.0); // 90 degrees
    assert!((rotated.0 - 0.0).abs() < 1e-10);
    assert!((rotated.1 - 0.0).abs() < 1e-10);
    assert!((rotated.2 - (-1.0)).abs() < 1e-10);

    // Test rotation around z-axis
    let point = (1.0, 0.0, 0.0);
    let rotated = rotate_around_z(point, PI / 2.0); // 90 degrees
    assert!((rotated.0 - 0.0).abs() < 1e-10);
    assert!((rotated.1 - 1.0).abs() < 1e-10);
    assert!((rotated.2 - 0.0).abs() < 1e-10);

    // Test that a full rotation returns to the original point
    let point = (1.0, 2.0, 3.0);
    let rotated = rotate_around_x(point, 2.0 * PI);
    assert!((rotated.0 - point.0).abs() < 1e-10);
    assert!((rotated.1 - point.1).abs() < 1e-10);
    assert!((rotated.2 - point.2).abs() < 1e-10);
}

#[test]
fn test_spheres_colliding() {
    // Test spheres that are colliding
    let center1 = (0.0, 0.0, 0.0);
    let radius1 = 2.0;
    let center2 = (3.0, 0.0, 0.0);
    let radius2 = 2.0;
    assert!(spheres_colliding(center1, radius1, center2, radius2), "Spheres should be colliding");

    // Test spheres that are just touching
    let center2 = (4.0, 0.0, 0.0);
    assert!(spheres_colliding(center1, radius1, center2, radius2), "Spheres should be just touching");

    // Test spheres that are not colliding
    let center2 = (5.0, 0.0, 0.0);
    assert!(!spheres_colliding(center1, radius1, center2, radius2), "Spheres should not be colliding");

    // Test in 3D space
    let center1 = (0.0, 0.0, 0.0);
    let center2 = (2.0, 2.0, 2.0);
    let radius1 = 2.0;
    let radius2 = 2.0;
    // Distance between centers = sqrt(12) ≈ 3.46
    // Sum of radii = 4
    // Since 3.46 < 4, spheres should be colliding
    assert!(spheres_colliding(center1, radius1, center2, radius2), "3D spheres should be colliding");
}

#[test]
fn test_sphere_collision_normal() {
    // Test horizontal direction
    let center1 = (0.0, 0.0, 0.0);
    let center2 = (5.0, 0.0, 0.0);
    let normal = sphere_collision_normal(center1, center2).unwrap();
    assert!((normal.0 - 1.0).abs() < 1e-10);
    assert!((normal.1 - 0.0).abs() < 1e-10);
    assert!((normal.2 - 0.0).abs() < 1e-10);

    // Test diagonal direction
    let center1 = (0.0, 0.0, 0.0);
    let center2 = (1.0, 1.0, 0.0);
    let normal = sphere_collision_normal(center1, center2).unwrap();
    let expected_val = 1.0 / (2.0_f64).sqrt();
    assert!((normal.0 - expected_val).abs() < 1e-10);
    assert!((normal.1 - expected_val).abs() < 1e-10);
    assert!((normal.2 - 0.0).abs() < 1e-10);

    // Test 3D direction
    let center1 = (0.0, 0.0, 0.0);
    let center2 = (1.0, 1.0, 1.0);
    let normal = sphere_collision_normal(center1, center2).unwrap();
    let expected_val = 1.0 / (3.0_f64).sqrt();
    assert!((normal.0 - expected_val).abs() < 1e-10);
    assert!((normal.1 - expected_val).abs() < 1e-10);
    assert!((normal.2 - expected_val).abs() < 1e-10);

    // Test for same centers (should return error)
    let center1 = (0.0, 0.0, 0.0);
    let center2 = (0.0, 0.0, 0.0);
    let result = sphere_collision_normal(center1, center2);
    assert!(result.is_err());
}