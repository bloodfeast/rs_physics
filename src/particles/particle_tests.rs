use crate::particles::Particle;
use crate::utils::{PhysicsConstants, PhysicsError};

#[test]
fn test_new_valid() {
    // Create a particle with valid parameters.
    let particle = Particle::new((0.0, 0.0), 10.0, (1.0, 2.0), 1.0)
        .expect("Failed to create particle with valid parameters");
    // Verify that the direction vector is normalized.
    let magnitude = (particle.direction.0.powi(2) + particle.direction.1.powi(2)).sqrt();
    assert!((magnitude - 1.0).abs() < 1e-6, "Direction vector is not normalized");
}

#[test]
fn test_new_invalid_mass() {
    // Attempt to create a particle with a non-positive mass.
    let result = Particle::new((0.0, 0.0), 10.0, (1.0, 2.0), 0.0);
    assert!(result.is_err(), "Particle creation should fail for non-positive mass");
    if let Err(err) = result {
        match err {
            PhysicsError::InvalidMass => (),
            _ => panic!("Unexpected error type for invalid mass"),
        }
    }
}

#[test]
fn test_new_zero_direction() {
    // Attempt to create a particle with a zero direction vector.
    let result = Particle::new((0.0, 0.0), 10.0, (0.0, 0.0), 1.0);
    assert!(result.is_err(), "Particle creation should fail for zero direction vector");
}

#[test]
fn test_update() {
    let constants = PhysicsConstants::default();
    // Create a particle with an initial velocity to the right.
    let mut particle = Particle::new((0.0, 0.0), 10.0, (1.0, 0.0), 1.0)
        .expect("Failed to create particle");
    let initial_position = particle.position;
    let dt = 0.016;
    // Update over a time step.
    particle.update(dt, &constants)
        .expect("Failed to update particle");
    // Compute expected positions.
    // Horizontal displacement: 10.0 * dt.
    let expected_x = initial_position.0 + 10.0 * dt;
    // Vertical: since initial vy is 0, after update:
    // vy becomes gravity * dt, and displacement = vy * dt = gravity * dt^2.
    let expected_y = initial_position.1 + constants.gravity * dt * dt;
    assert!((particle.position.0 - expected_x).abs() < 1e-6, "Horizontal position not updated correctly");
    assert!((particle.position.1 - expected_y).abs() < 1e-6, "Vertical position not updated correctly");
}

#[test]
fn test_update_with_effects() {
    let constants = PhysicsConstants::default();
    // Create a particle moving diagonally upward-right.
    let mut particle = Particle::new((0.0, 0.0), 10.0, (0.707, 0.707), 1.0)
        .expect("Failed to create particle");
    let initial_speed = particle.speed;
    // Update the particle with air resistance effects.
    particle.update_with_effects(0.016, &constants, Some((0.47, 1.0)))
        .expect("Failed to update particle with effects");
    // Assert that the speed is reduced due to the drag.
    assert!(particle.speed < initial_speed, "Speed did not reduce due to drag as expected");
}