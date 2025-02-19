use crate::particles::Simulation;
use crate::utils::PhysicsConstants;

#[test]
fn test_new_simulation() {
    // Create a simulation with 50 particles.
    let constants = PhysicsConstants::default();
    let sim = Simulation::new(50, (0.0, 0.0), 10.0, (1.0, 0.0), 1.0, constants, 0.016)
        .expect("Failed to create simulation");
    // Check that each property array has the correct length.
    assert_eq!(sim.positions_x.len(), 50);
    assert_eq!(sim.positions_y.len(), 50);
    assert_eq!(sim.speeds.len(), 50);
    // Check that the direction vector for the first particle is normalized.
    let mag = (sim.directions_x[0].powi(2) + sim.directions_y[0].powi(2)).sqrt();
    assert!((mag - 1.0).abs() < 1e-6, "Direction vector is not normalized");
}

#[test]
fn test_step_updates_positions() {
    // Create a simulation with 10 particles starting at (0, 0), moving right.
    let constants = PhysicsConstants::default();
    let mut sim = Simulation::new(10, (0.0, 0.0), 10.0, (1.0, 0.0), 1.0, constants, 0.016)
        .expect("Failed to create simulation");
    // Record the initial positions.
    let initial_x = sim.positions_x.clone();
    let initial_y = sim.positions_y.clone();
    // Run a single step.
    sim.step().expect("Step failed");
    // Each particle should have moved: x should increase (due to the initial velocity)
    // and y should increase because gravity is applied.
    for i in 0..sim.positions_x.len() {
        assert!(sim.positions_x[i] > initial_x[i], "Particle {} x did not increase", i);
        assert!(sim.positions_y[i] > initial_y[i], "Particle {} y did not increase", i);
    }
}

#[test]
fn test_simulate_updates_positions() {
    // Create a simulation with 10 particles.
    let constants = PhysicsConstants::default();
    let mut sim = Simulation::new(10, (0.0, 0.0), 10.0, (1.0, 0.0), 1.0, constants, 0.016)
        .expect("Failed to create simulation");
    // Run the simulation for 100 steps.
    sim.simulate(100).expect("Simulation failed");
    // After simulation, assert that each particle's x position has increased.
    for (i, &x) in sim.positions_x.iter().enumerate() {
        assert!(x > 0.0, "Particle {} x position did not increase", i);
    }
}