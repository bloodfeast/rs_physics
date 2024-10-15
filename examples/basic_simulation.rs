// examples/basic_simulation.rs

use rs_physics::apis::easy_physics::EasyPhysics;

fn main() -> Result<(), &'static str> {
    let physics = EasyPhysics::new();

    // Create two objects
    let mut obj1 = physics.create_object(1.0, 5.0, 0.0)?;
    let mut obj2 = physics.create_object(2.0, -3.0, 10.0)?;

    println!("Initial state:");
    println!("Object 1: mass={}, velocity={}, position={}", obj1.mass, obj1.velocity, obj1.position);
    println!("Object 2: mass={}, velocity={}, position={}", obj2.mass, obj2.velocity, obj2.position);

    // Calculate initial energies and momenta
    let initial_energy1 = physics.calculate_kinetic_energy(&obj1);
    let initial_energy2 = physics.calculate_kinetic_energy(&obj2);
    let initial_momentum1 = physics.calculate_momentum(&obj1);
    let initial_momentum2 = physics.calculate_momentum(&obj2);

    println!("Initial energy: {} J", initial_energy1 + initial_energy2);
    println!("Initial momentum: {} kg⋅m/s", initial_momentum1 + initial_momentum2);

    // Simulate a collision with drag coefficient and cross-sectional area
    let drag_coefficient = 0.47; // Approximate drag coefficient for a sphere
    let cross_sectional_area = 1.0; // Assuming unit area for simplicity
    physics.simulate_collision(&mut obj1, &mut obj2, 0.0, 0.1, drag_coefficient, cross_sectional_area)?;

    println!("\nAfter collision:");
    println!("Object 1: mass={}, velocity={}, position={}", obj1.mass, obj1.velocity, obj1.position);
    println!("Object 2: mass={}, velocity={}, position={}", obj2.mass, obj2.velocity, obj2.position);

    // Calculate final energies and momenta
    let final_energy1 = physics.calculate_kinetic_energy(&obj1);
    let final_energy2 = physics.calculate_kinetic_energy(&obj2);
    let final_momentum1 = physics.calculate_momentum(&obj1);
    let final_momentum2 = physics.calculate_momentum(&obj2);

    println!("Final energy: {} J", final_energy1 + final_energy2);
    println!("Final momentum: {} kg⋅m/s", final_momentum1 + final_momentum2);

    // Calculate gravitational force between objects
    let force = physics.calculate_gravity_force(&obj1, &obj2)?;
    println!("\nGravitational force between objects: {} N", force);

    Ok(())
}