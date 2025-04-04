# rs_physics

rs_physics is a Rust library for simulating advanced physics calculations and interactions. 
It provides a comprehensive set of tools for working with physical constants, performing calculations, and simulating object interactions across various domains of physics.

## Features

- Customizable physical constants
- Advanced physics calculations (e.g., velocity, acceleration, energy)
- Object interaction simulations (e.g., collisions, gravitational force)
- Rotational dynamics (e.g., torque, angular momentum) (optional, available behind feature flag)
- Thermodynamics (e.g., heat transfer, entropy change) (optional, available behind feature flag)
- Fluid dynamics (e.g., drag force, buoyant force) (optional, available behind feature flag)
- Fluid Simulation based on eulerian method (optional, available behind feature flag)
- Material properties and physics (e.g., density, specific heat capacity) (optional, available behind feature flag)
- Constraint solvers for connected bodies (optional, available behind feature flag)
- Particle system simulation (optional, available behind feature flag)
- WebAssembly (WASM) api for easy integration with web projects
- Comprehensive test suite


## Usage

Here's a quick example of how to use rs_physics:

```rust
use rs_physics::physics;
use rs_physics::interactions::{Object, elastic_collision};

fn main() {
    // Create physics constants (using default values)
    let constants = physics::create_constants(None, None, None, None);

    // Create two objects
    let mut obj1 = Object::new(1.0, 1.0, 0.0).unwrap();
    let mut obj2 = Object::new(1.0, -1.0, 1.0).unwrap();

    // Simulate an elastic collision
    elastic_collision(&constants, &mut obj1, &mut obj2, 0.0, 0.001, 0.47, 1.0).unwrap();

    println!("After collision:");
    println!("Object 1 velocity: {}", obj1.velocity);
    println!("Object 2 velocity: {}", obj2.velocity);
}
```

## API Overview

### Constants

- `PhysicsConstants`: Struct containing physical constants (gravity, air density, speed of sound, atmospheric pressure)
- `create_constants`: Function to create custom `PhysicsConstants`

### Physics Calculations

- Terminal velocity
- Air resistance
- Acceleration and deceleration
- Force and momentum
- Kinetic and potential energy
- Work and power
- Impulse
- Projectile motion
- Centripetal force
- Torque
- Angular velocity

### Object Interactions

- `Object`: Struct representing a physical object with mass, velocity, and position
- `elastic_collision`: Function to simulate elastic collisions between objects
- `gravitational_force`: Function to calculate gravitational force between objects
- `apply_force`: Function to apply a force to an object and update its state

### Rotational Dynamics

- `RotationalObject`: Struct for objects with rotational properties
- Moment of inertia calculations
- Angular momentum and rotational kinetic energy
- Torque application

### Thermodynamics

- `Thermodynamic`: Struct for thermodynamic systems
- Heat transfer calculations
- Entropy change
- Work done in thermodynamic processes
- Thermal efficiency
- Specific heat capacity

### Fluid Dynamics

- `Fluid`: Struct representing fluid properties
- Reynolds number calculation
- Drag force calculation
- Buoyant force calculation
- Pressure drop in pipes

### Constraint Solvers

- `Joint`: Struct for rigid connections between objects
- `Spring`: Struct for spring connections between objects
- `IterativeConstraintSolver`: Solver for systems with multiple constraints

### Particle System Simulation
- `Particle`: Struct for individual particles
- `Simulation`: Struct for particle system simulation
  - Includes support for AVX instructions for faster calculations (4-way SIMD implementation)
    - Fallback to scalar implementation for systems without AVX support using `Rayon` for parallelism

### WebAssembly Support

- WASM bindings for core library functionality
- Easy integration with web projects

## Testing

The library includes a comprehensive test suite. To run the tests, use:

```
cargo test
```

## WebAssembly Build

To build the WebAssembly module, navigate to the `rs_physics_wasm` directory and run:

```
wasm-pack build --target web
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
