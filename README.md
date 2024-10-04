# rs_physics

rs_physics is a Rust library for simulating basic physics calculations and interactions. It provides a set of tools for working with physical constants, performing calculations, and simulating object interactions.

## Features

- Customizable physical constants
- Basic physics calculations (e.g., velocity, acceleration, energy)
- Object interaction simulations (e.g., collisions, gravitational force)
- Comprehensive test suite

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rs_physics = "0.1.0"
```

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
    elastic_collision(&constants, &mut obj1, &mut obj2, 0.0, 0.001).unwrap();

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

## Testing

The library includes a comprehensive test suite. To run the tests, use:

```
cargo test
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
