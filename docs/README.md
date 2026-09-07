# rs_physics Documentation

Welcome to the **rs_physics** documentation. This is a comprehensive physics simulation library written in Rust, designed for advanced physics calculations, collision detection, and object interactions.

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Getting Started](#getting-started)
4. [Documentation Index](#documentation-index)
5. [Features](#features)
6. [Dependencies](#dependencies)

---

## Overview

**rs_physics** (v0.2.0) is a modular physics engine providing:

- **Core Physics**: Force, velocity, energy, momentum calculations
- **Collision Detection**: GJK/EPA algorithms for 3D, specialized 2D support
- **Continuous Collision Detection (CCD)**: Prevents tunneling through fast-moving objects
- **Rigid Body Dynamics**: Mass, inertia tensors, angular momentum
- **Constraint Solving**: Joints and springs for connected bodies
- **Particle Systems**: N-body simulation with Barnes-Hut optimization
- **Fluid Dynamics**: Reynolds number, drag, buoyancy calculations
- **Thermodynamics**: Heat transfer and entropy calculations
- **Material System**: Physical material properties and failure analysis

The library supports compilation to **WebAssembly** for web integration and provides a **Bevy-compatible** API.

> **Note**: This library is currently a work-in-progress and not yet ready for production use.

---

## Project Structure

```
rs_physics/
├── src/
│   ├── lib.rs                    # Main library entry point
│   ├── utils/                    # Utilities and constants
│   ├── physics/                  # Core physics calculations
│   ├── models/                   # Data structures (objects, shapes, quaternions)
│   ├── forces/                   # Force system (gravity, drag, spring, etc.)
│   ├── interactions/             # Collision detection and response
│   ├── rotational_dynamics/      # Angular momentum, torque (feature-gated)
│   ├── thermodynamics/           # Heat and entropy (feature-gated)
│   ├── fluid_dynamics/           # Fluid mechanics (feature-gated)
│   ├── materials/                # Material properties (feature-gated)
│   ├── constraints/              # Constraint solvers (feature-gated)
│   ├── particles/                # Particle systems (feature-gated)
│   └── apis/                     # High-level user APIs
├── benches/                      # Performance benchmarks
├── examples/                     # Usage examples
├── rs_physics_wasm/              # WebAssembly bindings
└── docs/                         # Documentation (you are here)
```

---

## Getting Started

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rs_physics = "0.2.0"
```

### Basic Usage

```rust
use rs_physics::apis::easy_physics::EasyPhysics;

fn main() -> Result<(), rs_physics::utils::errors::PhysicsError> {
    // Create physics environment with default constants
    let physics = EasyPhysics::new();

    // Create two objects: mass, velocity, position
    let mut obj1 = physics.create_object(1.0, 5.0, 0.0)?;
    let mut obj2 = physics.create_object(2.0, -3.0, 10.0)?;

    // Simulate elastic collision
    // Parameters: objects, angle, duration, drag_coefficient, area
    physics.simulate_collision(&mut obj1, &mut obj2, 0.0, 0.1, 0.47, 1.0)?;

    // Calculate energies
    let ke1 = physics.calculate_kinetic_energy(&obj1);
    let ke2 = physics.calculate_kinetic_energy(&obj2);

    println!("Final kinetic energies: {} J, {} J", ke1, ke2);
    Ok(())
}
```

### Feature Flags

Enable optional modules via Cargo features:

```toml
[dependencies]
rs_physics = { version = "0.2.0", features = ["all"] }
```

Available features:
- `constraints` (default) - Joint and spring constraint solvers
- `materials` (default) - Material properties and failure analysis
- `fluid_simulation` - Eulerian fluid simulation
- `thermodynamics` - Heat transfer and entropy
- `fluid_dynamics` - Fluid mechanics calculations
- `rotational_dynamics` - Angular momentum and torque
- `particles` - Particle system simulation
- `particles-cosmological` - N-body gravitational simulation
- `avx512-simd` - SIMD optimizations

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [Architecture](./ARCHITECTURE.md) | System design, module organization, design patterns |
| [Collision Detection](./COLLISION_DETECTION.md) | GJK/EPA algorithms, CCD, shape collisions |
| [Physics Systems](./PHYSICS_SYSTEMS.md) | Core physics, forces, interactions |
| [API Reference](./API_REFERENCE.md) | Public functions and types |

---

## Features

### Core Physics
- Kinematics: velocity, acceleration, projectile motion
- Dynamics: force, momentum, impulse
- Energy: kinetic, potential, work, power
- Circular motion: centripetal force, torque, angular velocity

### Collision Detection
- **GJK Algorithm**: Industry-standard convex collision detection
- **EPA Algorithm**: Penetration depth and contact normal extraction
- **Broad Phase**: AABB culling for performance
- **Fast Paths**: Specialized sphere-sphere checks
- **CCD**: Time-of-impact calculation for fast objects

### Supported Shapes
- Sphere
- Cuboid
- Beveled Cuboid (dice)
- Cylinder
- Arbitrary Convex Polyhedron

### Advanced Systems
- Quaternion-based rotations (no gimbal lock)
- Inertia tensor computation
- Constraint solving (iterative)
- Barnes-Hut N-body tree (O(N log N))
- Eulerian fluid simulation

---

## Dependencies

### Runtime
| Crate | Version | Purpose |
|-------|---------|---------|
| rayon | 1.10.0 | Data-level parallelism, SIMD |
| log | 0.4 | Logging infrastructure |
| env_logger | 0.11 | Environment-based logging |
| rand | 0.9.0-alpha.2 | Random number generation |
| approx | 0.5.1 | Floating-point comparisons |

### Development
| Crate | Version | Purpose |
|-------|---------|---------|
| criterion | 0.5 | Benchmarking with HTML reports |

---

## License

See the main repository for license information.

---

*Generated from codebase analysis - December 2025*
