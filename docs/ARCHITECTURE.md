# Architecture

This document describes the system architecture, module organization, and design patterns used in rs_physics.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Module Organization](#module-organization)
3. [Core Data Structures](#core-data-structures)
4. [Design Patterns](#design-patterns)
5. [Error Handling](#error-handling)
6. [Performance Considerations](#performance-considerations)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Applications                         │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    High-Level APIs (EasyPhysics)                │
│                         src/apis/                               │
└─────────────────────────────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌───────────────┐    ┌───────────────────┐    ┌───────────────────┐
│    Physics    │    │   Interactions    │    │  Optional Modules │
│  Calculations │    │   & Collisions    │    │  (feature-gated)  │
│ src/physics/  │    │ src/interactions/ │    │                   │
└───────────────┘    └───────────────────┘    └───────────────────┘
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Core Data Models                            │
│   Objects (1D/2D/3D) │ Shapes │ Quaternions │ Forces │ Simplex  │
│                       src/models/                                │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Utilities & Constants                         │
│      Math Helpers │ Physics Constants │ Error Types              │
│                        src/utils/                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Organization

### Core Modules (Always Available)

#### `src/utils/`
Foundation utilities used throughout the library.

| File | Purpose |
|------|---------|
| `constants.rs` | Default physical constants |
| `constants_config.rs` | `PhysicsConstants` struct for customization |
| `math_helpers.rs` | Fast trigonometric approximations |
| `errors.rs` | `PhysicsError` enum for error handling |

#### `src/models/`
Data structures representing physical entities.

| File | Purpose |
|------|---------|
| `objects.rs` | 1D `Object` struct |
| `object_2d.rs` | 2D objects: `ObjectIn2D`, `Axis2D`, `Velocity2D` |
| `object_3d.rs` | 3D objects: `ObjectIn3D`, `Axis3D`, `Velocity3D`, `PhysicalObject3D` |
| `shape_3d.rs` | `Shape3D` enum with volume, inertia, support functions |
| `quaternion.rs` | `Quaternion` for gimbal-lock-free rotations |
| `simplex.rs` | `Simplex` for GJK algorithm |

#### `src/physics/`
Core physics calculations.

| File | Purpose |
|------|---------|
| `physics.rs` | Force, velocity, energy, momentum, projectile calculations |

#### `src/forces/`
Force representation and application.

| File | Purpose |
|------|---------|
| `forces.rs` | `Force` enum: Gravity, Drag, Spring, Constant, Thrust |
| `forces_2d.rs` | 2D force calculations |

#### `src/interactions/`
Collision detection and physics interactions.

| File | Purpose |
|------|---------|
| `interactions.rs` | 1D elastic collision, gravitational force |
| `interactions_2d.rs` | 2D elastic collision |
| `interactions_3d.rs` | 3D elastic collision, vector math utilities |
| `gjk_collision_3d.rs` | GJK + EPA collision detection (~780 lines) |
| `continuous_collision_detection.rs` | CCD system (~2000 lines) |
| `shape_collisions_3d.rs` | Shape-specific collision handling (~1200 lines) |

#### `src/apis/`
High-level interfaces for users.

| File | Purpose |
|------|---------|
| `easy_physics.rs` | `EasyPhysics` simplified API wrapper |

### Feature-Gated Modules

These modules are conditionally compiled based on Cargo features.

#### `src/rotational_dynamics/` (`rotational_dynamics` feature)
| File | Purpose |
|------|---------|
| `rotational_dynamics.rs` | Angular momentum, moment of inertia, torque |

#### `src/thermodynamics/` (`thermodynamics` feature)
| File | Purpose |
|------|---------|
| `thermodynamics.rs` | Heat transfer, entropy change calculations |

#### `src/fluid_dynamics/` (`fluid_dynamics` feature)
| File | Purpose |
|------|---------|
| `fluid_dynamics.rs` | Reynolds number, drag coefficients, buoyancy |
| `fluid_simulation.rs` | Eulerian fluid simulation |

#### `src/materials/` (`materials` feature)
| File | Purpose |
|------|---------|
| `materials.rs` | `Material` struct, stress/strain, failure analysis |

#### `src/constraints/` (`constraints` feature)
| File | Purpose |
|------|---------|
| `constraint_solvers.rs` | `Joint` and `Spring` constraint implementations |

#### `src/particles/` (`particles` feature)
| File | Purpose |
|------|---------|
| `particle.rs` | Individual `Particle` struct |
| `particle_simulation.rs` | Particle system management |
| `particle_interactions_barnes_hut.rs` | Barnes-Hut tree for O(N log N) N-body |

---

## Core Data Structures

### Object Hierarchy

```
Object (1D)
├── mass: f64
├── velocity: f64
├── position: f64
└── forces: Vec<Force>

ObjectIn2D
├── mass: f64
├── velocity: Velocity2D { x, y }
├── position: Axis2D { x, y }
└── forces: Vec<(f64, f64)>

ObjectIn3D
├── mass: f64
├── velocity: Velocity3D { x, y, z }
├── position: Axis3D { x, y, z }
└── forces: Vec<(f64, f64, f64)>

PhysicalObject3D (for complex collisions)
├── object: ObjectIn3D
├── shape: Shape3D
├── orientation: (roll, pitch, yaw)
├── angular_velocity: (x, y, z)
└── inertia_tensor: [[f64; 3]; 3]
```

### Shape3D Variants

```rust
pub enum Shape3D {
    Sphere(f64),                                    // radius
    Cuboid(f64, f64, f64),                         // width, height, depth
    BeveledCuboid(f64, f64, f64, f64),            // w, h, d, bevel_radius
    Cylinder(f64, f64),                            // radius, height
    Polyhedron(Vec<(f64,f64,f64)>, Vec<Vec<usize>>), // vertices, face indices
}
```

Each shape provides:
- `volume()` - Calculate shape volume
- `moment_of_inertia(mass)` - Compute inertia tensor
- `support_point(direction)` - Furthest point in direction (for GJK)
- `get_bounds()` - AABB for broad-phase culling
- `get_world_vertices(position, orientation)` - Transform vertices to world space

### Quaternion

```rust
pub struct Quaternion {
    pub w: f64,  // scalar component
    pub x: f64,  // i component
    pub y: f64,  // j component
    pub z: f64,  // k component
}
```

Methods:
- `from_euler(roll, pitch, yaw)` - Create from Euler angles (ZYX convention)
- `from_axis_angle(axis, angle)` - Create from rotation axis and angle
- `to_euler()` - Convert back to Euler angles
- `normalize()` - Ensure unit quaternion
- `rotate_point(point)` - Apply rotation to a 3D point

### Simplex (GJK Algorithm)

```rust
pub struct Simplex {
    points: Vec<(f64, f64, f64)>,  // Up to 4 points
}
```

Used to build a tetrahedron containing the origin during GJK collision detection.

---

## Design Patterns

### 1. Modular Feature System

Uses Cargo feature flags for optional compilation:

```toml
[features]
default = ["constraints", "materials"]
all = ["constraints", "materials", "fluid_simulation", ...]
```

Code uses `#[cfg(feature = "...")]` to conditionally include modules.

### 2. Composition Over Inheritance

Objects are composed of simple data types rather than using inheritance hierarchies:

```rust
// Composition: PhysicalObject3D contains ObjectIn3D
pub struct PhysicalObject3D {
    pub object: ObjectIn3D,     // Embedded, not inherited
    pub shape: Shape3D,
    pub orientation: (f64, f64, f64),
    // ...
}
```

### 3. Trait-Based Polymorphism

Traits define shared behavior:

```rust
// Coordinate conversion traits
pub trait FromCoordinates<T> {
    fn from_coordinates(coords: T) -> Self;
}

pub trait ToCoordinates<T> {
    fn to_coordinates(&self) -> T;
}

// Constraint solving trait
pub trait ConstraintSolver {
    fn solve(&mut self, objects: &mut [PhysicalObject3D]) -> bool;
    fn calculate_error(&self, objects: &[PhysicalObject3D]) -> f64;
}
```

### 4. Result-Based Error Handling

Functions return `Result<T, PhysicsError>` for robust error propagation:

```rust
pub fn calculate_force(mass: f64, acceleration: f64) -> Result<f64, PhysicsError> {
    if mass <= 0.0 {
        return Err(PhysicsError::InvalidMass);
    }
    Ok(mass * acceleration)
}
```

### 5. Enum-Based Shape Dispatch

Shape behavior is handled through pattern matching:

```rust
impl Shape3D {
    pub fn volume(&self) -> f64 {
        match self {
            Shape3D::Sphere(r) => (4.0 / 3.0) * PI * r.powi(3),
            Shape3D::Cuboid(w, h, d) => w * h * d,
            Shape3D::Cylinder(r, h) => PI * r.powi(2) * h,
            // ...
        }
    }
}
```

---

## Error Handling

### PhysicsError Enum

```rust
pub enum PhysicsError {
    InvalidMass,           // Mass must be positive
    InvalidCoefficient,    // Coefficients must be in valid range
    InvalidArea,           // Area must be positive
    InvalidDistance,       // Distance cannot be zero/negative where inappropriate
    InvalidVelocity,       // Velocity constraints violated
    InvalidAngle,          // Angle out of expected range
    InvalidTime,           // Time must be positive
    InvalidRadius,         // Radius must be positive
    InvalidVolume,         // Volume must be positive
    DivisionByZero,        // Attempted division by zero
    CalculationError(String), // Generic calculation error with message
}
```

### Usage Pattern

```rust
// Input validation at function boundaries
pub fn gravitational_force(m1: f64, m2: f64, r: f64) -> Result<f64, PhysicsError> {
    if m1 <= 0.0 || m2 <= 0.0 {
        return Err(PhysicsError::InvalidMass);
    }
    if r <= 0.0 {
        return Err(PhysicsError::InvalidDistance);
    }

    let g = 6.67430e-11; // Gravitational constant
    Ok(g * m1 * m2 / (r * r))
}
```

---

## Performance Considerations

### Optimization Strategies

1. **Broad-Phase Culling**
   - AABB (Axis-Aligned Bounding Box) checks eliminate ~60% of impossible collisions
   - Performed before expensive GJK algorithm

2. **Specialized Fast Paths**
   - Sphere-sphere collisions use direct distance checks (~30% of cases)
   - Bypasses full GJK when applicable

3. **Fast Math Approximations**
   - `fast_atan()`, `fast_atan2()` for performance-critical paths
   - Configurable accuracy vs. speed trade-offs

4. **Iterative Convergence**
   - GJK: max 32 iterations with early exit
   - EPA: max 64 iterations with tolerance check
   - Stops when sufficient accuracy achieved

5. **Memory Pre-allocation**
   - Simplex allocated with capacity 4 (tetrahedron max)
   - Avoids repeated allocations during collision detection

6. **SIMD Support**
   - Available via `avx512-simd` feature
   - Rayon provides data-parallel operations

### Parallelization

```rust
// Particle systems support parallel updates
use rayon::prelude::*;

particles.par_iter_mut().for_each(|p| {
    p.update_position(dt);
});
```

### Algorithm Complexity

| Algorithm | Complexity | Notes |
|-----------|------------|-------|
| GJK Collision | O(1) amortized | Max 32 iterations |
| EPA Contact | O(n) | n = polytope faces (max 64 iter) |
| Broad Phase | O(n) | AABB comparison |
| Barnes-Hut | O(N log N) | N-body gravitational |
| Naive N-body | O(N²) | Avoided with Barnes-Hut |

### Constants Tuning

```rust
// GJK/EPA constants (src/interactions/gjk_collision_3d.rs)
const EPSILON: f64 = 1e-12;        // Numerical tolerance
const GJK_MAX_ITERATIONS: usize = 32;
const EPA_MAX_ITERATIONS: usize = 64;
const EPA_TOLERANCE: f64 = 1e-6;
```

---

## Thread Safety

The library primarily uses immutable data or `&mut` references, making it compatible with Rust's ownership model. For parallel operations:

- Particle systems use Rayon for safe parallelism
- Objects are typically processed in isolation
- No global mutable state

---

## WebAssembly Support

The `rs_physics_wasm/` directory contains bindings for WebAssembly compilation:

- Separate crate for WASM-specific bindings
- Exposes key functions to JavaScript
- Uses `wasm-bindgen` for interop

---

*See [COLLISION_DETECTION.md](./COLLISION_DETECTION.md) for detailed collision algorithm documentation.*
