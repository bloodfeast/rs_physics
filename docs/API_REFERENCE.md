# API Reference

Complete reference of public functions, structs, and enums in rs_physics.

## Table of Contents

1. [EasyPhysics API](#easyphysics-api)
2. [Physics Functions](#physics-functions)
3. [Force Types](#force-types)
4. [Object Types](#object-types)
5. [Shape Types](#shape-types)
6. [Collision Functions](#collision-functions)
7. [Interaction Functions](#interaction-functions)
8. [Vector Utilities](#vector-utilities)
9. [Error Types](#error-types)
10. [Constants](#constants)

---

## EasyPhysics API

**Location:** `src/apis/easy_physics.rs`

High-level wrapper for common physics operations.

### Construction

```rust
use rs_physics::apis::easy_physics::EasyPhysics;

// Default constants (Earth gravity, etc.)
let physics = EasyPhysics::new();

// Custom constants
let custom = EasyPhysics::with_constants(PhysicsConstants::with_gravity(1.62)); // Moon
```

### Object Creation

```rust
impl EasyPhysics {
    /// Create a 1D object
    /// - mass: Mass in kg
    /// - velocity: Velocity in m/s
    /// - position: Position in m
    pub fn create_object(
        &self,
        mass: f64,
        velocity: f64,
        position: f64
    ) -> Result<Object, PhysicsError>;

    /// Create a 2D object
    pub fn create_object_2d(
        &self,
        mass: f64,
        velocity: (f64, f64),
        position: (f64, f64)
    ) -> Result<ObjectIn2D, PhysicsError>;

    /// Create a 3D object
    pub fn create_object_3d(
        &self,
        mass: f64,
        velocity: (f64, f64, f64),
        position: (f64, f64, f64)
    ) -> Result<ObjectIn3D, PhysicsError>;
}
```

### Collision Simulation

```rust
impl EasyPhysics {
    /// Simulate elastic collision between two 1D objects
    /// - angle: Collision angle in radians
    /// - duration: Time duration in seconds
    /// - drag_coefficient: Aerodynamic drag coefficient
    /// - area: Cross-sectional area in m²
    pub fn simulate_collision(
        &self,
        obj1: &mut Object,
        obj2: &mut Object,
        angle: f64,
        duration: f64,
        drag_coefficient: f64,
        area: f64
    ) -> Result<(), PhysicsError>;
}
```

### Energy Calculations

```rust
impl EasyPhysics {
    /// Calculate kinetic energy: KE = ½mv²
    pub fn calculate_kinetic_energy(&self, obj: &Object) -> f64;

    /// Calculate potential energy: PE = mgh
    pub fn calculate_potential_energy(&self, obj: &Object) -> f64;

    /// Calculate momentum: p = mv
    pub fn calculate_momentum(&self, obj: &Object) -> f64;
}
```

### Force Calculations

```rust
impl EasyPhysics {
    /// Calculate gravitational force between two objects
    pub fn gravitational_force(
        &self,
        obj1: &Object,
        obj2: &Object
    ) -> Result<f64, PhysicsError>;

    /// Apply force to object over time
    pub fn apply_force(
        &self,
        obj: &mut Object,
        force: f64,
        time: f64
    ) -> Result<(), PhysicsError>;
}
```

---

## Physics Functions

**Location:** `src/physics/physics.rs`

### Kinematics

```rust
/// Calculate final velocity
/// v = v₀ + at
pub fn calculate_velocity(
    initial_velocity: f64,
    acceleration: f64,
    time: f64
) -> Result<f64, PhysicsError>;

/// Calculate average velocity
/// v_avg = (v₀ + v) / 2
pub fn calculate_average_velocity(
    initial_velocity: f64,
    final_velocity: f64
) -> f64;

/// Calculate acceleration
/// a = (v - v₀) / t
pub fn calculate_acceleration(
    initial_velocity: f64,
    final_velocity: f64,
    time: f64
) -> Result<f64, PhysicsError>;

/// Calculate deceleration (positive value)
pub fn calculate_deceleration(
    initial_velocity: f64,
    final_velocity: f64,
    time: f64
) -> Result<f64, PhysicsError>;

/// Calculate terminal velocity
/// v_t = √(2mg / ρAC_d)
pub fn calculate_terminal_velocity(
    mass: f64,
    drag_coefficient: f64,
    area: f64,
    constants: &PhysicsConstants
) -> Result<f64, PhysicsError>;

/// Calculate air resistance force
/// F_d = -½ρv²AC_d
pub fn calculate_air_resistance(
    velocity: f64,
    drag_coefficient: f64,
    area: f64,
    constants: &PhysicsConstants
) -> Result<f64, PhysicsError>;
```

### Dynamics

```rust
/// Calculate force (Newton's Second Law)
/// F = ma
pub fn calculate_force(
    mass: f64,
    acceleration: f64
) -> Result<f64, PhysicsError>;

/// Calculate momentum
/// p = mv
pub fn calculate_momentum(
    mass: f64,
    velocity: f64
) -> Result<f64, PhysicsError>;

/// Calculate impulse
/// J = Ft = Δp
pub fn calculate_impulse(
    force: f64,
    time: f64
) -> Result<f64, PhysicsError>;

/// Calculate coefficient of restitution
/// e = v₂' / v₁
pub fn calculate_coefficient_of_restitution(
    velocity_before: f64,
    velocity_after: f64
) -> Result<f64, PhysicsError>;
```

### Energy

```rust
/// Calculate kinetic energy
/// KE = ½mv²
pub fn calculate_kinetic_energy(
    mass: f64,
    velocity: f64
) -> Result<f64, PhysicsError>;

/// Calculate gravitational potential energy
/// PE = mgh
pub fn calculate_potential_energy(
    mass: f64,
    height: f64,
    constants: &PhysicsConstants
) -> Result<f64, PhysicsError>;

/// Calculate work done
/// W = Fd cos(θ)
pub fn calculate_work(
    force: f64,
    distance: f64,
    angle: f64
) -> Result<f64, PhysicsError>;

/// Calculate power
/// P = W / t
pub fn calculate_power(
    work: f64,
    time: f64
) -> Result<f64, PhysicsError>;
```

### Circular Motion

```rust
/// Calculate centripetal force
/// F = mv² / r
pub fn calculate_centripetal_force(
    mass: f64,
    velocity: f64,
    radius: f64
) -> Result<f64, PhysicsError>;

/// Calculate torque
/// τ = rF sin(θ)
pub fn calculate_torque(
    radius: f64,
    force: f64,
    angle: f64
) -> Result<f64, PhysicsError>;

/// Calculate angular velocity
/// ω = v / r
pub fn calculate_angular_velocity(
    velocity: f64,
    radius: f64
) -> Result<f64, PhysicsError>;
```

### Projectile Motion

```rust
/// Calculate time of flight
/// t = 2v₀ sin(θ) / g
pub fn calculate_projectile_time_of_flight(
    initial_velocity: f64,
    angle: f64,
    constants: &PhysicsConstants
) -> Result<f64, PhysicsError>;

/// Calculate maximum height
/// h = v₀² sin²(θ) / 2g
pub fn calculate_projectile_max_height(
    initial_velocity: f64,
    angle: f64,
    constants: &PhysicsConstants
) -> Result<f64, PhysicsError>;
```

---

## Force Types

**Location:** `src/forces/forces.rs`

```rust
pub enum Force {
    /// Gravitational force (downward)
    Gravity {
        mass: f64,
    },

    /// Drag force (opposes motion)
    Drag {
        drag_coefficient: f64,
        area: f64,
        velocity: f64,
    },

    /// Spring force (Hooke's Law)
    Spring {
        spring_constant: f64,
        displacement: f64,
    },

    /// Constant force
    Constant {
        magnitude: f64,
    },

    /// Thrust with direction
    Thrust {
        magnitude: f64,
        angle: f64,
    },
}

impl Force {
    /// Apply force as scalar (1D)
    pub fn apply(&self, constants: &PhysicsConstants) -> f64;

    /// Apply force as 2D vector
    pub fn apply_2d(&self, constants: &PhysicsConstants) -> (f64, f64);

    /// Apply force as 3D vector
    pub fn apply_3d(&self, constants: &PhysicsConstants) -> (f64, f64, f64);
}
```

---

## Object Types

**Location:** `src/models/`

### 1D Object

```rust
pub struct Object {
    pub mass: f64,
    pub velocity: f64,
    pub position: f64,
    pub forces: Vec<Force>,
}

impl Object {
    pub fn new(mass: f64, velocity: f64, position: f64) -> Result<Self, PhysicsError>;
    pub fn to_2d(&self) -> ObjectIn2D;
    pub fn to_3d(&self) -> ObjectIn3D;
}
```

### 2D Coordinates

```rust
pub struct Axis2D {
    pub x: f64,
    pub y: f64,
}

pub struct Velocity2D {
    pub x: f64,
    pub y: f64,
}

pub struct ObjectIn2D {
    pub mass: f64,
    pub velocity: Velocity2D,
    pub position: Axis2D,
    pub forces: Vec<(f64, f64)>,
}

impl ObjectIn2D {
    pub fn speed(&self) -> f64;
    pub fn direction(&self) -> f64;
    pub fn to_3d(&self) -> ObjectIn3D;
}
```

### 3D Coordinates

```rust
pub struct Axis3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub struct Velocity3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub struct ObjectIn3D {
    pub mass: f64,
    pub velocity: Velocity3D,
    pub position: Axis3D,
    pub forces: Vec<(f64, f64, f64)>,
}

impl ObjectIn3D {
    pub fn speed(&self) -> f64;
}
```

### Physical Object 3D

```rust
pub struct PhysicalObject3D {
    pub object: ObjectIn3D,
    pub shape: Shape3D,
    pub orientation: (f64, f64, f64),      // Euler angles (roll, pitch, yaw)
    pub angular_velocity: (f64, f64, f64),
    pub inertia_tensor: [[f64; 3]; 3],
}

impl PhysicalObject3D {
    pub fn new(
        mass: f64,
        position: (f64, f64, f64),
        velocity: (f64, f64, f64),
        shape: Shape3D
    ) -> Self;
}
```

### Quaternion

```rust
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    pub fn identity() -> Self;
    pub fn from_euler(roll: f64, pitch: f64, yaw: f64) -> Self;
    pub fn from_axis_angle(axis: (f64, f64, f64), angle: f64) -> Self;
    pub fn to_euler(&self) -> (f64, f64, f64);
    pub fn rotate_point(&self, point: (f64, f64, f64)) -> (f64, f64, f64);
    pub fn multiply(&self, other: &Quaternion) -> Quaternion;
    pub fn normalize(&self) -> Quaternion;
    pub fn inverse(&self) -> Quaternion;
    pub fn magnitude(&self) -> f64;
}
```

---

## Shape Types

**Location:** `src/models/shape_3d.rs`

```rust
pub enum Shape3D {
    /// Sphere with radius
    Sphere(f64),

    /// Cuboid with width, height, depth
    Cuboid(f64, f64, f64),

    /// Cuboid with beveled edges (for dice simulation)
    BeveledCuboid(f64, f64, f64, f64),  // w, h, d, bevel_radius

    /// Cylinder with radius and height
    Cylinder(f64, f64),

    /// Arbitrary convex polyhedron
    Polyhedron(
        Vec<(f64, f64, f64)>,  // vertices
        Vec<Vec<usize>>,       // face indices
    ),
}

impl Shape3D {
    /// Calculate volume
    pub fn volume(&self) -> f64;

    /// Calculate moment of inertia tensor
    pub fn moment_of_inertia(&self, mass: f64) -> [[f64; 3]; 3];

    /// Get radius (for sphere) or bounding radius
    pub fn get_radius(&self) -> f64;

    /// Get axis-aligned bounding box
    pub fn get_bounds(&self) -> ((f64, f64, f64), (f64, f64, f64));

    /// Get support point in direction (for GJK)
    pub fn support_point(&self, direction: (f64, f64, f64)) -> (f64, f64, f64);

    /// Get vertices transformed to world space
    pub fn get_world_vertices(
        &self,
        position: (f64, f64, f64),
        orientation: &Quaternion
    ) -> Vec<(f64, f64, f64)>;
}
```

---

## Collision Functions

**Location:** `src/interactions/gjk_collision_3d.rs`

### GJK Collision Detection

```rust
/// Detect collision between two 3D objects using GJK algorithm
/// Returns (collision_detected, simplex)
pub fn gjk_collision_detection(
    obj_a: &PhysicalObject3D,
    obj_b: &PhysicalObject3D
) -> (bool, Simplex);

/// Get support point of Minkowski difference
pub fn get_minkowski_support(
    obj_a: &PhysicalObject3D,
    obj_b: &PhysicalObject3D,
    direction: (f64, f64, f64)
) -> (f64, f64, f64);

/// Get support point for a shape in a direction
pub fn get_support_point(
    shape: &Shape3D,
    direction: (f64, f64, f64),
    orientation: &Quaternion
) -> (f64, f64, f64);
```

### EPA Contact Points

```rust
/// Extract contact information using EPA algorithm
/// Returns (penetration_depth, contact_normal, contact_point)
pub fn epa_contact_points(
    obj_a: &PhysicalObject3D,
    obj_b: &PhysicalObject3D,
    simplex: &Simplex
) -> Option<(f64, (f64, f64, f64), (f64, f64, f64))>;
```

### Continuous Collision Detection

**Location:** `src/interactions/continuous_collision_detection.rs`

```rust
/// Result of continuous collision detection
pub struct CcdCollisionResult {
    pub will_collide: bool,
    pub time_of_impact: f64,
    pub normal: Option<(f64, f64, f64)>,
    pub contact_points: Option<Vec<(f64, f64, f64)>>,
}

/// Check for collision during motion over timestep
pub fn check_continuous_collision(
    obj_a: &PhysicalObject3D,
    obj_b: &PhysicalObject3D,
    dt: f64
) -> CcdCollisionResult;

/// Apply collision response with CCD
pub fn apply_continuous_collision_response(
    obj_a: &mut PhysicalObject3D,
    obj_b: &mut PhysicalObject3D,
    result: &CcdCollisionResult,
    restitution: f64
);

/// Full physics update with CCD
pub fn update_physics_with_ccd(
    objects: &mut [PhysicalObject3D],
    dt: f64,
    gravity: (f64, f64, f64),
    restitution: f64
);
```

### Shape Collisions

**Location:** `src/interactions/shape_collisions_3d.rs`

```rust
/// Calculate impact point between two objects
pub fn calculate_impact_point(
    obj_a: &PhysicalObject3D,
    obj_b: &PhysicalObject3D,
    collision_normal: (f64, f64, f64)
) -> (f64, f64, f64);

/// Calculate velocity at a point (includes angular velocity)
pub fn calculate_point_velocity(
    obj: &PhysicalObject3D,
    point: (f64, f64, f64)
) -> (f64, f64, f64);

/// Calculate collision impulse magnitude
pub fn calculate_collision_impulse(
    obj_a: &PhysicalObject3D,
    obj_b: &PhysicalObject3D,
    contact_point: (f64, f64, f64),
    normal: (f64, f64, f64),
    restitution: f64
) -> f64;

/// Apply linear impulse to object
pub fn apply_linear_impulse(
    obj: &mut PhysicalObject3D,
    impulse: (f64, f64, f64)
);

/// Apply angular impulse to object
pub fn apply_angular_impulse(
    obj: &mut PhysicalObject3D,
    impulse: (f64, f64, f64),
    contact_point: (f64, f64, f64)
);
```

---

## Interaction Functions

**Location:** `src/interactions/`

### 1D Interactions

```rust
/// Elastic collision between two 1D objects
pub fn elastic_collision(
    obj1: &mut Object,
    obj2: &mut Object,
    angle: f64,
    duration: f64,
    drag_coefficient: f64,
    area: f64,
    constants: &PhysicsConstants
) -> Result<(), PhysicsError>;

/// Gravitational force between two 1D objects
pub fn gravitational_force(
    obj1: &Object,
    obj2: &Object
) -> Result<f64, PhysicsError>;

/// Apply force to 1D object over time
pub fn apply_force(
    obj: &mut Object,
    force: f64,
    time: f64,
    constants: &PhysicsConstants
) -> Result<(), PhysicsError>;
```

### 2D Interactions

```rust
/// Elastic collision between two 2D objects
pub fn elastic_collision_2d(
    obj1: &mut ObjectIn2D,
    obj2: &mut ObjectIn2D,
    collision_normal: (f64, f64),
    duration: f64,
    drag_coefficient: f64,
    area: f64,
    constants: &PhysicsConstants
) -> Result<(), PhysicsError>;
```

### 3D Interactions

```rust
/// Elastic collision between two 3D objects
pub fn elastic_collision_3d(
    obj1: &mut ObjectIn3D,
    obj2: &mut ObjectIn3D,
    collision_normal: (f64, f64, f64),
    duration: f64,
    drag_coefficient: f64,
    area: f64,
    constants: &PhysicsConstants
) -> Result<(), PhysicsError>;

/// Gravitational force between two 3D objects
pub fn gravitational_force_3d(
    obj1: &ObjectIn3D,
    obj2: &ObjectIn3D
) -> Result<(f64, f64, f64), PhysicsError>;

/// Apply 3D force to object over time
pub fn apply_force_3d(
    obj: &mut ObjectIn3D,
    force: (f64, f64, f64),
    time: f64
) -> Result<(), PhysicsError>;

/// Check if two spheres are colliding
pub fn spheres_colliding(
    s1: &PhysicalObject3D,
    s2: &PhysicalObject3D
) -> bool;

/// Get collision normal for spheres
pub fn sphere_collision_normal(
    s1: &PhysicalObject3D,
    s2: &PhysicalObject3D
) -> (f64, f64, f64);
```

---

## Vector Utilities

**Location:** `src/interactions/interactions_3d.rs`

```rust
/// Cross product of two 3D vectors
pub fn cross_product(
    a: (f64, f64, f64),
    b: (f64, f64, f64)
) -> (f64, f64, f64);

/// Dot product of two 3D vectors
pub fn dot_product(
    a: (f64, f64, f64),
    b: (f64, f64, f64)
) -> f64;

/// Magnitude of 3D vector
pub fn vector_magnitude(v: (f64, f64, f64)) -> f64;

/// Normalize 3D vector to unit length
pub fn normalize_vector(v: (f64, f64, f64)) -> (f64, f64, f64);

/// Add two 3D vectors
pub fn add(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64);

/// Subtract two 3D vectors
pub fn subtract(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64);

/// Scale 3D vector by scalar
pub fn scale(v: (f64, f64, f64), s: f64) -> (f64, f64, f64);

/// Negate 3D vector
pub fn negate(v: (f64, f64, f64)) -> (f64, f64, f64);
```

---

## Error Types

**Location:** `src/utils/errors.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum PhysicsError {
    /// Mass must be positive
    InvalidMass,

    /// Coefficient out of valid range
    InvalidCoefficient,

    /// Area must be positive
    InvalidArea,

    /// Distance cannot be zero or negative (where inappropriate)
    InvalidDistance,

    /// Velocity constraint violated
    InvalidVelocity,

    /// Angle out of expected range
    InvalidAngle,

    /// Time must be positive
    InvalidTime,

    /// Radius must be positive
    InvalidRadius,

    /// Volume must be positive
    InvalidVolume,

    /// Attempted division by zero
    DivisionByZero,

    /// Generic calculation error with description
    CalculationError(String),
}

impl std::fmt::Display for PhysicsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
}

impl std::error::Error for PhysicsError {}
```

---

## Constants

**Location:** `src/utils/constants.rs`, `src/utils/constants_config.rs`

### PhysicsConstants

```rust
pub struct PhysicsConstants {
    /// Gravitational acceleration (m/s²)
    /// Default: 9.80665 (Earth standard)
    pub gravity: f64,

    /// Air density (kg/m³)
    /// Default: 1.225 (at sea level, 15°C)
    pub air_density: f64,

    /// Speed of sound (m/s)
    /// Default: 343.0 (at 20°C in air)
    pub speed_of_sound: f64,

    /// Atmospheric pressure (Pa)
    /// Default: 101325 (1 atm)
    pub atmospheric_pressure: f64,

    /// Ground level reference (m)
    /// Default: 0.0
    pub ground_level: f64,
}

impl PhysicsConstants {
    /// Create with Earth defaults
    pub fn new() -> Self;

    /// Create with custom gravity
    pub fn with_gravity(gravity: f64) -> Self;
}

impl Default for PhysicsConstants {
    fn default() -> Self;
}
```

### Algorithm Constants

**Location:** `src/interactions/gjk_collision_3d.rs`

```rust
/// Numerical tolerance for floating-point comparisons
const EPSILON: f64 = 1e-12;

/// Maximum iterations for GJK algorithm
const GJK_MAX_ITERATIONS: usize = 32;

/// Maximum iterations for EPA algorithm
const EPA_MAX_ITERATIONS: usize = 64;

/// Convergence tolerance for EPA
const EPA_TOLERANCE: f64 = 1e-6;
```

### Physical Constants

```rust
/// Gravitational constant (N⋅m²/kg²)
pub const G: f64 = 6.67430e-11;

/// Speed of light (m/s)
pub const C: f64 = 299_792_458.0;

/// Planck constant (J⋅s)
pub const H: f64 = 6.62607015e-34;

/// Boltzmann constant (J/K)
pub const K_B: f64 = 1.380649e-23;
```

---

## Module Re-exports

**Location:** `src/lib.rs`

```rust
// Core modules (always available)
pub mod utils;
pub mod physics;
pub mod models;
pub mod forces;
pub mod interactions;
pub mod apis;

// Feature-gated modules
#[cfg(feature = "rotational_dynamics")]
pub mod rotational_dynamics;

#[cfg(feature = "thermodynamics")]
pub mod thermodynamics;

#[cfg(feature = "fluid_dynamics")]
pub mod fluid_dynamics;

#[cfg(feature = "materials")]
pub mod materials;

#[cfg(feature = "constraints")]
pub mod constraints;

#[cfg(feature = "particles")]
pub mod particles;
```

---

## Usage Examples

### Basic Collision

```rust
use rs_physics::apis::easy_physics::EasyPhysics;
use rs_physics::utils::errors::PhysicsError;

fn main() -> Result<(), PhysicsError> {
    let physics = EasyPhysics::new();

    let mut ball1 = physics.create_object(1.0, 5.0, 0.0)?;
    let mut ball2 = physics.create_object(2.0, -3.0, 10.0)?;

    physics.simulate_collision(&mut ball1, &mut ball2, 0.0, 0.1, 0.47, 1.0)?;

    println!("Ball 1 velocity: {} m/s", ball1.velocity);
    println!("Ball 2 velocity: {} m/s", ball2.velocity);

    Ok(())
}
```

### 3D Physics with GJK

```rust
use rs_physics::models::object_3d::PhysicalObject3D;
use rs_physics::models::shape_3d::Shape3D;
use rs_physics::interactions::gjk_collision_3d::*;

fn main() {
    let sphere = PhysicalObject3D::new(
        1.0,
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        Shape3D::Sphere(1.0)
    );

    let cube = PhysicalObject3D::new(
        2.0,
        (1.5, 0.0, 0.0),
        (-0.5, 0.0, 0.0),
        Shape3D::Cuboid(1.0, 1.0, 1.0)
    );

    let (collision, simplex) = gjk_collision_detection(&sphere, &cube);

    if collision {
        if let Some((depth, normal, point)) = epa_contact_points(&sphere, &cube, &simplex) {
            println!("Collision detected!");
            println!("Penetration depth: {}", depth);
            println!("Contact normal: {:?}", normal);
            println!("Contact point: {:?}", point);
        }
    }
}
```

### Custom Physics Constants

```rust
use rs_physics::utils::constants_config::PhysicsConstants;
use rs_physics::apis::easy_physics::EasyPhysics;

fn main() {
    // Simulate on the Moon
    let moon_constants = PhysicsConstants {
        gravity: 1.62,
        air_density: 0.0,  // No atmosphere
        ..Default::default()
    };

    let physics = EasyPhysics::with_constants(moon_constants);
    // ...
}
```

---

*For more detailed explanations, see [PHYSICS_SYSTEMS.md](./PHYSICS_SYSTEMS.md) and [COLLISION_DETECTION.md](./COLLISION_DETECTION.md).*
