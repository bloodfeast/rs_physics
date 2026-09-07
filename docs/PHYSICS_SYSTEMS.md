# Physics Systems

This document covers the core physics systems in rs_physics, including kinematics, dynamics, forces, energy calculations, and advanced physics modules.

## Table of Contents

1. [Core Physics](#core-physics)
2. [Force System](#force-system)
3. [Object Models](#object-models)
4. [Rotational Dynamics](#rotational-dynamics)
5. [Constraint Solving](#constraint-solving)
6. [Particle Systems](#particle-systems)
7. [Fluid Dynamics](#fluid-dynamics)
8. [Thermodynamics](#thermodynamics)
9. [Material System](#material-system)

---

## Core Physics

**Location:** `src/physics/physics.rs`

### Kinematics

Kinematics deals with motion without considering forces.

#### Velocity
```rust
/// Calculate final velocity: v = v₀ + at
pub fn calculate_velocity(
    initial_velocity: f64,
    acceleration: f64,
    time: f64
) -> Result<f64, PhysicsError>
```

#### Average Velocity
```rust
/// Calculate average velocity over time period
pub fn calculate_average_velocity(
    initial_velocity: f64,
    final_velocity: f64
) -> f64
```

#### Acceleration
```rust
/// Calculate acceleration: a = (v - v₀) / t
pub fn calculate_acceleration(
    initial_velocity: f64,
    final_velocity: f64,
    time: f64
) -> Result<f64, PhysicsError>
```

#### Terminal Velocity
```rust
/// Calculate terminal velocity for falling object
/// v_t = sqrt(2mg / ρAC_d)
pub fn calculate_terminal_velocity(
    mass: f64,
    drag_coefficient: f64,
    area: f64,
    constants: &PhysicsConstants
) -> Result<f64, PhysicsError>
```

### Dynamics

Dynamics considers forces and their effects on motion.

#### Force (Newton's Second Law)
```rust
/// Calculate force: F = ma
pub fn calculate_force(
    mass: f64,
    acceleration: f64
) -> Result<f64, PhysicsError>
```

#### Momentum
```rust
/// Calculate momentum: p = mv
pub fn calculate_momentum(
    mass: f64,
    velocity: f64
) -> Result<f64, PhysicsError>
```

#### Impulse
```rust
/// Calculate impulse: J = Ft = Δp
pub fn calculate_impulse(
    force: f64,
    time: f64
) -> Result<f64, PhysicsError>
```

#### Coefficient of Restitution
```rust
/// Calculate coefficient of restitution: e = v₂'/v₁
pub fn calculate_coefficient_of_restitution(
    velocity_before: f64,
    velocity_after: f64
) -> Result<f64, PhysicsError>
```

### Energy

#### Kinetic Energy
```rust
/// Calculate kinetic energy: KE = ½mv²
pub fn calculate_kinetic_energy(
    mass: f64,
    velocity: f64
) -> Result<f64, PhysicsError>
```

#### Potential Energy
```rust
/// Calculate gravitational potential energy: PE = mgh
pub fn calculate_potential_energy(
    mass: f64,
    height: f64,
    constants: &PhysicsConstants
) -> Result<f64, PhysicsError>
```

#### Work
```rust
/// Calculate work done: W = Fd cos(θ)
pub fn calculate_work(
    force: f64,
    distance: f64,
    angle: f64  // radians
) -> Result<f64, PhysicsError>
```

#### Power
```rust
/// Calculate power: P = W/t
pub fn calculate_power(
    work: f64,
    time: f64
) -> Result<f64, PhysicsError>
```

### Circular Motion

#### Centripetal Force
```rust
/// Calculate centripetal force: F = mv²/r
pub fn calculate_centripetal_force(
    mass: f64,
    velocity: f64,
    radius: f64
) -> Result<f64, PhysicsError>
```

#### Torque
```rust
/// Calculate torque: τ = rF sin(θ)
pub fn calculate_torque(
    radius: f64,
    force: f64,
    angle: f64
) -> Result<f64, PhysicsError>
```

#### Angular Velocity
```rust
/// Calculate angular velocity: ω = v/r
pub fn calculate_angular_velocity(
    velocity: f64,
    radius: f64
) -> Result<f64, PhysicsError>
```

### Projectile Motion

#### Time of Flight
```rust
/// Calculate time of flight for projectile
/// t = 2v₀sin(θ) / g
pub fn calculate_projectile_time_of_flight(
    initial_velocity: f64,
    angle: f64,
    constants: &PhysicsConstants
) -> Result<f64, PhysicsError>
```

#### Maximum Height
```rust
/// Calculate maximum height of projectile
/// h = v₀²sin²(θ) / 2g
pub fn calculate_projectile_max_height(
    initial_velocity: f64,
    angle: f64,
    constants: &PhysicsConstants
) -> Result<f64, PhysicsError>
```

### Air Resistance

#### Air Resistance Force
```rust
/// Calculate air resistance (drag force)
/// F_d = -½ρv²AC_d
pub fn calculate_air_resistance(
    velocity: f64,
    drag_coefficient: f64,
    area: f64,
    constants: &PhysicsConstants
) -> Result<f64, PhysicsError>
```

---

## Force System

**Location:** `src/forces/forces.rs`, `src/forces/forces_2d.rs`

### Force Types

```rust
pub enum Force {
    /// Gravitational force: F = mg (downward)
    Gravity { mass: f64 },

    /// Drag force: F = -½ρv²AC_d (opposes motion)
    Drag {
        drag_coefficient: f64,
        area: f64,
        velocity: f64,
    },

    /// Spring force (Hooke's law): F = -kx
    Spring {
        spring_constant: f64,
        displacement: f64,
    },

    /// Constant force in any direction
    Constant { magnitude: f64 },

    /// Thrust force with direction
    Thrust {
        magnitude: f64,
        angle: f64,  // radians
    },
}
```

### Force Application

#### Scalar (1D)
```rust
impl Force {
    /// Apply force as scalar value
    pub fn apply(&self, constants: &PhysicsConstants) -> f64 {
        match self {
            Force::Gravity { mass } => -mass * constants.gravity,
            Force::Drag { drag_coefficient, area, velocity } => {
                -0.5 * constants.air_density
                     * velocity.abs() * velocity
                     * drag_coefficient * area
            }
            Force::Spring { spring_constant, displacement } => {
                -spring_constant * displacement
            }
            Force::Constant { magnitude } => *magnitude,
            Force::Thrust { magnitude, angle } => {
                magnitude * angle.cos()
            }
        }
    }
}
```

#### Vector (2D)
```rust
impl Force {
    /// Apply force as 2D vector (fx, fy)
    pub fn apply_2d(&self, constants: &PhysicsConstants) -> (f64, f64) {
        match self {
            Force::Gravity { mass } => (0.0, -mass * constants.gravity),
            Force::Thrust { magnitude, angle } => (
                magnitude * angle.cos(),
                magnitude * angle.sin()
            ),
            // ...
        }
    }
}
```

#### Vector (3D)
```rust
impl Force {
    /// Apply force as 3D vector (fx, fy, fz)
    pub fn apply_3d(&self, constants: &PhysicsConstants) -> (f64, f64, f64) {
        match self {
            Force::Gravity { mass } => (0.0, -mass * constants.gravity, 0.0),
            // ...
        }
    }
}
```

### Force Accumulation

Objects accumulate forces over a timestep:

```rust
// Object with force collection
pub struct Object {
    pub mass: f64,
    pub velocity: f64,
    pub position: f64,
    pub forces: Vec<Force>,
}

impl Object {
    /// Sum all forces and calculate acceleration
    pub fn net_force(&self, constants: &PhysicsConstants) -> f64 {
        self.forces.iter()
            .map(|f| f.apply(constants))
            .sum()
    }

    /// Update velocity and position
    pub fn update(&mut self, dt: f64, constants: &PhysicsConstants) {
        let net_f = self.net_force(constants);
        let accel = net_f / self.mass;
        self.velocity += accel * dt;
        self.position += self.velocity * dt;
    }
}
```

---

## Object Models

**Location:** `src/models/`

### 1D Object

```rust
pub struct Object {
    pub mass: f64,
    pub velocity: f64,
    pub position: f64,
    pub forces: Vec<Force>,
}
```

### 2D Object

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
    /// Calculate speed magnitude
    pub fn speed(&self) -> f64 {
        (self.velocity.x.powi(2) + self.velocity.y.powi(2)).sqrt()
    }

    /// Calculate direction angle
    pub fn direction(&self) -> f64 {
        self.velocity.y.atan2(self.velocity.x)
    }
}
```

### 3D Object

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
```

### Physical Object (Full 3D with Shape)

```rust
pub struct PhysicalObject3D {
    pub object: ObjectIn3D,
    pub shape: Shape3D,
    pub orientation: (f64, f64, f64),  // Euler angles (roll, pitch, yaw)
    pub angular_velocity: (f64, f64, f64),
    pub inertia_tensor: [[f64; 3]; 3],
}

impl PhysicalObject3D {
    /// Create from basic parameters
    pub fn new(
        mass: f64,
        position: (f64, f64, f64),
        velocity: (f64, f64, f64),
        shape: Shape3D
    ) -> Self {
        let inertia = shape.moment_of_inertia(mass);
        Self {
            object: ObjectIn3D {
                mass,
                position: Axis3D { x: position.0, y: position.1, z: position.2 },
                velocity: Velocity3D { x: velocity.0, y: velocity.1, z: velocity.2 },
                forces: vec![],
            },
            shape,
            orientation: (0.0, 0.0, 0.0),
            angular_velocity: (0.0, 0.0, 0.0),
            inertia_tensor: inertia,
        }
    }
}
```

### Quaternion

```rust
pub struct Quaternion {
    pub w: f64,  // scalar
    pub x: f64,  // i
    pub y: f64,  // j
    pub z: f64,  // k
}

impl Quaternion {
    /// Create from Euler angles (ZYX convention)
    pub fn from_euler(roll: f64, pitch: f64, yaw: f64) -> Self;

    /// Create from axis-angle representation
    pub fn from_axis_angle(axis: (f64, f64, f64), angle: f64) -> Self;

    /// Convert to Euler angles
    pub fn to_euler(&self) -> (f64, f64, f64);

    /// Rotate a 3D point
    pub fn rotate_point(&self, point: (f64, f64, f64)) -> (f64, f64, f64);

    /// Quaternion multiplication
    pub fn multiply(&self, other: &Quaternion) -> Quaternion;

    /// Normalize to unit quaternion
    pub fn normalize(&self) -> Quaternion;

    /// Inverse rotation
    pub fn inverse(&self) -> Quaternion;
}
```

---

## Rotational Dynamics

**Location:** `src/rotational_dynamics/rotational_dynamics.rs`
**Feature:** `rotational_dynamics`

### Moment of Inertia

Each shape computes its inertia tensor:

```rust
impl Shape3D {
    /// Calculate moment of inertia tensor
    pub fn moment_of_inertia(&self, mass: f64) -> [[f64; 3]; 3] {
        match self {
            Shape3D::Sphere(r) => {
                // I = (2/5)mr² for solid sphere
                let i = 0.4 * mass * r.powi(2);
                [[i, 0.0, 0.0],
                 [0.0, i, 0.0],
                 [0.0, 0.0, i]]
            }
            Shape3D::Cuboid(w, h, d) => {
                // I_x = (1/12)m(h² + d²), etc.
                let ix = mass * (h.powi(2) + d.powi(2)) / 12.0;
                let iy = mass * (w.powi(2) + d.powi(2)) / 12.0;
                let iz = mass * (w.powi(2) + h.powi(2)) / 12.0;
                [[ix, 0.0, 0.0],
                 [0.0, iy, 0.0],
                 [0.0, 0.0, iz]]
            }
            Shape3D::Cylinder(r, h) => {
                // Cylinder about its axis
                let i_axial = 0.5 * mass * r.powi(2);
                let i_transverse = mass * (3.0 * r.powi(2) + h.powi(2)) / 12.0;
                [[i_transverse, 0.0, 0.0],
                 [0.0, i_axial, 0.0],
                 [0.0, 0.0, i_transverse]]
            }
            // ...
        }
    }
}
```

### Angular Momentum

```rust
/// Calculate angular momentum: L = Iω
pub fn calculate_angular_momentum(
    inertia_tensor: [[f64; 3]; 3],
    angular_velocity: (f64, f64, f64)
) -> (f64, f64, f64) {
    matrix_vector_multiply(inertia_tensor, angular_velocity)
}
```

### Torque and Angular Acceleration

```rust
/// Calculate angular acceleration from torque: α = I⁻¹τ
pub fn calculate_angular_acceleration(
    inertia_tensor: [[f64; 3]; 3],
    torque: (f64, f64, f64)
) -> (f64, f64, f64) {
    let inv_inertia = invert_3x3(inertia_tensor);
    matrix_vector_multiply(inv_inertia, torque)
}
```

---

## Constraint Solving

**Location:** `src/constraints/constraint_solvers.rs`
**Feature:** `constraints`

### Constraint Solver Trait

```rust
pub trait ConstraintSolver {
    /// Solve constraint, returns true if converged
    fn solve(&mut self, objects: &mut [PhysicalObject3D]) -> bool;

    /// Calculate current constraint error
    fn calculate_error(&self, objects: &[PhysicalObject3D]) -> f64;
}
```

### Joint Constraint

Maintains fixed distance between two objects:

```rust
pub struct Joint {
    pub object_a_index: usize,
    pub object_b_index: usize,
    pub anchor_a: (f64, f64, f64),  // Local anchor on A
    pub anchor_b: (f64, f64, f64),  // Local anchor on B
    pub rest_length: f64,
    pub stiffness: f64,
}

impl ConstraintSolver for Joint {
    fn solve(&mut self, objects: &mut [PhysicalObject3D]) -> bool {
        let world_a = get_world_anchor(objects, self.object_a_index, self.anchor_a);
        let world_b = get_world_anchor(objects, self.object_b_index, self.anchor_b);

        let delta = subtract(world_b, world_a);
        let distance = magnitude(delta);
        let error = distance - self.rest_length;

        if error.abs() < TOLERANCE {
            return true;
        }

        // Apply correction
        let correction = scale(normalize(delta), error * self.stiffness * 0.5);
        objects[self.object_a_index].object.position += correction;
        objects[self.object_b_index].object.position -= correction;

        false
    }
}
```

### Spring Constraint

```rust
pub struct Spring {
    pub object_a_index: usize,
    pub object_b_index: usize,
    pub rest_length: f64,
    pub spring_constant: f64,
    pub damping: f64,
}

impl ConstraintSolver for Spring {
    fn solve(&mut self, objects: &mut [PhysicalObject3D]) -> bool {
        let pos_a = objects[self.object_a_index].object.position;
        let pos_b = objects[self.object_b_index].object.position;

        let delta = subtract(pos_b, pos_a);
        let distance = magnitude(delta);
        let displacement = distance - self.rest_length;

        // Spring force: F = -kx
        let force_mag = self.spring_constant * displacement;
        let force_dir = normalize(delta);
        let force = scale(force_dir, force_mag);

        // Apply forces
        apply_force(&mut objects[self.object_a_index], force);
        apply_force(&mut objects[self.object_b_index], negate(force));

        // Add damping
        let vel_a = objects[self.object_a_index].object.velocity;
        let vel_b = objects[self.object_b_index].object.velocity;
        let rel_vel = subtract(vel_b, vel_a);
        let damping_force = scale(rel_vel, -self.damping);
        // ...

        displacement.abs() < TOLERANCE
    }
}
```

---

## Particle Systems

**Location:** `src/particles/`
**Feature:** `particles`, `particles-cosmological`

### Particle Structure

```rust
pub struct Particle {
    pub mass: f64,
    pub position: (f64, f64, f64),
    pub velocity: (f64, f64, f64),
    pub acceleration: (f64, f64, f64),
    pub lifetime: Option<f64>,  // None = infinite
    pub age: f64,
}
```

### Particle System

```rust
pub struct ParticleSystem {
    pub particles: Vec<Particle>,
    pub gravity: (f64, f64, f64),
    pub damping: f64,
}

impl ParticleSystem {
    /// Update all particles
    pub fn update(&mut self, dt: f64) {
        self.particles.retain_mut(|p| {
            // Apply gravity
            p.acceleration = add(p.acceleration, self.gravity);

            // Integrate velocity
            p.velocity = add(p.velocity, scale(p.acceleration, dt));
            p.velocity = scale(p.velocity, 1.0 - self.damping);

            // Integrate position
            p.position = add(p.position, scale(p.velocity, dt));

            // Reset acceleration
            p.acceleration = (0.0, 0.0, 0.0);

            // Update lifetime
            p.age += dt;
            match p.lifetime {
                Some(max_age) => p.age < max_age,
                None => true,
            }
        });
    }
}
```

### Barnes-Hut N-Body

**Location:** `src/particles/particle_interactions_barnes_hut.rs`

Optimizes gravitational calculations from O(N²) to O(N log N):

```rust
pub struct BarnesHutTree {
    root: Option<Box<OctreeNode>>,
    theta: f64,  // Opening angle (typically 0.5-1.0)
}

struct OctreeNode {
    center: (f64, f64, f64),
    half_size: f64,
    mass: f64,
    center_of_mass: (f64, f64, f64),
    children: [Option<Box<OctreeNode>>; 8],
    particle: Option<usize>,  // Leaf nodes store particle index
}

impl BarnesHutTree {
    /// Build tree from particles
    pub fn build(particles: &[Particle], bounds: Bounds) -> Self;

    /// Calculate forces on all particles
    pub fn calculate_forces(&self, particles: &mut [Particle], g: f64) {
        for i in 0..particles.len() {
            let force = self.calculate_force_on_particle(i, particles, g);
            particles[i].acceleration = scale(force, 1.0 / particles[i].mass);
        }
    }

    /// Recursive force calculation with approximation
    fn calculate_force_on_particle(
        &self,
        particle_idx: usize,
        particles: &[Particle],
        g: f64
    ) -> (f64, f64, f64) {
        self.traverse_node(&self.root, particle_idx, particles, g)
    }

    fn traverse_node(
        &self,
        node: &Option<Box<OctreeNode>>,
        particle_idx: usize,
        particles: &[Particle],
        g: f64
    ) -> (f64, f64, f64) {
        match node {
            None => (0.0, 0.0, 0.0),
            Some(n) => {
                let p = &particles[particle_idx];
                let d = distance(p.position, n.center_of_mass);

                // If node is far enough, use center of mass approximation
                if n.half_size / d < self.theta {
                    gravitational_force(p.mass, n.mass, p.position, n.center_of_mass, g)
                } else if let Some(other_idx) = n.particle {
                    // Leaf node: direct calculation
                    if other_idx != particle_idx {
                        let other = &particles[other_idx];
                        gravitational_force(p.mass, other.mass, p.position, other.position, g)
                    } else {
                        (0.0, 0.0, 0.0)
                    }
                } else {
                    // Internal node: recurse
                    let mut total = (0.0, 0.0, 0.0);
                    for child in &n.children {
                        total = add(total, self.traverse_node(child, particle_idx, particles, g));
                    }
                    total
                }
            }
        }
    }
}
```

---

## Fluid Dynamics

**Location:** `src/fluid_dynamics/`
**Feature:** `fluid_dynamics`, `fluid_simulation`

### Reynolds Number

```rust
/// Calculate Reynolds number: Re = ρvL/μ
pub fn reynolds_number(
    density: f64,
    velocity: f64,
    characteristic_length: f64,
    dynamic_viscosity: f64
) -> f64 {
    density * velocity * characteristic_length / dynamic_viscosity
}
```

### Drag Force

```rust
/// Calculate drag force: F_d = ½ρv²C_dA
pub fn drag_force(
    density: f64,
    velocity: f64,
    drag_coefficient: f64,
    area: f64
) -> f64 {
    0.5 * density * velocity.powi(2) * drag_coefficient * area
}
```

### Buoyancy

```rust
/// Calculate buoyant force (Archimedes): F_b = ρ_fluid * V * g
pub fn buoyant_force(
    fluid_density: f64,
    submerged_volume: f64,
    gravity: f64
) -> f64 {
    fluid_density * submerged_volume * gravity
}
```

### Eulerian Fluid Simulation

**Location:** `src/fluid_dynamics/fluid_simulation.rs`

Grid-based fluid simulation using velocity and pressure fields.

---

## Thermodynamics

**Location:** `src/thermodynamics/thermodynamics.rs`
**Feature:** `thermodynamics`

### Heat Transfer

```rust
/// Calculate heat transferred: Q = mcΔT
pub fn heat_transfer(
    mass: f64,
    specific_heat: f64,
    temperature_change: f64
) -> f64 {
    mass * specific_heat * temperature_change
}
```

### Entropy Change

```rust
/// Calculate entropy change: ΔS = Q/T (reversible process)
pub fn entropy_change(
    heat: f64,
    temperature: f64
) -> Result<f64, PhysicsError> {
    if temperature <= 0.0 {
        return Err(PhysicsError::DivisionByZero);
    }
    Ok(heat / temperature)
}
```

---

## Material System

**Location:** `src/materials/materials.rs`
**Feature:** `materials`

### Material Properties

```rust
pub struct Material {
    pub name: String,
    pub density: f64,           // kg/m³
    pub elastic_modulus: f64,   // Young's modulus (Pa)
    pub poisson_ratio: f64,     // 0.0 to 0.5
    pub yield_strength: f64,    // Pa
    pub ultimate_strength: f64, // Pa
    pub hardness: f64,          // Vickers hardness
    pub friction_coefficient: f64,
}

impl Material {
    /// Predefined materials
    pub fn steel() -> Self;
    pub fn aluminum() -> Self;
    pub fn wood() -> Self;
    pub fn polyurethane() -> Self;
    // ...
}
```

### Stress-Strain Analysis

```rust
/// Calculate stress: σ = F/A
pub fn calculate_stress(force: f64, area: f64) -> Result<f64, PhysicsError>;

/// Calculate strain: ε = ΔL/L
pub fn calculate_strain(change_in_length: f64, original_length: f64) -> Result<f64, PhysicsError>;

/// Check if material has failed (stress > ultimate strength)
pub fn check_failure(material: &Material, stress: f64) -> bool {
    stress > material.ultimate_strength
}
```

---

## Physics Constants

**Location:** `src/utils/constants_config.rs`

```rust
pub struct PhysicsConstants {
    pub gravity: f64,              // Default: 9.80665 m/s²
    pub air_density: f64,          // Default: 1.225 kg/m³
    pub speed_of_sound: f64,       // Default: 343.0 m/s
    pub atmospheric_pressure: f64, // Default: 101325 Pa
    pub ground_level: f64,         // Default: 0.0
}

impl PhysicsConstants {
    /// Create with default Earth values
    pub fn new() -> Self;

    /// Create with custom gravity (e.g., for Moon, Mars)
    pub fn with_gravity(gravity: f64) -> Self;
}
```

---

*See [API_REFERENCE.md](./API_REFERENCE.md) for complete function signatures.*
