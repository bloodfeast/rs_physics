//! # rs_physics
//!
//! A comprehensive physics simulation library for Rust, providing tools for
//! rigid body dynamics, collision detection, fluid simulation, thermodynamics,
//! and more.
//!
//! ## Features
//!
//! This crate is organized into feature-gated modules:
//!
//! - **Core** (always enabled): Basic physics types, vectors, forces, and rigid body dynamics
//! - **`constraints`**: Joint and spring constraint solvers
//! - **`materials`**: Material properties and collision response calculations
//! - **`fluid_dynamics`**: Analytical fluid calculations (drag, buoyancy, Reynolds number)
//! - **`fluid_simulation`**: Grid-based Eulerian fluid simulation (2D and 3D)
//! - **`thermodynamics`**: Heat transfer, thermodynamic processes, and thermal simulation
//! - **`gpu`**: GPU-accelerated computations using wgpu
//!
//! ## Quick Start
//!
//! ```rust
//! use rs_physics::physics::create_constants;
//!
//! // Create physics constants (gravity, air density, speed of sound, pressure, ground level)
//! let constants = create_constants(Some(9.81), Some(1.225), Some(343.0), Some(101325.0), Some(0.0));
//! ```
//!
//! ## Module Overview
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`physics`] | Core physics types: `PhysicsBody`, `Vector3D`, rigid body dynamics |
//! | [`interactions`] | Collision detection: GJK/EPA, broad-phase, continuous collision |
//! | [`forces`] | Force generators: gravity, springs, drag |
//! | [`constraints`] | Constraint solvers: joints, springs, iterative solver |
//! | [`materials`] | Material properties: density, elasticity, thermal properties |
//! | [`fluid_dynamics`] | Fluid calculations: drag, buoyancy, flow analysis |
//! | [`thermodynamics`] | Heat transfer, thermal grids, phase transitions |
//! | [`particles`] | Particle systems and emitters |
//! | [`world`] | Physics world container for managing simulations |
//!
//! ## Feature Flags
//!
//! Enable features in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! rs_physics = { version = "0.1", features = ["thermodynamics", "fluid_simulation"] }
//! ```
//!
//! ## Limitations
//!
//! - All calculations use f64 precision
//! - Units are SI (meters, kilograms, seconds, Kelvin, Pascals)
//! - The [`world`] module runs the simulation on a background thread and
//!   communicates with the main thread over channels; the calculation modules
//!   themselves are single-threaded (use rayon externally for parallelism)
//! - Collision detection is for convex shapes only (use convex decomposition for concave)

pub mod utils;
pub mod apis;
pub mod physics;
pub mod interactions;
pub mod forces;
pub mod constraints;
pub mod rotational_dynamics;
pub mod fluid_dynamics;
pub mod thermodynamics;
pub mod materials;
pub mod models;
pub mod particles;
pub mod world;

#[cfg(feature = "gpu")]
pub mod gpu;

/// Prelude module for convenient imports.
///
/// Import the entire prelude with:
/// ```rust
/// use rs_physics::prelude::*;
/// ```
///
/// This re-exports the most commonly used types and functions from the library.
pub mod prelude {
    // Core types
    pub use crate::utils::{PhysicsConstants, PhysicsError, DEFAULT_PHYSICS_CONSTANTS};
    pub use crate::utils::vector3::{Vec3, cross_product, dot_product, magnitude, normalize};

    // Physics calculations
    pub use crate::physics::{
        create_constants,
        calculate_force,
        calculate_acceleration,
        calculate_velocity,
        calculate_kinetic_energy,
        calculate_potential_energy,
        calculate_momentum,
        calculate_work,
        calculate_power,
    };

    // Collision detection - use the gjk_collision_3d module directly
    pub use crate::interactions::gjk_collision_3d::{
        gjk_collision_detection,
        gjk_collision_detection_ex,
        epa_contact_points,
        ContactInfo,
        GjkResult,
    };

    // Shape from models module
    pub use crate::models::Shape3D;

    // Materials
    pub use crate::materials::{Material, calculate_collision_response, calculate_stress};

    // Constraints
    pub use crate::constraints::{Joint, Spring, ConstraintSolver, IterativeConstraintSolver};

    // Thermodynamics (when enabled)
    #[cfg(feature = "thermodynamics")]
    pub use crate::thermodynamics::{
        ThermalGrid,
        ThermalGrid3D,
        ThermalBoundaryCondition,
        GridSide,
        Substance,
        celsius_to_kelvin,
        kelvin_to_celsius,
        carnot_efficiency,
    };

    // Fluid dynamics (when enabled)
    #[cfg(feature = "fluid_dynamics")]
    pub use crate::fluid_dynamics::{
        Fluid,
        calculate_reynolds_number,
        calculate_drag_force,
        calculate_buoyant_force,
    };

    // Fluid simulation (when enabled)
    #[cfg(feature = "fluid_simulation")]
    pub use crate::fluid_dynamics::{
        FluidGrid,
        FluidGrid3D,
        BoundaryType,
    };
}

/// ### General helper function
/// - Asserts that two floating point numbers are approximately equal.
///
/// ### Arguments
///
/// * `a` - The first floating point number.
/// * `b` - The second floating point number.
/// * `epsilon` - The maximum difference between `a` and `b` for them to be considered equal.
/// * `optional_message` - An optional message to display if the assertion fails.
///
pub fn assert_float_eq(a: f64, b: f64, epsilon: f64, optional_message: Option<&str>) {
    match optional_message {
        Some(message) => assert!((a - b).abs() < epsilon, "a: {:?},\nb: {:?},\nepsilon: {:?},\n message: {:?}", a, b, epsilon, message),
        None => assert!((a - b).abs() < epsilon, "Expected {} to be approximately equal to {} (epsilon: {})", a, b, epsilon),
    }
}

