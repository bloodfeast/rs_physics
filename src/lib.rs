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
//! | [`articulated`] | Skeletons of jointed capsules, contacts, and piles of them |
//! | [`materials`] | Material properties: density, elasticity, thermal properties |
//! | [`fluid_dynamics`] | Fluid calculations: drag, buoyancy, flow analysis |
//! | [`thermodynamics`] | Heat transfer, thermal grids, phase transitions |
//! | [`particles`] | Particle systems and emitters |
//! | [`world`] | Physics world container for managing simulations |
//!
//! ## Skeletons and piles
//!
//! [`articulated`] is the newest and largest of these, and it is a different shape from
//! [`constraints`]. A constraint there owns its two bodies by value, which is right for
//! one joint between two things and wrong for a skeleton -- a forearm is the second body
//! of the elbow and the first body of the wrist, and two copies of it do not converge.
//!
//! An [`articulated::Skeleton`] holds **one** set of bodies as parallel arrays and gives
//! joints indices into it, so a whole figure, or a few hundred of them, is solved at once:
//!
//! ```rust
//! use rs_physics::articulated::{Body, Joint, Skeleton};
//!
//! let mut s = Skeleton::new();
//! s.set_ground((0.0, 1.0, 0.0), 0.0);
//!
//! // A pinned anchor, and a limb hanging off it by a ball joint.
//! let anchor = s.add_body(Body::pinned((0.0, 2.0, 0.0)));
//! let limb = s.add_body(Body::capsule(4.0, 0.06, 0.4, (0.0, 1.6, 0.0)));
//! // A shoulder, with the range a shoulder has: the limb may swing 1.2 radians from
//! // the anchor's own axis and no further. `Joint::free_ball` is the same joint with no
//! // range on it.
//! s.add_joint(Joint::socket(
//!     anchor,
//!     limb,
//!     (0.0, 0.0, 0.0),
//!     (0.0, 0.2, 0.0),
//!     (0.0, -1.0, 0.0),
//!     (0.0, -1.0, 0.0),
//!     1.2,
//! ));
//!
//! for _ in 0..60 {
//!     s.step(1.0 / 60.0, (0.0, -9.80665, 0.0), 8);
//! }
//!
//! // The anchor is pinned, so it has not moved.
//! assert_eq!(s.position(anchor), (0.0, 2.0, 0.0));
//! ```
//!
//! What it provides: ball and hinge joints with a range of motion, capsule-against-capsule
//! and capsule-against-ground contacts with Coulomb friction and rolling resistance, a
//! uniform-grid broad phase, and a solve that is parallel across constraints -- the
//! constraints are graph-coloured so that no two in a colour name the same body, and a
//! pass hands the whole coloured plan to a pool of workers once rather than forking per
//! colour.
//!
//! **What has settled leaves the simulation.** Bodies that stop moving are put to sleep
//! and woken by anything that reaches them, in islands, so that a heap which has come to
//! rest costs the time it takes to scan a bit per body and nothing else. On ten thousand
//! capsules at rest that is the difference between milliseconds a step and tens of
//! nanoseconds. The threshold is not a tuned number: a body is settling when it moves
//! less than a small fraction of *its own size* over the time it would take to fall that
//! far, which means one rule serves a finger bone and a torso.
//!
//! It is **position-based** (XPBD), which is what makes a pile of hundreds of jointed
//! bodies tractable at all: corrections are applied to positions and velocities are read
//! back out of them, so the solve is stable at large timesteps where a force-based
//! integrator of the same scene would need many more, much smaller, steps.
//!
//! Its behaviour is pinned by `tests/articulated_laws.rs`, which checks the properties
//! that must hold however the solver is implemented: that the answer is bit-identical run
//! to run and does not depend on how many threads computed it, that `friction` really is
//! Coulomb's coefficient and means the same thing at any iteration count, that a resting
//! capsule sits exactly one radius above what it rests on, that a skeleton left to itself
//! does not move its own centre of mass, and that a closed system never ends with more
//! energy than it began with.
//!
//! Those laws are stated as physics rather than as recorded output, and they are measured
//! over several starting states rather than one, because a settling pile is chaotic: a
//! difference of one unit in the last place compounds, and a law that draws once reports
//! which draw it took rather than what the solver does. Three of them had to be rewritten
//! during development for exactly that reason, and each says so where its bound is set.
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
pub mod articulated;
pub mod rotational_dynamics;
pub mod fluid_dynamics;
pub mod thermodynamics;
pub mod materials;
/// Sound as a physical quantity: propagation, absorption, reflection and direction.
pub mod acoustics;
/// The state of the air, and the closed-form results for wind near the ground.
///
/// Owns [`atmosphere::Air`], which is the crate's single description of air — density,
/// viscosity, speed of sound and acoustic absorption all derive from the same three
/// numbers. Also holds [`atmosphere::boundary_layer`]: the logarithmic wind profile,
/// Jackson–Hunt speed-up over a rise, and the Cionco canopy profile. Not feature-gated,
/// because [`acoustics`] is not and depends on it.
pub mod atmosphere;
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

    // The air, and the wind near the ground. Ungated, because `atmosphere` is — see
    // `lib.rs`'s module list. Gating the re-export and not the module (or the reverse) is
    // the parity bug this crate's feature matrix is most exposed to.
    pub use crate::atmosphere::{
        Air, CanopyAttenuation, HillForm, Surface, WindProfile, LINEARISATION_SLOPE_LIMIT,
    };

    // Rigid-body rotation. Ungated, because `rotational_dynamics`'s inertia and angular
    // submodules are — only the legacy submodule sits behind the same-named feature.
    // Gating the re-export and not the module (or the reverse) is the parity bug this
    // crate's feature matrix is most exposed to.
    pub use crate::rotational_dynamics::{InertiaTensor, RigidBodyRotation, inertia_3d};

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
        // Thin-film surface flow. Gated identically to `mod thin_film` in
        // `fluid_dynamics/mod.rs`; the two gates must stay the same string or this
        // re-export names something that does not exist in a single-flag build.
        FilmFlow,
        FilmGrid,
        blood_apparent_viscosity,
        puddle_depth,
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

