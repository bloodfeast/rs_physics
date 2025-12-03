//! # Physics Module
//!
//! Core physics calculations for dynamics, kinematics, and mechanics.
//!
//! This module provides fundamental physics calculations including:
//! - Kinematics (velocity, acceleration, displacement)
//! - Dynamics (force, momentum, impulse)
//! - Energy (kinetic, potential, work, power)
//! - Projectile motion (time of flight, max height)
//! - Circular motion (centripetal force, angular velocity, torque)
//! - Collision mechanics (coefficient of restitution)
//!
//! ## Quick Start
//!
//! ```rust
//! use rs_physics::physics::{create_constants, calculate_kinetic_energy, calculate_force};
//! use rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS;
//!
//! // Create custom physics constants
//! let constants = create_constants(Some(9.81), Some(1.225), Some(343.0), Some(101325.0), Some(0.0));
//!
//! // Calculate kinetic energy: KE = 0.5 * m * v²
//! let kinetic_energy = calculate_kinetic_energy(&DEFAULT_PHYSICS_CONSTANTS, 10.0, 5.0);
//!
//! // Calculate force: F = m * a
//! let force = calculate_force(&DEFAULT_PHYSICS_CONSTANTS, 10.0, 9.81);
//! ```
//!
//! ## Physics Constants
//!
//! All calculations use a `PhysicsConstants` struct that contains:
//!
//! | Constant | Default Value | Unit |
//! |----------|---------------|------|
//! | Gravity | 9.81 | m/s² |
//! | Air density | 1.225 | kg/m³ |
//! | Speed of sound | 343.0 | m/s |
//! | Atmospheric pressure | 101325.0 | Pa |
//!
//! ## Error Handling
//!
//! Functions in this module log errors and attempt to recover when possible:
//! - Negative masses are converted to absolute values
//! - Invalid angles are clamped to valid ranges
//! - Division by zero returns 0.0 with an error log
//!
//! ## Limitations
//!
//! - All calculations assume SI units (meters, kilograms, seconds)
//! - Projectile motion assumes constant gravity and no air resistance
//! - Circular motion assumes uniform motion

mod physics;
#[cfg(test)]
mod physics_tests;

pub use physics::*;