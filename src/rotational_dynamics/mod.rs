//! Rotational Dynamics Module
//!
//! This module provides unified rotational physics for both 2D and 3D simulations.
//!
//! ## Core Components
//!
//! - **Inertia types**: `InertiaScalar` (2D), `InertiaTensor` (3D)
//! - **Angular states**: `AngularState2D`, `AngularState3D`
//! - **Utility functions**: collision impulse calculations, point velocity, etc.
//!
//! ## Feature Flags
//!
//! The legacy `rotational_dynamics` feature is still supported for backwards compatibility,
//! but new code should use the types directly from this module.
//!
//! ## Example
//!
//! ```rust,ignore
//! use rs_physics::rotational_dynamics::{
//!     InertiaScalar, InertiaTensor, AngularState2D, AngularState3D,
//!     inertia_3d, angular_impulse_from_collision,
//! };
//!
//! // 2D rotation
//! let mut state_2d = AngularState2D::new(1.0, 0.0);
//! let inertia_2d = InertiaScalar::from(2.0);
//! state_2d.apply_torque(4.0, 0.1, inertia_2d);
//!
//! // 3D rotation
//! let mut state_3d = AngularState3D::from_components(0.0, 1.0, 0.0);
//! let inertia_3d = InertiaTensor::uniform(5.0);
//! state_3d.apply_torque((1.0, 0.0, 0.0), 0.1, &inertia_3d);
//! ```

// New unified modules (always available)
mod inertia;
mod angular;

pub use inertia::*;
pub use angular::*;

// Legacy module (feature-gated for backwards compatibility)
#[cfg(feature = "rotational_dynamics")]
mod rotational_dynamics;
#[cfg(feature = "rotational_dynamics")]
pub use rotational_dynamics::*;

#[cfg(test)]
#[cfg(feature = "rotational_dynamics")]
mod rotational_dynamics_tests;
