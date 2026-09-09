//! Rotational Dynamics Module
//!
//! This module provides unified rotational physics for both 2D and 3D simulations.
//!
//! ## Core Components
//!
//! - **Inertia types**: `InertiaScalar` (2D), `InertiaTensor` (3D)
//! - **Rigid-body rotation in 3D**: [`RigidBodyRotation`] — Euler's equation,
//!   `ω̇ = I⁻¹(τ − ω × Iω)`, with a cached inverse tensor
//! - **Angular states**: `AngularState2D`, `AngularState3D` — torque and impulse
//!   *accumulators*, not integrators; see below
//! - **Utility functions**: collision impulse calculations, point velocity, etc.
//!
//! ## Which type to reach for in 3D
//!
//! **[`RigidBodyRotation`] is the rigid-body integrator.** If you want a body to rotate
//! the way a real one does — to wobble, precess, or tumble — that is the type. It owns
//! `ω`, so the only way to advance it is through `step`, and `step` cannot omit the
//! gyroscopic term.
//!
//! [`AngularState3D`] is a bare `ω` with helpers that add to it. Its `apply_torque`
//! integrates `Δω = I⁻¹τ·Δt` and **omits `ω × Iω`**, so a free body handed zero torque
//! through it never changes its angular velocity: no wobble, no intermediate-axis flip,
//! no tumble. It is deprecated for that reason. `apply_impulse` is *not* deprecated —
//! an impulse acts over a vanishing time, so `Δω = I⁻¹J` is exact.
//!
//! In 2D there is no such distinction. `AngularState2D::apply_torque` is complete as
//! written: a scalar moment about a fixed axis has no gyroscopic term.
//!
//! ## Feature Flags
//!
//! **This module is not gated by the feature that shares its name.** `InertiaTensor`,
//! `inertia_3d`, the angular states and [`RigidBodyRotation`] are all available in a
//! `--no-default-features` build. Only the legacy `rotational_dynamics` submodule sits
//! behind the `rotational_dynamics` flag, and it is kept for backwards compatibility;
//! new code should use the types directly from this module.
//!
//! ## Example
//!
//! ```rust,ignore
//! use rs_physics::rotational_dynamics::{
//!     InertiaScalar, InertiaTensor, AngularState2D, RigidBodyRotation,
//!     inertia_3d, angular_impulse_from_collision,
//! };
//!
//! // 2D rotation
//! let mut state_2d = AngularState2D::new(1.0, 0.0);
//! let inertia_2d = InertiaScalar::from(2.0);
//! state_2d.apply_torque(4.0, 0.1, inertia_2d);
//!
//! // 3D rotation of a rigid body
//! let mut body = RigidBodyRotation::new(inertia_3d::solid_cuboid(1.0, 0.2, 0.1, 0.05))?;
//! body.set_angular_velocity_body((0.05, 8.0, 0.0))?;
//! body.apply_torque_body((0.0, 0.0, 0.3))?;
//! body.step(1.0 / 120.0)?;
//! ```

// New unified modules (always available)
mod inertia;
mod angular;
mod rigid_body_rotation;

pub use inertia::*;
pub use angular::*;
pub use rigid_body_rotation::*;

// Legacy module (feature-gated for backwards compatibility)
#[cfg(feature = "rotational_dynamics")]
mod rotational_dynamics;
#[cfg(feature = "rotational_dynamics")]
pub use rotational_dynamics::*;

#[cfg(test)]
#[cfg(feature = "rotational_dynamics")]
mod rotational_dynamics_tests;
