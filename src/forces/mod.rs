//! # Forces Module
//!
//! Force generators and physics systems for 2D and 3D simulations.
//!
//! This module provides force generators and physics system management for
//! simulating various physical forces on objects in both 2D and 3D space.
//!
//! ## Features
//!
//! - **Force Generators**: Pre-built generators for common forces
//!   - Gravity (constant downward force)
//!   - Spring forces (Hooke's law)
//!   - Drag forces (velocity-dependent resistance)
//!   - Constant forces (user-defined direction and magnitude)
//!
//! - **Physics Systems**: Manage collections of objects with forces
//!   - 2D system (`PhysicsSystem2D`)
//!   - 3D system (`PhysicsSystem`)
//!
//! ## Quick Start - 3D
//!
//! ```rust,ignore
//! use rs_physics::forces::{PhysicsSystem, PhysicsObject, GravityForce};
//!
//! // Create a physics system
//! let mut system = PhysicsSystem::new();
//!
//! // Add an object
//! let object = PhysicsObject::new(1.0, [0.0, 10.0, 0.0], [0.0, 0.0, 0.0]);
//! system.add_object(object);
//!
//! // Apply gravity and update
//! system.apply_gravity();
//! system.update(0.016); // ~60 FPS timestep
//! ```
//!
//! ## Quick Start - 2D
//!
//! ```rust,ignore
//! use rs_physics::forces::{PhysicsSystem2D, PhysicsObject2D, GravityForce2D};
//!
//! // Create a 2D physics system
//! let mut system = PhysicsSystem2D::new();
//!
//! // Add an object with position and velocity
//! let object = PhysicsObject2D::new(1.0, [0.0, 10.0], [0.0, 0.0]);
//! system.add_object(object);
//!
//! // Apply gravity and update
//! system.apply_gravity();
//! system.update(0.016);
//! ```
//!
//! ## Force Types
//!
//! | Force | Formula | Description |
//! |-------|---------|-------------|
//! | Gravity | F = mg | Constant downward force |
//! | Spring | F = -kx | Restoring force proportional to displacement |
//! | Drag | F = -cv | Resistance proportional to velocity |
//! | Constant | F = const | User-defined force vector |
//!
//! ## Limitations
//!
//! - Forces are applied uniformly to point masses (no torque)
//! - Spring forces use simple Hooke's law (no damping by default)
//! - Drag is linear (not quadratic like real air resistance at high speeds)
//! - No collision detection or response (see `interactions` module)

mod forces;
mod forces_2d;

pub use forces::*;
pub use forces_2d::*;
#[cfg(test)]
mod forces_tests;
#[cfg(test)]
mod forces_2d_tests;