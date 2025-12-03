//! # Interactions Module
//!
//! Collision detection and response for physics simulations.
//!
//! This module provides comprehensive collision detection algorithms including
//! broad-phase culling, narrow-phase GJK/EPA, and continuous collision detection
//! for both 2D and 3D scenarios.
//!
//! ## Features
//!
//! - **GJK algorithm**: Gilbert-Johnson-Keerthi for convex shape intersection
//! - **EPA algorithm**: Expanding Polytope Algorithm for penetration depth
//! - **Broad-phase**: Spatial partitioning for efficient culling
//! - **CCD**: Continuous collision detection for fast-moving objects
//! - **2D and 3D support**: Separate optimized implementations
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use rs_physics::interactions::gjk_collision_3d::{ConvexShape, GJK};
//! use rs_physics::physics::Vector3D;
//!
//! // Create two spheres
//! let sphere1 = ConvexShape::Sphere {
//!     center: Vector3D::new(0.0, 0.0, 0.0),
//!     radius: 1.0,
//! };
//! let sphere2 = ConvexShape::Sphere {
//!     center: Vector3D::new(1.5, 0.0, 0.0),
//!     radius: 1.0,
//! };
//!
//! // Check for intersection
//! let gjk = GJK::new();
//! if gjk.intersects(&sphere1, &sphere2) {
//!     println!("Shapes are colliding!");
//! }
//! ```
//!
//! ## Algorithm Overview
//!
//! | Algorithm | Purpose | Complexity |
//! |-----------|---------|------------|
//! | GJK | Detect intersection | O(n) average |
//! | EPA | Find penetration depth | O(n²) worst case |
//! | Broad-phase | Cull non-colliding pairs | O(n log n) |
//! | CCD | Detect tunneling | O(iterations) |
//!
//! ## Supported Shapes
//!
//! - Sphere
//! - Box (AABB and OBB)
//! - Capsule
//! - Convex hull (arbitrary point cloud)
//! - Cylinder
//!
//! ## Limitations
//!
//! - **Convex shapes only**: Concave shapes must be decomposed into convex parts
//! - **No deformable bodies**: Shapes are assumed rigid
//! - **EPA precision**: May have numerical issues for nearly-touching objects
//! - **CCD iterations**: May miss very fast tunneling if iterations are too low

mod interactions;
mod interactions_2d;
mod interactions_3d;

pub use interactions::*;
pub use interactions_2d::*;
pub use interactions_3d::*;
pub mod shape_collisions_3d;
pub mod gjk_collision_3d;
pub mod continuous_collision_detection;

#[cfg(test)]
mod interactions_tests;
#[cfg(test)]
mod interactions_2d_tests;
#[cfg(test)]
mod interactions_3d_tests;
#[cfg(test)]
mod shape_collisions_3d_tests;
#[cfg(test)]
mod gjk_collision_3d_tests;
#[cfg(test)]
mod continuous_collision_detection_tests;