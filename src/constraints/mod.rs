//! # Constraints Module
//!
//! Physics constraint solvers for maintaining relationships between objects.
//!
//! This module provides constraint types and iterative solvers for enforcing
//! physical constraints like joints and springs between physics bodies.
//!
//! ## Features
//!
//! - **Joint constraints**: Maintain fixed distance between two objects
//! - **Spring constraints**: Elastic connections with configurable stiffness and damping
//! - **Iterative solver**: Converges constraints over multiple iterations
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use rs_physics::constraints::{Joint, Spring, IterativeConstraintSolver};
//! use rs_physics::models::Object;
//!
//! // Create an iterative solver
//! let mut solver = IterativeConstraintSolver::new(10, 0.001); // 10 iterations, 0.001 tolerance
//!
//! // Add a joint constraint between two objects
//! let joint = Joint {
//!     object1: Object::new(1.0, 0.0, 0.0),
//!     object2: Object::new(1.0, 5.0, 0.0),
//!     constraint_distance: 5.0,
//! };
//! solver.add_constraint(Box::new(joint));
//!
//! // Solve constraints
//! solver.solve(0.016).expect("Solver should converge");
//! ```
//!
//! ## Constraint Types
//!
//! | Type | Description | Use Case |
//! |------|-------------|----------|
//! | [`Joint`] | Fixed distance constraint (1D) | Rigid connections, chains |
//! | [`Spring`] | Elastic constraint with damping (1D) | Soft connections, suspension |
//! | [`Joint2D`] | Fixed distance constraint (2D) | 2D rigid connections |
//! | [`Spring2D`] | Elastic constraint with damping (2D) | 2D soft connections |
//! | [`Joint3D`] | Fixed distance constraint (3D) | 3D rigid connections |
//! | [`Spring3D`] | Elastic constraint with damping (3D) | 3D soft connections |
//! | [`Rope2D`] | Maximum length constraint (2D) | Ropes, cables, tethers |
//! | [`Rope3D`] | Maximum length constraint (3D) | Ropes, cables, tethers |
//! | [`RopeChain2D`] | Multi-segment rope (2D) | Realistic rope rendering |
//! | [`RopeChain3D`] | Multi-segment rope (3D) | Realistic rope rendering |
//! | [`Fixed2D`] | Fixed offset constraint (2D) | Weld joints, rigid attachments |
//! | [`Fixed3D`] | Fixed offset constraint (3D) | Weld joints, rigid attachments |
//! | [`Hinge3D`] | Revolute constraint (3D) | Doors, wheels, pendulums |
//! | [`Contact2D`] | Contact constraint (2D) | Collision response |
//! | [`Contact3D`] | Contact constraint (3D) | Collision response |
//! | [`UnifiedSolver2D`] | Unified 2D solver with warm starting | Multiple 2D constraints |
//! | [`UnifiedSolver3D`] | Unified 3D solver with warm starting | Multiple 3D constraints |
//!
//! ## Limitations
//!
//! - **Iterative solver for 1D only**: The IterativeConstraintSolver works with 1D constraints
//! - **2D/3D constraints standalone**: Joint2D, Spring2D, Joint3D, Spring3D work independently
//! - **Convergence**: Iterative solver may not converge for stiff systems
//!
//! ## Performance
//!
//! - Solver complexity: O(iterations × constraints) per timestep
//!
//! Requires the `constraints` feature flag.

#[cfg(feature = "constraints")]
mod constraint_solvers;
#[cfg(feature = "constraints")]
pub use constraint_solvers::*;

#[cfg(feature = "constraints")]
mod constraint_solvers_2d;
#[cfg(feature = "constraints")]
pub use constraint_solvers_2d::*;

#[cfg(feature = "constraints")]
mod constraint_solvers_3d;
#[cfg(feature = "constraints")]
pub use constraint_solvers_3d::*;

#[cfg(feature = "constraints")]
mod rope_constraint;
#[cfg(feature = "constraints")]
pub use rope_constraint::*;

#[cfg(feature = "constraints")]
mod fixed_constraint;
#[cfg(feature = "constraints")]
pub use fixed_constraint::*;

#[cfg(feature = "constraints")]
mod hinge_constraint;
#[cfg(feature = "constraints")]
pub use hinge_constraint::*;

#[cfg(feature = "constraints")]
mod contact_constraint;
#[cfg(feature = "constraints")]
pub use contact_constraint::*;

#[cfg(feature = "constraints")]
mod solver;
#[cfg(feature = "constraints")]
pub use solver::*;

#[cfg(feature = "constraints")]
#[cfg(test)]
mod constraint_solvers_tests;

#[cfg(feature = "constraints")]
#[cfg(test)]
mod solver_tests;