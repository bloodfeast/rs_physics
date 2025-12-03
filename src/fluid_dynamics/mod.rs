//! # Fluid Dynamics Module
//!
//! Fluid mechanics calculations and grid-based fluid simulation.
//!
//! This module provides two complementary approaches to fluid mechanics:
//! analytical calculations (drag, buoyancy, Reynolds number) and Eulerian
//! grid-based fluid simulation for 2D and 3D incompressible flow.
//!
//! ## Features
//!
//! ### Analytical (`fluid_dynamics` feature)
//! - Drag force calculation (sphere, cylinder, flat plate)
//! - Buoyancy force calculation
//! - Reynolds number and flow regime classification
//! - Bernoulli equation applications
//! - Predefined fluid properties (water, air, oil, etc.)
//!
//! ### Simulation (`fluid_simulation` feature)
//! - 2D/3D incompressible Navier-Stokes solver
//! - Pressure projection for incompressibility
//! - Configurable boundary conditions
//! - Particle-fluid coupling for two-way interaction
//!
//! ## Quick Start - Analytical
//!
//! ```rust,ignore
//! use rs_physics::fluid_dynamics::{Fluid, calculate_drag_force, reynolds_number};
//!
//! let water = Fluid::water();
//! let velocity = 2.0;  // m/s
//! let diameter = 0.1;  // m
//!
//! // Calculate Reynolds number
//! let re = reynolds_number(water.density, velocity, diameter, water.dynamic_viscosity);
//! println!("Reynolds number: {:.0}", re);
//!
//! // Calculate drag force on a sphere
//! let drag = calculate_drag_force(water.density, velocity, std::f64::consts::PI * diameter * diameter / 4.0, 0.47);
//! println!("Drag force: {:.2} N", drag);
//! ```
//!
//! ## Quick Start - Simulation
//!
//! ```rust,ignore
//! use rs_physics::fluid_dynamics::FluidGrid;
//!
//! // Create a 100x100 fluid grid
//! let mut grid = FluidGrid::new(100, 100, 0.01, 0.001, 1.0e-6)
//!     .expect("Valid grid parameters");
//!
//! // Add velocity source
//! grid.set_velocity(50, 50, 1.0, 0.0);
//!
//! // Simulate
//! for _ in 0..100 {
//!     grid.step();
//! }
//! ```
//!
//! ## Predefined Fluids
//!
//! | Fluid | Density (kg/m³) | Viscosity (Pa·s) |
//! |-------|-----------------|------------------|
//! | Water | 1000 | 1.0e-3 |
//! | Air | 1.225 | 1.8e-5 |
//! | Oil | 900 | 0.1 |
//! | Honey | 1400 | 2.5 |
//!
//! ## Flow Regimes
//!
//! Reynolds number determines flow behavior:
//! - Re < 1: Stokes flow (creeping)
//! - Re < 2300: Laminar flow
//! - 2300 < Re < 4000: Transitional
//! - Re > 4000: Turbulent flow
//!
//! ## Limitations
//!
//! - **Incompressible flow only**: No compressibility effects
//! - **No turbulence modeling**: DNS-style simulation
//! - **Fixed grid**: No adaptive mesh refinement
//! - **No multiphase flow**: Single fluid type per simulation
//! - **Explicit time integration**: Requires small timesteps
//!
//! ## Performance
//!
//! - Grid simulation: O(n) per substep for advection/diffusion
//! - Pressure solve: O(iterations × n) using Gauss-Seidel
//! - Memory: ~40 bytes per cell (velocity, pressure, density)

// Shared modules - available when any fluid feature is enabled
#[cfg(any(feature = "fluid_dynamics", feature = "fluid_simulation"))]
mod validation;
#[cfg(any(feature = "fluid_dynamics", feature = "fluid_simulation"))]
pub use validation::*;

#[cfg(any(feature = "fluid_dynamics", feature = "fluid_simulation"))]
mod solver;
#[cfg(any(feature = "fluid_dynamics", feature = "fluid_simulation"))]
pub use solver::*;

// Analytical fluid dynamics (drag, buoyancy, Reynolds number, etc.)
#[cfg(feature = "fluid_dynamics")]
mod fluid_dynamics;
#[cfg(feature = "fluid_dynamics")]
pub use fluid_dynamics::*;

// Grid-based fluid simulation (Eulerian solver)
#[cfg(feature = "fluid_simulation")]
mod fluid_simulation;
#[cfg(feature = "fluid_simulation")]
pub use fluid_simulation::*;

// 3D Grid-based fluid simulation
#[cfg(feature = "fluid_simulation")]
mod fluid_simulation_3d;
#[cfg(feature = "fluid_simulation")]
pub use fluid_simulation_3d::*;

// Particle-fluid coupling (two-way interaction)
#[cfg(feature = "fluid_simulation")]
mod particle_coupling;
#[cfg(feature = "fluid_simulation")]
pub use particle_coupling::*;

#[cfg(test)]
#[cfg(feature = "fluid_dynamics")]
mod fluid_dynamics_tests;
#[cfg(test)]
#[cfg(feature = "fluid_simulation")]
mod fluid_simulation_tests;