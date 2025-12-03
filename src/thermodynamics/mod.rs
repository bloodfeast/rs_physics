//! # Thermodynamics Module
//!
//! Comprehensive thermodynamic calculations and thermal simulation.
//!
//! This module provides tools for heat transfer analysis, thermodynamic processes,
//! heat engine cycles, phase transitions, and grid-based thermal simulations
//! in both 2D and 3D.
//!
//! ## Features
//!
//! - **Ideal gas law**: PV = nRT calculations and state functions
//! - **Heat transfer**: Conduction, convection, and radiation modes
//! - **Processes**: Isothermal, adiabatic, isobaric, isochoric transformations
//! - **Cycles**: Carnot, Otto, Diesel efficiency calculations
//! - **Phase transitions**: Latent heat, boiling/melting points
//! - **Thermal grids**: 2D and 3D heat diffusion simulation
//!
//! ## Quick Start
//!
//! ```rust
//! use rs_physics::thermodynamics::{ThermalGrid, ThermalBoundaryCondition, GridSide};
//!
//! // Create a 2D thermal grid (50x50, starting at 300K)
//! let mut grid = ThermalGrid::new(
//!     50, 50,           // dimensions
//!     300.0,            // initial temperature (K)
//!     1.0e-4,           // thermal diffusivity (m²/s)
//!     0.001,            // timestep (s)
//!     0.01,             // grid spacing (m)
//! ).expect("Valid grid parameters");
//!
//! // Set boundary conditions
//! grid.set_boundary_condition(GridSide::Left, ThermalBoundaryCondition::Dirichlet(400.0));
//! grid.set_boundary_condition(GridSide::Right, ThermalBoundaryCondition::Dirichlet(300.0));
//!
//! // Run simulation
//! for _ in 0..1000 {
//!     grid.step();
//! }
//!
//! println!("Average temperature: {:.1} K", grid.average_temperature());
//! ```
//!
//! ## Module Contents
//!
//! | Component | Description |
//! |-----------|-------------|
//! | Physical constants | R, k_B, σ (Stefan-Boltzmann) |
//! | Substances | Predefined water, air, copper, etc. |
//! | Ideal gas law | PV=nRT, internal energy, enthalpy |
//! | Heat transfer | Conduction, convection, radiation |
//! | Processes | Isothermal, adiabatic, isobaric, isochoric |
//! | Cycles | Carnot, Otto, Diesel efficiency |
//! | Phase transitions | Latent heat, boiling/melting points |
//! | [`ThermalGrid`] | 2D heat diffusion simulation |
//! | [`ThermalGrid3D`] | 3D heat diffusion simulation |
//!
//! ## Boundary Conditions
//!
//! The thermal grids support multiple boundary conditions:
//!
//! - **Dirichlet**: Fixed temperature at boundary
//! - **Neumann**: Fixed heat flux at boundary
//! - **Convective**: Heat transfer with ambient (Newton's law of cooling)
//! - **Insulated**: No heat flow (zero flux)
//!
//! ## Stability Considerations
//!
//! The thermal grid uses explicit finite difference (FTCS scheme).
//! For numerical stability, the CFL condition must be satisfied:
//!
//! - 2D: `α × dt / dx² ≤ 0.25`
//! - 3D: `α × dt / dx² ≤ 1/6 ≈ 0.167`
//!
//! Where α is thermal diffusivity, dt is timestep, dx is grid spacing.
//!
//! ## Example: Heat Engine Efficiency
//!
//! ```rust
//! use rs_physics::thermodynamics::{carnot_efficiency, otto_efficiency};
//!
//! // Carnot efficiency between 600K hot reservoir and 300K cold reservoir
//! let carnot = carnot_efficiency(600.0, 300.0).unwrap();
//! println!("Carnot efficiency: {:.1}%", carnot * 100.0); // 50%
//!
//! // Otto cycle with compression ratio 10:1
//! let otto = otto_efficiency(10.0, 1.4).unwrap(); // γ = 1.4 for air
//! println!("Otto efficiency: {:.1}%", otto * 100.0); // ~60%
//! ```
//!
//! ## Limitations
//!
//! - **Ideal gas only**: No real gas corrections (van der Waals, etc.)
//! - **Constant properties**: Material properties don't vary with temperature
//! - **No convection currents**: Thermal grids simulate conduction only
//! - **Single-phase**: No phase change during grid simulation
//! - **Isotropic**: Same thermal conductivity in all directions
//!
//! ## Performance
//!
//! - Thermal grid: O(n) per timestep where n = width × height (× depth for 3D)
//! - Memory: 16 bytes per cell (double-buffered f64 temperatures)
//!
//! Requires the `thermodynamics` feature flag.

// Shared utilities
#[cfg(feature = "thermodynamics")]
mod validation;
#[cfg(feature = "thermodynamics")]
pub use validation::*;

#[cfg(feature = "thermodynamics")]
mod constants;
#[cfg(feature = "thermodynamics")]
pub use constants::*;

#[cfg(feature = "thermodynamics")]
mod substance;
#[cfg(feature = "thermodynamics")]
pub use substance::*;

// Core thermodynamics
#[cfg(feature = "thermodynamics")]
mod thermodynamics;
#[cfg(feature = "thermodynamics")]
pub use thermodynamics::*;

// Heat transfer modes (conduction, convection, radiation)
#[cfg(feature = "thermodynamics")]
mod heat_transfer;
#[cfg(feature = "thermodynamics")]
pub use heat_transfer::*;

// Thermodynamic processes (isothermal, adiabatic, isobaric, isochoric)
#[cfg(feature = "thermodynamics")]
mod processes;
#[cfg(feature = "thermodynamics")]
pub use processes::*;

// Heat engines and thermodynamic cycles (Carnot, Otto, Diesel, Brayton)
#[cfg(feature = "thermodynamics")]
mod cycles;
#[cfg(feature = "thermodynamics")]
pub use cycles::*;

// Phase transitions and latent heat
#[cfg(feature = "thermodynamics")]
mod phase_transitions;
#[cfg(feature = "thermodynamics")]
pub use phase_transitions::*;

// 2D thermal grid simulation
#[cfg(feature = "thermodynamics")]
mod thermal_grid;
#[cfg(feature = "thermodynamics")]
pub use thermal_grid::*;

// 3D thermal grid simulation
#[cfg(feature = "thermodynamics")]
mod thermal_grid_3d;
#[cfg(feature = "thermodynamics")]
pub use thermal_grid_3d::*;

#[cfg(test)]
#[cfg(feature = "thermodynamics")]
mod thermodynamics_tests;