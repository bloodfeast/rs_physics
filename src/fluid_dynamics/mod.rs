//! # Fluid Dynamics Module
//!
//! Fluid mechanics calculations and grid-based fluid simulation.
//!
//! This module provides two complementary approaches to fluid mechanics:
//! analytical calculations (drag, buoyancy, Reynolds number) and simulation -- Eulerian
//! grids for 2D and 3D incompressible flow in a box, particles for splashes, and a
//! shallow-water heightfield for rivers, lakes and floods.
//!
//! ## Features
//!
//! ### Analytical (`fluid_dynamics` feature)
//! - Drag force `½ρv²·C_d·A`, from a drag coefficient and reference area you supply
//!   (there is no built-in `C_d(Re)` correlation)
//! - Buoyant force `ρ·V·g`, and the thermal (Boussinesq) buoyancy of hot gas
//! - Reynolds number `ρvL/μ` (no regime classifier; see *Flow Regimes* below)
//! - Darcy–Weisbach pipe pressure drop
//! - Fluid presets (water, seawater, oil, honey, glycerin, blood), and air derived from
//!   an [`crate::atmosphere::Air`] state through `Fluid::from_air`
//! - Thin-film surface flow -- the lubrication approximation (`FilmFlow`, `FilmGrid`)
//!
//! ### Simulation (`fluid_simulation` feature)
//! - `FluidGrid` / `FluidGrid3D`: incompressible Navier-Stokes in a closed box, with
//!   pressure projection. Every wall blocks normal flow, and is free-slip along it by
//!   default or no-slip on request (`WallCondition`); there is no inflow, outflow or
//!   free surface.
//! - `ShallowWater`: rivers, lakes and floods on a terrain heightfield -- a free surface,
//!   inflow and outflow edges, bed friction, wetting and drying, and exact conservation.
//!   This is the one to use for water a player walks beside.
//! - `SphFluid`: particles that are the fluid, for splashes and droplets.
//! - Particle-fluid coupling for two-way interaction with the grids.
//!
//! ## Quick Start - Analytical
//!
//! ```rust
//! # #[cfg(feature = "fluid_dynamics")] {
//! use rs_physics::fluid_dynamics::{Fluid, calculate_drag_force, calculate_reynolds_number};
//!
//! let water = Fluid::water();
//! let velocity = 1.0;  // m/s, relative to the water
//! let diameter = 0.1;  // m
//!
//! // Reynolds number on the diameter: about 10^5, where a sphere's C_d is about 0.47.
//! let re = calculate_reynolds_number(&water, velocity, diameter).unwrap();
//! assert!(re > 1.0e3 && re < 2.0e5);
//!
//! // Drag on a sphere. The area is the *frontal* area, pi d^2 / 4, because that is the
//! // area the 0.47 is defined against. The result is a magnitude in newtons, acting
//! // against the relative velocity.
//! let frontal_area = std::f64::consts::PI * diameter * diameter / 4.0;
//! let drag = calculate_drag_force(&water, velocity, frontal_area, 0.47).unwrap();
//! assert!((drag - 0.5 * 998.0 * 1.0 * frontal_area * 0.47).abs() < 1e-9);
//! # }
//! ```
//!
//! ## Quick Start - Simulation
//!
//! ```rust
//! # #[cfg(feature = "fluid_simulation")] {
//! use rs_physics::fluid_dynamics::FluidGrid;
//!
//! // A 100x100 grid: 98x98 fluid cells inside a one-cell boundary ring. Length is
//! // in domain widths, so diffusion and viscosity are in widths²/s and velocity in
//! // widths/s.
//! let mut grid = FluidGrid::new(100, 100, 1.0e-5, 1.0e-5, 1.0 / 60.0)
//!     .expect("Valid grid parameters");
//!
//! // A source inside the fluid (row and column 0 and 99 are the boundary ring)
//! grid.add_density(50, 50, 1.0).unwrap();
//! grid.add_velocity(50, 50, 0.5, 0.0).unwrap();
//!
//! // Simulate
//! for _ in 0..10 {
//!     grid.step();
//! }
//! assert!(grid.validate_state().is_ok());
//! # }
//! ```
//!
//! ## Predefined Fluids
//!
//! What the constructors actually return. The doctest below the table asserts every
//! row, so the table cannot drift from the code again.
//!
//! | Constructor | State | Density (kg/m³) | Viscosity (Pa·s) |
//! |-------------|-------|-----------------|------------------|
//! | `Fluid::water()` | 20 °C | 998 | 1.0e-3 |
//! | `Fluid::seawater()` | 20 °C | 1025 | 1.08e-3 |
//! | `Fluid::oil()` | SAE 30, 40 °C | 876 | 0.1 |
//! | `Fluid::honey()` | 20 °C | 1420 | 10.0 |
//! | `Fluid::glycerin()` | 20 °C | 1261 | 1.412 |
//! | `Fluid::blood()` | 37 °C, at 300 s⁻¹ | 1060 | 4.0e-3 |
//! | `Fluid::from_air(&Air::sea_level())` | ICAO, 15 °C | 1.225 | 1.79e-5 |
//!
//! There is no `Fluid::air()`: air is a state, not a constant, so it comes from an
//! [`crate::atmosphere::Air`].
//!
//! ```rust
//! # #[cfg(feature = "fluid_dynamics")] {
//! use rs_physics::atmosphere::Air;
//! use rs_physics::fluid_dynamics::Fluid;
//!
//! let rows = [
//!     (Fluid::water(), 998.0, 1.0e-3),
//!     (Fluid::seawater(), 1025.0, 1.08e-3),
//!     (Fluid::oil(), 876.0, 0.1),
//!     (Fluid::honey(), 1420.0, 10.0),
//!     (Fluid::glycerin(), 1261.0, 1.412),
//!     (Fluid::blood(), 1060.0, 4.0e-3),
//!     (Fluid::from_air(&Air::sea_level()), 1.225, 1.79e-5),
//! ];
//! for (fluid, density, viscosity) in rows {
//!     assert!((fluid.density - density).abs() / density < 1e-3);
//!     assert!((fluid.viscosity - viscosity).abs() / viscosity < 1e-3);
//! }
//! # }
//! ```
//!
//! ## Flow Regimes
//!
//! There is no regime classifier. `calculate_reynolds_number` returns Re, and where
//! the transitions fall depends on the geometry and on which length you passed:
//!
//! - **Pipe flow**, Re on the diameter: laminar below about 2300, turbulent above
//!   about 4000.
//! - **Open channels and rivers**, Re on the hydraulic radius: laminar below about
//!   500, turbulent above a few thousand (Chow, *Open-Channel Hydraulics*, 1959). Using
//!   the pipe thresholds with a hydraulic radius misplaces the transition by about 4×.
//! - **A sphere**, Re on the diameter: Stokes drag (`C_d = 24/Re`) only below about
//!   Re = 1; `C_d ≈ 0.4–0.5` from about 10³ to 2×10⁵.
//!
//! ## Limitations
//!
//! - **Incompressible flow only**: No compressibility effects
//! - **Turbulence: what the grids resolve, and two options for keeping it.** The grids
//!   solve the flow they can resolve and model nothing below a cell. Two options in
//!   `SolverConfig` keep more of what they resolve: `AdvectionScheme::MacCormack`
//!   (second-order, clamped advection that keeps a plume's edges and its eddies'
//!   strength) and `VorticityConfinement::MatchNumericalDissipation` (a force that puts
//!   back the rotation first-order advection removes, with its strength derived from
//!   that advection's numerical viscosity, so there is no constant to tune). Both are
//!   off by default. There is no sub-grid stress model, deliberately: at these
//!   resolutions first-order advection already removes more than a Smagorinsky model
//!   would (its numerical viscosity `h^2 a(1 - a) / (2 dt)` is of order `|u| h / 2` at a
//!   Courant number below one, against Smagorinsky's `(0.17 h)^2 |S|`, about
//!   `0.03 |u| h` for a shear of `|u| / h`), so adding one would only smooth further
//! - **Confinement on a collocated grid**: the projection cannot remove a divergence
//!   at the grid scale, and confinement feeds that scale, so a confined flow keeps a
//!   grid-scale divergence (rms 0.06% of the vorticity on a confined Taylor-Green
//!   array with the energy cap, 4% without it; the total over the box is still zero)
//! - **Fixed grid**: No adaptive mesh refinement
//! - **No multiphase flow**: Single fluid type per simulation; the box grids have no
//!   free surface (`ShallowWater` does, depth-averaged)
//! - **Grid units**: length is in domain widths (cell size `1 / width` on every
//!   axis), not metres; see `FluidGrid`
//! - **A variable pressure cost**: the pressure is solved by conjugate gradient to a
//!   relative tolerance (default 1e-2; see `SolverConfig::pressure_tolerance` for the
//!   cost of each setting), so a step costs more when it brings more new divergence. `PressureSolver::Relaxation` restores the old fixed-cost sweeps, which
//!   remove only about 1% of a large-scale divergence per step at 128²
//! - **Stability is not accuracy**: implicit diffusion and semi-Lagrangian advection
//!   are stable at any timestep, but a step that moves the flow more than a cell or
//!   so smears it (first-order numerical diffusion)
//!
//! ## Performance
//!
//! - Grid simulation: O(n) per substep for advection/diffusion
//! - Pressure solve: MIC(0)-preconditioned conjugate gradient, warm-started from the
//!   last step. With a steady source, about 13/19/29 iterations per step at 64²/128²/256²
//!   and 15/17 at 32³/64³ (default tolerance)
//! - Memory, all of it kept between steps so a step allocates nothing: 112 bytes per
//!   cell in 2D and 128 in 3D (velocity, density, two warm-start pressures, the
//!   preconditioner, the solver's work vectors and the step's copies of the diffused
//!   state). `SolverType::Jacobi` adds 8 and `PressureSolver::Relaxation` 16;
//!   `AdvectionScheme::MacCormack` adds 24 in either dimension (a reverse pass and the
//!   forward pass's bounds); `VorticityConfinement::MatchNumericalDissipation` adds 24
//!   in 2D and 32 in 3D (the curl, its magnitude and the uncapped force)

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

// Thin-film surface flow: the lubrication approximation. Gated with the analytical
// module rather than with `fluid_simulation`, because it is a *law* — pure functions
// on scalars, meant to be transcribed into a compute shader — and because it is built
// on `Fluid`, which lives behind this same flag. Gating it on `fluid_simulation`
// instead would let a `--features fluid_dynamics` build see `Fluid` and not the film
// law that consumes it.
#[cfg(feature = "fluid_dynamics")]
mod thin_film;
#[cfg(feature = "fluid_dynamics")]
pub use thin_film::*;

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

// Shallow-water equations on a terrain heightfield: rivers, lakes, floods. Depth-averaged,
// so its cost scales with the map's area rather than its volume.
#[cfg(feature = "fluid_simulation")]
mod shallow_water;
#[cfg(feature = "fluid_simulation")]
pub use shallow_water::*;

// Particle-fluid coupling (two-way interaction)
#[cfg(feature = "fluid_simulation")]
mod particle_coupling;
#[cfg(feature = "fluid_simulation")]
pub use particle_coupling::*;

// Smoothed-particle hydrodynamics: a Lagrangian solver where the particles *are*
// the fluid, rather than solids moving through one. Fills the gap between the
// Eulerian grid (too coarse for droplets) and particle coupling (no interaction
// between particles at all).
#[cfg(feature = "fluid_simulation")]
mod sph;
#[cfg(feature = "fluid_simulation")]
pub use sph::*;

#[cfg(test)]
#[cfg(feature = "fluid_dynamics")]
mod fluid_dynamics_tests;
#[cfg(test)]
#[cfg(feature = "fluid_simulation")]
mod fluid_simulation_tests;
#[cfg(test)]
#[cfg(feature = "fluid_dynamics")]
mod analytic_regression_tests;
#[cfg(test)]
#[cfg(feature = "fluid_simulation")]
mod grid_regression_tests;
#[cfg(test)]
#[cfg(feature = "fluid_simulation")]
mod turbulence_tests;
