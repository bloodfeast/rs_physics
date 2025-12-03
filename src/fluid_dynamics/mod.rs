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