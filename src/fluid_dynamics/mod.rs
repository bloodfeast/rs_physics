#[cfg(feature = "fluid_dynamics")]
mod fluid_dynamics;
#[cfg(feature = "fluid_dynamics")]
pub use fluid_dynamics::*;
#[cfg(feature = "fluid_simulation")]
mod fluid_simulation;
#[cfg(feature = "fluid_simulation")]
pub use fluid_simulation::*;

#[cfg(test)]
#[cfg(feature = "fluid_dynamics")]
mod fluid_dynamics_tests;
#[cfg(test)]
#[cfg(feature = "fluid_simulation")]
mod fluid_simulation_tests;