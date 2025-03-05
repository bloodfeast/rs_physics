#[cfg(feature = "particles")]
mod particle;
#[cfg(feature = "particles")]
mod particle_simulation;
#[cfg(feature = "particles")]
mod particle_interactions_barnes_hut;

#[cfg(feature = "particles")]
pub use particle::*;

#[cfg(feature = "particles")]
pub use particle_simulation::*;

#[cfg(feature = "particles")]
pub use particle_interactions_barnes_hut::*;
#[cfg(feature = "particles-cosmological")]
pub mod particle_interactions_barnes_hut_cosmological;

#[cfg(test)]
#[cfg(feature = "particles")]
mod particle_tests;
#[cfg(test)]
#[cfg(feature = "particles")]
mod particle_simulation_tests;
#[cfg(test)]
#[cfg(feature = "particles")]
mod particle_interactions_barnes_hut_tests;
#[cfg(test)]
#[cfg(feature = "particles-cosmological")]
mod particle_interactions_barnes_hut_cosmological_tests;