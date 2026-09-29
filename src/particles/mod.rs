#[cfg(feature = "particles")]
mod particle;
#[cfg(feature = "particles")]
mod particle_backend;
#[cfg(feature = "particles")]
mod particle_effects;
#[cfg(feature = "particles")]
mod particle_simulation;
#[cfg(feature = "particles")]
mod particle_interactions_barnes_hut;
/// Closed-form motion for effect particles and loose pieces (the law the Ridgeline engine's
/// effect records and gore pieces are evaluated with).
#[cfg(feature = "particles")]
pub mod analytic;

#[cfg(feature = "particles")]
pub use particle::*;

#[cfg(feature = "particles")]
pub use particle_backend::*;

#[cfg(feature = "particles")]
pub use particle_effects::*;

#[cfg(feature = "particles")]
pub use particle_simulation::*;

#[cfg(feature = "particles")]
pub use particle_interactions_barnes_hut::*;

#[cfg(test)]
#[cfg(feature = "particles")]
mod particle_tests;
#[cfg(test)]
#[cfg(feature = "particles")]
mod particle_simulation_tests;
#[cfg(test)]
#[cfg(feature = "particles")]
mod particle_interactions_barnes_hut_tests;
