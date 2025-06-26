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