mod interactions;
mod interactions_2d;
mod interactions_3d;

pub use interactions::*;
pub use interactions_2d::*;
pub use interactions_3d::*;

#[cfg(test)]
mod interactions_tests;
#[cfg(test)]
mod interactions_2d_tests;
#[cfg(test)]
mod interactions_3d_tests;