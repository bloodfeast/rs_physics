//! GPU-accelerated physics computations using wgpu
//!
//! This module provides GPU compute shader implementations for:
//! - Particle integration (position/velocity updates)
//! - 2D N-body gravitational simulation
//! - 3D N-body gravitational simulation (cosmological)
//!
//! # Example: Basic Particle Simulation
//!
//! ```ignore
//! use rs_physics::gpu::{GpuContext, GpuParticleSimulation, GpuParticle};
//!
//! let gpu = GpuContext::new().expect("Failed to initialize GPU");
//!
//! let particles: Vec<GpuParticle> = (0..10000)
//!     .map(|i| GpuParticle {
//!         pos: [(i % 100) as f32, (i / 100) as f32],
//!         vel: [0.0, 0.0],
//!         mass: 1.0,
//!         _padding: [0.0; 3],
//!     })
//!     .collect();
//!
//! let mut sim = GpuParticleSimulation::new(&gpu, &particles, 0.016, -9.81);
//!
//! for _ in 0..1000 {
//!     sim.step(&gpu);
//! }
//!
//! let results = sim.read_particles(&gpu);
//! ```
//!
//! # Example: 3D Cosmological N-body Simulation
//!
//! ```ignore
//! use rs_physics::gpu::{GpuContext, GpuNBody3DSimulation, NBody3DParticle};
//!
//! let gpu = GpuContext::new().expect("Failed to initialize GPU");
//!
//! // Create particles representing galaxies/dark matter halos
//! let particles: Vec<NBody3DParticle> = (0..50000)
//!     .map(|i| {
//!         let r = (i as f32 / 50000.0).powf(1.0/3.0) * 100.0;
//!         let theta = i as f32 * 2.399; // Golden angle
//!         let phi = (1.0 - 2.0 * (i as f32) / 50000.0).acos();
//!         NBody3DParticle::new(
//!             [r * phi.sin() * theta.cos(), r * phi.sin() * theta.sin(), r * phi.cos()],
//!             [0.0, 0.0, 0.0],
//!             1.0,
//!         )
//!     })
//!     .collect();
//!
//! // Periodic boundaries for cosmological simulation
//! let mut sim = GpuNBody3DSimulation::new(&gpu, &particles, 0.01, 1.0, 0.5, Some(200.0));
//!
//! for _ in 0..1000 {
//!     sim.step(&gpu);
//! }
//!
//! let results = sim.read_particles(&gpu);
//! ```

pub mod acoustics;
mod context;
mod particle_sim;
mod nbody;
mod nbody_3d;

pub use context::GpuContext;
pub use particle_sim::{GpuParticleSimulation, GpuParticle};
pub use nbody::{GpuNBodySimulation, NBodyParticle};
pub use nbody_3d::{GpuNBody3DSimulation, NBody3DParticle};
