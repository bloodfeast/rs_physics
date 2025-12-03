//! Particle-Fluid Coupling Module
//!
//! This module provides utilities for coupling particle systems with fluid simulations,
//! enabling two-way interaction between discrete particles and Eulerian fluid grids.
//!
//! Key features:
//! - Velocity sampling from fluid grids using interpolation
//! - Drag force calculations for particles in fluid
//! - Two-way coupling: particles can affect fluid velocity fields
//!
//! # Examples
//! ```
//! use rs_physics::fluid_dynamics::{FluidGrid, sample_velocity_2d, calculate_fluid_drag_2d};
//!
//! let mut fluid = FluidGrid::new(50, 50, 0.1, 0.001, 0.016).unwrap();
//! fluid.add_velocity(25, 25, 1.0, 0.5).unwrap();
//!
//! // Sample velocity at a continuous position
//! let (vx, vy) = sample_velocity_2d(&fluid, 25.5, 25.5);
//!
//! // Calculate drag force on a particle
//! let drag = calculate_fluid_drag_2d(
//!     (vx, vy),           // fluid velocity
//!     (0.0, 0.0),         // particle velocity
//!     0.1,                // particle radius
//!     1000.0,             // fluid density
//!     0.001,              // fluid viscosity
//! );
//! ```

use super::fluid_simulation::FluidGrid;
use super::fluid_simulation_3d::FluidGrid3D;
use std::f64::consts::PI;

// =============================================================================
// 2D Velocity Sampling
// =============================================================================

/// Samples fluid velocity at an arbitrary 2D position using bilinear interpolation.
///
/// # Arguments
/// * `grid` - The fluid grid to sample from
/// * `x` - The x-coordinate (can be fractional)
/// * `y` - The y-coordinate (can be fractional)
///
/// # Returns
/// A tuple (vx, vy) of interpolated velocity components.
///
/// # Note
/// Coordinates outside the grid are clamped to the boundary.
///
/// # Examples
/// ```
/// use rs_physics::fluid_dynamics::{FluidGrid, sample_velocity_2d};
///
/// let mut fluid = FluidGrid::new(20, 20, 0.0, 0.0, 0.016).unwrap();
/// fluid.add_velocity(10, 10, 2.0, 1.0).unwrap();
///
/// let (vx, vy) = sample_velocity_2d(&fluid, 10.0, 10.0);
/// assert!((vx - 2.0).abs() < 0.01);
/// assert!((vy - 1.0).abs() < 0.01);
/// ```
pub fn sample_velocity_2d(grid: &FluidGrid, x: f64, y: f64) -> (f64, f64) {
    let width = grid.get_width();
    let height = grid.get_height();

    // Clamp coordinates to valid range
    let x = x.clamp(0.0, (width - 1) as f64);
    let y = y.clamp(0.0, (height - 1) as f64);

    // Get integer cell indices
    let i0 = x.floor() as usize;
    let j0 = y.floor() as usize;
    let i1 = (i0 + 1).min(width - 1);
    let j1 = (j0 + 1).min(height - 1);

    // Interpolation weights
    let s = x - i0 as f64;
    let t = y - j0 as f64;

    // Sample velocities at the four corners
    let v00 = grid.get_velocity(i0, j0).unwrap_or((0.0, 0.0));
    let v10 = grid.get_velocity(i1, j0).unwrap_or((0.0, 0.0));
    let v01 = grid.get_velocity(i0, j1).unwrap_or((0.0, 0.0));
    let v11 = grid.get_velocity(i1, j1).unwrap_or((0.0, 0.0));

    // Bilinear interpolation
    let vx = (1.0 - s) * (1.0 - t) * v00.0
           + s * (1.0 - t) * v10.0
           + (1.0 - s) * t * v01.0
           + s * t * v11.0;

    let vy = (1.0 - s) * (1.0 - t) * v00.1
           + s * (1.0 - t) * v10.1
           + (1.0 - s) * t * v01.1
           + s * t * v11.1;

    (vx, vy)
}

/// Samples fluid density at an arbitrary 2D position using bilinear interpolation.
///
/// # Arguments
/// * `grid` - The fluid grid to sample from
/// * `x` - The x-coordinate (can be fractional)
/// * `y` - The y-coordinate (can be fractional)
///
/// # Returns
/// The interpolated density value.
pub fn sample_density_2d(grid: &FluidGrid, x: f64, y: f64) -> f64 {
    let width = grid.get_width();
    let height = grid.get_height();

    let x = x.clamp(0.0, (width - 1) as f64);
    let y = y.clamp(0.0, (height - 1) as f64);

    let i0 = x.floor() as usize;
    let j0 = y.floor() as usize;
    let i1 = (i0 + 1).min(width - 1);
    let j1 = (j0 + 1).min(height - 1);

    let s = x - i0 as f64;
    let t = y - j0 as f64;

    let d00 = grid.get_density(i0, j0).unwrap_or(0.0);
    let d10 = grid.get_density(i1, j0).unwrap_or(0.0);
    let d01 = grid.get_density(i0, j1).unwrap_or(0.0);
    let d11 = grid.get_density(i1, j1).unwrap_or(0.0);

    (1.0 - s) * (1.0 - t) * d00
    + s * (1.0 - t) * d10
    + (1.0 - s) * t * d01
    + s * t * d11
}

// =============================================================================
// 3D Velocity Sampling
// =============================================================================

/// Samples fluid velocity at an arbitrary 3D position using trilinear interpolation.
///
/// # Arguments
/// * `grid` - The 3D fluid grid to sample from
/// * `x` - The x-coordinate (can be fractional)
/// * `y` - The y-coordinate (can be fractional)
/// * `z` - The z-coordinate (can be fractional)
///
/// # Returns
/// A tuple (vx, vy, vz) of interpolated velocity components.
///
/// # Examples
/// ```
/// use rs_physics::fluid_dynamics::{FluidGrid3D, sample_velocity_3d};
///
/// let mut fluid = FluidGrid3D::new(10, 10, 10, 0.0, 0.0, 0.016).unwrap();
/// fluid.add_velocity(5, 5, 5, 1.0, 2.0, 3.0).unwrap();
///
/// let (vx, vy, vz) = sample_velocity_3d(&fluid, 5.0, 5.0, 5.0);
/// assert!((vx - 1.0).abs() < 0.01);
/// ```
pub fn sample_velocity_3d(grid: &FluidGrid3D, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let width = grid.get_width();
    let height = grid.get_height();
    let depth = grid.get_depth();

    let x = x.clamp(0.0, (width - 1) as f64);
    let y = y.clamp(0.0, (height - 1) as f64);
    let z = z.clamp(0.0, (depth - 1) as f64);

    let i0 = x.floor() as usize;
    let j0 = y.floor() as usize;
    let k0 = z.floor() as usize;
    let i1 = (i0 + 1).min(width - 1);
    let j1 = (j0 + 1).min(height - 1);
    let k1 = (k0 + 1).min(depth - 1);

    let s = x - i0 as f64;
    let t = y - j0 as f64;
    let u = z - k0 as f64;

    // Sample velocities at the eight corners
    let v000 = grid.get_velocity(i0, j0, k0).unwrap_or((0.0, 0.0, 0.0));
    let v100 = grid.get_velocity(i1, j0, k0).unwrap_or((0.0, 0.0, 0.0));
    let v010 = grid.get_velocity(i0, j1, k0).unwrap_or((0.0, 0.0, 0.0));
    let v110 = grid.get_velocity(i1, j1, k0).unwrap_or((0.0, 0.0, 0.0));
    let v001 = grid.get_velocity(i0, j0, k1).unwrap_or((0.0, 0.0, 0.0));
    let v101 = grid.get_velocity(i1, j0, k1).unwrap_or((0.0, 0.0, 0.0));
    let v011 = grid.get_velocity(i0, j1, k1).unwrap_or((0.0, 0.0, 0.0));
    let v111 = grid.get_velocity(i1, j1, k1).unwrap_or((0.0, 0.0, 0.0));

    // Trilinear interpolation for each component
    let vx = trilinear_interp(v000.0, v100.0, v010.0, v110.0, v001.0, v101.0, v011.0, v111.0, s, t, u);
    let vy = trilinear_interp(v000.1, v100.1, v010.1, v110.1, v001.1, v101.1, v011.1, v111.1, s, t, u);
    let vz = trilinear_interp(v000.2, v100.2, v010.2, v110.2, v001.2, v101.2, v011.2, v111.2, s, t, u);

    (vx, vy, vz)
}

/// Samples fluid density at an arbitrary 3D position using trilinear interpolation.
pub fn sample_density_3d(grid: &FluidGrid3D, x: f64, y: f64, z: f64) -> f64 {
    let width = grid.get_width();
    let height = grid.get_height();
    let depth = grid.get_depth();

    let x = x.clamp(0.0, (width - 1) as f64);
    let y = y.clamp(0.0, (height - 1) as f64);
    let z = z.clamp(0.0, (depth - 1) as f64);

    let i0 = x.floor() as usize;
    let j0 = y.floor() as usize;
    let k0 = z.floor() as usize;
    let i1 = (i0 + 1).min(width - 1);
    let j1 = (j0 + 1).min(height - 1);
    let k1 = (k0 + 1).min(depth - 1);

    let s = x - i0 as f64;
    let t = y - j0 as f64;
    let u = z - k0 as f64;

    let d000 = grid.get_density(i0, j0, k0).unwrap_or(0.0);
    let d100 = grid.get_density(i1, j0, k0).unwrap_or(0.0);
    let d010 = grid.get_density(i0, j1, k0).unwrap_or(0.0);
    let d110 = grid.get_density(i1, j1, k0).unwrap_or(0.0);
    let d001 = grid.get_density(i0, j0, k1).unwrap_or(0.0);
    let d101 = grid.get_density(i1, j0, k1).unwrap_or(0.0);
    let d011 = grid.get_density(i0, j1, k1).unwrap_or(0.0);
    let d111 = grid.get_density(i1, j1, k1).unwrap_or(0.0);

    trilinear_interp(d000, d100, d010, d110, d001, d101, d011, d111, s, t, u)
}

/// Helper function for trilinear interpolation.
#[inline]
fn trilinear_interp(
    v000: f64, v100: f64, v010: f64, v110: f64,
    v001: f64, v101: f64, v011: f64, v111: f64,
    s: f64, t: f64, u: f64
) -> f64 {
    let c00 = v000 * (1.0 - s) + v100 * s;
    let c10 = v010 * (1.0 - s) + v110 * s;
    let c01 = v001 * (1.0 - s) + v101 * s;
    let c11 = v011 * (1.0 - s) + v111 * s;

    let c0 = c00 * (1.0 - t) + c10 * t;
    let c1 = c01 * (1.0 - t) + c11 * t;

    c0 * (1.0 - u) + c1 * u
}

// =============================================================================
// Drag Force Calculations
// =============================================================================

/// Calculates the drag force on a spherical particle in a 2D fluid flow.
///
/// Uses Stokes drag for low Reynolds numbers (Re < 1) and a simplified
/// quadratic drag for higher Reynolds numbers.
///
/// # Arguments
/// * `fluid_velocity` - The fluid velocity at the particle position (vx, vy)
/// * `particle_velocity` - The particle's current velocity (vx, vy)
/// * `particle_radius` - The radius of the particle in grid units
/// * `fluid_density` - The density of the fluid (kg/m³)
/// * `fluid_viscosity` - The dynamic viscosity of the fluid (Pa·s)
///
/// # Returns
/// The drag force as (fx, fy) in Newtons.
///
/// # Examples
/// ```
/// use rs_physics::fluid_dynamics::calculate_fluid_drag_2d;
///
/// // Particle moving slower than fluid - drag pushes it forward
/// let drag = calculate_fluid_drag_2d(
///     (1.0, 0.0),   // fluid moving right at 1 m/s
///     (0.0, 0.0),   // particle stationary
///     0.01,         // 1cm radius
///     1000.0,       // water density
///     0.001,        // water viscosity
/// );
/// assert!(drag.0 > 0.0); // Force in positive x direction
/// ```
pub fn calculate_fluid_drag_2d(
    fluid_velocity: (f64, f64),
    particle_velocity: (f64, f64),
    particle_radius: f64,
    fluid_density: f64,
    fluid_viscosity: f64,
) -> (f64, f64) {
    // Relative velocity (fluid relative to particle)
    let rel_vx = fluid_velocity.0 - particle_velocity.0;
    let rel_vy = fluid_velocity.1 - particle_velocity.1;
    let rel_speed = (rel_vx * rel_vx + rel_vy * rel_vy).sqrt();

    if rel_speed < 1e-10 {
        return (0.0, 0.0);
    }

    // Calculate Reynolds number
    let re = fluid_density * rel_speed * 2.0 * particle_radius / fluid_viscosity;

    // Calculate drag coefficient based on Reynolds number
    let drag_coeff = if re < 1.0 {
        // Stokes regime
        24.0 / re.max(1e-10)
    } else if re < 1000.0 {
        // Intermediate regime (Schiller-Naumann correlation)
        24.0 / re * (1.0 + 0.15 * re.powf(0.687))
    } else {
        // Newton regime
        0.44
    };

    // Cross-sectional area (circle in 2D, but using sphere formula for 3D compatibility)
    let area = PI * particle_radius * particle_radius;

    // Drag force magnitude: F = 0.5 * rho * v^2 * Cd * A
    let drag_magnitude = 0.5 * fluid_density * rel_speed * rel_speed * drag_coeff * area;

    // Direction is along relative velocity
    let fx = drag_magnitude * rel_vx / rel_speed;
    let fy = drag_magnitude * rel_vy / rel_speed;

    (fx, fy)
}

/// Calculates the drag force on a spherical particle in a 3D fluid flow.
///
/// # Arguments
/// * `fluid_velocity` - The fluid velocity at the particle position (vx, vy, vz)
/// * `particle_velocity` - The particle's current velocity (vx, vy, vz)
/// * `particle_radius` - The radius of the particle in grid units
/// * `fluid_density` - The density of the fluid (kg/m³)
/// * `fluid_viscosity` - The dynamic viscosity of the fluid (Pa·s)
///
/// # Returns
/// The drag force as (fx, fy, fz) in Newtons.
///
/// # Examples
/// ```
/// use rs_physics::fluid_dynamics::calculate_fluid_drag_3d;
///
/// let drag = calculate_fluid_drag_3d(
///     (1.0, 0.0, 0.0),  // fluid velocity
///     (0.0, 0.0, 0.0),  // particle at rest
///     0.01,             // radius
///     1000.0,           // water density
///     0.001,            // water viscosity
/// );
/// assert!(drag.0 > 0.0);
/// ```
pub fn calculate_fluid_drag_3d(
    fluid_velocity: (f64, f64, f64),
    particle_velocity: (f64, f64, f64),
    particle_radius: f64,
    fluid_density: f64,
    fluid_viscosity: f64,
) -> (f64, f64, f64) {
    let rel_vx = fluid_velocity.0 - particle_velocity.0;
    let rel_vy = fluid_velocity.1 - particle_velocity.1;
    let rel_vz = fluid_velocity.2 - particle_velocity.2;
    let rel_speed = (rel_vx * rel_vx + rel_vy * rel_vy + rel_vz * rel_vz).sqrt();

    if rel_speed < 1e-10 {
        return (0.0, 0.0, 0.0);
    }

    let re = fluid_density * rel_speed * 2.0 * particle_radius / fluid_viscosity;

    let drag_coeff = if re < 1.0 {
        24.0 / re.max(1e-10)
    } else if re < 1000.0 {
        24.0 / re * (1.0 + 0.15 * re.powf(0.687))
    } else {
        0.44
    };

    let area = PI * particle_radius * particle_radius;
    let drag_magnitude = 0.5 * fluid_density * rel_speed * rel_speed * drag_coeff * area;

    let fx = drag_magnitude * rel_vx / rel_speed;
    let fy = drag_magnitude * rel_vy / rel_speed;
    let fz = drag_magnitude * rel_vz / rel_speed;

    (fx, fy, fz)
}

// =============================================================================
// Two-Way Coupling: Particle affects Fluid
// =============================================================================

/// Applies a force from a particle to the 2D fluid grid.
///
/// The force is distributed to nearby grid cells using bilinear interpolation weights.
/// This enables two-way coupling where particles can push the fluid.
///
/// # Arguments
/// * `grid` - The fluid grid to modify
/// * `particle_pos` - The particle position (x, y) in grid coordinates
/// * `force` - The force to apply (fx, fy)
/// * `particle_mass` - The mass of the particle (used to scale the momentum transfer)
///
/// # Note
/// The force applied to the fluid is the reaction force (Newton's third law),
/// so if drag pushes the particle, the fluid is pushed in the opposite direction.
pub fn apply_particle_force_to_grid_2d(
    grid: &mut FluidGrid,
    particle_pos: (f64, f64),
    force: (f64, f64),
    particle_mass: f64,
) {
    let width = grid.get_width();
    let height = grid.get_height();

    let x = particle_pos.0.clamp(0.0, (width - 1) as f64);
    let y = particle_pos.1.clamp(0.0, (height - 1) as f64);

    let i0 = x.floor() as usize;
    let j0 = y.floor() as usize;
    let i1 = (i0 + 1).min(width - 1);
    let j1 = (j0 + 1).min(height - 1);

    let s = x - i0 as f64;
    let t = y - j0 as f64;

    // Scale force by dt and inverse cell mass to get velocity change
    // Using a simplified model where force is distributed as velocity
    let dt = grid.get_dt();
    let scale = dt / particle_mass.max(1e-10);

    // Reaction force (opposite direction)
    let dvx = -force.0 * scale;
    let dvy = -force.1 * scale;

    // Distribute velocity change using bilinear weights
    let w00 = (1.0 - s) * (1.0 - t);
    let w10 = s * (1.0 - t);
    let w01 = (1.0 - s) * t;
    let w11 = s * t;

    let _ = grid.add_velocity(i0, j0, dvx * w00, dvy * w00);
    let _ = grid.add_velocity(i1, j0, dvx * w10, dvy * w10);
    let _ = grid.add_velocity(i0, j1, dvx * w01, dvy * w01);
    let _ = grid.add_velocity(i1, j1, dvx * w11, dvy * w11);
}

/// Applies a force from a particle to the 3D fluid grid.
///
/// The force is distributed to nearby grid cells using trilinear interpolation weights.
pub fn apply_particle_force_to_grid_3d(
    grid: &mut FluidGrid3D,
    particle_pos: (f64, f64, f64),
    force: (f64, f64, f64),
    particle_mass: f64,
) {
    let width = grid.get_width();
    let height = grid.get_height();
    let depth = grid.get_depth();

    let x = particle_pos.0.clamp(0.0, (width - 1) as f64);
    let y = particle_pos.1.clamp(0.0, (height - 1) as f64);
    let z = particle_pos.2.clamp(0.0, (depth - 1) as f64);

    let i0 = x.floor() as usize;
    let j0 = y.floor() as usize;
    let k0 = z.floor() as usize;
    let i1 = (i0 + 1).min(width - 1);
    let j1 = (j0 + 1).min(height - 1);
    let k1 = (k0 + 1).min(depth - 1);

    let s = x - i0 as f64;
    let t = y - j0 as f64;
    let u = z - k0 as f64;

    let dt = grid.get_dt();
    let scale = dt / particle_mass.max(1e-10);

    let dvx = -force.0 * scale;
    let dvy = -force.1 * scale;
    let dvz = -force.2 * scale;

    // Trilinear weights
    let w000 = (1.0 - s) * (1.0 - t) * (1.0 - u);
    let w100 = s * (1.0 - t) * (1.0 - u);
    let w010 = (1.0 - s) * t * (1.0 - u);
    let w110 = s * t * (1.0 - u);
    let w001 = (1.0 - s) * (1.0 - t) * u;
    let w101 = s * (1.0 - t) * u;
    let w011 = (1.0 - s) * t * u;
    let w111 = s * t * u;

    let _ = grid.add_velocity(i0, j0, k0, dvx * w000, dvy * w000, dvz * w000);
    let _ = grid.add_velocity(i1, j0, k0, dvx * w100, dvy * w100, dvz * w100);
    let _ = grid.add_velocity(i0, j1, k0, dvx * w010, dvy * w010, dvz * w010);
    let _ = grid.add_velocity(i1, j1, k0, dvx * w110, dvy * w110, dvz * w110);
    let _ = grid.add_velocity(i0, j0, k1, dvx * w001, dvy * w001, dvz * w001);
    let _ = grid.add_velocity(i1, j0, k1, dvx * w101, dvy * w101, dvz * w101);
    let _ = grid.add_velocity(i0, j1, k1, dvx * w011, dvy * w011, dvz * w011);
    let _ = grid.add_velocity(i1, j1, k1, dvx * w111, dvy * w111, dvz * w111);
}

// =============================================================================
// Particle State Update Helper
// =============================================================================

/// Represents a particle's state for fluid interaction.
#[derive(Debug, Clone, Copy)]
pub struct FluidParticle2D {
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub radius: f64,
    pub mass: f64,
}

impl FluidParticle2D {
    /// Creates a new 2D particle.
    pub fn new(x: f64, y: f64, radius: f64, mass: f64) -> Self {
        Self { x, y, vx: 0.0, vy: 0.0, radius, mass }
    }

    /// Updates the particle state based on fluid forces.
    ///
    /// # Arguments
    /// * `grid` - The fluid grid to interact with
    /// * `fluid_density` - Density of the fluid for drag calculation
    /// * `fluid_viscosity` - Viscosity of the fluid for drag calculation
    /// * `dt` - Time step
    /// * `two_way_coupling` - Whether to apply reaction force to the fluid
    pub fn update(&mut self, grid: &mut FluidGrid, fluid_density: f64, fluid_viscosity: f64, dt: f64, two_way_coupling: bool) {
        // Sample fluid velocity at particle position
        let fluid_vel = sample_velocity_2d(grid, self.x, self.y);

        // Calculate drag force
        let drag = calculate_fluid_drag_2d(
            fluid_vel,
            (self.vx, self.vy),
            self.radius,
            fluid_density,
            fluid_viscosity,
        );

        // Apply two-way coupling if enabled
        if two_way_coupling {
            apply_particle_force_to_grid_2d(grid, (self.x, self.y), drag, self.mass);
        }

        // Update particle velocity (F = ma, so a = F/m)
        let ax = drag.0 / self.mass;
        let ay = drag.1 / self.mass;
        self.vx += ax * dt;
        self.vy += ay * dt;

        // Update position
        self.x += self.vx * dt;
        self.y += self.vy * dt;
    }
}

/// Represents a particle's state for 3D fluid interaction.
#[derive(Debug, Clone, Copy)]
pub struct FluidParticle3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    pub radius: f64,
    pub mass: f64,
}

impl FluidParticle3D {
    /// Creates a new 3D particle.
    pub fn new(x: f64, y: f64, z: f64, radius: f64, mass: f64) -> Self {
        Self { x, y, z, vx: 0.0, vy: 0.0, vz: 0.0, radius, mass }
    }

    /// Updates the particle state based on fluid forces.
    pub fn update(&mut self, grid: &mut FluidGrid3D, fluid_density: f64, fluid_viscosity: f64, dt: f64, two_way_coupling: bool) {
        let fluid_vel = sample_velocity_3d(grid, self.x, self.y, self.z);

        let drag = calculate_fluid_drag_3d(
            fluid_vel,
            (self.vx, self.vy, self.vz),
            self.radius,
            fluid_density,
            fluid_viscosity,
        );

        if two_way_coupling {
            apply_particle_force_to_grid_3d(grid, (self.x, self.y, self.z), drag, self.mass);
        }

        let ax = drag.0 / self.mass;
        let ay = drag.1 / self.mass;
        let az = drag.2 / self.mass;
        self.vx += ax * dt;
        self.vy += ay * dt;
        self.vz += az * dt;

        self.x += self.vx * dt;
        self.y += self.vy * dt;
        self.z += self.vz * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_velocity_2d_exact() {
        let mut grid = FluidGrid::new(10, 10, 0.0, 0.0, 0.016).unwrap();
        grid.add_velocity(5, 5, 2.0, 3.0).unwrap();

        let (vx, vy) = sample_velocity_2d(&grid, 5.0, 5.0);
        assert!((vx - 2.0).abs() < 1e-10);
        assert!((vy - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_velocity_2d_interpolated() {
        let mut grid = FluidGrid::new(10, 10, 0.0, 0.0, 0.016).unwrap();
        grid.add_velocity(4, 4, 1.0, 0.0).unwrap();
        grid.add_velocity(5, 4, 3.0, 0.0).unwrap();

        // Sample midway between (4,4) and (5,4)
        let (vx, _) = sample_velocity_2d(&grid, 4.5, 4.0);
        assert!((vx - 2.0).abs() < 1e-10); // Average of 1.0 and 3.0
    }

    #[test]
    fn test_sample_velocity_3d_exact() {
        let mut grid = FluidGrid3D::new(10, 10, 10, 0.0, 0.0, 0.016).unwrap();
        grid.add_velocity(5, 5, 5, 1.0, 2.0, 3.0).unwrap();

        let (vx, vy, vz) = sample_velocity_3d(&grid, 5.0, 5.0, 5.0);
        assert!((vx - 1.0).abs() < 1e-10);
        assert!((vy - 2.0).abs() < 1e-10);
        assert!((vz - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_drag_force_direction() {
        // Fluid moving right, particle stationary
        let drag = calculate_fluid_drag_2d(
            (1.0, 0.0),
            (0.0, 0.0),
            0.01,
            1000.0,
            0.001,
        );
        // Drag should push particle in direction of fluid flow
        assert!(drag.0 > 0.0);
        assert!(drag.1.abs() < 1e-10);
    }

    #[test]
    fn test_drag_force_zero_relative_velocity() {
        let drag = calculate_fluid_drag_2d(
            (1.0, 1.0),
            (1.0, 1.0),
            0.01,
            1000.0,
            0.001,
        );
        assert!(drag.0.abs() < 1e-10);
        assert!(drag.1.abs() < 1e-10);
    }

    #[test]
    fn test_drag_force_3d() {
        let drag = calculate_fluid_drag_3d(
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            0.01,
            1000.0,
            0.001,
        );
        assert!(drag.0 > 0.0);
        assert!(drag.1.abs() < 1e-10);
        assert!(drag.2.abs() < 1e-10);
    }

    #[test]
    fn test_particle_2d_update() {
        let mut grid = FluidGrid::new(20, 20, 0.0, 0.0, 0.016).unwrap();
        grid.add_velocity(10, 10, 1.0, 0.0).unwrap();

        let mut particle = FluidParticle2D::new(10.0, 10.0, 0.1, 1.0);
        particle.update(&mut grid, 1000.0, 0.001, 0.016, false);

        // Particle should have gained some velocity in x direction
        assert!(particle.vx > 0.0);
    }

    #[test]
    fn test_particle_3d_update() {
        let mut grid = FluidGrid3D::new(10, 10, 10, 0.0, 0.0, 0.016).unwrap();
        grid.add_velocity(5, 5, 5, 0.0, 1.0, 0.0).unwrap();

        let mut particle = FluidParticle3D::new(5.0, 5.0, 5.0, 0.1, 1.0);
        particle.update(&mut grid, 1000.0, 0.001, 0.016, false);

        // Particle should have gained some velocity in y direction
        assert!(particle.vy > 0.0);
    }

    #[test]
    fn test_two_way_coupling() {
        let mut grid = FluidGrid::new(20, 20, 0.0, 0.0, 0.016).unwrap();

        // Apply a force to the grid from a particle
        apply_particle_force_to_grid_2d(&mut grid, (10.0, 10.0), (1.0, 0.0), 1.0);

        // The grid should now have some velocity (reaction force is opposite)
        let (vx, _) = grid.get_velocity(10, 10).unwrap();
        assert!(vx < 0.0); // Reaction force pushes fluid in opposite direction
    }

    #[test]
    fn test_sample_density_2d() {
        let mut grid = FluidGrid::new(10, 10, 0.0, 0.0, 0.016).unwrap();
        grid.add_density(5, 5, 1.0).unwrap();

        let density = sample_density_2d(&grid, 5.0, 5.0);
        assert!((density - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_density_3d() {
        let mut grid = FluidGrid3D::new(10, 10, 10, 0.0, 0.0, 0.016).unwrap();
        grid.add_density(5, 5, 5, 1.0).unwrap();

        let density = sample_density_3d(&grid, 5.0, 5.0, 5.0);
        assert!((density - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_trilinear_interpolation() {
        // Test the trilinear interpolation function directly
        // All corners have same value
        let result = trilinear_interp(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5);
        assert!((result - 1.0).abs() < 1e-10);

        // Linear interpolation along x
        let result = trilinear_interp(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0);
        assert!((result - 0.5).abs() < 1e-10);
    }
}
