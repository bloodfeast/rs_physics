//! 3D Grid-based fluid simulation using the Eulerian method.
//!
//! This module extends the 2D fluid simulation to three dimensions,
//! implementing Jos Stam's stable fluid solver for volumetric simulations.

use crate::utils::PhysicsError;
use super::validation::{validate_dimensions_3d, validate_non_negative, validate_positive, validate_position_3d};
use super::solver::{SolverConfig, BoundaryType};

/// A 3D grid-based fluid simulation using the Eulerian method.
///
/// This struct implements a stable fluid solver based on Jos Stam's method,
/// extended to three dimensions. The simulation handles density diffusion,
/// velocity diffusion, and advection in a 3D grid.
///
/// # Examples
/// ```
/// use rs_physics::fluid_dynamics::FluidGrid3D;
///
/// // Create a 50x50x50 grid with default solver
/// let mut fluid = FluidGrid3D::new(50, 50, 50, 0.1, 0.001, 0.016).unwrap();
///
/// // Add some density and velocity
/// fluid.add_density(25, 25, 25, 1.0).unwrap();
/// fluid.add_velocity(25, 25, 25, 0.0, 1.0, 0.0).unwrap();
///
/// // Run simulation step
/// fluid.step();
/// ```
pub struct FluidGrid3D {
    width: usize,
    height: usize,
    depth: usize,
    density: Vec<f64>,
    velocity_x: Vec<f64>,
    velocity_y: Vec<f64>,
    velocity_z: Vec<f64>,
    diffusion: f64,
    viscosity: f64,
    dt: f64,
    solver_config: SolverConfig,
}

impl FluidGrid3D {
    /// Creates a new 3D fluid simulation grid with the specified dimensions and properties.
    ///
    /// # Arguments
    /// * `width` - The width of the simulation grid (x-axis)
    /// * `height` - The height of the simulation grid (y-axis)
    /// * `depth` - The depth of the simulation grid (z-axis)
    /// * `diffusion` - The rate of diffusion (must be non-negative)
    /// * `viscosity` - The fluid viscosity (must be non-negative)
    /// * `dt` - The time step for the simulation (must be positive)
    ///
    /// # Returns
    /// * `Ok(FluidGrid3D)` - A new fluid simulation grid if all parameters are valid
    /// * `Err(PhysicsError)` - If any parameters are invalid
    pub fn new(
        width: usize,
        height: usize,
        depth: usize,
        diffusion: f64,
        viscosity: f64,
        dt: f64,
    ) -> Result<Self, PhysicsError> {
        Self::with_solver(width, height, depth, diffusion, viscosity, dt, SolverConfig::default())
    }

    /// Creates a new 3D fluid simulation grid with custom solver configuration.
    pub fn with_solver(
        width: usize,
        height: usize,
        depth: usize,
        diffusion: f64,
        viscosity: f64,
        dt: f64,
        solver_config: SolverConfig,
    ) -> Result<Self, PhysicsError> {
        validate_dimensions_3d(width, height, depth)?;
        validate_non_negative(diffusion, "diffusion").map_err(|_| PhysicsError::InvalidCoefficient)?;
        validate_non_negative(viscosity, "viscosity").map_err(|_| PhysicsError::InvalidCoefficient)?;
        validate_positive(dt, "dt").map_err(|_| PhysicsError::InvalidTime)?;

        let size = width * height * depth;
        Ok(Self {
            width,
            height,
            depth,
            density: vec![0.0; size],
            velocity_x: vec![0.0; size],
            velocity_y: vec![0.0; size],
            velocity_z: vec![0.0; size],
            diffusion,
            viscosity,
            dt,
            solver_config,
        })
    }

    /// Adds density to the fluid at a specific grid position.
    pub fn add_density(&mut self, x: usize, y: usize, z: usize, amount: f64) -> Result<(), PhysicsError> {
        validate_position_3d(x, y, z, self.width, self.height, self.depth)?;
        let idx = self.get_index(x, y, z);
        self.density[idx] += amount;
        Ok(())
    }

    /// Adds velocity to the fluid at a specific grid position.
    pub fn add_velocity(&mut self, x: usize, y: usize, z: usize, vx: f64, vy: f64, vz: f64) -> Result<(), PhysicsError> {
        validate_position_3d(x, y, z, self.width, self.height, self.depth)?;
        let idx = self.get_index(x, y, z);
        self.velocity_x[idx] += vx;
        self.velocity_y[idx] += vy;
        self.velocity_z[idx] += vz;
        Ok(())
    }

    /// Advances the fluid simulation by one time step.
    pub fn step(&mut self) {
        let size = self.width * self.height * self.depth;
        let mut velocity_x0 = vec![0.0; size];
        let mut velocity_y0 = vec![0.0; size];
        let mut velocity_z0 = vec![0.0; size];
        let mut density0 = vec![0.0; size];

        // Clone the current state
        velocity_x0.copy_from_slice(&self.velocity_x);
        velocity_y0.copy_from_slice(&self.velocity_y);
        velocity_z0.copy_from_slice(&self.velocity_z);
        density0.copy_from_slice(&self.density);

        // Diffuse velocity
        {
            let a = self.dt * self.viscosity * (self.width * self.height * self.depth) as f64;
            self.lin_solve(BoundaryType::VelocityX, &mut velocity_x0, &self.velocity_x, a, 1.0 + 6.0 * a);
            self.lin_solve(BoundaryType::VelocityY, &mut velocity_y0, &self.velocity_y, a, 1.0 + 6.0 * a);
            self.lin_solve(BoundaryType::VelocityZ, &mut velocity_z0, &self.velocity_z, a, 1.0 + 6.0 * a);
        }

        // Project velocity
        self.project(&mut velocity_x0, &mut velocity_y0, &mut velocity_z0);

        // Advect velocity
        {
            let mut next_velocity_x = vec![0.0; size];
            let mut next_velocity_y = vec![0.0; size];
            let mut next_velocity_z = vec![0.0; size];

            self.advect(BoundaryType::VelocityX, &mut next_velocity_x, &velocity_x0, &velocity_x0, &velocity_y0, &velocity_z0);
            self.advect(BoundaryType::VelocityY, &mut next_velocity_y, &velocity_y0, &velocity_x0, &velocity_y0, &velocity_z0);
            self.advect(BoundaryType::VelocityZ, &mut next_velocity_z, &velocity_z0, &velocity_x0, &velocity_y0, &velocity_z0);

            self.velocity_x = next_velocity_x;
            self.velocity_y = next_velocity_y;
            self.velocity_z = next_velocity_z;
        }

        // Project again
        {
            let mut next_velocity_x = self.velocity_x.clone();
            let mut next_velocity_y = self.velocity_y.clone();
            let mut next_velocity_z = self.velocity_z.clone();
            self.project(&mut next_velocity_x, &mut next_velocity_y, &mut next_velocity_z);
            self.velocity_x = next_velocity_x;
            self.velocity_y = next_velocity_y;
            self.velocity_z = next_velocity_z;
        }

        // Diffuse density
        {
            let a = self.dt * self.diffusion * (self.width * self.height * self.depth) as f64;
            self.lin_solve(BoundaryType::Density, &mut density0, &self.density, a, 1.0 + 6.0 * a);
        }

        // Advect density
        {
            let mut next_density = vec![0.0; size];
            self.advect(BoundaryType::Density, &mut next_density, &density0, &self.velocity_x, &self.velocity_y, &self.velocity_z);
            self.density = next_density;
        }
    }

    /// Gets the density value at a specific grid position.
    pub fn get_density(&self, x: usize, y: usize, z: usize) -> Result<f64, PhysicsError> {
        validate_position_3d(x, y, z, self.width, self.height, self.depth)?;
        Ok(self.density[self.get_index(x, y, z)])
    }

    /// Gets the velocity components at a specific grid position.
    pub fn get_velocity(&self, x: usize, y: usize, z: usize) -> Result<(f64, f64, f64), PhysicsError> {
        validate_position_3d(x, y, z, self.width, self.height, self.depth)?;
        let idx = self.get_index(x, y, z);
        Ok((self.velocity_x[idx], self.velocity_y[idx], self.velocity_z[idx]))
    }

    /// Converts 3D coordinates to a 1D array index.
    #[inline]
    fn get_index(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.width * self.height + y * self.width + x
    }

    /// Projects the velocity field to make it mass-conserving (divergence-free).
    fn project(&self, velocity_x: &mut Vec<f64>, velocity_y: &mut Vec<f64>, velocity_z: &mut Vec<f64>) {
        let size = self.width * self.height * self.depth;
        let mut p = vec![0.0; size];
        let mut div = vec![0.0; size];

        let h = 1.0 / self.width as f64;

        // Calculate divergence
        for i in 1..self.width-1 {
            for j in 1..self.height-1 {
                for k in 1..self.depth-1 {
                    let idx = self.get_index(i, j, k);
                    div[idx] = -0.5 * h * (
                        velocity_x[self.get_index(i+1, j, k)] - velocity_x[self.get_index(i-1, j, k)] +
                        velocity_y[self.get_index(i, j+1, k)] - velocity_y[self.get_index(i, j-1, k)] +
                        velocity_z[self.get_index(i, j, k+1)] - velocity_z[self.get_index(i, j, k-1)]
                    );
                    p[idx] = 0.0;
                }
            }
        }

        self.set_boundaries(BoundaryType::Density, &mut div);
        self.set_boundaries(BoundaryType::Density, &mut p);
        self.lin_solve(BoundaryType::Density, &mut p, &div, 1.0, 6.0);

        // Subtract pressure gradient
        for i in 1..self.width-1 {
            for j in 1..self.height-1 {
                for k in 1..self.depth-1 {
                    let idx = self.get_index(i, j, k);
                    velocity_x[idx] -= 0.5 * (p[self.get_index(i+1, j, k)] - p[self.get_index(i-1, j, k)]) / h;
                    velocity_y[idx] -= 0.5 * (p[self.get_index(i, j+1, k)] - p[self.get_index(i, j-1, k)]) / h;
                    velocity_z[idx] -= 0.5 * (p[self.get_index(i, j, k+1)] - p[self.get_index(i, j, k-1)]) / h;
                }
            }
        }

        self.set_boundaries(BoundaryType::VelocityX, velocity_x);
        self.set_boundaries(BoundaryType::VelocityY, velocity_y);
        self.set_boundaries(BoundaryType::VelocityZ, velocity_z);
    }

    /// Sets the boundary conditions for the 3D fluid simulation.
    fn set_boundaries(&self, boundary_type: BoundaryType, x: &mut Vec<f64>) {
        // Handle faces (6 faces)
        for i in 1..self.width-1 {
            for j in 1..self.height-1 {
                // Front face (k=0)
                x[self.get_index(i, j, 0)] = if boundary_type == BoundaryType::VelocityZ {
                    -x[self.get_index(i, j, 1)]
                } else {
                    x[self.get_index(i, j, 1)]
                };
                // Back face (k=depth-1)
                x[self.get_index(i, j, self.depth-1)] = if boundary_type == BoundaryType::VelocityZ {
                    -x[self.get_index(i, j, self.depth-2)]
                } else {
                    x[self.get_index(i, j, self.depth-2)]
                };
            }
        }

        for i in 1..self.width-1 {
            for k in 1..self.depth-1 {
                // Bottom face (j=0)
                x[self.get_index(i, 0, k)] = if boundary_type == BoundaryType::VelocityY {
                    -x[self.get_index(i, 1, k)]
                } else {
                    x[self.get_index(i, 1, k)]
                };
                // Top face (j=height-1)
                x[self.get_index(i, self.height-1, k)] = if boundary_type == BoundaryType::VelocityY {
                    -x[self.get_index(i, self.height-2, k)]
                } else {
                    x[self.get_index(i, self.height-2, k)]
                };
            }
        }

        for j in 1..self.height-1 {
            for k in 1..self.depth-1 {
                // Left face (i=0)
                x[self.get_index(0, j, k)] = if boundary_type == BoundaryType::VelocityX {
                    -x[self.get_index(1, j, k)]
                } else {
                    x[self.get_index(1, j, k)]
                };
                // Right face (i=width-1)
                x[self.get_index(self.width-1, j, k)] = if boundary_type == BoundaryType::VelocityX {
                    -x[self.get_index(self.width-2, j, k)]
                } else {
                    x[self.get_index(self.width-2, j, k)]
                };
            }
        }

        // Handle edges (12 edges) - average of adjacent face values
        for i in 1..self.width-1 {
            x[self.get_index(i, 0, 0)] = 0.5 * (x[self.get_index(i, 1, 0)] + x[self.get_index(i, 0, 1)]);
            x[self.get_index(i, self.height-1, 0)] = 0.5 * (x[self.get_index(i, self.height-2, 0)] + x[self.get_index(i, self.height-1, 1)]);
            x[self.get_index(i, 0, self.depth-1)] = 0.5 * (x[self.get_index(i, 1, self.depth-1)] + x[self.get_index(i, 0, self.depth-2)]);
            x[self.get_index(i, self.height-1, self.depth-1)] = 0.5 * (x[self.get_index(i, self.height-2, self.depth-1)] + x[self.get_index(i, self.height-1, self.depth-2)]);
        }

        for j in 1..self.height-1 {
            x[self.get_index(0, j, 0)] = 0.5 * (x[self.get_index(1, j, 0)] + x[self.get_index(0, j, 1)]);
            x[self.get_index(self.width-1, j, 0)] = 0.5 * (x[self.get_index(self.width-2, j, 0)] + x[self.get_index(self.width-1, j, 1)]);
            x[self.get_index(0, j, self.depth-1)] = 0.5 * (x[self.get_index(1, j, self.depth-1)] + x[self.get_index(0, j, self.depth-2)]);
            x[self.get_index(self.width-1, j, self.depth-1)] = 0.5 * (x[self.get_index(self.width-2, j, self.depth-1)] + x[self.get_index(self.width-1, j, self.depth-2)]);
        }

        for k in 1..self.depth-1 {
            x[self.get_index(0, 0, k)] = 0.5 * (x[self.get_index(1, 0, k)] + x[self.get_index(0, 1, k)]);
            x[self.get_index(self.width-1, 0, k)] = 0.5 * (x[self.get_index(self.width-2, 0, k)] + x[self.get_index(self.width-1, 1, k)]);
            x[self.get_index(0, self.height-1, k)] = 0.5 * (x[self.get_index(1, self.height-1, k)] + x[self.get_index(0, self.height-2, k)]);
            x[self.get_index(self.width-1, self.height-1, k)] = 0.5 * (x[self.get_index(self.width-2, self.height-1, k)] + x[self.get_index(self.width-1, self.height-2, k)]);
        }

        // Handle corners (8 corners) - average of 3 adjacent edge values
        x[self.get_index(0, 0, 0)] = (
            x[self.get_index(1, 0, 0)] + x[self.get_index(0, 1, 0)] + x[self.get_index(0, 0, 1)]
        ) / 3.0;
        x[self.get_index(self.width-1, 0, 0)] = (
            x[self.get_index(self.width-2, 0, 0)] + x[self.get_index(self.width-1, 1, 0)] + x[self.get_index(self.width-1, 0, 1)]
        ) / 3.0;
        x[self.get_index(0, self.height-1, 0)] = (
            x[self.get_index(1, self.height-1, 0)] + x[self.get_index(0, self.height-2, 0)] + x[self.get_index(0, self.height-1, 1)]
        ) / 3.0;
        x[self.get_index(self.width-1, self.height-1, 0)] = (
            x[self.get_index(self.width-2, self.height-1, 0)] + x[self.get_index(self.width-1, self.height-2, 0)] + x[self.get_index(self.width-1, self.height-1, 1)]
        ) / 3.0;
        x[self.get_index(0, 0, self.depth-1)] = (
            x[self.get_index(1, 0, self.depth-1)] + x[self.get_index(0, 1, self.depth-1)] + x[self.get_index(0, 0, self.depth-2)]
        ) / 3.0;
        x[self.get_index(self.width-1, 0, self.depth-1)] = (
            x[self.get_index(self.width-2, 0, self.depth-1)] + x[self.get_index(self.width-1, 1, self.depth-1)] + x[self.get_index(self.width-1, 0, self.depth-2)]
        ) / 3.0;
        x[self.get_index(0, self.height-1, self.depth-1)] = (
            x[self.get_index(1, self.height-1, self.depth-1)] + x[self.get_index(0, self.height-2, self.depth-1)] + x[self.get_index(0, self.height-1, self.depth-2)]
        ) / 3.0;
        x[self.get_index(self.width-1, self.height-1, self.depth-1)] = (
            x[self.get_index(self.width-2, self.height-1, self.depth-1)] + x[self.get_index(self.width-1, self.height-2, self.depth-1)] + x[self.get_index(self.width-1, self.height-1, self.depth-2)]
        ) / 3.0;
    }

    /// Solves a linear system using Gauss-Seidel relaxation in 3D.
    fn lin_solve(&self, boundary_type: BoundaryType, x: &mut Vec<f64>, x0: &Vec<f64>, a: f64, c: f64) {
        for _ in 0..self.solver_config.iterations {
            for i in 1..self.width-1 {
                for j in 1..self.height-1 {
                    for k in 1..self.depth-1 {
                        let idx = self.get_index(i, j, k);
                        x[idx] = (x0[idx] + a * (
                            x[self.get_index(i+1, j, k)] + x[self.get_index(i-1, j, k)] +
                            x[self.get_index(i, j+1, k)] + x[self.get_index(i, j-1, k)] +
                            x[self.get_index(i, j, k+1)] + x[self.get_index(i, j, k-1)]
                        )) / c;
                    }
                }
            }
            self.set_boundaries(boundary_type, x);
        }
    }

    /// Performs semi-Lagrangian advection in 3D.
    fn advect(&self, boundary_type: BoundaryType, d: &mut Vec<f64>, d0: &Vec<f64>,
              velocity_x: &Vec<f64>, velocity_y: &Vec<f64>, velocity_z: &Vec<f64>) {
        let dt0 = self.dt * self.width as f64;

        for i in 1..self.width-1 {
            for j in 1..self.height-1 {
                for k in 1..self.depth-1 {
                    let idx = self.get_index(i, j, k);

                    let mut x = i as f64 - dt0 * velocity_x[idx];
                    let mut y = j as f64 - dt0 * velocity_y[idx];
                    let mut z = k as f64 - dt0 * velocity_z[idx];

                    x = x.clamp(0.5, self.width as f64 - 1.5);
                    y = y.clamp(0.5, self.height as f64 - 1.5);
                    z = z.clamp(0.5, self.depth as f64 - 1.5);

                    let i0 = x.floor() as usize;
                    let i1 = i0 + 1;
                    let j0 = y.floor() as usize;
                    let j1 = j0 + 1;
                    let k0 = z.floor() as usize;
                    let k1 = k0 + 1;

                    let s1 = x - i0 as f64;
                    let s0 = 1.0 - s1;
                    let t1 = y - j0 as f64;
                    let t0 = 1.0 - t1;
                    let u1 = z - k0 as f64;
                    let u0 = 1.0 - u1;

                    // Trilinear interpolation
                    d[idx] = s0 * (
                        t0 * (u0 * d0[self.get_index(i0, j0, k0)] + u1 * d0[self.get_index(i0, j0, k1)]) +
                        t1 * (u0 * d0[self.get_index(i0, j1, k0)] + u1 * d0[self.get_index(i0, j1, k1)])
                    ) + s1 * (
                        t0 * (u0 * d0[self.get_index(i1, j0, k0)] + u1 * d0[self.get_index(i1, j0, k1)]) +
                        t1 * (u0 * d0[self.get_index(i1, j1, k0)] + u1 * d0[self.get_index(i1, j1, k1)])
                    );
                }
            }
        }

        self.set_boundaries(boundary_type, d);
    }

    // --- Accessor methods ---

    /// Gets the width of the simulation grid.
    pub fn get_width(&self) -> usize { self.width }

    /// Gets the height of the simulation grid.
    pub fn get_height(&self) -> usize { self.height }

    /// Gets the depth of the simulation grid.
    pub fn get_depth(&self) -> usize { self.depth }

    /// Gets the diffusion rate of the fluid.
    pub fn get_diffusion(&self) -> f64 { self.diffusion }

    /// Gets the viscosity of the fluid.
    pub fn get_viscosity(&self) -> f64 { self.viscosity }

    /// Gets the time step of the simulation.
    pub fn get_dt(&self) -> f64 { self.dt }

    /// Gets the solver configuration.
    pub fn get_solver_config(&self) -> SolverConfig { self.solver_config }

    /// Gets the number of solver iterations.
    pub fn get_solver_iterations(&self) -> usize { self.solver_config.iterations }

    /// Sets the number of solver iterations.
    pub fn set_solver_iterations(&mut self, iterations: usize) {
        self.solver_config.iterations = iterations.max(1);
    }

    /// Sets the solver configuration.
    pub fn set_solver_config(&mut self, config: SolverConfig) {
        self.solver_config = config;
    }

    /// Sets the diffusion rate of the fluid.
    pub fn set_diffusion(&mut self, diffusion: f64) -> Result<(), PhysicsError> {
        validate_non_negative(diffusion, "diffusion").map_err(|_| PhysicsError::InvalidCoefficient)?;
        self.diffusion = diffusion;
        Ok(())
    }

    /// Sets the viscosity of the fluid.
    pub fn set_viscosity(&mut self, viscosity: f64) -> Result<(), PhysicsError> {
        validate_non_negative(viscosity, "viscosity").map_err(|_| PhysicsError::InvalidCoefficient)?;
        self.viscosity = viscosity;
        Ok(())
    }

    /// Sets the time step of the simulation.
    pub fn set_dt(&mut self, dt: f64) -> Result<(), PhysicsError> {
        validate_positive(dt, "dt").map_err(|_| PhysicsError::InvalidTime)?;
        self.dt = dt;
        Ok(())
    }

    /// Resets the simulation to its initial state.
    pub fn reset(&mut self) {
        let size = self.width * self.height * self.depth;
        self.density = vec![0.0; size];
        self.velocity_x = vec![0.0; size];
        self.velocity_y = vec![0.0; size];
        self.velocity_z = vec![0.0; size];
    }

    /// Calculates the total mass (sum of density) in the simulation.
    pub fn get_total_mass(&self) -> f64 {
        self.density.iter().sum()
    }

    /// Calculates the average velocity magnitude in the simulation.
    pub fn get_average_velocity(&self) -> f64 {
        let size = self.width * self.height * self.depth;
        let total_velocity: f64 = (0..size)
            .map(|i| (self.velocity_x[i].powi(2) + self.velocity_y[i].powi(2) + self.velocity_z[i].powi(2)).sqrt())
            .sum();
        total_velocity / size as f64
    }

    /// Calculates the kinetic energy of the fluid.
    pub fn get_kinetic_energy(&self) -> f64 {
        let size = self.width * self.height * self.depth;
        (0..size)
            .map(|i| {
                0.5 * self.density[i] *
                    (self.velocity_x[i].powi(2) + self.velocity_y[i].powi(2) + self.velocity_z[i].powi(2))
            })
            .sum()
    }

    /// Checks if the simulation state is valid.
    pub fn validate_state(&self) -> Result<(), PhysicsError> {
        // Check density values
        if self.density.iter().any(|&d| d < 0.0 || !d.is_finite()) {
            return Err(PhysicsError::CalculationError(
                "Invalid density values detected".to_string()
            ));
        }

        // Check velocity values
        if self.velocity_x.iter()
            .chain(self.velocity_y.iter())
            .chain(self.velocity_z.iter())
            .any(|&v| !v.is_finite()) {
            return Err(PhysicsError::CalculationError(
                "Invalid velocity values detected".to_string()
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_creation() {
        let grid = FluidGrid3D::new(10, 10, 10, 0.1, 0.001, 0.016).unwrap();
        assert_eq!(grid.get_width(), 10);
        assert_eq!(grid.get_height(), 10);
        assert_eq!(grid.get_depth(), 10);
    }

    #[test]
    fn test_invalid_dimensions() {
        assert!(FluidGrid3D::new(0, 10, 10, 0.1, 0.001, 0.016).is_err());
        assert!(FluidGrid3D::new(10, 0, 10, 0.1, 0.001, 0.016).is_err());
        assert!(FluidGrid3D::new(10, 10, 0, 0.1, 0.001, 0.016).is_err());
    }

    #[test]
    fn test_add_density() {
        let mut grid = FluidGrid3D::new(10, 10, 10, 0.1, 0.001, 0.016).unwrap();
        grid.add_density(5, 5, 5, 1.0).unwrap();
        assert_eq!(grid.get_density(5, 5, 5).unwrap(), 1.0);
        assert_eq!(grid.get_total_mass(), 1.0);
    }

    #[test]
    fn test_add_velocity() {
        let mut grid = FluidGrid3D::new(10, 10, 10, 0.1, 0.001, 0.016).unwrap();
        grid.add_velocity(5, 5, 5, 1.0, 2.0, 3.0).unwrap();
        let (vx, vy, vz) = grid.get_velocity(5, 5, 5).unwrap();
        assert_eq!(vx, 1.0);
        assert_eq!(vy, 2.0);
        assert_eq!(vz, 3.0);
    }

    #[test]
    fn test_out_of_bounds() {
        let mut grid = FluidGrid3D::new(10, 10, 10, 0.1, 0.001, 0.016).unwrap();
        assert!(grid.add_density(10, 5, 5, 1.0).is_err());
        assert!(grid.add_density(5, 10, 5, 1.0).is_err());
        assert!(grid.add_density(5, 5, 10, 1.0).is_err());
    }

    #[test]
    fn test_step_basic() {
        let mut grid = FluidGrid3D::new(10, 10, 10, 0.1, 0.001, 0.016).unwrap();
        grid.add_density(5, 5, 5, 1.0).unwrap();
        grid.step();
        // Density should spread after step
        assert!(grid.validate_state().is_ok());
    }

    #[test]
    fn test_reset() {
        let mut grid = FluidGrid3D::new(10, 10, 10, 0.1, 0.001, 0.016).unwrap();
        grid.add_density(5, 5, 5, 1.0).unwrap();
        grid.add_velocity(5, 5, 5, 1.0, 1.0, 1.0).unwrap();
        grid.reset();
        assert_eq!(grid.get_total_mass(), 0.0);
        assert_eq!(grid.get_average_velocity(), 0.0);
    }

    #[test]
    fn test_solver_config() {
        let config = SolverConfig::high_quality();
        let grid = FluidGrid3D::with_solver(10, 10, 10, 0.1, 0.001, 0.016, config).unwrap();
        assert_eq!(grid.get_solver_iterations(), 20);
    }

    #[test]
    fn test_kinetic_energy() {
        let mut grid = FluidGrid3D::new(10, 10, 10, 0.1, 0.001, 0.016).unwrap();
        assert_eq!(grid.get_kinetic_energy(), 0.0);

        grid.add_density(5, 5, 5, 2.0).unwrap();
        grid.add_velocity(5, 5, 5, 3.0, 4.0, 0.0).unwrap();

        let energy = grid.get_kinetic_energy();
        // KE = 0.5 * m * v^2 = 0.5 * 2.0 * (9 + 16) = 25.0
        assert!((energy - 25.0).abs() < 1e-10);
    }
}
