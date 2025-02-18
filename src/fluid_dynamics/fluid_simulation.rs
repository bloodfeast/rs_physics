// src/fluid_simulation.rs

use crate::utils::PhysicsError;
use std::vec::Vec;

/// A 2D grid-based fluid simulation using the Eulerian method.
///
/// This struct implements a stable fluid solver based on Jos Stam's method,
/// which provides unconditionally stable fluid simulation. The simulation
/// handles density diffusion, velocity diffusion, and advection in a 2D grid.
///
/// # Fields
/// * `width` - The width of the simulation grid
/// * `height` - The height of the simulation grid
/// * `density` - The fluid density at each grid cell
/// * `velocity_x` - The x-component of velocity at each grid cell
/// * `velocity_y` - The y-component of velocity at each grid cell
/// * `diffusion` - The rate at which quantities diffuse through the fluid
/// * `viscosity` - The fluid's resistance to flow
/// * `dt` - The time step for the simulation
pub struct FluidGrid {
    width: usize,
    height: usize,
    density: Vec<f64>,
    velocity_x: Vec<f64>,
    velocity_y: Vec<f64>,
    diffusion: f64,
    viscosity: f64,
    dt: f64,
}

impl FluidGrid {
    /// Creates a new fluid simulation grid with the specified dimensions and properties.
    ///
    /// # Arguments
    /// * `width` - The width of the simulation grid
    /// * `height` - The height of the simulation grid
    /// * `diffusion` - The rate of diffusion (must be non-negative)
    /// * `viscosity` - The fluid viscosity (must be non-negative)
    /// * `dt` - The time step for the simulation (must be positive)
    ///
    /// # Returns
    /// * `Ok(FluidGrid)` - A new fluid simulation grid if all parameters are valid
    /// * `Err(PhysicsError)` - If any parameters are invalid
    ///
    /// # Examples
    /// ```
    /// use rs_physics::fluid_dynamics::FluidGrid;
    ///
    /// // Create a 100x100 grid with water-like properties
    /// let fluid = FluidGrid::new(100, 100, 0.1, 0.001, 0.016).unwrap();
    ///
    /// // Invalid parameters will return an error
    /// let invalid_fluid = FluidGrid::new(0, 100, 0.1, 0.001, 0.016);
    /// assert!(invalid_fluid.is_err());
    /// ```
    pub fn new(
        width: usize,
        height: usize,
        diffusion: f64,
        viscosity: f64,
        dt: f64,
    ) -> Result<Self, PhysicsError> {
        if width == 0 || height == 0 {
            return Err(PhysicsError::InvalidArea);
        }
        if diffusion < 0.0 || viscosity < 0.0 {
            return Err(PhysicsError::InvalidCoefficient);
        }
        if dt <= 0.0 {
            return Err(PhysicsError::InvalidTime);
        }

        let size = width * height;
        Ok(Self {
            width,
            height,
            density: vec![0.0; size],
            velocity_x: vec![0.0; size],
            velocity_y: vec![0.0; size],
            diffusion,
            viscosity,
            dt,
        })
    }

    /// Adds density to the fluid at a specific grid position.
    ///
    /// # Arguments
    /// * `x` - The x-coordinate in the grid
    /// * `y` - The y-coordinate in the grid
    /// * `amount` - The amount of density to add (can be negative to remove density)
    ///
    /// # Returns
    /// * `Ok(())` - If the density was successfully added
    /// * `Err(PhysicsError)` - If the position is out of bounds
    ///
    /// # Examples
    /// ```
    /// use rs_physics::fluid_dynamics::FluidGrid;
    ///
    /// let mut fluid = FluidGrid::new(100, 100, 0.1, 0.001, 0.016).unwrap();
    ///
    /// // Add smoke at position (50, 50)
    /// fluid.add_density(50, 50, 1.0).unwrap();
    ///
    /// // Remove some density (create a sink)
    /// fluid.add_density(50, 50, -0.5).unwrap();
    ///
    /// // Attempting to add density outside the grid returns an error
    /// assert!(fluid.add_density(100, 50, 1.0).is_err());
    /// ```
    pub fn add_density(&mut self, x: usize, y: usize, amount: f64) -> Result<(), PhysicsError> {
        if x >= self.width || y >= self.height {
            return Err(PhysicsError::CalculationError("Position out of bounds".to_string()));
        }
        let idx = self.get_index(x, y);
        self.density[idx] += amount;
        Ok(())
    }

    /// Adds velocity to the fluid at a specific grid position.
    ///
    /// # Arguments
    /// * `x` - The x-coordinate in the grid
    /// * `y` - The y-coordinate in the grid
    /// * `amount_x` - The amount of velocity to add in the x direction
    /// * `amount_y` - The amount of velocity to add in the y direction
    ///
    /// # Returns
    /// * `Ok(())` - If the velocity was successfully added
    /// * `Err(PhysicsError)` - If the position is out of bounds
    ///
    /// # Examples
    /// ```
    /// use rs_physics::fluid_dynamics::FluidGrid;
    ///
    /// let mut fluid = FluidGrid::new(100, 100, 0.1, 0.001, 0.016).unwrap();
    ///
    /// // Create an upward wind
    /// fluid.add_velocity(50, 50, 0.0, -1.0).unwrap();
    ///
    /// // Create a vortex with four velocity vectors
    /// fluid.add_velocity(45, 45, 1.0, 1.0).unwrap();
    /// fluid.add_velocity(45, 55, 1.0, -1.0).unwrap();
    /// fluid.add_velocity(55, 45, -1.0, 1.0).unwrap();
    /// fluid.add_velocity(55, 55, -1.0, -1.0).unwrap();
    /// ```
    pub fn add_velocity(&mut self, x: usize, y: usize, amount_x: f64, amount_y: f64) -> Result<(), PhysicsError> {
        if x >= self.width || y >= self.height {
            return Err(PhysicsError::CalculationError("Position out of bounds".to_string()));
        }
        let idx = self.get_index(x, y);
        self.velocity_x[idx] += amount_x;
        self.velocity_y[idx] += amount_y;
        Ok(())
    }

    /// Advances the fluid simulation by one time step.
    ///
    /// This method performs the main fluid simulation steps in the following order:
    /// 1. Velocity diffusion - Simulates viscous spreading of velocity
    /// 2. Mass conservation (projection) - Ensures incompressibility
    /// 3. Velocity advection - Moves velocity with the flow
    /// 4. Mass conservation (projection) - Ensures incompressibility after advection
    /// 5. Density diffusion - Simulates spreading of density
    /// 6. Density advection - Moves density with the flow
    ///
    /// # Examples
    /// ```
    /// use rs_physics::fluid_dynamics::FluidGrid;
    ///
    /// let mut fluid = FluidGrid::new(100, 100, 0.1, 0.001, 0.016).unwrap();
    ///
    /// // Set up initial conditions
    /// fluid.add_density(50, 50, 1.0).unwrap();
    /// fluid.add_velocity(50, 50, 0.0, -1.0).unwrap();
    ///
    /// // Simulate for 10 steps
    /// for _ in 0..10 {
    ///     fluid.step();
    /// }
    /// ```
    pub fn step(&mut self) {
        let size = self.width * self.height;
        let mut velocity_x0 = vec![0.0; size];
        let mut velocity_y0 = vec![0.0; size];
        let mut density0 = vec![0.0; size];

        // Clone the current state
        velocity_x0.copy_from_slice(&self.velocity_x);
        velocity_y0.copy_from_slice(&self.velocity_y);
        density0.copy_from_slice(&self.density);

        // Diffuse velocity
        {
            let a = self.dt * self.viscosity * (self.width * self.height) as f64;
            self.lin_solve(1, &mut velocity_x0, &self.velocity_x, a, 1.0 + 4.0 * a);
            self.lin_solve(2, &mut velocity_y0, &self.velocity_y, a, 1.0 + 4.0 * a);
        }

        // Project velocity
        self.project(&mut velocity_x0, &mut velocity_y0);

        // Advect velocity
        {
            let mut next_velocity_x = vec![0.0; size];
            let mut next_velocity_y = vec![0.0; size];

            self.advect(1, &mut next_velocity_x, &velocity_x0, &velocity_x0, &velocity_y0);
            self.advect(2, &mut next_velocity_y, &velocity_y0, &velocity_x0, &velocity_y0);

            self.velocity_x = next_velocity_x;
            self.velocity_y = next_velocity_y;
        }

        // Project again
        {
            let mut next_velocity_x = self.velocity_x.clone();
            let mut next_velocity_y = self.velocity_y.clone();
            self.project(&mut next_velocity_x, &mut next_velocity_y);
            self.velocity_x = next_velocity_x;
            self.velocity_y = next_velocity_y;
        }

        // Diffuse density
        {
            let a = self.dt * self.diffusion * (self.width * self.height) as f64;
            self.lin_solve(0, &mut density0, &self.density, a, 1.0 + 4.0 * a);
        }

        // Advect density
        {
            let mut next_density = vec![0.0; size];
            self.advect(0, &mut next_density, &density0, &self.velocity_x, &self.velocity_y);
            self.density = next_density;
        }
    }

    /// Gets the density value at a specific grid position.
    ///
    /// # Arguments
    /// * `x` - The x-coordinate in the grid
    /// * `y` - The y-coordinate in the grid
    ///
    /// # Returns
    /// * `Ok(f64)` - The density value at the specified position
    /// * `Err(PhysicsError)` - If the position is out of bounds
    ///
    /// # Examples
    /// ```
    /// use rs_physics::fluid_dynamics::FluidGrid;
    ///
    /// let mut fluid = FluidGrid::new(100, 100, 0.1, 0.001, 0.016).unwrap();
    /// fluid.add_density(50, 50, 1.0).unwrap();
    ///
    /// // Read the density at a position
    /// let density = fluid.get_density(50, 50).unwrap();
    /// assert_eq!(density, 1.0);
    ///
    /// // Reading outside the grid returns an error
    /// assert!(fluid.get_density(100, 50).is_err());
    /// ```
    pub fn get_density(&self, x: usize, y: usize) -> Result<f64, PhysicsError> {
        if x >= self.width || y >= self.height {
            return Err(PhysicsError::CalculationError("Position out of bounds".to_string()));
        }
        Ok(self.density[self.get_index(x, y)])
    }

    /// Gets the velocity components at a specific grid position.
    ///
    /// # Arguments
    /// * `x` - The x-coordinate in the grid
    /// * `y` - The y-coordinate in the grid
    ///
    /// # Returns
    /// * `Ok((f64, f64))` - A tuple of (x-velocity, y-velocity) at the specified position
    /// * `Err(PhysicsError)` - If the position is out of bounds
    ///
    /// # Examples
    /// ```
    /// use rs_physics::fluid_dynamics::FluidGrid;
    ///
    /// let mut fluid = FluidGrid::new(100, 100, 0.1, 0.001, 0.016).unwrap();
    /// fluid.add_velocity(50, 50, 1.0, -1.0).unwrap();
    ///
    /// // Read the velocity components
    /// let (vx, vy) = fluid.get_velocity(50, 50).unwrap();
    /// assert_eq!(vx, 1.0);
    /// assert_eq!(vy, -1.0);
    ///
    /// // Calculate the velocity magnitude
    /// let magnitude = (vx * vx + vy * vy).sqrt();
    /// assert_eq!(magnitude, 2.0_f64.sqrt());
    /// ```
    pub fn get_velocity(&self, x: usize, y: usize) -> Result<(f64, f64), PhysicsError> {
        if x >= self.width || y >= self.height {
            return Err(PhysicsError::CalculationError("Position out of bounds".to_string()));
        }
        let idx = self.get_index(x, y);
        Ok((self.velocity_x[idx], self.velocity_y[idx]))
    }

    /// Converts 2D coordinates to a 1D array index.
    fn get_index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    /// Projects the velocity field to make it mass-conserving.
    ///
    /// This method enforces incompressibility in the fluid by calculating and
    /// subtracting the pressure gradient. It follows the Helmholtz-Hodge
    /// decomposition to project the velocity field onto a divergence-free field.
    ///
    /// # Arguments
    /// * `velocity_x` - The x-component of the velocity field to be projected
    /// * `velocity_y` - The y-component of the velocity field to be projected
    ///
    /// # Note
    /// This is a key step in maintaining physical accuracy of the simulation,
    /// ensuring that the fluid behaves as an incompressible fluid would.
    fn project(&self, velocity_x: &mut Vec<f64>, velocity_y: &mut Vec<f64>) {
        let size = self.width * self.height;
        let mut p = vec![0.0; size];
        let mut div = vec![0.0; size];

        // Calculate divergence
        for i in 1..self.width-1 {
            for j in 1..self.height-1 {
                let idx = self.get_index(i, j);
                div[idx] = -0.5 * (
                    velocity_x[self.get_index(i+1, j)] -
                        velocity_x[self.get_index(i-1, j)] +
                        velocity_y[self.get_index(i, j+1)] -
                        velocity_y[self.get_index(i, j-1)]
                ) / self.width as f64;
                p[idx] = 0.0;
            }
        }

        self.set_boundaries(0, &mut div);
        self.set_boundaries(0, &mut p);
        self.lin_solve(0, &mut p, &div, 1.0, 4.0);

        // Subtract pressure gradient
        for i in 1..self.width-1 {
            for j in 1..self.height-1 {
                let idx = self.get_index(i, j);
                velocity_x[idx] -= 0.5 * (p[self.get_index(i+1, j)] - p[self.get_index(i-1, j)]) * self.width as f64;
                velocity_y[idx] -= 0.5 * (p[self.get_index(i, j+1)] - p[self.get_index(i, j-1)]) * self.height as f64;
            }
        }

        self.set_boundaries(1, velocity_x);
        self.set_boundaries(2, velocity_y);
    }

    /// Sets the boundary conditions for the fluid simulation.
    ///
    /// # Arguments
    /// * `b` - The boundary condition type:
    ///   * 0 for density (no-flux condition)
    ///   * 1 for x-velocity (no-slip condition)
    ///   * 2 for y-velocity (no-slip condition)
    /// * `x` - The field to apply boundary conditions to
    ///
    /// # Note
    /// The boundary conditions ensure that:
    /// - Fluid cannot flow through walls (no-slip condition)
    /// - Density is conserved at boundaries (no-flux condition)
    /// - Corner values are properly interpolated
    fn set_boundaries(&self, b: i32, x: &mut Vec<f64>) {
        for i in 1..self.width-1 {
            x[self.get_index(i, 0)] = if b == 2 {
                -x[self.get_index(i, 1)]
            } else {
                x[self.get_index(i, 1)]
            };
            x[self.get_index(i, self.height-1)] = if b == 2 {
                -x[self.get_index(i, self.height-2)]
            } else {
                x[self.get_index(i, self.height-2)]
            };
        }

        for j in 1..self.height-1 {
            x[self.get_index(0, j)] = if b == 1 {
                -x[self.get_index(1, j)]
            } else {
                x[self.get_index(1, j)]
            };
            x[self.get_index(self.width-1, j)] = if b == 1 {
                -x[self.get_index(self.width-2, j)]
            } else {
                x[self.get_index(self.width-2, j)]
            };
        }

        x[self.get_index(0, 0)] = 0.5 * (
            x[self.get_index(1, 0)] +
                x[self.get_index(0, 1)]
        );
        x[self.get_index(0, self.height-1)] = 0.5 * (
            x[self.get_index(1, self.height-1)] +
                x[self.get_index(0, self.height-2)]
        );
        x[self.get_index(self.width-1, 0)] = 0.5 * (
            x[self.get_index(self.width-2, 0)] +
                x[self.get_index(self.width-1, 1)]
        );
        x[self.get_index(self.width-1, self.height-1)] = 0.5 * (
            x[self.get_index(self.width-2, self.height-1)] +
                x[self.get_index(self.width-1, self.height-2)]
        );
    }

    /// Solves a linear system using Gauss-Seidel relaxation.
    ///
    /// # Arguments
    /// * `b` - The boundary condition type
    /// * `x` - The field to solve for
    /// * `x0` - The source field
    /// * `a` - The diffusion/viscosity rate multiplied by dt
    /// * `c` - The center cell coefficient (1 + 4a)
    ///
    /// # Note
    /// This method performs iterative relaxation to solve the diffusion equation:
    /// x = (x0 + a * (left + right + top + bottom)) / c
    /// The number of iterations affects the accuracy of the solution.
    fn lin_solve(&self, b: i32, x: &mut Vec<f64>, x0: &Vec<f64>, a: f64, c: f64) {
        let iter = 4;
        for _ in 0..iter {
            for i in 1..self.width-1 {
                for j in 1..self.height-1 {
                    let idx = self.get_index(i, j);
                    x[idx] = (x0[idx] + a * (
                        x[self.get_index(i+1, j)] +
                            x[self.get_index(i-1, j)] +
                            x[self.get_index(i, j+1)] +
                            x[self.get_index(i, j-1)]
                    )) / c;
                }
            }
            self.set_boundaries(b, x);
        }
    }

    /// Performs semi-Lagrangian advection of a quantity through the velocity field.
    ///
    /// # Arguments
    /// * `b` - The boundary condition type
    /// * `d` - The field to advect (output)
    /// * `d0` - The source field
    /// * `velocity_x` - The x-component of the velocity field
    /// * `velocity_y` - The y-component of the velocity field
    ///
    /// # Note
    /// This method:
    /// 1. Traces particles backwards through the velocity field
    /// 2. Interpolates the source field at the traced positions
    /// 3. Uses bilinear interpolation for smooth results
    /// 4. Ensures particles stay within the grid bounds
    fn advect(&self, b: i32, d: &mut Vec<f64>, d0: &Vec<f64>, velocity_x: &Vec<f64>, velocity_y: &Vec<f64>) {
        let dt0 = self.dt * self.width as f64;

        for i in 1..self.width-1 {
            for j in 1..self.height-1 {
                let mut x = i as f64 - dt0 * velocity_x[self.get_index(i, j)];
                let mut y = j as f64 - dt0 * velocity_y[self.get_index(i, j)];

                x = x.clamp(0.5, self.width as f64 - 1.5);
                y = y.clamp(0.5, self.height as f64 - 1.5);

                let i0 = x.floor() as usize;
                let i1 = i0 + 1;
                let j0 = y.floor() as usize;
                let j1 = j0 + 1;

                let s1 = x - i0 as f64;
                let s0 = 1.0 - s1;
                let t1 = y - j0 as f64;
                let t0 = 1.0 - t1;

                let idx = self.get_index(i, j);
                d[idx] = s0 * (t0 * d0[self.get_index(i0, j0)] + t1 * d0[self.get_index(i0, j1)]) +
                    s1 * (t0 * d0[self.get_index(i1, j0)] + t1 * d0[self.get_index(i1, j1)]);
            }
        }

        self.set_boundaries(b, d);
    }

    /// Gets the width of the simulation grid.
    ///
    /// # Returns
    /// The width of the grid in cells.
    pub fn get_width(&self) -> usize {
        self.width
    }

    /// Gets the height of the simulation grid.
    ///
    /// # Returns
    /// The height of the grid in cells.
    pub fn get_height(&self) -> usize {
        self.height
    }

    /// Gets the diffusion rate of the fluid.
    ///
    /// # Returns
    /// The diffusion coefficient.
    pub fn get_diffusion(&self) -> f64 {
        self.diffusion
    }

    /// Gets the viscosity of the fluid.
    ///
    /// # Returns
    /// The viscosity coefficient.
    pub fn get_viscosity(&self) -> f64 {
        self.viscosity
    }

    /// Gets the time step of the simulation.
    ///
    /// # Returns
    /// The time step in seconds.
    pub fn get_dt(&self) -> f64 {
        self.dt
    }

    /// Sets the diffusion rate of the fluid.
    ///
    /// # Arguments
    /// * `diffusion` - The new diffusion coefficient (must be non-negative)
    ///
    /// # Returns
    /// * `Ok(())` if the diffusion rate was successfully set
    /// * `Err(PhysicsError)` if the diffusion rate is negative
    pub fn set_diffusion(&mut self, diffusion: f64) -> Result<(), PhysicsError> {
        if diffusion < 0.0 {
            return Err(PhysicsError::InvalidCoefficient);
        }
        self.diffusion = diffusion;
        Ok(())
    }

    /// Sets the viscosity of the fluid.
    ///
    /// # Arguments
    /// * `viscosity` - The new viscosity coefficient (must be non-negative)
    ///
    /// # Returns
    /// * `Ok(())` if the viscosity was successfully set
    /// * `Err(PhysicsError)` if the viscosity is negative
    pub fn set_viscosity(&mut self, viscosity: f64) -> Result<(), PhysicsError> {
        if viscosity < 0.0 {
            return Err(PhysicsError::InvalidCoefficient);
        }
        self.viscosity = viscosity;
        Ok(())
    }

    /// Sets the time step of the simulation.
    ///
    /// # Arguments
    /// * `dt` - The new time step in seconds (must be positive)
    ///
    /// # Returns
    /// * `Ok(())` if the time step was successfully set
    /// * `Err(PhysicsError)` if the time step is zero or negative
    pub fn set_dt(&mut self, dt: f64) -> Result<(), PhysicsError> {
        if dt <= 0.0 {
            return Err(PhysicsError::InvalidTime);
        }
        self.dt = dt;
        Ok(())
    }

    /// Resets the simulation to its initial state.
    ///
    /// This method clears all density and velocity fields, setting them to zero.
    pub fn reset(&mut self) {
        let size = self.width * self.height;
        self.density = vec![0.0; size];
        self.velocity_x = vec![0.0; size];
        self.velocity_y = vec![0.0; size];
    }

    /// Calculates the total mass (sum of density) in the simulation.
    ///
    /// # Returns
    /// The total mass in the simulation.
    ///
    /// # Examples
    /// ```
    /// use rs_physics::fluid_dynamics::FluidGrid;
    ///
    /// let mut fluid = FluidGrid::new(100, 100, 0.1, 0.001, 0.016).unwrap();
    /// assert_eq!(fluid.get_total_mass(), 0.0);
    ///
    /// // Add some density and check total mass
    /// fluid.add_density(50, 50, 2.0).unwrap();
    /// fluid.add_density(51, 50, 3.0).unwrap();
    /// assert_eq!(fluid.get_total_mass(), 5.0);
    /// ```
    pub fn get_total_mass(&self) -> f64 {
        self.density.iter().sum()
    }

    /// Calculates the average velocity magnitude in the simulation.
    ///
    /// # Returns
    /// The average velocity magnitude across all cells.
    pub fn get_average_velocity(&self) -> f64 {
        let size = self.width * self.height;
        let total_velocity: f64 = (0..size)
            .map(|i| (self.velocity_x[i].powi(2) + self.velocity_y[i].powi(2)).sqrt())
            .sum();
        total_velocity / size as f64
    }

    /// Calculates the kinetic energy of the fluid.
    ///
    /// # Returns
    /// The total kinetic energy in the simulation.
    ///
    /// # Examples
    /// ```
    /// use rs_physics::fluid_dynamics::FluidGrid;
    ///
    /// let mut fluid = FluidGrid::new(100, 100, 0.1, 0.001, 0.016).unwrap();
    /// assert_eq!(fluid.get_kinetic_energy(), 0.0);
    ///
    /// // Add density and velocity to create kinetic energy
    /// fluid.add_density(50, 50, 2.0).unwrap();
    /// fluid.add_velocity(50, 50, 3.0, 4.0).unwrap();
    ///
    /// let energy = fluid.get_kinetic_energy();
    /// assert!(energy > 0.0);
    /// ```
    pub fn get_kinetic_energy(&self) -> f64 {
        let size = self.width * self.height;
        (0..size)
            .map(|i| {
                0.5 * self.density[i] *
                    (self.velocity_x[i].powi(2) + self.velocity_y[i].powi(2))
            })
            .sum()
    }

    /// Checks if the simulation state is valid.
    ///
    /// This method verifies that all density values are non-negative and
    /// that velocity values are finite.
    ///
    /// # Returns
    /// * `Ok(())` if the simulation state is valid
    /// * `Err(PhysicsError)` if any invalid values are found
    pub fn validate_state(&self) -> Result<(), PhysicsError> {
        // Check density values
        if self.density.iter().any(|&d| d < 0.0 || !d.is_finite()) {
            return Err(PhysicsError::CalculationError(
                "Invalid density values detected".to_string()
            ));
        }

        // Check velocity values
        if self.velocity_x.iter().chain(self.velocity_y.iter())
            .any(|&v| !v.is_finite()) {
            return Err(PhysicsError::CalculationError(
                "Invalid velocity values detected".to_string()
            ));
        }

        Ok(())
    }
}
