//! 3D Grid-based fluid simulation using the Eulerian method.
//!
//! This module extends the 2D fluid simulation to three dimensions,
//! implementing Jos Stam's stable fluid solver for volumetric simulations.

use crate::utils::PhysicsError;
use super::validation::{validate_dimensions_3d, validate_finite, validate_non_negative, validate_positive, validate_position_3d};
use super::solver::{
    pcg_solve, AdvectionScheme, BoundaryType, PressureSolver, SolverConfig, SolverType, VorticityConfinement,
};
use super::fluid_simulation::{cell_offset, fit, SolveScratch, MIC_SIGMA, MIC_TAU};

/// Buffers `step` reuses, so a step allocates nothing once the first has sized them.
/// The option buffers are sized the first time their option runs.
#[derive(Default)]
struct Workspace {
    solve: SolveScratch,
    /// The velocity and density as diffused, before advection.
    velocity_x0: Vec<f64>,
    velocity_y0: Vec<f64>,
    velocity_z0: Vec<f64>,
    density0: Vec<f64>,
    /// [`AdvectionScheme::MacCormack`]: the backward pass, and the range of the source
    /// cells the forward pass read for each cell.
    back: Vec<f64>,
    low: Vec<f64>,
    high: Vec<f64>,
    /// [`VorticityConfinement::MatchNumericalDissipation`]: the curl's three
    /// components and its magnitude.
    omega: [Vec<f64>; 4],
}

/// A 3D grid-based fluid simulation using the Eulerian method.
///
/// This struct implements a stable fluid solver based on Jos Stam's method,
/// extended to three dimensions. The simulation handles density diffusion,
/// velocity diffusion, and advection in a 3D grid.
///
/// # Units
/// As in [`FluidGrid`](crate::fluid_dynamics::FluidGrid), length is measured in
/// domain widths: every cell is `1 / width` on a side, on all three axes.
/// Velocities are in widths per second and `viscosity` and `diffusion` in widths²
/// per second. The outermost layer of cells is a boundary layer that each step
/// overwrites; the fluid is the `(width - 2) × (height - 2) × (depth - 2)` cells
/// inside it, each coordinate in `1..n-1`. [`FluidGrid3D::add_density`] and
/// [`FluidGrid3D::add_velocity`] refuse the boundary layer; the getters read it.
///
/// # Walls
/// Every wall stops the flow through it. Along it, walls are free-slip by default
/// and no-slip with [`WallCondition::NoSlip`](super::WallCondition) in the
/// [`SolverConfig`].
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
    /// The last conjugate-gradient pressure of each of a step's two projections: the
    /// starting guess for the same projection next step. The two solve different
    /// problems (the forced field, then the advected one), so one shared guess would
    /// start each from the other's answer.
    pressure: [Vec<f64>; 2],
    /// MIC(0) preconditioner for the pressure matrix, as inverse pivots `1/e`; it
    /// depends only on the grid's shape.
    precon: Vec<f64>,
    workspace: Workspace,
    last_pressure_iterations: usize,
}

impl FluidGrid3D {
    /// Creates a new 3D fluid simulation grid with the specified dimensions and properties.
    ///
    /// # Arguments
    /// * `width` - The width of the simulation grid (x-axis), in cells (at least 3)
    /// * `height` - The height of the simulation grid (y-axis), in cells (at least 3)
    /// * `depth` - The depth of the simulation grid (z-axis), in cells (at least 3)
    /// * `diffusion` - The rate of diffusion, widths²/s (must be finite and non-negative)
    /// * `viscosity` - The kinematic viscosity, widths²/s (must be finite and non-negative)
    /// * `dt` - The time step for the simulation, seconds (must be finite and positive)
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
    ///
    /// Takes the same arguments as [`FluidGrid3D::new`], plus the solver
    /// configuration; fewer than 1 iteration is raised to 1, as in
    /// [`FluidGrid3D::set_solver_iterations`]. Returns an error for a configuration
    /// that [`SolverConfig::validate`] rejects.
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
        // The boundary layer takes one cell on each side, and `set_boundaries`
        // indexes `n - 2`: a grid needs at least one fluid cell on each axis.
        if width < 3 || height < 3 || depth < 3 {
            return Err(PhysicsError::InvalidArea);
        }
        Self::validate_diffusion(diffusion)?;
        Self::validate_viscosity(viscosity)?;
        Self::validate_dt(dt)?;
        let solver_config = solver_config.checked()?;

        let size = width * height * depth;
        let mut grid = Self {
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
            pressure: [vec![0.0; size], vec![0.0; size]],
            precon: vec![0.0; size],
            workspace: Workspace::default(),
            last_pressure_iterations: 0,
        };
        grid.build_preconditioner();
        Ok(grid)
    }

    /// Refuses a position outside the fluid: off the grid, or in the boundary
    /// layer, which every step overwrites.
    fn validate_fluid_cell(&self, x: usize, y: usize, z: usize) -> Result<(), PhysicsError> {
        validate_position_3d(x, y, z, self.width, self.height, self.depth)?;
        if x == 0 || y == 0 || z == 0
            || x == self.width - 1 || y == self.height - 1 || z == self.depth - 1
        {
            return Err(PhysicsError::CalculationError(format!(
                "({x}, {y}, {z}) is in the boundary layer, which every step overwrites; \
                 the fluid is x in 1..={}, y in 1..={}, z in 1..={}",
                self.width - 2,
                self.height - 2,
                self.depth - 2
            )));
        }
        Ok(())
    }

    /// Adds density to the fluid at a specific grid position, a fluid cell (each
    /// coordinate in `1..n-1`).
    ///
    /// Returns an error if the position is out of bounds or in the boundary layer
    /// (which the next step would overwrite), or `amount` is not finite.
    pub fn add_density(&mut self, x: usize, y: usize, z: usize, amount: f64) -> Result<(), PhysicsError> {
        self.validate_fluid_cell(x, y, z)?;
        validate_finite(amount, "amount")?;
        let idx = self.get_index(x, y, z);
        self.density[idx] += amount;
        Ok(())
    }

    /// Adds velocity, in domain widths per second, to the fluid at a specific grid
    /// position, a fluid cell (each coordinate in `1..n-1`).
    ///
    /// Returns an error if the position is out of bounds or in the boundary layer,
    /// or a component is not finite.
    pub fn add_velocity(&mut self, x: usize, y: usize, z: usize, vx: f64, vy: f64, vz: f64) -> Result<(), PhysicsError> {
        self.validate_fluid_cell(x, y, z)?;
        if !vx.is_finite() || !vy.is_finite() || !vz.is_finite() {
            return Err(PhysicsError::InvalidVelocity);
        }
        let idx = self.get_index(x, y, z);
        self.velocity_x[idx] += vx;
        self.velocity_y[idx] += vy;
        self.velocity_z[idx] += vz;
        Ok(())
    }

    /// Advances the fluid simulation by one time step.
    ///
    /// The order is that of [`FluidGrid::step`](crate::fluid_dynamics::FluidGrid::step):
    /// diffuse and project the velocity, advect and project it again, then diffuse
    /// and advect the density. With the default [`PressureSolver::ConjugateGradient`]
    /// each projection solves the pressure to the configured relative tolerance (or
    /// iteration cap); see [`FluidGrid3D::get_last_pressure_iterations`].
    pub fn step(&mut self) {
        let mut ws = std::mem::take(&mut self.workspace);
        let [mut pressure, mut pressure_after_advection] = std::mem::take(&mut self.pressure);
        // The state is taken out for the step and each buffer reused in place, so a
        // step allocates nothing; as in the 2D grid, every buffer a pass writes is
        // overwritten whole, so the results are bit-identical to the allocating version.
        let mut velocity_x = std::mem::take(&mut self.velocity_x);
        let mut velocity_y = std::mem::take(&mut self.velocity_y);
        let mut velocity_z = std::mem::take(&mut self.velocity_z);
        let mut density = std::mem::take(&mut self.density);
        let size = self.width * self.height * self.depth;
        for v in [&mut ws.velocity_x0, &mut ws.velocity_y0, &mut ws.velocity_z0, &mut ws.density0] {
            fit(v, size);
        }

        // Clone the current state
        ws.velocity_x0.copy_from_slice(&velocity_x);
        ws.velocity_y0.copy_from_slice(&velocity_y);
        ws.velocity_z0.copy_from_slice(&velocity_z);
        ws.density0.copy_from_slice(&density);

        // Implicit diffusion: `a = dt * nu / h²`, with the cell size `h = 1 / width` on
        // every axis (the same `h` that `advect` and `project` use). The cell count
        // `width * height * depth` made viscosity and diffusion N times too strong
        // on an N³ grid, and resolution-dependent.
        let inv_h_sq = (self.width * self.width) as f64;

        // Diffuse velocity
        {
            let a = self.dt * self.viscosity * inv_h_sq;
            self.lin_solve(BoundaryType::VelocityX, &mut ws.velocity_x0, &velocity_x, a, 1.0 + 6.0 * a, &mut ws.solve.jacobi);
            self.lin_solve(BoundaryType::VelocityY, &mut ws.velocity_y0, &velocity_y, a, 1.0 + 6.0 * a, &mut ws.solve.jacobi);
            self.lin_solve(BoundaryType::VelocityZ, &mut ws.velocity_z0, &velocity_z, a, 1.0 + 6.0 * a, &mut ws.solve.jacobi);
        }

        // Project velocity
        let mut pressure_iterations =
            self.project(&mut ws.velocity_x0, &mut ws.velocity_y0, &mut ws.velocity_z0, &mut pressure, &mut ws.solve);

        // Advect velocity, into the buffers the old velocity held
        match self.solver_config.advection {
            AdvectionScheme::SemiLagrangian => {
                let (u, v, w) = (&ws.velocity_x0, &ws.velocity_y0, &ws.velocity_z0);
                self.advect(BoundaryType::VelocityX, &mut velocity_x, u, u, v, w);
                self.advect(BoundaryType::VelocityY, &mut velocity_y, v, u, v, w);
                self.advect(BoundaryType::VelocityZ, &mut velocity_z, w, u, v, w);
            }
            AdvectionScheme::MacCormack => {
                let Workspace { velocity_x0: u, velocity_y0: v, velocity_z0: w, back, low, high, .. } = &mut ws;
                let (u, v, w): (&[f64], &[f64], &[f64]) = (u, v, w);
                for (bt, out, source) in [
                    (BoundaryType::VelocityX, &mut velocity_x, u),
                    (BoundaryType::VelocityY, &mut velocity_y, v),
                    (BoundaryType::VelocityZ, &mut velocity_z, w),
                ] {
                    self.advect_maccormack(bt, out, source, [u, v, w], back, low, high);
                }
            }
        }

        if self.solver_config.vorticity_confinement == VorticityConfinement::MatchNumericalDissipation {
            self.confine_vorticity(
                [&mut velocity_x, &mut velocity_y, &mut velocity_z],
                [&ws.velocity_x0, &ws.velocity_y0, &ws.velocity_z0],
                &mut ws.omega,
            );
        }

        // Project again
        pressure_iterations +=
            self.project(&mut velocity_x, &mut velocity_y, &mut velocity_z, &mut pressure_after_advection, &mut ws.solve);

        // Diffuse density
        {
            let a = self.dt * self.diffusion * inv_h_sq;
            self.lin_solve(BoundaryType::Density, &mut ws.density0, &density, a, 1.0 + 6.0 * a, &mut ws.solve.jacobi);
        }

        // Advect density
        match self.solver_config.advection {
            AdvectionScheme::SemiLagrangian => {
                self.advect(BoundaryType::Density, &mut density, &ws.density0, &velocity_x, &velocity_y, &velocity_z);
            }
            AdvectionScheme::MacCormack => {
                let Workspace { density0, back, low, high, .. } = &mut ws;
                self.advect_maccormack(
                    BoundaryType::Density,
                    &mut density,
                    density0,
                    [&velocity_x, &velocity_y, &velocity_z],
                    back,
                    low,
                    high,
                );
            }
        }

        self.velocity_x = velocity_x;
        self.velocity_y = velocity_y;
        self.velocity_z = velocity_z;
        self.density = density;
        self.pressure = [pressure, pressure_after_advection];
        self.workspace = ws;
        self.last_pressure_iterations = pressure_iterations;
    }

    /// Gets the number of pressure-solve iterations the last [`FluidGrid3D::step`]
    /// ran, summed over its two projections: conjugate-gradient iterations, or
    /// relaxation sweeps under [`PressureSolver::Relaxation`]. A count of twice
    /// [`SolverConfig::pressure_max_iterations`] means the tolerance was not reached.
    pub fn get_last_pressure_iterations(&self) -> usize {
        self.last_pressure_iterations
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
    ///
    /// `z` is the fastest-varying index because every sweep in this file loops
    /// `x`, then `y`, then `z` innermost. With `x` fastest, the inner loop strode by
    /// a whole `width * height` slab (32 KiB at 64³), so every Gauss-Seidel update
    /// missed cache. Swapping the storage rather than the loops keeps the sweep
    /// order, and so every result, bit-identical.
    #[inline]
    fn get_index(&self, x: usize, y: usize, z: usize) -> usize {
        (x * self.height + y) * self.depth + z
    }

    /// Projects the velocity field to make it mass-conserving (divergence-free).
    ///
    /// The pressure solves `6p - Σ neighbours = -h/2 · (central divergence)`, a wall
    /// neighbour standing in as a copy of the cell itself. Returns the iterations the
    /// pressure solve ran.
    fn project(
        &self,
        velocity_x: &mut Vec<f64>,
        velocity_y: &mut Vec<f64>,
        velocity_z: &mut Vec<f64>,
        pressure: &mut Vec<f64>,
        ws: &mut SolveScratch,
    ) -> usize {
        match self.solver_config.pressure_solver {
            PressureSolver::Relaxation => {
                // Exactly the pre-conjugate-gradient projection: sweeps from zero. The
                // divergence is written whole before it is read; the pressure is cleared.
                let size = self.width * self.height * self.depth;
                fit(&mut ws.relax_pressure, size);
                fit(&mut ws.relax_divergence, size);
                let (p, div) = (&mut ws.relax_pressure, &mut ws.relax_divergence);
                p.fill(0.0);
                self.divergence(velocity_x, velocity_y, velocity_z, div);
                self.set_boundaries(BoundaryType::Density, div);
                self.set_boundaries(BoundaryType::Density, p);
                self.lin_solve(BoundaryType::Density, p, div, 1.0, 6.0, &mut ws.jacobi);
                self.subtract_pressure_gradient(velocity_x, velocity_y, velocity_z, p);
                self.solver_config.iterations
            }
            PressureSolver::ConjugateGradient => {
                let size = self.width * self.height * self.depth;
                if ws.rhs.len() != size {
                    // Zero-filled once: the boundary layer is never written.
                    ws.rhs = vec![0.0; size];
                }
                self.divergence(velocity_x, velocity_y, velocity_z, &mut ws.rhs);
                // The closed box's pressure matrix is singular (constants are its null
                // space); remove the rounding that would leave the system inconsistent.
                let mean = self.fluid_mean(&ws.rhs);
                self.for_each_fluid_cell(|idx| ws.rhs[idx] -= mean);

                let iterations = pcg_solve(
                    pressure,
                    &ws.rhs,
                    &mut ws.pcg,
                    self.solver_config.pressure_tolerance,
                    self.solver_config.pressure_max_iterations,
                    |x, out| self.apply_pressure_matrix(x, out),
                    |r, z| self.apply_preconditioner(r, z),
                );
                let mean = self.fluid_mean(pressure);
                self.for_each_fluid_cell(|idx| pressure[idx] -= mean);
                self.set_boundaries(BoundaryType::Density, pressure);
                self.subtract_pressure_gradient(velocity_x, velocity_y, velocity_z, pressure);
                iterations
            }
        }
    }

    /// Writes `-h/2 · (central-difference divergence)` into the fluid cells of `div`.
    fn divergence(&self, velocity_x: &[f64], velocity_y: &[f64], velocity_z: &[f64], div: &mut [f64]) {
        let h = 1.0 / self.width as f64;
        for i in 1..self.width-1 {
            for j in 1..self.height-1 {
                for k in 1..self.depth-1 {
                    let idx = self.get_index(i, j, k);
                    div[idx] = -0.5 * h * (
                        velocity_x[self.get_index(i+1, j, k)] - velocity_x[self.get_index(i-1, j, k)] +
                        velocity_y[self.get_index(i, j+1, k)] - velocity_y[self.get_index(i, j-1, k)] +
                        velocity_z[self.get_index(i, j, k+1)] - velocity_z[self.get_index(i, j, k-1)]
                    );
                }
            }
        }
    }

    fn subtract_pressure_gradient(&self, velocity_x: &mut Vec<f64>, velocity_y: &mut Vec<f64>, velocity_z: &mut Vec<f64>, p: &[f64]) {
        let h = 1.0 / self.width as f64;
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

    fn for_each_fluid_cell(&self, mut f: impl FnMut(usize)) {
        for i in 1..self.width-1 {
            for j in 1..self.height-1 {
                for k in 1..self.depth-1 {
                    f(self.get_index(i, j, k));
                }
            }
        }
    }

    fn fluid_mean(&self, field: &[f64]) -> f64 {
        let mut sum = 0.0;
        self.for_each_fluid_cell(|idx| sum += field[idx]);
        sum / ((self.width - 2) * (self.height - 2) * (self.depth - 2)) as f64
    }

    /// `out = A·x` for the pressure matrix on the fluid cells: `6x` minus the six
    /// neighbours, a wall neighbour standing in as a copy of the cell.
    fn apply_pressure_matrix(&self, x: &[f64], out: &mut [f64]) {
        let (w, h, d) = (self.width, self.height, self.depth);
        let (si, sj) = (h * d, d);
        for i in 1..w-1 {
            for j in 1..h-1 {
                let row = self.get_index(i, j, 0);
                // A missing neighbour row at a wall is the row itself: decided once
                // per row rather than per cell.
                let xm = if i > 1 { row - si } else { row };
                let xp = if i < w - 2 { row + si } else { row };
                let ym = if j > 1 { row - sj } else { row };
                let yp = if j < h - 2 { row + sj } else { row };
                for k in 1..d-1 {
                    let idx = row + k;
                    let c = x[idx];
                    let zm = if k > 1 { x[idx - 1] } else { c };
                    let zp = if k < d - 2 { x[idx + 1] } else { c };
                    out[idx] = 6.0 * c - (x[xm + k] + x[xp + k] + x[ym + k] + x[yp + k] + zm + zp);
                }
            }
        }
    }

    /// Builds the MIC(0) preconditioner (Bridson §5.4) for the pressure matrix, as
    /// inverse pivots `1/e` (see the 2D grid's `build_preconditioner`).
    fn build_preconditioner(&mut self) {
        let (w, h, d) = (self.width, self.height, self.depth);
        let (si, sj) = (h * d, d);
        let mut inv = std::mem::take(&mut self.precon);
        let has = |b: bool| if b { 1.0 } else { 0.0 };
        for i in 1..w-1 {
            for j in 1..h-1 {
                for k in 1..d-1 {
                    let idx = self.get_index(i, j, k);
                    let diag = has(i > 1) + has(i < w - 2) + has(j > 1) + has(j < h - 2) + has(k > 1) + has(k < d - 2);
                    let mut e = diag;
                    if i > 1 {
                        let q = inv[idx - si];
                        e -= q + MIC_TAU * q * (has(j < h - 2) + has(k < d - 2));
                    }
                    if j > 1 {
                        let q = inv[idx - sj];
                        e -= q + MIC_TAU * q * (has(i < w - 2) + has(k < d - 2));
                    }
                    if k > 1 {
                        let q = inv[idx - 1];
                        e -= q + MIC_TAU * q * (has(i < w - 2) + has(j < h - 2));
                    }
                    if e < MIC_SIGMA * diag {
                        e = diag;
                    }
                    inv[idx] = if e > 0.0 { 1.0 / e } else { 0.0 };
                }
            }
        }
        self.precon = inv;
    }

    /// `z = M⁻¹·r` for the MIC(0) preconditioner, in place in `z`, fluid cells only,
    /// written in terms of the inverse pivots as in the 2D grid.
    fn apply_preconditioner(&self, r: &[f64], z: &mut [f64]) {
        let (w, h, d) = (self.width, self.height, self.depth);
        let (si, sj) = (h * d, d);
        let inv = &self.precon;
        for i in 1..w-1 {
            for j in 1..h-1 {
                for k in 1..d-1 {
                    let idx = self.get_index(i, j, k);
                    let mut t = r[idx];
                    if i > 1 {
                        t += z[idx - si];
                    }
                    if j > 1 {
                        t += z[idx - sj];
                    }
                    if k > 1 {
                        t += z[idx - 1];
                    }
                    z[idx] = inv[idx] * t;
                }
            }
        }
        for i in (1..w-1).rev() {
            for j in (1..h-1).rev() {
                for k in (1..d-1).rev() {
                    let idx = self.get_index(i, j, k);
                    let mut t = z[idx];
                    if i < w - 2 {
                        t += inv[idx] * z[idx + si];
                    }
                    if j < h - 2 {
                        t += inv[idx] * z[idx + sj];
                    }
                    if k < d - 2 {
                        t += inv[idx] * z[idx + 1];
                    }
                    z[idx] = t;
                }
            }
        }
    }

    /// Sets the boundary conditions for the 3D fluid simulation.
    ///
    /// Each velocity component is negated at the pair of walls it crosses, so no
    /// fluid passes through a wall. At the other four walls it is copied under
    /// free-slip and negated under no-slip ([`SolverConfig::wall`]). Density and
    /// pressure are copied (no flux).
    fn set_boundaries(&self, boundary_type: BoundaryType, x: &mut Vec<f64>) {
        let wall = self.solver_config.wall;
        let flip_front_back = boundary_type.flips_at_wall(BoundaryType::VelocityZ, wall);
        let flip_top_bottom = boundary_type.flips_at_wall(BoundaryType::VelocityY, wall);
        let flip_left_right = boundary_type.flips_at_wall(BoundaryType::VelocityX, wall);

        // Handle faces (6 faces)
        for i in 1..self.width-1 {
            for j in 1..self.height-1 {
                // Front face (k=0)
                x[self.get_index(i, j, 0)] = if flip_front_back {
                    -x[self.get_index(i, j, 1)]
                } else {
                    x[self.get_index(i, j, 1)]
                };
                // Back face (k=depth-1)
                x[self.get_index(i, j, self.depth-1)] = if flip_front_back {
                    -x[self.get_index(i, j, self.depth-2)]
                } else {
                    x[self.get_index(i, j, self.depth-2)]
                };
            }
        }

        for i in 1..self.width-1 {
            for k in 1..self.depth-1 {
                // Bottom face (j=0)
                x[self.get_index(i, 0, k)] = if flip_top_bottom {
                    -x[self.get_index(i, 1, k)]
                } else {
                    x[self.get_index(i, 1, k)]
                };
                // Top face (j=height-1)
                x[self.get_index(i, self.height-1, k)] = if flip_top_bottom {
                    -x[self.get_index(i, self.height-2, k)]
                } else {
                    x[self.get_index(i, self.height-2, k)]
                };
            }
        }

        for j in 1..self.height-1 {
            for k in 1..self.depth-1 {
                // Left face (i=0)
                x[self.get_index(0, j, k)] = if flip_left_right {
                    -x[self.get_index(1, j, k)]
                } else {
                    x[self.get_index(1, j, k)]
                };
                // Right face (i=width-1)
                x[self.get_index(self.width-1, j, k)] = if flip_left_right {
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

    /// Solves a linear system by relaxation in 3D: Gauss-Seidel, SOR or Jacobi, per
    /// [`SolverConfig::solver_type`], `iterations` sweeps from the guess in `x`.
    /// `scratch` is Jacobi's second buffer.
    fn lin_solve(&self, boundary_type: BoundaryType, x: &mut Vec<f64>, x0: &[f64], a: f64, c: f64, scratch: &mut Vec<f64>) {
        match self.solver_config.solver_type {
            SolverType::GaussSeidel => {
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
            SolverType::SOR => {
                let omega = self.solver_config.relaxation;
                for _ in 0..self.solver_config.iterations {
                    for i in 1..self.width-1 {
                        for j in 1..self.height-1 {
                            for k in 1..self.depth-1 {
                                let idx = self.get_index(i, j, k);
                                let gauss_seidel = (x0[idx] + a * (
                                    x[self.get_index(i+1, j, k)] + x[self.get_index(i-1, j, k)] +
                                    x[self.get_index(i, j+1, k)] + x[self.get_index(i, j-1, k)] +
                                    x[self.get_index(i, j, k+1)] + x[self.get_index(i, j, k-1)]
                                )) / c;
                                x[idx] = (1.0 - omega) * x[idx] + omega * gauss_seidel;
                            }
                        }
                    }
                    self.set_boundaries(boundary_type, x);
                }
            }
            SolverType::Jacobi => {
                if scratch.len() != x.len() {
                    scratch.clear();
                    scratch.resize(x.len(), 0.0);
                }
                for _ in 0..self.solver_config.iterations {
                    for i in 1..self.width-1 {
                        for j in 1..self.height-1 {
                            for k in 1..self.depth-1 {
                                let idx = self.get_index(i, j, k);
                                scratch[idx] = (x0[idx] + a * (
                                    x[self.get_index(i+1, j, k)] + x[self.get_index(i-1, j, k)] +
                                    x[self.get_index(i, j+1, k)] + x[self.get_index(i, j-1, k)] +
                                    x[self.get_index(i, j, k+1)] + x[self.get_index(i, j, k-1)]
                                )) / c;
                            }
                        }
                    }
                    // The new sweep becomes `x`; `set_boundaries` rewrites its whole
                    // (stale) boundary layer from the fluid cells.
                    std::mem::swap(x, scratch);
                    self.set_boundaries(boundary_type, x);
                }
            }
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

    /// One semi-Lagrangian pass, along `+velocity` when `sign` is `-1` (the usual
    /// back-trace) or `-velocity` when it is `+1` (MacCormack's reverse step), writing
    /// the fluid cells of `d`. When `bounds` is given it also records, per cell, the
    /// smallest and largest of the eight source cells the interpolation read.
    ///
    /// The arithmetic of the `sign = -1` pass is [`FluidGrid3D::advect`]'s, operation
    /// for operation, so MacCormack's forward value is the first-order value exactly.
    fn trace_pass(
        &self,
        d: &mut [f64],
        d0: &[f64],
        velocity: [&[f64]; 3],
        sign: f64,
        mut bounds: Option<(&mut [f64], &mut [f64])>,
    ) {
        let dt0 = self.dt * self.width as f64;
        let [velocity_x, velocity_y, velocity_z] = velocity;
        for i in 1..self.width-1 {
            for j in 1..self.height-1 {
                for k in 1..self.depth-1 {
                    let idx = self.get_index(i, j, k);
                    let x = (i as f64 + sign * dt0 * velocity_x[idx]).clamp(0.5, self.width as f64 - 1.5);
                    let y = (j as f64 + sign * dt0 * velocity_y[idx]).clamp(0.5, self.height as f64 - 1.5);
                    let z = (k as f64 + sign * dt0 * velocity_z[idx]).clamp(0.5, self.depth as f64 - 1.5);

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

                    let c = [
                        d0[self.get_index(i0, j0, k0)],
                        d0[self.get_index(i0, j0, k1)],
                        d0[self.get_index(i0, j1, k0)],
                        d0[self.get_index(i0, j1, k1)],
                        d0[self.get_index(i1, j0, k0)],
                        d0[self.get_index(i1, j0, k1)],
                        d0[self.get_index(i1, j1, k0)],
                        d0[self.get_index(i1, j1, k1)],
                    ];
                    d[idx] = s0 * (t0 * (u0 * c[0] + u1 * c[1]) + t1 * (u0 * c[2] + u1 * c[3]))
                        + s1 * (t0 * (u0 * c[4] + u1 * c[5]) + t1 * (u0 * c[6] + u1 * c[7]));
                    if let Some((low, high)) = bounds.as_mut() {
                        low[idx] = c[0].min(c[1]).min(c[2].min(c[3])).min(c[4].min(c[5]).min(c[6].min(c[7])));
                        high[idx] = c[0].max(c[1]).max(c[2].max(c[3])).max(c[4].max(c[5]).max(c[6].max(c[7])));
                    }
                }
            }
        }
    }

    /// MacCormack advection of `d0` into `d` ([`AdvectionScheme::MacCormack`]); see
    /// the 2D grid's `advect_maccormack`. `back`, `low` and `high` are scratch.
    #[allow(clippy::too_many_arguments)]
    fn advect_maccormack(
        &self,
        boundary_type: BoundaryType,
        d: &mut Vec<f64>,
        d0: &[f64],
        velocity: [&[f64]; 3],
        back: &mut Vec<f64>,
        low: &mut Vec<f64>,
        high: &mut Vec<f64>,
    ) {
        let size = self.width * self.height * self.depth;
        fit(back, size);
        fit(low, size);
        fit(high, size);
        self.trace_pass(d, d0, velocity, -1.0, Some((low, high)));
        // The reverse pass interpolates the forward result, boundary layer included.
        self.set_boundaries(boundary_type, d);
        self.trace_pass(back, d, velocity, 1.0, None);
        for i in 1..self.width-1 {
            for j in 1..self.height-1 {
                let row = self.get_index(i, j, 0);
                for idx in row + 1..row + self.depth - 1 {
                    let corrected = d[idx] + 0.5 * (d0[idx] - back[idx]);
                    d[idx] = corrected.clamp(low[idx], high[idx]);
                }
            }
        }
        self.set_boundaries(boundary_type, d);
    }

    /// Adds the vorticity-confinement velocity
    /// ([`VorticityConfinement::MatchNumericalDissipation`]) to the fluid cells of
    /// `velocity`: `h e (N x omega)` with `e = mean over axes of a(1 - a) / 2`, `a` the
    /// fractional cell offset of each cell's departure point this step, from the
    /// velocity it was advected by. `omega` is scratch: the curl and its magnitude.
    fn confine_vorticity(&self, velocity: [&mut Vec<f64>; 3], advecting: [&Vec<f64>; 3], omega: &mut [Vec<f64>; 4]) {
        let (w, hgt, dep) = (self.width, self.height, self.depth);
        let (si, sj) = (hgt * dep, dep);
        for v in omega.iter_mut() {
            fit(v, w * hgt * dep);
        }
        let width = w as f64;
        let h = 1.0 / width;
        let inv_2h = 0.5 * width;
        let [velocity_x, velocity_y, velocity_z] = velocity;
        {
            let [ox, oy, oz, magnitude] = &mut *omega;
            for i in 1..w-1 {
                for j in 1..hgt-1 {
                    for k in 1..dep-1 {
                        let idx = self.get_index(i, j, k);
                        let dw_dy = velocity_z[idx + sj] - velocity_z[idx - sj];
                        let dv_dz = velocity_y[idx + 1] - velocity_y[idx - 1];
                        let du_dz = velocity_x[idx + 1] - velocity_x[idx - 1];
                        let dw_dx = velocity_z[idx + si] - velocity_z[idx - si];
                        let dv_dx = velocity_y[idx + si] - velocity_y[idx - si];
                        let du_dy = velocity_x[idx + sj] - velocity_x[idx - sj];
                        let (x, y, z) = ((dw_dy - dv_dz) * inv_2h, (du_dz - dw_dx) * inv_2h, (dv_dx - du_dy) * inv_2h);
                        ox[idx] = x;
                        oy[idx] = y;
                        oz[idx] = z;
                        magnitude[idx] = (x * x + y * y + z * z).sqrt();
                    }
                }
            }
            // A wall's ghost copies its neighbour, so |omega| has no gradient through it.
            self.set_boundaries(BoundaryType::Density, magnitude);
        }

        let [ox, oy, oz, magnitude] = &*omega;
        let [ax, ay, az] = advecting;
        let cells_per_velocity = self.dt * width;
        for i in 1..w-1 {
            for j in 1..hgt-1 {
                for k in 1..dep-1 {
                    let idx = self.get_index(i, j, k);
                    let gx = (magnitude[idx + si] - magnitude[idx - si]) * inv_2h;
                    let gy = (magnitude[idx + sj] - magnitude[idx - sj]) * inv_2h;
                    let gz = (magnitude[idx + 1] - magnitude[idx - 1]) * inv_2h;
                    let length = (gx * gx + gy * gy + gz * gz).sqrt();
                    if !(length > 0.0) {
                        continue;
                    }
                    let a = [ax[idx], ay[idx], az[idx]].map(|u| cell_offset(u * cells_per_velocity));
                    let e = (a[0] * (1.0 - a[0]) + a[1] * (1.0 - a[1]) + a[2] * (1.0 - a[2])) / 6.0;
                    let s = h * e / length;
                    let (wx, wy, wz) = (ox[idx], oy[idx], oz[idx]);
                    // N x omega, with N = g / |g| folded into `s`.
                    velocity_x[idx] += s * (gy * wz - gz * wy);
                    velocity_y[idx] += s * (gz * wx - gx * wz);
                    velocity_z[idx] += s * (gx * wy - gy * wx);
                }
            }
        }
        self.set_boundaries(BoundaryType::VelocityX, velocity_x);
        self.set_boundaries(BoundaryType::VelocityY, velocity_y);
        self.set_boundaries(BoundaryType::VelocityZ, velocity_z);
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

    /// Sets the solver configuration. Fewer than 1 iteration is raised to 1.
    ///
    /// Returns an error, keeping the previous configuration, if
    /// [`SolverConfig::validate`] rejects `config`.
    pub fn set_solver_config(&mut self, config: SolverConfig) -> Result<(), PhysicsError> {
        self.solver_config = config.checked()?;
        Ok(())
    }

    /// Sets the diffusion rate of the fluid, widths²/s (must be finite and non-negative).
    pub fn set_diffusion(&mut self, diffusion: f64) -> Result<(), PhysicsError> {
        Self::validate_diffusion(diffusion)?;
        self.diffusion = diffusion;
        Ok(())
    }

    /// Sets the kinematic viscosity of the fluid, widths²/s (must be finite and non-negative).
    pub fn set_viscosity(&mut self, viscosity: f64) -> Result<(), PhysicsError> {
        Self::validate_viscosity(viscosity)?;
        self.viscosity = viscosity;
        Ok(())
    }

    /// Sets the time step of the simulation, seconds (must be finite and positive).
    pub fn set_dt(&mut self, dt: f64) -> Result<(), PhysicsError> {
        Self::validate_dt(dt)?;
        self.dt = dt;
        Ok(())
    }

    // Infinity is positive, so the range checks alone let it through, and one step with
    // an infinite `dt` or coefficient turns every cell NaN.
    fn validate_diffusion(diffusion: f64) -> Result<(), PhysicsError> {
        validate_finite(diffusion, "diffusion")
            .and_then(|_| validate_non_negative(diffusion, "diffusion"))
            .map_err(|_| PhysicsError::InvalidCoefficient)
    }

    fn validate_viscosity(viscosity: f64) -> Result<(), PhysicsError> {
        validate_finite(viscosity, "viscosity")
            .and_then(|_| validate_non_negative(viscosity, "viscosity"))
            .map_err(|_| PhysicsError::InvalidCoefficient)
    }

    fn validate_dt(dt: f64) -> Result<(), PhysicsError> {
        validate_finite(dt, "dt")
            .and_then(|_| validate_positive(dt, "dt"))
            .map_err(|_| PhysicsError::InvalidTime)
    }

    /// Resets the simulation to its initial state.
    pub fn reset(&mut self) {
        let size = self.width * self.height * self.depth;
        self.density = vec![0.0; size];
        self.velocity_x = vec![0.0; size];
        self.velocity_y = vec![0.0; size];
        self.velocity_z = vec![0.0; size];
        self.pressure = [vec![0.0; size], vec![0.0; size]];
        self.last_pressure_iterations = 0;
    }

    /// Calculates the total mass (sum of density) in the simulation.
    ///
    /// Only the fluid cells are summed; the boundary layer holds copies of its
    /// neighbours and is not mass in the simulation.
    pub fn get_total_mass(&self) -> f64 {
        let mut total = 0.0;
        for i in 1..self.width-1 {
            for j in 1..self.height-1 {
                for k in 1..self.depth-1 {
                    total += self.density[self.get_index(i, j, k)];
                }
            }
        }
        total
    }

    /// Calculates the average velocity magnitude in the simulation.
    pub fn get_average_velocity(&self) -> f64 {
        let size = self.width * self.height * self.depth;
        // x fastest, so the sum is taken in the same order as it was before the
        // storage changed to z-fastest and the result is unchanged to the bit.
        let total_velocity: f64 = (0..self.depth)
            .flat_map(|z| (0..self.height).flat_map(move |y| (0..self.width).map(move |x| (x, y, z))))
            .map(|(x, y, z)| {
                let i = self.get_index(x, y, z);
                (self.velocity_x[i].powi(2) + self.velocity_y[i].powi(2) + self.velocity_z[i].powi(2)).sqrt()
            })
            .sum();
        total_velocity / size as f64
    }

    /// Calculates the kinetic energy of the fluid: `0.5 * density * |v|²` summed
    /// over the fluid cells (not the boundary layer).
    pub fn get_kinetic_energy(&self) -> f64 {
        let mut energy = 0.0;
        for i in 1..self.width-1 {
            for j in 1..self.height-1 {
                for k in 1..self.depth-1 {
                    let idx = self.get_index(i, j, k);
                    energy += 0.5 * self.density[idx] *
                        (self.velocity_x[idx].powi(2) + self.velocity_y[idx].powi(2) + self.velocity_z[idx].powi(2));
                }
            }
        }
        energy
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
