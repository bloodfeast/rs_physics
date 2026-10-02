// src/fluid_simulation.rs

use crate::utils::PhysicsError;
use super::validation::{validate_dimensions_2d, validate_finite, validate_non_negative, validate_positive, validate_position_2d};
use super::solver::{
    pcg_solve, AdvectionScheme, BoundaryType, PcgWorkspace, PressureSolver, SolverConfig, SolverType,
    VorticityConfinement,
};
use std::vec::Vec;

/// MIC(0) parameters from Bridson, *Fluid Simulation for Computer Graphics* (2nd ed.),
/// §5.4: the modification weight, and the fraction of the diagonal below which a
/// pivot is replaced by the diagonal. A closed box's pressure matrix is singular, so
/// full modification (τ = 1) would drive the last pivot to zero; at τ = 0.97 the
/// smallest pivot is 0.59 in 2D and 1.15 in 3D, the same from 34² to 258² and 18³
/// to 66³, and the σ guard never fires. It stays as Bridson's safety net.
pub(crate) const MIC_TAU: f64 = 0.97;
pub(crate) const MIC_SIGMA: f64 = 0.25;

/// Buffers the pressure and diffusion solves reuse.
#[derive(Default)]
pub(crate) struct SolveScratch {
    /// Jacobi's second buffer.
    pub(crate) jacobi: Vec<f64>,
    /// The projection's right-hand side, zero on the boundary ring.
    pub(crate) rhs: Vec<f64>,
    pub(crate) pcg: PcgWorkspace,
    /// [`PressureSolver::Relaxation`]'s pressure, zeroed before each solve.
    pub(crate) relax_pressure: Vec<f64>,
    /// [`PressureSolver::Relaxation`]'s divergence.
    pub(crate) relax_divergence: Vec<f64>,
}

/// Buffers `step` reuses, so a step allocates nothing once the first has sized them.
/// The option buffers are sized the first time their option runs, so a grid that
/// never turns an option on never pays for its memory.
#[derive(Default)]
struct Workspace {
    solve: SolveScratch,
    /// The velocity and density as diffused, before advection.
    velocity_x0: Vec<f64>,
    velocity_y0: Vec<f64>,
    density0: Vec<f64>,
    /// [`AdvectionScheme::MacCormack`]: the backward pass, and the range of the source
    /// cells the forward pass read for each cell.
    back: Vec<f64>,
    low: Vec<f64>,
    high: Vec<f64>,
    /// [`VorticityConfinement::MatchNumericalDissipation`]: the curl, and the
    /// confinement velocity before it is capped.
    omega: Vec<f64>,
    confine: [Vec<f64>; 2],
}

/// Sizes `v` to `len`, allocating only when the length changes. The contents are
/// left as they were (zero when newly allocated); every caller overwrites the cells
/// it reads.
pub(crate) fn fit(v: &mut Vec<f64>, len: usize) {
    if v.len() != len {
        v.clear();
        v.resize(len, 0.0);
    }
}

/// The largest `s` in `[0, 1]` for which adding `s * du` to a velocity field adds at
/// most `loss` kinetic energy, where adding all of it would add
/// `s * along + s^2 * spread`: `along` is the sum of `u . du` and `spread` the sum of
/// `|du|^2 / 2`, both over the fluid cells. Zero when there is no loss to put back.
pub(crate) fn confinement_scale(loss: f64, along: f64, spread: f64) -> f64 {
    if !(loss > 0.0) {
        return 0.0;
    }
    if !(spread > 0.0) {
        return if along > loss { loss / along } else { 1.0 };
    }
    // The positive root of spread s^2 + along s - loss = 0, which exists since
    // loss > 0 and spread > 0.
    let root = (-along + (along * along + 4.0 * spread * loss).sqrt()) / (2.0 * spread);
    root.clamp(0.0, 1.0)
}

/// Half the sum of `|u|^2` over a grid's fluid cells: kinetic energy per unit density
/// and cell volume, in the grid's units.
pub(crate) fn fluid_energy(fields: &[&[f64]], cells: impl Iterator<Item = usize>) -> f64 {
    let mut energy = 0.0;
    for idx in cells {
        for f in fields {
            energy += 0.5 * f[idx] * f[idx];
        }
    }
    energy
}

/// The fractional part of a step's displacement, in cells: the offset `a` at which
/// semi-Lagrangian advection interpolates, and so its numerical viscosity
/// `h^2 a(1 - a) / (2 dt)` (see [`VorticityConfinement`]).
#[inline]
pub(crate) fn cell_offset(displacement_cells: f64) -> f64 {
    let d = displacement_cells.abs();
    d - d.floor()
}

/// A 2D grid-based fluid simulation using the Eulerian method.
///
/// This struct implements a stable fluid solver based on Jos Stam's method,
/// which provides unconditionally stable fluid simulation. The simulation
/// handles density diffusion, velocity diffusion, and advection in a 2D grid.
///
/// # Units
/// Length is measured in *domain widths*: every cell is `1 / width` on a side, on
/// both axes, so a grid taller than it is wide is more than one unit tall.
/// Velocities are in widths per second (a step moves a quantity `dt * width * v`
/// cells), and `viscosity` and `diffusion` are in widths² per second. To work in
/// metres for a domain `L` metres wide, divide velocities by `L` and viscosity and
/// diffusion by `L²` on the way in.
///
/// The outermost ring of cells is a boundary layer that each step overwrites;
/// the fluid is the `(width - 2) × (height - 2)` cells inside it, `x` in
/// `1..width-1` and `y` in `1..height-1`. [`FluidGrid::add_density`] and
/// [`FluidGrid::add_velocity`] refuse the ring; the getters read it.
///
/// # Walls
/// Every wall stops the flow through it. Along it, walls are free-slip by default
/// and no-slip with [`WallCondition::NoSlip`](super::WallCondition) in the
/// [`SolverConfig`].
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
/// * `solver_config` - Configuration for the iterative solver
pub struct FluidGrid {
    width: usize,
    height: usize,
    density: Vec<f64>,
    velocity_x: Vec<f64>,
    velocity_y: Vec<f64>,
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

impl FluidGrid {
    /// Creates a new fluid simulation grid with the specified dimensions and properties.
    ///
    /// # Arguments
    /// * `width` - The width of the simulation grid, in cells (at least 3)
    /// * `height` - The height of the simulation grid, in cells (at least 3)
    /// * `diffusion` - The rate of diffusion, widths²/s (must be finite and non-negative)
    /// * `viscosity` - The kinematic viscosity, widths²/s (must be finite and non-negative)
    /// * `dt` - The time step for the simulation, seconds (must be finite and positive)
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
        Self::with_solver(width, height, diffusion, viscosity, dt, SolverConfig::default())
    }

    /// Creates a new fluid simulation grid with custom solver configuration.
    ///
    /// # Arguments
    /// * `width` - The width of the simulation grid, in cells (at least 3)
    /// * `height` - The height of the simulation grid, in cells (at least 3)
    /// * `diffusion` - The rate of diffusion, widths²/s (must be finite and non-negative)
    /// * `viscosity` - The kinematic viscosity, widths²/s (must be finite and non-negative)
    /// * `dt` - The time step for the simulation, seconds (must be finite and positive)
    /// * `solver_config` - Configuration for the iterative solver; fewer than 1
    ///   iteration is raised to 1, as in [`FluidGrid::set_solver_iterations`]
    ///
    /// # Returns
    /// * `Ok(FluidGrid)` - A new fluid simulation grid if all parameters are valid
    /// * `Err(PhysicsError)` - If any parameters are invalid, including a
    ///   `solver_config` that [`SolverConfig::validate`] rejects
    ///
    /// # Examples
    /// ```
    /// use rs_physics::fluid_dynamics::{FluidGrid, SolverConfig};
    ///
    /// // Create a high-quality simulation with more solver iterations
    /// let config = SolverConfig::high_quality();
    /// let fluid = FluidGrid::with_solver(100, 100, 0.1, 0.001, 0.016, config).unwrap();
    /// ```
    pub fn with_solver(
        width: usize,
        height: usize,
        diffusion: f64,
        viscosity: f64,
        dt: f64,
        solver_config: SolverConfig,
    ) -> Result<Self, PhysicsError> {
        validate_dimensions_2d(width, height)?;
        // The boundary ring takes one cell on each side, and `set_boundaries` indexes
        // `width - 2`: a grid needs at least one fluid cell on each axis.
        if width < 3 || height < 3 {
            return Err(PhysicsError::InvalidArea);
        }
        Self::validate_diffusion(diffusion)?;
        Self::validate_viscosity(viscosity)?;
        Self::validate_dt(dt)?;
        let solver_config = solver_config.checked()?;

        let size = width * height;
        let mut grid = Self {
            width,
            height,
            density: vec![0.0; size],
            velocity_x: vec![0.0; size],
            velocity_y: vec![0.0; size],
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

    /// Refuses a position outside the fluid: off the grid, or on the boundary ring,
    /// which every step overwrites.
    fn validate_fluid_cell(&self, x: usize, y: usize) -> Result<(), PhysicsError> {
        validate_position_2d(x, y, self.width, self.height)?;
        if x == 0 || y == 0 || x == self.width - 1 || y == self.height - 1 {
            return Err(PhysicsError::CalculationError(format!(
                "({x}, {y}) is on the boundary ring, which every step overwrites; \
                 the fluid is x in 1..={}, y in 1..={}",
                self.width - 2,
                self.height - 2
            )));
        }
        Ok(())
    }

    /// Adds density to the fluid at a specific grid position.
    ///
    /// # Arguments
    /// * `x` - The x-coordinate in the grid, a fluid cell: `1..width-1`
    /// * `y` - The y-coordinate in the grid, a fluid cell: `1..height-1`
    /// * `amount` - The amount of density to add (can be negative to remove density)
    ///
    /// # Returns
    /// * `Ok(())` - If the density was successfully added
    /// * `Err(PhysicsError)` - If the position is out of bounds or on the boundary
    ///   ring (which the next step would overwrite), or `amount` is not finite
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
    /// // Attempting to add density outside the fluid returns an error: off the
    /// // grid, or on the boundary ring
    /// assert!(fluid.add_density(100, 50, 1.0).is_err());
    /// assert!(fluid.add_density(0, 50, 1.0).is_err());
    /// ```
    pub fn add_density(&mut self, x: usize, y: usize, amount: f64) -> Result<(), PhysicsError> {
        self.validate_fluid_cell(x, y)?;
        // One non-finite cell reaches every cell within a few steps through the
        // pressure solve.
        validate_finite(amount, "amount")?;
        let idx = self.get_index(x, y);
        self.density[idx] += amount;
        Ok(())
    }

    /// Adds velocity to the fluid at a specific grid position.
    ///
    /// # Arguments
    /// * `x` - The x-coordinate in the grid, a fluid cell: `1..width-1`
    /// * `y` - The y-coordinate in the grid, a fluid cell: `1..height-1`
    /// * `amount_x` - The amount of velocity to add in the x direction
    /// * `amount_y` - The amount of velocity to add in the y direction
    ///
    /// Velocities are in domain widths per second (see [`FluidGrid`]'s units).
    ///
    /// # Returns
    /// * `Ok(())` - If the velocity was successfully added
    /// * `Err(PhysicsError)` - If the position is out of bounds or on the boundary
    ///   ring, or a component is not finite
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
        self.validate_fluid_cell(x, y)?;
        if !amount_x.is_finite() || !amount_y.is_finite() {
            return Err(PhysicsError::InvalidVelocity);
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
    /// 2. Mass conservation (projection) - Makes the velocity divergence-free
    /// 3. Velocity advection - Moves velocity with the flow
    /// 4. Mass conservation (projection) - Makes the velocity divergence-free again
    /// 5. Density diffusion - Simulates spreading of density
    /// 6. Density advection - Moves density with the flow
    ///
    /// With the default [`PressureSolver::ConjugateGradient`] each projection solves
    /// the pressure to the configured relative tolerance (or iteration cap), so the
    /// cost of a step depends on how much new divergence it brings; see
    /// [`FluidGrid::get_last_pressure_iterations`]. The diffusion solves run
    /// `iterations` sweeps of the configured [`SolverType`].
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
        let mut ws = std::mem::take(&mut self.workspace);
        let [mut pressure, mut pressure_after_advection] = std::mem::take(&mut self.pressure);
        // The state is taken out for the step and each buffer reused in place, so a
        // step allocates nothing. Every buffer a pass writes is overwritten whole (its
        // fluid cells by the pass, its ring by `set_boundaries`), so the results are
        // bit-identical to the allocating version this replaced.
        let mut velocity_x = std::mem::take(&mut self.velocity_x);
        let mut velocity_y = std::mem::take(&mut self.velocity_y);
        let mut density = std::mem::take(&mut self.density);
        let size = self.width * self.height;
        fit(&mut ws.velocity_x0, size);
        fit(&mut ws.velocity_y0, size);
        fit(&mut ws.density0, size);

        // Clone the current state
        ws.velocity_x0.copy_from_slice(&velocity_x);
        ws.velocity_y0.copy_from_slice(&velocity_y);
        ws.density0.copy_from_slice(&density);

        // Implicit diffusion: `a = dt * nu / h²`, with the cell size `h = 1 / width` on
        // both axes (the same `h` that `advect` and `project` use). `width * height`
        // is only equal to `1 / h²` on a square grid.
        let inv_h_sq = (self.width * self.width) as f64;

        // Diffuse velocity
        {
            let a = self.dt * self.viscosity * inv_h_sq;
            self.lin_solve(BoundaryType::VelocityX, &mut ws.velocity_x0, &velocity_x, a, 1.0 + 4.0 * a, &mut ws.solve.jacobi);
            self.lin_solve(BoundaryType::VelocityY, &mut ws.velocity_y0, &velocity_y, a, 1.0 + 4.0 * a, &mut ws.solve.jacobi);
        }

        // Project velocity
        let mut pressure_iterations =
            self.project(&mut ws.velocity_x0, &mut ws.velocity_y0, &mut pressure, &mut ws.solve);

        // Advect velocity, into the buffers the old velocity held
        match self.solver_config.advection {
            AdvectionScheme::SemiLagrangian => {
                self.advect(BoundaryType::VelocityX, &mut velocity_x, &ws.velocity_x0, &ws.velocity_x0, &ws.velocity_y0);
                self.advect(BoundaryType::VelocityY, &mut velocity_y, &ws.velocity_y0, &ws.velocity_x0, &ws.velocity_y0);
            }
            AdvectionScheme::MacCormack => {
                let Workspace { velocity_x0, velocity_y0, back, low, high, .. } = &mut ws;
                for (bt, out, source) in [
                    (BoundaryType::VelocityX, &mut velocity_x, &*velocity_x0),
                    (BoundaryType::VelocityY, &mut velocity_y, &*velocity_y0),
                ] {
                    self.advect_maccormack(bt, out, source, velocity_x0, velocity_y0, back, low, high);
                }
            }
        }

        if self.solver_config.vorticity_confinement == VorticityConfinement::MatchNumericalDissipation {
            self.confine_vorticity(
                &mut velocity_x,
                &mut velocity_y,
                &ws.velocity_x0,
                &ws.velocity_y0,
                &mut ws.omega,
                &mut ws.confine,
            );
        }

        // Project again
        pressure_iterations +=
            self.project(&mut velocity_x, &mut velocity_y, &mut pressure_after_advection, &mut ws.solve);

        // Diffuse density
        {
            let a = self.dt * self.diffusion * inv_h_sq;
            self.lin_solve(BoundaryType::Density, &mut ws.density0, &density, a, 1.0 + 4.0 * a, &mut ws.solve.jacobi);
        }

        // Advect density
        match self.solver_config.advection {
            AdvectionScheme::SemiLagrangian => {
                self.advect(BoundaryType::Density, &mut density, &ws.density0, &velocity_x, &velocity_y);
            }
            AdvectionScheme::MacCormack => {
                let Workspace { density0, back, low, high, .. } = &mut ws;
                self.advect_maccormack(BoundaryType::Density, &mut density, density0, &velocity_x, &velocity_y, back, low, high);
            }
        }

        self.velocity_x = velocity_x;
        self.velocity_y = velocity_y;
        self.density = density;
        self.pressure = [pressure, pressure_after_advection];
        self.workspace = ws;
        self.last_pressure_iterations = pressure_iterations;
    }

    /// Gets the number of pressure-solve iterations the last [`FluidGrid::step`] ran,
    /// summed over its two projections: conjugate-gradient iterations, or relaxation
    /// sweeps under [`PressureSolver::Relaxation`].
    ///
    /// The conjugate-gradient count is the step's variable cost. Each projection
    /// stops at [`SolverConfig::pressure_max_iterations`], so a count of twice that
    /// means the tolerance was not reached.
    pub fn get_last_pressure_iterations(&self) -> usize {
        self.last_pressure_iterations
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
        validate_position_2d(x, y, self.width, self.height)?;
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
        validate_position_2d(x, y, self.width, self.height)?;
        let idx = self.get_index(x, y);
        Ok((self.velocity_x[idx], self.velocity_y[idx]))
    }

    /// Converts 2D coordinates to a 1D array index.
    ///
    /// Column-major: `y` is the fastest-varying index because every sweep in this
    /// file loops `x` outside `y`. Row-major storage made the inner loop stride by a
    /// whole row, which cost 18% of a step at 256² (nothing once the grid fits in
    /// cache). Swapping the storage rather than the loops keeps the sweep order, and
    /// so every result, bit-identical.
    #[inline]
    fn get_index(&self, x: usize, y: usize) -> usize {
        x * self.height + y
    }

    /// Projects the velocity field to make it mass-conserving.
    ///
    /// This method enforces incompressibility in the fluid by calculating and
    /// subtracting the pressure gradient. It follows the Helmholtz-Hodge
    /// decomposition to project the velocity field onto a divergence-free field.
    ///
    /// The pressure solves `4p - Σ neighbours = -h/2 · (central divergence)`, with a
    /// wall neighbour standing in as a copy of the cell itself (zero normal
    /// gradient). Both solvers converge to the same answer; see [`PressureSolver`].
    ///
    /// # Arguments
    /// * `velocity_x` - The x-component of the velocity field to be projected
    /// * `velocity_y` - The y-component of the velocity field to be projected
    /// * `pressure` - The conjugate gradient's starting guess, left holding its answer
    ///
    /// # Returns
    /// The iterations the pressure solve ran.
    fn project(&self, velocity_x: &mut Vec<f64>, velocity_y: &mut Vec<f64>, pressure: &mut Vec<f64>, ws: &mut SolveScratch) -> usize {
        match self.solver_config.pressure_solver {
            PressureSolver::Relaxation => {
                // Exactly the pre-conjugate-gradient projection: sweeps from zero. The
                // divergence's fluid cells and ring are both written before they are
                // read, so only the pressure needs clearing.
                let size = self.width * self.height;
                fit(&mut ws.relax_pressure, size);
                fit(&mut ws.relax_divergence, size);
                let (p, div) = (&mut ws.relax_pressure, &mut ws.relax_divergence);
                p.fill(0.0);
                self.divergence(velocity_x, velocity_y, div);
                self.set_boundaries(BoundaryType::Density, div);
                self.set_boundaries(BoundaryType::Density, p);
                self.lin_solve(BoundaryType::Density, p, div, 1.0, 4.0, &mut ws.jacobi);
                self.subtract_pressure_gradient(velocity_x, velocity_y, p);
                self.solver_config.iterations
            }
            PressureSolver::ConjugateGradient => {
                let size = self.width * self.height;
                if ws.rhs.len() != size {
                    // Zero-filled once: the ring is never written and must stay zero.
                    ws.rhs = vec![0.0; size];
                }
                self.divergence(velocity_x, velocity_y, &mut ws.rhs);
                // A closed box's pressure matrix is singular, with the constants as its
                // null space: the right-hand side must have zero mean to be solvable.
                // It does in exact arithmetic (the walls' negated normal velocity
                // telescopes the sum away); this removes the rounding.
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
                // Only the gradient matters; keep the warm start from drifting.
                let mean = self.fluid_mean(pressure);
                self.for_each_fluid_cell(|idx| pressure[idx] -= mean);
                self.set_boundaries(BoundaryType::Density, pressure);
                self.subtract_pressure_gradient(velocity_x, velocity_y, pressure);
                iterations
            }
        }
    }

    /// Writes `-h/2 · (central-difference divergence)` into the fluid cells of `div`.
    fn divergence(&self, velocity_x: &[f64], velocity_y: &[f64], div: &mut [f64]) {
        for i in 1..self.width-1 {
            for j in 1..self.height-1 {
                let idx = self.get_index(i, j);
                div[idx] = -0.5 * (
                    velocity_x[self.get_index(i+1, j)] -
                        velocity_x[self.get_index(i-1, j)] +
                        velocity_y[self.get_index(i, j+1)] -
                        velocity_y[self.get_index(i, j-1)]
                ) / self.width as f64;
            }
        }
    }

    fn subtract_pressure_gradient(&self, velocity_x: &mut Vec<f64>, velocity_y: &mut Vec<f64>, p: &[f64]) {
        // Both components divide by the same `h = 1 / width` the divergence
        // multiplied by; scaling y by `height` instead left a non-square grid's
        // projection off by `height / width`.
        for i in 1..self.width-1 {
            for j in 1..self.height-1 {
                let idx = self.get_index(i, j);
                velocity_x[idx] -= 0.5 * (p[self.get_index(i+1, j)] - p[self.get_index(i-1, j)]) * self.width as f64;
                velocity_y[idx] -= 0.5 * (p[self.get_index(i, j+1)] - p[self.get_index(i, j-1)]) * self.width as f64;
            }
        }

        self.set_boundaries(BoundaryType::VelocityX, velocity_x);
        self.set_boundaries(BoundaryType::VelocityY, velocity_y);
    }

    fn for_each_fluid_cell(&self, mut f: impl FnMut(usize)) {
        for i in 1..self.width-1 {
            for j in 1..self.height-1 {
                f(self.get_index(i, j));
            }
        }
    }

    fn fluid_mean(&self, field: &[f64]) -> f64 {
        let mut sum = 0.0;
        self.for_each_fluid_cell(|idx| sum += field[idx]);
        sum / ((self.width - 2) * (self.height - 2)) as f64
    }

    /// `out = A·x` for the pressure matrix, on the fluid cells only: `4x` minus the
    /// four neighbours, a wall neighbour standing in as a copy of the cell. This is
    /// the system the relaxation solve's fixed point satisfies with the ghost-copy
    /// boundary, so both pressure solvers aim at the same answer.
    fn apply_pressure_matrix(&self, x: &[f64], out: &mut [f64]) {
        let (w, h) = (self.width, self.height);
        for i in 1..w-1 {
            let column = self.get_index(i, 0);
            // At a wall column the missing neighbour is the cell itself: read the
            // column's own values, decided once per column rather than per cell.
            let left = if i > 1 { column - h } else { column };
            let right = if i < w - 2 { column + h } else { column };
            for j in 1..h-1 {
                let idx = column + j;
                let c = x[idx];
                let down = if j > 1 { x[idx - 1] } else { c };
                let up = if j < h - 2 { x[idx + 1] } else { c };
                out[idx] = 4.0 * c - (x[left + j] + x[right + j] + down + up);
            }
        }
    }

    /// Builds the MIC(0) preconditioner (Bridson §5.4) for the pressure matrix.
    /// The matrix depends only on the grid's shape, so this runs once.
    ///
    /// Bridson stores `1/√e` per cell; this stores the inverse pivot `1/e`, which is
    /// all the application needs once it carries `y = q/√e` instead of `q` (see
    /// [`FluidGrid::apply_preconditioner`]). Same preconditioner, no square roots.
    fn build_preconditioner(&mut self) {
        let (w, h) = (self.width, self.height);
        let mut inv = std::mem::take(&mut self.precon);
        for i in 1..w-1 {
            for j in 1..h-1 {
                let idx = self.get_index(i, j);
                let diag = [i > 1, i < w - 2, j > 1, j < h - 2].iter().filter(|&&n| n).count() as f64;
                let mut e = diag;
                if i > 1 {
                    // The (i-1, j) row: its link to (i, j), and to (i-1, j+1).
                    let q = inv[idx - h];
                    e -= q;
                    if j < h - 2 {
                        e -= MIC_TAU * q;
                    }
                }
                if j > 1 {
                    let q = inv[idx - 1];
                    e -= q;
                    if i < w - 2 {
                        e -= MIC_TAU * q;
                    }
                }
                if e < MIC_SIGMA * diag {
                    e = diag;
                }
                inv[idx] = if e > 0.0 { 1.0 / e } else { 0.0 };
            }
        }
        self.precon = inv;
    }

    /// `z = M⁻¹·r` for the MIC(0) preconditioner `M = L·Lᵀ`: a forward and a backward
    /// triangular solve, in place in `z`, fluid cells only.
    ///
    /// Written in terms of the inverse pivots `1/e`: the forward pass leaves
    /// `y = q/√e` in `z` (where `q` solves `L·q = r`), and the backward pass turns it
    /// into the answer. Each cell's dependence on the cell before it is then one add
    /// and one multiply, which is the latency these serial sweeps are bound by.
    fn apply_preconditioner(&self, r: &[f64], z: &mut [f64]) {
        let (w, h) = (self.width, self.height);
        let inv = &self.precon;
        for i in 1..w-1 {
            for j in 1..h-1 {
                let idx = self.get_index(i, j);
                let mut t = r[idx];
                if i > 1 {
                    t += z[idx - h];
                }
                if j > 1 {
                    t += z[idx - 1];
                }
                z[idx] = inv[idx] * t;
            }
        }
        for i in (1..w-1).rev() {
            for j in (1..h-1).rev() {
                let idx = self.get_index(i, j);
                let mut t = z[idx];
                if i < w - 2 {
                    t += inv[idx] * z[idx + h];
                }
                if j < h - 2 {
                    t += inv[idx] * z[idx + 1];
                }
                z[idx] = t;
            }
        }
    }

    /// Sets the boundary conditions for the fluid simulation.
    ///
    /// # Arguments
    /// * `boundary_type` - The type of boundary condition to apply:
    ///   * `BoundaryType::Density` - no-flux condition (values continuous at boundary)
    ///   * `BoundaryType::VelocityX` - negated at the left/right walls (no flow through
    ///     them); at the top/bottom walls, copied (free-slip) or negated (no-slip)
    ///   * `BoundaryType::VelocityY` - negated at the top/bottom walls; at the
    ///     left/right walls, copied (free-slip) or negated (no-slip)
    /// * `x` - The field to apply boundary conditions to
    ///
    /// # Note
    /// The boundary conditions ensure that:
    /// - Fluid cannot flow through walls (the normal component is negated, so it
    ///   is zero half-way between ghost and fluid cell, at the wall)
    /// - Fluid slides along walls freely, or not at all, per [`SolverConfig::wall`]
    /// - Density is conserved at boundaries (no-flux condition)
    /// - Corner values are properly interpolated
    fn set_boundaries(&self, boundary_type: BoundaryType, x: &mut Vec<f64>) {
        let wall = self.solver_config.wall;
        let flip_top_bottom = boundary_type.flips_at_wall(BoundaryType::VelocityY, wall);
        let flip_left_right = boundary_type.flips_at_wall(BoundaryType::VelocityX, wall);

        // Top and bottom boundaries
        for i in 1..self.width-1 {
            x[self.get_index(i, 0)] = if flip_top_bottom {
                -x[self.get_index(i, 1)]
            } else {
                x[self.get_index(i, 1)]
            };
            x[self.get_index(i, self.height-1)] = if flip_top_bottom {
                -x[self.get_index(i, self.height-2)]
            } else {
                x[self.get_index(i, self.height-2)]
            };
        }

        // Left and right boundaries
        for j in 1..self.height-1 {
            x[self.get_index(0, j)] = if flip_left_right {
                -x[self.get_index(1, j)]
            } else {
                x[self.get_index(1, j)]
            };
            x[self.get_index(self.width-1, j)] = if flip_left_right {
                -x[self.get_index(self.width-2, j)]
            } else {
                x[self.get_index(self.width-2, j)]
            };
        }

        // Corner interpolation
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

    /// Solves a linear system by relaxation: Gauss-Seidel, SOR or Jacobi, per
    /// [`SolverConfig::solver_type`].
    ///
    /// # Arguments
    /// * `boundary_type` - The boundary condition type to apply after each iteration
    /// * `x` - The field to solve for, holding the starting guess
    /// * `x0` - The source field
    /// * `a` - The diffusion/viscosity rate multiplied by dt
    /// * `c` - The center cell coefficient (1 + 4a)
    /// * `scratch` - Jacobi's second buffer; unused by the other two
    ///
    /// # Note
    /// This method performs iterative relaxation to solve the diffusion equation:
    /// x = (x0 + a * (left + right + top + bottom)) / c
    /// The number of iterations is controlled by `solver_config.iterations`. Gauss-Seidel
    /// uses each new value as soon as it is computed; SOR moves each Gauss-Seidel update
    /// `relaxation` times as far; Jacobi computes a whole sweep from the previous one.
    fn lin_solve(&self, boundary_type: BoundaryType, x: &mut Vec<f64>, x0: &[f64], a: f64, c: f64, scratch: &mut Vec<f64>) {
        // With no coupling every Gauss-Seidel or Jacobi sweep computes `x0 / c` (the
        // neighbours are multiplied by zero), so the sweeps are skipped and the answer
        // written once: the same values, except that `-0 + 0 * neighbours` was `+0`
        // and is now `-0`. An inviscid grid's two diffusion solves were about a
        // quarter of its step. SOR is left alone: its update mixes in the old `x`.
        if a == 0.0 && matches!(self.solver_config.solver_type, SolverType::GaussSeidel | SolverType::Jacobi) {
            for i in 1..self.width-1 {
                for j in 1..self.height-1 {
                    let idx = self.get_index(i, j);
                    x[idx] = x0[idx] / c;
                }
            }
            self.set_boundaries(boundary_type, x);
            return;
        }
        match self.solver_config.solver_type {
            SolverType::GaussSeidel => {
                for _ in 0..self.solver_config.iterations {
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
                    self.set_boundaries(boundary_type, x);
                }
            }
            SolverType::SOR => {
                let omega = self.solver_config.relaxation;
                for _ in 0..self.solver_config.iterations {
                    for i in 1..self.width-1 {
                        for j in 1..self.height-1 {
                            let idx = self.get_index(i, j);
                            let gauss_seidel = (x0[idx] + a * (
                                x[self.get_index(i+1, j)] +
                                    x[self.get_index(i-1, j)] +
                                    x[self.get_index(i, j+1)] +
                                    x[self.get_index(i, j-1)]
                            )) / c;
                            x[idx] = (1.0 - omega) * x[idx] + omega * gauss_seidel;
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
                            let idx = self.get_index(i, j);
                            scratch[idx] = (x0[idx] + a * (
                                x[self.get_index(i+1, j)] +
                                    x[self.get_index(i-1, j)] +
                                    x[self.get_index(i, j+1)] +
                                    x[self.get_index(i, j-1)]
                            )) / c;
                        }
                    }
                    // The new sweep becomes `x`. Its ring is stale, and
                    // `set_boundaries` rewrites every ring cell from the fluid cells.
                    std::mem::swap(x, scratch);
                    self.set_boundaries(boundary_type, x);
                }
            }
        }
    }

    /// Performs semi-Lagrangian advection of a quantity through the velocity field.
    ///
    /// # Arguments
    /// * `boundary_type` - The boundary condition type to apply after advection
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
    fn advect(&self, boundary_type: BoundaryType, d: &mut Vec<f64>, d0: &Vec<f64>, velocity_x: &Vec<f64>, velocity_y: &Vec<f64>) {
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

        self.set_boundaries(boundary_type, d);
    }

    /// One semi-Lagrangian pass, along `+velocity` when `sign` is `-1` (the usual
    /// back-trace) or `-velocity` when it is `+1` (MacCormack's reverse step), writing
    /// the fluid cells of `d`. When `bounds` is given it also records, per cell, the
    /// smallest and largest of the four source cells the interpolation read.
    ///
    /// The arithmetic of the `sign = -1` pass is [`FluidGrid::advect`]'s, operation
    /// for operation, so MacCormack's forward value is the first-order value exactly.
    #[allow(clippy::too_many_arguments)]
    fn trace_pass(
        &self,
        d: &mut [f64],
        d0: &[f64],
        velocity_x: &[f64],
        velocity_y: &[f64],
        sign: f64,
        mut bounds: Option<(&mut [f64], &mut [f64])>,
    ) {
        let dt0 = self.dt * self.width as f64;
        for i in 1..self.width-1 {
            for j in 1..self.height-1 {
                let idx = self.get_index(i, j);
                let x = (i as f64 + sign * dt0 * velocity_x[idx]).clamp(0.5, self.width as f64 - 1.5);
                let y = (j as f64 + sign * dt0 * velocity_y[idx]).clamp(0.5, self.height as f64 - 1.5);

                let i0 = x.floor() as usize;
                let i1 = i0 + 1;
                let j0 = y.floor() as usize;
                let j1 = j0 + 1;

                let s1 = x - i0 as f64;
                let s0 = 1.0 - s1;
                let t1 = y - j0 as f64;
                let t0 = 1.0 - t1;

                let (a, b) = (d0[self.get_index(i0, j0)], d0[self.get_index(i0, j1)]);
                let (c, e) = (d0[self.get_index(i1, j0)], d0[self.get_index(i1, j1)]);
                d[idx] = s0 * (t0 * a + t1 * b) + s1 * (t0 * c + t1 * e);
                if let Some((low, high)) = bounds.as_mut() {
                    low[idx] = a.min(b).min(c.min(e));
                    high[idx] = a.max(b).max(c.max(e));
                }
            }
        }
    }

    /// MacCormack advection of `d0` into `d` ([`AdvectionScheme::MacCormack`]): a
    /// forward pass, a reverse pass from its result, half the round trip's error
    /// added back, and each cell clamped to the range of the cells its forward pass
    /// read. `back`, `low` and `high` are scratch.
    #[allow(clippy::too_many_arguments)]
    fn advect_maccormack(
        &self,
        boundary_type: BoundaryType,
        d: &mut Vec<f64>,
        d0: &[f64],
        velocity_x: &[f64],
        velocity_y: &[f64],
        back: &mut Vec<f64>,
        low: &mut Vec<f64>,
        high: &mut Vec<f64>,
    ) {
        let size = self.width * self.height;
        fit(back, size);
        fit(low, size);
        fit(high, size);
        self.trace_pass(d, d0, velocity_x, velocity_y, -1.0, Some((low, high)));
        // The reverse pass interpolates the forward result, ring included.
        self.set_boundaries(boundary_type, d);
        self.trace_pass(back, d, velocity_x, velocity_y, 1.0, None);
        for i in 1..self.width-1 {
            let column = self.get_index(i, 0);
            for idx in column + 1..column + self.height - 1 {
                let corrected = d[idx] + 0.5 * (d0[idx] - back[idx]);
                d[idx] = corrected.clamp(low[idx], high[idx]);
            }
        }
        self.set_boundaries(boundary_type, d);
    }

    /// Adds the vorticity-confinement velocity
    /// ([`VorticityConfinement::MatchNumericalDissipation`]) to the fluid cells of
    /// `velocity`: `h e (N x omega)` with `e = mean over axes of a(1 - a) / 2`, `a` the
    /// fractional cell offset of each cell's departure point this step, from the
    /// velocity it was advected by (`advecting_x`, `advecting_y`), which is also the
    /// velocity before this step's advection. The whole is scaled down, if need be, so
    /// it adds no more kinetic energy than the advection took out (see
    /// [`VorticityConfinement`]). `omega` and `confine` are scratch.
    fn confine_vorticity(
        &self,
        velocity_x: &mut Vec<f64>,
        velocity_y: &mut Vec<f64>,
        advecting_x: &[f64],
        advecting_y: &[f64],
        omega: &mut Vec<f64>,
        confine: &mut [Vec<f64>; 2],
    ) {
        let (w, hgt) = (self.width, self.height);
        fit(omega, w * hgt);
        for v in confine.iter_mut() {
            fit(v, w * hgt);
        }
        let fluid = || (1..w - 1).flat_map(move |i| (i * hgt + 1..i * hgt + hgt - 1));
        let loss = fluid_energy(&[advecting_x, advecting_y], fluid())
            - fluid_energy(&[&velocity_x[..], &velocity_y[..]], fluid());
        let width = w as f64;
        let h = 1.0 / width;
        let inv_2h = 0.5 * width;
        for i in 1..w-1 {
            for j in 1..hgt-1 {
                let idx = self.get_index(i, j);
                omega[idx] = ((velocity_y[idx + hgt] - velocity_y[idx - hgt])
                    - (velocity_x[idx + 1] - velocity_x[idx - 1])) * inv_2h;
            }
        }
        // A wall's ghost copies its neighbour, so |omega| has no gradient through it.
        self.set_boundaries(BoundaryType::Density, omega);

        let cells_per_velocity = self.dt * width;
        let [dx, dy] = confine;
        let (mut along, mut spread) = (0.0, 0.0);
        for i in 1..w-1 {
            for j in 1..hgt-1 {
                let idx = self.get_index(i, j);
                let gx = (omega[idx + hgt].abs() - omega[idx - hgt].abs()) * inv_2h;
                let gy = (omega[idx + 1].abs() - omega[idx - 1].abs()) * inv_2h;
                let length = (gx * gx + gy * gy).sqrt();
                if !(length > 0.0) {
                    dx[idx] = 0.0;
                    dy[idx] = 0.0;
                    continue;
                }
                let ax = cell_offset(advecting_x[idx] * cells_per_velocity);
                let ay = cell_offset(advecting_y[idx] * cells_per_velocity);
                let e = 0.25 * (ax * (1.0 - ax) + ay * (1.0 - ay));
                // N x (omega z) = (N_y omega, -N_x omega).
                let s = h * e * omega[idx] / length;
                let (cx, cy) = (s * gy, -s * gx);
                dx[idx] = cx;
                dy[idx] = cy;
                along += velocity_x[idx] * cx + velocity_y[idx] * cy;
                spread += 0.5 * (cx * cx + cy * cy);
            }
        }
        let scale = confinement_scale(loss, along, spread);
        if scale > 0.0 {
            for idx in fluid() {
                velocity_x[idx] += scale * dx[idx];
                velocity_y[idx] += scale * dy[idx];
            }
        }
        self.set_boundaries(BoundaryType::VelocityX, velocity_x);
        self.set_boundaries(BoundaryType::VelocityY, velocity_y);
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

    /// Gets the current solver configuration.
    ///
    /// # Returns
    /// A copy of the solver configuration.
    pub fn get_solver_config(&self) -> SolverConfig {
        self.solver_config
    }

    /// Gets the number of solver iterations.
    ///
    /// # Returns
    /// The number of iterations used in the linear solver.
    pub fn get_solver_iterations(&self) -> usize {
        self.solver_config.iterations
    }

    /// Sets the number of solver iterations.
    ///
    /// # Arguments
    /// * `iterations` - The new number of iterations (must be at least 1)
    ///
    /// # Note
    /// More iterations = more accurate but slower simulation.
    /// - 2-4: Fast, less accurate (good for real-time)
    /// - 10-20: High quality (good for offline rendering)
    pub fn set_solver_iterations(&mut self, iterations: usize) {
        self.solver_config.iterations = iterations.max(1);
    }

    /// Sets the solver configuration.
    ///
    /// The conjugate gradient's warm start is kept: the pressure it holds is a
    /// property of the flow, not of the solver.
    ///
    /// # Arguments
    /// * `config` - The new solver configuration; fewer than 1 iteration is raised
    ///   to 1, as in [`FluidGrid::set_solver_iterations`]
    ///
    /// # Returns
    /// * `Ok(())` if the configuration was set
    /// * `Err(PhysicsError)` if [`SolverConfig::validate`] rejects it; the grid keeps
    ///   its previous configuration
    pub fn set_solver_config(&mut self, config: SolverConfig) -> Result<(), PhysicsError> {
        self.solver_config = config.checked()?;
        Ok(())
    }

    /// Sets the diffusion rate of the fluid.
    ///
    /// # Arguments
    /// * `diffusion` - The new diffusion coefficient, widths²/s (must be finite and non-negative)
    ///
    /// # Returns
    /// * `Ok(())` if the diffusion rate was successfully set
    /// * `Err(PhysicsError)` if the diffusion rate is negative or not finite
    pub fn set_diffusion(&mut self, diffusion: f64) -> Result<(), PhysicsError> {
        Self::validate_diffusion(diffusion)?;
        self.diffusion = diffusion;
        Ok(())
    }

    /// Sets the viscosity of the fluid.
    ///
    /// # Arguments
    /// * `viscosity` - The new kinematic viscosity, widths²/s (must be finite and non-negative)
    ///
    /// # Returns
    /// * `Ok(())` if the viscosity was successfully set
    /// * `Err(PhysicsError)` if the viscosity is negative or not finite
    pub fn set_viscosity(&mut self, viscosity: f64) -> Result<(), PhysicsError> {
        Self::validate_viscosity(viscosity)?;
        self.viscosity = viscosity;
        Ok(())
    }

    /// Sets the time step of the simulation.
    ///
    /// # Arguments
    /// * `dt` - The new time step in seconds (must be finite and positive)
    ///
    /// # Returns
    /// * `Ok(())` if the time step was successfully set
    /// * `Err(PhysicsError)` if the time step is zero, negative or not finite
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
    ///
    /// This method clears all density and velocity fields, setting them to zero.
    pub fn reset(&mut self) {
        let size = self.width * self.height;
        self.density = vec![0.0; size];
        self.velocity_x = vec![0.0; size];
        self.velocity_y = vec![0.0; size];
        self.pressure = [vec![0.0; size], vec![0.0; size]];
        self.last_pressure_iterations = 0;
    }

    /// Calculates the total mass (sum of density) in the simulation.
    ///
    /// Only the fluid cells are summed. The boundary ring holds copies of its
    /// neighbours, so counting it would report mass that is not there (1.5625× the
    /// true total once a blob has spread across a 10×10 grid).
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
        let mut total = 0.0;
        for i in 1..self.width-1 {
            for j in 1..self.height-1 {
                total += self.density[self.get_index(i, j)];
            }
        }
        total
    }

    /// Calculates the average velocity magnitude in the simulation.
    ///
    /// # Returns
    /// The average velocity magnitude across all cells.
    pub fn get_average_velocity(&self) -> f64 {
        let size = self.width * self.height;
        // Row by row, so the sum is taken in the same order as it was before the
        // storage became column-major and the result is unchanged to the bit.
        let total_velocity: f64 = (0..self.height)
            .flat_map(|y| (0..self.width).map(move |x| (x, y)))
            .map(|(x, y)| {
                let i = self.get_index(x, y);
                (self.velocity_x[i].powi(2) + self.velocity_y[i].powi(2)).sqrt()
            })
            .sum();
        total_velocity / size as f64
    }

    /// Calculates the kinetic energy of the fluid.
    ///
    /// Sums `0.5 * density * |v|²` over the fluid cells; the boundary ring is not
    /// fluid (see [`FluidGrid::get_total_mass`]).
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
        let mut energy = 0.0;
        for i in 1..self.width-1 {
            for j in 1..self.height-1 {
                let idx = self.get_index(i, j);
                energy += 0.5 * self.density[idx] *
                    (self.velocity_x[idx].powi(2) + self.velocity_y[idx].powi(2));
            }
        }
        energy
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
