//! 3D Thermal Grid Simulation
//!
//! This module provides a grid-based heat diffusion simulation using the
//! finite difference method to solve the 3D heat equation.
//!
//! The heat equation: ∂T/∂t = α * (∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²)
//! where α is thermal diffusivity.

use crate::utils::PhysicsError;
use super::validation::validate_grid_dimensions_3d;
use super::thermal_grid::ThermalBoundaryCondition;

// ============================================================================
// Grid Face (for 3D boundary conditions)
// ============================================================================

/// Face of a 3D grid for boundary conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridFace {
    /// Negative X face (left)
    Left,
    /// Positive X face (right)
    Right,
    /// Negative Y face (front)
    Front,
    /// Positive Y face (back)
    Back,
    /// Negative Z face (bottom)
    Bottom,
    /// Positive Z face (top)
    Top,
}

// ============================================================================
// Heat Source 3D
// ============================================================================

/// A heat source in the 3D grid
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatSource3D {
    /// X position in grid
    pub x: usize,
    /// Y position in grid
    pub y: usize,
    /// Z position in grid
    pub z: usize,
    /// Power in Watts
    pub power: f64,
}

// ============================================================================
// ThermalGrid3D
// ============================================================================

/// 3D thermal grid for heat diffusion simulation
#[derive(Debug, Clone)]
pub struct ThermalGrid3D {
    /// Grid width (number of cells in x direction)
    width: usize,
    /// Grid height (number of cells in y direction)
    height: usize,
    /// Grid depth (number of cells in z direction)
    depth: usize,
    /// Temperature values at each cell in Kelvin
    temperature: Vec<f64>,
    /// Previous temperature values (for double buffering)
    prev_temperature: Vec<f64>,
    /// Thermal diffusivity α = k/(ρ·c) in m²/s
    thermal_diffusivity: f64,
    /// Time step in seconds
    dt: f64,
    /// Grid spacing in meters
    dx: f64,
    /// Heat sources in the grid
    heat_sources: Vec<HeatSource3D>,
    /// Boundary conditions [left, right, front, back, bottom, top]
    boundary_conditions: [ThermalBoundaryCondition; 6],
}

impl ThermalGrid3D {
    /// Creates a new 3D thermal grid
    ///
    /// # Arguments
    /// * `width` - Number of cells in x direction
    /// * `height` - Number of cells in y direction
    /// * `depth` - Number of cells in z direction
    /// * `initial_temp` - Initial temperature for all cells in K
    /// * `thermal_diffusivity` - Thermal diffusivity in m²/s
    /// * `dt` - Time step in seconds
    /// * `dx` - Grid spacing in meters
    ///
    /// # Returns
    /// New ThermalGrid3D or error if parameters are invalid
    pub fn new(
        width: usize,
        height: usize,
        depth: usize,
        initial_temp: f64,
        thermal_diffusivity: f64,
        dt: f64,
        dx: f64,
    ) -> Result<Self, PhysicsError> {
        validate_grid_dimensions_3d(width, height, depth)?;

        if thermal_diffusivity <= 0.0 {
            return Err(PhysicsError::CalculationError(
                "Thermal diffusivity must be positive".to_string(),
            ));
        }
        if dt <= 0.0 {
            return Err(PhysicsError::CalculationError(
                "Time step must be positive".to_string(),
            ));
        }
        if dx <= 0.0 {
            return Err(PhysicsError::CalculationError(
                "Grid spacing must be positive".to_string(),
            ));
        }
        if initial_temp <= 0.0 {
            return Err(PhysicsError::CalculationError(
                "Initial temperature must be positive (Kelvin)".to_string(),
            ));
        }

        // Check CFL stability condition for 3D: α*dt/dx² ≤ 1/6
        let stability_factor = thermal_diffusivity * dt / (dx * dx);
        if stability_factor > 1.0 / 6.0 {
            return Err(PhysicsError::CalculationError(format!(
                "CFL stability condition violated: α*dt/dx² = {} > 1/6. \
                Reduce dt or increase dx.",
                stability_factor
            )));
        }

        let size = width * height * depth;
        let temperature = vec![initial_temp; size];
        let prev_temperature = vec![initial_temp; size];

        Ok(Self {
            width,
            height,
            depth,
            temperature,
            prev_temperature,
            thermal_diffusivity,
            dt,
            dx,
            heat_sources: Vec::new(),
            boundary_conditions: [ThermalBoundaryCondition::Insulated; 6],
        })
    }

    /// Returns the width of the grid
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height of the grid
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the depth of the grid
    #[inline]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Converts 3D coordinates to linear index
    #[inline]
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.width * self.height + y * self.width + x
    }

    /// Gets the temperature at a specific cell
    pub fn get_temperature(&self, x: usize, y: usize, z: usize) -> Result<f64, PhysicsError> {
        if x >= self.width || y >= self.height || z >= self.depth {
            return Err(PhysicsError::CalculationError(
                "Index out of bounds".to_string(),
            ));
        }
        Ok(self.temperature[self.index(x, y, z)])
    }

    /// Sets the temperature at a specific cell
    pub fn set_temperature(&mut self, x: usize, y: usize, z: usize, temp: f64) -> Result<(), PhysicsError> {
        if x >= self.width || y >= self.height || z >= self.depth {
            return Err(PhysicsError::CalculationError(
                "Index out of bounds".to_string(),
            ));
        }
        if temp <= 0.0 {
            return Err(PhysicsError::CalculationError(
                "Temperature must be positive (Kelvin)".to_string(),
            ));
        }
        let idx = self.index(x, y, z);
        self.temperature[idx] = temp;
        Ok(())
    }

    /// Sets boundary condition for a face
    pub fn set_boundary_condition(&mut self, face: GridFace, condition: ThermalBoundaryCondition) {
        let idx = match face {
            GridFace::Left => 0,
            GridFace::Right => 1,
            GridFace::Front => 2,
            GridFace::Back => 3,
            GridFace::Bottom => 4,
            GridFace::Top => 5,
        };
        self.boundary_conditions[idx] = condition;
    }

    /// Adds a heat source at the specified position
    pub fn add_heat_source(&mut self, x: usize, y: usize, z: usize, power: f64) -> Result<(), PhysicsError> {
        if x >= self.width || y >= self.height || z >= self.depth {
            return Err(PhysicsError::CalculationError(
                "Index out of bounds".to_string(),
            ));
        }
        self.heat_sources.push(HeatSource3D { x, y, z, power });
        Ok(())
    }

    /// Removes heat source at specified position
    pub fn remove_heat_source(&mut self, x: usize, y: usize, z: usize) {
        self.heat_sources.retain(|s| s.x != x || s.y != y || s.z != z);
    }

    /// Clears all heat sources
    pub fn clear_heat_sources(&mut self) {
        self.heat_sources.clear();
    }

    /// Advances the simulation by one time step using FTCS scheme
    pub fn step(&mut self) {
        std::mem::swap(&mut self.temperature, &mut self.prev_temperature);

        let alpha = self.thermal_diffusivity;
        let factor = alpha * self.dt / (self.dx * self.dx);
        let width = self.width;
        let height = self.height;
        let depth = self.depth;
        let dt = self.dt;

        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    let idx = z * width * height + y * width + x;
                    let current = self.prev_temperature[idx];

                    // Get neighboring temperatures with boundary handling
                    let t_xm = if x > 0 {
                        self.prev_temperature[z * width * height + y * width + (x - 1)]
                    } else {
                        current
                    };
                    let t_xp = if x < width - 1 {
                        self.prev_temperature[z * width * height + y * width + (x + 1)]
                    } else {
                        current
                    };
                    let t_ym = if y > 0 {
                        self.prev_temperature[z * width * height + (y - 1) * width + x]
                    } else {
                        current
                    };
                    let t_yp = if y < height - 1 {
                        self.prev_temperature[z * width * height + (y + 1) * width + x]
                    } else {
                        current
                    };
                    let t_zm = if z > 0 {
                        self.prev_temperature[(z - 1) * width * height + y * width + x]
                    } else {
                        current
                    };
                    let t_zp = if z < depth - 1 {
                        self.prev_temperature[(z + 1) * width * height + y * width + x]
                    } else {
                        current
                    };

                    // 3D Laplacian: ∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²
                    let laplacian = t_xm + t_xp + t_ym + t_yp + t_zm + t_zp - 6.0 * current;

                    let mut new_temp = current + factor * laplacian;

                    // Apply heat sources
                    for source in &self.heat_sources {
                        if source.x == x && source.y == y && source.z == z {
                            // Q = P * dt, ΔT = Q / (ρ * c * V)
                            // Since α = k/(ρ*c), we use: ΔT = P * dt * α / (k * dx³)
                            // Simplified: treat power as direct temperature increase rate
                            new_temp += source.power * dt;
                        }
                    }

                    self.temperature[idx] = new_temp;
                }
            }
        }

        // Apply Dirichlet boundary conditions
        self.apply_boundary_conditions();
    }

    /// Applies boundary conditions to the grid faces
    fn apply_boundary_conditions(&mut self) {
        let width = self.width;
        let height = self.height;
        let depth = self.depth;

        // Left face (x = 0)
        if let ThermalBoundaryCondition::Dirichlet(temp) = self.boundary_conditions[0] {
            for z in 0..depth {
                for y in 0..height {
                    let idx = z * width * height + y * width + 0;
                    self.temperature[idx] = temp;
                }
            }
        }

        // Right face (x = width - 1)
        if let ThermalBoundaryCondition::Dirichlet(temp) = self.boundary_conditions[1] {
            for z in 0..depth {
                for y in 0..height {
                    let idx = z * width * height + y * width + (width - 1);
                    self.temperature[idx] = temp;
                }
            }
        }

        // Front face (y = 0)
        if let ThermalBoundaryCondition::Dirichlet(temp) = self.boundary_conditions[2] {
            for z in 0..depth {
                for x in 0..width {
                    let idx = z * width * height + 0 * width + x;
                    self.temperature[idx] = temp;
                }
            }
        }

        // Back face (y = height - 1)
        if let ThermalBoundaryCondition::Dirichlet(temp) = self.boundary_conditions[3] {
            for z in 0..depth {
                for x in 0..width {
                    let idx = z * width * height + (height - 1) * width + x;
                    self.temperature[idx] = temp;
                }
            }
        }

        // Bottom face (z = 0)
        if let ThermalBoundaryCondition::Dirichlet(temp) = self.boundary_conditions[4] {
            for y in 0..height {
                for x in 0..width {
                    let idx = 0 * width * height + y * width + x;
                    self.temperature[idx] = temp;
                }
            }
        }

        // Top face (z = depth - 1)
        if let ThermalBoundaryCondition::Dirichlet(temp) = self.boundary_conditions[5] {
            for y in 0..height {
                for x in 0..width {
                    let idx = (depth - 1) * width * height + y * width + x;
                    self.temperature[idx] = temp;
                }
            }
        }
    }

    /// Returns the average temperature across the grid
    pub fn average_temperature(&self) -> f64 {
        let sum: f64 = self.temperature.iter().sum();
        sum / (self.temperature.len() as f64)
    }

    /// Returns the maximum temperature in the grid
    pub fn max_temperature(&self) -> f64 {
        self.temperature.iter().cloned().fold(f64::MIN, f64::max)
    }

    /// Returns the minimum temperature in the grid
    pub fn min_temperature(&self) -> f64 {
        self.temperature.iter().cloned().fold(f64::MAX, f64::min)
    }

    /// Returns the total thermal energy in the grid
    /// Assumes unit volume cells and unit volumetric heat capacity
    pub fn total_thermal_energy(&self, density: f64, specific_heat: f64) -> f64 {
        let cell_volume = self.dx * self.dx * self.dx;
        self.temperature.iter().sum::<f64>() * density * specific_heat * cell_volume
    }

    /// Checks if the simulation has reached steady state
    pub fn is_steady_state(&self, tolerance: f64) -> bool {
        for (t_curr, t_prev) in self.temperature.iter().zip(self.prev_temperature.iter()) {
            if (t_curr - t_prev).abs() > tolerance {
                return false;
            }
        }
        true
    }

    /// Resets the grid to a uniform temperature
    pub fn reset(&mut self, initial_temp: f64) {
        for t in &mut self.temperature {
            *t = initial_temp;
        }
        for t in &mut self.prev_temperature {
            *t = initial_temp;
        }
    }

    /// Extracts an XY slice at specified Z coordinate
    pub fn get_slice_xy(&self, z: usize) -> Result<Vec<f64>, PhysicsError> {
        if z >= self.depth {
            return Err(PhysicsError::CalculationError(
                "Z index out of bounds".to_string(),
            ));
        }
        let mut slice = Vec::with_capacity(self.width * self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                slice.push(self.temperature[self.index(x, y, z)]);
            }
        }
        Ok(slice)
    }

    /// Extracts an XZ slice at specified Y coordinate
    pub fn get_slice_xz(&self, y: usize) -> Result<Vec<f64>, PhysicsError> {
        if y >= self.height {
            return Err(PhysicsError::CalculationError(
                "Y index out of bounds".to_string(),
            ));
        }
        let mut slice = Vec::with_capacity(self.width * self.depth);
        for z in 0..self.depth {
            for x in 0..self.width {
                slice.push(self.temperature[self.index(x, y, z)]);
            }
        }
        Ok(slice)
    }

    /// Extracts a YZ slice at specified X coordinate
    pub fn get_slice_yz(&self, x: usize) -> Result<Vec<f64>, PhysicsError> {
        if x >= self.width {
            return Err(PhysicsError::CalculationError(
                "X index out of bounds".to_string(),
            ));
        }
        let mut slice = Vec::with_capacity(self.height * self.depth);
        for z in 0..self.depth {
            for y in 0..self.height {
                slice.push(self.temperature[self.index(x, y, z)]);
            }
        }
        Ok(slice)
    }

    /// Returns the temperature data as a flat vector
    pub fn temperature_data(&self) -> &[f64] {
        &self.temperature
    }

    /// Checks if the current parameters satisfy CFL stability condition
    pub fn is_stable(&self) -> bool {
        let stability_factor = self.thermal_diffusivity * self.dt / (self.dx * self.dx);
        stability_factor <= 1.0 / 6.0
    }

    /// Sets a cubic region to a specific temperature
    pub fn set_region(
        &mut self,
        x_start: usize,
        y_start: usize,
        z_start: usize,
        x_end: usize,
        y_end: usize,
        z_end: usize,
        temp: f64,
    ) -> Result<(), PhysicsError> {
        if x_end > self.width || y_end > self.height || z_end > self.depth {
            return Err(PhysicsError::CalculationError(
                "Region out of bounds".to_string(),
            ));
        }
        if temp <= 0.0 {
            return Err(PhysicsError::CalculationError(
                "Temperature must be positive (Kelvin)".to_string(),
            ));
        }

        for z in z_start..z_end {
            for y in y_start..y_end {
                for x in x_start..x_end {
                    let idx = self.index(x, y, z);
                    self.temperature[idx] = temp;
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_3d_creation() {
        let grid = ThermalGrid3D::new(10, 10, 10, 300.0, 1e-5, 0.01, 0.01);
        assert!(grid.is_ok());
        let grid = grid.unwrap();
        assert_eq!(grid.width(), 10);
        assert_eq!(grid.height(), 10);
        assert_eq!(grid.depth(), 10);
    }

    #[test]
    fn test_grid_3d_creation_invalid_dimensions() {
        // Zero dimensions should fail
        let grid = ThermalGrid3D::new(0, 10, 10, 300.0, 1e-5, 0.01, 0.01);
        assert!(grid.is_err());

        let grid = ThermalGrid3D::new(10, 0, 10, 300.0, 1e-5, 0.01, 0.01);
        assert!(grid.is_err());

        let grid = ThermalGrid3D::new(10, 10, 0, 300.0, 1e-5, 0.01, 0.01);
        assert!(grid.is_err());
    }

    #[test]
    fn test_grid_3d_creation_invalid_parameters() {
        // Negative thermal diffusivity
        let grid = ThermalGrid3D::new(10, 10, 10, 300.0, -1e-5, 0.01, 0.01);
        assert!(grid.is_err());

        // Negative time step
        let grid = ThermalGrid3D::new(10, 10, 10, 300.0, 1e-5, -0.01, 0.01);
        assert!(grid.is_err());

        // Negative grid spacing
        let grid = ThermalGrid3D::new(10, 10, 10, 300.0, 1e-5, 0.01, -0.01);
        assert!(grid.is_err());

        // CFL violation (dt too large for 3D)
        let grid = ThermalGrid3D::new(10, 10, 10, 300.0, 1e-5, 100.0, 0.01);
        assert!(grid.is_err());
    }

    #[test]
    fn test_grid_3d_set_get_temperature() {
        let mut grid = ThermalGrid3D::new(10, 10, 10, 300.0, 1e-5, 0.01, 0.01).unwrap();

        assert!(grid.set_temperature(5, 5, 5, 400.0).is_ok());
        assert_eq!(grid.get_temperature(5, 5, 5).unwrap(), 400.0);

        // Out of bounds
        assert!(grid.set_temperature(15, 5, 5, 400.0).is_err());
        assert!(grid.get_temperature(15, 5, 5).is_err());
    }

    #[test]
    fn test_grid_3d_boundary_conditions() {
        let mut grid = ThermalGrid3D::new(5, 5, 5, 300.0, 1e-5, 0.01, 0.01).unwrap();

        // Set hot left face, cold right face
        grid.set_boundary_condition(GridFace::Left, ThermalBoundaryCondition::Dirichlet(400.0));
        grid.set_boundary_condition(GridFace::Right, ThermalBoundaryCondition::Dirichlet(300.0));

        // Run several steps
        for _ in 0..10 {
            grid.step();
        }

        // Check boundaries are maintained
        assert!((grid.get_temperature(0, 2, 2).unwrap() - 400.0).abs() < 1e-10);
        assert!((grid.get_temperature(4, 2, 2).unwrap() - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_grid_3d_heat_sources() {
        let mut grid = ThermalGrid3D::new(5, 5, 5, 300.0, 1e-5, 0.01, 0.01).unwrap();

        assert!(grid.add_heat_source(2, 2, 2, 10.0).is_ok());

        let initial_temp = grid.get_temperature(2, 2, 2).unwrap();
        grid.step();
        let after_temp = grid.get_temperature(2, 2, 2).unwrap();

        // Temperature should increase at heat source
        assert!(after_temp > initial_temp);
    }

    #[test]
    fn test_grid_3d_remove_heat_source() {
        let mut grid = ThermalGrid3D::new(5, 5, 5, 300.0, 1e-5, 0.01, 0.01).unwrap();

        grid.add_heat_source(2, 2, 2, 10.0).unwrap();
        grid.add_heat_source(3, 3, 3, 10.0).unwrap();

        grid.remove_heat_source(2, 2, 2);

        // Only one source should remain
        assert_eq!(grid.heat_sources.len(), 1);
        assert_eq!(grid.heat_sources[0].x, 3);
    }

    #[test]
    fn test_grid_3d_clear_heat_sources() {
        let mut grid = ThermalGrid3D::new(5, 5, 5, 300.0, 1e-5, 0.01, 0.01).unwrap();

        grid.add_heat_source(1, 1, 1, 10.0).unwrap();
        grid.add_heat_source(2, 2, 2, 10.0).unwrap();
        grid.add_heat_source(3, 3, 3, 10.0).unwrap();

        grid.clear_heat_sources();

        assert!(grid.heat_sources.is_empty());
    }

    #[test]
    fn test_grid_3d_step_diffusion() {
        let mut grid = ThermalGrid3D::new(5, 5, 5, 300.0, 1e-5, 0.01, 0.01).unwrap();

        // Create hot spot in center
        grid.set_temperature(2, 2, 2, 400.0).unwrap();

        let hot_before = grid.get_temperature(2, 2, 2).unwrap();
        let neighbor_before = grid.get_temperature(2, 2, 1).unwrap();

        grid.step();

        let hot_after = grid.get_temperature(2, 2, 2).unwrap();
        let neighbor_after = grid.get_temperature(2, 2, 1).unwrap();

        // Hot spot should cool, neighbors should warm
        assert!(hot_after < hot_before);
        assert!(neighbor_after > neighbor_before);
    }

    #[test]
    fn test_grid_3d_steady_state() {
        let mut grid = ThermalGrid3D::new(5, 5, 5, 300.0, 1e-5, 0.01, 0.01).unwrap();

        // Uniform temperature should be at steady state
        grid.step();
        assert!(grid.is_steady_state(1e-10));
    }

    #[test]
    fn test_grid_3d_slice_xy() {
        let mut grid = ThermalGrid3D::new(3, 3, 3, 300.0, 1e-5, 0.01, 0.01).unwrap();

        // Set bottom layer to different temp
        grid.set_temperature(1, 1, 0, 400.0).unwrap();

        let slice = grid.get_slice_xy(0).unwrap();
        assert_eq!(slice.len(), 9);
        assert_eq!(slice[4], 400.0); // Center of 3x3 slice

        // Out of bounds
        assert!(grid.get_slice_xy(10).is_err());
    }

    #[test]
    fn test_grid_3d_slice_xz() {
        let mut grid = ThermalGrid3D::new(3, 3, 3, 300.0, 1e-5, 0.01, 0.01).unwrap();

        grid.set_temperature(1, 1, 1, 400.0).unwrap();

        let slice = grid.get_slice_xz(1).unwrap();
        assert_eq!(slice.len(), 9); // 3 (width) * 3 (depth)
        assert_eq!(slice[4], 400.0); // Center

        assert!(grid.get_slice_xz(10).is_err());
    }

    #[test]
    fn test_grid_3d_slice_yz() {
        let mut grid = ThermalGrid3D::new(3, 3, 3, 300.0, 1e-5, 0.01, 0.01).unwrap();

        grid.set_temperature(1, 1, 1, 400.0).unwrap();

        let slice = grid.get_slice_yz(1).unwrap();
        assert_eq!(slice.len(), 9); // 3 (height) * 3 (depth)
        assert_eq!(slice[4], 400.0); // Center

        assert!(grid.get_slice_yz(10).is_err());
    }

    #[test]
    fn test_grid_3d_average_temperature() {
        let grid = ThermalGrid3D::new(10, 10, 10, 300.0, 1e-5, 0.01, 0.01).unwrap();
        assert!((grid.average_temperature() - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_grid_3d_max_min_temperature() {
        let mut grid = ThermalGrid3D::new(5, 5, 5, 300.0, 1e-5, 0.01, 0.01).unwrap();

        grid.set_temperature(1, 1, 1, 400.0).unwrap();
        grid.set_temperature(3, 3, 3, 200.0).unwrap();

        assert_eq!(grid.max_temperature(), 400.0);
        assert_eq!(grid.min_temperature(), 200.0);
    }

    #[test]
    fn test_grid_3d_total_thermal_energy() {
        let grid = ThermalGrid3D::new(2, 2, 2, 300.0, 1e-5, 0.01, 0.1).unwrap();

        // 8 cells, each with volume dx³ = 0.1³ = 0.001 m³
        // Total energy = sum(T) * density * specific_heat * cell_volume
        // = (8 * 300) * 1.0 * 1.0 * 0.001 = 2.4 J
        let energy = grid.total_thermal_energy(1.0, 1.0);
        assert!((energy - 2.4).abs() < 1e-10);
    }

    #[test]
    fn test_grid_3d_reset() {
        let mut grid = ThermalGrid3D::new(5, 5, 5, 300.0, 1e-5, 0.01, 0.01).unwrap();

        grid.set_temperature(2, 2, 2, 500.0).unwrap();
        grid.reset(350.0);

        assert_eq!(grid.get_temperature(2, 2, 2).unwrap(), 350.0);
        assert_eq!(grid.average_temperature(), 350.0);
    }

    #[test]
    fn test_grid_3d_stability_check() {
        let grid = ThermalGrid3D::new(10, 10, 10, 300.0, 1e-5, 0.01, 0.01).unwrap();
        assert!(grid.is_stable());
    }

    #[test]
    fn test_grid_3d_set_region() {
        let mut grid = ThermalGrid3D::new(10, 10, 10, 300.0, 1e-5, 0.01, 0.01).unwrap();

        // Set a 2x2x2 cube to 400K
        grid.set_region(2, 2, 2, 4, 4, 4, 400.0).unwrap();

        // Check corners of the cube
        assert_eq!(grid.get_temperature(2, 2, 2).unwrap(), 400.0);
        assert_eq!(grid.get_temperature(3, 3, 3).unwrap(), 400.0);

        // Outside the cube should still be 300K
        assert_eq!(grid.get_temperature(5, 5, 5).unwrap(), 300.0);

        // Out of bounds
        assert!(grid.set_region(0, 0, 0, 20, 20, 20, 400.0).is_err());
    }

    #[test]
    fn test_grid_3d_heat_flows_from_hot_to_cold() {
        let mut grid = ThermalGrid3D::new(5, 5, 5, 300.0, 1e-5, 0.01, 0.01).unwrap();

        // Set one side hot, other side cold
        for z in 0..5 {
            for y in 0..5 {
                grid.set_temperature(0, y, z, 400.0).unwrap();
                grid.set_temperature(4, y, z, 200.0).unwrap();
            }
        }

        // Run many steps
        for _ in 0..100 {
            grid.step();
        }

        // Middle should be between hot and cold
        let mid = grid.get_temperature(2, 2, 2).unwrap();
        assert!(mid > 200.0);
        assert!(mid < 400.0);
    }

    #[test]
    fn test_grid_3d_approaches_equilibrium() {
        // Use larger thermal diffusivity for faster equilibration
        let mut grid = ThermalGrid3D::new(3, 3, 3, 300.0, 1e-4, 0.001, 0.01).unwrap();

        // Create temperature gradient
        grid.set_temperature(0, 0, 0, 400.0).unwrap();
        grid.set_temperature(2, 2, 2, 200.0).unwrap();

        let initial_diff = grid.max_temperature() - grid.min_temperature();

        // Run many steps
        for _ in 0..10000 {
            grid.step();
            if grid.is_steady_state(0.001) {
                break;
            }
        }

        // Should approach more uniform temperature than initial
        let final_diff = grid.max_temperature() - grid.min_temperature();
        assert!(final_diff < initial_diff, "Temperature should become more uniform");
        assert!(final_diff < 100.0, "Max temperature difference should decrease significantly");
    }
}
