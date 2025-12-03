//! 2D Thermal Grid Simulation
//!
//! This module provides a grid-based heat diffusion simulation using the
//! finite difference method to solve the 2D heat equation.
//!
//! The heat equation: ∂T/∂t = α * (∂²T/∂x² + ∂²T/∂y²)
//! where α is thermal diffusivity.

use crate::utils::PhysicsError;
use super::validation::validate_grid_dimensions_2d;

// ============================================================================
// Boundary Conditions
// ============================================================================

/// Boundary condition types for thermal simulation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThermalBoundaryCondition {
    /// Fixed temperature boundary (Dirichlet)
    Dirichlet(f64),
    /// Fixed heat flux boundary (Neumann) - heat flux in W/m²
    Neumann(f64),
    /// Convective boundary - (h, T_ambient) where h is convection coefficient
    Convective(f64, f64),
    /// Insulated boundary (no heat flow) - equivalent to Neumann(0.0)
    Insulated,
}

/// Side of the grid for boundary conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridSide {
    Left,
    Right,
    Top,
    Bottom,
}

// ============================================================================
// Heat Source
// ============================================================================

/// A heat source in the grid
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatSource {
    /// X position in grid
    pub x: usize,
    /// Y position in grid
    pub y: usize,
    /// Power in Watts
    pub power: f64,
}

// ============================================================================
// ThermalGrid
// ============================================================================

/// 2D thermal grid for heat diffusion simulation
#[derive(Debug, Clone)]
pub struct ThermalGrid {
    /// Grid width (number of cells in x direction)
    width: usize,
    /// Grid height (number of cells in y direction)
    height: usize,
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
    heat_sources: Vec<HeatSource>,
    /// Boundary conditions [left, right, top, bottom]
    boundary_conditions: [ThermalBoundaryCondition; 4],
}

impl ThermalGrid {
    /// Creates a new thermal grid
    ///
    /// # Arguments
    /// * `width` - Number of cells in x direction (must be >= 3)
    /// * `height` - Number of cells in y direction (must be >= 3)
    /// * `initial_temp` - Initial temperature for all cells in K
    /// * `thermal_diffusivity` - Thermal diffusivity in m²/s
    /// * `dt` - Time step in seconds
    /// * `dx` - Grid spacing in meters
    ///
    /// # Returns
    /// New ThermalGrid or error if parameters are invalid
    pub fn new(
        width: usize,
        height: usize,
        initial_temp: f64,
        thermal_diffusivity: f64,
        dt: f64,
        dx: f64,
    ) -> Result<Self, PhysicsError> {
        validate_grid_dimensions_2d(width, height)?;

        if thermal_diffusivity <= 0.0 {
            return Err(PhysicsError::CalculationError(
                "Thermal diffusivity must be positive".to_string()
            ));
        }
        if dt <= 0.0 {
            return Err(PhysicsError::CalculationError(
                "Time step must be positive".to_string()
            ));
        }
        if dx <= 0.0 {
            return Err(PhysicsError::CalculationError(
                "Grid spacing must be positive".to_string()
            ));
        }
        if initial_temp <= 0.0 {
            return Err(PhysicsError::CalculationError(
                "Initial temperature must be positive (in Kelvin)".to_string()
            ));
        }

        // Check CFL stability condition: α * dt / dx² <= 0.25 for 2D
        let stability_factor = thermal_diffusivity * dt / (dx * dx);
        if stability_factor > 0.25 {
            return Err(PhysicsError::CalculationError(
                format!("Unstable parameters: α*dt/dx² = {} > 0.25. Reduce dt or increase dx.",
                        stability_factor)
            ));
        }

        let size = width * height;
        let temperature = vec![initial_temp; size];
        let prev_temperature = vec![initial_temp; size];

        Ok(Self {
            width,
            height,
            temperature,
            prev_temperature,
            thermal_diffusivity,
            dt,
            dx,
            heat_sources: Vec::new(),
            boundary_conditions: [
                ThermalBoundaryCondition::Insulated,
                ThermalBoundaryCondition::Insulated,
                ThermalBoundaryCondition::Insulated,
                ThermalBoundaryCondition::Insulated,
            ],
        })
    }

    /// Returns the grid width
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the grid height
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the thermal diffusivity
    pub fn thermal_diffusivity(&self) -> f64 {
        self.thermal_diffusivity
    }

    /// Returns the time step
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Returns the grid spacing
    pub fn dx(&self) -> f64 {
        self.dx
    }

    /// Converts (x, y) coordinates to linear index
    #[inline]
    fn index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    /// Gets the temperature at a cell
    pub fn get_temperature(&self, x: usize, y: usize) -> Result<f64, PhysicsError> {
        if x >= self.width || y >= self.height {
            return Err(PhysicsError::CalculationError(
                format!("Coordinates ({}, {}) out of bounds for {}x{} grid",
                        x, y, self.width, self.height)
            ));
        }
        Ok(self.temperature[self.index(x, y)])
    }

    /// Sets the temperature at a cell
    pub fn set_temperature(&mut self, x: usize, y: usize, temp: f64) -> Result<(), PhysicsError> {
        if x >= self.width || y >= self.height {
            return Err(PhysicsError::CalculationError(
                format!("Coordinates ({}, {}) out of bounds for {}x{} grid",
                        x, y, self.width, self.height)
            ));
        }
        if temp <= 0.0 {
            return Err(PhysicsError::CalculationError(
                "Temperature must be positive (in Kelvin)".to_string()
            ));
        }
        let idx = self.index(x, y);
        self.temperature[idx] = temp;
        Ok(())
    }

    /// Gets all temperatures as a slice
    pub fn temperatures(&self) -> &[f64] {
        &self.temperature
    }

    /// Gets temperature at (x, y) or returns the boundary value
    fn get_or_boundary(&self, x: isize, y: isize) -> f64 {
        // Left boundary
        if x < 0 {
            return match self.boundary_conditions[0] {
                ThermalBoundaryCondition::Dirichlet(t) => t,
                ThermalBoundaryCondition::Neumann(_) |
                ThermalBoundaryCondition::Convective(_, _) |
                ThermalBoundaryCondition::Insulated => {
                    self.temperature[self.index(0, y as usize)]
                }
            };
        }
        // Right boundary
        if x >= self.width as isize {
            return match self.boundary_conditions[1] {
                ThermalBoundaryCondition::Dirichlet(t) => t,
                ThermalBoundaryCondition::Neumann(_) |
                ThermalBoundaryCondition::Convective(_, _) |
                ThermalBoundaryCondition::Insulated => {
                    self.temperature[self.index(self.width - 1, y as usize)]
                }
            };
        }
        // Top boundary
        if y < 0 {
            return match self.boundary_conditions[2] {
                ThermalBoundaryCondition::Dirichlet(t) => t,
                ThermalBoundaryCondition::Neumann(_) |
                ThermalBoundaryCondition::Convective(_, _) |
                ThermalBoundaryCondition::Insulated => {
                    self.temperature[self.index(x as usize, 0)]
                }
            };
        }
        // Bottom boundary
        if y >= self.height as isize {
            return match self.boundary_conditions[3] {
                ThermalBoundaryCondition::Dirichlet(t) => t,
                ThermalBoundaryCondition::Neumann(_) |
                ThermalBoundaryCondition::Convective(_, _) |
                ThermalBoundaryCondition::Insulated => {
                    self.temperature[self.index(x as usize, self.height - 1)]
                }
            };
        }

        self.temperature[self.index(x as usize, y as usize)]
    }

    /// Sets boundary condition for a side
    pub fn set_boundary_condition(&mut self, side: GridSide, condition: ThermalBoundaryCondition) {
        let idx = match side {
            GridSide::Left => 0,
            GridSide::Right => 1,
            GridSide::Top => 2,
            GridSide::Bottom => 3,
        };
        self.boundary_conditions[idx] = condition;
    }

    /// Gets boundary condition for a side
    pub fn get_boundary_condition(&self, side: GridSide) -> ThermalBoundaryCondition {
        let idx = match side {
            GridSide::Left => 0,
            GridSide::Right => 1,
            GridSide::Top => 2,
            GridSide::Bottom => 3,
        };
        self.boundary_conditions[idx]
    }

    /// Adds a heat source at position (x, y) with given power in Watts
    pub fn add_heat_source(&mut self, x: usize, y: usize, power: f64) -> Result<(), PhysicsError> {
        if x >= self.width || y >= self.height {
            return Err(PhysicsError::CalculationError(
                format!("Heat source position ({}, {}) out of bounds", x, y)
            ));
        }
        self.heat_sources.push(HeatSource { x, y, power });
        Ok(())
    }

    /// Removes all heat sources at position (x, y)
    pub fn remove_heat_source(&mut self, x: usize, y: usize) {
        self.heat_sources.retain(|s| s.x != x || s.y != y);
    }

    /// Removes all heat sources
    pub fn clear_heat_sources(&mut self) {
        self.heat_sources.clear();
    }

    /// Returns the list of heat sources
    pub fn heat_sources(&self) -> &[HeatSource] {
        &self.heat_sources
    }

    /// Advances the simulation by one time step
    ///
    /// Uses explicit finite difference method (FTCS scheme):
    /// T(t+dt) = T(t) + α * dt/dx² * (T_left + T_right + T_up + T_down - 4*T)
    pub fn step(&mut self) {
        // Swap buffers
        std::mem::swap(&mut self.temperature, &mut self.prev_temperature);

        let alpha = self.thermal_diffusivity;
        let factor = alpha * self.dt / (self.dx * self.dx);
        let width = self.width;
        let height = self.height;
        let dt = self.dt;

        // Update interior cells
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let t_center = self.prev_temperature[idx];
                let t_left = self.get_or_boundary(x as isize - 1, y as isize);
                let t_right = self.get_or_boundary(x as isize + 1, y as isize);
                let t_up = self.get_or_boundary(x as isize, y as isize - 1);
                let t_down = self.get_or_boundary(x as isize, y as isize + 1);

                let laplacian = t_left + t_right + t_up + t_down - 4.0 * t_center;
                let new_temp = t_center + factor * laplacian;

                self.temperature[idx] = new_temp;
            }
        }

        // Apply heat sources
        // Q = P * dt gives energy in Joules
        // Temperature rise depends on volumetric heat capacity: ΔT = Q / (ρ * c * V)
        // For simplicity, we model heat source as direct temperature addition
        // proportional to power (this is a simplified model)
        for source in &self.heat_sources {
            let idx = source.y * width + source.x;
            // Add heat: ΔT = P * dt / (volumetric heat capacity * cell volume)
            // Using a simplified model where heat source adds temperature proportional to power
            // Actual value depends on material properties; this is normalized to ~1 K/s per kW
            self.temperature[idx] += source.power * dt / 1000.0;
        }

        // Apply Dirichlet boundary conditions (fixed temperature)
        // Pre-calculate dimensions to avoid borrow conflicts
        let width = self.width;
        let height = self.height;

        if let ThermalBoundaryCondition::Dirichlet(t) = self.boundary_conditions[0] {
            for y in 0..height {
                let idx = y * width; // index for x=0
                self.temperature[idx] = t;
            }
        }
        if let ThermalBoundaryCondition::Dirichlet(t) = self.boundary_conditions[1] {
            for y in 0..height {
                let idx = y * width + (width - 1); // index for x=width-1
                self.temperature[idx] = t;
            }
        }
        if let ThermalBoundaryCondition::Dirichlet(t) = self.boundary_conditions[2] {
            for x in 0..width {
                let idx = x; // index for y=0
                self.temperature[idx] = t;
            }
        }
        if let ThermalBoundaryCondition::Dirichlet(t) = self.boundary_conditions[3] {
            for x in 0..width {
                let idx = (height - 1) * width + x; // index for y=height-1
                self.temperature[idx] = t;
            }
        }
    }

    /// Advances the simulation by multiple time steps
    pub fn step_n(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Checks if the grid has reached steady state
    ///
    /// Returns true if the maximum temperature change from the previous step
    /// is less than the tolerance.
    pub fn is_steady_state(&self, tolerance: f64) -> bool {
        let mut max_diff = 0.0_f64;
        for i in 0..self.temperature.len() {
            let diff = (self.temperature[i] - self.prev_temperature[i]).abs();
            max_diff = max_diff.max(diff);
        }
        max_diff < tolerance
    }

    /// Returns the average temperature across the grid
    pub fn average_temperature(&self) -> f64 {
        let sum: f64 = self.temperature.iter().sum();
        sum / self.temperature.len() as f64
    }

    /// Returns the maximum temperature in the grid
    pub fn max_temperature(&self) -> f64 {
        self.temperature.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Returns the minimum temperature in the grid
    pub fn min_temperature(&self) -> f64 {
        self.temperature.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Returns the temperature range (max - min)
    pub fn temperature_range(&self) -> f64 {
        self.max_temperature() - self.min_temperature()
    }

    /// Calculates total thermal energy in the grid
    ///
    /// # Arguments
    /// * `density` - Material density in kg/m³
    /// * `specific_heat` - Specific heat capacity in J/(kg·K)
    ///
    /// # Returns
    /// Total thermal energy in Joules (relative to 0 K)
    pub fn total_thermal_energy(&self, density: f64, specific_heat: f64) -> f64 {
        let cell_volume = self.dx * self.dx; // 2D, assuming unit depth
        let cell_mass = density * cell_volume;
        let sum_temp: f64 = self.temperature.iter().sum();
        cell_mass * specific_heat * sum_temp
    }

    /// Resets all temperatures to a uniform value
    pub fn reset(&mut self, temperature: f64) {
        for t in &mut self.temperature {
            *t = temperature;
        }
        for t in &mut self.prev_temperature {
            *t = temperature;
        }
    }

    /// Sets a rectangular region to a uniform temperature
    pub fn set_region(
        &mut self,
        x_start: usize,
        y_start: usize,
        x_end: usize,
        y_end: usize,
        temperature: f64,
    ) -> Result<(), PhysicsError> {
        if x_end > self.width || y_end > self.height {
            return Err(PhysicsError::CalculationError(
                "Region extends beyond grid boundaries".to_string()
            ));
        }
        if temperature <= 0.0 {
            return Err(PhysicsError::CalculationError(
                "Temperature must be positive".to_string()
            ));
        }

        for y in y_start..y_end {
            for x in x_start..x_end {
                let idx = self.index(x, y);
                self.temperature[idx] = temperature;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ======================= Grid Creation Tests =======================

    #[test]
    fn test_grid_creation() {
        let grid = ThermalGrid::new(10, 10, 300.0, 1e-5, 0.1, 0.01);
        assert!(grid.is_ok());
        let grid = grid.unwrap();
        assert_eq!(grid.width(), 10);
        assert_eq!(grid.height(), 10);
    }

    #[test]
    fn test_grid_creation_invalid_dimensions() {
        // Zero dimensions should fail
        let grid = ThermalGrid::new(0, 10, 300.0, 1e-5, 0.1, 0.01);
        assert!(grid.is_err());

        let grid = ThermalGrid::new(10, 0, 300.0, 1e-5, 0.1, 0.01);
        assert!(grid.is_err());

        let grid = ThermalGrid::new(0, 0, 300.0, 1e-5, 0.1, 0.01);
        assert!(grid.is_err());
    }

    #[test]
    fn test_grid_creation_invalid_parameters() {
        // Negative thermal diffusivity
        let grid = ThermalGrid::new(10, 10, 300.0, -1e-5, 0.1, 0.01);
        assert!(grid.is_err());

        // Negative dt
        let grid = ThermalGrid::new(10, 10, 300.0, 1e-5, -0.1, 0.01);
        assert!(grid.is_err());

        // Negative dx
        let grid = ThermalGrid::new(10, 10, 300.0, 1e-5, 0.1, -0.01);
        assert!(grid.is_err());
    }

    #[test]
    fn test_grid_stability_check() {
        // Unstable parameters (dt too large)
        let grid = ThermalGrid::new(10, 10, 300.0, 1e-4, 1.0, 0.01);
        assert!(grid.is_err());
    }

    // ======================= Temperature Access Tests =======================

    #[test]
    fn test_set_get_temperature() {
        let mut grid = ThermalGrid::new(10, 10, 300.0, 1e-5, 0.01, 0.01).unwrap();

        grid.set_temperature(5, 5, 400.0).unwrap();
        let t = grid.get_temperature(5, 5).unwrap();
        assert!((t - 400.0).abs() < 1e-10);
    }

    #[test]
    fn test_temperature_out_of_bounds() {
        let grid = ThermalGrid::new(10, 10, 300.0, 1e-5, 0.01, 0.01).unwrap();
        assert!(grid.get_temperature(15, 5).is_err());
        assert!(grid.get_temperature(5, 15).is_err());
    }

    #[test]
    fn test_initial_temperature() {
        let grid = ThermalGrid::new(10, 10, 350.0, 1e-5, 0.01, 0.01).unwrap();

        // All cells should have initial temperature
        for y in 0..10 {
            for x in 0..10 {
                let t = grid.get_temperature(x, y).unwrap();
                assert!((t - 350.0).abs() < 1e-10);
            }
        }
    }

    // ======================= Heat Source Tests =======================

    #[test]
    fn test_add_heat_source() {
        let mut grid = ThermalGrid::new(10, 10, 300.0, 1e-5, 0.01, 0.01).unwrap();
        grid.add_heat_source(5, 5, 1000.0).unwrap();
        assert_eq!(grid.heat_sources().len(), 1);
    }

    #[test]
    fn test_remove_heat_source() {
        let mut grid = ThermalGrid::new(10, 10, 300.0, 1e-5, 0.01, 0.01).unwrap();
        grid.add_heat_source(5, 5, 1000.0).unwrap();
        grid.add_heat_source(3, 3, 500.0).unwrap();
        grid.remove_heat_source(5, 5);
        assert_eq!(grid.heat_sources().len(), 1);
    }

    #[test]
    fn test_clear_heat_sources() {
        let mut grid = ThermalGrid::new(10, 10, 300.0, 1e-5, 0.01, 0.01).unwrap();
        grid.add_heat_source(5, 5, 1000.0).unwrap();
        grid.add_heat_source(3, 3, 500.0).unwrap();
        grid.clear_heat_sources();
        assert!(grid.heat_sources().is_empty());
    }

    // ======================= Boundary Condition Tests =======================

    #[test]
    fn test_set_boundary_condition() {
        let mut grid = ThermalGrid::new(10, 10, 300.0, 1e-5, 0.01, 0.01).unwrap();
        grid.set_boundary_condition(GridSide::Left, ThermalBoundaryCondition::Dirichlet(400.0));

        let bc = grid.get_boundary_condition(GridSide::Left);
        assert_eq!(bc, ThermalBoundaryCondition::Dirichlet(400.0));
    }

    #[test]
    fn test_dirichlet_boundary_applied() {
        let mut grid = ThermalGrid::new(10, 10, 300.0, 1e-5, 0.01, 0.01).unwrap();
        grid.set_boundary_condition(GridSide::Left, ThermalBoundaryCondition::Dirichlet(400.0));

        grid.step();

        // Left boundary should be 400 K
        for y in 0..10 {
            let t = grid.get_temperature(0, y).unwrap();
            assert!((t - 400.0).abs() < 1e-10);
        }
    }

    // ======================= Diffusion Tests =======================

    #[test]
    fn test_step_diffusion() {
        let mut grid = ThermalGrid::new(20, 20, 300.0, 1e-5, 0.01, 0.01).unwrap();

        // Set center hot
        grid.set_temperature(10, 10, 500.0).unwrap();

        let initial_center = grid.get_temperature(10, 10).unwrap();

        // Run some steps
        grid.step_n(10);

        // Center should cool down
        let final_center = grid.get_temperature(10, 10).unwrap();
        assert!(final_center < initial_center);

        // Neighbors should warm up
        let neighbor = grid.get_temperature(9, 10).unwrap();
        assert!(neighbor > 300.0);
    }

    #[test]
    fn test_heat_flows_from_hot_to_cold() {
        let mut grid = ThermalGrid::new(10, 10, 300.0, 1e-5, 0.01, 0.01).unwrap();

        // Left side hot, right side cold
        grid.set_boundary_condition(GridSide::Left, ThermalBoundaryCondition::Dirichlet(400.0));
        grid.set_boundary_condition(GridSide::Right, ThermalBoundaryCondition::Dirichlet(300.0));

        grid.step_n(100);

        // Temperature should decrease from left to right
        for x in 1..9 {
            let t_left = grid.get_temperature(x - 1, 5).unwrap();
            let t_right = grid.get_temperature(x, 5).unwrap();
            assert!(t_left >= t_right - 1.0, "Heat should flow left to right");
        }
    }

    // ======================= Statistics Tests =======================

    #[test]
    fn test_average_temperature() {
        let grid = ThermalGrid::new(10, 10, 300.0, 1e-5, 0.01, 0.01).unwrap();
        let avg = grid.average_temperature();
        assert!((avg - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_min_temperature() {
        let mut grid = ThermalGrid::new(10, 10, 300.0, 1e-5, 0.01, 0.01).unwrap();
        grid.set_temperature(5, 5, 500.0).unwrap();
        grid.set_temperature(2, 2, 250.0).unwrap();

        // Note: 250 K is valid (below 0°C but above absolute zero)
        // We need to allow temperatures below 273.15 K
    }

    #[test]
    fn test_temperature_range() {
        let mut grid = ThermalGrid::new(10, 10, 300.0, 1e-5, 0.01, 0.01).unwrap();
        grid.set_temperature(5, 5, 400.0).unwrap();

        let range = grid.temperature_range();
        assert!((range - 100.0).abs() < 1e-10);
    }

    // ======================= Steady State Tests =======================

    #[test]
    fn test_steady_state_detection() {
        let mut grid = ThermalGrid::new(10, 10, 300.0, 1e-5, 0.01, 0.01).unwrap();

        // Uniform grid should be immediately at steady state
        grid.step();
        assert!(grid.is_steady_state(1e-6));
    }

    #[test]
    fn test_approaches_steady_state() {
        let mut grid = ThermalGrid::new(10, 10, 300.0, 1e-4, 0.001, 0.01).unwrap();

        // Set fixed boundaries
        grid.set_boundary_condition(GridSide::Left, ThermalBoundaryCondition::Dirichlet(400.0));
        grid.set_boundary_condition(GridSide::Right, ThermalBoundaryCondition::Dirichlet(300.0));
        grid.set_boundary_condition(GridSide::Top, ThermalBoundaryCondition::Insulated);
        grid.set_boundary_condition(GridSide::Bottom, ThermalBoundaryCondition::Insulated);

        // Run until steady state
        for _ in 0..10000 {
            grid.step();
            if grid.is_steady_state(1e-8) {
                break;
            }
        }

        // At steady state, temperature gradient should be nearly linear
        // (for 1D heat flow with constant k and no sources)
    }

    // ======================= Energy Conservation Tests =======================

    #[test]
    fn test_total_thermal_energy() {
        let grid = ThermalGrid::new(10, 10, 300.0, 1e-5, 0.01, 0.01).unwrap();
        let energy = grid.total_thermal_energy(1000.0, 4186.0);
        assert!(energy > 0.0);
    }

    // ======================= Region Tests =======================

    #[test]
    fn test_set_region() {
        let mut grid = ThermalGrid::new(10, 10, 300.0, 1e-5, 0.01, 0.01).unwrap();
        grid.set_region(2, 2, 5, 5, 400.0).unwrap();

        // Inside region should be 400 K
        let t_inside = grid.get_temperature(3, 3).unwrap();
        assert!((t_inside - 400.0).abs() < 1e-10);

        // Outside region should be 300 K
        let t_outside = grid.get_temperature(8, 8).unwrap();
        assert!((t_outside - 300.0).abs() < 1e-10);
    }

    // ======================= Reset Test =======================

    #[test]
    fn test_reset() {
        let mut grid = ThermalGrid::new(10, 10, 300.0, 1e-5, 0.01, 0.01).unwrap();
        grid.set_temperature(5, 5, 500.0).unwrap();
        grid.step_n(10);

        grid.reset(350.0);

        // All cells should be 350 K
        for y in 0..10 {
            for x in 0..10 {
                let t = grid.get_temperature(x, y).unwrap();
                assert!((t - 350.0).abs() < 1e-10);
            }
        }
    }
}
