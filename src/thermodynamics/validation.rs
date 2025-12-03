//! Shared validation utilities for thermodynamics modules
//!
//! This module provides common validation functions used across
//! thermodynamics calculations, heat transfer, and thermal simulation modules.

use crate::utils::PhysicsError;

/// Absolute zero in Celsius
pub const ABSOLUTE_ZERO_CELSIUS: f64 = -273.15;

/// Validates that a temperature in Kelvin is valid (> 0 K)
///
/// # Arguments
/// * `temperature` - The temperature in Kelvin
///
/// # Returns
/// * `Ok(())` if the temperature is valid (> 0 K)
/// * `Err(PhysicsError)` if the temperature is zero or negative
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::validate_temperature_kelvin;
///
/// assert!(validate_temperature_kelvin(300.0).is_ok());
/// assert!(validate_temperature_kelvin(0.0).is_err());
/// assert!(validate_temperature_kelvin(-50.0).is_err());
/// ```
#[inline]
pub fn validate_temperature_kelvin(temperature: f64) -> Result<(), PhysicsError> {
    if temperature <= 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("Temperature must be positive in Kelvin (> 0 K), got {} K", temperature)
        ));
    }
    Ok(())
}

/// Validates that a temperature in Celsius is valid (>= -273.15°C)
///
/// # Arguments
/// * `temperature` - The temperature in Celsius
///
/// # Returns
/// * `Ok(())` if the temperature is valid (>= -273.15°C)
/// * `Err(PhysicsError)` if the temperature is below absolute zero
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::validate_temperature_celsius;
///
/// assert!(validate_temperature_celsius(25.0).is_ok());
/// assert!(validate_temperature_celsius(-273.15).is_ok());
/// assert!(validate_temperature_celsius(-300.0).is_err());
/// ```
#[inline]
pub fn validate_temperature_celsius(temperature: f64) -> Result<(), PhysicsError> {
    if temperature < ABSOLUTE_ZERO_CELSIUS {
        return Err(PhysicsError::CalculationError(
            format!("Temperature cannot be below absolute zero (-273.15°C), got {}°C", temperature)
        ));
    }
    Ok(())
}

/// Validates that a pressure is valid (> 0)
///
/// # Arguments
/// * `pressure` - The pressure in Pascals
///
/// # Returns
/// * `Ok(())` if the pressure is positive
/// * `Err(PhysicsError)` if the pressure is zero or negative
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::validate_pressure;
///
/// assert!(validate_pressure(101325.0).is_ok());
/// assert!(validate_pressure(0.0).is_err());
/// assert!(validate_pressure(-100.0).is_err());
/// ```
#[inline]
pub fn validate_pressure(pressure: f64) -> Result<(), PhysicsError> {
    if pressure <= 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("Pressure must be positive, got {} Pa", pressure)
        ));
    }
    Ok(())
}

/// Validates that a volume is valid (> 0)
///
/// # Arguments
/// * `volume` - The volume in cubic meters
///
/// # Returns
/// * `Ok(())` if the volume is positive
/// * `Err(PhysicsError)` if the volume is zero or negative
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::validate_volume;
///
/// assert!(validate_volume(1.0).is_ok());
/// assert!(validate_volume(0.0).is_err());
/// assert!(validate_volume(-1.0).is_err());
/// ```
#[inline]
pub fn validate_volume(volume: f64) -> Result<(), PhysicsError> {
    if volume <= 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("Volume must be positive, got {} m³", volume)
        ));
    }
    Ok(())
}

/// Validates that the number of moles is valid (>= 0)
///
/// # Arguments
/// * `moles` - The amount of substance in moles
///
/// # Returns
/// * `Ok(())` if the moles is non-negative
/// * `Err(PhysicsError)` if the moles is negative
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::validate_moles;
///
/// assert!(validate_moles(1.0).is_ok());
/// assert!(validate_moles(0.0).is_ok());
/// assert!(validate_moles(-1.0).is_err());
/// ```
#[inline]
pub fn validate_moles(moles: f64) -> Result<(), PhysicsError> {
    if moles < 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("Number of moles must be non-negative, got {} mol", moles)
        ));
    }
    Ok(())
}

/// Validates that a heat capacity is valid (> 0)
///
/// # Arguments
/// * `heat_capacity` - The heat capacity in J/(kg·K) or J/(mol·K)
///
/// # Returns
/// * `Ok(())` if the heat capacity is positive
/// * `Err(PhysicsError)` if the heat capacity is zero or negative
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::validate_heat_capacity;
///
/// assert!(validate_heat_capacity(4186.0).is_ok());
/// assert!(validate_heat_capacity(0.0).is_err());
/// assert!(validate_heat_capacity(-100.0).is_err());
/// ```
#[inline]
pub fn validate_heat_capacity(heat_capacity: f64) -> Result<(), PhysicsError> {
    if heat_capacity <= 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("Heat capacity must be positive, got {}", heat_capacity)
        ));
    }
    Ok(())
}

/// Validates that a thermal conductivity is valid (> 0)
///
/// # Arguments
/// * `conductivity` - The thermal conductivity in W/(m·K)
///
/// # Returns
/// * `Ok(())` if the thermal conductivity is positive
/// * `Err(PhysicsError)` if the thermal conductivity is zero or negative
#[inline]
pub fn validate_thermal_conductivity(conductivity: f64) -> Result<(), PhysicsError> {
    if conductivity <= 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("Thermal conductivity must be positive, got {} W/(m·K)", conductivity)
        ));
    }
    Ok(())
}

/// Validates that an area is valid (> 0)
///
/// # Arguments
/// * `area` - The area in square meters
///
/// # Returns
/// * `Ok(())` if the area is positive
/// * `Err(PhysicsError)` if the area is zero or negative
#[inline]
pub fn validate_area(area: f64) -> Result<(), PhysicsError> {
    if area <= 0.0 {
        return Err(PhysicsError::InvalidArea);
    }
    Ok(())
}

/// Validates that a thickness/length is valid (> 0)
///
/// # Arguments
/// * `length` - The length/thickness in meters
///
/// # Returns
/// * `Ok(())` if the length is positive
/// * `Err(PhysicsError)` if the length is zero or negative
#[inline]
pub fn validate_length(length: f64) -> Result<(), PhysicsError> {
    if length <= 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("Length/thickness must be positive, got {} m", length)
        ));
    }
    Ok(())
}

/// Validates that an emissivity is valid (0 < ε <= 1)
///
/// # Arguments
/// * `emissivity` - The emissivity (dimensionless)
///
/// # Returns
/// * `Ok(())` if the emissivity is in the valid range
/// * `Err(PhysicsError)` if the emissivity is out of range
#[inline]
pub fn validate_emissivity(emissivity: f64) -> Result<(), PhysicsError> {
    if emissivity <= 0.0 || emissivity > 1.0 {
        return Err(PhysicsError::CalculationError(
            format!("Emissivity must be between 0 (exclusive) and 1 (inclusive), got {}", emissivity)
        ));
    }
    Ok(())
}

/// Validates that an efficiency is valid (0 <= η <= 1)
///
/// # Arguments
/// * `efficiency` - The efficiency (dimensionless)
///
/// # Returns
/// * `Ok(())` if the efficiency is in the valid range
/// * `Err(PhysicsError)` if the efficiency is out of range
#[inline]
pub fn validate_efficiency(efficiency: f64) -> Result<(), PhysicsError> {
    if efficiency < 0.0 || efficiency > 1.0 {
        return Err(PhysicsError::CalculationError(
            format!("Efficiency must be between 0 and 1, got {}", efficiency)
        ));
    }
    Ok(())
}

/// Validates that a mass is valid (> 0)
///
/// # Arguments
/// * `mass` - The mass in kilograms
///
/// # Returns
/// * `Ok(())` if the mass is positive
/// * `Err(PhysicsError)` if the mass is zero or negative
#[inline]
pub fn validate_mass(mass: f64) -> Result<(), PhysicsError> {
    if mass <= 0.0 {
        return Err(PhysicsError::InvalidMass);
    }
    Ok(())
}

/// Validates 2D grid dimensions for thermal simulation
///
/// # Arguments
/// * `width` - The width of the grid
/// * `height` - The height of the grid
///
/// # Returns
/// * `Ok(())` if both dimensions are at least 1
/// * `Err(PhysicsError::InvalidArea)` if either dimension is zero
#[inline]
pub fn validate_grid_dimensions_2d(width: usize, height: usize) -> Result<(), PhysicsError> {
    if width == 0 || height == 0 {
        return Err(PhysicsError::InvalidArea);
    }
    Ok(())
}

/// Validates 3D grid dimensions for thermal simulation
///
/// # Arguments
/// * `width` - The width of the grid (x)
/// * `height` - The height of the grid (y)
/// * `depth` - The depth of the grid (z)
///
/// # Returns
/// * `Ok(())` if all dimensions are at least 1
/// * `Err(PhysicsError::InvalidArea)` if any dimension is zero
#[inline]
pub fn validate_grid_dimensions_3d(width: usize, height: usize, depth: usize) -> Result<(), PhysicsError> {
    if width == 0 || height == 0 || depth == 0 {
        return Err(PhysicsError::InvalidArea);
    }
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_temperature_kelvin() {
        // Valid temperatures
        assert!(validate_temperature_kelvin(300.0).is_ok());
        assert!(validate_temperature_kelvin(0.001).is_ok());
        assert!(validate_temperature_kelvin(1e6).is_ok());

        // Invalid temperatures
        assert!(validate_temperature_kelvin(0.0).is_err());
        assert!(validate_temperature_kelvin(-1.0).is_err());
        assert!(validate_temperature_kelvin(-273.15).is_err());
    }

    #[test]
    fn test_validate_temperature_celsius() {
        // Valid temperatures
        assert!(validate_temperature_celsius(25.0).is_ok());
        assert!(validate_temperature_celsius(0.0).is_ok());
        assert!(validate_temperature_celsius(-100.0).is_ok());
        assert!(validate_temperature_celsius(-273.15).is_ok()); // Exactly absolute zero

        // Invalid temperatures (below absolute zero)
        assert!(validate_temperature_celsius(-273.16).is_err());
        assert!(validate_temperature_celsius(-300.0).is_err());
        assert!(validate_temperature_celsius(-1000.0).is_err());
    }

    #[test]
    fn test_validate_pressure() {
        // Valid pressures
        assert!(validate_pressure(101325.0).is_ok());
        assert!(validate_pressure(1.0).is_ok());
        assert!(validate_pressure(0.001).is_ok());

        // Invalid pressures
        assert!(validate_pressure(0.0).is_err());
        assert!(validate_pressure(-1.0).is_err());
        assert!(validate_pressure(-101325.0).is_err());
    }

    #[test]
    fn test_validate_volume() {
        // Valid volumes
        assert!(validate_volume(1.0).is_ok());
        assert!(validate_volume(0.001).is_ok());
        assert!(validate_volume(1e6).is_ok());

        // Invalid volumes
        assert!(validate_volume(0.0).is_err());
        assert!(validate_volume(-1.0).is_err());
    }

    #[test]
    fn test_validate_moles() {
        // Valid moles (including zero)
        assert!(validate_moles(1.0).is_ok());
        assert!(validate_moles(0.0).is_ok());
        assert!(validate_moles(0.001).is_ok());

        // Invalid moles
        assert!(validate_moles(-1.0).is_err());
        assert!(validate_moles(-0.001).is_err());
    }

    #[test]
    fn test_validate_heat_capacity() {
        // Valid heat capacities
        assert!(validate_heat_capacity(4186.0).is_ok()); // Water
        assert!(validate_heat_capacity(1.0).is_ok());
        assert!(validate_heat_capacity(0.001).is_ok());

        // Invalid heat capacities
        assert!(validate_heat_capacity(0.0).is_err());
        assert!(validate_heat_capacity(-1.0).is_err());
    }

    #[test]
    fn test_validate_thermal_conductivity() {
        // Valid thermal conductivities
        assert!(validate_thermal_conductivity(401.0).is_ok()); // Copper
        assert!(validate_thermal_conductivity(0.024).is_ok()); // Air

        // Invalid thermal conductivities
        assert!(validate_thermal_conductivity(0.0).is_err());
        assert!(validate_thermal_conductivity(-1.0).is_err());
    }

    #[test]
    fn test_validate_area() {
        // Valid areas
        assert!(validate_area(1.0).is_ok());
        assert!(validate_area(0.001).is_ok());

        // Invalid areas
        assert!(validate_area(0.0).is_err());
        assert!(validate_area(-1.0).is_err());
    }

    #[test]
    fn test_validate_length() {
        // Valid lengths
        assert!(validate_length(1.0).is_ok());
        assert!(validate_length(0.001).is_ok());

        // Invalid lengths
        assert!(validate_length(0.0).is_err());
        assert!(validate_length(-1.0).is_err());
    }

    #[test]
    fn test_validate_emissivity() {
        // Valid emissivities
        assert!(validate_emissivity(1.0).is_ok()); // Black body
        assert!(validate_emissivity(0.5).is_ok());
        assert!(validate_emissivity(0.001).is_ok());

        // Invalid emissivities
        assert!(validate_emissivity(0.0).is_err()); // Zero is invalid
        assert!(validate_emissivity(-0.1).is_err());
        assert!(validate_emissivity(1.1).is_err());
    }

    #[test]
    fn test_validate_efficiency() {
        // Valid efficiencies
        assert!(validate_efficiency(0.0).is_ok());
        assert!(validate_efficiency(0.5).is_ok());
        assert!(validate_efficiency(1.0).is_ok());

        // Invalid efficiencies
        assert!(validate_efficiency(-0.1).is_err());
        assert!(validate_efficiency(1.1).is_err());
    }

    #[test]
    fn test_validate_mass() {
        // Valid masses
        assert!(validate_mass(1.0).is_ok());
        assert!(validate_mass(0.001).is_ok());

        // Invalid masses
        assert!(validate_mass(0.0).is_err());
        assert!(validate_mass(-1.0).is_err());
    }

    #[test]
    fn test_validate_grid_dimensions_2d() {
        // Valid dimensions
        assert!(validate_grid_dimensions_2d(10, 10).is_ok());
        assert!(validate_grid_dimensions_2d(1, 1).is_ok());
        assert!(validate_grid_dimensions_2d(100, 50).is_ok());

        // Invalid dimensions
        assert!(validate_grid_dimensions_2d(0, 10).is_err());
        assert!(validate_grid_dimensions_2d(10, 0).is_err());
        assert!(validate_grid_dimensions_2d(0, 0).is_err());
    }

    #[test]
    fn test_validate_grid_dimensions_3d() {
        // Valid dimensions
        assert!(validate_grid_dimensions_3d(10, 10, 10).is_ok());
        assert!(validate_grid_dimensions_3d(1, 1, 1).is_ok());

        // Invalid dimensions
        assert!(validate_grid_dimensions_3d(0, 10, 10).is_err());
        assert!(validate_grid_dimensions_3d(10, 0, 10).is_err());
        assert!(validate_grid_dimensions_3d(10, 10, 0).is_err());
    }
}
