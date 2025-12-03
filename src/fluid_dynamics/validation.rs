//! Shared validation utilities for fluid dynamics modules
//!
//! This module provides common validation functions used across
//! both the analytical fluid dynamics and grid-based simulation modules.

use crate::utils::PhysicsError;

/// Validates that a value is strictly positive (> 0)
///
/// # Arguments
/// * `value` - The value to validate
/// * `name` - The name of the parameter (for error messages)
///
/// # Returns
/// * `Ok(())` if the value is positive
/// * `Err(PhysicsError)` if the value is zero or negative
#[inline]
pub fn validate_positive(value: f64, name: &str) -> Result<(), PhysicsError> {
    if value <= 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("{} must be positive, got {}", name, value)
        ));
    }
    Ok(())
}

/// Validates that a value is non-negative (>= 0)
///
/// # Arguments
/// * `value` - The value to validate
/// * `name` - The name of the parameter (for error messages)
///
/// # Returns
/// * `Ok(())` if the value is non-negative
/// * `Err(PhysicsError)` if the value is negative
#[inline]
pub fn validate_non_negative(value: f64, name: &str) -> Result<(), PhysicsError> {
    if value < 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("{} must be non-negative, got {}", name, value)
        ));
    }
    Ok(())
}

/// Validates that a value is finite (not NaN or infinite)
///
/// # Arguments
/// * `value` - The value to validate
/// * `name` - The name of the parameter (for error messages)
///
/// # Returns
/// * `Ok(())` if the value is finite
/// * `Err(PhysicsError)` if the value is NaN or infinite
#[inline]
pub fn validate_finite(value: f64, name: &str) -> Result<(), PhysicsError> {
    if !value.is_finite() {
        return Err(PhysicsError::CalculationError(
            format!("{} must be finite, got {}", name, value)
        ));
    }
    Ok(())
}

/// Validates 2D grid dimensions
///
/// # Arguments
/// * `width` - The width of the grid
/// * `height` - The height of the grid
///
/// # Returns
/// * `Ok(())` if both dimensions are at least 1
/// * `Err(PhysicsError::InvalidArea)` if either dimension is zero
#[inline]
pub fn validate_dimensions_2d(width: usize, height: usize) -> Result<(), PhysicsError> {
    if width == 0 || height == 0 {
        return Err(PhysicsError::InvalidArea);
    }
    Ok(())
}

/// Validates 3D grid dimensions
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
pub fn validate_dimensions_3d(width: usize, height: usize, depth: usize) -> Result<(), PhysicsError> {
    if width == 0 || height == 0 || depth == 0 {
        return Err(PhysicsError::InvalidArea);
    }
    Ok(())
}

/// Validates that a 2D position is within grid bounds
///
/// # Arguments
/// * `x` - The x-coordinate
/// * `y` - The y-coordinate
/// * `width` - The width of the grid
/// * `height` - The height of the grid
///
/// # Returns
/// * `Ok(())` if the position is within bounds
/// * `Err(PhysicsError)` if the position is out of bounds
#[inline]
pub fn validate_position_2d(x: usize, y: usize, width: usize, height: usize) -> Result<(), PhysicsError> {
    if x >= width || y >= height {
        return Err(PhysicsError::CalculationError(
            format!("Position ({}, {}) out of bounds for grid {}x{}", x, y, width, height)
        ));
    }
    Ok(())
}

/// Validates that a 3D position is within grid bounds
///
/// # Arguments
/// * `x` - The x-coordinate
/// * `y` - The y-coordinate
/// * `z` - The z-coordinate
/// * `width` - The width of the grid
/// * `height` - The height of the grid
/// * `depth` - The depth of the grid
///
/// # Returns
/// * `Ok(())` if the position is within bounds
/// * `Err(PhysicsError)` if the position is out of bounds
#[inline]
pub fn validate_position_3d(
    x: usize, y: usize, z: usize,
    width: usize, height: usize, depth: usize
) -> Result<(), PhysicsError> {
    if x >= width || y >= height || z >= depth {
        return Err(PhysicsError::CalculationError(
            format!("Position ({}, {}, {}) out of bounds for grid {}x{}x{}",
                    x, y, z, width, height, depth)
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_positive() {
        assert!(validate_positive(1.0, "test").is_ok());
        assert!(validate_positive(0.001, "test").is_ok());
        assert!(validate_positive(0.0, "test").is_err());
        assert!(validate_positive(-1.0, "test").is_err());
    }

    #[test]
    fn test_validate_non_negative() {
        assert!(validate_non_negative(1.0, "test").is_ok());
        assert!(validate_non_negative(0.0, "test").is_ok());
        assert!(validate_non_negative(-0.001, "test").is_err());
    }

    #[test]
    fn test_validate_finite() {
        assert!(validate_finite(1.0, "test").is_ok());
        assert!(validate_finite(0.0, "test").is_ok());
        assert!(validate_finite(f64::NAN, "test").is_err());
        assert!(validate_finite(f64::INFINITY, "test").is_err());
        assert!(validate_finite(f64::NEG_INFINITY, "test").is_err());
    }

    #[test]
    fn test_validate_dimensions_2d() {
        assert!(validate_dimensions_2d(10, 10).is_ok());
        assert!(validate_dimensions_2d(1, 1).is_ok());
        assert!(validate_dimensions_2d(0, 10).is_err());
        assert!(validate_dimensions_2d(10, 0).is_err());
        assert!(validate_dimensions_2d(0, 0).is_err());
    }

    #[test]
    fn test_validate_dimensions_3d() {
        assert!(validate_dimensions_3d(10, 10, 10).is_ok());
        assert!(validate_dimensions_3d(1, 1, 1).is_ok());
        assert!(validate_dimensions_3d(0, 10, 10).is_err());
        assert!(validate_dimensions_3d(10, 0, 10).is_err());
        assert!(validate_dimensions_3d(10, 10, 0).is_err());
    }

    #[test]
    fn test_validate_position_2d() {
        assert!(validate_position_2d(5, 5, 10, 10).is_ok());
        assert!(validate_position_2d(0, 0, 10, 10).is_ok());
        assert!(validate_position_2d(9, 9, 10, 10).is_ok());
        assert!(validate_position_2d(10, 5, 10, 10).is_err());
        assert!(validate_position_2d(5, 10, 10, 10).is_err());
    }

    #[test]
    fn test_validate_position_3d() {
        assert!(validate_position_3d(5, 5, 5, 10, 10, 10).is_ok());
        assert!(validate_position_3d(0, 0, 0, 10, 10, 10).is_ok());
        assert!(validate_position_3d(9, 9, 9, 10, 10, 10).is_ok());
        assert!(validate_position_3d(10, 5, 5, 10, 10, 10).is_err());
        assert!(validate_position_3d(5, 10, 5, 10, 10, 10).is_err());
        assert!(validate_position_3d(5, 5, 10, 10, 10, 10).is_err());
    }
}
