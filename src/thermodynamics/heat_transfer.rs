//! Heat Transfer Modes
//!
//! This module provides functions for calculating heat transfer through
//! conduction, convection, and radiation, as well as thermal resistance calculations.

use crate::utils::PhysicsError;
use super::constants::STEFAN_BOLTZMANN;
use super::validation::{
    validate_thermal_conductivity, validate_area, validate_length,
    validate_temperature_kelvin, validate_emissivity,
};

// ============================================================================
// Conduction (Fourier's Law)
// ============================================================================

/// Calculates the steady-state heat transfer rate through conduction
///
/// Uses Fourier's Law: Q = k * A * ΔT / L
///
/// # Arguments
/// * `k` - Thermal conductivity in W/(m·K)
/// * `area` - Cross-sectional area in m²
/// * `delta_t` - Temperature difference in K
/// * `thickness` - Material thickness in m
///
/// # Returns
/// Heat transfer rate in Watts (W)
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::conduction_rate;
///
/// // Heat through a 10cm steel plate, 1m² area, 50K difference
/// let q = conduction_rate(50.0, 1.0, 50.0, 0.1).unwrap();
/// assert!((q - 25000.0).abs() < 1.0);
/// ```
pub fn conduction_rate(k: f64, area: f64, delta_t: f64, thickness: f64) -> Result<f64, PhysicsError> {
    validate_thermal_conductivity(k)?;
    validate_area(area)?;
    validate_length(thickness)?;

    Ok(k * area * delta_t / thickness)
}

/// Calculates heat transfer through multiple layers in series
///
/// Q = ΔT / (Σ(L_i / k_i * A))
///
/// # Arguments
/// * `layers` - Vector of (thermal_conductivity, thickness) tuples for each layer
/// * `area` - Common cross-sectional area in m²
/// * `delta_t` - Total temperature difference in K
///
/// # Returns
/// Heat transfer rate in Watts
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::conduction_multilayer;
///
/// // Two layers: brick (0.7 W/mK, 20cm) and insulation (0.04 W/mK, 5cm)
/// let layers = vec![(0.7, 0.2), (0.04, 0.05)];
/// let q = conduction_multilayer(&layers, 1.0, 30.0).unwrap();
/// ```
pub fn conduction_multilayer(layers: &[(f64, f64)], area: f64, delta_t: f64) -> Result<f64, PhysicsError> {
    validate_area(area)?;

    if layers.is_empty() {
        return Err(PhysicsError::CalculationError(
            "At least one layer must be specified".to_string()
        ));
    }

    // Calculate total thermal resistance
    let mut total_resistance = 0.0;
    for (k, thickness) in layers {
        validate_thermal_conductivity(*k)?;
        validate_length(*thickness)?;
        total_resistance += thickness / (k * area);
    }

    Ok(delta_t / total_resistance)
}

/// Calculates heat transfer through a cylindrical wall (pipe)
///
/// Q = 2πkL(T_inner - T_outer) / ln(r_outer/r_inner)
///
/// # Arguments
/// * `k` - Thermal conductivity in W/(m·K)
/// * `length` - Pipe length in m
/// * `r_inner` - Inner radius in m
/// * `r_outer` - Outer radius in m
/// * `delta_t` - Temperature difference (T_inner - T_outer) in K
///
/// # Returns
/// Heat transfer rate in Watts
pub fn conduction_cylindrical(
    k: f64,
    length: f64,
    r_inner: f64,
    r_outer: f64,
    delta_t: f64,
) -> Result<f64, PhysicsError> {
    validate_thermal_conductivity(k)?;
    validate_length(length)?;

    if r_inner <= 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("Inner radius must be positive, got {} m", r_inner)
        ));
    }
    if r_outer <= r_inner {
        return Err(PhysicsError::CalculationError(
            format!("Outer radius ({}) must be greater than inner radius ({})", r_outer, r_inner)
        ));
    }

    let ln_ratio = (r_outer / r_inner).ln();
    Ok(2.0 * std::f64::consts::PI * k * length * delta_t / ln_ratio)
}

/// Calculates heat transfer through a spherical shell
///
/// Q = 4πk(T_inner - T_outer) / (1/r_inner - 1/r_outer)
///
/// # Arguments
/// * `k` - Thermal conductivity in W/(m·K)
/// * `r_inner` - Inner radius in m
/// * `r_outer` - Outer radius in m
/// * `delta_t` - Temperature difference (T_inner - T_outer) in K
///
/// # Returns
/// Heat transfer rate in Watts
pub fn conduction_spherical(
    k: f64,
    r_inner: f64,
    r_outer: f64,
    delta_t: f64,
) -> Result<f64, PhysicsError> {
    validate_thermal_conductivity(k)?;

    if r_inner <= 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("Inner radius must be positive, got {} m", r_inner)
        ));
    }
    if r_outer <= r_inner {
        return Err(PhysicsError::CalculationError(
            format!("Outer radius ({}) must be greater than inner radius ({})", r_outer, r_inner)
        ));
    }

    let reciprocal_diff = 1.0 / r_inner - 1.0 / r_outer;
    Ok(4.0 * std::f64::consts::PI * k * delta_t / reciprocal_diff)
}

// ============================================================================
// Convection (Newton's Law of Cooling)
// ============================================================================

/// Calculates heat transfer rate by convection
///
/// Uses Newton's Law of Cooling: Q = h * A * (T_surface - T_fluid)
///
/// # Arguments
/// * `h` - Convective heat transfer coefficient in W/(m²·K)
/// * `area` - Surface area in m²
/// * `t_surface` - Surface temperature in K
/// * `t_fluid` - Fluid temperature in K
///
/// # Returns
/// Heat transfer rate in Watts (positive = heat leaving surface)
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::convection_rate;
///
/// // Hot surface at 350K in 300K fluid with h=25 W/(m²·K)
/// let q = convection_rate(25.0, 1.0, 350.0, 300.0).unwrap();
/// assert!((q - 1250.0).abs() < 1.0);
/// ```
pub fn convection_rate(
    h: f64,
    area: f64,
    t_surface: f64,
    t_fluid: f64,
) -> Result<f64, PhysicsError> {
    if h <= 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("Convection coefficient must be positive, got {} W/(m²·K)", h)
        ));
    }
    validate_area(area)?;
    validate_temperature_kelvin(t_surface)?;
    validate_temperature_kelvin(t_fluid)?;

    Ok(h * area * (t_surface - t_fluid))
}

/// Convection type for estimating heat transfer coefficient
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvectionType {
    /// Natural/free convection (buoyancy-driven)
    Natural,
    /// Forced convection (external flow imposed)
    Forced,
}

/// Estimates the convective heat transfer coefficient
///
/// This provides approximate values based on empirical correlations.
/// For accurate results, use appropriate Nusselt number correlations.
///
/// # Arguments
/// * `convection_type` - Natural or Forced convection
/// * `is_gas` - true for gases, false for liquids
///
/// # Returns
/// Approximate heat transfer coefficient in W/(m²·K)
///
/// Typical ranges:
/// - Natural convection (gas): 2-25 W/(m²·K)
/// - Natural convection (liquid): 50-1000 W/(m²·K)
/// - Forced convection (gas): 25-250 W/(m²·K)
/// - Forced convection (liquid): 50-20000 W/(m²·K)
pub fn estimate_convection_coefficient(
    convection_type: ConvectionType,
    is_gas: bool,
) -> f64 {
    match (convection_type, is_gas) {
        (ConvectionType::Natural, true) => 10.0,   // Air natural convection
        (ConvectionType::Natural, false) => 500.0, // Water natural convection
        (ConvectionType::Forced, true) => 100.0,   // Air forced convection
        (ConvectionType::Forced, false) => 5000.0, // Water forced convection
    }
}

// ============================================================================
// Radiation (Stefan-Boltzmann Law)
// ============================================================================

/// Calculates radiative heat transfer rate from a surface
///
/// Uses Stefan-Boltzmann law: Q = ε * σ * A * (T_surface⁴ - T_surroundings⁴)
///
/// # Arguments
/// * `emissivity` - Surface emissivity (0 < ε ≤ 1)
/// * `area` - Surface area in m²
/// * `t_surface` - Surface temperature in K
/// * `t_surroundings` - Surrounding temperature in K
///
/// # Returns
/// Heat transfer rate in Watts (positive = heat leaving surface)
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::radiation_rate;
///
/// // Blackbody (ε=1) at 500K in 300K surroundings, 1m² area
/// let q = radiation_rate(1.0, 1.0, 500.0, 300.0).unwrap();
/// assert!(q > 0.0); // Heat is radiated away
/// ```
pub fn radiation_rate(
    emissivity: f64,
    area: f64,
    t_surface: f64,
    t_surroundings: f64,
) -> Result<f64, PhysicsError> {
    validate_emissivity(emissivity)?;
    validate_area(area)?;
    validate_temperature_kelvin(t_surface)?;
    validate_temperature_kelvin(t_surroundings)?;

    let t_s4 = t_surface.powi(4);
    let t_surr4 = t_surroundings.powi(4);

    Ok(emissivity * STEFAN_BOLTZMANN * area * (t_s4 - t_surr4))
}

/// Calculates radiative heat exchange between two surfaces
///
/// Q = F₁₂ * ε_eff * σ * A * (T₁⁴ - T₂⁴)
///
/// where ε_eff ≈ 1 / (1/ε₁ + 1/ε₂ - 1) for two large parallel plates
///
/// # Arguments
/// * `emissivity1` - Emissivity of surface 1
/// * `emissivity2` - Emissivity of surface 2
/// * `area` - Exchange area in m²
/// * `t1` - Temperature of surface 1 in K
/// * `t2` - Temperature of surface 2 in K
/// * `view_factor` - View factor F₁₂ (0 to 1)
///
/// # Returns
/// Heat transfer rate from surface 1 to surface 2 in Watts
pub fn radiation_between_surfaces(
    emissivity1: f64,
    emissivity2: f64,
    area: f64,
    t1: f64,
    t2: f64,
    view_factor: f64,
) -> Result<f64, PhysicsError> {
    validate_emissivity(emissivity1)?;
    validate_emissivity(emissivity2)?;
    validate_area(area)?;
    validate_temperature_kelvin(t1)?;
    validate_temperature_kelvin(t2)?;

    if view_factor < 0.0 || view_factor > 1.0 {
        return Err(PhysicsError::CalculationError(
            format!("View factor must be between 0 and 1, got {}", view_factor)
        ));
    }

    // Effective emissivity for two-surface enclosure
    let emissivity_eff = 1.0 / (1.0 / emissivity1 + 1.0 / emissivity2 - 1.0);

    let t1_4 = t1.powi(4);
    let t2_4 = t2.powi(4);

    Ok(view_factor * emissivity_eff * STEFAN_BOLTZMANN * area * (t1_4 - t2_4))
}

/// Calculates the blackbody emissive power
///
/// E_b = σ * T⁴
///
/// # Arguments
/// * `temperature` - Temperature in K
///
/// # Returns
/// Emissive power in W/m²
pub fn blackbody_emissive_power(temperature: f64) -> Result<f64, PhysicsError> {
    validate_temperature_kelvin(temperature)?;
    Ok(STEFAN_BOLTZMANN * temperature.powi(4))
}

// ============================================================================
// Thermal Resistance
// ============================================================================

/// Calculates thermal resistance for conduction
///
/// R = L / (k * A)
///
/// # Arguments
/// * `k` - Thermal conductivity in W/(m·K)
/// * `area` - Cross-sectional area in m²
/// * `thickness` - Material thickness in m
///
/// # Returns
/// Thermal resistance in K/W
pub fn thermal_resistance_conduction(k: f64, area: f64, thickness: f64) -> Result<f64, PhysicsError> {
    validate_thermal_conductivity(k)?;
    validate_area(area)?;
    validate_length(thickness)?;

    Ok(thickness / (k * area))
}

/// Calculates thermal resistance for convection
///
/// R = 1 / (h * A)
///
/// # Arguments
/// * `h` - Convective heat transfer coefficient in W/(m²·K)
/// * `area` - Surface area in m²
///
/// # Returns
/// Thermal resistance in K/W
pub fn thermal_resistance_convection(h: f64, area: f64) -> Result<f64, PhysicsError> {
    if h <= 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("Convection coefficient must be positive, got {} W/(m²·K)", h)
        ));
    }
    validate_area(area)?;

    Ok(1.0 / (h * area))
}

/// Calculates total thermal resistance for resistances in series
///
/// R_total = R₁ + R₂ + ... + Rₙ
///
/// # Arguments
/// * `resistances` - Slice of thermal resistances in K/W
///
/// # Returns
/// Total thermal resistance in K/W
#[inline]
pub fn thermal_resistance_series(resistances: &[f64]) -> f64 {
    resistances.iter().sum()
}

/// Calculates total thermal resistance for resistances in parallel
///
/// 1/R_total = 1/R₁ + 1/R₂ + ... + 1/Rₙ
///
/// # Arguments
/// * `resistances` - Slice of thermal resistances in K/W
///
/// # Returns
/// Total thermal resistance in K/W
pub fn thermal_resistance_parallel(resistances: &[f64]) -> Result<f64, PhysicsError> {
    if resistances.is_empty() {
        return Err(PhysicsError::CalculationError(
            "At least one resistance must be specified".to_string()
        ));
    }

    let sum_reciprocals: f64 = resistances.iter()
        .map(|r| {
            if *r <= 0.0 { 0.0 } else { 1.0 / r }
        })
        .sum();

    if sum_reciprocals <= 0.0 {
        return Err(PhysicsError::CalculationError(
            "All resistances must be positive".to_string()
        ));
    }

    Ok(1.0 / sum_reciprocals)
}

/// Calculates overall heat transfer coefficient from thermal resistances
///
/// U = 1 / (A * R_total)
///
/// # Arguments
/// * `total_resistance` - Total thermal resistance in K/W
/// * `area` - Reference area in m²
///
/// # Returns
/// Overall heat transfer coefficient in W/(m²·K)
pub fn overall_heat_transfer_coefficient(
    total_resistance: f64,
    area: f64,
) -> Result<f64, PhysicsError> {
    if total_resistance <= 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("Total resistance must be positive, got {} K/W", total_resistance)
        ));
    }
    validate_area(area)?;

    Ok(1.0 / (area * total_resistance))
}

/// Calculates heat transfer rate using thermal resistance
///
/// Q = ΔT / R
///
/// # Arguments
/// * `delta_t` - Temperature difference in K
/// * `resistance` - Thermal resistance in K/W
///
/// # Returns
/// Heat transfer rate in Watts
pub fn heat_rate_from_resistance(delta_t: f64, resistance: f64) -> Result<f64, PhysicsError> {
    if resistance <= 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("Thermal resistance must be positive, got {} K/W", resistance)
        ));
    }

    Ok(delta_t / resistance)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ======================= Conduction Tests =======================

    #[test]
    fn test_conduction_fourier_law() {
        // Q = k * A * ΔT / L
        // 50 W/mK * 1 m² * 100 K / 0.5 m = 10000 W
        let q = conduction_rate(50.0, 1.0, 100.0, 0.5).unwrap();
        assert!((q - 10000.0).abs() < 1.0);
    }

    #[test]
    fn test_conduction_multilayer() {
        // Two layers in series
        let layers = vec![(50.0, 0.1), (0.04, 0.05)];
        let q = conduction_multilayer(&layers, 1.0, 100.0).unwrap();

        // R1 = 0.1 / 50 = 0.002, R2 = 0.05 / 0.04 = 1.25
        // R_total = 1.252, Q = 100 / 1.252 ≈ 79.9
        assert!((q - 79.9).abs() < 1.0);
    }

    #[test]
    fn test_conduction_cylindrical() {
        // Pipe with inner radius 0.02m, outer 0.025m, length 1m
        let q = conduction_cylindrical(50.0, 1.0, 0.02, 0.025, 50.0).unwrap();
        // Q = 2π * 50 * 1 * 50 / ln(0.025/0.02)
        let expected = 2.0 * std::f64::consts::PI * 50.0 * 1.0 * 50.0 / (0.025_f64 / 0.02).ln();
        assert!((q - expected).abs() < 1.0);
    }

    #[test]
    fn test_conduction_spherical() {
        // Spherical shell with inner radius 0.1m, outer 0.2m
        let q = conduction_spherical(50.0, 0.1, 0.2, 100.0).unwrap();
        // Q = 4π * 50 * 100 / (1/0.1 - 1/0.2) = 4π * 5000 / 5 = 4π * 1000
        let expected = 4.0 * std::f64::consts::PI * 50.0 * 100.0 / (10.0 - 5.0);
        assert!((q - expected).abs() < 1.0);
    }

    #[test]
    fn test_conduction_invalid_inputs() {
        assert!(conduction_rate(0.0, 1.0, 100.0, 0.1).is_err()); // Zero conductivity
        assert!(conduction_rate(50.0, 0.0, 100.0, 0.1).is_err()); // Zero area
        assert!(conduction_rate(50.0, 1.0, 100.0, 0.0).is_err()); // Zero thickness

        assert!(conduction_cylindrical(50.0, 1.0, 0.0, 0.1, 50.0).is_err()); // Zero inner radius
        assert!(conduction_cylindrical(50.0, 1.0, 0.1, 0.05, 50.0).is_err()); // Outer < inner
    }

    // ======================= Convection Tests =======================

    #[test]
    fn test_convection_newton_cooling() {
        // Q = h * A * (T_s - T_f)
        let q = convection_rate(25.0, 2.0, 350.0, 300.0).unwrap();
        // 25 * 2 * 50 = 2500 W
        assert!((q - 2500.0).abs() < 1.0);
    }

    #[test]
    fn test_convection_cooling() {
        // Surface cooler than fluid - negative heat transfer (heat into surface)
        let q = convection_rate(25.0, 1.0, 280.0, 300.0).unwrap();
        assert!(q < 0.0, "Heat should flow into the surface");
    }

    #[test]
    fn test_estimate_convection_coefficient() {
        let h_nat_gas = estimate_convection_coefficient(ConvectionType::Natural, true);
        let h_nat_liq = estimate_convection_coefficient(ConvectionType::Natural, false);
        let h_forced_gas = estimate_convection_coefficient(ConvectionType::Forced, true);
        let h_forced_liq = estimate_convection_coefficient(ConvectionType::Forced, false);

        // Verify ordering: liquids > gases, forced > natural
        assert!(h_nat_liq > h_nat_gas);
        assert!(h_forced_gas > h_nat_gas);
        assert!(h_forced_liq > h_forced_gas);
    }

    #[test]
    fn test_convection_invalid_inputs() {
        assert!(convection_rate(0.0, 1.0, 350.0, 300.0).is_err()); // Zero h
        assert!(convection_rate(25.0, 0.0, 350.0, 300.0).is_err()); // Zero area
        assert!(convection_rate(25.0, 1.0, 0.0, 300.0).is_err()); // Zero temperature
    }

    // ======================= Radiation Tests =======================

    #[test]
    fn test_radiation_stefan_boltzmann() {
        // Blackbody at 500K in 300K surroundings
        let q = radiation_rate(1.0, 1.0, 500.0, 300.0).unwrap();

        // Q = σ * (500⁴ - 300⁴)
        let expected = STEFAN_BOLTZMANN * (500.0_f64.powi(4) - 300.0_f64.powi(4));
        assert!((q - expected).abs() < 1.0);
    }

    #[test]
    fn test_radiation_with_emissivity() {
        // Gray body with ε = 0.8
        let q = radiation_rate(0.8, 1.0, 500.0, 300.0).unwrap();
        let q_black = radiation_rate(1.0, 1.0, 500.0, 300.0).unwrap();

        assert!((q - 0.8 * q_black).abs() < 1.0);
    }

    #[test]
    fn test_radiation_between_surfaces() {
        // Two surfaces at 500K and 300K with ε = 0.9 each
        let q = radiation_between_surfaces(0.9, 0.9, 1.0, 500.0, 300.0, 1.0).unwrap();

        // ε_eff = 1 / (1/0.9 + 1/0.9 - 1) = 1 / (2.222 - 1) = 0.818
        let eps_eff = 1.0 / (1.0 / 0.9 + 1.0 / 0.9 - 1.0);
        let expected = eps_eff * STEFAN_BOLTZMANN * (500.0_f64.powi(4) - 300.0_f64.powi(4));
        assert!((q - expected).abs() < 1.0);
    }

    #[test]
    fn test_blackbody_emissive_power() {
        let e = blackbody_emissive_power(1000.0).unwrap();
        // σ * T⁴ = 5.67e-8 * 10^12 = 56700 W/m²
        assert!((e - 56703.7).abs() < 1.0);
    }

    #[test]
    fn test_radiation_net_exchange() {
        // If T_surface < T_surroundings, heat flows into the surface
        let q = radiation_rate(1.0, 1.0, 250.0, 300.0).unwrap();
        assert!(q < 0.0, "Net radiation should be negative (heat into surface)");
    }

    #[test]
    fn test_radiation_invalid_inputs() {
        assert!(radiation_rate(0.0, 1.0, 500.0, 300.0).is_err()); // Zero emissivity
        assert!(radiation_rate(1.5, 1.0, 500.0, 300.0).is_err()); // Emissivity > 1
        assert!(radiation_rate(1.0, 0.0, 500.0, 300.0).is_err()); // Zero area
        assert!(radiation_rate(1.0, 1.0, 0.0, 300.0).is_err()); // Zero temperature

        assert!(radiation_between_surfaces(0.9, 0.9, 1.0, 500.0, 300.0, 1.5).is_err()); // View factor > 1
    }

    // ======================= Thermal Resistance Tests =======================

    #[test]
    fn test_thermal_resistance_conduction() {
        // R = L / (k * A)
        let r = thermal_resistance_conduction(50.0, 2.0, 0.1).unwrap();
        // 0.1 / (50 * 2) = 0.001 K/W
        assert!((r - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_thermal_resistance_convection() {
        // R = 1 / (h * A)
        let r = thermal_resistance_convection(25.0, 2.0).unwrap();
        // 1 / (25 * 2) = 0.02 K/W
        assert!((r - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_thermal_resistance_series() {
        let resistances = vec![0.01, 0.02, 0.03];
        let r_total = thermal_resistance_series(&resistances);
        assert!((r_total - 0.06).abs() < 1e-6);
    }

    #[test]
    fn test_thermal_resistance_parallel() {
        // Two equal resistances in parallel: R_total = R/2
        let resistances = vec![0.1, 0.1];
        let r_total = thermal_resistance_parallel(&resistances).unwrap();
        assert!((r_total - 0.05).abs() < 1e-6);

        // Different resistances: 1/R = 1/0.1 + 1/0.2 = 15 → R = 0.0667
        let resistances2 = vec![0.1, 0.2];
        let r_total2 = thermal_resistance_parallel(&resistances2).unwrap();
        assert!((r_total2 - 0.0667).abs() < 0.001);
    }

    #[test]
    fn test_overall_heat_transfer_coefficient() {
        // U = 1 / (A * R)
        let u = overall_heat_transfer_coefficient(0.05, 2.0).unwrap();
        // 1 / (2 * 0.05) = 10 W/(m²·K)
        assert!((u - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_heat_rate_from_resistance() {
        // Q = ΔT / R
        let q = heat_rate_from_resistance(100.0, 0.05).unwrap();
        // 100 / 0.05 = 2000 W
        assert!((q - 2000.0).abs() < 1e-6);
    }
}
