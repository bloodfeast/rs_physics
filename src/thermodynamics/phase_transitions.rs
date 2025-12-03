//! Phase Transitions and Latent Heat
//!
//! This module provides functions for calculating heat requirements for phase changes,
//! determining phase states, and working with vapor pressure and phase diagrams.

use crate::utils::PhysicsError;
use super::constants::R;
use super::validation::{validate_temperature_kelvin, validate_pressure, validate_mass};

// ============================================================================
// Phase Transition Data
// ============================================================================

/// Data describing the phase transition properties of a substance
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseTransitionData {
    /// Melting point at 1 atm in Kelvin
    pub melting_point: f64,
    /// Boiling point at 1 atm in Kelvin
    pub boiling_point: f64,
    /// Latent heat of fusion (melting) in J/kg
    pub latent_heat_fusion: f64,
    /// Latent heat of vaporization (boiling) in J/kg
    pub latent_heat_vaporization: f64,
    /// Specific heat capacity of solid phase in J/(kg·K)
    pub specific_heat_solid: f64,
    /// Specific heat capacity of liquid phase in J/(kg·K)
    pub specific_heat_liquid: f64,
    /// Specific heat capacity of gas phase in J/(kg·K)
    pub specific_heat_gas: f64,
    /// Triple point (T in K, P in Pa)
    pub triple_point: (f64, f64),
    /// Critical point (T in K, P in Pa)
    pub critical_point: (f64, f64),
}

impl PhaseTransitionData {
    /// Phase transition data for water (H2O)
    pub fn water() -> Self {
        Self {
            melting_point: 273.15,        // 0°C
            boiling_point: 373.15,        // 100°C
            latent_heat_fusion: 334000.0, // 334 kJ/kg
            latent_heat_vaporization: 2260000.0, // 2260 kJ/kg
            specific_heat_solid: 2090.0,  // ice
            specific_heat_liquid: 4186.0, // water
            specific_heat_gas: 2010.0,    // steam
            triple_point: (273.16, 611.73),
            critical_point: (647.096, 22064000.0), // 22.064 MPa
        }
    }

    /// Phase transition data for ethanol (C2H5OH)
    pub fn ethanol() -> Self {
        Self {
            melting_point: 159.0,         // -114°C
            boiling_point: 351.5,         // 78.5°C
            latent_heat_fusion: 108000.0, // 108 kJ/kg
            latent_heat_vaporization: 841000.0, // 841 kJ/kg
            specific_heat_solid: 1000.0,
            specific_heat_liquid: 2440.0,
            specific_heat_gas: 1400.0,
            triple_point: (159.0, 4.3e-7),
            critical_point: (513.9, 6137000.0),
        }
    }

    /// Phase transition data for nitrogen (N2)
    pub fn nitrogen() -> Self {
        Self {
            melting_point: 63.15,
            boiling_point: 77.36,
            latent_heat_fusion: 25700.0,  // 25.7 kJ/kg
            latent_heat_vaporization: 199000.0, // 199 kJ/kg
            specific_heat_solid: 1040.0,
            specific_heat_liquid: 1040.0,
            specific_heat_gas: 1040.0,
            triple_point: (63.15, 12500.0),
            critical_point: (126.2, 3390000.0),
        }
    }

    /// Phase transition data for iron (Fe)
    pub fn iron() -> Self {
        Self {
            melting_point: 1811.0,        // 1538°C
            boiling_point: 3134.0,        // 2861°C
            latent_heat_fusion: 247000.0, // 247 kJ/kg
            latent_heat_vaporization: 6090000.0, // 6090 kJ/kg
            specific_heat_solid: 449.0,
            specific_heat_liquid: 824.0,
            specific_heat_gas: 450.0,
            triple_point: (1811.0, 0.0),  // Approximate
            critical_point: (9340.0, 10000000.0), // Approximate
        }
    }

    /// Phase transition data for copper (Cu)
    pub fn copper() -> Self {
        Self {
            melting_point: 1357.77,       // 1084.62°C
            boiling_point: 2835.0,        // 2562°C
            latent_heat_fusion: 206000.0, // 206 kJ/kg
            latent_heat_vaporization: 4730000.0, // 4730 kJ/kg
            specific_heat_solid: 385.0,
            specific_heat_liquid: 495.0,
            specific_heat_gas: 300.0,
            triple_point: (1357.77, 0.0),
            critical_point: (8000.0, 10000000.0),
        }
    }

    /// Phase transition data for ammonia (NH3)
    pub fn ammonia() -> Self {
        Self {
            melting_point: 195.4,
            boiling_point: 239.8,
            latent_heat_fusion: 332000.0,
            latent_heat_vaporization: 1370000.0,
            specific_heat_solid: 2060.0,
            specific_heat_liquid: 4700.0,
            specific_heat_gas: 2060.0,
            triple_point: (195.4, 6100.0),
            critical_point: (405.5, 11280000.0),
        }
    }

    /// Creates custom phase transition data
    pub fn new(
        melting_point: f64,
        boiling_point: f64,
        latent_heat_fusion: f64,
        latent_heat_vaporization: f64,
        specific_heat_solid: f64,
        specific_heat_liquid: f64,
        specific_heat_gas: f64,
    ) -> Result<Self, PhysicsError> {
        if melting_point <= 0.0 {
            return Err(PhysicsError::CalculationError(
                "Melting point must be positive".to_string()
            ));
        }
        if boiling_point <= melting_point {
            return Err(PhysicsError::CalculationError(
                "Boiling point must be greater than melting point".to_string()
            ));
        }
        if latent_heat_fusion <= 0.0 {
            return Err(PhysicsError::CalculationError(
                "Latent heat of fusion must be positive".to_string()
            ));
        }
        if latent_heat_vaporization <= 0.0 {
            return Err(PhysicsError::CalculationError(
                "Latent heat of vaporization must be positive".to_string()
            ));
        }

        Ok(Self {
            melting_point,
            boiling_point,
            latent_heat_fusion,
            latent_heat_vaporization,
            specific_heat_solid,
            specific_heat_liquid,
            specific_heat_gas,
            triple_point: (melting_point, 101325.0), // Default to 1 atm
            critical_point: (boiling_point * 1.5, 10000000.0), // Rough estimate
        })
    }
}

// ============================================================================
// Phase Determination
// ============================================================================

/// Phase states of matter
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseState {
    Solid,
    Liquid,
    Gas,
    /// Supercritical fluid (above critical point)
    Supercritical,
    /// At phase transition temperature
    Transitioning,
}

/// Determines the phase of a substance at given temperature and pressure
///
/// This is a simplified model that assumes:
/// - Solid below melting point
/// - Liquid between melting and boiling points
/// - Gas above boiling point
/// - Supercritical above critical point
///
/// For more accurate results, use a phase diagram or equation of state.
///
/// # Arguments
/// * `temperature` - Temperature in K
/// * `pressure` - Pressure in Pa
/// * `data` - Phase transition data for the substance
///
/// # Returns
/// The phase state of the substance
pub fn determine_phase(
    temperature: f64,
    pressure: f64,
    data: &PhaseTransitionData,
) -> Result<PhaseState, PhysicsError> {
    validate_temperature_kelvin(temperature)?;
    validate_pressure(pressure)?;

    // Check supercritical
    if temperature > data.critical_point.0 && pressure > data.critical_point.1 {
        return Ok(PhaseState::Supercritical);
    }

    // Check if at transition points (within 0.1 K tolerance)
    if (temperature - data.melting_point).abs() < 0.1 {
        return Ok(PhaseState::Transitioning);
    }
    if (temperature - data.boiling_point).abs() < 0.1 {
        return Ok(PhaseState::Transitioning);
    }

    // Simple phase determination
    if temperature < data.melting_point {
        Ok(PhaseState::Solid)
    } else if temperature < data.boiling_point {
        Ok(PhaseState::Liquid)
    } else {
        Ok(PhaseState::Gas)
    }
}

/// Determines the phase with pressure-dependent transition points
///
/// Uses simplified Clausius-Clapeyron relationship for better accuracy.
pub fn determine_phase_with_pressure(
    temperature: f64,
    pressure: f64,
    data: &PhaseTransitionData,
    molar_mass: f64,
) -> Result<PhaseState, PhysicsError> {
    validate_temperature_kelvin(temperature)?;
    validate_pressure(pressure)?;

    // Check supercritical
    if temperature > data.critical_point.0 && pressure > data.critical_point.1 {
        return Ok(PhaseState::Supercritical);
    }

    // Estimate boiling point at given pressure using Clausius-Clapeyron
    let t_boil = boiling_point_at_pressure(pressure, data, molar_mass).unwrap_or(data.boiling_point);

    if temperature < data.melting_point {
        Ok(PhaseState::Solid)
    } else if temperature < t_boil {
        Ok(PhaseState::Liquid)
    } else {
        Ok(PhaseState::Gas)
    }
}

// ============================================================================
// Heat Calculations for Phase Changes
// ============================================================================

/// Calculates heat required for a phase change
///
/// Q = m * L
///
/// # Arguments
/// * `mass` - Mass of substance in kg
/// * `latent_heat` - Latent heat in J/kg
///
/// # Returns
/// Heat required in Joules
pub fn heat_for_phase_change(mass: f64, latent_heat: f64) -> Result<f64, PhysicsError> {
    validate_mass(mass)?;
    if latent_heat <= 0.0 {
        return Err(PhysicsError::CalculationError(
            "Latent heat must be positive".to_string()
        ));
    }
    Ok(mass * latent_heat)
}

/// Calculates heat required to melt (fusion)
pub fn heat_for_melting(mass: f64, data: &PhaseTransitionData) -> Result<f64, PhysicsError> {
    heat_for_phase_change(mass, data.latent_heat_fusion)
}

/// Calculates heat required to vaporize (boiling)
pub fn heat_for_vaporization(mass: f64, data: &PhaseTransitionData) -> Result<f64, PhysicsError> {
    heat_for_phase_change(mass, data.latent_heat_vaporization)
}

/// Calculates heat required to freeze (negative of melting)
pub fn heat_for_freezing(mass: f64, data: &PhaseTransitionData) -> Result<f64, PhysicsError> {
    Ok(-heat_for_melting(mass, data)?)
}

/// Calculates heat required to condense (negative of vaporization)
pub fn heat_for_condensation(mass: f64, data: &PhaseTransitionData) -> Result<f64, PhysicsError> {
    Ok(-heat_for_vaporization(mass, data)?)
}

/// Calculates heat required to sublimate (solid to gas)
///
/// Sublimation heat ≈ fusion heat + vaporization heat
pub fn heat_for_sublimation(mass: f64, data: &PhaseTransitionData) -> Result<f64, PhysicsError> {
    validate_mass(mass)?;
    Ok(mass * (data.latent_heat_fusion + data.latent_heat_vaporization))
}

/// Calculates heat required for temperature change within a single phase
pub fn heat_for_temperature_change(
    mass: f64,
    specific_heat: f64,
    delta_t: f64,
) -> Result<f64, PhysicsError> {
    validate_mass(mass)?;
    if specific_heat <= 0.0 {
        return Err(PhysicsError::CalculationError(
            "Specific heat must be positive".to_string()
        ));
    }
    Ok(mass * specific_heat * delta_t)
}

/// Calculates total heat required to change temperature from T1 to T2,
/// including any phase transitions encountered along the way.
///
/// This handles heating or cooling through melting, boiling, freezing, or condensing.
///
/// # Arguments
/// * `mass` - Mass of substance in kg
/// * `t_initial` - Initial temperature in K
/// * `t_final` - Final temperature in K
/// * `data` - Phase transition data for the substance
///
/// # Returns
/// Total heat required in Joules (positive = heat added, negative = heat removed)
pub fn total_heat_for_temperature_range(
    mass: f64,
    t_initial: f64,
    t_final: f64,
    data: &PhaseTransitionData,
) -> Result<f64, PhysicsError> {
    validate_mass(mass)?;
    validate_temperature_kelvin(t_initial)?;
    validate_temperature_kelvin(t_final)?;

    let mut total_heat = 0.0;

    if t_final > t_initial {
        // Heating
        let mut t = t_initial;

        // Heat solid phase
        if t < data.melting_point {
            let t_end = t_final.min(data.melting_point);
            total_heat += mass * data.specific_heat_solid * (t_end - t);
            t = t_end;
        }

        // Melting transition
        if t == data.melting_point && t_final > data.melting_point {
            total_heat += mass * data.latent_heat_fusion;
            // Skip past transition point
        }

        // Heat liquid phase
        if t <= data.melting_point && t_final > data.melting_point {
            t = data.melting_point;
        }
        if t >= data.melting_point && t < data.boiling_point {
            let t_end = t_final.min(data.boiling_point);
            if t_end > t {
                total_heat += mass * data.specific_heat_liquid * (t_end - t);
            }
            t = t_end;
        }

        // Boiling transition
        if t == data.boiling_point && t_final > data.boiling_point {
            total_heat += mass * data.latent_heat_vaporization;
        }

        // Heat gas phase
        if t_final > data.boiling_point {
            let t_start = data.boiling_point.max(t_initial);
            total_heat += mass * data.specific_heat_gas * (t_final - t_start.max(data.boiling_point));
        }
    } else {
        // Cooling (reverse process, heat is negative)
        let mut t = t_initial;

        // Cool gas phase
        if t > data.boiling_point {
            let t_end = t_final.max(data.boiling_point);
            total_heat += mass * data.specific_heat_gas * (t_end - t);
            t = t_end;
        }

        // Condensation transition
        if t == data.boiling_point && t_final < data.boiling_point {
            total_heat -= mass * data.latent_heat_vaporization;
        }

        // Cool liquid phase
        if t <= data.boiling_point && t_final < data.boiling_point {
            t = data.boiling_point;
        }
        if t <= data.boiling_point && t > data.melting_point {
            let t_end = t_final.max(data.melting_point);
            if t_end < t {
                total_heat += mass * data.specific_heat_liquid * (t_end - t);
            }
            t = t_end;
        }

        // Freezing transition
        if t == data.melting_point && t_final < data.melting_point {
            total_heat -= mass * data.latent_heat_fusion;
        }

        // Cool solid phase
        if t_final < data.melting_point {
            let t_start = data.melting_point.min(t_initial);
            total_heat += mass * data.specific_heat_solid * (t_final - t_start.min(data.melting_point));
        }
    }

    Ok(total_heat)
}

// ============================================================================
// Vapor Pressure and Clausius-Clapeyron
// ============================================================================

/// Calculates vapor pressure using the Clausius-Clapeyron equation
///
/// ln(P2/P1) = (L_vap * M / R) * (1/T1 - 1/T2)
///
/// # Arguments
/// * `temperature` - Temperature at which to find vapor pressure in K
/// * `reference_point` - Reference (T, P) point, typically normal boiling point
/// * `latent_heat_vaporization` - Latent heat in J/kg
/// * `molar_mass` - Molar mass in kg/mol
///
/// # Returns
/// Vapor pressure at the given temperature in Pa
pub fn vapor_pressure(
    temperature: f64,
    reference_point: (f64, f64),
    latent_heat_vaporization: f64,
    molar_mass: f64,
) -> Result<f64, PhysicsError> {
    validate_temperature_kelvin(temperature)?;
    validate_temperature_kelvin(reference_point.0)?;
    validate_pressure(reference_point.1)?;

    if latent_heat_vaporization <= 0.0 {
        return Err(PhysicsError::CalculationError(
            "Latent heat must be positive".to_string()
        ));
    }
    if molar_mass <= 0.0 {
        return Err(PhysicsError::CalculationError(
            "Molar mass must be positive".to_string()
        ));
    }

    let (t_ref, p_ref) = reference_point;
    let l_molar = latent_heat_vaporization * molar_mass; // J/mol

    let exponent = (l_molar / R) * (1.0 / t_ref - 1.0 / temperature);
    let pressure = p_ref * exponent.exp();

    Ok(pressure)
}

/// Calculates boiling point at a given pressure using Clausius-Clapeyron
///
/// # Arguments
/// * `pressure` - Pressure at which to find boiling point in Pa
/// * `data` - Phase transition data for the substance
/// * `molar_mass` - Molar mass in kg/mol
///
/// # Returns
/// Boiling point at the given pressure in K
pub fn boiling_point_at_pressure(
    pressure: f64,
    data: &PhaseTransitionData,
    molar_mass: f64,
) -> Result<f64, PhysicsError> {
    validate_pressure(pressure)?;

    if molar_mass <= 0.0 {
        return Err(PhysicsError::CalculationError(
            "Molar mass must be positive".to_string()
        ));
    }

    // Reference point is normal boiling point at 1 atm
    let p_ref = 101325.0; // 1 atm
    let t_ref = data.boiling_point;
    let l_molar = data.latent_heat_vaporization * molar_mass; // J/mol

    // From Clausius-Clapeyron: ln(P/P_ref) = (L/R) * (1/T_ref - 1/T)
    // Solving for T: 1/T = 1/T_ref - (R/L) * ln(P/P_ref)
    let ln_ratio = (pressure / p_ref).ln();
    let inv_t = 1.0 / t_ref - (R / l_molar) * ln_ratio;

    if inv_t <= 0.0 {
        return Err(PhysicsError::CalculationError(
            "Invalid pressure range for boiling point calculation".to_string()
        ));
    }

    Ok(1.0 / inv_t)
}

/// Calculates the vapor pressure of water using a simplified Antoine equation
///
/// This provides better accuracy than Clausius-Clapeyron for water.
///
/// # Arguments
/// * `temperature` - Temperature in K (valid range: 273-473 K)
///
/// # Returns
/// Vapor pressure of water in Pa
pub fn water_vapor_pressure(temperature: f64) -> Result<f64, PhysicsError> {
    validate_temperature_kelvin(temperature)?;

    if temperature < 273.15 || temperature > 473.15 {
        return Err(PhysicsError::CalculationError(
            format!("Temperature {} K out of valid range (273-473 K)", temperature)
        ));
    }

    // Antoine equation constants for water (T in Celsius, P in mmHg)
    let t_celsius = temperature - 273.15;
    let a = 8.07131;
    let b = 1730.63;
    let c = 233.426;

    // log10(P_mmHg) = A - B/(C + T)
    let log_p_mmhg = a - b / (c + t_celsius);
    let p_mmhg = 10.0_f64.powf(log_p_mmhg);

    // Convert mmHg to Pa (1 mmHg = 133.322 Pa)
    Ok(p_mmhg * 133.322)
}

/// Calculates relative humidity
///
/// RH = (P_vapor / P_saturation) * 100%
///
/// # Arguments
/// * `actual_vapor_pressure` - Actual water vapor pressure in Pa
/// * `saturation_vapor_pressure` - Saturation vapor pressure at the temperature in Pa
///
/// # Returns
/// Relative humidity as a percentage (0-100)
pub fn relative_humidity(
    actual_vapor_pressure: f64,
    saturation_vapor_pressure: f64,
) -> Result<f64, PhysicsError> {
    if saturation_vapor_pressure <= 0.0 {
        return Err(PhysicsError::CalculationError(
            "Saturation vapor pressure must be positive".to_string()
        ));
    }
    if actual_vapor_pressure < 0.0 {
        return Err(PhysicsError::CalculationError(
            "Actual vapor pressure cannot be negative".to_string()
        ));
    }

    let rh = (actual_vapor_pressure / saturation_vapor_pressure) * 100.0;
    Ok(rh.min(100.0)) // Cap at 100%
}

/// Calculates dew point temperature
///
/// The temperature at which air becomes saturated with water vapor.
///
/// # Arguments
/// * `temperature` - Current temperature in K
/// * `relative_humidity` - Relative humidity as percentage (0-100)
///
/// # Returns
/// Dew point temperature in K
pub fn dew_point(temperature: f64, relative_humidity_pct: f64) -> Result<f64, PhysicsError> {
    validate_temperature_kelvin(temperature)?;

    if relative_humidity_pct <= 0.0 || relative_humidity_pct > 100.0 {
        return Err(PhysicsError::CalculationError(
            format!("Relative humidity must be 0-100%, got {}%", relative_humidity_pct)
        ));
    }

    let t_celsius = temperature - 273.15;

    // Magnus formula constants
    let a = 17.27;
    let b = 237.7;

    let gamma = (a * t_celsius) / (b + t_celsius) + (relative_humidity_pct / 100.0).ln();
    let dew_point_c = (b * gamma) / (a - gamma);

    Ok(dew_point_c + 273.15)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ======================= PhaseTransitionData Tests =======================

    #[test]
    fn test_water_phase_data() {
        let water = PhaseTransitionData::water();
        assert!((water.melting_point - 273.15).abs() < 0.01);
        assert!((water.boiling_point - 373.15).abs() < 0.01);
        assert!(water.latent_heat_fusion > 300000.0);
        assert!(water.latent_heat_vaporization > 2000000.0);
    }

    #[test]
    fn test_iron_phase_data() {
        let iron = PhaseTransitionData::iron();
        assert!(iron.melting_point > 1800.0); // ~1538°C
        assert!(iron.boiling_point > 3000.0); // ~2861°C
    }

    #[test]
    fn test_custom_phase_data_valid() {
        let data = PhaseTransitionData::new(
            300.0, 400.0, 100000.0, 500000.0, 1000.0, 2000.0, 1500.0
        );
        assert!(data.is_ok());
    }

    #[test]
    fn test_custom_phase_data_invalid() {
        // Boiling point < melting point
        let data = PhaseTransitionData::new(
            400.0, 300.0, 100000.0, 500000.0, 1000.0, 2000.0, 1500.0
        );
        assert!(data.is_err());

        // Negative latent heat
        let data2 = PhaseTransitionData::new(
            300.0, 400.0, -100000.0, 500000.0, 1000.0, 2000.0, 1500.0
        );
        assert!(data2.is_err());
    }

    // ======================= Phase Determination Tests =======================

    #[test]
    fn test_determine_phase_water() {
        let water = PhaseTransitionData::water();

        // Ice
        let phase = determine_phase(250.0, 101325.0, &water).unwrap();
        assert_eq!(phase, PhaseState::Solid);

        // Liquid water
        let phase = determine_phase(300.0, 101325.0, &water).unwrap();
        assert_eq!(phase, PhaseState::Liquid);

        // Steam
        let phase = determine_phase(400.0, 101325.0, &water).unwrap();
        assert_eq!(phase, PhaseState::Gas);
    }

    #[test]
    fn test_determine_phase_at_transition() {
        let water = PhaseTransitionData::water();

        // At melting point
        let phase = determine_phase(273.15, 101325.0, &water).unwrap();
        assert_eq!(phase, PhaseState::Transitioning);

        // At boiling point
        let phase = determine_phase(373.15, 101325.0, &water).unwrap();
        assert_eq!(phase, PhaseState::Transitioning);
    }

    #[test]
    fn test_determine_phase_supercritical() {
        let water = PhaseTransitionData::water();

        // Above critical point
        let phase = determine_phase(700.0, 25000000.0, &water).unwrap();
        assert_eq!(phase, PhaseState::Supercritical);
    }

    // ======================= Heat Calculation Tests =======================

    #[test]
    fn test_heat_for_phase_change() {
        let heat = heat_for_phase_change(1.0, 334000.0).unwrap();
        assert!((heat - 334000.0).abs() < 1.0);
    }

    #[test]
    fn test_heat_for_melting_water() {
        let water = PhaseTransitionData::water();
        let heat = heat_for_melting(1.0, &water).unwrap();
        assert!((heat - 334000.0).abs() < 1000.0);
    }

    #[test]
    fn test_heat_for_vaporization_water() {
        let water = PhaseTransitionData::water();
        let heat = heat_for_vaporization(1.0, &water).unwrap();
        assert!((heat - 2260000.0).abs() < 10000.0);
    }

    #[test]
    fn test_heat_for_freezing_is_negative() {
        let water = PhaseTransitionData::water();
        let heat = heat_for_freezing(1.0, &water).unwrap();
        assert!(heat < 0.0);
        assert!((heat + 334000.0).abs() < 1000.0);
    }

    #[test]
    fn test_heat_for_sublimation() {
        let water = PhaseTransitionData::water();
        let heat = heat_for_sublimation(1.0, &water).unwrap();
        // Should be sum of fusion + vaporization
        let expected = 334000.0 + 2260000.0;
        assert!((heat - expected).abs() < 10000.0);
    }

    #[test]
    fn test_heat_for_temperature_change() {
        // Heat 1 kg of water by 10 K
        let heat = heat_for_temperature_change(1.0, 4186.0, 10.0).unwrap();
        assert!((heat - 41860.0).abs() < 10.0);
    }

    #[test]
    fn test_total_heat_heating_through_melting() {
        let water = PhaseTransitionData::water();

        // Heat 1 kg from -10°C to +10°C (through melting)
        let t_initial = 263.15; // -10°C
        let t_final = 283.15;   // +10°C

        let heat = total_heat_for_temperature_range(1.0, t_initial, t_final, &water).unwrap();

        // Expected: solid heating + melting + liquid heating
        let expected_solid = 2090.0 * 10.0;  // Heat ice from -10 to 0°C
        let expected_melt = 334000.0;        // Melt the ice
        let expected_liquid = 4186.0 * 10.0; // Heat water from 0 to 10°C
        let expected_total = expected_solid + expected_melt + expected_liquid;

        assert!((heat - expected_total).abs() < 1000.0,
            "Expected {} J, got {} J", expected_total, heat);
    }

    #[test]
    fn test_total_heat_heating_through_boiling() {
        let water = PhaseTransitionData::water();

        // Heat 1 kg from 90°C to 110°C (through boiling)
        let t_initial = 363.15; // 90°C
        let t_final = 383.15;   // 110°C

        let heat = total_heat_for_temperature_range(1.0, t_initial, t_final, &water).unwrap();

        // Should include vaporization heat
        assert!(heat > 2000000.0, "Heat should be > 2 MJ due to vaporization");
    }

    // ======================= Vapor Pressure Tests =======================

    #[test]
    fn test_vapor_pressure() {
        let water = PhaseTransitionData::water();
        let molar_mass = 0.018; // kg/mol for water

        // At boiling point, vapor pressure should be ~1 atm
        let p = vapor_pressure(373.15, (373.15, 101325.0), water.latent_heat_vaporization, molar_mass).unwrap();
        assert!((p - 101325.0).abs() < 100.0);
    }

    #[test]
    fn test_vapor_pressure_increases_with_temperature() {
        let water = PhaseTransitionData::water();
        let molar_mass = 0.018;

        let p_low = vapor_pressure(350.0, (373.15, 101325.0), water.latent_heat_vaporization, molar_mass).unwrap();
        let p_high = vapor_pressure(400.0, (373.15, 101325.0), water.latent_heat_vaporization, molar_mass).unwrap();

        assert!(p_high > p_low);
    }

    #[test]
    fn test_boiling_point_at_pressure() {
        let water = PhaseTransitionData::water();
        let molar_mass = 0.018;

        // At lower pressure, boiling point is lower
        let t_low = boiling_point_at_pressure(50000.0, &water, molar_mass).unwrap();
        let t_atm = boiling_point_at_pressure(101325.0, &water, molar_mass).unwrap();
        let t_high = boiling_point_at_pressure(200000.0, &water, molar_mass).unwrap();

        assert!(t_low < t_atm);
        assert!(t_atm < t_high);
        assert!((t_atm - 373.15).abs() < 1.0); // At 1 atm should be ~100°C
    }

    #[test]
    fn test_water_vapor_pressure() {
        // At 100°C, should be approximately 1 atm
        let p = water_vapor_pressure(373.15).unwrap();
        assert!((p - 101325.0).abs() < 5000.0);

        // At 20°C, should be about 2.3 kPa
        let p_20c = water_vapor_pressure(293.15).unwrap();
        assert!(p_20c > 2000.0 && p_20c < 3000.0);
    }

    #[test]
    fn test_relative_humidity() {
        let rh = relative_humidity(1500.0, 2337.0).unwrap();
        assert!(rh > 60.0 && rh < 70.0);

        // Saturated air
        let rh_sat = relative_humidity(2337.0, 2337.0).unwrap();
        assert!((rh_sat - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_dew_point() {
        // At 100% humidity, dew point equals temperature
        let dp = dew_point(293.15, 100.0).unwrap();
        assert!((dp - 293.15).abs() < 1.0);

        // At lower humidity, dew point is lower
        let dp_low = dew_point(293.15, 50.0).unwrap();
        assert!(dp_low < 293.15);
    }

    #[test]
    fn test_dew_point_invalid() {
        assert!(dew_point(293.15, 0.0).is_err());
        assert!(dew_point(293.15, 150.0).is_err());
    }
}
