// src/thermodynamics.rs

use crate::utils::PhysicsError;
use super::constants::R;
use super::validation::{validate_temperature_kelvin, validate_pressure, validate_volume, validate_moles};

/// Type of ideal gas based on degrees of freedom
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GasType {
    /// Monatomic gas (He, Ne, Ar) - 3 translational degrees of freedom
    Monatomic,
    /// Diatomic gas (N2, O2, H2) - 5 degrees of freedom (3 translational + 2 rotational)
    Diatomic,
    /// Polyatomic gas with specified degrees of freedom
    Polyatomic(usize),
}

impl GasType {
    /// Returns the degrees of freedom for this gas type
    ///
    /// - Monatomic: 3 (translational only)
    /// - Diatomic: 5 (translational + rotational)
    /// - Polyatomic: specified value
    #[inline]
    pub fn degrees_of_freedom(&self) -> usize {
        match self {
            GasType::Monatomic => 3,
            GasType::Diatomic => 5,
            GasType::Polyatomic(f) => *f,
        }
    }

    /// Returns the heat capacity ratio (gamma = Cp/Cv)
    ///
    /// γ = (f + 2) / f where f is degrees of freedom
    #[inline]
    pub fn gamma(&self) -> f64 {
        let f = self.degrees_of_freedom() as f64;
        (f + 2.0) / f
    }

    /// Returns the molar heat capacity at constant volume (Cv)
    ///
    /// Cv = (f/2) * R
    #[inline]
    pub fn molar_cv(&self) -> f64 {
        let f = self.degrees_of_freedom() as f64;
        (f / 2.0) * R
    }

    /// Returns the molar heat capacity at constant pressure (Cp)
    ///
    /// Cp = Cv + R = ((f + 2)/2) * R
    #[inline]
    pub fn molar_cp(&self) -> f64 {
        let f = self.degrees_of_freedom() as f64;
        ((f + 2.0) / 2.0) * R
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thermodynamic {
    pub temperature: f64,
    pub pressure: f64,
    pub volume: f64,
}

impl Thermodynamic {
    /// Creates a new `Thermodynamic` instance with the given temperature, pressure, and volume.
    /// # Arguments
    /// * `temperature` - The temperature of the system in Kelvin (K).
    /// * `pressure` - The pressure of the system in Pascals (Pa).
    /// * `volume` - The volume of the system in cubic meters (m³).
    ///
    /// # Return
    /// Returns a `Result` containing the new `Thermodynamic` instance if successful,
    /// or a `PhysicsError` if the input parameters are invalid.
    ///
    /// # Errors
    /// Returns an error if:
    /// * The temperature is less than or equal to zero.
    /// * The pressure is less than or equal to zero.
    /// * The volume is less than or equal to zero.
    ///
    /// # Examples
    /// ```
    /// use rs_physics::thermodynamics::Thermodynamic;
    ///
    /// let state = Thermodynamic::new(300.0, 101325.0, 1.0).unwrap();
    /// ```
    pub fn new(temperature: f64, pressure: f64, volume: f64) -> Result<Self, PhysicsError> {
        if temperature <= 0.0 {
            return Err(PhysicsError::CalculationError("Temperature must be positive".to_string()));
        }
        if pressure <= 0.0 {
            return Err(PhysicsError::CalculationError("Pressure must be positive".to_string()));
        }
        if volume <= 0.0 {
            return Err(PhysicsError::InvalidVolume);
        }
        Ok(Self {
            temperature,
            pressure,
            volume,
        })
    }
}

/// Calculates the heat transfer through a material.
/// # Arguments
/// * `thermal_conductivity` - The thermal conductivity of the material in W/(m·K).
/// * `area` - The cross-sectional area of the material in m².
/// * `temperature_difference` - The temperature difference across the material in Kelvin (K).
/// * `thickness` - The thickness of the material in meters (m).
///
/// # Return
/// Returns a `Result` containing the calculated heat transfer rate in Watts (W) if successful,
/// or a `PhysicsError` if the input parameters are invalid.
///
/// # Errors
/// Returns an error if:
/// * The thermal conductivity is less than or equal to zero.
/// * The area is less than or equal to zero.
/// * The thickness is less than or equal to zero.
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::calculate_heat_transfer;
///
/// let heat_transfer = calculate_heat_transfer(0.5, 1.0, 10.0, 0.1).unwrap();
/// ```
pub fn calculate_heat_transfer(thermal_conductivity: f64, area: f64, temperature_difference: f64, thickness: f64) -> Result<f64, PhysicsError> {
    if thermal_conductivity <= 0.0 {
        return Err(PhysicsError::InvalidCoefficient);
    }
    if area <= 0.0 || thickness <= 0.0 {
        return Err(PhysicsError::InvalidArea);
    }
    if temperature_difference < 0.0 {
        return Err(PhysicsError::CalculationError("Temperature difference cannot be negative".to_string()));
    }
    Ok(thermal_conductivity * area * temperature_difference / thickness)
}

/// Calculates the change in entropy between two thermodynamic states.
/// # Arguments
/// * `initial_state` - A reference to the initial `Thermodynamic` state.
/// * `final_state` - A reference to the final `Thermodynamic` state.
/// * `heat_added` - The heat added to the system in Joules (J).
///
/// # Return
/// Returns a `Result` containing the calculated entropy change in J/K if successful,
/// or a `PhysicsError` if the input parameters are invalid.
///
/// # Errors
/// Returns an error if:
/// * The temperature of either the initial or final state is less than or equal to zero.
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::{Thermodynamic, calculate_entropy_change};
///
/// let initial = Thermodynamic::new(300.0, 101325.0, 1.0).unwrap();
/// let final_state = Thermodynamic::new(350.0, 101325.0, 1.2).unwrap();
/// let entropy_change = calculate_entropy_change(&initial, &final_state, 1000.0).unwrap();
/// ```
pub fn calculate_entropy_change(initial_state: &Thermodynamic, final_state: &Thermodynamic, heat_added: f64) -> Result<f64, PhysicsError> {
    if initial_state.temperature <= 0.0 || final_state.temperature <= 0.0 {
        return Err(PhysicsError::CalculationError("Temperature must be positive".to_string()));
    }
    // TODO: Maybe add a check for heat_added < 0.0 here? or should it be allowed to calculate a negative entropy change in this function?
    // TODO: Maybe it would be better to have a increase_entropy and decrease_entropy function instead?
    Ok(heat_added * (1.0 / initial_state.temperature - 1.0 / final_state.temperature))
}

/// Calculates the work done by a system during a thermodynamic process.
/// # Arguments
/// * `initial_state` - A reference to the initial `Thermodynamic` state.
/// * `final_state` - A reference to the final `Thermodynamic` state.
///
/// # Return
/// Returns a `Result` containing the calculated work done in Joules (J) if successful,
/// or a `PhysicsError` if the input parameters are invalid.
///
/// # Errors
/// Returns an error if:
/// * The pressure of either the initial or final state is less than or equal to zero.
/// * The volume of either the initial or final state is less than or equal to zero.
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::{Thermodynamic, calculate_work_done};
///
/// let initial = Thermodynamic::new(300.0, 101325.0, 1.0).unwrap();
/// let final_state = Thermodynamic::new(300.0, 101325.0, 1.2).unwrap();
/// let work = calculate_work_done(&initial, &final_state).unwrap();
/// ```
pub fn calculate_work_done(initial_state: &Thermodynamic, final_state: &Thermodynamic) -> Result<f64, PhysicsError> {
    if initial_state.pressure <= 0.0 || final_state.pressure <= 0.0 {
        return Err(PhysicsError::CalculationError("Pressure must be positive".to_string()));
    }
    if initial_state.volume <= 0.0 || final_state.volume <= 0.0 {
        return Err(PhysicsError::InvalidArea);
    }
    Ok(0.5 * (initial_state.pressure + final_state.pressure) * (final_state.volume - initial_state.volume))
}

/// Calculates the thermal efficiency of a heat engine.
/// # Arguments
/// * `work_output` - The work output of the heat engine in Joules (J).
/// * `heat_input` - The heat input to the heat engine in Joules (J).
///
/// # Return
/// Returns a `Result` containing the calculated efficiency (dimensionless) if successful,
/// or a `PhysicsError` if the input parameters are invalid.
///
/// # Errors
/// Returns an error if:
/// * The heat input is less than or equal to zero.
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::calculate_efficiency;
///
/// let efficiency = calculate_efficiency(300.0, 1000.0).unwrap();
/// ```
pub fn calculate_efficiency(work_output: f64, heat_input: f64) -> Result<f64, PhysicsError> {
    if heat_input <= 0.0 {
        return Err(PhysicsError::CalculationError("Heat input must be positive".to_string()));
    }
    Ok(work_output / heat_input)
}

/// Calculates the specific heat capacity of a substance.
/// # Arguments
/// * `mass` - The mass of the substance in kilograms (kg).
/// * `temperature_change` - The change in temperature in Kelvin (K).
/// * `heat_added` - The heat added to the substance in Joules (J).
///
/// # Return
/// Returns a `Result` containing the calculated specific heat capacity in J/(kg·K) if successful,
/// or a `PhysicsError` if the input parameters are invalid.
///
/// # Errors
/// Returns an error if:
/// * The mass is less than or equal to zero.
/// * The temperature change is zero.
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::calculate_specific_heat_capacity;
///
/// let specific_heat = calculate_specific_heat_capacity(1.0, 10.0, 4180.0).unwrap();
/// ```
pub fn calculate_specific_heat_capacity(mass: f64, temperature_change: f64, heat_added: f64) -> Result<f64, PhysicsError> {
    if mass <= 0.0 {
        return Err(PhysicsError::InvalidMass);
    }
    if temperature_change == 0.0 {
        return Err(PhysicsError::CalculationError("Temperature change cannot be zero".to_string()));
    }
    Ok(heat_added / (mass * temperature_change))
}

// ============================================================================
// Ideal Gas Law Functions
// ============================================================================

/// Calculates pressure using the ideal gas law: P = nRT/V
///
/// # Arguments
/// * `n` - Number of moles
/// * `temperature` - Temperature in Kelvin
/// * `volume` - Volume in cubic meters
///
/// # Returns
/// Pressure in Pascals
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::ideal_gas_pressure;
///
/// // 1 mole at 300 K in 0.025 m³ ≈ 100 kPa
/// let p = ideal_gas_pressure(1.0, 300.0, 0.025).unwrap();
/// assert!((p - 99858.0).abs() < 100.0);
/// ```
pub fn ideal_gas_pressure(n: f64, temperature: f64, volume: f64) -> Result<f64, PhysicsError> {
    validate_moles(n)?;
    validate_temperature_kelvin(temperature)?;
    validate_volume(volume)?;

    Ok(n * R * temperature / volume)
}

/// Calculates volume using the ideal gas law: V = nRT/P
///
/// # Arguments
/// * `n` - Number of moles
/// * `temperature` - Temperature in Kelvin
/// * `pressure` - Pressure in Pascals
///
/// # Returns
/// Volume in cubic meters
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::ideal_gas_volume;
///
/// // 1 mole at 300 K and 101325 Pa ≈ 0.0246 m³
/// let v = ideal_gas_volume(1.0, 300.0, 101325.0).unwrap();
/// assert!((v - 0.0246).abs() < 0.001);
/// ```
pub fn ideal_gas_volume(n: f64, temperature: f64, pressure: f64) -> Result<f64, PhysicsError> {
    validate_moles(n)?;
    validate_temperature_kelvin(temperature)?;
    validate_pressure(pressure)?;

    Ok(n * R * temperature / pressure)
}

/// Calculates temperature using the ideal gas law: T = PV/(nR)
///
/// # Arguments
/// * `pressure` - Pressure in Pascals
/// * `volume` - Volume in cubic meters
/// * `n` - Number of moles
///
/// # Returns
/// Temperature in Kelvin
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::ideal_gas_temperature;
///
/// let t = ideal_gas_temperature(101325.0, 0.0246, 1.0).unwrap();
/// assert!((t - 300.0).abs() < 1.0);
/// ```
pub fn ideal_gas_temperature(pressure: f64, volume: f64, n: f64) -> Result<f64, PhysicsError> {
    validate_pressure(pressure)?;
    validate_volume(volume)?;

    if n <= 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("Number of moles must be positive for temperature calculation, got {} mol", n)
        ));
    }

    Ok(pressure * volume / (n * R))
}

/// Calculates the number of moles using the ideal gas law: n = PV/(RT)
///
/// # Arguments
/// * `pressure` - Pressure in Pascals
/// * `volume` - Volume in cubic meters
/// * `temperature` - Temperature in Kelvin
///
/// # Returns
/// Number of moles
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::ideal_gas_moles;
///
/// let n = ideal_gas_moles(101325.0, 0.0246, 300.0).unwrap();
/// assert!((n - 1.0).abs() < 0.01);
/// ```
pub fn ideal_gas_moles(pressure: f64, volume: f64, temperature: f64) -> Result<f64, PhysicsError> {
    validate_pressure(pressure)?;
    validate_volume(volume)?;
    validate_temperature_kelvin(temperature)?;

    Ok(pressure * volume / (R * temperature))
}

// ============================================================================
// Internal Energy Functions
// ============================================================================

/// Calculates the internal energy of an ideal gas
///
/// U = (f/2) * n * R * T
///
/// where f is degrees of freedom:
/// - Monatomic: f = 3
/// - Diatomic: f = 5
/// - Polyatomic: f = specified
///
/// # Arguments
/// * `n` - Number of moles
/// * `temperature` - Temperature in Kelvin
/// * `gas_type` - Type of gas (determines degrees of freedom)
///
/// # Returns
/// Internal energy in Joules
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::{internal_energy, GasType};
///
/// // 1 mole of monatomic gas at 300 K
/// let u = internal_energy(1.0, 300.0, GasType::Monatomic).unwrap();
/// // U = (3/2) * 1 * 8.314 * 300 ≈ 3742 J
/// assert!((u - 3742.0).abs() < 10.0);
/// ```
pub fn internal_energy(n: f64, temperature: f64, gas_type: GasType) -> Result<f64, PhysicsError> {
    validate_moles(n)?;
    validate_temperature_kelvin(temperature)?;

    let f = gas_type.degrees_of_freedom() as f64;
    Ok((f / 2.0) * n * R * temperature)
}

/// Calculates the change in internal energy for an ideal gas
///
/// ΔU = (f/2) * n * R * ΔT = n * Cv * ΔT
///
/// # Arguments
/// * `n` - Number of moles
/// * `delta_t` - Temperature change in Kelvin
/// * `gas_type` - Type of gas
///
/// # Returns
/// Change in internal energy in Joules
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::{internal_energy_change, GasType};
///
/// // 1 mole of diatomic gas, temperature increases by 50 K
/// let delta_u = internal_energy_change(1.0, 50.0, GasType::Diatomic).unwrap();
/// // ΔU = (5/2) * 1 * 8.314 * 50 ≈ 1039 J
/// assert!((delta_u - 1039.0).abs() < 10.0);
/// ```
pub fn internal_energy_change(n: f64, delta_t: f64, gas_type: GasType) -> Result<f64, PhysicsError> {
    validate_moles(n)?;

    let cv = gas_type.molar_cv();
    Ok(n * cv * delta_t)
}

// ============================================================================
// Enthalpy Functions
// ============================================================================

/// Calculates the enthalpy of an ideal gas
///
/// H = U + PV = U + nRT
///
/// For an ideal gas: H = (f/2 + 1) * n * R * T = n * Cp * T
///
/// # Arguments
/// * `n` - Number of moles
/// * `temperature` - Temperature in Kelvin
/// * `gas_type` - Type of gas
///
/// # Returns
/// Enthalpy in Joules
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::{enthalpy, GasType};
///
/// // 1 mole of monatomic gas at 300 K
/// let h = enthalpy(1.0, 300.0, GasType::Monatomic).unwrap();
/// // H = (5/2) * 1 * 8.314 * 300 ≈ 6236 J
/// assert!((h - 6236.0).abs() < 10.0);
/// ```
pub fn enthalpy(n: f64, temperature: f64, gas_type: GasType) -> Result<f64, PhysicsError> {
    validate_moles(n)?;
    validate_temperature_kelvin(temperature)?;

    let cp = gas_type.molar_cp();
    Ok(n * cp * temperature)
}

/// Calculates the change in enthalpy for an ideal gas
///
/// ΔH = n * Cp * ΔT
///
/// # Arguments
/// * `n` - Number of moles
/// * `delta_t` - Temperature change in Kelvin
/// * `gas_type` - Type of gas
///
/// # Returns
/// Change in enthalpy in Joules
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::{enthalpy_change, GasType};
///
/// // 1 mole of diatomic gas, temperature increases by 50 K
/// let delta_h = enthalpy_change(1.0, 50.0, GasType::Diatomic).unwrap();
/// // ΔH = (7/2) * 1 * 8.314 * 50 ≈ 1455 J
/// assert!((delta_h - 1455.0).abs() < 10.0);
/// ```
pub fn enthalpy_change(n: f64, delta_t: f64, gas_type: GasType) -> Result<f64, PhysicsError> {
    validate_moles(n)?;

    let cp = gas_type.molar_cp();
    Ok(n * cp * delta_t)
}

/// Calculates the enthalpy from a thermodynamic state
///
/// H = U + PV where U = n * Cv * T
///
/// # Arguments
/// * `state` - The thermodynamic state (P, V, T)
/// * `n` - Number of moles
/// * `gas_type` - Type of gas
///
/// # Returns
/// Enthalpy in Joules
pub fn enthalpy_from_state(state: &Thermodynamic, n: f64, gas_type: GasType) -> Result<f64, PhysicsError> {
    validate_moles(n)?;

    let u = internal_energy(n, state.temperature, gas_type)?;
    Ok(u + state.pressure * state.volume)
}

// ============================================================================
// Heat Capacity Functions
// ============================================================================

/// Returns the molar heat capacity at constant volume for a given gas type
///
/// Cv = (f/2) * R
///
/// # Arguments
/// * `gas_type` - Type of gas
///
/// # Returns
/// Molar heat capacity at constant volume in J/(mol·K)
#[inline]
pub fn molar_heat_capacity_cv(gas_type: GasType) -> f64 {
    gas_type.molar_cv()
}

/// Returns the molar heat capacity at constant pressure for a given gas type
///
/// Cp = Cv + R = ((f+2)/2) * R
///
/// # Arguments
/// * `gas_type` - Type of gas
///
/// # Returns
/// Molar heat capacity at constant pressure in J/(mol·K)
#[inline]
pub fn molar_heat_capacity_cp(gas_type: GasType) -> f64 {
    gas_type.molar_cp()
}

/// Returns the heat capacity ratio (gamma) for a given gas type
///
/// γ = Cp/Cv = (f+2)/f
///
/// # Arguments
/// * `gas_type` - Type of gas
///
/// # Returns
/// Heat capacity ratio (dimensionless)
#[inline]
pub fn heat_capacity_ratio(gas_type: GasType) -> f64 {
    gas_type.gamma()
}