// src/thermodynamics.rs

use crate::errors::PhysicsError;

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