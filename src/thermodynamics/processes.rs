//! Thermodynamic Processes
//!
//! This module provides functions for calculating work, heat, and state changes
//! for common thermodynamic processes: isothermal, adiabatic, isobaric, isochoric,
//! and polytropic.

use crate::utils::PhysicsError;
use super::constants::R;
use super::thermodynamics::{GasType, Thermodynamic};
use super::validation::{validate_temperature_kelvin, validate_pressure, validate_volume, validate_moles};

/// Types of thermodynamic processes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProcessType {
    /// Isothermal process (constant temperature)
    Isothermal,
    /// Adiabatic process (no heat exchange, Q = 0)
    Adiabatic,
    /// Isobaric process (constant pressure)
    Isobaric,
    /// Isochoric process (constant volume)
    Isochoric,
    /// Polytropic process (PV^n = constant)
    Polytropic(f64),
}

/// Result of executing a thermodynamic process
#[derive(Debug, Clone)]
pub struct ProcessResult {
    /// Work done by the system in Joules (positive = work done by system)
    pub work: f64,
    /// Heat added to the system in Joules (positive = heat into system)
    pub heat: f64,
    /// Change in internal energy in Joules
    pub delta_internal_energy: f64,
    /// Final thermodynamic state
    pub final_state: Thermodynamic,
}

// ============================================================================
// Isothermal Process (T = constant)
// ============================================================================

/// Calculates work done in an isothermal (constant temperature) process
///
/// W = n * R * T * ln(V_final / V_initial)
///
/// For isothermal ideal gas: PV = constant, so W = nRT * ln(V2/V1)
///
/// # Arguments
/// * `n` - Number of moles
/// * `temperature` - Constant temperature in K
/// * `v_initial` - Initial volume in m³
/// * `v_final` - Final volume in m³
///
/// # Returns
/// Work done by the gas in Joules (positive for expansion)
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::isothermal_work;
///
/// // 1 mol at 300K expanding from 0.01 to 0.02 m³
/// let w = isothermal_work(1.0, 300.0, 0.01, 0.02).unwrap();
/// assert!(w > 0.0); // Expansion does positive work
/// ```
pub fn isothermal_work(n: f64, temperature: f64, v_initial: f64, v_final: f64) -> Result<f64, PhysicsError> {
    validate_moles(n)?;
    validate_temperature_kelvin(temperature)?;
    validate_volume(v_initial)?;
    validate_volume(v_final)?;

    Ok(n * R * temperature * (v_final / v_initial).ln())
}

/// Calculates heat transfer in an isothermal process
///
/// For isothermal process: ΔU = 0, so Q = W
///
/// # Arguments
/// * `n` - Number of moles
/// * `temperature` - Constant temperature in K
/// * `v_initial` - Initial volume in m³
/// * `v_final` - Final volume in m³
///
/// # Returns
/// Heat added to the system in Joules
pub fn isothermal_heat(n: f64, temperature: f64, v_initial: f64, v_final: f64) -> Result<f64, PhysicsError> {
    // For isothermal: Q = W (since ΔU = 0)
    isothermal_work(n, temperature, v_initial, v_final)
}

/// Calculates final pressure for isothermal process
///
/// P_final = P_initial * V_initial / V_final (from PV = constant)
///
/// # Arguments
/// * `p_initial` - Initial pressure in Pa
/// * `v_initial` - Initial volume in m³
/// * `v_final` - Final volume in m³
///
/// # Returns
/// Final pressure in Pa
pub fn isothermal_final_pressure(p_initial: f64, v_initial: f64, v_final: f64) -> Result<f64, PhysicsError> {
    validate_pressure(p_initial)?;
    validate_volume(v_initial)?;
    validate_volume(v_final)?;

    Ok(p_initial * v_initial / v_final)
}

/// Calculates final volume for isothermal process given final pressure
///
/// V_final = P_initial * V_initial / P_final
pub fn isothermal_final_volume(p_initial: f64, v_initial: f64, p_final: f64) -> Result<f64, PhysicsError> {
    validate_pressure(p_initial)?;
    validate_volume(v_initial)?;
    validate_pressure(p_final)?;

    Ok(p_initial * v_initial / p_final)
}

// ============================================================================
// Adiabatic Process (Q = 0)
// ============================================================================

/// Calculates work done in an adiabatic (no heat exchange) process
///
/// W = (P_i * V_i - P_f * V_f) / (γ - 1) = n * Cv * (T_i - T_f)
///
/// # Arguments
/// * `p_initial` - Initial pressure in Pa
/// * `v_initial` - Initial volume in m³
/// * `p_final` - Final pressure in Pa
/// * `v_final` - Final volume in m³
/// * `gamma` - Heat capacity ratio (Cp/Cv)
///
/// # Returns
/// Work done by the gas in Joules
pub fn adiabatic_work(
    p_initial: f64,
    v_initial: f64,
    p_final: f64,
    v_final: f64,
    gamma: f64,
) -> Result<f64, PhysicsError> {
    validate_pressure(p_initial)?;
    validate_volume(v_initial)?;
    validate_pressure(p_final)?;
    validate_volume(v_final)?;

    if gamma <= 1.0 {
        return Err(PhysicsError::CalculationError(
            format!("Gamma must be greater than 1, got {}", gamma)
        ));
    }

    Ok((p_initial * v_initial - p_final * v_final) / (gamma - 1.0))
}

/// Calculates work done in adiabatic process using moles and temperatures
///
/// W = n * Cv * (T_initial - T_final) = -ΔU
///
/// # Arguments
/// * `n` - Number of moles
/// * `t_initial` - Initial temperature in K
/// * `t_final` - Final temperature in K
/// * `gas_type` - Type of gas (determines Cv)
///
/// # Returns
/// Work done by the gas in Joules
pub fn adiabatic_work_from_temps(
    n: f64,
    t_initial: f64,
    t_final: f64,
    gas_type: GasType,
) -> Result<f64, PhysicsError> {
    validate_moles(n)?;
    validate_temperature_kelvin(t_initial)?;
    validate_temperature_kelvin(t_final)?;

    let cv = gas_type.molar_cv();
    Ok(n * cv * (t_initial - t_final))
}

/// Calculates final temperature for adiabatic process (given volume change)
///
/// T_final = T_initial * (V_initial / V_final)^(γ-1)
///
/// # Arguments
/// * `t_initial` - Initial temperature in K
/// * `v_initial` - Initial volume in m³
/// * `v_final` - Final volume in m³
/// * `gamma` - Heat capacity ratio (Cp/Cv)
///
/// # Returns
/// Final temperature in K
pub fn adiabatic_final_temperature(
    t_initial: f64,
    v_initial: f64,
    v_final: f64,
    gamma: f64,
) -> Result<f64, PhysicsError> {
    validate_temperature_kelvin(t_initial)?;
    validate_volume(v_initial)?;
    validate_volume(v_final)?;

    if gamma <= 1.0 {
        return Err(PhysicsError::CalculationError(
            format!("Gamma must be greater than 1, got {}", gamma)
        ));
    }

    Ok(t_initial * (v_initial / v_final).powf(gamma - 1.0))
}

/// Calculates final temperature for adiabatic process (given pressure change)
///
/// T_final = T_initial * (P_final / P_initial)^((γ-1)/γ)
pub fn adiabatic_final_temperature_from_pressure(
    t_initial: f64,
    p_initial: f64,
    p_final: f64,
    gamma: f64,
) -> Result<f64, PhysicsError> {
    validate_temperature_kelvin(t_initial)?;
    validate_pressure(p_initial)?;
    validate_pressure(p_final)?;

    if gamma <= 1.0 {
        return Err(PhysicsError::CalculationError(
            format!("Gamma must be greater than 1, got {}", gamma)
        ));
    }

    let exponent = (gamma - 1.0) / gamma;
    Ok(t_initial * (p_final / p_initial).powf(exponent))
}

/// Calculates final pressure for adiabatic process
///
/// P_final = P_initial * (V_initial / V_final)^γ
///
/// # Arguments
/// * `p_initial` - Initial pressure in Pa
/// * `v_initial` - Initial volume in m³
/// * `v_final` - Final volume in m³
/// * `gamma` - Heat capacity ratio (Cp/Cv)
///
/// # Returns
/// Final pressure in Pa
pub fn adiabatic_final_pressure(
    p_initial: f64,
    v_initial: f64,
    v_final: f64,
    gamma: f64,
) -> Result<f64, PhysicsError> {
    validate_pressure(p_initial)?;
    validate_volume(v_initial)?;
    validate_volume(v_final)?;

    if gamma <= 1.0 {
        return Err(PhysicsError::CalculationError(
            format!("Gamma must be greater than 1, got {}", gamma)
        ));
    }

    Ok(p_initial * (v_initial / v_final).powf(gamma))
}

/// Calculates final volume for adiabatic process given final pressure
///
/// V_final = V_initial * (P_initial / P_final)^(1/γ)
pub fn adiabatic_final_volume(
    v_initial: f64,
    p_initial: f64,
    p_final: f64,
    gamma: f64,
) -> Result<f64, PhysicsError> {
    validate_volume(v_initial)?;
    validate_pressure(p_initial)?;
    validate_pressure(p_final)?;

    if gamma <= 1.0 {
        return Err(PhysicsError::CalculationError(
            format!("Gamma must be greater than 1, got {}", gamma)
        ));
    }

    Ok(v_initial * (p_initial / p_final).powf(1.0 / gamma))
}

// ============================================================================
// Isobaric Process (P = constant)
// ============================================================================

/// Calculates work done in an isobaric (constant pressure) process
///
/// W = P * (V_final - V_initial) = P * ΔV
///
/// # Arguments
/// * `pressure` - Constant pressure in Pa
/// * `v_initial` - Initial volume in m³
/// * `v_final` - Final volume in m³
///
/// # Returns
/// Work done by the gas in Joules
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::isobaric_work;
///
/// // Expansion at 101325 Pa from 0.01 to 0.02 m³
/// let w = isobaric_work(101325.0, 0.01, 0.02).unwrap();
/// assert!((w - 1013.25).abs() < 1.0);
/// ```
pub fn isobaric_work(pressure: f64, v_initial: f64, v_final: f64) -> Result<f64, PhysicsError> {
    validate_pressure(pressure)?;
    validate_volume(v_initial)?;
    validate_volume(v_final)?;

    Ok(pressure * (v_final - v_initial))
}

/// Calculates heat transfer in an isobaric process
///
/// Q = n * Cp * ΔT
///
/// # Arguments
/// * `n` - Number of moles
/// * `t_initial` - Initial temperature in K
/// * `t_final` - Final temperature in K
/// * `gas_type` - Type of gas (determines Cp)
///
/// # Returns
/// Heat added to the system in Joules
pub fn isobaric_heat(n: f64, t_initial: f64, t_final: f64, gas_type: GasType) -> Result<f64, PhysicsError> {
    validate_moles(n)?;
    validate_temperature_kelvin(t_initial)?;
    validate_temperature_kelvin(t_final)?;

    let cp = gas_type.molar_cp();
    Ok(n * cp * (t_final - t_initial))
}

/// Calculates final volume for isobaric process
///
/// V_final = V_initial * (T_final / T_initial) (from V/T = constant)
pub fn isobaric_final_volume(v_initial: f64, t_initial: f64, t_final: f64) -> Result<f64, PhysicsError> {
    validate_volume(v_initial)?;
    validate_temperature_kelvin(t_initial)?;
    validate_temperature_kelvin(t_final)?;

    Ok(v_initial * t_final / t_initial)
}

/// Calculates final temperature for isobaric process given final volume
///
/// T_final = T_initial * (V_final / V_initial)
pub fn isobaric_final_temperature(t_initial: f64, v_initial: f64, v_final: f64) -> Result<f64, PhysicsError> {
    validate_temperature_kelvin(t_initial)?;
    validate_volume(v_initial)?;
    validate_volume(v_final)?;

    Ok(t_initial * v_final / v_initial)
}

// ============================================================================
// Isochoric Process (V = constant)
// ============================================================================

/// Work done in an isochoric (constant volume) process is always zero
///
/// W = ∫P dV = 0 (since dV = 0)
#[inline]
pub fn isochoric_work() -> f64 {
    0.0
}

/// Calculates heat transfer in an isochoric process
///
/// Q = n * Cv * ΔT = ΔU (since W = 0)
///
/// # Arguments
/// * `n` - Number of moles
/// * `t_initial` - Initial temperature in K
/// * `t_final` - Final temperature in K
/// * `gas_type` - Type of gas (determines Cv)
///
/// # Returns
/// Heat added to the system in Joules
pub fn isochoric_heat(n: f64, t_initial: f64, t_final: f64, gas_type: GasType) -> Result<f64, PhysicsError> {
    validate_moles(n)?;
    validate_temperature_kelvin(t_initial)?;
    validate_temperature_kelvin(t_final)?;

    let cv = gas_type.molar_cv();
    Ok(n * cv * (t_final - t_initial))
}

/// Calculates final pressure for isochoric process
///
/// P_final = P_initial * (T_final / T_initial) (from P/T = constant)
pub fn isochoric_final_pressure(p_initial: f64, t_initial: f64, t_final: f64) -> Result<f64, PhysicsError> {
    validate_pressure(p_initial)?;
    validate_temperature_kelvin(t_initial)?;
    validate_temperature_kelvin(t_final)?;

    Ok(p_initial * t_final / t_initial)
}

/// Calculates final temperature for isochoric process given final pressure
///
/// T_final = T_initial * (P_final / P_initial)
pub fn isochoric_final_temperature(t_initial: f64, p_initial: f64, p_final: f64) -> Result<f64, PhysicsError> {
    validate_temperature_kelvin(t_initial)?;
    validate_pressure(p_initial)?;
    validate_pressure(p_final)?;

    Ok(t_initial * p_final / p_initial)
}

// ============================================================================
// Polytropic Process (PV^n = constant)
// ============================================================================

/// Calculates work done in a polytropic process
///
/// W = (P_i * V_i - P_f * V_f) / (n - 1) for n ≠ 1
/// W = P_i * V_i * ln(V_f / V_i) for n = 1 (isothermal)
///
/// # Arguments
/// * `p_initial` - Initial pressure in Pa
/// * `v_initial` - Initial volume in m³
/// * `p_final` - Final pressure in Pa
/// * `v_final` - Final volume in m³
/// * `n` - Polytropic exponent
///
/// # Returns
/// Work done by the gas in Joules
pub fn polytropic_work(
    p_initial: f64,
    v_initial: f64,
    p_final: f64,
    v_final: f64,
    n: f64,
) -> Result<f64, PhysicsError> {
    validate_pressure(p_initial)?;
    validate_volume(v_initial)?;
    validate_pressure(p_final)?;
    validate_volume(v_final)?;

    if (n - 1.0).abs() < 1e-10 {
        // n ≈ 1, use isothermal formula
        Ok(p_initial * v_initial * (v_final / v_initial).ln())
    } else {
        Ok((p_initial * v_initial - p_final * v_final) / (n - 1.0))
    }
}

/// Calculates final pressure for polytropic process
///
/// P_final = P_initial * (V_initial / V_final)^n
pub fn polytropic_final_pressure(
    p_initial: f64,
    v_initial: f64,
    v_final: f64,
    n: f64,
) -> Result<f64, PhysicsError> {
    validate_pressure(p_initial)?;
    validate_volume(v_initial)?;
    validate_volume(v_final)?;

    Ok(p_initial * (v_initial / v_final).powf(n))
}

/// Calculates final temperature for polytropic process
///
/// T_final = T_initial * (V_initial / V_final)^(n-1)
pub fn polytropic_final_temperature(
    t_initial: f64,
    v_initial: f64,
    v_final: f64,
    n: f64,
) -> Result<f64, PhysicsError> {
    validate_temperature_kelvin(t_initial)?;
    validate_volume(v_initial)?;
    validate_volume(v_final)?;

    Ok(t_initial * (v_initial / v_final).powf(n - 1.0))
}

/// Determines the polytropic exponent from two states
///
/// n = ln(P1/P2) / ln(V2/V1)
pub fn polytropic_exponent(
    p1: f64,
    v1: f64,
    p2: f64,
    v2: f64,
) -> Result<f64, PhysicsError> {
    validate_pressure(p1)?;
    validate_volume(v1)?;
    validate_pressure(p2)?;
    validate_volume(v2)?;

    let ln_v_ratio = (v2 / v1).ln();
    if ln_v_ratio.abs() < 1e-10 {
        return Err(PhysicsError::CalculationError(
            "Cannot determine exponent when volumes are equal".to_string()
        ));
    }

    Ok((p1 / p2).ln() / ln_v_ratio)
}

// ============================================================================
// General Process Execution
// ============================================================================

/// Executes a thermodynamic process and returns the result
///
/// # Arguments
/// * `initial_state` - Initial thermodynamic state
/// * `process` - Type of process to execute
/// * `n_moles` - Number of moles
/// * `gas_type` - Type of gas
/// * `final_volume` - Final volume (required for isothermal, adiabatic, isobaric)
/// * `final_pressure` - Final pressure (alternative specification)
/// * `final_temperature` - Final temperature (for isobaric, isochoric)
///
/// # Returns
/// ProcessResult containing work, heat, ΔU, and final state
pub fn execute_process(
    initial_state: &Thermodynamic,
    process: ProcessType,
    n_moles: f64,
    gas_type: GasType,
    final_volume: Option<f64>,
    final_temperature: Option<f64>,
) -> Result<ProcessResult, PhysicsError> {
    validate_moles(n_moles)?;

    let gamma = gas_type.gamma();
    let cv = gas_type.molar_cv();

    match process {
        ProcessType::Isothermal => {
            let v_f = final_volume.ok_or_else(|| {
                PhysicsError::CalculationError("Final volume required for isothermal process".to_string())
            })?;
            validate_volume(v_f)?;

            let work = isothermal_work(n_moles, initial_state.temperature, initial_state.volume, v_f)?;
            let p_f = isothermal_final_pressure(initial_state.pressure, initial_state.volume, v_f)?;

            Ok(ProcessResult {
                work,
                heat: work, // Q = W for isothermal
                delta_internal_energy: 0.0, // ΔU = 0 for isothermal
                final_state: Thermodynamic::new(initial_state.temperature, p_f, v_f)?,
            })
        }

        ProcessType::Adiabatic => {
            let v_f = final_volume.ok_or_else(|| {
                PhysicsError::CalculationError("Final volume required for adiabatic process".to_string())
            })?;
            validate_volume(v_f)?;

            let t_f = adiabatic_final_temperature(initial_state.temperature, initial_state.volume, v_f, gamma)?;
            let p_f = adiabatic_final_pressure(initial_state.pressure, initial_state.volume, v_f, gamma)?;
            let delta_u = n_moles * cv * (t_f - initial_state.temperature);
            // For adiabatic: Q = 0, so from first law ΔU = Q - W = -W, thus W = -ΔU
            let work = -delta_u;

            Ok(ProcessResult {
                work,
                heat: 0.0, // Q = 0 for adiabatic
                delta_internal_energy: delta_u,
                final_state: Thermodynamic::new(t_f, p_f, v_f)?,
            })
        }

        ProcessType::Isobaric => {
            let t_f = final_temperature.ok_or_else(|| {
                PhysicsError::CalculationError("Final temperature required for isobaric process".to_string())
            })?;
            validate_temperature_kelvin(t_f)?;

            let v_f = isobaric_final_volume(initial_state.volume, initial_state.temperature, t_f)?;
            let work = isobaric_work(initial_state.pressure, initial_state.volume, v_f)?;
            let delta_u = n_moles * cv * (t_f - initial_state.temperature);
            // First law: Q = ΔU + W
            let heat = delta_u + work;

            Ok(ProcessResult {
                work,
                heat,
                delta_internal_energy: delta_u,
                final_state: Thermodynamic::new(t_f, initial_state.pressure, v_f)?,
            })
        }

        ProcessType::Isochoric => {
            let t_f = final_temperature.ok_or_else(|| {
                PhysicsError::CalculationError("Final temperature required for isochoric process".to_string())
            })?;
            validate_temperature_kelvin(t_f)?;

            let p_f = isochoric_final_pressure(initial_state.pressure, initial_state.temperature, t_f)?;
            let heat = isochoric_heat(n_moles, initial_state.temperature, t_f, gas_type)?;

            Ok(ProcessResult {
                work: 0.0, // W = 0 for isochoric
                heat,
                delta_internal_energy: heat, // ΔU = Q for isochoric
                final_state: Thermodynamic::new(t_f, p_f, initial_state.volume)?,
            })
        }

        ProcessType::Polytropic(n) => {
            let v_f = final_volume.ok_or_else(|| {
                PhysicsError::CalculationError("Final volume required for polytropic process".to_string())
            })?;
            validate_volume(v_f)?;

            let p_f = polytropic_final_pressure(initial_state.pressure, initial_state.volume, v_f, n)?;
            let t_f = polytropic_final_temperature(initial_state.temperature, initial_state.volume, v_f, n)?;
            let work = polytropic_work(initial_state.pressure, initial_state.volume, p_f, v_f, n)?;
            let delta_u = n_moles * cv * (t_f - initial_state.temperature);
            let heat = delta_u + work; // First law: Q = ΔU + W

            Ok(ProcessResult {
                work,
                heat,
                delta_internal_energy: delta_u,
                final_state: Thermodynamic::new(t_f, p_f, v_f)?,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ======================= Isothermal Tests =======================

    #[test]
    fn test_isothermal_work() {
        // W = nRT * ln(V2/V1)
        // 1 mol at 300K, doubling volume
        let w = isothermal_work(1.0, 300.0, 0.01, 0.02).unwrap();
        let expected = R * 300.0 * (2.0_f64).ln();
        assert!((w - expected).abs() < 1.0);
        assert!(w > 0.0, "Expansion should do positive work");
    }

    #[test]
    fn test_isothermal_compression() {
        // Compression should require negative work (work done ON gas)
        let w = isothermal_work(1.0, 300.0, 0.02, 0.01).unwrap();
        assert!(w < 0.0, "Compression should have negative work");
    }

    #[test]
    fn test_isothermal_heat() {
        // For isothermal: Q = W
        let q = isothermal_heat(1.0, 300.0, 0.01, 0.02).unwrap();
        let w = isothermal_work(1.0, 300.0, 0.01, 0.02).unwrap();
        assert!((q - w).abs() < 1e-10);
    }

    #[test]
    fn test_isothermal_final_pressure() {
        // PV = constant, so P2 = P1 * V1/V2
        let p2 = isothermal_final_pressure(200000.0, 0.01, 0.02).unwrap();
        assert!((p2 - 100000.0).abs() < 1.0);
    }

    #[test]
    fn test_isothermal_final_volume() {
        let v2 = isothermal_final_volume(200000.0, 0.01, 100000.0).unwrap();
        assert!((v2 - 0.02).abs() < 1e-6);
    }

    // ======================= Adiabatic Tests =======================

    #[test]
    fn test_adiabatic_work() {
        // Use PV relationship
        let gamma = 1.4; // Diatomic
        let p1 = 100000.0;
        let v1 = 0.01;
        let v2 = 0.02;
        let p2 = adiabatic_final_pressure(p1, v1, v2, gamma).unwrap();

        let w = adiabatic_work(p1, v1, p2, v2, gamma).unwrap();
        // Expansion should do positive work
        assert!(w > 0.0);
    }

    #[test]
    fn test_adiabatic_final_temperature() {
        // T2 = T1 * (V1/V2)^(γ-1)
        let gamma = 5.0 / 3.0; // Monatomic
        let t2 = adiabatic_final_temperature(300.0, 0.01, 0.02, gamma).unwrap();

        let expected = 300.0 * (0.5_f64).powf(gamma - 1.0);
        assert!((t2 - expected).abs() < 1.0);
        assert!(t2 < 300.0, "Expansion should cool the gas");
    }

    #[test]
    fn test_adiabatic_final_pressure() {
        // P2 = P1 * (V1/V2)^γ
        let gamma = 1.4;
        let p2 = adiabatic_final_pressure(100000.0, 0.01, 0.02, gamma).unwrap();

        let expected = 100000.0 * (0.5_f64).powf(gamma);
        assert!((p2 - expected).abs() < 1.0);
    }

    #[test]
    fn test_adiabatic_pv_gamma_relation() {
        // Verify PV^γ = constant
        let gamma = 1.4;
        let p1 = 100000.0;
        let v1 = 0.01;
        let v2 = 0.02;
        let p2 = adiabatic_final_pressure(p1, v1, v2, gamma).unwrap();

        let pv_gamma_1 = p1 * v1.powf(gamma);
        let pv_gamma_2 = p2 * v2.powf(gamma);

        assert!((pv_gamma_1 - pv_gamma_2).abs() < 1e-6);
    }

    // ======================= Isobaric Tests =======================

    #[test]
    fn test_isobaric_work() {
        // W = P * ΔV
        let w = isobaric_work(100000.0, 0.01, 0.02).unwrap();
        assert!((w - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_isobaric_heat() {
        // Q = n * Cp * ΔT
        let q = isobaric_heat(1.0, 300.0, 400.0, GasType::Monatomic).unwrap();
        let cp = GasType::Monatomic.molar_cp();
        let expected = cp * 100.0;
        assert!((q - expected).abs() < 1.0);
    }

    #[test]
    fn test_isobaric_final_volume() {
        // V2/V1 = T2/T1
        let v2 = isobaric_final_volume(0.01, 300.0, 600.0).unwrap();
        assert!((v2 - 0.02).abs() < 1e-6);
    }

    // ======================= Isochoric Tests =======================

    #[test]
    fn test_isochoric_work() {
        // W = 0 always
        assert_eq!(isochoric_work(), 0.0);
    }

    #[test]
    fn test_isochoric_heat() {
        // Q = n * Cv * ΔT = ΔU
        let q = isochoric_heat(1.0, 300.0, 400.0, GasType::Monatomic).unwrap();
        let cv = GasType::Monatomic.molar_cv();
        let expected = cv * 100.0;
        assert!((q - expected).abs() < 1.0);
    }

    #[test]
    fn test_isochoric_final_pressure() {
        // P2/P1 = T2/T1
        let p2 = isochoric_final_pressure(100000.0, 300.0, 600.0).unwrap();
        assert!((p2 - 200000.0).abs() < 1.0);
    }

    // ======================= Polytropic Tests =======================

    #[test]
    fn test_polytropic_work() {
        // Test with n = 1.3
        let p1 = 100000.0;
        let v1 = 0.01;
        let v2 = 0.02;
        let n = 1.3;
        let p2 = polytropic_final_pressure(p1, v1, v2, n).unwrap();

        let w = polytropic_work(p1, v1, p2, v2, n).unwrap();
        assert!(w > 0.0, "Expansion should do positive work");
    }

    #[test]
    fn test_polytropic_exponent() {
        // Create two states with known n
        let n_expected = 1.3;
        let p1 = 100000.0;
        let v1 = 0.01;
        let v2 = 0.02;
        let p2 = polytropic_final_pressure(p1, v1, v2, n_expected).unwrap();

        let n_calc = polytropic_exponent(p1, v1, p2, v2).unwrap();
        assert!((n_calc - n_expected).abs() < 1e-6);
    }

    #[test]
    fn test_polytropic_isothermal_limit() {
        // n = 1 should give isothermal behavior
        let p1 = 100000.0;
        let v1 = 0.01;
        let v2 = 0.02;
        let p2 = polytropic_final_pressure(p1, v1, v2, 1.0).unwrap();

        // P1*V1 should equal P2*V2
        assert!((p1 * v1 - p2 * v2).abs() < 1.0);
    }

    // ======================= Process Execution Tests =======================

    #[test]
    fn test_execute_isothermal_process() {
        let initial = Thermodynamic::new(300.0, 100000.0, 0.01).unwrap();
        let result = execute_process(
            &initial,
            ProcessType::Isothermal,
            1.0,
            GasType::Diatomic,
            Some(0.02),
            None,
        ).unwrap();

        // ΔU = 0 for isothermal
        assert!((result.delta_internal_energy).abs() < 1e-6);
        // Q = W
        assert!((result.heat - result.work).abs() < 1e-6);
        // Temperature unchanged
        assert!((result.final_state.temperature - 300.0).abs() < 1e-6);
    }

    #[test]
    fn test_execute_adiabatic_process() {
        let initial = Thermodynamic::new(300.0, 100000.0, 0.01).unwrap();
        let result = execute_process(
            &initial,
            ProcessType::Adiabatic,
            1.0,
            GasType::Diatomic,
            Some(0.02),
            None,
        ).unwrap();

        // Q = 0 for adiabatic
        assert!((result.heat).abs() < 1e-6);
        // Temperature decreases for expansion
        assert!(result.final_state.temperature < 300.0);
    }

    #[test]
    fn test_execute_isobaric_process() {
        let initial = Thermodynamic::new(300.0, 100000.0, 0.01).unwrap();
        let result = execute_process(
            &initial,
            ProcessType::Isobaric,
            1.0,
            GasType::Diatomic,
            None,
            Some(600.0),
        ).unwrap();

        // Pressure unchanged
        assert!((result.final_state.pressure - 100000.0).abs() < 1.0);
        // Volume doubled (T doubled)
        assert!((result.final_state.volume - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_execute_isochoric_process() {
        let initial = Thermodynamic::new(300.0, 100000.0, 0.01).unwrap();
        let result = execute_process(
            &initial,
            ProcessType::Isochoric,
            1.0,
            GasType::Diatomic,
            None,
            Some(600.0),
        ).unwrap();

        // W = 0 for isochoric
        assert!((result.work).abs() < 1e-6);
        // Q = ΔU
        assert!((result.heat - result.delta_internal_energy).abs() < 1e-6);
        // Volume unchanged
        assert!((result.final_state.volume - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_first_law_consistency() {
        // Verify ΔU = Q - W for all processes
        let initial = Thermodynamic::new(300.0, 100000.0, 0.01).unwrap();

        let processes = vec![
            (ProcessType::Isothermal, Some(0.02), None),
            (ProcessType::Adiabatic, Some(0.02), None),
            (ProcessType::Isobaric, None, Some(400.0)),
            (ProcessType::Isochoric, None, Some(400.0)),
        ];

        for (process, v, t) in processes {
            let result = execute_process(&initial, process, 1.0, GasType::Diatomic, v, t).unwrap();

            let delta_u_from_first_law = result.heat - result.work;
            assert!(
                (result.delta_internal_energy - delta_u_from_first_law).abs() < 1.0,
                "First law violated for {:?}: ΔU={}, Q-W={}",
                process, result.delta_internal_energy, delta_u_from_first_law
            );
        }
    }
}
