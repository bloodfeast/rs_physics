//! Heat Engines and Thermodynamic Cycles
//!
//! This module provides functions for calculating efficiency and performance
//! of common thermodynamic cycles: Carnot, Otto, Diesel, and refrigeration cycles.

use crate::utils::PhysicsError;
use super::validation::{validate_temperature_kelvin, validate_efficiency};

// ============================================================================
// Carnot Cycle - Theoretical Maximum Efficiency
// ============================================================================

/// Calculates the Carnot efficiency (maximum theoretical efficiency)
///
/// η = 1 - T_cold / T_hot
///
/// The Carnot efficiency represents the maximum possible efficiency for any heat engine
/// operating between two temperature reservoirs.
///
/// # Arguments
/// * `t_hot` - Temperature of hot reservoir in K
/// * `t_cold` - Temperature of cold reservoir in K
///
/// # Returns
/// Efficiency as a dimensionless fraction (0 to 1)
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::carnot_efficiency;
///
/// // Engine between 600K and 300K
/// let eta = carnot_efficiency(600.0, 300.0).unwrap();
/// assert!((eta - 0.5).abs() < 1e-10);
/// ```
pub fn carnot_efficiency(t_hot: f64, t_cold: f64) -> Result<f64, PhysicsError> {
    validate_temperature_kelvin(t_hot)?;
    validate_temperature_kelvin(t_cold)?;

    if t_cold >= t_hot {
        return Err(PhysicsError::CalculationError(
            "Hot temperature must be greater than cold temperature".to_string()
        ));
    }

    Ok(1.0 - t_cold / t_hot)
}

/// Calculates the Coefficient of Performance (COP) for a Carnot refrigerator
///
/// COP_refrigerator = T_cold / (T_hot - T_cold)
///
/// The COP measures how much heat can be removed from the cold reservoir per unit of work input.
///
/// # Arguments
/// * `t_hot` - Temperature of hot reservoir in K
/// * `t_cold` - Temperature of cold reservoir in K
///
/// # Returns
/// COP as a dimensionless number (can be > 1)
pub fn carnot_cop_refrigerator(t_hot: f64, t_cold: f64) -> Result<f64, PhysicsError> {
    validate_temperature_kelvin(t_hot)?;
    validate_temperature_kelvin(t_cold)?;

    if t_cold >= t_hot {
        return Err(PhysicsError::CalculationError(
            "Hot temperature must be greater than cold temperature".to_string()
        ));
    }

    Ok(t_cold / (t_hot - t_cold))
}

/// Calculates the Coefficient of Performance (COP) for a Carnot heat pump
///
/// COP_heat_pump = T_hot / (T_hot - T_cold)
///
/// The COP measures how much heat can be delivered to the hot reservoir per unit of work input.
///
/// # Arguments
/// * `t_hot` - Temperature of hot reservoir in K
/// * `t_cold` - Temperature of cold reservoir in K
///
/// # Returns
/// COP as a dimensionless number (always > 1)
pub fn carnot_cop_heat_pump(t_hot: f64, t_cold: f64) -> Result<f64, PhysicsError> {
    validate_temperature_kelvin(t_hot)?;
    validate_temperature_kelvin(t_cold)?;

    if t_cold >= t_hot {
        return Err(PhysicsError::CalculationError(
            "Hot temperature must be greater than cold temperature".to_string()
        ));
    }

    Ok(t_hot / (t_hot - t_cold))
}

/// Calculates work output per cycle for a Carnot engine
///
/// W = Q_hot * (1 - T_cold / T_hot) = Q_hot * η
///
/// # Arguments
/// * `q_hot` - Heat absorbed from hot reservoir in J
/// * `t_hot` - Temperature of hot reservoir in K
/// * `t_cold` - Temperature of cold reservoir in K
///
/// # Returns
/// Work output in Joules
pub fn carnot_work(q_hot: f64, t_hot: f64, t_cold: f64) -> Result<f64, PhysicsError> {
    let eta = carnot_efficiency(t_hot, t_cold)?;
    Ok(q_hot * eta)
}

/// Calculates heat rejected to cold reservoir in a Carnot cycle
///
/// Q_cold = Q_hot - W = Q_hot * T_cold / T_hot
///
/// # Arguments
/// * `q_hot` - Heat absorbed from hot reservoir in J
/// * `t_hot` - Temperature of hot reservoir in K
/// * `t_cold` - Temperature of cold reservoir in K
///
/// # Returns
/// Heat rejected in Joules
pub fn carnot_heat_rejected(q_hot: f64, t_hot: f64, t_cold: f64) -> Result<f64, PhysicsError> {
    validate_temperature_kelvin(t_hot)?;
    validate_temperature_kelvin(t_cold)?;

    if t_cold >= t_hot {
        return Err(PhysicsError::CalculationError(
            "Hot temperature must be greater than cold temperature".to_string()
        ));
    }

    Ok(q_hot * t_cold / t_hot)
}

/// Carnot cycle struct for more complex calculations
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CarnotCycle {
    /// Temperature of hot reservoir in K
    pub t_hot: f64,
    /// Temperature of cold reservoir in K
    pub t_cold: f64,
}

impl CarnotCycle {
    /// Creates a new Carnot cycle between two temperature reservoirs
    pub fn new(t_hot: f64, t_cold: f64) -> Result<Self, PhysicsError> {
        validate_temperature_kelvin(t_hot)?;
        validate_temperature_kelvin(t_cold)?;

        if t_cold >= t_hot {
            return Err(PhysicsError::CalculationError(
                "Hot temperature must be greater than cold temperature".to_string()
            ));
        }

        Ok(Self { t_hot, t_cold })
    }

    /// Returns the Carnot efficiency
    pub fn efficiency(&self) -> f64 {
        1.0 - self.t_cold / self.t_hot
    }

    /// Returns the COP if used as a refrigerator
    pub fn cop_refrigerator(&self) -> f64 {
        self.t_cold / (self.t_hot - self.t_cold)
    }

    /// Returns the COP if used as a heat pump
    pub fn cop_heat_pump(&self) -> f64 {
        self.t_hot / (self.t_hot - self.t_cold)
    }

    /// Calculates work output for a given heat input
    pub fn work_per_cycle(&self, q_hot: f64) -> f64 {
        q_hot * self.efficiency()
    }

    /// Calculates heat rejected for a given heat input
    pub fn heat_rejected(&self, q_hot: f64) -> f64 {
        q_hot * self.t_cold / self.t_hot
    }
}

// ============================================================================
// Otto Cycle - Spark Ignition (Gasoline) Engines
// ============================================================================

/// Calculates the efficiency of an Otto cycle (gasoline engine)
///
/// η = 1 - 1 / r^(γ-1)
///
/// where r is the compression ratio (V_max / V_min) and γ is the heat capacity ratio.
///
/// # Arguments
/// * `compression_ratio` - V_max / V_min (typically 8:1 to 12:1 for gasoline engines)
/// * `gamma` - Heat capacity ratio Cp/Cv (typically 1.4 for air)
///
/// # Returns
/// Efficiency as a dimensionless fraction (0 to 1)
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::otto_efficiency;
///
/// // Compression ratio 10:1 with γ = 1.4
/// let eta = otto_efficiency(10.0, 1.4).unwrap();
/// assert!(eta > 0.5 && eta < 0.7);
/// ```
pub fn otto_efficiency(compression_ratio: f64, gamma: f64) -> Result<f64, PhysicsError> {
    if compression_ratio <= 1.0 {
        return Err(PhysicsError::CalculationError(
            format!("Compression ratio must be greater than 1, got {}", compression_ratio)
        ));
    }
    if gamma <= 1.0 {
        return Err(PhysicsError::CalculationError(
            format!("Gamma must be greater than 1, got {}", gamma)
        ));
    }

    Ok(1.0 - compression_ratio.powf(1.0 - gamma))
}

/// Calculates the compression ratio required to achieve a target Otto efficiency
///
/// r = (1 - η)^(1/(1-γ))
///
/// # Arguments
/// * `efficiency` - Target efficiency (0 to 1)
/// * `gamma` - Heat capacity ratio Cp/Cv
///
/// # Returns
/// Required compression ratio
pub fn otto_compression_ratio_for_efficiency(efficiency: f64, gamma: f64) -> Result<f64, PhysicsError> {
    validate_efficiency(efficiency)?;
    if gamma <= 1.0 {
        return Err(PhysicsError::CalculationError(
            format!("Gamma must be greater than 1, got {}", gamma)
        ));
    }

    let exponent = 1.0 / (1.0 - gamma);
    Ok((1.0 - efficiency).powf(exponent))
}

/// Otto cycle struct for more complex calculations
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OttoCycle {
    /// Compression ratio V_max / V_min
    pub compression_ratio: f64,
    /// Heat capacity ratio Cp/Cv
    pub gamma: f64,
}

impl OttoCycle {
    /// Creates a new Otto cycle with given compression ratio and gamma
    pub fn new(compression_ratio: f64, gamma: f64) -> Result<Self, PhysicsError> {
        if compression_ratio <= 1.0 {
            return Err(PhysicsError::CalculationError(
                "Compression ratio must be greater than 1".to_string()
            ));
        }
        if gamma <= 1.0 {
            return Err(PhysicsError::CalculationError(
                "Gamma must be greater than 1".to_string()
            ));
        }
        Ok(Self { compression_ratio, gamma })
    }

    /// Returns the Otto cycle efficiency
    pub fn efficiency(&self) -> f64 {
        1.0 - self.compression_ratio.powf(1.0 - self.gamma)
    }

    /// Calculates work output for a given heat input
    pub fn work_per_cycle(&self, q_in: f64) -> f64 {
        q_in * self.efficiency()
    }
}

// ============================================================================
// Diesel Cycle - Compression Ignition Engines
// ============================================================================

/// Calculates the efficiency of a Diesel cycle
///
/// η = 1 - (1 / r^(γ-1)) * ((r_c^γ - 1) / (γ * (r_c - 1)))
///
/// where r is the compression ratio, r_c is the cutoff ratio, and γ is the heat capacity ratio.
///
/// # Arguments
/// * `compression_ratio` - V_max / V_min (typically 14:1 to 25:1 for diesel engines)
/// * `cutoff_ratio` - V_3 / V_2 (ratio of volume after to before heat addition)
/// * `gamma` - Heat capacity ratio Cp/Cv (typically 1.4 for air)
///
/// # Returns
/// Efficiency as a dimensionless fraction (0 to 1)
///
/// # Examples
/// ```
/// use rs_physics::thermodynamics::diesel_efficiency;
///
/// // Compression ratio 20:1, cutoff ratio 2, γ = 1.4
/// let eta = diesel_efficiency(20.0, 2.0, 1.4).unwrap();
/// assert!(eta > 0.5 && eta < 0.8);
/// ```
pub fn diesel_efficiency(compression_ratio: f64, cutoff_ratio: f64, gamma: f64) -> Result<f64, PhysicsError> {
    if compression_ratio <= 1.0 {
        return Err(PhysicsError::CalculationError(
            format!("Compression ratio must be greater than 1, got {}", compression_ratio)
        ));
    }
    if cutoff_ratio < 1.0 {
        return Err(PhysicsError::CalculationError(
            format!("Cutoff ratio must be at least 1, got {}", cutoff_ratio)
        ));
    }
    if gamma <= 1.0 {
        return Err(PhysicsError::CalculationError(
            format!("Gamma must be greater than 1, got {}", gamma)
        ));
    }

    let term1 = compression_ratio.powf(1.0 - gamma);
    let term2 = (cutoff_ratio.powf(gamma) - 1.0) / (gamma * (cutoff_ratio - 1.0));

    Ok(1.0 - term1 * term2)
}

/// Diesel cycle struct for more complex calculations
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DieselCycle {
    /// Compression ratio V_max / V_min
    pub compression_ratio: f64,
    /// Cutoff ratio V_3 / V_2
    pub cutoff_ratio: f64,
    /// Heat capacity ratio Cp/Cv
    pub gamma: f64,
}

impl DieselCycle {
    /// Creates a new Diesel cycle
    pub fn new(compression_ratio: f64, cutoff_ratio: f64, gamma: f64) -> Result<Self, PhysicsError> {
        if compression_ratio <= 1.0 {
            return Err(PhysicsError::CalculationError(
                "Compression ratio must be greater than 1".to_string()
            ));
        }
        if cutoff_ratio < 1.0 {
            return Err(PhysicsError::CalculationError(
                "Cutoff ratio must be at least 1".to_string()
            ));
        }
        if gamma <= 1.0 {
            return Err(PhysicsError::CalculationError(
                "Gamma must be greater than 1".to_string()
            ));
        }
        Ok(Self { compression_ratio, cutoff_ratio, gamma })
    }

    /// Returns the Diesel cycle efficiency
    pub fn efficiency(&self) -> f64 {
        let term1 = self.compression_ratio.powf(1.0 - self.gamma);
        let term2 = (self.cutoff_ratio.powf(self.gamma) - 1.0)
            / (self.gamma * (self.cutoff_ratio - 1.0));
        1.0 - term1 * term2
    }

    /// Calculates work output for a given heat input
    pub fn work_per_cycle(&self, q_in: f64) -> f64 {
        q_in * self.efficiency()
    }
}

// ============================================================================
// Brayton Cycle - Gas Turbines
// ============================================================================

/// Calculates the efficiency of a Brayton cycle (gas turbine)
///
/// η = 1 - 1 / r_p^((γ-1)/γ)
///
/// where r_p is the pressure ratio and γ is the heat capacity ratio.
///
/// # Arguments
/// * `pressure_ratio` - P_max / P_min
/// * `gamma` - Heat capacity ratio Cp/Cv
///
/// # Returns
/// Efficiency as a dimensionless fraction (0 to 1)
pub fn brayton_efficiency(pressure_ratio: f64, gamma: f64) -> Result<f64, PhysicsError> {
    if pressure_ratio <= 1.0 {
        return Err(PhysicsError::CalculationError(
            format!("Pressure ratio must be greater than 1, got {}", pressure_ratio)
        ));
    }
    if gamma <= 1.0 {
        return Err(PhysicsError::CalculationError(
            format!("Gamma must be greater than 1, got {}", gamma)
        ));
    }

    let exponent = (gamma - 1.0) / gamma;
    Ok(1.0 - pressure_ratio.powf(-exponent))
}

/// Brayton cycle struct for more complex calculations
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BraytonCycle {
    /// Pressure ratio P_max / P_min
    pub pressure_ratio: f64,
    /// Heat capacity ratio Cp/Cv
    pub gamma: f64,
}

impl BraytonCycle {
    /// Creates a new Brayton cycle
    pub fn new(pressure_ratio: f64, gamma: f64) -> Result<Self, PhysicsError> {
        if pressure_ratio <= 1.0 {
            return Err(PhysicsError::CalculationError(
                "Pressure ratio must be greater than 1".to_string()
            ));
        }
        if gamma <= 1.0 {
            return Err(PhysicsError::CalculationError(
                "Gamma must be greater than 1".to_string()
            ));
        }
        Ok(Self { pressure_ratio, gamma })
    }

    /// Returns the Brayton cycle efficiency
    pub fn efficiency(&self) -> f64 {
        let exponent = (self.gamma - 1.0) / self.gamma;
        1.0 - self.pressure_ratio.powf(-exponent)
    }
}

// ============================================================================
// Refrigeration and Heat Pump Cycles
// ============================================================================

/// Calculates the COP of an actual refrigerator
///
/// COP = Q_cold / W_input
///
/// # Arguments
/// * `q_cold` - Heat removed from cold reservoir in J
/// * `w_input` - Work input in J
///
/// # Returns
/// Coefficient of Performance (dimensionless)
pub fn refrigerator_cop(q_cold: f64, w_input: f64) -> Result<f64, PhysicsError> {
    if w_input <= 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("Work input must be positive, got {}", w_input)
        ));
    }
    if q_cold < 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("Heat removed must be non-negative, got {}", q_cold)
        ));
    }

    Ok(q_cold / w_input)
}

/// Calculates the COP of an actual heat pump
///
/// COP = Q_hot / W_input
///
/// # Arguments
/// * `q_hot` - Heat delivered to hot reservoir in J
/// * `w_input` - Work input in J
///
/// # Returns
/// Coefficient of Performance (dimensionless, always > 1 for ideal heat pump)
pub fn heat_pump_cop(q_hot: f64, w_input: f64) -> Result<f64, PhysicsError> {
    if w_input <= 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("Work input must be positive, got {}", w_input)
        ));
    }
    if q_hot < 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("Heat delivered must be non-negative, got {}", q_hot)
        ));
    }

    Ok(q_hot / w_input)
}

/// Calculates the second law efficiency of a heat engine
///
/// η_II = η_actual / η_Carnot
///
/// This measures how close an actual engine is to the theoretical maximum.
///
/// # Arguments
/// * `actual_efficiency` - Actual efficiency of the engine (0 to 1)
/// * `t_hot` - Hot reservoir temperature in K
/// * `t_cold` - Cold reservoir temperature in K
///
/// # Returns
/// Second law efficiency (0 to 1)
pub fn second_law_efficiency(actual_efficiency: f64, t_hot: f64, t_cold: f64) -> Result<f64, PhysicsError> {
    validate_efficiency(actual_efficiency)?;
    let carnot_eta = carnot_efficiency(t_hot, t_cold)?;

    Ok(actual_efficiency / carnot_eta)
}

/// Calculates the work required for a refrigeration cycle
///
/// W = Q_cold / COP
///
/// # Arguments
/// * `q_cold` - Heat to be removed in J
/// * `cop` - Coefficient of Performance
///
/// # Returns
/// Work required in J
pub fn refrigeration_work_required(q_cold: f64, cop: f64) -> Result<f64, PhysicsError> {
    if cop <= 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("COP must be positive, got {}", cop)
        ));
    }
    if q_cold < 0.0 {
        return Err(PhysicsError::CalculationError(
            format!("Heat must be non-negative, got {}", q_cold)
        ));
    }

    Ok(q_cold / cop)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ======================= Carnot Cycle Tests =======================

    #[test]
    fn test_carnot_efficiency() {
        // η = 1 - T_cold / T_hot
        let eta = carnot_efficiency(600.0, 300.0).unwrap();
        assert!((eta - 0.5).abs() < 1e-10);

        // Higher temperature difference = higher efficiency
        let eta2 = carnot_efficiency(800.0, 300.0).unwrap();
        assert!(eta2 > eta);
    }

    #[test]
    fn test_carnot_efficiency_invalid() {
        // Cold >= Hot should fail
        assert!(carnot_efficiency(300.0, 300.0).is_err());
        assert!(carnot_efficiency(300.0, 400.0).is_err());

        // Zero or negative temperatures
        assert!(carnot_efficiency(0.0, 300.0).is_err());
        assert!(carnot_efficiency(300.0, 0.0).is_err());
    }

    #[test]
    fn test_carnot_cop_refrigerator() {
        // COP_ref = T_cold / (T_hot - T_cold)
        let cop = carnot_cop_refrigerator(300.0, 250.0).unwrap();
        assert!((cop - 5.0).abs() < 1e-10); // 250 / 50 = 5
    }

    #[test]
    fn test_carnot_cop_heat_pump() {
        // COP_hp = T_hot / (T_hot - T_cold)
        let cop = carnot_cop_heat_pump(300.0, 250.0).unwrap();
        assert!((cop - 6.0).abs() < 1e-10); // 300 / 50 = 6

        // COP_hp = COP_ref + 1
        let cop_ref = carnot_cop_refrigerator(300.0, 250.0).unwrap();
        assert!((cop - cop_ref - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_carnot_work() {
        let w = carnot_work(1000.0, 600.0, 300.0).unwrap();
        assert!((w - 500.0).abs() < 1e-10);
    }

    #[test]
    fn test_carnot_heat_rejected() {
        let q_cold = carnot_heat_rejected(1000.0, 600.0, 300.0).unwrap();
        assert!((q_cold - 500.0).abs() < 1e-10);

        // Q_hot = W + Q_cold
        let w = carnot_work(1000.0, 600.0, 300.0).unwrap();
        assert!((1000.0 - w - q_cold).abs() < 1e-10);
    }

    #[test]
    fn test_carnot_cycle_struct() {
        let cycle = CarnotCycle::new(600.0, 300.0).unwrap();
        assert!((cycle.efficiency() - 0.5).abs() < 1e-10);
        assert!((cycle.cop_refrigerator() - 1.0).abs() < 1e-10);
        assert!((cycle.cop_heat_pump() - 2.0).abs() < 1e-10);
        assert!((cycle.work_per_cycle(1000.0) - 500.0).abs() < 1e-10);
    }

    // ======================= Otto Cycle Tests =======================

    #[test]
    fn test_otto_efficiency() {
        // η = 1 - r^(1-γ)
        let gamma = 1.4;
        let r = 10.0;
        let eta = otto_efficiency(r, gamma).unwrap();

        // Manual calculation: 1 - 10^(-0.4) ≈ 0.602
        let expected = 1.0 - r.powf(1.0 - gamma);
        assert!((eta - expected).abs() < 1e-10);
    }

    #[test]
    fn test_otto_efficiency_increases_with_compression() {
        let gamma = 1.4;
        let eta_8 = otto_efficiency(8.0, gamma).unwrap();
        let eta_10 = otto_efficiency(10.0, gamma).unwrap();
        let eta_12 = otto_efficiency(12.0, gamma).unwrap();

        assert!(eta_8 < eta_10);
        assert!(eta_10 < eta_12);
    }

    #[test]
    fn test_otto_efficiency_invalid() {
        assert!(otto_efficiency(0.5, 1.4).is_err()); // compression ratio < 1
        assert!(otto_efficiency(10.0, 0.5).is_err()); // gamma < 1
    }

    #[test]
    fn test_otto_compression_ratio_for_efficiency() {
        let gamma = 1.4;
        let target_eta = 0.6;
        let r = otto_compression_ratio_for_efficiency(target_eta, gamma).unwrap();

        // Verify by calculating efficiency at this compression ratio
        let eta = otto_efficiency(r, gamma).unwrap();
        assert!((eta - target_eta).abs() < 1e-6);
    }

    #[test]
    fn test_otto_cycle_struct() {
        let cycle = OttoCycle::new(10.0, 1.4).unwrap();
        let expected_eta = 1.0 - 10.0_f64.powf(-0.4);
        assert!((cycle.efficiency() - expected_eta).abs() < 1e-10);
    }

    // ======================= Diesel Cycle Tests =======================

    #[test]
    fn test_diesel_efficiency() {
        let eta = diesel_efficiency(20.0, 2.0, 1.4).unwrap();
        // Diesel efficiency should be less than equivalent Otto for same compression ratio
        // but with cutoff ratio = 1, should equal Otto efficiency
        assert!(eta > 0.5 && eta < 0.8);
    }

    #[test]
    fn test_diesel_efficiency_cutoff_ratio_1() {
        // When cutoff ratio = 1, Diesel cycle becomes Otto cycle
        // Note: formula has 0/0 when r_c = 1, so use limit: (r_c^γ - 1)/(γ*(r_c-1)) -> 1
        // For very small cutoff > 1, should approach Otto efficiency
        let r = 15.0;
        let gamma = 1.4;

        let otto_eta = otto_efficiency(r, gamma).unwrap();
        let diesel_eta = diesel_efficiency(r, 1.001, gamma).unwrap();

        assert!((diesel_eta - otto_eta).abs() < 0.01);
    }

    #[test]
    fn test_diesel_higher_compression_higher_efficiency() {
        let gamma = 1.4;
        let cutoff = 2.0;

        let eta_15 = diesel_efficiency(15.0, cutoff, gamma).unwrap();
        let eta_20 = diesel_efficiency(20.0, cutoff, gamma).unwrap();
        let eta_25 = diesel_efficiency(25.0, cutoff, gamma).unwrap();

        assert!(eta_15 < eta_20);
        assert!(eta_20 < eta_25);
    }

    #[test]
    fn test_diesel_efficiency_invalid() {
        assert!(diesel_efficiency(0.5, 2.0, 1.4).is_err()); // compression < 1
        assert!(diesel_efficiency(20.0, 0.5, 1.4).is_err()); // cutoff < 1
        assert!(diesel_efficiency(20.0, 2.0, 0.5).is_err()); // gamma < 1
    }

    #[test]
    fn test_diesel_cycle_struct() {
        let cycle = DieselCycle::new(20.0, 2.0, 1.4).unwrap();
        let eta = diesel_efficiency(20.0, 2.0, 1.4).unwrap();
        assert!((cycle.efficiency() - eta).abs() < 1e-10);
    }

    // ======================= Brayton Cycle Tests =======================

    #[test]
    fn test_brayton_efficiency() {
        // η = 1 - r_p^(-(γ-1)/γ)
        let gamma = 1.4;
        let r_p = 10.0;
        let eta = brayton_efficiency(r_p, gamma).unwrap();

        let exponent = (gamma - 1.0) / gamma;
        let expected = 1.0 - r_p.powf(-exponent);
        assert!((eta - expected).abs() < 1e-10);
    }

    #[test]
    fn test_brayton_efficiency_invalid() {
        assert!(brayton_efficiency(0.5, 1.4).is_err());
        assert!(brayton_efficiency(10.0, 0.5).is_err());
    }

    #[test]
    fn test_brayton_cycle_struct() {
        let cycle = BraytonCycle::new(10.0, 1.4).unwrap();
        let eta = brayton_efficiency(10.0, 1.4).unwrap();
        assert!((cycle.efficiency() - eta).abs() < 1e-10);
    }

    // ======================= Refrigeration Tests =======================

    #[test]
    fn test_refrigerator_cop() {
        let cop = refrigerator_cop(400.0, 100.0).unwrap();
        assert!((cop - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_heat_pump_cop() {
        let cop = heat_pump_cop(500.0, 100.0).unwrap();
        assert!((cop - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_heat_pump_cop_greater_than_1() {
        // Energy conservation: Q_hot = Q_cold + W
        // So COP_hp = Q_hot/W = (Q_cold + W)/W = Q_cold/W + 1 = COP_ref + 1 > 1
        let cop = heat_pump_cop(150.0, 100.0).unwrap();
        assert!(cop > 1.0);
    }

    #[test]
    fn test_refrigeration_cop_invalid() {
        assert!(refrigerator_cop(100.0, 0.0).is_err());
        assert!(refrigerator_cop(100.0, -50.0).is_err());
        assert!(refrigerator_cop(-100.0, 50.0).is_err());
    }

    #[test]
    fn test_second_law_efficiency() {
        // If actual = 30% and Carnot = 50%, second law = 60%
        let eta_II = second_law_efficiency(0.30, 600.0, 300.0).unwrap();
        assert!((eta_II - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_refrigeration_work_required() {
        let w = refrigeration_work_required(400.0, 4.0).unwrap();
        assert!((w - 100.0).abs() < 1e-10);
    }

    // ======================= Comparison Tests =======================

    #[test]
    fn test_diesel_vs_otto_same_compression() {
        // For the same compression ratio, Diesel is less efficient than Otto
        // (due to the cutoff ratio reducing efficiency)
        let gamma = 1.4;
        let r = 15.0;

        let otto_eta = otto_efficiency(r, gamma).unwrap();
        let diesel_eta = diesel_efficiency(r, 2.0, gamma).unwrap();

        assert!(diesel_eta < otto_eta);
    }

    #[test]
    fn test_brayton_equals_otto_at_same_ratio() {
        // Brayton and Otto have same efficiency formula structure
        // η_Brayton = 1 - r_p^(-(γ-1)/γ)
        // η_Otto = 1 - r^(1-γ)
        // These are equivalent when r_p^(1/γ) = r
        let gamma = 1.4;
        let r_otto: f64 = 10.0;
        let r_p_brayton = r_otto.powf(gamma);

        let otto_eta = otto_efficiency(r_otto, gamma).unwrap();
        let brayton_eta = brayton_efficiency(r_p_brayton, gamma).unwrap();

        assert!((otto_eta - brayton_eta).abs() < 1e-10);
    }
}
