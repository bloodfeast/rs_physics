//! Predefined substances with thermal properties
//!
//! This module provides a `Substance` struct representing materials with
//! thermodynamic properties, along with predefined common substances.

use crate::utils::PhysicsError;
use super::validation::{validate_heat_capacity, validate_thermal_conductivity, validate_mass};

/// Represents the phase (state of matter) of a substance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Solid phase
    Solid,
    /// Liquid phase
    Liquid,
    /// Gas/vapor phase
    Gas,
    /// Plasma phase (ionized gas)
    Plasma,
}

impl std::fmt::Display for Phase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Phase::Solid => write!(f, "Solid"),
            Phase::Liquid => write!(f, "Liquid"),
            Phase::Gas => write!(f, "Gas"),
            Phase::Plasma => write!(f, "Plasma"),
        }
    }
}

/// Represents a substance with thermodynamic properties
///
/// This struct holds the thermal properties needed for heat transfer
/// and thermodynamic calculations.
#[derive(Debug, Clone)]
pub struct Substance {
    /// Name of the substance
    pub name: String,
    /// Specific heat capacity in J/(kg·K)
    pub specific_heat_capacity: f64,
    /// Thermal conductivity in W/(m·K)
    pub thermal_conductivity: f64,
    /// Density in kg/m³
    pub density: f64,
    /// Molar mass in kg/mol
    pub molar_mass: f64,
    /// Phase at standard conditions
    pub phase: Phase,
}

impl Substance {
    /// Creates a new substance with custom properties
    ///
    /// # Arguments
    /// * `name` - Name of the substance
    /// * `specific_heat_capacity` - Specific heat capacity in J/(kg·K)
    /// * `thermal_conductivity` - Thermal conductivity in W/(m·K)
    /// * `density` - Density in kg/m³
    /// * `molar_mass` - Molar mass in kg/mol
    /// * `phase` - Phase at standard conditions
    ///
    /// # Returns
    /// * `Ok(Substance)` if all properties are valid
    /// * `Err(PhysicsError)` if any property is invalid
    ///
    /// # Examples
    /// ```
    /// use rs_physics::thermodynamics::{Substance, Phase};
    ///
    /// let custom = Substance::new(
    ///     "Custom Material",
    ///     1000.0,  // specific heat
    ///     50.0,    // thermal conductivity
    ///     2500.0,  // density
    ///     0.1,     // molar mass
    ///     Phase::Solid
    /// ).unwrap();
    /// ```
    pub fn new(
        name: &str,
        specific_heat_capacity: f64,
        thermal_conductivity: f64,
        density: f64,
        molar_mass: f64,
        phase: Phase,
    ) -> Result<Self, PhysicsError> {
        validate_heat_capacity(specific_heat_capacity)?;
        validate_thermal_conductivity(thermal_conductivity)?;
        validate_mass(density)?;

        if molar_mass <= 0.0 {
            return Err(PhysicsError::CalculationError(
                format!("Molar mass must be positive, got {} kg/mol", molar_mass)
            ));
        }

        Ok(Self {
            name: name.to_string(),
            specific_heat_capacity,
            thermal_conductivity,
            density,
            molar_mass,
            phase,
        })
    }

    /// Creates a substance representing liquid water at 25°C
    ///
    /// Properties:
    /// - Specific heat capacity: 4186 J/(kg·K)
    /// - Thermal conductivity: 0.606 W/(m·K)
    /// - Density: 997 kg/m³
    /// - Molar mass: 0.018015 kg/mol
    ///
    /// # Examples
    /// ```
    /// use rs_physics::thermodynamics::{Substance, Phase};
    ///
    /// let water = Substance::water();
    /// assert_eq!(water.phase, Phase::Liquid);
    /// assert!((water.specific_heat_capacity - 4186.0).abs() < 1.0);
    /// ```
    pub fn water() -> Self {
        Self {
            name: "Water".to_string(),
            specific_heat_capacity: 4186.0,
            thermal_conductivity: 0.606,
            density: 997.0,
            molar_mass: 0.018015,
            phase: Phase::Liquid,
        }
    }

    /// Creates a substance representing ice at 0°C
    ///
    /// Properties:
    /// - Specific heat capacity: 2090 J/(kg·K)
    /// - Thermal conductivity: 2.22 W/(m·K)
    /// - Density: 917 kg/m³
    /// - Molar mass: 0.018015 kg/mol
    pub fn ice() -> Self {
        Self {
            name: "Ice".to_string(),
            specific_heat_capacity: 2090.0,
            thermal_conductivity: 2.22,
            density: 917.0,
            molar_mass: 0.018015,
            phase: Phase::Solid,
        }
    }

    /// Creates a substance representing steam at 100°C, 1 atm
    ///
    /// Properties:
    /// - Specific heat capacity: 2010 J/(kg·K)
    /// - Thermal conductivity: 0.0248 W/(m·K)
    /// - Density: 0.598 kg/m³
    /// - Molar mass: 0.018015 kg/mol
    pub fn steam() -> Self {
        Self {
            name: "Steam".to_string(),
            specific_heat_capacity: 2010.0,
            thermal_conductivity: 0.0248,
            density: 0.598,
            molar_mass: 0.018015,
            phase: Phase::Gas,
        }
    }

    /// Creates a substance representing air at 25°C, 1 atm
    ///
    /// Properties:
    /// - Specific heat capacity: 1005 J/(kg·K)
    /// - Thermal conductivity: 0.0262 W/(m·K)
    /// - Density: 1.184 kg/m³
    /// - Molar mass: 0.02897 kg/mol (average)
    pub fn air() -> Self {
        Self {
            name: "Air".to_string(),
            specific_heat_capacity: 1005.0,
            thermal_conductivity: 0.0262,
            density: 1.184,
            molar_mass: 0.02897,
            phase: Phase::Gas,
        }
    }

    /// Creates an ideal gas with specified molar mass
    ///
    /// Uses ideal gas properties with C_p = 5R/2M for monatomic behavior.
    ///
    /// # Arguments
    /// * `molar_mass` - Molar mass in kg/mol
    ///
    /// # Returns
    /// * `Ok(Substance)` if molar mass is valid
    /// * `Err(PhysicsError)` if molar mass is invalid
    pub fn ideal_gas(molar_mass: f64) -> Result<Self, PhysicsError> {
        if molar_mass <= 0.0 {
            return Err(PhysicsError::CalculationError(
                format!("Molar mass must be positive, got {} kg/mol", molar_mass)
            ));
        }

        // For ideal gas: C_p = (5/2) * R / M for monatomic
        let r = super::constants::R;
        let specific_heat = 2.5 * r / molar_mass;

        // Thermal conductivity approximation for ideal gas
        let thermal_conductivity = 0.025; // Approximate value

        // Density at STP using ideal gas law: ρ = PM/(RT)
        let density = 101325.0 * molar_mass / (r * 298.15);

        Ok(Self {
            name: "Ideal Gas".to_string(),
            specific_heat_capacity: specific_heat,
            thermal_conductivity,
            density,
            molar_mass,
            phase: Phase::Gas,
        })
    }

    /// Creates a substance representing copper at 25°C
    ///
    /// Properties:
    /// - Specific heat capacity: 385 J/(kg·K)
    /// - Thermal conductivity: 401 W/(m·K)
    /// - Density: 8960 kg/m³
    /// - Molar mass: 0.06355 kg/mol
    pub fn copper() -> Self {
        Self {
            name: "Copper".to_string(),
            specific_heat_capacity: 385.0,
            thermal_conductivity: 401.0,
            density: 8960.0,
            molar_mass: 0.06355,
            phase: Phase::Solid,
        }
    }

    /// Creates a substance representing iron at 25°C
    ///
    /// Properties:
    /// - Specific heat capacity: 449 J/(kg·K)
    /// - Thermal conductivity: 80.4 W/(m·K)
    /// - Density: 7874 kg/m³
    /// - Molar mass: 0.05585 kg/mol
    pub fn iron() -> Self {
        Self {
            name: "Iron".to_string(),
            specific_heat_capacity: 449.0,
            thermal_conductivity: 80.4,
            density: 7874.0,
            molar_mass: 0.05585,
            phase: Phase::Solid,
        }
    }

    /// Creates a substance representing aluminum at 25°C
    ///
    /// Properties:
    /// - Specific heat capacity: 897 J/(kg·K)
    /// - Thermal conductivity: 237 W/(m·K)
    /// - Density: 2700 kg/m³
    /// - Molar mass: 0.02698 kg/mol
    pub fn aluminum() -> Self {
        Self {
            name: "Aluminum".to_string(),
            specific_heat_capacity: 897.0,
            thermal_conductivity: 237.0,
            density: 2700.0,
            molar_mass: 0.02698,
            phase: Phase::Solid,
        }
    }

    /// Creates a substance representing glass (soda-lime) at 25°C
    ///
    /// Properties:
    /// - Specific heat capacity: 840 J/(kg·K)
    /// - Thermal conductivity: 1.0 W/(m·K)
    /// - Density: 2500 kg/m³
    /// - Molar mass: 0.060 kg/mol (approximate)
    pub fn glass() -> Self {
        Self {
            name: "Glass".to_string(),
            specific_heat_capacity: 840.0,
            thermal_conductivity: 1.0,
            density: 2500.0,
            molar_mass: 0.060,
            phase: Phase::Solid,
        }
    }

    /// Creates a substance representing stainless steel at 25°C
    ///
    /// Properties:
    /// - Specific heat capacity: 500 J/(kg·K)
    /// - Thermal conductivity: 16.3 W/(m·K)
    /// - Density: 8000 kg/m³
    /// - Molar mass: 0.055 kg/mol (approximate, iron-based)
    pub fn stainless_steel() -> Self {
        Self {
            name: "Stainless Steel".to_string(),
            specific_heat_capacity: 500.0,
            thermal_conductivity: 16.3,
            density: 8000.0,
            molar_mass: 0.055,
            phase: Phase::Solid,
        }
    }

    /// Creates a substance representing concrete at 25°C
    ///
    /// Properties:
    /// - Specific heat capacity: 880 J/(kg·K)
    /// - Thermal conductivity: 1.7 W/(m·K)
    /// - Density: 2400 kg/m³
    /// - Molar mass: 0.100 kg/mol (approximate)
    pub fn concrete() -> Self {
        Self {
            name: "Concrete".to_string(),
            specific_heat_capacity: 880.0,
            thermal_conductivity: 1.7,
            density: 2400.0,
            molar_mass: 0.100,
            phase: Phase::Solid,
        }
    }

    /// Creates a substance representing wood (oak) at 25°C
    ///
    /// Properties:
    /// - Specific heat capacity: 2380 J/(kg·K)
    /// - Thermal conductivity: 0.17 W/(m·K)
    /// - Density: 750 kg/m³
    /// - Molar mass: 0.162 kg/mol (cellulose approximate)
    pub fn wood() -> Self {
        Self {
            name: "Wood (Oak)".to_string(),
            specific_heat_capacity: 2380.0,
            thermal_conductivity: 0.17,
            density: 750.0,
            molar_mass: 0.162,
            phase: Phase::Solid,
        }
    }

    /// Creates a substance representing gold at 25°C
    ///
    /// Properties:
    /// - Specific heat capacity: 129 J/(kg·K)
    /// - Thermal conductivity: 317 W/(m·K)
    /// - Density: 19300 kg/m³
    /// - Molar mass: 0.19697 kg/mol
    pub fn gold() -> Self {
        Self {
            name: "Gold".to_string(),
            specific_heat_capacity: 129.0,
            thermal_conductivity: 317.0,
            density: 19300.0,
            molar_mass: 0.19697,
            phase: Phase::Solid,
        }
    }

    /// Creates a substance representing silver at 25°C
    ///
    /// Properties:
    /// - Specific heat capacity: 235 J/(kg·K)
    /// - Thermal conductivity: 429 W/(m·K)
    /// - Density: 10500 kg/m³
    /// - Molar mass: 0.10787 kg/mol
    pub fn silver() -> Self {
        Self {
            name: "Silver".to_string(),
            specific_heat_capacity: 235.0,
            thermal_conductivity: 429.0,
            density: 10500.0,
            molar_mass: 0.10787,
            phase: Phase::Solid,
        }
    }

    /// Calculates the thermal diffusivity of the substance
    ///
    /// Thermal diffusivity (α) measures how quickly heat diffuses through a material.
    /// α = k / (ρ × c)
    ///
    /// # Returns
    /// Thermal diffusivity in m²/s
    ///
    /// # Examples
    /// ```
    /// use rs_physics::thermodynamics::Substance;
    ///
    /// let copper = Substance::copper();
    /// let alpha = copper.thermal_diffusivity();
    /// // Copper has high thermal diffusivity (~1.1e-4 m²/s)
    /// assert!(alpha > 1e-5);
    /// ```
    #[inline]
    pub fn thermal_diffusivity(&self) -> f64 {
        self.thermal_conductivity / (self.density * self.specific_heat_capacity)
    }

    /// Calculates the volumetric heat capacity
    ///
    /// Volumetric heat capacity = ρ × c
    ///
    /// # Returns
    /// Volumetric heat capacity in J/(m³·K)
    #[inline]
    pub fn volumetric_heat_capacity(&self) -> f64 {
        self.density * self.specific_heat_capacity
    }

    /// Calculates the heat required to change the temperature of a given mass
    ///
    /// Q = m × c × ΔT
    ///
    /// # Arguments
    /// * `mass` - Mass of the substance in kg
    /// * `delta_t` - Temperature change in K (or °C)
    ///
    /// # Returns
    /// * `Ok(f64)` - Heat energy in Joules
    /// * `Err(PhysicsError)` if mass is invalid
    pub fn heat_for_temperature_change(&self, mass: f64, delta_t: f64) -> Result<f64, PhysicsError> {
        validate_mass(mass)?;
        Ok(mass * self.specific_heat_capacity * delta_t)
    }

    /// Calculates the temperature change for a given heat input
    ///
    /// ΔT = Q / (m × c)
    ///
    /// # Arguments
    /// * `mass` - Mass of the substance in kg
    /// * `heat` - Heat energy in Joules
    ///
    /// # Returns
    /// * `Ok(f64)` - Temperature change in K
    /// * `Err(PhysicsError)` if mass is invalid
    pub fn temperature_change_for_heat(&self, mass: f64, heat: f64) -> Result<f64, PhysicsError> {
        validate_mass(mass)?;
        Ok(heat / (mass * self.specific_heat_capacity))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substance_water_liquid() {
        let water = Substance::water();
        assert_eq!(water.name, "Water");
        assert_eq!(water.phase, Phase::Liquid);
        assert!((water.specific_heat_capacity - 4186.0).abs() < 1.0);
        assert!((water.thermal_conductivity - 0.606).abs() < 0.01);
        assert!((water.density - 997.0).abs() < 1.0);
        assert!((water.molar_mass - 0.018015).abs() < 0.001);
    }

    #[test]
    fn test_substance_water_ice() {
        let ice = Substance::ice();
        assert_eq!(ice.name, "Ice");
        assert_eq!(ice.phase, Phase::Solid);
        assert!((ice.specific_heat_capacity - 2090.0).abs() < 1.0);
        assert!((ice.thermal_conductivity - 2.22).abs() < 0.01);
        // Ice is less dense than water
        assert!(ice.density < Substance::water().density);
    }

    #[test]
    fn test_substance_water_steam() {
        let steam = Substance::steam();
        assert_eq!(steam.name, "Steam");
        assert_eq!(steam.phase, Phase::Gas);
        assert!((steam.specific_heat_capacity - 2010.0).abs() < 1.0);
        // Steam is much less dense than liquid water
        assert!(steam.density < 1.0);
    }

    #[test]
    fn test_substance_air() {
        let air = Substance::air();
        assert_eq!(air.name, "Air");
        assert_eq!(air.phase, Phase::Gas);
        assert!((air.specific_heat_capacity - 1005.0).abs() < 1.0);
        assert!((air.density - 1.184).abs() < 0.01);
    }

    #[test]
    fn test_substance_ideal_gas() {
        // Test with helium-like molar mass
        let helium_mass = 0.004; // kg/mol
        let ideal = Substance::ideal_gas(helium_mass).unwrap();
        assert_eq!(ideal.phase, Phase::Gas);
        assert!(ideal.specific_heat_capacity > 0.0);
        assert!(ideal.density > 0.0);
    }

    #[test]
    fn test_substance_ideal_gas_invalid() {
        assert!(Substance::ideal_gas(0.0).is_err());
        assert!(Substance::ideal_gas(-1.0).is_err());
    }

    #[test]
    fn test_substance_copper() {
        let copper = Substance::copper();
        assert_eq!(copper.name, "Copper");
        assert_eq!(copper.phase, Phase::Solid);
        assert!((copper.thermal_conductivity - 401.0).abs() < 1.0);
        // Copper has high thermal conductivity
        assert!(copper.thermal_conductivity > 300.0);
    }

    #[test]
    fn test_substance_iron() {
        let iron = Substance::iron();
        assert_eq!(iron.name, "Iron");
        assert_eq!(iron.phase, Phase::Solid);
        assert!((iron.density - 7874.0).abs() < 1.0);
    }

    #[test]
    fn test_substance_custom_creation() {
        let custom = Substance::new(
            "Custom Material",
            1000.0,
            50.0,
            2500.0,
            0.1,
            Phase::Solid,
        ).unwrap();

        assert_eq!(custom.name, "Custom Material");
        assert_eq!(custom.phase, Phase::Solid);
        assert!((custom.specific_heat_capacity - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_substance_invalid_properties() {
        // Invalid specific heat capacity
        assert!(Substance::new("Test", 0.0, 50.0, 2500.0, 0.1, Phase::Solid).is_err());
        assert!(Substance::new("Test", -100.0, 50.0, 2500.0, 0.1, Phase::Solid).is_err());

        // Invalid thermal conductivity
        assert!(Substance::new("Test", 1000.0, 0.0, 2500.0, 0.1, Phase::Solid).is_err());
        assert!(Substance::new("Test", 1000.0, -50.0, 2500.0, 0.1, Phase::Solid).is_err());

        // Invalid density
        assert!(Substance::new("Test", 1000.0, 50.0, 0.0, 0.1, Phase::Solid).is_err());
        assert!(Substance::new("Test", 1000.0, 50.0, -2500.0, 0.1, Phase::Solid).is_err());

        // Invalid molar mass
        assert!(Substance::new("Test", 1000.0, 50.0, 2500.0, 0.0, Phase::Solid).is_err());
        assert!(Substance::new("Test", 1000.0, 50.0, 2500.0, -0.1, Phase::Solid).is_err());
    }

    #[test]
    fn test_thermal_diffusivity() {
        let copper = Substance::copper();
        let alpha = copper.thermal_diffusivity();

        // α = k / (ρ × c)
        let expected = copper.thermal_conductivity / (copper.density * copper.specific_heat_capacity);
        assert!((alpha - expected).abs() < 1e-10);

        // Copper has high thermal diffusivity (~1.1e-4 m²/s)
        assert!(alpha > 1e-5);
        assert!(alpha < 1e-3);
    }

    #[test]
    fn test_volumetric_heat_capacity() {
        let water = Substance::water();
        let vol_cap = water.volumetric_heat_capacity();

        // Should be ρ × c
        let expected = water.density * water.specific_heat_capacity;
        assert!((vol_cap - expected).abs() < 1e-6);

        // Water has high volumetric heat capacity
        assert!(vol_cap > 4e6);
    }

    #[test]
    fn test_heat_for_temperature_change() {
        let water = Substance::water();

        // Heat 1 kg of water by 10°C
        let heat = water.heat_for_temperature_change(1.0, 10.0).unwrap();
        // Q = 1 kg × 4186 J/(kg·K) × 10 K = 41860 J
        assert!((heat - 41860.0).abs() < 10.0);

        // Cooling should give negative heat
        let cooling_heat = water.heat_for_temperature_change(1.0, -10.0).unwrap();
        assert!((cooling_heat - (-41860.0)).abs() < 10.0);
    }

    #[test]
    fn test_heat_for_temperature_change_invalid() {
        let water = Substance::water();
        assert!(water.heat_for_temperature_change(0.0, 10.0).is_err());
        assert!(water.heat_for_temperature_change(-1.0, 10.0).is_err());
    }

    #[test]
    fn test_temperature_change_for_heat() {
        let water = Substance::water();

        // Add 41860 J to 1 kg of water
        let delta_t = water.temperature_change_for_heat(1.0, 41860.0).unwrap();
        // Should be approximately 10°C
        assert!((delta_t - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_phase_display() {
        assert_eq!(format!("{}", Phase::Solid), "Solid");
        assert_eq!(format!("{}", Phase::Liquid), "Liquid");
        assert_eq!(format!("{}", Phase::Gas), "Gas");
        assert_eq!(format!("{}", Phase::Plasma), "Plasma");
    }

    #[test]
    fn test_thermal_conductivity_ordering() {
        // Metals > Glass/Concrete > Wood > Air
        let copper = Substance::copper();
        let glass = Substance::glass();
        let wood = Substance::wood();
        let air = Substance::air();

        assert!(copper.thermal_conductivity > glass.thermal_conductivity);
        assert!(glass.thermal_conductivity > wood.thermal_conductivity);
        assert!(wood.thermal_conductivity > air.thermal_conductivity);
    }

    #[test]
    fn test_metal_properties() {
        // Silver has the highest thermal conductivity
        let silver = Substance::silver();
        let copper = Substance::copper();
        let gold = Substance::gold();
        let aluminum = Substance::aluminum();
        let iron = Substance::iron();

        assert!(silver.thermal_conductivity > copper.thermal_conductivity);
        assert!(copper.thermal_conductivity > gold.thermal_conductivity);
        assert!(gold.thermal_conductivity > aluminum.thermal_conductivity);
        assert!(aluminum.thermal_conductivity > iron.thermal_conductivity);
    }

    #[test]
    fn test_all_predefined_substances_valid() {
        // All predefined substances should have valid (positive) properties
        let substances = vec![
            Substance::water(),
            Substance::ice(),
            Substance::steam(),
            Substance::air(),
            Substance::copper(),
            Substance::iron(),
            Substance::aluminum(),
            Substance::glass(),
            Substance::stainless_steel(),
            Substance::concrete(),
            Substance::wood(),
            Substance::gold(),
            Substance::silver(),
        ];

        for s in substances {
            assert!(s.specific_heat_capacity > 0.0, "{} has invalid specific heat", s.name);
            assert!(s.thermal_conductivity > 0.0, "{} has invalid conductivity", s.name);
            assert!(s.density > 0.0, "{} has invalid density", s.name);
            assert!(s.molar_mass > 0.0, "{} has invalid molar mass", s.name);
            assert!(s.thermal_diffusivity() > 0.0, "{} has invalid diffusivity", s.name);
        }
    }
}
