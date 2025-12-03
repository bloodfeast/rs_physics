// src/materials.rs

use crate::utils::PhysicsError;

/// Represents different types of material failure.
///
/// This enum categorizes the mode of failure or deformation that occurs
/// when a material is subjected to stress or strain beyond certain limits.
///
/// # Examples
///
/// ```
/// use rs_physics::materials::{Material, BreakageType};
///
/// let steel = Material::steel();
/// let result = steel.will_break(500e6, 0.003, None);  // High stress
/// assert_eq!(result.breakage_type, BreakageType::TensileStress);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakageType {
    /// No failure - material remains in elastic region.
    /// Stress and strain are below yield point, material will return
    /// to original shape when load is removed.
    None,
    /// Immediate failure due to exceeding ultimate tensile strength.
    /// The applied stress has exceeded the maximum stress the material
    /// can withstand, causing catastrophic failure.
    TensileStress,
    /// Immediate failure due to exceeding ultimate strain.
    /// The material has stretched beyond its maximum allowable deformation,
    /// even if the stress hasn't exceeded the ultimate strength.
    TensileStrain,
    /// Plastic deformation (yield point exceeded).
    /// The material has permanently deformed but hasn't failed completely.
    /// Material will not return to original shape when load is removed.
    Plastic,
    /// Failure due to cyclic loading (fatigue).
    /// The material fails at stress levels below the yield strength
    /// due to repeated loading and unloading cycles.
    Fatigue,
}

/// Result of material failure analysis.
///
/// This struct contains comprehensive information about whether a material
/// will fail under given loading conditions and what type of failure would occur.
///
/// # Fields
///
/// * `will_break` - `true` if the material will catastrophically fail
/// * `breakage_type` - The mode of failure or deformation
/// * `safety_factor` - Ratio of allowable stress to applied stress (>1 = safe)
///
/// # Examples
///
/// ## Checking for safe operation
///
/// ```
/// use rs_physics::materials::{Material, BreakageType};
///
/// let steel = Material::steel();
///
/// // Check if stress is safe
/// let result = steel.will_break(100e6, 0.0005, None);
/// if result.will_break {
///     println!("Material will fail via {:?}!", result.breakage_type);
/// } else if result.safety_factor > 2.0 {
///     println!("Safe with factor of safety: {:.2}", result.safety_factor);
/// } else {
///     println!("Marginal safety factor: {:.2}", result.safety_factor);
/// }
/// ```
///
/// ## Fatigue analysis
///
/// ```
/// use rs_physics::materials::{Material, BreakageType};
///
/// let steel = Material::steel();
///
/// // Check for fatigue failure after 1 million cycles
/// let result = steel.will_break(150e6, 0.00075, Some(1_000_000));
/// match result.breakage_type {
///     BreakageType::Fatigue => println!("Fatigue failure expected"),
///     BreakageType::None => println!("Safe for cyclic loading"),
///     _ => println!("Other failure mode"),
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BreakageResult {
    /// Whether the material will break under given conditions.
    /// `true` indicates catastrophic failure (TensileStress, TensileStrain, or Fatigue).
    /// `false` means the material survives, though it may have yielded (Plastic).
    pub will_break: bool,
    /// Type of failure or deformation that occurs.
    /// See [`BreakageType`] for detailed descriptions of each failure mode.
    pub breakage_type: BreakageType,
    /// Ratio of allowable stress to applied stress.
    /// Values > 1.0 indicate the material is within safe limits.
    /// Values < 1.0 indicate the material has exceeded its limits.
    /// Typical engineering designs target safety factors of 1.5 to 3.0.
    pub safety_factor: f64,
}

/// Represents the physical properties of a material.
///
/// This struct encapsulates various material properties that affect physical interactions,
/// including mechanical, thermal, and collision behaviors.
///
/// # Properties
/// * `density` - Mass per unit volume in kg/m³
/// * `youngs_modulus` - Measure of material stiffness in Pascals (Pa)
/// * `poisson_ratio` - Ratio of transverse strain to axial strain (dimensionless)
/// * `friction_coefficient` - Coefficient of friction (dimensionless)
/// * `restitution_coefficient` - Coefficient of restitution for collisions (dimensionless)
/// * `rolling_resistance_coefficient` - Coefficient of rolling resistance (dimensionless)
/// * `thermal_conductivity` - Rate of heat transfer in W/(m·K)
/// * `specific_heat_capacity` - Energy required to raise temperature in J/(kg·K)
/// * `yield_strength` - Stress at which material begins to deform plastically in Pascals (Pa)
/// * `ultimate_strength` - Maximum stress before failure in Pascals (Pa)
#[derive(Debug, Clone, Copy)]
pub struct Material {
    /// Density of the material in kg/m³
    pub density: f64,
    /// Young's modulus in Pascals (Pa)
    pub youngs_modulus: f64,
    /// Poisson's ratio (dimensionless)
    pub poisson_ratio: f64,
    /// Coefficient of friction (dimensionless)
    pub friction_coefficient: f64,
    /// Coefficient of restitution (dimensionless)
    pub restitution_coefficient: f64,
    /// Coefficient of rolling resistance (dimensionless)
    /// Represents energy loss due to deformation at the contact patch during rolling.
    /// Typical values: 0.001-0.005 for hard materials (steel on steel),
    /// 0.01-0.03 for medium materials (rubber on concrete),
    /// 0.1-0.3 for soft materials (rubber on sand).
    pub rolling_resistance_coefficient: f64,
    /// Thermal conductivity in W/(m·K)
    pub thermal_conductivity: f64,
    /// Specific heat capacity in J/(kg·K)
    pub specific_heat_capacity: f64,
    /// Yield strength in Pascals (Pa)
    pub yield_strength: f64,
    /// Ultimate strength in Pascals (Pa)
    pub ultimate_strength: f64,
}

impl Material {
    /// Creates a new material with the specified properties.
    ///
    /// # Arguments
    ///
    /// * `density` - Mass per unit volume in kg/m³
    /// * `youngs_modulus` - Measure of material stiffness in Pascals (Pa)
    /// * `poisson_ratio` - Ratio of transverse strain to axial strain (dimensionless)
    /// * `friction_coefficient` - Coefficient of friction (dimensionless)
    /// * `restitution_coefficient` - Coefficient of restitution for collisions (dimensionless)
    /// * `rolling_resistance_coefficient` - Coefficient of rolling resistance (dimensionless)
    /// * `thermal_conductivity` - Rate of heat transfer in W/(m·K)
    /// * `specific_heat_capacity` - Energy required to raise temperature in J/(kg·K)
    /// * `yield_strength` - Stress at which material begins to deform plastically in Pascals (Pa)
    /// * `ultimate_strength` - Maximum stress before failure in Pascals (Pa)
    ///
    /// # Returns
    ///
    /// * `Ok(Material)` - A new Material instance with the specified properties
    /// * `Err(PhysicsError)` - If any of the input parameters are invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let steel = Material::new(
    ///     7850.0,   // density
    ///     200.0e9,  // Young's modulus
    ///     0.3,      // Poisson's ratio
    ///     0.74,     // friction coefficient
    ///     0.85,     // restitution coefficient
    ///     0.002,    // rolling resistance coefficient
    ///     43.0,     // thermal conductivity
    ///     490.0,    // specific heat capacity
    ///     250.0e6,  // yield strength
    ///     400.0e6   // ultimate strength
    /// ).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * Density is not positive
    /// * Young's modulus is not positive
    /// * Poisson's ratio is not between -1 and 0.5
    /// * Friction coefficient is negative
    /// * Restitution coefficient is not between 0 and 1
    /// * Rolling resistance coefficient is negative
    /// * Thermal conductivity is negative
    /// * Specific heat capacity is not positive
    /// * Yield strength is negative
    /// * Ultimate strength is less than yield strength
    pub fn new(
        density: f64,
        youngs_modulus: f64,
        poisson_ratio: f64,
        friction_coefficient: f64,
        restitution_coefficient: f64,
        rolling_resistance_coefficient: f64,
        thermal_conductivity: f64,
        specific_heat_capacity: f64,
        yield_strength: f64,
        ultimate_strength: f64,
    ) -> Result<Self, PhysicsError> {
        // Validate inputs
        if density <= 0.0 { return Err(PhysicsError::CalculationError("Density must be positive".to_string())); }
        if youngs_modulus <= 0.0 { return Err(PhysicsError::CalculationError("Young's modulus must be positive".to_string())); }
        if poisson_ratio <= -1.0 || poisson_ratio >= 0.5 { return Err(PhysicsError::CalculationError("Poisson's ratio must be between -1 and 0.5".to_string())); }
        if friction_coefficient < 0.0 { return Err(PhysicsError::InvalidCoefficient); }
        if restitution_coefficient < 0.0 || restitution_coefficient > 1.0 { return Err(PhysicsError::CalculationError("Coefficient of restitution must be between 0 and 1".to_string())); }
        if rolling_resistance_coefficient < 0.0 { return Err(PhysicsError::CalculationError("Rolling resistance coefficient must be non-negative".to_string())); }
        if thermal_conductivity < 0.0 { return Err(PhysicsError::InvalidCoefficient); }
        if specific_heat_capacity <= 0.0 { return Err(PhysicsError::CalculationError("Specific heat capacity must be positive".to_string())); }
        if yield_strength < 0.0 { return Err(PhysicsError::CalculationError("Yield strength must be non-negative".to_string())); }
        if ultimate_strength < yield_strength { return Err(PhysicsError::CalculationError("Ultimate strength must be greater than or equal to yield strength".to_string())); }

        Ok(Self {
            density,
            youngs_modulus,
            poisson_ratio,
            friction_coefficient,
            restitution_coefficient,
            rolling_resistance_coefficient,
            thermal_conductivity,
            specific_heat_capacity,
            yield_strength,
            ultimate_strength,
        })
    }

    /// Creates a new Material instance with properties of steel.
    ///
    /// # Returns
    ///
    /// A Material instance with typical properties of structural steel:
    /// * Density: 7850 kg/m³
    /// * Young's modulus: 200 GPa
    /// * Poisson's ratio: 0.3
    /// * Friction coefficient: 0.74
    /// * Restitution coefficient: 0.85
    /// * Rolling resistance coefficient: 0.002 (steel on steel)
    /// * Thermal conductivity: 43 W/(m·K)
    /// * Specific heat capacity: 490 J/(kg·K)
    /// * Yield strength: 250 MPa
    /// * Ultimate strength: 400 MPa
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let steel = Material::steel();
    /// assert_eq!(steel.density, 7850.0);
    /// ```
    pub fn steel() -> Self {
        Self::new(
            7850.0,             // density (kg/m³)
            200.0e9,            // Young's modulus (Pa)
            0.3,                // Poisson's ratio
            0.74,               // friction coefficient
            0.85,               // restitution coefficient
            0.002,              // rolling resistance coefficient (steel on steel)
            43.0,               // thermal conductivity (W/(m·K))
            490.0,              // specific heat capacity (J/(kg·K))
            250.0e6,            // yield strength (Pa)
            400.0e6,            // ultimate strength (Pa)
        ).expect("Failed to create steel material")
    }

    /// Creates a new Material instance with properties of aluminum.
    ///
    /// # Returns
    ///
    /// A Material instance with typical properties of aluminum:
    /// * Density: 2700 kg/m³
    /// * Young's modulus: 69 GPa
    /// * Poisson's ratio: 0.33
    /// * Friction coefficient: 0.61
    /// * Restitution coefficient: 0.75
    /// * Rolling resistance coefficient: 0.001 (aluminum on aluminum)
    /// * Thermal conductivity: 237 W/(m·K)
    /// * Specific heat capacity: 900 J/(kg·K)
    /// * Yield strength: 95 MPa
    /// * Ultimate strength: 110 MPa
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let aluminum = Material::aluminum();
    /// assert_eq!(aluminum.density, 2700.0);
    /// ```
    pub fn aluminum() -> Self {
        Self::new(
            2700.0,             // density (kg/m³)
            69.0e9,             // Young's modulus (Pa)
            0.33,               // Poisson's ratio
            0.61,               // friction coefficient
            0.75,               // restitution coefficient
            0.001,              // rolling resistance coefficient (aluminum on aluminum)
            237.0,              // thermal conductivity (W/(m·K))
            900.0,              // specific heat capacity (J/(kg·K))
            95.0e6,             // yield strength (Pa)
            110.0e6,            // ultimate strength (Pa)
        ).expect("Failed to create aluminum material")
    }

    /// Creates a new Material instance with properties of rubber.
    ///
    /// # Returns
    ///
    /// A Material instance with typical properties of rubber:
    /// * Density: 1100 kg/m³
    /// * Young's modulus: 0.01 GPa
    /// * Poisson's ratio: 0.49
    /// * Friction coefficient: 0.9
    /// * Restitution coefficient: 0.95
    /// * Rolling resistance coefficient: 0.02 (rubber deforms significantly)
    /// * Thermal conductivity: 0.16 W/(m·K)
    /// * Specific heat capacity: 2000 J/(kg·K)
    /// * Yield strength: 7 MPa
    /// * Ultimate strength: 15 MPa
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let rubber = Material::rubber();
    /// assert_eq!(rubber.density, 1100.0);
    /// ```
    pub fn rubber() -> Self {
        Self::new(
            1100.0,             // density (kg/m³)
            0.01e9,             // Young's modulus (Pa)
            0.49,               // Poisson's ratio
            0.9,                // friction coefficient
            0.95,               // restitution coefficient
            0.02,               // rolling resistance coefficient (rubber deforms significantly)
            0.16,               // thermal conductivity (W/(m·K))
            2000.0,             // specific heat capacity (J/(kg·K))
            7.0e6,              // yield strength (Pa)
            15.0e6,             // ultimate strength (Pa)
        ).expect("Failed to create rubber material")
    }

    /// Creates a new Material instance with properties of polyurethane.
    ///
    /// # Returns
    ///
    /// A Material instance with typical properties of polyurethane:
    /// * Density: 1200 kg/m³
    /// * Young's modulus: 0.02 GPa
    /// * Poisson's ratio: 0.45
    /// * Friction coefficient: 0.8
    /// * Restitution coefficient: 0.7
    /// * Rolling resistance coefficient: 0.015 (softer than rubber)
    /// * Thermal conductivity: 0.2 W/(m·K)
    /// * Specific heat capacity: 1800 J/(kg·K)
    /// * Yield strength: 35 MPa
    /// * Ultimate strength: 55 MPa
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let polyurethane = Material::polyurethane();
    /// assert_eq!(polyurethane.density, 1200.0);
    /// ```
    pub fn polyurethane() -> Self {
        Self::new(
            1200.0,             // density (kg/m³)
            0.02e9,             // Young's modulus (Pa)
            0.45,               // Poisson's ratio
            0.8,                // friction coefficient
            0.7,                // restitution coefficient
            0.015,              // rolling resistance coefficient (softer than rubber)
            0.2,                // thermal conductivity (W/(m·K))
            1800.0,             // specific heat capacity (J/(kg·K))
            35.0e6,             // yield strength (Pa)
            55.0e6,             // ultimate strength (Pa)
        ).expect("Failed to create polyurethane material")
    }

    /// Creates a new Material instance with properties of wood (hardwood).
    ///
    /// # Returns
    ///
    /// A Material instance with typical properties of hardwood:
    /// * Density: 700 kg/m³
    /// * Young's modulus: 12 GPa
    /// * Poisson's ratio: 0.3
    /// * Friction coefficient: 0.5
    /// * Restitution coefficient: 0.5
    /// * Rolling resistance coefficient: 0.01 (wood on wood)
    /// * Thermal conductivity: 0.15 W/(m·K)
    /// * Specific heat capacity: 1700 J/(kg·K)
    /// * Yield strength: 40 MPa
    /// * Ultimate strength: 70 MPa
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let wood = Material::wood();
    /// assert_eq!(wood.density, 700.0);
    /// ```
    pub fn wood() -> Self {
        Self::new(
            700.0,              // density (kg/m³)
            12.0e9,             // Young's modulus (Pa)
            0.3,                // Poisson's ratio
            0.5,                // friction coefficient
            0.5,                // restitution coefficient
            0.01,               // rolling resistance coefficient (wood on wood)
            0.15,               // thermal conductivity (W/(m·K))
            1700.0,             // specific heat capacity (J/(kg·K))
            40.0e6,             // yield strength (Pa)
            70.0e6,             // ultimate strength (Pa)
        ).expect("Failed to create wood material")
    }

    /// Creates a new Material instance with properties of copper.
    ///
    /// Copper is an excellent conductor of heat and electricity, making it
    /// ideal for thermal and electrical simulations.
    ///
    /// # Returns
    ///
    /// A Material instance with typical properties of pure copper:
    /// * Density: 8960 kg/m³
    /// * Young's modulus: 110 GPa
    /// * Poisson's ratio: 0.34
    /// * Friction coefficient: 0.4
    /// * Restitution coefficient: 0.75
    /// * Rolling resistance coefficient: 0.002 (copper on copper)
    /// * Thermal conductivity: 401 W/(m·K) (highest of common metals)
    /// * Specific heat capacity: 385 J/(kg·K)
    /// * Yield strength: 70 MPa (annealed)
    /// * Ultimate strength: 220 MPa
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let copper = Material::copper();
    /// assert_eq!(copper.density, 8960.0);
    /// assert!(copper.thermal_conductivity > 400.0);  // Excellent conductor
    /// ```
    pub fn copper() -> Self {
        Self::new(
            8960.0,             // density (kg/m³)
            110.0e9,            // Young's modulus (Pa)
            0.34,               // Poisson's ratio
            0.4,                // friction coefficient
            0.75,               // restitution coefficient
            0.002,              // rolling resistance coefficient (copper on copper)
            401.0,              // thermal conductivity (W/(m·K))
            385.0,              // specific heat capacity (J/(kg·K))
            70.0e6,             // yield strength (Pa) - annealed
            220.0e6,            // ultimate strength (Pa)
        ).expect("Failed to create copper material")
    }

    /// Creates a new Material instance with properties of titanium (Ti-6Al-4V).
    ///
    /// Titanium alloy offers excellent strength-to-weight ratio and corrosion resistance,
    /// commonly used in aerospace and medical applications.
    ///
    /// # Returns
    ///
    /// A Material instance with typical properties of Ti-6Al-4V:
    /// * Density: 4430 kg/m³
    /// * Young's modulus: 114 GPa
    /// * Poisson's ratio: 0.34
    /// * Friction coefficient: 0.36
    /// * Restitution coefficient: 0.8
    /// * Rolling resistance coefficient: 0.002 (titanium on titanium)
    /// * Thermal conductivity: 6.7 W/(m·K)
    /// * Specific heat capacity: 526 J/(kg·K)
    /// * Yield strength: 880 MPa
    /// * Ultimate strength: 950 MPa
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let titanium = Material::titanium();
    /// let steel = Material::steel();
    ///
    /// // Titanium is lighter than steel
    /// assert!(titanium.density < steel.density);
    /// // But has higher yield strength
    /// assert!(titanium.yield_strength > steel.yield_strength);
    /// ```
    pub fn titanium() -> Self {
        Self::new(
            4430.0,             // density (kg/m³)
            114.0e9,            // Young's modulus (Pa)
            0.34,               // Poisson's ratio
            0.36,               // friction coefficient
            0.8,                // restitution coefficient
            0.002,              // rolling resistance coefficient (titanium on titanium)
            6.7,                // thermal conductivity (W/(m·K))
            526.0,              // specific heat capacity (J/(kg·K))
            880.0e6,            // yield strength (Pa)
            950.0e6,            // ultimate strength (Pa)
        ).expect("Failed to create titanium material")
    }

    /// Creates a new Material instance with properties of concrete.
    ///
    /// Concrete is strong in compression but weak in tension.
    /// Note: The yield and ultimate strengths here represent compressive strength.
    ///
    /// # Returns
    ///
    /// A Material instance with typical properties of structural concrete:
    /// * Density: 2400 kg/m³
    /// * Young's modulus: 30 GPa
    /// * Poisson's ratio: 0.2
    /// * Friction coefficient: 0.6
    /// * Restitution coefficient: 0.2
    /// * Rolling resistance coefficient: 0.015 (rough surface)
    /// * Thermal conductivity: 1.7 W/(m·K)
    /// * Specific heat capacity: 880 J/(kg·K)
    /// * Yield strength: 25 MPa (compressive)
    /// * Ultimate strength: 40 MPa (compressive)
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let concrete = Material::concrete();
    /// assert_eq!(concrete.density, 2400.0);
    /// ```
    pub fn concrete() -> Self {
        Self::new(
            2400.0,             // density (kg/m³)
            30.0e9,             // Young's modulus (Pa)
            0.2,                // Poisson's ratio
            0.6,                // friction coefficient
            0.2,                // restitution coefficient (low - absorbs energy)
            0.015,              // rolling resistance coefficient (rough surface)
            1.7,                // thermal conductivity (W/(m·K))
            880.0,              // specific heat capacity (J/(kg·K))
            25.0e6,             // yield strength (Pa) - compressive
            40.0e6,             // ultimate strength (Pa) - compressive
        ).expect("Failed to create concrete material")
    }

    /// Creates a new Material instance with properties of glass (soda-lime).
    ///
    /// Glass is a brittle material with no significant plastic deformation.
    /// It fails catastrophically when stress exceeds the yield point.
    ///
    /// # Returns
    ///
    /// A Material instance with typical properties of soda-lime glass:
    /// * Density: 2500 kg/m³
    /// * Young's modulus: 70 GPa
    /// * Poisson's ratio: 0.22
    /// * Friction coefficient: 0.4
    /// * Restitution coefficient: 0.65
    /// * Rolling resistance coefficient: 0.003 (smooth, hard surface)
    /// * Thermal conductivity: 1.0 W/(m·K)
    /// * Specific heat capacity: 840 J/(kg·K)
    /// * Yield strength: 33 MPa (practical strength)
    /// * Ultimate strength: 33 MPa (brittle - no plastic region)
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let glass = Material::glass();
    /// // Glass is brittle - yield equals ultimate (no plastic deformation)
    /// assert_eq!(glass.yield_strength, glass.ultimate_strength);
    /// ```
    pub fn glass() -> Self {
        Self::new(
            2500.0,             // density (kg/m³)
            70.0e9,             // Young's modulus (Pa)
            0.22,               // Poisson's ratio
            0.4,                // friction coefficient
            0.65,               // restitution coefficient
            0.003,              // rolling resistance coefficient (smooth, hard surface)
            1.0,                // thermal conductivity (W/(m·K))
            840.0,              // specific heat capacity (J/(kg·K))
            33.0e6,             // yield strength (Pa) - practical strength
            33.0e6,             // ultimate strength (Pa) - same as yield (brittle)
        ).expect("Failed to create glass material")
    }

    /// Creates a new Material instance with properties of brass (70/30).
    ///
    /// Brass is a copper-zinc alloy with good machinability and corrosion resistance.
    ///
    /// # Returns
    ///
    /// A Material instance with typical properties of 70/30 brass:
    /// * Density: 8530 kg/m³
    /// * Young's modulus: 110 GPa
    /// * Poisson's ratio: 0.34
    /// * Friction coefficient: 0.35
    /// * Restitution coefficient: 0.6
    /// * Rolling resistance coefficient: 0.002 (brass on brass)
    /// * Thermal conductivity: 109 W/(m·K)
    /// * Specific heat capacity: 380 J/(kg·K)
    /// * Yield strength: 200 MPa
    /// * Ultimate strength: 400 MPa
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let brass = Material::brass();
    /// let copper = Material::copper();
    ///
    /// // Brass has lower thermal conductivity than pure copper
    /// assert!(brass.thermal_conductivity < copper.thermal_conductivity);
    /// ```
    pub fn brass() -> Self {
        Self::new(
            8530.0,             // density (kg/m³)
            110.0e9,            // Young's modulus (Pa)
            0.34,               // Poisson's ratio
            0.35,               // friction coefficient
            0.6,                // restitution coefficient
            0.002,              // rolling resistance coefficient (brass on brass)
            109.0,              // thermal conductivity (W/(m·K))
            380.0,              // specific heat capacity (J/(kg·K))
            200.0e6,            // yield strength (Pa)
            400.0e6,            // ultimate strength (Pa)
        ).expect("Failed to create brass material")
    }

    /// Creates a new Material instance with properties of ice (at 0°C).
    ///
    /// Ice is useful for thermal and phase transition simulations.
    ///
    /// # Returns
    ///
    /// A Material instance with typical properties of ice at 0°C:
    /// * Density: 917 kg/m³
    /// * Young's modulus: 9.3 GPa
    /// * Poisson's ratio: 0.33
    /// * Friction coefficient: 0.03 (very slippery)
    /// * Restitution coefficient: 0.3
    /// * Rolling resistance coefficient: 0.001 (very smooth surface)
    /// * Thermal conductivity: 2.2 W/(m·K)
    /// * Specific heat capacity: 2090 J/(kg·K)
    /// * Yield strength: 1 MPa
    /// * Ultimate strength: 2 MPa
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let ice = Material::ice();
    /// // Ice has very low friction
    /// assert!(ice.friction_coefficient < 0.1);
    /// ```
    pub fn ice() -> Self {
        Self::new(
            917.0,              // density (kg/m³)
            9.3e9,              // Young's modulus (Pa)
            0.33,               // Poisson's ratio
            0.03,               // friction coefficient (very low)
            0.3,                // restitution coefficient
            0.001,              // rolling resistance coefficient (very smooth surface)
            2.2,                // thermal conductivity (W/(m·K))
            2090.0,             // specific heat capacity (J/(kg·K))
            1.0e6,              // yield strength (Pa)
            2.0e6,              // ultimate strength (Pa)
        ).expect("Failed to create ice material")
    }

    /// Creates a new Material instance with properties of stainless steel (304).
    ///
    /// Stainless steel 304 is the most common stainless steel grade,
    /// offering good corrosion resistance with slightly different mechanical
    /// properties than carbon steel.
    ///
    /// # Returns
    ///
    /// A Material instance with typical properties of 304 stainless steel:
    /// * Density: 8000 kg/m³
    /// * Young's modulus: 193 GPa
    /// * Poisson's ratio: 0.29
    /// * Friction coefficient: 0.5
    /// * Restitution coefficient: 0.8
    /// * Rolling resistance coefficient: 0.002 (stainless steel on stainless steel)
    /// * Thermal conductivity: 16.2 W/(m·K)
    /// * Specific heat capacity: 500 J/(kg·K)
    /// * Yield strength: 215 MPa
    /// * Ultimate strength: 505 MPa
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let stainless = Material::stainless_steel();
    /// let carbon = Material::steel();
    ///
    /// // Stainless has lower thermal conductivity than carbon steel
    /// assert!(stainless.thermal_conductivity < carbon.thermal_conductivity);
    /// ```
    pub fn stainless_steel() -> Self {
        Self::new(
            8000.0,             // density (kg/m³)
            193.0e9,            // Young's modulus (Pa)
            0.29,               // Poisson's ratio
            0.5,                // friction coefficient
            0.8,                // restitution coefficient
            0.002,              // rolling resistance coefficient (stainless steel on stainless steel)
            16.2,               // thermal conductivity (W/(m·K))
            500.0,              // specific heat capacity (J/(kg·K))
            215.0e6,            // yield strength (Pa)
            505.0e6,            // ultimate strength (Pa)
        ).expect("Failed to create stainless steel material")
    }

    /// Calculates the shear modulus of the material.
    ///
    /// The shear modulus (G) is calculated from Young's modulus (E) and
    /// Poisson's ratio (ν) using the formula: G = E / (2(1 + ν))
    ///
    /// # Returns
    ///
    /// The shear modulus in Pascals (Pa)
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let steel = Material::steel();
    /// let shear_modulus = steel.shear_modulus();
    /// ```
    pub fn shear_modulus(&self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))
    }


    /// Calculates the bulk modulus of the material.
    ///
    /// The bulk modulus (K) is calculated from Young's modulus (E) and
    /// Poisson's ratio (ν) using the formula: K = E / (3(1 - 2ν))
    ///
    /// # Returns
    ///
    /// The bulk modulus in Pascals (Pa)
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let steel = Material::steel();
    /// let bulk_modulus = steel.bulk_modulus();
    /// ```
    pub fn bulk_modulus(&self) -> f64 {
        self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poisson_ratio))
    }


    /// Calculates the strain energy density at a given strain.
    ///
    /// # Arguments
    ///
    /// * `strain` - The strain value (dimensionless)
    ///
    /// # Returns
    ///
    /// The strain energy density in Joules per cubic meter (J/m³)
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let steel = Material::steel();
    /// let energy_density = steel.strain_energy_density(0.001);
    /// ```
    pub fn strain_energy_density(&self, strain: f64) -> f64 {
        0.5 * self.youngs_modulus * strain * strain
    }

    /// Determines if a material will break under given conditions.
    ///
    /// # Arguments
    ///
    /// * `stress` - Applied stress in Pascals (Pa)
    /// * `strain` - Applied strain (dimensionless)
    /// * `cycles` - Number of loading cycles (optional)
    ///
    /// # Returns
    ///
    /// A `BreakageResult` indicating if and how the material will fail
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::{Material, BreakageType};
    ///
    /// let steel = Material::steel();
    ///
    /// // Check elastic region (200 MPa stress, 0.1% strain)
    /// let elastic_result = steel.will_break(200e6, 0.001, None);
    /// assert_eq!(elastic_result.will_break, false);
    /// assert_eq!(elastic_result.breakage_type, BreakageType::None);
    ///
    /// // Check plastic region (280 MPa stress, 0.15% strain)
    /// let plastic_result = steel.will_break(280e6, 0.0015, None);
    /// assert_eq!(plastic_result.breakage_type, BreakageType::Plastic);
    ///
    /// // Check ultimate failure (420 MPa stress)
    /// let failure_result = steel.will_break(420e6, 0.0021, None);
    /// assert_eq!(failure_result.will_break, true);
    /// assert_eq!(failure_result.breakage_type, BreakageType::TensileStress);
    /// ```
    ///
    /// # Notes
    ///
    /// For steel (default properties):
    /// - Yield point: 250 MPa (stress), 0.125% (strain)
    /// - Ultimate strength: 400 MPa (stress), 0.2% (strain)
    ///
    /// The method determines failure mode based on both stress and strain:
    /// - Below yield: Elastic deformation (BreakageType::None)
    /// - Above yield but below ultimate: Plastic deformation (BreakageType::Plastic)
    /// - Above ultimate: Material failure (BreakageType::TensileStress or TensileStrain)
    pub fn will_break(&self, stress: f64, strain: f64, cycles: Option<u64>) -> BreakageResult {
        // Calculate material limits
        let yield_strain = self.yield_strength / self.youngs_modulus;
        let ultimate_strain = self.ultimate_strength / self.youngs_modulus;

        // Calculate the actual strain that would result from the applied stress
        let stress_induced_strain = stress / self.youngs_modulus;

        // Use the largest of the actual strain and the stress-induced strain
        let effective_strain = strain.max(stress_induced_strain);

        // Check failure modes in order of severity
        if stress >= self.ultimate_strength {
            return BreakageResult {
                will_break: true,
                breakage_type: BreakageType::TensileStress,
                safety_factor: self.ultimate_strength / stress,
            };
        }

        if effective_strain >= ultimate_strain {
            return BreakageResult {
                will_break: true,
                breakage_type: BreakageType::TensileStrain,
                safety_factor: ultimate_strain / effective_strain,
            };
        }

        // Check plastic deformation
        if stress >= self.yield_strength || effective_strain >= yield_strain {
            return BreakageResult {
                will_break: false,
                breakage_type: BreakageType::Plastic,
                safety_factor: (self.yield_strength / stress)
                    .min(yield_strain / effective_strain),
            };
        }

        // Check fatigue failure if cycles provided
        if let Some(cycle_count) = cycles {
            let fatigue_strength = self.calculate_fatigue_strength(cycle_count);
            if stress >= fatigue_strength {
                return BreakageResult {
                    will_break: true,
                    breakage_type: BreakageType::Fatigue,
                    safety_factor: fatigue_strength / stress,
                };
            }
        }

        // No failure detected
        BreakageResult {
            will_break: false,
            breakage_type: BreakageType::None,
            safety_factor: (self.yield_strength / stress)
                .min(yield_strain / effective_strain),
        }
    }

    /// Calculates the maximum allowable stress before failure.
    ///
    /// # Arguments
    ///
    /// * `cycles` - Optional number of loading cycles to consider fatigue
    /// * `safety_factor` - Desired safety factor (typically 1.5 to 3.0)
    ///
    /// # Returns
    ///
    /// Maximum allowable stress in Pascals (Pa)
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let steel = Material::steel();
    /// let max_stress = steel.maximum_allowable_stress(None, 2.0);
    /// ```
    pub fn maximum_allowable_stress(&self, cycles: Option<u64>, safety_factor: f64) -> f64 {
        let static_limit = self.yield_strength / safety_factor;

        if let Some(cycle_count) = cycles {
            let fatigue_limit = self.calculate_fatigue_strength(cycle_count) / safety_factor;
            static_limit.min(fatigue_limit)
        } else {
            static_limit
        }
    }

    /// Estimates the remaining cycles until failure under given loading conditions.
    ///
    /// # Arguments
    ///
    /// * `stress` - Applied stress in Pascals (Pa)
    /// * `mean_stress` - Mean stress in Pascals (Pa)
    ///
    /// # Returns
    ///
    /// Estimated number of cycles until failure
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let steel = Material::steel();
    /// let remaining_cycles = steel.estimate_remaining_cycles(200e6, 100e6);
    /// ```
    pub fn estimate_remaining_cycles(&self, stress: f64, mean_stress: f64) -> u64 {
        // Basquin's equation: S^m * N = C
        // where S is stress, N is cycles, m and C are material constants
        let m = 3.0; // Typical value for metals
        let endurance_limit = self.yield_strength * 0.5; // Simplified endurance limit

        // Goodman correction for mean stress
        let stress_amplitude = stress - mean_stress;
        let corrected_stress = stress_amplitude / (1.0 - mean_stress / self.ultimate_strength);

        if corrected_stress <= endurance_limit {
            return u64::MAX; // Infinite life
        }

        // Calculate cycles to failure
        let c = self.ultimate_strength.powf(m) * 1000.0; // Simplified material constant
        (c / corrected_stress.powf(m)) as u64
    }

    /// Calculates the fatigue strength based on number of cycles.
    ///
    /// This method implements a simplified S-N (stress-life) curve commonly used
    /// in fatigue analysis. The fatigue strength decreases logarithmically as
    /// the number of loading cycles increases.
    ///
    /// # Arguments
    ///
    /// * `cycles` - The number of loading cycles
    ///
    /// # Returns
    ///
    /// The fatigue strength in Pascals (Pa)
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let steel = Material::steel();
    ///
    /// // Low cycle fatigue (< 1000 cycles) - uses ultimate strength
    /// let low_cycle = steel.calculate_fatigue_strength(500);
    /// assert_eq!(low_cycle, steel.ultimate_strength);
    ///
    /// // High cycle fatigue (> 1,000,000 cycles) - uses endurance limit
    /// let high_cycle = steel.calculate_fatigue_strength(2_000_000);
    /// assert!((high_cycle - steel.yield_strength * 0.5).abs() < 1.0);
    /// ```
    ///
    /// # Physics Background
    ///
    /// The S-N curve (Wöhler curve) describes the relationship between stress
    /// amplitude and number of cycles to failure:
    /// - Low cycle fatigue (N < 10³): Strength approaches ultimate tensile strength
    /// - High cycle fatigue (N > 10⁶): Strength approaches endurance limit (~0.5 × yield)
    /// - Transition region: Log-linear interpolation between these limits
    pub fn calculate_fatigue_strength(&self, cycles: u64) -> f64 {
        // Simplified implementation of the S-N curve
        let endurance_limit = self.yield_strength * 0.5;
        if cycles < 1000 {
            self.ultimate_strength
        } else if cycles > 1_000_000 {
            endurance_limit
        } else {
            // Log-linear interpolation between ultimate strength and endurance limit
            let log_cycles = (cycles as f64).log10();
            let factor = (log_cycles - 3.0) / 3.0; // 3.0 represents log10(1000)
            self.ultimate_strength - (self.ultimate_strength - endurance_limit) * factor
        }
    }

    /// Calculates the thermal diffusivity of the material.
    ///
    /// Thermal diffusivity (α) measures how quickly a material can conduct heat
    /// relative to how much heat it can store. It is calculated as:
    /// α = k / (ρ × c)
    ///
    /// where:
    /// - k is thermal conductivity (W/(m·K))
    /// - ρ is density (kg/m³)
    /// - c is specific heat capacity (J/(kg·K))
    ///
    /// # Returns
    ///
    /// The thermal diffusivity in m²/s
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let copper = Material::copper();
    /// let aluminum = Material::aluminum();
    ///
    /// // Copper has higher thermal diffusivity than aluminum
    /// assert!(copper.thermal_diffusivity() > aluminum.thermal_diffusivity());
    /// ```
    ///
    /// # Physics Background
    ///
    /// Higher thermal diffusivity means the material reaches thermal equilibrium faster.
    /// Metals typically have high diffusivity (10⁻⁵ to 10⁻⁴ m²/s), while insulators
    /// have low diffusivity (10⁻⁷ to 10⁻⁶ m²/s).
    pub fn thermal_diffusivity(&self) -> f64 {
        self.thermal_conductivity / (self.density * self.specific_heat_capacity)
    }
}

/// Default implementation for Material.
///
/// Returns a steel material as the default, which is a common engineering reference material.
///
/// # Examples
///
/// ```
/// use rs_physics::materials::Material;
///
/// let material = Material::default();
/// assert_eq!(material.density, 7850.0);  // Steel density
/// ```
impl Default for Material {
    fn default() -> Self {
        Self::steel()
    }
}

/// PartialEq implementation for Material.
///
/// Compares all fields for approximate equality using a relative tolerance
/// of 1e-10 for floating point comparisons.
///
/// # Examples
///
/// ```
/// use rs_physics::materials::Material;
///
/// let steel1 = Material::steel();
/// let steel2 = Material::steel();
/// assert_eq!(steel1, steel2);
///
/// let aluminum = Material::aluminum();
/// assert_ne!(steel1, aluminum);
/// ```
impl PartialEq for Material {
    fn eq(&self, other: &Self) -> bool {
        const EPSILON: f64 = 1e-10;

        fn approx_eq(a: f64, b: f64) -> bool {
            if a == b { return true; }
            let diff = (a - b).abs();
            let max = a.abs().max(b.abs());
            if max == 0.0 { return diff < EPSILON; }
            diff / max < EPSILON
        }

        approx_eq(self.density, other.density)
            && approx_eq(self.youngs_modulus, other.youngs_modulus)
            && approx_eq(self.poisson_ratio, other.poisson_ratio)
            && approx_eq(self.friction_coefficient, other.friction_coefficient)
            && approx_eq(self.restitution_coefficient, other.restitution_coefficient)
            && approx_eq(self.rolling_resistance_coefficient, other.rolling_resistance_coefficient)
            && approx_eq(self.thermal_conductivity, other.thermal_conductivity)
            && approx_eq(self.specific_heat_capacity, other.specific_heat_capacity)
            && approx_eq(self.yield_strength, other.yield_strength)
            && approx_eq(self.ultimate_strength, other.ultimate_strength)
    }
}

/// Builder for creating custom materials with a fluent API.
///
/// The builder starts with default values (steel) and allows you to
/// customize individual properties before building the final material.
///
/// # Examples
///
/// ## Creating a custom material
///
/// ```
/// use rs_physics::materials::MaterialBuilder;
///
/// let custom = MaterialBuilder::new()
///     .density(5000.0)
///     .youngs_modulus(150e9)
///     .friction_coefficient(0.5)
///     .build()
///     .unwrap();
///
/// assert_eq!(custom.density, 5000.0);
/// ```
///
/// ## Starting from an existing material
///
/// ```
/// use rs_physics::materials::{Material, MaterialBuilder};
///
/// // Modify aluminum's friction coefficient
/// let modified_aluminum = MaterialBuilder::from(Material::aluminum())
///     .friction_coefficient(0.8)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MaterialBuilder {
    density: f64,
    youngs_modulus: f64,
    poisson_ratio: f64,
    friction_coefficient: f64,
    restitution_coefficient: f64,
    rolling_resistance_coefficient: f64,
    thermal_conductivity: f64,
    specific_heat_capacity: f64,
    yield_strength: f64,
    ultimate_strength: f64,
}

impl MaterialBuilder {
    /// Creates a new MaterialBuilder with default values (steel properties).
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::MaterialBuilder;
    ///
    /// let builder = MaterialBuilder::new();
    /// let material = builder.build().unwrap();
    /// assert_eq!(material.density, 7850.0);  // Steel default
    /// ```
    pub fn new() -> Self {
        // Start with steel as default
        let steel = Material::steel();
        Self {
            density: steel.density,
            youngs_modulus: steel.youngs_modulus,
            poisson_ratio: steel.poisson_ratio,
            friction_coefficient: steel.friction_coefficient,
            restitution_coefficient: steel.restitution_coefficient,
            rolling_resistance_coefficient: steel.rolling_resistance_coefficient,
            thermal_conductivity: steel.thermal_conductivity,
            specific_heat_capacity: steel.specific_heat_capacity,
            yield_strength: steel.yield_strength,
            ultimate_strength: steel.ultimate_strength,
        }
    }

    /// Sets the density in kg/m³.
    pub fn density(mut self, density: f64) -> Self {
        self.density = density;
        self
    }

    /// Sets the Young's modulus in Pascals (Pa).
    pub fn youngs_modulus(mut self, youngs_modulus: f64) -> Self {
        self.youngs_modulus = youngs_modulus;
        self
    }

    /// Sets the Poisson's ratio (dimensionless, typically -1 to 0.5).
    pub fn poisson_ratio(mut self, poisson_ratio: f64) -> Self {
        self.poisson_ratio = poisson_ratio;
        self
    }

    /// Sets the friction coefficient (dimensionless, >= 0).
    pub fn friction_coefficient(mut self, friction_coefficient: f64) -> Self {
        self.friction_coefficient = friction_coefficient;
        self
    }

    /// Sets the restitution coefficient (dimensionless, 0 to 1).
    pub fn restitution_coefficient(mut self, restitution_coefficient: f64) -> Self {
        self.restitution_coefficient = restitution_coefficient;
        self
    }

    /// Sets the rolling resistance coefficient (dimensionless, >= 0).
    /// Typical values: 0.001-0.005 for hard materials, 0.01-0.03 for rubber.
    pub fn rolling_resistance_coefficient(mut self, rolling_resistance_coefficient: f64) -> Self {
        self.rolling_resistance_coefficient = rolling_resistance_coefficient;
        self
    }

    /// Sets the thermal conductivity in W/(m·K).
    pub fn thermal_conductivity(mut self, thermal_conductivity: f64) -> Self {
        self.thermal_conductivity = thermal_conductivity;
        self
    }

    /// Sets the specific heat capacity in J/(kg·K).
    pub fn specific_heat_capacity(mut self, specific_heat_capacity: f64) -> Self {
        self.specific_heat_capacity = specific_heat_capacity;
        self
    }

    /// Sets the yield strength in Pascals (Pa).
    pub fn yield_strength(mut self, yield_strength: f64) -> Self {
        self.yield_strength = yield_strength;
        self
    }

    /// Sets the ultimate strength in Pascals (Pa).
    pub fn ultimate_strength(mut self, ultimate_strength: f64) -> Self {
        self.ultimate_strength = ultimate_strength;
        self
    }

    /// Builds the Material, validating all properties.
    ///
    /// # Returns
    ///
    /// * `Ok(Material)` - A valid Material with the specified properties
    /// * `Err(PhysicsError)` - If any property violates constraints
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::MaterialBuilder;
    ///
    /// // Valid material
    /// let material = MaterialBuilder::new()
    ///     .density(5000.0)
    ///     .build();
    /// assert!(material.is_ok());
    ///
    /// // Invalid material (negative density)
    /// let invalid = MaterialBuilder::new()
    ///     .density(-100.0)
    ///     .build();
    /// assert!(invalid.is_err());
    /// ```
    pub fn build(self) -> Result<Material, PhysicsError> {
        Material::new(
            self.density,
            self.youngs_modulus,
            self.poisson_ratio,
            self.friction_coefficient,
            self.restitution_coefficient,
            self.rolling_resistance_coefficient,
            self.thermal_conductivity,
            self.specific_heat_capacity,
            self.yield_strength,
            self.ultimate_strength,
        )
    }
}

impl Default for MaterialBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Material> for MaterialBuilder {
    /// Creates a MaterialBuilder from an existing Material.
    ///
    /// This allows you to start with a predefined material and modify
    /// only the properties you want to change.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::{Material, MaterialBuilder};
    ///
    /// let modified = MaterialBuilder::from(Material::steel())
    ///     .friction_coefficient(0.9)
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(modified.density, 7850.0);  // Unchanged
    /// assert_eq!(modified.friction_coefficient, 0.9);  // Modified
    /// ```
    fn from(material: Material) -> Self {
        Self {
            density: material.density,
            youngs_modulus: material.youngs_modulus,
            poisson_ratio: material.poisson_ratio,
            friction_coefficient: material.friction_coefficient,
            restitution_coefficient: material.restitution_coefficient,
            rolling_resistance_coefficient: material.rolling_resistance_coefficient,
            thermal_conductivity: material.thermal_conductivity,
            specific_heat_capacity: material.specific_heat_capacity,
            yield_strength: material.yield_strength,
            ultimate_strength: material.ultimate_strength,
        }
    }
}

/// Calculates the collision response between two materials.
///
/// # Arguments
///
/// * `material1` - Reference to the first material
/// * `material2` - Reference to the second material
/// * `relative_velocity` - The relative velocity between the materials in m/s
/// * `contact_angle` - The angle of contact in radians
///
/// # Returns
///
/// A tuple containing:
/// * The new normal velocity component
/// * The new tangential velocity component
///
/// # Examples
///
/// ```
/// use rs_physics::materials::{Material, calculate_collision_response};
/// use std::f64::consts::PI;
///
/// let steel = Material::steel();
/// let aluminum = Material::aluminum();
/// let (normal_v, tangential_v) = calculate_collision_response(&steel, &aluminum, 10.0, PI/4.0);
/// ```
pub fn calculate_collision_response(
    material1: &Material,
    material2: &Material,
    relative_velocity: f64,
    contact_angle: f64,
) -> (f64, f64) {
    // Calculate effective coefficient of restitution
    let effective_restitution = (material1.restitution_coefficient + material2.restitution_coefficient) / 2.0;

    // Calculate effective coefficient of friction
    let effective_friction = (material1.friction_coefficient * material2.friction_coefficient).sqrt();

    // Calculate normal and tangential components
    let normal_velocity = relative_velocity * contact_angle.cos();
    let tangential_velocity = relative_velocity * contact_angle.sin();

    // Apply restitution to normal component
    let new_normal_velocity = -normal_velocity * effective_restitution;

    // Apply friction to tangential component
    let friction_force = effective_friction * normal_velocity.abs();
    let new_tangential_velocity = if tangential_velocity.abs() <= friction_force {
        0.0 // Static friction case
    } else {
        // Dynamic friction case
        tangential_velocity - friction_force * tangential_velocity.signum()
    };

    (new_normal_velocity, new_tangential_velocity)
}

/// Calculates the heat generated during a collision between two materials.
///
/// # Arguments
///
/// * `material1` - Reference to the first material
/// * `material2` - Reference to the second material
/// * `relative_velocity` - The relative velocity between the materials in m/s
/// * `contact_area` - The area of contact during collision in m²
///
/// # Returns
///
/// The heat generated during the collision in Joules (J)
///
/// # Examples
///
/// ```
/// use rs_physics::materials::{Material, calculate_collision_heat_generation};
///
/// let steel = Material::steel();
/// let aluminum = Material::aluminum();
/// let heat = calculate_collision_heat_generation(&steel, &aluminum, 10.0, 0.01);
/// ```
///
/// # Notes
///
/// The heat generation is calculated based on the energy lost during collision,
/// which is determined by the coefficient of restitution of both materials.
/// This is a simplified model that assumes all lost kinetic energy is converted to heat.
pub fn calculate_collision_heat_generation(
    material1: &Material,
    material2: &Material,
    relative_velocity: f64,
    contact_area: f64,
) -> f64 {
    let effective_restitution = (material1.restitution_coefficient + material2.restitution_coefficient) / 2.0;
    let energy_loss = 0.5 * (1.0 - effective_restitution * effective_restitution) * relative_velocity * relative_velocity;

    // Convert lost kinetic energy to heat
    energy_loss * contact_area
}

/// Calculates the stress in a material under a given strain.
///
/// This function implements a combined elastic-plastic model:
/// - For strains resulting in stress below yield strength, uses linear elastic behavior (Hooke's law)
/// - For strains beyond yield point, uses a simplified plastic deformation model
///
/// # Arguments
///
/// * `material` - Reference to the material
/// * `strain` - The strain value (dimensionless)
///
/// # Returns
///
/// The stress in the material in Pascals (Pa)
///
/// # Examples
///
/// ```
/// use rs_physics::materials::{Material, calculate_stress};
///
/// let steel = Material::steel();
///
/// // Elastic region
/// let elastic_stress = calculate_stress(&steel, 0.001);
///
/// // Plastic region
/// let plastic_stress = calculate_stress(&steel, 0.01);
/// ```
///
/// # Notes
///
/// The plastic deformation model uses an exponential function to simulate
/// strain hardening, where stress increases more slowly after yielding
/// until reaching the ultimate strength. This is a simplified model and
/// may not accurately represent all materials' behavior in the plastic region.
///
/// The stress-strain relationship is:
/// - Elastic region (σ = E·ε): Linear relationship up to yield point
/// - Plastic region: Exponential approach to ultimate strength
///
/// # Physics Background
///
/// - Below yield strength: Uses Hooke's law (σ = E·ε)
/// - Above yield strength: Uses a continuous function that:
///   * Starts at yield strength
///   * Asymptotically approaches ultimate strength
///   * Has continuous first derivative at yield point
pub fn calculate_stress(material: &Material, strain: f64) -> f64 {
    // Using Hooke's law for linear elastic region
    if strain * material.youngs_modulus <= material.yield_strength {
        strain * material.youngs_modulus
    } else {
        // Simple plastic deformation model
        material.yield_strength +
            (material.ultimate_strength - material.yield_strength) *
                (1.0 - (-5.0 * (strain - material.yield_strength / material.youngs_modulus)).exp())
    }
}