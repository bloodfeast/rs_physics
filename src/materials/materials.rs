// src/materials.rs

use crate::utils::PhysicsError;

/// Represents different types of material failure
#[derive(Debug, PartialEq)]
pub enum BreakageType {
    None,           // No failure
    TensileStress,  // Immediate failure due to exceeding ultimate strength
    TensileStrain,  // Immediate failure due to exceeding ultimate strain
    Plastic,        // Plastic deformation (yield point exceeded)
    Fatigue,        // Failure due to cyclic loading
}
/// Result of material failure analysis
#[derive(Debug)]
pub struct BreakageResult {
    /// Whether the material will break under given conditions
    pub will_break: bool,
    /// Type of failure or deformation
    pub breakage_type: BreakageType,
    /// Ratio of allowable stress to applied stress
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
            7850.0,                    // density (kg/m³)
            200.0e9,            // Young's modulus (Pa)
            0.3,                  // Poisson's ratio
            0.74,             // friction coefficient
            0.85,           // restitution coefficient
            43.0,           // thermal conductivity (W/(m·K))
            490.0,         // specific heat capacity (J/(kg·K))
            250.0e6,             // yield strength (Pa)
            400.0e6,          // ultimate strength (Pa)
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
            2700.0,                 // density (kg/m³)
            69.0e9,          // Young's modulus (Pa)
            0.33,              // Poisson's ratio
            0.61,          // friction coefficient
            0.75,        // restitution coefficient
            237.0,        // thermal conductivity (W/(m·K))
            900.0,       // specific heat capacity (J/(kg·K))
            95.0e6,            // yield strength (Pa)
            110.0e6,         // ultimate strength (Pa)
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
            1100.0,            // density (kg/m³)
            0.01e9,            // Young's modulus (Pa)
            0.49,              // Poisson's ratio
            0.9,               // friction coefficient
            0.95,              // restitution coefficient
            0.16,              // thermal conductivity (W/(m·K))
            2000.0,            // specific heat capacity (J/(kg·K))
            7.0e6,             // yield strength (Pa)
            15.0e6,            // ultimate strength (Pa)
        ).expect("Failed to create rubber material")
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

    /// Helper method to calculate fatigue strength based on number of cycles.
    fn calculate_fatigue_strength(&self, cycles: u64) -> f64 {
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