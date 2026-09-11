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

/// How a material burns.
///
/// Three measured quantities and a residue fraction, and between them they determine
/// everything anybody normally wants to know about a fire: how hot, how long, and
/// whether it starts at all. These are what fire engineering actually measures, rather
/// than a "flammability" dial — a dial is a number somebody chose, and these can be
/// looked up.
///
/// # What comes out of them
///
/// * **How long a fire lasts** — fuel load over mass burning flux. Dry brush at about a
///   kilogram per square metre burns out in roughly a minute and a half, which is why a
///   grass fire passes over you and a fuel fire does not.
/// * **How fiercely** — heat release per unit area is flux times heat of combustion,
///   which is the single number a fire is characterised by.
/// * **Whether it catches** — the surface has to reach the ignition temperature, which
///   together with the material's own conductivity and heat capacity is the classic
///   thermally-thick ignition problem.
///
/// Nothing here is a game constant. They are properties of the substance, so two
/// simulations that read them cannot disagree about fire.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Combustion {
    /// Surface temperature at which flame becomes self-sustaining, in kelvin.
    ///
    /// The *piloted* ignition temperature — the one that applies when there is already a
    /// flame nearby, which is every case worth simulating.
    pub ignition_temperature: f64,
    /// Energy released per kilogram burned, in J/kg.
    ///
    /// The effective heat of combustion rather than the ideal: real fires burn
    /// incompletely, and the difference is the soot you can see.
    pub heat_of_combustion: f64,
    /// Mass lost per square metre per second by a fully developed fire, in kg/(m²·s).
    ///
    /// The mass burning flux, and the reason a thicker fuel burns *longer* rather than
    /// hotter: flux is a property of the substance, so twice the fuel is twice the time.
    pub mass_burning_flux: f64,
    /// Fraction of the original mass left as char and ash, 0 to 1.
    ///
    /// Wood leaves about a fifth of itself; a hydrocarbon liquid leaves almost nothing.
    /// It is what decides whether burnt ground reads as black or merely bare.
    pub residue_fraction: f64,
    /// Water carried per kilogram of dry fuel, as a fraction. Dry-weight basis, which is
    /// how forestry quotes it, so live foliage can exceed 1.0 and routinely does.
    ///
    /// **It has to be boiled off before anything can burn**, and that is a real energy
    /// bill: every kilogram of water needs about 2.6 MJ to reach 100 °C and evaporate,
    /// against roughly 0.5 MJ to bring a kilogram of dry fuel to ignition. Live foliage
    /// at 120% moisture spends five times as much energy drying as igniting, which is
    /// why a fire runs through dead grass and stops at a green hedge.
    pub moisture_fraction: f64,
    /// The gas-phase half of the fuel, for anything that evaporates. `None` for a
    /// solid, which is the honest answer rather than a defaulted one: wood does not
    /// have a flash point, and asking it for a burning velocity is asking the wrong
    /// question.
    pub volatile: Option<VolatileFuel>,
}

/// A fuel that puts flammable vapour above itself, and what that vapour does.
///
/// # Why a solid fuel has none of this
///
/// Everything in [`Combustion`] proper describes a *surface* burning. That is the whole
/// story for wood or grass: the flame heats the solid, the solid pyrolyses, and the
/// front advances no faster than the condensed phase can be brought to its ignition
/// temperature. Nothing is waiting in the air ahead of it.
///
/// A liquid or gelled hydrocarbon is a different problem, because there **is** something
/// waiting in the air ahead of it. Above its flash point the pool has already filled the
/// layer over itself with a flammable mixture, and the flame does not have to heat
/// anything to get there — it propagates through gas that was ready before it arrived.
/// That is a change of *regime*, not a change of rate, and it is the reason a pool fire
/// spreads at metres per second where a grass fire spreads at half of one.
///
/// These three numbers are what decide which of the two is happening, and how fast.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolatileFuel {
    /// Lowest temperature at which the fuel evaporates fast enough to make the air just
    /// above it flammable, in kelvin.
    ///
    /// **The regime switch.** Below it there is no standing vapour and a flame crosses
    /// the pool at centimetres per second, dragged along by surface-tension-gradient
    /// flow in the liquid. Above it the vapour is already there and the flame crosses at
    /// metres per second. Ambient temperature decides which, so the same fuel is two
    /// different weapons in winter and summer — except that the common hydrocarbons
    /// flash far below any weather, which is exactly why they are dangerous.
    pub flash_point: f64,
    /// Laminar burning velocity of the fuel's vapour in air, m/s, near stoichiometric
    /// at one atmosphere.
    ///
    /// How fast a premixed flame eats into the mixture *relative to the mixture*. It is
    /// a measured property of the fuel and it is much smaller than people expect —
    /// under half a metre a second for every ordinary hydrocarbon. What makes a
    /// premixed front look fast is [`VolatileFuel::expansion_ratio`], not this.
    pub laminar_burning_velocity: f64,
    /// Adiabatic flame temperature of the stoichiometric vapour-air mixture, in kelvin.
    ///
    /// Here to give the expansion ratio rather than to describe how hot the fire feels;
    /// [`Combustion::heat_release_rate`] is the quantity for that.
    pub adiabatic_flame_temperature: f64,
}

impl VolatileFuel {
    /// How much the gas expands on burning, as a ratio of volumes.
    ///
    /// Constant-pressure combustion at `ambient`, so it is a ratio of absolute
    /// temperatures and nothing else. Around eight for a hydrocarbon in air, which is
    /// the factor that turns a burning velocity nobody would describe as fast into a
    /// front nobody can stand in front of.
    pub fn expansion_ratio(&self, ambient: f64) -> f64 {
        if ambient <= 0.0 {
            return 1.0;
        }
        (self.adiabatic_flame_temperature / ambient).max(1.0)
    }
}

/// Latent heat of vaporisation of water at 100 °C, in J/kg.
const LATENT_HEAT_WATER: f64 = 2.257e6;
/// Specific heat capacity of liquid water, in J/(kg·K).
const SPECIFIC_HEAT_WATER: f64 = 4182.0;
/// Boiling point of water at one atmosphere, in kelvin.
const BOILING_POINT: f64 = 373.15;

impl Combustion {
    /// Heat release rate per unit area of a fully developed fire, in W/m².
    ///
    /// The product of the two quantities that are measured, rather than a third number
    /// kept in step with them by hand.
    pub fn heat_release_rate(&self) -> f64 {
        self.mass_burning_flux * self.heat_of_combustion
    }

    /// Speed at which a flame front crosses a **pool** of this fuel, in m/s.
    ///
    /// `None` for anything that is not a volatile liquid, and `None` below the fuel's
    /// flash point. Both are refusals rather than gaps — see below.
    ///
    /// # Which regime, and why it is not a matter of degree
    ///
    /// Flame spread over a liquid is two mechanisms with an order of magnitude between
    /// them, and the flash point is the switch.
    ///
    /// **Below it** the air above the pool is not flammable, so the flame has to bring
    /// the liquid up to temperature to make its own fuel. It does that by heating the
    /// surface just ahead of itself, which lowers the surface tension there and pulls
    /// warm liquid forward underneath the front. The front rides that flow, and it is
    /// slow — centimetres per second, and it pulses. This function returns `None`
    /// there, deliberately: the rate depends on the liquid's own convection, which is a
    /// different calculation, and returning a wrong-by-a-factor-of-fifty number would be
    /// worse than returning nothing.
    ///
    /// **Above it** the pool has already filled the layer over itself with a flammable
    /// mixture. Nothing has to be heated first; the flame is simply a premixed flame
    /// propagating through gas that was ready before it got there, and the condensed
    /// phase is not the rate-limiting step at all.
    ///
    /// # The derivation
    ///
    /// A premixed flame consumes unburnt mixture at the laminar burning velocity `Sʟ`
    /// *relative to that mixture*. Burning it raises its temperature from ambient to the
    /// flame temperature at constant pressure, so its volume grows by the ratio of the
    /// two. That expanding gas has to go somewhere, and the part of it that goes forward
    /// pushes the unburnt layer ahead of the flame — which the flame then rides. In the
    /// ground frame the front therefore advances at
    ///
    /// ```text
    ///     v = Sʟ · T_flame / T_ambient
    /// ```
    ///
    /// Two looked-up fuel properties and the ideal gas law. There is no chosen
    /// coefficient in it, and there is no length scale in it either, which is what makes
    /// it worth having: a spread rate written as a heated length over an ignition delay
    /// has a free parameter hiding in the length.
    ///
    /// # What it is, honestly
    ///
    /// **A ceiling.** The expansion is only converted into forward flow to the extent
    /// the layer is confined, and a pool in the open is confined by the ground on one
    /// side and by nothing at all on the other, so some of it vents upward instead. The
    /// true front is between `Sʟ` (everything vents) and this (nothing does), and
    /// measurements of above-flash pools sit in the upper half of that band. Taking the
    /// ceiling errs fast rather than inventing a venting fraction to sit in the middle
    /// of, and it says which way it errs.
    pub fn pool_flame_spread(&self, ambient: f64) -> Option<f64> {
        let volatile = self.volatile?;
        if ambient < volatile.flash_point {
            return None;
        }
        Some(volatile.laminar_burning_velocity * volatile.expansion_ratio(ambient))
    }

    /// How long a fuel load of `areal_density` kg/m² takes to burn out, in seconds.
    ///
    /// Only the combustible fraction is consumed; the char is left behind, which is why
    /// a wood fire dies down to embers rather than to nothing.
    pub fn burn_duration(&self, areal_density: f64) -> f64 {
        if self.mass_burning_flux <= 0.0 {
            return 0.0;
        }
        areal_density * (1.0 - self.residue_fraction) / self.mass_burning_flux
    }

    /// Energy needed to bring a kilogram of this fuel to the point of ignition, in J/kg.
    ///
    /// Two bills, and the second is usually the larger one: heating the dry fuel from
    /// ambient to its ignition temperature, and boiling off the water it carries.
    pub fn ignition_energy(&self, specific_heat: f64, ambient: f64) -> f64 {
        let dry = specific_heat * (self.ignition_temperature - ambient).max(0.0);
        let water = self.moisture_fraction
            * (SPECIFIC_HEAT_WATER * (BOILING_POINT - ambient).max(0.0) + LATENT_HEAT_WATER);
        dry + water
    }

    /// Seconds for a **thermally thin** fuel to ignite under a given heat flux.
    ///
    /// # Why leaves go first
    ///
    /// A leaf, a grass blade or a pine needle is thin enough to heat through as one
    /// lump, so its ignition delay is just the energy it needs divided by the energy
    /// arriving: **linear** in flux, and tiny, because there is almost no mass per unit
    /// area. A leaf at 0.2 kg/m² under a passing flame front ignites in well under a
    /// second.
    ///
    /// This is the dominant reason foliage catches before timber, ahead of moisture —
    /// though moisture is in here too, through [`Combustion::ignition_energy`], and it
    /// is what stops a fire in a green canopy that would run through a dead one.
    ///
    /// `areal_density` is kilograms of fuel per square metre of exposed surface;
    /// `heat_flux` is W/m² arriving at it.
    pub fn ignition_delay_thin(
        &self,
        areal_density: f64,
        specific_heat: f64,
        ambient: f64,
        heat_flux: f64,
    ) -> f64 {
        if heat_flux <= 0.0 {
            return f64::INFINITY;
        }
        areal_density * self.ignition_energy(specific_heat, ambient) / heat_flux
    }

    /// Seconds for a **thermally thick** solid to ignite under a given heat flux.
    ///
    /// # Why the trunk goes second, or not at all
    ///
    /// A trunk cannot heat through. The surface warms and conducts that heat away into
    /// cold wood behind it, so ignition depends on the material's *thermal inertia*
    /// `k·ρ·c` and the delay goes as the **inverse square** of the flux rather than the
    /// inverse. That exponent is the whole difference: halving the flux doubles a
    /// leaf's delay and quadruples a trunk's, so there is a wide band of fire intensity
    /// in which foliage burns readily and the wood it grew on never catches at all.
    ///
    /// Moisture is not in this one, and that is deliberate — the standard thermally-thick
    /// correlation is written for dry solids, and adding a term to it would be inventing
    /// a formula rather than using one.
    pub fn ignition_delay_thick(
        &self,
        conductivity: f64,
        density: f64,
        specific_heat: f64,
        ambient: f64,
        heat_flux: f64,
    ) -> f64 {
        if heat_flux <= 0.0 {
            return f64::INFINITY;
        }
        let rise = (self.ignition_temperature - ambient).max(0.0);
        let inertia = conductivity * density * specific_heat;
        (std::f64::consts::PI / 4.0) * inertia * rise * rise / (heat_flux * heat_flux)
    }
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
    /// How this material burns, if it burns at all.
    ///
    /// `None` means non-combustible - steel, concrete, glass. That is a different claim
    /// from "burns badly", and keeping it an `Option` rather than a zeroed set of numbers
    /// means a caller cannot compute a burn duration for a pane of glass and get a
    /// plausible-looking answer back.
    pub combustion: Option<Combustion>,
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
            // **Non-combustible until told otherwise.** `new` takes mechanical and
            // thermal properties, and adding four more positional arguments to a
            // ten-argument constructor would make every call site worse to read for the
            // sake of a property most materials do not have. Opt in with
            // [`Material::burning`].
            combustion: None,
        })
    }

    /// The same material, with how it burns.
    ///
    /// Builder-style because combustion is a property most materials lack and none of
    /// the existing call sites care about: `Material::steel()` should not have to say
    /// that steel does not burn.
    pub fn burning(mut self, combustion: Combustion) -> Self {
        self.combustion = Some(combustion);
        self
    }

    /// Whether this material burns at all.
    pub fn is_combustible(&self) -> bool {
        self.combustion.is_some()
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
            0.7,                // restitution coefficient (reduced from 0.95)
            0.02,               // rolling resistance coefficient (rubber deforms significantly)
            0.16,               // thermal conductivity (W/(m·K))
            2000.0,             // specific heat capacity (J/(kg·K))
            7.0e6,              // yield strength (Pa)
            15.0e6,             // ultimate strength (Pa)
        ).expect("Failed to create rubber material")
        // Burns dirty and long: a high heat of combustion with a slow flux, which is the
        // combination that makes a tyre fire last for days.
        .burning(Combustion {
            ignition_temperature: 653.0,
            heat_of_combustion: 32.0e6,
            mass_burning_flux: 0.010,
            residue_fraction: 0.15,
            moisture_fraction: 0.01,
            volatile: None,
        })
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
        // Foam plastics are the fastest common fuel there is: little mass, enormous
        // surface area, almost nothing left behind.
        .burning(Combustion {
            ignition_temperature: 583.0,
            heat_of_combustion: 26.0e6,
            mass_burning_flux: 0.025,
            residue_fraction: 0.03,
            moisture_fraction: 0.005,
            volatile: None,
        })
    }

    /// Creates a new Material instance with properties of rope/twine (natural fiber).
    ///
    /// Rope material is ideal for cable, rope, and tether constraints. It has
    /// low restitution for realistic behavior when going taut.
    ///
    /// # Returns
    ///
    /// A Material instance with typical properties of natural fiber rope:
    /// * Density: 1500 kg/m³ (dense fiber)
    /// * Young's modulus: 1 GPa (flexible but strong)
    /// * Poisson's ratio: 0.35
    /// * Friction coefficient: 0.6 (rough fiber surface)
    /// * Restitution coefficient: 0.15 (absorbs energy when taut)
    /// * Rolling resistance coefficient: 0.05
    /// * Thermal conductivity: 0.04 W/(m·K)
    /// * Specific heat capacity: 1400 J/(kg·K)
    /// * Yield strength: 30 MPa
    /// * Ultimate strength: 50 MPa
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let rope = Material::rope();
    /// assert_eq!(rope.restitution_coefficient, 0.15);
    /// ```
    pub fn rope() -> Self {
        Self::new(
            1500.0,             // density (kg/m³) - dense natural fiber
            1.0e9,              // Young's modulus (Pa) - flexible but strong
            0.35,               // Poisson's ratio
            0.6,                // friction coefficient - rough fiber surface
            0.15,               // restitution coefficient - absorbs energy when taut
            0.05,               // rolling resistance coefficient
            0.04,               // thermal conductivity (W/(m·K)) - poor conductor
            1400.0,             // specific heat capacity (J/(kg·K))
            30.0e6,             // yield strength (Pa)
            50.0e6,             // ultimate strength (Pa)
        ).expect("Failed to create rope material")
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
        // Softwood, as measured in a cone calorimeter. The residue fraction is why a
        // wood fire ends as embers and a charred stump rather than as bare ground.
        .burning(Combustion {
            ignition_temperature: 623.0,      // ~350 C piloted
            heat_of_combustion: 16.0e6,       // J/kg, effective
            mass_burning_flux: 0.011,         // kg/(m2 s)
            residue_fraction: 0.20,
            // Seasoned structural timber, not a living tree.
            moisture_fraction: 0.12,
            volatile: None,
        })
    }

    /// Dry standing vegetation: grass, scrub, thin brush.
    ///
    /// **The fuel a landscape is made of**, and the reason this belongs here rather than
    /// in whatever is drawing the fire. A grass fire is not a scaled-down wood fire —
    /// it is a fuel with a fifth of the density burning at twice the flux and leaving
    /// almost no char, which is exactly why it passes over in seconds and leaves black
    /// ground rather than embers.
    ///
    /// Mechanically it is close to worthless, and the numbers say so: it is not load
    /// bearing, and anything that asks it for a yield strength is asking the wrong
    /// question.
    pub fn dry_vegetation() -> Self {
        Self::new(
            120.0,              // density (kg/m³), loosely packed standing fuel
            0.4e9,              // Young's modulus (Pa), across the stems
            0.35,               // Poisson's ratio
            0.6,                // friction coefficient
            0.15,               // restitution coefficient
            0.05,               // rolling resistance coefficient
            0.07,               // thermal conductivity (W/(m·K)), mostly trapped air
            1800.0,             // specific heat capacity (J/(kg·K))
            0.4e6,              // yield strength (Pa)
            1.0e6,              // ultimate strength (Pa)
        ).expect("Failed to create dry vegetation material")
        .burning(Combustion {
            // Lower than wood: fine fuels have almost no thermal mass to heat through.
            ignition_temperature: 573.0,
            heat_of_combustion: 18.0e6,
            // Roughly twice wood's, which is the whole character of a grass fire.
            mass_burning_flux: 0.022,
            // Next to nothing survives. Burnt grassland is black earth, not embers.
            residue_fraction: 0.04,
            // Cured standing grass. Dead fine fuel holds very little.
            moisture_fraction: 0.08,
            volatile: None,
        })
    }

    /// Thickened hydrocarbon fuel: petrol with a gelling agent in it. Napalm.
    ///
    /// **The point of the thickener is not that it burns differently.** It burns like
    /// the petrol it is — same vapour, same flame, same burning velocity — and what the
    /// gel changes is everything mechanical: it sticks to what it lands on, it does not
    /// run off a slope or soak away, and it does not splash apart in flight. Those are
    /// the properties in the top half of this constructor, and they are why the numbers
    /// there describe something closer to a soft solid than to a liquid.
    ///
    /// The one thing the thickener does change about the *fire* is a subtraction. A free
    /// liquid below its flash point spreads flame by surface-tension-gradient flow,
    /// dragging warm fuel forward under the front; a gel cannot, because suppressing
    /// exactly that kind of flow is what a thickener is for. So gelling removes the slow
    /// regime and leaves the fast one, which is the opposite of the intuition that a
    /// thicker fuel must be a slower one.
    ///
    /// In practice it never matters, because the flash point below is two hundred and
    /// fifty kelvin under anything a battlefield sees.
    pub fn napalm() -> Self {
        Self::new(
            900.0,              // density (kg/m³), a little under water, like the petroleum it is
            5.0e3,              // Young's modulus (Pa) - a gel, not a solid
            0.499,              // Poisson's ratio - essentially incompressible
            0.9,                // friction coefficient - it sticks
            0.05,               // restitution coefficient - it splats
            0.3,                // rolling resistance coefficient
            0.13,               // thermal conductivity (W/(m·K))
            2100.0,             // specific heat capacity (J/(kg·K))
            2.0e2,              // yield strength (Pa) - the gel's yield stress, and it is tiny
            5.0e2,              // ultimate strength (Pa)
        ).expect("Failed to create napalm material")
        .burning(Combustion {
            // The *fire point* of the petrol carrier, and it is below freezing. A napalm
            // pool does not need warming up to be ignitable, which is the single fact
            // that decides how fire crosses it.
            ignition_temperature: 250.0,
            // Petrol's 43.7 MJ/kg ideal, at the ~0.9 combustion efficiency a sooty open
            // pool fire actually manages. The missing tenth is the black smoke.
            heat_of_combustion: 39.0e6,
            // Measured for open pool fires of thickened hydrocarbon, and more than twice
            // dry grass: this is a fuel, not a fine fuel.
            mass_burning_flux: 0.05,
            // Near enough nothing. What gets left on the ground after napalm is gel that
            // never caught, which is a different thing from char.
            residue_fraction: 0.02,
            moisture_fraction: 0.0,
            volatile: Some(VolatileFuel {
                // Petrol flashes at about -45 °C. Nothing outdoors is ever below this,
                // so a napalm pool is *always* in the above-flash regime — there is no
                // weather in which it creeps.
                flash_point: 228.0,
                // Petrol-air, stoichiometric, one atmosphere. Slower than a walk.
                laminar_burning_velocity: 0.40,
                // Stoichiometric petrol-air. Used for the expansion, not for the heat.
                adiabatic_flame_temperature: 2270.0,
            }),
        })
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
#[cfg(test)]
mod combustion_tests {
    use super::*;

    /// The materials that burn say so, and the ones that cannot stay silent.
    #[test]
    fn only_combustible_materials_carry_combustion() {
        for m in [Material::wood(), Material::rubber(), Material::polyurethane(), Material::dry_vegetation()] {
            assert!(m.is_combustible(), "a combustible material reported no combustion");
        }
        for m in [Material::steel(), Material::concrete(), Material::glass(), Material::ice()] {
            assert!(
                !m.is_combustible(),
                "a non-combustible material was given combustion properties, so something \
                 can now compute a burn duration for it and get a plausible answer",
            );
        }
    }

    /// Duration falls out of fuel load over flux, and the char is not consumed.
    #[test]
    fn burn_duration_follows_the_fuel_load() {
        let brush = Material::dry_vegetation().combustion.unwrap();

        // A kilogram of standing fuel per square metre — an ordinary grass load.
        let quick = brush.burn_duration(1.0);
        assert!(
            (40.0..60.0).contains(&quick),
            "a 1 kg/m2 grass load burned for {quick:.0} s, which is not a grass fire",
        );

        // Twice the fuel is twice the time: flux is a property of the substance.
        let double = brush.burn_duration(2.0);
        assert!((double - quick * 2.0).abs() < 1e-9, "duration is not linear in fuel load");

        // Char is left behind rather than burned, so wood outlasts grass per kilogram
        // by more than the flux ratio alone.
        let wood = Material::wood().combustion.unwrap();
        assert!(
            wood.burn_duration(1.0) > quick,
            "wood burned out faster than grass at the same load",
        );
    }

    /// Grass is the fiercer fire per square metre and the shorter one — which is the
    /// entire difference between a fire you walk through and one you do not.
    #[test]
    fn fine_fuels_burn_hotter_and_die_sooner() {
        let grass = Material::dry_vegetation().combustion.unwrap();
        let wood = Material::wood().combustion.unwrap();

        assert!(
            grass.heat_release_rate() > wood.heat_release_rate(),
            "grass releases {:.0} W/m2 against wood's {:.0} - a grass fire is supposed to \
             be the fiercer one per unit area",
            grass.heat_release_rate(),
            wood.heat_release_rate(),
        );
        assert!(
            grass.burn_duration(1.0) < wood.burn_duration(1.0),
            "grass outlasted wood at the same fuel load",
        );
    }

    /// A non-combustible material cannot be asked how long it burns by accident.
    #[test]
    fn glass_has_no_burn_duration_to_ask_for() {
        assert!(Material::glass().combustion.is_none());
    }

    /// Standard sea-level air, so the ambient in these tests is the one the atmosphere
    /// module already agrees on rather than a second opinion about room temperature.
    const STANDARD_AIR: f64 = crate::atmosphere::SEA_LEVEL_TEMPERATURE;

    /// The regime question, which is the one that decides the answer.
    ///
    /// A solid fuel has no vapour standing over it and no flash point to be above, so
    /// the pool-spread calculation does not apply to it and says so rather than
    /// returning a number that happens to be finite.
    #[test]
    fn only_pooled_fuels_have_a_pool_spread_rate() {
        for solid in [Material::wood(), Material::dry_vegetation()] {
            let c = solid.combustion.unwrap();
            assert!(
                c.pool_flame_spread(STANDARD_AIR).is_none(),
                "a solid fuel was given a pool flame spread rate, which means the \
                 regime distinction has been lost",
            );
        }
        assert!(
            Material::napalm()
                .combustion
                .unwrap()
                .pool_flame_spread(STANDARD_AIR)
                .is_some(),
            "napalm is a pooled volatile fuel and could not be asked how fast a flame \
             crosses it",
        );
    }

    /// Napalm is never in the slow regime, and that is a property of petrol rather than
    /// a decision anybody made.
    ///
    /// Worth a test because it is the whole argument: if the flash point were anywhere
    /// near ambient the spread rate would be a function of the weather, and the
    /// simulation reading it would need to care what season it is.
    #[test]
    fn napalm_is_above_its_flash_point_in_any_weather() {
        let gel = Material::napalm().combustion.unwrap();
        // Antarctic winter to a hot desert afternoon.
        for ambient in [223.0_f64, 250.0, 273.15, 288.15, 313.0] {
            let spread = gel.pool_flame_spread(ambient);
            if ambient < 228.0 {
                assert!(spread.is_none(), "petrol flashed below its own flash point");
                continue;
            }
            let v = spread.expect("napalm failed to spread above its flash point");
            assert!(
                (1.0..10.0).contains(&v),
                "a flame crosses a napalm pool at {v:.2} m/s at {ambient:.0} K, which is \
                 not a premixed front over a hydrocarbon — those are metres per second",
            );
        }
    }

    /// The expansion is the whole reason the front is fast, and it is worth saying so
    /// in a test: the burning velocity on its own would be a stroll.
    #[test]
    fn the_front_outruns_the_burning_velocity_by_the_expansion() {
        let gel = Material::napalm().combustion.unwrap();
        let volatile = gel.volatile.unwrap();
        let v = gel.pool_flame_spread(STANDARD_AIR).unwrap();

        assert!(
            volatile.laminar_burning_velocity < 0.5,
            "a laminar burning velocity of {:.2} m/s is not a hydrocarbon - every \
             ordinary fuel is under half a metre a second",
            volatile.laminar_burning_velocity,
        );
        let ratio = v / volatile.laminar_burning_velocity;
        assert!(
            (6.0..10.0).contains(&ratio),
            "the front ran {ratio:.1} times the burning velocity; constant-pressure \
             combustion in air expands by about eight",
        );
        assert!(
            (ratio - volatile.expansion_ratio(STANDARD_AIR)).abs() < 1e-12,
            "the spread rate is not the burning velocity times the expansion ratio, so \
             something other than the derivation is in it",
        );
    }

    /// Napalm is the fiercer fire, and a kilogram of it is therefore the *shorter* one.
    ///
    /// # The half of this that is counter-intuitive
    ///
    /// The first version of this test asserted that a kilo of gel outlasts a kilo of
    /// grass, and it failed, correctly. Mass burning flux is a *rate*: gel burns at more
    /// than twice grass's flux, so the same load is gone sooner. Napalm denies ground for
    /// longer than a grass fire does **because a shell puts far more fuel per square
    /// metre down**, not because the substance is slower. Those are different claims and
    /// only the second one is a property of the material.
    #[test]
    fn gel_is_the_fiercer_fire_and_the_shorter_one_per_kilogram() {
        let gel = Material::napalm().combustion.unwrap();
        let grass = Material::dry_vegetation().combustion.unwrap();

        assert!(
            gel.heat_release_rate() > 2.0 * grass.heat_release_rate(),
            "gel releases {:.0} W/m2 against grass at {:.0} - napalm is supposed to be \
             in a different class, not a slightly worse grass fire",
            gel.heat_release_rate(),
            grass.heat_release_rate(),
        );
        assert!(
            gel.burn_duration(1.0) < grass.burn_duration(1.0),
            "a kilo per square metre of gel outlasted the same load of grass, which \
             contradicts it burning at the higher flux",
        );
    }

    /// And it is much faster across the ground than a fire in a solid fuel.
    ///
    /// Not a coincidence and not a tuning: the gel front is a premixed flame in gas that
    /// was already flammable, and the grass front has to heat its fuel up first. The
    /// mechanisms are different, so the rates are not close.
    #[test]
    fn a_gel_front_outruns_a_fire_in_standing_fuel() {
        let gel = Material::napalm().combustion.unwrap();
        let v = gel.pool_flame_spread(STANDARD_AIR).unwrap();
        // A grass fire on the flat is about half a metre a second.
        assert!(
            v > 5.0 * 0.5,
            "flame crosses gel at {v:.2} m/s, which is within a factor of five of a \
             grass fire - the two regimes should not be that close",
        );
    }
}

#[cfg(test)]
mod ignition_tests {
    use super::*;

    /// Ambient, in kelvin. A cool day.
    const AMBIENT: f64 = 290.0;
    /// Heat arriving at a fuel from a passing flame front, in W/m². A moderate surface
    /// fire radiates this at a metre.
    const FRONT_FLUX: f64 = 25_000.0;

    /// **Leaves before trunk**, and by a very wide margin.
    ///
    /// The reason is thermal thickness rather than water: a leaf heats through as one
    /// lump and a trunk conducts heat away into itself.
    #[test]
    fn foliage_ignites_long_before_the_wood_it_grew_on() {
        let leaf_material = Material::dry_vegetation();
        let leaf = leaf_material.combustion.unwrap();
        let trunk_material = Material::wood();
        let trunk = trunk_material.combustion.unwrap();

        // **One leaf layer, not a litter bed.** A single leaf or needle runs about
        // 0.05 kg/m2 of its own surface; a few hundred grams per square metre is
        // accumulated litter several leaves deep, which is a slower fuel and a different
        // question. Getting that wrong is how a correct formula produces a wrong answer.
        let leaves = leaf.ignition_delay_thin(
            0.05,
            leaf_material.specific_heat_capacity,
            AMBIENT,
            FRONT_FLUX,
        );
        let bole = trunk.ignition_delay_thick(
            trunk_material.thermal_conductivity,
            trunk_material.density,
            trunk_material.specific_heat_capacity,
            AMBIENT,
            FRONT_FLUX,
        );

        assert!(
            leaves < 2.0,
            "foliage took {leaves:.1} s to catch under a flame front, which is not how \
             fine fuel behaves",
        );
        // An order of magnitude, not a fixed ratio: the exact figure moves with species,
        // flux and moisture, and pinning it would pin this test to one arbitrary trunk.
        // What matters is that the two regimes are not comparable - 1.4 s against 25.
        assert!(
            bole > leaves * 10.0,
            "a trunk ignited in {bole:.1} s against foliage's {leaves:.2} - the thermally \
             thick and thin regimes are supposed to be worlds apart",
        );
    }

    /// And the gap *widens* as the fire weakens, because the two regimes scale
    /// differently in flux: thin is linear, thick is inverse-square.
    ///
    /// That exponent is why a weak fire strips a canopy and leaves the timber standing.
    #[test]
    fn a_weaker_fire_spares_the_trunk_more_than_the_leaves() {
        let leaf_material = Material::dry_vegetation();
        let leaf = leaf_material.combustion.unwrap();
        let trunk_material = Material::wood();
        let trunk = trunk_material.combustion.unwrap();

        let thin_at = |q| {
            leaf.ignition_delay_thin(0.05, leaf_material.specific_heat_capacity, AMBIENT, q)
        };
        let thick_at = |q| {
            trunk.ignition_delay_thick(
                trunk_material.thermal_conductivity,
                trunk_material.density,
                trunk_material.specific_heat_capacity,
                AMBIENT,
                q,
            )
        };

        // Halve the flux: the leaf takes twice as long, the trunk four times.
        let leaf_ratio = thin_at(FRONT_FLUX / 2.0) / thin_at(FRONT_FLUX);
        let trunk_ratio = thick_at(FRONT_FLUX / 2.0) / thick_at(FRONT_FLUX);

        assert!((leaf_ratio - 2.0).abs() < 1e-9, "thin ignition is not linear in flux");
        assert!((trunk_ratio - 4.0).abs() < 1e-9, "thick ignition is not inverse-square");
    }

    /// Water is a real bill, and it is the larger one for anything living.
    #[test]
    fn moisture_is_most_of_what_a_green_fuel_costs_to_light() {
        let veg = Material::dry_vegetation();
        let cured = veg.combustion.unwrap();

        let mut green = cured;
        green.moisture_fraction = 1.20; // live foliage, dry-weight basis

        let dry_bill = cured.ignition_energy(veg.specific_heat_capacity, AMBIENT);
        let green_bill = green.ignition_energy(veg.specific_heat_capacity, AMBIENT);

        assert!(
            green_bill > dry_bill * 3.0,
            "green foliage cost {green_bill:.0} J/kg to light against cured fuel's \
             {dry_bill:.0} - boiling a kilogram of water is 2.6 MJ and should dominate",
        );
    }
}
