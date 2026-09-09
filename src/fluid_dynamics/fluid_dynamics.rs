// src/fluid_dynamics.rs

use crate::utils::PhysicsError;
use super::validation::{validate_finite, validate_positive};

/// Represents a fluid with physical properties for analytical calculations.
///
/// This struct is used for calculating Reynolds numbers, drag forces,
/// buoyant forces, and pressure drops.
pub struct Fluid {
    pub density: f64,
    pub viscosity: f64,
}

impl Fluid {

    /// Creates a new `Fluid` instance with the given density and viscosity.
    /// # Arguments
    /// * `density` - The density of the fluid in kg/m³.
    /// * `viscosity` - The dynamic viscosity of the fluid in Pa·s.
    ///
    /// # Return
    /// Returns a `Result` containing the new `Fluid` instance if successful,
    /// or a `PhysicsError` if the input parameters are invalid.
    ///
    /// # Errors
    /// Returns an error if:
    /// * The density is less than or equal to zero.
    /// * The viscosity is less than or equal to zero.
    ///
    /// # Examples
    /// ```
    /// use rs_physics::fluid_dynamics::Fluid;
    ///
    /// let water = Fluid::new(1000.0, 0.001).unwrap();
    /// ```
    pub fn new(density: f64, viscosity: f64) -> Result<Self, PhysicsError> {
        validate_positive(density, "density")?;
        validate_positive(viscosity, "viscosity")?;
        Ok(Self { density, viscosity })
    }

    /// Creates a `Fluid` representing water at 20°C.
    ///
    /// Properties:
    /// - Density: 998 kg/m³
    /// - Dynamic viscosity: 0.001 Pa·s (1 mPa·s)
    ///
    /// # Examples
    /// ```
    /// use rs_physics::fluid_dynamics::Fluid;
    ///
    /// let water = Fluid::water();
    /// assert!((water.density - 998.0).abs() < 1.0);
    /// ```
    pub fn water() -> Self {
        Self {
            density: 998.0,
            viscosity: 0.001,
        }
    }

    /// The bulk properties of a given state of the air.
    ///
    /// # Why there is no `Fluid::air()`
    ///
    /// There was, and it was a defect. It returned a frozen density of 1.225 kg/m³ and a
    /// frozen viscosity of 1.81×10⁻⁵ Pa·s, with a doc comment saying "20 °C" — which was
    /// wrong twice over, since 1.225 is the **15 °C** ICAO figure and 1.81e-5 is the
    /// 20 °C one. Worse, the crate simultaneously carried
    /// [`crate::atmosphere::Air`], a real thermodynamic state that knew the temperature,
    /// the humidity and the pressure, and the two could not be reconciled: a caller
    /// simulating a winter map got winter air for sound and summer air for drag, in the
    /// same frame, and no type, test or assertion objected. Winter air at 270 K is **9%
    /// denser** than summer air at 293 K, and density is linear in every drag force this
    /// module computes.
    ///
    /// So the nullary constructor is gone. There is no way to obtain a `Fluid` of air
    /// without saying which air, which makes the divergence unrepresentable rather than
    /// documented. For the old behaviour, ask for the state it meant:
    /// `Fluid::from_air(&Air::sea_level())`, whose density comes out at 1.225 because it
    /// is *computed* from the ICAO reference rather than copied from a table.
    ///
    /// # Arguments
    ///
    /// * `air` — the state of the air. See [`crate::atmosphere::Air`].
    ///
    /// # Examples
    /// ```
    /// use rs_physics::atmosphere::Air;
    /// use rs_physics::fluid_dynamics::Fluid;
    ///
    /// // The ICAO reference reproduces the constant this used to hardcode.
    /// let isa = Fluid::from_air(&Air::sea_level());
    /// assert!((isa.density - 1.225).abs() < 0.001);
    ///
    /// // And a cold day is genuinely a different fluid.
    /// let winter = Fluid::from_air(&Air::winter());
    /// assert!(winter.density > isa.density);
    /// ```
    #[inline]
    pub fn from_air(air: &crate::atmosphere::Air) -> Self {
        // Infallible: `Air`'s constructor guarantees a positive temperature and
        // pressure, so density and viscosity are both finite and strictly positive —
        // which is exactly what `Fluid::new` would have checked for.
        Self { density: air.density(), viscosity: air.dynamic_viscosity() }
    }

    /// Creates a `Fluid` representing motor oil (SAE 30) at 40°C.
    ///
    /// Properties:
    /// - Density: 876 kg/m³
    /// - Dynamic viscosity: 0.1 Pa·s (100 mPa·s)
    ///
    /// # Examples
    /// ```
    /// use rs_physics::fluid_dynamics::Fluid;
    ///
    /// let oil = Fluid::oil();
    /// assert!(oil.viscosity > Fluid::water().viscosity);
    /// ```
    pub fn oil() -> Self {
        Self {
            density: 876.0,
            viscosity: 0.1,
        }
    }

    /// Creates a `Fluid` representing honey at 20°C.
    ///
    /// Properties:
    /// - Density: 1420 kg/m³
    /// - Dynamic viscosity: 10.0 Pa·s (very viscous)
    ///
    /// # Examples
    /// ```
    /// use rs_physics::fluid_dynamics::Fluid;
    ///
    /// let honey = Fluid::honey();
    /// // Honey is much more viscous than water
    /// assert!(honey.viscosity > 1000.0 * Fluid::water().viscosity);
    /// ```
    pub fn honey() -> Self {
        Self {
            density: 1420.0,
            viscosity: 10.0,
        }
    }

    /// Creates a `Fluid` representing seawater at 20°C.
    ///
    /// Properties:
    /// - Density: 1025 kg/m³
    /// - Dynamic viscosity: 0.00108 Pa·s
    ///
    /// # Examples
    /// ```
    /// use rs_physics::fluid_dynamics::Fluid;
    ///
    /// let seawater = Fluid::seawater();
    /// // Seawater is slightly denser than freshwater
    /// assert!(seawater.density > Fluid::water().density);
    /// ```
    pub fn seawater() -> Self {
        Self {
            density: 1025.0,
            viscosity: 0.00108,
        }
    }

    /// Creates a `Fluid` representing whole human blood at 37 °C.
    ///
    /// Properties:
    /// - Density: 1060 kg/m³
    /// - Dynamic viscosity: 4.0 mPa·s — **at one stated shear rate, see below**
    ///
    /// # What this constructor cannot honestly claim
    ///
    /// **Blood is not Newtonian, and a single `viscosity` field cannot say so.** At
    /// rest, red cells stack into rouleaux and the suspension is nearly a solid; as
    /// it is sheared they break apart, then deform and align with the flow. Apparent
    /// viscosity therefore *falls* with shear rate — tens of mPa·s below 1 s⁻¹, an
    /// asymptote near 3.5 mPa·s above a few hundred. Blood also has a small but
    /// genuine **yield stress**, a few mPa, below which it does not flow at all.
    /// That is why a pool on a level floor holds an edge instead of creeping outward
    /// forever, and no `viscosity` however large reproduces it.
    ///
    /// So the number here is not "the viscosity of blood", because there is no such
    /// number. It is the apparent viscosity at **one shear rate, and the shear rate
    /// is named**: [`BLOOD_REFERENCE_SHEAR_RATE`], 300 s⁻¹, evaluated through
    /// [`blood_apparent_viscosity`]. A test asserts the two agree, so this is a
    /// derived figure with a source rather than a plausible one. 300 s⁻¹ is the
    /// middle of the range a film of spilt blood on ground actually runs at — see
    /// [`FilmFlow::shear_rate`], and the test that pins the whole plausible range of
    /// film depths to within a few percent of this value.
    ///
    /// Use this `Fluid` where the shear rate is high: drag, Reynolds numbers,
    /// buoyancy, and the thin-film flow in [`crate::fluid_dynamics::FilmFlow`]. Do
    /// **not** use it anywhere near rest — sedimentation, clotting, a vessel at
    /// diastole, a drop deciding whether to move at all — and reach for
    /// [`blood_apparent_viscosity`] instead, which knows what shear rate you meant.
    ///
    /// # Sources
    ///
    /// Density: Kenner, *The measurement of blood density and its meaning*, Basic
    /// Research in Cardiology 84 (1989), 111–124 — 1052–1063 kg/m³ for whole blood.
    /// Rheology: Merrill et al., *Rheology of human blood near and at zero flow*,
    /// Biophysical Journal 3 (1963), 199–213, and Cokelet et al. (1963), from whom
    /// the Casson constants in [`BLOOD_CASSON_VISCOSITY`] and
    /// [`BLOOD_YIELD_STRESS`] come.
    ///
    /// # Examples
    /// ```
    /// use rs_physics::fluid_dynamics::Fluid;
    ///
    /// let blood = Fluid::blood();
    /// // Denser than water and about four times as viscous — at high shear.
    /// assert!(blood.density > Fluid::water().density);
    /// assert!(blood.viscosity > 3.0 * Fluid::water().viscosity);
    /// ```
    pub fn blood() -> Self {
        Self {
            density: BLOOD_DENSITY,
            // Not a table value: the Casson fit evaluated at the shear rate this
            // approximation is declared valid at. `blood_apparent_viscosity` cannot
            // be called in a const context, so the number is written here and a test
            // (`newtonian_blood_is_the_casson_fit_at_its_stated_shear_rate`) is what
            // keeps the two from drifting.
            viscosity: 0.004,
        }
    }

    /// Creates a `Fluid` representing glycerin at 20°C.
    ///
    /// Properties:
    /// - Density: 1261 kg/m³
    /// - Dynamic viscosity: 1.5 Pa·s
    ///
    /// # Examples
    /// ```
    /// use rs_physics::fluid_dynamics::Fluid;
    ///
    /// let glycerin = Fluid::glycerin();
    /// assert!(glycerin.viscosity > Fluid::oil().viscosity);
    /// ```
    pub fn glycerin() -> Self {
        Self {
            density: 1261.0,
            viscosity: 1.5,
        }
    }

    /// Calculates the kinematic viscosity (ν = μ/ρ).
    ///
    /// # Returns
    /// The kinematic viscosity in m²/s.
    ///
    /// # Examples
    /// ```
    /// use rs_physics::fluid_dynamics::Fluid;
    ///
    /// let water = Fluid::water();
    /// let kinematic = water.kinematic_viscosity();
    /// // Approximately 1e-6 m²/s for water
    /// assert!((kinematic - 1e-6).abs() < 1e-7);
    /// ```
    #[inline]
    pub fn kinematic_viscosity(&self) -> f64 {
        self.viscosity / self.density
    }
}

/// Density of whole human blood at 37 °C, kg/m³.
///
/// Haematocrit around 45%. The measured range is 1052–1063 (Kenner 1989); 1060 is
/// the value in ordinary use and is what [`Fluid::blood`] carries.
pub const BLOOD_DENSITY: f64 = 1060.0;

/// Casson viscosity μ_c of whole blood, Pa·s — the **high-shear asymptote**.
///
/// This is the number blood's apparent viscosity approaches from above as shear rate
/// rises without bound, not a viscosity you should use directly at any real shear
/// rate. See [`blood_apparent_viscosity`]. Merrill et al. (1963), Hct ≈ 45%.
pub const BLOOD_CASSON_VISCOSITY: f64 = 0.0035;

/// Casson yield stress τ_y of whole blood, Pa.
///
/// **The property a Newtonian `Fluid` cannot express at all.** Below this shear
/// stress blood does not flow; rouleaux hold it together as a weak solid. Reported
/// between about 0.004 and 0.01 Pa at Hct 45% and strongly dependent on haematocrit
/// (Merrill et al. 1963); 0.005 sits at the low end of that band, which is the
/// conservative direction for a solver — it arrests less liquid, not more.
///
/// It is small: on a one-in-ten slope it holds only a film about 5 µm deep
/// (τ_y / ρ g sinθ). Yield stress is therefore *not* what makes a pool of blood hold
/// a millimetres-deep edge on level ground — surface tension is, see
/// [`puddle_depth`]. Both are real and they act at different scales.
pub const BLOOD_YIELD_STRESS: f64 = 0.005;

/// Surface tension of whole blood against air at 37 °C, N/m.
///
/// About 56 mN/m, appreciably below water's 72: plasma proteins are surface-active
/// and crowd the interface. Hrnčíř & Rosina, *Surface tension of blood*, Physiological
/// Research 46 (1997), 319–321.
pub const BLOOD_SURFACE_TENSION: f64 = 0.056;

/// The shear rate at which [`Fluid::blood`]'s single Newtonian viscosity is quoted, s⁻¹.
///
/// A Newtonian constructor for a shear-thinning fluid is only honest if it says which
/// shear rate it means. This is that declaration, and it is not arbitrary: a film of
/// spilt blood between a fifth of a millimetre and five millimetres deep, on grades
/// from a gentle slope to a steep one, runs at wall shear rates of roughly one to
/// eight hundred per second ([`FilmFlow::shear_rate`]). 300 s⁻¹ is the middle of that.
pub const BLOOD_REFERENCE_SHEAR_RATE: f64 = 300.0;

/// Apparent viscosity of whole blood at a given shear rate, Pa·s — the Casson model.
///
/// ```text
///   √τ = √τ_y + √(μ_c γ̇)          so      μ_app(γ̇) = τ/γ̇ = (√(τ_y/γ̇) + √μ_c)²
/// ```
///
/// This is what [`Fluid::blood`] is an approximation *of*. Blood thins under shear
/// because red cells are deformable and, at low shear, aggregated: the yield term
/// dominates below about 10 s⁻¹ and has all but vanished above a few hundred, where
/// the answer settles toward [`BLOOD_CASSON_VISCOSITY`].
///
/// Reach for this whenever the shear rate is low or unknown. Reach for
/// [`Fluid::blood`] when it is high and a few percent is not worth a square root.
///
/// # Arguments
///
/// * `shear_rate` — γ̇ in s⁻¹, strictly positive.
///
/// # Errors
///
/// Returns [`PhysicsError::CalculationError`] if the shear rate is not finite and
/// strictly positive. Zero is genuinely undefined here rather than merely awkward:
/// a fluid with a yield stress has *infinite* apparent viscosity at zero shear, and
/// returning `f64::INFINITY` would hand a value that poisons every arithmetic
/// expression downstream to a caller who only wanted to know if it moves. Ask
/// [`FilmFlow::arrest_thickness`] that question instead.
///
/// # Examples
/// ```
/// use rs_physics::fluid_dynamics::{blood_apparent_viscosity, Fluid};
///
/// // Blood thins as it is sheared harder. This is the whole point.
/// let slow = blood_apparent_viscosity(1.0).unwrap();
/// let fast = blood_apparent_viscosity(1000.0).unwrap();
/// assert!(slow > 4.0 * fast);
///
/// // And the Newtonian constructor is this curve at one point on it.
/// let quoted = blood_apparent_viscosity(300.0).unwrap();
/// assert!((quoted - Fluid::blood().viscosity).abs() < 1e-4);
/// ```
pub fn blood_apparent_viscosity(shear_rate: f64) -> Result<f64, PhysicsError> {
    if !(shear_rate > 0.0) || !shear_rate.is_finite() {
        return Err(PhysicsError::CalculationError(format!(
            "shear rate must be finite and strictly positive, got {}",
            shear_rate
        )));
    }
    let root = (BLOOD_YIELD_STRESS / shear_rate).sqrt() + BLOOD_CASSON_VISCOSITY.sqrt();
    Ok(root * root)
}

/// Depth of a static puddle of liquid standing on level ground, in metres.
///
/// ```text
///   h = 2 √(γ / ρg) · sin(θ_c / 2)
/// ```
///
/// **This is why a spill has an edge.** Gravity alone would flatten a puddle to a
/// molecular film; what stops it is that thinning further costs surface energy. The
/// balance is set by the capillary length √(γ/ρg) — 2.3 mm for blood, 2.7 for water —
/// and by how well the liquid wets what it is sitting on, through the contact angle.
/// A perfectly wetting liquid (θ_c = 0) forms no puddle at all and spreads without
/// limit; a non-wetting one (θ_c = π) beads to twice the capillary length.
///
/// It is worth having because it is the derived form of a number that otherwise gets
/// tuned. A flow solver needs a depth at which liquid stops levelling out, or a pool
/// spreads into an invisible sheet and the pass never converges; that depth is this
/// one, and it comes from the fluid rather than from how it looked.
///
/// The result is a *depth*, not a criterion for motion on a slope. For that see
/// [`FilmFlow::arrest_thickness`], which is the yield-stress condition and a much
/// smaller number.
///
/// # Arguments
///
/// * `surface_tension` — γ against air, N/m. Strictly positive.
/// * `density` — kg/m³. Strictly positive.
/// * `gravity` — m/s². Strictly positive.
/// * `contact_angle` — θ_c in **radians**, in `[0, π]`. Blood on dry soil is around
///   1.4 rad (80°); the crate carries no table of these because the substrate
///   decides it, not the liquid.
///
/// # Errors
///
/// Returns [`PhysicsError::CalculationError`] if any argument is not finite, if any
/// of the first three is not strictly positive, or if the contact angle lies outside
/// `[0, π]` — where the formula silently returns a negative depth.
///
/// # Examples
/// ```
/// use rs_physics::fluid_dynamics::{puddle_depth, BLOOD_DENSITY, BLOOD_SURFACE_TENSION};
///
/// // Blood on soil it wets poorly stands about three millimetres deep.
/// let h = puddle_depth(BLOOD_SURFACE_TENSION, BLOOD_DENSITY, 9.81, 80f64.to_radians()).unwrap();
/// assert!(h > 0.002 && h < 0.004);
///
/// // A liquid that wets perfectly makes no puddle: it spreads without limit.
/// assert_eq!(puddle_depth(BLOOD_SURFACE_TENSION, BLOOD_DENSITY, 9.81, 0.0).unwrap(), 0.0);
/// ```
pub fn puddle_depth(
    surface_tension: f64,
    density: f64,
    gravity: f64,
    contact_angle: f64,
) -> Result<f64, PhysicsError> {
    validate_finite(surface_tension, "surface_tension")?;
    validate_finite(density, "density")?;
    validate_finite(gravity, "gravity")?;
    validate_finite(contact_angle, "contact_angle")?;
    validate_positive(surface_tension, "surface_tension")?;
    validate_positive(density, "density")?;
    validate_positive(gravity, "gravity")?;
    if !(0.0..=std::f64::consts::PI).contains(&contact_angle) {
        return Err(PhysicsError::CalculationError(format!(
            "contact angle must be in [0, pi] radians, got {}",
            contact_angle
        )));
    }
    let capillary_length = (surface_tension / (density * gravity)).sqrt();
    Ok(2.0 * capillary_length * (contact_angle * 0.5).sin())
}

/// Calculates the Reynolds number for a fluid flow.
/// # Arguments
/// * `fluid` - A reference to the `Fluid` instance.
/// * `velocity` - The velocity of the fluid flow in m/s.
/// * `characteristic_length` - The characteristic length of the flow geometry in m.
///
/// # Return
/// Returns a `Result` containing the calculated Reynolds number (dimensionless) if successful,
/// or a `PhysicsError` if the input parameters are invalid.
///
/// # Errors
/// Returns an error if:
/// * The velocity is less than or equal to zero.
/// * The characteristic length is less than or equal to zero.
///
/// # Examples
/// ```
/// use rs_physics::fluid_dynamics::{Fluid, calculate_reynolds_number};
///
/// let water = Fluid::new(1000.0, 0.001).unwrap();
/// let re = calculate_reynolds_number(&water, 1.0, 0.1).unwrap();
/// ```
pub fn calculate_reynolds_number(fluid: &Fluid, velocity: f64, characteristic_length: f64) -> Result<f64, PhysicsError> {
    if velocity <= 0.0 {
        return Err(PhysicsError::InvalidVelocity);
    }
    if characteristic_length <= 0.0 {
        return Err(PhysicsError::InvalidArea);
    }
    Ok((fluid.density * velocity * characteristic_length) / fluid.viscosity)
}

/// Calculates the drag force on an object in a fluid.
/// # Arguments
/// * `fluid` - A reference to the `Fluid` instance.
/// * `velocity` - The relative velocity between the object and the fluid in m/s.
/// * `area` - The reference area of the object in m².
/// * `drag_coefficient` - The drag coefficient of the object (dimensionless).
///
/// # Return
/// Returns a `Result` containing the calculated drag force in Newtons (N) if successful,
/// or a `PhysicsError` if the input parameters are invalid.
///
/// # Errors
/// Returns an error if:
/// * The velocity is less than or equal to zero.
/// * The area is less than or equal to zero.
/// * The drag coefficient is less than or equal to zero.
///
/// # Examples
/// ```
/// use rs_physics::fluid_dynamics::{Fluid, calculate_drag_force};
///
/// let air = Fluid::new(1.225, 1.81e-5).unwrap();
/// let drag = calculate_drag_force(&air, 10.0, 1.0, 0.5).unwrap();
/// ```
pub fn calculate_drag_force(fluid: &Fluid, velocity: f64, area: f64, drag_coefficient: f64) -> Result<f64, PhysicsError> {
    if velocity <= 0.0 {
        return Err(PhysicsError::InvalidVelocity);
    }
    if area <= 0.0 {
        return Err(PhysicsError::InvalidArea);
    }
    if drag_coefficient <= 0.0 {
        return Err(PhysicsError::InvalidCoefficient);
    }
    Ok(0.5 * fluid.density * velocity * velocity * area * drag_coefficient)
}

/// Calculates the buoyant force on an object submerged in a fluid.
/// # Arguments
/// * `fluid` - A reference to the `Fluid` instance.
/// * `displaced_volume` - The volume of fluid displaced by the object in m³.
/// * `gravity` - The acceleration due to gravity in m/s².
///
/// # Return
/// Returns a `Result` containing the calculated buoyant force in Newtons (N) if successful,
/// or a `PhysicsError` if the input parameters are invalid.
///
/// # Errors
/// Returns an error if:
/// * The displaced volume is less than or equal to zero.
/// * The gravity is less than or equal to zero.
///
/// # Examples
/// ```
/// use rs_physics::fluid_dynamics::{Fluid, calculate_buoyant_force};
///
/// let water = Fluid::new(1000.0, 0.001).unwrap();
/// let buoyant_force = calculate_buoyant_force(&water, 0.1, 9.81).unwrap();
/// ```
pub fn calculate_buoyant_force(fluid: &Fluid, displaced_volume: f64, gravity: f64) -> Result<f64, PhysicsError> {
    if displaced_volume <= 0.0 {
        return Err(PhysicsError::InvalidArea);
    }
    if gravity <= 0.0 {
        return Err(PhysicsError::CalculationError("Gravity must be positive".to_string()));
    }
    Ok(fluid.density * displaced_volume * gravity)
}

/// Calculates the pressure drop in a pipe due to fluid flow.
/// # Arguments
/// * `fluid` - A reference to the `Fluid` instance.
/// * `pipe_length` - The length of the pipe in meters.
/// * `pipe_diameter` - The diameter of the pipe in meters.
/// * `velocity` - The average velocity of the fluid in the pipe in m/s.
/// * `friction_factor` - The Darcy friction factor (dimensionless).
///
/// # Return
/// Returns a `Result` containing the calculated pressure drop in Pascals (Pa) if successful,
/// or a `PhysicsError` if the input parameters are invalid.
///
/// # Errors
/// Returns an error if:
/// * The pipe length or diameter is less than or equal to zero.
/// * The velocity is less than or equal to zero.
/// * The friction factor is less than or equal to zero.
///
/// # Examples
/// ```
/// use rs_physics::fluid_dynamics::{Fluid, calculate_pressure_drop};
///
/// let water = Fluid::new(1000.0, 0.001).unwrap();
/// let pressure_drop = calculate_pressure_drop(&water, 10.0, 0.05, 2.0, 0.02).unwrap();
/// ```
pub fn calculate_pressure_drop(fluid: &Fluid, pipe_length: f64, pipe_diameter: f64, velocity: f64, friction_factor: f64) -> Result<f64, PhysicsError> {
    if pipe_length <= 0.0 || pipe_diameter <= 0.0 {
        return Err(PhysicsError::InvalidArea);
    }
    if velocity <= 0.0 {
        return Err(PhysicsError::InvalidVelocity);
    }
    if friction_factor <= 0.0 {
        return Err(PhysicsError::InvalidCoefficient);
    }
    Ok(friction_factor * (pipe_length / pipe_diameter) * 0.5 * fluid.density * velocity * velocity)
}
/// Upward acceleration of a parcel of fluid hotter than its surroundings, by the
/// Boussinesq approximation.
///
/// ```text
///   a = g * (T - T_ambient) / T_ambient
/// ```
///
/// This is why flames rise, why smoke climbs, and why a plume narrows as it goes: hot
/// gas is less dense than the air around it, and the buoyant acceleration is
/// proportional to how much hotter it is. Deriving a flame's rise from its
/// temperature rather than assigning it a speed means the two cannot disagree — a
/// cooler flame is automatically a slower one, and a dying fire visibly sags.
///
/// The Boussinesq approximation treats density as constant except in the buoyancy
/// term itself. It is accurate while the temperature difference is small relative to
/// the absolute temperature and increasingly optimistic beyond that; for a flame at
/// three or four times ambient it overstates the acceleration somewhat, which for
/// rendering is a forgiving direction to be wrong in.
///
/// Temperatures are absolute (kelvin). Returns m/s^2, positive upward.
///
/// # Errors
///
/// Returns an error if either temperature is at or below absolute zero.
///
/// # Examples
///
/// ```
/// use rs_physics::fluid_dynamics::thermal_buoyancy;
///
/// // A flame at 1500 K in 290 K air accelerates upward hard.
/// let flame = thermal_buoyancy(1500.0, 290.0, 9.81).unwrap();
/// assert!(flame > 30.0);
///
/// // Gas at ambient temperature does not rise at all.
/// let neutral = thermal_buoyancy(290.0, 290.0, 9.81).unwrap();
/// assert!(neutral.abs() < 1e-12);
///
/// // And something colder than its surroundings sinks.
/// let cold = thermal_buoyancy(250.0, 290.0, 9.81).unwrap();
/// assert!(cold < 0.0);
/// ```
pub fn thermal_buoyancy(
    temperature: f64,
    ambient: f64,
    gravity: f64,
) -> Result<f64, PhysicsError> {
    if temperature <= 0.0 || ambient <= 0.0 {
        return Err(PhysicsError::CalculationError(
            "temperatures must be absolute and above zero".to_string(),
        ));
    }
    Ok(gravity * (temperature - ambient) / ambient)
}

#[cfg(test)]
mod buoyancy_tests {
    use super::*;

    #[test]
    fn hotter_gas_rises_faster() {
        let warm = thermal_buoyancy(600.0, 290.0, 9.81).unwrap();
        let hot = thermal_buoyancy(1500.0, 290.0, 9.81).unwrap();
        assert!(hot > warm && warm > 0.0);
    }

    /// The property that makes a dying fire sag rather than stopping abruptly: rise
    /// falls away continuously with temperature, reaching zero exactly at ambient.
    #[test]
    fn buoyancy_falls_to_zero_at_ambient_and_reverses_below_it() {
        assert!(thermal_buoyancy(291.0, 290.0, 9.81).unwrap() > 0.0);
        assert_eq!(thermal_buoyancy(290.0, 290.0, 9.81).unwrap(), 0.0);
        assert!(thermal_buoyancy(289.0, 290.0, 9.81).unwrap() < 0.0);
    }

    #[test]
    fn absolute_zero_is_rejected_rather_than_dividing_by_it() {
        assert!(thermal_buoyancy(1000.0, 0.0, 9.81).is_err());
        assert!(thermal_buoyancy(0.0, 290.0, 9.81).is_err());
        assert!(thermal_buoyancy(-5.0, 290.0, 9.81).is_err());
    }
}
