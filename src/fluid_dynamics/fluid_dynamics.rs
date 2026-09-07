// src/fluid_dynamics.rs

use crate::utils::PhysicsError;
use super::validation::validate_positive;

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

    /// Creates a `Fluid` representing air at sea level and 20°C.
    ///
    /// Properties:
    /// - Density: 1.225 kg/m³
    /// - Dynamic viscosity: 1.81×10⁻⁵ Pa·s
    ///
    /// # Examples
    /// ```
    /// use rs_physics::fluid_dynamics::Fluid;
    ///
    /// let air = Fluid::air();
    /// assert!((air.density - 1.225).abs() < 0.01);
    /// ```
    pub fn air() -> Self {
        Self {
            density: 1.225,
            viscosity: 1.81e-5,
        }
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
