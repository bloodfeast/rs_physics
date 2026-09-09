//! # Atmosphere
//!
//! The state of the air, and what that state does to anything moving through it.
//!
//! ## Why this module exists
//!
//! Before it, the crate held **two descriptions of air that could not be reconciled**.
//! [`crate::acoustics`] carried an `Air` with a temperature, a humidity and a pressure,
//! and used it to compute the speed of sound and the absorption of a path — a real
//! thermodynamic state. [`crate::fluid_dynamics::Fluid`] carried a `Fluid::air()` whose
//! density was the literal `1.225` and whose viscosity was the literal `1.81e-5`, frozen
//! at one temperature that the doc comment named as 20 °C and that was in fact 15 °C.
//!
//! Nothing connected them. A caller could hold a winter atmosphere for sound and a
//! summer atmosphere for drag *in the same frame*, and no type, test or assertion would
//! object. Winter air at 270 K is **9% denser** than summer air at 293 K, and density is
//! linear in every drag force, every dynamic pressure and every acoustic impedance the
//! crate computes.
//!
//! So: there is now one air. [`Air`] is the state — three numbers that fix the gas — and
//! everything else is *derived from it*:
//!
//! * [`Air::density`] — ρ = pM/(RT), with the water-vapour correction.
//! * [`Air::dynamic_viscosity`] — Sutherland's law.
//! * [`Air::speed_of_sound`], [`Air::absorption_db_per_m`] — in [`crate::acoustics`],
//!   which keeps its own physics but no longer keeps its own idea of what air is.
//! * [`crate::fluid_dynamics::Fluid::from_air`] — the *only* way to obtain a `Fluid` of
//!   air. `Fluid::air()` has been removed, because a function that hands back air
//!   without being told which air is the bug this module exists to close.
//!
//! The three numbers are not interchangeable and never validated at the point of use,
//! so they are validated at the point of construction: [`Air::new`] returns a
//! `Result` and the fields are private. Every method below is then total — no clamps,
//! no `max(1.0)` guarding a `sqrt`, no divide that can produce an infinity.
//!
//! ## And the boundary layer
//!
//! [`boundary_layer`] holds the closed-form results for wind near the ground: how speed
//! varies with height over a given surface, how it accelerates over a rise, and how it
//! is damped inside a plant canopy. Every one is an algebraic expression with a
//! citation and a stated validity limit. **None of it is a solver** — see that module's
//! own documentation for why a grid is the wrong answer to this question.
//!
//! ## Example
//!
//! ```
//! use rs_physics::atmosphere::Air;
//!
//! // The same weather that darkens a distant gunshot also thickens the air a
//! // shell flies through, and now it is the same value doing both.
//! let summer = Air::standard();
//! let winter = Air::winter();
//!
//! assert!(winter.density() > summer.density() * 1.08);
//! assert!(winter.speed_of_sound() < summer.speed_of_sound());
//! ```

pub mod boundary_layer;

pub use boundary_layer::{
    canopy_profile, crest_speedup, dynamic_pressure, fractional_speedup_from_slope,
    friction_velocity, is_within_linearisation, log_profile, power_law, speedup_at_height,
    CanopyAttenuation, HillForm, Surface, WindProfile, LINEARISATION_SLOPE_LIMIT, VON_KARMAN,
};

use crate::utils::PhysicsError;

/// The molar gas constant, in J/(mol·K). Exact by the 2019 SI redefinition.
pub const MOLAR_GAS_CONSTANT: f64 = 8.314_462_618_153_24;

/// Molar mass of dry air, in kg/mol.
///
/// The ICAO standard atmosphere's value, which is what makes [`Air::sea_level`] come out
/// at the 1.225 kg/m³ every aerodynamics table quotes. Real air varies in the fourth
/// decimal with CO₂ concentration; nothing here is sensitive to that.
pub const MOLAR_MASS_DRY_AIR: f64 = 0.028_964_4;

/// Molar mass of water, in kg/mol.
pub const MOLAR_MASS_WATER: f64 = 0.018_015_28;

/// `1 - M_water / M_dry_air`, the coefficient of the humidity correction to density.
///
/// Water vapour is *lighter* than the nitrogen and oxygen it displaces, so humid air is
/// less dense than dry air at the same temperature and pressure — which surprises almost
/// everyone, and is why a pitcher gets more carry on a muggy day.
const HUMIDITY_DENSITY_COEFFICIENT: f64 = 1.0 - MOLAR_MASS_WATER / MOLAR_MASS_DRY_AIR;

/// Standard sea-level pressure, in pascals. One standard atmosphere.
pub const STANDARD_PRESSURE: f64 = 101_325.0;

/// ICAO standard atmosphere sea-level temperature, in kelvin. 15 °C.
pub const SEA_LEVEL_TEMPERATURE: f64 = 288.15;

/// The reference temperature of the acoustic absorption model, in kelvin. 20 °C.
pub const REFERENCE_TEMPERATURE: f64 = 293.15;

/// Triple-point isotherm of water, in kelvin. Appears in the saturation-vapour fit.
const TRIPLE_POINT: f64 = 273.16;

/// Absolute zero expressed in degrees Celsius, negated: the offset between the two
/// scales, in kelvin.
///
/// The crate's [`crate::thermodynamics::celsius_to_kelvin`] holds the same number, but
/// that module is behind a feature flag and [`Air`] is not, so a caller with acoustics
/// but not thermodynamics would otherwise write the constant themselves — which is
/// exactly what happened. See [`Air::from_celsius`].
pub const ABSOLUTE_ZERO_CELSIUS: f64 = 273.15;

/// Sutherland's reference viscosity for air, in Pa·s, at [`SUTHERLAND_T0`].
const SUTHERLAND_MU0: f64 = 1.716e-5;
/// Sutherland's reference temperature for air, in kelvin.
const SUTHERLAND_T0: f64 = 273.15;
/// Sutherland's constant for air, in kelvin.
const SUTHERLAND_S: f64 = 110.4;

/// Specific heat capacity of air at constant pressure, in J/(kg·K).
pub const SPECIFIC_HEAT_CAPACITY_AIR: f64 = 1005.0;

/// Prandtl number of air, dimensionless. Near-constant from 250 K to 1000 K.
pub const PRANDTL_NUMBER_AIR: f64 = 0.71;

/// The state of the air: the three numbers that fix it, and nothing else.
///
/// Everything the crate wants to know about air — how dense it is, how viscous, how fast
/// sound crosses it, how much of a sound it eats per metre — follows from these. None of
/// them is stored anywhere else, so two parts of a program cannot disagree about the
/// weather.
///
/// # Invariants
///
/// Held by [`Air::new`] and the `with_*` methods, which are the only ways to build one
/// other than the named presets. The fields are private *because* of this: every method
/// on `Air` divides by temperature or takes its square root, and a caller-assembled
/// struct literal with `temperature: 0.0` would put an infinity into a density, a drag
/// force, and eventually a position.
///
/// * `temperature` is finite and above 0 K.
/// * `pressure` is finite and above 0 Pa.
/// * `humidity` is in `0.0..=1.0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Air {
    /// Temperature in kelvin. Strictly positive.
    temperature: f64,
    /// Relative humidity, 0 to 1.
    humidity: f64,
    /// Static pressure in pascals. Strictly positive.
    pressure: f64,
}

impl Air {
    /// A state of the air, checked.
    ///
    /// # Arguments
    ///
    /// * `temperature_k` — temperature in kelvin. Must be finite and above zero.
    /// * `humidity` — relative humidity as a fraction, 0 to 1. Not a percentage.
    /// * `pressure_pa` — static pressure in pascals. Must be finite and above zero.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::CalculationError`] naming the offending quantity. The three are
    /// reported separately rather than as one "invalid air", because a caller that got
    /// the humidity in percent has a different bug from one whose temperature came out
    /// of a NaN-poisoned weather model.
    pub fn new(temperature_k: f64, humidity: f64, pressure_pa: f64) -> Result<Air, PhysicsError> {
        if !temperature_k.is_finite() || temperature_k <= 0.0 {
            return Err(PhysicsError::CalculationError(format!(
                "air temperature must be finite and above 0 K, got {temperature_k}"
            )));
        }
        if !humidity.is_finite() || !(0.0..=1.0).contains(&humidity) {
            return Err(PhysicsError::CalculationError(format!(
                "relative humidity is a fraction in 0..=1, not a percentage; got {humidity}"
            )));
        }
        if !pressure_pa.is_finite() || pressure_pa <= 0.0 {
            return Err(PhysicsError::CalculationError(format!(
                "air pressure must be finite and above 0 Pa, got {pressure_pa}"
            )));
        }
        Ok(Air { temperature: temperature_k, humidity, pressure: pressure_pa })
    }

    /// A state of the air with the temperature given in **degrees Celsius**.
    ///
    /// # Why this exists rather than leaving the caller to add 273.15
    ///
    /// Because a caller already got it wrong. `Air`'s temperature is absolute, and a
    /// temperate year written in Celsius sits in that field looking entirely plausible —
    /// 12.0 is a perfectly good number of kelvin. It is only wrong by a factor that makes
    /// sound travel at 70 m/s, and nothing in the type system had anything to say about
    /// it, because both readings are `f64` and both are positive.
    ///
    /// [`Air::new`] cannot catch this: the crate is also used for combustion at 3000 K, so
    /// there is no plausible-temperature band to check against. What *can* be done is to
    /// stop the conversion being the caller's job. The `+ 273.15` now exists once, here,
    /// where it is named in a signature instead of carried in a comment.
    ///
    /// This is the cheap end of the units argument. A full `Kelvin`/`Celsius` newtype pair
    /// would catch more and cost every call site a wrapper; a constructor per unit costs
    /// nothing and catches the case that actually occurred.
    ///
    /// # Arguments
    ///
    /// * `temperature_c` — temperature in degrees Celsius. Must be above −273.15.
    /// * `humidity` — relative humidity as a fraction, 0 to 1.
    /// * `pressure_pa` — static pressure in pascals.
    ///
    /// # Errors
    ///
    /// As [`Air::new`]. A temperature at or below absolute zero is reported in the units
    /// it was supplied in, so the message names Celsius rather than a kelvin figure the
    /// caller never typed.
    ///
    /// # Examples
    /// ```
    /// use rs_physics::atmosphere::{Air, STANDARD_PRESSURE};
    ///
    /// let spring = Air::from_celsius(12.0, 0.75, STANDARD_PRESSURE).unwrap();
    /// assert!((spring.temperature() - 285.15).abs() < 1e-9);
    ///
    /// // And the mistake the conversion existed to prevent is now a different function.
    /// assert!(Air::from_celsius(-300.0, 0.5, STANDARD_PRESSURE).is_err());
    /// ```
    pub fn from_celsius(
        temperature_c: f64,
        humidity: f64,
        pressure_pa: f64,
    ) -> Result<Air, PhysicsError> {
        if !temperature_c.is_finite() || temperature_c <= -ABSOLUTE_ZERO_CELSIUS {
            return Err(PhysicsError::CalculationError(format!(
                "air temperature must be finite and above absolute zero (-273.15 C), \
                 got {temperature_c} C"
            )));
        }
        Air::new(temperature_c + ABSOLUTE_ZERO_CELSIUS, humidity, pressure_pa)
    }

    /// This air's temperature in degrees Celsius.
    ///
    /// The other half of [`Air::from_celsius`]: a caller displaying a temperature should
    /// not be writing `− 273.15` either.
    #[inline]
    pub fn temperature_celsius(&self) -> f64 {
        self.temperature - ABSOLUTE_ZERO_CELSIUS
    }

    /// 20 °C, half humidity, sea level. The acoustics reference condition.
    pub const fn standard() -> Air {
        Air { temperature: REFERENCE_TEMPERATURE, humidity: 0.5, pressure: STANDARD_PRESSURE }
    }

    /// The ICAO standard atmosphere at sea level: 15 °C, dry, 101 325 Pa.
    ///
    /// This is the state whose density is the **1.225 kg/m³** that every aerodynamics
    /// table quotes and that this crate used to carry as a literal in three places. It
    /// is now computed, so it can be checked against the standard rather than trusted.
    pub const fn sea_level() -> Air {
        Air { temperature: SEA_LEVEL_TEMPERATURE, humidity: 0.0, pressure: STANDARD_PRESSURE }
    }

    /// A cold, dry day: −3 °C at 40% humidity.
    ///
    /// Carries high frequencies noticeably further than [`Air::standard`], and is about
    /// 9% denser — so the same shell flies shorter and the same shot is heard further.
    pub const fn winter() -> Air {
        Air { temperature: 270.0, humidity: 0.4, pressure: STANDARD_PRESSURE }
    }

    /// Temperature in kelvin.
    #[inline]
    pub const fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Relative humidity, 0 to 1.
    #[inline]
    pub const fn humidity(&self) -> f64 {
        self.humidity
    }

    /// Static pressure in pascals.
    #[inline]
    pub const fn pressure(&self) -> f64 {
        self.pressure
    }

    /// The same air at a different temperature, in kelvin.
    ///
    /// The replacement for `Air { temperature: t, ..base }`, which the private fields no
    /// longer allow. Returns `Result` for the same reason [`Air::new`] does.
    pub fn with_temperature(self, temperature_k: f64) -> Result<Air, PhysicsError> {
        Air::new(temperature_k, self.humidity, self.pressure)
    }

    /// The same air at a different relative humidity, 0 to 1.
    pub fn with_humidity(self, humidity: f64) -> Result<Air, PhysicsError> {
        Air::new(self.temperature, humidity, self.pressure)
    }

    /// The same air at a different static pressure, in pascals.
    pub fn with_pressure(self, pressure_pa: f64) -> Result<Air, PhysicsError> {
        Air::new(self.temperature, self.humidity, pressure_pa)
    }

    /// Saturation vapour pressure of water at this temperature, in pascals.
    ///
    /// ISO 9613-1's fit, which is what the absorption model is written against — so the
    /// crate has **one** water-vapour model rather than one for sound and another for
    /// density. Good to a couple of tenths of a per cent against steam tables between
    /// −20 °C and 50 °C.
    #[inline]
    pub fn saturation_vapour_pressure(&self) -> f64 {
        let t = self.temperature / TRIPLE_POINT;
        let exponent = -6.8346 * t.powf(-1.261) + 4.6151;
        STANDARD_PRESSURE * 10f64.powf(exponent)
    }

    /// Mole fraction of water vapour, dimensionless, 0 to 1.
    ///
    /// What actually matters is how many water molecules are present, and a given
    /// relative humidity is a very different number of them at 0 °C and at 30 °C. Both
    /// the density correction and the acoustic absorption model are written in terms of
    /// this rather than of relative humidity.
    ///
    /// Returned as a **fraction**, per the crate's SI convention. ISO 9613-1 writes its
    /// absorption formula in *percent*; that factor of 100 lives at exactly one place,
    /// inside [`Air::absorption_db_per_m`], next to the formula that needs it.
    ///
    /// Bounded above by 1 because a mole fraction cannot exceed one — that is the
    /// definition, not a guard. It binds only for states whose saturation pressure
    /// exceeds the static pressure, i.e. superheated air, which is outside anything the
    /// absorption model was fitted for anyway.
    #[inline]
    pub fn water_vapour_mole_fraction(&self) -> f64 {
        (self.humidity * self.saturation_vapour_pressure() / self.pressure).min(1.0)
    }

    /// Density, in kg/m³.
    ///
    /// The ideal gas law with the humidity correction:
    ///
    /// ```text
    /// rho = (p * M_dry) / (R * T) * (1 - (1 - M_water/M_dry) * x_v)
    /// ```
    ///
    /// where `x_v` is [`Air::water_vapour_mole_fraction`]. Three multiplies and a
    /// divide, and it is the number that was frozen at 1.225 across the whole crate.
    ///
    /// **This is the fix for the two-airs defect.** Compare
    /// `Air::winter().density()` at 1.306 against `Air::standard().density()` at 1.199:
    /// a 9% difference that runs straight into drag, dynamic pressure and acoustic
    /// impedance, and that the frozen constant threw away.
    ///
    /// Accurate to about 0.1% against CIPM-2007 for ordinary near-surface conditions.
    /// Air stops behaving ideally at pressures far above one atmosphere, which is well
    /// outside anything this crate is for.
    #[inline]
    pub fn density(&self) -> f64 {
        let dry = self.pressure * MOLAR_MASS_DRY_AIR
            / (MOLAR_GAS_CONSTANT * self.temperature);
        dry * (1.0 - HUMIDITY_DENSITY_COEFFICIENT * self.water_vapour_mole_fraction())
    }

    /// Dynamic viscosity, in Pa·s.
    ///
    /// Sutherland's law (Sutherland 1893), with the constants for air from White,
    /// *Viscous Fluid Flow* 3rd ed. Table 1-2:
    ///
    /// ```text
    /// mu = mu0 * (T0 + S)/(T + S) * (T/T0)^1.5
    /// ```
    ///
    /// Good to about 2% from 170 K to 1900 K, which covers everything from a polar night
    /// to the inside of a fire. Humidity changes air's viscosity by well under 1% and is
    /// not modelled; pressure does not enter at all, which is a real result for a gas and
    /// not an omission.
    #[inline]
    pub fn dynamic_viscosity(&self) -> f64 {
        let t = self.temperature;
        SUTHERLAND_MU0 * ((SUTHERLAND_T0 + SUTHERLAND_S) / (t + SUTHERLAND_S))
            * (t / SUTHERLAND_T0).powf(1.5)
    }

    /// Kinematic viscosity, in m²/s. `mu / rho`.
    #[inline]
    pub fn kinematic_viscosity(&self) -> f64 {
        self.dynamic_viscosity() / self.density()
    }

    /// Specific heat capacity at constant pressure, in J/(kg·K).
    ///
    /// Constant to within about 2% from 250 K to 450 K (Incropera & DeWitt,
    /// *Fundamentals of Heat and Mass Transfer*, Table A.4), which is why this is a
    /// constant and not a fit. It is a method rather than a bare `const` so that
    /// [`Air`] stays the single place to ask about air, and so that a temperature
    /// dependence can be added later without moving any call site.
    #[inline]
    pub fn specific_heat_capacity(&self) -> f64 {
        SPECIFIC_HEAT_CAPACITY_AIR
    }

    /// Thermal conductivity, in W/(m·K).
    ///
    /// `k = mu * c_p / Pr`. Air's Prandtl number is very nearly constant — 0.71 from
    /// about 250 K to 1000 K (Incropera & DeWitt, Table A.4) — so conductivity follows
    /// from the Sutherland viscosity rather than needing a fit of its own. That is worth
    /// having: it means the momentum and the heat sides of the module cannot disagree
    /// about what temperature it is.
    ///
    /// Reproduces the published 0.0243 W/(m·K) at 0 °C and 0.0262 at 25 °C to about 1%.
    #[inline]
    pub fn thermal_conductivity(&self) -> f64 {
        self.dynamic_viscosity() * self.specific_heat_capacity() / PRANDTL_NUMBER_AIR
    }
}

impl Default for Air {
    fn default() -> Self {
        Air::standard()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published ICAO figure, which is where the crate's old hardcoded 1.225 came
    /// from — except that it was labelled 20 °C and 1.225 is the **15 °C** value. This
    /// test is the one that would have caught that.
    #[test]
    fn sea_level_density_is_the_icao_standard() {
        let rho = Air::sea_level().density();
        assert!(
            (rho - 1.225).abs() < 0.001,
            "ISA sea level came out {rho:.4} kg/m3 against the standard's 1.225",
        );
    }

    /// And 20 °C at half humidity is *not* 1.225 — it is about 1.199, which is what the
    /// old `Fluid::air()` doc comment claimed to be describing and was 2% wrong about.
    #[test]
    fn room_temperature_air_is_lighter_than_the_isa_reference() {
        let rho = Air::standard().density();
        assert!(
            (rho - 1.199).abs() < 0.005,
            "20 C air at 50% RH came out {rho:.4} kg/m3 against a textbook 1.199",
        );
        assert!(rho < Air::sea_level().density());
    }

    /// **The defect this module was opened for.** A caller on a winter map used to get
    /// summer density for drag and winter temperature for sound, from the same frame.
    #[test]
    fn winter_air_is_about_nine_percent_denser_than_summer_air() {
        let ratio = Air::winter().density() / Air::standard().density();
        assert!(
            (1.08..1.11).contains(&ratio),
            "winter/summer density ratio came out {ratio:.4}; the physical answer is \
             about 1.09, and the old frozen 1.225 made it exactly 1.0",
        );
    }

    /// Humid air is *lighter*, because water displaces heavier nitrogen. The direction
    /// is the whole point of carrying the correction; getting it backwards would be
    /// invisible in magnitude and wrong in sign.
    #[test]
    fn humid_air_is_lighter_than_dry_air() {
        let base = Air::standard();
        let dry = base.with_humidity(0.0).unwrap();
        let damp = base.with_humidity(1.0).unwrap();
        assert!(
            damp.density() < dry.density(),
            "saturated air came out denser than dry air, which is backwards",
        );
        // And by a small amount: about 1% at 20 C, not 10%.
        let drop = 1.0 - damp.density() / dry.density();
        assert!((0.005..0.02).contains(&drop), "humidity moved density by {drop:.4}");
    }

    /// Ideal gas: halving absolute temperature at fixed pressure doubles density.
    #[test]
    fn density_is_inverse_in_absolute_temperature() {
        let dry = Air::new(600.0, 0.0, STANDARD_PRESSURE).unwrap();
        let colder = dry.with_temperature(300.0).unwrap();
        let ratio = colder.density() / dry.density();
        assert!((ratio - 2.0).abs() < 1e-9, "halving T gave a density ratio of {ratio}");
    }

    /// And linear in pressure.
    #[test]
    fn density_is_linear_in_pressure() {
        let low = Air::sea_level().with_pressure(STANDARD_PRESSURE / 2.0).unwrap();
        let ratio = Air::sea_level().density() / low.density();
        assert!((ratio - 2.0).abs() < 1e-9, "halving p gave a density ratio of {ratio}");
    }

    /// Published: 1.81e-5 Pa·s at 20 °C, and 1.789e-5 at the ISA reference. The first is
    /// the literal the old `Fluid::air()` carried, so Sutherland reproducing it is what
    /// says the replacement is not a regression.
    #[test]
    fn viscosity_matches_the_textbook() {
        let at_20c = Air::standard().dynamic_viscosity();
        assert!(
            (at_20c - 1.81e-5).abs() < 2e-7,
            "20 C viscosity came out {at_20c:.4e} against a textbook 1.81e-5",
        );
        let isa = Air::sea_level().dynamic_viscosity();
        assert!(
            (isa - 1.7894e-5).abs() < 1e-7,
            "ISA viscosity came out {isa:.5e} against the standard's 1.7894e-5",
        );
    }

    /// A gas gets *more* viscous when heated, which is the opposite of a liquid and the
    /// single most common way to get Sutherland's law in backwards.
    #[test]
    fn a_gas_thickens_when_heated() {
        let cold = Air::new(250.0, 0.0, STANDARD_PRESSURE).unwrap();
        let hot = Air::new(400.0, 0.0, STANDARD_PRESSURE).unwrap();
        assert!(
            hot.dynamic_viscosity() > cold.dynamic_viscosity(),
            "hot air came out less viscous than cold; this is the liquid intuition",
        );
    }

    /// Kinematic viscosity of air at 20 °C is about 1.5e-5 m²/s — an order of magnitude
    /// larger than water's, which is why Reynolds numbers in air are small.
    #[test]
    fn kinematic_viscosity_matches_the_textbook() {
        let nu = Air::standard().kinematic_viscosity();
        assert!(
            (nu - 1.51e-5).abs() < 1e-6,
            "kinematic viscosity came out {nu:.3e} against a textbook 1.51e-5",
        );
    }

    /// Saturation vapour pressure at 20 °C is 2339 Pa, and at 0 °C is 611 Pa. Both are
    /// steam-table values, not values this fit produced.
    #[test]
    fn saturation_vapour_pressure_matches_the_steam_tables() {
        let at_20c = Air::standard().saturation_vapour_pressure();
        assert!((at_20c - 2339.0).abs() < 15.0, "20 C: {at_20c:.0} Pa against 2339");
        let at_0c = Air::new(273.15, 0.0, STANDARD_PRESSURE)
            .unwrap()
            .saturation_vapour_pressure();
        assert!((at_0c - 611.0).abs() < 10.0, "0 C: {at_0c:.0} Pa against 611");
    }

    /// Published: 0.0243 W/(m·K) at 0 °C and 0.0262 at 25 °C. The second is the literal
    /// `Substance::air()` used to carry, so the constant-Prandtl relation reproducing it
    /// is what says deriving it is not a regression.
    #[test]
    fn thermal_conductivity_matches_the_textbook() {
        let at_0c = Air::new(273.15, 0.0, STANDARD_PRESSURE).unwrap().thermal_conductivity();
        assert!((at_0c - 0.0243).abs() < 0.0005, "0 C: {at_0c:.4} W/(m K) against 0.0243");
        let at_25c = Air::new(298.15, 0.0, STANDARD_PRESSURE).unwrap().thermal_conductivity();
        assert!((at_25c - 0.0262).abs() < 0.0005, "25 C: {at_25c:.4} W/(m K) against 0.0262");
    }

    /// **This is the test that keeps the airs from diverging again.**
    ///
    /// Three types in three modules describe air, and before this work each carried its
    /// own literal density: 1.225, 1.225 and 1.184, at three unstated temperatures. Now
    /// all three derive from the same [`Air`], so the assertion is not "they happen to
    /// match at one state" but "they match at every state" — and any future constructor
    /// that reintroduces a literal fails here rather than in someone's drag force.
    #[test]
    #[cfg(all(feature = "fluid_dynamics", feature = "thermodynamics"))]
    fn every_description_of_air_in_the_crate_agrees_at_every_state() {
        use crate::fluid_dynamics::Fluid;
        use crate::thermodynamics::Substance;

        for &t in &[250.0, 273.15, 288.15, 293.15, 320.0, 400.0] {
            for &rh in &[0.0, 0.35, 1.0] {
                for &p in &[80_000.0, STANDARD_PRESSURE, 110_000.0] {
                    let air = Air::new(t, rh, p).unwrap();
                    let fluid = Fluid::from_air(&air);
                    let substance = Substance::from_air(&air);

                    assert_eq!(
                        fluid.density, substance.density,
                        "Fluid and Substance disagree about the density of air at \
                         {t} K, {rh} RH, {p} Pa",
                    );
                    assert_eq!(
                        fluid.density,
                        air.density(),
                        "Fluid has drifted from Air at {t} K, {rh} RH, {p} Pa",
                    );
                    assert_eq!(
                        fluid.viscosity,
                        air.dynamic_viscosity(),
                        "Fluid viscosity has drifted from Air at {t} K",
                    );
                }
            }
        }
    }

    /// And the removal is not a regression: the ICAO reference reproduces the exact
    /// constant `Fluid::air()` used to hand back, so callers who wanted that number can
    /// still get it — by naming the state it belonged to.
    #[test]
    #[cfg(feature = "fluid_dynamics")]
    fn the_isa_reference_reproduces_the_constant_that_was_removed() {
        let isa = crate::fluid_dynamics::Fluid::from_air(&Air::sea_level());
        assert!((isa.density - 1.225).abs() < 0.001, "density {:.4}", isa.density);
        assert!(
            (isa.viscosity - 1.7894e-5).abs() < 1e-7,
            "viscosity {:.5e} against the ISA 1.7894e-5",
            isa.viscosity,
        );
    }

    /// The invariants the private fields exist to hold. Each of these used to be
    /// constructible as a struct literal and would have put an infinity or a NaN into
    /// every derived quantity.
    #[test]
    fn impossible_air_is_rejected_at_construction() {
        assert!(Air::new(0.0, 0.5, STANDARD_PRESSURE).is_err(), "0 K accepted");
        assert!(Air::new(-10.0, 0.5, STANDARD_PRESSURE).is_err(), "negative K accepted");
        assert!(Air::new(f64::NAN, 0.5, STANDARD_PRESSURE).is_err(), "NaN K accepted");
        assert!(Air::new(293.15, 50.0, STANDARD_PRESSURE).is_err(), "humidity in % accepted");
        assert!(Air::new(293.15, -0.1, STANDARD_PRESSURE).is_err(), "negative RH accepted");
        assert!(Air::new(293.15, f64::NAN, STANDARD_PRESSURE).is_err(), "NaN RH accepted");
        assert!(Air::new(293.15, 0.5, 0.0).is_err(), "vacuum accepted");
        assert!(Air::new(293.15, 0.5, f64::INFINITY).is_err(), "infinite pressure accepted");
    }

    /// The Celsius constructor round-trips, and refuses temperatures below absolute zero
    /// in the units they were supplied in.
    #[test]
    fn the_celsius_constructor_round_trips() {
        for &c in &[-40.0, -3.0, 0.0, 12.0, 26.0, 100.0] {
            let air = Air::from_celsius(c, 0.5, STANDARD_PRESSURE).unwrap();
            assert!(
                (air.temperature_celsius() - c).abs() < 1e-9,
                "{c} C went in and {} C came back",
                air.temperature_celsius(),
            );
            assert!((air.temperature() - (c + 273.15)).abs() < 1e-9);
        }
        assert!(Air::from_celsius(-273.15, 0.5, STANDARD_PRESSURE).is_err(), "0 K accepted");
        assert!(Air::from_celsius(-300.0, 0.5, STANDARD_PRESSURE).is_err(), "below 0 K accepted");
        assert!(Air::from_celsius(f64::NAN, 0.5, STANDARD_PRESSURE).is_err(), "NaN accepted");
    }

    /// **The consumer's shipped bug, as a test.** A temperate year written in Celsius and
    /// dropped into an absolute field is a plausible positive number that makes sound
    /// travel at a third of its speed. `Air::new` cannot reject it — 12 K is a legitimate
    /// state — so the fix is that the conversion is no longer the caller's to forget.
    #[test]
    fn celsius_and_kelvin_are_wildly_different_air_and_only_one_constructor_can_confuse_them() {
        let correct = Air::from_celsius(12.0, 0.5, STANDARD_PRESSURE).unwrap();
        let the_bug = Air::new(12.0, 0.5, STANDARD_PRESSURE).unwrap();
        assert!(the_bug.speed_of_sound() < correct.speed_of_sound() * 0.25);
        assert!(
            (correct.speed_of_sound() - 338.0).abs() < 3.0,
            "12 C air came out at {:.1} m/s against a textbook 338",
            correct.speed_of_sound(),
        );
    }

    /// And because they are rejected there, nothing downstream has to check. This is the
    /// property the private fields buy: every derived quantity is finite for every `Air`
    /// that exists.
    #[test]
    fn every_constructible_air_yields_finite_derived_quantities() {
        for &t in &[1.0, 100.0, 250.0, 293.15, 1000.0, 3000.0] {
            for &rh in &[0.0, 0.5, 1.0] {
                for &p in &[1.0, 1_000.0, STANDARD_PRESSURE, 1e7] {
                    let air = Air::new(t, rh, p).unwrap();
                    assert!(air.density().is_finite(), "density at {t}K {rh} {p}Pa");
                    assert!(air.density() > 0.0, "non-positive density at {t}K {rh} {p}Pa");
                    assert!(air.dynamic_viscosity().is_finite(), "viscosity at {t}K");
                    assert!(air.kinematic_viscosity().is_finite(), "nu at {t}K {rh} {p}Pa");
                    assert!(air.speed_of_sound().is_finite(), "c at {t}K");
                    assert!(
                        air.absorption_db_per_m(4_000.0).is_finite(),
                        "absorption at {t}K {rh} {p}Pa",
                    );
                }
            }
        }
    }
}
