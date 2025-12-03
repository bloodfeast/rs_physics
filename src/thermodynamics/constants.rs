//! Physical constants for thermodynamics calculations
//!
//! This module provides fundamental physical constants used in thermodynamics,
//! heat transfer, and related calculations. All values are in SI units.

/// Universal gas constant (R) in J/(mol·K)
///
/// The molar gas constant, also known as the ideal gas constant.
/// R = 8.314462618 J/(mol·K)
///
/// This is the proportionality constant in the ideal gas law: PV = nRT
pub const R: f64 = 8.314462618;

/// Boltzmann constant (k_B) in J/K
///
/// The Boltzmann constant relates the average kinetic energy of particles
/// in a gas to the temperature of the gas.
/// k_B = 1.380649 × 10⁻²³ J/K
///
/// Related to R by: R = k_B × N_A
pub const K_B: f64 = 1.380649e-23;

/// Avogadro's number (N_A) in 1/mol
///
/// The number of particles (atoms, molecules, ions, etc.) in one mole.
/// N_A = 6.02214076 × 10²³ mol⁻¹
pub const N_A: f64 = 6.02214076e23;

/// Stefan-Boltzmann constant (σ) in W/(m²·K⁴)
///
/// The proportionality constant in the Stefan-Boltzmann law for blackbody radiation.
/// σ = 5.670374419 × 10⁻⁸ W/(m²·K⁴)
///
/// Power radiated by a black body: P = σAT⁴
pub const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;

/// Standard atmospheric pressure in Pascals
///
/// The average atmospheric pressure at sea level.
/// P_atm = 101325 Pa = 1 atm
pub const STANDARD_ATMOSPHERE: f64 = 101325.0;

/// Standard temperature in Kelvin (25°C)
///
/// The standard temperature used in thermodynamic tables and calculations.
/// T_std = 298.15 K = 25°C
pub const STANDARD_TEMPERATURE: f64 = 298.15;

/// Triple point of water in Kelvin
///
/// The temperature at which water can exist in all three phases simultaneously.
/// T_triple = 273.16 K
pub const WATER_TRIPLE_POINT: f64 = 273.16;

/// Freezing point of water at 1 atm in Kelvin
///
/// T_freeze = 273.15 K = 0°C
pub const WATER_FREEZING_POINT: f64 = 273.15;

/// Boiling point of water at 1 atm in Kelvin
///
/// T_boil = 373.15 K = 100°C
pub const WATER_BOILING_POINT: f64 = 373.15;

/// Specific heat capacity of water in J/(kg·K)
///
/// The amount of heat required to raise 1 kg of water by 1 K.
/// c_water = 4186 J/(kg·K) at 25°C
pub const WATER_SPECIFIC_HEAT: f64 = 4186.0;

/// Thermal conductivity of air in W/(m·K)
///
/// At 25°C and 1 atm.
/// k_air = 0.0262 W/(m·K)
pub const AIR_THERMAL_CONDUCTIVITY: f64 = 0.0262;

/// Thermal conductivity of water in W/(m·K)
///
/// At 25°C.
/// k_water = 0.606 W/(m·K)
pub const WATER_THERMAL_CONDUCTIVITY: f64 = 0.606;

/// Thermal conductivity of copper in W/(m·K)
///
/// At 25°C.
/// k_copper = 401 W/(m·K)
pub const COPPER_THERMAL_CONDUCTIVITY: f64 = 401.0;

/// Latent heat of fusion of water in J/kg
///
/// The heat required to melt 1 kg of ice at 0°C.
/// L_f = 334000 J/kg
pub const WATER_LATENT_HEAT_FUSION: f64 = 334_000.0;

/// Latent heat of vaporization of water in J/kg
///
/// The heat required to vaporize 1 kg of water at 100°C.
/// L_v = 2260000 J/kg
pub const WATER_LATENT_HEAT_VAPORIZATION: f64 = 2_260_000.0;

/// Molar heat capacity at constant volume for monatomic ideal gas
///
/// C_v = (3/2)R for monatomic gases (He, Ne, Ar, etc.)
pub const CV_MONATOMIC: f64 = 1.5 * R;

/// Molar heat capacity at constant pressure for monatomic ideal gas
///
/// C_p = (5/2)R for monatomic gases
pub const CP_MONATOMIC: f64 = 2.5 * R;

/// Molar heat capacity at constant volume for diatomic ideal gas
///
/// C_v = (5/2)R for diatomic gases (N₂, O₂, H₂, etc.) at moderate temperatures
pub const CV_DIATOMIC: f64 = 2.5 * R;

/// Molar heat capacity at constant pressure for diatomic ideal gas
///
/// C_p = (7/2)R for diatomic gases at moderate temperatures
pub const CP_DIATOMIC: f64 = 3.5 * R;

/// Heat capacity ratio (gamma) for monatomic ideal gas
///
/// γ = C_p/C_v = 5/3 ≈ 1.667 for monatomic gases
pub const GAMMA_MONATOMIC: f64 = 5.0 / 3.0;

/// Heat capacity ratio (gamma) for diatomic ideal gas
///
/// γ = C_p/C_v = 7/5 = 1.4 for diatomic gases
pub const GAMMA_DIATOMIC: f64 = 7.0 / 5.0;

/// Planck constant in J·s
///
/// h = 6.62607015 × 10⁻³⁴ J·s
pub const PLANCK: f64 = 6.62607015e-34;

/// Speed of light in vacuum in m/s
///
/// c = 299792458 m/s (exact)
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Wien's displacement constant in m·K
///
/// The constant relating the peak wavelength of blackbody radiation to temperature.
/// b = 2.897771955 × 10⁻³ m·K
///
/// λ_max = b / T
pub const WIEN_DISPLACEMENT: f64 = 2.897771955e-3;

// ============================================================================
// Temperature Conversion Functions
// ============================================================================

/// Converts temperature from Celsius to Kelvin.
///
/// # Arguments
///
/// * `celsius` - Temperature in degrees Celsius
///
/// # Returns
///
/// Temperature in Kelvin
///
/// # Examples
///
/// ```rust
/// use rs_physics::thermodynamics::celsius_to_kelvin;
///
/// let kelvin = celsius_to_kelvin(25.0);
/// assert!((kelvin - 298.15).abs() < 0.01);
///
/// let freezing = celsius_to_kelvin(0.0);
/// assert!((freezing - 273.15).abs() < 0.01);
/// ```
#[inline]
pub fn celsius_to_kelvin(celsius: f64) -> f64 {
    celsius + 273.15
}

/// Converts temperature from Kelvin to Celsius.
///
/// # Arguments
///
/// * `kelvin` - Temperature in Kelvin
///
/// # Returns
///
/// Temperature in degrees Celsius
///
/// # Examples
///
/// ```rust
/// use rs_physics::thermodynamics::kelvin_to_celsius;
///
/// let celsius = kelvin_to_celsius(298.15);
/// assert!((celsius - 25.0).abs() < 0.01);
///
/// let absolute_zero = kelvin_to_celsius(0.0);
/// assert!((absolute_zero - (-273.15)).abs() < 0.01);
/// ```
#[inline]
pub fn kelvin_to_celsius(kelvin: f64) -> f64 {
    kelvin - 273.15
}

/// Converts temperature from Fahrenheit to Kelvin.
///
/// # Arguments
///
/// * `fahrenheit` - Temperature in degrees Fahrenheit
///
/// # Returns
///
/// Temperature in Kelvin
///
/// # Examples
///
/// ```rust
/// use rs_physics::thermodynamics::fahrenheit_to_kelvin;
///
/// let kelvin = fahrenheit_to_kelvin(32.0);  // Freezing point of water
/// assert!((kelvin - 273.15).abs() < 0.01);
///
/// let boiling = fahrenheit_to_kelvin(212.0);  // Boiling point of water
/// assert!((boiling - 373.15).abs() < 0.01);
/// ```
#[inline]
pub fn fahrenheit_to_kelvin(fahrenheit: f64) -> f64 {
    (fahrenheit - 32.0) * 5.0 / 9.0 + 273.15
}

/// Converts temperature from Kelvin to Fahrenheit.
///
/// # Arguments
///
/// * `kelvin` - Temperature in Kelvin
///
/// # Returns
///
/// Temperature in degrees Fahrenheit
///
/// # Examples
///
/// ```rust
/// use rs_physics::thermodynamics::kelvin_to_fahrenheit;
///
/// let fahrenheit = kelvin_to_fahrenheit(273.15);  // Freezing point
/// assert!((fahrenheit - 32.0).abs() < 0.01);
///
/// let room_temp = kelvin_to_fahrenheit(298.15);  // ~77°F
/// assert!((room_temp - 77.0).abs() < 0.1);
/// ```
#[inline]
pub fn kelvin_to_fahrenheit(kelvin: f64) -> f64 {
    (kelvin - 273.15) * 9.0 / 5.0 + 32.0
}

/// Converts temperature from Celsius to Fahrenheit.
///
/// # Arguments
///
/// * `celsius` - Temperature in degrees Celsius
///
/// # Returns
///
/// Temperature in degrees Fahrenheit
///
/// # Examples
///
/// ```rust
/// use rs_physics::thermodynamics::celsius_to_fahrenheit;
///
/// let fahrenheit = celsius_to_fahrenheit(0.0);
/// assert!((fahrenheit - 32.0).abs() < 0.01);
///
/// let boiling = celsius_to_fahrenheit(100.0);
/// assert!((boiling - 212.0).abs() < 0.01);
/// ```
#[inline]
pub fn celsius_to_fahrenheit(celsius: f64) -> f64 {
    celsius * 9.0 / 5.0 + 32.0
}

/// Converts temperature from Fahrenheit to Celsius.
///
/// # Arguments
///
/// * `fahrenheit` - Temperature in degrees Fahrenheit
///
/// # Returns
///
/// Temperature in degrees Celsius
///
/// # Examples
///
/// ```rust
/// use rs_physics::thermodynamics::fahrenheit_to_celsius;
///
/// let celsius = fahrenheit_to_celsius(32.0);
/// assert!((celsius - 0.0).abs() < 0.01);
///
/// let body_temp = fahrenheit_to_celsius(98.6);
/// assert!((body_temp - 37.0).abs() < 0.1);
/// ```
#[inline]
pub fn fahrenheit_to_celsius(fahrenheit: f64) -> f64 {
    (fahrenheit - 32.0) * 5.0 / 9.0
}

// ============================================================================
// Pressure Conversion Functions
// ============================================================================

/// Converts pressure from atmospheres to Pascals.
///
/// # Arguments
///
/// * `atm` - Pressure in atmospheres
///
/// # Returns
///
/// Pressure in Pascals (Pa)
///
/// # Examples
///
/// ```rust
/// use rs_physics::thermodynamics::atm_to_pascals;
///
/// let pa = atm_to_pascals(1.0);
/// assert!((pa - 101325.0).abs() < 1.0);
/// ```
#[inline]
pub fn atm_to_pascals(atm: f64) -> f64 {
    atm * STANDARD_ATMOSPHERE
}

/// Converts pressure from Pascals to atmospheres.
///
/// # Arguments
///
/// * `pascals` - Pressure in Pascals
///
/// # Returns
///
/// Pressure in atmospheres
///
/// # Examples
///
/// ```rust
/// use rs_physics::thermodynamics::pascals_to_atm;
///
/// let atm = pascals_to_atm(101325.0);
/// assert!((atm - 1.0).abs() < 0.0001);
/// ```
#[inline]
pub fn pascals_to_atm(pascals: f64) -> f64 {
    pascals / STANDARD_ATMOSPHERE
}

/// Converts pressure from bar to Pascals.
///
/// # Arguments
///
/// * `bar` - Pressure in bar
///
/// # Returns
///
/// Pressure in Pascals (Pa)
///
/// # Examples
///
/// ```rust
/// use rs_physics::thermodynamics::bar_to_pascals;
///
/// let pa = bar_to_pascals(1.0);
/// assert!((pa - 100_000.0).abs() < 1.0);
/// ```
#[inline]
pub fn bar_to_pascals(bar: f64) -> f64 {
    bar * 100_000.0
}

/// Converts pressure from Pascals to bar.
///
/// # Arguments
///
/// * `pascals` - Pressure in Pascals
///
/// # Returns
///
/// Pressure in bar
///
/// # Examples
///
/// ```rust
/// use rs_physics::thermodynamics::pascals_to_bar;
///
/// let bar = pascals_to_bar(100_000.0);
/// assert!((bar - 1.0).abs() < 0.0001);
/// ```
#[inline]
pub fn pascals_to_bar(pascals: f64) -> f64 {
    pascals / 100_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gas_constant_value() {
        // R should be approximately 8.314 J/(mol·K)
        assert!((R - 8.314).abs() < 0.001);
        assert!((R - 8.314462618).abs() < 1e-9);
    }

    #[test]
    fn test_boltzmann_constant_value() {
        // k_B should be approximately 1.38 × 10⁻²³ J/K
        assert!((K_B - 1.38e-23).abs() < 0.01e-23);
        assert!((K_B - 1.380649e-23).abs() < 1e-29);
    }

    #[test]
    fn test_avogadro_constant_value() {
        // N_A should be approximately 6.022 × 10²³
        assert!((N_A - 6.022e23).abs() < 0.001e23);
        assert!((N_A - 6.02214076e23).abs() < 1e17);
    }

    #[test]
    fn test_stefan_boltzmann_constant_value() {
        // σ should be approximately 5.67 × 10⁻⁸ W/(m²·K⁴)
        assert!((STEFAN_BOLTZMANN - 5.67e-8).abs() < 0.01e-8);
        assert!((STEFAN_BOLTZMANN - 5.670374419e-8).abs() < 1e-15);
    }

    #[test]
    fn test_gas_constant_relationship() {
        // R = k_B × N_A
        let calculated_r = K_B * N_A;
        assert!((calculated_r - R).abs() < 1e-6,
            "R = k_B × N_A should hold, got {} vs {}", calculated_r, R);
    }

    #[test]
    fn test_standard_atmosphere() {
        assert!((STANDARD_ATMOSPHERE - 101325.0).abs() < 1e-6);
    }

    #[test]
    fn test_standard_temperature() {
        // 25°C = 298.15 K
        assert!((STANDARD_TEMPERATURE - 298.15).abs() < 1e-6);
    }

    #[test]
    fn test_water_temperatures() {
        // Triple point
        assert!((WATER_TRIPLE_POINT - 273.16).abs() < 1e-6);

        // Freezing point (0°C)
        assert!((WATER_FREEZING_POINT - 273.15).abs() < 1e-6);

        // Boiling point (100°C)
        assert!((WATER_BOILING_POINT - 373.15).abs() < 1e-6);
    }

    #[test]
    fn test_heat_capacity_relationships() {
        // C_p - C_v = R for ideal gases
        let diff_monatomic = CP_MONATOMIC - CV_MONATOMIC;
        assert!((diff_monatomic - R).abs() < 1e-10,
            "C_p - C_v = R for monatomic gas: got {}", diff_monatomic);

        let diff_diatomic = CP_DIATOMIC - CV_DIATOMIC;
        assert!((diff_diatomic - R).abs() < 1e-10,
            "C_p - C_v = R for diatomic gas: got {}", diff_diatomic);
    }

    #[test]
    fn test_gamma_ratios() {
        // γ = C_p / C_v
        let gamma_mono_calc = CP_MONATOMIC / CV_MONATOMIC;
        assert!((gamma_mono_calc - GAMMA_MONATOMIC).abs() < 1e-10);
        assert!((GAMMA_MONATOMIC - 5.0/3.0).abs() < 1e-10);

        let gamma_di_calc = CP_DIATOMIC / CV_DIATOMIC;
        assert!((gamma_di_calc - GAMMA_DIATOMIC).abs() < 1e-10);
        assert!((GAMMA_DIATOMIC - 1.4).abs() < 1e-10);
    }

    #[test]
    fn test_cv_values() {
        // Monatomic: C_v = (3/2)R
        assert!((CV_MONATOMIC - 1.5 * R).abs() < 1e-10);

        // Diatomic: C_v = (5/2)R
        assert!((CV_DIATOMIC - 2.5 * R).abs() < 1e-10);
    }

    #[test]
    fn test_cp_values() {
        // Monatomic: C_p = (5/2)R
        assert!((CP_MONATOMIC - 2.5 * R).abs() < 1e-10);

        // Diatomic: C_p = (7/2)R
        assert!((CP_DIATOMIC - 3.5 * R).abs() < 1e-10);
    }

    #[test]
    fn test_water_specific_heat() {
        // Water specific heat should be around 4186 J/(kg·K)
        assert!((WATER_SPECIFIC_HEAT - 4186.0).abs() < 1.0);
    }

    #[test]
    fn test_water_latent_heats() {
        // Latent heat of fusion ~ 334 kJ/kg
        assert!((WATER_LATENT_HEAT_FUSION - 334_000.0).abs() < 1000.0);

        // Latent heat of vaporization ~ 2260 kJ/kg
        assert!((WATER_LATENT_HEAT_VAPORIZATION - 2_260_000.0).abs() < 10000.0);
    }

    #[test]
    fn test_thermal_conductivities_ordering() {
        // Copper > Water > Air (typical ordering)
        assert!(COPPER_THERMAL_CONDUCTIVITY > WATER_THERMAL_CONDUCTIVITY);
        assert!(WATER_THERMAL_CONDUCTIVITY > AIR_THERMAL_CONDUCTIVITY);
    }

    #[test]
    fn test_planck_constant() {
        assert!((PLANCK - 6.626e-34).abs() < 0.001e-34);
    }

    #[test]
    fn test_speed_of_light() {
        assert!((SPEED_OF_LIGHT - 299_792_458.0).abs() < 1.0);
    }

    #[test]
    fn test_wien_displacement() {
        // Wien's law: λ_max = b/T
        // For the sun (~5778 K), peak should be around 500 nm
        let sun_peak_wavelength = WIEN_DISPLACEMENT / 5778.0;
        assert!((sun_peak_wavelength - 500e-9).abs() < 50e-9,
            "Sun peak wavelength should be around 500 nm, got {} nm",
            sun_peak_wavelength * 1e9);
    }
}
