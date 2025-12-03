//! # Materials Module
//!
//! Physical material properties for physics simulations.
//!
//! This module provides a comprehensive `Material` struct that encapsulates
//! mechanical, thermal, and collision properties of materials, along with
//! predefined common materials and utility functions.
//!
//! ## Features
//!
//! - **Predefined materials**: Steel, stainless steel, aluminum, copper, titanium,
//!   brass, concrete, glass, rubber, polyurethane, wood, ice
//! - **Mechanical properties**: Young's modulus, yield/ultimate strength, Poisson's ratio
//! - **Thermal properties**: Thermal conductivity, specific heat capacity, thermal diffusivity
//! - **Collision properties**: Friction coefficient, restitution coefficient
//! - **Derived calculations**: Shear modulus, bulk modulus, strain energy
//! - **Failure analysis**: Breakage prediction, fatigue life estimation
//! - **Builder pattern**: `MaterialBuilder` for creating custom materials
//!
//! ## Quick Start
//!
//! ```rust
//! use rs_physics::materials::Material;
//!
//! // Use a predefined material
//! let steel = Material::steel();
//! println!("Steel density: {} kg/m³", steel.density);
//! println!("Steel shear modulus: {} Pa", steel.shear_modulus());
//!
//! // Check if material will break under stress
//! let result = steel.will_break(300e6, 0.0015, None); // 300 MPa stress
//! if result.will_break {
//!     println!("Material will fail!");
//! }
//! ```
//!
//! ## Predefined Materials
//!
//! | Material | Density (kg/m³) | Young's Modulus (GPa) | Use Case |
//! |----------|-----------------|----------------------|----------|
//! | Steel | 7850 | 200 | Structural components |
//! | Stainless Steel | 8000 | 193 | Corrosion resistant parts |
//! | Aluminum | 2700 | 69 | Lightweight structures |
//! | Copper | 8960 | 110 | Heat transfer, electrical |
//! | Titanium | 4430 | 114 | Aerospace, medical |
//! | Brass | 8530 | 110 | Decorative, low friction |
//! | Concrete | 2400 | 30 | Construction |
//! | Glass | 2500 | 70 | Windows, containers |
//! | Rubber | 1100 | 0.01 | Dampers, seals |
//! | Polyurethane | 1200 | 0.02 | Flexible components |
//! | Wood | 700 | 12 | Construction |
//! | Ice | 917 | 9.3 | Environmental simulation |
//!
//! ## Creating Custom Materials
//!
//! ### Using the Builder Pattern (Recommended)
//!
//! ```rust
//! use rs_physics::materials::MaterialBuilder;
//!
//! // Start fresh with default values
//! let custom = MaterialBuilder::new()
//!     .density(5000.0)
//!     .youngs_modulus(150e9)
//!     .friction_coefficient(0.5)
//!     .build()
//!     .expect("Valid material");
//!
//! // Or modify an existing material
//! use rs_physics::materials::Material;
//! let modified = MaterialBuilder::from(Material::aluminum())
//!     .friction_coefficient(0.8)
//!     .build()
//!     .expect("Valid material");
//! ```
//!
//! ### Using the Constructor
//!
//! ```rust
//! use rs_physics::materials::Material;
//!
//! let titanium = Material::new(
//!     4500.0,   // density (kg/m³)
//!     116.0e9,  // Young's modulus (Pa)
//!     0.32,     // Poisson's ratio
//!     0.36,     // friction coefficient
//!     0.5,      // restitution coefficient
//!     0.002,    // rolling resistance coefficient
//!     21.9,     // thermal conductivity (W/m·K)
//!     520.0,    // specific heat (J/kg·K)
//!     880.0e6,  // yield strength (Pa)
//!     950.0e6,  // ultimate strength (Pa)
//! ).expect("Valid material properties");
//! ```
//!
//! ## Limitations
//!
//! - Material properties are constant (no temperature dependence)
//! - Isotropic materials only (no anisotropic behavior)
//! - Simplified fatigue model (S-N curve approximation)
//! - Plastic deformation model is approximate
//!
//! Requires the `materials` feature flag.

#[cfg(feature = "materials")]
mod materials;
#[cfg(feature = "materials")]
pub use materials::*;

#[cfg(test)]
#[cfg(feature = "materials")]
mod materials_tests;