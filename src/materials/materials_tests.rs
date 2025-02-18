// src/materials_tests.rs

use crate::materials::{Material, BreakageType};
use crate::assert_float_eq;

/// Helper function to create test materials
fn create_test_materials() -> (Material, Material, Material) {
    (Material::steel(), Material::aluminum(), Material::rubber())
}

#[test]
fn test_material_creation() {
    let result = Material::new(
        7850.0,     // density
        200.0e9,    // youngs_modulus
        0.3,        // poisson_ratio
        0.74,       // friction_coefficient
        0.85,       // restitution_coefficient
        43.0,       // thermal_conductivity
        490.0,      // specific_heat_capacity
        250.0e6,    // yield_strength
        400.0e6,    // ultimate_strength
    );

    assert!(result.is_ok());
    let material = result.unwrap();
    assert_float_eq(material.density, 7850.0, 1e-6, Some("Density check"));
    assert_float_eq(material.youngs_modulus, 200.0e9, 1e-6, Some("Young's modulus check"));
}

#[test]
fn test_invalid_material_creation() {
    // Test negative density
    assert!(Material::new(
        -7850.0, 200.0e9, 0.3, 0.74, 0.85, 43.0, 490.0, 250.0e6, 400.0e6
    ).is_err());

    // Test invalid Poisson's ratio
    assert!(Material::new(
        7850.0, 200.0e9, 0.6, 0.74, 0.85, 43.0, 490.0, 250.0e6, 400.0e6
    ).is_err());

    // Test ultimate strength less than yield strength
    assert!(Material::new(
        7850.0, 200.0e9, 0.3, 0.74, 0.85, 43.0, 490.0, 400.0e6, 250.0e6
    ).is_err());
}

#[test]
fn test_predefined_materials() {
    let steel = Material::steel();
    let aluminum = Material::aluminum();
    let rubber = Material::rubber();

    // Test steel properties
    assert_float_eq(steel.density, 7850.0, 1e-6, Some("Steel density"));
    assert_float_eq(steel.youngs_modulus, 200.0e9, 1e-6, Some("Steel Young's modulus"));

    // Test aluminum properties
    assert_float_eq(aluminum.density, 2700.0, 1e-6, Some("Aluminum density"));
    assert_float_eq(aluminum.youngs_modulus, 69.0e9, 1e-6, Some("Aluminum Young's modulus"));

    // Test rubber properties
    assert_float_eq(rubber.density, 1100.0, 1e-6, Some("Rubber density"));
    assert_float_eq(rubber.youngs_modulus, 0.01e9, 1e-6, Some("Rubber Young's modulus"));
}

#[test]
fn test_elastic_region() {
    let steel = Material::steel();
    let yield_strain = steel.yield_strength / steel.youngs_modulus;

    // Test well within elastic region
    let result = steel.will_break(200e6, yield_strain * 0.5, None);
    assert_eq!(result.will_break, false);
    assert_eq!(result.breakage_type, BreakageType::None);
    assert!(result.safety_factor > 1.0);
}

#[test]
fn test_plastic_region() {
    let steel = Material::steel();
    let yield_strain = steel.yield_strength / steel.youngs_modulus;

    // Test just above yield point
    let result = steel.will_break(steel.yield_strength * 1.1, yield_strain * 1.1, None);
    assert_eq!(result.will_break, false);
    assert_eq!(result.breakage_type, BreakageType::Plastic);
    assert!(result.safety_factor <= 1.0);
}

#[test]
fn test_ultimate_failure() {
    let steel = Material::steel();

    // Test stress-based failure
    let stress_result = steel.will_break(steel.ultimate_strength * 1.1, 0.001, None);
    assert_eq!(stress_result.will_break, true);
    assert_eq!(stress_result.breakage_type, BreakageType::TensileStress);

    // Test strain-based failure
    let ultimate_strain = steel.ultimate_strength / steel.youngs_modulus;
    let strain_result = steel.will_break(steel.yield_strength, ultimate_strain * 1.1, None);
    assert_eq!(strain_result.will_break, true);
    assert_eq!(strain_result.breakage_type, BreakageType::TensileStrain);
}

#[test]
fn test_material_moduli() {
    let steel = Material::steel();

    // Calculate expected shear modulus
    let expected_shear = steel.youngs_modulus / (2.0 * (1.0 + steel.poisson_ratio));
    assert_float_eq(steel.shear_modulus(), expected_shear, 1e-6, Some("Shear modulus"));

    // Calculate expected bulk modulus
    let expected_bulk = steel.youngs_modulus / (3.0 * (1.0 - 2.0 * steel.poisson_ratio));
    assert_float_eq(steel.bulk_modulus(), expected_bulk, 1e-6, Some("Bulk modulus"));
}

#[test]
fn test_strain_energy() {
    let steel = Material::steel();
    let strain = 0.001;

    // Calculate expected strain energy density
    let expected_energy = 0.5 * steel.youngs_modulus * strain * strain;
    assert_float_eq(steel.strain_energy_density(strain), expected_energy, 1e-6, Some("Strain energy"));
}

#[test]
fn test_cyclic_loading() {
    let steel = Material::steel();

    // Test low-cycle fatigue
    let low_cycle = steel.will_break(steel.yield_strength * 0.9, 0.001, Some(100));

    // Test high-cycle fatigue
    let high_cycle = steel.will_break(steel.yield_strength * 0.5, 0.001, Some(1_000_000));

    // High-cycle should be more likely to fail than low-cycle at lower stress
    assert!(high_cycle.safety_factor <= low_cycle.safety_factor);
}

#[test]
fn test_safety_factors() {
    let steel = Material::steel();

    // Test elastic region safety factor
    let elastic = steel.will_break(steel.yield_strength * 0.5, 0.001, None);
    assert!(elastic.safety_factor > 1.0);

    // Test plastic region safety factor
    let plastic = steel.will_break(steel.yield_strength * 1.1, 0.001, None);
    assert!(plastic.safety_factor < 1.0);

    // Test ultimate region safety factor
    let ultimate = steel.will_break(steel.ultimate_strength * 1.1, 0.001, None);
    assert!(ultimate.safety_factor < 1.0);
}

#[test]
fn test_material_comparison() {
    let (steel, aluminum, rubber) = create_test_materials();

    // Compare yield strengths
    assert!(steel.yield_strength > aluminum.yield_strength);

    // Compare Young's moduli
    assert!(steel.youngs_modulus > rubber.youngs_modulus);

    // Compare densities
    assert!(aluminum.density < steel.density);
}

#[test]
fn test_edge_cases() {
    let steel = Material::steel();

    // Test exactly at yield point
    let at_yield = steel.will_break(steel.yield_strength, steel.yield_strength / steel.youngs_modulus, None);
    assert_eq!(at_yield.breakage_type, BreakageType::Plastic);

    // Test exactly at ultimate strength
    let at_ultimate = steel.will_break(steel.ultimate_strength, steel.ultimate_strength / steel.youngs_modulus, None);
    assert_eq!(at_ultimate.breakage_type, BreakageType::TensileStress);
}