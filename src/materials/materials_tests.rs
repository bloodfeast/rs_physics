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
        0.002,      // rolling_resistance_coefficient
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
        -7850.0, 200.0e9, 0.3, 0.74, 0.85, 0.002, 43.0, 490.0, 250.0e6, 400.0e6
    ).is_err());

    // Test invalid Poisson's ratio
    assert!(Material::new(
        7850.0, 200.0e9, 0.6, 0.74, 0.85, 0.002, 43.0, 490.0, 250.0e6, 400.0e6
    ).is_err());

    // Test ultimate strength less than yield strength
    assert!(Material::new(
        7850.0, 200.0e9, 0.3, 0.74, 0.85, 0.002, 43.0, 490.0, 400.0e6, 250.0e6
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
fn test_polyurethane_material() {
    let polyurethane = Material::polyurethane();

    // Check properties
    assert_float_eq(polyurethane.density, 1200.0, 1e-6, Some("Polyurethane density"));
    assert_float_eq(polyurethane.youngs_modulus, 0.02e9, 1e-6, Some("Polyurethane Young's modulus"));
    assert_float_eq(polyurethane.poisson_ratio, 0.45, 1e-6, Some("Polyurethane Poisson's ratio"));
    assert_float_eq(polyurethane.restitution_coefficient, 0.7, 1e-6, Some("Polyurethane restitution"));

    // Compare with other materials
    let rubber = Material::rubber();

    // Polyurethane should be stronger than rubber
    assert!(polyurethane.yield_strength > rubber.yield_strength);
    assert!(polyurethane.ultimate_strength > rubber.ultimate_strength);

    // But still significantly less stiff than metals
    let aluminum = Material::aluminum();
    assert!(polyurethane.youngs_modulus < aluminum.youngs_modulus);
}

#[test]
fn test_wood_material() {
    let wood = Material::wood();

    // Check properties
    assert_float_eq(wood.density, 700.0, 1e-6, Some("Wood density"));
    assert_float_eq(wood.youngs_modulus, 12.0e9, 1e-6, Some("Wood Young's modulus"));
    assert_float_eq(wood.poisson_ratio, 0.3, 1e-6, Some("Wood Poisson's ratio"));
    assert_float_eq(wood.restitution_coefficient, 0.5, 1e-6, Some("Wood restitution"));

    // Compare with other materials
    let steel = Material::steel();
    let rubber = Material::rubber();

    // Wood should be between steel and rubber in terms of stiffness
    assert!(wood.youngs_modulus < steel.youngs_modulus);
    assert!(wood.youngs_modulus > rubber.youngs_modulus);

    // Wood should be less dense than aluminum and steel
    let aluminum = Material::aluminum();
    assert!(wood.density < aluminum.density);
    assert!(wood.density < steel.density);
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

// ============================================
// NEW TESTS: Collision functions
// ============================================

#[test]
fn test_calculate_collision_response_head_on() {
    use crate::materials::calculate_collision_response;
    use std::f64::consts::PI;

    let steel = Material::steel();
    let aluminum = Material::aluminum();

    // Head-on collision (contact_angle = 0)
    let (normal_v, tangential_v) = calculate_collision_response(&steel, &aluminum, 10.0, 0.0);

    // Normal component should be reversed and reduced by restitution
    let effective_restitution = (steel.restitution_coefficient + aluminum.restitution_coefficient) / 2.0;
    assert_float_eq(normal_v, -10.0 * effective_restitution, 1e-6, Some("Normal velocity"));

    // Tangential component should be 0 for head-on (sin(0) = 0)
    assert_float_eq(tangential_v, 0.0, 1e-6, Some("Tangential velocity"));
}

#[test]
fn test_calculate_collision_response_glancing() {
    use crate::materials::calculate_collision_response;
    use std::f64::consts::PI;

    let rubber = Material::rubber();
    let steel = Material::steel();

    // Glancing collision at 45 degrees
    let (normal_v, tangential_v) = calculate_collision_response(&rubber, &steel, 10.0, PI / 4.0);

    // Normal component exists
    assert!(normal_v < 0.0, "Normal velocity should be negative (bouncing back)");

    // Tangential component may be reduced by friction
    assert!(tangential_v >= 0.0, "Tangential velocity should be non-negative");
}

#[test]
fn test_calculate_collision_response_same_material() {
    use crate::materials::calculate_collision_response;

    let steel1 = Material::steel();
    let steel2 = Material::steel();

    let (normal_v, _) = calculate_collision_response(&steel1, &steel2, 5.0, 0.0);

    // Same material: restitution is exactly the material's value
    assert_float_eq(normal_v, -5.0 * steel1.restitution_coefficient, 1e-6, None);
}

#[test]
fn test_calculate_collision_heat_generation() {
    use crate::materials::calculate_collision_heat_generation;

    let steel = Material::steel();
    let aluminum = Material::aluminum();

    let heat = calculate_collision_heat_generation(&steel, &aluminum, 10.0, 0.01);

    // Heat should be positive (energy is lost)
    assert!(heat > 0.0, "Heat generated should be positive");

    // Higher velocity should generate more heat
    let more_heat = calculate_collision_heat_generation(&steel, &aluminum, 20.0, 0.01);
    assert!(more_heat > heat, "Higher velocity should generate more heat");

    // Larger contact area should generate more heat
    let larger_area_heat = calculate_collision_heat_generation(&steel, &aluminum, 10.0, 0.02);
    assert!(larger_area_heat > heat, "Larger contact area should generate more heat");
}

#[test]
fn test_calculate_collision_heat_elastic() {
    use crate::materials::calculate_collision_heat_generation;

    // Create a perfectly elastic collision (restitution = 1.0 for both)
    let rubber = Material::rubber(); // Has high restitution (0.95)

    // Even with high restitution, some energy is lost
    let heat = calculate_collision_heat_generation(&rubber, &rubber, 10.0, 0.01);
    assert!(heat >= 0.0, "Heat should never be negative");
}

#[test]
fn test_calculate_stress_elastic() {
    use crate::materials::calculate_stress;

    let steel = Material::steel();
    let strain = 0.001;  // 0.1% strain (well within elastic region)

    let stress = calculate_stress(&steel, strain);

    // In elastic region, stress = E * strain
    let expected_stress = steel.youngs_modulus * strain;
    assert_float_eq(stress, expected_stress, 1e-6, Some("Elastic stress"));
}

#[test]
fn test_calculate_stress_plastic() {
    use crate::materials::calculate_stress;

    let steel = Material::steel();

    // Calculate strain that causes plastic deformation
    let yield_strain = steel.yield_strength / steel.youngs_modulus;
    let plastic_strain = yield_strain * 2.0;  // Well above yield

    let stress = calculate_stress(&steel, plastic_strain);

    // Stress should be above yield but below ultimate
    assert!(stress > steel.yield_strength, "Stress should exceed yield in plastic region");
    assert!(stress <= steel.ultimate_strength, "Stress should not exceed ultimate strength");
}

// ============================================
// NEW TESTS: New predefined materials
// ============================================

#[test]
fn test_copper_material() {
    let copper = Material::copper();

    assert_float_eq(copper.density, 8960.0, 1e-6, Some("Copper density"));
    assert!(copper.thermal_conductivity > 400.0, "Copper should have high thermal conductivity");
}

#[test]
fn test_titanium_material() {
    let titanium = Material::titanium();
    let steel = Material::steel();

    assert!(titanium.density < steel.density, "Titanium should be lighter than steel");
    assert!(titanium.yield_strength > steel.yield_strength, "Titanium should be stronger than steel");
}

#[test]
fn test_concrete_material() {
    let concrete = Material::concrete();

    assert_float_eq(concrete.density, 2400.0, 1e-6, Some("Concrete density"));
    assert!(concrete.restitution_coefficient < 0.5, "Concrete has low restitution");
}

#[test]
fn test_glass_material() {
    let glass = Material::glass();

    // Glass is brittle - yield equals ultimate
    assert_float_eq(glass.yield_strength, glass.ultimate_strength, 1e-6,
        Some("Glass yield should equal ultimate (brittle)"));
}

#[test]
fn test_brass_material() {
    let brass = Material::brass();
    let copper = Material::copper();

    assert!(brass.thermal_conductivity < copper.thermal_conductivity,
        "Brass should have lower thermal conductivity than pure copper");
}

#[test]
fn test_ice_material() {
    let ice = Material::ice();

    assert!(ice.friction_coefficient < 0.1, "Ice should have very low friction");
    assert!(ice.density < 1000.0, "Ice should be less dense than water");
}

#[test]
fn test_stainless_steel_material() {
    let stainless = Material::stainless_steel();
    let carbon = Material::steel();

    assert!(stainless.thermal_conductivity < carbon.thermal_conductivity,
        "Stainless steel should have lower thermal conductivity than carbon steel");
}

// ============================================
// NEW TESTS: Thermal diffusivity
// ============================================

#[test]
fn test_thermal_diffusivity() {
    let copper = Material::copper();
    let aluminum = Material::aluminum();
    let rubber = Material::rubber();

    // Calculate expected thermal diffusivities
    let copper_alpha = copper.thermal_conductivity / (copper.density * copper.specific_heat_capacity);
    let aluminum_alpha = aluminum.thermal_conductivity / (aluminum.density * aluminum.specific_heat_capacity);

    assert_float_eq(copper.thermal_diffusivity(), copper_alpha, 1e-12, Some("Copper thermal diffusivity"));
    assert_float_eq(aluminum.thermal_diffusivity(), aluminum_alpha, 1e-12, Some("Aluminum thermal diffusivity"));

    // Copper should have higher diffusivity than rubber
    assert!(copper.thermal_diffusivity() > rubber.thermal_diffusivity(),
        "Copper should have higher thermal diffusivity than rubber");
}

// ============================================
// NEW TESTS: Default and PartialEq
// ============================================

#[test]
fn test_material_default() {
    let default_material = Material::default();
    let steel = Material::steel();

    // Default should be steel
    assert_eq!(default_material.density, steel.density);
    assert_eq!(default_material.youngs_modulus, steel.youngs_modulus);
}

#[test]
fn test_material_partial_eq() {
    let steel1 = Material::steel();
    let steel2 = Material::steel();
    let aluminum = Material::aluminum();

    assert_eq!(steel1, steel2, "Same materials should be equal");
    assert_ne!(steel1, aluminum, "Different materials should not be equal");
}

// ============================================
// NEW TESTS: MaterialBuilder
// ============================================

#[test]
fn test_material_builder_basic() {
    use crate::materials::MaterialBuilder;

    let material = MaterialBuilder::new()
        .density(5000.0)
        .youngs_modulus(150e9)
        .build()
        .unwrap();

    assert_float_eq(material.density, 5000.0, 1e-6, Some("Builder density"));
    assert_float_eq(material.youngs_modulus, 150e9, 1e-6, Some("Builder Young's modulus"));
}

#[test]
fn test_material_builder_from_existing() {
    use crate::materials::MaterialBuilder;

    let modified = MaterialBuilder::from(Material::aluminum())
        .friction_coefficient(0.9)
        .build()
        .unwrap();

    // Should keep aluminum's density
    assert_float_eq(modified.density, 2700.0, 1e-6, Some("Preserved density"));
    // But have modified friction
    assert_float_eq(modified.friction_coefficient, 0.9, 1e-6, Some("Modified friction"));
}

#[test]
fn test_material_builder_validation() {
    use crate::materials::MaterialBuilder;

    // Invalid density should fail
    let result = MaterialBuilder::new()
        .density(-100.0)
        .build();
    assert!(result.is_err(), "Negative density should fail");

    // Invalid Poisson's ratio should fail
    let result = MaterialBuilder::new()
        .poisson_ratio(0.6)
        .build();
    assert!(result.is_err(), "Poisson ratio > 0.5 should fail");

    // Invalid restitution should fail
    let result = MaterialBuilder::new()
        .restitution_coefficient(1.5)
        .build();
    assert!(result.is_err(), "Restitution > 1 should fail");
}

#[test]
fn test_material_builder_default() {
    use crate::materials::MaterialBuilder;

    let builder1 = MaterialBuilder::new();
    let builder2 = MaterialBuilder::default();
    let material1 = builder1.build().unwrap();
    let material2 = builder2.build().unwrap();

    // Both should produce equivalent materials
    assert_eq!(material1, material2);
}

// ============================================
// NEW TESTS: Fatigue strength (now public)
// ============================================

#[test]
fn test_calculate_fatigue_strength_low_cycle() {
    let steel = Material::steel();

    // Low cycle fatigue (< 1000) should return ultimate strength
    let strength = steel.calculate_fatigue_strength(500);
    assert_float_eq(strength, steel.ultimate_strength, 1e-6,
        Some("Low cycle fatigue should use ultimate strength"));
}

#[test]
fn test_calculate_fatigue_strength_high_cycle() {
    let steel = Material::steel();

    // High cycle fatigue (> 1,000,000) should return endurance limit
    let strength = steel.calculate_fatigue_strength(2_000_000);
    let endurance_limit = steel.yield_strength * 0.5;
    assert_float_eq(strength, endurance_limit, 1e-6,
        Some("High cycle fatigue should use endurance limit"));
}

#[test]
fn test_calculate_fatigue_strength_transition() {
    let steel = Material::steel();

    // Transition region should be between ultimate and endurance limit
    let strength = steel.calculate_fatigue_strength(100_000);
    let endurance_limit = steel.yield_strength * 0.5;

    assert!(strength < steel.ultimate_strength,
        "Transition strength should be less than ultimate");
    assert!(strength > endurance_limit,
        "Transition strength should be greater than endurance limit");
}