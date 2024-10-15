// src/fluid_dynamics_tests.rs

use crate::assert_float_eq;
use crate::fluid_dynamics::{calculate_buoyant_force, calculate_drag_force, calculate_pressure_drop, calculate_reynolds_number, Fluid};

#[test]
fn test_fluid_creation() {
    let fluid = Fluid::new(1000.0, 0.001);
    assert!(fluid.is_ok());
    let fluid = fluid.unwrap();
    assert_float_eq(fluid.density, 1000.0, 1e-6, Some("Density should be 1000.0"));
    assert_float_eq(fluid.viscosity, 0.001, 1e-6, Some("Viscosity should be 0.001"));
}

#[test]
fn test_fluid_creation_invalid_inputs() {
    assert!(Fluid::new(-1000.0, 0.001).is_err(), "Should fail with negative density");
    assert!(Fluid::new(0.0, 0.001).is_err(), "Should fail with zero density");
    assert!(Fluid::new(1000.0, -0.001).is_err(), "Should fail with negative viscosity");
    assert!(Fluid::new(1000.0, 0.0).is_err(), "Should fail with zero viscosity");
}

#[test]
fn test_calculate_reynolds_number() {
    let fluid = Fluid::new(1000.0, 0.001).unwrap();
    let reynolds_number = calculate_reynolds_number(&fluid, 1.0, 0.1);
    assert!(reynolds_number.is_ok());
    assert_float_eq(reynolds_number.unwrap(), 100000.0, 1e-6, Some("Reynolds number calculation"));
}

#[test]
fn test_calculate_reynolds_number_invalid_inputs() {
    let fluid = Fluid::new(1000.0, 0.001).unwrap();
    assert!(calculate_reynolds_number(&fluid, 0.0, 0.1).is_err(), "Should fail with zero velocity");
    assert!(calculate_reynolds_number(&fluid, -1.0, 0.1).is_err(), "Should fail with negative velocity");
    assert!(calculate_reynolds_number(&fluid, 1.0, 0.0).is_err(), "Should fail with zero characteristic length");
    assert!(calculate_reynolds_number(&fluid, 1.0, -0.1).is_err(), "Should fail with negative characteristic length");
}

#[test]
fn test_calculate_drag_force() {
    let fluid = Fluid::new(1.225, 0.001).unwrap(); // Air at sea level
    let drag_force = calculate_drag_force(&fluid, 10.0, 1.0, 0.5);
    assert!(drag_force.is_ok());
    assert_float_eq(drag_force.unwrap(), 30.625, 1e-6, Some("Drag force calculation"));
}

#[test]
fn test_calculate_drag_force_invalid_inputs() {
    let fluid = Fluid::new(1.225, 0.001).unwrap();
    assert!(calculate_drag_force(&fluid, 0.0, 1.0, 0.5).is_err(), "Should fail with zero velocity");
    assert!(calculate_drag_force(&fluid, -10.0, 1.0, 0.5).is_err(), "Should fail with negative velocity");
    assert!(calculate_drag_force(&fluid, 10.0, 0.0, 0.5).is_err(), "Should fail with zero area");
    assert!(calculate_drag_force(&fluid, 10.0, -1.0, 0.5).is_err(), "Should fail with negative area");
    assert!(calculate_drag_force(&fluid, 10.0, 1.0, 0.0).is_err(), "Should fail with zero drag coefficient");
    assert!(calculate_drag_force(&fluid, 10.0, 1.0, -0.5).is_err(), "Should fail with negative drag coefficient");
}

#[test]
fn test_calculate_buoyant_force() {
    let fluid = Fluid::new(1000.0, 0.001).unwrap(); // Water
    let buoyant_force = calculate_buoyant_force(&fluid, 0.1, 9.81);
    assert!(buoyant_force.is_ok());
    assert_float_eq(buoyant_force.unwrap(), 981.0, 1e-6, Some("Buoyant force calculation"));
}

#[test]
fn test_calculate_buoyant_force_invalid_inputs() {
    let fluid = Fluid::new(1000.0, 0.001).unwrap();
    assert!(calculate_buoyant_force(&fluid, 0.0, 9.81).is_err(), "Should fail with zero displaced volume");
    assert!(calculate_buoyant_force(&fluid, -0.1, 9.81).is_err(), "Should fail with negative displaced volume");
    assert!(calculate_buoyant_force(&fluid, 0.1, 0.0).is_err(), "Should fail with zero gravity");
    assert!(calculate_buoyant_force(&fluid, 0.1, -9.81).is_err(), "Should fail with negative gravity");
}

#[test]
fn test_calculate_pressure_drop() {
    let fluid = Fluid::new(1000.0, 0.001).unwrap(); // Water
    let pressure_drop = calculate_pressure_drop(&fluid, 10.0, 0.05, 2.0, 0.02);
    assert!(pressure_drop.is_ok());
    let expected_pressure_drop = 0.02 * (10.0 / 0.05) * 0.5 * 1000.0 * 2.0 * 2.0;
    assert_float_eq(pressure_drop.unwrap(), expected_pressure_drop, 1e-6, Some("Pressure drop calculation"));
}
#[test]
fn test_calculate_pressure_drop_invalid_inputs() {
    let fluid = Fluid::new(1000.0, 0.001).unwrap();
    assert!(calculate_pressure_drop(&fluid, 0.0, 0.05, 2.0, 0.02).is_err(), "Should fail with zero pipe length");
    assert!(calculate_pressure_drop(&fluid, -10.0, 0.05, 2.0, 0.02).is_err(), "Should fail with negative pipe length");
    assert!(calculate_pressure_drop(&fluid, 10.0, 0.0, 2.0, 0.02).is_err(), "Should fail with zero pipe diameter");
    assert!(calculate_pressure_drop(&fluid, 10.0, -0.05, 2.0, 0.02).is_err(), "Should fail with negative pipe diameter");
    assert!(calculate_pressure_drop(&fluid, 10.0, 0.05, 0.0, 0.02).is_err(), "Should fail with zero velocity");
    assert!(calculate_pressure_drop(&fluid, 10.0, 0.05, -2.0, 0.02).is_err(), "Should fail with negative velocity");
    assert!(calculate_pressure_drop(&fluid, 10.0, 0.05, 2.0, 0.0).is_err(), "Should fail with zero friction factor");
    assert!(calculate_pressure_drop(&fluid, 10.0, 0.05, 2.0, -0.02).is_err(), "Should fail with negative friction factor");
}