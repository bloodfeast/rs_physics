use crate::assert_float_eq;
use crate::thermodynamics::{calculate_efficiency, calculate_entropy_change, calculate_heat_transfer, calculate_specific_heat_capacity, calculate_work_done, Thermodynamic};

#[test]
fn test_thermodynamic_creation() {
    let state = Thermodynamic::new(300.0, 101325.0, 1.0);
    assert!(state.is_ok());
    let state = state.unwrap();
    assert_float_eq(state.temperature, 300.0, 1e-6, Some("Temperature should be 300.0 K"));
    assert_float_eq(state.pressure, 101325.0, 1e-6, Some("Pressure should be 101325.0 Pa"));
    assert_float_eq(state.volume, 1.0, 1e-6, Some("Volume should be 1.0 m³"));
}

#[test]
fn test_thermodynamic_creation_invalid_inputs() {
    assert!(Thermodynamic::new(-300.0, 101325.0, 1.0).is_err(), "Should fail with negative temperature");
    assert!(Thermodynamic::new(300.0, -101325.0, 1.0).is_err(), "Should fail with negative pressure");
    assert!(Thermodynamic::new(300.0, 101325.0, -1.0).is_err(), "Should fail with negative volume");
    assert!(Thermodynamic::new(0.0, 101325.0, 1.0).is_err(), "Should fail with zero temperature");
    assert!(Thermodynamic::new(300.0, 0.0, 1.0).is_err(), "Should fail with zero pressure");
    assert!(Thermodynamic::new(300.0, 101325.0, 0.0).is_err(), "Should fail with zero volume");
}

#[test]
fn test_calculate_heat_transfer() {
    let heat_transfer = calculate_heat_transfer(0.5, 1.0, 10.0, 0.1);
    assert!(heat_transfer.is_ok());
    assert_float_eq(heat_transfer.unwrap(), 50.0, 1e-6, Some("Heat transfer calculation"));
}

#[test]
fn test_calculate_heat_transfer_invalid_inputs() {
    assert!(calculate_heat_transfer(-0.5, 1.0, 10.0, 0.1).is_err(), "Should fail with negative thermal conductivity");
    assert!(calculate_heat_transfer(0.5, -1.0, 10.0, 0.1).is_err(), "Should fail with negative area");
    assert!(calculate_heat_transfer(0.5, 1.0, 10.0, -0.1).is_err(), "Should fail with negative thickness");
    assert!(calculate_heat_transfer(0.0, 1.0, 10.0, 0.1).is_err(), "Should fail with zero thermal conductivity");
    assert!(calculate_heat_transfer(0.5, 0.0, 10.0, 0.1).is_err(), "Should fail with zero area");
    assert!(calculate_heat_transfer(0.5, 1.0, 10.0, 0.0).is_err(), "Should fail with zero thickness");
}

#[test]
fn test_calculate_entropy_change() {
    let initial = Thermodynamic::new(300.0, 101325.0, 1.0).unwrap();
    let final_state = Thermodynamic::new(350.0, 101325.0, 1.2).unwrap();
    let entropy_change = calculate_entropy_change(&initial, &final_state, 1000.0);
    assert!(entropy_change.is_ok());
    assert_float_eq(entropy_change.unwrap(), 0.476190476190476, 1e-6, Some("Entropy change calculation"));
}

#[test]
fn test_calculate_entropy_change_invalid_inputs() {
    let valid_state = Thermodynamic::new(300.0, 101325.0, 1.0).unwrap();
    let invalid_state_result = Thermodynamic::new(0.0, 101325.0, 1.0);

    assert!(invalid_state_result.is_err(), "Should fail to create Thermodynamic with zero temperature");

    if let Ok(invalid_state) = invalid_state_result {
        assert!(calculate_entropy_change(&invalid_state, &valid_state, 1000.0).is_err(), "Should fail with zero initial temperature");
        assert!(calculate_entropy_change(&valid_state, &invalid_state, 1000.0).is_err(), "Should fail with zero final temperature");
    }

    // Test with negative temperature
    let negative_temp_result = Thermodynamic::new(-300.0, 101325.0, 1.0);
    assert!(negative_temp_result.is_err(), "Should fail to create Thermodynamic with negative temperature");
}

#[test]
fn test_calculate_work_done() {
    let initial = Thermodynamic::new(300.0, 101325.0, 1.0).unwrap();
    let final_state = Thermodynamic::new(300.0, 101325.0, 1.2).unwrap();
    let work = calculate_work_done(&initial, &final_state);
    assert!(work.is_ok());
    assert_float_eq(work.unwrap(), 20265.0, 1e-6, Some("Work done calculation"));
}

#[test]
fn test_calculate_work_done_invalid_inputs() {
    let valid_state = Thermodynamic::new(300.0, 101325.0, 1.0).unwrap();

    // Test invalid pressure
    let invalid_pressure_result = Thermodynamic::new(300.0, 0.0, 1.0);
    assert!(invalid_pressure_result.is_err(), "Should fail to create Thermodynamic with zero pressure");

    // Test invalid volume
    let invalid_volume_result = Thermodynamic::new(300.0, 101325.0, 0.0);
    assert!(invalid_volume_result.is_err(), "Should fail to create Thermodynamic with zero volume");

    // Test negative pressure
    let negative_pressure_result = Thermodynamic::new(300.0, -101325.0, 1.0);
    assert!(negative_pressure_result.is_err(), "Should fail to create Thermodynamic with negative pressure");

    // Test negative volume
    let negative_volume_result = Thermodynamic::new(300.0, 101325.0, -1.0);
    assert!(negative_volume_result.is_err(), "Should fail to create Thermodynamic with negative volume");

    // Test calculate_work_done with invalid states (if they were somehow created)
    if let Ok(invalid_pressure) = invalid_pressure_result {
        assert!(calculate_work_done(&invalid_pressure, &valid_state).is_err(), "Should fail with zero initial pressure");
        assert!(calculate_work_done(&valid_state, &invalid_pressure).is_err(), "Should fail with zero final pressure");
    }

    if let Ok(invalid_volume) = invalid_volume_result {
        assert!(calculate_work_done(&invalid_volume, &valid_state).is_err(), "Should fail with zero initial volume");
        assert!(calculate_work_done(&valid_state, &invalid_volume).is_err(), "Should fail with zero final volume");
    }
}

#[test]
fn test_calculate_efficiency() {
    let efficiency = calculate_efficiency(300.0, 1000.0);
    assert!(efficiency.is_ok());
    assert_float_eq(efficiency.unwrap(), 0.3, 1e-6, Some("Efficiency calculation"));
}

#[test]
fn test_calculate_efficiency_invalid_inputs() {
    assert!(calculate_efficiency(300.0, 0.0).is_err(), "Should fail with zero heat input");
    assert!(calculate_efficiency(300.0, -1000.0).is_err(), "Should fail with negative heat input");
}

#[test]
fn test_calculate_specific_heat_capacity() {
    let specific_heat = calculate_specific_heat_capacity(1.0, 10.0, 4180.0);
    assert!(specific_heat.is_ok());
    assert_float_eq(specific_heat.unwrap(), 418.0, 1e-6, Some("Specific heat capacity calculation"));
}

#[test]
fn test_calculate_specific_heat_capacity_invalid_inputs() {
    assert!(calculate_specific_heat_capacity(0.0, 10.0, 4180.0).is_err(), "Should fail with zero mass");
    assert!(calculate_specific_heat_capacity(-1.0, 10.0, 4180.0).is_err(), "Should fail with negative mass");
    assert!(calculate_specific_heat_capacity(1.0, 0.0, 4180.0).is_err(), "Should fail with zero temperature change");
}