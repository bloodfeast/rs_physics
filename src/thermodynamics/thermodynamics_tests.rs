use crate::assert_float_eq;
use crate::thermodynamics::{
    calculate_efficiency, calculate_entropy_change, calculate_heat_transfer,
    calculate_specific_heat_capacity, calculate_work_done, Thermodynamic,
    // New imports for ideal gas and related functions
    GasType, ideal_gas_pressure, ideal_gas_volume, ideal_gas_temperature, ideal_gas_moles,
    internal_energy, internal_energy_change, enthalpy, enthalpy_change, enthalpy_from_state,
    molar_heat_capacity_cv, molar_heat_capacity_cp, heat_capacity_ratio, R,
};

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

// ============================================================================
// Ideal Gas Law Tests
// ============================================================================

#[test]
fn test_ideal_gas_pressure() {
    // P = nRT/V
    // 1 mol at 300 K in 0.0249 m³ should give ≈ 100 kPa
    let p = ideal_gas_pressure(1.0, 300.0, 0.0249).unwrap();
    assert!((p - 100116.0).abs() < 100.0, "Pressure should be approximately 100 kPa, got {}", p);

    // At STP: 1 mol, 273.15 K, ~22.4 L = 0.0224 m³ → 101325 Pa
    let p_stp = ideal_gas_pressure(1.0, 273.15, 0.0224).unwrap();
    assert!((p_stp - 101325.0).abs() < 1000.0, "Pressure at STP should be near 101325 Pa");
}

#[test]
fn test_ideal_gas_volume() {
    // V = nRT/P
    // 1 mol at 300 K and 100000 Pa
    let v = ideal_gas_volume(1.0, 300.0, 100000.0).unwrap();
    let expected = R * 300.0 / 100000.0;
    assert_float_eq(v, expected, 1e-6, Some("Volume calculation"));
}

#[test]
fn test_ideal_gas_temperature() {
    // T = PV/(nR)
    let t = ideal_gas_temperature(101325.0, 0.0224, 1.0).unwrap();
    // Should be approximately 273 K
    assert!((t - 273.0).abs() < 5.0, "Temperature should be near 273 K, got {}", t);
}

#[test]
fn test_ideal_gas_moles() {
    // n = PV/(RT)
    let n = ideal_gas_moles(101325.0, 0.0224, 273.15).unwrap();
    assert!((n - 1.0).abs() < 0.02, "Should be approximately 1 mol, got {}", n);
}

#[test]
fn test_ideal_gas_law_roundtrip() {
    // Start with known values and verify roundtrip
    let n = 2.5;
    let t = 350.0;
    let v = 0.05;

    let p = ideal_gas_pressure(n, t, v).unwrap();
    let n_calc = ideal_gas_moles(p, v, t).unwrap();
    assert!((n_calc - n).abs() < 1e-10, "Roundtrip moles: expected {}, got {}", n, n_calc);

    let v_calc = ideal_gas_volume(n, t, p).unwrap();
    assert!((v_calc - v).abs() < 1e-10, "Roundtrip volume: expected {}, got {}", v, v_calc);

    let t_calc = ideal_gas_temperature(p, v, n).unwrap();
    assert!((t_calc - t).abs() < 1e-10, "Roundtrip temperature: expected {}, got {}", t, t_calc);
}

#[test]
fn test_ideal_gas_invalid_inputs() {
    // Negative moles
    assert!(ideal_gas_pressure(-1.0, 300.0, 0.025).is_err());

    // Zero temperature
    assert!(ideal_gas_pressure(1.0, 0.0, 0.025).is_err());

    // Zero volume
    assert!(ideal_gas_pressure(1.0, 300.0, 0.0).is_err());

    // Zero pressure
    assert!(ideal_gas_volume(1.0, 300.0, 0.0).is_err());

    // Zero moles for temperature
    assert!(ideal_gas_temperature(101325.0, 0.025, 0.0).is_err());
}

// ============================================================================
// Internal Energy Tests
// ============================================================================

#[test]
fn test_internal_energy_monatomic() {
    // U = (3/2) * n * R * T for monatomic gas
    let u = internal_energy(1.0, 300.0, GasType::Monatomic).unwrap();
    let expected = 1.5 * R * 300.0;
    assert_float_eq(u, expected, 1e-6, Some("Monatomic internal energy"));
}

#[test]
fn test_internal_energy_diatomic() {
    // U = (5/2) * n * R * T for diatomic gas
    let u = internal_energy(1.0, 300.0, GasType::Diatomic).unwrap();
    let expected = 2.5 * R * 300.0;
    assert_float_eq(u, expected, 1e-6, Some("Diatomic internal energy"));
}

#[test]
fn test_internal_energy_change() {
    // ΔU = n * Cv * ΔT
    let delta_u = internal_energy_change(1.0, 100.0, GasType::Monatomic).unwrap();
    let expected = 1.5 * R * 100.0;
    assert_float_eq(delta_u, expected, 1e-6, Some("Internal energy change"));

    // Cooling (negative ΔT)
    let delta_u_cool = internal_energy_change(1.0, -50.0, GasType::Monatomic).unwrap();
    assert!(delta_u_cool < 0.0, "Cooling should decrease internal energy");
}

// ============================================================================
// Enthalpy Tests
// ============================================================================

#[test]
fn test_enthalpy() {
    // H = n * Cp * T for ideal gas
    let h = enthalpy(1.0, 300.0, GasType::Monatomic).unwrap();
    let expected = 2.5 * R * 300.0; // Cp = (5/2)R for monatomic
    assert_float_eq(h, expected, 1e-6, Some("Monatomic enthalpy"));
}

#[test]
fn test_enthalpy_change() {
    // ΔH = n * Cp * ΔT
    let delta_h = enthalpy_change(1.0, 100.0, GasType::Diatomic).unwrap();
    let expected = 3.5 * R * 100.0; // Cp = (7/2)R for diatomic
    assert_float_eq(delta_h, expected, 1e-6, Some("Enthalpy change"));
}

#[test]
fn test_enthalpy_from_state() {
    let state = Thermodynamic::new(300.0, 101325.0, 0.025).unwrap();
    let h = enthalpy_from_state(&state, 1.0, GasType::Monatomic).unwrap();

    // H = U + PV
    let u = internal_energy(1.0, 300.0, GasType::Monatomic).unwrap();
    let expected = u + 101325.0 * 0.025;

    assert_float_eq(h, expected, 1e-6, Some("Enthalpy from state"));
}

// ============================================================================
// Heat Capacity Tests
// ============================================================================

#[test]
fn test_cp_cv_relationship() {
    // Cp - Cv = R for ideal gases
    let cv_mono = molar_heat_capacity_cv(GasType::Monatomic);
    let cp_mono = molar_heat_capacity_cp(GasType::Monatomic);
    assert_float_eq(cp_mono - cv_mono, R, 1e-10, Some("Cp - Cv = R for monatomic"));

    let cv_di = molar_heat_capacity_cv(GasType::Diatomic);
    let cp_di = molar_heat_capacity_cp(GasType::Diatomic);
    assert_float_eq(cp_di - cv_di, R, 1e-10, Some("Cp - Cv = R for diatomic"));
}

#[test]
fn test_gamma_ratio() {
    // γ = Cp/Cv
    let gamma_mono = heat_capacity_ratio(GasType::Monatomic);
    assert_float_eq(gamma_mono, 5.0 / 3.0, 1e-10, Some("Gamma for monatomic"));

    let gamma_di = heat_capacity_ratio(GasType::Diatomic);
    assert_float_eq(gamma_di, 7.0 / 5.0, 1e-10, Some("Gamma for diatomic"));
}

#[test]
fn test_gas_type_degrees_of_freedom() {
    assert_eq!(GasType::Monatomic.degrees_of_freedom(), 3);
    assert_eq!(GasType::Diatomic.degrees_of_freedom(), 5);
    assert_eq!(GasType::Polyatomic(6).degrees_of_freedom(), 6);
}

#[test]
fn test_polyatomic_gas() {
    // CO2-like with 6 degrees of freedom
    let co2 = GasType::Polyatomic(6);

    let cv = molar_heat_capacity_cv(co2);
    assert_float_eq(cv, 3.0 * R, 1e-10, Some("Cv for 6 DOF"));

    let cp = molar_heat_capacity_cp(co2);
    assert_float_eq(cp, 4.0 * R, 1e-10, Some("Cp for 6 DOF"));

    let gamma = heat_capacity_ratio(co2);
    assert_float_eq(gamma, 8.0 / 6.0, 1e-10, Some("Gamma for 6 DOF"));
}