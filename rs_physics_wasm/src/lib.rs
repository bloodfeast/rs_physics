// rs_physics_wasm/src/lib.rs
// This is the public API for the wasm wrapper of the physics library.

use wasm_bindgen::prelude::*;
use rs_physics::apis::easy_physics::EasyPhysics;
use rs_physics::interactions::Object;

#[wasm_bindgen]
pub struct WasmPhysics {
    physics: EasyPhysics,
}

#[wasm_bindgen]
impl WasmPhysics {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            physics: EasyPhysics::new(),
        }
    }

    #[wasm_bindgen]
    pub fn create_object(&self, mass: f64, velocity: f64, position: f64) -> Result<WasmObject, JsValue> {
        self.physics.create_object(mass, velocity, position)
            .map(WasmObject)
            .map_err(|e| JsValue::from_str(e))
    }

    #[wasm_bindgen]
    pub fn simulate_collision(&self, obj1: &mut WasmObject, obj2: &mut WasmObject, angle: f64, duration: f64, drag_coefficient: f64, cross_sectional_area: f64) -> Result<(), JsValue> {
        self.physics.simulate_collision(&mut obj1.0, &mut obj2.0, angle, duration, drag_coefficient, cross_sectional_area)
            .map_err(|e| JsValue::from_str(e))
    }

    #[wasm_bindgen]
    pub fn calculate_gravity_force(&self, obj1: &WasmObject, obj2: &WasmObject) -> Result<f64, JsValue> {
        self.physics.calculate_gravity_force(&obj1.0, &obj2.0)
            .map_err(|e| JsValue::from_str(e))
    }

    #[wasm_bindgen]
    pub fn apply_force_to_object(&self, obj: &mut WasmObject, force: f64, time: f64) -> Result<(), JsValue> {
        self.physics.apply_force_to_object(&mut obj.0, force, time)
            .map_err(|e| JsValue::from_str(e))
    }

    #[wasm_bindgen]
    pub fn calculate_kinetic_energy(&self, obj: &WasmObject) -> f64 {
        self.physics.calculate_kinetic_energy(&obj.0)
    }

    #[wasm_bindgen]
    pub fn calculate_momentum(&self, obj: &WasmObject) -> f64 {
        self.physics.calculate_momentum(&obj.0)
    }
}

#[wasm_bindgen]
pub struct WasmObject(Object);

#[wasm_bindgen]
impl WasmObject {
    #[wasm_bindgen(getter)]
    pub fn mass(&self) -> f64 {
        self.0.mass
    }

    #[wasm_bindgen(getter)]
    pub fn velocity(&self) -> f64 {
        self.0.velocity
    }

    #[wasm_bindgen(getter)]
    pub fn position(&self) -> f64 {
        self.0.position
    }
}