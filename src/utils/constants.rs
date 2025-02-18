use crate::utils;

pub const DEFAULT_PHYSICS_CONSTANTS: utils::PhysicsConstants = utils::PhysicsConstants {
    gravity: 9.80665,
    air_density: 1.225,
    speed_of_sound: 343.0,
    atmospheric_pressure: 101_325.0,
};