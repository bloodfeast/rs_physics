// src/constants_config.rs
use std::f64::consts::PI;
use crate::utils::{
    DEFAULT_PHYSICS_CONSTANTS,
    errors::PhysicsError
};

#[derive(Debug, Clone, Copy)]
pub struct PhysicsConstants {
    pub gravity: f64,
    pub air_density: f64,
    pub speed_of_sound: f64,
    pub atmospheric_pressure: f64,
    pub ground_level: f64,
}


impl Default for PhysicsConstants {
    fn default() -> Self {
        Self {
            gravity: 9.80665,
            air_density: 1.225,
            speed_of_sound: 343.0,
            atmospheric_pressure: 101_325.0,
            ground_level: 0.0,
        }
    }
}

impl PhysicsConstants {
    pub fn new(
        gravity: Option<f64>,
        air_density: Option<f64>,
        speed_of_sound: Option<f64>,
        atmospheric_pressure: Option<f64>,
        ground_level: Option<f64>,
    ) -> Self {
        let default = DEFAULT_PHYSICS_CONSTANTS;
        Self {
            gravity: gravity.unwrap_or(default.gravity),
            air_density: air_density.unwrap_or(default.air_density),
            speed_of_sound: speed_of_sound.unwrap_or(default.speed_of_sound),
            atmospheric_pressure: atmospheric_pressure.unwrap_or(default.atmospheric_pressure),
            ground_level: ground_level.unwrap_or(default.ground_level),
        }
    }

    pub fn calculate_terminal_velocity(&self, mass: f64, drag_coefficient: f64, cross_sectional_area: f64) -> Result<f64, PhysicsError> {
        if mass <= 0.0 { return Err(PhysicsError::InvalidMass); }
        if drag_coefficient <= 0.0 { return Err(PhysicsError::InvalidCoefficient); }
        if cross_sectional_area <= 0.0 { return Err(PhysicsError::InvalidArea); }
        if self.air_density <= 0.0 { return Err(PhysicsError::CalculationError("Air density must be positive".to_string())); }

        Ok(((2.0 * mass * self.gravity) / (self.air_density * drag_coefficient * cross_sectional_area)).sqrt())
    }

    pub fn calculate_air_resistance(&self, velocity: f64, drag_coefficient: f64, cross_sectional_area: f64) -> Result<f64, PhysicsError> {
        if drag_coefficient < 0.0 { return Err(PhysicsError::InvalidCoefficient); }
        if cross_sectional_area < 0.0 { return Err(PhysicsError::InvalidArea); }
        if self.air_density < 0.0 { return Err(PhysicsError::CalculationError("Air density must be non-negative".to_string())); }

        Ok(0.5 * self.air_density * velocity * velocity * drag_coefficient * cross_sectional_area)
    }

    pub fn calculate_acceleration(&self, force: f64, mass: f64) -> Result<f64, PhysicsError> {
        if mass == 0.0 { return Err(PhysicsError::DivisionByZero); }
        if mass < 0.0 { return Err(PhysicsError::InvalidMass); }
        Ok(force / mass)
    }

    pub fn calculate_deceleration(&self, force: f64, mass: f64) -> Result<f64, PhysicsError> {
        self.calculate_acceleration(force, mass).map(|a| -a)
    }

    pub fn calculate_force(&self, mass: f64, acceleration: f64) -> Result<f64, PhysicsError> {
        if mass < 0.0 { return Err(PhysicsError::InvalidMass); }
        Ok(mass * acceleration)
    }

    pub fn calculate_momentum(&self, mass: f64, velocity: f64) -> Result<f64, PhysicsError> {
        if mass < 0.0 { return Err(PhysicsError::InvalidMass); }
        Ok(mass * velocity)
    }

    pub fn calculate_velocity(&self, initial_velocity: f64, acceleration: f64, time: f64) -> Result<f64, PhysicsError> {
        if time < 0.0 { return Err(PhysicsError::InvalidTime); }
        Ok(initial_velocity + acceleration * time)
    }

    pub fn calculate_average_velocity(&self, displacement: f64, time: f64) -> Result<f64, PhysicsError> {
        if time == 0.0 { return Err(PhysicsError::DivisionByZero); }
        Ok(displacement / time)
    }

    pub fn calculate_kinetic_energy(&self, mass: f64, velocity: f64) -> Result<f64, PhysicsError> {
        if mass < 0.0 { return Err(PhysicsError::InvalidMass); }
        Ok(0.5 * mass * velocity * velocity)
    }

    pub fn calculate_potential_energy(&self, mass: f64, height: f64) -> Result<f64, PhysicsError> {
        if mass < 0.0 { return Err(PhysicsError::InvalidMass); }
        Ok(mass * self.gravity * height)
    }

    pub fn calculate_work(&self, force: f64, displacement: f64) -> Result<f64, PhysicsError> {
        Ok(force * displacement)
    }

    pub fn calculate_power(&self, work: f64, time: f64) -> Result<f64, PhysicsError> {
        if time == 0.0 { return Err(PhysicsError::DivisionByZero); }
        Ok(work / time)
    }

    pub fn calculate_impulse(&self, force: f64, time: f64) -> Result<f64, PhysicsError> {
        if time < 0.0 { return Err(PhysicsError::InvalidTime); }
        Ok(force * time)
    }

    pub fn calculate_coefficient_of_restitution(&self, velocity_before: f64, velocity_after: f64) -> Result<f64, PhysicsError> {
        if velocity_before == 0.0 { return Err(PhysicsError::DivisionByZero); }
        Ok(-velocity_after / velocity_before)
    }

    pub fn calculate_projectile_time_of_flight(&self, initial_velocity: f64, angle: f64) -> Result<f64, PhysicsError> {
        if initial_velocity < 0.0 { return Err(PhysicsError::InvalidVelocity); }
        let normalized_angle = angle % (2.0 * PI);
        if normalized_angle < 0.0 || normalized_angle > PI / 2.0 { return Err(PhysicsError::InvalidAngle); }
        let v_y = initial_velocity * normalized_angle.sin();
        Ok(2.0 * v_y / self.gravity)
    }

    pub fn calculate_projectile_max_height(&self, initial_velocity: f64, angle: f64) -> Result<f64, PhysicsError> {
        if initial_velocity < 0.0 { return Err(PhysicsError::InvalidVelocity); }
        let normalized_angle = angle % (2.0 * PI);
        if normalized_angle < 0.0 || normalized_angle > PI / 2.0 { return Err(PhysicsError::InvalidAngle); }
        let v_y = initial_velocity * normalized_angle.sin();
        Ok((v_y * v_y) / (2.0 * self.gravity))
    }

    pub fn calculate_centripetal_force(&self, mass: f64, velocity: f64, radius: f64) -> Result<f64, PhysicsError> {
        if mass < 0.0 { return Err(PhysicsError::InvalidMass); }
        if radius == 0.0 { return Err(PhysicsError::DivisionByZero); }
        Ok(mass * velocity * velocity / radius)
    }

    pub fn calculate_torque(&self, force: f64, lever_arm: f64, angle: f64) -> Result<f64, PhysicsError> {
        if lever_arm < 0.0 {
            return Err(PhysicsError::InvalidDistance);
        }
        if !(0.0..=PI).contains(&angle) {
            return Err(PhysicsError::InvalidAngle);
        }
        // TODO: Should we allow negative forces? I'm leaning towards yes for now.
        // we may want to consider refactoring this into 2 functions based on torque direction
        // and have a separate function to calculate the net torque
        Ok(force * lever_arm * angle.sin())
    }

    pub fn calculate_angular_velocity(&self, linear_velocity: f64, radius: f64) -> Result<f64, PhysicsError> {
        if radius < 0.0 { return Err(PhysicsError::InvalidRadius); }
        if radius == 0.0 { return Err(PhysicsError::DivisionByZero); }
        Ok(linear_velocity / radius)
    }
}