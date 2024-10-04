// src/constants_config.rs

use std::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
pub struct PhysicsConstants {
    pub gravity: f64,
    pub air_density: f64,
    pub speed_of_sound: f64,
    pub atmospheric_pressure: f64,
}

impl Default for PhysicsConstants {
    fn default() -> Self {
        Self {
            gravity: 9.80665,
            air_density: 1.225,
            speed_of_sound: 343.0,
            atmospheric_pressure: 101_325.0,
        }
    }
}

impl PhysicsConstants {
    pub fn new(
        gravity: Option<f64>,
        air_density: Option<f64>,
        speed_of_sound: Option<f64>,
        atmospheric_pressure: Option<f64>
    ) -> Self {
        let default = PhysicsConstants::default();
        Self {
            gravity: gravity.unwrap_or(default.gravity),
            air_density: air_density.unwrap_or(default.air_density),
            speed_of_sound: speed_of_sound.unwrap_or(default.speed_of_sound),
            atmospheric_pressure: atmospheric_pressure.unwrap_or(default.atmospheric_pressure),
        }
    }

    pub fn calculate_terminal_velocity(&self, mass: f64, drag_coefficient: f64, cross_sectional_area: f64) -> Result<f64, &'static str> {
        if mass <= 0.0 { return Err("Mass must be positive"); }
        if drag_coefficient <= 0.0 { return Err("Drag coefficient must be positive"); }
        if cross_sectional_area <= 0.0 { return Err("Cross-sectional area must be positive"); }
        if self.air_density <= 0.0 { return Err("Air density must be positive"); }

        Ok(((2.0 * mass * self.gravity) / (self.air_density * drag_coefficient * cross_sectional_area)).sqrt())
    }

    pub fn calculate_air_resistance(&self, velocity: f64, drag_coefficient: f64, cross_sectional_area: f64) -> Result<f64, &'static str> {
        if drag_coefficient < 0.0 { return Err("Drag coefficient must be non-negative"); }
        if cross_sectional_area < 0.0 { return Err("Cross-sectional area must be non-negative"); }
        if self.air_density < 0.0 { return Err("Air density must be non-negative"); }

        Ok(0.5 * self.air_density * velocity * velocity * drag_coefficient * cross_sectional_area)
    }

    pub fn calculate_acceleration(&self, force: f64, mass: f64) -> Result<f64, &'static str> {
        if mass == 0.0 { return Err("Mass cannot be zero"); }
        Ok(force / mass)
    }

    pub fn calculate_deceleration(&self, force: f64, mass: f64) -> Result<f64, &'static str> {
        self.calculate_acceleration(force, mass).map(|a| -a)
    }

    pub fn calculate_force(&self, mass: f64, acceleration: f64) -> Result<f64, &'static str> {
        if mass < 0.0 { return Err("Mass must be non-negative"); }
        Ok(mass * acceleration)
    }

    pub fn calculate_momentum(&self, mass: f64, velocity: f64) -> Result<f64, &'static str> {
        if mass < 0.0 { return Err("Mass must be non-negative"); }
        Ok(mass * velocity)
    }

    pub fn calculate_velocity(&self, initial_velocity: f64, acceleration: f64, time: f64) -> Result<f64, &'static str> {
        if time < 0.0 { return Err("Time must be non-negative"); }
        Ok(initial_velocity + acceleration * time)
    }

    pub fn calculate_average_velocity(&self, displacement: f64, time: f64) -> Result<f64, &'static str> {
        if time == 0.0 { return Err("Time cannot be zero"); }
        Ok(displacement / time)
    }

    pub fn calculate_kinetic_energy(&self, mass: f64, velocity: f64) -> Result<f64, &'static str> {
        if mass < 0.0 { return Err("Mass must be non-negative"); }
        Ok(0.5 * mass * velocity * velocity)
    }

    pub fn calculate_potential_energy(&self, mass: f64, height: f64) -> Result<f64, &'static str> {
        if mass < 0.0 { return Err("Mass must be non-negative"); }
        Ok(mass * self.gravity * height)
    }

    pub fn calculate_work(&self, force: f64, displacement: f64) -> Result<f64, &'static str> {
        Ok(force * displacement)
    }

    pub fn calculate_power(&self, work: f64, time: f64) -> Result<f64, &'static str> {
        if time == 0.0 { return Err("Time cannot be zero"); }
        Ok(work / time)
    }

    pub fn calculate_impulse(&self, force: f64, time: f64) -> Result<f64, &'static str> {
        if time < 0.0 { return Err("Time must be non-negative"); }
        Ok(force * time)
    }

    pub fn calculate_coefficient_of_restitution(&self, velocity_before: f64, velocity_after: f64) -> Result<f64, &'static str> {
        if velocity_before == 0.0 { return Err("Velocity before collision cannot be zero"); }
        Ok(-velocity_after / velocity_before)
    }

    pub fn calculate_projectile_time_of_flight(&self, initial_velocity: f64, angle: f64) -> Result<f64, &'static str> {
        if initial_velocity < 0.0 { return Err("Initial velocity must be non-negative"); }
        let normalized_angle = angle % (2.0 * PI);
        if normalized_angle < 0.0 || normalized_angle > PI / 2.0 { return Err("Angle must be between 0 and π/2"); }
        let v_y = initial_velocity * normalized_angle.sin();
        Ok(2.0 * v_y / self.gravity)
    }

    pub fn calculate_projectile_max_height(&self, initial_velocity: f64, angle: f64) -> Result<f64, &'static str> {
        if initial_velocity < 0.0 { return Err("Initial velocity must be non-negative"); }
        let normalized_angle = angle % (2.0 * PI);
        if normalized_angle < 0.0 || normalized_angle > PI / 2.0 { return Err("Angle must be between 0 and π/2"); }
        let v_y = initial_velocity * normalized_angle.sin();
        Ok((v_y * v_y) / (2.0 * self.gravity))
    }

    pub fn calculate_centripetal_force(&self, mass: f64, velocity: f64, radius: f64) -> Result<f64, &'static str> {
        if mass < 0.0 { return Err("Mass must be non-negative"); }
        if radius == 0.0 { return Err("Radius cannot be zero"); }
        Ok(mass * velocity * velocity / radius)
    }

    pub fn calculate_torque(&self, force: f64, lever_arm: f64, angle: f64) -> Result<f64, &'static str> {
        if lever_arm < 0.0 { return Err("Lever arm must be non-negative"); }
        Ok(force * lever_arm * angle.sin())
    }

    pub fn calculate_angular_velocity(&self, linear_velocity: f64, radius: f64) -> Result<f64, &'static str> {
        if radius == 0.0 { return Err("Radius cannot be zero"); }
        Ok(linear_velocity / radius)
    }
}