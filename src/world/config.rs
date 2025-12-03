//! Configuration types for PhysicsWorld

use crate::utils::PhysicsConstants;

/// Configuration for creating a PhysicsWorld
#[derive(Debug, Clone)]
pub struct WorldConfig {
    /// Fixed timestep for physics simulation in seconds
    /// Default: 1/120 (120 Hz)
    pub timestep: f64,

    /// Gravity vector (x, y, z) in m/s²
    /// Default: (0.0, -9.81, 0.0)
    pub gravity: (f64, f64, f64),

    /// Maximum number of collision iterations per step
    /// Default: 4
    pub max_collision_iterations: usize,

    /// Number of physics ticks between state broadcasts
    /// Lower = more responsive but more overhead
    /// Default: 1 (broadcast every tick)
    pub broadcast_rate: usize,

    /// Physics constants (air density, etc.)
    pub constants: PhysicsConstants,

    /// Enable continuous collision detection (CCD)
    /// Default: true
    pub enable_ccd: bool,

    /// Velocity threshold for CCD activation
    /// Objects moving slower than this use discrete detection
    /// Default: 1.0 m/s
    pub ccd_velocity_threshold: f64,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            timestep: 1.0 / 120.0,  // 120 Hz
            gravity: (0.0, -9.81, 0.0),
            max_collision_iterations: 4,
            broadcast_rate: 1,
            constants: PhysicsConstants::default(),
            enable_ccd: true,
            ccd_velocity_threshold: 1.0,
        }
    }
}

impl WorldConfig {
    /// Create a new config with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder pattern: set timestep
    pub fn with_timestep(mut self, timestep: f64) -> Self {
        self.timestep = timestep;
        self
    }

    /// Builder pattern: set timestep from frequency (Hz)
    pub fn with_frequency(mut self, hz: f64) -> Self {
        self.timestep = 1.0 / hz;
        self
    }

    /// Builder pattern: set gravity
    pub fn with_gravity(mut self, x: f64, y: f64, z: f64) -> Self {
        self.gravity = (x, y, z);
        self
    }

    /// Builder pattern: set physics constants
    pub fn with_constants(mut self, constants: PhysicsConstants) -> Self {
        self.constants = constants;
        self
    }

    /// Builder pattern: disable CCD
    pub fn without_ccd(mut self) -> Self {
        self.enable_ccd = false;
        self
    }

    /// Builder pattern: set broadcast rate
    pub fn with_broadcast_rate(mut self, rate: usize) -> Self {
        self.broadcast_rate = rate.max(1);  // Minimum 1
        self
    }

    /// Create config for a zero-gravity environment (space)
    pub fn zero_gravity() -> Self {
        Self::default().with_gravity(0.0, 0.0, 0.0)
    }

    /// Create config for Moon gravity
    pub fn moon() -> Self {
        Self::default().with_gravity(0.0, -1.62, 0.0)
    }

    /// Create config for Mars gravity
    pub fn mars() -> Self {
        Self::default().with_gravity(0.0, -3.71, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = WorldConfig::default();
        assert!((config.timestep - 1.0/120.0).abs() < 1e-10);
        assert_eq!(config.gravity, (0.0, -9.81, 0.0));
        assert!(config.enable_ccd);
    }

    #[test]
    fn test_builder_pattern() {
        let config = WorldConfig::new()
            .with_frequency(60.0)
            .with_gravity(0.0, -10.0, 0.0)
            .without_ccd();

        assert!((config.timestep - 1.0/60.0).abs() < 1e-10);
        assert_eq!(config.gravity, (0.0, -10.0, 0.0));
        assert!(!config.enable_ccd);
    }

    #[test]
    fn test_presets() {
        let space = WorldConfig::zero_gravity();
        assert_eq!(space.gravity, (0.0, 0.0, 0.0));

        let moon = WorldConfig::moon();
        assert_eq!(moon.gravity, (0.0, -1.62, 0.0));
    }
}
