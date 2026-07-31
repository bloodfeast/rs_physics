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

    /// Pace the simulation against wall-clock time.
    ///
    /// When `true` (the default), the physics thread targets one tick every
    /// `timestep` seconds of real time, so one simulated second takes one real
    /// second. This is what you want behind a renderer.
    ///
    /// When `false`, the thread runs ticks back-to-back as fast as it can. Use
    /// this for headless batch runs where you want results sooner than real
    /// time, not for anything being displayed.
    ///
    /// Default: true
    pub real_time: bool,

    /// Maximum ticks the simulation will run back-to-back to catch up after
    /// falling behind wall clock (real-time mode only).
    ///
    /// If a tick overruns its deadline - a GC pause, a debugger break, a
    /// pathological collision frame - the thread runs extra ticks to catch up.
    /// Without a cap that becomes a death spiral: catching up costs more time,
    /// which puts you further behind, which demands more catch-up. On exceeding
    /// this many ticks of debt the simulation abandons the lost time and
    /// resynchronizes, trading a visible hitch for a recoverable one.
    ///
    /// Default: 4
    pub max_catchup_ticks: u32,
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
            real_time: true,
            max_catchup_ticks: 4,
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

    /// Builder pattern: pace the simulation to wall-clock time
    ///
    /// Enabled by default. Pass `false` only for headless runs that should
    /// complete as fast as the machine allows. See [`WorldConfig::real_time`].
    pub fn with_real_time(mut self, enabled: bool) -> Self {
        self.real_time = enabled;
        self
    }

    /// Builder pattern: set the catch-up ceiling
    ///
    /// See [`WorldConfig::max_catchup_ticks`]. Clamped to a minimum of 1.
    pub fn with_max_catchup_ticks(mut self, ticks: u32) -> Self {
        self.max_catchup_ticks = ticks.max(1);
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

    /// Create config optimized for real-time rendering
    ///
    /// Runs at 240 Hz paced to wall clock. Read it from the render thread with
    /// [`PhysicsHandle::get_interpolated_state`], which blends between ticks so
    /// motion stays smooth at any display refresh rate.
    ///
    /// [`PhysicsHandle::get_interpolated_state`]: super::PhysicsHandle::get_interpolated_state
    pub fn real_time() -> Self {
        Self::default()
            .with_frequency(240.0)
            .with_real_time(true)
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
        assert!(config.real_time, "wall-clock pacing is the default");
        assert_eq!(config.max_catchup_ticks, 4);
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

    #[test]
    fn test_real_time_config() {
        let config = WorldConfig::real_time();
        assert!(config.real_time);
        assert!((config.timestep - 1.0 / 240.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_catchup_ticks_cannot_be_zero() {
        // Zero would mean "never catch up", silently running in slow motion.
        assert_eq!(WorldConfig::new().with_max_catchup_ticks(0).max_catchup_ticks, 1);
    }
}
