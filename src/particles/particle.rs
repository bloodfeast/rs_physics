use crate::utils::{PhysicsConstants, PhysicsError};

#[derive(Debug, Clone)]
pub struct Particle {
    /// Position represented as (x, y).
    pub position: (f64, f64),
    /// Speed (scalar magnitude) of the particle.
    pub speed: f64,
    /// Normalized direction vector (weights for x and y) always between -1.0 and 1.0.
    pub direction: (f64, f64),
    /// Particle's mass.
    pub mass: f64,
}

impl Particle {
    /// Creates a new Particle.
    ///
    /// The provided `direction` vector is normalized to ensure that each component is within -1.0 and 1.0.
    ///
    /// # Errors
    ///
    /// Returns an error if `mass` is non-positive or if the provided direction vector is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::Particle;
    /// use rs_physics::utils::{PhysicsConstants, PhysicsError};
    ///
    /// let particle = Particle::new((0.0, 0.0), 10.0, (1.0, 2.0), 1.0)
    ///     .expect("Failed to create particle");
    /// // The direction vector (1.0, 2.0) is normalized automatically.
    /// let magnitude = (particle.direction.0 * particle.direction.0 + particle.direction.1 * particle.direction.1).sqrt();
    /// assert!((magnitude - 1.0).abs() < 1e-6, "Direction vector is not normalized");
    /// ```
    pub fn new(
        position: (f64, f64),
        speed: f64,
        direction: (f64, f64),
        mass: f64,
    ) -> Result<Self, PhysicsError> {
        if mass <= 0.0 {
            return Err(PhysicsError::InvalidMass);
        }
        let norm = (direction.0 * direction.0 + direction.1 * direction.1).sqrt();
        if norm == 0.0 {
            return Err(PhysicsError::CalculationError(
                "Direction vector cannot be zero".to_string(),
            ));
        }
        // Normalize the direction so each component is between -1.0 and 1.0.
        let normalized_direction = (direction.0 / norm, direction.1 / norm);
        Ok(Particle {
            position,
            speed,
            direction: normalized_direction,
            mass,
        })
    }

    /// Updates the particle's state over a time step `dt` using Euler integration.
    ///
    /// This method computes the current velocity vector from `speed` and `direction`,
    /// applies gravity to the vertical component, updates the position, and then recalculates the speed and
    /// normalized direction from the updated velocity.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::Particle;
    /// use rs_physics::utils::{PhysicsConstants, PhysicsError};
    ///
    /// let constants = PhysicsConstants::default();
    /// let mut particle = Particle::new((0.0, 0.0), 10.0, (1.0, 0.0), 1.0)
    ///     .expect("Failed to create particle");
    ///
    /// let initial_position = particle.position;
    /// particle.update(0.016, &constants)
    ///     .expect("Failed to update particle");
    ///
    /// // With gravity applied, both components of position should increase appropriately.
    /// assert!(particle.position.0 > initial_position.0, "Horizontal position did not increase as expected");
    /// assert!(particle.position.1 > initial_position.1, "Vertical position did not increase as expected");
    /// ```
    pub fn update(&mut self, dt: f64, constants: &PhysicsConstants) -> Result<(), PhysicsError> {
        let vx = self.speed * self.direction.0;
        let mut vy = self.speed * self.direction.1;

        // Apply gravity to the vertical component. For the horizontal component, vx remains constant.
        // This may be updated in future versions to support gravitational force on any axis, but for now
        // we'll assume gravity is always vertical.
        vy += constants.gravity * dt;

        self.position.0 += vx * dt;
        self.position.1 += vy * dt;

        // Compute new speed from updated velocity.
        let new_speed = (vx * vx + vy * vy).sqrt();

        if new_speed != 0.0 {
            self.direction = (vx / new_speed, vy / new_speed);
        }
        self.speed = new_speed;

        Ok(())
    }

    /// Updates the particle's state over a time step `dt` using Euler integration,
    /// optionally applying additional physical effects such as air resistance.
    ///
    /// The `drag` parameter is an optional tuple `(drag_coefficient, cross_sectional_area)`.
    /// If provided, air resistance is calculated using the physics constants (specifically `air_density`)
    /// to compute the drag force.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::Particle;
    /// use rs_physics::utils::{PhysicsConstants, PhysicsError};
    ///
    /// let constants = PhysicsConstants::default();
    /// let mut particle = Particle::new((0.0, 0.0), 10.0, (0.707, 0.707), 1.0)
    ///     .expect("Failed to create particle");
    ///
    /// let initial_speed = particle.speed;
    /// particle.update_with_effects(0.016, &constants, Some((0.47, 1.0)))
    ///     .expect("Failed to update particle with effects");
    ///
    /// // Air resistance should reduce the speed compared to the initial value.
    /// assert!(particle.speed < initial_speed, "Speed did not reduce due to drag as expected");
    /// ```
    pub fn update_with_effects(
        &mut self,
        dt: f64,
        constants: &PhysicsConstants,
        drag: Option<(f64, f64)>,
    ) -> Result<(), PhysicsError> {
        let mut vx = self.speed * self.direction.0;
        let mut vy = self.speed * self.direction.1;

        vy += constants.gravity * dt;

        let mut current_speed = (vx * vx + vy * vy).sqrt();

        // If drag parameters are provided, calculate air resistance and apply deceleration.
        match drag {
            Some((drag_coefficient, cross_sectional_area)) => {
                let drag_force = constants.calculate_air_resistance(current_speed, drag_coefficient, cross_sectional_area)?;
                let drag_acceleration = drag_force / self.mass;
                let drag_velocity = drag_acceleration * dt;

                current_speed = if current_speed > drag_velocity {
                    current_speed - drag_velocity
                } else {
                    0.0
                };

                if current_speed != 0.0 {
                    vx = self.direction.0 * current_speed;
                    vy = self.direction.1 * current_speed;
                } else {
                    vx = 0.0;
                    vy = 0.0;
                }
            }
            None => {}
        }

        self.position.0 += vx * dt;
        self.position.1 += vy * dt;

        let new_speed = (vx * vx + vy * vy).sqrt();
        if new_speed != 0.0 {
            self.direction = (vx / new_speed, vy / new_speed);
        }
        self.speed = new_speed;

        Ok(())
    }
}
