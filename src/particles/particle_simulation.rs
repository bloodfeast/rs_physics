//! This module provides a high-performance simulation of multiple particles.
//! The simulation uses a structure-of-arrays layout for particle properties
//! and leverages SIMD (AVX) intrinsics when available to update many particles
//! in parallel. When AVX is not available, the simulation falls back to using Rayon
//! to process particles in parallel.
//!
//! Each particle is represented by its position, speed, normalized direction, and mass.
//! Gravity and other constants are provided via a shared PhysicsConstants instance,
//! and Euler integration is used to update the state.
//!
//! # Example
//!
//! ```
//! use rs_physics::particles::Simulation;
//! use rs_physics::utils::{PhysicsConstants, PhysicsError};
//!
//! // Create default physics constants and choose a time step.
//! let constants = PhysicsConstants::default();
//! let dt = 0.016; // 16ms per simulation step (~60 FPS)
//!
//! // Initialize a simulation with 1000 identical particles.
//! let mut sim = Simulation::new(
//!     1000,
//!     (0.0, 0.0),
//!     10.0,
//!     (1.0, 0.0),
//!     1.0,
//!     constants,
//!     dt
//! ).expect("Failed to initialize simulation");
//!
//! // Run the simulation for 100 steps.
//! sim.simulate(100).expect("Simulation failed");
//!
//! // After simulation, inspect the position of the first particle.
//! println!("First particle position: ({}, {})", sim.positions_x[0], sim.positions_y[0]);
//! ```
use crate::utils::{PhysicsConstants, PhysicsError};
use rayon::prelude::*;

/// A high-performance simulation of multiple particles.
///
/// The simulation uses a structure-of-arrays layout:
/// - `positions_x` and `positions_y` store the x and y coordinates.
/// - `speeds` store the scalar speeds.
/// - `directions_x` and `directions_y` store the normalized direction components.
/// - `masses` store the mass of each particle.
///
/// Gravity and other constants are provided via `constants`, and `dt` is the time step.
pub struct Simulation {
    pub positions_x: Vec<f64>,
    pub positions_y: Vec<f64>,
    pub speeds: Vec<f64>,
    pub directions_x: Vec<f64>,
    pub directions_y: Vec<f64>,
    pub masses: Vec<f64>,
    pub constants: PhysicsConstants,
    pub dt: f64,
}

impl Simulation {
    /// Creates a new simulation with `num_particles` identical particles.
    ///
    /// Each particle is initialized with the given `initial_position`, `initial_speed`,
    /// `initial_direction`, and `mass`. The provided direction is normalized.
    ///
    /// # Errors
    ///
    /// Returns an error if `initial_direction` is the zero vector, or
    /// `PhysicsError::InvalidMass` if `mass` is not positive and finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::Simulation;
    /// use rs_physics::utils::{PhysicsConstants, PhysicsError};
    ///
    /// let constants = PhysicsConstants::default();
    /// let sim = Simulation::new(100, (0.0, 0.0), 10.0, (1.0, 0.0), 1.0, constants, 0.016)
    ///     .expect("Failed to create simulation");
    ///
    /// // Assert that the simulation arrays have the correct length.
    /// assert_eq!(sim.positions_x.len(), 100);
    /// assert_eq!(sim.positions_y.len(), 100);
    /// assert_eq!(sim.speeds.len(), 100);
    /// // Assert that the direction vector is normalized.
    /// let mag = (sim.directions_x[0].powi(2) + sim.directions_y[0].powi(2)).sqrt();
    /// assert!((mag - 1.0).abs() < 1e-6, "Direction vector not normalized");
    /// ```
    pub fn new(
        num_particles: usize,
        initial_position: (f64, f64),
        initial_speed: f64,
        initial_direction: (f64, f64),
        mass: f64,
        constants: PhysicsConstants,
        dt: f64,
    ) -> Result<Self, PhysicsError> {
        if !(mass > 0.0 && mass.is_finite()) {
            return Err(PhysicsError::InvalidMass);
        }
        let norm = (initial_direction.0.powi(2) + initial_direction.1.powi(2)).sqrt();
        if norm == 0.0 {
            return Err(PhysicsError::CalculationError(
                "initial_direction cannot be zero".to_string(),
            ));
        }
        let dir_x = initial_direction.0 / norm;
        let dir_y = initial_direction.1 / norm;

        Ok(Simulation {
            positions_x: vec![initial_position.0; num_particles],
            positions_y: vec![initial_position.1; num_particles],
            speeds: vec![initial_speed; num_particles],
            directions_x: vec![dir_x; num_particles],
            directions_y: vec![dir_y; num_particles],
            masses: vec![mass; num_particles],
            constants,
            dt,
        })
    }

    /// Advances the simulation by one time step.
    ///
    /// For each particle, the simulation computes the velocity components,
    /// applies gravity, updates the position, and recalculates the new speed and normalized direction.
    ///
    /// This method uses runtime detection to decide whether to use an AVX‑optimized update or a Rayon‑based fallback.
    ///
    /// # Errors
    ///
    /// Returns `PhysicsError::InvalidTime` if `dt` is not finite, and
    /// `PhysicsError::CalculationError` if the particle arrays (which are public)
    /// do not all have the same length. Nothing is updated in either case.
    ///
    /// # Examples (non-AVX)
    ///
    /// ```
    /// use rs_physics::particles::Simulation;
    /// use rs_physics::utils::{PhysicsConstants, PhysicsError};
    ///
    /// // Create a simulation with 10 particles.
    /// let constants = PhysicsConstants::default();
    /// let mut sim = Simulation::new(10, (0.0, 0.0), 10.0, (1.0, 0.0), 1.0, constants, 0.016)
    ///     .expect("Failed to create simulation");
    ///
    /// // Record the initial vertical position of the first particle.
    /// let initial_y = sim.positions_y[0];
    /// sim.step().expect("Step failed");
    /// // Assert that the vertical position has increased due to gravity.
    /// assert!(sim.positions_y[0] > initial_y, "Particle did not move vertically");
    /// ```
    pub fn step(&mut self) -> Result<(), PhysicsError> {
        if !self.dt.is_finite() {
            return Err(PhysicsError::InvalidTime);
        }
        let n = self.speeds.len();
        if self.positions_x.len() != n
            || self.positions_y.len() != n
            || self.directions_x.len() != n
            || self.directions_y.len() != n
        {
            return Err(PhysicsError::CalculationError(
                "positions, speeds and directions must all have the same length".to_string(),
            ));
        }

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: `step_avx` has two preconditions and both hold here. The CPU
            // supports AVX (detected at runtime on the line above), and all five
            // arrays it loads from and stores to have the same length `n` (checked
            // above), so every 4-wide access at `i` with `i + 4 <= n` is in bounds.
            unsafe { self.step_avx() };
            return Ok(());
        }

        Self::step_fallback(self);
        Ok(())
    }

    /// AVX-optimized update function.
    ///
    /// # Safety
    ///
    /// This function must only be called when AVX is available, and when
    /// `positions_x`, `positions_y`, `directions_x` and `directions_y` all have
    /// the same length as `speeds`: the loop is bounded by `speeds.len()` and reads
    /// and writes the others through raw pointers.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn step_avx(&mut self) {

        let n = self.speeds.len();
        let mut i = 0;
        // Process particles in blocks of 4 using AVX.
        use std::arch::x86_64::*;
        while i + 4 <= n {
            let speed = _mm256_loadu_pd(self.speeds.as_ptr().add(i));
            let dir_x = _mm256_loadu_pd(self.directions_x.as_ptr().add(i));
            let dir_y = _mm256_loadu_pd(self.directions_y.as_ptr().add(i));
            let pos_x = _mm256_loadu_pd(self.positions_x.as_ptr().add(i));
            let pos_y = _mm256_loadu_pd(self.positions_y.as_ptr().add(i));

            let vx = _mm256_mul_pd(speed, dir_x);
            let vy = _mm256_mul_pd(speed, dir_y);

            let gravity_dt = _mm256_set1_pd(self.constants.gravity * self.dt);
            let vy = _mm256_add_pd(vy, gravity_dt);

            let dt_vec = _mm256_set1_pd(self.dt);
            let new_pos_x = _mm256_add_pd(pos_x, _mm256_mul_pd(vx, dt_vec));
            let new_pos_y = _mm256_add_pd(pos_y, _mm256_mul_pd(vy, dt_vec));

            let vx2 = _mm256_mul_pd(vx, vx);
            let vy2 = _mm256_mul_pd(vy, vy);
            let sum = _mm256_add_pd(vx2, vy2);
            let new_speed = _mm256_sqrt_pd(sum);

            let zero = _mm256_set1_pd(0.0);
            let mask = _mm256_cmp_pd::<_CMP_EQ_OQ>(new_speed, zero);
            let new_dir_x = _mm256_blendv_pd(_mm256_div_pd(vx, new_speed), dir_x, mask);
            let new_dir_y = _mm256_blendv_pd(_mm256_div_pd(vy, new_speed), dir_y, mask);

            _mm256_storeu_pd(self.positions_x.as_mut_ptr().add(i), new_pos_x);
            _mm256_storeu_pd(self.positions_y.as_mut_ptr().add(i), new_pos_y);
            _mm256_storeu_pd(self.speeds.as_mut_ptr().add(i), new_speed);
            _mm256_storeu_pd(self.directions_x.as_mut_ptr().add(i), new_dir_x);
            _mm256_storeu_pd(self.directions_y.as_mut_ptr().add(i), new_dir_y);

            i += 4;
        }

        // The 0..=3 remaining particles, sequentially: handing them to Rayon costs
        // far more than the whole vectorized loop above.
        for j in i..n {
            Self::update_one(
                &mut self.positions_x[j],
                &mut self.positions_y[j],
                &mut self.speeds[j],
                &mut self.directions_x[j],
                &mut self.directions_y[j],
                self.dt,
                self.constants.gravity,
            );
        }
    }

    /// Fallback update function when AVX is not available.
    fn step_fallback(sim: &mut Simulation) {
        Self::update_slice(
            &mut sim.positions_x,
            &mut sim.positions_y,
            &mut sim.speeds,
            &mut sim.directions_x,
            &mut sim.directions_y,
            sim.dt,
            sim.constants.gravity,
        );
    }

    /// Runs the simulation for a specified number of steps.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::Simulation;
    /// use rs_physics::utils::{PhysicsConstants, PhysicsError};
    ///
    /// let constants = PhysicsConstants::default();
    /// let mut sim = Simulation::new(10, (0.0, 0.0), 10.0, (1.0, 0.0), 1.0, constants, 0.016)
    ///     .expect("Failed to create simulation");
    ///
    /// // Run the simulation for 100 steps.
    /// sim.simulate(100).expect("Simulation failed");
    ///
    /// // Assert that after simulation, the x positions have increased.
    /// for &x in &sim.positions_x {
    ///     assert!(x > 0.0, "Particle x position did not increase");
    /// }
    /// ```
    pub fn simulate(&mut self, steps: usize) -> Result<(), PhysicsError> {
        for _ in 0..steps {
            self.step()?;
        }
        Ok(())
    }


    /// Private helper function to update a slice of particle arrays in parallel.
    ///
    /// This function uses Rayon’s safe parallel iterators to process the given slices.
    ///
    /// # Parameters
    ///
    /// - `positions_x`, `positions_y`: Mutable slices of particle positions.
    /// - `speeds`: Mutable slice of particle speeds.
    /// - `directions_x`, `directions_y`: Mutable slices of particle normalized direction components.
    /// - `dt`: Time step.
    /// - `gravity`: Gravity constant.
    fn update_slice(
        positions_x: &mut [f64],
        positions_y: &mut [f64],
        speeds: &mut [f64],
        directions_x: &mut [f64],
        directions_y: &mut [f64],
        dt: f64,
        gravity: f64,
    ) {
        positions_x
            .par_iter_mut()
            .zip(positions_y.par_iter_mut())
            .zip(speeds.par_iter_mut())
            .zip(directions_x.par_iter_mut())
            .zip(directions_y.par_iter_mut())
            .for_each(|((((px, py), speed), dx), dy)| {
                Self::update_one(px, py, speed, dx, dy, dt, gravity);
            });
    }

    /// Advances one particle. Shared by the Rayon path and the AVX path's scalar
    /// tail so both apply exactly the same arithmetic as the vectorized loop.
    #[inline]
    fn update_one(
        px: &mut f64,
        py: &mut f64,
        speed: &mut f64,
        dx: &mut f64,
        dy: &mut f64,
        dt: f64,
        gravity: f64,
    ) {
        let vx = *speed * *dx;
        let vy = *speed * *dy + gravity * dt;
        *px += vx * dt;
        *py += vy * dt;
        let new_speed = (vx * vx + vy * vy).sqrt();
        if new_speed != 0.0 {
            *dx = vx / new_speed;
            *dy = vy / new_speed;
        }
        *speed = new_speed;
    }
}
