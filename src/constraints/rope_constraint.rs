//! Rope/cable constraints for physics simulations.
//!
//! This module provides rope constraints that only resist stretching.
//! Unlike joint constraints that maintain exact distances, ropes allow
//! objects to be closer than the max length (slack) but not further.

use crate::utils::PhysicsError;
use crate::models::{ObjectIn2D, ObjectIn3D};
use super::solver::{Constraint2D, Constraint3D};

/// A rope constraint between two 2D objects.
///
/// Ropes only apply corrections when stretched beyond their maximum length.
/// When the distance is less than max_length, no force is applied (rope is slack).
///
/// # Examples
///
/// ```rust,ignore
/// use rs_physics::constraints::Rope2D;
/// use rs_physics::models::ObjectIn2D;
///
/// let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Default::default());
/// let obj2 = ObjectIn2D::with_shape(1.0, (5.0, 0.0), Default::default());
/// let rope = Rope2D::new(obj1, obj2, 5.0).expect("Valid rope");
/// ```
pub struct Rope2D {
    /// First object connected by the rope
    pub object1: ObjectIn2D,
    /// Second object connected by the rope
    pub object2: ObjectIn2D,
    /// Maximum length before constraint activates (meters)
    pub max_length: f64,
    /// Baumgarte stabilization factor (0.0 to 1.0, default 0.2)
    pub baumgarte: f64,
    /// Accumulated impulse for warm starting
    pub lambda: f64,
}

impl Rope2D {
    /// Creates a new Rope2D constraint between two objects.
    ///
    /// # Arguments
    ///
    /// * `object1` - First object connected by the rope
    /// * `object2` - Second object connected by the rope
    /// * `max_length` - Maximum length before constraint activates (must be > 0)
    ///
    /// # Returns
    ///
    /// * `Ok(Rope2D)` - Valid rope constraint
    /// * `Err(PhysicsError)` - If max_length is not positive
    pub fn new(object1: ObjectIn2D, object2: ObjectIn2D, max_length: f64) -> Result<Self, PhysicsError> {
        if max_length <= 0.0 {
            return Err(PhysicsError::InvalidDistance);
        }
        Ok(Self {
            object1,
            object2,
            max_length,
            baumgarte: 0.2,
            lambda: 0.0,
        })
    }

    /// Sets the Baumgarte stabilization factor.
    pub fn with_baumgarte(mut self, factor: f64) -> Self {
        self.baumgarte = factor;
        self
    }

    /// Returns true if the rope is taut (stretched to or beyond max_length).
    pub fn is_taut(&self) -> bool {
        let dx = self.object2.position.x - self.object1.position.x;
        let dy = self.object2.position.y - self.object1.position.y;
        let current_length = (dx * dx + dy * dy).sqrt();
        current_length >= self.max_length
    }

    /// Solves the rope constraint for one iteration.
    ///
    /// Only applies corrections if the rope is taut. When taut, the rope
    /// behaves like a distance constraint (Joint2D), preserving tangential
    /// velocity while only correcting radial velocity and position.
    ///
    /// The key insight is that when position is corrected to keep the rope
    /// at max_length, we must also add velocity in the tangential direction
    /// so the object swings naturally rather than just dropping straight down.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    pub fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        let dx = self.object2.position.x - self.object1.position.x;
        let dy = self.object2.position.y - self.object1.position.y;
        let current_length = (dx * dx + dy * dy).sqrt();

        // Only apply constraint when rope is taut (stretched beyond max_length)
        if current_length <= self.max_length {
            return Ok(());
        }

        if current_length < 1e-10 {
            return Ok(());
        }

        let error = current_length - self.max_length;

        // Calculate inverse masses for mass-weighted corrections
        let inv_mass1 = if self.object1.mass.is_infinite() { 0.0 } else { 1.0 / self.object1.mass };
        let inv_mass2 = if self.object2.mass.is_infinite() { 0.0 } else { 1.0 / self.object2.mass };
        let total_inv_mass = inv_mass1 + inv_mass2;

        if total_inv_mass < 1e-10 {
            return Ok(());
        }

        // Normalize direction (points from object1 to object2)
        let nx = dx / current_length;
        let ny = dy / current_length;

        // Calculate relative velocity
        let rel_vx = self.object2.velocity.x - self.object1.velocity.x;
        let rel_vy = self.object2.velocity.y - self.object1.velocity.y;

        // Decompose velocity into radial component
        let radial_vel = rel_vx * nx + rel_vy * ny;

        // Position correction - project object2 back onto the constraint circle
        let position_correction = error.min(0.2);

        // Calculate target position on the constraint circle
        let target_length = self.max_length;
        let scale = target_length / current_length;
        let target_x2 = self.object1.position.x + dx * scale;
        let target_y2 = self.object1.position.y + dy * scale;

        // Position displacement caused by constraint
        let disp_x = target_x2 - self.object2.position.x;
        let disp_y = target_y2 - self.object2.position.y;

        // KEY FIX: Convert position displacement into velocity
        // This ensures that when the rope pulls the object onto the arc,
        // the object gains velocity in that direction to continue swinging.
        if dt > 1e-10 {
            let vel_from_correction = self.baumgarte * 0.5;
            if inv_mass2 > 0.0 {
                self.object2.velocity.x += disp_x * vel_from_correction / dt;
                self.object2.velocity.y += disp_y * vel_from_correction / dt;
            }
            if inv_mass1 > 0.0 {
                self.object1.velocity.x -= disp_x * vel_from_correction / dt * (inv_mass1 / inv_mass2).min(1.0);
                self.object1.velocity.y -= disp_y * vel_from_correction / dt * (inv_mass1 / inv_mass2).min(1.0);
            }
        }

        // Apply position correction (mass-weighted)
        self.object1.position.x += position_correction * (inv_mass1 / total_inv_mass) * nx;
        self.object1.position.y += position_correction * (inv_mass1 / total_inv_mass) * ny;
        self.object2.position.x -= position_correction * (inv_mass2 / total_inv_mass) * nx;
        self.object2.position.y -= position_correction * (inv_mass2 / total_inv_mass) * ny;

        // Handle velocity constraint - remove radial velocity component if separating
        if radial_vel > 0.0 {
            let bias = self.baumgarte * error / dt;
            let lambda = -(radial_vel + bias) / total_inv_mass;

            let max_impulse = 0.1 / dt;
            let clamped_lambda = lambda.min(max_impulse);

            self.lambda += clamped_lambda;

            self.object1.velocity.x -= clamped_lambda * inv_mass1 * nx;
            self.object1.velocity.y -= clamped_lambda * inv_mass1 * ny;
            self.object2.velocity.x += clamped_lambda * inv_mass2 * nx;
            self.object2.velocity.y += clamped_lambda * inv_mass2 * ny;
        }

        Ok(())
    }

    /// Calculates the current constraint error.
    ///
    /// # Returns
    ///
    /// 0 if rope is slack, positive value if rope is taut
    pub fn calculate_error(&self) -> f64 {
        let dx = self.object2.position.x - self.object1.position.x;
        let dy = self.object2.position.y - self.object1.position.y;
        let current_length = (dx * dx + dy * dy).sqrt();
        (current_length - self.max_length).max(0.0)
    }
}

impl Constraint2D for Rope2D {
    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        Rope2D::solve(self, dt)
    }

    fn calculate_error(&self) -> f64 {
        Rope2D::calculate_error(self)
    }

    fn get_lambda(&self) -> f64 {
        self.lambda
    }

    fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda;
    }
}

/// A rope constraint between two 3D objects.
///
/// Ropes only apply corrections when stretched beyond their maximum length.
/// When the distance is less than max_length, no force is applied (rope is slack).
///
/// # Examples
///
/// ```rust,ignore
/// use rs_physics::constraints::Rope3D;
/// use rs_physics::models::ObjectIn3D;
///
/// let mut obj1 = ObjectIn3D::default();
/// obj1.position = Axis3D { x: 0.0, y: 0.0, z: 0.0 };
/// let mut obj2 = ObjectIn3D::default();
/// obj2.position = Axis3D { x: 5.0, y: 0.0, z: 0.0 };
/// let rope = Rope3D::new(obj1, obj2, 5.0).expect("Valid rope");
/// ```
pub struct Rope3D {
    /// First object connected by the rope
    pub object1: ObjectIn3D,
    /// Second object connected by the rope
    pub object2: ObjectIn3D,
    /// Maximum length before constraint activates (meters)
    pub max_length: f64,
    /// Baumgarte stabilization factor (0.0 to 1.0, default 0.2)
    pub baumgarte: f64,
    /// Accumulated impulse for warm starting
    pub lambda: f64,
}

impl Rope3D {
    /// Creates a new Rope3D constraint between two objects.
    ///
    /// # Arguments
    ///
    /// * `object1` - First object connected by the rope
    /// * `object2` - Second object connected by the rope
    /// * `max_length` - Maximum length before constraint activates (must be > 0)
    ///
    /// # Returns
    ///
    /// * `Ok(Rope3D)` - Valid rope constraint
    /// * `Err(PhysicsError)` - If max_length is not positive
    pub fn new(object1: ObjectIn3D, object2: ObjectIn3D, max_length: f64) -> Result<Self, PhysicsError> {
        if max_length <= 0.0 {
            return Err(PhysicsError::InvalidDistance);
        }
        Ok(Self {
            object1,
            object2,
            max_length,
            baumgarte: 0.2,
            lambda: 0.0,
        })
    }

    /// Sets the Baumgarte stabilization factor.
    pub fn with_baumgarte(mut self, factor: f64) -> Self {
        self.baumgarte = factor;
        self
    }

    /// Returns true if the rope is taut (stretched to or beyond max_length).
    pub fn is_taut(&self) -> bool {
        let dx = self.object2.position.x - self.object1.position.x;
        let dy = self.object2.position.y - self.object1.position.y;
        let dz = self.object2.position.z - self.object1.position.z;
        let current_length = (dx * dx + dy * dy + dz * dz).sqrt();
        current_length >= self.max_length
    }

    /// Solves the rope constraint for one iteration.
    ///
    /// Only applies corrections if the rope is taut. When taut, the rope
    /// behaves like a distance constraint (Joint3D), preserving tangential
    /// velocity while only correcting radial velocity and position.
    ///
    /// The key insight is that when position is corrected to keep the rope
    /// at max_length, we must also add velocity in the tangential direction
    /// so the object swings naturally rather than just dropping straight down.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    pub fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        let dx = self.object2.position.x - self.object1.position.x;
        let dy = self.object2.position.y - self.object1.position.y;
        let dz = self.object2.position.z - self.object1.position.z;
        let current_length = (dx * dx + dy * dy + dz * dz).sqrt();

        // Only apply constraint when rope is taut (stretched beyond max_length)
        if current_length <= self.max_length {
            return Ok(());
        }

        if current_length < 1e-10 {
            return Ok(());
        }

        let error = current_length - self.max_length;

        // Calculate inverse masses for mass-weighted corrections
        let inv_mass1 = if self.object1.mass.is_infinite() { 0.0 } else { 1.0 / self.object1.mass };
        let inv_mass2 = if self.object2.mass.is_infinite() { 0.0 } else { 1.0 / self.object2.mass };
        let total_inv_mass = inv_mass1 + inv_mass2;

        if total_inv_mass < 1e-10 {
            return Ok(());
        }

        // Normalize direction (points from object1 to object2)
        let nx = dx / current_length;
        let ny = dy / current_length;
        let nz = dz / current_length;

        // Calculate relative velocity
        let rel_vx = self.object2.velocity.x - self.object1.velocity.x;
        let rel_vy = self.object2.velocity.y - self.object1.velocity.y;
        let rel_vz = self.object2.velocity.z - self.object1.velocity.z;

        // Decompose velocity into radial and tangential components
        let radial_vel = rel_vx * nx + rel_vy * ny + rel_vz * nz;

        // Only constrain the radial (stretching) component of velocity
        // The tangential component is preserved naturally
        // Radial velocity > 0 means objects are separating (stretching the rope)

        // For a rope (unilateral constraint), we only apply impulse when:
        // 1. The rope is stretched (error > 0) - already checked above
        // 2. Objects are moving apart (radial_vel > 0) OR we need position correction

        // Position correction - project object2 back onto the constraint sphere
        // This is always needed when taut to maintain the max_length
        let position_correction = error.min(0.2);  // Limit correction per iteration

        // Calculate what the position SHOULD be after correction
        let target_length = self.max_length;
        let scale = target_length / current_length;

        // New position for object2 on the constraint sphere
        let target_x2 = self.object1.position.x + dx * scale;
        let target_y2 = self.object1.position.y + dy * scale;
        let target_z2 = self.object1.position.z + dz * scale;

        // Position displacement caused by constraint
        let disp_x = target_x2 - self.object2.position.x;
        let disp_y = target_y2 - self.object2.position.y;
        let disp_z = target_z2 - self.object2.position.z;

        // KEY FIX: Convert position displacement into velocity
        // This ensures that when the rope pulls the object onto the arc,
        // the object gains velocity in that direction to continue swinging.
        // Without this, position correction moves the object but velocity
        // stays purely downward, causing it to immediately try to stretch again.
        if dt > 1e-10 {
            // Add velocity from position correction (scale by Baumgarte to avoid overshoot)
            let vel_from_correction = self.baumgarte * 0.5;  // Damped velocity addition
            if inv_mass2 > 0.0 {
                self.object2.velocity.x += disp_x * vel_from_correction / dt;
                self.object2.velocity.y += disp_y * vel_from_correction / dt;
                self.object2.velocity.z += disp_z * vel_from_correction / dt;
            }
            if inv_mass1 > 0.0 {
                self.object1.velocity.x -= disp_x * vel_from_correction / dt * (inv_mass1 / inv_mass2).min(1.0);
                self.object1.velocity.y -= disp_y * vel_from_correction / dt * (inv_mass1 / inv_mass2).min(1.0);
                self.object1.velocity.z -= disp_z * vel_from_correction / dt * (inv_mass1 / inv_mass2).min(1.0);
            }
        }

        // Apply position correction (mass-weighted)
        self.object1.position.x += position_correction * (inv_mass1 / total_inv_mass) * nx;
        self.object1.position.y += position_correction * (inv_mass1 / total_inv_mass) * ny;
        self.object1.position.z += position_correction * (inv_mass1 / total_inv_mass) * nz;
        self.object2.position.x -= position_correction * (inv_mass2 / total_inv_mass) * nx;
        self.object2.position.y -= position_correction * (inv_mass2 / total_inv_mass) * ny;
        self.object2.position.z -= position_correction * (inv_mass2 / total_inv_mass) * nz;

        // Now handle velocity constraint - remove radial velocity component if separating
        if radial_vel > 0.0 {
            // Objects are separating - apply impulse to stop radial separation
            // Baumgarte stabilization for remaining position error
            let bias = self.baumgarte * error / dt;
            let lambda = -(radial_vel + bias) / total_inv_mass;

            // Clamp impulse for stability
            let max_impulse = 0.1 / dt;
            let clamped_lambda = lambda.min(max_impulse);

            self.lambda += clamped_lambda;

            // Apply velocity corrections (mass-weighted, radial direction only)
            self.object1.velocity.x -= clamped_lambda * inv_mass1 * nx;
            self.object1.velocity.y -= clamped_lambda * inv_mass1 * ny;
            self.object1.velocity.z -= clamped_lambda * inv_mass1 * nz;
            self.object2.velocity.x += clamped_lambda * inv_mass2 * nx;
            self.object2.velocity.y += clamped_lambda * inv_mass2 * ny;
            self.object2.velocity.z += clamped_lambda * inv_mass2 * nz;
        }

        Ok(())
    }

    /// Calculates the current constraint error.
    ///
    /// # Returns
    ///
    /// 0 if rope is slack, positive value if rope is taut
    pub fn calculate_error(&self) -> f64 {
        let dx = self.object2.position.x - self.object1.position.x;
        let dy = self.object2.position.y - self.object1.position.y;
        let dz = self.object2.position.z - self.object1.position.z;
        let current_length = (dx * dx + dy * dy + dz * dz).sqrt();
        (current_length - self.max_length).max(0.0)
    }
}

impl Constraint3D for Rope3D {
    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        Rope3D::solve(self, dt)
    }

    fn calculate_error(&self) -> f64 {
        Rope3D::calculate_error(self)
    }

    fn get_lambda(&self) -> f64 {
        self.lambda
    }

    fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda;
    }
}

// ============================================================================
// RopeChain3D - Multi-segment rope simulation
// ============================================================================

use crate::models::{Axis3D, Velocity3D};

/// A particle in a rope chain.
///
/// Particles represent the nodes of the rope. Each particle has position,
/// velocity, and mass. The first particle (anchor) typically has infinite mass.
#[derive(Clone, Debug)]
pub struct RopeParticle {
    /// Position in 3D space
    pub position: Axis3D,
    /// Velocity in 3D space
    pub velocity: Velocity3D,
    /// Mass of the particle (use f64::INFINITY for fixed anchor points)
    pub mass: f64,
}

impl RopeParticle {
    /// Creates a new rope particle at the given position.
    pub fn new(x: f64, y: f64, z: f64, mass: f64) -> Self {
        Self {
            position: Axis3D { x, y, z },
            velocity: Velocity3D { x: 0.0, y: 0.0, z: 0.0 },
            mass,
        }
    }

    /// Creates a fixed anchor particle (infinite mass).
    pub fn anchor(x: f64, y: f64, z: f64) -> Self {
        Self::new(x, y, z, f64::INFINITY)
    }

    /// Returns the inverse mass (0 for infinite mass particles).
    pub fn inv_mass(&self) -> f64 {
        if self.mass.is_infinite() { 0.0 } else { 1.0 / self.mass }
    }
}

/// A multi-segment rope chain in 3D space.
///
/// RopeChain3D manages a chain of particles connected by distance constraints.
/// Unlike individual Rope3D constraints, this struct handles the state syncing
/// between segments automatically, making it suitable for rendering realistic
/// ropes and cables.
///
/// # Features
///
/// - Automatic particle state management across all segments
/// - Position-Based Dynamics (PBD) style constraint solving
/// - Multiple solver iterations for stability
/// - Gravity integration
/// - Easy position extraction for rendering
///
/// # Examples
///
/// ```rust,ignore
/// use rs_physics::constraints::RopeChain3D;
///
/// // Create a rope hanging from (0, 10, 0) with 10 segments, each 0.5 units long
/// let mut rope = RopeChain3D::new(
///     (0.0, 10.0, 0.0),  // anchor position
///     10,                 // number of segments
///     0.5,                // segment length
///     1.0,                // particle mass
/// ).expect("Valid rope chain");
///
/// // Simulation loop
/// let dt = 1.0 / 60.0;
/// for _ in 0..100 {
///     rope.apply_gravity(-9.81, dt);
///     rope.solve(dt, 10);  // 10 solver iterations
/// }
///
/// // Get positions for rendering
/// let positions = rope.get_particle_positions();
/// ```
#[derive(Debug)]
pub struct RopeChain3D {
    /// The particles making up the rope
    pub particles: Vec<RopeParticle>,
    /// The rest length of each segment (between consecutive particles)
    pub segment_lengths: Vec<f64>,
    /// Baumgarte stabilization factor (0.0 to 1.0, default 0.2)
    pub baumgarte: f64,
    /// Compliance factor for soft constraints (0 = rigid, higher = softer)
    pub compliance: f64,
}

impl RopeChain3D {
    /// Creates a new rope chain hanging vertically from an anchor point.
    ///
    /// # Arguments
    ///
    /// * `anchor` - Position of the fixed anchor point (x, y, z)
    /// * `num_segments` - Number of rope segments (creates num_segments + 1 particles)
    /// * `segment_length` - Length of each segment
    /// * `particle_mass` - Mass of each non-anchor particle
    ///
    /// # Returns
    ///
    /// * `Ok(RopeChain3D)` - Valid rope chain
    /// * `Err(PhysicsError)` - If parameters are invalid
    pub fn new(
        anchor: (f64, f64, f64),
        num_segments: usize,
        segment_length: f64,
        particle_mass: f64,
    ) -> Result<Self, PhysicsError> {
        if num_segments == 0 {
            return Err(PhysicsError::InvalidDimension);
        }
        if segment_length <= 0.0 {
            return Err(PhysicsError::InvalidDistance);
        }
        if particle_mass <= 0.0 || particle_mass.is_nan() {
            return Err(PhysicsError::InvalidMass);
        }

        let mut particles = Vec::with_capacity(num_segments + 1);
        let mut segment_lengths = Vec::with_capacity(num_segments);

        // Create anchor particle (infinite mass)
        particles.push(RopeParticle::anchor(anchor.0, anchor.1, anchor.2));

        // Create subsequent particles hanging vertically
        for i in 1..=num_segments {
            let y = anchor.1 - (i as f64) * segment_length;
            particles.push(RopeParticle::new(anchor.0, y, anchor.2, particle_mass));
            segment_lengths.push(segment_length);
        }

        Ok(Self {
            particles,
            segment_lengths,
            baumgarte: 0.2,
            compliance: 0.0,
        })
    }

    /// Creates a rope chain along a custom path of points.
    ///
    /// # Arguments
    ///
    /// * `points` - Positions of particles (first point is the anchor)
    /// * `particle_mass` - Mass of each non-anchor particle
    /// * `anchor_first` - If true, first particle is fixed (infinite mass)
    ///
    /// # Returns
    ///
    /// * `Ok(RopeChain3D)` - Valid rope chain
    /// * `Err(PhysicsError)` - If fewer than 2 points or invalid mass
    pub fn from_points(
        points: &[(f64, f64, f64)],
        particle_mass: f64,
        anchor_first: bool,
    ) -> Result<Self, PhysicsError> {
        if points.len() < 2 {
            return Err(PhysicsError::InvalidDimension);
        }
        if particle_mass <= 0.0 || particle_mass.is_nan() {
            return Err(PhysicsError::InvalidMass);
        }

        let mut particles = Vec::with_capacity(points.len());
        let mut segment_lengths = Vec::with_capacity(points.len() - 1);

        for (i, &(x, y, z)) in points.iter().enumerate() {
            let mass = if anchor_first && i == 0 {
                f64::INFINITY
            } else {
                particle_mass
            };
            particles.push(RopeParticle::new(x, y, z, mass));

            // Calculate segment length from previous particle
            if i > 0 {
                let prev = &points[i - 1];
                let dx = x - prev.0;
                let dy = y - prev.1;
                let dz = z - prev.2;
                let length = (dx * dx + dy * dy + dz * dz).sqrt();
                if length < 1e-10 {
                    return Err(PhysicsError::InvalidDistance);
                }
                segment_lengths.push(length);
            }
        }

        Ok(Self {
            particles,
            segment_lengths,
            baumgarte: 0.2,
            compliance: 0.0,
        })
    }

    /// Sets the Baumgarte stabilization factor.
    pub fn with_baumgarte(mut self, factor: f64) -> Self {
        self.baumgarte = factor.clamp(0.0, 1.0);
        self
    }

    /// Sets the compliance factor for softer constraints.
    pub fn with_compliance(mut self, compliance: f64) -> Self {
        self.compliance = compliance.max(0.0);
        self
    }

    /// Returns the number of particles in the rope.
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Returns the number of segments (constraints) in the rope.
    pub fn segment_count(&self) -> usize {
        self.segment_lengths.len()
    }

    /// Returns the total rest length of the rope.
    pub fn total_length(&self) -> f64 {
        self.segment_lengths.iter().sum()
    }

    /// Returns the current stretched length of the rope.
    pub fn current_length(&self) -> f64 {
        let mut length = 0.0;
        for i in 0..self.segment_lengths.len() {
            let p1 = &self.particles[i];
            let p2 = &self.particles[i + 1];
            let dx = p2.position.x - p1.position.x;
            let dy = p2.position.y - p1.position.y;
            let dz = p2.position.z - p1.position.z;
            length += (dx * dx + dy * dy + dz * dz).sqrt();
        }
        length
    }

    /// Gets the positions of all particles for rendering.
    ///
    /// Returns a vector of (x, y, z) tuples representing each particle position.
    pub fn get_particle_positions(&self) -> Vec<(f64, f64, f64)> {
        self.particles
            .iter()
            .map(|p| (p.position.x, p.position.y, p.position.z))
            .collect()
    }

    /// Gets mutable access to a particle by index.
    pub fn get_particle_mut(&mut self, index: usize) -> Option<&mut RopeParticle> {
        self.particles.get_mut(index)
    }

    /// Gets a reference to a particle by index.
    pub fn get_particle(&self, index: usize) -> Option<&RopeParticle> {
        self.particles.get(index)
    }

    /// Applies gravity to all non-fixed particles.
    ///
    /// # Arguments
    ///
    /// * `gravity` - Gravitational acceleration (typically -9.81 for downward)
    /// * `dt` - Timestep in seconds
    pub fn apply_gravity(&mut self, gravity: f64, dt: f64) {
        for particle in &mut self.particles {
            if particle.inv_mass() > 0.0 {
                particle.velocity.y += gravity * dt;
            }
        }
    }

    /// Applies an external force to a specific particle.
    ///
    /// # Arguments
    ///
    /// * `index` - Particle index
    /// * `force` - Force vector (fx, fy, fz)
    /// * `dt` - Timestep in seconds
    pub fn apply_force(&mut self, index: usize, force: (f64, f64, f64), dt: f64) {
        if let Some(particle) = self.particles.get_mut(index) {
            let inv_mass = particle.inv_mass();
            if inv_mass > 0.0 {
                particle.velocity.x += force.0 * inv_mass * dt;
                particle.velocity.y += force.1 * inv_mass * dt;
                particle.velocity.z += force.2 * inv_mass * dt;
            }
        }
    }

    /// Applies velocity damping to all particles.
    ///
    /// # Arguments
    ///
    /// * `damping` - Damping factor (0 = no damping, 1 = full stop)
    pub fn apply_damping(&mut self, damping: f64) {
        let factor = 1.0 - damping.clamp(0.0, 1.0);
        for particle in &mut self.particles {
            particle.velocity.x *= factor;
            particle.velocity.y *= factor;
            particle.velocity.z *= factor;
        }
    }

    /// Integrates particle positions based on velocities.
    ///
    /// Call this before solve() each frame.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    pub fn integrate(&mut self, dt: f64) {
        for particle in &mut self.particles {
            if particle.inv_mass() > 0.0 {
                particle.position.x += particle.velocity.x * dt;
                particle.position.y += particle.velocity.y * dt;
                particle.position.z += particle.velocity.z * dt;
            }
        }
    }

    /// Solves all distance constraints in the rope chain.
    ///
    /// Uses Position-Based Dynamics (PBD) style constraint solving with
    /// multiple iterations for stability.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    /// * `iterations` - Number of solver iterations (more = more stable but slower)
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Constraints solved successfully
    /// * `Err(PhysicsError)` - If an error occurred
    pub fn solve(&mut self, dt: f64, iterations: usize) -> Result<(), PhysicsError> {
        let iterations = iterations.max(1);

        for _ in 0..iterations {
            // Solve each segment constraint
            for i in 0..self.segment_lengths.len() {
                self.solve_segment(i, dt)?;
            }
        }

        Ok(())
    }

    /// Solves a single segment constraint between particles i and i+1.
    fn solve_segment(&mut self, segment_index: usize, dt: f64) -> Result<(), PhysicsError> {
        let rest_length = self.segment_lengths[segment_index];

        // Get positions
        let p1 = &self.particles[segment_index];
        let p2 = &self.particles[segment_index + 1];

        let dx = p2.position.x - p1.position.x;
        let dy = p2.position.y - p1.position.y;
        let dz = p2.position.z - p1.position.z;
        let current_length = (dx * dx + dy * dy + dz * dz).sqrt();

        if current_length < 1e-10 {
            return Ok(());
        }

        let error = current_length - rest_length;

        // For rope behavior, only constrain when stretched (error > 0)
        // Change to `error.abs()` for a stiff rod behavior
        if error <= 0.0 {
            return Ok(());
        }

        // Calculate inverse masses
        let inv_mass1 = self.particles[segment_index].inv_mass();
        let inv_mass2 = self.particles[segment_index + 1].inv_mass();
        let total_inv_mass = inv_mass1 + inv_mass2;

        if total_inv_mass < 1e-10 {
            return Ok(());
        }

        // Compliance (XPBD)
        let alpha = self.compliance / (dt * dt);
        let effective_mass = total_inv_mass + alpha;

        // Normalize direction
        let nx = dx / current_length;
        let ny = dy / current_length;
        let nz = dz / current_length;

        // Position correction magnitude
        let correction = error / effective_mass;

        // Apply position corrections (mass-weighted)
        let c1 = correction * (inv_mass1 / total_inv_mass);
        let c2 = correction * (inv_mass2 / total_inv_mass);

        // Apply to first particle
        if inv_mass1 > 0.0 {
            self.particles[segment_index].position.x += c1 * nx;
            self.particles[segment_index].position.y += c1 * ny;
            self.particles[segment_index].position.z += c1 * nz;
        }

        // Apply to second particle
        if inv_mass2 > 0.0 {
            self.particles[segment_index + 1].position.x -= c2 * nx;
            self.particles[segment_index + 1].position.y -= c2 * ny;
            self.particles[segment_index + 1].position.z -= c2 * nz;
        }

        // Velocity correction for stability
        // Calculate relative velocity along constraint direction
        let v1 = &self.particles[segment_index].velocity;
        let v2 = &self.particles[segment_index + 1].velocity;
        let rel_vx = v2.x - v1.x;
        let rel_vy = v2.y - v1.y;
        let rel_vz = v2.z - v1.z;
        let radial_vel = rel_vx * nx + rel_vy * ny + rel_vz * nz;

        // Only correct separating velocity
        if radial_vel > 0.0 {
            let bias = self.baumgarte * error / dt;
            let lambda = -(radial_vel + bias) / effective_mass;
            let lambda = lambda.min(0.5 / dt); // Clamp for stability

            // Apply velocity corrections
            if inv_mass1 > 0.0 {
                self.particles[segment_index].velocity.x -= lambda * inv_mass1 * nx;
                self.particles[segment_index].velocity.y -= lambda * inv_mass1 * ny;
                self.particles[segment_index].velocity.z -= lambda * inv_mass1 * nz;
            }
            if inv_mass2 > 0.0 {
                self.particles[segment_index + 1].velocity.x += lambda * inv_mass2 * nx;
                self.particles[segment_index + 1].velocity.y += lambda * inv_mass2 * ny;
                self.particles[segment_index + 1].velocity.z += lambda * inv_mass2 * nz;
            }
        }

        Ok(())
    }

    /// Performs a complete simulation step.
    ///
    /// This is a convenience method that performs:
    /// 1. Gravity integration
    /// 2. Position integration
    /// 3. Constraint solving
    /// 4. Optional damping
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    /// * `gravity` - Gravitational acceleration (typically -9.81)
    /// * `iterations` - Number of constraint solver iterations
    /// * `damping` - Optional velocity damping (0-1)
    pub fn step(&mut self, dt: f64, gravity: f64, iterations: usize, damping: Option<f64>) -> Result<(), PhysicsError> {
        // Apply gravity
        self.apply_gravity(gravity, dt);

        // Integrate positions
        self.integrate(dt);

        // Solve constraints
        self.solve(dt, iterations)?;

        // Apply damping if specified
        if let Some(d) = damping {
            self.apply_damping(d);
        }

        Ok(())
    }

    /// Calculates the maximum constraint error across all segments.
    pub fn max_error(&self) -> f64 {
        let mut max = 0.0_f64;
        for i in 0..self.segment_lengths.len() {
            let p1 = &self.particles[i];
            let p2 = &self.particles[i + 1];
            let dx = p2.position.x - p1.position.x;
            let dy = p2.position.y - p1.position.y;
            let dz = p2.position.z - p1.position.z;
            let current_length = (dx * dx + dy * dy + dz * dz).sqrt();
            let error = (current_length - self.segment_lengths[i]).max(0.0);
            max = max.max(error);
        }
        max
    }
}

// ============================================================================
// RopeChain2D - 2D version
// ============================================================================

/// A particle in a 2D rope chain.
#[derive(Clone, Debug)]
pub struct RopeParticle2D {
    /// Position in 2D space (x, y)
    pub position: (f64, f64),
    /// Velocity in 2D space (vx, vy)
    pub velocity: (f64, f64),
    /// Mass of the particle
    pub mass: f64,
}

impl RopeParticle2D {
    /// Creates a new rope particle at the given position.
    pub fn new(x: f64, y: f64, mass: f64) -> Self {
        Self {
            position: (x, y),
            velocity: (0.0, 0.0),
            mass,
        }
    }

    /// Creates a fixed anchor particle (infinite mass).
    pub fn anchor(x: f64, y: f64) -> Self {
        Self::new(x, y, f64::INFINITY)
    }

    /// Returns the inverse mass.
    pub fn inv_mass(&self) -> f64 {
        if self.mass.is_infinite() { 0.0 } else { 1.0 / self.mass }
    }
}

/// A multi-segment rope chain in 2D space.
///
/// See [`RopeChain3D`] for detailed documentation. This is the 2D equivalent.
pub struct RopeChain2D {
    /// The particles making up the rope
    pub particles: Vec<RopeParticle2D>,
    /// The rest length of each segment
    pub segment_lengths: Vec<f64>,
    /// Baumgarte stabilization factor
    pub baumgarte: f64,
    /// Compliance factor
    pub compliance: f64,
}

impl RopeChain2D {
    /// Creates a new rope chain hanging vertically from an anchor point.
    pub fn new(
        anchor: (f64, f64),
        num_segments: usize,
        segment_length: f64,
        particle_mass: f64,
    ) -> Result<Self, PhysicsError> {
        if num_segments == 0 {
            return Err(PhysicsError::InvalidDimension);
        }
        if segment_length <= 0.0 {
            return Err(PhysicsError::InvalidDistance);
        }
        if particle_mass <= 0.0 || particle_mass.is_nan() {
            return Err(PhysicsError::InvalidMass);
        }

        let mut particles = Vec::with_capacity(num_segments + 1);
        let mut segment_lengths = Vec::with_capacity(num_segments);

        particles.push(RopeParticle2D::anchor(anchor.0, anchor.1));

        for i in 1..=num_segments {
            let y = anchor.1 - (i as f64) * segment_length;
            particles.push(RopeParticle2D::new(anchor.0, y, particle_mass));
            segment_lengths.push(segment_length);
        }

        Ok(Self {
            particles,
            segment_lengths,
            baumgarte: 0.2,
            compliance: 0.0,
        })
    }

    /// Creates a rope chain along a custom path of points.
    pub fn from_points(
        points: &[(f64, f64)],
        particle_mass: f64,
        anchor_first: bool,
    ) -> Result<Self, PhysicsError> {
        if points.len() < 2 {
            return Err(PhysicsError::InvalidDimension);
        }
        if particle_mass <= 0.0 || particle_mass.is_nan() {
            return Err(PhysicsError::InvalidMass);
        }

        let mut particles = Vec::with_capacity(points.len());
        let mut segment_lengths = Vec::with_capacity(points.len() - 1);

        for (i, &(x, y)) in points.iter().enumerate() {
            let mass = if anchor_first && i == 0 { f64::INFINITY } else { particle_mass };
            particles.push(RopeParticle2D::new(x, y, mass));

            if i > 0 {
                let prev = &points[i - 1];
                let dx = x - prev.0;
                let dy = y - prev.1;
                let length = (dx * dx + dy * dy).sqrt();
                if length < 1e-10 {
                    return Err(PhysicsError::InvalidDistance);
                }
                segment_lengths.push(length);
            }
        }

        Ok(Self {
            particles,
            segment_lengths,
            baumgarte: 0.2,
            compliance: 0.0,
        })
    }

    /// Sets the Baumgarte stabilization factor.
    pub fn with_baumgarte(mut self, factor: f64) -> Self {
        self.baumgarte = factor.clamp(0.0, 1.0);
        self
    }

    /// Sets the compliance factor.
    pub fn with_compliance(mut self, compliance: f64) -> Self {
        self.compliance = compliance.max(0.0);
        self
    }

    /// Returns the number of particles.
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Returns the number of segments.
    pub fn segment_count(&self) -> usize {
        self.segment_lengths.len()
    }

    /// Gets positions for rendering.
    pub fn get_particle_positions(&self) -> Vec<(f64, f64)> {
        self.particles.iter().map(|p| p.position).collect()
    }

    /// Applies gravity.
    pub fn apply_gravity(&mut self, gravity: f64, dt: f64) {
        for particle in &mut self.particles {
            if particle.inv_mass() > 0.0 {
                particle.velocity.1 += gravity * dt;
            }
        }
    }

    /// Integrates positions.
    pub fn integrate(&mut self, dt: f64) {
        for particle in &mut self.particles {
            if particle.inv_mass() > 0.0 {
                particle.position.0 += particle.velocity.0 * dt;
                particle.position.1 += particle.velocity.1 * dt;
            }
        }
    }

    /// Solves constraints.
    pub fn solve(&mut self, dt: f64, iterations: usize) -> Result<(), PhysicsError> {
        let iterations = iterations.max(1);

        for _ in 0..iterations {
            for i in 0..self.segment_lengths.len() {
                self.solve_segment(i, dt)?;
            }
        }

        Ok(())
    }

    fn solve_segment(&mut self, segment_index: usize, dt: f64) -> Result<(), PhysicsError> {
        let rest_length = self.segment_lengths[segment_index];

        let p1 = &self.particles[segment_index];
        let p2 = &self.particles[segment_index + 1];

        let dx = p2.position.0 - p1.position.0;
        let dy = p2.position.1 - p1.position.1;
        let current_length = (dx * dx + dy * dy).sqrt();

        if current_length < 1e-10 {
            return Ok(());
        }

        let error = current_length - rest_length;

        if error <= 0.0 {
            return Ok(());
        }

        let inv_mass1 = self.particles[segment_index].inv_mass();
        let inv_mass2 = self.particles[segment_index + 1].inv_mass();
        let total_inv_mass = inv_mass1 + inv_mass2;

        if total_inv_mass < 1e-10 {
            return Ok(());
        }

        let alpha = self.compliance / (dt * dt);
        let effective_mass = total_inv_mass + alpha;

        let nx = dx / current_length;
        let ny = dy / current_length;

        let correction = error / effective_mass;
        let c1 = correction * (inv_mass1 / total_inv_mass);
        let c2 = correction * (inv_mass2 / total_inv_mass);

        if inv_mass1 > 0.0 {
            self.particles[segment_index].position.0 += c1 * nx;
            self.particles[segment_index].position.1 += c1 * ny;
        }

        if inv_mass2 > 0.0 {
            self.particles[segment_index + 1].position.0 -= c2 * nx;
            self.particles[segment_index + 1].position.1 -= c2 * ny;
        }

        let v1 = &self.particles[segment_index].velocity;
        let v2 = &self.particles[segment_index + 1].velocity;
        let rel_vx = v2.0 - v1.0;
        let rel_vy = v2.1 - v1.1;
        let radial_vel = rel_vx * nx + rel_vy * ny;

        if radial_vel > 0.0 {
            let bias = self.baumgarte * error / dt;
            let lambda = -(radial_vel + bias) / effective_mass;
            let lambda = lambda.min(0.5 / dt);

            if inv_mass1 > 0.0 {
                self.particles[segment_index].velocity.0 -= lambda * inv_mass1 * nx;
                self.particles[segment_index].velocity.1 -= lambda * inv_mass1 * ny;
            }
            if inv_mass2 > 0.0 {
                self.particles[segment_index + 1].velocity.0 += lambda * inv_mass2 * nx;
                self.particles[segment_index + 1].velocity.1 += lambda * inv_mass2 * ny;
            }
        }

        Ok(())
    }

    /// Performs a complete simulation step.
    pub fn step(&mut self, dt: f64, gravity: f64, iterations: usize, damping: Option<f64>) -> Result<(), PhysicsError> {
        self.apply_gravity(gravity, dt);
        self.integrate(dt);
        self.solve(dt, iterations)?;

        if let Some(d) = damping {
            let factor = 1.0 - d.clamp(0.0, 1.0);
            for particle in &mut self.particles {
                particle.velocity.0 *= factor;
                particle.velocity.1 *= factor;
            }
        }

        Ok(())
    }

    /// Calculates the maximum constraint error.
    pub fn max_error(&self) -> f64 {
        let mut max = 0.0_f64;
        for i in 0..self.segment_lengths.len() {
            let p1 = &self.particles[i];
            let p2 = &self.particles[i + 1];
            let dx = p2.position.0 - p1.position.0;
            let dy = p2.position.1 - p1.position.1;
            let current_length = (dx * dx + dy * dy).sqrt();
            let error = (current_length - self.segment_lengths[i]).max(0.0);
            max = max.max(error);
        }
        max
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Shape2DCollider;

    // ============================================================================
    // 2D Rope Tests
    // ============================================================================

    #[test]
    fn test_rope_2d_creation() {
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (3.0, 0.0), Shape2DCollider::Circle(1.0));
        let rope = Rope2D::new(obj1, obj2, 5.0);
        assert!(rope.is_ok());
    }

    #[test]
    fn test_rope_2d_invalid_length() {
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (3.0, 0.0), Shape2DCollider::Circle(1.0));
        let rope = Rope2D::new(obj1, obj2, 0.0);
        assert!(rope.is_err());

        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (3.0, 0.0), Shape2DCollider::Circle(1.0));
        let rope = Rope2D::new(obj1, obj2, -1.0);
        assert!(rope.is_err());
    }

    #[test]
    fn test_rope_2d_slack_no_correction() {
        // Rope is slack (objects closer than max_length)
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (3.0, 0.0), Shape2DCollider::Circle(1.0));
        let mut rope = Rope2D::new(obj1, obj2, 5.0).unwrap();

        let initial_pos1 = rope.object1.position.x;
        let initial_pos2 = rope.object2.position.x;

        rope.solve(0.1).unwrap();

        // No correction should be applied when slack
        assert!((rope.object1.position.x - initial_pos1).abs() < 1e-10);
        assert!((rope.object2.position.x - initial_pos2).abs() < 1e-10);
        assert!(!rope.is_taut());
    }

    #[test]
    fn test_rope_2d_taut_correction() {
        // Rope is taut (objects further than max_length)
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (7.0, 0.0), Shape2DCollider::Circle(1.0));
        let mut rope = Rope2D::new(obj1, obj2, 5.0).unwrap();

        assert!(rope.is_taut());
        let initial_error = rope.calculate_error();

        rope.solve(0.1).unwrap();

        let final_error = rope.calculate_error();
        assert!(final_error < initial_error, "Error should decrease after solving");
    }

    #[test]
    fn test_rope_2d_at_max_length() {
        // Rope exactly at max_length (should be considered taut)
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (5.0, 0.0), Shape2DCollider::Circle(1.0));
        let rope = Rope2D::new(obj1, obj2, 5.0).unwrap();

        assert!(rope.is_taut());
        assert!(rope.calculate_error() < 1e-10, "Error should be near zero at exact max_length");
    }

    #[test]
    fn test_rope_2d_diagonal() {
        // Rope stretched diagonally
        let obj1 = ObjectIn2D::with_shape(1.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));
        let obj2 = ObjectIn2D::with_shape(1.0, (6.0, 8.0), Shape2DCollider::Circle(1.0)); // Distance = 10
        let mut rope = Rope2D::new(obj1, obj2, 5.0).unwrap();

        assert!(rope.is_taut());
        let initial_error = rope.calculate_error();
        assert!((initial_error - 5.0).abs() < 1e-6, "Error should be 5.0 (10 - 5)");

        rope.solve(0.1).unwrap();

        let final_error = rope.calculate_error();
        assert!(final_error < initial_error, "Error should decrease after solving");
    }

    // ============================================================================
    // 3D Rope Tests
    // ============================================================================

    fn make_3d_object_at(x: f64, y: f64, z: f64) -> ObjectIn3D {
        use crate::models::{Axis3D, Velocity3D};
        ObjectIn3D {
            mass: 1.0,
            velocity: Velocity3D { x: 0.0, y: 0.0, z: 0.0 },
            position: Axis3D { x, y, z },
            forces: Vec::new(),
            material: None,
        }
    }

    #[test]
    fn test_rope_3d_creation() {
        let obj1 = make_3d_object_at(0.0, 0.0, 0.0);
        let obj2 = make_3d_object_at(3.0, 0.0, 0.0);
        let rope = Rope3D::new(obj1, obj2, 5.0);
        assert!(rope.is_ok());
    }

    #[test]
    fn test_rope_3d_slack_no_correction() {
        let obj1 = make_3d_object_at(0.0, 0.0, 0.0);
        let obj2 = make_3d_object_at(3.0, 0.0, 0.0);
        let mut rope = Rope3D::new(obj1, obj2, 5.0).unwrap();

        let initial_pos1 = rope.object1.position.x;
        let initial_pos2 = rope.object2.position.x;

        rope.solve(0.1).unwrap();

        // No correction should be applied when slack
        assert!((rope.object1.position.x - initial_pos1).abs() < 1e-10);
        assert!((rope.object2.position.x - initial_pos2).abs() < 1e-10);
        assert!(!rope.is_taut());
    }

    #[test]
    fn test_rope_3d_taut_correction() {
        let obj1 = make_3d_object_at(0.0, 0.0, 0.0);
        let obj2 = make_3d_object_at(7.0, 0.0, 0.0);
        let mut rope = Rope3D::new(obj1, obj2, 5.0).unwrap();

        assert!(rope.is_taut());
        let initial_error = rope.calculate_error();

        rope.solve(0.1).unwrap();

        let final_error = rope.calculate_error();
        assert!(final_error < initial_error, "Error should decrease after solving");
    }

    #[test]
    fn test_rope_3d_diagonal_xyz() {
        // Rope stretched diagonally in 3D space
        let obj1 = make_3d_object_at(0.0, 0.0, 0.0);
        let obj2 = make_3d_object_at(6.0, 6.0, 3.0); // Distance = 9
        let mut rope = Rope3D::new(obj1, obj2, 5.0).unwrap();

        assert!(rope.is_taut());
        let initial_error = rope.calculate_error();
        assert!(initial_error > 3.0, "Error should be > 3.0");

        rope.solve(0.1).unwrap();

        let final_error = rope.calculate_error();
        assert!(final_error < initial_error, "Error should decrease after solving");
    }

    // ============================================================================
    // RopeChain3D Tests
    // ============================================================================

    #[test]
    fn test_rope_chain_3d_creation() {
        let rope = RopeChain3D::new((0.0, 10.0, 0.0), 5, 1.0, 1.0);
        assert!(rope.is_ok());
        let rope = rope.unwrap();
        assert_eq!(rope.particle_count(), 6); // 5 segments + 1 anchor
        assert_eq!(rope.segment_count(), 5);
        assert!((rope.total_length() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_rope_chain_3d_invalid_params() {
        // Zero segments
        let rope = RopeChain3D::new((0.0, 10.0, 0.0), 0, 1.0, 1.0);
        assert!(rope.is_err());

        // Invalid segment length
        let rope = RopeChain3D::new((0.0, 10.0, 0.0), 5, 0.0, 1.0);
        assert!(rope.is_err());

        // Invalid mass
        let rope = RopeChain3D::new((0.0, 10.0, 0.0), 5, 1.0, 0.0);
        assert!(rope.is_err());
    }

    #[test]
    fn test_rope_chain_3d_from_points() {
        let points = vec![
            (0.0, 10.0, 0.0),
            (1.0, 9.0, 0.0),
            (2.0, 8.0, 0.0),
            (3.0, 7.0, 0.0),
        ];
        let rope = RopeChain3D::from_points(&points, 1.0, true);
        assert!(rope.is_ok());
        let rope = rope.unwrap();
        assert_eq!(rope.particle_count(), 4);
        assert_eq!(rope.segment_count(), 3);
        // First particle should be anchor (infinite mass)
        assert!(rope.particles[0].mass.is_infinite());
        // Other particles should have mass 1.0
        assert!((rope.particles[1].mass - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rope_chain_3d_from_points_no_anchor() {
        let points = vec![(0.0, 10.0, 0.0), (1.0, 9.0, 0.0)];
        let rope = RopeChain3D::from_points(&points, 2.0, false).unwrap();
        // All particles should have mass 2.0
        assert!((rope.particles[0].mass - 2.0).abs() < 1e-10);
        assert!((rope.particles[1].mass - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_rope_chain_3d_positions() {
        let rope = RopeChain3D::new((0.0, 10.0, 0.0), 3, 2.0, 1.0).unwrap();
        let positions = rope.get_particle_positions();
        assert_eq!(positions.len(), 4);
        assert!((positions[0].0 - 0.0).abs() < 1e-10);
        assert!((positions[0].1 - 10.0).abs() < 1e-10);
        assert!((positions[1].1 - 8.0).abs() < 1e-10);
        assert!((positions[2].1 - 6.0).abs() < 1e-10);
        assert!((positions[3].1 - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rope_chain_3d_gravity() {
        let mut rope = RopeChain3D::new((0.0, 10.0, 0.0), 3, 1.0, 1.0).unwrap();
        let dt = 0.1;
        let gravity = -9.81;

        rope.apply_gravity(gravity, dt);

        // Anchor (first particle) should not move
        assert!((rope.particles[0].velocity.y - 0.0).abs() < 1e-10);

        // Other particles should have velocity changed by gravity
        for particle in &rope.particles[1..] {
            assert!((particle.velocity.y - (gravity * dt)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rope_chain_3d_integration() {
        let mut rope = RopeChain3D::new((0.0, 10.0, 0.0), 2, 1.0, 1.0).unwrap();
        rope.particles[1].velocity.y = -1.0;
        rope.particles[2].velocity.y = -1.0;

        let dt = 0.1;
        rope.integrate(dt);

        // Anchor should not move
        assert!((rope.particles[0].position.y - 10.0).abs() < 1e-10);
        // Other particles should move
        assert!((rope.particles[1].position.y - (9.0 - 0.1)).abs() < 1e-10);
        assert!((rope.particles[2].position.y - (8.0 - 0.1)).abs() < 1e-10);
    }

    #[test]
    fn test_rope_chain_3d_constraint_solving() {
        let mut rope = RopeChain3D::new((0.0, 10.0, 0.0), 3, 1.0, 1.0).unwrap();

        // Stretch the rope by moving the last particle down
        rope.particles[3].position.y = 5.0; // Should be 7.0 (10 - 3)

        let initial_error = rope.max_error();
        assert!(initial_error > 0.0, "Should have constraint error");

        rope.solve(0.1, 10).unwrap();

        let final_error = rope.max_error();
        assert!(final_error < initial_error, "Error should decrease after solving");
    }

    #[test]
    fn test_rope_chain_3d_step() {
        let mut rope = RopeChain3D::new((0.0, 10.0, 0.0), 5, 0.5, 0.5).unwrap();

        let dt = 1.0 / 60.0;
        let gravity = -9.81;

        // Run simulation for a few steps
        for _ in 0..10 {
            rope.step(dt, gravity, 10, None).unwrap();
        }

        // Anchor should still be at original position
        assert!((rope.particles[0].position.x - 0.0).abs() < 1e-10);
        assert!((rope.particles[0].position.y - 10.0).abs() < 1e-10);
        assert!((rope.particles[0].position.z - 0.0).abs() < 1e-10);

        // Other particles should have moved down
        for particle in &rope.particles[1..] {
            assert!(particle.position.y < 10.0, "Particles should fall under gravity");
        }
    }

    #[test]
    fn test_rope_chain_3d_damping() {
        let mut rope = RopeChain3D::new((0.0, 10.0, 0.0), 2, 1.0, 1.0).unwrap();
        rope.particles[1].velocity.y = -10.0;

        rope.apply_damping(0.1);

        // Velocity should be reduced by 10%
        assert!((rope.particles[1].velocity.y - (-9.0)).abs() < 1e-10);
    }

    #[test]
    fn test_rope_chain_3d_apply_force() {
        let mut rope = RopeChain3D::new((0.0, 10.0, 0.0), 2, 1.0, 1.0).unwrap();

        let force = (10.0, 0.0, 0.0);
        let dt = 0.1;
        rope.apply_force(1, force, dt);

        // F = ma, a = F/m = 10/1 = 10, v = a*t = 10*0.1 = 1
        assert!((rope.particles[1].velocity.x - 1.0).abs() < 1e-10);

        // Force on anchor should have no effect
        rope.apply_force(0, force, dt);
        assert!((rope.particles[0].velocity.x - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_rope_chain_3d_current_length() {
        let rope = RopeChain3D::new((0.0, 10.0, 0.0), 3, 1.0, 1.0).unwrap();
        // Initial state: particles are spaced exactly segment_length apart
        assert!((rope.current_length() - 3.0).abs() < 1e-10);
    }

    // ============================================================================
    // RopeChain2D Tests
    // ============================================================================

    #[test]
    fn test_rope_chain_2d_creation() {
        let rope = RopeChain2D::new((0.0, 10.0), 5, 1.0, 1.0);
        assert!(rope.is_ok());
        let rope = rope.unwrap();
        assert_eq!(rope.particle_count(), 6);
        assert_eq!(rope.segment_count(), 5);
    }

    #[test]
    fn test_rope_chain_2d_invalid_params() {
        let rope = RopeChain2D::new((0.0, 10.0), 0, 1.0, 1.0);
        assert!(rope.is_err());

        let rope = RopeChain2D::new((0.0, 10.0), 5, -1.0, 1.0);
        assert!(rope.is_err());

        let rope = RopeChain2D::new((0.0, 10.0), 5, 1.0, -1.0);
        assert!(rope.is_err());
    }

    #[test]
    fn test_rope_chain_2d_from_points() {
        let points = vec![(0.0, 10.0), (1.0, 9.0), (2.0, 8.0)];
        let rope = RopeChain2D::from_points(&points, 1.0, true);
        assert!(rope.is_ok());
        let rope = rope.unwrap();
        assert_eq!(rope.particle_count(), 3);
        assert!(rope.particles[0].mass.is_infinite());
    }

    #[test]
    fn test_rope_chain_2d_positions() {
        let rope = RopeChain2D::new((0.0, 10.0), 3, 2.0, 1.0).unwrap();
        let positions = rope.get_particle_positions();
        assert_eq!(positions.len(), 4);
        assert!((positions[0].1 - 10.0).abs() < 1e-10);
        assert!((positions[1].1 - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_rope_chain_2d_step() {
        let mut rope = RopeChain2D::new((0.0, 10.0), 5, 0.5, 0.5).unwrap();

        let dt = 1.0 / 60.0;
        let gravity = -9.81;

        for _ in 0..10 {
            rope.step(dt, gravity, 10, None).unwrap();
        }

        // Anchor should be stationary
        assert!((rope.particles[0].position.1 - 10.0).abs() < 1e-10);

        // Other particles should fall
        for particle in &rope.particles[1..] {
            assert!(particle.position.1 < 10.0);
        }
    }

    #[test]
    fn test_rope_chain_2d_max_error() {
        let mut rope = RopeChain2D::new((0.0, 10.0), 3, 1.0, 1.0).unwrap();

        // Stretch the rope
        rope.particles[3].position.1 = 5.0;

        let error = rope.max_error();
        assert!(error > 0.0);
    }

    // ============================================================================
    // RopeParticle Tests
    // ============================================================================

    #[test]
    fn test_rope_particle_inv_mass() {
        let particle = RopeParticle::new(0.0, 0.0, 0.0, 2.0);
        assert!((particle.inv_mass() - 0.5).abs() < 1e-10);

        let anchor = RopeParticle::anchor(0.0, 0.0, 0.0);
        assert!((anchor.inv_mass() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_rope_particle_2d_inv_mass() {
        let particle = RopeParticle2D::new(0.0, 0.0, 4.0);
        assert!((particle.inv_mass() - 0.25).abs() < 1e-10);

        let anchor = RopeParticle2D::anchor(0.0, 0.0);
        assert!((anchor.inv_mass() - 0.0).abs() < 1e-10);
    }
}
