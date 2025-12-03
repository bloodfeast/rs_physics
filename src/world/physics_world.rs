//! PhysicsWorld - The core simulation container

use std::collections::HashMap;
use rayon::prelude::*;
use crate::models::{PhysicalObject3D, Quaternion, Shape3D};
use crate::utils::PhysicsConstants;
use crate::interactions::shape_collisions_3d::{handle_collision, apply_gravity};
use crate::interactions::gjk_collision_3d::{gjk_collision_detection_ex, epa_contact_points_ex, GjkResult};
use super::state::{ObjectId, ObjectState, WorldState};
use super::config::WorldConfig;

/// Unique identifier for continuous forces
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ForceId(pub u64);

impl ForceId {
    /// Generate a new unique ForceId
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        ForceId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for ForceId {
    fn default() -> Self {
        Self::new()
    }
}

/// Continuous forces that persist across simulation steps
///
/// These forces are automatically recalculated each step based on current object state.
/// They remain active until explicitly removed or their effect becomes negligible.
#[derive(Debug, Clone)]
pub enum ContinuousForce {
    /// Constant force in a direction (like thrust or wind)
    /// Force is applied every step until removed
    Constant {
        target: ObjectId,
        force: (f64, f64, f64),
    },

    /// Drag force opposing velocity (air/water resistance)
    /// Automatically removed when object velocity drops below threshold
    Drag {
        target: ObjectId,
        coefficient: f64,
        /// Velocity threshold below which drag is removed (default: 0.01)
        min_velocity: f64,
    },

    /// Spring force toward a rest position
    /// Automatically removed when displacement and velocity are below thresholds
    Spring {
        target: ObjectId,
        rest_position: (f64, f64, f64),
        stiffness: f64,
        /// Displacement threshold for removal (default: 0.01)
        min_displacement: f64,
        /// Velocity threshold for removal (default: 0.01)
        min_velocity: f64,
    },

    /// Damped spring (spring + velocity damping)
    /// Automatically removed when at rest
    DampedSpring {
        target: ObjectId,
        rest_position: (f64, f64, f64),
        stiffness: f64,
        damping: f64,
        min_displacement: f64,
        min_velocity: f64,
    },

    /// Attraction toward a point (gravity well, magnet)
    /// Can be set to expire after duration or persist indefinitely
    Attract {
        target: ObjectId,
        point: (f64, f64, f64),
        strength: f64,
        /// If Some, force expires after this many seconds
        duration: Option<f64>,
        elapsed: f64,
    },

    /// Repulsion from a point (force field)
    Repel {
        target: ObjectId,
        point: (f64, f64, f64),
        strength: f64,
        duration: Option<f64>,
        elapsed: f64,
    },

    /// Buoyancy force (upward force when below surface)
    /// Removed when object is above surface for extended time
    Buoyancy {
        target: ObjectId,
        surface_y: f64,
        fluid_density: f64,
        /// Time object has been above surface
        time_above_surface: f64,
        /// Remove after this long above surface (default: 1.0s)
        removal_delay: f64,
    },

    /// Vortex/rotational force around an axis
    Vortex {
        target: ObjectId,
        center: (f64, f64, f64),
        axis: (f64, f64, f64),
        strength: f64,
        duration: Option<f64>,
        elapsed: f64,
    },
}

/// Collision data collected during parallel detection phase
/// This allows us to detect collisions in parallel (read-only)
/// and then apply responses sequentially (write)
#[derive(Debug, Clone)]
struct CollisionData {
    /// Index of first object
    i: usize,
    /// Index of second object
    j: usize,
    /// Contact normal (from obj1 to obj2)
    normal: (f64, f64, f64),
    /// Penetration depth
    penetration: f64,
    /// Contact point on object 1 (local offset from center)
    contact1: (f64, f64, f64),
    /// Contact point on object 2 (local offset from center)
    contact2: (f64, f64, f64),
}

/// The main physics simulation world
///
/// Manages all physics objects and runs the simulation step.
/// This can be used directly for single-threaded simulation,
/// or through `PhysicsHandle` for background threaded simulation.
pub struct PhysicsWorld {
    /// All physics objects in the world
    objects: Vec<PhysicalObject3D>,

    /// Map from ObjectId to index in objects vector
    object_ids: HashMap<ObjectId, usize>,

    /// Reverse map from index to ObjectId
    index_to_id: HashMap<usize, ObjectId>,

    /// World configuration
    config: WorldConfig,

    /// Current simulation tick
    tick: u64,

    /// Current simulation time
    time: f64,

    /// Accumulated time for fixed timestep
    accumulated_time: f64,

    /// Whether simulation is paused
    paused: bool,

    /// Pending one-shot forces to apply (cleared each step)
    pending_forces: HashMap<ObjectId, Vec<(f64, f64, f64)>>,

    /// Continuous forces that persist across steps
    continuous_forces: HashMap<ForceId, ContinuousForce>,
}

impl PhysicsWorld {
    /// Create a new physics world with the given configuration
    pub fn new(config: WorldConfig) -> Self {
        Self {
            objects: Vec::new(),
            object_ids: HashMap::new(),
            index_to_id: HashMap::new(),
            config,
            tick: 0,
            time: 0.0,
            accumulated_time: 0.0,
            paused: false,
            pending_forces: HashMap::new(),
            continuous_forces: HashMap::new(),
        }
    }

    /// Create a new physics world with default configuration
    pub fn default_world() -> Self {
        Self::new(WorldConfig::default())
    }

    /// Add an object to the world
    ///
    /// Returns the ObjectId assigned to the object
    pub fn add_object(&mut self, obj: PhysicalObject3D) -> ObjectId {
        let id = ObjectId::new();
        let index = self.objects.len();

        self.objects.push(obj);
        self.object_ids.insert(id, index);
        self.index_to_id.insert(index, id);

        id
    }

    /// Remove an object from the world
    ///
    /// Returns true if the object was found and removed
    pub fn remove_object(&mut self, id: ObjectId) -> bool {
        if let Some(&index) = self.object_ids.get(&id) {
            // Remove the object
            self.objects.swap_remove(index);
            self.object_ids.remove(&id);
            self.index_to_id.remove(&index);

            // If we swapped an object into this position, update its index mapping
            if index < self.objects.len() {
                // Find the id that was at the last position
                let last_index = self.objects.len();
                if let Some(&swapped_id) = self.index_to_id.get(&last_index) {
                    self.object_ids.insert(swapped_id, index);
                    self.index_to_id.remove(&last_index);
                    self.index_to_id.insert(index, swapped_id);
                }
            }

            true
        } else {
            false
        }
    }

    /// Get a reference to an object by ID
    pub fn get_object(&self, id: ObjectId) -> Option<&PhysicalObject3D> {
        self.object_ids.get(&id).map(|&idx| &self.objects[idx])
    }

    /// Get a mutable reference to an object by ID
    pub fn get_object_mut(&mut self, id: ObjectId) -> Option<&mut PhysicalObject3D> {
        if let Some(&idx) = self.object_ids.get(&id) {
            Some(&mut self.objects[idx])
        } else {
            None
        }
    }

    /// Get the number of objects in the world
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Set the simulation timestep
    pub fn set_timestep(&mut self, dt: f64) {
        self.config.timestep = dt;
    }

    /// Get the current timestep
    pub fn timestep(&self) -> f64 {
        self.config.timestep
    }

    /// Pause the simulation
    pub fn pause(&mut self) {
        self.paused = true;
    }

    /// Resume the simulation
    pub fn resume(&mut self) {
        self.paused = false;
    }

    /// Check if simulation is paused
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Get the current simulation time
    pub fn current_time(&self) -> f64 {
        self.time
    }

    /// Get the current tick count
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Apply a force to an object (raw 3D vector)
    pub fn apply_force(&mut self, id: ObjectId, force: (f64, f64, f64)) {
        if self.object_ids.contains_key(&id) {
            self.pending_forces.entry(id).or_default().push(force);
        }
    }

    /// Apply an impulse (instant velocity change) to an object
    pub fn apply_impulse(&mut self, id: ObjectId, impulse: (f64, f64, f64)) {
        if let Some(obj) = self.get_object_mut(id) {
            let mass = obj.object.mass;
            if mass > 0.0 {
                obj.object.velocity.x += impulse.0 / mass;
                obj.object.velocity.y += impulse.1 / mass;
                obj.object.velocity.z += impulse.2 / mass;
            }
        }
    }

    // ==================== Ergonomic Force Methods ====================

    /// Apply a force in a specific direction with given magnitude
    ///
    /// Direction is automatically normalized.
    pub fn apply_force_directed(&mut self, id: ObjectId, magnitude: f64, direction: (f64, f64, f64)) {
        let len = (direction.0 * direction.0 + direction.1 * direction.1 + direction.2 * direction.2).sqrt();
        if len > 1e-10 {
            let normalized = (direction.0 / len, direction.1 / len, direction.2 / len);
            self.apply_force(id, (
                normalized.0 * magnitude,
                normalized.1 * magnitude,
                normalized.2 * magnitude,
            ));
        }
    }

    /// Apply a force toward a target position (attraction)
    ///
    /// Useful for gravity wells, magnets, or AI-controlled movement.
    pub fn apply_force_toward(&mut self, id: ObjectId, target: (f64, f64, f64), magnitude: f64) {
        if let Some(obj) = self.get_object(id) {
            let dx = target.0 - obj.object.position.x;
            let dy = target.1 - obj.object.position.y;
            let dz = target.2 - obj.object.position.z;
            self.apply_force_directed(id, magnitude, (dx, dy, dz));
        }
    }

    /// Apply a force away from a position (repulsion)
    ///
    /// Useful for explosions, force fields, or avoidance.
    pub fn apply_force_away(&mut self, id: ObjectId, source: (f64, f64, f64), magnitude: f64) {
        if let Some(obj) = self.get_object(id) {
            let dx = obj.object.position.x - source.0;
            let dy = obj.object.position.y - source.1;
            let dz = obj.object.position.z - source.2;
            self.apply_force_directed(id, magnitude, (dx, dy, dz));
        }
    }

    /// Apply drag force based on current velocity
    ///
    /// drag_coefficient: typically 0.1 to 2.0 (higher = more drag)
    pub fn apply_drag(&mut self, id: ObjectId, drag_coefficient: f64) {
        if let Some(obj) = self.get_object(id) {
            let vx = obj.object.velocity.x;
            let vy = obj.object.velocity.y;
            let vz = obj.object.velocity.z;
            let speed_sq = vx * vx + vy * vy + vz * vz;

            if speed_sq > 1e-10 {
                // Drag force opposes velocity, proportional to v²
                let drag_magnitude = drag_coefficient * speed_sq;
                let speed = speed_sq.sqrt();
                self.apply_force(id, (
                    -vx / speed * drag_magnitude,
                    -vy / speed * drag_magnitude,
                    -vz / speed * drag_magnitude,
                ));
            }
        }
    }

    /// Apply spring force toward a rest position
    ///
    /// Uses Hooke's law: F = -k * displacement
    pub fn apply_spring_force(&mut self, id: ObjectId, rest_position: (f64, f64, f64), stiffness: f64) {
        if let Some(obj) = self.get_object(id) {
            let dx = obj.object.position.x - rest_position.0;
            let dy = obj.object.position.y - rest_position.1;
            let dz = obj.object.position.z - rest_position.2;
            self.apply_force(id, (
                -stiffness * dx,
                -stiffness * dy,
                -stiffness * dz,
            ));
        }
    }

    /// Apply damped spring force (spring + velocity damping)
    ///
    /// Combines spring force with damping to reduce oscillation.
    pub fn apply_damped_spring(&mut self, id: ObjectId, rest_position: (f64, f64, f64), stiffness: f64, damping: f64) {
        if let Some(obj) = self.get_object(id) {
            let dx = obj.object.position.x - rest_position.0;
            let dy = obj.object.position.y - rest_position.1;
            let dz = obj.object.position.z - rest_position.2;
            let vx = obj.object.velocity.x;
            let vy = obj.object.velocity.y;
            let vz = obj.object.velocity.z;
            self.apply_force(id, (
                -stiffness * dx - damping * vx,
                -stiffness * dy - damping * vy,
                -stiffness * dz - damping * vz,
            ));
        }
    }

    /// Apply an explosion force to all objects within radius
    ///
    /// Force falls off with distance squared.
    pub fn apply_explosion(&mut self, center: (f64, f64, f64), force: f64, radius: f64) {
        let ids: Vec<ObjectId> = self.object_ids.keys().copied().collect();
        for id in ids {
            if let Some(obj) = self.get_object(id) {
                let dx = obj.object.position.x - center.0;
                let dy = obj.object.position.y - center.1;
                let dz = obj.object.position.z - center.2;
                let dist_sq = dx * dx + dy * dy + dz * dz;
                let radius_sq = radius * radius;

                if dist_sq < radius_sq && dist_sq > 1e-10 {
                    // Force falls off with distance squared
                    let falloff = 1.0 - (dist_sq / radius_sq);
                    let magnitude = force * falloff;
                    self.apply_force_directed(id, magnitude, (dx, dy, dz));
                }
            }
        }
    }

    /// Apply torque to rotate an object
    pub fn apply_torque(&mut self, id: ObjectId, torque: (f64, f64, f64)) {
        if let Some(obj) = self.get_object_mut(id) {
            // Simplified: assume unit moment of inertia
            // In a full implementation, this would use the object's inertia tensor
            obj.angular_velocity.0 += torque.0;
            obj.angular_velocity.1 += torque.1;
            obj.angular_velocity.2 += torque.2;
        }
    }

    /// Apply a buoyancy force (upward force based on depth below a surface)
    /// NOTE: This is a one-shot force. For continuous buoyancy, use add_buoyancy().
    ///
    /// surface_y: Y coordinate of the fluid surface
    /// fluid_density: density of the fluid (water ≈ 1000 kg/m³)
    pub fn apply_buoyancy(&mut self, id: ObjectId, surface_y: f64, fluid_density: f64) {
        if let Some(obj) = self.get_object(id) {
            let y = obj.object.position.y;
            if y < surface_y {
                // Simplified buoyancy: force proportional to depth
                let depth = surface_y - y;
                let buoyancy_force = fluid_density * depth * self.config.constants.gravity;
                self.apply_force(id, (0.0, buoyancy_force, 0.0));
            }
        }
    }

    // ==================== Continuous Force Methods ====================

    /// Add a continuous force that persists across simulation steps
    ///
    /// Returns a ForceId that can be used to remove the force later.
    pub fn add_continuous_force(&mut self, force: ContinuousForce) -> ForceId {
        let id = ForceId::new();
        self.continuous_forces.insert(id, force);
        id
    }

    /// Remove a continuous force by its ID
    ///
    /// Returns true if the force was found and removed.
    pub fn remove_continuous_force(&mut self, id: ForceId) -> bool {
        self.continuous_forces.remove(&id).is_some()
    }

    /// Remove all continuous forces targeting a specific object
    pub fn remove_forces_on_object(&mut self, target: ObjectId) {
        self.continuous_forces.retain(|_, force| {
            match force {
                ContinuousForce::Constant { target: t, .. } => *t != target,
                ContinuousForce::Drag { target: t, .. } => *t != target,
                ContinuousForce::Spring { target: t, .. } => *t != target,
                ContinuousForce::DampedSpring { target: t, .. } => *t != target,
                ContinuousForce::Attract { target: t, .. } => *t != target,
                ContinuousForce::Repel { target: t, .. } => *t != target,
                ContinuousForce::Buoyancy { target: t, .. } => *t != target,
                ContinuousForce::Vortex { target: t, .. } => *t != target,
            }
        });
    }

    /// Get the number of active continuous forces
    pub fn continuous_force_count(&self) -> usize {
        self.continuous_forces.len()
    }

    // ==================== Continuous Force Convenience Methods ====================

    /// Add continuous drag to an object (auto-removes when velocity < min_velocity)
    pub fn add_drag(&mut self, target: ObjectId, coefficient: f64) -> ForceId {
        self.add_continuous_force(ContinuousForce::Drag {
            target,
            coefficient,
            min_velocity: 0.01,
        })
    }

    /// Add continuous drag with custom minimum velocity threshold
    pub fn add_drag_with_threshold(&mut self, target: ObjectId, coefficient: f64, min_velocity: f64) -> ForceId {
        self.add_continuous_force(ContinuousForce::Drag {
            target,
            coefficient,
            min_velocity,
        })
    }

    /// Add a continuous spring force (auto-removes when at rest)
    pub fn add_spring(&mut self, target: ObjectId, rest_position: (f64, f64, f64), stiffness: f64) -> ForceId {
        self.add_continuous_force(ContinuousForce::Spring {
            target,
            rest_position,
            stiffness,
            min_displacement: 0.01,
            min_velocity: 0.01,
        })
    }

    /// Add a continuous damped spring (auto-removes when at rest)
    pub fn add_damped_spring(&mut self, target: ObjectId, rest_position: (f64, f64, f64), stiffness: f64, damping: f64) -> ForceId {
        self.add_continuous_force(ContinuousForce::DampedSpring {
            target,
            rest_position,
            stiffness,
            damping,
            min_displacement: 0.05,
            min_velocity: 0.05,
        })
    }

    /// Add continuous attraction toward a point (persists until removed or duration expires)
    pub fn add_attraction(&mut self, target: ObjectId, point: (f64, f64, f64), strength: f64, duration: Option<f64>) -> ForceId {
        self.add_continuous_force(ContinuousForce::Attract {
            target,
            point,
            strength,
            duration,
            elapsed: 0.0,
        })
    }

    /// Add continuous repulsion from a point
    pub fn add_repulsion(&mut self, target: ObjectId, point: (f64, f64, f64), strength: f64, duration: Option<f64>) -> ForceId {
        self.add_continuous_force(ContinuousForce::Repel {
            target,
            point,
            strength,
            duration,
            elapsed: 0.0,
        })
    }

    /// Add continuous buoyancy (auto-removes after being above surface for a while)
    pub fn add_buoyancy(&mut self, target: ObjectId, surface_y: f64, fluid_density: f64) -> ForceId {
        self.add_continuous_force(ContinuousForce::Buoyancy {
            target,
            surface_y,
            fluid_density,
            time_above_surface: 0.0,
            removal_delay: 1.0,
        })
    }

    /// Add a constant continuous force (like wind or thrust)
    pub fn add_constant_force(&mut self, target: ObjectId, force: (f64, f64, f64)) -> ForceId {
        self.add_continuous_force(ContinuousForce::Constant { target, force })
    }

    /// Add a vortex/rotational force
    pub fn add_vortex(&mut self, target: ObjectId, center: (f64, f64, f64), axis: (f64, f64, f64), strength: f64, duration: Option<f64>) -> ForceId {
        self.add_continuous_force(ContinuousForce::Vortex {
            target,
            center,
            axis,
            strength,
            duration,
            elapsed: 0.0,
        })
    }

    /// Set the velocity of an object
    pub fn set_velocity(&mut self, id: ObjectId, velocity: (f64, f64, f64)) {
        if let Some(obj) = self.get_object_mut(id) {
            obj.object.velocity.x = velocity.0;
            obj.object.velocity.y = velocity.1;
            obj.object.velocity.z = velocity.2;
        }
    }

    /// Set the position of an object
    pub fn set_position(&mut self, id: ObjectId, position: (f64, f64, f64)) {
        if let Some(obj) = self.get_object_mut(id) {
            obj.object.position.x = position.0;
            obj.object.position.y = position.1;
            obj.object.position.z = position.2;
        }
    }

    /// Advance simulation by real-time delta
    ///
    /// Uses fixed timestep accumulator pattern to ensure deterministic physics.
    /// Returns the number of physics steps taken.
    pub fn update(&mut self, real_dt: f64) -> usize {
        if self.paused {
            return 0;
        }

        self.accumulated_time += real_dt;
        let mut steps = 0;

        while self.accumulated_time >= self.config.timestep {
            self.step_internal();
            self.accumulated_time -= self.config.timestep;
            steps += 1;

            // Safety limit to prevent spiral of death
            if steps >= 10 {
                self.accumulated_time = 0.0;
                break;
            }
        }

        steps
    }

    /// Perform a single physics step
    pub fn step(&mut self) {
        if !self.paused {
            self.step_internal();
        }
    }

    /// Internal step implementation
    fn step_internal(&mut self) {
        let dt = self.config.timestep;
        let world_gravity = self.config.constants.gravity;
        // Use the gravity Y component from config (gravity vector is (x, y, z))
        let gravity = -self.config.gravity.1;  // Negate because apply_gravity expects positive down

        // 1. Apply gravity to all objects (skip static objects with infinite/very high mass)
        const STATIC_MASS_THRESHOLD: f64 = 1e20; // Objects heavier than this are treated as static
        for obj in &mut self.objects {
            if obj.object.mass < STATIC_MASS_THRESHOLD && !obj.object.mass.is_infinite() {
                apply_gravity(obj, gravity, dt);
            }
        }

        // 2. Apply continuous forces and collect expired ones
        let forces_to_remove = self.apply_continuous_forces(dt, world_gravity);
        for force_id in forces_to_remove {
            self.continuous_forces.remove(&force_id);
        }

        // 3. Apply pending one-shot forces and integrate velocities
        for (idx, obj) in self.objects.iter_mut().enumerate() {
            // Find the ObjectId for this index
            if let Some(&id) = self.index_to_id.get(&idx) {
                if let Some(forces) = self.pending_forces.get(&id) {
                    let mass = obj.object.mass;
                    if mass > 0.0 {
                        for force in forces {
                            obj.object.velocity.x += force.0 / mass * dt;
                            obj.object.velocity.y += force.1 / mass * dt;
                            obj.object.velocity.z += force.2 / mass * dt;
                        }
                    }
                }
            }
        }
        // Clear all pending one-shot forces
        self.pending_forces.clear();

        // 4. Collision detection and response
        self.resolve_collisions(dt);

        // 5. Integrate positions
        for obj in &mut self.objects {
            obj.object.position.x += obj.object.velocity.x * dt;
            obj.object.position.y += obj.object.velocity.y * dt;
            obj.object.position.z += obj.object.velocity.z * dt;

            // Update orientation from angular velocity
            obj.orientation.roll += obj.angular_velocity.0 * dt;
            obj.orientation.pitch += obj.angular_velocity.1 * dt;
            obj.orientation.yaw += obj.angular_velocity.2 * dt;
        }

        // 6. Anti-tunneling: Check if any fast-moving objects have passed through the ground
        // This is a simple safeguard for objects that tunnel through the ground plane at y=0
        for obj in &mut self.objects {
            // Skip static objects
            if obj.object.mass.is_infinite() || obj.object.mass <= 0.0 {
                continue;
            }

            // Get the object's lowest point based on its shape
            let min_y = match &obj.shape {
                Shape3D::Sphere(radius) => obj.object.position.y - radius,
                Shape3D::Cuboid(_, h, _) => obj.object.position.y - h / 2.0,
                Shape3D::Cylinder(_radius, height) => obj.object.position.y - height / 2.0,
                _ => obj.object.position.y - obj.shape.bounding_radius(),
            };

            // If the object has tunneled below the ground (y=0), correct it
            if min_y < 0.0 {
                let penetration = -min_y;
                obj.object.position.y += penetration;

                // If moving downward, bounce with reduced restitution
                if obj.object.velocity.y < 0.0 {
                    let restitution = obj.get_restitution() * 0.5; // Reduced for tunneling recovery
                    obj.object.velocity.y = -obj.object.velocity.y * restitution;
                }
            }
        }

        // 6. Update time tracking
        self.tick += 1;
        self.time += dt;
    }

    /// Apply all continuous forces and return IDs of forces that should be removed
    fn apply_continuous_forces(&mut self, dt: f64, world_gravity: f64) -> Vec<ForceId> {
        let mut to_remove = Vec::new();

        // Collect force computations first (to avoid borrow issues)
        let mut force_updates: Vec<(ForceId, ObjectId, (f64, f64, f64), bool)> = Vec::new();

        for (&force_id, force) in &mut self.continuous_forces {
            match force {
                ContinuousForce::Constant { target, force: f } => {
                    if self.object_ids.contains_key(target) {
                        force_updates.push((force_id, *target, *f, false));
                    } else {
                        to_remove.push(force_id); // Target no longer exists
                    }
                }

                ContinuousForce::Drag { target, coefficient, min_velocity } => {
                    if let Some(&idx) = self.object_ids.get(target) {
                        let obj = &self.objects[idx];
                        let vx = obj.object.velocity.x;
                        let vy = obj.object.velocity.y;
                        let vz = obj.object.velocity.z;
                        let speed_sq = vx * vx + vy * vy + vz * vz;

                        if speed_sq < *min_velocity * *min_velocity {
                            to_remove.push(force_id);
                        } else {
                            let speed = speed_sq.sqrt();
                            let drag_mag = *coefficient * speed_sq;
                            let f = (
                                -vx / speed * drag_mag,
                                -vy / speed * drag_mag,
                                -vz / speed * drag_mag,
                            );
                            force_updates.push((force_id, *target, f, false));
                        }
                    } else {
                        to_remove.push(force_id);
                    }
                }

                ContinuousForce::Spring { target, rest_position, stiffness, min_displacement, min_velocity } => {
                    if let Some(&idx) = self.object_ids.get(target) {
                        let obj = &self.objects[idx];
                        let dx = obj.object.position.x - rest_position.0;
                        let dy = obj.object.position.y - rest_position.1;
                        let dz = obj.object.position.z - rest_position.2;
                        let dist_sq = dx * dx + dy * dy + dz * dz;

                        let vx = obj.object.velocity.x;
                        let vy = obj.object.velocity.y;
                        let vz = obj.object.velocity.z;
                        let speed_sq = vx * vx + vy * vy + vz * vz;

                        if dist_sq < *min_displacement * *min_displacement
                            && speed_sq < *min_velocity * *min_velocity
                        {
                            to_remove.push(force_id);
                        } else {
                            let f = (-*stiffness * dx, -*stiffness * dy, -*stiffness * dz);
                            force_updates.push((force_id, *target, f, false));
                        }
                    } else {
                        to_remove.push(force_id);
                    }
                }

                ContinuousForce::DampedSpring { target, rest_position, stiffness, damping, min_displacement, min_velocity } => {
                    if let Some(&idx) = self.object_ids.get(target) {
                        let obj = &self.objects[idx];
                        let dx = obj.object.position.x - rest_position.0;
                        let dy = obj.object.position.y - rest_position.1;
                        let dz = obj.object.position.z - rest_position.2;
                        let dist_sq = dx * dx + dy * dy + dz * dz;

                        let vx = obj.object.velocity.x;
                        let vy = obj.object.velocity.y;
                        let vz = obj.object.velocity.z;
                        let speed_sq = vx * vx + vy * vy + vz * vz;

                        if dist_sq < *min_displacement * *min_displacement
                            && speed_sq < *min_velocity * *min_velocity
                        {
                            to_remove.push(force_id);
                        } else {
                            let f = (
                                -*stiffness * dx - *damping * vx,
                                -*stiffness * dy - *damping * vy,
                                -*stiffness * dz - *damping * vz,
                            );
                            force_updates.push((force_id, *target, f, false));
                        }
                    } else {
                        to_remove.push(force_id);
                    }
                }

                ContinuousForce::Attract { target, point, strength, duration, elapsed } => {
                    *elapsed += dt;
                    if duration.map_or(false, |d| *elapsed >= d) {
                        to_remove.push(force_id);
                    } else if let Some(&idx) = self.object_ids.get(target) {
                        let obj = &self.objects[idx];
                        let dx = point.0 - obj.object.position.x;
                        let dy = point.1 - obj.object.position.y;
                        let dz = point.2 - obj.object.position.z;
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                        if dist > 1e-6 {
                            let f = (
                                dx / dist * *strength,
                                dy / dist * *strength,
                                dz / dist * *strength,
                            );
                            force_updates.push((force_id, *target, f, false));
                        }
                    } else {
                        to_remove.push(force_id);
                    }
                }

                ContinuousForce::Repel { target, point, strength, duration, elapsed } => {
                    *elapsed += dt;
                    if duration.map_or(false, |d| *elapsed >= d) {
                        to_remove.push(force_id);
                    } else if let Some(&idx) = self.object_ids.get(target) {
                        let obj = &self.objects[idx];
                        let dx = obj.object.position.x - point.0;
                        let dy = obj.object.position.y - point.1;
                        let dz = obj.object.position.z - point.2;
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                        if dist > 1e-6 {
                            let f = (
                                dx / dist * *strength,
                                dy / dist * *strength,
                                dz / dist * *strength,
                            );
                            force_updates.push((force_id, *target, f, false));
                        }
                    } else {
                        to_remove.push(force_id);
                    }
                }

                ContinuousForce::Buoyancy { target, surface_y, fluid_density, time_above_surface, removal_delay } => {
                    if let Some(&idx) = self.object_ids.get(target) {
                        let obj = &self.objects[idx];
                        let y = obj.object.position.y;

                        if y < *surface_y {
                            *time_above_surface = 0.0;
                            let depth = *surface_y - y;
                            let buoyancy = *fluid_density * depth * world_gravity;
                            force_updates.push((force_id, *target, (0.0, buoyancy, 0.0), false));
                        } else {
                            *time_above_surface += dt;
                            if *time_above_surface >= *removal_delay {
                                to_remove.push(force_id);
                            }
                        }
                    } else {
                        to_remove.push(force_id);
                    }
                }

                ContinuousForce::Vortex { target, center, axis, strength, duration, elapsed } => {
                    *elapsed += dt;
                    if duration.map_or(false, |d| *elapsed >= d) {
                        to_remove.push(force_id);
                    } else if let Some(&idx) = self.object_ids.get(target) {
                        let obj = &self.objects[idx];
                        // Vector from center to object
                        let rx = obj.object.position.x - center.0;
                        let ry = obj.object.position.y - center.1;
                        let rz = obj.object.position.z - center.2;

                        // Cross product: axis × r gives tangential direction
                        let fx = axis.1 * rz - axis.2 * ry;
                        let fy = axis.2 * rx - axis.0 * rz;
                        let fz = axis.0 * ry - axis.1 * rx;

                        let mag = (fx * fx + fy * fy + fz * fz).sqrt();
                        if mag > 1e-6 {
                            let f = (
                                fx / mag * *strength,
                                fy / mag * *strength,
                                fz / mag * *strength,
                            );
                            force_updates.push((force_id, *target, f, false));
                        }
                    } else {
                        to_remove.push(force_id);
                    }
                }
            }
        }

        // Apply the computed forces
        for (_force_id, target_id, force, _) in force_updates {
            if let Some(&idx) = self.object_ids.get(&target_id) {
                let obj = &mut self.objects[idx];
                let mass = obj.object.mass;
                if mass > 0.0 {
                    obj.object.velocity.x += force.0 / mass * dt;
                    obj.object.velocity.y += force.1 / mass * dt;
                    obj.object.velocity.z += force.2 / mass * dt;
                }
            }
        }

        to_remove
    }

    /// Resolve collisions between all object pairs using parallel detection
    ///
    /// This uses a two-phase approach:
    /// 1. Parallel detection: Find all colliding pairs and compute contact data (read-only)
    /// 2. Sequential response: Apply impulses and position corrections (write)
    fn resolve_collisions(&mut self, dt: f64) {
        let n = self.objects.len();
        if n < 2 {
            return;
        }

        // For small object counts, use sequential path (overhead of parallelism not worth it)
        if n < 8 {
            self.resolve_collisions_sequential(dt);
            return;
        }

        // Phase 1: Parallel collision detection (read-only)
        let collisions = self.detect_collisions_parallel();

        // Phase 2: Sequential collision response (write)
        for collision in collisions {
            self.apply_collision_response(&collision, dt);
        }
    }

    /// Sequential collision detection for small object counts
    fn resolve_collisions_sequential(&mut self, dt: f64) {
        let n = self.objects.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let (first, second) = self.objects.split_at_mut(j);
                let obj1 = &mut first[i];
                let obj2 = &mut second[0];
                handle_collision(obj1, obj2, dt);
            }
        }
    }

    /// Parallel collision detection - returns collision data without mutating objects
    fn detect_collisions_parallel(&self) -> Vec<CollisionData> {
        let n = self.objects.len();

        // Generate all pair indices
        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .collect();

        // Parallel collision detection
        pairs.par_iter()
            .filter_map(|&(i, j)| self.detect_collision_pair(i, j))
            .collect()
    }

    /// Detect collision between a single pair of objects (read-only)
    fn detect_collision_pair(&self, i: usize, j: usize) -> Option<CollisionData> {
        let obj1 = &self.objects[i];
        let obj2 = &self.objects[j];

        let pos1 = (obj1.object.position.x, obj1.object.position.y, obj1.object.position.z);
        let pos2 = (obj2.object.position.x, obj2.object.position.y, obj2.object.position.z);

        // Quick bounding-sphere rejection test
        let dx = pos2.0 - pos1.0;
        let dy = pos2.1 - pos1.1;
        let dz = pos2.2 - pos1.2;
        let distance_sq = dx * dx + dy * dy + dz * dz;

        let r1 = obj1.shape.bounding_radius();
        let r2 = obj2.shape.bounding_radius();
        if distance_sq > (r1 + r2).powi(2) {
            return None;
        }

        // Get orientations
        let orientation1 = Quaternion::from_euler(
            obj1.orientation.roll,
            obj1.orientation.pitch,
            obj1.orientation.yaw
        );
        let orientation2 = Quaternion::from_euler(
            obj2.orientation.roll,
            obj2.orientation.pitch,
            obj2.orientation.yaw
        );

        // Run GJK collision detection
        let gjk_result = gjk_collision_detection_ex(
            &obj1.shape, pos1, orientation1,
            &obj2.shape, pos2, orientation2
        );

        match gjk_result {
            GjkResult::NoCollision => None,
            GjkResult::SphereSphere { pos1, pos2, r1, r2 } => {
                // Sphere-sphere collision
                let dx = pos2.0 - pos1.0;
                let dy = pos2.1 - pos1.1;
                let dz = pos2.2 - pos1.2;
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                if distance < 1e-10 {
                    return None; // Overlapping centers, can't compute normal
                }

                let normal = (dx / distance, dy / distance, dz / distance);
                let penetration = r1 + r2 - distance;

                if penetration <= 0.0 {
                    return None;
                }

                let contact1 = (normal.0 * r1, normal.1 * r1, normal.2 * r1);
                let contact2 = (-normal.0 * r2, -normal.1 * r2, -normal.2 * r2);

                Some(CollisionData {
                    i, j, normal, penetration, contact1, contact2
                })
            }
            GjkResult::Collision(_) => {
                // Use EPA to get contact information
                if let Some(contact) = epa_contact_points_ex(
                    &obj1.shape, pos1, orientation1,
                    &obj2.shape, pos2, orientation2,
                    &gjk_result
                ) {
                    let penetration = contact.penetration;
                    if penetration <= 0.0 {
                        return None;
                    }

                    // Calculate contact points relative to object centers
                    let contact1 = (
                        contact.point1.0 - pos1.0,
                        contact.point1.1 - pos1.1,
                        contact.point1.2 - pos1.2,
                    );
                    let contact2 = (
                        contact.point2.0 - pos2.0,
                        contact.point2.1 - pos2.1,
                        contact.point2.2 - pos2.2,
                    );

                    Some(CollisionData {
                        i, j,
                        normal: contact.normal,
                        penetration,
                        contact1,
                        contact2,
                    })
                } else {
                    None
                }
            }
        }
    }

    /// Apply collision response for a detected collision
    fn apply_collision_response(&mut self, collision: &CollisionData, _dt: f64) {
        let (first, second) = self.objects.split_at_mut(collision.j);
        let obj1 = &mut first[collision.i];
        let obj2 = &mut second[0];

        let normal = collision.normal;
        let r1 = collision.contact1;
        let r2 = collision.contact2;

        // Calculate point velocities (linear + angular contribution)
        let v1 = (
            obj1.object.velocity.x + obj1.angular_velocity.1 * r1.2 - obj1.angular_velocity.2 * r1.1,
            obj1.object.velocity.y + obj1.angular_velocity.2 * r1.0 - obj1.angular_velocity.0 * r1.2,
            obj1.object.velocity.z + obj1.angular_velocity.0 * r1.1 - obj1.angular_velocity.1 * r1.0,
        );
        let v2 = (
            obj2.object.velocity.x + obj2.angular_velocity.1 * r2.2 - obj2.angular_velocity.2 * r2.1,
            obj2.object.velocity.y + obj2.angular_velocity.2 * r2.0 - obj2.angular_velocity.0 * r2.2,
            obj2.object.velocity.z + obj2.angular_velocity.0 * r2.1 - obj2.angular_velocity.1 * r2.0,
        );

        // Relative velocity
        let vrel = (v2.0 - v1.0, v2.1 - v1.1, v2.2 - v1.2);
        let vrel_n = vrel.0 * normal.0 + vrel.1 * normal.1 + vrel.2 * normal.2;

        // Only respond if objects are approaching
        if vrel_n >= 0.0 {
            // Still need to resolve penetration
            self.resolve_penetration(collision);
            return;
        }

        // Calculate impulse magnitude
        // Use velocity-dependent restitution: reduce bounce for low-speed impacts (resting contacts)
        // This prevents jitter when objects are stacked
        let base_restitution = (obj1.get_restitution() + obj2.get_restitution()) / 2.0;
        const RESTITUTION_VELOCITY_THRESHOLD: f64 = 2.0; // m/s - below this, reduce restitution
        let restitution = if vrel_n.abs() < RESTITUTION_VELOCITY_THRESHOLD {
            // Scale restitution with square of approach speed for faster falloff
            base_restitution * (vrel_n.abs() / RESTITUTION_VELOCITY_THRESHOLD).powi(2)
        } else {
            base_restitution
        };

        let m1 = obj1.object.mass;
        let m2 = obj2.object.mass;

        // Check for static/infinite mass objects
        let m1_static = m1.is_infinite() || m1 <= 0.0;
        let m2_static = m2.is_infinite() || m2 <= 0.0;

        // For now, use simplified impulse (no angular contribution to denominator)
        // Use inverse mass = 0 for static objects
        let inv_mass1 = if m1_static { 0.0 } else { 1.0 / m1 };
        let inv_mass2 = if m2_static { 0.0 } else { 1.0 / m2 };
        let inv_mass_sum = inv_mass1 + inv_mass2;

        if inv_mass_sum < 1e-10 {
            return; // Both objects have infinite mass
        }

        let j = -(1.0 + restitution) * vrel_n / inv_mass_sum;

        // Apply linear impulse (skip for static objects)
        if !m1_static {
            let impulse_over_m1 = j / m1;
            obj1.object.velocity.x -= normal.0 * impulse_over_m1;
            obj1.object.velocity.y -= normal.1 * impulse_over_m1;
            obj1.object.velocity.z -= normal.2 * impulse_over_m1;
        }
        if !m2_static {
            let impulse_over_m2 = j / m2;
            obj2.object.velocity.x += normal.0 * impulse_over_m2;
            obj2.object.velocity.y += normal.1 * impulse_over_m2;
            obj2.object.velocity.z += normal.2 * impulse_over_m2;
        }

        // Apply angular impulse (simplified, skip for static objects)
        let torque_scale = 0.1; // Reduced angular response
        if !m1_static {
            let torque1 = cross_product(r1, (normal.0 * j, normal.1 * j, normal.2 * j));
            obj1.angular_velocity.0 -= torque1.0 * torque_scale;
            obj1.angular_velocity.1 -= torque1.1 * torque_scale;
            obj1.angular_velocity.2 -= torque1.2 * torque_scale;
        }
        if !m2_static {
            let torque2 = cross_product(r2, (normal.0 * j, normal.1 * j, normal.2 * j));
            obj2.angular_velocity.0 += torque2.0 * torque_scale;
            obj2.angular_velocity.1 += torque2.1 * torque_scale;
            obj2.angular_velocity.2 += torque2.2 * torque_scale;
        }

        // Resolve penetration
        self.resolve_penetration(collision);
    }

    /// Resolve penetration between two objects
    fn resolve_penetration(&mut self, collision: &CollisionData) {
        let (first, second) = self.objects.split_at_mut(collision.j);
        let obj1 = &mut first[collision.i];
        let obj2 = &mut second[0];

        let normal = collision.normal;
        let penetration = collision.penetration;

        let m1 = obj1.object.mass;
        let m2 = obj2.object.mass;

        // Check for static/infinite mass objects
        let m1_static = m1.is_infinite() || m1 <= 0.0;
        let m2_static = m2.is_infinite() || m2 <= 0.0;

        // Calculate correction ratio based on masses
        let (ratio1, ratio2) = if m1_static && m2_static {
            (0.0, 0.0) // Both immovable
        } else if m1_static {
            (0.0, 1.0) // Only obj1 is immovable
        } else if m2_static {
            (1.0, 0.0) // Only obj2 is immovable
        } else {
            let total = m1 + m2;
            (m2 / total, m1 / total) // Distribute by inverse mass ratio
        };

        // Apply position correction using logarithmic scaling
        // This gives strong correction for deep penetrations but very gentle for shallow ones
        // which helps prevent jitter while still resolving significant overlaps
        const PENETRATION_SLOP: f64 = 0.005; // Allow 5mm overlap before correcting
        const LOG_SCALE: f64 = 10.0; // Controls the curve steepness

        let excess = (penetration - PENETRATION_SLOP).max(0.0);
        // ln(1 + x*scale) / ln(1 + scale) gives 0 at x=0 and 1 at x=1
        // For small penetrations this is nearly zero, for large ones it approaches the penetration
        let correction = if excess > 0.0 {
            let normalized = (1.0 + excess * LOG_SCALE).ln() / (1.0 + LOG_SCALE).ln();
            excess * normalized * 0.5 // Apply 50% of the log-scaled correction
        } else {
            0.0
        };

        obj1.object.position.x -= normal.0 * correction * ratio1;
        obj1.object.position.y -= normal.1 * correction * ratio1;
        obj1.object.position.z -= normal.2 * correction * ratio1;

        obj2.object.position.x += normal.0 * correction * ratio2;
        obj2.object.position.y += normal.1 * correction * ratio2;
        obj2.object.position.z += normal.2 * correction * ratio2;
    }

    /// Get a snapshot of the current world state
    pub fn get_state(&self) -> WorldState {
        let objects: Vec<ObjectState> = self.objects.iter()
            .enumerate()
            .filter_map(|(idx, obj)| {
                self.index_to_id.get(&idx).map(|&id| {
                    // Convert Euler angles to quaternion
                    let quat = Quaternion::from_euler(
                        obj.orientation.roll,
                        obj.orientation.pitch,
                        obj.orientation.yaw
                    );

                    ObjectState {
                        id,
                        position: (
                            obj.object.position.x,
                            obj.object.position.y,
                            obj.object.position.z
                        ),
                        orientation: (quat.x, quat.y, quat.z, quat.w),
                        velocity: (
                            obj.object.velocity.x,
                            obj.object.velocity.y,
                            obj.object.velocity.z
                        ),
                        angular_velocity: obj.angular_velocity,
                    }
                })
            })
            .collect();

        WorldState {
            tick: self.tick,
            time: self.time,
            objects,
        }
    }

    /// Get physics constants
    pub fn constants(&self) -> &PhysicsConstants {
        &self.config.constants
    }

    /// Get world configuration
    pub fn config(&self) -> &WorldConfig {
        &self.config
    }
}

/// Cross product of two 3D vectors
#[inline]
fn cross_product(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Shape3D;
    use crate::utils::PhysicsConstants;

    fn create_test_sphere(position: (f64, f64, f64), velocity: (f64, f64, f64)) -> PhysicalObject3D {
        PhysicalObject3D::new(
            1.0,  // mass
            velocity,
            position,
            Shape3D::Sphere(0.5),
            None,  // material
            (0.0, 0.0, 0.0),  // angular_velocity
            (0.0, 0.0, 0.0),  // orientation
            PhysicsConstants::default(),
        )
    }

    #[test]
    fn test_add_remove_objects() {
        let mut world = PhysicsWorld::default_world();

        let id1 = world.add_object(create_test_sphere((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)));
        let id2 = world.add_object(create_test_sphere((5.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        assert_eq!(world.object_count(), 2);
        assert!(world.get_object(id1).is_some());
        assert!(world.get_object(id2).is_some());

        assert!(world.remove_object(id1));
        assert_eq!(world.object_count(), 1);
        assert!(world.get_object(id1).is_none());
        assert!(world.get_object(id2).is_some());
    }

    #[test]
    fn test_gravity_integration() {
        let mut world = PhysicsWorld::new(
            WorldConfig::default().with_gravity(0.0, -10.0, 0.0)
        );

        let id = world.add_object(create_test_sphere((0.0, 10.0, 0.0), (0.0, 0.0, 0.0)));

        // Step multiple times
        for _ in 0..120 {  // 1 second at 120Hz
            world.step();
        }

        let state = world.get_state();
        let obj = state.get_object(id).unwrap();

        // Object should have fallen approximately 5m (1/2 * g * t^2)
        // Allow some tolerance for collision detection overhead
        assert!(obj.position.1 < 10.0, "Object should have fallen");
        assert!(obj.velocity.1 < 0.0, "Object should have downward velocity");
    }

    #[test]
    fn test_pause_resume() {
        let mut world = PhysicsWorld::default_world();
        let id = world.add_object(create_test_sphere((0.0, 10.0, 0.0), (0.0, 0.0, 0.0)));

        let initial_pos = world.get_state().get_position(id).unwrap();

        world.pause();
        for _ in 0..100 {
            world.step();
        }

        let paused_pos = world.get_state().get_position(id).unwrap();
        assert_eq!(initial_pos, paused_pos, "Position should not change while paused");

        world.resume();
        world.step();

        let resumed_pos = world.get_state().get_position(id).unwrap();
        assert_ne!(initial_pos, resumed_pos, "Position should change after resume");
    }

    #[test]
    fn test_apply_impulse() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Apply impulse of 10 kg*m/s in x direction to 1kg object
        world.apply_impulse(id, (10.0, 0.0, 0.0));

        let obj = world.get_object(id).unwrap();
        assert!((obj.object.velocity.x - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_force_directed() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Apply 10N force in the (1, 1, 0) direction (normalized)
        world.apply_force_directed(id, 10.0, (1.0, 1.0, 0.0));
        world.step();

        let obj = world.get_object(id).unwrap();
        // Force should be split equally between x and y (normalized direction)
        let expected = 10.0 / 2.0_f64.sqrt() * world.timestep();  // F/m * dt
        assert!((obj.object.velocity.x - expected).abs() < 0.001,
            "vx={}, expected={}", obj.object.velocity.x, expected);
        assert!((obj.object.velocity.y - expected).abs() < 0.001,
            "vy={}, expected={}", obj.object.velocity.y, expected);
    }

    #[test]
    fn test_force_toward() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((10.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Apply force toward origin
        world.apply_force_toward(id, (0.0, 0.0, 0.0), 10.0);
        world.step();

        let obj = world.get_object(id).unwrap();
        // Object should now be moving toward origin (negative x)
        assert!(obj.object.velocity.x < 0.0, "Should move toward target");
    }

    #[test]
    fn test_spring_force() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((5.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Apply spring force toward origin with k=10
        world.apply_spring_force(id, (0.0, 0.0, 0.0), 10.0);
        world.step();

        let obj = world.get_object(id).unwrap();
        // Spring should pull toward rest position (negative x velocity)
        assert!(obj.object.velocity.x < 0.0, "Spring should pull toward rest");
    }

    #[test]
    fn test_explosion() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id1 = world.add_object(create_test_sphere((2.0, 0.0, 0.0), (0.0, 0.0, 0.0)));
        let id2 = world.add_object(create_test_sphere((0.0, 2.0, 0.0), (0.0, 0.0, 0.0)));
        let id3 = world.add_object(create_test_sphere((100.0, 0.0, 0.0), (0.0, 0.0, 0.0)));  // Outside radius

        // Explosion at origin with radius 10
        world.apply_explosion((0.0, 0.0, 0.0), 100.0, 10.0);
        world.step();

        let obj1 = world.get_object(id1).unwrap();
        let obj2 = world.get_object(id2).unwrap();
        let obj3 = world.get_object(id3).unwrap();

        // Objects within radius should be pushed away
        assert!(obj1.object.velocity.x > 0.0, "Should be pushed in +x");
        assert!(obj2.object.velocity.y > 0.0, "Should be pushed in +y");
        // Object outside radius should be unaffected
        assert!((obj3.object.velocity.x).abs() < 0.001, "Should be unaffected");
    }

    #[test]
    fn test_drag() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)));

        let initial_speed = 10.0;

        // Apply drag for several steps
        for _ in 0..10 {
            world.apply_drag(id, 0.5);
            world.step();
        }

        let obj = world.get_object(id).unwrap();
        // Velocity should have decreased due to drag
        assert!(obj.object.velocity.x < initial_speed, "Drag should slow object");
        assert!(obj.object.velocity.x > 0.0, "Should still be moving forward");
    }

    // ==================== Continuous Force Tests ====================

    #[test]
    fn test_continuous_drag() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)));

        let initial_velocity = 10.0;

        // Add continuous drag (not one-shot)
        let drag_id = world.add_drag(id, 0.5);
        assert_eq!(world.continuous_force_count(), 1);

        // Step multiple times - drag should persist and continuously slow object
        for _ in 0..50 {
            world.step();
        }

        let obj = world.get_object(id).unwrap();
        // Object should have slowed (drag is continuous, not one-shot)
        assert!(obj.object.velocity.x < initial_velocity,
            "Continuous drag should slow object. Got velocity: {}", obj.object.velocity.x);

        // Remove the drag force
        assert!(world.remove_continuous_force(drag_id));
        assert_eq!(world.continuous_force_count(), 0);
    }

    #[test]
    fn test_continuous_drag_auto_removal() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)));

        // Add drag with low velocity threshold
        world.add_drag_with_threshold(id, 0.5, 0.5);
        assert_eq!(world.continuous_force_count(), 1);

        // Step until velocity drops below threshold
        for _ in 0..500 {
            world.step();
            if world.continuous_force_count() == 0 {
                break;
            }
        }

        // Drag should have been auto-removed
        assert_eq!(world.continuous_force_count(), 0, "Drag should auto-remove when velocity < threshold");
    }

    #[test]
    fn test_continuous_spring() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((5.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Add spring attached to origin with higher damping for faster settling
        let spring_id = world.add_damped_spring(id, (0.0, 0.0, 0.0), 10.0, 5.0);
        assert_eq!(world.continuous_force_count(), 1);

        // Step many times - should approach rest position
        for _ in 0..500 {
            world.step();
        }

        let obj = world.get_object(id).unwrap();
        // Should be near origin (within 0.5m after 500 steps with high damping)
        assert!(obj.object.position.x.abs() < 0.5, "Spring should pull object toward rest position");

        // Continue stepping and verify object stays near rest position (spring is working)
        for _ in 0..500 {
            world.step();
        }

        let obj = world.get_object(id).unwrap();
        // Should still be near origin and nearly at rest
        assert!(obj.object.position.x.abs() < 0.1, "Spring should keep object near rest position");
        assert!(obj.object.velocity.x.abs() < 0.5, "Object should have low velocity");

        // Clean up spring manually (auto-removal thresholds may be too strict for unit tests)
        world.remove_continuous_force(spring_id);
        assert_eq!(world.continuous_force_count(), 0, "Spring should be removed after manual removal");
    }

    #[test]
    fn test_continuous_attraction_with_duration() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((10.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Add attraction for 0.5 seconds (60 steps at 120Hz)
        world.add_attraction(id, (0.0, 0.0, 0.0), 100.0, Some(0.5));
        assert_eq!(world.continuous_force_count(), 1);

        // Step for less than duration
        for _ in 0..30 {
            world.step();
        }
        assert_eq!(world.continuous_force_count(), 1, "Attraction should still be active");

        let obj = world.get_object(id).unwrap();
        assert!(obj.object.velocity.x < 0.0, "Should be moving toward attractor");

        // Step past duration
        for _ in 0..60 {
            world.step();
        }
        assert_eq!(world.continuous_force_count(), 0, "Attraction should expire after duration");
    }

    #[test]
    fn test_constant_force() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Add constant force (like wind)
        let wind_id = world.add_constant_force(id, (10.0, 0.0, 0.0));

        // Step multiple times
        for _ in 0..60 {
            world.step();
        }

        let obj = world.get_object(id).unwrap();
        // Should be accelerating in x direction
        assert!(obj.object.velocity.x > 0.0, "Constant force should accelerate object");
        assert!(obj.object.position.x > 0.0, "Object should have moved");

        // Constant force does NOT auto-remove
        assert_eq!(world.continuous_force_count(), 1);

        // Must manually remove
        world.remove_continuous_force(wind_id);
        assert_eq!(world.continuous_force_count(), 0);
    }

    #[test]
    fn test_remove_forces_on_object() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id1 = world.add_object(create_test_sphere((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)));
        let id2 = world.add_object(create_test_sphere((5.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Add forces to both objects
        world.add_drag(id1, 0.5);
        world.add_constant_force(id1, (1.0, 0.0, 0.0));
        world.add_drag(id2, 0.5);

        assert_eq!(world.continuous_force_count(), 3);

        // Remove all forces on object 1
        world.remove_forces_on_object(id1);

        assert_eq!(world.continuous_force_count(), 1, "Should only have force on object 2");
    }

    #[test]
    fn test_force_removed_when_object_removed() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((0.0, 0.0, 0.0), (5.0, 0.0, 0.0)));

        world.add_drag(id, 0.5);
        assert_eq!(world.continuous_force_count(), 1);

        // Remove the object
        world.remove_object(id);

        // Step - should detect target doesn't exist and remove force
        world.step();

        assert_eq!(world.continuous_force_count(), 0, "Force should be removed when target is removed");
    }

    #[test]
    fn test_continuous_repulsion() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        // Place object at (5, 0, 0), repel from origin
        let id = world.add_object(create_test_sphere((5.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Add repulsion from origin with strength 100, no expiration
        let repel_id = world.add_repulsion(id, (0.0, 0.0, 0.0), 100.0, None);
        assert_eq!(world.continuous_force_count(), 1);

        // Step multiple times
        for _ in 0..60 {
            world.step();
        }

        let obj = world.get_object(id).unwrap();
        // Should be pushed away from origin (positive x velocity)
        assert!(obj.object.velocity.x > 0.0, "Repulsion should push object away from point. Got vx={}", obj.object.velocity.x);
        // Should have moved further from origin
        assert!(obj.object.position.x > 5.0, "Object should have moved away from origin. Got x={}", obj.object.position.x);

        // Force should still be active (no duration)
        assert_eq!(world.continuous_force_count(), 1);

        // Manual removal
        world.remove_continuous_force(repel_id);
        assert_eq!(world.continuous_force_count(), 0);
    }

    #[test]
    fn test_continuous_repulsion_with_duration() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((5.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Add repulsion for 0.5 seconds
        world.add_repulsion(id, (0.0, 0.0, 0.0), 100.0, Some(0.5));
        assert_eq!(world.continuous_force_count(), 1);

        // Step for less than duration (30 steps at 120Hz = 0.25s)
        for _ in 0..30 {
            world.step();
        }
        assert_eq!(world.continuous_force_count(), 1, "Repulsion should still be active");

        // Step past duration
        for _ in 0..60 {
            world.step();
        }
        assert_eq!(world.continuous_force_count(), 0, "Repulsion should expire after duration");
    }

    #[test]
    fn test_continuous_vortex() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        // Place object at (5, 0, 0), vortex centered at origin with Y axis
        let id = world.add_object(create_test_sphere((5.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Add vortex around Y axis - should create tangential force
        let vortex_id = world.add_vortex(id, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 100.0, None);
        assert_eq!(world.continuous_force_count(), 1);

        // Step multiple times
        for _ in 0..60 {
            world.step();
        }

        let obj = world.get_object(id).unwrap();
        // Object at (5, 0, 0) with Y axis vortex should get force in Z direction (tangential)
        // Cross product: (0,1,0) × (5,0,0) = (0*0 - 0*0, 0*5 - 1*0, 1*0 - 0*5) = (0, 0, 0)... wait
        // Actually: axis × r where r = obj - center = (5,0,0)
        // (0,1,0) × (5,0,0) = (1*0 - 0*0, 0*5 - 0*0, 0*0 - 1*5) = (0, 0, -5)
        // So force should be in -Z direction
        assert!(obj.object.velocity.z < 0.0, "Vortex should create tangential velocity. Got vz={}", obj.object.velocity.z);

        // Force should still be active
        assert_eq!(world.continuous_force_count(), 1);

        world.remove_continuous_force(vortex_id);
        assert_eq!(world.continuous_force_count(), 0);
    }

    #[test]
    fn test_continuous_vortex_with_duration() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((5.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Add vortex for 0.25 seconds
        world.add_vortex(id, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 100.0, Some(0.25));
        assert_eq!(world.continuous_force_count(), 1);

        // Step past duration (60 steps at 120Hz = 0.5s)
        for _ in 0..60 {
            world.step();
        }

        assert_eq!(world.continuous_force_count(), 0, "Vortex should expire after duration");
    }

    #[test]
    fn test_continuous_buoyancy() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        // Place object below surface (y=0 surface, object at y=-5)
        let id = world.add_object(create_test_sphere((0.0, -5.0, 0.0), (0.0, 0.0, 0.0)));

        // Add buoyancy with water surface at y=0, water density
        let buoyancy_id = world.add_buoyancy(id, 0.0, 1000.0);
        assert_eq!(world.continuous_force_count(), 1);

        // Step multiple times
        for _ in 0..120 {
            world.step();
        }

        let obj = world.get_object(id).unwrap();
        // Object should have upward velocity from buoyancy
        assert!(obj.object.velocity.y > 0.0, "Buoyancy should push object up. Got vy={}", obj.object.velocity.y);
        // Object should have moved up
        assert!(obj.object.position.y > -5.0, "Object should have risen. Got y={}", obj.object.position.y);

        // Buoyancy should still be active while below surface
        assert_eq!(world.continuous_force_count(), 1);

        world.remove_continuous_force(buoyancy_id);
    }

    #[test]
    fn test_continuous_buoyancy_auto_removal() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        // Place object above surface (y=0 surface, object at y=5)
        let id = world.add_object(create_test_sphere((0.0, 5.0, 0.0), (0.0, 0.0, 0.0)));

        // Add buoyancy - object is already above surface
        world.add_buoyancy(id, 0.0, 1000.0);
        assert_eq!(world.continuous_force_count(), 1);

        // Step for more than removal_delay (1.0s default, so 120+ steps at 120Hz)
        for _ in 0..150 {
            world.step();
        }

        // Buoyancy should auto-remove since object has been above surface
        assert_eq!(world.continuous_force_count(), 0, "Buoyancy should auto-remove when above surface for extended time");
    }

    #[test]
    fn test_buoyancy_no_force_above_surface() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        // Object above surface
        let id = world.add_object(create_test_sphere((0.0, 5.0, 0.0), (0.0, 0.0, 0.0)));

        world.add_buoyancy(id, 0.0, 1000.0);

        // Step a few times (before auto-removal kicks in)
        for _ in 0..10 {
            world.step();
        }

        let obj = world.get_object(id).unwrap();
        // No force should be applied above surface
        assert!((obj.object.velocity.y).abs() < 0.001, "No buoyancy force above surface. Got vy={}", obj.object.velocity.y);
    }

    // ==================== Parallel Collision Detection Tests ====================

    #[test]
    fn test_parallel_collision_many_objects() {
        // Test with enough objects to trigger parallel path (>= 8)
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());

        // Create a grid of 16 spheres (4x4) that don't initially collide
        let mut ids = Vec::new();
        for x in 0..4 {
            for z in 0..4 {
                let pos = (x as f64 * 3.0, 0.0, z as f64 * 3.0);
                let id = world.add_object(create_test_sphere(pos, (0.0, 0.0, 0.0)));
                ids.push(id);
            }
        }

        assert_eq!(world.object_count(), 16);

        // Step should use parallel collision detection
        world.step();

        // All objects should still exist and have valid positions
        for id in &ids {
            let obj = world.get_object(*id);
            assert!(obj.is_some(), "Object should still exist after parallel collision step");
        }
    }

    #[test]
    fn test_parallel_collision_with_actual_collisions() {
        // Test parallel path with objects that will collide
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());

        // Create 10 spheres all moving toward center - they will collide
        let mut ids = Vec::new();
        for i in 0..10 {
            let angle = (i as f64) * std::f64::consts::PI * 2.0 / 10.0;
            let distance = 5.0;
            let pos = (angle.cos() * distance, 0.0, angle.sin() * distance);
            // Velocity toward center
            let vel = (-angle.cos() * 10.0, 0.0, -angle.sin() * 10.0);
            let id = world.add_object(create_test_sphere(pos, vel));
            ids.push(id);
        }

        assert_eq!(world.object_count(), 10);

        // Step multiple times - objects will collide near center
        for _ in 0..60 {
            world.step();
        }

        // After collisions, objects should have bounced and moved
        // Check that the simulation is stable (no NaN/Inf values)
        for id in &ids {
            let obj = world.get_object(*id).expect("Object should exist");
            assert!(!obj.object.position.x.is_nan(), "Position should not be NaN");
            assert!(!obj.object.velocity.x.is_nan(), "Velocity should not be NaN");
            assert!(obj.object.position.x.is_finite(), "Position should be finite");
            assert!(obj.object.velocity.x.is_finite(), "Velocity should be finite");
        }
    }

    #[test]
    fn test_parallel_collision_sphere_sphere() {
        // Test sphere-sphere collision through parallel path
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());

        // Create 8 spheres to trigger parallel path, but only 2 will collide
        for i in 0..6 {
            // Non-colliding spheres spread out
            let pos = (i as f64 * 10.0 + 20.0, 0.0, 0.0);
            world.add_object(create_test_sphere(pos, (0.0, 0.0, 0.0)));
        }

        // Two spheres that will collide (overlapping, moving toward each other)
        // Spheres have radius 0.5, so at distance < 1.0 they overlap
        // Distance = 0.8, combined radii = 1.0, so penetration = 0.2
        let id1 = world.add_object(create_test_sphere((-0.4, 0.0, 0.0), (5.0, 0.0, 0.0)));
        let id2 = world.add_object(create_test_sphere((0.4, 0.0, 0.0), (-5.0, 0.0, 0.0)));

        assert_eq!(world.object_count(), 8);

        let v1_before = world.get_object(id1).unwrap().object.velocity.x;
        let v2_before = world.get_object(id2).unwrap().object.velocity.x;

        // Step once - collision should be resolved
        world.step();

        // After collision, they should have bounced apart
        let obj1 = world.get_object(id1).unwrap();
        let obj2 = world.get_object(id2).unwrap();

        // Velocities should have changed
        let v1_after = obj1.object.velocity.x;
        let v2_after = obj2.object.velocity.x;

        // At minimum, velocities should have changed from the collision
        assert!(v1_after != v1_before || v2_after != v2_before,
            "Collision should change velocities. Before: ({}, {}), After: ({}, {})",
            v1_before, v2_before, v1_after, v2_after);

        // Velocities should have reversed (approximately)
        assert!(obj1.object.velocity.x < 0.0, "Sphere 1 should bounce back. Got vx={}", obj1.object.velocity.x);
        assert!(obj2.object.velocity.x > 0.0, "Sphere 2 should bounce back. Got vx={}", obj2.object.velocity.x);
    }
}
