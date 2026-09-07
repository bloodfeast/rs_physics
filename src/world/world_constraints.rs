//! World constraint types for PhysicsWorld integration.
//!
//! This module provides constraint wrappers that work with ObjectId references
//! instead of owned objects, allowing constraints to integrate with PhysicsWorld.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use super::state::ObjectId;
use crate::constraints::{RopeChain3D, Hinge3D};
use crate::models::PhysicalObject3D;

/// Unique identifier for constraints in the physics world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstraintId(pub u64);

impl ConstraintId {
    /// Generate a new unique ConstraintId
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        ConstraintId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for ConstraintId {
    fn default() -> Self {
        Self::new()
    }
}

/// A joint constraint between two objects in the world.
///
/// Maintains a fixed distance between two objects.
#[derive(Debug, Clone)]
pub struct WorldJoint3D {
    /// First object (can be static with None)
    pub object1: Option<ObjectId>,
    /// Second object
    pub object2: ObjectId,
    /// Fixed anchor position (used when object1 is None)
    pub anchor: Option<(f64, f64, f64)>,
    /// Target distance between objects
    pub distance: f64,
    /// Baumgarte stabilization factor (0.0 to 1.0)
    pub baumgarte: f64,
    /// Accumulated impulse for warm starting
    pub lambda: f64,
}

impl WorldJoint3D {
    /// Creates a joint between two objects.
    pub fn between(obj1: ObjectId, obj2: ObjectId, distance: f64) -> Self {
        Self {
            object1: Some(obj1),
            object2: obj2,
            anchor: None,
            distance,
            baumgarte: 0.2,
            lambda: 0.0,
        }
    }

    /// Creates a joint from a fixed anchor point to an object.
    pub fn anchored(anchor: (f64, f64, f64), obj: ObjectId, distance: f64) -> Self {
        Self {
            object1: None,
            object2: obj,
            anchor: Some(anchor),
            distance,
            baumgarte: 0.2,
            lambda: 0.0,
        }
    }
}

/// A spring constraint between two objects in the world.
///
/// Applies elastic forces to maintain a rest length.
#[derive(Debug, Clone)]
pub struct WorldSpring3D {
    /// First object (can be static with None)
    pub object1: Option<ObjectId>,
    /// Second object
    pub object2: ObjectId,
    /// Fixed anchor position (used when object1 is None)
    pub anchor: Option<(f64, f64, f64)>,
    /// Spring stiffness (N/m)
    pub stiffness: f64,
    /// Rest length
    pub rest_length: f64,
    /// Damping coefficient
    pub damping: f64,
}

impl WorldSpring3D {
    /// Creates a spring between two objects.
    pub fn between(obj1: ObjectId, obj2: ObjectId, stiffness: f64, rest_length: f64, damping: f64) -> Self {
        Self {
            object1: Some(obj1),
            object2: obj2,
            anchor: None,
            stiffness,
            rest_length,
            damping,
        }
    }

    /// Creates a spring from a fixed anchor point to an object.
    pub fn anchored(anchor: (f64, f64, f64), obj: ObjectId, stiffness: f64, rest_length: f64, damping: f64) -> Self {
        Self {
            object1: None,
            object2: obj,
            anchor: Some(anchor),
            stiffness,
            rest_length,
            damping,
        }
    }
}

/// A rope constraint between two objects in the world.
///
/// Only resists stretching beyond max_length (allows slack).
#[derive(Debug, Clone)]
pub struct WorldRope3D {
    /// First object (can be static with None)
    pub object1: Option<ObjectId>,
    /// Second object
    pub object2: ObjectId,
    /// Fixed anchor position (used when object1 is None)
    pub anchor: Option<(f64, f64, f64)>,
    /// Maximum length before constraint activates
    pub max_length: f64,
    /// Baumgarte stabilization factor
    pub baumgarte: f64,
    /// Accumulated impulse for warm starting
    pub lambda: f64,
}

impl WorldRope3D {
    /// Creates a rope between two objects.
    pub fn between(obj1: ObjectId, obj2: ObjectId, max_length: f64) -> Self {
        Self {
            object1: Some(obj1),
            object2: obj2,
            anchor: None,
            max_length,
            baumgarte: 0.2,
            lambda: 0.0,
        }
    }

    /// Creates a rope from a fixed anchor point to an object.
    pub fn anchored(anchor: (f64, f64, f64), obj: ObjectId, max_length: f64) -> Self {
        Self {
            object1: None,
            object2: obj,
            anchor: Some(anchor),
            max_length,
            baumgarte: 0.2,
            lambda: 0.0,
        }
    }
}

/// All constraint types that can be added to the physics world.
#[derive(Debug)]
pub enum WorldConstraint {
    /// Distance constraint (joint)
    Joint(WorldJoint3D),
    /// Spring constraint
    Spring(WorldSpring3D),
    /// Rope constraint (max length only)
    Rope(WorldRope3D),
    /// Self-contained rope chain (has its own particles)
    RopeChain(RopeChain3D),
    /// Hinge constraint (angular, with its own objects)
    Hinge(Hinge3D),
}

impl WorldConstraint {
    /// Solves the constraint for one iteration.
    ///
    /// # Arguments
    ///
    /// * `objects` - Map from ObjectId to index in objects vector
    /// * `object_data` - The objects vector
    /// * `dt` - Timestep in seconds
    /// Advances this constraint by one step.
    ///
    /// `gravity` is a positive-downward magnitude, as elsewhere in this module.
    /// It is threaded through because `Hinge3D::solve` otherwise falls back to a
    /// hardcoded -9.81 - its own documentation says to prefer
    /// `solve_with_gravity` with the simulation's real value - so a door in a
    /// world with stronger gravity swung open in slow motion.
    pub fn solve(
        &mut self,
        object_ids: &HashMap<ObjectId, usize>,
        objects: &mut [PhysicalObject3D],
        dt: f64,
        gravity: f64,
    ) {
        match self {
            WorldConstraint::Joint(joint) => {
                solve_joint(joint, object_ids, objects, dt);
            }
            WorldConstraint::Spring(spring) => {
                solve_spring(spring, object_ids, objects, dt);
            }
            WorldConstraint::Rope(rope) => {
                solve_rope(rope, object_ids, objects, dt);
            }
            WorldConstraint::RopeChain(chain) => {
                solve_rope_chain(chain, dt);
            }
            WorldConstraint::Hinge(hinge) => {
                // Hinge has its own internal objects, solve directly
                let _ = hinge.solve_with_gravity(dt, -gravity);
            }
        }
    }

    /// Applies gravity to constraint-owned particles (for RopeChain, Hinge).
    /// Applies gravity to constraint-owned particles.
    ///
    /// `gravity` is a positive-downward magnitude, matching
    /// [`crate::interactions::shape_collisions_3d::apply_gravity`], which the
    /// world uses for ordinary objects and which subtracts it from velocity.
    ///
    /// `RopeChain3D::apply_gravity` uses the opposite convention - it *adds* to
    /// velocity, so it wants a signed value - and the world was handing the same
    /// positive number to both. Objects fell and ropes rose. While the rope was
    /// never integrated that was invisible; once it was, it became an unbounded
    /// energy source that flung the bridge upward and blew up anything colliding
    /// with it.
    pub fn apply_gravity(&mut self, gravity: f64, dt: f64) {
        match self {
            WorldConstraint::RopeChain(chain) => {
                chain.apply_gravity(-gravity, dt);
            }
            WorldConstraint::Hinge(_hinge) => {
                // Hinge applies torque from gravity internally
            }
            _ => {}
        }
    }

    /// Gets particle positions for rendering (if applicable).
    pub fn get_particle_positions(&self) -> Option<Vec<(f64, f64, f64)>> {
        match self {
            WorldConstraint::RopeChain(chain) => Some(chain.get_particle_positions()),

            _ => None,
        }
    }

    /// Mass of one constraint-owned particle, if this constraint has any.
    ///
    /// Needed to scale impulses handed back from collisions: the contact solver
    /// computes them against an infinite-mass proxy, which is the right answer
    /// for the *other* body but far too large for the light particle standing
    /// behind it.
    pub fn particle_mass(&self, index: usize) -> Option<f64> {
        match self {
            WorldConstraint::RopeChain(chain) => chain.get_particle(index).map(|p| p.mass),
            _ => None,
        }
    }

    /// Applies a force to one constraint-owned particle, if this constraint has any.
    ///
    /// Returns `false` if the constraint has no particles or the index is out of
    /// range. A rope's particles are internal to the constraint, so nothing
    /// outside it can push on them - which means a rope bridge carries no load
    /// from the objects resting on it and hangs in the same curve whether it is
    /// bearing weight or not.
    pub fn apply_particle_force(
        &mut self,
        index: usize,
        force: (f64, f64, f64),
        dt: f64,
    ) -> bool {
        match self {
            WorldConstraint::RopeChain(chain) => {
                if index >= chain.particle_count() {
                    return false;
                }
                chain.apply_force(index, force, dt);
                true
            }
            _ => false,
        }
    }
}

/// Advances a rope chain by one step: integrate, project, then correct velocity.
///
/// Two things were missing here, and the second only bites once the first is
/// fixed.
///
/// Integration was never called. Gravity added velocity and the segment
/// constraints were solved, but nothing moved the particles by that velocity, so
/// the rope stayed frozen in whatever shape it was built with - it looked like a
/// hanging rope and behaved like a rigid one.
///
/// Adding integration alone then diverges to NaN within seconds. The solver is
/// position-based: it corrects positions to satisfy the segment lengths but
/// never touches velocity. Gravity therefore keeps adding velocity that the
/// constraint has no way to remove, and it grows without bound while the solver
/// fights to hold the positions. The fix is the standard PBD velocity update -
/// after projection, velocity is *defined* by how far the particle actually
/// moved. A taut segment then removes velocity along its own direction, which is
/// what a rope physically does.
#[cfg(feature = "constraints")]
fn solve_rope_chain(chain: &mut RopeChain3D, dt: f64) {
    if dt <= 0.0 {
        return;
    }

    let before = chain.get_particle_positions();
    chain.integrate(dt);

    // A single iteration leaves visible stretch on a loaded rope; the segments
    // are solved sequentially, so corrections need a few passes to propagate
    // from the anchors to the middle.
    let _ = chain.solve(dt, 4);

    let after = chain.get_particle_positions();
    for i in 0..after.len().min(before.len()) {
        let Some(particle) = chain.get_particle_mut(i) else { continue };
        if particle.inv_mass() <= 0.0 {
            continue;
        }
        particle.velocity.x = (after[i].0 - before[i].0) / dt;
        particle.velocity.y = (after[i].1 - before[i].1) / dt;
        particle.velocity.z = (after[i].2 - before[i].2) / dt;
    }

    // Small bleed, so a plucked bridge settles instead of ringing forever.
    chain.apply_damping(0.02);
}

/// Solves a joint constraint between world objects.
fn solve_joint(
    joint: &mut WorldJoint3D,
    object_ids: &HashMap<ObjectId, usize>,
    objects: &mut [PhysicalObject3D],
    dt: f64,
) {
    // Get position 1 (either from object or anchor)
    let (x1, y1, z1, inv_mass1) = if let Some(id1) = joint.object1 {
        if let Some(&idx) = object_ids.get(&id1) {
            let obj = &objects[idx];
            let inv_mass = if obj.object.mass.is_infinite() { 0.0 } else { 1.0 / obj.object.mass };
            (obj.object.position.x, obj.object.position.y, obj.object.position.z, inv_mass)
        } else {
            return; // Object doesn't exist
        }
    } else if let Some(anchor) = joint.anchor {
        (anchor.0, anchor.1, anchor.2, 0.0) // Static anchor
    } else {
        return;
    };

    // Get position 2
    let idx2 = match object_ids.get(&joint.object2) {
        Some(&idx) => idx,
        None => return,
    };
    let obj2 = &objects[idx2];
    let inv_mass2 = if obj2.object.mass.is_infinite() { 0.0 } else { 1.0 / obj2.object.mass };
    let (x2, y2, z2) = (obj2.object.position.x, obj2.object.position.y, obj2.object.position.z);

    let total_inv_mass = inv_mass1 + inv_mass2;
    if total_inv_mass < 1e-10 {
        return;
    }

    // Calculate current distance
    let dx = x2 - x1;
    let dy = y2 - y1;
    let dz = z2 - z1;
    let current_dist = (dx * dx + dy * dy + dz * dz).sqrt();

    if current_dist < 1e-10 {
        return;
    }

    let error = current_dist - joint.distance;
    let correction = error * joint.baumgarte;

    // Normalize
    let nx = dx / current_dist;
    let ny = dy / current_dist;
    let nz = dz / current_dist;

    // Apply corrections
    if let Some(id1) = joint.object1 {
        if let Some(&idx) = object_ids.get(&id1) {
            let ratio = inv_mass1 / total_inv_mass;
            let obj = &mut objects[idx];
            obj.object.position.x += correction * ratio * nx;
            obj.object.position.y += correction * ratio * ny;
            obj.object.position.z += correction * ratio * nz;
        }
    }

    {
        let ratio = inv_mass2 / total_inv_mass;
        let obj = &mut objects[idx2];
        obj.object.position.x -= correction * ratio * nx;
        obj.object.position.y -= correction * ratio * ny;
        obj.object.position.z -= correction * ratio * nz;
    }

    joint.lambda += correction / dt;
}

/// Solves a spring constraint between world objects.
fn solve_spring(
    spring: &WorldSpring3D,
    object_ids: &HashMap<ObjectId, usize>,
    objects: &mut [PhysicalObject3D],
    dt: f64,
) {
    // Get position 1 and velocity 1
    let (x1, y1, z1, vx1, vy1, vz1, inv_mass1) = if let Some(id1) = spring.object1 {
        if let Some(&idx) = object_ids.get(&id1) {
            let obj = &objects[idx];
            let inv_mass = if obj.object.mass.is_infinite() { 0.0 } else { 1.0 / obj.object.mass };
            (
                obj.object.position.x, obj.object.position.y, obj.object.position.z,
                obj.object.velocity.x, obj.object.velocity.y, obj.object.velocity.z,
                inv_mass
            )
        } else {
            return;
        }
    } else if let Some(anchor) = spring.anchor {
        (anchor.0, anchor.1, anchor.2, 0.0, 0.0, 0.0, 0.0)
    } else {
        return;
    };

    // Get position 2 and velocity 2
    let idx2 = match object_ids.get(&spring.object2) {
        Some(&idx) => idx,
        None => return,
    };
    let obj2 = &objects[idx2];
    let inv_mass2 = if obj2.object.mass.is_infinite() { 0.0 } else { 1.0 / obj2.object.mass };
    let (x2, y2, z2) = (obj2.object.position.x, obj2.object.position.y, obj2.object.position.z);
    let (vx2, vy2, vz2) = (obj2.object.velocity.x, obj2.object.velocity.y, obj2.object.velocity.z);

    let total_inv_mass = inv_mass1 + inv_mass2;
    if total_inv_mass < 1e-10 {
        return;
    }

    // Calculate displacement and velocity
    let dx = x2 - x1;
    let dy = y2 - y1;
    let dz = z2 - z1;
    let current_dist = (dx * dx + dy * dy + dz * dz).sqrt();

    if current_dist < 1e-10 {
        return;
    }

    let extension = current_dist - spring.rest_length;

    // Normalize
    let nx = dx / current_dist;
    let ny = dy / current_dist;
    let nz = dz / current_dist;

    // Relative velocity along spring
    let rel_vx = vx2 - vx1;
    let rel_vy = vy2 - vy1;
    let rel_vz = vz2 - vz1;
    let rel_v_along_spring = rel_vx * nx + rel_vy * ny + rel_vz * nz;

    // Spring force: F = -k * extension - c * velocity
    let force_magnitude = spring.stiffness * extension + spring.damping * rel_v_along_spring;

    // Force direction (from obj2 toward obj1 when stretched)
    let fx = -force_magnitude * nx;
    let fy = -force_magnitude * ny;
    let fz = -force_magnitude * nz;

    // Apply forces as velocity changes
    if let Some(id1) = spring.object1 {
        if let Some(&idx) = object_ids.get(&id1) {
            let obj = &mut objects[idx];
            obj.object.velocity.x -= fx * inv_mass1 * dt;
            obj.object.velocity.y -= fy * inv_mass1 * dt;
            obj.object.velocity.z -= fz * inv_mass1 * dt;
        }
    }

    {
        let obj = &mut objects[idx2];
        obj.object.velocity.x += fx * inv_mass2 * dt;
        obj.object.velocity.y += fy * inv_mass2 * dt;
        obj.object.velocity.z += fz * inv_mass2 * dt;
    }
}

/// Solves a rope constraint between world objects.
fn solve_rope(
    rope: &mut WorldRope3D,
    object_ids: &HashMap<ObjectId, usize>,
    objects: &mut [PhysicalObject3D],
    dt: f64,
) {
    // Get position 1
    let (x1, y1, z1, inv_mass1) = if let Some(id1) = rope.object1 {
        if let Some(&idx) = object_ids.get(&id1) {
            let obj = &objects[idx];
            let inv_mass = if obj.object.mass.is_infinite() { 0.0 } else { 1.0 / obj.object.mass };
            (obj.object.position.x, obj.object.position.y, obj.object.position.z, inv_mass)
        } else {
            return;
        }
    } else if let Some(anchor) = rope.anchor {
        (anchor.0, anchor.1, anchor.2, 0.0)
    } else {
        return;
    };

    // Get position 2
    let idx2 = match object_ids.get(&rope.object2) {
        Some(&idx) => idx,
        None => return,
    };
    let obj2 = &objects[idx2];
    let inv_mass2 = if obj2.object.mass.is_infinite() { 0.0 } else { 1.0 / obj2.object.mass };
    let (x2, y2, z2) = (obj2.object.position.x, obj2.object.position.y, obj2.object.position.z);

    // Calculate current distance
    let dx = x2 - x1;
    let dy = y2 - y1;
    let dz = z2 - z1;
    let current_dist = (dx * dx + dy * dy + dz * dz).sqrt();

    // Rope only activates when stretched beyond max_length
    if current_dist <= rope.max_length {
        return;
    }

    let total_inv_mass = inv_mass1 + inv_mass2;
    if total_inv_mass < 1e-10 {
        return;
    }

    if current_dist < 1e-10 {
        return;
    }

    let error = current_dist - rope.max_length;
    let correction = error * rope.baumgarte;

    // Normalize
    let nx = dx / current_dist;
    let ny = dy / current_dist;
    let nz = dz / current_dist;

    // Apply corrections
    if let Some(id1) = rope.object1 {
        if let Some(&idx) = object_ids.get(&id1) {
            let ratio = inv_mass1 / total_inv_mass;
            let obj = &mut objects[idx];
            obj.object.position.x += correction * ratio * nx;
            obj.object.position.y += correction * ratio * ny;
            obj.object.position.z += correction * ratio * nz;
        }
    }

    {
        let ratio = inv_mass2 / total_inv_mass;
        let obj = &mut objects[idx2];
        obj.object.position.x -= correction * ratio * nx;
        obj.object.position.y -= correction * ratio * ny;
        obj.object.position.z -= correction * ratio * nz;
    }

    rope.lambda += correction / dt;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_id_unique() {
        let id1 = ConstraintId::new();
        let id2 = ConstraintId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_world_joint_creation() {
        let id1 = ObjectId::new();
        let id2 = ObjectId::new();
        let joint = WorldJoint3D::between(id1, id2, 5.0);
        assert_eq!(joint.distance, 5.0);
        assert!(joint.object1.is_some());
    }

    #[test]
    fn test_world_joint_anchored() {
        let id = ObjectId::new();
        let joint = WorldJoint3D::anchored((0.0, 10.0, 0.0), id, 2.0);
        assert!(joint.object1.is_none());
        assert!(joint.anchor.is_some());
    }

    #[test]
    fn test_world_spring_creation() {
        let id1 = ObjectId::new();
        let id2 = ObjectId::new();
        let spring = WorldSpring3D::between(id1, id2, 100.0, 1.0, 0.5);
        assert_eq!(spring.stiffness, 100.0);
        assert_eq!(spring.rest_length, 1.0);
        assert_eq!(spring.damping, 0.5);
    }

    #[test]
    fn test_world_rope_creation() {
        let id1 = ObjectId::new();
        let id2 = ObjectId::new();
        let rope = WorldRope3D::between(id1, id2, 5.0);
        assert_eq!(rope.max_length, 5.0);
    }
}
