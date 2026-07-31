//! State types for physics world snapshots
//!
//! These types are designed to be lightweight and efficiently cloneable
//! for passing through channels between the physics thread and main thread.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[cfg(feature = "constraints")]
use super::world_constraints::ConstraintId;

/// Spherical linear interpolation between two quaternions `(x, y, z, w)`.
///
/// Takes the shortest arc (negating `b` when the quaternions are more than 90°
/// apart) and falls back to normalized lerp when the inputs are nearly parallel,
/// where `sin(theta)` underflows.
fn slerp_quat(
    a: (f64, f64, f64, f64),
    b: (f64, f64, f64, f64),
    t: f64,
) -> (f64, f64, f64, f64) {
    let mut dot = a.0 * b.0 + a.1 * b.1 + a.2 * b.2 + a.3 * b.3;

    // Shortest path: q and -q describe the same orientation.
    let b = if dot < 0.0 {
        dot = -dot;
        (-b.0, -b.1, -b.2, -b.3)
    } else {
        b
    };

    // Nearly parallel - slerp is numerically unstable here, and lerp is accurate.
    const PARALLEL_THRESHOLD: f64 = 0.9995;
    let (wa, wb) = if dot > PARALLEL_THRESHOLD {
        (1.0 - t, t)
    } else {
        let theta = dot.clamp(-1.0, 1.0).acos();
        let sin_theta = theta.sin();
        (((1.0 - t) * theta).sin() / sin_theta, (t * theta).sin() / sin_theta)
    };

    let q = (
        a.0 * wa + b.0 * wb,
        a.1 * wa + b.1 * wb,
        a.2 * wa + b.2 * wb,
        a.3 * wa + b.3 * wb,
    );

    let len_sq = q.0 * q.0 + q.1 * q.1 + q.2 * q.2 + q.3 * q.3;
    if len_sq < 1e-12 {
        // Degenerate input (a zero quaternion); fall back to the target.
        return b;
    }
    let inv_len = 1.0 / len_sq.sqrt();
    (q.0 * inv_len, q.1 * inv_len, q.2 * inv_len, q.3 * inv_len)
}

/// Linear interpolation between two 3-tuples.
#[inline]
fn lerp3(a: (f64, f64, f64), b: (f64, f64, f64), t: f64) -> (f64, f64, f64) {
    (
        a.0 + (b.0 - a.0) * t,
        a.1 + (b.1 - a.1) * t,
        a.2 + (b.2 - a.2) * t,
    )
}

/// Unique identifier for physics objects
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectId(pub u64);

impl ObjectId {
    /// Generate a new unique ObjectId
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        ObjectId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for ObjectId {
    fn default() -> Self {
        Self::new()
    }
}

/// State snapshot of a single physics object
#[derive(Debug, Clone)]
pub struct ObjectState {
    /// Unique identifier
    pub id: ObjectId,
    /// Position in world space (x, y, z)
    pub position: (f64, f64, f64),
    /// Orientation as quaternion (x, y, z, w)
    pub orientation: (f64, f64, f64, f64),
    /// Linear velocity (x, y, z)
    pub velocity: (f64, f64, f64),
    /// Angular velocity (x, y, z)
    pub angular_velocity: (f64, f64, f64),
}

impl Default for ObjectState {
    fn default() -> Self {
        Self {
            id: ObjectId(0),
            position: (0.0, 0.0, 0.0),
            orientation: (0.0, 0.0, 0.0, 1.0),  // Identity quaternion
            velocity: (0.0, 0.0, 0.0),
            angular_velocity: (0.0, 0.0, 0.0),
        }
    }
}

impl ObjectState {
    /// Blend between this state and `next` at `t` in `[0, 1]`.
    ///
    /// Position and velocities interpolate linearly; orientation uses slerp.
    /// The result carries `next`'s id.
    pub fn interpolate(&self, next: &ObjectState, t: f64) -> ObjectState {
        ObjectState {
            id: next.id,
            position: lerp3(self.position, next.position, t),
            orientation: slerp_quat(self.orientation, next.orientation, t),
            velocity: lerp3(self.velocity, next.velocity, t),
            angular_velocity: lerp3(self.angular_velocity, next.angular_velocity, t),
        }
    }
}

// ============================================================================
// Constraint State Types (requires "constraints" feature)
// ============================================================================

/// State snapshot of a joint constraint
#[cfg(feature = "constraints")]
#[derive(Debug, Clone)]
pub struct JointState {
    /// Unique identifier for this constraint
    pub id: ConstraintId,
    /// First object (None if anchored to world)
    pub object1: Option<ObjectId>,
    /// Second object
    pub object2: ObjectId,
    /// Anchor position (if object1 is None)
    pub anchor: Option<(f64, f64, f64)>,
    /// Target distance
    pub distance: f64,
}

/// State snapshot of a spring constraint
#[cfg(feature = "constraints")]
#[derive(Debug, Clone)]
pub struct SpringState {
    /// Unique identifier for this constraint
    pub id: ConstraintId,
    /// First object (None if anchored to world)
    pub object1: Option<ObjectId>,
    /// Second object
    pub object2: ObjectId,
    /// Anchor position (if object1 is None)
    pub anchor: Option<(f64, f64, f64)>,
    /// Spring stiffness
    pub stiffness: f64,
    /// Rest length
    pub rest_length: f64,
}

/// State snapshot of a rope constraint
#[cfg(feature = "constraints")]
#[derive(Debug, Clone)]
pub struct RopeState {
    /// Unique identifier for this constraint
    pub id: ConstraintId,
    /// First object (None if anchored to world)
    pub object1: Option<ObjectId>,
    /// Second object
    pub object2: ObjectId,
    /// Anchor position (if object1 is None)
    pub anchor: Option<(f64, f64, f64)>,
    /// Maximum rope length
    pub max_length: f64,
}

/// State snapshot of a rope chain (multi-segment rope with internal particles)
#[cfg(feature = "constraints")]
#[derive(Debug, Clone)]
pub struct RopeChainState {
    /// Unique identifier for this constraint
    pub id: ConstraintId,
    /// Anchor position
    pub anchor: (f64, f64, f64),
    /// Positions of all particles in the chain
    pub particle_positions: Vec<(f64, f64, f64)>,
    /// Segment length
    pub segment_length: f64,
}

/// State snapshot of a hinge constraint
#[cfg(feature = "constraints")]
#[derive(Debug, Clone)]
pub struct HingeState {
    /// Unique identifier for this constraint
    pub id: ConstraintId,
    /// Hinge anchor position
    pub anchor: (f64, f64, f64),
    /// Hinge axis (normalized)
    pub axis: (f64, f64, f64),
    /// Current angle in radians
    pub angle: f64,
    /// Angular velocity in rad/s
    pub angular_velocity: f64,
    /// Angle limits (min, max) if any
    pub limits: Option<(f64, f64)>,
}

/// State snapshot of any constraint type
#[cfg(feature = "constraints")]
#[derive(Debug, Clone)]
pub enum ConstraintState {
    /// Distance joint between objects
    Joint(JointState),
    /// Spring connection between objects
    Spring(SpringState),
    /// Rope (max length) connection between objects
    Rope(RopeState),
    /// Multi-segment rope chain with internal particles
    RopeChain(RopeChainState),
    /// Angular hinge constraint
    Hinge(HingeState),
}

#[cfg(feature = "constraints")]
impl ConstraintState {
    /// Returns the constraint ID
    pub fn id(&self) -> ConstraintId {
        match self {
            ConstraintState::Joint(j) => j.id,
            ConstraintState::Spring(s) => s.id,
            ConstraintState::Rope(r) => r.id,
            ConstraintState::RopeChain(rc) => rc.id,
            ConstraintState::Hinge(h) => h.id,
        }
    }

    /// Blend between this constraint snapshot and `next` at `t` in `[0, 1]`.
    ///
    /// Only the fields that change per tick are blended - anchors, rope-chain
    /// particle positions, and hinge angles. Configuration fields (stiffness,
    /// rest length, limits) are taken from `next`. Mismatched variants or
    /// differing particle counts fall back to `next` for the affected values,
    /// so a rope that gains segments pops rather than smearing across the
    /// whole chain.
    pub fn interpolate(&self, next: &ConstraintState, t: f64) -> ConstraintState {
        let t = t.clamp(0.0, 1.0);

        match (self, next) {
            (ConstraintState::Joint(a), ConstraintState::Joint(b)) => {
                ConstraintState::Joint(JointState {
                    anchor: lerp_opt3(a.anchor, b.anchor, t),
                    ..b.clone()
                })
            }
            (ConstraintState::Spring(a), ConstraintState::Spring(b)) => {
                ConstraintState::Spring(SpringState {
                    anchor: lerp_opt3(a.anchor, b.anchor, t),
                    ..b.clone()
                })
            }
            (ConstraintState::Rope(a), ConstraintState::Rope(b)) => {
                ConstraintState::Rope(RopeState {
                    anchor: lerp_opt3(a.anchor, b.anchor, t),
                    ..b.clone()
                })
            }
            (ConstraintState::RopeChain(a), ConstraintState::RopeChain(b)) => {
                let particle_positions = if a.particle_positions.len() == b.particle_positions.len()
                {
                    a.particle_positions
                        .iter()
                        .zip(b.particle_positions.iter())
                        .map(|(pa, pb)| lerp3(*pa, *pb, t))
                        .collect()
                } else {
                    b.particle_positions.clone()
                };

                ConstraintState::RopeChain(RopeChainState {
                    anchor: lerp3(a.anchor, b.anchor, t),
                    particle_positions,
                    ..b.clone()
                })
            }
            (ConstraintState::Hinge(a), ConstraintState::Hinge(b)) => {
                ConstraintState::Hinge(HingeState {
                    anchor: lerp3(a.anchor, b.anchor, t),
                    angle: lerp_angle(a.angle, b.angle, t),
                    angular_velocity: a.angular_velocity
                        + (b.angular_velocity - a.angular_velocity) * t,
                    ..b.clone()
                })
            }
            // Variant changed for this id - nothing meaningful to blend.
            _ => next.clone(),
        }
    }
}

/// Interpolate optional anchors, blending only when both sides are present.
#[cfg(feature = "constraints")]
#[inline]
fn lerp_opt3(
    a: Option<(f64, f64, f64)>,
    b: Option<(f64, f64, f64)>,
    t: f64,
) -> Option<(f64, f64, f64)> {
    match (a, b) {
        (Some(a), Some(b)) => Some(lerp3(a, b, t)),
        _ => b,
    }
}

/// Interpolate an angle in radians along the shortest arc.
///
/// A hinge crossing the -pi/pi boundary would otherwise sweep the long way
/// round, which reads as the object snapping backwards for one frame.
#[cfg(feature = "constraints")]
#[inline]
fn lerp_angle(a: f64, b: f64, t: f64) -> f64 {
    use std::f64::consts::{PI, TAU};
    let mut delta = (b - a) % TAU;
    if delta > PI {
        delta -= TAU;
    } else if delta < -PI {
        delta += TAU;
    }
    a + delta * t
}

/// Complete state snapshot of the physics world
#[derive(Debug, Clone)]
pub struct WorldState {
    /// Current simulation tick number
    pub tick: u64,
    /// Current simulation time in seconds
    pub time: f64,
    /// States of all objects in the world
    pub objects: Vec<ObjectState>,
    /// States of all constraints in the world (requires "constraints" feature)
    #[cfg(feature = "constraints")]
    pub constraints: Vec<ConstraintState>,
}

impl Default for WorldState {
    fn default() -> Self {
        Self {
            tick: 0,
            time: 0.0,
            objects: Vec::new(),
            #[cfg(feature = "constraints")]
            constraints: Vec::new(),
        }
    }
}

impl WorldState {
    /// Create a new empty world state
    pub fn new() -> Self {
        Self::default()
    }

    /// Blend between this snapshot and `next` at `t` in `[0, 1]`.
    ///
    /// Objects are matched by [`ObjectId`], not by position in the vector, so
    /// this stays correct across additions and removals. An object present in
    /// `next` but not in `self` (spawned during the interval) is taken from
    /// `next` unblended - there is no earlier state to blend from. Objects only
    /// in `self` (removed during the interval) are dropped.
    ///
    /// `tick` is taken from `next`; `time` interpolates.
    pub fn interpolate(&self, next: &WorldState, t: f64) -> WorldState {
        let t = t.clamp(0.0, 1.0);

        let mut objects = Vec::with_capacity(next.objects.len());
        for (i, cur) in next.objects.iter().enumerate() {
            // The two snapshots almost always share ordering, so check the same
            // index first and only fall back to a search when that misses.
            let prev = self
                .objects
                .get(i)
                .filter(|p| p.id == cur.id)
                .or_else(|| self.objects.iter().find(|p| p.id == cur.id));

            objects.push(match prev {
                Some(p) => p.interpolate(cur, t),
                None => cur.clone(),
            });
        }

        WorldState {
            tick: next.tick,
            time: self.time + (next.time - self.time) * t,
            objects,
            #[cfg(feature = "constraints")]
            constraints: {
                let mut out = Vec::with_capacity(next.constraints.len());
                for (i, cur) in next.constraints.iter().enumerate() {
                    let prev = self
                        .constraints
                        .get(i)
                        .filter(|p| p.id() == cur.id())
                        .or_else(|| self.constraints.iter().find(|p| p.id() == cur.id()));

                    out.push(match prev {
                        Some(p) => p.interpolate(cur, t),
                        None => cur.clone(),
                    });
                }
                out
            },
        }
    }

    /// Find an object state by its ID
    pub fn get_object(&self, id: ObjectId) -> Option<&ObjectState> {
        self.objects.iter().find(|obj| obj.id == id)
    }

    /// Get position of an object by ID, returns None if not found
    pub fn get_position(&self, id: ObjectId) -> Option<(f64, f64, f64)> {
        self.get_object(id).map(|obj| obj.position)
    }

    /// Get orientation of an object by ID as quaternion (x, y, z, w)
    pub fn get_orientation(&self, id: ObjectId) -> Option<(f64, f64, f64, f64)> {
        self.get_object(id).map(|obj| obj.orientation)
    }

    /// Get velocity of an object by ID
    pub fn get_velocity(&self, id: ObjectId) -> Option<(f64, f64, f64)> {
        self.get_object(id).map(|obj| obj.velocity)
    }

    /// Get angular velocity of an object by ID
    pub fn get_angular_velocity(&self, id: ObjectId) -> Option<(f64, f64, f64)> {
        self.get_object(id).map(|obj| obj.angular_velocity)
    }

    // ========================================================================
    // Constraint query methods (requires "constraints" feature)
    // ========================================================================

    /// Find a constraint state by its ID
    #[cfg(feature = "constraints")]
    pub fn get_constraint(&self, id: ConstraintId) -> Option<&ConstraintState> {
        self.constraints.iter().find(|c| c.id() == id)
    }

    /// Get the number of constraints in the world
    #[cfg(feature = "constraints")]
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// Get all rope chain particle positions for a given constraint ID
    #[cfg(feature = "constraints")]
    pub fn get_rope_chain_particles(&self, id: ConstraintId) -> Option<&Vec<(f64, f64, f64)>> {
        match self.get_constraint(id)? {
            ConstraintState::RopeChain(rc) => Some(&rc.particle_positions),
            _ => None,
        }
    }

    /// Get hinge state (angle, angular velocity) for a given constraint ID
    #[cfg(feature = "constraints")]
    pub fn get_hinge_state(&self, id: ConstraintId) -> Option<(f64, f64)> {
        match self.get_constraint(id)? {
            ConstraintState::Hinge(h) => Some((h.angle, h.angular_velocity)),
            _ => None,
        }
    }

    /// Iterate over all constraint states
    #[cfg(feature = "constraints")]
    pub fn constraints(&self) -> impl Iterator<Item = &ConstraintState> {
        self.constraints.iter()
    }
}

/// Double-buffered world state plus the timing needed to blend between the two.
///
/// The physics thread publishes a snapshot here on every broadcast; readers
/// sample it at whatever rate they render at. Sampling blends from the
/// second-most-recent snapshot toward the most recent, which means rendering
/// trails the simulation by one broadcast interval (8.3 ms at 120 Hz) but never
/// extrapolates past known state. Extrapolation is what produces overshoot and
/// visible correction snaps when an object stops abruptly.
#[derive(Debug)]
pub struct StateBuffer {
    previous: WorldState,
    current: WorldState,
    published_at: Instant,
    interval: Duration,
}

impl StateBuffer {
    /// Create a buffer for a publisher broadcasting every `interval`.
    pub fn new(interval: Duration) -> Self {
        Self {
            previous: WorldState::default(),
            current: WorldState::default(),
            published_at: Instant::now(),
            interval,
        }
    }

    /// Store a new snapshot, retiring the previous one.
    pub fn publish(&mut self, state: WorldState) {
        std::mem::swap(&mut self.previous, &mut self.current);
        self.current = state;
        self.published_at = Instant::now();
    }

    /// The most recent snapshot, unblended.
    pub fn current(&self) -> &WorldState {
        &self.current
    }

    /// How far into the current interval we are, clamped to `[0, 1]`.
    ///
    /// Clamping is what makes a stalled or paused publisher degrade gracefully:
    /// the blend parks on the newest snapshot instead of extrapolating away.
    pub fn alpha(&self) -> f64 {
        if self.interval.is_zero() {
            return 1.0;
        }
        (self.published_at.elapsed().as_secs_f64() / self.interval.as_secs_f64()).clamp(0.0, 1.0)
    }

    /// Blend the two buffered snapshots for the current wall-clock time.
    pub fn sample(&self) -> WorldState {
        self.previous.interpolate(&self.current, self.alpha())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_id_uniqueness() {
        let id1 = ObjectId::new();
        let id2 = ObjectId::new();
        let id3 = ObjectId::new();

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_world_state_get_object() {
        let mut state = WorldState::new();
        let id = ObjectId::new();

        state.objects.push(ObjectState {
            id,
            position: (1.0, 2.0, 3.0),
            ..Default::default()
        });

        let obj = state.get_object(id);
        assert!(obj.is_some());
        assert_eq!(obj.unwrap().position, (1.0, 2.0, 3.0));

        let missing = state.get_object(ObjectId::new());
        assert!(missing.is_none());
    }

    fn obj_at(id: ObjectId, pos: (f64, f64, f64)) -> ObjectState {
        ObjectState { id, position: pos, ..Default::default() }
    }

    #[test]
    fn test_interpolate_position_midpoint() {
        let id = ObjectId::new();
        let mut a = WorldState::new();
        let mut b = WorldState::new();
        a.objects.push(obj_at(id, (0.0, 0.0, 0.0)));
        b.objects.push(obj_at(id, (10.0, -4.0, 2.0)));
        b.tick = 7;
        b.time = 1.0;

        let mid = a.interpolate(&b, 0.5);
        let p = mid.get_position(id).unwrap();
        assert!((p.0 - 5.0).abs() < 1e-12);
        assert!((p.1 + 2.0).abs() < 1e-12);
        assert!((p.2 - 1.0).abs() < 1e-12);
        assert_eq!(mid.tick, 7, "tick should come from the newer snapshot");
        assert!((mid.time - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_interpolate_clamps_out_of_range_alpha() {
        let id = ObjectId::new();
        let mut a = WorldState::new();
        let mut b = WorldState::new();
        a.objects.push(obj_at(id, (0.0, 0.0, 0.0)));
        b.objects.push(obj_at(id, (10.0, 0.0, 0.0)));

        // A stalled publisher must park on the newest state, never overshoot.
        assert_eq!(a.interpolate(&b, 3.0).get_position(id).unwrap().0, 10.0);
        assert_eq!(a.interpolate(&b, -1.0).get_position(id).unwrap().0, 0.0);
    }

    #[test]
    fn test_interpolate_matches_by_id_not_index() {
        let (id1, id2) = (ObjectId::new(), ObjectId::new());
        let mut a = WorldState::new();
        let mut b = WorldState::new();
        a.objects.push(obj_at(id1, (0.0, 0.0, 0.0)));
        a.objects.push(obj_at(id2, (100.0, 0.0, 0.0)));
        // Reversed ordering - index-based matching would blend the wrong pair.
        b.objects.push(obj_at(id2, (110.0, 0.0, 0.0)));
        b.objects.push(obj_at(id1, (10.0, 0.0, 0.0)));

        let mid = a.interpolate(&b, 0.5);
        assert!((mid.get_position(id1).unwrap().0 - 5.0).abs() < 1e-12);
        assert!((mid.get_position(id2).unwrap().0 - 105.0).abs() < 1e-12);
    }

    #[test]
    fn test_interpolate_spawned_object_is_not_blended() {
        let (old, fresh) = (ObjectId::new(), ObjectId::new());
        let mut a = WorldState::new();
        let mut b = WorldState::new();
        a.objects.push(obj_at(old, (0.0, 0.0, 0.0)));
        b.objects.push(obj_at(old, (10.0, 0.0, 0.0)));
        b.objects.push(obj_at(fresh, (50.0, 0.0, 0.0)));

        let mid = a.interpolate(&b, 0.5);
        assert_eq!(mid.objects.len(), 2);
        // Blending from the origin would drag a spawned object in from (0,0,0).
        assert_eq!(mid.get_position(fresh).unwrap().0, 50.0);
    }

    #[test]
    fn test_interpolate_drops_removed_object() {
        let (kept, removed) = (ObjectId::new(), ObjectId::new());
        let mut a = WorldState::new();
        let mut b = WorldState::new();
        a.objects.push(obj_at(kept, (0.0, 0.0, 0.0)));
        a.objects.push(obj_at(removed, (0.0, 0.0, 0.0)));
        b.objects.push(obj_at(kept, (10.0, 0.0, 0.0)));

        let mid = a.interpolate(&b, 0.5);
        assert_eq!(mid.objects.len(), 1);
        assert!(mid.get_object(removed).is_none());
    }

    #[test]
    fn test_slerp_endpoints_and_normalization() {
        let a = (0.0, 0.0, 0.0, 1.0);
        // 90 degrees about Y
        let h = std::f64::consts::FRAC_PI_4;
        let b = (0.0, h.sin(), 0.0, h.cos());

        assert_eq!(slerp_quat(a, b, 0.0), a);
        let end = slerp_quat(a, b, 1.0);
        assert!((end.1 - b.1).abs() < 1e-12 && (end.3 - b.3).abs() < 1e-12);

        let mid = slerp_quat(a, b, 0.5);
        let len = (mid.0 * mid.0 + mid.1 * mid.1 + mid.2 * mid.2 + mid.3 * mid.3).sqrt();
        assert!((len - 1.0).abs() < 1e-12, "slerp must stay unit length, got {len}");
    }

    #[test]
    fn test_slerp_takes_shortest_arc() {
        let a = (0.0, 0.0, 0.0, 1.0);
        // Same orientation as `a` expressed with the opposite sign, rotated a little.
        let t = 0.1_f64;
        let b = (0.0, -(t.sin()), 0.0, -(t.cos()));

        let mid = slerp_quat(a, b, 0.5);
        // Shortest arc keeps w positive; the long way round would flip it.
        assert!(mid.3 > 0.99, "expected short arc, got w = {}", mid.3);
    }

    #[cfg(feature = "constraints")]
    #[test]
    fn test_lerp_angle_wraps_shortest_way() {
        use std::f64::consts::PI;
        // Just below +pi to just above -pi is a small step, not a full sweep back.
        let a = PI - 0.1;
        let b = -PI + 0.1;
        let mid = lerp_angle(a, b, 0.5);
        // Midpoint should sit at the wrap boundary, not near 0.
        assert!(mid.abs() > PI - 1e-9, "expected wrap through pi, got {mid}");
    }

    #[test]
    fn test_state_buffer_alpha_clamps_and_samples() {
        let mut buf = StateBuffer::new(Duration::from_millis(10));
        let id = ObjectId::new();

        let mut first = WorldState::new();
        first.objects.push(obj_at(id, (0.0, 0.0, 0.0)));
        buf.publish(first);

        let mut second = WorldState::new();
        second.objects.push(obj_at(id, (10.0, 0.0, 0.0)));
        buf.publish(second);

        assert_eq!(buf.current().get_position(id).unwrap().0, 10.0);

        // Well past one interval: alpha saturates and sampling yields `current`.
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(buf.alpha(), 1.0);
        assert_eq!(buf.sample().get_position(id).unwrap().0, 10.0);
    }

    #[test]
    fn test_state_buffer_zero_interval_is_not_a_divide_by_zero() {
        let buf = StateBuffer::new(Duration::ZERO);
        assert_eq!(buf.alpha(), 1.0);
    }
}
