//! State types for physics world snapshots
//!
//! These types are designed to be lightweight and efficiently cloneable
//! for passing through channels between the physics thread and main thread.

use std::sync::atomic::{AtomicU64, Ordering};

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

/// Complete state snapshot of the physics world
#[derive(Debug, Clone)]
pub struct WorldState {
    /// Current simulation tick number
    pub tick: u64,
    /// Current simulation time in seconds
    pub time: f64,
    /// States of all objects in the world
    pub objects: Vec<ObjectState>,
}

impl Default for WorldState {
    fn default() -> Self {
        Self {
            tick: 0,
            time: 0.0,
            objects: Vec::new(),
        }
    }
}

impl WorldState {
    /// Create a new empty world state
    pub fn new() -> Self {
        Self::default()
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
}
