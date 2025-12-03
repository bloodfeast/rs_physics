//! Physics World Module
//!
//! Provides a continuous physics simulation that runs on a background thread,
//! allowing game engines like Bevy to query object positions every render frame
//! without blocking.
//!
//! # Architecture
//!
//! - `PhysicsWorld`: The simulation container managing all physics objects
//! - `PhysicsHandle`: Control interface for the main thread
//! - `WorldState`: Lightweight snapshot sent through channels
//! - `spawn_physics_thread`: Entry point to start background simulation
//!
//! # Example
//!
//! ```ignore
//! use rs_physics::world::{spawn_physics_thread, WorldConfig};
//!
//! let physics = spawn_physics_thread(WorldConfig::default());
//!
//! // Add objects
//! let ball_id = physics.add_object(...);
//!
//! // In render loop - non-blocking read
//! let state = physics.get_latest_state();
//! for obj in &state.objects {
//!     println!("Object {} at {:?}", obj.id.0, obj.position);
//! }
//! ```

mod state;
mod config;
mod physics_world;
mod handle;
mod thread;

pub use state::{ObjectId, ObjectState, WorldState};
pub use config::WorldConfig;
pub use physics_world::{PhysicsWorld, ForceId, ContinuousForce};
pub use handle::{PhysicsHandle, PhysicsCommand};
pub use thread::spawn_physics_thread;
