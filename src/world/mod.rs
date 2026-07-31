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
//! - `StateBuffer`: Double-buffered snapshots enabling render-side interpolation
//! - `spawn_physics_thread`: Entry point to start background simulation
//!
//! # Timing
//!
//! The simulation advances at a fixed timestep, paced against wall-clock time,
//! on a thread of its own. That rate is deliberately unrelated to your display's
//! refresh rate. Reading raw snapshots from a renderer therefore stutters - some
//! frames land on a fresh tick and some repeat a stale one - so read through
//! [`PhysicsHandle::get_interpolated_state`], which blends between the two most
//! recent snapshots and stays smooth at 60, 144, 165 Hz or a variable rate.
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
//! // In the render loop - non-blocking, smooth at any refresh rate
//! let state = physics.get_interpolated_state();
//! for obj in &state.objects {
//!     println!("Object {} at {:?}", obj.id.0, obj.position);
//! }
//!
//! // For game logic, queries and tests, read the unblended simulation output
//! let exact = physics.get_latest_state();
//! ```

mod state;
mod config;
mod physics_world;
mod handle;
mod thread;

#[cfg(feature = "constraints")]
mod world_constraints;

pub use state::{ObjectId, ObjectState, StateBuffer, WorldState};
pub use config::WorldConfig;
pub use physics_world::{PhysicsWorld, ForceId, ContinuousForce};
pub use handle::{PhysicsHandle, PhysicsCommand};
pub use thread::{spawn_physics_thread, STATE_CHANNEL_CAPACITY};

#[cfg(feature = "constraints")]
pub use state::{
    ConstraintState, JointState, SpringState, RopeState, RopeChainState, HingeState,
};

#[cfg(feature = "constraints")]
pub use world_constraints::{
    ConstraintId, WorldConstraint, WorldJoint3D, WorldSpring3D, WorldRope3D,
};
