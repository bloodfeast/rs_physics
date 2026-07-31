//! PhysicsHandle - Thread-safe control interface for the physics simulation

use std::sync::{Arc, RwLock};
use crossbeam::channel::{Sender, Receiver, unbounded};
use crate::models::PhysicalObject3D;
use super::state::{ObjectId, StateBuffer, WorldState};
use super::physics_world::{ForceId, ContinuousForce};

#[cfg(feature = "constraints")]
use super::world_constraints::{ConstraintId, WorldConstraint};

/// Commands that can be sent to the physics thread
#[derive(Debug)]
pub enum PhysicsCommand {
    /// Add a new object to the world, returns ObjectId through oneshot channel
    AddObject {
        object: PhysicalObject3D,
        response: Sender<ObjectId>,
    },
    /// Remove an object from the world
    RemoveObject(ObjectId),
    /// Set the simulation timestep
    SetTimestep(f64),
    /// Apply a force to an object (x, y, z)
    ApplyForce(ObjectId, (f64, f64, f64)),
    /// Apply an impulse (instant velocity change) to an object
    ApplyImpulse(ObjectId, (f64, f64, f64)),
    /// Set the velocity of an object
    SetVelocity(ObjectId, (f64, f64, f64)),
    /// Set the position of an object
    SetPosition(ObjectId, (f64, f64, f64)),
    /// Apply a force in a direction with magnitude
    ApplyForceDirected(ObjectId, f64, (f64, f64, f64)),
    /// Apply force toward a target position
    ApplyForceToward(ObjectId, (f64, f64, f64), f64),
    /// Apply force away from a position
    ApplyForceAway(ObjectId, (f64, f64, f64), f64),
    /// Apply drag force (one-shot)
    ApplyDrag(ObjectId, f64),
    /// Apply spring force toward rest position (one-shot)
    ApplySpringForce(ObjectId, (f64, f64, f64), f64),
    /// Apply damped spring force (one-shot)
    ApplyDampedSpring(ObjectId, (f64, f64, f64), f64, f64),
    /// Apply explosion force to all objects in radius
    ApplyExplosion((f64, f64, f64), f64, f64),
    /// Apply torque to an object
    ApplyTorque(ObjectId, (f64, f64, f64)),
    /// Apply buoyancy force (one-shot)
    ApplyBuoyancy(ObjectId, f64, f64),
    // ==================== Continuous Force Commands ====================
    /// Add a continuous force, returns ForceId through response channel
    AddContinuousForce {
        force: ContinuousForce,
        response: Sender<ForceId>,
    },
    /// Remove a continuous force by ID
    RemoveContinuousForce(ForceId),
    /// Remove all continuous forces on an object
    RemoveForcesOnObject(ObjectId),
    /// Add continuous drag (auto-removes when slow)
    AddDrag {
        target: ObjectId,
        coefficient: f64,
        response: Sender<ForceId>,
    },
    /// Add continuous spring (auto-removes when at rest)
    AddSpring {
        target: ObjectId,
        rest_position: (f64, f64, f64),
        stiffness: f64,
        response: Sender<ForceId>,
    },
    /// Add continuous damped spring
    AddDampedSpring {
        target: ObjectId,
        rest_position: (f64, f64, f64),
        stiffness: f64,
        damping: f64,
        response: Sender<ForceId>,
    },
    /// Add continuous attraction
    AddAttraction {
        target: ObjectId,
        point: (f64, f64, f64),
        strength: f64,
        duration: Option<f64>,
        response: Sender<ForceId>,
    },
    /// Add continuous buoyancy
    AddBuoyancy {
        target: ObjectId,
        surface_y: f64,
        fluid_density: f64,
        response: Sender<ForceId>,
    },
    /// Add constant force (like wind)
    AddConstantForce {
        target: ObjectId,
        force: (f64, f64, f64),
        response: Sender<ForceId>,
    },
    /// Add continuous repulsion from a point
    AddRepulsion {
        target: ObjectId,
        point: (f64, f64, f64),
        strength: f64,
        duration: Option<f64>,
        response: Sender<ForceId>,
    },
    /// Add vortex/rotational force
    AddVortex {
        target: ObjectId,
        center: (f64, f64, f64),
        axis: (f64, f64, f64),
        strength: f64,
        duration: Option<f64>,
        response: Sender<ForceId>,
    },
    // ==================== Kinematic Object Commands ====================
    /// Set position of a kinematic object (computes velocity from displacement)
    SetPositionKinematic(ObjectId, (f64, f64, f64), f64), // id, position, dt
    // ==================== Constraint Commands (requires "constraints" feature) ====================
    /// Add a constraint to the world, returns ConstraintId through response channel
    #[cfg(feature = "constraints")]
    AddConstraint {
        constraint: WorldConstraint,
        response: Sender<ConstraintId>,
    },
    /// Remove a constraint from the world
    #[cfg(feature = "constraints")]
    RemoveConstraint(ConstraintId),
    // ==================== Control Commands ====================
    /// Pause the simulation
    Pause,
    /// Resume the simulation
    Resume,
    /// Shutdown the physics thread
    Shutdown,
}

/// Handle for controlling the physics simulation from the main thread
///
/// This is returned by `spawn_physics_thread()` and provides methods to:
/// - Query the latest physics state (non-blocking)
/// - Send commands to the physics thread
/// - Add/remove objects
/// - Control simulation (pause/resume/shutdown)
pub struct PhysicsHandle {
    /// Channel to send commands to the physics thread
    command_sender: Sender<PhysicsCommand>,
    /// Channel to receive state updates (optional, for polling)
    state_receiver: Receiver<WorldState>,
    /// Double-buffered latest state, shared with the physics thread
    latest_state: Arc<RwLock<StateBuffer>>,
}

impl PhysicsHandle {
    /// Create a new PhysicsHandle (called internally by spawn_physics_thread)
    pub(crate) fn new(
        command_sender: Sender<PhysicsCommand>,
        state_receiver: Receiver<WorldState>,
        latest_state: Arc<RwLock<StateBuffer>>,
    ) -> Self {
        Self {
            command_sender,
            state_receiver,
            latest_state,
        }
    }

    /// Read the shared buffer, tolerating a poisoned lock.
    ///
    /// A panic on the physics thread poisons the lock. Unwrapping here would
    /// turn that into a panic on the render thread every single frame, burying
    /// the original error. The buffered state is plain data and is still
    /// readable, so recover it and let the caller notice via a frozen `tick`.
    fn read_buffer(&self) -> std::sync::RwLockReadGuard<'_, StateBuffer> {
        self.latest_state.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Get the latest physics state (non-blocking)
    ///
    /// Returns the most recent snapshot exactly as the simulation produced it,
    /// with no blending. Use this when you want raw simulation output - logic,
    /// queries, tests, anything where an interpolated value would be wrong.
    ///
    /// For rendering, prefer [`Self::get_interpolated_state`].
    pub fn get_latest_state(&self) -> WorldState {
        self.read_buffer().current().clone()
    }

    /// Get the physics state blended for the current instant (non-blocking)
    ///
    /// This is the primary way to read physics state from a render thread. The
    /// simulation ticks at a fixed rate that has no relationship to your
    /// display's refresh rate; sampling it directly means some frames show a
    /// stale tick and some show a fresh one, which reads as stutter even though
    /// the simulation is perfectly regular.
    ///
    /// This blends between the last two snapshots based on how far into the
    /// current tick interval we are, so motion is smooth at any refresh rate,
    /// and stays smooth if the display rate changes mid-run. The cost is one
    /// tick of latency (4.2 ms at 240 Hz) - the alternative, extrapolating
    /// forward, overshoots whenever an object stops or bounces and then visibly
    /// snaps back.
    ///
    /// If the physics thread stalls, the blend saturates on the newest snapshot
    /// rather than drifting away from it.
    pub fn get_interpolated_state(&self) -> WorldState {
        self.read_buffer().sample()
    }

    /// How far the render clock is into the current physics tick, in `[0, 1]`
    ///
    /// Exposed for callers doing their own blending - for example interpolating
    /// only the handful of objects they actually draw, rather than paying for a
    /// full [`WorldState`] clone every frame.
    pub fn interpolation_alpha(&self) -> f64 {
        self.read_buffer().alpha()
    }

    /// Try to receive a state update from the channel (non-blocking)
    ///
    /// Returns `Some(state)` if a new state is available, `None` otherwise.
    /// Use this if you want to process every state update rather than
    /// just the latest one.
    ///
    /// The channel is bounded. If you do not drain it, the physics thread drops
    /// updates rather than queueing them forever - see
    /// [`STATE_CHANNEL_CAPACITY`](super::STATE_CHANNEL_CAPACITY). Callers that
    /// only ever read the newest state should use [`Self::get_latest_state`] or
    /// [`Self::get_interpolated_state`] and ignore this channel entirely.
    pub fn try_recv_state(&self) -> Option<WorldState> {
        self.state_receiver.try_recv().ok()
    }

    /// Send a command to the physics thread
    ///
    /// Returns `Ok(())` if the command was sent successfully,
    /// or `Err(())` if the physics thread has shut down.
    pub fn send_command(&self, cmd: PhysicsCommand) -> Result<(), ()> {
        self.command_sender.send(cmd).map_err(|_| ())
    }

    /// Add a new object to the physics world
    ///
    /// This blocks until the object is added and returns its ObjectId.
    pub fn add_object(&self, object: PhysicalObject3D) -> Result<ObjectId, ()> {
        let (response_tx, response_rx) = unbounded();

        self.send_command(PhysicsCommand::AddObject {
            object,
            response: response_tx,
        })?;

        // Wait for response
        response_rx.recv().map_err(|_| ())
    }

    /// Remove an object from the physics world
    pub fn remove_object(&self, id: ObjectId) -> Result<(), ()> {
        self.send_command(PhysicsCommand::RemoveObject(id))
    }

    /// Set the simulation timestep
    pub fn set_timestep(&self, dt: f64) -> Result<(), ()> {
        self.send_command(PhysicsCommand::SetTimestep(dt))
    }

    /// Apply a force to an object (raw 3D vector)
    pub fn apply_force(&self, id: ObjectId, force: (f64, f64, f64)) -> Result<(), ()> {
        self.send_command(PhysicsCommand::ApplyForce(id, force))
    }

    /// Apply an impulse to an object (instant velocity change)
    pub fn apply_impulse(&self, id: ObjectId, impulse: (f64, f64, f64)) -> Result<(), ()> {
        self.send_command(PhysicsCommand::ApplyImpulse(id, impulse))
    }

    /// Set the velocity of an object
    pub fn set_velocity(&self, id: ObjectId, velocity: (f64, f64, f64)) -> Result<(), ()> {
        self.send_command(PhysicsCommand::SetVelocity(id, velocity))
    }

    /// Set the position of an object
    pub fn set_position(&self, id: ObjectId, position: (f64, f64, f64)) -> Result<(), ()> {
        self.send_command(PhysicsCommand::SetPosition(id, position))
    }

    // ==================== Ergonomic Force Methods ====================

    /// Apply a force in a specific direction with given magnitude
    ///
    /// Direction is automatically normalized.
    /// ```ignore
    /// // Push object forward with 100N of force
    /// physics.apply_force_directed(id, 100.0, (1.0, 0.0, 0.0))?;
    /// ```
    pub fn apply_force_directed(&self, id: ObjectId, magnitude: f64, direction: (f64, f64, f64)) -> Result<(), ()> {
        self.send_command(PhysicsCommand::ApplyForceDirected(id, magnitude, direction))
    }

    /// Apply a force toward a target position (attraction)
    ///
    /// Useful for gravity wells, magnets, or AI-controlled movement.
    /// ```ignore
    /// // Pull object toward origin with 50N
    /// physics.apply_force_toward(id, (0.0, 0.0, 0.0), 50.0)?;
    /// ```
    pub fn apply_force_toward(&self, id: ObjectId, target: (f64, f64, f64), magnitude: f64) -> Result<(), ()> {
        self.send_command(PhysicsCommand::ApplyForceToward(id, target, magnitude))
    }

    /// Apply a force away from a position (repulsion)
    ///
    /// Useful for explosions, force fields, or avoidance.
    /// ```ignore
    /// // Push object away from player with 30N
    /// physics.apply_force_away(id, player_pos, 30.0)?;
    /// ```
    pub fn apply_force_away(&self, id: ObjectId, source: (f64, f64, f64), magnitude: f64) -> Result<(), ()> {
        self.send_command(PhysicsCommand::ApplyForceAway(id, source, magnitude))
    }

    /// Apply drag force based on current velocity
    ///
    /// drag_coefficient: typically 0.1 to 2.0 (higher = more drag)
    /// ```ignore
    /// // Apply air resistance
    /// physics.apply_drag(id, 0.5)?;
    /// ```
    pub fn apply_drag(&self, id: ObjectId, drag_coefficient: f64) -> Result<(), ()> {
        self.send_command(PhysicsCommand::ApplyDrag(id, drag_coefficient))
    }

    /// Apply spring force toward a rest position
    ///
    /// Uses Hooke's law: F = -k * displacement
    /// ```ignore
    /// // Tether object to anchor point
    /// physics.apply_spring_force(id, anchor_pos, 10.0)?;
    /// ```
    pub fn apply_spring_force(&self, id: ObjectId, rest_position: (f64, f64, f64), stiffness: f64) -> Result<(), ()> {
        self.send_command(PhysicsCommand::ApplySpringForce(id, rest_position, stiffness))
    }

    /// Apply damped spring force (spring + velocity damping)
    ///
    /// Combines spring force with damping to reduce oscillation.
    /// ```ignore
    /// // Smooth return to rest position
    /// physics.apply_damped_spring(id, rest_pos, 10.0, 2.0)?;
    /// ```
    pub fn apply_damped_spring(&self, id: ObjectId, rest_position: (f64, f64, f64), stiffness: f64, damping: f64) -> Result<(), ()> {
        self.send_command(PhysicsCommand::ApplyDampedSpring(id, rest_position, stiffness, damping))
    }

    /// Apply an explosion force to all objects within radius
    ///
    /// Force falls off with distance squared.
    /// ```ignore
    /// // Explosion at (0, 0, 0) with 1000N force, 10m radius
    /// physics.apply_explosion((0.0, 0.0, 0.0), 1000.0, 10.0)?;
    /// ```
    pub fn apply_explosion(&self, center: (f64, f64, f64), force: f64, radius: f64) -> Result<(), ()> {
        self.send_command(PhysicsCommand::ApplyExplosion(center, force, radius))
    }

    /// Apply torque to rotate an object
    /// ```ignore
    /// // Spin object around Y axis
    /// physics.apply_torque(id, (0.0, 1.0, 0.0))?;
    /// ```
    pub fn apply_torque(&self, id: ObjectId, torque: (f64, f64, f64)) -> Result<(), ()> {
        self.send_command(PhysicsCommand::ApplyTorque(id, torque))
    }

    /// Apply a buoyancy force (upward force based on depth below a surface)
    /// NOTE: This is a one-shot force. For continuous buoyancy, use add_buoyancy().
    ///
    /// surface_y: Y coordinate of the fluid surface
    /// fluid_density: density of the fluid (water ≈ 1000 kg/m³)
    /// ```ignore
    /// // Water surface at y=0, water density
    /// physics.apply_buoyancy(id, 0.0, 1000.0)?;
    /// ```
    pub fn apply_buoyancy(&self, id: ObjectId, surface_y: f64, fluid_density: f64) -> Result<(), ()> {
        self.send_command(PhysicsCommand::ApplyBuoyancy(id, surface_y, fluid_density))
    }

    // ==================== Continuous Force Methods ====================
    // These forces persist across simulation steps and auto-cleanup when their effect is negligible

    /// Add a continuous force (generic)
    ///
    /// Returns a ForceId that can be used to manually remove the force later.
    pub fn add_continuous_force(&self, force: ContinuousForce) -> Result<ForceId, ()> {
        let (response_tx, response_rx) = unbounded();
        self.send_command(PhysicsCommand::AddContinuousForce {
            force,
            response: response_tx,
        })?;
        response_rx.recv().map_err(|_| ())
    }

    /// Remove a continuous force by ID
    pub fn remove_continuous_force(&self, id: ForceId) -> Result<(), ()> {
        self.send_command(PhysicsCommand::RemoveContinuousForce(id))
    }

    /// Remove all continuous forces targeting an object
    pub fn remove_forces_on_object(&self, target: ObjectId) -> Result<(), ()> {
        self.send_command(PhysicsCommand::RemoveForcesOnObject(target))
    }

    /// Add continuous drag (auto-removes when velocity drops below threshold)
    ///
    /// Returns a ForceId for manual removal if needed.
    /// ```ignore
    /// let drag_id = physics.add_drag(ball_id, 0.5)?;
    /// // Drag will auto-remove when ball slows down
    /// // Or manually remove: physics.remove_continuous_force(drag_id)?;
    /// ```
    pub fn add_drag(&self, target: ObjectId, coefficient: f64) -> Result<ForceId, ()> {
        let (response_tx, response_rx) = unbounded();
        self.send_command(PhysicsCommand::AddDrag {
            target,
            coefficient,
            response: response_tx,
        })?;
        response_rx.recv().map_err(|_| ())
    }

    /// Add continuous spring force (auto-removes when at rest position)
    ///
    /// ```ignore
    /// // Tether ball to anchor - will oscillate then settle
    /// let spring_id = physics.add_spring(ball_id, anchor_pos, 10.0)?;
    /// ```
    pub fn add_spring(&self, target: ObjectId, rest_position: (f64, f64, f64), stiffness: f64) -> Result<ForceId, ()> {
        let (response_tx, response_rx) = unbounded();
        self.send_command(PhysicsCommand::AddSpring {
            target,
            rest_position,
            stiffness,
            response: response_tx,
        })?;
        response_rx.recv().map_err(|_| ())
    }

    /// Add continuous damped spring (auto-removes when settled)
    ///
    /// ```ignore
    /// // Smooth return to position with reduced oscillation
    /// let spring_id = physics.add_damped_spring(ball_id, rest_pos, 10.0, 2.0)?;
    /// ```
    pub fn add_damped_spring(&self, target: ObjectId, rest_position: (f64, f64, f64), stiffness: f64, damping: f64) -> Result<ForceId, ()> {
        let (response_tx, response_rx) = unbounded();
        self.send_command(PhysicsCommand::AddDampedSpring {
            target,
            rest_position,
            stiffness,
            damping,
            response: response_tx,
        })?;
        response_rx.recv().map_err(|_| ())
    }

    /// Add continuous attraction toward a point
    ///
    /// duration: Some(seconds) to auto-expire, None for permanent
    /// ```ignore
    /// // Temporary gravity well for 5 seconds
    /// let force_id = physics.add_attraction(ball_id, center, 50.0, Some(5.0))?;
    /// ```
    pub fn add_attraction(&self, target: ObjectId, point: (f64, f64, f64), strength: f64, duration: Option<f64>) -> Result<ForceId, ()> {
        let (response_tx, response_rx) = unbounded();
        self.send_command(PhysicsCommand::AddAttraction {
            target,
            point,
            strength,
            duration,
            response: response_tx,
        })?;
        response_rx.recv().map_err(|_| ())
    }

    /// Add continuous buoyancy (auto-removes when above surface for extended time)
    ///
    /// ```ignore
    /// // Ball will float in water
    /// let buoyancy_id = physics.add_buoyancy(ball_id, water_level, 1000.0)?;
    /// ```
    pub fn add_buoyancy(&self, target: ObjectId, surface_y: f64, fluid_density: f64) -> Result<ForceId, ()> {
        let (response_tx, response_rx) = unbounded();
        self.send_command(PhysicsCommand::AddBuoyancy {
            target,
            surface_y,
            fluid_density,
            response: response_tx,
        })?;
        response_rx.recv().map_err(|_| ())
    }

    /// Add a constant continuous force (like wind or thrust)
    ///
    /// Must be manually removed when no longer needed.
    /// ```ignore
    /// // Constant wind pushing objects
    /// let wind_id = physics.add_constant_force(ball_id, (10.0, 0.0, 0.0))?;
    /// ```
    pub fn add_constant_force(&self, target: ObjectId, force: (f64, f64, f64)) -> Result<ForceId, ()> {
        let (response_tx, response_rx) = unbounded();
        self.send_command(PhysicsCommand::AddConstantForce {
            target,
            force,
            response: response_tx,
        })?;
        response_rx.recv().map_err(|_| ())
    }

    /// Add continuous repulsion from a point (force field)
    ///
    /// duration: Some(seconds) to auto-expire, None for permanent
    /// ```ignore
    /// // Push objects away from hazard zone for 10 seconds
    /// let force_id = physics.add_repulsion(ball_id, hazard_center, 100.0, Some(10.0))?;
    /// ```
    pub fn add_repulsion(&self, target: ObjectId, point: (f64, f64, f64), strength: f64, duration: Option<f64>) -> Result<ForceId, ()> {
        let (response_tx, response_rx) = unbounded();
        self.send_command(PhysicsCommand::AddRepulsion {
            target,
            point,
            strength,
            duration,
            response: response_tx,
        })?;
        response_rx.recv().map_err(|_| ())
    }

    /// Add vortex/rotational force around an axis
    ///
    /// Creates a swirling force that pushes objects tangentially around the axis.
    /// duration: Some(seconds) to auto-expire, None for permanent
    /// ```ignore
    /// // Create tornado effect around Y axis
    /// let vortex_id = physics.add_vortex(ball_id, center, (0.0, 1.0, 0.0), 50.0, None)?;
    /// ```
    pub fn add_vortex(&self, target: ObjectId, center: (f64, f64, f64), axis: (f64, f64, f64), strength: f64, duration: Option<f64>) -> Result<ForceId, ()> {
        let (response_tx, response_rx) = unbounded();
        self.send_command(PhysicsCommand::AddVortex {
            target,
            center,
            axis,
            strength,
            duration,
            response: response_tx,
        })?;
        response_rx.recv().map_err(|_| ())
    }

    // ==================== Kinematic Object Methods ====================

    /// Set position of a kinematic object (computes velocity from displacement)
    ///
    /// This is used for externally-controlled objects (like rope bridge planks)
    /// that need to move smoothly while still participating in physics.
    /// ```ignore
    /// // Update plank position based on rope particle positions
    /// physics.set_position_kinematic(plank_id, new_pos, dt)?;
    /// ```
    pub fn set_position_kinematic(&self, id: ObjectId, position: (f64, f64, f64), dt: f64) -> Result<(), ()> {
        self.send_command(PhysicsCommand::SetPositionKinematic(id, position, dt))
    }

    // ==================== Constraint Methods (requires "constraints" feature) ====================

    /// Add a constraint to the world
    ///
    /// Returns a ConstraintId that can be used to remove the constraint later.
    /// ```ignore
    /// use rs_physics::world::WorldConstraint;
    ///
    /// // Add a rope chain constraint
    /// let constraint_id = physics.add_constraint(WorldConstraint::RopeChain { ... })?;
    /// ```
    #[cfg(feature = "constraints")]
    pub fn add_constraint(&self, constraint: WorldConstraint) -> Result<ConstraintId, ()> {
        let (response_tx, response_rx) = unbounded();
        self.send_command(PhysicsCommand::AddConstraint {
            constraint,
            response: response_tx,
        })?;
        response_rx.recv().map_err(|_| ())
    }

    /// Remove a constraint from the world
    #[cfg(feature = "constraints")]
    pub fn remove_constraint(&self, id: ConstraintId) -> Result<(), ()> {
        self.send_command(PhysicsCommand::RemoveConstraint(id))
    }

    // ==================== Simulation Control ====================

    /// Pause the simulation
    pub fn pause(&self) -> Result<(), ()> {
        self.send_command(PhysicsCommand::Pause)
    }

    /// Resume the simulation
    pub fn resume(&self) -> Result<(), ()> {
        self.send_command(PhysicsCommand::Resume)
    }

    /// Shutdown the physics thread
    ///
    /// After calling this, the handle is no longer usable.
    pub fn shutdown(&self) -> Result<(), ()> {
        self.send_command(PhysicsCommand::Shutdown)
    }

    // ==================== State Queries ====================

    /// Get the position of an object from the latest state
    ///
    /// Convenience method that combines get_latest_state() and lookup.
    pub fn get_position(&self, id: ObjectId) -> Option<(f64, f64, f64)> {
        self.get_latest_state().get_position(id)
    }

    /// Get the velocity of an object from the latest state
    pub fn get_velocity(&self, id: ObjectId) -> Option<(f64, f64, f64)> {
        self.get_latest_state().get_velocity(id)
    }

    /// Get the orientation of an object from the latest state as quaternion (x, y, z, w)
    pub fn get_orientation(&self, id: ObjectId) -> Option<(f64, f64, f64, f64)> {
        self.get_latest_state().get_orientation(id)
    }
}

// PhysicsHandle is automatically Send + Sync because all its fields are:
// - Sender<PhysicsCommand>: Send + Sync
// - Receiver<WorldState>: Send + Sync
// - Arc<RwLock<WorldState>>: Send + Sync
