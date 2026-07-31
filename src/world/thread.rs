//! Physics thread spawning and main loop

use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use crossbeam::channel::{bounded, unbounded, Receiver, Sender, TryRecvError};

use super::config::WorldConfig;
use super::handle::{PhysicsCommand, PhysicsHandle};
use super::physics_world::PhysicsWorld;
use super::state::{StateBuffer, WorldState};

/// Capacity of the state broadcast channel.
///
/// The channel exists for subscribers that want every tick rather than just the
/// newest one. It is bounded because most callers never read it at all - they
/// use `get_interpolated_state()` instead - and an unbounded channel nobody
/// drains is an unbounded memory leak that grows for as long as the simulation
/// runs. When the buffer is full the physics thread drops the update and keeps
/// stepping; a lagging subscriber must not be able to stall the simulation.
pub const STATE_CHANNEL_CAPACITY: usize = 64;

/// Raises OS timer resolution for the lifetime of the value.
///
/// Windows' default timer granularity is ~15.6 ms. A 240 Hz loop needs 4.2 ms
/// waits, so at the default granularity every `sleep` overshoots its deadline
/// and the simulation runs at roughly 64 Hz - irregularly, which is worse for
/// smoothness than simply running slower. `timeBeginPeriod(1)` drops granularity
/// to about 1 ms for the process. Without it the pacer below is forced to spin
/// the entire interval and burn a core.
///
/// On other platforms `nanosleep` is already fine-grained and this is a no-op.
#[cfg(windows)]
struct TimerResolutionGuard;

#[cfg(windows)]
#[link(name = "winmm")]
extern "system" {
    fn timeBeginPeriod(period: u32) -> u32;
    fn timeEndPeriod(period: u32) -> u32;
}

#[cfg(windows)]
impl TimerResolutionGuard {
    const PERIOD_MS: u32 = 1;

    fn acquire() -> Self {
        // SAFETY: timeBeginPeriod takes a scalar and has no precondition beyond
        // being matched by a timeEndPeriod with the same argument, which the
        // Drop impl below guarantees for every construction of this value.
        unsafe {
            timeBeginPeriod(Self::PERIOD_MS);
        }
        Self
    }
}

#[cfg(windows)]
impl Drop for TimerResolutionGuard {
    fn drop(&mut self) {
        // SAFETY: paired with the timeBeginPeriod in `acquire`.
        unsafe {
            timeEndPeriod(Self::PERIOD_MS);
        }
    }
}

#[cfg(not(windows))]
struct TimerResolutionGuard;

#[cfg(not(windows))]
impl TimerResolutionGuard {
    fn acquire() -> Self {
        Self
    }
}

/// Holds a fixed tick rate against wall-clock time.
///
/// Two things make this different from `sleep(period - elapsed)`:
///
/// 1. **Absolute deadlines.** Each tick targets `start + n * period` rather than
///    `period` from wherever the last tick happened to finish. Per-tick timing
///    error would otherwise accumulate, and the simulation would drift away from
///    real time permanently.
/// 2. **Measured sleep granularity.** Sleeping the whole remaining interval
///    overshoots on any OS whose timer is coarser than the interval. The pacer
///    sleeps in short hops while the remaining time comfortably exceeds the
///    granularity it has actually observed, then spins out the last fraction.
struct Pacer {
    period: Duration,
    next_deadline: Instant,
    /// Wall-clock cost of the shortest sleep this OS actually performs.
    sleep_granularity: Duration,
    max_catchup: u32,
}

impl Pacer {
    fn new(period: Duration, max_catchup: u32) -> Self {
        Self {
            period,
            next_deadline: Instant::now() + period,
            // Optimistic seed, corrected upward by the first few measurements.
            sleep_granularity: Duration::from_micros(1_500),
            max_catchup: max_catchup.max(1),
        }
    }

    /// Block until the next tick is due, then claim that deadline.
    ///
    /// Returns the number of ticks of debt that were abandoned, which is
    /// non-zero only when the simulation could not keep up.
    fn wait_for_next_tick(&mut self) -> u32 {
        let now = Instant::now();

        // Already past the deadline: the last tick overran. Run the next one
        // immediately to catch up, unless the backlog is beyond recovery.
        if now >= self.next_deadline {
            let debt = now - self.next_deadline;
            let budget = self.period * self.max_catchup;

            if debt > budget {
                let dropped = (debt.as_secs_f64() / self.period.as_secs_f64()) as u32;
                self.next_deadline = now + self.period;
                return dropped;
            }

            self.next_deadline += self.period;
            return 0;
        }

        loop {
            let remaining = self.next_deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }

            if remaining > self.sleep_granularity {
                let before = Instant::now();
                std::thread::sleep(Duration::from_millis(1));
                self.observe_sleep(before.elapsed());
            } else {
                std::hint::spin_loop();
            }
        }

        self.next_deadline += self.period;
        0
    }

    /// Update the granularity estimate from an observed sleep.
    fn observe_sleep(&mut self, actual: Duration) {
        if actual > self.sleep_granularity {
            // Rise immediately: underestimating granularity means overshooting
            // deadlines, which is the failure this whole mechanism exists to avoid.
            self.sleep_granularity = actual;
        } else {
            // Fall slowly, so one lucky sleep on a busy machine does not make
            // the pacer over-confident and cost it the next deadline.
            self.sleep_granularity = (self.sleep_granularity * 7 + actual) / 8;
        }
    }
}

/// Spawn a physics simulation on a background thread
///
/// Returns a `PhysicsHandle` that can be used to:
/// - Query the latest physics state (non-blocking)
/// - Send commands to add/remove objects
/// - Control the simulation (pause/resume/shutdown)
///
/// # Example
///
/// ```ignore
/// use rs_physics::world::{spawn_physics_thread, WorldConfig};
///
/// let physics = spawn_physics_thread(WorldConfig::default());
///
/// // Add an object
/// let ball_id = physics.add_object(my_object)?;
///
/// // In your render loop:
/// loop {
///     let state = physics.get_latest_state();
///     // Use state.objects to update your game objects
/// }
///
/// // When done:
/// physics.shutdown()?;
/// ```
pub fn spawn_physics_thread(config: WorldConfig) -> PhysicsHandle {
    let (cmd_tx, cmd_rx) = unbounded::<PhysicsCommand>();
    let (state_tx, state_rx) = bounded::<WorldState>(STATE_CHANNEL_CAPACITY);

    let broadcast_rate = config.broadcast_rate.max(1);
    // Readers blend across the gap between published snapshots, which is the
    // tick interval scaled by how often we actually broadcast - not the tick
    // interval itself.
    let broadcast_interval = Duration::from_secs_f64(config.timestep * broadcast_rate as f64);
    let latest_state = Arc::new(RwLock::new(StateBuffer::new(broadcast_interval)));

    let latest_state_clone = latest_state.clone();

    std::thread::Builder::new()
        .name("rs_physics".to_string())
        .spawn(move || {
            physics_thread_main(config, cmd_rx, state_tx, latest_state_clone, broadcast_rate);
        })
        .expect("Failed to spawn physics thread");

    PhysicsHandle::new(cmd_tx, state_rx, latest_state)
}

/// Main physics thread loop
fn physics_thread_main(
    config: WorldConfig,
    cmd_rx: Receiver<PhysicsCommand>,
    state_tx: Sender<WorldState>,
    latest_state: Arc<RwLock<StateBuffer>>,
    broadcast_rate: usize,
) {
    let mut world = PhysicsWorld::new(config.clone());
    let timestep = config.timestep;
    let real_time = config.real_time;

    // Held for the life of the thread; restores the OS setting on the way out.
    let _timer_guard = TimerResolutionGuard::acquire();

    let mut pacer = Pacer::new(
        Duration::from_secs_f64(timestep),
        config.max_catchup_ticks,
    );

    let mut tick_counter = 0usize;
    let mut running = true;

    if real_time {
        log::info!("Physics thread started at {:.1} Hz (timestep {:.4}s), paced to wall clock",
                   1.0 / timestep, timestep);
    } else {
        log::info!("Physics thread started at {:.1} Hz (timestep {:.4}s), running unpaced",
                   1.0 / timestep, timestep);
    }

    while running {
        // Process all pending commands
        loop {
            match cmd_rx.try_recv() {
                Ok(cmd) => {
                    running = process_command(&mut world, cmd);
                    if !running {
                        break;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    log::info!("Physics thread: command channel disconnected, shutting down");
                    running = false;
                    break;
                }
            }
        }

        if !running {
            break;
        }

        // Step the simulation
        world.step();
        tick_counter += 1;

        if tick_counter % broadcast_rate == 0 {
            let state = world.get_state();

            // Optional per-tick feed. Bounded, so a subscriber that stops
            // draining loses updates instead of growing the queue forever.
            let _ = state_tx.try_send(state.clone());

            // Publish for readers. A panicking reader poisons the lock, but the
            // simulation is still valid and other readers still want it, so
            // recover the guard rather than silently stopping all updates.
            let mut buffer = latest_state
                .write()
                .unwrap_or_else(|e| e.into_inner());
            buffer.publish(state);
        }

        // Pace against wall clock. Skipped entirely when unpaced, where the
        // point is to finish the simulation as fast as the machine allows.
        if real_time {
            let dropped = pacer.wait_for_next_tick();
            if dropped > 0 {
                log::warn!(
                    "Physics thread fell {} ticks behind wall clock (>{} allowed); \
                     abandoning the backlog and resynchronizing",
                    dropped,
                    config.max_catchup_ticks,
                );
            }
        }
    }

    log::info!("Physics thread shutting down after {} ticks", world.current_tick());
}

/// Process a single command, returns false if shutdown requested
fn process_command(world: &mut PhysicsWorld, cmd: PhysicsCommand) -> bool {
    match cmd {
        PhysicsCommand::AddObject { object, response } => {
            let id = world.add_object(object);
            let _ = response.send(id);
        }
        PhysicsCommand::RemoveObject(id) => {
            world.remove_object(id);
        }
        PhysicsCommand::SetTimestep(dt) => {
            world.set_timestep(dt);
        }
        PhysicsCommand::ApplyForce(id, force) => {
            world.apply_force(id, force);
        }
        PhysicsCommand::ApplyImpulse(id, impulse) => {
            world.apply_impulse(id, impulse);
        }
        PhysicsCommand::SetVelocity(id, velocity) => {
            world.set_velocity(id, velocity);
        }
        PhysicsCommand::SetPosition(id, position) => {
            world.set_position(id, position);
        }
        // Ergonomic force commands
        PhysicsCommand::ApplyForceDirected(id, magnitude, direction) => {
            world.apply_force_directed(id, magnitude, direction);
        }
        PhysicsCommand::ApplyForceToward(id, target, magnitude) => {
            world.apply_force_toward(id, target, magnitude);
        }
        PhysicsCommand::ApplyForceAway(id, source, magnitude) => {
            world.apply_force_away(id, source, magnitude);
        }
        PhysicsCommand::ApplyDrag(id, coefficient) => {
            world.apply_drag(id, coefficient);
        }
        PhysicsCommand::ApplySpringForce(id, rest_position, stiffness) => {
            world.apply_spring_force(id, rest_position, stiffness);
        }
        PhysicsCommand::ApplyDampedSpring(id, rest_position, stiffness, damping) => {
            world.apply_damped_spring(id, rest_position, stiffness, damping);
        }
        PhysicsCommand::ApplyExplosion(center, force, radius) => {
            world.apply_explosion(center, force, radius);
        }
        PhysicsCommand::ApplyTorque(id, torque) => {
            world.apply_torque(id, torque);
        }
        PhysicsCommand::ApplyBuoyancy(id, surface_y, fluid_density) => {
            world.apply_buoyancy(id, surface_y, fluid_density);
        }
        // Continuous force commands
        PhysicsCommand::AddContinuousForce { force, response } => {
            let id = world.add_continuous_force(force);
            let _ = response.send(id);
        }
        PhysicsCommand::RemoveContinuousForce(id) => {
            world.remove_continuous_force(id);
        }
        PhysicsCommand::RemoveForcesOnObject(target) => {
            world.remove_forces_on_object(target);
        }
        PhysicsCommand::AddDrag { target, coefficient, response } => {
            let id = world.add_drag(target, coefficient);
            let _ = response.send(id);
        }
        PhysicsCommand::AddSpring { target, rest_position, stiffness, response } => {
            let id = world.add_spring(target, rest_position, stiffness);
            let _ = response.send(id);
        }
        PhysicsCommand::AddDampedSpring { target, rest_position, stiffness, damping, response } => {
            let id = world.add_damped_spring(target, rest_position, stiffness, damping);
            let _ = response.send(id);
        }
        PhysicsCommand::AddAttraction { target, point, strength, duration, response } => {
            let id = world.add_attraction(target, point, strength, duration);
            let _ = response.send(id);
        }
        PhysicsCommand::AddBuoyancy { target, surface_y, fluid_density, response } => {
            let id = world.add_buoyancy(target, surface_y, fluid_density);
            let _ = response.send(id);
        }
        PhysicsCommand::AddConstantForce { target, force, response } => {
            let id = world.add_constant_force(target, force);
            let _ = response.send(id);
        }
        PhysicsCommand::AddRepulsion { target, point, strength, duration, response } => {
            let id = world.add_repulsion(target, point, strength, duration);
            let _ = response.send(id);
        }
        PhysicsCommand::AddVortex { target, center, axis, strength, duration, response } => {
            let id = world.add_vortex(target, center, axis, strength, duration);
            let _ = response.send(id);
        }
        // Kinematic object commands
        PhysicsCommand::SetPositionKinematic(id, position, dt) => {
            world.set_position_kinematic(id, position, dt);
        }
        // Constraint commands (requires "constraints" feature)
        #[cfg(feature = "constraints")]
        PhysicsCommand::AddConstraint { constraint, response } => {
            let id = world.add_constraint(constraint);
            let _ = response.send(id);
        }
        #[cfg(feature = "constraints")]
        PhysicsCommand::RemoveConstraint(id) => {
            world.remove_constraint(id);
        }
        // Simulation control
        PhysicsCommand::Pause => {
            world.pause();
            log::debug!("Physics simulation paused");
        }
        PhysicsCommand::Resume => {
            world.resume();
            log::debug!("Physics simulation resumed");
        }
        PhysicsCommand::Shutdown => {
            log::info!("Physics thread received shutdown command");
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{PhysicalObject3D, Shape3D};
    use crate::utils::PhysicsConstants;
    use std::time::Duration;

    fn create_test_sphere(pos: (f64, f64, f64), vel: (f64, f64, f64)) -> PhysicalObject3D {
        PhysicalObject3D::new(
            1.0,  // mass
            vel,
            pos,
            Shape3D::Sphere(0.5),
            None,  // material
            (0.0, 0.0, 0.0),  // angular_velocity
            (0.0, 0.0, 0.0),  // orientation
            PhysicsConstants::default(),
        )
    }

    #[test]
    fn test_spawn_and_shutdown() {
        let physics = spawn_physics_thread(WorldConfig::default());

        // Give it a moment to start
        std::thread::sleep(Duration::from_millis(50));

        // Should be able to get state
        let state = physics.get_latest_state();
        assert_eq!(state.objects.len(), 0);

        // Shutdown
        assert!(physics.shutdown().is_ok());

        // Give it a moment to shut down
        std::thread::sleep(Duration::from_millis(50));
    }

    #[test]
    fn test_add_object() {
        let physics = spawn_physics_thread(WorldConfig::default());

        // Add an object
        let sphere = create_test_sphere((0.0, 10.0, 0.0), (0.0, 0.0, 0.0));
        let id = physics.add_object(sphere).expect("Failed to add object");

        // Wait for physics to process
        std::thread::sleep(Duration::from_millis(100));

        // Check state
        let state = physics.get_latest_state();
        assert_eq!(state.objects.len(), 1);

        let obj_state = state.get_object(id);
        assert!(obj_state.is_some());

        physics.shutdown().ok();
    }

    #[test]
    fn test_gravity_simulation() {
        let config = WorldConfig::default()
            .with_gravity(0.0, -10.0, 0.0)
            .with_frequency(120.0);

        let physics = spawn_physics_thread(config);

        // Add a sphere at height 10
        let sphere = create_test_sphere((0.0, 10.0, 0.0), (0.0, 0.0, 0.0));
        let id = physics.add_object(sphere).expect("Failed to add object");

        // Let it fall for ~1 second (120 ticks at 120Hz)
        std::thread::sleep(Duration::from_secs(1));

        // Check that it fell
        let state = physics.get_latest_state();
        let obj = state.get_object(id).expect("Object should exist");

        assert!(obj.position.1 < 10.0, "Object should have fallen from y=10");
        assert!(obj.velocity.1 < 0.0, "Object should have downward velocity");

        physics.shutdown().ok();
    }

    #[test]
    fn test_pause_resume() {
        let config = WorldConfig::default()
            .with_gravity(0.0, -10.0, 0.0);

        let physics = spawn_physics_thread(config);

        let sphere = create_test_sphere((0.0, 100.0, 0.0), (0.0, 0.0, 0.0));
        let id = physics.add_object(sphere).expect("Failed to add object");

        // Let it run briefly
        std::thread::sleep(Duration::from_millis(50));

        // Pause
        physics.pause().ok();
        std::thread::sleep(Duration::from_millis(50));

        // Record position
        let paused_pos = physics.get_position(id).expect("Should have position");

        // Wait while paused
        std::thread::sleep(Duration::from_millis(200));

        // Position should be (nearly) the same
        let still_paused_pos = physics.get_position(id).expect("Should have position");
        assert!((paused_pos.1 - still_paused_pos.1).abs() < 0.01,
                "Position should not change while paused");

        // Resume and wait
        physics.resume().ok();
        std::thread::sleep(Duration::from_millis(200));

        // Position should have changed
        let resumed_pos = physics.get_position(id).expect("Should have position");
        assert!(resumed_pos.1 < paused_pos.1, "Object should fall after resume");

        physics.shutdown().ok();
    }

    // ==================== Continuous Force Tests (Threaded) ====================

    #[test]
    fn test_threaded_add_drag() {
        let config = WorldConfig::zero_gravity().with_frequency(120.0);
        let physics = spawn_physics_thread(config);

        // Add object with initial velocity
        let sphere = create_test_sphere((0.0, 0.0, 0.0), (10.0, 0.0, 0.0));
        let id = physics.add_object(sphere).expect("Failed to add object");

        // Add continuous drag
        let drag_id = physics.add_drag(id, 0.5).expect("Failed to add drag");

        // Let simulation run
        std::thread::sleep(Duration::from_millis(500));

        // Check velocity decreased
        let vel = physics.get_velocity(id).expect("Should have velocity");
        assert!(vel.0 < 10.0, "Drag should slow object. Got vx={}", vel.0);

        // Remove drag
        physics.remove_continuous_force(drag_id).ok();

        physics.shutdown().ok();
    }

    #[test]
    fn test_threaded_add_spring() {
        let config = WorldConfig::zero_gravity().with_frequency(120.0);
        let physics = spawn_physics_thread(config);

        // Add object displaced from origin
        let sphere = create_test_sphere((5.0, 0.0, 0.0), (0.0, 0.0, 0.0));
        let id = physics.add_object(sphere).expect("Failed to add object");

        // Add damped spring toward origin
        let spring_id = physics.add_damped_spring(id, (0.0, 0.0, 0.0), 10.0, 2.0)
            .expect("Failed to add spring");

        // Let simulation run
        std::thread::sleep(Duration::from_millis(500));

        // Check object moved toward rest position
        let pos = physics.get_position(id).expect("Should have position");
        assert!(pos.0 < 5.0, "Spring should pull object toward rest. Got x={}", pos.0);

        physics.remove_continuous_force(spring_id).ok();
        physics.shutdown().ok();
    }

    #[test]
    fn test_threaded_add_attraction() {
        let config = WorldConfig::zero_gravity().with_frequency(120.0);
        let physics = spawn_physics_thread(config);

        // Add object away from attractor point
        let sphere = create_test_sphere((10.0, 0.0, 0.0), (0.0, 0.0, 0.0));
        let id = physics.add_object(sphere).expect("Failed to add object");

        // Add attraction toward origin
        let _force_id = physics.add_attraction(id, (0.0, 0.0, 0.0), 100.0, None)
            .expect("Failed to add attraction");

        // Let simulation run
        std::thread::sleep(Duration::from_millis(300));

        // Check object moving toward attractor
        let vel = physics.get_velocity(id).expect("Should have velocity");
        assert!(vel.0 < 0.0, "Attraction should pull object toward point. Got vx={}", vel.0);

        physics.shutdown().ok();
    }

    #[test]
    fn test_threaded_add_repulsion() {
        let config = WorldConfig::zero_gravity().with_frequency(120.0);
        let physics = spawn_physics_thread(config);

        // Add object near repulsion point
        let sphere = create_test_sphere((5.0, 0.0, 0.0), (0.0, 0.0, 0.0));
        let id = physics.add_object(sphere).expect("Failed to add object");

        // Add repulsion from origin
        let _force_id = physics.add_repulsion(id, (0.0, 0.0, 0.0), 100.0, None)
            .expect("Failed to add repulsion");

        // Let simulation run
        std::thread::sleep(Duration::from_millis(300));

        // Check object moving away from repulsion point
        let vel = physics.get_velocity(id).expect("Should have velocity");
        assert!(vel.0 > 0.0, "Repulsion should push object away. Got vx={}", vel.0);

        let pos = physics.get_position(id).expect("Should have position");
        assert!(pos.0 > 5.0, "Object should have moved away. Got x={}", pos.0);

        physics.shutdown().ok();
    }

    #[test]
    fn test_threaded_add_vortex() {
        let config = WorldConfig::zero_gravity().with_frequency(120.0);
        let physics = spawn_physics_thread(config);

        // Add object at (5, 0, 0)
        let sphere = create_test_sphere((5.0, 0.0, 0.0), (0.0, 0.0, 0.0));
        let id = physics.add_object(sphere).expect("Failed to add object");

        // Add vortex around Y axis centered at origin
        let _force_id = physics.add_vortex(id, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 100.0, None)
            .expect("Failed to add vortex");

        // Let simulation run
        std::thread::sleep(Duration::from_millis(300));

        // Check object has tangential velocity (should be in -Z direction)
        let vel = physics.get_velocity(id).expect("Should have velocity");
        assert!(vel.2 < 0.0, "Vortex should create tangential velocity. Got vz={}", vel.2);

        physics.shutdown().ok();
    }

    #[test]
    fn test_threaded_add_buoyancy() {
        // NOTE: Buoyancy force is calculated as fluid_density * depth * gravity
        // So we need gravity to be non-zero for buoyancy to work
        let config = WorldConfig::default().with_frequency(120.0);
        let physics = spawn_physics_thread(config);

        // Add object below water surface
        let sphere = create_test_sphere((0.0, -5.0, 0.0), (0.0, 0.0, 0.0));
        let id = physics.add_object(sphere).expect("Failed to add object");

        // Add buoyancy (surface at y=0) with high fluid density to overcome gravity
        let _force_id = physics.add_buoyancy(id, 0.0, 2000.0)
            .expect("Failed to add buoyancy");

        // Let simulation run (longer time to ensure physics processes)
        std::thread::sleep(Duration::from_millis(800));

        // Check object has upward velocity (buoyancy - gravity should be positive)
        let vel = physics.get_velocity(id).expect("Should have velocity");
        assert!(vel.1 > 0.0, "Buoyancy should push object up. Got vy={}", vel.1);

        let pos = physics.get_position(id).expect("Should have position");
        assert!(pos.1 > -5.0, "Object should have risen. Got y={}", pos.1);

        physics.shutdown().ok();
    }

    #[test]
    fn test_threaded_add_constant_force() {
        let config = WorldConfig::zero_gravity().with_frequency(120.0);
        let physics = spawn_physics_thread(config);

        // Add stationary object
        let sphere = create_test_sphere((0.0, 0.0, 0.0), (0.0, 0.0, 0.0));
        let id = physics.add_object(sphere).expect("Failed to add object");

        // Add constant force (like wind)
        let force_id = physics.add_constant_force(id, (10.0, 0.0, 0.0))
            .expect("Failed to add constant force");

        // Let simulation run
        std::thread::sleep(Duration::from_millis(300));

        // Check object accelerating
        let vel = physics.get_velocity(id).expect("Should have velocity");
        assert!(vel.0 > 0.0, "Constant force should accelerate object. Got vx={}", vel.0);

        // Remove force
        physics.remove_continuous_force(force_id).ok();

        physics.shutdown().ok();
    }

    #[test]
    fn test_threaded_remove_forces_on_object() {
        let config = WorldConfig::zero_gravity().with_frequency(120.0);
        let physics = spawn_physics_thread(config);

        // Add object
        let sphere = create_test_sphere((0.0, 0.0, 0.0), (10.0, 0.0, 0.0));
        let id = physics.add_object(sphere).expect("Failed to add object");

        // Add multiple forces
        physics.add_drag(id, 0.5).expect("Failed to add drag");
        physics.add_constant_force(id, (1.0, 0.0, 0.0)).expect("Failed to add constant force");

        // Let simulation run briefly
        std::thread::sleep(Duration::from_millis(100));

        // Remove all forces on object
        physics.remove_forces_on_object(id).ok();

        // Record velocity after force removal
        std::thread::sleep(Duration::from_millis(50));
        let vel1 = physics.get_velocity(id).expect("Should have velocity");

        // Wait more - velocity should decrease due to damping (no external forces)
        std::thread::sleep(Duration::from_millis(200));
        let vel2 = physics.get_velocity(id).expect("Should have velocity");

        // With damping enabled, velocity should decrease over time (not increase)
        // The damping causes exponential decay, so vel2 should be less than vel1
        assert!(vel2.0 <= vel1.0, "Velocity should decrease or stay same due to damping. vel1={}, vel2={}", vel1.0, vel2.0);

        physics.shutdown().ok();
    }

    #[test]
    fn test_threaded_multiple_objects_multiple_forces() {
        let config = WorldConfig::zero_gravity().with_frequency(120.0);
        let physics = spawn_physics_thread(config);

        // Add two objects
        let sphere1 = create_test_sphere((0.0, 0.0, 0.0), (10.0, 0.0, 0.0));
        let sphere2 = create_test_sphere((10.0, 0.0, 0.0), (0.0, 0.0, 0.0));
        let id1 = physics.add_object(sphere1).expect("Failed to add object 1");
        let id2 = physics.add_object(sphere2).expect("Failed to add object 2");

        // Add different forces to each
        physics.add_drag(id1, 0.5).expect("Failed to add drag to object 1");
        physics.add_attraction(id2, (0.0, 0.0, 0.0), 50.0, None)
            .expect("Failed to add attraction to object 2");

        // Let simulation run
        std::thread::sleep(Duration::from_millis(500));

        // Check object 1 slowed down (drag)
        let vel1 = physics.get_velocity(id1).expect("Should have velocity");
        assert!(vel1.0 < 10.0, "Drag should slow object 1. Got vx={}", vel1.0);

        // Check object 2 moving toward origin (attraction)
        let vel2 = physics.get_velocity(id2).expect("Should have velocity");
        assert!(vel2.0 < 0.0, "Attraction should pull object 2 toward origin. Got vx={}", vel2.0);

        physics.shutdown().ok();
    }

    /// The simulation must advance at the configured rate in real time.
    ///
    /// This is the regression guard for the sleep-granularity bug: naive
    /// `sleep(period - elapsed)` on Windows quantizes to the ~15.6 ms system
    /// timer, so a 240 Hz world silently ran at roughly 64 Hz.
    #[test]
    fn test_paced_tick_rate_tracks_wall_clock() {
        const HZ: f64 = 240.0;
        const RUN: Duration = Duration::from_millis(1000);

        let physics = spawn_physics_thread(WorldConfig::default().with_frequency(HZ));

        // Let the pacer's granularity estimate settle before measuring.
        std::thread::sleep(Duration::from_millis(200));
        let start_tick = physics.get_latest_state().tick;

        let started = Instant::now();
        std::thread::sleep(RUN);
        let measured = started.elapsed();

        let ticks = physics.get_latest_state().tick - start_tick;
        physics.shutdown().ok();

        let expected = HZ * measured.as_secs_f64();
        let ratio = ticks as f64 / expected;
        eprintln!(
            "paced rate: {ticks} ticks in {:.3}s = {:.1} Hz (target {HZ} Hz, ratio {ratio:.3})",
            measured.as_secs_f64(),
            ticks as f64 / measured.as_secs_f64(),
        );

        // An empty world is cheap to step, so the only thing under test is
        // timing. Generous bounds: CI machines are noisy, but the old behaviour
        // sat near 0.27 and cannot pass this.
        assert!(
            (0.85..=1.15).contains(&ratio),
            "expected ~{expected:.0} ticks in {:.3}s at {HZ} Hz, got {ticks} (ratio {ratio:.3})",
            measured.as_secs_f64(),
        );
    }

    /// Unpaced mode exists to outrun wall clock; verify it actually does.
    #[test]
    fn test_unpaced_mode_outruns_wall_clock() {
        const HZ: f64 = 240.0;

        let physics = spawn_physics_thread(
            WorldConfig::default().with_frequency(HZ).with_real_time(false),
        );

        let started = Instant::now();
        std::thread::sleep(Duration::from_millis(200));
        let elapsed = started.elapsed();
        let ticks = physics.get_latest_state().tick;
        physics.shutdown().ok();

        let realtime_ticks = HZ * elapsed.as_secs_f64();
        assert!(
            ticks as f64 > realtime_ticks * 2.0,
            "unpaced mode should far outrun {realtime_ticks:.0} real-time ticks, got {ticks}",
        );
    }

    /// The state channel must not grow without bound when nobody drains it.
    ///
    /// This is the leak: the physics thread broadcasts every tick, and a
    /// renderer reading via `get_interpolated_state()` never touches the
    /// channel. Unbounded, that queued a snapshot per tick forever.
    #[test]
    fn test_undrained_state_channel_is_bounded() {
        let physics = spawn_physics_thread(WorldConfig::default().with_frequency(240.0));
        physics.add_object(create_test_sphere((0.0, 10.0, 0.0), (0.0, 0.0, 0.0)))
            .expect("Failed to add object");

        // Far more ticks than the channel can hold, with no consumer.
        std::thread::sleep(Duration::from_millis(500));

        let mut drained = 0usize;
        while physics.try_recv_state().is_some() {
            drained += 1;
            assert!(
                drained <= STATE_CHANNEL_CAPACITY,
                "channel exceeded its {STATE_CHANNEL_CAPACITY}-slot bound",
            );
        }
        physics.shutdown().ok();

        assert!(drained > 0, "expected the channel to hold some buffered states");
    }

    /// Interpolated reads must stay within the bracket of real simulated states.
    #[test]
    fn test_interpolated_state_is_bracketed_by_simulation() {
        let physics = spawn_physics_thread(
            WorldConfig::default().with_frequency(120.0).with_gravity(0.0, -10.0, 0.0),
        );
        let id = physics
            .add_object(create_test_sphere((0.0, 100.0, 0.0), (0.0, 0.0, 0.0)))
            .expect("Failed to add object");

        std::thread::sleep(Duration::from_millis(300));

        // Sampled repeatedly across tick boundaries, the blended height must
        // decrease monotonically - never jump ahead of, or behind, the sim.
        let mut last = f64::INFINITY;
        for _ in 0..40 {
            let y = physics
                .get_interpolated_state()
                .get_position(id)
                .expect("object should exist")
                .1;
            assert!(y.is_finite(), "interpolated position must stay finite");
            assert!(y <= last + 1e-9, "falling object rose: {last} -> {y}");
            last = y;
            std::thread::sleep(Duration::from_millis(3));
        }

        physics.shutdown().ok();
    }
}
