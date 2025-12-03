//! Physics thread spawning and main loop

use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use crossbeam::channel::{unbounded, Receiver, Sender, TryRecvError};

use super::config::WorldConfig;
use super::handle::{PhysicsCommand, PhysicsHandle};
use super::physics_world::PhysicsWorld;
use super::state::WorldState;

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
    let (state_tx, state_rx) = unbounded::<WorldState>();
    let latest_state = Arc::new(RwLock::new(WorldState::default()));

    let latest_state_clone = latest_state.clone();
    let broadcast_rate = config.broadcast_rate;

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
    latest_state: Arc<RwLock<WorldState>>,
    broadcast_rate: usize,
) {
    let mut world = PhysicsWorld::new(config.clone());
    let timestep = config.timestep;
    let target_frame_time = Duration::from_secs_f64(timestep);

    let mut tick_counter = 0usize;
    let mut running = true;

    log::info!("Physics thread started with timestep: {:.4}s ({:.1} Hz)",
               timestep, 1.0 / timestep);

    while running {
        let frame_start = Instant::now();

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

        // Broadcast state at configured rate
        if tick_counter % broadcast_rate == 0 {
            let state = world.get_state();

            // Update RwLock for thread-safe reads
            if let Ok(mut latest) = latest_state.write() {
                *latest = state.clone();
            }

            // Also send through channel for subscribers who want every update
            // Use try_send to avoid blocking if no one is listening
            let _ = state_tx.try_send(state);
        }

        // Sleep to maintain target framerate
        let elapsed = frame_start.elapsed();
        if elapsed < target_frame_time {
            std::thread::sleep(target_frame_time - elapsed);
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

        // Wait more - velocity should stay constant (no forces)
        std::thread::sleep(Duration::from_millis(200));
        let vel2 = physics.get_velocity(id).expect("Should have velocity");

        // Velocity should be approximately the same (no acceleration)
        let vel_diff = (vel2.0 - vel1.0).abs();
        assert!(vel_diff < 0.5, "Velocity should be constant after force removal. Diff={}", vel_diff);

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
}
