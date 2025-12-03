//! 3D Constraint visualization
//!
//! This visualizes various physics constraints:
//! - Pendulum chain (joints)
//! - Soft body grid (springs)
//! - Rope constraint
//! - Hinge door
//!
//! Controls:
//! - A/D or Left/Right: Rotate camera horizontally
//! - W/S or Up/Down: Rotate camera vertically
//! - Q/E: Zoom in/out
//! - Space: Pause/resume simulation
//! - R: Reset simulation
//! - 1-4: Switch scenarios
//! - G: Toggle gravity
//! - Escape: Exit

use bevy::prelude::*;
use bevy_visual_tests::{
    spawn_orbit_camera, spawn_info_text, particle_material,
    OrbitCameraPlugin, SimulationInfoText,
};
use rs_physics::models::ObjectIn3D;
use rs_physics::constraints::{Joint3D, Spring3D, Rope3D, Hinge3D};

const DT: f32 = 0.016; // 60 FPS timestep
const SOLVER_ITERATIONS: usize = 20; // Increased for better convergence
const MAX_VELOCITY: f64 = 20.0; // Clamp velocities for stability

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "3D Constraints Visualization".to_string(),
                resolution: (1280.0, 720.0).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(OrbitCameraPlugin)
        .init_resource::<SimulationState>()
        .add_systems(Startup, setup)
        .add_systems(Update, (
            simulation_step,
            sync_visuals,
            draw_constraint_lines,
            update_info_text,
            handle_input,
        ))
        .run();
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Scenario {
    Pendulum,
    SoftBody,
    Rope,
    Hinge,
}

impl Scenario {
    fn name(&self) -> &'static str {
        match self {
            Scenario::Pendulum => "Pendulum Chain",
            Scenario::SoftBody => "Soft Body Grid",
            Scenario::Rope => "Rope Swing",
            Scenario::Hinge => "Hinge Door",
        }
    }
}

/// Spring connection for soft body - stores indices into object array
#[derive(Clone)]
struct SpringConnection {
    idx1: usize,
    idx2: usize,
    rest_length: f64,
    stiffness: f64,
    damping: f64,
}

#[derive(Resource)]
struct SimulationState {
    scenario: Scenario,
    paused: bool,
    step_count: u64,
    gravity_enabled: bool,
    // Physics objects
    objects: Vec<ObjectIn3D>,
    // Constraint types stored separately for type-specific handling
    joints: Vec<Joint3D>,
    springs: Vec<Spring3D>,
    ropes: Vec<Rope3D>,
    hinges: Vec<Hinge3D>,
    // Track which objects are anchored (infinite mass)
    anchored: Vec<bool>,
    // Spring connections for soft body (stores indices into objects array)
    spring_connections: Vec<SpringConnection>,
}

impl Default for SimulationState {
    fn default() -> Self {
        Self {
            scenario: Scenario::Pendulum,
            paused: false,
            step_count: 0,
            gravity_enabled: true,
            objects: Vec::new(),
            joints: Vec::new(),
            springs: Vec::new(),
            ropes: Vec::new(),
            hinges: Vec::new(),
            anchored: Vec::new(),
            spring_connections: Vec::new(),
        }
    }
}

#[derive(Component)]
struct PhysicsObject;

#[derive(Resource)]
struct VisualizationAssets {
    sphere_mesh: Handle<Mesh>,
    cube_mesh: Handle<Mesh>,
    anchor_material: Handle<StandardMaterial>,
    object_material: Handle<StandardMaterial>,
    spring_material: Handle<StandardMaterial>,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut sim_state: ResMut<SimulationState>,
) {
    // Create visualization assets
    let sphere_mesh = meshes.add(Sphere::new(0.3));
    let cube_mesh = meshes.add(Cuboid::new(0.5, 0.5, 0.5));

    let anchor_material = particle_material(&mut materials, Color::srgb(0.8, 0.2, 0.2));
    let object_material = particle_material(&mut materials, Color::srgb(0.2, 0.6, 0.9));
    let spring_material = particle_material(&mut materials, Color::srgb(0.2, 0.9, 0.3));

    commands.insert_resource(VisualizationAssets {
        sphere_mesh,
        cube_mesh,
        anchor_material,
        object_material,
        spring_material,
    });

    // Initialize first scenario
    setup_pendulum_scenario(&mut sim_state);

    // Spawn camera
    spawn_orbit_camera(&mut commands, Vec3::new(0.0, 5.0, 20.0), Vec3::new(0.0, 0.0, 0.0));

    // Spawn lighting
    commands.spawn((
        PointLight {
            intensity: 2_000_000.0,
            range: 100.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(10.0, 20.0, 10.0),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(5.0, 10.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 300.0,
    });

    // Ground plane
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(50.0)))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.3, 0.35),
            perceptual_roughness: 0.9,
            ..default()
        })),
        Transform::from_xyz(0.0, -5.0, 0.0),
    ));

    // UI text
    spawn_info_text(&mut commands);
}

fn setup_pendulum_scenario(sim_state: &mut SimulationState) {
    sim_state.objects.clear();
    sim_state.joints.clear();
    sim_state.springs.clear();
    sim_state.ropes.clear();
    sim_state.hinges.clear();
    sim_state.anchored.clear();
    sim_state.scenario = Scenario::Pendulum;

    // Create a simple pendulum using stiff springs (more stable than joints)
    // Single pendulum bob hanging from anchor
    let anchor = ObjectIn3D::new(f64::INFINITY, 0.0, 0.0, 0.0, (0.0, 6.0, 0.0));
    let bob = ObjectIn3D::new(2.0, 0.0, 0.0, 0.0, (3.0, 3.0, 0.0)); // Displaced to the right

    sim_state.objects.push(anchor.clone());
    sim_state.objects.push(bob.clone());
    sim_state.anchored.push(true);
    sim_state.anchored.push(false);

    // Use a very stiff spring to simulate a rigid rod
    // High stiffness + critical damping for stability
    let rest_length = 4.24; // sqrt(3^2 + 3^2) ≈ distance from anchor to bob
    let spring = Spring3D::new(
        anchor,
        bob,
        500.0,  // Very high stiffness
        rest_length,
        10.0,   // Damping
    ).unwrap();
    sim_state.springs.push(spring);
}

fn setup_softbody_scenario(sim_state: &mut SimulationState) {
    sim_state.objects.clear();
    sim_state.joints.clear();
    sim_state.springs.clear();
    sim_state.ropes.clear();
    sim_state.hinges.clear();
    sim_state.anchored.clear();
    sim_state.spring_connections.clear();
    sim_state.scenario = Scenario::SoftBody;

    // Create a 4x4 grid of particles connected by springs
    let grid_size = 4;
    let spacing = 1.2;
    let start_x = -((grid_size - 1) as f64) * spacing / 2.0;
    let start_y = 6.0;
    let start_z = -((grid_size - 1) as f64) * spacing / 2.0;

    // Create particles
    for z in 0..grid_size {
        for x in 0..grid_size {
            let is_anchor = z == 0 && (x == 0 || x == grid_size - 1);
            let mass = if is_anchor { f64::INFINITY } else { 0.5 };
            let px = start_x + (x as f64) * spacing;
            let py = start_y;
            let pz = start_z + (z as f64) * spacing;

            let obj = ObjectIn3D::new(mass, 0.0, 0.0, 0.0, (px, py, pz));
            sim_state.objects.push(obj);
            sim_state.anchored.push(is_anchor);
        }
    }

    // Create spring connections - horizontal
    for z in 0..grid_size {
        for x in 0..(grid_size - 1) {
            let idx1 = z * grid_size + x;
            let idx2 = z * grid_size + x + 1;
            sim_state.spring_connections.push(SpringConnection {
                idx1,
                idx2,
                rest_length: spacing,
                stiffness: 200.0,
                damping: 5.0,
            });
        }
    }

    // Create spring connections - vertical (in Z direction)
    for z in 0..(grid_size - 1) {
        for x in 0..grid_size {
            let idx1 = z * grid_size + x;
            let idx2 = (z + 1) * grid_size + x;
            sim_state.spring_connections.push(SpringConnection {
                idx1,
                idx2,
                rest_length: spacing,
                stiffness: 200.0,
                damping: 5.0,
            });
        }
    }

    // Diagonal springs for shear stiffness
    for z in 0..(grid_size - 1) {
        for x in 0..(grid_size - 1) {
            let idx1 = z * grid_size + x;
            let idx2 = (z + 1) * grid_size + x + 1;
            let diag_length = (spacing * spacing * 2.0).sqrt();
            sim_state.spring_connections.push(SpringConnection {
                idx1,
                idx2,
                rest_length: diag_length,
                stiffness: 100.0,
                damping: 3.0,
            });
        }
    }
}

fn setup_rope_scenario(sim_state: &mut SimulationState) {
    sim_state.objects.clear();
    sim_state.joints.clear();
    sim_state.springs.clear();
    sim_state.ropes.clear();
    sim_state.hinges.clear();
    sim_state.anchored.clear();
    sim_state.scenario = Scenario::Rope;

    // Create a single anchor and a weight hanging from it
    // This is simpler and more stable than dual ropes
    let anchor = ObjectIn3D::new(f64::INFINITY, 0.0, 0.0, 0.0, (0.0, 8.0, 0.0));
    // Weight starts displaced to the right, so rope will swing
    let weight = ObjectIn3D::new(2.0, 0.0, 0.0, 0.0, (4.0, 3.0, 0.0));

    sim_state.objects.push(anchor.clone());
    sim_state.objects.push(weight.clone());
    sim_state.anchored.push(true);
    sim_state.anchored.push(false);

    // Distance from (0,8,0) to (4,3,0) = sqrt(16+25) = sqrt(41) ≈ 6.4
    // Set max_length just slightly longer so rope is nearly taut
    let rope = Rope3D::new(anchor, weight, 6.5).unwrap();
    sim_state.ropes.push(rope);
}

fn setup_hinge_scenario(sim_state: &mut SimulationState) {
    sim_state.objects.clear();
    sim_state.joints.clear();
    sim_state.springs.clear();
    sim_state.ropes.clear();
    sim_state.hinges.clear();
    sim_state.anchored.clear();
    sim_state.scenario = Scenario::Hinge;

    // Create a door frame (anchor) and door (swinging)
    let frame = ObjectIn3D::new(f64::INFINITY, 0.0, 0.0, 0.0, (0.0, 3.0, 0.0));
    // Door starts rotated open
    let door = ObjectIn3D::new(5.0, 0.0, 0.0, 0.5, (2.0, 3.0, 0.0));

    sim_state.objects.push(frame.clone());
    sim_state.objects.push(door.clone());
    sim_state.anchored.push(true);
    sim_state.anchored.push(false);

    // Create hinge at the frame position, rotating around Y axis
    let hinge = Hinge3D::new(
        frame,
        door,
        (0.0, 3.0, 0.0),  // anchor point
        (0.0, 1.0, 0.0),  // Y axis rotation
    ).unwrap().with_limits(-1.5, 1.5);  // Limit swing angle

    sim_state.hinges.push(hinge);
}

fn simulation_step(mut sim_state: ResMut<SimulationState>) {
    if sim_state.paused {
        return;
    }

    let dt = DT as f64;
    let gravity = if sim_state.gravity_enabled { -9.81 } else { 0.0 };
    let num_objects = sim_state.objects.len();
    let log_this_frame = sim_state.step_count % 120 == 0; // Log every 2 seconds

    if log_this_frame {
        println!("\n=== Step {} ({:?}) ===", sim_state.step_count, sim_state.scenario);
        println!("Before gravity/integration:");
        for (i, obj) in sim_state.objects.iter().enumerate() {
            println!("  obj[{}]: pos=({:.2}, {:.2}, {:.2}) vel=({:.2}, {:.2}, {:.2}) anchored={}",
                i, obj.position.x, obj.position.y, obj.position.z,
                obj.velocity.x, obj.velocity.y, obj.velocity.z,
                sim_state.anchored[i]);
        }
    }

    // Handle different scenarios appropriately
    match sim_state.scenario {
        Scenario::SoftBody => {
            // Soft body uses direct force calculation on particles
            // Step 1: Apply gravity to all non-anchored particles
            for i in 0..num_objects {
                if !sim_state.anchored[i] {
                    sim_state.objects[i].velocity.y += gravity * dt;
                }
            }

            // Step 2: Calculate and apply spring forces (without internal integration)
            // We accumulate forces from all springs, then integrate once
            let num_springs = sim_state.spring_connections.len();
            for spring_idx in 0..num_springs {
                let conn = &sim_state.spring_connections[spring_idx];
                let idx1 = conn.idx1;
                let idx2 = conn.idx2;
                let rest_length = conn.rest_length;
                let stiffness = conn.stiffness;
                let damping = conn.damping;

                // Get positions and velocities (copy to avoid borrow issues)
                let p1 = sim_state.objects[idx1].position.clone();
                let p2 = sim_state.objects[idx2].position.clone();
                let v1 = sim_state.objects[idx1].velocity.clone();
                let v2 = sim_state.objects[idx2].velocity.clone();
                let m1 = sim_state.objects[idx1].mass;
                let m2 = sim_state.objects[idx2].mass;
                let anchored1 = sim_state.anchored[idx1];
                let anchored2 = sim_state.anchored[idx2];

                let dx = p2.x - p1.x;
                let dy = p2.y - p1.y;
                let dz = p2.z - p1.z;
                let length = (dx * dx + dy * dy + dz * dz).sqrt();

                if length < 1e-10 {
                    continue;
                }

                let nx = dx / length;
                let ny = dy / length;
                let nz = dz / length;

                let stretch = length - rest_length;
                let spring_force = stiffness * stretch;

                let rel_vx = v2.x - v1.x;
                let rel_vy = v2.y - v1.y;
                let rel_vz = v2.z - v1.z;
                let rel_vel_along = rel_vx * nx + rel_vy * ny + rel_vz * nz;
                let damping_force = damping * rel_vel_along;

                let total_force = spring_force + damping_force;

                // Apply forces to velocities (F = ma, so dv = F/m * dt)
                if !m1.is_infinite() && !anchored1 {
                    let accel = total_force / m1;
                    sim_state.objects[idx1].velocity.x += accel * nx * dt;
                    sim_state.objects[idx1].velocity.y += accel * ny * dt;
                    sim_state.objects[idx1].velocity.z += accel * nz * dt;
                }
                if !m2.is_infinite() && !anchored2 {
                    let accel = total_force / m2;
                    sim_state.objects[idx2].velocity.x -= accel * nx * dt;
                    sim_state.objects[idx2].velocity.y -= accel * ny * dt;
                    sim_state.objects[idx2].velocity.z -= accel * nz * dt;
                }
            }

            // Step 3: Integrate positions (single integration after all forces applied)
            for i in 0..num_objects {
                if !sim_state.anchored[i] {
                    let obj = &mut sim_state.objects[i];
                    obj.position.x += obj.velocity.x * dt;
                    obj.position.y += obj.velocity.y * dt;
                    obj.position.z += obj.velocity.z * dt;
                }
            }
        }
        Scenario::Pendulum => {
            // Pendulum uses Spring3D which handles its own integration
            for spring in &mut sim_state.springs {
                if !spring.object1.mass.is_infinite() {
                    spring.object1.velocity.y += gravity * dt;
                }
                if !spring.object2.mass.is_infinite() {
                    spring.object2.velocity.y += gravity * dt;
                }
            }
        }
        _ => {
            // Rope and Hinge use external gravity + position integration
            for i in 0..num_objects {
                if !sim_state.anchored[i] {
                    sim_state.objects[i].velocity.y += gravity * dt;
                }
            }
            for i in 0..num_objects {
                if !sim_state.anchored[i] {
                    let obj = &mut sim_state.objects[i];
                    obj.position.x += obj.velocity.x * dt;
                    obj.position.y += obj.velocity.y * dt;
                    obj.position.z += obj.velocity.z * dt;
                }
            }
        }
    }

    let use_external_gravity = !matches!(sim_state.scenario, Scenario::Pendulum | Scenario::SoftBody);

    if log_this_frame {
        println!("After gravity+integration, before constraints:");
        for (i, obj) in sim_state.objects.iter().enumerate() {
            if !sim_state.anchored[i] {
                println!("  obj[{}]: pos=({:.2}, {:.2}, {:.2}) vel=({:.2}, {:.2}, {:.2})",
                    i, obj.position.x, obj.position.y, obj.position.z,
                    obj.velocity.x, obj.velocity.y, obj.velocity.z);
            }
        }
    }

    // Step 3: Copy object state TO constraints before solving (non-spring scenarios only)
    if use_external_gravity {
        update_constraints_from_objects(&mut sim_state);
    }

    if log_this_frame && !sim_state.joints.is_empty() {
        println!("Constraints before solve:");
        for (i, joint) in sim_state.joints.iter().enumerate() {
            println!("  joint[{}]: obj1=({:.2},{:.2},{:.2}) obj2=({:.2},{:.2},{:.2}) dist={:.2}",
                i, joint.object1.position.x, joint.object1.position.y, joint.object1.position.z,
                joint.object2.position.x, joint.object2.position.y, joint.object2.position.z,
                joint.constraint_distance);
        }
    }

    // Step 4: Solve constraints
    // Springs only need one solve per timestep (they integrate internally)
    for spring in &mut sim_state.springs {
        let _ = spring.solve(dt);
    }

    // Joint/rope/hinge constraints benefit from multiple iterations
    for _ in 0..SOLVER_ITERATIONS {
        for joint in &mut sim_state.joints {
            let _ = joint.solve(dt);
        }
        for rope in &mut sim_state.ropes {
            let _ = rope.solve(dt);
        }
        for hinge in &mut sim_state.hinges {
            let _ = hinge.solve(dt);
        }
    }

    if log_this_frame && !sim_state.joints.is_empty() {
        println!("Constraints after solve:");
        for (i, joint) in sim_state.joints.iter().enumerate() {
            println!("  joint[{}]: obj1=({:.2},{:.2},{:.2}) obj2=({:.2},{:.2},{:.2})",
                i, joint.object1.position.x, joint.object1.position.y, joint.object1.position.z,
                joint.object2.position.x, joint.object2.position.y, joint.object2.position.z);
        }
    }

    // Step 5: Copy corrected state FROM constraints back to objects
    update_objects_from_constraints(&mut sim_state);

    if log_this_frame {
        println!("After constraints copied back:");
        for (i, obj) in sim_state.objects.iter().enumerate() {
            println!("  obj[{}]: pos=({:.2}, {:.2}, {:.2}) vel=({:.2}, {:.2}, {:.2})",
                i, obj.position.x, obj.position.y, obj.position.z,
                obj.velocity.x, obj.velocity.y, obj.velocity.z);
        }
    }

    // Step 6: Floor collision and velocity clamping
    for i in 0..num_objects {
        if !sim_state.anchored[i] {
            let obj = &mut sim_state.objects[i];
            if obj.position.y < -4.5 {
                obj.position.y = -4.5;
                obj.velocity.y = -obj.velocity.y * 0.5;
                obj.velocity.x *= 0.9;
                obj.velocity.z *= 0.9;
            }

            // Clamp velocities for stability
            obj.velocity.x = obj.velocity.x.clamp(-MAX_VELOCITY, MAX_VELOCITY);
            obj.velocity.y = obj.velocity.y.clamp(-MAX_VELOCITY, MAX_VELOCITY);
            obj.velocity.z = obj.velocity.z.clamp(-MAX_VELOCITY, MAX_VELOCITY);
        }
    }

    sim_state.step_count += 1;
}

fn update_objects_from_constraints(sim_state: &mut SimulationState) {
    match sim_state.scenario {
        Scenario::Pendulum => {
            // Simple pendulum: anchor (0) and bob (1) connected by spring
            if let Some(spring) = sim_state.springs.first() {
                if sim_state.objects.len() > 1 && !sim_state.anchored[1] {
                    sim_state.objects[1].position = spring.object2.position.clone();
                    sim_state.objects[1].velocity = spring.object2.velocity.clone();
                }
            }
        }
        Scenario::SoftBody => {
            // Soft body: springs are solved and update their internal objects.
            // We need to average or accumulate corrections from all springs
            // that reference each object. For simplicity, use the spring objects
            // directly since they all update the same positions.
            let grid_size = 4usize;

            // Horizontal springs
            let mut spring_idx = 0;
            for z in 0..grid_size {
                for x in 0..(grid_size - 1) {
                    let idx1 = z * grid_size + x;
                    let idx2 = z * grid_size + x + 1;
                    if spring_idx < sim_state.springs.len() {
                        let spring = &sim_state.springs[spring_idx];
                        if !sim_state.anchored[idx1] {
                            sim_state.objects[idx1].position = spring.object1.position.clone();
                            sim_state.objects[idx1].velocity = spring.object1.velocity.clone();
                        }
                        if !sim_state.anchored[idx2] {
                            sim_state.objects[idx2].position = spring.object2.position.clone();
                            sim_state.objects[idx2].velocity = spring.object2.velocity.clone();
                        }
                        spring_idx += 1;
                    }
                }
            }
            // Note: vertical and diagonal springs will overwrite with their corrections
            // This is a simplification - proper approach would accumulate/average
        }
        Scenario::Rope => {
            // Rope: object 0 is anchor, object 1 is the weight
            if let Some(rope) = sim_state.ropes.first() {
                if sim_state.objects.len() > 1 && !sim_state.anchored[1] {
                    sim_state.objects[1].position = rope.object2.position.clone();
                    sim_state.objects[1].velocity = rope.object2.velocity.clone();
                }
            }
        }
        Scenario::Hinge => {
            // Hinge: object 0 is frame (anchor), object 1 is door
            if let Some(hinge) = sim_state.hinges.first() {
                if sim_state.objects.len() > 1 && !sim_state.anchored[1] {
                    sim_state.objects[1].position = hinge.object2.position.clone();
                    sim_state.objects[1].velocity = hinge.object2.velocity.clone();
                }
            }
        }
    }
}

fn update_constraints_from_objects(sim_state: &mut SimulationState) {
    match sim_state.scenario {
        Scenario::Pendulum => {
            // Simple pendulum: anchor (0) and bob (1) connected by spring
            if let Some(spring) = sim_state.springs.first_mut() {
                if sim_state.objects.len() > 1 {
                    spring.object1 = sim_state.objects[0].clone();
                    spring.object2 = sim_state.objects[1].clone();
                }
            }
        }
        Scenario::SoftBody => {
            let grid_size = 4usize;
            let mut spring_idx = 0;

            // Horizontal springs
            for z in 0..grid_size {
                for x in 0..(grid_size - 1) {
                    let idx1 = z * grid_size + x;
                    let idx2 = z * grid_size + x + 1;
                    if spring_idx < sim_state.springs.len() {
                        sim_state.springs[spring_idx].object1 = sim_state.objects[idx1].clone();
                        sim_state.springs[spring_idx].object2 = sim_state.objects[idx2].clone();
                        spring_idx += 1;
                    }
                }
            }

            // Vertical springs
            for z in 0..(grid_size - 1) {
                for x in 0..grid_size {
                    let idx1 = z * grid_size + x;
                    let idx2 = (z + 1) * grid_size + x;
                    if spring_idx < sim_state.springs.len() {
                        sim_state.springs[spring_idx].object1 = sim_state.objects[idx1].clone();
                        sim_state.springs[spring_idx].object2 = sim_state.objects[idx2].clone();
                        spring_idx += 1;
                    }
                }
            }

            // Diagonal springs
            for z in 0..(grid_size - 1) {
                for x in 0..(grid_size - 1) {
                    let idx1 = z * grid_size + x;
                    let idx2 = (z + 1) * grid_size + x + 1;
                    if spring_idx < sim_state.springs.len() {
                        sim_state.springs[spring_idx].object1 = sim_state.objects[idx1].clone();
                        sim_state.springs[spring_idx].object2 = sim_state.objects[idx2].clone();
                        spring_idx += 1;
                    }
                }
            }
        }
        Scenario::Rope => {
            // Rope scenario: anchor at 0, weight at 1
            if let Some(rope) = sim_state.ropes.first_mut() {
                if sim_state.objects.len() > 1 {
                    rope.object1 = sim_state.objects[0].clone();
                    rope.object2 = sim_state.objects[1].clone();
                }
            }
        }
        Scenario::Hinge => {
            // Hinge scenario: frame at 0, door at 1
            if let Some(hinge) = sim_state.hinges.first_mut() {
                if sim_state.objects.len() > 1 {
                    hinge.object1 = sim_state.objects[0].clone();
                    hinge.object2 = sim_state.objects[1].clone();
                }
            }
        }
    }
}

fn sync_visuals(
    sim_state: Res<SimulationState>,
    assets: Res<VisualizationAssets>,
    mut commands: Commands,
    query: Query<Entity, With<PhysicsObject>>,
) {
    // Remove old physics objects
    for entity in query.iter() {
        commands.entity(entity).despawn();
    }

    // Spawn new physics objects based on current state
    for (i, obj) in sim_state.objects.iter().enumerate() {
        let material = if sim_state.anchored[i] {
            assets.anchor_material.clone()
        } else if sim_state.scenario == Scenario::SoftBody {
            assets.spring_material.clone()
        } else {
            assets.object_material.clone()
        };

        let mesh = if sim_state.scenario == Scenario::Hinge && i == 1 {
            assets.cube_mesh.clone()
        } else {
            assets.sphere_mesh.clone()
        };

        let scale = if sim_state.scenario == Scenario::Hinge && i == 1 {
            Vec3::new(4.0, 6.0, 0.3) // Door shape
        } else if sim_state.anchored[i] {
            Vec3::splat(0.8)
        } else {
            Vec3::splat(1.0)
        };

        commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform::from_translation(Vec3::new(
                obj.position.x as f32,
                obj.position.y as f32,
                obj.position.z as f32,
            )).with_scale(scale),
            PhysicsObject,
        ));
    }
}

fn draw_constraint_lines(
    sim_state: Res<SimulationState>,
    mut gizmos: Gizmos,
) {
    let joint_color = Color::srgb(0.9, 0.9, 0.2);
    let spring_color = Color::srgb(0.2, 0.9, 0.3);
    let rope_color = Color::srgb(0.9, 0.5, 0.2);

    // Draw joint lines
    for joint in &sim_state.joints {
        let p1 = Vec3::new(
            joint.object1.position.x as f32,
            joint.object1.position.y as f32,
            joint.object1.position.z as f32,
        );
        let p2 = Vec3::new(
            joint.object2.position.x as f32,
            joint.object2.position.y as f32,
            joint.object2.position.z as f32,
        );
        gizmos.line(p1, p2, joint_color);
    }

    // Draw spring lines
    for spring in &sim_state.springs {
        let p1 = Vec3::new(
            spring.object1.position.x as f32,
            spring.object1.position.y as f32,
            spring.object1.position.z as f32,
        );
        let p2 = Vec3::new(
            spring.object2.position.x as f32,
            spring.object2.position.y as f32,
            spring.object2.position.z as f32,
        );
        gizmos.line(p1, p2, spring_color);
    }

    // Draw rope lines
    for rope in &sim_state.ropes {
        let p1 = Vec3::new(
            rope.object1.position.x as f32,
            rope.object1.position.y as f32,
            rope.object1.position.z as f32,
        );
        let p2 = Vec3::new(
            rope.object2.position.x as f32,
            rope.object2.position.y as f32,
            rope.object2.position.z as f32,
        );

        // Color based on tautness
        let color = if rope.is_taut() {
            Color::srgb(1.0, 0.3, 0.1) // Taut = red-orange
        } else {
            rope_color // Slack = orange
        };
        gizmos.line(p1, p2, color);
    }

    // Draw spring connection lines (for soft body)
    for conn in &sim_state.spring_connections {
        if conn.idx1 < sim_state.objects.len() && conn.idx2 < sim_state.objects.len() {
            let obj1 = &sim_state.objects[conn.idx1];
            let obj2 = &sim_state.objects[conn.idx2];
            let p1 = Vec3::new(
                obj1.position.x as f32,
                obj1.position.y as f32,
                obj1.position.z as f32,
            );
            let p2 = Vec3::new(
                obj2.position.x as f32,
                obj2.position.y as f32,
                obj2.position.z as f32,
            );
            gizmos.line(p1, p2, spring_color);
        }
    }

    // Draw hinge anchor point
    for hinge in &sim_state.hinges {
        let anchor = Vec3::new(
            hinge.anchor.0 as f32,
            hinge.anchor.1 as f32,
            hinge.anchor.2 as f32,
        );
        // Draw axis indicator
        let axis = Vec3::new(
            hinge.axis.0 as f32,
            hinge.axis.1 as f32,
            hinge.axis.2 as f32,
        );
        gizmos.line(anchor - axis * 0.5, anchor + axis * 0.5, Color::srgb(0.9, 0.2, 0.9));
        gizmos.sphere(Isometry3d::from_translation(anchor), 0.15, Color::srgb(0.9, 0.2, 0.9));
    }
}

fn update_info_text(
    sim_state: Res<SimulationState>,
    mut query: Query<&mut Text, With<SimulationInfoText>>,
) {
    for mut text in query.iter_mut() {
        let status = if sim_state.paused { "PAUSED" } else { "Running" };
        let gravity = if sim_state.gravity_enabled { "ON" } else { "OFF" };
        let sim_time = sim_state.step_count as f32 * DT;

        let constraint_count = sim_state.joints.len()
            + sim_state.springs.len()
            + sim_state.ropes.len()
            + sim_state.hinges.len()
            + sim_state.spring_connections.len();

        **text = format!(
            "3D Constraints Demo\n\
             Scenario: {} [1-4]\n\
             Status: {}\n\
             Gravity: {} [G]\n\
             Objects: {}\n\
             Constraints: {}\n\
             Steps: {}\n\
             Sim Time: {:.2}s\n\n\
             Controls:\n\
             A/D: Rotate | W/S: Tilt\n\
             Q/E: Zoom | Space: Pause\n\
             R: Reset | Esc: Exit",
            sim_state.scenario.name(),
            status,
            gravity,
            sim_state.objects.len(),
            constraint_count,
            sim_state.step_count,
            sim_time,
        );
    }
}

fn handle_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut sim_state: ResMut<SimulationState>,
    mut exit: EventWriter<AppExit>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        sim_state.paused = !sim_state.paused;
    }

    if keyboard.just_pressed(KeyCode::KeyG) {
        sim_state.gravity_enabled = !sim_state.gravity_enabled;
    }

    if keyboard.just_pressed(KeyCode::KeyR) {
        // Reset current scenario
        sim_state.step_count = 0;
        match sim_state.scenario {
            Scenario::Pendulum => setup_pendulum_scenario(&mut sim_state),
            Scenario::SoftBody => setup_softbody_scenario(&mut sim_state),
            Scenario::Rope => setup_rope_scenario(&mut sim_state),
            Scenario::Hinge => setup_hinge_scenario(&mut sim_state),
        }
    }

    // Scenario switching
    if keyboard.just_pressed(KeyCode::Digit1) {
        sim_state.step_count = 0;
        setup_pendulum_scenario(&mut sim_state);
    }
    if keyboard.just_pressed(KeyCode::Digit2) {
        sim_state.step_count = 0;
        setup_softbody_scenario(&mut sim_state);
    }
    if keyboard.just_pressed(KeyCode::Digit3) {
        sim_state.step_count = 0;
        setup_rope_scenario(&mut sim_state);
    }
    if keyboard.just_pressed(KeyCode::Digit4) {
        sim_state.step_count = 0;
        setup_hinge_scenario(&mut sim_state);
    }

    if keyboard.just_pressed(KeyCode::Escape) {
        exit.send(AppExit::Success);
    }
}
