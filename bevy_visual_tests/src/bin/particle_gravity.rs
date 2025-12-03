//! Simple particle gravity simulation visualization
//!
//! This visualizes particles falling under gravity using the GPU
//! particle integration shader - a simpler demo than N-body.
//!
//! Controls:
//! - Space: Pause/resume simulation
//! - R: Reset simulation
//! - G: Toggle gravity direction
//! - Escape: Exit

use bevy::prelude::*;
use bevy_visual_tests::{spawn_info_text, particle_material, SimulationInfoText};
use rs_physics::gpu::{GpuContext, GpuParticleSimulation, GpuParticle};
use rand::Rng;

const NUM_PARTICLES: usize = 5000;
const DT: f32 = 0.016;
const GRAVITY: f32 = -20.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "GPU Particle Gravity Simulation".to_string(),
                resolution: (1280.0, 720.0).into(),
                ..default()
            }),
            ..default()
        }))
        .init_resource::<SimulationState>()
        .add_systems(Startup, setup)
        .add_systems(Update, (
            simulation_step,
            sync_particles,
            update_info_text,
            handle_input,
            camera_controls,
        ))
        .run();
}

#[derive(Resource)]
struct SimulationState {
    gpu: Option<GpuContext>,
    simulation: Option<GpuParticleSimulation>,
    paused: bool,
    step_count: u64,
    gravity: f32,
}

impl Default for SimulationState {
    fn default() -> Self {
        Self {
            gpu: None,
            simulation: None,
            paused: false,
            step_count: 0,
            gravity: GRAVITY,
        }
    }
}

#[derive(Component)]
struct ParticleEntity(usize);

#[derive(Component)]
struct MainCamera;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut sim_state: ResMut<SimulationState>,
) {
    // Initialize GPU
    let gpu = match GpuContext::new() {
        Some(g) => {
            info!("GPU initialized: {:?}", g.adapter_info().name);
            g
        }
        None => {
            error!("Failed to initialize GPU!");
            return;
        }
    };

    // Create initial particle distribution
    let particles = create_initial_particles();

    // Create GPU simulation
    let simulation = GpuParticleSimulation::new(&gpu, &particles, DT, GRAVITY);

    sim_state.gpu = Some(gpu);
    sim_state.simulation = Some(simulation);

    // Create particle mesh (small sphere)
    let particle_mesh = meshes.add(Sphere::new(0.15));

    // Create a gradient of materials
    let num_colors = 10;
    let particle_materials: Vec<Handle<StandardMaterial>> = (0..num_colors)
        .map(|i| {
            let t = i as f32 / num_colors as f32;
            let color = Color::hsl(200.0 + t * 60.0, 0.8, 0.5 + t * 0.3);
            particle_material(&mut materials, color)
        })
        .collect();

    // Spawn particle entities
    for (i, particle) in particles.iter().enumerate() {
        let material_idx = (i * num_colors / NUM_PARTICLES) % num_colors;
        commands.spawn((
            Mesh3d(particle_mesh.clone()),
            MeshMaterial3d(particle_materials[material_idx].clone()),
            Transform::from_translation(Vec3::new(particle.pos[0], particle.pos[1], 0.0)),
            ParticleEntity(i),
        ));
    }

    // Spawn camera looking at the scene
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 100.0).looking_at(Vec3::ZERO, Vec3::Y),
        MainCamera,
    ));

    // Lighting
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 300.0,
    });

    commands.spawn((
        PointLight {
            intensity: 5_000_000.0,
            range: 200.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(0.0, 50.0, 50.0),
    ));

    // Spawn ground plane visual
    let ground_mesh = meshes.add(Cuboid::new(200.0, 1.0, 10.0));
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.3, 0.3, 0.35),
        ..default()
    });
    commands.spawn((
        Mesh3d(ground_mesh),
        MeshMaterial3d(ground_material),
        Transform::from_xyz(0.0, -50.0, 0.0),
    ));

    // UI text
    spawn_info_text(&mut commands);
}

fn create_initial_particles() -> Vec<GpuParticle> {
    let mut rng = rand::thread_rng();

    (0..NUM_PARTICLES)
        .map(|i| {
            // Grid-like initial positions with some randomness
            let cols = 100;
            let col = i % cols;
            let row = i / cols;

            let x = (col as f32 - cols as f32 / 2.0) * 0.8 + rng.gen_range(-0.2..0.2);
            let y = 30.0 + row as f32 * 0.8 + rng.gen_range(-0.2..0.2);

            // Small random initial velocities
            let vx = rng.gen_range(-2.0..2.0);
            let vy = rng.gen_range(-1.0..1.0);

            GpuParticle {
                pos: [x, y],
                vel: [vx, vy],
                mass: 1.0,
                _padding: [0.0; 3],
            }
        })
        .collect()
}

fn simulation_step(mut sim_state: ResMut<SimulationState>) {
    if sim_state.paused {
        return;
    }

    if let (Some(gpu), Some(simulation)) = (&sim_state.gpu, &sim_state.simulation) {
        simulation.step(gpu);
        sim_state.step_count += 1;
    }
}

fn sync_particles(
    sim_state: Res<SimulationState>,
    mut query: Query<(&ParticleEntity, &mut Transform)>,
) {
    if let (Some(gpu), Some(simulation)) = (&sim_state.gpu, &sim_state.simulation) {
        let particles = simulation.read_particles(gpu);

        for (particle_entity, mut transform) in query.iter_mut() {
            if let Some(p) = particles.get(particle_entity.0) {
                transform.translation.x = p.pos[0];
                transform.translation.y = p.pos[1];
            }
        }
    }
}

fn update_info_text(
    sim_state: Res<SimulationState>,
    mut query: Query<&mut Text, With<SimulationInfoText>>,
) {
    for mut text in query.iter_mut() {
        let status = if sim_state.paused { "PAUSED" } else { "Running" };
        let sim_time = sim_state.step_count as f32 * DT;
        let gravity_dir = if sim_state.gravity < 0.0 { "Down" } else { "Up" };

        **text = format!(
            "GPU Particle Gravity\n\
             Status: {}\n\
             Particles: {}\n\
             Steps: {}\n\
             Sim Time: {:.2}s\n\
             Gravity: {} ({:.1})\n\n\
             Controls:\n\
             Q/E: Zoom\n\
             Space: Pause\n\
             G: Flip gravity\n\
             R: Reset\n\
             Esc: Exit",
            status,
            NUM_PARTICLES,
            sim_state.step_count,
            sim_time,
            gravity_dir,
            sim_state.gravity,
        );
    }
}

fn camera_controls(
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut query: Query<&mut Transform, With<MainCamera>>,
) {
    let zoom_speed = 50.0;
    let delta = time.delta_secs();

    for mut transform in query.iter_mut() {
        if keyboard.pressed(KeyCode::KeyQ) {
            transform.translation.z = (transform.translation.z - zoom_speed * delta).max(20.0);
        }
        if keyboard.pressed(KeyCode::KeyE) {
            transform.translation.z = (transform.translation.z + zoom_speed * delta).min(300.0);
        }
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
        sim_state.gravity = -sim_state.gravity;
        if let (Some(gpu), Some(simulation)) = (&sim_state.gpu, &sim_state.simulation) {
            simulation.set_params(gpu, DT, sim_state.gravity);
        }
    }

    if keyboard.just_pressed(KeyCode::KeyR) {
        // Reset simulation
        if let Some(gpu) = &sim_state.gpu {
            let particles = create_initial_particles();
            sim_state.simulation = Some(GpuParticleSimulation::new(
                gpu,
                &particles,
                DT,
                sim_state.gravity,
            ));
            sim_state.step_count = 0;
        }
    }

    if keyboard.just_pressed(KeyCode::Escape) {
        exit.send(AppExit::Success);
    }
}
