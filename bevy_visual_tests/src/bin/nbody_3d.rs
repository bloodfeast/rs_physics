//! 3D N-body cosmological simulation visualization
//!
//! This visualizes particles gravitationally attracting each other,
//! forming clusters similar to galaxy formation in the early universe.
//!
//! Controls:
//! - A/D or Left/Right: Rotate camera horizontally
//! - W/S or Up/Down: Rotate camera vertically
//! - Q/E: Zoom in/out
//! - Space: Pause/resume simulation
//! - R: Reset simulation
//! - Escape: Exit

use bevy::prelude::*;
use bevy_visual_tests::{
    spawn_orbit_camera, spawn_info_text, particle_material,
    OrbitCameraPlugin, SimulationInfoText,
};
use rs_physics::gpu::{GpuContext, GpuNBody3DSimulation, NBody3DParticle};
use rand::Rng;

const NUM_PARTICLES: usize = 20000;
const BOX_SIZE: f32 = 100.0;
const DT: f32 = 0.005;
const G: f32 = 5.0;
const SOFTENING: f32 = 1.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "3D N-Body Cosmological Simulation".to_string(),
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
            sync_particles,
            update_info_text,
            handle_input,
        ))
        .run();
}

#[derive(Resource)]
struct SimulationState {
    gpu: Option<GpuContext>,
    simulation: Option<GpuNBody3DSimulation>,
    paused: bool,
    step_count: u64,
    steps_per_frame: u32,
}

impl Default for SimulationState {
    fn default() -> Self {
        Self {
            gpu: None,
            simulation: None,
            paused: false,
            step_count: 0,
            steps_per_frame: 5,
        }
    }
}

#[derive(Component)]
struct ParticleEntity(usize);

#[derive(Resource)]
#[allow(dead_code)]
struct ParticleMeshes {
    mesh: Handle<Mesh>,
    materials: Vec<Handle<StandardMaterial>>,
}

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
    let simulation = GpuNBody3DSimulation::new(
        &gpu,
        &particles,
        DT,
        G,
        SOFTENING,
        None, // No periodic boundaries for visualization
    );

    sim_state.gpu = Some(gpu);
    sim_state.simulation = Some(simulation);

    // Create particle mesh (small sphere)
    let particle_mesh = meshes.add(Sphere::new(0.2));

    // Create materials with different colors based on initial position
    let particle_materials: Vec<Handle<StandardMaterial>> = (0..NUM_PARTICLES)
        .map(|i| {
            let hue = (i as f32 / NUM_PARTICLES as f32) * 360.0;
            let color = Color::hsl(hue, 0.8, 0.6);
            particle_material(&mut materials, color)
        })
        .collect();

    commands.insert_resource(ParticleMeshes {
        mesh: particle_mesh.clone(),
        materials: particle_materials.clone(),
    });

    // Spawn particle entities
    for (i, particle) in particles.iter().enumerate() {
        commands.spawn((
            Mesh3d(particle_mesh.clone()),
            MeshMaterial3d(particle_materials[i].clone()),
            Transform::from_translation(Vec3::new(
                particle.pos[0],
                particle.pos[1],
                particle.pos[2],
            )),
            ParticleEntity(i),
        ));
    }

    // Spawn camera
    spawn_orbit_camera(&mut commands, Vec3::new(0.0, 50.0, 150.0), Vec3::ZERO);

    // Spawn lighting
    commands.spawn((
        PointLight {
            intensity: 10_000_000.0,
            range: 500.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(0.0, 100.0, 0.0),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 5000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(50.0, 100.0, 50.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Ambient light
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 200.0,
    });

    // UI text
    spawn_info_text(&mut commands);
}

fn create_initial_particles() -> Vec<NBody3DParticle> {
    let mut rng = rand::thread_rng();
    let mut particles = Vec::with_capacity(NUM_PARTICLES);

    // Create a roughly uniform sphere with some clustering
    for i in 0..NUM_PARTICLES {
        // Fibonacci sphere distribution for more uniform coverage
        let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;
        let theta = 2.0 * std::f32::consts::PI * (i as f32) / golden_ratio;
        let phi = (1.0 - 2.0 * (i as f32 + 0.5) / NUM_PARTICLES as f32).acos();

        // Randomize radius for 3D distribution
        let r = BOX_SIZE * 0.314 * rng.gen::<f32>().powf(1.0 / 3.0);

        let x = r * phi.sin() * theta.cos();
        let y = r * phi.sin() * theta.sin();
        let z = r * phi.cos();

        // Add some random perturbation
        let perturbation = 2.0;
        let px = x + rng.gen_range(-perturbation..perturbation);
        let py = y + rng.gen_range(-perturbation..perturbation);
        let pz = z + rng.gen_range(-perturbation..perturbation);

        // Small random initial velocities (slight rotation around Y axis)
        let dist = (px * px + pz * pz).sqrt();
        let tangent_speed = 0.05 * dist.sqrt();
        let vx = -pz / dist.max(0.1) * tangent_speed + rng.gen_range(-0.1..0.1);
        let vy = rng.gen_range(-0.1..0.1);
        let vz = px / dist.max(0.1) * tangent_speed + rng.gen_range(-0.1..0.1);

        // Vary mass slightly
        let mass = 1.0 + rng.gen_range(-0.5..0.5);

        particles.push(NBody3DParticle::new([px, py, pz], [vx, vy, vz], mass));
    }

    particles
}

fn simulation_step(mut sim_state: ResMut<SimulationState>) {
    if sim_state.paused {
        return;
    }

    let steps = sim_state.steps_per_frame;

    // Need to destructure to avoid borrow conflicts
    let SimulationState {
        gpu: ref gpu_opt,
        simulation: ref mut sim_opt,
        ..
    } = *sim_state;

    if let (Some(gpu), Some(simulation)) = (gpu_opt, sim_opt) {
        simulation.step_n(gpu, steps);
    }

    sim_state.step_count += steps as u64;
}

fn sync_particles(
    sim_state: Res<SimulationState>,
    mut query: Query<(&ParticleEntity, &mut Transform)>,
) {
    if let (Some(gpu), Some(simulation)) = (&sim_state.gpu, &sim_state.simulation) {
        let particles = simulation.read_particles(gpu);

        for (particle_entity, mut transform) in query.iter_mut() {
            if let Some(p) = particles.get(particle_entity.0) {
                transform.translation = Vec3::new(p.pos[0], p.pos[1], p.pos[2]);
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

        **text = format!(
            "3D N-Body Cosmological Simulation\n\
             Status: {}\n\
             Particles: {}\n\
             Steps: {}\n\
             Sim Time: {:.2}\n\
             Steps/Frame: {}\n\n\
             Controls:\n\
             A/D: Rotate | W/S: Tilt\n\
             Q/E: Zoom | Space: Pause\n\
             R: Reset | Esc: Exit",
            status,
            NUM_PARTICLES,
            sim_state.step_count,
            sim_time,
            sim_state.steps_per_frame,
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

    if keyboard.just_pressed(KeyCode::KeyR) {
        // Reset simulation
        if let Some(gpu) = &sim_state.gpu {
            let particles = create_initial_particles();
            sim_state.simulation = Some(GpuNBody3DSimulation::new(
                gpu,
                &particles,
                DT,
                G,
                SOFTENING,
                None,
            ));
            sim_state.step_count = 0;
        }
    }

    if keyboard.just_pressed(KeyCode::Equal) || keyboard.just_pressed(KeyCode::NumpadAdd) {
        sim_state.steps_per_frame = (sim_state.steps_per_frame + 1).min(20);
    }

    if keyboard.just_pressed(KeyCode::Minus) || keyboard.just_pressed(KeyCode::NumpadSubtract) {
        sim_state.steps_per_frame = (sim_state.steps_per_frame - 1).max(1);
    }

    if keyboard.just_pressed(KeyCode::Escape) {
        exit.send(AppExit::Success);
    }
}
