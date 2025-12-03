//! 2D N-body gravitational simulation visualization
//!
//! This visualizes particles gravitationally attracting each other in 2D,
//! useful for visualizing orbital mechanics and galaxy formation.
//!
//! Controls:
//! - A/D or Left/Right: Rotate camera
//! - Q/E: Zoom in/out
//! - Space: Pause/resume simulation
//! - R: Reset simulation
//! - 1-4: Load different presets
//! - Escape: Exit

use bevy::prelude::*;
use bevy_visual_tests::{spawn_info_text, particle_material, SimulationInfoText};
use rs_physics::gpu::{GpuContext, GpuNBodySimulation, NBodyParticle};
use rand::Rng;

const DT: f32 = 0.002;
const G: f32 = 10.0;
const SOFTENING: f32 = 0.5;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "2D N-Body Gravitational Simulation".to_string(),
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum Preset {
    RandomCloud,
    BinarySystem,
    SolarSystem,
    GalaxyCollision,
}

impl Preset {
    fn name(&self) -> &'static str {
        match self {
            Preset::RandomCloud => "Random Cloud",
            Preset::BinarySystem => "Binary System",
            Preset::SolarSystem => "Solar System",
            Preset::GalaxyCollision => "Galaxy Collision",
        }
    }
}

#[derive(Resource)]
struct SimulationState {
    gpu: Option<GpuContext>,
    simulation: Option<GpuNBodySimulation>,
    paused: bool,
    step_count: u64,
    steps_per_frame: u32,
    num_particles: usize,
    preset: Preset,
}

impl Default for SimulationState {
    fn default() -> Self {
        Self {
            gpu: None,
            simulation: None,
            paused: false,
            step_count: 0,
            steps_per_frame: 10,
            num_particles: 0,
            preset: Preset::RandomCloud,
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

    sim_state.gpu = Some(gpu);

    // Load initial preset
    load_preset(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut sim_state,
        Preset::RandomCloud,
    );

    // Spawn camera looking down at XY plane
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 150.0).looking_at(Vec3::ZERO, Vec3::Y),
        MainCamera,
    ));

    // Lighting
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 500.0,
    });

    // UI text
    spawn_info_text(&mut commands);
}

fn load_preset(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    sim_state: &mut SimulationState,
    preset: Preset,
) {
    // Create particles based on preset
    let particles = match preset {
        Preset::RandomCloud => create_random_cloud(5000),
        Preset::BinarySystem => create_binary_system(),
        Preset::SolarSystem => create_solar_system(),
        Preset::GalaxyCollision => create_galaxy_collision(),
    };

    let num_particles = particles.len();
    sim_state.num_particles = num_particles;
    sim_state.preset = preset;
    sim_state.step_count = 0;

    // Create GPU simulation
    if let Some(gpu) = &sim_state.gpu {
        sim_state.simulation = Some(GpuNBodySimulation::new(
            gpu,
            &particles,
            DT,
            G,
            SOFTENING,
        ));
    }

    // Create mesh for particles
    let particle_mesh = meshes.add(Sphere::new(0.5));

    // Spawn particle entities
    for (i, particle) in particles.iter().enumerate() {
        // Color based on mass (heavier = more yellow/white, lighter = more blue)
        let mass_factor = (particle.mass / 100.0).min(1.0);
        let color = Color::hsl(
            240.0 - mass_factor * 200.0, // Blue to yellow
            0.8,
            0.5 + mass_factor * 0.3,
        );

        // Size based on mass
        let scale = (particle.mass.sqrt() * 0.1).max(0.3).min(3.0);

        commands.spawn((
            Mesh3d(particle_mesh.clone()),
            MeshMaterial3d(particle_material(materials, color)),
            Transform::from_translation(Vec3::new(particle.pos[0], particle.pos[1], 0.0))
                .with_scale(Vec3::splat(scale)),
            ParticleEntity(i),
        ));
    }
}

fn create_random_cloud(n: usize) -> Vec<NBodyParticle> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| {
            let r = rng.gen::<f32>().sqrt() * 50.0;
            let theta = rng.gen::<f32>() * std::f32::consts::TAU;
            let x = r * theta.cos();
            let y = r * theta.sin();

            // Slight rotation
            let tangent_speed = 0.3 * r.sqrt();
            let vx = -y / r.max(0.1) * tangent_speed + rng.gen_range(-0.5..0.5);
            let vy = x / r.max(0.1) * tangent_speed + rng.gen_range(-0.5..0.5);

            NBodyParticle {
                pos: [x, y],
                vel: [vx, vy],
                mass: 1.0 + rng.gen_range(-0.3..0.3),
                _padding: [0.0; 3],
            }
        })
        .collect()
}

fn create_binary_system() -> Vec<NBodyParticle> {
    let m1 = 500.0;
    let m2 = 500.0;
    let sep = 30.0;

    // Orbital velocity for circular orbit
    let v = (G * (m1 + m2) / sep).sqrt() * 0.5;

    vec![
        NBodyParticle {
            pos: [-sep / 2.0, 0.0],
            vel: [0.0, -v * m2 / (m1 + m2)],
            mass: m1,
            _padding: [0.0; 3],
        },
        NBodyParticle {
            pos: [sep / 2.0, 0.0],
            vel: [0.0, v * m1 / (m1 + m2)],
            mass: m2,
            _padding: [0.0; 3],
        },
    ]
}

fn create_solar_system() -> Vec<NBodyParticle> {
    let sun_mass = 1000.0;
    let mut particles = vec![
        // Sun
        NBodyParticle {
            pos: [0.0, 0.0],
            vel: [0.0, 0.0],
            mass: sun_mass,
            _padding: [0.0; 3],
        },
    ];

    // Planets at different distances
    let planet_data = [
        (10.0, 5.0),   // Inner planet
        (20.0, 3.0),   //
        (35.0, 8.0),   //
        (50.0, 2.0),   //
        (70.0, 15.0),  // Gas giant
    ];

    for (distance, mass) in planet_data {
        let v = (G * sun_mass / distance).sqrt();
        particles.push(NBodyParticle {
            pos: [distance, 0.0],
            vel: [0.0, v],
            mass,
            _padding: [0.0; 3],
        });
    }

    particles
}

fn create_galaxy_collision() -> Vec<NBodyParticle> {
    let mut rng = rand::thread_rng();
    let mut particles = Vec::new();

    // Galaxy 1 (centered at -30, 0)
    let center1 = [-30.0, 10.0];
    let vel1 = [1.5, -0.5];
    for _ in 0..200 {
        let r = rng.gen::<f32>().sqrt() * 20.0;
        let theta = rng.gen::<f32>() * std::f32::consts::TAU;
        let x = center1[0] + r * theta.cos();
        let y = center1[1] + r * theta.sin();

        let tangent_speed = 0.5 * r.sqrt();
        let vx = vel1[0] - (y - center1[1]) / r.max(0.1) * tangent_speed;
        let vy = vel1[1] + (x - center1[0]) / r.max(0.1) * tangent_speed;

        particles.push(NBodyParticle {
            pos: [x, y],
            vel: [vx, vy],
            mass: 1.0,
            _padding: [0.0; 3],
        });
    }

    // Galaxy 2 (centered at 30, 0)
    let center2 = [30.0, -10.0];
    let vel2 = [-1.5, 0.5];
    for _ in 0..200 {
        let r = rng.gen::<f32>().sqrt() * 20.0;
        let theta = rng.gen::<f32>() * std::f32::consts::TAU;
        let x = center2[0] + r * theta.cos();
        let y = center2[1] + r * theta.sin();

        let tangent_speed = 0.5 * r.sqrt();
        let vx = vel2[0] + (y - center2[1]) / r.max(0.1) * tangent_speed;
        let vy = vel2[1] - (x - center2[0]) / r.max(0.1) * tangent_speed;

        particles.push(NBodyParticle {
            pos: [x, y],
            vel: [vx, vy],
            mass: 1.0,
            _padding: [0.0; 3],
        });
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

        **text = format!(
            "2D N-Body Simulation\n\
             Preset: {}\n\
             Status: {}\n\
             Particles: {}\n\
             Steps: {}\n\
             Sim Time: {:.2}\n\n\
             Controls:\n\
             Q/E: Zoom | Space: Pause\n\
             1: Random Cloud\n\
             2: Binary System\n\
             3: Solar System\n\
             4: Galaxy Collision\n\
             R: Reset | Esc: Exit",
            sim_state.preset.name(),
            status,
            sim_state.num_particles,
            sim_state.step_count,
            sim_time,
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
            transform.translation.z = (transform.translation.z + zoom_speed * delta).min(500.0);
        }
    }
}

fn handle_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut sim_state: ResMut<SimulationState>,
    particles_query: Query<Entity, With<ParticleEntity>>,
    mut exit: EventWriter<AppExit>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        sim_state.paused = !sim_state.paused;
    }

    let mut new_preset = None;

    if keyboard.just_pressed(KeyCode::Digit1) {
        new_preset = Some(Preset::RandomCloud);
    }
    if keyboard.just_pressed(KeyCode::Digit2) {
        new_preset = Some(Preset::BinarySystem);
    }
    if keyboard.just_pressed(KeyCode::Digit3) {
        new_preset = Some(Preset::SolarSystem);
    }
    if keyboard.just_pressed(KeyCode::Digit4) {
        new_preset = Some(Preset::GalaxyCollision);
    }
    if keyboard.just_pressed(KeyCode::KeyR) {
        new_preset = Some(sim_state.preset);
    }

    if let Some(preset) = new_preset {
        // Despawn existing particles
        for entity in particles_query.iter() {
            commands.entity(entity).despawn();
        }

        // Load new preset
        load_preset(
            &mut commands,
            &mut meshes,
            &mut materials,
            &mut sim_state,
            preset,
        );
    }

    if keyboard.just_pressed(KeyCode::Escape) {
        exit.send(AppExit::Success);
    }
}
