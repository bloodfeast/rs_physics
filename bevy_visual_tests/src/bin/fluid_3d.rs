//! 3D Fluid Simulation Visualization
//!
//! This visualizes the 3D Eulerian fluid simulation using FluidGrid3D.
//! Shows a volumetric view with density as particle opacity/size.
//!
//! Controls:
//! - WASD/Arrows: Rotate camera
//! - Q/E: Zoom in/out
//! - Space: Pause/resume simulation
//! - R: Reset simulation
//! - 1/2/3: Preset scenarios (smoke plume, explosion, vortex)
//! - Z/X: Move slice plane (for cross-section view)
//! - V: Toggle view mode (volume/slice)
//! - Escape: Exit

use bevy::prelude::*;
use bevy_visual_tests::{spawn_info_text, spawn_orbit_camera, OrbitCameraPlugin, SimulationInfoText};
use rs_physics::fluid_dynamics::{FluidGrid3D, SolverConfig};

const GRID_SIZE: usize = 24;
const DT: f64 = 0.016;
const CELL_SIZE: f32 = 1.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "3D Fluid Simulation".to_string(),
                resolution: (1280.0, 720.0).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(OrbitCameraPlugin)
        .init_resource::<SimulationState>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                simulation_step,
                add_sources,
                update_visualization,
                update_info_text,
                handle_keyboard_input,
            ),
        )
        .run();
}

#[derive(Clone, Copy, PartialEq)]
enum ViewMode {
    Volume,
    SliceXY,
    SliceXZ,
    SliceYZ,
}

#[derive(Clone, Copy, PartialEq)]
enum Scenario {
    SmokePlume,
    Explosion,
    Vortex,
}

#[derive(Resource)]
struct SimulationState {
    grid: FluidGrid3D,
    paused: bool,
    step_count: u64,
    viscosity: f64,
    diffusion: f64,
    view_mode: ViewMode,
    slice_position: usize,
    scenario: Scenario,
    density_threshold: f64,
}

impl Default for SimulationState {
    fn default() -> Self {
        let solver_config = SolverConfig::high_quality();
        let grid = FluidGrid3D::with_solver(
            GRID_SIZE, GRID_SIZE, GRID_SIZE,
            0.0001, 0.1, DT, solver_config,
        ).expect("Failed to create fluid grid");

        Self {
            grid,
            paused: false,
            step_count: 0,
            viscosity: 0.1,
            diffusion: 0.0001,
            view_mode: ViewMode::Volume,
            slice_position: GRID_SIZE / 2,
            scenario: Scenario::SmokePlume,
            density_threshold: 0.1,
        }
    }
}

#[derive(Component)]
struct FluidParticle;

fn setup(mut commands: Commands) {
    // Spawn orbit camera
    let camera_distance = GRID_SIZE as f32 * 2.0;
    spawn_orbit_camera(
        &mut commands,
        Vec3::new(camera_distance, camera_distance * 0.7, camera_distance),
        Vec3::ZERO,
    );

    // Lighting
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 200.0,
    });

    commands.spawn((
        PointLight {
            intensity: 2_000_000.0,
            range: 100.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(30.0, 30.0, 30.0),
    ));

    // Spawn UI text
    spawn_info_text(&mut commands);
}

fn add_sources(mut sim_state: ResMut<SimulationState>) {
    if sim_state.paused {
        return;
    }

    let center = GRID_SIZE / 2;

    match sim_state.scenario {
        Scenario::SmokePlume => {
            // Add smoke rising from bottom center
            for dx in 0..3 {
                for dz in 0..3 {
                    let x = center - 1 + dx;
                    let z = center - 1 + dz;
                    let _ = sim_state.grid.add_density(x, 2, z, 5.0);
                    let _ = sim_state.grid.add_velocity(x, 2, z, 0.0, 8.0, 0.0);
                }
            }
        }
        Scenario::Explosion => {
            // Periodic explosion from center
            if sim_state.step_count % 60 == 0 {
                for dx in 0..3 {
                    for dy in 0..3 {
                        for dz in 0..3 {
                            let x = center - 1 + dx;
                            let y = center - 1 + dy;
                            let z = center - 1 + dz;
                            let _ = sim_state.grid.add_density(x, y, z, 20.0);

                            // Radial velocity
                            let vx = (dx as f64 - 1.0) * 10.0;
                            let vy = (dy as f64 - 1.0) * 10.0;
                            let vz = (dz as f64 - 1.0) * 10.0;
                            let _ = sim_state.grid.add_velocity(x, y, z, vx, vy, vz);
                        }
                    }
                }
            }
        }
        Scenario::Vortex => {
            // Two opposing jets creating a vortex
            let _ = sim_state.grid.add_density(center - 5, center, center, 3.0);
            let _ = sim_state.grid.add_velocity(center - 5, center, center, 5.0, 0.0, 2.0);

            let _ = sim_state.grid.add_density(center + 5, center, center, 3.0);
            let _ = sim_state.grid.add_velocity(center + 5, center, center, -5.0, 0.0, -2.0);
        }
    }
}

fn simulation_step(mut sim_state: ResMut<SimulationState>) {
    if sim_state.paused {
        return;
    }

    sim_state.grid.step();
    sim_state.step_count += 1;
}

fn update_visualization(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    sim_state: Res<SimulationState>,
    query: Query<Entity, With<FluidParticle>>,
) {
    // Remove old particles
    for entity in query.iter() {
        commands.entity(entity).despawn();
    }

    let grid_offset = -(GRID_SIZE as f32 * CELL_SIZE) / 2.0;
    let threshold = sim_state.density_threshold;

    // Create mesh for particles
    let particle_mesh = meshes.add(Sphere::new(CELL_SIZE * 0.4));

    match sim_state.view_mode {
        ViewMode::Volume => {
            // Render all cells above threshold (sparse for performance)
            for z in (1..GRID_SIZE - 1).step_by(1) {
                for y in (1..GRID_SIZE - 1).step_by(1) {
                    for x in (1..GRID_SIZE - 1).step_by(1) {
                        let density = sim_state.grid.get_density(x, y, z).unwrap_or(0.0);
                        if density > threshold {
                            spawn_particle(
                                &mut commands,
                                &mut materials,
                                &particle_mesh,
                                &sim_state,
                                x, y, z,
                                density,
                                grid_offset,
                            );
                        }
                    }
                }
            }
        }
        ViewMode::SliceXY => {
            let z = sim_state.slice_position.min(GRID_SIZE - 2).max(1);
            for y in 1..GRID_SIZE - 1 {
                for x in 1..GRID_SIZE - 1 {
                    let density = sim_state.grid.get_density(x, y, z).unwrap_or(0.0);
                    if density > threshold * 0.5 {
                        spawn_particle(
                            &mut commands,
                            &mut materials,
                            &particle_mesh,
                            &sim_state,
                            x, y, z,
                            density,
                            grid_offset,
                        );
                    }
                }
            }
        }
        ViewMode::SliceXZ => {
            let y = sim_state.slice_position.min(GRID_SIZE - 2).max(1);
            for z in 1..GRID_SIZE - 1 {
                for x in 1..GRID_SIZE - 1 {
                    let density = sim_state.grid.get_density(x, y, z).unwrap_or(0.0);
                    if density > threshold * 0.5 {
                        spawn_particle(
                            &mut commands,
                            &mut materials,
                            &particle_mesh,
                            &sim_state,
                            x, y, z,
                            density,
                            grid_offset,
                        );
                    }
                }
            }
        }
        ViewMode::SliceYZ => {
            let x = sim_state.slice_position.min(GRID_SIZE - 2).max(1);
            for z in 1..GRID_SIZE - 1 {
                for y in 1..GRID_SIZE - 1 {
                    let density = sim_state.grid.get_density(x, y, z).unwrap_or(0.0);
                    if density > threshold * 0.5 {
                        spawn_particle(
                            &mut commands,
                            &mut materials,
                            &particle_mesh,
                            &sim_state,
                            x, y, z,
                            density,
                            grid_offset,
                        );
                    }
                }
            }
        }
    }
}

fn spawn_particle(
    commands: &mut Commands,
    materials: &mut Assets<StandardMaterial>,
    mesh: &Handle<Mesh>,
    sim_state: &SimulationState,
    x: usize, y: usize, z: usize,
    density: f64,
    grid_offset: f32,
) {
    let (vx, vy, vz) = sim_state.grid.get_velocity(x, y, z).unwrap_or((0.0, 0.0, 0.0));
    let vel_mag = (vx * vx + vy * vy + vz * vz).sqrt();

    // Map density to alpha and size
    let alpha = (density / 20.0).min(0.9).max(0.1) as f32;
    let scale = (density / 10.0).min(1.5).max(0.3) as f32;

    // Map velocity to hue
    let hue = if vel_mag > 0.1 {
        // Use velocity direction for color
        let angle = vz.atan2(vx);
        ((angle + std::f64::consts::PI) / (2.0 * std::f64::consts::PI) * 360.0) as f32
    } else {
        // Default orange/red for stationary
        30.0
    };

    let saturation = (vel_mag / 5.0).min(1.0) as f32;
    let color = Color::hsla(hue, saturation.max(0.5), 0.6, alpha);

    let pos_x = grid_offset + x as f32 * CELL_SIZE;
    let pos_y = grid_offset + y as f32 * CELL_SIZE;
    let pos_z = grid_offset + z as f32 * CELL_SIZE;

    commands.spawn((
        Mesh3d(mesh.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: color,
            alpha_mode: AlphaMode::Blend,
            emissive: LinearRgba::from(color) * 0.5,
            ..default()
        })),
        Transform::from_xyz(pos_x, pos_y, pos_z)
            .with_scale(Vec3::splat(scale)),
        FluidParticle,
    ));
}

fn update_info_text(
    sim_state: Res<SimulationState>,
    mut query: Query<&mut Text, With<SimulationInfoText>>,
) {
    for mut text in query.iter_mut() {
        let status = if sim_state.paused { "PAUSED" } else { "Running" };
        let sim_time = sim_state.step_count as f64 * DT;
        let kinetic_energy = sim_state.grid.get_kinetic_energy();
        let total_mass = sim_state.grid.get_total_mass();

        let view_mode_str = match sim_state.view_mode {
            ViewMode::Volume => "Volume",
            ViewMode::SliceXY => format!("Slice XY (z={})", sim_state.slice_position).leak(),
            ViewMode::SliceXZ => format!("Slice XZ (y={})", sim_state.slice_position).leak(),
            ViewMode::SliceYZ => format!("Slice YZ (x={})", sim_state.slice_position).leak(),
        };

        let scenario_str = match sim_state.scenario {
            Scenario::SmokePlume => "Smoke Plume",
            Scenario::Explosion => "Explosion",
            Scenario::Vortex => "Vortex",
        };

        **text = format!(
            "3D Fluid Simulation\n\
             Status: {}\n\
             Grid: {}^3\n\
             Scenario: {}\n\
             Steps: {}\n\
             Sim Time: {:.2}s\n\
             View: {}\n\
             Kinetic Energy: {:.2}\n\
             Total Mass: {:.2}\n\n\
             Controls:\n\
             WASD: Rotate camera\n\
             Q/E: Zoom\n\
             1/2/3: Scenarios\n\
             V: Toggle view mode\n\
             Z/X: Move slice\n\
             Space: Pause\n\
             R: Reset\n\
             Esc: Exit",
            status,
            GRID_SIZE,
            scenario_str,
            sim_state.step_count,
            sim_time,
            view_mode_str,
            kinetic_energy,
            total_mass,
        );
    }
}

fn handle_keyboard_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut sim_state: ResMut<SimulationState>,
    mut exit: EventWriter<AppExit>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        sim_state.paused = !sim_state.paused;
    }

    if keyboard.just_pressed(KeyCode::KeyR) {
        recreate_grid(&mut sim_state);
    }

    // Scenario selection
    if keyboard.just_pressed(KeyCode::Digit1) {
        sim_state.scenario = Scenario::SmokePlume;
        recreate_grid(&mut sim_state);
    }
    if keyboard.just_pressed(KeyCode::Digit2) {
        sim_state.scenario = Scenario::Explosion;
        recreate_grid(&mut sim_state);
    }
    if keyboard.just_pressed(KeyCode::Digit3) {
        sim_state.scenario = Scenario::Vortex;
        recreate_grid(&mut sim_state);
    }

    // View mode toggle
    if keyboard.just_pressed(KeyCode::KeyV) {
        sim_state.view_mode = match sim_state.view_mode {
            ViewMode::Volume => ViewMode::SliceXY,
            ViewMode::SliceXY => ViewMode::SliceXZ,
            ViewMode::SliceXZ => ViewMode::SliceYZ,
            ViewMode::SliceYZ => ViewMode::Volume,
        };
    }

    // Slice position controls
    if keyboard.just_pressed(KeyCode::KeyZ) {
        if sim_state.slice_position > 1 {
            sim_state.slice_position -= 1;
        }
    }
    if keyboard.just_pressed(KeyCode::KeyX) {
        if sim_state.slice_position < GRID_SIZE - 2 {
            sim_state.slice_position += 1;
        }
    }

    // Density threshold adjustment
    if keyboard.just_pressed(KeyCode::Minus) {
        sim_state.density_threshold = (sim_state.density_threshold - 0.1).max(0.01);
    }
    if keyboard.just_pressed(KeyCode::Equal) {
        sim_state.density_threshold = (sim_state.density_threshold + 0.1).min(5.0);
    }

    if keyboard.just_pressed(KeyCode::Escape) {
        exit.send(AppExit::Success);
    }
}

fn recreate_grid(sim_state: &mut SimulationState) {
    let solver_config = SolverConfig::high_quality();
    sim_state.grid = FluidGrid3D::with_solver(
        GRID_SIZE, GRID_SIZE, GRID_SIZE,
        sim_state.diffusion,
        sim_state.viscosity,
        DT,
        solver_config,
    ).expect("Failed to create fluid grid");
    sim_state.step_count = 0;
}
