//! 3D Thermal Simulation Visualization
//!
//! This visualizes the 3D heat diffusion simulation using ThermalGrid3D.
//! Temperature is shown using colored cubes/spheres with color gradient.
//!
//! Controls:
//! - WASD/Arrows: Rotate camera
//! - Q/E: Zoom in/out
//! - Space: Pause/resume simulation
//! - R: Reset simulation
//! - 1: Scenario 1 - Hot center (sphere)
//! - 2: Scenario 2 - Hot bottom, cold top
//! - 3: Scenario 3 - Corner heat sources
//! - Z/X: Move slice plane position
//! - V: Toggle view mode (volume/slice)
//! - Escape: Exit

use bevy::prelude::*;
use bevy_visual_tests::{spawn_info_text, spawn_orbit_camera, OrbitCameraPlugin, SimulationInfoText};
use rs_physics::thermodynamics::{ThermalGrid3D, ThermalBoundaryCondition, GridFace};

const GRID_SIZE: usize = 20;
const CELL_SIZE: f32 = 1.0;

// Thermal properties
const THERMAL_DIFFUSIVITY: f64 = 1.0e-4; // m^2/s
const DT: f64 = 0.001; // Time step
const DX: f64 = 0.01; // Grid spacing in meters

// Temperature range for visualization (Kelvin)
const T_MIN: f64 = 250.0; // Cold (blue)
const T_MAX: f64 = 450.0; // Hot (red)
const T_AMBIENT: f64 = 300.0; // Room temperature

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "3D Thermal Simulation".to_string(),
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
    HotCenter,
    HotColdGradient,
    CornerSources,
}

#[derive(Resource)]
struct SimulationState {
    grid: ThermalGrid3D,
    paused: bool,
    step_count: u64,
    view_mode: ViewMode,
    slice_position: usize,
    scenario: Scenario,
    temperature_threshold: f64,
}

impl Default for SimulationState {
    fn default() -> Self {
        let grid = ThermalGrid3D::new(
            GRID_SIZE,
            GRID_SIZE,
            GRID_SIZE,
            T_AMBIENT,
            THERMAL_DIFFUSIVITY,
            DT,
            DX,
        )
        .expect("Failed to create thermal grid");

        Self {
            grid,
            paused: false,
            step_count: 0,
            view_mode: ViewMode::Volume,
            slice_position: GRID_SIZE / 2,
            scenario: Scenario::HotCenter,
            temperature_threshold: 0.1,
        }
    }
}

#[derive(Component)]
struct ThermalCell3D {
    x: usize,
    y: usize,
    z: usize,
}

#[derive(Resource)]
#[allow(dead_code)]
struct ThermalMeshes {
    cube_mesh: Handle<Mesh>,
}

#[derive(Resource)]
struct ThermalMaterials {
    materials: Vec<Handle<StandardMaterial>>,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut sim_state: ResMut<SimulationState>,
) {
    // Spawn orbit camera
    let camera_distance = GRID_SIZE as f32 * 2.5;
    spawn_orbit_camera(
        &mut commands,
        Vec3::new(camera_distance, camera_distance * 0.7, camera_distance),
        Vec3::ZERO,
    );

    // Lighting
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 300.0,
    });

    commands.spawn((
        PointLight {
            intensity: 3_000_000.0,
            range: 150.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(40.0, 40.0, 40.0),
    ));

    commands.spawn((
        PointLight {
            intensity: 1_500_000.0,
            range: 100.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(-30.0, -20.0, -30.0),
    ));

    // Create mesh for cells
    let cube_mesh = meshes.add(Cuboid::new(CELL_SIZE * 0.9, CELL_SIZE * 0.9, CELL_SIZE * 0.9));

    // Create a gradient of materials from blue (cold) to red (hot)
    let num_materials = 64;
    let mut material_handles = Vec::with_capacity(num_materials);

    for i in 0..num_materials {
        let t = i as f32 / (num_materials - 1) as f32;
        let color = temperature_to_color(t);
        let material = materials.add(StandardMaterial {
            base_color: color,
            emissive: LinearRgba::from(color) * 0.3,
            ..default()
        });
        material_handles.push(material);
    }

    commands.insert_resource(ThermalMeshes { cube_mesh: cube_mesh.clone() });
    commands.insert_resource(ThermalMaterials { materials: material_handles.clone() });

    // Create grid cells
    let offset = -(GRID_SIZE as f32 * CELL_SIZE) / 2.0;

    for z in 0..GRID_SIZE {
        for y in 0..GRID_SIZE {
            for x in 0..GRID_SIZE {
                let pos = Vec3::new(
                    offset + x as f32 * CELL_SIZE + CELL_SIZE / 2.0,
                    offset + y as f32 * CELL_SIZE + CELL_SIZE / 2.0,
                    offset + z as f32 * CELL_SIZE + CELL_SIZE / 2.0,
                );

                commands.spawn((
                    Mesh3d(cube_mesh.clone()),
                    MeshMaterial3d(material_handles[num_materials / 2].clone()),
                    Transform::from_translation(pos).with_scale(Vec3::splat(0.8)),
                    ThermalCell3D { x, y, z },
                    Visibility::Hidden,
                ));
            }
        }
    }

    // Spawn UI text
    spawn_info_text(&mut commands);

    // Set up initial scenario
    setup_scenario_hot_center(&mut sim_state);
}

fn temperature_to_color(t: f32) -> Color {
    // Blue (cold) -> White (neutral) -> Red (hot)
    if t < 0.5 {
        let t2 = t * 2.0;
        Color::srgb(t2, t2, 1.0)
    } else {
        let t2 = (t - 0.5) * 2.0;
        Color::srgb(1.0, 1.0 - t2, 1.0 - t2)
    }
}

fn setup_scenario_hot_center(sim_state: &mut SimulationState) {
    sim_state.grid.reset(T_AMBIENT);

    // Hot sphere in center
    let center = GRID_SIZE / 2;
    let radius = 4;

    for z in 0..GRID_SIZE {
        for y in 0..GRID_SIZE {
            for x in 0..GRID_SIZE {
                let dx = x as i32 - center as i32;
                let dy = y as i32 - center as i32;
                let dz = z as i32 - center as i32;
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq <= (radius * radius) as i32 {
                    let _ = sim_state.grid.set_temperature(x, y, z, T_MAX);
                }
            }
        }
    }

    sim_state.scenario = Scenario::HotCenter;
    sim_state.step_count = 0;
}

fn setup_scenario_gradient(sim_state: &mut SimulationState) {
    sim_state.grid.reset(T_AMBIENT);

    // Hot bottom, cold top
    sim_state.grid.set_boundary_condition(GridFace::Bottom, ThermalBoundaryCondition::Dirichlet(T_MAX));
    sim_state.grid.set_boundary_condition(GridFace::Top, ThermalBoundaryCondition::Dirichlet(T_MIN));

    sim_state.scenario = Scenario::HotColdGradient;
    sim_state.step_count = 0;
}

fn setup_scenario_corners(sim_state: &mut SimulationState) {
    sim_state.grid.reset(T_AMBIENT);

    // Hot corners at (0,0,0) and (max,max,max), cold at other corners
    let max = GRID_SIZE - 1;
    let radius = 3;

    // Hot corner 1 (origin)
    for z in 0..radius {
        for y in 0..radius {
            for x in 0..radius {
                let _ = sim_state.grid.set_temperature(x, y, z, T_MAX);
            }
        }
    }

    // Hot corner 2 (opposite)
    for z in (max - radius + 1)..=max {
        for y in (max - radius + 1)..=max {
            for x in (max - radius + 1)..=max {
                let _ = sim_state.grid.set_temperature(x, y, z, T_MAX);
            }
        }
    }

    // Cold corner 3
    for z in 0..radius {
        for y in (max - radius + 1)..=max {
            for x in 0..radius {
                let _ = sim_state.grid.set_temperature(x, y, z, T_MIN);
            }
        }
    }

    // Cold corner 4
    for z in (max - radius + 1)..=max {
        for y in 0..radius {
            for x in (max - radius + 1)..=max {
                let _ = sim_state.grid.set_temperature(x, y, z, T_MIN);
            }
        }
    }

    sim_state.scenario = Scenario::CornerSources;
    sim_state.step_count = 0;
}

fn simulation_step(mut sim_state: ResMut<SimulationState>) {
    if sim_state.paused {
        return;
    }

    // Run multiple steps per frame for faster simulation
    for _ in 0..5 {
        sim_state.grid.step();
    }
    sim_state.step_count += 5;
}

fn update_visualization(
    sim_state: Res<SimulationState>,
    thermal_materials: Res<ThermalMaterials>,
    mut query: Query<(&ThermalCell3D, &mut Visibility, &mut MeshMaterial3d<StandardMaterial>, &mut Transform)>,
) {
    let num_materials = thermal_materials.materials.len();

    for (cell, mut visibility, mut material, mut transform) in query.iter_mut() {
        // Determine if this cell should be visible based on view mode
        let should_show = match sim_state.view_mode {
            ViewMode::Volume => true,
            ViewMode::SliceXY => cell.z == sim_state.slice_position,
            ViewMode::SliceXZ => cell.y == sim_state.slice_position,
            ViewMode::SliceYZ => cell.x == sim_state.slice_position,
        };

        if !should_show {
            *visibility = Visibility::Hidden;
            continue;
        }

        let temperature = sim_state.grid.get_temperature(cell.x, cell.y, cell.z).unwrap_or(T_AMBIENT);

        // Normalize temperature to 0-1 range
        let t_norm = ((temperature - T_MIN) / (T_MAX - T_MIN)).clamp(0.0, 1.0);

        // For volume view, only show cells that are significantly different from ambient
        if sim_state.view_mode == ViewMode::Volume {
            let deviation = (temperature - T_AMBIENT).abs() / (T_MAX - T_AMBIENT);
            if deviation < sim_state.temperature_threshold {
                *visibility = Visibility::Hidden;
                continue;
            }

            // Scale based on temperature deviation
            let scale = 0.4 + deviation as f32 * 0.6;
            transform.scale = Vec3::splat(scale);
        } else {
            transform.scale = Vec3::splat(0.9);
        }

        *visibility = Visibility::Visible;

        // Select material based on temperature
        let material_idx = ((t_norm * (num_materials - 1) as f64) as usize).min(num_materials - 1);
        *material = MeshMaterial3d(thermal_materials.materials[material_idx].clone());
    }
}

fn update_info_text(
    sim_state: Res<SimulationState>,
    mut query: Query<&mut Text, With<SimulationInfoText>>,
) {
    for mut text in query.iter_mut() {
        let status = if sim_state.paused { "PAUSED" } else { "Running" };
        let sim_time = sim_state.step_count as f64 * DT;

        let avg_temp = sim_state.grid.average_temperature();
        let max_temp = sim_state.grid.max_temperature();
        let min_temp = sim_state.grid.min_temperature();

        let scenario_name = match sim_state.scenario {
            Scenario::HotCenter => "Hot Center Sphere",
            Scenario::HotColdGradient => "Hot/Cold Gradient",
            Scenario::CornerSources => "Corner Sources",
        };

        let view_mode_name = match sim_state.view_mode {
            ViewMode::Volume => "Volume".to_string(),
            ViewMode::SliceXY => format!("Slice XY (z={})", sim_state.slice_position),
            ViewMode::SliceXZ => format!("Slice XZ (y={})", sim_state.slice_position),
            ViewMode::SliceYZ => format!("Slice YZ (x={})", sim_state.slice_position),
        };

        **text = format!(
            "3D Thermal Simulation\n\
             Status: {}\n\
             Scenario: {}\n\
             View: {}\n\
             Grid: {}^3\n\
             Steps: {}\n\
             Sim Time: {:.3}s\n\n\
             Temperature (K):\n\
             Avg: {:.1}\n\
             Max: {:.1}\n\
             Min: {:.1}\n\n\
             Controls:\n\
             WASD: Rotate camera\n\
             Q/E: Zoom\n\
             Space: Pause\n\
             1/2/3: Scenarios\n\
             V: View mode\n\
             Z/X: Move slice\n\
             R: Reset\n\
             Esc: Exit",
            status,
            scenario_name,
            view_mode_name,
            GRID_SIZE,
            sim_state.step_count,
            sim_time,
            avg_temp,
            max_temp,
            min_temp,
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
        // Reset to current scenario
        match sim_state.scenario {
            Scenario::HotCenter => setup_scenario_hot_center(&mut sim_state),
            Scenario::HotColdGradient => setup_scenario_gradient(&mut sim_state),
            Scenario::CornerSources => setup_scenario_corners(&mut sim_state),
        }
    }

    // Scenario selection
    if keyboard.just_pressed(KeyCode::Digit1) {
        setup_scenario_hot_center(&mut sim_state);
    }
    if keyboard.just_pressed(KeyCode::Digit2) {
        setup_scenario_gradient(&mut sim_state);
    }
    if keyboard.just_pressed(KeyCode::Digit3) {
        setup_scenario_corners(&mut sim_state);
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

    // Slice position adjustment
    if keyboard.just_pressed(KeyCode::KeyZ) {
        if sim_state.slice_position > 0 {
            sim_state.slice_position -= 1;
        }
    }
    if keyboard.just_pressed(KeyCode::KeyX) {
        if sim_state.slice_position < GRID_SIZE - 1 {
            sim_state.slice_position += 1;
        }
    }

    // Temperature threshold adjustment
    if keyboard.just_pressed(KeyCode::Minus) {
        sim_state.temperature_threshold = (sim_state.temperature_threshold - 0.05).max(0.0);
    }
    if keyboard.just_pressed(KeyCode::Equal) {
        sim_state.temperature_threshold = (sim_state.temperature_threshold + 0.05).min(1.0);
    }

    if keyboard.just_pressed(KeyCode::Escape) {
        exit.send(AppExit::Success);
    }
}
