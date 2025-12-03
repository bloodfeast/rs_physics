//! 2D Thermal Simulation Visualization
//!
//! This visualizes the 2D heat diffusion simulation using ThermalGrid.
//! Temperature is shown as a color gradient from blue (cold) to red (hot).
//!
//! Controls:
//! - Left mouse: Add heat source (increase temperature)
//! - Right mouse: Add cold spot (decrease temperature)
//! - Space: Pause/resume simulation
//! - R: Reset simulation
//! - 1: Scenario 1 - Hot center
//! - 2: Scenario 2 - Hot left, cold right
//! - 3: Scenario 3 - Checkerboard pattern
//! - Escape: Exit

use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy_visual_tests::{spawn_info_text, SimulationInfoText};
use rs_physics::thermodynamics::{ThermalGrid, ThermalBoundaryCondition, GridSide};

const GRID_WIDTH: usize = 80;
const GRID_HEIGHT: usize = 60;
const CELL_SIZE: f32 = 10.0;

// Thermal properties (copper-like)
const THERMAL_DIFFUSIVITY: f64 = 1.11e-4; // m^2/s for copper
const DT: f64 = 0.001; // Time step (small for stability)
const DX: f64 = 0.01; // Grid spacing in meters

// Temperature range for visualization (Kelvin)
const T_MIN: f64 = 250.0; // Cold (blue)
const T_MAX: f64 = 450.0; // Hot (red)
const T_AMBIENT: f64 = 300.0; // Room temperature

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "2D Thermal Simulation".to_string(),
                resolution: (
                    GRID_WIDTH as f32 * CELL_SIZE + 200.0,
                    GRID_HEIGHT as f32 * CELL_SIZE + 40.0,
                )
                    .into(),
                ..default()
            }),
            ..default()
        }))
        .init_resource::<SimulationState>()
        .init_resource::<MouseState>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                handle_mouse_input,
                simulation_step,
                update_visualization,
                update_info_text,
                handle_keyboard_input,
            ),
        )
        .run();
}

#[derive(Resource)]
struct SimulationState {
    grid: ThermalGrid,
    paused: bool,
    step_count: u64,
    scenario: usize,
}

impl Default for SimulationState {
    fn default() -> Self {
        let grid = ThermalGrid::new(
            GRID_WIDTH,
            GRID_HEIGHT,
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
            scenario: 0,
        }
    }
}

#[derive(Resource, Default)]
struct MouseState {
    last_pos: Option<Vec2>,
}

#[derive(Component)]
struct ThermalCell {
    x: usize,
    y: usize,
}

#[derive(Component)]
struct MainCamera;

fn setup(mut commands: Commands, mut sim_state: ResMut<SimulationState>) {
    // Calculate grid offset to center it
    let grid_offset_x = -(GRID_WIDTH as f32 * CELL_SIZE) / 2.0 + 100.0;
    let grid_offset_y = -(GRID_HEIGHT as f32 * CELL_SIZE) / 2.0;

    // Spawn 2D camera
    commands.spawn((Camera2d::default(), MainCamera));

    // Create grid cells as sprites
    for y in 0..GRID_HEIGHT {
        for x in 0..GRID_WIDTH {
            let pos_x = grid_offset_x + x as f32 * CELL_SIZE + CELL_SIZE / 2.0;
            let pos_y = grid_offset_y + y as f32 * CELL_SIZE + CELL_SIZE / 2.0;

            commands.spawn((
                Sprite {
                    color: Color::srgb(0.0, 0.0, 0.0),
                    custom_size: Some(Vec2::new(CELL_SIZE - 1.0, CELL_SIZE - 1.0)),
                    anchor: Anchor::Center,
                    ..default()
                },
                Transform::from_xyz(pos_x, pos_y, 0.0),
                ThermalCell { x, y },
            ));
        }
    }

    // UI text
    spawn_info_text(&mut commands);

    // Set up initial scenario
    setup_scenario_1(&mut sim_state);
}

fn setup_scenario_1(sim_state: &mut SimulationState) {
    // Hot center spot
    sim_state.grid.reset(T_AMBIENT);
    let cx = GRID_WIDTH / 2;
    let cy = GRID_HEIGHT / 2;
    let radius = 5;
    for y in (cy.saturating_sub(radius))..=(cy + radius).min(GRID_HEIGHT - 1) {
        for x in (cx.saturating_sub(radius))..=(cx + radius).min(GRID_WIDTH - 1) {
            let dx = x as i32 - cx as i32;
            let dy = y as i32 - cy as i32;
            if dx * dx + dy * dy <= (radius * radius) as i32 {
                let _ = sim_state.grid.set_temperature(x, y, T_MAX);
            }
        }
    }
    sim_state.scenario = 1;
    sim_state.step_count = 0;
}

fn setup_scenario_2(sim_state: &mut SimulationState) {
    // Hot left boundary, cold right boundary
    sim_state.grid.reset(T_AMBIENT);
    sim_state.grid.set_boundary_condition(GridSide::Left, ThermalBoundaryCondition::Dirichlet(T_MAX));
    sim_state.grid.set_boundary_condition(GridSide::Right, ThermalBoundaryCondition::Dirichlet(T_MIN));
    sim_state.scenario = 2;
    sim_state.step_count = 0;
}

fn setup_scenario_3(sim_state: &mut SimulationState) {
    // Checkerboard pattern
    sim_state.grid.reset(T_AMBIENT);
    for y in 0..GRID_HEIGHT {
        for x in 0..GRID_WIDTH {
            let checker_size = 10;
            let is_hot = ((x / checker_size) + (y / checker_size)) % 2 == 0;
            let temp = if is_hot { T_MAX } else { T_MIN };
            let _ = sim_state.grid.set_temperature(x, y, temp);
        }
    }
    sim_state.scenario = 3;
    sim_state.step_count = 0;
}

fn handle_mouse_input(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    camera_query: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    mut sim_state: ResMut<SimulationState>,
    mut mouse_state: ResMut<MouseState>,
) {
    let window = windows.single();
    let (camera, camera_transform) = camera_query.single();

    // Get cursor position in world coordinates
    let cursor_world_pos = window
        .cursor_position()
        .and_then(|cursor| camera.viewport_to_world_2d(camera_transform, cursor).ok());

    if let Some(world_pos) = cursor_world_pos {
        // Convert world position to grid coordinates
        let grid_offset_x = -(GRID_WIDTH as f32 * CELL_SIZE) / 2.0 + 100.0;
        let grid_offset_y = -(GRID_HEIGHT as f32 * CELL_SIZE) / 2.0;

        let grid_x = ((world_pos.x - grid_offset_x) / CELL_SIZE) as i32;
        let grid_y = ((world_pos.y - grid_offset_y) / CELL_SIZE) as i32;

        // Check if within grid bounds
        if grid_x >= 0
            && grid_x < GRID_WIDTH as i32
            && grid_y >= 0
            && grid_y < GRID_HEIGHT as i32
        {
            let x = grid_x as usize;
            let y = grid_y as usize;

            // Heat brush - affect a small area
            let brush_radius = 2i32;
            for dy in -brush_radius..=brush_radius {
                for dx in -brush_radius..=brush_radius {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && nx < GRID_WIDTH as i32 && ny >= 0 && ny < GRID_HEIGHT as i32 {
                        let nx = nx as usize;
                        let ny = ny as usize;

                        if buttons.pressed(MouseButton::Left) {
                            // Add heat at cursor position
                            if let Ok(current_temp) = sim_state.grid.get_temperature(nx, ny) {
                                let new_temp = (current_temp + 20.0).min(T_MAX + 100.0);
                                let _ = sim_state.grid.set_temperature(nx, ny, new_temp);
                            }
                        }

                        if buttons.pressed(MouseButton::Right) {
                            // Cool at cursor position
                            if let Ok(current_temp) = sim_state.grid.get_temperature(nx, ny) {
                                let new_temp = (current_temp - 20.0).max(T_MIN - 50.0);
                                let _ = sim_state.grid.set_temperature(nx, ny, new_temp);
                            }
                        }
                    }
                }
            }
        }

        mouse_state.last_pos = Some(world_pos);
    } else {
        mouse_state.last_pos = None;
    }
}

fn simulation_step(mut sim_state: ResMut<SimulationState>) {
    if sim_state.paused {
        return;
    }

    // Run multiple steps per frame for faster simulation
    for _ in 0..10 {
        sim_state.grid.step();
    }
    sim_state.step_count += 10;
}

fn update_visualization(
    sim_state: Res<SimulationState>,
    mut query: Query<(&ThermalCell, &mut Sprite)>,
) {
    for (cell, mut sprite) in query.iter_mut() {
        let temperature = sim_state.grid.get_temperature(cell.x, cell.y).unwrap_or(T_AMBIENT);

        // Map temperature to color (blue -> white -> red)
        let t_norm = ((temperature - T_MIN) / (T_MAX - T_MIN)).clamp(0.0, 1.0) as f32;

        let color = if t_norm < 0.5 {
            // Blue to white (cold to neutral)
            let t2 = t_norm * 2.0;
            Color::srgb(t2, t2, 1.0)
        } else {
            // White to red (neutral to hot)
            let t2 = (t_norm - 0.5) * 2.0;
            Color::srgb(1.0, 1.0 - t2, 1.0 - t2)
        };

        sprite.color = color;
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
            1 => "Hot Center",
            2 => "Hot/Cold Boundaries",
            3 => "Checkerboard",
            _ => "Custom",
        };

        **text = format!(
            "2D Thermal Simulation\n\
             Status: {}\n\
             Scenario: {}\n\
             Grid: {}x{}\n\
             Steps: {}\n\
             Sim Time: {:.3}s\n\n\
             Temperature (K):\n\
             Avg: {:.1}\n\
             Max: {:.1}\n\
             Min: {:.1}\n\n\
             Controls:\n\
             Left click: Add heat\n\
             Right click: Cool\n\
             Space: Pause\n\
             1/2/3: Scenarios\n\
             R: Reset\n\
             Esc: Exit",
            status,
            scenario_name,
            GRID_WIDTH,
            GRID_HEIGHT,
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
            1 => setup_scenario_1(&mut sim_state),
            2 => setup_scenario_2(&mut sim_state),
            3 => setup_scenario_3(&mut sim_state),
            _ => setup_scenario_1(&mut sim_state),
        }
    }

    // Scenario selection
    if keyboard.just_pressed(KeyCode::Digit1) {
        setup_scenario_1(&mut sim_state);
    }
    if keyboard.just_pressed(KeyCode::Digit2) {
        setup_scenario_2(&mut sim_state);
    }
    if keyboard.just_pressed(KeyCode::Digit3) {
        setup_scenario_3(&mut sim_state);
    }

    if keyboard.just_pressed(KeyCode::Escape) {
        exit.send(AppExit::Success);
    }
}
