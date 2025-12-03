//! 2D Fluid Simulation Visualization
//!
//! This visualizes the 2D Eulerian fluid simulation using the FluidGrid.
//! Density is shown as brightness, and velocity is shown as color hue.
//!
//! Controls:
//! - Left mouse: Add density
//! - Right mouse: Add velocity (drag)
//! - Space: Pause/resume simulation
//! - R: Reset simulation
//! - 1/2/3: Change viscosity (low/medium/high)
//! - Escape: Exit

use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy_visual_tests::{spawn_info_text, SimulationInfoText};
use rs_physics::fluid_dynamics::{FluidGrid, SolverConfig};

const GRID_WIDTH: usize = 80;
const GRID_HEIGHT: usize = 60;
const CELL_SIZE: f32 = 10.0;
const DT: f64 = 0.016;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "2D Fluid Simulation".to_string(),
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
    grid: FluidGrid,
    paused: bool,
    step_count: u64,
    viscosity: f64,
    diffusion: f64,
}

impl Default for SimulationState {
    fn default() -> Self {
        let solver_config = SolverConfig::high_quality();
        // API: FluidGrid::with_solver(width, height, diffusion, viscosity, dt, solver_config)
        let grid = FluidGrid::with_solver(GRID_WIDTH, GRID_HEIGHT, 0.0001, 0.1, DT, solver_config)
            .expect("Failed to create fluid grid");
        Self {
            grid,
            paused: false,
            step_count: 0,
            viscosity: 0.1,
            diffusion: 0.0001,
        }
    }
}

#[derive(Resource, Default)]
struct MouseState {
    last_pos: Option<Vec2>,
}

#[derive(Component)]
struct FluidCell {
    x: usize,
    y: usize,
}

#[derive(Component)]
struct MainCamera;

fn setup(mut commands: Commands) {
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
                FluidCell { x, y },
            ));
        }
    }

    // UI text
    spawn_info_text(&mut commands);
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
        if grid_x >= 1
            && grid_x < (GRID_WIDTH as i32 - 1)
            && grid_y >= 1
            && grid_y < (GRID_HEIGHT as i32 - 1)
        {
            let x = grid_x as usize;
            let y = grid_y as usize;

            if buttons.pressed(MouseButton::Left) {
                // Add density at cursor position
                let _ = sim_state.grid.add_density(x, y, 50.0);
            }

            if buttons.pressed(MouseButton::Right) {
                // Add velocity based on mouse movement
                if let Some(last_pos) = mouse_state.last_pos {
                    let delta = world_pos - last_pos;
                    let vx = (delta.x * 5.0) as f64;
                    let vy = (delta.y * 5.0) as f64;
                    let _ = sim_state.grid.add_velocity(x, y, vx, vy);
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

    sim_state.grid.step();
    sim_state.step_count += 1;
}

fn update_visualization(
    sim_state: Res<SimulationState>,
    mut query: Query<(&FluidCell, &mut Sprite)>,
) {
    for (cell, mut sprite) in query.iter_mut() {
        // Get density and velocity at this cell
        let density = sim_state.grid.get_density(cell.x, cell.y).unwrap_or(0.0);
        let (vx, vy) = sim_state.grid.get_velocity(cell.x, cell.y).unwrap_or((0.0, 0.0));

        // Calculate velocity magnitude for color
        let vel_mag = (vx * vx + vy * vy).sqrt();

        // Map density to brightness (clamped)
        let brightness = (density / 100.0).min(1.0) as f32;

        // Map velocity direction to hue
        let hue = if vel_mag > 0.01 {
            let angle = vy.atan2(vx);
            ((angle + std::f64::consts::PI) / (2.0 * std::f64::consts::PI) * 360.0) as f32
        } else {
            200.0 // Default blue when stationary
        };

        // Map velocity magnitude to saturation
        let saturation = (vel_mag / 10.0).min(1.0) as f32;

        // Create color: HSL with velocity-based hue and density-based lightness
        let color = Color::hsl(hue, saturation.max(0.3), brightness * 0.8 + 0.1);
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
        let kinetic_energy = sim_state.grid.get_kinetic_energy();
        let total_mass = sim_state.grid.get_total_mass();

        **text = format!(
            "2D Fluid Simulation\n\
             Status: {}\n\
             Grid: {}x{}\n\
             Steps: {}\n\
             Sim Time: {:.2}s\n\
             Viscosity: {:.4}\n\
             Kinetic Energy: {:.2}\n\
             Total Mass: {:.2}\n\n\
             Controls:\n\
             Left click: Add density\n\
             Right drag: Add velocity\n\
             Space: Pause\n\
             1/2/3: Viscosity\n\
             R: Reset\n\
             Esc: Exit",
            status,
            GRID_WIDTH,
            GRID_HEIGHT,
            sim_state.step_count,
            sim_time,
            sim_state.viscosity,
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
        // Reset simulation
        recreate_grid(&mut sim_state);
    }

    // Viscosity presets
    if keyboard.just_pressed(KeyCode::Digit1) {
        sim_state.viscosity = 0.001; // Low (like water)
        recreate_grid(&mut sim_state);
    }
    if keyboard.just_pressed(KeyCode::Digit2) {
        sim_state.viscosity = 0.1; // Medium
        recreate_grid(&mut sim_state);
    }
    if keyboard.just_pressed(KeyCode::Digit3) {
        sim_state.viscosity = 1.0; // High (like honey)
        recreate_grid(&mut sim_state);
    }

    if keyboard.just_pressed(KeyCode::Escape) {
        exit.send(AppExit::Success);
    }
}

fn recreate_grid(sim_state: &mut SimulationState) {
    let solver_config = SolverConfig::high_quality();
    // API: FluidGrid::with_solver(width, height, diffusion, viscosity, dt, solver_config)
    sim_state.grid = FluidGrid::with_solver(
        GRID_WIDTH,
        GRID_HEIGHT,
        sim_state.diffusion,
        sim_state.viscosity,
        DT,
        solver_config,
    )
    .expect("Failed to create fluid grid");
    sim_state.step_count = 0;
}
