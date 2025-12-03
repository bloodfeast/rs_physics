//! Common utilities for Bevy visual tests
//!
//! This crate provides visual demonstrations and tests for rs_physics
//! using the Bevy game engine for rendering.

use bevy::prelude::*;

/// Camera controller for orbiting around the scene
#[derive(Component)]
pub struct OrbitCamera {
    pub focus: Vec3,
    pub radius: f32,
    pub upside_down: bool,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            focus: Vec3::ZERO,
            radius: 50.0,
            upside_down: false,
        }
    }
}

/// Plugin for orbit camera controls
pub struct OrbitCameraPlugin;

impl Plugin for OrbitCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, orbit_camera_system);
    }
}

fn orbit_camera_system(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut query: Query<(&mut Transform, &mut OrbitCamera)>,
) {
    for (mut transform, mut orbit) in query.iter_mut() {
        let mut rotation = 0.0f32;
        let mut zoom = 0.0f32;
        let mut vertical = 0.0f32;

        if keyboard.pressed(KeyCode::KeyA) || keyboard.pressed(KeyCode::ArrowLeft) {
            rotation += 1.0;
        }
        if keyboard.pressed(KeyCode::KeyD) || keyboard.pressed(KeyCode::ArrowRight) {
            rotation -= 1.0;
        }
        if keyboard.pressed(KeyCode::KeyW) || keyboard.pressed(KeyCode::ArrowUp) {
            vertical += 1.0;
        }
        if keyboard.pressed(KeyCode::KeyS) || keyboard.pressed(KeyCode::ArrowDown) {
            vertical -= 1.0;
        }
        if keyboard.pressed(KeyCode::KeyQ) {
            zoom -= 1.0;
        }
        if keyboard.pressed(KeyCode::KeyE) {
            zoom += 1.0;
        }

        // Apply rotation
        let delta = time.delta_secs();
        let rot_speed = 2.0;
        let zoom_speed = 50.0;

        if rotation != 0.0 {
            let angle = rotation * rot_speed * delta;
            let rotation_quat = Quat::from_rotation_y(angle);
            let offset = transform.translation - orbit.focus;
            let new_offset = rotation_quat * offset;
            transform.translation = orbit.focus + new_offset;
        }

        if vertical != 0.0 {
            let angle = vertical * rot_speed * delta;
            let right = transform.right();
            let rotation_quat = Quat::from_axis_angle(*right, angle);
            let offset = transform.translation - orbit.focus;
            let new_offset = rotation_quat * offset;

            // Prevent flipping over the poles
            let up_dot = new_offset.normalize().dot(Vec3::Y);
            if up_dot.abs() < 0.99 {
                transform.translation = orbit.focus + new_offset;
            }
        }

        if zoom != 0.0 {
            orbit.radius = (orbit.radius + zoom * zoom_speed * delta).max(5.0);
            let direction = (transform.translation - orbit.focus).normalize();
            transform.translation = orbit.focus + direction * orbit.radius;
        }

        // Always look at focus
        transform.look_at(orbit.focus, Vec3::Y);
    }
}

/// Spawns a basic 3D camera with orbit controls
pub fn spawn_orbit_camera(commands: &mut Commands, position: Vec3, focus: Vec3) {
    let radius = (position - focus).length();
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(position).looking_at(focus, Vec3::Y),
        OrbitCamera {
            focus,
            radius,
            upside_down: false,
        },
    ));
}

/// UI text showing simulation info
#[derive(Component)]
pub struct SimulationInfoText;

/// Spawns UI text for simulation info
pub fn spawn_info_text(commands: &mut Commands) {
    commands.spawn((
        Text::new("Simulation Info"),
        TextFont {
            font_size: 20.0,
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
        SimulationInfoText,
    ));
}

/// Helper to create a colored material for particles
pub fn particle_material(
    materials: &mut Assets<StandardMaterial>,
    color: Color,
) -> Handle<StandardMaterial> {
    materials.add(StandardMaterial {
        base_color: color,
        emissive: LinearRgba::from(color) * 2.0,
        ..default()
    })
}
