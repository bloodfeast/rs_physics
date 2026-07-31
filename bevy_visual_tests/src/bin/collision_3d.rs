//! 3D Collision Detection Visual Demo
//!
//! This visualizes the physics world with collision detection
//! between multiple 3D objects (spheres, boxes).
//!
//! Controls:
//! - A/D or Left/Right: Rotate camera horizontally
//! - W/S or Up/Down: Rotate camera vertically
//! - Q/E: Zoom in/out
//! - Space: Pause/resume simulation
//! - R: Reset simulation
//! - 1-4: Load different scenarios
//! - Escape: Exit

use bevy::prelude::*;
use bevy_visual_tests::{
    spawn_orbit_camera, spawn_info_text,
    OrbitCameraPlugin, SimulationInfoText,
};
use rs_physics::world::{spawn_physics_thread, WorldConfig, PhysicsHandle, ObjectId};
use rs_physics::models::{PhysicalObject3D, Shape3D};
use rs_physics::materials::Material;
use rs_physics::utils::PhysicsConstants;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "3D Collision Detection Demo".to_string(),
                resolution: (1280.0, 720.0).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(OrbitCameraPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (
            sync_physics,
            update_info_text,
            handle_input,
        ))
        .run();
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Scenario {
    SphereBounce,
    BoxStack,
    BilliardBreak,
    Avalanche,
}

impl Scenario {
    fn name(&self) -> &'static str {
        match self {
            Scenario::SphereBounce => "Sphere Bounce",
            Scenario::BoxStack => "Box Stack",
            Scenario::BilliardBreak => "Billiard Break",
            Scenario::Avalanche => "Avalanche",
        }
    }
}

#[derive(Resource)]
struct SimulationState {
    physics: Option<PhysicsHandle>,
    paused: bool,
    scenario: Scenario,
    object_count: usize,
}

impl Default for SimulationState {
    fn default() -> Self {
        Self {
            physics: None,
            paused: false,
            scenario: Scenario::SphereBounce,
            object_count: 0,
        }
    }
}

#[derive(Component)]
struct PhysicsEntity {
    id: ObjectId,
    shape: ShapeType,
}

#[derive(Clone, Copy)]
enum ShapeType {
    Sphere(f32),
    Box(f32, f32, f32),
}

#[derive(Component)]
struct Ground;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Spawn camera
    spawn_orbit_camera(&mut commands, Vec3::new(0.0, 20.0, 40.0), Vec3::new(0.0, 5.0, 0.0));

    // Lighting
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(10.0, 20.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 300.0,
    });

    // Ground plane (visual only - collision handled by physics)
    let ground_mesh = meshes.add(Cuboid::new(100.0, 1.0, 100.0));
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.3, 0.5, 0.3),
        ..default()
    });
    commands.spawn((
        Mesh3d(ground_mesh),
        MeshMaterial3d(ground_material),
        Transform::from_xyz(0.0, -0.5, 0.0),
        Ground,
    ));

    // Initialize simulation state (will be set up by load_scenario)
    commands.init_resource::<SimulationState>();

    // UI text
    spawn_info_text(&mut commands);

    // Load default scenario after a frame (so resources are available)
}

fn load_scenario(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    sim_state: &mut SimulationState,
    scenario: Scenario,
    existing_entities: &Query<Entity, With<PhysicsEntity>>,
) {
    // Despawn existing physics entities
    for entity in existing_entities.iter() {
        commands.entity(entity).despawn();
    }

    // Shutdown old physics thread if exists
    if let Some(ref physics) = sim_state.physics {
        let _ = physics.shutdown();
    }

    // Create new physics world
    let config = WorldConfig {
        timestep: 1.0 / 120.0,
        gravity: (0.0, -9.81, 0.0),
        ..Default::default()
    };
    let physics = spawn_physics_thread(config);

    sim_state.scenario = scenario;
    sim_state.object_count = 0;

    // Add ground plane as a very heavy static box
    add_ground_plane(&physics);

    // Create objects based on scenario
    match scenario {
        Scenario::SphereBounce => {
            create_sphere_bounce(commands, meshes, materials, &physics, sim_state);
        }
        Scenario::BoxStack => {
            create_box_stack(commands, meshes, materials, &physics, sim_state);
        }
        Scenario::BilliardBreak => {
            create_billiard_break(commands, meshes, materials, &physics, sim_state);
        }
        Scenario::Avalanche => {
            create_avalanche(commands, meshes, materials, &physics, sim_state);
        }
    }

    sim_state.physics = Some(physics);
}

/// Add a ground plane to the physics world
/// Uses an infinite mass box positioned so top surface is at y=0
fn add_ground_plane(physics: &PhysicsHandle) {
    let ground_thickness = 2.0;
    let ground_size = 200.0;

    // Ground plane - infinite mass makes it static (no gravity, immovable)
    let ground = PhysicalObject3D::new(
        f64::INFINITY,  // Infinite mass = static object
        (0.0, 0.0, 0.0),  // No velocity
        (0.0, -ground_thickness / 2.0, 0.0),  // Position so top surface is at y=0
        Shape3D::Cuboid(ground_size, ground_thickness, ground_size),
        Some(Material::steel()),  // High restitution for bouncing
        (0.0, 0.0, 0.0),  // No angular velocity
        (0.0, 0.0, 0.0),  // No rotation
        PhysicsConstants::default(),
    );

    let _ = physics.add_object(ground);
}

fn create_sphere_bounce(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    physics: &PhysicsHandle,
    sim_state: &mut SimulationState,
) {
    let sphere_mesh = meshes.add(Sphere::new(1.0));
    let colors = [
        Color::srgb(1.0, 0.2, 0.2),
        Color::srgb(0.2, 1.0, 0.2),
        Color::srgb(0.2, 0.2, 1.0),
        Color::srgb(1.0, 1.0, 0.2),
        Color::srgb(1.0, 0.2, 1.0),
    ];

    // Create bouncing spheres at different heights
    for i in 0..5 {
        let x = (i as f64 - 2.0) * 4.0;
        let y = 5.0 + (i as f64) * 3.0;
        let radius = 1.0;

        let obj = PhysicalObject3D::new(
            1.0,
            (0.0, 0.0, 0.0),
            (x, y, 0.0),
            Shape3D::Sphere(radius),
            Some(Material::steel()),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            PhysicsConstants::default(),
        );

        if let Ok(id) = physics.add_object(obj) {
            let material = materials.add(StandardMaterial {
                base_color: colors[i % colors.len()],
                metallic: 0.8,
                perceptual_roughness: 0.3,
                ..default()
            });

            commands.spawn((
                Mesh3d(sphere_mesh.clone()),
                MeshMaterial3d(material),
                Transform::from_xyz(x as f32, y as f32, 0.0),
                PhysicsEntity {
                    id,
                    shape: ShapeType::Sphere(radius as f32),
                },
            ));
            sim_state.object_count += 1;
        }
    }
}

fn create_box_stack(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    physics: &PhysicsHandle,
    sim_state: &mut SimulationState,
) {
    let box_size = 2.0;
    let box_mesh = meshes.add(Cuboid::new(box_size as f32, box_size as f32, box_size as f32));

    // Stack of boxes - small gap to prevent initial overlap detection
    let gap = 0.02; // 2cm gap - small enough to not cause significant fall
    for layer in 0..5 {
        for i in 0..(5 - layer) {
            let x = (i as f64 - (4 - layer) as f64 / 2.0) * (box_size + gap);
            let y = box_size / 2.0 + layer as f64 * (box_size + gap);

            let obj = PhysicalObject3D::new(
                2.0,
                (0.0, 0.0, 0.0),
                (x, y, 0.0),
                Shape3D::Cuboid(box_size, box_size, box_size),
                Some(Material::wood()),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                PhysicsConstants::default(),
            );

            if let Ok(id) = physics.add_object(obj) {
                let hue = (layer as f32 * 60.0 + i as f32 * 30.0) % 360.0;
                let material = materials.add(StandardMaterial {
                    base_color: Color::hsl(hue, 0.7, 0.5),
                    ..default()
                });

                commands.spawn((
                    Mesh3d(box_mesh.clone()),
                    MeshMaterial3d(material),
                    Transform::from_xyz(x as f32, y as f32, 0.0),
                    PhysicsEntity {
                        id,
                        shape: ShapeType::Box(box_size as f32, box_size as f32, box_size as f32),
                    },
                ));
                sim_state.object_count += 1;
            }
        }
    }

    // Add a ball to knock them over
    let ball_radius = 1.5;
    let sphere_mesh = meshes.add(Sphere::new(ball_radius as f32));
    let obj = PhysicalObject3D::new(
        5.0,
        (10.0, 0.0, 0.0),
        (-15.0, 3.0, 0.0),
        Shape3D::Sphere(ball_radius),
        Some(Material::steel()),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        PhysicsConstants::default(),
    );

    if let Ok(id) = physics.add_object(obj) {
        let material = materials.add(StandardMaterial {
            base_color: Color::srgb(0.8, 0.1, 0.1),
            metallic: 0.9,
            perceptual_roughness: 0.1,
            ..default()
        });

        commands.spawn((
            Mesh3d(sphere_mesh),
            MeshMaterial3d(material),
            Transform::from_xyz(-15.0, 3.0, 0.0),
            PhysicsEntity {
                id,
                shape: ShapeType::Sphere(ball_radius as f32),
            },
        ));
        sim_state.object_count += 1;
    }
}

fn create_billiard_break(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    physics: &PhysicsHandle,
    sim_state: &mut SimulationState,
) {
    let ball_radius = 0.8;
    let sphere_mesh = meshes.add(Sphere::new(ball_radius as f32));

    // Triangle rack of balls
    let spacing = ball_radius * 2.1;
    let mut ball_index = 0;

    for row in 0..5 {
        for col in 0..=row {
            let x = row as f64 * spacing * 0.866; // cos(30°)
            let z = (col as f64 - row as f64 / 2.0) * spacing;
            let y = ball_radius;

            let obj = PhysicalObject3D::new(
                0.5,
                (0.0, 0.0, 0.0),
                (x, y, z),
                Shape3D::Sphere(ball_radius),
                Some(Material::polyurethane()),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                PhysicsConstants::default(),
            );

            if let Ok(id) = physics.add_object(obj) {
                let hue = (ball_index as f32 * 25.0) % 360.0;
                let material = materials.add(StandardMaterial {
                    base_color: Color::hsl(hue, 0.9, 0.5),
                    metallic: 0.2,
                    perceptual_roughness: 0.4,
                    ..default()
                });

                commands.spawn((
                    Mesh3d(sphere_mesh.clone()),
                    MeshMaterial3d(material),
                    Transform::from_xyz(x as f32, y as f32, z as f32),
                    PhysicsEntity {
                        id,
                        shape: ShapeType::Sphere(ball_radius as f32),
                    },
                ));
                sim_state.object_count += 1;
                ball_index += 1;
            }
        }
    }

    // Cue ball
    let obj = PhysicalObject3D::new(
        0.5,
        (15.0, 0.0, 0.0),
        (-10.0, ball_radius, 0.0),
        Shape3D::Sphere(ball_radius),
        Some(Material::polyurethane()),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        PhysicsConstants::default(),
    );

    if let Ok(id) = physics.add_object(obj) {
        let material = materials.add(StandardMaterial {
            base_color: Color::WHITE,
            metallic: 0.2,
            perceptual_roughness: 0.4,
            ..default()
        });

        commands.spawn((
            Mesh3d(sphere_mesh),
            MeshMaterial3d(material),
            Transform::from_xyz(-10.0, ball_radius as f32, 0.0),
            PhysicsEntity {
                id,
                shape: ShapeType::Sphere(ball_radius as f32),
            },
        ));
        sim_state.object_count += 1;
    }
}

fn create_avalanche(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    physics: &PhysicsHandle,
    sim_state: &mut SimulationState,
) {
    let sphere_mesh = meshes.add(Sphere::new(0.5));

    // Create many small spheres starting high up
    for i in 0..50 {
        let x = (i % 10) as f64 - 5.0 + (i as f64 * 0.1).sin() * 0.5;
        let y = 15.0 + (i / 10) as f64 * 1.5;
        let z = (i as f64 * 0.3).cos() * 3.0;
        let radius = 0.4 + (i as f64 * 0.01).sin().abs() * 0.2;

        let obj = PhysicalObject3D::new(
            0.5,
            ((i as f64).sin() * 2.0, 0.0, (i as f64).cos() * 2.0),
            (x, y, z),
            Shape3D::Sphere(radius),
            Some(Material::rubber()),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            PhysicsConstants::default(),
        );

        if let Ok(id) = physics.add_object(obj) {
            let hue = (i as f32 * 7.2) % 360.0;
            let material = materials.add(StandardMaterial {
                base_color: Color::hsl(hue, 0.8, 0.5),
                ..default()
            });

            let scaled_mesh = meshes.add(Sphere::new(radius as f32));
            commands.spawn((
                Mesh3d(scaled_mesh),
                MeshMaterial3d(material),
                Transform::from_xyz(x as f32, y as f32, z as f32),
                PhysicsEntity {
                    id,
                    shape: ShapeType::Sphere(radius as f32),
                },
            ));
            sim_state.object_count += 1;
        }
    }
}

fn sync_physics(
    sim_state: Res<SimulationState>,
    mut query: Query<(&PhysicsEntity, &mut Transform)>,
) {
    if let Some(ref physics) = sim_state.physics {
        // Interpolated: this drives transforms, so it must be smooth at our
        // refresh rate rather than snapping to the physics tick rate.
        let state = physics.get_interpolated_state();

        // Debug: log number of objects in physics world every 120 ticks
        if state.tick % 120 == 0 && state.tick > 0 {
            info!("Physics tick {}: {} objects in world", state.tick, state.objects.len());
            // Log first object (ground) position
            if let Some(first) = state.objects.first() {
                info!("  First object (ground?) at y={:.2}", first.position.1);
            }
            // Log second object position if it exists
            if state.objects.len() > 1 {
                if let Some(second) = state.objects.get(1) {
                    info!("  Second object at y={:.2}, vy={:.2}", second.position.1, second.velocity.1);
                }
            }
        }

        for (entity, mut transform) in query.iter_mut() {
            if let Some(obj_state) = state.get_object(entity.id) {
                transform.translation = Vec3::new(
                    obj_state.position.0 as f32,
                    obj_state.position.1 as f32,
                    obj_state.position.2 as f32,
                );

                // Convert quaternion orientation
                let (x, y, z, w) = obj_state.orientation;
                transform.rotation = Quat::from_xyzw(x as f32, y as f32, z as f32, w as f32);
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
        let tick = sim_state.physics.as_ref()
            .map(|p| p.get_latest_state().tick)
            .unwrap_or(0);

        **text = format!(
            "3D Collision Demo\n\
             Scenario: {}\n\
             Status: {}\n\
             Objects: {}\n\
             Physics Tick: {}\n\n\
             Controls:\n\
             WASD: Camera | Q/E: Zoom\n\
             Space: Pause\n\
             1: Sphere Bounce\n\
             2: Box Stack\n\
             3: Billiard Break\n\
             4: Avalanche\n\
             R: Reset | Esc: Exit",
            sim_state.scenario.name(),
            status,
            sim_state.object_count,
            tick,
        );
    }
}

fn handle_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut sim_state: ResMut<SimulationState>,
    existing_entities: Query<Entity, With<PhysicsEntity>>,
    mut exit: EventWriter<AppExit>,
) {
    // Initialize on first frame if not done
    if sim_state.physics.is_none() {
        load_scenario(
            &mut commands,
            &mut meshes,
            &mut materials,
            &mut sim_state,
            Scenario::SphereBounce,
            &existing_entities,
        );
        return;
    }

    if keyboard.just_pressed(KeyCode::Space) {
        sim_state.paused = !sim_state.paused;
        if let Some(ref physics) = sim_state.physics {
            if sim_state.paused {
                let _ = physics.pause();
            } else {
                let _ = physics.resume();
            }
        }
    }

    let mut new_scenario = None;

    if keyboard.just_pressed(KeyCode::Digit1) {
        new_scenario = Some(Scenario::SphereBounce);
    }
    if keyboard.just_pressed(KeyCode::Digit2) {
        new_scenario = Some(Scenario::BoxStack);
    }
    if keyboard.just_pressed(KeyCode::Digit3) {
        new_scenario = Some(Scenario::BilliardBreak);
    }
    if keyboard.just_pressed(KeyCode::Digit4) {
        new_scenario = Some(Scenario::Avalanche);
    }
    if keyboard.just_pressed(KeyCode::KeyR) {
        new_scenario = Some(sim_state.scenario);
    }

    if let Some(scenario) = new_scenario {
        load_scenario(
            &mut commands,
            &mut meshes,
            &mut materials,
            &mut sim_state,
            scenario,
            &existing_entities,
        );
    }

    if keyboard.just_pressed(KeyCode::Escape) {
        if let Some(ref physics) = sim_state.physics {
            let _ = physics.shutdown();
        }
        exit.send(AppExit::Success);
    }
}
