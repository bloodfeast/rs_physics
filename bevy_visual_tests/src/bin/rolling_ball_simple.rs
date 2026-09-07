//! Ball Playground - A playable physics demo using Real-Time Background Threading
//!
//! Control a bouncy rubber ball through an obstacle course featuring:
//! - Rolling physics with momentum (handled by background physics thread)
//! - Bouncy rubber ball (high restitution)
//! - Jump mechanic
//! - Rope bridges to cross
//! - Trap doors that swing open
//! - Bouncy trampolines
//! - Water pools with drag
//!
//! Controls:
//! - WASD or Arrow Keys: Roll the ball
//! - Space: Jump (when grounded)
//! - R: Reset ball position
//! - Escape: Exit
//!
//! Architecture:
//! - Physics runs in a background thread in REAL-TIME mode (as fast as possible)
//! - Render thread just reads the latest state each frame - always fresh
//! - No interpolation needed - physics is always "current enough"

use bevy::prelude::*;
use bevy_visual_tests::particle_material;
use rs_physics::constraints::{Hinge3D, RopeChain3D};
use rs_physics::materials::Material;
use rs_physics::models::{ObjectIn3D, PhysicalObject3D, Shape3D};
use rs_physics::utils::PhysicsConstants;
use rs_physics::world::{
    ObjectId, PhysicsHandle, WorldConfig, ForceId,
    ConstraintId, WorldConstraint, ConstraintState, WorldState,
    spawn_physics_thread,
};

const DT: f64 = 1.0 / 240.0;  // ~4ms timestep (240 Hz) - high enough for smooth physics
const BALL_RADIUS: f64 = 0.5;
const ROLL_FORCE: f64 = 100.0;  // Force applied for rolling (reduced from 500)
const JUMP_IMPULSE: f64 = 8.0;  // Impulse for jumping
const MAX_VELOCITY: f64 = 15.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Ball Playground - Physics Demo (Real-Time Background Thread)".to_string(),
                resolution: (1280.0, 720.0).into(),
                ..default()
            }),
            ..default()
        }))
        .init_resource::<CachedPhysicsState>()
        .add_systems(Startup, setup)
        // Fetch latest physics state at start of each frame
        .add_systems(First, fetch_physics_state)
        // Input and game logic in Update
        .add_systems(Update, (
            player_input,
            check_water_zones,
            update_kinematic_planks,
            camera_follow,
            update_ui,
        ))
        // Sync visuals directly from cached physics state (no interpolation needed)
        .add_systems(Update, (
            sync_player_transform,
            sync_rope_visuals,
            sync_plank_visuals,
            sync_hinge_visuals,
        ))
        .run();
}

// ============================================================================
// Components
// ============================================================================

#[derive(Component)]
struct Player {
    object_id: ObjectId,
    grounded: bool,
    jump_cooldown: f32,
}

#[derive(Component)]
struct Ground;

#[derive(Component)]
struct StaticPlatform {
    object_id: ObjectId,
}

#[derive(Component)]
struct Trampoline {
    object_id: ObjectId,
}

#[derive(Component)]
struct WaterZone {
    bounds_min: Vec3,
    bounds_max: Vec3,
}

#[derive(Component)]
struct RopeBridgeVisual {
    rope_index: usize,
    particle_index: usize,
}

#[derive(Component)]
struct BridgePlank {
    particle_index: usize,
    object_id: ObjectId,
}

#[derive(Component)]
struct HingeDoorVisual {
    hinge_index: usize,
    door_object_id: ObjectId,  // Collision body for the door
}

#[derive(Component)]
struct MainCamera;

#[derive(Component)]
struct UIText;

// ============================================================================
// Resources
// ============================================================================

#[derive(Resource)]
struct GameState {
    spawn_point: Vec3,
    physics: PhysicsHandle,
    player_id: Option<ObjectId>,
    rope_constraint_ids: Vec<ConstraintId>,
    hinge_constraint_ids: Vec<ConstraintId>,
    rope_particle_counts: Vec<usize>,
    score: u32,
    // Track if player is in water to avoid spamming buoyancy commands
    in_water: bool,
    water_buoyancy_id: Option<ForceId>,
}

// GameState is not Default because PhysicsHandle requires setup
// We'll create it in the setup system

/// Cached physics state - fetched once per frame from the background thread
///
/// In real-time mode, physics runs continuously in a background thread.
/// We just read the latest state each frame - no interpolation needed
/// because physics is always "current enough".
#[derive(Resource, Default)]
struct CachedPhysicsState {
    /// Latest physics state from the background thread
    state: WorldState,
}

/// Fetch the latest physics state from the background thread (runs in First schedule)
fn fetch_physics_state(
    game_state: Option<Res<GameState>>,
    mut cached: ResMut<CachedPhysicsState>,
) {
    if let Some(gs) = game_state {
        cached.state = gs.physics.get_latest_state();
    }
}

// ============================================================================
// Setup
// ============================================================================

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // === Create Physics World (Background Thread at 240Hz) ===
    // Use background threading with high update rate for smooth physics
    let config = WorldConfig::default()
        .with_frequency(240.0)           // 240 Hz physics - smooth enough without being wasteful
        .with_gravity(0.0, -15.0, 0.0)   // Slightly stronger gravity for gameplay
        .with_real_time(true);           // Broadcast every step for freshest state

    let physics = spawn_physics_thread(config);

    // === Materials ===
    let rubber_material = particle_material(&mut materials, Color::srgb(0.9, 0.2, 0.3));
    let stripe_material = particle_material(&mut materials, Color::srgb(1.0, 1.0, 1.0));
    let spot_material = particle_material(&mut materials, Color::srgb(0.2, 0.2, 0.8));
    let ground_material = particle_material(&mut materials, Color::srgb(0.3, 0.5, 0.3));
    let platform_material = particle_material(&mut materials, Color::srgb(0.4, 0.4, 0.5));
    let trampoline_material = particle_material(&mut materials, Color::srgb(0.9, 0.6, 0.1));
    let water_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.2, 0.4, 0.8, 0.6),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    let rope_material = particle_material(&mut materials, Color::srgb(0.6, 0.4, 0.2));
    let door_material = particle_material(&mut materials, Color::srgb(0.5, 0.3, 0.2));
    let anchor_material = particle_material(&mut materials, Color::srgb(0.3, 0.3, 0.3));

    let spawn_point = Vec3::new(0.0, 3.0, 0.0);

    // === Create Player Ball in PhysicsWorld ===
    let rubber = Material::rubber();
    let player_obj = PhysicalObject3D::new(
        2.0,  // mass
        (0.0, 0.0, 0.0),  // velocity
        (spawn_point.x as f64, spawn_point.y as f64, spawn_point.z as f64),
        Shape3D::Sphere(BALL_RADIUS),
        Some(rubber.clone()),
        (0.0, 0.0, 0.0),  // angular velocity
        (0.0, 0.0, 0.0),  // orientation
        PhysicsConstants::default(),
    );

    let player_id = physics.add_object(player_obj).expect("Failed to add player object");

    // Add continuous drag to the player (air resistance)
    let _ = physics.add_drag(player_id, 0.3);

    // === Player Ball Visual ===
    let ball_mesh = meshes.add(Sphere::new(BALL_RADIUS as f32));
    let stripe_mesh = meshes.add(Torus::new(BALL_RADIUS as f32 * 0.95, BALL_RADIUS as f32 * 0.05));
    let spot_mesh = meshes.add(Sphere::new(BALL_RADIUS as f32 * 0.15));

    commands.spawn((
        Mesh3d(ball_mesh),
        MeshMaterial3d(rubber_material),
        Transform::from_translation(spawn_point),
        Player {
            object_id: player_id,
            grounded: false,
            jump_cooldown: 0.0,
        },
    ))
    .with_children(|parent| {
        // Equator stripe
        parent.spawn((
            Mesh3d(stripe_mesh.clone()),
            MeshMaterial3d(stripe_material.clone()),
            Transform::IDENTITY,
        ));
        // Vertical stripe
        parent.spawn((
            Mesh3d(stripe_mesh.clone()),
            MeshMaterial3d(stripe_material.clone()),
            Transform::from_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
        ));
        // Top spot
        parent.spawn((
            Mesh3d(spot_mesh.clone()),
            MeshMaterial3d(spot_material.clone()),
            Transform::from_xyz(0.0, BALL_RADIUS as f32 * 0.9, 0.0),
        ));
        // Bottom spot
        parent.spawn((
            Mesh3d(spot_mesh.clone()),
            MeshMaterial3d(spot_material.clone()),
            Transform::from_xyz(0.0, -(BALL_RADIUS as f32 * 0.9), 0.0),
        ));
        // Front spot
        parent.spawn((
            Mesh3d(spot_mesh.clone()),
            MeshMaterial3d(stripe_material.clone()),
            Transform::from_xyz(0.0, 0.0, BALL_RADIUS as f32 * 0.9),
        ));
        // Back spot
        parent.spawn((
            Mesh3d(spot_mesh.clone()),
            MeshMaterial3d(stripe_material.clone()),
            Transform::from_xyz(0.0, 0.0, -(BALL_RADIUS as f32 * 0.9)),
        ));
        // Left spot
        parent.spawn((
            Mesh3d(spot_mesh.clone()),
            MeshMaterial3d(spot_material.clone()),
            Transform::from_xyz(-(BALL_RADIUS as f32 * 0.9), 0.0, 0.0),
        ));
        // Right spot
        parent.spawn((
            Mesh3d(spot_mesh.clone()),
            MeshMaterial3d(spot_material.clone()),
            Transform::from_xyz(BALL_RADIUS as f32 * 0.9, 0.0, 0.0),
        ));
    });

    // === Main Ground (static object in PhysicsWorld) ===
    // Use concrete material for realistic friction (0.6) instead of default (0.5)
    let concrete = Material::concrete();
    let ground_obj = PhysicalObject3D::new(
        f64::INFINITY,  // mass (infinite = static)
        (0.0, 0.0, 0.0),
        (0.0, -0.5, 0.0),
        Shape3D::Cuboid(60.0, 1.0, 60.0),
        Some(concrete.clone()),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        PhysicsConstants::default(),
    );
    let _ = physics.add_object(ground_obj);

    let ground_mesh = meshes.add(Cuboid::new(60.0, 1.0, 60.0));
    commands.spawn((
        Mesh3d(ground_mesh),
        MeshMaterial3d(ground_material.clone()),
        Transform::from_xyz(0.0, -0.5, 0.0),
        Ground,
    ));

    // === Platforms (static objects in PhysicsWorld) ===
    let platform_mesh = meshes.add(Cuboid::new(6.0, 0.5, 6.0));

    // Platform 1: Starting area
    let plat1_obj = PhysicalObject3D::new(
        f64::INFINITY,
        (0.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        Shape3D::Cuboid(6.0, 0.5, 6.0),
        Some(concrete.clone()),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        PhysicsConstants::default(),
    );
    let plat1_id = physics.add_object(plat1_obj).unwrap();
    commands.spawn((
        Mesh3d(platform_mesh.clone()),
        MeshMaterial3d(platform_material.clone()),
        Transform::from_xyz(0.0, 2.0, 0.0),
        StaticPlatform { object_id: plat1_id },
    ));

    // Platform 2: After rope bridge
    let plat2_obj = PhysicalObject3D::new(
        f64::INFINITY,
        (0.0, 0.0, 0.0),
        (15.0, 2.0, 0.0),
        Shape3D::Cuboid(6.0, 0.5, 6.0),
        Some(concrete.clone()),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        PhysicsConstants::default(),
    );
    let plat2_id = physics.add_object(plat2_obj).unwrap();
    commands.spawn((
        Mesh3d(platform_mesh.clone()),
        MeshMaterial3d(platform_material.clone()),
        Transform::from_xyz(15.0, 2.0, 0.0),
        StaticPlatform { object_id: plat2_id },
    ));

    // Platform 3: Trampoline target
    let plat3_obj = PhysicalObject3D::new(
        f64::INFINITY,
        (0.0, 0.0, 0.0),
        (15.0, 6.0, -10.0),
        Shape3D::Cuboid(6.0, 0.5, 6.0),
        Some(concrete.clone()),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        PhysicsConstants::default(),
    );
    let plat3_id = physics.add_object(plat3_obj).unwrap();
    commands.spawn((
        Mesh3d(platform_mesh.clone()),
        MeshMaterial3d(platform_material.clone()),
        Transform::from_xyz(15.0, 6.0, -10.0),
        StaticPlatform { object_id: plat3_id },
    ));

    // === Trampolines (special bounce zones - checked manually for extra bounce) ===
    let trampoline_mesh = meshes.add(Cuboid::new(4.0, 0.3, 4.0));
    // Create bouncy material for trampolines
    let bouncy = Material { restitution_coefficient: 2.0, ..Material::default() };

    // Trampoline 1
    let tramp1_obj = PhysicalObject3D::new(
        f64::INFINITY,
        (0.0, 0.0, 0.0),
        (-8.0, 0.15, 0.0),
        Shape3D::Cuboid(4.0, 0.3, 4.0),
        Some(bouncy.clone()),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        PhysicsConstants::default(),
    );
    let tramp1_id = physics.add_object(tramp1_obj).unwrap();
    commands.spawn((
        Mesh3d(trampoline_mesh.clone()),
        MeshMaterial3d(trampoline_material.clone()),
        Transform::from_xyz(-8.0, 0.15, 0.0),
        Trampoline { object_id: tramp1_id },
    ));

    // Trampoline 2
    let extra_bouncy = Material { restitution_coefficient: 2.5, ..Material::default() };
    let tramp2_obj = PhysicalObject3D::new(
        f64::INFINITY,
        (0.0, 0.0, 0.0),
        (15.0, 0.15, -5.0),
        Shape3D::Cuboid(4.0, 0.3, 4.0),
        Some(extra_bouncy),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        PhysicsConstants::default(),
    );
    let tramp2_id = physics.add_object(tramp2_obj).unwrap();
    commands.spawn((
        Mesh3d(trampoline_mesh.clone()),
        MeshMaterial3d(trampoline_material.clone()),
        Transform::from_xyz(15.0, 0.15, -5.0),
        Trampoline { object_id: tramp2_id },
    ));

    // === Water Pool (visual only - we'll add buoyancy force when player enters) ===
    let water_mesh = meshes.add(Cuboid::new(8.0, 1.5, 8.0));
    commands.spawn((
        Mesh3d(water_mesh),
        MeshMaterial3d(water_material),
        Transform::from_xyz(-8.0, 0.25, -10.0),
        WaterZone {
            bounds_min: Vec3::new(-12.0, -0.5, -14.0),
            bounds_max: Vec3::new(-4.0, 1.0, -6.0),
        },
    ));

    // === Rope Bridge ===
    let bridge_start_x = 3.0_f64;
    let bridge_end_x = 12.0_f64;
    let bridge_y = 2.5_f64;
    let bridge_width = 1.2_f64;
    let num_segments = 16;

    // Create LEFT rope
    let mut left_rope_points: Vec<(f64, f64, f64)> = Vec::new();
    for i in 0..=num_segments {
        let t = i as f64 / num_segments as f64;
        let x = bridge_start_x + (bridge_end_x - bridge_start_x) * t;
        let y = bridge_y - (t * (1.0 - t) * 1.5);
        let z = -bridge_width / 2.0;
        left_rope_points.push((x, y, z));
    }

    // Create RIGHT rope
    let mut right_rope_points: Vec<(f64, f64, f64)> = Vec::new();
    for i in 0..=num_segments {
        let t = i as f64 / num_segments as f64;
        let x = bridge_start_x + (bridge_end_x - bridge_start_x) * t;
        let y = bridge_y - (t * (1.0 - t) * 1.5);
        let z = bridge_width / 2.0;
        right_rope_points.push((x, y, z));
    }

    // Create rope chains
    let mut left_rope = RopeChain3D::from_points(&left_rope_points, 0.3, true).unwrap();
    let mut right_rope = RopeChain3D::from_points(&right_rope_points, 0.3, true).unwrap();

    // Anchor both ends
    left_rope.particles[0].mass = f64::INFINITY;
    left_rope.particles[num_segments].mass = f64::INFINITY;
    right_rope.particles[0].mass = f64::INFINITY;
    right_rope.particles[num_segments].mass = f64::INFINITY;

    // Add ropes to PhysicsWorld as constraints (via command API)
    let left_rope_id = physics.add_constraint(WorldConstraint::RopeChain(left_rope)).unwrap();
    let right_rope_id = physics.add_constraint(WorldConstraint::RopeChain(right_rope)).unwrap();

    let mut rope_constraint_ids = Vec::new();
    let mut rope_particle_counts = Vec::new();
    rope_constraint_ids.push(left_rope_id);
    rope_constraint_ids.push(right_rope_id);
    rope_particle_counts.push(num_segments + 1);
    rope_particle_counts.push(num_segments + 1);

    // Spawn rope segment visuals
    let rope_segment_mesh = meshes.add(Sphere::new(0.08));

    // Left rope visuals
    for i in 0..=num_segments {
        commands.spawn((
            Mesh3d(rope_segment_mesh.clone()),
            MeshMaterial3d(rope_material.clone()),
            Transform::from_xyz(
                left_rope_points[i].0 as f32,
                left_rope_points[i].1 as f32,
                left_rope_points[i].2 as f32,
            ),
            RopeBridgeVisual { rope_index: 0, particle_index: i },
        ));
    }

    // Right rope visuals
    for i in 0..=num_segments {
        commands.spawn((
            Mesh3d(rope_segment_mesh.clone()),
            MeshMaterial3d(rope_material.clone()),
            Transform::from_xyz(
                right_rope_points[i].0 as f32,
                right_rope_points[i].1 as f32,
                right_rope_points[i].2 as f32,
            ),
            RopeBridgeVisual { rope_index: 1, particle_index: i },
        ));
    }

    // Spawn planks as kinematic physics objects
    let plank_material = particle_material(&mut materials, Color::srgb(0.55, 0.35, 0.2));
    let plank_mesh = meshes.add(Cuboid::new(0.15, 0.08, bridge_width as f32 + 0.2));
    let plank_width = 0.15;
    let plank_height = 0.08;
    let plank_depth = bridge_width + 0.2;

    for i in 0..=num_segments {
        let mid_x = (left_rope_points[i].0 + right_rope_points[i].0) / 2.0;
        let mid_y = (left_rope_points[i].1 + right_rope_points[i].1) / 2.0;
        let mid_z = (left_rope_points[i].2 + right_rope_points[i].2) / 2.0;

        // Create physics object for this plank (kinematic - infinite mass, externally positioned)
        let plank_obj = PhysicalObject3D::new(
            f64::INFINITY,  // Kinematic object
            (0.0, 0.0, 0.0),
            (mid_x, mid_y, mid_z),
            Shape3D::Cuboid(plank_width, plank_height, plank_depth),
            None,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            PhysicsConstants::default(),
        );
        let plank_id = physics.add_object(plank_obj).unwrap();

        commands.spawn((
            Mesh3d(plank_mesh.clone()),
            MeshMaterial3d(plank_material.clone()),
            Transform::from_xyz(mid_x as f32, mid_y as f32, mid_z as f32),
            BridgePlank { particle_index: i, object_id: plank_id },
        ));
    }

    // Anchor posts
    let post_mesh = meshes.add(Cylinder::new(0.15, 1.5));
    commands.spawn((
        Mesh3d(post_mesh.clone()),
        MeshMaterial3d(anchor_material.clone()),
        Transform::from_xyz(bridge_start_x as f32, bridge_y as f32 - 0.25, -(bridge_width as f32) / 2.0),
    ));
    commands.spawn((
        Mesh3d(post_mesh.clone()),
        MeshMaterial3d(anchor_material.clone()),
        Transform::from_xyz(bridge_end_x as f32, bridge_y as f32 - 0.25, -(bridge_width as f32) / 2.0),
    ));
    commands.spawn((
        Mesh3d(post_mesh.clone()),
        MeshMaterial3d(anchor_material.clone()),
        Transform::from_xyz(bridge_start_x as f32, bridge_y as f32 - 0.25, (bridge_width as f32) / 2.0),
    ));
    commands.spawn((
        Mesh3d(post_mesh.clone()),
        MeshMaterial3d(anchor_material.clone()),
        Transform::from_xyz(bridge_end_x as f32, bridge_y as f32 - 0.25, (bridge_width as f32) / 2.0),
    ));

    // === Trap Door (Hinge) ===
    let hinge_pos = (20.0, 0.5, 0.0);
    let door_offset = 2.0_f64;

    let frame = ObjectIn3D::new(
        f64::INFINITY, 0.0, 0.0, 0.0,
        (hinge_pos.0 as f64, hinge_pos.1 as f64, hinge_pos.2 as f64),
    );
    let door = ObjectIn3D::new(
        3.0, 0.0, 0.0, 0.0,
        (hinge_pos.0 as f64, hinge_pos.1 as f64, hinge_pos.2 as f64 + door_offset),
    );

    let steel = Material::steel();
    let hinge = Hinge3D::new(
        frame,
        door,
        (hinge_pos.0 as f64, hinge_pos.1 as f64, hinge_pos.2 as f64),
        (1.0, 0.0, 0.0),
    ).unwrap()
        .with_limits(-0.1, std::f64::consts::FRAC_PI_2)
        .with_material(&steel)
        .with_angular_damping(0.05);

    let hinge_id = physics.add_constraint(WorldConstraint::Hinge(hinge)).unwrap();
    let mut hinge_constraint_ids = Vec::new();
    hinge_constraint_ids.push(hinge_id);

    // Door collision body (kinematic - we'll update position based on hinge angle)
    let door_collision = PhysicalObject3D::new(
        f64::INFINITY,  // Kinematic object
        (0.0, 0.0, 0.0),
        (hinge_pos.0 as f64, hinge_pos.1 as f64, hinge_pos.2 as f64 + door_offset),
        Shape3D::Cuboid(4.0, 0.2, 4.0),  // Same size as visual
        None,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        PhysicsConstants::default(),
    );
    let door_object_id = physics.add_object(door_collision).unwrap();

    // Door visual
    let door_mesh = meshes.add(Cuboid::new(4.0, 0.2, 4.0));
    commands.spawn((
        Mesh3d(door_mesh),
        MeshMaterial3d(door_material),
        Transform::from_xyz(hinge_pos.0, hinge_pos.1, hinge_pos.2 + 2.0),
        HingeDoorVisual { hinge_index: 0, door_object_id },
    ));

    // Hinge anchor
    commands.spawn((
        Mesh3d(post_mesh.clone()),
        MeshMaterial3d(anchor_material.clone()),
        Transform::from_xyz(hinge_pos.0, hinge_pos.1, hinge_pos.2)
            .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
    ));

    // === Ramps ===
    let ramp_mesh = meshes.add(Cuboid::new(4.0, 0.3, 6.0));
    commands.spawn((
        Mesh3d(ramp_mesh.clone()),
        MeshMaterial3d(platform_material.clone()),
        Transform::from_xyz(-5.0, 1.0, 0.0)
            .with_rotation(Quat::from_rotation_z(0.3)),
    ));

    // === Camera ===
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 10.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
        MainCamera,
    ));

    // === Lighting ===
    commands.spawn((
        DirectionalLight {
            illuminance: 15000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(10.0, 20.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 500.0,
    });

    // === UI ===
    commands.spawn((
        Text::new("Ball Playground (Background Thread)\nWASD: Move | Space: Jump | R: Reset"),
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
        UIText,
    ));

    // === Insert GameState resource ===
    commands.insert_resource(GameState {
        spawn_point,
        physics,
        player_id: Some(player_id),
        rope_constraint_ids,
        hinge_constraint_ids,
        rope_particle_counts,
        score: 0,
        in_water: false,
        water_buoyancy_id: None,
    });
}

// ============================================================================
// Systems
// ============================================================================

fn player_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut query: Query<&mut Player>,
    game_state: Res<GameState>,
    cached: Res<CachedPhysicsState>,
    time: Res<Time>,
    mut exit: EventWriter<AppExit>,
) {
    let Ok(mut player) = query.get_single_mut() else { return };
    let Some(player_id) = game_state.player_id else { return };

    let dt = time.delta_secs();

    // Movement input - apply forces through PhysicsHandle API
    let mut input_dir = (0.0_f64, 0.0_f64);

    if keyboard.pressed(KeyCode::KeyW) || keyboard.pressed(KeyCode::ArrowUp) {
        input_dir.1 -= 1.0;
    }
    if keyboard.pressed(KeyCode::KeyS) || keyboard.pressed(KeyCode::ArrowDown) {
        input_dir.1 += 1.0;
    }
    if keyboard.pressed(KeyCode::KeyA) || keyboard.pressed(KeyCode::ArrowLeft) {
        input_dir.0 -= 1.0;
    }
    if keyboard.pressed(KeyCode::KeyD) || keyboard.pressed(KeyCode::ArrowRight) {
        input_dir.0 += 1.0;
    }

    // Apply rolling force and torque via PhysicsHandle API
    // Rolling = linear force + torque perpendicular to movement direction
    let len = (input_dir.0 * input_dir.0 + input_dir.1 * input_dir.1).sqrt();
    if len > 0.0 {
        let force_mult = if player.grounded { ROLL_FORCE } else { ROLL_FORCE * 0.3 };
        let dir_x = input_dir.0 / len;
        let dir_z = input_dir.1 / len;

        // Linear force
        let force = (dir_x * force_mult, 0.0, dir_z * force_mult);
        let _ = game_state.physics.apply_force(player_id, force);

        // Rolling torque - small assist to help start rolling
        // The physics engine's friction should handle most of the rolling,
        // but we add a small torque to help with responsiveness.
        // Using 10% of force * radius (much smaller than before)
        let torque_mult = force_mult * BALL_RADIUS * 0.1;
        let torque = (
            dir_z * torque_mult,   // Movement in +Z creates +X rotation (rolling forward)
            0.0,
            -dir_x * torque_mult,  // Movement in +X creates -Z rotation (rolling forward)
        );
        let _ = game_state.physics.apply_torque(player_id, torque);
    }

    // Jump
    player.jump_cooldown -= dt;
    if keyboard.just_pressed(KeyCode::Space) && player.grounded && player.jump_cooldown <= 0.0 {
        let _ = game_state.physics.apply_impulse(player_id, (0.0, JUMP_IMPULSE, 0.0));
        player.grounded = false;
        player.jump_cooldown = 0.2;
    }

    // Reset
    if keyboard.just_pressed(KeyCode::KeyR) {
        let spawn = game_state.spawn_point;
        let _ = game_state.physics.set_position(player_id, (
            spawn.x as f64,
            spawn.y as f64,
            spawn.z as f64,
        ));
        let _ = game_state.physics.set_velocity(player_id, (0.0, 0.0, 0.0));
    }

    // Exit
    if keyboard.just_pressed(KeyCode::Escape) {
        let _ = game_state.physics.shutdown();
        exit.send(AppExit::Success);
    }

    // Update grounded state from cached physics state
    if let Some(obj) = cached.state.get_object(player_id) {
        let ball_bottom = obj.position.1 - BALL_RADIUS;
        player.grounded = ball_bottom < 0.1 && obj.velocity.1.abs() < 0.5;

        // Also ground if on platforms
        if !player.grounded {
            for plat_y in [2.25, 6.25] {
                if (obj.position.1 - BALL_RADIUS - plat_y).abs() < 0.1
                    && obj.velocity.1.abs() < 0.5
                {
                    player.grounded = true;
                    break;
                }
            }
        }

        // Clamp velocity if needed
        let vx = obj.velocity.0.clamp(-MAX_VELOCITY, MAX_VELOCITY);
        let vy = obj.velocity.1.clamp(-MAX_VELOCITY * 2.0, MAX_VELOCITY);
        let vz = obj.velocity.2.clamp(-MAX_VELOCITY, MAX_VELOCITY);

        if vx != obj.velocity.0 || vy != obj.velocity.1 || vz != obj.velocity.2 {
            let _ = game_state.physics.set_velocity(player_id, (vx, vy, vz));
        }

        // Fall off world reset
        if obj.position.1 < -10.0 {
            let spawn = game_state.spawn_point;
            let _ = game_state.physics.set_position(player_id, (
                spawn.x as f64,
                spawn.y as f64,
                spawn.z as f64,
            ));
            let _ = game_state.physics.set_velocity(player_id, (0.0, 0.0, 0.0));
        }
    }
}

/// Check if player is in water zone and apply buoyancy
fn check_water_zones(
    mut game_state: ResMut<GameState>,
    cached: Res<CachedPhysicsState>,
    water_zones: Query<&WaterZone>,
) {
    let Some(player_id) = game_state.player_id else { return };

    // Use cached state for logic
    let player_pos = if let Some(obj) = cached.state.get_object(player_id) {
        Vec3::new(obj.position.0 as f32, obj.position.1 as f32, obj.position.2 as f32)
    } else {
        return;
    };

    let mut currently_in_water = false;

    for water in &water_zones {
        if player_pos.x >= water.bounds_min.x
            && player_pos.x <= water.bounds_max.x
            && player_pos.z >= water.bounds_min.z
            && player_pos.z <= water.bounds_max.z
            && player_pos.y <= water.bounds_max.y
        {
            currently_in_water = true;

            // Add buoyancy force if not already in water
            if !game_state.in_water {
                // Add continuous buoyancy
                let buoyancy_id = game_state.physics.add_buoyancy(player_id, water.bounds_max.y as f64, 1000.0).unwrap();
                game_state.water_buoyancy_id = Some(buoyancy_id);
            }

            // Apply one-time drag force (water drag)
            let _ = game_state.physics.apply_drag(player_id, 2.0);
            break;
        }
    }

    // Remove buoyancy if left water
    if !currently_in_water && game_state.in_water {
        if let Some(buoyancy_id) = game_state.water_buoyancy_id.take() {
            let _ = game_state.physics.remove_continuous_force(buoyancy_id);
        }
    }

    game_state.in_water = currently_in_water;
}

/// Sync player visual transform from cached physics state (no interpolation - real-time physics)
fn sync_player_transform(
    cached: Res<CachedPhysicsState>,
    mut query: Query<(&Player, &mut Transform)>,
) {
    for (player, mut transform) in &mut query {
        if let Some(obj) = cached.state.get_object(player.object_id) {
            // Direct position from real-time physics - always fresh
            transform.translation = Vec3::new(
                obj.position.0 as f32,
                obj.position.1 as f32,
                obj.position.2 as f32,
            );
            transform.rotation = Quat::from_xyzw(
                obj.orientation.0 as f32,
                obj.orientation.1 as f32,
                obj.orientation.2 as f32,
                obj.orientation.3 as f32,
            );
        }
    }
}

/// Sync rope bridge visuals from cached physics state (no interpolation - real-time physics)
fn sync_rope_visuals(
    game_state: Res<GameState>,
    cached: Res<CachedPhysicsState>,
    mut query: Query<(&RopeBridgeVisual, &mut Transform)>,
) {
    for (visual, mut transform) in &mut query {
        if let Some(&constraint_id) = game_state.rope_constraint_ids.get(visual.rope_index) {
            if let Some(positions) = cached.state.get_rope_chain_particles(constraint_id) {
                if visual.particle_index < positions.len() {
                    let pos = positions[visual.particle_index];
                    transform.translation = Vec3::new(pos.0 as f32, pos.1 as f32, pos.2 as f32);
                }
            }
        }
    }
}

/// Update kinematic plank physics positions based on rope positions
fn update_kinematic_planks(
    game_state: Res<GameState>,
    cached: Res<CachedPhysicsState>,
    query: Query<&BridgePlank>,
) {
    if game_state.rope_constraint_ids.len() < 2 {
        return;
    }

    let left_rope_id = game_state.rope_constraint_ids[0];
    let right_rope_id = game_state.rope_constraint_ids[1];

    // Get current rope positions from cached state
    let curr_left = cached.state.get_rope_chain_particles(left_rope_id);
    let curr_right = cached.state.get_rope_chain_particles(right_rope_id);

    for plank in &query {
        let i = plank.particle_index;

        if let (Some(cl), Some(cr)) = (curr_left, curr_right) {
            if i < cl.len() && i < cr.len() {
                let mid_x = (cl[i].0 + cr[i].0) / 2.0;
                let mid_y = (cl[i].1 + cr[i].1) / 2.0;
                let mid_z = (cl[i].2 + cr[i].2) / 2.0;

                // Update kinematic physics object position
                let _ = game_state.physics.set_position_kinematic(
                    plank.object_id,
                    (mid_x, mid_y, mid_z),
                    DT,
                );
            }
        }
    }
}

/// Sync plank visuals from cached physics state (no interpolation - real-time physics)
fn sync_plank_visuals(
    game_state: Res<GameState>,
    cached: Res<CachedPhysicsState>,
    mut query: Query<(&BridgePlank, &mut Transform)>,
) {
    if game_state.rope_constraint_ids.len() < 2 {
        return;
    }

    let left_rope_id = game_state.rope_constraint_ids[0];
    let right_rope_id = game_state.rope_constraint_ids[1];

    let left_positions = cached.state.get_rope_chain_particles(left_rope_id);
    let right_positions = cached.state.get_rope_chain_particles(right_rope_id);

    for (plank, mut transform) in &mut query {
        let i = plank.particle_index;

        if let (Some(left), Some(right)) = (left_positions, right_positions) {
            if i < left.len() && i < right.len() {
                // Position at midpoint between left and right rope particles
                let mid_x = (left[i].0 + right[i].0) / 2.0;
                let mid_y = (left[i].1 + right[i].1) / 2.0;
                let mid_z = (left[i].2 + right[i].2) / 2.0;
                transform.translation = Vec3::new(mid_x as f32, mid_y as f32, mid_z as f32);

                // Rotation based on tilt between left and right rope
                let y_diff = (right[i].1 - left[i].1) as f32;
                let z_diff = (right[i].2 - left[i].2) as f32;
                transform.rotation = Quat::from_rotation_x(y_diff.atan2(z_diff));
            }
        }
    }
}

/// Sync hinge visuals and collision body from cached physics state (no interpolation - real-time physics)
fn sync_hinge_visuals(
    game_state: Res<GameState>,
    cached: Res<CachedPhysicsState>,
    mut query: Query<(&HingeDoorVisual, &mut Transform)>,
) {
    for (visual, mut transform) in &mut query {
        if let Some(&hinge_id) = game_state.hinge_constraint_ids.get(visual.hinge_index) {
            // Get hinge state (angle, angular_velocity)
            if let Some((angle, _)) = cached.state.get_hinge_state(hinge_id) {
                // Get anchor position from constraint state
                let anchor = if let Some(constraint) = cached.state.get_constraint(hinge_id) {
                    if let ConstraintState::Hinge(hinge_state) = constraint {
                        Vec3::new(
                            hinge_state.anchor.0 as f32,
                            hinge_state.anchor.1 as f32,
                            hinge_state.anchor.2 as f32,
                        )
                    } else {
                        continue;
                    }
                } else {
                    continue;
                };

                let rotation = Quat::from_rotation_x(angle as f32);
                let door_offset = Vec3::new(0.0, 0.0, 2.0);
                let rotated_offset = rotation * door_offset;

                let new_pos = anchor + rotated_offset;
                transform.translation = new_pos;
                transform.rotation = rotation;

                // Update the door collision body position (kinematic)
                let _ = game_state.physics.set_position_kinematic(
                    visual.door_object_id,
                    (new_pos.x as f64, new_pos.y as f64, new_pos.z as f64),
                    DT,
                );
            }
        }
    }
}

fn camera_follow(
    player_query: Query<&Transform, (With<Player>, Without<MainCamera>)>,
    mut camera_query: Query<&mut Transform, With<MainCamera>>,
) {
    let Ok(player_transform) = player_query.get_single() else { return };
    let Ok(mut camera_transform) = camera_query.get_single_mut() else { return };

    let target_pos = player_transform.translation + Vec3::new(0.0, 8.0, 15.0);
    camera_transform.translation = camera_transform.translation.lerp(target_pos, 0.05);

    let look_target = player_transform.translation + Vec3::new(0.0, 1.0, 0.0);
    camera_transform.look_at(look_target, Vec3::Y);
}

fn update_ui(
    player_query: Query<&Player>,
    game_state: Res<GameState>,
    cached: Res<CachedPhysicsState>,
    mut ui_query: Query<&mut Text, With<UIText>>,
) {
    let Ok(player) = player_query.get_single() else { return };
    let Ok(mut text) = ui_query.get_single_mut() else { return };
    let Some(player_id) = game_state.player_id else { return };

    // Use cached state for UI display
    let (pos, vel) = if let Some(obj) = cached.state.get_object(player_id) {
        (obj.position, obj.velocity)
    } else {
        return;
    };

    let status = if player.grounded { "Grounded" } else { "Airborne" };
    let water_status = if game_state.in_water { " (In Water)" } else { "" };

    **text = format!(
        "Ball Playground (Real-Time Background Thread)\n\
         WASD: Move | Space: Jump | R: Reset\n\n\
         Position: ({:.1}, {:.1}, {:.1})\n\
         Velocity: ({:.1}, {:.1}, {:.1})\n\
         Status: {}{}\n\
         Physics Tick: {}\n\
         Score: {}",
        pos.0, pos.1, pos.2,
        vel.0, vel.1, vel.2,
        status, water_status,
        cached.state.tick,
        game_state.score,
    );
}
