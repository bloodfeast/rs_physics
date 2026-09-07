//! Rolling Ball Simple - Custom inline physics playground
//!
//! A physics playground that uses custom inline physics code instead of
//! the rs_physics world simulation. Features:
//! - Rolling ball with manual physics
//! - Large test area with obstacles
//! - Trampolines (bouncy platforms)
//! - Rope bridge with particle physics
//! - Water pool with custom shader and wave physics
//! - Floating wooden raft with distance constraints
//!
//! Controls:
//! - WASD or Arrow Keys: Roll the ball
//! - Space: Jump (when grounded)
//! - R: Reset ball position
//! - Escape: Exit

use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderRef};
use bevy_visual_tests::particle_material;
use std::f32::consts::PI;

const DT: f32 = 1.0 / 60.0;
const BALL_RADIUS: f32 = 0.5;
const GRAVITY: f32 = -15.0;
const ROLL_FORCE: f32 = 50.0;
const JUMP_IMPULSE: f32 = 8.0;
const MAX_VELOCITY: f32 = 12.0;

// Water pool constants
const POOL_CENTER_X: f32 = -14.0;
const POOL_CENTER_Z: f32 = -4.0;
const POOL_WIDTH: f32 = 12.0;
const POOL_DEPTH: f32 = 16.0;
const POOL_SURFACE_Y: f32 = 2.0;
const POOL_BOTTOM_Y: f32 = -0.5;

// Wave grid resolution
const WAVE_GRID_SIZE: usize = 32;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Rolling Ball Simple - Custom Physics Playground".to_string(),
                resolution: (1280.0, 720.0).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(MaterialPlugin::<WaterMaterial>::default())
        .add_plugins(MaterialPlugin::<SplashMaterial>::default())
        .init_resource::<PhysicsState>()
        .add_systems(Startup, setup)
        .add_systems(Update, (
            player_input,
            physics_step,
            rope_bridge_physics,
            raft_physics,
            water_wave_physics,
            sync_transforms,
            sync_water_mesh,
            update_spillover_stream,
            camera_follow,
            update_ui,
        ).chain())
        .run();
}

// ============================================================================
// Water Shader Material
// ============================================================================

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct WaterMaterial {
    #[uniform(0)]
    deep_color: LinearRgba,
    #[uniform(0)]
    shallow_color: LinearRgba,
    #[uniform(0)]
    foam_color: LinearRgba,
    #[uniform(0)]
    specular_color: LinearRgba,
    #[uniform(0)]
    fresnel_power: f32,
    #[uniform(0)]
    specular_power: f32,
    #[uniform(0)]
    wave_speed: f32,
    #[uniform(0)]
    wave_scale: f32,
}

impl Material for WaterMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/water.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}

// Splash material for spillover droplets
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct SplashMaterial {
    #[uniform(0)]
    base_color: LinearRgba,
    #[uniform(0)]
    highlight_color: LinearRgba,
    #[uniform(0)]
    fresnel_power: f32,
    #[uniform(0)]
    opacity: f32,
    #[uniform(0)]
    _padding1: f32,
    #[uniform(0)]
    _padding2: f32,
}

impl Material for SplashMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/splash.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}

// ============================================================================
// Components
// ============================================================================

#[derive(Component)]
struct Player {
    grounded: bool,
    jump_cooldown: f32,
}

#[derive(Component)]
struct PhysicsBody {
    position: Vec3,
    velocity: Vec3,
    angular_velocity: Vec3,  // For tracking ball spin
    radius: f32,
    mass: f32,
    restitution: f32,
}

#[derive(Component)]
struct MainCamera;

#[derive(Component)]
struct Trampoline {
    bounds_min: Vec3,
    bounds_max: Vec3,
    bounce_strength: f32,
}

#[derive(Component)]
struct RopeBridgeSegment {
    particle_index: usize,
}

#[derive(Component)]
struct RaftPlank {
    particle_index: usize,
}

#[derive(Component)]
struct WaterSurface;

#[derive(Component)]
struct SpilloverParticle {
    velocity: Vec3,
    lifetime: f32,
}

#[derive(Resource)]
struct SpilloverResources {
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
}

#[derive(Component)]
struct UIText;

// ============================================================================
// Resources
// ============================================================================

#[derive(Clone)]
struct RopeParticle {
    position: Vec3,
    prev_position: Vec3,
    velocity: Vec3,
    fixed: bool,
}

#[derive(Clone)]
struct DistanceConstraint {
    particle_a: usize,
    particle_b: usize,
    rest_length: f32,
    stiffness: f32,
}

struct RopeBridge {
    particles: Vec<RopeParticle>,
    constraints: Vec<DistanceConstraint>,
}

impl RopeBridge {
    fn new(start: Vec3, end: Vec3, segments: usize) -> Self {
        let mut particles = Vec::new();
        let mut constraints = Vec::new();

        let segment_length = (end - start).length() / segments as f32;

        for i in 0..=segments {
            let t = i as f32 / segments as f32;
            let pos = start.lerp(end, t);
            // Add some initial sag
            let sag = (t * PI).sin() * 0.5;
            let sagged_pos = Vec3::new(pos.x, pos.y - sag, pos.z);

            particles.push(RopeParticle {
                position: sagged_pos,
                prev_position: sagged_pos,
                velocity: Vec3::ZERO,
                fixed: i == 0 || i == segments, // Fix endpoints
            });

            if i > 0 {
                constraints.push(DistanceConstraint {
                    particle_a: i - 1,
                    particle_b: i,
                    rest_length: segment_length,
                    stiffness: 0.8,
                });
            }
        }

        RopeBridge { particles, constraints }
    }
}

struct FloatingRaft {
    particles: Vec<RopeParticle>,
    constraints: Vec<DistanceConstraint>,
}

impl FloatingRaft {
    fn new(center: Vec3, width: f32, depth: f32, grid_w: usize, grid_d: usize) -> Self {
        let mut particles = Vec::new();
        let mut constraints = Vec::new();

        let spacing_x = width / (grid_w - 1).max(1) as f32;
        let spacing_z = depth / (grid_d - 1).max(1) as f32;
        let start_x = center.x - width / 2.0;
        let start_z = center.z - depth / 2.0;

        // Create grid of particles
        for z in 0..grid_d {
            for x in 0..grid_w {
                let px = start_x + x as f32 * spacing_x;
                let pz = start_z + z as f32 * spacing_z;
                particles.push(RopeParticle {
                    position: Vec3::new(px, center.y, pz),
                    prev_position: Vec3::new(px, center.y, pz),
                    velocity: Vec3::ZERO,
                    fixed: false,
                });
            }
        }

        // Create structural constraints
        for z in 0..grid_d {
            for x in 0..grid_w {
                let idx = z * grid_w + x;

                if x < grid_w - 1 {
                    constraints.push(DistanceConstraint {
                        particle_a: idx,
                        particle_b: idx + 1,
                        rest_length: spacing_x,
                        stiffness: 0.9,
                    });
                }

                if z < grid_d - 1 {
                    constraints.push(DistanceConstraint {
                        particle_a: idx,
                        particle_b: idx + grid_w,
                        rest_length: spacing_z,
                        stiffness: 0.9,
                    });
                }

                // Diagonal constraints
                if x < grid_w - 1 && z < grid_d - 1 {
                    let diag_len = (spacing_x * spacing_x + spacing_z * spacing_z).sqrt();
                    constraints.push(DistanceConstraint {
                        particle_a: idx,
                        particle_b: idx + grid_w + 1,
                        rest_length: diag_len,
                        stiffness: 0.8,
                    });
                    constraints.push(DistanceConstraint {
                        particle_a: idx + 1,
                        particle_b: idx + grid_w,
                        rest_length: diag_len,
                        stiffness: 0.8,
                    });
                }
            }
        }

        FloatingRaft { particles, constraints }
    }
}

struct WaterZone {
    bounds_min: Vec3,
    bounds_max: Vec3,
    buoyancy: f32,
    drag: f32,
    // Dynamic water level
    current_level: f32,
    base_level: f32,
    max_level: f32,
    spillover_edge: f32,  // Z coordinate where spillover happens (front edge)
    spillover_rate: f32,  // How fast water spills over
    refill_rate: f32,     // How fast water refills
}

impl WaterZone {
    fn contains(&self, pos: Vec3) -> bool {
        pos.x >= self.bounds_min.x && pos.x <= self.bounds_max.x &&
        pos.y >= self.bounds_min.y && pos.y <= self.current_level &&
        pos.z >= self.bounds_min.z && pos.z <= self.bounds_max.z
    }

    fn surface_y(&self) -> f32 {
        self.current_level
    }

    fn spillover_amount(&self) -> f32 {
        (self.current_level - self.spillover_edge).max(0.0)
    }
}

// Wave simulation data
struct WaveGrid {
    heights: Vec<Vec<f32>>,
    velocities: Vec<Vec<f32>>,
    size: usize,
    cell_size: f32,
    origin: Vec2,
    spillover_accumulated: f32,  // Track how much water spills over
}

impl WaveGrid {
    fn new(size: usize, width: f32, depth: f32, origin: Vec2) -> Self {
        let cell_size = width / size as f32;
        WaveGrid {
            heights: vec![vec![0.0; size]; size],
            velocities: vec![vec![0.0; size]; size],
            size,
            cell_size,
            origin,
            spillover_accumulated: 0.0,
        }
    }

    fn update(&mut self, dt: f32, _spillover_edge_z: usize) -> f32 {
        // Realistic pool water parameters
        let wave_speed_sq = 2.5;  // c² - slower, more realistic wave propagation
        let damping = 0.997;      // High damping for calm pool water
        let height_decay = 0.9995; // Gradual settling to flat

        // Track spillover at edge
        let mut spillover = 0.0;

        // Proper shallow water wave equation: acceleration = c² * laplacian
        for z in 1..self.size - 1 {
            for x in 1..self.size - 1 {
                // Calculate laplacian (second spatial derivative)
                let laplacian = (
                    self.heights[z][x - 1] +
                    self.heights[z][x + 1] +
                    self.heights[z - 1][x] +
                    self.heights[z + 1][x]
                    - 4.0 * self.heights[z][x]
                ) / (self.cell_size * self.cell_size);

                // Apply wave equation: acceleration = c² * laplacian
                let acceleration = wave_speed_sq * laplacian;
                self.velocities[z][x] += acceleration * dt;

                // Apply damping to velocity
                self.velocities[z][x] *= damping;
            }
        }

        // Update heights from velocities
        for z in 1..self.size - 1 {
            for x in 1..self.size - 1 {
                self.heights[z][x] += self.velocities[z][x] * dt;

                // Gentle decay towards equilibrium (water settles)
                self.heights[z][x] *= height_decay;

                // Clamp to reasonable range
                self.heights[z][x] = self.heights[z][x].clamp(-0.5, 0.5);
            }
        }

        // Handle spillover at the front edge (high z values)
        for x in 0..self.size {
            let edge_z = self.size - 1;
            if self.heights[edge_z][x] > 0.02 {
                spillover += self.heights[edge_z][x] * 0.2;
                self.heights[edge_z][x] *= 0.8;
                self.velocities[edge_z][x] = self.velocities[edge_z][x].min(0.0);
            }
        }

        // Absorbing boundaries (reduce reflections for more natural look)
        for x in 0..self.size {
            // Back edge - absorb
            self.heights[0][x] *= 0.9;
            self.velocities[0][x] *= 0.5;
        }
        for z in 0..self.size {
            // Left and right edges - absorb
            self.heights[z][0] *= 0.9;
            self.heights[z][self.size - 1] *= 0.9;
            self.velocities[z][0] *= 0.5;
            self.velocities[z][self.size - 1] *= 0.5;
        }

        self.spillover_accumulated += spillover;
        spillover
    }

    fn add_disturbance(&mut self, world_pos: Vec3, strength: f32) {
        let local_x = ((world_pos.x - self.origin.x) / self.cell_size) as i32;
        let local_z = ((world_pos.z - self.origin.y) / self.cell_size) as i32;

        let radius = 2;  // Smaller radius for more focused ripples
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let x = local_x + dx;
                let z = local_z + dy;
                if x >= 1 && x < (self.size - 1) as i32 && z >= 1 && z < (self.size - 1) as i32 {
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    // Gaussian-like falloff for smooth ripples
                    let falloff = (-dist * dist / 2.0).exp();
                    // Only add to height (velocity will follow naturally from wave equation)
                    self.heights[z as usize][x as usize] += strength * falloff;
                }
            }
        }
    }

    fn get_height(&self, world_pos: Vec3) -> f32 {
        let local_x = ((world_pos.x - self.origin.x) / self.cell_size).clamp(0.0, (self.size - 1) as f32);
        let local_z = ((world_pos.z - self.origin.y) / self.cell_size).clamp(0.0, (self.size - 1) as f32);

        let x0 = local_x as usize;
        let z0 = local_z as usize;
        let x1 = (x0 + 1).min(self.size - 1);
        let z1 = (z0 + 1).min(self.size - 1);

        let fx = local_x.fract();
        let fz = local_z.fract();

        let h00 = self.heights[z0][x0];
        let h10 = self.heights[z0][x1];
        let h01 = self.heights[z1][x0];
        let h11 = self.heights[z1][x1];

        let h0 = h00 * (1.0 - fx) + h10 * fx;
        let h1 = h01 * (1.0 - fx) + h11 * fx;

        h0 * (1.0 - fz) + h1 * fz
    }

    fn total_volume(&self) -> f32 {
        let mut total = 0.0;
        for row in &self.heights {
            for &h in row {
                total += h;
            }
        }
        total * self.cell_size * self.cell_size
    }

    /// Add a swirl/vortex effect at a world position based on angular velocity
    /// This simulates a spinning ball creating circular currents in water
    fn add_swirl(&mut self, world_pos: Vec3, angular_velocity: Vec3, radius: f32) {
        // Only the Y component of angular velocity affects water surface swirl
        let spin_strength = angular_velocity.y;
        if spin_strength.abs() < 0.1 {
            return;
        }

        let local_x = ((world_pos.x - self.origin.x) / self.cell_size) as i32;
        let local_z = ((world_pos.z - self.origin.y) / self.cell_size) as i32;

        let swirl_radius = ((radius * 2.0) / self.cell_size) as i32 + 2;

        for dy in -swirl_radius..=swirl_radius {
            for dx in -swirl_radius..=swirl_radius {
                let x = local_x + dx;
                let z = local_z + dy;
                if x >= 1 && x < (self.size - 1) as i32 && z >= 1 && z < (self.size - 1) as i32 {
                    let dist_sq = (dx * dx + dy * dy) as f32;
                    let max_dist_sq = (swirl_radius * swirl_radius) as f32;

                    if dist_sq < max_dist_sq && dist_sq > 0.01 {
                        let dist = dist_sq.sqrt();
                        // Swirl strength decreases with distance, but not at center
                        let falloff = (1.0 - dist / swirl_radius as f32).max(0.0);
                        let strength = spin_strength * falloff * 0.002;

                        // Perpendicular direction to create circular flow
                        // For a point (dx, dz) from center, perpendicular is (-dz, dx) for CCW
                        let perp_x = -dy as f32 / dist;
                        let perp_z = dx as f32 / dist;

                        // Apply velocity in perpendicular direction (creates circular flow)
                        // This affects neighboring cells' heights to simulate flow
                        let xu = x as usize;
                        let zu = z as usize;

                        // Push water in the tangential direction of the swirl
                        self.velocities[zu][xu] += strength * perp_z;

                        // Also create slight height displacement for visual effect
                        self.heights[zu][xu] += strength * 0.5;
                    }
                }
            }
        }
    }
}

#[derive(Resource)]
struct PhysicsState {
    spawn_point: Vec3,
    water_zone: Option<WaterZone>,
    rope_bridge: Option<RopeBridge>,
    raft: Option<FloatingRaft>,
    wave_grid: Option<WaveGrid>,
    time: f32,
}


impl Default for PhysicsState {
    fn default() -> Self {
        Self {
            spawn_point: Vec3::ZERO,
            water_zone: None,
            rope_bridge: None,
            raft: None,
            wave_grid: None,
            time: 0.0,
        }
    }
}

// ============================================================================
// Setup
// ============================================================================

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut water_materials: ResMut<Assets<WaterMaterial>>,
    mut physics: ResMut<PhysicsState>,
) {
    // === Materials ===
    let ball_material = particle_material(&mut materials, Color::srgb(0.9, 0.2, 0.3));
    let ground_material = particle_material(&mut materials, Color::srgb(0.3, 0.5, 0.3));
    let platform_material = particle_material(&mut materials, Color::srgb(0.4, 0.4, 0.5));
    let trampoline_material = particle_material(&mut materials, Color::srgb(0.9, 0.6, 0.1));
    let rope_material = particle_material(&mut materials, Color::srgb(0.6, 0.4, 0.2));
    let plank_material = particle_material(&mut materials, Color::srgb(0.5, 0.35, 0.2));

    // Spawn point
    let spawn_point = Vec3::new(0.0, 3.0, 0.0);
    physics.spawn_point = spawn_point;

    // === Player Ball ===
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(BALL_RADIUS))),
        MeshMaterial3d(ball_material),
        Transform::from_translation(spawn_point),
        Player {
            grounded: false,
            jump_cooldown: 0.0,
        },
        PhysicsBody {
            position: spawn_point,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            radius: BALL_RADIUS,
            mass: 1.0,
            restitution: 0.5,
        },
    ));

    // === Ground ===
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(80.0, 0.5, 80.0))),
        MeshMaterial3d(ground_material.clone()),
        Transform::from_xyz(0.0, -0.25, 0.0),
    ));

    // === Trampolines ===
    let trampoline_positions = [
        Vec3::new(5.0, 0.1, 5.0),
        Vec3::new(10.0, 0.1, -3.0),
        Vec3::new(-5.0, 0.1, 8.0),
        Vec3::new(15.0, 0.1, 10.0),
    ];

    for pos in trampoline_positions {
        let size = Vec3::new(3.0, 0.2, 3.0);
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(size.x, size.y, size.z))),
            MeshMaterial3d(trampoline_material.clone()),
            Transform::from_translation(pos),
            Trampoline {
                bounds_min: pos - size / 2.0,
                bounds_max: pos + size / 2.0,
                bounce_strength: 20.0,
            },
        ));
    }

    // === Platforms for rope bridge ===
    let bridge_start = Vec3::new(-5.0, 3.0, -10.0);
    let bridge_end = Vec3::new(5.0, 3.0, -10.0);

    // Start platform
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(3.0, 3.0, 4.0))),
        MeshMaterial3d(platform_material.clone()),
        Transform::from_xyz(bridge_start.x - 1.5, 1.5, bridge_start.z),
    ));

    // End platform
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(3.0, 3.0, 4.0))),
        MeshMaterial3d(platform_material.clone()),
        Transform::from_xyz(bridge_end.x + 1.5, 1.5, bridge_end.z),
    ));

    // === Rope Bridge ===
    let bridge = RopeBridge::new(bridge_start, bridge_end, 12);
    let plank_mesh = meshes.add(Cuboid::new(0.8, 0.1, 1.5));

    for (i, particle) in bridge.particles.iter().enumerate() {
        commands.spawn((
            Mesh3d(plank_mesh.clone()),
            MeshMaterial3d(rope_material.clone()),
            Transform::from_translation(particle.position),
            RopeBridgeSegment { particle_index: i },
        ));
    }

    physics.rope_bridge = Some(bridge);

    // === Water Pool ===
    let water_bounds_min = Vec3::new(
        POOL_CENTER_X - POOL_WIDTH / 2.0,
        POOL_BOTTOM_Y,
        POOL_CENTER_Z - POOL_DEPTH / 2.0,
    );
    let water_bounds_max = Vec3::new(
        POOL_CENTER_X + POOL_WIDTH / 2.0,
        POOL_SURFACE_Y,
        POOL_CENTER_Z + POOL_DEPTH / 2.0,
    );

    // Front edge barrier height - water only spills when waves crest OVER this barrier
    // Set slightly above base water surface so only active waves trigger spillover
    let spillover_height = POOL_SURFACE_Y + 0.05;  // Barrier is 5cm above resting water level

    physics.water_zone = Some(WaterZone {
        bounds_min: water_bounds_min,
        bounds_max: water_bounds_max,
        buoyancy: 25.0,
        drag: 4.0,
        current_level: POOL_SURFACE_Y,
        base_level: POOL_BOTTOM_Y + 0.5,  // Minimum water level
        max_level: POOL_SURFACE_Y + 0.2,  // Can rise slightly above normal
        spillover_edge: spillover_height,
        spillover_rate: 0.5,
        refill_rate: 0.02,  // Slow refill (like rain or inflow)
    });

    // Wave grid for fluid dynamics
    physics.wave_grid = Some(WaveGrid::new(
        WAVE_GRID_SIZE,
        POOL_WIDTH,
        POOL_DEPTH,
        Vec2::new(water_bounds_min.x, water_bounds_min.z),
    ));

    // Water surface mesh with wave grid
    let water_mesh = create_water_mesh(WAVE_GRID_SIZE, POOL_WIDTH, POOL_DEPTH);
    let water_mat = water_materials.add(WaterMaterial {
        deep_color: LinearRgba::new(0.1, 0.3, 0.5, 1.0),
        shallow_color: LinearRgba::new(0.3, 0.6, 0.8, 1.0),
        foam_color: LinearRgba::new(0.9, 0.95, 1.0, 1.0),
        specular_color: LinearRgba::new(1.0, 1.0, 1.0, 1.0),
        fresnel_power: 4.0,
        specular_power: 64.0,
        wave_speed: 1.0,
        wave_scale: 0.3,
    });

    commands.spawn((
        Mesh3d(meshes.add(water_mesh)),
        MeshMaterial3d(water_mat),
        Transform::from_xyz(POOL_CENTER_X, POOL_SURFACE_Y, POOL_CENTER_Z),
        WaterSurface,
    ));

    // Pool walls
    let wall_material = particle_material(&mut materials, Color::srgb(0.5, 0.5, 0.55));
    let wall_thickness = 0.4;
    let wall_height = POOL_SURFACE_Y - POOL_BOTTOM_Y + 0.5;
    let front_wall_height = spillover_height - POOL_BOTTOM_Y;  // Lower front wall for spillover

    // Back wall (full height)
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(POOL_WIDTH + wall_thickness * 2.0, wall_height, wall_thickness))),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_xyz(POOL_CENTER_X, POOL_BOTTOM_Y + wall_height / 2.0, water_bounds_min.z - wall_thickness / 2.0),
    ));
    // Front wall (LOWER - spillover edge)
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(POOL_WIDTH + wall_thickness * 2.0, front_wall_height, wall_thickness))),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_xyz(POOL_CENTER_X, POOL_BOTTOM_Y + front_wall_height / 2.0, water_bounds_max.z + wall_thickness / 2.0),
    ));
    // Left wall
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(wall_thickness, wall_height, POOL_DEPTH))),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_xyz(water_bounds_min.x - wall_thickness / 2.0, POOL_BOTTOM_Y + wall_height / 2.0, POOL_CENTER_Z),
    ));
    // Right wall
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(wall_thickness, wall_height, POOL_DEPTH))),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_xyz(water_bounds_max.x + wall_thickness / 2.0, POOL_BOTTOM_Y + wall_height / 2.0, POOL_CENTER_Z),
    ));

    // === Spillover Particles Resources ===
    // Create shared mesh and material for spillover droplets - water-like appearance
    let droplet_mesh = meshes.add(Sphere::new(0.15)); // Small water droplets
    let droplet_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.3, 0.6, 0.9, 0.8), // Translucent blue
        emissive: LinearRgba::new(0.1, 0.3, 0.5, 1.0), // Subtle blue glow
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    commands.insert_resource(SpilloverResources {
        mesh: droplet_mesh,
        material: droplet_mat,
    });

    // === Floating Raft ===
    let raft_center = Vec3::new(POOL_CENTER_X, POOL_SURFACE_Y + 0.1, POOL_CENTER_Z);
    let raft = FloatingRaft::new(raft_center, 4.0, 3.0, 4, 3);

    let raft_plank_mesh = meshes.add(Cuboid::new(1.0, 0.15, 1.0));
    for (i, particle) in raft.particles.iter().enumerate() {
        commands.spawn((
            Mesh3d(raft_plank_mesh.clone()),
            MeshMaterial3d(plank_material.clone()),
            Transform::from_translation(particle.position),
            RaftPlank { particle_index: i },
        ));
    }

    physics.raft = Some(raft);

    // === Lighting ===
    commands.spawn((
        DirectionalLight {
            illuminance: 15000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(10.0, 20.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        PointLight {
            intensity: 500000.0,
            range: 60.0,
            ..default()
        },
        Transform::from_xyz(-10.0, 15.0, 5.0),
    ));

    // === Camera ===
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 20.0, 30.0).looking_at(Vec3::ZERO, Vec3::Y),
        MainCamera,
    ));

    // === UI ===
    commands.spawn((
        Text::new("Rolling Ball Playground\nWASD: Move | Space: Jump | R: Reset"),
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
}

fn create_water_mesh(resolution: usize, width: f32, depth: f32) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    let half_width = width / 2.0;
    let half_depth = depth / 2.0;

    for z in 0..=resolution {
        for x in 0..=resolution {
            let px = (x as f32 / resolution as f32) * width - half_width;
            let pz = (z as f32 / resolution as f32) * depth - half_depth;

            positions.push([px, 0.0, pz]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([x as f32 / resolution as f32, z as f32 / resolution as f32]);
        }
    }

    for z in 0..resolution {
        for x in 0..resolution {
            let i = z * (resolution + 1) + x;
            indices.push(i as u32);
            indices.push((i + resolution + 1) as u32);
            indices.push((i + 1) as u32);

            indices.push((i + 1) as u32);
            indices.push((i + resolution + 1) as u32);
            indices.push((i + resolution + 2) as u32);
        }
    }

    Mesh::new(bevy::render::mesh::PrimitiveTopology::TriangleList, bevy::render::render_asset::RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(bevy::render::mesh::Indices::U32(indices))
}

// ============================================================================
// Systems
// ============================================================================

fn player_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut query: Query<(&mut PhysicsBody, &mut Player)>,
    physics: Res<PhysicsState>,
) {
    let Ok((mut body, mut player)) = query.get_single_mut() else { return };

    let mut force = Vec3::ZERO;

    if keyboard.pressed(KeyCode::KeyW) || keyboard.pressed(KeyCode::ArrowUp) {
        force.z -= 1.0;
    }
    if keyboard.pressed(KeyCode::KeyS) || keyboard.pressed(KeyCode::ArrowDown) {
        force.z += 1.0;
    }
    if keyboard.pressed(KeyCode::KeyA) || keyboard.pressed(KeyCode::ArrowLeft) {
        force.x -= 1.0;
    }
    if keyboard.pressed(KeyCode::KeyD) || keyboard.pressed(KeyCode::ArrowRight) {
        force.x += 1.0;
    }

    if force.length_squared() > 0.0 {
        force = force.normalize() * ROLL_FORCE;
        let mass = body.mass;
        body.velocity += force * DT / mass;
    }

    player.jump_cooldown -= DT;
    if keyboard.just_pressed(KeyCode::Space) && player.grounded && player.jump_cooldown <= 0.0 {
        body.velocity.y = JUMP_IMPULSE;
        player.grounded = false;
        player.jump_cooldown = 0.3;
    }

    if keyboard.just_pressed(KeyCode::KeyR) {
        body.position = physics.spawn_point;
        body.velocity = Vec3::ZERO;
    }

    let horizontal = Vec3::new(body.velocity.x, 0.0, body.velocity.z);
    if horizontal.length() > MAX_VELOCITY {
        let clamped = horizontal.normalize() * MAX_VELOCITY;
        body.velocity.x = clamped.x;
        body.velocity.z = clamped.z;
    }

    // Calculate angular velocity from linear velocity (rolling ball physics)
    // For rolling without slipping: ω = v / r
    // Rolling in X direction -> rotates around Z axis
    // Rolling in Z direction -> rotates around X axis
    let roll_factor = 1.0 / body.radius;
    body.angular_velocity.x = body.velocity.z * roll_factor;  // Forward/back roll
    body.angular_velocity.z = -body.velocity.x * roll_factor;  // Left/right roll
    // Y component is for spinning (like a top) - comes from turning quickly
    // We'll add a small spin when the ball changes direction rapidly in water
}

fn physics_step(
    mut query: Query<(&mut PhysicsBody, &mut Player)>,
    mut physics: ResMut<PhysicsState>,
    trampoline_query: Query<&Trampoline>,
) {
    let Ok((mut body, mut player)) = query.get_single_mut() else { return };

    // Gravity
    body.velocity.y += GRAVITY * DT;

    // Water physics
    let mut in_water = false;
    if let Some(ref water) = physics.water_zone {
        if water.contains(body.position) {
            in_water = true;
            let water_surface = water.surface_y();

            if body.position.y <= water_surface {
                let submerged_depth = (water_surface - body.position.y).max(0.0);
                let submerged_fraction = (submerged_depth / body.radius).min(1.0);
                let buoyancy_force = water.buoyancy * submerged_fraction;
                body.velocity.y += buoyancy_force * DT;
            }

            body.velocity *= 1.0 - water.drag * DT * 0.5;

            // Calculate spin from movement in water (turbulence creates swirl)
            // When moving fast horizontally, the ball can create a vertical spin
            let horizontal_speed = Vec3::new(body.velocity.x, 0.0, body.velocity.z).length();
            if horizontal_speed > 1.0 {
                // Cross product of velocity with up vector gives perpendicular spin direction
                // The Y component of angular velocity represents spinning like a top
                let spin_factor = (horizontal_speed - 1.0) * 0.5;
                // Spin direction based on movement (turning creates spin)
                body.angular_velocity.y = spin_factor * (body.velocity.x.signum() * body.velocity.z.abs()
                    - body.velocity.z.signum() * body.velocity.x.abs());
            }

            // Add disturbance and swirl to wave grid
            if let Some(ref mut wave_grid) = physics.wave_grid {
                // Add displacement disturbance (splash)
                if body.velocity.length() > 0.5 {
                    wave_grid.add_disturbance(body.position, body.velocity.length() * 0.015);
                }

                // Add swirl effect from ball spin
                wave_grid.add_swirl(body.position, body.angular_velocity, body.radius);
            }
        }
    }

    // Update position
    let velocity = body.velocity;
    body.position += velocity * DT;

    // Trampoline collision
    for trampoline in trampoline_query.iter() {
        if body.position.x >= trampoline.bounds_min.x - body.radius &&
           body.position.x <= trampoline.bounds_max.x + body.radius &&
           body.position.z >= trampoline.bounds_min.z - body.radius &&
           body.position.z <= trampoline.bounds_max.z + body.radius &&
           body.position.y <= trampoline.bounds_max.y + body.radius &&
           body.position.y >= trampoline.bounds_min.y
        {
            if body.velocity.y < 0.0 {
                body.position.y = trampoline.bounds_max.y + body.radius;
                body.velocity.y = trampoline.bounce_strength;
                player.grounded = false;
            }
        }
    }

    // Ground collision
    let ground_y = body.radius;
    if body.position.y < ground_y && !in_water {
        body.position.y = ground_y;
        if body.velocity.y < 0.0 {
            body.velocity.y = -body.velocity.y * body.restitution;
            if body.velocity.y.abs() < 0.5 {
                body.velocity.y = 0.0;
            }
        }
        player.grounded = true;
    }

    if player.grounded {
        body.velocity.x *= 0.95;
        body.velocity.z *= 0.95;
    }
}

fn rope_bridge_physics(mut physics: ResMut<PhysicsState>) {
    let bridge = match &mut physics.rope_bridge {
        Some(b) => b,
        None => return,
    };

    let dt = DT;

    // Physics for particles
    for particle in bridge.particles.iter_mut() {
        if particle.fixed { continue; }

        particle.velocity.y += GRAVITY * dt;
        particle.velocity *= 0.99; // Damping

        let vel = particle.velocity;
        particle.position += vel * dt;
    }

    // Solve constraints
    for _ in 0..10 {
        for i in 0..bridge.constraints.len() {
            let constraint = &bridge.constraints[i];
            let a_pos = bridge.particles[constraint.particle_a].position;
            let b_pos = bridge.particles[constraint.particle_b].position;
            let a_fixed = bridge.particles[constraint.particle_a].fixed;
            let b_fixed = bridge.particles[constraint.particle_b].fixed;

            let delta = b_pos - a_pos;
            let dist = delta.length();
            if dist < 0.0001 { continue; }

            let diff = (dist - constraint.rest_length) / dist;
            let correction = delta * diff * constraint.stiffness * 0.5;

            if !a_fixed && !b_fixed {
                bridge.particles[constraint.particle_a].position += correction;
                bridge.particles[constraint.particle_b].position -= correction;
            } else if !a_fixed {
                bridge.particles[constraint.particle_a].position += correction * 2.0;
            } else if !b_fixed {
                bridge.particles[constraint.particle_b].position -= correction * 2.0;
            }
        }
    }
}

fn raft_physics(mut physics: ResMut<PhysicsState>) {
    let (water_surface, water_bottom, water_bounds_min, water_bounds_max) = {
        let water_zone = match &physics.water_zone {
            Some(w) => w,
            None => return,
        };
        (water_zone.surface_y(), water_zone.bounds_min.y, water_zone.bounds_min, water_zone.bounds_max)
    };

    let raft = match &mut physics.raft {
        Some(r) => r,
        None => return,
    };

    let dt = DT;

    for particle in raft.particles.iter_mut() {
        if particle.fixed { continue; }

        particle.velocity.y += GRAVITY * dt;

        if particle.position.y <= water_surface {
            let submerged_depth = (water_surface - particle.position.y).max(0.0);
            let submerged_fraction = (submerged_depth / 0.3).min(1.0);
            let buoyancy_force = 25.0 * submerged_fraction;
            particle.velocity.y += buoyancy_force * dt;

            let drag = 4.0;
            particle.velocity.x *= 1.0 - drag * dt * 0.3;
            particle.velocity.y *= 1.0 - drag * dt * 0.8;
            particle.velocity.z *= 1.0 - drag * dt * 0.3;
        }

        let vel = particle.velocity;
        particle.position += vel * dt;

        if particle.position.y < water_bottom + 0.1 {
            particle.position.y = water_bottom + 0.1;
            particle.velocity.y = particle.velocity.y.max(0.0);
        }

        let margin = 0.3;
        particle.position.x = particle.position.x.clamp(water_bounds_min.x + margin, water_bounds_max.x - margin);
        particle.position.z = particle.position.z.clamp(water_bounds_min.z + margin, water_bounds_max.z - margin);

        particle.velocity *= 0.98;
    }

    // Solve constraints
    for _ in 0..8 {
        for i in 0..raft.constraints.len() {
            let constraint = &raft.constraints[i];
            let a_pos = raft.particles[constraint.particle_a].position;
            let b_pos = raft.particles[constraint.particle_b].position;
            let a_fixed = raft.particles[constraint.particle_a].fixed;
            let b_fixed = raft.particles[constraint.particle_b].fixed;

            let delta = b_pos - a_pos;
            let dist = delta.length();
            if dist < 0.0001 { continue; }

            let diff = (dist - constraint.rest_length) / dist;
            let correction = delta * diff * constraint.stiffness * 0.5;

            if !a_fixed && !b_fixed {
                raft.particles[constraint.particle_a].position += correction;
                raft.particles[constraint.particle_b].position -= correction;
            } else if !a_fixed {
                raft.particles[constraint.particle_a].position += correction * 2.0;
            } else if !b_fixed {
                raft.particles[constraint.particle_b].position -= correction * 2.0;
            }
        }
    }
}

fn water_wave_physics(mut physics: ResMut<PhysicsState>, time: Res<Time>) {
    physics.time += time.delta_secs();
    let dt = DT;

    // Get wave grid size first
    let wave_size = physics.wave_grid.as_ref().map(|g| g.size).unwrap_or(0);

    // Update wave grid and get spillover
    // Water is naturally calm - only the ball creates disturbances
    let spillover = if let Some(ref mut wave_grid) = physics.wave_grid {
        wave_grid.update(dt, wave_size - 1)
    } else {
        0.0
    };

    // Update water level based on spillover
    if let Some(ref mut water_zone) = physics.water_zone {
        // Calculate spillover based on wave heights at edge
        let effective_spillover = spillover * water_zone.spillover_rate;

        // Lower water level when spilling
        water_zone.current_level -= effective_spillover * dt * 0.1;

        // Slowly refill (like rain or underground spring)
        water_zone.current_level += water_zone.refill_rate * dt;

        // Clamp to valid range
        water_zone.current_level = water_zone.current_level.clamp(
            water_zone.base_level,
            water_zone.max_level
        );
    }
}

fn sync_transforms(
    physics: Res<PhysicsState>,
    mut player_query: Query<(&PhysicsBody, &mut Transform), With<Player>>,
    mut bridge_query: Query<(&RopeBridgeSegment, &mut Transform), (Without<Player>, Without<RaftPlank>)>,
    mut raft_query: Query<(&RaftPlank, &mut Transform), (Without<Player>, Without<RopeBridgeSegment>)>,
) {
    for (body, mut transform) in player_query.iter_mut() {
        transform.translation = body.position;
    }

    if let Some(ref bridge) = physics.rope_bridge {
        for (segment, mut transform) in bridge_query.iter_mut() {
            if segment.particle_index < bridge.particles.len() {
                transform.translation = bridge.particles[segment.particle_index].position;
            }
        }
    }

    if let Some(ref raft) = physics.raft {
        for (plank, mut transform) in raft_query.iter_mut() {
            if plank.particle_index < raft.particles.len() {
                transform.translation = raft.particles[plank.particle_index].position;
            }
        }
    }
}

fn sync_water_mesh(
    physics: Res<PhysicsState>,
    mut water_query: Query<(&Mesh3d, &mut Transform), With<WaterSurface>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let wave_grid = match &physics.wave_grid {
        Some(g) => g,
        None => return,
    };

    // Get current water level for surface positioning
    let current_level = physics.water_zone.as_ref().map(|w| w.current_level).unwrap_or(POOL_SURFACE_Y);

    for (mesh_handle, mut transform) in water_query.iter_mut() {
        // Update water surface Y position based on current level
        transform.translation.y = current_level;

        if let Some(mesh) = meshes.get_mut(mesh_handle.id()) {
            if let Some(positions) = mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION) {
                if let bevy::render::mesh::VertexAttributeValues::Float32x3(pos) = positions {
                    let resolution = wave_grid.size;
                    for z in 0..=resolution {
                        for x in 0..=resolution {
                            let idx = z * (resolution + 1) + x;
                            if idx < pos.len() {
                                let grid_x = x.min(resolution - 1);
                                let grid_z = z.min(resolution - 1);
                                pos[idx][1] = wave_grid.heights[grid_z][grid_x];
                            }
                        }
                    }
                }
            }

            // Recalculate normals
            if let Some(positions) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
                if let bevy::render::mesh::VertexAttributeValues::Float32x3(pos) = positions {
                    let resolution = wave_grid.size;
                    let mut normals = vec![[0.0f32, 1.0, 0.0]; pos.len()];

                    for z in 0..=resolution {
                        for x in 0..=resolution {
                            let idx = z * (resolution + 1) + x;
                            if idx < pos.len() {
                                let left = if x > 0 { pos[idx - 1][1] } else { pos[idx][1] };
                                let right = if x < resolution { pos[idx + 1][1] } else { pos[idx][1] };
                                let up = if z > 0 { pos[idx - resolution - 1][1] } else { pos[idx][1] };
                                let down = if z < resolution { pos[idx + resolution + 1][1] } else { pos[idx][1] };

                                let dx = right - left;
                                let dz = down - up;
                                let normal = Vec3::new(-dx, 2.0, -dz).normalize();
                                normals[idx] = [normal.x, normal.y, normal.z];
                            }
                        }
                    }

                    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
                }
            }
        }
    }
}

/// Update spillover particles - spawn droplets when waves crest the barrier
fn update_spillover_stream(
    mut commands: Commands,
    physics: Res<PhysicsState>,
    spillover_res: Option<Res<SpilloverResources>>,
    mut particle_query: Query<(Entity, &mut SpilloverParticle, &mut Transform)>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();

    // Log every ~2 seconds
    static mut LOG_TIMER: f32 = 0.0;
    let should_log = unsafe {
        LOG_TIMER += dt;
        if LOG_TIMER > 2.0 {
            LOG_TIMER = 0.0;
            true
        } else {
            false
        }
    };

    let water_zone = match &physics.water_zone {
        Some(w) => w,
        None => return,
    };

    let wave_grid = match &physics.wave_grid {
        Some(g) => g,
        None => return,
    };

    // Check all four edges for cresting waves
    // Edge indices in the wave grid
    let front_edge_z = wave_grid.size - 2; // +Z edge
    let back_edge_z = 1;                    // -Z edge
    let right_edge_x = wave_grid.size - 2;  // +X edge
    let left_edge_x = 1;                    // -X edge

    // Helper to spawn particles along an edge
    let spawn_edge_particles = |commands: &mut Commands, res: &SpilloverResources,
                                 spawn_pos: Vec3, outward_dir: Vec3, normalized_crest: f32| {
        let rand_offset = rand::random::<f32>() * 0.1;
        let pos = spawn_pos + Vec3::Y * rand_offset;

        // Velocity flows outward from pool
        let initial_vel = Vec3::new(
            outward_dir.x * (0.5 + rand::random::<f32>() * 1.0) + (rand::random::<f32>() - 0.5) * 0.5,
            rand::random::<f32>() * 2.0 * normalized_crest,
            outward_dir.z * (0.5 + rand::random::<f32>() * 1.0) + (rand::random::<f32>() - 0.5) * 0.5,
        );

        commands.spawn((
            SpilloverParticle {
                velocity: initial_vel,
                lifetime: 2.5,
            },
            Mesh3d(res.mesh.clone()),
            MeshMaterial3d(res.material.clone()),
            Transform::from_translation(pos)
                .with_scale(Vec3::splat(0.8 + rand::random::<f32>() * 0.6)),
        ));
    };

    // Calculate grid cell size
    let cell_width = POOL_WIDTH / (wave_grid.size - 1) as f32;
    let cell_depth = POOL_DEPTH / (wave_grid.size - 1) as f32;

    // Track total cresting for spawn rate
    let mut total_cresting = 0.0;
    let mut total_cells = 0;

    // Check front edge (+Z)
    for x in 1..wave_grid.size - 1 {
        let wave_height = wave_grid.heights[front_edge_z][x];
        let effective_height = POOL_SURFACE_Y + wave_height;
        let height_over = effective_height - water_zone.spillover_edge;
        if height_over > 0.0 {
            total_cresting += height_over;
            total_cells += 1;
        }
    }

    // Check back edge (-Z)
    for x in 1..wave_grid.size - 1 {
        let wave_height = wave_grid.heights[back_edge_z][x];
        let effective_height = POOL_SURFACE_Y + wave_height;
        let height_over = effective_height - water_zone.spillover_edge;
        if height_over > 0.0 {
            total_cresting += height_over;
            total_cells += 1;
        }
    }

    // Check left edge (-X)
    for z in 1..wave_grid.size - 1 {
        let wave_height = wave_grid.heights[z][left_edge_x];
        let effective_height = POOL_SURFACE_Y + wave_height;
        let height_over = effective_height - water_zone.spillover_edge;
        if height_over > 0.0 {
            total_cresting += height_over;
            total_cells += 1;
        }
    }

    // Check right edge (+X)
    for z in 1..wave_grid.size - 1 {
        let wave_height = wave_grid.heights[z][right_edge_x];
        let effective_height = POOL_SURFACE_Y + wave_height;
        let height_over = effective_height - water_zone.spillover_edge;
        if height_over > 0.0 {
            total_cresting += height_over;
            total_cells += 1;
        }
    }

    let total_edge_cells = ((wave_grid.size - 2) * 4) as f32;
    let normalized_crest = (total_cresting / total_edge_cells).clamp(0.0, 1.0);
    let is_cresting = total_cells >= 2 && normalized_crest > 0.005;

    if should_log {
        println!("=== SPILLOVER DEBUG (all edges) ===");
        println!("  Cresting cells: {}/{}, Is cresting: {}", total_cells, total_edge_cells as i32, is_cresting);
    }

    // Spawn particles at cresting locations
    if is_cresting {
        if let Some(ref res) = spillover_res {
            static mut SPAWN_ACCUMULATOR: f32 = 0.0;
            let spawn_rate = 20.0 + normalized_crest * 40.0;

            unsafe {
                SPAWN_ACCUMULATOR += spawn_rate * dt;

                while SPAWN_ACCUMULATOR >= 1.0 {
                    SPAWN_ACCUMULATOR -= 1.0;

                    // Pick a random cresting cell from any edge
                    let edge_choice = rand::random::<u32>() % 4;

                    match edge_choice {
                        0 => {
                            // Front edge (+Z)
                            let x = 1 + (rand::random::<usize>() % (wave_grid.size - 2));
                            let wave_height = wave_grid.heights[front_edge_z][x];
                            if POOL_SURFACE_Y + wave_height > water_zone.spillover_edge {
                                let world_x = water_zone.bounds_min.x + x as f32 * cell_width;
                                let spawn_pos = Vec3::new(world_x, water_zone.spillover_edge, water_zone.bounds_max.z);
                                spawn_edge_particles(&mut commands, res, spawn_pos, Vec3::new(0.0, 0.0, 1.0), normalized_crest);
                            }
                        }
                        1 => {
                            // Back edge (-Z)
                            let x = 1 + (rand::random::<usize>() % (wave_grid.size - 2));
                            let wave_height = wave_grid.heights[back_edge_z][x];
                            if POOL_SURFACE_Y + wave_height > water_zone.spillover_edge {
                                let world_x = water_zone.bounds_min.x + x as f32 * cell_width;
                                let spawn_pos = Vec3::new(world_x, water_zone.spillover_edge, water_zone.bounds_min.z);
                                spawn_edge_particles(&mut commands, res, spawn_pos, Vec3::new(0.0, 0.0, -1.0), normalized_crest);
                            }
                        }
                        2 => {
                            // Left edge (-X)
                            let z = 1 + (rand::random::<usize>() % (wave_grid.size - 2));
                            let wave_height = wave_grid.heights[z][left_edge_x];
                            if POOL_SURFACE_Y + wave_height > water_zone.spillover_edge {
                                let world_z = water_zone.bounds_min.z + z as f32 * cell_depth;
                                let spawn_pos = Vec3::new(water_zone.bounds_min.x, water_zone.spillover_edge, world_z);
                                spawn_edge_particles(&mut commands, res, spawn_pos, Vec3::new(-1.0, 0.0, 0.0), normalized_crest);
                            }
                        }
                        _ => {
                            // Right edge (+X)
                            let z = 1 + (rand::random::<usize>() % (wave_grid.size - 2));
                            let wave_height = wave_grid.heights[z][right_edge_x];
                            if POOL_SURFACE_Y + wave_height > water_zone.spillover_edge {
                                let world_z = water_zone.bounds_min.z + z as f32 * cell_depth;
                                let spawn_pos = Vec3::new(water_zone.bounds_max.x, water_zone.spillover_edge, world_z);
                                spawn_edge_particles(&mut commands, res, spawn_pos, Vec3::new(1.0, 0.0, 0.0), normalized_crest);
                            }
                        }
                    }
                }
            }
        }
    }

    // Update existing particles
    let gravity = -15.0;
    let ground_y = -0.5; // Despawn below this

    for (entity, mut particle, mut transform) in particle_query.iter_mut() {
        // Apply gravity
        particle.velocity.y += gravity * dt;

        // Update position
        transform.translation += particle.velocity * dt;

        // Update lifetime
        particle.lifetime -= dt;

        // Despawn if expired or below ground
        if particle.lifetime <= 0.0 || transform.translation.y < ground_y {
            commands.entity(entity).despawn();
        }
    }
}

fn camera_follow(
    player_query: Query<&PhysicsBody, With<Player>>,
    mut camera_query: Query<&mut Transform, (With<MainCamera>, Without<Player>)>,
) {
    let Ok(body) = player_query.get_single() else { return };
    let Ok(mut camera_transform) = camera_query.get_single_mut() else { return };

    let target_pos = body.position + Vec3::new(0.0, 15.0, 25.0);
    camera_transform.translation = camera_transform.translation.lerp(target_pos, 0.03);
    camera_transform.look_at(body.position, Vec3::Y);
}

fn update_ui(
    player_query: Query<&PhysicsBody, With<Player>>,
    mut text_query: Query<&mut Text, With<UIText>>,
    physics: Res<PhysicsState>,
) {
    let Ok(body) = player_query.get_single() else { return };
    let Ok(mut text) = text_query.get_single_mut() else { return };

    let water_info = if let Some(ref water) = physics.water_zone {
        format!("\nWater Level: {:.2} (spillover at {:.2})", water.current_level, water.spillover_edge)
    } else {
        String::new()
    };

    text.0 = format!(
        "Rolling Ball Playground\nWASD: Move | Space: Jump | R: Reset\nPosition: ({:.1}, {:.1}, {:.1})\nVelocity: {:.1}{}",
        body.position.x, body.position.y, body.position.z,
        body.velocity.length(),
        water_info
    );
}
