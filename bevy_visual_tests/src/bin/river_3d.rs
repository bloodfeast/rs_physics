//! River 3D - shallow-water river in a meandering valley
//!
//! A 160 m x 120 m valley on 1 m cells, simulated with `rs_physics`'s shallow-water
//! solver. A river enters at the west edge at 25 m³/s, winds down the valley, fills a
//! hollow on the way into a lake, and leaves at the east edge. The water mesh is the
//! solver's `bed + depth`, coloured by depth and flecked white where it runs fast; the
//! logs float on `sample()`'s surface height and drift with its current.
//!
//! Controls:
//! - C: Blast a crater in the next spot along the river (terrain changes under the water)
//! - F: Flood - quadruple the inflow for 15 seconds
//! - T: Burst a tank of 3000 m³ on the hillside
//! - Space: Pause / resume
//! - Left/Right, Up/Down: Orbit and tilt the camera; Q/E: zoom
//! - R: Reset
//! - Escape: Exit

use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::render::render_asset::RenderAssetUsages;
use rs_physics::fluid_dynamics::{Boundary, Edge, ShallowWater, Threading};
use std::f64::consts::PI;

const NX: usize = 160;
const NZ: usize = 120;
const DX: f64 = 1.0;
const INFLOW: f64 = 25.0;
const CHANNEL_HALF_WIDTH: f64 = 6.0;
/// Depth below which a cell is drawn as dry.
const WET: f64 = 0.02;
const LOGS: usize = 12;

/// Centre line of the river channel, z as a function of x.
fn channel_z(x: f64) -> f64 {
    60.0 + 22.0 * (2.0 * PI * x / 110.0).sin()
}

/// The valley: a gentle fall to the east, a carved channel, hills either side and a
/// hollow two-thirds of the way down that the river fills into a lake.
fn terrain(x: f64, z: f64) -> f64 {
    let fall = 0.012 * (NX as f64 * DX - x);
    let off = (z - channel_z(x)).abs();
    let carve = if off < CHANNEL_HALF_WIDTH {
        3.0 * (1.0 - (off / CHANNEL_HALF_WIDTH).powi(2))
    } else {
        0.0
    };
    let hills = 0.004 * (off - CHANNEL_HALF_WIDTH).max(0.0).powi(2);
    let (lx, lz) = (x - 105.0, z - 62.0);
    let hollow = 3.5 * (-(lx * lx / 300.0 + lz * lz / 220.0)).exp();
    let bumps = 0.25 * (0.31 * x).sin() * (0.27 * z).cos();
    fall - carve + hills.min(14.0) - hollow + bumps
}

fn build_river() -> ShallowWater {
    let bed: Vec<f64> = (0..NZ)
        .flat_map(|j| (0..NX).map(move |i| terrain((i as f64 + 0.5) * DX, (j as f64 + 0.5) * DX)))
        .collect();
    let mut river = ShallowWater::new(NX, NZ, DX, bed)
        .expect("valid grid")
        .with_manning(0.035)
        .expect("valid roughness")
        // On the calling thread, as a game would run it off its main loop; the answer is
        // the same bits as on rayon's pool.
        .with_threading(Threading::Serial);
    set_inflow(&mut river, INFLOW);
    river
        .set_boundary(Edge::MaxX, 0..NZ, Boundary::Open)
        .expect("valid boundary");
    river
}

fn inflow_cells() -> std::ops::Range<usize> {
    let centre = channel_z(0.0);
    let lo = ((centre - CHANNEL_HALF_WIDTH) / DX).floor().max(0.0) as usize;
    let hi = ((centre + CHANNEL_HALF_WIDTH) / DX).ceil().min(NZ as f64) as usize;
    lo..hi
}

fn set_inflow(river: &mut ShallowWater, discharge: f64) {
    river
        .set_boundary(Edge::MinX, inflow_cells(), Boundary::Inflow { discharge })
        .expect("valid inflow");
}

#[derive(Resource)]
struct Sim {
    river: ShallowWater,
    paused: bool,
    flood_left: f32,
    craters: usize,
    last_substeps: usize,
    last_ms: f32,
}

#[derive(Resource)]
struct MeshHandles {
    water: Handle<Mesh>,
    terrain: Handle<Mesh>,
}

#[derive(Component)]
struct Log {
    x: f64,
    z: f64,
    spin: f32,
}

#[derive(Component)]
struct Hud;

#[derive(Component)]
struct OrbitCamera {
    yaw: f32,
    pitch: f32,
    distance: f32,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "River 3D - Shallow Water".to_string(),
                resolution: (1280.0, 720.0).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.62, 0.74, 0.86)))
        .insert_resource(Sim {
            river: build_river(),
            paused: false,
            flood_left: 0.0,
            craters: 0,
            last_substeps: 0,
            last_ms: 0.0,
        })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (controls, step_river, sync_water, sync_terrain, float_logs, orbit_camera, update_hud)
                .chain(),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    sim: Res<Sim>,
) {
    let river = &sim.river;
    let terrain = meshes.add(grid_mesh(river, |k| river.beds()[k] as f32, terrain_colour));
    let water = meshes.add(grid_mesh(river, |k| water_height(river, k), |_, _| [0.2, 0.4, 0.6, 0.8]));

    commands.spawn((
        Mesh3d(terrain.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.95,
            ..default()
        })),
        Transform::default(),
    ));
    commands.spawn((
        Mesh3d(water.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            alpha_mode: AlphaMode::Blend,
            perceptual_roughness: 0.08,
            reflectance: 0.6,
            ..default()
        })),
        Transform::default(),
    ));
    commands.insert_resource(MeshHandles { water, terrain });

    let log_mesh = meshes.add(Cuboid::new(2.4, 0.35, 0.35));
    let log_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.42, 0.27, 0.14),
        perceptual_roughness: 0.9,
        ..default()
    });
    for n in 0..LOGS {
        commands.spawn((
            Mesh3d(log_mesh.clone()),
            MeshMaterial3d(log_material.clone()),
            Transform::default(),
            Log { x: -1.0 - 9.0 * n as f64, z: channel_z(0.0), spin: 0.0 },
        ));
    }

    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(40.0, 120.0, 30.0).looking_at(Vec3::new(80.0, 0.0, 60.0), Vec3::Y),
    ));
    commands.spawn((
        Camera3d::default(),
        Transform::default(),
        OrbitCamera { yaw: -0.6, pitch: 0.75, distance: 170.0 },
    ));
    commands.spawn((
        Text::new(""),
        TextFont { font_size: 18.0, ..default() },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
        Hud,
    ));
}

/// Draw height of the water at cell `k`: the surface where wet, tucked under the ground
/// where dry so the terrain hides it.
fn water_height(river: &ShallowWater, k: usize) -> f32 {
    let (h, b) = (river.depths()[k], river.beds()[k]);
    if h > WET {
        (b + h) as f32
    } else {
        (b - 0.3) as f32
    }
}

fn terrain_colour(river: &ShallowWater, k: usize) -> [f32; 4] {
    let b = river.beds()[k] as f32;
    let t = ((b + 2.0) / 14.0).clamp(0.0, 1.0);
    // Mud at the bottom, grass on the slopes, dry grass up high.
    let mud = [0.36, 0.30, 0.22];
    let grass = [0.28, 0.46, 0.20];
    let dry = [0.62, 0.58, 0.38];
    let mix = |a: [f32; 3], c: [f32; 3], s: f32| [0, 1, 2].map(|i| a[i] + (c[i] - a[i]) * s);
    let c = if t < 0.3 { mix(mud, grass, t / 0.3) } else { mix(grass, dry, (t - 0.3) / 0.7) };
    [c[0], c[1], c[2], 1.0]
}

fn water_colour(river: &ShallowWater, k: usize) -> [f32; 4] {
    let h = river.depths()[k];
    if h <= WET {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let speed = (river.discharges_x()[k].hypot(river.discharges_z()[k]) / h) as f32;
    let deep = (h as f32 / 3.0).clamp(0.0, 1.0);
    let foam = ((speed - 1.2) / 1.5).clamp(0.0, 0.7);
    let base = [0.30 - 0.22 * deep, 0.55 - 0.30 * deep, 0.62 - 0.18 * deep];
    let c = [0, 1, 2].map(|i| base[i] + (0.92 - base[i]) * foam);
    [c[0], c[1], c[2], 0.55 + 0.35 * deep.max(foam)]
}

/// A mesh with one vertex per cell centre.
fn grid_mesh(
    river: &ShallowWater,
    height: impl Fn(usize) -> f32,
    colour: impl Fn(&ShallowWater, usize) -> [f32; 4],
) -> Mesh {
    let (nx, nz) = (river.nx(), river.nz());
    let mut positions = Vec::with_capacity(nx * nz);
    let mut colours = Vec::with_capacity(nx * nz);
    for j in 0..nz {
        for i in 0..nx {
            let k = j * nx + i;
            let (x, z) = ((i as f64 + 0.5) * DX, (j as f64 + 0.5) * DX);
            positions.push([x as f32, height(k), z as f32]);
            colours.push(colour(river, k));
        }
    }
    let normals = grid_normals(&positions, nx, nz);
    let mut indices = Vec::with_capacity((nx - 1) * (nz - 1) * 6);
    for j in 0..nz - 1 {
        for i in 0..nx - 1 {
            let k = (j * nx + i) as u32;
            let (right, below) = (k + 1, k + nx as u32);
            indices.extend_from_slice(&[k, below, right, right, below, below + 1]);
        }
    }
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colours)
        .with_inserted_indices(Indices::U32(indices))
}

fn grid_normals(positions: &[[f32; 3]], nx: usize, nz: usize) -> Vec<[f32; 3]> {
    (0..nz)
        .flat_map(|j| (0..nx).map(move |i| (i, j)))
        .map(|(i, j)| {
            let y = |i: usize, j: usize| positions[j * nx + i][1];
            let dx = y((i + 1).min(nx - 1), j) - y(i.saturating_sub(1), j);
            let dz = y(i, (j + 1).min(nz - 1)) - y(i, j.saturating_sub(1));
            let n = Vec3::new(-dx, 2.0 * DX as f32, -dz).normalize();
            [n.x, n.y, n.z]
        })
        .collect()
}

fn controls(
    keys: Res<ButtonInput<KeyCode>>,
    mut sim: ResMut<Sim>,
    mut logs: Query<&mut Log>,
    mut exit: EventWriter<AppExit>,
) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.send(AppExit::Success);
    }
    if keys.just_pressed(KeyCode::Space) {
        sim.paused = !sim.paused;
    }
    if keys.just_pressed(KeyCode::KeyR) {
        sim.river = build_river();
        sim.flood_left = 0.0;
        sim.craters = 0;
        for (n, mut log) in logs.iter_mut().enumerate() {
            log.x = -1.0 - 9.0 * n as f64;
            log.z = channel_z(0.0);
        }
    }
    if keys.just_pressed(KeyCode::KeyF) {
        sim.flood_left = 15.0;
        set_inflow(&mut sim.river, 4.0 * INFLOW);
    }
    if keys.just_pressed(KeyCode::KeyT) {
        // A tank on the north hillside: 3000 m³ over a 10 x 10 patch.
        for j in 95..105 {
            for i in 40..50 {
                sim.river.add_water(i, j, 30.0).expect("inside the grid");
            }
        }
    }
    if keys.just_pressed(KeyCode::KeyC) {
        // A crater 4 m deep and 5 m across, stepping down the river each press.
        let cx = 25.0 + 22.0 * (sim.craters % 6) as f64;
        let cz = channel_z(cx) + 4.0;
        sim.craters += 1;
        for j in 0..NZ {
            for i in 0..NX {
                let (x, z) = ((i as f64 + 0.5) * DX, (j as f64 + 0.5) * DX);
                let r2 = (x - cx).powi(2) + (z - cz).powi(2);
                if r2 < 25.0 {
                    let bed = sim.river.bed(i, j) - 4.0 * (1.0 - r2 / 25.0);
                    sim.river.set_bed(i, j, bed).expect("finite");
                }
            }
        }
    }
}

fn step_river(time: Res<Time>, mut sim: ResMut<Sim>) {
    if sim.paused {
        return;
    }
    // A long frame (a window drag, a breakpoint) is capped rather than simulated in one
    // go; the solver would handle it, but nobody wants to watch it catch up.
    let dt = time.delta_secs().min(1.0 / 20.0);
    if sim.flood_left > 0.0 {
        sim.flood_left -= dt;
        if sim.flood_left <= 0.0 {
            set_inflow(&mut sim.river, INFLOW);
        }
    }
    let start = std::time::Instant::now();
    match sim.river.step(dt as f64) {
        Ok(substeps) => sim.last_substeps = substeps,
        Err(e) => {
            warn!("river step failed: {e:?}");
            sim.paused = true;
        }
    }
    sim.last_ms = start.elapsed().as_secs_f32() * 1000.0;
}

fn sync_water(sim: Res<Sim>, handles: Res<MeshHandles>, mut meshes: ResMut<Assets<Mesh>>) {
    let Some(mesh) = meshes.get_mut(&handles.water) else { return };
    let river = &sim.river;
    let Some(VertexAttributeValues::Float32x3(positions)) = mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
    else {
        return;
    };
    for (k, p) in positions.iter_mut().enumerate() {
        p[1] = water_height(river, k);
    }
    let normals = grid_normals(positions, river.nx(), river.nz());
    let colours: Vec<[f32; 4]> = (0..river.nx() * river.nz()).map(|k| water_colour(river, k)).collect();
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
}

/// Terrain only changes when a crater is blasted; rebuild it when the crater count moves.
fn sync_terrain(
    sim: Res<Sim>,
    handles: Res<MeshHandles>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut drawn: Local<Option<usize>>,
) {
    if *drawn == Some(sim.craters) {
        return;
    }
    *drawn = Some(sim.craters);
    let river = &sim.river;
    if let Some(mesh) = meshes.get_mut(&handles.terrain) {
        *mesh = grid_mesh(river, |k| river.beds()[k] as f32, terrain_colour);
    }
}

/// Logs ride the surface and relax towards the current, the way a light floating body
/// with drag does. Stranded or finished logs go back to the source.
fn float_logs(time: Res<Time>, sim: Res<Sim>, mut logs: Query<(&mut Log, &mut Transform)>) {
    if sim.paused {
        return;
    }
    let dt = time.delta_secs().min(1.0 / 20.0) as f64;
    let river = &sim.river;
    for (mut log, mut transform) in logs.iter_mut() {
        if log.x < 0.5 {
            // Queued upstream: feed in one at a time.
            log.x += 2.0 * dt;
            transform.translation = Vec3::new(-50.0, -50.0, 0.0);
            continue;
        }
        let here = river.sample(log.x, log.z);
        if here.depth < 0.15 || log.x > NX as f64 * DX - 1.0 {
            log.x = -1.0 - 20.0 * (log.spin.abs() as f64 % 3.0);
            log.z = channel_z(0.0);
            continue;
        }
        log.x += here.velocity[0] * dt;
        log.z += here.velocity[1] * dt;
        log.spin += (here.velocity[1] * dt * 0.8) as f32;
        let heading = (here.velocity[1] as f32).atan2(here.velocity[0] as f32);
        transform.translation = Vec3::new(log.x as f32, here.surface as f32 + 0.05, log.z as f32);
        transform.rotation = Quat::from_rotation_y(-heading) * Quat::from_rotation_x(log.spin);
    }
}

fn orbit_camera(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut cameras: Query<(&mut OrbitCamera, &mut Transform)>,
) {
    let dt = time.delta_secs();
    for (mut orbit, mut transform) in cameras.iter_mut() {
        if keys.pressed(KeyCode::ArrowLeft) {
            orbit.yaw -= dt;
        }
        if keys.pressed(KeyCode::ArrowRight) {
            orbit.yaw += dt;
        }
        if keys.pressed(KeyCode::ArrowUp) {
            orbit.pitch = (orbit.pitch + dt * 0.6).min(1.45);
        }
        if keys.pressed(KeyCode::ArrowDown) {
            orbit.pitch = (orbit.pitch - dt * 0.6).max(0.15);
        }
        if keys.pressed(KeyCode::KeyQ) {
            orbit.distance = (orbit.distance - 60.0 * dt).max(30.0);
        }
        if keys.pressed(KeyCode::KeyE) {
            orbit.distance = (orbit.distance + 60.0 * dt).min(320.0);
        }
        let target = Vec3::new(NX as f32 * DX as f32 / 2.0, 0.0, NZ as f32 * DX as f32 / 2.0);
        let offset = Vec3::new(
            orbit.yaw.cos() * orbit.pitch.cos(),
            orbit.pitch.sin(),
            orbit.yaw.sin() * orbit.pitch.cos(),
        ) * orbit.distance;
        *transform = Transform::from_translation(target + offset).looking_at(target, Vec3::Y);
    }
}

fn update_hud(sim: Res<Sim>, mut hud: Query<&mut Text, With<Hud>>) {
    let Ok(mut text) = hud.get_single_mut() else { return };
    let river = &sim.river;
    let wet = river.depths().iter().filter(|&&h| h > WET).count();
    text.0 = format!(
        "River 3D - shallow water, {}x{} cells of {} m\n\
         t = {:.0} s   water {:.0} m3   in {:.0} m3   out {:.0} m3   wet {:.0}%\n\
         step {:.2} ms, {} substep(s){}{}\n\
         C crater | F flood | T burst tank | Space pause | arrows/QE camera | R reset",
        river.nx(),
        river.nz(),
        river.cell_size(),
        river.time(),
        river.total_volume(),
        river.volume_in(),
        river.volume_out(),
        100.0 * wet as f64 / (river.nx() * river.nz()) as f64,
        sim.last_ms,
        sim.last_substeps,
        if sim.flood_left > 0.0 { format!("   FLOOD {:.0} s", sim.flood_left) } else { String::new() },
        if sim.paused { "   PAUSED" } else { "" },
    );
}
