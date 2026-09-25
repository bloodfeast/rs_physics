//! GPU-accelerated 3D N-body gravitational simulation
//!
//! This module provides a GPU compute shader implementation for 3D
//! N-body gravitational simulation, suitable for cosmological simulations
//! where particles represent galaxies, dark matter halos, or stars.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::GpuContext;

/// 3D N-body particle data stored on GPU
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct NBody3DParticle {
    /// Position (x, y, z)
    pub pos: [f32; 3],
    /// Mass
    pub mass: f32,
    /// Velocity (vx, vy, vz)
    pub vel: [f32; 3],
    /// Padding for alignment (total 32 bytes)
    pub _padding: f32,
}

impl NBody3DParticle {
    /// Create a new 3D particle
    pub fn new(pos: [f32; 3], vel: [f32; 3], mass: f32) -> Self {
        Self {
            pos,
            mass,
            vel,
            _padding: 0.0,
        }
    }
}

/// Simulation parameters for 3D N-body
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct NBody3DParams {
    /// Time step
    dt: f32,
    /// Gravitational constant
    g: f32,
    /// Softening factor (prevents singularity at r=0)
    softening: f32,
    /// Number of particles
    num_particles: u32,
    /// Box size for periodic boundaries (0 = no periodic boundaries)
    box_size: f32,
    /// Padding
    _padding: [f32; 3],
}

/// GPU-accelerated 3D N-body gravitational simulation
///
/// Uses a direct O(N²) approach which is highly parallelizable on GPUs.
/// Each particle computes its gravitational force from all other particles.
///
/// # Features
///
/// - Full 3D gravitational simulation
/// - Each particle has individual mass
/// - Optional periodic boundary conditions for cosmological simulations
/// - Softening parameter to prevent numerical singularities
///
/// # Example
///
/// ```ignore
/// use rs_physics::gpu::{GpuContext, GpuNBody3DSimulation, NBody3DParticle};
///
/// let gpu = GpuContext::new().unwrap();
///
/// // Create a galaxy-like distribution
/// let particles: Vec<NBody3DParticle> = (0..10000)
///     .map(|i| {
///         let angle = (i as f32) * 0.1;
///         let r = (i as f32).sqrt() * 0.5;
///         NBody3DParticle::new(
///             [angle.cos() * r, (i as f32 * 0.01).sin() * 0.1, angle.sin() * r],
///             [-angle.sin() * 0.1, 0.0, angle.cos() * 0.1],
///             1.0,
///         )
///     })
///     .collect();
///
/// let mut sim = GpuNBody3DSimulation::new(&gpu, &particles, 0.001, 1.0, 0.1, None);
///
/// for _ in 0..1000 {
///     sim.step(&gpu);
/// }
///
/// let results = sim.read_particles(&gpu);
/// ```
pub struct GpuNBody3DSimulation {
    /// Current particle state
    particle_buffer_a: wgpu::Buffer,
    /// Next particle state (ping-pong)
    particle_buffer_b: wgpu::Buffer,
    /// Parameters
    params_buffer: wgpu::Buffer,
    /// Bind groups for ping-pong (A->B and B->A)
    bind_group_a: wgpu::BindGroup,
    bind_group_b: wgpu::BindGroup,
    /// Which buffer is current (true = A, false = B)
    current_is_a: bool,
    /// Compute pipeline
    pipeline: wgpu::ComputePipeline,
    /// Number of particles
    num_particles: u32,
    /// Workgroup size
    workgroup_size: u32,
    /// Box size for periodic boundaries (None = no periodic)
    box_size: Option<f32>,
}

impl GpuNBody3DSimulation {
    /// Create a new 3D N-body simulation
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context
    /// * `particles` - Initial particle data
    /// * `dt` - Time step per simulation step
    /// * `g` - Gravitational constant (use ~6.67e-11 for SI units, or 1.0 for normalized)
    /// * `softening` - Softening factor to prevent singularities (typically 0.01-0.1)
    /// * `box_size` - Optional box size for periodic boundary conditions
    pub fn new(
        gpu: &GpuContext,
        particles: &[NBody3DParticle],
        dt: f32,
        g: f32,
        softening: f32,
        box_size: Option<f32>,
    ) -> Self {
        let num_particles = particles.len() as u32;
        let workgroup_size = 256u32;

        // Create double buffers for ping-pong
        let particle_buffer_a = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("NBody3D Particle Buffer A"),
            contents: bytemuck::cast_slice(particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let particle_buffer_b = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("NBody3D Particle Buffer B"),
            contents: bytemuck::cast_slice(particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        // Create params buffer
        let params = NBody3DParams {
            dt,
            g,
            softening,
            num_particles,
            box_size: box_size.unwrap_or(0.0),
            _padding: [0.0; 3],
        };
        let params_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("NBody3D Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create shader module
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("NBody3D Shader"),
            source: wgpu::ShaderSource::Wgsl(NBODY_3D_SHADER.into()),
        });

        // Create bind group layout
        let bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("NBody3D Bind Group Layout"),
            entries: &[
                // Input particles (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output particles (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create bind groups for ping-pong
        let bind_group_a = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("NBody3D Bind Group A->B"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particle_buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let bind_group_b = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("NBody3D Bind Group B->A"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particle_buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("NBody3D Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Create compute pipeline
        let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NBody3D Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            particle_buffer_a,
            particle_buffer_b,
            params_buffer,
            bind_group_a,
            bind_group_b,
            current_is_a: true,
            pipeline,
            num_particles,
            workgroup_size,
            box_size,
        }
    }

    /// Run one simulation step on the GPU
    pub fn step(&mut self, gpu: &GpuContext) {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("NBody3D Step Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("NBody3D Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);

            if self.current_is_a {
                compute_pass.set_bind_group(0, &self.bind_group_a, &[]);
            } else {
                compute_pass.set_bind_group(0, &self.bind_group_b, &[]);
            }

            let num_workgroups = (self.num_particles + self.workgroup_size - 1) / self.workgroup_size;
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the device was lost while waiting for the GPU");

        // Swap buffers
        self.current_is_a = !self.current_is_a;
    }

    /// Run multiple simulation steps
    pub fn step_n(&mut self, gpu: &GpuContext, steps: u32) {
        for _ in 0..steps {
            self.step(gpu);
        }
    }

    /// Read particle data back from GPU
    pub fn read_particles(&self, gpu: &GpuContext) -> Vec<NBody3DParticle> {
        let buffer_size = (self.num_particles as usize) * std::mem::size_of::<NBody3DParticle>();

        let source_buffer = if self.current_is_a {
            &self.particle_buffer_a
        } else {
            &self.particle_buffer_b
        };

        let staging_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NBody3D Staging Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("NBody3D Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(source_buffer, 0, &staging_buffer, 0, buffer_size as u64);
        gpu.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the device was lost while waiting for the GPU");
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice
            .get_mapped_range()
            .expect("the readback buffer was mapped by the callback above");
        let particles: Vec<NBody3DParticle> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        particles
    }

    /// Update simulation parameters
    pub fn set_params(&self, gpu: &GpuContext, dt: f32, g: f32, softening: f32) {
        let params = NBody3DParams {
            dt,
            g,
            softening,
            num_particles: self.num_particles,
            box_size: self.box_size.unwrap_or(0.0),
            _padding: [0.0; 3],
        };
        gpu.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    /// Get the number of particles
    pub fn num_particles(&self) -> u32 {
        self.num_particles
    }

    /// Check if periodic boundaries are enabled
    pub fn has_periodic_boundaries(&self) -> bool {
        self.box_size.is_some()
    }

    /// Calculate total kinetic energy of the system
    pub fn kinetic_energy(&self, gpu: &GpuContext) -> f32 {
        let particles = self.read_particles(gpu);
        particles.iter().map(|p| {
            let v_sq = p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1] + p.vel[2] * p.vel[2];
            0.5 * p.mass * v_sq
        }).sum()
    }

    /// Calculate total potential energy of the system
    pub fn potential_energy(&self, gpu: &GpuContext, g: f32, softening: f32) -> f32 {
        let particles = self.read_particles(gpu);
        let mut pe = 0.0f32;
        for i in 0..particles.len() {
            for j in (i + 1)..particles.len() {
                let dx = particles[j].pos[0] - particles[i].pos[0];
                let dy = particles[j].pos[1] - particles[i].pos[1];
                let dz = particles[j].pos[2] - particles[i].pos[2];
                let r = (dx * dx + dy * dy + dz * dz + softening * softening).sqrt();
                pe -= g * particles[i].mass * particles[j].mass / r;
            }
        }
        pe
    }
}

/// WGSL compute shader for 3D N-body gravitational simulation
const NBODY_3D_SHADER: &str = r#"
struct Particle {
    pos: vec3<f32>,
    mass: f32,
    vel: vec3<f32>,
    _padding: f32,
}

struct Params {
    dt: f32,
    g: f32,
    softening: f32,
    num_particles: u32,
    box_size: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<storage, read> particles_in: array<Particle>;
@group(0) @binding(1) var<storage, read_write> particles_out: array<Particle>;
@group(0) @binding(2) var<uniform> params: Params;

// Shared memory for tile-based optimization
var<workgroup> tile: array<Particle, 256>;

// Wrap position for periodic boundaries
fn wrap_position(pos: vec3<f32>, box_size: f32) -> vec3<f32> {
    if (box_size <= 0.0) {
        return pos;
    }
    var wrapped = pos;
    wrapped.x = pos.x - box_size * floor(pos.x / box_size);
    wrapped.y = pos.y - box_size * floor(pos.y / box_size);
    wrapped.z = pos.z - box_size * floor(pos.z / box_size);
    return wrapped;
}

// Get minimum image displacement for periodic boundaries
fn min_image_displacement(r: vec3<f32>, box_size: f32) -> vec3<f32> {
    if (box_size <= 0.0) {
        return r;
    }
    var d = r;
    let half_box = box_size * 0.5;
    if (d.x > half_box) { d.x = d.x - box_size; }
    if (d.x < -half_box) { d.x = d.x + box_size; }
    if (d.y > half_box) { d.y = d.y - box_size; }
    if (d.y < -half_box) { d.y = d.y + box_size; }
    if (d.z > half_box) { d.z = d.z - box_size; }
    if (d.z < -half_box) { d.z = d.z + box_size; }
    return d;
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let idx = global_id.x;
    let local_idx = local_id.x;

    // Bounds check
    if (idx >= params.num_particles) {
        return;
    }

    // Load this particle
    let p = particles_in[idx];
    var acc = vec3<f32>(0.0, 0.0, 0.0);

    // Process particles in tiles for better memory access
    let num_tiles = (params.num_particles + 255u) / 256u;

    for (var t = 0u; t < num_tiles; t = t + 1u) {
        // Load tile into shared memory
        let tile_idx = t * 256u + local_idx;
        if (tile_idx < params.num_particles) {
            tile[local_idx] = particles_in[tile_idx];
        }

        // Synchronize workgroup
        workgroupBarrier();

        // Compute forces from particles in this tile
        let tile_end = min(256u, params.num_particles - t * 256u);
        for (var j = 0u; j < tile_end; j = j + 1u) {
            let other_idx = t * 256u + j;

            // Skip self-interaction
            if (other_idx == idx) {
                continue;
            }

            let other = tile[j];

            // Compute displacement (with minimum image for periodic boundaries)
            var r = other.pos - p.pos;
            r = min_image_displacement(r, params.box_size);

            // Compute distance squared with softening
            let dist_sq = dot(r, r) + params.softening * params.softening;

            // Compute inverse distance cubed
            let inv_dist = inverseSqrt(dist_sq);
            let inv_dist_cubed = inv_dist * inv_dist * inv_dist;

            // Accumulate acceleration: a = G * m_other * r / |r|^3
            acc = acc + r * (params.g * other.mass * inv_dist_cubed);
        }

        // Synchronize before loading next tile
        workgroupBarrier();
    }

    // Update velocity and position using leapfrog integration
    let new_vel = p.vel + acc * params.dt;
    var new_pos = p.pos + new_vel * params.dt;

    // Apply periodic boundary conditions
    new_pos = wrap_position(new_pos, params.box_size);

    // Store result
    var result: Particle;
    result.pos = new_pos;
    result.mass = p.mass;
    result.vel = new_vel;
    result._padding = 0.0;
    particles_out[idx] = result;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nbody_3d_creation() {
        let gpu = match GpuContext::new() {
            Some(g) => g,
            None => {
                println!("No GPU available, skipping test");
                return;
            }
        };

        let particles: Vec<NBody3DParticle> = (0..100)
            .map(|i| {
                let angle = (i as f32) * std::f32::consts::PI * 2.0 / 100.0;
                NBody3DParticle::new(
                    [angle.cos() * 10.0, 0.0, angle.sin() * 10.0],
                    [0.0, 0.0, 0.0],
                    1.0,
                )
            })
            .collect();

        let sim = GpuNBody3DSimulation::new(&gpu, &particles, 0.01, 1.0, 0.1, None);
        assert_eq!(sim.num_particles(), 100);
    }

    #[test]
    fn test_nbody_3d_two_body() {
        let gpu = match GpuContext::new() {
            Some(g) => g,
            None => {
                println!("No GPU available, skipping test");
                return;
            }
        };

        // Two bodies attracting each other along x-axis
        let particles = vec![
            NBody3DParticle::new([-5.0, 0.0, 0.0], [0.0, 0.0, 0.0], 100.0),
            NBody3DParticle::new([5.0, 0.0, 0.0], [0.0, 0.0, 0.0], 100.0),
        ];

        let mut sim = GpuNBody3DSimulation::new(&gpu, &particles, 0.01, 1.0, 0.1, None);

        sim.step_n(&gpu, 100);

        let result = sim.read_particles(&gpu);

        // Particles should have moved toward each other
        assert!(result[0].pos[0] > -5.0, "Left particle should move right");
        assert!(result[1].pos[0] < 5.0, "Right particle should move left");

        // Velocities should be toward each other
        assert!(result[0].vel[0] > 0.0, "Left particle should have positive x velocity");
        assert!(result[1].vel[0] < 0.0, "Right particle should have negative x velocity");
    }

    #[test]
    fn test_nbody_3d_circular_orbit() {
        let gpu = match GpuContext::new() {
            Some(g) => g,
            None => {
                println!("No GPU available, skipping test");
                return;
            }
        };

        // Central massive body and orbiting body in x-z plane
        let g = 1.0f32;
        let m_central = 1000.0f32;
        let r = 10.0f32;
        let v_orbit = (g * m_central / r).sqrt();

        let particles = vec![
            NBody3DParticle::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], m_central),
            NBody3DParticle::new([r, 0.0, 0.0], [0.0, 0.0, v_orbit], 1.0),
        ];

        let mut sim = GpuNBody3DSimulation::new(&gpu, &particles, 0.001, g, 0.01, None);

        sim.step_n(&gpu, 1000);

        let result = sim.read_particles(&gpu);

        // Orbiting body should still be roughly at distance r
        let dx = result[1].pos[0] - result[0].pos[0];
        let dy = result[1].pos[1] - result[0].pos[1];
        let dz = result[1].pos[2] - result[0].pos[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        assert!(
            dist > r * 0.8 && dist < r * 1.2,
            "Orbiting body should maintain roughly circular orbit. Got distance {} (expected ~{})",
            dist, r
        );
    }

    #[test]
    fn test_nbody_3d_periodic_boundaries() {
        let gpu = match GpuContext::new() {
            Some(g) => g,
            None => {
                println!("No GPU available, skipping test");
                return;
            }
        };

        let box_size = 100.0f32;

        // Particle moving fast toward edge
        let particles = vec![
            NBody3DParticle::new([95.0, 50.0, 50.0], [10.0, 0.0, 0.0], 1.0),
            NBody3DParticle::new([50.0, 50.0, 50.0], [0.0, 0.0, 0.0], 1.0),
        ];

        let mut sim = GpuNBody3DSimulation::new(&gpu, &particles, 0.1, 0.0, 0.1, Some(box_size));

        // Run enough steps for particle to cross boundary
        sim.step_n(&gpu, 20);

        let result = sim.read_particles(&gpu);

        // First particle should have wrapped around
        assert!(
            result[0].pos[0] >= 0.0 && result[0].pos[0] < box_size,
            "Particle should be wrapped within box. Got x={}",
            result[0].pos[0]
        );
    }

    #[test]
    fn test_nbody_3d_cosmological_collapse() {
        let gpu = match GpuContext::new() {
            Some(g) => g,
            None => {
                println!("No GPU available, skipping test");
                return;
            }
        };

        // Create a uniform sphere of particles that should collapse under gravity
        let n = 500;
        let mut particles = Vec::with_capacity(n);

        // Use deterministic "random" distribution
        for i in 0..n {
            // Simple deterministic pseudo-random using golden ratio
            let phi = (1.0 + 5.0f32.sqrt()) / 2.0;
            let theta = 2.0 * std::f32::consts::PI * (i as f32) * phi;
            let r = 10.0 * ((i as f32) / (n as f32)).powf(1.0 / 3.0); // Uniform in volume
            let z = 1.0 - 2.0 * (i as f32) / (n as f32);
            let xy = (1.0 - z * z).sqrt();

            particles.push(NBody3DParticle::new(
                [r * xy * theta.cos(), r * xy * theta.sin(), r * z],
                [0.0, 0.0, 0.0],
                1.0,
            ));
        }

        let mut sim = GpuNBody3DSimulation::new(&gpu, &particles, 0.01, 1.0, 0.5, None);

        // Calculate initial center of mass spread
        let initial = sim.read_particles(&gpu);
        let initial_spread: f32 = initial.iter().map(|p| {
            (p.pos[0] * p.pos[0] + p.pos[1] * p.pos[1] + p.pos[2] * p.pos[2]).sqrt()
        }).sum::<f32>() / n as f32;

        // Run simulation
        sim.step_n(&gpu, 200);

        // Calculate final spread
        let final_particles = sim.read_particles(&gpu);
        let final_spread: f32 = final_particles.iter().map(|p| {
            (p.pos[0] * p.pos[0] + p.pos[1] * p.pos[1] + p.pos[2] * p.pos[2]).sqrt()
        }).sum::<f32>() / n as f32;

        println!("Initial avg radius: {}, Final avg radius: {}", initial_spread, final_spread);

        // System should have collapsed (average radius decreased)
        assert!(
            final_spread < initial_spread,
            "System should collapse under gravity. Initial spread: {}, Final spread: {}",
            initial_spread, final_spread
        );
    }

    #[test]
    fn test_nbody_3d_energy_conservation() {
        let gpu = match GpuContext::new() {
            Some(g) => g,
            None => {
                println!("No GPU available, skipping test");
                return;
            }
        };

        // Simple two-body system
        let g = 1.0f32;
        let softening = 0.1f32;

        let particles = vec![
            NBody3DParticle::new([-3.0, 0.0, 0.0], [0.0, 0.5, 0.0], 50.0),
            NBody3DParticle::new([3.0, 0.0, 0.0], [0.0, -0.5, 0.0], 50.0),
        ];

        let mut sim = GpuNBody3DSimulation::new(&gpu, &particles, 0.001, g, softening, None);

        let initial_ke = sim.kinetic_energy(&gpu);
        let initial_pe = sim.potential_energy(&gpu, g, softening);
        let initial_total = initial_ke + initial_pe;

        sim.step_n(&gpu, 500);

        let final_ke = sim.kinetic_energy(&gpu);
        let final_pe = sim.potential_energy(&gpu, g, softening);
        let final_total = final_ke + final_pe;

        let energy_drift = ((final_total - initial_total) / initial_total.abs()).abs();

        println!("Initial: KE={}, PE={}, Total={}", initial_ke, initial_pe, initial_total);
        println!("Final: KE={}, PE={}, Total={}", final_ke, final_pe, final_total);
        println!("Energy drift: {:.2}%", energy_drift * 100.0);

        // Energy should be conserved within 10% (leapfrog is symplectic but not perfect)
        assert!(
            energy_drift < 0.1,
            "Energy should be approximately conserved. Drift: {:.2}%",
            energy_drift * 100.0
        );
    }
}
