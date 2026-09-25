//! GPU-accelerated N-body gravitational simulation
//!
//! This module provides a GPU compute shader implementation for direct
//! N-body gravitational simulation. While this is O(N²), the massive
//! parallelism of GPUs makes it competitive with Barnes-Hut on CPU
//! for up to ~100K particles.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::GpuContext;

/// N-body particle data stored on GPU
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct NBodyParticle {
    /// Position (x, y)
    pub pos: [f32; 2],
    /// Velocity (vx, vy)
    pub vel: [f32; 2],
    /// Mass
    pub mass: f32,
    /// Padding for alignment
    pub _padding: [f32; 3],
}

/// Simulation parameters for N-body
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct NBodyParams {
    /// Time step
    dt: f32,
    /// Gravitational constant
    g: f32,
    /// Softening factor (prevents singularity at r=0)
    softening: f32,
    /// Number of particles
    num_particles: u32,
}

/// GPU-accelerated N-body gravitational simulation
///
/// Uses a direct O(N²) approach which is highly parallelizable on GPUs.
/// Each particle computes its force from all other particles in parallel.
///
/// # Performance
///
/// For N particles:
/// - CPU Barnes-Hut: O(N log N) per step, but serial tree traversal
/// - GPU Direct: O(N²) per step, but massively parallel
///
/// GPU direct typically wins for N < 100,000 particles.
///
/// # Example
///
/// ```ignore
/// use rs_physics::gpu::{GpuContext, GpuNBodySimulation, NBodyParticle};
///
/// let gpu = GpuContext::new().unwrap();
///
/// // Create a simple two-body system
/// let particles = vec![
///     NBodyParticle {
///         pos: [-1.0, 0.0],
///         vel: [0.0, 0.5],
///         mass: 1000.0,
///         _padding: [0.0; 3],
///     },
///     NBodyParticle {
///         pos: [1.0, 0.0],
///         vel: [0.0, -0.5],
///         mass: 1000.0,
///         _padding: [0.0; 3],
///     },
/// ];
///
/// let mut sim = GpuNBodySimulation::new(&gpu, &particles, 0.001, 6.67e-11, 0.01);
///
/// for _ in 0..1000 {
///     sim.step(&gpu);
/// }
/// ```
pub struct GpuNBodySimulation {
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
}

impl GpuNBodySimulation {
    /// Create a new N-body simulation
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context
    /// * `particles` - Initial particle data
    /// * `dt` - Time step per simulation step
    /// * `g` - Gravitational constant
    /// * `softening` - Softening factor to prevent singularities (typically 0.01-0.1)
    pub fn new(
        gpu: &GpuContext,
        particles: &[NBodyParticle],
        dt: f32,
        g: f32,
        softening: f32,
    ) -> Self {
        let num_particles = particles.len() as u32;
        let workgroup_size = 256u32;

        // Create double buffers for ping-pong
        let particle_buffer_a = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("NBody Particle Buffer A"),
            contents: bytemuck::cast_slice(particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let particle_buffer_b = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("NBody Particle Buffer B"),
            contents: bytemuck::cast_slice(particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        // Create params buffer
        let params = NBodyParams {
            dt,
            g,
            softening,
            num_particles,
        };
        let params_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("NBody Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create shader module
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("NBody Shader"),
            source: wgpu::ShaderSource::Wgsl(NBODY_SHADER.into()),
        });

        // Create bind group layout
        let bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("NBody Bind Group Layout"),
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
            label: Some("NBody Bind Group A->B"),
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
            label: Some("NBody Bind Group B->A"),
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
            label: Some("NBody Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Create compute pipeline
        let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NBody Compute Pipeline"),
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
        }
    }

    /// Run one simulation step on the GPU
    ///
    /// Uses ping-pong buffers to read from one buffer and write to another,
    /// avoiding race conditions in the parallel computation.
    pub fn step(&mut self, gpu: &GpuContext) {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("NBody Step Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("NBody Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);

            // Use appropriate bind group based on current buffer
            if self.current_is_a {
                compute_pass.set_bind_group(0, &self.bind_group_a, &[]);
            } else {
                compute_pass.set_bind_group(0, &self.bind_group_b, &[]);
            }

            // Dispatch enough workgroups to cover all particles
            let num_workgroups = (self.num_particles + self.workgroup_size - 1) / self.workgroup_size;
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));

        // Wait for GPU to finish
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
    pub fn read_particles(&self, gpu: &GpuContext) -> Vec<NBodyParticle> {
        let buffer_size = (self.num_particles as usize) * std::mem::size_of::<NBodyParticle>();

        // Determine which buffer has the current state
        let source_buffer = if self.current_is_a {
            &self.particle_buffer_a
        } else {
            &self.particle_buffer_b
        };

        // Create staging buffer for readback
        let staging_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NBody Staging Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from particle buffer to staging buffer
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("NBody Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(source_buffer, 0, &staging_buffer, 0, buffer_size as u64);
        gpu.queue.submit(std::iter::once(encoder.finish()));

        // Map the staging buffer and read data
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
        let particles: Vec<NBodyParticle> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        particles
    }

    /// Update simulation parameters
    pub fn set_params(&self, gpu: &GpuContext, dt: f32, g: f32, softening: f32) {
        let params = NBodyParams {
            dt,
            g,
            softening,
            num_particles: self.num_particles,
        };
        gpu.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    /// Get the number of particles
    pub fn num_particles(&self) -> u32 {
        self.num_particles
    }
}

/// WGSL compute shader for N-body gravitational simulation
///
/// Each thread computes the force on one particle from all other particles.
/// Uses tile-based loading into shared memory for better performance.
const NBODY_SHADER: &str = r#"
struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    mass: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct Params {
    dt: f32,
    g: f32,
    softening: f32,
    num_particles: u32,
}

@group(0) @binding(0) var<storage, read> particles_in: array<Particle>;
@group(0) @binding(1) var<storage, read_write> particles_out: array<Particle>;
@group(0) @binding(2) var<uniform> params: Params;

// Shared memory for tile-based optimization
var<workgroup> tile: array<Particle, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let idx = global_id.x;
    let local_idx = local_id.x;

    // Bounds check
    if (idx >= params.num_particles) {
        return;
    }

    // Load this particle
    let p = particles_in[idx];
    var acc = vec2<f32>(0.0, 0.0);

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

            // Compute displacement
            let r = other.pos - p.pos;

            // Compute distance squared with softening
            let dist_sq = dot(r, r) + params.softening * params.softening;

            // Compute inverse distance cubed (for force direction and magnitude)
            let inv_dist = inverseSqrt(dist_sq);
            let inv_dist_cubed = inv_dist * inv_dist * inv_dist;

            // Accumulate acceleration: a = G * m_other * r / |r|^3
            acc = acc + r * (params.g * other.mass * inv_dist_cubed);
        }

        // Synchronize before loading next tile
        workgroupBarrier();
    }

    // Update velocity and position using symplectic Euler (kick-drift)
    // First kick (half step velocity update)
    let new_vel = p.vel + acc * params.dt;
    // Drift (full step position update)
    let new_pos = p.pos + new_vel * params.dt;

    // Store result
    var result = p;
    result.pos = new_pos;
    result.vel = new_vel;
    particles_out[idx] = result;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_nbody_creation() {
        let gpu = match GpuContext::new() {
            Some(g) => g,
            None => {
                println!("No GPU available, skipping test");
                return;
            }
        };

        let particles: Vec<NBodyParticle> = (0..100)
            .map(|i| {
                let angle = (i as f32) * std::f32::consts::PI * 2.0 / 100.0;
                NBodyParticle {
                    pos: [angle.cos() * 10.0, angle.sin() * 10.0],
                    vel: [0.0, 0.0],
                    mass: 1.0,
                    _padding: [0.0; 3],
                }
            })
            .collect();

        let sim = GpuNBodySimulation::new(&gpu, &particles, 0.01, 1.0, 0.1);
        assert_eq!(sim.num_particles(), 100);
    }

    #[test]
    fn test_gpu_nbody_two_body() {
        let gpu = match GpuContext::new() {
            Some(g) => g,
            None => {
                println!("No GPU available, skipping test");
                return;
            }
        };

        // Two bodies attracting each other
        let particles = vec![
            NBodyParticle {
                pos: [-5.0, 0.0],
                vel: [0.0, 0.0],
                mass: 100.0,
                _padding: [0.0; 3],
            },
            NBodyParticle {
                pos: [5.0, 0.0],
                vel: [0.0, 0.0],
                mass: 100.0,
                _padding: [0.0; 3],
            },
        ];

        let mut sim = GpuNBodySimulation::new(&gpu, &particles, 0.01, 1.0, 0.1);

        // Run simulation
        sim.step_n(&gpu, 100);

        // Read back results
        let result = sim.read_particles(&gpu);

        // Particles should have moved toward each other
        assert!(result[0].pos[0] > -5.0, "Left particle should move right. Got x={}", result[0].pos[0]);
        assert!(result[1].pos[0] < 5.0, "Right particle should move left. Got x={}", result[1].pos[0]);

        // Velocities should be toward each other
        assert!(result[0].vel[0] > 0.0, "Left particle should have positive velocity");
        assert!(result[1].vel[0] < 0.0, "Right particle should have negative velocity");
    }

    #[test]
    fn test_gpu_nbody_circular_orbit() {
        let gpu = match GpuContext::new() {
            Some(g) => g,
            None => {
                println!("No GPU available, skipping test");
                return;
            }
        };

        // Central massive body and orbiting body
        // For circular orbit: v = sqrt(G*M/r)
        let g = 1.0f32;
        let m_central = 1000.0f32;
        let r = 10.0f32;
        let v_orbit = (g * m_central / r).sqrt();

        let particles = vec![
            // Central body (stationary)
            NBodyParticle {
                pos: [0.0, 0.0],
                vel: [0.0, 0.0],
                mass: m_central,
                _padding: [0.0; 3],
            },
            // Orbiting body
            NBodyParticle {
                pos: [r, 0.0],
                vel: [0.0, v_orbit],
                mass: 1.0,
                _padding: [0.0; 3],
            },
        ];

        let mut sim = GpuNBodySimulation::new(&gpu, &particles, 0.001, g, 0.01);

        // Run for a while
        sim.step_n(&gpu, 1000);

        let result = sim.read_particles(&gpu);

        // Orbiting body should still be roughly at distance r from center
        let dx = result[1].pos[0] - result[0].pos[0];
        let dy = result[1].pos[1] - result[0].pos[1];
        let dist = (dx * dx + dy * dy).sqrt();

        // Allow 20% deviation (not perfect due to numerical integration)
        assert!(
            dist > r * 0.8 && dist < r * 1.2,
            "Orbiting body should maintain roughly circular orbit. Got distance {} (expected ~{})",
            dist, r
        );
    }
}
