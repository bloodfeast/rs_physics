//! GPU-accelerated particle simulation
//!
//! This module provides a GPU compute shader implementation for particle
//! physics integration (position and velocity updates with gravity).

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::GpuContext;

/// Particle data stored on GPU (aligned for GPU access)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuParticle {
    /// Position (x, y)
    pub pos: [f32; 2],
    /// Velocity (vx, vy)
    pub vel: [f32; 2],
    /// Mass
    pub mass: f32,
    /// Padding for alignment
    pub _padding: [f32; 3],
}

/// Simulation parameters passed to the shader
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SimParams {
    /// Time step
    dt: f32,
    /// Gravity (typically negative for downward)
    gravity: f32,
    /// Number of particles
    num_particles: u32,
    /// Padding
    _padding: u32,
}

/// GPU-accelerated particle simulation
///
/// Uses compute shaders to update particle positions and velocities
/// with gravity integration. Much faster than CPU for large particle counts.
///
/// # Example
///
/// ```ignore
/// use rs_physics::gpu::{GpuContext, GpuParticleSimulation, GpuParticle};
///
/// let gpu = GpuContext::new().unwrap();
///
/// // Create initial particles
/// let particles: Vec<GpuParticle> = (0..10000)
///     .map(|i| GpuParticle {
///         pos: [(i % 100) as f32, (i / 100) as f32],
///         vel: [1.0, 0.0],
///         mass: 1.0,
///         _padding: [0.0; 3],
///     })
///     .collect();
///
/// let mut sim = GpuParticleSimulation::new(&gpu, &particles, 0.016, -9.81);
///
/// // Run 100 steps
/// for _ in 0..100 {
///     sim.step(&gpu);
/// }
///
/// let results = sim.read_particles(&gpu);
/// ```
pub struct GpuParticleSimulation {
    particle_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    num_particles: u32,
    workgroup_size: u32,
}

impl GpuParticleSimulation {
    /// Create a new GPU particle simulation
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context
    /// * `particles` - Initial particle data
    /// * `dt` - Time step per simulation step
    /// * `gravity` - Gravity acceleration (negative for downward)
    pub fn new(gpu: &GpuContext, particles: &[GpuParticle], dt: f32, gravity: f32) -> Self {
        let num_particles = particles.len() as u32;
        let workgroup_size = 256u32;

        // Create particle buffer
        let particle_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Buffer"),
            contents: bytemuck::cast_slice(particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        // Create params buffer
        let params = SimParams {
            dt,
            gravity,
            num_particles,
            _padding: 0,
        };
        let params_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create shader module
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Integration Shader"),
            source: wgpu::ShaderSource::Wgsl(PARTICLE_SHADER.into()),
        });

        // Create bind group layout
        let bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Particle Bind Group Layout"),
            entries: &[
                // Particles storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
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
                    binding: 1,
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

        // Create bind group
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Particle Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Particle Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Particle Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            particle_buffer,
            params_buffer,
            bind_group,
            pipeline,
            num_particles,
            workgroup_size,
        }
    }

    /// Run one simulation step on the GPU
    pub fn step(&self, gpu: &GpuContext) {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Particle Step Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Particle Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);

            // Dispatch enough workgroups to cover all particles
            let num_workgroups = (self.num_particles + self.workgroup_size - 1) / self.workgroup_size;
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));

        // Wait for GPU to finish
        gpu.device.poll(wgpu::Maintain::Wait);
    }

    /// Run multiple simulation steps
    pub fn step_n(&self, gpu: &GpuContext, steps: u32) {
        for _ in 0..steps {
            self.step(gpu);
        }
    }

    /// Read particle data back from GPU
    ///
    /// This is a synchronous operation that blocks until the data is available.
    pub fn read_particles(&self, gpu: &GpuContext) -> Vec<GpuParticle> {
        let buffer_size = (self.num_particles as usize) * std::mem::size_of::<GpuParticle>();

        // Create staging buffer for readback
        let staging_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from particle buffer to staging buffer
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.particle_buffer, 0, &staging_buffer, 0, buffer_size as u64);
        gpu.queue.submit(std::iter::once(encoder.finish()));

        // Map the staging buffer and read data
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        gpu.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let particles: Vec<GpuParticle> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        particles
    }

    /// Update simulation parameters
    pub fn set_params(&self, gpu: &GpuContext, dt: f32, gravity: f32) {
        let params = SimParams {
            dt,
            gravity,
            num_particles: self.num_particles,
            _padding: 0,
        };
        gpu.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    /// Get the number of particles in the simulation
    pub fn num_particles(&self) -> u32 {
        self.num_particles
    }
}

/// WGSL compute shader for particle integration
const PARTICLE_SHADER: &str = r#"
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
    gravity: f32,
    num_particles: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // Bounds check
    if (idx >= params.num_particles) {
        return;
    }

    // Load particle
    var p = particles[idx];

    // Apply gravity to velocity
    p.vel.y = p.vel.y + params.gravity * params.dt;

    // Update position
    p.pos = p.pos + p.vel * params.dt;

    // Store result
    particles[idx] = p;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_particle_creation() {
        let gpu = match GpuContext::new() {
            Some(g) => g,
            None => {
                println!("No GPU available, skipping test");
                return;
            }
        };

        let particles: Vec<GpuParticle> = (0..1000)
            .map(|i| GpuParticle {
                pos: [i as f32, 0.0],
                vel: [0.0, 0.0],
                mass: 1.0,
                _padding: [0.0; 3],
            })
            .collect();

        let sim = GpuParticleSimulation::new(&gpu, &particles, 0.016, -9.81);
        assert_eq!(sim.num_particles(), 1000);
    }

    #[test]
    fn test_gpu_particle_step() {
        let gpu = match GpuContext::new() {
            Some(g) => g,
            None => {
                println!("No GPU available, skipping test");
                return;
            }
        };

        println!("GPU adapter: {:?}", gpu.adapter_info().name);

        // Create particles at y=100 with zero velocity
        let particles: Vec<GpuParticle> = (0..100)
            .map(|i| GpuParticle {
                pos: [i as f32, 100.0],
                vel: [0.0, 0.0],
                mass: 1.0,
                _padding: [0.0; 3],
            })
            .collect();

        let sim = GpuParticleSimulation::new(&gpu, &particles, 0.016, -9.81);

        // Read initial state
        let initial = sim.read_particles(&gpu);
        println!("Initial particle 0: pos={:?} vel={:?}", initial[0].pos, initial[0].vel);

        // Run 10 steps
        sim.step_n(&gpu, 10);

        // Read back
        let result = sim.read_particles(&gpu);
        println!("After 10 steps particle 0: pos={:?} vel={:?}", result[0].pos, result[0].vel);

        // Check that particles have fallen (y decreased due to gravity)
        for (i, p) in result.iter().enumerate() {
            assert!(p.pos[1] < 100.0, "Particle {} should have fallen due to gravity. pos={:?} vel={:?}", i, p.pos, p.vel);
            assert!(p.vel[1] < 0.0, "Particle {} should have downward velocity", i);
        }
    }
}
