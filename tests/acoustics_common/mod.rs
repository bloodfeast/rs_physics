//! Shared by the acoustics oracles: a device, staging, readback, and scenes.
//!
//! Tests may block and poll; the library may not (oracle L8 scans it).
#![allow(dead_code)]

use rs_physics::gpu::acoustics::*;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub timing: bool,
}

/// A device, or `None` with the reason printed. Timestamps are asked for when offered.
pub fn gpu() -> Option<Gpu> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })) {
        Ok(a) => a,
        Err(e) => {
            println!("skipped: no GPU adapter ({e})");
            return None;
        }
    };
    let timing = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("acoustics oracles"),
        required_features: if timing { wgpu::Features::TIMESTAMP_QUERY } else { wgpu::Features::empty() },
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .ok()?;
    Some(Gpu { device, queue, timing })
}

impl Gpu {
    /// A mapped-at-creation staging buffer holding `bytes`, unmapped and ready to copy from.
    pub fn stage(&self, bytes: &[u8]) -> wgpu::Buffer {
        let size = (bytes.len().max(4) as u64).div_ceil(4) * 4;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test stage"),
            size,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        {
            let mut view = buffer.slice(..).get_mapped_range_mut().unwrap();
            view.slice(..bytes.len()).copy_from_slice(bytes);
        }
        buffer.unmap();
        buffer
    }

    pub fn encoder(&self) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(&Default::default())
    }

    pub fn submit_wait(&self, enc: wgpu::CommandEncoder) {
        self.queue.submit([enc.finish()]);
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }

    pub fn read(&self, src: &wgpu::Buffer, len: u64) -> Vec<u8> {
        let dst = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test read"),
            size: len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.encoder();
        enc.copy_buffer_to_buffer(src, 0, &dst, 0, len);
        self.submit_wait(enc);
        dst.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let out = dst.slice(..).get_mapped_range().unwrap().to_vec();
        dst.unmap();
        out
    }
}

/// A scene held on the CPU, so the oracles can march the same thing.
#[derive(Clone)]
pub struct Scene {
    pub heights: Vec<f32>,
    pub material: Vec<u8>,
    pub foliage: Option<Vec<u8>>,
    pub cols: u32,
    pub rows: u32,
    pub cell: f32,
    pub origin: [f32; 2],
    pub statics: Vec<Obb>,
    pub materials: Vec<AcousticMaterial>,
}

impl Scene {
    pub fn flat(cols: u32, rows: u32, cell: f32, height: f32) -> Scene {
        let n = (cols * rows) as usize;
        Scene {
            heights: vec![height; n],
            material: vec![0; n],
            foliage: None,
            cols,
            rows,
            cell,
            origin: [0.0, 0.0],
            statics: Vec::new(),
            materials: vec![AcousticMaterial { reflection: [0.9, 0.8, 0.7, 0.6] }],
        }
    }

    pub fn terrain(&self) -> TerrainGrid<'_> {
        TerrainGrid {
            heights: &self.heights,
            material: &self.material,
            cols: self.cols,
            rows: self.rows,
            cell_m: self.cell,
            origin: self.origin,
        }
    }

    pub fn height_at(&self, x: f64, z: f64) -> Option<f64> {
        let c = ((x - self.origin[0] as f64) / self.cell as f64).floor();
        let r = ((z - self.origin[1] as f64) / self.cell as f64).floor();
        if c < 0.0 || r < 0.0 || c >= self.cols as f64 || r >= self.rows as f64 {
            return None;
        }
        Some(self.heights[r as usize * self.cols as usize + c as usize] as f64)
    }

    /// Record the scene into `acoustics`.
    pub fn load(&self, gpu: &Gpu, acoustics: &mut GpuAcoustics) -> SceneLayout {
        let density = self.foliage.as_ref().map(|d| DensityGrid { density: d });
        let layout = GpuAcoustics::scene_bytes(
            &self.terrain(),
            &self.statics,
            self.materials.len() as u32,
            density.as_ref(),
        )
        .unwrap();
        let mut bytes = vec![0u8; layout.staged_bytes() as usize];
        GpuAcoustics::pack_scene(
            &layout,
            &self.terrain(),
            &self.statics,
            &self.materials,
            density.as_ref(),
            &mut bytes[..],
        )
        .unwrap();
        let stage = gpu.stage(&bytes);
        let mut enc = gpu.encoder();
        acoustics
            .encode_scene(&mut enc, &layout, Staged { buffer: &stage, offset: 0, len: layout.staged_bytes() })
            .unwrap();
        gpu.submit_wait(enc);
        layout
    }
}

pub fn listener_at(p: [f32; 3]) -> Listener {
    Listener { position: p, forward: [0.0, 0.0, -1.0], right: [1.0, 0.0, 0.0], velocity: [0.0; 3] }
}

/// Pack, record, submit, wait, and take the results of one dispatch.
pub fn dispatch(
    gpu: &Gpu,
    acoustics: &mut GpuAcoustics,
    header: &DispatchHeader,
    sources: &[Source],
    movers: &[Obb],
) -> TickResults {
    let mut bytes = vec![0u8; dispatch_bytes(sources.len(), movers.len())];
    let shape = GpuAcoustics::pack_dispatch(header, sources, movers, &mut bytes[..]).unwrap();
    let stage = gpu.stage(&bytes);
    let mut enc = gpu.encoder();
    let e = acoustics.encode(&mut enc, Staged { buffer: &stage, offset: 0, len: shape.bytes }, shape).unwrap();
    assert!(matches!(e, Encoded::Dispatched { .. }), "{e:?}");
    gpu.submit_wait(enc);
    let mut out = TickResults::default();
    assert!(acoustics.take_ready(&mut out), "no result was ready after a waited submit");
    out
}

/// Field rays of the last dispatch.
pub fn field_rays(gpu: &Gpu, acoustics: &GpuAcoustics) -> Vec<FieldRay> {
    let dst = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4096,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut enc = gpu.encoder();
    acoustics.copy_field_rays(&mut enc, &dst, 0);
    gpu.submit_wait(enc);
    bytemuck::cast_slice(&gpu.read(&dst, 4096)).to_vec()
}

/// A small deterministic generator (SplitMix64), so every oracle is reproducible.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}
