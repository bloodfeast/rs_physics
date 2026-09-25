//! Hooks for the oracles: the per-source law chain run over caller-chosen inputs.
//!
//! The source workgroup's lane 0 calls one WGSL function, `laws`, on the maxima its march
//! found. [`LawProbe`] runs that same function over cases the caller writes, so the f32
//! laws can be held against the f64 CPU laws (oracle L1) without a scene that happens to
//! produce each excess. It is not a second implementation: it is the shader's own entry
//! point over the same function.

use bytemuck::{Pod, Zeroable};

use super::records::{AcousticsError, DispatchHeader, HEADER_BYTES};
use super::{shader, storage_entry, Staged};

/// One case for [`LawProbe`]: a source, and the march's results for it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct LawCase {
    /// Source position, in metres.
    pub source: [f32; 3],
    /// Source directivity toward the listener.
    pub directivity: f32,
    /// Source velocity, in metres per second.
    pub velocity: [f32; 3],
    /// Tag, echoed.
    pub tag: u32,
    /// Main-edge signed excess per band, in metres.
    pub excess: [f32; 4],
    /// Main-edge signed excess at the probe band, in metres.
    pub probe_excess: f32,
    /// Path length through foliage, in metres.
    pub foliage_m: f32,
    /// Padding to 64 bytes.
    pub pad: [f32; 2],
}

const _: () = assert!(std::mem::size_of::<LawCase>() == 64);

/// The law chain on its own, for the oracles.
pub struct LawProbe {
    input: wgpu::Buffer,
    output: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
}

impl LawProbe {
    /// Cases one dispatch may carry.
    pub const MAX_CASES: u32 = 1_024;

    /// Build the probe.
    ///
    /// # Arguments
    ///
    /// * `device` - any device with compute.
    ///
    /// # Returns
    ///
    /// The probe.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::acoustics::oracle::LawProbe;
    ///
    /// # fn device() -> wgpu::Device { unimplemented!() }
    /// let probe = LawProbe::new(&device());
    /// ```
    pub fn new(device: &wgpu::Device) -> LawProbe {
        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let input = buffer(
            "law probe input",
            HEADER_BYTES + 64 * Self::MAX_CASES as u64,
            storage | wgpu::BufferUsages::COPY_DST,
        );
        let output = buffer(
            "law probe output",
            32 * Self::MAX_CASES as u64,
            storage | wgpu::BufferUsages::COPY_SRC,
        );
        let scene = buffer("law probe scene", 256, storage);
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("law probe"),
            entries: &[storage_entry(0, true), storage_entry(1, false), storage_entry(2, true)],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("acoustics query"),
            source: wgpu::ShaderSource::Wgsl(shader::query_source().into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = super::pipeline(device, &layout, &module, "probe_laws", &[]);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("law probe"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: scene.as_entire_binding() },
            ],
        });
        LawProbe { input, output, pipeline, group }
    }

    /// Pack a header and cases: the header's source count is set to the case count.
    ///
    /// # Arguments
    ///
    /// * `header` - the listener, the air and the legibility curve for every case.
    /// * `cases` - at most [`LawProbe::MAX_CASES`].
    /// * `out` - at least `160 + 64 x cases` bytes.
    ///
    /// # Returns
    ///
    /// The bytes written.
    ///
    /// # Errors
    ///
    /// [`AcousticsError::Limit`] over the case limit; [`AcousticsError::OutputTooSmall`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::acoustics::DispatchHeader;
    /// use rs_physics::gpu::acoustics::oracle::{LawCase, LawProbe};
    ///
    /// let mut out = [0u8; 224];
    /// let n = LawProbe::pack(&DispatchHeader::default(), &[LawCase::default()], &mut out[..]).unwrap();
    /// assert_eq!(n, 224);
    /// ```
    pub fn pack<'o>(
        header: &DispatchHeader,
        cases: &[LawCase],
        out: impl Into<super::Out<'o>>,
    ) -> Result<usize, AcousticsError> {
        let mut out = out.into();
        if cases.len() > Self::MAX_CASES as usize {
            return Err(AcousticsError::Limit {
                what: "law probe cases",
                got: cases.len() as u64,
                max: Self::MAX_CASES as u64,
            });
        }
        let need = HEADER_BYTES as usize + 64 * cases.len();
        if out.len() < need {
            return Err(AcousticsError::OutputTooSmall { need, got: out.len() });
        }
        let mut h = *header;
        h.counts[0] = cases.len() as u32;
        out.slice(..HEADER_BYTES as usize).copy_from_slice(bytemuck::bytes_of(&h));
        out.slice(HEADER_BYTES as usize..need).copy_from_slice(bytemuck::cast_slice(cases));
        Ok(need)
    }

    /// Record the probe over staged cases, and a copy of the results into `dst`.
    ///
    /// # Arguments
    ///
    /// * `enc` - the caller's encoder.
    /// * `staged` - the bytes [`LawProbe::pack`] wrote, staged.
    /// * `cases` - how many cases they hold.
    /// * `dst` - a `COPY_DST` buffer of at least `32 x cases` bytes, which receives one
    ///   [`super::SourceResult`] per case.
    ///
    /// # Returns
    ///
    /// Nothing; the work is recorded.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::acoustics::Staged;
    /// use rs_physics::gpu::acoustics::oracle::LawProbe;
    ///
    /// # fn setup() -> (LawProbe, wgpu::Device, wgpu::Buffer, wgpu::Buffer) { unimplemented!() }
    /// let (probe, device, stage, dst) = setup();
    /// let mut enc = device.create_command_encoder(&Default::default());
    /// probe.encode(&mut enc, Staged { buffer: &stage, offset: 0, len: 224 }, 1, &dst);
    /// ```
    pub fn encode(&self, enc: &mut wgpu::CommandEncoder, staged: Staged<'_>, cases: u32, dst: &wgpu::Buffer) {
        enc.copy_buffer_to_buffer(staged.buffer, staged.offset, &self.input, 0, staged.len);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("law probe"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.group, &[]);
            pass.set_pipeline(&self.pipeline);
            pass.dispatch_workgroups(cases.div_ceil(64).max(1), 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.output, 0, dst, 0, 32 * cases.max(1) as u64);
    }
}
