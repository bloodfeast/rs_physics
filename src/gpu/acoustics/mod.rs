//! # GPU acoustics
//!
//! The game's acoustics, on the GPU in f32: what a listener hears of each sound (gain per
//! ear, interaural delay, a low-pass, Doppler pitch) through the terrain, the static
//! obstacles, the large movers and the foliage between them, and what the listener's
//! surroundings do (early reflections, the mean free path and a reverberation time).
//! This is the one implementation; the f64 laws in [`crate::acoustics`] are its oracle.
//!
//! ## Shape
//!
//! * **The scene** is rs_physics's own: a terrain of 2 m columns with a max pyramid, up to
//!   4,096 static boxes binned into the same grid, a four-band material table and an
//!   optional foliage density. It is rays against this, on any adapter with compute; the
//!   engine's acceleration structures are not used.
//! * **One dispatch per source set**, recorded into the caller's encoder: a 64-lane
//!   workgroup per source and one for the listener field, in one compute pass.
//! * **No device calls after [`GpuAcoustics::new`].** Inputs are packed by pure functions
//!   into memory the caller owns (the engine's staging ring) and copied in while
//!   recording; results come back through four readback slots mapped by
//!   `map_buffer_on_submit`, whose callback fires inside the caller's own next
//!   `Queue::submit`. Nothing here polls the device or waits on it.
//!
//! ## What the dispatch computes
//!
//! **Per source.** The grid-clipped path from the source to the listener is split into 64
//! equal lengths, one per lane, and each lane walks its cells. In each cell it takes the
//! largest **signed excess path** `|SQ| + |QL| - |SL|` over the column top `Q` (positive
//! when the column stands above the line, negative for the clearance when it does not),
//! the same over the top edge of every static the path crosses there, and the foliage
//! density times the path length in the cell. Each lane also tests one mover. The
//! workgroup reduces to the largest excess per band (Deygout's main edge, found in the
//! march itself) and the foliage path. Then, in lane 0, the laws per band: air
//! (ISO 9613-1), the signed barrier law (Kurze-Anderson,
//! [`crate::acoustics::surfaces::barrier_insertion_db`]) and foliage (ISO 9613-2), fitted
//! to a broadband gain and a one-pole cutoff ([`crate::acoustics::band::fit_lowpass`]);
//! spreading; the head shadow and Woodworth delay ([`crate::acoustics::Ears::hear`]); and
//! the Doppler ratio.
//!
//! **The Fresnel rule, per path and per band.** An obstacle counts in a band only if its
//! half-width is at least the first Fresnel radius there
//! ([`crate::acoustics::surfaces::occludes`]); the terrain always counts. So a soldier does
//! not muffle a shot and a tank does, and a fence takes the treble and leaves the bass.
//!
//! **The listener field.** 64 rays weighted toward the horizon ([`field_directions`]),
//! marched through the pyramid and the static grid for up to two specular bounces, reduce
//! to eight early-reflection taps, the mean free path `4V/S` by quadrature over the rays'
//! first hits (`V = sum w l^3 / 3`, `S = sum w l^2 / |cos|`), the share of the sphere that
//! escapes, and an Eyring RT60 at 500 Hz in which escaped rays absorb everything.
//!
//! ## Examples
//!
//! The whole cycle, as a host drives it once a tick. `stage` stands for the caller's
//! staging ring; here it is a mapped buffer.
//!
//! ```no_run
//! use rs_physics::acoustics::Air;
//! use rs_physics::gpu::acoustics::*;
//!
//! # fn device() -> (wgpu::Device, wgpu::Queue) { unimplemented!() }
//! let (device, queue) = device();
//! let mut acoustics = GpuAcoustics::new(&device, AcousticLimits::MAX).unwrap();
//!
//! // Load: a 140 x 100 terrain at 2 m, no statics, one material.
//! let heights = vec![0.0f32; 140 * 100];
//! let material = vec![0u8; 140 * 100];
//! let terrain = TerrainGrid {
//!     heights: &heights, material: &material, cols: 140, rows: 100, cell_m: 2.0,
//!     origin: [0.0, 0.0],
//! };
//! let materials = [AcousticMaterial { reflection: [0.9; 4] }];
//! let layout = GpuAcoustics::scene_bytes(&terrain, &[], materials.len() as u32, None).unwrap();
//! let stage = device.create_buffer(&wgpu::BufferDescriptor {
//!     label: None, size: 1 << 20,
//!     usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
//!     mapped_at_creation: true,
//! });
//! {
//!     let mut view = stage.slice(..).get_mapped_range_mut().unwrap();
//!     GpuAcoustics::pack_scene(&layout, &terrain, &[], &materials, None, &mut view).unwrap();
//! }
//! stage.unmap();
//! let mut enc = device.create_command_encoder(&Default::default());
//! let staged = Staged { buffer: &stage, offset: 0, len: layout.staged_bytes() };
//! acoustics.encode_scene(&mut enc, &layout, staged).unwrap();
//! queue.submit([enc.finish()]);
//!
//! // A tick: pack the listener and the sources, record, submit; take what is ready.
//! let listener = Listener {
//!     position: [140.0, 1.6, 100.0], forward: [0.0, 0.0, -1.0], right: [1.0, 0.0, 0.0],
//!     velocity: [0.0; 3],
//! };
//! let header = DispatchHeader::new(&listener, &Air::standard(), &[]).unwrap();
//! let shot = Source::new([180.0, 1.0, 90.0], 1.0, [0.0; 3], 1, NO_MOVER).unwrap();
//! let mut bytes = [0u8; 512];
//! let shape = GpuAcoustics::pack_dispatch(&header, &[shot], &[], &mut bytes).unwrap();
//! // ... the host writes `bytes[..shape.bytes]` into its ring and gets a `Staged` back ...
//! # let tick = Staged { buffer: &stage, offset: 0, len: shape.bytes };
//! let mut enc = device.create_command_encoder(&Default::default());
//! acoustics.encode(&mut enc, tick, shape).unwrap();
//! queue.submit([enc.finish()]);
//!
//! let mut out = TickResults::default();
//! if acoustics.take_ready(&mut out) {
//!     for r in out.results() {
//!         println!("tag {}: {:.3} {:.3} cutoff {:.0} Hz", r.tag(), r.gain_l, r.gain_r, r.cutoff_hz);
//!     }
//! }
//! ```

#![warn(missing_docs)]

mod pack;
mod readback;
mod records;
mod shader;

pub mod oracle;

pub use pack::{
    dispatch_bytes, terrain_rect_bytes, DensityGrid, DispatchShape, GridRect, SceneLayout,
    TerrainGrid, HEADER_WORDS, PYRAMID_LEVELS,
};
pub use records::{
    AcousticCounters, AcousticLimits, AcousticMaterial, AcousticsError, DispatchHeader,
    FieldRay, Listener, ListenerField, Obb, Source, SourceResult, Tap, TickResults, BANDS_HZ,
    HEADER_BYTES, MAX_BOUNCES, MAX_CELLS_PER_STATIC, MAX_FIELD_RAYS, MAX_LEGIBILITY_POINTS,
    MAX_MOVERS, MAX_SOURCES, MAX_STATICS, MAX_TAG, MAX_TERRAIN_SIDE, NO_MOVER, PROBE_HZ,
    READBACK_BYTES,
};
pub use shader::{
    field_directions, series_coefficients, shadow_k, ASIN_COEFFICIENTS, ASIN_ERROR,
    COTH_TERMS, COT_TERMS, TAP_BINS, TAP_BIN_S,
};

use pack::*;
use readback::Ring;

/// A slice of a caller's buffer holding packed bytes, to be copied in while recording.
///
/// The buffer needs `COPY_SRC`; `offset` and `len` must be multiples of 4.
#[derive(Clone, Copy, Debug)]
pub struct Staged<'a> {
    /// The buffer: the engine's staging ring slot, or any `COPY_SRC` buffer.
    pub buffer: &'a wgpu::Buffer,
    /// Where the bytes start, bytes.
    pub offset: u64,
    /// How many, bytes.
    pub len: u64,
}

/// What [`GpuAcoustics::encode`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoded {
    /// Recorded, into readback slot `slot`, as dispatch number `sequence`.
    Dispatched {
        /// The readback slot, 0 to 3.
        slot: usize,
        /// The dispatch's sequence number, echoed in [`TickResults::sequence`].
        sequence: u64,
    },
    /// Nothing recorded: the next readback slot was still mapped or waiting for its map.
    /// Counted in [`AcousticCounters::skipped_busy`]; the next call tries the slot after.
    SkippedBusy,
}

/// The acoustics pass: its pipelines, its buffers and its readback ring.
///
/// See the module docs for what it computes and how a host drives it.
pub struct GpuAcoustics {
    device: wgpu::Device,
    limits: AcousticLimits,
    input: wgpu::Buffer,
    output: wgpu::Buffer,
    scene: wgpu::Buffer,
    layout: Option<SceneLayout>,
    query_bgl: wgpu::BindGroupLayout,
    build_bgl: wgpu::BindGroupLayout,
    query_group: wgpu::BindGroup,
    build_group: wgpu::BindGroup,
    sources: wgpu::ComputePipeline,
    field: wgpu::ComputePipeline,
    statics_bin: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    statics_fill: wgpu::ComputePipeline,
    pyramid: [wgpu::ComputePipeline; PYRAMID_LEVELS as usize],
    ring: Ring,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    entry: &str,
    constants: &[(&str, f64)],
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(layout),
        module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants,
            ..Default::default()
        },
        cache: None,
    })
}

fn check_staged(staged: &Staged<'_>, len: u64) -> Result<(), AcousticsError> {
    if staged.offset % 4 != 0 || staged.len % 4 != 0 {
        return Err(AcousticsError::Misaligned);
    }
    if staged.len != len {
        return Err(AcousticsError::Shape("the staged length is not the packed length"));
    }
    if staged.offset + staged.len > staged.buffer.size() {
        return Err(AcousticsError::Shape("the staged slice runs past its buffer"));
    }
    Ok(())
}

impl GpuAcoustics {
    /// Build the pass on a device: pipelines, the per-dispatch buffers, the readback ring
    /// and an empty scene.
    ///
    /// The only call that takes the device. The pass needs nothing beyond wgpu's default
    /// limits: three storage buffers per stage and 256 invocations per workgroup.
    ///
    /// # Arguments
    ///
    /// * `device` - the engine's device. A clone is kept, to size the scene buffer when a
    ///   scene arrives; it is never polled.
    /// * `limits` - the sizes to build for; see [`AcousticLimits`].
    ///
    /// # Returns
    ///
    /// The pass, with an empty scene: sources see no terrain and field rays escape until
    /// [`GpuAcoustics::encode_scene`] records one.
    ///
    /// # Errors
    ///
    /// [`AcousticsError::Limit`] when a limit exceeds its hard limit.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::acoustics::{AcousticLimits, GpuAcoustics};
    ///
    /// # fn device() -> wgpu::Device { unimplemented!() }
    /// let device = device();
    /// let limits = AcousticLimits { sources: 40, ..AcousticLimits::MAX };
    /// let acoustics = GpuAcoustics::new(&device, limits).unwrap();
    /// assert_eq!(acoustics.counters().dispatches, 0);
    /// ```
    pub fn new(device: &wgpu::Device, limits: AcousticLimits) -> Result<GpuAcoustics, AcousticsError> {
        limits.validate()?;
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
            "acoustics input",
            dispatch_bytes(limits.sources as usize, limits.movers as usize) as u64,
            storage | wgpu::BufferUsages::COPY_DST,
        );
        let output = buffer(
            "acoustics output",
            shader::OUT_WORDS as u64 * 4,
            storage | wgpu::BufferUsages::COPY_SRC,
        );
        let scene = buffer(
            "acoustics scene",
            (HEADER_WORDS as u64 + 1) * 4,
            storage | wgpu::BufferUsages::COPY_DST,
        );

        let query_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("acoustics query"),
            entries: &[storage_entry(0, true), storage_entry(1, false), storage_entry(2, true)],
        });
        let build_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("acoustics build"),
            entries: &[storage_entry(0, false)],
        });
        let query_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("acoustics query"),
            source: wgpu::ShaderSource::Wgsl(shader::query_source().into()),
        });
        let build_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("acoustics build"),
            source: wgpu::ShaderSource::Wgsl(shader::build_source().into()),
        });
        let layout_of = |bgl: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            })
        };
        let query_layout = layout_of(&query_bgl);
        let build_layout = layout_of(&build_bgl);
        let q = |entry: &str| pipeline(device, &query_layout, &query_module, entry, &[]);
        let b = |entry: &str, constants: &[(&str, f64)]| {
            pipeline(device, &build_layout, &build_module, entry, constants)
        };
        let pyramid = [
            b("pyramid", &[("LEVEL", 1.0)]),
            b("pyramid", &[("LEVEL", 2.0)]),
            b("pyramid", &[("LEVEL", 3.0)]),
        ];
        let (query_group, build_group) =
            Self::groups(device, &query_bgl, &build_bgl, &input, &output, &scene);
        Ok(GpuAcoustics {
            limits,
            sources: q("sources"),
            field: q("field"),
            statics_bin: b("statics_bin", &[]),
            scan: b("scan", &[]),
            statics_fill: b("statics_fill", &[]),
            pyramid,
            ring: Ring::new(device),
            device: device.clone(),
            input,
            output,
            scene,
            layout: None,
            query_bgl,
            build_bgl,
            query_group,
            build_group,
        })
    }

    fn groups(
        device: &wgpu::Device,
        query_bgl: &wgpu::BindGroupLayout,
        build_bgl: &wgpu::BindGroupLayout,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        scene: &wgpu::Buffer,
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let query = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("acoustics query"),
            layout: query_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: scene.as_entire_binding() },
            ],
        });
        let build = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("acoustics build"),
            layout: build_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: scene.as_entire_binding() }],
        });
        (query, build)
    }

    /// Validate a scene and lay out its buffer.
    ///
    /// # Arguments
    ///
    /// * `terrain` - the terrain grid; also the index the statics are binned into.
    /// * `statics` - static obstacles, each wholly or partly over the grid.
    /// * `materials` - the length of the material table the scene will carry.
    /// * `foliage` - foliage density on the terrain's grid, if any.
    ///
    /// # Returns
    ///
    /// The layout: the staged and resident sizes, and every region's offset.
    ///
    /// # Errors
    ///
    /// * [`AcousticsError::Limit`] - a side over 1,024 cells or more than 4,096 statics.
    /// * [`AcousticsError::Shape`] - slices that are not `cols x rows`, a non-positive cell.
    /// * [`AcousticsError::Material`] - a static naming a material past the table.
    /// * [`AcousticsError::Static`] - a static wholly off the grid, or over 4,096 cells.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::acoustics::{GpuAcoustics, Obb, TerrainGrid};
    ///
    /// let heights = vec![0.0f32; 140 * 100];
    /// let material = vec![0u8; 140 * 100];
    /// let terrain = TerrainGrid {
    ///     heights: &heights, material: &material, cols: 140, rows: 100, cell_m: 2.0,
    ///     origin: [0.0, 0.0],
    /// };
    /// let wall = Obb::upright([50.0, 2.0, 50.0], [8.0, 4.0, 0.5], 0.0, 0);
    /// let layout = GpuAcoustics::scene_bytes(&terrain, &[wall], 1, None).unwrap();
    /// assert_eq!(layout.statics(), 1);
    /// // A static off the map is refused, not dropped.
    /// let lost = Obb::upright([-500.0, 2.0, 50.0], [8.0, 4.0, 0.5], 0.0, 0);
    /// assert!(GpuAcoustics::scene_bytes(&terrain, &[lost], 1, None).is_err());
    /// ```
    pub fn scene_bytes(
        terrain: &TerrainGrid<'_>,
        statics: &[Obb],
        materials: u32,
        foliage: Option<&DensityGrid<'_>>,
    ) -> Result<SceneLayout, AcousticsError> {
        pack::layout(terrain, statics, materials, foliage)
    }

    /// Pack a scene into caller memory, for [`GpuAcoustics::encode_scene`].
    ///
    /// # Arguments
    ///
    /// * `layout` - from [`GpuAcoustics::scene_bytes`] for this same scene.
    /// * `terrain`, `statics`, `foliage` - the scene.
    /// * `materials` - the material table.
    /// * `out` - at least [`SceneLayout::staged_bytes`] bytes.
    ///
    /// # Returns
    ///
    /// `Ok(())` with the first `staged_bytes` of `out` written.
    ///
    /// # Errors
    ///
    /// Everything [`GpuAcoustics::scene_bytes`] refuses; [`AcousticsError::Shape`] when
    /// `layout` was made for a different scene; [`AcousticsError::Material`] for a terrain
    /// cell naming a material past the table; [`AcousticsError::OutputTooSmall`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::acoustics::{AcousticMaterial, GpuAcoustics, TerrainGrid};
    ///
    /// let heights = [0.0f32, 1.0, 2.0, 3.0];
    /// let terrain = TerrainGrid {
    ///     heights: &heights, material: &[0; 4], cols: 2, rows: 2, cell_m: 2.0, origin: [0.0; 2],
    /// };
    /// let table = [AcousticMaterial { reflection: [0.9; 4] }];
    /// let layout = GpuAcoustics::scene_bytes(&terrain, &[], 1, None).unwrap();
    /// let mut out = vec![0u8; layout.staged_bytes() as usize];
    /// GpuAcoustics::pack_scene(&layout, &terrain, &[], &table, None, &mut out).unwrap();
    /// assert_eq!(&out[..4], &2u32.to_le_bytes()); // the header opens with the columns
    /// ```
    pub fn pack_scene(
        layout: &SceneLayout,
        terrain: &TerrainGrid<'_>,
        statics: &[Obb],
        materials: &[AcousticMaterial],
        foliage: Option<&DensityGrid<'_>>,
        out: &mut [u8],
    ) -> Result<(), AcousticsError> {
        pack::pack_scene(layout, terrain, statics, materials, foliage, out)
    }

    /// Pack a rectangle of new terrain heights (a crater), for
    /// [`GpuAcoustics::encode_terrain_rect`].
    ///
    /// # Arguments
    ///
    /// * `rect` - the cells.
    /// * `heights` - the new heights, row-major within the rect, in metres.
    /// * `out` - at least [`terrain_rect_bytes`] bytes.
    ///
    /// # Returns
    ///
    /// The bytes written.
    ///
    /// # Errors
    ///
    /// [`AcousticsError::Shape`] for heights that are not `cols x rows`, empty, or not
    /// finite; [`AcousticsError::OutputTooSmall`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::acoustics::{GpuAcoustics, GridRect};
    ///
    /// let rect = GridRect { col: 10, row: 20, cols: 2, rows: 2 };
    /// let mut out = [0u8; 64];
    /// let n = GpuAcoustics::pack_terrain_rect(rect, &[-1.0; 4], &mut out).unwrap();
    /// assert_eq!(n, 32);
    /// ```
    pub fn pack_terrain_rect(
        rect: GridRect,
        heights: &[f32],
        out: &mut [u8],
    ) -> Result<usize, AcousticsError> {
        pack::pack_terrain_rect(rect, heights, out)
    }

    /// Pack a dispatch: the header, the sources and the movers.
    ///
    /// # Arguments
    ///
    /// * `header` - from [`DispatchHeader::new`]; its counts are filled here.
    /// * `sources` - at most [`MAX_SOURCES`]; zero is valid.
    /// * `movers` - at most [`MAX_MOVERS`], the large movers the Fresnel rule keeps.
    /// * `out` - at least [`dispatch_bytes`] bytes.
    ///
    /// # Returns
    ///
    /// The [`DispatchShape`] to hand to [`GpuAcoustics::encode`].
    ///
    /// # Errors
    ///
    /// [`AcousticsError::Limit`] over the hard limits; [`AcousticsError::OutputTooSmall`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::acoustics::*;
    ///
    /// let header = DispatchHeader::default();
    /// let s = Source::new([1.0, 1.0, 1.0], 1.0, [0.0; 3], 9, NO_MOVER).unwrap();
    /// let mut out = [0u8; 256];
    /// let shape = GpuAcoustics::pack_dispatch(&header, &[s], &[], &mut out).unwrap();
    /// assert_eq!(shape.bytes, 192);
    /// assert_eq!(shape.sources, 1);
    /// ```
    pub fn pack_dispatch(
        header: &DispatchHeader,
        sources: &[Source],
        movers: &[Obb],
        out: &mut [u8],
    ) -> Result<DispatchShape, AcousticsError> {
        pack::pack_dispatch(header, sources, movers, out)
    }

    /// Record a scene: the copy of the staged bytes, then the grid and pyramid build.
    ///
    /// The scene buffer grows here, from the kept device, if the scene is larger than any
    /// before it; that is the only allocation after `new`, and it happens on a load.
    ///
    /// # Arguments
    ///
    /// * `enc` - the caller's encoder.
    /// * `layout` - the scene's layout.
    /// * `staged` - the bytes [`GpuAcoustics::pack_scene`] wrote, staged.
    ///
    /// # Returns
    ///
    /// `Ok(())` once recorded. The scene takes effect in submission order.
    ///
    /// # Errors
    ///
    /// * [`AcousticsError::Limit`] - a grid or static count over this pass's limits.
    /// * [`AcousticsError::Misaligned`], [`AcousticsError::Shape`] - a bad staged slice.
    /// * [`AcousticsError::Device`] - a scene larger than one storage binding.
    ///
    /// # Examples
    ///
    /// See the module docs.
    pub fn encode_scene(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        layout: &SceneLayout,
        staged: Staged<'_>,
    ) -> Result<(), AcousticsError> {
        check_staged(&staged, layout.staged_bytes())?;
        let (cols, rows) = layout.grid();
        let over = |what, got: u32, max: u32| {
            if got > max {
                Err(AcousticsError::Limit { what, got: got as u64, max: max as u64 })
            } else {
                Ok(())
            }
        };
        over("terrain columns", cols, self.limits.terrain_cols)?;
        over("terrain rows", rows, self.limits.terrain_rows)?;
        over("statics", layout.statics(), self.limits.statics)?;
        let need = layout.resident_bytes();
        if need > self.device.limits().max_storage_buffer_binding_size as u64 {
            return Err(AcousticsError::Device("the scene exceeds one storage binding"));
        }
        if need > self.scene.size() {
            self.scene = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("acoustics scene"),
                size: need,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let (q, b) = Self::groups(
                &self.device,
                &self.query_bgl,
                &self.build_bgl,
                &self.input,
                &self.output,
                &self.scene,
            );
            self.query_group = q;
            self.build_group = b;
        }
        enc.copy_buffer_to_buffer(staged.buffer, staged.offset, &self.scene, 0, staged.len);
        let cells = cols as u64 * rows as u64;
        if cells > 0 {
            enc.clear_buffer(&self.scene, layout.words(H_OFF_COUNT) as u64 * 4, Some(cells * 4));
            enc.clear_buffer(&self.scene, layout.words(H_OFF_TOP) as u64 * 4, Some(cells * 4));
        }
        let n = layout.statics();
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("acoustics scene"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.build_group, &[]);
            if n > 0 {
                pass.set_pipeline(&self.statics_bin);
                pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
            }
            pass.set_pipeline(&self.scan);
            pass.dispatch_workgroups(1, 1, 1);
            if n > 0 {
                pass.set_pipeline(&self.statics_fill);
                pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
            }
            if cells > 0 {
                self.record_pyramid(&mut pass, GridRect { col: 0, row: 0, cols, rows });
            }
        }
        self.layout = Some(*layout);
        Ok(())
    }

    fn record_pyramid(&self, pass: &mut wgpu::ComputePass<'_>, rect: GridRect) {
        for level in 1..=PYRAMID_LEVELS {
            let c0 = rect.col >> level;
            let c1 = (rect.col + rect.cols - 1) >> level;
            let r0 = rect.row >> level;
            let r1 = (rect.row + rect.rows - 1) >> level;
            pass.set_pipeline(&self.pyramid[level as usize - 1]);
            pass.dispatch_workgroups((c1 - c0 + 1).div_ceil(8), (r1 - r0 + 1).div_ceil(8), 1);
        }
    }

    /// Record new heights for a rectangle of the terrain, and the pyramid above it.
    ///
    /// # Arguments
    ///
    /// * `enc` - the caller's encoder.
    /// * `rect` - the cells, inside the current scene's grid.
    /// * `staged` - the bytes [`GpuAcoustics::pack_terrain_rect`] wrote, staged.
    ///
    /// # Returns
    ///
    /// `Ok(())` once recorded: one copy per row of the rect and one dispatch per pyramid
    /// level.
    ///
    /// # Errors
    ///
    /// [`AcousticsError::Shape`] with no scene, a rect outside the grid or a staged length
    /// that is not the rect's; [`AcousticsError::Misaligned`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::acoustics::{GpuAcoustics, GridRect, Staged};
    ///
    /// # fn setup() -> (GpuAcoustics, wgpu::Device, wgpu::Buffer) { unimplemented!() }
    /// let (mut acoustics, device, stage) = setup();
    /// let rect = GridRect { col: 40, row: 30, cols: 4, rows: 4 };
    /// // ... pack_terrain_rect into the ring, then:
    /// let staged = Staged { buffer: &stage, offset: 0, len: 16 + 4 * 16 };
    /// let mut enc = device.create_command_encoder(&Default::default());
    /// acoustics.encode_terrain_rect(&mut enc, rect, staged).unwrap();
    /// ```
    pub fn encode_terrain_rect(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        rect: GridRect,
        staged: Staged<'_>,
    ) -> Result<(), AcousticsError> {
        let layout = self.layout.ok_or(AcousticsError::Shape("no scene has been encoded"))?;
        check_staged(&staged, terrain_rect_bytes(rect) as u64)?;
        let (cols, rows) = layout.grid();
        let inside = rect.cols > 0
            && rect.rows > 0
            && rect.col as u64 + rect.cols as u64 <= cols as u64
            && rect.row as u64 + rect.rows as u64 <= rows as u64;
        if !inside {
            return Err(AcousticsError::Shape("the rect is not inside the terrain grid"));
        }
        enc.copy_buffer_to_buffer(staged.buffer, staged.offset, &self.scene, H_RECT as u64 * 4, 16);
        let heights = layout.words(H_OFF_HEIGHTS) as u64;
        let row_bytes = rect.cols as u64 * 4;
        for r in 0..rect.rows as u64 {
            let src = staged.offset + 16 + r * row_bytes;
            let dst = (heights + (rect.row as u64 + r) * cols as u64 + rect.col as u64) * 4;
            enc.copy_buffer_to_buffer(staged.buffer, src, &self.scene, dst, row_bytes);
        }
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("acoustics terrain rect"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.build_group, &[]);
        self.record_pyramid(&mut pass, rect);
        Ok(())
    }

    /// Record one dispatch: the copy of the staged dispatch, the source and field
    /// workgroups, and the copy into a readback slot mapped on submit.
    ///
    /// # Arguments
    ///
    /// * `enc` - the caller's encoder; submit it with the caller's next submission.
    /// * `staged` - the bytes [`GpuAcoustics::pack_dispatch`] wrote, staged.
    /// * `shape` - what `pack_dispatch` returned for them.
    ///
    /// # Returns
    ///
    /// [`Encoded::Dispatched`], or [`Encoded::SkippedBusy`] when the next slot has not
    /// been taken yet (nothing is recorded, and the skip is counted).
    ///
    /// # Errors
    ///
    /// [`AcousticsError::Limit`] over this pass's source or mover limit;
    /// [`AcousticsError::Misaligned`], [`AcousticsError::Shape`] for a bad staged slice.
    ///
    /// # Examples
    ///
    /// See the module docs.
    pub fn encode(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        staged: Staged<'_>,
        shape: DispatchShape,
    ) -> Result<Encoded, AcousticsError> {
        self.encode_timed(enc, staged, shape, None)
    }

    /// [`GpuAcoustics::encode`], with timestamps written at the start and end of the
    /// compute pass: how the engine's clock (and the bench) takes the pass's GPU span.
    ///
    /// # Arguments
    ///
    /// * `enc`, `staged`, `shape` - as for [`GpuAcoustics::encode`].
    /// * `timestamps` - where to write the pass's begin and end timestamps; the device
    ///   needs `TIMESTAMP_QUERY`.
    ///
    /// # Returns
    ///
    /// As [`GpuAcoustics::encode`].
    ///
    /// # Errors
    ///
    /// As [`GpuAcoustics::encode`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::acoustics::*;
    ///
    /// # fn setup() -> (GpuAcoustics, wgpu::Device, wgpu::QuerySet, Staged<'static>, DispatchShape) { unimplemented!() }
    /// let (mut acoustics, device, queries, staged, shape) = setup();
    /// let mut enc = device.create_command_encoder(&Default::default());
    /// let marks = wgpu::ComputePassTimestampWrites {
    ///     query_set: &queries, beginning_of_pass_write_index: Some(0),
    ///     end_of_pass_write_index: Some(1),
    /// };
    /// acoustics.encode_timed(&mut enc, staged, shape, Some(marks)).unwrap();
    /// ```
    pub fn encode_timed(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        staged: Staged<'_>,
        shape: DispatchShape,
        timestamps: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) -> Result<Encoded, AcousticsError> {
        check_staged(&staged, shape.bytes)?;
        if shape.bytes != dispatch_bytes(shape.sources as usize, shape.movers as usize) as u64 {
            return Err(AcousticsError::Shape("the shape's bytes do not match its counts"));
        }
        if shape.sources > self.limits.sources {
            return Err(AcousticsError::Limit {
                what: "sources",
                got: shape.sources as u64,
                max: self.limits.sources as u64,
            });
        }
        if shape.movers > self.limits.movers {
            return Err(AcousticsError::Limit {
                what: "movers",
                got: shape.movers as u64,
                max: self.limits.movers as u64,
            });
        }
        let Some((slot, sequence)) = self.ring.claim(shape.sources) else {
            return Ok(Encoded::SkippedBusy);
        };
        enc.copy_buffer_to_buffer(staged.buffer, staged.offset, &self.input, 0, staged.len);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("acoustics"),
                timestamp_writes: timestamps,
            });
            pass.set_bind_group(0, &self.query_group, &[]);
            if shape.sources > 0 {
                pass.set_pipeline(&self.sources);
                pass.dispatch_workgroups(shape.sources, 1, 1);
            }
            pass.set_pipeline(&self.field);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.ring.record(enc, &self.output, slot);
        Ok(Encoded::Dispatched { slot, sequence })
    }

    /// Copy the newest ready result out, and release every ready slot.
    ///
    /// A slot is ready once its map callback has fired, which happens inside one of the
    /// caller's `Queue::submit` calls after the GPU finished the dispatch. This never polls
    /// the device and never waits.
    ///
    /// # Arguments
    ///
    /// * `out` - overwritten with the newest ready dispatch's results.
    ///
    /// # Returns
    ///
    /// `true` when a result was copied. Older ready results are released unread and
    /// counted in [`AcousticCounters::overwritten_ready`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::acoustics::{GpuAcoustics, TickResults};
    ///
    /// # fn setup() -> GpuAcoustics { unimplemented!() }
    /// let mut acoustics = setup();
    /// let mut out = TickResults::default();
    /// if acoustics.take_ready(&mut out) {
    ///     assert!(out.len <= out.sources.len());
    /// }
    /// ```
    pub fn take_ready(&mut self, out: &mut TickResults) -> bool {
        self.ring.take(out)
    }

    /// Counts over the pass's life.
    ///
    /// # Returns
    ///
    /// Dispatches recorded, dispatches skipped on a busy slot, ready results released
    /// unread, and failed maps.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::acoustics::GpuAcoustics;
    ///
    /// # fn setup() -> GpuAcoustics { unimplemented!() }
    /// let c = setup().counters();
    /// println!("{} dispatches, {} skipped", c.dispatches, c.skipped_busy);
    /// ```
    pub fn counters(&self) -> AcousticCounters {
        self.ring.counters()
    }

    /// The limits this pass was built for.
    ///
    /// # Returns
    ///
    /// The [`AcousticLimits`] given to `new`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::acoustics::{AcousticLimits, GpuAcoustics};
    ///
    /// # fn setup() -> GpuAcoustics { unimplemented!() }
    /// assert!(setup().limits().sources <= AcousticLimits::MAX.sources);
    /// ```
    pub fn limits(&self) -> AcousticLimits {
        self.limits
    }

    /// Record a copy of the last dispatch's per-ray field paths into a caller buffer, for
    /// the oracles: 64 [`FieldRay`] records, 4,096 bytes.
    ///
    /// # Arguments
    ///
    /// * `enc` - the caller's encoder, after the dispatch it reads.
    /// * `dst` - a `COPY_DST` buffer of at least 4,096 bytes from `offset`.
    /// * `offset` - where in `dst`, bytes; a multiple of 4.
    ///
    /// # Returns
    ///
    /// Nothing; the copy is recorded.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::acoustics::GpuAcoustics;
    ///
    /// # fn setup() -> (GpuAcoustics, wgpu::Device, wgpu::Buffer) { unimplemented!() }
    /// let (acoustics, device, dst) = setup();
    /// let mut enc = device.create_command_encoder(&Default::default());
    /// acoustics.copy_field_rays(&mut enc, &dst, 0);
    /// ```
    pub fn copy_field_rays(&self, enc: &mut wgpu::CommandEncoder, dst: &wgpu::Buffer, offset: u64) {
        let bytes = 16 * 4 * MAX_FIELD_RAYS as u64;
        enc.copy_buffer_to_buffer(&self.output, shader::OUT_RAYS as u64 * 4, dst, offset, bytes);
    }
}
