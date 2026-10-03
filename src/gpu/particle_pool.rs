//! The effect-particle pool resident on the GPU; see [`GpuParticlePool`].

#![warn(missing_docs)]

use std::collections::VecDeque;

use crate::gpu::GpuContext;
use crate::particles::{for_each_in_burst, stride_tables, DIGIT_TABLES, QUARTER_SINE};
use crate::particles::{
    BackendPolicy, Burst, EffectRng, GpuResidency, Landing, ParticleClass, ParticleEffects,
    VelocityGrid, MAX_CLASSES,
};

/// Threads in a workgroup of every pool pass; the buffers are rounded up to a multiple.
const WG: u32 = 64;

/// Words of the frame header at the front of the frame buffer: the WGSL `Frame` struct
/// is 84 words; the emission data starts at 512 bytes, a storage-binding offset every
/// device accepts.
const HEADER_WORDS: usize = 128;
const FRAME_WORDS: usize = 84;

/// Words a staged particle takes: position 3, velocity 3, remaining, lifetime, size,
/// class.
const RECORD_WORDS: usize = 10;

/// Words of a segment-table entry: first particle (in the frame), count, kind, payload
/// offset (words from the start of the emission data).
const SEGMENT_WORDS: usize = 4;
const SEGMENT_RECORDS: u32 = 0;
const SEGMENT_BURST: u32 = 1;

/// Words of a burst descriptor's payload: the state before its first particle's draws,
/// origin 3, class, speed, lifetime and size ranges (2 each), lift.
const BURST_WORDS: usize = 12;

/// Words of the emission region past `max_emit_per_frame * RECORD_WORDS`. A frame's
/// table and payloads take at most 10 words a particle plus a few at the ends (a record
/// run costs 4 over its records and a burst of two or more at least 4 under them; only
/// a burst cut by the frame's edges can be a single particle, 6 over); this covers the
/// ends, and a frame stops early rather than overrun it.
const DATA_SLACK_WORDS: usize = 64;

/// Words of the jump tables per level: a byte-sliced 32x32 GF(2) matrix.
const JUMP_TABLE_WORDS: usize = 1024;

/// Words a landing takes: position 3, impact speed, class, size.
const LANDING_WORDS: usize = 6;

const FLAG_AIR: u32 = 1;
const FLAG_GROUND: u32 = 2;

/// Words of the state buffer; see [`GpuParticlePool::state`].
const STATE_WORDS: usize = 9;

/// Words of the indirect-arguments buffer; see [`GpuParticlePool::draw_args`].
const ARGS_WORDS: usize = 12;

const POOL_SHADER: &str = include_str!("particle_pool.wgsl");
const EMIT_SHADER: &str = include_str!("particle_pool_emit.wgsl");
const AIR_SHADER: &str = include_str!("particle_pool_air.wgsl");

/// How the air field is stored on the device and fetched by the integrate.
///
/// The CPU fetch, [`VelocityGrid::sample`], is a trilinear interpolation in `f32`. A
/// hardware-filtered fetch is one texture sample, but the filter's weights have a few
/// bits of fraction (8 on current desktop hardware), so it differs from the CPU's by up
/// to about `3 D / 256` with `D` the largest difference between neighbouring cells, and
/// `rgba16float` storage adds a relative `2^-11` of the field's magnitude. The tests
/// measure both against the CPU fetch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FieldFormat {
    /// `rgba16float`, hardware trilinear filtering. 8 bytes a cell, filterable on every
    /// device. The cells are written as `f32` and converted on the device by a one-pass
    /// compute shader, so the CPU does no conversion.
    #[default]
    F16Filtered,
    /// `rgba32float`, hardware trilinear filtering. 16 bytes a cell. Needs the device
    /// feature `FLOAT32_FILTERABLE`.
    F32Filtered,
    /// `rgba32float`, eight loads and the CPU's seven lerps in `f32`: the CPU fetch's
    /// arithmetic, operation for operation. 16 bytes a cell, any device.
    F32Exact,
}

impl FieldFormat {
    fn texture_format(self) -> wgpu::TextureFormat {
        match self {
            FieldFormat::F16Filtered => wgpu::TextureFormat::Rgba16Float,
            FieldFormat::F32Filtered | FieldFormat::F32Exact => wgpu::TextureFormat::Rgba32Float,
        }
    }

    fn filtered(self) -> bool {
        !matches!(self, FieldFormat::F32Exact)
    }

    /// Bytes a cell of the field takes on the device, staging included.
    ///
    /// # Returns
    ///
    /// 24 for [`FieldFormat::F16Filtered`] (8 in the texture, 16 in the `f32` staging
    /// the conversion reads), 16 for the `f32` formats.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::FieldFormat;
    /// assert_eq!(FieldFormat::F32Exact.bytes_per_cell(), 16);
    /// ```
    pub fn bytes_per_cell(self) -> usize {
        match self {
            FieldFormat::F16Filtered => 8 + 16,
            FieldFormat::F32Filtered | FieldFormat::F32Exact => 16,
        }
    }
}

/// The shape of a [`GpuParticlePool`], fixed at construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuPoolConfig {
    /// The most particles alive at once. At least one, and at most
    /// `65535 * 64` (4,194,240), the most one dispatch dimension covers.
    pub capacity: u32,
    /// The most staged particles one [`GpuParticlePool::encode`] places, at most the
    /// capacity (a larger value is lowered to it). Particles staged beyond it wait for the
    /// next frame, in order; a burst cut by it carries its remainder over. Sizes the frame
    /// buffer: 40 bytes each (what [`GpuParticlePool::emit_one`] records take; a burst
    /// takes 64 bytes whatever its count), and the jump tables: 60 KB per hex digit of
    /// the largest particle index, 240 KB at the default 65,536.
    pub max_emit_per_frame: u32,
    /// The most landings one frame records; later landings that frame are counted in
    /// the state buffer but not written. 24 bytes each.
    pub landing_capacity: u32,
    /// How the air field is stored and fetched.
    pub field: FieldFormat,
    /// The vertex count written into the draw arguments, for a renderer that draws each
    /// particle as `sprite_vertices` vertices of one instance (6 for two triangles).
    pub sprite_vertices: u32,
}

impl GpuPoolConfig {
    /// A configuration for `capacity` particles with the defaults: up to 65,536 placed a
    /// frame (or the capacity, if smaller), 4,096 landings a frame, an `rgba16float`
    /// field, 6 vertices a sprite.
    ///
    /// # Arguments
    ///
    /// * `capacity` - the most particles alive at once.
    ///
    /// # Returns
    ///
    /// The configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::{FieldFormat, GpuPoolConfig};
    /// let config = GpuPoolConfig::new(100_000);
    /// assert_eq!(config.max_emit_per_frame, 65_536);
    /// assert_eq!(config.field, FieldFormat::F16Filtered);
    /// ```
    pub fn new(capacity: u32) -> GpuPoolConfig {
        GpuPoolConfig {
            capacity,
            max_emit_per_frame: capacity.clamp(1, 65_536),
            landing_capacity: 4_096,
            field: FieldFormat::default(),
            sprite_vertices: 6,
        }
    }
}

/// Why a [`GpuParticlePool`] could not be built or could not take a field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuPoolError {
    /// The device lacks a feature the configuration needs.
    MissingFeature(wgpu::Features),
    /// The configuration asks for more than the device allows, or for nothing.
    Limit(&'static str),
}

impl core::fmt::Display for GpuPoolError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GpuPoolError::MissingFeature(features) => write!(f, "the device lacks {features:?}"),
            GpuPoolError::Limit(what) => write!(f, "past a limit: {what}"),
        }
    }
}

impl std::error::Error for GpuPoolError {}

/// The engine's terrain heights, for ground contact on the device.
///
/// A 2D float texture of corner heights in metres (the engine's `R32Float` height map),
/// corner `(ix, iz)` at world `(min[0] + ix * cell, min[1] + iz * cell)`. The height
/// between corners is bilinear, and outside the map the nearest edge corner is read, as
/// the engine's own heightfield computes it on the CPU.
#[derive(Debug, Clone)]
pub struct GroundHeights {
    /// A view of the height texture. Its format must be readable as an unfilterable
    /// float (`R32Float`, `R16Float`), since the heights are loaded, not sampled.
    pub view: wgpu::TextureView,
    /// World `x` and `z` of corner `(0, 0)`, metres.
    pub min: [f32; 2],
    /// Spacing between corners, metres.
    pub cell: f32,
    /// Corners across (`x`) and down (`z`): the texture's width and height.
    pub corners: [u32; 2],
}

/// Where [`GpuParticlePool::encode_timed`] writes its timestamps: each `Some(i)` writes
/// the pass's start at query `i` and its end at `i + 1`.
///
/// The query set must hold timestamp queries, which needs the device feature
/// `TIMESTAMP_QUERY`.
#[derive(Debug, Clone, Copy)]
pub struct PoolTimestamps<'a> {
    /// The timestamp query set.
    pub query_set: &'a wgpu::QuerySet,
    /// The field conversion pass, written only on a frame that converts a new field
    /// (`rgba16float` only; see [`GpuParticlePool::field_pending`]).
    pub field: Option<u32>,
    /// The placement pass: new particles into slots, and the counts settled.
    pub emit: Option<u32>,
    /// The integrate pass.
    pub integrate: Option<u32>,
}

/// The pool's counters, read back by [`GpuParticlePool::read_counts_blocking`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GpuPoolCounts {
    /// Slots on the free stack.
    pub free: u32,
    /// Live particles.
    pub live: u32,
    /// One past the highest slot ever used: the integrate's and the draw's range.
    pub high_water: u32,
    /// Where the next overwrite of a full pool lands.
    pub cursor: u32,
    /// Landings in the last integrate, including any past the landing capacity.
    pub landings: u32,
    /// Particles retired since the pool was built.
    pub retired: u32,
    /// Particles placed since the pool was built, overwrites included.
    pub placed: u32,
    /// Particles placed over a live one because the pool was full.
    pub overwritten: u32,
    /// Particles not placed: the pool was full and the cursor's slot held a particle
    /// placed the same frame.
    pub dropped: u32,
}

/// One slot's contents, from [`GpuParticlePool::read_slots_blocking`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GpuSlot {
    /// Position, metres.
    pub position: [f32; 3],
    /// Seconds of life remaining; the slot is live while this is above zero.
    pub remaining: f32,
    /// Velocity, m/s.
    pub velocity: [f32; 3],
    /// Total lifetime, seconds.
    pub lifetime: f32,
    /// Class slot.
    pub class: u8,
    /// The renderer's scalar, as emitted.
    pub size: f32,
}

/// Bytes the pool needs in one of its own device resources, handed to a host that
/// carries its uploads through a staging ring of its own
/// ([`GpuParticlePool::stage_frame_with`], [`GpuParticlePool::upload_field_with`]).
///
/// The host copies `bytes` into its staging memory and records the copy (a
/// `copy_buffer_to_buffer` or a `copy_buffer_to_texture` with exactly this layout) so
/// that it executes before the pool's passes that read it: on the same command encoder
/// ahead of [`GpuParticlePool::encode_staged`], or in an earlier submission. Every length
/// and offset already meets wgpu's copy alignment (4 bytes for a buffer, 256-byte rows for
/// a texture).
#[derive(Debug, Clone, Copy)]
pub enum PoolWrite<'a> {
    /// Bytes for a buffer.
    Buffer {
        /// The pool's buffer to copy into.
        destination: &'a wgpu::Buffer,
        /// Byte offset in `destination`; a multiple of 4.
        offset: u64,
        /// The bytes; a multiple of 4 long.
        bytes: &'a [u8],
    },
    /// Bytes for the whole of a 3D texture, from offset 0 of the source.
    Texture {
        /// The pool's texture to copy into, mip level 0, origin zero, all aspects.
        destination: &'a wgpu::Texture,
        /// The layout of `bytes`: `bytes_per_row` a multiple of 256, `rows_per_image` the
        /// texture's height, `offset` 0 (add the staging offset when recording the copy).
        layout: wgpu::TexelCopyBufferLayout,
        /// The extent to copy: the whole texture.
        size: wgpu::Extent3d,
        /// The bytes, `bytes_per_row * height * depth` long.
        bytes: &'a [u8],
    },
}

impl PoolWrite<'_> {
    /// Carry this write out through `queue` at once: `write_buffer` or `write_texture`.
    /// What the pool's own [`GpuParticlePool::encode`] and
    /// [`GpuParticlePool::upload_field`] amount to; for a host without a ring.
    ///
    /// # Arguments
    ///
    /// * `queue` - the queue of the device the pool is on.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// pool.stage_frame_with(1.0 / 60.0, |write| write.write_now(&gpu.queue));
    /// let mut encoder = gpu.device.create_command_encoder(&Default::default());
    /// pool.encode_staged(&mut encoder, None);
    /// gpu.queue.submit([encoder.finish()]);
    /// ```
    pub fn write_now(self, queue: &wgpu::Queue) {
        match self {
            PoolWrite::Buffer {
                destination,
                offset,
                bytes,
            } => queue.write_buffer(destination, offset, bytes),
            PoolWrite::Texture {
                destination,
                layout,
                size,
                bytes,
            } => queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: destination,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytes,
                layout,
                size,
            ),
        }
    }
}

/// Emission staged and not yet placed.
#[derive(Debug, Clone, Copy)]
enum Pending {
    /// This many records, the next in `upload`.
    Records(usize),
    /// A burst's remaining particles: the descriptor payload (its state the one before
    /// the first remaining particle's draws) and how many remain.
    Burst([u32; BURST_WORDS], usize),
}

/// `state` advanced past `particles` particles' draws.
fn skip_particles(state: u32, particles: usize) -> u32 {
    let mut rng = EffectRng::new(state);
    rng.jump(Burst::DRAWS_PER_PARTICLE as u64 * particles as u64);
    rng.state()
}

/// The pool's device buffers and textures that depend on the field and the ground.
struct AirBinding {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// `[nx, ny, nz]` of the grid the texture holds.
    dims: [usize; 3],
    /// `f32` cells for the `rgba16float` conversion, and its bind group.
    staging: Option<(wgpu::Buffer, wgpu::BindGroup)>,
}

/// The effect-particle pool resident on the GPU: [`ParticleEffects`] with its integrate
/// in a compute shader and its particles in device buffers the renderer draws from.
///
/// # Why
///
/// The CPU pool's integrate costs about 3 ns a particle, and moving air costs more: at
/// a 20 Hz field each particle re-samples a third of its frames, a 10 ns trilinear fetch
/// each time, which puts [`ParticleEffects::integrate_in_air`] at 2.2 to 2.6 times
/// [`ParticleEffects::integrate`]. On the GPU the fetch is one texture sample, so every
/// particle samples the air every frame and the field can update as often as it likes
/// at no per-particle CPU cost. The particles never come back: the renderer draws from
/// the same buffers the integrate writes, and readback is confined to debug and test
/// paths.
///
/// The GPU is for presentation only. Nothing here is lockstep state.
///
/// # A frame
///
/// 1. The host calls [`GpuParticlePool::emit`] / [`GpuParticlePool::emit_one`] as it
///    would on the CPU pool. A burst is staged as a 64-byte descriptor whatever its
///    count, an explicit particle as a 40-byte record; nothing touches the device (see
///    "Emission" below).
/// 2. When the field has a new frame (10 to 20 times a second), the host calls
///    [`GpuParticlePool::upload_field`] from the frame thread: one copy of the cells.
/// 3. [`GpuParticlePool::encode`] writes one buffer (this frame's constants and the
///    staged descriptors and records, a single `write_buffer`) and records, into the
///    host's encoder, the placement of the new particles into free slots (expanding the
///    bursts as it goes), a one-thread pass that settles the counts, and one indirect
///    dispatch of the integrate over the slots in use.
/// 4. The renderer draws from [`GpuParticlePool::positions`] and the others, with the
///    instance count from [`GpuParticlePool::draw_args`], and drains the landings.
///
/// A host with a staging ring of its own replaces the pool's queue writes with copies
/// from the ring: [`GpuParticlePool::stage_frame_with`] then
/// [`GpuParticlePool::encode_staged`] in place of `encode`, and
/// [`GpuParticlePool::upload_field_with`] in place of `upload_field`. Each hands the host
/// the bytes and their destination as a [`PoolWrite`]; the device contents are the same
/// to the bit either way. wgpu allocates a fresh staging buffer for every `write_buffer`,
/// which is what a ring saves.
///
/// # Which backend: decided once, at startup
///
/// A host decides at startup, by whether an adapter exists, whether its effect particles
/// live in this pool or in a CPU [`ParticleEffects`], and keeps that for the run. There
/// is no switch between the two at run time: moving particles from the device back to
/// the CPU would need a readback, which costs a frame, and the device's fixed cost (about
/// 10 us a frame) makes it the cheaper home from a few thousand particles, which every
/// scene that matters exceeds. [`GpuParticlePool::register`] still declares the pool to
/// a [`BackendPolicy`] for hosts that report the policy's figures.
///
/// # Slots and the free list
///
/// A particle keeps the slot it was placed in until it retires; a slot is live while
/// its remaining life is above zero. Retired slots go on a free stack (an index buffer
/// and an atomic count) in the integrate itself, and emission takes slots off its top.
/// Every slot is either live or on the stack, exactly once. The integrate runs over the
/// highest slot ever used (the high water), skipping dead slots with one 16-byte read.
///
/// A free list rather than compaction because compaction needs a second copy of every
/// array (or a prefix sum of several passes) and moves every particle every frame;
/// the free list touches only the particles born or retired that frame (about 2.4% of
/// the pool a frame in the live sparks-and-dust workload), and slots that do not move
/// are what a renderer's per-particle state (a sort key, a trail) can hang off. Its
/// cost is the dead slots under the high water, which the integrate skips.
///
/// When the pool is full, new particles replace live ones at a rotating cursor, as the
/// CPU pool replaces its oldest: the newest event is the one drawn whole. Within one
/// frame's batch the newest particles take the free slots, and an older one whose turn
/// of the cursor falls on a slot filled that same frame is dropped rather than replace
/// a newer particle ([`GpuPoolCounts::dropped`]).
///
/// # Emission
///
/// [`GpuParticlePool::emit`] does not draw a burst's particles on the CPU. It stages a
/// descriptor: the [`EffectRng`] state, the origin, the class, the speed, lifetime and
/// size ranges and the lift (12 words), plus a 4-word segment entry. It then moves the
/// CPU's stream on by [`Burst::DRAWS_PER_PARTICLE`] draws a particle with
/// [`EffectRng::jump`], so the stream is exactly where [`ParticleEffects::emit`] would
/// leave it.
///
/// On the device the placement passes find each new particle's segment. Particle `i` of
/// a burst starts from the state `5 i` draws in: xorshift32 is linear over GF(2), so that
/// is one byte-sliced 32x32 bit matrix per non-zero hex digit of `i`, from the powers
/// `M^(5 d 16^k)` uploaded once at construction. It then takes its five draws with the
/// CPU's arithmetic, in the CPU's order:
///
/// - `f32` adds and multiplies are correctly rounded on the device as on the CPU;
/// - every product that feeds an add is forced to its own rounding first, since the
///   driver may otherwise fuse the two (and does: without it a velocity is 1 ulp off);
/// - the square root and the divide are the device's own result settled by exact integer
///   comparisons, so correctly rounded as the CPU's are;
/// - the direction's sine and cosine are [`sin_cos_turn`](crate::particles::sin_cos_turn),
///   integer arithmetic on the same table.
///
/// So a seed gives the same particles, to the bit, on the device and on the CPU pool.
/// Explicit particles ([`GpuParticlePool::emit_one`], [`GpuParticlePool::adopt`], and a
/// burst of one) stay 40-byte records drawn on the CPU.
///
/// A device may flush subnormal floats. No draw produces one from burst ranges of
/// normal magnitude; a subnormal `lift` or range end is read as zero on the device.
///
/// # Memory
///
/// 44 bytes a particle at capacity: position and remaining life (16), velocity and
/// total lifetime (16), class and size (8), the free stack (4). 44 MB at a million. The
/// CPU pool's 12-byte air sample has no counterpart, since every particle samples the
/// field every frame. Fixed costs besides: the frame buffer (512 bytes plus 40 a staged
/// particle at [`GpuPoolConfig::max_emit_per_frame`], plus the emission's tables: 60 KB
/// a hex digit of the largest index, 4 digits at the default, and the 16 KB sine), the
/// landings (24 bytes each), the
/// field texture (8 bytes a cell in `rgba16float`, 16 in `rgba32float`, plus a 16-byte
/// staging cell for `rgba16float`). [`GpuParticlePool::bytes`] adds it up.
///
/// # Cost
///
/// Measured 2026-10-03 on an RTX 3090 (Vulkan, indirect-call validation off as the
/// engine's release build runs it), beside another build, on the live sparks-and-dust pool
/// of `benches/particle_effects.rs` with both classes on an `rgba16float` field updated at
/// 20 Hz (`examples/pool_gpu_bench.rs`, timestamp queries, medians of 7 interleaved
/// rounds): the integrate pass 4.5 us at 10k, 10.2 us at 100k and 90.8 us at 1M (0.092 ns
/// a particle, the card's memory bandwidth at 72 bytes a particle); the emit pass 6.9 to
/// 13.9 us. A frame is 9.7 us plus 0.096 ns a particle of device time, under the CPU's
/// `integrate` (3 ns a particle) from about 3,300 particles. The field costs the integrate
/// 4 to 19% over no field; the three [`FieldFormat`]s are within 12% of each other.
///
/// On the frame thread the pool costs a descriptor and a jump of the stream a burst (no
/// per-particle work: POOL-GPU-RNG figures below), one `write_buffer` a frame and one a
/// field update. wgpu 30 allocates a fresh staging buffer for every `write_buffer`;
/// beside a compiler that measured 150 to 260 us a call regardless of size.
///
/// Measured 2026-10-03 on the same machine, no neighbour build seen at the start or the
/// end of the run (`examples/pool_gpu_bench.rs`, 7 interleaved rounds, the 0.3.3 path of
/// drawing every particle on the CPU and staging it as a record measured in the same
/// process as the comparison). At 10k, 100k and 1M live (about 240, 2,400 and 24,000
/// born a frame):
///
/// | | 10k | 100k | 1M |
/// |---|---|---|---|
/// | staging the emission (CPU) | 0.9 us | 1.0 us | 1.6 us |
/// | the same as records | 9.8 us | 91 us | 862 us |
/// | the frame's buffer write | 640 B | 640 B | 640 B |
/// | the same as records | 10 KB | 97 KB | 964 KB |
/// | the emit pass (device) | 9.9 us | 10.3 us | 17.7 us |
/// | the same as records | 7.4 us | 7.5 us | 14.3 us |
///
/// The emit pass grows by 2.5 to 3.4 us: each new particle's jump, integer sine and
/// cosine and settled square roots and divides, a short serial chain per thread.
///
/// # Determinism
///
/// Emission is exact: a seed gives the CPU pool's particles to the bit (see "Emission").
/// Beyond that none is promised. The order particles take free slots in, and the order
/// landings are appended in, depend on the device's scheduling; the integrate's
/// positions differ from the CPU's by float reordering (the device may fuse a multiply
/// and an add) and, with a filtered field, by the texture filter's precision. The tests
/// bound both. (With a power-of-two step, no field and no ground the integrate is exact
/// too, which is how the emission test follows particles to retirement.)
///
/// # Examples
///
/// ```no_run
/// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
/// use rs_physics::particles::{Burst, EffectRng, ParticleClass, VelocityGrid};
///
/// let gpu = GpuContext::new().expect("a GPU");
/// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(100_000)).unwrap();
/// pool.set_class(1, ParticleClass { gravity: 1.6, drag: 3.4, restitution: 0.0 });
/// pool.set_swirl(1, 1.0);
///
/// let mut air = VelocityGrid::new([-32.0, 0.0, -32.0], 2.0, [32, 32, 32]).unwrap();
/// air.fill([2.0, 0.0, 0.0]);
/// pool.upload_field(&air); // once a field update, not once a frame
///
/// let mut rng = EffectRng::new(7);
/// for _ in 0..60 {
///     pool.emit(&Burst {
///         origin: [0.0, 4.0, 0.0],
///         class: 1,
///         count: 100,
///         speed: 1.0..3.0,
///         lifetime: 2.0..4.0,
///         size: 1.0..1.0,
///         lift: 0.5,
///     }, &mut rng);
///     let mut encoder = gpu.device.create_command_encoder(&Default::default());
///     pool.encode(&mut encoder, 1.0 / 60.0);
///     // ... the renderer's passes draw from pool.positions() here ...
///     gpu.queue.submit([encoder.finish()]);
/// }
/// ```
pub struct GpuParticlePool {
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: GpuPoolConfig,
    /// Slots allocated: the capacity rounded up to a workgroup.
    slots: u32,

    classes: [ParticleClass; MAX_CLASSES],
    swirl: [f32; MAX_CLASSES],

    /// The frame buffer's contents: the header, then every staged record, oldest
    /// first, as 0.3.3 staged them; a frame appends its burst descriptors and segment
    /// table after the records it takes.
    upload: Vec<u32>,
    /// Staged emission, oldest first: runs of the records in `upload`, and bursts.
    pending: VecDeque<Pending>,
    /// Particles in `pending`.
    pending_count: usize,
    /// Scratch for a frame's burst payloads, its segment table, and the records it
    /// leaves staged.
    payload: Vec<u32>,
    table: Vec<u32>,
    carry: Vec<u32>,
    /// Words of the emission region; the jump tables follow it in the frame buffer.
    data_words: usize,
    /// Hex digits the jump tables cover: particle indices within a frame are below
    /// `16^jump_levels`.
    jump_levels: u32,

    frame: wgpu::Buffer,
    pos_life: wgpu::Buffer,
    vel: wgpu::Buffer,
    meta: wgpu::Buffer,
    free: wgpu::Buffer,
    state: wgpu::Buffer,
    landings: wgpu::Buffer,
    args: wgpu::Buffer,

    air: AirBinding,
    /// The header's field words for the current field, `None` with no field.
    air_words: Option<[f32; 20]>,
    field_pending: bool,
    /// The field's cells with each row padded to 256 bytes, for a texture copy from a
    /// host's staging ring ([`GpuParticlePool::upload_field_with`]); empty until needed.
    field_scratch: Vec<u8>,
    /// A frame staged by [`GpuParticlePool::stage_frame_with`] and not yet encoded: its
    /// step and the records it places.
    prepared: Option<(f32, usize)>,
    last_plume_step: Option<u64>,
    ground: Option<GroundHeights>,
    ground_placeholder: wgpu::TextureView,
    sampler: wgpu::Sampler,

    main_group: wgpu::BindGroup,
    air_group: wgpu::BindGroup,
    args_group: wgpu::BindGroup,
    air_layout: wgpu::BindGroupLayout,
    convert_layout: wgpu::BindGroupLayout,

    place_free: wgpu::ComputePipeline,
    place_overwrite: wgpu::ComputePipeline,
    finalize: wgpu::ComputePipeline,
    integrate: wgpu::ComputePipeline,
    convert: Option<wgpu::ComputePipeline>,
    module_source: String,
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

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// The section of `particle_pool_air.wgsl` after the marker `//== name`.
fn air_section(name: &str) -> &'static str {
    let marker = format!("//== {name}");
    let start = AIR_SHADER.find(&marker).expect("air shader section") + marker.len();
    let rest = &AIR_SHADER[start..];
    let end = rest.find("//== ").unwrap_or(rest.len());
    &rest[..end]
}

fn bytes_of(words: &[u32]) -> &[u8] {
    bytemuck::cast_slice(words)
}

impl GpuParticlePool {
    /// A pool of `config.capacity` slots on `gpu`'s device, every slot free.
    ///
    /// Allocates every device buffer here, once; nothing per frame allocates on the
    /// device. Compiles the pool's shaders.
    ///
    /// # Arguments
    ///
    /// * `gpu` - the context; [`GpuContext::from_device`] to run on the engine's device.
    /// * `config` - capacity, per-frame bounds and field format.
    ///
    /// # Returns
    ///
    /// The pool, with the default class table (real gravity, no drag), no field and no
    /// ground.
    ///
    /// # Errors
    ///
    /// [`GpuPoolError::MissingFeature`] for [`FieldFormat::F32Filtered`] on a device
    /// without `FLOAT32_FILTERABLE`; [`GpuPoolError::Limit`] for a zero capacity or
    /// emission bound, a capacity past one dispatch, or arrays past the device's
    /// storage-binding size.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(1 << 20)).unwrap();
    /// assert_eq!(pool.capacity(), 1 << 20);
    /// ```
    pub fn new(
        gpu: &GpuContext,
        mut config: GpuPoolConfig,
    ) -> Result<GpuParticlePool, GpuPoolError> {
        config.max_emit_per_frame = config.max_emit_per_frame.min(config.capacity);
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();
        if config.capacity == 0 || config.max_emit_per_frame == 0 {
            return Err(GpuPoolError::Limit(
                "capacity and max_emit_per_frame must be at least 1",
            ));
        }
        let limits = device.limits();
        let slots = config.capacity.div_ceil(WG) * WG;
        if slots / WG > limits.max_compute_workgroups_per_dimension {
            return Err(GpuPoolError::Limit("capacity past one dispatch"));
        }
        // Particle indices within a frame run below max_emit_per_frame: hex digits of the
        // largest.
        let index_bits = (32 - (config.max_emit_per_frame - 1).leading_zeros()).max(1);
        let jump_levels = index_bits.div_ceil(4);
        let data_words = config.max_emit_per_frame as usize * RECORD_WORDS + DATA_SLACK_WORDS;
        let table_words =
            jump_levels as usize * DIGIT_TABLES * JUMP_TABLE_WORDS + QUARTER_SINE.len();
        if (slots as u64) * 16 > limits.max_storage_buffer_binding_size as u64
            || ((data_words + table_words) as u64) * 4
                > limits.max_storage_buffer_binding_size as u64
        {
            return Err(GpuPoolError::Limit(
                "arrays past the device's storage binding size",
            ));
        }
        if limits.max_storage_buffers_per_shader_stage < 8 {
            return Err(GpuPoolError::Limit("the pool binds 8 storage buffers"));
        }
        if config.field == FieldFormat::F32Filtered
            && !device
                .features()
                .contains(wgpu::Features::FLOAT32_FILTERABLE)
        {
            return Err(GpuPoolError::MissingFeature(
                wgpu::Features::FLOAT32_FILTERABLE,
            ));
        }

        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let frame = buffer(
            "pool frame",
            ((HEADER_WORDS + data_words + table_words) * 4) as u64,
            wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        );
        // The emission's constant tables after the emission region, written once: the
        // jump tables (M^(5 d 16^k), byte-sliced) and the quarter-wave sine.
        let mut tables = stride_tables(Burst::DRAWS_PER_PARTICLE, jump_levels as usize);
        tables.extend(QUARTER_SINE.iter().map(|&v| v as u32));
        queue.write_buffer(
            &frame,
            ((HEADER_WORDS + data_words) * 4) as u64,
            bytes_of(&tables),
        );
        // Zeroed at creation, so every slot starts dead.
        let pos_life = buffer("pool position and life", slots as u64 * 16, rw);
        let vel = buffer("pool velocity and lifetime", slots as u64 * 16, rw);
        let meta = buffer("pool class and size", slots as u64 * 8, rw);
        let free = buffer("pool free stack", slots as u64 * 4, rw);
        let state = buffer("pool state", (STATE_WORDS * 4) as u64, rw);
        let landings = buffer(
            "pool landings",
            (config.landing_capacity.max(1) as usize * LANDING_WORDS * 4) as u64,
            rw,
        );
        let args = buffer(
            "pool indirect arguments",
            (ARGS_WORDS * 4) as u64,
            rw | wgpu::BufferUsages::INDIRECT,
        );

        // Every slot free, the lowest on top so the pool fills from slot 0 up.
        let stack: Vec<u32> = (0..config.capacity).rev().collect();
        queue.write_buffer(&free, 0, bytes_of(&stack));
        let mut initial = [0u32; STATE_WORDS];
        initial[0] = config.capacity;
        queue.write_buffer(&state, 0, bytes_of(&initial));
        // A first frame with nothing to integrate.
        queue.write_buffer(
            &args,
            0,
            bytes_of(&[0u32, 1, 1, 0, 1, 1, 0, 0, config.sprite_vertices, 0, 0, 0]),
        );

        let main_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pool main"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
                storage_entry(7, false),
            ],
        });
        let filtered = config.field.filtered();
        let mut air_entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float {
                    filterable: filtered,
                },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        }];
        if filtered {
            air_entries.push(wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        air_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        let air_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pool air and ground"),
            entries: &air_entries,
        });
        let args_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pool arguments"),
            entries: &[storage_entry(0, false)],
        });
        let convert_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pool field conversion"),
            entries: &[
                storage_entry(0, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });

        let module_source = format!(
            "{}\n{}\n{}",
            air_section(if filtered { "filtered" } else { "exact" }),
            POOL_SHADER,
            EMIT_SHADER
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle pool"),
            source: wgpu::ShaderSource::Wgsl(module_source.as_str().into()),
        });
        let layout_two = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pool main and air"),
            bind_group_layouts: &[Some(&main_layout), Some(&air_layout)],
            immediate_size: 0,
        });
        let layout_three = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pool main, air and arguments"),
            bind_group_layouts: &[Some(&main_layout), Some(&air_layout), Some(&args_layout)],
            immediate_size: 0,
        });
        let pipeline = |layout: &wgpu::PipelineLayout, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let place_free = pipeline(&layout_three, "place_free");
        let place_overwrite = pipeline(&layout_two, "place_overwrite");
        let finalize = pipeline(&layout_three, "finalize");
        let integrate = pipeline(&layout_two, "integrate");
        let convert = if config.field == FieldFormat::F16Filtered {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("pool field conversion"),
                source: wgpu::ShaderSource::Wgsl(air_section("convert").into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pool field conversion"),
                bind_group_layouts: &[Some(&convert_layout)],
                immediate_size: 0,
            });
            Some(
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("convert"),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some("convert"),
                    compilation_options: Default::default(),
                    cache: None,
                }),
            )
        } else {
            None
        };

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("pool air"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let ground_placeholder = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("pool no ground"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let air = Self::air_binding(&device, &convert_layout, config.field, [2, 2, 2]);

        let main_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool main"),
            layout: &main_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &frame,
                        offset: 0,
                        size: wgpu::BufferSize::new((HEADER_WORDS * 4) as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &frame,
                        offset: (HEADER_WORDS * 4) as u64,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pos_life.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: vel.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: free.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: landings.as_entire_binding(),
                },
            ],
        });
        let args_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool arguments"),
            layout: &args_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: args.as_entire_binding(),
            }],
        });
        let air_group = Self::air_group(
            &device,
            &air_layout,
            filtered,
            &air.view,
            &sampler,
            &ground_placeholder,
        );

        let mut upload = Vec::with_capacity(HEADER_WORDS + data_words);
        upload.resize(HEADER_WORDS, 0);

        Ok(GpuParticlePool {
            device,
            queue,
            config,
            slots,
            classes: [ParticleClass::default(); MAX_CLASSES],
            swirl: [0.0; MAX_CLASSES],
            upload,
            pending: VecDeque::new(),
            pending_count: 0,
            payload: Vec::new(),
            table: Vec::new(),
            carry: Vec::new(),
            data_words,
            jump_levels,
            frame,
            pos_life,
            vel,
            meta,
            free,
            state,
            landings,
            args,
            air,
            air_words: None,
            field_pending: false,
            field_scratch: Vec::new(),
            prepared: None,
            last_plume_step: None,
            ground: None,
            ground_placeholder,
            sampler,
            main_group,
            air_group,
            args_group,
            air_layout,
            convert_layout,
            place_free,
            place_overwrite,
            finalize,
            integrate,
            convert,
            module_source,
        })
    }

    fn air_binding(
        device: &wgpu::Device,
        convert_layout: &wgpu::BindGroupLayout,
        format: FieldFormat,
        dims: [usize; 3],
    ) -> AirBinding {
        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        if format == FieldFormat::F16Filtered {
            usage |= wgpu::TextureUsages::STORAGE_BINDING;
        }
        // The texture's x is the grid's z: the grid is z fastest, so its cells are the
        // texture's texels in order and upload with no transpose.
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pool air"),
            size: wgpu::Extent3d {
                width: dims[2] as u32,
                height: dims[1] as u32,
                depth_or_array_layers: dims[0] as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: format.texture_format(),
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let staging = if format == FieldFormat::F16Filtered {
            let cells = (dims[0] * dims[1] * dims[2]) as u64;
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pool air cells"),
                size: cells * 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pool field conversion"),
                layout: convert_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                ],
            });
            Some((buffer, group))
        } else {
            None
        };
        AirBinding {
            texture,
            view,
            dims,
            staging,
        }
    }

    fn air_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        filtered: bool,
        air: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        ground: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(air),
        }];
        if filtered {
            entries.push(wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: 2,
            resource: wgpu::BindingResource::TextureView(ground),
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool air and ground"),
            layout,
            entries: &entries,
        })
    }

    fn rebuild_air_group(&mut self) {
        let ground = match &self.ground {
            Some(g) => &g.view,
            None => &self.ground_placeholder,
        };
        self.air_group = Self::air_group(
            &self.device,
            &self.air_layout,
            self.config.field.filtered(),
            &self.air.view,
            &self.sampler,
            ground,
        );
    }

    /// Declare this pool to `policy` as a resident GPU backend, so
    /// [`BackendPolicy::choose`] may return [`Backend::Gpu`](crate::particles::Backend::Gpu).
    ///
    /// The policy's GPU costs stay at their seeds until something records a GPU
    /// timing; this pool never reads one back on its own (see [`GpuParticlePool`]).
    ///
    /// # Arguments
    ///
    /// * `policy` - the policy to register with, typically a CPU pool's
    ///   [`ParticleEffects::policy_mut`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// use rs_physics::particles::{Backend, ParticleEffects};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(1 << 20)).unwrap();
    /// let mut fx = ParticleEffects::with_capacity(1 << 20);
    /// pool.register(fx.policy_mut());
    /// assert!(fx.policy().gpu_available());
    /// assert_eq!(fx.policy_mut().choose(5_000_000), Backend::Gpu);
    /// ```
    pub fn register(&self, policy: &mut BackendPolicy) {
        policy.set_residency(GpuResidency::Resident);
        policy.set_gpu_available(true);
    }

    /// The configuration the pool was built with.
    ///
    /// # Returns
    ///
    /// The [`GpuPoolConfig`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// assert_eq!(pool.config().capacity, 64);
    /// ```
    pub fn config(&self) -> &GpuPoolConfig {
        &self.config
    }

    /// The most particles alive at once.
    ///
    /// # Returns
    ///
    /// [`GpuPoolConfig::capacity`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// assert_eq!(GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap().capacity(), 64);
    /// ```
    pub fn capacity(&self) -> u32 {
        self.config.capacity
    }

    /// Device bytes a particle slot takes: 44 (see [`GpuParticlePool`]).
    ///
    /// # Returns
    ///
    /// 44.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::GpuParticlePool;
    /// assert_eq!(GpuParticlePool::BYTES_PER_PARTICLE, 44);
    /// ```
    pub const BYTES_PER_PARTICLE: usize = 16 + 16 + 8 + 4;

    /// Every device byte the pool holds: the slots at capacity (rounded up to a
    /// workgroup of 64), the frame buffer, the landings, the counters and the field.
    ///
    /// # Returns
    ///
    /// Bytes.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(1 << 20)).unwrap();
    /// assert!(pool.bytes() > 44 << 20);
    /// ```
    pub fn bytes(&self) -> usize {
        let cells = self.air.dims.iter().product::<usize>();
        self.slots as usize * Self::BYTES_PER_PARTICLE
            + self.frame.size() as usize
            + self.landings.size() as usize
            + (STATE_WORDS + ARGS_WORDS) * 4
            + cells * self.config.field.bytes_per_cell()
    }

    // -- Classes --

    /// Set the behaviour every particle of class `index` shares, as
    /// [`ParticleEffects::set_class`]. Takes effect on the next [`Self::encode`].
    ///
    /// # Arguments
    ///
    /// * `index` - the class slot, `0..MAX_CLASSES`; larger values are clamped to the last.
    /// * `class` - gravity in m/s^2, drag in 1/s and ground restitution as a fraction.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// use rs_physics::particles::ParticleClass;
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// pool.set_class(2, ParticleClass { gravity: 1.6, drag: 3.4, restitution: 0.0 });
    /// assert_eq!(pool.class(2).drag, 3.4);
    /// ```
    pub fn set_class(&mut self, index: u8, class: ParticleClass) {
        self.classes[(index as usize).min(MAX_CLASSES - 1)] = class;
    }

    /// The behaviour of class `index`.
    ///
    /// # Arguments
    ///
    /// * `index` - the class slot; values past the table are clamped to the last slot.
    ///
    /// # Returns
    ///
    /// A copy of that class.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// use rs_physics::particles::ParticleClass;
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// assert_eq!(pool.class(0), ParticleClass::default());
    /// ```
    pub fn class(&self, index: u8) -> ParticleClass {
        self.classes[(index as usize).min(MAX_CLASSES - 1)]
    }

    /// Set how much of the moving air class `index` takes on, as
    /// [`ParticleEffects::set_swirl`]: the class relaxes towards the air at
    /// `fraction * drag`. 0, the default, ignores the air, and such a class integrates
    /// bit-identically with or without a field.
    ///
    /// # Arguments
    ///
    /// * `index` - the class slot; larger values are clamped to the last.
    /// * `fraction` - 0 to 1; values outside are clamped, and NaN is 0.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// pool.set_swirl(1, 1.0);
    /// assert_eq!(pool.swirl(1), 1.0);
    /// ```
    pub fn set_swirl(&mut self, index: u8, fraction: f32) {
        self.swirl[(index as usize).min(MAX_CLASSES - 1)] = fraction.max(0.0).min(1.0);
    }

    /// The fraction of the moving air class `index` takes on.
    ///
    /// # Arguments
    ///
    /// * `index` - the class slot; values past the table are clamped to the last slot.
    ///
    /// # Returns
    ///
    /// The fraction, 0 to 1.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// assert_eq!(GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap().swirl(3), 0.0);
    /// ```
    pub fn swirl(&self, index: u8) -> f32 {
        self.swirl[(index as usize).min(MAX_CLASSES - 1)]
    }

    /// Take every class and swirl fraction from a CPU pool, so the two integrate alike.
    ///
    /// # Arguments
    ///
    /// * `fx` - the CPU pool whose class table to copy.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// use rs_physics::particles::{ParticleClass, ParticleEffects};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let mut fx = ParticleEffects::with_capacity(64);
    /// fx.set_class(1, ParticleClass { gravity: 26.0, drag: 1.4, restitution: 0.32 });
    /// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// pool.copy_classes_from(&fx);
    /// assert_eq!(pool.class(1), fx.class(1));
    /// ```
    pub fn copy_classes_from(&mut self, fx: &ParticleEffects) {
        for c in 0..MAX_CLASSES as u8 {
            self.set_class(c, fx.class(c));
            self.set_swirl(c, fx.swirl(c));
        }
    }

    // -- Emission --

    fn stage(
        &mut self,
        pos: [f32; 3],
        vel: [f32; 3],
        remaining: f32,
        lifetime: f32,
        size: f32,
        class: u8,
    ) {
        let bits = |v: f32| v.to_bits();
        match self.pending.back_mut() {
            Some(Pending::Records(n)) => *n += 1,
            _ => self.pending.push_back(Pending::Records(1)),
        }
        self.pending_count += 1;
        self.upload.extend_from_slice(&[
            bits(pos[0]),
            bits(pos[1]),
            bits(pos[2]),
            bits(vel[0]),
            bits(vel[1]),
            bits(vel[2]),
            bits(remaining),
            bits(lifetime),
            bits(size),
            class as u32,
        ]);
    }

    /// Particles staged and not yet placed.
    ///
    /// # Returns
    ///
    /// The count; [`Self::encode`] places up to [`GpuPoolConfig::max_emit_per_frame`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// pool.emit_one([0.0; 3], [0.0; 3], 1.0, 1.0, 0);
    /// assert_eq!(pool.staged(), 1);
    /// ```
    pub fn staged(&self) -> usize {
        self.pending_count
    }

    /// Emit a burst, as [`ParticleEffects::emit`]: the same seed gives the same
    /// particles, to the bit. The burst is staged as a descriptor and expanded on the
    /// device by the next [`Self::encode`] (see "Emission" on [`GpuParticlePool`]); `rng`
    /// is moved on by [`Burst::DRAWS_PER_PARTICLE`] draws a particle at once, in at most
    /// 32 table lookups, so the CPU does no per-particle work. A burst of one is drawn on
    /// the CPU and staged as a record, which is smaller than a descriptor.
    ///
    /// Staged particles past the capacity drop the oldest, as before; a descriptor cut
    /// that way, or by [`GpuPoolConfig::max_emit_per_frame`], keeps its remainder with
    /// its state advanced past the particles cut.
    ///
    /// # Arguments
    ///
    /// * `burst` - where (metres), how many, and the ranges each particle's speed (m/s),
    ///   lifetime (s) and size are sampled from, and the lift. Finite values; a subnormal
    ///   one is read as zero on the device.
    /// * `rng` - the emission random stream; left where [`ParticleEffects::emit`] leaves
    ///   it.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// use rs_physics::particles::{Burst, EffectRng};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// pool.emit(&Burst {
    ///     origin: [0.0, 1.0, 0.0],
    ///     class: 0,
    ///     count: 16,
    ///     speed: 2.0..4.0,
    ///     lifetime: 0.5..1.0,
    ///     size: 1.0..1.0,
    ///     lift: 0.3,
    /// }, &mut EffectRng::new(1));
    /// assert_eq!(pool.staged(), 16);
    /// ```
    pub fn emit(&mut self, burst: &Burst, rng: &mut EffectRng) {
        match burst.count {
            0 => {}
            // A descriptor costs more words than the one record it would expand to.
            1 => for_each_in_burst(burst, rng, |pos, vel, life, size, class| {
                self.stage(pos, vel, life, life, size, class)
            }),
            count => {
                let f = |v: f32| v.to_bits();
                let payload = [
                    rng.state(),
                    f(burst.origin[0]),
                    f(burst.origin[1]),
                    f(burst.origin[2]),
                    burst.class.min((MAX_CLASSES - 1) as u8) as u32,
                    f(burst.speed.start),
                    f(burst.speed.end),
                    f(burst.lifetime.start),
                    f(burst.lifetime.end),
                    f(burst.size.start),
                    f(burst.size.end),
                    f(burst.lift),
                ];
                // The CPU stream moves on exactly as `ParticleEffects::emit` leaves it.
                rng.jump(Burst::DRAWS_PER_PARTICLE as u64 * count as u64);
                self.pending
                    .push_back(Pending::Burst(payload, count as usize));
                self.pending_count += count as usize;
            }
        }
        self.bound_staging();
    }

    /// Emit one particle with an explicit velocity, as [`ParticleEffects::emit_one`].
    ///
    /// # Arguments
    ///
    /// * `origin` - position, metres.
    /// * `velocity` - initial velocity, m/s.
    /// * `lifetime` - seconds before it retires; values below `f32::EPSILON` are raised to it.
    /// * `size` - the renderer's per-particle scalar.
    /// * `class` - class slot; out-of-range values are clamped.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// pool.emit_one([0.0, 2.0, 0.0], [3.0, 0.0, 0.0], 1.0, 1.0, 0);
    /// assert_eq!(pool.staged(), 1);
    /// ```
    pub fn emit_one(
        &mut self,
        origin: [f32; 3],
        velocity: [f32; 3],
        lifetime: f32,
        size: f32,
        class: u8,
    ) {
        let life = lifetime.max(f32::EPSILON);
        self.stage(
            origin,
            velocity,
            life,
            life,
            size,
            class.min((MAX_CLASSES - 1) as u8),
        );
        self.bound_staging();
    }

    /// Move every live particle of a CPU pool onto this one, and empty the CPU pool.
    ///
    /// For a host that started on the CPU pool and moves to this one once (the backend is
    /// decided at startup; see [`GpuParticlePool`]): the particles keep their positions,
    /// velocities, remaining and total lifetimes, sizes and classes, and are placed on the next [`Self::encode`] (over several frames if
    /// there are more than [`GpuPoolConfig::max_emit_per_frame`]). One upload, no
    /// readback. The class table is not copied; see [`Self::copy_classes_from`].
    ///
    /// # Arguments
    ///
    /// * `fx` - the CPU pool; empty afterwards.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// use rs_physics::particles::ParticleEffects;
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let mut fx = ParticleEffects::with_capacity(64);
    /// fx.emit_one([0.0; 3], [1.0, 0.0, 0.0], 2.0, 1.0, 0);
    /// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// pool.adopt(&mut fx);
    /// assert!(fx.is_empty());
    /// assert_eq!(pool.staged(), 1);
    /// ```
    pub fn adopt(&mut self, fx: &mut ParticleEffects) {
        for i in 0..fx.len() {
            let (remaining, lifetime) = fx.life_of(i);
            self.stage(
                fx.position(i),
                fx.velocity(i),
                remaining,
                lifetime,
                fx.size(i),
                fx.class_of(i),
            );
        }
        fx.clear();
        self.bound_staging();
    }

    /// More staged than the pool holds is the oldest of them replaced before they are
    /// ever drawn: drop them here, as the CPU pool's rotation would.
    fn bound_staging(&mut self) {
        let mut over = self
            .pending_count
            .saturating_sub(self.config.capacity as usize);
        while over > 0 {
            let front = self
                .pending
                .front_mut()
                .expect("pending_count counts pending");
            let (k, left) = match front {
                Pending::Records(n) => {
                    let k = (*n).min(over);
                    // The oldest records are first after the header.
                    self.upload
                        .drain(HEADER_WORDS..HEADER_WORDS + k * RECORD_WORDS);
                    *n -= k;
                    (k, *n)
                }
                Pending::Burst(payload, n) => {
                    let k = (*n).min(over);
                    payload[0] = skip_particles(payload[0], k);
                    *n -= k;
                    (k, *n)
                }
            };
            if left == 0 {
                self.pending.pop_front();
            }
            over -= k;
            self.pending_count -= k;
        }
    }

    // -- Field and ground --

    /// Write a new air field to the device: one copy of the cells, to be issued from
    /// the frame thread when the field has a new frame (10 to 20 times a second), never
    /// every frame.
    ///
    /// With [`FieldFormat::F16Filtered`] the cells go to an `f32` staging buffer and the
    /// next [`Self::encode`] converts them into the texture in one dispatch; the `f32`
    /// formats write the texture directly. A grid of new dimensions reallocates the
    /// texture (and its bind group) once.
    ///
    /// # Arguments
    ///
    /// * `air` - the air velocity, m/s; a [`SwirlField`](crate::particles::SwirlField)'s
    ///   or a plume frame's.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// use rs_physics::particles::VelocityGrid;
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// let mut air = VelocityGrid::new([0.0; 3], 2.0, [32, 32, 32]).unwrap();
    /// air.fill([1.0, 0.0, 0.0]);
    /// pool.upload_field(&air);
    /// assert!(pool.has_field());
    /// ```
    pub fn upload_field(&mut self, air: &VelocityGrid) {
        self.prepare_field(air);
        let dims = air.dims();
        let cells = bytemuck::cast_slice::<[f32; 4], u8>(air.cells());
        match &self.air.staging {
            Some((staging, _)) => {
                self.queue.write_buffer(staging, 0, cells);
                self.field_pending = true;
            }
            None => self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.air.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                cells,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(dims[2] as u32 * 16),
                    rows_per_image: Some(dims[1] as u32),
                },
                wgpu::Extent3d {
                    width: dims[2] as u32,
                    height: dims[1] as u32,
                    depth_or_array_layers: dims[0] as u32,
                },
            ),
        }
    }

    /// [`Self::upload_field`] through the host's staging ring: the pool hands `write` the
    /// field's bytes and where they go, and the host copies them there (see
    /// [`PoolWrite`]) before the next [`Self::encode`] or [`Self::encode_staged`]. The
    /// device ends up with the same contents as [`Self::upload_field`] gives it.
    ///
    /// With [`FieldFormat::F16Filtered`] the write is a [`PoolWrite::Buffer`] into the
    /// `f32` staging the conversion pass reads; with the `f32` formats it is a
    /// [`PoolWrite::Texture`], its rows padded to 256 bytes (a repack, into memory the
    /// pool keeps, only when a row of the grid is not already a multiple of 256 bytes:
    /// `nz * 16`, so 32 cells is not repacked and a plume's 34 is).
    ///
    /// # Arguments
    ///
    /// * `air` - the air velocity, m/s.
    /// * `write` - called once with the bytes and their destination.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig, PoolWrite};
    /// use rs_physics::particles::VelocityGrid;
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// let mut air = VelocityGrid::new([0.0; 3], 2.0, [32, 32, 32]).unwrap();
    /// air.fill([1.0, 0.0, 0.0]);
    /// pool.upload_field_with(&air, |write| match write {
    ///     PoolWrite::Buffer { bytes, .. } => assert_eq!(bytes.len(), 32 * 32 * 32 * 16),
    ///     PoolWrite::Texture { .. } => unreachable!("the default field is rgba16float"),
    /// });
    /// assert!(pool.has_field());
    /// ```
    pub fn upload_field_with<F: FnOnce(PoolWrite<'_>)>(&mut self, air: &VelocityGrid, write: F) {
        self.prepare_field(air);
        let dims = air.dims();
        let cells = bytemuck::cast_slice::<[f32; 4], u8>(air.cells());
        match &self.air.staging {
            Some((staging, _)) => {
                write(PoolWrite::Buffer {
                    destination: staging,
                    offset: 0,
                    bytes: cells,
                });
                self.field_pending = true;
            }
            None => {
                let row = dims[2] * 16;
                let padded = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
                let bytes: &[u8] = if padded == row {
                    cells
                } else {
                    let rows = dims[0] * dims[1];
                    self.field_scratch.clear();
                    self.field_scratch.resize(rows * padded, 0);
                    for (r, src) in cells.chunks_exact(row).enumerate() {
                        self.field_scratch[r * padded..r * padded + row].copy_from_slice(src);
                    }
                    &self.field_scratch
                };
                write(PoolWrite::Texture {
                    destination: &self.air.texture,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded as u32),
                        rows_per_image: Some(dims[1] as u32),
                    },
                    size: wgpu::Extent3d {
                        width: dims[2] as u32,
                        height: dims[1] as u32,
                        depth_or_array_layers: dims[0] as u32,
                    },
                    bytes,
                });
            }
        }
    }

    /// The field's placement into the frame header, and the texture reallocated when the
    /// grid's dimensions change: everything of an upload but the bytes.
    fn prepare_field(&mut self, air: &VelocityGrid) {
        let dims = air.dims();
        if dims != self.air.dims {
            self.air =
                Self::air_binding(&self.device, &self.convert_layout, self.config.field, dims);
            self.rebuild_air_group();
        }
        let (o, h) = (air.origin(), air.cell_size());
        let n = dims.map(|d| d as f32);
        // Texture order is z, y, x.
        let scale = [1.0 / (h * n[2]), 1.0 / (h * n[1]), 1.0 / (h * n[0])];
        let offset = [-o[2] * scale[0], -o[1] * scale[1], -o[0] * scale[2]];
        // As `VelocityGrid::new` computes them, so the exact fetch reads the same cells.
        let first = o.map(|v| v + 0.5 * h);
        self.air_words = Some([
            scale[0],
            scale[1],
            scale[2],
            0.0,
            offset[0],
            offset[1],
            offset[2],
            0.0,
            first[0],
            first[1],
            first[2],
            1.0 / h,
            n[0] - 1.0,
            n[1] - 1.0,
            n[2] - 1.0,
            0.0,
            n[0] - 2.0,
            n[1] - 2.0,
            n[2] - 2.0,
            0.0,
        ]);
    }

    /// Write a plume's newest frame to the device if it is not the one already there.
    ///
    /// The frame thread calls this every frame with the reader's
    /// [`latest`](crate::fluid_dynamics::PlumeReader::latest); it uploads only when the
    /// worker has published a new step, so the copy happens at the plume's rate. The
    /// worker thread never touches the device.
    ///
    /// # Arguments
    ///
    /// * `frame` - the plume frame.
    ///
    /// # Returns
    ///
    /// `true` when the frame was new and was uploaded.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::fluid_dynamics::{PlumeField, PlumeRegion, PlumeSource};
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// use rs_physics::particles::TurbulenceDrive;
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// let region = PlumeRegion { origin: [0.0; 3], cells: [8, 8, 8], cell_size: 1.0 };
    /// let source = PlumeSource {
    ///     position: [4.0, 0.5, 4.0],
    ///     drive: TurbulenceDrive::new(2.0, 3.0).unwrap(),
    ///     smoke_rate: 1.0,
    /// };
    /// let (mut plume, mut air) = PlumeField::new(region, source, [1.0, 0.0, 0.0], 1).unwrap();
    /// assert!(pool.upload_plume(air.latest()));
    /// assert!(!pool.upload_plume(air.latest()));
    /// plume.step_now(0.1);
    /// assert!(pool.upload_plume(air.latest()));
    /// ```
    #[cfg(feature = "fluid_simulation")]
    pub fn upload_plume(&mut self, frame: &crate::fluid_dynamics::PlumeFrame) -> bool {
        if self.last_plume_step == Some(frame.step()) && self.air_words.is_some() {
            return false;
        }
        self.last_plume_step = Some(frame.step());
        self.upload_field(frame.velocity());
        true
    }

    /// [`Self::upload_plume`] through the host's staging ring, as
    /// [`Self::upload_field_with`]: `write` is called only when the frame is new.
    ///
    /// # Arguments
    ///
    /// * `frame` - the plume frame.
    /// * `write` - called once with the bytes and their destination, when the frame is new.
    ///
    /// # Returns
    ///
    /// `true` when the frame was new and `write` was called.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::fluid_dynamics::{PlumeField, PlumeRegion, PlumeSource};
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// use rs_physics::particles::TurbulenceDrive;
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// let region = PlumeRegion { origin: [0.0; 3], cells: [8, 8, 8], cell_size: 1.0 };
    /// let source = PlumeSource {
    ///     position: [4.0, 0.5, 4.0],
    ///     drive: TurbulenceDrive::new(2.0, 3.0).unwrap(),
    ///     smoke_rate: 1.0,
    /// };
    /// let (_plume, mut air) = PlumeField::new(region, source, [1.0, 0.0, 0.0], 1).unwrap();
    /// assert!(pool.upload_plume_with(air.latest(), |write| write.write_now(&gpu.queue)));
    /// assert!(!pool.upload_plume_with(air.latest(), |_| unreachable!("not a new frame")));
    /// ```
    #[cfg(feature = "fluid_simulation")]
    pub fn upload_plume_with<F: FnOnce(PoolWrite<'_>)>(
        &mut self,
        frame: &crate::fluid_dynamics::PlumeFrame,
        write: F,
    ) -> bool {
        if self.last_plume_step == Some(frame.step()) && self.air_words.is_some() {
            return false;
        }
        self.last_plume_step = Some(frame.step());
        self.upload_field_with(frame.velocity(), write);
        true
    }

    /// Stop reading the air: every class integrates as in still air until the next
    /// [`Self::upload_field`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// pool.clear_field();
    /// assert!(!pool.has_field());
    /// ```
    pub fn clear_field(&mut self) {
        self.air_words = None;
        self.last_plume_step = None;
    }

    /// Whether a field is in use.
    ///
    /// # Returns
    ///
    /// `true` after [`Self::upload_field`] until [`Self::clear_field`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// assert!(!GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap().has_field());
    /// ```
    pub fn has_field(&self) -> bool {
        self.air_words.is_some()
    }

    /// Whether the next [`Self::encode`] converts a newly uploaded `rgba16float` field
    /// (and so writes [`PoolTimestamps::field`]).
    ///
    /// # Returns
    ///
    /// `true` between an upload to an `rgba16float` pool and the next encode.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// assert!(!GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap().field_pending());
    /// ```
    pub fn field_pending(&self) -> bool {
        self.field_pending && self.convert.is_some()
    }

    /// Bounce particles off the engine's terrain from now on, or stop with `None`.
    ///
    /// Each integrate then does what [`ParticleEffects::collide_ground_with`] does after
    /// [`ParticleEffects::integrate`]: a particle below the ground is put on it, its
    /// vertical velocity reversed and scaled by its class's restitution, its horizontal
    /// velocity scaled by 0.55, and a landing (position on the surface, impact speed,
    /// class, size) appended to [`Self::landings`]. Rebuilds one bind group; call it when
    /// the height texture changes, not every frame.
    ///
    /// # Arguments
    ///
    /// * `ground` - the height texture and its placement, or `None`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// pool.set_ground(None);
    /// ```
    pub fn set_ground(&mut self, ground: Option<GroundHeights>) {
        self.ground = ground;
        self.rebuild_air_group();
    }

    // -- A frame --

    /// Record this frame's placement and integrate into `encoder`, after one write of
    /// this frame's constants and staged particles.
    ///
    /// Submit the encoder once per call: the write lands at the next submission, and a
    /// second call before it would replace this one's data. A `dt` of zero or less (or
    /// not finite) places the staged particles and integrates nothing.
    ///
    /// # Arguments
    ///
    /// * `encoder` - the frame's encoder; the pool records two compute passes (three on
    ///   a frame that converts a new `rgba16float` field).
    /// * `dt` - the step, seconds.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// pool.emit_one([0.0, 2.0, 0.0], [3.0, 0.0, 0.0], 1.0, 1.0, 0);
    /// let mut encoder = gpu.device.create_command_encoder(&Default::default());
    /// pool.encode(&mut encoder, 1.0 / 60.0);
    /// gpu.queue.submit([encoder.finish()]);
    /// ```
    pub fn encode(&mut self, encoder: &mut wgpu::CommandEncoder, dt: f32) {
        self.encode_timed(encoder, dt, None);
    }

    /// [`Self::encode`], writing a timestamp at the start and end of each pass named in
    /// `timestamps`.
    ///
    /// # Arguments
    ///
    /// * `encoder` - the frame's encoder.
    /// * `dt` - the step, seconds.
    /// * `timestamps` - where each pass writes its timestamps, or `None`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig, PoolTimestamps};
    /// let gpu = GpuContext::with_features(wgpu::Features::TIMESTAMP_QUERY).expect("a GPU");
    /// let queries = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
    ///     label: None,
    ///     ty: wgpu::QueryType::Timestamp,
    ///     count: 4,
    /// });
    /// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// let mut encoder = gpu.device.create_command_encoder(&Default::default());
    /// pool.encode_timed(&mut encoder, 1.0 / 60.0, Some(PoolTimestamps {
    ///     query_set: &queries,
    ///     field: None,
    ///     emit: Some(0),
    ///     integrate: Some(2),
    /// }));
    /// gpu.queue.submit([encoder.finish()]);
    /// ```
    pub fn encode_timed(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
        timestamps: Option<PoolTimestamps<'_>>,
    ) {
        let queue = self.queue.clone();
        self.stage_frame_with(dt, |write| write.write_now(&queue));
        self.encode_staged(encoder, timestamps);
    }

    /// The first half of [`Self::encode`] for a host with a staging ring of its own: this
    /// frame's constants and staged particles, as one [`PoolWrite::Buffer`] handed to
    /// `write`, for the host to copy into the pool's frame buffer ahead of
    /// [`Self::encode_staged`] (see [`PoolWrite`]). Together the two leave the device
    /// exactly as [`Self::encode`] does, without the `write_buffer` and the staging buffer
    /// wgpu allocates for it.
    ///
    /// Call it once per [`Self::encode_staged`]; a second call before it replaces the
    /// first frame's constants, and its particles are placed with the second's.
    ///
    /// # Arguments
    ///
    /// * `dt` - the step, seconds; as [`Self::encode`].
    /// * `write` - called once with the bytes (512 bytes of constants, then 16 bytes a
    ///   segment of the frame's emission, plus 48 for a burst's descriptor or 40 a record)
    ///   and the frame buffer they go to.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig, PoolWrite};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// pool.emit_one([0.0, 2.0, 0.0], [3.0, 0.0, 0.0], 1.0, 1.0, 0);
    /// let mut encoder = gpu.device.create_command_encoder(&Default::default());
    /// pool.stage_frame_with(1.0 / 60.0, |write| {
    ///     let PoolWrite::Buffer { destination, offset, bytes } = write else { unreachable!() };
    ///     assert_eq!(bytes.len(), 512 + 16 + 40);
    ///     // A host's ring: the bytes into mapped staging memory, the copy onto the encoder.
    ///     let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
    ///         label: None,
    ///         size: bytes.len() as u64,
    ///         usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
    ///         mapped_at_creation: true,
    ///     });
    ///     staging.slice(..).get_mapped_range_mut().unwrap().copy_from_slice(bytes);
    ///     staging.unmap();
    ///     encoder.copy_buffer_to_buffer(&staging, 0, destination, offset, bytes.len() as u64);
    /// });
    /// pool.encode_staged(&mut encoder, None);
    /// gpu.queue.submit([encoder.finish()]);
    /// ```
    pub fn stage_frame_with<F: FnOnce(PoolWrite<'_>)>(&mut self, dt: f32, write: F) {
        let (emit, segments, table_at) = self.take_frame_emission();
        self.write_header(dt, emit as u32, segments as u32, table_at as u32);
        write(PoolWrite::Buffer {
            destination: &self.frame,
            offset: 0,
            bytes: bytes_of(&self.upload),
        });
        // The records left staged go back after the header.
        self.upload.truncate(HEADER_WORDS);
        self.upload.extend_from_slice(&self.carry);
        self.prepared = Some((dt, emit));
    }

    /// The oldest staged emission, up to [`GpuPoolConfig::max_emit_per_frame`]
    /// particles, laid out after the header in `upload` as the device reads it: the
    /// records this frame takes (already there, the oldest staged), then each burst's
    /// 12-word descriptor whatever its count, then the segment table. The records left
    /// staged are moved to `carry` meanwhile (none, unless the frame's bound cut them).
    /// A burst cut by the frame's bound leaves its remainder staged, its state advanced
    /// past the particles taken.
    ///
    /// Returns the particles and the segments taken, and the word of the emission data
    /// where the segment table starts.
    fn take_frame_emission(&mut self) -> (usize, usize, usize) {
        let want = self
            .pending_count
            .min(self.config.max_emit_per_frame as usize);
        self.payload.clear();
        self.table.clear();
        self.carry.clear();
        let (mut taken, mut records) = (0usize, 0usize);
        while taken < want {
            let used =
                records * RECORD_WORDS + self.payload.len() + self.table.len() + SEGMENT_WORDS;
            let Some(front) = self.pending.front_mut() else {
                break;
            };
            let (k, left) = match front {
                Pending::Records(n) => {
                    let room = self.data_words.saturating_sub(used) / RECORD_WORDS;
                    let k = (*n).min(want - taken).min(room);
                    if k == 0 {
                        break;
                    }
                    let at = records * RECORD_WORDS;
                    self.table
                        .extend([taken as u32, k as u32, SEGMENT_RECORDS, at as u32]);
                    records += k;
                    *n -= k;
                    (k, *n)
                }
                Pending::Burst(payload, n) => {
                    if used + BURST_WORDS > self.data_words {
                        break;
                    }
                    let k = (*n).min(want - taken);
                    // Offset among the bursts for now; the records' words go before them.
                    let at = self.payload.len();
                    self.table
                        .extend([taken as u32, k as u32, SEGMENT_BURST, at as u32]);
                    self.payload.extend_from_slice(payload);
                    *n -= k;
                    if *n > 0 {
                        payload[0] = skip_particles(payload[0], k);
                    }
                    (k, *n)
                }
            };
            if left == 0 {
                self.pending.pop_front();
            }
            taken += k;
            self.pending_count -= k;
        }
        let record_words = records * RECORD_WORDS;
        for entry in self.table.chunks_exact_mut(SEGMENT_WORDS) {
            if entry[2] == SEGMENT_BURST {
                entry[3] += record_words as u32;
            }
        }
        let end = HEADER_WORDS + record_words;
        self.carry.extend_from_slice(&self.upload[end..]);
        self.upload.truncate(end);
        self.upload.extend_from_slice(&self.payload);
        let table_at = record_words + self.payload.len();
        self.upload.extend_from_slice(&self.table);
        (taken, self.table.len() / SEGMENT_WORDS, table_at)
    }

    /// The second half of [`Self::encode`]: the passes of the frame
    /// [`Self::stage_frame_with`] staged, recorded into `encoder`, with timestamps as
    /// [`Self::encode_timed`]. The staged bytes must reach the frame buffer before these
    /// passes run (see [`PoolWrite`]).
    ///
    /// # Arguments
    ///
    /// * `encoder` - the frame's encoder.
    /// * `timestamps` - where each pass writes its timestamps, or `None`.
    ///
    /// # Panics
    ///
    /// If no frame was staged since the last encode.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// pool.stage_frame_with(1.0 / 60.0, |write| write.write_now(&gpu.queue));
    /// let mut encoder = gpu.device.create_command_encoder(&Default::default());
    /// pool.encode_staged(&mut encoder, None);
    /// gpu.queue.submit([encoder.finish()]);
    /// ```
    pub fn encode_staged(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        timestamps: Option<PoolTimestamps<'_>>,
    ) {
        let (dt, emit) = self
            .prepared
            .take()
            .expect("encode_staged needs a frame from stage_frame_with first");

        let writes = |pair: Option<u32>| {
            timestamps.and_then(|t| {
                pair.map(|i| wgpu::ComputePassTimestampWrites {
                    query_set: t.query_set,
                    beginning_of_pass_write_index: Some(i),
                    end_of_pass_write_index: Some(i + 1),
                })
            })
        };

        self.encode_field(encoder, writes(timestamps.and_then(|t| t.field)));

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pool emit"),
                timestamp_writes: writes(timestamps.and_then(|t| t.emit)),
            });
            pass.set_bind_group(0, &self.main_group, &[]);
            pass.set_bind_group(1, &self.air_group, &[]);
            pass.set_bind_group(2, &self.args_group, &[]);
            if emit > 0 {
                pass.set_pipeline(&self.place_free);
                pass.dispatch_workgroups((emit as u32).div_ceil(WG), 1, 1);
                pass.set_pipeline(&self.place_overwrite);
                pass.dispatch_workgroups_indirect(&self.args, 12);
            }
            pass.set_pipeline(&self.finalize);
            pass.dispatch_workgroups(1, 1, 1);
        }

        if dt.is_finite() && dt > 0.0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pool integrate"),
                timestamp_writes: writes(timestamps.and_then(|t| t.integrate)),
            });
            pass.set_pipeline(&self.integrate);
            pass.set_bind_group(0, &self.main_group, &[]);
            pass.set_bind_group(1, &self.air_group, &[]);
            pass.dispatch_workgroups_indirect(&self.args, 0);
        }
    }

    /// The pass converting a newly uploaded `rgba16float` field, when one is pending.
    fn encode_field(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        timestamps: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        if !self.field_pending {
            return;
        }
        self.field_pending = false;
        if let (Some(convert), Some((_, group))) = (&self.convert, &self.air.staging) {
            let cells = self.air.dims.iter().product::<usize>() as u32;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pool field"),
                timestamp_writes: timestamps,
            });
            pass.set_pipeline(convert);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(cells.div_ceil(64), 1, 1);
        }
    }

    /// [`Self::encode`] into an encoder of its own, submitted at once. For tests, tools
    /// and hosts with no frame encoder of their own; it does not wait for the device.
    ///
    /// # Arguments
    ///
    /// * `dt` - the step, seconds.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// pool.emit_one([0.0, 2.0, 0.0], [3.0, 0.0, 0.0], 1.0, 1.0, 0);
    /// pool.step(1.0 / 60.0);
    /// assert_eq!(pool.read_counts_blocking().live, 1);
    /// ```
    pub fn step(&mut self, dt: f32) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pool step"),
            });
        self.encode(&mut encoder, dt);
        self.queue.submit([encoder.finish()]);
    }

    fn write_header(&mut self, dt: f32, emit: u32, segments: u32, table_at: u32) {
        let h = &mut self.upload[..FRAME_WORDS];
        h.fill(0);
        let f = |v: f32| v.to_bits();
        h[0] = f(dt);
        h[1] = emit;
        h[2] = self.config.capacity;
        let mut flags = 0;
        if self.air_words.is_some() {
            flags |= FLAG_AIR;
        }
        if self.ground.is_some() {
            flags |= FLAG_GROUND;
        }
        h[3] = flags;
        h[4] = self.config.landing_capacity;
        h[5] = self.config.sprite_vertices;
        if let Some(words) = &self.air_words {
            for (slot, w) in h[8..28].iter_mut().zip(words) {
                *slot = f(*w);
            }
        }
        if let Some(g) = &self.ground {
            h[28] = f(g.min[0]);
            h[29] = f(g.min[1]);
            h[30] = f(g.cell);
            h[32] = g.corners[0].saturating_sub(1);
            h[33] = g.corners[1].saturating_sub(1);
        }
        // The CPU path's per-class rows, computed the same way so a class integrates
        // with the same constants on either.
        for c in 0..MAX_CLASSES {
            let class = self.classes[c];
            let k = (class.drag * dt).min(1.0);
            let row = [
                class.gravity * dt,
                1.0 - k,
                if class.gravity == 0.0 { 1.0 } else { 0.0 },
                self.swirl[c] * k.max(0.0),
            ];
            for (slot, w) in h[36 + 4 * c..40 + 4 * c].iter_mut().zip(row) {
                *slot = f(w);
            }
            h[68 + c] = f(class.restitution);
        }
        h[76] = segments;
        h[77] = self.data_words as u32;
        h[78] = self.jump_levels;
        // h[79] stays 0: the shader's opaque zero.
        h[80] = table_at;
    }

    // -- What the renderer reads --

    /// The position-and-life buffer, one `vec4<f32>` a slot: position in metres in
    /// `xyz`, seconds of life remaining in `w`. A slot is live while `w > 0`. Draw over
    /// the high water (see [`Self::draw_args`]) and discard slots with `w <= 0`.
    ///
    /// Read-only to anyone but the pool: bind it as read-only storage.
    ///
    /// # Returns
    ///
    /// The buffer, [`GpuPoolConfig::capacity`] slots (rounded up to 64).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// assert_eq!(pool.positions().size(), 64 * 16);
    /// ```
    pub fn positions(&self) -> &wgpu::Buffer {
        &self.pos_life
    }

    /// The velocity-and-lifetime buffer, one `vec4<f32>` a slot: velocity in m/s in
    /// `xyz`, total lifetime in seconds in `w`. The remaining fraction a renderer fades
    /// on is `positions[i].w / velocities[i].w`. Read-only to anyone but the pool.
    ///
    /// # Returns
    ///
    /// The buffer.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// assert_eq!(pool.velocities().size(), 64 * 16);
    /// ```
    pub fn velocities(&self) -> &wgpu::Buffer {
        &self.vel
    }

    /// The class-and-size buffer, one `vec2<u32>` a slot: the class slot in the low
    /// three bits of `x` (bit 31 is set on the frame a particle is placed, until its
    /// first integrate; mask it off), the renderer's size scalar as `f32` bits in `y`
    /// (`bitcast<f32>`). Read-only to anyone but the pool.
    ///
    /// # Returns
    ///
    /// The buffer.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// assert_eq!(pool.classes_and_sizes().size(), 64 * 8);
    /// ```
    pub fn classes_and_sizes(&self) -> &wgpu::Buffer {
        &self.meta
    }

    /// The counters, nine `u32`s: free slots, live particles, high water, overwrite
    /// cursor, landings this frame, then totals since the pool was built: retired,
    /// placed, overwritten and dropped (see [`GpuPoolCounts`]). The `STATE_*` constants are their byte offsets. Read-only to
    /// anyone but the pool.
    ///
    /// # Returns
    ///
    /// The buffer.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// assert_eq!(pool.state().size(), 36);
    /// ```
    pub fn state(&self) -> &wgpu::Buffer {
        &self.state
    }

    /// Byte offset in [`Self::state`] of the live count.
    pub const STATE_LIVE: u64 = 4;
    /// Byte offset in [`Self::state`] of the high water: one past the highest slot in use.
    pub const STATE_HIGH_WATER: u64 = 8;
    /// Byte offset in [`Self::state`] of this frame's landing count. It can exceed
    /// [`GpuPoolConfig::landing_capacity`]; only that many were written.
    pub const STATE_LANDINGS: u64 = 16;

    /// The landings of the last integrate, six 32-bit words each: position on the
    /// surface in metres (3 `f32`), impact speed in m/s (`f32`), class (`u32`), size
    /// (`f32` bits). The count is at [`Self::STATE_LANDINGS`], at most
    /// [`GpuPoolConfig::landing_capacity`] of them written. Valid from the integrate
    /// until the next [`Self::encode`]'s placement pass resets the count: drain it once
    /// a frame, between the two.
    ///
    /// # Returns
    ///
    /// The buffer.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// assert_eq!(pool.landings().size(), 4_096 * 24);
    /// ```
    pub fn landings(&self) -> &wgpu::Buffer {
        &self.landings
    }

    /// The indirect-arguments buffer. At [`Self::DRAW_ARGS_OFFSET`]: a non-indexed draw's
    /// four words, `[sprite_vertices, high_water, 0, 0]`, for `draw_indirect` of one
    /// instance a slot (discard the dead in the vertex stage). Written by every
    /// [`Self::encode`] before its integrate. Read it as `INDIRECT` only.
    ///
    /// # Returns
    ///
    /// The buffer.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// assert!(pool.draw_args().usage().contains(wgpu::BufferUsages::INDIRECT));
    /// ```
    pub fn draw_args(&self) -> &wgpu::Buffer {
        &self.args
    }

    /// Byte offset of the draw arguments in [`Self::draw_args`].
    pub const DRAW_ARGS_OFFSET: u64 = 32;

    // -- Debug and test readback: never on the frame thread --

    fn read_blocking(&self, source: &wgpu::Buffer, bytes: u64) -> Vec<u32> {
        if bytes == 0 {
            return Vec::new();
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pool readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
        self.queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device lost while reading the pool back");
        let words =
            bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range().expect("mapped readback"))
                .to_vec();
        staging.unmap();
        words
    }

    /// The counters, read back. **A debug and test path:** it submits a copy and waits
    /// for the device, which costs a frame; never call it on the frame thread.
    ///
    /// # Returns
    ///
    /// The [`GpuPoolCounts`] after every submitted frame.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// assert_eq!(pool.read_counts_blocking().free, 64);
    /// ```
    pub fn read_counts_blocking(&self) -> GpuPoolCounts {
        let w = self.read_blocking(&self.state, (STATE_WORDS * 4) as u64);
        GpuPoolCounts {
            free: w[0],
            live: w[1],
            high_water: w[2],
            cursor: w[3],
            landings: w[4],
            retired: w[5],
            placed: w[6],
            overwritten: w[7],
            dropped: w[8],
        }
    }

    /// Every slot below the high water, read back. A debug and test path; see
    /// [`Self::read_counts_blocking`].
    ///
    /// # Returns
    ///
    /// One [`GpuSlot`] per slot, live or not, in slot order.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// pool.emit_one([0.0, 2.0, 0.0], [3.0, 0.0, 0.0], 1.0, 1.0, 0);
    /// pool.step(0.0);
    /// assert_eq!(pool.read_slots_blocking()[0].position, [0.0, 2.0, 0.0]);
    /// ```
    pub fn read_slots_blocking(&self) -> Vec<GpuSlot> {
        let n = self.read_counts_blocking().high_water as u64;
        let a = self.read_blocking(&self.pos_life, n * 16);
        let v = self.read_blocking(&self.vel, n * 16);
        let m = self.read_blocking(&self.meta, n * 8);
        let f = f32::from_bits;
        (0..n as usize)
            .map(|i| GpuSlot {
                position: [f(a[4 * i]), f(a[4 * i + 1]), f(a[4 * i + 2])],
                remaining: f(a[4 * i + 3]),
                velocity: [f(v[4 * i]), f(v[4 * i + 1]), f(v[4 * i + 2])],
                lifetime: f(v[4 * i + 3]),
                class: m[2 * i] as u8,
                size: f(m[2 * i + 1]),
            })
            .collect()
    }

    /// The free stack, bottom first, read back. A debug and test path; see
    /// [`Self::read_counts_blocking`].
    ///
    /// # Returns
    ///
    /// The slots on the stack; the last is the next one taken.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// assert_eq!(pool.read_free_blocking().last(), Some(&0));
    /// ```
    pub fn read_free_blocking(&self) -> Vec<u32> {
        let n = self.read_counts_blocking().free as u64;
        self.read_blocking(&self.free, n * 4)
    }

    /// The last integrate's landings, read back. A debug and test path; see
    /// [`Self::read_counts_blocking`].
    ///
    /// # Returns
    ///
    /// The landings written (at most [`GpuPoolConfig::landing_capacity`]), in the order
    /// the device appended them.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// assert!(pool.read_landings_blocking().is_empty());
    /// ```
    pub fn read_landings_blocking(&self) -> Vec<Landing> {
        let n = self
            .read_counts_blocking()
            .landings
            .min(self.config.landing_capacity) as usize;
        let w = self.read_blocking(&self.landings, (n * LANDING_WORDS * 4) as u64);
        let f = f32::from_bits;
        (0..n)
            .map(|i| {
                let b = i * LANDING_WORDS;
                Landing {
                    position: [f(w[b]), f(w[b + 1]), f(w[b + 2])],
                    impact_speed: f(w[b + 3]),
                    class: w[b + 4] as u8,
                    size: f(w[b + 5]),
                }
            })
            .collect()
    }

    /// The air at each point, through the integrate's own fetch, read back. A debug and
    /// test path (it builds a pipeline and waits for the device); see
    /// [`Self::read_counts_blocking`].
    ///
    /// # Arguments
    ///
    /// * `points` - world positions, metres.
    ///
    /// # Returns
    ///
    /// The air velocity at each, m/s; zero everywhere when no field is in use.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
    /// use rs_physics::particles::VelocityGrid;
    /// let gpu = GpuContext::new().expect("a GPU");
    /// let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(64)).unwrap();
    /// let mut air = VelocityGrid::new([0.0; 3], 1.0, [4, 4, 4]).unwrap();
    /// air.fill([2.0, 0.0, 0.0]);
    /// pool.upload_field(&air);
    /// assert_eq!(pool.probe_air_blocking(&[[1.0, 1.0, 1.0]]), vec![[2.0, 0.0, 0.0]]);
    /// ```
    pub fn probe_air_blocking(&mut self, points: &[[f32; 3]]) -> Vec<[f32; 3]> {
        if points.is_empty() || self.air_words.is_none() {
            return vec![[0.0; 3]; points.len()];
        }
        // The field's words into the frame header; nothing placed, nothing stepped. The
        // next `encode` writes the header again.
        self.write_header(0.0, 0, 0, 0);
        self.queue
            .write_buffer(&self.frame, 0, bytes_of(&self.upload[..FRAME_WORDS]));
        let device = self.device.clone();

        let source = format!("{}\n{}", self.module_source, air_section("probe"));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pool probe"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pool probe"),
            entries: &[
                uniform_entry(0),
                storage_entry(8, true),
                storage_entry(9, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pool probe"),
            bind_group_layouts: &[Some(&layout), Some(&self.air_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("probe"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        let input: Vec<f32> = points
            .iter()
            .flat_map(|p| [p[0], p[1], p[2], 0.0])
            .collect();
        let bytes = (points.len() * 16) as u64;
        let probe_in = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe in"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&probe_in, 0, bytemuck::cast_slice(&input));
        let probe_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe out"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool probe"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.frame,
                        offset: 0,
                        size: wgpu::BufferSize::new((HEADER_WORDS * 4) as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: probe_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: probe_out.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        self.encode_field(&mut encoder, None);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.set_bind_group(1, &self.air_group, &[]);
            pass.dispatch_workgroups((points.len() as u32).div_ceil(64), 1, 1);
        }
        self.queue.submit([encoder.finish()]);
        let w = self.read_blocking(&probe_out, bytes);
        (0..points.len())
            .map(|i| {
                [
                    f32::from_bits(w[4 * i]),
                    f32::from_bits(w[4 * i + 1]),
                    f32::from_bits(w[4 * i + 2]),
                ]
            })
            .collect()
    }
}
