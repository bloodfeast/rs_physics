//! Emission expanded on the device against the CPU's, to the bit.
//!
//! A burst staged on [`GpuParticlePool::emit`] is a descriptor; the placement passes
//! expand it with xorshift32 jump-ahead and the CPU's arithmetic (integer sine and
//! cosine, correctly rounded square root and divide, every product rounded before an
//! add). These tests hold it to the CPU pool's particles bit for bit, the jump-ahead to
//! sequential steps, and the shader's arithmetic to the CPU's over sweeps.
//!
//! Every device test skips with a message when no adapter exists (CI has none).

use crate::gpu::{GpuContext, GpuParticlePool, GpuPoolConfig};
use crate::particles::{
    sin_cos_turn, stride_tables, Burst, EffectRng, ParticleClass, ParticleEffects, DIGIT_TABLES,
    QUARTER_SINE,
};

/// A power of two, so the integrate's `v dt` is exact and a fused `p + v dt` on the
/// device is the CPU's unfused one: the integrate is then bit-identical too, and the
/// born particles can be followed to retirement.
const DT: f32 = 1.0 / 64.0;

fn gpu() -> Option<GpuContext> {
    let gpu = GpuContext::with_features(wgpu::Features::empty());
    if gpu.is_none() {
        println!("No GPU adapter: skipping the device emission test");
    }
    gpu
}

/// Every class a different gravity and drag; none on the air, no ground.
fn classes() -> [ParticleClass; 8] {
    core::array::from_fn(|c| ParticleClass {
        gravity: [9.80665, 26.0, 1.6, 0.0, 3.0, 0.0, 60.0, 12.5][c],
        drag: [0.0, 1.4, 3.4, 0.7, 0.0, 0.0, 9.0, 0.25][c],
        restitution: 0.0,
    })
}

/// Bursts covering every option the draws read: each class (and one out of range,
/// clamped), lift from below zero to well past one, equal, reversed and negative
/// ranges (a negative lifetime is raised to `f32::EPSILON`), zero speed, and counts of
/// 0, 1 (the record path), 2 and up.
fn bursts(frame: usize) -> Vec<Burst> {
    let f = frame as f32;
    let mut out = Vec::new();
    let lifts = [0.0f32, 0.35, 1.0, 3.0, -0.5, 0.999_999];
    for class in 0..9u8 {
        let k = class as usize + frame;
        out.push(Burst {
            origin: [f * 0.25 - 3.0, 20.0 + class as f32, -1.5 * class as f32],
            class: if class == 8 { 200 } else { class },
            count: [0u32, 1, 2, 3, 17, 64, 131, 250, 9][k % 9],
            speed: match k % 4 {
                0 => 6.0..15.0,
                1 => 4.0..4.0,
                2 => 9.0..2.5,
                _ => 0.0..0.0,
            },
            lifetime: match k % 5 {
                0 => 0.05..0.6,
                1 => 0.3..0.3,
                2 => -1.0..-0.5,
                3 => 1.2..0.1,
                _ => 0.2..1.5,
            },
            size: match k % 3 {
                0 => 0.7..1.3,
                1 => 2.0..2.0,
                _ => -0.5..0.25,
            },
            lift: lifts[k % lifts.len()],
        });
    }
    out
}

/// A particle as bits: position, velocity, remaining, lifetime, size, class. The whole
/// record is its identity, so the CPU's and the device's particles are matched as
/// sorted lists of these.
type Bits = [u32; 10];

fn cpu_bits(fx: &ParticleEffects) -> Vec<Bits> {
    let mut out: Vec<Bits> = (0..fx.len())
        .map(|i| {
            let (p, v) = (fx.position(i), fx.velocity(i));
            let (remaining, lifetime) = fx.life_of(i);
            [
                p[0].to_bits(),
                p[1].to_bits(),
                p[2].to_bits(),
                v[0].to_bits(),
                v[1].to_bits(),
                v[2].to_bits(),
                remaining.to_bits(),
                lifetime.to_bits(),
                fx.size(i).to_bits(),
                fx.class_of(i) as u32,
            ]
        })
        .collect();
    out.sort_unstable();
    out
}

fn gpu_bits(pool: &GpuParticlePool) -> Vec<Bits> {
    let mut out: Vec<Bits> = pool
        .read_slots_blocking()
        .into_iter()
        .filter(|s| s.remaining > 0.0)
        .map(|s| {
            [
                s.position[0].to_bits(),
                s.position[1].to_bits(),
                s.position[2].to_bits(),
                s.velocity[0].to_bits(),
                s.velocity[1].to_bits(),
                s.velocity[2].to_bits(),
                s.remaining.to_bits(),
                s.lifetime.to_bits(),
                s.size.to_bits(),
                s.class as u32,
            ]
        })
        .collect();
    out.sort_unstable();
    out
}

/// The first difference between two sorted particle lists, for a failure message.
fn first_difference(cpu: &[Bits], device: &[Bits]) -> String {
    let f = |b: &Bits| -> Vec<f32> { b[..9].iter().map(|&w| f32::from_bits(w)).collect() };
    for (a, b) in cpu.iter().zip(device) {
        if a != b {
            return format!(
                "cpu {:?} class {}\ndev {:?} class {}",
                f(a),
                a[9],
                f(b),
                b[9]
            );
        }
    }
    format!("lengths {} and {}", cpu.len(), device.len())
}

/// (a) and (b): for every class and burst option, a CPU pool and the device pool fed the
/// same bursts from the same seed hold bit-identical particles (position, velocity,
/// remaining and total lifetime, size, class) after every one of 60 frames, through
/// retirement; and the emitting stream ends each frame where the CPU's emit left it.
#[test]
fn bursts_expanded_on_the_device_are_the_cpus_to_the_bit() {
    let Some(gpu) = gpu() else { return };
    let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(1 << 16)).unwrap();
    let mut fx = ParticleEffects::with_capacity(1 << 16);
    for (c, class) in classes().into_iter().enumerate() {
        fx.set_class(c as u8, class);
        pool.set_class(c as u8, class);
    }
    let (mut cpu_rng, mut gpu_rng) = (EffectRng::new(0xC0FFEE), EffectRng::new(0xC0FFEE));
    let (mut born, mut most_live, mut retired_by_now) = (0usize, 0usize, 0usize);
    for frame in 0..60 {
        for burst in bursts(frame) {
            fx.emit(&burst, &mut cpu_rng);
            pool.emit(&burst, &mut gpu_rng);
            born += burst.count as usize;
        }
        assert_eq!(
            cpu_rng.clone().next_u32(),
            gpu_rng.clone().next_u32(),
            "frame {frame}: the streams parted"
        );
        fx.integrate(DT);
        pool.step(DT);
        let (cpu, device) = (cpu_bits(&fx), gpu_bits(&pool));
        assert!(
            cpu == device,
            "frame {frame}: {} CPU particles, {} device; first difference:\n{}",
            cpu.len(),
            device.len(),
            first_difference(&cpu, &device)
        );
        most_live = most_live.max(cpu.len());
        retired_by_now = born - cpu.len();
    }
    let counts = pool.read_counts_blocking();
    assert_eq!(counts.placed as usize, born);
    assert_eq!(counts.retired as usize, retired_by_now);
    assert!(
        retired_by_now > born / 2,
        "the run should retire most of what it emits"
    );
    println!(
        "{born} particles born over 60 frames, {retired_by_now} retired, up to {most_live} live: every one bit-identical every frame"
    );
}

/// The frame bound cuts a burst and a full pool drops the oldest staged particles: a
/// descriptor's remainder carries on from the right draw either way, and records staged
/// among the bursts keep their order. Checked against the same particles staged as
/// records only (the bursts drawn by the CPU pool, in order).
#[test]
fn a_burst_cut_by_the_frame_bound_or_the_capacity_resumes_on_the_right_draw() {
    let Some(gpu) = gpu() else { return };
    let mut config = GpuPoolConfig::new(1_500);
    config.max_emit_per_frame = 211;
    let mut by_descriptor = GpuParticlePool::new(&gpu, config).unwrap();
    let mut by_record = GpuParticlePool::new(&gpu, config).unwrap();
    let (mut a, mut b) = (EffectRng::new(77), EffectRng::new(77));
    for frame in 0..12 {
        // 2,000 staged at once on the first frame: past the capacity, so the oldest 500
        // are dropped before they are placed, then 211 a frame.
        let count = if frame == 0 {
            2_000
        } else {
            37 + 50 * (frame % 3) as u32
        };
        let burst = Burst {
            origin: [1.0, 2.0, 3.0],
            class: (frame % 8) as u8,
            count,
            speed: 3.0..9.0,
            lifetime: 30.0..40.0,
            size: 0.5..1.5,
            lift: 0.4,
        };
        by_descriptor.emit(&burst, &mut a);
        let mut drawn = ParticleEffects::with_capacity(count as usize);
        drawn.emit(&burst, &mut b);
        for i in 0..drawn.len() {
            let (life, _) = drawn.life_of(i);
            by_record.emit_one(
                drawn.position(i),
                drawn.velocity(i),
                life,
                drawn.size(i),
                drawn.class_of(i),
            );
        }
        // Explicit particles between the bursts, so frames mix record runs and
        // descriptors, and the records cut by the bound carry over among them.
        for k in 0..7 {
            let v = [k as f32, frame as f32, 1.0];
            by_descriptor.emit_one([0.5; 3], v, 25.0, 3.0, 1);
            by_record.emit_one([0.5; 3], v, 25.0, 3.0, 1);
        }
        assert_eq!(by_descriptor.staged(), by_record.staged(), "frame {frame}");
        by_descriptor.step(DT);
        by_record.step(DT);
        let (x, y) = (gpu_bits(&by_descriptor), gpu_bits(&by_record));
        assert!(x == y, "frame {frame}: {}", first_difference(&x, &y));
    }
    assert_eq!(a.next_u32(), b.next_u32());
}

/// (d) is `a_hosts_staging_ring_gives_the_same_device_contents_as_the_queue` in
/// `particle_pool_tests.rs`, whose emission now includes bursts. This one checks the
/// frame's bytes: a burst of any size is one 16-word segment.
#[test]
fn a_burst_uploads_a_descriptor_not_its_particles() {
    let Some(gpu) = gpu() else { return };
    let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(1 << 16)).unwrap();
    let mut rng = EffectRng::new(5);
    let burst = |count| Burst {
        origin: [0.0; 3],
        class: 0,
        count,
        speed: 1.0..2.0,
        lifetime: 1.0..2.0,
        size: 1.0..1.0,
        lift: 0.0,
    };
    pool.emit(&burst(50_000), &mut rng);
    pool.emit(&burst(3), &mut rng);
    pool.emit_one([0.0; 3], [0.0; 3], 1.0, 1.0, 0);
    let mut bytes = 0;
    pool.stage_frame_with(DT, |write| {
        let crate::gpu::PoolWrite::Buffer { bytes: b, .. } = write else {
            unreachable!()
        };
        bytes = b.len();
        write.write_now(&gpu.queue);
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    pool.encode_staged(&mut encoder, None);
    gpu.queue.submit([encoder.finish()]);
    // The header, three segment entries, two burst payloads and one record.
    assert_eq!(bytes, 512 + 4 * (3 * 4 + 2 * 12 + 10));
    assert_eq!(pool.read_counts_blocking().live, 50_004);
}

/// Every burst option takes `Burst::DRAWS_PER_PARTICLE` draws a particle: the stride
/// the device jumps by.
#[test]
fn every_burst_option_takes_five_draws_a_particle() {
    for frame in 0..9 {
        for burst in bursts(frame) {
            let (mut a, mut b) = (EffectRng::new(91), EffectRng::new(91));
            ParticleEffects::with_capacity(512).emit(&burst, &mut a);
            b.jump(Burst::DRAWS_PER_PARTICLE as u64 * burst.count as u64);
            assert_eq!(a.next_u32(), b.next_u32(), "{burst:?}");
        }
    }
}

// -- The shader's arithmetic, swept against the CPU's --

const PROBE: &str = r#"
struct Probe {
    zero: u32,
    count: u32,
    levels: u32,
    tables: u32,
    sine: u32,
    inputs: u32,
    pad0: u32,
    pad1: u32,
}
@group(0) @binding(0) var<uniform> probe: Probe;
@group(0) @binding(1) var<storage, read> records: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

fn emit_zero() -> u32 {
    return probe.zero;
}

@compute @workgroup_size(256)
fn probe_sin_cos(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= probe.count) {
        return;
    }
    let sc = sin_cos_turn(probe.inputs + k, probe.sine);
    out[2u * k] = bitcast<u32>(sc.x);
    out[2u * k + 1u] = bitcast<u32>(sc.y);
}

@compute @workgroup_size(256)
fn probe_jump(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= probe.count) {
        return;
    }
    let b = probe.inputs + 2u * k;
    out[k] = jump_stride(records[b], records[b + 1u], probe.tables, probe.levels);
}

@compute @workgroup_size(256)
fn probe_sqrt_div(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= probe.count) {
        return;
    }
    let b = probe.inputs + 2u * k;
    let x = bitcast<f32>(records[b]);
    let y = bitcast<f32>(records[b + 1u]);
    out[4u * k] = bitcast<u32>(sqrt_rn(abs(x)));
    out[4u * k + 1u] = bitcast<u32>(div_rn(x, abs(y)));
    out[4u * k + 2u] = bitcast<u32>(sqrt_rn_by(abs(x), true));
    out[4u * k + 3u] = bitcast<u32>(div_rn_by(x, abs(y), true));
}
"#;

/// A pipeline over the emission WGSL with the probe entry points, the tables at word 0
/// of `records` and `inputs` after them.
struct Prober {
    gpu: GpuContext,
    module: wgpu::ShaderModule,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    levels: u32,
    tables: Vec<u32>,
}

impl Prober {
    fn new(gpu: GpuContext, levels: u32) -> Prober {
        let source = format!("{}\n{}", include_str!("particle_pool_emit.wgsl"), PROBE);
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("emission probe"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    entry(0, wgpu::BufferBindingType::Uniform),
                    entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let mut tables = stride_tables(Burst::DRAWS_PER_PARTICLE, levels as usize);
        tables.extend(QUARTER_SINE.iter().map(|&v| v as u32));
        Prober {
            gpu,
            module,
            layout,
            pipeline_layout,
            levels,
            tables,
        }
    }

    /// Runs `entry` over `count` threads with `inputs` (or, for the sine sweep, the
    /// first turn in `first`), returning `out_words` words.
    fn run(
        &self,
        entry: &str,
        count: u32,
        first: u32,
        inputs: &[u32],
        out_words: usize,
    ) -> Vec<u32> {
        use wgpu::util::DeviceExt;
        let device = &self.gpu.device;
        let jump_words = self.levels * DIGIT_TABLES as u32 * 1024;
        let header = [
            0u32,
            count,
            self.levels,
            0,
            jump_words,
            if inputs.is_empty() {
                first
            } else {
                self.tables.len() as u32
            },
            0,
            0,
        ];
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&header),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut words = self.tables.clone();
        words.extend_from_slice(inputs);
        let records = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&words),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let size = (out_words * 4) as u64;
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: records.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
            ],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&self.pipeline_layout),
            module: &self.module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(count.div_ceil(256), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, &read, 0, size);
        self.gpu.queue.submit([encoder.finish()]);
        let slice = read.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let words: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range().unwrap()).to_vec();
        read.unmap();
        words
    }
}

/// The device's `sin_cos_turn` equals the CPU's at all `2^24` turns: integer arithmetic
/// and the same table, so it must.
#[test]
fn the_device_sin_cos_equals_the_cpus_at_every_turn() {
    let Some(gpu) = gpu() else { return };
    let prober = Prober::new(gpu, 1);
    let started = std::time::Instant::now();
    let chunk = 1u32 << 22;
    let mut differ = 0usize;
    for c in 0..4u32 {
        let out = prober.run("probe_sin_cos", chunk, c * chunk, &[], 2 * chunk as usize);
        for k in 0..chunk {
            let (s, co) = sin_cos_turn(c * chunk + k);
            if out[2 * k as usize] != s.to_bits() || out[2 * k as usize + 1] != co.to_bits() {
                differ += 1;
            }
        }
    }
    println!(
        "sin_cos_turn: all 2^24 turns on the device in {:.2} s, {differ} differ from the CPU",
        started.elapsed().as_secs_f64()
    );
    assert_eq!(differ, 0);
}

/// (c) on the device: the stride tables reach `5 i` steps for `i` in a sweep of small
/// indices, powers of two and their neighbours, every single hex digit at every
/// position, a million, and the largest index six digits cover (past the largest
/// capacity, 4,194,240).
#[test]
fn the_device_jump_is_five_steps_a_particle() {
    let Some(gpu) = gpu() else { return };
    let digits = 6;
    let prober = Prober::new(gpu, digits);
    let mut cases = Vec::new();
    let mut indices: Vec<u32> = (0..300).collect();
    for b in 0..4 * digits {
        indices.extend([(1 << b) - 1, 1 << b, (1 << b) + 1]);
    }
    for k in 0..digits {
        indices.extend((1..16).map(|d| d << (4 * k)));
    }
    indices.extend([1_000_000, (1 << (4 * digits)) - 1]);
    for (n, &i) in indices.iter().enumerate() {
        let state = EffectRng::new(0x9E37_79B9 ^ (n as u32).wrapping_mul(0x85EB_CA6B)).next_u32();
        cases.push((state, i));
    }
    let inputs: Vec<u32> = cases.iter().flat_map(|&(s, i)| [s, i]).collect();
    let out = prober.run("probe_jump", cases.len() as u32, 0, &inputs, cases.len());
    for (k, &(state, i)) in cases.iter().enumerate() {
        // Sequential steps, against which `rng_jump`'s own tests hold the CPU jump.
        let mut stepped = EffectRng::new(state);
        for _ in 0..5 * i as u64 {
            stepped.next_u32();
        }
        assert_eq!(out[k], stepped.state(), "state {state:#x}, particle {i}");
    }
    println!(
        "jump_stride: {} indices up to {} on the device equal sequential steps",
        cases.len(),
        (1u32 << (4 * digits)) - 1
    );
}

/// The device's `sqrt_rn` and `div_rn` against the CPU's IEEE `sqrt` and `/`, over
/// every input shape the hemisphere feeds them (`1 - y^2` in `[0, 1]`, a squared length
/// up to a few, a component over a length) and a broad random sweep.
#[test]
fn the_device_sqrt_and_divide_are_correctly_rounded() {
    let Some(gpu) = gpu() else { return };
    let prober = Prober::new(gpu, 1);
    let mut rng = EffectRng::new(0x5151);
    let mut pairs: Vec<(f32, f32)> = Vec::new();
    for _ in 0..1 << 20 {
        // Exponents over a wide band, both signs for the dividend.
        let x = f32::from_bits((rng.next_u32() & 0x807F_FFFF) | ((90 + rng.next_u32() % 70) << 23));
        let y =
            f32::from_bits((rng.next_u32() & 0x007F_FFFF) | ((100 + rng.next_u32() % 50) << 23));
        pairs.push((x, y));
    }
    for k in 0..1u32 << 16 {
        // y = range(-1, 1) as the hemisphere draws it, and 1 - y^2.
        let y = -1.0 + 2.0 * ((k << 8) as f32 / 16_777_216.0);
        pairs.push((1.0 - y * y, 1.0 + y * y));
    }
    for v in [
        1.0f32,
        2.0,
        4.0,
        0.25,
        1.0 - f32::EPSILON / 2.0,
        1.0 + f32::EPSILON,
        3.0,
        1e-6,
        0.0,
        -0.0,
        -1.0,
    ] {
        pairs.push((v, 1.0));
        pairs.push((v, 3.0));
        pairs.push((v, 1e-6));
    }
    let inputs: Vec<u32> = pairs
        .iter()
        .flat_map(|&(x, y)| [x.to_bits(), y.to_bits()])
        .collect();
    let out = prober.run(
        "probe_sqrt_div",
        pairs.len() as u32,
        0,
        &inputs,
        4 * pairs.len(),
    );
    let mut checked = 0;
    for (k, &(x, y)) in pairs.iter().enumerate() {
        let (sq, q) = (x.abs().sqrt(), x / y.abs());
        // Below the normal range the device reads zero (documented); none arises in emission.
        // Both paths: the estimate settled by the bracket, and the fallback alone.
        if x.abs() >= f32::MIN_POSITIVE || x == 0.0 {
            assert_eq!(out[4 * k], sq.to_bits(), "sqrt {x:e}");
            assert_eq!(out[4 * k + 2], sq.to_bits(), "digit sqrt {x:e}");
        }
        if q.abs() >= f32::MIN_POSITIVE && q.is_finite() || x == 0.0 {
            assert_eq!(out[4 * k + 1], q.to_bits(), "{x:e} / {y:e}");
            assert_eq!(out[4 * k + 3], q.to_bits(), "long division {x:e} / {y:e}");
            checked += 1;
        }
    }
    println!("sqrt_rn and div_rn, each by both paths: {} square roots and {checked} quotients equal the CPU's", pairs.len());
}
