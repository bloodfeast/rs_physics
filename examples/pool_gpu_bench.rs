//! In-process, interleaved measurements for the resident GPU pool (POOL-GPU).
//!
//! The machine drifts with heat over minutes, so every configuration here is alternated
//! round by round inside one process and each figure is a median of rounds. GPU times
//! are timestamp queries around each pass; CPU times are `Instant` around the call.
//!
//! ```text
//! cargo run --release --example pool_gpu_bench --features "particles gpu" -- [rounds] [populations...]
//! ```
//!
//! The workload is the live sparks-and-dust pool of `benches/particle_effects.rs`, both
//! classes on a 32^3 swirl of 2 m cells. Per population and round, 60 frames each of:
//!
//! - CPU `integrate` (field off) and `integrate_in_air` with the field at 10 Hz and at
//!   20 Hz (each particle re-sampling once a field period), the field advanced outside
//!   the timed span;
//! - the GPU pool with an `rgba16float`, an `rgba32float` filtered (where the device
//!   can filter it) and an `rgba32float` exact field, the field advanced every 3 frames
//!   (20 Hz) and uploaded the frame it changes, and with no field;
//! - the `rgba16float` pool again with its emission staged the 0.3.3 way, as the
//!   "before" of POOL-GPU-RNG: every particle drawn on the CPU (the direction's sine and
//!   cosine from `f32::sin_cos`, as then) and staged as a 40-byte record, where the
//!   other GPU cases stage each burst as a descriptor and expand it on the device.
//!
//! Reported: GPU time per frame of the emit pass and the integrate pass, ns a particle;
//! the CPU cost on the frame thread of the GPU path (staging the emission, and the
//! encode with its one buffer write) per frame, and of a field upload; the bytes of the
//! frame's buffer write; the CPU paths in ns a particle; the crossover populations from
//! a straight-line fit of the GPU frame.

use std::time::Instant;

use rs_physics::gpu::{FieldFormat, GpuContext, GpuParticlePool, GpuPoolConfig, PoolTimestamps};
use rs_physics::particles::{
    Burst, EffectRng, ParticleClass, ParticleEffects, SwirlField, TurbulenceDrive,
};

const DT: f32 = 1.0 / 60.0;
const FRAMES: usize = 60;
const SPARK_LIFE: core::ops::Range<f32> = 0.2..0.5;
const DUST_LIFE: core::ops::Range<f32> = 20.0..40.0;

/// Where the live pool's emission goes.
trait Sink {
    fn burst(&mut self, burst: &Burst, rng: &mut EffectRng);
}

impl Sink for ParticleEffects {
    fn burst(&mut self, burst: &Burst, rng: &mut EffectRng) {
        self.emit(burst, rng);
    }
}

impl Sink for GpuParticlePool {
    fn burst(&mut self, burst: &Burst, rng: &mut EffectRng) {
        self.emit(burst, rng);
    }
}

/// The 0.3.3 emission into a GPU pool: each particle drawn on the CPU, its direction's
/// sine and cosine from `f32::sin_cos` of `unit() * TAU` as `EffectRng::hemisphere` was
/// then, and staged as a record with `emit_one`. The cost of what the descriptor path
/// replaced.
struct Records<'a>(&'a mut GpuParticlePool);

impl Sink for Records<'_> {
    fn burst(&mut self, burst: &Burst, rng: &mut EffectRng) {
        for _ in 0..burst.count {
            let azimuth = rng.unit() * core::f32::consts::TAU;
            let y = rng.range(-1.0, 1.0);
            let r = (1.0 - y * y).max(0.0).sqrt();
            let (sin, cos) = azimuth.sin_cos();
            let mut dir = [r * cos, y + burst.lift, r * sin];
            let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            if len > 1e-6 {
                dir = dir.map(|d| d / len);
            } else {
                dir = [0.0, 1.0, 0.0];
            }
            let speed = rng.range(burst.speed.start, burst.speed.end);
            let life = rng.range(burst.lifetime.start, burst.lifetime.end);
            let size = rng.range(burst.size.start, burst.size.end);
            self.0.emit_one(
                burst.origin,
                dir.map(|d| d * speed),
                life,
                size,
                burst.class,
            );
        }
    }
}

/// The steady-state emission of `benches/particle_effects.rs`: half sparks, half dust,
/// at the rates that hold `count` alive.
struct Feed {
    rng: EffectRng,
    owed: [f32; 2],
    rate: [f32; 2],
}

impl Feed {
    fn new(count: usize) -> Feed {
        let half = count as f32 / 2.0;
        Feed {
            rng: EffectRng::new(0xC0FFEE),
            owed: [0.0; 2],
            rate: [half / 0.35, half / 30.0],
        }
    }

    fn feed(&mut self, sink: &mut impl Sink) {
        for c in 0..2 {
            self.owed[c] += self.rate[c] * DT;
            let whole = self.owed[c].floor();
            self.owed[c] -= whole;
            sink.burst(
                &Burst {
                    origin: [0.0, 40.0, 0.0],
                    class: c as u8,
                    count: whole as u32,
                    speed: 6.0..15.0,
                    lifetime: if c == 0 { SPARK_LIFE } else { DUST_LIFE },
                    size: 0.7..1.3,
                    lift: 0.35,
                },
                &mut self.rng,
            );
        }
    }
}

fn classes() -> [ParticleClass; 2] {
    [
        ParticleClass {
            gravity: 26.0,
            drag: 1.4,
            restitution: 0.32,
        },
        ParticleClass {
            gravity: 1.6,
            drag: 3.4,
            restitution: 0.0,
        },
    ]
}

/// The swirl the field-on cases read: 32^3 cells of 2 m round the emitter, driven by a
/// 3 m/s plume 12 m wide, as in `benches/particle_effects.rs`.
fn swirl_field() -> SwirlField {
    let drive = TurbulenceDrive::new(3.0, 12.0).unwrap();
    SwirlField::new([-32.0, 8.0, -32.0], [32, 32, 32], 2.0, drive, 0x5EED).unwrap()
}

/// A CPU pool with its field, updated every `period` frames (0: no field).
struct Cpu {
    fx: ParticleEffects,
    feed: Feed,
    field: SwirlField,
    period: usize,
    frame: usize,
}

impl Cpu {
    fn new(count: usize, period: usize) -> Cpu {
        let mut fx = ParticleEffects::with_capacity(count + count / 4);
        for (c, class) in classes().into_iter().enumerate() {
            fx.set_class(c as u8, class);
            if period > 0 {
                fx.set_swirl(c as u8, 1.0);
            }
        }
        let mut cpu = Cpu {
            fx,
            feed: Feed::new(count),
            field: swirl_field(),
            period,
            frame: 0,
        };
        for _ in 0..(45.0 / DT) as usize {
            cpu.frame_ns();
        }
        cpu
    }

    /// One frame; returns the integrate's ns.
    fn frame_ns(&mut self) -> f64 {
        self.feed.feed(&mut self.fx);
        if self.period > 0 && self.frame % self.period == 0 {
            self.field.advance(self.period as f32 * DT);
        }
        self.frame += 1;
        let started = Instant::now();
        if self.period > 0 {
            self.fx
                .integrate_in_air(DT, self.field.velocity(), self.period as f32 * DT);
        } else {
            self.fx.integrate(DT);
        }
        started.elapsed().as_nanos() as f64
    }
}

/// A GPU pool with its field (`None`: no field), updated every 3 frames.
struct Gpu {
    pool: GpuParticlePool,
    feed: Feed,
    field: Option<SwirlField>,
    frame: usize,
    /// Emission staged as records drawn on the CPU (the 0.3.3 path), not descriptors.
    records: bool,
}

/// One round of a GPU pool: per-frame medians.
#[derive(Default, Clone, Copy)]
struct GpuRound {
    emit_gpu_ns: f64,
    integrate_gpu_ns: f64,
    stage_cpu_ns: f64,
    encode_cpu_ns: f64,
    upload_cpu_ns: f64,
    convert_gpu_ns: f64,
    frame_bytes: f64,
}

impl Gpu {
    fn new(
        gpu: &GpuContext,
        count: usize,
        format: Option<FieldFormat>,
        records: bool,
    ) -> Option<Gpu> {
        let mut config = GpuPoolConfig::new((count + count / 4) as u32);
        config.field = format.unwrap_or_default();
        let mut pool = GpuParticlePool::new(gpu, config).ok()?;
        for (c, class) in classes().into_iter().enumerate() {
            pool.set_class(c as u8, class);
            pool.set_swirl(c as u8, if format.is_some() { 1.0 } else { 0.0 });
        }
        let mut g = Gpu {
            pool,
            feed: Feed::new(count),
            field: format.map(|_| swirl_field()),
            frame: 0,
            records,
        };
        for chunk in 0..(45.0 / DT) as usize / FRAMES {
            let _ = chunk;
            g.round(gpu, None);
        }
        Some(g)
    }

    /// `FRAMES` frames, timed when `queries` is given.
    fn round(
        &mut self,
        gpu: &GpuContext,
        queries: Option<(&wgpu::QuerySet, &wgpu::Buffer, &wgpu::Buffer)>,
    ) -> GpuRound {
        let mut stage = Vec::with_capacity(FRAMES);
        let mut encode = Vec::with_capacity(FRAMES);
        let mut upload = Vec::new();
        let mut converted = Vec::new();
        let mut bytes = Vec::with_capacity(FRAMES);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for f in 0..FRAMES {
            let started = Instant::now();
            if self.records {
                self.feed.feed(&mut Records(&mut self.pool));
            } else {
                self.feed.feed(&mut self.pool);
            }
            stage.push(started.elapsed().as_nanos() as f64);
            if let Some(field) = &mut self.field {
                if self.frame % 3 == 0 {
                    field.advance(3.0 * DT);
                    let started = Instant::now();
                    self.pool.upload_field(field.velocity());
                    upload.push(started.elapsed().as_nanos() as f64);
                }
            }
            self.frame += 1;
            let base = (6 * f) as u32;
            let converting = self.pool.field_pending();
            if converting {
                converted.push(f);
            }
            let started = Instant::now();
            // One encoder a frame and one submission a frame, as a renderer would: what
            // `encode_timed` does, split to count the frame's bytes.
            let mut written = 0;
            self.pool.stage_frame_with(DT, |write| {
                if let rs_physics::gpu::PoolWrite::Buffer { bytes, .. } = write {
                    written = bytes.len();
                }
                write.write_now(&gpu.queue);
            });
            self.pool.encode_staged(
                &mut encoder,
                queries.map(|(set, _, _)| PoolTimestamps {
                    query_set: set,
                    field: Some(base),
                    emit: Some(base + 2),
                    integrate: Some(base + 4),
                }),
            );
            encode.push(started.elapsed().as_nanos() as f64);
            bytes.push(written as f64);
            gpu.queue.submit([std::mem::replace(
                &mut encoder,
                gpu.device.create_command_encoder(&Default::default()),
            )
            .finish()]);
        }
        let Some((set, resolve, read)) = queries else {
            gpu.queue.submit([encoder.finish()]);
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            return GpuRound::default();
        };
        encoder.resolve_query_set(set, 0..(6 * FRAMES) as u32, resolve, 0);
        encoder.copy_buffer_to_buffer(resolve, 0, read, 0, (6 * FRAMES * 8) as u64);
        gpu.queue.submit([encoder.finish()]);
        let slice = read.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let ticks: Vec<u64> = bytemuck::cast_slice(&slice.get_mapped_range().unwrap()).to_vec();
        read.unmap();
        let period = gpu.queue.get_timestamp_period() as f64;
        let span = |f: usize, k: usize| {
            (ticks[6 * f + k + 1].wrapping_sub(ticks[6 * f + k])) as f64 * period
        };
        let emit: Vec<f64> = (0..FRAMES).map(|f| span(f, 2)).collect();
        let integrate: Vec<f64> = (0..FRAMES).map(|f| span(f, 4)).collect();
        let convert: Vec<f64> = converted.iter().map(|&f| span(f, 0)).collect();
        GpuRound {
            emit_gpu_ns: median(emit),
            integrate_gpu_ns: median(integrate),
            stage_cpu_ns: median(stage),
            encode_cpu_ns: median(encode),
            upload_cpu_ns: median(upload),
            convert_gpu_ns: median(convert),
            frame_bytes: median(bytes),
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.retain(|x| !x.is_nan());
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let mut args = std::env::args().skip(1);
    let rounds: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(7);
    let mut populations: Vec<usize> = args.filter_map(|a| a.parse().ok()).collect();
    if populations.is_empty() {
        populations = vec![1_000, 10_000, 100_000, 1_000_000];
    }

    let Some(gpu) = GpuContext::with_features(
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::FLOAT32_FILTERABLE,
    ) else {
        eprintln!("no GPU adapter");
        return;
    };
    if !gpu
        .device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY)
    {
        eprintln!("the adapter has no timestamp queries");
        return;
    }
    let info = gpu.adapter_info();
    println!(
        "adapter: {} ({:?}), driver {}",
        info.name, info.backend, info.driver_info
    );
    println!(
        "rounds: {rounds}, {FRAMES} frames each, medians of per-frame times, then of rounds\n"
    );

    let set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("pool bench"),
        ty: wgpu::QueryType::Timestamp,
        count: (6 * FRAMES) as u32,
    });
    let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("resolve"),
        size: (6 * FRAMES * 8) as u64,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read"),
        size: (6 * FRAMES * 8) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let queries = Some((&set, &resolve, &read));

    overheads(&gpu);

    let formats: [(&str, Option<FieldFormat>, bool); 5] = [
        ("gpu f16", Some(FieldFormat::F16Filtered), false),
        ("gpu f16 records", Some(FieldFormat::F16Filtered), true),
        ("gpu f32 filtered", Some(FieldFormat::F32Filtered), false),
        ("gpu f32 exact", Some(FieldFormat::F32Exact), false),
        ("gpu no field", None, false),
    ];

    let mut fit = Vec::new();
    for &n in &populations {
        let mut cpu = [
            ("cpu integrate", Cpu::new(n, 0)),
            ("cpu air 10 Hz", Cpu::new(n, 6)),
            ("cpu air 20 Hz", Cpu::new(n, 3)),
        ];
        let mut gpus: Vec<(&str, Gpu)> = formats
            .iter()
            .filter_map(|(name, format, records)| {
                Gpu::new(&gpu, n, *format, *records).map(|g| (*name, g))
            })
            .collect();
        let mut cpu_rounds = vec![Vec::new(); cpu.len()];
        let mut gpu_rounds = vec![Vec::new(); gpus.len()];
        for _ in 0..rounds {
            for (k, (_, c)) in cpu.iter_mut().enumerate() {
                let per_frame: Vec<f64> = (0..FRAMES).map(|_| c.frame_ns()).collect();
                cpu_rounds[k].push(median(per_frame));
            }
            for (k, (_, g)) in gpus.iter_mut().enumerate() {
                gpu_rounds[k].push(g.round(&gpu, queries));
            }
        }
        let live = cpu[0].1.fx.len();
        println!("== {n} particles ({live} live on the CPU) ==");
        let mut cpu_ns = [0.0; 3];
        for (k, (name, _)) in cpu.iter().enumerate() {
            let t = median(cpu_rounds[k].clone());
            cpu_ns[k] = t / live as f64;
            println!(
                "  {name:<18} {:>10.1} us/frame  {:>6.2} ns/particle",
                t / 1e3,
                t / live as f64
            );
        }
        for (k, (name, g)) in gpus.iter().enumerate() {
            let r = &gpu_rounds[k];
            let pick = |f: fn(&GpuRound) -> f64| median(r.iter().map(f).collect());
            let (emit, integrate) = (pick(|r| r.emit_gpu_ns), pick(|r| r.integrate_gpu_ns));
            let counts = g.pool.read_counts_blocking();
            println!(
                "  {name:<18} gpu emit {:>7.1} us, integrate {:>7.1} us ({:.3} ns/particle, {} live, high water {}); cpu stage {:>6.1} us, encode {:>5.1} us, upload {:>6.1} us; convert {:>5.1} us; frame {:>8.0} bytes",
                emit / 1e3,
                integrate / 1e3,
                integrate / counts.live.max(1) as f64,
                counts.live,
                counts.high_water,
                pick(|r| r.stage_cpu_ns) / 1e3,
                pick(|r| r.encode_cpu_ns) / 1e3,
                pick(|r| r.upload_cpu_ns) / 1e3,
                pick(|r| r.convert_gpu_ns) / 1e3,
                pick(|r| r.frame_bytes),
            );
            if *name == "gpu f16" {
                fit.push((live as f64, emit + integrate, cpu_ns));
            }
        }
        println!();
    }

    // GPU frame = a + b n, least squares over the populations measured.
    if fit.len() >= 2 {
        let m = fit.len() as f64;
        let (sx, sy) = fit
            .iter()
            .fold((0.0, 0.0), |(a, b), (x, y, _)| (a + x, b + y));
        let (sxx, sxy) = fit
            .iter()
            .fold((0.0, 0.0), |(a, b), (x, y, _)| (a + x * x, b + x * y));
        let b = (m * sxy - sx * sy) / (m * sxx - sx * sx);
        let a = (sy - b * sx) / m;
        println!(
            "gpu f16 frame (emit + integrate) = {:.1} us + {:.3} ns x n",
            a / 1e3,
            b
        );
        for (k, name) in ["cpu integrate", "cpu air 10 Hz", "cpu air 20 Hz"]
            .iter()
            .enumerate()
        {
            let c = median(fit.iter().map(|(_, _, ns)| ns[k]).collect());
            if c > b {
                println!(
                    "  crossover against {name} ({c:.2} ns/particle): {:.0} particles",
                    a / (c - b)
                );
            } else {
                println!("  crossover against {name} ({c:.2} ns/particle): none");
            }
        }
    }
}

/// What the driver calls under an encode cost on the CPU, each alone: a 10 KB and a
/// 512 KiB `write_buffer`, an empty compute pass, and a command encoder's creation and
/// `finish`.
fn overheads(gpu: &GpuContext) {
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1 << 20,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let small = vec![0u8; 10 << 10];
    let big = vec![0u8; 512 << 10];
    let mut times = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for _ in 0..200 {
        let t = Instant::now();
        gpu.queue.write_buffer(&buffer, 0, &small);
        times[0].push(t.elapsed().as_nanos() as f64);
        let t = Instant::now();
        gpu.queue.write_buffer(&buffer, 0, &big);
        times[1].push(t.elapsed().as_nanos() as f64);
        let t = Instant::now();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        times[2].push(t.elapsed().as_nanos() as f64);
        let t = Instant::now();
        drop(encoder.begin_compute_pass(&Default::default()));
        times[3].push(t.elapsed().as_nanos() as f64);
        let t = Instant::now();
        let done = encoder.finish();
        times[4].push(t.elapsed().as_nanos() as f64);
        gpu.queue.submit([done]);
    }
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let names = [
        "write_buffer 10 KB",
        "write_buffer 512 KiB",
        "create encoder",
        "empty compute pass",
        "finish",
    ];
    for (name, t) in names.iter().zip(times) {
        println!("  {name:<22} {:>7.1} us", median(t) / 1e3);
    }
    println!();
}
