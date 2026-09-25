//! **L14, the cost gate**, at the 300-a-side shape: 40 sources (a tick's new starts and
//! the moving emitters), 64 movers, 200 statics, a 140 x 100 terrain at 2 m, and the full
//! listener field.
//!
//! * **GPU:** the pass's span from timestamps written at its start and end, per dispatch.
//! * **CPU:** the host's `pack_dispatch` (straight into mapped staging memory, as the
//!   engine's ring lends it) plus `encode`, per dispatch.
//!
//! **The gate is a budget we chose** (r1_contract.md): at most 20 us GPU, 1% of the 2 ms
//! stretch frame, and at most 10 us CPU. The bench prints the medians and exits non-zero
//! when either is over. Run it under the machine's measurement discipline:
//!
//!     py -3 tools/quiet.py -- target/release/deps/acoustics-<hash>.exe
//!
//! (`quiet.py` appends `--measure`, which this harness accepts and ignores.)

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::Criterion;
use rs_physics::acoustics::Air;
use rs_physics::gpu::acoustics::*;

const GATE_GPU_NS: f64 = 20_000.0;
const GATE_CPU_NS: f64 = 10_000.0;
const WARM: usize = 300;
const FRAMES: usize = 3_000;

struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// A staging slot as the engine's ring keeps one: mapped for writing, re-armed by
/// `map_buffer_on_submit` once the submission that copies out of it retires.
struct Slot {
    buffer: wgpu::Buffer,
    ready: Arc<AtomicBool>,
}

struct Rig {
    device: wgpu::Device,
    queue: wgpu::Queue,
    acoustics: GpuAcoustics,
    header: DispatchHeader,
    sources: Vec<Source>,
    movers: Vec<Obb>,
    stage: Vec<Slot>,
    next: usize,
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    period_ns: f64,
    out: TickResults,
}

impl Rig {
    fn new() -> Option<Rig> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()?;
        if !adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            println!("acoustics bench: the adapter has no timestamp queries; nothing to gate");
            return None;
        }
        let info = adapter.get_info();
        println!("acoustics bench: {} ({:?})", info.name, info.backend);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("acoustics bench"),
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .ok()?;
        let mut acoustics = GpuAcoustics::new(&device, AcousticLimits::MAX).unwrap();

        // The scene: rolling terrain, 200 statics, 4 materials.
        let (cols, rows) = (140u32, 100u32);
        let mut heights = vec![0.0f32; (cols * rows) as usize];
        for r in 0..rows as usize {
            for c in 0..cols as usize {
                let (x, z) = (c as f64 * 2.0, r as f64 * 2.0);
                heights[r * cols as usize + c] =
                    (4.0 * (x / 37.0).sin() * (z / 23.0).cos() + 2.0 * (x / 11.0 + z / 17.0).sin()) as f32;
            }
        }
        let material: Vec<u8> = (0..heights.len()).map(|i| (i % 4) as u8).collect();
        let terrain = TerrainGrid { heights: &heights, material: &material, cols, rows, cell_m: 2.0, origin: [0.0, 0.0] };
        let mut rng = Rng(0xbe4c);
        let statics: Vec<Obb> = (0..200)
            .map(|i| {
                Obb::upright(
                    [rng.range(10.0, 270.0) as f32, 3.0, rng.range(10.0, 190.0) as f32],
                    [rng.range(1.0, 12.0) as f32, rng.range(2.0, 9.0) as f32, rng.range(1.0, 12.0) as f32],
                    rng.range(0.0, 6.28) as f32,
                    i % 4,
                )
            })
            .collect();
        let materials = [0.98f32, 0.9, 0.7, 0.5].map(|r| AcousticMaterial { reflection: [r; 4] });
        let layout = GpuAcoustics::scene_bytes(&terrain, &statics, 4, None).unwrap();
        let scene_stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: layout.staged_bytes(),
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        {
            let mut view = scene_stage.slice(..).get_mapped_range_mut().unwrap();
            GpuAcoustics::pack_scene(&layout, &terrain, &statics, &materials, None, view.slice(..)).unwrap();
        }
        scene_stage.unmap();
        let mut enc = device.create_command_encoder(&Default::default());
        let staged = Staged { buffer: &scene_stage, offset: 0, len: layout.staged_bytes() };
        acoustics.encode_scene(&mut enc, &layout, staged).unwrap();
        queue.submit([enc.finish()]);

        let listener = Listener { position: [140.0, 1.6, 100.0], forward: [0.0, 0.0, -1.0], right: [1.0, 0.0, 0.0], velocity: [0.0; 3] };
        let ladder = [(0.0f32, 20_000.0f32), (45.0, 1_400.0), (130.0, 480.0)];
        let header = DispatchHeader::new(&listener, &Air::standard(), &ladder).unwrap();
        let sources = (0..40)
            .map(|i| {
                let a = rng.range(0.0, 6.28);
                let d = rng.range(5.0, 75.0);
                Source::new(
                    [(140.0 + d * a.cos()) as f32, 6.0, (100.0 + d * a.sin()) as f32],
                    1.0,
                    [rng.range(-10.0, 10.0) as f32, 0.0, rng.range(-10.0, 10.0) as f32],
                    i,
                    NO_MOVER,
                )
                .unwrap()
            })
            .collect();
        let movers = (0..64)
            .map(|_| {
                Obb::upright(
                    [rng.range(80.0, 200.0) as f32, 3.0, rng.range(40.0, 160.0) as f32],
                    [2.5, 3.0, 5.0],
                    rng.range(0.0, 6.28) as f32,
                    0,
                )
            })
            .collect();
        let stage = (0..4)
            .map(|_| Slot {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("bench ring slot"),
                    size: 8192,
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                }),
                ready: Arc::new(AtomicBool::new(true)),
            })
            .collect();
        let queries = device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 2 });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let period_ns = queue.get_timestamp_period() as f64;
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        Some(Rig {
            device,
            queue,
            acoustics,
            header,
            sources,
            movers,
            stage,
            next: 0,
            queries,
            resolve,
            read,
            period_ns,
            out: TickResults::default(),
        })
    }

    /// One tick: pack into a ring slot and encode (timed on the CPU), resolve the pass's
    /// timestamps, submit, and read them back. Returns (cpu ns, gpu ns).
    fn tick(&mut self) -> (f64, f64) {
        let k = self.next % self.stage.len();
        self.next += 1;
        let slot = &self.stage[k];
        assert!(slot.ready.load(Ordering::Acquire), "the bench ring ran dry");
        let mut enc = self.device.create_command_encoder(&Default::default());
        let began = Instant::now();
        let shape = {
            let mut view = slot.buffer.slice(..).get_mapped_range_mut().unwrap();
            GpuAcoustics::pack_dispatch(&self.header, &self.sources, &self.movers, view.slice(..)).unwrap()
        };
        slot.buffer.unmap();
        let staged = Staged { buffer: &slot.buffer, offset: 0, len: shape.bytes };
        let marks = wgpu::ComputePassTimestampWrites {
            query_set: &self.queries,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        };
        let encoded = self.acoustics.encode_timed(&mut enc, staged, shape, Some(marks)).unwrap();
        let cpu = began.elapsed().as_nanos() as f64;
        assert!(matches!(encoded, Encoded::Dispatched { .. }));
        slot.ready.store(false, Ordering::Release);
        let ready = Arc::clone(&slot.ready);
        enc.map_buffer_on_submit(&slot.buffer, wgpu::MapMode::Write, .., move |r| {
            ready.store(r.is_ok(), Ordering::Release)
        });
        enc.resolve_query_set(&self.queries, 0..2, &self.resolve, 0);
        enc.copy_buffer_to_buffer(&self.resolve, 0, &self.read, 0, 16);
        enc.map_buffer_on_submit(&self.read, wgpu::MapMode::Read, .., |_| {});
        self.queue.submit([enc.finish()]);
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let ticks: [u64; 2] = {
            let view = self.read.slice(..).get_mapped_range().unwrap();
            bytemuck::pod_read_unaligned(&view[..16])
        };
        self.read.unmap();
        assert!(self.acoustics.take_ready(&mut self.out));
        let gpu = ticks[1].wrapping_sub(ticks[0]) as f64 * self.period_ns;
        (cpu, gpu)
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn fmt(ns: f64) -> String {
    if ns < 1_000.0 { format!("{ns:.0} ns") } else if ns < 1_000_000.0 { format!("{:.2} us", ns / 1e3) } else { format!("{:.3} ms", ns / 1e6) }
}

fn main() {
    let Some(mut rig) = Rig::new() else { return };
    for _ in 0..WARM {
        rig.tick();
    }
    let mut cpu = Vec::with_capacity(FRAMES);
    let mut gpu = Vec::with_capacity(FRAMES);
    for _ in 0..FRAMES {
        let (c, g) = rig.tick();
        cpu.push(c);
        gpu.push(g);
    }
    let (c50, g50) = (median(&mut cpu), median(&mut gpu));
    let (c99, g99) = (cpu[cpu.len() * 99 / 100], gpu[gpu.len() * 99 / 100]);

    // Criterion's view of the host's share, for its report.
    let mut criterion = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3));
    criterion.bench_function("acoustics pack + encode (40 sources, 64 movers)", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                total += Duration::from_nanos(rig.tick().0 as u64);
            }
            total
        })
    });
    criterion.final_summary();

    println!(
        "acoustics gate over {FRAMES} dispatches: GPU median {} (p99 {}), gate {}; CPU pack + encode median {} (p99 {}), gate {}",
        fmt(g50), fmt(g99), fmt(GATE_GPU_NS), fmt(c50), fmt(c99), fmt(GATE_CPU_NS)
    );
    let over_gpu = g50 > GATE_GPU_NS;
    let over_cpu = c50 > GATE_CPU_NS;
    if over_gpu || over_cpu {
        println!(
            "acoustics gate FAILED:{}{}",
            if over_gpu { " GPU over 20 us" } else { "" },
            if over_cpu { " CPU over 10 us" } else { "" }
        );
        std::process::exit(1);
    }
    println!("acoustics gate passed");
}
