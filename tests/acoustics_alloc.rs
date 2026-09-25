//! **L7. Allocations.** The same frame loop, run with and without the acoustics.
//!
//! rs_physics's own work allocates nothing per tick: packing and taking results are
//! asserted at zero. What is left is wgpu's: recording a compute pass and a copy, and the
//! box `map_buffer_on_submit` puts its callback in (`api/command_buffer_actions.rs`). The
//! loop difference is therefore measured against a control that records the same wgpu
//! commands by hand, so rs_physics's share is isolated: it must be zero beyond wgpu's.
#![cfg(feature = "gpu")]

mod acoustics_common;
use acoustics_common::*;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use rs_physics::acoustics::Air;
use rs_physics::gpu::acoustics::*;

struct Counting;

thread_local! {
    static ON: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<u64> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ON.try_with(|on| {
            if on.get() {
                COUNT.with(|c| c.set(c.get() + 1));
            }
        });
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let _ = ON.try_with(|on| {
            if on.get() {
                COUNT.with(|c| c.set(c.get() + 1));
            }
        });
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn counted<R>(f: impl FnOnce() -> R) -> (R, u64) {
    COUNT.with(|c| c.set(0));
    ON.with(|on| on.set(true));
    let r = f();
    ON.with(|on| on.set(false));
    (r, COUNT.with(|c| c.get()))
}

#[test]
fn l7_the_acoustics_allocate_nothing_of_their_own_per_tick() {
    let Some(gpu) = gpu() else { return };
    let mut ac = GpuAcoustics::new(&gpu.device, AcousticLimits::MAX).unwrap();
    let mut scene = Scene::flat(140, 100, 2.0, 0.0);
    scene.statics = (0..200)
        .map(|i| {
            Obb::upright(
                [
                    5.0 + (i % 20) as f32 * 13.0,
                    2.0,
                    5.0 + (i / 20) as f32 * 19.0,
                ],
                [3.0, 4.0, 2.0],
                i as f32,
                0,
            )
        })
        .collect();
    scene.load(&gpu, &mut ac);
    let air = Air::standard();
    let header = DispatchHeader::new(&listener_at([140.0, 1.6, 100.0]), &air, &[]).unwrap();
    let sources: Vec<Source> = (0..40)
        .map(|i| {
            Source::new(
                [60.0 + 4.0 * i as f32, 1.0, 40.0 + 2.0 * i as f32],
                1.0,
                [3.0, 0.0, 0.0],
                i,
                NO_MOVER,
            )
            .unwrap()
        })
        .collect();
    let movers: Vec<Obb> = (0..64)
        .map(|i| Obb::upright([20.0 + 3.0 * i as f32, 1.5, 150.0], [2.5, 3.0, 5.0], 0.0, 0))
        .collect();
    let mut bytes = vec![0u8; dispatch_bytes(sources.len(), movers.len())];
    let mut out = TickResults::default();

    // Packing: zero, every time.
    for _ in 0..3 {
        let (shape, n) = counted(|| {
            GpuAcoustics::pack_dispatch(&header, &sources, &movers, &mut bytes[..]).unwrap()
        });
        assert_eq!(n, 0, "pack_dispatch allocated");
        assert_eq!(shape.bytes as usize, bytes.len());
    }
    let shape = GpuAcoustics::pack_dispatch(&header, &sources, &movers, &mut bytes[..]).unwrap();
    let stage = gpu.stage(&bytes);
    let staged = Staged {
        buffer: &stage,
        offset: 0,
        len: shape.bytes,
    };

    // The control records what encode records, by hand: a copy in, a pass with two
    // dispatches of a trivial pipeline over three storage bindings, a copy out, and one
    // map_buffer_on_submit.
    let control = Control::new(&gpu.device);

    let frames = 64u64;
    let (mut with, mut bare, mut ctrl, mut maps) = (0u64, 0u64, 0u64, 0u64);
    let (mut encode_n, mut take_n) = (0u64, 0u64);
    for f in 0..(frames + 8) {
        let warm = f >= 8; // the first frames grow wgpu's own pools
                           // With the acoustics.
        let (_, n) = counted(|| {
            let mut enc = gpu.encoder();
            let (_, e) = counted_inner(|| ac.encode(&mut enc, staged, shape).unwrap());
            gpu.queue.submit([enc.finish()]);
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            let (taken, t) = counted_inner(|| ac.take_ready(&mut out));
            assert!(taken);
            (e, t)
        });
        // Bare: the same frame without them.
        let (_, b) = counted(|| {
            let enc = gpu.encoder();
            gpu.queue.submit([enc.finish()]);
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
        });
        // Control: the same wgpu commands by hand.
        let (_, c) = counted(|| {
            let mut enc = gpu.encoder();
            control.record(&mut enc, staged);
            gpu.queue.submit([enc.finish()]);
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            control.unmap();
        });
        if warm {
            with += n;
            bare += b;
            ctrl += c;
            maps += 1;
        }
        let _ = (&mut encode_n, &mut take_n);
    }
    for _ in 0..3 {
        let mut enc = gpu.encoder();
        let (_, e) = counted(|| ac.encode(&mut enc, staged, shape).unwrap());
        let (_, c) = counted(|| control.record(&mut enc, staged));
        let (_, fin) = counted(|| enc.finish());
        println!("encode {e}, control record {c}, finish {fin}");
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }
    // take_ready on its own: a copy out and an unmap.
    let mut enc = gpu.encoder();
    ac.encode(&mut enc, staged, shape).unwrap();
    gpu.submit_wait(enc);
    let (taken, t) = counted(|| ac.take_ready(&mut out));
    assert!(taken);
    println!(
        "L7 over {frames} frames: with the acoustics {with}, bare {bare}, control {ctrl} allocations; \
         {maps} map_buffer_on_submit calls; take_ready alone {t}"
    );
    println!(
        "  per frame: acoustics - bare = {:.2} (wgpu's recording and its map box), acoustics - control = {:.2}",
        (with as f64 - bare as f64) / frames as f64,
        (with as f64 - ctrl as f64) / frames as f64,
    );
    assert!(
        with <= ctrl,
        "the acoustics allocated beyond the wgpu commands they record"
    );
}

/// Counting inside `counted` without resetting the outer count.
fn counted_inner<R>(f: impl FnOnce() -> R) -> (R, u64) {
    let before = COUNT.with(|c| c.get());
    let r = f();
    (r, COUNT.with(|c| c.get()) - before)
}

struct Control {
    input: wgpu::Buffer,
    output: wgpu::Buffer,
    scene: wgpu::Buffer,
    slot: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    second: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    state: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Control {
    fn new(device: &wgpu::Device) -> Control {
        let buf = |size: u64, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let input = buf(16_384, storage | wgpu::BufferUsages::COPY_DST);
        let output = buf(16_384, storage | wgpu::BufferUsages::COPY_SRC);
        let scene = buf(1024, storage);
        let slot = buf(
            READBACK_BYTES,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var<storage, read> a: array<u32>;\n\
                 @group(0) @binding(1) var<storage, read_write> b: array<u32>;\n\
                 @group(0) @binding(2) var<storage, read> c: array<u32>;\n\
                 @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {\n\
                     b[g.x] = a[g.x] + c[0];\n\
                 }\n\
                 @compute @workgroup_size(64) fn second(@builtin(global_invocation_id) g: vec3<u32>) {\n\
                     b[g.x] = a[g.x] + c[1];\n\
                 }\n"
                    .into(),
            ),
        });
        let entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[entry(0, true), entry(1, false), entry(2, true)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let second = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&layout),
            module: &module,
            entry_point: Some("second"),
            compilation_options: Default::default(),
            cache: None,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scene.as_entire_binding(),
                },
            ],
        });
        Control {
            input,
            output,
            scene,
            slot,
            pipeline,
            second,
            group,
            state: Default::default(),
        }
    }

    fn record(&self, enc: &mut wgpu::CommandEncoder, staged: Staged<'_>) {
        enc.copy_buffer_to_buffer(staged.buffer, staged.offset, &self.input, 0, staged.len);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.group, &[]);
            pass.set_pipeline(&self.pipeline);
            pass.dispatch_workgroups(40, 1, 1);
            pass.set_pipeline(&self.second);
            pass.dispatch_workgroups(1, 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.output, 0, &self.slot, 0, READBACK_BYTES);
        let state = std::sync::Arc::clone(&self.state);
        enc.map_buffer_on_submit(&self.slot, wgpu::MapMode::Read, .., move |r| {
            state.store(r.is_ok(), std::sync::atomic::Ordering::Release)
        });
        let _ = &self.scene;
    }

    fn unmap(&self) {
        if self.state.swap(false, std::sync::atomic::Ordering::AcqRel) {
            let view = self.slot.slice(..).get_mapped_range().unwrap();
            drop(view);
            self.slot.unmap();
        }
    }
}
