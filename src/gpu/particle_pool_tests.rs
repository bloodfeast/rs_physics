//! The resident pool against the CPU pool it mirrors.
//!
//! Tolerance tests, not bit-identity: the device may fuse a multiply and the add after
//! it into one rounding, and a filtered field fetch has the texture unit's precision.
//! Every bound below is derived from the operations the integrate does and the field
//! it reads, not tuned; each test prints what it measured beside its bound.
//!
//! Every test skips with a message when no adapter exists (CI has none).

use std::collections::{HashMap, HashSet};

use crate::gpu::{FieldFormat, GpuContext, GpuParticlePool, GpuPoolConfig, GroundHeights};
use crate::particles::{
    Backend, BackendPolicy, EffectRng, ParticleClass, ParticleEffects, SwirlField, TurbulenceDrive,
    VelocityGrid,
};

const DT: f32 = 1.0 / 60.0;
/// The unit roundoff of `f32`: one rounding changes a result by at most `U |r|`.
const U: f32 = f32::EPSILON / 2.0;

fn gpu(features: wgpu::Features) -> Option<GpuContext> {
    let gpu = GpuContext::with_features(features);
    if gpu.is_none() {
        println!("No GPU adapter: skipping the resident pool test");
    }
    gpu
}

/// Sparks and dust on the air, as the bench's live pool, and a heavy class that
/// ignores it.
fn classes(swirl: bool) -> [(ParticleClass, f32); 3] {
    let s = if swirl { 1.0 } else { 0.0 };
    [
        (ParticleClass { gravity: 26.0, drag: 1.4, restitution: 0.32 }, s),
        (ParticleClass { gravity: 1.6, drag: 3.4, restitution: 0.0 }, s),
        (ParticleClass { gravity: 9.8, drag: 0.2, restitution: 0.5 }, 0.0),
    ]
}

fn set_classes(fx: &mut ParticleEffects, pool: &mut GpuParticlePool, swirl: bool) {
    for (c, (class, s)) in classes(swirl).into_iter().enumerate() {
        fx.set_class(c as u8, class);
        fx.set_swirl(c as u8, s);
    }
    pool.copy_classes_from(fx);
}

/// A swirl of 16^3 cells of 2 m over the 32 m round the emitter, advanced so its
/// octaves have structure.
fn swirl_air() -> VelocityGrid {
    let drive = TurbulenceDrive::new(3.0, 12.0).unwrap();
    let mut field = SwirlField::new([-16.0, 0.0, -16.0], [16, 16, 16], 2.0, drive, 0x5EED).unwrap();
    for _ in 0..5 {
        field.advance(0.1);
    }
    // A breeze on top, so the field is not zero on its outer layer and the clamp is read.
    let mut air = field.velocity().clone();
    let [nx, ny, nz] = air.dims();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let v = air.get(i, j, k);
                air.set(i, j, k, [v[0] + 1.5, v[1] + 0.4, v[2] - 0.7]);
            }
        }
    }
    air
}

/// The largest magnitude of a cell (`A`), and the largest difference between two
/// neighbouring cells along an axis (`D`), over every component.
fn field_extent(air: &VelocityGrid) -> (f32, f32) {
    let [nx, ny, nz] = air.dims();
    let (mut a, mut d) = (0.0f32, 0.0f32);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let v = air.get(i, j, k);
                a = a.max(v.iter().fold(0.0, |m, c| m.max(c.abs())));
                for (di, dj, dk) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] {
                    if i + di < nx && j + dj < ny && k + dk < nz {
                        let w = air.get(i + di, j + dj, k + dk);
                        for c in 0..3 {
                            d = d.max((w[c] - v[c]).abs());
                        }
                    }
                }
            }
        }
    }
    (a, d)
}

/// How far the device's fetch may be from `VelocityGrid::sample`, from the format's
/// arithmetic:
///
/// * exact: the CPU's seven lerps in three nested levels; a fused multiply-add at a
///   level changes it by one rounding, `U A`, and the levels are convex combinations,
///   so the differences add: `3 U A`.
/// * filtered: the texture unit's interpolation weights carry 8 bits of fraction
///   (Vulkan's `subTexelPrecisionBits`, 8 on desktop hardware), so each axis' weight is
///   off by up to `2^-8` and the value by `2^-8 D` an axis: `3 D / 256`. The texel
///   coordinate is computed as `p * scale + offset` rather than the CPU's
///   `(p - centre) / h`, a few roundings of the coordinate: `4 U A` covers it.
/// * `rgba16float` adds the storage: each cell rounded to 11 significant bits, off by
///   up to `A 2^-10` toward zero (`2^-11` to nearest), and the filter's own arithmetic
///   on half-precision texels up to as much again: `2 A 2^-10`.
fn fetch_bound(format: FieldFormat, a: f32, d: f32) -> f32 {
    match format {
        FieldFormat::F32Exact => 3.0 * U * a,
        FieldFormat::F32Filtered => 3.0 * d / 256.0 + 4.0 * U * a,
        FieldFormat::F16Filtered => 3.0 * d / 256.0 + 4.0 * U * a + 2.0 * a / 1024.0,
    }
}

/// How far a particle's position may drift from the CPU's over `n` steps, given the
/// fetch bound `fetch` (m/s), the largest speed `v` and coordinate `p` the run reaches,
/// and the field's steepest gradient `g` (1/s).
///
/// A step's velocity is `v d + t a` (`d = 1 - k`, `t = swirl k <= k`). The device may
/// fuse the add into the multiply before it: one rounding, at most `U V`. The air read
/// is off by the fetch's own error plus `g dp` when the position is off by `dp`. Since
/// `t <= 1 - d`, the velocity error is a convex mix of its last value and the air's
/// error, plus the rounding, so after `m` steps it is at most `fetch + g dp_m + m U V`.
/// The position `p + v dt` may also fuse: `U P` a step. So, step by step,
///
/// ```text
/// dp_(m+1) <= (1 + g dt) dp_m + dt (fetch + m U V) + U P
/// ```
///
/// the discrete Gronwall bound: a steep field amplifies any difference, by up to
/// `e^(g n dt)`, because two particles a little apart are carried apart by it. That is
/// the field's physics, not the device's error, and the bound includes it.
fn trajectory_bound(n: usize, fetch: f32, v: f32, p: f32, g: f32) -> f32 {
    let mut dp = 0.0f64;
    for m in 0..n {
        dp = (1.0 + g as f64 * DT as f64) * dp
            + DT as f64 * (fetch as f64 + m as f64 * U as f64 * v as f64)
            + U as f64 * p as f64;
    }
    dp as f32
}

/// Particle `id` carries `id` as its size, so the two pools can be matched after the
/// CPU's compaction reorders its particles and the device's free list scatters them.
struct Emitter {
    rng: EffectRng,
    next_id: u32,
}

impl Emitter {
    fn new(seed: u32) -> Emitter {
        Emitter { rng: EffectRng::new(seed), next_id: 1 }
    }

    /// `count` particles round the emitter, every class, lifetimes from a fifth of a
    /// second to two seconds so some retire during a run.
    fn emit(&mut self, count: usize, fx: &mut ParticleEffects, pool: &mut GpuParticlePool) {
        for _ in 0..count {
            let r = &mut self.rng;
            let pos = [r.range(-10.0, 10.0), r.range(2.0, 14.0), r.range(-10.0, 10.0)];
            let dir = r.hemisphere(0.4);
            let speed = r.range(1.0, 12.0);
            let vel = [dir[0] * speed, dir[1] * speed, dir[2] * speed];
            let life = r.range(0.2, 2.0);
            let class = (self.next_id % 3) as u8;
            let id = self.next_id as f32;
            self.next_id += 1;
            fx.emit_one(pos, vel, life, id, class);
            pool.emit_one(pos, vel, life, id, class);
        }
    }
}

/// The CPU pool's particles by id: position, velocity, class.
fn cpu_by_id(fx: &ParticleEffects) -> HashMap<u32, ([f32; 3], [f32; 3])> {
    (0..fx.len()).map(|i| (fx.size(i) as u32, (fx.position(i), fx.velocity(i)))).collect()
}

/// The device pool's live particles by id.
fn gpu_by_id(pool: &GpuParticlePool) -> HashMap<u32, ([f32; 3], [f32; 3])> {
    let mut out = HashMap::new();
    for slot in pool.read_slots_blocking() {
        if slot.remaining > 0.0 {
            let previous = out.insert(slot.size as u32, (slot.position, slot.velocity));
            assert!(previous.is_none(), "particle {} is in two slots", slot.size);
        }
    }
    out
}

fn max_abs(v: [f32; 3]) -> f32 {
    v.iter().fold(0.0f32, |m, c| m.max(c.abs()))
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    max_abs([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
}

/// Runs the CPU's `integrate_in_air` (every particle re-sampling every frame, the
/// device's behaviour) and the device's integrate side by side for `frames`, and checks
/// the population and every particle's position and velocity against the bounds.
fn track_the_cpu(format: FieldFormat, features: wgpu::Features) {
    let Some(gpu) = gpu(features) else { return };
    if format == FieldFormat::F32Filtered && !gpu.device.features().contains(wgpu::Features::FLOAT32_FILTERABLE) {
        println!("No FLOAT32_FILTERABLE: skipping the f32 filtered case");
        return;
    }
    let mut config = GpuPoolConfig::new(8_192);
    config.field = format;
    let mut pool = GpuParticlePool::new(&gpu, config).unwrap();
    let mut fx = ParticleEffects::with_capacity(8_192);
    set_classes(&mut fx, &mut pool, true);
    let air = swirl_air();
    pool.upload_field(&air);
    let (a, d) = field_extent(&air);
    let g = d / air.cell_size();

    let mut emitter = Emitter::new(0xBEEF);
    let frames = 60;
    let (mut v_max, mut p_max) = (0.0f32, 0.0f32);
    for frame in 0..frames {
        emitter.emit(if frame == 0 { 2_000 } else { 40 }, &mut fx, &mut pool);
        fx.integrate_in_air(DT, &air, DT);
        pool.step(DT);
        for (p, v) in cpu_by_id(&fx).values() {
            v_max = v_max.max(max_abs(*v));
            p_max = p_max.max(max_abs(*p));
        }
    }

    let emitted = (emitter.next_id - 1) as usize;
    let counts = pool.read_counts_blocking();
    assert_eq!(counts.live as usize, fx.len(), "live counts differ");
    assert_eq!(counts.retired as usize, emitted - fx.len(), "retire counts differ");
    assert_eq!(counts.placed as usize, emitted);
    assert_eq!(counts.free + counts.live, pool.capacity());

    let cpu = cpu_by_id(&fx);
    let device = gpu_by_id(&pool);
    assert_eq!(cpu.keys().collect::<HashSet<_>>(), device.keys().collect::<HashSet<_>>());

    let fetch = fetch_bound(format, a, d);
    let bound = trajectory_bound(frames, fetch, v_max, p_max, g);
    let (mut worst_p, mut worst_v) = (0.0f32, 0.0f32);
    for (id, (p, v)) in &cpu {
        let (gp, gv) = device[id];
        worst_p = worst_p.max(distance(*p, gp));
        worst_v = worst_v.max(distance(*v, gv));
    }
    // The velocity's share of the same derivation: `fetch + g dp + n U V`.
    let v_bound = fetch + g * bound + frames as f32 * U * v_max;
    println!(
        "{format:?}: {} live; position {worst_p:.3e} m (bound {bound:.3e}), velocity {worst_v:.3e} m/s (bound {v_bound:.3e}); A {a:.3}, D {d:.3}, V {v_max:.2}, P {p_max:.2}",
        cpu.len()
    );
    assert!(worst_p <= bound, "{format:?}: position {worst_p} past {bound}");
    assert!(worst_v <= v_bound, "{format:?}: velocity {worst_v} past {v_bound}");
}

#[test]
fn the_exact_fetch_integrate_tracks_the_cpu_within_float_reordering() {
    track_the_cpu(FieldFormat::F32Exact, wgpu::Features::empty());
}

#[test]
fn the_f16_field_integrate_tracks_the_cpu_within_the_filter_bound() {
    track_the_cpu(FieldFormat::F16Filtered, wgpu::Features::empty());
}

#[test]
fn the_f32_filtered_integrate_tracks_the_cpu_within_the_filter_bound() {
    track_the_cpu(FieldFormat::F32Filtered, wgpu::Features::FLOAT32_FILTERABLE);
}

/// The fetch alone, at points inside the grid, on its edges and outside it, against
/// `VelocityGrid::sample`.
#[test]
fn each_fetch_is_within_its_format_precision() {
    let Some(gpu) = gpu(wgpu::Features::FLOAT32_FILTERABLE) else { return };
    let air = swirl_air();
    let (a, d) = field_extent(&air);
    let mut rng = EffectRng::new(17);
    let points: Vec<[f32; 3]> = (0..4_096)
        .map(|_| [rng.range(-20.0, 20.0), rng.range(-4.0, 36.0), rng.range(-20.0, 20.0)])
        .collect();
    let mut by_format = Vec::new();
    for format in [FieldFormat::F32Exact, FieldFormat::F32Filtered, FieldFormat::F16Filtered] {
        let mut config = GpuPoolConfig::new(64);
        config.field = format;
        let Ok(mut pool) = GpuParticlePool::new(&gpu, config) else {
            println!("{format:?}: not on this device");
            continue;
        };
        pool.upload_field(&air);
        let fetched = pool.probe_air_blocking(&points);
        let worst = points
            .iter()
            .zip(&fetched)
            .map(|(p, f)| distance(air.sample(*p), *f))
            .fold(0.0f32, f32::max);
        let bound = fetch_bound(format, a, d);
        println!("{format:?}: fetch {worst:.3e} m/s (bound {bound:.3e}; A {a:.3}, D {d:.3})");
        assert!(worst <= bound, "{format:?}: fetch {worst} past {bound}");
        by_format.push((format, fetched));
    }
    // The half-precision storage alone: the f16 field against the f32 field through the
    // same hardware filter. Rounding 11 significant bits is `A 2^-11` to nearest,
    // `A 2^-10` toward zero, and the filter's half-precision arithmetic as much again.
    let f32_filtered = by_format.iter().find(|(f, _)| *f == FieldFormat::F32Filtered);
    let f16_filtered = by_format.iter().find(|(f, _)| *f == FieldFormat::F16Filtered);
    if let (Some((_, wide)), Some((_, half))) = (f32_filtered, f16_filtered) {
        let worst = wide.iter().zip(half).map(|(w, h)| distance(*w, *h)).fold(0.0f32, f32::max);
        let bound = 2.0 * a / 1024.0;
        println!("f16 storage against the f32 field: {worst:.3e} m/s (bound {bound:.3e}, {:.4}% of A)", 100.0 * worst / a);
        assert!(worst <= bound, "f16 storage {worst} past {bound}");
    }
}

/// The keep-the-field-off guarantee, on the device: a class that ignores the air
/// integrates to the bit as it does with no field at all.
#[test]
fn a_class_off_the_air_is_bit_identical_with_and_without_a_field() {
    let Some(gpu) = gpu(wgpu::Features::empty()) else { return };
    let mut on = GpuParticlePool::new(&gpu, GpuPoolConfig::new(4_096)).unwrap();
    let mut off = GpuParticlePool::new(&gpu, GpuPoolConfig::new(4_096)).unwrap();
    let mut fx = ParticleEffects::with_capacity(4_096);
    set_classes(&mut fx, &mut on, true);
    off.copy_classes_from(&fx);
    on.upload_field(&swirl_air());
    let mut a = Emitter::new(5);
    let mut b = Emitter::new(5);
    let mut scratch = ParticleEffects::with_capacity(4_096);
    for frame in 0..60 {
        let n = if frame == 0 { 1_000 } else { 20 };
        a.emit(n, &mut scratch, &mut on);
        b.emit(n, &mut scratch, &mut off);
        scratch.clear();
        on.step(DT);
        off.step(DT);
    }
    let (with, without) = (gpu_by_id(&on), gpu_by_id(&off));
    let mut compared = 0;
    for (id, (p, v)) in &without {
        if id % 3 == 2 {
            let (wp, wv) = with[id];
            assert_eq!(p.map(f32::to_bits), wp.map(f32::to_bits), "particle {id} moved");
            assert_eq!(v.map(f32::to_bits), wv.map(f32::to_bits), "particle {id} moved");
            compared += 1;
        }
    }
    assert!(compared > 100);
}

/// Ground contact on the device against `collide_ground_with` after `integrate` on the
/// CPU, over a sloped height map: the same particles land on the same frames, where
/// the CPU puts them.
#[test]
fn landings_match_the_cpu_in_count_and_position() {
    let Some(gpu) = gpu(wgpu::Features::empty()) else { return };
    let corners = [33u32, 29];
    let (min, cell) = ([-16.0f32, -14.0], 1.0f32);
    let heights: Vec<f32> = (0..corners[1])
        .flat_map(|iz| (0..corners[0]).map(move |ix| 0.12 * ix as f32 - 0.08 * iz as f32 + ((ix * 7 + iz * 3) % 5) as f32 * 0.1))
        .collect();
    // The engine's heightfield on the CPU: bilinear between corners, edges clamped.
    let at = |ix: i64, iz: i64| {
        let ix = ix.clamp(0, corners[0] as i64 - 1) as usize;
        let iz = iz.clamp(0, corners[1] as i64 - 1) as usize;
        heights[iz * corners[0] as usize + ix]
    };
    let height = |x: f32, z: f32| {
        let u = (x - min[0]) / cell;
        let v = (z - min[1]) / cell;
        let (ix, iz) = (u.floor(), v.floor());
        let (fx, fz) = (u - ix, v - iz);
        let (ix, iz) = (ix as i64, iz as i64);
        let top = at(ix, iz) * (1.0 - fx) + at(ix + 1, iz) * fx;
        let bottom = at(ix, iz + 1) * (1.0 - fx) + at(ix + 1, iz + 1) * fx;
        top * (1.0 - fz) + bottom * fz
    };
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test heights"),
        size: wgpu::Extent3d { width: corners[0], height: corners[1], depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&heights),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(corners[0] * 4), rows_per_image: None },
        wgpu::Extent3d { width: corners[0], height: corners[1], depth_or_array_layers: 1 },
    );

    let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(4_096)).unwrap();
    let mut fx = ParticleEffects::with_capacity(4_096);
    set_classes(&mut fx, &mut pool, false);
    pool.set_ground(Some(GroundHeights {
        view: texture.create_view(&Default::default()),
        min,
        cell,
        corners,
    }));

    let mut emitter = Emitter::new(99);
    let frames = 60;
    let (mut total_cpu, mut total_gpu, mut borderline) = (0usize, 0usize, 0usize);
    let mut worst = 0.0f32;
    let mut p_max = 0.0f32;
    for frame in 0..frames {
        emitter.emit(if frame == 0 { 1_500 } else { 30 }, &mut fx, &mut pool);
        fx.integrate(DT);
        let before: HashMap<u32, f32> = (0..fx.len()).map(|i| (fx.size(i) as u32, fx.position(i)[1])).collect();
        let mut cpu = HashMap::new();
        fx.collide_ground_with(height, |l| {
            cpu.insert(l.size as u32, l);
        });
        pool.step(DT);
        let device: HashMap<u32, _> = pool.read_landings_blocking().into_iter().map(|l| (l.size as u32, l)).collect();
        for i in 0..fx.len() {
            p_max = p_max.max(max_abs(fx.position(i)));
        }
        total_cpu += cpu.len();
        total_gpu += device.len();
        for (id, l) in &cpu {
            match device.get(id) {
                Some(d) => worst = worst.max(distance(l.position, d.position)),
                None => {
                    borderline += 1;
                    println!("frame {frame}: particle {id} landed on the CPU only, {:.3e} m below", l.position[1] - before[id]);
                }
            }
        }
        for id in device.keys().filter(|id| !cpu.contains_key(id)) {
            borderline += 1;
            println!("frame {frame}: particle {id} landed on the device only, {:.3e} m above", before[id] - height_of(&fx, *id, &height));
        }
    }
    // No air, so velocities agree to the bit until a bounce and positions differ only
    // by a fused `p + v dt`: `n U P`, through the bounce's scaling (each a factor at
    // most 1) unchanged. The height adds the device's division (2.5 ulp in Vulkan) and
    // a fused lerp or two: `4 U P` more.
    let bound = trajectory_bound(frames, 0.0, 0.0, p_max, 0.0) + 4.0 * U * p_max;
    println!("landings: {total_cpu} on the CPU, {total_gpu} on the device, {borderline} borderline; position {worst:.3e} m (bound {bound:.3e})");
    assert!(total_cpu > 100, "too few landings to say anything");
    assert_eq!(borderline, 0, "a landing on one side only");
    assert_eq!(total_cpu, total_gpu);
    assert!(worst <= bound, "landing position {worst} past {bound}");
}

fn height_of(fx: &ParticleEffects, id: u32, height: &impl Fn(f32, f32) -> f32) -> f32 {
    let i = (0..fx.len()).find(|&i| fx.size(i) as u32 == id).expect("particle alive");
    let p = fx.position(i);
    height(p[0], p[2])
}

/// Fill the pool, let half of it retire, refill past the free slots, and check after
/// every phase that the free stack and the live slots partition the pool: no slot both
/// free and live, none on the stack twice, no particle in two slots, and every
/// particle that was not overwritten still where it was.
#[test]
fn the_free_list_never_hands_out_a_live_slot() {
    let Some(gpu) = gpu(wgpu::Features::empty()) else { return };
    let capacity = 1_000u32;
    let mut pool = GpuParticlePool::new(&gpu, GpuPoolConfig::new(capacity)).unwrap();
    pool.set_class(0, ParticleClass { gravity: 0.0, drag: 0.0, restitution: 0.0 });

    let check = |pool: &GpuParticlePool| {
        let counts = pool.read_counts_blocking();
        let slots = pool.read_slots_blocking();
        let free = pool.read_free_blocking();
        let free_set: HashSet<u32> = free.iter().copied().collect();
        assert_eq!(free_set.len(), free.len(), "a slot on the free stack twice");
        let mut ids = HashSet::new();
        let mut live = 0;
        for (i, slot) in slots.iter().enumerate() {
            if slot.remaining > 0.0 {
                live += 1;
                assert!(!free_set.contains(&(i as u32)), "slot {i} is live and free");
                assert!(ids.insert(slot.size as u32), "particle {} in two slots", slot.size);
            }
        }
        assert!(free.iter().all(|&s| s < capacity), "a slot past the capacity");
        assert_eq!(live, counts.live);
        assert_eq!(counts.free, free.len() as u32);
        assert_eq!(counts.live + counts.free, capacity, "slots lost or made");
        assert!(counts.high_water <= capacity);
        (counts, slots)
    };

    // Fill: alternate short (0.1 s) and long (100 s) lives; each particle sits at x = id.
    for id in 1..=capacity {
        let life = if id % 2 == 0 { 0.1 } else { 100.0 };
        pool.emit_one([id as f32, 0.0, 0.0], [0.0; 3], life, id as f32, 0);
    }
    pool.step(DT);
    let (counts, _) = check(&pool);
    assert_eq!((counts.live, counts.free, counts.high_water), (capacity, 0, capacity));

    // The short half retires.
    for _ in 0..10 {
        pool.step(DT);
    }
    let (counts, _) = check(&pool);
    assert_eq!((counts.live, counts.retired), (capacity / 2, capacity / 2));

    // Refill past the free slots: the newest 500 into free slots, the 100 older ones
    // over live particles at the cursor (slots 0 to 99), except where the cursor meets
    // a slot filled this frame, where the older particle is dropped.
    for id in 1..=600 {
        pool.emit_one([0.0, 1.0, 0.0], [0.0; 3], 100.0, (10_000 + id) as f32, 0);
    }
    pool.step(DT);
    let (counts, slots) = check(&pool);
    assert_eq!((counts.live, counts.free), (capacity, 0));
    assert_eq!(counts.overwritten + counts.dropped, 100);
    // Half of slots 0 to 99 held short-lived particles and were refilled from the stack.
    assert_eq!((counts.overwritten, counts.dropped), (50, 50));
    let new: HashSet<u32> =
        slots.iter().filter(|s| s.remaining > 0.0 && s.size >= 10_000.0).map(|s| s.size as u32 - 10_000).collect();
    assert_eq!(new.len() as u32, 500 + counts.overwritten, "a new particle was placed over another new one");
    assert!((101..=600).all(|id| new.contains(&id)), "one of the newest 500 is missing");
    // Every surviving old particle is untouched.
    let old: Vec<_> = slots.iter().filter(|s| s.remaining > 0.0 && s.size < 10_000.0).collect();
    assert_eq!(old.len() as u32, 500 - counts.overwritten);
    for s in old {
        assert_eq!(s.position[0], s.size, "an old particle was corrupted");
    }

    // And again from empty: everything retires, and the pool refills from its stack.
    for _ in 0..(100.0 / DT) as usize + 2 {
        pool.step(DT);
    }
    let (counts, _) = check(&pool);
    assert_eq!(counts.live, 0);
    for id in 1..=capacity {
        pool.emit_one([0.0; 3], [0.0; 3], 1.0, (20_000 + id) as f32, 0);
    }
    pool.step(DT);
    let (counts, _) = check(&pool);
    assert_eq!((counts.live, counts.free), (capacity, 0));
}

/// More staged in a frame than the bound: the rest wait for the next frame, in order.
#[test]
fn emission_past_the_frame_bound_carries_over() {
    let Some(gpu) = gpu(wgpu::Features::empty()) else { return };
    let mut config = GpuPoolConfig::new(1_000);
    config.max_emit_per_frame = 300;
    let mut pool = GpuParticlePool::new(&gpu, config).unwrap();
    for id in 0..700 {
        pool.emit_one([0.0; 3], [0.0; 3], 10.0, id as f32, 0);
    }
    pool.step(DT);
    assert_eq!(pool.staged(), 400);
    assert_eq!(pool.read_counts_blocking().live, 300);
    pool.step(DT);
    pool.step(DT);
    assert_eq!(pool.staged(), 0);
    assert_eq!(pool.read_counts_blocking().live, 700);
}

/// The pool runs on a device it did not open, and the policy may then choose it.
#[test]
fn a_shared_device_runs_the_pool_and_the_policy_can_choose_it() {
    let Some(own) = gpu(wgpu::Features::empty()) else { return };
    let shared = GpuContext::from_device(own.device.clone(), own.queue.clone());
    assert_eq!(shared.adapter_info().name, own.adapter_info().name);
    let mut pool = GpuParticlePool::new(&shared, GpuPoolConfig::new(1_024)).unwrap();

    let mut fx = ParticleEffects::with_capacity(1_024);
    fx.emit_one([0.0, 5.0, 0.0], [1.0, 0.0, 0.0], 3.0, 1.0, 0);
    let mut policy = BackendPolicy::default();
    assert_eq!(policy.choose(50_000_000), Backend::Cpu);
    pool.register(&mut policy);
    assert_eq!(policy.choose(50_000_000), Backend::Gpu);

    pool.adopt(&mut fx);
    assert!(fx.is_empty());
    pool.step(DT);
    let slots = pool.read_slots_blocking();
    assert_eq!(slots.len(), 1);
    assert!((slots[0].remaining - (3.0 - DT)).abs() < 1e-6);
    assert!(slots[0].position[0] > 0.0);
}
