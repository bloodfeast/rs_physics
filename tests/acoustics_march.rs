//! **L2** (occlusion against a reference march) and **L3** (the main edge against a dense
//! evaluation), on three synthetic terrains with random statics and movers.
//!
//! The CPU side is f64 and knows nothing of the shader's walk: it samples the path every
//! centimetre (L3) or every 0.1 m (L2), adds every cell-boundary crossing (where the max
//! grid takes the higher of the cells that meet), and for L3 refines each cell's stretch by
//! golden-section search. The scene semantics are the ones the docs state: a column per
//! cell at its height, a static or mover an upright column over its footprint from its
//! bottom to its top, and the Fresnel rule per band at the crossing's midpoint.
#![cfg(feature = "gpu")]

mod acoustics_common;
use acoustics_common::*;

use rs_physics::acoustics::Air;
use rs_physics::gpu::acoustics::*;

const U: f64 = 1.0 / (1u64 << 24) as f64;

/// Three terrains of 140 x 100 cells at 2 m: rolling hills, sharp ridges, rough noise.
fn terrains() -> Vec<(&'static str, Scene)> {
    let mut out = Vec::new();
    let mut hills = Scene::flat(140, 100, 2.0, 0.0);
    let mut ridges = Scene::flat(140, 100, 2.0, 0.0);
    let mut rough = Scene::flat(140, 100, 2.0, 0.0);
    let mut rng = Rng(77);
    for r in 0..100usize {
        for c in 0..140usize {
            let (x, z) = (c as f64 * 2.0, r as f64 * 2.0);
            hills.heights[r * 140 + c] =
                (4.0 * (x / 37.0).sin() * (z / 23.0).cos() + 2.0 * (x / 11.0 + z / 17.0).sin()) as f32;
            let band = ((x + 0.3 * z) / 40.0).floor() as i64;
            ridges.heights[r * 140 + c] = if band % 2 == 0 { 0.0 } else { 3.0 + (band % 3) as f32 * 2.5 };
            rough.heights[r * 140 + c] = (rng.range(0.0, 1.5) + 2.0 * (x / 60.0).sin()) as f32;
        }
    }
    out.push(("hills", hills));
    out.push(("ridges", ridges));
    out.push(("rough", rough));
    out
}

fn random_box(rng: &mut Rng, scene: &Scene, big: bool) -> Obb {
    let (w, d) = if big {
        (rng.range(1.0, 14.0), rng.range(1.0, 14.0))
    } else {
        (rng.range(0.5, 4.0), rng.range(0.5, 6.0))
    };
    let h = rng.range(1.0, 9.0);
    let x = rng.range(10.0, 270.0);
    let z = rng.range(10.0, 190.0);
    let ground = scene.height_at(x, z).unwrap();
    // One in six floats, like a dropship at cruise.
    let lift = if rng.unit() < 1.0 / 6.0 { rng.range(3.0, 10.0) } else { 0.0 };
    Obb::upright(
        [x as f32, (ground + lift + h / 2.0) as f32, z as f32],
        [w as f32, h as f32, d as f32],
        rng.range(0.0, std::f64::consts::TAU) as f32,
        0,
    )
}

/// An obstacle as the CPU sees it, in f64 from the same f32 rows.
struct Col {
    inv: [[f64; 4]; 3],
    top: f64,
    bottom: f64,
    half_width: f64,
}

fn col(o: &Obb) -> Col {
    let r = o.rows.map(|row| row.map(|v| v as f64));
    let a = [[r[0][0], r[0][1], r[0][2]], [r[1][0], r[1][1], r[1][2]], [r[2][0], r[2][1], r[2][2]]];
    let cross = |p: [f64; 3], q: [f64; 3]| [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]];
    let (c0, c1, c2) = (cross(a[1], a[2]), cross(a[2], a[0]), cross(a[0], a[1]));
    let det = a[0][0] * c0[0] + a[0][1] * c0[1] + a[0][2] * c0[2];
    let t = [r[0][3], r[1][3], r[2][3]];
    let mut inv = [[0.0; 4]; 3];
    for i in 0..3 {
        let row = [c0[i] / det, c1[i] / det, c2[i] / det];
        inv[i] = [row[0], row[1], row[2], -(row[0] * t[0] + row[1] * t[1] + row[2] * t[2])];
    }
    let half_y = 0.5 * (r[1][0].abs() + r[1][1].abs() + r[1][2].abs());
    Col { inv, top: r[1][3] + half_y, bottom: r[1][3] - half_y, half_width: o.half_width as f64 }
}

fn local(inv: &[[f64; 4]; 3], p: [f64; 3], w: f64) -> [f64; 3] {
    [0, 1, 2].map(|i| inv[i][0] * p[0] + inv[i][1] * p[1] + inv[i][2] * p[2] + w * inv[i][3])
}

/// The segment's crossing of an upright footprint, in t, if any.
fn crossing(c: &Col, s: [f64; 3], d: [f64; 3]) -> Option<(f64, f64)> {
    let o = local(&c.inv, s, 1.0);
    let dl = local(&c.inv, d, 0.0);
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for k in [0, 2] {
        if dl[k].abs() < 1e-300 {
            if o[k].abs() > 0.5 {
                return None;
            }
            continue;
        }
        let (a, b) = ((-0.5 - o[k]) / dl[k], (0.5 - o[k]) / dl[k]);
        t0 = t0.max(a.min(b));
        t1 = t1.min(a.max(b));
    }
    (t0 < t1).then_some((t0, t1))
}

fn excess(s: [f64; 3], l: [f64; 3], t: f64, h: f64) -> f64 {
    let p = [s[0] + t * (l[0] - s[0]), s[1] + t * (l[1] - s[1]), s[2] + t * (l[2] - s[2])];
    let dist = |a: [f64; 3], b: [f64; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let q = [p[0], h, p[2]];
    let e = dist(q, s) + dist(l, q) - dist(l, s);
    if h > p[1] { e } else { -e }
}

/// The largest signed excess over [t0, t1] at height h: samples every `step_t`, the ends,
/// and a golden-section refinement.
fn dense_max(s: [f64; 3], l: [f64; 3], t0: f64, t1: f64, h: f64, step_t: f64) -> f64 {
    let mut best = excess(s, l, t0, h).max(excess(s, l, t1, h));
    let n = ((t1 - t0) / step_t).ceil().max(1.0) as usize;
    let mut arg = t0;
    for i in 0..=n {
        let t = t0 + (t1 - t0) * i as f64 / n as f64;
        let e = excess(s, l, t, h);
        if e > best {
            best = e;
            arg = t;
        }
    }
    // Refine around the best sample.
    let g = (5f64.sqrt() - 1.0) / 2.0;
    let (mut a, mut b) = ((arg - step_t).max(t0), (arg + step_t).min(t1));
    for _ in 0..80 {
        let (c, d) = (b - g * (b - a), a + g * (b - a));
        if excess(s, l, c, h) > excess(s, l, d, h) {
            b = d;
        } else {
            a = c;
        }
    }
    best.max(excess(s, l, 0.5 * (a + b), h))
}

struct CpuEdges {
    /// Four bands and the probe.
    edges: [f64; 5],
    /// A Fresnel admission within rounding of its threshold: counted, not asserted.
    ambiguous: bool,
    /// A footprint crossing under 0.3 m, too short for L2's 0.1 m march to be sure of.
    short: bool,
    /// The largest |d t| of any t the GPU computes (cell boundaries, footprint crossings),
    /// from the f32 inputs' magnitudes and the crossing's angle.
    dt: f64,
}

fn cpu_edges(scene: &Scene, cols: &[Col], s32: [f32; 3], l32: [f32; 3], c: f64, step_m: f64) -> CpuEdges {
    let s = s32.map(|v| v as f64);
    let l = l32.map(|v| v as f64);
    let d = [l[0] - s[0], l[1] - s[1], l[2] - s[2]];
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let dh = (d[0] * d[0] + d[2] * d[2]).sqrt().max(1e-9);
    let step_t = step_m / dh;
    let mut edges = [f64::MIN; 5];
    let mut ambiguous = false;
    let mut short = false;
    let mut dt = 8.0 * U;
    // The terrain: every cell piece, max-grid heights at the boundaries.
    let cell = scene.cell as f64;
    let (ox, oz) = (scene.origin[0] as f64, scene.origin[1] as f64);
    let (hx, hz) = (ox + scene.cols as f64 * cell, oz + scene.rows as f64 * cell);
    let (mut c0, mut c1) = (0.0f64, 1.0f64);
    for (o, dd, lo, hi) in [(s[0], d[0], ox, hx), (s[2], d[2], oz, hz)] {
        if dd.abs() < 1e-300 {
            if o < lo || o > hi {
                c1 = -1.0;
            }
            continue;
        }
        let (a, b) = ((lo - o) / dd, (hi - o) / dd);
        c0 = c0.max(a.min(b));
        c1 = c1.min(a.max(b));
    }
    if c0 < c1 {
        let mut cuts = vec![c0, c1];
        for (o, dd, origin, n, mag) in [(s[0], d[0], ox, scene.cols, hx.abs().max(ox.abs())), (s[2], d[2], oz, scene.rows, hz.abs().max(oz.abs()))] {
            if dd.abs() > 1e-300 {
                for k in 0..=n {
                    let t = (origin + k as f64 * cell - o) / dd;
                    if t > c0 && t < c1 {
                        cuts.push(t);
                    }
                }
                dt = dt.max(4.0 * U * (mag + o.abs()) / dd.abs());
            }
        }
        cuts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let h_at = |t: f64| {
            let x = s[0] + t * d[0];
            let z = s[2] + t * d[2];
            let ci = (((x - ox) / cell).floor() as i64).clamp(0, scene.cols as i64 - 1) as usize;
            let cj = (((z - oz) / cell).floor() as i64).clamp(0, scene.rows as i64 - 1) as usize;
            scene.heights[cj * scene.cols as usize + ci] as f64
        };
        for w in cuts.windows(2) {
            let (t0, t1) = (w[0], w[1]);
            if t1 <= t0 {
                continue;
            }
            let h = h_at(0.5 * (t0 + t1));
            let e = dense_max(s, l, t0, t1, h, step_t);
            for b in edges.iter_mut() {
                *b = b.max(e);
            }
        }
    }
    // Obstacles, each band admitted or not at the crossing's midpoint.
    let mut lams = [0.0; 5];
    for b in 0..4 {
        lams[b] = c / BANDS_HZ[b] as f64;
    }
    lams[4] = c / PROBE_HZ as f64;
    for cc in cols {
        let Some((ta, tb)) = crossing(cc, s, d) else { continue };
        let (ya, yb) = (s[1] + ta * d[1], s[1] + tb * d[1]);
        if ya < cc.bottom && yb < cc.bottom {
            if (ya - cc.bottom).abs() < 1e-3 || (yb - cc.bottom).abs() < 1e-3 {
                short = true;
            }
            continue;
        }
        if (tb - ta) * dh < 0.3 {
            short = true;
        }
        let dl = local(&cc.inv, d, 0.0);
        for k in [0, 2] {
            if dl[k].abs() > 1e-300 {
                let mag: f64 = cc.inv[k].iter().map(|v| v.abs()).sum::<f64>() * (1.0 + s.iter().map(|v| v.abs()).sum::<f64>());
                dt = dt.max(8.0 * U * mag / dl[k].abs());
            }
        }
        let e = dense_max(s, l, ta, tb, cc.top, step_t);
        let tm = 0.5 * (ta + tb);
        let (d1, d2) = (tm * len, len - tm * len);
        for b in 0..5 {
            let zone = lams[b] * d1 * d2 / len;
            let hw2 = cc.half_width * cc.half_width;
            if (hw2 - zone).abs() <= 1e-4 * hw2.max(zone) {
                ambiguous = true;
            }
            if hw2 >= zone {
                edges[b] = edges[b].max(e);
            }
        }
    }
    CpuEdges { edges, ambiguous, short, dt }
}

/// A march at `step` metres plus every cell boundary: blocked at the probe band, and the
/// least clearance with the local height step there.
fn cpu_blocked(scene: &Scene, cols: &[Col], s32: [f32; 3], l32: [f32; 3], c: f64) -> (bool, f64, f64) {
    let s = s32.map(|v| v as f64);
    let l = l32.map(|v| v as f64);
    let d = [l[0] - s[0], l[1] - s[1], l[2] - s[2]];
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let dh = (d[0] * d[0] + d[2] * d[2]).sqrt().max(1e-9);
    let n = (dh / 0.1).ceil() as usize;
    let mut ts: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();
    let cell = scene.cell as f64;
    for (o, dd, n) in [(s[0], d[0], scene.cols), (s[2], d[2], scene.rows)] {
        if dd.abs() > 1e-300 {
            for k in 0..=n {
                let t = (k as f64 * cell - o) / dd;
                if t > 0.0 && t < 1.0 {
                    ts.push(t);
                    ts.push((t - 1e-9).max(0.0));
                    ts.push((t + 1e-9).min(1.0));
                }
            }
        }
    }
    let (mut blocked, mut clear, mut step_at) = (false, f64::INFINITY, 0.0f64);
    let lam = c / PROBE_HZ as f64;
    for &t in &ts {
        let p = [s[0] + t * d[0], s[1] + t * d[1], s[2] + t * d[2]];
        if let Some(h) = scene.height_at(p[0], p[2]) {
            if h > p[1] {
                blocked = true;
            }
            if (p[1] - h).abs() < clear {
                clear = (p[1] - h).abs();
                let ci = (p[0] / cell).floor() as i64;
                let cj = (p[2] / cell).floor() as i64;
                let mut step: f64 = 0.0;
                for (di, dj) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                    if let Some(hn) = scene.height_at(((ci + di) as f64 + 0.5) * cell, ((cj + dj) as f64 + 0.5) * cell) {
                        step = step.max((hn - h).abs());
                    }
                }
                step_at = step;
            }
        }
        for cc in cols {
            let Some((ta, tb)) = crossing(cc, s, d) else { continue };
            if t < ta || t > tb {
                continue;
            }
            let tm = 0.5 * (ta + tb);
            let zone = lam * tm * len * (len - tm * len) / len;
            if cc.half_width * cc.half_width < zone {
                continue;
            }
            if p[1] <= cc.top && p[1] >= cc.bottom {
                blocked = true;
            }
            let m = (p[1] - cc.top).abs().min((p[1] - cc.bottom).abs());
            if m < clear {
                clear = m;
                step_at = 0.0;
            }
        }
    }
    (blocked, clear, step_at)
}

fn random_point(rng: &mut Rng, scene: &Scene) -> [f32; 3] {
    let x = rng.range(1.0, 279.0);
    let z = rng.range(1.0, 199.0);
    let h = scene.height_at(x, z).unwrap();
    [x as f32, (h + rng.range(0.3, 4.0)) as f32, z as f32]
}

#[test]
fn l2_l3_the_march_finds_the_edge_and_the_blocked_flag_agrees() {
    let Some(gpu) = gpu() else { return };
    let air = Air::standard();
    let c = air.speed_of_sound();
    let mut rng = Rng(0xacd5_0003);
    let (mut l3_pairs, mut l3_counted, mut l3_worst) = (0usize, 0usize, 0.0f64);
    let (mut l2_pairs, mut l2_counted, mut l2_blocked) = (0usize, 0usize, 0usize);
    for (name, mut scene) in terrains() {
        let mut ac = GpuAcoustics::new(&gpu.device, AcousticLimits::MAX).unwrap();
        scene.statics = (0..40).map(|_| random_box(&mut rng, &scene, true)).collect();
        scene.load(&gpu, &mut ac);
        let static_cols: Vec<Col> = scene.statics.iter().map(col).collect();
        let (mut t_name_pairs, mut t_asserted) = (0usize, 0usize);
        for _round in 0..6 {
            let listener = random_point(&mut rng, &scene);
            let movers: Vec<Obb> = (0..MAX_MOVERS).map(|_| random_box(&mut rng, &scene, false)).collect();
            let sources: Vec<Source> = (0..MAX_SOURCES)
                .map(|i| {
                    // Some sources sit inside a mover, which they then ignore.
                    let (p, ignore) = if i % 16 == 5 {
                        let m = (i % 64) as usize;
                        let r = movers[m].rows;
                        ([r[0][3], r[1][3], r[2][3]], m as u8)
                    } else {
                        (random_point(&mut rng, &scene), NO_MOVER)
                    };
                    Source::new(p, 1.0, [0.0; 3], i, ignore).unwrap()
                })
                .collect();
            let header = DispatchHeader::new(&listener_at(listener), &air, &[]).unwrap();
            let out = dispatch(&gpu, &mut ac, &header, &sources, &movers);
            let dst = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 4096,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let mut enc = gpu.encoder();
            ac.copy_source_edges(&mut enc, &dst, 0);
            gpu.submit_wait(enc);
            let edges: Vec<SourceEdges> = bytemuck::cast_slice(&gpu.read(&dst, 4096)).to_vec();
            let c32 = header.listener[3] as f64;
            for (i, src) in sources.iter().enumerate() {
                let mut cols: Vec<Col> = scene.statics.iter().map(col).collect();
                for (j, m) in movers.iter().enumerate() {
                    if j as u8 != src.ignore_mover() {
                        cols.push(col(m));
                    }
                }
                let _ = &static_cols;
                // L3: every band and the probe.
                let cpu = cpu_edges(&scene, &cols, src.position, listener, c32, 0.01);
                l3_pairs += 1;
                t_name_pairs += 1;
                if cpu.ambiguous {
                    l3_counted += 1;
                } else {
                    let s = src.position.map(|v| v as f64);
                    let l = listener.map(|v| v as f64);
                    let len = ((l[0] - s[0]).powi(2) + (l[1] - s[1]).powi(2) + (l[2] - s[2]).powi(2)).sqrt();
                    let mag = s.iter().chain(l.iter()).map(|v| v.abs()).fold(0.0, f64::max);
                    let gpu_e = [edges[i].excess_m[0], edges[i].excess_m[1], edges[i].excess_m[2], edges[i].excess_m[3], edges[i].probe_excess_m];
                    for b in 0..5 {
                        let (g, want) = (gpu_e[b] as f64, cpu.edges[b]);
                        if want == f64::MIN {
                            assert_eq!(g, f32::MIN as f64, "{name}: pair {i} band {b} saw an edge the CPU did not");
                            continue;
                        }
                        // Two lengths, each a three-term dot and a sqrt (10.5 u), a sum and
                        // a difference; the points' own rounding; and the evaluation point's
                        // t error times the excess's slope in t (at most 2 |SL|).
                        let edge_len = 2.0 * len + want.abs() + 2.0 * mag;
                        let bound = 16.0 * U * edge_len + 8.0 * U * mag + 2.0 * len * cpu.dt;
                        let diff = (g - want).abs();
                        assert!(diff <= bound, "{name}: pair {i} band {b}: gpu {g} cpu {want} diff {diff:e} bound {bound:e}");
                        l3_worst = l3_worst.max(diff / bound);
                    }
                    assert_eq!(out.sources[i].excess_m, edges[i].probe_excess_m);
                }
                // L2: the blocked flag.
                let (blocked, clear, step) = cpu_blocked(&scene, &cols, src.position, listener, c32);
                l2_pairs += 1;
                if clear <= step.max(1e-6) || cpu.ambiguous || cpu.short {
                    l2_counted += 1;
                } else {
                    assert_eq!(out.sources[i].blocked(), blocked, "{name}: pair {i} clearance {clear} step {step}");
                    t_asserted += 1;
                    l2_blocked += blocked as usize;
                }
            }
        }
        println!("{name}: {t_name_pairs} pairs, {t_asserted} asserted by L2");
        assert!(4 * t_asserted >= t_name_pairs, "{name}: under a quarter of the pairs were asserted");
    }
    println!(
        "L3: {l3_pairs} pairs x 5 bands, {l3_counted} counted (a Fresnel admission within rounding of its threshold), worst {l3_worst:.3} of the bound"
    );
    println!("L2: {l2_pairs} pairs, {l2_counted} within one cell's height step (counted), {l2_blocked} blocked among the asserted");
}

/// L3 past the static queue's capacity: 1,200 statics stacked on a few cells, so a lane's
/// cells hold more than the 512-entry queue and the overflow is tested inline. Every band
/// still matches the CPU, and field rays through the crowd still hit the nearest face.
#[test]
fn l3_a_crowded_cell_overflows_the_queue_and_loses_nothing() {
    let Some(gpu) = gpu() else { return };
    let air = Air::standard();
    let mut scene = Scene::flat(60, 60, 2.0, 0.0);
    let mut rng = Rng(0xc20d);
    // 1,200 boxes, each 2.6 m across (wider than the 4 kHz Fresnel zone here), crowded into a 6 m
    // square: every cell there lists hundreds of them.
    scene.statics = (0..1_200)
        .map(|_| {
            Obb::upright(
                [rng.range(57.0, 63.0) as f32, rng.range(0.5, 3.0) as f32, rng.range(57.0, 63.0) as f32],
                [3.0, rng.range(1.0, 6.0) as f32, 2.6],
                rng.range(0.0, 3.14) as f32,
                0,
            )
        })
        .collect();
    let mut ac = GpuAcoustics::new(&gpu.device, AcousticLimits::MAX).unwrap();
    scene.load(&gpu, &mut ac);
    let cols: Vec<Col> = scene.statics.iter().map(col).collect();
    let listener = [30.0f32, 1.6, 60.0];
    let header = DispatchHeader::new(&listener_at(listener), &air, &[]).unwrap();
    let sources: Vec<Source> = (0..32)
        .map(|i| Source::new([90.0, 0.5 + 0.2 * i as f32, 50.0 + 0.6 * i as f32], 1.0, [0.0; 3], i, NO_MOVER).unwrap())
        .collect();
    let out = dispatch(&gpu, &mut ac, &header, &sources, &[]);
    let dst = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4096,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut enc = gpu.encoder();
    ac.copy_source_edges(&mut enc, &dst, 0);
    gpu.submit_wait(enc);
    let edges: Vec<SourceEdges> = bytemuck::cast_slice(&gpu.read(&dst, 4096)).to_vec();
    let c32 = header.listener[3] as f64;
    let (mut asserted, mut counted, mut worst) = (0, 0, 0.0f64);
    for (i, src) in sources.iter().enumerate() {
        let cpu = cpu_edges(&scene, &cols, src.position, listener, c32, 0.01);
        if cpu.ambiguous {
            counted += 1;
            continue;
        }
        let s = src.position.map(|v| v as f64);
        let l = listener.map(|v| v as f64);
        let len = ((l[0] - s[0]).powi(2) + (l[1] - s[1]).powi(2) + (l[2] - s[2]).powi(2)).sqrt();
        let mag = s.iter().chain(l.iter()).map(|v| v.abs()).fold(0.0, f64::max);
        let gpu_e = [edges[i].excess_m[0], edges[i].excess_m[1], edges[i].excess_m[2], edges[i].excess_m[3], edges[i].probe_excess_m];
        for b in 0..5 {
            let (g, want) = (gpu_e[b] as f64, cpu.edges[b]);
            let bound = 16.0 * U * (2.0 * len + want.abs() + 2.0 * mag) + 8.0 * U * mag + 2.0 * len * cpu.dt;
            assert!((g - want).abs() <= bound, "crowd: pair {i} band {b}: gpu {g} cpu {want}");
            worst = worst.max((g - want).abs() / bound);
        }
        asserted += 1;
    }
    assert!(out.results().iter().any(|r| r.blocked()), "the crowd blocked nothing");
    // The queue really overflowed: the cells one path crosses list more statics than it holds.
    let per_cell = |ci: i64, cj: i64| {
        scene.statics.iter().filter(|o| {
            let r = o.rows;
            let hx = 0.5 * (r[0][0].abs() + r[0][1].abs() + r[0][2].abs());
            let hz = 0.5 * (r[2][0].abs() + r[2][1].abs() + r[2][2].abs());
            let (x0, x1) = (((r[0][3] - hx) / 2.0).floor() as i64, ((r[0][3] + hx) / 2.0).floor() as i64);
            let (z0, z1) = (((r[2][3] - hz) / 2.0).floor() as i64, ((r[2][3] + hz) / 2.0).floor() as i64);
            (x0..=x1).contains(&ci) && (z0..=z1).contains(&cj)
        }).count()
    };
    let mut most = 0usize;
    for src in &sources {
        let (s, l) = (src.position.map(|v| v as f64), listener.map(|v| v as f64));
        let mut cells = std::collections::BTreeSet::new();
        for k in 0..=20_000 {
            let t = k as f64 / 20_000.0;
            cells.insert((((s[0] + t * (l[0] - s[0])) / 2.0).floor() as i64, ((s[2] + t * (l[2] - s[2])) / 2.0).floor() as i64));
        }
        most = most.max(cells.iter().map(|&(i, j)| per_cell(i, j)).sum());
    }
    assert!(most > 512, "the busiest path lists only {most} statics: the queue did not overflow");
    assert!(asserted >= 8, "only {asserted} pairs were asserted ({counted} counted)");
    println!("L3 crowd: {asserted} pairs asserted, the busiest path listing {most} statics against a queue of 512, {counted} counted, worst {worst:.3} of the bound");
}
