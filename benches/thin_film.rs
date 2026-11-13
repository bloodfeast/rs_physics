//! What the film law and the reference solver actually cost.
//!
//! Two questions, and the second one is the honest version of a claim it would be easy
//! to make without checking.
//!
//! **Does the batch entry point earn its existence?** `flux_batch` is offered as "the
//! form a solver reaches for", which implies it is faster than the obvious loop calling
//! `flux` per face. It might not be: `flux` is `#[inline]`, so a caller's own loop can
//! vectorize just as well. The one thing `flux_batch` does that the naive loop cannot is
//! **unswitch the yield-stress branch out of the loop**, which for a Newtonian fluid
//! removes a `vdivpd` from every four faces. `newtonian_batch` against
//! `newtonian_scalar_loop` is that difference and nothing else — same data, same law,
//! same answer. Whatever the number says is what goes in the docs.
//!
//! **What does a step cost per cell?** The `map` case is a game's stain buffer at its
//! real size — 1400 × 1000 texels over a 280 × 200 m map — because "it is O(n) and
//! branch-free" is not a frame budget. It is also the case that says whether visiting
//! every cell is affordable at all, or whether the caller has to track which tiles are
//! wet. The answer decides an API, so it is worth a number rather than an opinion.

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use rs_physics::fluid_dynamics::{FilmFlow, FilmGrid, Fluid, BLOOD_YIELD_STRESS};

const G: f64 = 9.81;

/// Face slopes and donor depths standing in for a wet region of a grid. Deterministic,
/// so two runs measure the same work, and it includes dry faces because a real sweep is
/// mostly dry and the zero-flux path is the one taken most often.
fn faces(count: usize) -> (Vec<f64>, Vec<f64>) {
    (0..count)
        .map(|i| {
            let t = i as f64 * 0.000_431;
            let slope = 0.25 * (t * 7.0).sin();
            let depth = if i % 5 == 0 { 0.0 } else { 0.0005 + 0.004 * (t.sin() * t.sin()) };
            (slope, depth)
        })
        .unzip()
}

/// A slope with a spill on it, at a given resolution.
fn spill(width: usize, height: usize, cell: f64) -> FilmGrid {
    let mut grid = FilmGrid::new(width, height, cell).unwrap();
    let bed: Vec<f64> = (0..width * height)
        .map(|i| {
            let (x, y) = ((i % width) as f64, (i / width) as f64);
            -0.12 * x * cell + 0.4 * (y * cell * 0.3).sin()
        })
        .collect();
    grid.set_ground_from(&bed).unwrap();
    // A percent or so of the map wet, which is what a fight leaves behind.
    for y in (height / 4)..(height / 4 + height / 16).max(height / 4 + 1) {
        for x in (width / 8)..(width / 8 + width / 16).max(width / 8 + 1) {
            grid.set_thickness(x, y, 0.003).unwrap();
        }
    }
    grid
}

/// The stain battlefield: see `step_stain_battlefield_bingham`.
fn stain_battlefield() -> FilmGrid {
    let (width, height, cell) = (1400usize, 1000usize, 0.2f64);
    let mut grid = FilmGrid::new(width, height, cell).unwrap();
    const FALL_UNIT_M: f64 = 0.0008;
    // Integrate the stain test's stored falls into a height field: the drop from `c` to
    // `c + 1` along x, and along z down the first column.
    let mut bed = vec![0.0f64; width * height];
    for y in 0..height {
        let row = y * width;
        let start = if y == 0 {
            0.0
        } else {
            bed[row - width] - (((row - width) % 23) as f64 - 11.0) * FALL_UNIT_M
        };
        bed[row] = start;
        for x in 1..width {
            let c = row + x - 1;
            bed[row + x] = bed[c] - ((c % 37) as f64 - 18.0) * FALL_UNIT_M;
        }
    }
    grid.set_ground_from(&bed).unwrap();
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f64 / (1u64 << 24) as f64
    };
    let r = (0.85 / cell).ceil() as i64;
    for &(u, v, metres, deaths) in &[(0.3, 0.4, 40.0, 300), (0.7, 0.7, 25.0, 300)] {
        let (cx, cy) = (u * 280.0, v * 200.0);
        for _ in 0..deaths {
            let (x, y) = (cx + (next() - 0.5) * metres, cy + (next() - 0.5) * metres);
            let (px, py) = ((x / cell) as i64, (y / cell) as i64);
            for dy in -r..=r {
                for dx in -r..=r {
                    let (qx, qy) = (px + dx, py + dy);
                    if qx < 0 || qy < 0 || qx >= width as i64 || qy >= height as i64 {
                        continue;
                    }
                    if ((dx * dx + dy * dy) as f64).sqrt() * cell <= 0.85 {
                        grid.set_thickness(qx as usize, qy as usize, 0.002).unwrap();
                    }
                }
            }
        }
    }
    grid
}

fn benchmark(c: &mut Criterion) {
    let newtonian = FilmFlow::new(&Fluid::blood(), G).unwrap();
    let yielding = newtonian.with_yield_stress(BLOOD_YIELD_STRESS).unwrap();

    const FACES: usize = 100_000;
    let (slopes, depths) = faces(FACES);
    let mut out = vec![0.0; FACES];

    let mut group = c.benchmark_group("thin_film_law");

    // `black_box` goes around the *slices*, once, and never around an element inside a
    // loop. A `black_box` per element is a barrier per element and stops the very
    // auto-vectorization the comparison is about — it would hand the batch form a win it
    // had not earned. The first draft of this file did exactly that and reported 7.7x.
    group.bench_function("newtonian_batch", |b| {
        b.iter(|| {
            newtonian
                .flux_batch(black_box(&slopes), black_box(&depths), black_box(&mut out))
                .unwrap();
            black_box(out[0])
        })
    });

    // The loop a caller writes if `flux_batch` does not exist. Identical arithmetic; the
    // only structural difference is that the yield branch is inside it rather than
    // hoisted, so this is exactly what the unswitch is worth.
    group.bench_function("newtonian_scalar_loop", |b| {
        b.iter(|| {
            let (s, d, o) = (black_box(&slopes), black_box(&depths), black_box(&mut out));
            for i in 0..FACES {
                o[i] = newtonian.flux(s[i], d[i]);
            }
            black_box(o[0])
        })
    });

    // The yield-stress path, which cannot avoid the divide and so is the honest ceiling.
    group.bench_function("bingham_batch", |b| {
        b.iter(|| {
            yielding
                .flux_batch(black_box(&slopes), black_box(&depths), black_box(&mut out))
                .unwrap();
            black_box(out[0])
        })
    });

    // And the same loop for a fluid that really does yield, where there is nothing to
    // hoist. If the unswitch is the whole story, this and `bingham_batch` should be a
    // wash — which is the prediction, and worth checking rather than asserting.
    group.bench_function("bingham_scalar_loop", |b| {
        b.iter(|| {
            let (s, d, o) = (black_box(&slopes), black_box(&depths), black_box(&mut out));
            for i in 0..FACES {
                o[i] = yielding.flux(s[i], d[i]);
            }
            black_box(o[0])
        })
    });

    group.finish();

    let mut group = c.benchmark_group("thin_film_step");

    let mut small = spill(128, 128, 0.05);
    let dt = small.max_step(&newtonian) * 0.25;
    group.bench_function("step_128x128", |b| {
        b.iter(|| small.step(black_box(&newtonian), black_box(dt)))
    });

    // A game's stain buffer at its real size: 5 texels/m over a 280 x 200 m map.
    let mut map = spill(1400, 1000, 0.2);
    let map_dt = map.max_step(&newtonian) * 0.25;
    group.bench_function("step_1400x1000_map", |b| {
        b.iter(|| map.step(black_box(&newtonian), black_box(map_dt)))
    });

    // The whole-grid CFL scan, which a caller pays only if it does not already know its
    // deepest cell.
    group.bench_function("max_step_1400x1000_map", |b| {
        b.iter(|| black_box(map.max_step(black_box(&newtonian))))
    });

    // STAIN-MOVE step 1: the same map and workload Ridgeline's stain settle is measured on
    // (`terrain_stain::heave::measure_stain_phases`): 1400 x 1000 at 0.2 m, rolling ground
    // built from the stain test's fall pattern (`cell % 37 - 18` and `cell % 23 - 11` units
    // of 0.8 mm a texel), and six hundred bodies dropped by the same xorshift into two
    // fights, each a 0.85 m disc of standing blood 2 mm deep. The dense step visits every
    // cell, wet or not, so this is what the reference solver costs on that map; the stain
    // advances at 60 Hz, so the per-settle figure is this times the substeps the stable
    // step needs inside 1/60 s (printed once below).
    let mut stain = stain_battlefield();
    let stain_dt = stain.max_step(&yielding);
    let substeps = (1.0 / 60.0 / stain_dt).ceil().max(1.0);
    println!("stain battlefield: stable step {stain_dt:.3e} s, {substeps} substeps per 1/60 s");
    group.bench_function("step_stain_battlefield_bingham", |b| {
        b.iter(|| stain.step(black_box(&yielding), black_box(stain_dt)))
    });

    group.finish();

    // Does `step` get slower per cell as the grid outgrows cache?
    //
    // This is the measurement that decides whether hand-written SIMD is worth writing.
    // `FilmGrid` holds five `f64` arrays (`ground`, `thickness`, `flux_x`, `flux_y`,
    // `work`), so the working set is ~40 bytes a cell and these sizes walk it from
    // comfortably-in-L2 to several times L3:
    //
    //   64 x 64   =>  164 KB      512 x 512   =>  10.5 MB
    //   128 x 128 =>  655 KB      1024 x 1024 =>  42 MB
    //   256 x 256 =>  2.6 MB
    //
    // The arithmetic per cell is identical at every size. So if ns/cell is flat, the
    // loop is compute-bound and wider vectors would buy something; if it climbs once
    // the working set passes L3, the large grid is waiting on memory and no amount of
    // SIMD moves it. Throughput is set to Elements so Criterion reports the per-cell
    // figure directly rather than leaving it to be divided out by hand.
    let mut group = c.benchmark_group("thin_film_scaling");

    for &(w, h) in &[(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)] {
        let cells = (w * h) as u64;
        let mut grid = spill(w, h, 0.05);
        let step_dt = grid.max_step(&newtonian) * 0.25;
        group.throughput(Throughput::Elements(cells));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{w}x{h}")),
            &step_dt,
            |b, &d| b.iter(|| grid.step(black_box(&newtonian), black_box(d))),
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
