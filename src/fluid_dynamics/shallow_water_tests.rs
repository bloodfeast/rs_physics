//! Tests for the shallow-water solver, each against an exact answer: a lake at rest,
//! conservation, Stoker's and Ritter's dam-break solutions, Manning's normal depth,
//! Bernoulli over a bump, symmetry, and determinism across thread counts.

use super::*;

const G: f64 = 9.81;

/// A one-cell-wide channel along `x`, walls on its long sides.
fn channel(nx: usize, dx: f64, bed: impl Fn(f64) -> f64) -> ShallowWater {
    let beds = (0..nx).map(|i| bed((i as f64 + 0.5) * dx)).collect();
    ShallowWater::new(nx, 1, dx, beds).unwrap()
}

fn cell_x(i: usize, dx: f64) -> f64 {
    (i as f64 + 0.5) * dx
}

/// A deterministic, bumpy bed with no structure a grid could line up with.
fn rough_bed(nx: usize, nz: usize) -> Vec<f64> {
    (0..nz)
        .flat_map(|j| {
            (0..nx).map(move |i| {
                let (x, z) = (i as f64, j as f64);
                0.35 * (0.37 * x).sin() * (0.23 * z).cos()
                    + 0.2 * (1.7 * x + 2.3 * z).sin() * (0.9 * x - 1.1 * z).cos()
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------------
// The four promises
// ---------------------------------------------------------------------------------

/// **Well-balanced.** Still water over uneven ground, with islands poking out of it,
/// stays still: no current, a flat surface, and dry land stays dry. Every kind of
/// boundary is present, and the `Level` one agrees with the lake.
#[test]
fn a_lake_at_rest_over_rough_ground_stays_at_rest() {
    let (nx, nz) = (48, 40);
    let bed = rough_bed(nx, nz);
    let level = 0.3;
    let mut lake = ShallowWater::new(nx, nz, 1.5, bed.clone()).unwrap();
    lake.fill_to_level(level).unwrap();
    lake.set_boundary(Edge::MinX, 0..nz, Boundary::Open).unwrap();
    lake.set_boundary(Edge::MaxX, 0..nz, Boundary::Level { surface: level }).unwrap();

    let islands = bed.iter().filter(|&&b| b > level).count();
    assert!(islands > 50, "the test needs dry islands; it has {islands} cells");

    for _ in 0..300 {
        lake.step(0.1).unwrap();
    }
    for k in 0..nx * nz {
        let (i, j) = (k % nx, k / nx);
        let [u, v] = lake.velocity(i, j);
        assert!(u.abs() < 1e-10 && v.abs() < 1e-10, "cell ({i}, {j}) moved at ({u}, {v})");
        if bed[k] >= level {
            assert_eq!(lake.depth(i, j), 0.0, "island cell ({i}, {j}) got wet");
        } else {
            let error = lake.surface(i, j) - level;
            assert!(error.abs() < 1e-10, "surface at ({i}, {j}) moved by {error:e} m");
        }
    }
}

/// **Conservative.** A hump of water sloshing in a closed basin over rough ground,
/// wetting and drying its slopes, keeps its volume to rounding, and nothing crosses
/// a wall.
#[test]
fn a_closed_basin_conserves_volume_to_rounding() {
    let (nx, nz) = (40, 36);
    let mut basin = ShallowWater::new(nx, nz, 1.0, rough_bed(nx, nz)).unwrap();
    basin.fill_to_level(0.1).unwrap();
    for j in 12..24 {
        for i in 10..22 {
            basin.add_water(i, j, 1.2).unwrap();
        }
    }
    let start = basin.total_volume();
    for _ in 0..400 {
        basin.step(0.05).unwrap();
        assert!(basin.depths().iter().all(|&h| h >= 0.0 && h.is_finite()));
    }
    let drift = (basin.total_volume() - start) / start;
    assert!(drift.abs() < 1e-12, "volume drifted by {drift:e} of itself");
    assert_eq!(basin.volume_out(), 0.0, "water left through a wall");
    assert!(basin.time() > 19.99);
}

/// Water that comes in through an inflow and leaves through an open edge is metered to
/// rounding: what is on the grid is what came in minus what went out.
#[test]
fn inflow_and_outflow_are_metered_exactly() {
    let mut river = channel(60, 2.0, |x| 0.5 - 0.004 * x);
    river.set_boundary(Edge::MinX, 0..1, Boundary::Inflow { discharge: 3.0 }).unwrap();
    river.set_boundary(Edge::MaxX, 0..1, Boundary::Open).unwrap();
    for _ in 0..3000 {
        river.step(1.0 / 30.0).unwrap();
    }
    let expected_in = 3.0 * river.time();
    assert!(
        (river.volume_in() - expected_in).abs() < 1e-9 * expected_in,
        "an inflow of 3 m3/s for {:.1} s let in {:.6} m3, not {expected_in:.6}",
        river.time(),
        river.volume_in(),
    );
    assert!(river.volume_out() > 0.0, "nothing reached the outlet");
    let balance = river.volume_in() - river.volume_out() - river.total_volume();
    assert!(balance.abs() < 1e-10 * river.volume_in(), "{balance:e} m3 unaccounted for");
}

/// **Deterministic.** The same river on one thread and on four is the same bits, and so
/// is the serial sweep that small grids use instead of the thread pool.
#[test]
fn the_result_does_not_depend_on_the_thread_count() {
    let run = |threads: usize, threshold: usize, threading: Threading| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| {
            let (nx, nz) = (37, 29);
            let mut water = ShallowWater::new(nx, nz, 1.0, rough_bed(nx, nz))
                .unwrap()
                .with_parallel_threshold(threshold)
                .with_threading(threading);
            water.fill_to_level(0.0).unwrap();
            water.set_boundary(Edge::MinX, 5..20, Boundary::Inflow { discharge: 12.0 }).unwrap();
            water.set_boundary(Edge::MaxZ, 0..nx, Boundary::Open).unwrap();
            for _ in 0..200 {
                water.step(1.0 / 60.0).unwrap();
            }
            water
        })
    };
    let one = run(1, 0, Threading::Auto);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    // The pool on four threads, the internal serial path, and the public serial switch
    // on a grid the pool would otherwise take.
    for other in [run(4, 0, Threading::Auto), run(3, usize::MAX, Threading::Auto), run(4, 0, Threading::Serial)] {
        assert_eq!(bits(one.depths()), bits(other.depths()));
        assert_eq!(bits(one.discharges_x()), bits(other.discharges_x()));
        assert_eq!(bits(one.discharges_z()), bits(other.discharges_z()));
        assert_eq!(one.volume_out().to_bits(), other.volume_out().to_bits());
        assert_eq!(one.time().to_bits(), other.time().to_bits());
    }
}

/// A symmetric setup stays symmetric: swapping `x` and `z` in the input swaps them in
/// the output. An index-order bug in either sweep breaks this at once.
#[test]
fn swapping_x_and_z_swaps_the_answer() {
    let n = 41;
    let bed: Vec<f64> = (0..n * n)
        .map(|k| {
            let (x, z) = ((k % n) as f64 - 18.0, (k / n) as f64 - 18.0);
            0.2 * (-(x * x + z * z) / 60.0).exp() + 0.01 * (x + z)
        })
        .collect();
    let mut water = ShallowWater::new(n, n, 0.5, bed).unwrap();
    water.fill_to_level(0.3).unwrap();
    for j in 0..n {
        for i in 0..n {
            let (x, z) = (i as f64 - 12.0, j as f64 - 12.0);
            if x * x + z * z < 30.0 {
                water.add_water(i, j, 0.25).unwrap();
            }
        }
    }
    water.set_boundary(Edge::MaxX, 0..n, Boundary::Open).unwrap();
    water.set_boundary(Edge::MaxZ, 0..n, Boundary::Open).unwrap();
    for _ in 0..150 {
        water.step(0.02).unwrap();
    }
    for j in 0..n {
        for i in 0..n {
            let (a, b) = (water.depth(i, j), water.depth(j, i));
            assert!((a - b).abs() < 1e-12, "depth ({i},{j}) {a} vs ({j},{i}) {b}");
            let (ua, va) = (water.velocity(i, j)[0], water.velocity(i, j)[1]);
            let (ub, vb) = (water.velocity(j, i)[0], water.velocity(j, i)[1]);
            assert!((ua - vb).abs() < 1e-10 && (va - ub).abs() < 1e-10);
        }
    }
}

// ---------------------------------------------------------------------------------
// Exact solutions
// ---------------------------------------------------------------------------------

/// Stoker's solution for a dam break onto still water: a rarefaction running back into
/// the reservoir and a bore running out over the tailwater, with a plateau between.
fn stoker(h_l: f64, h_r: f64, xi: f64) -> f64 {
    let c_l = (G * h_l).sqrt();
    let mismatch = |h_m: f64| {
        let rarefaction = 2.0 * (c_l - (G * h_m).sqrt());
        let shock = (h_m - h_r) * (G * (h_m + h_r) / (2.0 * h_m * h_r)).sqrt();
        rarefaction - shock
    };
    let (mut lo, mut hi) = (h_r, h_l);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mismatch(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let h_m = 0.5 * (lo + hi);
    let u_m = 2.0 * (c_l - (G * h_m).sqrt());
    let c_m = (G * h_m).sqrt();
    let shock = h_m * u_m / (h_m - h_r);
    if xi < -c_l {
        h_l
    } else if xi < u_m - c_m {
        (2.0 * c_l - xi).powi(2) / (9.0 * G)
    } else if xi < shock {
        h_m
    } else {
        h_r
    }
}

/// Ritter's solution for a dam break onto dry ground.
fn ritter(h0: f64, xi: f64) -> f64 {
    let c0 = (G * h0).sqrt();
    if xi < -c0 {
        h0
    } else if xi < 2.0 * c0 {
        (2.0 * c0 - xi).powi(2) / (9.0 * G)
    } else {
        0.0
    }
}

/// Run a frictionless dam break in a 100 m channel with the dam at `dam` m, stepping
/// the solver in frames of `frame` seconds, and return it at `t`.
fn dam_break(dx: f64, dam: f64, h_l: f64, h_r: f64, t: f64, frame: f64) -> ShallowWater {
    let nx = (100.0 / dx).round() as usize;
    let mut water = channel(nx, dx, |_| 0.0).with_manning(0.0).unwrap();
    for i in 0..nx {
        let depth = if cell_x(i, dx) < dam { h_l } else { h_r };
        water.add_water(i, 0, depth * dx * dx).unwrap();
    }
    let frames = (t / frame).round() as usize;
    for _ in 0..frames {
        water.step(frame).unwrap();
    }
    assert!((water.time() - t).abs() < 1e-9);
    water
}

fn l1_error(water: &ShallowWater, exact: impl Fn(f64) -> f64) -> f64 {
    let dx = water.cell_size();
    (0..water.nx())
        .map(|i| (water.depth(i, 0) - exact(cell_x(i, dx))).abs() * dx)
        .sum()
}

/// **Stoker.** A 2 m reservoir released onto 0.5 m of tailwater, after 4 s: the depth
/// profile against the exact solution, the plateau depth, the bore's position, and
/// first-order convergence as the cells halve.
#[test]
fn a_wet_dam_break_matches_stokers_solution() {
    let (h_l, h_r, dam, t) = (2.0, 0.5, 50.0, 4.0);
    let exact = |x: f64| stoker(h_l, h_r, (x - dam) / t);
    // The signal is the area between the exact profile and the tailwater.
    let signal: f64 = (0..4000).map(|k| (exact((k as f64 + 0.5) * 0.025) - h_r) * 0.025).sum();

    let coarse = l1_error(&dam_break(0.5, dam, h_l, h_r, t, 1.0 / 60.0), exact);
    let fine_water = dam_break(0.25, dam, h_l, h_r, t, 1.0 / 60.0);
    let fine = l1_error(&fine_water, exact);
    assert!(fine < 0.02 * signal, "L1 error {fine:.4} m2 is over 2% of {signal:.3} m2");
    assert!(
        fine < 0.75 * coarse,
        "halving the cells took the error from {coarse:.4} to {fine:.4}: not converging",
    );

    // The plateau between the rarefaction and the bore sits at Stoker's depth.
    let plateau = stoker(h_l, h_r, 3.0);
    let mid = fine_water.depth((62.0 / 0.25) as usize, 0);
    assert!(
        (mid - plateau).abs() < 0.01 * plateau,
        "the plateau is {mid:.4} m against Stoker's {plateau:.4}",
    );
    // And the bore is where it should be, to within two cells.
    let half = 0.5 * (plateau + h_r);
    let bore = (0..fine_water.nx())
        .rev()
        .find(|&i| fine_water.depth(i, 0) > half)
        .map(|i| cell_x(i, 0.25))
        .unwrap();
    let exact_bore = (dam..100.0)
        .step_by_f64(0.001)
        .find(|&x| exact(x) < half)
        .unwrap();
    assert!(
        (bore - exact_bore).abs() < 0.5,
        "the bore is at {bore:.2} m against {exact_bore:.2}",
    );
}

/// **Ritter.** A 1 m reservoir released onto dry ground: the profile, the front, and no
/// negative depth anywhere on the way.
#[test]
fn a_dry_dam_break_matches_ritters_solution() {
    let (h0, dam, t) = (1.0, 40.0, 5.0);
    let water = dam_break(0.25, dam, h0, 0.0, t, 1.0 / 60.0);
    let exact = |x: f64| ritter(h0, (x - dam) / t);
    let signal: f64 = (0..4000).map(|k| exact((k as f64 + 0.5) * 0.025) * 0.025).sum();
    let error = l1_error(&water, exact);
    assert!(error < 0.02 * signal, "L1 error {error:.4} m2 is over 2% of {signal:.3} m2");
    assert!(water.depths().iter().all(|&h| h >= 0.0));

    // Where the flood thins to 5% of the reservoir: 1.33 c0 t past the dam.
    let c0 = (G * h0).sqrt();
    let exact_front = dam + (2.0 - 3.0 * 0.05f64.sqrt()) * c0 * t;
    let front = (0..water.nx())
        .rev()
        .find(|&i| water.depth(i, 0) > 0.05 * h0)
        .map(|i| cell_x(i, 0.25))
        .unwrap();
    assert!(
        (front - exact_front).abs() < 0.03 * (exact_front - dam),
        "the 5% contour is at {front:.2} m against Ritter's {exact_front:.2}",
    );
}

/// Manning's normal depth: steady uniform flow down a slope `S` carrying `q` per metre
/// of width settles at `h = (q n / √S)^{3/5}`. Runs a channel of `length` metres from a
/// rough start for `seconds` and returns it with the normal depth and the outflow rate
/// per metre over the last 100 s.
fn normal_depth_run(
    slope: f64,
    q: f64,
    dx: f64,
    length: f64,
    seconds: f64,
) -> (ShallowWater, f64, f64) {
    let n = 0.03;
    let nx = (length / dx).round() as usize;
    let mut river = channel(nx, dx, |x| slope * (length - x)).with_manning(n).unwrap();
    river.set_boundary(Edge::MinX, 0..1, Boundary::Inflow { discharge: q * dx }).unwrap();
    river.set_boundary(Edge::MaxX, 0..1, Boundary::Open).unwrap();
    for i in 0..nx {
        river.add_water(i, 0, 1.0 * dx * dx).unwrap();
        river.set_velocity(i, 0, [q, 0.0]).unwrap();
    }
    for _ in 0..(seconds / 10.0) as usize {
        river.step(10.0).unwrap();
    }
    let before = river.volume_out();
    river.step(100.0).unwrap();
    let outflow = (river.volume_out() - before) / 100.0 / dx;
    (river, (q * n / slope.sqrt()).powf(0.6), outflow)
}

/// On gentle river slopes the middle of the channel sits within 1% of Manning's depth,
/// and in steady state exactly the inflow leaves through the open mouth -- no backwater
/// piles up against it.
///
/// The cells' own discharge reads up to about 1.5% below `q`: at first order the flux
/// *between* cells on a slope is `q` while the cell averages lag it by a term of order
/// `bed drop per cell × wave speed`. The outflow rate, which is the face flux, is exact.
#[test]
fn a_river_settles_at_mannings_normal_depth() {
    for slope in [0.001, 0.005] {
        let q = 2.0;
        let (river, normal, outflow) = normal_depth_run(slope, q, 10.0, 2_000.0, 6_000.0);
        let nx = river.nx();
        for i in (nx * 3 / 10)..(nx * 7 / 10) {
            let h = river.depth(i, 0);
            assert!(
                (h - normal).abs() < 0.01 * normal,
                "slope {slope}: depth {h:.4} m at cell {i} against Manning's {normal:.4}",
            );
            let flow = river.discharges_x()[i];
            assert!((flow - q).abs() < 0.015 * q, "slope {slope}: discharge {flow:.5} at cell {i}");
        }
        assert!(
            (outflow - q).abs() < 1e-3 * q,
            "slope {slope}: {outflow:.5} m2/s leaves against {q} coming in",
        );
    }
}

/// **The documented limit.** On a steep bed (2%) with 10 m cells, the ground falls half
/// the water's depth from one cell to the next, and first-order hydrostatic
/// reconstruction under-drives the flow: the river runs about 10% too deep. Finer cells
/// are the cure, and the error falls in proportion to the cell size.
#[test]
fn on_a_steep_bed_the_error_falls_with_the_cell_size() {
    let error = |dx: f64| {
        let (river, normal, _) = normal_depth_run(0.02, 1.0, dx, 500.0, 1_000.0);
        let mid = river.depth(river.nx() / 2, 0);
        (mid - normal) / normal
    };
    let (coarse, fine) = (error(10.0), error(2.5));
    assert!(coarse > 0.05 && coarse < 0.15, "10 m cells were {:.1}% deep", 100.0 * coarse);
    assert!(fine.abs() < 0.03, "2.5 m cells were {:.1}% deep", 100.0 * fine);
    assert!(fine.abs() < 0.35 * coarse, "quartering the cells took {coarse:.3} to {fine:.3}");
}

/// Steady subcritical flow over a bump (the Goutal–Maurel benchmark) on cells of `dx`
/// metres, run to steady state. Returns the worst relative depth error against the
/// exact profile, and the channel.
fn flow_over_a_bump(dx: f64) -> (f64, ShallowWater) {
    let (q, h_out) = (4.42, 2.0);
    let bump = |x: f64| (0.2 - 0.05 * (x - 10.0) * (x - 10.0)).max(0.0);
    let nx = (25.0 / dx).round() as usize;
    let mut flow = channel(nx, dx, bump).with_manning(0.0).unwrap();
    flow.set_boundary(Edge::MinX, 0..1, Boundary::Inflow { discharge: q * dx }).unwrap();
    flow.set_boundary(Edge::MaxX, 0..1, Boundary::Level { surface: h_out }).unwrap();
    flow.fill_to_level(h_out).unwrap();
    for i in 0..nx {
        flow.set_velocity(i, 0, [q / flow.depth(i, 0), 0.0]).unwrap();
    }
    for _ in 0..100 {
        flow.step(1.0).unwrap();
    }

    // Without friction the Bernoulli head q²/(2gh²) + h + b is the same everywhere, and
    // the depth is its subcritical root.
    let head = q * q / (2.0 * G * h_out * h_out) + h_out;
    let critical = (q * q / G).cbrt();
    let mut worst: f64 = 0.0;
    for i in 0..nx {
        let b = bump(cell_x(i, dx));
        let (mut lo, mut hi) = (critical, head - b);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if q * q / (2.0 * G * mid * mid) + mid > head - b {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let exact = 0.5 * (lo + hi);
        worst = worst.max(((flow.depth(i, 0) - exact) / exact).abs());
        let discharge = flow.discharges_x()[i];
        assert!((discharge - q).abs() < 0.015 * q, "discharge {discharge:.4} at cell {i}");
    }
    (worst, flow)
}

/// Bernoulli over the bump, to 1.5% on 25 cm cells and converging at first order: the
/// worst error is on the bump's lee slope and halves with the cells. What leaves the
/// channel is exactly what enters it.
#[test]
fn steady_flow_over_a_bump_conserves_bernoulli_head() {
    let (coarse, mut flow) = flow_over_a_bump(0.25);
    let (fine, _) = flow_over_a_bump(0.125);
    assert!(coarse < 0.015, "worst depth error {:.2}% on 25 cm cells", 100.0 * coarse);
    assert!(fine < 0.6 * coarse, "halving the cells took {coarse:.4} to {fine:.4}");

    let before = flow.volume_out();
    flow.step(10.0).unwrap();
    let outflow = (flow.volume_out() - before) / 10.0 / 0.25;
    assert!((outflow - 4.42).abs() < 1e-3 * 4.42, "{outflow:.5} m2/s leaves against 4.42");
}

/// The frame rate is not a physical parameter. A dam break stepped in 30 Hz frames and
/// in 144 Hz frames differs by less than a fifth of either's own error against Stoker.
#[test]
fn the_frame_rate_does_not_change_the_answer() {
    let (h_l, h_r, dam, t) = (2.0, 0.5, 50.0, 4.0);
    let exact = |x: f64| stoker(h_l, h_r, (x - dam) / t);
    let slow = dam_break(0.25, dam, h_l, h_r, t, 1.0 / 30.0);
    let fast = dam_break(0.25, dam, h_l, h_r, t, 1.0 / 144.0);
    let between: f64 = (0..slow.nx())
        .map(|i| (slow.depth(i, 0) - fast.depth(i, 0)).abs() * 0.25)
        .sum();
    let scheme = l1_error(&slow, exact).min(l1_error(&fast, exact));
    assert!(
        between < 0.2 * scheme,
        "30 Hz and 144 Hz differ by {between:.5} m2; the scheme's own error is {scheme:.5}",
    );
}

// ---------------------------------------------------------------------------------
// Game behaviour
// ---------------------------------------------------------------------------------

/// Digging a crater under a lake: the water column drops into it, the lake drains
/// into the hole, and it comes back to rest at a lower, flat level with not a drop lost.
#[test]
fn a_crater_under_a_lake_fills_and_the_lake_settles() {
    let n = 30;
    let mut lake = ShallowWater::new(n, n, 1.0, vec![0.0; n * n]).unwrap();
    lake.fill_to_level(1.0).unwrap();
    let start = lake.total_volume();
    for j in 12..18 {
        for i in 12..18 {
            lake.set_bed(i, j, -2.0).unwrap();
        }
    }
    assert!((lake.total_volume() - start).abs() < 1e-12, "moving the bed changed the volume");
    for _ in 0..1200 {
        lake.step(0.5).unwrap();
    }
    // 36 m² of crater two metres deep takes 72 m³ out of a 900 m² lake.
    let level = 1.0 - 72.0 / 900.0;
    for j in 0..n {
        for i in 0..n {
            assert!(
                (lake.surface(i, j) - level).abs() < 1e-3,
                "surface at ({i},{j}) is {:.5} m against {level:.5}",
                lake.surface(i, j),
            );
        }
    }
    assert!((lake.total_volume() - start).abs() < 1e-9 * start);
}

/// A slug of water dropped high on a steep, rough hillside runs down it as a thin sheet
/// over dry ground: the case that breaks naive schemes with negative depths and
/// runaway speeds. Nothing goes negative, nothing goes non-finite, and it is conserved.
#[test]
fn runoff_down_a_steep_dry_hillside_stays_finite_and_positive() {
    let (nx, nz) = (60, 20);
    let bed: Vec<f64> = rough_bed(nx, nz)
        .iter()
        .enumerate()
        .map(|(k, b)| b + 0.3 * (nx - k % nx) as f64)
        .collect();
    let mut hill = ShallowWater::new(nx, nz, 1.0, bed).unwrap().with_manning(0.05).unwrap();
    for j in 6..14 {
        for i in 2..8 {
            hill.add_water(i, j, 0.5).unwrap();
        }
    }
    let start = hill.total_volume();
    for _ in 0..600 {
        hill.step(1.0 / 60.0).unwrap();
        assert!(hill.depths().iter().all(|h| h.is_finite() && *h >= 0.0));
        assert!(hill.discharges_x().iter().all(|q| q.is_finite()));
    }
    assert!(((hill.total_volume() - start) / start).abs() < 1e-12);
    // And it went downhill.
    let centre: f64 = (0..nx * nz).map(|k| hill.depths()[k] * (k % nx) as f64).sum::<f64>()
        / hill.depths().iter().sum::<f64>();
    assert!(centre > 8.0, "the water's centre is still at column {centre:.1}");
}

/// Sampling a lake at rest reads its level anywhere inside it, no current, and holds
/// the edge values outside the grid.
#[test]
fn sampling_reads_the_surface_and_current() {
    let bed: Vec<f64> = (0..20 * 10).map(|k| 0.01 * (k % 20) as f64).collect();
    let mut lake = ShallowWater::new(20, 10, 2.0, bed)
        .unwrap()
        .with_origin([100.0, -50.0])
        .unwrap();
    lake.fill_to_level(0.5).unwrap();
    for (x, z) in [(113.3, -41.7), (101.0, -49.0), (138.9, -30.2), (500.0, 500.0)] {
        let s = lake.sample(x, z);
        assert!((s.surface - 0.5).abs() < 1e-12, "surface {} at ({x}, {z})", s.surface);
        assert_eq!(s.velocity, [0.0, 0.0]);
    }
    assert_eq!(lake.cell_at(101.0, -49.0), Some((0, 0)));
    assert_eq!(lake.cell_at(139.9, -30.1), Some((19, 9)));
    assert_eq!(lake.cell_at(99.9, -40.0), None);
    assert_eq!(lake.cell_at(140.0, -40.0), None);
}

/// Bad input is refused rather than turned into NaN.
#[test]
fn invalid_input_is_rejected() {
    assert!(ShallowWater::new(4, 4, 1.0, vec![0.0; 15]).is_err());
    assert!(ShallowWater::new(0, 4, 1.0, vec![]).is_err());
    assert!(ShallowWater::new(2, 2, 0.0, vec![0.0; 4]).is_err());
    assert!(ShallowWater::new(2, 2, f64::NAN, vec![0.0; 4]).is_err());
    assert!(ShallowWater::new(2, 2, 1.0, vec![0.0, f64::INFINITY, 0.0, 0.0]).is_err());

    let mut water = ShallowWater::new(4, 3, 1.0, vec![0.0; 12]).unwrap();
    assert!(water.clone().with_manning(-0.01).is_err());
    assert!(water.clone().with_gravity(0.0).is_err());
    assert!(water.step(f64::NAN).is_err());
    assert!(water.step(-1.0).is_err());
    assert_eq!(water.step(0.0).unwrap(), 0);
    assert!(water.set_boundary(Edge::MinX, 0..4, Boundary::Open).is_err(), "MinX has 3 cells");
    assert!(water.set_boundary(Edge::MinZ, 2..2, Boundary::Open).is_err());
    assert!(water
        .set_boundary(Edge::MinZ, 0..4, Boundary::Inflow { discharge: -1.0 })
        .is_err());
    assert!(water.add_water(4, 0, 1.0).is_err());
    assert!(water.add_water(0, 0, -1.0).is_err());
    assert!(water.set_bed(0, 0, f64::NAN).is_err());
    assert!(water.set_velocity(0, 0, [f64::NAN, 0.0]).is_err());
}

/// A step that would need more substeps than allowed stops with an error rather than
/// freezing the frame, and reports how far it got.
#[test]
fn an_absurd_timestep_is_bounded() {
    let mut water = ShallowWater::new(10, 10, 0.1, vec![0.0; 100])
        .unwrap()
        .with_max_substeps(50)
        .unwrap();
    water.fill_to_level(5.0).unwrap();
    assert!(water.step(60.0).is_err());
    assert!(water.time() > 0.0 && water.time() < 60.0);
}

trait StepByF64 {
    fn step_by_f64(self, step: f64) -> Box<dyn Iterator<Item = f64>>;
}

impl StepByF64 for std::ops::Range<f64> {
    fn step_by_f64(self, step: f64) -> Box<dyn Iterator<Item = f64>> {
        let n = ((self.end - self.start) / step) as usize;
        Box::new((0..n).map(move |k| self.start + k as f64 * step))
    }
}





/// A state pushed past what f64 can hold is reported as an error, not carried on as
/// NaN: the dry-bed reconstruction's `max(0)` would otherwise hide a NaN depth from the
/// wave speeds for good.
#[test]
fn a_runaway_state_is_reported() {
    let mut water = ShallowWater::new(8, 8, 1.0, vec![0.0; 64]).unwrap();
    water.fill_to_level(1.0).unwrap();
    water.set_velocity(3, 3, [1e200, 0.0]).unwrap();
    assert!(water.step(0.1).is_err());
}

/// Lowering the water with `fill_to_level` is metered as water out, not as negative
/// water in.
#[test]
fn lowering_the_level_is_metered_as_outflow() {
    let mut water = ShallowWater::new(5, 4, 2.0, vec![0.0; 20]).unwrap();
    water.fill_to_level(1.0).unwrap();
    water.fill_to_level(0.25).unwrap();
    assert_eq!(water.volume_in(), 80.0);
    assert_eq!(water.volume_out(), 60.0);
    assert_eq!(water.total_volume(), 20.0);
}

