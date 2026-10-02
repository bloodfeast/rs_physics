//! Regression tests from the particles-module correctness and performance review,
//! `docs/reviews/2026-09-29-correctness-performance.md`. Each test's doc comment
//! names the review finding (F0–F16) it reproduces; the test names match the ones
//! the review cites.
//!
//! Every oracle here is independent of the code under test: an O(n²) direct sum
//! for Barnes-Hut, a hand-written scalar reference for the SIMD paths, and
//! `Particle::update` as the zero-drag oracle for `update_with_effects`.
//!
//! Tests marked `known defect` reproduce findings this change deliberately left
//! for follow-up; they are ignored so the suite stays green, and un-ignoring one
//! is the acceptance test for its fix. Timing tests are ignored too -- run them
//! with `cargo test --release --lib --features particles review_perf -- --ignored
//! --nocapture --test-threads=1`.

use crate::particles::{
    build_tree, compute_net_force, Backend, BarnesHutNode, ParticleClass, ParticleData,
    ParticleEffects, Particle, Quad, Simulation,
};
#[cfg(target_arch = "x86_64")]
use crate::particles::{
    collect_approx_nodes, compute_force_scalar, compute_force_simd_avx,
    compute_force_simd_avx_low_precision, ApproxNode,
};
use crate::utils::PhysicsConstants;

// ── helpers ────────────────────────────────────────────────────────────────────

/// Small deterministic LCG so the tests do not depend on `rand`'s stream.
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

fn random_particles(n: usize, seed: u64) -> Vec<ParticleData> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|_| ParticleData {
            x: rng.range(-1.0, 1.0),
            y: rng.range(-1.0, 1.0),
            mass: rng.range(0.5, 2.0),
        })
        .collect()
}

/// O(n²) oracle using the library's own softening (`+1e-12` on r²), excluding
/// particle `i` by *index*, not by position. Also returns Σ|F_ij| so a comparison
/// can be normalised against the scale of the individual terms rather than the
/// (possibly cancelled) net force.
fn direct_force(ps: &[ParticleData], i: usize, g: f64) -> (f64, f64, f64) {
    let p = ps[i];
    let (mut fx, mut fy, mut scale) = (0.0, 0.0, 0.0);
    for (j, q) in ps.iter().enumerate() {
        if j == i {
            continue;
        }
        let dx = q.x - p.x;
        let dy = q.y - p.y;
        let d2 = dx * dx + dy * dy + 1e-12;
        let d = d2.sqrt();
        let f = g * p.mass * q.mass / d2;
        fx += f * dx / d;
        fy += f * dy / d;
        scale += f.abs();
    }
    (fx, fy, scale)
}

fn tree_mass(node: &BarnesHutNode) -> f64 {
    match node {
        BarnesHutNode::Empty(_) => 0.0,
        BarnesHutNode::Leaf(_, p) => p.mass,
        BarnesHutNode::Internal { mass, .. } => *mass,
    }
}

/// Sum of the masses actually stored in leaves (what a θ=0 walk can see).
fn leaf_mass(node: &BarnesHutNode) -> f64 {
    match node {
        BarnesHutNode::Empty(_) => 0.0,
        BarnesHutNode::Leaf(_, p) => p.mass,
        BarnesHutNode::Internal { nw, ne, sw, se, .. } => {
            leaf_mass(nw) + leaf_mass(ne) + leaf_mass(sw) + leaf_mass(se)
        }
    }
}

fn tree_depth(node: &BarnesHutNode) -> usize {
    match node {
        BarnesHutNode::Internal { nw, ne, sw, se, .. } => {
            1 + tree_depth(nw).max(tree_depth(ne)).max(tree_depth(sw)).max(tree_depth(se))
        }
        _ => 0,
    }
}

// ── Barnes-Hut: things that are correct (kept as regression guards) ───────────

/// θ = 0 opens every node, so the tree walk *is* a direct sum in a different order.
#[test]
fn review_bh_theta_zero_matches_direct_sum() {
    let ps = random_particles(300, 42);
    let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    let tree = build_tree(&ps, quad);
    assert!((tree_mass(&tree) - ps.iter().map(|p| p.mass).sum::<f64>()).abs() < 1e-12);

    let mut worst = 0.0f64;
    for i in 0..ps.len() {
        let (ex, ey, scale) = direct_force(&ps, i, 1.0);
        let (tx, ty) = tree.compute_force(ps[i], 0.0, 1.0);
        let (nx, ny) = compute_net_force(&tree, ps[i], 0.0, 1.0); // 299 nodes -> f64 AVX path
        for (ax, ay) in [(tx, ty), (nx, ny)] {
            let e = ((ax - ex).hypot(ay - ey)) / scale;
            worst = worst.max(e);
        }
    }
    assert!(worst < 1e-12, "theta=0 disagrees with direct sum: worst normalised error {worst:e}");
}

/// θ = 0.5 is an approximation; it should stay within a percent of the direct sum.
#[test]
fn review_bh_theta_half_is_close_to_direct_sum() {
    let ps = random_particles(2000, 7);
    let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    let tree = build_tree(&ps, quad);
    let mut errs: Vec<f64> = (0..ps.len())
        .map(|i| {
            let (ex, ey, scale) = direct_force(&ps, i, 1.0);
            let (tx, ty) = tree.compute_force(ps[i], 0.5, 1.0);
            (tx - ex).hypot(ty - ey) / scale
        })
        .collect();
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = errs[errs.len() / 2];
    let max = *errs.last().unwrap();
    println!("theta=0.5: median normalised error {median:e}, max {max:e}");
    assert!(median < 1e-2 && max < 1e-1, "median {median:e} max {max:e}");
}

/// The f64 AVX kernel against the scalar kernel at every remainder length 0..=3
/// and across several full 4-wide blocks.
#[cfg(target_arch = "x86_64")]
#[test]
fn review_simd_f64_matches_scalar_for_worklist_len_1_to_17() {
    if !std::is_x86_feature_detected!("avx") {
        return;
    }
    let mut rng = Lcg(99);
    for n in 1..=17usize {
        let wl: Vec<ApproxNode> = (0..n)
            .map(|_| ApproxNode {
                mass: rng.range(0.1, 10.0),
                com_x: rng.range(-3.0, 3.0),
                com_y: rng.range(-3.0, 3.0),
            })
            .collect();
        let p = ParticleData { x: 0.25, y: -0.4, mass: 1.7 };
        let s = compute_force_scalar(p, &wl, 1.0);
        let v = unsafe { compute_force_simd_avx(p, &wl, 1.0) };
        let scale: f64 = wl
            .iter()
            .map(|q| {
                let d2 = (q.com_x - p.x).powi(2) + (q.com_y - p.y).powi(2) + 1e-12;
                p.mass * q.mass / d2
            })
            .sum();
        let e = (s.0 - v.0).hypot(s.1 - v.1) / scale;
        assert!(e < 1e-14, "n={n}: scalar {s:?} vs avx {v:?} (normalised err {e:e})");
    }
}

/// The f32 kernel at every remainder length 0..=7 and across full 8-wide blocks,
/// at unit scale where f32 is adequate. Guards the lane/remainder handling and
/// the horizontal sum.
#[cfg(target_arch = "x86_64")]
#[test]
fn review_simd_f32_matches_scalar_for_worklist_len_1_to_17() {
    if !std::is_x86_feature_detected!("avx") {
        return;
    }
    let mut rng = Lcg(1234);
    for n in 1..=17usize {
        let wl: Vec<ApproxNode> = (0..n)
            .map(|_| ApproxNode {
                mass: rng.range(0.1, 10.0),
                com_x: rng.range(-3.0, 3.0),
                com_y: rng.range(-3.0, 3.0),
            })
            .collect();
        let p = ParticleData { x: 0.25, y: -0.4, mass: 1.7 };
        let s = compute_force_scalar(p, &wl, 1.0);
        let v = unsafe { compute_force_simd_avx_low_precision(p, &wl, 1.0) };
        let scale: f64 = wl
            .iter()
            .map(|q| {
                let d2 = (q.com_x - p.x).powi(2) + (q.com_y - p.y).powi(2) + 1e-12;
                p.mass * q.mass / d2
            })
            .sum();
        let e = (s.0 - v.0 as f64).hypot(s.1 - v.1 as f64) / scale;
        assert!(e < 1e-5, "n={n}: scalar {s:?} vs f32 avx {v:?} (normalised err {e:e})");
    }
}

// ── Barnes-Hut: fixed defects ──────────────────────────────────────────────────

/// F3. Two particles at the same position used to recurse until the child quads
/// stopped containing the point, and were then dropped; the root's aggregated
/// mass is built from the children, so the pair vanished from the tree entirely.
#[test]
fn review_bh_coincident_particles_vanish_from_build_tree() {
    let ps = [
        ParticleData { x: 0.3, y: 0.3, mass: 1.0 },
        ParticleData { x: 0.3, y: 0.3, mass: 1.0 },
        ParticleData { x: -0.5, y: -0.5, mass: 1.0 },
    ];
    let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    let tree = build_tree(&ps, quad);

    // Force on the third particle, which is 1.131 m from the pair.
    let (ex, ey, _) = direct_force(&ps, 2, 1.0);
    let (tx, ty) = tree.compute_force(ps[2], 0.5, 1.0);
    println!(
        "root mass {} (expected 3), depth {}; force ({tx:e},{ty:e}) expected ({ex:e},{ey:e})",
        tree_mass(&tree),
        tree_depth(&tree)
    );
    assert!((tree_mass(&tree) - 3.0).abs() < 1e-12, "root mass {} != 3", tree_mass(&tree));
    assert!((tx - ex).abs() < 1e-9 * ex.abs(), "fx {tx} != {ex}");
}

/// F3. The same pair through `insert`: the aggregated mass survived on the path
/// but the leaves did not, so any walk that opens the node (θ = 0 always does)
/// lost it.
#[test]
fn review_bh_coincident_particles_vanish_from_inserted_leaves() {
    let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    let mut root = BarnesHutNode::new(quad);
    let ps = [
        ParticleData { x: 0.3, y: 0.3, mass: 1.0 },
        ParticleData { x: 0.3, y: 0.3, mass: 1.0 },
        ParticleData { x: -0.5, y: -0.5, mass: 1.0 },
    ];
    for p in ps {
        root.insert(p);
    }
    let (ex, _, _) = direct_force(&ps, 2, 1.0);
    let (tx, _) = root.compute_force(ps[2], 0.0, 1.0);
    println!(
        "aggregated mass {}, mass in leaves {}, fx {tx:e} expected {ex:e}",
        tree_mass(&root),
        leaf_mass(&root)
    );
    assert!((leaf_mass(&root) - 3.0).abs() < 1e-12, "leaves hold {} of 3", leaf_mass(&root));
    assert!((tx - ex).abs() < 1e-9 * ex.abs(), "fx {tx} != {ex}");
}

/// F2. The natural root for a set of particles is their bounding square. `contains`
/// is half-open, so the particle defining the upper edge used to be silently
/// dropped -- while a *lone* out-of-range particle was kept, because the
/// `len()==1` shortcut never checks containment.
#[test]
fn review_bh_particle_on_upper_edge_of_root_is_dropped() {
    let ps = [
        ParticleData { x: -1.0, y: -1.0, mass: 1.0 },
        ParticleData { x: 1.0, y: 1.0, mass: 1.0 }, // on the upper edge
        ParticleData { x: 0.2, y: -0.3, mass: 1.0 },
    ];
    let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 }; // the bounding square
    let tree = build_tree(&ps, quad);

    let lone = build_tree(&[ParticleData { x: 5.0, y: 5.0, mass: 1.0 }], quad);
    println!(
        "bounding-square root mass {} (expected 3); lone out-of-range particle kept with mass {}",
        tree_mass(&tree),
        tree_mass(&lone)
    );
    assert!((tree_mass(&tree) - 3.0).abs() < 1e-12, "root mass {} != 3", tree_mass(&tree));
}

/// F5. With θ ≥ 1/√2 the opening test used to accept a node that *contains the
/// target particle*. Its aggregated mass includes the particle itself and its COM
/// is dragged towards it, so the particle attracted itself. Two particles are
/// enough.
#[test]
fn review_bh_node_containing_the_particle_is_accepted_self_interaction() {
    let p = ParticleData { x: -0.99, y: -0.99, mass: 1.0 };
    let q = ParticleData { x: 0.99, y: 0.99, mass: 9.0 };
    let ps = [p, q];
    let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    let tree = build_tree(&ps, quad);

    let (ex, ey, _) = direct_force(&ps, 0, 1.0);
    let exact = ex.hypot(ey);
    for theta in [0.8, 1.0] {
        let (tx, ty) = tree.compute_force(p, theta, 1.0);
        let (nx, ny) = compute_net_force(&tree, p, theta, 1.0);
        let rel_walk = (tx.hypot(ty) - exact) / exact;
        let rel_list = (nx.hypot(ny) - exact) / exact;
        println!(
            "theta={theta}: |F| walk {:.6} list {:.6} exact {exact:.6} -> rel err {rel_walk:+.3} / {rel_list:+.3}",
            tx.hypot(ty),
            nx.hypot(ny)
        );
        assert!(rel_walk.abs() < 1e-9, "theta={theta}: self-interaction error {rel_walk:+.3}");
        assert!(rel_list.abs() < 1e-9, "theta={theta}: self-interaction error {rel_list:+.3}");
    }
}

/// F1. `compute_net_force` used to switch silently to f32 once the worklist
/// exceeded 1000 entries. In SI units `m_p * m_node` overflows f32 for any pair of
/// masses whose product exceeds 3.4e38 -- e.g. two 1e20 kg asteroids -- and the
/// force came back non-finite, while the f64 tree walk is fine.
#[test]
fn review_bh_low_precision_path_is_non_finite_for_si_masses() {
    const G: f64 = 6.674_30e-11;
    // 34 x 33 grid of Earth-mass bodies 1e7 m apart; θ = 0 → 1121-entry worklist.
    let mut ps = Vec::new();
    for i in 0..34 {
        for j in 0..33 {
            ps.push(ParticleData { x: i as f64 * 1e7, y: j as f64 * 1e7, mass: 5.97e24 });
        }
    }
    let quad = Quad { cx: 1.7e8, cy: 1.7e8, half_size: 2.0e8 };
    let tree = build_tree(&ps, quad);
    let p = ps[500];
    let (tx, ty) = tree.compute_force(p, 0.0, G);
    let (nx, ny) = compute_net_force(&tree, p, 0.0, G);
    println!("f64 walk ({tx:e}, {ty:e}); compute_net_force ({nx:e}, {ny:e})");
    assert!(tx.is_finite() && ty.is_finite());
    assert!(nx.is_finite() && ny.is_finite(), "compute_net_force returned ({nx}, {ny})");
}

/// F1. The same switch at unit masses: once positions are offset from the origin
/// the f32 subtraction `com_x as f32 - p.x as f32` cancels catastrophically. A
/// cluster of 0.01 m spacing sitting at x = 1e4 m.
#[test]
fn review_bh_low_precision_path_loses_accuracy_away_from_origin() {
    let mut rng = Lcg(5);
    let ps: Vec<ParticleData> = (0..1100)
        .map(|_| ParticleData {
            x: 1.0e4 + rng.range(0.0, 0.33),
            y: 1.0e4 + rng.range(0.0, 0.33),
            mass: 1.0,
        })
        .collect();
    let quad = Quad { cx: 1.0e4 + 0.165, cy: 1.0e4 + 0.165, half_size: 0.2 };
    let tree = build_tree(&ps, quad);
    let mut worst = 0.0f64;
    for i in (0..ps.len()).step_by(37) {
        let (ex, ey, scale) = direct_force(&ps, i, 1.0);
        let (nx, ny) = compute_net_force(&tree, ps[i], 0.0, 1.0); // 1099 entries -> f32
        worst = worst.max((nx - ex).hypot(ny - ey) / scale);
    }
    println!("worst normalised error of compute_net_force at x=1e4: {worst:e}");
    assert!(worst < 1e-6, "f32 path error {worst:e}");
}

/// F11 (not fixed). The Barnes-Hut softening is a hard-coded 1e-12 m² -- an
/// absolute length of 1 µm in an SI library. At sub-micron separations the force
/// is wrong by orders of magnitude, and it cannot be changed.
#[test]
#[ignore = "known defect, not fixed in this change: see docs/reviews/2026-09-29-correctness-performance.md"]
fn review_bh_hardcoded_softening_is_not_scale_invariant() {
    let d = 1e-7; // 100 nm
    let ps = [
        ParticleData { x: 0.0, y: 0.0, mass: 1.0 },
        ParticleData { x: d, y: 0.0, mass: 1.0 },
    ];
    let tree = build_tree(&ps, Quad { cx: 0.0, cy: 0.0, half_size: 1e-6 });
    let (fx, _) = tree.compute_force(ps[0], 0.5, 1.0);
    let newton = 1.0 / (d * d);
    println!("F = {fx:e}, Newton = {newton:e}, ratio {:e}", fx / newton);
    assert!((fx / newton - 1.0).abs() < 0.01, "ratio to Newton {}", fx / newton);
}

// ── Simulation ─────────────────────────────────────────────────────────────────

/// The AVX step against a hand-written scalar reference for every n where the
/// remainder is 0..=3.
#[test]
fn review_simulation_step_matches_scalar_reference_for_n_1_to_9() {
    let c = PhysicsConstants::default();
    let dt = 0.013;
    for n in 1..=9usize {
        let mut sim = Simulation::new(n, (0.0, 0.0), 1.0, (1.0, 0.0), 1.0, c, dt).unwrap();
        let mut rng = Lcg(n as u64);
        for i in 0..n {
            sim.positions_x[i] = rng.range(-5.0, 5.0);
            sim.positions_y[i] = rng.range(-5.0, 5.0);
            sim.speeds[i] = if i == 2 { 0.0 } else { rng.range(0.0, 20.0) };
            let a = rng.range(0.0, std::f64::consts::TAU);
            sim.directions_x[i] = a.cos();
            sim.directions_y[i] = a.sin();
        }
        let (mut px, mut py, mut s, mut dx, mut dy) = (
            sim.positions_x.clone(),
            sim.positions_y.clone(),
            sim.speeds.clone(),
            sim.directions_x.clone(),
            sim.directions_y.clone(),
        );
        for _ in 0..5 {
            sim.step().unwrap();
            for i in 0..n {
                let vx = s[i] * dx[i];
                let vy = s[i] * dy[i] + c.gravity * dt;
                px[i] += vx * dt;
                py[i] += vy * dt;
                let ns = (vx * vx + vy * vy).sqrt();
                if ns != 0.0 {
                    dx[i] = vx / ns;
                    dy[i] = vy / ns;
                }
                s[i] = ns;
            }
        }
        assert_eq!(sim.positions_x, px, "n={n}");
        assert_eq!(sim.positions_y, py, "n={n}");
        assert_eq!(sim.speeds, s, "n={n}");
        assert_eq!(sim.directions_x, dx, "n={n}");
        assert_eq!(sim.directions_y, dy, "n={n}");
    }
}

/// F0. `Simulation`'s arrays are `pub`, so their lengths can disagree in safe code.
/// `step_avx` bounds its loop by `speeds.len()` and does raw `loadu/storeu` on
/// all five arrays, so it used to write past the end of the shorter one. Here the
/// write would land in the Vec's spare capacity (so the test can observe it
/// without crashing); with a shorter allocation it was a heap overflow.
#[test]
fn review_simulation_step_with_mismatched_lengths_writes_past_the_end() {
    let c = PhysicsConstants::default();
    let mut sim = Simulation::new(8, (0.0, 0.0), 10.0, (1.0, 0.0), 1.0, c, 0.5).unwrap();
    sim.positions_x.truncate(4); // len 4, capacity 8 -- entirely safe code
    assert!(sim.positions_x.capacity() >= 8);

    let result = sim.step();

    // Observe the slot *past the end* that step() should never have touched.
    // SAFETY (test only): the slot was initialised by `vec![0.0; 8]` and is within
    // the allocation; we read it only to show the library wrote to it.
    let past_end = unsafe { sim.positions_x.spare_capacity_mut()[0].assume_init() };
    println!("step() returned {result:?}; positions_x[len] (past the end) = {past_end}");
    assert!(result.is_err(), "mismatched lengths accepted");
    assert_eq!(past_end, 0.0, "step() wrote past the end of positions_x");
}

/// F0. With an empty `positions_x` the pointer is dangling, and the first AVX load
/// used to segfault and take the test process down. It must now be refused.
#[test]
fn review_simulation_step_with_empty_field_segfaults() {
    let c = PhysicsConstants::default();
    let mut sim = Simulation::new(4, (0.0, 0.0), 10.0, (1.0, 0.0), 1.0, c, 0.5).unwrap();
    sim.positions_x = Vec::new();
    assert!(sim.step().is_err(), "an empty positions_x was accepted");
}

/// F6. A particle at rest under zero drag must fall exactly as `update()` does.
/// `update_with_effects` used to rebuild the velocity from the *pre-gravity*
/// direction, so gravity's change of direction was thrown away and its speed was
/// added along whatever direction the particle had -- sideways here.
#[test]
fn review_update_with_effects_discards_gravity_direction() {
    let c = PhysicsConstants::default();
    let dt = 0.1;
    let mut a = Particle::new((0.0, 0.0), 0.0, (1.0, 0.0), 1.0).unwrap();
    let mut b = a.clone();
    for _ in 0..10 {
        a.update(dt, &c).unwrap();
        b.update_with_effects(dt, &c, Some((0.0, 1.0))).unwrap(); // zero drag coefficient
    }
    println!("update: {:?}   update_with_effects(Cd=0): {:?}", a.position, b.position);
    assert!((a.position.0 - b.position.0).abs() < 1e-12, "x: {} vs {}", a.position.0, b.position.0);
    assert!((a.position.1 - b.position.1).abs() < 1e-12, "y: {} vs {}", a.position.1, b.position.1);
}

/// F8 (not fixed: the sign is an API decision). `Simulation`/`Particle` add
/// `PhysicsConstants::gravity` (+9.80665, documented as "positive down" in
/// `world/physics_world.rs`) to +y, so a dropped particle moves to +y.
/// `ParticleEffects` in the same module subtracts it -- the same drop goes to −y --
/// and `GpuParticleSimulation` takes "negative for downward".
#[test]
#[ignore = "known defect, not fixed in this change: see docs/reviews/2026-09-29-correctness-performance.md"]
fn review_gravity_sign_disagrees_within_particles_module() {
    let c = PhysicsConstants::default();
    let mut sim = Simulation::new(1, (0.0, 0.0), 0.0, (1.0, 0.0), 1.0, c, 0.1).unwrap();
    sim.step().unwrap();

    let mut fx = ParticleEffects::with_capacity(1);
    fx.set_class(0, ParticleClass { gravity: c.gravity as f32, drag: 0.0, restitution: 0.0, swirl: 0.0 });
    fx.emit_one([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 10.0, 1.0, 0);
    fx.integrate(0.1);
    println!("Simulation y after one step: {:+e}; ParticleEffects y: {:+e}", sim.positions_y[0], fx.position(0)[1]);
    assert_eq!(
        sim.positions_y[0].signum(),
        fx.position(0)[1].signum() as f64,
        "the two particle integrators fall in opposite directions"
    );
}

// ── ParticleEffects ────────────────────────────────────────────────────────────

/// Backward swap-remove compaction retires exactly the expired particles.
#[test]
fn review_effects_retire_keeps_exactly_the_live_particles() {
    let mut fx = ParticleEffects::with_capacity(64);
    fx.set_class(0, ParticleClass { gravity: 0.0, drag: 0.0, restitution: 0.0, swirl: 0.0 });
    let dt = 0.1;
    let mut expect_alive = Vec::new();
    for k in 0..40u32 {
        // Every 3rd particle and runs of adjacent ones expire this frame.
        let life = if k % 3 == 0 || (10..14).contains(&k) || k >= 37 { 0.05 } else { 1.0 };
        fx.emit_one([0.0; 3], [0.0; 3], life, k as f32, 0);
        if life > dt {
            expect_alive.push(k);
        }
    }
    fx.integrate(dt);
    let mut alive: Vec<u32> = (0..fx.len()).map(|i| fx.size(i) as u32).collect();
    alive.sort();
    assert_eq!(alive, expect_alive);
}

/// F15. `ParticleEffects` has no GPU backend, but once a caller declares one
/// available the policy picks `Gpu` and the CPU path runs; its timing used to be
/// recorded as a GPU sample, calibrating the GPU model on CPU numbers.
#[test]
fn review_effects_record_cpu_timings_as_gpu_samples() {
    let n = 200_000;
    let mut fx = ParticleEffects::with_capacity(n);
    fx.set_class(0, ParticleClass { gravity: 9.8, drag: 0.5, restitution: 0.0, swirl: 0.0 });
    for _ in 0..n {
        fx.emit_one([0.0; 3], [1.0, 2.0, 3.0], 1e6, 1.0, 0);
    }
    fx.policy_mut().set_gpu_available(true);
    let seed = fx.policy().gpu_per_particle_ns();
    for _ in 0..8 {
        fx.integrate(1.0 / 60.0);
    }
    println!(
        "backend reported {:?}; gpu_per_particle_ns {seed} -> {}",
        fx.policy().current(),
        fx.policy().gpu_per_particle_ns()
    );
    assert!(
        !(fx.policy().current() == Backend::Gpu && fx.policy().gpu_per_particle_ns() != seed),
        "CPU work recorded as GPU samples"
    );
}

// ── Performance (run with --release -- --ignored --nocapture) ─────────────────

fn time<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    f(); // warm-up
    let t = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() / iters as f64
}

/// F9. A remainder of 1..=3 particles used to be handed to a Rayon `par_iter`,
/// which cost far more than the whole AVX body for small and medium populations
/// (n = 1027 took 10–17× as long as n = 1024).
#[test]
#[ignore = "timing; run in --release"]
fn review_perf_simulation_step_remainder_through_rayon() {
    let c = PhysicsConstants::default();
    // 65_540 has no remainder: a control separating array size from the tail.
    for n in [4usize, 5, 1024, 1027, 65_536, 65_539, 65_540] {
        let mut sim = Simulation::new(n, (0.0, 0.0), 10.0, (1.0, 0.5), 1.0, c, 1e-3).unwrap();
        let t = time(20_000, || {
            sim.step().unwrap();
            std::hint::black_box(&sim.positions_x);
        });
        println!("Simulation::step n={n:>6}: {:>9.1} ns/step", t * 1e9);
    }
}

/// F12 (not fixed). Per-particle cost of the force paths on one tree.
/// `compute_net_force` allocates a fresh worklist per call.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "timing; run in --release"]
fn review_perf_bh_force_paths() {
    if !std::is_x86_feature_detected!("avx") {
        return;
    }
    let ps = random_particles(20_000, 3);
    let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    let tree = build_tree(&ps, quad);
    let theta = 0.5;
    let ps = std::hint::black_box(ps);
    let n = ps.len() as f64;

    let t_walk = time(5, || {
        for &p in &ps {
            std::hint::black_box(tree.compute_force(p, theta, 1.0));
        }
    });
    let t_net = time(5, || {
        for &p in &ps {
            std::hint::black_box(compute_net_force(&tree, p, theta, 1.0));
        }
    });
    let mut wl = Vec::new();
    let t_reuse_simd = time(5, || {
        for &p in &ps {
            wl.clear();
            collect_approx_nodes(&tree, p, theta, &mut wl);
            std::hint::black_box(unsafe { compute_force_simd_avx(p, &wl, 1.0) });
        }
    });
    let t_reuse_scalar = time(5, || {
        for &p in &ps {
            wl.clear();
            collect_approx_nodes(&tree, p, theta, &mut wl);
            std::hint::black_box(compute_force_scalar(p, &wl, 1.0));
        }
    });
    let t_collect = time(5, || {
        for &p in &ps {
            wl.clear();
            collect_approx_nodes(&tree, p, theta, &mut wl);
            std::hint::black_box(&wl);
        }
    });
    let mut total = 0usize;
    for &p in &ps {
        wl.clear();
        collect_approx_nodes(&tree, p, theta, &mut wl);
        total += wl.len();
    }
    println!("mean worklist length {:.1}", total as f64 / n);
    println!("recursive compute_force         : {:>7.1} ns/particle", t_walk / n * 1e9);
    println!("compute_net_force (alloc + AVX) : {:>7.1} ns/particle", t_net / n * 1e9);
    println!("reused worklist + AVX           : {:>7.1} ns/particle", t_reuse_simd / n * 1e9);
    println!("reused worklist + scalar        : {:>7.1} ns/particle", t_reuse_scalar / n * 1e9);
    println!("collect_approx_nodes only       : {:>7.1} ns/particle", t_collect / n * 1e9);
}

/// F13 (not fixed). `build_tree` forks with `rayon::join` at every level down to
/// single leaves. A sequential copy of the algorithm as it stood at review time,
/// and a copy that only forks above 4096 particles, for comparison.
#[test]
#[ignore = "timing; run in --release"]
fn review_perf_build_tree_join_granularity() {
    fn build(ps: &[ParticleData], quad: Quad, cutoff: usize) -> BarnesHutNode {
        if ps.is_empty() {
            return BarnesHutNode::Empty(quad);
        }
        if ps.len() == 1 {
            return BarnesHutNode::Leaf(quad, ps[0]);
        }
        let (a, b, c, d) = quad.subdivide();
        let (mut va, mut vb, mut vc, mut vd) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for &p in ps {
            if a.contains(p.x, p.y) {
                va.push(p)
            } else if b.contains(p.x, p.y) {
                vb.push(p)
            } else if c.contains(p.x, p.y) {
                vc.push(p)
            } else if d.contains(p.x, p.y) {
                vd.push(p)
            }
        }
        let (ta, tb, tc, td) = if ps.len() > cutoff {
            let ((ta, tb), (tc, td)) = rayon::join(
                || rayon::join(|| build(&va, a, cutoff), || build(&vb, b, cutoff)),
                || rayon::join(|| build(&vc, c, cutoff), || build(&vd, d, cutoff)),
            );
            (ta, tb, tc, td)
        } else {
            (build(&va, a, cutoff), build(&vb, b, cutoff), build(&vc, c, cutoff), build(&vd, d, cutoff))
        };
        let (mut m, mut x, mut y) = (0.0, 0.0, 0.0);
        for t in [&ta, &tb, &tc, &td] {
            if let Some((tm, tx, ty)) = crate::particles::get_mass_com(t) {
                m += tm;
                x += tx * tm;
                y += ty * tm;
            }
        }
        if m > 0.0 {
            x /= m;
            y /= m;
        }
        BarnesHutNode::Internal {
            quad, mass: m, com_x: x, com_y: y,
            nw: Box::new(ta), ne: Box::new(tb), sw: Box::new(tc), se: Box::new(td),
        }
    }
    let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    for n in [1_000usize, 20_000, 200_000] {
        let ps = std::hint::black_box(random_particles(n, 11));
        let iters = (2_000_000 / n).max(3);
        let t_lib = time(iters, || { std::hint::black_box(build_tree(&ps, quad)); });
        let t_seq = time(iters, || { std::hint::black_box(build(&ps, quad, usize::MAX)); });
        let t_cut = time(iters, || { std::hint::black_box(build(&ps, quad, 4096)); });
        println!(
            "n={n:>7}: build_tree {:>8.1} us | sequential {:>8.1} us | join above 4096 {:>8.1} us",
            t_lib * 1e6, t_seq * 1e6, t_cut * 1e6
        );
    }
}

/// F3. Two coincident particles at the root centre (e.g. two bodies at the
/// origin): `build_tree` used to follow the point down the lower-left corners
/// until the half-size underflowed to 0 -- ~1075 levels of `rayon::join` -- and
/// overflow the stack in a debug build. They must share one leaf instead.
#[test]
fn review_bh_coincident_particles_at_centre_overflow_build_tree() {
    let ps = [ParticleData { x: 0.0, y: 0.0, mass: 1.0 }; 2];
    let tree = build_tree(&ps, Quad { cx: 0.0, cy: 0.0, half_size: 1.0 });
    assert_eq!(tree_depth(&tree), 0, "coincident pair was subdivided");
    assert!((leaf_mass(&tree) - 2.0).abs() < 1e-12, "leaves hold {} of 2", leaf_mass(&tree));
}

/// F3. The same pair through `insert` used to build a ~1075-deep chain whose
/// leaves held mass 0 (and overflow the stack in a debug build).
#[test]
fn review_bh_coincident_particles_at_centre_insert_depth() {
    let mut root = BarnesHutNode::new(Quad { cx: 0.0, cy: 0.0, half_size: 1.0 });
    root.insert(ParticleData { x: 0.0, y: 0.0, mass: 1.0 });
    root.insert(ParticleData { x: 0.0, y: 0.0, mass: 1.0 });
    assert_eq!(tree_depth(&root), 0, "coincident pair was subdivided");
    assert!((leaf_mass(&root) - 2.0).abs() < 1e-12, "leaves hold {} of 2", leaf_mass(&root));
}

/// F13 (not fixed). Where `build_tree`'s time goes: the same tree built by partitioning one slice
/// in place (no per-level `Vec`s, no particle copies), sequential and with a
/// 4096-particle fork cutoff. The Box-per-node layout is unchanged.
#[test]
#[ignore = "timing; run in --release"]
fn review_perf_build_tree_in_place_partition() {
    fn split<F: Fn(&ParticleData) -> bool>(s: &mut [ParticleData], pred: F) -> usize {
        let mut i = 0;
        for j in 0..s.len() {
            if pred(&s[j]) {
                s.swap(i, j);
                i += 1;
            }
        }
        i
    }
    fn build(ps: &mut [ParticleData], quad: Quad, cutoff: usize) -> BarnesHutNode {
        if ps.is_empty() {
            return BarnesHutNode::Empty(quad);
        }
        if ps.len() == 1 {
            return BarnesHutNode::Leaf(quad, ps[0]);
        }
        let (a, b, c, d) = quad.subdivide(); // NW, NE, SW, SE
        let k = split(ps, |p| p.y >= quad.cy);
        let (north, south) = ps.split_at_mut(k);
        let w = split(north, |p| p.x < quad.cx);
        let (nw, ne) = north.split_at_mut(w);
        let w = split(south, |p| p.x < quad.cx);
        let (sw, se) = south.split_at_mut(w);
        let n = nw.len() + ne.len() + sw.len() + se.len();
        let (ta, tb, tc, td) = if n > cutoff {
            let ((ta, tb), (tc, td)) = rayon::join(
                || rayon::join(|| build(nw, a, cutoff), || build(ne, b, cutoff)),
                || rayon::join(|| build(sw, c, cutoff), || build(se, d, cutoff)),
            );
            (ta, tb, tc, td)
        } else {
            (build(nw, a, cutoff), build(ne, b, cutoff), build(sw, c, cutoff), build(se, d, cutoff))
        };
        let (mut m, mut x, mut y) = (0.0, 0.0, 0.0);
        for t in [&ta, &tb, &tc, &td] {
            if let Some((tm, tx, ty)) = crate::particles::get_mass_com(t) {
                m += tm;
                x += tx * tm;
                y += ty * tm;
            }
        }
        if m > 0.0 {
            x /= m;
            y /= m;
        }
        BarnesHutNode::Internal {
            quad, mass: m, com_x: x, com_y: y,
            nw: Box::new(ta), ne: Box::new(tb), sw: Box::new(tc), se: Box::new(td),
        }
    }
    let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    for n in [1_000usize, 20_000, 200_000] {
        let ps = std::hint::black_box(random_particles(n, 11));
        // Same answer as the library at θ = 0.5 for a spot check.
        let mut scratch = ps.clone();
        let t_ref = build_tree(&ps, quad);
        let t_new = build(&mut scratch, quad, 4096);
        let (a, b) = (t_ref.compute_force(ps[3], 0.5, 1.0), t_new.compute_force(ps[3], 0.5, 1.0));
        assert!((a.0 - b.0).abs() <= 1e-12 * a.0.abs() && (a.1 - b.1).abs() <= 1e-12 * a.1.abs());

        let iters = (2_000_000 / n).max(3);
        let t_lib = time(iters, || { std::hint::black_box(build_tree(&ps, quad)); });
        let t_seq = time(iters, || {
            scratch.copy_from_slice(&ps);
            std::hint::black_box(build(&mut scratch, quad, usize::MAX));
        });
        let t_cut = time(iters, || {
            scratch.copy_from_slice(&ps);
            std::hint::black_box(build(&mut scratch, quad, 4096));
        });
        println!(
            "n={n:>7}: build_tree {:>8.1} us | in-place sequential {:>8.1} us | in-place, join above 4096 {:>8.1} us",
            t_lib * 1e6, t_seq * 1e6, t_cut * 1e6
        );
    }
}

/// Measurement only, no finding. `Simulation` stores speed + unit direction, so
/// every step pays a sqrt and two divides per particle to renormalise. The same
/// update on stored (vx, vy).
#[test]
#[ignore = "timing; run in --release"]
fn review_perf_simulation_speed_direction_representation() {
    let c = PhysicsConstants::default();
    let dt = 1e-3;
    for n in [1_024usize, 65_536, 1_048_576] {
        let mut sim = Simulation::new(n, (0.0, 0.0), 10.0, (1.0, 0.5), 1.0, c, dt).unwrap();
        let iters = (50_000_000 / n).max(10);
        let t_lib = time(iters, || {
            sim.step().unwrap();
            std::hint::black_box(&sim.positions_x);
        });
        let (mut x, mut y) = (vec![0.0f64; n], vec![0.0f64; n]);
        let (vx, mut vy) = (vec![8.94f64; n], vec![4.47f64; n]);
        let g = c.gravity;
        let t_vel = time(iters, || {
            let n = x.len();
            let (x, y, vx, vy) = (&mut x[..n], &mut y[..n], &vx[..n], &mut vy[..n]);
            for i in 0..n {
                vy[i] += g * dt;
                x[i] += vx[i] * dt;
                y[i] += vy[i] * dt;
            }
            std::hint::black_box(&*x);
        });
        println!(
            "n={n:>8}: Simulation::step {:>8.2} ns/particle | stored (vx,vy) loop {:>6.2} ns/particle",
            t_lib / n as f64 * 1e9, t_vel / n as f64 * 1e9
        );
    }
}

/// F7. Debris resting on the ground is multiplied by the 0.55 contact friction every
/// frame (and by the drag factor in `integrate`). Once |v| reaches the subnormal
/// range it never reaches zero -- `round(0.55 · m) == m` for a 1-ulp subnormal --
/// so the tangential velocity used to park on a subnormal forever.
#[test]
fn review_effects_resting_debris_velocity_parks_on_a_subnormal() {
    let mut fx = ParticleEffects::with_capacity(4);
    fx.set_class(0, ParticleClass { gravity: 26.0, drag: 1.4, restitution: 0.32, swirl: 0.0 });
    fx.emit_one([0.0, 0.0, 0.0], [5.0, 0.0, 3.0], 1e6, 1.0, 0);
    for _ in 0..2_000 {
        fx.integrate(1.0 / 60.0);
        fx.collide_ground(|_, _| 0.0);
    }
    let v = fx.velocity(0);
    println!("after 2000 frames at rest: vx = {:e} (subnormal: {}), vz = {:e}", v[0], v[0].is_subnormal(), v[2]);
    assert!(!v[0].is_subnormal() && !v[2].is_subnormal(), "velocity stuck at a subnormal: {v:?}");
}

/// F7. Long-lived particles with drag reached the same state in free flight, and
/// every arithmetic op on a subnormal operand takes a microcode assist on x86
/// (2.3 → 31 ns/particle before the fix; "aged" should now match "fresh").
#[test]
#[ignore = "timing; run in --release"]
fn review_perf_effects_subnormal_velocities() {
    let n = 10_000;
    let mut fx = ParticleEffects::with_capacity(n);
    fx.set_class(0, ParticleClass { gravity: 26.0, drag: 1.4, restitution: 0.32, swirl: 0.0 });
    fx.set_class(1, ParticleClass { gravity: 1.6, drag: 3.4, restitution: 0.0, swirl: 0.0 });
    let mut rng = crate::particles::EffectRng::new(0xC0FFEE);
    for class in [0u8, 1] {
        fx.emit(
            &crate::particles::Burst {
                origin: [0.0, 40.0, 0.0], class, count: (n / 2) as u32,
                speed: 6.0..15.0, lifetime: 10_000.0..10_001.0, size: 0.7..1.3, lift: 0.35,
            },
            &mut rng,
        );
    }
    let frame = |fx: &mut ParticleEffects| fx.integrate(std::hint::black_box(1.0 / 60.0));
    let fresh = time(300, || frame(&mut fx));
    for _ in 0..6_000 {
        frame(&mut fx);
    }
    let subnormal = (0..fx.len()).filter(|&i| fx.velocity(i)[0].is_subnormal()).count();
    let aged = time(300, || frame(&mut fx));
    println!(
        "n={n}: fresh {:.2} ns/particle | after 6000 frames {:.2} ns/particle ({subnormal} of {n} vx subnormal)",
        fresh / n as f64 * 1e9, aged / n as f64 * 1e9
    );
}

/// F7 context. Fresh-pool integrate cost (kept well short of the subnormal regime).
#[test]
#[ignore = "timing; run in --release"]
fn review_perf_effects_integrate_fresh() {
    for n in [10_000usize, 100_000, 1_000_000] {
        let mut samples = Vec::new();
        for rep in 0..5 {
            let mut fx = ParticleEffects::with_capacity(n);
            fx.set_class(0, ParticleClass { gravity: 26.0, drag: 1.4, restitution: 0.32, swirl: 0.0 });
            fx.set_class(1, ParticleClass { gravity: 1.6, drag: 3.4, restitution: 0.0, swirl: 0.0 });
            let mut rng = crate::particles::EffectRng::new(0xC0FFEE + rep);
            for class in [0u8, 1] {
                fx.emit(
                    &crate::particles::Burst {
                        origin: [0.0, 40.0, 0.0], class, count: (n / 2) as u32,
                        speed: 6.0..15.0, lifetime: 10_000.0..10_001.0, size: 0.7..1.3, lift: 0.35,
                    },
                    &mut rng,
                );
            }
            samples.push(time(100, || fx.integrate(std::hint::black_box(1.0 / 60.0))) / n as f64 * 1e9);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("n={n:>8}: integrate (fresh) median {:.2} ns/particle, min {:.2}", samples[2], samples[0]);
    }
}

/// F10. The three worklist kernels on a 2000-entry worklist. Run once with the
/// repo's `.cargo/config.toml` (+avx crate-wide) and once with `RUSTFLAGS=""`,
/// which is how the crate is compiled as a dependency. The f32 kernel used to have
/// no `#[target_feature(enable = "avx")]`, so there its intrinsics were not
/// inlined (23.9 vs 2.55 ns/node).
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "timing; run in --release, with and without RUSTFLAGS=\"\""]
fn review_perf_worklist_kernels() {
    if !std::is_x86_feature_detected!("avx") {
        return;
    }
    let mut rng = Lcg(77);
    let wl: Vec<ApproxNode> = (0..2000)
        .map(|_| ApproxNode { mass: rng.range(0.5, 2.0), com_x: rng.range(-1.0, 1.0), com_y: rng.range(-1.0, 1.0) })
        .collect();
    let wl = std::hint::black_box(wl);
    let p = std::hint::black_box(ParticleData { x: 0.01, y: 0.02, mass: 1.0 });
    let n = wl.len() as f64;
    let t_s = time(20_000, || { std::hint::black_box(compute_force_scalar(p, &wl, 1.0)); });
    let t_d = time(20_000, || { std::hint::black_box(unsafe { compute_force_simd_avx(p, &wl, 1.0) }); });
    let t_f = time(20_000, || { std::hint::black_box(unsafe { compute_force_simd_avx_low_precision(p, &wl, 1.0) }); });
    println!(
        "scalar f64 {:.2} ns/node | avx f64 {:.2} ns/node | avx f32 {:.2} ns/node",
        t_s / n * 1e9, t_d / n * 1e9, t_f / n * 1e9
    );
}

/// F14. `Simulation::new` used to validate nothing but the direction (unlike
/// `Particle::new`, which rejects mass <= 0), and `step` returned `Ok` while
/// writing NaN into every particle when `dt` was NaN.
#[test]
fn review_simulation_accepts_invalid_mass_and_dt() {
    let c = PhysicsConstants::default();
    let bad_mass = Simulation::new(4, (0.0, 0.0), 1.0, (1.0, 0.0), -1.0, c, 0.01);
    let mut nan_dt = Simulation::new(4, (0.0, 0.0), 1.0, (1.0, 0.0), 1.0, c, f64::NAN).unwrap();
    let stepped = nan_dt.step();
    println!(
        "mass=-1 accepted: {}; dt=NaN step() -> {stepped:?}, x[0] = {}",
        bad_mass.is_ok(),
        nan_dt.positions_x[0]
    );
    assert!(bad_mass.is_err(), "Simulation::new accepted mass = -1");
    assert!(stepped.is_err() || nan_dt.positions_x[0].is_finite(), "NaN dt poisoned the state and returned Ok");
}
