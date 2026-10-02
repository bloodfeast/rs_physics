//! Regression tests for `SphFluid`, from the 2026-09-29 correctness and performance
//! review: see `docs/reviews/2026-09-29-correctness-performance.md`.
//!
//! A child module of `sph` rather than an integration test, so it can reach the
//! private passes (`build_grid`, the recorded neighbour lists, ...) and scratch fields and
//! check each against an independent oracle: a kernel integral, brute-force
//! neighbour search, conservation of momentum. Tests marked `#[ignore]` with a
//! "known defect" reason reproduce findings deliberately left for follow-up work.

use super::*;

fn flat(_x: f64, _z: f64) -> f64 {
    0.0
}

/// Deterministic xorshift in [-0.5, 0.5).
fn rng(seed: u32) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s >> 8) as f64 / ((1u32 << 24) as f64) - 0.5
    }
}

fn total_momentum(f: &SphFluid) -> [f64; 3] {
    let m = f.params.particle_mass;
    let mut p = [0.0; 3];
    for i in 0..f.len() {
        let v = f.velocity(i);
        for a in 0..3 {
            p[a] += m * v[a];
        }
    }
    p
}

/// Particle `i`'s neighbours (itself excluded) as the density pass recorded them for
/// the force pass, as particle indices, in walk order.
fn cached_neighbours(f: &SphFluid, i: usize) -> Vec<usize> {
    let k = f.slot_of[i] as usize;
    let (c, local) = (k / SPH_CHUNK, k % SPH_CHUNK);
    let list = &f.lists[c];
    let begin = if local == 0 { 0 } else { list.end[local - 1] as usize };
    let end = list.end[local] as usize;
    list.index[begin..end]
        .iter()
        .map(|&s| f.order[s as usize] as usize)
        .collect()
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

// ── Kernels ──────────────────────────────────────────────────────────────────

/// The density kernel, *as the solver evaluates it*, integrates to one over its
/// support. W(r) is recovered from the solver itself: the density a particle at
/// distance r adds to a lone particle, divided by the particle mass.
#[test]
fn poly6_as_evaluated_integrates_to_one() {
    let params = SphParams::water();
    let h = params.smoothing_radius;
    let m = params.particle_mass;

    let lone_density = || {
        let mut lone = SphFluid::new(params, 1).unwrap();
        lone.spawn([0.0; 3], [0.0; 3]);
        lone.build_grid();
        lone.compute_density_and_pressure();
        lone.density[0]
    };
    let self_only = lone_density();
    let w_of = |r: f64| -> f64 {
        let mut pair = SphFluid::new(params, 2).unwrap();
        pair.spawn([0.0; 3], [0.0; 3]);
        pair.spawn([r, 0.0, 0.0], [0.0; 3]);
        pair.build_grid();
        pair.compute_density_and_pressure();
        (pair.density[0] - self_only) / m
    };

    // Composite Simpson on 4 pi r^2 W(r), r in [0, h]. At r = 0, W is the
    // self-contribution over m.
    let n = 400;
    let dr = h / n as f64;
    let mut integral = 0.0;
    for k in 0..=n {
        let r = k as f64 * dr;
        let w = if k == 0 { self_only / m } else { w_of(r) };
        let coef = if k == 0 || k == n {
            1.0
        } else if k % 2 == 1 {
            4.0
        } else {
            2.0
        };
        integral += coef * 4.0 * core::f64::consts::PI * r * r * w;
    }
    integral *= dr / 3.0;
    assert!(
        (integral - 1.0).abs() < 1e-6,
        "poly6 integrates to {integral}, not 1"
    );
}

/// Spiky gradient and viscosity Laplacian constants, checked against the kernels
/// they are derived from (spiky W = 15/(pi h^6) (h-r)^3, normalised; viscosity W
/// of Mueller 2003, normalised), by numerical integration and differentiation.
#[test]
fn spiky_and_viscosity_kernels_are_normalised_and_differentiated() {
    let h = 0.04_f64;
    let pi = core::f64::consts::PI;
    let spiky = |r: f64| 15.0 / (pi * h.powi(6)) * (h - r).powi(3);
    let visc = |r: f64| {
        15.0 / (2.0 * pi * h.powi(3))
            * (-(r * r * r) / (2.0 * h * h * h) + r * r / (h * h) + h / (2.0 * r) - 1.0)
    };
    let simpson = |f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize| {
        let dr = (b - a) / n as f64;
        let mut s = 0.0;
        for k in 0..=n {
            let r = a + k as f64 * dr;
            let c = if k == 0 || k == n {
                1.0
            } else if k % 2 == 1 {
                4.0
            } else {
                2.0
            };
            s += c * f(r);
        }
        s * dr / 3.0
    };
    let int_spiky = simpson(&|r| 4.0 * pi * r * r * spiky(r), 0.0, h, 2000);
    let int_visc = simpson(&|r| 4.0 * pi * r * r * visc(r), 1e-12, h, 20000);
    assert!(
        (int_spiky - 1.0).abs() < 1e-9,
        "spiky integrates to {int_spiky}"
    );
    assert!(
        (int_visc - 1.0).abs() < 1e-4,
        "viscosity integrates to {int_visc}"
    );

    // The solver's constants (apply_forces): spiky_grad = -45/(pi h^6), applied as
    // spiky_grad * (h-r)^2; visc_lap = 45/(pi h^6), applied as visc_lap * (h-r).
    let spiky_grad = -45.0 / (pi * h.powi(6));
    let visc_lap = 45.0 / (pi * h.powi(6));
    for &r in &[0.1 * h, 0.37 * h, 0.5 * h, 0.81 * h] {
        let e = 1e-7 * h;
        let dwdr = (spiky(r + e) - spiky(r - e)) / (2.0 * e);
        let code = spiky_grad * (h - r) * (h - r);
        assert!((dwdr - code).abs() / code.abs() < 1e-6);
        // Laplacian of a radial function: W'' + 2 W' / r. Second differences need
        // a coarser step than first differences.
        let e2 = 1e-5 * h;
        let d1 = (visc(r + e) - visc(r - e)) / (2.0 * e);
        let d2 = (visc(r + e2) - 2.0 * visc(r) + visc(r - e2)) / (e2 * e2);
        let lap = d2 + 2.0 * d1 / r;
        let code = visc_lap * (h - r);
        assert!(
            (lap - code).abs() / code.abs() < 1e-4,
            "lap {lap} vs {code}"
        );
    }
}

/// A repulsive pressure force pushes a compressed pair *apart*.
#[test]
fn pressure_pushes_a_compressed_pair_apart() {
    let mut p = SphParams::water();
    p.viscosity = 0.0;
    p.cohesion = 0.0;
    let s = p.smoothing_radius * 0.1;
    let mut f = SphFluid::new(p, 2).unwrap();
    f.spawn([0.0, 10.0, 0.0], [0.0; 3]);
    f.spawn([s, 10.0, 0.0], [0.0; 3]);
    f.build_grid();
    f.compute_density_and_pressure();
    // Both set, so which sorted slot each particle took does not matter.
    f.s_pressure[0] = 1000.0;
    f.s_pressure[1] = 1000.0;
    f.apply_forces(0.0);
    let (a0, a1) = (f.ax[f.slot_of[0] as usize], f.ax[f.slot_of[1] as usize]);
    assert!(a0 < 0.0 && a1 > 0.0, "accelerations {a0} and {a1}");
}

// ── Momentum ─────────────────────────────────────────────────────────────────

/// An asymmetric lump of fluid in zero gravity, far from the ground and slow enough
/// never to touch the speed cap. Returns (|total momentum| after, sum of m|v| after,
/// peak speed, speed ceiling).
fn free_lump(cohesion: f64, steps: usize) -> (f64, f64, f64, f64) {
    let mut p = SphParams::blood();
    p.cohesion = cohesion;
    let s = p.smoothing_radius * 0.5;
    let dt = 1.0 / 240.0;
    let mut f = SphFluid::new(p, 4096).unwrap();
    // An L-shaped block: asymmetric, so surface and interior densities differ in a
    // way that does not cancel by symmetry.
    for x in 0..8 {
        for y in 0..8 {
            for z in 0..4 {
                if x >= 4 && y >= 4 {
                    continue;
                }
                f.spawn([x as f64 * s, 50.0 + y as f64 * s, z as f64 * s], [0.0; 3]);
            }
        }
    }
    let ceiling = f.speed_ceiling(dt);
    let mut peak: f64 = 0.0;
    for _ in 0..steps {
        f.step(dt, 0.0, flat);
        for i in 0..f.len() {
            peak = peak.max(norm(f.velocity(i)));
        }
    }
    let m = f.params.particle_mass;
    let scale: f64 = (0..f.len()).map(|i| m * norm(f.velocity(i))).sum();
    (norm(total_momentum(&f)), scale, peak, ceiling)
}

/// Pressure + viscosity alone conserve linear momentum to rounding.
#[test]
fn pressure_and_viscosity_conserve_momentum() {
    let (p, scale, peak, ceiling) = free_lump(0.0, 240);
    assert!(
        peak < ceiling * 0.9,
        "hit the cap ({peak} vs {ceiling}); test invalid"
    );
    assert!(scale > 0.0);
    assert!(
        p <= 1e-9 * scale,
        "net momentum {p:.3e} against sum m|v| {scale:.3e}"
    );
}

/// Cohesion is equal and opposite between a pair. It used to divide particle i's
/// acceleration by rho_i and j's by rho_j, so any pair with different densities (a
/// surface particle next to an interior one) put a net force on itself and a free
/// lump of fluid self-propelled: 1.8e-4 kg m/s from rest in one second here.
#[test]
fn cohesion_conserves_momentum() {
    let (p, scale, peak, ceiling) = free_lump(SphParams::blood().cohesion, 240);
    assert!(
        peak < ceiling * 0.9,
        "hit the cap ({peak} vs {ceiling}); test invalid"
    );
    assert!(
        p <= 1e-9 * scale,
        "a free lump acquired net momentum {p:.3e} kg m/s from internal forces \
         alone; sum m|v| = {scale:.3e}"
    );
}

// ── Neighbour search ─────────────────────────────────────────────────────────

/// The hashed grid finds exactly the brute-force neighbour set, on clouds that
/// straddle zero on every axis, sit wholly in negative space, and put particles
/// exactly on cell faces. Covers the out-of-reach cell skip in the walk, and the
/// neighbour list the density pass caches for the force pass.
#[test]
fn neighbour_search_matches_brute_force_including_negative_coords() {
    let params = SphParams::blood();
    let h = params.smoothing_radius;
    for (seed, centre, spread) in [
        (1u32, [0.0, 0.0, 0.0], 0.3),
        (2, [-1.0, -1.0, -1.0], 0.2),
        (3, [-0.01, 0.01, -0.02], 0.1),
        (4, [123.4, -56.7, -0.5], 0.25),
        (5, [0.0, 0.0, 0.0], 60.0), // sparse, many empty cells, collisions
    ] {
        let mut r = rng(seed);
        let mut f = SphFluid::new(params, 2048).unwrap();
        for k in 0..1500 {
            let mut p = [
                centre[0] + r() * spread,
                centre[1] + r() * spread,
                centre[2] + r() * spread,
            ];
            // Every tenth particle snapped exactly onto a cell face.
            if k % 10 == 0 {
                p[0] = (p[0] / h).round() * h;
            }
            f.spawn(p, [0.0; 3]);
        }
        f.build_grid();
        f.compute_density_and_pressure();
        for i in 0..f.len() {
            let mut got = cached_neighbours(&f, i);
            got.sort_unstable();
            let dup = got.windows(2).any(|w| w[0] == w[1]);
            assert!(!dup, "seed {seed}: particle {i} visited a neighbour twice");
            let mut want = Vec::new();
            let a = f.position(i);
            for j in 0..f.len() {
                let b = f.position(j);
                let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
                if j != i && d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= h * h {
                    want.push(j);
                }
            }
            assert_eq!(got, want, "seed {seed}: cached neighbours of {i} at {a:?}");
        }
    }
}

/// `cell_of` floors rather than truncating toward zero.
#[test]
fn cell_coord_floors_negative_coordinates() {
    let c = |p: [f64; 3]| [cell_of(p[0], 1.0), cell_of(p[1], 1.0), cell_of(p[2], 1.0)];
    assert_eq!(c([-0.5, 0.5, -1e-12]), [-1, 0, -1]);
    assert_eq!(c([-1.0, 1.0, 0.0]), [-1, 1, 0]);
}

/// Positions past i32::MAX cells used to saturate in `cell_of`, and the 27-cell
/// walk then computed `base + 1`: a panic in debug builds ("attempt to add with
/// overflow") and a wrap to the far side of the map in release. They now clamp one
/// short of the range, and two such particles still find each other.
#[test]
fn far_particles_do_not_overflow_the_neighbour_walk() {
    let mut f = SphFluid::new(SphParams::blood(), 4).unwrap();
    // 1e9 m / 0.04 m = 2.5e10 cells, past i32::MAX.
    f.spawn([1e9, 10.0, 0.0], [0.0; 3]);
    f.spawn([1e9 + 0.01, 10.0, 0.0], [0.0; 3]);
    f.spawn([-1e9, 10.0, 0.0], [0.0; 3]);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        f.build_grid();
        f.compute_density_and_pressure();
    }));
    assert!(r.is_ok(), "the neighbour walk panicked at x = 1e9 m");
    assert_eq!(cached_neighbours(&f, 0), vec![1]);
    assert_eq!(cached_neighbours(&f, 2), Vec::<usize>::new());
    f.step(1.0 / 240.0, 9.81, flat);
}

// ── Validation / NaN ─────────────────────────────────────────────────────────

/// `SphFluid::new` used to test `<= 0.0`, which NaN passes, and checked three of
/// the eight fields. A NaN turned every particle NaN on the first step.
#[test]
fn new_rejects_nan_and_out_of_range_parameters() {
    let bad: [fn(&mut SphParams); 12] = [
        |p| p.smoothing_radius = f64::NAN,
        |p| p.particle_mass = f64::NAN,
        |p| p.rest_density = f64::NAN,
        |p| p.smoothing_radius = f64::INFINITY,
        |p| p.stiffness = -1.0,
        |p| p.stiffness = f64::NAN,
        |p| p.viscosity = -0.1,
        |p| p.cohesion = f64::NAN,
        |p| p.restitution = 1.5,
        |p| p.restitution = -0.1,
        |p| p.friction = 1.01,
        |p| p.friction = f64::NAN,
    ];
    for (k, spoil) in bad.iter().enumerate() {
        let mut p = SphParams::water();
        spoil(&mut p);
        assert!(
            SphFluid::new(p, 16).is_err(),
            "case {k} was accepted: {p:?}"
        );
    }
    for p in [SphParams::water(), SphParams::blood(), SphParams::napalm()] {
        assert!(SphFluid::new(p, 16).is_ok(), "a preset was rejected: {p:?}");
    }
    // The ends of the fraction ranges are legal.
    let mut p = SphParams::water();
    p.friction = 0.0;
    p.restitution = 1.0;
    assert!(SphFluid::new(p, 16).is_ok());
}

/// One particle spawned with a non-finite velocity used to turn its neighbours NaN
/// within one step, through the viscosity term (vel[j] - vel[i]). `spawn` now
/// refuses it.
#[test]
fn spawn_rejects_non_finite_state() {
    let mut f = SphFluid::new(SphParams::blood(), 64).unwrap();
    let s = f.params.smoothing_radius * 0.5;
    for k in 0..8 {
        f.spawn([k as f64 * s, 1.0, 0.0], [0.0; 3]);
    }
    assert!(!f.spawn([3.5 * s, 1.0, 0.0], [f64::NAN, 0.0, 0.0]));
    assert!(!f.spawn([3.5 * s, 1.0, 0.0], [0.0, f64::INFINITY, 0.0]));
    assert!(!f.spawn([f64::NAN, 1.0, 0.0], [0.0; 3]));
    assert_eq!(f.len(), 8);
    f.step(1.0 / 240.0, 9.81, flat);
    for i in 0..f.len() {
        assert!(f.velocity(i).iter().all(|c| c.is_finite()));
        assert!(f.position(i).iter().all(|c| c.is_finite()));
    }
}

/// `step` used to guard only `dt <= 0.0`, which NaN and +inf pass: every particle
/// went NaN (and a debug build panicked on the `debug_assert!` in `integrate`).
/// A non-finite `dt` or `gravity` is now a no-op.
#[test]
fn step_ignores_non_finite_dt_and_gravity() {
    for (dt, g) in [
        (f64::NAN, 9.81),
        (f64::INFINITY, 9.81),
        (1.0 / 240.0, f64::NAN),
        (1.0 / 240.0, f64::INFINITY),
    ] {
        let mut f = SphFluid::new(SphParams::blood(), 4).unwrap();
        f.spawn([0.0, 1.0, 0.0], [0.1, 0.0, 0.0]);
        let before = f.position(0);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            f.step(dt, g, flat);
        }));
        assert!(r.is_ok(), "dt = {dt}, g = {g}: step panicked");
        assert_eq!(f.position(0), before, "dt = {dt}, g = {g}: step was not a no-op");
    }
}

/// The speed cap squared the speed first. A finite velocity whose square overflows
/// (|v| > ~1.3e154) was scaled by max/inf = 0 and *stopped* rather than capped.
#[test]
fn speed_cap_survives_overflowing_speed() {
    let dt = 1.0 / 240.0;
    let mut f = SphFluid::new(SphParams::blood(), 4).unwrap();
    f.spawn([0.0, 10.0, 0.0], [1e155, 0.0, 0.0]);
    let ceiling = f.speed_ceiling(dt);
    f.step(dt, 0.0, flat);
    let v = f.velocity(0);
    assert!(
        (norm(v) - ceiling).abs() < 1e-9 * ceiling,
        "a 1e155 m/s particle left the cap at {v:?}, expected speed {ceiling}"
    );
}

// ── Boundary / lifecycle ─────────────────────────────────────────────────────

/// `friction` used to be a per-substep velocity multiplier applied on every step of
/// ground contact, so the slide was v0 * dt / (1 - friction): proportional to dt,
/// and zero as the solver is refined (0.166 m at 240 Hz, 0.042 m at 960 Hz here). A
/// slippery 0.95 is used so the decay spans many substeps and first-order
/// integration error is not what is being measured.
#[test]
fn ground_friction_is_timestep_independent() {
    fn slide(dt: f64) -> f64 {
        let mut p = SphParams::blood();
        p.friction = 0.95;
        let mut f = SphFluid::new(p, 4).unwrap();
        // A lone drop resting on the ground, sliding at 2 m/s.
        f.spawn([0.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
        let steps = (0.5 / dt).round() as usize;
        for _ in 0..steps {
            f.step(dt, 9.81, flat);
        }
        f.position(0)[0]
    }
    let coarse = slide(1.0 / 240.0);
    let fine = slide(1.0 / 960.0);
    assert!(
        (coarse - fine).abs() < 0.1 * coarse.max(fine),
        "same drop, same 0.5 s: slid {coarse:.5} m at 240 Hz but {fine:.5} m at 960 Hz"
    );
}

/// Two particles spawned at the same point with the same velocity never separate:
/// the force loop skips r <= 1e-9 entirely and every other neighbour acts on both
/// identically, so they move as one particle of double mass forever. Shown inside a
/// compressed block, where pressure is positive and *should* push them apart.
#[test]
#[ignore = "known defect, not fixed in this change: see docs/reviews/2026-09-29-correctness-performance.md"]
fn coincident_particles_separate() {
    let p = SphParams::water();
    let s = p.smoothing_radius * 0.5 * 0.9; // 10% compressed: positive pressure
    let mut f = SphFluid::new(p, 1024).unwrap();
    for x in 0..7 {
        for y in 0..7 {
            for z in 0..7 {
                f.spawn([x as f64 * s, 5.0 + y as f64 * s, z as f64 * s], [0.0; 3]);
            }
        }
    }
    let centre = (3 * 7 + 3) * 7 + 3;
    let dup = f.len();
    f.spawn(f.position(centre), [0.0; 3]);
    f.build_grid();
    f.compute_density_and_pressure();
    let p0 = f.s_pressure[f.slot_of[centre] as usize];
    for _ in 0..240 {
        f.step(1.0 / 240.0, 0.0, flat);
    }
    let (a, b) = (f.position(centre), f.position(dup));
    let d = norm([a[0] - b[0], a[1] - b[1], a[2] - b[2]]);
    assert!(
        d > 0.1 * s,
        "coincident pair (initial pressure {p0:.0} Pa) still {d:e} m apart after 1 s"
    );
}

/// `step`'s doc: "a splash on a slope runs downhill". The ground response only
/// clamps y and never applies a normal, so gravity has no downhill component: a
/// drop set down on a 30-degree slope does not move, and is then retired as
/// settled.
#[test]
#[ignore = "known defect, not fixed in this change: see docs/reviews/2026-09-29-correctness-performance.md"]
fn a_drop_on_a_slope_runs_downhill() {
    let slope = 30f64.to_radians().tan();
    let ground = move |x: f64, _z: f64| -slope * x; // downhill is +x
    let mut f = SphFluid::new(SphParams::water(), 4).unwrap();
    f.spawn([0.0, 0.0, 0.0], [0.0; 3]);
    let mut retired_at = None;
    for _ in 0..480 {
        f.step(1.0 / 240.0, 9.81, ground);
        f.drain_settled(|s| retired_at = Some(s.position));
        if f.is_empty() {
            break;
        }
    }
    let x = retired_at.unwrap_or_else(|| f.position(0))[0];
    assert!(
        x > 0.01,
        "a drop on a 30-degree slope moved {x:e} m downhill in 2 s (retired as \
         settled: {})",
        retired_at.is_some()
    );
}

/// Same contact model, other direction: with `friction = 1.0` (a legal value, "keeps
/// all tangential velocity") a drop sliding into an upslope climbs it at undiminished
/// speed. The y-clamp lifts it without charging its kinetic energy, so mechanical
/// energy grows without bound.
#[test]
#[ignore = "known defect, not fixed in this change: see docs/reviews/2026-09-29-correctness-performance.md"]
fn climbing_a_slope_costs_kinetic_energy() {
    let slope = 30f64.to_radians().tan();
    let ground = move |x: f64, _z: f64| slope * x; // uphill is +x
    let mut p = SphParams::water();
    p.friction = 1.0;
    let dt = 1.0 / 240.0;
    let mut f = SphFluid::new(p, 4).unwrap();
    f.spawn([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    let e0 = 0.5 * 1.0f64.powi(2);
    for _ in 0..480 {
        f.step(dt, 9.81, ground);
    }
    let v = f.velocity(0);
    let y = f.position(0)[1];
    let e1 = 0.5 * norm(v).powi(2) + 9.81 * y;
    assert!(
        e1 <= e0 * 1.01,
        "specific mechanical energy went {e0:.3} -> {e1:.3} J/kg: climbed {y:.3} m with \
         speed still {:.3} m/s",
        norm(v)
    );
}

// ── Diagnostics (tools, not assertions) ──────────────────────────────────────

/// A bit-exact fingerprint of a 1 s splash, for checking that a change meant to be
/// a pure optimisation leaves the solver's output untouched.
#[test]
#[ignore = "diagnostic tool: run with --release --ignored --nocapture and compare the printed fingerprint"]
fn fingerprint() {
    let mut f = SphFluid::new(SphParams::blood(), 2048).unwrap();
    let mut r = rng(4242);
    for _ in 0..1500 {
        f.spawn(
            [r() * 0.4, 1.0 + r() * 0.4, r() * 0.4],
            [r() * 4.0, r() * 2.0, r() * 4.0],
        );
    }
    let mut h: u64 = 0xcbf29ce484222325;
    for k in 0..240 {
        f.step(1.0 / 240.0, 9.81, |x, _| 0.1 * x - 0.2);
        if k % 60 == 59 {
            f.drain_settled(|_| {});
        }
    }
    for i in 0..f.len() {
        for c in f.position(i) {
            h ^= c.to_bits();
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    eprintln!("fingerprint n={} {h:016x}", f.len());
}

/// Where a step's time goes, and how many candidates the 27-cell block holds
/// against how many are really within h.
#[test]
#[ignore = "perf diagnostic: run with --release --ignored --nocapture"]
fn perf_phase_breakdown() {
    use std::time::Instant;
    for &n in &[1024usize, 4096, 16384] {
        let params = SphParams::blood();
        let spacing = params.smoothing_radius * 0.5;
        let mut f = SphFluid::new(params, n).unwrap();
        let side = (n as f64).cbrt().ceil() as usize;
        'o: for x in 0..side {
            for y in 0..side {
                for z in 0..side {
                    let p = [
                        x as f64 * spacing,
                        1.0 + y as f64 * spacing,
                        z as f64 * spacing,
                    ];
                    if !f.spawn(p, [0.0; 3]) {
                        break 'o;
                    }
                }
            }
        }
        // Candidate statistics over the full 27-cell block.
        f.build_grid();
        let h = f.params.smoothing_radius;
        let (mut scanned, mut collided, mut inside) = (0usize, 0usize, 0usize);
        for i in 0..f.len() {
            let base = [f.cell_x[i], f.cell_y[i], f.cell_z[i]];
            for dz in -1..=1i32 {
                for dy in -1..=1i32 {
                    for dx in -1..=1i32 {
                        let c = [base[0] + dx, base[1] + dy, base[2] + dz];
                        let b = bucket(row_hash(c[1], c[2]), c[0], f.table_mask);
                        for slot in f.bucket_start[b] as usize..f.bucket_start[b + 1] as usize {
                            let j = f.order[slot] as usize;
                            scanned += 1;
                            if [f.cell_x[j], f.cell_y[j], f.cell_z[j]] != c {
                                collided += 1;
                                continue;
                            }
                            let (pa, pb) = (f.position(i), f.position(j));
                            let d: f64 = (0..3).map(|a| (pa[a] - pb[a]).powi(2)).sum();
                            if d <= h * h {
                                inside += 1;
                            }
                        }
                    }
                }
            }
        }
        let reps = (200_000 / n).max(5);
        let (mut tg, mut td, mut tf, mut ti) = (0.0, 0.0, 0.0, 0.0);
        for _ in 0..reps {
            let t = Instant::now();
            f.build_grid();
            tg += t.elapsed().as_secs_f64();
            let t = Instant::now();
            f.compute_density_and_pressure();
            td += t.elapsed().as_secs_f64();
            let t = Instant::now();
            f.apply_forces(0.0);
            tf += t.elapsed().as_secs_f64();
            // Undo velocity so the cube stays a cube across reps.
            for v in [&mut f.vx, &mut f.vy, &mut f.vz] {
                v.fill(0.0);
            }
            let t = Instant::now();
            f.integrate(1.0 / 240.0, &flat, None);
            ti += t.elapsed().as_secs_f64();
        }
        let r = reps as f64 * 1e-6;
        eprintln!(
            "n={n}: grid {:.0} us, density {:.0} us, forces {:.0} us, integrate {:.0} us; \
             per particle in 27 cells: {:.1} candidates, {:.2} hash-collided, {:.1} within h",
            tg / r,
            td / r,
            tf / r,
            ti / r,
            scanned as f64 / n as f64,
            collided as f64 / n as f64,
            inside as f64 / n as f64
        );
    }
}
