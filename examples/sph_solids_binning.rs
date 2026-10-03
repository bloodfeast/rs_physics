//! What does binning solids into an SPH fluid cost, scene by scene?
//!
//! Six scenes, stepped in turn so they share the machine's state, each timed by the
//! solver's own timers (`SphSolidStats::binning` and `SphPhaseTimes`), medians over the
//! steps:
//!
//! - (a) `far`: 24 limbs and 4 crates out of every particle's reach, over a packed cube
//!   of 4,096 (the `sph/solids` bench's `cube_far`).
//! - (b) `hull_1k`: one 4 m hull capsule (radius 0.5 m) across a resting pool of 1,024.
//! - (c) `hull_16k`: the same hull over 16,384 drops spread over a 20 m square of rolling
//!   ground (1 m of relief either way), so the fluid's cell range is 20 m by 2 m by 20 m.
//! - (d) `melee_16k`: 150 limb capsules (15 actors, 10 limbs each, inside a 6 m square)
//!   over the same spread field.
//! - (e) `pool_off` and (f) `pool_wade`: a resting pool of 1,024 stepped plainly, and
//!   the same pool with a shin wading back and forth through it at 1 m/s (the
//!   `sph/solids` bench's `pool_off` and `pool_wade`): the wading cost is (f)'s total
//!   less (e)'s.
//!
//! For each it prints the binning, grid, move (`integrate`) and whole-step medians, then
//! a checksum of every particle's position and velocity bits, the last step's bin
//! entries and contact tests and the contacts over the run, so two builds can be
//! compared for bit identity.
//! `sph_solids_binning [steps]`, default 200.

use std::time::Duration;

use rs_physics::fluid_dynamics::{SphFluid, SphParams, SphSolids};

/// Rolling ground: about a metre of relief either way over the 20 m field.
fn hills(x: f64, z: f64) -> f64 {
    0.6 * (0.31 * x).sin() + 0.5 * (0.23 * z).cos()
}

fn flat(_x: f64, _z: f64) -> f64 {
    0.0
}

/// A fixed xorshift so the scenes are the same in every build.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn packed_cube(n: usize) -> SphFluid {
    let params = SphParams::blood();
    let spacing = params.smoothing_radius * 0.5;
    let mut fluid = SphFluid::new(params, n).unwrap();
    let side = (n as f64).cbrt().ceil() as usize;
    'outer: for x in 0..side {
        for y in 0..side {
            for z in 0..side {
                let p = [
                    x as f64 * spacing,
                    1.0 + y as f64 * spacing,
                    z as f64 * spacing,
                ];
                if !fluid.spawn(p, [0.0; 3]) {
                    break 'outer;
                }
            }
        }
    }
    fluid
}

/// A resting pool four particles deep on level ground.
fn pool(n: usize) -> SphFluid {
    let params = SphParams::blood();
    let spacing = params.smoothing_radius * 0.5;
    let mut fluid = SphFluid::new(params, n).unwrap();
    let side = ((n / 4) as f64).sqrt().ceil() as usize;
    'outer: for x in 0..side {
        for z in 0..side {
            for y in 0..4 {
                let p = [
                    x as f64 * spacing,
                    0.5 * spacing + y as f64 * spacing,
                    z as f64 * spacing,
                ];
                if !fluid.spawn(p, [0.0; 3]) {
                    break 'outer;
                }
            }
        }
    }
    for _ in 0..240 {
        fluid.step(1.0 / 240.0, 9.81, flat);
    }
    fluid
}

/// Drops scattered over a 20 m square, each resting a centimetre above the hills.
fn field(n: usize) -> SphFluid {
    let mut fluid = SphFluid::new(SphParams::blood(), n).unwrap();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    while fluid.len() < n {
        let (x, z) = (20.0 * rng.unit(), 20.0 * rng.unit());
        fluid.spawn([x, hills(x, z) + 0.01, z], [0.0; 3]);
    }
    fluid
}

fn far_solids() -> SphSolids {
    let mut solids = SphSolids::with_capacity(24, 4);
    for k in 0..24 {
        let x = 3.0 + 0.3 * (k % 6) as f64;
        let z = 0.3 * (k / 6) as f64;
        solids.push_capsule(
            [x, 0.1, z],
            [x, 0.5, z],
            0.06,
            [1.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        );
    }
    for k in 0..4 {
        let x = -3.0 - 0.8 * k as f64;
        solids.push_box([x, 0.25, 0.0], [0.25; 3], 0.3 * k as f64, [0.0; 3]);
    }
    solids
}

/// A 4 m hull capsule of radius 0.5 m whose belly sits at `y0`, centred on
/// `(cx, cz)`, along x and turned 30 degrees, moving at 1 m/s along its axis.
fn hull(cx: f64, y0: f64, cz: f64) -> SphSolids {
    let (s, c) = (0.5f64, 0.75f64.sqrt());
    let half = [2.0 * c, 0.0, 2.0 * s];
    let y = y0 + 0.5;
    let v = [c, 0.0, s];
    let mut solids = SphSolids::with_capacity(1, 0);
    solids.push_capsule(
        [cx - half[0], y, cz - half[2]],
        [cx + half[0], y, cz + half[2]],
        0.5,
        v,
        v,
    );
    solids
}

/// Fifteen actors in a 6 m square at the field's centre, ten limbs each: two shins,
/// two thighs, two forearms, two upper arms, a torso and a head-and-neck, as capsules.
fn melee() -> SphSolids {
    let mut solids = SphSolids::with_capacity(150, 0);
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    // (offset x, z, bottom, top, radius) of each limb about an actor's feet.
    let limbs: [(f64, f64, f64, f64, f64); 10] = [
        (-0.12, 0.0, 0.08, 0.5, 0.06),
        (0.12, 0.0, 0.08, 0.5, 0.06),
        (-0.12, 0.05, 0.5, 0.95, 0.08),
        (0.12, -0.05, 0.5, 0.95, 0.08),
        (-0.3, 0.15, 0.95, 1.25, 0.05),
        (0.3, 0.2, 0.95, 1.25, 0.05),
        (-0.25, 0.0, 1.25, 1.5, 0.055),
        (0.25, 0.0, 1.25, 1.5, 0.055),
        (0.0, 0.0, 1.0, 1.5, 0.16),
        (0.0, 0.0, 1.55, 1.75, 0.11),
    ];
    for _ in 0..15 {
        let (x, z) = (7.0 + 6.0 * rng.unit(), 7.0 + 6.0 * rng.unit());
        let g = hills(x, z);
        let v = [2.0 * rng.unit() - 1.0, 0.0, 2.0 * rng.unit() - 1.0];
        for &(ox, oz, lo, hi, r) in &limbs {
            let lean = 0.15 * (2.0 * rng.unit() - 1.0);
            solids.push_capsule(
                [x + ox, g + lo, z + oz],
                [x + ox + lean, g + hi, z + oz - lean],
                r,
                v,
                [2.0 * v[0], 0.0, 2.0 * v[2]],
            );
        }
    }
    solids
}

/// A shin wading back and forth across `fluid`'s pool at 1 m/s, at substep `s`: the
/// `sph/solids` bench's `wading_shin`.
fn wading_shin(fluid: &SphFluid, s: u64, solids: &mut SphSolids) {
    let spacing = fluid.params().smoothing_radius * 0.5;
    let side = ((fluid.capacity() / 4) as f64).sqrt().ceil() * spacing;
    // A triangle wave from 0.1 m before the pool to 0.1 m past it.
    let span = side + 0.2;
    let travel = (s as f64 / 240.0) % (2.0 * span);
    let (x, vx) = if travel < span {
        (travel - 0.1, 1.0)
    } else {
        (2.0 * span - travel - 0.1, -1.0)
    };
    solids.clear();
    solids.push_capsule(
        [x, -0.05, 0.5 * side],
        [x, 0.4, 0.5 * side],
        0.06,
        [vx, 0.0, 0.0],
        [vx, 0.0, 0.0],
    );
}

fn checksum(fluid: &SphFluid) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for i in 0..fluid.len() {
        for c in fluid.position(i).iter().chain(&fluid.velocity(i)) {
            h ^= c.to_bits();
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let steps: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(200);
    let dt = 1.0 / 240.0;
    const SCENES: usize = 6;
    let names = [
        "(a) far",
        "(b) hull_1k",
        "(c) hull_16k",
        "(d) melee_16k",
        "(e) pool_off",
        "(f) pool_wade",
    ];
    let small_pool = pool(1_024);
    let spread = field(16_384);
    let mut fluids = [
        packed_cube(4_096),
        small_pool.clone(),
        spread.clone(),
        spread,
        small_pool.clone(),
        small_pool,
    ];
    let mut solids = [
        far_solids(),
        hull(0.16, 0.0, 0.16),
        hull(10.0, hills(10.0, 10.0), 10.0),
        melee(),
        SphSolids::new(),
        SphSolids::with_capacity(1, 0),
    ];
    // Per scene, per step: binning, grid, integrate, total, microseconds.
    let mut times = vec![vec![[0.0f64; 4]; steps]; SCENES];
    let mut contacts = [0usize; SCENES];
    for s in 0..steps {
        for k in 0..SCENES {
            let gravity = if k == 0 { 0.0 } else { 9.81 };
            match k {
                1 => fluids[k].step_with_solids(dt, gravity, flat, &solids[k]),
                4 => fluids[k].step(dt, gravity, flat),
                5 => {
                    wading_shin(&fluids[k], s as u64, &mut solids[k]);
                    fluids[k].step_with_solids(dt, gravity, flat, &solids[k]);
                }
                _ => fluids[k].step_with_solids(dt, gravity, hills, &solids[k]),
            }
            let us = |d: Duration| d.as_secs_f64() * 1e6;
            let t = fluids[k].phase_times();
            times[k][s] = [
                us(fluids[k].solid_stats().binning),
                us(t.grid),
                us(t.integrate),
                us(t.total()),
            ];
            contacts[k] += fluids[k].solid_stats().contacts;
        }
    }
    for k in 0..SCENES {
        let st = fluids[k].solid_stats();
        let med = |c: usize| {
            let mut col: Vec<f64> = times[k].iter().map(|t| t[c]).collect();
            median(&mut col)
        };
        println!(
            "{:<14} n={:>5} solids={:>3}  binning {:8.2} us  grid {:8.1} us  move {:8.1} us  \
             total {:8.1} us  entries {:>6} ray-tested {:>6} contacts {:>7}  checksum {:016x}",
            names[k],
            fluids[k].len(),
            st.solids,
            med(0),
            med(1),
            med(2),
            med(3),
            st.bin_entries,
            st.ray_tested,
            contacts[k],
            checksum(&fluids[k]),
        );
    }
}
