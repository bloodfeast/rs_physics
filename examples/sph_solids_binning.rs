//! What does binning solids into an SPH fluid cost, scene by scene?
//!
//! Four scenes, stepped in turn so they share the machine's state, each timed by the
//! solver's own binning timer (`SphSolidStats::binning`), medians over the steps:
//!
//! - (a) `far`: 24 limbs and 4 crates out of every particle's reach, over a packed cube
//!   of 4,096 (the `sph/solids` bench's `cube_far`).
//! - (b) `hull_1k`: one 4 m hull capsule (radius 0.5 m) across a resting pool of 1,024.
//! - (c) `hull_16k`: the same hull over 16,384 drops spread over a 20 m square of rolling
//!   ground (1 m of relief either way), so the fluid's cell range is 20 m by 2 m by 20 m.
//! - (d) `melee_16k`: 150 limb capsules (15 actors, 10 limbs each, inside a 6 m square)
//!   over the same spread field.
//!
//! After the timed steps it prints a checksum of every particle's position and velocity
//! bits and the last step's bin entries, so two builds can be compared for bit identity.
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

/// A 4 m hull capsule of radius 0.5 m whose belly clears `y0` by 5 cm, centred on
/// `(cx, cz)`, along x and turned 30 degrees, moving at 1 m/s along its axis.
fn hull(cx: f64, y0: f64, cz: f64) -> SphSolids {
    let (s, c) = (0.5f64, 0.75f64.sqrt());
    let half = [2.0 * c, 0.0, 2.0 * s];
    let y = y0 + 0.55;
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
    let names = ["(a) far", "(b) hull_1k", "(c) hull_16k", "(d) melee_16k"];
    let small_pool = pool(1_024);
    let spread = field(16_384);
    let mut fluids = [packed_cube(4_096), small_pool, spread.clone(), spread];
    let solids = [
        far_solids(),
        hull(0.16, 0.0, 0.16),
        hull(10.0, hills(10.0, 10.0), 10.0),
        melee(),
    ];
    let mut binning = vec![Vec::with_capacity(steps); 4];
    let mut grid = vec![Vec::with_capacity(steps); 4];
    for _ in 0..steps {
        for k in 0..4 {
            let gravity = if k == 0 { 0.0 } else { 9.81 };
            if k == 1 {
                fluids[k].step_with_solids(dt, gravity, flat, &solids[k]);
            } else {
                fluids[k].step_with_solids(dt, gravity, hills, &solids[k]);
            }
            let us = |d: Duration| d.as_secs_f64() * 1e6;
            binning[k].push(us(fluids[k].solid_stats().binning));
            grid[k].push(us(fluids[k].phase_times().grid));
        }
    }
    for k in 0..4 {
        let st = fluids[k].solid_stats();
        println!(
            "{:<14} n={:>5} solids={:>3}  binning median {:8.2} us  grid {:8.1} us  \
             entries {:>6} ray-tested {:>6} contacts {:>5}  checksum {:016x}",
            names[k],
            fluids[k].len(),
            st.solids,
            median(&mut binning[k]),
            median(&mut grid[k]),
            st.bin_entries,
            st.ray_tested,
            st.contacts,
            checksum(&fluids[k]),
        );
    }
}
