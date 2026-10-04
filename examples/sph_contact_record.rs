//! What does the contact record cost?
//!
//! Three resting pools of blood, 1,024, 4,096 and 16,384 particles four deep, each with a
//! hundred capsules standing in it (a ten by ten grid over the pool, every one wading
//! back and forth at 0.5 m/s), stepped in turn so they share the machine's state. Each
//! step is timed by the solver's own timers (`SphPhaseTimes`, `SphSolidStats`), and the
//! medians over the steps are printed: binning, move (`integrate`, where the contact and
//! its record run) and the whole step, with the contacts a step and a checksum of every
//! particle's position and velocity bits, so two builds can be compared for bit
//! identity as well as for time.
//!
//! The example uses only the 0.3.7 surface, so the same file builds against the library
//! before and after the record and the two binaries can be run interleaved.
//!
//! `sph_contact_record [steps]`, default 400.

use std::time::Duration;

use rs_physics::fluid_dynamics::{SphFluid, SphParams, SphSolids};

const DT: f64 = 1.0 / 240.0;
const SCALES: [usize; 3] = [1_024, 4_096, 16_384];

fn flat(_x: f64, _z: f64) -> f64 {
    0.0
}

/// The side of the pool `pool(n)` lays, metres.
fn pool_side(n: usize) -> f64 {
    let spacing = SphParams::blood().smoothing_radius * 0.5;
    ((n / 4) as f64).sqrt().ceil() * spacing
}

/// A resting pool four particles deep on level ground, settled for a second.
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
        fluid.step(DT, 9.81, flat);
    }
    fluid
}

/// A hundred capsules on a ten by ten grid over the pool of side `side`, at substep `s`:
/// each stands from below the ground to above the pool, its radius a quarter of the
/// grid pitch (at most 3 cm), and wades along x at 0.5 m/s over half the pitch, each
/// one a different phase of its triangle wave. Given at its pose at the end of the
/// substep, with the velocity that carried it there.
fn wading_grid(side: f64, s: u64, solids: &mut SphSolids) {
    let pitch = side / 10.0;
    let radius = (0.25 * pitch).min(0.03);
    let speed = 0.5;
    let span = 0.5 * pitch;
    let period = 2.0 * span / speed;
    solids.clear();
    for k in 0..100u64 {
        let (gx, gz) = ((k % 10) as f64, (k / 10) as f64);
        let t = ((s + 1) as f64 * DT + period * k as f64 / 100.0) % period;
        let travel = t * speed;
        let (dx, vx) = if travel < span {
            (travel, speed)
        } else {
            (2.0 * span - travel, -speed)
        };
        let x = (gx + 0.25) * pitch + dx;
        let z = (gz + 0.5) * pitch;
        solids.push_capsule(
            [x, -0.05, z],
            [x, 0.2, z],
            radius,
            [vx, 0.0, 0.0],
            [vx, 0.0, 0.0],
        );
    }
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
        .unwrap_or(400);
    let mut fluids: Vec<SphFluid> = SCALES.iter().map(|&n| pool(n)).collect();
    let mut solids: Vec<SphSolids> = SCALES
        .iter()
        .map(|_| SphSolids::with_capacity(100, 0))
        .collect();
    // Per scale, per step: binning, integrate, total, microseconds.
    let mut times = vec![vec![[0.0f64; 3]; steps]; SCALES.len()];
    let mut contacts = [0usize; SCALES.len()];
    for s in 0..steps {
        for k in 0..SCALES.len() {
            wading_grid(pool_side(SCALES[k]), s as u64, &mut solids[k]);
            fluids[k].step_with_solids(DT, 9.81, flat, &solids[k]);
            let us = |d: Duration| d.as_secs_f64() * 1e6;
            let t = fluids[k].phase_times();
            let st = fluids[k].solid_stats();
            times[k][s] = [us(st.binning), us(t.integrate), us(t.total())];
            contacts[k] += st.contacts;
        }
    }
    for k in 0..SCALES.len() {
        let med = |c: usize| {
            let mut col: Vec<f64> = times[k].iter().map(|t| t[c]).collect();
            median(&mut col)
        };
        println!(
            "n={:>5} solids={:>3}  binning {:8.2} us  move {:8.1} us  total {:8.1} us  \
             contacts a step {:8.1}  checksum {:016x}",
            fluids[k].len(),
            solids[k].len(),
            med(0),
            med(1),
            med(2),
            contacts[k] as f64 / steps as f64,
            checksum(&fluids[k]),
        );
    }
}
