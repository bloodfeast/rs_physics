//! Per-phase microseconds of `SphFluid::step` on a packed cube, at 1, 4 and 8 threads,
//! from the solver's own phase timers. Medians over forty steps, rounds alternated in one
//! process. `r2_phases [rounds]`.

use rs_physics::fluid_dynamics::{SphFluid, SphParams};

fn packed_cube(n: usize) -> SphFluid {
    let params = SphParams::blood();
    let spacing = params.smoothing_radius * 0.5;
    let mut fluid = SphFluid::new(params, n).unwrap();
    let side = (n as f64).cbrt().ceil() as usize;
    'outer: for x in 0..side {
        for y in 0..side {
            for z in 0..side {
                let p = [x as f64 * spacing, 1.0 + y as f64 * spacing, z as f64 * spacing];
                if !fluid.spawn(p, [0.0; 3]) {
                    break 'outer;
                }
            }
        }
    }
    fluid
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let rounds: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(4);
    let sizes = [1_024usize, 4_096, 16_384];
    for round in 0..rounds {
        for &n in &sizes {
            let scene = packed_cube(n);
            for threads in [1usize, 4, 8] {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                let mut f = scene.clone();
                let mut ph = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                pool.install(|| {
                    f.step(1.0 / 240.0, 9.81, |_, _| 0.0);
                    for _ in 0..40 {
                        f.step(1.0 / 240.0, 9.81, |_, _| 0.0);
                        let t = f.phase_times();
                        for (k, d) in [t.grid, t.density, t.forces, t.integrate, t.total()].iter().enumerate() {
                            ph[k].push(d.as_secs_f64() * 1e6);
                        }
                    }
                });
                let m: Vec<String> = ph.iter_mut().map(|v| format!("{:9.1}", median(v))).collect();
                println!("round {round} n={n:>5} threads={threads}  grid/density/forces/integrate/total us {}", m.join(" "));
            }
        }
    }
}
