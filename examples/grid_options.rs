//! Interleaved medians for the grids' turbulence options: a `FluidGrid3D` step at 32^3
//! and 64^3 with each option, on the forced source of `benches/fluid_grid.rs`, every
//! configuration alternated round by round in one process so they share the machine's
//! state (it drifts with heat over minutes).
//!
//! ```text
//! cargo run --release --example grid_options --features fluid_simulation -- [rounds]
//! ```

use std::time::Instant;

use rs_physics::fluid_dynamics::{
    AdvectionScheme, FluidGrid3D, SolverConfig, VorticityConfinement,
};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn source_3d(grid: &mut FluidGrid3D, n: usize) {
    for i in n / 4..3 * n / 4 {
        grid.add_density(i, n / 2, n / 2, 1.0).unwrap();
        grid.add_velocity(i, n / 2, n / 2, 0.0, 0.05, 0.0).unwrap();
    }
}

fn grid(rounds: usize) {
    println!(
        "FluidGrid3D step: ms, median of {rounds} interleaved rounds (and pressure iterations)"
    );
    let configs = [
        ("off", SolverConfig::default()),
        (
            "maccormack",
            SolverConfig::default().with_advection(AdvectionScheme::MacCormack),
        ),
        (
            "confinement",
            SolverConfig::default()
                .with_vorticity_confinement(VorticityConfinement::MatchNumericalDissipation),
        ),
        (
            "both",
            SolverConfig::default()
                .with_advection(AdvectionScheme::MacCormack)
                .with_vorticity_confinement(VorticityConfinement::MatchNumericalDissipation),
        ),
    ];
    for &n in &[32usize, 64] {
        let mut grids: Vec<FluidGrid3D> = configs
            .iter()
            .map(|(_, c)| {
                let mut g = FluidGrid3D::with_solver(n, n, n, 1e-5, 1e-5, 1.0 / 60.0, *c).unwrap();
                for _ in 0..10 {
                    source_3d(&mut g, n);
                    g.step();
                }
                g
            })
            .collect();
        let steps = if n == 32 { 10 } else { 2 };
        let mut times = vec![Vec::new(); configs.len()];
        let mut iterations = vec![0usize; configs.len()];
        for _ in 0..rounds {
            for (c, g) in grids.iter_mut().enumerate() {
                let started = Instant::now();
                for _ in 0..steps {
                    source_3d(g, n);
                    g.step();
                    iterations[c] += g.get_last_pressure_iterations();
                }
                times[c].push(started.elapsed().as_secs_f64() * 1e3 / steps as f64);
            }
        }
        let off = median(times[0].clone());
        print!("  {n}^3:");
        for (c, (name, _)) in configs.iter().enumerate() {
            let t = median(times[c].clone());
            print!(
                "  {name} {t:.2} ({:+.0}%, {:.1} it)",
                100.0 * (t / off - 1.0),
                iterations[c] as f64 / (rounds * steps) as f64
            );
        }
        println!();
    }
}

fn main() {
    let rounds: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(9);
    grid(rounds);
}
