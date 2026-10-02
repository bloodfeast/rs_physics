//! Bit checksums of the grid fluids and the particle pool after fixed scenarios.
//!
//! A change that claims to leave a solver bit-identical is checked by running this on
//! the commit before it and the commit after it and comparing the lines. Every value is
//! hashed by its bit pattern, so a single flipped bit anywhere changes the line.
//!
//! ```text
//! cargo run --release --example grid_checksum --features "particles fluid_simulation"
//! ```

use rs_physics::fluid_dynamics::{
    FluidGrid, FluidGrid3D, PressureSolver, SolverConfig, WallCondition,
};
use rs_physics::particles::{Burst, EffectRng, ParticleClass, ParticleEffects};

/// FNV-1a over 64-bit words.
struct Hash(u64);

impl Hash {
    fn new() -> Hash {
        Hash(0xcbf2_9ce4_8422_2325)
    }
    fn add(&mut self, bits: u64) {
        for b in bits.to_le_bytes() {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }
}

fn configs() -> Vec<(&'static str, SolverConfig)> {
    vec![
        ("default", SolverConfig::default()),
        ("jacobi", SolverConfig::jacobi(6)),
        ("sor", SolverConfig::sor(6, 1.5)),
        (
            "relaxation",
            SolverConfig::default().with_pressure_solver(PressureSolver::Relaxation),
        ),
        (
            "no_slip",
            SolverConfig::default().with_wall_condition(WallCondition::NoSlip),
        ),
    ]
}

fn grid_2d(name: &str, config: SolverConfig, coefficient: f64) {
    let n = 48;
    let mut g =
        FluidGrid::with_solver(n, n + 8, coefficient, coefficient, 1.0 / 60.0, config).unwrap();
    let mut iterations = 0;
    for step in 0..60 {
        for i in n / 4..3 * n / 4 {
            g.add_density(i, n / 3, 1.0).unwrap();
            let swirl = ((step + i) as f64 * 0.37).sin();
            g.add_velocity(i, n / 3, 0.3 * swirl, 0.4).unwrap();
        }
        g.step();
        iterations += g.get_last_pressure_iterations();
    }
    let mut h = Hash::new();
    for x in 0..n {
        for y in 0..n + 8 {
            let (vx, vy) = g.get_velocity(x, y).unwrap();
            h.add(g.get_density(x, y).unwrap().to_bits());
            h.add(vx.to_bits());
            h.add(vy.to_bits());
        }
    }
    println!(
        "grid2d {name:<12} {:016x} pressure_iterations {iterations}",
        h.0
    );
}

fn grid_3d(name: &str, config: SolverConfig, coefficient: f64) {
    let n = 20;
    let mut g = FluidGrid3D::with_solver(
        n,
        n + 4,
        n - 2,
        coefficient,
        coefficient,
        1.0 / 60.0,
        config,
    )
    .unwrap();
    let mut iterations = 0;
    for step in 0..30 {
        for i in n / 4..3 * n / 4 {
            g.add_density(i, n / 3, n / 2, 1.0).unwrap();
            let swirl = ((step + i) as f64 * 0.37).sin();
            g.add_velocity(i, n / 3, n / 2, 0.3 * swirl, 0.4, -0.2 * swirl)
                .unwrap();
        }
        g.step();
        iterations += g.get_last_pressure_iterations();
    }
    let mut h = Hash::new();
    for x in 0..n {
        for y in 0..n + 4 {
            for z in 0..n - 2 {
                let (vx, vy, vz) = g.get_velocity(x, y, z).unwrap();
                h.add(g.get_density(x, y, z).unwrap().to_bits());
                h.add(vx.to_bits());
                h.add(vy.to_bits());
                h.add(vz.to_bits());
            }
        }
    }
    println!(
        "grid3d {name:<12} {:016x} pressure_iterations {iterations}",
        h.0
    );
}

fn particles() {
    let mut fx = ParticleEffects::with_capacity(20_000);
    #[allow(clippy::needless_update)]
    {
        fx.set_class(
            0,
            ParticleClass {
                gravity: 26.0,
                drag: 1.4,
                restitution: 0.32,
                ..Default::default()
            },
        );
        fx.set_class(
            1,
            ParticleClass {
                gravity: 1.6,
                drag: 3.4,
                restitution: 0.0,
                ..Default::default()
            },
        );
        fx.set_class(
            2,
            ParticleClass {
                gravity: 0.0,
                drag: 400.0,
                restitution: 0.5,
                ..Default::default()
            },
        );
    }
    let mut rng = EffectRng::new(0xC0FFEE);
    let mut h = Hash::new();
    for step in 0..900 {
        for class in 0..3u8 {
            fx.emit(
                &Burst {
                    origin: [step as f32 * 0.01, 4.0, -1.0],
                    class,
                    count: 7,
                    speed: 2.0..15.0,
                    lifetime: if class == 0 { 0.2..0.5 } else { 5.0..12.0 },
                    size: 0.7..1.3,
                    lift: 0.35,
                },
                &mut rng,
            );
        }
        fx.integrate(1.0 / 60.0);
        fx.collide_ground(|x, z| (x * 0.1).sin() * 0.5 + z * 0.01);
    }
    for i in 0..fx.len() {
        for c in fx.position(i).into_iter().chain(fx.velocity(i)) {
            h.add(c.to_bits() as u64);
        }
    }
    println!("particles {:016x} live {}", h.0, fx.len());
}

/// Every scenario at a diffusion and viscosity of 1e-4 widths^2/s, then again at zero,
/// where the diffusion solve has nothing to do.
fn main() {
    for coefficient in [1e-4, 0.0] {
        println!("diffusion and viscosity {coefficient:e}");
        for (name, config) in configs() {
            grid_2d(name, config, coefficient);
        }
        for (name, config) in configs() {
            grid_3d(name, config, coefficient);
        }
    }
    particles();
}
