//! Tests for the swirl field and the particle pool's coupling to moving air.
//!
//! Oracles: the curl's own identity (a central-difference curl has no
//! central-difference divergence), Kolmogorov's law as written in
//! [`TurbulenceDrive`], the cross-fade's definition, and the plain
//! [`ParticleEffects::integrate`] for a class that ignores the air.

use super::{
    Burst, EffectRng, ParticleClass, ParticleEffects, SwirlField, TurbulenceDrive, VelocityGrid,
};

fn rms_speed(grid: &VelocityGrid, margin: usize) -> f64 {
    let n = grid.dims();
    let (mut sum, mut count) = (0.0f64, 0usize);
    for i in margin..n[0] - margin {
        for j in margin..n[1] - margin {
            for k in margin..n[2] - margin {
                let v = grid.get(i, j, k);
                sum += v.iter().map(|c| (*c as f64).powi(2)).sum::<f64>();
                count += 1;
            }
        }
    }
    (sum / count as f64).sqrt()
}

/// The swirl's divergence is zero to `f32` rounding at every cell whose differences
/// do not reach the still outer layer.
#[test]
fn the_swirl_is_divergence_free_to_float_precision() {
    let drive = TurbulenceDrive::new(3.0, 40.0).unwrap();
    let mut swirl = SwirlField::new([-32.0, 0.0, -32.0], [32, 32, 32], 2.0, drive, 11).unwrap();
    swirl.advance(0.37);
    let grid = swirl.velocity();
    let peak = (0..32)
        .flat_map(|i| (0..32).flat_map(move |j| (0..32).map(move |k| (i, j, k))))
        .map(|(i, j, k)| grid.get(i, j, k).iter().fold(0.0f32, |m, c| m.max(c.abs())))
        .fold(0.0f32, f32::max);
    assert!(peak > 0.5, "the field is too weak to test: {peak}");
    // The scale of one difference: a velocity over a cell. Rounding is a few `f32`
    // epsilons of that.
    let scale = peak / grid.cell_size();
    let mut rng = EffectRng::new(5);
    let mut worst = 0.0f32;
    for _ in 0..2000 {
        let pick = |rng: &mut EffectRng| 2 + (rng.next_u32() % 28) as usize;
        let (i, j, k) = (pick(&mut rng), pick(&mut rng), pick(&mut rng));
        worst = worst.max(grid.divergence_at(i, j, k).abs());
    }
    println!("worst divergence {worst:e} against a velocity gradient scale {scale:e}");
    assert!(
        worst <= 16.0 * f32::EPSILON * scale,
        "divergence {worst} against {scale}"
    );
}

/// Each octave's rms speed is Kolmogorov's eddy velocity at its size, measured on a
/// 64^3 field one octave at a time.
#[test]
fn each_octave_moves_at_its_kolmogorov_eddy_velocity() {
    let (u, l, h) = (4.0f32, 80.0f32, 1.5f32);
    let drive = TurbulenceDrive::new(u, l).unwrap();
    // The law itself: an eddy an eighth of the plume's size turns at half its speed.
    assert!((drive.eddy_velocity(l / 8.0) - u / 2.0).abs() < 1e-6);
    assert!((drive.turnover_time(l / 8.0) - (l / 8.0) / (u / 2.0)).abs() < 1e-5);

    for octave in 0..3 {
        // Average over seeds for the coarse octaves, which have fewer lattice points.
        let seeds = [1u32, 2, 3, 4];
        let mut measured = 0.0;
        let mut law = 0.0;
        for &seed in &seeds {
            let mut swirl = SwirlField::new([0.0; 3], [64, 64, 64], h, drive, seed).unwrap();
            swirl.isolate_octave(octave);
            let (scale, speed, turnover) = swirl.octave_law(octave);
            assert_eq!(scale, h * (2 << octave) as f32);
            assert!((speed - u * (scale / l).cbrt()).abs() < 1e-5 * u);
            assert!((turnover - scale / speed).abs() < 1e-5 * turnover);
            measured += rms_speed(swirl.velocity(), 2) / seeds.len() as f64;
            law = speed as f64;
        }
        let error = measured / law - 1.0;
        println!(
            "octave {octave}: rms {measured:.4} m/s against the law's {law:.4} ({:+.1}%)",
            100.0 * error
        );
        // A realisation scatters about its expectation by about one over the square
        // root of its lattice points: 1% for the finest octave, 10% for the coarsest.
        let tolerance = [0.03, 0.06, 0.12][octave];
        assert!(error.abs() < tolerance, "octave {octave} off by {error}");
    }
}

/// An octave's swirl changes over its eddies' turnover time: a cross-fade of two
/// draws by angle `(t / tau) * pi / 2`, so its correlation with where it started is
/// `cos` of that angle, and after one turnover the swirl is the second draw.
#[test]
fn an_octave_changes_over_its_turnover_time() {
    let drive = TurbulenceDrive::new(3.0, 60.0).unwrap();
    let mut swirl = SwirlField::new([0.0; 3], [32, 32, 32], 2.0, drive, 9).unwrap();
    let octave = 0;
    let (_, _, tau) = swirl.octave_law(octave);
    let start = swirl.lattice_now(octave);
    let correlation = |a: &[f32], b: &[f32]| {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum();
        let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum();
        dot / (na * nb).sqrt()
    };

    swirl.advance(tau / 3.0);
    let third = swirl.lattice_now(octave);
    let want = (std::f64::consts::FRAC_PI_2 / 3.0).cos();
    let got = correlation(&start, &third);
    println!("correlation after a third of a turnover: {got:.4}, construction {want:.4}");
    // Two independent draws of 9000 points correlate by about 0.01.
    assert!((got - want).abs() < 0.03, "{got} vs {want}");

    swirl.advance(2.0 * tau / 3.0);
    let whole = swirl.lattice_now(octave);
    let got = correlation(&start, &whole);
    println!("correlation after a whole turnover: {got:.4}");
    assert!(got.abs() < 0.03, "{got}");
    assert_eq!(swirl.lattice_spacing(octave), 2);

    // The velocity follows the potential: after a turnover the finest octave's swirl
    // has moved on too.
    let mut a = SwirlField::new([0.0; 3], [32, 32, 32], 2.0, drive, 9).unwrap();
    a.isolate_octave(0);
    let before: Vec<f32> = a
        .velocity()
        .cells()
        .iter()
        .flat_map(|c| [c[0], c[1], c[2]])
        .collect();
    a.advance(tau);
    let after: Vec<f32> = a
        .velocity()
        .cells()
        .iter()
        .flat_map(|c| [c[0], c[1], c[2]])
        .collect();
    let got = correlation(&before, &after);
    println!("velocity correlation after a turnover: {got:.4}");
    assert!(got.abs() < 0.05, "{got}");
}

/// A plume narrower than the smallest eddy the grid can carry has no swirl on it.
#[test]
fn no_octave_fits_below_the_grid() {
    let drive = TurbulenceDrive::new(3.0, 3.9).unwrap();
    let mut swirl = SwirlField::new([0.0; 3], [8, 8, 8], 2.0, drive, 1).unwrap();
    assert!(swirl.octave_scales().is_empty());
    swirl.advance(1.0);
    assert!(swirl.velocity().cells().iter().all(|c| *c == [0.0; 4]));
}

/// A fetch at a cell centre is that cell, between centres it is linear, and outside the
/// grid it is the nearest edge.
#[test]
fn a_fetch_is_trilinear_and_clamped() {
    let mut grid = VelocityGrid::new([10.0, 0.0, -4.0], 2.0, [4, 3, 5]).unwrap();
    let f = |i: usize, j: usize, k: usize| {
        [
            i as f32 + 0.5 * j as f32 - k as f32,
            2.0 * j as f32,
            0.25 * k as f32,
        ]
    };
    for i in 0..4 {
        for j in 0..3 {
            for k in 0..5 {
                grid.set(i, j, k, f(i, j, k));
            }
        }
    }
    let centre = |i: usize, j: usize, k: usize| {
        [
            10.0 + 2.0 * i as f32 + 1.0,
            2.0 * j as f32 + 1.0,
            -4.0 + 2.0 * k as f32 + 1.0,
        ]
    };
    assert_eq!(grid.sample(centre(2, 1, 3)), f(2, 1, 3));
    // The field is linear in the indices, so trilinear interpolation reproduces it.
    let p = [
        10.0 + 1.0 + 2.0 * 1.25,
        1.0 + 2.0 * 0.5,
        -4.0 + 1.0 + 2.0 * 2.75,
    ];
    let want = [1.25 + 0.25 - 2.75, 1.0, 0.25 * 2.75];
    let got = grid.sample(p);
    for a in 0..3 {
        assert!((got[a] - want[a]).abs() < 1e-5, "{got:?} vs {want:?}");
    }
    // Far outside on every axis: the corner cell.
    assert_eq!(grid.sample([-1e6, 1e6, -1e6]), f(0, 2, 0));
    assert_eq!(grid.sample([f32::NAN, 1.0, -3.0]), f(0, 0, 0));
}

/// A class with the air off integrates bit-identically to [`ParticleEffects::integrate`],
/// even in a pool where another class rides a strong swirl, over a long run with
/// retirement and re-emission.
#[test]
fn a_class_that_ignores_the_air_is_bit_identical_to_integrate() {
    let drive = TurbulenceDrive::new(5.0, 30.0).unwrap();
    let mut swirl = SwirlField::new([-16.0, 0.0, -16.0], [16, 16, 16], 2.0, drive, 3).unwrap();
    let classes = [
        ParticleClass {
            gravity: 26.0,
            drag: 1.4,
            restitution: 0.32,
        },
        ParticleClass {
            gravity: 1.6,
            drag: 3.4,
            restitution: 0.0,
        },
        // drag * dt above one: the damping clamp's case, where -0.0 can appear.
        ParticleClass {
            gravity: 4.0,
            drag: 400.0,
            restitution: 0.0,
        },
    ];
    let mut with_air = ParticleEffects::with_capacity(4096);
    let mut plain = ParticleEffects::with_capacity(4096);
    for (c, class) in classes.into_iter().enumerate() {
        with_air.set_class(c as u8, class);
        plain.set_class(c as u8, class);
    }
    with_air.set_swirl(1, 1.0);
    let mut rng_a = EffectRng::new(77);
    let mut rng_b = EffectRng::new(77);
    let dt = 1.0 / 60.0;
    let mut compared = 0usize;
    for step in 0..600 {
        for class in 0..3u8 {
            let burst = Burst {
                origin: [0.0, 6.0, 0.0],
                class,
                count: 5,
                speed: 1.0..9.0,
                lifetime: 0.3..3.0,
                size: 1.0..1.0,
                lift: 0.4,
            };
            with_air.emit(&burst, &mut rng_a);
            plain.emit(&burst, &mut rng_b);
        }
        if step % 6 == 0 {
            swirl.advance(6.0 * dt);
        }
        // The field updates every 6 frames and the samples refresh over that period.
        with_air.integrate_in_air(dt, swirl.velocity(), 6.0 * dt);
        plain.integrate(dt);
        // Retirement depends only on lifetimes, which neither path touches
        // differently, so the pools stay index-aligned.
        assert_eq!(with_air.len(), plain.len());
        for i in 0..plain.len() {
            assert_eq!(with_air.class_of(i), plain.class_of(i));
            if with_air.class_of(i) == 1 {
                continue;
            }
            let (a, b) = (with_air.position(i), plain.position(i));
            let (va, vb) = (with_air.velocity(i), plain.velocity(i));
            for k in 0..3 {
                assert_eq!(
                    a[k].to_bits(),
                    b[k].to_bits(),
                    "step {step} particle {i} position axis {k}"
                );
                assert_eq!(
                    va[k].to_bits(),
                    vb[k].to_bits(),
                    "step {step} particle {i} velocity axis {k}"
                );
            }
            compared += 1;
        }
    }
    assert!(compared > 100_000, "{compared}");
}

/// With no class on the air, the call is `integrate` itself.
#[test]
fn no_class_on_the_air_is_integrate() {
    let air = {
        let mut g = VelocityGrid::new([0.0; 3], 1.0, [4, 4, 4]).unwrap();
        g.fill([3.0, 3.0, 3.0]);
        g
    };
    let mut a = ParticleEffects::with_capacity(64);
    let mut b = ParticleEffects::with_capacity(64);
    let class = ParticleClass {
        gravity: 9.0,
        drag: 2.0,
        restitution: 0.0,
    };
    a.set_class(0, class);
    b.set_class(0, class);
    for i in 0..32 {
        a.emit_one([i as f32, 1.0, 0.0], [1.0, 2.0, -1.0], 5.0, 1.0, 0);
        b.emit_one([i as f32, 1.0, 0.0], [1.0, 2.0, -1.0], 5.0, 1.0, 0);
    }
    for _ in 0..100 {
        a.integrate_in_air(1.0 / 60.0, &air, 0.1);
        b.integrate(1.0 / 60.0);
    }
    for i in 0..a.len() {
        assert_eq!(a.position(i), b.position(i));
        assert_eq!(a.velocity(i), b.velocity(i));
    }
}

/// Swirl 1 is relaxation towards the air at the drag rate: in uniform air a
/// gravity-free particle's velocity approaches the air's as `1 - (1 - drag dt)^n`.
#[test]
fn full_coupling_relaxes_to_the_air_at_the_drag_rate() {
    let mut air = VelocityGrid::new([-50.0; 3], 10.0, [10, 10, 10]).unwrap();
    air.fill([0.0, 0.0, 4.0]);
    let mut fx = ParticleEffects::with_capacity(4);
    let (drag, dt) = (2.5f32, 1.0f32 / 60.0);
    fx.set_class(
        0,
        ParticleClass {
            gravity: 0.0,
            drag,
            restitution: 0.0,
        },
    );
    fx.set_swirl(0, 1.0);
    fx.emit_one([0.0; 3], [0.0; 3], 10.0, 1.0, 0);
    for step in 1..=30 {
        fx.integrate_in_air(dt, &air, dt);
        let want = 4.0 * (1.0 - (1.0 - drag * dt).powi(step));
        assert!(
            (fx.velocity(0)[2] - want).abs() < 1e-5,
            "step {step}: {} vs {want}",
            fx.velocity(0)[2]
        );
    }
}

/// The fast fetch is the portable one to the bit, inside the grid, outside it, and at
/// non-finite positions.
#[test]
fn the_fast_fetch_matches_the_portable_one() {
    let drive = TurbulenceDrive::new(4.0, 30.0).unwrap();
    let mut swirl = SwirlField::new([-10.0, 2.0, -7.0], [12, 9, 15], 1.5, drive, 21).unwrap();
    swirl.advance(0.3);
    let grid = swirl.velocity();
    let mut rng = EffectRng::new(8);
    let mut points: Vec<[f32; 3]> = (0..20_000)
        .map(|_| {
            [
                rng.range(-14.0, 12.0),
                rng.range(-2.0, 20.0),
                rng.range(-11.0, 20.0),
            ]
        })
        .collect();
    points.push([f32::NAN, 3.0, 1.0]);
    points.push([f32::INFINITY, f32::NEG_INFINITY, 0.0]);
    for p in points {
        let (fast, reference) = (grid.sample(p), grid.sample_reference(p));
        for a in 0..3 {
            assert_eq!(
                fast[a].to_bits(),
                reference[a].to_bits(),
                "{p:?}: {fast:?} vs {reference:?}"
            );
        }
    }
}

/// Staggered refresh: with the air updated every `p` frames, each particle re-samples it
/// once in any `p` consecutive calls, a `1 / p` share of the pool a call, and keeps its
/// last sample between. Shown by changing a uniform field and counting who has seen it.
#[test]
fn each_particle_resamples_the_air_once_a_period() {
    let mut air = VelocityGrid::new([-100.0; 3], 50.0, [4, 4, 4]).unwrap();
    air.fill([1.0, 0.0, 0.0]);
    let mut fx = ParticleEffects::with_capacity(600);
    fx.set_class(
        0,
        ParticleClass {
            gravity: 0.0,
            drag: 1.0,
            restitution: 0.0,
        },
    );
    fx.set_swirl(0, 1.0);
    for i in 0..600 {
        fx.emit_one([i as f32 * 0.01, 0.0, 0.0], [0.0; 3], 1000.0, 1.0, 0);
    }
    // A step and period exact in binary, so the share a call is exactly 100.
    let dt = 0.25;
    let period = 6;
    // The first call samples every new particle.
    fx.integrate_in_air(dt, &air, period as f32 * dt);
    assert!((0..fx.len()).all(|i| fx.air_sample(i) == [1.0, 0.0, 0.0]));

    air.fill([0.0, 2.0, 0.0]);
    for call in 1..=period {
        fx.integrate_in_air(dt, &air, period as f32 * dt);
        let seen = (0..fx.len())
            .filter(|&i| fx.air_sample(i) == [0.0, 2.0, 0.0])
            .count();
        // A sixth of the pool a call, rounded up.
        assert_eq!(seen, (call * 100).min(600), "call {call}");
        // Nobody holds anything but the old field or the new one.
        assert!((0..fx.len()).all(|i| {
            let a = fx.air_sample(i);
            a == [1.0, 0.0, 0.0] || a == [0.0, 2.0, 0.0]
        }));
    }
}

/// A particle emitted between two calls is sampled on the next one, whether it was
/// appended or overwrote the oldest slot of a full pool, so it never rides a stale or
/// missing sample for a period.
#[test]
fn new_particles_are_sampled_on_their_first_call() {
    let mut air = VelocityGrid::new([-100.0; 3], 50.0, [4, 4, 4]).unwrap();
    air.fill([0.0, 0.0, 3.0]);
    let mut fx = ParticleEffects::with_capacity(64);
    fx.set_class(
        0,
        ParticleClass {
            gravity: 0.0,
            drag: 1.0,
            restitution: 0.0,
        },
    );
    fx.set_swirl(0, 1.0);
    let dt = 1.0 / 60.0;
    // A long period: the turn alone would reach a new particle only after many calls.
    let refresh = 100.0 * dt;
    for i in 0..40 {
        fx.emit_one([i as f32, 0.0, 0.0], [0.0; 3], 1000.0, 1.0, 0);
    }
    fx.integrate_in_air(dt, &air, refresh);
    air.fill([5.0, 0.0, 0.0]);
    // Appended.
    for i in 0..10 {
        fx.emit_one([i as f32, 1.0, 0.0], [0.0; 3], 1000.0, 1.0, 0);
    }
    fx.integrate_in_air(dt, &air, refresh);
    let appended = (40..50)
        .filter(|&i| fx.air_sample(i) == [5.0, 0.0, 0.0])
        .count();
    assert_eq!(appended, 10);
    // Fill the pool, then overwrite: the overwritten slots hold new particles.
    for i in 0..14 {
        fx.emit_one([i as f32, 2.0, 0.0], [0.0; 3], 1000.0, 1.0, 0);
    }
    fx.integrate_in_air(dt, &air, refresh);
    air.fill([0.0, -7.0, 0.0]);
    for i in 0..20 {
        fx.emit_one([i as f32, 3.0, 0.0], [0.0; 3], 1000.0, 1.0, 0);
    }
    fx.integrate_in_air(dt, &air, refresh);
    let fresh = (0..fx.len())
        .filter(|&i| fx.position(i)[1] > 2.5)
        .collect::<Vec<_>>();
    assert_eq!(fresh.len(), 20);
    assert!(fresh.iter().all(|&i| fx.air_sample(i) == [0.0, -7.0, 0.0]));
}

/// The paired fetch the pool samples with gives each particle exactly what
/// [`VelocityGrid::sample`] gives it, odd counts and points outside the grid included.
#[test]
fn a_run_of_samples_matches_one_at_a_time() {
    let drive = TurbulenceDrive::new(4.0, 30.0).unwrap();
    let swirl = SwirlField::new([-10.0, 2.0, -7.0], [12, 9, 15], 1.5, drive, 5).unwrap();
    let grid = swirl.velocity();
    let mut rng = EffectRng::new(17);
    let n = 1001;
    let p: Vec<[f32; 3]> = (0..n)
        .map(|_| {
            [
                rng.range(-14.0, 12.0),
                rng.range(-2.0, 20.0),
                rng.range(-11.0, 20.0),
            ]
        })
        .collect();
    let (px, py, pz): (Vec<f32>, Vec<f32>, Vec<f32>) = (
        p.iter().map(|q| q[0]).collect(),
        p.iter().map(|q| q[1]).collect(),
        p.iter().map(|q| q[2]).collect(),
    );
    let (mut ax, mut ay, mut az) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    grid.sample_run([&px, &py, &pz], [&mut ax, &mut ay, &mut az]);
    for i in 0..n {
        let want = grid.sample(p[i]);
        assert_eq!(
            [ax[i].to_bits(), ay[i].to_bits(), az[i].to_bits()],
            want.map(f32::to_bits),
            "{i}"
        );
    }
}
