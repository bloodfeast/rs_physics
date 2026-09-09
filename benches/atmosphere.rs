//! What the boundary-layer closed forms cost per point.
//!
//! The whole argument for [`rs_physics::atmosphere::boundary_layer`] over a grid solver is
//! that the consumer's question — *what is the wind at this plant* — is asked once per
//! plant, at bake time, tens of thousands of times, and then never again. That argument is
//! only worth making if the per-point cost is what it claims to be: a handful of floating
//! point operations, no allocation, and no error path.
//!
//! `plants` is the shape of the real call: 40 000 points, each with its own height and its
//! own locally-sampled upwind gradient, through one `WindProfile` built once.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rs_physics::atmosphere::{
    canopy_profile, fractional_speedup_from_slope, Air, CanopyAttenuation, HillForm, Surface,
    WindProfile,
};

/// Heights and gradients standing in for a heightfield sample. Deterministic, so two runs
/// measure the same work.
fn field(count: usize) -> Vec<(f64, f64)> {
    (0..count)
        .map(|i| {
            let t = i as f64 * 0.000_157;
            (0.2 + 2.0 * (t.sin() * t.sin()), 0.45 * (t * 3.1).sin())
        })
        .collect()
}

fn benchmark(c: &mut Criterion) {
    let profile = WindProfile::new(9.0, 10.0, Surface::SCRUB).unwrap();
    let points = field(40_000);

    let mut group = c.benchmark_group("atmosphere");

    // The single per-point call a foliage bake makes.
    group.bench_function("wind_over_slope", |b| {
        b.iter(|| {
            let mut total = 0.0;
            for &(height, slope) in &points {
                total += profile.over_slope(black_box(height), black_box(slope), HillForm::Ridge);
            }
            black_box(total)
        })
    });

    // The two halves separately, to say which one costs what.
    group.bench_function("log_profile_only", |b| {
        b.iter(|| {
            let mut total = 0.0;
            for &(height, _) in &points {
                total += profile.at_height(black_box(height));
            }
            black_box(total)
        })
    });

    group.bench_function("speedup_only", |b| {
        b.iter(|| {
            let mut total = 0.0;
            for &(_, slope) in &points {
                total += fractional_speedup_from_slope(black_box(slope), HillForm::Ridge);
            }
            black_box(total)
        })
    });

    group.bench_function("canopy_profile", |b| {
        b.iter(|| {
            let mut total = 0.0;
            for &(height, _) in &points {
                total += canopy_profile(
                    black_box(6.0),
                    black_box(height),
                    black_box(3.0),
                    CanopyAttenuation::DENSE_FOREST,
                );
            }
            black_box(total)
        })
    });

    // The state derivations, which a caller does once per weather change rather than per
    // point — measured so nobody has to guess whether they could be done per point.
    let air = Air::winter();
    group.bench_function("air_density", |b| b.iter(|| black_box(black_box(&air).density())));
    group.bench_function("air_viscosity", |b| {
        b.iter(|| black_box(black_box(&air).dynamic_viscosity()))
    });

    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
