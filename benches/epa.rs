//! GJK and EPA per call, on the four overlaps the 2026-09-29 review timed by hand
//! (`perf_gjk_epa_per_call` in `interactions/gjk_epa_regression_tests.rs`): rotated boxes,
//! a box resting on a slab, a sphere against a rotated box, and a cylinder on a slab.
//!
//! Public API only, so the same file measures any revision of the crate that has
//! `gjk_collision_detection_ex` and `epa_contact_points_ex`.

use criterion::{criterion_group, criterion_main, Criterion};
use rs_physics::interactions::gjk_collision_3d::{epa_contact_points_ex, gjk_collision_detection_ex};
use rs_physics::models::{Quaternion, Shape3D};

type Case = (&'static str, Shape3D, (f64, f64, f64), Quaternion, Shape3D, (f64, f64, f64), Quaternion);

fn cases() -> Vec<Case> {
    let q1 = Quaternion::from_axis_angle((0.3, 1.0, 0.2), 0.7);
    let q2 = Quaternion::from_axis_angle((1.0, 0.1, -0.4), 1.1);
    vec![
        ("box_box_rotated", Shape3D::Cuboid(1.0, 1.0, 1.0), (0.0, 0.0, 0.0), q1,
         Shape3D::Cuboid(1.0, 1.0, 1.0), (0.8, 0.3, 0.1), q2),
        ("box_on_ground", Shape3D::Cuboid(1.0, 1.0, 1.0), (0.2, 0.48, 0.1),
         Quaternion::from_axis_angle((0.0, 1.0, 0.0), 0.4),
         Shape3D::Cuboid(20.0, 1.0, 20.0), (0.0, -0.5, 0.0), Quaternion::identity()),
        ("sphere_box_rotated", Shape3D::Sphere(0.5), (0.9, 0.2, 0.1), Quaternion::identity(),
         Shape3D::Cuboid(1.0, 1.0, 1.0), (0.0, 0.0, 0.0), q2),
        ("cylinder_box", Shape3D::Cylinder(0.4, 1.0), (0.0, 0.95, 0.1), q1,
         Shape3D::Cuboid(4.0, 1.0, 4.0), (0.0, 0.0, 0.0), Quaternion::identity()),
    ]
}

fn epa(c: &mut Criterion) {
    let mut group = c.benchmark_group("epa");
    for (name, s1, p1, q1, s2, p2, q2) in cases() {
        let g = gjk_collision_detection_ex(&s1, p1, q1, &s2, p2, q2);
        group.bench_function(name, |b| {
            b.iter(|| {
                epa_contact_points_ex(
                    std::hint::black_box(&s1), std::hint::black_box(p1), q1, &s2, p2, q2,
                    std::hint::black_box(&g),
                )
            })
        });
    }
    group.finish();
}

fn gjk(c: &mut Criterion) {
    let mut group = c.benchmark_group("gjk");
    for (name, s1, p1, q1, s2, p2, q2) in cases() {
        group.bench_function(name, |b| {
            b.iter(|| {
                gjk_collision_detection_ex(
                    std::hint::black_box(&s1), std::hint::black_box(p1), q1, &s2, p2, q2,
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, epa, gjk);
criterion_main!(benches);
