//! What free-body rotation actually costs, and whether the batch entry point earns its
//! keep.
//!
//! The consumer that motivated [`RigidBodyRotation`] integrates hundreds of tumbling
//! pieces per frame — Ridgeline's debris path runs 520 at once at 240 Hz — so the number
//! that matters is the whole batch, not one body. `BODIES` is that workload.
//!
//! Measured against each other:
//!
//! - `step_many` over a contiguous slice: `dt` validated once, no `Result` per body,
//!   memory order — against the loop a caller would write by hand,
//!   `for b in &mut bodies { b.step(dt) }`, which pays a `Result` and a `dt` check per
//!   body. Both take the slice through `black_box`, so neither gets an advantage from
//!   the benchmark's own shape.
//! - `omega_only/cached_inverse` against `omega_only/recomputed_inverse`: **the same
//!   arithmetic**, differing only in whether `I⁻¹` is a field or a fresh 3×3 inversion
//!   inside every derivative evaluation. This is the pair that prices the cache. It has
//!   to be its own pair rather than a comparison against `step`, because `step` also
//!   integrates the orientation and comparing it to an ω-only variant would credit the
//!   cache with work that simply is not there.
//!
//! Also here: `substeps_1` against `substeps_4`, the same physics at a spin four times
//! faster, which is what the adaptive substep rule costs when it decides it needs more.
//!
//! # What it measured
//!
//! On a Windows x86-64 desktop, `cargo bench --bench rigid_body_rotation`:
//!
//! ```text
//!   step_many/520                        62.5 µs      8.3 Melem/s
//!   step_loop/520                        61.2 µs      8.5 Melem/s
//!   step_angular_velocity_loop/520       32 µs
//!   omega_only/cached_inverse/520        24.1 µs     21.5 Melem/s
//!   omega_only/recomputed_inverse/520    49.3 µs     10.5 Melem/s
//!   substeps_1                            124 ns
//!   substeps_4                            278 ns
//! ```
//!
//! **Read the ratios, not the microseconds.** Repeated runs on a machine with other work
//! on it move every absolute number by ±25% together, while the ratios below held to
//! within a few percent across every run. Each conclusion is stated as the ratio it
//! actually is.
//!
//! **Caching the inverse tensor is worth 2.0–2.1×** on identical arithmetic. That is the
//! one clear win, and it is the thing `RigidBodyRotation` does that
//! `InertiaTensor::apply_inverse_to_torque` cannot.
//!
//! **Not carrying the orientation is worth 2×.** `step_angular_velocity` costs about
//! half of `step`, which is what the four quaternion Runge-Kutta stages and the
//! renormalisation come to. The caller whose orientation lives in an ECS transform gets
//! that back for free and gets bit-identical `ω`.
//!
//! **`step_many` is not faster than the hand-written loop** — a wash, or a few percent
//! worse. Said plainly because the opposite is the easy assumption: what `step_many`
//! actually buys is contract, not throughput. One `dt` validation for the whole slice,
//! no `Result` per body, and a batch-level skip count instead of an error the caller has
//! to fold. Anyone reaching for it expecting a speed-up should reach for the loop
//! instead and lose nothing.
//!
//! **The adaptive substep rule beats a fixed worst-case count.** Four substeps cost
//! 2.2–2.4× one, not 4× — roughly a third of a body's cost is fixed and two thirds is
//! per-substep — so forcing every body to the count the fastest one needs would roughly
//! double a workload in which most bodies need one substep. The per-body branch that
//! chooses the count is cheaper than the work it avoids.
//!
//! **LLVM does auto-vectorize, partially and within a body.** Disassembling
//! `step_many` from `--emit=asm` gives 32 packed SSE ops (`mulpd`/`addpd`/`subpd`)
//! against 101 scalar ones — about a quarter of the arithmetic paired 2-wide, which is
//! LLVM finding pairs inside the 3-vector and quaternion expressions. There is **no**
//! cross-body vectorization and there will not be without a structure-of-arrays layout,
//! because a `RigidBodyRotation` is 176 bytes of interleaved fields and each body's
//! substep count differs.
//!
//! **And none of it is worth tuning further.** 520 bodies at 62 µs is 1.5% of a 4.17 ms
//! frame at 240 Hz. The Dzhanibekov test is the deliverable; this file exists to say
//! what the cost is, not to chase it.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rs_physics::rotational_dynamics::{inertia_3d, InertiaTensor, RigidBodyRotation};

/// Ridgeline's measured debris count. Named rather than inlined so the number has one home.
const BODIES: usize = 520;

/// 240 Hz — the frame rate the caller targets.
const DT: f64 = 1.0 / 240.0;

/// A spread of limb-sized boxes with distinct principal moments, so every one of them is
/// actually tumbling rather than sitting on a principal axis.
fn debris(count: usize) -> Vec<RigidBodyRotation> {
    (0..count)
        .map(|i| {
            let t = i as f64 / count as f64;
            let tensor = inertia_3d::solid_cuboid(
                0.5 + 2.0 * t,
                0.06 + 0.05 * t,
                0.20 + 0.10 * t,
                0.05 + 0.03 * t,
            );
            let mut body = RigidBodyRotation::new(tensor).expect("box tensors are realizable");
            body.set_angular_velocity_body((
                4.0 + 3.0 * t,
                0.7 - 1.4 * t,
                11.0 - 6.0 * t,
            ))
            .expect("finite");
            body
        })
        .collect()
}

/// One RK4 step of `ω̇ = −I⁻¹(ω × Iω)`, orientation left alone, parameterised on where
/// `I⁻¹` comes from. Identical arithmetic in both arms except for that.
#[inline(always)]
fn omega_only_step(
    inertia: &InertiaTensor,
    inverse: Option<&InertiaTensor>,
    w: (f64, f64, f64),
    h: f64,
) -> (f64, f64, f64) {
    let deriv = |w: (f64, f64, f64)| {
        let l = inertia.multiply_vector(w);
        let gyro = (
            -(w.1 * l.2 - w.2 * l.1),
            -(w.2 * l.0 - w.0 * l.2),
            -(w.0 * l.1 - w.1 * l.0),
        );
        match inverse {
            // A field read.
            Some(inv) => inv.multiply_vector(gyro),
            // A fresh 3x3 inversion, which is what `apply_inverse_to_torque` does.
            None => inertia.apply_inverse_to_torque(gyro),
        }
    };
    let k1 = deriv(w);
    let k2 = deriv((w.0 + 0.5 * h * k1.0, w.1 + 0.5 * h * k1.1, w.2 + 0.5 * h * k1.2));
    let k3 = deriv((w.0 + 0.5 * h * k2.0, w.1 + 0.5 * h * k2.1, w.2 + 0.5 * h * k2.2));
    let k4 = deriv((w.0 + h * k3.0, w.1 + h * k3.1, w.2 + h * k3.2));
    let s = h / 6.0;
    (
        w.0 + s * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0),
        w.1 + s * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1),
        w.2 + s * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2),
    )
}

fn bench_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("rigid_body_rotation");
    group.throughput(criterion::Throughput::Elements(BODIES as u64));

    let seed = debris(BODIES);

    group.bench_function("step_many/520", |b| {
        let mut bodies = seed.clone();
        b.iter(|| {
            RigidBodyRotation::step_many(black_box(&mut bodies), black_box(DT)).unwrap()
        });
    });

    group.bench_function("step_loop/520", |b| {
        let mut bodies = seed.clone();
        b.iter(|| {
            let slice: &mut [RigidBodyRotation] = black_box(&mut bodies);
            for body in slice.iter_mut() {
                body.step(black_box(DT)).unwrap();
            }
        });
    });

    group.bench_function("step_angular_velocity_loop/520", |b| {
        let mut bodies = seed.clone();
        b.iter(|| {
            let slice: &mut [RigidBodyRotation] = black_box(&mut bodies);
            for body in slice.iter_mut() {
                body.step_angular_velocity(black_box(DT)).unwrap();
            }
        });
    });

    let tensors: Vec<InertiaTensor> = seed.iter().map(|b| *b.inertia()).collect();
    let inverses: Vec<InertiaTensor> = tensors.iter().map(|t| t.inverse()).collect();

    group.bench_function("omega_only/cached_inverse/520", |b| {
        let mut velocities: Vec<(f64, f64, f64)> =
            seed.iter().map(|b| b.angular_velocity_body()).collect();
        b.iter(|| {
            let v: &mut [(f64, f64, f64)] = black_box(&mut velocities);
            for ((t, inv), w) in tensors.iter().zip(inverses.iter()).zip(v.iter_mut()) {
                *w = omega_only_step(t, Some(inv), *w, black_box(DT));
            }
        });
    });

    group.bench_function("omega_only/recomputed_inverse/520", |b| {
        let mut velocities: Vec<(f64, f64, f64)> =
            seed.iter().map(|b| b.angular_velocity_body()).collect();
        b.iter(|| {
            let v: &mut [(f64, f64, f64)] = black_box(&mut velocities);
            for (t, w) in tensors.iter().zip(v.iter_mut()) {
                *w = omega_only_step(t, None, *w, black_box(DT));
            }
        });
    });

    group.finish();
}

fn bench_substeps(c: &mut Criterion) {
    let mut group = c.benchmark_group("rigid_body_rotation_substeps");

    // |ω|·dt / MAX_STEP_RADIANS decides the count. At 240 Hz a 40 rad/s body wants one
    // substep and a 160 rad/s body wants four.
    for (name, rate) in [("substeps_1", 40.0_f64), ("substeps_4", 160.0)] {
        group.bench_function(name, |b| {
            let mut body =
                RigidBodyRotation::new(inertia_3d::solid_cuboid(1.28, 0.09, 0.26, 0.07)).unwrap();
            body.set_angular_velocity_body((rate * 0.3, rate * 0.1, rate * 0.94))
                .unwrap();
            b.iter(|| black_box(body.step(black_box(DT)).unwrap()));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_batch, bench_substeps);
criterion_main!(benches);
