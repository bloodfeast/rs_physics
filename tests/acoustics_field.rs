//! **L4** (field rays are specular, each against its own analytic path), **L5** (the mean
//! free path converges on 4V/S; RT60 against Eyring on the true V/S) and **L12** (empty
//! inputs are valid and well defined).
#![cfg(feature = "gpu")]

mod acoustics_common;
use acoustics_common::*;

use rs_physics::acoustics::Air;
use rs_physics::gpu::acoustics::*;

const U: f64 = 1.0 / (1u64 << 24) as f64;

/// A shoebox of inner size (lx, ly, lz) with its floor corner at `lo`, as six 0.5 m walls,
/// over a flat terrain far below it.
struct Shoebox {
    lo: [f64; 3],
    size: [f64; 3],
}

const WALL: f64 = 0.5;

impl Shoebox {
    fn scene(&self) -> Scene {
        let mut s = Scene::flat(140, 100, 2.0, -100.0);
        s.materials = vec![AcousticMaterial { reflection: [0.95, 0.9, 0.85, 0.8] }];
        let [x0, y0, z0] = self.lo;
        let [lx, ly, lz] = self.size;
        let (cx, cy, cz) = (x0 + lx / 2.0, y0 + ly / 2.0, z0 + lz / 2.0);
        let (ox, oy, oz) = (lx + 2.0 * WALL, ly + 2.0 * WALL, lz + 2.0 * WALL);
        let walls = [
            ([x0 - WALL / 2.0, cy, cz], [WALL, oy, oz]),
            ([x0 + lx + WALL / 2.0, cy, cz], [WALL, oy, oz]),
            ([cx, y0 - WALL / 2.0, cz], [ox, WALL, oz]),
            ([cx, y0 + ly + WALL / 2.0, cz], [ox, WALL, oz]),
            ([cx, cy, z0 - WALL / 2.0], [ox, oy, WALL]),
            ([cx, cy, z0 + lz + WALL / 2.0], [ox, oy, WALL]),
        ];
        s.statics = walls
            .iter()
            .map(|(c, sz)| Obb::upright(c.map(|v| v as f32), sz.map(|v| v as f32), 0.0, 0))
            .collect();
        s
    }

    /// The analytic first hit from `o` along unit `d`: distance, the axis hit, and the
    /// reflected direction.
    fn hit(&self, o: [f64; 3], d: [f64; 3]) -> (f64, usize, [f64; 3]) {
        let mut best = (f64::INFINITY, 0usize);
        for k in 0..3 {
            if d[k] > 0.0 {
                best = best.min_by_t(((self.lo[k] + self.size[k] - o[k]) / d[k], k));
            } else if d[k] < 0.0 {
                best = best.min_by_t(((self.lo[k] - o[k]) / d[k], k));
            }
        }
        let mut r = d;
        r[best.1] = -r[best.1];
        (best.0, best.1, r)
    }

    fn mfp(&self) -> f64 {
        let [a, b, c] = self.size;
        4.0 * a * b * c / (2.0 * (a * b + b * c + a * c))
    }
}

trait MinByT {
    fn min_by_t(self, o: (f64, usize)) -> (f64, usize);
}

impl MinByT for (f64, usize) {
    fn min_by_t(self, o: (f64, usize)) -> (f64, usize) {
        if o.0 < self.0 { o } else { self }
    }
}

/// Bound on the GPU's hit distance against the analytic one, "within f32 of the geometry".
///
/// The wall is axis-aligned, so the shader's inverse is `1 / s` per axis to four roundings
/// (a product in the cofactor, one in the determinant, the reciprocal and the product),
/// and its translation one more. The local coordinate `inv o + inv_w` is two roundings on
/// terms of size `|o| / s` and `|c| / s`, the local direction one; the slab's `(+-0.5 - o) /
/// d` a subtraction and a 2.5 ulp division. The ray's origin after a bounce carries the
/// first hit's error `e0` (and its 1 mm offset, exact in the analytic path).
fn hit_bound(o: [f64; 3], d: [f64; 3], k: usize, t: f64, wall: f64, center: f64, e0: f64) -> f64 {
    let inv = 1.0 / wall;
    let local = (o[k].abs() + center.abs()) * inv;
    let num = 8.0 * U * (local + 0.5) + e0 * inv;
    let dl = (d[k] * inv).abs();
    num / dl + t * 12.0 * U + 4.0 * U * t
}

#[test]
fn l4_l5_field_rays_are_specular_and_the_mean_free_path_is_4v_over_s() {
    let Some(gpu) = gpu() else { return };
    let boxes = [
        Shoebox { lo: [100.0, 0.0, 80.0], size: [12.0, 5.0, 8.0] },
        Shoebox { lo: [40.0, -3.0, 30.0], size: [30.0, 9.0, 21.0] },
        Shoebox { lo: [200.0, 2.0, 150.0], size: [6.0, 3.0, 6.0] },
    ];
    let air = Air::standard();
    let dirs = field_directions();
    let mut rng = Rng(0xf1e1d);
    let (mut worst_t, mut worst_dir) = (0.0f64, 0.0f64);
    for (bi, b) in boxes.iter().enumerate() {
        let scene = b.scene();
        let mut ac = GpuAcoustics::new(&gpu.device, AcousticLimits::MAX).unwrap();
        scene.load(&gpu, &mut ac);
        for trial in 0..4 {
            let o = [0, 1, 2].map(|k| b.lo[k] + b.size[k] * rng.range(0.15, 0.85));
            let o32 = o.map(|v| v as f32);
            let header = DispatchHeader::new(&listener_at(o32), &air, &[]).unwrap();
            let out = dispatch(&gpu, &mut ac, &header, &[], &[]);
            let rays = field_rays(&gpu, &ac);
            let o = o32.map(|v| v as f64);
            let range = TAP_BIN_S as f64 * TAP_BINS as f64 * header.listener[3] as f64 / 2.0;
            // ---- L4: each ray against the analytic specular path for its own direction.
            let mut sums = (0.0, 0.0, 0.0f64);
            let mut per_ray = Vec::new();
            for (r, dw) in rays.iter().zip(dirs.iter()) {
                let d = [dw[0] as f32 as f64, dw[1] as f32 as f64, dw[2] as f32 as f64];
                let (t, k, refl) = b.hit(o, d);
                if t > range {
                    assert!(r.first_m < 0.0, "a ray past the range hit at {}", r.first_m);
                    continue;
                }
                let face = if d[k] > 0.0 { b.lo[k] + b.size[k] + WALL / 2.0 } else { b.lo[k] - WALL / 2.0 };
                let e1 = hit_bound(o, d, k, t, WALL, face, 0.0);
                let dt = (r.first_m as f64 - t).abs();
                assert!(dt <= e1, "box {bi} trial {trial}: first hit {} against {t} (bound {e1:e})", r.first_m);
                worst_t = worst_t.max(dt / e1);
                for j in 0..3 {
                    let bound = 60.0 * U * d[j].abs() + 1e-30;
                    let dd = (r.first_dir[j] as f64 - refl[j]).abs();
                    assert!(dd <= bound, "box {bi}: reflected {:?} against {refl:?}", r.first_dir);
                    worst_dir = worst_dir.max(dd / bound);
                }
                // The second leg, from the first hit less its millimetre offset.
                let mut p1 = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
                p1[k] -= d[k].signum() * 1e-3;
                let (t2, k2, _) = b.hit(p1, refl);
                if t2 <= range && r.second_m >= 0.0 {
                    let face2 = if refl[k2] > 0.0 { b.lo[k2] + b.size[k2] + WALL / 2.0 } else { b.lo[k2] - WALL / 2.0 };
                    let e2 = hit_bound(p1, refl, k2, t2, WALL, face2, e1 + 4.0 * U * p1[k].abs());
                    let d2 = (r.second_m as f64 - t2).abs();
                    assert!(d2 <= e2, "box {bi}: second hit {} against {t2} (bound {e2:e})", r.second_m);
                }
                // L5's GPU quadrature, from the rays' own records.
                let cos = d[k].abs();
                assert!((r.first_cos as f64 - cos).abs() <= 64.0 * U, "cos {} against {cos}", r.first_cos);
                let (a, s) = (dw[3] * t * t * t, dw[3] * t * t / cos);
                sums.0 += a;
                sums.1 += s;
                per_ray.push((a, s));
            }
            assert_eq!(out.field.clear_fraction, 0.0, "a closed box let a ray out");
            check_taps(&out.field, &rays);
            // ---- L5, the GPU's 64 rays: 4V/S within three standard errors of its own
            // sample, the error reported.
            let n = per_ray.len() as f64;
            let ratio = sums.0 / sums.1;
            let var: f64 = per_ray.iter().map(|(a, s)| (a - ratio * s).powi(2)).sum::<f64>() * n / (n - 1.0);
            let se = (4.0 / 3.0) * var.sqrt() / sums.1;
            let truth = b.mfp();
            let gpu_mfp = out.field.mfp_m as f64;
            println!(
                "box {bi} trial {trial}: mfp {gpu_mfp:.4} m against 4V/S {truth:.4} m: error {:.4} m = {:.2} standard errors ({se:.4} m)",
                gpu_mfp - truth,
                (gpu_mfp - truth).abs() / se,
            );
            assert!((gpu_mfp - truth).abs() <= 3.0 * se, "mfp off by more than three standard errors");
            // RT60 against Eyring on the true V/S, within what the mfp error implies.
            let r500 = scene.materials[0].reflection[1] as f64;
            let alpha = 1.0 - r500 * r500;
            let c = header.listener[3] as f64;
            let eyring = 24.0 * 10f64.ln() / c * (truth / 4.0) / -(1.0 - alpha).ln();
            let rt = out.field.rt60_s as f64;
            assert!(
                (rt - eyring).abs() <= eyring * 3.0 * se / truth + 1e-5 * eyring,
                "rt60 {rt} against Eyring {eyring}"
            );
        }
    }
    println!("L4: worst first-hit {worst_t:.3}, worst reflected direction {worst_dir:.3} of their bounds");
}

/// L5's convergence half: the same estimator (V = sum w l^3 / 3, S = sum w l^2 / |cos|) on
/// uniformly random directions, in f64, closes on 4V/S as the ray count grows.
#[test]
fn l5_the_quadrature_converges_on_4v_over_s() {
    let b = Shoebox { lo: [0.0, 0.0, 0.0], size: [12.0, 5.0, 8.0] };
    let o = [4.3, 1.7, 3.1];
    let truth = b.mfp();
    let mut rng = Rng(0xc0ffee);
    let mut last_se = f64::INFINITY;
    for n in [64usize, 256, 1024, 4096] {
        let mut per = Vec::with_capacity(n);
        for _ in 0..n {
            let z = rng.range(-1.0, 1.0);
            let phi = rng.range(0.0, std::f64::consts::TAU);
            let rr = (1.0 - z * z).sqrt();
            let d = [rr * phi.cos(), z, rr * phi.sin()];
            let (t, k, _) = b.hit(o, d);
            per.push((t * t * t, t * t / d[k].abs()));
        }
        let (a, s): (f64, f64) = per.iter().fold((0.0, 0.0), |acc, p| (acc.0 + p.0, acc.1 + p.1));
        let ratio = a / s;
        let var: f64 = per.iter().map(|(x, y)| (x - ratio * y).powi(2)).sum::<f64>() * n as f64 / (n as f64 - 1.0);
        let se = (4.0 / 3.0) * var.sqrt() / s;
        let mfp = 4.0 * a / (3.0 * s);
        println!("{n:>5} rays: mfp {mfp:.4} against {truth:.4}: error {:+.4} m, standard error {se:.4} m", mfp - truth);
        assert!((mfp - truth).abs() <= 3.0 * se, "{n} rays: off by more than three standard errors");
        assert!(se < last_se, "the standard error did not fall with the ray count");
        last_se = se;
    }
}

/// L12: no scene, no sources, no statics, no movers: valid bindings and defined answers.
#[test]
fn l12_empty_inputs_dispatch_and_answer() {
    let Some(gpu) = gpu() else { return };
    let air = Air::standard();
    // Before any scene: nothing to hit.
    let mut ac = GpuAcoustics::new(&gpu.device, AcousticLimits::MAX).unwrap();
    let header = DispatchHeader::new(&listener_at([10.0, 1.6, 10.0]), &air, &[]).unwrap();
    let out = dispatch(&gpu, &mut ac, &header, &[], &[]);
    assert_eq!(out.len, 0);
    assert!(out.results().is_empty());
    assert_eq!((out.field.mfp_m, out.field.rt60_s, out.field.clear_fraction), (0.0, 0.0, 1.0));
    assert!(field_rays(&gpu, &ac).iter().all(|r| r.first_m < 0.0));
    // A source with no scene: spreading only, and no edge.
    let s = Source::new([20.0, 1.6, 10.0], 1.0, [0.0; 3], 7, NO_MOVER).unwrap();
    let out = dispatch(&gpu, &mut ac, &header, &[s], &[]);
    assert_eq!(out.len, 1);
    assert_eq!(out.sources[0].excess_m, f32::MIN);
    assert!(!out.sources[0].blocked() && !out.sources[0].diffracted());

    // A flat terrain with no statics and no movers: rays hit the ground or escape.
    let scene = Scene::flat(40, 40, 2.0, 0.0);
    scene.load(&gpu, &mut ac);
    let o = [40.0f32, 1.6, 40.0];
    let header = DispatchHeader::new(&listener_at(o), &air, &[]).unwrap();
    let out = dispatch(&gpu, &mut ac, &header, &[], &[]);
    assert_eq!(out.len, 0);
    let range = TAP_BIN_S as f64 * TAP_BINS as f64 * header.listener[3] as f64 / 2.0;
    let rays = field_rays(&gpu, &ac);
    let mut hits = 0;
    for (r, d) in rays.iter().zip(field_directions().iter()) {
        let dy = d[1] as f32 as f64;
        if dy < 0.0 {
            let t = 1.6 / -dy;
            // The terrain is 80 m wide; a ray that leaves it first escapes.
            let x = o[0] as f64 + t * d[0];
            let z = o[2] as f64 + t * d[2];
            let inside = (0.0..80.0).contains(&x) && (0.0..80.0).contains(&z);
            if t <= range && inside {
                assert!((r.first_m as f64 - t).abs() < 1e-3, "a ground ray hit at {} against {t}", r.first_m);
                hits += 1;
            }
        } else {
            assert!(r.first_m < 0.0, "an upward ray hit something with no statics");
        }
    }
    assert!(hits > 0);
    assert!(out.field.clear_fraction > 0.0 && out.field.clear_fraction < 1.0);
}

/// The taps against a CPU selection over the rays' own arrival records: the strongest
/// arrival in each 5 ms bin, the eight strongest bins (ties to the earlier bin), in delay
/// order, the earliest-indexed arrival on a tie within a bin.
fn check_taps(field: &ListenerField, rays: &[FieldRay]) {
    let mut arrivals = Vec::new();
    for (i, r) in rays.iter().enumerate() {
        if r.first_m >= 0.0 {
            arrivals.push((2 * i, r.first_arrival));
            if r.second_m >= 0.0 {
                arrivals.push((2 * i + 1, r.second_arrival));
            }
        }
    }
    let mut bins: Vec<Option<(usize, [f32; 3])>> = vec![None; TAP_BINS as usize];
    for &(idx, a) in &arrivals {
        let bin = (a[0] / TAP_BIN_S) as usize;
        if a[1] <= 0.0 || bin >= TAP_BINS as usize {
            continue;
        }
        let better = match bins[bin] {
            None => true,
            Some((j, b)) => a[1] > b[1] || (a[1] == b[1] && idx < j),
        };
        if better {
            bins[bin] = Some((idx, a));
        }
    }
    let mut order: Vec<usize> = (0..bins.len()).filter(|&b| bins[b].is_some()).collect();
    order.sort_by(|&x, &y| {
        let (gx, gy) = (bins[x].unwrap().1[1], bins[y].unwrap().1[1]);
        gy.partial_cmp(&gx).unwrap().then(x.cmp(&y))
    });
    let mut chosen: Vec<usize> = order.into_iter().take(8).collect();
    chosen.sort();
    for (slot, tap) in field.taps.iter().enumerate() {
        match chosen.get(slot) {
            Some(&b) => {
                let a = bins[b].unwrap().1;
                assert_eq!([tap.delay_s, tap.gain, tap.pan], [a[0], a[1], a[2].clamp(-1.0, 1.0)], "tap {slot}");
            }
            None => assert_eq!(tap.gain, 0.0, "tap {slot} should be empty"),
        }
    }
}
