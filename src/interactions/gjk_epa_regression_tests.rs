//! Regression tests for the GJK/EPA narrow phase, checked against analytic answers
//! (`docs/reviews/2026-09-29-correctness-performance.md`).
//!
//! Every test here failed (or panicked) on the code as reviewed; `box_vs_box_matches_sat_oracle`
//! passed and is a guard that the fixes do not regress the case that already worked.

use crate::interactions::gjk_collision_3d::{
    epa_contact_points_ex, get_support_point_for_shape, gjk_collision_detection_ex, ContactInfo,
    GjkResult,
};
use crate::models::{PhysicalObject3D, Quaternion, Shape3D};
use crate::utils::PhysicsConstants;
use crate::world::{PhysicsWorld, WorldConfig};

type V = (f64, f64, f64);

fn add(a: V, b: V) -> V { (a.0 + b.0, a.1 + b.1, a.2 + b.2) }
fn sub(a: V, b: V) -> V { (a.0 - b.0, a.1 - b.1, a.2 - b.2) }
fn dot(a: V, b: V) -> f64 { a.0 * b.0 + a.1 * b.1 + a.2 * b.2 }
fn scale(a: V, s: f64) -> V { (a.0 * s, a.1 * s, a.2 * s) }
fn cross(a: V, b: V) -> V {
    (a.1 * b.2 - a.2 * b.1, a.2 * b.0 - a.0 * b.2, a.0 * b.1 - a.1 * b.0)
}
fn len(a: V) -> f64 { dot(a, a).sqrt() }
fn norm(a: V) -> V { scale(a, 1.0 / len(a)) }

/// Deterministic xorshift64* so every failure is reproducible from its seed.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn uniform(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + (hi - lo) * self.uniform() }
    fn unit(&mut self) -> V {
        loop {
            let v = (self.range(-1.0, 1.0), self.range(-1.0, 1.0), self.range(-1.0, 1.0));
            let l = len(v);
            if l > 0.1 && l <= 1.0 { return scale(v, 1.0 / l); }
        }
    }
    fn quat(&mut self) -> Quaternion {
        Quaternion::from_axis_angle(self.unit(), self.range(0.0, std::f64::consts::TAU))
    }
}

/// Local axes of a rotation in world space, via the crate's own `rotate_point`.
fn axes(q: Quaternion) -> [V; 3] {
    [q.rotate_point((1.0, 0.0, 0.0)), q.rotate_point((0.0, 1.0, 0.0)), q.rotate_point((0.0, 0.0, 1.0))]
}

/// Exact signed distance from `p` to an oriented box (negative inside) and the outward normal
/// at the closest boundary point.
fn point_obb_signed(p: V, c: V, q: Quaternion, half: V) -> (f64, V) {
    let ax = axes(q);
    let d = sub(p, c);
    let l = [dot(d, ax[0]), dot(d, ax[1]), dot(d, ax[2])];
    let h = [half.0, half.1, half.2];
    let mut outside = [0.0; 3];
    let mut any_out = false;
    for i in 0..3 {
        let e = l[i].abs() - h[i];
        if e > 0.0 { outside[i] = e * l[i].signum(); any_out = true; }
    }
    if any_out {
        let dist = (outside[0] * outside[0] + outside[1] * outside[1] + outside[2] * outside[2]).sqrt();
        let n = add(add(scale(ax[0], outside[0] / dist), scale(ax[1], outside[1] / dist)), scale(ax[2], outside[2] / dist));
        (dist, n)
    } else {
        let (mut best, mut bi) = (f64::INFINITY, 0);
        for i in 0..3 {
            let e = h[i] - l[i].abs();
            if e < best { best = e; bi = i; }
        }
        (-best, scale(ax[bi], if l[bi] >= 0.0 { 1.0 } else { -1.0 }))
    }
}

/// Exact signed distance from `p` to a cylinder whose axis is local Y (the crate's convention).
fn point_cyl_signed(p: V, c: V, q: Quaternion, radius: f64, height: f64) -> (f64, V) {
    let ax = axes(q);
    let d = sub(p, c);
    let l = (dot(d, ax[0]), dot(d, ax[1]), dot(d, ax[2]));
    let rho = (l.0 * l.0 + l.2 * l.2).sqrt();
    let dr = rho - radius;
    let dy = l.1.abs() - height / 2.0;
    let radial = if rho > 1e-15 { norm(add(scale(ax[0], l.0), scale(ax[2], l.2))) } else { ax[0] };
    let axial = scale(ax[1], l.1.signum());
    if dr > 0.0 || dy > 0.0 {
        let (a, b) = (dr.max(0.0), dy.max(0.0));
        let dist = (a * a + b * b).sqrt();
        (dist, norm(add(scale(radial, a), scale(axial, b))))
    } else if dr > dy {
        (dr, radial)
    } else {
        (dy, axial)
    }
}

/// SAT over the 15 axes of two OBBs: the minimum overlap is the exact penetration depth.
fn obb_obb_sat(ca: V, qa: Quaternion, ha: V, cb: V, qb: Quaternion, hb: V) -> (f64, V) {
    let (a, b) = (axes(qa), axes(qb));
    let (hav, hbv) = ([ha.0, ha.1, ha.2], [hb.0, hb.1, hb.2]);
    let t = sub(cb, ca);
    let mut cands: Vec<V> = vec![a[0], a[1], a[2], b[0], b[1], b[2]];
    for i in 0..3 {
        for j in 0..3 {
            let c = cross(a[i], b[j]);
            if len(c) > 1e-9 { cands.push(norm(c)); }
        }
    }
    let (mut best, mut best_axis) = (f64::INFINITY, (0.0, 0.0, 0.0));
    for n in cands {
        let ra: f64 = (0..3).map(|i| hav[i] * dot(a[i], n).abs()).sum();
        let rb: f64 = (0..3).map(|i| hbv[i] * dot(b[i], n).abs()).sum();
        let dist = dot(t, n);
        let overlap = ra + rb - dist.abs();
        if overlap < best {
            best = overlap;
            best_axis = if dist >= 0.0 { n } else { scale(n, -1.0) };
        }
    }
    (best, best_axis)
}

fn run(s1: &Shape3D, p1: V, q1: Quaternion, s2: &Shape3D, p2: V, q2: Quaternion) -> (bool, Option<ContactInfo>) {
    let g = gjk_collision_detection_ex(s1, p1, q1, s2, p2, q2);
    let hit = !matches!(g, GjkResult::NoCollision);
    (hit, epa_contact_points_ex(s1, p1, q1, s2, p2, q2, &g))
}

/// Sphere (shape 1) against `make`'s shape (shape 2) placed near contact at random poses.
/// Returns human-readable failures: missed overlap, missing contact, wrong depth, NaN, panic.
fn sphere_vs_oracle_failures(
    seed: u64,
    n: usize,
    scale_s: f64,
    make: &dyn Fn(&mut Rng) -> (Shape3D, Box<dyn Fn(V, V, Quaternion) -> (f64, V)>),
) -> Vec<String> {
    let mut rng = Rng(seed);
    let mut failures = Vec::new();
    for _ in 0..n {
        let (shape, oracle) = make(&mut rng);
        let qb = rng.quat();
        let cb = (rng.range(-1.0, 1.0) * scale_s, rng.range(-1.0, 1.0) * scale_s, rng.range(-1.0, 1.0) * scale_s);
        let r = rng.range(0.05, 1.5) * scale_s;
        let mut cs = add(cb, scale(rng.unit(), rng.range(0.0, 3.0) * scale_s));
        for _ in 0..3 {
            let (sd, nrm) = oracle(cs, cb, qb);
            let target = r + rng.range(-0.5 * r, 0.2 * scale_s);
            cs = add(cs, scale(nrm, target - sd));
        }
        let (sd, nrm) = oracle(cs, cb, qb);
        let gap = sd - r;
        // Skip the touching band and centre-inside cases (the oracle depth formula needs sd >= 0).
        if gap.abs() < 1e-7 * scale_s || sd < 0.0 || gap > 0.0 {
            continue;
        }
        let s1 = Shape3D::Sphere(r);
        let res = std::panic::catch_unwind(|| run(&s1, cs, Quaternion::identity(), &shape, cb, qb));
        let tag = format!("r={r:?} cs={cs:?} cb={cb:?} qb={qb:?} shape={shape:?}");
        match res {
            Err(_) => failures.push(format!("PANIC {tag}")),
            Ok((false, _)) => failures.push(format!("MISSED overlap {} {tag}", -gap)),
            Ok((true, None)) => failures.push(format!("NO CONTACT for overlap {} {tag}", -gap)),
            Ok((true, Some(c))) => {
                let exp_n = scale(nrm, -1.0); // sphere -> shape
                let finite = c.penetration.is_finite() && c.normal.0.is_finite() && c.normal.1.is_finite() && c.normal.2.is_finite();
                if !finite || (c.penetration + gap).abs() > 2e-3 * (scale_s - gap) || dot(c.normal, exp_n) < 0.99 {
                    failures.push(format!("WRONG depth {} (expected {}) normal {:?} (expected {:?}) {tag}", c.penetration, -gap, c.normal, exp_n));
                }
            }
        }
    }
    failures
}

fn report(label: &str, failures: &[String]) {
    assert!(
        failures.is_empty(),
        "{label}: {} failures; first: {}",
        failures.len(),
        failures.iter().take(3).cloned().collect::<Vec<_>>().join("\n  ")
    );
}

// ---------------------------------------------------------------------------
// EPA face orientation / degenerate GJK simplex
// ---------------------------------------------------------------------------

/// A sphere centred on a cube's body diagonal. GJK's first two support points lie on the
/// diagonal on either side of the origin, so the tetrahedron it hands EPA has the origin on an
/// edge. EPA orients faces by the sign of the origin distance (here ~1e-16), turns one inward,
/// deletes every face, and indexes an empty Vec.
#[test]
fn epa_sphere_on_box_body_diagonal_does_not_panic() {
    let r = 0.2 * 3f64.sqrt() + 0.1; // exactly 0.1 deep at the corner (1,1,1)
    let res = std::panic::catch_unwind(|| {
        run(&Shape3D::Sphere(r), (1.2, 1.2, 1.2), Quaternion::identity(),
            &Shape3D::Cuboid(2.0, 2.0, 2.0), (0.0, 0.0, 0.0), Quaternion::identity())
    });
    let (hit, contact) = res.expect("EPA panicked on a sphere resting on a cube corner");
    assert!(hit);
    let c = contact.expect("overlapping sphere and box produced no contact");
    assert!((c.penetration - 0.1).abs() < 1e-3, "depth {} expected 0.1", c.penetration);
    let expected = norm((-1.0, -1.0, -1.0));
    assert!(dot(c.normal, expected) > 0.999, "normal {:?} expected {:?}", c.normal, expected);
}

/// A ball against the side of an upright cylinder. The mirror plane through the cylinder axis
/// and the ball centre makes GJK's first triangle coplanar with the origin, EPA flips that face
/// inward, and the reported depth is 16x the truth with a normal perpendicular to the true one.
#[test]
fn epa_sphere_beside_upright_cylinder_depth_is_analytic() {
    let cs = (1.0606601717798214, 0.30000000000000004, 1.0606601717798212); // 1.5 m out at 45 deg
    let (hit, contact) = run(&Shape3D::Sphere(0.6), cs, Quaternion::identity(),
                             &Shape3D::Cylinder(1.0, 2.0), (0.0, 0.0, 0.0), Quaternion::identity());
    assert!(hit);
    let c = contact.expect("no contact");
    assert!((c.penetration - 0.1).abs() < 1e-3, "depth {} expected 0.1 (normal {:?})", c.penetration, c.normal);
    let expected = norm((-1.0, 0.0, -1.0));
    assert!(dot(c.normal, expected) > 0.999, "normal {:?} expected {:?}", c.normal, expected);
}

/// Randomised sphere-vs-cylinder against the analytic distance. About 10% of overlapping
/// contacts come back with a grossly wrong depth/normal on the reviewed code.
#[test]
fn sphere_vs_cylinder_matches_analytic_oracle() {
    let failures = sphere_vs_oracle_failures(12, 1500, 1.0, &|rng: &mut Rng| {
        let (rad, h) = (rng.range(0.1, 2.0), rng.range(0.1, 3.0));
        (Shape3D::Cylinder(rad, h), Box::new(move |p, c, q| point_cyl_signed(p, c, q, rad, h)))
    });
    report("sphere vs cylinder", &failures);
}

/// Randomised sphere-vs-rotated-box against the analytic distance (corner Voronoi regions are
/// where it breaks: the Minkowski difference is locally a sphere around a fixed vertex).
#[test]
fn sphere_vs_rotated_box_matches_analytic_oracle() {
    let failures = sphere_vs_oracle_failures(11, 8000, 1.0, &|rng: &mut Rng| {
        let half = (rng.range(0.1, 2.0), rng.range(0.1, 2.0), rng.range(0.1, 2.0));
        (Shape3D::Cuboid(2.0 * half.0, 2.0 * half.1, 2.0 * half.2),
         Box::new(move |p, c, q| point_obb_signed(p, c, q, half)))
    });
    report("sphere vs rotated box", &failures);
}

/// Same hull as a box, but described with coplanar face/edge points and a duplicate vertex.
#[test]
fn sphere_vs_hull_with_coplanar_and_duplicate_points_matches_box_oracle() {
    let failures = sphere_vs_oracle_failures(13, 8000, 1.0, &|rng: &mut Rng| {
        let half = (rng.range(0.1, 2.0), rng.range(0.1, 2.0), rng.range(0.1, 2.0));
        let mut v = Vec::new();
        for sx in [-1.0, 0.0, 1.0] {
            for sy in [-1.0, 0.0, 1.0] {
                for sz in [-1.0, 0.0, 1.0] {
                    if (sx, sy, sz) != (0.0, 0.0, 0.0) { v.push((sx * half.0, sy * half.1, sz * half.2)); }
                }
            }
        }
        v.push((half.0, half.1, half.2));
        (Shape3D::Polyhedron(v, vec![]), Box::new(move |p, c, q| point_obb_signed(p, c, q, half)))
    });
    report("sphere vs hull", &failures);
}

// ---------------------------------------------------------------------------
// EPA iteration cap -> None (false negative after GJK reported a hit)
// ---------------------------------------------------------------------------

/// Octahedron vertex pointing straight at a sphere. GJK reports the overlap; EPA runs out of
/// iterations refining the locally spherical Minkowski surface and returns None, so the
/// world drops a 0.3 m-deep contact.
#[test]
fn epa_sphere_on_octahedron_vertex_returns_contact() {
    let octa = Shape3D::Polyhedron(
        vec![(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)],
        vec![],
    );
    let (hit, contact) = run(&Shape3D::Sphere(0.5), (1.2, 0.0, 0.0), Quaternion::identity(),
                             &octa, (0.0, 0.0, 0.0), Quaternion::identity());
    assert!(hit, "GJK should report the overlap");
    let c = contact.expect("GJK reported a 0.3 m overlap but EPA returned no contact");
    assert!((c.penetration - 0.3).abs() < 1e-3, "depth {}", c.penetration);
    assert!(c.normal.0 < -0.999, "normal {:?}", c.normal);
}

/// At 100 m scale the absolute EPA_TOLERANCE (1e-6 m) is out of reach in 64 iterations for a
/// curved partner, and EPA returns None for genuinely overlapping pairs.
#[test]
fn sphere_vs_box_at_100m_scale_always_yields_contact() {
    let failures = sphere_vs_oracle_failures(14, 8000, 100.0, &|rng: &mut Rng| {
        let half = (rng.range(10.0, 200.0), rng.range(10.0, 200.0), rng.range(10.0, 200.0));
        (Shape3D::Cuboid(2.0 * half.0, 2.0 * half.1, 2.0 * half.2),
         Box::new(move |p, c, q| point_obb_signed(p, c, q, half)))
    });
    report("sphere vs box at 100 m scale", &failures);
}

// ---------------------------------------------------------------------------
// Guard: box-box already matches SAT exactly and must keep doing so
// ---------------------------------------------------------------------------

#[test]
fn box_vs_box_matches_sat_oracle() {
    let mut rng = Rng(0xdead_beef_cafe_f00d);
    let mut failures = Vec::new();
    for _ in 0..5000 {
        let ha = (rng.range(0.1, 2.0), rng.range(0.1, 2.0), rng.range(0.1, 2.0));
        let hb = (rng.range(0.1, 2.0), rng.range(0.1, 2.0), rng.range(0.1, 2.0));
        let (qa, qb) = (rng.quat(), rng.quat());
        let d = rng.unit();
        let mut cb = scale(d, rng.range(0.0, 5.0));
        let (ov, _) = obb_obb_sat((0.0, 0.0, 0.0), qa, ha, cb, qb, hb);
        let target = rng.range(-0.3, 0.8);
        cb = add(cb, scale(d, ov - target));
        let (ov, _) = obb_obb_sat((0.0, 0.0, 0.0), qa, ha, cb, qb, hb);
        if ov.abs() < 1e-7 { continue; }
        let s1 = Shape3D::Cuboid(2.0 * ha.0, 2.0 * ha.1, 2.0 * ha.2);
        let s2 = Shape3D::Cuboid(2.0 * hb.0, 2.0 * hb.1, 2.0 * hb.2);
        let (hit, c) = run(&s1, (0.0, 0.0, 0.0), qa, &s2, cb, qb);
        if hit != (ov > 0.0) {
            failures.push(format!("hit={hit} but SAT overlap {ov}"));
        } else if hit {
            match c {
                Some(c) if (c.penetration - ov).abs() <= 1e-4 * (1.0 + ov) => {}
                other => failures.push(format!("depth {:?} vs SAT {ov}", other.map(|c| c.penetration))),
            }
        }
    }
    report("box vs box", &failures);
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

/// EPA builds its horizon from a HashMap, so with tied closest faces the chosen normal depends
/// on the per-HashMap random seed: the same call returns (0,1,0) one time and (0,0,1) the next.
/// (Probabilistic on the reviewed code: fails with overwhelming likelihood in 100 calls.)
#[test]
fn epa_is_deterministic_for_symmetric_cube_overlap() {
    let s = Shape3D::Cuboid(1.0, 1.0, 1.0);
    let off = (0.508601565045107, 0.508601565045107, 0.508601565045107);
    let first = run(&s, (0.0, 0.0, 0.0), Quaternion::identity(), &s, off, Quaternion::identity()).1.expect("contact");
    for i in 0..100 {
        let c = run(&s, (0.0, 0.0, 0.0), Quaternion::identity(), &s, off, Quaternion::identity()).1.expect("contact");
        assert!(
            c.normal == first.normal && c.penetration == first.penetration && c.point1 == first.point1,
            "call {i} returned normal {:?}, first call returned {:?}", c.normal, first.normal
        );
    }
}

// ---------------------------------------------------------------------------
// BeveledCuboid support function
// ---------------------------------------------------------------------------

/// The bevel removes material, so no support point may leave the unbevelled w x h x d box.
/// The face case adds `0.1 * bevel` along the direction, and near-corner directions below the
/// 0.3 threshold return the sharp corner.
#[test]
fn beveled_cuboid_support_never_leaves_its_box_and_is_a_valid_support_map() {
    let (w, h, d, b) = (1.0, 1.0, 1.0, 0.1);
    let shape = Shape3D::BeveledCuboid(w, h, d, b);
    let down = get_support_point_for_shape(&shape, (0.0, 0.0, 0.0), Quaternion::identity(), (0.0, -1.0, 0.0));
    assert!((down.1 + 0.5).abs() < 1e-12, "lowest point of the die is y = {} (true bottom face is y = -0.5)", down.1);

    let mut rng = Rng(77);
    let mut pts = Vec::new();
    for _ in 0..2000 {
        let dir = rng.unit();
        let s = get_support_point_for_shape(&shape, (0.0, 0.0, 0.0), Quaternion::identity(), dir);
        let h_box = dir.0.abs() * w / 2.0 + dir.1.abs() * h / 2.0 + dir.2.abs() * d / 2.0;
        assert!(dot(s, dir) <= h_box + 1e-12, "support {s:?} in direction {dir:?} lies outside the unbevelled box");
        pts.push((dir, s));
    }
    // A support map must return the maximiser: no other returned point may beat it along `dir`.
    for (dir, s) in &pts {
        for (_, other) in &pts {
            assert!(dot(*other, *dir) <= dot(*s, *dir) + 1e-9,
                    "support in {dir:?} is {s:?}, but {other:?} is further along it");
        }
    }
}

/// Consequence: a die hovering 5 mm above a table is reported as touching it.
#[test]
fn beveled_die_hovering_above_table_is_not_in_contact() {
    let die = Shape3D::BeveledCuboid(1.0, 1.0, 1.0, 0.1);
    let table = Shape3D::Cuboid(10.0, 1.0, 10.0); // top face at y = 0
    let (hit, c) = run(&die, (0.0, 0.5 + 0.005, 0.0), Quaternion::identity(),
                       &table, (0.0, -0.5, 0.0), Quaternion::identity());
    assert!(!hit, "5 mm gap reported as contact with depth {:?}", c.map(|c| c.penetration));
}

// ---------------------------------------------------------------------------
// World call site
// ---------------------------------------------------------------------------

fn zero_g_world() -> PhysicsWorld {
    let mut cfg = WorldConfig::default();
    cfg.gravity = (0.0, 0.0, 0.0);
    cfg.aerodynamic_drag = false;
    PhysicsWorld::new(cfg)
}

fn body(mass: f64, pos: V, shape: Shape3D) -> PhysicalObject3D {
    PhysicalObject3D::new(mass, (0.0, 0.0, 0.0), pos, shape, None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), PhysicsConstants::default())
}

/// The world gates the narrow phase on `Shape3D::bounding_radius()`, which for a Polyhedron is
/// measured from the vertex centroid; GJK's support function places vertices relative to the
/// body position. A hull whose local origin is not its centroid is under-bounded, and a real,
/// GJK-confirmed overlap is never handed to GJK at all.
#[test]
fn world_resolves_overlap_with_off_centre_polyhedron() {
    let hull = Shape3D::Polyhedron(vec![(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)], vec![]);
    let sphere_pos = (4.3, 0.05, 0.05); // 0.5 m sphere swallowing the (4,0,0) vertex
    let g = gjk_collision_detection_ex(&Shape3D::Sphere(0.5), sphere_pos, Quaternion::identity(),
                                       &hull, (0.0, 0.0, 0.0), Quaternion::identity());
    assert!(!matches!(g, GjkResult::NoCollision), "precondition: GJK sees the overlap");

    let mut world = zero_g_world();
    world.add_object(body(f64::INFINITY, (0.0, 0.0, 0.0), hull));
    let id = world.add_object(body(1.0, sphere_pos, Shape3D::Sphere(0.5)));
    for _ in 0..10 { world.step(); }
    let p = &world.get_object(id).unwrap().object.position;
    assert!(p.x > sphere_pos.0 + 1e-3, "sphere never pushed out of the hull: x = {}", p.x);
}

/// Two spheres spawned at the same point are never separated: the SphereSphere branch returns
/// None when the centres coincide instead of picking a fallback normal
/// (`sphere_sphere_contact` already does this).
#[test]
fn world_separates_coincident_spheres() {
    let mut world = zero_g_world();
    let a = world.add_object(body(1.0, (0.0, 0.0, 0.0), Shape3D::Sphere(0.5)));
    let b = world.add_object(body(1.0, (0.0, 0.0, 0.0), Shape3D::Sphere(0.5)));
    for _ in 0..30 { world.step(); }
    let pa = world.get_object(a).unwrap().object.position.clone();
    let pb = world.get_object(b).unwrap().object.position.clone();
    let sep = ((pa.x - pb.x).powi(2) + (pa.y - pb.y).powi(2) + (pa.z - pb.z).powi(2)).sqrt();
    assert!(sep > 0.5, "coincident spheres still {sep} m apart after 30 steps");
}

// ---------------------------------------------------------------------------
// Performance probe: cargo test --release --lib perf_gjk_epa_per_call -- --ignored --nocapture
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn perf_gjk_epa_per_call() {
    use std::hint::black_box;
    use std::time::Instant;
    let q1 = Quaternion::from_axis_angle((0.3, 1.0, 0.2), 0.7);
    let q2 = Quaternion::from_axis_angle((1.0, 0.1, -0.4), 1.1);
    let cases = vec![
        ("box-box rotated", Shape3D::Cuboid(1.0, 1.0, 1.0), (0.0, 0.0, 0.0), q1, Shape3D::Cuboid(1.0, 1.0, 1.0), (0.8, 0.3, 0.1), q2),
        ("box on ground", Shape3D::Cuboid(1.0, 1.0, 1.0), (0.2, 0.48, 0.1), Quaternion::from_axis_angle((0.0, 1.0, 0.0), 0.4),
         Shape3D::Cuboid(20.0, 1.0, 20.0), (0.0, -0.5, 0.0), Quaternion::identity()),
        ("sphere-box rotated", Shape3D::Sphere(0.5), (0.9, 0.2, 0.1), Quaternion::identity(), Shape3D::Cuboid(1.0, 1.0, 1.0), (0.0, 0.0, 0.0), q2),
        ("cylinder-box", Shape3D::Cylinder(0.4, 1.0), (0.0, 0.95, 0.1), q1, Shape3D::Cuboid(4.0, 1.0, 4.0), (0.0, 0.0, 0.0), Quaternion::identity()),
    ];
    for (name, s1, p1, q1, s2, p2, q2) in cases {
        let n = 200_000;
        let t = Instant::now();
        for _ in 0..n {
            black_box(gjk_collision_detection_ex(black_box(&s1), black_box(p1), q1, &s2, p2, q2));
        }
        let gjk_ns = t.elapsed().as_nanos() as f64 / n as f64;
        let g = gjk_collision_detection_ex(&s1, p1, q1, &s2, p2, q2);
        let t = Instant::now();
        for _ in 0..n {
            black_box(epa_contact_points_ex(black_box(&s1), black_box(p1), q1, &s2, p2, q2, black_box(&g)));
        }
        let epa_ns = t.elapsed().as_nanos() as f64 / n as f64;
        println!("{name:22} GJK {gjk_ns:7.0} ns   EPA {epa_ns:7.0} ns");
    }
    // Support function alone vs. the same computation with the rotation precomputed once.
    let s = Shape3D::Cuboid(1.0, 2.0, 3.0);
    let q = Quaternion::from_axis_angle((0.3, 1.0, 0.2), 0.7);
    let n = 5_000_000;
    let mut acc = (0.0, 0.0, 0.0);
    let t = Instant::now();
    for i in 0..n {
        let d = (1.0, (i & 7) as f64 * 0.1, -0.3);
        acc = add(acc, get_support_point_for_shape(black_box(&s), (0.0, 0.0, 0.0), black_box(q), d));
    }
    let quat_ns = t.elapsed().as_nanos() as f64 / n as f64;
    black_box(acc);
    let ax = axes(q);
    let mut acc = (0.0, 0.0, 0.0);
    let t = Instant::now();
    for i in 0..n {
        let d = black_box((1.0, (i & 7) as f64 * 0.1, -0.3));
        let l = (dot(d, ax[0]), dot(d, ax[1]), dot(d, ax[2]));
        let ls = (if l.0 >= 0.0 { 0.5 } else { -0.5 }, if l.1 >= 0.0 { 1.0 } else { -1.0 }, if l.2 >= 0.0 { 1.5 } else { -1.5 });
        acc = add(acc, add(add(scale(ax[0], ls.0), scale(ax[1], ls.1)), scale(ax[2], ls.2)));
    }
    let mat_ns = t.elapsed().as_nanos() as f64 / n as f64;
    black_box(acc);
    println!("cuboid support: current {quat_ns:.1} ns/call, rotation precomputed {mat_ns:.1} ns/call");
}
