//! Tests for the SPH solids: the swept-ray contact, the binning and the slope-normal
//! ground. A child module of `sph`, so the binning can be checked against a brute-force
//! oracle on the solver's own grid.

use super::*;

const DT: f64 = 1.0 / 240.0;

fn flat(_x: f64, _z: f64) -> f64 {
    0.0
}

fn far_below(_x: f64, _z: f64) -> f64 {
    -100.0
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Distance from `p` to the segment `a`-`b`.
fn segment_distance(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let ab = sub(b, a);
    let t = (dot(sub(p, a), ab) / dot(ab, ab)).clamp(0.0, 1.0);
    let c = [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t];
    dot(sub(p, c), sub(p, c)).sqrt()
}

/// Every position, velocity and density, as bits.
fn bits(f: &SphFluid) -> Vec<u64> {
    let mut out = Vec::new();
    for i in 0..f.len() {
        let (p, v) = (f.position(i), f.velocity(i));
        out.extend(p.iter().chain(v.iter()).map(|c| c.to_bits()));
        out.push(f.density(i).to_bits());
    }
    out
}

/// A drop set on a horizontal capsule comes to rest on it with its centre one contact
/// radius off the surface.
#[test]
fn a_drop_rests_on_a_static_capsule_at_the_contact_radius() {
    let mut f = SphFluid::new(SphParams::water(), 4).unwrap();
    let (a, b, radius) = ([-0.2, 0.5, 0.0], [0.2, 0.5, 0.0], 0.05);
    let mut solids = SphSolids::new();
    assert!(solids.push_capsule(a, b, radius, [0.0; 3], [0.0; 3]));
    f.spawn([0.0, 0.7, 0.0], [0.0; 3]);
    for _ in 0..480 {
        f.step_with_solids(DT, 9.81, far_below, &solids);
    }
    let p = f.position(0);
    let off = segment_distance(p, a, b) - radius;
    let rc = f.contact_radius();
    // One substep of gravity sinks it by g dt^2 before the ray sets it back.
    let sag = 9.81 * DT * DT;
    assert!(
        (off - rc).abs() <= sag,
        "the drop rests {off:.6} m off the capsule, against a contact radius of {rc:.6} m"
    );
    let v = f.velocity(0);
    assert!(dot(v, v).sqrt() < 0.05, "still moving at {v:?} after 2 s");
}

/// A resting pool, a few particles deep, on level ground.
fn pool() -> SphFluid {
    let params = SphParams::water();
    let spacing = params.smoothing_radius * 0.5;
    let mut f = SphFluid::new(params, 2048).unwrap();
    for x in 0..24 {
        for y in 0..4 {
            for z in 0..16 {
                f.spawn(
                    [
                        x as f64 * spacing,
                        0.5 * spacing + y as f64 * spacing,
                        z as f64 * spacing,
                    ],
                    [0.0; 3],
                );
            }
        }
    }
    for _ in 0..120 {
        f.step(DT, 9.81, flat);
    }
    f
}

/// A shin wading through a pool: no particle is ever left inside it, and the particles
/// it meets head-on leave at its speed.
#[test]
fn a_capsule_swept_through_a_pool_carries_it_and_leaves_nothing_inside() {
    let mut f = pool();
    let radius = 0.04;
    let speed = 1.0;
    let z = 0.15;
    let mut solids = SphSolids::new();
    let mut ahead = Vec::new();
    for s in 0..72 {
        let x = -0.08 + speed * DT * (s + 1) as f64;
        let (a, b) = ([x, -0.02, z], [x, 0.3, z]);
        solids.clear();
        solids.push_capsule(a, b, radius, [speed, 0.0, 0.0], [speed, 0.0, 0.0]);
        f.step_with_solids(DT, 9.81, flat, &solids);
        for i in 0..f.len() {
            let d = segment_distance(f.position(i), a, b);
            assert!(
                d >= radius * (1.0 - 1e-12),
                "step {s}: particle {i} is {d:.6} m from the axis, inside the {radius} m capsule"
            );
        }
        if s == 71 {
            // The particles the capsule set on its surface this substep (exactly a contact
            // radius out), dead ahead of it and off the ground: a particle on the ground
            // sits at exactly zero, and the ground's friction takes its share after the
            // solid's.
            let rc = f.contact_radius();
            for i in 0..f.len() {
                let p = f.position(i);
                let off = [p[0] - x, 0.0, p[2] - z];
                let d = dot(off, off).sqrt();
                if (d - (radius + rc)).abs() < 1e-9 && off[0] > 0.9 * d && p[1] > 0.0 {
                    ahead.push(f.velocity(i)[0]);
                }
            }
        }
    }
    assert!(
        f.solid_stats().contacts > 0,
        "the capsule never met the pool"
    );
    assert!(
        ahead.len() >= 2,
        "no particle was being pushed head-on: {}",
        ahead.len()
    );
    let mean = ahead.iter().sum::<f64>() / ahead.len() as f64;
    // Head-on, the contact leaves a particle at the surface's speed plus `restitution`
    // of its approach: between 1 and 1 + restitution times the capsule's speed.
    let rest = f.params().restitution;
    assert!(
        mean >= speed * 0.999 && mean <= speed * (1.0 + rest) * 1.001,
        "particles pushed head-on move at {mean:.3} m/s ahead of a {speed} m/s capsule \
         ({} of them)",
        ahead.len()
    );
}

/// The reason for a ray over a point test: a droplet at the speed ceiling aimed at a box
/// one spacing thick, and at a 2 mm blade turned off the axes, ends on the near side. A
/// plain step, which knows no solid, passes straight through, so the geometry is one a
/// droplet does cross in a substep or two.
#[test]
fn a_droplet_at_the_speed_ceiling_cannot_tunnel_through_a_thin_box() {
    let params = SphParams::water();
    let spacing = params.smoothing_radius * 0.5;
    for (half_x, yaw) in [(0.5 * spacing, 0.0), (0.001, 0.5f64)] {
        let normal = [yaw.cos(), 0.0, -yaw.sin()];
        let centre = [0.1 * normal[0], 1.0, 0.1 * normal[2]];
        let mut solids = SphSolids::new();
        assert!(solids.push_box(centre, [half_x, 0.2, 0.2], yaw, [0.0; 3]));

        let run = |with_solids: bool| -> f64 {
            let mut f = SphFluid::new(params, 4).unwrap();
            let c = f.speed_ceiling(DT);
            f.spawn([0.0, 1.0, 0.0], [c * normal[0], 0.0, c * normal[2]]);
            let mut furthest = f64::NEG_INFINITY;
            for _ in 0..60 {
                if with_solids {
                    f.step_with_solids(DT, 0.0, far_below, &solids);
                } else {
                    f.step(DT, 0.0, far_below);
                }
                // Signed distance along the normal from the box's centre plane.
                let along = dot(sub(f.position(0), centre), normal);
                furthest = furthest.max(along);
            }
            furthest
        };

        let near_face = -half_x;
        let with = run(true);
        assert!(
            with <= near_face - 0.999 * SphFluid::new(params, 1).unwrap().contact_radius(),
            "half-thickness {half_x}: the droplet reached {with:.5} m along the normal, past \
             the near face at {near_face:.5}"
        );
        let without = run(false);
        assert!(
            without > half_x,
            "half-thickness {half_x}: without solids the droplet only reached {without:.5} m; \
             the test is not aiming through the box"
        );
    }
}

/// A box moved onto resting particles, so that they start the substep inside it, throws
/// them out through the face they are nearest, at the box's speed: what makes a falling
/// corpse splash. Checked square and yawed.
#[test]
fn a_box_pushed_onto_resting_particles_ejects_them_along_the_normal() {
    for yaw in [0.0, 0.3f64] {
        let normal = [yaw.cos(), 0.0, -yaw.sin()];
        let half = [0.1, 0.3, 0.1];
        let centre = [0.0, 1.0, 0.0];
        let speed = 1.0;
        let mut f = SphFluid::new(SphParams::water(), 16).unwrap();
        // Four millimetres inside the +x face, spread along the face's height.
        for j in 0..5 {
            let depth = half[0] - 0.004;
            f.spawn(
                [
                    centre[0] + depth * normal[0],
                    centre[1] - 0.1 + 0.05 * j as f64,
                    centre[2] + depth * normal[2],
                ],
                [0.0; 3],
            );
        }
        let mut solids = SphSolids::new();
        solids.push_box(
            centre,
            half,
            yaw,
            [speed * normal[0], 0.0, speed * normal[2]],
        );
        f.step_with_solids(DT, 0.0, far_below, &solids);

        let rc = f.contact_radius();
        assert_eq!(f.solid_stats().contacts, 5);
        for i in 0..f.len() {
            let p = sub(f.position(i), centre);
            let out = dot(p, normal);
            assert!(
                (out - (half[0] + rc)).abs() < 1e-9,
                "yaw {yaw}: particle {i} left {out:.6} m along the normal, not on the face \
                 plus the contact radius"
            );
            let v = f.velocity(i);
            let vn = dot(v, normal);
            let across = dot(
                sub(v, [vn * normal[0], vn * normal[1], vn * normal[2]]),
                [1.0; 3],
            );
            assert!(
                vn >= speed,
                "yaw {yaw}: thrown at {vn:.4} m/s along the normal"
            );
            assert!(
                across.abs() < 1e-9,
                "yaw {yaw}: thrown off the normal, {v:?}"
            );
        }
    }
}

/// SPH-F5 closed, behind the solids opt-in: a drop set on a 20 degree incline runs down
/// it, and stays on it. The plain step, which has no slope normal, leaves it where it is.
#[test]
fn the_slope_normal_runs_a_drop_downhill() {
    let grade = 20f64.to_radians().tan();
    let ground = move |x: f64, _z: f64| -grade * x; // downhill is +x
    let run = |slope: bool| -> [f64; 3] {
        let mut f = SphFluid::new(SphParams::water(), 4).unwrap();
        f.spawn([0.0, 0.0, 0.0], [0.0; 3]);
        let none = SphSolids::new();
        for _ in 0..480 {
            if slope {
                f.step_with_solids(DT, 9.81, ground, &none);
            } else {
                f.step(DT, 9.81, ground);
            }
        }
        f.position(0)
    };
    let p = run(true);
    assert!(
        p[0] > 0.01,
        "a drop on a 20 degree slope moved {:.5} m downhill in 2 s",
        p[0]
    );
    assert!(p[2].abs() < 1e-12, "it ran sideways: z = {:e}", p[2]);
    assert!(
        (p[1] - ground(p[0], p[2])).abs() < 1e-3,
        "it left the slope: {:.5} m above it",
        p[1] - ground(p[0], p[2])
    );
    let still = run(false);
    assert!(
        still[0].abs() < 1e-9,
        "the plain step moved it {:e} m",
        still[0]
    );
}

/// A splash on level ground, stepped with no solids, with solids none of it can reach,
/// and plainly: all three bit-identical. The far solids cost the binning and write no
/// bin.
#[test]
fn empty_or_unreachable_solids_are_bit_identical_to_step() {
    let scene = || {
        let mut f = SphFluid::new(SphParams::blood(), 512).unwrap();
        for i in 0..300usize {
            let (x, y, z) = (i % 7, (i / 7) % 7, i / 49);
            f.spawn(
                [x as f64 * 0.01, 0.2 + y as f64 * 0.01, z as f64 * 0.01],
                [0.3 * (i % 5) as f64 - 0.6, -1.0, 0.2 * (i % 3) as f64 - 0.2],
            );
        }
        f
    };
    let empty = SphSolids::new();
    let mut far = SphSolids::new();
    far.push_capsule([20.0, 0.0, 0.0], [20.0, 1.8, 0.0], 0.1, [1.0; 3], [1.0; 3]);
    far.push_box([-20.0, 0.5, 3.0], [0.9, 0.3, 0.3], 1.0, [0.0, -3.0, 0.0]);

    let (mut a, mut b, mut c) = (scene(), scene(), scene());
    for s in 0..150 {
        a.step(DT, 9.81, flat);
        b.step_with_solids(DT, 9.81, flat, &empty);
        c.step_with_solids(DT, 9.81, flat, &far);
        assert_eq!(
            c.solid_stats().bin_entries,
            0,
            "step {s}: a far solid was binned"
        );
        if s == 89 {
            a.drain_settled(|_| {});
            b.drain_settled(|_| {});
            c.drain_settled(|_| {});
        }
    }
    assert!(a.len() > 50, "the splash drained before it was compared");
    assert!(bits(&a) == bits(&b), "an empty solid set changed the fluid");
    assert!(
        bits(&a) == bits(&c),
        "solids out of reach changed the fluid"
    );
}

/// The binning's promise, against brute force: every particle that starts within reach
/// of a solid's bounds (the speed ceiling's travel plus the contact radius) finds that
/// solid in its own cell's bin.
#[test]
fn every_particle_within_reach_of_a_solid_finds_it_in_its_bin() {
    let params = SphParams::blood();
    let mut f = SphFluid::new(params, 2048).unwrap();
    let mut seed = 31u32;
    let mut rand = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed >> 8) as f64 / ((1u32 << 24) as f64) - 0.5
    };
    for _ in 0..1500 {
        f.spawn([rand() * 0.6, 0.3 + rand() * 0.6, rand() * 0.6], [0.0; 3]);
    }
    let mut solids = SphSolids::new();
    for k in 0..6 {
        let o = k as f64 * 0.1 - 0.25;
        solids.push_capsule(
            [o, 0.1, -0.2],
            [o + 0.05, 0.5, 0.2],
            0.03,
            [0.0; 3],
            [0.0; 3],
        );
        solids.push_box([-o, 0.3, o], [0.04, 0.1, 0.02], o * 3.0, [0.0; 3]);
    }
    f.build_grid();
    f.bin_solids(&solids);
    let h = params.smoothing_radius;
    let reach = CFL_FRACTION * h + f.contact_radius();
    let contact = solids::Contact::new(&f.solid_bins, &solids, 0.0, 0.0, 1.0);
    let mut checked = 0;
    for i in 0..f.len() {
        let p = f.position(i);
        let binned = contact.binned(f.bucket_of[i]);
        for id in 0..solids.len() {
            let (lo, hi) = solids.bounds(id);
            let gap2: f64 = (0..3)
                .map(|a| {
                    let g = (lo[a] - p[a]).max(p[a] - hi[a]).max(0.0);
                    g * g
                })
                .sum();
            if gap2 <= reach * reach {
                checked += 1;
                assert!(
                    binned.contains(&(id as u32)),
                    "particle {i} at {p:?} is within reach of solid {id} but its bin {binned:?} \
                     does not hold it"
                );
            }
        }
    }
    assert!(
        checked > 500,
        "only {checked} particle-solid pairs were in reach"
    );
    f.solid_bins.reset();
    assert!(
        f.solid_bins.range.iter().all(|r| *r == [0, 0]),
        "reset left a bucket set"
    );
}

/// The bytes the module documentation declares for a solid and a bin, held to the types.
#[test]
fn a_solid_and_a_bin_cost_the_declared_bytes() {
    use std::mem::size_of;
    assert_eq!(size_of::<solids::Capsule>(), 144);
    assert_eq!(size_of::<solids::YawBox>(), 88);
    // A bucket's range, and an entry: its pair (bucket, solid) and its slot.
    assert_eq!(size_of::<[u32; 2]>(), 8);
    assert_eq!(2 * size_of::<u32>() + size_of::<u32>(), 12);
}
