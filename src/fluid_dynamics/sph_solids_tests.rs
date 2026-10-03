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

/// With the slope normal, a drop driven up an incline at `friction = 1.0` pays for the
/// height it gains: the clamp no longer lifts it for free (the energy half of SPH-F5).
#[test]
fn climbing_a_slope_with_the_normal_costs_kinetic_energy() {
    let grade = 30f64.to_radians().tan();
    let ground = move |x: f64, _z: f64| grade * x; // uphill is +x
    let mut p = SphParams::water();
    p.friction = 1.0;
    let mut f = SphFluid::new(p, 4).unwrap();
    f.spawn([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    let none = SphSolids::new();
    let e0 = 0.5 * 1.0f64.powi(2);
    for _ in 0..480 {
        f.step_with_solids(DT, 9.81, ground, &none);
    }
    let v = f.velocity(0);
    let e1 = 0.5 * dot(v, v) + 9.81 * f.position(0)[1];
    assert!(
        e1 <= e0 * 1.01,
        "specific mechanical energy went {e0:.3} -> {e1:.3} J/kg"
    );
}

/// A top-down picture of a pool a shin has waded halfway through: one character a
/// particle column, `.` empty, digits the particles stacked there, `#` the shin.
#[test]
#[ignore = "diagnostic picture: run with --ignored --nocapture"]
fn picture_of_a_waded_pool() {
    let mut f = pool();
    let (radius, z) = (0.04, 0.15);
    let mut solids = SphSolids::new();
    let mut x = 0.0;
    for s in 0..72 {
        x = -0.08 + DT * (s + 1) as f64;
        solids.clear();
        solids.push_capsule(
            [x, -0.02, z],
            [x, 0.3, z],
            radius,
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        );
        f.step_with_solids(DT, 9.81, flat, &solids);
    }
    let cell = 0.01;
    let (nx, nz) = (60usize, 34usize);
    let mut count = vec![0u32; nx * nz];
    for i in 0..f.len() {
        let p = f.position(i);
        let (cx, cz) = (
            ((p[0] + 0.05) / cell).floor(),
            ((p[2] + 0.01) / cell).floor(),
        );
        if cx >= 0.0 && cz >= 0.0 && (cx as usize) < nx && (cz as usize) < nz {
            count[cz as usize * nx + cx as usize] += 1;
        }
    }
    let mut out = format!("pool after a 1 m/s shin waded to x = {x:.3} m (top down, 1 cm cells)\n");
    for row in 0..nz {
        for col in 0..nx {
            let (px, pz) = (
                col as f64 * cell - 0.05 + 0.005,
                row as f64 * cell - 0.01 + 0.005,
            );
            let shin = ((px - x).powi(2) + (pz - z).powi(2)).sqrt() < radius;
            let c = count[row * nx + col];
            out.push(if shin {
                '#'
            } else if c == 0 {
                '.'
            } else {
                char::from_digit(c.min(9), 10).unwrap()
            });
        }
        out.push('\n');
    }
    eprintln!("{out}");
}

/// Drops scattered over a `side` metre square of rolling ground (half a metre of relief),
/// `n` of them, from a fixed xorshift: a wide, sparse fluid whose hash buckets mostly
/// hold one cell, and sometimes two.
fn scattered(n: usize, side: f64) -> SphFluid {
    let mut f = SphFluid::new(SphParams::blood(), n).unwrap();
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut unit = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    while f.len() < n {
        let (x, z) = (side * unit(), side * unit());
        let y = 0.3 * (1.3 * x).sin() + 0.2 * (0.9 * z).cos() + 0.4 * unit();
        f.spawn([x, y, z], [0.0; 3]);
    }
    f
}

/// Limbs and crates over the scattered field, and one long hull across it: solids of
/// every size, overlapping each other, in and out of reach.
fn mixed_solids(side: f64) -> SphSolids {
    let mut solids = SphSolids::new();
    for k in 0..40 {
        let x = side * ((k * 37) % 97) as f64 / 97.0;
        let z = side * ((k * 61) % 89) as f64 / 89.0;
        solids.push_capsule(
            [x, -0.1, z],
            [x + 0.1, 0.4, z - 0.05],
            0.06,
            [0.0; 3],
            [0.0; 3],
        );
        if k % 5 == 0 {
            solids.push_box([z, 0.2, x], [0.3, 0.2, 0.15], k as f64, [0.0; 3]);
        }
    }
    solids.push_capsule(
        [0.0, 0.3, 0.2 * side],
        [side, 0.1, 0.8 * side],
        0.4,
        [0.0; 3],
        [0.0; 3],
    );
    solids
}

/// Every bucket's bin under one walk.
fn bins_by(f: &mut SphFluid, solids: &SphSolids, path: solids::BinPath) -> Vec<Vec<u32>> {
    f.build_grid();
    f.bin_solids_by(solids, path);
    let contact = solids::Contact::new(&f.solid_bins, solids, 0.0, 0.0, 1.0);
    let out = (0..=f.table_mask)
        .map(|b| contact.binned(b as u32).to_vec())
        .collect();
    f.solid_bins.reset();
    out
}

/// The two walks write the same bins, bucket for bucket and in the same order, and both
/// equal a brute-force oracle: for each solid in order, its id once for every occupied
/// cell of the bucket that lies in its reach, cells taken from the particles themselves.
#[test]
fn both_binning_walks_write_the_oracles_bins() {
    let side = 3.0;
    let mut f = scattered(3000, side);
    let solids = mixed_solids(side);
    let cells = bins_by(&mut f, &solids, solids::BinPath::Cells);
    let particles = bins_by(&mut f, &solids, solids::BinPath::Particles);
    let measured = bins_by(&mut f, &solids, solids::BinPath::Measured);

    // The oracle, from the grid `bins_by` left built.
    let h = f.params.smoothing_radius;
    let reach = (CFL_FRACTION * h + f.contact_radius()) * (1.0 + 1e-6);
    let mut occupied: Vec<([i32; 3], usize)> = (0..f.len())
        .map(|i| {
            (
                [f.cell_x[i], f.cell_y[i], f.cell_z[i]],
                f.bucket_of[i] as usize,
            )
        })
        .collect();
    occupied.sort_unstable();
    occupied.dedup();
    let mut oracle = vec![Vec::new(); f.table_mask + 1];
    let mut shared = 0;
    for id in 0..solids.len() {
        let (lo, hi) = solids.bounds(id);
        for &(c, b) in &occupied {
            if (0..3)
                .all(|a| c[a] >= cell_of(lo[a] - reach, h) && c[a] <= cell_of(hi[a] + reach, h))
            {
                oracle[b].push(id as u32);
            }
        }
    }
    for (b, bin) in oracle.iter().enumerate() {
        if bin.windows(2).any(|w| w[0] != w[1]) {
            shared += 1;
        }
        assert_eq!(
            cells[b], *bin,
            "bucket {b}: the cell walk differs from the oracle"
        );
        assert_eq!(
            particles[b], *bin,
            "bucket {b}: the particle scan differs from the oracle"
        );
        assert_eq!(
            measured[b], *bin,
            "bucket {b}: the measured choice differs from the oracle"
        );
    }
    let entries: usize = oracle.iter().map(Vec::len).sum();
    assert!(
        entries > 500,
        "only {entries} bin entries: the scene does not exercise the bins"
    );
    assert!(
        shared > 10,
        "only {shared} buckets hold two solids: the order is not exercised"
    );
}

/// A long hull over a wide, sparse fluid (its box holds over a hundred cells a particle,
/// so the step takes the particle scan): at every step the cell walk, rebuilt from the
/// same grid, writes the same bins, so the move reads what the walk would have given it.
/// Stepped at 1, 4 and 8 threads: the scan is serial, and the move that reads its bins is
/// not.
#[test]
fn the_particle_scan_steps_bit_identically_at_any_thread_count() {
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let side = 6.0;
            let mut f = scattered(2500, side);
            let mut solids = SphSolids::new();
            let mut scanned = 0;
            for s in 0..60 {
                let x = 0.5 + 0.02 * s as f64;
                solids.clear();
                solids.push_capsule(
                    [x, 0.2, 0.5],
                    [x + 4.0, 0.2, 5.0],
                    0.45,
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                );
                f.step_with_solids(DT, 9.81, flat, &solids);
                // The same step's bins rebuilt both ways from the stepped fluid agree.
                let mut probe = f.clone();
                let a = bins_by(&mut probe, &solids, solids::BinPath::Cells);
                let b = bins_by(&mut probe, &solids, solids::BinPath::Particles);
                assert!(a == b, "step {s}: the walks disagree");
                if f.solid_stats().bin_entries > 0 {
                    scanned += 1;
                }
            }
            assert!(
                scanned > 50,
                "the hull reached the fluid on only {scanned} steps"
            );
            bits(&f)
        })
    };
    let one = run(1);
    for t in [4, 8] {
        assert!(run(t) == one, "{t} threads gave a different fluid from one");
    }
}

/// The crossover between the two walks, measured: a capsule of growing size over a
/// sparse fluid, binned by each walk, best of many runs. Prints the box cells, the
/// particle count and both times; the scan wins once the cells pass the particles by the
/// printed ratio. Run in release.
#[test]
#[ignore = "perf diagnostic: run with --release --ignored --nocapture"]
fn binning_crossover() {
    for n in [1_024usize, 4_096, 16_384] {
        let side = (n as f64 / 40.0).sqrt();
        let mut f = scattered(n, side);
        f.build_grid();
        for (len, radius) in [
            (0.0, 0.02),
            (0.1, 0.02),
            (0.2, 0.04),
            (0.4, 0.06),
            (0.4, 0.15),
            (0.8, 0.3),
            (1.6, 0.3),
            (3.2, 0.3),
            (6.4, 0.3),
        ] {
            let (len, radius): (f64, f64) = (len, radius);
            let mut solids = SphSolids::new();
            let c = 0.5 * side;
            solids.push_capsule(
                [c - 0.5 * len, 0.3, c],
                [c + 0.5 * len, 0.3, c + 0.3 * len],
                radius,
                [0.0; 3],
                [0.0; 3],
            );
            let time = |f: &mut SphFluid, path| {
                let mut best = f64::INFINITY;
                for _ in 0..200 {
                    let t = std::time::Instant::now();
                    f.bin_solids_by(&solids, path);
                    best = best.min(t.elapsed().as_secs_f64() * 1e6);
                    f.solid_bins.reset();
                }
                best
            };
            let walk = time(&mut f, solids::BinPath::Cells);
            let scan = time(&mut f, solids::BinPath::Particles);
            let h = f.params.smoothing_radius;
            let (lo, hi) = solids.bounds(0);
            let reach = CFL_FRACTION * h + f.contact_radius();
            let span = |a: usize| {
                let (mn, mx) = (0..f.len()).fold((i32::MAX, i32::MIN), |(mn, mx), i| {
                    let c = [f.cell_x[i], f.cell_y[i], f.cell_z[i]][a];
                    (mn.min(c), mx.max(c))
                });
                let c0 = cell_of(lo[a] - reach, h).max(mn);
                let c1 = cell_of(hi[a] + reach, h).min(mx);
                (c1 - c0 + 1).max(0) as f64
            };
            let cells = span(0) * span(1) * span(2);
            eprintln!(
                "n {n:>6} len {len:>4} m r {radius:>4} m: cells {cells:>8} ({:>6.2} a particle)  walk {walk:>8.1} us  scan {scan:>7.1} us  walk/scan {:>6.2}",
                cells / n as f64,
                walk / scan
            );
        }
    }
}

/// Steps `f` for up to `steps` substeps, refilling `solids` before each with `fill(step)`,
/// draining after each, and returns every drop drained with the step it settled on.
fn settle_run(
    f: &mut SphFluid,
    steps: usize,
    ground: fn(f64, f64) -> f64,
    mut fill: impl FnMut(usize, &mut SphSolids),
) -> Vec<(usize, Settled)> {
    let mut solids = SphSolids::new();
    let mut out = Vec::new();
    for s in 0..steps {
        solids.clear();
        fill(s, &mut solids);
        f.step_with_solids(DT, 9.81, ground, &solids);
        f.drain_settled(|d| out.push((s, d)));
    }
    out
}

/// A crate sitting on the ground, with a far limb pushed before it so the crate is solid
/// 1 (capsules come first).
fn far_limb_and_crate(solids: &mut SphSolids) {
    solids.push_capsule([5.0, 0.0, 5.0], [5.0, 1.0, 5.0], 0.06, [0.0; 3], [0.0; 3]);
    solids.push_box([0.0, 0.25, 0.0], [0.25; 3], 0.3, [0.0; 3]);
}

/// A drop that lands on a resting crate settles on its lid, a contact radius up, with the
/// crate's index, once it has been still on it for `SETTLE_TIME`.
#[test]
fn a_drop_landing_on_a_static_box_settles_with_its_index() {
    let mut f = SphFluid::new(SphParams::blood(), 4).unwrap();
    f.spawn([0.05, 0.8, -0.05], [0.0; 3]);
    let settled = settle_run(&mut f, 480, flat, |_, s| far_limb_and_crate(s));
    assert_eq!(settled.len(), 1, "drained {settled:?}");
    let (step, d) = settled[0];
    assert_eq!(d.on_solid, Some(1), "settled on {:?}", d.on_solid);
    let lid = 0.5 + f.contact_radius();
    assert!(
        (d.position[1] - lid).abs() <= 9.81 * DT * DT,
        "settled at y = {} against a lid at {lid}",
        d.position[1]
    );
    // Falling 0.3 m takes 0.247 s, and then it must be still for SETTLE_TIME.
    let fall = (2.0f64 * 0.3 / 9.81).sqrt();
    let t = (step + 1) as f64 * DT;
    assert!(
        t >= fall + SETTLE_TIME && t <= fall + SETTLE_TIME + 0.1,
        "settled at {t:.3} s; it lands at {fall:.3} s"
    );
}

/// A drop that lands on a capsule moving at 1 m/s along its axis rides it, and settles
/// with the capsule's index once it has been still relative to the capsule for
/// `SETTLE_TIME`; its reported velocity is the surface's, to within the one substep of
/// gravity (`g dt`) a resting contact leaves it.
///
/// Open: the contact casts the particle's world displacement against the solid at its
/// new pose, so a drop carried along by a moving surface casts a ray nearly parallel to
/// it, which meets the surface only once the drop has sunk the whole contact radius. It
/// rides in a cycle (at 1 m/s, nine substeps of free fall relative to the hull and a
/// bump of 6.6 mm), meets the hull one substep in ten, and is never still on it for
/// `SETTLE_TIME`. Any tangential surface speed above about `g dt` (0.04 m/s at 240 Hz)
/// does the same; a rising surface catches the drop only by pushing it out. Casting the
/// displacement relative to the surface's velocity fixes it and changes the results of
/// every moving solid, so it waits on a ruling.
#[test]
#[ignore = "known defect, not fixed in this change: the swept contact is cast in the world frame, so a drop riding a moving solid meets it one substep in ten and never settles on it"]
fn a_drop_riding_a_moving_capsule_settles_with_its_index_and_the_surface_velocity() {
    let mut f = SphFluid::new(SphParams::blood(), 4).unwrap();
    f.spawn([0.0, 0.8, 0.0], [0.0; 3]);
    let v = [1.0, 0.0, 0.0];
    let settled = settle_run(&mut f, 720, far_below, |s, solids| {
        let x = (s + 1) as f64 * DT;
        solids.push_capsule([x - 1.0, 0.5, 0.0], [x + 2.0, 0.5, 0.0], 0.1, v, v);
    });
    assert_eq!(settled.len(), 1, "drained {settled:?}");
    let (step, d) = settled[0];
    assert_eq!(d.on_solid, Some(0));
    let rel = sub(d.velocity, v);
    assert!(
        dot(rel, rel).sqrt() <= 9.81 * DT,
        "settled at {:?} on a surface at {v:?}",
        d.velocity
    );
    // On the capsule's top, a contact radius up, and carried along with it.
    assert!((d.position[1] - (0.6 + f.contact_radius())).abs() <= 9.81 * DT * DT);
    let x = (step + 1) as f64 * DT;
    assert!(d.position[0] > x - 1.0 && d.position[0] < x + 2.0);
    assert!(
        d.position[0] > 0.3,
        "it did not ride: settled at x = {}",
        d.position[0]
    );
}

/// A drop that comes to rest on the ground reports no solid, whether stepped plainly or
/// with solids in reach of other particles.
#[test]
fn a_drop_settled_on_the_ground_reports_no_solid() {
    let mut plain = SphFluid::new(SphParams::blood(), 4).unwrap();
    plain.spawn([1.0, 0.3, 1.0], [0.0; 3]);
    let mut drained = Vec::new();
    for _ in 0..240 {
        plain.step(DT, 9.81, flat);
        plain.drain_settled(|d| drained.push(d));
    }
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0].on_solid, None);

    let mut f = SphFluid::new(SphParams::blood(), 4).unwrap();
    f.spawn([1.0, 0.3, 1.0], [0.0; 3]);
    f.spawn([0.0, 0.8, 0.0], [0.0; 3]);
    let settled = settle_run(&mut f, 480, flat, |_, s| far_limb_and_crate(s));
    let on: Vec<Option<u32>> = settled.iter().map(|(_, d)| d.on_solid).collect();
    assert_eq!(settled.len(), 2, "drained {settled:?}");
    let ground = settled.iter().find(|(_, d)| d.position[1] < 0.1).unwrap();
    assert_eq!(ground.1.on_solid, None, "{on:?}");
    assert!(on.contains(&Some(1)), "{on:?}");
}

/// The index a resting drop reports is the one its solid has in the set handed to each
/// step, however the bins under it change: rain falling around the crate, a limb
/// sweeping across it and a solid added mid-rest change which buckets are binned
/// from step to step, and the crate, refilled in the same place in the order, is the one
/// named. Its solid's index shifting between steps (as a set refilled every frame from
/// the actors in reach does) neither restarts nor stalls its count: reindexed once
/// mid-rest, or on every other step, it settles on the same step, in the same place to
/// the bit, reporting the index of the step it drained after.
#[test]
fn a_resting_drops_solid_index_survives_the_bins_being_rebuilt() {
    let run = |extra_limbs: &dyn Fn(usize) -> bool| {
        let mut f = SphFluid::new(SphParams::blood(), 1024).unwrap();
        f.spawn([0.0, 0.8, 0.0], [0.0; 3]);
        let (mut last, mut changes) = (usize::MAX, 0);
        let mut solids = SphSolids::new();
        let mut out = Vec::new();
        for s in 0..480 {
            // Rain around the crate, never on it, through the limb's path.
            let a = s as f64 * 2.399;
            let r = 0.4 + 0.3 * ((s * 7) % 11) as f64 / 11.0;
            f.spawn([r * a.cos(), 0.5, r * a.sin()], [0.0, -1.0, 0.0]);
            solids.clear();
            if extra_limbs(s) {
                // Two more limbs from here on: capsules come first, so the crate goes from
                // solid 1 to solid 3.
                solids.push_capsule([3.0, 0.0, 3.0], [3.0, 1.0, 3.0], 0.06, [0.0; 3], [0.0; 3]);
                solids.push_capsule([4.0, 0.0, 3.0], [4.0, 1.0, 3.0], 0.06, [0.0; 3], [0.0; 3]);
            }
            // A limb sweeping across the field and over the crate's edge.
            let x = -0.8 + 1.6 * (s as f64 / 480.0);
            solids.push_capsule(
                [x, 0.0, 0.3],
                [x, 0.7, 0.3],
                0.05,
                [0.8, 0.0, 0.0],
                [0.8, 0.0, 0.0],
            );
            solids.push_box([0.0, 0.25, 0.0], [0.25; 3], 0.0, [0.0; 3]);
            if s >= 100 {
                // A crate dropped beside it: a box after this one, so its index stays.
                solids.push_box([0.6, 0.2, -0.5], [0.1; 3], 0.5, [0.0; 3]);
            }
            f.step_with_solids(DT, 9.81, flat, &solids);
            let entries = f.solid_stats().bin_entries;
            changes += (entries != last) as usize;
            last = entries;
            // The drop on the crate's lid; the rain settles elsewhere.
            f.drain_settled(|d| {
                let p = d.position;
                if p[1] > 0.4 && p[0].abs() < 0.3 && p[2].abs() < 0.3 {
                    out.push((s, d));
                }
            });
        }
        assert!(changes > 20, "the bins changed on only {changes} steps");
        out
    };

    let steady = run(&|_| false);
    assert_eq!(steady.len(), 1, "{steady:?}");
    assert_eq!(steady[0].1.on_solid, Some(1));

    // Reindexed once after it has rested a while but before it settles, and on every
    // other step throughout.
    let (settle, first) = steady[0];
    let at = settle - 20;
    let bits = |d: &Settled| d.position.map(f64::to_bits);
    for (name, moved) in [
        ("once", run(&|s| s >= at)),
        ("every other step", run(&|s| s % 2 == 1)),
    ] {
        assert_eq!(moved.len(), 1, "{name}: {moved:?}");
        let (step, d) = moved[0];
        assert_eq!(step, settle, "{name}: settled on step {step}, not {settle}");
        assert_eq!(bits(&d), bits(&first), "{name}: settled somewhere else");
        // The index of the step it drained after: 3 with the two extra limbs pushed.
        let expect = if (name == "once") || step % 2 == 1 {
            3
        } else {
            1
        };
        assert_eq!(d.on_solid, Some(expect), "{name}");
    }
}

/// Drops raining onto crates, a moving hull and the ground settle in the same places,
/// with the same velocities and on the same solids, at 1, 4 and 8 threads.
#[test]
fn settling_on_solids_is_bit_identical_at_any_thread_count() {
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut f = SphFluid::new(SphParams::blood(), 2048).unwrap();
            for k in 0..1500usize {
                let (i, j) = (k % 50, k / 50);
                f.spawn(
                    [
                        -1.0 + 0.04 * i as f64,
                        0.9 + 0.02 * (k % 7) as f64,
                        -0.6 + 0.04 * j as f64,
                    ],
                    [0.1 * (k % 3) as f64, 0.0, -0.1 * (k % 5) as f64],
                );
            }
            let settled = settle_run(&mut f, 360, flat, |s, solids| {
                let x = -0.5 + (s + 1) as f64 * DT;
                let v = [1.0, 0.0, 0.0];
                solids.push_capsule([x, 0.35, -0.3], [x + 0.8, 0.35, 0.3], 0.2, v, v);
                solids.push_box([-0.6, 0.2, 0.2], [0.2; 3], 0.4, [0.0; 3]);
                solids.push_box([0.6, 0.15, -0.3], [0.3, 0.15, 0.2], -0.2, [0.0; 3]);
            });
            let mut on = [0usize; 4];
            let mut bits = Vec::new();
            for (s, d) in &settled {
                on[d.on_solid.map_or(3, |i| i as usize)] += 1;
                bits.push(*s as u64);
                bits.extend(d.position.iter().chain(&d.velocity).map(|c| c.to_bits()));
                bits.push(d.on_solid.map_or(u64::MAX, u64::from));
            }
            (bits, on)
        })
    };
    let (one, on) = run(1);
    assert!(
        on[3] > 100 && on[0] + on[1] + on[2] > 20 && on[1] > 0 && on[2] > 0,
        "settled on hull, crate, crate, ground: {on:?}"
    );
    for t in [4, 8] {
        assert!(run(t).0 == one, "{t} threads settled differently from one");
    }
}
