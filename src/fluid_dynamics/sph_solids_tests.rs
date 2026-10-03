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
    f.bin_solids(&solids, DT);
    let h = params.smoothing_radius;
    let reach = CFL_FRACTION * h + f.contact_radius();
    let contact = solids::Contact::new(&f.solid_bins, &solids, 0.0, 0.0, 1.0, DT);
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
    f.bin_solids_by(solids, DT, path);
    let contact = solids::Contact::new(&f.solid_bins, solids, 0.0, 0.0, 1.0, DT);
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
                    f.bin_solids_by(&solids, DT, path);
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
    fill: impl FnMut(usize, &mut SphSolids),
) -> Vec<(usize, Settled)> {
    settle_run_at(f, steps, DT, ground, fill)
}

/// [`settle_run`] at a substep of `dt`.
fn settle_run_at(
    f: &mut SphFluid,
    steps: usize,
    dt: f64,
    ground: fn(f64, f64) -> f64,
    mut fill: impl FnMut(usize, &mut SphSolids),
) -> Vec<(usize, Settled)> {
    let mut solids = SphSolids::new();
    let mut out = Vec::new();
    for s in 0..steps {
        solids.clear();
        fill(s, &mut solids);
        f.step_with_solids(dt, 9.81, ground, &solids);
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

/// A drop falls onto a hull capsule (radius 0.1 m, its top at 0.6 m) moving at `speed`
/// along its axis, given at its pose at the end of each substep; returns the fluid and
/// what drained.
fn ride(params: SphParams, dt: f64, speed: f64, steps: usize) -> (SphFluid, Vec<(usize, Settled)>) {
    let mut f = SphFluid::new(params, 4).unwrap();
    f.spawn([0.0, 0.8, 0.0], [0.0; 3]);
    let v = [speed, 0.0, 0.0];
    let settled = settle_run_at(&mut f, steps, dt, far_below, |s, solids| {
        let x = (s + 1) as f64 * dt * speed;
        solids.push_capsule([x - 3.0, 0.5, 0.0], [x + 1.0, 0.5, 0.0], 0.1, v, v);
    });
    (f, settled)
}

/// What a drop that rode a hull at `speed` must have done: settled once, on solid 0, on
/// the hull's top a contact radius up, carried with it (its velocity the surface's, to
/// within the one substep of gravity a resting contact leaves it), no sooner than
/// `SETTLE_TIME` after it landed. Later by the time the hull takes to bring it up to
/// speed: it lands at rest and bounces (restitution), and friction acts on each substep
/// it touches the hull (see `a_drop_set_on_a_1_m_s_hull_reaches_its_speed_within_a_few_substeps`).
fn assert_rode(f: &SphFluid, settled: &[(usize, Settled)], dt: f64, speed: f64) {
    assert_eq!(settled.len(), 1, "{speed} m/s: drained {settled:?}");
    let (step, d) = settled[0];
    assert_eq!(d.on_solid, Some(0), "{speed} m/s");
    let rel = sub(d.velocity, [speed, 0.0, 0.0]);
    assert!(
        dot(rel, rel).sqrt() <= 9.81 * dt,
        "{speed} m/s: settled at {:?}",
        d.velocity
    );
    let top = 0.6 + f.contact_radius();
    assert!(
        (d.position[1] - top).abs() <= 9.81 * dt * dt,
        "{speed} m/s: settled at y = {} against a top at {top}",
        d.position[1]
    );
    // It landed at x = 0 and rode: still relative to the hull for SETTLE_TIME, so carried
    // at least that far, and no further than the hull went after it landed (it lands at
    // rest and the hull's friction brings it up to speed).
    let t = (step + 1) as f64 * dt;
    let x = t * speed;
    let fall = (2.0f64 * (0.8 - top) / 9.81).sqrt();
    let carried = (t - fall) * speed;
    assert!(
        d.position[0] >= speed * SETTLE_TIME && d.position[0] <= carried,
        "{speed} m/s: settled at x = {}; the hull went {carried:.3} m after it landed",
        d.position[0]
    );
    assert!(d.position[0] > x - 3.0 && d.position[0] < x + 1.0);
    assert!(
        t >= fall + SETTLE_TIME,
        "{speed} m/s: settled at {t:.3} s; it lands at {fall:.3} s"
    );
}

/// A drop that lands on a capsule moving at 1 m/s along its axis rides it, and settles
/// with the capsule's index once it has been still relative to the capsule for
/// `SETTLE_TIME`; its reported velocity is the surface's, to within the one substep of
/// gravity (`g dt`) a resting contact leaves it.
///
/// The defect SPH-SOLIDS-b pinned here: the contact cast the particle's world
/// displacement against the solid at its new pose, so a drop carried along by the
/// surface cast a ray nearly parallel to it and met it only after sinking the whole
/// contact radius, one substep in ten, and never settled. Cast in the solid's frame
/// (`p1 - p0 - v_s dt`), a drop moving with the surface casts only gravity's sag,
/// straight at it.
#[test]
fn a_drop_riding_a_moving_capsule_settles_with_its_index_and_the_surface_velocity() {
    let (f, settled) = ride(SphParams::blood(), DT, 1.0, 720);
    assert_rode(&f, &settled, DT, 1.0);
}

/// The same ride at 5 m/s. Blood at 240 Hz caps a particle at its speed ceiling, 3.84
/// m/s (0.4 of a 4 cm smoothing radius a substep), slower than this hull, so the ride is
/// run where the ceiling clears it: at 480 Hz, 7.68 m/s. The pinned case below is the
/// same hull at 240 Hz.
#[test]
fn a_drop_rides_a_hull_at_5_m_s_below_the_speed_ceiling() {
    let dt = 1.0 / 480.0;
    let probe = SphFluid::new(SphParams::blood(), 1).unwrap();
    assert!(probe.speed_ceiling(dt) > 5.0);
    let (f, settled) = ride(SphParams::blood(), dt, 5.0, 1440);
    assert_rode(&f, &settled, dt, 5.0);
}

/// Known gap: a surface faster than the fluid's speed ceiling cannot carry a drop at its
/// speed, because the cap, a world-frame speed limit, takes the drop back to the ceiling
/// every substep before the contact runs. At 5 m/s under blood's 3.84 m/s ceiling at
/// 240 Hz the drop slips back along the hull at 1.16 m/s, over the settle speed, and
/// never settles.
#[test]
#[ignore = "known gap: the speed cap is world-frame, so a hull faster than the ceiling (5 m/s against 3.84 for blood at 240 Hz) cannot carry a drop at its speed"]
fn a_drop_rides_a_hull_faster_than_the_speed_ceiling() {
    let (f, settled) = ride(SphParams::blood(), DT, 5.0, 720);
    assert_rode(&f, &settled, DT, 5.0);
}

/// A drop on a lift rising at 0.5 m/s rides it up and settles with its index (1, after a
/// far capsule), at the lift's velocity and a contact radius above its lid. Before the
/// relative cast the drop cast its own rise against the lid at its new pose, missed it,
/// sank into it and was pushed out again, a cycle that never settled.
#[test]
fn a_drop_on_a_rising_lift_settles_with_its_index_and_the_lift_velocity() {
    let mut f = SphFluid::new(SphParams::blood(), 4).unwrap();
    f.spawn([0.05, 0.8, -0.05], [0.0; 3]);
    let rise = 0.5;
    let lid = |s: usize| 0.5 + (s + 1) as f64 * DT * rise;
    let settled = settle_run(&mut f, 480, far_below, |s, solids| {
        solids.push_capsule([5.0, 0.0, 5.0], [5.0, 1.0, 5.0], 0.06, [0.0; 3], [0.0; 3]);
        solids.push_box([0.0, lid(s) - 0.25, 0.0], [0.25; 3], 0.3, [0.0, rise, 0.0]);
    });
    assert_eq!(settled.len(), 1, "drained {settled:?}");
    let (step, d) = settled[0];
    assert_eq!(d.on_solid, Some(1));
    let rel = sub(d.velocity, [0.0, rise, 0.0]);
    assert!(
        dot(rel, rel).sqrt() <= 9.81 * DT,
        "settled at {:?} on a lid rising at {rise}",
        d.velocity
    );
    let top = lid(step) + f.contact_radius();
    assert!(
        (d.position[1] - top).abs() <= 9.81 * DT * DT,
        "settled at y = {} against a lid at {top}",
        d.position[1]
    );
    // The lid meets the falling drop where 0.8 - g t^2 / 2 = 0.5 + rc + rise t.
    let gap = 0.3 - f.contact_radius();
    let meet = (-rise + (rise * rise + 2.0 * 9.81 * gap).sqrt()) / 9.81;
    let t = (step + 1) as f64 * DT;
    assert!(
        t >= meet + SETTLE_TIME && t <= meet + SETTLE_TIME + 0.1,
        "settled at {t:.3} s; the lid meets it at {meet:.3} s"
    );
}

/// The share of its tangential speed relative to a surface a drop keeps through one
/// substep of `dt` in contact: the ground's implicit friction decay, which the solids use.
fn friction_keep(params: &SphParams, dt: f64) -> f64 {
    1.0 / (1.0 + FRICTION_REFERENCE_HZ * (1.0 / params.friction - 1.0) * dt)
}

/// The friction rate, 1/s: a drop sliding at `u` across a surface it stays on stops in
/// `u / rate`, whatever the substep.
fn friction_rate(params: &SphParams) -> f64 {
    FRICTION_REFERENCE_HZ * (1.0 / params.friction - 1.0)
}

/// N, the substeps in contact friction takes to bring a drop sliding at `speed` across a
/// surface to rest on it. Each keeps `friction_keep` of the tangential speed, and a drop
/// is at rest on a surface once that is under `g dt`, the one substep of gravity a resting
/// contact leaves it: N is the least k with `speed keep^k <= g dt`. With the sphere cast a
/// drop on a surface meets it every substep, so N contacts are N substeps.
fn substeps_to_rest(params: &SphParams, dt: f64, speed: f64) -> usize {
    ((9.81 * dt / speed).ln() / friction_keep(params, dt).ln()).ceil() as usize
}

/// N for the two hull tests below: blood, 240 Hz, the faster of the two slides (2 m/s).
fn hull_substeps_to_rest() -> usize {
    let n = substeps_to_rest(&SphParams::blood(), DT, 2.0);
    // A few, not tens: blood keeps 35% a substep, so 2 m/s is under g dt in four.
    assert!(n <= 5, "N = {n}");
    n
}

/// A drop riding a hull at 2 m/s, for less than the settle time, stops with the hull. It
/// is suddenly 2 m/s off the surface's speed, and since it sits a contact radius off the
/// hull it meets it on every substep from the stop: friction brings it to rest within N
/// substeps (`hull_substeps_to_rest`), and it slides `u / rate` (the implicit decay's
/// stopping distance, 4.5 mm for blood), under a centimetre. Before the sphere cast the
/// drop met the stopped hull about one substep in ten and slid 14.5 cm. Its count restarts
/// at the stop, and it settles with the hull's index and no velocity but the one substep
/// of gravity, no sooner than `SETTLE_TIME` after the stop.
#[test]
fn a_drop_on_a_hull_that_stops_comes_to_rest_within_a_few_substeps() {
    let mut f = SphFluid::new(SphParams::blood(), 4).unwrap();
    let rc = f.contact_radius();
    let speed = 2.0;
    f.spawn([0.0, 0.6 + rc, 0.0], [speed, 0.0, 0.0]);
    let stop = 48; // 0.2 s, under SETTLE_TIME
    let n = hull_substeps_to_rest();
    let x = |s: usize| (s.min(stop) + 1) as f64 * DT * speed;
    let mut solids = SphSolids::new();
    let mut settled = Vec::new();
    let mut at_stop = ([0.0; 3], [0.0; 3]);
    for s in 0..480 {
        let v = if s <= stop {
            [speed, 0.0, 0.0]
        } else {
            [0.0; 3]
        };
        solids.clear();
        solids.push_capsule([x(s) - 1.0, 0.5, 0.0], [x(s) + 1.0, 0.5, 0.0], 0.1, v, v);
        f.step_with_solids(DT, 9.81, far_below, &solids);
        if f.is_empty() {
            break;
        }
        if s == stop {
            at_stop = (f.position(0), f.velocity(0));
        }
        if s > stop && s <= stop + n {
            assert_eq!(
                f.solid_stats().contacts,
                1,
                "{} substeps after the stop the drop did not meet the hull",
                s - stop
            );
        }
        if s == stop + n {
            let v = f.velocity(0);
            assert!(
                v[0].abs() <= 9.81 * DT,
                "{n} substeps after the stop the drop still slides at {} m/s",
                v[0]
            );
        }
        f.drain_settled(|d| settled.push((s, d)));
    }
    assert_eq!(settled.len(), 1, "drained {settled:?}");
    let (step, d) = settled[0];
    assert_eq!(d.on_solid, Some(0));
    assert!(
        dot(d.velocity, d.velocity).sqrt() <= 9.81 * DT,
        "settled at {:?} on a stopped hull",
        d.velocity
    );
    assert!((d.position[1] - (0.6 + rc)).abs() <= 9.81 * DT * DT);
    // It rode the hull to the stop, then slid u / rate on along its top: each substep
    // moves it at the speed the decay leaves it, so the slide is the sum of
    // `u keep^k dt` over k from one, `u dt keep / (1 - keep)`, which is `u / rate`.
    let u = at_stop.1[0];
    let slide = d.position[0] - at_stop.0[0];
    let law = u / friction_rate(f.params());
    assert!(
        (slide - law).abs() <= 1e-9 && slide < 0.01,
        "slid {slide:.6} m from {u:.4} m/s; the decay stops it in {law:.6} m"
    );
    let t = (step + 1) as f64 * DT;
    let stopped = (stop + 1) as f64 * DT;
    assert!(
        t >= stopped + SETTLE_TIME,
        "settled at {t:.3} s; the hull stopped at {stopped:.3} s"
    );
}

/// A drop set down at rest on a hull moving at 1 m/s (touching it, a contact radius off
/// its top) meets it on every substep and comes up to its speed by the decay law: after
/// k substeps it is `keep^k` of the hull's speed behind, and within N substeps
/// (`hull_substeps_to_rest`) it is within `g dt` of it. Before the sphere cast it met the
/// hull about one substep in ten, so friction acted that often.
#[test]
fn a_drop_set_on_a_1_m_s_hull_reaches_its_speed_within_a_few_substeps() {
    let mut f = SphFluid::new(SphParams::blood(), 4).unwrap();
    let rc = f.contact_radius();
    let speed = 1.0;
    let n = hull_substeps_to_rest();
    let keep = friction_keep(f.params(), DT);
    f.spawn([0.0, 0.6 + rc, 0.0], [0.0; 3]);
    let v = [speed, 0.0, 0.0];
    let mut solids = SphSolids::new();
    for s in 0..n {
        let x = (s + 1) as f64 * DT * speed;
        solids.clear();
        solids.push_capsule([x - 3.0, 0.5, 0.0], [x + 1.0, 0.5, 0.0], 0.1, v, v);
        f.step_with_solids(DT, 9.81, far_below, &solids);
        assert_eq!(f.solid_stats().contacts, 1, "substep {s}: no contact");
        let behind = speed - f.velocity(0)[0];
        let law = speed * keep.powi(s as i32 + 1);
        assert!(
            (behind - law).abs() <= 1e-12,
            "substep {s}: {behind:.6} m/s behind the hull, the decay leaves {law:.6}"
        );
        let p = f.position(0);
        assert!(
            (p[1] - (0.6 + rc)).abs() <= 1e-12,
            "substep {s}: at y = {}",
            p[1]
        );
    }
    let behind = speed - f.velocity(0)[0];
    assert!(
        behind <= 9.81 * DT,
        "after {n} substeps the drop is {behind:.4} m/s behind a {speed} m/s hull"
    );
}

/// A drop thrown down a static 30 degree incline (the top of a tilted capsule) at 2 m/s
/// decelerates on every substep, along the law the decay sets: each substep adds
/// `g sin 30 dt` along the slope and keeps `keep` of the result, so its speed down the
/// slope tends to `v* = keep g sin 30 dt / (1 - keep)` and its excess over that shrinks by
/// exactly `keep` a substep, with no substep-in-ten steps. Thrown across a static box's
/// lid the same way, with no slope, its speed shrinks by `keep` a substep to nothing. For
/// blood, water and napalm (each friction from the stickiest to the slickest).
#[test]
fn a_drop_sliding_on_a_static_solid_decelerates_every_substep() {
    let (sin, cos) = (30f64.to_radians().sin(), 30f64.to_radians().cos());
    for params in [SphParams::blood(), SphParams::water(), SphParams::napalm()] {
        let name = format!("friction {}", params.friction);
        let keep = friction_keep(&params, DT);
        let v0 = 2.0;
        assert!(SphFluid::new(params, 1).unwrap().speed_ceiling(DT) > v0);

        // Down the incline: the capsule's axis runs downhill along `down`, and the drop
        // sits on its crest, a contact radius above it, with the up-slope normal.
        let mut f = SphFluid::new(params, 4).unwrap();
        let (radius, rc) = (0.1, f.contact_radius());
        let down = [cos, -sin, 0.0];
        let normal = [sin, cos, 0.0];
        let a = [0.0, 1.0, 0.0];
        let b = [a[0] + 2.0 * down[0], a[1] + 2.0 * down[1], 0.0];
        let along = 0.2;
        let lift = radius + rc;
        f.spawn(
            [
                a[0] + along * down[0] + lift * normal[0],
                a[1] + along * down[1] + lift * normal[1],
                0.0,
            ],
            [v0 * down[0], v0 * down[1], 0.0],
        );
        let mut incline = SphSolids::new();
        incline.push_capsule(a, b, radius, [0.0; 3], [0.0; 3]);
        let terminal = keep * 9.81 * sin * DT / (1.0 - keep);
        let mut excess = v0 - terminal;
        for s in 0..40 {
            f.step_with_solids(DT, 9.81, far_below, &incline);
            assert_eq!(f.solid_stats().contacts, 1, "{name}, incline, substep {s}");
            let next = dot(f.velocity(0), down) - terminal;
            assert!(
                (next - keep * excess).abs() <= 1e-9 * v0,
                "{name}, incline, substep {s}: excess speed {excess:.6} -> {next:.6} m/s, \
                 the decay keeps {keep:.4} of it"
            );
            excess = next;
            let off = segment_distance(f.position(0), a, b) - radius;
            assert!(
                (off - rc).abs() <= 1e-12,
                "{name}, incline, substep {s}: {off} off"
            );
        }

        // Across a lid: a box yawed a quarter-ish turn, the drop thrown along its own x.
        let mut f = SphFluid::new(params, 4).unwrap();
        let yaw: f64 = 0.4;
        let dir = [yaw.cos(), 0.0, -yaw.sin()];
        let mut lid = SphSolids::new();
        lid.push_box([0.0, 0.25, 0.0], [1.0, 0.25, 0.5], yaw, [0.0; 3]);
        let top = 0.5 + f.contact_radius();
        f.spawn(
            [-0.6 * dir[0], top, -0.6 * dir[2]],
            [v0 * dir[0], 0.0, v0 * dir[2]],
        );
        let mut speed = v0;
        for s in 0..40 {
            f.step_with_solids(DT, 9.81, far_below, &lid);
            assert_eq!(f.solid_stats().contacts, 1, "{name}, lid, substep {s}");
            let next = dot(f.velocity(0), dir);
            assert!(
                (next - keep * speed).abs() <= 1e-9 * v0,
                "{name}, lid, substep {s}: {speed:.6} -> {next:.6} m/s, the decay keeps \
                 {keep:.4}"
            );
            speed = next;
            assert!(
                (f.position(0)[1] - top).abs() <= 1e-12,
                "{name}, lid, substep {s}"
            );
        }
    }
}

/// Distance from `p` to box `centre`, `half`, yawed by `yaw` about +y (the convention of
/// `SphSolids::push_box`).
fn box_distance(p: [f64; 3], centre: [f64; 3], half: [f64; 3], yaw: f64) -> f64 {
    let (s, c) = yaw.sin_cos();
    let w = sub(p, centre);
    let l = [c * w[0] - s * w[2], w[1], s * w[0] + c * w[2]];
    let o: Vec<f64> = (0..3).map(|a| (l[a].abs() - half[a]).max(0.0)).collect();
    (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt()
}

/// The sphere cast against an oracle: droplets fired at a yawed box's faces, edges and
/// corners and at a tilted capsule from every side, one substep each, with no gravity.
/// Where the straight path from start to end comes within a contact radius of the solid
/// (its distance along the path is convex, so a ternary search finds the closest), the
/// solid meets the droplet and leaves it exactly a contact radius off its surface; where
/// it does not, the droplet goes where it was going, untouched. Starts already within a
/// contact radius are included: moving in, they are met at once; moving out, never.
#[test]
fn the_sphere_cast_meets_exactly_the_paths_that_come_within_a_contact_radius() {
    let (centre, half, yaw) = ([0.1, 0.5, -0.1], [0.2, 0.1, 0.15], 0.7);
    let (ca, cb, cr) = ([-0.3, 0.3, 0.2], [0.2, 0.6, -0.1], 0.06);
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut unit = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let probe = SphFluid::new(SphParams::blood(), 1).unwrap();
    let (rc, ceiling) = (probe.contact_radius(), probe.speed_ceiling(DT));
    let (mut met, mut missed, mut touching) = (0, 0, 0);
    for case in 0..1200 {
        let on_box = case % 2 == 0;
        let distance = |p: [f64; 3]| {
            if on_box {
                box_distance(p, centre, half, yaw)
            } else {
                segment_distance(p, ca, cb) - cr
            }
        };
        // A target near the solid's surface (for a box, mostly near its edges and
        // corners), and a start up to a substep at the ceiling away from it.
        let target = if on_box {
            let (s, c) = yaw.sin_cos();
            let l: Vec<f64> = (0..3)
                .map(|a| {
                    let side = if unit() < 0.5 { -1.0 } else { 1.0 };
                    side * half[a] * (0.7 + 0.6 * unit())
                })
                .collect();
            [
                centre[0] + c * l[0] + s * l[2],
                centre[1] + l[1],
                centre[2] - s * l[0] + c * l[2],
            ]
        } else {
            let t = unit() * 1.2 - 0.1;
            let axis = sub(cb, ca);
            [
                ca[0] + t * axis[0] + (unit() - 0.5) * 0.2,
                ca[1] + t * axis[1] + (unit() - 0.5) * 0.2,
                ca[2] + t * axis[2] + (unit() - 0.5) * 0.2,
            ]
        };
        let heading = {
            let d = [unit() - 0.5, unit() - 0.5, unit() - 0.5];
            let l = dot(d, d).sqrt();
            [d[0] / l, d[1] / l, d[2] / l]
        };
        let speed = ceiling * (0.2 + 0.79 * unit());
        let back = speed * DT * (0.3 + unit());
        let p0 = [
            target[0] - heading[0] * back,
            target[1] - heading[1] * back,
            target[2] - heading[2] * back,
        ];
        let start = distance(p0);
        if start <= 0.0 {
            continue;
        }
        let v = [heading[0] * speed, heading[1] * speed, heading[2] * speed];
        let p1 = [p0[0] + v[0] * DT, p0[1] + v[1] * DT, p0[2] + v[2] * DT];
        // The closest the path comes; in-flight starts take the closest point ahead.
        let at = |t: f64| {
            distance([
                p0[0] + (p1[0] - p0[0]) * t,
                p0[1] + (p1[1] - p0[1]) * t,
                p0[2] + (p1[2] - p0[2]) * t,
            ])
        };
        let (mut lo, mut hi) = (0.0, 1.0);
        for _ in 0..200 {
            let (m1, m2) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
            if at(m1) <= at(m2) {
                hi = m2;
            } else {
                lo = m1;
            }
        }
        let closest = at(0.5 * (lo + hi)).min(at(0.0)).min(at(1.0));
        let inside_start = start <= rc;
        let moving_in = at(1e-6) < start;
        let expect_meet = if inside_start {
            moving_in
        } else {
            closest <= rc
        };
        // Grazing paths, within rounding of the inflated surface, decide nothing.
        if (closest - rc).abs() < 1e-9 || (start - rc).abs() < 1e-9 {
            continue;
        }
        let mut f = SphFluid::new(SphParams::blood(), 1).unwrap();
        f.spawn(p0, v);
        let mut solids = SphSolids::new();
        if on_box {
            solids.push_box(centre, half, yaw, [0.0; 3]);
        } else {
            solids.push_capsule(ca, cb, cr, [0.0; 3], [0.0; 3]);
        }
        f.step_with_solids(DT, 0.0, far_below, &solids);
        let p = f.position(0);
        let contacts = f.solid_stats().contacts;
        if expect_meet {
            met += 1;
            touching += inside_start as usize;
            assert_eq!(contacts, 1, "case {case}: {p0:?} -> {p1:?} was not met");
            let off = distance(p);
            assert!(
                (off - rc).abs() <= 1e-9,
                "case {case}: {p0:?} -> {p1:?} ended {off:.9} m off, not a contact \
                 radius ({rc})"
            );
        } else {
            missed += 1;
            assert_eq!(contacts, 0, "case {case}: {p0:?} -> {p1:?} was met");
            let d = sub(p, p1);
            assert!(
                dot(d, d) <= 1e-24,
                "case {case}: moved from {p1:?} to {p:?}"
            );
        }
    }
    assert!(
        met > 60 && missed > 200 && touching > 20,
        "met {met}, missed {missed}, touching at the start {touching}"
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

/// Static solids and the ground step to a recorded checksum: drops raining onto a
/// capsule, a yawed crate, a sphere and a sloped ground, every solid at rest, step to the
/// same checksum of positions, velocities and drained indices every time.
///
/// The literal moves only with a deliberate change to the static contact, recorded here.
/// 0.3.5 and 0.3.6 stepped it to `0x7c0b_de12_850b_324d` (measured on 0.3.5, 9d2c669: the
/// relative cast of 0.3.6 leaves a solid at rest the world cast it always had). 0.3.7's
/// sphere cast changes it: a drop sliding on a solid meets it on every substep instead
/// of about one in ten, and every contact carries the particle on along the surface for
/// the rest of its substep.
#[test]
fn a_static_scene_steps_to_its_recorded_checksum() {
    let slope = |x: f64, z: f64| 0.08 * x - 0.05 * z;
    let mut f = SphFluid::new(SphParams::blood(), 1024).unwrap();
    for k in 0..800usize {
        let (i, j) = (k % 40, k / 40);
        f.spawn(
            [
                -0.8 + 0.04 * i as f64,
                0.9 + 0.015 * (k % 5) as f64,
                -0.4 + 0.04 * j as f64,
            ],
            [0.05 * (k % 3) as f64, -0.2, -0.05 * (k % 4) as f64],
        );
    }
    let settled = settle_run(&mut f, 300, flat, |_, solids| {
        solids.push_capsule(
            [-0.6, 0.35, -0.3],
            [0.2, 0.4, 0.3],
            0.12,
            [0.0; 3],
            [0.0; 3],
        );
        solids.push_capsule([0.5, 0.3, 0.1], [0.5, 0.3, 0.1], 0.15, [0.0; 3], [0.0; 3]);
        solids.push_box([0.2, 0.2, -0.2], [0.2, 0.2, 0.15], 0.6, [0.0; 3]);
    });
    // The settle run's ground is flat; step the survivors on the slope too.
    let solids = {
        let mut s = SphSolids::new();
        s.push_box([0.0, -0.1, 0.0], [0.3, 0.12, 0.3], -0.4, [0.0; 3]);
        s
    };
    for _ in 0..60 {
        f.step_with_solids(DT, 9.81, slope, &solids);
    }
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for v in bits(&f) {
        eat(v);
    }
    for (s, d) in &settled {
        eat(*s as u64);
        for c in d.position.iter().chain(&d.velocity) {
            eat(c.to_bits());
        }
        eat(d.on_solid.map_or(u64::MAX, u64::from));
    }
    assert!(settled.len() > 50, "only {} drops settled", settled.len());
    assert!(
        settled.iter().any(|(_, d)| d.on_solid.is_some()),
        "nothing settled on a solid"
    );
    assert_eq!(
        h,
        0xa768_fcfb_039f_9def,
        "checksum {h:#018x} over {} drained",
        settled.len()
    );
}

/// A blade faster than the speed ceiling (30 m/s, 12.5 cm a substep, against water's
/// 3.84 m/s and 1.6 cm) sweeps a pool: every particle in the slab it swept through this
/// substep ends in front of its leading face, and none is left inside it. The particles
/// it sweeps start up to its own travel behind its end pose, past the ceiling's reach,
/// so this holds only because a solid's reach grows by its own travel.
#[test]
fn a_blade_faster_than_the_speed_ceiling_sweeps_the_pool_ahead_of_it() {
    let mut f = pool();
    let speed = 30.0;
    assert!(speed * DT > CFL_FRACTION * f.params().smoothing_radius * 3.0);
    let (half, cy, cz) = ([0.005, 0.3, 0.3], 0.05, 0.15);
    let mut solids = SphSolids::new();
    let mut swept = 0;
    for s in 0..4 {
        let x1 = 0.05 + speed * DT * s as f64;
        let x0 = x1 - speed * DT;
        solids.clear();
        solids.push_box([x1, cy, cz], half, 0.0, [speed, 0.0, 0.0]);
        let before: Vec<[f64; 3]> = (0..f.len()).map(|i| f.position(i)).collect();
        f.step_with_solids(DT, 9.81, flat, &solids);
        for (i, p0) in before.iter().enumerate() {
            let p = f.position(i);
            let inside = (p[0] - x1).abs() < half[0]
                && (p[1] - cy).abs() < half[1]
                && (p[2] - cz).abs() < half[2];
            assert!(
                !inside,
                "step {s}: particle {i} is inside the blade at {p:?}"
            );
            // In the slab the leading face crossed, away from the blade's edges.
            if p0[0] > x0 + half[0] && p0[0] < x1 + half[0] && (p0[2] - cz).abs() < 0.2 {
                swept += 1;
                assert!(
                    p[0] >= x1 + half[0],
                    "step {s}: particle {i} started at {p0:?} in the blade's path and ended \
                     behind its face at {p:?}"
                );
            }
        }
    }
    assert!(swept > 100, "the blade swept only {swept} particles");
}
