//! Tests for the contact record: which solid each particle met in a step and how hard,
//! read through `SphFluid::contacts`. A child module of `sph`, beside the solids tests.

use std::collections::HashMap;

use super::*;

const DT: f64 = 1.0 / 240.0;
const G: f64 = 9.81;

fn flat(_x: f64, _z: f64) -> f64 {
    0.0
}

fn far_below(_x: f64, _z: f64) -> f64 {
    -100.0
}

/// A post five metres off, pushed first so the solid under test is index 1.
fn far_post(solids: &mut SphSolids) {
    solids.push_capsule([5.0, 0.0, 5.0], [5.0, 1.0, 5.0], 0.06, [0.0; 3], [0.0; 3]);
}

/// A post of radius 5 cm standing on the y axis, from the ground to 2 m.
const POST_RADIUS: f64 = 0.05;

fn post(solids: &mut SphSolids) {
    solids.push_capsule(
        [0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        POST_RADIUS,
        [0.0; 3],
        [0.0; 3],
    );
}

/// Every step's contacts, with the record's own count checked against the walk and the
/// solids' statistics on the way.
fn record(f: &SphFluid) -> Vec<Contact> {
    let contacts: Vec<Contact> = f.contacts().collect();
    assert_eq!(contacts.len(), f.contacts_len(), "the walk and the count disagree");
    assert_eq!(
        f.contacts_len(),
        f.solid_stats().contacts,
        "the record and the solids' statistics disagree"
    );
    contacts
}

/// A drop thrown at a post, on a path through its axis and climbing along it, strikes
/// it once, reported as the post's index with the normal part of its throw as the
/// approach speed. Gravity runs along the post's axis, so it changes only the
/// tangential part and the normal part at the strike is the thrown one.
#[test]
fn a_drop_thrown_at_a_capsule_reports_it_with_the_normal_part_of_its_speed() {
    let mut f = SphFluid::new(SphParams::blood(), 4).unwrap();
    // Across the ground at (2, 1) m/s, from 5 cm behind the axis line, so its path in
    // the ground plane crosses the axis at x = 0, z = 0; and 0.3 m/s up the post.
    let throw = [2.0, 0.3, 1.0];
    f.spawn([-0.1, 1.0, -0.05], throw);
    let normal = (throw[0] * throw[0] + throw[2] * throw[2]).sqrt();
    let mut solids = SphSolids::new();
    far_post(&mut solids);
    post(&mut solids);
    let mut strike = None;
    for s in 0..24 {
        f.step_with_solids(DT, G, far_below, &solids);
        let contacts = record(&f);
        if strike.is_none() && !contacts.is_empty() {
            assert_eq!(contacts.len(), 1, "step {s}: {contacts:?}");
            strike = Some((s, contacts[0]));
        }
    }
    let (s, c) = strike.expect("the drop never struck the post");
    assert_eq!((c.particle, c.solid), (0, 1), "step {s}: {c:?}");
    assert!(
        (c.approach_speed - normal).abs() <= 0.01 * normal,
        "struck at {} m/s into the post, against a thrown normal part of {normal} m/s",
        c.approach_speed
    );
    // It reaches the inflated post (radius plus a contact radius) in the step the
    // throw's path says it does.
    let gap = (0.1f64.hypot(0.05) - POST_RADIUS - f.contact_radius()) / normal;
    assert_eq!(s, (gap / DT).floor() as usize, "struck on step {s}");
}

/// A drop thrown past a post, half a contact radius clear of where it would touch it,
/// is cast against the post (it starts within its reach) and never meets it: nothing is
/// recorded.
#[test]
fn a_drop_that_misses_reports_nothing() {
    let mut f = SphFluid::new(SphParams::blood(), 4).unwrap();
    let rc = f.contact_radius();
    f.spawn([-0.1, 1.0, POST_RADIUS + 1.5 * rc], [2.0, 0.0, 0.0]);
    let mut solids = SphSolids::new();
    post(&mut solids);
    let mut tested = 0;
    for s in 0..24 {
        f.step_with_solids(DT, G, far_below, &solids);
        assert!(record(&f).is_empty(), "step {s}: {:?}", record(&f));
        tested += f.solid_stats().ray_tested;
    }
    assert!(tested > 0, "the drop never came within the post's reach");
    assert!(f.position(0)[0] > 0.1, "the drop did not pass the post");
}

/// A drop at rest on a crate's lid strikes the crate every step, at no more than one
/// substep's gravity: the response leaves it at most `restitution` of the last approach
/// upward, and one substep's gravity, `g dt`, is all it gains before the next.
#[test]
fn a_drop_at_rest_on_a_box_reports_the_box_every_step() {
    for params in [SphParams::blood(), SphParams::water(), SphParams::napalm()] {
        let mut f = SphFluid::new(params, 4).unwrap();
        let mut solids = SphSolids::new();
        far_post(&mut solids);
        solids.push_box([0.0, 0.25, 0.0], [0.25; 3], 0.3, [0.0; 3]);
        f.spawn([0.05, 0.5 + f.contact_radius(), -0.05], [0.0; 3]);
        for s in 0..120 {
            f.step_with_solids(DT, G, flat, &solids);
            let contacts = record(&f);
            assert_eq!(contacts.len(), 1, "step {s}: {contacts:?}");
            let c = contacts[0];
            assert_eq!((c.particle, c.solid), (0, 1), "step {s}: {c:?}");
            assert!(
                c.approach_speed >= 0.0 && c.approach_speed <= G * DT * (1.0 + 1e-9),
                "step {s}: a resting drop came in at {} m/s, against g dt = {}",
                c.approach_speed,
                G * DT
            );
        }
    }
}

/// The record is the last step's alone: after a step with a contact, a step with no
/// solids, a plain step, a step whose solids reach nothing and a step that refuses its
/// `dt` each leave it empty.
#[test]
fn the_record_is_cleared_between_steps() {
    let mut solids = SphSolids::new();
    post(&mut solids);
    let mut far = SphSolids::new();
    far_post(&mut far);
    let empty = SphSolids::new();
    let clearing: [&dyn Fn(&mut SphFluid); 5] = [
        &|f| f.step_with_solids(DT, G, far_below, &empty),
        &|f| f.step(DT, G, far_below),
        &|f| f.step_with_solids(DT, G, far_below, &far),
        &|f| f.step_with_solids(0.0, G, far_below, &solids),
        &|f| f.step_with_solids(DT, f64::NAN, far_below, &solids),
    ];
    for (k, clear) in clearing.iter().enumerate() {
        let mut f = SphFluid::new(SphParams::blood(), 4).unwrap();
        // Inside the post: pushed out on the first step.
        f.spawn([0.0, 1.0, 0.01], [0.0; 3]);
        f.step_with_solids(DT, G, far_below, &solids);
        assert_eq!(record(&f).len(), 1, "case {k}: the push-out was not recorded");
        clear(&mut f);
        assert_eq!(f.contacts_len(), 0, "case {k}");
        assert_eq!(f.contacts().count(), 0, "case {k}");
    }
}

/// Drops falling onto two crates settle with `Settled::on_solid` naming the crate, and
/// the contact the record held for each the step it drained names the same index.
#[test]
fn the_record_indexes_solids_as_settled_on_solid_does() {
    let mut f = SphFluid::new(SphParams::blood(), 16).unwrap();
    for k in 0..4 {
        let o = 0.1 * k as f64 - 0.15;
        f.spawn([-1.0 + o, 0.7, o], [0.0; 3]);
        f.spawn([1.0 + o, 0.5, 0.5 - o], [0.0; 3]);
    }
    let mut solids = SphSolids::new();
    far_post(&mut solids);
    far_post(&mut solids);
    solids.push_box([-1.0, 0.25, 0.0], [0.25; 3], 0.3, [0.0; 3]);
    solids.push_box([1.0, 0.15, 0.5], [0.3, 0.15, 0.2], -0.4, [0.0; 3]);
    let mut on = [0usize; 4];
    for s in 0..480 {
        f.step_with_solids(DT, G, flat, &solids);
        let mut met: HashMap<[u64; 3], u32> = HashMap::new();
        for c in record(&f) {
            met.insert(f.position(c.particle).map(f64::to_bits), c.solid);
        }
        f.drain_settled(|d| {
            let solid = d.on_solid.expect("a drop settled on the ground");
            assert_eq!(
                met.get(&d.position.map(f64::to_bits)),
                Some(&solid),
                "step {s}: a drop settled on solid {solid} without striking it this step"
            );
            on[solid as usize] += 1;
        });
        // Draining keeps the record at the particle indices.
        assert_eq!(f.contacts().count(), f.contacts_len());
        assert!(f.contacts().all(|c| c.particle < f.len()));
    }
    assert_eq!(on, [0, 0, 4, 4], "settled on each solid: {on:?}");
}

/// Regression: a plain `step`, and `step_with_solids` with an empty set, never record a
/// contact, however many particles hit the ground.
#[test]
fn a_step_without_solids_leaves_the_record_empty() {
    let params = SphParams::blood();
    let spacing = params.smoothing_radius * 0.5;
    let mut plain = SphFluid::new(params, 512).unwrap();
    for k in 0..512usize {
        let p = [
            (k % 8) as f64 * spacing,
            0.05 + (k / 64) as f64 * spacing,
            ((k / 8) % 8) as f64 * spacing,
        ];
        plain.spawn(p, [0.0, -1.0, 0.0]);
    }
    let mut empty = plain.clone();
    let none = SphSolids::new();
    for s in 0..120 {
        plain.step(DT, G, flat);
        empty.step_with_solids(DT, G, flat, &none);
        for f in [&plain, &empty] {
            assert_eq!((f.contacts_len(), f.contacts().count()), (0, 0), "step {s}");
        }
    }
}

/// The record is bit-identical at any thread count, and its count is the solids'
/// statistics' contact count every step, in a pool a capsule wades through.
#[test]
fn the_record_is_bit_identical_at_any_thread_count() {
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let params = SphParams::blood();
            let spacing = params.smoothing_radius * 0.5;
            let mut f = SphFluid::new(params, 2048).unwrap();
            for k in 0..2048usize {
                let p = [
                    (k % 32) as f64 * spacing,
                    0.5 * spacing + (k / 512) as f64 * spacing,
                    ((k / 32) % 16) as f64 * spacing,
                ];
                f.spawn(p, [0.0; 3]);
            }
            let mut solids = SphSolids::new();
            let mut out = Vec::new();
            for s in 0..96 {
                let x = -0.08 + (s + 1) as f64 * DT;
                let v = [1.0, 0.0, 0.0];
                solids.clear();
                solids.push_capsule([x, -0.02, 0.15], [x, 0.3, 0.15], 0.04, v, v);
                solids.push_box([0.3, 0.05, 0.1], [0.05; 3], 0.2, [0.0; 3]);
                f.step_with_solids(DT, G, flat, &solids);
                for c in record(&f) {
                    out.push([c.particle as u64, c.solid as u64, c.approach_speed.to_bits()]);
                }
            }
            out
        })
    };
    let one = run(1);
    assert!(one.len() > 1000, "only {} contacts recorded", one.len());
    for t in [3, 8] {
        assert!(run(t) == one, "{t} threads recorded differently from one");
    }
}
