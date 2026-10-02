//! Tests for [`PlumeField`]: the summed field's divergence, a plume leaning in the
//! wind, and the triple buffer never handing the reader a half-written frame.

use super::plume_field::test_access::triple_buffer;
use super::{PlumeField, PlumeRegion, PlumeSource};
use crate::particles::{TurbulenceDrive, VelocityGrid};

fn source_at(position: [f32; 3]) -> PlumeSource {
    PlumeSource {
        position,
        drive: TurbulenceDrive::new(3.0, 4.0).unwrap(),
        smoke_rate: 1.0,
    }
}

fn checksum(grid: &VelocityGrid) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for c in grid.cells() {
        for v in c {
            for b in v.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
    h
}

/// The summed field's divergence is the grid's (what its projection leaves) plus the
/// swirl's, which is zero to `f32` rounding: summing adds nothing to it.
#[test]
fn the_summed_field_is_as_divergence_free_as_the_grid() {
    let region = PlumeRegion {
        origin: [-8.0, 0.0, -8.0],
        cells: [16, 16, 16],
        cell_size: 1.0,
    };
    let (mut plume, mut air) =
        PlumeField::new(region, source_at([0.0, 0.5, 0.0]), [1.5, 0.0, 0.5], 3).unwrap();
    for _ in 0..20 {
        plume.step_now(0.05);
    }
    let frame = air.latest();
    let summed = frame.velocity();
    let dims = summed.dims();

    // The grid's flow alone, in m/s, on the same cells.
    let mut flow = VelocityGrid::new(summed.origin(), 1.0, dims).unwrap();
    let scale = plume.metres_per_grid_unit();
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            for k in 0..dims[2] {
                let (u, v, w) = plume.fluid().get_velocity(i, j, k).unwrap();
                flow.set(
                    i,
                    j,
                    k,
                    [(u * scale) as f32, (v * scale) as f32, (w * scale) as f32],
                );
            }
        }
    }
    let swirl = plume.swirl_for_tests().velocity();

    let (mut rms_summed, mut rms_flow, mut worst_gap, mut peak) = (0.0f64, 0.0f64, 0.0f32, 0.0f32);
    let mut count = 0usize;
    for i in 2..dims[0] - 2 {
        for j in 2..dims[1] - 2 {
            for k in 2..dims[2] - 2 {
                let (s, f, w) = (
                    summed.divergence_at(i, j, k),
                    flow.divergence_at(i, j, k),
                    swirl.divergence_at(i, j, k),
                );
                rms_summed += (s as f64).powi(2);
                rms_flow += (f as f64).powi(2);
                worst_gap = worst_gap.max((s - f).abs()).max(w.abs());
                peak = peak.max(
                    summed
                        .get(i, j, k)
                        .iter()
                        .fold(0.0f32, |m, c| m.max(c.abs())),
                );
                count += 1;
            }
        }
    }
    let (rms_summed, rms_flow) = (
        (rms_summed / count as f64).sqrt(),
        (rms_flow / count as f64).sqrt(),
    );
    println!("rms divergence: summed {rms_summed:e}, grid alone {rms_flow:e}; worst swirl contribution {worst_gap:e} (peak speed {peak})");
    // A cell's velocities are rounded to f32 and differenced: a few epsilons of the
    // peak speed over a cell.
    let rounding = 32.0 * f32::EPSILON * peak / region.cell_size;
    assert!(
        worst_gap <= rounding,
        "the swirl added divergence {worst_gap} (rounding {rounding})"
    );
    assert!(
        rms_summed <= rms_flow + rounding as f64,
        "{rms_summed} vs {rms_flow}"
    );
}

/// A buoyant column in a side wind leans downwind: the smoke's centroid at height
/// moves the way the wind blows, and with no wind it stays over the source.
#[test]
fn a_plume_leans_downwind() {
    let centroid_at_height = |wind: f32| {
        let region = PlumeRegion {
            origin: [-10.0, 0.0, -10.0],
            cells: [20, 22, 20],
            cell_size: 1.0,
        };
        let (mut plume, _air) =
            PlumeField::new(region, source_at([0.0, 0.5, 0.0]), [wind, 0.0, 0.0], 5).unwrap();
        for _ in 0..80 {
            plume.step_now(0.05);
        }
        // Fluid cell `i` is centred at origin + (i - 0.5) h; the layer 12 m up.
        let j = 12;
        let (mut mass, mut moment) = (0.0f64, 0.0f64);
        for i in 1..=20 {
            for k in 1..=20 {
                let rho = plume.fluid().get_density(i, j, k).unwrap();
                mass += rho;
                moment += rho * (-10.0 + i as f64 - 0.5);
            }
        }
        assert!(mass > 0.0, "no smoke reached 12 m in wind {wind}");
        moment / mass
    };
    let still = centroid_at_height(0.0);
    let east = centroid_at_height(2.0);
    let west = centroid_at_height(-2.0);
    println!("smoke centroid 12 m up: still {still:.2} m, wind +2 m/s {east:.2} m, wind -2 m/s {west:.2} m");
    assert!(still.abs() < 0.5, "the still plume drifted to {still}");
    assert!(east > still + 1.0, "{east} vs {still}");
    assert!(west < still - 1.0, "{west} vs {still}");
}

/// The reader never sees a half-written frame: a writer on another thread fills each
/// frame whole with its own step number and publishes it, and every frame the reader
/// takes is uniform, carries the step it was filled with, and is never older than the
/// last one it took.
#[test]
fn the_triple_buffer_never_tears() {
    let (mut write, mut reader) = triple_buffer([16, 16, 16]);
    let last = 3000u64;
    let writer = std::thread::spawn(move || {
        for step in 1..=last {
            write(step as f32, step);
        }
    });
    let (mut seen, mut previous, mut distinct) = (0u64, 0u64, 0u64);
    while previous < last {
        let frame = reader.latest();
        let step = frame.step();
        assert!(step >= previous, "went back from {previous} to {step}");
        if step != previous {
            distinct += 1;
        }
        let want = [step as f32, -(step as f32), step as f32, 0.0];
        assert!(
            frame.velocity().cells().iter().all(|c| *c == want),
            "frame {step} is torn"
        );
        previous = step;
        seen += 1;
    }
    writer.join().unwrap();
    println!("{seen} reads, {distinct} distinct frames");
    assert!(distinct > 1);
}

/// The same with the real worker: every frame read while it steps is exactly the
/// state a synchronous run reaches at that step number.
#[test]
fn every_frame_read_while_the_worker_steps_is_a_finished_step() {
    let region = PlumeRegion {
        origin: [-6.0, 0.0, -6.0],
        cells: [12, 12, 12],
        cell_size: 1.0,
    };
    let wind = [1.0, 0.0, -0.5];
    let rate = 500.0f32;
    let steps = 12u64;

    let (mut replica, mut replica_air) =
        PlumeField::new(region, source_at([0.0, 0.5, 0.0]), wind, 9).unwrap();
    let mut expected = vec![checksum(replica_air.latest().velocity())];
    for _ in 0..steps {
        replica.step_now(1.0 / rate);
        expected.push(checksum(replica_air.latest().velocity()));
    }

    let (plume, mut air) = PlumeField::new(region, source_at([0.0, 0.5, 0.0]), wind, 9).unwrap();
    let worker = plume.spawn(rate).unwrap();
    let mut reads = 0u64;
    loop {
        let frame = air.latest();
        let step = frame.step();
        if step > steps {
            break;
        }
        assert_eq!(
            checksum(frame.velocity()),
            expected[step as usize],
            "frame {step} is not a finished step"
        );
        reads += 1;
        std::thread::yield_now();
    }
    drop(worker);
    println!("{reads} reads while the worker stepped");
    assert!(reads > 0);
}
