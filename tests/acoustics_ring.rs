//! **L10** (results echo their tag, and a reassigned slot's stale result is identifiable)
//! and **L11** (a busy readback slot is skipped and counted, the next free slot is used,
//! and wgpu reports no validation error).
#![cfg(feature = "gpu")]

mod acoustics_common;
use acoustics_common::*;

use rs_physics::acoustics::Air;
use rs_physics::gpu::acoustics::*;

fn staged_dispatch(
    gpu: &Gpu,
    header: &DispatchHeader,
    sources: &[Source],
) -> (wgpu::Buffer, DispatchShape) {
    let mut bytes = vec![0u8; dispatch_bytes(sources.len(), 0)];
    let shape = GpuAcoustics::pack_dispatch(header, sources, &[], &mut bytes[..]).unwrap();
    (gpu.stage(&bytes), shape)
}

fn ridge_scene() -> Scene {
    let mut scene = Scene::flat(60, 60, 2.0, 0.0);
    for r in 0..60 {
        for c in 30..32 {
            scene.heights[r * 60 + c] = 8.0;
        }
    }
    scene
}

#[test]
fn l10_results_echo_their_tags_and_a_reassigned_slot_is_identifiable() {
    let Some(gpu) = gpu() else { return };
    let mut ac = GpuAcoustics::new(&gpu.device, AcousticLimits::MAX).unwrap();
    ridge_scene().load(&gpu, &mut ac);
    let header =
        DispatchHeader::new(&listener_at([20.0, 1.6, 60.0]), &Air::standard(), &[]).unwrap();
    // Tags across the whole 24 bits, sources behind the ridge (flags set) and in front, and
    // an ignore-mover byte that must not leak into the tag.
    let tags = [
        0u32, 1, 2, 0x7F_FFFF, 0x80_0000, MAX_TAG, 12_345, 999_999, 3, 4,
    ];
    let sources: Vec<Source> = tags
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let x = if i % 2 == 0 { 100.0 } else { 30.0 };
            Source::new(
                [x, 1.0, 20.0 + 8.0 * i as f32],
                1.0,
                [0.0; 3],
                t,
                (i as u8) * 20,
            )
            .unwrap()
        })
        .collect();
    // Voice slots: who owns slot i when the dispatch is recorded.
    let mut owner = tags;
    let (stage, shape) = staged_dispatch(&gpu, &header, &sources);
    let mut enc = gpu.encoder();
    ac.encode(
        &mut enc,
        Staged {
            buffer: &stage,
            offset: 0,
            len: shape.bytes,
        },
        shape,
    )
    .unwrap();
    gpu.queue.submit([enc.finish()]);
    // Between dispatch and readback, two voices are replaced.
    owner[3] = 77;
    owner[7] = 78;
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let mut out = TickResults::default();
    assert!(ac.take_ready(&mut out));
    assert_eq!(out.len, tags.len());
    let mut stale = Vec::new();
    for (i, r) in out.results().iter().enumerate() {
        assert_eq!(r.tag(), tags[i], "result {i} did not echo its source's tag");
        if r.tag() != owner[i] {
            stale.push(i);
        }
    }
    assert_eq!(
        stale,
        vec![3, 7],
        "the reassigned slots are exactly the stale ones"
    );
    // The flags ride above the tag without touching it.
    assert!(
        out.results()[0].blocked(),
        "a source behind the ridge was not blocked"
    );
    assert_eq!(out.results()[5].tag(), MAX_TAG);
}

#[test]
fn l11_a_busy_slot_is_skipped_and_counted_and_nothing_is_invalid() {
    let Some(gpu) = gpu() else { return };
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut ac = GpuAcoustics::new(&gpu.device, AcousticLimits::MAX).unwrap();
    ridge_scene().load(&gpu, &mut ac);
    let header =
        DispatchHeader::new(&listener_at([20.0, 1.6, 60.0]), &Air::standard(), &[]).unwrap();
    let s = Source::new([100.0, 1.0, 60.0], 1.0, [0.0; 3], 1, NO_MOVER).unwrap();
    let (stage, shape) = staged_dispatch(&gpu, &header, &[s]);
    let staged = Staged {
        buffer: &stage,
        offset: 0,
        len: shape.bytes,
    };

    // The injected hitch: dispatch 1 is recorded into slot 0 and its encoder held back, so
    // slot 0 waits on a map that has not been submitted.
    let mut held = gpu.encoder();
    assert_eq!(
        ac.encode(&mut held, staged, shape).unwrap(),
        Encoded::Dispatched {
            slot: 0,
            sequence: 1
        }
    );
    for (slot, sequence) in [(1, 2), (2, 3), (3, 4)] {
        let mut enc = gpu.encoder();
        assert_eq!(
            ac.encode(&mut enc, staged, shape).unwrap(),
            Encoded::Dispatched { slot, sequence }
        );
        gpu.submit_wait(enc);
    }
    let mut out = TickResults::default();
    assert!(ac.take_ready(&mut out));
    assert_eq!(out.sequence, 4, "the newest ready result is the one taken");
    assert_eq!(ac.counters().overwritten_ready, 2);

    // Slot 0 is still pending: skipped and counted, and nothing is recorded.
    let mut enc = gpu.encoder();
    assert_eq!(
        ac.encode(&mut enc, staged, shape).unwrap(),
        Encoded::SkippedBusy
    );
    assert_eq!(ac.counters().skipped_busy, 1);
    // The next free slot is used after it.
    assert_eq!(
        ac.encode(&mut enc, staged, shape).unwrap(),
        Encoded::Dispatched {
            slot: 1,
            sequence: 5
        }
    );
    gpu.submit_wait(enc);

    // The hitch clears: the held dispatch lands late, and the newer result still wins.
    gpu.submit_wait(held);
    assert!(ac.take_ready(&mut out));
    assert_eq!(out.sequence, 5);
    assert_eq!(ac.counters().overwritten_ready, 3);
    assert_eq!(ac.counters().dispatches, 5);

    // An encoder dropped unsubmitted hands its slot back rather than leaking it.
    {
        let mut dropped = gpu.encoder();
        assert!(matches!(
            ac.encode(&mut dropped, staged, shape).unwrap(),
            Encoded::Dispatched { slot: 2, .. }
        ));
    }
    for _ in 0..4 {
        let mut enc = gpu.encoder();
        assert!(matches!(
            ac.encode(&mut enc, staged, shape).unwrap(),
            Encoded::Dispatched { .. }
        ));
        gpu.submit_wait(enc);
        assert!(ac.take_ready(&mut out));
    }
    assert_eq!(ac.counters().skipped_busy, 1);

    // Hard limits are errors, not truncation, and record nothing.
    let too_many = vec![s; MAX_SOURCES as usize + 1];
    let mut big = vec![0u8; dispatch_bytes(too_many.len(), 0)];
    assert!(matches!(
        GpuAcoustics::pack_dispatch(&header, &too_many, &[], &mut big[..]),
        Err(AcousticsError::Limit { .. })
    ));
    let small = AcousticLimits {
        sources: 1,
        ..AcousticLimits::MAX
    };
    let mut ac1 = GpuAcoustics::new(&gpu.device, small).unwrap();
    let two = [s, s];
    let (stage2, shape2) = staged_dispatch(&gpu, &header, &two);
    let mut enc = gpu.encoder();
    let err = ac1.encode(
        &mut enc,
        Staged {
            buffer: &stage2,
            offset: 0,
            len: shape2.bytes,
        },
        shape2,
    );
    assert!(matches!(
        err,
        Err(AcousticsError::Limit {
            what: "sources",
            got: 2,
            max: 1
        })
    ));
    assert_eq!(ac1.counters().dispatches, 0);
    gpu.submit_wait(enc);

    let error = pollster::block_on(scope.pop());
    assert!(
        error.is_none(),
        "wgpu reported a validation error: {error:?}"
    );
}
