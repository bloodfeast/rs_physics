//! The readback ring: four `MAP_READ` slots, each mapped by `map_buffer_on_submit` and
//! freed by [`Ring::take`]. The map callback fires inside a later `Queue::submit` (which
//! is where wgpu polls); nothing here polls or waits.
//!
//! A slot is idle, pending (copied into, its map requested) or ready (mapped, its callback
//! fired). A dispatch whose slot is not idle is skipped and counted, rather than recorded
//! into a buffer that is still mapped, which wgpu would reject at submit.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use super::records::{AcousticCounters, ListenerField, SourceResult, TickResults, READBACK_BYTES};

const IDLE: u8 = 0;
const PENDING: u8 = 1;
const READY: u8 = 2;
const FAILED: u8 = 3;

/// Readback slots. Four covers the two frames wgpu keeps in flight plus a frame of the
/// caller not having taken its results yet (Carmack R7).
pub(crate) const SLOTS: usize = 4;

/// Sets a slot's state when its map answers, and returns it to idle if the encoder that
/// carried the request was dropped unsubmitted (wgpu then drops the callback uncalled).
struct Answer(Arc<AtomicU8>);

impl Answer {
    fn answer(self, ok: bool) {
        self.0
            .store(if ok { READY } else { FAILED }, Ordering::Release);
    }
}

impl Drop for Answer {
    fn drop(&mut self) {
        let _ = self
            .0
            .compare_exchange(PENDING, IDLE, Ordering::AcqRel, Ordering::Acquire);
    }
}

struct Slot {
    buffer: wgpu::Buffer,
    state: Arc<AtomicU8>,
    sequence: u64,
    len: u32,
}

pub(crate) struct Ring {
    slots: [Slot; SLOTS],
    next: usize,
    sequence: u64,
    counters: AcousticCounters,
}

impl Ring {
    pub(crate) fn new(device: &wgpu::Device) -> Ring {
        let slot = |i: usize| Slot {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(
                    [
                        "acoustics readback 0",
                        "acoustics readback 1",
                        "acoustics readback 2",
                        "acoustics readback 3",
                    ][i],
                ),
                size: READBACK_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            state: Arc::new(AtomicU8::new(IDLE)),
            sequence: 0,
            len: 0,
        };
        Ring {
            slots: [slot(0), slot(1), slot(2), slot(3)],
            next: 0,
            sequence: 0,
            counters: AcousticCounters::default(),
        }
    }

    /// The next slot, if it is idle: its index and the dispatch's sequence number. A busy
    /// slot is skipped and counted, and the cursor moves past it.
    pub(crate) fn claim(&mut self, sources: u32) -> Option<(usize, u64)> {
        let k = self.next % SLOTS;
        self.next = self.next.wrapping_add(1);
        let slot = &mut self.slots[k];
        match slot.state.load(Ordering::Acquire) {
            IDLE => {}
            FAILED => {
                slot.state.store(IDLE, Ordering::Release);
                self.counters.map_failures += 1;
            }
            _ => {
                self.counters.skipped_busy += 1;
                return None;
            }
        }
        self.sequence += 1;
        slot.sequence = self.sequence;
        slot.len = sources;
        Some((k, self.sequence))
    }

    /// Copy the output into slot `k` and ask for it to be mapped when `enc` is submitted.
    pub(crate) fn record(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        output: &wgpu::Buffer,
        k: usize,
    ) {
        let slot = &self.slots[k];
        enc.copy_buffer_to_buffer(output, 0, &slot.buffer, 0, READBACK_BYTES);
        slot.state.store(PENDING, Ordering::Release);
        let answer = Answer(Arc::clone(&slot.state));
        enc.map_buffer_on_submit(&slot.buffer, wgpu::MapMode::Read, .., move |r| {
            answer.answer(r.is_ok())
        });
        self.counters.dispatches += 1;
    }

    pub(crate) fn take(&mut self, out: &mut TickResults) -> bool {
        let mut newest: Option<(usize, u64)> = None;
        for (k, slot) in self.slots.iter().enumerate() {
            match slot.state.load(Ordering::Acquire) {
                READY => {
                    if newest.is_none_or(|(_, seq)| seq < slot.sequence) {
                        newest = Some((k, slot.sequence));
                    }
                }
                FAILED => {
                    slot.state.store(IDLE, Ordering::Release);
                    self.counters.map_failures += 1;
                }
                _ => {}
            }
        }
        let newest = newest.map(|(k, _)| k);
        let Some(newest) = newest else {
            return false;
        };
        for (k, slot) in self.slots.iter().enumerate() {
            if slot.state.load(Ordering::Acquire) != READY {
                continue;
            }
            if k == newest {
                let view = slot
                    .buffer
                    .slice(..)
                    .get_mapped_range()
                    .expect("a slot whose map callback reported success is mapped");
                let bytes: &[u8] = &view;
                let results = MAX_RESULT_BYTES;
                out.sources
                    .copy_from_slice(bytemuck::cast_slice::<u8, SourceResult>(&bytes[..results]));
                out.field = bytemuck::pod_read_unaligned::<ListenerField>(&bytes[results..]);
                out.len = slot.len as usize;
                out.sequence = slot.sequence;
                drop(view);
            } else {
                self.counters.overwritten_ready += 1;
            }
            slot.buffer.unmap();
            slot.state.store(IDLE, Ordering::Release);
        }
        true
    }

    pub(crate) fn counters(&self) -> AcousticCounters {
        self.counters
    }
}

const MAX_RESULT_BYTES: usize = super::records::MAX_SOURCES as usize * 32;
