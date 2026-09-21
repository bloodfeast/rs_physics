//! Handing a whole solver pass to the thread pool once, instead of once per colour.
//!
//! # What this replaces, and the measurement that demanded it
//!
//! A colour must not run alongside the next one, so the obvious shape is a parallel
//! region per colour: `set.par_iter().for_each(..)`, once for each. That is correct, and
//! on the heap this module is benchmarked on it is where most of the time goes.
//!
//! Measured, with the arithmetic and the memory traffic held fixed and only the number of
//! regions varied -- the same `solve_contact` calls over the same 35,925 contacts,
//! writing nothing:
//!
//! ```text
//!   in 1 parallel region                    0.39 ms
//!   in 20, one per colour                   1.66 ms
//!   so a fork and a join cost               67-96 us
//! ```
//!
//! A coloured pass is 2.23 ms, of which those nineteen extra forks are 1.27 ms -- more
//! than half, against seventeen per cent for the contact solve itself. And that is a
//! lower bound on the fork share, because the joint and ground colours are six more
//! regions the measurement does not charge for.
//!
//! **So the pass is handed out once.** One [`rayon::broadcast`] per pass gives every
//! worker the whole list of stages; each takes its own slice of each stage and waits at a
//! [`Gate`] before the next. Twenty-six regions become one, and a barrier costs a
//! handful of microseconds where a fork costs seventy.
//!
//! # Why this needs no new `unsafe`, and what it does need
//!
//! The corrections are still written in place through the raw-pointer views in
//! [`super::scatter`], on exactly the argument that file makes: no two constraints in a
//! colour name the same body. Two things extend it, and neither is new unsafety:
//!
//! * **Lanes within a colour take disjoint slices.** [`Lane::span`] cuts the colour's
//!   list into half-open ranges that partition it exactly, so if no two constraints in
//!   the colour share a body then no two lanes address one body either. The debug check
//!   in [`super::scatter::disjoint`] runs over the whole colour, which is a superset of
//!   every lane's slice, so it still covers precisely what the workers touch.
//! * **The barrier carries the ordering the join used to.** A colour reads bodies the
//!   colour before it wrote. Previously the fork-join gave that happens-before for free.
//!   Now [`Gate::wait`] does: a lane's writes are released when it arrives, and acquired
//!   by every lane that leaves. Without it, colour `n + 1` could read a stale body
//!   through a pointer the compiler is entitled to assume nobody else touched.
//!
//! # Spinning, and where it is allowed
//!
//! The gate spins before it yields, which is only acceptable because of *when* it
//! happens: inside a `step`, between two colours that are microseconds apart, on threads
//! that are already doing the caller's work. Nothing here spins between steps -- the
//! broadcast returns, and rayon's own threads park as they always did. A caller who steps
//! one small skeleton a frame never reaches this code at all; see [`PASS_FLOOR`].
//!
//! The spin budget is not a tuned number. A lane that spins for as long as yielding would
//! have cost has, at worst, spent what it was about to spend anyway, and at best has
//! skipped it entirely -- so the budget is the measured cost of a yield, and being wrong
//! about it by a factor of two costs a few microseconds a pass either way.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Constraints in a pass below which the pass runs on the calling thread.
///
/// **A floor on the pass, where the old one was a floor per colour**, and that is the
/// point rather than an implementation detail. A contact set is divided by twenty-odd
/// colours before it meets a per-colour floor, so the colours drop under it long before
/// the set does: measured on a thousand-body heap, only five colours of thirty-two were
/// over the old floor of 256, and the other twenty-seven ran on the calling thread while
/// the pool sat idle. One region for the whole pass has nothing to divide, so the floor
/// applies once and a colour's own size stops mattering.
///
/// The number is where one broadcast pays for itself: a broadcast costs about 46 us and
/// each stage's barrier about 15 us, against roughly 390 ns of one thread's time per
/// constraint. Measured, a pass of 267 constraints in sixteen stages is faster on one
/// thread and a pass of 1,420 in twenty-six is faster on all of them, so the crossing is
/// somewhere between and this sits in the middle of it. Nothing is sensitive to exactly
/// where: being wrong by a factor of two costs one broadcast on a pass that was cheap
/// anyway.
pub(super) const PASS_FLOOR: usize = 512;

/// How many times a lane spins before yielding. See the module header for why this is the
/// measured cost of a yield rather than a number somebody liked.
const SPINS_PER_YIELD: u32 = 64;

/// How many lanes a pass of `work` constraints in `stages` stages should use.
///
/// **Every thread, or one.** That is not laziness, it is what the measurement says: the
/// step at eight iterations against a forced lane count, on a thirty-six thread machine,
///
/// ```text
///   work    stages     1      2      4      8     12     16     24     36 lanes
///  1,420        26   3.41   4.35   4.06   3.90   3.60   4.31   3.95   3.37 ms
///  4,912        36  12.64   9.11   7.25   7.02   6.64   6.91   6.83   6.81 ms
/// 12,997        34  29.14  19.21  14.70  12.02  10.96   9.80   9.01   8.66 ms
/// 45,525        24 137.80  48.94  30.95  21.66  18.74  16.22  13.85  13.19 ms
/// ```
///
/// There is no interior optimum to find. A first attempt derived a lane count by
/// balancing the barrier against the arithmetic -- a barrier over thirty-six lanes is
/// 15.4 us, a constraint is about 174 ns -- and it chose thirty lanes for the largest
/// case and made it forty per cent slower than using all thirty-six. The model was wrong
/// because a barrier's cost is mostly the wait for the slowest lane to notice, which does
/// not grow with the lane count the way contention on its counter does.
///
/// So the only judgement left is whether to go parallel at all, which is [`PASS_FLOOR`].
fn lanes_for(work: usize, threads: usize) -> usize {
    if work >= PASS_FLOOR {
        threads
    } else {
        1
    }
}

/// Which lane is running, and which stage of the pass it is running.
#[derive(Clone, Copy, Debug)]
pub(super) struct Lane {
    /// Index into the caller's list of stages.
    pub stage: usize,
    index: usize,
    lanes: usize,
}

impl Lane {
    /// This lane's share of `len` items, as a half-open range.
    ///
    /// **Static and exact**: the ranges partition `0..len` with no gap and no overlap,
    /// and they depend only on the lane count, not on who finishes first. That is what
    /// keeps the answer bit-identical across thread counts -- and it is also why there is
    /// no work stealing here, which would make the slicing a function of scheduling.
    #[inline]
    pub fn span(&self, len: usize) -> std::ops::Range<usize> {
        let from = len * self.index / self.lanes;
        let to = len * (self.index + 1) / self.lanes;
        from..to
    }

    /// Whether this is the lane that runs work which cannot be split. Exactly one lane
    /// answers yes.
    #[inline]
    pub fn is_only(&self) -> bool {
        self.index == 0
    }
}

/// A sense-reversing barrier for a fixed number of lanes.
///
/// Sense-reversing so that it can be reused without being reset: a lane flips its own
/// copy of the sense, the last one in flips the shared copy, and there is no window in
/// which a fast lane can run through the next barrier before a slow one has left this
/// one.
struct Gate {
    arrived: AtomicUsize,
    sense: AtomicBool,
    lanes: usize,
}

impl Gate {
    fn new(lanes: usize) -> Self {
        Gate {
            arrived: AtomicUsize::new(0),
            sense: AtomicBool::new(false),
            lanes,
        }
    }

    /// Wait until every lane has arrived.
    ///
    /// `local` is this lane's own copy of the sense and must start `false` and be passed
    /// back unchanged between calls.
    ///
    /// The orderings are what makes this a memory barrier as well as a rendezvous. A lane
    /// arriving releases everything it wrote this stage; a lane leaving acquires
    /// everything every other lane released. That is the happens-before a colour needs
    /// over the colour before it, and before this existed it came from the join.
    fn wait(&self, local: &mut bool) {
        *local = !*local;
        if self.arrived.fetch_add(1, Ordering::AcqRel) == self.lanes - 1 {
            self.arrived.store(0, Ordering::Relaxed);
            self.sense.store(*local, Ordering::Release);
            return;
        }
        let mut spins = 0u32;
        while self.sense.load(Ordering::Acquire) != *local {
            spins += 1;
            if spins < SPINS_PER_YIELD {
                std::hint::spin_loop();
            } else {
                spins = 0;
                std::thread::yield_now();
            }
        }
    }
}

/// Run `work` for every stage, in order, across the pool.
///
/// `work` is called once per lane per stage, and every lane has finished stage `n` before
/// any lane begins stage `n + 1`. Below [`PASS_FLOOR`] worth of work -- which the caller
/// weighs, because only it knows what a stage costs -- the whole thing runs on the
/// calling thread and the pool is never touched.
///
/// The closure must be `Sync` because every lane calls it at once, and it must be
/// prepared for `lanes` to be one: that is not a special case, it is the small-input path.
pub(super) fn each_stage(stages: usize, work_items: usize, work: impl Fn(Lane) + Sync) {
    if stages == 0 {
        return;
    }
    let lanes = lanes_for(work_items, rayon::current_num_threads());
    if lanes <= 1 {
        for stage in 0..stages {
            work(Lane {
                stage,
                index: 0,
                lanes: 1,
            });
        }
        return;
    }

    let gate = Gate::new(lanes);
    rayon::broadcast(|ctx| {
        // The pool hands the broadcast to every thread it has; the ones past the lane
        // count take no work and, crucially, do not join the barrier -- a gate sized for
        // `lanes` that thirty-six threads arrive at would never open the same number of
        // times twice.
        if ctx.index() >= lanes {
            return;
        }
        let mut sense = false;
        for stage in 0..stages {
            work(Lane {
                stage,
                index: ctx.index(),
                lanes,
            });
            gate.wait(&mut sense);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    /// The spans partition the list exactly, at every lane count and every length. A gap
    /// would drop a constraint silently and an overlap would be two threads on one body,
    /// which is the data race everything here is built to avoid.
    #[test]
    fn the_lanes_partition_the_work_exactly() {
        for lanes in 1..=17usize {
            for len in 0..=200usize {
                let mut covered = vec![0u32; len];
                for index in 0..lanes {
                    let lane = Lane {
                        stage: 0,
                        index,
                        lanes,
                    };
                    for i in lane.span(len) {
                        covered[i] += 1;
                    }
                }
                for (i, &times) in covered.iter().enumerate() {
                    assert_eq!(
                        times, 1,
                        "with {lanes} lanes over {len} items, item {i} was taken {times} \
                         times",
                    );
                }
            }
        }
        // And exactly one lane owns work that cannot be split.
        for lanes in 1..=17usize {
            let only = (0..lanes)
                .filter(|&index| {
                    Lane {
                        stage: 0,
                        index,
                        lanes,
                    }
                    .is_only()
                })
                .count();
            assert_eq!(only, 1, "with {lanes} lanes, {only} of them think they are the one");
        }
    }

    /// **Every lane finishes a stage before any lane starts the next.** This is the
    /// property the colouring depends on: a colour reads what the colour before it wrote,
    /// and without it the whole scheme is a data race dressed up as a schedule.
    ///
    /// Written as a test that would actually catch the failure rather than one that
    /// passes because the machine happened to be fast: every lane writes the stage number
    /// into its own slot, and every lane then checks every other slot. A lane that had
    /// run ahead would be seen by the others as being in the wrong stage.
    #[test]
    fn a_stage_is_finished_everywhere_before_the_next_begins() {
        let lanes = rayon::current_num_threads().max(2);
        let at: Vec<AtomicU64> = (0..lanes).map(|_| AtomicU64::new(u64::MAX)).collect();
        let wrong = AtomicUsize::new(0);
        const STAGES: usize = 200;

        each_stage(STAGES, usize::MAX, |lane| {
            at[lane.index].store(lane.stage as u64, Ordering::Release);
            // Everything the other lanes wrote in the stage before this one is visible,
            // and nothing they write in the stage after it can be.
            for slot in at.iter() {
                let seen = slot.load(Ordering::Acquire);
                if seen != u64::MAX && seen.abs_diff(lane.stage as u64) > 1 {
                    wrong.fetch_add(1, Ordering::Relaxed);
                }
            }
        });

        assert_eq!(
            wrong.load(Ordering::Relaxed),
            0,
            "a lane saw another more than one stage away, so the barrier is not holding \
             the stages apart",
        );
        for slot in at.iter() {
            assert_eq!(
                slot.load(Ordering::Relaxed),
                (STAGES - 1) as u64,
                "a lane did not run every stage",
            );
        }
    }

    /// The narrow path runs every stage too, on the calling thread. A small skeleton must
    /// get the same answer as a large one, not a different code path that skips work.
    #[test]
    fn the_narrow_path_runs_every_stage_once() {
        let seen = AtomicUsize::new(0);
        each_stage(7, 0, |lane| {
            assert_eq!(lane.lanes, 1, "the narrow path should be one lane");
            assert!(lane.is_only());
            seen.fetch_add(1 << lane.stage, Ordering::Relaxed);
        });
        assert_eq!(seen.load(Ordering::Relaxed), (1 << 7) - 1);
    }

    /// No stages is not a deadlock. A skeleton with no joints and nothing touching still
    /// takes steps.
    #[test]
    fn no_stages_is_not_a_deadlock() {
        each_stage(0, usize::MAX, |_| unreachable!("there were no stages to run"));
    }
}
