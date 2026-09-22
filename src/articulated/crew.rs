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
//! # Spinning, and why it has a floor under it
//!
//! The gate spins before it blocks, which is only acceptable because of *when* it
//! happens: inside a `step`, between two colours that are microseconds apart, on threads
//! that are already doing the caller's work. Nothing here spins between steps -- the
//! broadcast returns, and rayon's own threads park as they always did. A caller who steps
//! one small skeleton a frame never reaches this code at all; see [`PASS_FLOOR`].
//!
//! The spin budget is not a tuned number. A lane that spins for as long as blocking would
//! have cost has, at worst, spent what it was about to spend anyway, and at best has
//! skipped it entirely -- so the budget is the cost of a park and a wake, and being wrong
//! about it by a factor of two costs a few microseconds a pass either way.
//!
//! **A spin barrier is correct only while every lane is actually executing, and that is a
//! statement about the lane count rather than about the spinning.** A lane is handed to
//! each thread the pool has, and a hardware thread is not a core: this machine reports
//! thirty-six of the first and eighteen of the second, so at least half the lanes are
//! descheduled at any instant *by construction*, and every one of the eighty to two
//! hundred and fifty barriers in a step waits on a context switch rather than on a cache
//! line. Measured on `pile` at eight iterations, one binary, three clean rounds of each:
//!
//! ```text
//!   36 lanes   6.39  6.11  6.04 ms   medians spread 5.8 per cent
//!   18 lanes   5.26  5.21  5.25 ms   medians spread 1.0 per cent
//! ```
//!
//! **Read that as a fact about `pile`, not about the crate.** Two independent
//! re-measurements have since disagreed with each other and with it, on three different
//! fixtures, and the disagreement is larger than any of the effects:
//!
//! ```text
//!   this table, pile, one binary per count, 3 rounds     18 beats 36 by 16%
//!   a heap of rigs, counts alternated in one process     36 beats 18 by 25-33%
//!   a dense overlapping heap, ABBA in one process        18 and 27 beat 36 by 5-25%
//! ```
//!
//! So **the lane count that wins is a property of the scene**, and nothing here is
//! entitled to state one answer. What the three agree on is narrower and is what the
//! design actually rests on: oversubscribing a spin barrier is not free, the cost is real
//! enough to measure, and a caller who knows its workload should be able to say so --
//! which is what [`super::Skeleton::set_lanes`] is for.
//!
//! Two methodological notes, because both re-measurements were provoked by this table and
//! neither settles it:
//!
//! * **The numbers above were taken across separate runs**, and this machine ramps
//!   thermally -- an unchanged binary on an unchanged scene has read 10.01 ms early in a
//!   session and 24.87 ms an hour later. A 16 per cent difference between two runs is
//!   inside that. Alternating the variants *within one process* is the only form of this
//!   comparison worth making, and the two rows below the table are.
//! * **The third row's fixture packs six hundred rigs into about ten metres square**, so
//!   the bodies interpenetrate and the contact count is nothing like `pile`'s -- its step
//!   is 127 ms where `pile`'s is six. It is evidence that the answer moves with the scene
//!   and is not evidence about `pile`.
//!
//! What would settle it is one fixture, phase-timed rather than wall-timed, with the lane
//! count alternated inside the process -- which is a measurement nobody has made yet.
//!
//! Sixteen per cent, and the spread is the more expensive half: several sections of
//! [`super`]'s header apologise for a variance they put down to the machine -- one fixture
//! "moved by a factor of two on the unchanged binary between rounds" -- and then read
//! their own small differences through it.
//!
//! **Blocking instead of spinning was built and it is worse, which is what says the lane
//! count is the fix.** The reasoning was that a lane which parks hands its core to whoever
//! is runnable, so a descheduled lane should cost one wake-up rather than every other
//! lane's quantum. What actually happens when the pool is oversubscribed is that *every*
//! barrier has lanes which outlast any spin budget, so the wake-up is not an exception, it
//! is the common case, and a couple of hundred barriers a step turn into thousands of
//! kernel round trips:
//!
//! ```text
//!   spin, 36 lanes   6.39  6.11  6.04 ms        park, 36 lanes   10.74  9.92  9.68 ms
//!   spin, 18 lanes   5.26  5.21  5.25 ms        park, 18 lanes    5.32  5.32  5.25 ms
//! ```
//!
//! Parking costs nothing when the barrier's precondition holds and sixty per cent when it
//! does not, so it does not buy the precondition -- it prices it. The lane count is what
//! has to give, and since this crate is a subsystem rather than an application, the lane
//! count is not the crate's to choose: see [`lanes_for`].

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
/// cost of a yield rather than a number somebody liked.
const SPINS_PER_YIELD: u32 = 64;

/// How many lanes a pass of `work` constraints should use, given the budget the caller
/// has declared.
///
/// **All of the budget, or one.** The only judgement left here is whether to go parallel
/// at all, which is [`PASS_FLOOR`]. An earlier version derived an interior lane count by
/// balancing the barrier against the arithmetic and chose thirty lanes for the largest
/// case, which was forty per cent slower than using every one; the model was wrong because
/// a barrier's cost is mostly the wait for the slowest lane to notice, which does not grow
/// with the lane count the way contention on its counter does.
///
/// **What the budget is, though, is not this crate's to decide**, and reading it off
/// `rayon::current_num_threads()` was this module assuming it owned the machine. It does
/// not: a solver is a subsystem of something with a frame to fill, and the threads it may
/// have are whatever is left after rendering, audio and everything else. That is what
/// [`super::Skeleton::set_lanes`] is for, and it defaults to the whole pool because within
/// a pool that is the fastest answer -- see there for the measurement, and for the thing it
/// is *not* a remedy for, which is a pool with more threads than the machine can run.
fn lanes_for(work: usize, budget: usize) -> usize {
    if work >= PASS_FLOOR {
        budget.max(1)
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
    /// **Set when a lane has left the stage loop early**, because its own work panicked.
    ///
    /// A lane that unwinds never arrives, so the count the gate waits for can never be
    /// reached and every other lane waits for ever -- and because `rayon::broadcast` only
    /// delivers a panic once every job has returned, the whole pool dies with it, including
    /// callers with nothing to do with this crate. Measured before this existed: a single
    /// panicking lane hung the process, with thirty-five threads spinning.
    ///
    /// The answer is to abandon the pass rather than to finish it. Once this is set, every
    /// gate is open and every lane stops at the top of its next stage -- nothing else is
    /// owed, because the broadcast is going to unwind regardless and the only thing the
    /// other lanes have to do is stop. They are not racing the lane that fell over: it is
    /// not writing anything any more.
    abandoned: AtomicBool,
}

impl Gate {
    fn new(lanes: usize) -> Self {
        Gate {
            arrived: AtomicUsize::new(0),
            sense: AtomicBool::new(false),
            lanes,
            abandoned: AtomicBool::new(false),
        }
    }

    /// Abandon the pass. Called from the unwinding path, where it must not itself be able
    /// to fail, which is why it is one store and no lock.
    fn abandon(&self) {
        self.abandoned.store(true, Ordering::Release);
    }

    /// Whether some lane fell over and the pass is being given up.
    fn is_abandoned(&self) -> bool {
        self.abandoned.load(Ordering::Acquire)
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
            // The lane this one is waiting for may have fallen over, in which case it is
            // never coming and there is nothing left to wait for.
            if self.is_abandoned() {
                return;
            }
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
/// Returns how many lanes it used, which is one when it ran on the calling thread. The
/// caller keeps that so it can be asked -- see [`super::Skeleton::solved_in_parallel`],
/// which exists because a law that cannot tell whether the parallel path ran is a law that
/// passes by running the serial one twice.
pub(super) fn each_stage(
    stages: usize,
    work_items: usize,
    budget: usize,
    work: impl Fn(Lane) + Sync,
) -> usize {
    if stages == 0 {
        return 0;
    }
    // Never more lanes than the pool has threads to put them on, whatever the caller asked
    // for: a lane the broadcast cannot deliver is a lane the gate would wait for for ever.
    let lanes = lanes_for(work_items, budget.min(rayon::current_num_threads()));
    if lanes <= 1 {
        for stage in 0..stages {
            work(Lane {
                stage,
                index: 0,
                lanes: 1,
            });
        }
        return 1;
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
        // **Every path out of the stage loop has to tell the gate.** Unwinding past it
        // without saying so is the one way to leave a barrier that can never open: the
        // count it waits for can no longer be reached, every other lane waits for ever,
        // and since `rayon::broadcast` only propagates a panic once every job has
        // returned, the whole pool dies with it -- including callers with nothing to do
        // with this crate. See [`Gate::depart`].
        let fell = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for stage in 0..stages {
                if gate.is_abandoned() {
                    return;
                }
                work(Lane {
                    stage,
                    index: ctx.index(),
                    lanes,
                });
                gate.wait(&mut sense);
            }
        }));
        if let Err(payload) = fell {
            gate.abandon();
            // Resumed rather than swallowed, so the caller gets the original panic with
            // its own message: the broadcast collects it once every lane has returned,
            // which is now something that happens.
            std::panic::resume_unwind(payload);
        }
    });
    lanes
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    /// **A lane that falls over does not take the pool with it.**
    ///
    /// The barrier waits for a count of arrivals, and a lane that unwinds never arrives, so
    /// before [`Gate::abandon`] existed the count could not be reached and every other lane
    /// spun for ever. `rayon::broadcast` only delivers a panic once every job has returned,
    /// so the panic never surfaced either: the process hung with thirty-five threads at a
    /// hundred per cent and no stack to look at, and the pool was dead for everything else
    /// in the program, not only for this crate.
    ///
    /// It is worth being clear about why this was not a theoretical hazard. The debug
    /// assertions inside [`super::scatter::Cells`] fire *inside a lane*. Every one of them
    /// could only ever hang rather than report, which is the opposite of what an assertion
    /// is for, and is why the one check that matters -- [`super::scatter::disjoint`] -- runs
    /// on the calling thread instead.
    ///
    /// Stated with a timeout rather than by calling `each_stage` directly, because the
    /// failure being guarded against is a hang, and a test that reproduces it by hanging
    /// cannot report anything either.
    #[test]
    fn a_lane_that_panics_does_not_hang_the_others() {
        let (done, listen) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            // The panic is expected; its message would otherwise be printed by the default
            // hook and read as a test failure.
            let hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let fell = std::panic::catch_unwind(|| {
                each_stage(8, usize::MAX, rayon::current_num_threads(), |lane: Lane| {
                    assert!(
                        !(lane.stage == 3 && lane.index == 1),
                        "the lane this test exists to knock over",
                    );
                });
            });
            std::panic::set_hook(hook);
            let _ = done.send(fell.is_err());
        });

        match listen.recv_timeout(std::time::Duration::from_secs(20)) {
            Ok(panicked) => assert!(
                panicked,
                "the pass swallowed a lane's panic; the caller would be handed a step that \
                 silently did not happen",
            ),
            Err(_) => panic!(
                "a panicking lane hung the pass for twenty seconds, so every other lane is \
                 still waiting at a barrier that can never open",
            ),
        }
    }

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

        each_stage(STAGES, usize::MAX, rayon::current_num_threads(), |lane| {
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
        each_stage(7, 0, rayon::current_num_threads(), |lane| {
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
        each_stage(0, usize::MAX, rayon::current_num_threads(), |_| {
            unreachable!("there were no stages to run")
        });
    }
}
