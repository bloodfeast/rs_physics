//! Which bodies the solver may leave out of a step, and what brings them back.
//!
//! # A settled heap costs full price for ever, and that is the largest single waste
//!
//! Measured on ten thousand bodies, a step is about a hundred milliseconds and
//! eighty-seven of them are the solve: `iterations` passes over every joint and every
//! contact. None of that arithmetic changes anything for a body that has stopped moving,
//! and a heap that has arrived is the state a heap spends nearly all of its life in.
//! Skipping it is not an optimisation of the solve, it is the removal of the solve.
//!
//! # The awake set is a bitset, because the interesting case is a run of zeroes
//!
//! One bit per body. Walking it is a walk over words: a word that is zero means
//! sixty-four sleeping bodies skipped by one test, which is the behaviour a settled heap
//! wants and is not reachable from a per-body branch. Every sweep and every constraint
//! list is filtered through it.
//!
//! # Islands, because a body cannot be woken by something it does not touch
//!
//! Bodies are grouped into connected components over the joints and contacts between
//! them, by union-find with path halving and union by size. A component is the unit that
//! sleeps: a body in the middle of a resting stack is not still because it has come to
//! rest on its own, it is still because everything holding it is, and the moment one of
//! them moves it is no longer entitled to be asleep. So the whole component goes to sleep
//! together and the whole component wakes together, and waking is an OR of the island's
//! stored words into the awake set rather than a walk over a graph that is no longer
//! being built.
//!
//! **Pinned bodies do not join islands.** A body of infinite mass never moves and cannot
//! carry a disturbance, so letting it into a component would weld every pile resting on
//! the same anchor into one island that can only sleep or wake as a unit. This is the
//! same reason a static body does not merge islands in any other solver.
//!
//! # What counts as still
//!
//! Two decisions, and both are quantities rather than epsilons.
//!
//! * **How far.** A body is still while it stays within [`STILL_FRACTION`] of its own
//!   radius of where the window started -- position and the sweep of its far end taken
//!   together. A fraction of the body's own size rather than an absolute distance,
//!   because the same rule then has to serve a finger bone and a torso: half a millimetre
//!   is nothing to one and a visible slide to the other. It is measured against the start
//!   of the window rather than step to step on purpose: a resting body in a position
//!   solver does not read back as motionless, it reads back as jittering, because gravity
//!   moves it `g dt^2` every step before the contact puts it back. Step-to-step speed
//!   therefore never falls below `g dt` -- 0.16 m/s at sixty hertz -- and a criterion
//!   below that would never fire at all. Drift over a window does not have that floor.
//! * **How long.** The window is the time a body of that radius takes to fall its own
//!   radius under the gravity the caller is actually using, `sqrt(2 r / g)`. That is the
//!   shortest interval over which "it has not moved" carries information: anything
//!   released from rest is inside the threshold for less than that, so a shorter window
//!   would put a body to sleep at the top of a bounce. It scales with size the right way
//!   -- a larger body takes longer to be believed -- and with gravity the right way.
//!
//! What would make either wrong: a caller whose bodies are far larger than the motion
//! that matters on them (a vehicle whose doors must be seen to settle), or one running
//! with no gravity at all, where there is no free-fall time to compare against and the
//! window falls back to standard gravity. Both are visible as a pile that stops moving
//! slightly too early rather than as an error.



/// How far a body may drift over a settling window and still be called still, as a
/// fraction of its own radius. See the module header for why it is a fraction.
pub(super) const STILL_FRACTION: f64 = 0.02;

/// The gravity a settling window is derived from when the caller supplies none.
const STANDARD_GRAVITY: f64 = 9.80665;

/// No island. `u32` rather than `Option<u32>` so the array is four bytes a body.
pub(super) const NO_ISLAND: u32 = u32::MAX;

/// One bit per body.
///
/// A `Vec<bool>` would be a byte a body and would still be read one body at a time. The
/// point of the words is [`BitSet::for_each_set`]: a zero word is sixty-four bodies
/// dismissed by one test, which is what a settled heap is made of.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct BitSet {
    words: Vec<u64>,
    len: usize,
}

impl BitSet {
    /// Grows to `len` bits, with the new bits set to `value`.
    pub(super) fn resize(&mut self, len: usize, value: bool) {
        let words = len.div_ceil(64);
        if value {
            // The tail of the last word is padding and must stay zero, or `count` and
            // `any` report bodies that do not exist.
            for i in self.len..len {
                if i >> 6 < self.words.len() {
                    self.words[i >> 6] |= 1u64 << (i & 63);
                }
            }
            self.words.resize(words, u64::MAX);
            self.len = len;
            self.trim();
            return;
        }
        self.words.resize(words, 0);
        self.len = len;
    }

    /// Clears the bits past `len` in the final word.
    fn trim(&mut self) {
        let spare = self.words.len() * 64 - self.len;
        if spare > 0 && spare < 64 {
            if let Some(last) = self.words.last_mut() {
                *last &= u64::MAX >> spare;
            }
        }
    }

    #[inline]
    pub(super) fn set(&mut self, i: usize) {
        self.words[i >> 6] |= 1u64 << (i & 63);
    }

    #[inline]
    pub(super) fn unset(&mut self, i: usize) {
        self.words[i >> 6] &= !(1u64 << (i & 63));
    }

    #[inline]
    pub(super) fn get(&self, i: usize) -> bool {
        self.words[i >> 6] & (1u64 << (i & 63)) != 0
    }

    pub(super) fn clear(&mut self) {
        self.words.iter_mut().for_each(|w| *w = 0);
    }

    pub(super) fn fill(&mut self) {
        self.words.iter_mut().for_each(|w| *w = u64::MAX);
        self.trim();
    }

    pub(super) fn any(&self) -> bool {
        self.words.iter().any(|&w| w != 0)
    }

    pub(super) fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub(super) fn words(&self) -> &[u64] {
        &self.words
    }

    pub(super) fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }


    /// `self |= other`, and whether anything changed.
    pub(super) fn union(&mut self, other: &BitSet) -> bool {
        let mut changed = false;
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            let merged = *a | *b;
            changed |= merged != *a;
            *a = merged;
        }
        changed
    }

    /// `self = a & !b`.
    pub(super) fn difference(&mut self, a: &BitSet, b: &BitSet) {
        self.words.resize(a.words.len(), 0);
        self.len = a.len;
        for (into, (x, y)) in self
            .words
            .iter_mut()
            .zip(a.words.iter().zip(b.words.iter()))
        {
            *into = *x & !*y;
        }
    }

    /// Every set bit, in order, a word at a time.
    #[inline]
    pub(super) fn for_each_set(&self, mut f: impl FnMut(usize)) {
        for (index, &word) in self.words.iter().enumerate() {
            let mut word = word;
            while word != 0 {
                f((index << 6) | word.trailing_zeros() as usize);
                word &= word - 1;
            }
        }
    }
}

/// Connected components over whatever is unioned into it, by union-find with path halving
/// and union by size.
///
/// Near enough linear, which matters because it is rebuilt every step: the joints do not
/// change but the contacts do, and a component that has lost its last contact is a
/// component that may sleep separately from the one it was part of.
#[derive(Clone, Debug, Default)]
pub(super) struct Components {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl Components {
    /// Makes every body in `members` a component of its own, and touches no others.
    ///
    /// Only those, because only they are ever unioned or looked up: a component that
    /// could reach a body outside the set would have to have been joined to it, and the
    /// caller joins nothing that is not in it. On a heap where one body in a hundred has
    /// settled that is a hundredth of the writes a full reset does.
    pub(super) fn reset_members(&mut self, len: usize, members: &BitSet) {
        if self.parent.len() < len {
            self.parent.resize(len, 0);
            self.size.resize(len, 1);
        }
        let (parent, size) = (&mut self.parent, &mut self.size);
        members.for_each_set(|i| {
            parent[i] = i as u32;
            size[i] = 1;
        });
    }

    #[inline]
    pub(super) fn find(&mut self, mut i: u32) -> u32 {
        // Path halving: every second link on the way up is pointed at its grandparent.
        // One pass, no recursion, and the same amortised bound as full compression.
        while self.parent[i as usize] != i {
            let grandparent = self.parent[self.parent[i as usize] as usize];
            self.parent[i as usize] = grandparent;
            i = grandparent;
        }
        i
    }

    #[inline]
    pub(super) fn union(&mut self, a: usize, b: usize) {
        let (mut ra, mut rb) = (self.find(a as u32), self.find(b as u32));
        if ra == rb {
            return;
        }
        if self.size[ra as usize] < self.size[rb as usize] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb as usize] = ra;
        self.size[ra as usize] += self.size[rb as usize];
    }
}

/// The islands that are asleep, stored as the words they occupy in the awake set.
///
/// A sleeping island is only ever read to be woken, and waking it is `awake |= island`.
/// Storing the members as `(word, mask)` pairs rather than as a list of indices makes
/// that an OR per sixty-four bodies instead of a scatter per body, and costs nothing
/// extra because the bodies of one rig are consecutive indices: a seventeen-bone skeleton
/// is one pair, or two when it straddles a word boundary.
#[derive(Clone, Debug, Default)]
pub(super) struct Islands {
    words: Vec<(u32, u64)>,
    /// Where each island's words live, as a half-open range. An empty range is an island
    /// that has woken and whose slot is on `free`.
    span: Vec<(u32, u32)>,
    free: Vec<u32>,
    /// Words held by islands that have since woken, so the garbage can be compacted once
    /// it is worth more than the copy.
    dead: usize,
}

impl Islands {
    /// Records a sleeping island from an **ascending** list of body indices, and returns
    /// its id. Ascending because that is what turns the list into whole words in one pass
    /// with no scratch buffer, and the caller's counting sort produces it that way.
    pub(super) fn freeze_sorted(&mut self, members: &[u32]) -> u32 {
        let from = self.words.len() as u32;
        let mut current = u32::MAX;
        let mut mask = 0u64;
        for &i in members {
            let word = i >> 6;
            if word != current {
                if mask != 0 {
                    self.words.push((current, mask));
                }
                current = word;
                mask = 0;
            }
            mask |= 1u64 << (i & 63);
        }
        if mask != 0 {
            self.words.push((current, mask));
        }
        let span = (from, self.words.len() as u32);
        match self.free.pop() {
            Some(id) => {
                self.span[id as usize] = span;
                id
            }
            None => {
                self.span.push(span);
                self.span.len() as u32 - 1
            }
        }
    }

    /// ORs island `id` into `awake` and releases it, clearing each member's island and
    /// restarting its settling window. Returns whether anything was added.
    pub(super) fn thaw(
        &mut self,
        id: u32,
        awake: &mut [u64],
        island_of: &mut [u32],
        still_steps: &mut [u32],
    ) -> bool {
        let (from, upto) = self.span[id as usize];
        if from == upto {
            return false;
        }
        for &(index, word) in &self.words[from as usize..upto as usize] {
            awake[index as usize] |= word;
            let mut bits = word;
            while bits != 0 {
                let i = ((index as usize) << 6) | bits.trailing_zeros() as usize;
                bits &= bits - 1;
                island_of[i] = NO_ISLAND;
                still_steps[i] = 0;
            }
        }
        self.span[id as usize] = (0, 0);
        self.free.push(id);
        self.dead += (upto - from) as usize;
        true
    }

    pub(super) fn clear(&mut self) {
        self.words.clear();
        self.span.clear();
        self.free.clear();
        self.dead = 0;
    }

    /// Squeezes out the words of islands that have woken, once half the buffer is theirs.
    ///
    /// Amortised: the compaction copies what is live, and cannot run again until as much
    /// again has died, so the total copying is linear in the number of islands frozen.
    pub(super) fn compact_if_worthwhile(&mut self) {
        if self.dead * 2 < self.words.len() || self.words.is_empty() {
            return;
        }
        let mut write = 0usize;
        for span in self.span.iter_mut() {
            let (from, upto) = *span;
            if from == upto {
                continue;
            }
            let count = (upto - from) as usize;
            self.words.copy_within(from as usize..upto as usize, write);
            *span = (write as u32, (write + count) as u32);
            write += count;
        }
        self.words.truncate(write);
        self.dead = 0;
    }
}

/// How many steps a body of this `radius` must stay still before it may sleep: the time
/// it would take to fall its own radius, in steps of `dt`.
///
/// See the module header. Clamped below at one step, because a body large enough for the
/// window to be long is also one nobody will see twitch, and clamped above so that an
/// enormous body in weak gravity does not become un-sleepable.
#[inline]
pub(super) fn settling_steps(reach: f64, gravity: f64, dt: f64) -> u32 {
    let g = if gravity > 1e-6 { gravity } else { STANDARD_GRAVITY };
    let seconds = (2.0 * reach.max(1e-6) / g).sqrt();
    ((seconds / dt).ceil() as u32).clamp(1, 600)
}
