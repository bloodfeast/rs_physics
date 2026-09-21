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
//! together.
//!
//! **It does not wake together, and that is a separate question with a different answer.**
//! Sleeping is a claim about a set -- nobody in it is moving -- and waking is a claim about
//! one body, that something has happened to *it*. Reading the island as the unit of both
//! is the defect the section below is about; [`Islands::release`] is what waking does to an
//! island now, which is to take one body out of it and leave the rest of it asleep.
//!
//! **Pinned bodies do not join islands.** A body of infinite mass never moves and cannot
//! carry a disturbance, so letting it into a component would weld every pile resting on
//! the same anchor into one island that can only sleep or wake as a unit. This is the
//! same reason a static body does not merge islands in any other solver.
//!
//! **And not joining an island is not the same as holding one awake.** A body that is not
//! ready to sleep disqualifies its ready neighbours -- a body holding up something that is
//! still moving has no business stopping -- and there are three ways to be unready: to be
//! moving, to be pinned, or to be asleep already. The last two are the *stillest* things in
//! the simulation. Reading all three as "still moving" left every skeleton hung off an
//! anchor permanently awake: measured, a limb of three capsules hanging from a pinned root
//! at exactly its own equilibrium, with every velocity bitwise zero, was solved for twelve
//! thousand steps and never slept. What disqualifies a neighbour is that it is *awake*,
//! which is the one bit that means "being solved and not settled", and
//! `a_limb_hanging_from_an_anchor_goes_to_sleep` is the guard on it. See
//! [`super::Skeleton::settle`].
//!
//! # Four contacts used to wake four hundred bodies
//!
//! A field of four hundred loose capsules settled flat on the plane is **one island**: they
//! all touch, or come within a body's length of touching, so the union-find joins the lot.
//! Waking by island then means that the first thing to disturb any of them puts every one
//! of them back into the step. Measured on a field of two thousand and forty-eight with one
//! heavy body driven through it at a walking pace, timed halfway down the field:
//!
//! ```text
//!                       awake        contacts   1 pass          8 passes
//!   waking the island   2049 / 2049       250   1.70 .. 1.74 ms   2.88 .. 3.09 ms
//!   waking a body        346 / 2049       292   1.00 .. 1.06 ms   1.80 .. 2.43 ms
//! ```
//!
//! That is `benches/articulated.rs`'s `ploughing`, five alternating rounds of prebuilt
//! binaries. The field is a settled world with one thing happening in a small part of it,
//! which is the case sleeping exists for and was the one it handled worst: the disturbance
//! is local and the response was global. Note which way the contact count went: the field
//! that stays asleep is a *denser* scene by the time it is timed, and is a third cheaper
//! anyway, because what a settled body costs a step is not its contacts.
//!
//! # So what wakes a body is that something moving reached it
//!
//! Three rules, and between them they replace the island thaw. Each is stated where it
//! runs; this is why the three of them are needed and no fewer.
//!
//! * **Waking takes one body out of its island** and leaves the rest of it asleep --
//!   [`Islands::release`], and [`super::Skeleton::wake`]. On its own this changes nothing
//!   about which bodies end up awake, because the broad phase sweeps outward in rounds from
//!   the awake set and a settled field is connected: the thaw simply arrives a round at a
//!   time instead of all at once. Measured, the field above went from 401 of 401 awake at
//!   the step of first contact to 401 of 401 awake thirty steps later.
//! * **What may carry the front outward is the set that is moving**, not the set that is
//!   awake -- [`super::Skeleton::find_pairs`]. This is the one that bites. Nearly every
//!   body in a settled field is within a body's length of something awake, so waking on
//!   proximity to *anything awake* grows the awake region by a ring a step for ever; waking
//!   on proximity to something that has moved further than [`STILL_FRACTION`] of its own
//!   reach makes the region follow the disturbance and stop where it stops. It also makes
//!   the rounds terminate on their own: a body that has just been woken has not moved, so
//!   the round after the one that woke it produces pairs and no wakes.
//! * **And a touch wakes whatever it touched**, whether or not anything was moving fast
//!   enough to be called a disturbance -- [`super::Skeleton::wake_touched`]. That is the
//!   floor under the second rule, and it is what makes the threshold in it safe rather than
//!   a tolerance: below the speed at which the front leads, the overlap a body can present
//!   a sleeping neighbour with before the contact wakes it is bounded by the same fraction
//!   of its own reach.
//!
//! **Waking on the touch alone does not work, and the stack says why.** It is the tempting
//! rule -- causal, no distances in it, the same shape as the one that keeps a contact
//! alive -- and it drives a stack apart: a capsule dropped onto a settled stack of three
//! went through it at fifteen of the twenty-eight drop heights swept below, twelve of them
//! consecutive and including the one the guard uses. The reason is the fact this module has
//! already recorded twice: **a pile
//! that has settled perfectly carries no contacts at all**, because the solve removes the
//! whole overlap. What re-acquires them is the woken body's own sag of `g dt^2` onto what
//! it is resting on, and that takes a step per layer. Waking on the touch gives the pile no
//! steps; waking on proximity gives it the body's own length divided by the speed of
//! whatever is coming, which at a metre a second is a dozen.
//!
//! # What it costs everywhere else, which is the same scene doing the same work
//!
//! Every other fixture in `benches/articulated.rs` reaches **bit-identical named states**
//! under both rules -- the same bodies, joints, colours, contacts and awake counts -- so
//! there is no scene difference to read a timing as:
//!
//! ```text
//!                       contacts   awake
//!   one                       13   16 / 17
//!   joints_only                0   10200 / 10200
//!   pile                    7800   9600 / 10200
//!   arriving               12000   10200 / 10200
//!   has_settled, asleep        0   0 / 9999
//! ```
//!
//! What is added to a step is a bit written per awake body in [`super::Skeleton::settle`],
//! two bit tests per contact in [`super::Skeleton::wake_touched`], and one more bitset read
//! per swept body in the broad phase. The timings agree: over five alternating rounds of
//! prebuilt binaries, in both orders, every one of those fixtures has the two ranges
//! overlapping -- `one` 100..158 us against 106..161, `joints_only` 3.3..5.7 ms against
//! 4.0..5.6, `pile` 7.2..18.1 against 9.9..23.2, `arriving` 14.1..22.3 against 18.6..22.3.
//! Those spreads are the machine rather than the change: `pile` moved by a factor of two
//! and a half on the *unchanged* binary between rounds of the same session. Read them as
//! "no signal" and re-measure on a quiet machine if a number ever has to be quoted.
//!
//! # What the drop guard actually measures, which is less than it appears to
//!
//! `a_body_dropped_on_a_sleeping_stack_lands_on_top_of_it` drops a capsule from three
//! metres onto a settled stack of three and asserts it finishes above the stack's top. It
//! is the guard on all of the above and it is worth knowing what it can see. Swept over
//! twenty-eight drop heights from one metre to three and seven tenths, a tenth apart:
//!
//! ```text
//!   waking the island   12 of 28 go through
//!   waking a body        9 of 28 go through
//! ```
//!
//! **Both fail about a third of the heights, on the commit this was written against as much
//! as after it.** What happens is not a body passing through a pile because the pile was
//! slow to wake -- it is that a first contact deeper than a capsule's radius drives two
//! parallel capsules into each other by more than the narrow phase can undo, and they stay
//! there: the stack ends up with two of its bodies a tenth of a metre inside one another.
//! That is a defect in the solve at a deep first overlap and it is nothing to do with
//! sleeping; the guard is one sample of a chaotic family and its own height is one of the
//! ones that passes. Anybody changing the waking rules should sweep the family rather than
//! trust the single height, and anybody going after the interpenetration should know that
//! fixing it would make this guard mean what it says.
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
//! **It is not the criterion that keeps a jointed rig awake**, and that was worth ruling
//! out rather than assuming. A rig that will not sleep is not jittering below the
//! threshold: it translates as a rigid body at twenty to forty millimetres a second, its
//! median bone moving 0.039 m relative to the rig's centre while the centre moves 0.276 m,
//! and its net travel over thirty-two windows is 0.99 of the path it walked getting there.
//! Loosening anything here would put a rig to sleep while it was visibly walking. See the
//! header of [`super`] for what does keep it awake.
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

    /// Takes **one** body out of island `id`, leaving the rest of it asleep, and says
    /// whether that emptied the island.
    ///
    /// This is the whole of what waking writes to an island, and it is one body rather
    /// than the component for the reason in the module header: the island is the unit
    /// that *sleeps* and the body is the unit that *wakes*. The scan is over the
    /// island's words, which is one entry per sixty-four consecutive members -- a
    /// seventeen-bone rig is one or two, a four-hundred-body field is seven -- and it
    /// has to look at all of them anyway to tell whether anything is left.
    ///
    /// The caller clears `island_of` for the body. A word left holding no members is left
    /// in place rather than squeezed out: nothing reads an island except this, and
    /// [`Islands::compact_if_worthwhile`] collects the whole span once the island empties.
    pub(super) fn release(&mut self, id: u32, i: u32) -> bool {
        let (from, upto) = self.span[id as usize];
        if from == upto {
            return true;
        }
        let word = i >> 6;
        let mut empty = true;
        for slot in &mut self.words[from as usize..upto as usize] {
            if slot.0 == word {
                slot.1 &= !(1u64 << (i & 63));
            }
            empty &= slot.1 == 0;
        }
        if empty {
            self.span[id as usize] = (0, 0);
            self.free.push(id);
            self.dead += (upto - from) as usize;
        }
        empty
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
