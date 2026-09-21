//! Which pairs of bodies are close enough to be worth a narrow-phase test.
//!
//! # Why a grid and not a sweep
//!
//! Testing every pair is quadratic, and quadratic is not a constant factor away from
//! workable here: a few hundred bodies is fine and ten thousand is fifty million tests a
//! step, which is more arithmetic than the entire solve.
//!
//! Sorting along an axis and sweeping is less code and works well for things spread out
//! along that axis. It degenerates exactly where this is pointed, though -- a heap is a
//! tall stack of bodies sharing one footprint, so every body overlaps every other on the
//! two horizontal axes and most of them on the vertical one, and the sweep falls back to
//! the quadratic it was meant to avoid. A grid does not care how the bodies are arranged.
//!
//! # The cell size is not a tuning knob
//!
//! Each body goes in the single cell holding its centre, and each body looks at the
//! twenty-seven cells around its own. That is only correct if no two bodies can touch
//! from further apart than a cell, so the edge is **twice the largest reach in the set**,
//! a reach being a body's radius plus its half-length. It follows from the geometry and
//! there is nothing to tune: a smaller cell misses contacts and a larger one only costs
//! time.
//!
//! # One large body used to coarsen the grid for every small one
//!
//! That is what the rule costs, and it is not a small cost: the cell is a length, so a
//! body four times the reach of the rest of the set multiplies the volume every other
//! body scans by sixty-four. A skeleton's bones are within a few times each other's size
//! and it never bit there. Driving one heavy body through a field of small ones it bit
//! hard, and the `ploughing` benchmark's fixture had been cut down to a lane to get away
//! from it. Restored to a field -- 4,800 capsules of reach 0.35 with a roller of reach
//! 2.25 driven through them -- the bodies the scan looks at in one step:
//!
//! ```text
//!   the roller in the grid   cell 4.50 m   1,495,561
//!   the roller out of it     cell 0.70 m     105,362
//! ```
//!
//! Fourteen times, and it is a cube of a ratio rather than a constant factor, so it gets
//! worse with the size of the outlier and not better.
//!
//! So a body far enough above the rest of the set is **kept out of the grid**, and tested
//! against it directly: the cells its own bound overlaps, which is cheap because there are
//! few such bodies and the grid they are asking is now fine. Outliers are tested against
//! each other quadratically. The cell is then sized by the population rather than by the
//! one body, and the pair set is the same pair set --
//! `the_grid_finds_every_pair_the_quadratic_search_would` is the guard, on a fixture with
//! an outlier in it.
//!
//! **Far enough** is [`OVERSIZE`], which falls out of the neighbourhood rather than being
//! chosen. How many may come out is [`oversize_cap`], and that is what stops this firing
//! on a set that is not outlying but merely graded: where half the bodies are twice the
//! size of the other half there is no outlier, there are two populations, and taking one
//! of them out of the grid would be quadratic in half the set. Two grids would be the
//! answer to that and this is not that.
//!
//! Downstream an outlier is a body like any other. It is reported in `pairs`, so it joins
//! islands -- which are built over pairs and not over contacts, exactly so that two bodies
//! resting against each other land in one island. It is swept in the same rounds, so a
//! sleeping body beside it wakes. A sleeping *outlier* beside something awake wakes one
//! round later than a gridded body would, because nothing in the grid looks at a body that
//! is not in it, so an outlier has to do all of its own looking; the caller already sweeps
//! in rounds and that is the round it costs.
//!
//! # A cell is one integer, and the scan carries it
//!
//! The cell a body sits in is packed into a `u64`, twenty-one bits an axis, rather than
//! kept as three `i32`s. Two things follow, and both are in the inner loop:
//!
//! * The test that tells a real neighbour from a hash collision is one integer compare
//!   instead of three.
//! * The cell travels **inside the bucket entry**, so the scan reads it off the record it
//!   is already looking at. The earlier version stored bucket entries as indices and went
//!   back to a side table for each one's cell, which is a scattered load per candidate --
//!   and a body in a heap has three hundred candidates.
//!
//! Twenty-one bits an axis is a grid of two million cells on a side, which at the cell
//! size a ragdoll produces is a world four hundred kilometres across. Coordinates past
//! that are clamped rather than wrapped, so a body thrown to infinity lands in an edge
//! cell instead of aliasing onto one in the middle of the heap.
//!
//! # What was actually slow, measured
//!
//! On a heap of 9,600 shaped bodies the broad phase was 9.7 ms a step. Split:
//!
//! ```text
//!   the neighbour scan          5.3 ms
//!   the jointed-pair rejection  4.4 ms
//!   building the grid           0.1 ms
//! ```
//!
//! **Nearly half of it was one `binary_search`.** Two bones either side of an elbow
//! overlap by construction, so every jointed pair survives the distance test and asks the
//! question, and the answer was fourteen scattered probes into a list of every joint in
//! the skeleton. A body is jointed to three or four others and knows which, so the list
//! is now per body -- a run of `u32`s, looked up once for the outer body of the scan and
//! then in cache for the whole of its neighbourhood. The same rejection, 4.4 ms to 0.1.
//!
//! The scan itself is per body independent, so it runs in chunks across the pool, each
//! chunk filling a buffer the grid owns and the caller concatenating them in order.
//! Chunked by a fixed count rather than by the thread count, so the pairs come out in the
//! same order on every machine.
//!
//! # Allocation
//!
//! Bucketing is a counting sort into buffers the [`Grid`] owns and reuses, so rebuilding
//! it every step -- which it must be, since everything moves -- allocates nothing after
//! the first.

use super::*;

/// Fibonacci hashing: the reciprocal of the golden ratio in 64 bits. Multiplying by it
/// and taking the **high** bits spreads keys that differ by one -- which cells adjacent
/// along an axis do, by construction -- across the whole table.
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// Bits of the packed cell key given to each axis.
const LANE: u32 = 21;
/// Half the range of a lane, so a signed cell index can be stored unsigned.
const BIAS: i64 = 1 << (LANE - 1);
const LANE_MASK: u64 = (1 << LANE) - 1;

/// The smallest table, so that a handful of bodies does not hash into two buckets.
const MIN_BUCKETS: usize = 64;

/// Bodies per chunk of the parallel scan. Fixed rather than derived from the thread
/// count, so that the pair list is the same list in the same order whatever machine it
/// runs on.
const CHUNK: usize = 512;

/// How much larger than the rest of the set a body has to be before it is cheaper to keep
/// it out of the grid than to let it size the cell.
///
/// **This falls out of the neighbourhood; it is not a number that was tried.** A body of
/// reach `r` in a grid the same body sizes sits in a cell of edge `2r` and scans the
/// twenty-seven cells around its own, which is a box of edge `6r`. The same body tested
/// directly against a grid whose widest member reaches `w` scans the cells its own bound
/// overlaps -- a box of half-extent `r + w`, rounded out to cell boundaries, so an edge of
/// at most `2(r + w) + 2(2w)`, which is `2r + 6w`. Direct is the cheaper of the two for
/// that body alone when
///
/// ```text
///     2r + 6w  <  6r     which is     r  >  1.5 w
/// ```
///
/// -- the three being the cells on a side of the neighbourhood and the two being the
/// reaches in a cell. Both are the grid's own contract, stated at the top of this file, and
/// if either of them ever changes this moves with it.
///
/// It is a floor rather than a balance, and deliberately the conservative one: it is the
/// point at which coming out of the grid pays for the outlier *itself*, and by then it has
/// already stopped charging the other `n - 1` bodies the cube of how much it was inflating
/// their cell, which is the whole of what this is for. A body at exactly the break-even
/// saves the population a factor of `1.5^3`, a little over three.
///
/// **What would make it wrong** is a set whose large bodies are a population rather than
/// an outlier -- half the set twice the size of the other half. Then taking them out is
/// taking out half the grid and the quadratic below it is not a handful of pairs.
/// [`oversize_cap`] is what says no to that.
pub(super) const OVERSIZE: f64 = 1.5;

/// The most bodies that may be held out of a grid holding `gridded` of them.
///
/// Bodies out of the grid are tested against each other directly, and that is quadratic.
/// Quadratic is free only while it is smaller than a linear pass over the population --
/// which [`Grid::rebuild`] makes several of anyway, so it is a pass the step is already
/// paying for. So the bound is the largest `k` with `k(k - 1)/2 <= gridded`.
///
/// It is not there to be reached. A set with an outlier in it has one or two, and the
/// peel in [`Grid::grid_ceiling`] stops long before this. What it is there for is the set
/// that is merely graded, where peeling would walk on down the sizes taking half the
/// bodies out of the grid and testing them against each other; this is the statement that
/// such a set has no outlier in it and should be left alone.
pub(super) fn oversize_cap(gridded: usize) -> usize {
    // The positive root of `k^2 - k - 2n = 0`, floored. A budget rather than a bound that
    // has to be tight, so the square root's last bit does not matter.
    (0.5 * (1.0 + (1.0 + 8.0 * gridded as f64).sqrt())) as usize
}

/// One body's entry in a bucket: which cell it is really in, and which body it is.
///
/// The cell is here rather than in a side table because the scan needs it for every
/// candidate it rejects; see the module header.
///
/// # Carrying the centre and the reach here too, which does not pay
///
/// The obvious next step, and it was built: the scan rejects a candidate on its cell, its
/// centre and its reach, and the last two are scattered loads into arrays ten thousand
/// long, indexed by a body number with no relationship to where the scan is reading. A body
/// in a heap has around three hundred candidates. Putting all three in the record the scan
/// is already looking at is the same argument the header makes for `cell`, carried the rest
/// of the way.
///
/// **Measured, it is nothing**, and the way it is nothing is worth more than the change
/// would have been. Three alternating rounds of prebuilt binaries on `pile` at eight
/// iterations said 6.99 / 5.81 / 6.35 ms against 5.84 / 5.79 / 5.85 -- a clean eight per
/// cent with the "after" build strikingly steady. Four more rounds with **the order of the
/// two binaries reversed** said 6.47 / 6.50 / 6.56 / 6.24 against 5.69 / 6.32 / 5.71 /
/// 6.32. It flipped: whichever binary runs first in a round is the slow one, by about the
/// size of the effect being looked for.
///
/// So the record stays at sixteen bytes, and the method note is the finding. Alternating
/// *rounds* is not enough on this machine -- the order of the two builds within a round has
/// to alternate as well, or the first one measured carries a penalty that reads as a result.
/// Several tables in [`super`]'s header were taken by alternating rounds only.
#[derive(Clone, Copy, Debug, Default)]
struct Member {
    cell: u64,
    body: u32,
}

/// A uniform grid over the shaped bodies, rebuilt each step.
#[derive(Clone, Debug, Default)]
pub(super) struct Grid {
    /// Which bodies have a shape at all, in increasing order -- which is what lets the
    /// scan produce each pair once, by reporting only neighbours of a higher index.
    shaped: Vec<u32>,
    /// The packed cell of each of those, in the same order.
    keys: Vec<u64>,
    /// Each shaped body's radius plus half-length, indexed by **body**. One load in the
    /// inner loop where reading the two arrays was two.
    ///
    /// Filled for every body, gridded or not, because the oversized sweep reads it for
    /// bodies that are in no cell.
    reach: Vec<f64>,
    /// The bodies too big to be worth gridding, in increasing order -- see [`OVERSIZE`].
    /// They are **not** in `shaped`, `keys` or `members`, so the neighbourhood scan never
    /// sees one; [`Grid::sweep_oversized`] is where their pairs come from.
    ///
    /// In increasing body order because it is built by walking `shaped`, which is, and
    /// because two of this module's laws are that the pair list does not depend on the run
    /// or on the thread count.
    oversized: Vec<u32>,
    /// The largest reach left in the grid, which is half the cell. The oversized sweep
    /// needs it to know how far from itself a gridded body's centre can be and still be
    /// touching.
    widest: f64,
    /// Bucket entries in bucket order, and where each bucket starts. The counting sort's
    /// two halves.
    members: Vec<Member>,
    starts: Vec<u32>,
    cursor: Vec<u32>,
    /// Where each chunk of the parallel scan puts its pairs before they are concatenated,
    /// and the sleeping bodies it found next to a moving one. Owned so that a step
    /// allocates nothing.
    scratch: Vec<Vec<(usize, usize)>>,
    woken: Vec<Vec<usize>>,
    inv_cell: f64,
    /// `64 - log2(buckets)`, which is the shift Fibonacci hashing folds with.
    shift: u32,
}

impl Grid {
    /// Rebuilds the grid over every body with a radius.
    pub(super) fn rebuild(
        &mut self,
        position: &[(f64, f64, f64)],
        radius: &[f64],
        half_length: &[f64],
    ) {
        self.shaped.clear();
        self.oversized.clear();
        self.reach.clear();
        self.reach.resize(position.len(), 0.0);
        let mut widest: f64 = 0.0;
        let mut narrowest = f64::INFINITY;
        for i in 0..position.len() {
            let reach = radius[i] + half_length[i];
            self.reach[i] = reach;
            if radius[i] > 0.0 {
                self.shaped.push(i as u32);
                widest = widest.max(reach);
                narrowest = narrowest.min(reach);
            }
        }
        if self.shaped.is_empty() {
            self.widest = 0.0;
            return;
        }

        // Whatever is too big to be worth gridding comes out before the cell is fixed,
        // because the cell is exactly what it was taking from everything else.
        let ceiling = self.grid_ceiling(widest, narrowest);
        if ceiling < widest {
            let mut kept = 0;
            widest = 0.0;
            for nth in 0..self.shaped.len() {
                let body = self.shaped[nth];
                if self.reach[body as usize] > ceiling {
                    self.oversized.push(body);
                } else {
                    widest = widest.max(self.reach[body as usize]);
                    self.shaped[kept] = body;
                    kept += 1;
                }
            }
            self.shaped.truncate(kept);
        }
        self.widest = widest;

        // Twice the largest reach **in the grid**, so two gridded bodies that touch are
        // never more than one cell apart. See the module header.
        self.inv_cell = 1.0 / (2.0 * widest).max(1e-6);

        let buckets = (2 * self.shaped.len()).next_power_of_two().max(MIN_BUCKETS);
        self.shift = 64 - buckets.trailing_zeros();

        self.keys.clear();
        self.keys.extend(
            self.shaped
                .iter()
                .map(|&i| cell_of(position[i as usize], self.inv_cell)),
        );

        // Counting sort: how many land in each bucket, where each bucket therefore
        // begins, then a second pass that puts them there.
        self.starts.clear();
        self.starts.resize(buckets + 1, 0);
        for key in self.keys.iter() {
            self.starts[bucket_of(*key, self.shift)] += 1;
        }
        let mut running = 0u32;
        for slot in self.starts.iter_mut() {
            let count = *slot;
            *slot = running;
            running += count;
        }
        self.cursor.clear();
        self.cursor.extend_from_slice(&self.starts);

        self.members.clear();
        self.members.resize(self.shaped.len(), Member::default());
        for (nth, key) in self.keys.iter().enumerate() {
            let bucket = bucket_of(*key, self.shift);
            let at = self.cursor[bucket] as usize;
            self.members[at] = Member {
                cell: *key,
                body: self.shaped[nth],
            };
            self.cursor[bucket] += 1;
        }
    }

    /// The largest reach the grid will hold; everything above it is an outlier.
    ///
    /// Peels the top of the set while the body setting the cell is more than [`OVERSIZE`]
    /// times the widest body that would be left behind, so what it returns is a fixed
    /// point: nothing left in the grid is oversized for the grid that remains, and nothing
    /// taken out of it would have been better left in.
    ///
    /// Each round drops the ceiling by at least a factor of [`OVERSIZE`] and takes at least
    /// one more body out, so the rounds are bounded by [`oversize_cap`]; on every set that
    /// has an outlier in it at all there is one round, and on a set that has none the first
    /// compare answers it.
    fn grid_ceiling(&self, widest: f64, narrowest: f64) -> f64 {
        // Nothing in the set is far enough below the widest body for that body to be an
        // outlier in it. The usual answer, and it costs one compare on two numbers the
        // caller had already.
        if widest <= OVERSIZE * narrowest {
            return widest;
        }
        let cap = oversize_cap(self.shaped.len());
        let mut ceiling = widest;
        let mut top = widest;
        loop {
            // The widest body that is more than the break-even below the one setting the
            // cell: the cell this grid would have if everything above it came out.
            let mut next = f64::NEG_INFINITY;
            for &body in self.shaped.iter() {
                let reach = self.reach[body as usize];
                if reach * OVERSIZE < top {
                    next = next.max(reach);
                }
            }
            if !next.is_finite() {
                // Everything left is within the break-even of the body sizing the cell, so
                // the cell is the population's and there is no outlier under it.
                return ceiling;
            }
            let bar = OVERSIZE * next;
            let taken = self
                .shaped
                .iter()
                .filter(|&&body| self.reach[body as usize] > bar)
                .count();
            if taken > cap {
                // Not an outlier: a population. See [`oversize_cap`].
                return ceiling;
            }
            ceiling = bar;
            top = next;
        }
    }

    /// Which bodies were held out of the grid, in increasing order.
    ///
    /// For the laws, which have to be able to say that a fixture still has an outlier in
    /// it. The solve never asks: an outlier is a body like any other to everything past
    /// this file.
    #[cfg(test)]
    pub(super) fn oversized(&self) -> &[u32] {
        &self.oversized
    }

    /// Appends every pair worth testing to `out`, skipping pairs the caller has said are
    /// not candidates.
    ///
    /// **Only bodies in `frontier` are swept.** A body outside it is still a collider and
    /// is still reported as the other half of a pair; it simply does not look around
    /// itself, which is what lets a sleeping heap cost nothing here. It is also why the
    /// caller sweeps in rounds: a body found in neither `frontier` nor `swept` was asleep,
    /// so it is appended to `reached` and the caller sweeps from it next -- but **only when
    /// the body that found it is in `moving`**, which is what stops the rounds spreading
    /// out to the whole component. See [`super::Skeleton::find_pairs`].
    ///
    /// Each pair is still produced exactly once, and the rule that makes it so is now
    /// three cases rather than one:
    ///
    /// * both ends sweeping -- the lower index reports it, as before;
    /// * the other end already `swept` -- it reported the pair when it was the one
    ///   sweeping, so this side keeps quiet;
    /// * the other end in neither -- nobody has looked from it and nobody will until it is
    ///   woken, so this side reports it.
    ///
    /// With everything awake the second and third cases never arise and this is the sweep
    /// it replaced, which is what
    /// `the_grid_finds_every_pair_the_quadratic_search_would` checks by passing an
    /// all-set frontier.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn pairs(
        &mut self,
        position: &[(f64, f64, f64)],
        inv_mass: &[f64],
        jointed: Jointed<'_>,
        frontier: &BitSet,
        swept: &BitSet,
        moving: &BitSet,
        out: &mut Vec<(usize, usize)>,
        reached: &mut Vec<usize>,
    ) {
        let shaped = self.shaped.len();
        if shaped >= PARALLEL_FLOOR {
            // Taken out so the chunks can borrow the grid's read-only half while filling
            // it; put back below, so nothing here allocates after the first few steps.
            let mut scratch = std::mem::take(&mut self.scratch);
            let mut woken = std::mem::take(&mut self.woken);
            let chunks = shaped.div_ceil(CHUNK);
            if scratch.len() < chunks {
                scratch.resize_with(chunks, Vec::new);
            }
            if woken.len() < chunks {
                woken.resize_with(chunks, Vec::new);
            }
            scratch[..chunks]
                .par_iter_mut()
                .zip(woken[..chunks].par_iter_mut())
                .enumerate()
                .for_each(|(chunk, (into, wake))| {
                    into.clear();
                    wake.clear();
                    let from = chunk * CHUNK;
                    let upto = (from + CHUNK).min(shaped);
                    self.scan(
                        from, upto, position, inv_mass, jointed, frontier, swept, moving,
                        into, wake,
                    );
                });
            for filled in scratch[..chunks].iter() {
                out.extend_from_slice(filled);
            }
            for filled in woken[..chunks].iter() {
                reached.extend_from_slice(filled);
            }
            self.scratch = scratch;
            self.woken = woken;
        } else if shaped > 0 {
            self.scan(
                0, shaped, position, inv_mass, jointed, frontier, swept, moving, out,
                reached,
            );
        }
        if !self.oversized.is_empty() {
            self.sweep_oversized(
                position, inv_mass, jointed, frontier, swept, moving, out, reached,
            );
        }
    }

    /// The pairs of the bodies that were too big to be gridded: each against the grid, and
    /// then against the others that came out of it.
    ///
    /// **Nothing in the grid ever looks at a body that is not in it**, so an outlier does
    /// all of its own looking and reports every pair it finds, whatever the other end is
    /// doing. That is the one place this differs from [`Grid::scan`], where the three cases
    /// exist precisely because either end may be the one sweeping.
    ///
    /// The consequence is for sleeping. A gridded body beside an awake outlier is found and
    /// woken here exactly as the scan would have done. An outlier beside an awake *gridded*
    /// body is nobody's neighbour until it looks for itself, so when it is asleep this only
    /// wakes it -- it reports nothing -- and it sweeps properly in the next of the caller's
    /// rounds. Reporting from here as well would be the pair twice, once from each round.
    ///
    /// Outlier against outlier is quadratic and may be: [`oversize_cap`] bounds the count
    /// at the point where the quadratic is smaller than a pass over the population, and in
    /// a set that has an outlier at all the count is one or two.
    #[allow(clippy::too_many_arguments)]
    fn sweep_oversized(
        &self,
        position: &[(f64, f64, f64)],
        inv_mass: &[f64],
        jointed: Jointed<'_>,
        frontier: &BitSet,
        swept: &BitSet,
        moving: &BitSet,
        out: &mut Vec<(usize, usize)>,
        reached: &mut Vec<usize>,
    ) {
        for &packed in self.oversized.iter() {
            let a = packed as usize;
            if swept.get(a) {
                // It swept in an earlier round and reported then everything it can see;
                // nothing has moved since.
                continue;
            }
            let sweeping = frontier.get(a);
            // The same rule the neighbourhood scan wakes by, asked of an outlier: what
            // wakes a sleeping body is that something **moving** reached it, not that
            // something awake is beside it. See [`super::Skeleton::find_pairs`].
            let disturbing = moving.get(a);
            let mut wake_a = false;
            self.near(position[a], self.reach[a], |b| {
                if !self.worth_testing(a, b, position, inv_mass, jointed) {
                    return;
                }
                if sweeping {
                    out.push((a.min(b), a.max(b)));
                    if disturbing && !frontier.get(b) && !swept.get(b) {
                        // Nobody has looked from it and nobody will until it is woken.
                        reached.push(b);
                    }
                } else if moving.get(b) && (frontier.get(b) || swept.get(b)) {
                    // Asleep with something moving beside it. Waking it is all this round
                    // does; see the doc comment for why it does not also report.
                    wake_a = true;
                }
            });
            if wake_a {
                reached.push(a);
            }
        }

        // And against each other. The same three cases the neighbourhood scan uses, and
        // for the same reason: here both ends can be the one sweeping.
        for (nth, &packed) in self.oversized.iter().enumerate() {
            let a = packed as usize;
            if !frontier.get(a) {
                continue;
            }
            let disturbing = moving.get(a);
            for (mth, &other) in self.oversized.iter().enumerate() {
                if mth == nth {
                    continue;
                }
                let b = other as usize;
                let looking = frontier.get(b);
                if looking && b < a {
                    continue;
                }
                if !looking && swept.get(b) {
                    continue;
                }
                if !self.worth_testing(a, b, position, inv_mass, jointed) {
                    continue;
                }
                out.push((a.min(b), a.max(b)));
                if disturbing && !looking {
                    reached.push(b);
                }
            }
        }
    }

    /// The rejections [`Grid::scan`] makes once a pair is in hand, for a sweep that has no
    /// neighbourhood to amortise them over.
    ///
    /// The scan hoists every one of these out of its inner loop because it asks them of a
    /// three-hundred-body neighbourhood; this asks them of a handful of bodies a step, so
    /// the hoisting would be the more expensive half.
    #[inline]
    fn worth_testing(
        &self,
        a: usize,
        b: usize,
        position: &[(f64, f64, f64)],
        inv_mass: &[f64],
        jointed: Jointed<'_>,
    ) -> bool {
        if a == b {
            return false;
        }
        // Two pinned bodies can never be moved apart, so a test between them has no
        // outcome to produce.
        if inv_mass[a] <= 0.0 && inv_mass[b] <= 0.0 {
            return false;
        }
        let apart = sub(position[a], position[b]);
        let allowed = self.reach[a] + self.reach[b];
        if dot(apart, apart) > allowed * allowed {
            return false;
        }
        if jointed.of(a).contains(&(b as u32)) {
            return false;
        }
        // `u32::MAX` where no skeleton owns the body, which cannot match another body's.
        let skeleton_a = jointed.component.get(a).copied().unwrap_or(u32::MAX);
        jointed.component.get(b) != Some(&skeleton_a)
    }

    /// Every gridded body sitting in a cell that could hold something touching a body of
    /// this `reach` at `p`.
    ///
    /// The cells overlapping a box of half-extent `reach + widest` about the point, which
    /// is where anything it can touch has to be: no body in the grid reaches further than
    /// [`Grid::widest`], so one whose centre is beyond that is not touching whatever is at
    /// `p`. Cells rather than a sphere because a cell is the resolution the grid has.
    ///
    /// **It walks the members instead when the box covers more cells than the grid holds
    /// bodies.** A body a thousand times the size of the set would otherwise walk a
    /// thousand cubed mostly empty cells to reach the same few hundred bodies. The two
    /// counts are in the same units and the crossover is where they cross, so there is
    /// nothing here to choose either.
    fn near(&self, p: (f64, f64, f64), reach: f64, mut visit: impl FnMut(usize)) {
        let span = reach + self.widest;
        // `as` saturates and the clamp is the grid's own, so a body at an absurd
        // coordinate asks about an edge cell rather than one in the middle of the heap.
        let edge = |v: f64| ((v * self.inv_cell).floor() as i64).clamp(-BIAS, BIAS - 1);
        let lo = (edge(p.0 - span), edge(p.1 - span), edge(p.2 - span));
        let hi = (edge(p.0 + span), edge(p.1 + span), edge(p.2 + span));
        // In `i128` because each side of the box can be the whole two million cells of a
        // lane and the product of three of those is not a `u64`.
        let cells = (hi.0 - lo.0 + 1) as i128
            * (hi.1 - lo.1 + 1) as i128
            * (hi.2 - lo.2 + 1) as i128;
        if cells > self.members.len() as i128 {
            for member in self.members.iter() {
                visit(member.body as usize);
            }
            return;
        }
        for x in lo.0..=hi.0 {
            for y in lo.1..=hi.1 {
                for z in lo.2..=hi.2 {
                    let cell = pack(x, y, z);
                    let bucket = bucket_of(cell, self.shift);
                    let start = self.starts[bucket] as usize;
                    let end = self.starts[bucket + 1] as usize;
                    for member in &self.members[start..end] {
                        // A bucket holds every cell that hashed to it, so the cell itself
                        // still has to be checked.
                        if member.cell == cell {
                            visit(member.body as usize);
                        }
                    }
                }
            }
        }
    }

    /// The neighbourhood scan for one run of the shaped list.
    #[allow(clippy::too_many_arguments)]
    fn scan(
        &self,
        from: usize,
        upto: usize,
        position: &[(f64, f64, f64)],
        inv_mass: &[f64],
        jointed: Jointed<'_>,
        frontier: &BitSet,
        swept: &BitSet,
        moving: &BitSet,
        out: &mut Vec<(usize, usize)>,
        reached: &mut Vec<usize>,
    ) {
        for nth in from..upto {
            let a = self.shaped[nth] as usize;
            if !frontier.get(a) {
                continue;
            }
            // Whether anything this body finds asleep is woken by finding it. Read once
            // for the whole neighbourhood, like everything else about the outer body.
            let disturbing = moving.get(a);
            // Everything about the outer body, read once for its whole neighbourhood
            // rather than for each of the three hundred candidates in it.
            let here = position[a];
            let reach_a = self.reach[a];
            let pinned_a = inv_mass[a] <= 0.0;
            let jointed_to = jointed.of(a);
            // `u32::MAX` where no skeleton owns this body, which never matches another
            // body's, so a set with self-collision off still tests loose bodies normally.
            let skeleton_a = jointed.component.get(a).copied().unwrap_or(u32::MAX);
            let (x, y, z) = unpack(self.keys[nth]);

            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let cell = pack(x + dx, y + dy, z + dz);
                        let bucket = bucket_of(cell, self.shift);
                        let start = self.starts[bucket] as usize;
                        let end = self.starts[bucket + 1] as usize;
                        for member in &self.members[start..end] {
                            // A bucket holds every cell that hashed to it, so the cell
                            // itself has to be checked; and ordering by body index is
                            // what keeps each pair to one appearance.
                            let b = member.body as usize;
                            if member.cell != cell || b == a {
                                continue;
                            }
                            // Which of the two reports the pair. See the doc comment.
                            let looking = frontier.get(b);
                            if looking && b < a {
                                continue;
                            }
                            if !looking && swept.get(b) {
                                continue;
                            }
                            // Two pinned bodies can never be moved apart, so a test
                            // between them has no outcome to produce.
                            if pinned_a && inv_mass[b] <= 0.0 {
                                continue;
                            }
                            // The rejection the grid cannot do, and it comes first
                            // deliberately: cells are sized for the largest body in the
                            // set, so most cell neighbours are nowhere near each other,
                            // and this is arithmetic on two values already in hand.
                            let apart = sub(here, position[b]);
                            let allowed = reach_a + self.reach[b];
                            if dot(apart, apart) > allowed * allowed {
                                continue;
                            }
                            if jointed_to.contains(&(b as u32)) {
                                continue;
                            }
                            // The same rejection over a whole skeleton, when the caller
                            // has asked for it. `skeleton_a` is `u32::MAX` when the slice
                            // is empty, and a body's own index when it carries no joint,
                            // so neither case can match.
                            if jointed.component.get(b) == Some(&skeleton_a) {
                                continue;
                            }
                            out.push((a.min(b), a.max(b)));
                            // Asleep, and something that is moving is now a body's length
                            // away from it.
                            if disturbing && !looking {
                                reached.push(b);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Which bodies each body is directly jointed to, as one run per body.
///
/// The pairs contact generation must not produce. Two bones either side of an elbow share
/// an anchor point, so their capsules overlap by construction and a contact between them
/// would be the joint and the contact fighting each other forever.
///
/// [`Jointed::component`] is the same rejection widened to a whole skeleton, and it is
/// empty unless the caller has asked for it. See [`super::Skeleton::set_self_collision`]
/// for what it costs and what it buys.
///
/// A run per body rather than one sorted list of pairs, because the question is asked
/// from inside the neighbourhood scan and the answer is the same for a whole
/// neighbourhood: the run is found once per outer body and then read out of cache. The
/// sorted list cost fourteen scattered probes every time it was asked; see the module
/// header for what that was worth.
#[derive(Clone, Copy)]
pub(super) struct Jointed<'a> {
    pub start: &'a [u32],
    pub to: &'a [u32],
    /// Which skeleton each body belongs to, as the root of its joint-connected component,
    /// or **empty** when a skeleton is allowed to touch itself. Empty is the default and
    /// costs one test on an already-loaded slice.
    pub component: &'a [u32],
}

impl Jointed<'_> {
    /// The bodies jointed to `a`. Empty for a body with no joints, and for any body at
    /// all when the skeleton has none.
    #[inline]
    pub(super) fn of(&self, a: usize) -> &[u32] {
        if a + 1 >= self.start.len() {
            return &[];
        }
        &self.to[self.start[a] as usize..self.start[a + 1] as usize]
    }

    /// Whether `a` and `b` are held together by a joint.
    pub(super) fn holds(&self, a: usize, b: usize) -> bool {
        self.of(a).contains(&(b as u32))
    }
}

/// Which cell a point falls in, packed.
#[inline]
fn cell_of(p: (f64, f64, f64), inv_cell: f64) -> u64 {
    pack(
        (p.0 * inv_cell).floor() as i64,
        (p.1 * inv_cell).floor() as i64,
        (p.2 * inv_cell).floor() as i64,
    )
}

/// Three signed cell indices in one word, twenty-one bits each. Clamped rather than
/// wrapped: a body at an absurd coordinate should land in an edge cell, not alias onto
/// one in the middle of the heap.
#[inline]
fn pack(x: i64, y: i64, z: i64) -> u64 {
    let lane = |v: i64| (v.clamp(-BIAS, BIAS - 1) + BIAS) as u64;
    (lane(x) << (2 * LANE)) | (lane(y) << LANE) | lane(z)
}

/// The inverse of [`pack`], for the one place that needs the neighbouring cells.
#[inline]
fn unpack(key: u64) -> (i64, i64, i64) {
    let lane = |v: u64| (v & LANE_MASK) as i64 - BIAS;
    (
        lane(key >> (2 * LANE)),
        lane(key >> LANE),
        lane(key),
    )
}

/// Fibonacci hashing, folded into the table. One multiply and one shift, where the
/// classic three-prime spatial hash is three multiplies and two exclusive-ors.
#[inline]
fn bucket_of(cell: u64, shift: u32) -> usize {
    (cell.wrapping_mul(GOLDEN) >> shift) as usize
}
