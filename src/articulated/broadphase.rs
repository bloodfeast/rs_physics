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
//! The cost of that rule is that one large body coarsens the grid for every small one.
//! A skeleton's bones are within a few times each other's size, so it does not bite here;
//! a set holding one enormous collider and ten thousand small ones would want that
//! collider kept out of the grid and tested separately.
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

/// One body's entry in a bucket: which cell it is really in, and which body it is.
///
/// The cell is here rather than in a side table because the scan needs it for every
/// candidate it rejects; see the module header.
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
    reach: Vec<f64>,
    /// Bucket entries in bucket order, and where each bucket starts. The counting sort's
    /// two halves.
    members: Vec<Member>,
    starts: Vec<u32>,
    cursor: Vec<u32>,
    /// Where each chunk of the parallel scan puts its pairs before they are concatenated,
    /// and the sleeping bodies it found next to an awake one. Owned so that a step
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
        self.reach.clear();
        self.reach.resize(position.len(), 0.0);
        let mut widest: f64 = 0.0;
        for i in 0..position.len() {
            let reach = radius[i] + half_length[i];
            self.reach[i] = reach;
            if radius[i] > 0.0 {
                self.shaped.push(i as u32);
                widest = widest.max(reach);
            }
        }
        if self.shaped.is_empty() {
            return;
        }

        // Twice the largest reach, so two bodies that touch are never more than one cell
        // apart. See the module header.
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

    /// Appends every pair worth testing to `out`, skipping pairs the caller has said are
    /// not candidates.
    ///
    /// **Only bodies in `frontier` are swept.** A body outside it is still a collider and
    /// is still reported as the other half of a pair; it simply does not look around
    /// itself, which is what lets a sleeping heap cost nothing here. It is also why the
    /// caller sweeps in rounds: a body found in neither `frontier` nor `swept` was
    /// asleep, so it is appended to `reached` and the caller sweeps from it next.
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
        out: &mut Vec<(usize, usize)>,
        reached: &mut Vec<usize>,
    ) {
        let shaped = self.shaped.len();
        if shaped == 0 {
            return;
        }
        if shaped < PARALLEL_FLOOR {
            self.scan(
                0, shaped, position, inv_mass, jointed, frontier, swept, out, reached,
            );
            return;
        }

        // Taken out so the chunks can borrow the grid's read-only half while filling it;
        // put back below, so nothing here allocates after the first few steps.
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
                    from, upto, position, inv_mass, jointed, frontier, swept, into, wake,
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
        out: &mut Vec<(usize, usize)>,
        reached: &mut Vec<usize>,
    ) {
        for nth in from..upto {
            let a = self.shaped[nth] as usize;
            if !frontier.get(a) {
                continue;
            }
            // Everything about the outer body, read once for its whole neighbourhood
            // rather than for each of the three hundred candidates in it.
            let here = position[a];
            let reach_a = self.reach[a];
            let pinned_a = inv_mass[a] <= 0.0;
            let jointed_to = jointed.of(a);
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
                            out.push((a.min(b), a.max(b)));
                            // Close enough to be worth a narrow-phase test, and nobody
                            // has looked from it: it was asleep, and something is beside
                            // it now.
                            if !looking {
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
/// A run per body rather than one sorted list of pairs, because the question is asked
/// from inside the neighbourhood scan and the answer is the same for a whole
/// neighbourhood: the run is found once per outer body and then read out of cache. The
/// sorted list cost fourteen scattered probes every time it was asked; see the module
/// header for what that was worth.
#[derive(Clone, Copy)]
pub(super) struct Jointed<'a> {
    pub start: &'a [u32],
    pub to: &'a [u32],
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
