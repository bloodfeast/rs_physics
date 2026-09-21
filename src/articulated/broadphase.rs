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
//! # Allocation
//!
//! Bucketing is a counting sort into buffers the [`Grid`] owns and reuses, so rebuilding
//! it every step -- which it must be, since everything moves -- allocates nothing after
//! the first.

use super::*;

/// Multipliers for the classic spatial hash. Large primes, so that cells adjacent in any
/// axis land far apart in the table and a moving body does not sweep one bucket.
const HASH: (i64, i64, i64) = (73_856_093, 19_349_663, 83_492_791);

/// The smallest table, so that a handful of bodies does not hash into two buckets.
const MIN_BUCKETS: usize = 64;

/// A uniform grid over the shaped bodies, rebuilt each step.
#[derive(Clone, Debug, Default)]
pub(super) struct Grid {
    /// Which bodies have a shape at all. Everything below indexes into this, not into
    /// the skeleton.
    shaped: Vec<usize>,
    /// The cell each of those bodies sits in, kept so that a hash collision can be told
    /// from a real neighbour.
    keys: Vec<(i32, i32, i32)>,
    /// Body indices ordered by bucket, and where each bucket starts. The counting sort's
    /// two halves.
    members: Vec<usize>,
    starts: Vec<u32>,
    cursor: Vec<u32>,
    cell: f64,
    mask: u64,
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
        let mut reach: f64 = 0.0;
        for i in 0..position.len() {
            if radius[i] > 0.0 {
                self.shaped.push(i);
                reach = reach.max(radius[i] + half_length[i]);
            }
        }
        if self.shaped.is_empty() {
            return;
        }

        // Twice the largest reach, so two bodies that touch are never more than one cell
        // apart. See the module header.
        self.cell = (2.0 * reach).max(1e-6);

        let buckets = (2 * self.shaped.len()).next_power_of_two().max(MIN_BUCKETS);
        self.mask = buckets as u64 - 1;

        self.keys.clear();
        self.keys
            .extend(self.shaped.iter().map(|&i| cell_of(position[i], self.cell)));

        // Counting sort: how many land in each bucket, where each bucket therefore
        // begins, then a second pass that puts them there.
        self.starts.clear();
        self.starts.resize(buckets + 1, 0);
        for key in self.keys.iter() {
            self.starts[bucket_of(*key, self.mask)] += 1;
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
        self.members.resize(self.shaped.len(), 0);
        for (nth, key) in self.keys.iter().enumerate() {
            let bucket = bucket_of(*key, self.mask);
            let at = self.cursor[bucket] as usize;
            self.members[at] = nth;
            self.cursor[bucket] += 1;
        }
    }

    /// Appends every pair worth testing to `out`, skipping pairs the caller has said are
    /// not candidates.
    ///
    /// Each pair is produced once: a body only reports neighbours that come after it in
    /// the grid's own order, and any body it can touch is within the cells it looks at,
    /// so the other side of the pair finds it instead.
    pub(super) fn pairs(
        &self,
        position: &[(f64, f64, f64)],
        radius: &[f64],
        half_length: &[f64],
        inv_mass: &[f64],
        jointed: &[(usize, usize)],
        out: &mut Vec<(usize, usize)>,
    ) {
        for (nth, &(x, y, z)) in self.keys.iter().enumerate() {
            let a = self.shaped[nth];
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let neighbour = (x + dx, y + dy, z + dz);
                        let bucket = bucket_of(neighbour, self.mask);
                        let from = self.starts[bucket] as usize;
                        let upto = self.starts[bucket + 1] as usize;
                        for &other in &self.members[from..upto] {
                            // A bucket holds every cell that hashed to it, so the cell
                            // itself has to be checked; and ordering by the grid's index
                            // is what keeps each pair to one appearance.
                            if other <= nth || self.keys[other] != neighbour {
                                continue;
                            }
                            let b = self.shaped[other];
                            // Two pinned bodies can never be moved apart, so a test
                            // between them has no outcome to produce.
                            if inv_mass[a] <= 0.0 && inv_mass[b] <= 0.0 {
                                continue;
                            }
                            // The rejection the grid cannot do, and it comes first
                            // deliberately: cells are sized for the largest body in the
                            // set, so most cell neighbours are nowhere near each other,
                            // and this is arithmetic on two values already in hand.
                            //
                            // The jointed test below is a binary search over every joint
                            // in the skeleton -- fourteen scattered reads for ten
                            // thousand of them. Measured with the two the other way
                            // round, a heap of six hundred bodies spent **thirty-five
                            // milliseconds a step** in the broad phase, almost all of it
                            // searching that list on behalf of pairs that were about to
                            // be thrown away for being metres apart.
                            let apart = length(sub(position[a], position[b]));
                            if apart > radius[a] + half_length[a] + radius[b] + half_length[b] {
                                continue;
                            }
                            let pair = (a.min(b), a.max(b));
                            if jointed.binary_search(&pair).is_ok() {
                                continue;
                            }
                            out.push(pair);
                        }
                    }
                }
            }
        }
    }
}

/// Which cell a point falls in.
#[inline]
fn cell_of(p: (f64, f64, f64), cell: f64) -> (i32, i32, i32) {
    (
        (p.0 / cell).floor() as i32,
        (p.1 / cell).floor() as i32,
        (p.2 / cell).floor() as i32,
    )
}

/// Teschner's spatial hash, folded into the table.
#[inline]
fn bucket_of(cell: (i32, i32, i32), mask: u64) -> usize {
    let hashed = (cell.0 as i64).wrapping_mul(HASH.0)
        ^ (cell.1 as i64).wrapping_mul(HASH.1)
        ^ (cell.2 as i64).wrapping_mul(HASH.2);
    ((hashed as u64) & mask) as usize
}
