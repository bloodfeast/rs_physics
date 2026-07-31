//! PhysicsWorld - The core simulation container

use std::collections::HashMap;
use rayon::prelude::*;
use log::info;
use crate::models::{PhysicalObject3D, Quaternion, Shape3D};
use crate::utils::PhysicsConstants;
use crate::interactions::shape_collisions_3d::apply_gravity;
use crate::interactions::gjk_collision_3d::{gjk_collision_detection_ex, epa_contact_points_ex, GjkResult};
use super::state::{ObjectId, ObjectState, WorldState};
use super::config::WorldConfig;

#[cfg(feature = "constraints")]
use super::world_constraints::{ConstraintId, WorldConstraint};

/// Unique identifier for continuous forces
///
/// `Ord` matters here: forces are stored in a `HashMap` but must be *applied*
/// in a stable order, or float accumulation varies between runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ForceId(pub u64);

impl ForceId {
    /// Generate a new unique ForceId
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        ForceId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for ForceId {
    fn default() -> Self {
        Self::new()
    }
}

/// Continuous forces that persist across simulation steps
///
/// These forces are automatically recalculated each step based on current object state.
/// They remain active until explicitly removed or their effect becomes negligible.
#[derive(Debug, Clone)]
pub enum ContinuousForce {
    /// Constant force in a direction (like thrust or wind)
    /// Force is applied every step until removed
    Constant {
        target: ObjectId,
        force: (f64, f64, f64),
    },

    /// Drag force opposing velocity (air/water resistance)
    /// Automatically removed when object velocity drops below threshold
    Drag {
        target: ObjectId,
        coefficient: f64,
        /// Velocity threshold below which drag is removed (default: 0.01)
        min_velocity: f64,
    },

    /// Spring force toward a rest position
    /// Automatically removed when displacement and velocity are below thresholds
    Spring {
        target: ObjectId,
        rest_position: (f64, f64, f64),
        stiffness: f64,
        /// Displacement threshold for removal (default: 0.01)
        min_displacement: f64,
        /// Velocity threshold for removal (default: 0.01)
        min_velocity: f64,
    },

    /// Damped spring (spring + velocity damping)
    /// Automatically removed when at rest
    DampedSpring {
        target: ObjectId,
        rest_position: (f64, f64, f64),
        stiffness: f64,
        damping: f64,
        min_displacement: f64,
        min_velocity: f64,
    },

    /// Attraction toward a point (gravity well, magnet)
    /// Can be set to expire after duration or persist indefinitely
    Attract {
        target: ObjectId,
        point: (f64, f64, f64),
        strength: f64,
        /// If Some, force expires after this many seconds
        duration: Option<f64>,
        elapsed: f64,
    },

    /// Repulsion from a point (force field)
    Repel {
        target: ObjectId,
        point: (f64, f64, f64),
        strength: f64,
        duration: Option<f64>,
        elapsed: f64,
    },

    /// Buoyancy force (upward force when below surface)
    /// Removed when object is above surface for extended time
    Buoyancy {
        target: ObjectId,
        surface_y: f64,
        fluid_density: f64,
        /// Time object has been above surface
        time_above_surface: f64,
        /// Remove after this long above surface (default: 1.0s)
        removal_delay: f64,
    },

    /// Vortex/rotational force around an axis
    Vortex {
        target: ObjectId,
        center: (f64, f64, f64),
        axis: (f64, f64, f64),
        strength: f64,
        duration: Option<f64>,
        elapsed: f64,
    },
}

/// Collision data collected during parallel detection phase
/// This allows us to detect collisions in parallel (read-only)
/// and then apply responses sequentially (write)
#[derive(Debug, Clone, PartialEq)]
struct CollisionData {
    /// Index of first object
    i: usize,
    /// Index of second object
    j: usize,
    /// Contact normal (from obj1 to obj2)
    normal: (f64, f64, f64),
    /// Penetration depth
    penetration: f64,
    /// Contact point on object 1 (local offset from center)
    contact1: (f64, f64, f64),
    /// Contact point on object 2 (local offset from center)
    contact2: (f64, f64, f64),
}

/// Grid coordinate of a broad-phase cell.
type CellKey = (i32, i32, i32);

/// Per-phase time accumulator, so optimization targets come from measurement
/// rather than intuition. Test builds only; `phase!` compiles to nothing else.
#[cfg(test)]
#[derive(Default, Clone, Copy, Debug)]
pub(crate) struct PhaseTimings {
    pub gravity: std::time::Duration,
    pub continuous_forces: std::time::Duration,
    pub pending_forces: std::time::Duration,
    pub broad_rebuild: std::time::Duration,
    pub narrow_phase: std::time::Duration,
    pub response: std::time::Duration,
    pub constraints: std::time::Duration,
    pub damping: std::time::Duration,
    pub integrate: std::time::Duration,
    pub tunneling: std::time::Duration,
}

/// Sub-phase timings inside the broad-phase rebuild.
#[cfg(test)]
#[derive(Default, Clone, Copy, Debug)]
pub(crate) struct BroadPhaseTimings {
    pub fill: std::time::Duration,
    pub median: std::time::Duration,
    pub scatter: std::time::Duration,
    pub collect: std::time::Duration,
    pub sort_dedup: std::time::Duration,
}

/// Time a phase into `self.profile`, or expand to just the body outside tests.
#[cfg(test)]
macro_rules! phase {
    ($self:ident, $field:ident, $body:expr) => {{
        let __start = std::time::Instant::now();
        let __result = $body;
        $self.profile.$field += __start.elapsed();
        __result
    }};
}

#[cfg(not(test))]
macro_rules! phase {
    ($self:ident, $field:ident, $body:expr) => {
        $body
    };
}

/// Spatial broad phase over packed position/radius data.
///
/// Two things make this fast, and they are independent:
///
/// **Layout.** A rejection test reads four numbers per object - centre and
/// bounding radius. Reading those out of `PhysicalObject3D`, which also carries
/// shape, material, mass, orientation and angular velocity, pulls a cache line
/// of data the test never looks at, for each of the two objects, with the inner
/// index striding across the array. The packed arrays here stream instead.
///
/// **Algorithm.** Testing every pair is O(n^2): at 4096 objects that is 8.4M
/// rejections whether or not anything is near anything else. A uniform grid
/// sized to the objects means each object only considers the cells that could
/// possibly contain something touching it.
///
/// Objects far larger than typical - a ground plane, a wall - would force a
/// cell size so coarse that the grid degenerates back to one bucket, so they
/// are pulled out and tested against everything. In practice there are a
/// handful of these and many small objects, which is the case the split is for.
///
/// The grid itself is an open hash table built by counting sort into two flat
/// arrays, not a `HashMap<CellKey, Vec<u32>>`. The map version cost a SipHash
/// and a probe per lookup, times 28 lookups per object per step, and scattered
/// every bucket into its own heap allocation. Here a lookup is an integer hash
/// and two array reads, and the storage is two buffers reused across steps.
/// Distinct cells may collide into the same bucket; that only adds candidates,
/// which the bounding-sphere test rejects, so results are unaffected.
/// One grid of a [`BroadPhase`], holding objects whose diameter fits `cell_size`.
#[derive(Default)]
struct GridLevel {
    /// Edge length of a cell at this level.
    cell_size: f64,
    /// `1.0 / cell_size`, kept to turn a division into a multiply per lookup.
    inv_cell: f64,
    /// `table_size - 1`; table size is always a power of two.
    table_mask: u32,
    /// Bucket boundaries: bucket `b` owns `items[cell_start[b]..cell_start[b+1]]`.
    cell_start: Vec<u32>,
    /// Write cursors used while scattering; kept to avoid a per-step allocation.
    cursor: Vec<u32>,
    /// Object indices at this level, grouped by bucket, ascending within a bucket.
    items: Vec<u32>,
    /// Cell hash of each entry in `items`, used to reject bucket collisions.
    item_keys: Vec<u64>,
    /// One bit per bucket: set means "may hold something", clear means empty.
    ///
    /// Most neighbour probes in any non-crowded scene land on empty buckets, and
    /// discovering that via `cell_start` is a random read into a table far
    /// larger than L1. This bitset is 1/32nd the size and answers the common
    /// case without touching it.
    occupied: Vec<u64>,
}

#[derive(Default)]
struct BroadPhase {
    /// Packed centres, one entry per object, parallel to `PhysicsWorld::objects`.
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    /// Packed bounding radii.
    radius: Vec<f64>,
    /// Orientation per object, computed once per step rather than once per pair.
    orientations: Vec<Quaternion>,
    /// True for objects handled by the oversized path rather than the grid.
    is_oversized: Vec<bool>,
    /// Indices of oversized objects.
    oversized: Vec<u32>,
    /// Cell of each object *at its own level*. Oversized entries are unused.
    cell_coords: Vec<CellKey>,
    /// Cell hash of each object at its own level; avoids rehashing per pass.
    cell_hashes: Vec<u64>,
    /// Which grid level each object belongs to.
    object_level: Vec<u8>,
    /// Grids from finest to coarsest. A uniform-radius scene produces exactly
    /// one, in which case this behaves identically to a single flat grid.
    levels: Vec<GridLevel>,
    /// Candidate pairs surviving the broad phase, sorted and deduplicated.
    ///
    /// Packed as `(lo << 32) | hi` rather than `(u32, u32)`: sorting compares a
    /// single register instead of running derived lexicographic comparison on a
    /// tuple, and packed order is identical to tuple order because `lo < hi`.
    pairs: Vec<u64>,
    /// Scratch buffer for the median calculation, kept to avoid a per-step alloc.
    radius_scratch: Vec<f64>,
    /// Accumulated sub-phase timings (test builds only)
    #[cfg(test)]
    pub(crate) profile: BroadPhaseTimings,
}

impl BroadPhase {
    /// A radius this many times the median marks an object as oversized.
    ///
    /// This is an outlier test, not a percentile split: in a scene of balls plus
    /// a ground plane the ground is orders of magnitude larger, while the balls
    /// cluster near the median. A percentile would misclassify a fixed fraction
    /// of ordinary objects no matter how uniform they were.
    ///
    /// Tied to the level count, because the levels span exactly this ratio of
    /// radii. Anything the grid can hold belongs in the grid: the oversized path
    /// is O(k*n), so misrouting even a small *fraction* of objects into it -
    /// rather than a fixed handful - is quadratic. A scene of 90% debris and 10%
    /// crates put every crate on that path at the old factor of 4 and cost 40 ms
    /// a step.
    const OVERSIZE_FACTOR: f64 = (1u64 << (Self::MAX_LEVELS - 1)) as f64;

    /// Smallest usable cell size, guarding against a world of zero-radius points.
    const MIN_CELL_SIZE: f64 = 1e-6;

    /// Ceiling on grid levels.
    ///
    /// Each level above an object's own costs it a full 27-cell sweep, so the
    /// levels have to stay few. Four covers a 16x radius span, and the oversized
    /// path already absorbs true outliers beyond that.
    const MAX_LEVELS: usize = 4;

    /// The half of the 3x3x3 neighbourhood that is lexicographically after the
    /// centre cell.
    ///
    /// Scanning all 27 neighbours per object visits every cell pair twice and
    /// then discards half the results. Scanning only the forward half and
    /// emitting every pair found there covers each unordered cell pair exactly
    /// once, halving the lookups. The centre cell is handled separately, where
    /// `j > i` breaks the tie within a single cell.
    const FORWARD_OFFSETS: [CellKey; 13] = [
        (0, 0, 1),
        (0, 1, -1),
        (0, 1, 0),
        (0, 1, 1),
        (1, -1, -1),
        (1, -1, 0),
        (1, -1, 1),
        (1, 0, -1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, -1),
        (1, 1, 0),
        (1, 1, 1),
    ];

    /// Hash a cell to 64 bits.
    ///
    /// The low bits select the bucket; the full value is stored per item and
    /// compared on lookup. That comparison is what keeps an empty cell empty:
    /// without it, a query for a vacant cell returns whatever unrelated objects
    /// happen to share its bucket, and a sparse scene generates tens of
    /// thousands of candidates that only the narrow phase can reject.
    ///
    /// Two distinct cells colliding across all 64 bits would produce a spurious
    /// candidate, never a wrong result - the bounding-sphere test still runs.
    /// Three *independent* multiplies, and nothing after them.
    ///
    /// This sits on the critical path 14 times per object per step, so latency
    /// matters more than avalanche quality. A splitmix-style finalizer chained
    /// two more dependent multiplies onto the end and measured ~10.6 ns per
    /// lookup; these three issue in parallel and the xors are free by
    /// comparison. Bucket selection takes the high bits instead - see
    /// [`Self::bucket_index`] - which a multiply already mixes well.
    /// Three *independent* multiplies, and nothing after them.
    ///
    /// Z-order (Morton) coding was tried here to make neighbour probes
    /// cache-local, in two forms: the raw code masked for the bucket, and a
    /// hybrid passing an 8x8x8 block through untouched while scattering the
    /// block address above it. Both regressed every scene - raw Morton by 16%
    /// sparse / 11% dense / 36% clustered, the hybrid by more. The low bits of a
    /// Morton code only distinguish cells inside a ~25-cell cube, so any
    /// real-sized scene aliases into long bucket chains, and the scan cost of
    /// those chains dwarfs the locality it buys. Locality here is worth less
    /// than distribution; do not re-litigate without measuring all three scenes.
    #[inline]
    fn cell_hash(cell: CellKey) -> u64 {
        let (x, y, z) = cell;
        (x as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (y as i64 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
            ^ (z as i64 as u64).wrapping_mul(0x1656_67B1_9E37_79F9)
    }

    /// Select a bucket from a cell hash.
    ///
    /// Uses the high half: the low bits of `value * odd_constant` barely move
    /// (bit 0 is just bit 0 of the input), so masking them directly would pile
    /// axis-aligned scenes into a handful of buckets.
    #[inline]
    fn bucket_index(hash: u64, mask: u32) -> usize {
        (((hash >> 32) as u32) & mask) as usize
    }

    /// Recompute all derived data for the current object positions.
    fn rebuild(&mut self, objects: &[PhysicalObject3D]) {
        let n = objects.len();

        self.x.clear();
        self.y.clear();
        self.z.clear();
        self.radius.clear();
        self.orientations.clear();
        self.is_oversized.clear();
        self.oversized.clear();
        self.pairs.clear();
        self.cell_coords.clear();
        self.cell_hashes.clear();
        self.object_level.clear();

        phase!(self, fill, {
            for obj in objects {
                self.x.push(obj.object.position.x);
                self.y.push(obj.object.position.y);
                self.z.push(obj.object.position.z);
                self.radius.push(obj.shape.bounding_radius());
                self.orientations.push(Quaternion::from_euler(
                    obj.orientation.roll,
                    obj.orientation.pitch,
                    obj.orientation.yaw,
                ));
            }
        });

        if n < 2 {
            return;
        }

        // Median radius, via selection rather than a full sort.
        let median = phase!(self, median, {
            self.radius_scratch.clear();
            self.radius_scratch.extend_from_slice(&self.radius);
            let mid = n / 2;
            self.radius_scratch.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
            self.radius_scratch[mid]
        });

        let oversize_threshold = if median > 0.0 {
            median * Self::OVERSIZE_FACTOR
        } else {
            // Degenerate: mostly zero-radius objects. Nothing is an outlier;
            // fall through to a single coarse grid rather than divide by zero.
            f64::INFINITY
        };

        let mut max_gridded_radius: f64 = 0.0;
        self.is_oversized.reserve(n);
        for i in 0..n {
            let r = self.radius[i];
            let oversized = r > oversize_threshold;
            self.is_oversized.push(oversized);
            if oversized {
                self.oversized.push(i as u32);
            } else {
                max_gridded_radius = max_gridded_radius.max(r);
            }
        }

        // A single grid must size its cells to the largest object it holds, so
        // one big object coarsens the grid for every small one - they pile up
        // many per cell and each neighbourhood sweep returns dozens of
        // candidates that are nowhere near touching. Instead, bin objects by
        // size into levels whose cell sizes double, so every object sits in a
        // grid scaled to itself.
        let mut min_gridded_radius = f64::INFINITY;
        for i in 0..n {
            if !self.is_oversized[i] {
                min_gridded_radius = min_gridded_radius.min(self.radius[i]);
            }
        }
        if !min_gridded_radius.is_finite() {
            min_gridded_radius = 0.0;
        }

        // One level per doubling of radius. Uniform radii give a span of 1 and
        // therefore exactly one level, identical to a flat grid.
        let span = if min_gridded_radius > 0.0 {
            max_gridded_radius / min_gridded_radius
        } else {
            f64::INFINITY
        };
        let level_count = if span.is_finite() {
            ((span.log2().ceil().max(0.0) as usize) + 1).clamp(1, Self::MAX_LEVELS)
        } else {
            Self::MAX_LEVELS
        };

        // The coarsest level fits the largest gridded object exactly; each level
        // below it halves. An object joins the finest level whose cells are at
        // least its diameter, which is what bounds every query to +/-1 cell.
        let top_cell = (2.0 * max_gridded_radius).max(Self::MIN_CELL_SIZE);
        self.levels.resize_with(level_count, GridLevel::default);
        for (l, level) in self.levels.iter_mut().enumerate() {
            level.cell_size = top_cell / (1u64 << (level_count - 1 - l)) as f64;
            level.inv_cell = 1.0 / level.cell_size;
            level.items.clear();
            level.item_keys.clear();
        }

        for i in 0..n {
            let needed = 2.0 * self.radius[i];
            let mut l = 0usize;
            while l + 1 < level_count && self.levels[l].cell_size < needed {
                l += 1;
            }
            self.object_level.push(l as u8);
            let cell =
                Self::cell_of(self.x[i], self.y[i], self.z[i], self.levels[l].inv_cell);
            self.cell_coords.push(cell);
            self.cell_hashes.push(Self::cell_hash(cell));
        }

        #[cfg(test)]
        let scatter_start = std::time::Instant::now();

        for l in 0..level_count {
            // Load factor near 0.125. A sparser table means shorter bucket
            // chains - which the Morton experiment showed is what this phase is
            // actually bound on - and more probes resolving in the occupancy
            // bitset without touching `cell_start` at all.
            let members = (0..n)
                .filter(|&i| !self.is_oversized[i] && self.object_level[i] as usize == l)
                .count();
            let table_size = (members.saturating_mul(8).max(16)).next_power_of_two();

            let level = &mut self.levels[l];
            level.table_mask = (table_size - 1) as u32;
            level.cell_start.clear();
            level.cell_start.resize(table_size + 1, 0);
            level.occupied.clear();
            level.occupied.resize(table_size / 64 + 1, 0);
            level.items.resize(members, 0);
            level.item_keys.resize(members, 0);

            // Counting sort: tally, prefix-sum, then scatter in ascending object
            // order so bucket contents stay ordered and reproducible.
            for i in 0..n {
                if self.is_oversized[i] || self.object_level[i] as usize != l {
                    continue;
                }
                let b = Self::bucket_index(self.cell_hashes[i], level.table_mask);
                level.cell_start[b + 1] += 1;
                level.occupied[b >> 6] |= 1u64 << (b & 63);
            }
            for b in 0..table_size {
                level.cell_start[b + 1] += level.cell_start[b];
            }

            level.cursor.clear();
            level.cursor.extend_from_slice(&level.cell_start[..table_size]);

            for i in 0..n {
                if self.is_oversized[i] || self.object_level[i] as usize != l {
                    continue;
                }
                let hash = self.cell_hashes[i];
                let b = Self::bucket_index(hash, level.table_mask);
                let slot = level.cursor[b] as usize;
                level.items[slot] = i as u32;
                level.item_keys[slot] = hash;
                level.cursor[b] += 1;
            }
        }

        #[cfg(test)]
        {
            self.profile.scatter += scatter_start.elapsed();
        }

        self.collect_pairs(n);
    }

    /// Range within `items` for the bucket belonging to `hash`.
    ///
    /// Returns indices rather than a slice so the caller keeps `items` and
    /// `pairs` as separate field borrows; a `&self` method would borrow both.
    /// Entries in the range still have to be checked against `hash` via
    /// `item_keys` - the bucket may hold unrelated cells.
    #[inline]
    fn bucket_range(cell_start: &[u32], mask: u32, hash: u64) -> (usize, usize) {
        let b = Self::bucket_index(hash, mask);
        (cell_start[b] as usize, cell_start[b + 1] as usize)
    }

    /// Pack an ordered index pair into one sortable word.
    #[inline]
    fn pack_pair(lo: u32, hi: u32) -> u64 {
        ((lo as u64) << 32) | hi as u64
    }

    /// Cheap "definitely empty" test, answered from the bitset.
    #[inline]
    fn bucket_is_empty(occupied: &[u64], mask: u32, hash: u64) -> bool {
        let b = Self::bucket_index(hash, mask);
        occupied[b >> 6] & (1u64 << (b & 63)) == 0
    }

    /// Map a position to its grid cell.
    ///
    /// A non-finite coordinate - a NaN that leaked in from a degenerate
    /// collision - would otherwise produce an unpredictable cell. Bucket those
    /// at the origin so they stay visible to collision detection, which rejects
    /// them properly, instead of silently vanishing from the broad phase.
    #[inline]
    fn cell_of(x: f64, y: f64, z: f64, inv_cell: f64) -> CellKey {
        let q = |v: f64| -> i32 {
            if v.is_finite() {
                // `as` saturates rather than wrapping. Clamp one short of the
                // limits so the +/-1 neighbour scan cannot overflow: an object
                // flung to 1e12 by a bad impulse must not panic the simulation.
                (v * inv_cell).floor().clamp(
                    (i32::MIN + 1) as f64,
                    (i32::MAX - 1) as f64,
                ) as i32
            } else {
                0
            }
        };
        (q(x), q(y), q(z))
    }

    /// Build the candidate pair list.
    ///
    /// Iterates objects, not cells.
    ///
    /// Hoisting the thirteen neighbourhood probes to once per *cell* was tried,
    /// grouping the bucket-ordered `items` into runs that share a cell. It
    /// measured 12% worse on the dense scene - at 1.06-1.07 objects per occupied
    /// cell, which is what uniform radii produce, run detection is pure overhead
    /// with nothing to amortize - and 20% better only on the mixed-radii scene,
    /// where it is swamped by candidate volume anyway. Net effect on totals was
    /// inside noise, so the simpler form stays.
    fn collect_pairs(&mut self, n: usize) {
        #[cfg(test)]
        let collect_start = std::time::Instant::now();

        // Split the field borrows so the level tables can be read while `pairs`
        // is written.
        let BroadPhase {
            x, y, z, cell_coords, cell_hashes, object_level, is_oversized, levels, pairs, ..
        } = self;

        for i in 0..n {
            if is_oversized[i] {
                continue;
            }
            let iu = i as u32;
            let own_level = object_level[i] as usize;

            // --- The object's own level ---
            //
            // Both objects of a same-level pair run this sweep, so the ordering
            // tests below are what keep each pair to a single emission.
            {
                let level = &levels[own_level];
                let (cx, cy, cz) = cell_coords[i];
                let own_hash = cell_hashes[i];

                // Own cell: `j > i` picks each within-cell pair once.
                let (lo, hi) = Self::bucket_range(&level.cell_start, level.table_mask, own_hash);
                for k in lo..hi {
                    if level.item_keys[k] != own_hash {
                        continue;
                    }
                    let j = level.items[k];
                    if j > iu {
                        pairs.push(Self::pack_pair(iu, j));
                    }
                }

                // Forward half of the neighbourhood: every pair found is new, so
                // there is no ordering test to apply here.
                for &(dx, dy, dz) in &Self::FORWARD_OFFSETS {
                    let hash = Self::cell_hash((cx + dx, cy + dy, cz + dz));
                    if Self::bucket_is_empty(&level.occupied, level.table_mask, hash) {
                        continue;
                    }
                    let (lo, hi) = Self::bucket_range(&level.cell_start, level.table_mask, hash);
                    for k in lo..hi {
                        if level.item_keys[k] != hash {
                            continue;
                        }
                        let j = level.items[k];
                        pairs.push(Self::pack_pair(iu.min(j), iu.max(j)));
                    }
                }
            }

            // --- Coarser levels ---
            //
            // Only the finer object of a cross-level pair looks upward, so these
            // sweeps need no ordering test and must cover all 27 cells rather
            // than the forward half. Radius still bounds the search to +/-1 cell:
            // both objects fit within half a cell of this level, so their centres
            // cannot be a full cell apart and still touch.
            for coarser in levels.iter().skip(own_level + 1) {
                if coarser.items.is_empty() {
                    continue;
                }
                let (cx, cy, cz) = Self::cell_of(x[i], y[i], z[i], coarser.inv_cell);

                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            let hash = Self::cell_hash((cx + dx, cy + dy, cz + dz));
                            if Self::bucket_is_empty(
                                &coarser.occupied,
                                coarser.table_mask,
                                hash,
                            ) {
                                continue;
                            }
                            let (lo, hi) = Self::bucket_range(
                                &coarser.cell_start,
                                coarser.table_mask,
                                hash,
                            );
                            for k in lo..hi {
                                if coarser.item_keys[k] != hash {
                                    continue;
                                }
                                let j = coarser.items[k];
                                pairs.push(Self::pack_pair(iu.min(j), iu.max(j)));
                            }
                        }
                    }
                }
            }
        }

        // Oversized objects against everything else...
        for &o in &self.oversized {
            for k in 0..n as u32 {
                if k == o || self.is_oversized[k as usize] {
                    continue;
                }
                self.pairs.push(Self::pack_pair(o.min(k), o.max(k)));
            }
        }

        // ...and against each other.
        for a in 0..self.oversized.len() {
            for b in (a + 1)..self.oversized.len() {
                let (i, j) = (self.oversized[a], self.oversized[b]);
                self.pairs.push(Self::pack_pair(i.min(j), i.max(j)));
            }
        }

        // Sorting is not just tidiness. Collision response is applied in list
        // order and mutates objects, so the order decides the result. Sorting
        // by (i, j) reproduces exactly the order the old all-pairs loop
        // produced, which keeps this a pure optimization and keeps results
        // reproducible run to run.
        #[cfg(test)]
        {
            self.profile.collect += collect_start.elapsed();
        }

        #[cfg(test)]
        #[cfg(test)]
        let sort_start = std::time::Instant::now();
        self.pairs.sort_unstable();
        self.pairs.dedup();
        #[cfg(test)]
        {
            self.profile.sort_dedup += sort_start.elapsed();
        }
    }
}

/// The main physics simulation world
///
/// Manages all physics objects and runs the simulation step.
/// This can be used directly for single-threaded simulation,
/// or through `PhysicsHandle` for background threaded simulation.
pub struct PhysicsWorld {
    /// All physics objects in the world
    objects: Vec<PhysicalObject3D>,

    /// Map from ObjectId to index in objects vector
    object_ids: HashMap<ObjectId, usize>,

    /// Reverse map from index to ObjectId
    index_to_id: HashMap<usize, ObjectId>,

    /// World configuration
    config: WorldConfig,

    /// Current simulation tick
    tick: u64,

    /// Current simulation time
    time: f64,

    /// Accumulated time for fixed timestep
    accumulated_time: f64,

    /// Whether simulation is paused
    paused: bool,

    /// Pending one-shot forces to apply (cleared each step)
    pending_forces: HashMap<ObjectId, Vec<(f64, f64, f64)>>,

    /// Continuous forces that persist across steps
    continuous_forces: HashMap<ForceId, ContinuousForce>,

    /// Constraints between objects (requires "constraints" feature)
    #[cfg(feature = "constraints")]
    constraints: HashMap<ConstraintId, WorldConstraint>,

    /// Number of constraint solver iterations per step
    #[cfg(feature = "constraints")]
    constraint_iterations: usize,

    /// Active contacts from the last physics step (object pairs that are touching)
    /// Key is the object ID, value is a list of all objects it's currently in contact with
    active_contacts: HashMap<ObjectId, Vec<ObjectId>>,

    /// Spatial acceleration structure, rebuilt each step and reused across steps
    broad_phase: BroadPhase,

    /// Accumulated per-phase timings (test builds only)
    #[cfg(test)]
    pub(crate) profile: PhaseTimings,
}

impl PhysicsWorld {
    /// Create a new physics world with the given configuration
    pub fn new(config: WorldConfig) -> Self {
        Self {
            objects: Vec::new(),
            object_ids: HashMap::new(),
            index_to_id: HashMap::new(),
            config,
            tick: 0,
            time: 0.0,
            accumulated_time: 0.0,
            paused: false,
            pending_forces: HashMap::new(),
            continuous_forces: HashMap::new(),
            #[cfg(feature = "constraints")]
            constraints: HashMap::new(),
            #[cfg(feature = "constraints")]
            constraint_iterations: 8, // Default iterations for Gauss-Seidel solver
            active_contacts: HashMap::new(),
            broad_phase: BroadPhase::default(),
            #[cfg(test)]
            profile: PhaseTimings::default(),
        }
    }

    /// Create a new physics world with default configuration
    pub fn default_world() -> Self {
        Self::new(WorldConfig::default())
    }

    /// Add an object to the world
    ///
    /// Returns the ObjectId assigned to the object
    pub fn add_object(&mut self, obj: PhysicalObject3D) -> ObjectId {
        let id = ObjectId::new();
        let index = self.objects.len();

        self.objects.push(obj);
        self.object_ids.insert(id, index);
        self.index_to_id.insert(index, id);

        id
    }

    /// Remove an object from the world
    ///
    /// Returns true if the object was found and removed
    pub fn remove_object(&mut self, id: ObjectId) -> bool {
        if let Some(&index) = self.object_ids.get(&id) {
            // Remove the object
            self.objects.swap_remove(index);
            self.object_ids.remove(&id);
            self.index_to_id.remove(&index);

            // If we swapped an object into this position, update its index mapping
            if index < self.objects.len() {
                // Find the id that was at the last position
                let last_index = self.objects.len();
                if let Some(&swapped_id) = self.index_to_id.get(&last_index) {
                    self.object_ids.insert(swapped_id, index);
                    self.index_to_id.remove(&last_index);
                    self.index_to_id.insert(index, swapped_id);
                }
            }

            true
        } else {
            false
        }
    }

    /// Get a reference to an object by ID
    pub fn get_object(&self, id: ObjectId) -> Option<&PhysicalObject3D> {
        self.object_ids.get(&id).map(|&idx| &self.objects[idx])
    }

    /// Get a mutable reference to an object by ID
    pub fn get_object_mut(&mut self, id: ObjectId) -> Option<&mut PhysicalObject3D> {
        if let Some(&idx) = self.object_ids.get(&id) {
            Some(&mut self.objects[idx])
        } else {
            None
        }
    }

    /// Get the number of objects in the world
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    // ========================================================================
    // Collision Query API
    // ========================================================================

    /// Get a list of all objects currently in contact with the given object.
    ///
    /// Returns the ObjectIds of all objects that collided with this object
    /// during the last physics step. The list is empty if no contacts occurred.
    ///
    /// # Example
    /// ```ignore
    /// let ball_id = world.add_object(ball);
    /// world.step();
    /// let contacts = world.get_contacts(ball_id);
    /// for other_id in contacts {
    ///     println!("Ball is touching object {:?}", other_id);
    /// }
    /// ```
    pub fn get_contacts(&self, id: ObjectId) -> Vec<ObjectId> {
        self.active_contacts
            .get(&id)
            .cloned()
            .unwrap_or_default()
    }

    /// Check if two objects are currently in contact.
    ///
    /// Returns true if the objects collided during the last physics step.
    pub fn are_in_contact(&self, id1: ObjectId, id2: ObjectId) -> bool {
        self.active_contacts
            .get(&id1)
            .map(|contacts| contacts.contains(&id2))
            .unwrap_or(false)
    }

    /// Check if an object has any contacts.
    ///
    /// Returns true if the object is touching any other object.
    pub fn has_contacts(&self, id: ObjectId) -> bool {
        self.active_contacts
            .get(&id)
            .map(|contacts| !contacts.is_empty())
            .unwrap_or(false)
    }

    /// Get the number of objects currently in contact with the given object.
    pub fn contact_count(&self, id: ObjectId) -> usize {
        self.active_contacts
            .get(&id)
            .map(|contacts| contacts.len())
            .unwrap_or(0)
    }

    /// Set the simulation timestep
    pub fn set_timestep(&mut self, dt: f64) {
        self.config.timestep = dt;
    }

    /// Get the current timestep
    pub fn timestep(&self) -> f64 {
        self.config.timestep
    }

    /// Pause the simulation
    pub fn pause(&mut self) {
        self.paused = true;
    }

    /// Resume the simulation
    pub fn resume(&mut self) {
        self.paused = false;
    }

    /// Check if simulation is paused
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Get the current simulation time
    pub fn current_time(&self) -> f64 {
        self.time
    }

    /// Get the current tick count
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Apply a force to an object (raw 3D vector)
    pub fn apply_force(&mut self, id: ObjectId, force: (f64, f64, f64)) {
        if self.object_ids.contains_key(&id) {
            self.pending_forces.entry(id).or_default().push(force);
        }
    }

    /// Apply an impulse (instant velocity change) to an object
    pub fn apply_impulse(&mut self, id: ObjectId, impulse: (f64, f64, f64)) {
        if let Some(obj) = self.get_object_mut(id) {
            let mass = obj.object.mass;
            if mass > 0.0 {
                obj.object.velocity.x += impulse.0 / mass;
                obj.object.velocity.y += impulse.1 / mass;
                obj.object.velocity.z += impulse.2 / mass;
            }
        }
    }

    // ==================== Kinematic Object Methods ====================

    /// Set position for a kinematic object (externally controlled).
    ///
    /// This method:
    /// - Sets the object's position directly
    /// - Computes velocity from the position change (for proper collision response)
    ///
    /// Use this for objects like planks that follow rope particles or
    /// platforms that move along paths. The object should have infinite mass
    /// so it's not affected by forces, but can push other objects.
    ///
    /// # Arguments
    /// * `id` - The object ID
    /// * `new_pos` - The new position (x, y, z)
    /// * `dt` - Time step (used to compute velocity from position change)
    pub fn set_position_kinematic(
        &mut self,
        id: ObjectId,
        new_pos: (f64, f64, f64),
        dt: f64,
    ) {
        if let Some(obj) = self.get_object_mut(id) {
            // Compute velocity from position change
            if dt > 0.0 {
                obj.object.velocity.x = (new_pos.0 - obj.object.position.x) / dt;
                obj.object.velocity.y = (new_pos.1 - obj.object.position.y) / dt;
                obj.object.velocity.z = (new_pos.2 - obj.object.position.z) / dt;
            }
            // Set new position directly
            obj.object.position.x = new_pos.0;
            obj.object.position.y = new_pos.1;
            obj.object.position.z = new_pos.2;
        }
    }

    /// Set orientation for a kinematic object (externally controlled).
    ///
    /// Angles are Euler `(roll, pitch, yaw)` in radians. Without this, a body
    /// driven externally could be moved but never turned, so a swinging door's
    /// collider stayed axis-aligned while its visual rotated - the collider and
    /// the thing the player can see describing different worlds.
    pub fn set_orientation_kinematic(
        &mut self,
        id: ObjectId,
        orientation: (f64, f64, f64),
        dt: f64,
    ) {
        if let Some(obj) = self.get_object_mut(id) {
            if dt > 0.0 {
                obj.angular_velocity = (
                    (orientation.0 - obj.orientation.roll) / dt,
                    (orientation.1 - obj.orientation.pitch) / dt,
                    (orientation.2 - obj.orientation.yaw) / dt,
                );
            }
            obj.orientation.roll = orientation.0;
            obj.orientation.pitch = orientation.1;
            obj.orientation.yaw = orientation.2;
        }
    }

    // ==================== Ergonomic Force Methods ====================

    /// Apply a force in a specific direction with given magnitude
    ///
    /// Direction is automatically normalized.
    pub fn apply_force_directed(&mut self, id: ObjectId, magnitude: f64, direction: (f64, f64, f64)) {
        let len = (direction.0 * direction.0 + direction.1 * direction.1 + direction.2 * direction.2).sqrt();
        if len > 1e-10 {
            let normalized = (direction.0 / len, direction.1 / len, direction.2 / len);
            self.apply_force(id, (
                normalized.0 * magnitude,
                normalized.1 * magnitude,
                normalized.2 * magnitude,
            ));
        }
    }

    /// Apply a force toward a target position (attraction)
    ///
    /// Useful for gravity wells, magnets, or AI-controlled movement.
    pub fn apply_force_toward(&mut self, id: ObjectId, target: (f64, f64, f64), magnitude: f64) {
        if let Some(obj) = self.get_object(id) {
            let dx = target.0 - obj.object.position.x;
            let dy = target.1 - obj.object.position.y;
            let dz = target.2 - obj.object.position.z;
            self.apply_force_directed(id, magnitude, (dx, dy, dz));
        }
    }

    /// Apply a force away from a position (repulsion)
    ///
    /// Useful for explosions, force fields, or avoidance.
    pub fn apply_force_away(&mut self, id: ObjectId, source: (f64, f64, f64), magnitude: f64) {
        if let Some(obj) = self.get_object(id) {
            let dx = obj.object.position.x - source.0;
            let dy = obj.object.position.y - source.1;
            let dz = obj.object.position.z - source.2;
            self.apply_force_directed(id, magnitude, (dx, dy, dz));
        }
    }

    /// Apply drag force based on current velocity
    ///
    /// drag_coefficient: typically 0.1 to 2.0 (higher = more drag)
    pub fn apply_drag(&mut self, id: ObjectId, drag_coefficient: f64) {
        if let Some(obj) = self.get_object(id) {
            let vx = obj.object.velocity.x;
            let vy = obj.object.velocity.y;
            let vz = obj.object.velocity.z;
            let speed_sq = vx * vx + vy * vy + vz * vz;

            if speed_sq > 1e-10 {
                // Drag force opposes velocity, proportional to v²
                let drag_magnitude = drag_coefficient * speed_sq;
                let speed = speed_sq.sqrt();
                self.apply_force(id, (
                    -vx / speed * drag_magnitude,
                    -vy / speed * drag_magnitude,
                    -vz / speed * drag_magnitude,
                ));
            }
        }
    }

    /// Apply spring force toward a rest position
    ///
    /// Uses Hooke's law: F = -k * displacement
    pub fn apply_spring_force(&mut self, id: ObjectId, rest_position: (f64, f64, f64), stiffness: f64) {
        if let Some(obj) = self.get_object(id) {
            let dx = obj.object.position.x - rest_position.0;
            let dy = obj.object.position.y - rest_position.1;
            let dz = obj.object.position.z - rest_position.2;
            self.apply_force(id, (
                -stiffness * dx,
                -stiffness * dy,
                -stiffness * dz,
            ));
        }
    }

    /// Apply damped spring force (spring + velocity damping)
    ///
    /// Combines spring force with damping to reduce oscillation.
    pub fn apply_damped_spring(&mut self, id: ObjectId, rest_position: (f64, f64, f64), stiffness: f64, damping: f64) {
        if let Some(obj) = self.get_object(id) {
            let dx = obj.object.position.x - rest_position.0;
            let dy = obj.object.position.y - rest_position.1;
            let dz = obj.object.position.z - rest_position.2;
            let vx = obj.object.velocity.x;
            let vy = obj.object.velocity.y;
            let vz = obj.object.velocity.z;
            self.apply_force(id, (
                -stiffness * dx - damping * vx,
                -stiffness * dy - damping * vy,
                -stiffness * dz - damping * vz,
            ));
        }
    }

    /// Apply an explosion force to all objects within radius
    ///
    /// Force falls off with distance squared.
    pub fn apply_explosion(&mut self, center: (f64, f64, f64), force: f64, radius: f64) {
        let ids: Vec<ObjectId> = self.object_ids.keys().copied().collect();
        for id in ids {
            if let Some(obj) = self.get_object(id) {
                let dx = obj.object.position.x - center.0;
                let dy = obj.object.position.y - center.1;
                let dz = obj.object.position.z - center.2;
                let dist_sq = dx * dx + dy * dy + dz * dz;
                let radius_sq = radius * radius;

                if dist_sq < radius_sq && dist_sq > 1e-10 {
                    // Force falls off with distance squared
                    let falloff = 1.0 - (dist_sq / radius_sq);
                    let magnitude = force * falloff;
                    self.apply_force_directed(id, magnitude, (dx, dy, dz));
                }
            }
        }
    }

    /// Apply torque to rotate an object
    ///
    /// Torque is converted to angular acceleration using moment of inertia,
    /// then integrated over the timestep to get change in angular velocity:
    /// Δω = τ * dt / I
    pub fn apply_torque(&mut self, id: ObjectId, torque: (f64, f64, f64)) {
        let dt = self.config.timestep;
        if let Some(obj) = self.get_object_mut(id) {
            // Get moment of inertia from shape
            // Returns [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
            let inertia = obj.shape.moment_of_inertia(obj.object.mass);

            // Angular acceleration α = τ / I
            // Change in angular velocity: Δω = α * dt = τ * dt / I
            // Using diagonal elements (Ixx, Iyy, Izz) for principal axes
            let ixx = inertia[0].max(0.001); // Prevent division by zero
            let iyy = inertia[1].max(0.001);
            let izz = inertia[2].max(0.001);

            obj.angular_velocity.0 += torque.0 * dt / ixx;
            obj.angular_velocity.1 += torque.1 * dt / iyy;
            obj.angular_velocity.2 += torque.2 * dt / izz;
        }
    }

    /// Apply a buoyancy force (upward force based on depth below a surface)
    /// NOTE: This is a one-shot force. For continuous buoyancy, use add_buoyancy().
    ///
    /// surface_y: Y coordinate of the fluid surface
    /// fluid_density: density of the fluid (water ≈ 1000 kg/m³)
    pub fn apply_buoyancy(&mut self, id: ObjectId, surface_y: f64, fluid_density: f64) {
        if let Some(obj) = self.get_object(id) {
            let y = obj.object.position.y;
            if y < surface_y {
                // Simplified buoyancy: force proportional to depth
                let depth = surface_y - y;
                let buoyancy_force = fluid_density * depth * self.config.constants.gravity;
                self.apply_force(id, (0.0, buoyancy_force, 0.0));
            }
        }
    }

    // ==================== Continuous Force Methods ====================

    /// Add a continuous force that persists across simulation steps
    ///
    /// Returns a ForceId that can be used to remove the force later.
    pub fn add_continuous_force(&mut self, force: ContinuousForce) -> ForceId {
        let id = ForceId::new();
        self.continuous_forces.insert(id, force);
        id
    }

    /// Remove a continuous force by its ID
    ///
    /// Returns true if the force was found and removed.
    pub fn remove_continuous_force(&mut self, id: ForceId) -> bool {
        self.continuous_forces.remove(&id).is_some()
    }

    /// Remove all continuous forces targeting a specific object
    pub fn remove_forces_on_object(&mut self, target: ObjectId) {
        self.continuous_forces.retain(|_, force| {
            match force {
                ContinuousForce::Constant { target: t, .. } => *t != target,
                ContinuousForce::Drag { target: t, .. } => *t != target,
                ContinuousForce::Spring { target: t, .. } => *t != target,
                ContinuousForce::DampedSpring { target: t, .. } => *t != target,
                ContinuousForce::Attract { target: t, .. } => *t != target,
                ContinuousForce::Repel { target: t, .. } => *t != target,
                ContinuousForce::Buoyancy { target: t, .. } => *t != target,
                ContinuousForce::Vortex { target: t, .. } => *t != target,
            }
        });
    }

    /// Get the number of active continuous forces
    pub fn continuous_force_count(&self) -> usize {
        self.continuous_forces.len()
    }

    // ==================== Continuous Force Convenience Methods ====================

    /// Add continuous drag to an object (auto-removes when velocity < min_velocity)
    pub fn add_drag(&mut self, target: ObjectId, coefficient: f64) -> ForceId {
        self.add_continuous_force(ContinuousForce::Drag {
            target,
            coefficient,
            min_velocity: 0.01,
        })
    }

    /// Add continuous drag with custom minimum velocity threshold
    pub fn add_drag_with_threshold(&mut self, target: ObjectId, coefficient: f64, min_velocity: f64) -> ForceId {
        self.add_continuous_force(ContinuousForce::Drag {
            target,
            coefficient,
            min_velocity,
        })
    }

    /// Add a continuous spring force (auto-removes when at rest)
    pub fn add_spring(&mut self, target: ObjectId, rest_position: (f64, f64, f64), stiffness: f64) -> ForceId {
        self.add_continuous_force(ContinuousForce::Spring {
            target,
            rest_position,
            stiffness,
            min_displacement: 0.01,
            min_velocity: 0.01,
        })
    }

    /// Add a continuous damped spring (auto-removes when at rest)
    pub fn add_damped_spring(&mut self, target: ObjectId, rest_position: (f64, f64, f64), stiffness: f64, damping: f64) -> ForceId {
        self.add_continuous_force(ContinuousForce::DampedSpring {
            target,
            rest_position,
            stiffness,
            damping,
            min_displacement: 0.05,
            // A settled spring removes itself, so the residual velocity it
            // tolerates has to be small enough that the coast afterwards stays
            // inside min_displacement. Only LINEAR_DAMPING (0.1) acts once the
            // spring is gone, and an object at velocity v coasts v / 0.1 = 10v
            // before stopping. At the previous 0.05 that was a 0.5 m drift -
            // ten times the displacement the removal check had just verified,
            // so the object quietly wandered off the rest position it had
            // reached. 0.005 bounds the coast to 0.05.
            min_velocity: 0.005,
        })
    }

    /// Add continuous attraction toward a point (persists until removed or duration expires)
    pub fn add_attraction(&mut self, target: ObjectId, point: (f64, f64, f64), strength: f64, duration: Option<f64>) -> ForceId {
        self.add_continuous_force(ContinuousForce::Attract {
            target,
            point,
            strength,
            duration,
            elapsed: 0.0,
        })
    }

    /// Add continuous repulsion from a point
    pub fn add_repulsion(&mut self, target: ObjectId, point: (f64, f64, f64), strength: f64, duration: Option<f64>) -> ForceId {
        self.add_continuous_force(ContinuousForce::Repel {
            target,
            point,
            strength,
            duration,
            elapsed: 0.0,
        })
    }

    /// Add continuous buoyancy (auto-removes after being above surface for a while)
    pub fn add_buoyancy(&mut self, target: ObjectId, surface_y: f64, fluid_density: f64) -> ForceId {
        self.add_continuous_force(ContinuousForce::Buoyancy {
            target,
            surface_y,
            fluid_density,
            time_above_surface: 0.0,
            removal_delay: 1.0,
        })
    }

    /// Add a constant continuous force (like wind or thrust)
    pub fn add_constant_force(&mut self, target: ObjectId, force: (f64, f64, f64)) -> ForceId {
        self.add_continuous_force(ContinuousForce::Constant { target, force })
    }

    /// Add a vortex/rotational force
    pub fn add_vortex(&mut self, target: ObjectId, center: (f64, f64, f64), axis: (f64, f64, f64), strength: f64, duration: Option<f64>) -> ForceId {
        self.add_continuous_force(ContinuousForce::Vortex {
            target,
            center,
            axis,
            strength,
            duration,
            elapsed: 0.0,
        })
    }

    // ==================== Constraint Methods ====================

    /// Add a constraint to the world
    ///
    /// Returns the ConstraintId assigned to the constraint.
    #[cfg(feature = "constraints")]
    pub fn add_constraint(&mut self, constraint: WorldConstraint) -> ConstraintId {
        let id = ConstraintId::new();
        self.constraints.insert(id, constraint);
        id
    }

    /// Remove a constraint from the world
    ///
    /// Returns true if the constraint was found and removed.
    #[cfg(feature = "constraints")]
    pub fn remove_constraint(&mut self, id: ConstraintId) -> bool {
        self.constraints.remove(&id).is_some()
    }

    /// Get a reference to a constraint by ID
    #[cfg(feature = "constraints")]
    pub fn get_constraint(&self, id: ConstraintId) -> Option<&WorldConstraint> {
        self.constraints.get(&id)
    }

    /// Get a mutable reference to a constraint by ID
    #[cfg(feature = "constraints")]
    pub fn get_constraint_mut(&mut self, id: ConstraintId) -> Option<&mut WorldConstraint> {
        self.constraints.get_mut(&id)
    }

    /// Get the number of constraints in the world
    #[cfg(feature = "constraints")]
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// Set the number of constraint solver iterations
    #[cfg(feature = "constraints")]
    pub fn set_constraint_iterations(&mut self, iterations: usize) {
        self.constraint_iterations = iterations;
    }

    /// Get the current constraint solver iteration count
    #[cfg(feature = "constraints")]
    pub fn constraint_iterations(&self) -> usize {
        self.constraint_iterations
    }

    /// Get an iterator over all constraint IDs
    #[cfg(feature = "constraints")]
    pub fn constraint_ids(&self) -> impl Iterator<Item = ConstraintId> + '_ {
        self.constraints.keys().copied()
    }

    /// Clear all constraints from the world
    #[cfg(feature = "constraints")]
    pub fn clear_constraints(&mut self) {
        self.constraints.clear();
    }

    /// Set the velocity of an object
    pub fn set_velocity(&mut self, id: ObjectId, velocity: (f64, f64, f64)) {
        if let Some(obj) = self.get_object_mut(id) {
            obj.object.velocity.x = velocity.0;
            obj.object.velocity.y = velocity.1;
            obj.object.velocity.z = velocity.2;
        }
    }

    /// Set the position of an object
    pub fn set_position(&mut self, id: ObjectId, position: (f64, f64, f64)) {
        if let Some(obj) = self.get_object_mut(id) {
            obj.object.position.x = position.0;
            obj.object.position.y = position.1;
            obj.object.position.z = position.2;
        }
    }

    /// Advance simulation by real-time delta
    ///
    /// Uses fixed timestep accumulator pattern to ensure deterministic physics.
    /// Returns the number of physics steps taken.
    pub fn update(&mut self, real_dt: f64) -> usize {
        if self.paused {
            return 0;
        }

        self.accumulated_time += real_dt;
        let mut steps = 0;

        while self.accumulated_time >= self.config.timestep {
            self.step_internal();
            self.accumulated_time -= self.config.timestep;
            steps += 1;

            // Safety limit to prevent spiral of death
            if steps >= 10 {
                self.accumulated_time = 0.0;
                break;
            }
        }

        steps
    }

    /// Size at which the determinism tests exercise a realistically large world.
    #[cfg(test)]
    const LARGE_WORLD: usize = 192;

    /// Apply an independent per-object update.
    ///
    /// These stages were parallelized with `par_iter_mut` and measured slower at
    /// every size from 64 to 4096 objects (+121% at n=64, +0.4% at n=4096): the
    /// per-object work is a handful of multiply-adds, so rayon's fixed fork-join
    /// cost per parallel section exceeds the work being scheduled, and the step
    /// is dominated by the O(n^2) narrow phase regardless. Kept sequential
    /// deliberately - see `bench_step_cost` to re-check that decision.
    ///
    /// `f` must only touch the object it is given.
    fn for_each_object<F>(objects: &mut [PhysicalObject3D], f: F)
    where
        F: Fn(&mut PhysicalObject3D),
    {
        objects.iter_mut().for_each(f);
    }

    /// Perform a single physics step
    pub fn step(&mut self) {
        if !self.paused {
            self.step_internal();
        }
    }

    /// Internal step implementation
    fn step_internal(&mut self) {
        let dt = self.config.timestep;
        let world_gravity = self.config.constants.gravity;
        // Use the gravity Y component from config (gravity vector is (x, y, z))
        let gravity = -self.config.gravity.1;  // Negate because apply_gravity expects positive down

        // 1. Apply gravity to all objects (skip static objects with infinite/very high mass)
        const STATIC_MASS_THRESHOLD: f64 = 1e20; // Objects heavier than this are treated as static
        phase!(self, gravity, {
        Self::for_each_object(&mut self.objects, |obj| {
            if obj.object.mass < STATIC_MASS_THRESHOLD && !obj.object.mass.is_infinite() {
                apply_gravity(obj, gravity, dt);
            }
        });
        });

        // 2. Apply continuous forces and collect expired ones
        phase!(self, continuous_forces, {
            let forces_to_remove = self.apply_continuous_forces(dt, world_gravity);
            for force_id in forces_to_remove {
                self.continuous_forces.remove(&force_id);
            }
        });

        // 3. Apply pending one-shot forces and integrate velocities
        //
        // Each object reads only its own force list, and that list is a Vec, so
        // the accumulation order within an object is fixed regardless of how the
        // objects are distributed across threads.
        // Nothing pending is the common case, and the loop below would still do
        // a HashMap lookup per object to discover that.
        if !self.pending_forces.is_empty() {
            // Split the borrows so the object slice can be handed to rayon while
            // the lookup tables stay readable.
            let index_to_id = &self.index_to_id;
            let pending_forces = &self.pending_forces;

            let apply_pending = |(idx, obj): (usize, &mut PhysicalObject3D)| {
                let Some(&id) = index_to_id.get(&idx) else { return };
                let Some(forces) = pending_forces.get(&id) else { return };

                let mass = obj.object.mass;
                if mass <= 0.0 {
                    return;
                }
                for force in forces {
                    obj.object.velocity.x += force.0 / mass * dt;
                    obj.object.velocity.y += force.1 / mass * dt;
                    obj.object.velocity.z += force.2 / mass * dt;
                }
            };

            #[cfg(test)]
            #[cfg(test)]
            let start = std::time::Instant::now();
            self.objects.iter_mut().enumerate().for_each(apply_pending);
            #[cfg(test)]
            {
                self.profile.pending_forces += start.elapsed();
            }
        }
        // Clear all pending one-shot forces
        self.pending_forces.clear();

        // 4. Collision detection and response
        self.resolve_collisions(dt);

        // 4.5. Apply gravity to constraint-owned particles and solve constraints
        #[cfg(feature = "constraints")]
        phase!(self, constraints, self.solve_constraints(dt, gravity));

        // 4.6. Apply damping (air resistance and angular friction)
        // Using exponential decay for frame-rate independence
        // damping_factor = (1 - damping)^dt approximated as e^(-damping * dt)
        // Decay per second: exp(-coefficient). Linear 0.1 sheds ~9.5%/s.
        //
        // Angular was 2.0, which sheds ~86.5%/s - twenty times the linear rate,
        // so a ball lost most of its spin within a second even in mid-air, where
        // nothing should be removing angular momentum. That reads as rotation
        // decoupled from motion. 0.3 sheds ~26%/s: still above the linear rate,
        // which is defensible because rolling resistance is real, without the
        // spin visibly dying on its own.
        const LINEAR_DAMPING: f64 = 0.1;   // Linear velocity damping coefficient
        const ANGULAR_DAMPING: f64 = 0.3;  // Angular velocity damping coefficient

        let linear_decay = (-LINEAR_DAMPING * dt).exp();
        let angular_decay = (-ANGULAR_DAMPING * dt).exp();

        phase!(self, damping, Self::for_each_object(&mut self.objects, |obj| {
            // Skip static objects
            if obj.object.mass.is_infinite() || obj.object.mass <= 0.0 {
                return;
            }

            // Apply linear damping (air resistance)
            obj.object.velocity.x *= linear_decay;
            obj.object.velocity.y *= linear_decay;
            obj.object.velocity.z *= linear_decay;

            // Apply angular damping (rotational friction)
            obj.angular_velocity.0 *= angular_decay;
            obj.angular_velocity.1 *= angular_decay;
            obj.angular_velocity.2 *= angular_decay;
        }));

        // 5. Integrate positions
        phase!(self, integrate, Self::for_each_object(&mut self.objects, |obj| {
            obj.object.position.x += obj.object.velocity.x * dt;
            obj.object.position.y += obj.object.velocity.y * dt;
            obj.object.position.z += obj.object.velocity.z * dt;

            // Update orientation from angular velocity
            obj.orientation.roll += obj.angular_velocity.0 * dt;
            obj.orientation.pitch += obj.angular_velocity.1 * dt;
            obj.orientation.yaw += obj.angular_velocity.2 * dt;
        }));

        // 6. Optional implicit floor, for worlds with no ground collider.
        //    Off by default - see WorldConfig::ground_plane for why.
        if let Some(floor) = self.config.ground_plane {
            phase!(self, tunneling, Self::for_each_object(&mut self.objects, |obj| {
                // Skip static objects
                if obj.object.mass.is_infinite() || obj.object.mass <= 0.0 {
                    return;
                }

                // Lowest point of the object. Note the cuboid case ignores
                // orientation, so a rotated box reports its unrotated extent;
                // another reason this is a net rather than a collider.
                let min_y = match &obj.shape {
                    Shape3D::Sphere(radius) => obj.object.position.y - radius,
                    Shape3D::Cuboid(_, h, _) => obj.object.position.y - h / 2.0,
                    Shape3D::Cylinder(_radius, height) => obj.object.position.y - height / 2.0,
                    _ => obj.object.position.y - obj.shape.bounding_radius(),
                };

                if min_y < floor {
                    obj.object.position.y += floor - min_y;

                    // If moving downward, bounce with reduced restitution
                    if obj.object.velocity.y < 0.0 {
                        let restitution = obj.get_restitution() * 0.5; // Reduced for tunneling recovery
                        obj.object.velocity.y = -obj.object.velocity.y * restitution;
                    }
                }
            }));
        }

        // 6. Update time tracking
        self.tick += 1;
        self.time += dt;
    }

    /// Apply all continuous forces and return IDs of forces that should be removed
    fn apply_continuous_forces(&mut self, dt: f64, world_gravity: f64) -> Vec<ForceId> {
        let mut to_remove = Vec::new();

        // Collect force computations first (to avoid borrow issues)
        let mut force_updates: Vec<(ForceId, ObjectId, (f64, f64, f64), bool)> = Vec::new();

        for (&force_id, force) in &mut self.continuous_forces {
            match force {
                ContinuousForce::Constant { target, force: f } => {
                    if self.object_ids.contains_key(target) {
                        force_updates.push((force_id, *target, *f, false));
                    } else {
                        to_remove.push(force_id); // Target no longer exists
                    }
                }

                ContinuousForce::Drag { target, coefficient, min_velocity } => {
                    if let Some(&idx) = self.object_ids.get(target) {
                        let obj = &self.objects[idx];
                        let vx = obj.object.velocity.x;
                        let vy = obj.object.velocity.y;
                        let vz = obj.object.velocity.z;
                        let speed_sq = vx * vx + vy * vy + vz * vz;

                        if speed_sq < *min_velocity * *min_velocity {
                            to_remove.push(force_id);
                        } else {
                            let speed = speed_sq.sqrt();
                            let drag_mag = *coefficient * speed_sq;
                            let f = (
                                -vx / speed * drag_mag,
                                -vy / speed * drag_mag,
                                -vz / speed * drag_mag,
                            );
                            force_updates.push((force_id, *target, f, false));
                        }
                    } else {
                        to_remove.push(force_id);
                    }
                }

                ContinuousForce::Spring { target, rest_position, stiffness, min_displacement, min_velocity } => {
                    if let Some(&idx) = self.object_ids.get(target) {
                        let obj = &self.objects[idx];
                        let dx = obj.object.position.x - rest_position.0;
                        let dy = obj.object.position.y - rest_position.1;
                        let dz = obj.object.position.z - rest_position.2;
                        let dist_sq = dx * dx + dy * dy + dz * dz;

                        let vx = obj.object.velocity.x;
                        let vy = obj.object.velocity.y;
                        let vz = obj.object.velocity.z;
                        let speed_sq = vx * vx + vy * vy + vz * vz;

                        if dist_sq < *min_displacement * *min_displacement
                            && speed_sq < *min_velocity * *min_velocity
                        {
                            to_remove.push(force_id);
                        } else {
                            let f = (-*stiffness * dx, -*stiffness * dy, -*stiffness * dz);
                            force_updates.push((force_id, *target, f, false));
                        }
                    } else {
                        to_remove.push(force_id);
                    }
                }

                ContinuousForce::DampedSpring { target, rest_position, stiffness, damping, min_displacement, min_velocity } => {
                    if let Some(&idx) = self.object_ids.get(target) {
                        let obj = &self.objects[idx];
                        let dx = obj.object.position.x - rest_position.0;
                        let dy = obj.object.position.y - rest_position.1;
                        let dz = obj.object.position.z - rest_position.2;
                        let dist_sq = dx * dx + dy * dy + dz * dz;

                        let vx = obj.object.velocity.x;
                        let vy = obj.object.velocity.y;
                        let vz = obj.object.velocity.z;
                        let speed_sq = vx * vx + vy * vy + vz * vz;

                        if dist_sq < *min_displacement * *min_displacement
                            && speed_sq < *min_velocity * *min_velocity
                        {
                            to_remove.push(force_id);
                        } else {
                            let f = (
                                -*stiffness * dx - *damping * vx,
                                -*stiffness * dy - *damping * vy,
                                -*stiffness * dz - *damping * vz,
                            );
                            force_updates.push((force_id, *target, f, false));
                        }
                    } else {
                        to_remove.push(force_id);
                    }
                }

                ContinuousForce::Attract { target, point, strength, duration, elapsed } => {
                    *elapsed += dt;
                    if duration.map_or(false, |d| *elapsed >= d) {
                        to_remove.push(force_id);
                    } else if let Some(&idx) = self.object_ids.get(target) {
                        let obj = &self.objects[idx];
                        let dx = point.0 - obj.object.position.x;
                        let dy = point.1 - obj.object.position.y;
                        let dz = point.2 - obj.object.position.z;
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                        if dist > 1e-6 {
                            let f = (
                                dx / dist * *strength,
                                dy / dist * *strength,
                                dz / dist * *strength,
                            );
                            force_updates.push((force_id, *target, f, false));
                        }
                    } else {
                        to_remove.push(force_id);
                    }
                }

                ContinuousForce::Repel { target, point, strength, duration, elapsed } => {
                    *elapsed += dt;
                    if duration.map_or(false, |d| *elapsed >= d) {
                        to_remove.push(force_id);
                    } else if let Some(&idx) = self.object_ids.get(target) {
                        let obj = &self.objects[idx];
                        let dx = obj.object.position.x - point.0;
                        let dy = obj.object.position.y - point.1;
                        let dz = obj.object.position.z - point.2;
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                        if dist > 1e-6 {
                            let f = (
                                dx / dist * *strength,
                                dy / dist * *strength,
                                dz / dist * *strength,
                            );
                            force_updates.push((force_id, *target, f, false));
                        }
                    } else {
                        to_remove.push(force_id);
                    }
                }

                ContinuousForce::Buoyancy { target, surface_y, fluid_density, time_above_surface, removal_delay } => {
                    if let Some(&idx) = self.object_ids.get(target) {
                        let obj = &self.objects[idx];
                        let y = obj.object.position.y;

                        if y < *surface_y {
                            *time_above_surface = 0.0;
                            let depth = *surface_y - y;
                            let buoyancy = *fluid_density * depth * world_gravity;
                            force_updates.push((force_id, *target, (0.0, buoyancy, 0.0), false));
                        } else {
                            *time_above_surface += dt;
                            if *time_above_surface >= *removal_delay {
                                to_remove.push(force_id);
                            }
                        }
                    } else {
                        to_remove.push(force_id);
                    }
                }

                ContinuousForce::Vortex { target, center, axis, strength, duration, elapsed } => {
                    *elapsed += dt;
                    if duration.map_or(false, |d| *elapsed >= d) {
                        to_remove.push(force_id);
                    } else if let Some(&idx) = self.object_ids.get(target) {
                        let obj = &self.objects[idx];
                        // Vector from center to object
                        let rx = obj.object.position.x - center.0;
                        let ry = obj.object.position.y - center.1;
                        let rz = obj.object.position.z - center.2;

                        // Cross product: axis × r gives tangential direction
                        let fx = axis.1 * rz - axis.2 * ry;
                        let fy = axis.2 * rx - axis.0 * rz;
                        let fz = axis.0 * ry - axis.1 * rx;

                        let mag = (fx * fx + fy * fy + fz * fz).sqrt();
                        if mag > 1e-6 {
                            let f = (
                                fx / mag * *strength,
                                fy / mag * *strength,
                                fz / mag * *strength,
                            );
                            force_updates.push((force_id, *target, f, false));
                        }
                    } else {
                        to_remove.push(force_id);
                    }
                }
            }
        }

        // Apply the computed forces
        // `continuous_forces` is a HashMap, so the collection order above varies
        // between process runs. These are float accumulations into velocity, and
        // float addition is not associative - two forces on the same object
        // would sum differently run to run, and the simulation would diverge.
        // Sorting by ForceId (monotonic, unique) pins a stable order.
        force_updates.sort_unstable_by_key(|&(force_id, ..)| force_id);

        for (_force_id, target_id, force, _) in force_updates {
            if let Some(&idx) = self.object_ids.get(&target_id) {
                let obj = &mut self.objects[idx];
                let mass = obj.object.mass;
                if mass > 0.0 {
                    obj.object.velocity.x += force.0 / mass * dt;
                    obj.object.velocity.y += force.1 / mass * dt;
                    obj.object.velocity.z += force.2 / mass * dt;
                }
            }
        }

        to_remove
    }

    /// Solve all constraints using iterative Gauss-Seidel solver
    #[cfg(feature = "constraints")]
    fn solve_constraints(&mut self, dt: f64, gravity: f64) {
        if self.constraints.is_empty() {
            return;
        }

        // First pass: Apply gravity to constraint-owned particles (RopeChain, etc.)
        for constraint in self.constraints.values_mut() {
            constraint.apply_gravity(gravity, dt);
        }

        // Iterative constraint solving (Gauss-Seidel)
        let iterations = self.constraint_iterations;
        for _ in 0..iterations {
            for constraint in self.constraints.values_mut() {
                constraint.solve(&self.object_ids, &mut self.objects, dt);
            }
        }
    }

    /// Resolve collisions between all object pairs using parallel detection
    ///
    /// This uses a two-phase approach:
    /// 1. Parallel detection: Find all colliding pairs and compute contact data (read-only)
    /// 2. Sequential response: Apply impulses and position corrections (write)
    fn resolve_collisions(&mut self, dt: f64) {
        // Clear contacts from last frame
        self.active_contacts.clear();

        let n = self.objects.len();
        if n < 2 {
            return;
        }

        // There used to be a separate sequential path here for n < 8, on the
        // theory that the broad phase wasn't worth its overhead for a handful of
        // objects. It cost a few hundred nanoseconds and bought a second,
        // divergent collision-response implementation - which had an inverted
        // penetration correction that drove objects *through* surfaces instead
        // of out of them. Two implementations of the same physics is a bug
        // factory; the broad phase handles small worlds fine.

        // Phase 0: Rebuild the spatial index. Note this runs before position
        // integration (step 5), so contacts are resolved against the positions
        // this tick started with.
        phase!(self, broad_rebuild, {
            let objects = &self.objects;
            self.broad_phase.rebuild(objects);
        });

        // Phase 1: Parallel collision detection (read-only)
        let collisions = phase!(self, narrow_phase, self.detect_collisions_parallel());

        // Phase 2: Sequential collision response (write) and contact tracking
        #[cfg(test)]
        #[cfg(test)]
        let response_start = std::time::Instant::now();
        for collision in &collisions {
            // Track contacts bidirectionally
            if let (Some(&id1), Some(&id2)) = (
                self.index_to_id.get(&collision.i),
                self.index_to_id.get(&collision.j),
            ) {
                self.active_contacts.entry(id1).or_default().push(id2);
                self.active_contacts.entry(id2).or_default().push(id1);
            }
            self.apply_collision_response(collision, dt);
        }
        #[cfg(test)]
        {
            self.profile.response += response_start.elapsed();
        }
    }


    /// Parallel collision detection - returns collision data without mutating objects
    ///
    /// Assumes `self.broad_phase` has already been rebuilt for the current
    /// positions. The candidate list is sorted by `(i, j)`, and rayon's indexed
    /// `collect` preserves input order, so the result is identical - in content
    /// and order - to testing every pair.
    fn detect_collisions_parallel(&self) -> Vec<CollisionData> {
        self.broad_phase
            .pairs
            .par_iter()
            .filter_map(|&packed| {
                let (i, j) = ((packed >> 32) as usize, (packed as u32) as usize);
                self.detect_collision_pair(i, j)
            })
            .collect()
    }

    /// Detect collision between a pair, reading the packed broad-phase arrays.
    ///
    /// The rejection test - the overwhelming majority of calls - touches four
    /// contiguous arrays instead of two fat structs, and the orientations were
    /// computed once per object during the rebuild rather than once per pair.
    ///
    /// Requires `self.broad_phase` to match the current object positions.
    fn detect_collision_pair(&self, i: usize, j: usize) -> Option<CollisionData> {
        let bp = &self.broad_phase;
        self.detect_collision_with(
            i,
            j,
            (bp.x[i], bp.y[i], bp.z[i]),
            (bp.x[j], bp.y[j], bp.z[j]),
            bp.radius[i],
            bp.radius[j],
            bp.orientations[i],
            bp.orientations[j],
        )
    }

    /// Detect collision between a pair, reading live object state.
    ///
    /// The sequential path mutates objects as it goes, so the packed arrays go
    /// stale mid-loop; it reads through here instead. Only used for small
    /// worlds, where the layout advantage would not have paid anyway.
    fn detect_collision_pair_live(&self, i: usize, j: usize) -> Option<CollisionData> {
        let obj1 = &self.objects[i];
        let obj2 = &self.objects[j];

        self.detect_collision_with(
            i,
            j,
            (obj1.object.position.x, obj1.object.position.y, obj1.object.position.z),
            (obj2.object.position.x, obj2.object.position.y, obj2.object.position.z),
            obj1.shape.bounding_radius(),
            obj2.shape.bounding_radius(),
            Quaternion::from_euler(obj1.orientation.roll, obj1.orientation.pitch, obj1.orientation.yaw),
            Quaternion::from_euler(obj2.orientation.roll, obj2.orientation.pitch, obj2.orientation.yaw),
        )
    }

    /// Shared narrow-phase body for both detection entry points.
    #[allow(clippy::too_many_arguments)]
    fn detect_collision_with(
        &self,
        i: usize,
        j: usize,
        pos1: (f64, f64, f64),
        pos2: (f64, f64, f64),
        r1: f64,
        r2: f64,
        orientation1: Quaternion,
        orientation2: Quaternion,
    ) -> Option<CollisionData> {
        // Quick bounding-sphere rejection test
        let dx = pos2.0 - pos1.0;
        let dy = pos2.1 - pos1.1;
        let dz = pos2.2 - pos1.2;
        let distance_sq = dx * dx + dy * dy + dz * dz;

        if distance_sq > (r1 + r2).powi(2) {
            return None;
        }

        let obj1 = &self.objects[i];
        let obj2 = &self.objects[j];

        // Run GJK collision detection
        let gjk_result = gjk_collision_detection_ex(
            &obj1.shape, pos1, orientation1,
            &obj2.shape, pos2, orientation2
        );

        match gjk_result {
            GjkResult::NoCollision => None,
            GjkResult::SphereSphere { pos1, pos2, r1, r2 } => {
                // Sphere-sphere collision
                let dx = pos2.0 - pos1.0;
                let dy = pos2.1 - pos1.1;
                let dz = pos2.2 - pos1.2;
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                if distance < 1e-10 {
                    return None; // Overlapping centers, can't compute normal
                }

                let normal = (dx / distance, dy / distance, dz / distance);
                let penetration = r1 + r2 - distance;

                if penetration <= 0.0 {
                    return None;
                }

                let contact1 = (normal.0 * r1, normal.1 * r1, normal.2 * r1);
                let contact2 = (-normal.0 * r2, -normal.1 * r2, -normal.2 * r2);

                Some(CollisionData {
                    i, j, normal, penetration, contact1, contact2
                })
            }
            GjkResult::Collision(_) => {
                // Use EPA to get contact information
                if let Some(contact) = epa_contact_points_ex(
                    &obj1.shape, pos1, orientation1,
                    &obj2.shape, pos2, orientation2,
                    &gjk_result
                ) {
                    let penetration = contact.penetration;
                    if penetration <= 0.0 {
                        return None;
                    }

                    // Calculate contact points relative to object centers
                    let contact1 = (
                        contact.point1.0 - pos1.0,
                        contact.point1.1 - pos1.1,
                        contact.point1.2 - pos1.2,
                    );
                    let contact2 = (
                        contact.point2.0 - pos2.0,
                        contact.point2.1 - pos2.1,
                        contact.point2.2 - pos2.2,
                    );

                    Some(CollisionData {
                        i, j,
                        normal: contact.normal,
                        penetration,
                        contact1,
                        contact2,
                    })
                } else {
                    None
                }
            }
        }
    }

    /// Apply collision response for a detected collision
    fn apply_collision_response(&mut self, collision: &CollisionData, dt: f64) {
        let (first, second) = self.objects.split_at_mut(collision.j);
        let obj1 = &mut first[collision.i];
        let obj2 = &mut second[0];

        let normal = collision.normal;
        let r1 = collision.contact1;
        let r2 = collision.contact2;

        // Calculate point velocities (linear + angular contribution)
        let v1 = (
            obj1.object.velocity.x + obj1.angular_velocity.1 * r1.2 - obj1.angular_velocity.2 * r1.1,
            obj1.object.velocity.y + obj1.angular_velocity.2 * r1.0 - obj1.angular_velocity.0 * r1.2,
            obj1.object.velocity.z + obj1.angular_velocity.0 * r1.1 - obj1.angular_velocity.1 * r1.0,
        );
        let v2 = (
            obj2.object.velocity.x + obj2.angular_velocity.1 * r2.2 - obj2.angular_velocity.2 * r2.1,
            obj2.object.velocity.y + obj2.angular_velocity.2 * r2.0 - obj2.angular_velocity.0 * r2.2,
            obj2.object.velocity.z + obj2.angular_velocity.0 * r2.1 - obj2.angular_velocity.1 * r2.0,
        );

        // Relative velocity
        let vrel = (v2.0 - v1.0, v2.1 - v1.1, v2.2 - v1.2);
        let vrel_n = vrel.0 * normal.0 + vrel.1 * normal.1 + vrel.2 * normal.2;

        // Only respond if objects are approaching
        if vrel_n >= 0.0 {
            // Still need to resolve penetration
            self.resolve_penetration(collision);
            return;
        }

        // Calculate impulse magnitude
        // Use velocity-dependent restitution: reduce bounce for low-speed impacts (resting contacts)
        // This prevents jitter when objects are stacked
        let base_restitution = (obj1.get_restitution() + obj2.get_restitution()) / 2.0;
        const RESTITUTION_VELOCITY_THRESHOLD: f64 = 2.0; // m/s - below this, reduce restitution
        let restitution = if vrel_n.abs() < RESTITUTION_VELOCITY_THRESHOLD {
            // Scale restitution with square of approach speed for faster falloff
            base_restitution * (vrel_n.abs() / RESTITUTION_VELOCITY_THRESHOLD).powi(2)
        } else {
            base_restitution
        };

        let m1 = obj1.object.mass;
        let m2 = obj2.object.mass;

        // Check for static/infinite mass objects
        let m1_static = m1.is_infinite() || m1 <= 0.0;
        let m2_static = m2.is_infinite() || m2 <= 0.0;

        // For now, use simplified impulse (no angular contribution to denominator)
        // Use inverse mass = 0 for static objects
        let inv_mass1 = if m1_static { 0.0 } else { 1.0 / m1 };
        let inv_mass2 = if m2_static { 0.0 } else { 1.0 / m2 };
        let inv_mass_sum = inv_mass1 + inv_mass2;

        if inv_mass_sum < 1e-10 {
            return; // Both objects have infinite mass
        }

        let j = -(1.0 + restitution) * vrel_n / inv_mass_sum;

        // Apply linear impulse (skip for static objects)
        if !m1_static {
            let impulse_over_m1 = j / m1;
            obj1.object.velocity.x -= normal.0 * impulse_over_m1;
            obj1.object.velocity.y -= normal.1 * impulse_over_m1;
            obj1.object.velocity.z -= normal.2 * impulse_over_m1;
        }
        if !m2_static {
            let impulse_over_m2 = j / m2;
            obj2.object.velocity.x += normal.0 * impulse_over_m2;
            obj2.object.velocity.y += normal.1 * impulse_over_m2;
            obj2.object.velocity.z += normal.2 * impulse_over_m2;
        }

        // Apply angular impulse from normal force (simplified, skip for static objects)
        let torque_scale = 0.1; // Reduced angular response
        if !m1_static {
            let torque1 = cross_product(r1, (normal.0 * j, normal.1 * j, normal.2 * j));
            obj1.angular_velocity.0 -= torque1.0 * torque_scale;
            obj1.angular_velocity.1 -= torque1.1 * torque_scale;
            obj1.angular_velocity.2 -= torque1.2 * torque_scale;
        }
        if !m2_static {
            let torque2 = cross_product(r2, (normal.0 * j, normal.1 * j, normal.2 * j));
            obj2.angular_velocity.0 += torque2.0 * torque_scale;
            obj2.angular_velocity.1 += torque2.1 * torque_scale;
            obj2.angular_velocity.2 += torque2.2 * torque_scale;
        }

        // =====================================================================
        // FRICTION IMPULSE (Coulomb friction model)
        // =====================================================================
        // Calculate tangential velocity at contact point (perpendicular to normal)
        let tangent_vel = (
            vrel.0 - vrel_n * normal.0,
            vrel.1 - vrel_n * normal.1,
            vrel.2 - vrel_n * normal.2,
        );
        let tangent_speed = (tangent_vel.0.powi(2) + tangent_vel.1.powi(2) + tangent_vel.2.powi(2)).sqrt();

        if tangent_speed > 1e-6 {
            // `vrel = v2 - v1`, so `tangent_vel` points along obj2's motion
            // relative to obj1 - the opposite of obj1's slide direction.
            // Negating here makes `tangent` point along obj1's slide, so the
            // `-=` on obj1 and `+=` on obj2 below each oppose their own motion.
            //
            // Without this negation every application was inverted: friction
            // accelerated each body along the direction it was already sliding,
            // which grew the next tick's tangential impulse. A ball dropped
            // straight down onto a static surface spun up to 35 rad/s and was
            // flung off it, manufacturing mechanical energy from nothing.
            let tangent = (
                -tangent_vel.0 / tangent_speed,
                -tangent_vel.1 / tangent_speed,
                -tangent_vel.2 / tangent_speed,
            );

            // Get friction coefficient (geometric mean of both objects)
            // NOTE: Use obj1.get_friction() not obj1.object.get_friction()
            // because PhysicalObject3D.material is separate from ObjectIn3D.material
            let friction1 = obj1.get_friction();
            let friction2 = obj2.get_friction();
            let friction = (friction1 * friction2).sqrt();

            // Calculate friction impulse magnitude
            // Coulomb model: friction_impulse <= friction * normal_force
            // For resting contacts, use weight-based normal force since collision impulse is ~0
            let gravity = self.config.gravity;
            let gravity_mag = (gravity.0.powi(2) + gravity.1.powi(2) + gravity.2.powi(2)).sqrt();

            let weight_impulse = if gravity_mag > 1e-10 {
                let gravity_dir = (gravity.0 / gravity_mag, gravity.1 / gravity_mag, gravity.2 / gravity_mag);
                // How much is gravity pushing obj1 into obj2?
                // normal points from obj1 to obj2, gravity pushes obj1 in gravity_dir
                // alignment is positive when gravity pushes obj1 into obj2 (into the surface)
                let alignment = gravity_dir.0 * normal.0 + gravity_dir.1 * normal.1 + gravity_dir.2 * normal.2;

                if alignment > 0.0 {
                    // Weight component pressing into surface
                    // Use the lighter object's mass contribution to normal force
                    let effective_mass = if m2_static {
                        m1  // Ball on static ground: ball's weight creates normal force
                    } else if m1_static {
                        m2  // Object on static surface from above
                    } else {
                        (m1 * m2) / (m1 + m2)  // Reduced mass for two dynamic objects
                    };

                    effective_mass * gravity_mag * dt * alignment
                } else {
                    0.0  // Not pressing into surface (e.g., hitting from below)
                }
            } else {
                0.0  // No gravity
            };

            // Use the larger of collision impulse or weight-based impulse
            let max_friction_impulse = friction * j.abs().max(weight_impulse);

            // The impulse needed to stop tangential motion
            let friction_impulse_needed = tangent_speed / inv_mass_sum;

            // Apply the smaller of the two (clamped Coulomb friction)
            let friction_j = friction_impulse_needed.min(max_friction_impulse);

            // DEBUG: Log friction values (rate-limited to once per second at 240Hz)
            static DEBUG_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let count = DEBUG_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count % 240 == 0 && tangent_speed > 0.01 {
                log::info!(
                    "[FRICTION] µ1={:.2} µ2={:.2} µ={:.2} | j={:.4} weight={:.4} max_fric={:.4} | tan_spd={:.3} fric_j={:.4} | ω=({:.2},{:.2},{:.2})",
                    friction1, friction2, friction,
                    j.abs(), weight_impulse, max_friction_impulse,
                    tangent_speed, friction_j,
                    obj1.angular_velocity.0, obj1.angular_velocity.1, obj1.angular_velocity.2
                );
            }

            // Apply friction linear impulse (opposes tangential motion)
            // tangent points in direction of obj1's sliding relative to obj2
            // Friction opposes this: subtract from obj1, add to obj2
            if !m1_static {
                let friction_impulse_over_m1 = friction_j / m1;
                obj1.object.velocity.x -= tangent.0 * friction_impulse_over_m1;
                obj1.object.velocity.y -= tangent.1 * friction_impulse_over_m1;
                obj1.object.velocity.z -= tangent.2 * friction_impulse_over_m1;
            }
            if !m2_static {
                let friction_impulse_over_m2 = friction_j / m2;
                obj2.object.velocity.x += tangent.0 * friction_impulse_over_m2;
                obj2.object.velocity.y += tangent.1 * friction_impulse_over_m2;
                obj2.object.velocity.z += tangent.2 * friction_impulse_over_m2;
            }

            // Apply friction torque (creates rolling motion)
            // Angular impulse L = r × J_friction
            // Change in angular velocity: Δω = L / I (moment of inertia)
            // For solid sphere: I = 0.4 * m * r²
            if !m1_static {
                let r1_len = (r1.0.powi(2) + r1.1.powi(2) + r1.2.powi(2)).sqrt().max(0.1);
                let moment_of_inertia1 = 0.4 * m1 * r1_len * r1_len;
                let angular_impulse1 = cross_product(r1, (-tangent.0 * friction_j, -tangent.1 * friction_j, -tangent.2 * friction_j));

                // Δω = L / I
                obj1.angular_velocity.0 += angular_impulse1.0 / moment_of_inertia1;
                obj1.angular_velocity.1 += angular_impulse1.1 / moment_of_inertia1;
                obj1.angular_velocity.2 += angular_impulse1.2 / moment_of_inertia1;
            }
            if !m2_static {
                let r2_len = (r2.0.powi(2) + r2.1.powi(2) + r2.2.powi(2)).sqrt().max(0.1);
                let moment_of_inertia2 = 0.4 * m2 * r2_len * r2_len;
                let angular_impulse2 = cross_product(r2, (tangent.0 * friction_j, tangent.1 * friction_j, tangent.2 * friction_j));

                obj2.angular_velocity.0 += angular_impulse2.0 / moment_of_inertia2;
                obj2.angular_velocity.1 += angular_impulse2.1 / moment_of_inertia2;
                obj2.angular_velocity.2 += angular_impulse2.2 / moment_of_inertia2;
            }
        }

        // Rolling resistance - opposes angular velocity proportional to normal force
        // This makes rolling objects slow down naturally based on their material properties
        let rolling_resistance1 = obj1.object.get_rolling_resistance();
        let rolling_resistance2 = obj2.object.get_rolling_resistance();
        let rolling_resistance = (rolling_resistance1 * rolling_resistance2).sqrt();

        // Normal force: use the larger of collision impulse or weight-based impulse
        // This ensures rolling resistance works for resting contacts where j ≈ 0
        let gravity = self.config.gravity;
        let gravity_mag = (gravity.0.powi(2) + gravity.1.powi(2) + gravity.2.powi(2)).sqrt();
        let weight_normal_force = if gravity_mag > 1e-10 {
            let gravity_dir = (gravity.0 / gravity_mag, gravity.1 / gravity_mag, gravity.2 / gravity_mag);
            // For ground contact: gravity_dir=(0,-1,0), normal=(0,-1,0) → dot=+1
            // Positive alignment means gravity is pushing object against surface
            let alignment = gravity_dir.0 * normal.0 + gravity_dir.1 * normal.1 + gravity_dir.2 * normal.2;
            if alignment > 0.0 {
                let effective_mass = if m2_static { m1 } else if m1_static { m2 } else { (m1 * m2) / (m1 + m2) };
                effective_mass * gravity_mag * dt * alignment
            } else { 0.0 }
        } else { 0.0 };
        let normal_force = j.abs().max(weight_normal_force);

        // Apply rolling resistance torque (opposes angular velocity)
        if !m1_static && rolling_resistance > 0.0 {
            let omega1_mag = (obj1.angular_velocity.0.powi(2) + obj1.angular_velocity.1.powi(2) + obj1.angular_velocity.2.powi(2)).sqrt();
            if omega1_mag > 1e-6 {
                // Rolling resistance: angular_impulse = Crr * N * r
                // To convert to Δω, divide by moment of inertia
                // Approximate as solid sphere: I = 0.4 * m * r²
                let r1_len = (r1.0.powi(2) + r1.1.powi(2) + r1.2.powi(2)).sqrt().max(0.1);
                let angular_impulse = rolling_resistance * normal_force * r1_len;

                // Moment of inertia approximation (solid sphere: 0.4, hollow sphere: 0.67)
                let moment_of_inertia = 0.4 * m1 * r1_len * r1_len;
                let resistance_magnitude = angular_impulse / moment_of_inertia.max(0.01);

                // Clamp to not reverse angular velocity
                let max_reduction = omega1_mag * 0.5; // Don't reduce by more than half per collision
                let actual_resistance = resistance_magnitude.min(max_reduction);

                // Apply as impulse opposing angular velocity
                obj1.angular_velocity.0 -= (obj1.angular_velocity.0 / omega1_mag) * actual_resistance;
                obj1.angular_velocity.1 -= (obj1.angular_velocity.1 / omega1_mag) * actual_resistance;
                obj1.angular_velocity.2 -= (obj1.angular_velocity.2 / omega1_mag) * actual_resistance;
            }
        }

        if !m2_static && rolling_resistance > 0.0 {
            let omega2_mag = (obj2.angular_velocity.0.powi(2) + obj2.angular_velocity.1.powi(2) + obj2.angular_velocity.2.powi(2)).sqrt();
            if omega2_mag > 1e-6 {
                let r2_len = (r2.0.powi(2) + r2.1.powi(2) + r2.2.powi(2)).sqrt().max(0.1);
                let angular_impulse = rolling_resistance * normal_force * r2_len;

                // Moment of inertia approximation (solid sphere: 0.4)
                let moment_of_inertia = 0.4 * m2 * r2_len * r2_len;
                let resistance_magnitude = angular_impulse / moment_of_inertia.max(0.01);

                let max_reduction = omega2_mag * 0.5;
                let actual_resistance = resistance_magnitude.min(max_reduction);

                obj2.angular_velocity.0 -= (obj2.angular_velocity.0 / omega2_mag) * actual_resistance;
                obj2.angular_velocity.1 -= (obj2.angular_velocity.1 / omega2_mag) * actual_resistance;
                obj2.angular_velocity.2 -= (obj2.angular_velocity.2 / omega2_mag) * actual_resistance;
            }
        }

        // Resolve penetration
        self.resolve_penetration(collision);
    }

    /// Resolve penetration between two objects
    fn resolve_penetration(&mut self, collision: &CollisionData) {
        let (first, second) = self.objects.split_at_mut(collision.j);
        let obj1 = &mut first[collision.i];
        let obj2 = &mut second[0];

        let normal = collision.normal;
        let penetration = collision.penetration;

        let m1 = obj1.object.mass;
        let m2 = obj2.object.mass;

        // Check for static/infinite mass objects
        let m1_static = m1.is_infinite() || m1 <= 0.0;
        let m2_static = m2.is_infinite() || m2 <= 0.0;

        // Calculate correction ratio based on masses
        let (ratio1, ratio2) = if m1_static && m2_static {
            (0.0, 0.0) // Both immovable
        } else if m1_static {
            (0.0, 1.0) // Only obj1 is immovable
        } else if m2_static {
            (1.0, 0.0) // Only obj2 is immovable
        } else {
            let total = m1 + m2;
            (m2 / total, m1 / total) // Distribute by inverse mass ratio
        };

        // Apply position correction using logarithmic scaling
        // This gives strong correction for deep penetrations but very gentle for shallow ones
        // which helps prevent jitter while still resolving significant overlaps
        const PENETRATION_SLOP: f64 = 0.005; // Allow 5mm overlap before correcting
        const LOG_SCALE: f64 = 10.0; // Controls the curve steepness

        let excess = (penetration - PENETRATION_SLOP).max(0.0);
        // ln(1 + x*scale) / ln(1 + scale) gives 0 at x=0 and 1 at x=1
        // For small penetrations this is nearly zero, for large ones it approaches the penetration
        let correction = if excess > 0.0 {
            let normalized = (1.0 + excess * LOG_SCALE).ln() / (1.0 + LOG_SCALE).ln();
            excess * normalized * 0.5 // Apply 50% of the log-scaled correction
        } else {
            0.0
        };

        obj1.object.position.x -= normal.0 * correction * ratio1;
        obj1.object.position.y -= normal.1 * correction * ratio1;
        obj1.object.position.z -= normal.2 * correction * ratio1;

        obj2.object.position.x += normal.0 * correction * ratio2;
        obj2.object.position.y += normal.1 * correction * ratio2;
        obj2.object.position.z += normal.2 * correction * ratio2;
    }

    /// Get a snapshot of the current world state
    pub fn get_state(&self) -> WorldState {
        let objects: Vec<ObjectState> = self.objects.iter()
            .enumerate()
            .filter_map(|(idx, obj)| {
                self.index_to_id.get(&idx).map(|&id| {
                    // Convert Euler angles to quaternion
                    let quat = Quaternion::from_euler(
                        obj.orientation.roll,
                        obj.orientation.pitch,
                        obj.orientation.yaw
                    );

                    ObjectState {
                        id,
                        position: (
                            obj.object.position.x,
                            obj.object.position.y,
                            obj.object.position.z
                        ),
                        orientation: (quat.x, quat.y, quat.z, quat.w),
                        velocity: (
                            obj.object.velocity.x,
                            obj.object.velocity.y,
                            obj.object.velocity.z
                        ),
                        angular_velocity: obj.angular_velocity,
                        contacts: self
                            .active_contacts
                            .get(&id)
                            .cloned()
                            .unwrap_or_default(),
                    }
                })
            })
            .collect();

        WorldState {
            tick: self.tick,
            time: self.time,
            objects,
            #[cfg(feature = "constraints")]
            constraints: self.collect_constraint_states(),
        }
    }

    /// Collect constraint states for snapshots
    #[cfg(feature = "constraints")]
    fn collect_constraint_states(&self) -> Vec<super::state::ConstraintState> {
        use super::state::*;

        self.constraints
            .iter()
            .map(|(&id, constraint)| match constraint {
                WorldConstraint::Joint(j) => ConstraintState::Joint(JointState {
                    id,
                    object1: j.object1,
                    object2: j.object2,
                    anchor: j.anchor,
                    distance: j.distance,
                }),
                WorldConstraint::Spring(s) => ConstraintState::Spring(SpringState {
                    id,
                    object1: s.object1,
                    object2: s.object2,
                    anchor: s.anchor,
                    stiffness: s.stiffness,
                    rest_length: s.rest_length,
                }),
                WorldConstraint::Rope(r) => ConstraintState::Rope(RopeState {
                    id,
                    object1: r.object1,
                    object2: r.object2,
                    anchor: r.anchor,
                    max_length: r.max_length,
                }),
                WorldConstraint::RopeChain(chain) => {
                    // Get anchor position from first particle
                    let anchor_pos = &chain.particles[0].position;
                    ConstraintState::RopeChain(RopeChainState {
                        id,
                        anchor: (anchor_pos.x, anchor_pos.y, anchor_pos.z),
                        particle_positions: chain.get_particle_positions(),
                        segment_length: chain.segment_lengths.first().copied().unwrap_or(0.0),
                    })
                },
                WorldConstraint::Hinge(hinge) => ConstraintState::Hinge(HingeState {
                    id,
                    anchor: hinge.anchor,
                    axis: hinge.axis,
                    angle: hinge.angle,
                    angular_velocity: hinge.angular_velocity,
                    limits: match (hinge.angle_min, hinge.angle_max) {
                        (Some(min), Some(max)) => Some((min, max)),
                        _ => None,
                    },
                }),
            })
            .collect()
    }

    /// Get physics constants
    pub fn constants(&self) -> &PhysicsConstants {
        &self.config.constants
    }

    /// Get world configuration
    pub fn config(&self) -> &WorldConfig {
        &self.config
    }
}

/// Cross product of two 3D vectors
#[inline]
fn cross_product(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Shape3D;
    use crate::utils::PhysicsConstants;

    fn create_test_sphere(position: (f64, f64, f64), velocity: (f64, f64, f64)) -> PhysicalObject3D {
        PhysicalObject3D::new(
            1.0,  // mass
            velocity,
            position,
            Shape3D::Sphere(0.5),
            None,  // material
            (0.0, 0.0, 0.0),  // angular_velocity
            (0.0, 0.0, 0.0),  // orientation
            PhysicsConstants::default(),
        )
    }

    #[test]
    fn test_add_remove_objects() {
        let mut world = PhysicsWorld::default_world();

        let id1 = world.add_object(create_test_sphere((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)));
        let id2 = world.add_object(create_test_sphere((5.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        assert_eq!(world.object_count(), 2);
        assert!(world.get_object(id1).is_some());
        assert!(world.get_object(id2).is_some());

        assert!(world.remove_object(id1));
        assert_eq!(world.object_count(), 1);
        assert!(world.get_object(id1).is_none());
        assert!(world.get_object(id2).is_some());
    }

    #[test]
    fn test_gravity_integration() {
        let mut world = PhysicsWorld::new(
            WorldConfig::default().with_gravity(0.0, -10.0, 0.0)
        );

        let id = world.add_object(create_test_sphere((0.0, 10.0, 0.0), (0.0, 0.0, 0.0)));

        // Step multiple times
        for _ in 0..120 {  // 1 second at 120Hz
            world.step();
        }

        let state = world.get_state();
        let obj = state.get_object(id).unwrap();

        // Object should have fallen approximately 5m (1/2 * g * t^2)
        // Allow some tolerance for collision detection overhead
        assert!(obj.position.1 < 10.0, "Object should have fallen");
        assert!(obj.velocity.1 < 0.0, "Object should have downward velocity");
    }

    #[test]
    fn test_pause_resume() {
        let mut world = PhysicsWorld::default_world();
        let id = world.add_object(create_test_sphere((0.0, 10.0, 0.0), (0.0, 0.0, 0.0)));

        let initial_pos = world.get_state().get_position(id).unwrap();

        world.pause();
        for _ in 0..100 {
            world.step();
        }

        let paused_pos = world.get_state().get_position(id).unwrap();
        assert_eq!(initial_pos, paused_pos, "Position should not change while paused");

        world.resume();
        world.step();

        let resumed_pos = world.get_state().get_position(id).unwrap();
        assert_ne!(initial_pos, resumed_pos, "Position should change after resume");
    }

    #[test]
    fn test_apply_impulse() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Apply impulse of 10 kg*m/s in x direction to 1kg object
        world.apply_impulse(id, (10.0, 0.0, 0.0));

        let obj = world.get_object(id).unwrap();
        assert!((obj.object.velocity.x - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_force_directed() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Apply 10N force in the (1, 1, 0) direction (normalized)
        world.apply_force_directed(id, 10.0, (1.0, 1.0, 0.0));
        world.step();

        let obj = world.get_object(id).unwrap();
        // Force should be split equally between x and y (normalized direction)
        let expected = 10.0 / 2.0_f64.sqrt() * world.timestep();  // F/m * dt
        assert!((obj.object.velocity.x - expected).abs() < 0.001,
            "vx={}, expected={}", obj.object.velocity.x, expected);
        assert!((obj.object.velocity.y - expected).abs() < 0.001,
            "vy={}, expected={}", obj.object.velocity.y, expected);
    }

    #[test]
    fn test_force_toward() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((10.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Apply force toward origin
        world.apply_force_toward(id, (0.0, 0.0, 0.0), 10.0);
        world.step();

        let obj = world.get_object(id).unwrap();
        // Object should now be moving toward origin (negative x)
        assert!(obj.object.velocity.x < 0.0, "Should move toward target");
    }

    #[test]
    fn test_spring_force() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((5.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Apply spring force toward origin with k=10
        world.apply_spring_force(id, (0.0, 0.0, 0.0), 10.0);
        world.step();

        let obj = world.get_object(id).unwrap();
        // Spring should pull toward rest position (negative x velocity)
        assert!(obj.object.velocity.x < 0.0, "Spring should pull toward rest");
    }

    #[test]
    fn test_explosion() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id1 = world.add_object(create_test_sphere((2.0, 0.0, 0.0), (0.0, 0.0, 0.0)));
        let id2 = world.add_object(create_test_sphere((0.0, 2.0, 0.0), (0.0, 0.0, 0.0)));
        let id3 = world.add_object(create_test_sphere((100.0, 0.0, 0.0), (0.0, 0.0, 0.0)));  // Outside radius

        // Explosion at origin with radius 10
        world.apply_explosion((0.0, 0.0, 0.0), 100.0, 10.0);
        world.step();

        let obj1 = world.get_object(id1).unwrap();
        let obj2 = world.get_object(id2).unwrap();
        let obj3 = world.get_object(id3).unwrap();

        // Objects within radius should be pushed away
        assert!(obj1.object.velocity.x > 0.0, "Should be pushed in +x");
        assert!(obj2.object.velocity.y > 0.0, "Should be pushed in +y");
        // Object outside radius should be unaffected
        assert!((obj3.object.velocity.x).abs() < 0.001, "Should be unaffected");
    }

    #[test]
    fn test_drag() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)));

        let initial_speed = 10.0;

        // Apply drag for several steps
        for _ in 0..10 {
            world.apply_drag(id, 0.5);
            world.step();
        }

        let obj = world.get_object(id).unwrap();
        // Velocity should have decreased due to drag
        assert!(obj.object.velocity.x < initial_speed, "Drag should slow object");
        assert!(obj.object.velocity.x > 0.0, "Should still be moving forward");
    }

    // ==================== Continuous Force Tests ====================

    #[test]
    fn test_continuous_drag() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)));

        let initial_velocity = 10.0;

        // Add continuous drag (not one-shot)
        let drag_id = world.add_drag(id, 0.5);
        assert_eq!(world.continuous_force_count(), 1);

        // Step multiple times - drag should persist and continuously slow object
        for _ in 0..50 {
            world.step();
        }

        let obj = world.get_object(id).unwrap();
        // Object should have slowed (drag is continuous, not one-shot)
        assert!(obj.object.velocity.x < initial_velocity,
            "Continuous drag should slow object. Got velocity: {}", obj.object.velocity.x);

        // Remove the drag force
        assert!(world.remove_continuous_force(drag_id));
        assert_eq!(world.continuous_force_count(), 0);
    }

    #[test]
    fn test_continuous_drag_auto_removal() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)));

        // Add drag with low velocity threshold
        world.add_drag_with_threshold(id, 0.5, 0.5);
        assert_eq!(world.continuous_force_count(), 1);

        // Step until velocity drops below threshold
        for _ in 0..500 {
            world.step();
            if world.continuous_force_count() == 0 {
                break;
            }
        }

        // Drag should have been auto-removed
        assert_eq!(world.continuous_force_count(), 0, "Drag should auto-remove when velocity < threshold");
    }
    #[test]
    fn test_continuous_spring() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((5.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Add spring attached to origin with higher damping for faster settling
        let spring_id = world.add_damped_spring(id, (0.0, 0.0, 0.0), 10.0, 5.0);
        assert_eq!(world.continuous_force_count(), 1);

        // Step many times - should approach rest position
        for _ in 0..500 {
            world.step();
        }

        let obj = world.get_object(id).unwrap();
        // Should be near origin (within 0.5m after 500 steps with high damping)
        assert!(obj.object.position.x.abs() < 0.5, "Spring should pull object toward rest position");

        // Continue stepping and verify object stays near rest position (spring is working)
        for _ in 0..500 {
            world.step();
        }

        let obj = world.get_object(id).unwrap();
        // Should still be near origin and nearly at rest
        assert!(obj.object.position.x.abs() < 0.1, "Spring should keep object near rest position");
        assert!(obj.object.velocity.x.abs() < 0.5, "Object should have low velocity");

        // Clean up spring manually (auto-removal thresholds may be too strict for unit tests)
        world.remove_continuous_force(spring_id);
        assert_eq!(world.continuous_force_count(), 0, "Spring should be removed after manual removal");
    }

    #[test]
    fn test_continuous_attraction_with_duration() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((10.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Add attraction for 0.5 seconds (60 steps at 120Hz)
        world.add_attraction(id, (0.0, 0.0, 0.0), 100.0, Some(0.5));
        assert_eq!(world.continuous_force_count(), 1);

        // Step for less than duration
        for _ in 0..30 {
            world.step();
        }
        assert_eq!(world.continuous_force_count(), 1, "Attraction should still be active");

        let obj = world.get_object(id).unwrap();
        assert!(obj.object.velocity.x < 0.0, "Should be moving toward attractor");

        // Step past duration
        for _ in 0..60 {
            world.step();
        }
        assert_eq!(world.continuous_force_count(), 0, "Attraction should expire after duration");
    }

    #[test]
    fn test_constant_force() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Add constant force (like wind)
        let wind_id = world.add_constant_force(id, (10.0, 0.0, 0.0));

        // Step multiple times
        for _ in 0..60 {
            world.step();
        }

        let obj = world.get_object(id).unwrap();
        // Should be accelerating in x direction
        assert!(obj.object.velocity.x > 0.0, "Constant force should accelerate object");
        assert!(obj.object.position.x > 0.0, "Object should have moved");

        // Constant force does NOT auto-remove
        assert_eq!(world.continuous_force_count(), 1);

        // Must manually remove
        world.remove_continuous_force(wind_id);
        assert_eq!(world.continuous_force_count(), 0);
    }

    #[test]
    fn test_remove_forces_on_object() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id1 = world.add_object(create_test_sphere((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)));
        let id2 = world.add_object(create_test_sphere((5.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Add forces to both objects
        world.add_drag(id1, 0.5);
        world.add_constant_force(id1, (1.0, 0.0, 0.0));
        world.add_drag(id2, 0.5);

        assert_eq!(world.continuous_force_count(), 3);

        // Remove all forces on object 1
        world.remove_forces_on_object(id1);

        assert_eq!(world.continuous_force_count(), 1, "Should only have force on object 2");
    }

    #[test]
    fn test_force_removed_when_object_removed() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((0.0, 0.0, 0.0), (5.0, 0.0, 0.0)));

        world.add_drag(id, 0.5);
        assert_eq!(world.continuous_force_count(), 1);

        // Remove the object
        world.remove_object(id);

        // Step - should detect target doesn't exist and remove force
        world.step();

        assert_eq!(world.continuous_force_count(), 0, "Force should be removed when target is removed");
    }

    #[test]
    fn test_continuous_repulsion() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        // Place object at (5, 0, 0), repel from origin
        let id = world.add_object(create_test_sphere((5.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Add repulsion from origin with strength 100, no expiration
        let repel_id = world.add_repulsion(id, (0.0, 0.0, 0.0), 100.0, None);
        assert_eq!(world.continuous_force_count(), 1);

        // Step multiple times
        for _ in 0..60 {
            world.step();
        }

        let obj = world.get_object(id).unwrap();
        // Should be pushed away from origin (positive x velocity)
        assert!(obj.object.velocity.x > 0.0, "Repulsion should push object away from point. Got vx={}", obj.object.velocity.x);
        // Should have moved further from origin
        assert!(obj.object.position.x > 5.0, "Object should have moved away from origin. Got x={}", obj.object.position.x);

        // Force should still be active (no duration)
        assert_eq!(world.continuous_force_count(), 1);

        // Manual removal
        world.remove_continuous_force(repel_id);
        assert_eq!(world.continuous_force_count(), 0);
    }

    #[test]
    fn test_continuous_repulsion_with_duration() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((5.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Add repulsion for 0.5 seconds
        world.add_repulsion(id, (0.0, 0.0, 0.0), 100.0, Some(0.5));
        assert_eq!(world.continuous_force_count(), 1);

        // Step for less than duration (30 steps at 120Hz = 0.25s)
        for _ in 0..30 {
            world.step();
        }
        assert_eq!(world.continuous_force_count(), 1, "Repulsion should still be active");

        // Step past duration
        for _ in 0..60 {
            world.step();
        }
        assert_eq!(world.continuous_force_count(), 0, "Repulsion should expire after duration");
    }

    #[test]
    fn test_continuous_vortex() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        // Place object at (5, 0, 0), vortex centered at origin with Y axis
        let id = world.add_object(create_test_sphere((5.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Add vortex around Y axis - should create tangential force
        let vortex_id = world.add_vortex(id, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 100.0, None);
        assert_eq!(world.continuous_force_count(), 1);

        // Step multiple times
        for _ in 0..60 {
            world.step();
        }

        let obj = world.get_object(id).unwrap();
        // Object at (5, 0, 0) with Y axis vortex should get force in Z direction (tangential)
        // Cross product: (0,1,0) × (5,0,0) = (0*0 - 0*0, 0*5 - 1*0, 1*0 - 0*5) = (0, 0, 0)... wait
        // Actually: axis × r where r = obj - center = (5,0,0)
        // (0,1,0) × (5,0,0) = (1*0 - 0*0, 0*5 - 0*0, 0*0 - 1*5) = (0, 0, -5)
        // So force should be in -Z direction
        assert!(obj.object.velocity.z < 0.0, "Vortex should create tangential velocity. Got vz={}", obj.object.velocity.z);

        // Force should still be active
        assert_eq!(world.continuous_force_count(), 1);

        world.remove_continuous_force(vortex_id);
        assert_eq!(world.continuous_force_count(), 0);
    }

    #[test]
    fn test_continuous_vortex_with_duration() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        let id = world.add_object(create_test_sphere((5.0, 0.0, 0.0), (0.0, 0.0, 0.0)));

        // Add vortex for 0.25 seconds
        world.add_vortex(id, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 100.0, Some(0.25));
        assert_eq!(world.continuous_force_count(), 1);

        // Step past duration (60 steps at 120Hz = 0.5s)
        for _ in 0..60 {
            world.step();
        }

        assert_eq!(world.continuous_force_count(), 0, "Vortex should expire after duration");
    }

    #[test]
    fn test_continuous_buoyancy() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        // Place object below surface (y=0 surface, object at y=-5)
        let id = world.add_object(create_test_sphere((0.0, -5.0, 0.0), (0.0, 0.0, 0.0)));

        // Add buoyancy with water surface at y=0, water density
        let buoyancy_id = world.add_buoyancy(id, 0.0, 1000.0);
        assert_eq!(world.continuous_force_count(), 1);

        // Step multiple times
        for _ in 0..120 {
            world.step();
        }

        let obj = world.get_object(id).unwrap();
        // Object should have upward velocity from buoyancy
        assert!(obj.object.velocity.y > 0.0, "Buoyancy should push object up. Got vy={}", obj.object.velocity.y);
        // Object should have moved up
        assert!(obj.object.position.y > -5.0, "Object should have risen. Got y={}", obj.object.position.y);

        // Buoyancy should still be active while below surface
        assert_eq!(world.continuous_force_count(), 1);

        world.remove_continuous_force(buoyancy_id);
    }

    #[test]
    fn test_continuous_buoyancy_auto_removal() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        // Place object above surface (y=0 surface, object at y=5)
        let id = world.add_object(create_test_sphere((0.0, 5.0, 0.0), (0.0, 0.0, 0.0)));

        // Add buoyancy - object is already above surface
        world.add_buoyancy(id, 0.0, 1000.0);
        assert_eq!(world.continuous_force_count(), 1);

        // Step for more than removal_delay (1.0s default, so 120+ steps at 120Hz)
        for _ in 0..150 {
            world.step();
        }

        // Buoyancy should auto-remove since object has been above surface
        assert_eq!(world.continuous_force_count(), 0, "Buoyancy should auto-remove when above surface for extended time");
    }

    #[test]
    fn test_buoyancy_no_force_above_surface() {
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());
        // Object above surface
        let id = world.add_object(create_test_sphere((0.0, 5.0, 0.0), (0.0, 0.0, 0.0)));

        world.add_buoyancy(id, 0.0, 1000.0);

        // Step a few times (before auto-removal kicks in)
        for _ in 0..10 {
            world.step();
        }

        let obj = world.get_object(id).unwrap();
        // No force should be applied above surface
        assert!((obj.object.velocity.y).abs() < 0.001, "No buoyancy force above surface. Got vy={}", obj.object.velocity.y);
    }

    // ==================== Parallel Collision Detection Tests ====================

    #[test]
    fn test_parallel_collision_many_objects() {
        // Test with enough objects to trigger parallel path (>= 8)
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());

        // Create a grid of 16 spheres (4x4) that don't initially collide
        let mut ids = Vec::new();
        for x in 0..4 {
            for z in 0..4 {
                let pos = (x as f64 * 3.0, 0.0, z as f64 * 3.0);
                let id = world.add_object(create_test_sphere(pos, (0.0, 0.0, 0.0)));
                ids.push(id);
            }
        }

        assert_eq!(world.object_count(), 16);

        // Step should use parallel collision detection
        world.step();

        // All objects should still exist and have valid positions
        for id in &ids {
            let obj = world.get_object(*id);
            assert!(obj.is_some(), "Object should still exist after parallel collision step");
        }
    }

    #[test]
    fn test_parallel_collision_with_actual_collisions() {
        // Test parallel path with objects that will collide
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());

        // Create 10 spheres all moving toward center - they will collide
        let mut ids = Vec::new();
        for i in 0..10 {
            let angle = (i as f64) * std::f64::consts::PI * 2.0 / 10.0;
            let distance = 5.0;
            let pos = (angle.cos() * distance, 0.0, angle.sin() * distance);
            // Velocity toward center
            let vel = (-angle.cos() * 10.0, 0.0, -angle.sin() * 10.0);
            let id = world.add_object(create_test_sphere(pos, vel));
            ids.push(id);
        }

        assert_eq!(world.object_count(), 10);

        // Step multiple times - objects will collide near center
        for _ in 0..60 {
            world.step();
        }

        // After collisions, objects should have bounced and moved
        // Check that the simulation is stable (no NaN/Inf values)
        for id in &ids {
            let obj = world.get_object(*id).expect("Object should exist");
            assert!(!obj.object.position.x.is_nan(), "Position should not be NaN");
            assert!(!obj.object.velocity.x.is_nan(), "Velocity should not be NaN");
            assert!(obj.object.position.x.is_finite(), "Position should be finite");
            assert!(obj.object.velocity.x.is_finite(), "Velocity should be finite");
        }
    }

    #[test]
    fn test_parallel_collision_sphere_sphere() {
        // Test sphere-sphere collision through parallel path
        let mut world = PhysicsWorld::new(WorldConfig::zero_gravity());

        // Create 8 spheres to trigger parallel path, but only 2 will collide
        for i in 0..6 {
            // Non-colliding spheres spread out
            let pos = (i as f64 * 10.0 + 20.0, 0.0, 0.0);
            world.add_object(create_test_sphere(pos, (0.0, 0.0, 0.0)));
        }

        // Two spheres that will collide (overlapping, moving toward each other)
        // Spheres have radius 0.5, so at distance < 1.0 they overlap
        // Distance = 0.8, combined radii = 1.0, so penetration = 0.2
        let id1 = world.add_object(create_test_sphere((-0.4, 0.0, 0.0), (5.0, 0.0, 0.0)));
        let id2 = world.add_object(create_test_sphere((0.4, 0.0, 0.0), (-5.0, 0.0, 0.0)));

        assert_eq!(world.object_count(), 8);

        let v1_before = world.get_object(id1).unwrap().object.velocity.x;
        let v2_before = world.get_object(id2).unwrap().object.velocity.x;

        // Step once - collision should be resolved
        world.step();

        // After collision, they should have bounced apart
        let obj1 = world.get_object(id1).unwrap();
        let obj2 = world.get_object(id2).unwrap();

        // Velocities should have changed
        let v1_after = obj1.object.velocity.x;
        let v2_after = obj2.object.velocity.x;

        // At minimum, velocities should have changed from the collision
        assert!(v1_after != v1_before || v2_after != v2_before,
            "Collision should change velocities. Before: ({}, {}), After: ({}, {})",
            v1_before, v2_before, v1_after, v2_after);

        // Velocities should have reversed (approximately)
        assert!(obj1.object.velocity.x < 0.0, "Sphere 1 should bounce back. Got vx={}", obj1.object.velocity.x);
        assert!(obj2.object.velocity.x > 0.0, "Sphere 2 should bounce back. Got vx={}", obj2.object.velocity.x);
    }

    // ========================================================================
    // Contact physics
    //
    // These assert what a player sees rather than what the code does: a ball
    // dropped on a surface comes to rest on it and stays put, and a sliding
    // ball slows down. That catches an inverted friction impulse, an inverted
    // penetration correction, and a contact that never runs - none of which the
    // broad-phase equivalence tests below can see, because those compare pair
    // *selection* and these are all failures of response.
    //
    // Surfaces here sit well above y = 0 deliberately, so the real collider is
    // what responds rather than any implicit ground handling.
    // ========================================================================

    fn static_box(
        position: (f64, f64, f64),
        (w, h, d): (f64, f64, f64),
    ) -> PhysicalObject3D {
        PhysicalObject3D::new(
            f64::INFINITY,
            (0.0, 0.0, 0.0),
            position,
            Shape3D::Cuboid(w, h, d),
            None,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            PhysicsConstants::default(),
        )
    }

    fn contact_world() -> PhysicsWorld {
        PhysicsWorld::new(
            WorldConfig::default()
                .with_frequency(240.0)
                .with_gravity(0.0, -15.0, 0.0),
        )
    }

    #[test]
    fn test_ball_dropped_straight_down_comes_to_rest_on_surface() {
        const SURFACE_TOP: f64 = 5.0;
        const RADIUS: f64 = 0.5;

        let mut world = contact_world();
        world.add_object(static_box((0.0, SURFACE_TOP - 0.25, 0.0), (20.0, 0.5, 20.0)));
        world.add_object(sphere_of(RADIUS, (0.0, 8.0, 0.0)));

        for _ in 0..1440 {
            world.step();
        }

        let ball = &world.objects[1];
        let (p, v, w) = (&ball.object.position, &ball.object.velocity, ball.angular_velocity);
        let expected_y = SURFACE_TOP + RADIUS;

        assert!(
            (p.y - expected_y).abs() < 1e-3,
            "ball should rest on the surface at y={expected_y}, got {} \
             (below means it sank through, above means it never settled)",
            p.y,
        );
        assert!(
            v.x.abs() < 0.05 && v.z.abs() < 0.05,
            "ball fell straight down with no lateral input; it must not acquire \
             horizontal velocity from contact. got vx={}, vz={}",
            v.x, v.z,
        );
        let spin = (w.0 * w.0 + w.1 * w.1 + w.2 * w.2).sqrt();
        assert!(
            spin < 0.5,
            "ball fell straight down; contact must not spin it up. got |w|={spin}",
        );
    }

    #[test]
    fn test_sliding_ball_loses_tangential_speed() {
        const SURFACE_TOP: f64 = 5.0;
        const START_VX: f64 = 5.0;

        let mut world = contact_world();
        world.add_object(static_box((0.0, SURFACE_TOP - 0.25, 0.0), (400.0, 0.5, 400.0)));

        let mut ball = sphere_of(0.5, (0.0, SURFACE_TOP + 0.5, 0.0));
        ball.object.velocity.x = START_VX;
        world.add_object(ball);

        for _ in 0..240 {
            world.step();
        }

        let v = &world.objects[1].object.velocity;
        assert!(
            v.x < START_VX,
            "friction must oppose sliding, never drive it. vx went {START_VX} -> {} \
             (an increase means the tangential impulse is applied along the slide \
             instead of against it, which manufactures energy every contact)",
            v.x,
        );
        assert!(
            v.x > -START_VX,
            "friction must not reverse the slide outright, got vx={}",
            v.x,
        );
    }

    #[test]
    fn test_sliding_ball_starts_rolling() {
        // Rolling is not scripted anywhere - it has to emerge from a tangential
        // friction impulse applied at the contact point, which torques the body.
        // A ball given pure lateral velocity must therefore spin up, and spin in
        // the direction that *reduces* slip. If contact applies no torque (or
        // the wrong one) this stays at zero and the ball skids forever.
        const SURFACE_TOP: f64 = 5.0;
        const RADIUS: f64 = 0.5;

        let mut world = contact_world();
        world.add_object(static_box((0.0, SURFACE_TOP - 0.25, 0.0), (400.0, 0.5, 400.0)));

        let mut ball = sphere_of(RADIUS, (0.0, SURFACE_TOP + RADIUS, 0.0));
        ball.object.velocity.x = 5.0;
        world.add_object(ball);

        for _ in 0..240 {
            world.step();
        }

        let b = &world.objects[1];
        let vx = b.object.velocity.x;
        let wz = b.angular_velocity.2;

        assert!(
            wz.abs() > 0.5,
            "contact must torque a sliding ball into rotation, got wz={wz}",
        );
        // Rolling without slipping about +x travel is wz = -vx / r.
        let slip = vx + wz * RADIUS;
        assert!(
            slip.abs() < vx.abs(),
            "friction must reduce slip, not increase it. vx={vx}, wz={wz}, slip={slip}",
        );
    }

    // ========================================================================
    // Broad phase
    //
    // The grid is only an optimization if it selects exactly the pairs the
    // all-pairs loop would have tested to a positive result. A broad phase that
    // silently misses contacts produces objects that sink through each other,
    // and it does so intermittently, which is close to undebuggable. These
    // tests compare against brute force directly.
    // ========================================================================

    /// Deterministic pseudo-random source; avoids a dependency and keeps
    /// failures reproducible.
    struct Lcg(u64);

    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }

        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + self.next_f64() * (hi - lo)
        }
    }

    fn sphere_of(radius: f64, position: (f64, f64, f64)) -> PhysicalObject3D {
        PhysicalObject3D::new(
            1.0,
            (0.0, 0.0, 0.0),
            position,
            Shape3D::Sphere(radius),
            None,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            PhysicsConstants::default(),
        )
    }

    /// Every pair, tested through the same narrow phase the grid path uses.
    fn brute_force_collisions(world: &PhysicsWorld) -> Vec<CollisionData> {
        let n = world.objects.len();
        let mut out = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if let Some(c) = world.detect_collision_pair(i, j) {
                    out.push(c);
                }
            }
        }
        out
    }

    fn assert_broad_phase_matches_brute_force(world: &mut PhysicsWorld, label: &str) {
        world.broad_phase.rebuild(&world.objects);

        let expected = brute_force_collisions(world);
        let actual = world.detect_collisions_parallel();

        assert_eq!(
            expected.len(),
            actual.len(),
            "{label}: broad phase found {} collisions, brute force found {}",
            actual.len(),
            expected.len(),
        );
        assert_eq!(
            expected, actual,
            "{label}: broad phase and brute force disagree in content or order",
        );
        assert!(
            !expected.is_empty(),
            "{label}: scene produced no collisions at all - the test proves nothing",
        );
    }

    #[test]
    fn test_broad_phase_matches_brute_force_uniform_sizes() {
        let mut rng = Lcg(0x1234_5678);
        let mut world = PhysicsWorld::default_world();

        // Dense enough that plenty of pairs genuinely overlap.
        for _ in 0..400 {
            world.add_object(sphere_of(
                0.5,
                (rng.range(-10.0, 10.0), rng.range(-10.0, 10.0), rng.range(-10.0, 10.0)),
            ));
        }
        assert_broad_phase_matches_brute_force(&mut world, "uniform");
    }

    #[test]
    fn test_broad_phase_matches_brute_force_mixed_sizes() {
        let mut rng = Lcg(0xdead_beef);
        let mut world = PhysicsWorld::default_world();

        // Radii spanning an order of magnitude: the grid is sized from the
        // largest gridded object, so this checks nothing escapes its cells.
        for _ in 0..300 {
            world.add_object(sphere_of(
                rng.range(0.2, 2.0),
                (rng.range(-12.0, 12.0), rng.range(-12.0, 12.0), rng.range(-12.0, 12.0)),
            ));
        }
        assert_broad_phase_matches_brute_force(&mut world, "mixed sizes");
    }

    #[test]
    fn test_broad_phase_matches_brute_force_bimodal_sizes() {
        // Two distinct size populations, both inside the grid. This is the case
        // that exercises cross-level pairing: a small object must find a large
        // one by sweeping the coarser level, and each such pair must be emitted
        // exactly once, by the finer object only.
        let mut rng = Lcg(0xb1_40da1);
        let mut world = PhysicsWorld::default_world();

        for _ in 0..400 {
            let radius = if rng.next_f64() < 0.85 { 0.25 } else { 1.5 };
            world.add_object(sphere_of(
                radius,
                (rng.range(-8.0, 8.0), rng.range(-8.0, 8.0), rng.range(-8.0, 8.0)),
            ));
        }

        assert!(
            world.broad_phase.levels.len() > 1
                || {
                    world.broad_phase.rebuild(&world.objects);
                    world.broad_phase.levels.len() > 1
                },
            "scene should span multiple grid levels or it proves nothing",
        );
        assert_broad_phase_matches_brute_force(&mut world, "bimodal");
    }

    #[test]
    fn test_broad_phase_matches_brute_force_with_oversized_objects() {
        let mut rng = Lcg(0x0bad_f00d);
        let mut world = PhysicsWorld::default_world();

        // A ground plane and two walls: exactly the case that would collapse a
        // naive grid into a single bucket, and the reason for the oversized path.
        world.add_object(PhysicalObject3D::new(
            f64::INFINITY,
            (0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            Shape3D::Cuboid(200.0, 2.0, 200.0),
            None,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            PhysicsConstants::default(),
        ));
        world.add_object(PhysicalObject3D::new(
            f64::INFINITY,
            (0.0, 0.0, 0.0),
            (-15.0, 5.0, 0.0),
            Shape3D::Cuboid(2.0, 60.0, 120.0),
            None,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            PhysicsConstants::default(),
        ));

        for _ in 0..250 {
            world.add_object(sphere_of(
                rng.range(0.3, 0.8),
                (rng.range(-14.0, 14.0), rng.range(-0.5, 8.0), rng.range(-14.0, 14.0)),
            ));
        }
        assert_broad_phase_matches_brute_force(&mut world, "oversized");
    }

    #[test]
    fn test_broad_phase_handles_coincident_and_extreme_positions() {
        let mut world = PhysicsWorld::default_world();

        // Coincident centres, a far-flung outlier, and a cluster - the cases
        // where cell arithmetic tends to go wrong.
        world.add_object(sphere_of(1.0, (0.0, 0.0, 0.0)));
        world.add_object(sphere_of(1.0, (0.0, 0.0, 0.0)));
        world.add_object(sphere_of(1.0, (0.5, 0.0, 0.0)));
        world.add_object(sphere_of(1.0, (1e12, 0.0, 0.0)));
        world.add_object(sphere_of(1.0, (-1e12, 0.0, 0.0)));
        for i in 0..20 {
            world.add_object(sphere_of(1.0, (i as f64 * 0.3, 0.0, 0.0)));
        }

        world.broad_phase.rebuild(&world.objects);
        let expected = brute_force_collisions(&world);
        let actual = world.detect_collisions_parallel();
        assert_eq!(expected, actual, "degenerate placement diverged from brute force");
    }

    #[test]
    fn test_broad_phase_actually_culls() {
        // The point of the grid is to test far fewer pairs. If it ever stops
        // culling, the perf work has silently regressed while staying correct.
        let mut world = PhysicsWorld::default_world();
        for i in 0..1000 {
            let f = i as f64;
            world.add_object(sphere_of(0.5, (f * 10.0, 0.0, 0.0)));
        }
        world.broad_phase.rebuild(&world.objects);

        let all_pairs = 1000 * 999 / 2;
        let candidates = world.broad_phase.pairs.len();
        assert!(
            candidates * 100 < all_pairs,
            "broad phase kept {candidates} of {all_pairs} pairs; expected a >100x reduction",
        );
    }

    // ========================================================================
    // Determinism
    //
    // Narrow-phase collision detection runs on a rayon pool, and the continuous
    // forces live in a HashMap. Both are places where iteration or completion
    // order can leak into float accumulation. These tests pin the property that
    // repeated runs of an identical setup stay bit-identical - not merely close
    // - which is what replays, lockstep networking and reproducible bug reports
    // all depend on.
    // ========================================================================

    /// Build a reasonably large world, spread out so objects fall freely rather
    /// than colliding.
    fn build_parallel_sized_world() -> PhysicsWorld {
        let count = PhysicsWorld::LARGE_WORLD;
        let mut world = PhysicsWorld::default_world();

        for i in 0..count {
            let f = i as f64;
            world.add_object(create_test_sphere(
                (f * 5.0, 50.0 + f * 0.25, f * 3.0),
                (f * 0.01, 0.0, -f * 0.02),
            ));
        }
        world
    }

    fn fingerprint(world: &PhysicsWorld) -> Vec<(u64, u64, u64, u64, u64, u64)> {
        // Compare raw bit patterns: "approximately equal" would hide exactly the
        // last-bit divergence that breaks replays and lockstep networking.
        world
            .objects
            .iter()
            .map(|o| {
                (
                    o.object.position.x.to_bits(),
                    o.object.position.y.to_bits(),
                    o.object.position.z.to_bits(),
                    o.object.velocity.x.to_bits(),
                    o.object.velocity.y.to_bits(),
                    o.object.velocity.z.to_bits(),
                )
            })
            .collect()
    }

    /// Timing harness for the per-object stages. Run with:
    /// `cargo test --release --lib bench_step_cost -- --ignored --nocapture`
    #[test]
    #[ignore = "timing measurement, not a pass/fail assertion"]
    fn bench_step_cost() {
        // Sparse is a broad phase's best case; dense is its worst, because most
        // candidate pairs are genuinely near and reach the narrow phase. Report
        // both, or the speedup is measured on the flattering scene only.
        // Three shapes of scene, because a spatial index can be tuned to look
        // good on any one of them:
        //   sparse  - objects strung far apart. Note x == z exactly, which is a
        //             worst case for any locality scheme keyed on interleaved
        //             coordinate bits.
        //   dense   - a packed lattice; nearly every probe finds neighbours.
        //   cluster - pseudo-random inside a compact box. Closest to a real
        //             scene, and the case locality optimizations should win on.
        //   mixed   - varying radii packed tightly. Cell size follows the
        //             largest object, so many small ones share a cell; this is
        //             the only scene where per-cell work can be amortized.
        //   bimodal - lots of small debris plus a minority of large crates. The
        //             realistic heterogeneous case, and the one a size-binned
        //             grid should actually win on.
        let scenes: [(&str, u8); 5] = [
            ("sparse ", 0), ("dense  ", 1), ("cluster", 2), ("mixed  ", 3), ("bimodal", 4),
        ];

        for (label, kind) in scenes {
        for &count in &[64usize, 256, 1024, 2048, 4096] {
            let mut world = PhysicsWorld::default_world();
            let side = (count as f64).cbrt().ceil() as usize;
            let mut rng = Lcg(0x5eed_1234);
            let box_side = 4.0 * (count as f64).cbrt();

            for i in 0..count {
                let f = i as f64;
                let position = match kind {
                    // Lattice at ~1.2 diameters: every object has neighbours in
                    // range, so the grid cannot cull much.
                    1 => {
                        let (ix, iy, iz) = (i % side, (i / side) % side, i / (side * side));
                        (ix as f64 * 1.2, 20.0 + iy as f64 * 1.2, iz as f64 * 1.2)
                    }
                    2 => (
                        rng.range(-box_side, box_side),
                        20.0 + rng.range(0.0, box_side),
                        rng.range(-box_side, box_side),
                    ),
                    3 => {
                        let s = 0.75 * (count as f64).cbrt();
                        (rng.range(-s, s), 20.0 + rng.range(0.0, s), rng.range(-s, s))
                    }
                    4 => {
                        let s = 0.95 * (count as f64).cbrt();
                        (rng.range(-s, s), 20.0 + rng.range(0.0, s), rng.range(-s, s))
                    }
                    _ => (f * 50.0, 50.0 + f * 0.25, f * 50.0),
                };

                if kind == 3 {
                    // Radii within 4x of the median, so none are classified
                    // oversized and all go through the grid.
                    world.add_object(sphere_of(rng.range(0.3, 1.2), position));
                } else if kind == 4 {
                    // 90% debris, 10% crates - a 6x size split.
                    let radius = if rng.next_f64() < 0.9 { 0.25 } else { 1.5 };
                    world.add_object(sphere_of(radius, position));
                } else {
                    world.add_object(create_test_sphere(position, (f * 0.01, 0.0, -f * 0.02)));
                }
            }

            for _ in 0..50 {
                world.step();
            }
            // Warmup must not pollute the breakdown; it is divided by the
            // measured iteration count only.
            world.profile = PhaseTimings::default();
            world.broad_phase.profile = BroadPhaseTimings::default();

            // Narrow phase is O(n^2); keep total runtime bounded as n grows.
            let iterations = (500 * 256 / count).max(20);
            let started = std::time::Instant::now();
            for _ in 0..iterations {
                world.step();
            }
            let elapsed = started.elapsed();

            eprintln!(
                "{label}  n={count:5}  {:9.1} us/step  {:8.1} ns/object/step  {:8} candidate pairs",
                elapsed.as_secs_f64() * 1e6 / iterations as f64,
                elapsed.as_secs_f64() * 1e9 / (iterations * count) as f64,
                world.broad_phase.pairs.len(),
            );

            if count == 4096 {
                let p = world.profile;
                let per = |d: std::time::Duration| d.as_secs_f64() * 1e6 / iterations as f64;
                eprintln!(
                    "        breakdown us/step: grav {:.1}  contforce {:.1}  pending {:.1}  \
                     broad {:.1}  narrow {:.1}  response {:.1}  constr {:.1}  damp {:.1}  \
                     integ {:.1}  tunnel {:.1}",
                    per(p.gravity), per(p.continuous_forces), per(p.pending_forces),
                    per(p.broad_rebuild), per(p.narrow_phase), per(p.response),
                    per(p.constraints), per(p.damping), per(p.integrate), per(p.tunneling),
                );
                let b = world.broad_phase.profile;
                // Objects per occupied cell decides whether hoisting work to the
                // cell level can pay at all: at 1.0 there is nothing to amortize.
                let occupied: u32 = world
                    .broad_phase
                    .levels
                    .iter()
                    .flat_map(|l| l.occupied.iter())
                    .map(|w| w.count_ones())
                    .sum();
                let gridded: usize =
                    world.broad_phase.levels.iter().map(|l| l.items.len()).sum();
                eprintln!(
                    "        broad phase us/step: fill {:.1}  median {:.1}  scatter {:.1}  \
                     collect {:.1}  sort+dedup {:.1}   [{} levels, {:.2} objects/occupied cell]",
                    per(b.fill), per(b.median), per(b.scatter), per(b.collect), per(b.sort_dedup),
                    world.broad_phase.levels.len(),
                    gridded as f64 / occupied.max(1) as f64,
                );
            }
        }
        }
    }

    #[test]
    fn test_parallel_step_is_bit_identical_across_runs() {
        let mut a = build_parallel_sized_world();
        let mut b = build_parallel_sized_world();
        assert!(
            a.objects.len() >= PhysicsWorld::LARGE_WORLD,
            "test must exercise the parallel path",
        );

        for _ in 0..120 {
            a.step();
            b.step();
        }

        assert_eq!(
            fingerprint(&a),
            fingerprint(&b),
            "identical setups diverged; the parallel path is not deterministic",
        );
    }

    #[test]
    fn test_continuous_force_application_order_is_stable() {
        // Several forces on one object: the sum is order-dependent because float
        // addition is not associative, and they live in a HashMap.
        fn run() -> (u64, u64, u64) {
            let mut world = PhysicsWorld::default_world();
            let id = world.add_object(create_test_sphere((0.0, 100.0, 0.0), (0.0, 0.0, 0.0)));

            world.add_constant_force(id, (0.1, 0.0, 0.0));
            world.add_constant_force(id, (0.02, 0.0, 0.0));
            world.add_constant_force(id, (0.003, 0.0, 0.0));
            world.add_constant_force(id, (0.0004, 0.0, 0.0));
            world.add_constant_force(id, (0.00005, 0.0, 0.0));

            for _ in 0..200 {
                world.step();
            }
            let v = &world.objects[0].object.velocity;
            (v.x.to_bits(), v.y.to_bits(), v.z.to_bits())
        }

        let first = run();
        for attempt in 1..8 {
            assert_eq!(
                first,
                run(),
                "continuous-force accumulation diverged on attempt {attempt}; \
                 HashMap iteration order is leaking into the result",
            );
        }
    }

    #[test]
    fn test_pending_forces_hit_the_right_object_when_parallel() {
        // `par_iter_mut().enumerate()` must pair each object with its own index;
        // an off-by-one here would silently push the wrong body.
        // Compare against an identical unpushed world rather than against the
        // pre-step state: damping and gravity move every object each tick, so
        // "changed at all" is not evidence of anything.
        let mut pushed = build_parallel_sized_world();
        let mut control = build_parallel_sized_world();

        let target_idx = PhysicsWorld::LARGE_WORLD / 2 + 7;
        let target_id = *pushed.index_to_id.get(&target_idx).expect("index should map");

        pushed.apply_force(target_id, (1000.0, 0.0, 0.0));
        pushed.step();
        control.step();

        let (a, b) = (fingerprint(&pushed), fingerprint(&control));
        for (i, (p, c)) in a.iter().zip(b.iter()).enumerate() {
            if i == target_idx {
                assert_ne!(p, c, "the targeted object should have moved differently");
            } else {
                assert_eq!(
                    p, c,
                    "object {i} was affected, but only object {target_idx} was pushed",
                );
            }
        }
    }
}
