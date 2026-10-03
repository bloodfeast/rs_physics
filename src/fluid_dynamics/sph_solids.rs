//! Solids the liquid meets: capsules and yawed boxes a caller places every frame, binned
//! into the SPH hash once a step and met by a short swept ray in `integrate`.
//!
//! A child module of `sph` so the binning can read the step's grid (the particle cells,
//! the bucket table) without widening anything public. The design, the one-way rule and
//! the cost model are in the parent module's "Solids" section.

use std::time::Duration;

use super::{bucket, cell_of, row_hash, SphFluid, CFL_FRACTION};

/// A capsule as the contact reads it: the caller's numbers plus the axis terms every ray
/// test needs, folded once when it is pushed.
#[derive(Debug, Clone, Copy)]
pub(super) struct Capsule {
    a: [f64; 3],
    b: [f64; 3],
    velocity_a: [f64; 3],
    velocity_b: [f64; 3],
    radius: f64,
    /// `b - a`.
    axis: [f64; 3],
    /// `|b - a|^2`; zero for a sphere.
    axis_len2: f64,
    /// `1 / |b - a|^2`, or zero for a sphere, so the closest-point parameter clamps to
    /// the `a` end without a branch on the degenerate case.
    inv_axis_len2: f64,
}

/// A box yawed about +y, with its yaw's sine and cosine taken once when it is pushed.
#[derive(Debug, Clone, Copy)]
pub(super) struct YawBox {
    centre: [f64; 3],
    half: [f64; 3],
    sin: f64,
    cos: f64,
    velocity: [f64; 3],
}

/// The solids a [`SphFluid`] meets in one step: capsules and yawed boxes, each with the
/// velocity of its surface, handed to [`SphFluid::step_with_solids`].
///
/// Presentation only and one way: the solids push the liquid and the liquid never pushes
/// a solid back, so nothing here can reach a lockstep simulation. The caller clears and
/// refills the set each frame from wherever its actors, corpses and props are; the step
/// reads it and never writes it. Each solid is given at its pose at the end of the
/// substep, with the velocity that carried it there.
///
/// Storage is two `Vec`s of whole records (a capsule is 144 bytes, a box 88), because the
/// contact reads every field of the one solid it is testing; cleared rather than freed,
/// so a set refilled every frame stops allocating once it has held its largest frame.
///
/// # Examples
///
/// ```
/// use rs_physics::fluid_dynamics::SphSolids;
///
/// let mut solids = SphSolids::new();
/// // A shin, swinging forward: the knee at 1 m/s, the ankle at 2.
/// assert!(solids.push_capsule(
///     [0.0, 0.5, 0.0],
///     [0.0, 0.1, 0.0],
///     0.06,
///     [1.0, 0.0, 0.0],
///     [2.0, 0.0, 0.0],
/// ));
/// // A crate, turned a quarter, at rest.
/// assert!(solids.push_box([1.0, 0.25, 0.0], [0.25; 3], std::f64::consts::FRAC_PI_2, [0.0; 3]));
/// assert_eq!((solids.capsule_count(), solids.box_count()), (1, 1));
/// solids.clear();
/// assert!(solids.is_empty());
/// ```
#[derive(Debug, Clone, Default)]
pub struct SphSolids {
    capsules: Vec<Capsule>,
    boxes: Vec<YawBox>,
}

impl SphSolids {
    /// An empty set.
    ///
    /// # Returns
    ///
    /// A set holding no solids and owning no memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::SphSolids;
    ///
    /// assert!(SphSolids::new().is_empty());
    /// ```
    pub fn new() -> SphSolids {
        SphSolids::default()
    }

    /// An empty set with room for `capsules` capsules and `boxes` boxes before it
    /// allocates.
    ///
    /// # Arguments
    ///
    /// * `capsules` - capsules to reserve for.
    /// * `boxes` - boxes to reserve for.
    ///
    /// # Returns
    ///
    /// An empty set.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::SphSolids;
    ///
    /// let solids = SphSolids::with_capacity(64, 8);
    /// assert!(solids.is_empty());
    /// ```
    pub fn with_capacity(capsules: usize, boxes: usize) -> SphSolids {
        SphSolids {
            capsules: Vec::with_capacity(capsules),
            boxes: Vec::with_capacity(boxes),
        }
    }

    /// Remove every solid, keeping the memory for the next frame's.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::SphSolids;
    ///
    /// let mut solids = SphSolids::new();
    /// solids.push_box([0.0; 3], [0.1; 3], 0.0, [0.0; 3]);
    /// solids.clear();
    /// assert!(solids.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.capsules.clear();
        self.boxes.clear();
    }

    /// How many solids the set holds, capsules and boxes together.
    ///
    /// # Returns
    ///
    /// [`Self::capsule_count`] plus [`Self::box_count`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::SphSolids;
    ///
    /// let mut solids = SphSolids::new();
    /// solids.push_box([0.0; 3], [0.1; 3], 0.0, [0.0; 3]);
    /// assert_eq!(solids.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.capsules.len() + self.boxes.len()
    }

    /// Whether the set holds no solid.
    ///
    /// # Returns
    ///
    /// `true` when [`Self::len`] is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::SphSolids;
    ///
    /// assert!(SphSolids::new().is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.capsules.is_empty() && self.boxes.is_empty()
    }

    /// How many capsules the set holds.
    ///
    /// # Returns
    ///
    /// The capsule count.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::SphSolids;
    ///
    /// let mut solids = SphSolids::new();
    /// solids.push_capsule([0.0; 3], [0.0, 1.0, 0.0], 0.1, [0.0; 3], [0.0; 3]);
    /// assert_eq!(solids.capsule_count(), 1);
    /// ```
    pub fn capsule_count(&self) -> usize {
        self.capsules.len()
    }

    /// How many boxes the set holds.
    ///
    /// # Returns
    ///
    /// The box count.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::SphSolids;
    ///
    /// let mut solids = SphSolids::new();
    /// solids.push_box([0.0; 3], [0.1; 3], 0.0, [0.0; 3]);
    /// assert_eq!(solids.box_count(), 1);
    /// ```
    pub fn box_count(&self) -> usize {
        self.boxes.len()
    }

    /// Add a capsule: every point within `radius` of the segment from `a` to `b`. An
    /// actor's limb, a boot, a blade.
    ///
    /// The surface velocity is given at each end and blended along the axis, so a limb
    /// swinging about one end carries liquid faster at its tip than at its root; give the
    /// same velocity twice for a capsule that only translates. `a == b` is a sphere.
    ///
    /// # Arguments
    ///
    /// * `a`, `b` - the axis end points, metres, where they are at the END of the
    ///   substep the set is stepped with: the capsule stood `velocity * dt` further back
    ///   when it began.
    /// * `radius` - metres, positive.
    /// * `velocity_a`, `velocity_b` - the surface velocity at each end, m/s: what
    ///   carried each end to `a` and `b`. The contact casts each particle relative to it,
    ///   so a drop moving with the surface stays on it.
    ///
    /// # Returns
    ///
    /// `false`, and nothing added, if any number is not finite, the radius is not
    /// positive, or the set already holds `u32::MAX` solids.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::SphSolids;
    ///
    /// let mut solids = SphSolids::new();
    /// assert!(solids.push_capsule([0.0; 3], [0.0, 1.0, 0.0], 0.1, [0.0; 3], [0.0; 3]));
    /// assert!(!solids.push_capsule([0.0; 3], [0.0, 1.0, 0.0], 0.0, [0.0; 3], [0.0; 3]));
    /// assert!(!solids.push_capsule([f64::NAN; 3], [0.0; 3], 0.1, [0.0; 3], [0.0; 3]));
    /// ```
    pub fn push_capsule(
        &mut self,
        a: [f64; 3],
        b: [f64; 3],
        radius: f64,
        velocity_a: [f64; 3],
        velocity_b: [f64; 3],
    ) -> bool {
        let finite = a
            .iter()
            .chain(&b)
            .chain(&velocity_a)
            .chain(&velocity_b)
            .all(|c| c.is_finite());
        if !finite || !(radius > 0.0 && radius.is_finite()) || self.full() {
            return false;
        }
        let axis = sub(b, a);
        let axis_len2 = dot(axis, axis);
        if !axis_len2.is_finite() {
            return false;
        }
        let inv_axis_len2 = if axis_len2 > 0.0 {
            1.0 / axis_len2
        } else {
            0.0
        };
        self.capsules.push(Capsule {
            a,
            b,
            velocity_a,
            velocity_b,
            radius,
            axis,
            axis_len2,
            inv_axis_len2,
        });
        true
    }

    /// Add a box yawed about +y: debris, a crate, a corpse's bounds. The whole box moves
    /// at one velocity (it does not spin).
    ///
    /// # Arguments
    ///
    /// * `centre` - metres, where the box is at the END of the substep the set is stepped
    ///   with: it stood `velocity * dt` further back when it began.
    /// * `half_extents` - half the box's size along its own x, y and z, metres, each
    ///   positive.
    /// * `yaw` - radians, a right-handed rotation about +y: the box's own +x axis lies
    ///   along `(cos yaw, 0, -sin yaw)` and its +z along `(sin yaw, 0, cos yaw)`. Its sine
    ///   and cosine are taken here, once, so the step's inner loop has no transcendental.
    /// * `velocity` - the surface velocity, m/s: what carried the box to `centre`. The
    ///   contact casts each particle relative to it, so a drop moving with the box stays
    ///   on it.
    ///
    /// # Returns
    ///
    /// `false`, and nothing added, if any number is not finite, a half extent is not
    /// positive, or the set already holds `u32::MAX` solids.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::SphSolids;
    ///
    /// let mut solids = SphSolids::new();
    /// assert!(solids.push_box([0.0, 0.5, 0.0], [0.9, 0.2, 0.3], 0.4, [0.0, -2.0, 0.0]));
    /// assert!(!solids.push_box([0.0; 3], [0.1, 0.0, 0.1], 0.0, [0.0; 3]));
    /// ```
    pub fn push_box(
        &mut self,
        centre: [f64; 3],
        half_extents: [f64; 3],
        yaw: f64,
        velocity: [f64; 3],
    ) -> bool {
        let finite = centre
            .iter()
            .chain(&velocity)
            .chain(std::iter::once(&yaw))
            .all(|c| c.is_finite());
        let positive = half_extents.iter().all(|&e| e > 0.0 && e.is_finite());
        if !finite || !positive || self.full() {
            return false;
        }
        let (sin, cos) = yaw.sin_cos();
        self.boxes.push(YawBox {
            centre,
            half: half_extents,
            sin,
            cos,
            velocity,
        });
        true
    }

    /// Solid ids are `u32` in the bins.
    fn full(&self) -> bool {
        self.len() >= u32::MAX as usize
    }

    /// The world-space bounds of solid `id`: capsules are ids `0..capsule_count`, boxes
    /// follow.
    pub(super) fn bounds(&self, id: usize) -> ([f64; 3], [f64; 3]) {
        if id < self.capsules.len() {
            let c = &self.capsules[id];
            let r = c.radius;
            let lo = [
                c.a[0].min(c.b[0]) - r,
                c.a[1].min(c.b[1]) - r,
                c.a[2].min(c.b[2]) - r,
            ];
            let hi = [
                c.a[0].max(c.b[0]) + r,
                c.a[1].max(c.b[1]) + r,
                c.a[2].max(c.b[2]) + r,
            ];
            (lo, hi)
        } else {
            let b = &self.boxes[id - self.capsules.len()];
            let (s, c) = (b.sin.abs(), b.cos.abs());
            let ex = c * b.half[0] + s * b.half[2];
            let ez = s * b.half[0] + c * b.half[2];
            let e = [ex, b.half[1], ez];
            (
                [b.centre[0] - e[0], b.centre[1] - e[1], b.centre[2] - e[2]],
                [b.centre[0] + e[0], b.centre[1] + e[1], b.centre[2] + e[2]],
            )
        }
    }

    /// The fastest surface of solid `id`, m/s: the faster end of a capsule (its blend
    /// along the axis is never faster), a box's one speed.
    pub(super) fn max_speed(&self, id: usize) -> f64 {
        let norm = |v: [f64; 3]| dot(v, v).sqrt();
        if id < self.capsules.len() {
            let c = &self.capsules[id];
            norm(c.velocity_a).max(norm(c.velocity_b))
        } else {
            norm(self.boxes[id - self.capsules.len()].velocity)
        }
    }
}

/// What one [`SphFluid::step_with_solids`] did with its solids, read with
/// [`SphFluid::solid_stats`]. All zero after a plain [`SphFluid::step`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SphSolidStats {
    /// Solids handed to the step.
    pub solids: usize,
    /// `(solid, cell)` entries the binning wrote: one for every cell holding particles
    /// that a solid's reach covers, filed under the cell's hash bucket. Zero means no
    /// particle could meet a solid and the move ran exactly as it does without solids.
    ///
    /// Since 0.3.5 a cell counts only if it holds a particle itself. Before, a cell whose
    /// bucket held another cell's particles was binned too, which bound no particle to
    /// anything it could reach (the far cell is out of reach) but wrote, at 16,384
    /// particles spread over 20 m, a hundred thousand entries for a hull that reaches
    /// three hundred cells. The fluid is bit-identical either way; this count, and
    /// `ray_tested`, are smaller.
    pub bin_entries: usize,
    /// Particles whose bucket held at least one solid, so that ran the contact test.
    pub ray_tested: usize,
    /// Particles a solid moved: a swept hit, or a push out of a solid that moved onto
    /// them.
    pub contacts: usize,
    /// Wall time of the binning, which the step's [`super::SphPhaseTimes::grid`] includes.
    pub binning: Duration,
}

/// How [`SphFluid::bin_solids_by`] finds the solids' occupied cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) enum BinPath {
    /// Whichever is cheaper for the step's solids: the cell walk while their clipped
    /// boxes hold at most a [`SCAN_CROSSOVER`]th of the particle count in cells between
    /// them, the particle scan past that.
    Measured,
    /// The cells of each solid's clipped box.
    Cells,
    /// Every particle's cell, tested against the boxes.
    Particles,
}

/// The crossover between the two walks: the particle scan runs once the solids' clipped
/// boxes hold more than `n / SCAN_CROSSOVER` cells between them.
///
/// Measured by `binning_crossover` in the solids tests (release, one capsule of growing
/// size over a sparse field, best of 200): the walk costs about 6 ns a cell (a hash, a
/// bucket read and a look at the bucket's cells) and the scan about 1.4 ns a particle
/// (a branch-free union test, 64 particles to a word) plus its block grid. They cross at
/// about 0.45, 0.3 and 0.25 cells a particle at 1,024, 4,096 and 16,384 particles, and a
/// third picks the faster walk at every one of the 27 measured sizes.
const SCAN_CROSSOVER: u64 = 3;

/// The solids binned into the step's hash buckets: for each bucket, a range of
/// `entries` naming the solids whose reach overlaps a cell of that bucket.
///
/// `range` is indexed by bucket and is all `[0, 0]` between steps: the binning writes
/// only the buckets it touches and [`SolidBins::reset`] zeroes exactly those again, so a
/// step pays for the buckets solids reach, never for the whole table.
#[derive(Debug, Clone, Default)]
pub(super) struct SolidBins {
    pub(super) range: Vec<[u32; 2]>,
    pair_bucket: Vec<u32>,
    pair_solid: Vec<u32>,
    entries: Vec<u32>,
    touched: Vec<u32>,
    /// Each binned solid's clipped box in cells, `[min, max]` inclusive, and its id.
    boxes: Vec<[[i32; 3]; 2]>,
    box_id: Vec<u32>,
    /// The particle scan's block grid: prefix sums into `block_box`, one a block plus a
    /// terminator, and the boxes (indices into `boxes`) each block overlaps.
    block_start: Vec<u32>,
    block_box: Vec<u32>,
}

impl SolidBins {
    /// Zero the buckets this step touched.
    pub(super) fn reset(&mut self) {
        for &b in &self.touched {
            self.range[b as usize] = [0, 0];
        }
        self.touched.clear();
        self.pair_bucket.clear();
        self.pair_solid.clear();
        self.entries.clear();
        self.boxes.clear();
        self.box_id.clear();
    }

    /// Entries written by the last binning.
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}

impl SphFluid {
    /// The contact radius, metres: how far from a solid's surface the contact keeps a
    /// particle's centre, and how far past its substep the swept ray reaches.
    ///
    /// Half the rest spacing, which [`super::SphParams::with_spacing`] sets at half the
    /// smoothing radius, so a particle rests against a solid as it rests against its
    /// neighbours: one spacing between centres and the surface halfway.
    ///
    /// # Returns
    ///
    /// A quarter of the smoothing radius, metres.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{SphFluid, SphParams};
    ///
    /// let blood = SphFluid::new(SphParams::blood(), 8).unwrap();
    /// // Blood is spaced at 2 cm, so its surface sits 1 cm off a solid.
    /// assert!((blood.contact_radius() - 0.01).abs() < 1e-12);
    /// ```
    pub fn contact_radius(&self) -> f64 {
        self.params.smoothing_radius * 0.25
    }

    /// What the last [`Self::step_with_solids`] did with its solids.
    ///
    /// # Returns
    ///
    /// The counts and the binning time of the most recent step; all zero after a plain
    /// [`Self::step`] or before the first.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{SphFluid, SphParams, SphSolids};
    ///
    /// let mut fluid = SphFluid::new(SphParams::water(), 8).unwrap();
    /// fluid.spawn([0.0, 1.0, 0.0], [0.0; 3]);
    /// let mut solids = SphSolids::new();
    /// solids.push_box([0.0, 1.0, 0.0], [0.1; 3], 0.0, [0.0; 3]);
    /// fluid.step_with_solids(1.0 / 240.0, 9.81, |_, _| 0.0, &solids);
    /// let stats = fluid.solid_stats();
    /// assert_eq!((stats.solids, stats.ray_tested, stats.contacts), (1, 1, 1));
    /// ```
    pub fn solid_stats(&self) -> super::SphSolidStats {
        self.solid_stats
    }

    /// Bin every solid's reach into the hash buckets of this step's grid, for a substep
    /// of `dt` seconds.
    ///
    /// A solid's reach is its bounds grown by the farthest a particle that can meet it
    /// starts from it: the longer of the speed ceiling's travel and the solid's own
    /// travel in the substep (its fastest surface speed times `dt`), plus the contact
    /// radius. The contact casts each particle in the solid's frame (see
    /// [`Contact::resolve`]), from `p0 + v_s dt` to `p1`, so a hit point lies at
    /// `p0 + (1 - l) v_s dt + l (p1 - p0)` for `l` up to one, then up to a contact radius
    /// further: within `max(|p1 - p0|, |v_s| dt)` of `p0` plus the contact radius, and
    /// `|p1 - p0|` is at most the ceiling's travel. A particle inside the solid's start
    /// pose is within `|v_s| dt` of it. A solid slower than the ceiling therefore reaches
    /// exactly what it reached before solids were cast in their own frame. Any particle
    /// whose ray can meet the solid, or which starts inside it, starts within that
    /// reach, so its own cell
    /// (the cell `build_grid` gave it, from where the substep starts) is binned: one
    /// bucket read a particle finds every solid it can touch. Cells are clipped to the
    /// cells the fluid occupies.
    ///
    /// A solid writes one entry, under the cell's bucket, for every cell in its clipped
    /// reach that holds a particle: an occupied cell, not merely a cell whose bucket some
    /// other cell's particles share. It finds them by whichever of two walks is cheaper for
    /// it ([`BinPath::Measured`]): the cells of its box, each one hash and a look at its
    /// bucket's cells, or every particle's cell tested against the box, which costs the
    /// particle count however long the solid. Both write the same entries, so the choice
    /// changes no bin.
    ///
    /// Serial and in solid order, so the bins, and the order a particle tests its solids
    /// in, are fixed by the data.
    pub(super) fn bin_solids(&mut self, solids: &SphSolids, dt: f64) {
        self.bin_solids_by(solids, dt, BinPath::Measured);
    }

    /// [`Self::bin_solids`], with the walk chosen by `path`.
    pub(super) fn bin_solids_by(&mut self, solids: &SphSolids, dt: f64, path: BinPath) {
        self.solid_bins.reset();
        let n = self.len();
        if solids.is_empty() || n == 0 {
            return;
        }
        let h = self.params.smoothing_radius;
        // A particle's travel: a substep at the speed ceiling (CFL_FRACTION of h,
        // whatever dt is). A solid's reach is the longer of that and its own travel,
        // plus the contact radius, with room for rounding in the cap.
        let ceiling = CFL_FRACTION * h;
        let contact_radius = self.contact_radius();

        // The cells the fluid occupies: a serial min and max an axis over the integer
        // cells, which vectorises: the whole binning of 28 solids out of reach measured
        // 4.8 us at 4,096 particles with it. A rayon reduction here measured 53 us,
        // nearly all of it waking the pool.
        let span = |c: &[i32]| {
            let lo = c.iter().fold(i32::MAX, |m, &v| m.min(v));
            let hi = c.iter().fold(i32::MIN, |m, &v| m.max(v));
            (lo, hi)
        };
        let (x, y, z) = (span(&self.cell_x), span(&self.cell_y), span(&self.cell_z));
        let (lo, hi) = ([x.0, y.0, z.0], [x.1, y.1, z.1]);

        let grid = SortedCells {
            x: &self.s_cell_x[..n],
            y: &self.s_cell_y[..n],
            z: &self.s_cell_z[..n],
            start: &self.bucket_start,
            mask: self.table_mask,
        };
        let bins = &mut self.solid_bins;

        // Each solid's reach in cells, clipped to the fluid's; a solid that reaches no
        // cell the fluid spans drops out here.
        let mut cells = 0u64;
        for id in 0..solids.len() {
            let (smin, smax) = solids.bounds(id);
            // `max` returns the ceiling itself for any solid slower than it, so such a
            // solid's reach is the same number it was before the relative cast.
            let travel = ceiling.max(solids.max_speed(id) * dt);
            let reach = (travel + contact_radius) * (1.0 + 1e-6);
            let mut c0 = [0i32; 3];
            let mut c1 = [0i32; 3];
            for a in 0..3 {
                c0[a] = cell_of(smin[a] - reach, h).max(lo[a]);
                c1[a] = cell_of(smax[a] + reach, h).min(hi[a]);
            }
            if (0..3).any(|a| c0[a] > c1[a]) {
                continue;
            }
            let count = (0..3)
                .map(|a| (c1[a] as i64 - c0[a] as i64 + 1) as u64)
                .fold(1u64, u64::saturating_mul);
            cells = cells.saturating_add(count);
            bins.boxes.push([c0, c1]);
            bins.box_id.push(id as u32);
        }
        if bins.boxes.is_empty() {
            return;
        }
        let scan = match path {
            BinPath::Measured => cells.saturating_mul(SCAN_CROSSOVER) > n as u64,
            BinPath::Cells => false,
            BinPath::Particles => true,
        };
        if scan {
            bins.scan_particles(&grid, [lo, hi]);
        } else {
            bins.walk_cells(&grid);
        }
        if bins.pair_bucket.is_empty() {
            return;
        }

        // A counting sort of the pairs by bucket, over the touched buckets only.
        if bins.range.len() <= grid.mask {
            bins.range.resize(grid.mask + 1, [0, 0]);
        }
        for &b in &bins.pair_bucket {
            let r = &mut bins.range[b as usize];
            if r[1] == 0 {
                bins.touched.push(b);
            }
            r[1] += 1;
        }
        let mut at = 0u32;
        for &b in &bins.touched {
            let r = &mut bins.range[b as usize];
            let count = r[1];
            *r = [at, at];
            at += count;
        }
        bins.entries.resize(at as usize, 0);
        for (&b, &id) in bins.pair_bucket.iter().zip(&bins.pair_solid) {
            let r = &mut bins.range[b as usize];
            bins.entries[r[1] as usize] = id;
            r[1] += 1;
        }
        // The cell walk writes in solid order, so each bucket's solids are already
        // ascending. The particle scan writes in particle order; putting each bucket's
        // solids in ascending order makes its bins the walk's exactly (equal ids are
        // equal entries, so the sort's instability cannot show).
        if scan {
            for &b in &bins.touched {
                let [s, e] = bins.range[b as usize];
                let run = &mut bins.entries[s as usize..e as usize];
                if run.len() > 1 {
                    run.sort_unstable();
                }
            }
        }
    }
}

/// The step's particle cells in sorted order and the bucket table over them, borrowed
/// for the binning.
pub(super) struct SortedCells<'a> {
    x: &'a [i32],
    y: &'a [i32],
    z: &'a [i32],
    start: &'a [u32],
    mask: usize,
}

impl SortedCells<'_> {
    /// Whether bucket `b`'s run holds a particle in cell `(x, y, z)`, not only particles
    /// of another cell that hashes to the same bucket.
    #[inline]
    fn holds(&self, b: usize, x: i32, y: i32, z: i32) -> bool {
        let (s, e) = (self.start[b] as usize, self.start[b + 1] as usize);
        (s..e).any(|j| self.x[j] == x && self.y[j] == y && self.z[j] == z)
    }

    /// Whether sorted slot `k`, in bucket `b`, is the first of its cell in the bucket's
    /// run.
    #[inline]
    fn first_of_cell(&self, k: usize, b: usize) -> bool {
        let (x, y, z) = (self.x[k], self.y[k], self.z[k]);
        (self.start[b] as usize..k).all(|j| self.x[j] != x || self.y[j] != y || self.z[j] != z)
    }
}

impl SolidBins {
    /// The cell walk: each solid's clipped box, cell by cell, one hash and a look at the
    /// bucket's run a cell. Writes one pair for each occupied cell, in solid order.
    fn walk_cells(&mut self, grid: &SortedCells) {
        for (&[c0, c1], &id) in self.boxes.iter().zip(&self.box_id) {
            for cz in c0[2]..=c1[2] {
                for cy in c0[1]..=c1[1] {
                    let row = row_hash(cy, cz);
                    for cx in c0[0]..=c1[0] {
                        let b = bucket(row, cx, grid.mask);
                        if grid.holds(b, cx, cy, cz) {
                            self.pair_bucket.push(b as u32);
                            self.pair_solid.push(id);
                        }
                    }
                }
            }
        }
    }

    /// The particle scan: every particle's cell against the solids whose boxes could hold
    /// it. Writes the same pairs as [`Self::walk_cells`], in particle order.
    ///
    /// The boxes are first listed in a dense grid of blocks, `2^s` cells a side, over
    /// their union. `s` is chosen per step by the cost it implies, each term a count of
    /// simple operations: the blocks to clear and sum, the box-block entries to write, and
    /// a box test for each particle in the union for each box its block lists (the
    /// particles in the union estimated from its share of the fluid's cell range). A
    /// single hull takes one block; a melee's limbs take blocks about a limb wide. A
    /// particle outside the union costs one branch on six compares; one inside tests its
    /// cell against its block's boxes, in solid order. A cell is written at its first
    /// particle in its bucket's run, so once however many particles it holds.
    fn scan_particles(&mut self, grid: &SortedCells, span: [[i32; 3]; 2]) {
        let SolidBins {
            pair_bucket,
            pair_solid,
            boxes,
            box_id,
            block_start,
            block_box,
            ..
        } = self;
        let n = grid.x.len();
        let mut u0 = [i32::MAX; 3];
        let mut u1 = [i32::MIN; 3];
        for [c0, c1] in boxes.iter() {
            for a in 0..3 {
                u0[a] = u0[a].min(c0[a]);
                u1[a] = u1[a].max(c1[a]);
            }
        }
        let width = |lo: i32, hi: i32| (hi as i64 - lo as i64 + 1) as f64;
        let share = (0..3)
            .map(|a| width(u0[a], u1[a]) / width(span[0][a], span[1][a]))
            .product::<f64>();
        let in_union = n as f64 * share;

        // A cell's offset into the union, in blocks of 2^s cells: under 2^32.
        let off = |c: i32, a: usize, s: u32| ((c as i64 - u0[a] as i64) as u64) >> s;
        let shape = |s: u32| {
            let dims = [
                off(u1[0], 0, s) + 1,
                off(u1[1], 1, s) + 1,
                off(u1[2], 2, s) + 1,
            ];
            let blocks = dims[0].saturating_mul(dims[1]).saturating_mul(dims[2]);
            let listed = boxes
                .iter()
                .map(|[c0, c1]| {
                    (0..3)
                        .map(|a| off(c1[a], a, s) - off(c0[a], a, s) + 1)
                        .fold(1u64, u64::saturating_mul)
                })
                .fold(0u64, u64::saturating_add);
            (dims, blocks, listed)
        };
        let cost = |s: u32| {
            let (_, blocks, listed) = shape(s);
            let (b, l) = (blocks as f64, listed as f64);
            b + l + in_union * l / b
        };
        // Past the shift that makes the union one block a side nothing changes, so the
        // search ends there: about log2 of the union's width, each try one pass over the
        // boxes. The first of equal costs wins.
        let widest = (0..3).map(|a| off(u1[a], a, 0)).max().unwrap_or(0);
        let last = 64 - widest.leading_zeros();
        let mut s = last;
        let mut best = cost(last);
        for t in 0..last {
            let c = cost(t);
            if c < best || (c == best && t < s) {
                best = c;
                s = t;
            }
        }
        let (dims, blocks, _) = shape(s);
        let blocks = blocks as usize;
        let index = |b: [u64; 3]| ((b[2] * dims[1] + b[1]) * dims[0] + b[0]) as usize;
        let block_box_of = |c0: [i32; 3], c1: [i32; 3]| {
            (
                [off(c0[0], 0, s), off(c0[1], 1, s), off(c0[2], 2, s)],
                [off(c1[0], 0, s), off(c1[1], 1, s), off(c1[2], 2, s)],
            )
        };

        // The block-box list, a counting sort in box order, so each block's boxes are in
        // solid order.
        block_start.clear();
        block_start.resize(blocks + 1, 0);
        for &[c0, c1] in boxes.iter() {
            let (b0, b1) = block_box_of(c0, c1);
            for bz in b0[2]..=b1[2] {
                for by in b0[1]..=b1[1] {
                    for bx in b0[0]..=b1[0] {
                        block_start[index([bx, by, bz]) + 1] += 1;
                    }
                }
            }
        }
        for k in 0..blocks {
            block_start[k + 1] += block_start[k];
        }
        block_box.clear();
        block_box.resize(block_start[blocks] as usize, 0);
        // Fill through `block_start[k]` as the cursor, which leaves it at block k's end,
        // then shift the table back by one.
        for (i, &[c0, c1]) in boxes.iter().enumerate() {
            let (b0, b1) = block_box_of(c0, c1);
            for bz in b0[2]..=b1[2] {
                for by in b0[1]..=b1[1] {
                    for bx in b0[0]..=b1[0] {
                        let k = index([bx, by, bz]);
                        block_box[block_start[k] as usize] = i as u32;
                        block_start[k] += 1;
                    }
                }
            }
        }
        for k in (1..=blocks).rev() {
            block_start[k] = block_start[k - 1];
        }
        block_start[0] = 0;

        let inside = |c: [i32; 3], c0: [i32; 3], c1: [i32; 3]| {
            (c[0] >= c0[0])
                & (c[0] <= c1[0])
                & (c[1] >= c0[1])
                & (c[1] <= c1[1])
                & (c[2] >= c0[2])
                & (c[2] <= c1[2])
        };
        // The union test as data, 64 particles to a word, so it compiles to compares and
        // masks with no branch a particle; as a branch it measured 7 ns a particle, much
        // of it mispredicted. Only the particles whose bit is set go on.
        let ext = [0, 1, 2].map(|a| u1[a].wrapping_sub(u0[a]) as u32);
        let mut bits = 0u64;
        let mut base = 0usize;
        let mut k = 0usize;
        loop {
            if bits == 0 {
                if base >= n {
                    break;
                }
                let end = (base + 64).min(n);
                let (xs, ys, zs) = (&grid.x[base..end], &grid.y[base..end], &grid.z[base..end]);
                for j in 0..xs.len() {
                    // One unsigned compare an axis: below the union's start wraps high.
                    let hit = ((xs[j].wrapping_sub(u0[0]) as u32) <= ext[0])
                        & ((ys[j].wrapping_sub(u0[1]) as u32) <= ext[1])
                        & ((zs[j].wrapping_sub(u0[2]) as u32) <= ext[2]);
                    bits |= (hit as u64) << j;
                }
                k = base;
                base = end;
                continue;
            }
            let slot = k + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let c = [grid.x[slot], grid.y[slot], grid.z[slot]];
            let blk = index([off(c[0], 0, s), off(c[1], 1, s), off(c[2], 2, s)]);
            let (j0, j1) = (block_start[blk] as usize, block_start[blk + 1] as usize);
            // The bucket and whether this slot is its cell's first, worked out at the
            // first box that holds the cell.
            let mut first: Option<(usize, bool)> = None;
            for &i in &block_box[j0..j1] {
                let [c0, c1] = boxes[i as usize];
                if !inside(c, c0, c1) {
                    continue;
                }
                let (b, write) = *first.get_or_insert_with(|| {
                    let b = bucket(row_hash(c[1], c[2]), c[0], grid.mask);
                    (b, grid.first_of_cell(slot, b))
                });
                if !write {
                    break;
                }
                pair_bucket.push(b as u32);
                pair_solid.push(box_id[i as usize]);
            }
        }
    }
}

/// Everything the contact reads, borrowed for one move.
pub(super) struct Contact<'a> {
    pub(super) solids: &'a SphSolids,
    pub(super) range: &'a [[u32; 2]],
    pub(super) entries: &'a [u32],
    pub(super) contact_radius: f64,
    pub(super) restitution: f64,
    pub(super) friction_keep: f64,
    /// The substep, seconds: how far back along its velocity a moving solid's start pose
    /// lies from the end pose it is given at.
    pub(super) dt: f64,
}

impl<'a> Contact<'a> {
    pub(super) fn new(
        fluid_bins: &'a SolidBins,
        solids: &'a SphSolids,
        contact_radius: f64,
        restitution: f64,
        friction_keep: f64,
        dt: f64,
    ) -> Contact<'a> {
        Contact {
            solids,
            range: &fluid_bins.range,
            entries: &fluid_bins.entries,
            contact_radius,
            restitution,
            friction_keep,
            dt,
        }
    }

    /// The solids binned in `bucket`, in bin order.
    #[inline]
    pub(super) fn binned(&self, bucket: u32) -> &'a [u32] {
        let [s, e] = self.range[bucket as usize];
        &self.entries[s as usize..e as usize]
    }

    /// Meet the solids `ids` on the substep from `p0` to `p1` at velocity `v`, each in
    /// its own frame.
    ///
    /// A solid is given at its pose at the END of the substep, moving at its surface
    /// velocity, so it stood that velocity times `dt` further back when the substep
    /// began. Seen from the solid, the particle started at `p0 + v_s dt` and ends at
    /// `p1`: its displacement relative to the surface is `p1 - p0 - v_s dt`, and that is
    /// what it casts, against the solid at the given pose. A drop carried with the
    /// surface casts only what gravity added, straight at the surface; a particle the
    /// surface swept onto casts back along the solid's motion and meets its leading face.
    /// `v_s` is the velocity of the surface nearest `p0` (a capsule's blend at the axis
    /// parameter of `p0`, a box's one velocity). A solid at rest casts exactly the world
    /// displacement it always did.
    ///
    /// A particle that starts inside a solid, in that solid's frame (inside the pose the
    /// solid had when the substep began), is pushed out of the first such solid, in bin
    /// order, along the nearest surface normal. Otherwise the nearest surface along the
    /// rays, each reaching the contact radius past `p1`, stops it: it is set on the
    /// surface, pushed out by the contact radius along the normal. "Nearest" is the
    /// distance along each solid's own ray. Either way its velocity relative to the
    /// surface loses its approaching normal part to `restitution` and its tangential part
    /// to the ground's friction decay, and then takes the surface's velocity.
    ///
    /// Returns, if a solid moved the particle, that solid's id and the square of the
    /// particle's speed relative to the surface after the response, which is what its
    /// stillness on the solid is judged by.
    #[inline]
    pub(super) fn resolve(
        &self,
        ids: &[u32],
        p0: [f64; 3],
        p1: &mut [f64; 3],
        v: &mut [f64; 3],
    ) -> Option<(u32, f64)> {
        // The world ray, which every solid at rest casts, worked out at the first one: a
        // particle meeting only moving solids never needs it.
        let mut world: Option<Ray> = None;

        let caps = self.solids.capsules.len();
        let mut best = f64::INFINITY;
        let mut best_id = u32::MAX;
        let mut best_ray = Ray {
            origin: p0,
            dir: [0.0; 3],
            reach: 0.0,
            moving: false,
        };
        for &id in ids {
            let k = id as usize;
            let vs = if k < caps {
                capsule_velocity_near(&self.solids.capsules[k], p0)
            } else {
                self.solids.boxes[k - caps].velocity
            };
            // A solid at rest takes the world ray itself, so its arithmetic is the
            // world cast's operation for operation (and -0.0 stays -0.0).
            let ray = if vs == [0.0; 3] {
                *world.get_or_insert_with(|| Ray::between(p0, *p1, self.contact_radius))
            } else {
                Ray::between(add(p0, scale(vs, self.dt)), *p1, self.contact_radius)
            };
            let inside = if k < caps {
                capsule_inside(&self.solids.capsules[k], ray.origin)
            } else {
                box_inside(&self.solids.boxes[k - caps], ray.origin)
            };
            if let Some(hit) = inside {
                return Some((id, self.respond(hit, p1, v)));
            }
            if !ray.moving {
                continue;
            }
            let s = if k < caps {
                capsule_ray(&self.solids.capsules[k], ray.origin, ray.dir)
            } else {
                box_ray(&self.solids.boxes[k - caps], ray.origin, ray.dir)
            };
            if s <= ray.reach && s < best {
                best = s;
                best_id = id;
                best_ray = ray;
            }
        }
        if best_id == u32::MAX {
            return None;
        }
        let q = add(best_ray.origin, scale(best_ray.dir, best));
        let k = best_id as usize;
        let hit = if k < caps {
            capsule_surface(&self.solids.capsules[k], q)
        } else {
            box_surface(&self.solids.boxes[k - caps], q)
        };
        Some((best_id, self.respond(hit, p1, v)))
    }

    /// Set the particle on the surface, a contact radius out, and apply the response.
    /// Returns the square of its speed relative to the surface after it.
    #[inline]
    fn respond(&self, hit: Hit, p: &mut [f64; 3], v: &mut [f64; 3]) -> f64 {
        *p = add(hit.point, scale(hit.normal, self.contact_radius));
        let n = hit.normal;
        let rel = sub(*v, hit.velocity);
        let vn = dot(rel, n);
        let tangential = sub(rel, scale(n, vn));
        // Only an approaching particle bounces: one already leaving the surface faster
        // than it would keep its normal speed, or it would be sent back into the solid.
        let normal = if vn < 0.0 { -vn * self.restitution } else { vn };
        let kept = add(scale(tangential, self.friction_keep), scale(n, normal));
        *v = add(hit.velocity, kept);
        dot(kept, kept)
    }
}

/// A particle's swept ray in one solid's frame: from where it started, as that solid
/// sees it, towards where it ends.
#[derive(Debug, Clone, Copy)]
struct Ray {
    origin: [f64; 3],
    /// Unit direction, or zero when the particle did not move in this frame.
    dir: [f64; 3],
    /// The displacement's length plus the contact radius: how far along `dir` a hit
    /// still counts. Zero when not moving.
    reach: f64,
    moving: bool,
}

impl Ray {
    /// The ray from `origin` to `end`, reaching `contact_radius` past it.
    #[inline]
    fn between(origin: [f64; 3], end: [f64; 3], contact_radius: f64) -> Ray {
        let d = sub(end, origin);
        let len2 = dot(d, d);
        let moving = len2 > 0.0;
        let (dir, reach) = if moving {
            let len = len2.sqrt();
            (scale(d, 1.0 / len), len + contact_radius)
        } else {
            ([0.0; 3], 0.0)
        };
        Ray {
            origin,
            dir,
            reach,
            moving,
        }
    }
}

/// The surface velocity of capsule `c` nearest `p`: its one velocity if both ends move
/// alike, else the ends' blend at `p`'s axis parameter.
#[inline]
fn capsule_velocity_near(c: &Capsule, p: [f64; 3]) -> [f64; 3] {
    if c.velocity_a == c.velocity_b {
        return c.velocity_a;
    }
    let t = capsule_param(c, p);
    add(c.velocity_a, scale(sub(c.velocity_b, c.velocity_a), t))
}

/// A point on a solid's surface, its outward unit normal, and the surface's velocity
/// there.
#[derive(Debug, Clone, Copy)]
struct Hit {
    point: [f64; 3],
    normal: [f64; 3],
    velocity: [f64; 3],
}

/// The closest point of `c`'s axis to `p`, as the parameter along it in `[0, 1]`.
#[inline]
fn capsule_param(c: &Capsule, p: [f64; 3]) -> f64 {
    (dot(sub(p, c.a), c.axis) * c.inv_axis_len2).clamp(0.0, 1.0)
}

/// The surface point of `c` nearest `p` (outward from the axis), its normal and the
/// surface velocity, given `p`'s offset from the axis.
#[inline]
fn capsule_hit(c: &Capsule, p: [f64; 3], t: f64) -> Hit {
    let on_axis = add(c.a, scale(c.axis, t));
    let off = sub(p, on_axis);
    let len2 = dot(off, off);
    let normal = if len2 > 0.0 {
        scale(off, 1.0 / len2.sqrt())
    } else {
        perpendicular(c.axis, c.axis_len2)
    };
    Hit {
        point: add(on_axis, scale(normal, c.radius)),
        normal,
        velocity: add(c.velocity_a, scale(sub(c.velocity_b, c.velocity_a), t)),
    }
}

/// A unit vector perpendicular to `axis`, for a point exactly on a capsule's axis: up
/// with the axis taken out, or +x for an upright axis. Any fixed choice is as right as
/// another; it only has to be the same every time.
#[inline]
fn perpendicular(axis: [f64; 3], axis_len2: f64) -> [f64; 3] {
    if axis_len2 > 0.0 {
        let up = [0.0, 1.0, 0.0];
        let along = scale(axis, dot(up, axis) / axis_len2);
        let p = sub(up, along);
        let l2 = dot(p, p);
        if l2 > 0.25 {
            return scale(p, 1.0 / l2.sqrt());
        }
        let x = [1.0, 0.0, 0.0];
        let p = sub(x, scale(axis, dot(x, axis) / axis_len2));
        return scale(p, 1.0 / dot(p, p).sqrt());
    }
    [0.0, 1.0, 0.0]
}

/// `Some` surface contact if `p` is strictly inside capsule `c`.
#[inline]
fn capsule_inside(c: &Capsule, p: [f64; 3]) -> Option<Hit> {
    let t = capsule_param(c, p);
    let off = sub(p, add(c.a, scale(c.axis, t)));
    if dot(off, off) < c.radius * c.radius {
        Some(capsule_hit(c, p, t))
    } else {
        None
    }
}

/// The contact at surface point `q` of capsule `c`.
#[inline]
fn capsule_surface(c: &Capsule, q: [f64; 3]) -> Hit {
    capsule_hit(c, q, capsule_param(c, q))
}

/// Distance along the unit ray `o + s rd` to where it enters capsule `c`, for an origin
/// outside it; infinite on a miss. The first point of the ray in the capsule is the first
/// in any of its three convex parts (the side of the cylinder, the two end spheres), so
/// it is the nearest of three closed-form entries; a ray entering through a flat end of
/// the cylinder meets that end's sphere first.
#[inline]
fn capsule_ray(c: &Capsule, o: [f64; 3], rd: [f64; 3]) -> f64 {
    let r2 = c.radius * c.radius;
    let mut best = f64::INFINITY;
    let baba = c.axis_len2;
    if baba > 0.0 {
        let oa = sub(o, c.a);
        let bard = dot(c.axis, rd);
        let baoa = dot(c.axis, oa);
        let qa = baba - bard * bard;
        // Parallel to the axis the side is never entered; the spheres take it.
        if qa > 1e-12 * baba {
            let qb = baba * dot(rd, oa) - baoa * bard;
            let qc = baba * dot(oa, oa) - baoa * baoa - r2 * baba;
            let disc = qb * qb - qa * qc;
            if disc >= 0.0 {
                let s = (-qb - disc.sqrt()) / qa;
                let y = baoa + s * bard;
                if s >= 0.0 && y >= 0.0 && y <= baba {
                    best = s;
                }
            }
        }
    }
    for centre in [c.a, c.b] {
        let oc = sub(o, centre);
        let b = dot(rd, oc);
        let disc = b * b - (dot(oc, oc) - r2);
        if disc >= 0.0 {
            let s = -b - disc.sqrt();
            if s >= 0.0 && s < best {
                best = s;
            }
        }
    }
    best
}

/// World vector `w` in box `b`'s frame.
#[inline]
fn to_box(b: &YawBox, w: [f64; 3]) -> [f64; 3] {
    [
        b.cos * w[0] - b.sin * w[2],
        w[1],
        b.sin * w[0] + b.cos * w[2],
    ]
}

/// Box `b`'s local vector `l` in the world.
#[inline]
fn from_box(b: &YawBox, l: [f64; 3]) -> [f64; 3] {
    [
        b.cos * l[0] + b.sin * l[2],
        l[1],
        -b.sin * l[0] + b.cos * l[2],
    ]
}

/// The contact on face `axis` of box `b` (outward along `sign`) nearest local point `l`.
#[inline]
fn box_face(b: &YawBox, l: [f64; 3], axis: usize, sign: f64) -> Hit {
    let mut q = l;
    q[axis] = sign * b.half[axis];
    let mut n = [0.0; 3];
    n[axis] = sign;
    Hit {
        point: add(b.centre, from_box(b, q)),
        normal: from_box(b, n),
        velocity: b.velocity,
    }
}

/// `Some` contact on the nearest face if `p` is strictly inside box `b`.
#[inline]
fn box_inside(b: &YawBox, p: [f64; 3]) -> Option<Hit> {
    let l = to_box(b, sub(p, b.centre));
    let mut axis = 0;
    let mut depth = f64::INFINITY;
    for a in 0..3 {
        let d = b.half[a] - l[a].abs();
        if d <= 0.0 {
            return None;
        }
        if d < depth {
            depth = d;
            axis = a;
        }
    }
    let sign = if l[axis] >= 0.0 { 1.0 } else { -1.0 };
    Some(box_face(b, l, axis, sign))
}

/// The contact at surface point `q` of box `b`: the face `q` lies on, taken as the axis
/// where `q` is deepest outside, or least inside, its slab.
#[inline]
fn box_surface(b: &YawBox, q: [f64; 3]) -> Hit {
    let l = to_box(b, sub(q, b.centre));
    let mut axis = 0;
    let mut out = f64::NEG_INFINITY;
    for a in 0..3 {
        let o = l[a].abs() - b.half[a];
        if o > out {
            out = o;
            axis = a;
        }
    }
    let sign = if l[axis] >= 0.0 { 1.0 } else { -1.0 };
    box_face(b, l, axis, sign)
}

/// Distance along the unit ray `o + s rd` to where it enters box `b`, for an origin
/// outside it; infinite on a miss. The slab test in the box's frame: three divisions.
#[inline]
fn box_ray(b: &YawBox, o: [f64; 3], rd: [f64; 3]) -> f64 {
    let lo = to_box(b, sub(o, b.centre));
    let ld = to_box(b, rd);
    let mut enter = f64::NEG_INFINITY;
    let mut exit = f64::INFINITY;
    for a in 0..3 {
        let h = b.half[a];
        if ld[a] == 0.0 {
            if lo[a].abs() >= h {
                return f64::INFINITY;
            }
            continue;
        }
        let inv = 1.0 / ld[a];
        let (t0, t1) = ((-h - lo[a]) * inv, (h - lo[a]) * inv);
        let (near, far) = if t0 < t1 { (t0, t1) } else { (t1, t0) };
        enter = enter.max(near);
        exit = exit.min(far);
    }
    if enter <= exit && enter >= 0.0 {
        enter
    } else {
        f64::INFINITY
    }
}

#[inline]
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
