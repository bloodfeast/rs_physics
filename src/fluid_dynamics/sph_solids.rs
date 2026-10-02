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
/// reads it and never writes it.
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
    /// * `a`, `b` - the axis end points, metres.
    /// * `radius` - metres, positive.
    /// * `velocity_a`, `velocity_b` - the surface velocity at each end, m/s.
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
    /// * `centre` - metres.
    /// * `half_extents` - half the box's size along its own x, y and z, metres, each
    ///   positive.
    /// * `yaw` - radians, a right-handed rotation about +y: the box's own +x axis lies
    ///   along `(cos yaw, 0, -sin yaw)` and its +z along `(sin yaw, 0, cos yaw)`. Its sine
    ///   and cosine are taken here, once, so the step's inner loop has no transcendental.
    /// * `velocity` - the surface velocity, m/s.
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
}

/// What one [`SphFluid::step_with_solids`] did with its solids, read with
/// [`SphFluid::solid_stats`]. All zero after a plain [`SphFluid::step`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SphSolidStats {
    /// Solids handed to the step.
    pub solids: usize,
    /// `(solid, cell)` entries the binning wrote: one for every hash bucket holding
    /// particles that a solid's reach overlaps. Zero means no particle could meet a solid
    /// and the move ran exactly as it does without solids.
    pub bin_entries: usize,
    /// Particles whose cell held at least one solid, so that ran the contact test.
    pub ray_tested: usize,
    /// Particles a solid moved: a swept hit, or a push out of a solid that moved onto
    /// them.
    pub contacts: usize,
    /// Wall time of the binning, which the step's [`super::SphPhaseTimes::grid`] includes.
    pub binning: Duration,
}

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

    /// Bin every solid's reach into the hash buckets of this step's grid.
    ///
    /// A solid's reach is its bounds grown by the longest swept ray a particle can cast,
    /// the speed ceiling's travel plus the contact radius. Any particle whose ray can meet
    /// the solid, or which starts inside it, starts within that reach, so its own cell
    /// (the cell `build_grid` gave it, from where the substep starts) is binned: one
    /// bucket read a particle finds every solid it can touch. Cells are clipped to the
    /// cells the fluid occupies, and a cell whose bucket holds no particle writes nothing.
    ///
    /// Serial and in solid order, so the bins, and the order a particle tests its solids
    /// in, are fixed by the data.
    pub(super) fn bin_solids(&mut self, solids: &SphSolids) {
        self.solid_bins.reset();
        let n = self.len();
        if solids.is_empty() || n == 0 {
            return;
        }
        let h = self.params.smoothing_radius;
        // The longest ray: a substep at the speed ceiling (CFL_FRACTION of h, whatever
        // dt is) plus the contact radius, with room for rounding in the cap.
        let reach = (CFL_FRACTION * h + self.contact_radius()) * (1.0 + 1e-6);

        // The cells the fluid occupies: a serial min and max an axis over the integer
        // cells, which vectorises and costs about a microsecond at 4,096 particles. A rayon
        // reduction here measured 53 us at 4,096, all of it waking the pool.
        let span = |c: &[i32]| {
            let lo = c.iter().fold(i32::MAX, |m, &v| m.min(v));
            let hi = c.iter().fold(i32::MIN, |m, &v| m.max(v));
            (lo, hi)
        };
        let (x, y, z) = (span(&self.cell_x), span(&self.cell_y), span(&self.cell_z));
        let (lo, hi) = ([x.0, y.0, z.0], [x.1, y.1, z.1]);

        let mask = self.table_mask;
        let start = &self.bucket_start;
        let bins = &mut self.solid_bins;
        for id in 0..solids.len() {
            let (smin, smax) = solids.bounds(id);
            let mut c0 = [0i32; 3];
            let mut c1 = [0i32; 3];
            for a in 0..3 {
                c0[a] = cell_of(smin[a] - reach, h).max(lo[a]);
                c1[a] = cell_of(smax[a] + reach, h).min(hi[a]);
            }
            if (0..3).any(|a| c0[a] > c1[a]) {
                continue;
            }
            for cz in c0[2]..=c1[2] {
                for cy in c0[1]..=c1[1] {
                    let row = row_hash(cy, cz);
                    for cx in c0[0]..=c1[0] {
                        let b = bucket(row, cx, mask);
                        if start[b] < start[b + 1] {
                            bins.pair_bucket.push(b as u32);
                            bins.pair_solid.push(id as u32);
                        }
                    }
                }
            }
        }
        if bins.pair_bucket.is_empty() {
            return;
        }

        // A counting sort of the pairs by bucket, over the touched buckets only.
        if bins.range.len() <= mask {
            bins.range.resize(mask + 1, [0, 0]);
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
}

impl<'a> Contact<'a> {
    pub(super) fn new(
        fluid_bins: &'a SolidBins,
        solids: &'a SphSolids,
        contact_radius: f64,
        restitution: f64,
        friction_keep: f64,
    ) -> Contact<'a> {
        Contact {
            solids,
            range: &fluid_bins.range,
            entries: &fluid_bins.entries,
            contact_radius,
            restitution,
            friction_keep,
        }
    }

    /// The solids binned in `bucket`, in bin order.
    #[inline]
    pub(super) fn binned(&self, bucket: u32) -> &'a [u32] {
        let [s, e] = self.range[bucket as usize];
        &self.entries[s as usize..e as usize]
    }

    /// Meet the solids `ids` on the substep from `p0` to `p1` at velocity `v`.
    ///
    /// A particle that starts inside a solid (the solid moved onto it) is pushed out of
    /// the first such solid, in bin order, along the nearest surface normal. Otherwise
    /// it casts a ray from `p0` along its displacement, reaching the contact radius past
    /// `p1`, and the nearest surface it meets stops it: it is set on the surface, pushed
    /// out by the contact radius along the normal. Either way its velocity relative to
    /// the surface loses its approaching normal part to `restitution` and its tangential
    /// part to the ground's friction decay, and then takes the surface's velocity.
    ///
    /// Returns whether a solid moved the particle.
    #[inline]
    pub(super) fn resolve(
        &self,
        ids: &[u32],
        p0: [f64; 3],
        p1: &mut [f64; 3],
        v: &mut [f64; 3],
    ) -> bool {
        let d = sub(*p1, p0);
        let len2 = dot(d, d);
        let moving = len2 > 0.0;
        let (rd, s_max) = if moving {
            let len = len2.sqrt();
            (scale(d, 1.0 / len), len + self.contact_radius)
        } else {
            ([0.0; 3], 0.0)
        };

        let caps = self.solids.capsules.len();
        let mut best = f64::INFINITY;
        let mut best_id = u32::MAX;
        for &id in ids {
            let k = id as usize;
            let inside = if k < caps {
                capsule_inside(&self.solids.capsules[k], p0)
            } else {
                box_inside(&self.solids.boxes[k - caps], p0)
            };
            if let Some(hit) = inside {
                self.respond(hit, p1, v);
                return true;
            }
            if !moving {
                continue;
            }
            let s = if k < caps {
                capsule_ray(&self.solids.capsules[k], p0, rd)
            } else {
                box_ray(&self.solids.boxes[k - caps], p0, rd)
            };
            if s <= s_max && s < best {
                best = s;
                best_id = id;
            }
        }
        if best_id == u32::MAX {
            return false;
        }
        let q = add(p0, scale(rd, best));
        let k = best_id as usize;
        let hit = if k < caps {
            capsule_surface(&self.solids.capsules[k], q)
        } else {
            box_surface(&self.solids.boxes[k - caps], q)
        };
        self.respond(hit, p1, v);
        true
    }

    /// Set the particle on the surface, a contact radius out, and apply the response.
    #[inline]
    fn respond(&self, hit: Hit, p: &mut [f64; 3], v: &mut [f64; 3]) {
        *p = add(hit.point, scale(hit.normal, self.contact_radius));
        let n = hit.normal;
        let rel = sub(*v, hit.velocity);
        let vn = dot(rel, n);
        let tangential = sub(rel, scale(n, vn));
        // Only an approaching particle bounces: one already leaving the surface faster
        // than it would keep its normal speed, or it would be sent back into the solid.
        let normal = if vn < 0.0 { -vn * self.restitution } else { vn };
        *v = add(
            hit.velocity,
            add(scale(tangential, self.friction_keep), scale(n, normal)),
        );
    }
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
