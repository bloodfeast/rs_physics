//! A divergence-free swirl for effect particles: curl noise whose amplitude and rate
//! come from the plume that drives it.
//!
//! Smoke and dust from [`ParticleEffects`](crate::particles::ParticleEffects) see
//! gravity and drag and nothing else, so a column of smoke rises as a column. Real
//! smoke rolls, because the air it rides is turbulent. This module supplies that air:
//!
//! - [`VelocityGrid`] is a coarse 3D grid of air velocity (`f32`, metres per second)
//!   that a particle reads with one trilinear fetch. It is the one thing the particle
//!   loop touches.
//! - [`SwirlField`] fills one with *curl noise* (Bridson, Hourihan and Nordenstam,
//!   "Curl-Noise for Procedural Fluid Flow", SIGGRAPH 2007): band-limited noise in a
//!   vector potential `A`, whose curl is the velocity. A curl has no divergence, so the
//!   swirl never bunches smoke up or thins it out.
//! - [`TurbulenceDrive`] is where the numbers come from. A smoke source already knows
//!   how fast its plume rises (`U`) and how wide it is (`L`); Kolmogorov's scaling
//!   gives the speed of an eddy of any smaller size `l` as `U (l / L)^(1/3)`, and its
//!   turnover time as `l` over that. The field has no amplitude or rate parameter of
//!   its own.
//!
//! # Example
//!
//! ```
//! use rs_physics::particles::{ParticleClass, ParticleEffects, SwirlField, TurbulenceDrive};
//!
//! // A smoke source 6 m wide rising at 3 m/s, in a 32 x 32 x 32 field of 2 m cells.
//! let drive = TurbulenceDrive::new(3.0, 6.0).unwrap();
//! let mut swirl = SwirlField::new([-32.0, 0.0, -32.0], [32, 32, 32], 2.0, drive, 7).unwrap();
//!
//! let mut fx = ParticleEffects::with_capacity(256);
//! // Smoke: light, draggy, and fully carried by the air it is in.
//! fx.set_class(0, ParticleClass { gravity: 0.5, drag: 2.0, restitution: 0.0 });
//! fx.set_swirl(0, 1.0);
//! for i in 0..64 {
//!     fx.emit_one([i as f32 * 0.1, 10.0, 0.0], [0.0, 3.0, 0.0], 30.0, 1.0, 0);
//! }
//! // The swirl updates at 10 Hz, and each particle re-samples it once a period.
//! for frame in 0..60 {
//!     if frame % 6 == 0 {
//!         swirl.advance(0.1);
//!     }
//!     fx.integrate_in_air(1.0 / 60.0, swirl.velocity(), 0.1);
//! }
//! assert_eq!(fx.len(), 64);
//! ```

#![warn(missing_docs)]

use crate::particles::EffectRng;
use crate::utils::PhysicsError;

/// A 3D grid of air velocity that particles sample, metres per second.
///
/// Cell `(i, j, k)` covers `origin + [i, j, k] * h` to one cell further, and its
/// velocity is the air's at the cell's centre. [`VelocityGrid::sample`] interpolates
/// between centres trilinearly and clamps outside them, so a point beyond the grid
/// reads the outermost cell: a grid whose outer layer is zero (as [`SwirlField`]
/// writes it) reads as still air everywhere outside.
///
/// The cells are stored as four `f32`s (`x`, `y`, `z` and a zero pad), `z` fastest,
/// so each of a fetch's eight taps is one 16-byte load. Memory: 16 bytes per
/// cell, 512 KiB for 32^3.
#[derive(Debug, Clone)]
pub struct VelocityGrid {
    origin: [f32; 3],
    h: f32,
    dims: [usize; 3],
    /// `origin + h / 2`: the centre of cell `(0, 0, 0)`.
    first_centre: [f32; 3],
    inv_h: f32,
    /// `dims - 1` per axis, the largest sample coordinate.
    last: [f32; 3],
    /// `dims - 2` per axis, the largest base cell of a fetch.
    last_base: [f32; 3],
    cells: Vec<[f32; 4]>,
}

impl VelocityGrid {
    /// A grid of still air.
    ///
    /// # Arguments
    ///
    /// * `origin` - the low corner of cell `(0, 0, 0)`, metres.
    /// * `h` - the cell size, metres; finite and positive.
    /// * `dims` - cells along `x`, `y` and `z`; at least 2 on each axis, so a sample
    ///   always has two centres to interpolate between.
    ///
    /// # Returns
    ///
    /// The grid, every cell zero.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidDistance`] for a bad `h` or a non-finite `origin`, and
    /// [`PhysicsError::InvalidDimension`] for an axis of fewer than 2 cells.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::VelocityGrid;
    /// let grid = VelocityGrid::new([0.0; 3], 2.0, [32, 32, 32]).unwrap();
    /// assert_eq!(grid.sample([10.0, 10.0, 10.0]), [0.0; 3]);
    /// assert!(VelocityGrid::new([0.0; 3], 2.0, [1, 32, 32]).is_err());
    /// ```
    pub fn new(origin: [f32; 3], h: f32, dims: [usize; 3]) -> Result<VelocityGrid, PhysicsError> {
        if !(h.is_finite() && h > 0.0) || origin.iter().any(|o| !o.is_finite()) {
            return Err(PhysicsError::InvalidDistance);
        }
        if dims.iter().any(|&n| n < 2) {
            return Err(PhysicsError::InvalidDimension);
        }
        let cells = dims[0]
            .checked_mul(dims[1])
            .and_then(|c| c.checked_mul(dims[2]))
            .ok_or(PhysicsError::InvalidDimension)?;
        Ok(VelocityGrid {
            origin,
            h,
            dims,
            first_centre: origin.map(|o| o + 0.5 * h),
            inv_h: 1.0 / h,
            last: dims.map(|n| (n - 1) as f32),
            last_base: dims.map(|n| (n - 2) as f32),
            cells: vec![[0.0; 4]; cells],
        })
    }

    /// The low corner of cell `(0, 0, 0)`, metres.
    ///
    /// # Returns
    ///
    /// `[x, y, z]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::VelocityGrid;
    /// let grid = VelocityGrid::new([1.0, 2.0, 3.0], 2.0, [4, 4, 4]).unwrap();
    /// assert_eq!(grid.origin(), [1.0, 2.0, 3.0]);
    /// ```
    pub fn origin(&self) -> [f32; 3] {
        self.origin
    }

    /// The cell size, metres.
    ///
    /// # Returns
    ///
    /// The `h` the grid was built with.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::VelocityGrid;
    /// assert_eq!(VelocityGrid::new([0.0; 3], 1.5, [4, 4, 4]).unwrap().cell_size(), 1.5);
    /// ```
    pub fn cell_size(&self) -> f32 {
        self.h
    }

    /// Cells along each axis.
    ///
    /// # Returns
    ///
    /// `[nx, ny, nz]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::VelocityGrid;
    /// assert_eq!(VelocityGrid::new([0.0; 3], 1.0, [4, 5, 6]).unwrap().dims(), [4, 5, 6]);
    /// ```
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// The memory the cells take, bytes: 16 per cell.
    ///
    /// # Returns
    ///
    /// The size of the cell array.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::VelocityGrid;
    /// assert_eq!(VelocityGrid::new([0.0; 3], 2.0, [32, 32, 32]).unwrap().bytes(), 524_288);
    /// ```
    pub fn bytes(&self) -> usize {
        self.cells.len() * std::mem::size_of::<[f32; 4]>()
    }

    #[inline(always)]
    fn index(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.dims[1] + j) * self.dims[2] + k
    }

    /// The velocity stored at cell `(i, j, k)`, m/s.
    ///
    /// # Arguments
    ///
    /// * `i`, `j`, `k` - the cell, each below its axis's [`Self::dims`].
    ///
    /// # Returns
    ///
    /// `[x, y, z]`.
    ///
    /// # Panics
    ///
    /// If the cell is outside the grid.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::VelocityGrid;
    /// let mut grid = VelocityGrid::new([0.0; 3], 1.0, [4, 4, 4]).unwrap();
    /// grid.set(1, 2, 3, [1.0, 0.0, -1.0]);
    /// assert_eq!(grid.get(1, 2, 3), [1.0, 0.0, -1.0]);
    /// ```
    pub fn get(&self, i: usize, j: usize, k: usize) -> [f32; 3] {
        assert!(
            i < self.dims[0] && j < self.dims[1] && k < self.dims[2],
            "cell outside the grid"
        );
        let c = self.cells[self.index(i, j, k)];
        [c[0], c[1], c[2]]
    }

    /// Stores a velocity at cell `(i, j, k)`, for a caller that fills the grid from its
    /// own source of air motion.
    ///
    /// # Arguments
    ///
    /// * `i`, `j`, `k` - the cell, each below its axis's [`Self::dims`].
    /// * `velocity` - m/s; a non-finite component is stored as zero, so one bad value
    ///   cannot poison every particle that samples near it.
    ///
    /// # Panics
    ///
    /// If the cell is outside the grid.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::VelocityGrid;
    /// let mut grid = VelocityGrid::new([0.0; 3], 1.0, [4, 4, 4]).unwrap();
    /// grid.set(0, 0, 0, [f32::NAN, 2.0, 0.0]);
    /// assert_eq!(grid.get(0, 0, 0), [0.0, 2.0, 0.0]);
    /// ```
    pub fn set(&mut self, i: usize, j: usize, k: usize, velocity: [f32; 3]) {
        assert!(
            i < self.dims[0] && j < self.dims[1] && k < self.dims[2],
            "cell outside the grid"
        );
        let idx = self.index(i, j, k);
        let v = velocity.map(|c| if c.is_finite() { c } else { 0.0 });
        self.cells[idx] = [v[0], v[1], v[2], 0.0];
    }

    /// Sets every cell to `velocity`.
    ///
    /// # Arguments
    ///
    /// * `velocity` - m/s; non-finite components are stored as zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::VelocityGrid;
    /// let mut grid = VelocityGrid::new([0.0; 3], 1.0, [4, 4, 4]).unwrap();
    /// grid.fill([3.0, 0.0, 0.0]);
    /// assert_eq!(grid.sample([100.0, -5.0, 2.0]), [3.0, 0.0, 0.0]);
    /// ```
    pub fn fill(&mut self, velocity: [f32; 3]) {
        let v = velocity.map(|c| if c.is_finite() { c } else { 0.0 });
        self.cells.fill([v[0], v[1], v[2], 0.0]);
    }

    pub(crate) fn cells_mut(&mut self) -> &mut [[f32; 4]] {
        &mut self.cells
    }

    pub(crate) fn cells(&self) -> &[[f32; 4]] {
        &self.cells
    }

    /// The air velocity at a point, m/s: one trilinear fetch of the eight cell centres
    /// around it. Outside the grid the point is clamped to the outermost centres.
    ///
    /// # Arguments
    ///
    /// * `p` - the point, metres. A non-finite coordinate reads cell 0 on that axis.
    ///
    /// # Returns
    ///
    /// `[x, y, z]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::VelocityGrid;
    /// let mut grid = VelocityGrid::new([0.0; 3], 1.0, [2, 2, 2]).unwrap();
    /// grid.set(1, 0, 0, [2.0, 0.0, 0.0]);
    /// grid.set(1, 1, 0, [2.0, 0.0, 0.0]);
    /// grid.set(1, 0, 1, [2.0, 0.0, 0.0]);
    /// grid.set(1, 1, 1, [2.0, 0.0, 0.0]);
    /// // Half-way between the centres at x = 0.5 and x = 1.5.
    /// assert_eq!(grid.sample([1.0, 0.7, 1.2]), [1.0, 0.0, 0.0]);
    /// ```
    #[inline(always)]
    pub fn sample(&self, p: [f32; 3]) -> [f32; 3] {
        #[cfg(target_arch = "x86_64")]
        {
            self.sample_sse(p)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.sample_portable(p)
        }
    }

    /// [`Self::sample`] with SSE2, which every x86-64 has: the three axes' clamp, floor
    /// and fraction in one register, each tap one 16-byte load, and the seven lerps on
    /// all three components at once. The result is the portable version's to the bit
    /// (the same operations in the same order, lane by lane), which a test checks.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn sample_sse(&self, p: [f32; 3]) -> [f32; 3] {
        use std::arch::x86_64::*;
        let [fc, last, last_base] =
            [self.first_centre, self.last, self.last_base].map(|v| [v[0], v[1], v[2], 0.0f32]);
        let sy = self.dims[2];
        let sx = self.dims[1] * sy;
        // SAFETY: SSE2 is part of the x86-64 baseline. Every load is inside `cells`:
        // the base cell's index on each axis is at most `dims - 2` (the clamp to
        // `last_base` happens before the conversion, and `max` sends NaN to 0), so the
        // farthest tap, `base + sx + sy + 1`, is the last cell at most. Each tap reads
        // the four `f32`s of one `[f32; 4]`.
        unsafe {
            let g = _mm_mul_ps(
                _mm_sub_ps(_mm_set_ps(0.0, p[2], p[1], p[0]), _mm_loadu_ps(fc.as_ptr())),
                _mm_set1_ps(self.inv_h),
            );
            // `max(g, 0)` returns its second operand when `g` is NaN.
            let g = _mm_min_ps(_mm_max_ps(g, _mm_setzero_ps()), _mm_loadu_ps(last.as_ptr()));
            let base = _mm_cvttps_epi32(_mm_min_ps(g, _mm_loadu_ps(last_base.as_ptr())));
            let f = _mm_sub_ps(g, _mm_cvtepi32_ps(base));
            let i = _mm_cvtsi128_si32(base) as usize;
            let j = _mm_cvtsi128_si32(_mm_shuffle_epi32(base, 0x55)) as usize;
            let k = _mm_cvtsi128_si32(_mm_shuffle_epi32(base, 0xAA)) as usize;
            let tap = self.cells.as_ptr().add(i * sx + j * sy + k) as *const f32;
            let load = |offset: usize| _mm_loadu_ps(tap.add(4 * offset));
            let lerp =
                |a: __m128, b: __m128, t: __m128| _mm_add_ps(a, _mm_mul_ps(_mm_sub_ps(b, a), t));
            let (fx, fy, fz) = (
                _mm_shuffle_ps(f, f, 0x00),
                _mm_shuffle_ps(f, f, 0x55),
                _mm_shuffle_ps(f, f, 0xAA),
            );
            let x0 = lerp(
                lerp(load(0), load(1), fz),
                lerp(load(sy), load(sy + 1), fz),
                fy,
            );
            let x1 = lerp(
                lerp(load(sx), load(sx + 1), fz),
                lerp(load(sx + sy), load(sx + sy + 1), fz),
                fy,
            );
            let mut out = [0.0f32; 4];
            _mm_storeu_ps(out.as_mut_ptr(), lerp(x0, x1, fx));
            [out[0], out[1], out[2]]
        }
    }

    /// Samples a run of particles into the air arrays: `a[i] = sample(p[i])`, all six
    /// slices the same length. The same values as calling [`Self::sample`] on each, to
    /// the bit; on x86-64 two particles go through at once, their instructions
    /// interleaved, so the out-of-order core always has a second fetch to work on while
    /// the first waits on its loads (one fetch is a single dependent chain of about 50
    /// cycles, and two in flight take little longer than one).
    pub(crate) fn sample_run(&self, p: [&[f32]; 3], a: [&mut [f32]; 3]) {
        let [px, py, pz] = p;
        let [ax, ay, az] = a;
        let n = px
            .len()
            .min(py.len())
            .min(pz.len())
            .min(ax.len())
            .min(ay.len())
            .min(az.len());
        let mut i = 0;
        #[cfg(target_arch = "x86_64")]
        while i + 2 <= n {
            let (u, v) =
                self.sample_pair_sse([px[i], py[i], pz[i]], [px[i + 1], py[i + 1], pz[i + 1]]);
            ax[i] = u[0];
            ay[i] = u[1];
            az[i] = u[2];
            ax[i + 1] = v[0];
            ay[i + 1] = v[1];
            az[i + 1] = v[2];
            i += 2;
        }
        while i < n {
            let u = self.sample([px[i], py[i], pz[i]]);
            ax[i] = u[0];
            ay[i] = u[1];
            az[i] = u[2];
            i += 1;
        }
    }

    /// Two [`Self::sample_sse`] fetches, written side by side.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn sample_pair_sse(&self, p: [f32; 3], q: [f32; 3]) -> ([f32; 3], [f32; 3]) {
        use std::arch::x86_64::*;
        let [fc, last, last_base] =
            [self.first_centre, self.last, self.last_base].map(|v| [v[0], v[1], v[2], 0.0f32]);
        let sy = self.dims[2];
        let sx = self.dims[1] * sy;
        // SAFETY: as in `sample_sse`, for each of the two points.
        unsafe {
            let (fc, last, last_base, inv_h) = (
                _mm_loadu_ps(fc.as_ptr()),
                _mm_loadu_ps(last.as_ptr()),
                _mm_loadu_ps(last_base.as_ptr()),
                _mm_set1_ps(self.inv_h),
            );
            let gp = _mm_mul_ps(_mm_sub_ps(_mm_set_ps(0.0, p[2], p[1], p[0]), fc), inv_h);
            let gq = _mm_mul_ps(_mm_sub_ps(_mm_set_ps(0.0, q[2], q[1], q[0]), fc), inv_h);
            let gp = _mm_min_ps(_mm_max_ps(gp, _mm_setzero_ps()), last);
            let gq = _mm_min_ps(_mm_max_ps(gq, _mm_setzero_ps()), last);
            let bp = _mm_cvttps_epi32(_mm_min_ps(gp, last_base));
            let bq = _mm_cvttps_epi32(_mm_min_ps(gq, last_base));
            let fp = _mm_sub_ps(gp, _mm_cvtepi32_ps(bp));
            let fq = _mm_sub_ps(gq, _mm_cvtepi32_ps(bq));
            let index = |b: __m128i| {
                let i = _mm_cvtsi128_si32(b) as usize;
                let j = _mm_cvtsi128_si32(_mm_shuffle_epi32(b, 0x55)) as usize;
                let k = _mm_cvtsi128_si32(_mm_shuffle_epi32(b, 0xAA)) as usize;
                i * sx + j * sy + k
            };
            let (tp, tq) = (
                self.cells.as_ptr().add(index(bp)) as *const f32,
                self.cells.as_ptr().add(index(bq)) as *const f32,
            );
            let lp = |offset: usize| _mm_loadu_ps(tp.add(4 * offset));
            let lq = |offset: usize| _mm_loadu_ps(tq.add(4 * offset));
            let lerp =
                |a: __m128, b: __m128, t: __m128| _mm_add_ps(a, _mm_mul_ps(_mm_sub_ps(b, a), t));
            let (pzf, qzf) = (_mm_shuffle_ps(fp, fp, 0xAA), _mm_shuffle_ps(fq, fq, 0xAA));
            let (pyf, qyf) = (_mm_shuffle_ps(fp, fp, 0x55), _mm_shuffle_ps(fq, fq, 0x55));
            let (pxf, qxf) = (_mm_shuffle_ps(fp, fp, 0x00), _mm_shuffle_ps(fq, fq, 0x00));
            let p00 = lerp(lp(0), lp(1), pzf);
            let q00 = lerp(lq(0), lq(1), qzf);
            let p01 = lerp(lp(sy), lp(sy + 1), pzf);
            let q01 = lerp(lq(sy), lq(sy + 1), qzf);
            let p10 = lerp(lp(sx), lp(sx + 1), pzf);
            let q10 = lerp(lq(sx), lq(sx + 1), qzf);
            let p11 = lerp(lp(sx + sy), lp(sx + sy + 1), pzf);
            let q11 = lerp(lq(sx + sy), lq(sx + sy + 1), qzf);
            let (px0, qx0) = (lerp(p00, p01, pyf), lerp(q00, q01, qyf));
            let (px1, qx1) = (lerp(p10, p11, pyf), lerp(q10, q11, qyf));
            let (up, uq) = (lerp(px0, px1, pxf), lerp(qx0, qx1, qxf));
            let (mut op, mut oq) = ([0.0f32; 4], [0.0f32; 4]);
            _mm_storeu_ps(op.as_mut_ptr(), up);
            _mm_storeu_ps(oq.as_mut_ptr(), uq);
            ([op[0], op[1], op[2]], [oq[0], oq[1], oq[2]])
        }
    }

    /// [`Self::sample`] in plain Rust, for targets without SSE2 and as the reference
    /// the SSE version is tested against.
    #[cfg_attr(all(target_arch = "x86_64", not(test)), allow(dead_code))]
    #[inline(always)]
    fn sample_portable(&self, p: [f32; 3]) -> [f32; 3] {
        let mut base = [0usize; 3];
        let mut f = [0.0f32; 3];
        for a in 0..3 {
            // `max` then `min` rather than `clamp`, so NaN lands on 0 instead of passing.
            let g = ((p[a] - self.first_centre[a]) * self.inv_h)
                .max(0.0)
                .min(self.last[a]);
            // Truncation is the floor here, `g` being non-negative; the last centre
            // belongs to the cell pair below it.
            let i = g.min(self.last_base[a]) as usize;
            base[a] = i;
            f[a] = g - i as f32;
        }
        let sy = self.dims[2];
        let sx = self.dims[1] * sy;
        let b = base[0] * sx + base[1] * sy + base[2];
        // Every tap is inside: each base is at most `dims - 2`.
        let c = &self.cells[b..b + sx + sy + 2];
        let lerp = |a: [f32; 4], b: [f32; 4], t: f32| -> [f32; 4] {
            [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
                0.0,
            ]
        };
        let x0 = lerp(lerp(c[0], c[1], f[2]), lerp(c[sy], c[sy + 1], f[2]), f[1]);
        let x1 = lerp(
            lerp(c[sx], c[sx + 1], f[2]),
            lerp(c[sx + sy], c[sx + sy + 1], f[2]),
            f[1],
        );
        let v = lerp(x0, x1, f[0]);
        [v[0], v[1], v[2]]
    }

    /// The portable fetch, for checking the SSE one against.
    #[cfg(test)]
    pub(crate) fn sample_reference(&self, p: [f32; 3]) -> [f32; 3] {
        self.sample_portable(p)
    }

    /// The central-difference divergence at cell `(i, j, k)`, 1/s, for checking a field
    /// that should have none. Needs a neighbour on each side.
    #[cfg(test)]
    pub(crate) fn divergence_at(&self, i: usize, j: usize, k: usize) -> f32 {
        let d = |a: [f32; 3], b: [f32; 3], axis: usize| b[axis] - a[axis];
        (d(self.get(i - 1, j, k), self.get(i + 1, j, k), 0)
            + d(self.get(i, j - 1, k), self.get(i, j + 1, k), 1)
            + d(self.get(i, j, k - 1), self.get(i, j, k + 1), 2))
            * (0.5 * self.inv_h)
    }
}

/// The plume a turbulent field is driven by: how fast it moves and how wide it is.
///
/// Turbulence takes its energy at the scale of whatever drives it (a plume's width)
/// and hands it down a cascade of smaller eddies. In the inertial range, between that
/// scale and the tiny one where viscosity takes over, Kolmogorov's 1941 scaling gives
/// the speed of an eddy of size `l` from the energy flux alone, and the flux is set at
/// the top: `u_l = U (l / L)^(1/3)`. Its turnover time, how long it takes to go round
/// once and so how fast the swirl it makes changes, is `tau_l = l / u_l`.
///
/// Valid for `l < L`. At `l >= L` there is no eddy of that size to speak of: it is the
/// plume itself, which a [`SwirlField`] does not supply (the plume's own large-scale
/// motion is a fluid solver's job; see `PlumeField` with the `fluid_simulation`
/// feature).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurbulenceDrive {
    velocity: f32,
    scale: f32,
}

impl TurbulenceDrive {
    /// A drive of speed `velocity` at scale `scale`.
    ///
    /// # Arguments
    ///
    /// * `velocity` - `U`, m/s: the plume's driving speed, a smoke column's rise speed;
    ///   finite and non-negative (zero gives still air).
    /// * `scale` - `L`, metres: the driving scale, the source's width; finite and
    ///   positive.
    ///
    /// # Returns
    ///
    /// The drive.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidVelocity`] for a bad `velocity`,
    /// [`PhysicsError::InvalidDistance`] for a bad `scale`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::TurbulenceDrive;
    /// assert!(TurbulenceDrive::new(3.0, 6.0).is_ok());
    /// assert!(TurbulenceDrive::new(3.0, 0.0).is_err());
    /// ```
    pub fn new(velocity: f32, scale: f32) -> Result<TurbulenceDrive, PhysicsError> {
        if !(velocity.is_finite() && velocity >= 0.0) {
            return Err(PhysicsError::InvalidVelocity);
        }
        if !(scale.is_finite() && scale > 0.0) {
            return Err(PhysicsError::InvalidDistance);
        }
        Ok(TurbulenceDrive { velocity, scale })
    }

    /// `U`, m/s.
    ///
    /// # Returns
    ///
    /// The driving velocity.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::TurbulenceDrive;
    /// assert_eq!(TurbulenceDrive::new(3.0, 6.0).unwrap().velocity(), 3.0);
    /// ```
    pub fn velocity(&self) -> f32 {
        self.velocity
    }

    /// `L`, metres.
    ///
    /// # Returns
    ///
    /// The driving scale.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::TurbulenceDrive;
    /// assert_eq!(TurbulenceDrive::new(3.0, 6.0).unwrap().scale(), 6.0);
    /// ```
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// The speed of an eddy of size `l`, `U (l / L)^(1/3)`, m/s.
    ///
    /// # Arguments
    ///
    /// * `l` - the eddy size, metres. The law holds below `L`; at or above it this
    ///   returns `U`, the plume's own speed.
    ///
    /// # Returns
    ///
    /// The eddy velocity.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::TurbulenceDrive;
    /// let drive = TurbulenceDrive::new(4.0, 8.0).unwrap();
    /// // An eddy an eighth of the plume's width turns at half its speed.
    /// assert!((drive.eddy_velocity(1.0) - 2.0).abs() < 1e-6);
    /// ```
    pub fn eddy_velocity(&self, l: f32) -> f32 {
        self.velocity * (l / self.scale).clamp(0.0, 1.0).cbrt()
    }

    /// The turnover time of an eddy of size `l`, `l / u_l`, seconds: how long the swirl
    /// it makes takes to change. Infinite in still air.
    ///
    /// # Arguments
    ///
    /// * `l` - the eddy size, metres.
    ///
    /// # Returns
    ///
    /// The turnover time.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::TurbulenceDrive;
    /// let drive = TurbulenceDrive::new(4.0, 8.0).unwrap();
    /// assert!((drive.turnover_time(1.0) - 0.5).abs() < 1e-6);
    /// ```
    pub fn turnover_time(&self, l: f32) -> f32 {
        let u = self.eddy_velocity(l);
        if u > 0.0 {
            l / u
        } else {
            f32::INFINITY
        }
    }
}

/// One octave of the swirl: a lattice of random potential values at a spacing of
/// `spacing` cells, cross-faded between two draws as its eddies turn over.
#[derive(Debug, Clone)]
struct Octave {
    /// Lattice spacing, cells: 2, 4 or 8.
    spacing: usize,
    /// Eddy size, metres: `spacing * h`.
    scale: f32,
    /// Its eddy speed and turnover time from the drive.
    speed: f32,
    turnover: f32,
    /// What multiplies a unit-variance lattice so the octave's rms speed is `speed`.
    amplitude: f32,
    /// Lattice points per axis.
    dims: [usize; 3],
    /// The draw fading out and the one fading in, three channels each.
    current: [Vec<f32>; 3],
    next: [Vec<f32>; 3],
    /// Progress through the current fade, 0 to 1; one fade lasts one turnover.
    phase: f32,
}

/// Variance of a lattice value drawn uniformly from `[-1, 1]`.
const LATTICE_VARIANCE: f64 = 1.0 / 3.0;

/// One 1D cubic B-spline subdivision (Lane and Riesenfeld): halve the spacing of
/// `coarse`, writing every fine point that has its full support. Points `q` of the
/// result sit where `q + 2` would in an untrimmed subdivision, which keeps a margin of
/// two points below the first cell at every level (see [`SwirlField`]).
#[inline]
fn subdivide_1d(coarse: &[f32], fine: &mut [f32]) {
    for (q, out) in fine.iter_mut().enumerate() {
        let full = q + 2;
        let m = full / 2;
        *out = if full % 2 == 0 {
            (coarse[m - 1] + 6.0 * coarse[m] + coarse[m + 1]) * 0.125
        } else {
            (coarse[m] + coarse[m + 1]) * 0.5
        };
    }
}

/// Lattice points a level of spacing `s` cells needs on an axis of `n` cells: from two
/// points below cell 0 to two above cell `n - 1`, with one more so the next finer level
/// has the support it needs.
fn lattice_points(n: usize, s: usize) -> usize {
    (n - 1).div_ceil(s) + 5
}

/// The exact expected statistics of one octave subdivided `levels` times: `(S, D)`
/// with `S` the mean square of the composite weights and `D` the mean square of their
/// central difference (half the two-sided step), both per lattice point. A field of
/// unit-variance lattice values then has variance `S` per axis it is smooth along and
/// a central-difference gradient variance `D` (per cell^2) along the axis it is
/// differenced on.
fn octave_statistics(levels: u32) -> (f64, f64) {
    // A single lattice value, subdivided `levels` times with no trimming.
    let mut response = vec![0.0f64; 9];
    response[4] = 1.0;
    for _ in 0..levels {
        let mut fine = vec![0.0f64; 2 * response.len() + 1];
        for (m, &c) in response.iter().enumerate() {
            // The transpose of `subdivide_1d`: where coarse point `m` lands.
            fine[2 * m] += 0.75 * c;
            if 2 * m >= 1 {
                fine[2 * m - 1] += 0.5 * c;
            }
            fine[2 * m + 1] += 0.5 * c;
            if 2 * m >= 2 {
                fine[2 * m - 2] += 0.125 * c;
            }
            fine[2 * m + 2] += 0.125 * c;
        }
        response = fine;
    }
    let spacing = (1u64 << levels) as f64;
    let s: f64 = response.iter().map(|r| r * r).sum::<f64>() / spacing;
    let d: f64 = (1..response.len() - 1)
        .map(|q| (0.5 * (response[q + 1] - response[q - 1])).powi(2))
        .sum::<f64>()
        / spacing;
    (s, d)
}

/// A swirling, divergence-free air velocity over a box: curl noise driven by a
/// [`TurbulenceDrive`].
///
/// # What it holds
///
/// Up to three octaves of noise in a vector potential `A`, with lattice spacings of 2,
/// 4 and 8 cells. An octave of spacing `s` makes eddies about `l = s h` across, so the
/// smallest is `2h`: the smallest a grid of spacing `h` can carry (an eddy of `h` would
/// be a wave at the grid's Nyquist limit). Octaves at or above the drive's `L` are left
/// out, being the plume rather than its turbulence; with `L <= 2h` there is none and
/// the field is still air, which is the right answer when every eddy is below the
/// grid.
///
/// # Amplitude, derived
///
/// Each octave's rms speed is the drive's eddy velocity at its size,
/// `u_l = U (l / L)^(1/3)` ([`TurbulenceDrive::eddy_velocity`]). The noise's amplitude
/// is set from that exactly, not tuned: the lattice values have variance 1/3, the
/// subdivision that smooths them is linear with weights known in closed form, and the
/// curl by central differences is linear too, so the expected squared speed of an
/// octave of amplitude `a` is `6 (a^2 / 3) S^2 D / h^2` with `S` and `D` sums of the
/// composite weights (computed once at construction). Setting that to `u_l^2` gives
/// `a`. A single realisation scatters about the expectation, more for the coarse
/// octaves on a small grid (fewer lattice points); the test measures it on 64^3.
///
/// # Rate, derived
///
/// Each octave cross-fades between two independent draws, `cos(t) A1 + sin(t) A2` with
/// `t` running from 0 to a right angle over one turnover time `l / u_l`
/// ([`TurbulenceDrive::turnover_time`]); then the second draw becomes the first and a
/// fresh one fades in. The sum of squares of the weights is one throughout, so the
/// amplitude never dips mid-fade, and after one turnover an octave's swirl is a new one.
/// Large eddies change slowly and small ones fast, as they do.
///
/// # Divergence
///
/// The velocity is the curl of `A` by central differences on the cell centres, which
/// has zero central-difference divergence exactly (the difference operators commute),
/// so to `f32` rounding in practice. The outermost layer of cells is set to zero, so
/// outside the box [`VelocityGrid::sample`] reads still air. That layer is the one
/// place the field is not divergence-free: the swirl stops over the last cell.
///
/// # Cost
///
/// [`SwirlField::advance`] rebuilds the potential (a subdivision pyramid, finest pass
/// once for all octaves) and its curl: measured 2026-10-02 at 0.67 ms for 32^3 and
/// 4.5 ms for 64^3 on one core, beside another build (`benches/particle_effects.rs`,
/// `swirl_advance`). Memory, as [`SwirlField::bytes`] reports it: 1.69 MB at 32^3
/// (512 KiB of velocity grid, 384 KiB of potential, the rest pyramid scratch and
/// lattices) and 12.0 MB at 64^3.
#[derive(Debug, Clone)]
pub struct SwirlField {
    drive: TurbulenceDrive,
    h: f32,
    dims: [usize; 3],
    octaves: Vec<Octave>,
    rng: EffectRng,
    /// The potential at the cell centres, one array per channel.
    potential: [Vec<f32>; 3],
    /// Pyramid scratch: a level's lattice, the two half-subdivided stages, and a
    /// strided line's coarse and fine copies.
    level: Vec<f32>,
    stage: [Vec<f32>; 2],
    lines: [Vec<f32>; 2],
    velocity: VelocityGrid,
}

impl SwirlField {
    /// A swirl over the box of `dims` cells of size `h` from `origin`, driven by
    /// `drive`, at its first frame.
    ///
    /// # Arguments
    ///
    /// * `origin` - the box's low corner, metres.
    /// * `dims` - cells per axis; at least 3 each (the outer layer is still air).
    /// * `h` - the cell size, metres. A 32^3 field of 2 m cells covers a 64 m region.
    /// * `drive` - the plume's speed and width, from which the swirl's amplitude and
    ///   rate follow.
    /// * `seed` - the noise's random stream; the same seed gives the same swirl.
    ///
    /// # Returns
    ///
    /// The field, with its velocity built.
    ///
    /// # Errors
    ///
    /// As [`VelocityGrid::new`], and [`PhysicsError::InvalidDimension`] for an axis of
    /// fewer than 3 cells.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{SwirlField, TurbulenceDrive};
    /// let drive = TurbulenceDrive::new(3.0, 20.0).unwrap();
    /// let swirl = SwirlField::new([0.0; 3], [16, 16, 16], 2.0, drive, 1).unwrap();
    /// // Eddies of 4, 8 and 16 m: all three octaves are below the 20 m plume.
    /// assert_eq!(swirl.octave_scales().len(), 3);
    /// ```
    pub fn new(
        origin: [f32; 3],
        dims: [usize; 3],
        h: f32,
        drive: TurbulenceDrive,
        seed: u32,
    ) -> Result<SwirlField, PhysicsError> {
        if dims.iter().any(|&n| n < 3) {
            return Err(PhysicsError::InvalidDimension);
        }
        let velocity = VelocityGrid::new(origin, h, dims)?;
        let mut rng = EffectRng::new(seed);
        let mut octaves = Vec::new();
        for o in 0..3u32 {
            let spacing = 2usize << o;
            let scale = spacing as f32 * h;
            if scale >= drive.scale() || drive.velocity() == 0.0 {
                break;
            }
            let speed = drive.eddy_velocity(scale);
            let (s, d) = octave_statistics(o + 1);
            // E|u|^2 = 6 (a^2 sigma^2) S^2 D / h^2, solved for a with E|u|^2 = speed^2.
            let amplitude = (speed as f64 * h as f64
                / (S_D_FACTOR * LATTICE_VARIANCE * s * s * d).sqrt())
                as f32;
            let lattice_dims = dims.map(|n| lattice_points(n, spacing));
            let points = lattice_dims.iter().product::<usize>();
            let mut draw = || {
                std::array::from_fn(|_| {
                    (0..points)
                        .map(|_| rng.range(-1.0, 1.0))
                        .collect::<Vec<f32>>()
                })
            };
            let current = draw();
            let next = draw();
            octaves.push(Octave {
                spacing,
                scale,
                speed,
                turnover: drive.turnover_time(scale),
                amplitude,
                dims: lattice_dims,
                current,
                next,
                phase: 0.0,
            });
        }
        let cells = dims.iter().product::<usize>();
        let mut field = SwirlField {
            drive,
            h,
            dims,
            octaves,
            rng,
            potential: std::array::from_fn(|_| vec![0.0; cells]),
            level: Vec::new(),
            stage: [Vec::new(), Vec::new()],
            lines: [Vec::new(), Vec::new()],
            velocity,
        };
        field.rebuild();
        Ok(field)
    }

    /// The drive this field was built from.
    ///
    /// # Returns
    ///
    /// The [`TurbulenceDrive`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{SwirlField, TurbulenceDrive};
    /// let drive = TurbulenceDrive::new(3.0, 20.0).unwrap();
    /// let swirl = SwirlField::new([0.0; 3], [8, 8, 8], 2.0, drive, 1).unwrap();
    /// assert_eq!(swirl.drive(), drive);
    /// ```
    pub fn drive(&self) -> TurbulenceDrive {
        self.drive
    }

    /// The eddy size of each octave the field holds, metres, smallest first.
    ///
    /// # Returns
    ///
    /// `2h`, `4h` and `8h`, as many as are below the drive's scale.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{SwirlField, TurbulenceDrive};
    /// let drive = TurbulenceDrive::new(3.0, 10.0).unwrap();
    /// let swirl = SwirlField::new([0.0; 3], [8, 8, 8], 2.0, drive, 1).unwrap();
    /// // 4 m and 8 m eddies; a 16 m one would be wider than the 10 m plume.
    /// assert_eq!(swirl.octave_scales(), vec![4.0, 8.0]);
    /// ```
    pub fn octave_scales(&self) -> Vec<f32> {
        self.octaves.iter().map(|o| o.scale).collect()
    }

    /// The air velocity the particles read.
    ///
    /// # Returns
    ///
    /// The field's [`VelocityGrid`], as of the last [`Self::advance`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{SwirlField, TurbulenceDrive};
    /// let drive = TurbulenceDrive::new(3.0, 20.0).unwrap();
    /// let swirl = SwirlField::new([0.0; 3], [8, 8, 8], 2.0, drive, 1).unwrap();
    /// assert_eq!(swirl.velocity().dims(), [8, 8, 8]);
    /// ```
    pub fn velocity(&self) -> &VelocityGrid {
        &self.velocity
    }

    /// Everything the field keeps, bytes: the velocity grid, the potential, the
    /// lattices and the pyramid scratch.
    ///
    /// # Returns
    ///
    /// The total.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{SwirlField, TurbulenceDrive};
    /// let drive = TurbulenceDrive::new(3.0, 20.0).unwrap();
    /// let swirl = SwirlField::new([0.0; 3], [32, 32, 32], 2.0, drive, 1).unwrap();
    /// assert!(swirl.bytes() > swirl.velocity().bytes());
    /// ```
    pub fn bytes(&self) -> usize {
        let floats = |v: &Vec<f32>| v.capacity() * 4;
        let mut total = self.velocity.bytes();
        total += self.potential.iter().map(floats).sum::<usize>();
        total += floats(&self.level)
            + self
                .stage
                .iter()
                .chain(self.lines.iter())
                .map(floats)
                .sum::<usize>();
        for o in &self.octaves {
            total += o
                .current
                .iter()
                .chain(o.next.iter())
                .map(floats)
                .sum::<usize>();
        }
        total
    }

    /// Moves the swirl on by `dt` and rebuilds the velocity.
    ///
    /// Each octave advances its cross-fade by `dt` over its own turnover time; the
    /// cost is the rebuild, the same whatever `dt` is, so a caller sets the update rate
    /// by how often it calls this (the swirl changes over a turnover time, a second or
    /// so for metre-scale eddies, so 10 to 20 Hz is plenty) and passes the time since
    /// the last call.
    ///
    /// # Arguments
    ///
    /// * `dt` - seconds since the last call. Zero, negative or non-finite rebuilds
    ///   nothing.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{SwirlField, TurbulenceDrive};
    /// let drive = TurbulenceDrive::new(3.0, 20.0).unwrap();
    /// let mut swirl = SwirlField::new([0.0; 3], [8, 8, 8], 2.0, drive, 1).unwrap();
    /// let before = swirl.velocity().get(4, 4, 4);
    /// swirl.advance(0.5);
    /// assert_ne!(swirl.velocity().get(4, 4, 4), before);
    /// ```
    pub fn advance(&mut self, dt: f32) {
        if !(dt.is_finite() && dt > 0.0) || self.octaves.is_empty() {
            return;
        }
        for o in &mut self.octaves {
            o.phase += dt / o.turnover;
            // A long gap is one or more whole fades: draw what is needed, at most two.
            let whole = o.phase.floor();
            if whole >= 1.0 {
                o.phase -= whole;
                if whole >= 2.0 {
                    for c in 0..3 {
                        for v in o.current[c].iter_mut() {
                            *v = self.rng.range(-1.0, 1.0);
                        }
                    }
                } else {
                    std::mem::swap(&mut o.current, &mut o.next);
                }
                for c in 0..3 {
                    for v in o.next[c].iter_mut() {
                        *v = self.rng.range(-1.0, 1.0);
                    }
                }
            }
        }
        self.rebuild();
    }

    /// Rebuilds the potential through the pyramid, then its curl.
    fn rebuild(&mut self) {
        let n = self.dims;
        if self.octaves.is_empty() {
            self.velocity.fill([0.0; 3]);
            return;
        }
        let coarsest = self.octaves.len() - 1;
        for channel in 0..3 {
            // The coarsest octave's lattice, faded.
            let o = &self.octaves[coarsest];
            let mut dims = o.dims;
            fade_into(o, channel, &mut self.level);
            // Down the pyramid: subdivide, then add the next finer octave's lattice.
            for finer in (0..coarsest).rev() {
                let target = self.octaves[finer].dims;
                subdivide_3d(&self.level, dims, target, &mut self.stage, &mut self.lines);
                dims = target;
                std::mem::swap(&mut self.level, &mut self.stage[1]);
                add_faded(&self.octaves[finer], channel, &mut self.level);
            }
            // The last subdivision lands on the cells, two points in from the margin.
            let cells_dims = n.map(|c| c + 4);
            subdivide_3d(
                &self.level,
                dims,
                cells_dims,
                &mut self.stage,
                &mut self.lines,
            );
            let fine = &self.stage[1];
            let out = &mut self.potential[channel];
            for i in 0..n[0] {
                for j in 0..n[1] {
                    let src = ((i + 2) * cells_dims[1] + (j + 2)) * cells_dims[2] + 2;
                    let dst = (i * n[1] + j) * n[2];
                    out[dst..dst + n[2]].copy_from_slice(&fine[src..src + n[2]]);
                }
            }
        }
        self.curl();
    }

    /// The velocity is `curl A` by central differences, zero on the outer layer.
    fn curl(&mut self) {
        let n = self.dims;
        let (sx, sy) = (n[1] * n[2], n[2]);
        let inv_2h = 0.5 / self.h;
        let [ax, ay, az] = &self.potential;
        let cells = self.velocity.cells_mut();
        cells.fill([0.0; 4]);
        for i in 1..n[0] - 1 {
            for j in 1..n[1] - 1 {
                for k in 1..n[2] - 1 {
                    let c = (i * n[1] + j) * n[2] + k;
                    let x = (az[c + sy] - az[c - sy]) - (ay[c + 1] - ay[c - 1]);
                    let y = (ax[c + 1] - ax[c - 1]) - (az[c + sx] - az[c - sx]);
                    let z = (ay[c + sx] - ay[c - sx]) - (ax[c + sy] - ax[c - sy]);
                    cells[c] = [x * inv_2h, y * inv_2h, z * inv_2h, 0.0];
                }
            }
        }
    }

    /// The rms speed an octave is built to have, m/s, by its index (0 is the
    /// finest), with its eddy size and turnover time.
    #[cfg(test)]
    pub(crate) fn octave_law(&self, octave: usize) -> (f32, f32, f32) {
        let o = &self.octaves[octave];
        (o.scale, o.speed, o.turnover)
    }

    /// Keeps only octave `octave`, for measuring one at a time.
    #[cfg(test)]
    pub(crate) fn isolate_octave(&mut self, octave: usize) {
        for (i, o) in self.octaves.iter_mut().enumerate() {
            if i != octave {
                o.amplitude = 0.0;
            }
        }
        self.rebuild();
    }

    /// The faded lattice of octave `octave`, channel 0, for checking the fade.
    #[cfg(test)]
    pub(crate) fn lattice_now(&self, octave: usize) -> Vec<f32> {
        let mut out = Vec::new();
        fade_into(&self.octaves[octave], 0, &mut out);
        out
    }

    #[cfg(test)]
    pub(crate) fn lattice_spacing(&self, octave: usize) -> usize {
        self.octaves[octave].spacing
    }
}

/// `6` in `E|u|^2 = 6 sigma^2 a^2 S^2 D / h^2`: three velocity components, each the
/// difference of two independent potential derivatives.
const S_D_FACTOR: f64 = 6.0;

/// The octave's lattice for `channel` at its current fade, times its amplitude.
fn fade_into(o: &Octave, channel: usize, out: &mut Vec<f32>) {
    let t = o.phase * core::f32::consts::FRAC_PI_2;
    let (s, c) = t.sin_cos();
    let (wc, wn) = (c * o.amplitude, s * o.amplitude);
    out.clear();
    out.extend(
        o.current[channel]
            .iter()
            .zip(&o.next[channel])
            .map(|(a, b)| wc * a + wn * b),
    );
}

/// Adds the octave's faded lattice to `level`.
fn add_faded(o: &Octave, channel: usize, level: &mut [f32]) {
    let t = o.phase * core::f32::consts::FRAC_PI_2;
    let (s, c) = t.sin_cos();
    let (wc, wn) = (c * o.amplitude, s * o.amplitude);
    for ((v, a), b) in level
        .iter_mut()
        .zip(&o.current[channel])
        .zip(&o.next[channel])
    {
        *v += wc * a + wn * b;
    }
}

/// Subdivides a 3D lattice of `from` points per axis to `to`, one axis at a time; the
/// result lands in `stage[1]`. Every buffer keeps its capacity, so once the first
/// rebuild has sized them nothing allocates.
fn subdivide_3d(
    src: &[f32],
    from: [usize; 3],
    to: [usize; 3],
    stage: &mut [Vec<f32>; 2],
    lines: &mut [Vec<f32>; 2],
) {
    let [s0, s1] = stage;
    // Along z: (from0, from1, to2).
    s0.clear();
    s0.resize(from[0] * from[1] * to[2], 0.0);
    for line in 0..from[0] * from[1] {
        subdivide_1d(
            &src[line * from[2]..(line + 1) * from[2]],
            &mut s0[line * to[2]..(line + 1) * to[2]],
        );
    }
    // Along y: (from0, to1, to2), a strided line per (x, z).
    s1.clear();
    s1.resize(from[0] * to[1] * to[2], 0.0);
    let [coarse, fine] = lines;
    coarse.clear();
    coarse.resize(from[1], 0.0);
    fine.clear();
    fine.resize(to[1], 0.0);
    for x in 0..from[0] {
        for z in 0..to[2] {
            for (y, c) in coarse.iter_mut().enumerate() {
                *c = s0[(x * from[1] + y) * to[2] + z];
            }
            subdivide_1d(coarse, fine);
            for (y, f) in fine.iter().enumerate() {
                s1[(x * to[1] + y) * to[2] + z] = *f;
            }
        }
    }
    // Along x: whole (y, z) planes at a time, so the inner loop is contiguous.
    let plane = to[1] * to[2];
    s0.clear();
    s0.resize(to[0] * plane, 0.0);
    for q in 0..to[0] {
        let full = q + 2;
        let m = full / 2;
        let out = &mut s0[q * plane..(q + 1) * plane];
        if full % 2 == 0 {
            let (a, b, c) = (
                &s1[(m - 1) * plane..m * plane],
                &s1[m * plane..(m + 1) * plane],
                &s1[(m + 1) * plane..(m + 2) * plane],
            );
            for p in 0..plane {
                out[p] = (a[p] + 6.0 * b[p] + c[p]) * 0.125;
            }
        } else {
            let (a, b) = (
                &s1[m * plane..(m + 1) * plane],
                &s1[(m + 1) * plane..(m + 2) * plane],
            );
            for p in 0..plane {
                out[p] = (a[p] + b[p]) * 0.5;
            }
        }
    }
    std::mem::swap(s0, s1);
}
