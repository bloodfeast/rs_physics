//! **The ground as a height field**, for callers whose ground is not a plane and does not
//! hold still.
//!
//! # Why this is borrowed for a step rather than owned by the skeleton
//!
//! Because a height field that a skeleton owned would be a *copy*, and a copy has to be
//! kept up to date. The callers who need this are the ones whose ground is being changed
//! underneath the bodies standing on it -- a shell crater, a collapsing bank, ground that
//! is churned by the same event that knocked the bodies over -- and for them every
//! owned-copy design is the same design: notice the change, work out which cells moved,
//! write them across. That is a synchronisation problem, it is the caller's to get wrong,
//! and it is wrong in the one direction that matters: a stale cell is a body resting on
//! ground that is no longer there.
//!
//! Borrowing removes the problem rather than solving it. [`Skeleton::step_over`] takes the
//! field for exactly the length of one step, reads whatever the caller's own array says
//! right now, and keeps nothing. There is no copy to go stale because there is no copy.
//!
//! The price is a lifetime on a hot signature, which is why [`Skeleton::step`] still exists
//! and still takes the stored plane: a caller whose ground is a plane should not pay for
//! the general case.
//!
//! # The layout
//!
//! Vertex heights, row-major, on a square grid: `columns * rows` of them, spaced `cell`
//! apart, with vertex zero at `origin`. This is the layout a height field has in every
//! program that has one, so for most callers it is the array they already hold and this
//! borrows it without touching it.
//!
//! Sampling **off the edge clamps** rather than wrapping or returning nothing, which makes
//! the border extend outward for ever. A body that walks off the edge of the field keeps
//! ground under it at the height of the edge it left. The alternative -- no ground out
//! there -- is a body that falls for ever, and a solver cannot tell that apart from a bug.

use super::{dot, normalized};

/// A borrowed grid of heights. See the module header for why it is borrowed.
///
/// `Copy`, and `Send + Sync` because it is a shared slice and some numbers, so the parallel
/// solve can hand it to every lane without any of them owning it.
#[derive(Clone, Copy, Debug)]
pub struct Field<'a> {
    heights: &'a [f64],
    columns: usize,
    rows: usize,
    cell: f64,
    origin: (f64, f64),
}

impl<'a> Field<'a> {
    /// A field over `heights`, laid out row-major as `columns * rows` vertices `cell` apart,
    /// with the first vertex at `origin` in the ground plane's two axes.
    ///
    /// `None` rather than a panic for any grid that cannot describe a surface: too few
    /// vertices to span a cell, a spacing that is not a positive finite number, a length
    /// that does not match the shape claimed for it, or any height that is not finite.
    ///
    /// **The height check is worth its cost and the others are free.** A single NaN cell
    /// reaches the solve as a NaN normal, and from there it is one step to a NaN position,
    /// which spreads through every contact that body ever makes and cannot be traced back.
    /// Rejecting the field is the only place the cause is still visible. It is a linear
    /// scan once per step against a solve that is many passes over every body, and a caller
    /// who cannot afford it is a caller who should hoist the field out of the loop.
    pub fn new(
        heights: &'a [f64],
        columns: usize,
        rows: usize,
        cell: f64,
        origin: (f64, f64),
    ) -> Option<Self> {
        if columns < 2 || rows < 2 || !(cell > 0.0 && cell.is_finite()) {
            return None;
        }
        if !origin.0.is_finite() || !origin.1.is_finite() {
            return None;
        }
        if heights.len() != columns.checked_mul(rows)? {
            return None;
        }
        if heights.iter().any(|h| !h.is_finite()) {
            return None;
        }
        Some(Field {
            heights,
            columns,
            rows,
            cell,
            origin,
        })
    }

    /// One vertex, with both indices clamped into the grid. See the module header for why
    /// the border extends rather than ending.
    #[inline]
    fn vertex(&self, ix: i64, iz: i64) -> f64 {
        let ix = ix.clamp(0, self.columns as i64 - 1) as usize;
        let iz = iz.clamp(0, self.rows as i64 - 1) as usize;
        self.heights[iz * self.columns + ix]
    }

    /// The four corners of the cell `(x, z)` falls in, and how far across it the point is.
    #[inline]
    fn cell_at(&self, x: f64, z: f64) -> ([f64; 4], f64, f64) {
        let fx = (x - self.origin.0) / self.cell;
        let fz = (z - self.origin.1) / self.cell;
        // `floor` rather than a cast: a cast truncates toward zero, which folds the cells
        // either side of the origin into one and puts a crease along it.
        let (bx, bz) = (fx.floor(), fz.floor());
        let (tx, tz) = (fx - bx, fz - bz);
        // Saturating rather than wrapping: the clamp in `vertex` handles an index outside
        // the grid, but only if the cast to `i64` did not wrap round first, and a position
        // far enough out for that is one the solver should survive rather than mis-sample.
        let (ix, iz) = (saturate(bx), saturate(bz));
        (
            [
                self.vertex(ix, iz),
                self.vertex(ix + 1, iz),
                self.vertex(ix, iz + 1),
                self.vertex(ix + 1, iz + 1),
            ],
            tx,
            tz,
        )
    }

    /// The surface height at `(x, z)`, bilinearly across the cell.
    pub fn height_at(&self, x: f64, z: f64) -> f64 {
        let ([h00, h10, h01, h11], tx, tz) = self.cell_at(x, z);
        let low = h00 + (h10 - h00) * tx;
        let high = h01 + (h11 - h01) * tx;
        low + (high - low) * tz
    }

    /// **The plane tangent to the surface under `(x, z)`**, as the `(normal, distance)` pair
    /// the rest of this module means by a ground: the plane `dot(normal, p) = distance`,
    /// with the normal pointing up out of the ground.
    ///
    /// # Why the gradient is exact rather than a difference
    ///
    /// The surface between four vertices *is* the bilinear patch, so its slope there has a
    /// closed form and there is nothing to approximate. A central difference would need an
    /// epsilon, and every epsilon is wrong at one of the two ends: small enough to stay
    /// inside the cell and it is differencing two numbers that agree to most of their
    /// digits; large enough to be well conditioned and it has crossed into the next cell and
    /// is reporting a slope the body is not on. The closed form has neither failure and
    /// costs less than the difference, because it reuses the four corners the height has
    /// already fetched.
    ///
    /// The seam between two cells is the one place this is not the surface's own slope,
    /// because a bilinear field has a crease there. What a body gets is the slope of the
    /// cell it is over, which changes as it crosses -- continuously in the height, in a step
    /// in the normal. That step is bounded by how much the ground itself turns, which is the
    /// one bound that matters: **a normal that jumps is what makes a body walk**, and here it
    /// jumps by the angle between two adjacent cells rather than by anything the solver
    /// invented.
    pub fn plane_at(&self, x: f64, z: f64) -> ((f64, f64, f64), f64) {
        let ([h00, h10, h01, h11], tx, tz) = self.cell_at(x, z);
        let low = h00 + (h10 - h00) * tx;
        let high = h01 + (h11 - h01) * tx;
        let height = low + (high - low) * tz;
        // The bilinear patch's own derivatives, divided by the cell to turn a step across
        // the cell into a step in world units.
        let along_x = ((h10 - h00) * (1.0 - tz) + (h11 - h01) * tz) / self.cell;
        let along_z = ((h01 - h00) * (1.0 - tx) + (h11 - h10) * tx) / self.cell;
        // The normal of a height field `y = h(x, z)` is `(-dh/dx, 1, -dh/dz)`. It cannot
        // fail to normalise -- the `y` component is one before scaling -- but `normalized`
        // is what the rest of the module uses, and a level plane is the right answer to a
        // degenerate one.
        let normal = normalized((-along_x, 1.0, -along_z)).unwrap_or((0.0, 1.0, 0.0));
        (normal, dot(normal, (x, height, z)))
    }
}

/// A floored coordinate as an index, without the wrap a bare cast does at the extremes.
#[inline]
fn saturate(v: f64) -> i64 {
    if v.is_nan() {
        0
    } else if v >= i64::MAX as f64 {
        i64::MAX
    } else if v <= i64::MIN as f64 {
        i64::MIN
    } else {
        v as i64
    }
}
