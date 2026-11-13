//! A sparse-tile linear fade: a byte field that relaxes to zero at a fixed rate, visiting
//! only the tiles that still hold anything.
//!
//! The law is the simplest relaxation there is, `dh/dt = -rate` clamped at zero, on a
//! `u8` field cut into [`TILE`]-texel tiles (the same tiles as
//! [`crate::fluid_dynamics::TileFilm`]). Ridgeline uses it for the ridge a burrowing unit
//! throws up: loose soil standing proud that settles back over a fixed life. It fades the
//! amplitude **in place**; it does not move material to neighbours, so it is not an
//! angle-of-repose slump and the footprint never widens. Moved here from Ridgeline's
//! `terrain_stain` (STAIN-MOVE, design note `docs/design/STAIN_MOVE.md`); the arithmetic
//! is unchanged, and Ridgeline tests it step for step against a frozen copy of the
//! original.
//!
//! - **A banked rate, not an interval.** A field that loses 255 units over a life of
//!   many steps loses a fraction of a unit a step, and an integer decrement of a fraction
//!   is nothing. Shedding one unit every N steps rounds the life to a multiple of N, which
//!   can be far off (Ridgeline's 4.71 would be 17.0 s or 21.3 s against 20.0). So the
//!   fade owes the fraction in one `f32` for the whole field and spends it the moment it
//!   adds up to a whole unit: exact in the long run, and on most steps the owed amount is
//!   under one and [`TileFade::step`] returns before it looks at a tile.
//! - **Sparse.** A tile is flagged by [`TileFade::raise`] when the caller writes into it
//!   and cleared by the step that takes its last texel to zero, so the cost is the
//!   features standing on the field now, not every feature there has ever been.
//! - **Branchless rows.** Inside a flagged tile every texel takes a `saturating_sub` and
//!   the row's `or` decides whether the tile still stands. Skipping texels already at
//!   zero spends a branch per texel and stops the row vectorising; Ridgeline measured the
//!   skip at 0.977 ms a step against 0.172 over 150 ridges.
//!
//! `step` allocates nothing: the flags are sized in [`TileFade::new`].

use super::film_tiles::TILE;

/// What one [`TileFade::step`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FadeStep {
    /// The step spent at least one unit on at least one flagged tile.
    pub moved: bool,
    /// Bounding box of the tiles the step visited, `[min, max)` in texels; empty
    /// (`min > max`) when nothing moved. Every visited tile is in it, including one whose
    /// last texel reached zero on an earlier step: that is the step that clears its flag,
    /// and it costs one extra tile in the box rather than a branch per texel.
    pub dirty_min: (usize, usize),
    pub dirty_max: (usize, usize),
    /// Units taken off every texel this step (zero on a step that only banked).
    pub drop: u8,
    /// Tiles the step visited.
    pub tiles: usize,
}

/// The sparse state of a fading `u8` field: which tiles hold anything, and the rate owed
/// but not yet spent. Owns no texels: [`TileFade::step`] fades the caller's slice.
#[derive(Clone, Debug)]
pub struct TileFade {
    width: usize,
    height: usize,
    tiles_x: usize,
    standing: Vec<bool>,
    /// Units the fade accrues a step.
    per_step: f32,
    /// Units accrued and not yet spent, always in `[0, 1)` after a step.
    owed: f32,
}

impl TileFade {
    /// A flat field of `width` x `height` texels that, once raised, loses `per_step` units
    /// a step. A field that should fade from `full` to nothing over `life` seconds at a
    /// step of `dt` passes `full / life * dt`, in that order (it is the order Ridgeline's
    /// stain evaluates, and the fade is identical only if the `f32` is).
    pub fn new(width: usize, height: usize, per_step: f32) -> TileFade {
        let tiles_x = width.div_ceil(TILE);
        let tiles_y = height.div_ceil(TILE);
        TileFade {
            width,
            height,
            tiles_x,
            standing: vec![false; tiles_x * tiles_y],
            per_step,
            owed: 0.0,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// Units the fade accrues a step.
    pub fn per_step(&self) -> f32 {
        self.per_step
    }

    /// Flag the tile holding `(x, y)`: the caller raised a texel there. Cheap enough to
    /// call per texel written.
    #[inline]
    pub fn raise(&mut self, x: usize, y: usize) {
        self.standing[(y / TILE) * self.tiles_x + x / TILE] = true;
    }

    /// The tile flags, row-major, [`TILE`]-texel tiles. A tile is flagged from the
    /// [`Self::raise`] that wrote into it until the step that takes its last texel to zero.
    pub fn standing(&self) -> &[bool] {
        &self.standing
    }

    /// Whether any tile still holds anything.
    pub fn any_standing(&self) -> bool {
        self.standing.iter().any(|&s| s)
    }

    /// Units accrued and not yet spent.
    pub fn owed(&self) -> f32 {
        self.owed
    }

    /// Set what is owed, for a caller restoring saved state or a test forcing a step that
    /// spends (owed `1.0` makes the next step drop at least one unit).
    pub fn set_owed(&mut self, owed: f32) {
        self.owed = owed;
    }

    /// Every flag down and nothing owed. For a field that was cleared.
    pub fn clear(&mut self) {
        self.standing.fill(false);
        self.owed = 0.0;
    }

    /// One step of the fade over `field` (`width * height` bytes, row-major): accrue
    /// [`Self::per_step`], and if a whole unit is owed, take it off every texel of every
    /// flagged tile (saturating at zero) and clear the flag of any tile left at zero.
    pub fn step(&mut self, field: &mut [u8]) -> FadeStep {
        debug_assert_eq!(field.len(), self.width * self.height);
        let mut report = FadeStep {
            dirty_min: (self.width, self.height),
            ..FadeStep::default()
        };
        self.owed += self.per_step;
        let drop = self.owed as u8;
        if drop == 0 {
            return report;
        }
        self.owed -= drop as f32;
        report.drop = drop;

        for (t, flag) in self.standing.iter_mut().enumerate() {
            if !*flag {
                continue;
            }
            let (tx, ty) = (t % self.tiles_x, t / self.tiles_x);
            let (x0, y0) = (tx * TILE, ty * TILE);
            let x1 = (x0 + TILE).min(self.width);
            let y1 = (y0 + TILE).min(self.height);

            // Branchless over the row (see the module note): one `saturating_sub` a lane,
            // and an `or` reduction for whether anything is left.
            let mut standing = false;
            for y in y0..y1 {
                let row = y * self.width;
                let mut any = 0u8;
                for h in &mut field[row + x0..row + x1] {
                    *h = h.saturating_sub(drop);
                    any |= *h;
                }
                standing |= any != 0;
            }
            *flag = standing;
            report.tiles += 1;
            report.moved = true;
            report.dirty_min.0 = report.dirty_min.0.min(x0);
            report.dirty_min.1 = report.dirty_min.1.min(y0);
            report.dirty_max.0 = report.dirty_max.0.max(x1);
            report.dirty_max.1 = report.dirty_max.1.max(y1);
        }
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 255 units over 20 s at 60 Hz: Ridgeline's ridge.
    const PER_STEP: f32 = 255.0 / 20.0 * (1.0 / 60.0);

    fn field(width: usize, height: usize) -> (TileFade, Vec<u8>) {
        (TileFade::new(width, height, PER_STEP), vec![0u8; width * height])
    }

    fn raise(fade: &mut TileFade, f: &mut [u8], width: usize, x: usize, y: usize, h: u8) {
        f[y * width + x] = h;
        fade.raise(x, y);
    }

    #[test]
    fn a_full_byte_is_gone_in_its_life_and_not_before() {
        let (w, h) = (65, 65);
        let (mut fade, mut f) = field(w, h);
        raise(&mut fade, &mut f, w, 32, 32, 255);
        let life = (20.0f32 * 60.0).ceil() as usize;
        for _ in 0..life * 9 / 10 {
            fade.step(&mut f);
        }
        let left = f[32 * w + 32];
        assert!((20..=30).contains(&left), "nine tenths through its life the byte is {left}, not about 25");
        for _ in 0..life - life * 9 / 10 + 2 {
            fade.step(&mut f);
        }
        assert_eq!(f[32 * w + 32], 0, "the byte outlived its life");
        assert!(!fade.any_standing(), "the last texel is zero and its tile is still flagged");
    }

    #[test]
    fn the_rate_is_banked_and_spent_exactly() {
        // 0.2125 a step: four steps in five bank, the fifth spends one unit.
        let (mut fade, mut f) = field(32, 32);
        raise(&mut fade, &mut f, 32, 0, 0, 200);
        let mut spent = 0u32;
        let steps = 4800;
        for _ in 0..steps {
            spent += fade.step(&mut f).drop as u32;
            assert!((0.0..1.0).contains(&fade.owed()));
        }
        let expected = PER_STEP * steps as f32;
        assert!((spent as f32 - expected).abs() <= 1.0, "spent {spent} against {expected}");
        assert_eq!(f[0], 200u32.saturating_sub(spent) as u8);
    }

    #[test]
    fn a_step_that_only_banks_touches_nothing() {
        let (mut fade, mut f) = field(64, 64);
        raise(&mut fade, &mut f, 64, 10, 10, 9);
        let step = fade.step(&mut f);
        assert_eq!(step, FadeStep { dirty_min: (64, 64), ..FadeStep::default() });
        assert_eq!(f[10 * 64 + 10], 9);
    }

    #[test]
    fn only_flagged_tiles_fade_and_the_box_holds_them_all() {
        let (w, h) = (100, 70);
        let (mut fade, mut f) = field(w, h);
        raise(&mut fade, &mut f, w, 5, 5, 10);
        raise(&mut fade, &mut f, w, 99, 69, 1);
        // Written without a raise: the fade must not know about it.
        f[40 * w + 40] = 50;
        fade.set_owed(1.0);
        let step = fade.step(&mut f);
        assert!(step.moved);
        assert_eq!(step.drop, 1);
        assert_eq!(step.tiles, 2);
        assert_eq!(step.dirty_min, (0, 0));
        // The far tile is clipped to the field: x 96..100, y 64..70.
        assert_eq!(step.dirty_max, (100, 70));
        assert_eq!(f[5 * w + 5], 9);
        assert_eq!(f[69 * w + 99], 0);
        assert_eq!(f[40 * w + 40], 50, "an unflagged tile faded");
        let (tx, _) = (w.div_ceil(TILE), h.div_ceil(TILE));
        assert!(fade.standing()[0], "a tile with 9 left was cleared");
        assert!(!fade.standing()[2 * tx + 3], "a tile at zero kept its flag");
    }

    #[test]
    fn a_tile_cleared_this_step_is_still_in_the_box_and_then_costs_nothing() {
        let (mut fade, mut f) = field(64, 64);
        raise(&mut fade, &mut f, 64, 40, 40, 1);
        fade.set_owed(1.0);
        let first = fade.step(&mut f);
        assert_eq!((first.dirty_min, first.dirty_max, first.tiles), ((32, 32), (64, 64), 1));
        fade.set_owed(1.0);
        let second = fade.step(&mut f);
        assert!(!second.moved && second.tiles == 0, "a finished tile was walked again");
    }

    #[test]
    fn saturates_at_zero_under_any_drop() {
        let (mut fade, mut f) = field(32, 32);
        raise(&mut fade, &mut f, 32, 3, 3, 2);
        fade.set_owed(5.0);
        let step = fade.step(&mut f);
        assert_eq!(step.drop, 5);
        assert_eq!(f[3 * 32 + 3], 0);
        assert!(!fade.any_standing());
    }

    #[test]
    fn clear_drops_the_flags_and_the_debt() {
        let (mut fade, mut f) = field(64, 32);
        raise(&mut fade, &mut f, 64, 33, 1, 7);
        fade.set_owed(0.9);
        fade.clear();
        assert!(!fade.any_standing());
        assert_eq!(fade.owed(), 0.0);
    }
}
