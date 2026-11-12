//! A sparse-tile thin-film step: the lubrication law on a fixed-point depth field,
//! visiting only the tiles that hold liquid.
//!
//! [`crate::fluid_dynamics::FilmGrid`] is the reference solver: dense, `f64`, every cell
//! every step. That is the right shape for a validation and the wrong one for a game's
//! stain buffer, where a fight wets a few percent of a 1400 x 1000 map and the rest is
//! dry. This is the other shape, moved here from Ridgeline's `terrain_stain` (STAIN-MOVE,
//! design note `docs/design/STAIN_MOVE.md`):
//!
//! - **Sparse.** The field is cut into [`TILE`]-texel tiles. A step visits the tiles
//!   flagged active and nothing else; a tile goes to sleep when nothing in it moved and is
//!   woken by a neighbour whose liquid reached the seam, or by the caller
//!   ([`TileFilm::wake`]) when it lands liquid.
//! - **Fixed-point.** Depth is an `i32` of sub-units whose high byte is the display depth.
//!   The caller owns the storage ([`FilmField`]), so a renderer that samples that byte
//!   needs no copy and no quantisation step.
//! - **Eight neighbours.** Liquid leaves a texel toward every downhill neighbour in
//!   proportion to the drop over the distance ([`INV_DISTANCE`]): the multiple-flow-
//!   direction rule, which spreads a spill round rather than as a diamond.
//! - **Two passes, order-free.** Pass one writes one [`TileFlux`] per texel from the old
//!   depth; pass two recomputes each sender's share from the drop and the sender's
//!   carried reciprocal depth, so what leaves is what arrives and no texel's result
//!   depends on traversal order. That is also what makes it a compute shader.
//!
//! The rate is the Nusselt film flux `q = rho g S h^3 / (3 mu)` as a share of the
//! reference film's Courant number, with the Bingham plug correction (it tapers to zero as
//! the wall shear falls to the yield stress). The threshold a sender holds back is either
//! the contact-line retention of the ground under it ([`FilmField::retention`]) or, toward
//! a neighbour already wet, the fluid's yield head; both scale with one over the depth.
//!
//! `step` allocates nothing: every buffer is sized in [`TileFilm::new`].

use std::time::Instant;

/// Texels on a tile's side.
pub const TILE: usize = 32;

/// Reciprocal of the distance to each of the eight neighbours, in texels, in the order
/// every function here uses: E, W, S, N, SE, NE, SW, NW.
///
/// This is what makes the spread round rather than square: a diagonal neighbour with the
/// same drop offers a gentler slope by exactly the factor its distance implies.
pub const INV_DISTANCE: [f32; 8] = [
    1.0,
    1.0,
    1.0,
    1.0,
    std::f32::consts::FRAC_1_SQRT_2,
    std::f32::consts::FRAC_1_SQRT_2,
    std::f32::consts::FRAC_1_SQRT_2,
    std::f32::consts::FRAC_1_SQRT_2,
];

/// The film law, fixed per fluid and grid, in the field's own units: depth in display
/// units (`subunits` of storage each), heads in display units per texel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileFilmLaw {
    /// Share of a texel the reference film moves in one step: `u(h_ref, S_ref) dt / dx`.
    pub courant: f32,
    /// Display units of head per texel, as a fraction of the reference grade.
    pub grade_per_head: f32,
    /// Reference grade times reference depth: `grade * film * drive_gain` is the wall
    /// shear over `rho g`, in metres.
    pub drive_gain: f32,
    /// `tau_y / (rho g)`, metres: the shear length the drive must beat to move at all.
    pub yield_length: f32,
    /// The wet-to-wet threshold at unit reciprocal depth (the yield stress as a head).
    pub yield_head: f32,
    /// Display depth at or below which nothing is loose.
    pub film: u8,
    /// Storage units per display unit: a power of two, so the display byte is a shift.
    pub subunits: i32,
    /// Display units per stored fall unit.
    pub fall_gain: f32,
    /// The constant-threshold arm, for an A/B of the retention law; `None` in play.
    pub legacy_cohesion: Option<f32>,
}

impl TileFilmLaw {
    /// The reciprocal of a depth in display units: the one divide the law costs.
    ///
    /// `depth` must be positive; the step only calls this above [`Self::film`].
    #[inline(always)]
    pub fn inv_depth(&self, depth: i32) -> f32 {
        self.subunits as f32 / depth as f32
    }

    /// The head a sender holds back toward one neighbour, in display units per texel.
    ///
    /// `scale` is the sender's [`Self::inv_depth`] (thinned), `retention` the ground's
    /// contact-line head under the sender, `neighbour_wet` whether the receiver held
    /// liquid at the start of the step: toward a wet neighbour there is no contact line
    /// and only the yield stress resists.
    #[inline(always)]
    pub fn threshold(&self, scale: f32, retention: f32, neighbour_wet: bool) -> f32 {
        if let Some(legacy) = self.legacy_cohesion {
            legacy
        } else if neighbour_wet {
            self.yield_head * scale
        } else {
            retention * scale
        }
    }

    /// The share of a texel's depth that leaves it this step, given its steepest
    /// thresholded slope and its depth: Nusselt cube, Bingham plug, Courant cap.
    /// `None` when it does not move.
    #[inline(always)]
    pub fn outflow(&self, steepest: f32, depth: i32) -> Option<f32> {
        let film = depth as f32 / (255.0 * self.subunits as f32);
        let grade = steepest * self.grade_per_head;
        let drive = grade * film * self.drive_gain;
        if drive <= self.yield_length {
            return None;
        }
        let x = self.yield_length / drive;
        let plug = 1.0 - 1.5 * x + 0.5 * x * x * x;
        Some((self.courant * grade * film * film * plug).min(self.courant))
    }
}

/// What the caller stores and what its ground does.
///
/// The step is generic over this, so every call monomorphises and inlines: the trait is
/// the seam between the law and a particular buffer layout, not a dynamic dispatch.
pub trait FilmField {
    /// Depth at `cell`, in sub-units.
    fn depth(&self, cell: usize) -> i32;
    /// Write a depth, clamped to what the storage holds.
    fn set_depth(&mut self, cell: usize, depth: i32);
    /// The display byte of the depth (its high byte).
    fn display(&self, cell: usize) -> u8;
    /// Whether any texel of row `y` in `x0..x1` holds any depth at all.
    fn row_wet(&self, y: usize, x0: usize, x1: usize) -> bool;
    /// The liquid surface (depth plus whatever the ground adds or removes, in display
    /// units) and whether it is wet (display depth above the film), for row `y`,
    /// `x0..x1`, into `surface` and `wet` (both `x1 - x0` long).
    fn snapshot_row(&self, y: usize, x0: usize, x1: usize, surface: &mut [i16], wet: &mut [bool]);
    /// Contact-line head of the ground under `cell`, at unit reciprocal depth.
    fn retention(&self, cell: usize) -> f32;
    /// Fraction of its yield a film of `depth` keeps at `(x, y)` (1 with nothing mixed in).
    fn thinning(&self, x: usize, y: usize, depth: i32) -> f32;
    /// Whether a film at rest at `(x, y)` may dry this step.
    fn may_dry(&self, x: usize, y: usize) -> bool;
    /// Liquid arrived at `cell` this step; the ground drinks what it will. Returns the
    /// sub-units drunk (the step subtracts them). See [`crate::fluid_dynamics::infiltration`].
    fn drink(&mut self, cell: usize, arrived: f32) -> i32;
}

/// One texel's outgoing flux for a step: how much leaves, what to divide a neighbour's
/// share by, and the sender's reciprocal depth so the receiver can reproduce its threshold.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct TileFlux {
    /// Depth leaving this texel this step, in sub-units.
    pub magnitude: f32,
    /// Reciprocal of the sum of the downhill thresholded slopes.
    pub inv_total: f32,
    /// The sender's thinned reciprocal depth.
    pub inv_depth: f32,
}

impl TileFlux {
    /// Nothing is moving here.
    pub const STILL: TileFlux = TileFlux { magnitude: 0.0, inv_total: 0.0, inv_depth: 0.0 };
}

/// What one [`TileFilm::step`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TileStep {
    /// A display byte changed somewhere: the picture moved.
    pub moved: bool,
    /// Bounding box of the tiles whose display bytes changed, `[min, max)` in texels;
    /// empty (`min > max`) when nothing moved.
    pub dirty_min: (usize, usize),
    pub dirty_max: (usize, usize),
    /// Tiles the step visited.
    pub tiles: usize,
    /// Texels the ground drank from.
    pub soaks: u64,
    /// Wall clock of the two passes, nanoseconds.
    pub flux_ns: u64,
    pub gather_ns: u64,
}

/// The sparse state of a film: tile flags, the flux and surface snapshot buffers and the
/// live tile list. Owns no depth: that is the caller's [`FilmField`].
#[derive(Clone, Debug)]
pub struct TileFilm {
    width: usize,
    height: usize,
    tiles_x: usize,
    tiles_y: usize,
    active: Vec<bool>,
    liquid: Vec<bool>,
    flux: Vec<TileFlux>,
    surface: Vec<i16>,
    wet: Vec<bool>,
    live: Vec<usize>,
}

impl TileFilm {
    /// A dry film over a `width` x `height` field. Every buffer the step uses is sized here.
    pub fn new(width: usize, height: usize) -> TileFilm {
        let tiles_x = width.div_ceil(TILE);
        let tiles_y = height.div_ceil(TILE);
        TileFilm {
            width,
            height,
            tiles_x,
            tiles_y,
            active: vec![false; tiles_x * tiles_y],
            liquid: vec![false; tiles_x * tiles_y],
            flux: vec![TileFlux::STILL; width * height],
            surface: vec![0; width * height],
            wet: vec![false; width * height],
            live: Vec::with_capacity(tiles_x * tiles_y),
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// Tiles across and down.
    pub fn tiles(&self) -> (usize, usize) {
        (self.tiles_x, self.tiles_y)
    }

    /// Wake the tile `(x, y)` is in and the eight around it, and mark it as holding
    /// liquid. A caller that lands liquid calls this; the neighbours matter because liquid
    /// against a seam leaves through the next tile's pass.
    #[inline]
    pub fn wake(&mut self, x: usize, y: usize) {
        let tx = x / TILE;
        let ty = y / TILE;
        for ny in ty.saturating_sub(1)..(ty + 2).min(self.tiles_y) {
            for nx in tx.saturating_sub(1)..(tx + 2).min(self.tiles_x) {
                self.active[ny * self.tiles_x + nx] = true;
            }
        }
        self.liquid[ty * self.tiles_x + tx] = true;
    }

    /// Whether the tile holding `(x, y)` held liquid after the last step.
    #[inline]
    pub fn holds_liquid(&self, x: usize, y: usize) -> bool {
        self.liquid[(y / TILE) * self.tiles_x + (x / TILE)]
    }

    /// Whether any tile is active (will be visited by the next non-drying step).
    pub fn any_active(&self) -> bool {
        self.active.iter().any(|&a| a)
    }

    /// Whether any tile holds liquid.
    pub fn any_liquid(&self) -> bool {
        self.liquid.iter().any(|&l| l)
    }

    /// Tiles in the whole field.
    pub fn tile_count(&self) -> usize {
        self.active.len()
    }

    /// The liquid surface snapshot the last step took, in display units.
    pub fn surface(&self) -> &[i16] {
        &self.surface
    }

    /// Every tile asleep and dry, every buffer zeroed. For a field that was cleared.
    pub fn clear(&mut self) {
        self.active.fill(false);
        self.liquid.fill(false);
        self.flux.fill(TileFlux::STILL);
        self.surface.fill(0);
        self.wet.fill(false);
        self.live.clear();
    }

    /// One step over every active tile. `drying` (the caller's clock decides, once in
    /// however many steps its film should take to dry) also visits every tile holding
    /// liquid and dries one sub-unit's worth of display depth off films at rest below the
    /// law's `film`. `fall_x[c]` is the stored drop from `c` to `c + 1`, `fall_z[c]` from
    /// `c` to `c + width`.
    pub fn step<F: FilmField>(
        &mut self,
        field: &mut F,
        law: &TileFilmLaw,
        fall_x: &[i8],
        fall_z: &[i8],
        drying: bool,
    ) -> TileStep {
        let mut report = TileStep {
            dirty_min: (self.width, self.height),
            dirty_max: (0, 0),
            ..TileStep::default()
        };
        if drying {
            for (active, liquid) in self.active.iter_mut().zip(self.liquid.iter()) {
                *active |= *liquid;
            }
        }
        // Snapshot the live set before either pass: the gather wakes tiles, and a tile
        // woken this step must not be gathered before its flux exists.
        let mut live = std::mem::take(&mut self.live);
        live.clear();
        live.extend((0..self.active.len()).filter(|&t| self.active[t]));
        report.tiles = live.len();

        let started = Instant::now();
        for &t in &live {
            self.flux_tile(t, field, law, fall_x, fall_z);
        }
        report.flux_ns = started.elapsed().as_nanos() as u64;

        let started = Instant::now();
        for &t in &live {
            self.active[t] = false;
            let (moved, stirred, edge, liquid, soaks) =
                self.gather_tile(t, drying, field, law, fall_x, fall_z);
            self.liquid[t] = liquid;
            report.soaks += soaks;
            let (tx, ty) = (t % self.tiles_x, t / self.tiles_x);
            let (x0, y0) = (tx * TILE, ty * TILE);
            let x1 = (x0 + TILE - 1).min(self.width - 1);
            let y1 = (y0 + TILE - 1).min(self.height - 1);
            if stirred {
                self.active[t] = true;
            }
            if edge {
                // Only when liquid changed on the tile's outer ring: liquid crosses at
                // most one texel a step, so an interior change cannot reach next door.
                self.wake(x0, y0);
                self.wake(x1, y1);
            }
            if moved {
                report.moved = true;
                report.dirty_min.0 = report.dirty_min.0.min(x0);
                report.dirty_min.1 = report.dirty_min.1.min(y0);
                report.dirty_max.0 = report.dirty_max.0.max(x1 + 1);
                report.dirty_max.1 = report.dirty_max.1.max(y1 + 1);
            }
        }
        report.gather_ns = started.elapsed().as_nanos() as u64;
        self.live = live;
        report
    }

    /// Surface height of each of the eight neighbours of `(x, y)` less its own, in
    /// display units: the terrain rise from the stored falls plus the liquid snapshot.
    /// Off the field reads as level.
    #[inline]
    fn rises(&self, x: usize, y: usize, law: &TileFilmLaw, fall_x: &[i8], fall_z: &[i8]) -> [f32; 8] {
        let w = self.width;
        let cell = y * w + x;
        let g = law.fall_gain;

        let east = x + 1 < w;
        let west = x >= 1;
        let south = y + 1 < self.height;
        let north = y >= 1;

        let tx_e = if east { -(fall_x[cell] as f32) * g } else { 0.0 };
        let tx_w = if west { fall_x[cell - 1] as f32 * g } else { 0.0 };
        let tz_s = if south { -(fall_z[cell] as f32) * g } else { 0.0 };
        let tz_n = if north { fall_z[cell - w] as f32 * g } else { 0.0 };
        // A diagonal's vertical step is taken in the column it lands in, so the diagonal
        // is the sum of the two axis steps that reach it (exact for a bilinear surface).
        let tz_s_e = if south && east { -(fall_z[cell + 1] as f32) * g } else { 0.0 };
        let tz_n_e = if north && east { fall_z[cell + 1 - w] as f32 * g } else { 0.0 };
        let tz_s_w = if south && west { -(fall_z[cell - 1] as f32) * g } else { 0.0 };
        let tz_n_w = if north && west { fall_z[cell - 1 - w] as f32 * g } else { 0.0 };

        // The snapshot, never the live depth: pass two writes in place.
        let s = &self.surface[..];
        let here = s[cell] as f32;
        // The index is computed only if the neighbour exists (no underflow on row 0).
        let liquid = |c: Option<usize>| c.map_or(0.0, |c| s[c] as f32 - here);

        [
            tx_e + liquid(east.then(|| cell + 1)),
            tx_w + liquid(west.then(|| cell - 1)),
            tz_s + liquid(south.then(|| cell + w)),
            tz_n + liquid(north.then(|| cell - w)),
            tx_e + tz_s_e + liquid((east && south).then(|| cell + w + 1)),
            tx_e + tz_n_e + liquid((east && north).then(|| cell + 1 - w)),
            tx_w + tz_s_w + liquid((west && south).then(|| cell + w - 1)),
            tx_w + tz_n_w + liquid((west && north).then(|| cell - w - 1)),
        ]
    }

    /// Whether each neighbour held liquid at the start of the step, in [`INV_DISTANCE`]
    /// order. Off the field reads as dry.
    #[inline]
    fn wet_neighbours(&self, x: usize, y: usize) -> [bool; 8] {
        let w = self.width;
        let cell = y * w + x;
        let east = x + 1 < w;
        let west = x >= 1;
        let south = y + 1 < self.height;
        let north = y >= 1;
        let at = |ok: bool, c: usize| ok && self.wet[c];
        [
            at(east, cell + 1),
            at(west, cell.wrapping_sub(1)),
            at(south, cell + w),
            at(north, cell.wrapping_sub(w)),
            at(east && south, cell + w + 1),
            at(east && north, (cell + 1).wrapping_sub(w)),
            at(south && west, (cell + w).wrapping_sub(1)),
            at(north && west, cell.wrapping_sub(w + 1)),
        ]
    }

    /// The eight neighbours of a texel, in [`INV_DISTANCE`] order, or `None` off the field.
    #[inline]
    fn neighbours(cell: usize, x: usize, y: usize, w: usize, h: usize) -> [Option<usize>; 8] {
        let east = x + 1 < w;
        let west = x >= 1;
        let south = y + 1 < h;
        let north = y >= 1;
        [
            east.then(|| cell + 1),
            west.then(|| cell - 1),
            south.then(|| cell + w),
            north.then(|| cell - w),
            (east && south).then(|| cell + w + 1),
            (east && north).then(|| cell + 1 - w),
            (west && south).then(|| cell + w - 1),
            (west && north).then(|| cell - w - 1),
        ]
    }

    /// Pass one: how much leaves each texel of the tile and its one-texel halo, and how
    /// that is shared out. Reads the old depth; writes only `flux` and the snapshot.
    fn flux_tile<F: FilmField>(
        &mut self,
        tile: usize,
        field: &F,
        law: &TileFilmLaw,
        fall_x: &[i8],
        fall_z: &[i8],
    ) {
        let (tx, ty) = (tile % self.tiles_x, tile / self.tiles_x);
        let x0 = (tx * TILE).saturating_sub(1);
        let x1 = (tx * TILE + TILE + 1).min(self.width);
        let y0 = (ty * TILE).saturating_sub(1);
        let y1 = (ty * TILE + TILE + 1).min(self.height);

        // The surface two texels out, before any of it is read: flux over the halo asks
        // the halo's neighbours how high they are. Rewritten every step, never skipped,
        // so no stale height from a texel that dried survives as a phantom neighbour.
        {
            let sx0 = (tx * TILE).saturating_sub(2);
            let sx1 = (tx * TILE + TILE + 2).min(self.width);
            let sy0 = (ty * TILE).saturating_sub(2);
            let sy1 = (ty * TILE + TILE + 2).min(self.height);
            for y in sy0..sy1 {
                let row = y * self.width;
                field.snapshot_row(
                    y,
                    sx0,
                    sx1,
                    &mut self.surface[row + sx0..row + sx1],
                    &mut self.wet[row + sx0..row + sx1],
                );
            }
        }

        for y in y0..y1 {
            for x in x0..x1 {
                let cell = y * self.width + x;
                if field.display(cell) <= law.film {
                    self.flux[cell] = TileFlux::STILL;
                    continue;
                }
                let depth = field.depth(cell);
                // Thinned by whatever is mixed into it, folded into the depth term so the
                // receiver sees the same thinning through `inv_depth`.
                let inv_depth = law.inv_depth(depth) * field.thinning(x, y, depth);
                let retention = field.retention(cell);

                let rises = self.rises(x, y, law, fall_x, fall_z);
                let wet = self.wet_neighbours(x, y);
                let mut total = 0.0f32;
                let mut steepest = 0.0f32;
                for (i, rise) in rises.iter().enumerate() {
                    let slope = -rise * INV_DISTANCE[i] - law.threshold(inv_depth, retention, wet[i]);
                    if slope > 0.0 {
                        total += slope;
                        steepest = steepest.max(slope);
                    }
                }
                if total <= 0.0 {
                    self.flux[cell] = TileFlux::STILL;
                    continue;
                }
                let Some(share) = law.outflow(steepest, depth) else {
                    self.flux[cell] = TileFlux::STILL;
                    continue;
                };
                // Never more than is loose: what is left at the film clings.
                let loose = (depth - law.film as i32 * law.subunits).max(0) as f32;
                let magnitude = (share * depth as f32).min(loose);
                // Below a sub-unit the storage cannot hold the change.
                if magnitude < 1.0 {
                    self.flux[cell] = TileFlux::STILL;
                    continue;
                }
                self.flux[cell] = TileFlux { magnitude, inv_total: 1.0 / total, inv_depth };
            }
        }
    }

    /// Pass two: what arrives at each texel of the tile, what the ground drinks of it, and
    /// the drying. Returns (display changed, any sub-unit flowed, flowed on the outer
    /// ring, holds liquid, texels drunk from).
    fn gather_tile<F: FilmField>(
        &mut self,
        tile: usize,
        drying: bool,
        field: &mut F,
        law: &TileFilmLaw,
        fall_x: &[i8],
        fall_z: &[i8],
    ) -> (bool, bool, bool, bool, u64) {
        let (tx, ty) = (tile % self.tiles_x, tile / self.tiles_x);
        let x0 = tx * TILE;
        let x1 = (x0 + TILE).min(self.width);
        let y0 = ty * TILE;
        let y1 = (y0 + TILE).min(self.height);

        let w = self.width;
        let h = self.height;
        let mut moved = false;
        let mut stirred = false;
        let mut edge = false;
        let mut any_liquid = false;
        let mut soaks = 0u64;
        let film_depth = law.film as i32 * law.subunits;
        let shift = law.subunits.trailing_zeros();

        let sx0 = x0.saturating_sub(1);
        let sx1 = (x1 + 1).min(w);

        for y in y0..y1 {
            // Which columns anything can arrive at: three rows of the flux buffer as
            // slices, which vectorises, then three bools per texel.
            let mut column: [bool; TILE + 2] = [false; TILE + 2];
            {
                let mut mark = |at: usize| {
                    for (i, f) in self.flux[at * w + sx0..at * w + sx1].iter().enumerate() {
                        column[i] |= f.magnitude != 0.0;
                    }
                };
                mark(y);
                if y >= 1 {
                    mark(y - 1);
                }
                if y + 1 < h {
                    mark(y + 1);
                }
            }
            let nearby = column.iter().any(|c| *c);
            let reaches = |x: usize| {
                let i = x - sx0;
                column[i.saturating_sub(1)] || column[i] || column[i + 1]
            };

            if !nearby {
                if drying && field.row_wet(y, x0, x1) {
                    // fall through to the drying loop below
                } else {
                    any_liquid |= field.row_wet(y, x0, x1);
                    continue;
                }
            }

            for x in x0..x1 {
                let cell = y * w + x;
                let before = field.depth(cell);
                let mut depth = before;

                let mut arrived = 0.0f32;
                if nearby && reaches(x) {
                    let rises = self.rises(x, y, law, fall_x, fall_z);
                    let neighbours = Self::neighbours(cell, x, y, w, h);
                    let wet = self.wet[cell];
                    for (i, from) in neighbours.iter().enumerate() {
                        let Some(from) = *from else { continue };
                        let f = self.flux[from];
                        if f.magnitude == 0.0 {
                            continue;
                        }
                        // The sender's threshold, from the sender's carried depth and the
                        // ground under the sender: the share it divided by.
                        let slope = rises[i] * INV_DISTANCE[i]
                            - law.threshold(f.inv_depth, field.retention(from), wet);
                        if slope > 0.0 {
                            arrived += f.magnitude * slope * f.inv_total;
                        }
                    }
                    depth += (arrived - self.flux[cell].magnitude) as i32;
                }

                // Running liquid soaks into what it runs over.
                if arrived > 0.0 {
                    let drunk = field.drink(cell, arrived);
                    if drunk > 0 {
                        soaks += 1;
                        depth -= drunk;
                    }
                }

                let flowed = depth != before;
                // A film too thin to stand dries where it lies, only where nothing is
                // arriving (or a rim fed from behind would never stop drying).
                if drying && arrived == 0.0 && depth > 0 && depth < film_depth && field.may_dry(x, y) {
                    depth = (depth - law.subunits).max(0);
                }

                if depth != before {
                    let seen = (before >> shift) != (depth >> shift);
                    field.set_depth(cell, depth);
                    moved |= seen;
                    // Drying is not stirring: a tile that only dried needs no help
                    // staying awake (the drying step re-activates liquid tiles itself).
                    stirred |= flowed;
                    edge |= flowed && (x == x0 || x + 1 == x1 || y == y0 || y + 1 == y1);
                }
                any_liquid |= depth > 0;
            }
        }

        (moved, stirred, edge, any_liquid, soaks)
    }
}

#[cfg(test)]
#[path = "film_tiles_tests.rs"]
mod tests;
