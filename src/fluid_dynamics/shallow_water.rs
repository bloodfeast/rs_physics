//! Rivers, lakes and floods: the shallow-water equations on a terrain heightfield.
//!
//! # Why this and not a volume solver
//!
//! A river is a few metres deep and kilometres long. Every volume solver in this crate
//! would spend almost all of its cells on the vertical, which is the one direction
//! nothing interesting happens in: over a river's depth the flow is close to uniform,
//! and the pressure is hydrostatic. The shallow-water (Saint-Venant) equations take that
//! as given and integrate it out, leaving three numbers per column of water -- the depth
//! `h` and the two horizontal discharges `hu`, `hv` -- over a 2D grid laid on the
//! terrain:
//!
//! ```text
//!   ∂h/∂t  + ∂(hu)/∂x + ∂(hv)/∂z = 0
//!   ∂(hu)/∂t + ∂(hu² + gh²/2)/∂x + ∂(huv)/∂z = −gh ∂b/∂x − friction
//!   ∂(hv)/∂t + ∂(huv)/∂x + ∂(hv² + gh²/2)/∂z = −gh ∂b/∂z − friction
//! ```
//!
//! where `b` is the bed elevation. The water surface `b + h` is a real 3D surface that
//! runs downhill, pools in hollows, backs up behind a dam, jumps at a weir and floods
//! over its banks; what it cannot represent is anything that folds over on itself -- a
//! breaking wave, the inside of a waterfall, spray. Those belong to a particle layer
//! ([`crate::fluid_dynamics::SphFluid`] or the particle effects) spawned where this one
//! says the water is fast or falling.
//!
//! This is also how games that simulate rivers do it, because it is the model whose cost
//! scales with the *map* rather than with the map times its depth.
//!
//! # The scheme, and the four promises it keeps
//!
//! Finite volumes, first order, with an HLL Riemann flux at every cell face and the
//! **hydrostatic reconstruction** of Audusse et al. (2004) for the bed slope. Friction
//! is Manning's law, applied implicitly. Each of those choices buys one of these:
//!
//! 1. **A lake at rest stays at rest**, exactly, over any terrain -- including terrain
//!    that pokes out of it as islands. A scheme that is not *well-balanced* makes every
//!    still pond on uneven ground slosh forever, driven by nothing but the grid. The
//!    hydrostatic reconstruction is what makes the pressure across a face and the
//!    slope of the bed beneath it cancel to the last bit.
//! 2. **Depth never goes negative**, so banks, beaches and floodplains can wet and dry
//!    without a special case. That is a property of the HLL flux with dry-bed wave
//!    speeds under the timestep limit, which [`ShallowWater::step`] enforces by
//!    substepping.
//! 3. **Water is conserved to rounding.** Whatever leaves one cell enters its
//!    neighbour through the same face flux, and the only other ways in or out are the
//!    boundaries and [`ShallowWater::add_water`], which are metered:
//!    [`ShallowWater::volume_in`] and [`ShallowWater::volume_out`] account for every
//!    cubic metre.
//! 4. **The result does not depend on the thread count.** Rows are computed in parallel,
//!    but each cell's update is the same arithmetic in the same order whichever thread
//!    does it, and the only reduction is a maximum. Two machines stepping the same river
//!    get the same bits, which a lockstep game needs.
//!
//! Friction is Manning's formula, the one river engineers use: in steady flow down a
//! slope `S` a river settles at the depth where `q = h^{5/3} √S / n`. Manning's `n` is a
//! property of the bed -- about 0.03 s/m^{1/3} for a clean natural channel, 0.05 for a
//! weedy or stony one, 0.1 for a floodplain in brush -- so the same river runs fast over
//! gravel and slow through reeds without anyone choosing a speed.
//!
//! # Conventions
//!
//! * `y` is up, and the grid lies in the horizontal `x`–`z` plane (Bevy's convention).
//! * Cell `(i, j)` is column `i` along `x` and row `j` along `z`. It is stored at
//!   `j * nx + i` and its centre is at `origin + ((i + ½)·dx, (j + ½)·dx)`.
//! * `u` is velocity along `x` and `v` along `z`, in m/s. Discharges `hu`, `hv` are in
//!   m²/s (volume per second per metre of width).
//! * Every edge of the map is a [`Boundary::Wall`] until told otherwise.
//!
//! # A river in a game
//!
//! ```
//! use rs_physics::fluid_dynamics::{Boundary, Edge, ShallowWater};
//!
//! // A channel 200 m long and 20 m wide, falling 1 m over its length, on 5 m cells.
//! let (nx, nz, dx) = (40, 4, 5.0);
//! let bed: Vec<f64> = (0..nz)
//!     .flat_map(|_| (0..nx).map(move |i| 1.0 - (i as f64 + 0.5) * dx / 200.0))
//!     .collect();
//! let mut river = ShallowWater::new(nx, nz, dx, bed).unwrap();
//!
//! // 40 m³/s enters across the upstream edge and leaves freely at the downstream one.
//! river.set_boundary(Edge::MinX, 0..nz, Boundary::Inflow { discharge: 40.0 }).unwrap();
//! river.set_boundary(Edge::MaxX, 0..nz, Boundary::Open).unwrap();
//!
//! // Ten minutes of game time at 30 frames a second.
//! for _ in 0..(30 * 600) {
//!     river.step(1.0 / 30.0).unwrap();
//! }
//!
//! // The water reached the far end and is flowing downstream through it.
//! let mouth = river.sample(195.0, 10.0);
//! assert!(mouth.depth > 0.3 && mouth.velocity[0] > 0.3);
//! // And nothing was lost: everything that came in is in the channel or went out.
//! let balance = river.volume_in() - river.volume_out() - river.total_volume();
//! assert!(balance.abs() < 1e-6 * river.volume_in());
//! ```
//!
//! What a game does with it each frame:
//!
//! * **Render** -- [`ShallowWater::depths`] and [`ShallowWater::beds`] are the two
//!   heightfields; the water mesh is `bed + depth` wherever `depth` is above a small
//!   threshold. [`ShallowWater::velocity`] drives flow-map scrolling and foam.
//! * **Float things** -- [`ShallowWater::sample`] gives the surface height, the depth and
//!   the current at any point. Buoyancy is the displaced volume below `surface`, and the
//!   drag of the current is the drag of the *relative* velocity, which the analytic
//!   helpers in [`crate::fluid_dynamics`] compute.
//! * **Change the world** -- [`ShallowWater::set_bed`] moves terrain under the water
//!   (a crater, a collapsed bank, a dam) and keeps the water column on top of it, and
//!   [`ShallowWater::add_water`] pours water in (rain, a burst tank, a spring).
//!
//! # Limits
//!
//! * **First order.** Sharp features -- a bore, the front of a flood -- are smeared over a
//!   few cells. Cell size is the resolution knob; the tests measure the error.
//! * **The bed drop across one cell should be small against the depth.** Water a few
//!   centimetres deep on a steep hillside, on cells coarse enough that the ground falls
//!   more than the water is deep from one to the next, flows too slowly -- the known
//!   weakness of first-order hydrostatic reconstruction. A river is deep compared with
//!   its bed's fall per cell; sheet runoff on a mountainside is not.
//! * **The timestep is limited by the wave speed** `|u| + √(gh)`. [`ShallowWater::step`]
//!   takes whatever `dt` a frame has and substeps inside it, and returns how many
//!   substeps it took. On 1 m cells one substep covers a 60 Hz frame for water up to
//!   about 15 m deep. Halving the cell size doubles the substeps and quadruples the
//!   cells, so the cost goes as the inverse cube of the cell size.
//!
//! # Budget: what rate, what threads
//!
//! **Design point: presentation, on one worker thread, at 20–30 Hz.** A river is
//! something the player sees and floats things on, not something a fixed-rate
//! simulation tick should wait for. Run it off the main loop on its own thread, step it
//! with the real elapsed time (the substepping makes any rate correct), and let the
//! renderer interpolate between the last two states if it draws faster than that.
//!
//! One frame at 30 Hz on **one thread** ([`Threading::Serial`]), 1 m cells, a river in
//! steady flow wet from edge to edge -- the worst case, since dry cells are cheaper.
//! Measured with `cargo bench --bench shallow_water --features fluid_simulation`:
//!
//! | Grid | Per 30 Hz frame | Share of one core |
//! |---|---:|---:|
//! | 64 × 64 | 0.51 ms | 1.5% |
//! | 128 × 128 | 2.1 ms | 6% |
//! | 256 × 256 | 8.8 ms | 26% |
//! | 512 × 512 | 35 ms | more than a frame: use 2 m cells, or the pool |
//!
//! A real map is mostly dry: the `river_3d` demo's 160 × 120 valley, 12% wet, costs
//! 2.3 ms a frame on one thread at 30 Hz (two substeps, for its fastest water), about 7%
//! of a core.
//!
//! **Threads.** [`Threading::Auto`], the default, sweeps rows in parallel on rayon's
//! *current* pool for grids of 2048 cells or more; that is rayon's global pool unless
//! `step` is called inside `pool.install(..)`, which is how a caller bounds or isolates
//! the threads so they do not compete with a renderer's own. [`Threading::Serial`] keeps
//! every sweep on the calling thread. The answer is the same bits either way, and
//! nothing is allocated after construction.
//!
//! References: Audusse, Bouchut, Bristeau, Klein & Perthame, *A fast and stable
//! well-balanced scheme with hydrostatic reconstruction for shallow water flows*, SIAM J.
//! Sci. Comput. 25 (2004); Toro, *Shock-Capturing Methods for Free-Surface Shallow
//! Flows* (2001), for the HLL flux and its dry-bed wave speeds.

use std::ops::Range;

use rayon::prelude::*;

use crate::utils::PhysicsError;

/// Depth below which a cell counts as dry, in metres: it keeps its water but carries no
/// momentum. A micrometre is far below anything a player can see, and far above the
/// depths at which `hu / h` stops meaning anything.
pub const DRY_DEPTH: f64 = 1e-6;

/// Fraction of the positivity limit each substep uses. The scheme keeps depths
/// non-negative while `dt · (max wave speed along x + along z) ≤ dx / 2`; this runs at
/// 90% of that.
const CFL: f64 = 0.45;

/// Default Manning roughness, s/m^{1/3}: a clean natural channel.
pub const DEFAULT_MANNING: f64 = 0.03;

/// Default number of substeps one call to [`ShallowWater::step`] may take before it
/// gives up, so a pathological `dt` costs a bounded frame rather than a hang.
pub const DEFAULT_MAX_SUBSTEPS: usize = 10_000;

/// Standard gravity, m/s².
const STANDARD_GRAVITY: f64 = 9.81;

/// Grids with at least this many cells sweep their rows on rayon's pool; smaller ones
/// on the calling thread, where they finish before a hand-off would.
const PARALLEL_CELLS: usize = 2_048;

/// How [`ShallowWater::step`] uses threads. Either way the answer is the same bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Threading {
    /// Sweep rows in parallel on rayon's *current* pool when the grid has at least 2048
    /// cells, and on the calling thread below that. To bound or isolate the threads,
    /// call `step` inside your own pool: `my_pool.install(|| river.step(dt))`.
    #[default]
    Auto,
    /// Always on the calling thread. For a worker thread that must not compete with a
    /// renderer's pool, or a caller that runs several rivers on its own threads.
    Serial,
}

/// One edge of the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    /// The `x = origin.x` edge, along which `i = 0`. Its cells are numbered by `j`.
    MinX,
    /// The far `x` edge, `i = nx − 1`. Numbered by `j`.
    MaxX,
    /// The `z = origin.z` edge, `j = 0`. Numbered by `i`.
    MinZ,
    /// The far `z` edge, `j = nz − 1`. Numbered by `i`.
    MaxZ,
}

impl Edge {
    fn slot(self) -> usize {
        match self {
            Edge::MinX => 0,
            Edge::MaxX => 1,
            Edge::MinZ => 2,
            Edge::MaxZ => 3,
        }
    }
}

/// What happens to water at a stretch of the map's edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Boundary {
    /// Nothing crosses. Water reflects as it would off a vertical wall. The default.
    Wall,
    /// Water leaves freely: a river's mouth, or the map edge it runs off. The river is
    /// taken to carry on beyond the edge the way it arrives -- the water surface keeps
    /// its slope and the current its speed -- so a river in steady flow passes straight
    /// out at its own depth instead of backing up. Still water beside an open edge stays
    /// still, and water piling up against one is never pushed back in.
    Open,
    /// Water enters at a fixed total discharge, m³/s, spread evenly across the stretch
    /// of edge it is set on, flowing straight in. A river's source.
    Inflow {
        /// Total volume per second across the whole stretch, m³/s. Not negative.
        discharge: f64,
    },
    /// The water surface just outside is held at this elevation, m: a lake or the sea.
    /// Water flows in if the level is above the water inside and out if it is below.
    Level {
        /// Elevation of the outside water surface, m.
        surface: f64,
    },
}

/// A boundary as stored per edge cell: an inflow is kept per metre of edge.
#[derive(Debug, Clone, Copy, PartialEq)]
enum EdgeCell {
    Wall,
    Open,
    Inflow { q: f64 },
    Level { surface: f64 },
}

/// The flux through one face, in the face's normal frame, and the hydrostatic
/// corrections for the cells on either side of it.
#[derive(Debug, Clone, Copy, Default)]
struct FaceFlux {
    /// Volume per second per metre of face, towards the high-index cell. m²/s.
    mass: f64,
    /// Flux of the face-normal momentum. m³/s².
    normal: f64,
    /// Flux of the tangential momentum. m³/s².
    tangential: f64,
    /// `g/2 (h² − h*²)` for the low-index cell: the pressure that the bed step under
    /// this face holds up, added to the normal flux that cell sees.
    corr_lo: f64,
    /// The same for the high-index cell.
    corr_hi: f64,
}

/// One cell's state in a face's normal frame.
#[derive(Debug, Clone, Copy)]
struct Column {
    h: f64,
    normal: f64,
    tangential: f64,
    bed: f64,
}

/// What [`ShallowWater::sample`] reports at a point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaterSample {
    /// Water depth, m. Zero on dry ground.
    pub depth: f64,
    /// Bed elevation, m.
    pub bed: f64,
    /// Water surface elevation, `bed + depth`, m. On dry ground this is the ground.
    pub surface: f64,
    /// Depth-averaged current `[u, v]` along `x` and `z`, m/s. Zero on dry ground.
    pub velocity: [f64; 2],
}

/// A shallow-water simulation over a terrain heightfield. See the module docs.
#[derive(Debug, Clone)]
pub struct ShallowWater {
    nx: usize,
    nz: usize,
    dx: f64,
    origin: [f64; 2],
    gravity: f64,
    max_substeps: usize,
    /// Grids with at least this many cells sweep their rows in parallel.
    parallel_threshold: usize,
    threading: Threading,
    bed: Vec<f64>,
    h: Vec<f64>,
    hu: Vec<f64>,
    hv: Vec<f64>,
    manning: Vec<f64>,
    /// Per edge (indexed by [`Edge::slot`]), one entry per cell along it.
    edges: [Vec<EdgeCell>; 4],
    /// Faces normal to `x`: `(nx + 1)` per row, `nz` rows.
    fx: Vec<FaceFlux>,
    /// Faces normal to `z`: `nx` per row, `nz + 1` rows.
    fz: Vec<FaceFlux>,
    time: f64,
    volume_in: f64,
    volume_out: f64,
}

impl ShallowWater {
    /// A dry grid of `nx × nz` square cells of side `cell_size` metres over the given bed
    /// elevations, stored row by row (`bed[j * nx + i]`).
    ///
    /// Gravity is 9.81 m/s², Manning's `n` is [`DEFAULT_MANNING`] everywhere, the origin
    /// is `(0, 0)`, and every edge is a wall.
    pub fn new(
        nx: usize,
        nz: usize,
        cell_size: f64,
        bed: Vec<f64>,
    ) -> Result<ShallowWater, PhysicsError> {
        let cells = nx.checked_mul(nz).ok_or(PhysicsError::InvalidDimension)?;
        if nx == 0 || nz == 0 || bed.len() != cells {
            return Err(PhysicsError::InvalidDimension);
        }
        if !(cell_size.is_finite() && cell_size > 0.0) {
            return Err(PhysicsError::InvalidDimension);
        }
        if bed.iter().any(|b| !b.is_finite()) {
            return Err(PhysicsError::CalculationError(
                "bed elevations must be finite".to_string(),
            ));
        }
        Ok(ShallowWater {
            nx,
            nz,
            dx: cell_size,
            origin: [0.0, 0.0],
            gravity: STANDARD_GRAVITY,
            max_substeps: DEFAULT_MAX_SUBSTEPS,
            parallel_threshold: PARALLEL_CELLS,
            threading: Threading::Auto,
            bed,
            h: vec![0.0; cells],
            hu: vec![0.0; cells],
            hv: vec![0.0; cells],
            manning: vec![DEFAULT_MANNING; cells],
            edges: [
                vec![EdgeCell::Wall; nz],
                vec![EdgeCell::Wall; nz],
                vec![EdgeCell::Wall; nx],
                vec![EdgeCell::Wall; nx],
            ],
            fx: vec![FaceFlux::default(); (nx + 1) * nz],
            fz: vec![FaceFlux::default(); nx * (nz + 1)],
            time: 0.0,
            volume_in: 0.0,
            volume_out: 0.0,
        })
    }

    /// The world position of the grid's corner, `[x, z]` in metres.
    pub fn with_origin(mut self, origin: [f64; 2]) -> Result<ShallowWater, PhysicsError> {
        if !(origin[0].is_finite() && origin[1].is_finite()) {
            return Err(PhysicsError::InvalidDimension);
        }
        self.origin = origin;
        Ok(self)
    }

    /// Gravitational acceleration, m/s², positive.
    pub fn with_gravity(mut self, gravity: f64) -> Result<ShallowWater, PhysicsError> {
        if !(gravity.is_finite() && gravity > 0.0) {
            return Err(PhysicsError::InvalidCoefficient);
        }
        self.gravity = gravity;
        Ok(self)
    }

    /// The same Manning roughness for every cell, s/m^{1/3}. Zero is frictionless.
    pub fn with_manning(mut self, n: f64) -> Result<ShallowWater, PhysicsError> {
        validate_manning(n)?;
        self.manning.fill(n);
        Ok(self)
    }

    /// How many substeps one [`ShallowWater::step`] may take before returning an error.
    pub fn with_max_substeps(mut self, max_substeps: usize) -> Result<ShallowWater, PhysicsError> {
        if max_substeps == 0 {
            return Err(PhysicsError::InvalidCoefficient);
        }
        self.max_substeps = max_substeps;
        Ok(self)
    }

    /// How `step` uses threads; see [`Threading`]. The default is [`Threading::Auto`].
    pub fn with_threading(mut self, threading: Threading) -> ShallowWater {
        self.threading = threading;
        self
    }

    /// Change how `step` uses threads; see [`Threading`].
    pub fn set_threading(&mut self, threading: Threading) {
        self.threading = threading;
    }

    /// How `step` uses threads.
    pub fn threading(&self) -> Threading {
        self.threading
    }

    /// Force the parallel or the serial sweep, to test that they agree.
    #[cfg(test)]
    fn with_parallel_threshold(mut self, cells: usize) -> ShallowWater {
        self.parallel_threshold = cells;
        self
    }

    /// Cells along `x`.
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Cells along `z`.
    pub fn nz(&self) -> usize {
        self.nz
    }

    /// Side of a cell, m.
    pub fn cell_size(&self) -> f64 {
        self.dx
    }

    /// World position of the grid's corner, `[x, z]`.
    pub fn origin(&self) -> [f64; 2] {
        self.origin
    }

    /// Simulated time so far, s.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Water depth in every cell, row by row, m.
    pub fn depths(&self) -> &[f64] {
        &self.h
    }

    /// Bed elevation in every cell, row by row, m.
    pub fn beds(&self) -> &[f64] {
        &self.bed
    }

    /// Discharge along `x` in every cell, row by row, m²/s.
    pub fn discharges_x(&self) -> &[f64] {
        &self.hu
    }

    /// Discharge along `z` in every cell, row by row, m²/s.
    pub fn discharges_z(&self) -> &[f64] {
        &self.hv
    }

    /// Water depth in cell `(i, j)`, m. Panics outside the grid, like slice indexing.
    pub fn depth(&self, i: usize, j: usize) -> f64 {
        self.h[self.index(i, j)]
    }

    /// Bed elevation in cell `(i, j)`, m.
    pub fn bed(&self, i: usize, j: usize) -> f64 {
        self.bed[self.index(i, j)]
    }

    /// Water surface elevation in cell `(i, j)`, `bed + depth`, m.
    pub fn surface(&self, i: usize, j: usize) -> f64 {
        let k = self.index(i, j);
        self.bed[k] + self.h[k]
    }

    /// Depth-averaged velocity `[u, v]` in cell `(i, j)`, m/s. Zero where dry.
    pub fn velocity(&self, i: usize, j: usize) -> [f64; 2] {
        let k = self.index(i, j);
        if self.h[k] > DRY_DEPTH {
            [self.hu[k] / self.h[k], self.hv[k] / self.h[k]]
        } else {
            [0.0, 0.0]
        }
    }

    /// Manning roughness of cell `(i, j)`, s/m^{1/3}.
    pub fn manning(&self, i: usize, j: usize) -> f64 {
        self.manning[self.index(i, j)]
    }

    /// Total water on the grid, m³.
    pub fn total_volume(&self) -> f64 {
        self.h.iter().sum::<f64>() * self.dx * self.dx
    }

    /// Water that has entered through the boundaries and [`ShallowWater::add_water`]
    /// since the grid was made, m³.
    pub fn volume_in(&self) -> f64 {
        self.volume_in
    }

    /// Water that has left through the boundaries since the grid was made, m³.
    pub fn volume_out(&self) -> f64 {
        self.volume_out
    }

    /// The cell containing world point `(x, z)`, if any.
    pub fn cell_at(&self, x: f64, z: f64) -> Option<(usize, usize)> {
        let fi = (x - self.origin[0]) / self.dx;
        let fj = (z - self.origin[1]) / self.dx;
        if !(fi >= 0.0 && fj >= 0.0) {
            return None;
        }
        let (i, j) = (fi as usize, fj as usize);
        (i < self.nx && j < self.nz).then_some((i, j))
    }

    /// Depth, bed, surface and current at world point `(x, z)`, interpolated
    /// bilinearly between cell centres and held at the edge values outside the grid.
    ///
    /// The current is the interpolated discharge over the interpolated depth, so a dry
    /// neighbour does not drag a river's speed down at its bank.
    pub fn sample(&self, x: f64, z: f64) -> WaterSample {
        let (i0, i1, tx) = axis_weights((x - self.origin[0]) / self.dx - 0.5, self.nx);
        let (j0, j1, tz) = axis_weights((z - self.origin[1]) / self.dx - 0.5, self.nz);
        let lerp2 = |field: &[f64]| {
            let a = field[j0 * self.nx + i0] * (1.0 - tx) + field[j0 * self.nx + i1] * tx;
            let b = field[j1 * self.nx + i0] * (1.0 - tx) + field[j1 * self.nx + i1] * tx;
            a * (1.0 - tz) + b * tz
        };
        let depth = lerp2(&self.h);
        let bed = lerp2(&self.bed);
        let velocity = if depth > DRY_DEPTH {
            [lerp2(&self.hu) / depth, lerp2(&self.hv) / depth]
        } else {
            [0.0, 0.0]
        };
        WaterSample {
            depth,
            bed,
            surface: bed + depth,
            velocity,
        }
    }

    /// Set how a stretch of one edge behaves. `cells` counts along the edge: `j` for
    /// [`Edge::MinX`] and [`Edge::MaxX`], `i` for [`Edge::MinZ`] and [`Edge::MaxZ`].
    ///
    /// An [`Boundary::Inflow`]'s discharge is the total over the stretch, so the same
    /// river enters at the same rate whatever the cell size.
    pub fn set_boundary(
        &mut self,
        edge: Edge,
        cells: Range<usize>,
        boundary: Boundary,
    ) -> Result<(), PhysicsError> {
        let len = self.edges[edge.slot()].len();
        if cells.start >= cells.end || cells.end > len {
            return Err(PhysicsError::InvalidDimension);
        }
        let stored = match boundary {
            Boundary::Wall => EdgeCell::Wall,
            Boundary::Open => EdgeCell::Open,
            Boundary::Inflow { discharge } => {
                if !(discharge.is_finite() && discharge >= 0.0) {
                    return Err(PhysicsError::InvalidVelocity);
                }
                EdgeCell::Inflow {
                    q: discharge / ((cells.end - cells.start) as f64 * self.dx),
                }
            }
            Boundary::Level { surface } => {
                if !surface.is_finite() {
                    return Err(PhysicsError::InvalidDimension);
                }
                EdgeCell::Level { surface }
            }
        };
        self.edges[edge.slot()][cells].fill(stored);
        Ok(())
    }

    /// Fill every cell to the water surface `surface`, m, at rest: depth
    /// `max(surface − bed, 0)` and no current. Ground above the level stays dry. Water
    /// this adds is counted in [`ShallowWater::volume_in`], and water it removes in
    /// [`ShallowWater::volume_out`].
    pub fn fill_to_level(&mut self, surface: f64) -> Result<(), PhysicsError> {
        if !surface.is_finite() {
            return Err(PhysicsError::InvalidDimension);
        }
        let before = self.total_volume();
        for k in 0..self.h.len() {
            self.h[k] = (surface - self.bed[k]).max(0.0);
            self.hu[k] = 0.0;
            self.hv[k] = 0.0;
        }
        let change = self.total_volume() - before;
        if change > 0.0 {
            self.volume_in += change;
        } else {
            self.volume_out -= change;
        }
        Ok(())
    }

    /// Pour `volume` m³ of still water into cell `(i, j)`. Counted in
    /// [`ShallowWater::volume_in`]. The cell's momentum is unchanged, so the water it
    /// already held keeps moving and the new water slows it by dilution.
    pub fn add_water(&mut self, i: usize, j: usize, volume: f64) -> Result<(), PhysicsError> {
        let k = self.checked_index(i, j)?;
        if !(volume.is_finite() && volume >= 0.0) {
            return Err(PhysicsError::InvalidDimension);
        }
        self.h[k] += volume / (self.dx * self.dx);
        self.volume_in += volume;
        Ok(())
    }

    /// Set the depth-averaged velocity `[u, v]` in cell `(i, j)`, m/s. Ignored where
    /// the cell is dry.
    pub fn set_velocity(&mut self, i: usize, j: usize, velocity: [f64; 2]) -> Result<(), PhysicsError> {
        let k = self.checked_index(i, j)?;
        if !(velocity[0].is_finite() && velocity[1].is_finite()) {
            return Err(PhysicsError::InvalidVelocity);
        }
        if self.h[k] > DRY_DEPTH {
            self.hu[k] = self.h[k] * velocity[0];
            self.hv[k] = self.h[k] * velocity[1];
        }
        Ok(())
    }

    /// Move the ground in cell `(i, j)` to `elevation`, m. The water column moves with
    /// it -- its depth is kept -- so digging a crater under a river makes a hole the river
    /// then fills, and raising a bank lifts the water on it, which runs off. Volume is
    /// conserved either way.
    pub fn set_bed(&mut self, i: usize, j: usize, elevation: f64) -> Result<(), PhysicsError> {
        let k = self.checked_index(i, j)?;
        if !elevation.is_finite() {
            return Err(PhysicsError::CalculationError(
                "bed elevations must be finite".to_string(),
            ));
        }
        self.bed[k] = elevation;
        Ok(())
    }

    /// Manning roughness of cell `(i, j)`, s/m^{1/3}.
    pub fn set_manning(&mut self, i: usize, j: usize, n: f64) -> Result<(), PhysicsError> {
        let k = self.checked_index(i, j)?;
        validate_manning(n)?;
        self.manning[k] = n;
        Ok(())
    }

    /// Advance the water by `dt` seconds, in as many substeps as the wave speed
    /// requires. Returns the number of substeps taken.
    ///
    /// Any `dt` a game frame produces is fine: the substeps are sized from the flow, so
    /// the answer does not depend on the frame rate beyond the scheme's own error. A
    /// `dt` of zero does nothing. If the flow would need more than the configured maximum
    /// number of substeps (see [`ShallowWater::with_max_substeps`]), this stops there and
    /// returns an error, having advanced [`ShallowWater::time`] by as much as it did.
    pub fn step(&mut self, dt: f64) -> Result<usize, PhysicsError> {
        if !(dt.is_finite() && dt >= 0.0) {
            return Err(PhysicsError::InvalidTime);
        }
        let mut remaining = dt;
        let mut substeps = 0;
        while remaining > 0.0 {
            if substeps == self.max_substeps {
                return Err(PhysicsError::CalculationError(format!(
                    "shallow water needed more than {} substeps for dt = {dt} s",
                    self.max_substeps
                )));
            }
            let used = self.substep(remaining)?;
            remaining = if used >= remaining { 0.0 } else { remaining - used };
            substeps += 1;
        }
        Ok(substeps)
    }

    /// One explicit update of at most `limit` seconds. Returns the time it covered.
    fn substep(&mut self, limit: f64) -> Result<f64, PhysicsError> {
        let (sx, sz) = self.compute_fluxes();
        if sx.is_nan() || sz.is_nan() {
            return Err(PhysicsError::CalculationError(
                "shallow water state became non-finite".to_string(),
            ));
        }
        let speed = sx + sz;
        let dt = if speed > 0.0 {
            (CFL * self.dx / speed).min(limit)
        } else {
            limit
        };
        if !(dt > 0.0) {
            return Err(PhysicsError::CalculationError(
                "shallow water timestep collapsed to zero".to_string(),
            ));
        }
        self.meter_boundaries(dt);
        let finite = self.apply_fluxes(dt);
        self.time += dt;
        if !finite {
            // A dry-bed `max(0)` inside the reconstruction would otherwise hide a NaN
            // depth from the wave speeds for good.
            return Err(PhysicsError::CalculationError(
                "shallow water state became non-finite".to_string(),
            ));
        }
        Ok(dt)
    }

    /// Fill `fx` and `fz` from the current state. Returns the largest wave speed seen
    /// along `x` and along `z` (NaN if any state was NaN).
    fn compute_fluxes(&mut self) -> (f64, f64) {
        let parallel = self.parallel();
        let state = State {
            nx: self.nx,
            nz: self.nz,
            g: self.gravity,
            bed: &self.bed,
            h: &self.h,
            hu: &self.hu,
            hv: &self.hv,
            edges: &self.edges,
        };
        let (nx, fx, fz) = (self.nx, &mut self.fx, &mut self.fz);
        if parallel {
            let sx = fx
                .par_chunks_mut(nx + 1)
                .enumerate()
                .map(|(j, row)| state.x_faces(j, row))
                .reduce(|| 0.0, nan_max);
            let sz = fz
                .par_chunks_mut(nx)
                .enumerate()
                .map(|(j, row)| state.z_faces(j, row))
                .reduce(|| 0.0, nan_max);
            (sx, sz)
        } else {
            let sx = fx
                .chunks_mut(nx + 1)
                .enumerate()
                .map(|(j, row)| state.x_faces(j, row))
                .fold(0.0, nan_max);
            let sz = fz
                .chunks_mut(nx)
                .enumerate()
                .map(|(j, row)| state.z_faces(j, row))
                .fold(0.0, nan_max);
            (sx, sz)
        }
    }

    /// Count what crosses the map's edges this substep. A serial pass over the
    /// perimeter, in a fixed order, so the totals are deterministic.
    fn meter_boundaries(&mut self, dt: f64) {
        let (nx, nz) = (self.nx, self.nz);
        let scale = dt * self.dx;
        let (mut came_in, mut went_out) = (0.0, 0.0);
        let mut inward = |flux: f64| {
            if flux > 0.0 {
                came_in += flux * scale;
            } else {
                went_out -= flux * scale;
            }
        };
        for j in 0..nz {
            inward(self.fx[j * (nx + 1)].mass);
            inward(-self.fx[j * (nx + 1) + nx].mass);
        }
        for i in 0..nx {
            inward(self.fz[i].mass);
            inward(-self.fz[nz * nx + i].mass);
        }
        self.volume_in += came_in;
        self.volume_out += went_out;
    }

    /// Update every cell from the face fluxes, then apply friction. Returns whether
    /// every cell is still finite.
    fn apply_fluxes(&mut self, dt: f64) -> bool {
        let parallel = self.parallel();
        let update = Update {
            nx: self.nx,
            g: self.gravity,
            dt,
            lambda: dt / self.dx,
            fx: &self.fx,
            fz: &self.fz,
            manning: &self.manning,
        };
        let nx = self.nx;
        if parallel {
            self.h
                .par_chunks_mut(nx)
                .zip(self.hu.par_chunks_mut(nx))
                .zip(self.hv.par_chunks_mut(nx))
                .enumerate()
                .map(|(j, ((h, hu), hv))| update.row(j, h, hu, hv))
                .reduce(|| true, |a, b| a && b)
        } else {
            self.h
                .chunks_mut(nx)
                .zip(self.hu.chunks_mut(nx))
                .zip(self.hv.chunks_mut(nx))
                .enumerate()
                .map(|(j, ((h, hu), hv))| update.row(j, h, hu, hv))
                .fold(true, |a, b| a && b)
        }
    }

    /// Whether this grid is big enough for rows to be worth handing to other threads.
    /// Below that, the hand-off costs more than the rows do. Both paths run the same
    /// arithmetic on each row, so the choice never changes the answer.
    fn parallel(&self) -> bool {
        self.threading == Threading::Auto && self.h.len() >= self.parallel_threshold
    }

    fn index(&self, i: usize, j: usize) -> usize {
        assert!(
            i < self.nx && j < self.nz,
            "cell ({i}, {j}) is outside a {} x {} shallow-water grid",
            self.nx,
            self.nz
        );
        j * self.nx + i
    }

    fn checked_index(&self, i: usize, j: usize) -> Result<usize, PhysicsError> {
        if i < self.nx && j < self.nz {
            Ok(j * self.nx + i)
        } else {
            Err(PhysicsError::InvalidDimension)
        }
    }
}

/// The read-only state a flux sweep needs.
struct State<'a> {
    nx: usize,
    nz: usize,
    g: f64,
    bed: &'a [f64],
    h: &'a [f64],
    hu: &'a [f64],
    hv: &'a [f64],
    edges: &'a [Vec<EdgeCell>; 4],
}

impl State<'_> {
    /// Cell `(i, j)` in a face frame whose normal is `x` (`along_x`) or `z`.
    #[inline]
    fn column(&self, i: usize, j: usize, along_x: bool) -> Column {
        let k = j * self.nx + i;
        let (u, v) = velocity_of(self.h[k], self.hu[k], self.hv[k]);
        let (normal, tangential) = if along_x { (u, v) } else { (v, u) };
        Column { h: self.h[k], normal, tangential, bed: self.bed[k] }
    }

    /// Row `j` of the faces normal to `x`: face `i` lies between cells `(i − 1, j)` and
    /// `(i, j)`. Returns the fastest wave speed in the row.
    fn x_faces(&self, j: usize, row: &mut [FaceFlux]) -> f64 {
        let nx = self.nx;
        let mut fastest = 0.0;
        for (i, slot) in row.iter_mut().enumerate() {
            let (flux, speed) = if i == 0 {
                let beyond = (nx > 1).then(|| self.column(1, j, true));
                let kind = self.edges[Edge::MinX.slot()][j];
                boundary_face(self.g, kind, self.column(0, j, true), beyond, false)
            } else if i == nx {
                let beyond = (nx > 1).then(|| self.column(nx - 2, j, true));
                let kind = self.edges[Edge::MaxX.slot()][j];
                boundary_face(self.g, kind, self.column(nx - 1, j, true), beyond, true)
            } else {
                interior_face(self.g, self.column(i - 1, j, true), self.column(i, j, true))
            };
            *slot = flux;
            fastest = nan_max(fastest, speed);
        }
        fastest
    }

    /// Row `j` of the faces normal to `z`: face `i` lies between cells `(i, j − 1)` and
    /// `(i, j)`. Returns the fastest wave speed in the row.
    fn z_faces(&self, j: usize, row: &mut [FaceFlux]) -> f64 {
        let nz = self.nz;
        let mut fastest = 0.0;
        for (i, slot) in row.iter_mut().enumerate() {
            let (flux, speed) = if j == 0 {
                let beyond = (nz > 1).then(|| self.column(i, 1, false));
                let kind = self.edges[Edge::MinZ.slot()][i];
                boundary_face(self.g, kind, self.column(i, 0, false), beyond, false)
            } else if j == nz {
                let beyond = (nz > 1).then(|| self.column(i, nz - 2, false));
                let kind = self.edges[Edge::MaxZ.slot()][i];
                boundary_face(self.g, kind, self.column(i, nz - 1, false), beyond, true)
            } else {
                interior_face(self.g, self.column(i, j - 1, false), self.column(i, j, false))
            };
            *slot = flux;
            fastest = nan_max(fastest, speed);
        }
        fastest
    }
}

/// What the cell update needs besides the cells themselves.
struct Update<'a> {
    nx: usize,
    g: f64,
    dt: f64,
    lambda: f64,
    fx: &'a [FaceFlux],
    fz: &'a [FaceFlux],
    manning: &'a [f64],
}

impl Update<'_> {
    /// Update row `j` from its four faces per cell, then apply friction. Returns
    /// whether every cell in the row is still finite.
    fn row(&self, j: usize, h_row: &mut [f64], hu_row: &mut [f64], hv_row: &mut [f64]) -> bool {
        let (nx, lambda) = (self.nx, self.lambda);
        let mut finite = true;
        for i in 0..nx {
            let west = &self.fx[j * (nx + 1) + i];
            let east = &self.fx[j * (nx + 1) + i + 1];
            let south = &self.fz[j * nx + i];
            let north = &self.fz[(j + 1) * nx + i];

            let h = h_row[i] - lambda * ((east.mass - west.mass) + (north.mass - south.mass));
            let hu = hu_row[i]
                - lambda
                    * ((east.normal + east.corr_lo - west.normal - west.corr_hi)
                        + (north.tangential - south.tangential));
            let hv = hv_row[i]
                - lambda
                    * ((east.tangential - west.tangential)
                        + (north.normal + north.corr_lo - south.normal - south.corr_hi));

            if h <= DRY_DEPTH {
                // `max` only ever removes rounding here: the timestep keeps the update
                // non-negative in exact arithmetic.
                h_row[i] = h.max(0.0);
                hu_row[i] = 0.0;
                hv_row[i] = 0.0;
                continue;
            }
            // Manning friction, implicit in the speed so that a thin fast layer is
            // slowed rather than reversed: `S_f = g n² |u| u / h^{4/3}`.
            let n = self.manning[j * nx + i];
            let damping = if n > 0.0 {
                let speed = (hu * hu + hv * hv).sqrt() / h;
                1.0 / (1.0 + self.dt * self.g * n * n * speed / (h * h.cbrt()))
            } else {
                1.0
            };
            h_row[i] = h;
            hu_row[i] = hu * damping;
            hv_row[i] = hv * damping;
            finite &= h.is_finite() && hu.is_finite() && hv.is_finite();
        }
        finite
    }
}

fn validate_manning(n: f64) -> Result<(), PhysicsError> {
    if n.is_finite() && n >= 0.0 {
        Ok(())
    } else {
        Err(PhysicsError::InvalidCoefficient)
    }
}

/// Velocity of a column, zero where it is dry.
#[inline]
fn velocity_of(h: f64, hu: f64, hv: f64) -> (f64, f64) {
    if h > DRY_DEPTH {
        (hu / h, hv / h)
    } else {
        (0.0, 0.0)
    }
}

/// Maximum that propagates NaN, so a broken state is reported rather than hidden.
#[inline]
fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// Lower index, upper index and weight along one axis, clamped to the grid.
fn axis_weights(f: f64, n: usize) -> (usize, usize, f64) {
    if !(f > 0.0) {
        return (0, 0, 0.0);
    }
    let last = (n - 1) as f64;
    if f >= last {
        return (n - 1, n - 1, 0.0);
    }
    let i0 = f as usize;
    (i0, i0 + 1, f - i0 as f64)
}

/// The flux through a face between two cells, with the hydrostatic reconstruction:
/// both sides are cut to the higher of the two beds before the Riemann problem is
/// solved, and each side is told how much pressure the step holds up.
#[inline]
fn interior_face(g: f64, lo: Column, hi: Column) -> (FaceFlux, f64) {
    let bed = lo.bed.max(hi.bed);
    let h_lo = (lo.h + lo.bed - bed).max(0.0);
    let h_hi = (hi.h + hi.bed - bed).max(0.0);
    let (mass, normal, tangential, speed) = hll(
        g,
        h_lo,
        lo.normal,
        lo.tangential,
        h_hi,
        hi.normal,
        hi.tangential,
    );
    (
        FaceFlux {
            mass,
            normal,
            tangential,
            corr_lo: 0.5 * g * (lo.h * lo.h - h_lo * h_lo),
            corr_hi: 0.5 * g * (hi.h * hi.h - h_hi * h_hi),
        },
        speed,
    )
}

/// The flux through a face on the map's edge. `inside` is the edge cell and `beyond`
/// the next one in from it, if the grid has one. `inside_is_lo` says whether the edge
/// cell is on the low-index side of the face (the `Max` edges) or the high one.
#[inline]
fn boundary_face(
    g: f64,
    kind: EdgeCell,
    inside: Column,
    beyond: Option<Column>,
    inside_is_lo: bool,
) -> (FaceFlux, f64) {
    let ghost = match kind {
        EdgeCell::Wall => Column { normal: -inside.normal, ..inside },
        EdgeCell::Open => {
            // Carry the surface's slope across the edge, but never above the inside
            // surface. Copying the edge cell instead (a flat continuation) is not a
            // continuation of a sloping river at all: at first order the cells of a
            // river on a slope carry about 1% less discharge than the faces between
            // them, the copied face passes only the cell's, and the difference piles
            // up at the mouth as a backwater that never stops rising.
            let surface = inside.h + inside.bed;
            let ghost_surface = match beyond {
                Some(next) if inside.h > DRY_DEPTH && next.h > DRY_DEPTH => {
                    (2.0 * surface - (next.h + next.bed)).min(surface)
                }
                _ => surface,
            };
            Column {
                h: (ghost_surface - inside.bed).max(0.0),
                ..inside
            }
        }
        EdgeCell::Level { surface } => Column {
            h: (surface - inside.bed).max(0.0),
            ..inside
        },
        EdgeCell::Inflow { q } => {
            // The discharge is imposed rather than solved for, so exactly `q` enters.
            // The depth it enters at continues the inside surface's slope upstream (a
            // river arriving down a slope arrives from higher water, and that head is
            // what pushes the first cell), and is never less than the critical depth
            // for `q` -- the least depth that can carry it, which is also what it
            // enters dry ground at.
            let surface = inside.h + inside.bed;
            let upstream = match beyond {
                Some(next) if inside.h > DRY_DEPTH && next.h > DRY_DEPTH => {
                    (2.0 * surface - (next.h + next.bed)).max(surface) - inside.bed
                }
                _ => inside.h,
            };
            let depth = upstream.max((q * q / g).cbrt());
            if depth <= 0.0 {
                return (FaceFlux::default(), 0.0);
            }
            let into = if inside_is_lo { -q } else { q };
            let flux = FaceFlux {
                mass: into,
                normal: q * q / depth + 0.5 * g * depth * depth,
                tangential: 0.0,
                corr_lo: 0.0,
                corr_hi: 0.0,
            };
            return (flux, q / depth + (g * depth).sqrt());
        }
    };
    if inside_is_lo {
        interior_face(g, inside, ghost)
    } else {
        interior_face(g, ghost, inside)
    }
}

/// HLL flux for the shallow-water equations in a face's normal frame, with Toro's
/// dry-bed wave speeds. The tangential momentum is carried upwind with the mass.
/// Returns `(mass, normal, tangential, largest |wave speed|)`.
#[inline]
fn hll(
    g: f64,
    h_l: f64,
    un_l: f64,
    ut_l: f64,
    h_r: f64,
    un_r: f64,
    ut_r: f64,
) -> (f64, f64, f64, f64) {
    if h_l <= 0.0 && h_r <= 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let c_l = (g * h_l).sqrt();
    let c_r = (g * h_r).sqrt();
    let (s_l, s_r) = if h_l <= 0.0 {
        (un_r - 2.0 * c_r, un_r + c_r)
    } else if h_r <= 0.0 {
        (un_l - c_l, un_l + 2.0 * c_l)
    } else {
        ((un_l - c_l).min(un_r - c_r), (un_l + c_l).max(un_r + c_r))
    };

    let mass_l = h_l * un_l;
    let mass_r = h_r * un_r;
    let normal_l = mass_l * un_l + 0.5 * g * h_l * h_l;
    let normal_r = mass_r * un_r + 0.5 * g * h_r * h_r;

    let (mass, normal) = if s_l >= 0.0 {
        (mass_l, normal_l)
    } else if s_r <= 0.0 {
        (mass_r, normal_r)
    } else {
        let inv = 1.0 / (s_r - s_l);
        (
            (s_r * mass_l - s_l * mass_r + s_l * s_r * (h_r - h_l)) * inv,
            (s_r * normal_l - s_l * normal_r + s_l * s_r * (mass_r - mass_l)) * inv,
        )
    };
    let tangential = if mass >= 0.0 { mass * ut_l } else { mass * ut_r };
    (mass, normal, tangential, s_l.abs().max(s_r.abs()))
}

#[cfg(test)]
#[path = "shallow_water_tests.rs"]
mod tests;
