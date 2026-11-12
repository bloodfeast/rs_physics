# STAIN-MOVE: the stain's flow laws move into rs_physics

Ridgeline's `terrain_stain.rs` hand-rolls three laws on its 1400 x 1000 stain buffer: the
film flow (`Flux`, `flux_tile`, `gather_tile`), the soak of running blood into the ground
(`soak`, `absorb`, the drink in `gather_tile`) and the ridge slump (`slump`). Ruling
(Brandon 2026-11-12, "Measure, then move"): add the laws here as standalone, unit-tested
functions, switch the stain to them, delete the hand-rolled copies. Bar: no slower
(profiler, both arms one run), look unchanged (captures). Painting, channels and upload
stay in Ridgeline. Nothing here touches `shallow_water.rs` (FLUIDS-B-SWE owns it).

## Why not `FilmGrid`

`FilmGrid::step` is the reference lubrication solver: dense, `f64`, five arrays, every cell
every step. The stain is sparse (a fight wets a few percent of the map; the tile flags
skip the rest), fixed-point (the depth's high byte is the texel the shader samples, so
there is no quantisation step), eight-neighbour (the multiple-flow-direction stencil that
makes a spill round, not a diamond), and carries a contact-line retention term per sender
that `FilmGrid` does not have. Porting the stain onto `FilmGrid` would change the look and
pay for the dry map. So the move adds a second entry point beside it, for sparse callers.

## 1. `fluid_dynamics::film_tiles` -- the sparse-tile film step

```rust
pub const TILE: usize = 32;
pub const INV_DISTANCE: [f32; 8];          // E W S N SE NE SW NW

/// The film law, fixed per fluid and grid: the Nusselt cube with a Bingham plug,
/// Courant-capped, in display units of depth and sub-units of storage.
#[derive(Clone, Copy)]
pub struct TileFilmLaw {
    pub courant: f32,         // share of a texel per step at the reference film
    pub grade_per_head: f32,  // display units of head per texel -> grade / reference grade
    pub drive_gain: f32,      // grade * film -> wall shear length (m)
    pub yield_length: f32,    // tau_y / (rho g), m
    pub yield_head: f32,      // wet-to-wet threshold, display units at unit inv_depth
    pub film: u8,             // display units below which nothing is loose
    pub subunits: i32,        // storage units per display unit
    pub fall_gain: f32,       // stored fall unit -> display units
    pub legacy_cohesion: Option<f32>, // the constant-threshold ablation arm
}

/// What the caller stores and what its ground does: the storage stays the caller's
/// (Ridgeline keeps depth's high byte inside its RGBA texture), so the step costs no copy.
/// Every method is per texel and `#[inline]`-able; the step is generic over it, so
/// monomorphisation gives the same code the hand-rolled loop compiled to.
pub trait FilmField {
    fn depth(&self, cell: usize) -> i32;                 // sub-units
    fn set_depth(&mut self, cell: usize, depth: i32);    // clamps to storage
    fn display(&self, cell: usize) -> u8;                // depth >> 16
    fn snapshot_row(&self, row: usize, x0: usize, x1: usize,
                    surface: &mut [i16], wet: &mut [bool]); // liquid surface + wet flags
    fn retention(&self, cell: usize) -> f32;             // contact-line head, unit inv_depth
    fn thinning(&self, x: usize, y: usize, depth: i32) -> f32; // yield kept under water
    fn may_dry(&self, x: usize, y: usize) -> bool;       // no water standing on it
    fn drink(&mut self, cell: usize, arrived: f32) -> i32;     // soak (law 2), sub-units
}

/// The sparse state: tile flags, the flux and snapshot buffers, the live list. Owns no
/// depth. Allocates in `new` and never in `step`.
pub struct TileFilm { /* tiles_x, tiles_y, active, liquid, flux: Vec<TileFlux>,
                         surface: Vec<i16>, wet: Vec<bool>, live: Vec<usize>, ... */ }

impl TileFilm {
    pub fn new(width: usize, height: usize) -> TileFilm;
    pub fn wake(&mut self, x: usize, y: usize);                 // a mark landed liquid here
    pub fn step<F: FilmField>(&mut self, field: &mut F, law: &TileFilmLaw,
                              fall_x: &[i8], fall_z: &[i8], drying: bool,
                              touched: &mut impl FnMut(usize, usize)) -> TileStep;
    pub fn tiles_visited(&self) -> usize;
    pub fn clear(&mut self);
}
pub struct TileStep { pub moved: bool, pub flux_ns: u64, pub gather_ns: u64, pub soaks: u64 }
```

`step` is the stain's two passes verbatim: pass one (`flux`) reads the old depth and the
surface snapshot and writes one `TileFlux` per texel; pass two (`gather`) recomputes each
sender's share from the drop and the sender's carried `inv_depth`, so mass is conserved
by construction and the result does not depend on traversal order. The arithmetic is
moved, not rewritten, in the same order and the same types, so the output is
bit-identical; a Ridgeline test runs both arms on clones of one battlefield and compares
every byte before the hand-rolled arm is deleted.

One change rides along: the stain's `settle` collected the live tile list into a fresh
`Vec` every step. `TileFilm` keeps it (no allocation in a per-step path).

## 2. `fluid_dynamics::porous` -- the soak law

```rust
/// Uptake of `offered` units by a pore volume holding `held` of `ceiling`: the offer
/// takes `offered / 255` of the headroom left (compression toward saturation).
pub fn soak_uptake(held: u8, offered: u32, ceiling: u8) -> u8;
/// What a porous bed refuses of an offer: offered less what it took.
pub fn refused(offered: u32, before: u8, after: u8) -> u32;
/// Infiltration from a running film: of `arrived` sub-units, the share the remaining pore
/// capacity drinks, `arrived * (min(ceiling, capacity) - held) / (capacity * passes)`.
pub fn infiltration(arrived: i32, held: u8, ceiling: u8, capacity: u8, passes: i32) -> i32;
```

Integer, branch-light, the stain's exact arithmetic. Ridgeline's `soak`, the refusal in
`absorb` and the drink in `gather_tile` call these. The soak *front* in water
(RAIN-SKIN/CANOPY-EDGE, per FLUIDS-B-INTEGRATION) is a different quantity (a wetting
depth, in the shader) and is not touched; these functions are there for it to use.

## 3. The slump -- a design fork, not moved yet

The ruling names "a standalone granular slump law (angle of repose)". The stain's
`slump` is not one: it fades the ridge's amplitude in place by `255 / HEAVE_LIFE_S` a
second and leaves the footprint alone (its own test says so: a true repose slump would
widen the band toward `2 * BURROW_HEAVE_M / REPOSE_TAN`). A repose law moves material to
neighbours (reads them, conserves volume, widens the band) and so changes the look and
the cost; the bar says the look is unchanged. This is put to Brandon as a question; until
it is answered the slump stays in Ridgeline.

## How terrain_stain calls them

`StainMap` keeps its buffers and implements `FilmField` on a borrowed view
(`pixels`, `pool_fraction`, `ground`, `sheet`, `pools`). `blood_settles` calls
`tile_film.step(&mut view, &LAW, &fall_x, &fall_z, drying, &mut touched)` where it called
`settle`, then `slump`. Painting calls `tile_film.wake` where it called `wake`. `LAW` is a
`const` built from the constants the stain already derives (`COURANT`, `GRADE_PER_HEAD`,
...), so no number crosses the crate boundary twice.
