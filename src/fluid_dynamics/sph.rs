//! Smoothed-particle hydrodynamics -- liquid that holds itself together.
//!
//! # The gap this fills
//!
//! `rs_physics` had two fluid models and neither could do a splash. [`FluidGrid3D`]
//! is Eulerian: excellent for smoke and large volumes, hopeless for droplets, since
//! resolving a 3 mm drop over a 280 m map would need a grid nobody can afford.
//! [`FluidParticle3D`] is a *solid particle moving through* a fluid medium -- a
//! bubble or a grain of sediment -- and its particles do not interact with each
//! other at all.
//!
//! SPH is the third thing: **the particles are the liquid**. Each one carries a
//! smoothed density sampled from its neighbours, and pressure, viscosity and
//! cohesion forces between them are what make a body of fluid behave like one. That
//! is what produces the behaviour a splash needs and the other two cannot give:
//! a sheet of blood that stretches, thins, and breaks into droplets on its own.
//!
//! [`FluidGrid3D`]: crate::fluid_dynamics::FluidGrid3D
//! [`FluidParticle3D`]: crate::fluid_dynamics::FluidParticle3D
//!
//! # The kernels
//!
//! Standard Muller-style SPH with an Akinci cohesion term:
//!
//! - **Poly6** for density. Smooth and cheap, but its gradient vanishes at zero
//!   distance, which is why it is not used for pressure.
//! - **Spiky gradient** for pressure. Its gradient *grows* as particles approach,
//!   which is exactly what stops them collapsing onto each other.
//! - **Viscosity Laplacian** for damping, which is what makes honey differ from
//!   water rather than just being slower.
//! - **Akinci cohesion** for surface tension. This is the term that makes a splash
//!   look like a liquid instead of like dust: without it particles disperse, and
//!   with it they pull into rounded blobs and separate into droplets.
//!
//! # Layout
//!
//! Structure-of-arrays throughout: one `Vec<f64>` a component. Until 2026-09-28 the
//! particle state was `Vec<[f64; 3]>` and the neighbour walk chased a `sorted` index
//! into it for every candidate; now each step copies the state into cell order once,
//! so every candidate range the walk reads is contiguous memory.
//!
//! The cell hash puts a row of cells (fixed `y` and `z`) in consecutive buckets, so
//! the three cells of a row that a particle's neighbourhood covers are one run of the
//! sorted arrays: at most nine runs a cell rather than twenty-seven hash lookups a
//! particle. The nine intervals are merged where two rows' buckets overlap, so every
//! bucket is walked once, and they are built once a cell (the sorted order keeps a
//! cell's particles together). Every particle within a smoothing radius lies in one of
//! the 27 cells, so the distance test alone is the membership test: a particle from a
//! far cell that shares a bucket fails it, and none is visited twice.
//!
//! The density pass records each particle's neighbours (a `u32` a neighbour, in a
//! list owned by a fixed chunk of particles), and the force pass reads that list
//! rather than searching again. Positions do not move between the two passes, so the
//! list is exact.
//!
//! # Parallelism and determinism
//!
//! Density and forces run over fixed chunks of [`SPH_CHUNK`] particles with rayon, on
//! whatever pool the caller is running in (`pool.install(|| fluid.step(..))`), or the
//! global one; the O(n) passes (binning, the gathers into and out of cell order, the
//! move) run over larger fixed chunks the same way. The answer is **bit-identical at any thread count**, by construction
//! rather than by care:
//!
//! - Every particle's sums are gathers, written only to that particle's own slot, in
//!   an order fixed by the data: the nine rows in order, each row's slots in sorted
//!   order. No thread ever adds into another particle's total.
//! - The sums run in four interleaved partial sums (a candidate's lane is its position
//!   in its row, a neighbour's its position in the list) combined as
//!   `(a0 + a1) + (a2 + a3)`, which lets the adds pipeline and vectorise without
//!   reassociation that a thread count could change.
//! - The counting sort is serial and O(n), and it is what fixes the order. Every pass
//!   back to particle order is a gather through the sort's inverse, one particle a
//!   slot, so it too is independent of the schedule. The ground callback runs serially:
//!   it is the caller's closure and is not assumed to be `Sync`.
//!
//! `parallel_steps_are_bit_identical_at_any_thread_count` asserts it at 1, 3 and 8
//! threads. No hash iteration and no transcendentals in the inner loops, so the
//! solver is deterministic on one machine; across machines it is as deterministic as
//! `f64` `sqrt` and division, which IEEE 754 fixes.
//!
//! # Solids
//!
//! [`SphFluid::step`] knows one solid, the ground. [`SphFluid::step_with_solids`] adds a
//! [`SphSolids`] set the caller fills each frame: capsules (a limb, a boot, a blade, with
//! a surface velocity at each end, blended along the axis) and boxes yawed about +y
//! (debris, a crate, a corpse's bounds, with one velocity).
//!
//! **One way.** Solids push the liquid; the liquid never pushes a solid. The set is
//! borrowed read-only for the step and nothing is written back, so no fluid state can
//! reach anything a lockstep simulation reads.
//!
//! **Binning.** After the particle grid is built, each solid's bounds, grown by the
//! longest ray a particle can cast (the speed ceiling's travel, `0.4 h`, plus the
//! contact radius, `h / 4`), are binned into the hash buckets of the occupied cells they
//! cover, clipped to the cells the fluid spans: one entry a solid for each cell in its
//! reach that holds a particle, filed under that cell's bucket. Any particle that can
//! meet a solid starts the substep within that reach, so it finds the solid in its own
//! cell's bin, one bucket read: the bins are the cells the ray can cross, gathered on the
//! solid's side. A bucket's solids are in solid order, which is the order a particle
//! tests them in.
//!
//! Two walks find the occupied cells, and write the same bins. The **cell walk** visits
//! each solid's clipped box cell by cell: a hash, a bucket read and a look at the
//! bucket's cells (so a cell is binned for its own particles, never for another cell's
//! that hashes beside it), about 6 ns a cell. The **particle scan** tests every
//! particle's cell against the boxes instead: the boxes are listed in a grid of blocks
//! over their union, sized each step by the cost it implies, and a particle outside the
//! union costs one branch-free test, 64 to a word, about 1.4 ns a particle. A long solid
//! over a wide, sparse fluid (a 4 m hull over 16,384 drops spread across 20 m covers
//! 200,000 cells) costs the walk milliseconds and the scan tens of microseconds; a limb in
//! a pool costs the walk a few hundred cells. The step takes the scan once the clipped
//! boxes hold more than a third of the particle count in cells between them, the
//! measured crossover (`SCAN_CROSSOVER`). Serial, after one serial min and max over the
//! particle cells (O(n) integer work that vectorises), into reused buffers: a counting
//! sort over only the buckets touched, which are zeroed again after the step, so the
//! table is never cleared whole.
//!
//! **Contact.** In the move, after the velocity update and before the ground, a particle
//! whose bin is not empty casts a ray from where it was to where it is going, extended
//! by the contact radius. Closed form, no iteration: a capsule is the nearest of its
//! cylinder side and its two end spheres (three square roots), a box is the slab test in
//! its own frame (three divisions; its yaw's sine and cosine are taken once, when it is
//! pushed). At the nearest hit the particle is set on the surface a contact radius out
//! along the normal; a particle that starts inside a solid (the solid moved onto it) is
//! pushed out along the nearest normal instead. Its velocity relative to the surface
//! keeps `restitution` of an approaching normal part, decays its tangential part at the
//! ground's `friction` rate, and takes on the surface velocity: the ground's response,
//! with no new coefficient. Swept rather than a point test, so a droplet at the speed
//! ceiling cannot pass through a box one spacing thick or a blade. One contact a
//! substep: a particle pushed from one solid into another meets the second next substep.
//!
//! **Settling.** A particle a solid's contact set on its surface, moving slower than
//! the ground's settle speed relative to that surface (`SETTLE_SPEED`, the ground's own
//! test seen from the surface's frame), is still on that solid; held for `SETTLE_TIME`
//! on the same surface, it is drained by [`SphFluid::drain_settled`] with
//! [`Settled::on_solid`] naming the solid's index in the set (capsules first, then
//! boxes, so a set refilled in the same order keeps its indices). The ground, which runs
//! after the solids, wins a particle that touched both, and a particle that comes to rest
//! on another surface counts again. One known gap: the contact casts the particle's world
//! displacement, so a drop carried along a moving surface meets it only every few
//! substeps and does not settle on it (pinned by an ignored test in the solids tests).
//!
//! **The slope.** With solids the ground has a normal too: a particle in ground contact
//! samples `ground_height` a rest spacing along +x and along +z (two extra calls, and
//! only in contact, so a particle in flight pays nothing) and meets the ground along
//! that slope's normal, so a drop on an incline runs downhill (SPH-F5). Where both
//! differences are zero the level response runs unchanged, so on level ground with no
//! solid in reach `step_with_solids` is bit-identical to `step`.
//!
//! **Cost.** A solid covers about `(L / h + 2.3)` cells along each axis of length `L`
//! (its extent plus `1.3 h` of reach); a 0.4 m limb of radius 6 cm in blood (`h` = 4 cm)
//! spans 0.52 m by 0.12 m, about `5 x 15 x 5`, near 400 cells. The binning costs the
//! cheaper of the walk over those cells and the scan over the particles (above); it
//! writes an entry only where particles are. A particle pays one bucket read when no
//! solid is near it, and the contact test against each solid in its bin when one is.
//! Solids that reach no particle write no entry, and then the move is the plain one: the
//! binning is the whole price.
//!
//! **Memory.** A capsule is 144 bytes and a box 88 in [`SphSolids`]. The fluid's bins
//! are 8 bytes a bucket (two to four buckets a particle, allocated on the first step with
//! solids) plus 12 bytes an entry and 4 a touched bucket, and 28 bytes a solid in reach
//! for its clipped box; the scan's block grid is 4 bytes a block plus 4 a block a box
//! lists, its size picked each step by cost (one block for a lone hull). All kept at
//! their high-water marks.
//!
//! # Examples
//!
//! ```
//! use rs_physics::fluid_dynamics::{SphFluid, SphParams};
//!
//! let mut blood = SphFluid::new(SphParams::blood(), 512).unwrap();
//! for i in 0..64 {
//!     let (x, z) = ((i % 8) as f64 * 0.01, (i / 8) as f64 * 0.01);
//!     blood.spawn([x, 0.5, z], [0.0, -1.0, 0.0]);
//! }
//! for _ in 0..240 {
//!     blood.step(1.0 / 240.0, 9.81, |_, _| 0.0);
//! }
//! assert!((0..blood.len()).all(|i| blood.position(i)[1] >= 0.0));
//! ```

#![warn(missing_docs)]

use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::utils::PhysicsError;

#[path = "sph_solids.rs"]
mod solids;
pub use solids::{SphSolidStats, SphSolids};

/// Fluid behaviour. The four numbers that separate water from blood from honey.
#[derive(Debug, Clone, Copy)]
pub struct SphParams {
    /// Target density, kg/m^3. Pressure pushes particles apart above this and lets
    /// them draw together below it.
    pub rest_density: f64,
    /// Pressure stiffness. Higher resists compression harder but needs a smaller
    /// timestep; too high and the simulation explodes.
    pub stiffness: f64,
    /// Internal friction. Water is near zero, blood is a few times that, honey is
    /// enormous.
    pub viscosity: f64,
    /// Surface tension. **The term that makes droplets.** At zero the fluid behaves
    /// like a gas of independent particles; raised, it pulls the surface into
    /// rounded blobs and pinches sheets into drops.
    pub cohesion: f64,
    /// Kernel support radius, metres. Every interaction is zero beyond this, so it
    /// sets both the physics and the cost.
    pub smoothing_radius: f64,
    /// Mass of a single particle, kg.
    pub particle_mass: f64,
    /// Fraction of normal velocity kept when hitting the ground.
    pub restitution: f64,
    /// Fraction of tangential velocity kept when hitting the ground. Below one, a
    /// splash spreads and then stops rather than sliding forever.
    ///
    /// Stated per 1/240 s substep of contact and applied as the equivalent decay
    /// rate, so how far a drop slides does not depend on the substep it is solved
    /// at.
    pub friction: f64,
}

impl SphParams {
    /// Set the smoothing radius and particle mass from a target particle spacing.
    ///
    /// **These three numbers are not independent, and getting that wrong is silent.**
    /// A particle represents a cube of fluid `spacing` on a side, so its mass must be
    /// `rest_density * spacing^3`; and the kernel needs roughly two particles of
    /// support in every direction, so the smoothing radius is about twice the
    /// spacing.
    ///
    /// Set them inconsistently and the solver does not fail -- it computes a sampled
    /// density far below the rest density, which makes pressure clamp to zero, which
    /// silently removes incompressibility altogether. The fluid still moves, still
    /// looks plausible, and is no longer a fluid. The first draft of this module had
    /// exactly that bug and a density test is what caught it.
    ///
    /// # Arguments
    ///
    /// * `spacing` - distance between neighbouring particles at rest, metres. It is
    ///   the real handle on cost: halving it is eight times the particles for the
    ///   same volume.
    ///
    /// # Returns
    ///
    /// The same parameters with `smoothing_radius` and `particle_mass` derived.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::SphParams;
    ///
    /// let p = SphParams::water().with_spacing(0.05);
    /// assert_eq!(p.smoothing_radius, 0.1);
    /// assert!((p.particle_mass - 1000.0 * 0.05f64.powi(3)).abs() < 1e-12);
    /// ```
    pub fn with_spacing(mut self, spacing: f64) -> SphParams {
        self.smoothing_radius = spacing * 2.0;
        self.particle_mass = self.rest_density * spacing * spacing * spacing;
        self
    }

    /// Blood: a little denser than water, several times as viscous, and with high
    /// surface tension -- which is why it travels as discrete drops rather than a
    /// spray, and why a splash beads on the ground instead of spreading thin.
    ///
    /// Spaced at 2 cm, so a particle is about 8 ml and a splash of a few dozen is a
    /// realistic quarter-litre rather than a bathtub.
    ///
    /// # Returns
    ///
    /// Blood's parameters at a 2 cm spacing.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::SphParams;
    ///
    /// assert_eq!(SphParams::blood().rest_density, 1060.0);
    /// ```
    pub fn blood() -> SphParams {
        SphParams {
            rest_density: 1060.0,
            stiffness: 60.0,
            viscosity: 9.0,
            cohesion: 0.55,
            smoothing_radius: 0.0,
            particle_mass: 0.0,
            restitution: 0.08,
            friction: 0.35,
        }
        .with_spacing(0.02)
    }

    /// Napalm: gelled hydrocarbon fuel.
    ///
    /// Every number here is doing the same job -- making the fluid **refuse to
    /// spread**. Thickened fuel is the whole point of the weapon: petrol splashes,
    /// runs off and burns out in seconds, so it is gelled until it clings to whatever
    /// it lands on and burns there. That behaviour is not decoration on top of the
    /// mechanic, it *is* the mechanic.
    ///
    /// - Density 900 kg/m^3, lighter than water: it is a hydrocarbon.
    /// - Viscosity an order of magnitude above blood, which is what makes it crawl
    ///   downhill in lobes instead of sheeting out like spilt water.
    /// - Cohesion far above anything else here. This is the term that produces
    ///   surface tension, and surface tension is what makes gel form fat rounded
    ///   blobs with a visible skin rather than a thin film.
    /// - Restitution near zero and friction near one: it splats and stays put. A gel
    ///   that bounced would read as rubber.
    ///
    /// Spaced at 12 cm rather than blood's 2 cm. This is a fluid measured in tens of
    /// kilograms thrown across metres of ground, not a droplet-scale splash, and
    /// resolving it at droplet scale would spend the entire particle budget on one
    /// shell.
    ///
    /// # Returns
    ///
    /// Napalm's parameters at a 12 cm spacing.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::SphParams;
    ///
    /// assert!(SphParams::napalm().viscosity > SphParams::blood().viscosity);
    /// ```
    pub fn napalm() -> SphParams {
        SphParams {
            rest_density: 900.0,
            stiffness: 45.0,
            viscosity: 95.0,
            cohesion: 2.6,
            smoothing_radius: 0.0,
            particle_mass: 0.0,
            restitution: 0.02,
            friction: 0.88,
        }
        .with_spacing(0.12)
    }

    /// Water: lighter, thinner, and less cohesive, so it sheets and spreads where
    /// blood beads.
    ///
    /// # Returns
    ///
    /// Water's parameters at a 2 cm spacing.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::SphParams;
    ///
    /// assert!(SphParams::water().cohesion < SphParams::blood().cohesion);
    /// ```
    pub fn water() -> SphParams {
        SphParams {
            rest_density: 1000.0,
            stiffness: 80.0,
            viscosity: 2.5,
            cohesion: 0.22,
            smoothing_radius: 0.0,
            particle_mass: 0.0,
            restitution: 0.12,
            friction: 0.55,
        }
        .with_spacing(0.02)
    }
}

/// A particle that has come to rest, reported by [`SphFluid::drain_settled`].
///
/// This is the handoff a splash needs: the fluid runs while it is *moving and
/// interesting*, and the moment a drop stops it leaves the simulation and becomes a
/// mark on the ground. Keeping settled particles in the solver would cost neighbour
/// searches forever to render something that is no longer changing.
#[derive(Debug, Clone, Copy)]
pub struct Settled {
    /// Where it came to rest, metres.
    pub position: [f64; 3],
    /// How much fluid this particle represented, kg. A stain can scale with it.
    pub mass: f64,
    /// Velocity at the moment it came to rest.
    ///
    /// Reported because *how* a drop arrived decides what it leaves. A droplet
    /// landing straight down makes a round mark; one still carrying sideways speed
    /// smears along its travel. Without this the caller can only draw circles, which
    /// is exactly what a splash does not look like.
    ///
    /// World velocity, m/s. A drop that settled on a moving solid reports about the
    /// solid's surface velocity where it rests (it was still relative to that surface,
    /// not to the ground); subtract the surface's own velocity for the drop's motion
    /// across it.
    pub velocity: [f64; 3],
    /// What it came to rest on: the index of a solid in the [`SphSolids`] handed to the
    /// last [`SphFluid::step_with_solids`] (capsules first, then boxes, in the order they
    /// were pushed), or `None` for the ground.
    ///
    /// A drop on a hull, a corpse or a limb is still when its velocity relative to that
    /// solid's surface is, so it settles riding a moving hull; the caller can parent its
    /// stain to whatever that index named this frame. Added in 0.3.5.
    pub on_solid: Option<u32>,
}

/// Wall time of each phase of the last [`SphFluid::step`], read with
/// [`SphFluid::phase_times`].
///
/// Taken in-process with [`Instant`] on every step, which costs five clock reads, so a
/// caller can put the phases on a HUD or in a trace without a profiler, and a
/// benchmark can compare thread counts inside one process rather than across a
/// machine whose clock drifts with heat.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SphPhaseTimes {
    /// Binning into cells, the counting sort, and the copy into cell order. Serial.
    pub grid: Duration,
    /// The neighbour search and the density and pressure it gives. Parallel.
    pub density: Duration,
    /// Pressure, viscosity and cohesion over the recorded neighbours, and the velocity
    /// update. Parallel, then a serial scatter.
    pub forces: Duration,
    /// The speed cap, the move and the ground. Serial: the ground is the caller's
    /// closure.
    pub integrate: Duration,
}

impl SphPhaseTimes {
    /// The whole step.
    ///
    /// # Returns
    ///
    /// The sum of the four phases.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    /// use rs_physics::fluid_dynamics::SphPhaseTimes;
    ///
    /// let t = SphPhaseTimes { grid: Duration::from_micros(1), ..Default::default() };
    /// assert_eq!(t.total(), Duration::from_micros(1));
    /// ```
    pub fn total(&self) -> Duration {
        self.grid + self.density + self.forces + self.integrate
    }
}

/// Particles per unit of parallel work.
///
/// Both parallel passes hand rayon whole chunks, so this is the scheduling grain, and
/// it never changes an answer: each particle's sums are its own whichever chunk or
/// thread computes them. At about a microsecond a particle a chunk is about 64 us of
/// work, far above the few microseconds a rayon split costs, and small enough that
/// 1,024 particles still give sixteen chunks to spread over eight threads.
pub const SPH_CHUNK: usize = 64;

/// Particles per unit of parallel work in the O(n) streaming passes (binning, the
/// gathers into and out of cell order, the move).
///
/// A few nanoseconds a particle, so the grain is sixteen times [`SPH_CHUNK`]: about
/// 5 to 20 us a piece, still well above a split's cost. Below it a pass runs as one
/// piece on the calling thread. Like `SPH_CHUNK`, it never changes an answer.
const STREAM_CHUNK: usize = SPH_CHUNK * 16;

/// One chunk's neighbour lists: `index[end[k - 1]..end[k]]` are the sorted slots
/// of particle `k`'s neighbours, itself excluded.
///
/// Owned per chunk so the parallel density pass can append without coordination.
/// Cleared rather than freed, so after the first steps nothing here allocates.
#[derive(Debug, Clone, Default)]
struct NeighbourList {
    index: Vec<u32>,
    end: Vec<u32>,
}

/// A body of SPH fluid.
///
/// Structure-of-arrays: every pass touches positions and velocities and little else.
/// See the module documentation for the layout and the determinism argument.
#[derive(Debug, Clone)]
pub struct SphFluid {
    // Particle state, in the order particles were spawned (compacted by
    // `drain_settled`). Indices are stable across a step.
    px: Vec<f64>,
    py: Vec<f64>,
    pz: Vec<f64>,
    /// Where each particle was at the *start* of the last completed substep.
    ///
    /// Kept so a renderer running faster than the solver can draw **between** two
    /// solved states instead of on them. SPH has to substep -- pressure is stiff -- so
    /// a caller accumulating frame time always finishes a frame with a remainder it
    /// cannot spend, and drawing `pos` directly shows the fluid advancing in visible
    /// quanta at the substep rate rather than tracking render time.
    ///
    /// Maintained in lockstep with the position by every operation that changes the
    /// particle set -- `spawn` pushes, `drain_settled` swap-removes, `clear` clears --
    /// so equal lengths are an invariant of the type rather than something
    /// [`SphFluid::interpolated_position`] has to check.
    ppx: Vec<f64>,
    ppy: Vec<f64>,
    ppz: Vec<f64>,
    vx: Vec<f64>,
    vy: Vec<f64>,
    vz: Vec<f64>,
    density: Vec<f64>,
    /// Seconds this particle has been below the settle speed. A drop is only retired
    /// once it has been still for a moment, so a drip that is briefly slow at the
    /// apex of a bounce is not mistaken for one that has stopped.
    still_for: Vec<f64>,
    /// The surface each particle's `still_for` is counted against: the index of a solid
    /// in the last step's [`SphSolids`], or [`NO_SOLID`] for the ground. A particle that
    /// comes to rest on another surface starts its count again.
    rest_on: Vec<u32>,

    params: SphParams,
    capacity: usize,

    // Scratch, all owned by the solver, sized for `capacity` in `new`, and cleared
    // rather than reallocated. The engine's rule: a `step` that allocates is a `step`
    // that stutters, and this one runs every frame for as long as there is fluid on
    // screen.
    /// Cell coordinate of each particle, in particle order.
    cell_x: Vec<i32>,
    cell_y: Vec<i32>,
    cell_z: Vec<i32>,
    /// Which hash bucket each particle landed in, in particle order.
    bucket_of: Vec<u32>,
    /// Prefix sums into the sorted arrays, one entry per bucket plus a terminator.
    bucket_start: Vec<u32>,
    /// Write cursor for the counting sort.
    bucket_cursor: Vec<u32>,
    /// Sorted slot to particle index.
    order: Vec<u32>,
    /// Particle index to sorted slot: the inverse of `order`, so every pass back to
    /// particle order is a parallel gather rather than a serial scatter.
    slot_of: Vec<u32>,
    // The state in sorted (cell) order, which is what the neighbour walk reads.
    sx: Vec<f64>,
    sy: Vec<f64>,
    sz: Vec<f64>,
    svx: Vec<f64>,
    svy: Vec<f64>,
    svz: Vec<f64>,
    s_cell_x: Vec<i32>,
    s_cell_y: Vec<i32>,
    s_cell_z: Vec<i32>,
    s_density: Vec<f64>,
    s_pressure: Vec<f64>,
    s_inv_density: Vec<f64>,
    /// Acceleration in sorted order, accumulated before any velocity is touched so
    /// the result does not depend on the order particles are visited.
    ax: Vec<f64>,
    ay: Vec<f64>,
    az: Vec<f64>,
    lists: Vec<NeighbourList>,
    /// Bucket count, always a power of two so the hash reduces with a mask.
    table_mask: usize,
    times: SphPhaseTimes,
    /// The solids of the current `step_with_solids`, binned by bucket. Empty, and owning
    /// no memory, until the first step with solids.
    solid_bins: solids::SolidBins,
    solid_stats: SphSolidStats,
    /// Per particle, in particle order, written by the move with solids: the solid whose
    /// contact set it on its surface this substep, slower relative to that surface than
    /// [`SETTLE_SPEED`], or [`NO_SOLID`]. Read by the ground pass.
    still_on_solid: Vec<u32>,
}

/// Below this speed, and touching ground, a particle is considered to have landed.
///
/// On a solid the same number is applied to the particle's velocity relative to the
/// surface it was set on (the contact's surface velocity at that point), not to its world
/// velocity. That is the ground's own test seen from the surface: the ground is a surface
/// at rest, and a speed relative to it is a world speed, so the ground's threshold moved
/// into a moving surface's frame is this one, unchanged, and a drop riding a hull at
/// 1 m/s is as still as one on the ground. What it admits is the same in both frames:
/// held for [`SETTLE_TIME`], at most `SETTLE_SPEED * SETTLE_TIME` = 8.75 cm of creep
/// across the surface, and a drop at rest on a solid sits far below it, since each
/// substep's gravity, `g dt` (0.041 m/s at 240 Hz), is all the response leaves it after
/// keeping `restitution` of the approach. A bounce that leaves it faster than this
/// relative to the surface starts the count again, as a bounce on the ground does.
const SETTLE_SPEED: f64 = 0.35;
/// ...and it must stay that way for this long before it is retired.
const SETTLE_TIME: f64 = 0.25;

/// `rest_on` and `still_on_solid`: no solid, the ground. Solid indices stop short of it,
/// since [`SphSolids`] holds fewer than `u32::MAX` solids.
const NO_SOLID: u32 = u32::MAX;

/// Buckets for `n` particles: about two a particle keeps collisions rare, and a power
/// of two reduces the hash with a mask.
fn table_size(n: usize) -> usize {
    (n * 2).next_power_of_two().max(64)
}

impl SphFluid {
    /// An empty fluid that will hold at most `capacity` particles.
    ///
    /// Every buffer the step uses is allocated here, for `capacity`. The neighbour
    /// lists are the one exception: their length is the number of neighbours, which
    /// depends on how packed the fluid is, so they grow to their high-water mark over
    /// the first steps and are reused after.
    ///
    /// # Arguments
    ///
    /// * `params` - the fluid; see [`SphParams::with_spacing`] for the three numbers
    ///   that have to agree.
    /// * `capacity` - the most particles alive at once.
    ///
    /// # Returns
    ///
    /// An empty fluid.
    ///
    /// # Errors
    ///
    /// * [`PhysicsError::InvalidDistance`] if the smoothing radius is not a positive
    ///   finite number.
    /// * [`PhysicsError::InvalidMass`] if the particle mass is not a positive finite
    ///   number.
    /// * [`PhysicsError::CalculationError`] if the rest density is not a positive finite
    ///   number, if the stiffness, viscosity or cohesion is negative or not finite, or if
    ///   `capacity` does not fit the `u32` neighbour indices.
    /// * [`PhysicsError::InvalidCoefficient`] if the restitution or the friction is
    ///   outside `0..=1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{SphFluid, SphParams};
    ///
    /// let fluid = SphFluid::new(SphParams::water(), 1_000).unwrap();
    /// assert_eq!(fluid.capacity(), 1_000);
    ///
    /// let mut bad = SphParams::water();
    /// bad.particle_mass = 0.0;
    /// assert!(SphFluid::new(bad, 1_000).is_err());
    /// ```
    pub fn new(params: SphParams, capacity: usize) -> Result<SphFluid, PhysicsError> {
        // Written as "is not a positive finite number" rather than `<= 0.0`, which
        // NaN passes, and a NaN here turns every particle NaN on the first step.
        let positive = |v: f64| v > 0.0 && v.is_finite();
        let non_negative = |v: f64| v >= 0.0 && v.is_finite();
        let fraction = |v: f64| (0.0..=1.0).contains(&v);

        if !positive(params.smoothing_radius) {
            return Err(PhysicsError::InvalidDistance);
        }
        if !positive(params.particle_mass) {
            return Err(PhysicsError::InvalidMass);
        }
        if !positive(params.rest_density) {
            return Err(PhysicsError::CalculationError(
                "rest density must be positive".to_string(),
            ));
        }
        // Negative stiffness or viscosity feeds energy in instead of taking it out.
        if !non_negative(params.stiffness)
            || !non_negative(params.viscosity)
            || !non_negative(params.cohesion)
        {
            return Err(PhysicsError::CalculationError(
                "stiffness, viscosity and cohesion must be finite and non-negative"
                    .to_string(),
            ));
        }
        // Fractions of velocity kept at the ground: above one, every contact gains
        // energy.
        if !fraction(params.restitution) || !fraction(params.friction) {
            return Err(PhysicsError::InvalidCoefficient);
        }
        if capacity >= u32::MAX as usize / 2 {
            return Err(PhysicsError::CalculationError(
                "capacity must fit the solver's u32 neighbour indices".to_string(),
            ));
        }

        let f = || Vec::<f64>::with_capacity(capacity);
        let c = || Vec::<i32>::with_capacity(capacity);
        let table = table_size(capacity);
        Ok(SphFluid {
            px: f(),
            py: f(),
            pz: f(),
            ppx: f(),
            ppy: f(),
            ppz: f(),
            vx: f(),
            vy: f(),
            vz: f(),
            density: f(),
            still_for: f(),
            rest_on: Vec::with_capacity(capacity),
            params,
            capacity,
            cell_x: c(),
            cell_y: c(),
            cell_z: c(),
            bucket_of: Vec::with_capacity(capacity),
            bucket_start: Vec::with_capacity(table + 1),
            bucket_cursor: Vec::with_capacity(table + 1),
            order: Vec::with_capacity(capacity),
            slot_of: Vec::with_capacity(capacity),
            sx: f(),
            sy: f(),
            sz: f(),
            svx: f(),
            svy: f(),
            svz: f(),
            s_cell_x: c(),
            s_cell_y: c(),
            s_cell_z: c(),
            s_density: f(),
            s_pressure: f(),
            s_inv_density: f(),
            ax: f(),
            ay: f(),
            az: f(),
            lists: vec![NeighbourList::default(); capacity.div_ceil(SPH_CHUNK)],
            table_mask: 0,
            times: SphPhaseTimes::default(),
            solid_bins: solids::SolidBins::default(),
            solid_stats: SphSolidStats::default(),
            still_on_solid: Vec::with_capacity(capacity),
        })
    }

    /// How many particles are alive.
    ///
    /// # Returns
    ///
    /// The live count, at most [`Self::capacity`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{SphFluid, SphParams};
    ///
    /// let mut fluid = SphFluid::new(SphParams::blood(), 8).unwrap();
    /// fluid.spawn([0.0, 1.0, 0.0], [0.0; 3]);
    /// assert_eq!(fluid.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.px.len()
    }

    /// Whether no particle is alive.
    ///
    /// # Returns
    ///
    /// `true` when [`Self::len`] is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{SphFluid, SphParams};
    ///
    /// assert!(SphFluid::new(SphParams::blood(), 8).unwrap().is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.px.is_empty()
    }

    /// The most particles the fluid holds, fixed at construction.
    ///
    /// # Returns
    ///
    /// The capacity passed to [`Self::new`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{SphFluid, SphParams};
    ///
    /// assert_eq!(SphFluid::new(SphParams::blood(), 8).unwrap().capacity(), 8);
    /// ```
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The fluid's parameters.
    ///
    /// # Returns
    ///
    /// The [`SphParams`] passed to [`Self::new`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{SphFluid, SphParams};
    ///
    /// let fluid = SphFluid::new(SphParams::napalm(), 8).unwrap();
    /// assert_eq!(fluid.params().rest_density, 900.0);
    /// ```
    pub fn params(&self) -> &SphParams {
        &self.params
    }

    /// Remove every particle at once, keeping the allocations.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{SphFluid, SphParams};
    ///
    /// let mut fluid = SphFluid::new(SphParams::blood(), 8).unwrap();
    /// fluid.spawn([0.0, 1.0, 0.0], [0.0; 3]);
    /// fluid.clear();
    /// assert!(fluid.is_empty());
    /// ```
    pub fn clear(&mut self) {
        for v in [
            &mut self.px,
            &mut self.py,
            &mut self.pz,
            &mut self.ppx,
            &mut self.ppy,
            &mut self.ppz,
            &mut self.vx,
            &mut self.vy,
            &mut self.vz,
            &mut self.density,
            &mut self.still_for,
        ] {
            v.clear();
        }
        self.rest_on.clear();
    }

    /// Position of particle `i`, metres, as of the last completed substep.
    ///
    /// # Arguments
    ///
    /// * `i` - particle index, `0..len()`. Stable across a step; `drain_settled`
    ///   compacts with `swap_remove`.
    ///
    /// # Returns
    ///
    /// `[x, y, z]`.
    ///
    /// # Panics
    ///
    /// If `i >= len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{SphFluid, SphParams};
    ///
    /// let mut fluid = SphFluid::new(SphParams::blood(), 8).unwrap();
    /// fluid.spawn([1.0, 2.0, 3.0], [0.0; 3]);
    /// assert_eq!(fluid.position(0), [1.0, 2.0, 3.0]);
    /// ```
    pub fn position(&self, i: usize) -> [f64; 3] {
        [self.px[i], self.py[i], self.pz[i]]
    }

    /// Where particle `i` was at the start of the last completed substep.
    ///
    /// The other half of [`Self::interpolated_position`]; exposed so a caller that
    /// wants to do its own blending does not have to keep a shadow copy that
    /// `drain_settled`'s `swap_remove` would silently misalign.
    ///
    /// # Arguments
    ///
    /// * `i` - particle index, `0..len()`.
    ///
    /// # Returns
    ///
    /// `[x, y, z]`, metres.
    ///
    /// # Panics
    ///
    /// If `i >= len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{SphFluid, SphParams};
    ///
    /// let mut fluid = SphFluid::new(SphParams::blood(), 8).unwrap();
    /// fluid.spawn([0.0, 1.0, 0.0], [0.0; 3]);
    /// fluid.step(1.0 / 240.0, 9.81, |_, _| 0.0);
    /// assert_eq!(fluid.previous_position(0), [0.0, 1.0, 0.0]);
    /// ```
    pub fn previous_position(&self, i: usize) -> [f64; 3] {
        [self.ppx[i], self.ppy[i], self.ppz[i]]
    }

    /// Particle `i` drawn `alpha` of the way through the last completed substep.
    ///
    /// # Why a solver offers this at all
    ///
    /// Because the alternative is that every caller gets it wrong in the same way.
    /// SPH must substep, so a renderer accumulating frame time always ends a frame
    /// with a remainder shorter than one substep and no way to spend it. Reading
    /// [`Self::position`] draws the last *solved* state, so the fluid advances in
    /// quanta at the substep rate however smooth the frame rate is -- visible
    /// immediately as a liquid running downhill in steps.
    ///
    /// The obvious client-side fix -- keep a `Vec` of previous positions -- does not
    /// work, because [`Self::drain_settled`] compacts with `swap_remove`. A shadow
    /// buffer would need to mirror that, so the buffer belongs to whatever performs
    /// the removal. That is this type.
    ///
    /// # What it is not
    ///
    /// Read-only, and it never touches solver state: the fluid still advances by
    /// whole substeps and integrates nothing between them. This blends the two most
    /// recent solved states the way a fixed-tick renderer blends ticks, so the drawn
    /// position lags the solved one by up to one substep. At the rates SPH needs --
    /// 240 Hz is four milliseconds -- that lag is well under a frame.
    ///
    /// `alpha` is clamped to `[0, 1]`, so an accumulator that ran past a substep
    /// cannot extrapolate a particle somewhere the solver never put it.
    ///
    /// # Arguments
    ///
    /// * `i` - particle index, `0..len()`.
    /// * `alpha` - fraction of a substep elapsed since the last one completed,
    ///   normally `accumulator / substep`. NaN or infinite draws the solved state.
    ///
    /// # Returns
    ///
    /// `[x, y, z]`, metres.
    ///
    /// # Panics
    ///
    /// If `i >= len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{SphFluid, SphParams};
    ///
    /// let mut fluid = SphFluid::new(SphParams::blood(), 8).unwrap();
    /// fluid.spawn([0.0, 1.0, 0.0], [0.0, -1.0, 0.0]);
    /// fluid.step(1.0 / 240.0, 0.0, |_, _| 0.0);
    /// let halfway = fluid.interpolated_position(0, 0.5)[1];
    /// assert!(halfway < 1.0 && halfway > fluid.position(0)[1]);
    /// ```
    pub fn interpolated_position(&self, i: usize, alpha: f64) -> [f64; 3] {
        // NaN would take the false branch of every comparison inside `clamp` and is
        // not representable as "somewhere between", so it resolves to the solved
        // state rather than propagating into a vertex position.
        let a = if alpha.is_finite() {
            alpha.clamp(0.0, 1.0)
        } else {
            1.0
        };
        let lerp = |from: f64, to: f64| from + (to - from) * a;
        [
            lerp(self.ppx[i], self.px[i]),
            lerp(self.ppy[i], self.py[i]),
            lerp(self.ppz[i], self.pz[i]),
        ]
    }

    /// The fastest a particle can actually travel in this solver, m/s, at `dt`.
    ///
    /// # Why this is public
    ///
    /// Because [`Self::step`] enforces it silently and callers have been building
    /// designs on top of speeds it deletes. The cap is a CFL condition -- a particle
    /// may not cross more than a fraction of a smoothing radius in one step, or the
    /// neighbour search stops finding the neighbours whose forces were meant to act
    /// on it -- so it is not negotiable and not a tuning knob. But it *is* invisible:
    /// [`Self::spawn`] accepts any velocity and the next step quietly truncates it,
    /// which reads as the solver working and the emitter's numbers not mattering.
    ///
    /// Blood spawned along a 42 m/s round at "40% of its speed" and blood spawned at
    /// wound-cavity speeds arrive here as the same number. Nothing errors, nothing
    /// logs, and the two-population spray the caller wrote is one population. Asking
    /// first is the difference between a designed spread and an accidental constant.
    ///
    /// Raise it by raising the smoothing radius (a coarser, cheaper fluid) or by
    /// stepping faster; there is no third way, and both are the caller's decision.
    ///
    /// # Arguments
    ///
    /// * `dt` - the substep this fluid is stepped with, seconds.
    ///
    /// # Returns
    ///
    /// The speed cap, m/s.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{SphFluid, SphParams};
    ///
    /// let fluid = SphFluid::new(SphParams::blood(), 8).unwrap();
    /// // 0.4 of a 4 cm smoothing radius a 1/240 s substep.
    /// assert!((fluid.speed_ceiling(1.0 / 240.0) - 3.84).abs() < 1e-9);
    /// ```
    pub fn speed_ceiling(&self, dt: f64) -> f64 {
        cfl_speed_ceiling(self.params.smoothing_radius, dt)
    }

    /// Velocity of particle `i`, m/s.
    ///
    /// # Arguments
    ///
    /// * `i` - particle index, `0..len()`.
    ///
    /// # Returns
    ///
    /// `[x, y, z]`.
    ///
    /// # Panics
    ///
    /// If `i >= len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{SphFluid, SphParams};
    ///
    /// let mut fluid = SphFluid::new(SphParams::blood(), 8).unwrap();
    /// fluid.spawn([0.0, 1.0, 0.0], [0.5, 0.0, 0.0]);
    /// assert_eq!(fluid.velocity(0), [0.5, 0.0, 0.0]);
    /// ```
    pub fn velocity(&self, i: usize) -> [f64; 3] {
        [self.vx[i], self.vy[i], self.vz[i]]
    }

    /// Sampled density at a particle, which for a surface particle is well below
    /// rest density. Useful for rendering: the sparse ones are the spray.
    ///
    /// # Arguments
    ///
    /// * `i` - particle index, `0..len()`.
    ///
    /// # Returns
    ///
    /// Density in kg/m^3 as of the last step; the rest density before the first.
    ///
    /// # Panics
    ///
    /// If `i >= len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{SphFluid, SphParams};
    ///
    /// let mut fluid = SphFluid::new(SphParams::blood(), 8).unwrap();
    /// fluid.spawn([0.0, 1.0, 0.0], [0.0; 3]);
    /// fluid.step(1.0 / 240.0, 9.81, |_, _| 0.0);
    /// // A lone drop samples only itself.
    /// assert!(fluid.density(0) < fluid.params().rest_density);
    /// ```
    pub fn density(&self, i: usize) -> f64 {
        self.density[i]
    }

    /// Wall time of each phase of the last [`Self::step`].
    ///
    /// # Returns
    ///
    /// The phases of the most recent step that did work; all zero before the first.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{SphFluid, SphParams};
    ///
    /// let mut fluid = SphFluid::new(SphParams::blood(), 8).unwrap();
    /// fluid.spawn([0.0, 1.0, 0.0], [0.0; 3]);
    /// fluid.step(1.0 / 240.0, 9.81, |_, _| 0.0);
    /// assert!(fluid.phase_times().total() > std::time::Duration::ZERO);
    /// ```
    pub fn phase_times(&self) -> SphPhaseTimes {
        self.times
    }

    /// Add one particle. Silently refuses once full rather than growing, so a long
    /// fight cannot turn the solver into a slideshow.
    ///
    /// **`velocity` is a request, not a promise.** The first step clamps it to
    /// [`Self::speed_ceiling`], which at typical droplet spacings is a few metres per
    /// second -- far below anything a projectile or an explosion would suggest. Ask
    /// the ceiling before spreading emitter speeds across a range, or the spread
    /// collapses to a single value and the emission design stops existing.
    ///
    /// A non-finite position or velocity is refused too: one NaN particle turns its
    /// neighbours NaN within a step, through the viscosity term.
    ///
    /// # Arguments
    ///
    /// * `position` - metres.
    /// * `velocity` - m/s, clamped by the next step.
    ///
    /// # Returns
    ///
    /// `false` if the fluid was full, or the state was not finite, and nothing was
    /// added.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{SphFluid, SphParams};
    ///
    /// let mut fluid = SphFluid::new(SphParams::blood(), 1).unwrap();
    /// assert!(fluid.spawn([0.0, 1.0, 0.0], [0.0; 3]));
    /// assert!(!fluid.spawn([0.0, 1.0, 0.0], [0.0; 3]));
    /// ```
    pub fn spawn(&mut self, position: [f64; 3], velocity: [f64; 3]) -> bool {
        if self.len() >= self.capacity {
            return false;
        }
        if !position.iter().chain(&velocity).all(|c| c.is_finite()) {
            return false;
        }
        self.px.push(position[0]);
        self.py.push(position[1]);
        self.pz.push(position[2]);
        // A particle that has never been stepped is where it is: `alpha` blending on
        // the spawn frame must give the spawn point, not a lerp from stale memory.
        self.ppx.push(position[0]);
        self.ppy.push(position[1]);
        self.ppz.push(position[2]);
        self.vx.push(velocity[0]);
        self.vy.push(velocity[1]);
        self.vz.push(velocity[2]);
        self.density.push(self.params.rest_density);
        self.still_for.push(0.0);
        self.rest_on.push(NO_SOLID);
        true
    }

    /// Advance the fluid one substep.
    ///
    /// Density and forces run in parallel on the current rayon pool; the result is
    /// bit-identical whatever that pool's size (see the module documentation).
    /// Allocates nothing once the neighbour lists have reached their high-water mark.
    ///
    /// # Arguments
    ///
    /// * `dt` - the substep, seconds. A `dt` that is not a positive finite number does
    ///   nothing, rather than step every particle to NaN.
    /// * `gravity` - downward acceleration, m/s^2. A non-finite value does nothing.
    /// * `ground_height` - terrain height in metres at a world `(x, z)`, sampled per
    ///   particle, so the fluid rests on terrain rather than a flat plane. The contact
    ///   only clamps height and reflects vertical velocity: it has no slope normal, so
    ///   a drop on an incline does not run downhill on its own (SPH-F5 in
    ///   `docs/reviews/2026-09-29-correctness-performance.md`);
    ///   [`Self::step_with_solids`] gives it one.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{SphFluid, SphParams};
    ///
    /// let mut fluid = SphFluid::new(SphParams::water(), 8).unwrap();
    /// fluid.spawn([0.0, 0.01, 0.0], [0.0, -1.0, 0.0]);
    /// for _ in 0..24 {
    ///     fluid.step(1.0 / 240.0, 9.81, |x, _| 0.1 * x);
    /// }
    /// assert!(fluid.position(0)[1] >= 0.0);
    /// ```
    pub fn step<F>(&mut self, dt: f64, gravity: f64, ground_height: F)
    where
        F: Fn(f64, f64) -> f64,
    {
        self.step_inner(dt, gravity, &ground_height, None);
    }

    /// Advance the fluid one substep against a set of solids: capsules and yawed boxes
    /// that push the liquid, and a ground that has a slope.
    ///
    /// Everything [`Self::step`] does, and two things more (the module documentation's
    /// "Solids" section has the design and the cost):
    ///
    /// - **The solids.** Each particle casts a short ray along its substep, from where it
    ///   was to where it is going, reaching [`Self::contact_radius`] further, against the
    ///   solids binned in its cell. The nearest surface it meets stops it a contact
    ///   radius out; a particle a solid moved onto is pushed out along the nearest
    ///   normal. Either way its velocity relative to the surface keeps `restitution` of
    ///   its approaching normal part and decays its tangential part at the ground's
    ///   `friction` rate, then takes on the surface's velocity: a boot displaces a pool,
    ///   a falling corpse throws a splash. Swept, so a droplet at the speed ceiling
    ///   cannot pass through a blade.
    /// - **The slope.** A particle in ground contact samples `ground_height` twice more,
    ///   a rest spacing along +x and along +z, and meets the ground along that slope's
    ///   normal with the same response, so a drop on an incline runs downhill (SPH-F5).
    ///   A particle in flight samples nothing extra, and a level ground gives exactly
    ///   what [`Self::step`] gives.
    ///
    /// One way: the solids are read and never written, and the liquid exerts nothing on
    /// them. With an empty set, on level ground, the result is bit-identical to
    /// [`Self::step`]; on a slope it differs by exactly the slope normal. Bit-identical
    /// at any thread count, like [`Self::step`]. Allocates nothing once the bins have
    /// reached their high-water mark.
    ///
    /// # Arguments
    ///
    /// * `dt` - the substep, seconds. A `dt` that is not a positive finite number does
    ///   nothing.
    /// * `gravity` - downward acceleration, m/s^2. A non-finite value does nothing.
    /// * `ground_height` - terrain height in metres at a world `(x, z)`, as for
    ///   [`Self::step`].
    /// * `solids` - the solids this substep, at their poses for it. Read only.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{SphFluid, SphParams, SphSolids};
    ///
    /// // A drop falls onto a crate and stays on its lid.
    /// let mut fluid = SphFluid::new(SphParams::water(), 8).unwrap();
    /// fluid.spawn([0.0, 0.8, 0.0], [0.0; 3]);
    /// let mut solids = SphSolids::new();
    /// solids.push_box([0.0, 0.25, 0.0], [0.25; 3], 0.0, [0.0; 3]);
    /// for _ in 0..240 {
    ///     fluid.step_with_solids(1.0 / 240.0, 9.81, |_, _| 0.0, &solids);
    /// }
    /// let lid = 0.5 + fluid.contact_radius();
    /// assert!((fluid.position(0)[1] - lid).abs() < 1e-3);
    /// ```
    pub fn step_with_solids<F>(
        &mut self,
        dt: f64,
        gravity: f64,
        ground_height: F,
        solids: &SphSolids,
    ) where
        F: Fn(f64, f64) -> f64,
    {
        self.step_inner(dt, gravity, &ground_height, Some(solids));
    }

    /// The step, with or without solids.
    fn step_inner<F>(
        &mut self,
        dt: f64,
        gravity: f64,
        ground_height: &F,
        solids: Option<&SphSolids>,
    ) where
        F: Fn(f64, f64) -> f64,
    {
        if !(dt > 0.0 && dt.is_finite() && gravity.is_finite()) || self.is_empty() {
            return;
        }
        let t0 = Instant::now();

        // Snapshot before anything moves, so `interpolated_position` has the two ends
        // of the interval the renderer is blending across.
        self.ppx.copy_from_slice(&self.px);
        self.ppy.copy_from_slice(&self.py);
        self.ppz.copy_from_slice(&self.pz);

        self.build_grid();
        let mut binning = Duration::ZERO;
        if let Some(solids) = solids {
            let tb = Instant::now();
            self.bin_solids(solids);
            binning = tb.elapsed();
        }
        let t1 = Instant::now();
        self.compute_density_and_pressure();
        let t2 = Instant::now();
        self.apply_forces(gravity);
        let t3 = Instant::now();
        let (ray_tested, contacts) = self.integrate(dt, ground_height, solids);
        let t4 = Instant::now();

        self.times = SphPhaseTimes {
            grid: t1 - t0,
            density: t2 - t1,
            forces: t3 - t2,
            integrate: t4 - t3,
        };
        self.solid_stats = SphSolidStats {
            solids: solids.map_or(0, SphSolids::len),
            bin_entries: self.solid_bins.len(),
            ray_tested,
            contacts,
            binning,
        };
        if solids.is_some() {
            self.solid_bins.reset();
        }
    }

    // -- Neighbour search -----------------------------------------------------

    /// Bin particles into a spatial hash so neighbour lookups cost what the
    /// particle count costs, not what the map size costs, and copy the state into
    /// bucket order.
    ///
    /// # Why not a dense grid
    ///
    /// The first version allocated a uniform grid spanning the particles' bounding
    /// box. That is fine for one splash and catastrophic for two: blood at opposite
    /// ends of a 100 m map produced a grid covering the gap, which at a 4 cm cell is
    /// about 10^10 cells. Measured, the same 256 particles cost 274 us clustered and
    /// 268,000 us spread -- a thousandfold cliff -- and the scratch buffer stayed
    /// resident at that size afterwards, because `Vec::resize` grows and never
    /// shrinks.
    ///
    /// Hashing cell coordinates into a table sized by particle count removes both
    /// problems at once: memory is O(n) whatever the spread, and empty space between
    /// splashes costs nothing at all.
    ///
    /// Counting sort into flat arrays: no allocation per cell, and within a bucket
    /// particles keep index order, so the sorted order is fixed by the data.
    fn build_grid(&mut self) {
        let h = self.params.smoothing_radius;
        let n = self.len();
        let table = table_size(n);
        self.table_mask = table - 1;
        let mask = self.table_mask;

        for v in [&mut self.cell_x, &mut self.cell_y, &mut self.cell_z] {
            v.clear();
            v.resize(n, 0);
        }
        self.bucket_of.clear();
        self.bucket_of.resize(n, 0);

        // Cells and buckets: independent a particle, so parallel.
        (
            self.cell_x.par_chunks_mut(STREAM_CHUNK),
            self.cell_y.par_chunks_mut(STREAM_CHUNK),
            self.cell_z.par_chunks_mut(STREAM_CHUNK),
            self.bucket_of.par_chunks_mut(STREAM_CHUNK),
            self.px.par_chunks(STREAM_CHUNK),
            self.py.par_chunks(STREAM_CHUNK),
            self.pz.par_chunks(STREAM_CHUNK),
        )
            .into_par_iter()
            .for_each(|(cx, cy, cz, bo, px, py, pz)| {
                for i in 0..bo.len() {
                    cx[i] = cell_of(px[i], h);
                    cy[i] = cell_of(py[i], h);
                    cz[i] = cell_of(pz[i], h);
                    bo[i] = bucket(row_hash(cy[i], cz[i]), cx[i], mask) as u32;
                }
            });

        // The counting sort: serial, O(n), and what fixes the order.
        self.bucket_start.clear();
        self.bucket_start.resize(table + 1, 0);
        for &b in &self.bucket_of {
            self.bucket_start[b as usize + 1] += 1;
        }
        for b in 0..table {
            self.bucket_start[b + 1] += self.bucket_start[b];
        }
        self.bucket_cursor.clear();
        self.bucket_cursor.extend_from_slice(&self.bucket_start);
        self.order.clear();
        self.order.resize(n, 0);
        self.slot_of.clear();
        self.slot_of.resize(n, 0);
        for i in 0..n {
            let b = self.bucket_of[i] as usize;
            let slot = self.bucket_cursor[b];
            self.order[slot as usize] = i as u32;
            self.slot_of[i] = slot;
            self.bucket_cursor[b] = slot + 1;
        }

        // Gather into sorted order: one scattered read a particle here, so that the
        // neighbour walk's hundreds of reads a particle are contiguous. Parallel.
        for v in [
            &mut self.sx,
            &mut self.sy,
            &mut self.sz,
            &mut self.svx,
            &mut self.svy,
            &mut self.svz,
        ] {
            v.clear();
            v.resize(n, 0.0);
        }
        for v in [&mut self.s_cell_x, &mut self.s_cell_y, &mut self.s_cell_z] {
            v.clear();
            v.resize(n, 0);
        }
        let (px, py, pz) = (&self.px, &self.py, &self.pz);
        let (vx, vy, vz) = (&self.vx, &self.vy, &self.vz);
        let (cx, cy, cz) = (&self.cell_x, &self.cell_y, &self.cell_z);
        (
            (
                self.sx.par_chunks_mut(STREAM_CHUNK),
                self.sy.par_chunks_mut(STREAM_CHUNK),
                self.sz.par_chunks_mut(STREAM_CHUNK),
                self.svx.par_chunks_mut(STREAM_CHUNK),
                self.svy.par_chunks_mut(STREAM_CHUNK),
                self.svz.par_chunks_mut(STREAM_CHUNK),
            ),
            (
                self.s_cell_x.par_chunks_mut(STREAM_CHUNK),
                self.s_cell_y.par_chunks_mut(STREAM_CHUNK),
                self.s_cell_z.par_chunks_mut(STREAM_CHUNK),
                self.order.par_chunks(STREAM_CHUNK),
            ),
        )
            .into_par_iter()
            .for_each(|((sx, sy, sz, svx, svy, svz), (scx, scy, scz, ord))| {
                // One source array at a time: each pass's random reads then share
                // the cache with one array rather than nine.
                fn pass<T: Copy>(dst: &mut [T], src: &[T], ord: &[u32]) {
                    for (d, &i) in dst.iter_mut().zip(ord) {
                        *d = src[i as usize];
                    }
                }
                pass(sx, px, ord);
                pass(sy, py, ord);
                pass(sz, pz, ord);
                pass(svx, vx, ord);
                pass(svy, vy, ord);
                pass(svz, vz, ord);
                pass(scx, cx, ord);
                pass(scy, cy, ord);
                pass(scz, cz, ord);
            });
    }

    /// Search each particle's neighbourhood, record its neighbours, and sum its
    /// density and pressure. Parallel over chunks; see the module documentation.
    fn compute_density_and_pressure(&mut self) {
        let n = self.len();
        let h = self.params.smoothing_radius;
        let h2 = h * h;
        let mass = self.params.particle_mass;
        let poly6 = 315.0 / (64.0 * core::f64::consts::PI * h.powi(9));
        let (stiffness, rest) = (self.params.stiffness, self.params.rest_density);

        self.s_density.clear();
        self.s_density.resize(n, 0.0);
        self.s_pressure.clear();
        self.s_pressure.resize(n, 0.0);
        self.s_inv_density.clear();
        self.s_inv_density.resize(n, 0.0);
        let chunks = n.div_ceil(SPH_CHUNK);
        if self.lists.len() < chunks {
            self.lists.resize_with(chunks, NeighbourList::default);
        }

        let grid = Grid {
            x: &self.sx,
            y: &self.sy,
            z: &self.sz,
            cx: &self.s_cell_x,
            cy: &self.s_cell_y,
            cz: &self.s_cell_z,
            start: &self.bucket_start,
            mask: self.table_mask,
            h,
            h2,
        };

        self.lists[..chunks]
            .par_iter_mut()
            .zip(self.s_density.par_chunks_mut(SPH_CHUNK))
            .zip(self.s_pressure.par_chunks_mut(SPH_CHUNK))
            .zip(self.s_inv_density.par_chunks_mut(SPH_CHUNK))
            .enumerate()
            .for_each(|(c, (((list, dens), press), inv))| {
                list.end.clear();
                let first = c * SPH_CHUNK;
                let mut runs = Runs::default();
                let mut len = 0usize;
                for local in 0..dens.len() {
                    let k = first + local;
                    let begin = len;
                    len = grid.neighbours(k, &mut runs, &mut list.index, len);
                    list.end.push(len as u32);
                    // The particle itself, at r = 0, is the one term every density
                    // has, so a lone drop's density is never zero.
                    let sum = h2 * h2 * h2 + grid.density_sum(k, &list.index[begin..len]);

                    let density = (mass * poly6 * sum).max(1e-9);
                    dens[local] = density;
                    inv[local] = 1.0 / density;
                    // Ideal-gas pressure, clamped non-negative. Negative pressure
                    // would make sparse regions suck inward, which is what cohesion is
                    // for and it does it far more stably.
                    press[local] = (stiffness * (density - rest)).max(0.0);
                }
            });

        let s_density = &self.s_density;
        self.density
            .par_chunks_mut(STREAM_CHUNK)
            .zip(self.slot_of.par_chunks(STREAM_CHUNK))
            .for_each(|(d, slots)| {
                for (d, &k) in d.iter_mut().zip(slots) {
                    *d = s_density[k as usize];
                }
            });
    }

    /// Pressure, viscosity and cohesion over each particle's recorded neighbours:
    /// the acceleration, in sorted order. `integrate` applies it.
    fn apply_forces(&mut self, gravity: f64) {
        let n = self.len();
        let h = self.params.smoothing_radius;
        let mass = self.params.particle_mass;
        let spiky_grad = -45.0 / (core::f64::consts::PI * h.powi(6));
        let visc_lap = 45.0 / (core::f64::consts::PI * h.powi(6));
        let k = Kernels {
            h,
            // Pressure: the symmetric form `-(m (p_i + p_j) / 2 rho_j) grad W`.
            pressure: -mass * 0.5 * spiky_grad,
            viscosity: self.params.viscosity * mass * visc_lap,
            cohesion: self.params.cohesion * mass * 32.0 / (core::f64::consts::PI * h.powi(9)),
            cohesion_floor: h.powi(6) / 64.0,
        };

        for a in [&mut self.ax, &mut self.ay, &mut self.az] {
            a.clear();
            a.resize(n, 0.0);
        }
        let chunks = n.div_ceil(SPH_CHUNK);
        let state = Sorted {
            x: &self.sx,
            y: &self.sy,
            z: &self.sz,
            vx: &self.svx,
            vy: &self.svy,
            vz: &self.svz,
            pressure: &self.s_pressure,
            inv_density: &self.s_inv_density,
        };

        self.lists[..chunks]
            .par_iter()
            .zip(self.ax.par_chunks_mut(SPH_CHUNK))
            .zip(self.ay.par_chunks_mut(SPH_CHUNK))
            .zip(self.az.par_chunks_mut(SPH_CHUNK))
            .enumerate()
            .for_each(|(c, (((list, ax), ay), az))| {
                let first = c * SPH_CHUNK;
                let mut begin = 0usize;
                for local in 0..ax.len() {
                    let i = first + local;
                    let end = list.end[local] as usize;
                    let f = state.force(i, &list.index[begin..end], &k);
                    begin = end;
                    let inv = state.inv_density[i];
                    ax[local] = f[0] * inv;
                    ay[local] = f[1] * inv - gravity;
                    az[local] = f[2] * inv;
                }
            });
    }

    /// The speed cap, the move, the solids (when given) and the ground. Returns how many
    /// particles ran the solid contact test and how many a solid moved.
    fn integrate<F>(
        &mut self,
        dt: f64,
        ground_height: &F,
        solids: Option<&SphSolids>,
    ) -> (usize, usize)
    where
        F: Fn(f64, f64) -> f64,
    {
        // A hard speed cap. SPH goes unstable by way of one particle acquiring an
        // enormous velocity and dragging its neighbours after it; clamping turns
        // that from an explosion into a brief wobble.
        //
        // It is also the single most surprising thing about this solver from the
        // outside, which is why the arithmetic lives in one shared function that
        // callers can query through `speed_ceiling` rather than being written twice.
        let max_speed = cfl_speed_ceiling(self.params.smoothing_radius, dt);
        let max_sq = max_speed * max_speed;

        // The acceleration (read through `slot_of`, a gather), the cap and the move:
        // independent a particle, so parallel, and branch-free. Scaling by exactly 1.0
        // leaves an uncapped velocity bit for bit.
        let (ax, ay, az) = (&self.ax, &self.ay, &self.az);
        let restitution = self.params.restitution;
        let friction_rate = FRICTION_REFERENCE_HZ * (1.0 / self.params.friction - 1.0);
        let friction_keep = 1.0 / (1.0 + friction_rate * dt);

        // With solids binned where the fluid is, the move meets them: still one particle
        // at a time, reading only the frozen bins, so still parallel and independent of
        // the schedule. The velocity update is the plain pass's arithmetic, operation for
        // operation. With nothing binned the plain pass runs, so solids no particle can
        // reach cost the binning and nothing here.
        if let Some(solids) = solids.filter(|_| self.solid_bins.len() > 0) {
            let contact = solids::Contact::new(
                &self.solid_bins,
                solids,
                self.contact_radius(),
                restitution,
                friction_keep,
            );
            let n = self.px.len();
            self.still_on_solid.clear();
            self.still_on_solid.resize(n, NO_SOLID);
            let counts = (
                self.vx.par_chunks_mut(STREAM_CHUNK),
                self.vy.par_chunks_mut(STREAM_CHUNK),
                self.vz.par_chunks_mut(STREAM_CHUNK),
                self.px.par_chunks_mut(STREAM_CHUNK),
                self.py.par_chunks_mut(STREAM_CHUNK),
                self.pz.par_chunks_mut(STREAM_CHUNK),
                self.slot_of.par_chunks(STREAM_CHUNK),
                self.bucket_of.par_chunks(STREAM_CHUNK),
                self.still_on_solid.par_chunks_mut(STREAM_CHUNK),
            )
                .into_par_iter()
                .map(|(vx, vy, vz, px, py, pz, slots, buckets, still)| {
                    let (mut tested, mut moved) = (0usize, 0usize);
                    for i in 0..slots.len() {
                        let k = slots[i] as usize;
                        let (mut x, mut y, mut z) =
                            (vx[i] + ax[k] * dt, vy[i] + ay[k] * dt, vz[i] + az[k] * dt);
                        let sq = x * x + y * y + z * z;
                        let scale = if sq > max_sq {
                            max_speed / speed_of(x, y, z, sq)
                        } else {
                            1.0
                        };
                        x *= scale;
                        y *= scale;
                        z *= scale;
                        let p0 = [px[i], py[i], pz[i]];
                        let mut p = [p0[0] + x * dt, p0[1] + y * dt, p0[2] + z * dt];
                        let mut v = [x, y, z];
                        // The particle's bucket is its cell at the start of the substep,
                        // where its ray starts.
                        let ids = contact.binned(buckets[i]);
                        if !ids.is_empty() {
                            tested += 1;
                            if let Some((id, rel_sq)) = contact.resolve(ids, p0, &mut p, &mut v)
                            {
                                moved += 1;
                                if rel_sq < SETTLE_SPEED * SETTLE_SPEED {
                                    still[i] = id;
                                }
                            }
                        }
                        vx[i] = v[0];
                        vy[i] = v[1];
                        vz[i] = v[2];
                        px[i] = p[0];
                        py[i] = p[1];
                        pz[i] = p[2];
                    }
                    (tested, moved)
                })
                .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
            self.ground(dt, ground_height, true, true, restitution, friction_keep);
            return counts;
        }

        (
            self.vx.par_chunks_mut(STREAM_CHUNK),
            self.vy.par_chunks_mut(STREAM_CHUNK),
            self.vz.par_chunks_mut(STREAM_CHUNK),
            self.px.par_chunks_mut(STREAM_CHUNK),
            self.py.par_chunks_mut(STREAM_CHUNK),
            self.pz.par_chunks_mut(STREAM_CHUNK),
            self.slot_of.par_chunks(STREAM_CHUNK),
        )
            .into_par_iter()
            .for_each(|(vx, vy, vz, px, py, pz, slots)| {
                for i in 0..slots.len() {
                    let k = slots[i] as usize;
                    let (mut x, mut y, mut z) =
                        (vx[i] + ax[k] * dt, vy[i] + ay[k] * dt, vz[i] + az[k] * dt);
                    let sq = x * x + y * y + z * z;
                    let scale = if sq > max_sq { max_speed / speed_of(x, y, z, sq) } else { 1.0 };
                    x *= scale;
                    y *= scale;
                    z *= scale;
                    vx[i] = x;
                    vy[i] = y;
                    vz[i] = z;
                    px[i] += x * dt;
                    py[i] += y * dt;
                    pz[i] += z * dt;
                }
            });
        let slope = solids.is_some();
        self.ground(dt, ground_height, slope, false, restitution, friction_keep);
        (0, 0)
    }

    /// The ground: one call into the caller's closure a particle, so scalar.
    ///
    /// Ground friction is a decay *rate*, not a per-step multiplier. A resting or
    /// sliding particle is in contact on every substep, so multiplying by `friction`
    /// each time made the slide distance proportional to `dt`: four times shorter at
    /// 960 Hz than at 240 Hz, and zero in the limit. Implicit decay at a rate
    /// calibrated so that one 240 Hz substep keeps exactly `friction` slides
    /// `v / rate` whatever the substep, in rational arithmetic with no transcendental
    /// in the loop. The solids decay at the same rate.
    ///
    /// With `slope`, a particle in contact samples the ground twice more, a rest spacing
    /// along +x and along +z, and the response runs along that slope's normal. A
    /// particle not in contact samples nothing extra, and where both differences are
    /// exactly zero the level response below runs unchanged, so level ground is
    /// bit-identical either way.
    ///
    /// Then each particle's stillness: on the ground, below [`SETTLE_SPEED`]; otherwise,
    /// with `solids` (the move with solids ran), set on a solid this substep and below it
    /// relative to that solid's surface (`still_on_solid`). The ground, which runs last,
    /// wins a particle that touched both. A particle still on a different surface from
    /// the one it was counting against starts its count again.
    fn ground<F>(
        &mut self,
        dt: f64,
        ground_height: &F,
        slope: bool,
        solids: bool,
        restitution: f64,
        friction_keep: f64,
    ) where
        F: Fn(f64, f64) -> f64,
    {
        let n = self.len();
        let spacing = self.params.smoothing_radius * 0.5;
        for i in 0..n {
            let floor = ground_height(self.px[i], self.pz[i]);
            let mut on_ground = false;
            if self.py[i] < floor {
                self.py[i] = floor;
                let (gx, gz) = if slope {
                    (
                        (ground_height(self.px[i] + spacing, self.pz[i]) - floor) / spacing,
                        (ground_height(self.px[i], self.pz[i] + spacing) - floor) / spacing,
                    )
                } else {
                    (0.0, 0.0)
                };
                if gx == 0.0 && gz == 0.0 {
                    self.vy[i] = -self.vy[i] * restitution;
                    self.vx[i] *= friction_keep;
                    self.vz[i] *= friction_keep;
                } else {
                    // The same response along the normal `(-gx, 1, -gz)`, normalised:
                    // the normal part reflected with `restitution`, the tangential part
                    // decayed, exactly as the level branch does to `vy` and `(vx, vz)`.
                    let inv = 1.0 / (1.0 + gx * gx + gz * gz).sqrt();
                    let nrm = [-gx * inv, inv, -gz * inv];
                    let v = [self.vx[i], self.vy[i], self.vz[i]];
                    let vn = v[0] * nrm[0] + v[1] * nrm[1] + v[2] * nrm[2];
                    let kept = -vn * restitution;
                    self.vx[i] = (v[0] - vn * nrm[0]) * friction_keep + kept * nrm[0];
                    self.vy[i] = (v[1] - vn * nrm[1]) * friction_keep + kept * nrm[1];
                    self.vz[i] = (v[2] - vn * nrm[2]) * friction_keep + kept * nrm[2];
                }
                on_ground = true;
            }

            let speed_sq = self.vx[i] * self.vx[i] + self.vy[i] * self.vy[i] + self.vz[i] * self.vz[i];
            let still_on = if on_ground {
                (speed_sq < SETTLE_SPEED * SETTLE_SPEED).then_some(NO_SOLID)
            } else if solids && self.still_on_solid[i] != NO_SOLID {
                Some(self.still_on_solid[i])
            } else {
                None
            };
            match still_on {
                Some(surface) => {
                    if self.rest_on[i] != surface {
                        self.rest_on[i] = surface;
                        self.still_for[i] = 0.0;
                    }
                    self.still_for[i] += dt;
                }
                None => self.still_for[i] = 0.0,
            }

            debug_assert!(
                self.px[i].is_finite() && self.py[i].is_finite() && self.pz[i].is_finite(),
                "sph particle {i} went non-finite"
            );
        }
    }

    /// Remove every particle that has come to rest and report where it stopped.
    ///
    /// This is the seam between the fluid and whatever it leaves behind: the solver
    /// handles the splash while it is moving, and hands the caller a position the
    /// instant it is not. Walked backwards so `swap_remove` never skips a particle;
    /// the last particle takes a removed one's index.
    ///
    /// # Arguments
    ///
    /// * `on_settled` - called once for each particle removed, before it is removed.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{SphFluid, SphParams};
    ///
    /// let mut fluid = SphFluid::new(SphParams::blood(), 8).unwrap();
    /// fluid.spawn([0.0, 0.0, 0.0], [0.0; 3]);
    /// for _ in 0..120 {
    ///     fluid.step(1.0 / 240.0, 9.81, |_, _| 0.0);
    /// }
    /// let mut stains = 0;
    /// fluid.drain_settled(|_| stains += 1);
    /// assert_eq!((stains, fluid.len()), (1, 0));
    /// ```
    pub fn drain_settled<F: FnMut(Settled)>(&mut self, mut on_settled: F) {
        let mass = self.params.particle_mass;
        let mut i = self.len();
        while i > 0 {
            i -= 1;
            if self.still_for[i] < SETTLE_TIME {
                continue;
            }
            let on = self.rest_on[i];
            on_settled(Settled {
                position: self.position(i),
                mass,
                velocity: self.velocity(i),
                on_solid: (on != NO_SOLID).then_some(on),
            });

            // The previous positions are removed with the rest, which is the whole
            // reason that buffer lives in the solver rather than in the renderer.
            for v in [
                &mut self.px,
                &mut self.py,
                &mut self.pz,
                &mut self.ppx,
                &mut self.ppy,
                &mut self.ppz,
                &mut self.vx,
                &mut self.vy,
                &mut self.vz,
                &mut self.density,
                &mut self.still_for,
            ] {
                v.swap_remove(i);
            }
            self.rest_on.swap_remove(i);
        }
    }
}

/// The sorted positions and cells the neighbour search reads, borrowed for one pass.
struct Grid<'a> {
    x: &'a [f64],
    y: &'a [f64],
    z: &'a [f64],
    cx: &'a [i32],
    cy: &'a [i32],
    cz: &'a [i32],
    start: &'a [u32],
    mask: usize,
    h: f64,
    h2: f64,
}

/// The runs of sorted slots a cell's neighbourhood covers: its nine rows' bucket
/// intervals, merged so no bucket appears twice. Every particle of a cell shares them,
/// and the sorted order keeps a cell's particles together, so they are built once a
/// cell rather than once a particle.
#[derive(Default)]
struct Runs {
    cell: [i32; 3],
    fresh: bool,
    count: usize,
    slots: [(u32, u32); 18],
    /// For a run that is exactly one row's three cells, the row, `(dy + 1) + 3 (dz + 1)`;
    /// [`MIXED_RUN`] for a run merged from several rows or split at the table's end.
    row: [u8; 18],
    /// The first bucket of a single-row run: its cells are buckets `first..first + 3`.
    first: [u32; 18],
}

/// [`Runs::row`] of a run the walk takes whole, because it is not one row.
const MIXED_RUN: u8 = u8::MAX;

impl Grid<'_> {
    /// Build `runs` for the cell `(cx, cy, cz)`.
    ///
    /// A row's three cells are three consecutive buckets, one interval, or two where
    /// it wraps the table. Two rows can hash to overlapping intervals; merging them is
    /// what lets the walk skip any per-candidate cell check. With each bucket visited
    /// once, each particle is visited at most once, and every particle within `h` lies
    /// in one of the 27 cells and so in a visited bucket. The distance test is then
    /// the whole membership test: a particle from a colliding far cell simply fails it.
    fn build_runs(&self, cell: [i32; 3], runs: &mut Runs) {
        let table = self.mask + 1;
        let mut iv = [(0usize, 0usize, MIXED_RUN); 18];
        let mut m = 0;
        for dz in -1..=1i32 {
            for dy in -1..=1i32 {
                let tag = ((dy + 1) + 3 * (dz + 1)) as u8;
                let row = row_hash(cell[1].wrapping_add(dy), cell[2].wrapping_add(dz));
                let b0 = bucket(row, cell[0].wrapping_sub(1), self.mask);
                if b0 + 3 <= table {
                    iv[m] = (b0, b0 + 3, tag);
                    m += 1;
                } else {
                    iv[m] = (b0, table, MIXED_RUN);
                    iv[m + 1] = (0, b0 + 3 - table, MIXED_RUN);
                    m += 2;
                }
            }
        }
        // Insertion sort: at most eighteen, nearly always nine.
        for a in 1..m {
            let mut b = a;
            while b > 0 && iv[b - 1].0 > iv[b].0 {
                iv.swap(b - 1, b);
                b -= 1;
            }
        }
        let mut count = 0;
        let mut cur = iv[0];
        let mut emit = |run: (usize, usize, u8), count: &mut usize| {
            let (s, e) = (self.start[run.0], self.start[run.1]);
            if s < e {
                runs.slots[*count] = (s, e);
                runs.row[*count] = run.2;
                runs.first[*count] = run.0 as u32;
                *count += 1;
            }
        };
        for &next in &iv[1..m] {
            if next.0 <= cur.1 {
                cur.1 = cur.1.max(next.1);
                cur.2 = MIXED_RUN;
            } else {
                emit(cur, &mut count);
                cur = next;
            }
        }
        emit(cur, &mut count);
        runs.count = count;
        runs.cell = cell;
        runs.fresh = true;
    }

    /// Write the sorted slots of every particle within one smoothing radius of `k`,
    /// itself excluded, into `out` from `len`, and return the new length.
    ///
    /// `out` is a buffer that only grows: it is resized when a run could overflow it
    /// and never shrunk, so after the first steps the walk writes into memory it
    /// already owns. Branch-free per candidate: every candidate's slot is written and
    /// the cursor advances only on a hit.
    #[inline]
    fn neighbours(&self, k: usize, runs: &mut Runs, out: &mut Vec<u32>, len: usize) -> usize {
        let cell = [self.cx[k], self.cy[k], self.cz[k]];
        if !runs.fresh || runs.cell != cell {
            self.build_runs(cell, runs);
        }
        let (x, y, z) = (self.x[k], self.y[k], self.z[k]);
        let h2 = self.h2;
        // Out-of-reach cells: the squared distance from the particle to the near face of
        // the cell below, level with and above its own on each axis. A cell whose nearest
        // point is beyond `h` holds no neighbour, so a single-row run is trimmed to the
        // cells in reach and skipped when none is (about a quarter of the candidates on a
        // packed block). What is dropped is what the distance test rejects, so the list,
        // and its order, is what the whole run gives. Each gap is shortened by a few units
        // in the last place of the coordinate and the reach lengthened by 1e-12, so
        // rounding in `floor` at a cell face can keep a cell, never drop one.
        let h = self.h;
        let gaps = |p: f64, c: i32| -> [f64; 3] {
            let slack = (p.abs() + h) * (4.0 * f64::EPSILON);
            let below = (p - c as f64 * h).clamp(0.0, h);
            let (b, a) = ((below - slack).max(0.0), (h - below - slack).max(0.0));
            [b * b, 0.0, a * a]
        };
        let (gx, gy, gz) = (gaps(x, cell[0]), gaps(y, cell[1]), gaps(z, cell[2]));
        let reach = h2 * (1.0 + 1e-12);
        let mut w = len;
        for r in 0..runs.count {
            let (mut s, mut e) = runs.slots[r];
            let row = runs.row[r];
            if row != MIXED_RUN {
                let g = gy[(row % 3) as usize] + gz[(row / 3) as usize];
                if g > reach {
                    continue;
                }
                let b0 = runs.first[r] as usize;
                let lo = if g + gx[0] > reach { 1 } else { 0 };
                let hi = if g + gx[2] > reach { 2 } else { 3 };
                (s, e) = (self.start[b0 + lo], self.start[b0 + hi]);
                if s >= e {
                    continue;
                }
            }
            let (s, e) = (s as usize, e as usize);
            if out.len() < w + (e - s) {
                out.resize(w + (e - s), 0);
            }
            let (xs, ys, zs) = (&self.x[s..e], &self.y[s..e], &self.z[s..e]);
            let slots = &mut out[w..w + (e - s)];
            let mut hits = 0usize;
            // Distances four at a time into lane arrays, which vectorise; then the
            // compaction, which cannot, as four scalar stores.
            let blocks = (e - s) / 4;
            for b in 0..blocks {
                let o = b * 4;
                let mut r2 = [0.0f64; 4];
                for l in 0..4 {
                    let (dx, dy, dz) = (x - xs[o + l], y - ys[o + l], z - zs[o + l]);
                    r2[l] = dx * dx + dy * dy + dz * dz;
                }
                for l in 0..4 {
                    let j = s + o + l;
                    slots[hits] = j as u32;
                    hits += ((r2[l] <= h2) & (j != k)) as usize;
                }
            }
            for o in blocks * 4..e - s {
                let (dx, dy, dz) = (x - xs[o], y - ys[o], z - zs[o]);
                let j = s + o;
                slots[hits] = j as u32;
                hits += ((dx * dx + dy * dy + dz * dz <= h2) & (j != k)) as usize;
            }
            w += hits;
        }
        w
    }

    /// The sum of `(h^2 - r^2)^3` over `k`'s neighbours: the poly6 density without its
    /// constant. Four partial sums, a neighbour's lane its position in the list.
    #[inline]
    fn density_sum(&self, k: usize, nbrs: &[u32]) -> f64 {
        let (x, y, z) = (self.x[k], self.y[k], self.z[k]);
        let h2 = self.h2;
        let term = |j: u32| {
            let j = j as usize;
            let (dx, dy, dz) = (x - self.x[j], y - self.y[j], z - self.z[j]);
            let d = h2 - (dx * dx + dy * dy + dz * dz);
            d * d * d
        };
        let mut acc = [0.0f64; 4];
        let mut blocks = nbrs.chunks_exact(4);
        for block in &mut blocks {
            for l in 0..4 {
                acc[l] += term(block[l]);
            }
        }
        for (l, &j) in blocks.remainder().iter().enumerate() {
            acc[l] += term(j);
        }
        (acc[0] + acc[1]) + (acc[2] + acc[3])
    }
}

/// The sorted state the force pass reads, borrowed for one pass.
struct Sorted<'a> {
    x: &'a [f64],
    y: &'a [f64],
    z: &'a [f64],
    vx: &'a [f64],
    vy: &'a [f64],
    vz: &'a [f64],
    pressure: &'a [f64],
    inv_density: &'a [f64],
}

/// The force kernels' constants, folded with the particle mass once a step.
struct Kernels {
    h: f64,
    /// `-m * spiky / 2`, positive: multiplies `(p_i + p_j) / rho_j * (h - r)^2`.
    pressure: f64,
    /// `mu * m * lap`: multiplies `(h - r) / rho_j`.
    viscosity: f64,
    /// `sigma * m * 32 / (pi h^9)`: Akinci's normalisation.
    cohesion: f64,
    /// `h^6 / 64`, the inner branch's offset.
    cohesion_floor: f64,
}

impl Sorted<'_> {
    /// The force on sorted particle `i` from its neighbours `nbrs`, before dividing
    /// by its density.
    ///
    /// Four neighbours a block, each lane its own partial sum, combined in a fixed
    /// order. The block is gathered into lane arrays first and every operation after
    /// is element-wise over them, so the arithmetic (the square roots and divisions
    /// above all) compiles to packed instructions. A short final block is padded with
    /// `i` itself: at `r = 0` every term is selected to zero, so the padding adds
    /// exactly `+0.0`. Coincident particles (`r <= 1e-9`) contribute nothing the same
    /// way.
    #[inline]
    fn force(&self, i: usize, nbrs: &[u32], k: &Kernels) -> [f64; 3] {
        let mut fx = [0.0f64; 4];
        let mut fy = [0.0f64; 4];
        let mut fz = [0.0f64; 4];
        let mut blocks = nbrs.chunks_exact(4);
        for b in &mut blocks {
            let js = [b[0] as usize, b[1] as usize, b[2] as usize, b[3] as usize];
            self.block(i, js, k, &mut fx, &mut fy, &mut fz);
        }
        let rest = blocks.remainder();
        if !rest.is_empty() {
            let mut js = [i; 4];
            for (l, &j) in rest.iter().enumerate() {
                js[l] = j as usize;
            }
            self.block(i, js, k, &mut fx, &mut fy, &mut fz);
        }
        let sum = |a: [f64; 4]| (a[0] + a[1]) + (a[2] + a[3]);
        [sum(fx), sum(fy), sum(fz)]
    }

    /// Four neighbours' pressure, viscosity and cohesion on `i`, added lane by lane.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        i: usize,
        js: [usize; 4],
        k: &Kernels,
        fx: &mut [f64; 4],
        fy: &mut [f64; 4],
        fz: &mut [f64; 4],
    ) {
        let (x, y, z) = (self.x[i], self.y[i], self.z[i]);
        let (vx, vy, vz) = (self.vx[i], self.vy[i], self.vz[i]);
        let (pi, h, inv_di) = (self.pressure[i], k.h, self.inv_density[i]);

        let mut dx = [0.0f64; 4];
        let mut dy = [0.0f64; 4];
        let mut dz = [0.0f64; 4];
        let mut dvx = [0.0f64; 4];
        let mut dvy = [0.0f64; 4];
        let mut dvz = [0.0f64; 4];
        let mut pj = [0.0f64; 4];
        let mut inv_dj = [0.0f64; 4];
        for l in 0..4 {
            let j = js[l];
            dx[l] = x - self.x[j];
            dy[l] = y - self.y[j];
            dz[l] = z - self.z[j];
            dvx[l] = self.vx[j] - vx;
            dvy[l] = self.vy[j] - vy;
            dvz[l] = self.vz[j] - vz;
            pj[l] = self.pressure[j];
            inv_dj[l] = self.inv_density[j];
        }

        let mut r = [0.0f64; 4];
        for l in 0..4 {
            r[l] = (dx[l] * dx[l] + dy[l] * dy[l] + dz[l] * dz[l]).sqrt();
        }
        let mut radial = [0.0f64; 4];
        let mut visc = [0.0f64; 4];
        for l in 0..4 {
            let live = r[l] > 1e-9;
            let inv_r = 1.0 / if live { r[l] } else { 1.0 };
            let hr = h - r[l];
            let pressure = k.pressure * (pi + pj[l]) * inv_dj[l] * hr * hr;
            // Akinci's spline: zero at both ends, peaked between, so particles neither
            // collapse together nor pull from beyond the kernel.
            let a3r3 = hr * hr * hr * r[l] * r[l] * r[l];
            let spline = if 2.0 * r[l] > h { a3r3 } else { 2.0 * a3r3 - k.cohesion_floor };
            // Scaled by 2 rho_i / (rho_i + rho_j), written in the inverse densities the
            // block already holds, so that after the division by rho_i the pair's
            // accelerations are equal and opposite. Divided by rho_i alone, a surface
            // particle beside a denser interior one pulled harder than it was pulled, and
            // a free blob self-propelled. Unchanged wherever the density is uniform.
            let symmetric = 2.0 * inv_dj[l] / (inv_di + inv_dj[l]);
            let cohesion = if r[l] <= h { k.cohesion * spline * symmetric } else { 0.0 };
            radial[l] = if live { (pressure - cohesion) * inv_r } else { 0.0 };
            visc[l] = if live { k.viscosity * hr * inv_dj[l] } else { 0.0 };
        }
        for l in 0..4 {
            fx[l] += radial[l] * dx[l] + visc[l] * dvx[l];
            fy[l] += radial[l] * dy[l] + visc[l] * dvy[l];
            fz[l] += radial[l] * dz[l] + visc[l] * dvz[l];
        }
    }
}

/// The Courant condition this solver enforces: a particle may cross no more than
/// [`CFL_FRACTION`] of a smoothing radius per step.
///
/// One definition, used by both the enforcement in `integrate` and the
/// [`SphFluid::speed_ceiling`] a caller reads. Two copies of this line is how a
/// documented ceiling and an enforced ceiling come to differ.
#[inline]
fn cfl_speed_ceiling(smoothing_radius: f64, dt: f64) -> f64 {
    smoothing_radius / dt.max(1e-6) * CFL_FRACTION
}

/// The length of `(x, y, z)`, given its square `sq`, safe where the square overflows.
///
/// Squaring overflows past about 1.3e154 m/s, and `max / inf` would stop the particle
/// dead instead of capping it, so there the vector is rescaled by its largest component
/// first. Only the capped branch calls this, so the common path is one square root.
#[inline]
fn speed_of(x: f64, y: f64, z: f64, sq: f64) -> f64 {
    if sq.is_finite() {
        return sq.sqrt();
    }
    let big = x.abs().max(y.abs()).max(z.abs());
    let (ux, uy, uz) = (x / big, y / big, z / big);
    big * (ux * ux + uy * uy + uz * uz).sqrt()
}

/// How much of a smoothing radius a particle may cross in one step.
///
/// Below one because the neighbour search is built once per step: a particle that
/// moved a whole radius has left the set of neighbours whose forces were computed
/// for it, so the forces it received were for somewhere it no longer is.
const CFL_FRACTION: f64 = 0.4;

/// The substep rate `SphParams::friction` is calibrated at: one substep of ground
/// contact at this rate keeps exactly `friction` of the tangential velocity, and
/// other rates keep whatever gives the same slide distance.
const FRICTION_REFERENCE_HZ: f64 = 240.0;

/// Akinci's cohesion spline, normalised over the kernel support. The scalar form the
/// force pass inlines, kept for the tests that pin its shape.
///
/// Zero at both ends and peaked around `h/2`, which is what makes it stable: it
/// cannot pull particles that are already touching any closer, and it has no reach
/// beyond the neighbour radius.
#[cfg(test)]
fn cohesion_kernel(r: f64, h: f64) -> f64 {
    if r <= 0.0 || r > h {
        return 0.0;
    }
    let norm = 32.0 / (core::f64::consts::PI * h.powi(9));
    let a = h - r;
    if 2.0 * r > h {
        norm * a * a * a * r * r * r
    } else {
        norm * (2.0 * a * a * a * r * r * r - h.powi(6) / 64.0)
    }
}

/// Integer cell coordinate of one axis, at a given cell size.
///
/// `floor` rather than a cast: casting truncates toward zero, so positions either
/// side of an axis would share a cell and neighbours would be found asymmetrically.
///
/// Clamped one short of the `i32` range, so the neighbour walk's `cell +- 1` can never
/// overflow. Anything past it (8.6e7 m at blood's spacing) shares the edge cell, where
/// the distance test still separates what is and is not a neighbour.
#[inline]
fn cell_of(p: f64, cell_size: f64) -> i32 {
    const LO: f64 = i32::MIN as f64 + 1.0;
    const HI: f64 = i32::MAX as f64 - 1.0;
    (p / cell_size).floor().clamp(LO, HI) as i32
}

/// Hash of a row of cells (fixed `y`, `z`). Integer arithmetic only, so it produces
/// the same buckets on every machine -- a hash that varied would make the neighbour
/// walk order vary with it.
#[inline]
fn row_hash(cy: i32, cz: i32) -> i64 {
    const P2: i64 = 19_349_663;
    const P3: i64 = 83_492_791;
    (cy as i64).wrapping_mul(P2) ^ (cz as i64).wrapping_mul(P3)
}

/// The bucket of cell `cx` in a row: consecutive cells of a row are consecutive
/// buckets, which is what makes a row's three cells one run of the sorted arrays.
#[inline]
fn bucket(row: i64, cx: i32, mask: usize) -> usize {
    (row.wrapping_add(cx as i64) as usize) & mask
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(_x: f64, _z: f64) -> f64 {
        0.0
    }

    /// A block of particles at rest spacing must measure close to rest density in
    /// its interior.
    ///
    /// This is the load-bearing test of the whole module. Sampled density feeds
    /// pressure, and pressure clamps at zero, so a density that comes out too low
    /// does not error -- it quietly deletes incompressibility and leaves something
    /// that still moves and is no longer a fluid. The first draft failed this at
    /// 34 against a rest density of 1000, because mass, spacing and rest density had
    /// been set independently. See [`SphParams::with_spacing`].
    #[test]
    fn a_packed_block_measures_near_rest_density() {
        let spacing = 0.02;
        let params = SphParams::water().with_spacing(spacing);
        let rest = params.rest_density;

        let mut fluid = SphFluid::new(params, 4096).unwrap();
        const SIDE: usize = 9;
        for x in 0..SIDE {
            for y in 0..SIDE {
                for z in 0..SIDE {
                    fluid.spawn(
                        [x as f64 * spacing, y as f64 * spacing, z as f64 * spacing],
                        [0.0; 3],
                    );
                }
            }
        }

        fluid.build_grid();
        fluid.compute_density_and_pressure();

        // Centre of the block, where the kernel is fully supported.
        let centre = ((SIDE / 2) * SIDE + SIDE / 2) * SIDE + SIDE / 2;
        let d = fluid.density(centre);
        let error = (d - rest).abs() / rest;
        assert!(
            error < 0.35,
            "interior density {d:.0} against a rest density of {rest:.0} \
             ({:.0}% off) -- mass, spacing and rest density are inconsistent",
            error * 100.0
        );
    }

    /// The consistency relationship itself, asserted directly so nobody re-derives
    /// it wrongly.
    #[test]
    fn spacing_sets_mass_and_smoothing_radius_together() {
        let p = SphParams::water().with_spacing(0.05);
        assert!((p.smoothing_radius - 0.10).abs() < 1e-12);
        assert!((p.particle_mass - 1000.0 * 0.05_f64.powi(3)).abs() < 1e-12);
    }


    /// Cohesion is the whole reason this module exists rather than reusing the
    /// existing particle systems, so it gets asserted directly: a loose cloud of
    /// particles must pull *together*, not disperse.
    #[test]
    fn cohesion_pulls_a_scattered_blob_together() {
        fn spread(cohesion: f64) -> f64 {
            let mut params = SphParams::blood();
            params.cohesion = cohesion;
            let mut fluid = SphFluid::new(params, 512).unwrap();

            // A small loose cloud, in zero gravity so only the fluid forces act.
            let mut seed = 12345u32;
            let mut rand = || {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                (seed >> 8) as f64 / ((1u32 << 24) as f64) - 0.5
            };
            for _ in 0..60 {
                fluid.spawn([rand() * 0.5, 5.0 + rand() * 0.5, rand() * 0.5], [0.0; 3]);
            }

            for _ in 0..60 {
                fluid.step(1.0 / 240.0, 0.0, flat);
            }

            // Mean distance from the centroid.
            let n = fluid.len() as f64;
            let mut centre = [0.0; 3];
            for i in 0..fluid.len() {
                for a in 0..3 {
                    centre[a] += fluid.position(i)[a] / n;
                }
            }
            let mut spread = 0.0;
            for i in 0..fluid.len() {
                let p = fluid.position(i);
                let d: f64 = (0..3).map(|a| (p[a] - centre[a]).powi(2)).sum();
                spread += d.sqrt() / n;
            }
            spread
        }

        let cohesive = spread(1.4);
        let loose = spread(0.0);
        assert!(
            cohesive < loose,
            "cohesion did not draw the blob in: cohesive {cohesive:.4} vs loose {loose:.4}"
        );
    }

    #[test]
    fn the_cohesion_kernel_is_zero_at_both_ends() {
        let h = 0.3;
        assert_eq!(cohesion_kernel(0.0, h), 0.0);
        assert_eq!(cohesion_kernel(h * 1.01, h), 0.0);
        assert!(cohesion_kernel(h * 0.5, h) > 0.0, "should peak in the middle");
    }

    /// The presets are not interchangeable, and the ordering between them is what
    /// makes each one look like the substance it is named after. If gel ever stops
    /// being the thickest and stickiest of the three, napalm stops crawling and
    /// starts sheeting, and it no longer reads as gel however it is drawn.
    #[test]
    fn gel_is_thicker_and_stickier_than_the_thin_fluids() {
        let gel = SphParams::napalm();
        let blood = SphParams::blood();
        let water = SphParams::water();

        assert!(gel.viscosity > blood.viscosity * 5.0, "gel should crawl");
        assert!(blood.viscosity > water.viscosity, "blood is thicker than water");

        assert!(gel.cohesion > blood.cohesion, "gel should cling hardest");
        assert!(blood.cohesion > water.cohesion, "blood beads, water sheets");

        // Splats rather than bounces, and stays where it lands.
        assert!(gel.restitution < water.restitution);
        assert!(gel.friction > water.friction);

        // A hydrocarbon, so lighter than either.
        assert!(gel.rest_density < water.rest_density);
    }

    /// A splash has to end. Particles that stop must be handed back so the caller
    /// can bake them into whatever they leave behind, and must leave the solver.
    #[test]
    fn settled_particles_are_drained_and_reported() {

        let mut fluid = SphFluid::new(SphParams::blood(), 256).unwrap();
        for i in 0..12 {
            fluid.spawn([i as f64 * 0.05, 0.6, 0.0], [0.0, -2.0, 0.0]);
        }

        let mut settled = Vec::new();
        for _ in 0..600 {
            fluid.step(1.0 / 240.0, 9.81, flat);
            fluid.drain_settled(|s| settled.push(s));
        }

        assert!(!settled.is_empty(), "nothing ever came to rest");
        assert!(
            fluid.len() < 12,
            "settled particles were reported but not removed"
        );
        for s in &settled {
            assert!(s.position.iter().all(|c| c.is_finite()));
            assert!(s.position[1].abs() < 0.2, "settled above the ground");
        }
    }

    #[test]
    fn a_falling_splash_stays_finite() {
        let mut fluid = SphFluid::new(SphParams::blood(), 1024).unwrap();
        let mut seed = 99u32;
        let mut rand = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 8) as f64 / ((1u32 << 24) as f64) - 0.5
        };
        for _ in 0..120 {
            fluid.spawn(
                [rand() * 0.3, 2.0 + rand() * 0.3, rand() * 0.3],
                [rand() * 6.0, 3.0, rand() * 6.0],
            );
        }

        for _ in 0..1200 {
            // Sloped ground, so the splash runs downhill rather than settling on a
            // convenient plane.
            fluid.step(1.0 / 240.0, 9.81, |x, z| (x + z) * 0.08);
            fluid.drain_settled(|_| {});
            for i in 0..fluid.len() {
                assert!(fluid.position(i).iter().all(|c| c.is_finite()));
                assert!(fluid.velocity(i).iter().all(|c| c.is_finite()));
            }
        }
    }

    /// Cost must track the particle count, not how far apart the particles are.
    ///
    /// The first neighbour search was a dense grid over the bounding box, which made
    /// two splashes at opposite ends of a map allocate a grid spanning the gap --
    /// measured at 274 us clustered against 268,000 us spread over 100 m, with the
    /// scratch buffer staying resident at that size afterwards. This is the guard
    /// against anyone reintroducing that.
    #[test]
    fn spreading_particles_across_a_map_does_not_blow_up_the_cost() {
        use std::time::Instant;

        fn one_step(spread: f64) -> f64 {
            let mut fluid = SphFluid::new(SphParams::blood(), 512).unwrap();
            let mut seed = 7u32;
            let mut rand = || {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                (seed >> 8) as f64 / ((1u32 << 24) as f64) - 0.5
            };
            for _ in 0..256 {
                fluid.spawn([rand() * spread, 1.0 + rand() * 0.2, rand() * spread], [0.0; 3]);
            }
            // One warm step so allocation is not being timed.
            fluid.step(1.0 / 240.0, 9.81, |_, _| 0.0);

            let start = Instant::now();
            for _ in 0..5 {
                fluid.step(1.0 / 240.0, 9.81, |_, _| 0.0);
            }
            start.elapsed().as_secs_f64() / 5.0
        }

        let clustered = one_step(0.2);
        let scattered = one_step(100.0);

        // Scattered is legitimately *cheaper* (fewer neighbours each), so the only
        // thing being asserted is that it is not dramatically worse. A generous
        // bound: the failure this catches was three orders of magnitude.
        assert!(
            scattered < clustered * 4.0,
            "spreading particles over 100 m cost {scattered:.6} s against {clustered:.6} s \
             clustered -- the neighbour search is scaling with map size again"
        );
    }

    #[test]
    fn the_pool_refuses_to_grow_past_capacity() {

        let mut fluid = SphFluid::new(SphParams::water(), 4).unwrap();
        for _ in 0..10 {
            fluid.spawn([0.0; 3], [0.0; 3]);
        }
        assert_eq!(fluid.len(), 4);
    }

    #[test]
    fn bad_parameters_are_rejected_rather_than_producing_nan() {
        let mut p = SphParams::water();
        p.smoothing_radius = 0.0;
        assert!(SphFluid::new(p, 16).is_err());

        let mut p = SphParams::water();
        p.particle_mass = -1.0;
        assert!(SphFluid::new(p, 16).is_err());
    }

    /// The ceiling a caller reads must be the ceiling the solver enforces.
    ///
    /// Asserted against a *measured* peak speed rather than against the formula,
    /// because the formula is what would be copied wrongly. This is the guard on the
    /// thing that silently deleted a caller's emission design: blood spawned at
    /// 17 m/s and blood spawned at 3 m/s came out of `step` as the same number, and
    /// nothing said so.
    #[test]
    fn nothing_ever_moves_faster_than_the_advertised_ceiling() {
        let dt = 1.0 / 240.0;
        let mut fluid = SphFluid::new(SphParams::blood(), 64).unwrap();
        let ceiling = fluid.speed_ceiling(dt);

        // Absurdly fast, in every direction, from a spread of places.
        for i in 0..24 {
            let f = i as f64 * 0.01;
            fluid.spawn([f, 3.0, -f], [400.0, 250.0, -300.0]);
        }

        let mut peak: f64 = 0.0;
        for _ in 0..240 {
            fluid.step(dt, 9.81, flat);
            for i in 0..fluid.len() {
                let v = fluid.velocity(i);
                peak = peak.max((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
            }
        }

        assert!(
            peak <= ceiling * 1.001,
            "a particle reached {peak:.3} m/s against an advertised ceiling of \
             {ceiling:.3} m/s"
        );

        // And the ceiling is genuinely reached, or it would be advertising a bound
        // that some *other* limit is actually doing the work of.
        assert!(
            peak > ceiling * 0.99,
            "peak {peak:.3} m/s never approached the ceiling {ceiling:.3} m/s -- \
             something else is limiting the fluid and `speed_ceiling` is not it"
        );
    }

    /// The ceiling is a real constraint on what a caller may ask for, stated in the
    /// units a caller thinks in: metres of throw.
    ///
    /// Blood spaced for droplets at 240 Hz cannot be flung across a field however
    /// hard it is launched, and an emitter that believes otherwise is writing
    /// constants that do nothing. The number here is the one that matters at the call
    /// site -- a splash throws metres, not tens of metres.
    #[test]
    fn a_droplet_splash_throws_metres_not_tens_of_metres() {
        let dt = 1.0 / 240.0;
        let mut fluid = SphFluid::new(SphParams::blood(), 64).unwrap();

        // Launched at a rifle round's speed, flat out along +X from chest height.
        for i in 0..16 {
            fluid.spawn([0.0, 1.2, i as f64 * 0.01], [42.0, 4.0, 0.0]);
        }

        let mut furthest: f64 = 0.0;
        let mut note = |p: [f64; 3]| {
            let d = (p[0] * p[0] + p[2] * p[2]).sqrt();
            if d > furthest {
                furthest = d;
            }
        };
        for _ in 0..(240 * 20) {
            fluid.step(dt, 9.81, flat);
            let mut settled = Vec::new();
            fluid.drain_settled(|s| settled.push(s.position));
            for p in settled {
                note(p);
            }
            if fluid.is_empty() {
                break;
            }
        }
        for i in 0..fluid.len() {
            note(fluid.position(i));
        }

        // The ballistic range of the *requested* 42 m/s is 180 m. What the solver can
        // represent is `ceiling^2 / g` plus the drop from launch height, a couple of
        // metres. Both bounds are asserted: the upper one is the bug this catches, the
        // lower one stops a future change from making a splash that goes nowhere.
        let ceiling = fluid.speed_ceiling(dt);
        let ballistic = ceiling * ceiling / 9.81;
        assert!(
            furthest < ballistic + 2.0,
            "a splash reached {furthest:.2} m, past the {:.2} m its own speed ceiling \
             of {ceiling:.2} m/s can carry it",
            ballistic + 2.0
        );
        assert!(
            furthest > 0.5,
            "a splash reached only {furthest:.2} m -- that is a puddle, not a spray"
        );
    }

    /// The render buffer has to survive the two things that move particles around in
    /// the arrays: spawning and `swap_remove`.
    ///
    /// If `prev_pos` ever falls out of step with `pos`, `interpolated_position`
    /// blends one particle's history into another particle's present -- which draws as
    /// a droplet streaking across the map between two unrelated splashes, and is
    /// exactly the failure a client-side shadow buffer would have.
    #[test]
    fn interpolation_stays_aligned_across_spawns_and_drains() {
        let dt = 1.0 / 240.0;
        let mut fluid = SphFluid::new(SphParams::blood(), 256).unwrap();

        for round in 0..40 {
            // Two clusters, far apart, so a misalignment is a huge distance rather
            // than a subtle one.
            for i in 0..3 {
                let f = i as f64 * 0.01;
                fluid.spawn([f, 0.4, f], [0.2, 1.0, 0.0]);
                fluid.spawn([80.0 + f, 0.4, 80.0 + f], [-0.2, 1.0, 0.0]);
            }
            for _ in 0..6 {
                fluid.step(dt, 9.81, flat);
            }
            fluid.drain_settled(|_| {});

            for i in 0..fluid.len() {
                let from = fluid.previous_position(i);
                let to = fluid.position(i);
                let mid = fluid.interpolated_position(i, 0.5);

                // One substep at the ceiling is 0.4 of a smoothing radius. Anything
                // further means the two ends belong to different particles.
                let travelled = ((to[0] - from[0]).powi(2)
                    + (to[1] - from[1]).powi(2)
                    + (to[2] - from[2]).powi(2))
                .sqrt();
                assert!(
                    travelled <= fluid.params().smoothing_radius * CFL_FRACTION + 1e-9,
                    "round {round}: particle {i} moved {travelled:.4} m in one substep \
                     -- prev_pos is aligned with a different particle"
                );

                for a in 0..3 {
                    let lo = from[a].min(to[a]);
                    let hi = from[a].max(to[a]);
                    assert!(mid[a] >= lo - 1e-9 && mid[a] <= hi + 1e-9);
                }
            }
        }
    }

    /// The two ends, and nothing outside them.
    #[test]
    fn interpolation_is_a_blend_and_never_an_extrapolation() {
        let dt = 1.0 / 240.0;
        let mut fluid = SphFluid::new(SphParams::blood(), 16).unwrap();
        fluid.spawn([0.0, 5.0, 0.0], [1.0, 0.0, 0.0]);

        // On the spawn frame there is no history, so every alpha is the spawn point.
        for a in [0.0, 0.5, 1.0] {
            assert_eq!(fluid.interpolated_position(0, a), [0.0, 5.0, 0.0]);
        }

        fluid.step(dt, 9.81, flat);
        let from = fluid.previous_position(0);
        let to = fluid.position(0);
        assert_ne!(from, to, "the step did not move it, so this proves nothing");

        assert_eq!(fluid.interpolated_position(0, 0.0), from);
        assert_eq!(fluid.interpolated_position(0, 1.0), to);
        // Clamped, not extrapolated: an accumulator that overran must not invent a
        // position the solver never produced.
        assert_eq!(fluid.interpolated_position(0, 4.0), to);
        assert_eq!(fluid.interpolated_position(0, -3.0), from);
        assert_eq!(fluid.interpolated_position(0, f64::NAN), to);
    }

    /// The reported symptom, measured: a fluid drawn at a frame rate that is not a
    /// multiple of the substep rate advances in uneven jumps.
    ///
    /// This is what "not smooth, running at a fixed rate instead of interpolated"
    /// actually is. At 144 fps over a 240 Hz solver a frame consumes 1.67 substeps, so
    /// some frames advance one substep's worth and some two -- the drawn position moves
    /// twice as far on some frames as on others, and the eye reads that as stutter
    /// however high the frame rate is. Sixty exactly would have hidden it: four
    /// substeps every frame, perfectly even, which is why this test does not use it.
    ///
    /// Asserted as a ratio of the largest per-frame step to the smallest, which is a
    /// property of the *motion* rather than of any number this module returns.
    #[test]
    fn drawing_at_an_awkward_frame_rate_advances_evenly() {
        let substep = 1.0 / 240.0;
        let frame = 1.0 / 144.0;

        fn evenness(substep: f64, frame: f64, interpolate: bool) -> f64 {
            let mut fluid = SphFluid::new(SphParams::blood(), 64).unwrap();
            // One drop, alone, in **zero gravity**, well clear of the ground.
            //
            // Deliberately not a parabola. The first version of this test fell under
            // gravity and measured a ratio of 1.89 even when interpolated -- because a
            // falling drop genuinely does cover more ground on a late frame than an
            // early one, so a global max-over-min was measuring acceleration and
            // calling it stutter. Under no forces the true motion is exactly linear,
            // and *every* departure from a constant step is the sampling.
            fluid.spawn([0.0, 40.0, 0.0], [2.0, 0.0, 1.0]);

            let mut accumulator = 0.0;
            let mut drawn: Vec<[f64; 3]> = Vec::new();
            for _ in 0..120 {
                accumulator += frame;
                while accumulator >= substep {
                    accumulator -= substep;
                    fluid.step(substep, 0.0, flat);
                }
                let alpha = accumulator / substep;
                drawn.push(if interpolate {
                    fluid.interpolated_position(0, alpha)
                } else {
                    fluid.position(0)
                });
            }

            let steps: Vec<f64> = drawn
                .windows(2)
                .map(|w| {
                    ((w[1][0] - w[0][0]).powi(2)
                        + (w[1][1] - w[0][1]).powi(2)
                        + (w[1][2] - w[0][2]).powi(2))
                    .sqrt()
                })
                // The first frames are still filling the accumulator.
                .skip(4)
                .collect();

            let biggest = steps.iter().cloned().fold(0.0f64, f64::max);
            let smallest = steps.iter().cloned().fold(f64::INFINITY, f64::min);
            biggest / smallest
        }

        let stepped = evenness(substep, frame, false);
        let blended = evenness(substep, frame, true);

        // Reading `position` at 144 fps over 240 Hz: some frames move two substeps'
        // worth and some one, so the ratio is close to two.
        assert!(
            stepped > 1.9,
            "reading the solved position gave a step ratio of {stepped:.2} -- this test \
             is supposed to reproduce the quantisation before asserting it is gone"
        );

        // Interpolated, every frame covers the same amount of simulated time. The
        // residual is the drop accelerating under gravity, which is real motion.
        assert!(
            blended < 1.001,
            "interpolated drawing still stepped unevenly: ratio {blended:.3} against \
             {stepped:.2} unblended"
        );
    }

    /// Interpolating must not feed back into the solver.
    ///
    /// The failure it guards against is a render-side convenience that ends up
    /// writing to solver state -- at which point the fluid's behaviour depends on the
    /// frame rate, which is the one thing substepping exists to prevent.
    #[test]
    fn reading_interpolated_positions_does_not_perturb_the_solver() {
        fn run(read_between_steps: bool) -> Vec<[f64; 3]> {
            let mut fluid = SphFluid::new(SphParams::blood(), 256).unwrap();
            for i in 0..40 {
                let f = i as f64 * 0.03;
                fluid.spawn([f, 1.5 + f * 0.5, -f], [1.0, 2.0, 0.5]);
            }
            for s in 0..300 {
                fluid.step(1.0 / 240.0, 9.81, flat);
                if read_between_steps {
                    let alpha = (s % 7) as f64 / 7.0;
                    let mut sink = 0.0;
                    for i in 0..fluid.len() {
                        sink += fluid.interpolated_position(i, alpha)[1];
                    }
                    assert!(sink.is_finite());
                }
            }
            (0..fluid.len()).map(|i| fluid.position(i)).collect()
        }
        assert_eq!(run(false), run(true));
    }

    #[test]
    fn the_same_splash_twice_gives_the_same_result() {
        fn run() -> Vec<[f64; 3]> {
            let mut fluid = SphFluid::new(SphParams::blood(), 256).unwrap();
            for i in 0..40 {
                let f = i as f64 * 0.03;
                fluid.spawn([f, 1.5 + f * 0.5, -f], [1.0, 2.0, 0.5]);
            }
            for _ in 0..300 {
                fluid.step(1.0 / 240.0, 9.81, flat);
            }
            (0..fluid.len()).map(|i| fluid.position(i)).collect()
        }
        assert_eq!(run(), run());
    }

    /// The parallel passes write each particle's sums only into its own slot, in an
    /// order fixed by the data, so the thread count must not change a single bit.
    /// A splash large enough for dozens of chunks, stepped through a fall, an impact
    /// and a settle, on pools of 1, 3 and 8 threads; and again with solids, a shin
    /// wading through the splash and a crate dropping into it, on its sloped ground.
    #[test]
    fn parallel_steps_are_bit_identical_at_any_thread_count() {
        parallel_bit_identity(false);
        parallel_bit_identity(true);
    }

    fn parallel_bit_identity(with_solids: bool) {
        let run = move || -> Vec<u64> {
            let mut solids = SphSolids::new();
            let mut contacts = 0usize;
            let mut fluid = SphFluid::new(SphParams::blood(), 4096).unwrap();
            let spacing = fluid.params().smoothing_radius * 0.5;
            for i in 0..3000usize {
                let (x, y, z) = (i % 15, (i / 15) % 15, i / 225);
                fluid.spawn(
                    [
                        x as f64 * spacing,
                        0.3 + y as f64 * spacing,
                        z as f64 * spacing,
                    ],
                    [0.4 * (i % 7) as f64 - 1.2, -1.0, 0.3 * (i % 5) as f64 - 0.6],
                );
            }
            for step in 0..90 {
                let ground = |x: f64, z: f64| 0.05 * (x - z);
                if with_solids {
                    let t = (step + 1) as f64 / 240.0;
                    let x = -0.1 + 1.0 * t;
                    solids.clear();
                    solids.push_capsule(
                        [x, -0.05, 0.12],
                        [x + 0.05, 0.45, 0.12],
                        0.04,
                        [1.0, 0.0, 0.0],
                        [1.5, 0.0, 0.0],
                    );
                    solids.push_box(
                        [0.15, 0.6 - 2.0 * t, 0.15],
                        [0.06, 0.04, 0.05],
                        0.4,
                        [0.0, -2.0, 0.0],
                    );
                    fluid.step_with_solids(1.0 / 240.0, 9.81, ground, &solids);
                    contacts += fluid.solid_stats().contacts;
                } else {
                    fluid.step(1.0 / 240.0, 9.81, ground);
                }
                if step % 30 == 29 {
                    fluid.drain_settled(|_| {});
                }
            }
            let mut bits = Vec::new();
            for i in 0..fluid.len() {
                let (p, v) = (fluid.position(i), fluid.velocity(i));
                bits.extend(p.iter().chain(v.iter()).map(|c| c.to_bits()));
                bits.push(fluid.density(i).to_bits());
            }
            assert!(
                !with_solids || contacts > 1000,
                "the solids met the splash only {contacts} times"
            );
            bits
        };

        let answers: Vec<Vec<u64>> = [1usize, 3, 8]
            .iter()
            .map(|&t| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build()
                    .unwrap();
                pool.install(run)
            })
            .collect();
        assert!(
            answers[0].len() > 1000,
            "the splash drained before it was compared"
        );
        for (t, other) in [3, 8].iter().zip(&answers[1..]) {
            assert!(
                answers[0] == *other,
                "stepping on {t} threads gave a different fluid from stepping on one"
            );
        }
    }

    /// The neighbour walk rejects candidates by their real cell, so a hash collision
    /// cannot count a particle twice or from across the map. Checked against brute
    /// force: every density equals the all-pairs sum to rounding.
    #[test]
    fn the_neighbour_walk_finds_exactly_the_all_pairs_neighbours() {
        let params = SphParams::water();
        let mut fluid = SphFluid::new(params, 600).unwrap();
        let mut seed = 11u32;
        let mut rand = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 8) as f64 / ((1u32 << 24) as f64)
        };
        // Half a dense blob, half scattered over 50 m so distant cells collide.
        for i in 0..600 {
            let spread = if i % 2 == 0 { 0.2 } else { 50.0 };
            fluid.spawn([rand() * spread, 1.0 + rand() * 0.2, rand() * spread], [0.0; 3]);
        }
        let before: Vec<[f64; 3]> = (0..fluid.len()).map(|i| fluid.position(i)).collect();
        fluid.step(1e-9, 0.0, |_, _| -10.0);

        let h = params.smoothing_radius;
        let poly6 = 315.0 / (64.0 * core::f64::consts::PI * h.powi(9));
        for (i, a) in before.iter().enumerate() {
            let mut sum = 0.0;
            for b in &before {
                let r2 = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
                if r2 <= h * h {
                    sum += (h * h - r2).powi(3);
                }
            }
            let expect = params.particle_mass * poly6 * sum;
            let got = fluid.density(i);
            assert!(
                (got - expect).abs() <= 1e-9 * expect,
                "particle {i}: walked density {got}, all-pairs {expect}"
            );
        }
    }
}

#[cfg(test)]
#[path = "sph_regression_tests.rs"]
mod regression_tests;

#[cfg(test)]
#[path = "sph_solids_tests.rs"]
mod solids_tests;
