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
//! sorted arrays: nine ranges a particle rather than twenty-seven hash lookups. A
//! candidate from a colliding row is rejected by comparing its real cell coordinate,
//! which is also what guarantees each neighbour is visited exactly once.
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
//! global one. The answer is **bit-identical at any thread count**, by construction
//! rather than by care:
//!
//! - Every particle's sums are gathers, written only to that particle's own slot, in
//!   an order fixed by the data: the nine rows in order, each row's slots in sorted
//!   order. No thread ever adds into another particle's total.
//! - The sums run in four interleaved partial sums (a candidate's lane is its position
//!   in its row, a neighbour's its position in the list) combined as
//!   `(a0 + a1) + (a2 + a3)`, which lets the adds pipeline and vectorise without
//!   reassociation that a thread count could change.
//! - The grid build, the scatter back to particle order and the integration are
//!   serial and O(n); the ground callback is the caller's closure and is not assumed
//!   to be `Sync`.
//!
//! `parallel_steps_are_bit_identical_at_any_thread_count` asserts it at 1, 3 and 8
//! threads. No hash iteration and no transcendentals in the inner loops, so the
//! solver is deterministic on one machine; across machines it is as deterministic as
//! `f64` `sqrt` and division, which IEEE 754 fixes.
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
    pub velocity: [f64; 3],
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
}

/// Below this speed, and touching ground, a particle is considered to have landed.
const SETTLE_SPEED: f64 = 0.35;
/// ...and it must stay that way for this long before it is retired.
const SETTLE_TIME: f64 = 0.25;

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
    /// * [`PhysicsError::InvalidDistance`] if the smoothing radius is not positive.
    /// * [`PhysicsError::InvalidMass`] if the particle mass is not positive.
    /// * [`PhysicsError::CalculationError`] if the rest density is not positive, or if
    ///   `capacity` does not fit the `u32` neighbour indices.
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
        if params.smoothing_radius <= 0.0 {
            return Err(PhysicsError::InvalidDistance);
        }
        if params.particle_mass <= 0.0 {
            return Err(PhysicsError::InvalidMass);
        }
        if params.rest_density <= 0.0 {
            return Err(PhysicsError::CalculationError(
                "rest density must be positive".to_string(),
            ));
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
            params,
            capacity,
            cell_x: c(),
            cell_y: c(),
            cell_z: c(),
            bucket_of: Vec::with_capacity(capacity),
            bucket_start: Vec::with_capacity(table + 1),
            bucket_cursor: Vec::with_capacity(table + 1),
            order: Vec::with_capacity(capacity),
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
    /// # Arguments
    ///
    /// * `position` - metres.
    /// * `velocity` - m/s, clamped by the next step.
    ///
    /// # Returns
    ///
    /// `false` if the fluid was full and nothing was added.
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
    /// * `dt` - the substep, seconds. Zero or less does nothing.
    /// * `gravity` - downward acceleration, m/s^2.
    /// * `ground_height` - terrain height in metres at a world `(x, z)`, sampled per
    ///   particle, so the fluid follows terrain rather than a flat plane.
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
        if dt <= 0.0 || self.is_empty() {
            return;
        }
        let t0 = Instant::now();

        // Snapshot before anything moves, so `interpolated_position` has the two ends
        // of the interval the renderer is blending across.
        self.ppx.copy_from_slice(&self.px);
        self.ppy.copy_from_slice(&self.py);
        self.ppz.copy_from_slice(&self.pz);

        self.build_grid();
        let t1 = Instant::now();
        self.compute_density_and_pressure();
        let t2 = Instant::now();
        self.apply_forces(dt, gravity);
        let t3 = Instant::now();
        self.integrate(dt, &ground_height);
        let t4 = Instant::now();

        self.times = SphPhaseTimes {
            grid: t1 - t0,
            density: t2 - t1,
            forces: t3 - t2,
            integrate: t4 - t3,
        };
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

        self.cell_x.clear();
        self.cell_y.clear();
        self.cell_z.clear();
        self.bucket_of.clear();
        self.bucket_start.clear();
        self.bucket_start.resize(table + 1, 0);

        for i in 0..n {
            let cx = cell_of(self.px[i], h);
            let cy = cell_of(self.py[i], h);
            let cz = cell_of(self.pz[i], h);
            let b = bucket(row_hash(cy, cz), cx, mask);
            self.cell_x.push(cx);
            self.cell_y.push(cy);
            self.cell_z.push(cz);
            self.bucket_of.push(b as u32);
            self.bucket_start[b + 1] += 1;
        }
        for b in 0..table {
            self.bucket_start[b + 1] += self.bucket_start[b];
        }

        self.bucket_cursor.clear();
        self.bucket_cursor.extend_from_slice(&self.bucket_start);
        self.order.clear();
        self.order.resize(n, 0);
        for i in 0..n {
            let b = self.bucket_of[i] as usize;
            self.order[self.bucket_cursor[b] as usize] = i as u32;
            self.bucket_cursor[b] += 1;
        }

        // Gather into sorted order: one scattered read a particle here, so that the
        // neighbour walk's thousands of reads a particle are contiguous.
        macro_rules! gather {
            ($dst:ident, $src:ident) => {
                self.$dst.clear();
                self.$dst.extend(self.order.iter().map(|&i| self.$src[i as usize]));
            };
        }
        gather!(sx, px);
        gather!(sy, py);
        gather!(sz, pz);
        gather!(svx, vx);
        gather!(svy, vy);
        gather!(svz, vz);
        gather!(s_cell_x, cell_x);
        gather!(s_cell_y, cell_y);
        gather!(s_cell_z, cell_z);
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
            h2,
        };

        self.lists[..chunks]
            .par_iter_mut()
            .zip(self.s_density.par_chunks_mut(SPH_CHUNK))
            .zip(self.s_pressure.par_chunks_mut(SPH_CHUNK))
            .zip(self.s_inv_density.par_chunks_mut(SPH_CHUNK))
            .enumerate()
            .for_each(|(c, (((list, dens), press), inv))| {
                list.index.clear();
                list.end.clear();
                let first = c * SPH_CHUNK;
                for local in 0..dens.len() {
                    let k = first + local;
                    // The particle itself, at r = 0: the one term every density has,
                    // so a lone drop's density is never zero.
                    let sum = h2 * h2 * h2 + grid.neighbours(k, &mut list.index);
                    list.end.push(list.index.len() as u32);

                    let density = (mass * poly6 * sum).max(1e-9);
                    dens[local] = density;
                    inv[local] = 1.0 / density;
                    // Ideal-gas pressure, clamped non-negative. Negative pressure
                    // would make sparse regions suck inward, which is what cohesion is
                    // for and it does it far more stably.
                    press[local] = (stiffness * (density - rest)).max(0.0);
                }
            });

        for (k, &i) in self.order.iter().enumerate() {
            self.density[i as usize] = self.s_density[k];
        }
    }

    /// Pressure, viscosity and cohesion over each particle's recorded neighbours,
    /// then the velocity update.
    fn apply_forces(&mut self, dt: f64, gravity: f64) {
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

        for (k, &i) in self.order.iter().enumerate() {
            let i = i as usize;
            self.vx[i] += self.ax[k] * dt;
            self.vy[i] += self.ay[k] * dt;
            self.vz[i] += self.az[k] * dt;
        }
    }

    fn integrate<F>(&mut self, dt: f64, ground_height: &F)
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
        let n = self.len();

        // The cap and the move: branch-free over equal-length slices, so it
        // vectorises. Scaling by exactly 1.0 leaves an uncapped velocity bit for bit.
        {
            let (vx, vy, vz) = (&mut self.vx[..n], &mut self.vy[..n], &mut self.vz[..n]);
            let (px, py, pz) = (&mut self.px[..n], &mut self.py[..n], &mut self.pz[..n]);
            for i in 0..n {
                let sq = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
                let scale = if sq > max_sq { max_speed / sq.sqrt() } else { 1.0 };
                vx[i] *= scale;
                vy[i] *= scale;
                vz[i] *= scale;
                px[i] += vx[i] * dt;
                py[i] += vy[i] * dt;
                pz[i] += vz[i] * dt;
            }
        }

        // The ground: one call into the caller's closure a particle, so scalar.
        let (restitution, friction) = (self.params.restitution, self.params.friction);
        for i in 0..n {
            let floor = ground_height(self.px[i], self.pz[i]);
            let mut on_ground = false;
            if self.py[i] < floor {
                self.py[i] = floor;
                self.vy[i] = -self.vy[i] * restitution;
                self.vx[i] *= friction;
                self.vz[i] *= friction;
                on_ground = true;
            }

            let speed_sq = self.vx[i] * self.vx[i] + self.vy[i] * self.vy[i] + self.vz[i] * self.vz[i];
            if on_ground && speed_sq < SETTLE_SPEED * SETTLE_SPEED {
                self.still_for[i] += dt;
            } else {
                self.still_for[i] = 0.0;
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
            on_settled(Settled {
                position: self.position(i),
                mass,
                velocity: self.velocity(i),
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
    h2: f64,
}

impl Grid<'_> {
    /// Append the sorted slots of every particle within one smoothing radius of `k`
    /// (itself excluded) to `out`, and return the sum of `(h^2 - r^2)^3` over them.
    ///
    /// Nine rows of three cells. Each row is one or two contiguous runs of slots,
    /// since a row's cells hash to consecutive buckets. A candidate counts only if its
    /// real cell is in the row being walked and within one cell of `k`'s, which
    /// rejects hash collisions and visits every neighbour once.
    ///
    /// Branch-free per candidate: every candidate's slot is written and the cursor
    /// advances only on a hit, and the density term is selected to zero on a miss,
    /// into one of four partial sums chosen by the candidate's place in its run.
    #[inline]
    fn neighbours(&self, k: usize, out: &mut Vec<u32>) -> f64 {
        let (x, y, z) = (self.x[k], self.y[k], self.z[k]);
        let (cx, cy, cz) = (self.cx[k], self.cy[k], self.cz[k]);
        let h2 = self.h2;
        let mut acc = [0.0f64; 4];

        for dz in -1..=1i32 {
            for dy in -1..=1i32 {
                let (ty, tz) = (cy.wrapping_add(dy), cz.wrapping_add(dz));
                let b0 = bucket(row_hash(ty, tz), cx.wrapping_sub(1), self.mask);
                let b3 = b0 + 3;
                let table = self.mask + 1;
                // The run of buckets b0..b0+3, split in two where it wraps the table.
                let runs = if b3 <= table {
                    [(b0, b3), (0, 0)]
                } else {
                    [(b0, table), (0, b3 - table)]
                };
                for (from, to) in runs {
                    let (s, e) = (self.start[from] as usize, self.start[to] as usize);
                    if s >= e {
                        continue;
                    }
                    let base = out.len();
                    out.resize(base + (e - s), 0);
                    let slots = &mut out[base..];
                    let mut w = 0usize;
                    for j in s..e {
                        let (ddx, ddy, ddz) = (x - self.x[j], y - self.y[j], z - self.z[j]);
                        let r2 = ddx * ddx + ddy * ddy + ddz * ddz;
                        let in_row = (self.cy[j] == ty)
                            & (self.cz[j] == tz)
                            & (self.cx[j].wrapping_sub(cx.wrapping_sub(1)) as u32 <= 2);
                        let hit = in_row & (r2 <= h2) & (j != k);
                        slots[w] = j as u32;
                        w += hit as usize;
                        let d = h2 - r2;
                        acc[(j - s) & 3] += if hit { d * d * d } else { 0.0 };
                    }
                    out.truncate(base + w);
                }
            }
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
    /// Four lanes a block of neighbours, each lane its own partial sum, combined in a
    /// fixed order. Coincident particles (`r <= 1e-9`) contribute nothing, selected
    /// rather than branched so the block stays straight-line code.
    #[inline]
    fn force(&self, i: usize, nbrs: &[u32], k: &Kernels) -> [f64; 3] {
        let (x, y, z) = (self.x[i], self.y[i], self.z[i]);
        let (vx, vy, vz) = (self.vx[i], self.vy[i], self.vz[i]);
        let pi = self.pressure[i];
        let h = k.h;

        let term = |j: usize| -> [f64; 3] {
            let (dx, dy, dz) = (x - self.x[j], y - self.y[j], z - self.z[j]);
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            let live = r > 1e-9;
            let inv_r = if live { 1.0 / r } else { 0.0 };
            let hr = h - r;
            let inv_dj = self.inv_density[j];

            let pressure = k.pressure * (pi + self.pressure[j]) * inv_dj * hr * hr;
            let visc = if live { k.viscosity * hr * inv_dj } else { 0.0 };
            // Akinci's spline: zero at both ends, peaked between, so particles neither
            // collapse together nor pull from beyond the kernel.
            let a3r3 = hr * hr * hr * r * r * r;
            let spline = if 2.0 * r > h { a3r3 } else { 2.0 * a3r3 - k.cohesion_floor };
            let cohesion = if r <= h { k.cohesion * spline } else { 0.0 };

            let radial = (pressure - cohesion) * inv_r;
            [
                radial * dx + visc * (self.vx[j] - vx),
                radial * dy + visc * (self.vy[j] - vy),
                radial * dz + visc * (self.vz[j] - vz),
            ]
        };

        let mut fx = [0.0f64; 4];
        let mut fy = [0.0f64; 4];
        let mut fz = [0.0f64; 4];
        let mut blocks = nbrs.chunks_exact(4);
        for block in &mut blocks {
            for l in 0..4 {
                let t = term(block[l] as usize);
                fx[l] += t[0];
                fy[l] += t[1];
                fz[l] += t[2];
            }
        }
        for (l, &j) in blocks.remainder().iter().enumerate() {
            let t = term(j as usize);
            fx[l] += t[0];
            fy[l] += t[1];
            fz[l] += t[2];
        }
        let sum = |a: [f64; 4]| (a[0] + a[1]) + (a[2] + a[3]);
        [sum(fx), sum(fy), sum(fz)]
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

/// How much of a smoothing radius a particle may cross in one step.
///
/// Below one because the neighbour search is built once per step: a particle that
/// moved a whole radius has left the set of neighbours whose forces were computed
/// for it, so the forces it received were for somewhere it no longer is.
const CFL_FRACTION: f64 = 0.4;

/// Akinci's cohesion spline, normalised over the kernel support. The scalar form the
/// force pass inlines, kept for the tests that pin its shape.
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
#[inline]
fn cell_of(p: f64, cell_size: f64) -> i32 {
    (p / cell_size).floor() as i32
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
    /// does not error — it quietly deletes incompressibility and leaves something
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
             ({:.0}% off) — mass, spacing and rest density are inconsistent",
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

    /// A splash has to end. Particles that stop must be handed back so the caller
    /// can bake them into whatever they leave behind, and must leave the solver.
    #[test]
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
    /// two splashes at opposite ends of a map allocate a grid spanning the gap —
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
             clustered — the neighbour search is scaling with map size again"
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
            "peak {peak:.3} m/s never approached the ceiling {ceiling:.3} m/s — \
             something else is limiting the fluid and `speed_ceiling` is not it"
        );
    }

    /// The ceiling is a real constraint on what a caller may ask for, stated in the
    /// units a caller thinks in: metres of throw.
    ///
    /// Blood spaced for droplets at 240 Hz cannot be flung across a field however
    /// hard it is launched, and an emitter that believes otherwise is writing
    /// constants that do nothing. The number here is the one that matters at the call
    /// site — a splash throws metres, not tens of metres.
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
            "a splash reached only {furthest:.2} m — that is a puddle, not a spray"
        );
    }

    /// The render buffer has to survive the two things that move particles around in
    /// the arrays: spawning and `swap_remove`.
    ///
    /// If `prev_pos` ever falls out of step with `pos`, `interpolated_position`
    /// blends one particle's history into another particle's present — which draws as
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
                     — prev_pos is aligned with a different particle"
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
    /// some frames advance one substep's worth and some two — the drawn position moves
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
            // gravity and measured a ratio of 1.89 even when interpolated — because a
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
            "reading the solved position gave a step ratio of {stepped:.2} — this test \
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
    /// writing to solver state — at which point the fluid's behaviour depends on the
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
    /// and a settle, on pools of 1, 3 and 8 threads.
    #[test]
    fn parallel_steps_are_bit_identical_at_any_thread_count() {
        fn run() -> Vec<u64> {
            let mut fluid = SphFluid::new(SphParams::blood(), 4096).unwrap();
            let spacing = fluid.params().smoothing_radius * 0.5;
            for i in 0..3000usize {
                let (x, y, z) = (i % 15, (i / 15) % 15, i / 225);
                fluid.spawn(
                    [x as f64 * spacing, 0.3 + y as f64 * spacing, z as f64 * spacing],
                    [0.4 * (i % 7) as f64 - 1.2, -1.0, 0.3 * (i % 5) as f64 - 0.6],
                );
            }
            for step in 0..90 {
                fluid.step(1.0 / 240.0, 9.81, |x, z| 0.05 * (x - z));
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
            bits
        }

        let answers: Vec<Vec<u64>> = [1usize, 3, 8]
            .iter()
            .map(|&t| {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
                pool.install(run)
            })
            .collect();
        assert!(answers[0].len() > 1000, "the splash drained before it was compared");
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
