//! A smoke plume's air: a coarse Navier-Stokes grid for its large-scale motion, a
//! swirl for its eddies, summed into one [`VelocityGrid`] that particles read, stepped
//! on a worker thread.
//!
//! A [`SwirlField`] alone has turbulence and no plume: nothing leans in the wind,
//! nothing rises as a column. [`PlumeField`] adds that motion with a small
//! [`FluidGrid3D`] over the source's region, with MacCormack advection (which keeps
//! the plume's edges) and vorticity confinement (which keeps its roll) on, and lays the
//! swirl over it for the eddies the grid is too coarse to carry. The particle loop
//! still makes one trilinear fetch a particle: the two are summed into a single grid
//! before any particle sees them.
//!
//! The grid is too slow for a frame thread (milliseconds a step) and too slow-moving
//! to need one (a plume changes over seconds), so it steps on a worker at a rate the
//! caller sets, 10 to 20 Hz by design. The worker writes each finished grid into a
//! triple buffer and publishes it with one atomic swap; the frame thread's
//! [`PlumeReader::latest`] takes the newest finished grid with another, never waits,
//! never copies, and holds no lock while particles read it.
//!
//! # Example
//!
//! ```
//! use rs_physics::fluid_dynamics::{PlumeField, PlumeRegion, PlumeSource};
//! use rs_physics::particles::{ParticleClass, ParticleEffects, TurbulenceDrive};
//!
//! // A fire 4 m wide whose smoke rises at 3 m/s, in a 16 m cube of 1 m cells, with a
//! // 2 m/s wind along x.
//! let region = PlumeRegion { origin: [-8.0, 0.0, -8.0], cells: [16, 16, 16], cell_size: 1.0 };
//! let source = PlumeSource {
//!     position: [0.0, 0.5, 0.0],
//!     drive: TurbulenceDrive::new(3.0, 4.0).unwrap(),
//!     smoke_rate: 1.0,
//! };
//! let (mut plume, mut air) = PlumeField::new(region, source, [2.0, 0.0, 0.0], 7).unwrap();
//! // One step on this thread; `spawn` runs it on a worker instead.
//! plume.step_now(0.05);
//!
//! let mut fx = ParticleEffects::with_capacity(64);
//! fx.set_class(0, ParticleClass { gravity: 0.3, drag: 2.0, restitution: 0.0, swirl: 1.0 });
//! for i in 0..16 {
//!     fx.emit_one([0.0, 1.0 + i as f32 * 0.5, 0.0], [0.0; 3], 10.0, 1.0, 0);
//! }
//! let frame = air.latest();
//! assert_eq!(frame.step(), 1);
//! for _ in 0..30 {
//!     fx.integrate_in_air(1.0 / 60.0, frame.velocity());
//! }
//! // The wind has the smoke moving downwind.
//! assert!((0..fx.len()).all(|i| fx.velocity(i)[0] > 0.0));
//! ```

#![warn(missing_docs)]

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::{AdvectionScheme, FluidGrid3D, SolverConfig, VorticityConfinement};
use crate::particles::{SwirlField, TurbulenceDrive, VelocityGrid};
use crate::utils::PhysicsError;

/// The box a [`PlumeField`] simulates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlumeRegion {
    /// The low corner of the region, metres.
    pub origin: [f32; 3],
    /// Fluid cells along `x`, `y` and `z`; at least 3 each. About 32 is the design
    /// point: 32³ cells of 1 to 2 m cover a 32 to 64 m plume.
    pub cells: [usize; 3],
    /// The cell size, metres.
    pub cell_size: f32,
}

/// A fire or smoke source: where it is, and the plume it drives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlumeSource {
    /// The centre of the source's base, metres.
    pub position: [f32; 3],
    /// `U`, the speed the smoke leaves the source at (m/s), and `L`, the source's width
    /// (m). The grid holds the air over the source rising at `U`, across a disc of
    /// diameter `L`; the swirl's eddies take their speed and rate from the same two
    /// numbers ([`TurbulenceDrive`]).
    pub drive: TurbulenceDrive,
    /// Smoke the source adds a second, in the grid's density units (a concentration
    /// the caller reads back as it likes); zero for air motion alone.
    pub smoke_rate: f32,
}

/// One finished step of a plume: the air velocity particles read, and when it was.
#[derive(Debug, Clone)]
pub struct PlumeFrame {
    velocity: VelocityGrid,
    step: u64,
    time: f64,
    step_cost: Duration,
}

impl PlumeFrame {
    /// The air velocity, m/s: the grid's flow plus the swirl, with the wind outside the
    /// region.
    ///
    /// # Returns
    ///
    /// The grid to pass to
    /// [`ParticleEffects::integrate_in_air`](crate::particles::ParticleEffects::integrate_in_air).
    ///
    /// # Examples
    ///
    /// See the [module documentation](self).
    pub fn velocity(&self) -> &VelocityGrid {
        &self.velocity
    }

    /// How many steps the plume had taken when this frame was written; 0 is the state
    /// it was built in.
    ///
    /// # Returns
    ///
    /// The step count.
    ///
    /// # Examples
    ///
    /// See the [module documentation](self).
    pub fn step(&self) -> u64 {
        self.step
    }

    /// The plume's simulated time at this frame, seconds.
    ///
    /// # Returns
    ///
    /// The sum of every step's `dt`.
    ///
    /// # Examples
    ///
    /// See the [module documentation](self).
    pub fn time(&self) -> f64 {
        self.time
    }

    /// What the step that wrote this frame cost on the thread that ran it: sources,
    /// grid step, swirl update and writing the frame.
    ///
    /// # Returns
    ///
    /// The wall time of that step.
    ///
    /// # Examples
    ///
    /// See the [module documentation](self).
    pub fn step_cost(&self) -> Duration {
        self.step_cost
    }
}

/// The three frames and which one is waiting to be read.
///
/// A slot index is owned by exactly one party at a time: the writer's back frame, the
/// reader's front frame, or the middle one, which neither touches and both only swap.
/// `middle` holds the middle index with `FRESH` set when the writer has published it
/// since the reader last took one.
struct Slots {
    frames: [UnsafeCell<PlumeFrame>; 3],
    middle: AtomicUsize,
}

const FRESH: usize = 4;
const INDEX: usize = 3;

// SAFETY: a frame is only ever touched through the index its owner holds (`Writer::back`
// or `PlumeReader::front`), and the three indices are always a permutation of 0, 1, 2:
// ownership moves only by swapping `middle`, an atomic exchange. The writer's swap is a
// release and the reader's an acquire (both `AcqRel`), so everything the writer wrote
// into a frame happens before the reader reads it. Each party is unique (neither type is
// `Clone`) and acts through `&mut self`.
unsafe impl Sync for Slots {}

struct Writer {
    slots: Arc<Slots>,
    back: usize,
}

impl Writer {
    fn back_mut(&mut self) -> &mut PlumeFrame {
        // SAFETY: `back` is this writer's own slot; see `Slots`.
        unsafe { &mut *self.slots.frames[self.back].get() }
    }

    /// Hands the back frame to the reader and takes the middle one as the new back.
    fn publish(&mut self) {
        let previous = self.slots.middle.swap(self.back | FRESH, Ordering::AcqRel);
        self.back = previous & INDEX;
    }
}

/// The frame thread's end of a [`PlumeField`]: the newest finished air, without
/// waiting and without a lock.
///
/// There is one reader per plume (it is not `Clone`), and it can be moved to any
/// thread.
pub struct PlumeReader {
    slots: Arc<Slots>,
    front: usize,
}

impl PlumeReader {
    /// The newest frame the plume has finished. If the worker has published one since
    /// the last call, it is swapped in (one atomic exchange); otherwise the same frame
    /// is returned again. The frame stays valid and unchanged until the next call: the
    /// worker never writes a frame the reader holds.
    ///
    /// # Returns
    ///
    /// The frame.
    ///
    /// # Examples
    ///
    /// See the [module documentation](self).
    pub fn latest(&mut self) -> &PlumeFrame {
        if self.slots.middle.load(Ordering::Acquire) & FRESH != 0 {
            let previous = self.slots.middle.swap(self.front, Ordering::AcqRel);
            self.front = previous & INDEX;
        }
        // SAFETY: `front` is this reader's own slot; see `Slots`.
        unsafe { &*self.slots.frames[self.front].get() }
    }
}

/// A plume's air: a Navier-Stokes grid over the source's region for the plume's
/// large-scale motion, a [`SwirlField`] for its eddies, and the triple buffer the
/// sum is published through. See the [module documentation](self).
///
/// # The grid
///
/// A [`FluidGrid3D`] of the region's cells plus a ghost layer, with
/// [`AdvectionScheme::MacCormack`] and
/// [`VorticityConfinement::MatchNumericalDissipation`] on and the conjugate-gradient
/// pressure solve at its default tolerance. Air's viscosity (1.5e-5 m²/s) is far below
/// what the grid's own numerics add at metre cells, so the grid runs inviscid, and the
/// smoke is not diffused beyond what advection does.
///
/// - The source holds the air in a disc of diameter `L` at its cell layer rising at
///   `U`, and adds `smoke_rate` of smoke there each second.
/// - The wind is a far field: the walls hold the normal flow to it, so it blows in
///   one side of the box and out the other, and the region starts filled with it.
///   Outside the region particles read the wind.
/// - Solid cells (buildings) are not supported: the grid has none, and this package
///   adds no obstacles.
///
/// # Memory
///
/// [`PlumeField::bytes`] reports it. For a 32³ region (34³ cells with the ghost layer):
/// the grid's 184 bytes a cell with both options (7.2 MB), the swirl (about 1.4 MB) and
/// three published frames of 16 bytes a cell (1.9 MB), about 10.5 MB in all.
pub struct PlumeField {
    grid: FluidGrid3D,
    swirl: SwirlField,
    region: PlumeRegion,
    source: PlumeSource,
    wind: [f32; 3],
    /// Metres in one of the grid's length units (a domain width).
    metres_per_width: f64,
    /// The fluid cells the source holds.
    source_cells: Vec<[usize; 3]>,
    writer: Writer,
    step: u64,
    time: f64,
}

impl PlumeField {
    /// A plume over `region` from `source`, in a `wind`, and the reader its frames are
    /// published to. Frame 0, the state it starts in (the region filled with the wind,
    /// the swirl at its first frame), is published already.
    ///
    /// # Arguments
    ///
    /// * `region` - the box simulated, and its cells.
    /// * `source` - the fire or smoke source; its position should be inside the region
    ///   (it is clamped to the nearest fluid cell).
    /// * `wind` - the far-field wind, m/s. Usually horizontal; a vertical component
    ///   blows through the ground and the top alike.
    /// * `seed` - the swirl's random stream.
    ///
    /// # Returns
    ///
    /// The plume and its reader.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidDimension`] for an axis of fewer than 3 cells,
    /// [`PhysicsError::InvalidDistance`] for a bad cell size or a non-finite position,
    /// and [`PhysicsError::InvalidVelocity`] for a non-finite wind.
    ///
    /// # Examples
    ///
    /// See the [module documentation](self).
    pub fn new(
        region: PlumeRegion,
        source: PlumeSource,
        wind: [f32; 3],
        seed: u32,
    ) -> Result<(PlumeField, PlumeReader), PhysicsError> {
        if region.cells.iter().any(|&n| n < 3) {
            return Err(PhysicsError::InvalidDimension);
        }
        let h = region.cell_size;
        if !(h.is_finite() && h > 0.0)
            || region.origin.iter().chain(source.position.iter()).any(|c| !c.is_finite())
            || !source.smoke_rate.is_finite()
        {
            return Err(PhysicsError::InvalidDistance);
        }
        if wind.iter().any(|w| !w.is_finite()) {
            return Err(PhysicsError::InvalidVelocity);
        }
        let dims = region.cells.map(|n| n + 2);
        let config = SolverConfig::default()
            .with_advection(AdvectionScheme::MacCormack)
            .with_vorticity_confinement(VorticityConfinement::MatchNumericalDissipation);
        let grid = FluidGrid3D::with_solver(dims[0], dims[1], dims[2], 0.0, 0.0, 0.05, config)?;
        // The ghost layer sits one cell outside the region.
        let grid_origin = region.origin.map(|o| o - h);
        let swirl = SwirlField::new(grid_origin, dims, h, source.drive, seed)?;
        let frame = || -> Result<UnsafeCell<PlumeFrame>, PhysicsError> {
            Ok(UnsafeCell::new(PlumeFrame {
                velocity: VelocityGrid::new(grid_origin, h, dims)?,
                step: 0,
                time: 0.0,
                step_cost: Duration::ZERO,
            }))
        };
        let slots = Arc::new(Slots { frames: [frame()?, frame()?, frame()?], middle: AtomicUsize::new(2) });
        let reader = PlumeReader { slots: Arc::clone(&slots), front: 1 };
        let mut plume = PlumeField {
            grid,
            swirl,
            region,
            source,
            wind,
            metres_per_width: dims[0] as f64 * h as f64,
            source_cells: Vec::new(),
            writer: Writer { slots, back: 0 },
            step: 0,
            time: 0.0,
        };
        plume.place_source();
        plume.apply_wind();
        // Start the region in the wind, so the first steps do not spend themselves
        // filling the box.
        let w = plume.wind_in_grid_units();
        for i in 1..dims[0] - 1 {
            for j in 1..dims[1] - 1 {
                for k in 1..dims[2] - 1 {
                    plume.grid.add_velocity(i, j, k, w[0], w[1], w[2])?;
                }
            }
        }
        plume.publish(Duration::ZERO);
        Ok((plume, reader))
    }

    fn wind_in_grid_units(&self) -> [f64; 3] {
        self.wind.map(|w| w as f64 / self.metres_per_width)
    }

    fn apply_wind(&mut self) {
        let w = self.wind_in_grid_units();
        self.grid.set_far_field(if w == [0.0; 3] { None } else { Some(w) });
    }

    /// The fluid cells of the source's disc: centres within `L / 2` of its position
    /// across, at the cell layer of its height; the one cell under it if the disc is
    /// narrower than a cell.
    fn place_source(&mut self) {
        let h = self.region.cell_size;
        let n = self.region.cells;
        // Fluid cell `c` (1-based in the grid) has its centre at origin + (c - 0.5) h.
        let cell_of = |axis: usize, x: f32| -> usize {
            let c = ((x - self.region.origin[axis]) / h).floor() as i64 + 1;
            c.clamp(1, n[axis] as i64) as usize
        };
        let p = self.source.position;
        let (ci, cj, ck) = (cell_of(0, p[0]), cell_of(1, p[1]), cell_of(2, p[2]));
        let radius = 0.5 * self.source.drive.scale();
        self.source_cells.clear();
        for i in 1..=n[0] {
            for k in 1..=n[2] {
                let x = self.region.origin[0] + (i as f32 - 0.5) * h - p[0];
                let z = self.region.origin[2] + (k as f32 - 0.5) * h - p[2];
                if x * x + z * z <= radius * radius {
                    self.source_cells.push([i, cj, k]);
                }
            }
        }
        if self.source_cells.is_empty() {
            self.source_cells.push([ci, cj, ck]);
        }
    }

    /// The grid the plume steps, for reading its density (the smoke) or flow.
    ///
    /// # Returns
    ///
    /// The [`FluidGrid3D`], in its own units (domain widths; see
    /// [`PlumeField::metres_per_grid_unit`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{PlumeField, PlumeRegion, PlumeSource};
    /// use rs_physics::particles::TurbulenceDrive;
    /// let region = PlumeRegion { origin: [0.0; 3], cells: [8, 8, 8], cell_size: 1.0 };
    /// let source = PlumeSource {
    ///     position: [4.0, 0.5, 4.0],
    ///     drive: TurbulenceDrive::new(2.0, 3.0).unwrap(),
    ///     smoke_rate: 1.0,
    /// };
    /// let (mut plume, _air) = PlumeField::new(region, source, [0.0; 3], 1).unwrap();
    /// plume.step_now(0.1);
    /// assert!(plume.fluid().get_total_mass() > 0.0);
    /// ```
    pub fn fluid(&self) -> &FluidGrid3D {
        &self.grid
    }

    /// Metres in the grid's unit of length (one domain width: the cell count along
    /// `x`, ghost layer included, times the cell size).
    ///
    /// # Returns
    ///
    /// The factor that turns the grid's velocities into m/s.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{PlumeField, PlumeRegion, PlumeSource};
    /// use rs_physics::particles::TurbulenceDrive;
    /// let region = PlumeRegion { origin: [0.0; 3], cells: [8, 8, 8], cell_size: 2.0 };
    /// let source = PlumeSource {
    ///     position: [8.0, 1.0, 8.0],
    ///     drive: TurbulenceDrive::new(2.0, 3.0).unwrap(),
    ///     smoke_rate: 0.0,
    /// };
    /// let (plume, _air) = PlumeField::new(region, source, [0.0; 3], 1).unwrap();
    /// assert_eq!(plume.metres_per_grid_unit(), 20.0);
    /// ```
    pub fn metres_per_grid_unit(&self) -> f64 {
        self.metres_per_width
    }

    /// Sets the far-field wind, m/s, from the next step.
    ///
    /// # Arguments
    ///
    /// * `wind` - the wind; a non-finite component is taken as zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{PlumeField, PlumeRegion, PlumeSource};
    /// use rs_physics::particles::TurbulenceDrive;
    /// let region = PlumeRegion { origin: [0.0; 3], cells: [8, 8, 8], cell_size: 1.0 };
    /// let source = PlumeSource {
    ///     position: [4.0, 0.5, 4.0],
    ///     drive: TurbulenceDrive::new(2.0, 3.0).unwrap(),
    ///     smoke_rate: 0.0,
    /// };
    /// let (mut plume, _air) = PlumeField::new(region, source, [0.0; 3], 1).unwrap();
    /// plume.set_wind([0.0, 0.0, -3.0]);
    /// plume.step_now(0.1);
    /// ```
    pub fn set_wind(&mut self, wind: [f32; 3]) {
        self.wind = wind.map(|w| if w.is_finite() { w } else { 0.0 });
        self.apply_wind();
    }

    /// Moves or changes the source, from the next step. The swirl keeps the drive it
    /// was built with.
    ///
    /// # Arguments
    ///
    /// * `source` - the new source; a non-finite position or rate is ignored.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{PlumeField, PlumeRegion, PlumeSource};
    /// use rs_physics::particles::TurbulenceDrive;
    /// let region = PlumeRegion { origin: [0.0; 3], cells: [8, 8, 8], cell_size: 1.0 };
    /// let mut source = PlumeSource {
    ///     position: [4.0, 0.5, 4.0],
    ///     drive: TurbulenceDrive::new(2.0, 3.0).unwrap(),
    ///     smoke_rate: 1.0,
    /// };
    /// let (mut plume, _air) = PlumeField::new(region, source, [0.0; 3], 1).unwrap();
    /// source.position = [2.0, 0.5, 2.0];
    /// plume.set_source(source);
    /// plume.step_now(0.1);
    /// ```
    pub fn set_source(&mut self, source: PlumeSource) {
        if source.position.iter().any(|c| !c.is_finite()) || !source.smoke_rate.is_finite() {
            return;
        }
        self.source = source;
        self.place_source();
    }

    /// Everything the plume keeps, bytes: the grid with its workspace, the swirl, and
    /// the three frames.
    ///
    /// # Returns
    ///
    /// The total.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{PlumeField, PlumeRegion, PlumeSource};
    /// use rs_physics::particles::TurbulenceDrive;
    /// let region = PlumeRegion { origin: [0.0; 3], cells: [8, 8, 8], cell_size: 1.0 };
    /// let source = PlumeSource {
    ///     position: [4.0, 0.5, 4.0],
    ///     drive: TurbulenceDrive::new(2.0, 3.0).unwrap(),
    ///     smoke_rate: 1.0,
    /// };
    /// let (mut plume, _air) = PlumeField::new(region, source, [0.0; 3], 1).unwrap();
    /// plume.step_now(0.1);
    /// assert!(plume.bytes() > 1000 * 16 * 3);
    /// ```
    pub fn bytes(&self) -> usize {
        let frames = 3 * self.writer.back_frame_bytes();
        self.grid.bytes() + self.swirl.bytes() + frames + self.source_cells.capacity() * 24
    }

    /// Advances the plume by `dt` on this thread and publishes the frame: the source,
    /// one grid step, the swirl, and their sum. For tests, and for a caller with no
    /// thread to spare; [`PlumeField::spawn`] runs the same steps on a worker.
    ///
    /// # Arguments
    ///
    /// * `dt` - seconds. Zero, negative or non-finite does nothing.
    ///
    /// # Examples
    ///
    /// See the [module documentation](self).
    pub fn step_now(&mut self, dt: f32) {
        if !(dt.is_finite() && dt > 0.0) {
            return;
        }
        let started = Instant::now();
        if (self.grid.get_dt() - dt as f64).abs() > 0.0 {
            // Validated above: finite and positive.
            let _ = self.grid.set_dt(dt as f64);
        }
        self.apply_source(dt);
        self.grid.step();
        self.swirl.advance(dt);
        self.step += 1;
        self.time += dt as f64;
        self.publish(started.elapsed());
    }

    /// Holds the source's disc rising at `U` and adds its smoke.
    fn apply_source(&mut self, dt: f32) {
        let rise = self.source.drive.velocity() as f64 / self.metres_per_width;
        let smoke = (self.source.smoke_rate * dt) as f64;
        for &[i, j, k] in &self.source_cells {
            let (_, vy, _) = self.grid.get_velocity(i, j, k).unwrap_or((0.0, rise, 0.0));
            // Source cells are fluid cells, which the grid accepts.
            let _ = self.grid.add_velocity(i, j, k, 0.0, rise - vy, 0.0);
            if smoke != 0.0 {
                let _ = self.grid.add_density(i, j, k, smoke);
            }
        }
    }

    /// Writes the grid's flow plus the swirl into the back frame, the wind into its
    /// outer layer, and publishes it.
    fn publish(&mut self, cost_so_far: Duration) {
        let started = Instant::now();
        let scale = self.metres_per_width;
        let wind = self.wind;
        let [vx, vy, vz] = self.grid.velocity_slices();
        let swirl = self.swirl.velocity().cells();
        let (step, time) = (self.step, self.time);
        let frame = self.writer.back_mut();
        let dims = frame.velocity.dims();
        let cells = frame.velocity.cells_mut();
        for i in 0..dims[0] {
            for j in 0..dims[1] {
                let row = (i * dims[1] + j) * dims[2];
                let edge_row = i == 0 || j == 0 || i == dims[0] - 1 || j == dims[1] - 1;
                for k in 0..dims[2] {
                    let c = row + k;
                    cells[c] = if edge_row || k == 0 || k == dims[2] - 1 {
                        [wind[0], wind[1], wind[2], 0.0]
                    } else {
                        let s = swirl[c];
                        [
                            (vx[c] * scale) as f32 + s[0],
                            (vy[c] * scale) as f32 + s[1],
                            (vz[c] * scale) as f32 + s[2],
                            0.0,
                        ]
                    };
                }
            }
        }
        frame.step = step;
        frame.time = time;
        frame.step_cost = cost_so_far + started.elapsed();
        self.writer.publish();
    }

    /// Runs the plume on a worker thread, stepping by `1 / rate_hz` seconds of
    /// simulated time every `1 / rate_hz` seconds of wall time, and publishing each
    /// step to the reader. If a step takes longer than its period the worker runs
    /// behind (the plume's time slows) rather than stepping twice to catch up.
    ///
    /// # Arguments
    ///
    /// * `rate_hz` - steps a second, finite and positive; 10 to 20 is the design point.
    ///
    /// # Returns
    ///
    /// The worker's handle. Dropping it stops the worker; [`PlumeWorker::stop`] also
    /// hands the plume back.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidTime`] for a bad rate, and
    /// [`PhysicsError::CalculationError`] if the thread cannot be started.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{PlumeField, PlumeRegion, PlumeSource};
    /// use rs_physics::particles::TurbulenceDrive;
    /// let region = PlumeRegion { origin: [0.0; 3], cells: [8, 8, 8], cell_size: 1.0 };
    /// let source = PlumeSource {
    ///     position: [4.0, 0.5, 4.0],
    ///     drive: TurbulenceDrive::new(2.0, 3.0).unwrap(),
    ///     smoke_rate: 1.0,
    /// };
    /// let (plume, mut air) = PlumeField::new(region, source, [1.0, 0.0, 0.0], 1).unwrap();
    /// let worker = plume.spawn(200.0).unwrap();
    /// while air.latest().step() < 3 {
    ///     std::thread::yield_now();
    /// }
    /// let plume = worker.stop();
    /// assert!(plume.fluid().get_total_mass() > 0.0);
    /// ```
    pub fn spawn(self, rate_hz: f32) -> Result<PlumeWorker, PhysicsError> {
        if !(rate_hz.is_finite() && rate_hz > 0.0) {
            return Err(PhysicsError::InvalidTime);
        }
        let stop = Arc::new(AtomicBool::new(false));
        let controls = Arc::new(Mutex::new(Controls::default()));
        let (thread_stop, thread_controls) = (Arc::clone(&stop), Arc::clone(&controls));
        let mut plume = self;
        let handle = std::thread::Builder::new()
            .name("rs_physics plume".into())
            .spawn(move || {
                let period = Duration::from_secs_f32(1.0 / rate_hz);
                let dt = 1.0 / rate_hz;
                let mut next = Instant::now();
                while !thread_stop.load(Ordering::Acquire) {
                    // Only the worker and a setter ever take this lock, never a reader.
                    if let Ok(mut c) = thread_controls.lock() {
                        if let Some(wind) = c.wind.take() {
                            plume.set_wind(wind);
                        }
                        if let Some(source) = c.source.take() {
                            plume.set_source(source);
                        }
                    }
                    plume.step_now(dt);
                    next += period;
                    let now = Instant::now();
                    if next > now {
                        std::thread::sleep(next - now);
                    } else {
                        next = now;
                    }
                }
                plume
            })
            .map_err(|e| PhysicsError::CalculationError(format!("plume worker did not start: {e}")))?;
        Ok(PlumeWorker { handle: Some(handle), stop, controls })
    }
}

impl PlumeField {
    /// The swirl, for tests that take the sum apart.
    #[cfg(test)]
    pub(crate) fn swirl_for_tests(&self) -> &SwirlField {
        &self.swirl
    }
}

impl Writer {
    fn back_frame_bytes(&self) -> usize {
        // SAFETY: `back` is this writer's own slot; see `Slots`.
        unsafe { (*self.slots.frames[self.back].get()).velocity.bytes() }
    }
}

/// Changes waiting for the worker's next step.
#[derive(Default)]
struct Controls {
    wind: Option<[f32; 3]>,
    source: Option<PlumeSource>,
}

/// A [`PlumeField`] stepping on its own thread. Dropping it stops the thread.
pub struct PlumeWorker {
    handle: Option<JoinHandle<PlumeField>>,
    stop: Arc<AtomicBool>,
    controls: Arc<Mutex<Controls>>,
}

impl PlumeWorker {
    /// Sets the wind from the worker's next step ([`PlumeField::set_wind`]).
    ///
    /// # Arguments
    ///
    /// * `wind` - m/s.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{PlumeField, PlumeRegion, PlumeSource};
    /// use rs_physics::particles::TurbulenceDrive;
    /// let region = PlumeRegion { origin: [0.0; 3], cells: [8, 8, 8], cell_size: 1.0 };
    /// let source = PlumeSource {
    ///     position: [4.0, 0.5, 4.0],
    ///     drive: TurbulenceDrive::new(2.0, 3.0).unwrap(),
    ///     smoke_rate: 1.0,
    /// };
    /// let (plume, _air) = PlumeField::new(region, source, [0.0; 3], 1).unwrap();
    /// let worker = plume.spawn(100.0).unwrap();
    /// worker.set_wind([2.0, 0.0, 0.0]);
    /// drop(worker);
    /// ```
    pub fn set_wind(&self, wind: [f32; 3]) {
        if let Ok(mut c) = self.controls.lock() {
            c.wind = Some(wind);
        }
    }

    /// Moves or changes the source from the worker's next step
    /// ([`PlumeField::set_source`]).
    ///
    /// # Arguments
    ///
    /// * `source` - the new source.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{PlumeField, PlumeRegion, PlumeSource};
    /// use rs_physics::particles::TurbulenceDrive;
    /// let region = PlumeRegion { origin: [0.0; 3], cells: [8, 8, 8], cell_size: 1.0 };
    /// let mut source = PlumeSource {
    ///     position: [4.0, 0.5, 4.0],
    ///     drive: TurbulenceDrive::new(2.0, 3.0).unwrap(),
    ///     smoke_rate: 1.0,
    /// };
    /// let (plume, _air) = PlumeField::new(region, source, [0.0; 3], 1).unwrap();
    /// let worker = plume.spawn(100.0).unwrap();
    /// source.smoke_rate = 0.0;
    /// worker.set_source(source);
    /// drop(worker);
    /// ```
    pub fn set_source(&self, source: PlumeSource) {
        if let Ok(mut c) = self.controls.lock() {
            c.source = Some(source);
        }
    }

    /// Stops the worker after its current step and hands the plume back.
    ///
    /// # Returns
    ///
    /// The plume, as of its last step.
    ///
    /// # Panics
    ///
    /// If the worker thread panicked.
    ///
    /// # Examples
    ///
    /// See [`PlumeField::spawn`].
    pub fn stop(mut self) -> PlumeField {
        self.stop.store(true, Ordering::Release);
        let handle = self.handle.take().expect("the worker is joined only once");
        match handle.join() {
            Ok(plume) => plume,
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }
}

impl Drop for PlumeWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
pub(crate) mod test_access {
    //! The triple buffer on its own, for the test that it never tears.
    use super::*;

    /// A writer and reader over three frames of `dims` cells.
    pub(crate) fn triple_buffer(dims: [usize; 3]) -> (impl FnMut(f32, u64), PlumeReader) {
        let frame = || {
            UnsafeCell::new(PlumeFrame {
                velocity: VelocityGrid::new([0.0; 3], 1.0, dims).unwrap(),
                step: 0,
                time: 0.0,
                step_cost: Duration::ZERO,
            })
        };
        let slots = Arc::new(Slots { frames: [frame(), frame(), frame()], middle: AtomicUsize::new(2) });
        let reader = PlumeReader { slots: Arc::clone(&slots), front: 1 };
        let mut writer = Writer { slots, back: 0 };
        let write = move |value: f32, step: u64| {
            let frame = writer.back_mut();
            frame.velocity.fill([value, -value, value]);
            frame.step = step;
            writer.publish();
        };
        (write, reader)
    }
}
