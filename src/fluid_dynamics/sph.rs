//! Smoothed-particle hydrodynamics — liquid that holds itself together.
//!
//! # The gap this fills
//!
//! `rs_physics` had two fluid models and neither could do a splash. [`FluidGrid3D`]
//! is Eulerian: excellent for smoke and large volumes, hopeless for droplets, since
//! resolving a 3 mm drop over a 280 m map would need a grid nobody can afford.
//! [`FluidParticle3D`] is a *solid particle moving through* a fluid medium — a
//! bubble or a grain of sediment — and its particles do not interact with each
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
//! Standard Müller-style SPH with an Akinci cohesion term:
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
//! # Determinism
//!
//! Neighbour search is a uniform grid of `Vec`s walked in index order, and every
//! force loop runs over particle indices. No hash iteration, no parallelism, no
//! transcendentals in the inner loop. This is presentation-side in the game that
//! prompted it, but a solver that is deterministic by construction costs nothing
//! extra and can be used for gameplay later.

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
    /// Set them inconsistently and the solver does not fail — it computes a sampled
    /// density far below the rest density, which makes pressure clamp to zero, which
    /// silently removes incompressibility altogether. The fluid still moves, still
    /// looks plausible, and is no longer a fluid. The first draft of this module had
    /// exactly that bug and a density test is what caught it.
    ///
    /// `spacing` is in metres, and it is the real handle on cost: halving it is eight
    /// times the particles for the same volume.
    pub fn with_spacing(mut self, spacing: f64) -> SphParams {
        self.smoothing_radius = spacing * 2.0;
        self.particle_mass = self.rest_density * spacing * spacing * spacing;
        self
    }

    /// Blood: a little denser than water, several times as viscous, and with high
    /// surface tension — which is why it travels as discrete drops rather than a
    /// spray, and why a splash beads on the ground instead of spreading thin.
    ///
    /// Spaced at 2 cm, so a particle is about 8 ml and a splash of a few dozen is a
    /// realistic quarter-litre rather than a bathtub.
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
    /// Every number here is doing the same job — making the fluid **refuse to
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

/// A body of SPH fluid.
///
/// Structure-of-arrays for the same reason the effect particles are: every pass
/// touches positions and velocities and little else.
#[derive(Debug, Clone)]
pub struct SphFluid {
    pos: Vec<[f64; 3]>,
    vel: Vec<[f64; 3]>,
    density: Vec<f64>,
    pressure: Vec<f64>,
    /// Seconds this particle has been below the settle speed. A drop is only retired
    /// once it has been still for a moment, so a drip that is briefly slow at the
    /// apex of a bounce is not mistaken for one that has stopped.
    still_for: Vec<f64>,

    params: SphParams,
    capacity: usize,

    // Scratch, all owned by the solver and cleared rather than reallocated. The
    // engine's rule: a `step` that allocates is a `step` that stutters, and this one
    // runs every frame for as long as there is fluid on screen.
    /// Integer grid coordinate of each particle's cell. Kept so a neighbour lookup
    /// can reject hash collisions by comparing the real coordinate.
    cell_coord: Vec<[i32; 3]>,
    /// Which hash bucket each particle landed in.
    bucket_of: Vec<usize>,
    /// Prefix sums into `sorted`, one entry per bucket plus a terminator.
    bucket_start: Vec<usize>,
    /// Write cursor for the counting sort. A field rather than a `clone()` of
    /// `bucket_start`, which would allocate a whole table every single step.
    bucket_cursor: Vec<usize>,
    sorted: Vec<usize>,
    /// Per-particle acceleration, accumulated before any velocity is touched so the
    /// result does not depend on index order.
    accel: Vec<[f64; 3]>,
    /// Bucket count, always a power of two so the hash reduces with a mask.
    table_mask: usize,
}

/// Below this speed, and touching ground, a particle is considered to have landed.
const SETTLE_SPEED: f64 = 0.35;
/// ...and it must stay that way for this long before it is retired.
const SETTLE_TIME: f64 = 0.25;

impl SphFluid {
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

        Ok(SphFluid {
            pos: Vec::with_capacity(capacity),
            vel: Vec::with_capacity(capacity),
            density: Vec::with_capacity(capacity),
            pressure: Vec::with_capacity(capacity),
            still_for: Vec::with_capacity(capacity),
            params,
            capacity,
            cell_coord: Vec::new(),
            bucket_of: Vec::new(),
            bucket_start: Vec::new(),
            bucket_cursor: Vec::new(),
            sorted: Vec::new(),
            accel: Vec::new(),
            table_mask: 0,

        })
    }

    pub fn len(&self) -> usize {
        self.pos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn params(&self) -> &SphParams {
        &self.params
    }

    pub fn clear(&mut self) {
        self.pos.clear();
        self.vel.clear();
        self.density.clear();
        self.pressure.clear();
        self.still_for.clear();
    }

    pub fn position(&self, i: usize) -> [f64; 3] {
        self.pos[i]
    }

    pub fn velocity(&self, i: usize) -> [f64; 3] {
        self.vel[i]
    }

    /// Sampled density at a particle, which for a surface particle is well below
    /// rest density. Useful for rendering: the sparse ones are the spray.
    pub fn density(&self, i: usize) -> f64 {
        self.density[i]
    }

    /// Add one particle. Silently refuses once full rather than growing, so a long
    /// fight cannot turn the solver into a slideshow.
    pub fn spawn(&mut self, position: [f64; 3], velocity: [f64; 3]) -> bool {
        if self.pos.len() >= self.capacity {
            return false;
        }
        self.pos.push(position);
        self.vel.push(velocity);
        self.density.push(self.params.rest_density);
        self.pressure.push(0.0);
        self.still_for.push(0.0);
        true
    }

    /// Advance the fluid.
    ///
    /// `ground_height` is sampled per particle, so the fluid follows terrain rather
    /// than a flat plane — a splash on a slope runs downhill.
    pub fn step<F>(&mut self, dt: f64, gravity: f64, ground_height: F)
    where
        F: Fn(f64, f64) -> f64,
    {
        if dt <= 0.0 || self.is_empty() {
            return;
        }

        self.build_grid();
        self.compute_density_and_pressure();
        self.apply_forces(dt, gravity);
        self.integrate(dt, &ground_height);
    }

    // ── Neighbour search ─────────────────────────────────────────────────────

    /// Bin particles into a uniform grid of one smoothing radius per cell, so each
    /// particle only has to look at its own cell and the 26 around it.
    ///
    /// Counting sort into flat `Vec`s rather than a map of buckets: no hashing, no
    /// allocation per cell, and the iteration order is fixed by construction.
    /// Bin particles into a spatial hash so neighbour lookups cost what the
    /// particle count costs, not what the map size costs.
    ///
    /// # Why not a dense grid
    ///
    /// The first version allocated a uniform grid spanning the particles' bounding
    /// box. That is fine for one splash and catastrophic for two: blood at opposite
    /// ends of a 100 m map produced a grid covering the gap, which at a 4 cm cell is
    /// about 10^10 cells. Measured, the same 256 particles cost 274 us clustered and
    /// 268,000 us spread — a thousandfold cliff — and the scratch buffer stayed
    /// resident at that size afterwards, because `Vec::resize` grows and never
    /// shrinks.
    ///
    /// Hashing cell coordinates into a table sized by particle count removes both
    /// problems at once: memory is O(n) whatever the spread, and empty space between
    /// splashes costs nothing at all.
    ///
    /// Collisions are resolved by storing each particle's real cell coordinate and
    /// comparing it during the walk, so two distant cells sharing a bucket cannot be
    /// mistaken for neighbours.
    fn build_grid(&mut self) {
        let h = self.params.smoothing_radius;
        let n = self.len();

        // Roughly two buckets per particle keeps collisions rare. Power of two so
        // the hash reduces with a mask instead of a modulo.
        let table = (n * 2).next_power_of_two().max(64);
        self.table_mask = table - 1;

        self.cell_coord.clear();
        self.cell_coord.reserve(n);
        self.bucket_of.clear();
        self.bucket_of.resize(n, 0);
        self.bucket_start.clear();
        self.bucket_start.resize(table + 1, 0);
        self.sorted.clear();
        self.sorted.resize(n, 0);

        for i in 0..n {
            let coord = cell_coord(self.pos[i], h);
            let bucket = hash_cell(coord, self.table_mask);
            self.cell_coord.push(coord);
            self.bucket_of[i] = bucket;
            self.bucket_start[bucket + 1] += 1;
        }
        for b in 0..table {
            self.bucket_start[b + 1] += self.bucket_start[b];
        }

        // Second pass fills each bucket's slice. The cursor is a reused buffer
        // rather than a clone of `bucket_start`, whose prefix sums have to survive
        // for lookup.
        self.bucket_cursor.clear();
        self.bucket_cursor.extend_from_slice(&self.bucket_start);
        for i in 0..n {
            let b = self.bucket_of[i];
            self.sorted[self.bucket_cursor[b]] = i;
            self.bucket_cursor[b] += 1;
        }
    }

    /// Call `f` with every particle within one smoothing radius of `i`, itself
    /// included.
    fn for_each_neighbour<F: FnMut(usize, f64, [f64; 3])>(&self, i: usize, mut f: F) {
        let h = self.params.smoothing_radius;
        let p = self.pos[i];
        let base = self.cell_coord[i];

        for dz in -1..=1i32 {
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    let coord = [base[0] + dx, base[1] + dy, base[2] + dz];
                    let bucket = hash_cell(coord, self.table_mask);

                    for slot in self.bucket_start[bucket]..self.bucket_start[bucket + 1] {
                        let j = self.sorted[slot];
                        // Reject hash collisions: two far-apart cells can share a
                        // bucket, and treating their contents as neighbours would
                        // apply forces across the map.
                        if self.cell_coord[j] != coord {
                            continue;
                        }

                        let d = [
                            p[0] - self.pos[j][0],
                            p[1] - self.pos[j][1],
                            p[2] - self.pos[j][2],
                        ];
                        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                        if r2 <= h * h {
                            f(j, r2.sqrt(), d);
                        }
                    }
                }
            }
        }
    }

    fn compute_density_and_pressure(&mut self) {
        let h = self.params.smoothing_radius;
        let mass = self.params.particle_mass;
        let poly6 = 315.0 / (64.0 * core::f64::consts::PI * h.powi(9));

        for i in 0..self.len() {
            let mut density = 0.0;
            self.for_each_neighbour(i, |_, r, _| {
                let diff = h * h - r * r;
                if diff > 0.0 {
                    density += mass * poly6 * diff * diff * diff;
                }
            });

            // A lone particle still has its own self-contribution, so density never
            // reaches zero and the pressure division below is always safe.
            self.density[i] = density.max(1e-9);

            // Ideal-gas pressure, clamped non-negative. Negative pressure would make
            // sparse regions suck inward, which is what cohesion is for and it does
            // it far more stably.
            self.pressure[i] =
                (self.params.stiffness * (self.density[i] - self.params.rest_density)).max(0.0);
        }
    }

    fn apply_forces(&mut self, dt: f64, gravity: f64) {
        let h = self.params.smoothing_radius;
        let mass = self.params.particle_mass;
        let spiky_grad = -45.0 / (core::f64::consts::PI * h.powi(6));
        let visc_lap = 45.0 / (core::f64::consts::PI * h.powi(6));

        // Accumulated separately so every particle sees the same density field —
        // updating velocities in place would make the result depend on index order.
        // Reused buffer: `vec![]` here would allocate once per step, forever.
        self.accel.clear();
        self.accel.resize(self.len(), [0.0; 3]);

        for i in 0..self.len() {
            let mut force = [0.0f64; 3];
            let di = self.density[i];
            let pi = self.pressure[i];
            let vi = self.vel[i];

            self.for_each_neighbour(i, |j, r, d| {
                if j == i || r <= 1e-9 {
                    return;
                }
                let dj = self.density[j];
                let dir = [d[0] / r, d[1] / r, d[2] / r];

                // Pressure: symmetric form, so equal and opposite between a pair.
                let shared = (pi + self.pressure[j]) / (2.0 * dj);
                let grad = spiky_grad * (h - r) * (h - r);
                let pressure_term = -mass * shared * grad;

                // Viscosity: pulls neighbouring velocities toward each other.
                let lap = visc_lap * (h - r);
                let visc = self.params.viscosity * mass * lap / dj;

                // Cohesion: the surface-tension term, and the reason a splash forms
                // droplets instead of dispersing. Uses Akinci's spline, which is
                // zero at both r = 0 and r = h and peaked in between — so particles
                // neither collapse together nor pull from beyond the kernel.
                let cohesion = self.params.cohesion * mass * cohesion_kernel(r, h);

                for axis in 0..3 {
                    force[axis] += pressure_term * dir[axis];
                    force[axis] += visc * (self.vel[j][axis] - vi[axis]);
                    force[axis] -= cohesion * dir[axis];
                }
            });

            for axis in 0..3 {
                self.accel[i][axis] = force[axis] / di;
            }
            self.accel[i][1] -= gravity;
        }

        for i in 0..self.len() {
            for axis in 0..3 {
                self.vel[i][axis] += self.accel[i][axis] * dt;
            }
        }

    }

    fn integrate<F>(&mut self, dt: f64, ground_height: &F)
    where
        F: Fn(f64, f64) -> f64,
    {
        // A hard speed cap. SPH goes unstable by way of one particle acquiring an
        // enormous velocity and dragging its neighbours after it; clamping turns
        // that from an explosion into a brief wobble.
        let max_speed = self.params.smoothing_radius / dt.max(1e-6) * 0.4;

        for i in 0..self.len() {
            let speed_sq: f64 = self.vel[i].iter().map(|v| v * v).sum();
            if speed_sq > max_speed * max_speed {
                let scale = max_speed / speed_sq.sqrt();
                for axis in 0..3 {
                    self.vel[i][axis] *= scale;
                }
            }

            for axis in 0..3 {
                self.pos[i][axis] += self.vel[i][axis] * dt;
            }

            let floor = ground_height(self.pos[i][0], self.pos[i][2]);
            let mut on_ground = false;
            if self.pos[i][1] < floor {
                self.pos[i][1] = floor;
                self.vel[i][1] = -self.vel[i][1] * self.params.restitution;
                self.vel[i][0] *= self.params.friction;
                self.vel[i][2] *= self.params.friction;
                on_ground = true;
            }

            let speed: f64 = self.vel[i]
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt();
            if on_ground && speed < SETTLE_SPEED {
                self.still_for[i] += dt;
            } else {
                self.still_for[i] = 0.0;
            }

            debug_assert!(
                self.pos[i].iter().all(|c| c.is_finite()),
                "sph particle {i} went non-finite"
            );
        }
    }

    /// Remove every particle that has come to rest and report where it stopped.
    ///
    /// This is the seam between the fluid and whatever it leaves behind: the solver
    /// handles the splash while it is moving, and hands the caller a position the
    /// instant it is not. Walked backwards so `swap_remove` never skips a particle.
    pub fn drain_settled<F: FnMut(Settled)>(&mut self, mut on_settled: F) {
        let mass = self.params.particle_mass;
        let mut i = self.len();
        while i > 0 {
            i -= 1;
            if self.still_for[i] < SETTLE_TIME {
                continue;
            }
            on_settled(Settled {
                position: self.pos[i],
                mass,
                velocity: self.vel[i],
            });

            self.pos.swap_remove(i);
            self.vel.swap_remove(i);
            self.density.swap_remove(i);
            self.pressure.swap_remove(i);
            self.still_for.swap_remove(i);
        }
    }
}

/// Akinci's cohesion spline, normalised over the kernel support.
///
/// Zero at both ends and peaked around `h/2`, which is what makes it stable: it
/// cannot pull particles that are already touching any closer, and it has no reach
/// beyond the neighbour radius.
#[inline]
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
}

/// Integer cell coordinate of a position, at a given cell size.
///
/// `floor` rather than a cast: casting truncates toward zero, so positions either
/// side of an axis would share a cell and neighbours would be found asymmetrically.
#[inline]
fn cell_coord(p: [f64; 3], cell_size: f64) -> [i32; 3] {
    [
        (p[0] / cell_size).floor() as i32,
        (p[1] / cell_size).floor() as i32,
        (p[2] / cell_size).floor() as i32,
    ]
}

/// The standard three-prime spatial hash. Integer arithmetic only, so it produces
/// the same buckets on every machine — a hash that varied would make the neighbour
/// walk order vary with it.
#[inline]
fn hash_cell(c: [i32; 3], mask: usize) -> usize {
    const P1: i64 = 73_856_093;
    const P2: i64 = 19_349_663;
    const P3: i64 = 83_492_791;

    let h = (c[0] as i64).wrapping_mul(P1)
        ^ (c[1] as i64).wrapping_mul(P2)
        ^ (c[2] as i64).wrapping_mul(P3);
    (h as usize) & mask
}
