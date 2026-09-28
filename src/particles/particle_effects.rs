//! Short-lived 3D particle effects -- sparks, dust, debris, smoke.
//!
//! This module fills a genuine gap. Every other particle system in this crate is a
//! *simulation* system: a fixed population of mutually-interacting bodies
//! integrated for accuracy -- Barnes-Hut N-body, SPH fluid coupling, the GPU
//! compute path. All of them are two-dimensional, and **none of them has a
//! lifetime**.
//!
//! Visual effects are the opposite workload:
//!
//! | | simulation particles | effect particles |
//! |---|---|---|
//! | population | fixed | churns constantly |
//! | interaction | every pair | none |
//! | cost driver | O(n log n) forces | memory bandwidth |
//! | precision | `f64`, accuracy matters | `f32`, nobody can see the difference |
//! | central concept | forces | **lifetime** |
//!
//! Bending an N-body solver into an emitter would cost more than this module and
//! read worse, so this is a separate thing that shares the crate rather than a
//! layer on top of the existing one.
//!
//! # Why `f32` here when the crate is `f64`
//!
//! A deliberate second exception, alongside the low-precision Barnes-Hut path. The
//! error in a spark's position is invisible at any zoom a human uses, and halving
//! the bytes per particle doubles the number that fit in cache -- which is the whole
//! cost model for a system whose per-particle work is a handful of multiply-adds.
//! It is also what any GPU backend wants. Precision would buy nothing and cost the
//! only thing that matters here.
//!
//! # Layout
//!
//! Structure-of-arrays, because every pass touches position and velocity and
//! nothing else. An array-of-structs carrying colour, size and class alongside
//! would drag all of it through cache on every integration step to read six floats.
//!
//! The split between [`ParticleEffects::integrate`] and
//! [`ParticleEffects::collide_ground`] is deliberate and is the seam a GPU backend
//! drops into: `integrate` is pure data-parallel arithmetic over flat arrays with
//! no callbacks and no branching on external state -- a direct translation to a
//! CUDA or compute-shader kernel. Ground collision needs the host's heightmap, so
//! it stays a separate, optional, host-side pass.
//!
//! # Example
//!
//! ```
//! use rs_physics::particles::{Burst, ParticleClass, ParticleEffects, EffectRng};
//!
//! let mut fx = ParticleEffects::with_capacity(4096);
//! fx.set_class(0, ParticleClass { gravity: 26.0, drag: 1.4, restitution: 0.32 });
//!
//! let mut rng = EffectRng::new(0xC0FFEE);
//! fx.emit(&Burst {
//!     origin: [0.0, 1.0, 0.0],
//!     class: 0,
//!     count: 64,
//!     speed: 6.0..15.0,
//!     lifetime: 0.2..0.5,
//!     size: 0.7..1.3,
//!     lift: 0.35,
//! }, &mut rng);
//!
//! assert_eq!(fx.len(), 64);
//! fx.integrate(1.0 / 60.0);
//! // Particles retire on their own once their lifetime runs out.
//! for _ in 0..60 { fx.integrate(1.0 / 60.0); }
//! assert_eq!(fx.len(), 0);
//! ```

#![warn(missing_docs)]

use core::ops::Range;
use std::time::Instant;

use crate::particles::particle_backend::{Backend, BackendPolicy};

/// How many distinct behaviours a single [`ParticleEffects`] pool can hold.
///
/// A small fixed count rather than a `Vec` of classes: the integration loop reads
/// the class table per particle, and a fixed-size array keeps it in registers and
/// lets the bounds check fold away. Eight covers sparks, embers, dust, smoke,
/// debris and blood with room spare; if a project needs more, it wants a second
/// pool rather than a bigger table.
pub const MAX_CLASSES: usize = 8;

/// Per-class physical behaviour.
///
/// Deliberately not per-particle. Every particle of a class shares these, so the
/// integration loop loads three floats once instead of three per particle -- and
/// "all the sparks behave like sparks" is what an artist wants anyway.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleClass {
    /// Downward acceleration, m/s². Sparks want more than real gravity (they read
    /// as light and fast); dust wants far less so it hangs.
    pub gravity: f32,
    /// Linear velocity damping per second. The dominant control over whether
    /// something reads as heavy debris or as a cloud.
    pub drag: f32,
    /// Fraction of vertical speed retained when bouncing off ground.
    pub restitution: f32,
}

impl Default for ParticleClass {
    fn default() -> Self {
        ParticleClass {
            gravity: 9.80665,
            drag: 0.0,
            restitution: 0.0,
        }
    }
}

/// A burst of particles emitted from a point.
#[derive(Debug, Clone)]
pub struct Burst {
    /// Where every particle of the burst starts, metres.
    pub origin: [f32; 3],
    /// Index into the class table. Out-of-range values are clamped rather than
    /// rejected -- a bad class is a visual bug, not a reason to fail an emit.
    pub class: u8,
    /// How many particles to emit.
    pub count: u32,
    /// Initial speed, sampled uniformly.
    pub speed: Range<f32>,
    /// Seconds before the particle retires, sampled uniformly.
    pub lifetime: Range<f32>,
    /// Arbitrary per-particle scalar the renderer can use for size, brightness or
    /// mass. The simulation does not read it.
    pub size: Range<f32>,
    /// Upward bias applied to the emission direction, roughly 0 to 1. Zero is a
    /// uniform sphere; higher values bloom the burst upward so half of it is not
    /// swallowed by the ground on the first frame.
    pub lift: f32,
}

/// A particle touching down on the ground, reported by
/// [`ParticleEffects::collide_ground_with`].
///
/// This is the hook decals hang off: a blood particle landing is where a stain
/// belongs, and the spatter pattern that results is the one the physics actually
/// produced rather than one a designer approximated with a texture.
#[derive(Debug, Clone, Copy)]
pub struct Landing {
    /// Where it touched down, on the surface.
    pub position: [f32; 3],
    /// Downward speed at the moment of contact, metres per second. Always
    /// non-negative. A decal that scales with this reads as force.
    pub impact_speed: f32,
    /// The particle's class slot.
    pub class: u8,
    /// The particle's renderer scalar, as emitted.
    pub size: f32,
}

/// A pool of live effect particles.
///
/// Capacity is fixed at construction. When full, the oldest particle is replaced --
/// which keeps the *most recent* event fully drawn, since that is the one the
/// viewer is looking at. An unbounded pool is the same resource-exhaustion hazard
/// as any other unbounded collection; it just fails as a stutter rather than a
/// crash.
#[derive(Debug, Clone)]
pub struct ParticleEffects {
    pos_x: Vec<f32>,
    pos_y: Vec<f32>,
    pos_z: Vec<f32>,
    vel_x: Vec<f32>,
    vel_y: Vec<f32>,
    vel_z: Vec<f32>,
    /// Seconds remaining. Retirement is `remaining <= 0`.
    remaining: Vec<f32>,
    /// Total lifetime, so a renderer can compute a 0..1 fade without a second pass.
    lifetime: Vec<f32>,
    size: Vec<f32>,
    class: Vec<u8>,

    classes: [ParticleClass; MAX_CLASSES],
    capacity: usize,
    /// Rotating write cursor for the full-pool case.
    oldest: usize,
    /// Decides CPU or GPU per frame from measured cost. See [`BackendPolicy`].
    policy: BackendPolicy,
}


impl ParticleEffects {
    /// An empty pool that will hold at most `capacity` live particles.
    ///
    /// Every array is allocated here, once; nothing in the per-step path allocates.
    ///
    /// # Arguments
    ///
    /// * `capacity` - the most particles alive at once. Must be at least one.
    ///
    /// # Returns
    ///
    /// An empty pool with the default class table (real gravity, no drag).
    ///
    /// # Panics
    ///
    /// If `capacity` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{ParticleClass, ParticleEffects};
    /// let fx = ParticleEffects::with_capacity(256);
    /// assert_eq!(fx.capacity(), 256);
    /// assert!(fx.is_empty());
    /// ```
    pub fn with_capacity(capacity: usize) -> ParticleEffects {
        assert!(capacity > 0, "particle pool needs room for at least one");
        ParticleEffects {
            pos_x: Vec::with_capacity(capacity),
            pos_y: Vec::with_capacity(capacity),
            pos_z: Vec::with_capacity(capacity),
            vel_x: Vec::with_capacity(capacity),
            vel_y: Vec::with_capacity(capacity),
            vel_z: Vec::with_capacity(capacity),
            remaining: Vec::with_capacity(capacity),
            lifetime: Vec::with_capacity(capacity),
            size: Vec::with_capacity(capacity),
            class: Vec::with_capacity(capacity),
            classes: [ParticleClass::default(); MAX_CLASSES],
            capacity,
            oldest: 0,
            // No GPU backend is registered until a caller supplies one, so `Auto`
            // resolves to the CPU until then. The policy still calibrates its CPU
            // cost meanwhile, so `crossover()` is meaningful before any GPU exists --
            // which is exactly when you want to know whether building one is worth it.
            policy: BackendPolicy::default(),
        }
    }

    /// The CPU/GPU dispatch policy. Use this to force a backend, declare that a GPU
    /// backend is available, or read the measured crossover for a HUD.
    ///
    /// # Returns
    ///
    /// The pool's [`BackendPolicy`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{ParticleClass, ParticleEffects};
    /// let fx = ParticleEffects::with_capacity(8);
    /// let _policy = fx.policy();
    /// ```
    pub fn policy(&self) -> &BackendPolicy {
        &self.policy
    }

    /// Mutable access to the CPU/GPU dispatch policy, to force a backend or register one.
    ///
    /// # Returns
    ///
    /// The pool's [`BackendPolicy`], mutably.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{ParticleClass, ParticleEffects};
    /// let mut fx = ParticleEffects::with_capacity(8);
    /// let _policy = fx.policy_mut();
    /// ```
    pub fn policy_mut(&mut self) -> &mut BackendPolicy {
        &mut self.policy
    }

    /// Set the behaviour every particle of class `index` shares.
    ///
    /// Takes effect on the next step for particles already alive, since the table is read
    /// per step rather than copied into each particle.
    ///
    /// # Arguments
    ///
    /// * `index` - the class slot, `0..MAX_CLASSES`; larger values are clamped to the last.
    /// * `class` - gravity in m/s^2, drag in 1/s and ground restitution as a fraction.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{ParticleClass, ParticleEffects};
    /// let mut fx = ParticleEffects::with_capacity(8);
    /// fx.set_class(2, ParticleClass { gravity: 1.6, drag: 3.4, restitution: 0.0 });
    /// assert_eq!(fx.class(2).drag, 3.4);
    /// ```
    pub fn set_class(&mut self, index: u8, class: ParticleClass) {
        let index = (index as usize).min(MAX_CLASSES - 1);
        self.classes[index] = class;
    }

    /// The behaviour of class `index`.
    ///
    /// # Arguments
    ///
    /// * `index` - the class slot; values past the table are clamped to the last slot.
    ///
    /// # Returns
    ///
    /// A copy of that class.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{ParticleClass, ParticleEffects};
    /// let fx = ParticleEffects::with_capacity(8);
    /// assert_eq!(fx.class(0), ParticleClass::default());
    /// ```
    pub fn class(&self, index: u8) -> ParticleClass {
        self.classes[(index as usize).min(MAX_CLASSES - 1)]
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
    /// use rs_physics::particles::{ParticleClass, ParticleEffects};
    /// let mut fx = ParticleEffects::with_capacity(8);
    /// fx.emit_one([0.0; 3], [0.0; 3], 1.0, 1.0, 0);
    /// assert_eq!(fx.len(), 1);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.pos_x.len()
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
    /// use rs_physics::particles::{ParticleClass, ParticleEffects};
    /// assert!(ParticleEffects::with_capacity(8).is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The most particles the pool holds at once, fixed at construction.
    ///
    /// # Returns
    ///
    /// The capacity passed to [`Self::with_capacity`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{ParticleClass, ParticleEffects};
    /// assert_eq!(ParticleEffects::with_capacity(8).capacity(), 8);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Retire every particle at once, keeping the allocation and the class table.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{ParticleClass, ParticleEffects};
    /// let mut fx = ParticleEffects::with_capacity(8);
    /// fx.emit_one([0.0; 3], [0.0; 3], 1.0, 1.0, 0);
    /// fx.clear();
    /// assert!(fx.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.pos_x.clear();
        self.pos_y.clear();
        self.pos_z.clear();
        self.vel_x.clear();
        self.vel_y.clear();
        self.vel_z.clear();
        self.remaining.clear();
        self.lifetime.clear();
        self.size.clear();
        self.class.clear();
        self.oldest = 0;
    }

    // ── Emission ─────────────────────────────────────────────────────────────

    /// Emit a burst: `burst.count` particles from one point in random directions.
    ///
    /// When the pool is full each new particle replaces the oldest, so the newest event
    /// is always drawn whole.
    ///
    /// # Arguments
    ///
    /// * `burst` - where, how many, and the ranges each particle's speed (m/s), lifetime
    ///   (s) and size are sampled from.
    /// * `rng` - the emission random stream; the same seed gives the same burst.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{Burst, EffectRng, ParticleEffects};
    /// let mut fx = ParticleEffects::with_capacity(64);
    /// let mut rng = EffectRng::new(1);
    /// fx.emit(&Burst {
    ///     origin: [0.0, 1.0, 0.0],
    ///     class: 0,
    ///     count: 16,
    ///     speed: 2.0..4.0,
    ///     lifetime: 0.5..1.0,
    ///     size: 1.0..1.0,
    ///     lift: 0.3,
    /// }, &mut rng);
    /// assert_eq!(fx.len(), 16);
    /// ```
    pub fn emit(&mut self, burst: &Burst, rng: &mut EffectRng) {
        let class = burst.class.min((MAX_CLASSES - 1) as u8);

        for _ in 0..burst.count {
            let dir = rng.hemisphere(burst.lift);
            let speed = rng.range(burst.speed.start, burst.speed.end);
            let life = rng
                .range(burst.lifetime.start, burst.lifetime.end)
                .max(f32::EPSILON);
            let size = rng.range(burst.size.start, burst.size.end);

            self.push(
                burst.origin,
                [dir[0] * speed, dir[1] * speed, dir[2] * speed],
                life,
                size,
                class,
            );
        }
    }

    /// Emit a single particle with an explicit velocity, for cases an isotropic
    /// burst does not cover -- a directed jet, a trail, a shaped charge.
    ///
    /// # Arguments
    ///
    /// * `origin` - position, metres.
    /// * `velocity` - initial velocity, m/s.
    /// * `lifetime` - seconds before it retires; values below `f32::EPSILON` are raised to it.
    /// * `size` - the renderer's per-particle scalar; the simulation does not read it.
    /// * `class` - class slot; out-of-range values are clamped.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{ParticleClass, ParticleEffects};
    /// let mut fx = ParticleEffects::with_capacity(8);
    /// fx.emit_one([0.0, 2.0, 0.0], [3.0, 0.0, 0.0], 1.0, 1.0, 0);
    /// assert_eq!(fx.velocity(0), [3.0, 0.0, 0.0]);
    /// ```
    pub fn emit_one(
        &mut self,
        origin: [f32; 3],
        velocity: [f32; 3],
        lifetime: f32,
        size: f32,
        class: u8,
    ) {
        self.push(
            origin,
            velocity,
            lifetime.max(f32::EPSILON),
            size,
            class.min((MAX_CLASSES - 1) as u8),
        );
    }

    fn push(&mut self, pos: [f32; 3], vel: [f32; 3], lifetime: f32, size: f32, class: u8) {
        if self.len() < self.capacity {
            self.pos_x.push(pos[0]);
            self.pos_y.push(pos[1]);
            self.pos_z.push(pos[2]);
            self.vel_x.push(vel[0]);
            self.vel_y.push(vel[1]);
            self.vel_z.push(vel[2]);
            self.remaining.push(lifetime);
            self.lifetime.push(lifetime);
            self.size.push(size);
            self.class.push(class);
            return;
        }

        // Full: overwrite in a rotation. O(1), and no reallocation in the hot path.
        let i = self.oldest;
        self.pos_x[i] = pos[0];
        self.pos_y[i] = pos[1];
        self.pos_z[i] = pos[2];
        self.vel_x[i] = vel[0];
        self.vel_y[i] = vel[1];
        self.vel_z[i] = vel[2];
        self.remaining[i] = lifetime;
        self.lifetime[i] = lifetime;
        self.size[i] = size;
        self.class[i] = class;
        self.oldest = (i + 1) % self.capacity;
    }

    // ── Integration ──────────────────────────────────────────────────────────

    /// Advance every particle and retire the expired ones.
    ///
    /// This is the GPU-portable half: flat arrays, no callbacks, no branching on
    /// anything the host owns. A CUDA or compute-shader backend replaces the body
    /// of the private `integrate_free_flight` and leaves compaction on the host.
    ///
    /// A velocity component too small to move its particle is set to zero, exactly: the
    /// position it would have reached is bit for bit the position it keeps. Drag alone
    /// never reaches zero and used to strand long-lived particles in the subnormal range,
    /// where a step cost about ten times as much. The derivation is on the private
    /// `integrate_free_flight`.
    ///
    /// # Arguments
    ///
    /// * `dt` - the step, seconds. A step of zero or less does nothing.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{ParticleClass, ParticleEffects};
    /// let mut fx = ParticleEffects::with_capacity(8);
    /// fx.set_class(0, ParticleClass { gravity: 0.0, drag: 2.0, restitution: 0.0 });
    /// fx.emit_one([10.0, 5.0, 0.0], [1.0, 0.0, 0.0], 1_000.0, 1.0, 0);
    /// // Drag alone never reaches zero; the flush does, once the velocity moves nothing.
    /// for _ in 0..3_000 { fx.integrate(1.0 / 60.0); }
    /// assert_eq!(fx.velocity(0)[0], 0.0);
    /// ```
    pub fn integrate(&mut self, dt: f32) {
        if dt <= 0.0 || self.is_empty() {
            return;
        }

        let count = self.len();
        let backend = self.policy.choose(count);

        let started = Instant::now();
        match backend {
            // Until a GPU backend is registered, `choose` never returns `Gpu`, so
            // this arm is the fallback for a caller who forced it and then removed
            // the device. Falling back beats failing: a dropped frame of sparks is
            // not worth an error path through a renderer.
            Backend::Cpu | Backend::Gpu => self.integrate_free_flight(dt),
        }
        self.policy.record(backend, count, started.elapsed());

        self.retire_expired();
    }


    /// Pure arithmetic over the arrays. Split out so a GPU backend has one function
    /// to replace and the benchmark has one thing to measure.
    ///
    /// # Velocities too small to move a particle are zero
    ///
    /// Drag multiplies each velocity by `d < 1` every step, so on its own it never
    /// reaches zero: it decays through the subnormal range and, at the bottom of it,
    /// `v * d` rounds back to `v` and stays there. Every multiply on a subnormal takes a
    /// microcode assist. Measured 2026-09-28 (`examples/r2_bench.rs`, medians of four
    /// rounds in one process): a pool that never retires, after 500 s of drag, cost
    /// 27.2 ns a particle at 16k and 22.3 ns at 100k, against 2.45 and 2.17 ns for the
    /// same pool fresh. A game pool with a long-lived drifting class (dust living 20 to
    /// 40 s beside short sparks) cost 8.3 ns, because its dust got there before it
    /// retired. With the flush: 3.05 ns aged, 3.19 ns live, 2.86 ns fresh; the check
    /// itself costs about 0.4 ns a particle, which the live pool repays 2.6 times over.
    ///
    /// The fix is physical rather than a tuned epsilon. A component is set to zero when
    /// the displacement it would produce this step, `v * dt`, is below a quarter of the
    /// spacing between `f32` values at the particle's position: `position + v * dt` then
    /// rounds back to `position`, so the component moves nothing. Drag only ever shrinks
    /// it afterwards (and ground contact only shrinks the horizontal ones), so it would
    /// never move anything again. Zeroing it is therefore exact: at a fixed `dt` no
    /// position differs, by a single bit, from what the unflushed loop computes. The
    /// quarter, not a half, covers a position at a power of two, where the spacing
    /// below is half the spacing above.
    ///
    /// The vertical component is flushed only for classes with no gravity: gravity adds
    /// `g * dt` every step, so a small vertical velocity is on its way somewhere and is
    /// not decaying. Positions within about `1e-30` m of an axis floor the resolution at
    /// [`f32::MIN_POSITIVE`], so the step itself is never subnormal either.
    fn integrate_free_flight(&mut self, dt: f32) {
        // A quarter of an `f32` spacing, relative to the power of two below a value.
        const ROUNDS_AWAY: f32 = f32::EPSILON * 0.25;
        const EXPONENT: u32 = 0x7f80_0000;

        // Hoisted per-class so the inner loop reads registers rather than chasing
        // the class table per particle.
        let mut gravity = [0.0f32; MAX_CLASSES];
        let mut damping = [0.0f32; MAX_CLASSES];
        // 1 where the vertical velocity only decays (no gravity), 0 where it does not.
        let mut decays_y = [0.0f32; MAX_CLASSES];
        for c in 0..MAX_CLASSES {
            gravity[c] = self.classes[c].gravity * dt;
            // Clamped so a large `drag * dt` cannot flip the velocity sign, which
            // would turn heavy damping into a bounce.
            damping[c] = 1.0 - (self.classes[c].drag * dt).min(1.0);
            decays_y[c] = if self.classes[c].gravity == 0.0 { 1.0 } else { 0.0 };
        }

        // The spacing a step has to beat to move a coordinate. Exponent bits only, so
        // it is a power of two (or zero) and the multiply is exact.
        let resolution = |p: f32| {
            (f32::from_bits(p.to_bits() & EXPONENT) * ROUNDS_AWAY).max(f32::MIN_POSITIVE)
        };

        let n = self.len();
        // Equal-length slices, so the bounds checks fold out of the loop and it
        // vectorises across particles.
        let (px, py, pz) = (&mut self.pos_x[..n], &mut self.pos_y[..n], &mut self.pos_z[..n]);
        let (vx, vy, vz) = (&mut self.vel_x[..n], &mut self.vel_y[..n], &mut self.vel_z[..n]);
        let remaining = &mut self.remaining[..n];
        let class = &self.class[..n];

        for i in 0..n {
            let c = (class[i] as usize) & (MAX_CLASSES - 1);
            let d = damping[c];

            let mut x = vx[i] * d;
            let mut y = (vy[i] - gravity[c]) * d;
            let mut z = vz[i] * d;

            let (sx, sy, sz) = (x * dt, y * dt, z * dt);
            x = if sx.abs() < resolution(px[i]) { 0.0 } else { x };
            y = if sy.abs() < resolution(py[i]) * decays_y[c] { 0.0 } else { y };
            z = if sz.abs() < resolution(pz[i]) { 0.0 } else { z };

            vx[i] = x;
            vy[i] = y;
            vz[i] = z;
            px[i] += x * dt;
            py[i] += y * dt;
            pz[i] += z * dt;
            remaining[i] -= dt;
        }
    }

    /// Compact out the dead with swap-remove.
    ///
    /// Walks backward so a swapped-in survivor is never re-examined, which makes it
    /// one pass rather than the repeated shuffling a forward `retain` over ten
    /// parallel arrays would do.
    fn retire_expired(&mut self) {
        let mut i = self.len();
        while i > 0 {
            i -= 1;
            if self.remaining[i] > 0.0 {
                continue;
            }
            self.pos_x.swap_remove(i);
            self.pos_y.swap_remove(i);
            self.pos_z.swap_remove(i);
            self.vel_x.swap_remove(i);
            self.vel_y.swap_remove(i);
            self.vel_z.swap_remove(i);
            self.remaining.swap_remove(i);
            self.lifetime.swap_remove(i);
            self.size.swap_remove(i);
            self.class.swap_remove(i);
        }
        // Compaction moved everything, so the rotation cursor no longer refers to
        // the particle it was pointing at. Reset rather than track it -- being
        // approximately-oldest is all this needs to be.
        if self.oldest >= self.len() {
            self.oldest = 0;
        }
    }

    /// Bounce particles off a height field.
    ///
    /// A separate host-side pass because it needs the caller's terrain, which no
    /// GPU kernel here can see. Callers that do not need ground contact simply do
    /// not call it and pay nothing.
    ///
    /// # Arguments
    ///
    /// * `ground_height` - terrain height in metres at a world `(x, z)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{ParticleClass, ParticleEffects};
    /// let mut fx = ParticleEffects::with_capacity(8);
    /// fx.emit_one([0.0, -1.0, 0.0], [0.0, -5.0, 0.0], 1.0, 1.0, 0);
    /// fx.collide_ground(|_, _| 0.0);
    /// assert_eq!(fx.position(0)[1], 0.0);
    /// assert!(fx.velocity(0)[1] >= 0.0);
    /// ```
    pub fn collide_ground<F>(&mut self, ground_height: F)
    where
        F: Fn(f32, f32) -> f32,
    {
        self.collide_ground_with(ground_height, |_| {});
    }

    /// Bounce off a height field, and report each particle that touched down.
    ///
    /// The reporting variant exists because *where a particle landed* is usually
    /// more interesting than the particle. Blood decides where a stain goes, sparks
    /// decide where a scorch mark goes, debris decides where a dent goes -- and all
    /// of that information is generated here and thrown away by the plain version.
    /// Recovering it afterwards is impossible: by the next frame the particle has
    /// either bounced or been retired.
    ///
    /// Only the *first* contact of each particle in a given call is reported, which
    /// is the one decals care about; a particle that bounces reports again on the
    /// frame it lands again.
    ///
    /// # Arguments
    ///
    /// * `ground_height` - terrain height in metres at a world `(x, z)`.
    /// * `on_land` - called once for each particle found below the ground this call.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{ParticleClass, ParticleEffects};
    /// let mut fx = ParticleEffects::with_capacity(8);
    /// fx.emit_one([2.0, -0.5, 0.0], [0.0, -8.0, 0.0], 1.0, 1.0, 0);
    /// let mut hits = Vec::new();
    /// fx.collide_ground_with(|_, _| 0.0, |landing| hits.push(landing));
    /// assert_eq!(hits.len(), 1);
    /// assert_eq!(hits[0].impact_speed, 8.0);
    /// ```
    pub fn collide_ground_with<F, G>(&mut self, ground_height: F, mut on_land: G)
    where
        F: Fn(f32, f32) -> f32,
        G: FnMut(Landing),
    {
        for i in 0..self.len() {
            let floor = ground_height(self.pos_x[i], self.pos_z[i]);
            if self.pos_y[i] >= floor {
                continue;
            }
            let c = (self.class[i] as usize) & (MAX_CLASSES - 1);
            let restitution = self.classes[c].restitution;

            // Reported before the bounce, so `speed` is the speed it arrived at
            // rather than the speed it left with. A decal should scale with the
            // impact, not with what survived it.
            on_land(Landing {
                position: [self.pos_x[i], floor, self.pos_z[i]],
                impact_speed: -self.vel_y[i].min(0.0),
                class: self.class[i],
                size: self.size[i],
            });

            self.pos_y[i] = floor;
            self.vel_y[i] = -self.vel_y[i] * restitution;
            // Tangential friction on contact, so debris slides to a stop instead of
            // skating forever along the surface.
            self.vel_x[i] *= 0.55;
            self.vel_z[i] *= 0.55;
        }
    }


    // ── Reading ──────────────────────────────────────────────────────────────

    /// Position of particle `i`, metres.
    ///
    /// # Arguments
    ///
    /// * `i` - particle index, `0..len()`. Indices are not stable across a step, since
    ///   retirement compacts the arrays.
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
    /// use rs_physics::particles::{ParticleClass, ParticleEffects};
    /// let mut fx = ParticleEffects::with_capacity(8);
    /// fx.emit_one([1.0, 2.0, 3.0], [0.0; 3], 1.0, 1.0, 0);
    /// assert_eq!(fx.position(0), [1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    pub fn position(&self, i: usize) -> [f32; 3] {
        [self.pos_x[i], self.pos_y[i], self.pos_z[i]]
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
    /// use rs_physics::particles::{ParticleClass, ParticleEffects};
    /// let mut fx = ParticleEffects::with_capacity(8);
    /// fx.emit_one([0.0; 3], [4.0, 5.0, 6.0], 1.0, 1.0, 0);
    /// assert_eq!(fx.velocity(0), [4.0, 5.0, 6.0]);
    /// ```
    #[inline]
    pub fn velocity(&self, i: usize) -> [f32; 3] {
        [self.vel_x[i], self.vel_y[i], self.vel_z[i]]
    }

    /// The renderer's scalar for particle `i`, as sampled at emission.
    ///
    /// # Arguments
    ///
    /// * `i` - particle index, `0..len()`.
    ///
    /// # Returns
    ///
    /// The size value; the simulation never reads it.
    ///
    /// # Panics
    ///
    /// If `i >= len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{ParticleClass, ParticleEffects};
    /// let mut fx = ParticleEffects::with_capacity(8);
    /// fx.emit_one([0.0; 3], [0.0; 3], 1.0, 0.25, 0);
    /// assert_eq!(fx.size(0), 0.25);
    /// ```
    #[inline]
    pub fn size(&self, i: usize) -> f32 {
        self.size[i]
    }

    /// The class slot of particle `i`.
    ///
    /// # Arguments
    ///
    /// * `i` - particle index, `0..len()`.
    ///
    /// # Returns
    ///
    /// The slot, already clamped to `0..MAX_CLASSES`.
    ///
    /// # Panics
    ///
    /// If `i >= len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{ParticleClass, ParticleEffects};
    /// let mut fx = ParticleEffects::with_capacity(8);
    /// fx.emit_one([0.0; 3], [0.0; 3], 1.0, 1.0, 3);
    /// assert_eq!(fx.class_of(0), 3);
    /// ```
    #[inline]
    pub fn class_of(&self, i: usize) -> u8 {
        self.class[i]
    }

    /// Fraction of life remaining, 1.0 at birth down to 0.0 at retirement. The
    /// value a renderer fades on.
    ///
    /// # Arguments
    ///
    /// * `i` - particle index, `0..len()`.
    ///
    /// # Returns
    ///
    /// Remaining lifetime over total lifetime, clamped to `[0, 1]`.
    ///
    /// # Panics
    ///
    /// If `i >= len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{ParticleClass, ParticleEffects};
    /// let mut fx = ParticleEffects::with_capacity(8);
    /// fx.emit_one([0.0, 50.0, 0.0], [0.0; 3], 1.0, 1.0, 0);
    /// fx.integrate(0.25);
    /// assert!((fx.remaining_fraction(0) - 0.75).abs() < 1e-6);
    /// ```
    #[inline]
    pub fn remaining_fraction(&self, i: usize) -> f32 {
        let total = self.lifetime[i];
        if total <= 0.0 {
            0.0
        } else {
            (self.remaining[i] / total).clamp(0.0, 1.0)
        }
    }

    /// Raw component slices, for bulk upload to a GPU buffer or a vertex stream
    /// without a per-particle copy.
    ///
    /// # Returns
    ///
    /// The `x`, `y` and `z` position arrays, each [`Self::len`] long, metres.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{ParticleClass, ParticleEffects};
    /// let mut fx = ParticleEffects::with_capacity(8);
    /// fx.emit_one([1.0, 2.0, 3.0], [0.0; 3], 1.0, 1.0, 0);
    /// let (x, y, z) = fx.positions_soa();
    /// assert_eq!((x[0], y[0], z[0]), (1.0, 2.0, 3.0));
    /// ```
    pub fn positions_soa(&self) -> (&[f32], &[f32], &[f32]) {
        (&self.pos_x, &self.pos_y, &self.pos_z)
    }
}

/// Seeded xorshift32 for emission.
///
/// Small and specified, so effects are reproducible from a seed when a caller wants
/// that -- a replay, a regression screenshot, or a lockstep game that wants both
/// peers to see identical sparks. Callers that do not care simply never reuse a
/// seed.
#[derive(Debug, Clone)]
pub struct EffectRng(u32);

impl EffectRng {
    /// A stream seeded with `seed`.
    ///
    /// # Arguments
    ///
    /// * `seed` - any value; zero, which xorshift cannot leave, is replaced by a fixed one.
    ///
    /// # Returns
    ///
    /// The generator.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::EffectRng;
    /// let mut a = EffectRng::new(9);
    /// let mut b = EffectRng::new(9);
    /// assert_eq!(a.next_u32(), b.next_u32());
    /// ```
    pub fn new(seed: u32) -> EffectRng {
        // xorshift is degenerate at zero and never escapes it.
        EffectRng(if seed == 0 { 0x1234_5678 } else { seed })
    }

    /// The next raw value of the xorshift32 sequence.
    ///
    /// # Returns
    ///
    /// A non-zero `u32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::EffectRng;
    /// assert_ne!(EffectRng::new(0).next_u32(), 0);
    /// ```
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Uniform in `[0, 1)`, taken from the top bits -- xorshift's low bits are the
    /// weakest.
    ///
    /// # Returns
    ///
    /// A value in `[0, 1)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::EffectRng;
    /// let u = EffectRng::new(4).unit();
    /// assert!((0.0..1.0).contains(&u));
    /// ```
    #[inline]
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / ((1u32 << 24) as f32)
    }

    /// Uniform in `[min, max)`.
    ///
    /// # Arguments
    ///
    /// * `min` - the lower bound, included.
    /// * `max` - the upper bound, excluded (equal to `min` gives `min`).
    ///
    /// # Returns
    ///
    /// The sample.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::EffectRng;
    /// let v = EffectRng::new(4).range(2.0, 3.0);
    /// assert!((2.0..3.0).contains(&v));
    /// ```
    #[inline]
    pub fn range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.unit()
    }

    /// A direction on the unit sphere, biased upward by `lift`.
    ///
    /// Samples `y` uniformly before taking the ring radius, which gives a genuinely
    /// uniform sphere. Sampling two angles instead clumps points at the poles, and
    /// a burst built that way visibly favours straight up and straight down.
    ///
    /// # Arguments
    ///
    /// * `lift` - upward bias, roughly `0..1`; zero is a uniform sphere.
    ///
    /// # Returns
    ///
    /// A unit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::EffectRng;
    /// let d = EffectRng::new(4).hemisphere(0.35);
    /// let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    /// assert!((len - 1.0).abs() < 1e-5);
    /// ```
    pub fn hemisphere(&mut self, lift: f32) -> [f32; 3] {
        let azimuth = self.unit() * core::f32::consts::TAU;
        let y = self.range(-1.0, 1.0);
        let r = (1.0 - y * y).max(0.0).sqrt();

        let (sin, cos) = azimuth.sin_cos();
        let mut dir = [r * cos, y + lift, r * sin];

        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if len > 1e-6 {
            dir[0] /= len;
            dir[1] /= len;
            dir[2] /= len;
        } else {
            dir = [0.0, 1.0, 0.0];
        }
        dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sparks() -> ParticleEffects {
        let mut fx = ParticleEffects::with_capacity(1024);
        fx.set_class(
            0,
            ParticleClass {
                gravity: 26.0,
                drag: 1.4,
                restitution: 0.32,
            },
        );
        fx
    }

    fn burst(count: u32) -> Burst {
        Burst {
            origin: [0.0, 5.0, 0.0],
            class: 0,
            count,
            speed: 6.0..15.0,
            lifetime: 0.2..0.5,
            size: 0.7..1.3,
            lift: 0.35,
        }
    }

    #[test]
    fn emitted_particles_are_alive_and_finite() {
        let mut fx = sparks();
        let mut rng = EffectRng::new(7);
        fx.emit(&burst(200), &mut rng);

        assert_eq!(fx.len(), 200);
        for i in 0..fx.len() {
            let p = fx.position(i);
            let v = fx.velocity(i);
            assert!(p.iter().all(|c| c.is_finite()));
            assert!(v.iter().all(|c| c.is_finite()));
            assert!(fx.remaining_fraction(i) > 0.0);
        }
    }

    #[test]
    fn particles_retire_when_their_lifetime_runs_out() {
        let mut fx = sparks();
        let mut rng = EffectRng::new(11);
        fx.emit(&burst(500), &mut rng);

        // Longest possible lifetime is 0.5 s; a full second must clear the pool.
        for _ in 0..60 {
            fx.integrate(1.0 / 60.0);
        }
        assert_eq!(fx.len(), 0, "particles outlived their lifetime");
    }

    #[test]
    fn a_full_pool_replaces_rather_than_grows() {
        let mut fx = ParticleEffects::with_capacity(64);
        let mut rng = EffectRng::new(3);
        fx.emit(&burst(1000), &mut rng);

        assert_eq!(fx.len(), 64);
        assert_eq!(fx.capacity(), 64);
    }

    #[test]
    fn gravity_pulls_particles_down() {
        let mut fx = sparks();
        fx.emit_one([0.0, 10.0, 0.0], [0.0, 0.0, 0.0], 5.0, 1.0, 0);
        let start = fx.position(0)[1];

        for _ in 0..10 {
            fx.integrate(1.0 / 60.0);
        }
        assert!(fx.position(0)[1] < start);
    }

    #[test]
    fn drag_removes_speed_and_never_reverses_it() {
        let mut fx = ParticleEffects::with_capacity(16);
        // Drag high enough that `drag * dt` exceeds 1 -- the case the damping clamp
        // exists for. Without it the velocity flips sign and heavy air resistance
        // turns into a bounce.
        fx.set_class(
            0,
            ParticleClass {
                gravity: 0.0,
                drag: 400.0,
                restitution: 0.0,
            },
        );
        fx.emit_one([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], 5.0, 1.0, 0);

        fx.integrate(1.0 / 60.0);
        let vx = fx.velocity(0)[0];
        assert!(vx >= 0.0, "drag reversed the velocity: {vx}");
        assert!(vx < 10.0, "drag did not slow the particle: {vx}");
    }

    #[test]
    fn ground_collision_bounces_and_does_not_sink() {
        let mut fx = sparks();
        fx.emit_one([0.0, 1.0, 0.0], [0.0, -12.0, 0.0], 5.0, 1.0, 0);

        for _ in 0..120 {
            fx.integrate(1.0 / 60.0);
            fx.collide_ground(|_, _| 0.0);
            assert!(
                fx.position(0)[1] >= -1e-4,
                "particle sank through the floor to {}",
                fx.position(0)[1]
            );
        }
    }

    /// The landing hook is what decals are built on, so it has to report the right
    /// things: only particles that actually touched down, at the surface, with the
    /// speed they arrived at rather than the speed they left with.
    #[test]
    fn landings_are_reported_with_their_impact_speed() {
        let mut fx = sparks();
        fx.emit_one([3.0, 5.0, -2.0], [0.0, -20.0, 0.0], 5.0, 1.0, 0);
        // A second particle well above the ground, which must not be reported.
        fx.emit_one([0.0, 40.0, 0.0], [0.0, 0.0, 0.0], 5.0, 1.0, 0);

        let mut landings = Vec::new();
        for _ in 0..30 {
            fx.integrate(1.0 / 60.0);
            fx.collide_ground_with(|_, _| 0.0, |l| landings.push(l));
        }

        assert!(!landings.is_empty(), "the falling particle never reported");
        let first = landings[0];
        assert!(first.impact_speed > 15.0, "impact speed {} too low", first.impact_speed);
        assert!((first.position[1]).abs() < 1e-5, "reported off the surface");
        assert!(
            (first.position[0] - 3.0).abs() < 0.5 && (first.position[2] + 2.0).abs() < 0.5,
            "reported at the wrong place: {:?}",
            first.position
        );
    }

    #[test]
    fn a_particle_that_never_touches_down_is_never_reported() {
        let mut fx = ParticleEffects::with_capacity(8);
        fx.set_class(0, ParticleClass { gravity: 0.0, drag: 0.0, restitution: 0.0 });
        fx.emit_one([0.0, 10.0, 0.0], [1.0, 0.0, 0.0], 100.0, 1.0, 0);

        let mut count = 0;
        for _ in 0..120 {
            fx.integrate(1.0 / 60.0);
            fx.collide_ground_with(|_, _| 0.0, |_| count += 1);
        }
        assert_eq!(count, 0);
    }

    #[test]
    fn emission_is_reproducible_from_a_seed() {

        let mut a = sparks();
        let mut b = sparks();
        a.emit(&burst(128), &mut EffectRng::new(0xABCD));
        b.emit(&burst(128), &mut EffectRng::new(0xABCD));

        for i in 0..a.len() {
            assert_eq!(a.position(i), b.position(i));
            assert_eq!(a.velocity(i), b.velocity(i));
        }
    }

    /// A burst sampled from two angles clumps at the poles. Check the distribution
    /// is actually spread: with zero lift, roughly half should go up and half down.
    #[test]
    fn unlifted_bursts_are_not_pole_biased() {
        let mut fx = ParticleEffects::with_capacity(4096);
        let mut rng = EffectRng::new(99);
        let mut b = burst(2000);
        b.lift = 0.0;
        fx.emit(&b, &mut rng);

        let up = (0..fx.len()).filter(|&i| fx.velocity(i)[1] > 0.0).count();
        let ratio = up as f32 / fx.len() as f32;
        assert!(
            (0.42..0.58).contains(&ratio),
            "emission is biased: {ratio} went up"
        );
    }

    /// The drag flush zeroes a velocity only once it can no longer move its particle,
    /// so it must not change a single position bit against the loop without it, over
    /// a run long enough (500 s) for drag to take every velocity through the whole
    /// subnormal range. And it must leave no subnormal behind.
    #[test]
    fn the_drag_flush_moves_nothing_and_leaves_no_subnormal() {
        let mut fx = ParticleEffects::with_capacity(512);
        fx.set_class(0, ParticleClass { gravity: 26.0, drag: 1.4, restitution: 0.3 });
        fx.set_class(1, ParticleClass { gravity: 0.0, drag: 3.4, restitution: 0.0 });
        let mut rng = EffectRng::new(21);
        for class in [0u8, 1] {
            let mut b = burst(256);
            b.class = class;
            b.lifetime = 1_000.0..1_001.0;
            fx.emit(&b, &mut rng);
        }

        // The loop as it was before the flush.
        let n = fx.len();
        let mut p: Vec<[f32; 3]> = (0..n).map(|i| fx.position(i)).collect();
        let mut v: Vec<[f32; 3]> = (0..n).map(|i| fx.velocity(i)).collect();
        let dt = 1.0f32 / 60.0;
        for _ in 0..30_000 {
            fx.integrate(dt);
            for i in 0..n {
                let class = fx.class(fx.class_of(i));
                let d = 1.0 - (class.drag * dt).min(1.0);
                v[i][1] -= class.gravity * dt;
                for a in 0..3 {
                    v[i][a] *= d;
                    p[i][a] += v[i][a] * dt;
                }
            }
        }

        let mut zeroed = 0;
        for i in 0..n {
            let (got, want) = (fx.position(i), p[i]);
            for a in 0..3 {
                assert_eq!(got[a].to_bits(), want[a].to_bits(), "particle {i} axis {a} moved");
                let vel = fx.velocity(i)[a];
                assert!(!vel.is_subnormal(), "particle {i} axis {a} left subnormal: {vel:e}");
                zeroed += (vel == 0.0) as usize;
            }
            // The reference is stuck in the subnormal range, which is the defect.
            assert!(v[i][0] == 0.0 || v[i][0].is_subnormal());
        }
        // Horizontal for every particle, vertical for the gravity-free class.
        assert_eq!(zeroed, 2 * n + n / 2);
    }

    #[test]
    fn long_runs_stay_finite() {
        let mut fx = sparks();
        let mut rng = EffectRng::new(5);
        for step in 0..2000 {
            if step % 7 == 0 {
                fx.emit(&burst(20), &mut rng);
            }
            fx.integrate(1.0 / 60.0);
            fx.collide_ground(|x, z| (x * 0.05).min(z * 0.05));

            for i in 0..fx.len() {
                assert!(fx.position(i).iter().all(|c| c.is_finite()));
                assert!(fx.velocity(i).iter().all(|c| c.is_finite()));
            }
        }
    }
}
