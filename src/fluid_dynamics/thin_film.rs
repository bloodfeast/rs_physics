//! Thin-film surface flow — the lubrication approximation, and why a sheet of
//! liquid running downhill breaks into rivulets.
//!
//! # The gap this fills
//!
//! The crate had three fluid models and none of them could do a film.
//! `FluidGrid3D` resolves a volume, and a film is not a volume; it is two
//! millimetres of liquid smeared over twenty metres of ground, and a grid fine
//! enough to hold it in the vertical is a grid nobody can afford in the horizontal.
//! `SphFluid` says so itself: surface tension makes SPH produce *"blobs with a
//! visible skin rather than a thin film"*, which is the right behaviour for a splash
//! and the wrong behaviour for what the splash leaves behind.
//!
//! A film is the case where the geometry does the work for you. When the liquid is
//! far thinner than it is wide, inertia is negligible against viscosity, the velocity
//! profile across the depth is known analytically, and the whole Navier–Stokes problem
//! collapses to one scalar per unit area — the depth — obeying a conservation law.
//! That is the **lubrication approximation**, and it is why a two-dimensional buffer
//! can carry a three-dimensional flow honestly.
//!
//!
//! # The law
//!
//! For a Newtonian liquid on a slope, the volumetric flux per unit width is
//!
//! ```text
//!   q = ρ g sinθ h³ / (3 μ)                      [m²/s]
//! ```
//!
//! **The cube is the entire story.** Everything a film does that a diffusion model
//! does not comes out of that exponent:
//!
//! - A patch 10% thicker than its neighbours carries **33% more** flux, and its front
//!   advances **21% faster** — the front speed is `q/h ∝ h²`. So a straight advancing
//!   edge does not stay straight: any bulge outruns the rest of the line, and once
//!   ahead it drains the liquid beside it and grows further. That positive feedback is
//!   what **rivulets** are. They are not an effect to be added to the model; they are
//!   what the exponent does, and they appear on their own the moment the flux law is
//!   right. (Huppert, *Flow and instability of a viscous current down a slope*, Nature
//!   300 (1982), 427–429.)
//! - A linear flux law `q ∝ h` has a front speed independent of depth, so a perturbed
//!   edge translates rigidly and never sharpens. It is a diffusion; it makes puddles
//!   and never makes channels. That comparison is a test in this module
//!   (`the_cube_is_what_makes_rivulets`) and it is the whole justification for the
//!   exponent.
//! - Halving the depth divides the flow by eight. A trail therefore has an *end*
//!   rather than an asymptote.
//!
//! Written for a solver, the driving slope is not the ground's slope but the slope of
//! the liquid's **free surface**, ground plus depth:
//!
//! ```text
//!   q = -(ρ g h³ / 3μ) ∇(z_bed + h)
//! ```
//!
//! which is the same law and also does the levelling: on flat ground the ∇h term
//! spreads a pool, and on a slope the ∇z term outruns it. One expression, both
//! behaviours, no blend factor.
//!
//! # Where the yield stress goes
//!
//! A Newtonian film never stops; it only gets slower. Real spilt liquids stop, because
//! most of them — blood especially, see [`crate::fluid_dynamics::Fluid::blood`] — have
//! a small yield stress. For a Bingham film the top of the layer moves as an unsheared
//! plug and only the part below it carries shear, which changes the flux by an exactly
//! known factor:
//!
//! ```text
//!   q = (ρ g sinθ h³ / 3μ) · (1 - 1.5X + 0.5X³),   X = h_y/h,  h_y = τ_y/(ρ g sinθ)
//! ```
//!
//! At `X = 0` the factor is 1 and the law is the Newtonian one. At `X = 1` it is
//! exactly 0, so the flux is not switched off at a threshold — it *reaches* zero
//! continuously at [`FilmFlow::arrest_thickness`]. A solver therefore gets a film that
//! comes to rest and stays there without a tuned cutoff, and one that terminates.
//!
//! # Why this is a law and not a solver
//!
//! **A GPU compute shader is the intended consumer.** So the flux law, the arrest
//! criterion, the wave speed and the stability limit are pure functions of scalars
//! plus a [`FilmFlow`] — three `f64` fields, each a precomputed grouping of the
//! fluid's constants, none recomputed per cell. Transcribing that into WGSL is a
//! uniform struct and four lines of arithmetic:
//!
//! ```wgsl
//! struct FilmFlow { hydrostatic: f32, mobility: f32, yield_length: f32 };
//!
//! fn film_flux(f: FilmFlow, slope: f32, h: f32) -> f32 {
//!     let drive = abs(slope) * h;
//!     let x = f.yield_length / drive;
//!     let q = f.mobility * slope * h * h * h * (1.0 - 1.5 * x + 0.5 * x * x * x);
//!     return select(0.0, q, drive > f.yield_length);
//! }
//! ```
//!
//! One thing a shader author must copy along with the formula: **difference the ground
//! and the depth separately.** The free-surface slope is
//! `((z_i - z_j) + (h_i - h_j)) / dx`, never `((z_i + h_i) - (z_j + h_j)) / dx`. Bed
//! elevations are metres and film depths are millimetres, so forming the sum first
//! throws away the depth difference — which *is* the levelling term. In `f32` at
//! map-scale elevations it is not an accuracy loss, it is total: a pool stops levelling
//! altogether. Written the first way, `f32` is fine on the GPU.
//!
//! # Layout and vectorization
//!
//! [`FilmGrid`] is structure-of-arrays: depth, bed elevation and the two face-flux
//! buffers are separate contiguous `Vec<f64>`, never a `Vec<Cell>`. That is what lets
//! a pass over depth vectorize, and it is also exactly what a GPU wants, which is a
//! good sign it is the right shape.
//!
//! Every loop in [`FilmGrid::step`] is flat and branch-free over the whole grid. The
//! trick that buys that is **padding the flux buffers rather than testing for edges**:
//! `flux_x` holds `cells + 1` faces indexed by the receiving cell and `flux_y` holds
//! `cells + width`, so "the face above cell `i`" is always `flux_y[i]` and the ones
//! that do not exist are zeros nobody ever writes. There is no `if x > 0` anywhere in
//! the step, which is the difference between a loop LLVM vectorizes and one it does
//! not.
//!
//! Checked rather than hoped for, since "it should vectorize" is the same class of
//! claim as "it should be fast". Against `rustc` 1.9x in release,
//! `cargo rustc --release --emit asm`, the body of `FilmGrid::step` contains 42
//! `vmulpd`, 18 `vaddpd`, 12 `vsubpd`, 12 `vmaxpd` and 10 `vcmpltpd`/`vblendvpd` pairs
//! on `%ymm` registers — four `f64` lanes at a time, with the NaN sink emitted as a
//! blend rather than a branch. Every bounds check LLVM could not discharge sits in the
//! cold tail of the function, past the last vector instruction, so none of them is in a
//! loop body. That is also why the first pass is written a row at a time:
//! `row[x + 1]` against a slice of length `width` is provable and
//! `flux_x[y * width + x + 1]` against `width * height + 1` is not.
//!
//! Counted rather than eyeballed: of the double-precision arithmetic in
//! [`FilmFlow::flux_batch`], **76% is packed** (65 `pd` against 20 `sd`, 26 of the packed
//! ops on `%ymm`); in [`FilmGrid::step`], **64%** (244 against 139, 108 on `%ymm`). The
//! scalar remainder is loop prologues, epilogues and tail elements across six loops, not
//! arithmetic that failed to vectorize.
//!
//! No hand-written intrinsics, and deliberately none: the production consumer is a
//! compute shader, so `unsafe` architecture-specific code here would be maintained
//! forever to accelerate a reference implementation.
//!
//! # What it costs
//!
//! Measured, `cargo bench --bench thin_film`, release, one Windows x86-64 desktop. The
//! ratios are the durable part; the absolute figures move with the machine.
//!
//! | | per call | per element |
//! |---|---|---|
//! | `flux_batch`, Newtonian, 100 k faces | 72 µs | **0.72 ns** |
//! | the same loop written by hand, calling `flux` | 318 µs | 3.2 ns |
//! | `flux_batch`, with a yield stress | 91 µs | 0.91 ns |
//! | the same, by hand | 295 µs | 2.9 ns |
//! | `FilmGrid::step`, 128 × 128 | 101 µs | 6.1 ns/cell |
//! | `FilmGrid::step`, 1400 × 1000 | **16.8 ms** | 12 ns/cell |
//! | `FilmGrid::max_step`, 1400 × 1000 | 5.2 ms | 3.7 ns/cell |
//!
//! Two things follow, and neither was safe to assume.
//!
//! **The batch entry point is worth 3.3–4.4×**, not merely tidier. Part of that is the
//! yield-stress unswitch — the Bingham divide costs 26% — but only part: the two
//! hand-written loops land within 8% of each other, so most of the gap is the slice
//! shape, which is what lets LLVM discharge the bounds checks and vectorize at all. The
//! prediction going in was that the unswitch was the whole story. It was not.
//!
//! **A whole-map sweep is not a frame budget.** 1400 × 1000 is Ridgeline's stain buffer
//! at 5 texels per metre over a 280 × 200 m map, and 16.8 ms of it is a frame. That is
//! the measurement behind [`FilmGrid`] declining to own the sweep: liquid covers a
//! percent or two of a map, and a caller that visits only the tiles holding any pays a
//! percent or two of that. It is also why [`FilmGrid::max_step`] documents an escape —
//! 5.2 ms to scan for the deepest cell is absurd beside one [`FilmFlow::max_step`] call
//! with a depth the caller already knows.
//!
//! Per-cell functions are **total** — no `Result`, no panic, no allocation. The
//! validation the rest of this module does per call at the API edge is done here once,
//! at [`FilmFlow::new`] and at [`FilmGrid`]'s setters, because a discriminant check per
//! cell is a branch per cell and its answer cannot change. [`FilmFlow::flux_batch`] is
//! the entry point a solver should reach for; [`FilmFlow::flux`] is the same arithmetic
//! for one cell, and is what the WGSL above and the tests are written against.
//!
//! # Why `f64` here
//!
//! Deliberately, and not merely because the crate is. The state is a depth in metres
//! sitting on a bed elevation in metres, and the two differ by four orders of
//! magnitude; the levelling term is the difference between neighbouring depths, which
//! can be micrometres. Differencing separately (above) keeps that safe in `f64` with
//! room to spare, and would keep it *just* safe in `f32` — but the same buffers also
//! carry the mass-conservation guarantee across tens of thousands of steps, and `f32`'s
//! seven digits put a visible drift inside a minute of a match. The GPU implementation
//! will be `f32` and should be: it re-uploads from a `f64` authority, or it accepts
//! drift in something nobody is measuring. That is a decision the consumer makes with
//! its own error budget; the library keeps the accurate one.
//!
//! # What this is not
//!
//! - Not inertial. The lubrication approximation assumes the reduced Reynolds number is
//!   small, which for millimetre films of anything thicker than water it is. A splash
//!   arriving at 40 m/s is not a film; that is `SphFluid`'s job, and the film begins
//!   where the splash has come to rest.
//! - Not capillary. Surface tension is absent from the flux law, so this model thins a
//!   film without limit rather than letting it break into dry patches and beads.
//!   [`crate::fluid_dynamics::puddle_depth`] is the scale at which that assumption
//!   fails.
//! - Not a wetting model. Whether liquid sticks to what it runs over, soaks in, or
//!   dries is substrate behaviour and belongs to the caller.

use crate::utils::PhysicsError;
use super::fluid_dynamics::Fluid;
use super::validation::{validate_finite, validate_positive};

/// `h^N`, resolved at monomorphisation.
///
/// Exists so the cubic law and the linear law the tests compare it against run through
/// *the same code*, and the comparison is therefore about the exponent and nothing
/// else. For `N = 3` this is two multiplies, not a `powi` call.
#[inline(always)]
fn shape<const N: u32>(h: f64) -> f64 {
    match N {
        1 => h,
        3 => h * h * h,
        _ => h.powi(N as i32),
    }
}

/// The flux law, as the compiler sees it in an inner loop: total, branch-free, and
/// free of anything that would stop a loop containing it from vectorizing.
///
/// `YIELDS` unswitches the Bingham plug correction, which is a property of the fluid
/// and therefore loop-invariant. Hoisting it by hand rather than hoping LLVM does it
/// also removes an unconditional divide from the Newtonian path.
///
/// The final `if` is a select, not a jump: `drive > yield_length` is false for a NaN in
/// either argument, so a NaN is *stopped here* rather than spreading to every cell it
/// touches within a few steps. The checking that keeps NaN out in the first place is at
/// [`FilmFlow::new`] and [`FilmGrid`]'s setters, where it costs once instead of per
/// cell.
#[inline(always)]
fn flux_kernel<const N: u32, const YIELDS: bool>(
    mobility: f64,
    yield_length: f64,
    gain: f64,
    slope: f64,
    thickness: f64,
) -> f64 {
    // `drive` is the wall shear stress over ρg: |slope|·h. Written this way the yield
    // test needs no division.
    let drive = slope.abs() * thickness;
    let base = gain * mobility * slope * shape::<N>(thickness);
    let q = if YIELDS {
        let x = yield_length / drive;
        base * (1.0 - 1.5 * x + 0.5 * x * x * x)
    } else {
        base
    };
    if drive > yield_length {
        q
    } else {
        0.0
    }
}

/// Everything about a fluid and a gravity that a film flux needs, grouped once.
///
/// Three numbers, each a precomputed combination that would otherwise be recomputed per
/// cell per step. Build it when the fluid changes — approximately never — and pass it
/// to the per-cell functions, or upload it to a shader as a twelve-byte uniform.
///
/// # Examples
///
/// ```
/// use rs_physics::fluid_dynamics::{Fluid, FilmFlow};
///
/// let blood = FilmFlow::new(&Fluid::blood(), 9.81).unwrap();
/// let honey = FilmFlow::new(&Fluid::honey(), 9.81).unwrap();
///
/// // Same depth, same slope: blood runs and honey creeps, and the ratio is not a
/// // dial — it is the fluids' own densities and viscosities.
/// let (slope, depth) = (0.1, 0.002);
/// assert!(blood.velocity(slope, depth) > 1000.0 * honey.velocity(slope, depth));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilmFlow {
    hydrostatic: f64,
    mobility: f64,
    yield_length: f64,
}

impl FilmFlow {
    /// Groups a Newtonian fluid and a gravity into a film law.
    ///
    /// # Arguments
    ///
    /// * `fluid` — the liquid. Only its density and viscosity are used.
    /// * `gravity` — m/s², strictly positive.
    ///
    /// # Errors
    ///
    /// Returns [`PhysicsError::CalculationError`] if the gravity or either of the
    /// fluid's properties is not finite and strictly positive.
    ///
    /// **The fluid is re-validated here rather than trusted.** [`Fluid`]'s fields are
    /// public, so `Fluid { density: -1.0, viscosity: f64::NAN }` is a legal value that
    /// [`Fluid::new`]'s checks never saw. A NaN viscosity makes `mobility` NaN, and
    /// from there every flux is NaN and every bound check against one silently takes
    /// the false branch — the failure would be total and completely quiet. This is the
    /// boundary where it is cheap to stop, and it is the *only* validation on the path,
    /// which is what lets everything downstream of it be total.
    pub fn new(fluid: &Fluid, gravity: f64) -> Result<FilmFlow, PhysicsError> {
        validate_finite(fluid.density, "density")?;
        validate_finite(fluid.viscosity, "viscosity")?;
        validate_finite(gravity, "gravity")?;
        validate_positive(fluid.density, "density")?;
        validate_positive(fluid.viscosity, "viscosity")?;
        validate_positive(gravity, "gravity")?;
        let hydrostatic = fluid.density * gravity;
        Ok(FilmFlow {
            hydrostatic,
            mobility: hydrostatic / (3.0 * fluid.viscosity),
            yield_length: 0.0,
        })
    }

    /// Adds a yield stress, making the film one that can come to rest.
    ///
    /// See the module docs for the Bingham plug correction this switches on. For blood,
    /// pass [`crate::fluid_dynamics::BLOOD_YIELD_STRESS`].
    ///
    /// # Arguments
    ///
    /// * `yield_stress` — τ_y in Pa, finite and non-negative. Zero restores the
    ///   Newtonian law exactly, and is also the faster path: it takes a divide out of
    ///   the inner loop.
    ///
    /// # Errors
    ///
    /// Returns [`PhysicsError::CalculationError`] if the yield stress is not finite or
    /// is negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{Fluid, FilmFlow, BLOOD_YIELD_STRESS};
    ///
    /// let blood = FilmFlow::new(&Fluid::blood(), 9.81).unwrap()
    ///     .with_yield_stress(BLOOD_YIELD_STRESS).unwrap();
    ///
    /// // On level ground nothing runs, however deep it is.
    /// assert_eq!(blood.flux(0.0, 0.01), 0.0);
    /// // On a slope, a film below the arrest thickness is held.
    /// let held = blood.arrest_thickness(0.1) * 0.5;
    /// assert_eq!(blood.flux(0.1, held), 0.0);
    /// ```
    pub fn with_yield_stress(mut self, yield_stress: f64) -> Result<FilmFlow, PhysicsError> {
        validate_finite(yield_stress, "yield_stress")?;
        if yield_stress < 0.0 {
            return Err(PhysicsError::CalculationError(format!(
                "yield stress must be non-negative, got {}",
                yield_stress
            )));
        }
        self.yield_length = yield_stress / self.hydrostatic;
        Ok(self)
    }

    /// ρg, in Pa/m: the pressure a metre of this liquid puts on what is under it.
    #[inline]
    pub fn hydrostatic(&self) -> f64 {
        self.hydrostatic
    }

    /// ρg/3μ, in 1/(m·s): the coefficient of `slope · h³` in the flux.
    #[inline]
    pub fn mobility(&self) -> f64 {
        self.mobility
    }

    /// τ_y/ρg, in metres: divide by the slope to get [`Self::arrest_thickness`].
    #[inline]
    pub fn yield_length(&self) -> f64 {
        self.yield_length
    }

    /// Volumetric flux per unit width, m²/s, signed with the slope.
    ///
    /// ```text
    ///   q = mobility · slope · h³ · (1 - 1.5X + 0.5X³)
    /// ```
    ///
    /// The scalar form: what the WGSL in the module docs transcribes and what the tests
    /// are written against. For a whole row or grid use [`Self::flux_batch`], which is
    /// the same arithmetic in a shape the compiler can put four cells through at a
    /// time.
    ///
    /// # Arguments
    ///
    /// * `slope` — the free-surface gradient, dimensionless and **signed**: positive
    ///   means downhill in the positive coordinate direction, and the flux comes back
    ///   with the same sign. For a bed of angle θ this is `tanθ`; the difference from
    ///   the `sinθ` of the derivation is far below the error in knowing the depth.
    /// * `thickness` — h in metres, non-negative.
    ///
    /// # Behaviour at the edges, which is deliberate
    ///
    /// Total: no `Result`, no panic, no allocation, and it returns exactly `0.0` for a
    /// zero slope, a zero thickness, a film below the arrest thickness, **and for a NaN
    /// in either argument**. The inner loop of a solver is the wrong place to raise an
    /// error, and a NaN that leaked in here would otherwise reach every cell within a
    /// few steps. It is stopped rather than propagated; the checking that keeps it out
    /// is at [`Self::new`] and [`FilmGrid`]'s setters.
    #[inline]
    pub fn flux(&self, slope: f64, thickness: f64) -> f64 {
        flux_kernel::<3, true>(self.mobility, self.yield_length, 1.0, slope, thickness)
    }

    /// [`Self::flux`] over contiguous slices — the form a solver should reach for.
    ///
    /// One length check, then a flat loop with no bounds checks and no branches, which
    /// is what the auto-vectorizer needs. Allocates nothing.
    ///
    /// **Measured at 3.3–4.4× the same loop written by hand around [`Self::flux`]** —
    /// 0.72 ns a face against 3.2 — so this is a speed-up and not only a tidier
    /// signature. Two things buy it: the yield-stress branch is hoisted out of the loop
    /// by a const generic, worth 26%, and re-slicing all three inputs to one known
    /// length lets LLVM drop the bounds checks and vectorize, which is the rest. See the
    /// module docs for the whole table.
    ///
    /// # Arguments
    ///
    /// * `slopes` — free-surface gradients, one per face.
    /// * `thickness` — the **donor** depth at each face: the depth on the uphill side.
    ///   Averaging the two sides instead would let a dry cell donate liquid.
    /// * `out` — filled with the flux at each face, m²/s.
    ///
    /// # Errors
    ///
    /// Returns [`PhysicsError::CalculationError`] if the three slices differ in length.
    /// Once per batch, not once per cell — that is the point.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::fluid_dynamics::{Fluid, FilmFlow};
    ///
    /// let flow = FilmFlow::new(&Fluid::blood(), 9.81).unwrap();
    /// let slopes = [0.1, 0.1, 0.0, -0.1];
    /// let depths = [0.001, 0.002, 0.002, 0.002];
    /// let mut flux = [0.0; 4];
    /// flow.flux_batch(&slopes, &depths, &mut flux).unwrap();
    ///
    /// assert!((flux[1] / flux[0] - 8.0).abs() < 1e-9); // twice as deep, eight times
    /// assert_eq!(flux[2], 0.0);                        // level ground, no flow
    /// assert!(flux[3] < 0.0);                          // and downhill is a direction
    /// ```
    pub fn flux_batch(
        &self,
        slopes: &[f64],
        thickness: &[f64],
        out: &mut [f64],
    ) -> Result<(), PhysicsError> {
        if slopes.len() != thickness.len() || slopes.len() != out.len() {
            return Err(PhysicsError::CalculationError(format!(
                "flux_batch needs equal-length slices, got {}, {} and {}",
                slopes.len(),
                thickness.len(),
                out.len()
            )));
        }
        if self.yield_length > 0.0 {
            self.flux_slice::<3, true>(slopes, thickness, out);
        } else {
            self.flux_slice::<3, false>(slopes, thickness, out);
        }
        Ok(())
    }

    /// The batch loop, unswitched on whether the fluid yields. Lengths are equal by the
    /// caller's check; the re-slicing is what tells LLVM so and removes the bounds
    /// checks.
    #[inline]
    fn flux_slice<const N: u32, const YIELDS: bool>(
        &self,
        slopes: &[f64],
        thickness: &[f64],
        out: &mut [f64],
    ) {
        let n = out.len();
        let slopes = &slopes[..n];
        let thickness = &thickness[..n];
        let (mobility, yield_length) = (self.mobility, self.yield_length);
        for i in 0..n {
            out[i] = flux_kernel::<N, YIELDS>(
                mobility,
                yield_length,
                1.0,
                slopes[i],
                thickness[i],
            );
        }
    }

    /// Depth-averaged velocity of the film, m/s, signed with the slope.
    ///
    /// `u = q/h`. This is the Nusselt result `ρ g sinθ h² / 3μ` for a Newtonian liquid
    /// — the speed a trail of spilt liquid visibly runs at, and the quantity to reach
    /// for when asking "how long before it gets there". Returns `0.0` for zero or
    /// negative thickness.
    #[inline]
    pub fn velocity(&self, slope: f64, thickness: f64) -> f64 {
        if !(thickness > 0.0) {
            return 0.0;
        }
        self.flux(slope, thickness) / thickness
    }

    /// Speed at which a change in depth travels, m/s, always non-negative.
    ///
    /// `c = dq/dh = 3 · mobility · h · (|slope|·h - yield_length)`, which for a
    /// Newtonian film is **three times the depth-averaged velocity**. That factor of
    /// three is not a curiosity: it is why the leading edge of a spill outruns the
    /// liquid in it, and it is the speed a grid must respect rather than the flow
    /// speed. Confusing the two gives a solver that looks stable in a test and
    /// oscillates at three times the timestep in the field.
    #[inline]
    pub fn wave_speed(&self, slope: f64, thickness: f64) -> f64 {
        let drive = slope.abs() * thickness;
        if !(drive > self.yield_length) {
            return 0.0;
        }
        3.0 * self.mobility * thickness * (drive - self.yield_length)
    }

    /// Wall shear rate in the film, s⁻¹, always non-negative.
    ///
    /// `γ̇ = ρ g |slope| h / μ`, the velocity gradient at the ground, where it is
    /// steepest. **This is the function that says whether a Newtonian viscosity was a
    /// defensible choice**: hand it to
    /// [`crate::fluid_dynamics::blood_apparent_viscosity`] and compare the answer with
    /// the viscosity you built the [`FilmFlow`] from. For blood films between half a
    /// millimetre and five deep on ordinary ground the two agree to about 15%, which is
    /// what makes [`Fluid::blood`] usable here — asserted by a test rather than in
    /// prose.
    #[inline]
    pub fn shear_rate(&self, slope: f64, thickness: f64) -> f64 {
        3.0 * self.mobility * slope.abs() * thickness
    }

    /// The film depth below which this liquid does not move on this slope, metres.
    ///
    /// `h = τ_y / (ρ g slope)`. Infinite on level ground, which is the correct answer: a
    /// yield-stress fluid on the flat never runs downhill, because there is no down.
    /// Zero for a Newtonian fluid, which never stops.
    ///
    /// Small. For blood on a one-in-ten slope it is about five micrometres — nowhere
    /// near what holds a visible pool together. If you want the depth a spill *settles*
    /// at on level ground, that is surface tension's doing and lives in
    /// [`crate::fluid_dynamics::puddle_depth`].
    #[inline]
    pub fn arrest_thickness(&self, slope: f64) -> f64 {
        let slope = slope.abs();
        if !(slope > 0.0) {
            return if self.yield_length > 0.0 {
                f64::INFINITY
            } else {
                0.0
            };
        }
        self.yield_length / slope
    }

    /// The largest timestep a grid of this cell size may take, seconds.
    ///
    /// The CFL condition for the kinematic wave, `dt ≤ dx/c`. At exactly this value a
    /// disturbance crosses one whole cell in one step, which is the boundary of what the
    /// grid can represent; take a fraction of it. [`f64::INFINITY`] when nothing moves.
    ///
    /// Because `c` goes as `h²`, **the deepest cell on the map sets the timestep for all
    /// of them**, and quadratically: a spill twice as deep needs a step four times
    /// smaller. That is the number to watch when a film solver mysteriously destabilises
    /// after a big splash. [`FilmGrid::max_step`] finds it for a whole grid.
    #[inline]
    pub fn max_step(&self, slope: f64, thickness: f64, cell_size: f64) -> f64 {
        let speed = self.wave_speed(slope, thickness);
        if !(speed > 0.0) {
            return f64::INFINITY;
        }
        cell_size / speed
    }
}

/// A CPU reference solver for [`FilmFlow`] on a regular square grid.
///
/// Finite-volume, donor-cell upwinded, with the flux coming from the same kernel
/// [`FilmFlow::flux`] and [`FilmFlow::flux_batch`] use rather than a private copy of the
/// arithmetic. It exists to be tested, and to be the thing a shader is checked against;
/// the intended production consumer is the shader.
///
/// # Guarantees
///
/// - **No allocation in [`Self::step`].** Every buffer is sized once, in [`Self::new`].
/// - **Mass is conserved** to float rounding: each face flux is added to one cell and
///   subtracted from another, and the padded boundary faces are permanent zeros, so the
///   liquid has nowhere to leak to.
/// - **Depth never goes negative**, at any timestep. A cell that would give away more
///   than it holds has all its outgoing faces scaled by one common factor. A step above
///   the CFL limit gives a wrong answer rather than an exploding one, which is a much
///   easier failure to see.
/// - **Flat and level stays exactly flat.** Not "to within an epsilon": a zero slope
///   gives a flux of exactly `0.0`, and adding `0.0` to a depth returns the same bits.
///   A film solver that drifts on still ground repaints its whole buffer forever.
///
/// # Layout
///
/// Structure-of-arrays, and the flux buffers are **padded rather than bounds-tested**:
/// `flux_x[i]` is the face entering cell `i` from the left and `flux_x[i+1]` the one
/// leaving to the right; `flux_y[i]` and `flux_y[i+width]` likewise. The faces that do
/// not exist are indices nobody writes, so they stay `0.0` forever and there is no `if`
/// for an edge anywhere in the step. Four flat, branch-free passes over the grid.
///
/// It visits every cell whether or not it holds liquid, and **at map scale that is the
/// wrong sweep by a wide measured margin**: 1400 × 1000 cells cost 16.8 ms a step, which
/// is a whole frame spent on a buffer that is 99% dry. A caller must track which tiles
/// hold liquid and step only those. That bookkeeping is not here because it depends on
/// how the caller stores its liquid, and because at 6.1 ns a cell the arithmetic is not
/// what needs fixing — the number of cells is.
#[derive(Debug, Clone)]
pub struct FilmGrid {
    width: usize,
    height: usize,
    cell_size: f64,
    ground: Vec<f64>,
    thickness: Vec<f64>,
    /// `cells + 1` entries. `flux_x[i]` is the flux from cell `i-1` into cell `i`;
    /// index `y*width` for every `y`, and index `cells`, are faces that do not exist
    /// and are never written.
    flux_x: Vec<f64>,
    /// `cells + width` entries. `flux_y[i]` is the flux from cell `i-width` into cell
    /// `i`; the first and last `width` entries are faces that do not exist.
    flux_y: Vec<f64>,
    /// Per-cell outflow, then the factor that limits it. One buffer, two uses, no
    /// allocation.
    work: Vec<f64>,
}

impl FilmGrid {
    /// A grid of dry, level ground.
    ///
    /// # Arguments
    ///
    /// * `width`, `height` — cells, both at least 1.
    /// * `cell_size` — metres per cell, finite and strictly positive. Square: the flux
    ///   law is isotropic and rectangular cells would need two of it.
    ///
    /// # Errors
    ///
    /// Returns [`PhysicsError::CalculationError`] for a zero dimension, a non-positive
    /// or non-finite cell size, or a cell count that overflows `usize`.
    pub fn new(width: usize, height: usize, cell_size: f64) -> Result<FilmGrid, PhysicsError> {
        if width == 0 || height == 0 {
            return Err(PhysicsError::CalculationError(format!(
                "grid must have at least one cell in each direction, got {}x{}",
                width, height
            )));
        }
        validate_finite(cell_size, "cell_size")?;
        validate_positive(cell_size, "cell_size")?;
        // Checked, because `width * height` is a product of two caller-supplied
        // `usize`s and a wrapped one would size the buffers for far fewer cells than
        // the indices later computed from `width` and `height` — which is an
        // out-of-bounds panic at best and, in the padded buffers, an off-by-`width`
        // silent corruption at worst. The flux buffers are longer than the grid, so
        // their lengths are checked too rather than assumed to follow.
        let too_big = || {
            PhysicsError::CalculationError(format!("a {}x{} grid does not fit in memory", width, height))
        };
        let cells = width.checked_mul(height).ok_or_else(too_big)?;
        cells
            .checked_add(width)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(too_big)?;
        Ok(FilmGrid {
            width,
            height,
            cell_size,
            ground: vec![0.0; cells],
            thickness: vec![0.0; cells],
            flux_x: vec![0.0; cells + 1],
            flux_y: vec![0.0; cells + width],
            work: vec![0.0; cells],
        })
    }

    /// Cells across.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Cells down.
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Metres per cell.
    #[inline]
    pub fn cell_size(&self) -> f64 {
        self.cell_size
    }

    /// The depth of liquid in every cell, row-major. Contiguous, so a caller can
    /// vectorize its own pass — a shader upload, a colour ramp — over it directly.
    #[inline]
    pub fn thickness(&self) -> &[f64] {
        &self.thickness
    }

    /// The bed elevation of every cell, row-major.
    #[inline]
    pub fn ground(&self) -> &[f64] {
        &self.ground
    }

    /// Set the bed elevation at a cell, metres.
    ///
    /// # Errors
    ///
    /// Out of bounds, or a non-finite elevation.
    pub fn set_ground(&mut self, x: usize, y: usize, elevation: f64) -> Result<(), PhysicsError> {
        let i = self.index(x, y)?;
        validate_finite(elevation, "elevation")?;
        self.ground[i] = elevation;
        Ok(())
    }

    /// Bed elevation in metres at a cell.
    ///
    /// # Errors
    ///
    /// Out of bounds.
    pub fn ground_at(&self, x: usize, y: usize) -> Result<f64, PhysicsError> {
        Ok(self.ground[self.index(x, y)?])
    }

    /// Replace the whole bed in one call, row-major.
    ///
    /// The bulk form, because setting a 1400×1000 terrain one `Result` at a time is
    /// a million branches for a check whose answer is the same every time. One length
    /// check and one validity scan, then a memory copy.
    ///
    /// # Errors
    ///
    /// A length mismatch, or any non-finite elevation — and in that case **nothing is
    /// written**, so a rejected terrain cannot leave the grid half-updated.
    pub fn set_ground_from(&mut self, elevations: &[f64]) -> Result<(), PhysicsError> {
        if elevations.len() != self.ground.len() {
            return Err(PhysicsError::CalculationError(format!(
                "expected {} elevations for a {}x{} grid, got {}",
                self.ground.len(),
                self.width,
                self.height,
                elevations.len()
            )));
        }
        if let Some(bad) = elevations.iter().position(|e| !e.is_finite()) {
            return Err(PhysicsError::CalculationError(format!(
                "elevation {} at index {} is not finite; the bed was left unchanged",
                elevations[bad], bad
            )));
        }
        self.ground.copy_from_slice(elevations);
        Ok(())
    }

    /// Set the depth of liquid in a cell, metres.
    ///
    /// # Errors
    ///
    /// Out of bounds, or a depth that is not finite and non-negative.
    ///
    /// **This is where float validity is established for the whole solver.** The step
    /// loop does no `is_finite` checks because it does not need to: nothing can put a
    /// NaN or a negative depth into the buffer except through here and
    /// [`Self::add_thickness`], and both refuse.
    pub fn set_thickness(&mut self, x: usize, y: usize, depth: f64) -> Result<(), PhysicsError> {
        let i = self.index(x, y)?;
        Self::check_depth(depth)?;
        self.thickness[i] = depth;
        Ok(())
    }

    /// Add liquid to a cell, metres of depth. The usual way a spill enters the grid.
    ///
    /// # Errors
    ///
    /// Out of bounds, or an addition that is not finite and non-negative.
    pub fn add_thickness(&mut self, x: usize, y: usize, depth: f64) -> Result<(), PhysicsError> {
        let i = self.index(x, y)?;
        Self::check_depth(depth)?;
        self.thickness[i] += depth;
        Ok(())
    }

    /// Depth of liquid in a cell, metres.
    ///
    /// # Errors
    ///
    /// Out of bounds.
    pub fn thickness_at(&self, x: usize, y: usize) -> Result<f64, PhysicsError> {
        Ok(self.thickness[self.index(x, y)?])
    }

    /// Total liquid on the grid, cubic metres. The quantity a step must not change.
    pub fn total_volume(&self) -> f64 {
        let cell_area = self.cell_size * self.cell_size;
        self.thickness.iter().sum::<f64>() * cell_area
    }

    /// The largest timestep this grid may currently take, seconds.
    ///
    /// The tightest [`FilmFlow::max_step`] over every interior face, using the same face
    /// slopes and donor depths the step itself will. [`f64::INFINITY`] if nothing on the
    /// grid is moving.
    ///
    /// It changes as the liquid moves — a splash lands, the depth quadruples, the
    /// allowable step falls by sixteen — so it is a per-frame question, not a setup-time
    /// one. O(cells) and allocation-free, but it is a whole extra sweep and it is priced
    /// accordingly: **5.2 ms on a 1400 × 1000 grid**, which is most of a frame to
    /// rediscover something the caller usually already knows. A caller that has just
    /// deposited the deepest liquid on the map should call [`FilmFlow::max_step`] with
    /// that depth and skip this entirely.
    pub fn max_step(&self, flow: &FilmFlow) -> f64 {
        let (w, h) = (self.width, self.height);
        let inv_dx = 1.0 / self.cell_size;
        let cells = w * h;
        let ground = &self.ground[..cells];
        let depth = &self.thickness[..cells];

        let mut smallest = f64::INFINITY;
        for y in 0..h {
            let r = y * w;
            let bed = &ground[r..r + w];
            let film = &depth[r..r + w];
            for x in 0..w - 1 {
                let slope = ((bed[x] - bed[x + 1]) + (film[x] - film[x + 1])) * inv_dx;
                let donor = if slope > 0.0 { film[x] } else { film[x + 1] };
                smallest = smallest.min(flow.max_step(slope, donor, self.cell_size));
            }
        }
        for i in 0..cells - w {
            let j = i + w;
            let slope = ((ground[i] - ground[j]) + (depth[i] - depth[j])) * inv_dx;
            let donor = if slope > 0.0 { depth[i] } else { depth[j] };
            smallest = smallest.min(flow.max_step(slope, donor, self.cell_size));
        }
        smallest
    }

    /// Advance the film by `dt` seconds.
    ///
    /// Allocates nothing and branches on nothing per cell. See [`Self::max_step`] for
    /// what `dt` may be; exceeding it costs accuracy, not stability, because the
    /// positivity limiter holds regardless.
    ///
    /// A `dt` that is zero, negative or not finite is a no-op: the caller gets its grid
    /// back unchanged rather than an error, because this is called from a frame loop
    /// where there is nothing useful to do with an `Err`, and because a NaN `dt` would
    /// otherwise erase the whole map in one call.
    pub fn step(&mut self, flow: &FilmFlow, dt: f64) {
        if flow.yield_length > 0.0 {
            self.step_inner::<3, true>(flow, dt, 1.0);
        } else {
            self.step_inner::<3, false>(flow, dt, 1.0);
        }
    }

    /// The step with a substitutable flux exponent, for the test that shows what the
    /// cube is for. `gain` normalises a different exponent to the same flux at a
    /// reference depth, so the comparison is about the shape of the law and not its
    /// magnitude.
    fn step_inner<const N: u32, const YIELDS: bool>(
        &mut self,
        flow: &FilmFlow,
        dt: f64,
        gain: f64,
    ) {
        if !(dt > 0.0) || !dt.is_finite() {
            return;
        }

        let (w, h) = (self.width, self.height);
        let cells = w * h;
        let inv_dx = 1.0 / self.cell_size;
        let scale = dt * inv_dx;
        let (mobility, yield_length) = (flow.mobility, flow.yield_length);

        // Exact-length reborrows: this is what tells LLVM the indices below are in
        // range, and it is the difference between four vectorized loops and four loops
        // full of panic branches.
        let ground = &self.ground[..cells];
        let depth = &mut self.thickness[..cells];
        let flux_x = &mut self.flux_x[..=cells];
        let flux_y = &mut self.flux_y[..cells + w];
        let work = &mut self.work[..cells];

        // -- Pass 1: the flux across every face, from the donor cell's depth. ---------
        //
        // Upwinding is not a numerical nicety: the liquid that crosses a face is the
        // liquid on the uphill side of it, and averaging the two depths would let a dry
        // cell donate. The `if` is a select over two loaded values, not a jump.
        //
        // The two differences are taken separately — `(z_i - z_j) + (h_i - h_j)` and
        // never `(z_i + h_i) - (z_j + h_j)` — because bed elevations are metres and
        // depths are millimetres, and forming the sums first rounds the depth difference
        // away. That difference *is* the levelling term.
        // Taken a row at a time, not because the arithmetic differs but because
        // `row[x + 1]` against a slice of known length `w` is a bound LLVM can
        // discharge, and `flux_x[y * w + x + 1]` against a length of `w * h + 1` is
        // not. The difference is a vectorized loop against one carrying a bounds check
        // per cell.
        for y in 0..h {
            let r = y * w;
            let bed = &ground[r..r + w];
            let film = &depth[r..r + w];
            let across = &mut flux_x[r..r + w];
            for x in 0..w - 1 {
                let slope = ((bed[x] - bed[x + 1]) + (film[x] - film[x + 1])) * inv_dx;
                let donor = if slope > 0.0 { film[x] } else { film[x + 1] };
                across[x + 1] =
                    flux_kernel::<N, YIELDS>(mobility, yield_length, gain, slope, donor);
            }
        }
        for i in 0..cells - w {
            let j = i + w;
            let slope = ((ground[i] - ground[j]) + (depth[i] - depth[j])) * inv_dx;
            let donor = if slope > 0.0 { depth[i] } else { depth[j] };
            flux_y[j] = flux_kernel::<N, YIELDS>(mobility, yield_length, gain, slope, donor);
        }

        // -- Pass 2: no cell may give away more than it has. --------------------------
        //
        // Flat over the whole grid, because the padding means "the face above me" is
        // `flux_y[i]` for every cell including the first row. Outgoing faces are the
        // ones whose sign points away: `flux_x[i+1] > 0` to the right, `flux_x[i] < 0`
        // to the left.
        //
        // **Every face has exactly one donor**, decided by its sign, so the factor a
        // cell computes for its own outgoing faces can never conflict with another
        // cell's. That is what makes one pass enough and the result independent of
        // visiting order.
        for i in 0..cells {
            let leaving = flux_x[i + 1].max(0.0)
                + (-flux_x[i]).max(0.0)
                + flux_y[i + w].max(0.0)
                + (-flux_y[i]).max(0.0);
            let demanded = leaving * scale;
            // `demanded > depth[i] >= 0` means it is strictly positive, so the divide is
            // safe; the comparison is false for a zero demand and the factor is 1.
            work[i] = if demanded > depth[i] {
                depth[i] / demanded
            } else {
                1.0
            };
        }

        // -- Pass 3: apply each face's donor factor to that face. ---------------------
        //
        // Done to the face rather than to the cell, so both sides of it see the same
        // reduced number and the conservation in pass 4 still telescopes. `flux_x[cells]`
        // and the last row of `flux_y` are permanent zeros and are skipped.
        for i in 0..cells - 1 {
            let q = flux_x[i + 1];
            flux_x[i + 1] = q * if q > 0.0 { work[i] } else { work[i + 1] };
        }
        for i in 0..cells - w {
            let q = flux_y[i + w];
            flux_y[i + w] = q * if q > 0.0 { work[i] } else { work[i + w] };
        }

        // -- Pass 4: the conservation law itself. -------------------------------------
        //
        // Each face appears once with each sign, which is what makes the total conserved
        // rather than approximately conserved. Flat, branch-free, four loads and a
        // fused-multiply-add per cell.
        for i in 0..cells {
            let net = (flux_x[i] - flux_x[i + 1]) + (flux_y[i] - flux_y[i + w]);
            // `.max(0.0)` catches the rounding-level negatives the limiter cannot, and
            // is a no-op for any non-negative depth — including the flat case, where
            // `net` is exactly zero and the depth comes back bit-identical.
            depth[i] = (depth[i] + net * scale).max(0.0);
        }
    }

    #[inline]
    fn check_depth(depth: f64) -> Result<(), PhysicsError> {
        validate_finite(depth, "depth")?;
        if depth < 0.0 {
            return Err(PhysicsError::CalculationError(format!(
                "film depth must be non-negative, got {}",
                depth
            )));
        }
        Ok(())
    }

    #[inline]
    fn index(&self, x: usize, y: usize) -> Result<usize, PhysicsError> {
        if x >= self.width || y >= self.height {
            return Err(PhysicsError::CalculationError(format!(
                "cell ({}, {}) is outside a {}x{} grid",
                x, y, self.width, self.height
            )));
        }
        Ok(y * self.width + x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_float_eq;
    use crate::fluid_dynamics::{
        blood_apparent_viscosity, puddle_depth, BLOOD_CASSON_VISCOSITY, BLOOD_DENSITY,
        BLOOD_REFERENCE_SHEAR_RATE, BLOOD_SURFACE_TENSION, BLOOD_YIELD_STRESS,
    };

    const G: f64 = 9.81;

    fn blood() -> FilmFlow {
        FilmFlow::new(&Fluid::blood(), G).unwrap()
    }

    // ---------------------------------------------------------------- the law

    /// **The claim the whole module rests on.** Double the depth, get eight times the
    /// flux — and the expected factor is written as `2³` rather than as `8.0`, so it
    /// tracks the exponent instead of restating it.
    #[test]
    fn flux_is_cubic_in_thickness() {
        let flow = blood();
        let slope = 0.1;
        let thin = flow.flux(slope, 0.001);
        let thick = flow.flux(slope, 0.002);
        assert_float_eq(thick / thin, 2f64.powi(3), 1e-12, Some("flux must go as h^3"));

        for depth in [1e-4, 5e-4, 2e-3, 6e-3] {
            let doubled = flow.flux(slope, depth * 2.0) / flow.flux(slope, depth);
            assert_float_eq(doubled, 8.0, 1e-9, Some("cubic at every depth"));
        }
    }

    #[test]
    fn flux_is_linear_in_slope_and_signed_by_it() {
        let flow = blood();
        let depth = 0.002;
        let gentle = flow.flux(0.05, depth);
        let steep = flow.flux(0.15, depth);
        assert_float_eq(steep / gentle, 3.0, 1e-12, Some("flux is linear in slope"));
        assert_float_eq(
            flow.flux(-0.05, depth),
            -gentle,
            1e-18,
            Some("uphill slope, flux the other way, same size"),
        );
    }

    /// The front of a spill outruns the liquid in it by exactly three, which is what
    /// makes a leading edge sharpen instead of diffusing.
    #[test]
    fn a_depth_change_travels_three_times_faster_than_the_liquid() {
        let flow = blood();
        let (slope, depth) = (0.12, 0.0015);
        assert_float_eq(
            flow.wave_speed(slope, depth),
            3.0 * flow.velocity(slope, depth),
            1e-15,
            Some("kinematic wave speed is dq/dh = 3u for a Newtonian film"),
        );
    }

    /// Nothing here may turn a NaN into a plausible number, and nothing may pass one on.
    /// Zero is the documented sink.
    #[test]
    fn nan_and_degenerate_inputs_produce_exactly_zero_flux() {
        let flow = blood();
        assert_eq!(flow.flux(f64::NAN, 0.002), 0.0);
        assert_eq!(flow.flux(0.1, f64::NAN), 0.0);
        assert_eq!(flow.flux(0.0, 0.002), 0.0);
        assert_eq!(flow.flux(0.1, 0.0), 0.0);
        assert_eq!(flow.velocity(0.1, 0.0), 0.0);
        assert_eq!(flow.wave_speed(f64::NAN, 0.002), 0.0);
    }

    /// The batch form and the scalar form must be the *same* law, bit for bit, or the
    /// WGSL transcription is written against a function nobody runs.
    #[test]
    fn the_batch_form_agrees_with_the_scalar_form_exactly() {
        let slopes = [0.0, 0.1, -0.25, 1.0, -0.02, f64::NAN, 0.3, 0.05];
        let depths = [0.002, 0.0, 0.001, 0.004, f64::NAN, 0.002, 0.0035, 1e-9];
        for flow in [blood(), blood().with_yield_stress(BLOOD_YIELD_STRESS).unwrap()] {
            let mut out = [0.0; 8];
            flow.flux_batch(&slopes, &depths, &mut out).unwrap();
            for i in 0..8 {
                assert_eq!(
                    out[i],
                    flow.flux(slopes[i], depths[i]),
                    "batch and scalar disagree at {}",
                    i
                );
            }
        }
        // The length check is at the batch boundary, once, and it is a real check.
        let mut short = [0.0; 3];
        assert!(blood().flux_batch(&slopes, &depths, &mut short).is_err());
        assert!(blood().flux_batch(&slopes[..2], &depths, &mut out_of(8)).is_err());
    }

    fn out_of(n: usize) -> Vec<f64> {
        vec![0.0; n]
    }

    /// A public `Fluid` field means `Fluid::new`'s validation can be walked around.
    /// This is the boundary that has to notice, because it is the only one on the path.
    #[test]
    fn a_fluid_built_around_its_own_constructor_is_rejected_here() {
        assert!(FilmFlow::new(&Fluid { density: -1.0, viscosity: 0.004 }, G).is_err());
        assert!(FilmFlow::new(&Fluid { density: 1060.0, viscosity: 0.0 }, G).is_err());
        assert!(FilmFlow::new(&Fluid { density: 1060.0, viscosity: f64::NAN }, G).is_err());
        assert!(FilmFlow::new(&Fluid::blood(), 0.0).is_err());
        assert!(FilmFlow::new(&Fluid::blood(), f64::INFINITY).is_err());
        assert!(blood().with_yield_stress(-1e-3).is_err());
        assert!(blood().with_yield_stress(f64::NAN).is_err());
    }

    // ------------------------------------------------------- the fluids differ

    /// Blood runs and honey creeps, **and by how much comes from the fluids
    /// themselves** rather than a number typed into this test. The ratio of
    /// depth-averaged velocities at equal depth and slope is `(ρ_b/μ_b)/(ρ_h/μ_h)`, so
    /// the expected value is built from the two constructors.
    #[test]
    fn blood_runs_where_honey_creeps() {
        let (blood_fluid, honey_fluid) = (Fluid::blood(), Fluid::honey());
        let blood = FilmFlow::new(&blood_fluid, G).unwrap();
        let honey = FilmFlow::new(&honey_fluid, G).unwrap();

        let (slope, depth) = (0.1, 0.002);
        let expected = (blood_fluid.density / blood_fluid.viscosity)
            / (honey_fluid.density / honey_fluid.viscosity);
        let measured = blood.velocity(slope, depth) / honey.velocity(slope, depth);
        assert_float_eq(measured, expected, 1e-9, Some("velocity ratio is rho/mu"));

        // The qualitative claim, stated separately so a change of constants that
        // reverses it fails loudly: honey does not run off a battlefield.
        assert!(measured > 1000.0, "blood should outrun honey by three orders");
        assert!(
            honey.velocity(slope, depth) < 1e-3,
            "honey at 2 mm on a 1-in-10 slope should creep at under a mm/s, got {}",
            honey.velocity(slope, depth)
        );
        // And a millimetre of blood on the same slope runs at a walking pace.
        let running = blood.velocity(slope, 0.001);
        assert!(running > 0.05 && running < 1.0, "blood ran at {} m/s", running);
    }

    /// Water sheets faster than blood at the same depth. Two effects fight — blood is
    /// denser (pushes harder) and more viscous (resists more) — and viscosity wins by
    /// nearly four to one, which is why a water spill covers ground a blood spill does
    /// not reach.
    #[test]
    fn water_outruns_blood() {
        let water = FilmFlow::new(&Fluid::water(), G).unwrap();
        let blood = blood();
        let (slope, depth) = (0.1, 0.002);
        assert!(water.velocity(slope, depth) > 3.0 * blood.velocity(slope, depth));
    }

    // -------------------------------------------------- blood is not Newtonian

    /// `Fluid::blood`'s viscosity is not a table value: it is the Casson fit at a
    /// declared shear rate. This is the assertion that keeps the two from drifting, and
    /// the reason the constructor is allowed to claim a source.
    #[test]
    fn newtonian_blood_is_the_casson_fit_at_its_stated_shear_rate() {
        let quoted = blood_apparent_viscosity(BLOOD_REFERENCE_SHEAR_RATE).unwrap();
        assert_float_eq(
            Fluid::blood().viscosity,
            quoted,
            1e-5,
            Some("Fluid::blood().viscosity must be blood_apparent_viscosity(300)"),
        );
    }

    #[test]
    fn blood_thins_as_it_is_sheared_and_settles_on_the_casson_asymptote() {
        let mut previous = f64::INFINITY;
        for rate in [0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0] {
            let mu = blood_apparent_viscosity(rate).unwrap();
            assert!(mu < previous, "apparent viscosity must fall with shear rate");
            previous = mu;
        }
        // Below 1 s^-1 it is an order of magnitude off the high-shear figure, which is
        // exactly why `Fluid::blood()` may not be used there.
        assert!(blood_apparent_viscosity(0.1).unwrap() > 10.0 * Fluid::blood().viscosity);
        // And above a few thousand it has all but reached the asymptote.
        assert!((blood_apparent_viscosity(1e5).unwrap() - BLOOD_CASSON_VISCOSITY).abs() < 1e-4);
        assert!(blood_apparent_viscosity(0.0).is_err());
        assert!(blood_apparent_viscosity(f64::NAN).is_err());
        assert!(blood_apparent_viscosity(-1.0).is_err());
    }

    /// **The test that licenses the whole approximation.** Over every film depth and
    /// slope this module is meant for, the Casson apparent viscosity at the film's own
    /// shear rate stays within 16% of the single Newtonian number `Fluid::blood()`
    /// carries. If a change to either constant breaks that, the constructor's claim
    /// stops being true and this fails.
    #[test]
    fn the_newtonian_approximation_holds_across_the_film_regime() {
        let flow = blood();
        let quoted = Fluid::blood().viscosity;
        let mut worst = 0.0f64;
        for &depth in &[0.0005, 0.001, 0.002, 0.003, 0.005] {
            for &slope in &[0.05, 0.1, 0.2, 0.3] {
                let rate = flow.shear_rate(slope, depth);
                assert!(rate > 1.0, "a running film is never at low shear, got {}", rate);
                let error = (blood_apparent_viscosity(rate).unwrap() - quoted).abs() / quoted;
                worst = worst.max(error);
            }
        }
        assert!(worst < 0.16, "worst-case Newtonian error was {:.1}%", worst * 100.0);
    }

    /// Blood's yield stress is real but tiny, and the two things that stop a spill act
    /// at scales three orders apart. Saying so in a test stops anyone reaching for the
    /// wrong one.
    #[test]
    fn yield_stress_arrests_a_film_far_thinner_than_surface_tension_holds_a_pool() {
        let flow = blood().with_yield_stress(BLOOD_YIELD_STRESS).unwrap();
        let arrest = flow.arrest_thickness(0.1);
        assert!(arrest > 1e-6 && arrest < 1e-5, "arrest thickness was {} m", arrest);

        let pool =
            puddle_depth(BLOOD_SURFACE_TENSION, BLOOD_DENSITY, G, 80f64.to_radians()).unwrap();
        assert!(pool > 100.0 * arrest, "a pool's depth is set by capillarity, not yield");

        // Below the arrest thickness nothing moves; above it the flux climbs
        // continuously from zero rather than switching on.
        assert_eq!(flow.flux(0.1, arrest * 0.99), 0.0);
        assert!(flow.flux(0.1, arrest * 1.01) > 0.0);
        assert!(flow.flux(0.1, arrest * 1.01) < 1e-4 * flow.flux(0.1, arrest * 10.0));

        // On level ground a yield-stress fluid never runs, at any depth.
        assert_eq!(flow.arrest_thickness(0.0), f64::INFINITY);
        assert_eq!(flow.flux(0.0, 1.0), 0.0);

        // Deep enough and the plug correction all but vanishes, which is what makes the
        // Newtonian `Fluid::blood()` an honest thing to hand a film solver. The size of
        // the correction is `1.5·X` to leading order, `X = arrest/h`, so at 5 mm it is
        // about 0.15% — and it is written that way rather than as a number, so the
        // bound tracks the constants instead of restating them.
        let deep = 0.005;
        let newtonian = blood().flux(0.1, deep);
        let correction = (newtonian - flow.flux(0.1, deep)) / newtonian;
        let leading_order = 1.5 * arrest / deep;
        assert!(correction > 0.0, "a yield stress can only reduce the flux");
        assert_float_eq(
            correction,
            leading_order,
            1e-5,
            Some("the plug correction is 1.5*h_y/h to leading order"),
        );
        assert!(correction < 0.002, "at 5 mm the correction is {:.4}%", correction * 100.0);
    }

    // ------------------------------------------------------------- the solver

    /// A grid tilted along +x, with the ground dropping `grade` metres per metre.
    fn sloped_grid(width: usize, height: usize, cell: f64, grade: f64) -> FilmGrid {
        let mut grid = FilmGrid::new(width, height, cell).unwrap();
        let bed: Vec<f64> = (0..width * height)
            .map(|i| -grade * (i % width) as f64 * cell)
            .collect();
        grid.set_ground_from(&bed).unwrap();
        grid
    }

    #[test]
    fn setters_refuse_what_would_poison_the_grid() {
        let mut grid = FilmGrid::new(4, 4, 0.2).unwrap();
        assert!(grid.set_thickness(0, 0, f64::NAN).is_err());
        assert!(grid.set_thickness(0, 0, -0.001).is_err());
        assert!(grid.set_thickness(4, 0, 0.001).is_err());
        assert!(grid.set_ground(0, 0, f64::INFINITY).is_err());
        assert!(grid.add_thickness(0, 0, f64::NAN).is_err());
        assert!(grid.thickness_at(0, 4).is_err());
        assert!(FilmGrid::new(0, 4, 0.2).is_err());
        assert!(FilmGrid::new(4, 4, 0.0).is_err());
        assert!(FilmGrid::new(4, 4, f64::NAN).is_err());
        assert!(FilmGrid::new(usize::MAX, usize::MAX, 0.2).is_err());

        // A rejected bulk terrain leaves the bed untouched rather than half written.
        let mut bed = vec![1.0; 16];
        bed[9] = f64::NAN;
        assert!(grid.set_ground_from(&bed).is_err());
        assert!(grid.ground().iter().all(|&z| z == 0.0));
        assert!(grid.set_ground_from(&[0.0; 4]).is_err());
    }

    /// Liquid is neither created nor destroyed by a step, on a slope, with the film
    /// actively running off the interior and piling against the boundary.
    #[test]
    fn mass_is_conserved_across_a_step() {
        let flow = blood();
        let mut grid = sloped_grid(24, 8, 0.2, 0.1);
        for y in 2..6 {
            for x in 2..6 {
                grid.set_thickness(x, y, 0.003).unwrap();
            }
        }
        let before = grid.total_volume();
        assert!(before > 0.0);

        let dt = grid.max_step(&flow) * 0.4;
        for _ in 0..400 {
            grid.step(&flow, dt);
        }
        let after = grid.total_volume();
        assert!(
            (after - before).abs() / before < 1e-12,
            "volume drifted from {} to {}",
            before,
            after
        );
    }

    /// Zero slope, zero flux, and **exactly** zero drift — asserted on the bits, not
    /// with an epsilon. A film solver that creeps on still ground never stops
    /// re-uploading its buffer.
    #[test]
    fn a_level_uniform_film_does_not_move_at_all() {
        let flow = blood();
        let mut grid = FilmGrid::new(16, 16, 0.2).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                grid.set_thickness(x, y, 0.002).unwrap();
            }
        }
        let before = grid.thickness().to_vec();
        for _ in 0..1000 {
            grid.step(&flow, 1.0 / 60.0);
        }
        assert_eq!(grid.thickness(), &before[..], "a level film must not drift");

        // And dry ground on a slope stays dry, which is the case that dominates a map.
        let mut dry = sloped_grid(16, 16, 0.2, 0.3);
        let before = dry.thickness().to_vec();
        for _ in 0..100 {
            dry.step(&flow, 1.0 / 60.0);
        }
        assert_eq!(dry.thickness(), &before[..]);
    }

    /// A cell may never owe more liquid than it has, at any timestep — including one far
    /// above the CFL limit, where the answer is wrong but must not be negative.
    #[test]
    fn no_timestep_can_drive_a_cell_negative() {
        let flow = blood();
        let mut grid = sloped_grid(20, 4, 0.2, 0.4);
        for y in 0..4 {
            grid.set_thickness(3, y, 0.02).unwrap();
        }
        let before = grid.total_volume();
        let reckless = grid.max_step(&flow) * 100.0;
        for _ in 0..50 {
            grid.step(&flow, reckless);
            for &h in grid.thickness() {
                assert!(h >= 0.0 && h.is_finite(), "depth went to {}", h);
            }
        }
        // Wrong, but still conservative: the limiter throws nothing away.
        assert!((grid.total_volume() - before).abs() / before < 1e-12);
    }

    /// On level ground a pool spreads and levels rather than sitting in a column: the
    /// free-surface gradient does that, without a second law.
    #[test]
    fn a_pool_on_level_ground_spreads_and_levels() {
        let flow = blood();
        let mut grid = FilmGrid::new(21, 21, 0.05).unwrap();
        grid.set_thickness(10, 10, 0.01).unwrap();
        let before = grid.total_volume();
        let dt = grid.max_step(&flow) * 0.2;
        for _ in 0..2000 {
            grid.step(&flow, dt);
        }
        assert!(grid.thickness_at(10, 10).unwrap() < 0.01, "the column must slump");
        assert!(grid.thickness_at(12, 10).unwrap() > 0.0, "and reach its neighbours");
        assert!((grid.total_volume() - before).abs() / before < 1e-12);
    }

    /// **The test that justifies the exponent.**
    ///
    /// A sheet released across a slope with a 3% transverse ripple in its depth and
    /// nothing else different between the rows. Under the cubic law the thicker rows
    /// run faster — front speed goes as `h²` — pull ahead, and the sheet stops
    /// advancing as one line: the spread in how far each row has got *grows* from zero.
    /// That differential advance is what a rivulet is before it is a channel.
    ///
    /// The comparison is the same grid, the same solver, **the same code path**, with
    /// the exponent changed to 1 and a gain that makes the two laws give identical flux
    /// at the mean depth — so it is not a slower fluid, it is the same fluid with a
    /// different law. Under it, every row advances at the same speed regardless of
    /// depth: the perturbation is still there, and it never becomes anything.
    ///
    /// # Why they are compared at equal *distance* rather than equal *time*
    ///
    /// A cubic front decelerates as the sheet drains and thins; a linear one does not,
    /// because its speed does not depend on depth at all. Run for the same number of
    /// steps, the linear film simply reaches the bottom of the grid and stops, and any
    /// comparison after that is with a puddle against a wall. So each is run until its
    /// mean position has travelled the same metre of ground, and the spreads are
    /// compared there. It is also the fairer question: after the same distance, how
    /// much has the front come apart?
    ///
    /// The measure is the depth-weighted centroid of each row, in metres, which is
    /// continuous — a threshold crossing quantised to whole cells cannot see a
    /// half-cell difference, and half a cell is what a 3% ripple earns over a metre.
    /// The *ratio* is what matters, and it is around two orders of magnitude.
    #[test]
    fn the_cube_is_what_makes_rivulets() {
        const W: usize = 96;
        const H: usize = 24;
        const CELL: f64 = 0.05;
        const GRADE: f64 = 0.15;
        const MEAN: f64 = 0.002;
        const RIPPLE: f64 = 0.03;
        /// How far the sheet's mean position must travel before the two are compared.
        const TRAVEL: f64 = 1.0;
        /// Enough steps to get there under either law, with a wide margin; a cap only
        /// so a regression that stops the flow fails as a failure rather than a hang.
        const CAP: u32 = 50_000;

        let flow = blood();

        // Depth-weighted centre of each row, in metres downslope.
        let centroids = |grid: &FilmGrid| -> Vec<f64> {
            (0..H)
                .map(|y| {
                    let mut moment = 0.0;
                    let mut mass = 0.0;
                    for x in 0..W {
                        let depth = grid.thickness_at(x, y).unwrap();
                        moment += x as f64 * CELL * depth;
                        mass += depth;
                    }
                    moment / mass
                })
                .collect()
        };
        let spread = |c: &[f64]| {
            c.iter().cloned().fold(f64::MIN, f64::max) - c.iter().cloned().fold(f64::MAX, f64::min)
        };
        let mean = |c: &[f64]| c.iter().sum::<f64>() / c.len() as f64;

        let build = || {
            let mut grid = sloped_grid(W, H, CELL, GRADE);
            for y in 0..H {
                // Two full transverse cycles, so the ripple is smooth and periodic and
                // no row is special.
                let phase = (y as f64 / H as f64) * 4.0 * std::f64::consts::PI;
                let depth = MEAN * (1.0 + RIPPLE * phase.sin());
                for x in 0..8 {
                    grid.set_thickness(x, y, depth).unwrap();
                }
            }
            grid
        };

        // The same flux at the mean depth: gain·m·s·h = m·s·h³ when h = MEAN.
        let gain = MEAN * MEAN;
        let dt = build().max_step(&flow) * 0.25;
        assert!(spread(&centroids(&build())) < 1e-12, "both sheets start level");

        let start = mean(&centroids(&build()));
        let run = |cubic: bool| -> (f64, u32) {
            let mut grid = build();
            let mut steps = 0;
            while mean(&centroids(&grid)) - start < TRAVEL && steps < CAP {
                if cubic {
                    grid.step(&flow, dt);
                } else {
                    grid.step_inner::<1, false>(&flow, dt, gain);
                }
                steps += 1;
            }
            (spread(&centroids(&grid)), steps)
        };

        let (cubic_spread, cubic_steps) = run(true);
        let (linear_spread, linear_steps) = run(false);

        assert!(cubic_steps < CAP, "the cubic sheet never travelled a metre");
        assert!(linear_steps < CAP, "the linear sheet never travelled a metre");

        // Half a cell of front separation, out of a 3% ripple, over one metre.
        assert!(
            cubic_spread > 0.4 * CELL,
            "a cubic law must pull the front apart; spread was {:.4} m ({:.2} cells)",
            cubic_spread,
            cubic_spread / CELL
        );
        // And the linear law must leave it where it found it.
        assert!(
            linear_spread < 0.05 * CELL,
            "a linear law must not finger: front speed does not depend on depth, yet \
             the spread was {:.5} m ({:.3} cells)",
            linear_spread,
            linear_spread / CELL
        );
        assert!(
            cubic_spread > 20.0 * linear_spread,
            "cubic {:.5} m vs linear {:.5} m — the exponent is not doing anything",
            cubic_spread,
            linear_spread
        );
    }


    /// The mechanism behind the test above, isolated from the solver: a thicker patch
    /// does not merely carry more liquid, it carries **disproportionately** more, and
    /// the disproportion is the exponent minus one in the front speed.
    #[test]
    fn a_ten_percent_thicker_patch_runs_twenty_one_percent_faster() {
        let flow = blood();
        let slope = 0.1;
        let thin = flow.velocity(slope, 0.002);
        let thick = flow.velocity(slope, 0.002 * 1.1);
        assert_float_eq(thick / thin, 1.1f64.powi(2), 1e-12, Some("front speed goes as h^2"));
        assert_float_eq(
            flow.flux(slope, 0.002 * 1.1) / flow.flux(slope, 0.002),
            1.1f64.powi(3),
            1e-12,
            Some("flux goes as h^3"),
        );
    }

    /// The stability limit is a real limit: at a quarter of it a spill on a steep slope
    /// stays smooth, and the deepest cell is what sets it.
    #[test]
    fn the_stable_step_keeps_a_steep_spill_smooth() {
        let flow = blood();
        let mut grid = sloped_grid(48, 4, 0.1, 0.3);
        for y in 0..4 {
            for x in 0..4 {
                grid.set_thickness(x, y, 0.004).unwrap();
            }
        }
        let dt = grid.max_step(&flow) * 0.25;
        assert!(dt.is_finite() && dt > 0.0);
        for _ in 0..3000 {
            grid.step(&flow, dt);
            for &h in grid.thickness() {
                assert!(h.is_finite() && h >= 0.0 && h < 0.02, "depth blew up to {}", h);
            }
        }
        // Deeper liquid means a smaller allowable step, quadratically.
        let shallow = flow.max_step(0.3, 0.002, 0.1);
        let deep = flow.max_step(0.3, 0.004, 0.1);
        assert_float_eq(shallow / deep, 4.0, 1e-12, Some("dt limit goes as 1/h^2"));
        assert_eq!(flow.max_step(0.0, 0.002, 0.1), f64::INFINITY);
    }

    /// The stepper refuses a `dt` it cannot use rather than doing something with it.
    #[test]
    fn a_degenerate_timestep_is_a_no_op() {
        let flow = blood();
        let mut grid = sloped_grid(8, 2, 0.2, 0.2);
        grid.set_thickness(1, 0, 0.003).unwrap();
        let before = grid.thickness().to_vec();
        for dt in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            grid.step(&flow, dt);
            assert_eq!(grid.thickness(), &before[..], "dt = {} changed the grid", dt);
        }
    }

    /// Liquid runs downhill, not up. Obvious, and the sign of a slope is exactly the
    /// kind of thing that is wrong for a month without anyone noticing.
    #[test]
    fn a_spill_runs_downhill() {
        let flow = blood();
        let mut grid = sloped_grid(32, 3, 0.1, 0.2);
        for y in 0..3 {
            grid.set_thickness(8, y, 0.004).unwrap();
        }
        let dt = grid.max_step(&flow) * 0.25;
        for _ in 0..2000 {
            grid.step(&flow, dt);
        }
        let downhill: f64 = (9..32).map(|x| grid.thickness_at(x, 1).unwrap()).sum();
        let uphill: f64 = (0..8).map(|x| grid.thickness_at(x, 1).unwrap()).sum();
        assert!(downhill > 100.0 * uphill, "down {} up {}", downhill, uphill);
    }

    /// A one-cell-wide grid has no faces in one direction, and the padded buffers must
    /// still be indexable. The degenerate shapes are where an off-by-one in the padding
    /// would show up.
    #[test]
    fn degenerate_grid_shapes_still_step() {
        let flow = blood();
        for (w, h) in [(1, 1), (1, 8), (8, 1), (2, 2)] {
            let mut grid = sloped_grid(w, h, 0.1, 0.2);
            grid.set_thickness(0, 0, 0.003).unwrap();
            let before = grid.total_volume();
            for _ in 0..100 {
                grid.step(&flow, 1e-4);
            }
            assert!((grid.total_volume() - before).abs() / before < 1e-12, "{}x{}", w, h);
            assert!(grid.thickness().iter().all(|h| h.is_finite() && *h >= 0.0));
        }
    }
}
