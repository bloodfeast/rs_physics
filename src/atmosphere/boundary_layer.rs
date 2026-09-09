//! # The atmospheric boundary layer, in closed form
//!
//! Wind near the ground: how its speed varies with height, how it accelerates over a
//! rise, and how far into a plant canopy it reaches. Every result here is an algebraic
//! expression, most of them a single line, each with a citation and a stated range
//! outside which it is not a model of anything.
//!
//! ## This is deliberately not a solver
//!
//! [`crate::fluid_dynamics`] already holds an Eulerian grid with iterative pressure
//! projection, and it is the wrong instrument for this question. The question is *what
//! is the wind at this point on this hillside* — asked tens of thousands of times, once,
//! at bake time, for a value that then never changes. A grid answers a different
//! question (what does this flow field do next) at a cost that scales with the volume
//! rather than with the number of points you asked about, and it needs boundary
//! conditions that are themselves the answer you wanted.
//!
//! The closed forms below are what the wind-engineering literature uses for exactly this
//! reason. They are `f64` in, `f64` out. Nothing here allocates, nothing here iterates,
//! and the per-point calls are infallible by construction — see [`WindProfile`].
//!
//! ## The three results
//!
//! **1. Speed against height — the logarithmic profile.** In a neutrally stratified
//! surface layer, momentum flux is constant with height and mixing length grows linearly
//! with it, which integrates to `u(z) = (u*/κ)·ln((z−d)/z₀)`. That is [`log_profile`],
//! and the surface it is over is a [`Surface`]: a roughness length `z₀` and a
//! displacement height `d`. It is the reason grass and canopy move differently — not
//! because they are different plants, but because their tips are at different heights in
//! the same profile, and over grass the profile is steep near the ground while over
//! forest the whole thing is lifted by `d`.
//!
//! **2. Speed-up over a rise — Jackson–Hunt.** Flow crossing a hill has to fit through a
//! layer the hill has thinned, so it accelerates; the linearised solution puts the
//! fractional speed-up at the crest at `K·H/L`. That is [`crest_speedup`], and its
//! validity limit is [`LINEARISATION_SLOPE_LIMIT`] — past which the flow separates and
//! the linearisation is describing a flow that does not occur.
//!
//! **3. Damping inside a canopy — Cionco.** Below the top of standing vegetation the
//! profile is not logarithmic at all; drag on the foliage makes it decay exponentially
//! toward the ground. That is [`canopy_profile`], and it is the honest form of
//! "sheltering" for anything standing among other plants.
//!
//! ## What is deliberately absent
//!
//! **Wakes behind solid or porous obstacles.** There is no defensible closed form. The
//! published shelterbelt results (Hagen & Skidmore 1971; Wang & Takle 1997; Vigiak et
//! al. 2003) are empirical curve fits with a different fit per porosity, per barrier
//! aspect ratio and per approach profile, and they disagree with each other by tens of
//! per cent in the near wake — which is the region anyone asking the question cares
//! about. A single-parameter curve that looked plausible would be an invented number
//! wearing a citation, so there is not one here.
//!
//! **Separated lee flow.** The lee side of a *gentle* rise is genuinely covered by the
//! same Jackson–Hunt expression with a negative slope, and that is supported. Past
//! [`LINEARISATION_SLOPE_LIMIT`] the flow separates into a recirculating bubble whose
//! length depends on the downwind slope, the approach turbulence and the crest curvature.
//! The functions here clamp at the limit rather than extrapolate, so a steeper hill
//! returns the steepest answer the theory has, not a bigger one it does not have.
//!
//! **Stability.** Everything here is the neutral case. Stable and unstable
//! stratification change the profile through the Monin–Obukhov length, which needs a
//! surface heat flux the caller almost certainly does not have. Adding it would be a
//! real extension; guessing an Obukhov length would not.

use crate::atmosphere::Air;
use crate::utils::PhysicsError;

/// The von Kármán constant, dimensionless.
///
/// The proportionality between mixing length and height in the surface layer. Measured
/// rather than derived; published values run 0.38–0.41 and 0.40 is the conventional
/// choice, used here and in ESDU 85020.
pub const VON_KARMAN: f64 = 0.40;

/// The gradient past which the Jackson–Hunt linearisation stops describing the flow.
///
/// `H/L = 0.3`, about 17°. Below it the flow stays attached over the hill and the linear
/// theory holds; above it the boundary layer separates on the lee side and a linear
/// speed-up is not what happens (Taylor, Mason & Bradley 1987; ESDU 91043).
///
/// **The clamp at this value is part of the physics, not a safety net.** A result used
/// outside its own validity range is worse than no result — it is a wrong answer with the
/// authority of an equation behind it. Clamping means a 45° scarp is given the answer for
/// the steepest slope the theory covers, and it also fixes the floor on the sheltered
/// side: a full lee gets `1 − 2 × 0.3 = 0.4` of the open-ground speed and nothing gets
/// less.
pub const LINEARISATION_SLOPE_LIMIT: f64 = 0.3;

/// Decay rate of the crest speed-up with height, in units of `1/L`.
///
/// **This one is an empirical fit, not a theory result**, and is flagged as such because
/// the coefficients in [`HillForm`] are not. Jackson–Hunt gives the speed-up at the top
/// of the inner layer; its decay above that is quoted as `exp(−A·z/L)` with published `A`
/// between about 2.5 and 3.5 depending on hill form and on how `L` was measured
/// (Taylor & Lee 1984; Lemelin, Surry & Davenport 1988). 3.0 is the middle of that range.
const SPEEDUP_HEIGHT_DECAY: f64 = 3.0;

// ---------------------------------------------------------------------------------
// The ground
// ---------------------------------------------------------------------------------

/// What the wind is blowing over: a roughness length and a displacement height.
///
/// # Why these are one type and not two `f64`
///
/// Both are lengths in metres, so no dimensional newtype would stop them being swapped,
/// and both appear in the same logarithm where getting them the wrong way round produces
/// a plausible-looking number rather than an error. More to the point, `z₀` is the one
/// quantity in this module a caller is most likely to invent: it is not a height anyone
/// can measure with a tape, it comes from a *published classification table*, and the
/// right way to get one is to name the terrain rather than to type a decimal.
///
/// So the constants below are the interface, the table is in the type, and `z₀ > 0` is
/// checked once here rather than guarded at each of the four places that take its
/// logarithm.
///
/// # The numbers
///
/// Roughness lengths follow the Davenport classification as revised by Wieringa (1992),
/// which is also what ESDU 85020 and Eurocode EN 1991-1-4 Table 4.1 use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Surface {
    /// Aerodynamic roughness length `z₀`, in metres. Strictly positive.
    roughness_m: f64,
    /// Zero-plane displacement `d`, in metres. Non-negative.
    displacement_m: f64,
}

impl Surface {
    /// Open sea, ice, tidal flat. `z₀ = 0.0002 m`.
    pub const OPEN_SEA: Surface = Surface { roughness_m: 0.0002, displacement_m: 0.0 };
    /// Snow-covered flat ground with no vegetation showing. `z₀ = 0.005 m`.
    pub const SNOW: Surface = Surface { roughness_m: 0.005, displacement_m: 0.0 };
    /// Open flat terrain: mown grass, bare soil, airfield. `z₀ = 0.03 m`.
    ///
    /// The reference class. Most quoted "open country" wind figures are for this.
    pub const OPEN_GRASS: Surface = Surface { roughness_m: 0.03, displacement_m: 0.0 };
    /// Low crops, occasional large obstacles well separated. `z₀ = 0.10 m`.
    pub const LOW_CROPS: Surface = Surface { roughness_m: 0.10, displacement_m: 0.0 };
    /// High crops, scattered obstacles at 12–15 obstacle heights. `z₀ = 0.25 m`.
    pub const HIGH_CROPS: Surface = Surface { roughness_m: 0.25, displacement_m: 0.0 };
    /// Parkland, bushes, scrub; obstacles at about 8 obstacle heights. `z₀ = 0.5 m`.
    pub const SCRUB: Surface = Surface { roughness_m: 0.5, displacement_m: 0.0 };
    /// Regular cover by large obstacles: mature forest, low-rise suburb. `z₀ = 1.0 m`.
    ///
    /// Note the displacement is zero here: this is the class for *flow over* a forest
    /// measured from the ground datum a long way away. For a profile near or inside a
    /// specific canopy, use [`Surface::canopy`], which sets `d` as well.
    pub const FOREST: Surface = Surface { roughness_m: 1.0, displacement_m: 0.0 };
    /// City centre with tall and irregular buildings. `z₀ = 2.0 m`.
    pub const CITY: Surface = Surface { roughness_m: 2.0, displacement_m: 0.0 };

    /// A surface from a measured roughness length and displacement height.
    ///
    /// # Arguments
    ///
    /// * `roughness_m` — aerodynamic roughness length `z₀` in metres. Strictly positive.
    /// * `displacement_m` — zero-plane displacement `d` in metres. Non-negative.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::CalculationError`] if either is non-finite, if `z₀ ≤ 0` (the log
    /// profile takes `ln(z/z₀)`, so zero roughness is an infinity and not a smooth
    /// surface), or if `d < 0`.
    pub fn new(roughness_m: f64, displacement_m: f64) -> Result<Surface, PhysicsError> {
        if !roughness_m.is_finite() || roughness_m <= 0.0 {
            return Err(PhysicsError::CalculationError(format!(
                "roughness length z0 must be finite and above zero, got {roughness_m}; \
                 a perfectly smooth surface has no logarithmic profile"
            )));
        }
        if !displacement_m.is_finite() || displacement_m < 0.0 {
            return Err(PhysicsError::CalculationError(format!(
                "displacement height d must be finite and non-negative, got {displacement_m}"
            )));
        }
        Ok(Surface { roughness_m, displacement_m })
    }

    /// The surface presented by standing vegetation of a given height.
    ///
    /// Uses the standard rules of thumb `z₀ ≈ 0.1 h` and `d ≈ 0.7 h` (Brutsaert,
    /// *Evaporation into the Atmosphere*, 1982, §5.2; the same ratios appear in Stull
    /// 1988 §9.7). They hold to within a factor of about 1.5 for closed canopies of
    /// uniform height, which is as well as anything without a site measurement does.
    ///
    /// The displacement is what makes this different from picking a `z₀` off the table:
    /// over a 20 m forest the wind profile behaves as though the ground were at 14 m, so
    /// a plant at 2 m is not merely low in the profile, it is *below the profile
    /// entirely* — which is the whole reason [`canopy_profile`] exists.
    ///
    /// # Arguments
    ///
    /// * `height_m` — mean height of the canopy top above ground, in metres.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::CalculationError`] if the height is not finite and positive.
    pub fn canopy(height_m: f64) -> Result<Surface, PhysicsError> {
        if !height_m.is_finite() || height_m <= 0.0 {
            return Err(PhysicsError::CalculationError(format!(
                "canopy height must be finite and above zero, got {height_m}"
            )));
        }
        Surface::new(0.1 * height_m, 0.7 * height_m)
    }

    /// Aerodynamic roughness length `z₀`, in metres.
    #[inline]
    pub const fn roughness_m(&self) -> f64 {
        self.roughness_m
    }

    /// Zero-plane displacement `d`, in metres.
    #[inline]
    pub const fn displacement_m(&self) -> f64 {
        self.displacement_m
    }

    /// The height below which the logarithmic profile gives zero, in metres: `d + z₀`.
    ///
    /// Not a fudge — the log law's own zero crossing. Below it the expression is negative
    /// and then, at `d`, undefined; physically you are inside the roughness elements,
    /// where there is no surface-layer profile to speak of. [`canopy_profile`] is what
    /// covers that region for vegetation.
    #[inline]
    pub const fn profile_floor_m(&self) -> f64 {
        self.displacement_m + self.roughness_m
    }

    /// The power-law exponent `α` corresponding to this roughness.
    ///
    /// Counihan's (1975) fit, `α = 0.24 + 0.096·log₁₀ z₀ + 0.016·(log₁₀ z₀)²`, valid for
    /// `0.001 ≤ z₀ ≤ 10 m`. Open grassland comes out at 0.13 against the traditional
    /// one-seventh, and suburban terrain at 0.24.
    ///
    /// Provided so a caller who wants [`power_law`] does not have to invent an exponent,
    /// but the log profile is the better model and this is here for compatibility with
    /// codes and data that are quoted as power laws.
    #[inline]
    pub fn power_law_exponent(&self) -> f64 {
        let l = self.roughness_m.log10();
        0.24 + 0.096 * l + 0.016 * l * l
    }
}

// ---------------------------------------------------------------------------------
// Speed against height
// ---------------------------------------------------------------------------------

/// Friction velocity `u*`, in m/s, from one measured speed at one height.
///
/// `u* = κ·u_ref / ln((z_ref − d)/z₀)`. This is the quantity that is actually constant
/// through the surface layer — the wind speed is not — so it is what surface stress
/// (`τ = ρ u*²`) and every scaling argument is written in terms of.
///
/// # Arguments
///
/// * `reference_speed_ms` — measured wind speed, m/s.
/// * `reference_height_m` — height above ground at which it was measured, m.
/// * `surface` — the ground it was measured over.
///
/// # Errors
///
/// [`PhysicsError::CalculationError`] if the reference height is at or below
/// [`Surface::profile_floor_m`], where the profile has no scale to speak of, or if either
/// input is non-finite.
pub fn friction_velocity(
    reference_speed_ms: f64,
    reference_height_m: f64,
    surface: Surface,
) -> Result<f64, PhysicsError> {
    if !reference_speed_ms.is_finite() || reference_speed_ms < 0.0 {
        return Err(PhysicsError::CalculationError(format!(
            "reference wind speed must be finite and non-negative, got {reference_speed_ms}"
        )));
    }
    if !reference_height_m.is_finite() {
        return Err(PhysicsError::CalculationError(format!(
            "reference height must be finite, got {reference_height_m}"
        )));
    }
    let above = reference_height_m - surface.displacement_m;
    if above <= surface.roughness_m {
        return Err(PhysicsError::CalculationError(format!(
            "reference height {reference_height_m} m is at or below d + z0 = {} m for this \
             surface; a wind speed quoted from inside the roughness elements does not fix \
             a surface-layer profile",
            surface.profile_floor_m()
        )));
    }
    Ok(VON_KARMAN * reference_speed_ms / (above / surface.roughness_m).ln())
}

/// Wind speed at a height, in m/s, from the friction velocity.
///
/// `u(z) = (u*/κ)·ln((z − d)/z₀)` — the logarithmic wind profile, the neutral-stability
/// solution for a constant-flux surface layer. Prandtl (1932); see Stull, *An
/// Introduction to Boundary Layer Meteorology* (1988) §9.6, or ESDU 85020.
///
/// # Validity
///
/// Neutral stratification, and the **surface layer only** — the lowest 10% or so of the
/// boundary layer, which in practice means below roughly 100 m. Above that the wind turns
/// with height and the profile bends toward the geostrophic value, and this expression
/// keeps rising logarithmically instead. Strong daytime heating or a clear-night
/// inversion also break it; see the module docs on stability.
///
/// Returns `0.0` at or below `d + z₀`, which is the expression's own zero crossing rather
/// than a clamp.
///
/// # Arguments
///
/// * `friction_velocity_ms` — `u*`, m/s.
/// * `height_m` — height above ground, m.
/// * `surface` — the ground.
#[inline]
#[must_use]
pub fn log_profile(friction_velocity_ms: f64, height_m: f64, surface: Surface) -> f64 {
    let above = height_m - surface.displacement_m;
    if !(above > surface.roughness_m) {
        // Includes NaN, which lands here rather than propagating out of a logarithm.
        return 0.0;
    }
    friction_velocity_ms / VON_KARMAN * (above / surface.roughness_m).ln()
}

/// Wind speed at a height, in m/s, by the power law.
///
/// `u(z) = u_ref · (z/z_ref)^α`. Hellmann (1916); the familiar `α = 1/7` is the open-country
/// value. **This is an empirical fit and not a derived result** — it has no boundary
/// condition at the ground, no roughness length in it, and no theory behind the exponent.
/// It is here because a great deal of published wind data and several building codes are
/// quoted in this form, so a caller matching such data needs it. For anything where you
/// have a choice, [`log_profile`] is the better model.
///
/// Returns `0.0` for non-positive or non-finite heights.
///
/// # Arguments
///
/// * `reference_speed_ms` — speed at the reference height, m/s.
/// * `reference_height_m` — that height, m. Must be positive.
/// * `height_m` — the height wanted, m.
/// * `exponent` — `α`, dimensionless. See [`Surface::power_law_exponent`].
#[inline]
#[must_use]
pub fn power_law(
    reference_speed_ms: f64,
    reference_height_m: f64,
    height_m: f64,
    exponent: f64,
) -> f64 {
    if !(height_m > 0.0) || !(reference_height_m > 0.0) {
        return 0.0;
    }
    reference_speed_ms * (height_m / reference_height_m).powf(exponent)
}

// ---------------------------------------------------------------------------------
// Speed-up over a rise
// ---------------------------------------------------------------------------------

/// The shape of the rise the wind is crossing, which fixes the speed-up coefficient `K`.
///
/// Flow has to get *around* a three-dimensional hill as well as over it, so less of it is
/// squeezed through the thinned layer at the crest and the speed-up is smaller. A long
/// ridge gives the flow nowhere to go sideways and speeds it up most.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HillForm {
    /// A long ridge across the wind, effectively two-dimensional. `K = 2.0`.
    ///
    /// The standard figure, and the one wind-loading codes use for a 2-D ridge
    /// (Eurocode EN 1991-1-4 Annex A.3).
    Ridge,
    /// An isolated, roughly axisymmetric hill. `K = 1.6`.
    Hill,
    /// An escarpment or step: it rises and stays risen. `K = 0.8`.
    ///
    /// Half the ridge value, because there is no downwind slope to complete the
    /// squeeze — the streamlines relax again once past the edge.
    Escarpment,
}

impl HillForm {
    /// The speed-up coefficient `K`, dimensionless.
    ///
    /// Published values scatter by roughly ±20% because they depend on exactly how the
    /// half-length `L` was defined and on the approach roughness; these are the
    /// mid-range figures used in wind-loading practice.
    #[inline]
    #[must_use]
    pub const fn coefficient(self) -> f64 {
        match self {
            HillForm::Ridge => 2.0,
            HillForm::Hill => 1.6,
            HillForm::Escarpment => 0.8,
        }
    }
}

/// Fractional speed-up at the crest of a hill, dimensionless.
///
/// `Δs = K · H/L`, so the wind at the crest is `u_open · (1 + Δs)`. Returning the
/// *fraction* rather than the factor is deliberate and is the module's single convention:
/// the height decay in [`speedup_at_height`] multiplies the fraction, and a `1 +` floating
/// around between two functions is how the third call site gets it wrong.
///
/// # The result
///
/// Jackson & Hunt (1975), *Turbulent wind flow over a low hill*, Q. J. R. Meteorol. Soc.
/// **101**, 929–955. The flow over a hill divides into an outer region where the
/// streamlines are displaced and a thin inner region where turbulent stress matters; the
/// perturbation is linearised in `H/L`, and at the crest it comes out proportional to the
/// hill's aspect ratio with a coefficient that depends only on its form. See also Taylor,
/// Mason & Bradley (1987), *Boundary-Layer Meteorol.* **39**, 107–132, for the measured
/// coefficients, and ESDU 91043.
///
/// This is mass conservation with the turbulence done properly: the same flux crossing
/// the hill has to fit through a layer the hill has thinned.
///
/// # Validity
///
/// The linearisation is in `H/L`, and holds while the flow stays attached — up to about
/// `H/L = 0.3`, [`LINEARISATION_SLOPE_LIMIT`]. Beyond that this function returns the
/// value **at** the limit rather than extrapolating; see that constant for why the clamp
/// is physics rather than a guard. Use [`is_within_linearisation`] if you need to know
/// whether the clamp bound.
///
/// Also assumes neutral stratification, an isolated hill (not one of a range), and an
/// approach flow in equilibrium with the upwind surface.
///
/// # Arguments
///
/// * `height_m` — crest height above the surrounding ground, `H`, in metres.
/// * `half_length_m` — horizontal distance from the crest to where the ground is at half
///   the crest height, `L`, in metres. **Not** the full base width, and not the distance
///   to the foot.
/// * `form` — see [`HillForm`].
///
/// # Errors
///
/// [`PhysicsError::CalculationError`] if either length is non-finite or non-positive.
pub fn crest_speedup(
    height_m: f64,
    half_length_m: f64,
    form: HillForm,
) -> Result<f64, PhysicsError> {
    if !height_m.is_finite() || height_m <= 0.0 {
        return Err(PhysicsError::CalculationError(format!(
            "hill height must be finite and above zero, got {height_m}"
        )));
    }
    if !half_length_m.is_finite() || half_length_m <= 0.0 {
        return Err(PhysicsError::CalculationError(format!(
            "hill half-length must be finite and above zero, got {half_length_m}"
        )));
    }
    Ok(fractional_speedup_from_slope(height_m / half_length_m, form))
}

/// Fractional speed-up from a sampled ground gradient, dimensionless.
///
/// The same `Δs = K · H/L` as [`crest_speedup`], entered from the other end: for the
/// canonical hill the aspect ratio `H/L` *is* the mean upwind gradient, so terrain you can
/// only sample — a heightfield, where you know the local rise per metre but not where the
/// crest is — goes in here.
///
/// A **negative** gradient is the lee of the rise, and gives a slow-down. That is not an
/// extension of the result; the linearised solution is antisymmetric for a symmetric hill,
/// so the sheltered side is the same equation with the sign it already had. It is valid to
/// the same `H/L = 0.3` — past which the lee flow separates and there is no closed form at
/// all, which is why this clamps rather than continues.
///
/// # Totality
///
/// Infallible, non-allocating, and safe to call per element — this is the entry point a
/// per-plant or per-vertex bake wants, and returning a `Result` there would mean either a
/// `String` allocation in a loop over 10⁵ points or an `unwrap` that is a panic waiting
/// for a heightfield edge.
///
/// An infinite gradient has a sign, so it clamps like any other over-steep slope and gets
/// the limit value. **A NaN gradient does not**, and returns `0.0` — flat ground. That is
/// a real trade and worth naming: the alternative is propagating a NaN into a vertex
/// buffer, where it becomes a mesh that renders as a black spike three subsystems away
/// from the terrain sampler that produced it. A `debug_assert!` fires first, so the case is
/// loud where you can afford loud and total where you cannot.
///
/// # Arguments
///
/// * `slope` — the upwind gradient, rise over run, dimensionless. Positive uphill.
/// * `form` — see [`HillForm`].
#[inline]
#[must_use]
pub fn fractional_speedup_from_slope(slope: f64, form: HillForm) -> f64 {
    debug_assert!(!slope.is_nan(), "NaN upwind slope treated as flat ground");
    if slope.is_nan() {
        return 0.0;
    }
    form.coefficient() * slope.clamp(-LINEARISATION_SLOPE_LIMIT, LINEARISATION_SLOPE_LIMIT)
}

/// Whether a gradient is inside the range Jackson–Hunt actually covers.
///
/// `|H/L| ≤ 0.3`. The speed-up functions clamp silently by design — a game does not want
/// an error per vertex — so this is how a caller finds out that the clamp bound, for
/// instance when validating a heightfield or reporting that a map has terrain the model
/// cannot speak to.
#[inline]
#[must_use]
pub fn is_within_linearisation(slope: f64) -> bool {
    slope.is_finite() && slope.abs() <= LINEARISATION_SLOPE_LIMIT
}

/// The crest speed-up carried up to a height above the crest, dimensionless.
///
/// `Δs(z) = Δs_crest · exp(−A·z/L)`. The speed-up is a surface effect: it is largest at
/// the ground and gone by a height comparable with the hill's own length, because that is
/// the scale over which the streamline displacement relaxes.
///
/// **The decay constant is an empirical fit, unlike `K`.** Published `A` runs about
/// 2.5–3.5 (Taylor & Lee 1984, *Climatological Bulletin* **18**, 3–32; Lemelin, Surry &
/// Davenport 1988, *J. Wind Eng. Ind. Aerodyn.* **28**, 117–127) and 3.0 is used here. The
/// distinction matters: `K = 2` is what a linearised theory produces, `A = 3` is what
/// somebody measured, and only one of those tightens if you look harder.
///
/// No horizontal decay is supplied. It exists in the same references, but it needs the
/// distance from the crest, which is exactly what a caller sampling a local gradient does
/// not know — and a horizontal factor applied to a locally-sampled slope would be counting
/// the same falloff twice.
///
/// Returns the crest value unchanged at `z = 0`, and `0.0` for non-finite inputs.
///
/// # Arguments
///
/// * `crest_speedup` — the fraction from [`crest_speedup`] or
///   [`fractional_speedup_from_slope`].
/// * `height_m` — height above the crest, m. Negative heights are treated as the crest.
/// * `half_length_m` — the hill's `L`, m, as in [`crest_speedup`].
#[inline]
#[must_use]
pub fn speedup_at_height(crest_speedup: f64, height_m: f64, half_length_m: f64) -> f64 {
    if !crest_speedup.is_finite() || !height_m.is_finite() || !(half_length_m > 0.0) {
        return 0.0;
    }
    crest_speedup * (-SPEEDUP_HEIGHT_DECAY * height_m.max(0.0) / half_length_m).exp()
}

// ---------------------------------------------------------------------------------
// Inside a canopy
// ---------------------------------------------------------------------------------

/// Wind speed inside standing vegetation, in m/s.
///
/// `u(z) = u_h · exp(−a·(1 − z/h))` for `0 ≤ z ≤ h`, where `u_h` is the speed at the
/// canopy top and `a` is an attenuation coefficient. Cionco (1965), *A mathematical model
/// for air flow in a vegetative canopy*, J. Appl. Meteorol. **4**, 517–522; see also
/// Brutsaert 1982 §5.
///
/// # This is the defensible form of sheltering
///
/// Below the canopy top the logarithmic profile does not apply — momentum is absorbed by
/// the foliage all the way down rather than only at the ground, so the flux is not
/// constant with height and the log law's premise is gone. The exponential is the solution
/// for a canopy of uniform leaf area with a constant drag coefficient. It is why a plant
/// standing among other plants of its own height feels a fraction of the wind that the same
/// plant in the open would, and it needs no obstacle wake model to say so.
///
/// # Validity
///
/// A closed canopy of reasonably uniform height and density; it does not describe an
/// isolated bush or a windbreak with a gap under it. `a` is the fitted parameter and it
/// varies a lot — Cionco's own measurements run from about 0.4 for immature crops to
/// about 4 for dense forest. See [`CanopyAttenuation`] for the useful range.
///
/// Above `h` this returns `speed_at_top_ms` unchanged rather than extrapolating, because
/// above the canopy the log profile is the right model and this one is not.
///
/// # Arguments
///
/// * `speed_at_top_ms` — wind speed at the canopy top, m/s. Typically
///   [`log_profile`] evaluated at `h` for a [`Surface::canopy`].
/// * `height_m` — height above ground, m.
/// * `canopy_height_m` — height of the canopy top above ground, m.
/// * `attenuation` — `a`, dimensionless. See [`CanopyAttenuation`].
#[inline]
#[must_use]
pub fn canopy_profile(
    speed_at_top_ms: f64,
    height_m: f64,
    canopy_height_m: f64,
    attenuation: f64,
) -> f64 {
    if !(canopy_height_m > 0.0) || !height_m.is_finite() {
        return 0.0;
    }
    if height_m >= canopy_height_m {
        return speed_at_top_ms;
    }
    let fraction = (height_m.max(0.0) / canopy_height_m).clamp(0.0, 1.0);
    speed_at_top_ms * (-attenuation.max(0.0) * (1.0 - fraction)).exp()
}

/// Published attenuation coefficients `a` for [`canopy_profile`].
///
/// From Cionco's (1965) fitted values and the compilations in Brutsaert (1982) §5. They
/// scatter, and the scatter is real rather than experimental: the coefficient encodes leaf
/// area density, which is the difference between a spring and an autumn wood.
pub struct CanopyAttenuation;

impl CanopyAttenuation {
    /// Grass and immature crops. Barely attenuating; the profile is nearly uniform.
    pub const GRASS: f64 = 0.5;
    /// Mature cereals and low shrub.
    pub const CROPS: f64 = 1.5;
    /// Open woodland, pine forest — trunks below, foliage above.
    pub const OPEN_WOODLAND: f64 = 1.0;
    /// Dense broadleaf forest in leaf. The far end of the published range.
    pub const DENSE_FOREST: f64 = 3.0;
}

// ---------------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------------

/// Dynamic pressure of moving air, in pascals.
///
/// `q = ½ρu²`. One line, and it is here because it is the point where the two halves of
/// this work meet: `ρ` comes from [`Air::density`], which is a function of the actual
/// weather, so a winter gale loads a structure about 9% harder than a summer one of the
/// same speed. Under the old frozen 1.225 it loaded it identically, and no test in the
/// crate could tell.
///
/// # Arguments
///
/// * `air` — the state of the air.
/// * `speed_ms` — wind speed, m/s. Sign is irrelevant; the square takes care of it.
#[inline]
#[must_use]
pub fn dynamic_pressure(air: &Air, speed_ms: f64) -> f64 {
    0.5 * air.density() * speed_ms * speed_ms
}

// ---------------------------------------------------------------------------------
// The composed profile
// ---------------------------------------------------------------------------------

/// A wind profile pinned to one measured speed over one surface.
///
/// # Why this exists rather than six-argument free functions
///
/// The free functions above take a reference speed, a reference height, a query height, a
/// roughness and a displacement — five `f64`, three of them lengths in metres and
/// therefore mutually substitutable by any type system that is not going to be built here.
/// Composing them per query means restating four arguments that never change, at every one
/// of the tens of thousands of call sites a heightfield bake makes.
///
/// So the invariant moves up a rung by moving the validation to construction. `WindProfile`
/// **cannot exist** with a reference below the roughness sublayer, with a non-finite speed,
/// or over a zero roughness length — [`WindProfile::new`] returns a `Result` and is the
/// only way in. Every query method is then infallible, allocation-free, `#[inline]`, and
/// takes only the arguments that actually vary per point.
///
/// That is the trade this type is making: one `Result` at setup, in exchange for a hot loop
/// with no error path in it and no way to pass the wrong metre.
///
/// # Example
///
/// ```
/// use rs_physics::atmosphere::{Surface, WindProfile, HillForm};
///
/// // Ten metres per second at the standard ten-metre mast, over open grass.
/// let wind = WindProfile::new(10.0, 10.0, Surface::OPEN_GRASS).unwrap();
///
/// // A blade of grass at 30 cm feels far less than the mast reads.
/// let at_grass = wind.at_height(0.3);
/// let at_mast = wind.at_height(10.0);
/// assert!(at_grass < at_mast * 0.7);
///
/// // And a shrub on an exposed shoulder feels more than one in the hollow below it.
/// let shoulder = wind.over_slope(1.0, 0.2, HillForm::Ridge);
/// let hollow = wind.over_slope(1.0, -0.2, HillForm::Ridge);
/// assert!(shoulder > hollow * 2.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindProfile {
    friction_velocity_ms: f64,
    surface: Surface,
}

impl WindProfile {
    /// A profile through one measured speed at one height.
    ///
    /// # Arguments
    ///
    /// * `reference_speed_ms` — measured wind speed, m/s, non-negative.
    /// * `reference_height_m` — the height it was measured at, m. Ten metres is the
    ///   meteorological standard and what most quoted wind speeds mean.
    /// * `surface` — the ground it was measured over. If the reference was taken over
    ///   different terrain from the terrain being queried, that is an internal boundary
    ///   layer problem and neither this type nor anything else here solves it.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::CalculationError`] if the speed is negative or non-finite, or if
    /// the reference height is at or below [`Surface::profile_floor_m`].
    pub fn new(
        reference_speed_ms: f64,
        reference_height_m: f64,
        surface: Surface,
    ) -> Result<WindProfile, PhysicsError> {
        let friction_velocity_ms =
            friction_velocity(reference_speed_ms, reference_height_m, surface)?;
        Ok(WindProfile { friction_velocity_ms, surface })
    }

    /// A profile from a friction velocity directly, in m/s.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::CalculationError`] if `u*` is negative or non-finite.
    pub fn from_friction_velocity(
        friction_velocity_ms: f64,
        surface: Surface,
    ) -> Result<WindProfile, PhysicsError> {
        if !friction_velocity_ms.is_finite() || friction_velocity_ms < 0.0 {
            return Err(PhysicsError::CalculationError(format!(
                "friction velocity must be finite and non-negative, got {friction_velocity_ms}"
            )));
        }
        Ok(WindProfile { friction_velocity_ms, surface })
    }

    /// The friction velocity `u*` of this profile, in m/s.
    #[inline]
    pub const fn friction_velocity_ms(&self) -> f64 {
        self.friction_velocity_ms
    }

    /// The surface this profile is over.
    #[inline]
    pub const fn surface(&self) -> Surface {
        self.surface
    }

    /// Wind speed at a height above flat ground, in m/s. See [`log_profile`].
    #[inline]
    #[must_use]
    pub fn at_height(&self, height_m: f64) -> f64 {
        log_profile(self.friction_velocity_ms, height_m, self.surface)
    }

    /// Wind speed at a height over sloping ground, in m/s.
    ///
    /// The log profile scaled by the Jackson–Hunt speed-up for the local upwind gradient:
    /// `u(z)·(1 + Δs)`. This is the per-point call a terrain bake wants — pass the height
    /// of the thing and the rise per metre of the ground upwind of it, and everything else
    /// is already in the profile.
    ///
    /// The speed-up is applied at the surface value rather than decayed with height,
    /// because the caller who has a sampled gradient does not have the hill's `L` and so
    /// cannot evaluate the decay. For anything more than a few metres above ground over a
    /// hill whose dimensions you know, compose [`crest_speedup`] with
    /// [`speedup_at_height`] instead.
    ///
    /// # Arguments
    ///
    /// * `height_m` — height above ground, m.
    /// * `upwind_slope` — rise over run of the ground upwind, dimensionless.
    /// * `form` — see [`HillForm`].
    #[inline]
    #[must_use]
    pub fn over_slope(&self, height_m: f64, upwind_slope: f64, form: HillForm) -> f64 {
        self.at_height(height_m) * (1.0 + fractional_speedup_from_slope(upwind_slope, form))
    }

    /// Wind speed at a height inside a canopy of a given height, in m/s.
    ///
    /// Evaluates the log profile at the canopy top and hands it to [`canopy_profile`].
    /// Note that for this to be right the profile's [`Surface`] should be the canopy's own
    /// — see [`Surface::canopy`] — or the speed at the top will be the speed over
    /// whatever the profile was built for.
    ///
    /// # Arguments
    ///
    /// * `height_m` — height above ground, m.
    /// * `canopy_height_m` — height of the canopy top, m.
    /// * `attenuation` — `a`, dimensionless. See [`CanopyAttenuation`].
    #[inline]
    #[must_use]
    pub fn in_canopy(&self, height_m: f64, canopy_height_m: f64, attenuation: f64) -> f64 {
        let at_top = self.at_height(canopy_height_m);
        canopy_profile(at_top, height_m, canopy_height_m, attenuation)
    }

    /// Dynamic pressure at a height above flat ground, in pascals. See
    /// [`dynamic_pressure`].
    #[inline]
    #[must_use]
    pub fn dynamic_pressure_at(&self, air: &Air, height_m: f64) -> f64 {
        dynamic_pressure(air, self.at_height(height_m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------------
    // The log profile, against published behaviour and against its own limits.
    // -----------------------------------------------------------------------------

    /// The textbook worked example: over open country (`z₀ = 0.03 m`), a 10 m/s wind at
    /// the standard 10 m mast implies `u* ≈ 0.69 m/s`. That is `κu/ln(z/z₀)` with the
    /// numbers a meteorology course uses, not a number this code produced.
    #[test]
    fn friction_velocity_matches_the_worked_example() {
        let u_star = friction_velocity(10.0, 10.0, Surface::OPEN_GRASS).unwrap();
        // ln(10/0.03) = 5.809, so u* = 0.40*10/5.809 = 0.6886.
        assert!(
            (u_star - 0.689).abs() < 0.005,
            "u* came out {u_star:.4} m/s against a hand-worked 0.689",
        );
    }

    /// The profile must pass back through the point it was pinned to. If it does not, the
    /// reference height and the query height have been swapped somewhere.
    #[test]
    fn the_profile_reproduces_its_own_reference() {
        for &(speed, height) in &[(10.0, 10.0), (3.5, 2.0), (25.0, 80.0)] {
            for surface in [Surface::OPEN_SEA, Surface::OPEN_GRASS, Surface::FOREST] {
                let wind = WindProfile::new(speed, height, surface).unwrap();
                let back = wind.at_height(height);
                assert!(
                    (back - speed).abs() < 1e-9,
                    "pinned at {speed} m/s at {height} m over z0={}, read back {back}",
                    surface.roughness_m(),
                );
            }
        }
    }

    /// **The result the consumer wants.** Rough ground bleeds more momentum out of the
    /// air near the surface, so for the same wind aloft the profile over forest is far
    /// steeper — a plant at knee height in a wood feels much less than one on a lawn.
    #[test]
    fn rough_ground_shelters_the_ground_more_than_smooth_ground() {
        // Pin both to the same speed high up, well above either surface's influence.
        let smooth = WindProfile::new(20.0, 100.0, Surface::OPEN_GRASS).unwrap();
        let rough = WindProfile::new(20.0, 100.0, Surface::FOREST).unwrap();
        let low = 0.5;
        assert!(
            rough.at_height(low) < smooth.at_height(low) * 0.7,
            "at {low} m: forest {:.2} m/s against grass {:.2} m/s — the profiles are not \
             separating, which means roughness is not reaching the answer",
            rough.at_height(low),
            smooth.at_height(low),
        );
    }

    /// The zero crossing is `d + z₀`, and it is the expression's own, not a clamp bolted
    /// on. Below it there is no profile: you are among the roughness elements.
    #[test]
    fn the_profile_goes_to_zero_at_the_roughness_length() {
        let wind = WindProfile::new(10.0, 10.0, Surface::OPEN_GRASS).unwrap();
        let z0 = Surface::OPEN_GRASS.roughness_m();
        assert_eq!(wind.at_height(z0), 0.0, "not zero at z0");
        assert_eq!(wind.at_height(0.0), 0.0, "not zero at the ground");
        assert_eq!(wind.at_height(-5.0), 0.0, "not zero below the ground");
        assert!(wind.at_height(z0 * 1.001) > 0.0, "still zero just above z0");
    }

    /// A canopy lifts the whole profile by its displacement height, so the zero crossing
    /// moves from a few centimetres up to most of the canopy's height.
    #[test]
    fn a_canopy_lifts_the_profile_off_the_ground() {
        let forest = Surface::canopy(20.0).unwrap();
        assert!((forest.roughness_m() - 2.0).abs() < 1e-12);
        assert!((forest.displacement_m() - 14.0).abs() < 1e-12);
        assert!((forest.profile_floor_m() - 16.0).abs() < 1e-12);

        let wind = WindProfile::new(15.0, 60.0, forest).unwrap();
        assert_eq!(wind.at_height(10.0), 0.0, "log profile still non-zero inside the trunks");
    }

    /// A reference measured from inside the roughness sublayer does not pin a profile,
    /// and the type refuses rather than returning a negative or infinite `u*`.
    #[test]
    fn a_reference_below_the_roughness_sublayer_is_rejected() {
        let forest = Surface::canopy(20.0).unwrap();
        assert!(
            WindProfile::new(5.0, 10.0, forest).is_err(),
            "accepted a reference speed measured 10 m up inside a 20 m canopy",
        );
        assert!(WindProfile::new(5.0, 0.03, Surface::OPEN_GRASS).is_err(), "accepted z = z0");
        assert!(WindProfile::new(f64::NAN, 10.0, Surface::OPEN_GRASS).is_err(), "accepted NaN");
        assert!(WindProfile::new(-3.0, 10.0, Surface::OPEN_GRASS).is_err(), "accepted -3 m/s");
    }

    /// Zero roughness is not a smooth surface, it is `ln(z/0)`. The constructor is where
    /// that stops, so the four functions that take the logarithm do not each need a guard.
    #[test]
    fn impossible_surfaces_are_rejected_at_construction() {
        assert!(Surface::new(0.0, 0.0).is_err(), "z0 = 0 accepted");
        assert!(Surface::new(-0.1, 0.0).is_err(), "negative z0 accepted");
        assert!(Surface::new(f64::NAN, 0.0).is_err(), "NaN z0 accepted");
        assert!(Surface::new(0.03, -1.0).is_err(), "negative displacement accepted");
        assert!(Surface::canopy(0.0).is_err(), "zero-height canopy accepted");
    }

    /// And having been rejected there, no constructible profile can produce a non-finite
    /// speed at any height a caller might ask for — including the ones a heightfield edge
    /// produces.
    #[test]
    fn every_constructible_profile_is_finite_at_every_height() {
        for surface in [
            Surface::OPEN_SEA,
            Surface::OPEN_GRASS,
            Surface::CITY,
            Surface::canopy(30.0).unwrap(),
        ] {
            let wind = WindProfile::new(12.0, 100.0, surface).unwrap();
            for &z in &[-1e6, -1.0, 0.0, 1e-9, 0.01, 1.0, 100.0, 1e6] {
                let u = wind.at_height(z);
                assert!(u.is_finite(), "at_height({z}) over z0={} gave {u}", surface.roughness_m());
                assert!(u >= 0.0, "at_height({z}) gave a negative speed {u}");
                // NaN is deliberately absent: it trips the `debug_assert` in
                // `fractional_speedup_from_slope` by design, and has its own test below.
                for slope in [f64::NEG_INFINITY, -1e6, -0.5, 0.0, 0.5, 1e6, f64::INFINITY] {
                    let s = wind.over_slope(z, slope, HillForm::Ridge);
                    assert!(s.is_finite(), "over_slope({z}, {slope}) gave {s}");
                    assert!(s >= 0.0, "over_slope({z}, {slope}) gave {s}");
                }
            }
        }
    }

    // -----------------------------------------------------------------------------
    // The power law.
    // -----------------------------------------------------------------------------

    /// Counihan's fit must reproduce the two exponents everybody quotes: about one
    /// seventh over open country and about a quarter over suburbs.
    #[test]
    fn counihans_exponents_match_the_quoted_values() {
        let open = Surface::OPEN_GRASS.power_law_exponent();
        assert!((0.12..0.16).contains(&open), "open country came out {open:.3}, not ~1/7");
        let suburb = Surface::FOREST.power_law_exponent();
        assert!((0.22..0.26).contains(&suburb), "suburban came out {suburb:.3}, not ~0.24");
        assert!(
            Surface::OPEN_SEA.power_law_exponent() < open,
            "the sea came out rougher than a field",
        );
    }

    /// The power law and the log law should not disagree wildly over the range they are
    /// both meant for. They are different models, so this is a sanity band rather than an
    /// equality — but a factor-of-two divergence at 20 m would mean one of them is wrong.
    #[test]
    fn the_power_law_tracks_the_log_law_over_the_surface_layer() {
        let surface = Surface::OPEN_GRASS;
        let wind = WindProfile::new(10.0, 10.0, surface).unwrap();
        let alpha = surface.power_law_exponent();
        for &z in &[2.0, 5.0, 20.0, 50.0] {
            let logged = wind.at_height(z);
            let powered = power_law(10.0, 10.0, z, alpha);
            let ratio = powered / logged;
            assert!(
                (0.85..1.15).contains(&ratio),
                "at {z} m the power law gave {powered:.2} and the log law {logged:.2}",
            );
        }
    }

    // -----------------------------------------------------------------------------
    // Jackson-Hunt.
    // -----------------------------------------------------------------------------

    /// The published result: a 2-D ridge of aspect ratio `H/L = 0.1` speeds the wind at
    /// its crest by 20%. That is `K·H/L` with `K = 2`, which is the figure in the
    /// wind-loading codes.
    #[test]
    fn crest_speedup_matches_the_published_coefficient() {
        let s = crest_speedup(50.0, 500.0, HillForm::Ridge).unwrap();
        assert!((s - 0.2).abs() < 1e-12, "H/L = 0.1 ridge gave {s}, not 0.20");
        let hill = crest_speedup(50.0, 500.0, HillForm::Hill).unwrap();
        assert!((hill - 0.16).abs() < 1e-12, "H/L = 0.1 hill gave {hill}, not 0.16");
    }

    /// It is the aspect ratio that matters and nothing else — a 10 m knoll and a 1000 m
    /// mountain of the same shape speed the wind up by the same fraction. That is the
    /// content of the linearisation and the easiest thing to get wrong by carrying an
    /// absolute height somewhere.
    #[test]
    fn only_the_aspect_ratio_matters() {
        let small = crest_speedup(10.0, 100.0, HillForm::Ridge).unwrap();
        let large = crest_speedup(1000.0, 10_000.0, HillForm::Ridge).unwrap();
        assert!((small - large).abs() < 1e-12, "{small} against {large} for the same H/L");
    }

    /// **Flat ground does nothing.** The limit that catches a stray `+1` or a coefficient
    /// applied to the wrong quantity.
    #[test]
    fn flat_ground_has_no_speedup() {
        for form in [HillForm::Ridge, HillForm::Hill, HillForm::Escarpment] {
            assert_eq!(fractional_speedup_from_slope(0.0, form), 0.0);
        }
        let wind = WindProfile::new(10.0, 10.0, Surface::OPEN_GRASS).unwrap();
        assert!(
            (wind.over_slope(2.0, 0.0, HillForm::Ridge) - wind.at_height(2.0)).abs() < 1e-12,
            "a zero slope changed the speed",
        );
    }

    /// The lee is the same equation with the sign it already had, and it is a slow-down of
    /// exactly the magnitude the windward side is a speed-up.
    #[test]
    fn the_lee_is_the_windward_side_with_the_sign_reversed() {
        for slope in [0.05, 0.1, 0.2, 0.3] {
            let up = fractional_speedup_from_slope(slope, HillForm::Ridge);
            let down = fractional_speedup_from_slope(-slope, HillForm::Ridge);
            assert!((up + down).abs() < 1e-12, "slope {slope}: {up} up against {down} down");
            assert!(up > 0.0 && down < 0.0, "slope {slope} did not accelerate uphill");
        }
    }

    /// **The validity boundary, tested as a boundary.** Past `H/L = 0.3` the answer stops
    /// changing, because past there the model has stopped being one. A 45° scarp gets the
    /// 17° answer, not a bigger one.
    #[test]
    fn the_linearisation_clamps_at_its_own_validity_limit() {
        let at_limit = fractional_speedup_from_slope(LINEARISATION_SLOPE_LIMIT, HillForm::Ridge);
        assert!((at_limit - 0.6).abs() < 1e-12, "the limit gave {at_limit}, not 2 * 0.3");
        for beyond in [0.31, 0.5, 1.0, 10.0, f64::INFINITY] {
            let s = fractional_speedup_from_slope(beyond, HillForm::Ridge);
            assert!(
                (s - at_limit).abs() < 1e-12,
                "slope {beyond} gave {s}, extrapolating past the limit's {at_limit}",
            );
            assert!(!is_within_linearisation(beyond), "{beyond} reported as valid");
        }
        assert!(is_within_linearisation(0.3), "the limit itself reported as invalid");
        assert!(is_within_linearisation(-0.3), "the negative limit reported as invalid");
    }

    /// And the clamp fixes the floor: nothing is ever sheltered below 40% of the open-ground
    /// speed by this term, and in particular the speed never goes negative — which is what
    /// an unclamped `1 + K·slope` does at a slope of −0.5.
    #[test]
    fn the_sheltered_side_has_a_floor_and_never_reverses() {
        let wind = WindProfile::new(10.0, 10.0, Surface::OPEN_GRASS).unwrap();
        let open = wind.at_height(2.0);
        for slope in [-0.3, -0.5, -1.0, -100.0] {
            let sheltered = wind.over_slope(2.0, slope, HillForm::Ridge);
            assert!(sheltered > 0.0, "slope {slope} gave a negative wind speed {sheltered}");
            assert!(
                (sheltered / open - 0.4).abs() < 1e-9,
                "slope {slope} gave {:.4} of open ground, not the 1 - 2*0.3 = 0.4 floor",
                sheltered / open,
            );
        }
    }

    /// A NaN gradient — which is what a heightfield sampled off its own edge produces —
    /// must not reach a vertex buffer. The function is loud about it in a debug build and
    /// total in a release one, and this asserts whichever applies rather than testing only
    /// the configuration it happens to be run in.
    #[test]
    fn a_nan_slope_is_loud_in_debug_and_flat_in_release() {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let outcome = std::panic::catch_unwind(|| {
            fractional_speedup_from_slope(f64::NAN, HillForm::Ridge)
        });
        std::panic::set_hook(previous);

        match outcome {
            Ok(value) => {
                assert!(!cfg!(debug_assertions), "a debug build did not assert on a NaN slope");
                assert_eq!(value, 0.0, "a release build gave {value} for a NaN slope");
            }
            Err(_) => assert!(cfg!(debug_assertions), "a release build panicked on a NaN slope"),
        }
        assert!(!is_within_linearisation(f64::NAN), "NaN reported as inside the linearisation");
    }

    /// Degenerate hills are refused rather than divided by.
    #[test]
    fn degenerate_hills_are_rejected() {
        assert!(crest_speedup(0.0, 100.0, HillForm::Ridge).is_err(), "zero-height hill accepted");
        assert!(crest_speedup(10.0, 0.0, HillForm::Ridge).is_err(), "zero-length hill accepted");
        assert!(crest_speedup(f64::NAN, 100.0, HillForm::Ridge).is_err(), "NaN height accepted");
        assert!(
            crest_speedup(10.0, f64::INFINITY, HillForm::Ridge).is_err(),
            "infinite hill accepted",
        );
    }

    /// The speed-up is a surface effect and is gone within about a hill length above the
    /// crest. At `z = 0` it is untouched; at `z = L` it is down to about 5%.
    #[test]
    fn the_speedup_decays_away_above_the_crest() {
        let crest = crest_speedup(50.0, 500.0, HillForm::Ridge).unwrap();
        assert!((speedup_at_height(crest, 0.0, 500.0) - crest).abs() < 1e-12, "changed at z=0");
        let aloft = speedup_at_height(crest, 500.0, 500.0);
        assert!(aloft < crest * 0.06, "a full hill-length up it was still {aloft:.4}");
        assert!(aloft > 0.0, "it went to exactly zero, which an exponential does not");
        // Monotone all the way up.
        let mut previous = crest;
        for step in 1..20 {
            let z = step as f64 * 50.0;
            let s = speedup_at_height(crest, z, 500.0);
            assert!(s < previous, "speed-up rose between {} and {z} m", z - 50.0);
            previous = s;
        }
    }

    // -----------------------------------------------------------------------------
    // The canopy.
    // -----------------------------------------------------------------------------

    /// At the canopy top the exponential is 1: the two profiles meet, which is the whole
    /// point of pinning it there.
    #[test]
    fn the_canopy_profile_matches_the_log_profile_at_the_canopy_top() {
        let forest = Surface::canopy(20.0).unwrap();
        let wind = WindProfile::new(15.0, 60.0, forest).unwrap();
        let at_top_log = wind.at_height(20.0);
        let at_top_canopy = wind.in_canopy(20.0, 20.0, CanopyAttenuation::DENSE_FOREST);
        assert!(
            (at_top_log - at_top_canopy).abs() < 1e-9,
            "the two profiles disagree at the canopy top: {at_top_log} against {at_top_canopy}",
        );
    }

    /// **The published number.** Cionco's model gives `u(0)/u(h) = exp(−a)`, so a dense
    /// forest at `a = 3` leaves about 5% of the canopy-top wind at the forest floor.
    /// That is a strong, checkable statement and it is why a wood is still.
    #[test]
    fn a_dense_forest_floor_is_nearly_still() {
        let at_floor = canopy_profile(10.0, 0.0, 20.0, CanopyAttenuation::DENSE_FOREST);
        let expected = 10.0 * (-3.0f64).exp();
        assert!(
            (at_floor - expected).abs() < 1e-9,
            "forest floor came out {at_floor:.3} m/s against exp(-a) * 10 = {expected:.3}",
        );
        assert!(at_floor < 0.6, "5% of 10 m/s is 0.5; got {at_floor:.3}");
    }

    /// And grass barely attenuates at all, which is what makes the coefficient worth
    /// carrying rather than fixing.
    #[test]
    fn grass_barely_shelters_its_own_base() {
        let grass = canopy_profile(10.0, 0.0, 0.4, CanopyAttenuation::GRASS);
        let forest = canopy_profile(10.0, 0.0, 20.0, CanopyAttenuation::DENSE_FOREST);
        assert!(grass > 5.0, "grass base came out {grass:.2} m/s from a 10 m/s top");
        assert!(grass > forest * 10.0, "grass and forest are not separating");
    }

    /// Monotone from floor to top, and it does not extrapolate above the canopy — above
    /// `h` the log law is the right model and this one hands back its boundary value.
    #[test]
    fn the_canopy_profile_is_monotone_and_stops_at_the_top() {
        let h = 12.0;
        let mut previous = -1.0;
        for step in 0..=24 {
            let z = step as f64 * h / 24.0;
            let u = canopy_profile(8.0, z, h, CanopyAttenuation::CROPS);
            assert!(u > previous, "canopy wind fell going up, at {z} m");
            assert!(u.is_finite() && u >= 0.0, "canopy wind at {z} m came out {u}");
            previous = u;
        }
        assert_eq!(canopy_profile(8.0, h * 2.0, h, CanopyAttenuation::CROPS), 8.0);
        assert_eq!(canopy_profile(8.0, 1e9, h, CanopyAttenuation::CROPS), 8.0);
    }

    /// Zero attenuation is a canopy that is not there — the profile is flat at the top
    /// value. The limit that catches a sign error in the exponent.
    #[test]
    fn a_canopy_with_no_attenuation_does_nothing() {
        for z in [0.0, 1.0, 5.0, 10.0] {
            assert!((canopy_profile(7.0, z, 10.0, 0.0) - 7.0).abs() < 1e-12);
        }
    }

    // -----------------------------------------------------------------------------
    // Loading, and the tie back to the air.
    // -----------------------------------------------------------------------------

    /// Dynamic pressure at 10 m/s in ISA air is 61 Pa: `0.5 * 1.225 * 100`. Textbook.
    #[test]
    fn dynamic_pressure_matches_the_textbook() {
        let q = dynamic_pressure(&Air::sea_level(), 10.0);
        assert!((q - 61.25).abs() < 0.05, "10 m/s in ISA air gave {q:.2} Pa against 61.25");
        assert!((dynamic_pressure(&Air::sea_level(), -10.0) - q).abs() < 1e-12, "sign leaked");
        assert_eq!(dynamic_pressure(&Air::sea_level(), 0.0), 0.0);
    }

    /// **The reconciliation, measured.** Winter loads harder than summer at the same wind
    /// speed, by the density ratio — and under the crate's old frozen 1.225 this test
    /// would have asserted a ratio of exactly 1.
    #[test]
    fn winter_wind_loads_harder_than_summer_wind_at_the_same_speed() {
        let winter = dynamic_pressure(&Air::winter(), 20.0);
        let summer = dynamic_pressure(&Air::standard(), 20.0);
        let ratio = winter / summer;
        assert!(
            (1.08..1.11).contains(&ratio),
            "winter/summer loading came out {ratio:.4}; the air is 9% denser so the load is \
             9% greater, and a frozen density would have made this 1.000",
        );
    }
}
