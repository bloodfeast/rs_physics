//! # Wetting
//!
//! How water takes to a material: whether it soaks in, beads up or spreads into a film,
//! and what that does to the light the surface sends back.
//!
//! ## Why a wet thing is darker
//!
//! Two separate effects, and a material can have either, both or neither.
//!
//! **The film** (Angstrom 1925; Lekner and Dorf, "Why some things are darker when wet",
//! *Applied Optics* 27(7), 1988). A thin layer of water over a surface that scatters
//! diffusely traps part of the scattered light: what leaves the surface at a grazing
//! angle meets the water-air boundary past its critical angle and is turned back down
//! for a second chance at being absorbed. Each round trip costs the surface's own
//! absorption again, so the darkening grows as the albedo falls. A surface that is
//! already smooth under a clear coat (gloss paint, glass) gains almost nothing, because
//! the trap is already there: water over it changes which boundary is the outer one,
//! not whether there is one. A metal has nothing to trap at all: its reflectance comes
//! off the surface, not scattered back from inside. So the film is applied only to a
//! [`Finish::Diffuse`] surface; on a [`Finish::Smooth`] one (metal, glass, gloss paint)
//! water adds a mirror and darkens nothing. [`film_albedo`] is the law for the diffuse
//! case.
//!
//! **The pores** (Twomey, Bohren and Mergenthaler, "Reflectance and albedo differences
//! between wet and dry surfaces", *Applied Optics* 25(3), 1986). A porous material scatters
//! light at every boundary between its solid and the air in its pores. Fill the pores
//! with water and each boundary's index step shrinks (cellulose against air is 1.53 to
//! 1.00; against water it is 1.53 to 1.33), so the material scatters far less per unit
//! depth and light goes deeper and is absorbed more before it comes back. This is why
//! soaked cotton is several times darker, not a few percent: [`pore_filled_albedo`]
//! takes it through the Kubelka-Munk two-flux model.
//!
//! ## Model limits
//!
//! - Albedos are linear reflectances (linear light), for one wavelength band at a time;
//!   call per channel for colour. The pore term is nonlinear in the albedo, so an
//!   sRGB-encoded value gives a wrong answer: decode it first.
//! - Open porosity is a gate, not a quantity in the albedo: any open porosity that water
//!   wets gives the full pore-filling collapse at saturation 1. In the Twomey, Bohren and
//!   Mergenthaler picture a porous material scatters at its pore boundaries, and filling
//!   them changes every boundary's index step however many there are; the porosity sets
//!   how much water the material holds ([`Wetting::saturation_water_content`]), not how
//!   dark it gets. A sealed coat with a few cracks is better described as sealed.
//! - The pore term scales the scattering by the normal-incidence Fresnel reflectance of
//!   one boundary, a proxy for the scattering strength of a random pore network. It is
//!   a ratio, so the material's own absolute scattering never enters.
//! - The film is thin (no absorption in the water) and the illumination diffuse.
//!
//! ## Examples
//!
//! ```
//! use rs_physics::materials::{Material, Wetting};
//!
//! // Soaked canvas against a wet steel plate, both 0.5 albedo dry: the canvas goes
//! // several times darker, the steel not at all (it only gains gloss).
//! let canvas = Material::canvas().wetting.unwrap();
//! let steel = Material::steel().wetting.unwrap();
//! assert!(canvas.wet_albedo(0.5, 1.0) < 0.25);
//! assert_eq!(steel.wet_albedo(0.5, 1.0), 0.5);
//! assert_eq!(steel.pore_scattering_ratio(1.0), 1.0);
//! assert!(Wetting::new(0.3, 40.0, 1.55).is_ok());
//! ```

use crate::utils::PhysicsError;

/// Refractive index of liquid water at 589 nm and 20 C.
pub const WATER_REFRACTIVE_INDEX: f64 = 1.333;

/// Density of liquid water at 20 C, kg/m³.
pub const WATER_DENSITY: f64 = 998.2;

/// **Normal-incidence Fresnel reflectance** of the boundary between two dielectrics.
///
/// `((n1 - n2) / (n1 + n2))²`: the share of light a smooth boundary reflects head on.
/// Water against air gives 0.020, the specular reflectance a wet film adds to anything.
///
/// # Arguments
///
/// * `n1` - Refractive index on one side.
/// * `n2` - Refractive index on the other.
///
/// # Returns
///
/// The reflectance, 0 to 1.
///
/// # Examples
///
/// ```
/// use rs_physics::materials::{normal_reflectance, WATER_REFRACTIVE_INDEX};
///
/// let water = normal_reflectance(1.0, WATER_REFRACTIVE_INDEX);
/// assert!((water - 0.0204).abs() < 1e-3);
/// ```
pub fn normal_reflectance(n1: f64, n2: f64) -> f64 {
    let r = (n1 - n2) / (n1 + n2);
    r * r
}

/// **Diffuse light's reflectance at a water-air boundary, from the air side**: the
/// hemispherical average of the Fresnel reflectance for light arriving evenly from every
/// direction of the sky, for n = 1.333 (Lekner and Dorf 1988, table 1).
pub const WATER_DIFFUSE_REFLECTANCE: f64 = 0.066;

/// **Diffuse light's reflectance at a water-air boundary, from inside the water.**
///
/// What a diffusely lit film turns back down onto the surface under it. By reciprocity
/// the light that escapes from inside is the light that enters from outside, divided by
/// n² for the narrower cone it squeezes into: `1 - (1 - R_e) / n²` = 0.474 for water.
/// About half of what a wet diffuse surface scatters upward is sent back down.
pub const fn water_internal_reflectance() -> f64 {
    1.0 - (1.0 - WATER_DIFFUSE_REFLECTANCE) / (WATER_REFRACTIVE_INDEX * WATER_REFRACTIVE_INDEX)
}

/// **The albedo of a diffuse surface under a thin water film** (Angstrom's model as
/// Lekner and Dorf 1988 complete it).
///
/// Light enters the film with probability `1 - R_e`, is scattered by the surface with its
/// albedo `a`, escapes with probability `1 - R_i`, and otherwise is turned back to the
/// surface to be scattered again. Summing the round trips:
///
/// `A = (1 - R_e) * a * (1 - R_i) / (1 - a * R_i)`
///
/// The film's own specular reflection (`R_e`) is not included: it is a mirror term that a
/// renderer draws as gloss, not part of the diffuse albedo.
///
/// # Arguments
///
/// * `albedo` - The dry surface's diffuse albedo, linear (not sRGB-encoded), 0 to 1.
///
/// # Returns
///
/// The wet diffuse albedo, never above the dry one.
///
/// # Examples
///
/// ```
/// use rs_physics::materials::film_albedo;
///
/// // A mid-grey diffuse surface keeps about two thirds of its brightness.
/// let wet = film_albedo(0.5);
/// assert!(wet > 0.30 && wet < 0.36);
/// assert_eq!(film_albedo(0.0), 0.0);
/// ```
pub fn film_albedo(albedo: f64) -> f64 {
    let a = albedo.clamp(0.0, 1.0);
    let r_i = water_internal_reflectance();
    (1.0 - WATER_DIFFUSE_REFLECTANCE) * a * (1.0 - r_i) / (1.0 - a * r_i)
}

/// **The albedo of a thick scattering layer whose scattering is scaled by `ratio`**,
/// through the Kubelka-Munk infinite-layer reflectance.
///
/// The dry albedo fixes the material's absorption-to-scattering ratio,
/// `K/S = (1 - R)² / 2R`; water in the pores leaves `K` as it was and multiplies `S` by
/// `ratio` (see [`Wetting::pore_scattering_ratio`]); the new ratio gives the new albedo,
/// `R = 1 + K/S - sqrt((K/S)² + 2 K/S)`.
///
/// # Arguments
///
/// * `dry_albedo` - The dry material's albedo, linear (not sRGB-encoded), 0 to 1.
/// * `ratio` - Wet scattering over dry, 0 to 1 (1 leaves the albedo as it is).
///
/// # Returns
///
/// The albedo with its pores filled, never above the dry one; 0 stays 0.
///
/// # Examples
///
/// ```
/// use rs_physics::materials::pore_filled_albedo;
///
/// assert!((pore_filled_albedo(0.6, 1.0) - 0.6).abs() < 1e-12);
/// assert!(pore_filled_albedo(0.6, 0.1) < 0.35);
/// assert_eq!(pore_filled_albedo(0.0, 0.1), 0.0);
/// ```
pub fn pore_filled_albedo(dry_albedo: f64, ratio: f64) -> f64 {
    // Black absorbs everything whatever the scattering; without this the clamp below
    // would lift it to a few 1e-8.
    if dry_albedo <= 0.0 {
        return 0.0;
    }
    let r = dry_albedo.clamp(1e-6, 1.0 - 1e-9);
    let ratio = ratio.clamp(1e-6, 1.0);
    if ratio >= 1.0 {
        return dry_albedo.clamp(0.0, 1.0);
    }
    let k_over_s = (1.0 - r) * (1.0 - r) / (2.0 * r) / ratio;
    1.0 + k_over_s - (k_over_s * k_over_s + 2.0 * k_over_s).sqrt()
}

/// **The optical finish of a surface's top boundary**, which decides whether a water film
/// over it darkens it ([`Wetting::wet_albedo`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// A rough or matt surface that scatters light at its top: bare wood, concrete,
    /// cloth, rubber, matt paint. A film over it traps part of that light
    /// ([`film_albedo`]).
    Diffuse,
    /// An optically smooth top boundary, or a metal: glass, gloss paint, and any metal
    /// whatever its roughness (a metal's reflectance comes off its surface rather than
    /// being scattered back from inside, so there is nothing under a film to trap). A film
    /// adds a mirror reflection and darkens nothing.
    Smooth,
}

/// **How water takes to a material**: whether it soaks in, and what it does to the light.
///
/// Mechanical and thermal properties say what a material does when struck or heated; this
/// says what it does when it rains. Kept as an `Option` on [`crate::materials::Material`]
/// for the same reason as its combustion: `None` means not characterised, a different claim
/// from "does not wet". The presets left `None` on purpose are `ice` (water on ice is a
/// melt film, another law), `napalm` (a gel fuel) and `dry_vegetation` (a leaf's cuticle
/// runs from wetting to strongly water-repellent by species and age, so no one value
/// describes it).
///
/// The contact angles on the presets are typical as-received values for each surface;
/// clean, freshly prepared metal is more hydrophilic and a contaminated one less.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wetting {
    /// Open porosity: the volume fraction of connected pores water can fill, 0 to 1.
    /// Zero for a sealed solid (metal, glass, rubber): water stays on the surface.
    pub open_porosity: f64,
    /// Static contact angle of water on a smooth, clean, flat sample, degrees.
    ///
    /// Under 90 the water spreads into a film; over 90 it beads. A porous material with a
    /// low angle wicks water in, which is what "soaks" means.
    pub contact_angle_deg: f64,
    /// Refractive index of the solid phase in the visible (589 nm), for the pore term.
    /// Unused when the material has no open porosity.
    pub solid_index: f64,
    /// The finish of the top boundary: whether a film over it darkens it.
    pub finish: Finish,
}

impl Wetting {
    /// Creates a wetting description with a [`Finish::Diffuse`] top, checked.
    ///
    /// # Arguments
    ///
    /// * `open_porosity` - Volume fraction of connected pores, 0 to 1.
    /// * `contact_angle_deg` - Water contact angle, 0 to 180 degrees.
    /// * `solid_index` - Refractive index of the solid phase, finite and at least 1.
    ///
    /// # Returns
    ///
    /// The description, or a `PhysicsError` naming the value out of range (NaN and the
    /// infinities are out of range).
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Wetting;
    ///
    /// assert!(Wetting::new(0.1, 30.0, 1.55).is_ok());
    /// assert!(Wetting::new(1.5, 30.0, 1.55).is_err());
    /// assert!(Wetting::new(0.1, 30.0, f64::INFINITY).is_err());
    /// ```
    pub fn new(open_porosity: f64, contact_angle_deg: f64, solid_index: f64) -> Result<Self, PhysicsError> {
        if !(0.0..=1.0).contains(&open_porosity) {
            return Err(PhysicsError::CalculationError("Open porosity must be between 0 and 1".to_string()));
        }
        if !(0.0..=180.0).contains(&contact_angle_deg) {
            return Err(PhysicsError::CalculationError("Contact angle must be between 0 and 180 degrees".to_string()));
        }
        if !solid_index.is_finite() || solid_index < 1.0 {
            return Err(PhysicsError::CalculationError("Solid refractive index must be finite and at least 1".to_string()));
        }
        Ok(Self { open_porosity, contact_angle_deg, solid_index, finish: Finish::Diffuse })
    }

    /// A sealed solid: no open pores, water stays on the surface at `contact_angle_deg`.
    /// The finish is [`Finish::Diffuse`]; a metal, glass or gloss coat sets
    /// [`Finish::Smooth`] with [`Wetting::with_finish`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Wetting;
    ///
    /// let sealed = Wetting::sealed(70.0);
    /// assert_eq!(sealed.open_porosity, 0.0);
    /// assert!(!sealed.soaks());
    /// ```
    pub fn sealed(contact_angle_deg: f64) -> Self {
        Self {
            open_porosity: 0.0,
            contact_angle_deg: contact_angle_deg.clamp(0.0, 180.0),
            solid_index: 1.5,
            finish: Finish::Diffuse,
        }
    }

    /// The same description with its top boundary's finish replaced.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::{Finish, Wetting};
    ///
    /// let polished = Wetting::sealed(70.0).with_finish(Finish::Smooth);
    /// assert_eq!(polished.wet_albedo(0.5, 1.0), 0.5);
    /// assert!(Wetting::sealed(70.0).wet_albedo(0.5, 1.0) < 0.5);
    /// ```
    pub fn with_finish(self, finish: Finish) -> Self {
        Self { finish, ..self }
    }

    /// Water held by a cubic metre of the material at saturation, kg: the open pores full.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let canvas = Material::canvas().wetting.unwrap();
    /// assert!(canvas.saturation_water_content() > 500.0);
    /// ```
    pub fn saturation_water_content(&self) -> f64 {
        self.open_porosity * crate::materials::WATER_DENSITY
    }

    /// Whether water beads on a smooth sample (contact angle 90 degrees or more).
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// assert!(Material::rubber().wetting.unwrap().beads());
    /// assert!(!Material::glass().wetting.unwrap().beads());
    /// ```
    pub fn beads(&self) -> bool {
        self.contact_angle_deg >= 90.0
    }

    /// Whether the material soaks water in: open pores that water wets (contact angle
    /// under 90 degrees, so capillary pressure pulls water in rather than keeping it out).
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// assert!(Material::canvas().wetting.unwrap().soaks());
    /// assert!(!Material::steel().wetting.unwrap().soaks());
    /// ```
    pub fn soaks(&self) -> bool {
        self.open_porosity > 0.0 && self.contact_angle_deg < 90.0
    }

    /// **Wet scattering over dry**, at `saturation` (the filled share of the open pores,
    /// 0 to 1).
    ///
    /// Each filled pore's boundary scatters as the Fresnel step from the solid to water
    /// rather than to air, so the filled share scatters `r` times as much, with
    /// `r = R(n_s, n_water) / R(n_s, 1)`: 0.108 for cellulose. 1 for a material that
    /// does not soak.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// let canvas = Material::canvas().wetting.unwrap();
    /// assert!(canvas.pore_scattering_ratio(1.0) < 0.15);
    /// assert_eq!(canvas.pore_scattering_ratio(0.0), 1.0);
    /// ```
    pub fn pore_scattering_ratio(&self, saturation: f64) -> f64 {
        if !self.soaks() {
            return 1.0;
        }
        let dry = normal_reflectance(self.solid_index, 1.0);
        if dry <= 0.0 {
            return 1.0;
        }
        let r = normal_reflectance(self.solid_index, WATER_REFRACTIVE_INDEX) / dry;
        1.0 - saturation.clamp(0.0, 1.0) * (1.0 - r)
    }

    /// **The diffuse albedo of the material wet**, at `saturation` (0 dry, 1 soaked with
    /// a film over it): the pores filled ([`pore_filled_albedo`]), then, on a
    /// [`Finish::Diffuse`] top only, the film ([`film_albedo`]) blended in by the
    /// saturation. A [`Finish::Smooth`] material that does not soak (metal, glass, gloss
    /// paint) keeps its albedo: water adds gloss, a mirror term for the renderer, not a
    /// change of albedo.
    ///
    /// # Arguments
    ///
    /// * `dry_albedo` - The dry albedo, linear (not sRGB-encoded), 0 to 1.
    /// * `saturation` - 0 dry to 1 soaked. NaN and anything at or below 0 read as dry.
    ///
    /// # Returns
    ///
    /// The wet albedo, never above the dry one. At `saturation` 0 it is `dry_albedo`
    /// exactly as given; once wet, an albedo outside 0 to 1 is clamped into it first.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::materials::Material;
    ///
    /// // Undyed cotton canvas, taken as 0.68 dry (an assumed value), soaked: about a
    /// // quarter as bright. Indicative only: a geotextile bag photographed wet below and
    /// // dry above in one frame measures 0.26, a single uncontrolled photograph.
    /// let canvas = Material::canvas().wetting.unwrap();
    /// let ratio = canvas.wet_albedo(0.68, 1.0) / 0.68;
    /// assert!(ratio > 0.22 && ratio < 0.32);
    /// assert_eq!(canvas.wet_albedo(0.68, 0.0), 0.68);
    /// // Steel only gains gloss.
    /// assert_eq!(Material::steel().wetting.unwrap().wet_albedo(0.68, 1.0), 0.68);
    /// ```
    pub fn wet_albedo(&self, dry_albedo: f64, saturation: f64) -> f64 {
        // `!(saturation > 0)` also catches NaN, which `clamp` would pass through.
        if !(saturation > 0.0) {
            return dry_albedo;
        }
        let s = saturation.min(1.0);
        let filled = pore_filled_albedo(dry_albedo.clamp(0.0, 1.0), self.pore_scattering_ratio(s));
        match self.finish {
            Finish::Smooth => filled,
            Finish::Diffuse => filled + (film_albedo(filled) - filled) * s,
        }
    }
}
