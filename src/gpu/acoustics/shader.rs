//! The WGSL, generated: every constant the shaders use comes from the f64 CPU laws or the
//! records, formatted as the f32 the GPU will hold, so nothing is typed twice.

use std::fmt::Write;

use super::pack::{HEADER_FIELDS, PYRAMID_LEVELS};
use super::records::*;
use crate::acoustics::{self, band, spatial, surfaces};

/// Word offset of the results in the output buffer.
pub(crate) const OUT_SOURCES: u32 = 0;
/// Word offset of the listener field in the output buffer.
pub(crate) const OUT_FIELD: u32 = MAX_SOURCES * 8;
/// Word offset of the per-ray diagnostics in the output buffer.
pub(crate) const OUT_RAYS: u32 = OUT_FIELD + 36;
/// Word offset of the per-source main edges in the output buffer.
pub(crate) const OUT_EDGES: u32 = OUT_RAYS + 16 * MAX_FIELD_RAYS;
/// Words in the output buffer.
pub(crate) const OUT_WORDS: u32 = OUT_EDGES + 8 * MAX_SOURCES;

/// Seconds a tap bin spans: arrivals closer than this are one early reflection to the ear.
pub const TAP_BIN_S: f32 = 0.005;
/// Tap bins: 64 of 5 ms, a 320 ms window. A field ray is marched to the distance whose
/// first-order echo lands at the window's end, `window x c / 2` (about 55 m at 343 m/s);
/// a ray that meets nothing by then is counted as escaped.
pub const TAP_BINS: u32 = 64;

/// How far past a surface a reflected ray starts, and how far from the listener the first
/// ray does, in metres: a millimetre, so a ray never re-hits the face it left.
const T_START_M: f32 = 1e-3;

/// A floor under `|cos|` in the surface-area quadrature, only to keep a division finite
/// for a ray exactly parallel to the face it is reported to hit.
const COS_FLOOR: f32 = 1e-4;

/// Bernoulli numbers B_2 to B_18 as exact rationals, for the series of `x coth x` and
/// `x cot x`: mathematics, not tuning.
const BERNOULLI: [(f64, f64); 9] = [
    (1.0, 6.0),
    (1.0, 30.0),
    (1.0, 42.0),
    (1.0, 30.0),
    (5.0, 66.0),
    (691.0, 2730.0),
    (7.0, 6.0),
    (3617.0, 510.0),
    (43867.0, 798.0),
];

/// `|c_n| = 2^(2n) |B_2n| / (2n)!`, the magnitude of the n-th coefficient of both
/// `x coth x = 1 + sum (-1)^(n+1) c_n x^(2n)` and `x cot x = 1 - sum c_n x^(2n)`,
/// for n = 1 to 9.
pub fn series_coefficients() -> [f64; 9] {
    let mut out = [0.0; 9];
    for (k, (num, den)) in BERNOULLI.iter().enumerate() {
        let n = k as i32 + 1;
        let mut fact = 1.0f64;
        for m in 1..=(2 * n) {
            fact *= m as f64;
        }
        out[k] = 2f64.powi(2 * n) * num / den / fact;
    }
    out
}

/// Terms of `x coth x` kept under x = 1, and of `x cot x` on the lit zone. The remainder
/// of each is under half an f32 ulp of 1 (the oracle L1 computes it).
pub const COTH_TERMS: usize = 7;
/// See [`COTH_TERMS`].
pub const COT_TERMS: usize = 8;

/// Abramowitz and Stegun 4.4.46: `acos x = sqrt(1 - x) (a0 + a1 x + ... + a7 x^7)` on
/// `[0, 1]`, with an absolute error of at most 2e-8. The published coefficients.
pub const ASIN_COEFFICIENTS: [f64; 8] = [
    1.570_796_305_0,
    -0.214_598_801_6,
    0.088_978_987_4,
    -0.050_174_304_6,
    0.030_891_881_0,
    -0.017_088_125_6,
    0.006_670_090_1,
    -0.001_262_491_1,
];

/// The published bound of [`ASIN_COEFFICIENTS`], radians.
pub const ASIN_ERROR: f64 = 2e-8;

/// The field's 64 directions and the solid angle each stands for, `[x, y, z, sr]`.
///
/// # Derivation
///
/// A listener stands 1.6 m above the ground, and what reflects sound back to it (terrain,
/// walls, vehicles) lies near the horizon; straight up is sky. So the sphere is split into
/// a band within +-20 degrees of horizontal and two caps, and the band gets three quarters
/// of the rays:
///
/// * **The band, 48 rays:** four rings of 12, each ring at the centre, in `sin(elevation)`,
///   of an equal-area quarter of the band (about -14.9, -4.9, 4.9 and 14.9 degrees), with
///   alternate rings turned by 15 degrees so no two rings share an azimuth.
/// * **Each cap, 8 rays:** two rings of 4 at the centres of equal-area halves of the cap
///   (about 30.4 and 56.7 degrees from the horizon), the second turned by 45 degrees.
///
/// Stratifying by equal area makes each ray stand for the solid angle of its stratum,
/// `4 pi sin(20 deg) / 48` in the band and `2 pi (1 - sin(20 deg)) / 8` in a cap, and those
/// weights are what the field's sums are weighted by, so the estimates stay unbiased
/// despite the uneven density. The weights sum to `4 pi`.
///
/// # Returns
///
/// The table, in the world frame (y up): it does not turn with the listener, so the field
/// does not flicker as the camera yaws.
///
/// # Examples
///
/// ```
/// use rs_physics::gpu::acoustics::field_directions;
///
/// let dirs = field_directions();
/// let total: f64 = dirs.iter().map(|d| d[3]).sum();
/// assert!((total - 4.0 * std::f64::consts::PI).abs() < 1e-9);
/// let near_horizon = dirs.iter().filter(|d| d[1].abs() <= 20f64.to_radians().sin()).count();
/// assert_eq!(near_horizon, 48);
/// ```
pub fn field_directions() -> [[f64; 4]; 64] {
    use std::f64::consts::PI;
    let s20 = 20f64.to_radians().sin();
    let mut out = [[0.0; 4]; 64];
    let mut k = 0;
    let band_w = 4.0 * PI * s20 / 48.0;
    for ring in 0..4 {
        let s = -s20 + (ring as f64 + 0.5) * (2.0 * s20 / 4.0);
        let cos_el = (1.0 - s * s).sqrt();
        for j in 0..12 {
            let az = (j as f64 + 0.5 * (ring % 2) as f64) * (2.0 * PI / 12.0);
            out[k] = [cos_el * az.cos(), s, cos_el * az.sin(), band_w];
            k += 1;
        }
    }
    let cap_w = 2.0 * PI * (1.0 - s20) / 8.0;
    for sign in [1.0, -1.0] {
        for ring in 0..2 {
            let s = s20 + (ring as f64 + 0.5) * (1.0 - s20) / 2.0;
            let cos_el = (1.0 - s * s).sqrt();
            for j in 0..4 {
                let az = (j as f64 + 0.5 * ring as f64) * (PI / 2.0);
                out[k] = [cos_el * az.cos(), sign * s, cos_el * az.sin(), cap_w];
                k += 1;
            }
        }
    }
    out
}

fn f(v: f64) -> String {
    format!("{:?}", v as f32)
}

fn vec4(v: [f64; 4]) -> String {
    format!("vec4<f32>({}, {}, {}, {})", f(v[0]), f(v[1]), f(v[2]), f(v[3]))
}

fn prelude(out: &mut String) {
    for (name, word) in HEADER_FIELDS {
        writeln!(out, "const {name}: u32 = {word}u;").unwrap();
    }
}

/// The query module's source: the generated constants, then `query.wgsl`.
pub(crate) fn query_source() -> String {
    let mut s = String::with_capacity(32 * 1024);
    prelude(&mut s);
    let k = law_constants();
    let series = series_coefficients();
    let c = |n: &str, v: String, s: &mut String| writeln!(s, "const {n} = {v};").unwrap();
    c("OUT_SOURCES", format!("{OUT_SOURCES}u"), &mut s);
    c("OUT_FIELD", format!("{OUT_FIELD}u"), &mut s);
    c("OUT_RAYS", format!("{OUT_RAYS}u"), &mut s);
    c("OUT_EDGES", format!("{OUT_EDGES}u"), &mut s);
    c("MAX_STATICS", format!("{MAX_STATICS}u"), &mut s);
    c("FIELD_RAYS", format!("{MAX_FIELD_RAYS}u"), &mut s);
    c("MAX_LANE_STEPS", format!("{}u", 2 * MAX_TERRAIN_SIDE / 64 + 4), &mut s);
    c("MAX_BLOCK_STEPS", format!("{}u", 2 * MAX_TERRAIN_SIDE / 8 + 4), &mut s);
    c("MAX_CELL_STEPS", format!("{}u", 2 * 8 + 4), &mut s);
    c("MAX_LEGIBILITY", format!("{MAX_LEGIBILITY_POINTS}u"), &mut s);
    c("PYRAMID_TOP", format!("{PYRAMID_LEVELS}u"), &mut s);
    c("BAND_INV_HZ", vec4(BANDS_HZ.map(|hz| 1.0 / hz as f64)), &mut s);
    c("PROBE_INV_HZ", f(1.0 / PROBE_HZ as f64), &mut s);
    c("FOLIAGE_DB_PER_M", vec4(k.foliage_db_per_m), &mut s);
    c("FOLIAGE_MAX_M", f(surfaces::FOLIAGE_MAX_CREDITED_M), &mut s);
    c("BARRIER_GRAZING_DB", f(surfaces::BARRIER_GRAZING_DB), &mut s);
    c("BARRIER_CAP_DB", f(surfaces::BARRIER_CAP_DB), &mut s);
    let n0 = surfaces::lit_zone_limit();
    c("LIT_N0", f(n0), &mut s);
    c("LIT_X0", f((2.0 * std::f64::consts::PI * n0).sqrt()), &mut s);
    c("TWO_PI", f(2.0 * std::f64::consts::PI), &mut s);
    c("HALF_PI", f(std::f64::consts::FRAC_PI_2), &mut s);
    c("DB_PER_LOG2", f(20.0 * 2f64.log10()), &mut s);
    c("LOG2_E", f(std::f64::consts::LOG2_E), &mut s);
    c("LOG2_10_OVER_10", f(10f64.log2() / 10.0), &mut s);
    for n in 0..COTH_TERMS {
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        c(&format!("COTH_C{}", n + 1), f(sign * series[n]), &mut s);
    }
    for n in 0..COT_TERMS {
        c(&format!("COT_C{}", n + 1), f(-series[n]), &mut s);
    }
    c("FIT_W", vec4(k.fit_weights), &mut s);
    c("FIT_X_MEAN", f(k.fit_x_mean), &mut s);
    c("FIT_REFERENCE_HZ", f(band::FIT_REFERENCE_HZ), &mut s);
    c("NO_LOWPASS_HZ", f(band::NO_LOWPASS_HZ), &mut s);
    c("REFERENCE_M", f(acoustics::REFERENCE_M), &mut s);
    c("HEAD_RADIUS_M", f(spatial::HEAD_RADIUS_M), &mut s);
    c("SHADOW_K", f(shadow_k()), &mut s);
    c("DOPPLER_MAX", f(acoustics::DOPPLER_MAX_CLOSING), &mut s);
    for (i, a) in ASIN_COEFFICIENTS.iter().enumerate() {
        c(&format!("ASIN_A{i}"), f(*a), &mut s);
    }
    let dirs = field_directions();
    let mut table = String::from("array<vec4<f32>, 64>(");
    for (i, d) in dirs.iter().enumerate() {
        if i > 0 {
            table.push_str(", ");
        }
        table.push_str(&vec4(*d));
    }
    table.push(')');
    c("FIELD_DIR_TABLE", table, &mut s);
    let total: f64 = dirs.iter().map(|d| d[3] as f32 as f64).sum();
    c("FIELD_TOTAL_WEIGHT", f(total), &mut s);
    c("TAP_BIN_S", f(TAP_BIN_S as f64), &mut s);
    c("TAP_BINS", format!("{TAP_BINS}u"), &mut s);
    c("TAP_WINDOW_S", f(TAP_BIN_S as f64 * TAP_BINS as f64), &mut s);
    c("T_START", f(T_START_M as f64), &mut s);
    c("COS_FLOOR", f(COS_FLOOR as f64), &mut s);
    c("SABINE_LN", f(24.0 * 10f64.ln()), &mut s);
    s.push_str(include_str!("query.wgsl"));
    s
}

/// The build module's source.
pub(crate) fn build_source() -> String {
    let mut s = String::with_capacity(8 * 1024);
    prelude(&mut s);
    writeln!(s, "const MAX_CELLS_PER_STATIC = {MAX_CELLS_PER_STATIC}u;").unwrap();
    let chunk = (MAX_TERRAIN_SIDE * MAX_TERRAIN_SIDE).div_ceil(256);
    writeln!(s, "const MAX_SCAN_CHUNK = {chunk}u;").unwrap();
    s.push_str(include_str!("build.wgsl"));
    s
}

/// `0.9 x (f/f0)^2 / (1 + (f/f0)^2)` at the probe band: the far ear's shadow per unit of
/// lateral cosine, from [`spatial::Ears::hear`]'s own law.
pub fn shadow_k() -> f64 {
    let ears = spatial::Ears::at([0.0; 3]);
    let air = acoustics::Air::standard();
    // Read the law off the CPU function rather than restating it: a source dead right has
    // |beside| = 1, so its far-ear gain is 1 - k.
    let side = ears.hear([10.0, 0.0, 0.0], &air, PROBE_HZ as f64);
    1.0 - side.left_gain
}
