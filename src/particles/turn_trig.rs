//! Sine and cosine of a fraction of a turn, in integer arithmetic, so the CPU and a
//! GPU compute the same bits; see [`sin_cos_turn`].
//!
//! Emission's direction draw used `f32::sin_cos`, which is the platform's maths
//! library: the Windows UCRT, glibc and the macOS libm give different last bits, and
//! none of them is correctly rounded (the UCRT is 1 ulp off the correctly rounded value
//! on 43,448 of the 2^24 azimuths a draw can take). A shader's `sin` and `cos` are looser
//! still (2^-11 absolute on Vulkan). Neither can be reproduced bit for bit elsewhere.
//! Integer arithmetic can: every add, shift and 32-bit multiply below is exact on any
//! CPU and any GPU, and the one conversion to `f32` at the end is of an integer below
//! 2^24, which is exact too.

#![warn(missing_docs)]

/// Bits of the turn fraction [`sin_cos_turn`] reads: a whole turn is `2^24`.
pub const TURN_BITS: u32 = 24;

/// Bits of a quarter turn in the same units.
const QUARTER_BITS: u32 = TURN_BITS - 2;

/// Intervals of the quarter-wave table: `2^12`.
const TABLE_BITS: u32 = 12;

/// Bits of the position inside one table interval, interpolated linearly.
const FRACTION_BITS: u32 = QUARTER_BITS - TABLE_BITS;

/// Entries of [`QUARTER_SINE`]: the `2^12 + 1` knots from 0 to a quarter turn, and a
/// copy of the last so the interpolation of the end point reads in bounds with no
/// branch.
pub(crate) const QUARTER_SINE_LEN: usize = (1 << TABLE_BITS) + 2;

/// The scale of the integers [`sin_cos_turn`] interpolates: 1.0 is `2^24`.
const ONE: f32 = 16_777_216.0;

/// `round(2^24 sin(i pi / 2^13))` for `i` in `0..=2^12`: the sine over a quarter turn in
/// `2^12` intervals, at the scale where 1.0 is `2^24`, then the end knot again.
///
/// Built at compile time in integer arithmetic (a Taylor series in 61-bit fixed point,
/// error below `2^-55`), not from the platform's `f64::sin`, so the table is the same
/// on every build. The GPU pool uploads this array as it stands.
pub(crate) static QUARTER_SINE: [i32; QUARTER_SINE_LEN] = quarter_sine();

const fn quarter_sine() -> [i32; QUARTER_SINE_LEN] {
    // round(pi 2^61).
    const PI_Q61: i128 = 7_244_019_458_077_122_842;
    let mut table = [0i32; QUARTER_SINE_LEN];
    let mut i = 0;
    while i <= 1 << TABLE_BITS {
        // The knot's angle, i pi / 2^13, in Q61: below 2^61 pi / 2.
        let x = (i as i128 * PI_Q61) >> (TABLE_BITS + 1);
        let x2 = (x * x) >> 61;
        // sin x = x - x^3/3! + x^5/5! - ...; each term from the last by -x^2 / (n+1)(n+2).
        let mut term = x;
        let mut sum = 0i128;
        let mut n = 1i128;
        while term != 0 {
            sum += term;
            term = -((term * x2) >> 61) / ((n + 1) * (n + 2));
            n += 2;
        }
        // Q61 to Q24, to nearest.
        table[i] = ((sum + (1 << 36)) >> 37) as i32;
        i += 1;
    }
    table[QUARTER_SINE_LEN - 1] = table[QUARTER_SINE_LEN - 2];
    table
}

/// The sine of a position `p` in `0..=2^22` (a quarter turn), at the scale where 1.0 is
/// `2^24`: the table's knot below and the linear step to the next, rounded to nearest.
#[inline(always)]
fn quarter(p: u32) -> i32 {
    let i = (p >> FRACTION_BITS) as usize;
    let f = (p & ((1 << FRACTION_BITS) - 1)) as i32;
    let a = QUARTER_SINE[i];
    let b = QUARTER_SINE[i + 1];
    // b - a is at most 2^24 pi / 2^13 (13 bits), f at most 10 bits: the product fits.
    a + (((b - a) * f + (1 << (FRACTION_BITS - 1))) >> FRACTION_BITS)
}

/// The sine and cosine of `turn / 2^24` of a full turn, the same bits on every CPU and
/// in the GPU particle pool's shader.
///
/// The angle is a fraction of a turn in fixed point, as an emission draw yields it
/// (`EffectRng::next_u32() >> 8`), so there is no multiply by `2 pi` and no rounding of
/// the angle. The top two bits pick the quadrant; within it a quarter-wave table of
/// `2^12` intervals (16 KB, built in integer arithmetic at compile time) is read at the
/// position and the next knot and interpolated linearly in integers, and the quadrant's
/// symmetry swaps and negates the pair with bit masks, no branch. Everything is `i32`
/// arithmetic, exact on any CPU or GPU, so the result is identical across Windows,
/// Linux, macOS and the device. Only the last step converts: the integers are at most
/// `2^24` in magnitude, so `as f32` is exact, and the scale by `2^-24` is a power of two.
///
/// # Precision
///
/// At most `1.293 * 2^-24` (7.7e-8) from `f64` `sin` and `cos` of `2 pi turn / 2^24`,
/// absolute, over all `2^24` inputs (measured exhaustively; the table's rounding is
/// half a unit, the interpolation `(pi / 2^13)^2 / 8` of a unit, and the step's rounding
/// half a unit). The error is absolute, not relative: near a zero of the sine the result
/// has fewer significant bits than an `f32` holds, which no direction drawn from it can
/// show. `f32::sin_cos` of the same angle (after the angle's own rounding in
/// `unit() * TAU`) was up to `6.9 * 2^-24` from the exact value on the same sweep.
///
/// # Cost
///
/// 4.7 ns a call on the CPU against 21.0 ns for `f32::sin_cos` (Windows, i9-10980XE,
/// `--release`, 2026-10-03, beside another project's test run). A 2^11-interval table
/// measured 4.7 ns too, and a 26-step CORDIC (shifts and adds only) 42.6 ns.
///
/// # Arguments
///
/// * `turn` - the angle in units of `2^-24` of a turn; only the low 24 bits are read.
///
/// # Returns
///
/// `(sin, cos)`, each a multiple of `2^-24` in `[-1, 1]`.
///
/// # Examples
///
/// ```
/// use rs_physics::particles::sin_cos_turn;
/// // A quarter turn: exactly (1, 0).
/// assert_eq!(sin_cos_turn(1 << 22), (1.0, 0.0));
/// // An eighth: both 1/sqrt(2), to the routine's 1.3e-7.
/// let (s, c) = sin_cos_turn(1 << 21);
/// assert!((s - core::f32::consts::FRAC_1_SQRT_2).abs() < 1.3e-7);
/// assert_eq!(s, c);
/// ```
#[inline]
pub fn sin_cos_turn(turn: u32) -> (f32, f32) {
    let q = (turn >> QUARTER_BITS) & 3;
    let p = turn & ((1 << QUARTER_BITS) - 1);
    let s = quarter(p);
    let c = quarter((1 << QUARTER_BITS) - p);
    // Quadrant 1 and 3 swap the pair; 2 and 3 negate the sine, 1 and 2 the cosine.
    let swap = ((q & 1) as i32).wrapping_neg();
    let (a, b) = (s ^ ((s ^ c) & swap), c ^ ((s ^ c) & swap));
    let neg_s = ((q >> 1) as i32).wrapping_neg();
    let neg_c = (((q ^ (q >> 1)) & 1) as i32).wrapping_neg();
    let sin = (a ^ neg_s).wrapping_sub(neg_s);
    let cos = (b ^ neg_c).wrapping_sub(neg_c);
    (sin as f32 * (1.0 / ONE), cos as f32 * (1.0 / ONE))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every knot is the correctly rounded value: within half a unit of `f64`'s sine
    /// (whose own error is about `2^-29` of a unit here).
    #[test]
    fn the_table_is_the_rounded_quarter_sine() {
        for (i, &v) in QUARTER_SINE[..QUARTER_SINE_LEN - 1].iter().enumerate() {
            let exact = (i as f64 * core::f64::consts::PI / 8192.0).sin() * ONE as f64;
            assert!(
                (v as f64 - exact).abs() <= 0.5 + 1e-6,
                "knot {i}: {v} against {exact}"
            );
        }
        assert_eq!(QUARTER_SINE[0], 0);
        assert_eq!(QUARTER_SINE[1 << TABLE_BITS], 1 << 24);
        assert_eq!(QUARTER_SINE[QUARTER_SINE_LEN - 1], 1 << 24);
    }

    /// All `2^24` inputs against `f64`: the documented bound, and the symmetries the
    /// quadrant masks are meant to give exactly.
    #[test]
    fn every_turn_is_within_the_documented_bound() {
        let bound = 1.293 / ONE as f64 + 1e-12;
        let (mut worst_s, mut worst_c) = (0f64, 0f64);
        for turn in 0..1u32 << TURN_BITS {
            let (s, c) = sin_cos_turn(turn);
            let a = turn as f64 / ONE as f64 * core::f64::consts::TAU;
            worst_s = worst_s.max((s as f64 - a.sin()).abs());
            worst_c = worst_c.max((c as f64 - a.cos()).abs());
            // Half a turn on is the negation, to the bit (zero keeps its sign as +0).
            let (s2, c2) = sin_cos_turn(turn + (1 << 23));
            assert_eq!((s2, c2), (-s + 0.0, -c + 0.0), "turn {turn}");
        }
        println!(
            "sin_cos_turn: worst {:.4} (sin) and {:.4} (cos) x 2^-24 from f64",
            worst_s * ONE as f64,
            worst_c * ONE as f64
        );
        assert!(worst_s <= bound && worst_c <= bound);
    }

    #[test]
    fn only_the_low_24_bits_are_read() {
        assert_eq!(sin_cos_turn(0xFF00_0123), sin_cos_turn(0x0000_0123));
    }
}
