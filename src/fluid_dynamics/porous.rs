//! Porous absorption: how much of a liquid a porous bed takes in, and what it refuses.
//!
//! Two laws on a pore volume stored as a byte of fill (`0` dry, `ceiling` saturated):
//!
//! - [`soak_uptake`]: an offer of liquid is taken in as a share of the headroom left,
//!   `offered / 255` of it, so a dry bed drinks most of an offer and a nearly full one
//!   almost none of it. That compression toward saturation is what makes the fill
//!   readable as a history (a patch that took one fight and one that took six differ).
//! - [`infiltration`]: a film *running over* the bed loses to it a share of what arrives
//!   in proportion to the pore capacity left, spread over `passes` arrivals. It is what
//!   gives a trail a length: it stains the ground it crosses and shortens as it goes, and
//!   only reaches far once the ground ahead is full.
//!
//! [`refused`] is the remainder of an offer, which a caller stands on top of the bed.
//! A ceiling of zero (a sealed surface, concrete) takes nothing: every offer is refused.
//!
//! Integer and branch-light, because they run per texel inside a flow step. Moved here from
//! Ridgeline's `terrain_stain` (STAIN-MOVE); the arithmetic is unchanged.

/// The fill after offering `offered` units to a bed holding `held` of `ceiling`.
///
/// Takes `offered / 255` of the headroom `ceiling - held`, never past `ceiling`; a bed
/// already at or over its ceiling takes nothing and keeps what it holds.
#[inline]
pub fn soak_uptake(held: u8, offered: u32, ceiling: u8) -> u8 {
    if held >= ceiling {
        return held;
    }
    let headroom = (ceiling as u32) - held as u32;
    (held as u32 + (offered * headroom) / u8::MAX as u32).min(ceiling as u32) as u8
}

/// What a bed refused of an offer: the offer less what the fill actually rose by.
///
/// `before` and `after` are the fill either side of [`soak_uptake`] (`after >= before`).
#[inline]
pub fn refused(offered: u32, before: u8, after: u8) -> u32 {
    offered.saturating_sub(after.saturating_sub(before) as u32)
}

/// Units a bed drinks from `arrived` units of a film running over it.
///
/// The share is the pore capacity left, `min(ceiling, capacity) - held`, over
/// `capacity * passes`: a dry bed drinks `1 / passes` of what arrives, a full one none.
/// `capacity` is the fill scale the share is measured against (the bed's nominal
/// saturation), `ceiling` this bed's own limit under it. Truncates toward zero, and is
/// zero for a non-positive `arrived`.
#[inline]
pub fn infiltration(arrived: i32, held: u8, ceiling: u8, capacity: u8, passes: i32) -> i32 {
    let dry = ceiling.min(capacity).saturating_sub(held) as i32;
    if arrived <= 0 || capacity == 0 || passes <= 0 {
        return 0;
    }
    arrived * dry / (capacity as i32 * passes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dry_bed_takes_the_offer_as_a_share_of_its_headroom() {
        // 255 offered to an empty full-scale bed fills it.
        assert_eq!(soak_uptake(0, 255, 255), 255);
        // Half the scale offered takes half the headroom (truncated).
        assert_eq!(soak_uptake(0, 128, 255), 128);
        assert_eq!(soak_uptake(100, 128, 255), 100 + (128 * 155) / 255);
    }

    #[test]
    fn uptake_compresses_toward_saturation_and_never_passes_the_ceiling() {
        let mut fill = 0u8;
        let mut gains = Vec::new();
        for _ in 0..20 {
            let next = soak_uptake(fill, 40, 200);
            gains.push(next - fill);
            fill = next;
            assert!(fill <= 200);
        }
        assert!(gains.windows(2).all(|w| w[1] <= w[0]), "gains must fall: {gains:?}");
        assert_eq!(soak_uptake(200, 255, 200), 200);
        assert_eq!(soak_uptake(250, 255, 200), 250, "over the ceiling keeps what it holds");
    }

    #[test]
    fn a_sealed_bed_refuses_everything() {
        let after = soak_uptake(0, 90, 0);
        assert_eq!(after, 0);
        assert_eq!(refused(90, 0, after), 90);
        assert_eq!(infiltration(1_000_000, 0, 0, 235, 3), 0);
    }

    #[test]
    fn what_is_refused_is_what_did_not_go_in() {
        for held in [0u8, 50, 200, 235] {
            for offered in [0u32, 1, 17, 255, 2000] {
                let after = soak_uptake(held, offered, 235);
                let took = (after - held) as u32;
                // The bed never takes more than it was offered, so nothing is lost or
                // made: refused plus taken is the offer.
                assert!(took <= offered);
                assert_eq!(refused(offered, held, after) + took, offered);
            }
        }
    }

    #[test]
    fn a_running_film_loses_a_share_set_by_the_capacity_left() {
        // Dry bed: one pass-th of what arrives.
        assert_eq!(infiltration(3 * 235 * 100, 0, 255, 235, 3), 235 * 100);
        // Half full: half that.
        let half = infiltration(3 * 235 * 100, 117, 255, 235, 3);
        assert!((half - 118 * 100).abs() <= 1, "{half}");
        // Full or over: nothing.
        assert_eq!(infiltration(1_000_000, 235, 255, 235, 3), 0);
        assert_eq!(infiltration(1_000_000, 240, 255, 235, 3), 0);
        // A lower ceiling caps what is left.
        assert_eq!(infiltration(3 * 235, 0, 100, 235, 3), 100);
        // Nothing arriving, nothing drunk.
        assert_eq!(infiltration(0, 0, 255, 235, 3), 0);
        assert_eq!(infiltration(-5, 0, 255, 235, 3), 0);
    }
}
