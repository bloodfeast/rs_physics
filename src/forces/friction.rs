//! Friction laws as pure functions: Coulomb's bound on an accumulated impulse, and the
//! twist (spin) friction of a contact patch about its normal.
//!
//! These are the laws without a solver around them. A solver that iterates (several passes
//! over the same contact in one step) keeps a running total of the impulse a contact has
//! spent and asks each pass for what it wants now; the functions here say how much of that
//! it may have and what the new total is. Keeping the total rather than the sum of what each
//! pass applied is what lets a pass that reverses the motion give budget back.

/// **Coulomb's bound on a running total**, in one dimension: the impulse to apply this pass
/// and the new total, given the total spent so far, the impulse that would cancel what is
/// left of the motion, and the most the contact may carry.
///
/// The total plus `wanted` is clamped to `[-limit, limit]`, and what is applied is the
/// difference. That is `wanted` exactly while the bound does not bind, and towards the bound
/// rather than past it when it does. A negative or non-finite `limit` is read as zero: a
/// contact that carries nothing holds nothing.
///
/// Returns `(applied, total)`.
#[inline]
pub fn coulomb_clamp(accumulated: f64, wanted: f64, limit: f64) -> (f64, f64) {
    let limit = if limit.is_finite() { limit.max(0.0) } else { 0.0 };
    let total = (accumulated + wanted).clamp(-limit, limit);
    (total - accumulated, total)
}

/// **The most twist friction a straight contact patch can carry**, as an angular impulse:
/// `friction * normal_impulse * half_length / 2`.
///
/// A patch resists turning about its normal with friction spread along it. With the load
/// uniform along a segment of half-length `h`, every element of the patch slides at a speed
/// proportional to its distance from the centre when the patch turns, and Coulomb's force on
/// it acts at that distance, so the resisting moment is `friction * N` times the mean lever,
/// which over `[-h, h]` is `h / 2`. A patch that is a point (`half_length` zero) resists no
/// turn at all, which is right: friction acting at one point has no moment about it.
#[inline]
pub fn twist_friction_limit(friction: f64, normal_impulse: f64, half_length: f64) -> f64 {
    (friction * normal_impulse * half_length * 0.5).max(0.0)
}

/// **Twist friction about a contact normal**: the angular impulse that takes back a turn
/// about the normal, on Coulomb's bound, and the new running total.
///
/// `turn` is the angle the body has turned about the normal that is still to be taken back
/// (radians, signed about the normal), and `ease` is the body's inverse moment of inertia
/// about the normal, so the impulse that would undo the whole turn is `-turn / ease`. The
/// answer is that, clamped by [`coulomb_clamp`] against `limit` (see
/// [`twist_friction_limit`]). The turn the returned impulse makes is `applied * ease`.
///
/// It opposes the turn and never exceeds it: friction stops a spin, it does not reverse one.
/// A body that cannot turn about the normal (`ease` not positive) gets nothing.
///
/// Returns `(applied, total)`.
#[inline]
pub fn twist_friction(turn: f64, ease: f64, accumulated: f64, limit: f64) -> (f64, f64) {
    if !(ease > 1e-12) || turn == 0.0 {
        return (0.0, accumulated);
    }
    coulomb_clamp(accumulated, -turn / ease, limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coulomb_clamp_passes_what_fits_and_stops_at_the_bound() {
        assert_eq!(coulomb_clamp(0.0, 0.3, 1.0), (0.3, 0.3));
        let (applied, total) = coulomb_clamp(0.8, 0.5, 1.0);
        assert!((applied - 0.2).abs() < 1e-15 && total == 1.0);
        assert_eq!(coulomb_clamp(0.0, -2.0, 1.0), (-1.0, -1.0));
    }

    #[test]
    fn coulomb_clamp_gives_budget_back_when_the_motion_reverses() {
        // A pass that over-applied is answered by a later pass asking the other way, and the
        // running total, not the sum of the magnitudes, is what is bounded.
        let (_, total) = coulomb_clamp(0.0, 1.5, 1.0);
        let (applied, total) = coulomb_clamp(total, -0.4, 1.0);
        assert!((applied + 0.4).abs() < 1e-15 && (total - 0.6).abs() < 1e-15);
    }

    #[test]
    fn coulomb_clamp_with_no_budget_holds_nothing() {
        assert_eq!(coulomb_clamp(0.0, 0.5, 0.0), (0.0, 0.0));
        assert_eq!(coulomb_clamp(0.0, 0.5, -1.0), (0.0, 0.0));
        assert_eq!(coulomb_clamp(0.0, 0.5, f64::NAN), (0.0, 0.0));
    }

    #[test]
    fn twist_friction_limit_is_the_mean_lever_of_a_uniform_line_load() {
        // Integrate |x| over a uniform load on [-h, h] numerically and compare.
        let (friction, normal, h) = (0.5, 2.0, 0.25);
        let n = 100_000;
        let mut moment = 0.0;
        for k in 0..n {
            let x = -h + (k as f64 + 0.5) * 2.0 * h / n as f64;
            moment += friction * normal / n as f64 * x.abs();
        }
        assert!((twist_friction_limit(friction, normal, h) - moment).abs() < 1e-9);
        assert_eq!(twist_friction_limit(friction, normal, 0.0), 0.0);
    }

    #[test]
    fn twist_friction_stops_a_turn_it_can_afford_exactly() {
        let ease = 40.0;
        let (applied, total) = twist_friction(-5e-3, ease, 0.0, 1.0);
        assert!((applied * ease - 5e-3).abs() < 1e-15);
        assert_eq!(total, applied);
    }

    #[test]
    fn twist_friction_saturates_and_never_reverses_the_turn() {
        let ease = 40.0;
        let limit = 1e-5;
        let (applied, total) = twist_friction(-5e-3, ease, 0.0, limit);
        assert_eq!((applied, total), (limit, limit));
        // Opposes, and takes back less than the whole turn.
        assert!(applied > 0.0 && applied * ease < 5e-3);
    }

    #[test]
    fn twist_friction_on_a_body_that_cannot_turn_is_nothing() {
        assert_eq!(twist_friction(1e-3, 0.0, 0.25, 1.0), (0.0, 0.25));
        assert_eq!(twist_friction(0.0, 40.0, 0.25, 1.0), (0.0, 0.25));
    }

    #[test]
    fn twist_friction_dissipates_a_spin_over_steps_at_a_fixed_rate() {
        // A free spin, one step at a time, with the impulse it may spend each step: the spin
        // falls by limit * ease a step until it stops, and then stays stopped.
        let (ease, dt, limit) = (40.0, 1.0 / 60.0, 1e-4);
        let mut spin = 0.5_f64;
        let mut steps = 0;
        while spin != 0.0 && steps < 1000 {
            let (applied, _) = twist_friction(spin * dt, ease, 0.0, limit);
            spin += applied * ease / dt;
            if spin.abs() < 1e-12 {
                spin = 0.0;
            }
            steps += 1;
        }
        let per_step = limit * ease / dt;
        assert_eq!(spin, 0.0);
        assert_eq!(steps, (0.5 / per_step).ceil() as usize);
    }
}
