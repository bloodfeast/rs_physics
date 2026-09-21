//! Writing a colour's corrections straight into the body arrays, from the threads that
//! computed them.
//!
//! # What this replaces, and the measurement that demanded it
//!
//! The first version of the colour solve had two halves: a parallel one that computed
//! every correction in the colour into a buffer, and a serial one that walked the buffer
//! and applied them. That is safe without any of this, and on a heap of ten thousand
//! bodies it cost more than the arithmetic it was protecting.
//!
//! ```text
//!   contact pass, one colour set, 37,600 contacts
//!     the arithmetic, one thread          3.26 ms
//!     the same, parallel, into a buffer   4.90 ms
//!     the serial half that applies it     3.11 ms
//! ```
//!
//! The parallel half was *slower than doing the whole thing on one thread*. An answer is
//! two [`Correction`]s and a running impulse -- 272 bytes -- so a pass wrote ten megabytes
//! into the buffer and the serial half read them all back, and neither half was doing any
//! more useful work for it. Ten megabytes a pass, eight passes a step.
//!
//! Applying in place deletes both numbers at once: the correction is written where it was
//! computed, by the thread that computed it, and no buffer exists.
//!
//! # Why this is safe, precisely
//!
//! Every constraint here names its bodies by index and touches no others. A **colour** is
//! a set of constraints in which no two name the same body -- that is the definition the
//! colouring is built to satisfy, and it is the same invariant that already made the old
//! serial half's *order* irrelevant. If two corrections in one colour could land on one
//! body, the previous version's answer already depended on which of them was applied
//! last, and the module would have been wrong before any of this.
//!
//! So, for a colour solved through the types here:
//!
//! * **Bodies are partitioned, not shared.** Constraint `k` reads and writes bodies `a`
//!   and `b` and nothing else; every other constraint in the colour names a disjoint
//!   pair. No two threads address the same element of any of the four body arrays, so
//!   there is no data race and no ordering to depend on.
//! * **The running impulses are addressed by something unique to the constraint within
//!   the colour**, so each is touched by exactly one thread. For pair contacts that is
//!   the contact's own index. For ground contacts it is the **body**, because the two
//!   ends of a capsule on the plane pool one Coulomb budget -- see
//!   [`Skeleton::build_contacts`] -- and that is sound for exactly the reason the body
//!   arrays are: within a ground set each body appears at most once, which is the
//!   property [`disjoint`] already checks there, since it walks the set's bodies rather
//!   than its contacts. A change that put both of a body's ground contacts in one set
//!   would race the budget as well as the body, and would fail that same check.
//! * **Nothing is read through a shared reference while it is written.** The arrays are
//!   reached only through [`Cells`], which reads and writes elements through a raw
//!   pointer and never forms a `&` or `&mut` over the buffer. The read-only arrays --
//!   inverse mass, inverse inertia, radius -- are ordinary slices, because nothing here
//!   writes them.
//! * **Indices are in range.** They came from the body count the arrays were sized to,
//!   and [`Cells`] carries the length and checks it in debug builds.
//!
//! # Which code establishes it, and what would break it
//!
//! The invariant is not a property of the types; it is produced by two functions, and a
//! change to either of them is what would make everything here unsound:
//!
//! * [`Skeleton::add_joint`] partitions the joints. It gives each joint the lowest colour
//!   neither of its bodies is already using, so a colour names each body at most once.
//!   It runs as each joint arrives, and the assignment is never revisited: adding an edge
//!   to a proper edge-colouring leaves it proper.
//! * [`Skeleton::colour_contacts`] does the same for the contacts, every step, with a bit
//!   per colour per body. A contact whose two bodies have between them used all
//!   sixty-four bits goes to `contact_overflow` instead, **which is solved one at a time
//!   on the calling thread** and never through a parallel colour.
//!
//! A solve may be handed a *restriction* of a colour rather than the whole of it -- the
//! constraints touching an awake body, foreground first, which is what
//! [`Skeleton::find_live_joints`] and the two-pass half of [`Skeleton::colour_contacts`]
//! produce. That cannot break the invariant, and it is worth saying why rather than
//! leaving it to be noticed: a subset of a set in which no body appears twice is still
//! such a set, and reordering one is a permutation, which is also still such a set.
//! [`disjoint`] runs on the slice that is actually solved, so the check follows the
//! restriction rather than trusting it.
//!
//! If either grew a case that put two constraints on one body in one colour -- a colour
//! cap that assigned the last colour rather than overflowing, say, or a merge of two
//! colours to cut the fork count -- then two threads would read-modify-write one body and
//! this would be a data race. Neither function may do that without this file changing
//! with it.
//!
//! So it is checked rather than trusted, three ways:
//!
//! * **At runtime under `cfg(debug_assertions)`**: [`disjoint`] walks the colour before it
//!   is solved and panics if a body appears twice, so a debug test run fails loudly where
//!   a release build would race quietly. Every parallel colour goes through it.
//! * **By tests whose whole purpose is the invariant**:
//!   `no_two_joints_in_a_colour_share_a_body` on a branching skeleton and
//!   `no_two_contacts_in_a_colour_share_a_body` on a pile, both asserting the property
//!   directly rather than exercising a happy path.
//! * **By a determinism test.** The order corrections are applied in is now whatever the
//!   thread pool chose, so if the bodies in a colour were *not* disjoint the arithmetic
//!   would depend on that order and float addition is not associative.
//!   `the_same_crowd_twice_lands_on_the_same_bits` runs one simulation twice in one
//!   process and requires every `f64` to match to the bit, over a workload it asserts is
//!   big enough to have gone through the pool. A scheduling-dependent answer fails it.
//!
//! Nothing here accumulates across constraints, which is why bit-determinism is available
//! at all: a correction is read, composed and written by one thread, and no two threads
//! contribute to one sum.

use super::*;

/// Panics unless `touched` names each body at most once -- which is what every unsafe
/// block in this file rests on. Compiled out of release builds; see the module header for
/// which functions are supposed to guarantee it.
///
/// A flag per body rather than a set, because this runs before every parallel colour of
/// every pass of every step in a debug test run, and a hash per constraint would make the
/// suite too slow to keep it switched on.
#[cfg(debug_assertions)]
pub(super) fn disjoint(bodies: usize, what: &str, touched: impl Iterator<Item = usize>) {
    let mut seen = vec![false; bodies];
    for body in touched {
        assert!(
            body < bodies,
            "a {what} colour names body {body}, which is outside the {bodies} there are",
        );
        assert!(
            !seen[body],
            "a {what} colour names body {body} twice; solving it in parallel is a data \
             race, and the colouring that produced it is broken",
        );
        seen[body] = true;
    }
}

/// One array of body-indexed state, reachable from several threads at once **on the
/// understanding that they address disjoint elements**. See the module header.
///
/// Deliberately not `&[UnsafeCell<T>]`: that would make every ordinary sweep over the
/// same arrays go through `UnsafeCell` as well, for the benefit of the one place that
/// needs it.
pub(super) struct Cells<T> {
    at: *mut T,
    len: usize,
}

impl<T> Clone for Cells<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Cells<T> {}

// SAFETY: the pointer is a plain `*mut T` into a buffer the caller owns for the lifetime
// of the parallel region, and the disjointness argument in the module header is what
// makes sharing it across threads sound.
unsafe impl<T: Send> Send for Cells<T> {}
unsafe impl<T: Send> Sync for Cells<T> {}

impl<T: Copy> Cells<T> {
    pub(super) fn of(slice: &mut [T]) -> Self {
        Cells {
            at: slice.as_mut_ptr(),
            len: slice.len(),
        }
    }

    /// # Safety
    /// `i` must be in range, and no other thread may be writing element `i`.
    #[inline]
    pub(super) unsafe fn get(&self, i: usize) -> T {
        debug_assert!(i < self.len, "body {i} is outside the {} there are", self.len);
        *self.at.add(i)
    }

    /// # Safety
    /// `i` must be in range, and this thread must be the only one touching element `i`.
    #[inline]
    pub(super) unsafe fn set(&self, i: usize, value: T) {
        debug_assert!(i < self.len, "body {i} is outside the {} there are", self.len);
        *self.at.add(i) = value;
    }
}

/// The four arrays a correction writes, bundled so a colour's closure carries one value.
#[derive(Clone, Copy)]
pub(super) struct Bodies {
    position: Cells<(f64, f64, f64)>,
    orientation: Cells<Quaternion>,
    prev_position: Cells<(f64, f64, f64)>,
    prev_orientation: Cells<Quaternion>,
}

impl Bodies {
    pub(super) fn of(
        position: &mut [(f64, f64, f64)],
        orientation: &mut [Quaternion],
        prev_position: &mut [(f64, f64, f64)],
        prev_orientation: &mut [Quaternion],
    ) -> Self {
        Bodies {
            position: Cells::of(position),
            orientation: Cells::of(orientation),
            prev_position: Cells::of(prev_position),
            prev_orientation: Cells::of(prev_orientation),
        }
    }

    /// Where body `i` is and how hard it is to move, with its world-space inverse inertia
    /// built once. See [`Pose`].
    ///
    /// # Safety
    /// `i` must be in range and owned by this thread for the whole constraint.
    #[inline]
    pub(super) unsafe fn pose(
        &self,
        i: usize,
        inv_mass: &[f64],
        inv_inertia: &[(f64, f64, f64)],
    ) -> Pose {
        let orientation = self.orientation.get(i);
        let inertia = inv_inertia[i];
        Pose {
            position: self.position.get(i),
            orientation,
            inv_mass: inv_mass[i],
            inv_inertia: inertia,
            world_inv_inertia: SymMat3::of(orientation, inertia),
        }
    }

    /// The same, plus what a contact needs that a joint does not: where the body was when
    /// the step began, and how fat it is.
    ///
    /// # Safety
    /// As [`Bodies::pose`].
    #[inline]
    pub(super) unsafe fn gather(
        &self,
        i: usize,
        inv_mass: &[f64],
        inv_inertia: &[(f64, f64, f64)],
        radius: &[f64],
    ) -> Gathered {
        Gathered {
            now: self.pose(i, inv_mass, inv_inertia),
            prev_position: self.prev_position.get(i),
            prev_orientation: self.prev_orientation.get(i),
            radius: radius[i],
        }
    }

    /// Move and turn the bodies one constraint asked for.
    ///
    /// # Safety
    /// Every body named by `corrections` must be owned by this thread -- which within a
    /// colour it is, by the colouring. See the module header.
    #[inline]
    pub(super) unsafe fn apply(&self, corrections: [Correction; 2]) {
        for correction in corrections {
            if correction.body == usize::MAX {
                continue;
            }
            let i = correction.body;
            let moved = add(correction.translation, correction.free_translation);
            self.position.set(i, add(self.position.get(i), moved));
            let turn = correction.free_rotation.multiply(&correction.rotation);
            if !turn.is_near_identity(1e-12) {
                let spun = turn.multiply(&self.orientation.get(i));
                self.orientation.set(i, renormalized(spun));
            }

            // The free part moves where the body came from as well, so the velocity read
            // back at the end of the step does not see it at all. See
            // [`Correction::free_translation`].
            if correction.free_translation != (0.0, 0.0, 0.0) {
                let was = self.prev_position.get(i);
                self.prev_position
                    .set(i, add(was, correction.free_translation));
            }
            if !correction.free_rotation.is_near_identity(1e-12) {
                let spun = correction
                    .free_rotation
                    .multiply(&self.prev_orientation.get(i));
                self.prev_orientation.set(i, renormalized(spun));
            }

        }
    }

    /// **The third correction kind: move where the body came from instead of where it
    /// is.** The velocity pass, and nothing else, is applied through here.
    ///
    /// [`Bodies::apply`] moves `position`, and moves `prev_position` with it for the free
    /// share so that the read-back sees nothing of that share. This moves `prev_position`
    /// *instead*, so the read-back sees only this and the body stays exactly where the
    /// positional solve left it. The fields are the same fields -- see [`Charge`] for why
    /// -- and `translation` is the same quantity it always was, an impulse times an
    /// inverse mass, because a velocity change `dv` held for one step is a displacement of
    /// `dv * dt` and a velocity-level impulse `J` is a positional impulse `J * dt`. The
    /// two `dt`s cancel and nothing on this path carries one.
    ///
    /// # Safety
    /// As [`Bodies::apply`]: every body named must be owned by this thread, which within a
    /// colour it is. The two arrays written are two of the four that function writes, so
    /// the module header's argument covers this without an addition.
    #[inline]
    pub(super) unsafe fn apply_velocity(&self, corrections: [Correction; 2]) {
        for correction in corrections {
            if correction.body == usize::MAX {
                continue;
            }
            let i = correction.body;
            // Subtracted rather than added: pulling the start of the step backwards is
            // what makes the body read as having travelled further over it.
            if correction.translation != (0.0, 0.0, 0.0) {
                let was = self.prev_position.get(i);
                self.prev_position.set(i, sub(was, correction.translation));
            }
            // The turn takes one more step than the translation, because the read-back is
            // a quotient rather than a difference. What the sweep reads is
            // `orientation * prev_orientation^-1`; composing the delta onto that and
            // solving back for `prev_orientation` -- `(delta * turn)^-1 * orientation` --
            // is the assignment that changes the quotient and leaves `orientation` exactly
            // where the positional solve put it.
            if !correction.rotation.is_near_identity(1e-12) {
                let now = self.orientation.get(i);
                let turn = now.multiply(&self.prev_orientation.get(i).conjugate());
                let wanted = correction.rotation.multiply(&turn);
                self.prev_orientation
                    .set(i, renormalized(wanted.conjugate().multiply(&now)));
            }
        }
    }
}
