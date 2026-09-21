//! Articulated rigid bodies: a shared body array, joints that reference it by index, and
//! one solver that sees all of them at once.
//!
//! # Why this exists next to [`crate::constraints`]
//!
//! The constraint types already in this crate -- [`crate::constraints::Joint3D`],
//! [`crate::constraints::Hinge3D`], [`crate::constraints::Fixed3D`] -- each **own their
//! two bodies by value**:
//!
//! ```ignore
//! pub struct Hinge3D { pub object1: ObjectIn3D, pub object2: ObjectIn3D, .. }
//! ```
//!
//! That is the right shape for one constraint between two things, and the wrong shape for
//! a *skeleton*. A forearm is the second body of the elbow and the first body of the
//! wrist: with ownership by value there are two copies of it, and a correction applied
//! through the elbow is invisible to the wrist until somebody copies it across. Iterating
//! such a set does not converge on the joint set, it oscillates between two pictures of
//! the same limb.
//!
//! So this module keeps **one** set of bodies and gives joints indices into it.
//!
//! # Structure of arrays, and the reason it is not a style preference
//!
//! The bodies are eight parallel arrays rather than a `Vec<Body>`. Two things need that,
//! and neither is cache-line arithmetic:
//!
//! * **SIMD on the streaming passes.** Predicting and reading velocities back are pure
//!   sweeps over every body. As arrays they vectorise and parallelise by chunk; as a
//!   `Vec<Body>` each lane would be a gather out of a 152-byte struct.
//! * **The GPU, if it is ever asked for.** A warp reading `bodies[tid].position` out of an
//!   interleaved struct wastes most of every memory transaction -- coalescing wants the
//!   field contiguous. Converting later would mean rewriting whatever had been built on
//!   top, which is why it is done before contacts rather than after.
//!
//! [`Body`] still exists as the thing you hand to [`Skeleton::add_body`] and get back from
//! [`Skeleton::body`]. It is a *view*, assembled on demand; the storage is the arrays.
//!
//! # Colouring, because the layout was only half the problem
//!
//! The first cut of this solver was Gauss-Seidel -- each joint reading the corrections the
//! last one made and writing immediately -- which is **inherently serial whatever the
//! layout is**: two joints sharing a body cannot run at once. Structure of arrays alone
//! would have parallelised the sweeps and left the solve exactly as serial as it was, and
//! the solve is where the time goes once contacts arrive.
//!
//! So the joints are partitioned into **colours**, where no two joints in a colour touch
//! the same body. Every colour runs fully parallel; within a colour it is still
//! Gauss-Seidel, so convergence is not traded away. A limb colours in two or three, a
//! whole skeleton in four or five.
//!
//! The alternative is Jacobi -- accumulate every correction and apply them at the end --
//! which parallelises without colouring and converges slower, so it needs more iterations
//! to hold a knee. Colouring keeps the iteration count.
//!
//! A colour is solved **and applied** on the threads that solved it. That is what
//! disjointness was always for, and doing anything else gives it away: an earlier version
//! computed a colour's corrections into a buffer and applied them from one thread
//! afterwards, and on a heap of ten thousand bodies the buffer cost more than the
//! arithmetic did. See [`scatter`], which holds the measurement and the safety argument.
//!
//! # Going parallel is not free, and below a size it is a loss
//!
//! Measured, one skeleton of seventeen bodies against a pile of ten thousand:
//!
//! ```text
//!   one skeleton   207 us a step    12.2 us per body
//!   the pile       2.43 ms a step    0.24 us per body
//! ```
//!
//! Fifty times worse per body on the small one, for the same arithmetic. Handing a
//! seventeen-element sweep to a thread pool costs more in scheduling than the sweep costs
//! to run, and a solver is usually called on one skeleton at a time.
//!
//! So each sweep goes parallel only above [`PARALLEL_FLOOR`], and runs on the calling
//! thread below it. The solve has its own floor, on the whole pass rather than on each
//! colour, for the reason [`crew::PASS_FLOOR`] gives.
//!
//! **And a fork is dearer than it looks.** Measured on a twenty-four core machine, one
//! `par_iter` over a few thousand items with an empty body costs 26 to 72 us before any
//! work happens -- the pool has that many threads to wake. A heap colours its contacts in
//! about twenty sets, so eight passes over them is a hundred and sixty forks a step, and
//! the scheduling is a real fraction of the solve. The answers are to ask the pool for
//! fewer, larger things: the three predict sweeps are one sweep, the narrow phase and the
//! broad phase are one fork each over fixed-size chunks, and a colour no longer has a
//! second traversal to apply what it computed.
//!
//! **And then the last of it: the solve asks once per pass, not once per colour.** A fork
//! per colour was the floor of the previous structure and it was most of what a pass
//! cost. [`crew`] replaces it with one broadcast for the whole pass and a barrier between
//! colours -- 46 us once, plus 15 us a colour, against 46 us a colour -- which also means
//! a colour too small to be worth its own fork is no longer too small to be worth
//! anything. Measured end to end on the heap, a pass went from 5.5 ms to 1.3 ms and the
//! step at eight iterations from 48 ms to 14 ms.
//!
//! # Contacts, and the two laws that turned out to be needed
//!
//! [`contacts`] adds capsule-versus-capsule and capsule-versus-ground constraints to the
//! same solve: found once a step from the predicted positions, coloured the same way the
//! joints are, and solved in the same passes. Three things there were not obvious, and
//! each of them was a measurement rather than a guess.
//!
//! * **A contact may turn into velocity only the overlap it made this step.** A
//!   position-based solver derives velocity from how far a body moved, so lifting a body
//!   out of an overlap it was already in reads back as speed: bodies spawned inside one
//!   another leave at metres a second, and a heap dropped in as a heap detonates on its
//!   first frame. Separating the two at the source costs a second pair of fields on
//!   [`Correction`] and needs no rate limit, no clamp and no tuning.
//! * **Coulomb's limit is a budget for the step, not for each pass.** Spending it per
//!   pass multiplies friction by the iteration count -- a slope that should have let go
//!   at twenty-seven degrees held past forty -- and dividing it between the passes fails
//!   the other way, because the first pass removes nearly all the overlap and leaves the
//!   rest almost no normal impulse to be a fraction of. Carrying the totals across the
//!   step gets the angle right and makes it the same at four iterations and at
//!   thirty-two. What is carried is a **cone on the resultant**, not a running total of
//!   magnitude: the friction direction reverses between passes, and charging both
//!   directions against one total spends the coefficient to produce no net impulse. See
//!   [`contacts::Spent`].
//! * **A contact patch is not a point, and the difference is a couple.** Friction acts at
//!   the surface, below the centre of mass, so it tips a body forward over its contact.
//!   For a body touching at one point that is the whole story and it should tip. A body
//!   resting on a patch moves its normal load within the patch instead and does not tip,
//!   and modelling it as a point makes a resting body ratchet itself clear of the plane
//!   over the passes until its contacts report no depth and friction stops acting. See
//!   [`contacts::patch_arm`].
//! * **Friction does not resist rolling, so a heap of capsules rolls apart.** The contact
//!   point of a rolling body is instantaneously still, so there is nothing for Coulomb to
//!   act on. Measured, a pile of forty settled onto the ground perfectly happily and then
//!   spread to twenty metres over thirty seconds. Rolling resistance -- the same law on
//!   the same budget, one dimension over -- holds it at about a metre and a quarter.
//!
//! # Sleeping, which is the removal of the solve rather than an optimisation of it
//!
//! `iterations` multiplies every joint and every contact, and none of that arithmetic
//! changes anything for a body that has stopped moving. [`sleep`] takes settled bodies
//! out of the step entirely -- an awake bitset walked a word at a time, islands over
//! joints plus contacts as the unit that sleeps and wakes together -- and a step where
//! nothing is awake returns before it touches memory. Measured on ten thousand capsules
//! resting on the ground in stacks of three: **4.5 to 6.5 ms a step down to nothing
//! measurable**, the spread being what a shared machine does to a twenty-step timing.
//! With nothing asleep it costs nothing that can be told from the noise, which is what
//! the translation-only early exit in [`Skeleton::settle`] is for.
//!
//! # Why the iteration count is what it is, and what will not move it
//!
//! Two things that look like levers and are not, both measured rather than argued.
//!
//! * **There is no impulse here to warm-start.** `contact_impulse` is not an applied
//!   impulse, it is the Coulomb budget; the positional correction is re-derived from the
//!   current geometry on every pass. Carrying the previous step's budget forward changes
//!   the slope a body holds to by under a per cent at four iterations and by nothing at
//!   eight, because the seed sits on the limit line and cancels. Nor is there an active
//!   set to predict: a contact is active iff `depth > 0`, which is three flops and
//!   cheaper than any prediction of it could be. **The N passes are Gauss-Seidel
//!   propagation whose count is set by graph distance from the ground to the top of a
//!   stack, not by combinatorics.**
//! * **Substepping does not pay.** N substeps of one pass against one step of N passes,
//!   at the same total solve work and with the broad and narrow phases still run once:
//!   the creep below is unchanged, the cost is flat within noise, the resting height and
//!   the pile's footprint improve in the third decimal, and at eight substeps of one pass
//!   Coulomb's angle breaks outright -- a thirty-degree slope holds. Macklin et al (2019)
//!   argue for it in general and it is the right thing to have tried; it is not what is
//!   wrong here.
//!
//! # The creep, which is what stops any of this settling
//!
//! A pile resting on the plane **drifts for ever**, and the drift over a window is
//! exactly linear in the window -- forty capsules left for twenty-five seconds, median
//! surface displacement as a fraction of each body's own reach, over windows of 15, 30,
//! 60, 120, 240 and 480 steps:
//!
//! ```text
//!   translation               0.0057  0.0085  0.0166  0.0359  0.0630  0.1207
//!   surface sweep of the turn 0.0077  0.0099  0.0216  0.0466  0.0922  0.1535
//!   together                  0.0126  0.0208  0.0458  0.0906  0.1697  0.2749
//! ```
//!
//! It is not a convergence failure: it is the same at eight iterations and at sixty-four,
//! and only worse at four. It is not Coulomb slip either, because quadrupling the
//! coefficient changes it by a tenth. The faults that were *within* a step have been
//! fixed -- the cone, the patch couple, the pooled ground budget -- and this survived all
//! of them, because it is **across** steps: friction compared surface points with where
//! they were at the start of *this* step, so whatever slip a step failed to remove was
//! forgiven by the next one, which re-anchored at the new position.
//!
//! And what was left of it was **straight**. A settled pile's net travel over four hundred
//! and eighty steps was 0.905 of the path it walked getting there, where a pile jostling in
//! place would be about a fifth of it. That is not slip being forgiven, it is a ratchet.
//!
//! **The patch against the plane now carries its own anchor** -- where it stuck, and the
//! Coulomb budget it stuck under -- so the slip a step fails to take off is asked for again
//! next step instead of being written off. See [`contacts::Anchor`] for what bounds it and
//! [`Skeleton::anchor_ground`] for when it is dropped. Measured over eight draws a part in
//! a hundred thousand million apart, the median body's surface drift over four hundred and
//! eighty steps and the straightness of that travel:
//!
//! ```text
//!   pile of   20            40            60
//!   before    0.075 / 0.88  0.141 / 0.80  0.217 / 0.72
//!   after     0.007 / 0.37  0.103 / 0.46  0.126 / 0.53
//! ```
//!
//! What is left is the same fault at the contacts *between* bodies, which have no anchor:
//! measured before this landed, with friction between bodies switched off and only the
//! ground's left, settled piles of twenty and of forty stopped outright. Anchors on those
//! contacts too have been tried on this branch and are not the answer -- see the section
//! below.
//!
//! # The ground patch, and the four corrections that could not stand in for it
//!
//! Every capsule lying on level ground used to slide, at between ten and forty
//! millimetres a second depending only on how long it was, while turning by nothing at
//! all; a sphere did not move. That was a contact patch sampled at each end and solved
//! one stage after the other, and it is fixed where it was caused rather than downstream
//! -- see [`contacts::solve_ground`], and the four corrections that each fixed one case
//! and broke another are recorded there too.
//!
//! ```text
//!   half-length   0.02   0.04   0.05   0.06   0.08   0.10   0.15   0.20   0.25   0.40
//!   two samples   17.4    9.0    8.5   41.8   36.0   26.1   15.9   20.3    9.7    0.4  mm/s
//!   one patch      0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
//! ```
//!
//! The resting height is exactly one radius at every one of them. A seventeen-bone rig
//! went from skating at 14 mm a second to 0.62, a stack of ten capsules from never
//! settling to asleep at step 1319, and a settled pile of twenty from covering 0.275 of a
//! body's reach in eight seconds to 0.094.
//!
//! The same change one level up -- [`contacts::capsule_contact`] emitting the two ends of
//! a *pair* patch as one constraint rather than two -- has been built and measured and is
//! not merged. It buys what it was meant to (two capsules stacked go from drifting at
//! 2.3 mm a second to nothing measurable, a ten-high stack from settling in 1866 steps to
//! 318) and it does not make a rig sleep, because a rig's self-contacts are crossed
//! rather than parallel and a crossed pair takes the same one-end branch either way.
//!
//! It was held back because a settled pile of sixty drifted further under it, and that
//! reason turns out not to survive the measurement. The pile statistic it failed on is
//! chaotic -- see `a_settled_pile_wanders_but_does_not_drift`, where eight starting
//! heights differing by seven parts in a million million give ratios from 15.0 to 40.3 on
//! the branch that passes -- so the single number it failed by sits inside the spread of
//! the branch it was compared against. Giving the unified constraint the rolling authority
//! its two contacts had between them, which was the standing hypothesis for the
//! regression, was measured and is not it: the uncoupled per-end budget takes a ten-high
//! stack from settling in 318 steps to 1370 and moves the pile's drift by less than its
//! own spread. Whoever picks the pair patch up again should judge it on the median long
//! window over several draws, not on one ratio.
//!
//! # Why a rig walked, and what stopped it
//!
//! Read in order: this section is the fault, the one after it is everything that was tried
//! against it and did not work, and the last of them is what did. **The walk is gone** --
//! a settled seventeen-bone rig that may touch itself covered 0.239 of a body's reach over
//! four hundred and eighty steps with a straightness of 0.72, and now covers 0.0003 with a
//! straightness of 0.001, which is a rig sitting still. What is *not* fixed is that such a
//! rig mostly still does not go to sleep: one draw of six does, where none did. The rest
//! are held awake by a bone or two jittering inside the loop rather than by the rig going
//! anywhere, and the sleeping test is about drift rather than jitter. See
//! `a_settled_rig_stays_where_it_settled`, which is the law this is now held to.
//!
//! **A contact between two bodies of the same skeleton closes a loop with the joints.**
//! The joints hold the pair in a small overlap -- 0.3 to 2.1 mm, measured on a settled
//! seventeen-bone rig -- the contact pushes them apart, the joints put them back, and the
//! two corrections are applied one after the other rather than together. Rigid
//! displacements about different points do not commute, so the round trip does not return
//! the loop to where it started. The configuration repeats every step, so the leftover is
//! the same small screw every step, and it integrates.
//!
//! What says it is a ratchet and not a leak is that it is **straight**: measured over
//! thirty-two windows of fifteen steps, a settled rig's net travel is 0.99 of the path it
//! walked getting there, against 0.18 for a body wandering in place. And it is the order
//! of composition that sets it: solving the stages in the opposite order every other step
//! takes an eleven-bone rig's straightness from 0.99 to 0.59 and its drift from 1.08 of a
//! reach over four hundred and eighty steps to 0.14. That is a diagnosis and not a fix,
//! for a reason worth its own section; see below.
//!
//! Everything else was measured and is not it. The residual is the same at eight, at
//! thirty-two and at sixty-four iterations; with pair friction off, with pair rolling
//! resistance off, with both off; with ground friction off and with ground rolling
//! resistance off; with hinges and with balls; with the hinge limits taken off; and with
//! every body's velocity relative to the rig's own momentum and every body's spin **zeroed
//! at the end of each step**, which is what says the motion is manufactured inside the
//! step rather than carried into it. The rig's kinetic energy sits at a third of a joule
//! for as long as it is watched while its potential energy does not move: a steady state
//! with the solver feeding one side of it and Coulomb taking from the other.
//!
//! It is not the sleep criterion being wrong, either. The rig translates as a rigid body
//! -- the median bone moves 0.039 m relative to the rig's centre while the centre moves
//! 0.276 m -- at twenty to forty millimetres a second, which is two metres a minute and
//! not something a viewer would fail to see.
//!
//! **The smallest thing that fails is three bodies**: a root with two one-capsule arms
//! ball-jointed to it at a right angle, whose anchors put the arms' axes 0.113 m apart
//! while their radii sum to 0.12. Move the anchors out to 0.10 m so the arms clear one
//! another and it sleeps at step 118; leave them touching and it never sleeps. A jointed
//! line of capsules sleeps at step 22 because a line has no loop in it, and two capsules
//! crossed on the ground with no joints at all sleep at step 20 at every crossing angle
//! from parallel to square.
//!
//! **What removes it, and it is the only thing that does.** Not solving a skeleton's
//! contacts with itself. Measured on thirteen rigs built by taking the pieces of a
//! seventeen-bone one away, six never sleep with self-collision on and all thirteen sleep
//! with it off, the seventeen-bone rig at step 1373. That is what
//! [`Skeleton::set_self_collision`] is, and it is a switch rather than a fix because it
//! costs a rig the right to stop its own limbs passing through each other.
//!
//! Narrowing it by joint-graph distance is not the answer and was measured: the
//! self-contacts a settled rig cannot shed sit two, three, four and six joints apart, and
//! rejecting everything within two joints fixes three of the thirteen and breaks a fourth
//! that used to sleep. How far apart two bodies are in the joint graph does not say
//! whether the joints will let them separate.
//!
//! # What has been tried against it, with numbers, and what is left
//!
//! **Reversing the stage order every other step is not a fix, and cannot be one.** It is
//! the obvious thing to try, since the ratchet weakens when the order of composition
//! reverses, and on the numbers above it does weaken it. It also stops a rig sleeping at
//! all: a seventeen-bone rig that cannot touch itself, which sleeps at step 1373 as things
//! are, never sleeps in six thousand steps under an alternating sweep, and its pelvis
//! moves 0.21 mm one step and 0.65 mm the next for as long as it is watched. The reason is
//! structural rather than bad luck. A body may sleep only where the step has a fixed
//! point; the forward sweep and the reversed one settle on different answers -- 0.27 mm
//! apart, measured below -- so a rig alternating between them is oscillating between two
//! equilibria and is never still. **Anything that alternates across steps buys the
//! cancellation and pays for it in exactly that coin.**
//!
//! **A symmetric sweep inside the step converges better and does nothing to the ratchet.**
//! Walking the stages forwards and then backwards within each step is the textbook
//! symmetric sweep, and being a palindrome it leaves one map per step, so sleeping stays
//! possible. It takes the rig that cannot touch itself to sleep at step 234 instead of
//! 1373, and a settled pile of twenty to sleep outright -- drift 0.0000 of a reach on all
//! eight draws, against a median of 0.075. The ratchet is untouched: the self-colliding
//! rig moves at 0.15 m/s against 0.036, and piles of forty and sixty drift 0.254 and 0.324
//! of a reach against 0.141 and 0.217, with the eight-draw spreads disjoint. Neither
//! variant is free, either, though its arithmetic is: `pile/8` 3.25 ms becomes 4.16 and
//! 4.13, `one/8` 36.8 us becomes 44.0 and 62.0, `arriving/8` 6.44 ms becomes 7.59 and
//! 6.79. None of that is the reversed index; it is what a step costs when less of the
//! skeleton is asleep.
//!
//! **The iteration count is not where this lives, and more of it is worse.** A single step
//! does converge: at four thousand and ninety-six iterations against four thousand and
//! ninety-seven, a bone moves 1e-6 mm, so the loop's constraints do have a common solution
//! and the sweep reaches it. But the forward sweep and the reversed one converge to
//! answers **0.27 mm apart** on a settled self-colliding rig, and 0.15 mm apart on one
//! that cannot touch itself. That gap is the bias, and it is the right size: a quarter of a
//! millimetre a step at sixty hertz is 16 mm/s, which is the walk. Converging harder walks
//! faster rather than slower -- 0.09 m/s at two thousand and forty-eight iterations against
//! 0.036 at eight -- because the answer the sweep is converging *to* is the one that
//! depends on the order.
//!
//! **The walk leaves through the ground rather than out of the joints.** Set the friction
//! coefficient to zero and the same rig's drift falls from 0.276 of a reach to 0.0001 and
//! its straightness from 0.96 to 0.01, while quadrupling the coefficient instead changes
//! neither much. With no gravity and no ground, the internal corrections move the rig's
//! centre of mass by nothing measurable over two thousand steps, which is a law of its own
//! -- `a_skeleton_left_to_itself_does_not_move_its_own_centre_of_mass`. So the loop
//! manufactures a cycle inside the rig and the contact with the ground rectifies it into
//! travel, which is how a crawling thing gets along.
//!
//! **What solving the loop as a loop would buy, measured before building it.** The
//! precedent is the ground patch one level down: two samples that skated became one block
//! solve and stopped, because a single correction has nothing to fail to commute with. A
//! joint and a contact cannot be merged that way, so the general version is a block or
//! direct solve over an island's whole constraint set in place of coloured Gauss-Seidel
//! over it -- a different solver, not a rule. What such a solve has that the sweep does not
//! is an answer independent of the order its constraints were visited in, and that property
//! can be had without writing it, slowly, by solving a pass **simultaneously**: every
//! constraint reading one state, each body taking the mean of the corrections that named
//! it. Standing in for the block solve that way, a self-colliding rig sleeps in 4 of 16
//! runs -- iteration counts of 128 to 1024, four draws each -- where the ordered sweep
//! sleeps in 0 of 16. So an island solved together is worth something and is not on its own
//! a cure, and that is the number a block solve has to beat rather than a hope it has to
//! carry.
//!
//! # What stopped it: the ground patch remembers where it stuck
//!
//! Friction compared the surface with where it was at the start of *this* step, so whatever
//! slip a step failed to remove was forgiven by the next one, which re-anchored at the new
//! position. The loop shakes the rig, the ground's friction rectifies the shake into
//! travel, and nothing ever asks for the travel back. **So the patch against the plane now
//! remembers**: where it stuck, the piece of its own surface that is stuck, and the Coulomb
//! budget it stuck under. [`contacts::Anchor`] is the whole of what is remembered and why
//! each part of it has to be there; [`Skeleton::anchor_ground`] decides who is stuck.
//!
//! Three bounds, and every one of them was measured rather than chosen. Take any of them
//! away and something a solver must do stops working:
//!
//! * **It is dropped the moment the patch slips**, meaning its friction reached Coulomb's
//!   limit. A contact at its limit is sliding, and where it used to be is not a fact about
//!   it any more. This is also why a sliding or rolling body behaves exactly as it did
//!   before: both spend their whole budget every step, so neither ever carries an anchor.
//! * **It names a material point, not a place under the body.** A body that *rolls* is not
//!   sliding, and friction has no business resisting it; tracking the piece of surface that
//!   stuck means rolling shows up as that piece rising off the plane, which is exactly when
//!   the memory is dropped. Anchoring the load point instead charges a rolling capsule for
//!   the whole of its roll, and `without_rolling_resistance_the_same_pile_comes_apart`
//!   catches it: the pile spread to 1.9 m rather than the 4 m it spreads to with nothing
//!   holding it.
//! * **What it remembers is bounded twice**: by the budget it was banked at, so a resting
//!   step's slip cannot be redeemed at an impact's authority, and by how far a step of
//!   gravity can drive a resting body into what it stands on, which is the scale of the
//!   residue a step can leave. With neither bound the rig walks again at 0.347 of a reach,
//!   which is where it started.
//!
//! What it buys, measured over several draws rather than one: a rig that may touch itself
//! stops travelling (0.239 of a reach to 0.0003, straightness 0.72 to 0.001) and one draw
//! in six now sleeps; a rig that cannot touch itself sleeps in all six draws at steps 112
//! to 230, where before four of six slept and took between 198 and 1520; and a settled pile
//! of forty and of sixty drift below the whole spread they used to sit in.
//!
//! It costs nothing, measured as a matched pair run back to back because this machine's
//! variance between sessions is larger than the effect: `one/8` is nine per cent faster and
//! `pile/8` twenty-six, while `arriving/8` does not move (p = 0.85). What is faster is a
//! scene that settles sooner rather than arithmetic saved -- a step with anchors costs one
//! extra pass over the ground contacts, two vectors, a float and a bit a body.
//!
//! **What it costs is patience with a tall jumble.** A stack of capsules dropped in a
//! column falls into a heap that has to shake itself out, and the ground bodies underneath
//! it can no longer creep while it does: stacks of three, four and five settle every time
//! and sooner than they did, while the taller ones take longer and their spread widens.
//! The cause is the asymmetry this fix has: the ground contact has a memory and the
//! contacts between bodies do not, so the two disagree about where the past was.
//!
//! # Anchors on the contacts between bodies, and why they are not here
//!
//! Closing that gap has been built three ways and measured, and none of them is better
//! than the asymmetry. Read this before building a fourth: the obstruction is not the one
//! that was expected, and it is stated at the end.
//!
//! **How any of this has to be judged.** Every number below is the median and spread of
//! sixteen draws a relative 1e-12 apart, because one draw says nothing. That matters for
//! the numbers already on this page too: over sixteen draws *this commit* leaves two draws
//! of a column of eight still awake after twelve thousand steps, and two draws of the
//! seventeen-bone rig `a_settled_rig_stays_where_it_settled` is written against travel 6.37
//! and 0.32 of a reach with straightnesses of 0.94 and 0.50 -- which is the walk, at the
//! size it was before the ground anchor, on seeds next door to the one the law runs. The
//! law passes on its own seed. Whoever picks this up should fix that spread rather than
//! trust either column of figures below to two significant figures.
//!
//! ```text
//!   steps to sleep      column of 3  4         5          6          7          8
//!   this commit                   381  366..378  531..1768  243..2305  775..2704  676..8239
//!                                                                             and 2 of 16 never
//!   one memory per pair patch, with the kinematic bound three paragraphs down:
//!                       206..1269  79..1139  197..3082  132..3492  236..2059  359..1927
//!                                                    2 never                    1 never
//! ```
//!
//! **A memory per contact is the shape that fails, and not because it is a memory.** Two
//! near-parallel capsules are given a contact at each end of the line they touch along;
//! anchoring those separately remembers the pair's relative *rotation* as well as its
//! relative position, out of two configurations banked at two different moments. A column
//! of three then goes from settling at step 381 on every draw to between 421 and 4452, and
//! a column of eight stops settling. Pooling them -- one memory for the pair, at the point
//! its load stands, with the budget the weakest its ends reported, which is exactly the
//! shape [`contacts::solve_ground`] gives the ground patch -- recovers all of that. So the
//! four bounds do change the answer, and the earlier report that a relative memory is
//! hopeless was a report about the wrong granularity.
//!
//! **What is frame-independent is that two pieces of surface were touching.** A pair anchor
//! that remembers a place fights every rigid motion the two make together; one that names a
//! material point on each body and remembers how they stood relative to one another does
//! not, and the ground's material-point rule then carries over unchanged -- the memory is
//! dropped when the two points stop being pressed together, which is what a pair that has
//! rolled looks like. So does the `hold` bound and the cap at `g dt^2`. Three of the four
//! bounds need nothing for a moving frame.
//!
//! **The fourth does not carry over, and that is the real difference.** Against the plane,
//! "the friction stayed strictly inside its cone" is a sufficient test for having stuck,
//! because a body that is sliding spends its whole Coulomb budget doing it. Between two
//! bodies it is not sufficient. A pair jostling in a heap can sit strictly inside its cone
//! for a step and still have slid a millimetre across itself, because the friction
//! correction that removed the slide is undone inside the same step by the normal
//! corrections of the contacts either body has with everything else. The anchor then banks
//! real motion as if it were the residue of a step, and hands it back next step as
//! velocity, which is what redeeming an anchor does. Two bodies that can each do that to
//! the other have a loop to pump, and it pumps: a seven-capsule heap settled into a
//! period-three limit cycle -- its contact count cycling 9, 7, 6 -- in which every body
//! moved at `g dt`, which is one whole step's worth of the cap spent every step. It ran
//! sixty thousand steps without settling and repeated its state every fifteen thousand.
//!
//! **Adding the kinematic half of "stuck" -- the slip over the step no larger than `g dt^2`,
//! the same residue the redemption is capped at -- removes that cycle** and is what the
//! second row above is. It takes the worst column of eight from 8239 steps to 1927, and it
//! moves the mode rather than removing it: a column of six gains two draws of sixteen that
//! do not settle. Three further bounds were measured and each buys one number and spends
//! another:
//!
//! * Dropping a memory that has grown past one step's residue, rather than clamping what it
//!   may redeem -- a memory that cannot be cashed is not a fact. Three draws of sixteen of a
//!   column of *three* then stop settling.
//! * Charging the redeemed share as a free correction rather than as velocity, which is the
//!   only thing that stops the pumping at the source. The self-colliding rig then sleeps in
//!   three draws of sixteen, the best anything has managed and better than this commit's
//!   two; a settled pile of forty drifts out to 0.30 of a reach and a column of seven takes
//!   8827 steps.
//! * Restricting the anchors to a spanning forest of the graph of bodies plus one node for
//!   the ground, which is exactly the condition for a set of *relative* memories to be
//!   jointly satisfiable and therefore the textbook cure for the over-determination. A
//!   column of five then settles at a median of 97 steps against this commit's 900, the best
//!   single number any of this produced -- and five draws of a hundred and twenty-eight stop
//!   settling, and the rig travels 0.63 of a reach. **That is the result that says the
//!   obstruction is not over-determination.** The forest makes the memories satisfiable and
//!   the scenes still ring, so what is left is the energy the redemption puts in.
//!
//! So the state of it: the granularity was wrong in the earlier attempt and pooling fixes
//! it; the moving frame costs nothing; and what is unsolved is that redeeming a memory
//! returns energy, which a loop of two bodies can pump and the plane cannot. Anything built
//! on top of this has to answer that before it answers anything else.
//!
//! # Allocation
//!
//! [`Skeleton::step`] allocates nothing once it is warm. The predicted state, the colour
//! sets, the contact buffers and the per-chunk buffers the broad and narrow phases fill
//! all live in the struct and are reused. Joint colouring happens as each joint arrives
//! -- see [`Skeleton::add_joint`] -- and contact colouring has to happen every step,
//! because the contacts do.

use rayon::prelude::*;

use crate::models::Quaternion;

mod broadphase;
mod contacts;
mod crew;
mod scatter;
mod sleep;

use broadphase::{Grid, Jointed};
use contacts::{
    capsule_contact, ground_contacts, solve_contact, solve_ground, Contact, GroundContact, Spent,
};
use scatter::Bodies;
use sleep::{settling_steps, BitSet, Components, Islands, NO_ISLAND, STILL_FRACTION};

/// Below this many items, a sweep or a colour runs on the calling thread.
///
/// See the module header for the measurement. A thread pool has a fixed cost per split --
/// a task, a queue, a join -- and a few dozen elements of arithmetic does not repay it.
///
/// Raising it to a thousand was tried, on the reasoning that a fork costs tens of
/// microseconds and a colour of three hundred contacts is worth twenty. On a loaded
/// machine it looked like a win and on an idle one it was a loss, which is the answer:
/// the fork is dear because the pool's threads are asleep, and on an idle machine they
/// are not. Left where it was, since that is the case the solver is meant for.
const PARALLEL_FLOOR: usize = 256;

/// Candidate pairs per chunk of the narrow phase. See [`Skeleton::build_contacts`].
const NARROW_CHUNK: usize = 1024;

/// Words of the awake set per chunk of a streaming sweep.
///
/// The sweeps walk the awake set a word at a time so that one test can dismiss sixty-four
/// sleeping bodies, but a word is far too small to be a unit of work for the thread pool:
/// at one word a chunk a ten-thousand-body predict is a hundred and sixty splits of about
/// a microsecond each, and the scheduling costs more than the sweep. Sixteen words is a
/// thousand bodies a chunk, which is the granularity the flat sweep this replaced had.
const SWEEP_WORDS: usize = 16;
const SWEEP_BLOCK: usize = SWEEP_WORDS * 64;

/// Coulomb friction between two bodies, unless a caller says otherwise.
///
/// Not tuned against how a pile looks: it is the measured static coefficient for cloth on
/// cloth, which is what is actually in contact when two clothed bodies rest on each other,
/// and it sits in the same band as skin on most dry surfaces. A pile of rubber wants
/// more and a pile of ice wants far less, which is what [`Skeleton::set_friction`] is for.
const DEFAULT_FRICTION: f64 = 0.5;

/// Rolling resistance between a body and whatever it is resting on.
///
/// **A capsule is perfectly round and a limb is not**, and that difference has to be paid
/// for somewhere. Coulomb friction does not resist rolling at all -- the contact point of
/// a rolling body is instantaneously still, so there is no sliding for friction to act on
/// -- so a heap of ideal capsules converts its sliding into rolling and then rolls apart
/// for ever. Measured on a pile of forty dropped together, bodies were still leaving the
/// heap at half a metre a second after thirty seconds, and had reached twenty metres out.
///
/// Rolling resistance is the real effect the round shape threw away: a deformable body
/// flattens against what it rests on, the support moves ahead of the contact point, and
/// the offset is a torque against the roll. The coefficient is the offset as a fraction
/// of the radius, and a quarter is the band measured for soft bodies on soft ground --
/// far above a steel wheel on rail, which is thousandths.
const DEFAULT_ROLLING_RESISTANCE: f64 = 0.25;

/// A rigid body, as a value. The storage is [`Skeleton`]'s arrays; this is what crosses
/// the API in either direction.
///
/// Deliberately not [`crate::models::PhysicalObject3D`], which carries a `Vec<Force>` per
/// body and an Euler-angle orientation. A skeleton is hundreds of these: a heap
/// allocation each is a cost nothing here needs, and Euler angles gimbal-lock exactly
/// where a shoulder lives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Body {
    pub position: (f64, f64, f64),
    pub orientation: Quaternion,
    pub velocity: (f64, f64, f64),
    pub angular_velocity: (f64, f64, f64),
    /// Reciprocal mass. **Zero pins the body**, which is how a skeleton is anchored to
    /// something the solver does not own -- a corpse's root, a hand still on a blade.
    pub inv_mass: f64,
    /// Reciprocal of the inertia tensor's diagonal, in the body's own frame.
    ///
    /// Diagonal because every shape a limb is made of -- a capsule, a box, a sphere --
    /// has its principal axes along its own, so the off-diagonal terms are zero in the
    /// frame the body is authored in. Storing three numbers instead of nine is not an
    /// approximation here; it is the same tensor written where it is diagonal.
    pub inv_inertia: (f64, f64, f64),
    /// The capsule this body collides as: a segment of `2 * half_length` down its own
    /// +Y, with `radius` around it.
    ///
    /// **Radius zero means no extent and no contacts.** That is what a body used purely
    /// as a joint anchor wants, and it is the default, so a caller who has not thought
    /// about collision does not silently get it.
    pub radius: f64,
    pub half_length: f64,
}

impl Body {
    /// A body at rest at `position`, with the inertia of a solid capsule of this `mass`,
    /// `radius` and segment `length`, its long axis along local **+Y**.
    ///
    /// The axis is +Y because that is the convention skeletal formats put a bone's own
    /// length down, so a segment authored in one arrives pointing the right way.
    pub fn capsule(mass: f64, radius: f64, length: f64, position: (f64, f64, f64)) -> Self {
        // A capsule's inertia, taken as the cylinder it mostly is: `m r^2 / 2` about the
        // long axis and `m (3 r^2 + L^2) / 12` across it. The hemispherical caps move
        // both terms by a few percent and are not worth the algebra for a corpse.
        let along = 0.5 * mass * radius * radius;
        let across = mass * (3.0 * radius * radius + length * length) / 12.0;
        let inv = |i: f64| if i > 0.0 { 1.0 / i } else { 0.0 };
        Body {
            position,
            orientation: Quaternion::identity(),
            velocity: (0.0, 0.0, 0.0),
            angular_velocity: (0.0, 0.0, 0.0),
            inv_mass: if mass > 0.0 { 1.0 / mass } else { 0.0 },
            inv_inertia: (inv(across), inv(along), inv(across)),
            radius,
            half_length: 0.5 * length,
        }
    }

    /// The same body given a capsule to collide as.
    ///
    /// Pairs with [`Body::pinned`], which has no extent of its own: a pinned body handed
    /// a shape is an immovable collider -- a floor, a wall, a vehicle that everything
    /// else piles against -- and the solver moves the rest of the world around it.
    pub fn shaped(mut self, radius: f64, length: f64) -> Self {
        self.radius = radius.max(0.0);
        self.half_length = 0.5 * length.max(0.0);
        self
    }

    /// The same body, pinned: infinite mass and infinite inertia, so the solver moves
    /// everything else around it.
    pub fn pinned(position: (f64, f64, f64)) -> Self {
        Body {
            position,
            orientation: Quaternion::identity(),
            velocity: (0.0, 0.0, 0.0),
            angular_velocity: (0.0, 0.0, 0.0),
            inv_mass: 0.0,
            inv_inertia: (0.0, 0.0, 0.0),
            radius: 0.0,
            half_length: 0.0,
        }
    }
}

/// What holds two bodies together, by index into the [`Skeleton`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Joint {
    /// **A point shared by two bodies**, each anchor given in its own body's frame. The
    /// shoulder and the hip: three degrees of rotational freedom, none of translation.
    Ball {
        a: usize,
        b: usize,
        anchor_a: (f64, f64, f64),
        anchor_b: (f64, f64, f64),
    },
    /// **A point shared, plus an axis shared, plus a range on the angle about it.** The
    /// elbow and the knee: one degree of freedom, and it does not go backwards.
    ///
    /// `axis_a` and `axis_b` are the hinge axis written in each body's own frame, and
    /// `min`/`max` bound the angle from `b`'s rest orientation about it, in radians.
    Hinge {
        a: usize,
        b: usize,
        anchor_a: (f64, f64, f64),
        anchor_b: (f64, f64, f64),
        axis_a: (f64, f64, f64),
        axis_b: (f64, f64, f64),
        min: f64,
        max: f64,
    },
}

impl Joint {
    fn bodies(&self) -> (usize, usize) {
        match *self {
            Joint::Ball { a, b, .. } => (a, b),
            Joint::Hinge { a, b, .. } => (a, b),
        }
    }
}

// -- small vector helpers, local because this module is the only user -------------

#[inline]
fn add(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
}

#[inline]
fn sub(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 - b.0, a.1 - b.1, a.2 - b.2)
}

#[inline]
fn scale(a: (f64, f64, f64), k: f64) -> (f64, f64, f64) {
    (a.0 * k, a.1 * k, a.2 * k)
}

#[inline]
fn dot(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

#[inline]
fn cross(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    )
}

#[inline]
fn length(a: (f64, f64, f64)) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
fn normalized(a: (f64, f64, f64)) -> Option<(f64, f64, f64)> {
    let n = length(a);
    if n > 1e-12 {
        Some(scale(a, 1.0 / n))
    } else {
        None
    }
}

/// `q v q*` for a **unit** quaternion, as the cross-product form rather than as two
/// quaternion multiplies.
///
/// [`Quaternion::rotate_point`] is the general one: it normalises its receiver, takes a
/// true inverse and multiplies twice, which is a square root and eight divisions before
/// any rotating happens. That is the right answer for a quaternion of unknown length and
/// the wrong one here, where every orientation is unit by construction --
/// [`Skeleton::add_body`] and [`Skeleton::set_body`] normalise on the way in and every
/// write inside the solver ends in `normalized`.
///
/// Measured on a heap of ten thousand bodies, a solver pass calls this about thirty times
/// per contact, and swapping the general form for this one took the contact pass from
/// 11.6 ms to 4.1 ms. It is the same rotation to the last bit the general form would give
/// a unit quaternion; it is not an approximation.
#[inline]
fn rotate(q: Quaternion, v: (f64, f64, f64)) -> (f64, f64, f64) {
    let u = (q.x, q.y, q.z);
    let t = scale(cross(u, v), 2.0);
    add(add(v, scale(t, q.w)), cross(u, t))
}

/// The same rotation backwards, `q* v q`. The conjugate is the inverse for a unit
/// quaternion, so this costs nothing the forward one does not.
#[inline]
fn rotate_inv(q: Quaternion, v: (f64, f64, f64)) -> (f64, f64, f64) {
    rotate(
        Quaternion {
            w: q.w,
            x: -q.x,
            y: -q.y,
            z: -q.z,
        },
        v,
    )
}

/// A quaternion made unit, for the case this module is always in: one that is already
/// nearly unit.
///
/// [`Quaternion::normalized`] divides all four components by the magnitude -- a square
/// root and **four divisions**, and a division is a dozen-odd cycles that do not pipeline
/// with each other. Multiplying by the reciprocal once is the same answer for a tenth of
/// the latency.
///
/// And the near-unit case skips the square root as well. Every write inside the solver is
/// a product of two unit quaternions, so its magnitude is one to within a few ulp before
/// this is called, and one step of Newton's method from a guess of one is
/// `(3 - m) / 2`. Its error is `(3/8)(m - 1)^2`, which over the band it is allowed here
/// -- a part in a billion -- is four parts in `10^19`, under an ulp of the value itself.
/// It is not an approximation at this distance from unit; it is the same double.
///
/// Measured on the heap, the solver renormalises about seventy-five thousand times a pass
/// and the sweeps ten thousand more, and this took a solve pass from 2.14 ms to 1.83 ms.
#[inline]
fn renormalized(q: Quaternion) -> Quaternion {
    let m = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    let k = if (m - 1.0).abs() < 1e-9 {
        0.5 * (3.0 - m)
    } else if m > 1e-20 {
        1.0 / m.sqrt()
    } else {
        return Quaternion::identity();
    };
    Quaternion {
        w: q.w * k,
        x: q.x * k,
        y: q.y * k,
        z: q.z * k,
    }
}

/// A body's inverse inertia written out in world space, as the symmetric matrix it is.
///
/// `I^-1` is diagonal in the body's own frame, so `R I^-1 R^T` in world space, and the
/// obvious way to apply it is to rotate the vector into the body, scale by three numbers
/// and rotate back. That is right, and it is the wrong shape when a body's tensor is
/// wanted five or six times over: a contact asks for it twice for the generalised inverse
/// masses, twice for the impulse folds, once for the rolling resistance and once more for
/// friction, and each of those was a pair of quaternion rotations.
///
/// Built flat it is one rotation matrix and three outer products, and every use after that
/// is nine multiplies. Only six of the nine entries are stored because `R I^-1 R^T` is
/// symmetric for any `R`, which is not an approximation but the shape of the thing.
#[derive(Clone, Copy, Debug)]
pub(super) struct SymMat3 {
    xx: f64,
    yy: f64,
    zz: f64,
    xy: f64,
    xz: f64,
    yz: f64,
}

impl SymMat3 {
    /// `R diag(inv_inertia) R^T`, for the unit quaternion `q`.
    #[inline]
    fn of(q: Quaternion, inv_inertia: (f64, f64, f64)) -> Self {
        let (w, x, y, z) = (q.w, q.x, q.y, q.z);
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, xz, yz) = (x * y, x * z, y * z);
        let (wx, wy, wz) = (w * x, w * y, w * z);
        // The columns of R, which are the body's own axes written in world space.
        let c0 = (1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy));
        let c1 = (2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx));
        let c2 = (2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy));
        let (ia, ib, ic) = inv_inertia;
        SymMat3 {
            xx: ia * c0.0 * c0.0 + ib * c1.0 * c1.0 + ic * c2.0 * c2.0,
            yy: ia * c0.1 * c0.1 + ib * c1.1 * c1.1 + ic * c2.1 * c2.1,
            zz: ia * c0.2 * c0.2 + ib * c1.2 * c1.2 + ic * c2.2 * c2.2,
            xy: ia * c0.0 * c0.1 + ib * c1.0 * c1.1 + ic * c2.0 * c2.1,
            xz: ia * c0.0 * c0.2 + ib * c1.0 * c1.2 + ic * c2.0 * c2.2,
            yz: ia * c0.1 * c0.2 + ib * c1.1 * c1.2 + ic * c2.1 * c2.2,
        }
    }

    #[inline]
    fn apply(&self, v: (f64, f64, f64)) -> (f64, f64, f64) {
        (
            self.xx * v.0 + self.xy * v.1 + self.xz * v.2,
            self.xy * v.0 + self.yy * v.1 + self.yz * v.2,
            self.xz * v.0 + self.yz * v.1 + self.zz * v.2,
        )
    }
}

/// Where a body is and how hard it is to move, gathered once for a constraint.
///
/// The solve used to index eight parallel arrays a dozen times over per constraint --
/// `position[a]`, `orientation[a]`, `inv_mass[a]` and so on, each a bounds-checked load
/// from a different cache line. Gathering the body once and passing it down makes the
/// scattered half of the access pattern two reads per body instead of a dozen, and it is
/// what lets [`SymMat3`] be built once rather than implied six times.
#[derive(Clone, Copy, Debug)]
pub(super) struct Pose {
    pub position: (f64, f64, f64),
    pub orientation: Quaternion,
    pub inv_mass: f64,
    pub inv_inertia: (f64, f64, f64),
    pub world_inv_inertia: SymMat3,
}

impl Pose {
    /// Whether the solver can move this body at all. A pinned one has neither mass nor
    /// inertia to give.
    #[inline]
    fn movable(&self) -> bool {
        self.inv_mass > 0.0 || self.inv_inertia != (0.0, 0.0, 0.0)
    }
}

/// A [`Pose`] plus what a contact needs and a joint does not: where the body was when the
/// step began, and how fat it is.
#[derive(Clone, Copy, Debug)]
pub(super) struct Gathered {
    pub now: Pose,
    pub prev_position: (f64, f64, f64),
    pub prev_orientation: Quaternion,
    pub radius: f64,
}

/// One body's share of one constraint's correction: where to move it and how to turn it.
///
/// **It never leaves the thread that made it.** A constraint computes its pair of these
/// and writes them into the bodies itself, so this is a local that lives in registers for
/// a few dozen instructions and is gone. That is worth saying because it used to be the
/// opposite: a colour's worth of them went into a buffer for a serial half to read back,
/// and at a hundred and twenty bytes each -- the free fields doubled it -- a pass over a
/// heap of ten thousand bodies moved ten megabytes through that buffer twice. Splitting
/// the type into a lean one for joints and a fat one for contacts would have halved a
/// cost that did not need to exist. See [`scatter`].
#[derive(Clone, Copy, Debug)]
struct Correction {
    body: usize,
    translation: (f64, f64, f64),
    /// The quaternion *delta* to left-multiply, already weighted. Identity when the body
    /// is only being moved.
    rotation: Quaternion,
    /// **The part of the same correction that must not read back as velocity.**
    ///
    /// A position-based solver derives velocity from how far a body moved, so a body
    /// lifted out of an overlap it was already in reads as having travelled under its own
    /// power: spawn two bodies inside each other and they leave at several metres a
    /// second. But a body that drove into a surface *during this step* must read as
    /// having been stopped, or nothing ever collides.
    ///
    /// Both are position corrections and no amount of clamping tells them apart, so they
    /// are separated at the source: a contact may turn into velocity only the overlap it
    /// made this step, and whatever was already there is carried here. The applying half
    /// moves the previous position by exactly this much as well, which leaves the
    /// difference the velocity is read from untouched.
    free_translation: (f64, f64, f64),
    free_rotation: Quaternion,
}

impl Correction {
    fn none() -> Self {
        Correction {
            body: usize::MAX,
            translation: (0.0, 0.0, 0.0),
            rotation: Quaternion::identity(),
            free_translation: (0.0, 0.0, 0.0),
            free_rotation: Quaternion::identity(),
        }
    }
}

/// One indivisible piece of a solver pass: a colour, or the serial tail.
///
/// A pass is a list of these, and every lane walks the same list. They exist as a list
/// rather than as four loops because the pool is handed the whole pass at once and has to
/// be told what the pass *is*; see [`crew`].
///
/// A stage carries **how much of its list to run** as well as which list, because that is
/// the only thing that differs between the full plan and the reduced one a background
/// island gets, and because a list shortened by sleeping is the same list with a smaller
/// count. A stage whose count reaches zero is left out of the plan entirely rather than
/// costing a barrier for no work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Joints { colour: u32, upto: u32 },
    Contacts { colour: u32, upto: u32 },
    /// The contacts colouring could not place. Run by one lane, because each reads the
    /// positions the one before it wrote.
    Overflow,
    /// The joints colouring could not place. Empty unless a body carries more than
    /// sixty-four joints; see [`Skeleton::add_joint`].
    JointOverflow,
    Ground,
}

/// A set of bodies and the joints between them, solved together.
///
/// See the module header for why the bodies are arrays, and why the joints are coloured.
#[derive(Clone, Debug)]
pub struct Skeleton {
    position: Vec<(f64, f64, f64)>,
    orientation: Vec<Quaternion>,
    velocity: Vec<(f64, f64, f64)>,
    angular_velocity: Vec<(f64, f64, f64)>,
    inv_mass: Vec<f64>,
    inv_inertia: Vec<(f64, f64, f64)>,
    radius: Vec<f64>,
    half_length: Vec<f64>,

    joints: Vec<Joint>,
    /// Joint indices grouped so that no two joints in a group share a body. Extended as
    /// each joint arrives; see [`Skeleton::add_joint`].
    colours: Vec<Vec<usize>>,
    /// One word per body: bit `c` set means this body already carries a joint of colour
    /// `c`. Persistent, which is what makes colouring a new joint two loads and a
    /// `trailing_ones` rather than a pass over the whole joint set.
    joint_bits: Vec<u64>,
    /// Joints that could not be coloured inside the sixty-four bits, solved one at a time
    /// on the calling thread. Reaching this needs a body with sixty-five joints on it.
    joint_overflow: Vec<usize>,
    /// Whether the jointed-neighbour runs still describe the joint set. See
    /// [`Skeleton::rebuild_jointed`].
    jointed_built: bool,
    /// Each colour's joints that are live this step, foreground first, and how much of
    /// each is foreground. Built once per step from the awake set rather than tested
    /// inside every pass -- `iterations` multiplies the pass.
    ///
    /// **A slice of one of these is still a proper colour**, which is what lets it be
    /// handed to [`scatter`]: taking a subset of a set in which no body appears twice
    /// cannot make a body appear twice, and ordering the set foreground-first is a
    /// permutation, which cannot either. `scatter::disjoint` is run on the slice that is
    /// actually solved, so the check follows the restriction rather than the whole set.
    live_joints: Vec<Vec<usize>>,
    live_joints_near: Vec<usize>,

    prev_position: Vec<(f64, f64, f64)>,
    prev_orientation: Vec<Quaternion>,

    /// Which bodies are directly jointed to each body, as one run per body, so contact
    /// generation can skip them. Rebuilt with the colouring. See [`Jointed`].
    jointed_start: Vec<u32>,
    jointed_to: Vec<u32>,
    /// Which skeleton each body belongs to, as the smallest body index its joints can
    /// reach. Filled only while [`Skeleton::set_self_collision`] is off, and empty
    /// otherwise, which is what the broad phase tests.
    jointed_component: Vec<u32>,
    self_collision: bool,
    /// Candidate pairs from the broad phase, and the contacts that survived the narrow
    /// one. Both are cleared and refilled per step rather than reallocated.
    pairs: Vec<(usize, usize)>,
    contacts: Vec<Contact>,
    /// Where each chunk of the narrow phase puts its contacts before they are
    /// concatenated. See [`Skeleton::build_contacts`].
    contact_scratch: Vec<Vec<Contact>>,
    contact_colours: Vec<Vec<usize>>,
    /// How much of each contact colour is foreground. See [`Skeleton::set_background`].
    contact_colours_near: Vec<usize>,
    /// Contacts on a body that has already used every colour the bitmask can hold. See
    /// [`Skeleton::colour_contacts`]; solved serially, and in practice empty.
    contact_overflow: Vec<usize>,
    /// One word per body: bit `c` set means this body already has a contact in colour
    /// `c`. A bitmask rather than a set per body because this is rebuilt every step.
    colour_bits: Vec<u64>,
    /// The plane everything rests on, as a unit normal and the distance along it, or
    /// `None` for a skeleton that hangs in space. See [`Skeleton::set_ground`].
    ground: Option<((f64, f64, f64), f64)>,
    ground_contacts: Vec<GroundContact>,
    /// Which ground contacts to solve. **One set, not two**: a body's whole contact with
    /// the plane is one constraint now (see [`contacts::solve_ground`]), so there is one
    /// per body and no two of them name the same body -- which is the colouring's whole
    /// requirement, met by construction. The pair of sets that used to be here existed
    /// only so that the second end of a capsule could read what the first had spent.
    ground_colours: Vec<usize>,
    friction: f64,
    rolling_resistance: f64,
    /// What each contact has already spent this step. See [`Spent`], which is also where
    /// the tangential half's shape is argued.
    ///
    /// **`contact_impulse` is indexed by contact and `ground_impulse` by body**, and the
    /// difference is deliberate: see [`Skeleton::build_contacts`] on why the two ends of
    /// one capsule on the plane share one budget, and [`Skeleton::solve_ground_colour`]
    /// on why indexing it by body is still sound under the parallel scatter.
    ///
    /// **Coulomb's limit is a budget for the whole step, not for each solver pass**, and
    /// it has to be carried across the passes or the coefficient stops meaning anything.
    /// Spending the full limit every pass multiplies the friction by the iteration count:
    /// measured, a slope that should have let go at twenty-seven degrees still held at
    /// forty. Dividing the limit between the passes instead fails the other way, because
    /// the first pass removes nearly all the overlap and the later ones have almost no
    /// normal impulse left to be a fraction of -- at thirty-two iterations that version
    /// slid at five degrees.
    ///
    /// Totals have neither problem: the tangential impulse over the step is held under
    /// `friction` times the normal impulse over the step, which is the law itself, and
    /// the answer stops depending on the quality dial.
    contact_impulse: Vec<Spent>,
    ground_impulse: Vec<Spent>,

    /// Where each body's ground patch was when it stuck, and what it stuck under. Only
    /// meaningful where [`Skeleton::ground_stuck`] says so. See [`contacts::Anchor`] for
    /// what it is for and [`Skeleton::anchor_ground`] for what maintains it.
    ground_anchor: Vec<contacts::Anchor>,
    /// One bit per body: its patch against the plane ended the last step strictly inside
    /// its friction cone, so it is stuck and the anchor above is live.
    ground_stuck: BitSet,
    /// The bodies that will be stuck when the next step reads the set, gathered while the
    /// last step's totals are still in hand. Scratch, reused; see
    /// [`Skeleton::anchor_ground`].
    ground_sticking: Vec<u32>,
    /// The furthest a step of gravity drives a resting body into what it is standing on,
    /// recomputed each step; the longest offset an anchor may remember.
    anchor_reach: f64,

    /// The stages of a pass, in order, with the empty colours left out. Rebuilt when the
    /// colouring is, not per pass. See [`Skeleton::plan_pass`].
    plan: Vec<Stage>,
    /// The same pass with every background island's constraints left out. See
    /// [`Skeleton::plan_pass`].
    plan_near: Vec<Stage>,
    plan_work: usize,
    plan_near_work: usize,

    /// The broad phase. See [`broadphase`] for why it is a grid.
    grid: Grid,

    // -- sleeping. See [`sleep`] for the whole of the reasoning. ------------------
    /// Whether settled bodies may be left out of a step at all.
    sleeping: bool,
    /// One bit per body: set means the body is simulated this step. A pinned body is
    /// never set, because it cannot move and there is nothing to simulate.
    awake: BitSet,
    /// The bodies the broad phase has already swept as it walks outward from the awake
    /// set, and the ones it is sweeping now. See [`Skeleton::find_pairs`].
    swept: BitSet,
    frontier: BitSet,
    next_frontier: BitSet,
    /// Bodies the sweep reached that were asleep, to be woken before the next round.
    reached: Vec<usize>,
    /// Bodies that have now been still for their whole settling window. The union-find
    /// below only looks at constraints with one of these at each end; see
    /// [`Skeleton::settle`].
    ready: BitSet,
    /// Ready bodies found holding up one that is still moving, set aside during the edge
    /// scan and disqualified after it. See [`Skeleton::settle`].
    held: Vec<u32>,
    /// Where a body was when its settling window opened, and how long it has been there.
    /// See [`sleep`] for why the drift is measured against the window rather than against
    /// the previous step.
    still_from: Vec<(f64, f64, f64)>,
    still_turn: Vec<Quaternion>,
    still_steps: Vec<u32>,
    /// Which sleeping island a body belongs to, or [`NO_ISLAND`] while it is awake.
    island_of: Vec<u32>,
    islands: Islands,
    components: Components,
    /// Scratch for grouping the awake bodies by island: a counting sort keyed on the
    /// component root, which is a body index, so the tallies are one word a body.
    island_tally: Vec<u32>,
    island_list: Vec<u32>,
    /// Per component root, whether any member of it is still moving.
    unsettled: Vec<bool>,
    /// Bodies the caller has said are not worth full quality, and what they get instead.
    /// See [`Skeleton::set_background`].
    background: BitSet,
    background_iterations: usize,
    any_background: bool,
}

impl Default for Skeleton {
    /// Empty, and with [`DEFAULT_FRICTION`] between its bodies. Written out rather than
    /// derived because a derived one would start at zero friction, and a pile with no
    /// friction slides flat without anything reporting an error.
    fn default() -> Self {
        Skeleton {
            position: Vec::new(),
            orientation: Vec::new(),
            velocity: Vec::new(),
            angular_velocity: Vec::new(),
            inv_mass: Vec::new(),
            inv_inertia: Vec::new(),
            radius: Vec::new(),
            half_length: Vec::new(),
            joints: Vec::new(),
            colours: Vec::new(),
            joint_bits: Vec::new(),
            joint_overflow: Vec::new(),
            jointed_built: false,
            live_joints: Vec::new(),
            live_joints_near: Vec::new(),
            prev_position: Vec::new(),
            prev_orientation: Vec::new(),
            jointed_start: Vec::new(),
            jointed_to: Vec::new(),
            jointed_component: Vec::new(),
            self_collision: true,
            pairs: Vec::new(),
            contacts: Vec::new(),
            contact_scratch: Vec::new(),
            contact_colours: Vec::new(),
            contact_colours_near: Vec::new(),
            contact_overflow: Vec::new(),
            colour_bits: Vec::new(),
            ground: None,
            ground_contacts: Vec::new(),
            ground_colours: Vec::new(),
            friction: DEFAULT_FRICTION,
            rolling_resistance: DEFAULT_ROLLING_RESISTANCE,
            contact_impulse: Vec::new(),
            ground_impulse: Vec::new(),
            ground_anchor: Vec::new(),
            ground_stuck: BitSet::default(),
            ground_sticking: Vec::new(),
            anchor_reach: 0.0,
            plan: Vec::new(),
            plan_near: Vec::new(),
            plan_work: 0,
            plan_near_work: 0,
            grid: Grid::default(),
            sleeping: true,
            awake: BitSet::default(),
            swept: BitSet::default(),
            frontier: BitSet::default(),
            next_frontier: BitSet::default(),
            reached: Vec::new(),
            ready: BitSet::default(),
            held: Vec::new(),
            still_from: Vec::new(),
            still_turn: Vec::new(),
            still_steps: Vec::new(),
            island_of: Vec::new(),
            islands: Islands::default(),
            components: Components::default(),
            island_tally: Vec::new(),
            island_list: Vec::new(),
            unsettled: Vec::new(),
            background: BitSet::default(),
            background_iterations: 1,
            any_background: false,
        }
    }
}

impl Skeleton {
    pub fn new() -> Self {
        Skeleton::default()
    }

    /// The Coulomb coefficient between every pair of bodies. Zero turns friction off and
    /// leaves only the non-penetration constraint.
    pub fn set_friction(&mut self, friction: f64) {
        self.friction = friction.max(0.0);
        // A material change is a disturbance no settling test can see: a body held on a
        // slope by the old coefficient is asleep, and has to be given the chance to find
        // out that the new one does not hold it.
        self.wake_all();
    }

    /// Rolling resistance between bodies, as a fraction of the contact radius. Zero lets
    /// a capsule roll like the ideal cylinder it is; see [`DEFAULT_ROLLING_RESISTANCE`]
    /// for why that is not what a pile wants.
    pub fn set_rolling_resistance(&mut self, resistance: f64) {
        self.rolling_resistance = resistance.max(0.0);
        self.wake_all();
    }

    /// **The ground**: the plane `dot(normal, p) = distance`, which every shaped body
    /// rests on. `None` removes it.
    ///
    /// A plane rather than a wide pinned body, because the two are not equivalent where
    /// it matters. A capsule lying across a cylinder touches it at one point however fat
    /// the cylinder is, and one point cannot hold a body flat -- it rocks, and the solve
    /// spends its iterations on that instead of on the pile. Against a plane the same
    /// capsule gets a contact at each end. Terrain that is not flat is a caller's
    /// problem for now; this is the half of it every pile needs.
    pub fn set_ground(&mut self, normal: (f64, f64, f64), distance: f64) {
        self.ground = normalized(normal).map(|n| (n, distance));
        // The floor moving is the one thing that can reach a sleeping body without
        // touching it.
        self.wake_all();
    }

    /// **Whether the bodies of one skeleton collide with each other.** On by default, and
    /// the one lever that lets a jointed rig come to rest.
    ///
    /// # What it is for, and it is not a performance switch
    ///
    /// A contact between two bodies of the same skeleton closes a loop: the joints hold
    /// the pair in a small overlap, the contact pushes them apart, the joints put them
    /// back, and the two corrections are applied one after the other rather than together.
    /// Rigid displacements about different points do not commute, so the round trip does
    /// not return the loop to where it started; the leftover is the same small screw every
    /// step, because the configuration is the same every step, and it integrates. Measured
    /// on a seventeen-bone rig lying on the plane, the whole thing walks in a straight line
    /// -- net travel over a hundred and sixty steps is 0.99 of the path it walked getting
    /// there -- at twenty to forty millimetres a second, for as long as it is watched.
    ///
    /// That is what keeps a rig out of [`sleep`], and it is nothing else: it is the same
    /// at eight, thirty-two and sixty-four iterations, with and without friction, with and
    /// without rolling resistance, with hinges and with balls, and with the hinge limits
    /// taken off. The bodies of a rig with every relative velocity and every spin zeroed
    /// at the end of each step still walk, which is what says the motion is made inside
    /// the step rather than carried into it.
    ///
    /// Measured on thirteen rigs built by taking the pieces of a seventeen-bone one away:
    /// with self-collision on, six of them never sleep; with it off, all thirteen do, and
    /// the seventeen-bone rig sleeps at step 1373. A settled island costs about sixty
    /// nanoseconds a step against milliseconds awake, so that is the whole of the
    /// difference.
    ///
    /// # What it costs
    ///
    /// Limbs pass through one another. A knee can fold into a thigh and a hand can lie
    /// inside a chest, and nothing will stop them. Contacts with *other* skeletons and
    /// with loose bodies are untouched -- only a body and its own skeleton stop seeing
    /// each other -- so a heap of rigs still piles up as a heap.
    ///
    /// It is off rather than on by default because the trade is the caller's: a rig that
    /// is looked at closely wants its limbs to collide, and a rig in a crowd wants to stop
    /// costing anything once it has landed.
    ///
    /// # Why the narrower rejection already in here is not enough
    ///
    /// Bodies either side of one joint have never collided, because their capsules overlap
    /// by construction -- see [`broadphase::Jointed`]. Widening that to two joints was
    /// measured and is not the answer: the self-contacts that a settled rig cannot shed
    /// sit two, three, four and six joints apart, at penetrations of 0.3 to 2.1 mm, and a
    /// two-joint rejection fixes three of the thirteen rigs above and breaks a fourth that
    /// used to sleep. How far apart two bodies are in the joint graph does not say whether
    /// the joints will let them separate.
    pub fn set_self_collision(&mut self, collide: bool) {
        if self.self_collision == collide {
            return;
        }
        self.self_collision = collide;
        self.jointed_built = false;
        self.wake_all();
    }

    /// Whether the bodies of one skeleton collide with each other. See
    /// [`Skeleton::set_self_collision`].
    pub fn self_collision(&self) -> bool {
        self.self_collision
    }

    /// Removes the ground plane.
    pub fn clear_ground(&mut self) {
        self.ground = None;
        self.wake_all();
    }

    /// How many contacts the last [`Skeleton::step`] found. The number a broad phase is
    /// judged against, and the one that says whether a pile is resting or interpenetrating.
    pub fn contact_count(&self) -> usize {
        self.contacts.len()
    }

    pub fn len(&self) -> usize {
        self.position.len()
    }

    pub fn is_empty(&self) -> bool {
        self.position.is_empty()
    }

    /// Adds a body and returns its index, which is what joints are written against.
    /// Orientations are normalised on the way in, and that is load-bearing rather than
    /// tidy: the solver's rotation is the unit-quaternion form (see [`rotate`]), which is
    /// only the right answer for a quaternion of length one. This and
    /// [`Skeleton::set_body`] are the only places one can arrive from outside; everything
    /// the solver itself writes is already unit.
    pub fn add_body(&mut self, mut body: Body) -> usize {
        body.orientation = body.orientation.normalized();
        self.position.push(body.position);
        self.orientation.push(body.orientation);
        self.velocity.push(body.velocity);
        self.angular_velocity.push(body.angular_velocity);
        self.inv_mass.push(body.inv_mass);
        self.inv_inertia.push(body.inv_inertia);
        self.radius.push(body.radius);
        self.half_length.push(body.half_length);
        self.prev_position.push(body.position);
        self.prev_orientation.push(body.orientation);
        self.colour_bits.push(0);
        self.joint_bits.push(0);
        self.still_from.push(body.position);
        self.still_turn.push(body.orientation);
        self.still_steps.push(0);
        self.ground_anchor.push(contacts::Anchor {
            position: body.position,
            orientation: body.orientation,
            hold: 0.0,
            local: (0.0, 0.0, 0.0),
        });
        self.island_of.push(NO_ISLAND);
        let i = self.position.len() - 1;
        let n = self.position.len();
        self.awake.resize(n, false);
        self.ground_stuck.resize(n, false);
        self.swept.resize(n, false);
        self.frontier.resize(n, false);
        self.next_frontier.resize(n, false);
        self.ready.resize(n, false);
        self.background.resize(n, false);
        // A new body arrives awake unless it is pinned, and a pinned body is never awake:
        // it does not move, so there is nothing for a step to do to it.
        if body.inv_mass > 0.0 {
            self.awake.set(i);
        }
        // The jointed runs are indexed by body, so the last one no longer covers the set.
        self.jointed_built = false;
        i
    }

    /// One body, gathered out of the arrays.
    pub fn body(&self, i: usize) -> Body {
        Body {
            position: self.position[i],
            orientation: self.orientation[i],
            velocity: self.velocity[i],
            angular_velocity: self.angular_velocity[i],
            inv_mass: self.inv_mass[i],
            inv_inertia: self.inv_inertia[i],
            radius: self.radius[i],
            half_length: self.half_length[i],
        }
    }

    /// Writes one body back. The whole body, because a caller that has one has usually
    /// changed more than one field of it.
    ///
    /// **Wakes the body's island.** A caller writing a body is the one disturbance the
    /// solver cannot see coming, and a teleported body that stays asleep is a body that
    /// never collides with anything again.
    pub fn set_body(&mut self, i: usize, body: Body) {
        self.wake(i);
        // A body that has been put somewhere else is not stuck to where it was. Leaving the
        // anchor live would have friction drag it back towards a place the caller has just
        // taken it from.
        self.ground_stuck.unset(i);
        self.position[i] = body.position;
        // Unit on the way in; see [`Skeleton::add_body`].
        self.orientation[i] = body.orientation.normalized();
        self.velocity[i] = body.velocity;
        self.angular_velocity[i] = body.angular_velocity;
        self.inv_mass[i] = body.inv_mass;
        self.inv_inertia[i] = body.inv_inertia;
        self.radius[i] = body.radius;
        self.half_length[i] = body.half_length;
    }

    pub fn position(&self, i: usize) -> (f64, f64, f64) {
        self.position[i]
    }

    pub fn orientation(&self, i: usize) -> Quaternion {
        self.orientation[i]
    }

    pub fn velocity(&self, i: usize) -> (f64, f64, f64) {
        self.velocity[i]
    }

    pub fn angular_velocity(&self, i: usize) -> (f64, f64, f64) {
        self.angular_velocity[i]
    }

    /// Sets a body's angular velocity, and **wakes its island**: a caller pushing a body
    /// is a disturbance the settling test cannot see, and it has to reach the bodies
    /// leaning on it as well as the one that was pushed.
    pub fn set_angular_velocity(&mut self, i: usize, w: (f64, f64, f64)) {
        self.wake(i);
        self.angular_velocity[i] = w;
    }

    /// Sets a body's velocity, and **wakes its island**. See
    /// [`Skeleton::set_angular_velocity`].
    pub fn set_velocity(&mut self, i: usize, v: (f64, f64, f64)) {
        self.wake(i);
        self.velocity[i] = v;
    }

    /// Adds a joint. Returns `false` and adds nothing if it names a body that does not
    /// exist, or joints a body to itself -- an out-of-range index is a caller's bug and
    /// panicking in a solver that runs per frame is worse than refusing.
    ///
    /// **Coloured on arrival, in constant time and without allocating.** The greedy rule
    /// is "the lowest colour neither body is already using", which depends on nothing but
    /// the two bodies -- so an edge added to a proper edge-colouring leaves it proper, and
    /// there is never anything to recolour.
    ///
    /// The version this replaced set a dirty flag and coloured the whole joint set on the
    /// next step, through a `vec![Vec::new(); bodies]` -- ten thousand heap allocations on
    /// a heap that size -- with a linear scan per colour probe. A caller that adds a rig
    /// per frame paid it every frame, and neither bench could see it because both build
    /// the skeleton once and then step it. Measured on ten thousand bodies and nine
    /// thousand six hundred joints: **1688 us against 3.2 us to add a rig.**
    ///
    /// The colour count is unchanged by this. Greedy takes the joints in the order they
    /// were added either way, so it lands on exactly the assignment the from-scratch pass
    /// produced -- which `colouring_joints_as_they_arrive_matches_colouring_them_all_at_once`
    /// asserts against a from-scratch pass rather than leaving implicit, because the
    /// colour count is what decides how parallel the solve can be.
    pub fn add_joint(&mut self, joint: Joint) -> bool {
        let (a, b) = joint.bodies();
        let n = self.position.len();
        if a >= n || b >= n || a == b {
            return false;
        }
        let index = self.joints.len();
        self.joints.push(joint);

        let taken = self.joint_bits[a] | self.joint_bits[b];
        if taken == u64::MAX {
            // Sixty-five joints on one body. Nothing a skeleton does reaches it, and a
            // serial tail is a better answer than a colour nobody can parallelise.
            self.joint_overflow.push(index);
        } else {
            let colour = taken.trailing_ones() as usize;
            let bit = 1u64 << colour;
            self.joint_bits[a] |= bit;
            self.joint_bits[b] |= bit;
            if colour >= self.colours.len() {
                self.colours.resize_with(colour + 1, Vec::new);
                self.live_joints.resize_with(colour + 1, Vec::new);
                self.live_joints_near.resize(colour + 1, 0);
            }
            self.colours[colour].push(index);
        }
        self.jointed_built = false;

        // A joint arriving between a sleeping body and anything else is a new way for a
        // disturbance to travel, and the islands were frozen without it.
        self.wake(a);
        self.wake(b);
        true
    }

    pub fn joints(&self) -> &[Joint] {
        &self.joints
    }

    /// How the joints were partitioned. Exposed because the colour count is the thing
    /// that decides how parallel a step can be, and a caller tuning a rig wants to see it.
    ///
    /// **Greedy colouring**: each joint takes the lowest colour no joint already on
    /// either of its bodies is using. Greedy rather than optimal because optimal
    /// colouring is NP-hard and the gain would be at most a colour or two on a graph
    /// where every vertex has degree three or four. A skeleton lands on four or five
    /// either way. It happens in [`Skeleton::add_joint`], one joint at a time.
    pub fn colours(&self) -> &[Vec<usize>] {
        &self.colours
    }

    /// Rebuilds the jointed-neighbour runs, which are indexed by body and so do not
    /// survive a body or a joint arriving.
    ///
    /// Still a full pass, unlike the colouring, because the runs are a compressed
    /// adjacency and an insertion into one moves every run after it. It is `O(bodies +
    /// joints)` of `u32` writes against the colouring pass's allocation per body, and on
    /// the heap workload it is the difference between 1688 us a frame and the number in
    /// [`Skeleton::add_joint`].
    fn rebuild_jointed(&mut self) {
        if self.jointed_built {
            return;
        }
        let bodies = self.position.len();
        self.jointed_start.clear();
        self.jointed_start.resize(bodies + 1, 0);
        for joint in self.joints.iter() {
            let (a, b) = joint.bodies();
            self.jointed_start[a] += 1;
            self.jointed_start[b] += 1;
        }
        let mut running = 0u32;
        for slot in self.jointed_start.iter_mut() {
            let count = *slot;
            *slot = running;
            running += count;
        }
        self.jointed_to.clear();
        self.jointed_to.resize(running as usize, 0);
        // A cursor per body, allocated here rather than kept on the struct because this
        // runs when the rig changes and not per step.
        let mut cursor = self.jointed_start.clone();
        for joint in self.joints.iter() {
            let (a, b) = joint.bodies();
            for (from, to) in [(a, b), (b, a)] {
                self.jointed_to[cursor[from] as usize] = to as u32;
                cursor[from] += 1;
            }
        }
        self.rebuild_components(bodies);
        self.jointed_built = true;
    }

    /// Which skeleton each body belongs to, as the smallest body index its joints reach.
    ///
    /// Left empty while a skeleton may touch itself, which is the default: the broad
    /// phase reads the emptiness rather than a flag, so the ordinary path carries no
    /// branch that a caller who has never heard of this could pay for.
    ///
    /// The labelling is a relaxation rather than a union-find: every joint pulls both its
    /// ends down to the lower of their two labels, repeated until a sweep changes nothing.
    /// A skeleton is a few dozen bodies wide and this runs when the rig changes rather
    /// than per step, so the passes cost nothing worth a second structure. The result does
    /// not depend on the order the joints are in, which a union-find's roots would.
    fn rebuild_components(&mut self, bodies: usize) {
        self.jointed_component.clear();
        if self.self_collision {
            return;
        }
        self.jointed_component.extend(0..bodies as u32);
        loop {
            let mut changed = false;
            for joint in self.joints.iter() {
                let (a, b) = joint.bodies();
                let lowest = self.jointed_component[a].min(self.jointed_component[b]);
                for end in [a, b] {
                    if self.jointed_component[end] != lowest {
                        self.jointed_component[end] = lowest;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    /// Which bodies each body is jointed to. See [`Jointed`].
    fn jointed(&self) -> Jointed<'_> {
        Jointed {
            start: &self.jointed_start,
            to: &self.jointed_to,
            component: &self.jointed_component,
        }
    }

    /// Whether a joint holds these two bodies together, which is the pair contact
    /// generation must not produce.
    pub fn is_jointed(&mut self, a: usize, b: usize) -> bool {
        self.rebuild_jointed();
        self.jointed().holds(a, b)
    }

    /// Candidate pairs for the narrow phase, from the broad phase.
    ///
    /// **Only awake bodies are swept**, which is the broad phase's whole share of what
    /// sleeping saves: a settled heap produces no outer loop at all, and the grid is left
    /// standing because nothing in it moved. A sleeping body is still in the grid and
    /// still a collider -- it has to be, or an awake body would fall through the heap it
    /// landed on -- it simply does not go looking.
    ///
    /// A sleeping body found within reach of an awake one is woken, and then has to be
    /// swept itself, because *its* neighbours further into the heap have not been looked
    /// at by anybody. So the sweep runs outward in rounds until a round wakes nothing:
    /// the awake set first, then whatever that reached, and so on. One round on a quiet
    /// frame, two or three where something has just landed.
    fn find_pairs(&mut self) {
        let mut pairs = std::mem::take(&mut self.pairs);
        pairs.clear();
        if !self.awake.any() {
            // Nothing moved, so the grid still describes where everything is and there is
            // no outer loop to run.
            self.pairs = pairs;
            return;
        }
        let mut grid = std::mem::take(&mut self.grid);
        grid.rebuild(&self.position, &self.radius, &self.half_length);

        self.swept.clear();
        self.frontier.clear();
        self.frontier.union(&self.awake);
        let mut reached = std::mem::take(&mut self.reached);
        loop {
            reached.clear();
            grid.pairs(
                &self.position,
                &self.inv_mass,
                Jointed {
                    start: &self.jointed_start,
                    to: &self.jointed_to,
                    component: &self.jointed_component,
                },
                &self.frontier,
                &self.swept,
                &mut pairs,
                &mut reached,
            );
            self.swept.union(&self.frontier);
            if reached.is_empty() {
                break;
            }
            // Everything the round reached is now awake, and its island with it: a body
            // underneath the one that was touched is just as disturbed as the one that
            // was.
            for &i in reached.iter() {
                self.wake(i);
            }
            self.next_frontier.difference(&self.awake, &self.swept);
            std::mem::swap(&mut self.frontier, &mut self.next_frontier);
            if !self.frontier.any() {
                break;
            }
        }
        self.reached = reached;
        self.grid = grid;
        self.pairs = pairs;
    }

    /// The narrow phase: which candidates are actually touching, and where.
    fn build_contacts(&mut self) {
        let mut contacts = std::mem::take(&mut self.contacts);
        contacts.clear();
        let position = &self.position;
        let orientation = &self.orientation;
        let radius = &self.radius;
        let half_length = &self.half_length;
        let test = |&(a, b): &(usize, usize)| {
            capsule_contact(a, b, position, orientation, radius, half_length)
                .into_iter()
                .flatten()
        };
        // Chunked into buffers this struct owns, rather than `par_extend` over a
        // flat-mapping parallel iterator. The number of contacts a pair yields is not
        // known in advance, so `par_extend` cannot write in place: it builds a tree of
        // collections and folds them together, and measured on a heap of ten thousand
        // bodies that cost 3.7 ms against 0.5 for the same tests run this way. Chunked by
        // a fixed count rather than by the thread count, so the contact list is the same
        // list in the same order on every machine.
        if self.pairs.len() >= PARALLEL_FLOOR {
            let mut scratch = std::mem::take(&mut self.contact_scratch);
            let chunks = self.pairs.len().div_ceil(NARROW_CHUNK);
            if scratch.len() < chunks {
                scratch.resize_with(chunks, Vec::new);
            }
            scratch[..chunks]
                .par_iter_mut()
                .zip(self.pairs.par_chunks(NARROW_CHUNK))
                .for_each(|(into, chunk)| {
                    into.clear();
                    into.extend(chunk.iter().flat_map(test));
                });
            for filled in scratch[..chunks].iter() {
                contacts.extend_from_slice(filled);
            }
            self.contact_scratch = scratch;
        } else {
            contacts.extend(self.pairs.iter().flat_map(test));
        }
        self.contacts = contacts;
        self.contact_impulse.clear();
        self.contact_impulse
            .resize(self.contacts.len(), Spent::default());

        self.ground_contacts.clear();
        self.ground_colours.clear();
        let Some((normal, distance)) = self.ground else {
            return;
        };
        // Awake bodies only, and in increasing order, which is the order the loop this
        // replaced produced: a sleeping body is already resting on the plane -- that is
        // most of why it went to sleep -- and the plane cannot arrive underneath it.
        let awake = std::mem::take(&mut self.awake);
        awake.for_each_set(|i| {
            let before = self.ground_contacts.len();
            ground_contacts(
                i,
                self.position[i],
                self.orientation[i],
                self.radius[i],
                self.half_length[i],
                normal,
                distance,
                &mut self.ground_contacts,
            );
            for index in before..self.ground_contacts.len() {
                self.ground_colours.push(index);
            }
        });
        self.awake = awake;

        // One running impulse per contact, like every other constraint here. It used to
        // have to be one per *body*, pooled across the two ends of a capsule's patch,
        // because Coulomb's limit belongs to the patch and not to a sample of it -- and
        // that made it the one exception the [`scatter`] safety argument had to make a
        // case for. The patch is one constraint now, so the exception is gone: the
        // budget is the constraint's, indexed by the constraint.
        self.ground_impulse.clear();
        self.ground_impulse
            .resize(self.ground_contacts.len(), Spent::default());
    }

    /// **Greedy colouring again, but every step**, because the contact set is new every
    /// step where the joint set is not.
    ///
    /// The taken-colour test is a bit in a word per body instead of the joint version's
    /// vector per body: same algorithm, no allocation, and the whole pass is a handful of
    /// instructions per contact. Sixty-four colours is the price -- a body touched by
    /// more than sixty-four others at once sends its extra contacts to
    /// `contact_overflow`, which is solved serially. Reaching that means a body buried
    /// under sixty-four neighbours, and a pile that deep has worse problems than a
    /// serial tail.
    fn colour_contacts(&mut self) {
        for set in self.contact_colours.iter_mut() {
            set.clear();
        }
        self.contact_colours_near.clear();
        self.contact_colours_near
            .resize(self.contact_colours.len(), 0);
        self.contact_overflow.clear();
        for bits in self.colour_bits.iter_mut() {
            *bits = 0;
        }

        // Foreground contacts are coloured first, so each colour's list is its foreground
        // prefix followed by its background tail and a reduced pass stops at
        // `contact_colours_near`. Which colour a contact gets is free to choose; that a
        // colour names each body once is not, and taking a prefix of a set with that
        // property cannot break it. See [`scatter`].
        for background in [false, true] {
            if background && !self.any_background {
                break;
            }
            for (index, contact) in self.contacts.iter().enumerate() {
                let (a, b) = (contact.a, contact.b);
                // Background only where *both* ends are: a contact on the boundary
                // between the two is solved at full quality, because it is the one
                // carrying whatever the foreground is leaning on.
                if self.any_background
                    && (self.background.get(a) && self.background.get(b)) != background
                {
                    continue;
                }
                let taken = self.colour_bits[a] | self.colour_bits[b];
                if taken == u64::MAX {
                    self.contact_overflow.push(index);
                    continue;
                }
                let colour = taken.trailing_ones() as usize;
                let bit = 1u64 << colour;
                self.colour_bits[a] |= bit;
                self.colour_bits[b] |= bit;
                if colour >= self.contact_colours.len() {
                    self.contact_colours.resize_with(colour + 1, Vec::new);
                    self.contact_colours_near.resize(colour + 1, 0);
                }
                self.contact_colours[colour].push(index);
            }
            if !background {
                for (near, set) in self
                    .contact_colours_near
                    .iter_mut()
                    .zip(self.contact_colours.iter())
                {
                    *near = set.len();
                }
            }
        }
    }

    /// Which joints are worth solving this step, per colour, foreground first.
    ///
    /// Built once rather than tested inside every pass: `iterations` is the multiplier on
    /// everything in the solve, so a test that has to happen per constraint wants to
    /// happen once a step and not eight times.
    fn find_live_joints(&mut self) {
        let Skeleton {
            colours,
            live_joints,
            live_joints_near,
            joints,
            awake,
            background,
            any_background,
            ..
        } = self;
        live_joints.resize_with(colours.len(), Vec::new);
        live_joints_near.resize(colours.len(), 0);
        for (colour, set) in colours.iter().enumerate() {
            let live = &mut live_joints[colour];
            live.clear();
            for pass in [false, true] {
                if pass && !*any_background {
                    break;
                }
                for &k in set.iter() {
                    let (a, b) = joints[k].bodies();
                    if !awake.get(a) && !awake.get(b) {
                        continue;
                    }
                    if *any_background && (background.get(a) && background.get(b)) != pass {
                        continue;
                    }
                    live.push(k);
                }
                if !pass {
                    live_joints_near[colour] = live.len();
                }
            }
        }
    }

    /// **One step.** Predict under `gravity`, run `iterations` passes over the coloured
    /// joint sets, then read the velocities back out of what moved.
    ///
    /// Allocates nothing. `iterations` is the quality dial: four is enough for a corpse,
    /// and the cost is linear in it.
    pub fn step(&mut self, dt: f64, gravity: (f64, f64, f64), iterations: usize) {
        if dt <= 0.0 || self.position.is_empty() {
            return;
        }
        // Everything has settled and nothing has disturbed it. There is no state a step
        // could change, so the cheapest honest answer is the whole step: `bodies / 64`
        // word tests and no memory touched. This is what sleeping is for.
        if !self.awake.any() {
            return;
        }
        self.rebuild_jointed();
        // The furthest a step of gravity can drive a resting body into what it is standing
        // on, which is the scale of the slip a step can leave behind. See
        // [`contacts::Anchor`].
        self.anchor_reach = length(gravity) * dt * dt;

        self.prev_position.copy_from_slice(&self.position);
        self.prev_orientation.copy_from_slice(&self.orientation);

        // **One sweep, not three.** Falling, travelling and spinning were a pass each,
        // which is three reads of every array and three handings of the same ten thousand
        // bodies to the thread pool. Nothing in them crosses bodies, so they fuse: the
        // arrays are read once and the pool is asked once. Measured on the heap, a fork
        // and a join cost 30 to 70 us on their own, and the whole predict is under a
        // millisecond -- the scheduling was a real fraction of it.
        let g = gravity;
        let wide = self.position.len() >= PARALLEL_FLOOR;

        let predict = |((((p, v), q), &inv_m), w): (
            (((&mut (f64, f64, f64), &mut (f64, f64, f64)), &mut Quaternion), &f64),
            &(f64, f64, f64),
        )| {
            if inv_m > 0.0 {
                *v = add(*v, scale(g, dt));
            }
            *p = add(*p, scale(*v, dt));
            *q = integrate_spin(*q, *w, dt);
        };

        // **And only over the bodies that are awake**, a word of the awake set at a
        // time. A word of zeroes is sixty-four sleeping bodies dismissed by one test,
        // which is the shape a settled heap has; a word of ones runs a straight loop with
        // no bit arithmetic in it, so a skeleton where nothing sleeps runs what it ran
        // before. Only the ragged edge between them pays for the `trailing_zeros`.
        let block = |(((p, v), q), ((inv_m, w), words)): (
            (
                (&mut [(f64, f64, f64)], &mut [(f64, f64, f64)]),
                &mut [Quaternion],
            ),
            ((&[f64], &[(f64, f64, f64)]), &[u64]),
        )| {
            // A chunk with nothing asleep in it runs the zipped sweep it always ran.
            // Walking it by index instead costs a bounds check on five slices a body, and
            // measured on a heap where nothing sleeps that was most of what sleeping cost
            // when it was saving nothing.
            if words.iter().all(|&word| word == u64::MAX) {
                p.iter_mut()
                    .zip(v.iter_mut())
                    .zip(q.iter_mut())
                    .zip(inv_m.iter())
                    .zip(w.iter())
                    .for_each(predict);
                return;
            }
            let mut run = |i: usize| predict(((((&mut p[i], &mut v[i]), &mut q[i]), &inv_m[i]), &w[i]));
            for (nth, &word) in words.iter().enumerate() {
                let base = nth * 64;
                if word == u64::MAX {
                    let upto = (base + 64).min(inv_m.len());
                    for i in base..upto {
                        run(i);
                    }
                    continue;
                }
                let mut word = word;
                while word != 0 {
                    let i = base + word.trailing_zeros() as usize;
                    word &= word - 1;
                    run(i);
                }
            }
        };
        let words = self.awake.words();
        if wide {
            self.position
                .par_chunks_mut(SWEEP_BLOCK)
                .zip(self.velocity.par_chunks_mut(SWEEP_BLOCK))
                .zip(self.orientation.par_chunks_mut(SWEEP_BLOCK))
                .zip(
                    self.inv_mass
                        .par_chunks(SWEEP_BLOCK)
                        .zip(self.angular_velocity.par_chunks(SWEEP_BLOCK))
                        .zip(words.par_chunks(SWEEP_WORDS)),
                )
                .for_each(block);
        } else {
            self.position
                .chunks_mut(SWEEP_BLOCK)
                .zip(self.velocity.chunks_mut(SWEEP_BLOCK))
                .zip(self.orientation.chunks_mut(SWEEP_BLOCK))
                .zip(
                    self.inv_mass
                        .chunks(SWEEP_BLOCK)
                        .zip(self.angular_velocity.chunks(SWEEP_BLOCK))
                        .zip(words.chunks(SWEEP_WORDS)),
                )
                .for_each(block);
        }

        // Contacts are found once, from the predicted positions, and then solved on every
        // iteration. Regenerating them per iteration would double the narrow phase for a
        // set that barely changes across a step, and would let a pair appear and vanish
        // between passes so that nothing ever converged.
        self.find_pairs();
        self.build_contacts();
        self.colour_contacts();
        self.find_live_joints();
        self.plan_pass();

        let iterations = iterations.max(1);
        // Past this count a pass runs the foreground plan instead of the whole one. The
        // two plans are built together and differ only in how much of each stage's list
        // they name, so the choice is one branch a pass rather than a test per
        // constraint. See [`Skeleton::set_background`].
        let background = self.background_iterations.clamp(1, iterations);
        for pass in 0..iterations {
            self.solve_pass(self.any_background && pass >= background);
        }

        // And one sweep to read both velocities back, for the same reason the predict is
        // one.
        let inv_dt = 1.0 / dt;
        let read_back = |((((v, w), p), prev_p), (q, prev_q)): (
            (
                ((&mut (f64, f64, f64), &mut (f64, f64, f64)), &(f64, f64, f64)),
                &(f64, f64, f64),
            ),
            (&Quaternion, &Quaternion),
        )| {
            *v = scale(sub(*p, *prev_p), inv_dt);
            // The rotation that happened, as an axis-angle, divided by the step. The
            // `w < 0` flip keeps the short way round: a quaternion and its negation are
            // the same orientation, and without the check a body can read as spinning
            // almost a full turn when it barely moved.
            let delta = q.multiply(&prev_q.conjugate());
            let sign = if delta.w < 0.0 { -1.0 } else { 1.0 };
            *w = scale((delta.x, delta.y, delta.z), 2.0 * inv_dt * sign);
        };

        // Awake bodies only, in blocks of sixty-four, for the same reason the predict is.
        let block = |(((v, w), p), ((prev_p, (q, prev_q)), words)): (
            (
                (&mut [(f64, f64, f64)], &mut [(f64, f64, f64)]),
                &[(f64, f64, f64)],
            ),
            ((&[(f64, f64, f64)], (&[Quaternion], &[Quaternion])), &[u64]),
        )| {
            // See the predict sweep: a chunk with nothing asleep in it runs the zipped
            // form, and only a ragged one pays for the bits.
            if words.iter().all(|&word| word == u64::MAX) {
                v.iter_mut()
                    .zip(w.iter_mut())
                    .zip(p.iter())
                    .zip(prev_p.iter())
                    .zip(q.iter().zip(prev_q.iter()))
                    .for_each(read_back);
                return;
            }
            let mut run = |i: usize| {
                read_back((
                    (((&mut v[i], &mut w[i]), &p[i]), &prev_p[i]),
                    (&q[i], &prev_q[i]),
                ))
            };
            for (nth, &word) in words.iter().enumerate() {
                let base = nth * 64;
                if word == u64::MAX {
                    let upto = (base + 64).min(p.len());
                    for i in base..upto {
                        run(i);
                    }
                    continue;
                }
                let mut word = word;
                while word != 0 {
                    let i = base + word.trailing_zeros() as usize;
                    word &= word - 1;
                    run(i);
                }
            }
        };
        let words = self.awake.words();
        if wide {
            self.velocity
                .par_chunks_mut(SWEEP_BLOCK)
                .zip(self.angular_velocity.par_chunks_mut(SWEEP_BLOCK))
                .zip(self.position.par_chunks(SWEEP_BLOCK))
                .zip(
                    self.prev_position
                        .par_chunks(SWEEP_BLOCK)
                        .zip(
                            self.orientation
                                .par_chunks(SWEEP_BLOCK)
                                .zip(self.prev_orientation.par_chunks(SWEEP_BLOCK)),
                        )
                        .zip(words.par_chunks(SWEEP_WORDS)),
                )
                .for_each(block);
        } else {
            self.velocity
                .chunks_mut(SWEEP_BLOCK)
                .zip(self.angular_velocity.chunks_mut(SWEEP_BLOCK))
                .zip(self.position.chunks(SWEEP_BLOCK))
                .zip(
                    self.prev_position
                        .chunks(SWEEP_BLOCK)
                        .zip(
                            self.orientation
                                .chunks(SWEEP_BLOCK)
                                .zip(self.prev_orientation.chunks(SWEEP_BLOCK)),
                        )
                        .zip(words.chunks(SWEEP_WORDS)),
                )
                .for_each(block);
        }

        self.settle(dt, length(gravity));
        // After the solve rather than inside it, because what decides whether a patch is
        // stuck is the Coulomb total the whole step spent, which only exists once the last
        // pass has run. See [`Skeleton::anchor_ground`].
        self.anchor_ground();
    }

    /// The four body arrays every correction writes, as the disjoint-scatter view a
    /// colour is applied through. See [`scatter`] for why, and for the safety argument.
    #[inline]
    fn writable(&mut self) -> Bodies {
        Bodies::of(
            &mut self.position,
            &mut self.orientation,
            &mut self.prev_position,
            &mut self.prev_orientation,
        )
    }

    /// The stages of one pass, in the order they have to run, with the empty ones left
    /// out so that no lane waits at a barrier for work that does not exist.
    ///
    /// Built once a step rather than once a pass, because the colouring does not change
    /// between the passes of a step -- only the positions do. Reused rather than
    /// reallocated, like everything else here.
    ///
    /// **Two plans**, and they differ only in how much of each list they name: the whole
    /// of it, and the foreground prefix a background island's constraints are left out
    /// of. Building both here rather than choosing per stage per pass keeps the choice to
    /// one branch a pass. A colour with nothing live in it -- every constraint on
    /// sleeping bodies -- is in neither, so a mostly-settled skeleton produces a shorter
    /// plan rather than a plan full of empty stages.
    fn plan_pass(&mut self) {
        self.plan.clear();
        self.plan_near.clear();
        self.plan_work = 0;
        self.plan_near_work = 0;
        for colour in 0..self.live_joints.len() {
            let live = self.live_joints[colour].len();
            let near = self.live_joints_near[colour];
            if live > 0 {
                self.plan.push(Stage::Joints {
                    colour: colour as u32,
                    upto: live as u32,
                });
                self.plan_work += live;
            }
            if near > 0 {
                self.plan_near.push(Stage::Joints {
                    colour: colour as u32,
                    upto: near as u32,
                });
                self.plan_near_work += near;
            }
        }
        if !self.joint_overflow.is_empty() {
            self.plan.push(Stage::JointOverflow);
            self.plan_near.push(Stage::JointOverflow);
        }
        for colour in 0..self.contact_colours.len() {
            let live = self.contact_colours[colour].len();
            let near = self.contact_colours_near[colour];
            if live > 0 {
                self.plan.push(Stage::Contacts {
                    colour: colour as u32,
                    upto: live as u32,
                });
                self.plan_work += live;
            }
            if near > 0 {
                self.plan_near.push(Stage::Contacts {
                    colour: colour as u32,
                    upto: near as u32,
                });
                self.plan_near_work += near;
            }
        }
        if !self.contact_overflow.is_empty() {
            self.plan.push(Stage::Overflow);
            self.plan_near.push(Stage::Overflow);
        }
        // The ground is never reduced and never dropped: there is one contact per body,
        // it is the cheapest constraint in the step, and the artefact of under-solving one
        // is a body sinking through the floor, which is the one no distance excuses. It is
        // also one stage where it was two, because the second used to have to read what
        // the first had spent and there is no longer a first.
        if !self.ground_colours.is_empty() {
            self.plan.push(Stage::Ground);
            self.plan_near.push(Stage::Ground);
            self.plan_work += self.ground_colours.len();
            self.plan_near_work += self.ground_colours.len();
        }
    }

    /// **One pass over every colour**, handed to the pool once.
    ///
    /// Each lane walks the same list of stages and takes its own slice of each, waiting
    /// at a barrier before the next -- so a colour is still finished everywhere before
    /// the next one starts, which is what the colouring requires, but it costs a barrier
    /// rather than a fork. See [`crew`] for the measurement that demanded it and for why
    /// this adds no unsafety to what [`scatter`] already argued.
    fn solve_pass(&mut self, near_only: bool) {
        #[cfg(debug_assertions)]
        self.check_colours_are_disjoint();

        // Taken before the shared borrows: these hold raw pointers rather than
        // references, so the mutable borrow each one needs ends here.
        let bodies = self.writable();
        let contact_impulse = scatter::Cells::of(&mut self.contact_impulse);
        let ground_impulse = scatter::Cells::of(&mut self.ground_impulse);

        let (plan, work) = if near_only {
            (&self.plan_near, self.plan_near_work)
        } else {
            (&self.plan, self.plan_work)
        };
        let colours = &self.live_joints;
        let contact_colours = &self.contact_colours;
        let ground_colours = &self.ground_colours;
        let overflow = &self.contact_overflow;
        let joint_overflow = &self.joint_overflow;
        let awake = &self.awake;
        let joints = &self.joints;
        let contacts = &self.contacts;
        let ground_contacts = &self.ground_contacts;
        let inv_mass = &self.inv_mass;
        let inv_inertia = &self.inv_inertia;
        let radius = &self.radius;
        let friction = self.friction;
        let rolling = self.rolling_resistance;
        let ground = self.ground;
        let ground_anchor = &self.ground_anchor;
        let ground_stuck = &self.ground_stuck;
        let anchor_reach = self.anchor_reach;

        // SAFETY: the argument is the one [`scatter`] makes, and every part of it still
        // holds here.
        //
        // * Within a stage, each constraint reads and writes only the bodies it names,
        //   and no two constraints in a colour name the same body -- established by
        //   `Skeleton::recolour` for the joints and `Skeleton::colour_contacts` for the
        //   contacts, and checked over every colour by `check_colours_are_disjoint` in
        //   debug builds. The ground sets come from `Skeleton::build_contacts`, which
        //   puts a body's first contact in one set and its second in the other.
        // * Within a colour, the lanes take the disjoint slices `Lane::span` cuts, which
        //   partition the colour exactly -- so a body named once in the colour is
        //   addressed by exactly one lane. The debug check covers this because it runs
        //   over the whole colour, which is the union of the lanes' slices.
        // * The running impulses for joints and pair contacts are indexed by the
        //   constraint, not by the body, so each is touched by the one lane that owns
        //   that constraint.
        // * `ground_impulse` is indexed by the constraint like the others, and used to be
        //   the one exception this argument had to make a case for -- indexed by body,
        //   because the two ends of a capsule pooled one Coulomb budget. A body's contact
        //   with the plane is one constraint now, so there is one entry per body either
        //   way and the exception has gone: see [`contacts::solve_ground`].
        // * Across stages, `crew::each_stage` puts a barrier between them, and its
        //   release-acquire pair is what makes one colour's writes visible to the next.
        //   That ordering used to come from the join; losing it without replacing it
        //   would be the one way this change could be unsound.
        // * `Stage::Overflow` is run by a single lane, so nothing in it is shared at all.
        let run = |lane: crew::Lane| unsafe {
            match plan[lane.stage] {
                Stage::Joints { colour, upto } => {
                    let set = &colours[colour as usize][..upto as usize];
                    for &k in &set[lane.span(set.len())] {
                        let joint = joints[k];
                        let (a, b) = joint.bodies();
                        let first = bodies.pose(a, inv_mass, inv_inertia);
                        let second = bodies.pose(b, inv_mass, inv_inertia);
                        bodies.apply(solve_joint(joint, &first, &second));
                    }
                }
                Stage::JointOverflow => {
                    if lane.is_only() {
                        for &k in joint_overflow.iter() {
                            let joint = joints[k];
                            let (a, b) = joint.bodies();
                            if !awake.get(a) && !awake.get(b) {
                                continue;
                            }
                            let first = bodies.pose(a, inv_mass, inv_inertia);
                            let second = bodies.pose(b, inv_mass, inv_inertia);
                            bodies.apply(solve_joint(joint, &first, &second));
                        }
                    }
                }
                Stage::Contacts { colour, upto } => {
                    let set = &contact_colours[colour as usize][..upto as usize];
                    let span = lane.span(set.len());
                    solve_some_contacts(
                        &set[span],
                        contacts,
                        &bodies,
                        &contact_impulse,
                        inv_mass,
                        inv_inertia,
                        radius,
                        friction,
                        rolling,
                    );
                }
                Stage::Overflow => {
                    if lane.is_only() {
                        solve_some_contacts(
                            overflow,
                            contacts,
                            &bodies,
                            &contact_impulse,
                            inv_mass,
                            inv_inertia,
                            radius,
                            friction,
                            rolling,
                        );
                    }
                }
                Stage::Ground => {
                    let Some((normal, distance)) = ground else {
                        return;
                    };
                    let set = ground_colours;
                    for &k in &set[lane.span(set.len())] {
                        let contact = ground_contacts[k];
                        let body = bodies.gather(contact.body, inv_mass, inv_inertia, radius);
                        let (correction, totals) = solve_ground(
                            contact,
                            &body,
                            friction,
                            rolling,
                            normal,
                            distance,
                            ground_impulse.get(k),
                            ground_stuck
                                .get(contact.body)
                                .then(|| ground_anchor[contact.body]),
                            anchor_reach,
                        );
                        ground_impulse.set(k, totals);
                        bodies.apply([correction, Correction::none()]);
                    }
                }
            }
        };

        crew::each_stage(plan.len(), work, run);
    }

    /// **Which ground patches are stuck, and where they stuck**, decided once a step while
    /// the step's own Coulomb totals are still in hand.
    ///
    /// A patch is stuck when the friction it spent stayed *strictly* inside its cone:
    /// Coulomb's limit was never reached, so nothing slid, and the piece of plane it is
    /// standing on is the piece it was standing on. One that reached the limit did slide,
    /// and its memory of where it was is worth nothing -- so it is dropped, and the next
    /// step measures from where the body now is, which is what this module did everywhere
    /// before anchors existed. That is also what makes a rolling capsule and a sliding one
    /// behave as they always did: both spend their whole budget every step, so neither ever
    /// carries an anchor.
    ///
    /// The anchor is set to where the body was at the **start** of the step it stuck in
    /// rather than where it ended, because what is being remembered is precisely the slip
    /// that step failed to take off.
    ///
    /// A body with no patch this step -- in the air, asleep, or resting on something other
    /// than the plane -- is not in the contact list, so its bit is cleared and its memory
    /// goes with it. An anchor is never older than the contact holding it.
    fn anchor_ground(&mut self) {
        self.ground_sticking.clear();
        let Some((normal, distance)) = self.ground else {
            self.ground_stuck.clear();
            return;
        };
        for k in 0..self.ground_contacts.len() {
            let i = self.ground_contacts[k].body;
            let spent = self.ground_impulse[k];
            let budget = self.friction * spent.normal;
            if spent.normal <= 0.0 || length(spent.tangential) >= budget {
                continue;
            }
            // The piece of surface that is stuck has to still be against the plane. A body
            // that has rolled is touching somewhere else, and what its old contact point
            // did on the way round is not a slide. See [`contacts::Anchor`].
            let stuck = self.ground_stuck.get(i) && {
                let anchor = self.ground_anchor[i];
                let at = add(self.position[i], rotate(self.orientation[i], anchor.local));
                distance - dot(normal, at) > 0.0
            };
            if stuck {
                // The stored offset is the sum of what several steps banked, so the
                // authority that covers all of it is the weakest of theirs. See
                // [`contacts::Anchor`].
                self.ground_anchor[i].hold = self.ground_anchor[i].hold.min(budget);
            } else {
                // The point of the body that is against the plane, in body coordinates:
                // straight down the normal from the centre, as far as the centre stands.
                let drop = dot(normal, self.prev_position[i]) - distance;
                self.ground_anchor[i] = contacts::Anchor {
                    position: self.prev_position[i],
                    orientation: self.prev_orientation[i],
                    hold: budget,
                    local: rotate_inv(self.prev_orientation[i], scale(normal, -drop)),
                };
            }
            self.ground_sticking.push(i as u32);
        }
        // Gathered first and written after, because the loop above has to read the set it
        // is replacing: a body is newly stuck or still stuck, and those start different
        // anchors.
        self.ground_stuck.clear();
        for &i in self.ground_sticking.iter() {
            self.ground_stuck.set(i as usize);
        }
    }

    /// Every colour, checked to name each body at most once. The precondition of every
    /// `unsafe` block in [`scatter`], asserted rather than assumed; compiled out of
    /// release builds.
    #[cfg(debug_assertions)]
    fn check_colours_are_disjoint(&self) {
        // The *live* lists rather than the colours, because they are the sets the lanes
        // actually walk. A live list is a subset of its colour reordered
        // foreground-first, so checking it is strictly what is needed: a subset of a set
        // in which no body appears twice is still such a set, and so is a permutation of
        // one, but it is the set that is touched that has to be asserted.
        let bodies = self.position.len();
        for set in self.live_joints.iter() {
            scatter::disjoint(
                bodies,
                "joint",
                set.iter().flat_map(|&k| {
                    let (a, b) = self.joints[k].bodies();
                    [a, b]
                }),
            );
        }
        for set in self.contact_colours.iter() {
            scatter::disjoint(
                bodies,
                "contact",
                set.iter()
                    .flat_map(|&k| [self.contacts[k].a, self.contacts[k].b]),
            );
        }
        scatter::disjoint(
            bodies,
            "ground",
            self.ground_colours
                .iter()
                .map(|&k| self.ground_contacts[k].body),
        );
    }

    // -- sleeping. See [`sleep`] for the reasoning behind all of it. ------------------

    /// Whether settled bodies may be left out of a step. On by default.
    ///
    /// Turning it off wakes everything, because a caller that has just turned it off is
    /// asking for the whole set to be simulated and not for whatever was asleep to stay
    /// where it was.
    pub fn set_sleeping(&mut self, sleeping: bool) {
        self.sleeping = sleeping;
        if !sleeping {
            self.wake_all();
        }
    }

    /// Whether this body is currently being simulated. A pinned body is never awake:
    /// it cannot move, so there is nothing for a step to do to it.
    pub fn is_awake(&self, i: usize) -> bool {
        self.awake.get(i)
    }

    /// How many bodies a step is currently doing any work for. The number that says
    /// whether a heap has arrived.
    pub fn awake_count(&self) -> usize {
        self.awake.count()
    }

    /// **Wakes the body's whole island**, not the body.
    ///
    /// A body in the middle of a resting stack is held still by everything around it, so
    /// disturbing it disturbs them: waking one and leaving its neighbours asleep would
    /// let it push through bodies that are no longer being solved. The island is stored
    /// as the words it occupies in the awake set, so this is an OR per sixty-four bodies.
    pub fn wake(&mut self, i: usize) {
        let Skeleton {
            awake,
            islands,
            island_of,
            still_steps,
            still_from,
            still_turn,
            position,
            orientation,
            inv_mass,
            ..
        } = self;
        let id = island_of[i];
        if id != NO_ISLAND {
            islands.thaw(id, awake.words_mut(), island_of, still_steps);
            islands.compact_if_worthwhile();
        }
        if inv_mass[i] > 0.0 {
            awake.set(i);
        }
        still_steps[i] = 0;
        still_from[i] = position[i];
        still_turn[i] = orientation[i];
    }

    /// Wakes every body. What a caller reaches for when it has changed something the
    /// solver has no way to notice -- the ground plane, a material, the whole world.
    pub fn wake_all(&mut self) {
        self.islands.clear();
        for id in self.island_of.iter_mut() {
            *id = NO_ISLAND;
        }
        for steps in self.still_steps.iter_mut() {
            *steps = 0;
        }
        self.awake.fill();
        for i in 0..self.inv_mass.len() {
            if self.inv_mass[i] <= 0.0 {
                self.awake.unset(i);
            }
            self.still_from[i] = self.position[i];
            self.still_turn[i] = self.orientation[i];
        }
    }

    /// Marks a body as background: a constraint with background at both ends is solved
    /// with [`Skeleton::set_background_iterations`] passes instead of the count
    /// [`Skeleton::step`] is given.
    ///
    /// The solver is told which bodies matter and nothing about why. A caller that knows
    /// where the attention is -- a camera, a listener, a player -- owns that judgement;
    /// a solver that tried to own it would need to be told about all three and would
    /// still be guessing. Ground contacts are not reduced: there are at most two per
    /// body, they are the cheapest constraint in the step, and the failure they produce
    /// when under-solved is a body sinking through the floor, which is the one artefact
    /// no distance excuses.
    pub fn set_background(&mut self, i: usize, background: bool) {
        if background {
            self.background.set(i);
            self.any_background = true;
        } else {
            self.background.unset(i);
            self.any_background = self.background.any();
        }
    }

    /// How many passes a background constraint gets. Clamped to the step's own count, so
    /// a caller cannot ask for more quality out there than in here.
    pub fn set_background_iterations(&mut self, iterations: usize) {
        self.background_iterations = iterations.max(1);
    }

    /// Decide what may sleep, after the step has run.
    ///
    /// Two passes over the awake bodies and one union-find. The order matters: stillness
    /// is a property of a body, sleeping is a property of an island, and a body is only
    /// entitled to sleep because everything holding it up has stopped too.
    fn settle(&mut self, dt: f64, gravity: f64) {
        if !self.sleeping || !self.awake.any() {
            return;
        }
        let n = self.position.len();

        // -- what has stopped moving ------------------------------------------------
        let Skeleton {
            position,
            orientation,
            still_from,
            still_turn,
            still_steps,
            radius,
            half_length,
            awake,
            ready,
            ..
        } = self;
        // Whether any body has now been still for its whole window. Everything below this
        // is the island machinery, and none of it can put anything to sleep unless at
        // least one body is ready -- so a heap that is still moving pays one pass over
        // its awake bodies and nothing else. That is the common case while a heap is
        // arriving, and it is worth an early exit: the union-find below is over every
        // joint and every contact, which on the heap workload is fifty thousand edges.
        let mut any_ready = false;
        ready.clear();
        awake.for_each_set(|i| {
            // The body's reach -- how far its surface is from its own centre -- is the
            // length everything here is measured against, so that the same rule serves a
            // finger bone and a torso. The floor is for a body with no extent at all,
            // which is a joint anchor rather than a thing anyone watches.
            let reach = (radius[i] + half_length[i]).max(1e-3);
            let limit = STILL_FRACTION * reach;
            // The translation alone, squared, decides most bodies while a heap is
            // arriving -- and it costs a subtract and a dot, where the whole test costs a
            // square root and two quaternion products. The turn is only asked about for a
            // body that has not already moved too far on its own.
            let gap = sub(position[i], still_from[i]);
            let moved = if dot(gap, gap) > limit * limit {
                f64::INFINITY
            } else {
                length(gap)
                    + swept(
                        turned_since(orientation[i], still_turn[i]),
                        orientation[i],
                        radius[i],
                        half_length[i],
                    )
            };
            if moved > limit {
                still_steps[i] = 0;
                still_from[i] = position[i];
                still_turn[i] = orientation[i];
            } else {
                still_steps[i] += 1;
                if still_steps[i] >= settling_steps(reach, gravity, dt) {
                    ready.set(i);
                    any_ready = true;
                }
            }
        });
        if !any_ready {
            return;
        }

        // -- which of them are held up by each other --------------------------------
        //
        // **Only over the constraints with a body ready to sleep at each end.** An island
        // sleeps only if every member is ready, so a constraint with an unready end
        // cannot be inside one -- and the union-find is the expensive half of this, fifty
        // thousand edges of pointer-chasing on the heap workload, against two bit tests
        // to reject one. While a heap is arriving almost every edge is rejected.
        //
        // The edges that straddle the boundary still have to be looked at, because a
        // ready body holding up an unready one is not entitled to sleep either. They mark
        // the ready side's component instead of joining it, which they can only do once
        // every union is in, so it is two passes rather than one.
        self.components.reset_members(n, &self.ready);
        self.unsettled.clear();
        self.unsettled.resize(n, false);
        {
            let Skeleton {
                components,
                joints,
                pairs,
                ready,
                held,
                ..
            } = self;
            held.clear();
            // **One scan, and two bit tests an edge.** Every other lookup here was a
            // random access into an array the size of the body count, and there are
            // eighteen thousand edges on the heap workload: the inverse masses this used
            // to test are already implied, because a pinned body is never awake and only
            // an awake body is ever ready.
            //
            // An edge with a ready body at one end and a moving one at the other cannot
            // be inside a sleeping island, but it does disqualify the ready side -- a
            // body holding up something that is still moving is not entitled to stop. Its
            // component is not known until every union is in, so the ready end is set
            // aside here and looked up afterwards, which costs a walk over the boundary
            // rather than a second walk over every edge.
            let mut edge = |a: usize, b: usize| match (ready.get(a), ready.get(b)) {
                (true, true) => components.union(a, b),
                (true, false) => held.push(a as u32),
                (false, true) => held.push(b as u32),
                (false, false) => {}
            };
            for joint in joints.iter() {
                let (a, b) = joint.bodies();
                edge(a, b);
            }
            // **The broad phase's pairs, not the narrow phase's contacts.** Two bodies
            // resting exactly against each other overlap by nothing, so there is no
            // contact between them -- and the better the solve gets the more often that
            // is true. Islands built on contacts then come apart under a stack that has
            // settled perfectly, every body sleeps in its own island, and dropping
            // something on the top wakes only the top. What an island needs is who
            // *could* touch, which is the list the broad phase already produced.
            for &(a, b) in pairs.iter() {
                edge(a, b);
            }
        }
        {
            let Skeleton {
                components,
                unsettled,
                held,
                ..
            } = self;
            for &i in held.iter() {
                unsettled[components.find(i) as usize] = true;
            }
        }

        // -- and which whole islands may go ------------------------------------------
        self.island_tally.clear();
        self.island_tally.resize(n + 1, 0);

        // Counting sort of the sleeping candidates by their island's root, so that each
        // island's members come out together and in increasing order -- which is what
        // lets the island be stored as words rather than as a list of indices.
        let Skeleton {
            components,
            unsettled,
            island_tally,
            island_list,
            ready,
            ..
        } = self;
        ready.for_each_set(|i| {
            let root = components.find(i as u32) as usize;
            if !unsettled[root] {
                island_tally[root] += 1;
            }
        });
        let mut running = 0u32;
        for slot in island_tally.iter_mut() {
            let count = *slot;
            *slot = running;
            running += count;
        }
        if running == 0 {
            return;
        }
        island_list.clear();
        island_list.resize(running as usize, 0);
        ready.for_each_set(|i| {
            let root = components.find(i as u32) as usize;
            if !unsettled[root] {
                island_list[island_tally[root] as usize] = i as u32;
                island_tally[root] += 1;
            }
        });

        // `island_tally[root]` now holds the end of that root's run, and the previous
        // root's end is where it began, so the runs are read off in one pass.
        let mut from = 0usize;
        for root in 0..n {
            let upto = self.island_tally[root] as usize;
            if upto == from {
                continue;
            }
            let members = &self.island_list[from..upto];
            let id = self.islands.freeze_sorted(members);
            for &member in members {
                let i = member as usize;
                self.awake.unset(i);
                self.island_of[i] = id;
                // A sleeping body must read back as stopped rather than as whatever the
                // last correction happened to leave on it, or it wakes with a shove.
                self.velocity[i] = (0.0, 0.0, 0.0);
                self.angular_velocity[i] = (0.0, 0.0, 0.0);
                self.prev_position[i] = self.position[i];
                self.prev_orientation[i] = self.orientation[i];
            }
            from = upto;
        }
    }
}

/// One lane's worth of contacts, solved and written back in place.
///
/// Shared between a colour's slice and the serial overflow, because the two differ only
/// in which list they walk and how many lanes are walking it.
///
/// # Safety
/// Every body named by `set` must be owned by the calling lane for the duration: within a
/// colour that is the colouring plus [`crew::Lane::span`], and for the overflow it is the
/// single lane that runs it. See [`Skeleton::solve_pass`] for the argument in full.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn solve_some_contacts(
    set: &[usize],
    contacts: &[Contact],
    bodies: &Bodies,
    impulse: &scatter::Cells<Spent>,
    inv_mass: &[f64],
    inv_inertia: &[(f64, f64, f64)],
    radius: &[f64],
    friction: f64,
    rolling: f64,
) {
    for &k in set {
        let contact = contacts[k];
        let first = bodies.gather(contact.a, inv_mass, inv_inertia, radius);
        let second = bodies.gather(contact.b, inv_mass, inv_inertia, radius);
        let (corrections, totals) =
            solve_contact(contact, &first, &second, friction, rolling, impulse.get(k));
        impulse.set(k, totals);
        bodies.apply(corrections);
    }

}

/// How far a turn of `turned` actually carries the surface of a capsule of this size.
///
/// **A capsule is a surface of revolution, so a turn about its own long axis moves
/// nothing.** Treating every axis alike costs a settling test most of its budget on a
/// bone that is merely spinning where it lies: measured on a settled pile of forty, the
/// axis-blind version reported a surface sweep half again as large as the real one and
/// left the pile permanently awake. Across the axis the far end travels the body's whole
/// reach; along it, the surface only goes round at the radius.
#[inline]
fn swept(
    turned: (f64, f64, f64),
    orientation: Quaternion,
    radius: f64,
    half_length: f64,
) -> f64 {
    let axis = rotate(orientation, (0.0, 1.0, 0.0));
    let along = dot(turned, axis);
    let across = length(sub(turned, scale(axis, along)));
    across * (radius + half_length) + along.abs() * radius
}

/// `q` advanced by angular velocity `w` over `dt`, renormalised. The small-angle
/// integrator every position-based solver uses: exact enough over a frame and far cheaper
/// than an exponential map.
#[inline]
fn integrate_spin(q: Quaternion, w: (f64, f64, f64), dt: f64) -> Quaternion {
    let spin = Quaternion {
        w: 0.0,
        x: w.0,
        y: w.1,
        z: w.2,
    }
    .multiply(&q);
    renormalized(Quaternion {
        w: q.w + 0.5 * dt * spin.w,
        x: q.x + 0.5 * dt * spin.x,
        y: q.y + 0.5 * dt * spin.y,
        z: q.z + 0.5 * dt * spin.z,
    })
}

/// Everything one joint wants done, as two corrections. Reads only; the caller applies.
fn solve_joint(joint: Joint, first: &Pose, second: &Pose) -> [Correction; 2] {
    let mut out = [Correction::none(); 2];
    let (a, b) = joint.bodies();
    out[0].body = a;
    out[1].body = b;

    let (anchor_a, anchor_b) = match joint {
        Joint::Ball {
            anchor_a, anchor_b, ..
        } => (anchor_a, anchor_b),
        Joint::Hinge {
            anchor_a, anchor_b, ..
        } => (anchor_a, anchor_b),
    };

    // -- the positional half: the two anchors are one point ----------------------
    let ra = rotate(first.orientation, anchor_a);
    let rb = rotate(second.orientation, anchor_b);
    let error = sub(add(second.position, rb), add(first.position, ra));
    if let Some(n) = normalized(error) {
        let c = length(error);
        let wa = generalised_inverse_mass(first, ra, n);
        let wb = generalised_inverse_mass(second, rb, n);
        let total = wa + wb;
        if total > 1e-12 {
            let impulse = scale(n, c / total);
            accumulate(&mut out[0], first, ra, impulse, false);
            accumulate(&mut out[1], second, rb, scale(impulse, -1.0), false);
        }
    }

    // -- and, for a hinge, the axis and the range on it --------------------------
    if let Joint::Hinge {
        axis_a,
        axis_b,
        min,
        max,
        ..
    } = joint
    {
        let world_a = rotate(first.orientation, axis_a);
        let world_b = rotate(second.orientation, axis_b);
        let misaligned = cross(world_b, world_a);
        if let Some(n) = normalized(misaligned) {
            let angle = length(misaligned).clamp(-1.0, 1.0).asin();
            share_turn(&mut out, first, second, n, angle);
        }

        if let Some(axis) = normalized(world_a) {
            let angle = hinge_angle(
                first.orientation,
                second.orientation,
                axis,
                axis_a,
                axis_b,
            );
            let excess = if angle < min {
                angle - min
            } else if angle > max {
                angle - max
            } else {
                0.0
            };
            if excess != 0.0 {
                share_turn(&mut out, first, second, axis, excess);
            }
        }
    }

    out
}

/// The angle between two bodies about a hinge, measured from a reference perpendicular to
/// the axis in each -- so it is the swing, with the twist the axis constraint removes left
/// out of it.
fn hinge_angle(
    a: Quaternion,
    b: Quaternion,
    axis: (f64, f64, f64),
    axis_a: (f64, f64, f64),
    axis_b: (f64, f64, f64),
) -> f64 {
    let reference = perpendicular(axis);
    let in_a = rotate(a, reference);
    let in_b = rotate(b, rotate_into(reference, axis_b, axis_a));
    dot(cross(in_a, in_b), axis).atan2(dot(in_b, in_a))
}

/// `w = inv_m + (r x n) . I^-1 (r x n)`: how much a unit impulse along `n` applied at `r`
/// actually moves this body. The denominator of every positional correction.
#[inline]
fn generalised_inverse_mass(body: &Pose, r: (f64, f64, f64), n: (f64, f64, f64)) -> f64 {
    let rn = cross(r, n);
    body.inv_mass + dot(rn, body.world_inv_inertia.apply(rn))
}

/// Fold one impulse at `r` into a body's correction.
fn accumulate(
    into: &mut Correction,
    body: &Pose,
    r: (f64, f64, f64),
    impulse: (f64, f64, f64),
    free: bool,
) {
    if !body.movable() {
        return;
    }
    let move_by = scale(impulse, body.inv_mass);
    if free {
        into.free_translation = add(into.free_translation, move_by);
    } else {
        into.translation = add(into.translation, move_by);
    }

    // The orientation update is `q + (1/2) dw q`, and `dw q` factors, so the *delta* to
    // left-multiply is `1 + (1/2) dw` -- independent of the orientation it will be
    // applied to, which is exactly what lets this be computed now and applied later.
    let dw = body.world_inv_inertia.apply(cross(r, impulse));
    let delta = renormalized(Quaternion {
        w: 1.0,
        x: 0.5 * dw.0,
        y: 0.5 * dw.1,
        z: 0.5 * dw.2,
    });
    // Composed onto whatever this body has already been asked to do by this joint.
    if free {
        into.free_rotation = renormalized(delta.multiply(&into.free_rotation));
    } else {
        into.rotation = renormalized(delta.multiply(&into.rotation));
    }
}

/// Turn two bodies apart about a world axis, split by their inertias, into their
/// corrections.
fn share_turn(
    out: &mut [Correction; 2],
    a: &Pose,
    b: &Pose,
    axis: (f64, f64, f64),
    angle: f64,
) {
    if angle.abs() < 1e-9 {
        return;
    }
    let ia = dot(axis, a.world_inv_inertia.apply(axis));
    let ib = dot(axis, b.world_inv_inertia.apply(axis));
    let total = ia + ib;
    if total <= 1e-12 {
        return;
    }
    if a.inv_inertia != (0.0, 0.0, 0.0) {
        let turn = Quaternion::from_axis_angle(axis, angle * ia / total);
        out[0].rotation = renormalized(turn.multiply(&out[0].rotation));
    }
    if b.inv_inertia != (0.0, 0.0, 0.0) {
        let turn = Quaternion::from_axis_angle(axis, -angle * ib / total);
        out[1].rotation = renormalized(turn.multiply(&out[1].rotation));
    }
}

/// How far a body has turned since the step began, as an axis-angle vector.
///
/// The `w < 0` flip keeps the short way round, for the same reason the velocity writeback
/// does: a quaternion and its negation are the same orientation.
#[inline]
fn turned_since(now: Quaternion, before: Quaternion) -> (f64, f64, f64) {
    let delta = now.multiply(&before.conjugate());
    let sign = if delta.w < 0.0 { -1.0 } else { 1.0 };
    scale((delta.x, delta.y, delta.z), 2.0 * sign)
}

/// Rolling resistance at a contact, as the angular half of Coulomb.
///
/// The resisting torque a real contact patch applies is the coefficient times the normal
/// force times the radius, so over a step the resisting angular impulse is bounded by the
/// coefficient times the radius times the normal impulse -- carried across the passes
/// exactly like the tangential one, and for the same reason, including that what is
/// bounded is the resultant rather than the distance it walked. A pass asks for exactly
/// the roll that has happened and no more, so resistance never turns into a push; what it
/// may do, once the cone has clipped an earlier pass, is give back some of what that pass
/// over-applied.
///
/// Only rotation about axes *in* the contact plane is resisted. Rotation about the normal
/// is a body spinning on the spot, which is a different effect with a different arm.
/// `b` is `None` when the other side of the contact is the ground, which does not turn
/// and takes none of the correction.
#[allow(clippy::too_many_arguments)]
fn resist_rolling(
    out: &mut [Correction; 2],
    a: &Gathered,
    b: Option<&Gathered>,
    normal: (f64, f64, f64),
    arm: f64,
    normal_impulse: f64,
    spent: (f64, f64, f64),
) -> (f64, f64, f64) {
    let mut relative = turned_since(a.now.orientation, a.prev_orientation);
    if let Some(b) = b {
        relative = sub(relative, turned_since(b.now.orientation, b.prev_orientation));
    }
    let rolled = sub(relative, scale(normal, dot(relative, normal)));
    let Some(axis) = normalized(rolled) else {
        return spent;
    };
    let ia = dot(axis, a.now.world_inv_inertia.apply(axis));
    let ib = match b {
        Some(b) => dot(axis, b.now.world_inv_inertia.apply(axis)),
        None => 0.0,
    };
    let total = ia + ib;
    if total <= 1e-12 {
        return spent;
    }

    // A cone on the resultant, not a running total of what has been spent: a roll that
    // reverses between passes has to give its budget back, or the reversals eat the
    // coefficient. Exactly the argument [`contacts::Spent`] makes for the tangential
    // half, one dimension over -- and this is the dimension where reversals are most
    // likely, because the friction impulse that turns a body is applied at an arm and the
    // normal impulse that untilts it is applied at another.
    let wanted = scale(axis, -length(rolled) / total);
    let (delta, total_impulse) = contacts::cone(spent, wanted, arm * normal_impulse);
    let size = length(delta);
    let Some(along) = normalized(delta) else {
        return spent;
    };

    // Split by inertia, the same way a hinge's range is, and opposing the roll. The
    // angular impulse turns each body by its own inverse inertia along the axis it acts
    // on, which is the axis of the correction rather than of the roll once anything has
    // been carried over from an earlier pass.
    let ia = dot(along, a.now.world_inv_inertia.apply(along));
    let ib = match b {
        Some(b) => dot(along, b.now.world_inv_inertia.apply(along)),
        None => 0.0,
    };
    if ia > 0.0 {
        let turn = Quaternion::from_axis_angle(along, size * ia);
        out[0].rotation = renormalized(turn.multiply(&out[0].rotation));
    }
    if ib > 0.0 {
        let turn = Quaternion::from_axis_angle(along, -size * ib);
        out[1].rotation = renormalized(turn.multiply(&out[1].rotation));
    }
    total_impulse
}

/// Any unit vector at right angles to `axis`. Which one does not matter -- it is only ever
/// a shared reference for measuring an angle, and both bodies measure from the same one.
fn perpendicular(axis: (f64, f64, f64)) -> (f64, f64, f64) {
    let candidate = if axis.0.abs() < 0.9 {
        (1.0, 0.0, 0.0)
    } else {
        (0.0, 1.0, 0.0)
    };
    normalized(cross(axis, candidate)).unwrap_or((0.0, 1.0, 0.0))
}

/// Carry a vector given about `from` over to the frame where the hinge axis is `to`, so
/// both bodies measure their angle from the same reference.
fn rotate_into(
    v: (f64, f64, f64),
    from: (f64, f64, f64),
    to: (f64, f64, f64),
) -> (f64, f64, f64) {
    let (Some(f), Some(t)) = (normalized(from), normalized(to)) else {
        return v;
    };
    let Some(axis) = normalized(cross(f, t)) else {
        return v;
    };
    let angle = dot(f, t).clamp(-1.0, 1.0).acos();
    rotate(Quaternion::from_axis_angle(axis, angle), v)
}

#[cfg(test)]
mod tests;
