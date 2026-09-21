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
//!   `Vec<Body>` each lane would be a gather out of a 160-byte struct.
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
//! joints plus contacts as the unit that *sleeps* -- and a step where nothing is awake
//! returns before it touches memory. Measured on ten thousand capsules resting on the
//! ground in stacks of three: **4.5 to 6.5 ms a step down to nothing measurable**, the
//! spread being what a shared machine does to a twenty-step timing. With nothing asleep it
//! costs nothing that can be told from the noise, which is what the translation-only early
//! exit in [`Skeleton::settle`] is for.
//!
//! **The island is not the unit that wakes**, and reading it as one cost more than
//! anything else sleeping saved on the workload sleeping is for. A settled field is one
//! island, so the first thing to touch any of it woke all of it: four contacts woke four
//! hundred bodies, and a step that cost nothing went to milliseconds. What wakes a body is
//! now that something *moving* came within reach of it, or that something touched it --
//! [`Skeleton::find_pairs`] and [`Skeleton::wake_touched`], with the whole argument and the
//! measurements in [`sleep`]. Measured on a field of two thousand with one heavy body
//! ploughing through it, halfway down: 2049 of 2049 awake at 2.88 to 3.09 ms a step
//! becomes 346 of 2049 at 1.80 to 2.43 ms, against a *higher* contact count.
//!
//! # What the fixtures carry, and how to read the tables below
//!
//! Almost every claim in this file is a before-and-after on a named benchmark fixture, and
//! each of those tables carries a `contacts` column. **Those columns record what the
//! fixture carried when that measurement was taken, and several of them no longer match
//! what it carries now.** They are left as they were, because rewriting a number inside a
//! comparison nobody has re-run would be inventing a measurement. What follows is the
//! authority on the present state; `cargo bench --bench articulated -- --test` prints it in
//! a few seconds and is the way to check it rather than to trust it.
//!
//! ```text
//!   one              17 bodies, 16 joints, 8 contacts, 16 of 17 awake
//!   joints_only  10,200 bodies, 9,600 joints, 0 contacts, 4 colours, all awake
//!   pile         10,200 bodies, 9,600 joints, 4,800 contacts, 9,600 awake
//!   arriving     10,200 bodies, 9,600 joints, 11,600 contacts, all awake
//!   ploughing     2,049 bodies over 23.2 by 15.8 m, 273 contacts, 330 awake
//!   crushing      4,800 bodies, 9,582 candidate pairs, 0 contacts settled
//! ```
//!
//! The one that matters most is `pile`, because it is the fixture most of this file
//! divides by: it carries **4,800** contacts, where tables below say 8,400 and then 7,800.
//! Any "cost per contact" read off those is out by a factor of one and a half to one and
//! three quarters. The counts move when the *solve* changes and not only when the fixture
//! does -- the hinge fix further down this page changed how a heap collapses, and therefore
//! how much of it is touching at the step the bench names -- which is why a contact count
//! quoted in a table is a fact about that measurement rather than about the fixture.
//!
//! The retirement table further down is the one to be most careful with: it is headed as a
//! lane of eight hundred capsules, and that fixture has since been widened into a field of
//! 4,801 and renamed `crushing`. Its rows cannot be reproduced by running the bench today.
//! The conclusion drawn from it -- that the cost per live body is flat while three quarters
//! of the field retires -- is restated on the current fixture in `benches/articulated.rs`,
//! which is where to read it.
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
//! sixteen draws a relative 1e-12 apart, because one draw says nothing. That mattered for
//! the numbers already on this page too: over sixteen draws the commit this was written
//! against left two draws of a column of eight still awake after twelve thousand steps, and
//! two draws of the seventeen-bone rig `a_settled_rig_stays_where_it_settled` is written
//! against travelled 6.37 and 0.32 of a reach with straightnesses of 0.94 and 0.50 -- which
//! is the walk, at the size it was before the ground anchor, on seeds next door to the one
//! the law runs. That spread is what the section below on the staggered sub-passes went
//! after, so read the figures in this section as the comparison between three anchor shapes
//! that they are rather than as the solver's current numbers.
//!
//! ```text
//!   steps to sleep      column of 3  4         5          6          7          8
//!   before the stagger            381  366..378  531..1768  243..2305  775..2704  676..8239
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
//! # A pass solves every normal before it solves any friction
//!
//! The section above closes on the measurement that explains the rest of this page: a pair
//! in a heap can sit strictly inside its Coulomb cone and still slide a millimetre, because
//! the friction correction that removed the slide is undone *inside the same step* by the
//! normal corrections of the contacts either body has elsewhere. It is also why the cone
//! test proves sticking against the plane -- a reference that does not move, whose contact
//! is solved as one patch -- and does not prove it between two bodies.
//!
//! A contact used to compute and apply its normal correction and its friction correction in
//! one call, so within a single pass contact A's friction was applied and then contact B's
//! normal correction moved one of A's bodies and undid it. **Friction was being asked to
//! hold against a surface that was still moving underneath it.** So a pass is now two
//! sub-passes over the same coloured stage list -- every normal correction, then every
//! tangential one -- and friction acts on a configuration the normals have already agreed
//! on. Nothing else about the pass changes: the colours are the colours, so a lane's slice
//! of a stage is disjoint in exactly the way [`scatter`] requires, and the second traversal
//! is checked by the same `check_colours_are_disjoint`.
//!
//! **The tangential half is the whole tangential half**, friction and rolling resistance
//! together, and it is gated on the normal impulse the step has spent rather than on the
//! overlap that is left -- see [`contacts::solve_contact_friction`], which is where that
//! argument is, because getting it wrong makes friction stop acting entirely.
//!
//! Measured against the chaos spread rather than single runs. The rig is the seventeen-bone
//! one `a_settled_rig_stays_where_it_settled` is written against, over twenty-four draws a
//! relative 1e-12 apart; the columns are capsules lying flat on the plane, sixteen draws,
//! capped at twelve thousand steps; the piles are eight draws of twenty, forty and sixty
//! capsules dropped in a heap, the median body's surface travel over four hundred and
//! eighty steps as a fraction of its own reach.
//!
//! ```text
//!   the rig, median bone's drift over 32 windows      past 0.113   worst   straightness
//!   before                                              3 of 24     6.37       0.94
//!   after                                               1 of 24     0.19       0.31
//!
//!   steps to sleep    column of 3    4         5          6          7          8
//!   before                    381  366..378  531..1768  243..2305  775..2704  676..8239
//!                                                                          and 2 of 16 never
//!   after                     325       220   207..224   298..519  648..1132  593..2961
//!
//!   settled pile of                20              40              60
//!   before                 0.000..0.137    0.063..0.135    0.051..0.193
//!   after                  0.000..0.008    0.037..0.105    0.107..0.203
//! ```
//!
//! The bolt is the result. What used to happen is that one draw in a dozen found somewhere
//! to go and travelled six reaches with a straightness of 0.94, which is a rig being
//! carried; the worst draw of twenty-four now covers a fifth of a reach at a straightness of
//! 0.31, which is under the 0.5 that separates carried from jostled. The columns lose their
//! tail in the same way -- nothing fails to settle, and the spread of a column of eight
//! falls from a factor of twelve to a factor of five. The typical rig draw is *worse*, from
//! 0.002 of a reach to 0.021: the ratchet has become a jostle, and there is more of it.
//!
//! **It costs about five to ten per cent**, which is the second traversal of the stage list
//! and its barriers, not arithmetic -- the flops are the same ones. Two runs of each, at
//! eight iterations: `one/8` 36.1 and 37.6 us against 37.8 and 40.5; `pile/8` 3.11 and 3.07
//! ms against 3.54 and 3.21; `arriving/8` 6.85 and 6.13 ms against 7.04 and 6.90.
//!
//! # The iteration count cannot come down, and that is the point
//!
//! The reason eight passes was the number is in the section above on the iteration count,
//! and it is unchanged. What has changed is the direction of the error. Before this, *more
//! passes walked further* -- 0.090 m/s at two thousand and forty-eight against 0.036 at
//! eight -- because the bias was baked into the order and every extra pass applied it
//! again. It is now a convergence shortfall instead, which iteration reduces. The rig,
//! sixteen draws, how many go to sleep inside two thousand steps:
//!
//! ```text
//!   passes                8       10      12      16      24
//!   before             2/16     3/16    3/16    2/16    7/16
//!   after              0/16     4/16    2/16    5/16    8/16
//!   after, median window travel of a bone, as a fraction of its reach:
//!                     0.032    0.011   0.004   0.010   0.008
//! ```
//!
//! So the count may not come down; the residual at eight passes is what is left of the
//! solve rather than a bias in it, and it sits a little above the 0.02 of a reach a body
//! must stay inside to be called still. That is the one defect of the four this did not
//! close: a self-colliding rig still does not reliably sleep, and at eight passes it sleeps
//! in none of sixteen draws where it used to sleep in two. What holds it awake is no longer
//! a direction -- it is that the rig now props itself up on its own friction instead of
//! slumping, and a propped rig sits at a marginal equilibrium that eight passes cannot
//! resolve to stillness. Measured, a settled rig holds 0.73 to 0.82 J of kinetic energy
//! where it used to hold 0.07 to 0.36, and stands about a fifth higher.
//!
//! # Four ways of taking the energy back out, and why none of them is here
//!
//! * **Leave rolling resistance in the normal half**, so that only friction is staggered.
//!   It is the natural minimal version and it is much worse than doing nothing: the rig
//!   bolts in four draws of twelve, at 8.8, 3.9, 3.1 and 1.8 reaches, against one in twelve
//!   before the change. A column of six gains a draw that never settles and the piles drift
//!   further. Whatever the tangential half is doing for the rig, it is doing it as one
//!   thing.
//! * **Refuse friction where the contact has opened**, by more than one step of gravity's
//!   sag -- `g dt^2`, the scale a resting contact's own overlap has, already derived here as
//!   `anchor_reach`. It is the obvious objection to the gate the tangential half uses, and
//!   it buys real things: the rig sleeps in three draws of sixteen, the best any of this has
//!   produced, and the columns tighten again. It also brings the bolt straight back -- two
//!   draws of twelve, the worst at 2.7 reaches. **Intermittent friction is what the rig
//!   walks on.** Any variant that lets the tangential half skip a pass the normal half ran
//!   reproduces the defect, which is the same statement as the one at the top of this
//!   section, seen from the other side.
//! * **Bound what a normal correction may read back as velocity** by the closing the step
//!   itself drove, measured over the predict before any correction existed, and charge the
//!   rest as `Correction::free_translation`. This is the "do not put the energy in" answer
//!   and it is derived rather than tuned: the slide a pass sees includes every correction
//!   applied since the step began, the tangential half's included, so charging the whole of
//!   it as velocity hands the bodies the solver's own repair work as momentum they never
//!   earned. It does not work, and the reason is worth more than the attempt: **in a
//!   position-based solver, charging its own corrections as velocity is the only channel by
//!   which "stopped" propagates up a stack.** Two capsules resting one on the other sag
//!   together, so the closing measured over the predict at the contact between them is zero,
//!   and it is the ground's correction to the lower body that closes the pair and stops the
//!   upper one. Bound that and the upper body is never stopped at all: a body dropped on a
//!   sleeping stack sinks three millimetres into it and a settled pile of forty never comes
//!   to rest. Bounding only the ground's share is worse again, four unit tests rather than
//!   two.
//! * **The same bound pooled over the body**, which is the obvious repair of the one above
//!   and the third time this module has met the granularity question -- the ground patch
//!   was two point constraints until it became one block solve, and contact memory was per
//!   contact until it was pooled per pair patch. It was built and measured, and the
//!   pooling does exactly what the pattern says it should: the propagation channel stays
//!   open, `a_pile_settles_into_a_heap_rather_than_rolling_away` passes again, and so do
//!   all thirteen laws, the centre-of-mass one included -- once the fraction charged is
//!   the *same* for both ends of a pair, which it has to be, because the share of an
//!   impulse that is momentum exchange rather than positional repair cannot be one number
//!   for the body that receives it and another for the body that gives it. Charging each
//!   end against its own bound moves a free skeleton's centre of mass by 1.8 m.
//!
//!   It still does not work, and the reason closes the whole family rather than this
//!   member of it. **The bound is satisfied by the very thing it is aimed at.** "No faster
//!   than it arrived" bounds growth, not a cycle: a body going round a limit cycle at a
//!   constant speed arrives at that speed every step, so the bound is slack at exactly the
//!   state it was built to forbid. Measured on the settled rig, the median bone's speed is
//!   0.054 to 0.063 m/s on every draw -- a third of `g dt`, the same third on all six, which
//!   is a cycle and not a spread. And what it does bite on it cannot dissipate: the share it
//!   refuses to charge goes to `Correction::free_translation`, which moves the body without
//!   reading back as velocity, so the momentum it declines to take out is still in the body
//!   next step. Measured: the rig travels past the jostling allowance in ten draws of
//!   twenty-four, the worst at 1.19 reaches with a straightness of 0.62, a settled pile of
//!   twenty drifts 0.000..0.199 of a reach against 0.000..0.008, and a capsule dropped on a
//!   sleeping stack of three still finishes six millimetres inside the top of it.
//!
//!   There is no third destination for the refused share. Charging it is this commit;
//!   making it free preserves it; dropping it is under-relaxation, which is measured and
//!   rejected in the section above. A sink would have to be something that removes momentum
//!   without being asked, and this module does not have one and may not have a tuned one.
//!
//! So the state of it was: the walk gone, the sleeping not, and what was left a limit
//! cycle at a third of one step of gravity that eight passes did not resolve and no bound
//! on the read-back velocity could see. That closing argument is what the section below
//! is: the one shape this module did not have.
//!
//! # The velocity pass, which is the half of XPBD this was missing
//!
//! Everything above is a **positional** solver: constraints move bodies, and velocity is
//! whatever the movement divided by the step says it is. Muller's formulation has a second
//! traversal after that one, over the same contacts, which corrects *velocities* directly
//! -- it is where restitution lives, and where friction acts on what the positional solve
//! left moving. This module never had it, and the limit cycle above is what its absence
//! looks like: a contact push read back as speed, a joint pulling it back, and nothing
//! anywhere in the step entitled to say what the relative velocity of two resting surfaces
//! ought to be.
//!
//! **It needed a correction shape that did not exist.** `Correction::translation` moves the
//! body and lets the step read the move back as speed; `Correction::free_translation` moves
//! it and hides the move, by shifting where it came from as well. Neither can change a
//! velocity without changing a position. The third kind moves where the body came from
//! *instead* of where it is -- see [`Charge`] and [`scatter::Bodies::apply_velocity`] --
//! and it is the only thing in the module that can take momentum out and leave the solve's
//! answer standing. That is the sink the section above says does not exist; it exists now.
//!
//! ## What the pass says, and where each number comes from
//!
//! * **A resting contact is perfectly inelastic.** The relative normal velocity of two
//!   loaded surfaces should be zero when the step ends, and restitution is the coefficient
//!   that would make it something else. Zero is a statement about rigid capsules lying on
//!   one another, not a damping constant; a caller who wants a bouncy contact wants
//!   `-e * closing` as the target and this is the only place in the module that number
//!   could act.
//! * **And a contact may take back only what it gave.** Cancelling a separating velocity
//!   means removing normal impulse from the step, and the most that can be removed is the
//!   part the contact actually handed the bodies as momentum -- the *driven* share, which
//!   is the overlap the step itself made and the only share charged as velocity. Take more
//!   and the pair is being pulled together, which a unilateral constraint may never do.
//!   [`contacts::Spent::driven`] is that total; it is the same "budget for the step" shape
//!   Coulomb's cone already has, one law over.
//! * **Coulomb at the velocity level, out of the same cone.** The tangential half asks for
//!   the impulse that stops the slip and adds it to the resultant the positional passes
//!   accumulated, clipped by the same `friction * spent.normal`. It does **not** get a
//!   second budget, and that is not conservatism: a contact that could spend
//!   `friction * N` twice in one step has a coefficient of twice `friction`, and a slope
//!   that should let go at Coulomb's angle would hold to twice its tangent. Sharing the
//!   cone also sorts the two cases with nothing to decide -- a sliding contact has spent it
//!   in the positional passes and slides exactly as it did, and a sticking one has spent
//!   only what it took to hold still.
//!
//! ## It is simultaneous, and that is the whole difference
//!
//! The first version was an ordered sweep over the coloured stages, like the positional
//! solve, and it is the same mistake one level up. A Gauss-Seidel sweep round a loop leaves
//! the same small bias every step and the ground rectifies it into travel -- which is the
//! diagnosis this page already carries for the positional ratchet, arriving a second time
//! by the same route. Measured on the seventeen-bone rig over twenty-four draws: the worst
//! drift went from 0.19 of a reach to 6.1 and its straightness from 0.31 to 0.90, and
//! *converging* that pass -- eight sweeps instead of one -- made all twenty-four walk two
//! reaches at a straightness of 0.56, which is what a fixed point that depends on the order
//! looks like.
//!
//! So the pass reads one state and every constraint answers against it. The read-back sweep
//! runs **before** the pass as well as after: the first gives it the velocities, the second
//! delivers what it decided. Nothing rewrites those arrays while the pass is running, so
//! the answer does not depend on which contact was visited first, and determinism is
//! unaffected -- within a colour the bodies are disjoint exactly as [`scatter`] requires,
//! and across colours the barrier order is fixed.
//!
//! What a simultaneous pass costs is that a body named by three contacts is asked for three
//! full corrections at once. [`Skeleton::velocity_share`] is the mean that answers it, and
//! **both ends of a pair take the same one** -- the smaller of the two -- because the share
//! of an impulse that is momentum exchange cannot be one figure for the body receiving it
//! and another for the body giving it. Charging each end against its own is the mistake
//! that moved a free skeleton's centre of mass by 1.8 m, and
//! `a_skeleton_left_to_itself_does_not_move_its_own_centre_of_mass` is the guard on it.
//!
//! ## Two things deliberately left out of it
//!
//! * **The plane's tangential half.** The ground patch already remembers where it stuck and
//!   what it stuck under, and its friction answers this step's slip and the slip earlier
//!   steps left behind, at two different authorities -- that memory is what stopped the walk
//!   in the first place. A velocity-level friction on the same patch answers the same slip
//!   a second time with none, and the two disagree about what the past was. Measured, adding
//!   it takes the rig from nine draws of sixteen going to sleep to one, while the drift does
//!   not move. The plane is the one surface whose tangential half is already complete.
//! * **Rolling resistance.** It is the module's model of a *deformed patch* -- the load
//!   moves ahead of the contact point and the offset is a torque -- and it exists precisely
//!   because Coulomb cannot see a roll: the contact point of a rolling body is
//!   instantaneously still, so there is no relative surface velocity there for a
//!   velocity-level law to act on. What an angular version would act on instead is the
//!   relative spin of the two bodies, which between two bones of one skeleton is mostly the
//!   joints doing their job. Measured, resisting it takes the rig from no draw of
//!   twenty-four travelling past the jostling allowance to six, the worst at 4.4 reaches.
//!
//! ## What it buys
//!
//! The rig is the seventeen-bone one, self-collision on, eight passes. The residual is the
//! median bone's speed on a settled rig, six draws, as a fraction of `g dt`; the cycle is
//! what the section above measured at a third of one step of gravity, the same third on
//! every draw. Columns are capsules lying flat on the plane, sixteen draws, capped at
//! twelve thousand steps; piles are eight draws of the median body's surface travel over
//! four hundred and eighty steps.
//!
//! ```text
//!   rig, 16 draws, how many sleep inside 2000 steps at 8 passes
//!     before   0 of 16                      after   9 of 16   (14 of 32)
//!   rig, settled residual as a fraction of g dt
//!     before   0.335 .. 0.388               after   0.062 .. 0.182
//!   rig, 24 draws, drift over 32 windows and the worst straightness
//!     before   1 past 0.113, worst 0.19 at 0.31
//!     after    0 past 0.113, worst 0.083 at 0.12
//!
//!   steps to sleep    column of 3    4         5          6          7          8
//!   before                    325       220   207..224   298..519  648..1132  593..2961
//!   after                     363   630..767   225..339   204..564   746..920   304..427
//!
//!   settled pile of                20              40              60
//!   before                 0.000..0.008    0.037..0.105    0.107..0.203
//!   after                  0.000..0.090    0.073..0.124    0.077..0.218
//! ```
//!
//! **The limit cycle is dissipated**: the residual falls by a factor of two to five and,
//! more to the point, stops being the same third on every draw -- it is a spread now rather
//! than an attractor. The rig also stops propping itself up: its total energy settles 10 to
//! 25 per cent lower and does not creep, where before it held 0.73 to 0.82 J of kinetic
//! energy for as long as it was watched. And the walk does not come back -- the worst draw
//! of twenty-four is better than it was on every one of the three statistics the law
//! weighs.
//!
//! **What it does not close.** Nine draws of sixteen is not most of them, and at
//! twenty-four passes the rig goes to sleep in five draws of sixteen where it used to
//! manage eight -- more passes no longer help as much, because what is left is not a
//! convergence shortfall. Two of sixteen draws still sit at 0.43 and 0.58 of `g dt`, so
//! the attractor is gone but something in the same band is still reachable. A settled pile
//! of twenty gains a tail it did not have (0.008 to 0.090 over eight draws) and one of
//! forty drifts about a fifth further, both well inside what
//! `a_settled_pile_wanders_but_does_not_drift` allows and both a real cost. A column of
//! four takes three times as long to settle, while a column of eight takes a seventh as
//! long and the worst column of any height falls from 2961 steps to 920.
//!
//! ## Sweeping it more than once, which does not work and is not a convergence question
//!
//! The pass is one Jacobi sweep of a velocity-level complementarity problem, so it
//! propagates a correction exactly one contact deep, and in a heap a body's velocity
//! depends on its neighbours' through the contacts it shares. The obvious next move is to
//! sweep it two or three times, and the obvious expectation is that the draws which fail
//! to sleep -- close rather than badly wrong -- come over. [`VELOCITY_SWEEPS`] is the
//! count, the sweeps share one step's budget so that N of them cannot spend the
//! coefficient N times, and the whole of it was measured:
//!
//! ```text
//!   sweeps                    1        2        3        4
//!   draws that sleep      9/16     1/16    13/16     0/16
//!   residual, fraction of g dt
//!                    .062..182  .10..2.1  .19..213  .024..064
//!   a settled rig's energy   stable  UNSTABLE   stable   stable
//! ```
//!
//! **Good at one, catastrophic at two, best at three, useless at four.** That is not an
//! iteration converging; it is a coefficient that happens to suit a scene, and it is the
//! same shape this page rejects under-relaxation for. At two sweeps a settled rig gains
//! energy, peaking at 1024 J against the 215 it settles from. The reason it is not a
//! convergence question is the one the joints already taught: each sweep leaves the
//! *joints'* velocity constraints violated by what it just did, only the next step's
//! positional solve repairs that, and the repair is charged as velocity -- so sweeping
//! harder feeds the loop rather than closing it. The count stays at one.
//!
//! **And four sweeps names what is left of the defect.** It is the stillest the rig has
//! ever been -- 0.024 to 0.064 of `g dt` against 0.062 to 0.182, and a median drift of
//! 0.0000 of a reach over four hundred and eighty steps -- and it sleeps in none of sixteen
//! draws over twelve thousand steps. A drift criterion that a frozen median bone will not
//! satisfy means what is still moving is one or two bones of seventeen rather than the rig.
//! So whatever remains is **local to a bone**, not a cycle around the whole loop, and it is
//! a different thing to go after than the one the velocity pass was built for.
//!
//! **What it costs, and the cost lands exactly where the pass runs.** A traversal and a
//! half -- two velocity stages over the contacts, one over the ground, and the extra
//! read-back sweep -- against eight positional passes. Three alternating rounds of
//! prebuilt binaries, so that neither build state nor a scene drifting under the benchmark
//! is in the comparison; every fixture is timed from one named state, and `benches` has
//! the account of why that had to be fixed first.
//!
//! ```text
//!                   contacts        before              after
//!   joints_only/8          0    2.65 .. 2.71 ms    2.65 .. 2.75 ms
//!   one/8                 14    88.5 .. 96.1 us     109 .. 111 us
//!   pile/8             8,400    5.82 .. 6.67 ms    7.39 .. 8.19 ms
//!   arriving/8        23,000    7.42 .. 8.60 ms    8.60 .. 10.2 ms
//! ```
//!
//! **A workload with no contacts pays nothing**, which is the check that says the cost is
//! the pass and not a regression somewhere else: `joints_only` is six hundred rigs with
//! self-collision off and no plane, the whole step is joints, `plan_velocity` is empty and
//! the guard in [`Skeleton::step`] skips the lot -- and the two ranges are the same range.
//! Where there are contacts it costs about a fifth to a quarter, and `arriving` is too
//! noisy to say more than that its two ranges touch.
//!
//! **And two more vectors on [`Correction`] cost more than the pass does** -- taking it
//! from 120 bytes to 176 was worth 30 to 40 per cent on its own, with the pass itself
//! unchanged, because every joint and every contact in the positional solve carries one
//! whether it has anything velocity-level to say or not. That is why the third correction
//! kind shares the first's fields rather than adding its own. See [`Charge`].
//!
//! # What was left is one bone, and it was a contact that existed on alternate steps
//!
//! **This is closed**; read it for the mechanism, the two fixes that were rejected for it,
//! and then the section that ends them. The section above ends on the measurement that
//! localised this: at four velocity sweeps
//! the rig is the stillest it has ever been and sleeps in none of sixteen draws, so what is
//! still moving is one or two bones rather than the rig. It is **one bone**, and it is the
//! same one in all sixteen draws.
//!
//! At four sweeps every other bone of the seventeen is frozen -- median drift 0.0000 of a
//! reach over four hundred and eighty steps -- and body 10, the free end of a limb resting
//! on another bone, sits in an exact **period-two** cycle:
//!
//! ```text
//!   step  gap 7-10 at the start   after the predict   contact?   after the solve
//!   even        +0.0319 mm            +0.0446 mm         no          -4.1551 mm
//!   odd         -4.1551 mm            -8.3159 mm        yes          +0.0319 mm
//! ```
//!
//! On the odd step the contact exists and puts the bone back exactly where it was. On the
//! even step the narrow phase finds the pair clear -- by **thirty-two microns** -- emits
//! nothing, and the bone is unconstrained for the whole step: it falls 1.36 mm and turns
//! 0.0231 rad about its joint, which is 5.7 mm of surface against the 3.7 mm
//! [`sleep::STILL_FRACTION`] allows it. It misses by half as much again, every other step,
//! for ever, while nothing else in the rig moves at all.
//!
//! **Three facts compose into that, and none of them is wrong on its own.** The positional
//! solve removes the whole overlap, so a resting pair ends the step touching to within
//! microns. The narrow phase's test is `distance >= r_a + r_b`, so a pair resting exactly
//! against another has no contact -- a fact this module has already met once and recorded,
//! which is why islands are built on the broad phase's pairs rather than on contacts. And
//! with no contact in the step, the **joints** drive the pair 4.16 mm together, because last
//! step's contact pushed them 8.35 mm apart and left the joint that much to repair; the
//! repair is charged as velocity, and the next predict doubles it into the 8.35 mm the
//! contact then removes. The cycle is closed and it is stable.
//!
//! At one sweep the same mechanism runs with more company. The rig settles into an arch --
//! pelvis and one whole leg flat on the plane, reading 0.0005 m/s and never woken; the other
//! leg part down; both arms propped on the leg and on each other -- and the bones that never
//! satisfy the settling test are the propped ones, bones 4 to 10, never a bone lying on the
//! plane. The heaviest of them, a hand on a foot, carries 10 to 12 mm of overlap at the end
//! of **every** step and never sheds it: the pair contact resolves it to zero in every pass
//! and the ground stage, which runs last in a pass and is effectively rigid, turns the foot
//! 0.096 rad back and puts all of it back. That ordering is not the defect either -- running
//! the ground first instead takes the rig from nine draws of sixteen asleep to none.
//!
//! # Two fixes for it that were built, measured, and are not here
//!
//! **Speculative contacts.** Give the pair a constraint while it is still clear, by as much
//! as one step of gravity's sag -- `|g| dt^2`, which the module already derives as
//! [`contacts::Anchor`]'s `anchor_reach` and which is exactly the distance a resting pair
//! can close in a step. It costs nothing while the gap is open, because
//! [`contacts::solve_contact_normal`] returns on `depth <= 0`, and it catches the joints'
//! push in the same step they make it. Confined to the crossed branch of the narrow phase --
//! a near-parallel pair touching at one end already has its patch, and speculation is about
//! a pair with no contact at all -- it leaves a settled stack of three **bit-identical** and
//! takes the rig from 14 draws of 32 asleep to 26, the best this defect has ever measured.
//!
//! **It brings the walk back, and that is disqualifying.** Over twenty-four draws the median
//! drift improves (0.0024 of a reach to 0.0016) and the tail explodes: six draws travel past
//! the jostling allowance where none did, the worst at 2.20 reaches with a straightness of
//! 0.67, and `a_settled_rig_stays_where_it_settled` allows **no** draw to look carried. The
//! variants are all worse. Without the velocity share below: eight of twenty-four, worst
//! straightness 0.9996, which is a rig being carried outright. With the margin at a tenth of
//! a millimetre instead: the median settled residual goes to 0.301 of `g dt`, past the
//! quarter `a_settled_rig_does_not_sit_on_a_limit_cycle` allows. Applied to both branches
//! rather than one: a settled stack of three drifts 0.354 m against 0.048 and leans 1.6
//! degrees, so a body dropped on it misses and lands beside it. And letting a speculative
//! contact skip its tangential half -- a pair that was not touching has no patch to stick to
//! -- is the worst of the lot, nine draws of twenty-four and a worst drift of 13.5 reaches at
//! a straightness of 0.997.
//!
//! The reason is the one this page gives twice already. A speculative contact's normal is
//! frozen while the pair is still apart and spends itself a pass or a step later, so it is a
//! push whose direction is a little wrong every time **in the same way**; and its tangential
//! half acting only on the passes its normal half reached is the third time this page has
//! found that intermittent friction is what the rig walks on. Whatever closes this has to
//! give the pair a constraint without giving it a stale frame -- which is a narrow phase that
//! can run inside the solve, not a margin on the one that runs before it. See the section
//! below for what that turned out to need, which is less than it sounds.
//!
//! **The velocity share taken over the loaded constraints only.** The arithmetically exact
//! reading of a simultaneous pass, and measured worse on its own. See
//! [`Skeleton::share_velocity`], which carries the numbers.
//!
//! # What closed it: a contact outlives the step it stopped being needed in
//!
//! The section above asks for a narrow phase that can run inside the solve, so that a pair
//! resting exactly against another is still known to be touching and its normal is current
//! rather than a step stale. **The cheaper half of that is enough, and it is the half with
//! no stale frame in it at all.** Re-running the narrow phase between passes is what costs:
//! it is about 1.4 ms on the heap, and eight passes of it would add ten. What the defect
//! actually needs is for the contact to *stop vanishing* -- and a contact that survives the
//! step boundary has its normal and its surface points re-derived from the predicted
//! positions at the top of the next step, which is exactly as fresh as every other
//! contact's and is not a frozen frame spent later.
//!
//! So [`contacts::capsule_contact`] takes one more argument: whether this pair was carrying
//! normal impulse last step. If it was, it keeps its contact whether or not the surfaces
//! still overlap. [`Skeleton::persist_contacts`] is the list, rebuilt from scratch every
//! step out of the contacts that spent anything, so a pair is kept for exactly one step
//! past the last one it did work in.
//!
//! **This is not a speculative contact, and the difference is the whole of why it works.**
//! A speculative contact is generated from a *distance*: every pair inside a margin gets
//! one, including pairs that are merely approaching and will never touch, and its frame is
//! fixed while the pair is still millimetres apart. There is no distance in this rule. What
//! renews a contact is that the contact *did work* -- so the set is causally derived from
//! what was actually loaded, a pair that has never touched is never given one, and a pair
//! flung apart loses its own after a single inert step. And a revived contact is provably
//! free while the gap is open: [`contacts::solve_contact_normal`] returns on `depth <= 0`
//! and every other half returns on `spent.normal <= 0`, so the only behaviour it can change
//! is a gap that closes *during* the step, which is the case it exists for.
//!
//! **One bound had to be found, and the stack of three found it.** A near-parallel pair is
//! a patch with a contact at each end, and a pair resting at a slight relative tilt has one
//! end loaded and the other clear. Reviving that clear end gives a second constraint to a
//! pair that already had one -- and a settled stack of three then shears 0.354 m sideways
//! and leans 1.6 degrees, so `a_body_dropped_on_a_sleeping_stack_lands_on_top_of_it` misses
//! the stack and lands beside it. That is the same 0.354 m the speculative variant measured
//! when it was applied to both branches of the narrow phase, arriving a second time by a
//! different route. **A revived contact is for a pair that has lost every constraint**, so
//! the patch is asked for as it stands first and revived only if that leaves it with
//! nothing. Under that rule a settled stack of three is **bit-identical** to what it was.
//!
//! ## What it buys
//!
//! The rig is the seventeen-bone one, self-collision on, eight passes, draws a relative
//! 1e-12 apart. Columns are capsules lying flat on the plane, sixteen draws, capped at
//! twelve thousand steps; piles are eight draws of the median body's surface travel over
//! four hundred and eighty steps.
//!
//! ```text
//!   rig, how many draws sleep inside 2000 steps
//!     before   9 of 16, 14 of 32, at steps 312..1312
//!     after   16 of 16, 32 of 32, at steps 112..296
//!   rig, 24 draws, drift over 32 windows and the worst straightness
//!     before   0 past 0.113, worst 0.083 at 0.12, median 0.0017
//!     after    0 past 0.113, worst 0.014 at 0.14, median 0.0034
//!   rig, settled residual as a fraction of g dt, eight draws
//!     before   0.038 .. 0.182, median 0.143
//!     after    0.037 .. 0.099, median 0.089
//!   rig, kinetic energy held by a settled rig over six thousand steps
//!     before   0.39 .. 0.68 J, sitting 199.3 J up
//!     after    0.02 .. 0.04 J, sitting 187.0 J up
//!
//!   steps to sleep    column of 3    4         5          6          7          8
//!   before                    362  629..766   224..338   203..563   745..919   303..426
//!   after                     362  468..715        199   428..458   364..480   462..693
//!
//!   settled pile of                20              40              60
//!   before                 0.000..0.090    0.073..0.124    0.077..0.218
//!   after                  0.000..0.000    0.000..0.113    0.054..0.153
//! ```
//!
//! **The period-two cycle is gone.** Traced on a settled rig with sleeping off, the nearest
//! self-contacting pair's surface gap alternated between +0.31 mm and -7.70 mm on
//! consecutive steps while the contact count swung between seven and nine; it now sits
//! between +0.21 and +0.57 mm with the count steady, and the median bone's speed falls
//! from 0.022..0.046 m/s to 0.010..0.024 and keeps falling. **And the walk did not come
//! back**, which is what disqualified the speculative version: no draw of twenty-four is
//! past the jostling allowance, the worst straightness is 0.14 where the law's line is 0.5,
//! and the worst draw's drift is six times smaller than it was.
//!
//! The rig also stops propping itself up on its own friction, which is the defect the
//! staggered sub-passes left behind: it settles twelve joules of potential lower and holds
//! a twentieth of the kinetic energy, flat over six thousand steps.
//!
//! ## What it costs
//!
//! Three alternating rounds of prebuilt binaries and then five more, because this machine's
//! noise is larger than the effect: `joints_only` is the control and its point estimate
//! moved by a factor of two on the *unchanged* binary between rounds, so what follows is
//! medians of eight rounds rather than a range.
//!
//! ```text
//!                   contacts             before     after
//!   joints_only/8          0 ->     0    2.42 ms    2.52 ms
//!   one/8                 14 ->    13     116 us      95 us
//!   pile/8             8,400 -> 7,800    10.00 ms    7.67 ms
//!   arriving/8         8,950 ->12,000    10.78 ms   11.15 ms
//! ```
//!
//! **A workload with no contacts pays exactly nothing**, and that is arithmetic rather than
//! a measurement: [`Skeleton::persist_contacts`] walks a contact list that is empty and the
//! narrow phase searches a list that is empty, so `joints_only`'s two figures are one figure
//! plus this machine's noise. Where a scene is settling the change pays for itself -- `one`
//! and `pile` are both faster, because a rig that goes to sleep and a pile that settles
//! further are less work, and `pile` carries six hundred fewer contacts than it did.
//!
//! **Where it costs is a scene in flight.** `arriving` is a heap on its way down, where
//! pairs touch and part every step, and every one of them carries an inert contact for one
//! step afterwards: a third more contacts, for four per cent on the median, with the two
//! ranges overlapping. That is the honest price of a rule with no distance in it -- the
//! grace is one step whether the pair parted by a micron or by a metre -- and it buys the
//! absence of a margin to tune.
//!
//! ## And one thing it moves that is worth recording
//!
//! [`VELOCITY_SWEEPS`] is one, and the count above it was already known to be a coefficient
//! rather than an iteration -- good at one, catastrophic at two, best at three, useless at
//! four. Revived contacts move *which* counts are which: at four sweeps, where the rig used
//! to be the stillest it had ever been at 0.004 to 0.013 m/s, it now thrashes at 2.5 m/s.
//! At the shipped count of one a settled rig's total energy is flat to a part in ten
//! thousand over six thousand steps, which is the measurement that matters; the four-sweep
//! figure is recorded because it is the second time this page has caught that count behaving
//! like a tuned number, and the next person to reach for it should know it has moved again.
//!
//! # A revived contact hands the bodies no momentum, and without that a rig flies
//!
//! The section above is right about what a revived contact is *for* and wrong about one
//! thing it may do. It closes on the claim that reviving one is provably free while the gap
//! is open, which it is; what it does not say is what happens the moment the gap closes.
//!
//! **A ragdoll takes off.** Measured on the seventeen-bone rig resting on the plane, ten
//! draws a relative 1e-12 apart, settled kinetic energy after eighteen hundred steps:
//!
//! ```text
//!   passes             8         12         16         24         32         64
//!   median        0.0146   470.1123     0.0195  1277.1570     0.0117     0.0003  J
//!   worst         0.0297   661.3257  1166.6445  2729.2404  1911.6905     0.0278  J
//!   over 1 J     0 of 10   10 of 10    3 of 10    6 of 10    2 of 10    0 of 10
//! ```
//!
//! Traced at twenty-four passes, the draws that hold a kilojoule are **airborne**: every one
//! of seventeen bones off the ground, bodies at ten to fifteen metres a second, three to
//! eleven contacts and all of them between the rig's own limbs, holding one to two
//! kilojoules for **twelve thousand steps** without decaying. A rig that never touches the
//! ground is not taking that from the ground. It is propelling itself.
//!
//! ## Where it comes from
//!
//! `Correction::free_translation` exists because a contact may turn into velocity only the
//! overlap the step itself drove, and the share it drove is measured as the slide along the
//! normal since the step began. **For a revived contact that measurement is meaningless.**
//! The pair was not overlapping when the narrow phase looked, so the bodies drove into
//! nothing; whatever overlap the constraint later finds was made by the step's own
//! corrections -- the joints pulling a resting pair back together, which is the case
//! revival exists for. The slide cannot tell the two apart, because it includes every
//! correction applied since the step began. This page already says so, one section up, about
//! a different fix: charging the whole of it as velocity hands the bodies the solver's own
//! repair work as momentum they never earned.
//!
//! A revived contact is the one constraint for which that is the *whole* of what it charges,
//! and it is charged again on every pass -- which is why the defect scales with the pass
//! count and why eight, the count everything here was measured at, is one of the two that
//! happen to be quiet.
//!
//! ## The fix, which is one line and a bit of bookkeeping
//!
//! [`contacts::Contact::revived`] says the narrow phase found this pair clear. A revived
//! contact charges velocity on the **first pass that finds it loaded** and none on any pass
//! after that. The first pass sees the closing the bodies and their joints actually made;
//! every pass after it sees only what the earlier passes failed to remove, which is the
//! solver's own work and not momentum anybody earned. That is the same shape every
//! dissipative rule on this page has had to take -- a budget for the step rather than a
//! charge per pass -- arriving for the fourth time.
//!
//! Charging a revived contact *nothing* was tried and is worse, which is the measurement
//! that says the first pass matters: a rig then stops propagating "stopped" through a
//! resting self-contact, `a_rig_that_touches_itself_comes_to_rest` loses two draws of
//! sixteen, and `a_settled_rig_stays_where_it_settled` gains a draw that is carried. That is
//! the objection this page raises against every bound on the read-back velocity, and it
//! applies here too -- but only to the passes after the first.
//!
//! ```text
//!   passes             8         12         16         24         32         64
//!   median        0.1340     0.1514     0.2290     0.3077     0.0431     0.0791  J
//!   worst         1.5218     0.8125     1.0007     2.9172     1.7477     4.3381  J
//!   over 1 J     1 of 10    0 of 10    1 of 10    2 of 10    2 of 10    1 of 10
//!
//!   draws of sixteen that go to sleep, and the steps they take
//!     before     16/16      0/16       9/16       5/16       9/16      16/16
//!     after      16/16     15/16      14/16      14/16      15/16      16/16
//!     after, median steps
//!                  131       539        395        413        385        720
//! ```
//!
//! **The kilojoules are gone at every count**, the worst draw anywhere is four joules
//! against two thousand seven hundred, and the sleeping the section above bought is kept --
//! sixteen of sixteen at eight passes at 131 steps, where it was 148 to 297, and twelve to
//! sixteen everywhere else where it was nought to sixteen. It costs a `bool` on
//! [`contacts::Contact`], 112 bytes to 120, and one test per contact per pass. Timed on ten
//! thousand two hundred bodies as prebuilt binaries run alternately, the two builds settle
//! into different scenes -- 21,100 contacts against 15,450 at the same step -- so the
//! per-step totals are not comparable and the per-contact figures are: 1.53 to 2.42 us
//! against 1.82 to 2.30, which is one range.
//!
//! ## And there is no distance bound on revival, which was measured rather than assumed
//!
//! The obvious guard on a revived contact is a margin -- revive only a pair clear by less
//! than the furthest a step can close it, which this module already derives as
//! [`contacts::Anchor`]'s `anchor_reach`. It was built, and it is **worse than either
//! extreme**: see [`contacts::capsule_contact`] for the table. A bound is not a weaker
//! revival, it is an **intermittent** one -- the constraint appears and disappears as the
//! gap crosses the margin -- and this page has now recorded four separate occasions on
//! which an intermittent constraint is what a rig walks on. Reviving always, or never, is
//! stable; reviving sometimes is not.
//!
//! ## What was watching for this, and what is now
//!
//! Nothing was. Every law in `tests/articulated_laws.rs` ran the rig at eight passes, so a
//! defect that is quiet at eight and at sixty-four and loud at everything between was
//! invisible. `asking_for_more_passes_does_not_make_a_settled_rig_worse` is the law that
//! closes that gap: it holds the rig to the same quarter of `g dt` that
//! `a_settled_rig_does_not_sit_on_a_limit_cycle` holds it to, at eight, twelve, sixteen,
//! twenty-four and thirty-two passes. Run against the defect it reports 12.3, 22.6 and 17.2
//! steps of gravity at twelve, twenty-four and thirty-two against 0.091 at eight.
//!
//! It asserts one bound at every count rather than a trend, and that is deliberate: a
//! settling rig is chaotic, so two counts differ by which configuration the rig lands in as
//! well as by how well it was solved, and a monotone assertion would fail on noise while a
//! real doubling slipped past. **The property is that the bound does not depend on the
//! dial.**
//!
//! `a_settled_rig_stays_where_it_settled` needed its straightness predicate fixed to survive
//! this, and the fix makes it stricter. A bone's straightness is its net travel over the path
//! it walked, guarded by `path > 0.0` -- which only catches a bone that is still to the last
//! bit. Once rigs started freezing outright, four draws of eight came back with a drift of
//! 0.0000 of a reach and a straightness of **1.0000**, which is a rig sitting perfectly
//! still failing a law about rigs that travel. The guard is now the law's own
//! `STILL_FRACTION`: a bone that has not covered one settling window's allowance across all
//! thirty-two of them has not gone anywhere, and asking it which direction is dividing
//! arithmetic noise by itself.
//!
//! # Every hinge in the module was inside out, and it was the pile
//!
//! The section above and the six before it are a hunt for why a jointed rig will not go to
//! sleep, conducted entirely on **one** rig in **one** pose: the seventeen-bone rig dropped
//! upright from a metre. Run the same rig at a set of landing poses instead -- twenty rigs
//! far enough apart that none touches another, so each is an independent sample, eight
//! draws a relative 1e-12 apart, a hundred and sixty samples in all -- and it comes to rest
//! in **89 of 160**. The pose the rest of this page was measured at is one of the good ones.
//!
//! The ensemble is worth stating exactly, because it is the instrument the rest of this
//! section is read off: twenty copies of the rig `tests/articulated_laws.rs` builds, on a
//! 2.5 m grid so that no two can reach each other, each turned by its own axis and angle
//! before it is dropped from 0.55 m, and a rig counted as settled on the step every one of
//! its seventeen bodies is asleep. Eight draws, differing only in a relative 1e-12 on the
//! drop height. One skeleton holds all twenty, which costs nothing here: separate rigs
//! never touch, so they are separate islands and each sleeps on its own.
//!
//! That is what makes a pile look like a different problem from a rig. It is not one. A
//! pile sleeps as an island, so it sleeps when its *last* body does; a heap of twenty rigs
//! sleeps only if all twenty do, and at 56 per cent each that never happens. Measured, and
//! this is the honest baseline the fix below is against -- eight draws each, capped at
//! sixty seconds:
//!
//! ```text
//!                                  bodies   draws that came to rest
//!   one rig, upright from a metre      17   8 of 8, at 133..867 steps
//!   one rig, tumbling                  17   4 of 8
//!   four rigs in a heap                68   0 of 8
//!   twenty rigs in a heap             340   0 of 8
//!   twenty rigs in a shallow field    340   0 of 8
//!   forty rigs in a deep heap         680   0 of 8
//! ```
//!
//! **Neither coefficient is what is wrong, and that was measured first because it would
//! have been the cheapest answer.** A hundred and sixty samples a point, as the fraction of
//! rigs that come to rest inside sixty seconds:
//!
//! ```text
//!   friction        0.0    0.25    0.5    0.8    1.2    2.0
//!                    5%     56%    56%    65%    76%    71%
//!   rolling         0.0     0.25   0.5    1.0    2.0
//!                   48%      56%   68%    43%    21%
//! ```
//!
//! Friction buys something and buys it slowly -- at 1.2 the median rig takes 1400 steps to
//! settle against 900 at 0.5 -- and rolling resistance is worse on both sides of where it
//! is. Nothing here is a pile settling.
//!
//! ## What it was
//!
//! Bisect to the smallest thing that fails and it is one limb. On a rig that will not
//! sleep, two bones of one arm -- the first and last of its three segments, two joints
//! apart and so not excluded as a jointed pair -- sit **33 to 55 mm inside each other** with
//! their radii summing to 120 mm, in an exact period-two cycle, each carrying 700 to 900 N
//! where its own weight is 39 N, while the bone between them swings at 0.9 m/s for as long
//! as it is watched. Pin every other body in the rig, take the ground and gravity away, and
//! solve that one step at rising pass counts, and the overlap does not come out:
//!
//! ```text
//!   passes         1       2       4       8      16      32      64     128     256
//!   overlap     52.1    34.8    35.3    39.2    44.1    46.1    46.8    44.6    41.9  mm
//!   load         0.7     122     336     817    1840    3534    6633   12218   20661  N
//! ```
//!
//! **The impulse grows linearly with the pass count and the overlap does not move.** That is
//! not a loop converging slowly; it is a pass applying a correction that the next constraint
//! takes straight back out, for ever.
//!
//! **And the constraint that takes it back out is not the contact's partner in the loop but
//! the hinge's own second half.** A hinge holds two things -- that the two
//! bodies' axes are one axis, and that the swing about it stays inside `min`..`max` -- and
//! the alignment half turned the axes onto one another about `cross(world_b, world_a)`,
//! which is the wrong way round: that correction drives them **apart**. The cross product
//! vanishes at a half turn as well as at none, so both are fixed points, and with the sign
//! reversed the stable one is the wrong one. Measured on the rig, every one of its eight
//! hinges started with its axes parallel and was **inverted within five steps** of the drop,
//! and stayed inverted for the rest of the run:
//!
//! ```text
//!   step                0     1     2     5    20   100  1500
//!   cos(axis_a, axis_b) +1    +1    +1  -0.78 -1.00 -1.00 -1.00
//! ```
//!
//! An inverted hinge still holds a single axis *line*, which is most of what a hinge looks
//! like from outside, and that is why nothing caught it: `a_hinge_stays_inside_its_range`
//! measures the swing with [`hinge_angle`], which reads the angle from a reference carried
//! in each body's own frame -- so on an inverted joint it asks the same wrong question the
//! solver does and gets a consistent answer. What is actually lost is the *zero and the
//! sign* of the range. `min: -0.1, max: 2.2` is then enforced on a number that is not the
//! angle anybody asked about, and the limit drives the elbow to the wrong end and holds it
//! there against a contact that cannot win.
//!
//! The angle was wrong as well, in the way that made the half turn a slow place to leave
//! rather than one a hinge passes through: `asin` of the cross product's length cannot tell
//! an angle from its supplement, so past a quarter turn the correction *shrinks* as the
//! error grows. The sign of `a . b` says which side of the quarter turn the pair is on,
//! which is the whole of what `asin` cannot see -- so the repair is one compare, and it is
//! not the dearer `atan2` because this runs on every hinge on every pass. A hinge handed to
//! the solver exactly opposed -- no cross product to turn about -- is turned back about any
//! perpendicular, because every one of them leaves the same hinge.
//!
//! ## What it buys, and what it does not
//!
//! The same hundred and sixty samples, and the same piles:
//!
//! ```text
//!   rigs that come to rest inside 60 s     before  89/160 (56%)   after  144/160 (90%)
//!   median steps to rest, of those          before     889        after      837
//!   the deepest overlap inside a rig        before   33..55 mm    after   0.2..2.6 mm
//!   one step at 256 passes, load            before   20,661 N     after   converges at 228 N
//! ```
//!
//! **What moves is which rigs stop, not how long a rig that was going to stop takes.** The
//! median is the same within its own noise, and that is the right shape for the defect: a
//! jammed limb does not settle slowly, it never settles at all, so removing the jam moves a
//! draw from "never" to "somewhere in the distribution" rather than shortening the ones that
//! already worked.
//!
//! **And the size of that statistic's own noise is worth having**, because everything above
//! is a count over a chaotic fixture. The angle was first written as `atan2(|a x b|, a . b)`
//! and then as the arithmetically identical `asin` with the supplement taken when the dot
//! product is negative -- the same algorithm, differing in the last bits -- and the two give
//! 136 and 144 of 160. Read a difference of five points as nothing.
//!
//! **The pass count stops being a lever, which is the clearest thing the fix does to the
//! rest of this page.** Before it, raising the count was the only thing that moved the
//! settling rate at all, and it moved it a long way -- that was passes partly overcoming a
//! jam rather than passes converging. After it, they buy speed and not certainty:
//!
//! ```text
//!   passes                    8      16      32      64
//!   rigs that come to rest   90%     93%     90%     93%     (before the fix: 56 76 90 93)
//!   median steps, of those   837     484     365     304
//! ```
//!
//! A fixture that settles twice as fast at sixteen passes and no more often is a fixture
//! whose failures are not a convergence shortfall. Whatever holds the last tenth awake is
//! something eight passes already reach.
//!
//! The coefficients do not come back either. Re-run on the fixed solver, friction at 1.2
//! gives 68 of 80 against the same build's 68 of 80 at 0.5, and rolling resistance at 0.5
//! gives 65 -- so the gain friction showed before the fix was it papering over the jam, and
//! there is nothing left for it to buy.
//!
//! **And the piles still do not sleep**, which is the honest end of this section. A tenth
//! of landing poses still hold a rig awake, an island sleeps when its last body
//! does, and twenty rigs is twenty chances to fail. What is left looks nothing like what was
//! removed -- millimetres of overlap rather than centimetres, a solve that converges rather
//! than one that never does -- and it is the propped-rig residual the sections above are
//! about, met again at the poses those sections never ran. Measured on one such rig after
//! the fix: bones with 0.2 to 2.6 mm of overlap, moving at 0.03 to 0.15 m/s, where the
//! settling window allows about 0.01, and three hundred of that scene's three hundred and
//! forty bodies never move at all.
//!
//! **So the next person to go after a pile should go after one rig at the poses that fail,
//! and should measure the fraction rather than the time.** The two statistics say different
//! things -- the time is what a rig that settles takes and it is chaotic over a factor of
//! five, the fraction is what fails outright and a pile is the product of it over every rig
//! it holds. Eight draws of a heap were 0 of 8 before this and are 0 of 8 after, which is
//! the same number and a different scene behind it.
//!
//! **What it costs is nothing anybody can measure, and the arithmetic says so first.** The
//! correction is the same `asin` it always was, plus a dot product and a compare, on a
//! branch that already computed a cross product, a normalisation and two
//! `Quaternion::from_axis_angle` calls -- five flops against a few hundred, per hinge per
//! pass. `atan2` would not have been free at that rate, which is why the supplement is
//! taken by hand. Three alternating rounds of prebuilt binaries, at eight passes, medians:
//!
//! ```text
//!                    before                      after
//!   joints_only/8    4.57  4.66  3.17 ms      5.08  4.57  4.83 ms
//!   one/8            114.6  87.6  88.6 us     86.4  88.3  96.7 us
//!   pile/8           9.87  8.46  5.09 ms      8.29  7.96  7.91 ms
//! ```
//!
//! `joints_only` is the control the claim rests on -- it is nine thousand six hundred
//! joints, half of them hinges, and no contacts at all, so it is the fixture where this
//! branch is the largest fraction of the step. **Its own spread on one unchanged binary is
//! 3.17 to 4.66 ms**, a factor of one and a half, against six per cent between the two
//! medians. The other two fixtures are not comparable at all in the strict sense, because
//! the fix changes how their rigs settle and so which scene is being timed.
//!
//! **One thing it changes that a caller can meet.** A hinge that is inverted now comes
//! back, and a rigid constraint undoing a half turn does it in about a step: measured on a
//! knee handed to the solver exactly inside out, the shin peaks at 120 rad/s against the
//! 188 that delivering the whole error once would be, and is at rest again within ten
//! seconds. That is the same thing this module already does to a body spawned a metre from
//! its anchor, and the bound worth guarding is not the size of the snap but that there is
//! one of it rather than one per pass --
//! `a_hinge_inverted_exactly_turns_itself_back` holds it to `PI / dt`. Before, such a hinge
//! stayed inverted for ever and nothing was reported at all.
//!
//! `a_hinge_does_not_turn_itself_inside_out` is the law that would have caught it, and it is
//! stated where the geometry puts it rather than next to the measurement: the correction has
//! two fixed points, no angle at all and a half turn, and only one of them is a hinge, so the
//! watershed between their basins is a quarter turn and a working hinge never reaches one.
//! Against the defect it reports a cosine of -1.0000.
//!
//! # What a body is carrying, and why it is `driven` rather than `normal`
//!
//! [`Skeleton::normal_load`] reports, per body, the normal impulse the last step actually
//! handed it, as a mean force in newtons. It is the quantity a caller needs to decide that
//! something has been crushed -- and the crate has no opinion on how much is too much,
//! because what load breaks a body is a material judgement that varies by what the bodies
//! represent. There is no threshold in here and there may not be one.
//!
//! **The quantity already existed, and only one of the two totals is it.**
//! [`contacts::Spent`] separates `normal`, the whole normal impulse, from `driven`, the
//! part the step itself drove and so the only part charged to the bodies as momentum. A
//! body recovering from a careless spawn is separated by an enormous `normal` while
//! nothing whatever presses on it; `driven` reads zero through the same separation. Using
//! `normal` would make a badly placed body the most crushed thing in the scene, which
//! `an_overlapping_spawn_is_not_a_crushed_body` is the guard against.
//!
//! **It is two divisions by the step, and that is not cosmetic.** The solver's impulses
//! are in the convention `correction = impulse * inv_mass`, so a raw one is kilogram
//! metres -- not the newton seconds the word suggests. One division gives the momentum,
//! two give the mean force over the step. Reporting the raw total would make the same
//! physical squeeze read four times smaller at half the timestep, and silently change the
//! meaning of whatever rule a caller had written against it. The calibration is that a
//! body lying on the plane reads its own weight, exactly: it sags `g dt^2`, the plane
//! drives that back out, and `m g dt^2 / dt^2` is `m g`.
//!
//! **It costs nothing in the inner loop.** Every contact and every ground patch already
//! carries its `Spent` across the passes, because Coulomb's cone needs the step's totals
//! and the velocity pass needs to know what it is allowed to take back. So the load is
//! already computed when the step ends, and [`Skeleton::gather_normal_load`] is one pass
//! over two lists that are still in cache rather than a write inside eight passes over
//! them. It runs *after* the velocity pass, so what it reports is the net the step handed
//! the body rather than the gross the positional passes applied before some of it was
//! taken back.
//!
//! **And it measures as free.** Three alternating rounds of prebuilt binaries, medians, at
//! eight iterations:
//!
//! ```text
//!                   contacts     before     after
//!   joints_only/8          0    3.78 ms   3.73 ms
//!   one/8                 13     102 us     98 us
//!   pile/8             7,800    8.77 ms   8.28 ms
//!   arriving/8        12,000    16.0 ms   16.2 ms
//! ```
//!
//! Every figure is inside the run-to-run spread and the point estimates move in both
//! directions, which is what no cost looks like on a machine whose noise is larger than the
//! effect. `joints_only` is the check that says so by arithmetic rather than by
//! measurement: it has no contacts of any kind, so the gather walks two empty lists and
//! cannot cost anything. Five further rounds were taken and are not quoted -- another
//! process was benchmarking on the same machine throughout them, and `arriving` reported
//! 269 ms on one of them against its usual sixteen.
//!
//! One consequence is worth stating where a caller will meet it: **a sleeping body reads
//! zero**. A step that does not solve a body drives nothing into it, so what this reports
//! is load arriving. A caller whose rule has to see a static load has to keep those bodies
//! awake.
//!
//! # Retirement, which is destruction as subtraction
//!
//! [`Skeleton::retire`] takes a body and its joints out of the solve for good. Structural
//! failure under load is general physics and belongs here; what load counts as failure is
//! a material judgement that does not, so the crate reports and the caller decides.
//!
//! **It makes the simulation cheaper rather than dearer.** Nothing is emitted, nothing
//! flies off, and no body is created: a field driven through leaves fewer bodies behind it
//! than in front. That is the opposite of the usual shape of destruction and it is most of
//! why this is worth having as a primitive -- and it is the general primitive, since
//! taking a body and its joints out is what cutting a skeleton apart needs too.
//!
//! Four things it has to be, and each is a decision rather than an implementation detail;
//! [`Skeleton::retire`] carries the argument for each. **Its joints go with it**, so
//! retiring a hip lets the leg come away. **Body indices stay stable**, because callers
//! hold them and the ground anchors are indexed by body -- the arrays are never compacted,
//! and joint indices do shift. **It costs nothing afterwards**, and not by being tested
//! for: a body with no radius is invisible to the broad phase and to the plane, and a body
//! with no mass can never be woken, which are the two rules that already keep an anchor
//! body and a pinned body out of the work. And **its neighbours wake**, because taking a
//! support away changes the situation for whatever was leaning on it.
//!
//! Removing joints is the one thing the incremental colouring cannot absorb, which
//! [`Skeleton::add_joint`] anticipated: a full recolour is wanted "if a caller removes
//! joints, or never". Retirement is rare, so it pays for one -- and the replay is the same
//! greedy rule over the joints that remain in the order they arrived, so what it leaves
//! behind is the state adding those joints would have produced, which is what keeps the
//! incremental path *correct* across a removal rather than merely unbroken.
//!
//! **What it is worth, measured.** `benches`'s `ploughing` fixture is a settled lane of
//! eight hundred capsules with a heavy roller driven down it, retiring what it crushes.
//! Medians of four runs:
//!
//! ```text
//!   steps driven        30      330      630      930
//!   bodies still live  784      585      387      189
//!   a step            3.24 ms  2.12 ms  1.71 ms  0.89 ms
//!   per live body     4.13 us  3.62 us  4.42 us  4.70 us
//! ```
//!
//! The cost falls by a factor of 3.6 while the drive runs, and the bottom row says why:
//! the cost per live body is flat, so a step pays for the bodies that are left. Every
//! other fixture in `benches` measures a scene that costs what it costs; this is the one
//! number that goes *down* as its fixture runs.
//!
//! # What is left of the pile, which is rocking rather than sliding
//!
//! After the hinge fix, nine rigs in ten come to rest on their own and a heap of twenty
//! still does not -- because a heap sleeps when its last body does, and one rig in ten
//! never stopping is twenty chances to fail. This is what the ones that do not stop are
//! doing, measured on twenty seventeen-bone rigs dropped together and left for sixty
//! seconds.
//!
//! **They are rocking in place, not sliding.** The settling test measures translation plus
//! the distance the surface sweeps as the body turns; over one settling window, of three
//! hundred and forty bodies:
//!
//! ```text
//!   over the limit on both            23
//!   over on the turn only             68
//!   over on the move only             15
//!   under the limit on neither       234
//! ```
//!
//! The turn is sixty-two per cent of the measured motion at the median and the sole cause
//! of sixty-eight of the hundred and six failures, against fifteen for translation. Two
//! hundred and thirty-four bodies in three hundred and forty pass any *given* window; what
//! they cannot do is string sixteen consecutive steps together, because the rocking crosses
//! the bound intermittently and resets the count.
//!
//! **And the residual is positional, not kinetic.** That is the part that says where the
//! fix is not. Three separate interventions on the velocity of a body the settling test
//! already calls still -- the whole velocity zeroed, halved, and the angular part alone
//! zeroed -- were built and measured against a baseline of 334 of 340 awake:
//!
//! ```text
//!   baseline                334 awake, 112 ready
//!   velocity zeroed         336 awake,  99 ready
//!   velocity halved         332 awake, 116 ready
//!   angular part zeroed     328 awake,  97 ready
//! ```
//!
//! All four are the same number. A body is not rocking because it carries angular
//! momentum from the step before; it is rocking because the solve puts it back each step.
//! Anything that damps, thresholds or resists *velocity* is treating the wrong quantity --
//! including the tempting one, a deadband under which motion is ignored, which is why it
//! is recorded here rather than tried again.
//!
//! # Where the pile's contacts come from, and the constraint that is missing
//!
//! Two fifths of a heap's contacts are a rig against **itself**. Measured on heaps of one,
//! four and twenty of the seventeen-bone rig, run to sixty seconds:
//!
//! ```text
//!   rigs   self-collision   asleep at   contacts   of them within one rig
//!      1        on             254          2               2
//!      1        off             87          0               0
//!      4        on            never        34              26
//!      4        off           1532          1               0
//!     20        on            never       308             126
//!     20        off           never       111               0
//! ```
//!
//! A heap of four never comes to rest with self-collision and comes to rest in
//! twenty-five seconds without it; one rig settles three times sooner. At twenty it is not
//! sufficient on its own -- that heap does not settle either way -- but it removes
//! sixty-four per cent of the contacts, and by the section below, most of those were
//! single-point contacts free to rotate. Self-collision is the largest single lever
//! anything has found on this problem.
//!
//! **And the reason it is load-bearing is that a constraint is missing.** [`Joint::Hinge`]
//! carries `min` and `max`; [`Joint::Ball`] carries no range at all. An elbow cannot fold
//! backwards, but a shoulder can rotate an arm straight through the chest it is attached
//! to, and the only thing that stops it is a contact. A shoulder has about a hundred and
//! twenty degrees of cone and a hip rather less; with that written down, a limb cannot
//! reach the body it hangs off, and the contact it is standing in for is not needed.
//!
//! That is the shape of the fix rather than turning self-collision off, which was ruled
//! against for a good reason -- a rig whose limbs pass through each other is wrong in a way
//! anybody can see. A cone on the ball joints keeps the limbs out of the torso *and* takes
//! the contacts away, where switching self-collision off only does the second. It is also
//! the cheaper constraint: a cone limit is one dot product and a `share_turn`, against a
//! narrow-phase test and a solved contact for every limb pair in every rig, every step.
//!
//! ## What the cone was worth, which is not what it was built for
//!
//! It was built, and it is [`Joint::socket`]. Measured against the same rig with
//! [`Joint::free_ball`] everywhere, with anatomical cones -- a third of a radian at each
//! spine link, one and a half at a shoulder, one and a fifth at a hip:
//!
//! ```text
//!   rigs   cones   self-collision   asleep at   contacts
//!      1     no          on             254          2
//!      1    yes          on            1987          5
//!      4     no          on           never         34
//!      4    yes          on           never         27
//!     20     no          on           never        308
//!     20    yes          on           never        234
//!     20     no         off           never        111
//!     20    yes         off           never        159
//! ```
//!
//! **A quarter of the pile's contacts gone, and no pile settles that did not settle
//! before.** Worse, a lone rig takes eight times as long: 254 steps against 1,987. The
//! reason is the one this page keeps arriving at from different directions -- a limb
//! resting *against* its cone is an active constraint being re-enforced every step, and a
//! joint that is being corrected is a joint that is not still. It is the propped-rig
//! residual reached by a new road, and a tighter cone makes it worse rather than better,
//! which is the opposite of what a settling fix does.
//!
//! **A limit that dissipates was the obvious next idea and it was built.** Charging the
//! cone's correction as [`Charge::Free`] turns the limb back inside its range and turns the
//! orientation it came from with it, so the step reads back no spin from the move -- which
//! is what a ligament does at the end of its travel, where a spring gives the energy back.
//! Measured against the ordinary cone:
//!
//! ```text
//!   rigs   self-collision   no cone   cone   absorbing cone
//!      1        on            254     1987      never
//!      1        off            87      194        105
//!      4        off          1532     1384      never
//!     20        off       111 contacts  159         95
//! ```
//!
//! It is not better, it is differently bad, and the reason is the one this page has now
//! arrived at five times: **the settling test asks how far a body moved, and a free
//! correction moves it exactly as far.** It only stops the move counting as speed. Every
//! intervention on the velocity channel -- zeroing it, halving it, zeroing the angular part
//! alone, and now absorbing a joint limit -- lands in the same place, because the residual
//! is positional.
//!
//! ## And a cone does not replace self-collision, which was the hope
//!
//! The whole reason to want cones was that two fifths of a heap's contacts are a rig
//! against itself. If a limb cannot reach the body it hangs off, the contact is not needed.
//! Measured on four rigs, the worst any rig's own bones ever overlap each other, for pairs
//! no joint already holds apart, against a bone radius of 0.060 m:
//!
//! ```text
//!   cones   self-collision   worst ever   where it finishes
//!     no         no            0.1600 m       0.1591 m
//!    yes         no            0.1564         0.1245
//!     no        yes            0.1747         0.0019
//!    yes        yes            0.1554         0.0021
//! ```
//!
//! **No.** Without self-collision the bones finish a tenth of a metre inside each other --
//! more than two radii, which is one limb wholly inside another -- and the cone takes that
//! from 0.159 to 0.125, which is not a fix. With self-collision the figure is 0.002 m,
//! which is bodies touching.
//!
//! The reason is structural rather than a matter of choosing better cones. **A cone is
//! joint-local and interpenetration is global.** A cone holds a limb against its own
//! parent; a folded rig brings *non-adjacent* bones together, a hand into a thigh, a shin
//! across the opposite shin, and no per-joint limit can see those pairs at all.
//!
//! So self-collision stays, the contacts stay, and the thing to fix is what those contacts
//! *are* -- which is the section below, and the reason for it is now measured rather than
//! argued: if a pile must carry three hundred contacts, they should be contacts that
//! resist the motion the pile is failing to stop.
//!
//! **Detached pieces are unaffected**, which is what makes this safe for a caller that cuts
//! bodies apart. Self-collision is rejected per *jointed component*, and a piece that has
//! been severed is no longer in the component it came from -- see
//! [`Skeleton::set_self_collision`] and the broad phase's `jointed.component` test. A cut
//! arm collides with the corpse it came off; the arm still attached does not need to.
//!
//! # And the rocking is the shape, which this module already says somewhere else
//!
//! [`contacts::capsule_contact`] carries the diagnosis in its own doc comment: *"two
//! cylinders lying against each other touch along a line, and a single point taken from
//! the middle of it leaves them free to rotate about that point: a pile of parallel limbs
//! then rocks forever instead of resting, however many solver iterations it is given."*
//! That is exactly the measurement above, and the answer already built for it is the
//! two-point patch a near-parallel pair gets.
//!
//! **It reaches one per cent of a pile.** Of the three hundred and five touching pairs in
//! the heap above, three got a patch and three hundred and two got a single point, because
//! the patch is only offered to pairs within [`contacts::PARALLEL_SINE`] of parallel -- a
//! little under three degrees -- and a heap is not parallel:
//!
//! ```text
//!   sine of the angle between the axes   p10 0.160   p25 0.407   median 0.774
//!   the patch is offered below                                          0.05
//! ```
//!
//! Ninety per cent of touching pairs are more than nine degrees from parallel and the
//! median pair is near fifty. So ninety-nine per cent of a pile's contacts are a single
//! point with no moment arm, and the bodies are free to turn about them. **The solve is not
//! failing to converge; it is converging on a constraint that does not resist the motion
//! being measured.**
//!
//! Which makes it a question about the shape rather than about the solver. A capsule is a
//! swept sphere: its surface is curved everywhere, so it touches anything at a point or a
//! line and there is no first-order resistance to rolling about that contact. A body with
//! flats touches on a *face*, the contact is a polygon rather than a point, and tipping it
//! lifts one edge while pressing another -- which is a restoring torque out of geometry
//! rather than out of a coefficient. It is why a box on a table does not rock and a can
//! does.
//!
//! Two ways out, and the cheap one has not been tried:
//!
//! * **Give every contact the patch, not only the parallel ones.** Two spheres of radii
//!   `ra` and `rb` overlapping by `d` meet in a circle of radius about `sqrt(2 d r)` with
//!   `r` the harmonic mean -- pure geometry, no material constant, nothing to tune, and it
//!   grows with load because a heavier contact is a deeper one. That is a derived moment
//!   arm for every contact in the set, where today there is one for one in a hundred.
//! * **Give the bodies flats.** An octagonal prism is the shape that suits this: its face
//!   normals are eight fixed directions in body space plus two caps, so there is no support
//!   search; its inertia is closed form, so there is no polyhedral integration; and a fixed
//!   sixteen vertices means separating-axis rather than GJK, which terminates by
//!   construction rather than on a tolerance -- and a solver whose behaviour is pinned by a
//!   bit-identity law should prefer the one that does not iterate. It costs perhaps twenty
//!   to forty times a capsule pair in the narrow phase, against a settled heap costing
//!   nothing at all, so the trade is worth measuring rather than assuming. The new failure
//!   mode to watch for is that flats have metastable states of their own: eight faces is
//!   eight resting orientations to flip between.
//!
//! # The GPU question, which double precision answers
//!
//! A graph-coloured constraint solve is a GPU-shaped workload: a colour is thousands of
//! independent constraints and nothing in it crosses. That is what PhysX and FleX do, and
//! it is the obvious thing to reach for when a step costs milliseconds. An earlier attempt
//! here was withdrawn on the grounds that its entire margin was the host's fork overhead --
//! a verdict since made doubtful, because the host side it was compared against was
//! measured on an oversubscribed pool and with the benchmark ordering defect
//! `benches/articulated.rs` now documents.
//!
//! **So it was measured again, and the thing that decides it is not the solver at all. It
//! is that this module is `f64`.**
//!
//! Consumer NVIDIA silicon runs double precision at a fraction of its single-precision
//! rate -- one sixty-fourth on Ampere. Measured on an RTX 3090 against an i9-10980XE,
//! through CubeCL's CUDA runtime, a dependent FMA chain with eight independent accumulators
//! per thread so that both sides are throughput-bound rather than latency-bound:
//!
//! ```text
//!   gpu f32    18.6 .. 19.8 TFLOP/s
//!   gpu f64     0.536 .. 0.538      97 per cent of the card's f64 peak
//!   cpu f64     0.244 .. 0.267      with `-C target-cpu=native`
//! ```
//!
//! **Thirty-five times between the GPU's own two precisions, and two times between the GPU
//! and the CPU in the one this crate uses.** That two is a ceiling, not an estimate: it is
//! pure arithmetic with no kernel launches, no transfers, no divergence and no irregular
//! access, and a coloured solve has all four. Twenty-six stages times eight passes is two
//! hundred launches a step before any work happens.
//!
//! Two things about that measurement are worth keeping, because both were nearly reported
//! wrong. The CPU figure moved by a factor of forty when `-C target-cpu=native` was added:
//! without it `f64::mul_add` compiles to a libm call, a correctly-rounded software FMA, and
//! the CPU appeared to lose by eighty times rather than two. And a single accumulator per
//! thread measures FMA *latency* rather than throughput, which flatters whichever side has
//! more threads to hide it with -- always the GPU. A comparison that gets either wrong
//! produces a confident number pointing the wrong way.
//!
//! # So the precision is the fork, not the vendor
//!
//! Dropping to `f32` recovers a factor of thirty-five and costs
//! `the_same_simulation_twice_is_bit_identical`, which is not a trade this module can make:
//! a caller running lockstep needs the same bits on every machine, and GPU float behaviour
//! varies by driver and by vendor. A caller who does *not* need that -- one animating
//! bodies nobody's simulation state depends on -- is a different caller, and for them the
//! thirty-five is the whole story.
//!
//! That is the shape a GPU path here would have to take: not this solver moved to the
//! device, but a separate single-precision one for bodies whose positions nothing reads
//! back. `src/particles/particle_backend.rs` already carries the machinery for deciding
//! when such a path pays -- a measured cost model with a `crossover` that returns `None`
//! when the answer is never -- and it is the right instrument to point at this before any
//! kernel is written.
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
    capsule_contact, ground_contacts, solve_contact_friction, solve_contact_normal,
    solve_contact_velocity_friction, solve_contact_velocity_normal, solve_ground_friction,
    solve_ground_normal, solve_ground_velocity_normal, Contact, GroundContact, Spent,
};
use scatter::Bodies;
use sleep::{settling_steps, BitSet, Components, Islands, NO_ISLAND, STILL_FRACTION};

/// **How many times the velocity pass sweeps its constraints before the step ends.**
///
/// The pass is a simultaneous solve -- every constraint answering against one frozen
/// velocity state, a body taking the mean of what named it -- which is one Jacobi sweep of
/// a velocity-level complementarity problem, and one sweep propagates a correction exactly
/// one contact deep. In a heap a body's velocity depends on its neighbours' through the
/// contacts it shares, so the obvious question is whether a second sweep carries it
/// further.
///
/// **It is one, and the answer is not that one is enough -- it is that the count is a
/// tuned number and this module does not take those.** Measured on the seventeen-bone rig
/// with self-collision on, sixteen draws at eight positional passes, capped at twelve
/// thousand steps:
///
/// ```text
///   sweeps                    1        2        3        4
///   draws that sleep      9/16     1/16    13/16     0/16
///   median bone's residual, as a fraction of g dt, six draws:
///                    .062..182  .10..2.1  .19..213  .024..064
///   total energy of a settled rig     stable  UNSTABLE   stable   stable
/// ```
///
/// Three sweeps is the best figure anything in this module has produced and two is the
/// worst -- a rig at two sweeps gains energy, peaking at 1024 J against the 215 it settles
/// from, and one draw of sixteen sleeps. A count that is good at one, catastrophic at two,
/// best at three and useless at four is not converging on anything; it is a coefficient
/// that happens to suit a scene, which is exactly the shape the module header rejects
/// under-relaxation for. The velocity pass is a Jacobi sweep of a velocity-level problem
/// and the obvious reading is that more sweeps solve it better, and the measurement says
/// it is not that kind of iteration at all: each sweep disturbs the *joints'* velocity
/// constraints, which only the next step's positional solve repairs, and that repair is
/// charged as velocity. So sweeping harder feeds the loop rather than closing it.
///
/// **And four sweeps names what is left.** It is the stillest the rig has ever been -- the
/// median bone at 0.024 to 0.064 of `g dt` against 0.062 to 0.182 at one sweep, and a
/// median drift of 0.0000 of a reach over four hundred and eighty steps -- and it never
/// sleeps, in none of sixteen draws over twelve thousand steps. A drift criterion that a
/// frozen median bone does not satisfy means the bodies still moving are one or two of
/// seventeen rather than the rig. Whatever is left of this defect is local to a bone, not
/// a cycle in the whole loop, and that is a different thing to go after than the one the
/// velocity pass was built for.
///
/// Whatever the count, the sweeps **share one step's budget**: each carries the running
/// totals in [`contacts::Spent`] forward the way the positional passes do. A sweep that
/// re-read the positional figures would be handing back N times what a contact ever gave
/// and spending Coulomb's coefficient N times, which is the same defect as giving the pass
/// a fresh budget, reached by a different route. That accounting is what makes the row
/// above a measurement of iteration rather than of a coefficient being multiplied, and at
/// one sweep it is provably free: the answer is bit-identical with and without it.
///
/// **And the row has moved since**, which is the strongest thing anybody has said about it
/// being a coefficient. Once the narrow phase started keeping a loaded pair's contact alive
/// -- see the module header -- four sweeps went from the stillest the rig had ever been, at
/// 0.004 to 0.013 m/s, to thrashing at 2.5 m/s. At one sweep a settled rig's total energy is
/// flat to a part in ten thousand over six thousand steps. So the row above is a record of
/// what that count did on one commit rather than a property of the pass, and anybody raising
/// it has to re-measure the whole row.
const VELOCITY_SWEEPS: usize = 1;

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
    /// **How many flats the body has around its own axis.** Zero is a capsule; eight is an
    /// octagonal prism of circumradius `radius`, its faces running down the same +Y.
    ///
    /// A capsule is a swept sphere, so it is curved everywhere and touches anything at a
    /// point or a line -- and a point contact offers no first-order resistance to rolling
    /// about itself, which is why a pile of them rocks and never comes to rest. Measured on
    /// a heap of twenty rigs: three hundred and two of three hundred and five touching
    /// pairs get a single point. A body with flats resting on a face touches over a
    /// *polygon*, and tipping it lifts one edge while pressing another, which is a
    /// restoring torque out of the geometry rather than out of a coefficient. It is why a
    /// box on a table does not rock and a can does.
    ///
    /// **Fewer than three is a capsule**, since two flats do not enclose anything. The
    /// module's own header carries the argument for eight in particular: enough that the
    /// silhouette reads round at arm's length, few enough that two of them can be tested
    /// by separating axes rather than by an iterative search -- which matters for a solver
    /// whose behaviour is pinned by a bit-identity law, because a fixed set of axes
    /// terminates by construction where GJK terminates on a tolerance.
    pub facets: u32,
}

impl Body {
    /// **A regular `facets`-sided prism** at rest at `position`, of circumradius `radius`
    /// and segment `length`, its long axis along local **+Y** like everything else here.
    ///
    /// `facets` below three is a capsule, because two flats enclose nothing;
    /// [`Body::capsule`] is the direct way to say that.
    ///
    /// The inertia is the prism's own and not the capsule's, and it is closed form, which
    /// is half the reason this shape rather than a general hull. For a regular `n`-gon of
    /// circumradius `R` the second moment about the axis through its centre is
    /// `R^2 (1 + 2 cos^2(pi/n)) / 6` per unit mass -- which tends to `R^2 / 2`, the
    /// cylinder's, as `n` grows, and is less than it for any finite `n` because the corners
    /// are the only part of a circle a polygon keeps. The transverse axes are that halved
    /// plus the rod term `h^2 / 3`, exactly as a cylinder's are.
    pub fn prism(mass: f64, radius: f64, length: f64, facets: u32, position: (f64, f64, f64)) -> Body {
        let mut body = Body::capsule(mass, radius, length, position);
        if facets < 3 {
            return body;
        }
        body.facets = facets;
        let half = 0.5 * length;
        let across = (std::f64::consts::PI / facets as f64).cos();
        // About the long axis, then the two transverse axes.
        let spin = radius * radius * (1.0 + 2.0 * across * across) / 6.0;
        let over = half * half / 3.0 + 0.5 * spin;
        let inv = |i: f64| if mass > 0.0 && i > 0.0 { 1.0 / (mass * i) } else { 0.0 };
        body.inv_inertia = (inv(over), inv(spin), inv(over));
        body
    }

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
            facets: 0,
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
            facets: 0,
        }
    }
}

/// What holds two bodies together, by index into the [`Skeleton`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Joint {
    /// **A point shared by two bodies**, each anchor given in its own body's frame. The
    /// shoulder and the hip: three degrees of rotational freedom, none of translation.
    ///
    /// **`cone` is the range on those three**, and it is what keeps a limb out of the body
    /// it hangs off. `axis_a` is the cone's axis in `a`'s frame, `axis_b` is the direction
    /// the limb points in `b`'s own frame, and the joint holds the angle between them to
    /// `cone` radians or less. A shoulder is about two fifths of a turn; a hip rather less.
    ///
    /// A `cone` of `PI` or more is a ball joint with no range, which is what this was
    /// before the range existed -- [`Joint::free_ball`] spells that, and the axes are then
    /// not read at all. The range is the caller's to state because only the caller knows
    /// which joint this is; the crate holds no anatomy.
    ///
    /// **Without it a contact has to do the job.** An elbow cannot fold backwards because
    /// [`Joint::Hinge`]'s `min` and `max` say so, but a ball could rotate its limb straight
    /// through the chest, and the only thing stopping it was a self-collision contact.
    /// Measured, two fifths of a heap's contacts are a rig against itself, and a heap of
    /// four rigs that never comes to rest with them comes to rest in twenty-five seconds
    /// without -- so the missing constraint was being paid for twice, in the narrow phase
    /// and in a pile that would not settle. See this module's header.
    Ball {
        a: usize,
        b: usize,
        anchor_a: (f64, f64, f64),
        anchor_b: (f64, f64, f64),
        axis_a: (f64, f64, f64),
        axis_b: (f64, f64, f64),
        cone: f64,
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
    /// **A ball joint with no range**, which is every ball joint this module had before
    /// ranges existed. The limb may point anywhere.
    ///
    /// Use it where the joint really has no meaningful limit, or where the caller has not
    /// yet measured one. A shoulder is not one of those: see [`Joint::socket`].
    pub fn free_ball(
        a: usize,
        b: usize,
        anchor_a: (f64, f64, f64),
        anchor_b: (f64, f64, f64),
    ) -> Joint {
        Joint::Ball {
            a,
            b,
            anchor_a,
            anchor_b,
            // Not read, because the cone is wider than the sphere. Any direction will do
            // and this one is the module's own convention for a body's length.
            axis_a: (0.0, 1.0, 0.0),
            axis_b: (0.0, 1.0, 0.0),
            cone: f64::INFINITY,
        }
    }

    /// **A ball joint with a cone on it**: the shoulder and the hip.
    ///
    /// `axis_a` is the cone's axis in `a`'s frame and `axis_b` the limb's direction in
    /// `b`'s, both taken as directions and normalised on the way in. `cone` is the
    /// half-angle in radians, so a shoulder that swings two fifths of a turn from its rest
    /// direction is `cone: 1.2` or so. `PI` or more is [`Joint::free_ball`].
    pub fn socket(
        a: usize,
        b: usize,
        anchor_a: (f64, f64, f64),
        anchor_b: (f64, f64, f64),
        axis_a: (f64, f64, f64),
        axis_b: (f64, f64, f64),
        cone: f64,
    ) -> Joint {
        Joint::Ball {
            a,
            b,
            anchor_a,
            anchor_b,
            axis_a,
            axis_b,
            cone,
        }
    }

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

/// **Which of the three things one impulse folded into a correction is.**
///
/// The first two are the two pairs of fields on [`Correction`]. The third is the same
/// fields as the first, delivered by [`scatter::Bodies::apply_velocity`] instead of
/// [`scatter::Bodies::apply`] -- and it is a third kind rather than a spelling of the
/// first, because what the applying half does with it is the opposite: it moves where the
/// body *came from* instead of where it is, so the read-back velocity changes and the
/// position does not.
///
/// It is the shape a position-based solver needs and does not otherwise have, and without
/// it a contact's energy has nowhere to go: charging a correction as velocity puts
/// momentum in, charging it as free leaves the momentum where it was, and there is no
/// third answer until something can take momentum out without moving anything. See the
/// module header's section on the velocity pass.
///
/// **Sharing the fields is a measurement rather than a shortcut.** Two more vectors on
/// [`Correction`] took it from 120 bytes to 176, and every joint and every contact in the
/// positional solve carries one whether or not it has anything velocity-level to say:
/// measured, `one/8` went from 38.8 to 53.6 us and `pile/8` from 3.24 to 4.92 ms, where
/// the pass itself is one traversal in nine. A correction is either positional or
/// velocity-level and never both -- the velocity pass has its own stages -- so the fields
/// are the same fields and the stage decides what they mean.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Charge {
    /// Move the body, and let the step read the move back as velocity. Every joint, and
    /// the share of a contact's overlap the step itself drove.
    Moving,
    /// Move the body without the step reading it back: the overlap a contact inherited
    /// rather than made.
    Free,
    /// Change what the step reads back without moving the body. The velocity pass, and
    /// nothing else.
    Still,
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
    Contacts { colour: u32, upto: u32, half: Half },
    /// The contacts colouring could not place. Run by one lane, because each reads the
    /// positions the one before it wrote.
    Overflow { half: Half },
    /// The joints colouring could not place. Empty unless a body carries more than
    /// sixty-four joints; see [`Skeleton::add_joint`].
    JointOverflow,
    Ground { half: Half },
}

/// **Which half of a contact a stage is running.**
///
/// A contact is two constraints that do not commute -- a normal one and a tangential one
/// -- and a pass runs every contact's normal half before any contact's tangential half, so
/// that friction acts on a configuration the normals have already agreed on rather than one
/// still moving underneath it. See the module header for the measurement that demanded it.
///
/// The colours are the same colours in both halves, so this doubles the stage list and
/// changes nothing about what a lane may touch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Half {
    Normal,
    Friction,
    /// The velocity pass's normal half: a resting contact is perfectly inelastic, so it
    /// may not leave the two surfaces separating. Runs once a step, after the passes.
    VelocityNormal,
    /// The velocity pass's tangential half: Coulomb and rolling resistance on what the
    /// positional solve left moving, out of what is left of the step's own cone.
    VelocityFriction,
}

/// **Which plan a call to [`Skeleton::solve_pass`] is walking.**
///
/// The three differ only in which stage list they name, which is why they are one
/// function: the whole positional plan, the foreground prefix of it a background island
/// gets, and the velocity pass. See [`Skeleton::plan_pass`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Which {
    Full,
    Near,
    Velocity,
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
    /// How many flats each body has around its own axis; zero is a capsule. See
    /// [`Body::facets`].
    facets: Vec<u32>,

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
    /// **The pairs that carried normal impulse last step**, sorted, so that the narrow
    /// phase can keep their contact alive for one more step while they are clear. See
    /// [`Skeleton::persist_contacts`].
    persisting: Vec<(u32, u32)>,
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
    ground_impulse: Vec<contacts::Patch>,

    /// **How hard each body was squeezed over the last step**, as a mean force in
    /// newtons. See [`Skeleton::normal_load`], which is the whole of the argument for
    /// what the number is and why it is that one.
    normal_load: Vec<f64>,
    /// Which entries of `normal_load` the last step wrote, so that clearing it costs the
    /// bodies that carried load rather than the whole set. A body may appear twice; the
    /// clear is idempotent, and a list that is never searched is cheaper to fill
    /// carelessly than to keep unique.
    normal_loaded: Vec<u32>,

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
    /// The velocity pass's plan: the same contact colours and the same ground set, in the
    /// two velocity-level halves, and no joints. See [`Skeleton::plan_pass`].
    plan_velocity: Vec<Stage>,
    plan_work: usize,
    plan_near_work: usize,
    plan_velocity_work: usize,
    /// **What fraction of one constraint's velocity-level correction a body may take.**
    ///
    /// The velocity pass reads one frozen state and every constraint answers against it,
    /// so a body named by three contacts is asked for three full corrections and would be
    /// corrected three times over. The reciprocal of how many constraints name it is the
    /// mean of what they asked for, which is what a simultaneous pass means.
    ///
    /// A pair takes the **smaller** of its two ends' shares, and takes it at both ends.
    /// The two must be one number: the share of an impulse that is momentum exchange
    /// cannot be one figure for the body receiving it and another for the body giving it,
    /// or the pair stops being equal and opposite and
    /// `a_skeleton_left_to_itself_does_not_move_its_own_centre_of_mass` fails. Measured on
    /// an earlier version of exactly this mistake, a free skeleton's centre moved 1.8 m.
    velocity_share: Vec<f64>,

    /// The broad phase. See [`broadphase`] for why it is a grid.
    grid: Grid,

    // -- sleeping. See [`sleep`] for the whole of the reasoning. ------------------
    /// Whether settled bodies may be left out of a step at all.
    sleeping: bool,
    /// How many lanes a parallel pass may use, or `None` for the default. See
    /// [`Skeleton::set_lanes`].
    lanes: Option<usize>,
    /// Whether the last solve pass was divided across lanes. See
    /// [`Skeleton::solved_in_parallel`].
    solved_in_parallel: bool,
    /// One bit per body: set means the body is simulated this step. A pinned body is
    /// never set, because it cannot move and there is nothing to simulate.
    awake: BitSet,
    /// One bit per body: the caller has taken it out of the solve for good. See
    /// [`Skeleton::retire`], which is also where the argument for why this is a bit
    /// rather than a shape test is.
    retired: BitSet,
    /// Bodies the settling test last found **moving** -- the ones whose neighbours the
    /// broad phase wakes ahead of them. See [`Skeleton::find_pairs`] and [`sleep`].
    disturbing: BitSet,
    /// The bodies the broad phase has already swept as it walks outward from the awake
    /// set, and the ones it is sweeping now. See [`Skeleton::find_pairs`].
    swept: BitSet,
    frontier: BitSet,
    next_frontier: BitSet,
    /// Sleeping bodies the sweep found within reach of a moving one, woken before the next
    /// round. See [`Skeleton::find_pairs`].
    reached: Vec<usize>,
    /// Where [`Skeleton::wake_island`] reads an island's members, so that retiring a body
    /// allocates nothing.
    island_members: Vec<usize>,
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
            facets: Vec::new(),
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
            persisting: Vec::new(),
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
            normal_load: Vec::new(),
            normal_loaded: Vec::new(),
            ground_anchor: Vec::new(),
            ground_stuck: BitSet::default(),
            ground_sticking: Vec::new(),
            anchor_reach: 0.0,
            plan: Vec::new(),
            plan_near: Vec::new(),
            plan_velocity: Vec::new(),
            plan_work: 0,
            plan_near_work: 0,
            plan_velocity_work: 0,
            velocity_share: Vec::new(),
            grid: Grid::default(),
            sleeping: true,
            lanes: None,
            solved_in_parallel: false,
            awake: BitSet::default(),
            retired: BitSet::default(),
            disturbing: BitSet::default(),
            swept: BitSet::default(),
            frontier: BitSet::default(),
            next_frontier: BitSet::default(),
            reached: Vec::new(),
            island_members: Vec::new(),
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
    /// **It is on by default**, because a rig whose limbs pass through each other is
    /// wrong in a way anybody can see, where the cost of it is a frame-budget question the
    /// caller is better placed to answer. The trade is real and it is the one above: a rig
    /// that is looked at closely wants its limbs to collide, and a rig in a crowd wants to
    /// stop costing anything once it has landed. A caller with crowds turns it off.
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
    ///
    /// **A few of them may not be overlapping.** A pair that carried load last step keeps
    /// its constraint for one more, so that the step which solves a resting pair exactly
    /// together does not leave it unconstrained -- see the module header. Such a contact is
    /// inert while its gap is open, so this counts constraints in the step rather than
    /// surfaces in collision, and the two differ by the pairs that have just parted.
    pub fn contact_count(&self) -> usize {
        self.contacts.len()
    }

    /// How many candidate pairs the last [`Skeleton::step`]'s broad phase handed the
    /// narrow phase.
    ///
    /// The number a broad phase is actually judged by, and the one that says whether it
    /// is still a broad phase. The contacts it leads to are a property of the scene; this
    /// is a property of the *search*, so a grid whose cell has been coarsened -- by one
    /// body far larger than the rest, say -- shows up here, and in the step time, and in
    /// nothing else at all.
    pub fn candidate_pairs(&self) -> usize {
        self.pairs.len()
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
        self.facets.push(body.facets);
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
        self.normal_load.push(0.0);
        let i = self.position.len() - 1;
        let n = self.position.len();
        self.awake.resize(n, false);
        self.retired.resize(n, false);
        self.ground_stuck.resize(n, false);
        self.swept.resize(n, false);
        self.frontier.resize(n, false);
        self.next_frontier.resize(n, false);
        self.ready.resize(n, false);
        self.background.resize(n, false);
        // **A body that has just arrived counts as moving**, whatever it is doing, because
        // it may have been put down inside a settled pile and nothing in the pile has had
        // a chance to notice. The settling test takes the bit off again at the end of the
        // first step it spends still.
        self.disturbing.resize(n, true);
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
            facets: self.facets[i],
        }
    }

    /// Writes one body back. The whole body, because a caller that has one has usually
    /// changed more than one field of it.
    ///
    /// **Wakes the body's island.** A caller writing a body is the one disturbance the
    /// solver cannot see coming, and a teleported body that stays asleep is a body that
    /// never collides with anything again.
    ///
    /// **Does nothing to a retired body**, silently. Retirement is permanent by
    /// definition -- see [`Skeleton::retire`] -- and a write that restored a shape and a
    /// mass would quietly bring one back into the solve with no joints and no history.
    pub fn set_body(&mut self, i: usize, body: Body) {
        if self.retired.get(i) {
            return;
        }
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
        self.facets[i] = body.facets;
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

    /// **How hard this body was squeezed by the last step**, as a mean force in newtons.
    ///
    /// The sum over every normal constraint that named the body -- pair contacts and the
    /// plane alike -- of the normal impulse it was actually handed, divided by the step.
    /// Zero for a body nothing pressed on, and for a body that has never been stepped.
    ///
    /// # Which impulse, and why the other one is wrong
    ///
    /// [`contacts::Spent`] distinguishes two totals, and only one of them is a load.
    /// `normal` is the whole normal impulse a contact applied, which includes the solver
    /// lifting the bodies out of an overlap they were **already** in when the step began;
    /// `driven` is the part the step itself drove, and so the only part the bodies were
    /// handed as momentum. This is the sum of `driven`.
    ///
    /// The difference is the difference between a crushed body and a carelessly placed
    /// one. A body spawned half inside another is separated over its first few steps by
    /// an enormous `normal` while nothing whatever is pressing on it, and reading that
    /// would make a spawn look like an impact. `driven` reads near zero through the same
    /// separation, because the overlap was inherited rather than made. See
    /// [`contacts::solve_contact_normal`], where the split is taken, and
    /// `an_overlapping_spawn_is_not_a_crushed_body`, which is the guard on it.
    ///
    /// # What the number is, dimensionally
    ///
    /// The solver's impulses are in the convention `correction = impulse * inv_mass`, so
    /// a raw one is a mass times a distance -- kilogram metres, not the newton seconds a
    /// reader would assume from the word. Dividing by the step turns it into the momentum
    /// the body was handed (`kg m / s`, which *is* newton seconds) and dividing again
    /// turns that into the mean force over the step, in newtons. **This is the second
    /// one**, and it is divided by `dt` twice for that reason.
    ///
    /// Newtons rather than the raw total because the raw total scales with `dt * dt`: a
    /// caller who halved their timestep would find the same physical squeeze reading a
    /// quarter as large, and whatever rule they had written against it would silently
    /// change meaning. A force does not move. The check that it is the right force is
    /// that **a body lying on the plane reads its own weight**: it sags `g dt^2` in a
    /// step, the plane drives that back out, the impulse is `m g dt^2`, and two divisions
    /// by `dt` leave `m g`.
    ///
    /// # Turning it into a stress
    ///
    /// Divide by the area the load is carried over. The crate does not know that area --
    /// a capsule's contact patch depends on how far the two surfaces flatten, which is a
    /// property of the material and not of the geometry -- so a caller who wants a stress
    /// supplies it. `2 * radius * half_length` is the projected side of a capsule and is
    /// the usual stand-in; the resulting pascals are then comparable with a compressive
    /// strength.
    ///
    /// # What it sums, and what that means for two-sided load
    ///
    /// The sum of the **magnitudes** of the normal impulses, not their vector sum. Two
    /// opposed forces -- the plane below and something heavy above -- add, which is what
    /// being crushed is; a body merely being accelerated by one push adds the same way,
    /// which overstates it by the factor the second surface would have contributed.
    /// A body genuinely in a vice is the case this is for, and it is the case the scalar
    /// sum is exactly right for.
    ///
    /// The crate has no opinion on how much is too much. What load breaks a body is a
    /// material judgement that varies by what the bodies represent, and there is no
    /// threshold, no strength, and nothing resembling one anywhere in here: the solver
    /// reports, and the caller decides. [`Skeleton::retire`] is what a caller who has
    /// decided reaches for.
    pub fn normal_load(&self, i: usize) -> f64 {
        self.normal_load[i]
    }

    /// **Takes a body and its joints out of the solve, for good.** Returns `false` and
    /// changes nothing for an index that does not exist or a body that was already
    /// retired.
    ///
    /// What it is for is structural failure: a body has been loaded past what whatever it
    /// represents can carry -- [`Skeleton::normal_load`] is how a caller sees that -- and
    /// what is left is no longer a rigid body. **Destruction here is subtraction.**
    /// Nothing is emitted, nothing flies off, and no body is created; the simulation gets
    /// *cheaper*, so a field driven through leaves fewer bodies behind than in front of
    /// it. The crate holds no threshold and no material strength: what load is too much
    /// varies by what the bodies represent, so the solver reports and the caller decides.
    ///
    /// It is also the general primitive rather than a special case of crushing -- taking
    /// a body and its joints out is what cutting a skeleton apart needs too.
    ///
    /// # Its joints go with it
    ///
    /// A smashed bone that still anchors its neighbours is wrong: retiring a hip has to
    /// let the leg come away. So every joint naming the body is removed, which is the one
    /// thing the incremental colouring cannot absorb -- see [`Skeleton::recolour_joints`],
    /// which is paid here because retirement is rare.
    ///
    /// **Joint indices shift.** [`Skeleton::joints`] is a vector and removing from the
    /// middle of it moves everything after; a caller holding an index into that slice has
    /// to re-read it. Body indices do not, which is the next section.
    ///
    /// # Indices stay stable
    ///
    /// The arrays are **not compacted**. Callers hold body indices, the ground anchors are
    /// indexed by body, and a body index has meant the same body for the skeleton's life
    /// because bodies are only ever added. [`Skeleton::len`] therefore counts retired
    /// bodies, and [`Skeleton::body`] still answers for one -- with the shape and the mass
    /// taken off it, at the place it was retired.
    ///
    /// # It costs nothing afterwards, and not by being tested for
    ///
    /// There is no "is it retired" branch anywhere in the step. Two facts already in the
    /// solver do the whole of it, and each of them is load-bearing:
    ///
    /// * **No radius means no broad phase and no plane.** [`broadphase::Grid::rebuild`]
    ///   puts a body in the grid only if `radius > 0`, so a retired body is neither an
    ///   outer body of the scan nor a candidate in anybody else's neighbourhood; and
    ///   [`contacts::ground_contacts`] returns on the same test. It can therefore take no
    ///   contact of any kind, and appears in no contact colour and in no ground colour.
    /// * **No mass means never awake.** [`Skeleton::wake`] sets the awake bit only for a
    ///   body with `inv_mass > 0` and [`Skeleton::wake_all`] clears it again for one
    ///   without, which is how a *pinned* body is already kept out of every sweep. So a
    ///   retired body is permanently unready as well: [`Skeleton::settle`] only ever looks
    ///   at awake bodies, so it can never be marked still, never join an island, and never
    ///   be woken by a neighbour.
    ///
    /// The bit this sets is not consulted by the step at all. It exists so that retirement
    /// is *permanent* -- [`Skeleton::set_body`] would otherwise restore a shape and a mass
    /// and quietly bring the body back with no joints and no history, and
    /// [`Skeleton::add_joint`] would otherwise anchor a new joint to it.
    ///
    /// # A joint the caller still holds is handled, not refused
    ///
    /// Retiring a body some joint still names is the ordinary case rather than an error:
    /// it is what "the leg comes away" means. So the joints go quietly and nothing is
    /// refused. The precedent is [`Skeleton::add_joint`], which returns `false` for a bad
    /// index rather than panicking -- a solver called every frame on data a caller
    /// assembled is the wrong place to unwind -- and the same reading applies here: the
    /// only `false` is for a body that is not there or is already gone, and a caller that
    /// does not look has still not been surprised.
    ///
    /// # Its neighbours wake
    ///
    /// Taking a support away changes the situation for whatever was leaning on it, and
    /// everything that was is asleep precisely because it had stopped. So the body's own
    /// island is thawed -- which is what wakes a sleeping stack it was part of -- and then
    /// everything jointed to it and everything the broad phase last paired it with. The
    /// broad phase's pairs rather than the narrow phase's contacts, for the reason
    /// [`Skeleton::settle`] gives: two bodies resting exactly against one another overlap
    /// by nothing and have no contact, and the better the solve gets the more often that
    /// is true.
    #[must_use = "a body that refused to retire is still in the solve, and nothing else \
                  will say so"]
    pub fn retire(&mut self, i: usize) -> bool {
        if i >= self.position.len() || self.retired.get(i) {
            return false;
        }

        // First, while the body is still in the graph. **Its own island goes with it**,
        // which is the one place left that wakes a component rather than a body: a body
        // asleep in a stack is asleep *with* the bodies it is holding up, and taking it out
        // is the one disturbance the broad phase cannot see coming. Nothing arrives, so
        // nothing comes within reach; the stack is perfectly settled, so there are no
        // contacts to lose. See [`Skeleton::wake_island`].
        self.wake_island(i);
        for k in 0..self.joints.len() {
            let (a, b) = self.joints[k].bodies();
            if a == i {
                self.wake(b);
            } else if b == i {
                self.wake(a);
            }
        }
        for k in 0..self.pairs.len() {
            let (a, b) = self.pairs[k];
            if a == i {
                self.wake(b);
            } else if b == i {
                self.wake(a);
            }
        }

        self.retired.set(i);
        // No extent: out of the grid, out of every neighbourhood, and out of the plane's
        // contact generation. See the doc comment.
        self.radius[i] = 0.0;
        self.half_length[i] = 0.0;
        // A retired body is not any shape at all.
        self.facets[i] = 0;
        // No mass: never awake again, by the rule that already keeps pinned bodies out.
        self.inv_mass[i] = 0.0;
        self.inv_inertia[i] = (0.0, 0.0, 0.0);
        // Nothing left reads these, but a stale velocity on a body a caller can still ask
        // about would be a lie about a body that is not moving.
        self.velocity[i] = (0.0, 0.0, 0.0);
        self.angular_velocity[i] = (0.0, 0.0, 0.0);
        self.prev_position[i] = self.position[i];
        self.prev_orientation[i] = self.orientation[i];
        self.awake.unset(i);
        self.ready.unset(i);
        self.disturbing.unset(i);
        self.ground_stuck.unset(i);
        self.island_of[i] = NO_ISLAND;
        self.still_steps[i] = 0;
        self.normal_load[i] = 0.0;

        let before = self.joints.len();
        self.joints.retain(|joint| {
            let (a, b) = joint.bodies();
            a != i && b != i
        });
        if self.joints.len() != before {
            self.recolour_joints();
            // The jointed-neighbour runs and the component labels are both built from the
            // joint set.
            self.jointed_built = false;
        }
        true
    }

    /// **How many threads a parallel pass may use.** The default is the whole pool.
    ///
    /// This exists because a solver is a subsystem of something with a frame to fill: the
    /// threads it may have are whatever is left after rendering, audio and the rest, and
    /// only the caller knows what that is. A caller that wants physics to keep out of two
    /// cores says so here.
    ///
    /// **It is not the dial for the cost of an oversubscribed pool, and it was nearly
    /// shipped as one.** The measurement that started this was `pile` at eight iterations
    /// under `RAYON_NUM_THREADS=18` against the default thirty-six, on a machine with
    /// eighteen cores and thirty-six hardware threads -- 5.24 ms against 6.18, and a
    /// round-to-round spread of one per cent against six. That looks exactly like a spin
    /// barrier with more lanes than can run at once, and it was read as one. Holding the
    /// lane count fixed and varying only the pool says otherwise:
    ///
    /// ```text
    ///   pool 36, 18 lanes   6.61  6.77 ms
    ///   pool 18, 18 lanes   5.26  5.21  5.25 ms
    ///   pool 18,  9 lanes   7.07  7.10 ms
    ///   pool 12,  6 lanes   8.91  9.11 ms
    /// ```
    ///
    /// Eighteen lanes cost 6.6 ms in a pool of thirty-six and 5.2 in a pool of eighteen, so
    /// what the first measurement found was **the pool**, not the lanes: the eighteen extra
    /// workers take no lane and return from the broadcast at once, then spend the pass
    /// looking for work that is not there, a couple of hundred times a step. And within a
    /// given pool, fewer lanes is plainly worse -- halving them costs a third. More lanes
    /// is better up to the pool size, which is what [`crew::lanes_for`] already said.
    ///
    /// So the thing a caller should not do is hand this crate a rayon pool larger than the
    /// machine can run, and that is a property of the pool rather than anything this crate
    /// can fix from the inside. Lowering the budget does not recover it; it makes it worse.
    ///
    /// Zero is read as one. The count is capped at the pool's own thread count, since a
    /// lane the broadcast cannot deliver is a lane the gate would wait for for ever.
    ///
    /// **It does not change the answer.** How many lanes ran is not something the result
    /// may depend on -- `the_answer_does_not_depend_on_how_many_threads_ran_it` is the law
    /// -- so this is a performance dial and nothing else.
    pub fn set_lanes(&mut self, lanes: usize) {
        self.lanes = Some(lanes.max(1));
    }

    /// The lane budget in force. See [`Skeleton::set_lanes`].
    pub fn lanes(&self) -> usize {
        self.lanes.unwrap_or_else(rayon::current_num_threads)
    }

    /// **Whether the last solve divided its work across lanes**, rather than running the
    /// whole of it on the calling thread.
    ///
    /// This is here because of what it is like to be without it. The crate's sharpest
    /// promise is that the answer does not depend on how many threads computed it, and the
    /// law that states it ran a heap of forty bodies at one, two and eight threads and
    /// compared the results. That heap's widest pass is about two hundred constraints
    /// against a [`crew::PASS_FLOOR`] of five hundred and twelve, so every one of those
    /// runs took the serial path: the law was comparing single-threaded output with
    /// single-threaded output and could not have failed. The parallel writes it was
    /// written to police are the raw-pointer scatter in [`scatter`], so it was the guard on
    /// the unsafe as well.
    ///
    /// A fixture can drift under a floor without anybody noticing, and no amount of care
    /// in the test will see it, because the thing to assert is not visible from outside. So
    /// it is visible now. A caller sizing scenes can use it for the same reason: it says
    /// whether a scene is large enough to be worth the pool at all.
    pub fn solved_in_parallel(&self) -> bool {
        self.solved_in_parallel
    }

    /// Whether this body has been retired. See [`Skeleton::retire`].
    pub fn is_retired(&self, i: usize) -> bool {
        self.retired.get(i)
    }

    /// Forgets the last step's loads. Costs the bodies that carried one, not the set.
    fn clear_normal_load(&mut self) {
        for &i in self.normal_loaded.iter() {
            self.normal_load[i as usize] = 0.0;
        }
        self.normal_loaded.clear();
    }

    /// **The per-body total of [`Skeleton::normal_load`]**, gathered once at the end of
    /// the step out of the running totals the constraints already keep.
    ///
    /// Nothing is added to the inner loop for this. Every contact and every ground patch
    /// already carries its `Spent` across the passes, because Coulomb's cone needs the
    /// step's totals and the velocity pass needs to know what it is allowed to take back;
    /// so the load is already computed by the time the step ends and this is one pass over
    /// two lists that are in cache, not a write inside eight passes over them. Running it
    /// after the velocity pass rather than after the positional one is deliberate as well:
    /// that pass **removes** normal impulse where a resting contact ended the step
    /// separating, and what a caller wants is the net momentum the step handed the body
    /// rather than the gross the positional passes applied before it was taken back.
    ///
    /// Deterministic because it sums in list order, and both lists are built in a fixed
    /// order on every machine -- the contacts by the narrow phase's fixed-size chunks, the
    /// ground patches by increasing body index.
    fn gather_normal_load(&mut self, dt: f64) {
        self.clear_normal_load();
        // Twice, and the second division is what makes this a force rather than a
        // momentum. See [`Skeleton::normal_load`].
        let per_step = 1.0 / (dt * dt);
        for (k, contact) in self.contacts.iter().enumerate() {
            let driven = self.contact_impulse[k].driven;
            if driven <= 0.0 {
                continue;
            }
            let load = driven * per_step;
            for end in [contact.a, contact.b] {
                if self.normal_load[end] == 0.0 {
                    self.normal_loaded.push(end as u32);
                }
                self.normal_load[end] += load;
            }
        }
        for (k, ground) in self.ground_contacts.iter().enumerate() {
            let driven = self.ground_impulse[k].spent.driven;
            if driven <= 0.0 {
                continue;
            }
            let i = ground.body;
            if self.normal_load[i] == 0.0 {
                self.normal_loaded.push(i as u32);
            }
            self.normal_load[i] += driven * per_step;
        }
    }

    /// Sets a body's angular velocity, and **wakes its island**: a caller pushing a body
    /// is a disturbance the settling test cannot see, and it has to reach the bodies
    /// leaning on it as well as the one that was pushed.
    ///
    /// Does nothing to a retired body, for the reason [`Skeleton::set_body`] gives.
    pub fn set_angular_velocity(&mut self, i: usize, w: (f64, f64, f64)) {
        if self.retired.get(i) {
            return;
        }
        self.wake(i);
        self.angular_velocity[i] = w;
    }

    /// Sets a body's velocity, and **wakes its island**. See
    /// [`Skeleton::set_angular_velocity`].
    pub fn set_velocity(&mut self, i: usize, v: (f64, f64, f64)) {
        if self.retired.get(i) {
            return;
        }
        self.wake(i);
        self.velocity[i] = v;
    }

    /// Adds a joint. Returns `false` and adds nothing if it names a body that does not
    /// exist, joints a body to itself, or names a body that has been **retired** -- an
    /// out-of-range index is a caller's bug and panicking in a solver that runs per frame
    /// is worse than refusing, and a retired body is exactly as absent as one that was
    /// never added. See [`Skeleton::retire`].
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
    #[must_use = "a refused joint is not in the skeleton, and nothing else will say so"]
    pub fn add_joint(&mut self, mut joint: Joint) -> bool {
        let (a, b) = joint.bodies();
        let n = self.position.len();
        if a >= n || b >= n || a == b || self.retired.get(a) || self.retired.get(b) {
            return false;
        }
        // **A hinge axis is a direction, so it is stored as one.** The alignment
        // correction takes its angle as `asin` of the cross product's length, which is
        // `|axis_a| |axis_b| sin t` and only equals `sin t` for unit axes. Given a pair of
        // axes twice as long it reads four times the sine, saturates the clamp at a
        // quarter turn, and asks for a quarter turn whatever the error actually is --
        // which walks the hinge into the inverted basin
        // `a_hinge_does_not_turn_itself_inside_out` exists to forbid, and it does not come
        // back. `(2, 0, 0)` is the same hinge as `(1, 0, 0)` and must behave like it, so
        // the contract is made true here rather than written in the doc and hoped for.
        //
        // An axis of no length is refused rather than normalised. There is no direction to
        // recover, and letting it through degrades the hinge to a ball joint in silence:
        // every branch of the hinge solve is guarded by a `normalized` that would hand
        // back `None`, so the range would stop being enforced and nothing would say so.
        // A cone's axes are read the same way a hinge's are, so they get the same
        // treatment: normalised here so the contract is true rather than documented, and a
        // joint with no direction to point in is refused. A ball with no cone never reads
        // them, so it is left alone -- `free_ball` fills them with a placeholder and
        // nothing may depend on what it is.
        let coned = match joint {
            Joint::Hinge { .. } => true,
            Joint::Ball { cone, .. } => cone < std::f64::consts::PI,
        };
        if coned {
            let (axis_a, axis_b) = match &mut joint {
                Joint::Hinge { axis_a, axis_b, .. } => (axis_a, axis_b),
                Joint::Ball { axis_a, axis_b, .. } => (axis_a, axis_b),
            };
            let (Some(unit_a), Some(unit_b)) = (normalized(*axis_a), normalized(*axis_b))
            else {
                return false;
            };
            *axis_a = unit_a;
            *axis_b = unit_b;
        }
        let index = self.joints.len();
        self.joints.push(joint);
        self.colour_joint(index);
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

    /// **One joint's colour**: the lowest one neither of its bodies is already using.
    ///
    /// Lifted out of [`Skeleton::add_joint`] so that the rule has one statement rather
    /// than two. [`Skeleton::recolour_joints`] replays it over a joint set a retirement
    /// has taken something out of, and a greedy colouring that differed between the two
    /// would leave `joint_bits` describing an assignment `colours` did not have -- after
    /// which the next arrival would be coloured against a fiction.
    fn colour_joint(&mut self, index: usize) {
        let (a, b) = self.joints[index].bodies();
        let taken = self.joint_bits[a] | self.joint_bits[b];
        if taken == u64::MAX {
            // Sixty-five joints on one body. Nothing a skeleton does reaches it, and a
            // serial tail is a better answer than a colour nobody can parallelise.
            self.joint_overflow.push(index);
            return;
        }
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

    /// **The whole joint set coloured again from nothing**, which is what removing a
    /// joint costs.
    ///
    /// [`Skeleton::add_joint`] never needs this: an edge added to a proper edge-colouring
    /// leaves it proper, so an arrival is two loads and a `trailing_ones`. Removal is the
    /// other case, and it is the one the incremental comment there already named -- a
    /// full recolour is wanted "if a caller removes joints, or never". Removing a joint
    /// frees colours on both of its bodies and, because the joints are named by their
    /// index into one vector, shifts the index of every joint after it; there is nothing
    /// incremental left of either.
    ///
    /// **It is the same greedy rule in the same order**, so what it leaves behind is
    /// exactly the state adding the surviving joints one at a time would have produced.
    /// That is what keeps the incremental path correct across a removal rather than
    /// merely unbroken: the next [`Skeleton::add_joint`] reads a `joint_bits` that agrees
    /// with `colours`, which is the invariant the constant-time colouring rests on.
    ///
    /// Trailing empty colours are dropped, so a set that used to need four and now needs
    /// three reports three -- the colour count is what says how parallel a step can be,
    /// and a caller reading it should not be told about colours nothing is in.
    fn recolour_joints(&mut self) {
        for set in self.colours.iter_mut() {
            set.clear();
        }
        self.joint_overflow.clear();
        for bits in self.joint_bits.iter_mut() {
            *bits = 0;
        }
        for index in 0..self.joints.len() {
            self.colour_joint(index);
        }
        while self.colours.last().is_some_and(|set| set.is_empty()) {
            self.colours.pop();
        }
        self.live_joints.truncate(self.colours.len());
        self.live_joints_near.truncate(self.colours.len());
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
    /// **A sleeping body is woken when a body that is actually moving comes within reach
    /// of it**, and is swept on the next step rather than this one.
    ///
    /// Both halves of that carry weight. Waking from the *awake* set instead is what the
    /// island thaw does one step at a time: nearly every body in a settled field is within
    /// a body's length of something awake, so the awake region then grows a ring a step
    /// whether or not anything is happening, out to the whole component -- measured below,
    /// and it is the defect this rule exists to close. What may carry the front is
    /// therefore the set the settling test last found moving, which is the same threshold
    /// that decides sleeping and so costs no second constant. A body woken here has not
    /// moved, so it carries the front no further until something pushes it.
    ///
    /// The lead this buys is a length rather than a choice. A pair is reported as soon as
    /// the centres are within the two reaches, so the neighbour of a moving body wakes a
    /// body's own size before anything touches it -- and it needs that lead rather than
    /// the touch itself, because a pile that has settled perfectly carries **no contacts**:
    /// the solve removes the whole overlap, and it is the woken body's own sag of `g dt^2`
    /// that gives it back a contact with what it is resting on. Waking on the touch alone
    /// is measured in [`sleep`] and it drives a stack apart.
    ///
    /// Slower than that threshold there is no lead and none is needed, and
    /// [`Skeleton::wake_touched`] is what catches it: a body the settling test calls still
    /// is one that crosses less than [`STILL_FRACTION`] of its own reach in a window, so
    /// the overlap it can present a sleeping neighbour with before the touch wakes it is
    /// bounded by the same fraction of its own size.
    ///
    /// **The sweep still runs in rounds, and now it terminates by itself.** A body woken by
    /// a round is still swept by the next one -- it has to be, or its own neighbours
    /// further into the pile are never looked at by anybody, and the contact that carries
    /// the disturbance on is a step late. What the round after it cannot do is wake
    /// anything: a body woken and not yet moved is not in `disturbing`, so the round
    /// produces its pairs and no wakes and the loop ends. **Two rounds, always**, where the
    /// same loop gated on the awake set ran until it had covered the component.
    ///
    /// Letting a woken body carry the front one round further -- the same rounds, gated on
    /// the round before's wakes -- was built and measured. It reproduces the island thaw
    /// exactly on a three-body stack, which is the whole of what it can reach there, and on
    /// the drop sweep in [`sleep`] it goes back to failing the twelve heights of
    /// twenty-eight that the thaw fails. There is nothing to gate it with beyond that
    /// either, which is what says the count is not a quantity: one round is "what is
    /// moving", and every count after it is a number somebody chose.
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
                &self.disturbing,
                &mut pairs,
                &mut reached,
            );
            self.swept.union(&self.frontier);
            if reached.is_empty() {
                break;
            }
            // In the order the chunks were concatenated, which is the same order on every
            // machine. A body may be reached by more than one neighbour; waking it twice
            // is the second call finding it awake already.
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

    /// **Wakes whatever the narrow phase found something touching**, before the solve
    /// runs, so that a contact never names a body the step is not simulating.
    ///
    /// **This is the floor under the threshold [`Skeleton::find_pairs`] wakes on, and it is
    /// what makes that threshold safe rather than a tolerance.** Waking ahead of something
    /// moving needs the thing to be moving by the settling test's own measure; below that
    /// the front has no lead and needs none, because a body the settling test calls still
    /// crosses less than [`STILL_FRACTION`] of its own reach in a window, so the overlap
    /// it can present a sleeping neighbour with before this wakes it is bounded by the
    /// same fraction of its own size. Nothing creeps into a sleeping pile unnoticed.
    ///
    /// It is the same causal rule [`Skeleton::persist_contacts`] already keeps a contact
    /// alive by -- what the pair *did*, not how far apart it is -- and it needs no margin
    /// and no second distance. What it cannot do on its own is carry a disturbance into a
    /// pile that has settled perfectly, because such a pile has no contacts to make; see
    /// [`sleep`] for the measurement that closed that off.
    fn wake_touched(&mut self) {
        if !self.sleeping {
            return;
        }
        // In contact order, which is the order the narrow phase produced and therefore the
        // same on every machine. Taken out so `wake` may borrow the rest of the struct.
        let contacts = std::mem::take(&mut self.contacts);
        for contact in contacts.iter() {
            if !self.awake.get(contact.a) {
                self.wake(contact.a);
            }
            if !self.awake.get(contact.b) {
                self.wake(contact.b);
            }
        }
        self.contacts = contacts;
    }

    /// The narrow phase: which candidates are actually touching, and where.
    fn build_contacts(&mut self) {
        let mut contacts = std::mem::take(&mut self.contacts);
        contacts.clear();
        let position = &self.position;
        let orientation = &self.orientation;
        let radius = &self.radius;
        let half_length = &self.half_length;
        // Sorted, so this is a binary search over a list the size of last step's loaded
        // contact set. See [`Skeleton::persist_contacts`].
        let persisting = &self.persisting;
        let test = |&(a, b): &(usize, usize)| {
            let alive = persisting.binary_search(&(a as u32, b as u32)).is_ok();
            capsule_contact(a, b, position, orientation, radius, half_length, alive)
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
    }

    /// The plane's share of the narrow phase, which runs **after** [`Skeleton::wake_touched`]
    /// so that a body woken by a contact this step still gets its ground contact in the
    /// same step. Without that it would be pushed by the contact that woke it with nothing
    /// under it, and would meet the plane a step late.
    fn build_ground_contacts(&mut self) {
        self.ground_contacts.clear();
        self.ground_colours.clear();
        let Some((normal, distance)) = self.ground else {
            self.ground_impulse.clear();
            return;
        };
        // Awake bodies only, and in increasing order, which is the order the loop this
        // replaced produced: a sleeping body is already resting on the plane -- that is
        // most of why it went to sleep -- and the plane cannot arrive underneath it.
        // **Destructured rather than taken out and put back.** The closure needs the body
        // arrays while `awake` is borrowed, and `mem::take` is the usual way to buy that --
        // but a take that is restored after the loop is not restored if the loop unwinds,
        // and what would be left behind here is an *empty* awake set: every body asleep,
        // `step` returning at its first line for ever, and a skeleton that looks fine to
        // every accessor. A caller that catches a panic rather than dying gets a corpse.
        let Skeleton {
            awake,
            position,
            orientation,
            radius,
            half_length,
            ground_contacts: out,
            ground_colours,
            ..
        } = self;
        awake.for_each_set(|i| {
            let before = out.len();
            ground_contacts(
                i,
                position[i],
                orientation[i],
                radius[i],
                half_length[i],
                normal,
                distance,
                out,
            );
            for index in before..out.len() {
                ground_colours.push(index);
            }
        });

        // One running impulse per contact, like every other constraint here. It used to
        // have to be one per *body*, pooled across the two ends of a capsule's patch,
        // because Coulomb's limit belongs to the patch and not to a sample of it -- and
        // that made it the one exception the [`scatter`] safety argument had to make a
        // case for. The patch is one constraint now, so the exception is gone: the
        // budget is the constraint's, indexed by the constraint.
        self.ground_impulse.clear();
        self.ground_impulse
            .resize(self.ground_contacts.len(), contacts::Patch::default());
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
    /// `iterations` is the quality dial: four is enough for a corpse, and the cost is
    /// linear in it. Zero is read as one, since a step that ran no passes would integrate
    /// gravity and solve nothing, which is a worse answer than a cheap one.
    ///
    /// **A `dt` that is not a positive, finite number does nothing**, and the spelling of
    /// that test is load-bearing twice over.
    ///
    /// `dt <= 0.0` lets a NaN straight through, because every comparison with a NaN is
    /// false -- and one NaN step puts NaN in every position and orientation in the
    /// skeleton, with nothing that clears it and no accessor that admits to it. A caller
    /// dividing by a frame rate can produce one on the frame a timer wraps.
    ///
    /// `dt > 0.0` alone still admits an infinity, which is worse than it sounds: it does
    /// not merely produce a wrong answer, it reaches a `clamp` on a NaN inside the solve
    /// and **panics**, from a call the caller had every reason to think was total.
    /// `a_step_of_no_time_or_of_nonsense_does_nothing` covers all four of zero, negative,
    /// NaN and infinite.
    pub fn step(&mut self, dt: f64, gravity: (f64, f64, f64), iterations: usize) {
        if !(dt > 0.0 && dt.is_finite()) || self.position.is_empty() {
            return;
        }
        // Everything has settled and nothing has disturbed it. There is no state a step
        // could change, so the cheapest honest answer is the whole step: `bodies / 64`
        // word tests and no memory touched. This is what sleeping is for.
        if !self.awake.any() {
            // Nothing was pressed on, so nothing carries a load. This walks the list of
            // bodies that carried one last step -- empty from the second idle step on --
            // rather than the whole set, so a settled scene still returns without
            // touching memory that is proportional to its size.
            self.clear_normal_load();
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
        // **Sized by what is awake, not by what exists**, which on the workload sleeping
        // is for is the whole of the difference. The sweep walks the awake set a word at a
        // time and does nothing for a clear bit, so a heap of ten thousand with three
        // body moving in it has one body's worth of work here -- and asking the pool for it
        // cost three forks, at the 26 to 72 us a fork this module has measured elsewhere.
        // Measured on 9,999 capsules resting in stacks of three, one body woken each step,
        // the two predicates alternated over three rounds:
        //
        // ```text
        //   by body count    709.2  734.0  732.5 us a step
        //   by awake count   468.0  463.3  464.6 us a step
        // ```
        //
        // Disjoint every round, and a third of the step gone. On `pile`, where everything
        // is awake, the same build reads 6.23 and 6.60 ms against a 6.0 to 6.4 baseline --
        // no signal, which is what the paragraph below predicts.
        //
        // With everything awake the two predicates are the same predicate, so nothing that
        // was parallel stops being parallel. `BitSet::count` is a popcount per sixty-four
        // bodies, which is the sweep's own loop again and did not show.
        let wide = self.awake.count() >= PARALLEL_FLOOR;

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
        // Before anything that reads the awake set: a contact that names a sleeping body
        // has to name an awake one by the time the solve reaches it, and a body woken here
        // needs the plane under it in the same step.
        self.wake_touched();
        self.build_ground_contacts();
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
            let which = if self.any_background && pass >= background {
                Which::Near
            } else {
                Which::Full
            };
            self.solve_pass(which, dt);
        }

        self.read_velocities(dt);

        // **The velocity pass**, which is the one thing an XPBD solver has that this did
        // not: a traversal that corrects velocities rather than positions, where zero
        // restitution and Coulomb's law act on what the positional solve left moving.
        //
        // It is **simultaneous**, and that is not an optimisation: every constraint reads
        // the velocities the sweep above has just written and none of them is rewritten
        // until the sweep below runs again, so the answer does not depend on the order the
        // contacts were visited in. An *ordered* velocity sweep was built first and is the
        // same mistake the positional solve's own header describes one level up -- a
        // Gauss-Seidel sweep round a loop leaves the same small bias every step, and the
        // ground rectifies it into travel. Measured on the seventeen-bone rig over
        // twenty-four draws, the ordered version took the worst drift from 0.19 of a reach
        // to 6.1 and its straightness from 0.31 to 0.90, and converging it harder made
        // every one of the twenty-four walk two reaches at a straightness of 0.56, which
        // is the signature of a fixed point that depends on the order.
        //
        // What a simultaneous pass costs is that a body named by several contacts is asked
        // for several full corrections at once; [`Skeleton::velocity_share`] is the mean
        // that answers it, and why both ends of a pair must take the same one.
        //
        // Skipped when there is nothing to solve, which is a skeleton with no contacts of
        // any kind -- a free chain, a pendulum -- so that those pay nothing for it.
        //
        // The read-back sweep runs again between sweeps as well as after the last, because
        // a simultaneous pass is defined by the state it reads: two sweeps over the *same*
        // frozen velocities are one sweep applied twice, which is a relaxation factor and
        // not an iteration.
        if !self.plan_velocity.is_empty() {
            self.share_velocity();
            for _ in 0..VELOCITY_SWEEPS {
                self.solve_pass(Which::Velocity, dt);
                self.read_velocities(dt);
            }
        }

        self.settle(dt, length(gravity));
        // After the solve rather than inside it, because what decides whether a patch is
        // stuck is the Coulomb total the whole step spent, which only exists once the last
        // pass has run. See [`Skeleton::anchor_ground`].
        self.anchor_ground();
        // And for the same reason: which pairs were carrying load is a fact about the
        // whole step. See [`Skeleton::persist_contacts`].
        self.persist_contacts();
        // And for the same reason again, one law over: how hard a body was squeezed is
        // the whole step's normal impulse, which only exists once the velocity pass has
        // finished taking back what it is entitled to.
        self.gather_normal_load(dt);
    }

    /// **One sweep to read both velocities back out of how far everything moved**, for
    /// the same reason the predict is one sweep.
    ///
    /// A method rather than a tail of [`Skeleton::step`] because the step runs it twice:
    /// once to give the velocity pass the state it reads, and once to deliver what that
    /// pass decided. See the module header on the velocity pass.
    fn read_velocities(&mut self, dt: f64) {
        // **Sized by what is awake, not by what exists**, which on the workload sleeping
        // is for is the whole of the difference. The sweep walks the awake set a word at a
        // time and does nothing for a clear bit, so a heap of ten thousand with three
        // body moving in it has one body's worth of work here -- and asking the pool for it
        // cost three forks, at the 26 to 72 us a fork this module has measured elsewhere.
        // Measured on 9,999 capsules resting in stacks of three, one body woken each step,
        // the two predicates alternated over three rounds:
        //
        // ```text
        //   by body count    709.2  734.0  732.5 us a step
        //   by awake count   468.0  463.3  464.6 us a step
        // ```
        //
        // Disjoint every round, and a third of the step gone. On `pile`, where everything
        // is awake, the same build reads 6.23 and 6.60 ms against a 6.0 to 6.4 baseline --
        // no signal, which is what the paragraph below predicts.
        //
        // With everything awake the two predicates are the same predicate, so nothing that
        // was parallel stops being parallel. `BitSet::count` is a popcount per sixty-four
        // bodies, which is the sweep's own loop again and did not show.
        let wide = self.awake.count() >= PARALLEL_FLOOR;
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
    }
    /// The four body arrays every correction writes, as the disjoint-scatter view a
    /// colour is applied through. See [`scatter`] for why, and for the safety argument.
    #[inline]
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
        self.plan_velocity.clear();
        self.plan_work = 0;
        self.plan_near_work = 0;
        self.plan_velocity_work = 0;
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
        // **Every normal half, and then every tangential half.** The two groups are
        // built by the same loop run twice rather than interleaved, because that ordering
        // is the point: see the module header. Nothing else about the stage list changes
        // -- the colours are the same colours, so a lane's slice of one is still disjoint
        // in exactly the way [`scatter`] requires.
        for half in [Half::Normal, Half::Friction] {
            for colour in 0..self.contact_colours.len() {
                let live = self.contact_colours[colour].len();
                let near = self.contact_colours_near[colour];
                if live > 0 {
                    self.plan.push(Stage::Contacts {
                        colour: colour as u32,
                        upto: live as u32,
                        half,
                    });
                    self.plan_work += live;
                }
                if near > 0 {
                    self.plan_near.push(Stage::Contacts {
                        colour: colour as u32,
                        upto: near as u32,
                        half,
                    });
                    self.plan_near_work += near;
                }
            }
            if !self.contact_overflow.is_empty() {
                self.plan.push(Stage::Overflow { half });
                self.plan_near.push(Stage::Overflow { half });
            }
            // The ground is never reduced and never dropped: there is one contact per
            // body, it is the cheapest constraint in the step, and the artefact of
            // under-solving one is a body sinking through the floor, which is the one no
            // distance excuses.
            if !self.ground_colours.is_empty() {
                self.plan.push(Stage::Ground { half });
                self.plan_near.push(Stage::Ground { half });
                self.plan_work += self.ground_colours.len();
                self.plan_near_work += self.ground_colours.len();
            }
        }

        // **And the velocity pass's own plan**, which is the same list twice more: every
        // contact's velocity-level normal half, then every contact's velocity-level
        // tangential half.
        //
        // **No joints**, and that was measured rather than assumed. A joint has an obvious
        // velocity constraint -- the two anchors are one point, so they have one velocity
        // -- and solving the contacts without it looks over-determined, since a contact's
        // velocity correction leaves the joint's violated. It is much worse: twenty-three
        // draws of twenty-four travel, the worst eighteen reaches. The reason is that the
        // velocity of a material point read back from a position-based step is
        // `(position - prev_position) / dt` plus `w x r`, and for a body that has turned
        // over the step those two do not add up to where the point actually went. The
        // error is second order in the turn, it is not zero for a rigid articulated
        // motion, and a bilateral constraint hammers it every step. A contact's normal and
        // tangent tolerate it because they are unilateral and bounded by what the
        // positional solve spent; a joint has no such bound.
        //
        // It names the whole of each list rather than a foreground prefix. A background
        // island is given fewer *passes*; this runs once a step whatever the pass count
        // is, so there is nothing in it to reduce and a reduced copy would be the same
        // list.
        for half in [Half::VelocityNormal, Half::VelocityFriction] {
            for colour in 0..self.contact_colours.len() {
                let live = self.contact_colours[colour].len();
                if live > 0 {
                    self.plan_velocity.push(Stage::Contacts {
                        colour: colour as u32,
                        upto: live as u32,
                        half,
                    });
                    self.plan_velocity_work += live;
                }
            }
            if !self.contact_overflow.is_empty() {
                self.plan_velocity.push(Stage::Overflow { half });
            }
            // **The plane's tangential half is not in this pass**, and that is the one
            // asymmetry in it. See [`contacts::solve_ground_friction`]: the ground patch
            // already carries a memory of where it stuck and what it stuck under, and its
            // friction answers this step's slip and the slip earlier steps left behind, at
            // two different authorities. A velocity-level friction on the same patch
            // answers the same slip a second time with no memory, and the two disagree
            // about what the past was. Measured on the seventeen-bone rig at eight passes,
            // adding it takes the draws that go to sleep from nine of sixteen to one,
            // while the drift does not move. The plane is the one surface whose tangential
            // half is already complete.
            if half == Half::VelocityNormal && !self.ground_colours.is_empty() {
                self.plan_velocity.push(Stage::Ground { half });
                self.plan_velocity_work += self.ground_colours.len();
            }
        }
    }

    /// How many constraints name each body this step, as the reciprocal. See
    /// [`Skeleton::velocity_share`].
    ///
    /// **Every constraint that names the body, not only the ones that carried load.** The
    /// narrower rule is the arithmetically exact one -- a contact that spent no normal
    /// impulse makes no velocity correction either, because both velocity halves return on
    /// `spent.normal <= 0` before they read anything, so counting it divides the others by a
    /// constraint that will not answer. It was built and measured and it is worse, which is
    /// worth recording: over twenty-four draws of the seventeen-bone rig the median settled
    /// residual goes from 0.170 of `g dt` to 0.337 -- past the quarter
    /// `a_settled_rig_does_not_sit_on_a_limit_cycle` allows -- two draws travel past the
    /// jostling allowance where none did, and the rig sleeps in 1 draw of 32 against 14.
    /// The reason is the one the plan already gives for keeping joints out of this pass: a
    /// larger velocity correction at a contact leaves the joints of the body it acts on
    /// more violated, and only the next step's positional solve repairs that, charged as
    /// velocity. The mean is deliberately the conservative one.
    fn share_velocity(&mut self) {
        self.velocity_share.clear();
        self.velocity_share.resize(self.position.len(), 0.0);
        for contact in self.contacts.iter() {
            self.velocity_share[contact.a] += 1.0;
            self.velocity_share[contact.b] += 1.0;
        }
        for contact in self.ground_contacts.iter() {
            self.velocity_share[contact.body] += 1.0;
        }
        for share in self.velocity_share.iter_mut() {
            if *share > 0.0 {
                *share = 1.0 / *share;
            }
        }
    }

    /// **One pass over every colour**, handed to the pool once.
    ///
    /// Each lane walks the same list of stages and takes its own slice of each, waiting
    /// at a barrier before the next -- so a colour is still finished everywhere before
    /// the next one starts, which is what the colouring requires, but it costs a barrier
    /// rather than a fork. See [`crew`] for the measurement that demanded it and for why
    /// this adds no unsafety to what [`scatter`] already argued.
    fn solve_pass(&mut self, which: Which, dt: f64) {
        #[cfg(debug_assertions)]
        self.check_colours_are_disjoint();

        // Read before the field borrows below, because it asks the whole skeleton and
        // those hold parts of it.
        let budget = self.lanes();
        // **Written as direct field borrows rather than through a method**, because each
        // of these views now carries the lifetime of the slice it was built from -- see
        // [`scatter::Cells`]. A method taking `&mut self` would borrow the whole skeleton
        // and the next line could not take another field mutably; three disjoint fields
        // taken by name can. The compiler is checking what was previously a comment.
        let bodies = scatter::Bodies::of(
            &mut self.position,
            &mut self.orientation,
            &mut self.prev_position,
            &mut self.prev_orientation,
        );
        let contact_impulse = scatter::Cells::of(&mut self.contact_impulse);
        let ground_impulse = scatter::Cells::of(&mut self.ground_impulse);

        let (plan, work) = match which {
            Which::Full => (&self.plan, self.plan_work),
            Which::Near => (&self.plan_near, self.plan_near_work),
            Which::Velocity => (&self.plan_velocity, self.plan_velocity_work),
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
        // Read-only, and frozen for the whole of a velocity pass: the read-back sweep
        // wrote them before the pass began and the next one will rewrite them after it.
        // That is what makes the pass simultaneous rather than a second ordered sweep.
        let velocity = &self.velocity;
        let angular_velocity = &self.angular_velocity;
        let velocity_share = &self.velocity_share;

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
        // * **The two halves of a contact are two stages, not one**, and that adds nothing
        //   to this argument: a stage names a colour, the colours are unchanged, and the
        //   tangential stage walks the same list the normal one did. A body named once in
        //   the colour is still addressed by exactly one lane in each, `scatter::disjoint`
        //   still runs over the whole colour, and the barrier between the two halves is the
        //   same barrier that separates any other pair of stages -- which is what makes the
        //   normals' writes visible to the frictions that read them.
        // * **The velocity pass is two more stages of the same kind**, over the same
        //   colours and the same ground set, so it adds nothing to this argument either. It
        //   writes the same two running-impulse arrays the positional halves write, indexed
        //   the same way -- by the constraint, so each is touched by the one lane that owns
        //   it -- and it writes `prev_position` and `prev_orientation`, which are two of the
        //   four [`scatter::Bodies`] already covers. Running it more than once changes
        //   nothing here: a sweep is a pass, and a pass is what this argument is about.
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
                Stage::Contacts { colour, upto, half } => {
                    let set = &contact_colours[colour as usize][..upto as usize];
                    let span = lane.span(set.len());
                    solve_some_contacts(
                        &set[span],
                        half,
                        contacts,
                        &bodies,
                        &contact_impulse,
                        inv_mass,
                        inv_inertia,
                        radius,
                        friction,
                        rolling,
                        dt,
                        velocity,
                        angular_velocity,
                        velocity_share,
                    );
                }
                Stage::Overflow { half } => {
                    if lane.is_only() {
                        solve_some_contacts(
                            overflow,
                            half,
                            contacts,
                            &bodies,
                            &contact_impulse,
                            inv_mass,
                            inv_inertia,
                            radius,
                            friction,
                            rolling,
                            dt,
                            velocity,
                            angular_velocity,
                            velocity_share,
                        );
                    }
                }
                Stage::Ground { half } => {
                    let Some((normal, distance)) = ground else {
                        return;
                    };
                    let set = ground_colours;
                    for &k in &set[lane.span(set.len())] {
                        let contact = ground_contacts[k];
                        let body = bodies.gather(contact.body, inv_mass, inv_inertia, radius);
                        let patch = ground_impulse.get(k);
                        let correction = match half {
                            Half::Normal => {
                                let (correction, totals) =
                                    solve_ground_normal(contact, &body, normal, distance, patch);
                                ground_impulse.set(k, totals);
                                correction
                            }
                            Half::Friction => {
                                let (correction, totals) = solve_ground_friction(
                                    contact,
                                    &body,
                                    friction,
                                    rolling,
                                    normal,
                                    distance,
                                    patch,
                                    ground_stuck
                                        .get(contact.body)
                                        .then(|| ground_anchor[contact.body]),
                                    anchor_reach,
                                );
                                ground_impulse.set(k, totals);
                                correction
                            }
                            // One of each a step, so nothing is carried and nothing is
                            // written back.
                            Half::VelocityNormal => {
                                let (correction, totals) = solve_ground_velocity_normal(
                                    contact,
                                    &body,
                                    normal,
                                    patch,
                                    dt,
                                    velocity[contact.body],
                                    angular_velocity[contact.body],
                                    velocity_share[contact.body],
                                );
                                ground_impulse.set(k, totals);
                                correction
                            }
                            // The plane has no tangential half in the velocity pass, so it
                            // is never given the stage. See [`Skeleton::plan_pass`].
                            Half::VelocityFriction => Correction::none(),
                        };
                        match half {
                            Half::Normal | Half::Friction => {
                                bodies.apply([correction, Correction::none()])
                            }
                            Half::VelocityNormal | Half::VelocityFriction => {
                                bodies.apply_velocity([correction, Correction::none()])
                            }
                        }
                    }
                }
            }
        };

        let lanes = crew::each_stage(plan.len(), work, budget, run);
        self.solved_in_parallel = lanes > 1;
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
            let spent = self.ground_impulse[k].spent;
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

    /// **Which pairs carried normal impulse this step**, so that the narrow phase can give
    /// them a constraint next step whether or not they are still overlapping.
    ///
    /// A pair is kept for exactly one step past the last one it did anything in: this list
    /// is rebuilt from scratch every step out of the contacts that spent normal impulse, so
    /// a constraint that was alive for a whole step and never once found positive depth is
    /// not renewed. A pair flung apart therefore carries an inert constraint for one step
    /// and then loses it, and a pair resting exactly against another keeps one for as long
    /// as it rests. There is no distance in this rule and no margin to choose: what renews
    /// a contact is that it did work.
    ///
    /// The two contacts of a line patch name the same pair, so the list is deduplicated
    /// after sorting -- it is a set of pairs, and the narrow phase asks it one question per
    /// pair.
    fn persist_contacts(&mut self) {
        let mut persisting = std::mem::take(&mut self.persisting);
        persisting.clear();
        for (k, contact) in self.contacts.iter().enumerate() {
            if self.contact_impulse[k].normal > 0.0 {
                persisting.push((contact.a as u32, contact.b as u32));
            }
        }
        persisting.sort_unstable();
        persisting.dedup();
        self.persisting = persisting;
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

    /// **Wakes this body**, and leaves the rest of its island asleep for the broad phase
    /// to reach.
    ///
    /// Its neighbours are woken too, if the disturbance actually reaches them:
    /// [`Skeleton::find_pairs`] wakes what a **moving** body comes within reach of, and
    /// [`Skeleton::wake_touched`] wakes what anything touches. Waking the component instead
    /// is what this used to do, and [`sleep`] holds the measurement that says what it cost.
    /// **Wakes the body's whole island**, for the one caller that has to.
    ///
    /// [`Skeleton::wake`] wakes a body and leaves its island to be reached, because every
    /// ordinary disturbance is something arriving and the broad phase can see it coming --
    /// the whole argument is in [`sleep`]. A retirement is not: the collider simply stops
    /// existing, nothing moves toward what it was holding up, and a stack that has settled
    /// perfectly carries no contacts to lose. Left to the ordinary rules the bodies above
    /// it stay asleep in the air, which is what
    /// `taking_a_body_out_from_under_a_sleeping_stack_drops_it` measures.
    ///
    /// The component is the right unit here for the reason it is the right unit for going
    /// to sleep: a body in a resting stack is still because everything holding it is, so
    /// the set that has to reconsider is the set that was holding it.
    fn wake_island(&mut self, i: usize) {
        let id = self.island_of[i];
        if id != NO_ISLAND {
            let mut members = std::mem::take(&mut self.island_members);
            members.clear();
            self.islands.members(id, &mut members);
            for &m in members.iter() {
                self.wake(m);
            }
            self.island_members = members;
        }
        self.wake(i);
    }

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
            if islands.release(id, i as u32) {
                islands.compact_if_worthwhile();
            }
            island_of[i] = NO_ISLAND;
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
            disturbing,
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
                // **The same test decides what may wake its neighbours.** A body that has
                // moved further than it is allowed to and stay asleep is exactly a body
                // whose neighbours have no business staying asleep either, so the wake
                // front carries no threshold of its own. See [`Skeleton::find_pairs`].
                disturbing.set(i);
            } else {
                disturbing.unset(i);
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
        //
        // **And the component is the right unit here, which is not obvious and was
        // measured the hard way.** Blocking only the body the disturbance reaches, and
        // letting the rest of its component sleep, is the exact mirror of what waking
        // does, and on a settled field with a local disturbance it is worth a great deal:
        // `benches`'s `ploughing` went from 330 of 2,049 awake to 174. It also fails
        // `a_rig_that_does_not_touch_itself_comes_to_rest` and
        // `a_rig_that_touches_itself_comes_to_rest`, with bones coming to rest **below the
        // ground**, at y = -0.0413.
        //
        // That is the thing the component rule is really for, and it is not the one stated
        // above it. A body can be locally still while the structure it belongs to is still
        // being corrected -- held down by a joint whose other end has not resolved, sunk
        // through the plane by a contact the solve has not finished undoing. Sleeping it
        // then does not save work, it *freezes an unconverged state*, and nothing will
        // ever come back to fix it because the body is no longer being solved. Waiting for
        // the whole component is how a body knows the structure around it has converged
        // and not merely that it personally stopped moving.
        //
        // So the cost is real and is paid deliberately: a heap sleeps when its last body
        // does. What that costs on a heap of jointed rigs is written up in this module's
        // header, along with what the rigs are actually doing, which is rocking.
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
                awake,
                ..
            } = self;
            held.clear();
            // **One scan, and two bit tests an edge**, for the pairs that matter: the
            // union-find is the expensive half of this and the tests reject an edge with
            // no random access into anything.
            //
            // An edge with a ready body at one end and a moving one at the other cannot
            // be inside a sleeping island, but it does disqualify the ready side -- a
            // body holding up something that is still moving is not entitled to stop. Its
            // component is not known until every union is in, so the ready end is set
            // aside here and looked up afterwards, which costs a walk over the boundary
            // rather than a second walk over every edge.
            //
            // **"Not ready" is not the same as "still moving", and conflating them left a
            // whole shape of scene permanently awake.** There are three ways to be
            // unready: to be moving, to be *pinned*, or to be *asleep* already. The last
            // two are the stillest things in the simulation -- a pinned body can never
            // move at all, and a sleeping one is not being solved -- so neither can be
            // what is holding a neighbour up. Without the test below, every body jointed
            // to a pinned anchor is disqualified on every step for ever: measured, a limb
            // of three capsules hanging from a pinned root, at rest to the last bit with
            // every velocity reading exactly zero, stayed awake for twelve thousand steps.
            // A sleeping neighbour cannot arise through the broad phase, which wakes
            // whatever it reaches, but it can through a joint the caller has just added,
            // and the same argument covers it.
            let holding = |i: usize| awake.get(i);
            let mut edge = |a: usize, b: usize| match (ready.get(a), ready.get(b)) {
                (true, true) => components.union(a, b),
                (true, false) if holding(b) => held.push(a as u32),
                (false, true) if holding(a) => held.push(b as u32),
                _ => {}
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
    half: Half,
    contacts: &[Contact],
    bodies: &Bodies,
    impulse: &scatter::Cells<Spent>,
    inv_mass: &[f64],
    inv_inertia: &[(f64, f64, f64)],
    radius: &[f64],
    friction: f64,
    rolling: f64,
    dt: f64,
    velocity: &[(f64, f64, f64)],
    angular_velocity: &[(f64, f64, f64)],
    velocity_share: &[f64],
) {
    for &k in set {
        let contact = contacts[k];
        let first = bodies.gather(contact.a, inv_mass, inv_inertia, radius);
        let second = bodies.gather(contact.b, inv_mass, inv_inertia, radius);
        let spent = impulse.get(k);
        let corrections = match half {
            Half::Normal => {
                let (corrections, totals) = solve_contact_normal(contact, &first, &second, spent);
                impulse.set(k, totals);
                corrections
            }
            Half::Friction => {
                let (corrections, totals) =
                    solve_contact_friction(contact, &first, &second, friction, rolling, spent);
                impulse.set(k, totals);
                corrections
            }
            // The velocity pass runs once a step, so there is one of each of these per
            // contact: nothing accumulates, and what the positional solve spent is read
            // rather than added to.
            // Applied through [`scatter::Bodies::apply_velocity`], which is the difference
            // between the two halves above and the two below. They carry a running total
            // like the positional halves, because the pass runs [`VELOCITY_SWEEPS`] times
            // and the sweeps share one step's budget rather than getting one each.
            Half::VelocityNormal => {
                let (corrections, totals) = solve_contact_velocity_normal(
                    contact,
                    &first,
                    &second,
                    spent,
                    dt,
                    (velocity[contact.a], angular_velocity[contact.a]),
                    (velocity[contact.b], angular_velocity[contact.b]),
                    velocity_share[contact.a].min(velocity_share[contact.b]),
                );
                impulse.set(k, totals);
                corrections
            }
            Half::VelocityFriction => {
                let (corrections, totals) = solve_contact_velocity_friction(
                    contact,
                    &first,
                    &second,
                    friction,
                    spent,
                    dt,
                    (velocity[contact.a], angular_velocity[contact.a]),
                    (velocity[contact.b], angular_velocity[contact.b]),
                    velocity_share[contact.a].min(velocity_share[contact.b]),
                );
                impulse.set(k, totals);
                corrections
            }
        };
        match half {
            Half::Normal | Half::Friction => bodies.apply(corrections),
            Half::VelocityNormal | Half::VelocityFriction => bodies.apply_velocity(corrections),
        }
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
            accumulate(&mut out[0], first, ra, impulse, Charge::Moving);
            accumulate(&mut out[1], second, rb, scale(impulse, -1.0), Charge::Moving);
        }
    }

    // -- and, for a ball with a range, the cone -----------------------------------
    //
    // **The same correction the hinge's alignment makes, stopped short of zero.** A hinge
    // drives the angle between its two axes to nothing; a cone drives it to `cone`, and
    // only when it is outside. Inside, the joint is the ball joint it always was and this
    // does nothing at all -- which is why a limb swings freely through its range and stops
    // dead at the edge of it, rather than being pulled toward a rest pose.
    //
    // It shares the hinge's arithmetic deliberately, including the two things that were
    // wrong with it: the correction axis is `cross(world_a, world_b)`, which turns the two
    // *together* where the other order drives them apart, and the angle takes the
    // supplement when `a . b` is negative, because `asin` of the cross product's length
    // cannot tell an angle from its supplement. A limb outside a cone is very often past
    // the quarter turn where that second one bites.
    if let Joint::Ball {
        axis_a,
        axis_b,
        cone,
        ..
    } = joint
    {
        if cone < std::f64::consts::PI {
            let world_a = rotate(first.orientation, axis_a);
            let world_b = rotate(second.orientation, axis_b);
            let across = cross(world_a, world_b);
            let along = dot(world_a, world_b);
            let acute = length(across).clamp(-1.0, 1.0).asin();
            let angle = if along >= 0.0 {
                acute
            } else {
                std::f64::consts::PI - acute
            };
            let excess = angle - cone;
            if excess > 0.0 {
                // **Charged as an ordinary correction, and absorbing it was measured and
                // is worse.** A limit that reads back no velocity is what a ligament does
                // -- it takes the energy out rather than returning it -- and
                // [`share_turn_charged`]'s doc is that, and it does not help,
                // because the settling test measures how far a body *moved* and a free
                // correction moves it just as far; it only stops the move counting as
                // speed. Measured on a lone rig with self-collision, against 254 steps
                // with no cones and 1,987 with these: absorbing never settled at all. See
                // this module's header, under what is left of the pile.
                if let Some(n) = normalized(across) {
                    share_turn(&mut out, first, second, n, excess);
                } else if along < 0.0 {
                    // Exactly opposed, and outside any cone short of a half turn. There is
                    // no cross product to turn about and any perpendicular will do, for the
                    // same reason an inverted hinge is turned back about one.
                    share_turn(
                        &mut out,
                        first,
                        second,
                        perpendicular(world_a),
                        excess,
                    );
                }
            }
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
        // **The two axes are turned onto one another, and both halves of that sentence
        // had to be fixed.**
        //
        // The axis to turn about is `a x b` -- turning `a` the positive way about it
        // carries `a` towards `b`, and [`share_turn`] turns `b` the other way, so the two
        // meet. Written `b x a` the same correction drives them *apart*, and since the
        // cross product vanishes at a half turn as well as at none, the axes then settle
        // **anti-parallel**: the constraint reports itself satisfied while the child is
        // flipped end for end about the joint. Measured on a seventeen-bone rig, every one
        // of its eight hinges started parallel and was inverted within five steps and held
        // there for the rest of the run.
        //
        // The angle is the whole angle and not `asin|a x b|`, because the sine alone
        // cannot tell an angle from its supplement: past a quarter turn `asin` reports the
        // supplement, so the correction *shrinks* as the error grows and a hinge crawls
        // the last quarter turn instead of covering it. The reversed axis is what made the
        // half turn an attractor; this is what made it a slow one to leave. The sign of
        // `a . b` says which side of the quarter turn the pair is on, which is the whole of
        // what `asin` cannot see -- so the repair is one compare rather than the dearer
        // `atan2`, and this runs on every hinge on every pass.
        //
        // A hinge that is *exactly* opposed has no cross product to turn about and needs
        // one, or it stays inverted for ever: any axis perpendicular to the two will undo
        // the half turn, and which one is arbitrary because they all leave the same hinge.
        let across = cross(world_a, world_b);
        let along = dot(world_a, world_b);
        let acute = length(across).clamp(-1.0, 1.0).asin();
        let angle = if along >= 0.0 {
            acute
        } else {
            std::f64::consts::PI - acute
        };
        if let Some(n) = normalized(across) {
            share_turn(&mut out, first, second, n, angle);
        } else if along < 0.0 {
            // `perpendicular` already hands back a unit vector, and a zero axis cannot
            // reach here because it has no negative dot product to report.
            share_turn(&mut out, first, second, perpendicular(world_a), std::f64::consts::PI);
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
    // **The reference is taken in `a`'s own frame, and that is the whole of this
    // function's correctness.** It is turned into the world by `a`'s orientation below, so
    // it has to start as a body vector; built from the *world* axis instead it is a world
    // vector being rotated as though it were a body one, and what comes back is not
    // perpendicular to the hinge at all. Writing `c` for `reference . axis_a`, that
    // version returns `atan2((1 - c^2) sin t, (1 - c^2) cos t + c^2)` rather than `t`, so
    // the angle collapses towards zero as `|c|` approaches one -- and since `c` is built
    // from the body's *current* world orientation, the same joint at the same swing reads
    // differently depending on which way the skeleton happens to be facing. Measured, a
    // hinge held at 0.9000 rad read 0.9000, 0.7130 and 0.4640 in three world frames.
    //
    // It survived because `perpendicular` returns `(1, 0, 0)` whenever `|axis.x| < 0.9`,
    // so for the `axis_a = (1, 0, 0)` every fixture in this crate uses, the world version
    // came out orthogonal to the axis by accident and `c` stayed near zero. A hinge whose
    // axis runs along its own bone, or any diagonal, does not get that luck.
    //
    // Taking it in the body frame also removes `perpendicular`'s discontinuity at
    // `|x| = 0.9` from the answer: `axis_a` is a constant of the joint, so whichever
    // perpendicular is picked is the same one on both sides and cancels between `in_a` and
    // `in_b`. `a_hinge_reads_the_same_angle_whichever_way_the_world_faces` is the guard.
    let reference = perpendicular(axis_a);
    let in_a = rotate(a, reference);
    // Carried `axis_a -> axis_b`, the direction that puts it in `b`'s frame. The other way
    // round measures from a reference the second body does not share.
    let in_b = rotate(b, rotate_into(reference, axis_a, axis_b));
    dot(cross(in_a, in_b), axis).atan2(dot(in_b, in_a))
}

/// `w = inv_m + (r x n) . I^-1 (r x n)`: how much a unit impulse along `n` applied at `r`
/// actually moves this body. The denominator of every positional correction.
#[inline]
fn generalised_inverse_mass(body: &Pose, r: (f64, f64, f64), n: (f64, f64, f64)) -> f64 {
    let rn = cross(r, n);
    body.inv_mass + dot(rn, body.world_inv_inertia.apply(rn))
}

/// Fold one impulse at `r` into a body's correction, as one of the three things a
/// correction may be. See [`Charge`] and [`Correction`].
///
/// **The impulse is at the positional scale whichever kind it is**, which is what lets
/// this be one function: a velocity-level impulse `J` moves a body by `J * inv_mass * dt`
/// over the step it acts in, and a positional impulse `J * dt` moves it by exactly the
/// same, so the two arrive here as the same number and the caller converts once where the
/// velocity is read.
fn accumulate(
    into: &mut Correction,
    body: &Pose,
    r: (f64, f64, f64),
    impulse: (f64, f64, f64),
    charge: Charge,
) {
    if !body.movable() {
        return;
    }
    let move_by = scale(impulse, body.inv_mass);
    match charge {
        // `Still` lands in the same fields as `Moving`: see [`Charge`] for why, and
        // [`scatter::Bodies::apply_velocity`] for what is then done with them.
        Charge::Moving | Charge::Still => into.translation = add(into.translation, move_by),
        Charge::Free => into.free_translation = add(into.free_translation, move_by),
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
    match charge {
        Charge::Moving | Charge::Still => {
            into.rotation = renormalized(delta.multiply(&into.rotation))
        }
        Charge::Free => into.free_rotation = renormalized(delta.multiply(&into.free_rotation)),
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
    share_turn_charged(out, a, b, axis, angle, Charge::Moving);
}

/// **A turn that absorbs rather than returns was built here and is not kept.** Charging
/// the correction as [`Charge::Free`] turns the bodies and turns the orientation they came
/// from with them, so the step reads back no angular velocity from the move -- which is
/// what a ligament does at the end of its travel, and the obvious answer to a joint limit
/// that behaves like a spring. It does not work, for a reason that has now been measured
/// five different ways: the settling test asks how far a body *moved*, and a free
/// correction moves it exactly as far. It only stops the move counting as speed. Measured
/// on a lone rig with self-collision, against 254 steps with no cones and 1,987 with
/// ordinary ones, the absorbing version never settled at all. See [`super`]'s header.
///
/// Turns `a` by `+angle` about `axis` and `b` by `-angle`, split by how hard each is to
/// turn about it, into whichever of [`Correction`]'s two channels `charge` names.
fn share_turn_charged(
    out: &mut [Correction; 2],
    a: &Pose,
    b: &Pose,
    axis: (f64, f64, f64),
    angle: f64,
    charge: Charge,
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
    let into = |slot: &mut Correction, turn: Quaternion| match charge {
        Charge::Moving | Charge::Still => {
            slot.rotation = renormalized(turn.multiply(&slot.rotation));
        }
        Charge::Free => {
            slot.free_rotation = renormalized(turn.multiply(&slot.free_rotation));
        }
    };
    if a.inv_inertia != (0.0, 0.0, 0.0) {
        into(
            &mut out[0],
            Quaternion::from_axis_angle(axis, angle * ia / total),
        );
    }
    if b.inv_inertia != (0.0, 0.0, 0.0) {
        into(
            &mut out[1],
            Quaternion::from_axis_angle(axis, -angle * ib / total),
        );
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
