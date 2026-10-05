# Changelog

Notable changes to `rs_physics`. Versions before 0.3.0 are recorded only in the git log and
`development_log/`.

## Unreleased

- Doctests under `--features all` no longer fail to compile on Windows machines with many
  cores and little commit headroom. rustdoc compiled one doctest per core, each a full rustc
  and link.exe against wgpu at about 0.55 GB, and 36 at once ran a 64 GB machine with no
  paging file out of commit (os error 1455, reported as E0786, E0462 and LNK1102).
  `.cargo/config.toml` now caps doctest compiles at 8 with `rustdocflags`; unit and
  integration tests are unaffected. Config only: no source, API or doctest changes.

## 0.3.8 (2026-10-04)

Package SPH-CONTACT-OWNER: who a drop splashes against. Additive: no signature, type or
result changes, and every existing test stands; the fluid steps bit-identically to 0.3.7.

### Added

- **The contact record.** After a `SphFluid::step_with_solids`, `SphFluid::contacts()`
  lists every particle a solid's contact moved in that step as a `Contact`: the
  particle's index, the solid's index (the indexing of `Settled::on_solid`: capsules
  first, then boxes, in push order) and its approach speed, its velocity relative to the
  surface along the inward normal before the response, in m/s (zero when it was not
  moving into the surface). `SphFluid::contacts_len()` counts them without the walk, and
  equals `SphSolidStats::contacts`. A drop that strikes a solid and bounces or runs off
  never settles there, so `Settled` never named it; now the surface it struck can be
  credited with the strike. A drop resting on a solid is listed every step, at about
  `g dt`.
- One contact a particle a step: the contact resolves a particle against one solid a
  step (the nearest meeting, or the first it starts inside), so a particle touching two
  reports the one it was resolved against.
- The record is cleared at the start of every step: any `step`, a `step_with_solids`
  with no solid in reach, and a step that refuses its `dt` or gravity leave it empty.
  `spawn`, `drain_settled` (which moves the last particle's contact with its
  `swap_remove`) and `clear` keep it at the particle indices.
- Kinematics only. How much a strike wets a surface is the caller's law; the `Contact`
  documentation points at the Weber number, `rho u^2 d / sigma`, as the usual input.
- Structure-of-arrays and always on: two arrays reserved at the fluid's capacity (12
  bytes a particle), written in the parallel move where the contact resolves: one
  write a particle the move visits and one more a contact; no allocation in the step.
- Tests (`sph_contact_tests.rs`): a drop thrown at a capsule reports it with the
  throw's normal part as its approach speed, within 1%, on the step its path reaches the
  inflated surface; a drop passing half a contact radius clear reports nothing; a drop
  at rest on a box reports the box every step at no more than `g dt` (blood, water,
  napalm); the record is cleared by an empty set, a plain step, an unreachable set and
  a refused `dt` or gravity; drops settled on two crates name the index their last
  contact named; a regression that `step`, and `step_with_solids` with an empty set,
  never record; and the record is bit-identical at 1, 3 and 8 threads, its count the
  statistics' every step.
- `examples/sph_contact_record.rs`: three resting pools (1,024, 4,096 and 16,384) with a
  hundred wading capsules each, timed by the solver's phase timers, with a checksum; on
  the 0.3.7 surface only, so the same file measures both sides.

### Measured

`examples/sph_contact_record 400`, release. Each figure is the median of 400 steps, in
microseconds, taken from the solver's own phase timers. The scene is three resting pools
of blood with a hundred capsules wading in each, stepped in turn. "Before" is 0.3.7
(ff26755) with the example added (d8067fe). "After" is this change. There were eight runs,
ordered before, after, after, before, before, after, after, before, and the table gives
the range over each side's four runs. Every run of both builds printed the same checksums
and contact counts.

| particles | contacts a step | move before | move after | step before | step after |
|---|---|---|---|---|---|
| 1,024 | 264 | 252.9 to 255.1 | 254.2 to 256.8 | 672.0 to 675.0 | 677.8 to 683.6 |
| 4,096 | 922 | 289.5 to 298.2 | 287.6 to 293.0 | 1,048.2 to 1,079.9 | 1,049.7 to 1,064.6 |
| 16,384 | 2,198 | 357.1 to 369.6 | 353.0 to 358.6 | 2,250.2 to 2,286.9 | 2,245.9 to 2,272.2 |

At 4k and 16k the two sides overlap in every phase. At 1k the move (where the record is
written) differs by about 1.5 us, inside its run-to-run spread. The whole step's ranges do
not overlap there: the after side is about 6 us (1%) slower. Most of that gap is in phases
whose code this change does not touch. For example, the binning, which is byte-for-byte
the same source, reads 38.8 to 39.0 after against 37.6 to 38.2 before. The 1k gap is
therefore code layout in the rebuilt library, not the record's cost. That cost is one
`u32` write for each particle the move visits, plus one `f64` write for each contact:
about 1,300 writes at 1k.

### Changed

- Two earlier entries named the consumer whose build ran beside a measurement; they now
  say a consumer build, so the changelog names no consumer.

## 0.3.7 (2026-10-03)

Package SPH-SOLIDS-d: the sphere-cast contact, so a sliding drop gets friction every
substep. No signature changes and no new public item; every result involving a solid
changes (under Changed), static solids included. The ground path is untouched.

### Changed

- **The contact casts a sphere of the contact radius.** Each particle's ray is tested
  against each solid inflated by the contact radius, the exact Minkowski sum with the
  sphere: a capsule of radius `R` as the capsule of radius `R + h / 4` on the same axis,
  a box as the rounded box (faces out by `h / 4`, edges quarter cylinders, corners eighth
  spheres). Before, the ray was a point's, extended a contact radius past the substep,
  and the particle was set a contact radius out only after it met the surface, so a
  drop sliding across a solid faster than about `g dt` cast a ray nearly parallel to it
  and met it only once it had sunk the contact radius, about one substep in ten:
  friction and the surface velocity acted that often. Now a particle within a contact
  radius of a surface and moving into it meets it at once, so a drop on a solid meets it
  every substep. The push-out for a particle starting inside a solid stays. A start
  within `1e-9` (of the squared distance) past the inflated surface counts as touching,
  so rounding cannot turn a drop set on the surface into a miss.
- **A contact carries the particle on along the surface.** At the meeting the rest of
  the cast keeps its tangential part, scaled by the friction decay, and the end is set
  back on the inflated surface. Before, the particle stopped at the meeting point, which
  a drop meeting a surface every substep would never leave. Moving the rest of the
  substep at the decayed speed is the implicit friction step for the position as well
  as the velocity, so a drop sliding at `u` stops in exactly `u / rate`: a 2 m/s drop on
  a hull that stops slides 4.5 mm (it slid 14.5 cm), and a drop set on a 1 m/s hull is
  within `g dt` of its speed in four substeps of blood.
- Static solids' results change with both, so the static-scene checksum test
  (`a_static_scene_steps_to_its_recorded_checksum`, renamed from `..._it_had_in_0_3_5`)
  takes `0xa768_fcfb_039f_9def`, with 0.3.5 and 0.3.6's `0x7c0b_de12_850b_324d` recorded in
  its doc comment.

### Fixed

- A drop sliding across a solid gets friction and the surface velocity every substep
  (SPH-SOLIDS-c's known sliding gap), on solids at rest and moving.

### Added

- Tests: a drop on a hull that stops comes to rest within N substeps (derived from the
  friction decay and `g dt`: four for blood at 2 m/s) and slides `u / rate`, under a
  centimetre; a drop set on a 1 m/s hull follows the decay law to its speed every
  substep; a drop thrown down a 30 degree incline and across a yawed box's lid
  decelerates by the decay every substep (blood, water, napalm); and an oracle test of
  the sphere cast, droplets fired at a capsule and at a yawed box's faces, edges and
  corners, met exactly where the path comes within a contact radius and left a contact
  radius off.

## 0.3.6 (2026-10-03)

Package SPH-SOLIDS-c: drops ride moving solids. The solid contact is cast in each solid's
frame. No signature changes; moving solids' results change (under Changed), and static
solids and the ground are bit-identical to 0.3.5.

### Changed

- **A solid's pose is the end of the substep.** Stated in the module's "Solids" section,
  on `SphSolids`, `push_capsule`, `push_box` and `step_with_solids`: a solid is given
  where it is when the substep ends, with the surface velocity that carried it there, so
  it stood `v dt` further back when the substep began. This is what a caller with one
  pose a frame and a velocity has, and what the existing tests and examples already did.
- **The contact casts relative to the surface.** Each particle casts, against each
  solid in its bin, its displacement relative to that solid, `p1 - p0 - v_s dt`, from
  `p0 + v_s dt` (where it started as the solid sees it) to `p1`, against the solid at its
  end pose; `v_s` is the velocity of the surface nearest `p0` (a capsule's blend at
  `p0`'s axis parameter, a box's one velocity). Before, it cast its world displacement
  against the new pose, so a drop carried along a surface moving tangentially faster
  than about `g dt` (0.04 m/s at 240 Hz) cast a ray nearly parallel to it and met it only
  after sinking the whole contact radius, one substep in ten at 1 m/s, and a rising
  surface met a drop only by pushing it out; neither settled. Now a drop moving with the
  surface casts only one substep's gravity sag, straight at it, and settles on it as on a
  solid at rest. A particle a surface sweeps onto casts back along the solid's motion and
  meets its leading face; the push-out runs for a particle inside the solid's start pose.
  Every result involving a moving solid changes; a solid with zero velocity takes the
  world ray itself, the same arithmetic as 0.3.5.
- **A fast solid's reach grows by its travel.** The binning reach is now per solid: the
  longer of the speed ceiling's travel and the solid's own (its fastest surface speed
  times `dt`), plus the contact radius. That is the exact bound on where a particle that
  can meet the solid starts, under the relative cast (a hit lies within the longer of
  the two travels of `p0`, plus the contact radius). A solid slower than the ceiling
  (3.84 m/s for blood and water at 240 Hz) reaches exactly what it did; only a faster
  one pays for a wider box.

### Fixed

- A drop riding a moving solid settles on it (SPH-SOLIDS-b's known gap): the ignored
  `a_drop_riding_a_moving_capsule_settles_with_its_index_and_the_surface_velocity` runs
  and passes at 1 m/s, with a 5 m/s ride, a lift rising at 0.5 m/s and a hull that stops
  beside it.

### Added

- Tests: the 5 m/s ride (at 480 Hz, where blood's ceiling, 7.68 m/s, clears it), a drop
  on a rising lift, a drop on a hull that stops, a 30 m/s blade sweeping a pool (no
  particle left inside or behind its face: the reach growth at work), a static scene
  stepped to the checksum it had on 0.3.5, and an ignored known-gap test for a hull
  faster than the ceiling.
- `examples/sph_solids_binning.rs` prints the move and whole-step medians beside the
  binning, and steps two more scenes, (e) `pool_off` and (f) `pool_wade`, the bench's
  wading pair.

### Known gaps

- **Sliding.** A drop sliding across a surface, moving or at rest, faster than about
  `g dt` relative to it still meets it only every few substeps (the same parallel-ray
  geometry, now in the surface's frame), so friction acts that often. A drop that lands
  at rest on a moving hull takes longer to come up to its speed before its settle time
  starts (measured 0.08 s at 1 m/s, 0.27 s at 5 m/s), and a drop at 2 m/s on a hull that
  stops slides 14.5 cm over 0.15 s rather than a few substeps. Static solids keep this
  behaviour, bit for bit.
- **Faster than the ceiling.** The speed cap is a world-frame limit, applied before the
  contact, so a surface faster than the fluid's ceiling has its riders capped back to the
  ceiling every substep: at 5 m/s under blood's 3.84 m/s at 240 Hz the drop slips back
  along the hull at 1.16 m/s and does not settle. Pinned by
  `a_drop_rides_a_hull_faster_than_the_speed_ceiling` (ignored).

### Measured

`examples/sph_solids_binning 400`, release, medians of 400 steps, microseconds; before is
0.3.5 with the example's new scenes (fd21248), after is this change: five before runs
and three of the final build, interleaved with each other and with probe builds, with a
consumer build's `cargo` and `rustc` running beside them. Ranges over the runs:

| scene | binning before | after | move before | after | step before | after |
|---|---|---|---|---|---|---|
| (a) far, 4,096 | 4.8 to 5.3 | 5.2 to 5.5 | 96 to 102 | 99 to 101 | 722 to 739 | 721 to 744 |
| (b) hull_1k | 3.8 to 4.1 | 3.5 to 3.9 | 90 to 97 | 99 to 100 | 441 to 460 | 458 to 468 |
| (c) hull_16k | 58.8 to 59.4 | 59.6 to 60.5 | 1,359 to 1,380 | 1,364 to 1,372 | 2,660 to 2,712 | 2,675 to 2,707 |
| (d) melee_16k | 90.5 to 92.5 | 91.3 to 93.1 | 1,339 to 1,349 | 1,336 to 1,344 | 2,684 to 2,712 | 2,676 to 2,706 |
| (f) less (e), wading | | | 18.7 to 20.7 | 19.6 to 21.2 | 0.0 to 6.5 | -3.7 to 1.7 |

- (a) is bit-identical (its solids are out of reach), and so is every static scene: the
  checksum test holds 0.3.5's literal.
- (b) is the one scene that moves: every one of its 1,024 particles tests the moving hull
  each substep, and the move pays 4 to 8 us for it (the adjacent before and after pairs),
  about 5 ns a particle-solid test for the surface velocity and the shifted ray, 1 to 2%
  of the step. A probe build that kept the new code but cast the world ray measured the
  same, so it is the per-test work, not the different trajectories.
- The wading shin (1 m/s, under the ceiling) costs what it did: about 20 us in the move
  in both, and nothing measurable on the whole step.
- (c) and (d) are within their spread; their checksums differ from 0.3.5's because their
  solids move.

## 0.3.5 (2026-10-03)

Package SPH-SOLIDS-b: solids binned by the cells they actually reach, and drops that
settle on a solid. Additive apart from one field on `Settled`, under Changed. The fluid
is bit-identical to 0.3.4's in every scene measured.

### Added

- `Settled::on_solid: Option<u32>`: what a drained particle came to rest on, the index of
  a solid in the `SphSolids` handed to the last `step_with_solids` (capsules first, then
  boxes, in push order), or `None` for the ground. A particle a solid's contact set on
  its surface, slower than the settle speed (0.35 m/s) relative to that surface, is
  still on that solid, and drains after `SETTLE_TIME` (0.25 s) still on solids.
  The threshold is the ground's, applied in the surface's frame (the ground is a
  surface at rest), derived in the module's "Solids" section and on the constant. The ground runs after the
  solids and wins a particle that touched both. The count restarts when a particle moves
  between the ground and a solid, not when its solid's index changes: a set refilled
  every frame from the actors in reach shifts indices most frames, and the index
  reported is the one from the step the particle drained after.
- The binning's particle scan (internal): every particle's cell tested against the
  solids' boxes, listed in a block grid over their union whose block size is chosen each
  step by the cost it implies, with a branch-free union test 64 particles to a word.
  The step takes it over the cell walk once the solids' clipped boxes hold more than a
  third of the particle count in cells between them, the measured crossover
  (`binning_crossover`, an ignored perf test in the solids tests).
- `examples/sph_solids_binning.rs`: the binning's median for four scenes and a checksum
  of each stepped fluid, for comparing two builds.

### Changed

- `Settled` has a fourth public field, `on_solid`. Code that only reads `Settled` (as
  `drain_settled`'s callback receives it) is unaffected; a struct literal of `Settled`
  outside this crate needs the field.
- The solid bins hold a cell only for its own particles. Before, every cell in a solid's
  reach whose hash bucket held any particle was binned, including cells whose bucket
  held only another cell's particles, out of the solid's reach. Those entries met
  nothing (the fluid is bit-identical) but cost the binning, its counting sort and a
  contact test for every particle in the bucket: a 4 m hull over 16,384 drops spread over
  20 m wrote 101,114 entries for the 329 cells it reaches, and 13,437 particles ran the
  contact test, 542 now. `SphSolidStats::bin_entries` and `ray_tested` count the exact
  bins.
- Memory: 8 bytes a particle more (the surface a particle's stillness is counted
  against, and the move's per-particle contact record), allocated at capacity; 28 bytes
  a solid in reach and the scan's block grid in the bins, kept at their high-water marks.

### Known gap

- A drop riding a moving solid does not settle on it. The swept contact casts the
  particle's world displacement against the solid at its new pose, so a drop carried
  along a surface moving tangentially faster than about `g dt` (0.04 m/s at 240 Hz) casts
  a ray nearly parallel to it and meets it only once it has sunk the whole contact
  radius: at 1 m/s, one substep in ten, with a 6.6 mm bump each time. Pinned by
  `a_drop_riding_a_moving_capsule_settles_with_its_index_and_the_surface_velocity`
  (ignored). Casting the displacement relative to the surface fixes it and changes the
  results of every moving solid, so it waits on a ruling.

### Measured

`examples/sph_solids_binning`, release, medians of 400 steps of each scene's binning
(`SphSolidStats::binning`); "before" is 0.3.4 with the example added, and the builds ran
before, after, after, before. A consumer
build's `cargo` and `cargo-nextest` were running beside every run.

| scene | before | after | |
|---|---|---|---|
| (a) 28 solids out of reach, 4,096 packed | 5.0, 4.7 us | 4.4, 4.8 us | unchanged |
| (b) 4 m hull across a 1,024 pool | 3.2, 2.9 us | 3.5, 3.3 us | +0.3 us |
| (c) 4 m hull over 16,384 spread over 20 m | 1,669, 1,667 us | 58.0, 56.8 us | 29x |
| (d) 150 limbs over the same 16,384 | 975, 972 us | 88.9, 88.1 us | 11x |

- **Gate:** (c) and (d) down at least 5x. Met, at 29x and 11x.
- **(b) moves by 0.3 us**, inside its own spread (2.9 to 3.5 us across the four runs);
  the old walk wrote its false entries cheaply and paid for them in the counting sort
  and the move, and the step's grid phase, which includes the binning, is 1.5 us
  shorter after (33 against 35 us).
- Every scene's checksum of positions and velocities is the same before and after.
- The rest of (c) is the min and max over the cells (about 12 us) and the scan reading
  the sorted cells another thread wrote (about 40 us at 16k).

## 0.3.4 (2026-10-03)

Package POOL-GPU-RNG: the GPU pool expands a burst on the device, drawing each particle
with `EffectRng`'s exact sequence, so a seed emits the same particles to the bit on the
CPU pool and the GPU pool. No signature changes; one behaviour change, under Changed.

### Added

- `particles::sin_cos_turn(turn)` and `particles::TURN_BITS`: the sine and cosine of
  `turn / 2^24` of a turn, in integer arithmetic (a quarter-wave table of 2^12 intervals
  built at compile time, integer interpolation, quadrant symmetry by bit masks). The
  same bits on every CPU and in the GPU pool's shader. At most `1.293 * 2^-24` from
  `f64` over all 2^24 inputs; 4.7 ns a call against 21.0 ns for `f32::sin_cos`.
- `EffectRng::jump(steps)`: skip any number of draws. xorshift32 is linear over GF(2), so
  a jump is the product of precomputed matrix powers, byte-sliced (four loads and three
  XORs each, at most 32 of them; 128 KB of tables built on first use).
- `Burst::DRAWS_PER_PARTICLE` (5): the draws `emit` takes a particle, whatever the burst.
- `GpuParticlePool::emit` stages a burst as a 64-byte descriptor (the stream's state,
  origin, class, the three ranges and the lift, and a segment entry) whatever its count,
  and moves the CPU's stream on with `jump`, leaving it exactly where
  `ParticleEffects::emit` would. The placement passes expand it: particle `i` starts at
  the state `5 i` draws in (one byte-sliced matrix per non-zero hex digit of `i`, the
  powers `M^(5 d 16^k)` uploaded once at construction) and takes its five draws with the
  CPU's arithmetic. Products that feed an add are forced to their own rounding (the
  driver fuses them otherwise, and a velocity came out 1 ulp off on the first frame
  without it). The square root and divide are the device's estimate settled by exact
  integer comparisons, with digit-by-digit and long division as the fallback. The sine
  and cosine come from `sin_cos_turn`. `emit_one`, `adopt` and a burst of one stay
  40-byte records.

### Changed

- `EffectRng::hemisphere` takes the azimuth as a 24-bit fraction of a turn and its sine
  and cosine from `sin_cos_turn`, no longer `f32::sin_cos` of `unit() * TAU`.
  - Why: the platform's `sin_cos` is the maths library's. On Windows the UCRT is not
    correctly rounded: 1 ulp off on 43,448 of the 2^24 azimuths, so no shader could
    reproduce it, and glibc and macOS give other last bits again.
  - Effect: the same draws in the same order, so `next_u32` sequences and draw counts are
    unchanged, but a seed's directions differ from 0.3.3's: by up to 4.8e-7 in a
    component of the unit direction (8 units of 2^-24, most of it the old path's own
    rounding of the angle in `unit() * TAU`), 5.5e-7 rad in angle, over 2^22 draws.
    Emission from a seed is now identical on Windows, Linux and macOS and in the GPU
    pool. `examples/grid_checksum`'s particle checksum changes with it.
- `GpuParticlePool`'s frame buffer carries a segment table and payloads after its
  512-byte header: 16 bytes a segment, then 48 for a burst's descriptor or 40 a record.
  It also holds the emission's tables after the emission region, written once: 60 KB a
  hex digit of the largest particle index (240 KB at the default 65,536 a frame) and the
  16 KB sine table. `PoolWrite` and `stage_frame_with` are unchanged in signature; the
  bytes they hand over follow the new layout.
- `GpuParticlePool` docs: an "Emission" section, and "Determinism" now says emission is
  exact. With a power-of-two step, no field and no ground, the integrate is exact too.

### Measured

RTX 3090, Vulkan, wgpu 30, indirect-call validation off; `examples/pool_gpu_bench.rs`, 7
interleaved rounds. No neighbour build was running at the start or the end of the run.
The "before" is the 0.3.3 path, measured in the same process: every particle drawn on
the CPU with `f32::sin_cos` and staged as a 40-byte record. The live sparks-and-dust
pool at 10k, 100k and 1M live is about 240, 2,400 and 24,000 born a frame.

| | 10k | 100k | 1M |
|---|---|---|---|
| staging the emission (frame thread), before | 9.8 us | 91 us | 862 us |
| staging the emission, after | 0.9 us | 1.0 us | **1.6 us** |
| the frame's buffer write, before | 10,168 B | 96,888 B | 964,008 B |
| the frame's buffer write, after | 640 B | 640 B | 640 B |
| `encode` (one `write_buffer` and the passes), before | 163 us | 156 us | 228 us |
| `encode`, after | 154 us | 150 us | 154 us |
| the emit pass (device), before | 7.4 us | 7.5 us | 14.3 us |
| the emit pass, after | 9.9 us | 10.3 us | 17.7 us |

- **Gate:** staging under 50 us at 1M. Met, at 1.6 us.
- **The emit pass grows 2.5 to 3.4 us.** It is a short serial chain per new particle.
  As first written it grew 7.7 us at 100k and 10 us at 1M. Two changes brought it down:
  - Settling the device's square root and divide with six exact comparisons, in place
    of a 25-step digit loop, saved 3.8 and 4.8 us.
  - Taking the jump by hex digit, in place of by bit, saved 1.0 and 1.8 us.
- **Records are staged in place after the header**, as 0.3.3 did, so `emit_one` and
  `adopt` cost what they did.
- **`f32::sin_cos` against `sin_cos_turn`, per call:** 21.0 ns against 4.7 ns. A
  26-step CORDIC measured 42.6 ns, and a 2^11-interval table 4.7 ns.

Bit identity (`src/gpu/particle_pool_emit_tests.rs`, all passing on the RTX 3090):

- **Bursts against the CPU pool:** 28,620 particles born from 9 bursts a frame over 60
  frames, covering:
  - every class and one out of range;
  - lift -0.5 to 3;
  - equal, reversed and negative ranges;
  - counts 0, 1, 2 and up.

  16,975 of them retired. Every live particle's position, velocity, remaining and total
  lifetime, size and class equals the CPU pool's to the bit after every frame, and the
  two streams agree after every frame.
- **A burst cut by the frame bound and by the capacity:** 211 a frame, with records
  between the bursts. It equals the same particles staged as records, every frame.
- **The jump:** on the device it equals sequential steps at 464 indices up to 2^24 - 1,
  including every single hex digit at every position. On the CPU the same holds over a
  sweep up to 5,000,000 steps, and a jump by the period is the identity.
- **`sin_cos_turn`:** equal on the device and the CPU at all 2^24 turns, swept in 1.3 to
  1.9 s.
- **Square root and divide:** equal to the CPU's IEEE results on 1.1 million inputs,
  both by the settled estimate and by the fallback loops alone.
- **The contraction guard:** with it removed, the first frame already differs, by 1 ulp
  in a velocity.

## 0.3.3 (2026-10-03)

The particle half of package TURB, and package POOL-GPU half A: the effect pool resident
on the GPU. Additive: `ParticleClass` is unchanged, and every new item sits beside the
old ones.

### Added (POOL-GPU)

- `gpu::GpuParticlePool` (features `gpu` and `particles`): the effect pool's arrays as
  device buffers at capacity (44 bytes a particle: position and life, velocity and
  lifetime, class and size, the free stack; 44 MB at a million), emission staged on the
  CPU and placed by one `write_buffer` and a compute pass a frame into slots from a GPU
  free stack the integrate refills, and one indirect dispatch of the integrate over the
  slots in use. The integrate is `integrate_in_air`'s arithmetic with the air sampled
  every frame from a 3D texture, ground contact against the engine's height texture
  (restitution and the 0.55 tangential friction as `collide_ground_with`) and retirement
  in the same pass. Landings append to a bounded buffer (4,096 a frame by default, 24
  bytes each). The positions, velocities, classes and sizes, the counters and draw
  arguments are exposed as buffers for an indirect or instanced draw with no CPU copy.
  Readback exists only in the `*_blocking` debug and test methods.
- `gpu::FieldFormat`: `rgba16float` with hardware trilinear filtering (the default),
  `rgba32float` filtered (needs `FLOAT32_FILTERABLE`), and `rgba32float` with the CPU's
  fetch in `f32`. `upload_field` and `upload_plume` write a field once a field update,
  from the frame thread.
- `GpuParticlePool::stage_frame_with` and `encode_staged`, `upload_field_with` and
  `upload_plume_with`, and `gpu::PoolWrite`: the same frame and field uploads handed to a
  host's own staging ring as bytes and a destination, in place of the pool's
  `write_buffer` (wgpu allocates a staging buffer for each). The device contents are
  bit-identical to `encode` and `upload_field`'s, which stay the default path.
- The backend is decided once at startup, by whether an adapter exists; there is no
  runtime switch between the CPU and GPU pools and no readback path (documented on
  `GpuParticlePool` and `BackendPolicy`).
- `GpuParticlePool::register` declares the pool to a `BackendPolicy` as a resident GPU
  backend, so `choose` can return `Backend::Gpu`; `adopt` moves a CPU pool's particles
  onto the device in the next frame's write.
- `GpuContext::from_device(device, queue)`, to run on an engine's device (owned clones
  of wgpu 30's reference-counted handles, so no lifetime ties the pool to the engine),
  and `GpuContext::with_features`.
- `examples/pool_gpu_bench.rs` (interleaved CPU and GPU timings, timestamp queries) and
  the `particle_effects/gpu_pool` group in `benches/particle_effects.rs`.

### Changed (POOL-GPU)

- `BackendPolicy`'s GPU seeds are the resident pool's measured costs: 10 us a frame
  fixed and 0.1 ns a particle (were 60 us and 0.4 ns, estimates), which moves the seeded
  resident crossover from about 23,000 particles to about 3,400.

### Measured (POOL-GPU)

RTX 3090, Vulkan, wgpu 30, indirect-call validation off (as the engine's release build
runs), beside another build. The live sparks-and-dust pool with the air at 20 Hz:

| particles | GPU emit pass | GPU integrate pass | integrate a particle | CPU `integrate` | CPU air 10 Hz |
|---|---|---|---|---|---|
| 10k | 6.9 us | 4.5 us | 0.46 ns | 2.76 ns | 5.04 ns |
| 100k | 7.0 us | 10.2 us | 0.104 ns | 2.93 ns | 5.57 ns |
| 1M | 13.9 us | 90.8 us | 0.092 ns | 4.16 ns | 8.17 ns |

GPU frame = 9.7 us + 0.096 ns a particle; it costs less device time than the CPU's
`integrate` from about 3,300 particles and than the 10 Hz air path from about 1,800.

The particle half of TURB, carried into this version:

### Added (TURB)

- `VelocityGrid`, `TurbulenceDrive` and `SwirlField`: curl noise in three octaves at
  2h, 4h and 8h, each octave's rms speed set exactly to Kolmogorov's `U (l / L)^(1/3)`
  and cross-faded over its turnover time `l / u_l`.
- `ParticleEffects::set_swirl` and `swirl`, and `ParticleEffects::integrate_in_air(dt,
  air, refresh)`: drag relaxes a class towards the local air at `swirl * drag`, each
  particle re-sampling the air once per field period, staggered across the frames
  (12 bytes a particle for the samples). A class with `swirl` 0 integrates
  bit-identically to `integrate`.
- `PlumeField` and its reader, frame and worker: a `FluidGrid3D` with both turbulence
  options over a source's region plus the swirl, summed into one `VelocityGrid` and
  published through a lock-free triple buffer from a worker thread.

### Gate (TURB)

`integrate_in_air` against `integrate`, same run, gate 2x: 1.8 to 1.97x with the field
at 10 Hz, 2.2 to 2.6x at 20 Hz (beside another build). Not met at 20 Hz on the CPU;
the 20 Hz field is the GPU pool's (above), whose integrate pass costs 1.04 to 1.19 times
its field-off cost.

## 0.3.2 (2026-10-02)

Turbulence options on the grid fluids (package TURB, first half). Additive: with the new
options at their defaults every grid is bit-identical (`examples/grid_checksum.rs` on
every solver type, pressure solver and wall condition, at a diffusion and viscosity of
1e-4 and of zero).

### Added

- `AdvectionScheme::MacCormack` for `FluidGrid` and `FluidGrid3D` (Selle et al. 2008):
  forward and reverse semi-Lagrangian passes, half the round trip's error added back,
  clamped to the forward pass's source cells. On an inviscid 4 x 4 Taylor-Green array at
  64^2 it keeps 0.80 of the enstrophy over 2 s where first-order advection keeps 0.36.
  Adds 41 to 50% to a 3D step and 24 bytes a cell.
- `VorticityConfinement::MatchNumericalDissipation` (Fedkiw, Stam and Jensen 2001):
  `epsilon h (N x omega)` before the second projection, with `epsilon` derived cell by
  cell from first-order advection's numerical viscosity `h^2 a(1 - a) / (2 dt)` at the
  grid scale, and each step's force capped so it adds no more kinetic energy than that
  step's advection removed. No constant to tune. On the inviscid 2 x 2 array over 2 s:
  enstrophy 0.98 of the start (0.73 without), energy 0.94. Adds 13 to 14% to a 3D step,
  24 bytes a cell in 2D and 32 in 3D.
- `SolverConfig::advection` and `SolverConfig::vorticity_confinement`, with
  `with_advection` and `with_vorticity_confinement`. (`SolverConfig` literals already
  need `..Default::default()`, since 0.3.0.)
- `examples/grid_checksum.rs` (bit checksums for comparing two commits) and
  `examples/grid_options.rs` (interleaved medians of the options).

### Changed

- `FluidGrid::step` and `FluidGrid3D::step` allocate nothing: the step's copies and the
  relaxation projection's buffers moved into the workspace (112 bytes a cell kept in 2D,
  128 in 3D). Bit-identical.
- A Gauss-Seidel or Jacobi diffusion solve at a zero coefficient writes its answer once
  instead of sweeping (an inviscid grid's diffusion sweeps were about a quarter of its
  step). Bit-identical in the checksum scenarios; in principle a `-0.0` the sweeps turned
  into `+0.0` now stays `-0.0`.

## 0.3.1 (2026-10-02)

Liquids react to actors and debris. Additive: `step` is unchanged and bit-identical.

### Added

- `SphSolids`: capsules (two end points, a radius, a surface velocity at each end) and
  boxes yawed about +y (one velocity), filled by the caller each frame.
- `SphFluid::step_with_solids`: each particle casts a short ray along its substep,
  extended by `SphFluid::contact_radius` (half the rest spacing), against the solids
  binned in its cell, and stops on the nearest surface; a particle a solid moved onto is
  pushed out along the nearest normal. The response reuses `restitution` and `friction`
  and adds the surface velocity. Swept, so a droplet at the speed ceiling cannot tunnel
  through a blade. One way: the liquid never pushes a solid. Bit-identical at any thread
  count.
- The same opt-in gives the ground a slope normal from two extra `ground_height` samples
  for particles in contact, so a drop on an incline runs downhill (SPH-F5). Level ground
  is bit-identical to `step`.
- `SphFluid::solid_stats` (`SphSolidStats`): solids, bin entries, particles tested,
  contacts and the binning time of the last step.
- `benches/sph.rs`: `sph/solids`, solids out of reach and a shin wading a pool, each
  beside its plain-step twin in the same run, with phase medians.

## 0.3.0 (2026-09-30)

Master's correctness and performance reviews (GJK/EPA, CCD, joints, particles, SPH, acoustics,
the grid fluids and the thin film) merged onto `articulated-bodies`, with the shallow-water
solver. The reviews are in `docs/reviews/`.

### Breaking

- `Hinge3D` has a new public field, `axial_offset` (m): struct literals need it or
  `..Default::default()`.
- `Hinge3D::angular_damping` is a rate in 1/s, not a per-call factor. The default, 1.2 /s,
  matches the old feel at 60 Hz.
- `FluidGrid::set_solver_config` and `FluidGrid3D::set_solver_config` return
  `Result<(), PhysicsError>`.
- `SolverConfig` has new public fields (`pressure_solver`, `pressure_tolerance`,
  `pressure_max_iterations`, `wall`): struct literals need `..Default::default()`.
- Writes to a grid's ghost ring return `Err` instead of being dropped.
- `SphFluid::new` rejects non-finite parameters, negative stiffness, viscosity or cohesion, and a
  restitution or friction outside `0..=1` (`PhysicsError::InvalidCoefficient`).
- `SphFluid::spawn` refuses a non-finite position or velocity (returns `false`), and
  `SphFluid::step` is a no-op for a non-finite `dt` or gravity.
- `Simulation::new` and `Simulation::step` return `Err` on invalid input.

### Behaviour changes

- Constraints: `PhysicsWorld` solves hinges, rope chains and springs once a step (they ran once
  per solver iteration, 8x), world joints and ropes correct velocity, and `RopeChain` XPBD scales
  its correction by inverse mass. Hinges swing at real time and ropes are as stiff as their
  iteration count says.
- SPH: cohesion is symmetric (momentum-conserving; identical at uniform density, different at the
  free surface), and ground friction is a decay rate calibrated at 240 Hz, so a slide no longer
  depends on the substep. Identical at 240 Hz.
- Acoustics: `impedance` uses the P-wave modulus; reflection coefficients move by under 0.006
  for every solid preset except napalm, 0.674 to 0.970. A receding listener is clamped in
  `doppler_ratio`, and `barrier_insertion_db` at 0 Hz is the 5 dB grazing limit.
- Grid fluids: the pressure solve is a warm-started MIC(0) conjugate gradient and flows are
  incompressible; `PressureSolver::Relaxation` restores the old solver bit for bit. 3D viscosity
  and diffusion lost an N-times units error. Glycerin is 1.412 Pa s.
- Thin film: the exact inclined law, 2% slower at a grade of 0.1 and 8% at 0.2; level ground is
  identical.
- EPA orients faces from an interior point, keeps confirmed contacts at the iteration cap, and is
  deterministic; CCD normals follow one convention.
- `ParticleEffects::integrate` records timings under the backend that actually ran.

### Added

- `fluid_dynamics::ShallowWater`: a river, lake and flood solver on a heightfield.
- `particles::analytic`: closed-form flight, landing, bounce and slide for effect particles and
  thrown pieces.
- Benches: `epa` (GJK and EPA per call), `fluid_grid`, `shallow_water`.
