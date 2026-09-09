# rs_physics — repo profile

The architecture + language baseline every reviewer agent reads before it reviews.
Regenerate or refresh with `/profile-repo`.

---

## What this is

A **Rust physics simulation library** (`rs_physics`, v0.2.0, edition 2021, MIT). Rigid-body
dynamics, collision detection, constraints, fluids, thermodynamics, particles. It is a
**library, not an application** — there is no server, no database, no auth, no tenancy, no
network boundary. Reviews written against a web-app hazard model land nowhere here.

Status per the README: *work in progress, not production-ready.* Breaking changes to the
public API are cheap right now; that is a live input to any "should we keep this
compatible?" question.

## Stack

- **Language:** Rust 2021. No `no_std`. No async runtime — concurrency is OS threads +
  channels, not futures.
- **Deps (core):** `rayon` (data parallelism), `crossbeam` (channels), `log` + `env_logger`,
  `rand`, `approx`.
- **Deps (optional, `gpu` feature):** `wgpu` 23, `pollster`, `bytemuck`.
- **Dev:** `criterion` (benches, `harness = false`).
- **Workspace members:** `.` (the library), `rs_physics_wasm` (WASM bindings),
  `bevy_visual_tests` (Bevy 0.15 visual harness, `publish = false`).
- **Numerics:** `f64` everywhere in the core. SI units throughout — metres, kilograms,
  seconds, Kelvin, Pascals. One deliberate `f32` exception in the SIMD low-precision
  Barnes-Hut path.

## Layout

```
src/
  lib.rs                  crate docs, module tree, `prelude`, assert_float_eq
  utils/                  PhysicsConstants, PhysicsError, Vec3, math_helpers, constants
  physics/                core scalar physics (force, accel, energy, momentum, work, power)
  models/                 Object2D/3D, Shape3D, Quaternion, Simplex
  interactions/           collision: GJK/EPA 3D, shape_collisions_3d, CCD, 2D/3D response
  forces/                 gravity, springs, drag generators (2D and 3D)
  constraints/            joints, springs, rope, hinge, fixed, contact; iterative solver
  materials/              material properties, collision response, stress
  rotational_dynamics/    torque, angular momentum, inertia tensors
  fluid_dynamics/         analytical fluid calcs + Eulerian grid sim (2D/3D) + coupling
  thermodynamics/         heat transfer, thermal grids (2D/3D), processes, cycles, phases
  particles/              particle sim, Barnes-Hut N-body (with AVX SIMD paths)
  world/                  the threaded simulation container — see below
  apis/                   easy_physics convenience wrapper
  gpu/                    wgpu compute: nbody, nbody_3d, particle_sim  [feature = "gpu"]
benches/math_helpers.rs   the only criterion bench
bevy_visual_tests/src/bin/  11 runnable visual demos
```

~47k lines of Rust in `src/`, ~806 `#[test]` functions.

## Feature flags

`default = ["constraints", "materials"]`. Also: `fluid_simulation`, `thermodynamics`,
`fluid_dynamics`, `rotational_dynamics`, `particles`, `particles-cosmological`,
`avx512-simd`, `gpu`. `all` turns on everything.

**This is a primary hazard surface.** Ten flags means a large combination space, `#[cfg]`
gates scattered through `lib.rs`'s `prelude` and across modules, and CI that almost
certainly does not build every combination. A change that compiles under `--all-features`
and under the default set can still break `--no-default-features` or any single-flag build.
Check the gates when a diff touches a `#[cfg(feature = ...)]` boundary or a re-export.

## The concurrency model — `src/world/`

The only real concurrency in the crate, and the place to look first for a race.

- `spawn_physics_thread(WorldConfig)` starts a **background simulation thread**.
- That thread advances at a **fixed timestep paced against wall-clock**, deliberately
  decoupled from display refresh rate.
- `PhysicsWorld` (4.4k lines) owns the sim state; the main thread never touches it.
- Communication is **crossbeam channels** carrying `WorldState` snapshots
  (`STATE_CHANNEL_CAPACITY` bounds it).
- `StateBuffer` double-buffers the two most recent snapshots. Renderers must read through
  `PhysicsHandle::get_interpolated_state()`; reading `get_latest_state()` from a render loop
  stutters, and that is documented and intentional.
- `PhysicsHandle` sends `PhysicsCommand`s in; there is no shared mutable state to lock.

Broad phase lives in `physics_world.rs` as a **multi-level uniform grid** (`GridLevel`,
`CellKey = (i32, i32, i32)`), with tests asserting it matches brute force across uniform,
mixed, bimodal, oversized, coincident, and extreme-position cases.

## Error handling and panics — the local convention

The library returns `Result<_, PhysicsError>`. `PhysicsError` (in `utils/errors.rs`) is a
plain enum with ~15 domain variants — `InvalidMass`, `DivisionByZero`, `InvalidVelocity`,
`ObjectsAtSamePosition`, `CalculationError(String)` — implementing `Display` + `Error`. No
`thiserror`, no `anyhow`.

**The convention is upheld well.** A raw grep finds ~657 `unwrap`/`expect`/`panic!` in
`src/`, but nearly all of them are inside inline `#[cfg(test)] mod` blocks:
`physics_world.rs` has **0** before its test module, `thermal_grid_3d.rs` **0**,
`thread.rs` **1**, `solver.rs` **4**. Do not open a finding on the raw count without
checking which side of the `#[cfg(test)]` line it falls on.

There are **78** `is_nan`/`is_finite`/`is_infinite` guards in non-test code — float
validity is treated as a real concern here, not an afterthought.

## `unsafe` — all of it, in three places

1. `particles/particle_interactions_barnes_hut.rs` — `compute_force_simd_avx` and
   `compute_force_simd_avx_low_precision` (`f32`), dispatched behind a runtime feature check.
2. `particles/particle_simulation.rs` — `step_avx`.
3. `world/thread.rs` — Windows `timeBeginPeriod`/`timeEndPeriod` in a
   `TimerResolutionGuard` RAII pair. **This one carries proper `// SAFETY:` notes** naming
   the precondition and the `Drop` that upholds it. It is the template the SIMD sites should
   match.

The SIMD sites are the highest-value `unsafe` to interrogate: target-feature preconditions,
lane counts, and alignment assumptions.

## Testing

- Tests are **colocated** — either `*_tests.rs` beside the module, or inline
  `#[cfg(test)] mod` at the bottom of the file. Both patterns are in use.
- Float comparison goes through `rs_physics::assert_float_eq(a, b, epsilon, msg)` or
  `approx`. Never `==`.
- `physics_world.rs` carries a **`#[cfg(test)]`-only `PhaseTimings` instrumentation struct**
  and a `phase!` macro that compiles to nothing in release. Measurement infrastructure
  exists — use it rather than speculating about cost.
- Only one criterion bench (`benches/math_helpers.rs`) against ~47k lines. Perf claims about
  anything else are currently unmeasured.
- `bevy_visual_tests` is the qualitative test layer: 11 binaries you actually watch.

## Where the danger lives

For the inversion pass. This is a numerics-and-threading codebase, so the hazard classes are
*not* the web ones:

1. **NaN / Inf propagation.** One bad divide poisons a body, then the grid cell, then the
   snapshot, and the sim never recovers. Zero mass, zero-length normals, coincident
   positions, degenerate simplices in GJK.
2. **Solver divergence.** Iterative constraint solving that gains energy instead of losing
   it. A stack that jitters, a rope that whips, a system that explodes on frame 400 and is
   fine on frame 10 — tests that run 10 steps will not see it.
3. **Tunnelling.** Fast bodies through thin geometry. `continuous_collision_detection.rs`
   exists to stop this; check whether the path in question actually routes through it.
4. **Timestep assumptions.** `dt = 0`, a huge `dt` after a stall, an accumulator spiral where
   catching up costs more than the frame budget.
5. **Float determinism.** Rayon changes reduction order; SIMD changes association;
   `f32`/`f64` mixing in the Barnes-Hut path. If anything is meant to be reproducible,
   nothing currently enforces it.
6. **`unsafe` SIMD preconditions.** Runtime feature detection that does not match the
   `target_feature` the function assumes.
7. **Feature-flag combinations.** A `#[cfg]` gate that compiles in the default set and
   breaks in `--no-default-features` or a lone-flag build.
8. **Thread lifecycle.** A panic on the physics thread — does the handle report it, block
   forever, or silently stop stepping? Channel full, channel disconnected, shutdown ordering.
9. **Unit and frame confusion.** SI is the contract; degrees vs radians, local vs world
   space, and per-second vs per-step quantities are where it silently breaks.
10. **Public API breakage.** `lib.rs`'s `prelude` is the compatibility surface. Pre-1.0, so
    breaking it is allowed — but it should be deliberate, not incidental.

## Who uses this

Two distinct populations, and they want opposite things:

- **Rust game/simulation developers integrating the crate.** They read
  `lib.rs`, the `prelude`, and the docs.rs page. They want: SI units stated, `Result` not
  panic, a fixed-timestep story that doesn't fight their render loop, and feature flags that
  don't explode. They do not want to learn the internals to place a ball on a floor.
- **The author, iterating on the solver.** Reaches for `bevy_visual_tests` binaries and
  watches. Wants fast `cargo check`, fast tests, and demos that fail *visibly* rather than
  subtly.

The `bevy_visual_tests` binaries are the only "interface" a human looks at. They are a
**debugging instrument** — judge them on whether they tell the truth about what the solver
is doing, not on whether they look polished.

## Conventions worth not breaking

- SI units, `f64`, no silent unit conversion at API edges.
- `Result<_, PhysicsError>` out of anything fallible; no panics in library paths.
- Tests colocated; floats compared with an epsilon.
- Feature-gated modules re-exported through `prelude` behind matching `#[cfg]`.
- Doc comments with `# Arguments` sections on public functions.
- `// SAFETY:` on every `unsafe` block (aspirational — `world/thread.rs` does it, the SIMD
  sites do not yet).

---

## Learnings

Reviewers append dated one-liners here when a review turns up a load-bearing fact this
baseline lacked.

- 2026-09-03 — Baseline created. `unsafe` is confined to two SIMD sites and one Windows
  timer guard; only the timer guard carries `SAFETY:` notes.
- 2026-09-08 — A second, unnamed population exists: **`C:/dev/ridgeline`**, a lockstep-deterministic
  RTS whose `ridgeline_sim` crate **deliberately cannot depend on rs_physics** (rayon, hand-written
  AVX, and `HashMap` iteration all break bit-exact replay). rs_physics sits on the *presentation*
  side of that boundary — debris, ragdolls, VFX — so API breakage there is cheap, but numeric
  behaviour is not: `ridgeline_sim::brush` copies `Material::dry_vegetation()`'s mass-burning flux
  (0.022) and residue fraction (0.04) by hand, and `ridgeline/tests/combustion_agrees.rs` fails if
  they drift. Changing a material constant here silently fails a test in another repo.
- 2026-09-08 — `C:/dev/ridgeline` **has no repo profile of its own** and therefore no `## Who uses
  this`; a HUD review there had to assume its population (the author iterating, plus an RTS-literate
  playtester in one 35-minute match). Run `/profile-repo` in that repo. Two load-bearing facts a
  reviewer needs there: (1) `Season::for_match` rolls winter in one match in four, and winter paints
  the playfield at albedo (0.82, 0.85, 0.90) — so **every world-space HUD mark must be contrast-judged
  against a near-white ground**, where the client's `ACCENT`, team colours and health bars all measure
  1.1–1.4:1; (2) the client renders at whatever the machine manages while the sim is a fixed 20 Hz,
  so any on-screen duration or rate must name which clock.
- 2026-09-08 — Concept-art pass on `C:/dev/ridgeline` turned up two arithmetic facts its own
  `docs/ref-palette.html` §5 gets wrong, both of which change what gets modelled. (1) The default
  camera sits **38° above the horizon**, so there are **three** px/m scales, not one: 16.05 vertical,
  20.37 horizontal across the camera, **12.54 horizontal along it**. §5's "≥ 6×6 screen px ⇒ a patch
  ≥ 0.36 m" was derived against the vertical scale only; a *top-surface* team panel — the only one
  visible from every heading, since units turn and the camera does not — needs **0.478 m along the
  axis of travel**. Worse, the panel has to be **square**: which of its two axes lands on the
  12.54 direction depends on the unit's heading, and units turn. A 0.48 × 0.30 m deck panel is
  6.1 × 6.0 px facing one way and **9.8 × 3.8 px** facing the other. The rule that survives every
  heading is **vertical ≥ 0.38 × 0.38 m, top deck ≥ 0.48 × 0.48 m**. (2) The six faction material hexes have
  a **hue** spread but almost no **value** spread: `plate:machined` measures **1.30:1** (WEST) and
  **1.28:1** (EAST), under the project's own 1.618 φ threshold for "reads as a separate thing" — and
  since every mechanical unit is built from exactly those two families, **every mech is one flat value
  mass**. Darkening machined to `#26211e` / `#191d20` takes it to 1.83 / 1.64 with no hue change and
  no shader edit. Related: against the *scrub* valley floor (L 0.062) no faction material exceeds
  1.41:1, so units cannot be built to contrast with the ground — only with themselves.
- 2026-09-08 — Ridgeline's Crawler was redesigned mid-review from a grader to a machine that fires a
  **12 m, ~1450 t sphere with backspin** (a billiards draw shot: solid sphere, returns iff
  `ω0·r > 2.5·v0`; road length `L = v0²/(2μg)` has no ω in it, so spin decides *whether* and the
  ground decides *how far*). Three `units.rs`/roster numbers no longer hold at that scale and are
  logged for routing: Crawler **radius 5.50 m** cannot straddle a 12 m ball (needs ~11.0);
  **mass 14 000 kg** is inconsistent with the ball at any density (14 t in an 11 m sphere is
  20 kg/m³, lighter than balsa); and the energy formulas (0.4 kJ/kg capacity, 3.31 kJ/s·m²·r²
  rejection) give refill ∝ m/r², which at capital scale is minutes rather than roster v2's prized
  25–56 s band. Separately confirmed load-bearing for the presentation side:
  `rs_physics::acoustics::Air::speed_of_sound` (`20.05·√T`) is what `ridgeline/src/audio.rs`
  delays every sound by, so a shockwave ring and its report are the same `distance / c` — and
  winter air at 270 K is 329.5 m/s, ~5% slower than summer, which an existing Ridgeline test asserts.
- 2026-09-08 — Two Ridgeline design corrections that reverse earlier specs, both now reflected in
  `docs/concept-*.html`. (1) **Everything is destructible** is a general rule: roster v2's Redoubt
  ("only occupants take damage, you cannot kill the bunker") is withdrawn, so it takes 900 hp and a
  normal health bar. The load-bearing consequence is that occupants must be **ejected** rather than
  killed when it falls — otherwise the concrete (900 effective hp vs kinetic) is strictly a cheaper
  target than the garrison (210 hp ÷ 0.55 Heavy ÷ 0.35 garrison = 1091 effective) and the ×0.35
  bonus is never computed. Ejected, the two routes price at 1110 vs 1091, within 2%. (2) The
  **Sapper and Spotter merge into one unit** (proposed name *Pioneer*); neither exists in
  `ridgeline_sim`, so nothing breaks. Checked rather than assumed: `cards.rs` `SAPPERS` (id 6) and
  `SPOTTERS` (id 60) are both entirely `Scope::All` doctrine cards that never name a unit, so the
  seven-landmarks-vs-six-slots reasoning at `cards.rs:393` is untouched; only `PRODDERS`
  (`Scope::Kind(UnitKind::Spotter)`) needs a rename. New finding filed: a **gunless unit inside
  `SKIRMISH_LINE`'s `LighterThan(80.0)` band takes the +30% speed and pays none of the accuracy
  cost** — true of the old 72 kg Spotter too, never written down, and worth an assertion in
  `loadout::validate`.
- 2026-09-08 — Ridgeline gains a **caster archetype**, and the definition matters more than the units:
  *a caster's supply does nothing unless you are driving it* (no autoattack) — "has a button" cannot
  be the test because on paper every unit in that roster already has one. Two consequences worth
  carrying: (1) **`ridgeline_sim` has no energy system at all** — grep finds no kJ, capacitor or
  ability charge anywhere, so every energy figure in roster-v2 and the concept sheets is paper, and
  a per-unit energy float would have to enter `checksum.rs`; (2) because the crate has no RNG and
  cannot have one, **the only caster that works under lockstep is one whose effects are objects** —
  a thing with a position, a hit-point pool and an expiry — rather than buffs, procs or variable
  durations. That single object type serves the placed charge, the mine and the proposed tier-three
  stake. Also filed against Ridgeline's own roster: the Courser's stated premise ("for being
  somewhere else in time") is false against `units.rs` — `SCOUT_SPEED` 10.0 × `speed_ratio(Raider)`
  1.70 = **17.0 m/s against the Courser's 14.5**, and `can_leap()` matches only the Raider. Its real
  differentiator is `moving_cooldown ×1.15`, the only exemption from the number roster-v2 §5 itself
  uses to close kiting (Skirmisher 1.55, Bulwark 1.80, Sniper 3.00).
- 2026-09-08 — Three more Ridgeline results worth carrying. (1) **Emissive is the only channel with
  headroom left**: the reflectance budget is full (audit measures `Armour` at Y = 0.1902 against a
  0.20 ceiling), and emissive is not reflected light so it is not in that budget. But **no mark of
  any luminance beats today's snow** — 3:1 above L 0.3375 requires L = 1.113, past white — so
  `ref-palette.html` §2's ground fix stops being a supporting change and becomes a *prerequisite*.
  Against a ground capped at 0.20, an emissive mark at **L 0.70 measures exactly 3.00:1**. Bloom then
  cuts the required team-mark surface **6.5×** (0.2304 m² → 0.0352 m²), which kills the team-colour
  justification for re-topologising the Skirmisher — a decal goes on curved geometry. Critical
  distinction to hold: an **emissive surface** is a material term and free; a **point light** is 600
  entries in Bevy's clustered-forward list and is not. (2) `camera.rs:798` foreshortens by the *sine*
  of camera pitch where the projection needs the *cosine* — 27% overstatement; the palette uses cos
  and is right. Every px figure in the Ridgeline design docs is a **1080p** figure and the game opens
  at 1440×860 (×0.796 linear, 1.674× fewer pixels). (3) Ridgeline's fire constants draw a roster line
  nobody designed: `FULL_COVER_SECONDS` 15.3 × `BRUSH_DAMAGE_PER_SECOND` 7.0 = **107 damage**, the
  ceiling of one burning cell — and *every biological unit is under 107 hp while every mechanical one
  except the Siege is over it*. "Strong against light and biological" needs no armour row.
- 2026-09-08 — Four measured corrections against Ridgeline's palette, from rendered 1920×1080
  captures rather than arithmetic. (1) **Vertical projects by cos of camera pitch, depth by sin** —
  I had them swapped, and so did `camera.rs:798`. At 51.87° that makes **vertical the smallest
  scale** (12.58 px/m at 64 m, 15.48 at the 52 m the camera actually uses), not the largest. Team
  panel rule becomes **0.295 wide × 0.477 tall** for a vertical face and **0.374 square** for a deck
  — the old 0.48 square was 64% too strict by area. (2) **Darkening the ground made team colour
  worse**: with `high_colour` at 0.420, WEST renders at L 0.20 against ground 0.19–0.22 = **1.04:1**,
  down from 1.26:1. WEST 0.257 and EAST 0.386 *straddle* any ground budget, so no ground value
  separates from both. The resolution is that **detectability and identification are different jobs**
  — the bracket carries detection, hue carries identification, and hue was never doing detection.
  Emissive is *not* the answer for world marks: a 3.6 m ground ring at L 0.70 is 21× the selection
  ring's luminous ink. (3) `UNDERSTROKE`'s **alpha 0.85 caps every bracketed mark at 2.28:1** against
  a claimed 5.98 and a 3:1 bar; opaque gives 3.1–3.2:1. One constant, nine marks. (4) An acceptance
  test can name a map that cannot produce the condition it tests — `terraces` is a summer map and the
  winter budget test needed `--map=plateau`.
- 2026-09-08 — The crate held **three independent descriptions of air** and nothing could tell
  them apart: `acoustics::Air` (a real T/RH/p state, no density), `Fluid::air()` (frozen
  1.225 kg/m³ + 1.81e-5 Pa·s, doc-commented "20 °C" when 1.225 is the **15 °C** ICAO figure),
  and `Substance::air()` (frozen 1.184 at 25 °C). `PhysicsConstants::air_density` is a fourth
  and is **still frozen at 1.225** — it is a struct field on the hot `interactions_*` drag
  paths, so unifying it is a separate call. Reconciled by moving `Air` into a new ungated
  `src/atmosphere/`, deriving ρ (ideal gas + humidity), μ (Sutherland), k (constant-Prandtl)
  from it, replacing `Fluid::air()` with `Fluid::from_air(&Air)` and adding
  `Substance::from_air`. `atmosphere::tests::every_description_of_air_in_the_crate_agrees_at_every_state`
  is the mechanism that stops them diverging again. Winter air (270 K) is **9.0% denser** than
  summer (293.15 K); the frozen constant made that ratio exactly 1.
- 2026-09-08 — `Air`'s fields are now **private** with a validating `Air::new`; the three
  defensive `.max(1.0)` / `.clamp(0.0,1.0)` guards scattered through `speed_of_sound` and the
  absorption model are gone, because the constructor is the only way in. `Air::new` **cannot**
  catch the Celsius/kelvin confusion that Ridgeline actually shipped (12 K is a legitimate
  state — the crate does combustion at 3000 K), so `Air::from_celsius` / `temperature_celsius`
  exist to take the `+273.15` off the caller. `celsius_to_kelvin` already existed in
  `thermodynamics`, but that module is feature-gated and `acoustics`/`atmosphere` are not,
  which is why the consumer wrote its own `FREEZING_K`.
- 2026-09-08 — **`cargo check --no-default-features` has been broken since before this work**
  (6 × E0432): `lib.rs`'s prelude re-exports `Material`, `calculate_collision_response`,
  `calculate_stress`, `Joint`, `Spring`, `ConstraintSolver`, `IterativeConstraintSolver`
  ungated while `materials`/`constraints` are feature-gated, and `acoustics/surfaces.rs:26`,
  `models/object_2d.rs:2`, `object_3d.rs:2`, `shape_3d.rs:2` import `materials::Material`
  ungated. Verified pre-existing by `git stash`: 6 errors before, 6 after. Every single-flag
  build over `constraints,materials` is clean. This is the feature-parity bug the profile
  predicted and nothing in CI catches it.
- 2026-09-08 — `cargo test --doc --all-features` fails wholesale (~204) with E0460/E0462
  linker errors from the `cdylib` crate-type interacting with `wgpu`; it is an environment
  artefact, not a code defect. Doctests pass under any feature set that excludes `gpu`. Do
  not read an all-features doctest failure as a regression without checking that first.
- 2026-09-08 — Ridgeline depends on rs_physics **by path with no version pin**, so any API
  change here breaks its build in the same commit. `cd C:/dev/ridgeline && cargo check
  --workspace --all-targets` is the real acceptance test for an rs_physics API change, and
  `--all-targets` matters: the consumer's lib and bin can be green while its `#[cfg(test)]`
  code is not.

- 2026-09-08: **Ridgeline `economy.rs` contradicts itself about its own economy.** The module header (lines 17-28) states "There are no resource nodes and no gathering"; forty lines below, `WORKERS_PER_IRON` says "Now that a deposit is a row of piles a worker walks to", and `IRON_PATCHES_PER_BASE 7`, `MINE_SECONDS 3.6`, `DROP_SECONDS 0.4`, `IRON_PILE_R` and a 13 m round trip all exist. The header describes a replaced model. A design pass was built on the header and had to be redone.
- 2026-09-08: **The Assembly already pays a body for a building** (`units.rs:1290`, "paid for twice, once in iron and once in the worker that is not mining"). Bodies-as-currency is not a novel mechanic in this game; faction two extends an existing verb.
- 2026-09-08: **Cross-faction silhouette separation does not come from height.** Every Yield/Assembly pair rendered at true size measures x1.17-x1.32 on height against a 1.618 threshold. Recognition rests entirely on ground contact, limb count and material value. The ground-contact taxonomy is load-bearing, not stylistic.
- 2026-09-08: **The phi 1.618 threshold is being used for two different jobs** — "this is a different unit" and "this unit changed state" — and only the first is what it was derived for. A laden/unladen state read at x1.38 is correct; the same number would be a failure for two units.
- 2026-09-08: The Assembly's main base is named `"Base"` (`building.rs:916`), the only category word in a roster of specific nouns (Depot, Extractor, Works, Foundry, Barracks, Picket, Lance, Redoubt).
- 2026-09-08: **Ridgeline naming rule, learned by getting it wrong.** The roster's register (Bulwark, Redoubt, Anvil, Picket) is *concrete nouns you could point at*. A name that works by punning across senses is clever rather than plain, and **a name that has to be explained before it lands is one syllable of wordplay too far**. Faction two is **The Strain** — genetic strain, lineage, and what a body does under load, each literally true — after *The Yield* was proposed and rejected for being abstract. Tell: the paragraph defending a name has to *argue* rather than *state*.
- 2026-09-08 — **`SphFluid::step` enforces a CFL speed cap and used to do it silently**, and a
  consumer built a whole emission design on top of speeds it deleted. The cap is
  `smoothing_radius / dt * 0.4`; for `SphParams::blood()` (2 cm spacing → h = 0.04) at 240 Hz
  that is **3.84 m/s**, so `spawn` accepted 24 m/s and the next step clamped it with no error,
  no log and no way to ask. Ridgeline's `blood.rs` wrote two populations "an order of magnitude
  apart" (a carried jet at 40% of a 60 m/s round, a slow cloud at 3–8 m/s) and *both* clipped to
  the same 3.84 — measured in the running game, every drop settled within 2.16 m of its wound and
  the distribution did not move between a 24 m/s tool and a 60 m/s rifle. Now exposed as
  `SphFluid::speed_ceiling(dt)`, with the arithmetic in one `cfl_speed_ceiling` shared by the
  enforcement and the accessor, and a test that asserts a measured peak against the advertised
  bound in both directions. **When a solver silently clamps a caller's input, the clamp is part
  of the API whether it is published or not.**
- 2026-09-08 — `SphFluid` now carries `prev_pos` and `interpolated_position(i, alpha)`, because
  the client-side fix for substep quantisation *cannot work*: `drain_settled` compacts with
  `swap_remove`, so a renderer's shadow buffer of previous positions silently misaligns and
  blends one particle's history into another's present. The buffer has to belong to whatever
  performs the removal. Measured symptom, at 144 fps over a 240 Hz solver with a lone drop in
  zero gravity: per-frame drawn displacement varied by exactly **2.0×** reading `position`, and
  **1.000×** interpolated. Cost is below the bench's ±10% run-to-run noise (one
  `extend_from_slice` of `n × 24` bytes per step against a 27-cell neighbour walk), plus 11.5 KB
  resident at Ridgeline's 480-particle pool.
- 2026-09-08 — Ridgeline's "blood lands 30 m from my units" was **not** the SPH solver and not
  air drag. Measured by capture: `corpses.rs` capped a blast's throw at a flat `MAX_THROW = 20.0`
  m/s whose own comment claimed "a body thrown clear across a crater, which is as far as anything
  here should ever go" — 20 m/s at that launch angle with no drag is a **40 m** parabola, and
  `gibs.rs::shed_while_flying` bleeds a piece for its whole flight, so the blood followed the body
  part. Two lessons worth carrying. (1) **Air drag does not rescue a thrown body part and the
  arithmetic says so up front**: linear drag goes as `1/r`, so `air::linear_drag` is 0.9/s for a
  1.5 mm droplet and **0.03/s** for a 13.5 kg torso — thirty times weaker, worth a tenth of the
  flight. Reaching for the module that already computes drag was the obvious fix and it was the
  wrong one. (2) The first repair capped the *increment* while `Corpse::shove` accumulates, so a
  barrage still stacked to 18.6 m/s against a 10.4 m/s per-blast ceiling — **a bound on the delta
  is not a bound on the quantity.** Fixed by deriving the cap from the weapon's own
  `splash_radius` and clamping the post-shove speed. Measured on the identical `--scene=gore`
  capture: farthest blood mark from the body that shed it **29.58 m → 20.49 m**, mean 7.53 → 5.60,
  fastest bleeding piece 19.92 → 10.71 m/s. The residual is honest terrain — a piece going off a
  5 m ledge, landing and skidding.
- 2026-09-09: **Ridgeline energy: a correct table under a false summary.** `roster-v2` published an accurate refill table and the sentence "refill time scales with radius" above it. The scaling reading needs constant density; the roster's density falls 7.6x (153 kg/m3 Worker to 20 at the Crawler), so refill is not monotone in radius anywhere. The true claim is a **bound** — capacity spans 254x, refill 2.2x — and it was already in the document one paragraph later. **A right table under a wrong summary is more dangerous than a wrong table**, because the table lends the summary its credibility.
- 2026-09-09: **Heat rejection splits on chassis and the published law did not.** Mechanical is area (`3.31 kJ/s/m2 x r2`), biological is mass (`CAPACITY_KJ_PER_KG / BIO_REFILL_S x kg`), which makes flesh refill flat at 30 s by *identity* rather than approximation. `roster-v2` applied r2 to everything and `ref-setting.html` said flesh was flat; both were published and disagreed. Fixed in the earlier document, since the later one is what got implemented.
- 2026-09-09: **A pricing rule enforced on half the price list.** `MAX_USES_PER_FILL` is tested against `Ability::price()`, but cards modify prices and are not in that enum — so all five base prices pass and **all four card-modified prices fail**. Cutting Torch 5 uses, Long Legs 4, Marksman Drill 5. The test that exists to prevent this cannot see where it happened. Generalises: **a constraint checked at the source of a value does not cover the things that transform it.**
- 2026-09-09: **Vault settled: 12 kJ, one use from full.** A cooldown was protecting a real property ("enter high ground by leaping or leave it, not both in one engagement") with the wrong instrument. One-use-from-full is an identity (price > half the bar > 11.0 kJ), and 12 kJ re-earns in 16.4 s against the 12 s the cooldown encoded. The old 8 kJ moved onto the Long Legs card, which now buys the round trip rather than being a flat discount. Cost model beats cooldown because sprint and leap now compete for one bar.
- 2026-09-09: **A four-value table that is monotone in the wrong quantity.** The season term (x1.25/1.10/1.00/0.75) was specified as keyed off `Season::moisture_fraction` but is not monotone in moisture — autumn is wetter than summer and wants a higher factor. It is monotone in *temperature*; `Season` had none until `ambient_k()`. **The figures were right and the quantity they were attributed to was wrong**, which is the same defect class as the radius summary above and showed up in the same week.
- 2026-09-09: **Ridgeline grime: the fine noise octave never renders during play.** At gameplay zoom one pixel covers 0.093 m, so `px_per_patch = GRAIN_M / footprint = 0.76`, and `detail = smoothstep(2, 8, 0.76) = 0`. The authored 7 cm dust and 2 cm oil are dead below `GRAIN_FADE_PX`; what renders is the coarse octave at `14/9`, a **0.643 m / 13.1 px** patch — 22-57% of a drawn unit's width. Per-unit grime variation is therefore worth a shader instruction, and the seed must come from sim id, never from unit position.
- 2026-09-09: **Withdrawn, and it was my own claim: the flat bio refill law does not favour The Strain.** I compared Maul (340 kg, bio) against Bulwark (260 kg, mech) and called the 1.90x unearned — different units, different masses, a category error. The real discriminator is **areal density: flesh beats metal iff m/r2 > 248 kg/m2**. The biggest beneficiary is the Assembly's own Bulwark at x1.45, and the Assembly's internal spread is x0.83 (Courser) to x1.86 (Crawler), a 2.2x range inside one faction. The Skirmisher lands on exactly x1.00, which is the pin arriving as a check. **Generalises: comparing two different units measures their difference, not the rule's.**
- 2026-09-09: **`rotational_dynamics::AngularState3D::apply_torque` is not Euler's equation.** It integrates `Δω = I⁻¹τ·Δt` and **omits the `ω × Iω` gyroscopic term entirely**, so a free body handed zero torque does not tumble, wobble, or exhibit the intermediate-axis instability — it holds one axis forever. The name and signature read like the function for rigid-body rotation and it is a torque integrator. Anyone reaching for this module to make debris tumble (Ridgeline just did, in `ridgeline/src/tumble.rs`) has to write the `−ω × Iω` term themselves. It also calls `inertia.apply_inverse_to_torque`, which recomputes the full 3×3 matrix inverse on **every call** — fine once, wrong per-body-per-substep-per-frame. If this is meant to be the crate's rotational integrator, the missing term is a defect; if it is meant to be a torque accumulator, the doc comment should say so, because `# Arguments` naming an inertia tensor implies otherwise.
- 2026-09-09: **The `rotational_dynamics` *module* is not gated by the `rotational_dynamics` *feature*.** `lib.rs:66` declares `pub mod rotational_dynamics;` unconditionally, and `mod.rs` marks `inertia` and `angular` "always available"; only the legacy `rotational_dynamics.rs` submodule sits behind the flag. So `InertiaTensor`, `AngularState3D` and `inertia_3d::*` are reachable from a `--no-default-features` build, and a consumer does **not** need to add the feature to use them — Ridgeline uses `inertia_3d::solid_cuboid` with the feature off. A same-named module and feature that gate differently is a real trip hazard in both directions; worth a sentence in the module docs.
