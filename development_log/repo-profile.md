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
- 2026-09-08 — A second population exists: a **lockstep-deterministic game** whose simulation
  crate deliberately cannot depend on rs_physics (rayon, hand-written AVX and `HashMap` iteration
  all break bit-exact replay). rs_physics sits on the *presentation* side of that boundary —
  debris, ragdolls, VFX — so API breakage there is cheap, but numeric behaviour is not: the
  consumer copies `Material::dry_vegetation()`'s mass-burning flux (0.022) and residue fraction
  (0.04) by hand and pins them with a test. Changing a material constant here can silently fail a
  test in another repo.
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
  catch the Celsius/kelvin confusion that a downstream consumer actually shipped (12 K is a legitimate
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
- 2026-09-08 — A downstream consumer depends on rs_physics **by path with no version pin**, so any
  API change here breaks its build in the same commit. Its `cargo check --workspace --all-targets`
  is the real acceptance test for an rs_physics API change, and `--all-targets` matters: the
  consumer's lib and bin can be green while its `#[cfg(test)]` code is not.
- 2026-09-08 — **`SphFluid::step` enforces a CFL speed cap and used to do it silently**, and a
  consumer built a whole emission design on top of speeds it deleted. The cap is
  `smoothing_radius / dt * 0.4`; for `SphParams::blood()` (2 cm spacing → h = 0.04) at 240 Hz
  that is **3.84 m/s**, so `spawn` accepted 24 m/s and the next step clamped it with no error,
  no log and no way to ask. A consumer's blood emitter wrote two populations "an order of magnitude
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
  resident at a 480-particle pool.
- 2026-09-09: **`rotational_dynamics::AngularState3D::apply_torque` is not Euler's equation.** It integrates `Δω = I⁻¹τ·Δt` and **omits the `ω × Iω` gyroscopic term entirely**, so a free body handed zero torque does not tumble, wobble, or exhibit the intermediate-axis instability — it holds one axis forever. The name and signature read like the function for rigid-body rotation and it is a torque integrator. Anyone reaching for this module to make debris tumble (a downstream consumer just did) has to write the `−ω × Iω` term themselves. It also calls `inertia.apply_inverse_to_torque`, which recomputes the full 3×3 matrix inverse on **every call** — fine once, wrong per-body-per-substep-per-frame. If this is meant to be the crate's rotational integrator, the missing term is a defect; if it is meant to be a torque accumulator, the doc comment should say so, because `# Arguments` naming an inertia tensor implies otherwise.
- 2026-09-09: **The `rotational_dynamics` *module* is not gated by the `rotational_dynamics` *feature*.** `lib.rs:66` declares `pub mod rotational_dynamics;` unconditionally, and `mod.rs` marks `inertia` and `angular` "always available"; only the legacy `rotational_dynamics.rs` submodule sits behind the flag. So `InertiaTensor`, `AngularState3D` and `inertia_3d::*` are reachable from a `--no-default-features` build, and a consumer does **not** need to add the feature to use them — a downstream consumer uses `inertia_3d::solid_cuboid` with the feature off. A same-named module and feature that gate differently is a real trip hazard in both directions; worth a sentence in the module docs.
- 2026-09-09: **`fluid_dynamics::puddle_depth` is the fourth blood constant a downstream consumer has had
  to hand-copy**, and the copies now outnumber anything a version pin could protect.
  The consumer already carries `BLOOD_YIELD_STRESS`, `BLOOD_DENSITY`,
  `BLOOD_VISCOSITY` and `BLOOD_SURFACE_TENSION` as literals with pin-tests, because they live
  on branch **`thin-film-blood`** and the workspace resolves `rs_physics` **by path to a
  checkout on `master`**, which has none of them. A blood-retention fix there needed
  `puddle_depth(σ, ρ, g, θ)` too and copied it the same way — plus `Surface::contact_angle`,
  which `rs_physics` deliberately does *not* carry a table of ("the substrate decides it, not
  the liquid") while its own doc quotes 80° for soil, so the consumer has taken the doc
  comment as the source. Two consequences: (1) **merging `thin-film-blood` is now the cheapest
  thing on this list**, since every day it stays on a branch adds another literal; (2) the
  doc-comment figure is being *used as data* and should be a named constant or the sentence
  should stop quoting a number.
- 2026-09-09: **A result `thin_film.rs` does not have and should**: `FilmFlow::arrest_thickness`
  is the *yield-stress* criterion and is three orders of magnitude too small to arrest anything
  on a real slope — 0.005 Pa over ρg is **0.5 µm**, against the 380 µm film a consumer actually
  draws. What holds a film on an incline is the contact line, and the force balance is
  one line: retention force per unit width is `σ(1−cosθ) = ½ρg h_p²` — *exactly* the puddle
  thrust, so `puddle_depth` already contains it and no separate hysteresis number is needed —
  set against the wall shear `ρ g h S` of a film of depth `h`, giving
  **`head across one cell ≤ h_p²/(2h)`**. ρ, g and the cell size all cancel, so it is
  mesh-independent, which is the check that it is a force balance rather than a fitted curve.
  The yield criterion by contrast is on the *gradient*, so its head form does carry the cell
  size — one is a body stress, the other a line force, and they scale differently by
  construction. Measured against blood on soil (θ = 80°, h_p = 2.98 mm) it is 46× the yield
  term at every depth. This belongs next to `puddle_depth` as `capillary_retention_head` or
  similar; the consumer derived it because the crate did not have it.
- 2026-09-09: **A threshold below one quantum of its own quantised input is not a threshold.**
  The general form of the bug this fixed, and worth carrying because it is invisible to
  inspection: a consumer's stain shader compared a head threshold of 24 display units against a terrain
  fall baked as an `i8` whose **one unit is 51 display units**. Every representable slope beat
  it and the flat did not, so a tunable "cohesion" was in fact the boolean `fall != 0` — a
  property of the terrain with no liquid in it — and the texels that arrested formed a
  **contour line**, which is what the user reported as "banks along a hard edge". Both numbers
  were individually defensible and neither was ever compared to the other. Whenever a constant
  is set against a quantised quantity, the ratio of the constant to the quantum is the first
  thing to compute.
- 2026-09-09: **`Fluid`'s fields are `pub`, so `Fluid::new`'s validation is optional.** `Fluid { density: -1.0, viscosity: f64::NAN }` is a legal value that the constructor's `validate_positive` never sees, and every consumer that takes `&Fluid` and divides by `viscosity` inherits the hole. `FilmFlow::new` re-validates rather than trusting, which is what lets everything downstream of it be a total function; anything else consuming `&Fluid` on a hot path should do the same or say why not. Same shape as `SphParams`, whose fields are also public and whose `with_spacing` docs already warn that setting them inconsistently fails silently.
- 2026-09-09: **Whole-map per-cell sweeps in this crate are frame-sized, and now there is a number.** A `FilmGrid::step` over 1400 × 1000 cells — a game's stain buffer at 5 texels/m over a 280 × 200 m map — costs **16.8 ms** in release, and the whole-grid CFL scan `FilmGrid::max_step` costs **5.2 ms**, at 6–12 ns a cell with 64% of the arithmetic packed. The arithmetic is not what needs fixing; the cell count is. Any grid API this crate offers a map-scale consumer has to leave the *sweep* to the caller (who knows which tiles are wet) and own only the *law*. A consumer reached the same conclusion independently and empirically — its 32-texel active-set tiles exist because a full sweep cost 14 ms.
- 2026-09-09: **A batch entry point over slices is worth 3.3–4.4× here, and the reason is not the obvious one.** `FilmFlow::flux_batch` runs at 0.72 ns/face against 3.2 ns for the identical loop written by hand around the `#[inline]` scalar `flux`. Hoisting the fluid-dependent branch out with a const generic is only 26% of it; the rest is re-slicing every input to one known length, which is what lets LLVM discharge the bounds checks and vectorize at all. **Caution for anyone benchmarking this class of change:** a `black_box` around each *element* inside the loop is a barrier per element and suppresses the vectorization under test — it inflated this same comparison to a false 7.7×. `black_box` the slices, once, outside.
- 2026-09-09: **Stock `cargo build --release` gives this crate SSE2, not AVX — and forcing AVX made the film grid *slower*.** Baseline `x86-64` has no `%ymm`, so `FilmGrid::step` ships 2-wide: 66% of its double-precision ops packed, every one on `%xmm`. Rebuilt with `-C target-cpu=native` it goes 4-wide (154 `%ymm` ops) and regresses **+38% at 512×512, +27% at 1024×1024, +18% on a 1400×1000 map**; only a 64×64 grid improves, non-significantly. Cause is in the shape: the penalty is absent when the grid fits cache and severe when it does not, and the two builds plateau at *different* rates (92 vs 71 Melem/s), so this is not a shared bandwidth ceiling — it is AVX's own access pattern, most likely `Vec`'s 16-byte alignment making 32-byte loads straddle cache lines. **Generalises: before hand-rolling SIMD anywhere in this crate, check whether the loop is memory-bound; wider vectors can lose.** A per-cell scaling sweep from 164 KB to 42 MB of working set costs nothing to write and answers it.
- 2026-09-09: **Beware quoting an asm count without naming the build it came from.** I measured `FilmGrid::step` under `target-cpu=native`, wrote "108 ops on `%ymm`" into the module docs, and it was wrong about the crate as shipped — which has zero. An instruction count is only meaningful with its `RUSTFLAGS` attached. Same failure mode as the `black_box`-per-element benchmark logged above: the measurement was real, the thing it measured was not the thing being documented.
- 2026-09-09: **`continuous_collision_detection.rs:2006` had a dead comparison that also blocked `cargo clippy` crate-wide.** `const SUB_STEPS: usize = 1` with `if step < SUB_STEPS - 1` is `step < 0` on an unsigned type — never true, so the damping clause reduced to its gravity test alone and the comment above it described behaviour the code had lost. It trips `clippy::absurd_extreme_comparisons`, which is deny-by-default, so *every* clippy run on the crate failed at it. Fixed to `step + 1 < SUB_STEPS`, which is behaviour-identical at 1 and cannot underflow. Worth grepping for other `CONST - 1` on `usize` constants.
- 2026-09-09: **Resolved, and the fix was structural rather than a missing line.** `RigidBodyRotation` (`rotational_dynamics/rigid_body_rotation.rs`) now owns `ω` and is the only thing in the crate that moves it forward in time, so the gyroscopic term cannot be omitted or double-applied. **Adding the term to `AngularState3D::apply_torque` would have been wrong**: `apply_torque` is an *accumulator* called once per torque, and `ω × Iω` belongs to advancing time once — a caller with three torques would have got it three times. It is deprecated instead, not changed. **The inverse recompute was real** and caching it measures 2.0-2.1x on identical arithmetic (`benches/rigid_body_rotation.rs`).
- 2026-09-09: **A physical inertia tensor's principal moments obey `I₁ + I₂ ≥ I₃`, and that inequality is load-bearing for any substep rule.** In principal axes `ω̇ᵢ = (Iⱼ − I_k)/Iᵢ · ωⱼω_k`; the triangle inequality is exactly what makes that coefficient ≤ 1, hence `|ω̇| ≲ |ω|²`, hence "substep so `|ω|·h ≤ 0.25`" is a derivation rather than a tuned constant. It is checkable with no eigendecomposition: it holds iff `(tr I / 2)·Id − I` is positive semi-definite. Note that `InertiaTensor::diagonal_only(2, 4, 8)` — used in this crate's own older tests — violates it, and `inertia_3d::thin_rod_center` returns a singular tensor by design. Anything integrating a caller-supplied tensor must check both.
- 2026-09-09: **`cargo check --lib --no-default-features` does not compile, and has not for a while.** `lib.rs`'s `prelude` re-exports `crate::materials::{Material, ...}` and `crate::constraints::{Joint, ...}` **ungated** while the module contents are behind the `materials` / `constraints` default features — 6 unresolved-import errors. This is the prelude gate-parity hazard the profile already warns about, landing for real. Unrelated to and pre-dating the rotational work; `--all-features` and the default set are both clean.
- 2026-09-09: **Two agents in one working tree will lose each other's work.** Two sessions ran in `C:/dev/rs_physics` simultaneously; `HEAD` was switched between branches mid-task twice, one agent's `git add -A` swept up the other's uncommitted file, and a checkout reverted six files that had already been committed elsewhere. Nothing was ultimately lost, but only because the commits existed. If a second agent may be active, **commit after every green test run**, add explicit paths rather than `-A`, and prefer `git worktree add` to sharing the tree.
- 2026-09-29 — Five-area correctness/performance review (GJK/EPA, CCD, joints, particles, SPH); full
  findings in `docs/reviews/2026-09-29-correctness-performance.md`. Load-bearing facts for the next
  reviewer:
  - **`PhysicsWorld` does not run CCD at all**: `WorldConfig::enable_ccd` is read nowhere. "Does
    this path route through CCD?" (hazard 3) is answered *no* for everything in the world.
  - **The crate has three normal conventions**. `ContactInfo` and constraint `Contact*` point
    1 → 2; `CcdCollisionResult` points 2 → 1. Check which one a call site assumes.
  - **Anything `solve()`d inside the world's iteration loop must be a pure relaxation step.** A
    constraint that integrates time or applies `F·dt` inside `solve()` runs `constraint_iterations`
    times per step. `WorldConstraint::is_iterative` now gates this.
  - **Per-call coefficients are a recurring defect class here.** Hinge damping, rope-chain bleed and
    SPH ground friction were all per call. Write damping as `exp(−c·dt)` or `1/(1 + c·dt)`.
  - **EPA must orient faces from an interior point, never from the origin.** GJK hands over the
    origin on a face whenever the pair has a mirror plane through the centres.
  - **`--features particles` did not compile off x86_64 until this review.** The SIMD sites are now
    `cfg(target_arch)`-gated; keep new intrinsics behind the same gate.
- 2026-09-29 — Acoustics review (`docs/reviews/2026-09-29-acoustics.md`). The ray tracer lives
  downstream and was not reviewed; this covers what it asks `rs_physics` for.
  - **The ISO 9613-1 absorption model is exact**: it matches an independent implementation to
    1e-15 and all 48 values of ISO 9613-2 Table 2. Do not re-derive it; tighten tests against the
    table instead.
  - **No preset is acoustically soft.** Reflection comes from the bulk-impedance mismatch alone, so
    every solid reflects >99.7% of pressure, including `dry_vegetation`. Porous absorption (flow
    resistivity, Miki/Delany–Bazley) is the missing term, and it is an API decision (ACU-1).
  - **`barrier_insertion_db` has no lit side.** It steps 0 → 5 dB at the shadow boundary. Fixing it
    needs a signed path difference and a matching caller change downstream (ACU-2).
  - **`impedance` is the P-wave (bulk) impedance** as of this review, not `sqrt(Eρ)`.
- 2026-09-30 — Analytic fluids + thin-film review (tests in
  `fluid_dynamics/analytic_regression_tests.rs`, IDs FLD-n / FILM-n).
  - **The film law is a diffusion as well as a wave, and the step limit only knew about the
    wave.** `FilmFlow::max_step` / `FilmGrid::max_step` now return `min(dx/c, dx²/(4·ρgh³/3μ))`.
    On level or gentle ground the old figure was up to ~3000× too large and a ¼-step grew a
    ±15% checkerboard that the positivity limiter then held; positivity is not stability.
  - **`validate_positive(NaN)` used to be `Ok`** (`value <= 0.0` is false for NaN). It and
    `validate_non_negative` now reject NaN, which also tightens `FluidGrid`/`FluidGrid3D`
    constructors (a NaN `dt`, diffusion or viscosity is now an error). Write range checks
    as `!(x > 0.0)` in this crate.
  - **The film solver is right**: Huppert's t^{1/5} pool and t^{1/3} slope current both
    reproduced (first-order in dx), and the Bingham factor matches an integrated profile
    to 1e-10. FILM-2 (exact inclined law, `cos⁴θ`) and FLD-2 (zero speed is `Ok(0.0)`) were
    fixed in a second pass. A claim that a downstream shader mirrors `thin_film` turned out to be
    wrong (owner, on PR #27): **don't claim downstream breakage from a change to `thin_film` or
    the grid solvers without checking what consumers actually call.**
- 2026-09-30 — Eulerian grid review (`FluidGrid`, `FluidGrid3D`; findings GRID-1..13, tests in
  `src/fluid_dynamics/grid_regression_tests.rs`). Load-bearing facts for the next reviewer:
  - **The grids' unit of length is the domain width on every axis**: `h = 1 / width`, velocities in
    widths/s, viscosity and diffusion in widths²/s. Anything that uses `height` or `depth` as `1/h`,
    or the cell count as `1/h²`, is a bug; the 3D solver's viscosity was N× too strong that way.
  - **The outer ring of cells is a ghost layer**, overwritten every step. The fluid is cells
    `1..n-1`; reductions must skip the ring, and `add_density`/`add_velocity` refuse it with `Err`
    (GRID-11; they used to return `Ok` and the write vanished).
  - **Lexicographic Gauss-Seidel on a 5/7-point stencil gives the same iterates in any loop
    nesting**, so the grids are x↔y symmetric to rounding, and the storage is now
    innermost-loop-fastest (column-major in 2D, z-fastest in 3D). Anything that walks the raw
    `Vec` in index order sees a different order than before; go through `get_index`.
  - **The pressure is a warm-started MIC(0) conjugate gradient to a relative tolerance** (GRID-9),
    one warm-start field per projection: a step's two projections solve different problems, and
    sharing one guess erased the warm start's benefit (128², tol 1e-5: 111 iterations per step shared, 31
    per-projection). Step cost now varies with the flow; `get_last_pressure_iterations` reports it.
    `PressureSolver::Relaxation` + Gauss-Seidel reproduces the old solver bit-for-bit. The
    default tolerance is **1e-2**, chosen by measurement after review: it is cheaper per step
    than the old four sweeps and within 0.5% of a converged flow; 1e-4 cost up to 72% more for
    no visible change. Judge a tolerance by the flow it produces, not by the residual.
  - **Walls are free-slip by default and no-slip on request** (`WallCondition`, GRID-10). Either
    way the grids are closed boxes with no through-flow; `ShallowWater` is the river solver.
- 2026-09-30 — `ShallowWater` (`fluid_dynamics/shallow_water.rs`, `fluid_simulation`) is the river,
  lake and flood model: depth-averaged Saint-Venant, HLL plus Audusse hydrostatic reconstruction,
  implicit Manning friction. Report: `docs/reviews/2026-09-30-river-solver.md`.
  - **Its promises are tested exactly, so keep them:** a lake at rest stays still to 1e-10, volume
    is conserved to 1e-12 and metered through `volume_in`/`volume_out`, and results are
    bit-identical across thread counts and the serial path. A change that breaks any of these is a
    defect, not a tolerance to loosen.
  - **Boundaries are where river bugs hide.** A copied (zero-gradient) outlet on a sloping bed built
    an ever-rising backwater, because first-order cell discharges lag face fluxes by about 1% on a
    slope. Test outlets with Manning's normal depth on a slope, never on flat ground.
  - **Steep, thin flow is under-driven** where the bed falls more than the depth per cell (10% deep
    at S = 0.02 on 10 m cells). The cure is finer cells; the test pins first-order convergence.
  - **Performance is memory-bound at 512²** on its 40-byte face buffers; arithmetic tweaks don't
    help. The next step is a fused sweep, then a GPU port.
  - **Its design point is presentation on one worker thread at 20–30 Hz**, not a fixed-rate sim
    tick (owner, on PR #27: the game's simulation never calls rs_physics). `Threading::Serial` keeps it
    on the calling thread; `Auto` uses rayon's *current* pool, so `pool.install` bounds it.
    128² wet costs 2.1 ms per 30 Hz frame on one thread.
