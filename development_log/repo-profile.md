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
