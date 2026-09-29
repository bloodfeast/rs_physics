# Correctness and performance review: GJK/EPA, CCD, joints, particles, SPH

2026-09-29 · base `9c765dc`

This review covers five areas:

1. The GJK/EPA narrow phase
2. Continuous collision detection (CCD)
3. The joint and constraint solvers, including how `PhysicsWorld` steps them
4. The `particles` module (Barnes-Hut, SIMD, particle effects)
5. The SPH fluid and its particle coupling

Each area was reviewed independently against `development_log/repo-profile.md` ("Where the danger lives").

Every finding was checked against an analytic or brute-force oracle, not against the code's own output. The oracles include:

- pendulum periods, free-fall distance, spring frequency
- momentum and energy balances
- exact point-to-OBB and point-to-cylinder distance, and 15-axis SAT for box–box
- an exact swept-sphere TOI
- O(n²) direct-sum gravity, and brute-force neighbour search
- a scalar reference for every SIMD kernel

A finding marked **VERIFIED** has a test that failed on the base commit. Fixed findings keep that test as a regression test. Open findings keep it as an `#[ignore]`d test that pins the defect. Run `cargo test --lib <features> -- --ignored` to see what is still wrong.

## Summary

| Area | Critical | High | Medium | Low | Fixed here | Open |
|---|---:|---:|---:|---:|---:|---:|
| GJK / EPA | 2 | 2 | 3 | 2 | 7 | 2 |
| CCD | 5 | 8 | 4 | 5 | 11 (+1 partial) | 10 |
| Joints / constraints | 3 | 5 | 8 | 4 | 15 (+1 partial) | 4 |
| Particles / Barnes-Hut | 1 | 4 | 6 | 6 | 13 | 4 |
| SPH / coupling | 0 | 1 | 7 | 7 | 11 | 4 |

The five things most likely to hurt someone today:

1. **`PhysicsWorld` never runs CCD** (CCD-C1). `WorldConfig::enable_ccd` and `ccd_velocity_threshold` are read nowhere. A 0.1 m ball at 120 m/s goes straight through a 0.1 m wall with `enable_ccd = true`. The CCD module also cannot simply be switched on: every TOI path except sphere–sphere had a blocking bug.
2. **The world stepped hinges, rope chains and springs 8× per step** (JNT-C1), and **world joints and ropes never corrected velocity** (JNT-C2). A hinge pendulum ran at 8× real time. A bob hanging from a world joint reached 163 m/s and a 2.6 m "1 m" joint within 2000 steps. *Fixed.*
3. **EPA could panic, and was grossly wrong on curved contacts** (GJK-1, GJK-2).
   - A sphere on a cube's body diagonal indexed an empty `Vec`.
   - About 10% of sphere–cylinder contacts came back with 10–40× the true depth and an unrelated normal.

   *Fixed.*
4. **Undefined behaviour in `Simulation::step`** (PRT-F0). The AVX path trusted that five `pub` arrays had equal lengths. Truncating one gave an out-of-bounds write; emptying one gave SIGSEGV. *Fixed.*
5. **`RopeChain` XPBD step was scaled by particle mass** (JNT-C3). A chain of 5 kg particles reached 10¹⁵⁴ m on the first step. *Fixed.*

---

## 1. GJK / EPA (`interactions/gjk_collision_3d.rs`)

GJK's yes/no answer is sound. It had no false negatives or false positives over about 80k sphere/box/cylinder/hull pairs checked against analytic distance, at scales from 1 mm to 100 m. Box–box depths already matched SAT exactly. The defects were in EPA and in support/bounds data.

| ID | Sev | Finding | Status | Test |
|---|---|---|---|---|
| GJK-1 | Critical | EPA panics (`faces[0]` on an empty `Vec`) when GJK hands over a tetrahedron with the origin on an edge, e.g. a sphere centred on a cube's body diagonal. | **Fixed** | `epa_sphere_on_box_body_diagonal_does_not_panic` |
| GJK-2 | Critical | EPA oriented faces by the *sign of their distance from the origin*. GJK routinely delivers the origin on a face (sphere vs anything axisymmetric; sphere in a vertex region), so faces flipped inward and the polytope opened. Result: depth 10–40× too large with a perpendicular normal on 10.2% of sphere–cylinder contacts, and a smaller share on rotated boxes and hulls. Faces are now oriented away from the GJK tetrahedron's centroid. | **Fixed** | `epa_sphere_beside_upright_cylinder_depth_is_analytic`, `sphere_vs_{cylinder,rotated_box,hull_*}_matches_*_oracle` |
| GJK-3 | High | EPA discarded its result at 64 iterations, so contacts GJK had confirmed were dropped on curved or large (100 m) surfaces. It now returns the closest face, which is a lower bound within one refinement step. | **Fixed** | `epa_sphere_on_octahedron_vertex_returns_contact`, `sphere_vs_box_at_100m_scale_always_yields_contact` |
| GJK-4 | High | `BeveledCuboid` support was a threshold heuristic, not a support map. It returned points outside the unbevelled box by up to 0.1·bevel, so a die hovering 5 mm above a table read as contact. It is now the exact rounded-box support. | **Fixed** | `beveled_cuboid_support_never_leaves_its_box_*`, `beveled_die_hovering_above_table_is_not_in_contact` |
| GJK-5 | Medium | `Shape3D::bounding_radius` for a `Polyhedron` was measured from the vertex centroid, but support places vertices relative to the local origin. Off-centre hulls were culled before GJK, so real overlaps were never resolved. | **Fixed** | `world_resolves_overlap_with_off_centre_polyhedron` |
| GJK-6 | Medium (perf) | Every support call normalises, inverts and applies the quaternion twice (~95 ns for a cuboid). A 3×3 rotation built once per query makes GJK 1.6–2.1× faster. | Open | `perf_gjk_epa_per_call` (ignored) |
| GJK-7 | Low / Medium (perf) | EPA built its horizon from a per-call `HashMap`. With tied faces, `RandomState` order made the returned normal nondeterministic between identical calls (127 of 400 symmetric inputs). A `Vec` edge list is deterministic and makes EPA 1.4–2.2× faster. | **Fixed** | `epa_is_deterministic_for_symmetric_cube_overlap` |
| GJK-8 | Low | The world's sphere–sphere branch returned "no contact" for coincident centres instead of picking a fallback normal. | **Fixed** | `world_separates_coincident_spheres` |
| GJK-9 | Low (perf) | GJK allocates a fresh `Vec` on every simplex region change; a fixed `[SupportPoint; 4]` would remove it. EPA's absolute 1e-6 m tolerance keeps curved partners near the iteration cap. | Open (UNVERIFIED magnitude) | — |

Note: `test_drag_depends_on_area_to_mass_ratio` spawned both of its spheres at the origin and only passed because of GJK-8. The beach ball now starts at y = 5.

## 2. Continuous collision detection (`interactions/continuous_collision_detection.rs`)

Sphere–sphere TOI is correct. It matched an independent closed form over 20,000 random cases, with the right root, negative times rejected, and the `[0, dt]` window respected. Everything else had at least one blocking bug.

Before this change, the axis-aligned sphere–plate sweep had these errors:

| Error | Count |
|---|---:|
| Face hits with the wrong TOI and normal | 385 / 2556 |
| Gross false positives | 262 |
| Rotated-plate hits missed | 2559 / 2559 |

Afterwards, face TOIs are exact to 4e-11 s (axis-aligned) and 7e-10 s (rotated), with no face misses.

| ID | Sev | Finding | Status | Test (`ccd_review_repro_tests`) |
|---|---|---|---|---|
| CCD-C1 | Critical | **`PhysicsWorld` never calls CCD.** `enable_ccd` and `ccd_velocity_threshold` are read nowhere, and `resolve_collisions` is discrete only. A 0.1 m ball at 120 m/s passes a 0.1 m wall with CCD "enabled". The fields are now documented as not implemented. Wiring CCD in needs CCD-C5, H2, H3 and H5 fixed first, plus a swept pass in `resolve_collisions`. | Open (documented) | `ccd_repro_world_bullet_tunnels_thin_wall_with_ccd_enabled` (ignored) |
| CCD-C2 | Critical | Sphere vs AABB set `toi = 0` with normal `(±1,0,0)` whenever the sphere was inside the x slab, without testing y or z. A ball dropped on a floor ended at y = −5. Replaced with a slab test. | **Fixed** | `ccd_repro_sphere_dropped_on_floor_*`, `ccd_repro_update_physics_ball_falls_through_floor`, `ccd_repro_sweep_sphere_vs_axis_aligned_plate` |
| CCD-C3 | Critical | The rotated-box face impact was computed as `(r − d) / −v_n`, which is negative for every sphere not already touching. No hit on any non-identity-oriented box was ever found. | **Fixed** | `ccd_repro_rotated_box_face_hit_is_missed`, `ccd_repro_sweep_sphere_vs_rotated_plate` |
| CCD-C4 | Critical | The CCD normal is documented obj2 → obj1, but four paths used obj1 → obj2: cuboid–cuboid, the t = 0 EPA path, conservative advancement's EPA branch, and `resolve_penetrations`. Closing pairs got no impulse, and the penetration pass pushed overlapping spheres from 1.9 m apart to 1.2 m. | **Fixed** | `ccd_repro_cuboid_cuboid_normal_is_reversed`, `ccd_repro_cuboid_bullet_tunnels_wall_*`, `ccd_repro_overlap_at_t0_normal_is_reversed`, `ccd_repro_overlapping_approaching_spheres_get_no_impulse`, `ccd_repro_penetration_pass_pushes_spheres_deeper` |
| CCD-C5 | Critical | Conservative advancement (every rotating box pair, and anything with a Cylinder, BeveledCuboid or Polyhedron) uses the gap between *bounding spheres* as its distance. It declares contact with normal `(1,0,0)` when that gap is below 1 mm. A spinning bullet 5 m above a 10 m floor is "in contact" at t = 0, and an untouched box acquires −0.9 m/s. Needs a GJK closest-distance query. | Open | `ccd_repro_spinning_bullet_*`, `ccd_repro_conservative_advancement_false_positive_*` (ignored) |
| CCD-H1 | High | Bodies were only integrated when there were no CCD pairs. Static-static pairs were only skipped for `mass <= 0`, not the world's `INFINITY`, so two overlapping floor tiles froze every other body permanently. Bodies in no pair are now integrated, and static-static pairs are skipped. A body skipped because its partner was already processed still does not move. | Partly fixed | `ccd_repro_bystander_is_not_integrated_*`, `ccd_repro_overlapping_static_tiles_freeze_the_world` |
| CCD-H2 | High | There is one TOI event per body per step, and the remaining time is integrated blind. A ball rebounding off a wall passes straight through the ball behind it. Needs a TOI loop. | Open | `ccd_repro_rebound_tunnels_through_body_behind` (ignored) |
| CCD-H3 | High | Rotated-box *edge and corner* hits are never found (551 of 2559 remain after CCD-C3). | Open | `ccd_repro_rotated_box_edge_hit_is_missed`, `ccd_repro_sweep_sphere_vs_rotated_plate_edges` (ignored) |
| CCD-H4 | High | `update_physics_with_ccd*` applied gravity to static bodies (mass 0 and `INFINITY`). | **Fixed** | `ccd_repro_update_physics_applies_gravity_to_static_bodies` |
| CCD-H5 | High | Sphere–cuboid ignores the box's angular velocity: a blade spinning at 50 rad/s never hits a ball in its path. | Open | `ccd_repro_spinning_blade_vs_sphere_*` (ignored) |
| CCD-H6 | High | The non-rotating box–box path ignores static orientation, so a wall pitched 90° is tested with its unrotated extents. | Open | `ccd_repro_cuboid_cuboid_ignores_static_orientation` (ignored) |
| CCD-H7 | High | A mass-0 static body has inertia 0, so `(r×n)²/0` gave 0/0 = NaN, which slipped past the `denom < EPSILON` guard and poisoned the ball's velocity. | **Fixed** | `ccd_repro_zero_mass_static_body_poisons_velocity_with_nan` |
| CCD-H8 | High | The "centres approaching" early-out rejected sphere–box hits on long walls. It is now bypassed for the analytic sphere–box and box–box routines. It is kept for pairs that fall through to conservative advancement, where it hides CCD-C5 false positives. | **Fixed** (analytic pairs) | `ccd_repro_centre_approach_early_out_skips_hit_on_long_wall` |
| CCD-M1 | Medium | Restitution was hard-coded to 0.8. It is now the mean of the materials, as in `shape_collisions_3d`. `test_collision_response` now expects the 0.5 default. | **Fixed** | `ccd_repro_response_ignores_material_restitution` |
| CCD-M2 | Medium | Positions were advanced to the TOI but orientations were not, so rotation over `[0, toi]` was lost. | **Fixed** | `ccd_repro_rotation_before_toi_is_dropped` |
| CCD-M3 | Medium | The TOI was documented as a fraction in `[0, 1]` but is seconds in `[0, dt]`. The rotated-box path rejected anything after 1.0 s. | **Fixed** | `ccd_repro_rotated_box_hit_after_one_second_is_dropped` |
| CCD-P1 | Medium (perf) | `update_physics_with_ccd` is O(n²) with no broad phase: 3 full pair sweeps per call, plus up to 3 more in `resolve_penetrations`. Measured 4.9 ms at 250 bodies, 41 ms at 1000, 682 ms at 4000, with zero contacts. | Open | `ccd_perf_update_physics_with_ccd_scaling` (ignored) |
| CCD-L1 | Low | The sphere–sphere "touching" tolerance was compared against `c = |x|² − R²` (m²), so 1 mm spheres 0.2 mm apart were reported in contact. It is now a length. The contact points in that branch had the wrong sign. | **Fixed** | `ccd_repro_small_spheres_reported_touching_across_gap` |
| CCD-L2 | Low | NaN or negative `dt` panicked inside `f64::clamp`. Every public entry point now no-ops. | **Fixed** | `ccd_repro_nan_dt_panics_in_response` |
| CCD-L3 | Low | The AABB path treats the Minkowski sum's edges as square, so hits near edges are slightly early or phantom (within r√3). | Open | swept, not asserted |
| CCD-L4 | Low | The body-frame inertia tensor is combined with world-frame r and n without rotation. | Open (UNVERIFIED) | — |
| CCD-L5 | Low | The `MIN_ADVANCEMENT` floor (1e-4 s) breaks conservative advancement's bound at high speed. Masked by CCD-C5. | Open (UNVERIFIED) | — |

`docs/COLLISION_DETECTION.md` also describes a `binary_search_toi` and signatures that do not exist.

## 3. Joints and constraint solvers (`constraints/`, `world/world_constraints.rs`)

The Jacobians are right: n = (p₂−p₁)/|…|, and static bodies get inverse mass 0. The failures were in how the world drives the constraints, and in several unit and scaling errors that only show up away from 1 kg and 60 Hz.

| ID | Sev | Finding | Status | Test |
|---|---|---|---|---|
| JNT-C1 | Critical | `PhysicsWorld::solve_constraints` called every constraint `constraint_iterations` (8) times per step, including hinges and rope chains (which integrate themselves) and springs (which apply `F·dt`). Measured: hinge period 0.251 s against an analytic 2.007 s; spring period ÷√8; a rope chain moved 7.5× `v·dt`. Now only joints and ropes iterate; the rest run once per step. | **Fixed** | `review_world_hinge_pendulum_period_matches_analytic`, `review_world_spring_period_matches_analytic`, `review_world_rope_chain_moves_v_dt_per_step` |
| JNT-C2 | Critical | World `Joint` and `Rope` corrected position only, so gravity's velocity was never removed. A 1 m joint was 2.64 m long with the bob at 163 m/s after 2000 steps. They now remove the relative normal velocity (ropes only when separating). | **Fixed** | `review_world_{joint,rope}_hanging_bob_stays_bounded` |
| JNT-C3 | Critical | `RopeChain2D/3D::solve_segment` scaled the XPBD Δλ by mass *fraction* `w/Σw` instead of `w`. The step was ∝ particle mass: 10 kg overshot to 0.1 m, 0.1 kg under-corrected, and 5 kg chains diverged immediately. | **Fixed** | `review_rope_chain_projection_is_mass_independent`, `review_rope_chain_heavy_particles_stay_bounded` |
| JNT-H1 | High | The `Joint`/`Fixed` impulse clamp `0.1 / dt` has units of 1/s, not N·s, so it capped heavy bodies to almost nothing. A 1000 kg bob hung from `Joint3D` drifted 612 m; from `Fixed3D`, 979 m. Removed in 1D/2D/3D Joint and Fixed2D/3D. Dead `lambda.min(+x)` clamps in `Rope2D/3D` were removed too. | **Fixed** | `review_joint3d_holds_heavy_bob`, `review_fixed3d_holds_heavy_body` |
| JNT-H2 | High | `Hinge3D` projected the arm onto the swing plane and dropped the axial offset, so a door centred 0.5 m along its hinge line teleported onto the anchor plane. | **Fixed** | `review_hinge_preserves_offset_along_axis` |
| JNT-H3 | High | `Contact3D` friction built the tangent from the *pre-impulse* normal velocity, so friction cancelled the bounce (e = 1, μ = 0.5 head-on gave vy = 0, not 1). `Contact2D` was already right. | **Fixed** | `review_contact3d_friction_does_not_act_along_normal` |
| JNT-H4 | High | The contact normal was documented as object2 → object1, but the code works only with object1 → object2. Following the docs gave no response at all. Three `solver_tests` floor contacts used the documented convention and so did nothing (they only asserted `iterations > 0`). The docs, examples, inline comments and those tests are corrected. | **Fixed** | `review_contact3d_documented_normal_convention_bounces` |
| JNT-H5 | High (design) | `UnifiedSolver2D/3D` constraints own copies of their bodies, so a body shared by two constraints is two independent bodies and chains/ragdolls are uncoupled pairs. There is no accessor to read results back. `Hinge3D` and `Spring*` also integrate inside `solve()`, so one `solve(dt)` advances them `iterations × dt`. | Open | `review_unified_solver_advances_{hinge,spring}_by_one_dt_per_solve` (ignored) |
| JNT-M1 | Medium | The `Rope2D/3D` "KEY FIX" block added `disp·β·0.5/dt` velocity whenever the rope was stretched, even while the ends approached (KE 0.5 → 2.0 J). Its `min(w1/w2, 1)` weighting was not momentum-conserving (0 → −9 kg·m/s). Removed; the existing one-sided impulse is the rope. | **Fixed** | `review_rope3d_does_not_accelerate_approaching_bodies`, `review_rope3d_conserves_momentum_light_object1` |
| JNT-M2 | Medium | `Contact2D/3D` projected the penetration out *and* biased the full penetration into velocity, which launched e = 0 bodies 6 cm off the surface. The bias now covers only what the projection could not remove. | **Fixed** | `review_contact3d_inelastic_contact_does_not_launch_body` |
| JNT-M3 | Medium | The contact position correction was re-applied every iteration, because the stored penetration was never reduced (10 iterations pushed a 5 cm overlap out by 40 cm). | **Fixed** | `review_contact3d_position_correction_not_reapplied_per_iteration` |
| JNT-M4 | Medium | `dt = 0` wrote NaN into static anchors (0·∞) and rope-chain positions (`compliance/dt²` = 0/0). Every `solve` now no-ops at `dt <= 0`. | **Fixed** | `review_joint3d_dt_zero_does_not_nan`, `review_rope_chain_dt_zero_does_not_nan` |
| JNT-M5 | Medium | Hinge damping was `ω *= 1 − c` per call: 3.3 rad/s peak at 60 Hz vs 0.8 at 600 Hz. It is now a rate in 1/s; the default of 1.2 /s matches the old feel at 60 Hz. | **Fixed** | `review_hinge_damping_is_timestep_independent` |
| JNT-M6 | Medium | The world rope chain's "small bleed" was 2% per call (so 16% per step with JNT-C1), giving a terminal velocity near 2 m/s in free fall. It is now 0.05 /s. | **Fixed** | `review_world_rope_chain_free_fall_distance` |
| JNT-M7 | Medium | 1D `Spring` multiplied damping by `signum(dx)`, making it *anti*-damped on the −x side (1.25 J → 9.35 J). | **Fixed** | `review_spring_1d_damping_is_mirror_symmetric` |
| JNT-M8 | Medium | `Hinge3D` never applies a reaction to object1 and overwrites object2's velocity every call. With a dynamic frame, momentum is not conserved. | Open | `review_hinge_applies_reaction_to_dynamic_object1` (ignored) |
| JNT-L1 | Low | `critical_damping()` was `m₁m₂/(m₁+m₂)` = ∞/∞ = NaN with a static anchor. | **Fixed** | `review_spring3d_critical_damping_with_static_anchor` |
| JNT-L2 | Low | Warm starting does nothing: `set_lambda` stores a number no constraint reads. Trajectories with it on and off are bit-identical. | Open | `review_warm_starting_has_an_effect` (ignored) |
| JNT-L3 | Low (perf) | Rope chains did 8× the work and 16 heap allocations per step because of JNT-C1. JNT-C1 removes the 8×. Remaining: repeated SipHash `object_ids` lookups per iteration, and a `HashMap` of constraints walked `iterations` times. | Partly fixed | — |
| JNT-L4 | Low | Gauss-Seidel order follows `HashMap` iteration, so coupled constraints solve in a different order each process run. | Open (UNVERIFIED) | — |

Behaviour changes to expect in the visual demos:

- Rope chains are now as stiff as their iteration count says. They used to be soft for the 0.08 kg particles the demos use.
- Hinges swing at real time.
- `Hinge3D::angular_damping` is now per second.

## 4. Particles (`particles/`)

What checked out:

- **θ = 0:** the Barnes-Hut walk matches an O(n²) direct sum to 1e-12.
- **θ = 0.5:** median error 2.2e-3.
- **SIMD kernels:** the f64 AVX kernel agrees with the scalar one to 1e-14 at every worklist length from 1 to 17, and `step_avx` is bit-exact against a scalar reference.
- **Determinism:** there are no rayon float reductions.
- **Pool lifecycle:** `ParticleEffects` retirement is correct, and the pool is bounded.

| ID | Sev | Finding | Status | Test (`particle_regression_tests.rs`) |
|---|---|---|---|---|
| PRT-F0 | Critical (UB) | `Simulation`'s five arrays are `pub`, and `step_avx` bounded its loop by `speeds.len()` while doing raw 4-wide loads and stores on the other four. A shorter array was written past its end; an empty one segfaulted. `step` now checks the lengths and returns `Err`, and the `SAFETY` note names both preconditions. | **Fixed** | `review_simulation_step_with_mismatched_lengths_writes_past_the_end`, `review_simulation_step_with_empty_field_segfaults` |
| PRT-F1 | High | `compute_net_force` silently switched to the f32 kernel above 1000 nodes. In SI units, `m₁·m₂` overflows f32 for Earth masses (result NaN), and positions 10 km from the origin lost 60% of their accuracy. The function is now always f64; the f32 kernel is an explicit opt-in. | **Fixed** | `review_bh_low_precision_path_is_non_finite_for_si_masses`, `review_bh_low_precision_path_loses_accuracy_away_from_origin` |
| PRT-F2 | High | Particles on the root quad's upper edge or outside it were silently dropped: `contains` is half-open, and the if/else chain had no final `else`. The natural bounding-square root puts the max-x or max-y particle exactly on that edge. Particles are now classified against the node centre. | **Fixed** | `review_bh_particle_on_upper_edge_of_root_is_dropped` |
| PRT-F3 | High | Coincident particles recursed until the child quads stopped containing them and were then dropped. At the quad centre this overflowed the stack, in `build_tree` on a rayon worker. `Simulation::new` places every particle at one point. Coincident particles now share a leaf, with a depth cap of 60. | **Fixed** | `review_bh_coincident_particles_*` |
| PRT-F4 | High | `is_x86_feature_detected!`, `std::arch::x86_64` and `#[target_feature(enable = "avx")]` were ungated, so `--features particles` (and therefore `all`) did not compile on aarch64, wasm32 or i686. They are now gated on `target_arch = "x86_64"` with scalar fallbacks. `cargo check --target aarch64-unknown-linux-gnu --features particles` is clean. | **Fixed** | cross-check |
| PRT-F5 | Medium | The opening criterion could accept a node containing the particle itself (θ > 1/√2), so it attracted itself: +37% force at θ = 0.8. Such nodes are always opened. | **Fixed** | `review_bh_node_containing_the_particle_is_accepted_self_interaction` |
| PRT-F6 | Medium | `Particle::update_with_effects` rebuilt the velocity along the *pre-gravity* direction, so a particle at rest with direction (1,0) slid sideways and never fell. | **Fixed** | `review_update_with_effects_discards_gravity_direction` |
| PRT-F7 | Medium (perf) | `ParticleEffects` velocities parked on subnormal floats under drag and contact friction. Aged pools ran at 31 ns/particle against 2.6 fresh. The criterion bench measured that regime, inflating the GPU-crossover numbers `BackendPolicy` calibrates on by about 11×. Tiny speeds now snap to rest, and the bench re-seeds per sample: 10k particles now measure 30 µs, down from 306. | **Fixed** | `review_effects_resting_debris_velocity_parks_on_a_subnormal`, `review_perf_effects_subnormal_velocities` (ignored timing) |
| PRT-F8 | Medium | `Simulation` and `Particle` add `constants.gravity` to +y. `ParticleEffects`, `apply_gravity`, `PhysicsWorld` and the GPU sim all subtract it. This is an API decision: existing tests and docs assert +y. | Open | `review_gravity_sign_disagrees_within_particles_module` (ignored) |
| PRT-F9 | Medium (perf) | The AVX step's 1–3 particle tail went through rayon: n = 1027 took 46–57 µs against 3.3–4.5 µs for n = 1024. It now runs sequentially through the same helper: 2.9 µs. | **Fixed** | `review_perf_simulation_step_remainder_through_rayon` (ignored timing) |
| PRT-F10 | Medium (perf) | The f32 kernel had no `#[target_feature]`. When built as a dependency (no `.cargo/config.toml` `+avx`), it ran 9.4× slower than scalar f64. | **Fixed** | `review_perf_worklist_kernels` (ignored timing) |
| PRT-F11 | Low | Softening (`+1e-12` m²) and the self-match tolerance are hard-coded absolutes in an SI library. Two 1 kg particles 100 nm apart feel 10⁻³ of Newton's force. | Open | `review_bh_hardcoded_softening_is_not_scale_invariant` (ignored) |
| PRT-F12 | Low (perf) | The "SIMD" `compute_net_force` path (allocate, collect, AVX) is slower than the plain recursive walk (5.9 vs 5.3 µs/particle at n = 20k). One `Box` per node, including empty children, dominates. | Open | `review_perf_bh_force_paths` (ignored) |
| PRT-F13 | Low (perf) | `build_tree` allocates four `Vec`s per level and forks rayon down to 2-particle subtrees. In-place partitioning with a fork cutoff near 4096 is 1.5×+ faster at 200k. | Open | `review_perf_build_tree_*` (ignored) |
| PRT-F14 | Low | `Simulation::new` accepted negative mass, and `step` returned `Ok` after writing NaN for `dt = NaN`. | **Fixed** | `review_simulation_accepts_invalid_mass_and_dt` |
| PRT-F15 | Low | With `set_gpu_available(true)`, the CPU path's timings were recorded as GPU samples, so the GPU model was calibrated on CPU numbers. | **Fixed** | `review_effects_record_cpu_timings_as_gpu_samples` |
| PRT-F16 | Low | The SIMD-vs-scalar test had an absolute tolerance of 1e-10 against 9e-10 forces, and only n = 8. The `compute_force_simd_avx` doctest imported a type that does not exist, and was compiled out. | **Fixed** | `review_simd_f{32,64}_matches_scalar_for_worklist_len_1_to_17` |

The `particles-cosmological` and `avx512-simd` feature flags gate no code.

## 5. SPH and particle coupling (`fluid_dynamics/sph.rs`, `particle_coupling.rs`)

Both modules sit behind `fluid_simulation`. The SPH core is sound:

- **Kernels:** poly6, as the solver actually evaluates it, integrates to 1 within 1e-6. The spiky and viscosity kernels are normalised and match finite-difference derivatives, and h is the support radius throughout.
- **Forces:** pressure and viscosity are pairwise antisymmetric, so momentum is conserved to 1e-9.
- **Neighbour search:** it matches brute force exactly, including negative coordinates, particles on cell faces, and hash collisions.
- **Lifecycle:** `drain_settled` keeps all six arrays aligned.

The defects were in the particle–grid coupling and at SPH's boundaries.

| ID | Sev | Finding | Status | Test |
|---|---|---|---|---|
| SPH-F1 | High | `FluidParticle2D/3D` integrated drag with explicit Euler. For dt > 2τ it overshoots and diverges: a 0.1 mm sand grain in water at 60 Hz went 18 → −2943 → … → −∞ in 8 steps. Drag relaxation is now implicit and cannot overshoot the fluid velocity. | **Fixed** | `drag_relaxation_is_stable_for_light_particles{,_3d}` |
| SPH-F2 | Medium | The grid's reaction is `Δv_cell = −F·dt / m_particle`, as if the fluid had the particle's mass. The same force pushes the fluid 1000× harder for a 1 g particle than for a 1 kg one. Fixing it needs the cell mass, which `FluidGrid` has no physical size to derive (an API decision). | Open | `reaction_impulse_is_independent_of_particle_mass` (ignored) |
| SPH-F3 | Medium | Two-way coupling integrated the particle with the caller's `dt` but gave the grid a reaction computed with `grid.get_dt()`. Sub-stepping particles 4× gave the fluid 4× the impulse. The grid now receives the impulse the particle actually got. | **Fixed** | `two_way_coupling_uses_one_timestep{,_3d}` |
| SPH-F4 | Medium (units) | `FluidGrid` velocities are in domain widths per second: `advect` moves `dt·width·v` cells. Tracers moved `v·dt` cells, so they fell 64× behind the dye on a 64-wide grid. Tracers now match `advect`, and the unit is documented. Existing callers will see tracers move `width`× faster. | **Fixed** | `tracer_moves_with_the_grid_advection{,_3d}` |
| SPH-F5 | Medium | Ground contact clamps y and reflects `vel.y` only; it has no contact normal. A drop on a 30° slope never moves and is retired as "settled". With `friction = 1.0` a drop climbs a slope at undiminished speed. The `step` docs claimed "a splash on a slope runs downhill" and now describe what it actually does. | Open | `a_drop_on_a_slope_runs_downhill`, `climbing_a_slope_costs_kinetic_energy` (ignored) |
| SPH-F6 | Medium | Ground friction multiplied tangential velocity per substep, so slide distance ∝ dt (0.166 m at 240 Hz, 0.042 m at 960 Hz). It is now a rate `1/(1 + k·dt)` equal to the old multiplier at 240 Hz, so presets look the same there. | **Fixed** | `ground_friction_is_timestep_independent` |
| SPH-F7 | Medium | `new` tested `<= 0.0`, which NaN passes, and did not check stiffness, viscosity, cohesion, restitution or friction. `step` accepted NaN or ∞ `dt` and hit a `debug_assert!` panic in library code. `spawn` accepted non-finite state, which spread NaN to 4 of 8 neighbours in one step. All three are validated now. | **Fixed** | `new_rejects_nan_and_out_of_range_parameters`, `step_ignores_non_finite_dt_and_gravity`, `spawn_rejects_non_finite_state` |
| SPH-F8 | Medium (perf) | The force pass repeated the whole 27-cell neighbour walk (about 174 candidates to find 26). It now iterates the list the density pass found. Output is bit-identical. Criterion: −26% to −39% per step at 256–4096 particles. | **Fixed** | `neighbour_search_matches_brute_force_*` (also checks the cache) |
| SPH-F9 | Low | Cohesion's acceleration divided by ρᵢ only, so pairs with unequal density (surface next to interior) self-propelled a free lump, carrying 7% of Σm\|v\| as net momentum. It now uses 2/(ρᵢ+ρⱼ), which is unchanged at uniform density. | **Fixed** | `cohesion_conserves_momentum` |
| SPH-F10 | Low | Coincident particles (`r <= 1e-9`) skip each other entirely and never separate, even under 47 kPa. | Open | `coincident_particles_separate` (ignored) |
| SPH-F11 | Low | `speed_sq` overflowed to ∞ above ~1e154 m/s, so the "cap" stopped the particle dead (or produced NaN). | **Fixed** | `speed_cap_survives_overflowing_speed` |
| SPH-F12 | Low | `cell_coord` saturates at `i32::MAX`, then `base + 1` overflowed (a debug panic for a particle at x = 1e9 m). It is now clamped. | **Fixed** | `far_particles_do_not_overflow_the_neighbour_walk` |
| SPH-F13 | Low (perf) | Neighbour cells whose near face is beyond h are now skipped (with a 1e-12 margin). Density −25%, forces −17% on their own. Bit-identical. | **Fixed** | `perf_phase_breakdown`, `fingerprint` (ignored tools) |
| SPH-F14 | Low (perf) | Per-particle gather parallelism with rayon keeps the output bit-identical (−30% at 4096 on a loaded machine). The module's "no parallelism for determinism" rule is not needed for this pattern. | Open (UNVERIFIED speedup) | — |
| SPH-F15 | Low (bench) | `benches/sph.rs` stepped one fluid under gravity for every iteration, so after about 100 steps it measured a puddle, not the "packed cube". It now steps in zero gravity. Bench numbers are not comparable with earlier baselines. | **Fixed** | — |

The simulation fingerprint (n = 338) changes only because of SPH-F6 and SPH-F9, which change the physics by design. Every other SPH change leaves normal runs bit-identical.

---

## Cross-cutting observations

- **Three conflicting normal conventions.**
  - `ContactInfo` (GJK/EPA) and the constraint `Contact*` normals point 1 → 2.
  - `CcdCollisionResult` points 2 → 1.
  - CCD mixed the two in four places (CCD-C4).

  Pick one crate-wide and state it once.
- **Per-call vs per-second coefficients.** Hinge damping (JNT-M5), rope-chain bleed (JNT-M6) and SPH ground friction (SPH-F6) were all per call, so behaviour changed with the step rate and, in the world, with the iteration count. `ω *= exp(−c·dt)` or `1/(1 + c·dt)` is the pattern.
- **Static-body conventions.** The world uses `mass = f64::INFINITY`; the CCD module used `mass <= 0`. Code on either side had to guard both (CCD-H1, CCD-H4, CCD-H7).
- **Tests that could not fail.** Several existing tests asserted only `iterations > 0`, accepted `None`, or ran 10 steps. The new tests all compare against an analytic answer or an O(n²) oracle, and several run 2000+ steps or randomised sweeps.

## Validation of this change

On the branch head:

| Command | Result |
|---|---|
| `cargo test --lib`, default features | 643 passed, 0 failed, 16 ignored |
| `cargo test --lib`, every feature except `gpu` | 1083 passed, 0 failed, 32 ignored |
| `RUSTFLAGS="" cargo test --lib --features particles` (the build a downstream crate gets, without the repo's `+avx`) | green |
| `cargo check --lib --features particles --target aarch64-unknown-linux-gnu` | clean; it failed with 7 errors before |
| `cargo check -p rs_physics_wasm` | clean |
| `cargo bench --no-run` | clean |

The ignored tests are open-defect repros, timing probes, and the pre-existing `bench_step_cost`. Each open-defect repro was confirmed to still fail with `--ignored`, so nothing is ignored that already passes.

The two Bevy demos that set hinge damping were converted to the new per-second unit. Each keeps the damping its author set per call; the only change is the removal of the 8× stepping bug.

- `constraints_3d`: 2%/call at 60 Hz → 1.2 /s
- `ball_playground`: 5%/call at 240 Hz → 12 /s

## Reproducing

```sh
# everything except gpu (CI's `--features all` also builds wgpu)
cargo test --lib --features "constraints materials fluid_simulation thermodynamics fluid_dynamics rotational_dynamics particles particles-cosmological avx512-simd"

# the defects still open, pinned by ignored tests
cargo test --lib --features "..." -- --ignored

# timing probes
cargo test --release --lib --features "..." perf -- --ignored --nocapture
```

Regression tests live in:

- `src/interactions/gjk_epa_regression_tests.rs`
- the `ccd_review_repro_tests` module in `src/interactions/continuous_collision_detection_tests.rs`
- `src/constraints/regression_tests.rs`
- `src/particles/particle_regression_tests.rs`
- `src/fluid_dynamics/sph_regression_tests.rs`
- `src/fluid_dynamics/particle_coupling_regression_tests.rs`
