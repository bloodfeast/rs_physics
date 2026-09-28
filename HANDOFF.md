# HANDOFF: perf-r2 (denormal fix, parallel SPH)

Package R2 from the rs_physics perf pass. Branch `perf-r2` off `articulated-bodies`; PR targets
`articulated-bodies`.

Build target: `CARGO_TARGET_DIR=H:/dev/targets/r2`, `CARGO_BUILD_JOBS=4`, tests with
`-- --test-threads=3`. Lib tests only build in debug (`check_no_joint_straddles_the_awake_set` is
`cfg(debug_assertions)`; pre-existing).

## Status

- [x] Particle drag: exact physical velocity flush; criterion bench retires and re-emits
- [x] SPH: SoA, sorted merged bucket runs, neighbour lists, parallel, bit-identical at any thread count
- [x] Measurements: quiet.py (full quiet), ABBA, medians of four, in-process
- [x] Docs to the bar (`#![warn(missing_docs)]` on both modules), figures corrected (`CPU_SEED_NS`)
- [x] Whole suite: every non-GPU feature (1,390 tests), default, each feature on top of default,
      `--features all` test build (wgpu 30.0.1), `gpu` release build
- [x] PR

## Measuring

`examples/r2_bench.rs` (particles fresh / aged / live, SPH at 1, 4, 8 threads) and
`examples/r2_phases.rs` (SPH phase timers). Build with
`cargo build --release --examples --features "particles fluid_simulation"`, copy the exe, run each
through `py -3 tools/quiet.py -- <exe> 1` one round a run, alternate base/new ABBA. The crate's
`.cargo/config.toml` builds these with `+avx`; Ridgeline builds rs_physics with plain x86-64 (SSE2).

## Open

- The flush costs about 0.4 ns a particle on a fresh pool (2.45 -> 2.86 ns); the live pool repays
  it 2.6x. The particle loop does not vectorise (class-table lookups; it is also near L2 bandwidth
  at 16k). A per-particle drag/gravity copy was tried in a micro-bench and did not help.
- The SPH candidate walk (density phase, ~60% of a step) is scalar compaction at ~2 ns a candidate.
  Finer cells or per-particle row trimming were considered and not tried.
- `SphFluid::step` now uses the calling rayon pool; a Bevy caller that does not `install` gets the
  global pool, which INTENT (2026-09-27) rules out for presentation physics. The R3 physics thread
  should `install` its own pool around the step.
