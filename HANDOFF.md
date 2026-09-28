# HANDOFF: perf-r2 (denormal fix, parallel SPH)

Package R2 from the rs_physics perf pass. Branch `perf-r2` off `articulated-bodies`; PR targets
`articulated-bodies`.

Build target: `CARGO_TARGET_DIR=H:/dev/targets/r2`, `CARGO_BUILD_JOBS=4`, tests with
`-- --test-threads=3`.

## Status

- [ ] Particle drag: physical velocity flush, bench retires and re-emits
- [ ] SPH: SoA, sorted neighbour ranges, parallel and bit-identical at any thread count
- [ ] Measurements (quiet.py, medians of four)
- [ ] Docs and every number quoting the old bench corrected
- [ ] Whole suite, every feature set, `gpu` build
- [ ] PR
