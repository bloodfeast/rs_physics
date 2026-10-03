# Changelog

Notable changes to `rs_physics`. Versions before 0.3.0 are recorded only in the git log and
`development_log/`.

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
