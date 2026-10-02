# Changelog

Notable changes to `rs_physics`. Versions before 0.3.0 are recorded only in the git log and
`development_log/`.

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
