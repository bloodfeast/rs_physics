# Changelog

Notable changes to `rs_physics`. Versions before 0.3.0 are recorded only in the git log and
`development_log/`.

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
