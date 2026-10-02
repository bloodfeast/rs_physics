//! Solver configuration and types for fluid dynamics simulations
//!
//! This module provides configurable solver parameters and type-safe
//! boundary condition handling for fluid simulation, and the conjugate-gradient
//! pressure solve the grids share.

use crate::utils::PhysicsError;

/// Type of iterative solver used for the diffusion solves, and for the pressure
/// solve when [`SolverConfig::pressure_solver`] is [`PressureSolver::Relaxation`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolverType {
    /// Gauss-Seidel relaxation (default)
    /// - Good convergence properties
    /// - Sequential updates (harder to parallelize)
    #[default]
    GaussSeidel,

    /// Jacobi iteration
    /// - Slower convergence than Gauss-Seidel
    /// - Embarrassingly parallel (all cells update independently)
    Jacobi,

    /// Successive Over-Relaxation (SOR): each Gauss-Seidel update is moved
    /// `relaxation` times as far, `x ← (1 − ω)·x + ω·x_GS`
    /// - Faster convergence with a relaxation factor near the optimum
    ///   (`2 / (1 + sin(π/n))` for the pressure solve on an `n`-cell grid)
    /// - `relaxation = 1` is Gauss-Seidel
    SOR,
}

/// How the grids solve the pressure Poisson equation that makes the velocity
/// divergence-free.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PressureSolver {
    /// Conjugate gradient, preconditioned with modified incomplete Cholesky (MIC(0),
    /// Bridson's *Fluid Simulation for Computer Graphics*, §5.4), warm-started from
    /// the previous projection's pressure. It stops when the residual's 2-norm falls
    /// to [`SolverConfig::pressure_tolerance`] times the right-hand side's, or after
    /// [`SolverConfig::pressure_max_iterations`]. The default.
    #[default]
    ConjugateGradient,

    /// [`SolverConfig::iterations`] sweeps of [`SolverConfig::solver_type`] from zero
    /// pressure, which is how every grid solved it before 2026-09-30 and reproduces
    /// those results bit-for-bit with Gauss-Seidel. Fixed and cheap, but it removes
    /// little of a large-scale divergence on a big grid: about 1% per step at 128²
    /// with 4 sweeps.
    Relaxation,
}

/// What a wall does to the velocity *along* it. Every wall stops the velocity
/// *through* it, under either condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WallCondition {
    /// The fluid slides along the wall without friction: the ghost cell copies the
    /// tangential velocity of its neighbour. The default, and Stam's choice; right
    /// for smoke in a room, wrong for a river bed.
    #[default]
    FreeSlip,

    /// The fluid is at rest at the wall: the ghost cell holds the negated tangential
    /// velocity of its neighbour, so the wall, half-way between them, sees zero.
    /// Walls drag on the flow and grow a boundary layer.
    NoSlip,
}

/// How the grids carry velocity and density along the flow each step.
///
/// # Examples
/// ```
/// use rs_physics::fluid_dynamics::{AdvectionScheme, FluidGrid, SolverConfig};
///
/// let config = SolverConfig::default().with_advection(AdvectionScheme::MacCormack);
/// let mut grid = FluidGrid::with_solver(32, 32, 0.0, 0.0, 1.0 / 60.0, config).unwrap();
/// grid.add_density(16, 16, 1.0).unwrap();
/// grid.add_velocity(16, 16, 0.3, 0.1).unwrap();
/// grid.step();
/// assert!(grid.validate_state().is_ok());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AdvectionScheme {
    /// First-order semi-Lagrangian advection (Stam 1999): trace each cell back along
    /// the velocity for one step and interpolate linearly where it lands. Stable at any
    /// step, and diffusive: linear interpolation at a fractional cell offset `a` acts
    /// as a viscosity `h² a(1 - a) / (2 dt)` per axis, which is what smooths a plume's
    /// edges and spins its eddies down. The default, and bit-identical to every grid
    /// before 2026-10-02.
    #[default]
    SemiLagrangian,

    /// MacCormack advection (Selle, Fedkiw, Kim, Liu and Rossignac, "An Unconditionally
    /// Stable MacCormack Method", J. Sci. Comput. 2008), a BFECC-class scheme: a forward
    /// semi-Lagrangian step `f = A(q)`, a backward one from its result
    /// `b = A_reverse(f)`, and the corrected value `f + (q - b) / 2`, which cancels the
    /// forward step's leading error and makes the scheme second order in smooth flow.
    /// The correction can overshoot at a sharp edge, so each cell is clamped to the
    /// range of the source cells its forward interpolation read; that clamp is what
    /// keeps it stable at any step, and it falls back towards first order exactly
    /// where the field has an extremum. Costs a second interpolation pass and a
    /// combine pass for every advected field.
    MacCormack,
}

/// Vorticity confinement (Fedkiw, Stam and Jensen, "Visual Simulation of Smoke",
/// SIGGRAPH 2001): a body force that puts back the small-scale rotation the
/// advection's numerical dissipation takes out.
///
/// With `omega = curl u` and `N = grad|omega| / |grad|omega||` (the unit vector
/// towards stronger rotation), the force is `f = epsilon h (N x omega)`, with `h` the
/// cell size, exactly as the paper writes it, added to the velocity before the
/// projection that follows advection. The paper leaves `epsilon` free: "used to
/// control the amount of small scale detail added back into the flow field", with no
/// range given. Here it is derived instead, so there is no number to tune.
///
/// # The derivation
///
/// For `omega > 0`, `f` is `-epsilon h (|omega| / |grad|omega||) (z x grad omega)`,
/// and a viscous force is `nu (z x grad omega)`: confinement is a negative viscosity
/// `nu_c = epsilon h l`, with `l = |omega| / |grad|omega||` the length over which the
/// rotation changes. The structures confinement exists to keep are the ones at the
/// grid scale, `l = h`, so `nu_c = epsilon h²`.
///
/// What first-order semi-Lagrangian advection removes is also a viscosity. Linear
/// interpolation at a fractional offset `a` of a cell has the modified equation
/// `q_t + u q_x = nu_num q_xx` with `nu_num = h² a (1 - a) / (2 dt)`, per axis; at a
/// Courant number below one, `a` is the Courant number itself and this is first-order
/// upwind's familiar `|u| h (1 - a) / 2`. Setting `nu_c = nu_num` gives
///
/// `epsilon = a (1 - a) / (2 dt)`, averaged over the axes,
///
/// in 1/s, so the velocity a step adds is `dt f = h e (N x omega)` with the
/// dimensionless `e = mean over axes of a (1 - a) / 2`: between 0 (a step that moves
/// the flow a whole number of cells, which semi-Lagrangian advection does exactly) and
/// 1/8 (half a cell, the most diffusive), and 1/12 on average over offsets. Each cell
/// uses the offsets of its own departure point this step, so confinement is strong
/// where the advection was diffusive and absent where it was exact.
///
/// The match is made at the grid scale, as the method intends: a structure much
/// larger than a cell is confined more than its numerical dissipation (the force does
/// not shrink with `l`), which is the method's known character, not this derivation's.
/// Under [`AdvectionScheme::MacCormack`] the advection dissipates less than the
/// first-order rate this uses, so the two together put back more than was lost; see
/// the tests for what that does to a decaying vortex.
///
/// # Examples
/// ```
/// use rs_physics::fluid_dynamics::{FluidGrid3D, SolverConfig, VorticityConfinement};
///
/// let config = SolverConfig::default()
///     .with_vorticity_confinement(VorticityConfinement::MatchNumericalDissipation);
/// let mut grid = FluidGrid3D::with_solver(16, 16, 16, 0.0, 0.0, 1.0 / 60.0, config).unwrap();
/// grid.add_velocity(8, 8, 8, 0.2, 0.0, 0.1).unwrap();
/// grid.step();
/// assert!(grid.validate_state().is_ok());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VorticityConfinement {
    /// No confinement force. The default, and bit-identical to every grid before
    /// 2026-10-02.
    #[default]
    Off,

    /// Confinement with `epsilon` matched, cell by cell, to the numerical viscosity of
    /// first-order semi-Lagrangian advection at that cell's offsets this step (see the
    /// type's documentation). Costs a curl, a gradient and a force pass per step.
    MatchNumericalDissipation,
}

/// Configuration for the iterative solver
#[derive(Debug, Clone, Copy)]
pub struct SolverConfig {
    /// Number of sweeps for the diffusion solves, and for the pressure solve under
    /// [`PressureSolver::Relaxation`] (default: 4). Fewer than 1 is raised to 1.
    /// More iterations = more accurate but slower
    pub iterations: usize,

    /// Relaxation factor ω for [`SolverType::SOR`] (default: 1.9). Must lie in the
    /// open interval (0, 2), where SOR converges; the grids reject anything else,
    /// whatever the solver type.
    /// - 1.0 = equivalent to Gauss-Seidel
    /// - 1.0 to 2.0 = over-relaxation (faster convergence)
    pub relaxation: f64,

    /// The relaxation solver to use (default: Gauss-Seidel)
    pub solver_type: SolverType,

    /// How the pressure is solved (default: [`PressureSolver::ConjugateGradient`])
    pub pressure_solver: PressureSolver,

    /// Relative residual at which the conjugate-gradient pressure solve stops:
    /// `|b − A·p|₂ ≤ tolerance · |b|₂` (default: 1e-2). Must lie in (0, 1).
    ///
    /// The trade, measured at 128² on a forced plume over 2 s against a converged solve,
    /// and against one cold step of a smooth divergence:
    ///
    /// | tolerance | step cost vs the old 4 sweeps | flow after 2 s | divergence energy left |
    /// |---|---|---|---|
    /// | 0.3 | −24% | velocity 5% off | 1e-4 |
    /// | **1e-2 (default)** | −12% | 0.5% off | 2e-9 |
    /// | 1e-4 | +72% | 0.004% off | 2e-13 |
    /// | 1e-8 | +440% | converged | rounding |
    ///
    /// The old relaxation (`PressureSolver::Relaxation`) left the same plume 91% off and
    /// 99% of the divergence in place. The default is where the flow stops changing
    /// visibly and the step is still cheaper than the solver it replaced.
    pub pressure_tolerance: f64,

    /// Cap on conjugate-gradient iterations per projection (default: 200); each step
    /// projects twice. Fewer than 1 is raised to 1.
    pub pressure_max_iterations: usize,

    /// What walls do to the tangential velocity (default: [`WallCondition::FreeSlip`])
    pub wall: WallCondition,

    /// How velocity and density are advected (default:
    /// [`AdvectionScheme::SemiLagrangian`])
    pub advection: AdvectionScheme,

    /// Whether a vorticity-confinement force is added before the second projection
    /// (default: [`VorticityConfinement::Off`])
    pub vorticity_confinement: VorticityConfinement,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            iterations: 4,
            relaxation: 1.9,
            solver_type: SolverType::GaussSeidel,
            pressure_solver: PressureSolver::ConjugateGradient,
            pressure_tolerance: 1e-2,
            pressure_max_iterations: 200,
            wall: WallCondition::FreeSlip,
            advection: AdvectionScheme::SemiLagrangian,
            vorticity_confinement: VorticityConfinement::Off,
        }
    }
}

impl SolverConfig {
    /// Creates a new solver configuration with the specified number of iterations
    pub fn new(iterations: usize) -> Self {
        Self {
            iterations,
            ..Default::default()
        }
    }

    /// Creates a Gauss-Seidel solver configuration
    pub fn gauss_seidel(iterations: usize) -> Self {
        Self {
            iterations,
            solver_type: SolverType::GaussSeidel,
            ..Default::default()
        }
    }

    /// Creates a Jacobi solver configuration
    pub fn jacobi(iterations: usize) -> Self {
        Self {
            iterations,
            solver_type: SolverType::Jacobi,
            ..Default::default()
        }
    }

    /// Creates an SOR solver configuration with the specified relaxation factor.
    ///
    /// The factor is not checked here: [`SolverConfig::validate`], and every grid
    /// constructor and setter that takes a config, rejects one outside (0, 2).
    pub fn sor(iterations: usize, relaxation: f64) -> Self {
        Self {
            iterations,
            relaxation,
            solver_type: SolverType::SOR,
            ..Default::default()
        }
    }

    /// Returns a high-quality configuration with more iterations
    pub fn high_quality() -> Self {
        Self {
            iterations: 20,
            ..Default::default()
        }
    }

    /// Returns a fast configuration with fewer iterations
    pub fn fast() -> Self {
        Self {
            iterations: 2,
            ..Default::default()
        }
    }

    /// Returns this configuration with the pressure solved by `solver`.
    pub fn with_pressure_solver(mut self, solver: PressureSolver) -> Self {
        self.pressure_solver = solver;
        self
    }

    /// Returns this configuration with the conjugate-gradient stopping rule set.
    pub fn with_pressure_tolerance(mut self, tolerance: f64, max_iterations: usize) -> Self {
        self.pressure_tolerance = tolerance;
        self.pressure_max_iterations = max_iterations;
        self
    }

    /// Returns this configuration with walls under `wall`.
    pub fn with_wall_condition(mut self, wall: WallCondition) -> Self {
        self.wall = wall;
        self
    }

    /// Returns this configuration with velocity and density advected by `scheme`.
    ///
    /// # Arguments
    /// * `scheme` - the advection scheme; see [`AdvectionScheme`] for what each costs
    ///
    /// # Returns
    /// The updated configuration.
    ///
    /// # Examples
    /// ```
    /// use rs_physics::fluid_dynamics::{AdvectionScheme, SolverConfig};
    /// let config = SolverConfig::default().with_advection(AdvectionScheme::MacCormack);
    /// assert_eq!(config.advection, AdvectionScheme::MacCormack);
    /// ```
    pub fn with_advection(mut self, scheme: AdvectionScheme) -> Self {
        self.advection = scheme;
        self
    }

    /// Returns this configuration with vorticity confinement set to `confinement`.
    ///
    /// # Arguments
    /// * `confinement` - off, or matched to the advection's numerical dissipation; see
    ///   [`VorticityConfinement`] for the derivation
    ///
    /// # Returns
    /// The updated configuration.
    ///
    /// # Examples
    /// ```
    /// use rs_physics::fluid_dynamics::{SolverConfig, VorticityConfinement};
    /// let config = SolverConfig::default()
    ///     .with_vorticity_confinement(VorticityConfinement::MatchNumericalDissipation);
    /// assert_eq!(config.vorticity_confinement, VorticityConfinement::MatchNumericalDissipation);
    /// ```
    pub fn with_vorticity_confinement(mut self, confinement: VorticityConfinement) -> Self {
        self.vorticity_confinement = confinement;
        self
    }

    /// Checks the fields a solver cannot run with.
    ///
    /// # Returns
    /// * `Ok(())` if the configuration is usable
    /// * `Err(PhysicsError::InvalidCoefficient)` if `relaxation` is outside the open
    ///   interval (0, 2) or not finite, or `pressure_tolerance` is outside (0, 1) or
    ///   not finite
    ///
    /// Iteration counts of 0 are not errors; the grids raise them to 1.
    pub fn validate(&self) -> Result<(), PhysicsError> {
        // Written so that NaN fails: every comparison with NaN is false.
        if !(self.relaxation > 0.0 && self.relaxation < 2.0) {
            return Err(PhysicsError::InvalidCoefficient);
        }
        if !(self.pressure_tolerance > 0.0 && self.pressure_tolerance < 1.0) {
            return Err(PhysicsError::InvalidCoefficient);
        }
        Ok(())
    }

    /// The configuration a grid keeps: validated, with iteration counts of 0 raised to 1.
    pub(crate) fn checked(mut self) -> Result<Self, PhysicsError> {
        self.validate()?;
        self.iterations = self.iterations.max(1);
        self.pressure_max_iterations = self.pressure_max_iterations.max(1);
        Ok(self)
    }
}

/// Scratch vectors for [`pcg_solve`], kept by a grid so a solve allocates nothing.
#[derive(Debug, Default, Clone)]
pub(crate) struct PcgWorkspace {
    r: Vec<f64>,
    z: Vec<f64>,
    s: Vec<f64>,
    t: Vec<f64>,
}

impl PcgWorkspace {
    /// The elements the four work vectors hold, as allocated.
    pub(crate) fn capacity(&self) -> usize {
        self.r.capacity() + self.z.capacity() + self.s.capacity() + self.t.capacity()
    }

    /// Sizes every vector to `len`, zero-filled. Called on every solve, but only
    /// allocates when the length changes; the cells a solve never writes (the grid's
    /// boundary ring) are zero from here on.
    fn fit(&mut self, len: usize) {
        for v in [&mut self.r, &mut self.z, &mut self.s, &mut self.t] {
            if v.len() != len {
                v.clear();
                v.resize(len, 0.0);
            }
        }
    }
}

/// `a · b` with eight partial sums combined in a fixed order. A single running sum
/// is one long chain of dependent adds, which made the dot products most of an
/// unpreconditioned iteration's cost; eight lanes break the chain and vectorise.
/// The order is fixed, so the result is the same on every run and thread count.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut lanes = [0.0f64; 8];
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let tail: f64 = chunks_a.remainder().iter().zip(chunks_b.remainder()).map(|(x, y)| x * y).sum();
    for (ca, cb) in chunks_a.zip(chunks_b) {
        for lane in 0..8 {
            lanes[lane] += ca[lane] * cb[lane];
        }
    }
    ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7])) + tail
}

/// Solves `A·p = b` by preconditioned conjugate gradient, starting from the `p`
/// passed in, until `|b − A·p|₂ ≤ tolerance · |b|₂` or `max_iterations`.
///
/// Every vector is a whole grid, boundary ring included. `b` must be zero on the
/// ring, and `apply` (`out = A·x`) and `precondition` (`out = M⁻¹·x`) must write
/// only fluid cells, so the ring stays zero in every work vector and whole-vector
/// sums are sums over the fluid. `A` may be singular (a closed box's Neumann
/// Laplacian is, with the constants as its null space) if `b` is in its range.
///
/// Serial on purpose: every reduction runs in index order, so the answer does not
/// depend on the thread count. Returns the iterations taken, 0 if the start was
/// already within tolerance.
pub(crate) fn pcg_solve(
    p: &mut [f64],
    b: &[f64],
    ws: &mut PcgWorkspace,
    tolerance: f64,
    max_iterations: usize,
    apply: impl Fn(&[f64], &mut [f64]),
    precondition: impl Fn(&[f64], &mut [f64]),
) -> usize {
    ws.fit(p.len());
    let PcgWorkspace { r, z, s, t } = ws;

    let target = tolerance * dot(b, b).sqrt();
    apply(p, t);
    for ((ri, bi), ti) in r.iter_mut().zip(b).zip(t.iter()) {
        *ri = bi - ti;
    }
    if dot(r, r).sqrt() <= target {
        return 0;
    }

    precondition(r, z);
    s.copy_from_slice(z);
    let mut sigma = dot(z, r);
    for iteration in 1..=max_iterations {
        apply(s, t);
        let curvature = dot(s, t);
        if !(curvature > 0.0) || !sigma.is_finite() {
            // `s` has no component outside A's null space left: nothing to do.
            return iteration - 1;
        }
        let alpha = sigma / curvature;
        for (pi, si) in p.iter_mut().zip(s.iter()) {
            *pi += alpha * si;
        }
        for (ri, ti) in r.iter_mut().zip(t.iter()) {
            *ri -= alpha * ti;
        }
        if dot(r, r).sqrt() <= target {
            return iteration;
        }
        precondition(r, z);
        let sigma_next = dot(z, r);
        let beta = sigma_next / sigma;
        for (si, zi) in s.iter_mut().zip(z.iter()) {
            *si = zi + beta * *si;
        }
        sigma = sigma_next;
    }
    max_iterations
}

/// Type of boundary condition to apply
///
/// Replaces magic numbers (0, 1, 2, 3) with descriptive enum variants.
/// Used in `set_boundaries` to determine how values are reflected at walls. Every
/// wall blocks the velocity component normal to it; what happens to the tangential
/// components is the grid's [`WallCondition`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryType {
    /// Density or pressure field - no-flux boundary (continuous across boundary)
    /// Values at boundary equal adjacent interior values
    Density,

    /// X-component of velocity: negated at the left/right walls, so nothing crosses
    /// them. At the other walls, copied under [`WallCondition::FreeSlip`] and negated
    /// under [`WallCondition::NoSlip`].
    VelocityX,

    /// Y-component of velocity: negated at the top/bottom walls, so nothing crosses
    /// them. At the other walls, copied under [`WallCondition::FreeSlip`] and negated
    /// under [`WallCondition::NoSlip`].
    VelocityY,

    /// Z-component of velocity (3D only): negated at the front/back walls, so nothing
    /// crosses them. At the other walls, copied under [`WallCondition::FreeSlip`] and
    /// negated under [`WallCondition::NoSlip`].
    VelocityZ,
}

impl BoundaryType {
    /// Converts to the legacy integer representation
    /// Used for compatibility with existing code during migration
    #[inline]
    pub fn to_legacy_int(self) -> i32 {
        match self {
            BoundaryType::Density => 0,
            BoundaryType::VelocityX => 1,
            BoundaryType::VelocityY => 2,
            BoundaryType::VelocityZ => 3,
        }
    }

    /// Creates from legacy integer representation
    /// Returns None for invalid values
    #[inline]
    pub fn from_legacy_int(b: i32) -> Option<Self> {
        match b {
            0 => Some(BoundaryType::Density),
            1 => Some(BoundaryType::VelocityX),
            2 => Some(BoundaryType::VelocityY),
            3 => Some(BoundaryType::VelocityZ),
            _ => None,
        }
    }

    /// Whether the wall that `normal` crosses (`VelocityX` for the left and right
    /// walls, and so on) negates this field in its ghost cell under `wall`.
    #[inline]
    pub(crate) fn flips_at_wall(self, normal: BoundaryType, wall: WallCondition) -> bool {
        match self {
            BoundaryType::Density => false,
            _ => self == normal || wall == WallCondition::NoSlip,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_config_default() {
        let config = SolverConfig::default();
        assert_eq!(config.iterations, 4);
        assert_eq!(config.solver_type, SolverType::GaussSeidel);
        assert_eq!(config.pressure_solver, PressureSolver::ConjugateGradient);
        assert_eq!(config.wall, WallCondition::FreeSlip);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_solver_config_new() {
        let config = SolverConfig::new(10);
        assert_eq!(config.iterations, 10);
        assert_eq!(config.solver_type, SolverType::GaussSeidel);
    }

    #[test]
    fn test_solver_config_gauss_seidel() {
        let config = SolverConfig::gauss_seidel(8);
        assert_eq!(config.iterations, 8);
        assert_eq!(config.solver_type, SolverType::GaussSeidel);
    }

    #[test]
    fn test_solver_config_jacobi() {
        let config = SolverConfig::jacobi(12);
        assert_eq!(config.iterations, 12);
        assert_eq!(config.solver_type, SolverType::Jacobi);
    }

    #[test]
    fn test_solver_config_sor() {
        let config = SolverConfig::sor(6, 1.5);
        assert_eq!(config.iterations, 6);
        assert_eq!(config.relaxation, 1.5);
        assert_eq!(config.solver_type, SolverType::SOR);
        assert!(config.validate().is_ok());
    }

    /// `sor` used to `assert!` the factor, a panic in a library path; it now builds
    /// the config and `validate` (and every grid that takes it) refuses it.
    #[test]
    fn test_solver_config_sor_out_of_range_is_rejected_not_panicking() {
        for bad in [2.0, 0.0, -0.5, 2.5, f64::NAN, f64::INFINITY] {
            let config = SolverConfig::sor(6, bad);
            assert_eq!(config.validate(), Err(PhysicsError::InvalidCoefficient), "relaxation {bad}");
        }
    }

    #[test]
    fn test_solver_config_rejects_bad_pressure_tolerance() {
        for bad in [0.0, -1e-6, 1.0, f64::NAN, f64::INFINITY] {
            let config = SolverConfig::default().with_pressure_tolerance(bad, 100);
            assert!(config.validate().is_err(), "tolerance {bad}");
        }
        assert!(SolverConfig::default().with_pressure_tolerance(1e-10, 0).validate().is_ok());
    }

    #[test]
    fn test_solver_config_high_quality() {
        let config = SolverConfig::high_quality();
        assert_eq!(config.iterations, 20);
    }

    #[test]
    fn test_solver_config_fast() {
        let config = SolverConfig::fast();
        assert_eq!(config.iterations, 2);
    }

    #[test]
    fn test_boundary_type_to_legacy() {
        assert_eq!(BoundaryType::Density.to_legacy_int(), 0);
        assert_eq!(BoundaryType::VelocityX.to_legacy_int(), 1);
        assert_eq!(BoundaryType::VelocityY.to_legacy_int(), 2);
        assert_eq!(BoundaryType::VelocityZ.to_legacy_int(), 3);
    }

    #[test]
    fn test_boundary_type_from_legacy() {
        assert_eq!(BoundaryType::from_legacy_int(0), Some(BoundaryType::Density));
        assert_eq!(BoundaryType::from_legacy_int(1), Some(BoundaryType::VelocityX));
        assert_eq!(BoundaryType::from_legacy_int(2), Some(BoundaryType::VelocityY));
        assert_eq!(BoundaryType::from_legacy_int(3), Some(BoundaryType::VelocityZ));
        assert_eq!(BoundaryType::from_legacy_int(4), None);
        assert_eq!(BoundaryType::from_legacy_int(-1), None);
    }

    #[test]
    fn test_boundary_type_roundtrip() {
        for b in [BoundaryType::Density, BoundaryType::VelocityX,
                  BoundaryType::VelocityY, BoundaryType::VelocityZ] {
            assert_eq!(BoundaryType::from_legacy_int(b.to_legacy_int()), Some(b));
        }
    }

    #[test]
    fn test_solver_type_default() {
        assert_eq!(SolverType::default(), SolverType::GaussSeidel);
    }

    /// The conjugate gradient against a direct answer: a 1D Neumann Laplacian with
    /// a mean-zero right-hand side, where `p` is known in closed form up to a constant.
    #[test]
    fn test_pcg_solves_a_singular_neumann_system() {
        // Cells 1..=n are unknowns; 0 and n+1 are the always-zero ring.
        let n = 40;
        let len = n + 2;
        let apply = |x: &[f64], out: &mut [f64]| {
            for i in 1..=n {
                let l = if i > 1 { x[i - 1] } else { x[i] };
                let r = if i < n { x[i + 1] } else { x[i] };
                out[i] = 2.0 * x[i] - l - r;
            }
        };
        let identity = |x: &[f64], out: &mut [f64]| {
            out[1..=n].copy_from_slice(&x[1..=n]);
        };
        // A known solution, then its right-hand side.
        let exact: Vec<f64> = (0..len)
            .map(|i| if i == 0 || i == len - 1 { 0.0 } else { (0.3 * i as f64).cos() })
            .collect();
        let mut b = vec![0.0; len];
        apply(&exact, &mut b);
        let mut p = vec![0.0; len];
        let mut ws = PcgWorkspace::default();
        let iterations = pcg_solve(&mut p, &b, &mut ws, 1e-12, 1000, apply, identity);
        // Unpreconditioned CG solves an n-unknown system in at most n steps.
        assert!(iterations <= n, "{iterations} iterations");
        let shift = (1..=n).map(|i| exact[i] - p[i]).sum::<f64>() / n as f64;
        for i in 1..=n {
            assert!((p[i] + shift - exact[i]).abs() < 1e-9, "cell {i}: {} vs {}", p[i] + shift, exact[i]);
        }
        assert_eq!((p[0], p[len - 1]), (0.0, 0.0), "the ring must stay untouched");
    }
}
