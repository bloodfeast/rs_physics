//! Solver configuration and types for fluid dynamics simulations
//!
//! This module provides configurable solver parameters and type-safe
//! boundary condition handling for fluid simulation.

/// Type of iterative solver used for diffusion and pressure equations
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

    /// Successive Over-Relaxation (SOR)
    /// - Faster convergence with optimal relaxation factor
    /// - Relaxation factor between 1.0 and 2.0
    SOR,
}

/// Configuration for the iterative solver
#[derive(Debug, Clone, Copy)]
pub struct SolverConfig {
    /// Number of iterations for the linear solver (default: 4)
    /// More iterations = more accurate but slower
    pub iterations: usize,

    /// Relaxation factor for SOR solver (default: 1.9)
    /// - 1.0 = equivalent to Gauss-Seidel
    /// - 1.0 to 2.0 = over-relaxation (faster convergence)
    /// - Values outside (0, 2) will diverge
    pub relaxation: f64,

    /// The type of solver to use
    pub solver_type: SolverType,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            iterations: 4,
            relaxation: 1.9,
            solver_type: SolverType::GaussSeidel,
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

    /// Creates an SOR solver configuration with the specified relaxation factor
    ///
    /// # Panics
    /// Panics if relaxation factor is not in the range (0, 2)
    pub fn sor(iterations: usize, relaxation: f64) -> Self {
        assert!(relaxation > 0.0 && relaxation < 2.0,
            "SOR relaxation factor must be in range (0, 2), got {}", relaxation);
        Self {
            iterations,
            relaxation,
            solver_type: SolverType::SOR,
        }
    }

    /// Returns a high-quality configuration with more iterations
    pub fn high_quality() -> Self {
        Self {
            iterations: 20,
            relaxation: 1.9,
            solver_type: SolverType::GaussSeidel,
        }
    }

    /// Returns a fast configuration with fewer iterations
    pub fn fast() -> Self {
        Self {
            iterations: 2,
            relaxation: 1.9,
            solver_type: SolverType::GaussSeidel,
        }
    }
}

/// Type of boundary condition to apply
///
/// Replaces magic numbers (0, 1, 2, 3) with descriptive enum variants.
/// Used in `set_boundaries` to determine how values are reflected at walls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryType {
    /// Density or pressure field - no-flux boundary (continuous across boundary)
    /// Values at boundary equal adjacent interior values
    Density,

    /// X-component of velocity - no-slip boundary in X direction
    /// X-velocity is negated at left/right walls, continuous at top/bottom
    VelocityX,

    /// Y-component of velocity - no-slip boundary in Y direction
    /// Y-velocity is negated at top/bottom walls, continuous at left/right
    VelocityY,

    /// Z-component of velocity - no-slip boundary in Z direction (3D only)
    /// Z-velocity is negated at front/back walls, continuous at other faces
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_config_default() {
        let config = SolverConfig::default();
        assert_eq!(config.iterations, 4);
        assert_eq!(config.solver_type, SolverType::GaussSeidel);
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
    }

    #[test]
    #[should_panic(expected = "SOR relaxation factor must be in range (0, 2)")]
    fn test_solver_config_sor_invalid_relaxation_high() {
        SolverConfig::sor(6, 2.0);
    }

    #[test]
    #[should_panic(expected = "SOR relaxation factor must be in range (0, 2)")]
    fn test_solver_config_sor_invalid_relaxation_low() {
        SolverConfig::sor(6, 0.0);
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
}
