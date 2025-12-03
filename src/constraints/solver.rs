//! Unified constraint solvers with warm starting support.
//!
//! This module provides unified constraint solvers that can work with any
//! constraint type implementing the `Constraint2D` or `Constraint3D` traits.

use crate::utils::PhysicsError;

/// Result of a solver iteration.
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
    /// Maximum constraint error after solving.
    pub max_error: f64,
}

/// Trait for 2D constraints that can be solved by the unified solver.
pub trait Constraint2D {
    /// Solves the constraint for one iteration.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError>;

    /// Calculates the current constraint error.
    ///
    /// # Returns
    ///
    /// The magnitude of the constraint violation.
    fn calculate_error(&self) -> f64;

    /// Gets the cached lambda (impulse accumulator) for warm starting.
    fn get_lambda(&self) -> f64;

    /// Sets the lambda for warm starting.
    fn set_lambda(&mut self, lambda: f64);
}

/// Trait for 3D constraints that can be solved by the unified solver.
pub trait Constraint3D {
    /// Solves the constraint for one iteration.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError>;

    /// Calculates the current constraint error.
    ///
    /// # Returns
    ///
    /// The magnitude of the constraint violation.
    fn calculate_error(&self) -> f64;

    /// Gets the cached lambda (impulse accumulator) for warm starting.
    fn get_lambda(&self) -> f64;

    /// Sets the lambda for warm starting.
    fn set_lambda(&mut self, lambda: f64);
}

/// Unified 2D constraint solver with optional warm starting.
///
/// This solver uses Gauss-Seidel iteration to solve a collection of 2D
/// constraints. Warm starting can be enabled to improve convergence by
/// using cached impulse values from the previous frame.
///
/// # Examples
///
/// ```rust,ignore
/// use rs_physics::constraints::{UnifiedSolver2D, Constraint2D};
///
/// let mut solver = UnifiedSolver2D::new(10, 0.001).unwrap();
/// solver.add_constraint(Box::new(my_joint));
/// let result = solver.solve(0.016).unwrap();
/// println!("Converged: {}, iterations: {}", result.converged, result.iterations);
/// ```
pub struct UnifiedSolver2D {
    /// Constraints managed by this solver.
    constraints: Vec<Box<dyn Constraint2D>>,
    /// Maximum iterations per solve.
    max_iterations: usize,
    /// Convergence tolerance.
    tolerance: f64,
    /// Whether warm starting is enabled.
    warm_start: bool,
    /// Cached lambda values for warm starting.
    lambda_cache: Vec<f64>,
}

impl UnifiedSolver2D {
    /// Creates a new UnifiedSolver2D.
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - Maximum number of solver iterations (must be > 0)
    /// * `tolerance` - Convergence tolerance (must be > 0)
    ///
    /// # Returns
    ///
    /// * `Ok(UnifiedSolver2D)` - Valid solver
    /// * `Err(PhysicsError)` - If parameters are invalid
    pub fn new(max_iterations: usize, tolerance: f64) -> Result<Self, PhysicsError> {
        if max_iterations == 0 {
            return Err(PhysicsError::CalculationError(
                "max_iterations must be greater than 0".to_string(),
            ));
        }
        if tolerance <= 0.0 {
            return Err(PhysicsError::CalculationError(
                "tolerance must be positive".to_string(),
            ));
        }

        Ok(Self {
            constraints: Vec::new(),
            max_iterations,
            tolerance,
            warm_start: false,
            lambda_cache: Vec::new(),
        })
    }

    /// Enables or disables warm starting.
    ///
    /// Warm starting uses cached impulse values from previous frames
    /// to improve solver convergence.
    pub fn with_warm_starting(mut self, enabled: bool) -> Self {
        self.warm_start = enabled;
        self
    }

    /// Adds a constraint to the solver.
    pub fn add_constraint(&mut self, constraint: Box<dyn Constraint2D>) {
        self.constraints.push(constraint);
        self.lambda_cache.push(0.0);
    }

    /// Removes a constraint at the specified index.
    ///
    /// # Panics
    ///
    /// Panics if index is out of bounds.
    pub fn remove_constraint(&mut self, index: usize) {
        self.constraints.remove(index);
        self.lambda_cache.remove(index);
    }

    /// Removes all constraints from the solver.
    pub fn clear(&mut self) {
        self.constraints.clear();
        self.lambda_cache.clear();
    }

    /// Returns the number of constraints in the solver.
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// Resets the warm start cache.
    ///
    /// Call this when constraints have changed significantly
    /// or when starting a new simulation.
    pub fn reset_warm_start(&mut self) {
        for lambda in &mut self.lambda_cache {
            *lambda = 0.0;
        }
        for constraint in &mut self.constraints {
            constraint.set_lambda(0.0);
        }
    }

    /// Solves all constraints.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    ///
    /// # Returns
    ///
    /// * `Ok(SolverResult)` - Result containing iteration count, convergence status, and max error
    /// * `Err(PhysicsError)` - If any constraint fails to solve
    pub fn solve(&mut self, dt: f64) -> Result<SolverResult, PhysicsError> {
        if self.constraints.is_empty() {
            return Ok(SolverResult {
                iterations: 0,
                converged: true,
                max_error: 0.0,
            });
        }

        // Apply warm starting if enabled
        if self.warm_start {
            for (i, constraint) in self.constraints.iter_mut().enumerate() {
                constraint.set_lambda(self.lambda_cache[i]);
            }
        }

        let mut iterations = 0;
        let mut converged = false;
        let mut max_error = 0.0;

        for iter in 0..self.max_iterations {
            iterations = iter + 1;
            max_error = 0.0;

            // Solve each constraint
            for constraint in &mut self.constraints {
                constraint.solve(dt)?;
                let error = constraint.calculate_error();
                if error > max_error {
                    max_error = error;
                }
            }

            // Check for convergence
            if max_error < self.tolerance {
                converged = true;
                break;
            }
        }

        // Cache lambda values for warm starting
        if self.warm_start {
            for (i, constraint) in self.constraints.iter().enumerate() {
                self.lambda_cache[i] = constraint.get_lambda();
            }
        }

        Ok(SolverResult {
            iterations,
            converged,
            max_error,
        })
    }
}

/// Unified 3D constraint solver with optional warm starting.
///
/// This solver uses Gauss-Seidel iteration to solve a collection of 3D
/// constraints. Warm starting can be enabled to improve convergence by
/// using cached impulse values from the previous frame.
///
/// # Examples
///
/// ```rust,ignore
/// use rs_physics::constraints::{UnifiedSolver3D, Constraint3D};
///
/// let mut solver = UnifiedSolver3D::new(10, 0.001).unwrap();
/// solver.add_constraint(Box::new(my_joint));
/// let result = solver.solve(0.016).unwrap();
/// println!("Converged: {}, iterations: {}", result.converged, result.iterations);
/// ```
pub struct UnifiedSolver3D {
    /// Constraints managed by this solver.
    constraints: Vec<Box<dyn Constraint3D>>,
    /// Maximum iterations per solve.
    max_iterations: usize,
    /// Convergence tolerance.
    tolerance: f64,
    /// Whether warm starting is enabled.
    warm_start: bool,
    /// Cached lambda values for warm starting.
    lambda_cache: Vec<f64>,
}

impl UnifiedSolver3D {
    /// Creates a new UnifiedSolver3D.
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - Maximum number of solver iterations (must be > 0)
    /// * `tolerance` - Convergence tolerance (must be > 0)
    ///
    /// # Returns
    ///
    /// * `Ok(UnifiedSolver3D)` - Valid solver
    /// * `Err(PhysicsError)` - If parameters are invalid
    pub fn new(max_iterations: usize, tolerance: f64) -> Result<Self, PhysicsError> {
        if max_iterations == 0 {
            return Err(PhysicsError::CalculationError(
                "max_iterations must be greater than 0".to_string(),
            ));
        }
        if tolerance <= 0.0 {
            return Err(PhysicsError::CalculationError(
                "tolerance must be positive".to_string(),
            ));
        }

        Ok(Self {
            constraints: Vec::new(),
            max_iterations,
            tolerance,
            warm_start: false,
            lambda_cache: Vec::new(),
        })
    }

    /// Enables or disables warm starting.
    ///
    /// Warm starting uses cached impulse values from previous frames
    /// to improve solver convergence.
    pub fn with_warm_starting(mut self, enabled: bool) -> Self {
        self.warm_start = enabled;
        self
    }

    /// Adds a constraint to the solver.
    pub fn add_constraint(&mut self, constraint: Box<dyn Constraint3D>) {
        self.constraints.push(constraint);
        self.lambda_cache.push(0.0);
    }

    /// Removes a constraint at the specified index.
    ///
    /// # Panics
    ///
    /// Panics if index is out of bounds.
    pub fn remove_constraint(&mut self, index: usize) {
        self.constraints.remove(index);
        self.lambda_cache.remove(index);
    }

    /// Removes all constraints from the solver.
    pub fn clear(&mut self) {
        self.constraints.clear();
        self.lambda_cache.clear();
    }

    /// Returns the number of constraints in the solver.
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// Resets the warm start cache.
    ///
    /// Call this when constraints have changed significantly
    /// or when starting a new simulation.
    pub fn reset_warm_start(&mut self) {
        for lambda in &mut self.lambda_cache {
            *lambda = 0.0;
        }
        for constraint in &mut self.constraints {
            constraint.set_lambda(0.0);
        }
    }

    /// Solves all constraints.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    ///
    /// # Returns
    ///
    /// * `Ok(SolverResult)` - Result containing iteration count, convergence status, and max error
    /// * `Err(PhysicsError)` - If any constraint fails to solve
    pub fn solve(&mut self, dt: f64) -> Result<SolverResult, PhysicsError> {
        if self.constraints.is_empty() {
            return Ok(SolverResult {
                iterations: 0,
                converged: true,
                max_error: 0.0,
            });
        }

        // Apply warm starting if enabled
        if self.warm_start {
            for (i, constraint) in self.constraints.iter_mut().enumerate() {
                constraint.set_lambda(self.lambda_cache[i]);
            }
        }

        let mut iterations = 0;
        let mut converged = false;
        let mut max_error = 0.0;

        for iter in 0..self.max_iterations {
            iterations = iter + 1;
            max_error = 0.0;

            // Solve each constraint
            for constraint in &mut self.constraints {
                constraint.solve(dt)?;
                let error = constraint.calculate_error();
                if error > max_error {
                    max_error = error;
                }
            }

            // Check for convergence
            if max_error < self.tolerance {
                converged = true;
                break;
            }
        }

        // Cache lambda values for warm starting
        if self.warm_start {
            for (i, constraint) in self.constraints.iter().enumerate() {
                self.lambda_cache[i] = constraint.get_lambda();
            }
        }

        Ok(SolverResult {
            iterations,
            converged,
            max_error,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test constraint for 2D
    struct TestConstraint2D {
        error: f64,
        lambda: f64,
        solve_reduces_error: bool,
    }

    impl TestConstraint2D {
        fn new(error: f64) -> Self {
            Self {
                error,
                lambda: 0.0,
                solve_reduces_error: true,
            }
        }

        fn with_fixed_error(error: f64) -> Self {
            Self {
                error,
                lambda: 0.0,
                solve_reduces_error: false,
            }
        }
    }

    impl Constraint2D for TestConstraint2D {
        fn solve(&mut self, _dt: f64) -> Result<(), PhysicsError> {
            if self.solve_reduces_error {
                self.error *= 0.5; // Reduce error by half each iteration
            }
            self.lambda += 1.0;
            Ok(())
        }

        fn calculate_error(&self) -> f64 {
            self.error
        }

        fn get_lambda(&self) -> f64 {
            self.lambda
        }

        fn set_lambda(&mut self, lambda: f64) {
            self.lambda = lambda;
        }
    }

    // Simple test constraint for 3D
    struct TestConstraint3D {
        error: f64,
        lambda: f64,
        solve_reduces_error: bool,
    }

    impl TestConstraint3D {
        fn new(error: f64) -> Self {
            Self {
                error,
                lambda: 0.0,
                solve_reduces_error: true,
            }
        }

        fn with_fixed_error(error: f64) -> Self {
            Self {
                error,
                lambda: 0.0,
                solve_reduces_error: false,
            }
        }
    }

    impl Constraint3D for TestConstraint3D {
        fn solve(&mut self, _dt: f64) -> Result<(), PhysicsError> {
            if self.solve_reduces_error {
                self.error *= 0.5;
            }
            self.lambda += 1.0;
            Ok(())
        }

        fn calculate_error(&self) -> f64 {
            self.error
        }

        fn get_lambda(&self) -> f64 {
            self.lambda
        }

        fn set_lambda(&mut self, lambda: f64) {
            self.lambda = lambda;
        }
    }

    // 2D Solver Tests

    #[test]
    fn test_solver_2d_creation() {
        let solver = UnifiedSolver2D::new(10, 0.001);
        assert!(solver.is_ok());
        let solver = solver.unwrap();
        assert_eq!(solver.constraint_count(), 0);
    }

    #[test]
    fn test_solver_2d_invalid_iterations() {
        let solver = UnifiedSolver2D::new(0, 0.001);
        assert!(solver.is_err());
    }

    #[test]
    fn test_solver_2d_invalid_tolerance() {
        let solver = UnifiedSolver2D::new(10, 0.0);
        assert!(solver.is_err());
        let solver = UnifiedSolver2D::new(10, -0.001);
        assert!(solver.is_err());
    }

    #[test]
    fn test_solver_2d_add_constraint() {
        let mut solver = UnifiedSolver2D::new(10, 0.001).unwrap();
        solver.add_constraint(Box::new(TestConstraint2D::new(1.0)));
        assert_eq!(solver.constraint_count(), 1);
        solver.add_constraint(Box::new(TestConstraint2D::new(2.0)));
        assert_eq!(solver.constraint_count(), 2);
    }

    #[test]
    fn test_solver_2d_remove_constraint() {
        let mut solver = UnifiedSolver2D::new(10, 0.001).unwrap();
        solver.add_constraint(Box::new(TestConstraint2D::new(1.0)));
        solver.add_constraint(Box::new(TestConstraint2D::new(2.0)));
        assert_eq!(solver.constraint_count(), 2);
        solver.remove_constraint(0);
        assert_eq!(solver.constraint_count(), 1);
    }

    #[test]
    fn test_solver_2d_clear() {
        let mut solver = UnifiedSolver2D::new(10, 0.001).unwrap();
        solver.add_constraint(Box::new(TestConstraint2D::new(1.0)));
        solver.add_constraint(Box::new(TestConstraint2D::new(2.0)));
        solver.clear();
        assert_eq!(solver.constraint_count(), 0);
    }

    #[test]
    fn test_solver_2d_empty_solve() {
        let mut solver = UnifiedSolver2D::new(10, 0.001).unwrap();
        let result = solver.solve(0.016).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.max_error, 0.0);
    }

    #[test]
    fn test_solver_2d_single_constraint() {
        let mut solver = UnifiedSolver2D::new(10, 0.001).unwrap();
        solver.add_constraint(Box::new(TestConstraint2D::new(1.0)));
        let result = solver.solve(0.016).unwrap();
        assert!(result.converged);
        assert!(result.max_error < 0.001);
    }

    #[test]
    fn test_solver_2d_multiple_constraints() {
        let mut solver = UnifiedSolver2D::new(20, 0.001).unwrap();
        solver.add_constraint(Box::new(TestConstraint2D::new(1.0)));
        solver.add_constraint(Box::new(TestConstraint2D::new(2.0)));
        solver.add_constraint(Box::new(TestConstraint2D::new(0.5)));
        let result = solver.solve(0.016).unwrap();
        assert!(result.converged);
        assert!(result.max_error < 0.001);
    }

    #[test]
    fn test_solver_2d_convergence() {
        let mut solver = UnifiedSolver2D::new(10, 0.001).unwrap();
        solver.add_constraint(Box::new(TestConstraint2D::new(0.5)));
        let result = solver.solve(0.016).unwrap();
        assert!(result.converged);
        assert!(result.iterations <= 10);
    }

    #[test]
    fn test_solver_2d_early_termination() {
        let mut solver = UnifiedSolver2D::new(100, 0.01).unwrap();
        // Small error should converge quickly
        solver.add_constraint(Box::new(TestConstraint2D::new(0.1)));
        let result = solver.solve(0.016).unwrap();
        assert!(result.converged);
        assert!(result.iterations < 100, "Should terminate early");
    }

    #[test]
    fn test_solver_2d_no_convergence() {
        let mut solver = UnifiedSolver2D::new(5, 0.001).unwrap();
        // Large error with few iterations won't converge
        solver.add_constraint(Box::new(TestConstraint2D::with_fixed_error(10.0)));
        let result = solver.solve(0.016).unwrap();
        assert!(!result.converged);
        assert_eq!(result.iterations, 5);
    }

    #[test]
    fn test_solver_2d_warm_starting() {
        let mut solver = UnifiedSolver2D::new(10, 0.001).unwrap().with_warm_starting(true);
        solver.add_constraint(Box::new(TestConstraint2D::new(1.0)));

        // First solve
        solver.solve(0.016).unwrap();

        // Second solve should use cached lambda
        let result = solver.solve(0.016).unwrap();
        assert!(result.converged);
    }

    #[test]
    fn test_solver_2d_warm_start_cache_reset() {
        let mut solver = UnifiedSolver2D::new(10, 0.001).unwrap().with_warm_starting(true);
        solver.add_constraint(Box::new(TestConstraint2D::new(1.0)));
        solver.solve(0.016).unwrap();

        // Reset cache
        solver.reset_warm_start();

        // Lambda should be reset
        assert_eq!(solver.lambda_cache[0], 0.0);
    }

    // 3D Solver Tests

    #[test]
    fn test_solver_3d_creation() {
        let solver = UnifiedSolver3D::new(10, 0.001);
        assert!(solver.is_ok());
        let solver = solver.unwrap();
        assert_eq!(solver.constraint_count(), 0);
    }

    #[test]
    fn test_solver_3d_invalid_iterations() {
        let solver = UnifiedSolver3D::new(0, 0.001);
        assert!(solver.is_err());
    }

    #[test]
    fn test_solver_3d_invalid_tolerance() {
        let solver = UnifiedSolver3D::new(10, 0.0);
        assert!(solver.is_err());
        let solver = UnifiedSolver3D::new(10, -0.001);
        assert!(solver.is_err());
    }

    #[test]
    fn test_solver_3d_add_constraint() {
        let mut solver = UnifiedSolver3D::new(10, 0.001).unwrap();
        solver.add_constraint(Box::new(TestConstraint3D::new(1.0)));
        assert_eq!(solver.constraint_count(), 1);
        solver.add_constraint(Box::new(TestConstraint3D::new(2.0)));
        assert_eq!(solver.constraint_count(), 2);
    }

    #[test]
    fn test_solver_3d_remove_constraint() {
        let mut solver = UnifiedSolver3D::new(10, 0.001).unwrap();
        solver.add_constraint(Box::new(TestConstraint3D::new(1.0)));
        solver.add_constraint(Box::new(TestConstraint3D::new(2.0)));
        assert_eq!(solver.constraint_count(), 2);
        solver.remove_constraint(0);
        assert_eq!(solver.constraint_count(), 1);
    }

    #[test]
    fn test_solver_3d_clear() {
        let mut solver = UnifiedSolver3D::new(10, 0.001).unwrap();
        solver.add_constraint(Box::new(TestConstraint3D::new(1.0)));
        solver.add_constraint(Box::new(TestConstraint3D::new(2.0)));
        solver.clear();
        assert_eq!(solver.constraint_count(), 0);
    }

    #[test]
    fn test_solver_3d_empty_solve() {
        let mut solver = UnifiedSolver3D::new(10, 0.001).unwrap();
        let result = solver.solve(0.016).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.max_error, 0.0);
    }

    #[test]
    fn test_solver_3d_single_constraint() {
        let mut solver = UnifiedSolver3D::new(10, 0.001).unwrap();
        solver.add_constraint(Box::new(TestConstraint3D::new(1.0)));
        let result = solver.solve(0.016).unwrap();
        assert!(result.converged);
        assert!(result.max_error < 0.001);
    }

    #[test]
    fn test_solver_3d_multiple_constraints() {
        let mut solver = UnifiedSolver3D::new(20, 0.001).unwrap();
        solver.add_constraint(Box::new(TestConstraint3D::new(1.0)));
        solver.add_constraint(Box::new(TestConstraint3D::new(2.0)));
        solver.add_constraint(Box::new(TestConstraint3D::new(0.5)));
        let result = solver.solve(0.016).unwrap();
        assert!(result.converged);
        assert!(result.max_error < 0.001);
    }

    #[test]
    fn test_solver_3d_convergence() {
        let mut solver = UnifiedSolver3D::new(10, 0.001).unwrap();
        solver.add_constraint(Box::new(TestConstraint3D::new(0.5)));
        let result = solver.solve(0.016).unwrap();
        assert!(result.converged);
        assert!(result.iterations <= 10);
    }

    #[test]
    fn test_solver_3d_early_termination() {
        let mut solver = UnifiedSolver3D::new(100, 0.01).unwrap();
        solver.add_constraint(Box::new(TestConstraint3D::new(0.1)));
        let result = solver.solve(0.016).unwrap();
        assert!(result.converged);
        assert!(result.iterations < 100, "Should terminate early");
    }

    #[test]
    fn test_solver_3d_no_convergence() {
        let mut solver = UnifiedSolver3D::new(5, 0.001).unwrap();
        solver.add_constraint(Box::new(TestConstraint3D::with_fixed_error(10.0)));
        let result = solver.solve(0.016).unwrap();
        assert!(!result.converged);
        assert_eq!(result.iterations, 5);
    }

    #[test]
    fn test_solver_3d_warm_starting() {
        let mut solver = UnifiedSolver3D::new(10, 0.001).unwrap().with_warm_starting(true);
        solver.add_constraint(Box::new(TestConstraint3D::new(1.0)));

        // First solve
        solver.solve(0.016).unwrap();

        // Second solve should use cached lambda
        let result = solver.solve(0.016).unwrap();
        assert!(result.converged);
    }

    #[test]
    fn test_solver_3d_warm_start_cache_reset() {
        let mut solver = UnifiedSolver3D::new(10, 0.001).unwrap().with_warm_starting(true);
        solver.add_constraint(Box::new(TestConstraint3D::new(1.0)));
        solver.solve(0.016).unwrap();

        // Reset cache
        solver.reset_warm_start();

        // Lambda should be reset
        assert_eq!(solver.lambda_cache[0], 0.0);
    }
}
