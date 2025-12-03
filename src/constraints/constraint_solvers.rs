//! Constraint solvers for physics simulations.
//!
//! This module provides constraint types and solvers for maintaining
//! physical relationships between objects during simulation.

use crate::utils::PhysicsError;
use crate::models::Object;

/// A rigid joint constraint between two objects.
///
/// Maintains a fixed distance between two objects by applying position
/// and velocity corrections each timestep. Uses a position-based dynamics
/// approach with velocity projection.
///
/// # Fields
///
/// * `object1` - First object in the joint
/// * `object2` - Second object in the joint
/// * `constraint_distance` - Target distance to maintain between objects (in meters)
///
/// # Examples
///
/// ```rust,ignore
/// use rs_physics::constraints::Joint;
/// use rs_physics::models::Object;
///
/// let joint = Joint {
///     object1: Object::new(1.0, 0.0, 0.0),  // 1kg at x=0
///     object2: Object::new(1.0, 5.0, 0.0),  // 1kg at x=5
///     constraint_distance: 5.0,             // maintain 5m distance
/// };
/// ```
///
/// # Notes
///
/// - Corrections are clamped to ±0.1 units per solve to ensure stability
/// - Both position and velocity are corrected to maintain constraint
pub struct Joint {
    /// First object connected by the joint
    pub object1: Object,
    /// Second object connected by the joint
    pub object2: Object,
    /// Target distance to maintain between objects (meters)
    pub constraint_distance: f64,
    /// Baumgarte stabilization factor (0.0 to 1.0, default 0.2)
    pub baumgarte: f64,
}

impl Joint {
    /// Creates a new Joint constraint between two objects.
    ///
    /// # Arguments
    ///
    /// * `object1` - First object in the joint
    /// * `object2` - Second object in the joint
    /// * `constraint_distance` - Target distance to maintain (must be >= 0)
    ///
    /// # Returns
    ///
    /// * `Ok(Joint)` - Valid joint constraint
    /// * `Err(PhysicsError)` - If constraint_distance is negative
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use rs_physics::constraints::Joint;
    /// use rs_physics::models::Object;
    ///
    /// let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();
    /// let obj2 = Object::new(1.0, 5.0, 0.0).unwrap();
    /// let joint = Joint::new(obj1, obj2, 5.0).expect("Valid joint");
    /// ```
    pub fn new(object1: Object, object2: Object, constraint_distance: f64) -> Result<Self, PhysicsError> {
        if constraint_distance < 0.0 {
            return Err(PhysicsError::InvalidDistance);
        }
        Ok(Self {
            object1,
            object2,
            constraint_distance,
            baumgarte: 0.2, // Default Baumgarte stabilization factor
        })
    }

    /// Sets the Baumgarte stabilization factor.
    ///
    /// # Arguments
    ///
    /// * `factor` - Stabilization factor between 0.0 and 1.0
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_baumgarte(mut self, factor: f64) -> Self {
        self.baumgarte = factor;
        self
    }
}

/// A spring constraint between two objects.
///
/// Models an elastic connection with configurable spring constant and damping.
/// Implements Hooke's law with viscous damping for realistic spring behavior.
///
/// # Physics Model
///
/// The spring force is calculated as:
/// ```text
/// F_spring = k × (current_length - rest_length)
/// F_damping = c × relative_velocity
/// F_total = F_spring + F_damping
/// ```
///
/// Where:
/// - `k` is the spring constant (N/m)
/// - `c` is the damping factor (N·s/m)
///
/// # Fields
///
/// * `object1` - First object connected to spring
/// * `object2` - Second object connected to spring
/// * `spring_constant` - Stiffness of the spring (N/m)
/// * `rest_length` - Natural length of the spring (m)
/// * `damping_factor` - Viscous damping coefficient (N·s/m)
///
/// # Examples
///
/// ```rust,ignore
/// use rs_physics::constraints::Spring;
/// use rs_physics::models::Object;
///
/// // Create a spring with k=100 N/m, rest length 2m, damping 0.5
/// let spring = Spring {
///     object1: Object::new(1.0, 0.0, 0.0),
///     object2: Object::new(1.0, 3.0, 0.0),  // stretched 1m beyond rest
///     spring_constant: 100.0,
///     rest_length: 2.0,
///     damping_factor: 0.5,
/// };
/// ```
///
/// # Notes
///
/// - Higher spring constants lead to stiffer springs but may require smaller timesteps
/// - Damping helps prevent oscillation and improves stability
/// - For critical damping, use: `damping_factor = 2 × √(spring_constant × reduced_mass)`
pub struct Spring {
    /// First object connected to the spring
    pub object1: Object,
    /// Second object connected to the spring
    pub object2: Object,
    /// Spring stiffness constant (N/m)
    pub spring_constant: f64,
    /// Natural/rest length of the spring (m)
    pub rest_length: f64,
    /// Viscous damping coefficient (N·s/m)
    pub damping_factor: f64,
}

impl Spring {
    /// Creates a new Spring constraint between two objects.
    ///
    /// # Arguments
    ///
    /// * `object1` - First object connected to the spring
    /// * `object2` - Second object connected to the spring
    /// * `spring_constant` - Spring stiffness (N/m, must be > 0)
    /// * `rest_length` - Natural length of the spring (m, must be >= 0)
    /// * `damping_factor` - Viscous damping coefficient (N·s/m, must be >= 0)
    ///
    /// # Returns
    ///
    /// * `Ok(Spring)` - Valid spring constraint
    /// * `Err(PhysicsError)` - If any parameter is invalid
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use rs_physics::constraints::Spring;
    /// use rs_physics::models::Object;
    ///
    /// let obj1 = Object::new(1.0, 0.0, 0.0).unwrap();
    /// let obj2 = Object::new(1.0, 3.0, 0.0).unwrap();
    /// let spring = Spring::new(obj1, obj2, 100.0, 2.0, 0.5).expect("Valid spring");
    /// ```
    pub fn new(
        object1: Object,
        object2: Object,
        spring_constant: f64,
        rest_length: f64,
        damping_factor: f64,
    ) -> Result<Self, PhysicsError> {
        if spring_constant <= 0.0 {
            return Err(PhysicsError::InvalidCoefficient);
        }
        if rest_length < 0.0 {
            return Err(PhysicsError::InvalidDistance);
        }
        if damping_factor < 0.0 {
            return Err(PhysicsError::InvalidCoefficient);
        }
        Ok(Self {
            object1,
            object2,
            spring_constant,
            rest_length,
            damping_factor,
        })
    }

    /// Calculates the critical damping coefficient for this spring.
    ///
    /// Critical damping is the minimum damping that prevents oscillation.
    /// For a spring-mass system: c_critical = 2 × √(k × m_reduced)
    ///
    /// # Returns
    ///
    /// The critical damping coefficient in N·s/m
    pub fn critical_damping(&self) -> f64 {
        let m_reduced = (self.object1.mass * self.object2.mass)
            / (self.object1.mass + self.object2.mass);
        2.0 * (self.spring_constant * m_reduced).sqrt()
    }

    /// Returns the damping ratio (ζ) for this spring.
    ///
    /// - ζ < 1: Underdamped (oscillates)
    /// - ζ = 1: Critically damped (no oscillation, fastest settling)
    /// - ζ > 1: Overdamped (slow return, no oscillation)
    pub fn damping_ratio(&self) -> f64 {
        self.damping_factor / self.critical_damping()
    }
}

/// Trait for constraint solvers.
///
/// Defines the interface that all constraint types must implement to work
/// with the iterative constraint solver.
///
/// # Required Methods
///
/// * `solve` - Apply corrections to satisfy the constraint
/// * `calculate_error` - Measure how far the constraint is from being satisfied
/// * `as_any` - Enable downcasting for type-specific operations
///
/// # Implementing a Custom Constraint
///
/// ```rust,ignore
/// use rs_physics::constraints::ConstraintSolver;
/// use rs_physics::utils::PhysicsError;
///
/// struct MyConstraint { /* fields */ }
///
/// impl ConstraintSolver for MyConstraint {
///     fn as_any(&mut self) -> &mut dyn std::any::Any { self }
///
///     fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
///         // Apply corrections to satisfy constraint
///         Ok(())
///     }
///
///     fn calculate_error(&self) -> f64 {
///         // Return distance from constraint satisfaction (0 = satisfied)
///         0.0
///     }
/// }
/// ```
pub trait ConstraintSolver: std::any::Any {
    /// Returns a mutable reference to self as Any for downcasting.
    fn as_any(&mut self) -> &mut dyn std::any::Any;

    /// Solves the constraint for one iteration.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Constraint solved successfully
    /// * `Err(PhysicsError)` - If constraint cannot be solved
    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError>;

    /// Calculates the current constraint error.
    ///
    /// # Returns
    ///
    /// The magnitude of constraint violation (0.0 = perfectly satisfied)
    fn calculate_error(&self) -> f64;
}


impl ConstraintSolver for Joint {
    fn as_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        let dx = self.object2.position - self.object1.position;
        let current_distance = dx.abs();
        let error = current_distance - self.constraint_distance;

        if current_distance < 1e-10 {
            // Objects at same position, can't determine direction
            return Ok(());
        }

        // Calculate inverse masses for mass-weighted corrections
        let inv_mass1 = if self.object1.mass.is_infinite() { 0.0 } else { 1.0 / self.object1.mass };
        let inv_mass2 = if self.object2.mass.is_infinite() { 0.0 } else { 1.0 / self.object2.mass };
        let total_inv_mass = inv_mass1 + inv_mass2;

        if total_inv_mass < 1e-10 {
            // Both objects have infinite mass, can't correct
            return Ok(());
        }

        // Baumgarte stabilization: bias velocity toward zero error
        let bias = self.baumgarte * error / dt;

        // Calculate relative velocity along constraint direction
        let direction = dx.signum();
        let relative_velocity = (self.object2.velocity - self.object1.velocity) * direction;

        // Velocity constraint: relative_velocity + bias = 0
        // Impulse magnitude: lambda = -(relative_velocity + bias) / total_inv_mass
        let lambda = -(relative_velocity + bias) / total_inv_mass;

        // Clamp impulse for stability
        let max_impulse = 0.1 / dt;
        let lambda = lambda.clamp(-max_impulse, max_impulse);

        // Apply velocity corrections (mass-weighted)
        self.object1.velocity -= lambda * inv_mass1 * direction;
        self.object2.velocity += lambda * inv_mass2 * direction;

        // Position correction (mass-weighted)
        let max_correction = 0.1;
        let position_correction = error.clamp(-max_correction * 2.0, max_correction * 2.0);

        self.object1.position += position_correction * (inv_mass1 / total_inv_mass) * direction;
        self.object2.position -= position_correction * (inv_mass2 / total_inv_mass) * direction;

        Ok(())
    }

    fn calculate_error(&self) -> f64 {
        let dx = self.object2.position - self.object1.position;
        let current_distance = dx.abs();
        (current_distance - self.constraint_distance).abs()
    }
}
impl ConstraintSolver for Spring {
    fn as_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        let dx = self.object2.position - self.object1.position;
        let current_length = dx.abs();
        let stretch = current_length - self.rest_length;

        // Calculate spring force
        let spring_force = self.spring_constant * stretch;

        // Calculate relative velocity
        let relative_velocity = self.object2.velocity - self.object1.velocity;

        // Calculate damping force
        let damping_force = self.damping_factor * relative_velocity;

        // Total force
        let total_force = (spring_force + damping_force) * dx.signum();

        // Apply forces
        let acceleration1 = total_force / self.object1.mass;
        let acceleration2 = -total_force / self.object2.mass;

        self.object1.velocity += acceleration1 * dt;
        self.object2.velocity += acceleration2 * dt;

        self.object1.position += self.object1.velocity * dt;
        self.object2.position += self.object2.velocity * dt;

        Ok(())
    }
    fn calculate_error(&self) -> f64 {
        let dx = self.object2.position - self.object1.position;
        let current_length = dx.abs();
        (current_length - self.rest_length).abs()
    }
}

/// An iterative constraint solver using the Gauss-Seidel method.
///
/// Solves multiple constraints simultaneously by iterating until convergence
/// or reaching the maximum iteration limit.
///
/// # Algorithm
///
/// 1. For each iteration:
///    - Solve each constraint sequentially
///    - Calculate RMS (root mean square) error
///    - If RMS error < tolerance, converged
/// 2. Return error if max iterations reached without convergence
///
/// # Fields
///
/// * `constraints` - Vector of constraint objects to solve
/// * `max_iterations` - Maximum solver iterations before giving up
/// * `tolerance` - RMS error threshold for convergence
///
/// # Examples
///
/// ```rust,ignore
/// use rs_physics::constraints::{IterativeConstraintSolver, Joint, Spring};
/// use rs_physics::models::Object;
///
/// // Create solver with 20 iterations max, 0.001 tolerance
/// let mut solver = IterativeConstraintSolver::new(20, 0.001);
///
/// // Add constraints
/// let joint = Joint {
///     object1: Object::new(1.0, 0.0, 0.0),
///     object2: Object::new(1.0, 5.0, 0.0),
///     constraint_distance: 5.0,
/// };
/// solver.add_constraint(Box::new(joint));
///
/// // Solve with 16ms timestep
/// match solver.solve(0.016) {
///     Ok(()) => println!("Constraints solved!"),
///     Err(e) => println!("Failed to converge: {}", e),
/// }
/// ```
///
/// # Performance
///
/// - Time complexity: O(max_iterations × num_constraints)
///
/// # Notes
///
/// - Lower tolerance requires more iterations but gives more accurate results
/// - If constraints are stiff (high spring constants), increase max_iterations
pub struct IterativeConstraintSolver {
    /// Collection of constraints to solve
    constraints: Vec<Box<dyn ConstraintSolver>>,
    /// Maximum number of iterations before returning error
    max_iterations: usize,
    /// RMS error threshold for convergence
    tolerance: f64,
}

impl IterativeConstraintSolver {
    /// Creates a new iterative constraint solver.
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - Maximum iterations before solver gives up
    /// * `tolerance` - RMS error threshold for convergence (smaller = more accurate)
    ///
    /// # Returns
    ///
    /// A new `IterativeConstraintSolver` with no constraints
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use rs_physics::constraints::IterativeConstraintSolver;
    ///
    /// // High accuracy solver
    /// let precise_solver = IterativeConstraintSolver::new(100, 0.0001);
    ///
    /// // Fast but less accurate solver
    /// let fast_solver = IterativeConstraintSolver::new(5, 0.01);
    /// ```
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            constraints: Vec::new(),
            max_iterations,
            tolerance,
        }
    }

    /// Adds a constraint to the solver.
    ///
    /// # Arguments
    ///
    /// * `constraint` - A boxed constraint implementing `ConstraintSolver`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use rs_physics::constraints::{IterativeConstraintSolver, Joint};
    ///
    /// let mut solver = IterativeConstraintSolver::new(10, 0.001);
    /// solver.add_constraint(Box::new(joint));
    /// solver.add_constraint(Box::new(spring));
    /// ```
    pub fn add_constraint(&mut self, constraint: Box<dyn ConstraintSolver>) {
        self.constraints.push(constraint);
    }

    /// Solves all constraints iteratively until convergence or max iterations.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Constraints converged within tolerance
    /// * `Err(PhysicsError)` - Failed to converge within max_iterations
    ///
    /// # Algorithm Details
    ///
    /// Uses Gauss-Seidel iteration with SIMD-accelerated error accumulation.
    /// Each constraint is solved sequentially, and the RMS error is calculated
    /// after each full iteration pass.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let result = solver.solve(0.016); // 60 FPS timestep
    /// if result.is_err() {
    ///     // Consider: increasing max_iterations, relaxing tolerance,
    ///     // or using smaller timesteps
    /// }
    /// ```
    /// Solves all constraints iteratively until convergence or max iterations.
    ///
    /// # Arguments
    ///
    /// * `dt` - Timestep in seconds
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Constraints converged within tolerance
    /// * `Err(PhysicsError)` - Failed to converge within max_iterations
    pub fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        if self.constraints.is_empty() {
            return Ok(());
        }

        for _iteration in 0..self.max_iterations {
            let mut total_error_squared = 0.0;

            for constraint in &mut self.constraints {
                constraint.solve(dt)?;
                let error = constraint.calculate_error();
                total_error_squared += error * error;
            }

            let rms_error = (total_error_squared / self.constraints.len() as f64).sqrt();

            if rms_error < self.tolerance {
                return Ok(());
            }
        }

        Err(PhysicsError::CalculationError(format!(
            "Iterative solver did not converge within {} iterations", self.max_iterations
        )))
    }

    /// Returns the number of constraints in the solver.
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// Clears all constraints from the solver.
    pub fn clear(&mut self) {
        self.constraints.clear();
    }
}
