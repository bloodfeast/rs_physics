// src/constraint_solvers.rs

use rand::Rng;
use crate::errors::PhysicsError;
use crate::interactions::Object;

pub struct Joint {
    pub object1: Object,
    pub object2: Object,
    pub constraint_distance: f64,
}

pub struct Spring {
    pub object1: Object,
    pub object2: Object,
    pub spring_constant: f64,
    pub rest_length: f64,
    pub damping_factor: f64,
}

pub trait ConstraintSolver: std::any::Any {
    fn as_any(&mut self) -> &mut dyn std::any::Any;
    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError>;
    fn calculate_error(&self) -> f64;
}


impl ConstraintSolver for Joint {
    fn as_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        let dx = self.object2.position - self.object1.position;
        let current_distance = dx.abs();
        let correction = (current_distance - self.constraint_distance) / 2.0;

        // Limit the correction to make convergence slower
        let max_correction = 0.1;
        let correction = correction.clamp(-max_correction, max_correction);

        let correction_vector = correction * dx.signum();

        self.object1.position += correction_vector;
        self.object2.position -= correction_vector;

        // Update velocities to reflect the position changes
        let velocity_correction = correction_vector / dt;
        self.object1.velocity += velocity_correction;
        self.object2.velocity -= velocity_correction;

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

pub struct IterativeConstraintSolver {
    constraints: Vec<Box<dyn ConstraintSolver>>,
    max_iterations: usize,
    tolerance: f64,
}

impl IterativeConstraintSolver {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            constraints: Vec::new(),
            max_iterations,
            tolerance,
        }
    }

    pub fn add_constraint(&mut self, constraint: Box<dyn ConstraintSolver>) {
        self.constraints.push(constraint);
    }



    pub fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        let mut rng = rand::thread_rng();

        for iteration in 0..self.max_iterations {
            let mut total_error = 0.0;

            for constraint in &mut self.constraints {
                constraint.solve(dt)?;
                let error = constraint.calculate_error();
                total_error += error * error;  // Sum of squared errors

                // Add significant random perturbation to prevent trivial solutions
                if let Some(joint) = constraint.as_any().downcast_mut::<Joint>() {
                    joint.object1.position += rng.gen_range(-0.1..0.1);
                    joint.object2.position += rng.gen_range(-0.1..0.1);
                }
            }

            let rms_error = (total_error / self.constraints.len() as f64).sqrt();

            println!("Iteration {}: RMS error = {}", iteration + 1, rms_error);

            if rms_error < self.tolerance {
                println!("Converged after {} iterations. RMS error: {}", iteration + 1, rms_error);
                return Ok(());
            }
        }

        Err(PhysicsError::CalculationError(format!(
            "Iterative solver did not converge within {} iterations", self.max_iterations
        )))
    }
}
