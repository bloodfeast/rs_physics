// src/forces.rs

use crate::utils::PhysicsConstants;
use crate::models::{Direction2D, Direction3D, Object, Velocity2D, Velocity3D};

#[derive(Debug, Clone, Copy)]
pub enum Force {
    Gravity(f64),
    Drag { coefficient: f64, area: f64 },
    Spring { k: f64, x: f64 },
    Constant(f64),
    Thrust { magnitude: f64, angle: f64 },
}

impl Force {
    /// Applies the force to an object with the given mass and velocity.
    /// # Arguments
    /// * `mass` - The mass of the object in kilograms.
    /// * `velocity` - The velocity of the object in meters per second.
    ///
    /// # Returns
    /// The force applied to the object.
    ///
    /// # Example
    /// ```
    /// let force = rs_physics::forces::Force::Gravity(9.81);
    /// let mass = 10.0;
    /// let velocity = 5.0;
    /// let applied_force = force.apply(mass, velocity);
    /// ```
    pub fn apply(&self, mass: f64, velocity: f64) -> f64 {
        match *self {
            Force::Gravity(g) => mass * g,
            Force::Drag { coefficient, area } => -0.5 * coefficient * area * velocity.abs() * velocity,
            Force::Spring { k, x } => -k * x,
            Force::Constant(f) => f,
            Force::Thrust { magnitude, angle } => magnitude * angle.cos(),
        }
    }

    /// Applies the force to an object and returns the force components as a vector.
    /// This method is useful for 2D and 3D simulations.
    ///
    /// # Arguments
    /// * `mass` - The mass of the object in kilograms.
    /// * `velocity` - The speed of the object in meters per second.
    /// * `direction` - The direction of the object's motion.
    ///
    /// # Returns
    /// A tuple containing the x and y components of the force in Newtons.
    ///
    pub fn apply_vector(&self, mass: f64, velocity: f64, direction: &Direction2D) -> (f64, f64) {
        match *self {
            Force::Gravity(g) => (0.0, -g * mass),
            Force::Drag { coefficient, area } => {
                // Compute drag magnitude. Note: velocity^2 gives the proper scaling.
                // Note: I adjusted the drag constant to -0.25 to make the motion feel better.
                // the realistic formula is -0.5 * coefficient * area * velocity.powi(2)
                let drag_magnitude = -0.25 * coefficient * area * velocity.powi(2);
                // The drag force vector is drag_magnitude times the unit vector in the direction of motion.
                // Since drag opposes the velocity, we multiply by the normalized direction.
                (drag_magnitude * direction.x, drag_magnitude * direction.y)
            },
            Force::Spring { k, x } => (-k * x, 0.0),
            Force::Constant(f) => (f, f), // Not really directional, this is currently unused
            Force::Thrust { magnitude, angle } => {
                (magnitude * angle.cos(), magnitude * angle.sin())
            },
        }
    }

    /// Applies the force to an object and returns the force components as a 3D vector.
    /// This method is useful for 3D simulations.
    ///
    /// # Arguments
    /// * `mass` - The mass of the object in kilograms.
    /// * `velocity` - The speed of the object in meters per second.
    /// * `direction` - The direction of the object's motion in 3D space.
    ///
    /// # Returns
    /// A tuple containing the x, y, and z components of the force in Newtons.
    ///
    pub fn apply_vector_3d(&self, mass: f64, velocity: f64, direction: &Direction3D) -> (f64, f64, f64) {
        match *self {
            Force::Gravity(g) => (0.0, -g * mass, 0.0), // Assuming gravity acts in the -y direction
            Force::Drag { coefficient, area } => {
                let drag_magnitude = -0.25 * coefficient * area * velocity.powi(2);
                (drag_magnitude * direction.x, drag_magnitude * direction.y, drag_magnitude * direction.z)
            },
            Force::Spring { k, x } => (-k * x, 0.0, 0.0), // Assuming spring acts along the x axis
            Force::Constant(f) => (f, f, f), // Not really directional
            Force::Thrust { magnitude, angle } => {
                // For 3D, we'd need more angles, but for simplicity we'll assume thrust in xy plane
                (magnitude * angle.cos(), magnitude * angle.sin(), 0.0)
            },
        }
    }

    /// Applies the force directly to a 2D velocity vector.
    ///
    /// # Arguments
    /// * `mass` - The mass of the object in kilograms.
    /// * `velocity` - The velocity vector of the object.
    ///
    /// # Returns
    /// A tuple containing the x and y components of the force in Newtons.
    ///
    pub fn apply_to_velocity_2d(&self, mass: f64, velocity: &Velocity2D) -> (f64, f64) {
        match *self {
            Force::Gravity(g) => (0.0, -g * mass),
            Force::Drag { coefficient, area } => {
                let speed = velocity.magnitude();
                if speed == 0.0 {
                    return (0.0, 0.0);
                }
                let drag_magnitude = -0.25 * coefficient * area * speed.powi(2);
                (drag_magnitude * velocity.x / speed, drag_magnitude * velocity.y / speed)
            },
            Force::Spring { k, x } => (-k * x, 0.0),
            Force::Constant(f) => (f, f),
            Force::Thrust { magnitude, angle } => {
                (magnitude * angle.cos(), magnitude * angle.sin())
            },
        }
    }

    /// Applies the force directly to a 3D velocity vector.
    ///
    /// # Arguments
    /// * `mass` - The mass of the object in kilograms.
    /// * `velocity` - The velocity vector of the object in 3D space.
    ///
    /// # Returns
    /// A tuple containing the x, y, and z components of the force in Newtons.
    ///
    pub fn apply_to_velocity_3d(&self, mass: f64, velocity: &Velocity3D) -> (f64, f64, f64) {
        match *self {
            Force::Gravity(g) => (0.0, -g * mass, 0.0),
            Force::Drag { coefficient, area } => {
                let speed = velocity.magnitude();
                if speed == 0.0 {
                    return (0.0, 0.0, 0.0);
                }
                let drag_magnitude = -0.25 * coefficient * area * speed.powi(2);
                (
                    drag_magnitude * velocity.x / speed,
                    drag_magnitude * velocity.y / speed,
                    drag_magnitude * velocity.z / speed
                )
            },
            Force::Spring { k, x } => (-k * x, 0.0, 0.0),
            Force::Constant(f) => (f, f, f),
            Force::Thrust { magnitude, angle } => {
                (magnitude * angle.cos(), magnitude * angle.sin(), 0.0)
            },
        }
    }
}

pub struct PhysicsSystem {
    pub objects: Vec<Object>,
    pub constants: PhysicsConstants,
}
impl PhysicsSystem {

    /// Creates a new `PhysicsSystem` with the given physical constants.
    /// # Arguments
    /// * `constants` - An instance of `PhysicsConstants` containing the physical constants for the simulation.
    ///
    /// # Returns
    /// A new instance of `PhysicsSystem`.
    ///
    /// # Example
    /// ```
    /// let system = rs_physics::forces::PhysicsSystem::new(rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS);
    /// ```
    pub fn new(constants: PhysicsConstants) -> Self {
        PhysicsSystem {
            objects: Vec::new(),
            constants,
        }
    }

    /// Adds a new `Object` to the physics system.
    /// # Arguments
    /// * `object` - An instance of `Object` to be added to the system.
    ///
    /// # Example
    /// ```
    /// let object = rs_physics::models::Object::new(10.0, 5.0, 0.0).unwrap();
    /// let mut system = rs_physics::forces::PhysicsSystem::new(rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS);
    /// system.add_object(object);
    /// ```
    pub fn add_object(&mut self, object: Object) {
        self.objects.push(object);
    }

    /// Updates the state of all objects in the system over a time step `dt`.
    ///
    /// This method calculates the total force on each object, computes the resulting
    /// acceleration, and updates the object's velocity and position.
    /// # Arguments
    ///
    /// * `dt` - The time step duration in seconds.
    ///
    /// # Example
    /// ```
    /// let mut system = rs_physics::forces::PhysicsSystem::new(rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS);
    /// system.update(0.01); // Update the system for a 10ms time step
    /// ```
    pub fn update(&mut self, dt: f64) {
        for obj in &mut self.objects {
            let total_force = obj.forces.iter()
                .map(|force| force.apply(obj.mass, obj.velocity))
                .sum::<f64>();

            let acceleration = total_force / obj.mass;
            let initial_velocity = obj.velocity;
            obj.velocity += acceleration * dt;
            obj.position += 0.5 * (initial_velocity + obj.velocity) * dt;
        }
    }

    /// Applies gravitational force to all objects in the system.
    /// This method adds a `Force::Gravity` to each object in the system,
    /// using the gravitational constant from `self.constants`.
    /// # Example
    ///
    /// ```
    /// let mut system = rs_physics::forces::PhysicsSystem::new(rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS);
    /// system.apply_gravity();
    /// ```
    pub fn apply_gravity(&mut self) {
        let gravity = Force::Gravity(self.constants.gravity);
        for obj in &mut self.objects {
            obj.add_force(gravity);
        }
    }

    /// Applies drag force to all objects in the system with the given drag coefficient and area.
    /// # Arguments
    /// * `drag_coefficient` - The drag coefficient to be used in the drag force calculation.
    /// * `area` - The reference area to be used in the drag force calculation.
    ///
    /// # Example
    /// ```
    /// let mut system = rs_physics::forces::PhysicsSystem::new(rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS);
    /// system.apply_drag(0.47, 1.0); // Apply drag with Cd = 0.47 and area = 1.0 m^2
    /// ```
    pub fn apply_drag(&mut self, drag_coefficient: f64, area: f64) {
        let drag = Force::Drag { coefficient: drag_coefficient, area };
        for obj in &mut self.objects {
            obj.add_force(drag);
        }
    }

    /// Applies spring force to all objects in the system with the given spring constant and displacement.
    /// # Arguments
    /// * `k` - The spring constant in N/m.
    /// * `x` - The displacement from the equilibrium position in meters.
    ///
    /// # Example
    /// ```
    /// let mut system = rs_physics::forces::PhysicsSystem::new(rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS);
    /// system.apply_spring(10.0, 0.1); // Apply spring force with k = 10 N/m and x = 0.1 m
    /// ```
    pub fn apply_spring(&mut self, k: f64, x: f64) {
        let spring = Force::Spring { k, x };
        for obj in &mut self.objects {
            obj.add_force(spring);
        }
    }

    /// Removes all forces from all objects in the system.
    /// This method is useful for resetting the forces on objects before
    /// applying a new set of forces for the next simulation step.
    ///
    /// # Example
    /// ```
    /// let mut system = rs_physics::forces::PhysicsSystem::new(rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS);
    /// system.clear_forces();
    /// ```
    pub fn clear_forces(&mut self) {
        for obj in &mut self.objects {
            obj.clear_forces();
        }
    }
}