// src/forces.rs

use crate::constants_config::PhysicsConstants;
use crate::interactions::Object;

#[derive(Debug, Clone, Copy)]
pub enum Force {
    Gravity(f64),
    Drag { coefficient: f64, area: f64 },
    Spring { k: f64, x: f64 },
    Constant(f64),
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
    /// let system = rs_physics::forces::PhysicsSystem::new(rs_physics::DEFAULT_PHYSICS_CONSTANTS);
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
    /// let object = rs_physics::interactions::Object::new(10.0, 5.0, 0.0).unwrap();
    /// let mut system = rs_physics::forces::PhysicsSystem::new(rs_physics::DEFAULT_PHYSICS_CONSTANTS);
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
    /// let mut system = rs_physics::forces::PhysicsSystem::new(rs_physics::DEFAULT_PHYSICS_CONSTANTS);
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
    /// let mut system = rs_physics::forces::PhysicsSystem::new(rs_physics::DEFAULT_PHYSICS_CONSTANTS);
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
    /// let mut system = rs_physics::forces::PhysicsSystem::new(rs_physics::DEFAULT_PHYSICS_CONSTANTS);
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
    /// let mut system = rs_physics::forces::PhysicsSystem::new(rs_physics::DEFAULT_PHYSICS_CONSTANTS);
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
    /// let mut system = rs_physics::forces::PhysicsSystem::new(rs_physics::DEFAULT_PHYSICS_CONSTANTS);
    /// system.clear_forces();
    /// ```
    pub fn clear_forces(&mut self) {
        for obj in &mut self.objects {
            obj.clear_forces();
        }
    }
}