use log::warn;
use rayon::prelude::*;
use crate::forces::Force;
use crate::models::ObjectIn2D;
use crate::utils::PhysicsConstants;

pub struct PhysicsSystem2D {
    objects: Vec<ObjectIn2D>,
    constants: PhysicsConstants,
}
impl PhysicsSystem2D {
    pub fn new(constants: PhysicsConstants) -> Self {
        PhysicsSystem2D {
            objects: Vec::new(),
            constants,
        }
    }

    pub fn update_ground_level(&mut self, new_ground_level: f64) {
        self.constants.ground_level = new_ground_level;
    }

    /// Adds an object to the system.
    /// # Arguments
    /// * `object` - The object to be added.
    /// # Example
    /// ```
    /// use rs_physics::forces::PhysicsSystem2D;
    /// let mut system = rs_physics::forces::PhysicsSystem2D::new(rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS);
    /// let object = rs_physics::models::ObjectIn2D::default();
    /// system.add_object(object);
    /// assert!(system.get_object(0).is_some());
    /// ```
    pub fn add_object(&mut self, object: ObjectIn2D) {
        self.objects.push(object);
    }

    /// Clears all objects from the system.
    /// # Example
    /// ```
    /// use rs_physics::forces::PhysicsSystem2D;
    /// let mut system = rs_physics::forces::PhysicsSystem2D::new(rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS);
    /// let object = rs_physics::models::ObjectIn2D::default();
    /// system.add_object(object);
    /// system.clear_objects();
    /// assert!(system.get_object(0).is_none());
    /// ```
    pub fn clear_objects(&mut self) {
        self.objects.clear();
    }

    /// Gets an object from the system.
    /// # Arguments
    /// * `index` - The index of the object to be retrieved.
    /// # Returns
    /// An `Option` containing the object at the given index, or `None` if the index is out of bounds.
    /// # Example
    /// ```
    /// use rs_physics::forces::PhysicsSystem2D;
    /// let mut system = rs_physics::forces::PhysicsSystem2D::new(rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS);
    /// let object = rs_physics::models::ObjectIn2D::default();
    /// system.add_object(object);
    /// assert!(system.get_object(0).is_some());
    /// ```
    pub fn get_object(&self, index: usize) -> Option<&ObjectIn2D> {
        self.objects.get(index)
    }

    /// Gets a mutable reference to an object in the system.
    /// # Arguments
    /// * `index` - The index of the object to be retrieved.
    /// # Returns
    /// An `Option` containing a mutable reference to the object at the given index, or `None` if the index is out of bounds.
    /// # Example
    /// ```
    /// use rs_physics::forces::PhysicsSystem2D;
    /// let mut system = rs_physics::forces::PhysicsSystem2D::new(rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS);
    /// let object = rs_physics::models::ObjectIn2D::default();
    /// system.add_object(object);
    /// assert!(system.get_object_mut(0).is_some());
    /// ```
    pub fn get_object_mut(&mut self, index: usize) -> Option<&mut ObjectIn2D> {
        self.objects.get_mut(index)
    }

    /// Removes an object from the system.
    /// # Arguments
    /// * `index` - The index of the object to be removed.
    /// # Returns
    /// An `Option` containing the object that was removed, or `None` if the index was out of bounds.
    /// # Example
    /// ```
    /// use rs_physics::forces::PhysicsSystem2D;
    /// let mut system = rs_physics::forces::PhysicsSystem2D::new(rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS);
    /// let object = rs_physics::models::ObjectIn2D::default();
    /// system.add_object(object);
    /// system.remove_object(0);
    /// assert!(system.get_object(0).is_none());
    /// ```
    pub fn remove_object(&mut self, index: usize) -> Option<ObjectIn2D> {
        if index >= self.objects.len() {
            warn!("Index out of bounds, no object removed.\nReturning None.");
            return None;
        }
        Some(self.objects.remove(index))
    }

    /// Updates the positions of all objects in the system.
    /// # Arguments
    /// * `time_step` - The time step to update the system by.
    /// # Example
    /// ```
    /// use rs_physics::forces::PhysicsSystem2D;
    /// use rs_physics::models::{ObjectIn2D, Velocity2D};
    ///
    /// let mut system = rs_physics::forces::PhysicsSystem2D::new(rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS);
    /// system.update(0.1);
    ///
    /// let mut object = ObjectIn2D::default();
    /// object.velocity.x = 5.0;
    /// system.add_object(object);
    /// system.update(0.1);
    /// assert_eq!(system.get_object(0).unwrap().position.x, 0.5);
    /// ```
    pub fn update(&mut self, time_step: f64) {
        self.objects.par_iter_mut().for_each(|object| {
            // Sum forces as vectors.
            let mut total_fx = 0.0;
            let mut total_fy = 0.0;
            for force in object.forces.iter() {
                // We need to adapt force application for the new velocity structure
                // Get the current direction for the force calculation
                let direction = object.direction();
                let speed = object.speed();

                let (fx, fy) = force.apply_vector(object.mass, speed, &direction);
                total_fx += fx;
                total_fy += fy;
            }

            // Compute acceleration components.
            let ax = total_fx / object.mass;
            let ay = total_fy / object.mass;

            // Update velocity components with acceleration.
            object.velocity.x += ax * time_step;
            object.velocity.y += ay * time_step;

            // Update position.
            object.position.x += object.velocity.x * time_step;
            object.position.y += object.velocity.y * time_step;

            // Check ground collision using the ground_level from constants.
            if object.position.y <= self.constants.ground_level {
                object.position.y = self.constants.ground_level;
                // Only process if the object is falling.
                if object.velocity.y < -0.1 {
                    // Define restitution and friction values.
                    let restitution = 0.0; // 0 = fully inelastic (no bounce)
                    let friction = 0.1;    // adjust friction to reduce horizontal momentum on landing
                    // Apply restitution to vertical velocity.
                    object.velocity.y = -restitution * object.velocity.y;
                    // Reduce horizontal velocity by friction.
                    object.velocity.x *= 1.0 - friction;
                } else {
                    // If not falling, simply zero out vertical velocity.
                    object.velocity.y = 0.0;
                }
            }

            // No need to recompose velocity and direction as we now use velocity components directly

            // Optionally clear forces after applying them, or let them persist for a duration.
            object.forces.retain(|f| matches!(f, Force::Gravity(_)) || matches!(f, Force::Drag { .. }));
        });
    }

    /// Applies gravity to all objects in the system.
    /// # Example
    /// ```
    /// use rs_physics::forces::PhysicsSystem2D;
    /// let mut system = rs_physics::forces::PhysicsSystem2D::new(rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS);
    /// system.apply_gravity();
    /// ```
    pub fn apply_gravity(&mut self) {
        self.objects
            .par_iter_mut()
            .for_each(|object| {
                let force = Force::Gravity(self.constants.gravity);
                object.forces.retain(|f| !matches!(f, Force::Gravity(_)));
                object.add_force(force);
            });
    }

    /// Applies drag to all objects in the system.
    /// # Arguments
    /// * `drag_coefficient` - The drag coefficient to be used in the drag force calculation.
    /// * `cross_sectional_area` - The reference area to be used in the drag force calculation.
    /// # Example
    /// ```
    /// use rs_physics::forces::PhysicsSystem2D;
    /// let mut system = rs_physics::forces::PhysicsSystem2D::new(rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS);
    /// system.apply_drag(0.45, 1.0);
    /// ```
    pub fn apply_drag(&mut self, drag_coefficient: f64, cross_sectional_area: f64) {
        self.objects
            .par_iter_mut()
            .for_each(|object| {
                let force = Force::Drag { coefficient: drag_coefficient, area: cross_sectional_area };
                object.forces.retain(|f| !matches!(f, Force::Drag { .. }));
                object.add_force(force);
            });
    }

    /// Applies spring force to all objects in the system.
    /// # Arguments
    /// * `spring_constant` - The spring constant in N/m.
    /// * `displacement` - The displacement from the equilibrium position in meters.
    /// # Example
    /// ```
    /// use rs_physics::forces::PhysicsSystem2D;
    /// let mut system = rs_physics::forces::PhysicsSystem2D::new(rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS);
    /// system.apply_spring_force(1.0, 0.1);
    /// ```
    pub fn apply_spring_force(&mut self, spring_constant: f64, displacement: f64) {
        self.objects
            .par_iter_mut()
            .for_each(|object| {
                let force = Force::Spring { k: spring_constant, x: displacement };
                object.forces.retain(|f| !matches!(f, Force::Spring { .. }));
                object.add_force(force);
            });
    }

    /// Clears all forces from all objects in the system.
    /// # Example
    /// ```
    /// use rs_physics::forces::PhysicsSystem2D;
    /// let mut system = rs_physics::forces::PhysicsSystem2D::new(rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS);
    /// system.clear_forces();
    /// ```
    pub fn clear_forces(&mut self) {
        self.objects
            .par_iter_mut()
            .for_each(|object| object.clear_forces());
    }
}