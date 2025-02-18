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
    /// let mut system = rs_physics::forces::PhysicsSystem2D::new(rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS);
    /// system.update(0.1);
    ///
    /// let mut object = rs_physics::models::ObjectIn2D::default();
    /// object.velocity = 5.0;
    /// object.direction.x = 1.0;
    /// system.add_object(object);
    /// system.update(0.1);
    /// assert_eq!(system.get_object(0).unwrap().position.x, 0.5);
    /// ```
    pub fn update(&mut self, time_step: f64) {
        self.objects
            .par_iter_mut()
            .for_each(|object| {
                let mut total_force = 0.0;
                for force in object.forces.iter() {
                    total_force += force.apply(object.mass, object.velocity);
                }
                let acceleration = total_force / object.mass;
                object.velocity += acceleration * time_step;
                let (x_velocity, y_velocity) = object.get_directional_velocities();
                object.position.x += x_velocity * time_step;
                object.position.y += y_velocity * time_step;
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
                object.add_force(force);
            });
    }

    pub fn clear_forces(&mut self) {
        self.objects
            .par_iter_mut()
            .for_each(|object| object.clear_forces());
    }
}