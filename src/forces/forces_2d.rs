use log::warn;
use crate::forces::Force;
use crate::models::ObjectIn2D;
use crate::utils::PhysicsConstants;

struct PhysicsSystem2D {
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
    pub fn add_object(&mut self, object: ObjectIn2D) {
        self.objects.push(object);
    }
    pub fn clear_objects(&mut self) {
        self.objects.clear();
    }
    pub fn get_object(&self, index: usize) -> Option<&ObjectIn2D> {
        self.objects.get(index)
    }
    pub fn get_object_mut(&mut self, index: usize) -> Option<&mut ObjectIn2D> {
        self.objects.get_mut(index)
    }
    pub fn remove_object(&mut self, index: usize) -> Option<ObjectIn2D> {
        if index >= self.objects.len() {
            warn!("Index out of bounds, no object removed.\nReturning None.");
            return None;
        }
        Some(self.objects.remove(index))
    }
    pub fn update(&mut self, time_step: f64) {
        for object in self.objects.iter_mut() {
            let mut total_force = 0.0;
            for force in object.forces.iter() {
                total_force += force.apply(object.mass, object.velocity);
            }
            let acceleration = total_force / object.mass;
            object.velocity += acceleration * time_step;
            let (x_velocity, y_velocity) = object.get_directional_velocities();
            object.position.x += x_velocity * time_step;
            object.position.y += y_velocity * time_step;
        }
    }

    pub fn clear_forces(&mut self) {
        for object in self.objects.iter_mut() {
            object.clear_forces();
        }
    }
}