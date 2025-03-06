use crate::forces::Force;
use crate::models::{Axis2D, Axis3D, Velocity2D, Velocity3D, ObjectIn2D, ObjectIn3D, ToObjectIn2D, ToObjectIn3D};

pub trait FromCoordinates <T> {
    /// Creates a new instance of the struct from the given coordinates.
    /// # Arguments
    /// * `position` - The coordinates to create the struct from.
    /// # Returns
    /// A new instance of the struct.
    /// # Example
    /// ```
    /// use rs_physics::models::Axis2D;
    /// use rs_physics::models::FromCoordinates;
    ///
    /// let axis = Axis2D::from_coord((1.0, 2.0));
    /// assert_eq!(axis.x, 1.0);
    /// assert_eq!(axis.y, 2.0);
    /// ```
    fn from_coord(position: T) -> Self;
}

pub trait ToCoordinates <T> {
    /// Converts the struct to a tuple of coordinates.
    /// # Returns
    /// A tuple of coordinates.
    /// # Example
    /// ```
    /// use rs_physics::models::Axis2D;
    /// use rs_physics::models::ToCoordinates;
    ///
    /// let axis = Axis2D { x: 1.0, y: 2.0 };
    /// let coordinates = axis.to_coord();
    ///
    /// assert_eq!(coordinates.0, 1.0);
    /// assert_eq!(coordinates.1, 2.0);
    /// ```
    fn to_coord(&self) -> T;
}

#[derive(Debug, Clone)]
pub struct Object {
    pub mass: f64,
    pub velocity: f64,
    pub position: f64,
    pub forces: Vec<Force>,
}

impl Default for Object {
    /// Creates a new `Object` with default values.
    /// # Returns
    /// A new `Object` with default values.
    /// # Example
    /// ```
    /// use rs_physics::models::Object;
    /// let obj = Object::default();
    /// assert_eq!(obj.mass, 1.0);
    /// assert_eq!(obj.velocity, 0.0);
    /// assert_eq!(obj.position, 0.0);
    /// assert_eq!(obj.forces.len(), 0);
    /// ```
    fn default() -> Self {
        Object {
            mass: 1.0,
            velocity: 0.0,
            position: 0.0,
            forces: Vec::new(),
        }
    }
}

impl ToObjectIn2D for Object {
    /// Converts a 1D object into a 2D object.
    /// The 1D velocity becomes the x component of the 2D velocity.
    ///
    /// # Returns
    /// A 2D representation of the object.
    ///
    /// # Example
    /// ```
    /// use rs_physics::models::{Object, ObjectIn2D, ToObjectIn2D, Velocity2D};
    ///
    /// let obj = Object { mass: 2.0, velocity: 5.0, position: 10.0, forces: vec![] };
    /// let obj_2d = obj.to_2d();
    ///
    /// assert_eq!(obj_2d.mass, 2.0);
    /// assert_eq!(obj_2d.velocity.x, 5.0);
    /// assert_eq!(obj_2d.velocity.y, 0.0);
    /// assert_eq!(obj_2d.position.x, 10.0);
    /// assert_eq!(obj_2d.position.y, 0.0);
    /// ```
    fn to_2d(&self) -> ObjectIn2D {
        ObjectIn2D {
            mass: self.mass,
            velocity: Velocity2D { x: self.velocity, y: 0.0 },
            position: Axis2D { x: self.position, y: 0.0 },
            forces: self.forces.to_owned(),
        }
    }
}

impl ToObjectIn3D for Object {
    /// Converts a 1D object into a 3D object.
    /// The 1D velocity becomes the x component of the 3D velocity.
    ///
    /// # Returns
    /// A 3D representation of the object.
    ///
    /// # Example
    /// ```
    /// use rs_physics::models::{Object, ObjectIn3D, ToObjectIn3D, Velocity3D};
    ///
    /// let obj = Object { mass: 2.0, velocity: 5.0, position: 10.0, forces: vec![] };
    /// let obj_3d = obj.to_3d();
    ///
    /// assert_eq!(obj_3d.mass, 2.0);
    /// assert_eq!(obj_3d.velocity.x, 5.0);
    /// assert_eq!(obj_3d.velocity.y, 0.0);
    /// assert_eq!(obj_3d.velocity.z, 0.0);
    /// assert_eq!(obj_3d.position.x, 10.0);
    /// assert_eq!(obj_3d.position.y, 0.0);
    /// assert_eq!(obj_3d.position.z, 0.0);
    /// ```
    fn to_3d(&self) -> ObjectIn3D {
        ObjectIn3D {
            mass: self.mass,
            velocity: Velocity3D { x: self.velocity, y: 0.0, z: 0.0 },
            position: Axis3D { x: self.position, y: 0.0, z: 0.0 },
            forces: self.forces.to_owned(),
        }
    }
}