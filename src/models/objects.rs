use crate::forces::Force;
use crate::models::{Axis2D, Axis3D, Direction2D, Direction3D, ObjectIn2D, ObjectIn3D, ToObjectIn2D, ToObjectIn3D};

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
    fn to_2d(&self) -> ObjectIn2D {
        ObjectIn2D {
            mass: self.mass,
            velocity: self.velocity,
            direction: Direction2D { x: 0.0, y: 0.0 },
            position: Axis2D { x: self.position, y: 0.0 },
            forces: self.forces.to_owned(),
        }
    }
}

impl ToObjectIn3D for Object {
    fn to_3d(&self) -> ObjectIn3D {
        ObjectIn3D {
            mass: self.mass,
            velocity: self.velocity,
            direction: Direction3D { x: 0.0, y: 0.0, z: 0.0 },
            position: Axis3D { x: self.position, y: 0.0, z: 0.0 },
            forces: self.forces.to_owned(),
        }
    }
}