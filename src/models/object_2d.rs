use crate::forces::Force;
use crate::models::{FromCoordinates, ObjectIn3D, To3D, ToCoordinates};

#[derive(Debug, Clone)]
pub struct Axis2D {
    pub x: f64,
    pub y: f64,
}

impl PartialEq for Axis2D {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

pub trait To2D<T> {
    /// Converts the struct to a 2D representation.
    /// # Returns
    /// A 2D representation of the struct.
    /// # Example
    /// ```
    /// use rs_physics::models::{Axis2D, Axis3D};
    /// use rs_physics::models::To2D;
    ///
    /// let axis = Axis3D { x: 1.0, y: 2.0, z: 3.0 };
    /// let axis_2d: Axis2D = axis.to_2d();
    ///
    /// assert_eq!(axis_2d.x, 1.0);
    /// assert_eq!(axis_2d.y, 2.0);
    /// ```
    fn to_2d(&self) -> T;
}

impl <T: FromCoordinates<(f64, f64, f64)>>To3D<T> for Axis2D {
    /// Converts the struct to a 3D representation.
    /// # Returns
    /// A 3D representation of the struct.
    /// # Example
    /// ```
    /// use rs_physics::models::{Axis2D, Axis3D};
    /// use rs_physics::models::To3D;
    ///
    /// let axis = Axis2D { x: 1.0, y: 2.0 };
    /// let axis_3d: Axis3D = axis.to_3d();
    ///
    /// assert_eq!(axis_3d.x, 1.0);
    /// assert_eq!(axis_3d.y, 2.0);
    /// assert_eq!(axis_3d.z, 0.0);
    /// ```
    fn to_3d(&self) -> T
    where
        T: FromCoordinates<(f64, f64, f64)>
    {
        T::from_coord((self.x, self.y, 0.0))
    }
}

impl FromCoordinates<(f64, f64)> for Axis2D {
    fn from_coord(position: (f64, f64)) -> Self {
        Axis2D {
            x: position.0,
            y: position.1,
        }
    }
}

impl ToCoordinates<(f64, f64)> for Axis2D {
    fn to_coord(&self) -> (f64, f64) {
        (self.x, self.y)
    }
}

#[derive(Debug, Clone)]
pub struct Direction2D {
    pub x: f64,
    pub y: f64,
}

impl <T: FromCoordinates<(f64, f64, f64)>>To3D<T> for Direction2D {
    /// Converts the struct to a 3D representation.
    /// # Returns
    /// A 3D representation of the struct.
    /// # Example
    /// ```
    /// use rs_physics::models::{Direction2D, Direction3D};
    /// use rs_physics::models::To3D;
    ///
    /// let direction = Direction2D { x: 1.0, y: 2.0 };
    /// let direction_3d: Direction3D = direction.to_3d();
    ///
    /// assert_eq!(direction_3d.x, 1.0);
    /// assert_eq!(direction_3d.y, 2.0);
    /// assert_eq!(direction_3d.z, 0.0);
    /// ```
    fn to_3d(&self) -> T
    where
        T: FromCoordinates<(f64, f64, f64)>
    {
        T::from_coord((self.x, self.y, 0.0))
    }
}

impl PartialEq for Direction2D {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl FromCoordinates<(f64, f64)> for Direction2D {
    fn from_coord(position: (f64, f64)) -> Self {
        Direction2D {
            x: position.0,
            y: position.1,
        }
    }
}

impl ToCoordinates<(f64, f64)> for Direction2D {
    fn to_coord(&self) -> (f64, f64) {
        (self.x, self.y)
    }
}

#[derive(Debug, Clone)]
pub struct ObjectIn2D {
    pub mass: f64,
    pub velocity: f64,
    pub direction: Direction2D,
    pub position: Axis2D,
    pub forces: Vec<Force>,
}

impl Default for ObjectIn2D {
    /// Creates a new `ObjectIn2D` with default values.
    /// # Returns
    /// A new `ObjectIn2D` with default values.
    /// # Example
    /// ```
    /// use rs_physics::models::ObjectIn2D;
    /// use rs_physics::models::Axis2D;
    /// use rs_physics::models::Direction2D;
    /// let obj = ObjectIn2D::default();
    /// assert_eq!(obj.mass, 1.0);
    /// assert_eq!(obj.velocity, 0.0);
    /// assert_eq!(obj.direction, Direction2D { x: 0.0, y: 0.0 });
    /// assert_eq!(obj.position, Axis2D { x: 0.0, y: 0.0 });
    /// assert_eq!(obj.forces.len(), 0);
    /// ```
    fn default() -> Self {
        ObjectIn2D {
            mass: 1.0,
            velocity: 0.0,
            direction: Direction2D { x: 0.0, y: 0.0 },
            position: Axis2D { x: 0.0, y: 0.0 },
            forces: Vec::new(),
        }
    }
}

pub trait ToObjectIn3D {
    /// Converts the struct to a 3D representation.
    /// # Returns
    /// A 3D representation of the struct.
    /// # Example
    /// ```
    /// use rs_physics::models::{ObjectIn2D, ObjectIn3D};
    /// use rs_physics::models::ToObjectIn3D;
    ///
    /// let obj = ObjectIn2D::default();
    /// let obj_3d: ObjectIn3D = obj.to_3d();
    ///
    /// assert_eq!(obj_3d.mass, 1.0);
    /// assert_eq!(obj_3d.velocity, 0.0);
    /// assert_eq!(obj_3d.direction.x, 0.0);
    /// assert_eq!(obj_3d.direction.y, 0.0);
    /// assert_eq!(obj_3d.direction.z, 0.0);
    /// assert_eq!(obj_3d.position.x, 0.0);
    /// assert_eq!(obj_3d.position.y, 0.0);
    /// assert_eq!(obj_3d.position.z, 0.0);
    /// assert_eq!(obj_3d.forces.len(), 0);
    /// ```
    fn to_3d(&self) -> ObjectIn3D;
}

impl ToObjectIn3D for ObjectIn2D {
    fn to_3d(&self) -> ObjectIn3D {
        ObjectIn3D {
            mass: self.mass,
            velocity: self.velocity,
            direction: self.direction.to_3d(),
            position: self.position.to_3d(),
            forces: self.forces.to_owned(),
        }
    }
}