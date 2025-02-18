use crate::forces::Force;
use crate::models::{FromCoordinates, ObjectIn2D, To2D, ToCoordinates};

#[derive(Debug, Clone)]
pub struct Axis3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl PartialEq for Axis3D {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z
    }
}

impl FromCoordinates<(f64, f64, f64)> for Axis3D {
    fn from_coord(position: (f64, f64, f64)) -> Self {
        Axis3D {
            x: position.0,
            y: position.1,
            z: position.2,
        }
    }
}

impl ToCoordinates<(f64, f64, f64)> for Axis3D {
    fn to_coord(&self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }
}


pub trait To3D<T> {
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
    fn to_3d(&self) -> T;
}

impl <T: FromCoordinates<(f64, f64)>>To2D<T> for Axis3D {
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
    fn to_2d(&self) -> T
    where
        T: FromCoordinates<(f64, f64)>
    {
        T::from_coord((self.x, self.y))
    }
}

#[derive(Debug, Clone)]
pub struct Direction3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl <T: FromCoordinates<(f64, f64)>>To2D<T> for Direction3D {
    /// Converts the struct to a 2D representation.
    /// # Returns
    /// A 2D representation of the struct.
    /// # Example
    /// ```
    /// use rs_physics::models::{Direction2D, Direction3D};
    /// use rs_physics::models::To2D;
    ///
    /// let direction = Direction3D { x: 1.0, y: 2.0, z: 3.0 };
    /// let direction_2d: Direction2D = direction.to_2d();
    ///
    /// assert_eq!(direction_2d.x, 1.0);
    /// assert_eq!(direction_2d.y, 2.0);
    /// ```
    fn to_2d(&self) -> T
    where
        T: FromCoordinates<(f64, f64)>
    {
        T::from_coord((self.x, self.y))
    }
}

impl PartialEq for Direction3D {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z
    }
}

impl FromCoordinates<(f64, f64, f64)> for Direction3D {
    fn from_coord(position: (f64, f64, f64)) -> Self {
        Direction3D {
            x: position.0,
            y: position.1,
            z: position.2,
        }
    }
}

impl ToCoordinates<(f64, f64, f64)> for Direction3D {
    fn to_coord(&self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone)]
pub struct ObjectIn3D {
    pub mass: f64,
    pub velocity: f64,
    pub direction: Direction3D,
    pub position: Axis3D,
    pub forces: Vec<Force>,
}

impl Default for ObjectIn3D {
    /// Creates a new `ObjectIn3D` with default values.
    /// # Returns
    /// A new `ObjectIn3D` with default values.
    /// # Example
    /// ```
    /// use rs_physics::models::ObjectIn3D;
    /// use rs_physics::models::Axis3D;
    /// use rs_physics::models::Direction3D;
    ///
    /// let obj = ObjectIn3D::default();
    /// assert_eq!(obj.mass, 1.0);
    /// assert_eq!(obj.velocity, 0.0);
    /// assert_eq!(obj.direction, Direction3D { x: 0.0, y: 0.0, z: 0.0 });
    /// assert_eq!(obj.position, Axis3D { x: 0.0, y: 0.0, z: 0.0 });
    /// assert_eq!(obj.forces.len(), 0);
    /// ```
    fn default() -> Self {
        ObjectIn3D {
            mass: 1.0,
            velocity: 0.0,
            direction: Direction3D { x: 0.0, y: 0.0, z: 0.0 },
            position: Axis3D { x: 0.0, y: 0.0, z: 0.0 },
            forces: Vec::new(),
        }
    }
}

pub trait ToObjectIn2D {
    /// Converts the struct to a 2D representation.
    /// # Returns
    /// A 2D representation of the struct.
    /// # Example
    /// ```
    /// use rs_physics::models::{ObjectIn2D, ObjectIn3D, ToObjectIn2D, Axis2D, Direction2D};
    ///
    /// let obj = ObjectIn3D::default();
    /// let obj_2d: ObjectIn2D = obj.to_2d();
    ///
    /// assert_eq!(obj_2d.mass, 1.0);
    /// assert_eq!(obj_2d.velocity, 0.0);
    /// assert_eq!(obj_2d.direction, Direction2D { x: 0.0, y: 0.0 });
    /// assert_eq!(obj_2d.position, Axis2D { x: 0.0, y: 0.0 });
    /// assert_eq!(obj_2d.forces.len(), 0);
    /// ```
    fn to_2d(&self) -> ObjectIn2D;
}

impl ToObjectIn2D for ObjectIn3D {
    fn to_2d(&self) -> ObjectIn2D {
        ObjectIn2D {
            mass: self.mass,
            velocity: self.velocity,
            direction: self.direction.to_2d(),
            position: self.position.to_2d(),
            forces: self.forces.to_owned(),
        }
    }
}

