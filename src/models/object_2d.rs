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

/// A 2D velocity vector representing both speed and direction.
/// - x: Velocity in the x direction (positive = right, negative = left)
/// - y: Velocity in the y direction (positive = up, negative = down)
#[derive(Debug, Clone)]
pub struct Velocity2D {
    pub x: f64,
    pub y: f64,
}

impl PartialEq for Velocity2D {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl FromCoordinates<(f64, f64)> for Velocity2D {
    fn from_coord(velocity: (f64, f64)) -> Self {
        Velocity2D {
            x: velocity.0,
            y: velocity.1,
        }
    }
}

impl ToCoordinates<(f64, f64)> for Velocity2D {
    fn to_coord(&self) -> (f64, f64) {
        (self.x, self.y)
    }
}

impl <T: FromCoordinates<(f64, f64, f64)>>To3D<T> for Velocity2D {
    /// Converts the struct to a 3D representation.
    /// # Returns
    /// A 3D representation of the struct.
    /// # Example
    /// ```
    /// use rs_physics::models::{Velocity2D, Velocity3D};
    /// use rs_physics::models::To3D;
    ///
    /// let velocity = Velocity2D { x: 3.0, y: 4.0 };
    /// let velocity_3d: Velocity3D = velocity.to_3d();
    ///
    /// assert_eq!(velocity_3d.x, 3.0);
    /// assert_eq!(velocity_3d.y, 4.0);
    /// assert_eq!(velocity_3d.z, 0.0);
    /// ```
    fn to_3d(&self) -> T
    where
        T: FromCoordinates<(f64, f64, f64)>
    {
        T::from_coord((self.x, self.y, 0.0))
    }
}

impl Velocity2D {
    /// Calculate the magnitude (speed) of the velocity vector
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Calculate the direction of the velocity vector as a normalized unit vector
    pub fn direction(&self) -> Direction2D {
        let magnitude = self.magnitude();
        if magnitude == 0.0 {
            Direction2D { x: 0.0, y: 0.0 }
        } else {
            Direction2D {
                x: self.x / magnitude,
                y: self.y / magnitude,
            }
        }
    }
}

/// A 2D direction represented as a unit vector.
/// The x and y values should be between -1.0 and 1.0.
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
    /// let direction = Direction2D { x: 0.5, y: 0.5 };
    /// let direction_3d: Direction3D = direction.to_3d();
    ///
    /// assert_eq!(direction_3d.x, 0.5);
    /// assert_eq!(direction_3d.y, 0.5);
    /// assert_eq!(direction_3d.z, 0.0);
    /// ```
    fn to_3d(&self) -> T
    where
        T: FromCoordinates<(f64, f64, f64)>
    {
        let x = self.x.clamp(-1.0, 1.0);
        let y = self.y.clamp(-1.0, 1.0);
        T::from_coord((x, y, 0.0))
    }
}

impl PartialEq for Direction2D {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl FromCoordinates<(f64, f64)> for Direction2D {
    fn from_coord(position: (f64, f64)) -> Self {
        let x = position.0.clamp(-1.0, 1.0);
        let y = position.1.clamp(-1.0, 1.0);
        Direction2D {
            x,
            y,
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
    pub velocity: Velocity2D,
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
    /// use rs_physics::models::Velocity2D;
    /// let obj = ObjectIn2D::default();
    /// assert_eq!(obj.mass, 1.0);
    /// assert_eq!(obj.velocity, Velocity2D { x: 0.0, y: 0.0 });
    /// assert_eq!(obj.position, Axis2D { x: 0.0, y: 0.0 });
    /// assert_eq!(obj.forces.len(), 0);
    /// ```
    fn default() -> Self {
        ObjectIn2D {
            mass: 1.0,
            velocity: Velocity2D { x: 0.0, y: 0.0 },
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
    /// assert_eq!(obj_3d.velocity.x, 0.0);
    /// assert_eq!(obj_3d.velocity.y, 0.0);
    /// assert_eq!(obj_3d.velocity.z, 0.0);
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
            velocity: self.velocity.to_3d(),
            position: self.position.to_3d(),
            forces: self.forces.to_owned(),
        }
    }
}