use crate::forces::Force;

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
pub struct Axis2D {
    pub x: f64,
    pub y: f64,
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
pub struct Axis3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
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

#[derive(Debug, Clone)]
pub struct Object {
    pub mass: f64,
    pub velocity: f64,
    pub position: f64,
    pub forces: Vec<Force>,
}

#[derive(Debug, Clone)]
pub struct ObjectIn2D {
    pub mass: f64,
    pub velocity: f64,
    pub position: Axis2D,
    pub forces: Vec<Force>,
}

#[derive(Debug, Clone)]
pub struct ObjectIn3D {
    pub mass: f64,
    pub velocity: f64,
    pub position: Axis3D,
    pub forces: Vec<Force>,
}