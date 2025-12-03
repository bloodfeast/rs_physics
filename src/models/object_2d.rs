use crate::forces::Force;
use crate::models::{FromCoordinates, ObjectIn3D, To3D, ToCoordinates};
use crate::rotational_dynamics::{AngularState2D, InertiaScalar, Shape2D};

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

/// A 2D shape for collision and inertia calculations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shape2DCollider {
    /// A circle with the given radius
    Circle(f64),
    /// A rectangle with width and height
    Rectangle(f64, f64),
}

impl Shape2DCollider {
    /// Calculate the moment of inertia for this shape
    pub fn moment_of_inertia(&self, mass: f64) -> InertiaScalar {
        match self {
            Shape2DCollider::Circle(r) => Shape2D::Disk(*r).moment_of_inertia(mass),
            Shape2DCollider::Rectangle(w, h) => Shape2D::Rectangle(*w, *h).moment_of_inertia(mass),
        }
    }

    /// Get the bounding radius for broad-phase collision detection
    pub fn bounding_radius(&self) -> f64 {
        match self {
            Shape2DCollider::Circle(r) => *r,
            Shape2DCollider::Rectangle(w, h) => (w * w / 4.0 + h * h / 4.0).sqrt(),
        }
    }
}

impl Default for Shape2DCollider {
    fn default() -> Self {
        Shape2DCollider::Circle(1.0)
    }
}

#[derive(Debug, Clone)]
pub struct ObjectIn2D {
    pub mass: f64,
    pub velocity: Velocity2D,
    pub position: Axis2D,
    pub forces: Vec<Force>,
    /// Angular state (velocity and orientation)
    pub angular: AngularState2D,
    /// Shape for collision detection and inertia calculations
    pub shape: Shape2DCollider,
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
            angular: AngularState2D::default(),
            shape: Shape2DCollider::default(),
        }
    }
}

impl ObjectIn2D {
    /// Create a new 2D object with the given mass, position, and shape
    pub fn with_shape(mass: f64, position: (f64, f64), shape: Shape2DCollider) -> Self {
        Self {
            mass,
            velocity: Velocity2D { x: 0.0, y: 0.0 },
            position: Axis2D { x: position.0, y: position.1 },
            forces: Vec::new(),
            angular: AngularState2D::default(),
            shape,
        }
    }

    /// Get the moment of inertia for this object based on its shape and mass
    pub fn moment_of_inertia(&self) -> InertiaScalar {
        self.shape.moment_of_inertia(self.mass)
    }

    /// Apply a torque for a given duration
    pub fn apply_torque(&mut self, torque: f64, dt: f64) {
        let inertia = self.moment_of_inertia();
        self.angular.apply_torque(torque, dt, inertia);
    }

    /// Apply an angular impulse (instantaneous change in angular momentum)
    pub fn apply_angular_impulse(&mut self, impulse: f64) {
        let inertia = self.moment_of_inertia();
        self.angular.apply_impulse(impulse, inertia);
    }

    /// Get the angular velocity in radians per second
    pub fn angular_velocity(&self) -> f64 {
        self.angular.velocity
    }

    /// Set the angular velocity in radians per second
    pub fn set_angular_velocity(&mut self, velocity: f64) {
        self.angular.velocity = velocity;
    }

    /// Get the orientation angle in radians
    pub fn angle(&self) -> f64 {
        self.angular.angle
    }

    /// Set the orientation angle in radians
    pub fn set_angle(&mut self, angle: f64) {
        self.angular.angle = angle;
    }

    /// Integrate the angular position over time
    pub fn integrate_angular(&mut self, dt: f64) {
        self.angular.integrate(dt);
    }

    /// Calculate the rotational kinetic energy
    pub fn rotational_kinetic_energy(&self) -> f64 {
        self.angular.kinetic_energy(self.moment_of_inertia())
    }

    /// Calculate the angular momentum
    pub fn angular_momentum(&self) -> f64 {
        self.angular.momentum(self.moment_of_inertia())
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

#[cfg(all(test, feature = "rotational_dynamics"))]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ========== Shape2DCollider Tests ==========

    #[test]
    fn test_shape2d_collider_circle_default() {
        let shape = Shape2DCollider::default();
        assert!(matches!(shape, Shape2DCollider::Circle(r) if (r - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_shape2d_collider_circle_moment_of_inertia() {
        let shape = Shape2DCollider::Circle(2.0);
        let mass = 5.0;
        let inertia = shape.moment_of_inertia(mass);
        // For disk: I = (1/2) * m * r^2 = 0.5 * 5 * 4 = 10
        assert!((inertia.value() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_shape2d_collider_rectangle_moment_of_inertia() {
        let shape = Shape2DCollider::Rectangle(4.0, 6.0);
        let mass = 3.0;
        let inertia = shape.moment_of_inertia(mass);
        // For rectangle: I = (1/12) * m * (w^2 + h^2) = (1/12) * 3 * (16 + 36) = (1/12) * 3 * 52 = 13
        assert!((inertia.value() - 13.0).abs() < 1e-10);
    }

    #[test]
    fn test_shape2d_collider_circle_bounding_radius() {
        let shape = Shape2DCollider::Circle(3.5);
        assert!((shape.bounding_radius() - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_shape2d_collider_rectangle_bounding_radius() {
        let shape = Shape2DCollider::Rectangle(6.0, 8.0);
        // Bounding radius = sqrt((w/2)^2 + (h/2)^2) = sqrt(9 + 16) = 5
        assert!((shape.bounding_radius() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_shape2d_collider_square_bounding_radius() {
        let shape = Shape2DCollider::Rectangle(4.0, 4.0);
        // Bounding radius = sqrt((2)^2 + (2)^2) = sqrt(8) = 2 * sqrt(2)
        let expected = 2.0 * 2.0_f64.sqrt();
        assert!((shape.bounding_radius() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_shape2d_collider_eq() {
        let c1 = Shape2DCollider::Circle(1.0);
        let c2 = Shape2DCollider::Circle(1.0);
        let c3 = Shape2DCollider::Circle(2.0);
        let r1 = Shape2DCollider::Rectangle(1.0, 2.0);

        assert_eq!(c1, c2);
        assert_ne!(c1, c3);
        assert_ne!(c1, r1);
    }

    // ========== ObjectIn2D Rotation Tests ==========

    #[test]
    fn test_object2d_with_shape() {
        let obj = ObjectIn2D::with_shape(5.0, (10.0, 20.0), Shape2DCollider::Circle(2.0));
        assert!((obj.mass - 5.0).abs() < 1e-10);
        assert!((obj.position.x - 10.0).abs() < 1e-10);
        assert!((obj.position.y - 20.0).abs() < 1e-10);
        assert!(matches!(obj.shape, Shape2DCollider::Circle(r) if (r - 2.0).abs() < 1e-10));
        assert!((obj.angular_velocity() - 0.0).abs() < 1e-10);
        assert!((obj.angle() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_object2d_moment_of_inertia() {
        let obj = ObjectIn2D::with_shape(4.0, (0.0, 0.0), Shape2DCollider::Circle(3.0));
        let inertia = obj.moment_of_inertia();
        // I = 0.5 * m * r^2 = 0.5 * 4 * 9 = 18
        assert!((inertia.value() - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_object2d_angular_velocity_getset() {
        let mut obj = ObjectIn2D::default();
        assert!((obj.angular_velocity() - 0.0).abs() < 1e-10);

        obj.set_angular_velocity(5.0);
        assert!((obj.angular_velocity() - 5.0).abs() < 1e-10);

        obj.set_angular_velocity(-3.0);
        assert!((obj.angular_velocity() - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_object2d_angle_getset() {
        let mut obj = ObjectIn2D::default();
        assert!((obj.angle() - 0.0).abs() < 1e-10);

        obj.set_angle(PI / 4.0);
        assert!((obj.angle() - PI / 4.0).abs() < 1e-10);

        obj.set_angle(-PI);
        assert!((obj.angle() - (-PI)).abs() < 1e-10);
    }

    #[test]
    fn test_object2d_apply_torque() {
        // Circle with mass 2, radius 1: I = 0.5 * 2 * 1 = 1
        let mut obj = ObjectIn2D::with_shape(2.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));

        // Apply torque of 5 N·m for 0.5 seconds
        // α = τ / I = 5 / 1 = 5 rad/s²
        // Δω = α * dt = 5 * 0.5 = 2.5 rad/s
        obj.apply_torque(5.0, 0.5);
        assert!((obj.angular_velocity() - 2.5).abs() < 1e-10);

        // Apply more torque
        obj.apply_torque(5.0, 0.5);
        assert!((obj.angular_velocity() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_object2d_apply_angular_impulse() {
        // Circle with mass 2, radius 1: I = 0.5 * 2 * 1 = 1
        let mut obj = ObjectIn2D::with_shape(2.0, (0.0, 0.0), Shape2DCollider::Circle(1.0));

        // Apply angular impulse of 3 kg·m²/s
        // Δω = J / I = 3 / 1 = 3 rad/s
        obj.apply_angular_impulse(3.0);
        assert!((obj.angular_velocity() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_object2d_integrate_angular() {
        let mut obj = ObjectIn2D::default();
        obj.set_angular_velocity(2.0);

        // Integrate for 0.5 seconds
        // Δθ = ω * dt = 2 * 0.5 = 1 radian
        obj.integrate_angular(0.5);
        assert!((obj.angle() - 1.0).abs() < 1e-10);

        // Continue integration
        obj.integrate_angular(0.5);
        assert!((obj.angle() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_object2d_integrate_angular_wrap() {
        let mut obj = ObjectIn2D::default();
        obj.set_angular_velocity(PI);

        // After 3 seconds, angle = 3π which normalizes to -π or π
        obj.integrate_angular(3.0);
        // Should wrap to (-π, π] range
        let angle = obj.angle();
        assert!(angle >= -PI && angle <= PI);
    }

    #[test]
    fn test_object2d_rotational_kinetic_energy() {
        // Circle with mass 2, radius 2: I = 0.5 * 2 * 4 = 4
        let mut obj = ObjectIn2D::with_shape(2.0, (0.0, 0.0), Shape2DCollider::Circle(2.0));
        obj.set_angular_velocity(3.0);

        // KE = 0.5 * I * ω² = 0.5 * 4 * 9 = 18 J
        let ke = obj.rotational_kinetic_energy();
        assert!((ke - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_object2d_angular_momentum() {
        // Circle with mass 2, radius 2: I = 0.5 * 2 * 4 = 4
        let mut obj = ObjectIn2D::with_shape(2.0, (0.0, 0.0), Shape2DCollider::Circle(2.0));
        obj.set_angular_velocity(5.0);

        // L = I * ω = 4 * 5 = 20 kg·m²/s
        let momentum = obj.angular_momentum();
        assert!((momentum - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_object2d_rectangle_rotation() {
        // Rectangle 4x2 with mass 3: I = (1/12) * 3 * (16 + 4) = (1/12) * 60 = 5
        let mut obj = ObjectIn2D::with_shape(3.0, (0.0, 0.0), Shape2DCollider::Rectangle(4.0, 2.0));

        // Apply torque of 10 N·m for 1 second
        // α = τ / I = 10 / 5 = 2 rad/s²
        // Δω = α * dt = 2 * 1 = 2 rad/s
        obj.apply_torque(10.0, 1.0);
        assert!((obj.angular_velocity() - 2.0).abs() < 1e-10);

        // Integrate for 1 second
        obj.integrate_angular(1.0);
        assert!((obj.angle() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_object2d_default_has_angular_state() {
        let obj = ObjectIn2D::default();
        assert!((obj.angular_velocity() - 0.0).abs() < 1e-10);
        assert!((obj.angle() - 0.0).abs() < 1e-10);
        assert!(matches!(obj.shape, Shape2DCollider::Circle(_)));
    }

    // ========== Velocity2D Tests ==========

    #[test]
    fn test_velocity2d_magnitude() {
        let v = Velocity2D { x: 3.0, y: 4.0 };
        assert!((v.magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_velocity2d_magnitude_zero() {
        let v = Velocity2D { x: 0.0, y: 0.0 };
        assert!((v.magnitude() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_velocity2d_direction() {
        let v = Velocity2D { x: 3.0, y: 4.0 };
        let d = v.direction();
        assert!((d.x - 0.6).abs() < 1e-10);
        assert!((d.y - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_velocity2d_direction_zero() {
        let v = Velocity2D { x: 0.0, y: 0.0 };
        let d = v.direction();
        assert!((d.x - 0.0).abs() < 1e-10);
        assert!((d.y - 0.0).abs() < 1e-10);
    }

    // ========== Direction2D Tests ==========

    #[test]
    fn test_direction2d_clamp_on_construction() {
        let d = Direction2D::from_coord((5.0, -3.0));
        assert!((d.x - 1.0).abs() < 1e-10);
        assert!((d.y - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_direction2d_to_3d() {
        use crate::models::{Direction3D, To3D};
        let d = Direction2D { x: 0.5, y: -0.5 };
        let d3: Direction3D = d.to_3d();
        assert!((d3.x - 0.5).abs() < 1e-10);
        assert!((d3.y - (-0.5)).abs() < 1e-10);
        assert!((d3.z - 0.0).abs() < 1e-10);
    }

    // ========== Axis2D Tests ==========

    #[test]
    fn test_axis2d_to_3d() {
        use crate::models::{Axis3D, To3D};
        let a = Axis2D { x: 10.0, y: 20.0 };
        let a3: Axis3D = a.to_3d();
        assert!((a3.x - 10.0).abs() < 1e-10);
        assert!((a3.y - 20.0).abs() < 1e-10);
        assert!((a3.z - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_axis2d_from_coord() {
        let a = Axis2D::from_coord((5.5, 6.6));
        assert!((a.x - 5.5).abs() < 1e-10);
        assert!((a.y - 6.6).abs() < 1e-10);
    }

    #[test]
    fn test_axis2d_to_coord() {
        let a = Axis2D { x: 1.0, y: 2.0 };
        let (x, y) = a.to_coord();
        assert!((x - 1.0).abs() < 1e-10);
        assert!((y - 2.0).abs() < 1e-10);
    }

    // ========== ObjectIn2D to 3D Conversion ==========

    #[test]
    fn test_object2d_to_3d_conversion() {
        let mut obj = ObjectIn2D::default();
        obj.mass = 5.0;
        obj.position = Axis2D { x: 1.0, y: 2.0 };
        obj.velocity = Velocity2D { x: 3.0, y: 4.0 };

        let obj3d = obj.to_3d();
        assert!((obj3d.mass - 5.0).abs() < 1e-10);
        assert!((obj3d.position.x - 1.0).abs() < 1e-10);
        assert!((obj3d.position.y - 2.0).abs() < 1e-10);
        assert!((obj3d.position.z - 0.0).abs() < 1e-10);
        assert!((obj3d.velocity.x - 3.0).abs() < 1e-10);
        assert!((obj3d.velocity.y - 4.0).abs() < 1e-10);
        assert!((obj3d.velocity.z - 0.0).abs() < 1e-10);
    }
}