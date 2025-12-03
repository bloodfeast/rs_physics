use crate::forces::Force;
use crate::models::{Axis2D, Direction2D, FromCoordinates, ObjectIn2D, Velocity2D, Shape2DCollider};
use crate::rotational_dynamics::AngularState2D;
use crate::utils::PhysicsConstants;

impl ObjectIn2D {
    /// Creates a new `ObjectIn2D` with the given mass, velocity components, and position.
    ///
    /// # Arguments
    /// * `mass` - The mass of the object in kilograms.
    /// * `vx` - The velocity in the x direction in meters per second.
    /// * `vy` - The velocity in the y direction in meters per second.
    /// * `position` - The position as (x, y) coordinates.
    ///
    /// # Returns
    /// A new `ObjectIn2D` with the specified properties.
    ///
    /// # Example
    /// ```
    /// use rs_physics::models::ObjectIn2D;
    /// let obj = ObjectIn2D::new(1.0, 2.0, 3.0, (0.0, 0.0));
    /// assert_eq!(obj.mass, 1.0);
    /// assert_eq!(obj.velocity.x, 2.0);
    /// assert_eq!(obj.velocity.y, 3.0);
    /// ```
    pub fn new(mass: f64, vx: f64, vy: f64, position: (f64, f64)) -> Self {
        ObjectIn2D {
            mass,
            velocity: Velocity2D { x: vx, y: vy },
            position: Axis2D::from_coord(position),
            forces: Vec::new(),
            angular: AngularState2D::default(),
            shape: Shape2DCollider::default(),
            material: None,
        }
    }

    /// Creates a new `ObjectIn2D` with the given mass, velocity magnitude, direction, and position.
    /// This method maintains backward compatibility with the old API.
    ///
    /// # Arguments
    /// * `mass` - The mass of the object in kilograms.
    /// * `speed` - The magnitude of velocity in meters per second.
    /// * `direction` - The direction as (x, y) components (will be normalized).
    /// * `position` - The position as (x, y) coordinates.
    ///
    /// # Returns
    /// A new `ObjectIn2D` with the specified properties.
    ///
    /// # Example
    /// ```
    /// use rs_physics::models::ObjectIn2D;
    /// let obj = ObjectIn2D::new_with_direction(1.0, 5.0, (1.0, 0.0), (0.0, 0.0));
    /// assert_eq!(obj.mass, 1.0);
    /// assert_eq!(obj.velocity.x, 5.0);
    /// assert_eq!(obj.velocity.y, 0.0);
    /// ```
    pub fn new_with_direction(mass: f64, speed: f64, direction: (f64, f64), position: (f64, f64)) -> Self {
        let dir = Direction2D::from_coord(direction);

        // Normalize direction components if they're not already
        let magnitude = (dir.x * dir.x + dir.y * dir.y).sqrt();
        let (x_normalized, y_normalized) = if magnitude > 0.0 {
            (dir.x / magnitude, dir.y / magnitude)
        } else {
            (0.0, 0.0)
        };

        ObjectIn2D {
            mass,
            velocity: Velocity2D {
                x: speed * x_normalized,
                y: speed * y_normalized
            },
            position: Axis2D::from_coord(position),
            forces: Vec::new(),
            angular: AngularState2D::default(),
            shape: Shape2DCollider::default(),
            material: None,
        }
    }

    /// Get the speed (magnitude of velocity)
    pub fn speed(&self) -> f64 {
        self.velocity.magnitude()
    }

    /// Get the direction as a normalized vector
    pub fn direction(&self) -> Direction2D {
        self.velocity.direction()
    }

    pub fn add_force(&mut self, force: Force) {
        self.forces.push(force);
    }

    pub fn clear_forces(&mut self) {
        self.forces.clear();
    }
}

/// Simulates a 2D elastic collision between two objects.
///
/// # Arguments
///
/// * `constants` - Structure containing gravity and air density.
/// * `obj1`, `obj2` - Mutable references to the two colliding objects.
/// * `angle` - The angle of collision in radians.
/// * `duration` - The duration of the collision in seconds (must be positive).
/// * `drag_coefficient` - The coefficient of drag (must be non-negative).
/// * `cross_sectional_area` - The cross-sectional area of the objects (must be non-negative).
///
/// # Returns
///
/// `Ok(())` if the collision was successfully computed, or an error message on invalid parameters.
///
/// # Examples
///
/// ```
/// use rs_physics::interactions::elastic_collision_2d;
/// use rs_physics::models::ObjectIn2D;
///
/// // Create objects with x and y velocity components
/// let mut obj1 = ObjectIn2D::new(1.0, 2.0, 0.0, (0.0, 0.0));
/// let mut obj2 = ObjectIn2D::new(1.0, -1.0, 0.0, (1.0, 0.0));
///
/// elastic_collision_2d(&rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS,
///                     &mut obj1, &mut obj2, 0.0, 1.0, 0.45, 1.0).unwrap();
///
/// assert!((obj1.velocity.x - (-0.724375)).abs() < 1e-6);
/// assert!((obj2.velocity.x - 1.44875).abs() < 1e-6);
/// ```
///
/// # Notes
///
/// This function projects each object's velocity onto axes parallel and perpendicular to the collision angle,
/// applies a 1D elastic collision formula to the parallel components, then updates velocities considering gravity and drag.
pub fn elastic_collision_2d(
    constants: &PhysicsConstants,
    obj1: &mut ObjectIn2D,
    obj2: &mut ObjectIn2D,
    angle: f64,
    duration: f64,
    drag_coefficient: f64,
    cross_sectional_area: f64
) -> Result<(), &'static str> {
    if obj1.mass <= 0.0 || obj2.mass <= 0.0 {
        return Err("Mass must be positive");
    }
    if duration <= 0.0 {
        return Err("Collision duration must be positive");
    }
    if drag_coefficient < 0.0 {
        return Err("Drag coefficient must be non-negative");
    }
    if cross_sectional_area < 0.0 {
        return Err("Cross-sectional area must be non-negative");
    }

    let vx1 = obj1.velocity.x;
    let vy1 = obj1.velocity.y;
    let vx2 = obj2.velocity.x;
    let vy2 = obj2.velocity.y;

    // Project velocities onto collision axis (angle) and perpendicular axis (angle + 90)
    let v1_along = vx1 * angle.cos() + vy1 * angle.sin();
    let v1_perp = -vx1 * angle.sin() + vy1 * angle.cos();
    let v2_along = vx2 * angle.cos() + vy2 * angle.sin();
    let v2_perp = -vx2 * angle.sin() + vy2 * angle.cos();

    // Calculate final velocities along collision axis using a 1D elastic collision formula
    let m1 = obj1.mass;
    let m2 = obj2.mass;
    let total_mass = m1 + m2;
    let mass_diff = m1 - m2;

    let v1_final_along = (mass_diff * v1_along + 2.0 * m2 * v2_along) / total_mass;
    let v2_final_along = (2.0 * m1 * v1_along - mass_diff * v2_along) / total_mass;

    // Approximate gravity effect (if needed)
    let gravity_effect = constants.gravity * duration * angle.sin();

    // Simplified air resistance
    let air_res_force = 0.5 * constants.air_density * drag_coefficient * cross_sectional_area * duration;
    let v1_drag = air_res_force * v1_final_along.abs() / m1;
    let v2_drag = air_res_force * v2_final_along.abs() / m2;

    // Update velocities along collision axis with gravity and drag
    let v1_along_updated = v1_final_along - gravity_effect - v1_drag * v1_final_along.signum();
    let v2_along_updated = v2_final_along - gravity_effect - v2_drag * v2_final_along.signum();

    // Recompose velocities
    let new_vx1 = v1_along_updated * angle.cos() - v1_perp * angle.sin();
    let new_vy1 = v1_along_updated * angle.sin() + v1_perp * angle.cos();
    let new_vx2 = v2_along_updated * angle.cos() - v2_perp * angle.sin();
    let new_vy2 = v2_along_updated * angle.sin() + v2_perp * angle.cos();

    // Update velocity vectors directly
    obj1.velocity.x = new_vx1;
    obj1.velocity.y = new_vy1;
    obj2.velocity.x = new_vx2;
    obj2.velocity.y = new_vy2;

    Ok(())
}

/// Result of a material-based 2D collision.
///
/// Contains the updated velocities and optionally the heat generated
/// during the collision when the thermodynamics feature is enabled.
#[derive(Debug, Clone)]
pub struct Collision2DResult {
    /// Updated velocity for object 1
    pub velocity1: Velocity2D,
    /// Updated velocity for object 2
    pub velocity2: Velocity2D,
    /// Heat generated during collision (in Joules)
    /// Only calculated when the `thermodynamics` feature is enabled.
    #[cfg(feature = "thermodynamics")]
    pub heat_generated: f64,
}

/// Simulates a 2D collision between two objects using their material properties.
///
/// This function uses the objects' material properties (restitution and friction)
/// to calculate the collision response. If materials are not set, default values
/// are used (restitution: 0.8, friction: 0.5).
///
/// # Arguments
///
/// * `obj1`, `obj2` - The two colliding objects.
/// * `contact_normal` - The collision normal vector (pointing from obj2 to obj1).
///
/// # Returns
///
/// A `Collision2DResult` containing the new velocities for both objects,
/// and optionally the heat generated (when thermodynamics feature is enabled).
///
/// # Examples
///
/// ```
/// use rs_physics::interactions::{material_collision_2d, Collision2DResult};
/// use rs_physics::models::{ObjectIn2D, Shape2DCollider};
/// use rs_physics::materials::Material;
///
/// // Create two objects with materials
/// let obj1 = ObjectIn2D::with_material(
///     1.0, (0.0, 0.0),
///     Shape2DCollider::Circle(1.0),
///     Material::steel()
/// );
/// let mut obj2 = ObjectIn2D::with_material(
///     1.0, (2.0, 0.0),
///     Shape2DCollider::Circle(1.0),
///     Material::rubber()
/// );
/// obj2.velocity.x = -5.0;  // Moving toward obj1
///
/// let result = material_collision_2d(&obj1, &obj2, (1.0, 0.0));
/// ```
pub fn material_collision_2d(
    obj1: &ObjectIn2D,
    obj2: &ObjectIn2D,
    contact_normal: (f64, f64),
) -> Collision2DResult {
    let m1 = obj1.mass;
    let m2 = obj2.mass;

    // Get material properties (use defaults if no material set)
    let restitution1 = obj1.get_restitution();
    let restitution2 = obj2.get_restitution();
    let friction1 = obj1.get_friction();
    let friction2 = obj2.get_friction();

    // Calculate effective coefficients
    let effective_restitution = (restitution1 + restitution2) / 2.0;
    let effective_friction = (friction1 * friction2).sqrt();

    // Normalize contact normal
    let normal_len = (contact_normal.0 * contact_normal.0 + contact_normal.1 * contact_normal.1).sqrt();
    let normal = if normal_len > 0.0 {
        (contact_normal.0 / normal_len, contact_normal.1 / normal_len)
    } else {
        (1.0, 0.0)
    };

    // Relative velocity
    let rel_vx = obj1.velocity.x - obj2.velocity.x;
    let rel_vy = obj1.velocity.y - obj2.velocity.y;

    // Relative velocity along normal
    let rel_vel_normal = rel_vx * normal.0 + rel_vy * normal.1;

    // Only process if objects are moving toward each other
    if rel_vel_normal > 0.0 {
        return Collision2DResult {
            velocity1: obj1.velocity.clone(),
            velocity2: obj2.velocity.clone(),
            #[cfg(feature = "thermodynamics")]
            heat_generated: 0.0,
        };
    }

    // Calculate impulse magnitude
    let impulse_magnitude = -(1.0 + effective_restitution) * rel_vel_normal / (1.0 / m1 + 1.0 / m2);

    // Apply impulse along normal
    let impulse_x = impulse_magnitude * normal.0;
    let impulse_y = impulse_magnitude * normal.1;

    let new_v1x = obj1.velocity.x + impulse_x / m1;
    let new_v1y = obj1.velocity.y + impulse_y / m1;
    let new_v2x = obj2.velocity.x - impulse_x / m2;
    let new_v2y = obj2.velocity.y - impulse_y / m2;

    // Calculate tangential (friction) component
    let tangent = (-normal.1, normal.0);
    let rel_vel_tangent = rel_vx * tangent.0 + rel_vy * tangent.1;

    // Apply friction impulse
    let friction_impulse_mag = effective_friction * impulse_magnitude.abs();
    let friction_impulse = friction_impulse_mag * rel_vel_tangent.signum();

    let mut final_v1x = new_v1x - friction_impulse * tangent.0 / m1;
    let mut final_v1y = new_v1y - friction_impulse * tangent.1 / m1;
    let mut final_v2x = new_v2x + friction_impulse * tangent.0 / m2;
    let mut final_v2y = new_v2y + friction_impulse * tangent.1 / m2;

    // Limit friction to not reverse tangential velocity
    if rel_vel_tangent.abs() < friction_impulse_mag * (1.0 / m1 + 1.0 / m2) {
        // Static friction case - zero out tangential velocity difference
        let avg_tangent_vel = (obj1.velocity.x * tangent.0 + obj1.velocity.y * tangent.1) * m1
            + (obj2.velocity.x * tangent.0 + obj2.velocity.y * tangent.1) * m2;
        let avg_tangent_vel = avg_tangent_vel / (m1 + m2);

        final_v1x = new_v1x - (obj1.velocity.x * tangent.0 + obj1.velocity.y * tangent.1 - avg_tangent_vel) * tangent.0;
        final_v1y = new_v1y - (obj1.velocity.x * tangent.0 + obj1.velocity.y * tangent.1 - avg_tangent_vel) * tangent.1;
        final_v2x = new_v2x - (obj2.velocity.x * tangent.0 + obj2.velocity.y * tangent.1 - avg_tangent_vel) * tangent.0;
        final_v2y = new_v2y - (obj2.velocity.x * tangent.0 + obj2.velocity.y * tangent.1 - avg_tangent_vel) * tangent.1;
    }

    // Calculate heat generated (energy lost to inelastic collision)
    #[cfg(feature = "thermodynamics")]
    let heat_generated = {
        let initial_ke = 0.5 * m1 * (obj1.velocity.x.powi(2) + obj1.velocity.y.powi(2))
            + 0.5 * m2 * (obj2.velocity.x.powi(2) + obj2.velocity.y.powi(2));
        let final_ke = 0.5 * m1 * (final_v1x.powi(2) + final_v1y.powi(2))
            + 0.5 * m2 * (final_v2x.powi(2) + final_v2y.powi(2));
        (initial_ke - final_ke).max(0.0)
    };

    Collision2DResult {
        velocity1: Velocity2D { x: final_v1x, y: final_v1y },
        velocity2: Velocity2D { x: final_v2x, y: final_v2y },
        #[cfg(feature = "thermodynamics")]
        heat_generated,
    }
}