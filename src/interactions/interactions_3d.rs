use crate::forces::Force;
use crate::models::{Axis3D, Direction3D, FromCoordinates, ObjectIn3D, Velocity3D};
use crate::utils::PhysicsConstants;

impl ObjectIn3D {
    /// Creates a new `ObjectIn3D` with the given mass, velocity components, and position.
    ///
    /// # Arguments
    /// * `mass` - The mass of the object in kilograms.
    /// * `vx` - The velocity in the x direction in meters per second.
    /// * `vy` - The velocity in the y direction in meters per second.
    /// * `vz` - The velocity in the z direction in meters per second.
    /// * `position` - The position as (x, y, z) coordinates.
    ///
    /// # Returns
    /// A new `ObjectIn3D` with the specified properties.
    ///
    /// # Example
    /// ```
    /// use rs_physics::models::ObjectIn3D;
    /// let obj = ObjectIn3D::new(1.0, 2.0, 3.0, 4.0, (0.0, 0.0, 0.0));
    /// assert_eq!(obj.mass, 1.0);
    /// assert_eq!(obj.velocity.x, 2.0);
    /// assert_eq!(obj.velocity.y, 3.0);
    /// assert_eq!(obj.velocity.z, 4.0);
    /// ```
    pub fn new(mass: f64, vx: f64, vy: f64, vz: f64, position: (f64, f64, f64)) -> Self {
        ObjectIn3D {
            mass,
            velocity: Velocity3D { x: vx, y: vy, z: vz },
            position: Axis3D::from_coord(position),
            forces: Vec::new(),
        }
    }

    /// Get the speed (magnitude of velocity)
    pub fn speed(&self) -> f64 {
        self.velocity.magnitude()
    }

    /// Get the direction as a normalized vector
    pub fn direction(&self) -> Direction3D {
        self.velocity.direction()
    }

    pub fn add_force(&mut self, force: Force) {
        self.forces.push(force);
    }

    pub fn clear_forces(&mut self) {
        self.forces.clear();
    }
}

/// Simulates a 3D elastic collision between two objects.
///
/// # Arguments
///
/// * `constants` - Structure containing gravity and air density.
/// * `obj1`, `obj2` - Mutable references to the two colliding objects.
/// * `collision_normal` - The normal vector to the collision surface (should be normalized).
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
/// use rs_physics::interactions::elastic_collision_3d;
/// use rs_physics::models::ObjectIn3D;
///
/// // Create objects with x, y, and z velocity components
/// let mut obj1 = ObjectIn3D::new(1.0, 2.0, 0.0, 0.0, (0.0, 0.0, 0.0));
/// let mut obj2 = ObjectIn3D::new(1.0, -1.0, 0.0, 0.0, (1.0, 0.0, 0.0));
/// // Normal pointing in x direction for a head-on collision
/// let normal = (1.0, 0.0, 0.0);
///
/// elastic_collision_3d(&rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS,
///                     &mut obj1, &mut obj2, normal, 1.0, 0.45, 1.0).unwrap();
///
/// assert!((obj1.velocity.x - (-1.0)).abs() < 1.0);
/// assert!((obj2.velocity.x - 2.0).abs() < 1.0);
/// ```
///
/// # Notes
///
/// This function generalizes the 2D elastic collision to 3D space by:
/// 1. Projecting the velocities onto the collision normal direction
/// 2. Applying an elastic collision on these projected components
/// 3. Keeping tangential velocity components unchanged
/// 4. Recombining all velocity components and applying gravity and drag effects
pub fn elastic_collision_3d(
    constants: &PhysicsConstants,
    obj1: &mut ObjectIn3D,
    obj2: &mut ObjectIn3D,
    collision_normal: (f64, f64, f64),
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

    // Normalize the collision normal if it's not already normalized
    let normal_len = (collision_normal.0 * collision_normal.0 +
        collision_normal.1 * collision_normal.1 +
        collision_normal.2 * collision_normal.2).sqrt();

    if normal_len == 0.0 {
        return Err("Collision normal cannot be zero vector");
    }

    let normal = (
        collision_normal.0 / normal_len,
        collision_normal.1 / normal_len,
        collision_normal.2 / normal_len
    );

    // Extract velocity components
    let v1 = (obj1.velocity.x, obj1.velocity.y, obj1.velocity.z);
    let v2 = (obj2.velocity.x, obj2.velocity.y, obj2.velocity.z);

    // Project velocities onto the normal direction (dot product)
    let v1_normal = v1.0 * normal.0 + v1.1 * normal.1 + v1.2 * normal.2;
    let v2_normal = v2.0 * normal.0 + v2.1 * normal.1 + v2.2 * normal.2;

    // Compute tangential components (velocity - projection_onto_normal)
    let v1_tangent = (
        v1.0 - v1_normal * normal.0,
        v1.1 - v1_normal * normal.1,
        v1.2 - v1_normal * normal.2
    );

    let v2_tangent = (
        v2.0 - v2_normal * normal.0,
        v2.1 - v2_normal * normal.1,
        v2.2 - v2_normal * normal.2
    );

    // Calculate new normal velocities after elastic collision
    let m1 = obj1.mass;
    let m2 = obj2.mass;
    let total_mass = m1 + m2;
    let mass_diff = m1 - m2;

    let v1_normal_final = (mass_diff * v1_normal + 2.0 * m2 * v2_normal) / total_mass;
    let v2_normal_final = (2.0 * m1 * v1_normal - mass_diff * v2_normal) / total_mass;

    // Apply gravity effect in the y-direction
    // Note: This assumes that gravity acts in the negative y direction
    let gravity_effect_y = constants.gravity * duration;

    // Calculate air resistance factor
    let air_res_factor = 0.5 * constants.air_density * drag_coefficient * cross_sectional_area * duration;

    // Compute final velocities by combining normal and tangential components
    let new_v1 = (
        v1_normal_final * normal.0 + v1_tangent.0,
        v1_normal_final * normal.1 + v1_tangent.1 - gravity_effect_y,
        v1_normal_final * normal.2 + v1_tangent.2
    );

    let new_v2 = (
        v2_normal_final * normal.0 + v2_tangent.0,
        v2_normal_final * normal.1 + v2_tangent.1 - gravity_effect_y,
        v2_normal_final * normal.2 + v2_tangent.2
    );

    // Apply air resistance
    let v1_speed = (new_v1.0 * new_v1.0 + new_v1.1 * new_v1.1 + new_v1.2 * new_v1.2).sqrt();
    let v2_speed = (new_v2.0 * new_v2.0 + new_v2.1 * new_v2.1 + new_v2.2 * new_v2.2).sqrt();

    let v1_drag = air_res_factor * v1_speed / m1;
    let v2_drag = air_res_factor * v2_speed / m2;

    // Update velocities with drag effect
    if v1_speed > 0.0 {
        let drag_factor = (v1_speed - v1_drag) / v1_speed;
        obj1.velocity.x = new_v1.0 * drag_factor;
        obj1.velocity.y = new_v1.1 * drag_factor;
        obj1.velocity.z = new_v1.2 * drag_factor;
    } else {
        obj1.velocity.x = new_v1.0;
        obj1.velocity.y = new_v1.1;
        obj1.velocity.z = new_v1.2;
    }

    if v2_speed > 0.0 {
        let drag_factor = (v2_speed - v2_drag) / v2_speed;
        obj2.velocity.x = new_v2.0 * drag_factor;
        obj2.velocity.y = new_v2.1 * drag_factor;
        obj2.velocity.z = new_v2.2 * drag_factor;
    } else {
        obj2.velocity.x = new_v2.0;
        obj2.velocity.y = new_v2.1;
        obj2.velocity.z = new_v2.2;
    }

    Ok(())
}

/// Calculates the gravitational force between two 3D objects.
///
/// # Arguments
/// * `constants` - The physics constants to use for the calculation.
/// * `obj1` - A reference to the first object.
/// * `obj2` - A reference to the second object.
///
/// # Returns
/// A tuple containing the x, y, and z components of the gravitational force vector in Newtons.
///
/// # Errors
/// Returns an error if the objects are at the same position.
///
/// # Example
/// ```
/// use rs_physics::interactions::gravitational_force_3d;
/// use rs_physics::models::ObjectIn3D;
/// use rs_physics::physics::create_constants;
///
/// let constants = create_constants(Some(6.67430e-11), None, None, None, None);
/// let obj1 = ObjectIn3D::new(5.97e24, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0)); // Earth
/// let obj2 = ObjectIn3D::new(7.34e22, 0.0, 0.0, 0.0, (3.84e8, 0.0, 0.0)); // Moon
///
/// let (fx, fy, fz) = gravitational_force_3d(&constants, &obj1, &obj2).unwrap();
/// // Force should be along the x-axis in this example
/// assert!(fx.abs() > 0.0);
/// assert!(fy.abs() < 1e-10);
/// assert!(fz.abs() < 1e-10);
/// ```
pub fn gravitational_force_3d(
    constants: &PhysicsConstants,
    obj1: &ObjectIn3D,
    obj2: &ObjectIn3D
) -> Result<(f64, f64, f64), &'static str> {
    // Calculate distance vector between objects
    let dx = obj2.position.x - obj1.position.x;
    let dy = obj2.position.y - obj1.position.y;
    let dz = obj2.position.z - obj1.position.z;

    // Calculate square of distance
    let distance_squared = dx * dx + dy * dy + dz * dz;

    if distance_squared == 0.0 {
        return Err("Objects cannot be at the same position");
    }

    // Calculate distance
    let distance = distance_squared.sqrt();

    // Calculate gravitational force magnitude using Newton's law of universal gravitation
    let force_magnitude = constants.gravity * obj1.mass * obj2.mass / distance_squared;

    // Calculate direction unit vector components
    let dir_x = dx / distance;
    let dir_y = dy / distance;
    let dir_z = dz / distance;

    // Return force components
    Ok((force_magnitude * dir_x, force_magnitude * dir_y, force_magnitude * dir_z))
}

/// Applies a force to a 3D object for a given time and updates its velocity and position.
///
/// # Arguments
/// * `constants` - The physics constants to use for the calculation.
/// * `obj` - A mutable reference to the object to which the force is applied.
/// * `force_vector` - The force vector (x, y, z) in Newtons.
/// * `time` - The duration for which the force is applied in seconds.
///
/// # Returns
/// `Ok(())` if the force was successfully applied and the object's state updated.
///
/// # Errors
/// Returns an error if the time is negative.
///
/// # Example
/// ```
/// use rs_physics::interactions::apply_force_3d;
/// use rs_physics::models::ObjectIn3D;
/// use rs_physics::utils::{PhysicsConstants, DEFAULT_PHYSICS_CONSTANTS};
///
/// let constants = DEFAULT_PHYSICS_CONSTANTS; // Gravity is 9.80665 m/s²
/// let mut obj = ObjectIn3D::new(1.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0));
/// apply_force_3d(&constants, &mut obj, (10.0, 5.0, 2.0), 1.0).unwrap();
///
/// // x-component: only affected by applied force
/// assert_eq!(obj.velocity.x, 10.0);  // a_x = F_x/m = 10/1 = 10 m/s²
/// assert_eq!(obj.position.x, 5.0);   // Δx = v₀t + ½at² = 0 + ½(10)(1)² = 5
///
/// // y-component: affected by both applied force and gravity
/// assert_eq!(obj.velocity.y, 5.0); // a_y = F_y/m = 5 (this is just the starting velocity in this case)
/// assert_eq!(obj.position.y, 2.5 - 0.5 * constants.gravity); // Δy = v₀t + ½(a_y)t²
///
/// // z-component: only affected by applied force
/// assert_eq!(obj.velocity.z, 2.0);  // a_z = F_z/m = 2/1 = 2 m/s²
/// assert_eq!(obj.position.z, 1.0);  // Δz = v₀t + ½at² = 0 + ½(2)(1)² = 1
pub fn apply_force_3d(
    constants: &PhysicsConstants,
    obj: &mut ObjectIn3D,
    force_vector: (f64, f64, f64),
    time: f64
) -> Result<(), &'static str> {
    if time < 0.0 {
        return Err("Time cannot be negative");
    }

    // Calculate accelerations in each direction (F = ma, so a = F/m)
    let ax = force_vector.0 / obj.mass;
    let ay = force_vector.1 / obj.mass;
    let az = force_vector.2 / obj.mass;

    // Store initial velocities
    let initial_vx = obj.velocity.x;
    let initial_vy = obj.velocity.y - constants.gravity;
    let initial_vz = obj.velocity.z;

    // Update velocities: v = v₀ + at
    obj.velocity.x += ax * time;
    obj.velocity.y += ay * time;
    obj.velocity.z += az * time;

    // Update positions using average velocity over the time period
    // Δx = (v₀ + v)/2 * t = v₀t + ½at²
    obj.position.x += 0.5 * (initial_vx + obj.velocity.x) * time;
    obj.position.y += 0.5 * (initial_vy + obj.velocity.y) * time;
    obj.position.z += 0.5 * (initial_vz + obj.velocity.z) * time;

    Ok(())
}

/// Calculates the cross product of two 3D vectors.
///
/// # Arguments
/// * `v1` - The first vector as a tuple (x, y, z).
/// * `v2` - The second vector as a tuple (x, y, z).
///
/// # Returns
/// The cross product vector as a tuple (x, y, z).
///
/// # Example
/// ```
/// use rs_physics::interactions::cross_product;
///
/// let v1 = (1.0, 0.0, 0.0);
/// let v2 = (0.0, 1.0, 0.0);
/// let result = cross_product(v1, v2);
///
/// assert_eq!(result, (0.0, 0.0, 1.0));
/// ```
pub fn cross_product(v1: (f64, f64, f64), v2: (f64, f64, f64)) -> (f64, f64, f64) {
    (
        v1.1 * v2.2 - v1.2 * v2.1,
        v1.2 * v2.0 - v1.0 * v2.2,
        v1.0 * v2.1 - v1.1 * v2.0
    )
}

/// Calculates the dot product of two 3D vectors.
///
/// # Arguments
/// * `v1` - The first vector as a tuple (x, y, z).
/// * `v2` - The second vector as a tuple (x, y, z).
///
/// # Returns
/// The dot product as a scalar.
///
/// # Example
/// ```
/// use rs_physics::interactions::dot_product;
///
/// let v1 = (1.0, 2.0, 3.0);
/// let v2 = (4.0, 5.0, 6.0);
/// let result = dot_product(v1, v2);
///
/// assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
/// ```
pub fn dot_product(v1: (f64, f64, f64), v2: (f64, f64, f64)) -> f64 {
    v1.0 * v2.0 + v1.1 * v2.1 + v1.2 * v2.2
}

/// Calculates the vector magnitude (length) of a 3D vector.
///
/// # Arguments
/// * `v` - The vector as a tuple (x, y, z).
///
/// # Returns
/// The magnitude of the vector.
///
/// # Example
/// ```
/// use rs_physics::interactions::vector_magnitude;
///
/// let v = (3.0, 4.0, 5.0);
/// let magnitude = vector_magnitude(v);
///
/// assert!((magnitude - 7.0710678118654755).abs() < 1e-10);
/// ```
pub fn vector_magnitude(v: (f64, f64, f64)) -> f64 {
    (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt()
}

/// Normalizes a 3D vector (makes it a unit vector).
///
/// # Arguments
/// * `v` - The vector to normalize as a tuple (x, y, z).
///
/// # Returns
/// The normalized vector as a tuple (x, y, z).
///
/// # Errors
/// Returns an error if the input is a zero vector.
///
/// # Example
/// ```
/// use rs_physics::interactions::normalize_vector;
///
/// let v = (3.0, 0.0, 4.0);
/// let normalized = normalize_vector(v).unwrap();
///
/// assert!((normalized.0 - 0.6).abs() < 1e-10);
/// assert!((normalized.1 - 0.0).abs() < 1e-10);
/// assert!((normalized.2 - 0.8).abs() < 1e-10);
/// ```
pub fn normalize_vector(v: (f64, f64, f64)) -> Result<(f64, f64, f64), &'static str> {
    let magnitude = vector_magnitude(v);

    if magnitude == 0.0 {
        return Err("Cannot normalize a zero vector");
    }

    Ok((v.0 / magnitude, v.1 / magnitude, v.2 / magnitude))
}

/// Rotates a 3D point around the x-axis.
///
/// # Arguments
/// * `point` - The point to rotate as a tuple (x, y, z).
/// * `angle` - The angle of rotation in radians.
///
/// # Returns
/// The rotated point as a tuple (x, y, z).
///
/// # Example
/// ```
/// use rs_physics::interactions::rotate_around_x;
/// use std::f64::consts::PI;
///
/// let point = (1.0, 1.0, 0.0);
/// let rotated = rotate_around_x(point, PI/2.0);
///
/// assert!((rotated.0 - 1.0).abs() < 1e-10);
/// assert!((rotated.1 - 0.0).abs() < 1e-10);
/// assert!((rotated.2 - 1.0).abs() < 1e-10);
/// ```
pub fn rotate_around_x(point: (f64, f64, f64), angle: f64) -> (f64, f64, f64) {
    let cos_angle = angle.cos();
    let sin_angle = angle.sin();

    (
        point.0,
        point.1 * cos_angle - point.2 * sin_angle,
        point.1 * sin_angle + point.2 * cos_angle
    )
}

/// Rotates a 3D point around the y-axis.
///
/// # Arguments
/// * `point` - The point to rotate as a tuple (x, y, z).
/// * `angle` - The angle of rotation in radians.
///
/// # Returns
/// The rotated point as a tuple (x, y, z).
///
/// # Example
/// ```
/// use rs_physics::interactions::rotate_around_y;
/// use std::f64::consts::PI;
///
/// let point = (1.0, 0.0, 1.0);
/// let rotated = rotate_around_y(point, PI/2.0);
///
/// assert!((rotated.0 - 1.0).abs() < 1e-10);
/// assert!((rotated.1 - 0.0).abs() < 1e-10);
/// assert!((rotated.2 - -1.0).abs() < 1e-10);
/// ```
pub fn rotate_around_y(point: (f64, f64, f64), angle: f64) -> (f64, f64, f64) {
    let cos_angle = angle.cos();
    let sin_angle = angle.sin();

    (
        point.0 * cos_angle + point.2 * sin_angle,
        point.1,
        -point.0 * sin_angle + point.2 * cos_angle
    )
}

/// Rotates a 3D point around the z-axis.
///
/// # Arguments
/// * `point` - The point to rotate as a tuple (x, y, z).
/// * `angle` - The angle of rotation in radians.
///
/// # Returns
/// The rotated point as a tuple (x, y, z).
///
/// # Example
/// ```
/// use rs_physics::interactions::rotate_around_z;
/// use std::f64::consts::PI;
///
/// let point = (1.0, 1.0, 0.0);
/// let rotated = rotate_around_z(point, PI/2.0);
///
/// assert!((rotated.0 - -1.0).abs() < 1e-10);
/// assert!((rotated.1 - 1.0).abs() < 1e-10);
/// assert!((rotated.2 - 0.0).abs() < 1e-10);
/// ```
pub fn rotate_around_z(point: (f64, f64, f64), angle: f64) -> (f64, f64, f64) {
    let cos_angle = angle.cos();
    let sin_angle = angle.sin();

    (
        point.0 * cos_angle - point.1 * sin_angle,
        point.0 * sin_angle + point.1 * cos_angle,
        point.2
    )
}

/// Calculates if two 3D spheres are colliding.
///
/// # Arguments
/// * `center1` - The center position of the first sphere as a tuple (x, y, z).
/// * `radius1` - The radius of the first sphere.
/// * `center2` - The center position of the second sphere as a tuple (x, y, z).
/// * `radius2` - The radius of the second sphere.
///
/// # Returns
/// `true` if the spheres are colliding, `false` otherwise.
///
/// # Example
/// ```
/// use rs_physics::interactions::spheres_colliding;
///
/// let center1 = (0.0, 0.0, 0.0);
/// let radius1 = 1.0;
/// let center2 = (1.5, 0.0, 0.0);
/// let radius2 = 1.0;
///
/// assert!(spheres_colliding(center1, radius1, center2, radius2));
///
/// let center3 = (3.0, 0.0, 0.0);
/// assert!(!spheres_colliding(center1, radius1, center3, radius2));
/// ```
pub fn spheres_colliding(
    center1: (f64, f64, f64),
    radius1: f64,
    center2: (f64, f64, f64),
    radius2: f64
) -> bool {
    let dx = center2.0 - center1.0;
    let dy = center2.1 - center1.1;
    let dz = center2.2 - center1.2;

    let distance_squared = dx * dx + dy * dy + dz * dz;
    let radius_sum = radius1 + radius2;

    distance_squared <= radius_sum * radius_sum
}

/// Calculates the collision normal between two 3D spheres.
///
/// # Arguments
/// * `center1` - The center position of the first sphere as a tuple (x, y, z).
/// * `center2` - The center position of the second sphere as a tuple (x, y, z).
///
/// # Returns
/// The normalized collision normal vector pointing from sphere1 to sphere2.
///
/// # Errors
/// Returns an error if the spheres are at the same position.
///
/// # Example
/// ```
/// use rs_physics::interactions::sphere_collision_normal;
///
/// let center1 = (0.0, 0.0, 0.0);
/// let center2 = (3.0, 4.0, 0.0);
///
/// let normal = sphere_collision_normal(center1, center2).unwrap();
/// assert!((normal.0 - 0.6).abs() < 1e-10);
/// assert!((normal.1 - 0.8).abs() < 1e-10);
/// assert!((normal.2 - 0.0).abs() < 1e-10);
/// ```
pub fn sphere_collision_normal(
    center1: (f64, f64, f64),
    center2: (f64, f64, f64)
) -> Result<(f64, f64, f64), &'static str> {
    let dx = center2.0 - center1.0;
    let dy = center2.1 - center1.1;
    let dz = center2.2 - center1.2;

    normalize_vector((dx, dy, dz))
}