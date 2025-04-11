use crate::interactions::{cross_product, dot_product};
use crate::models::{normalize_angle, rotate_point, PhysicalObject3D, Shape3D, ToCoordinates};
use rand::{rng, Rng};

/// Checks for collision between two 3D shapes at specified positions
///
/// Performs an efficient collision detection by first using a bounding sphere test
/// for quick rejection, then applying more precise shape-specific tests.
///
/// # Arguments
/// * `shape1` - Reference to the first shape
/// * `shape2` - Reference to the second shape
/// * `position1` - Position of the first shape as (x, y, z) coordinates
/// * `position2` - Position of the second shape as (x, y, z) coordinates
///
/// # Returns
/// `true` if the shapes are colliding, `false` otherwise
///
/// # Examples
/// ```
/// use rs_physics::interactions::shape_collisions_3d;
/// use rs_physics::models::Shape3D;
///
/// let sphere = Shape3D::new_sphere(1.0);
/// let cube = Shape3D::new_cuboid(2.0, 2.0, 2.0);
///
/// // Shapes are 2 units apart - not colliding
/// let colliding = shape_collisions_3d::check_collision(
///     &sphere, (0.0, 0.0, 0.0),
///     &cube, (3.0, 0.0, 0.0)
/// );
/// assert_eq!(colliding, false);
///
/// // Shapes are 1 unit apart - colliding
/// let colliding = shape_collisions_3d::check_collision(
///     &sphere, (0.0, 0.0, 0.0),
///     &cube, (2.0, 0.0, 0.0)
/// );
/// assert_eq!(colliding, true);
/// ```
pub fn check_collision(
    shape1: &Shape3D,
    position1: (f64, f64, f64),
    shape2: &Shape3D,
    position2: (f64, f64, f64),
) -> bool {
    // First, do a quick bounding sphere check for early rejection
    let dx = position2.0 - position1.0;
    let dy = position2.1 - position1.1;
    let dz = position2.2 - position1.2;

    let distance_squared = dx.powi(2) + dy.powi(2) + dz.powi(2);
    let r1 = shape1.bounding_radius();
    let r2 = shape2.bounding_radius();

    if distance_squared > (r1 + r2).powi(2) {
        return false;
    }

    // More specific collision detection based on shape types
    match (shape1, shape2) {
        // Sphere-sphere collision
        (Shape3D::Sphere(r1), Shape3D::Sphere(r2)) => distance_squared <= (r1 + r2).powi(2),

        // Cuboid-cuboid collision (AABB)
        (Shape3D::Cuboid(w1, h1, d1), Shape3D::Cuboid(w2, h2, d2))
        | (Shape3D::BeveledCuboid(w1, h1, d1, _), Shape3D::BeveledCuboid(w2, h2, d2, _))
        | (Shape3D::Cuboid(w1, h1, d1), Shape3D::BeveledCuboid(w2, h2, d2, _))
        | (Shape3D::BeveledCuboid(w1, h1, d1, _), Shape3D::Cuboid(w2, h2, d2)) => {
            // Axis-Aligned Bounding Box test
            let w1_half = w1 / 2.0;
            let h1_half = h1 / 2.0;
            let d1_half = d1 / 2.0;

            let w2_half = w2 / 2.0;
            let h2_half = h2 / 2.0;
            let d2_half = d2 / 2.0;

            // Check overlap in all three axes
            dx.abs() <= (w1_half + w2_half)
                && dy.abs() <= (h1_half + h2_half)
                && dz.abs() <= (d1_half + d2_half)
        }

        // Sphere-cuboid collision
        (Shape3D::Sphere(radius), Shape3D::Cuboid(w, h, d))
        | (Shape3D::Sphere(radius), Shape3D::BeveledCuboid(w, h, d, _)) => {
            // Find the closest point on the cuboid to the sphere
            let w_half = w / 2.0;
            let h_half = h / 2.0;
            let d_half = d / 2.0;

            // Find the closest point on the cuboid (clamped to cuboid bounds)
            let closest_x = (-dx).clamp(-w_half, w_half);
            let closest_y = (-dy).clamp(-h_half, h_half);
            let closest_z = (-dz).clamp(-d_half, d_half);

            // Calculate squared distance from closest point to sphere center
            let closest_dist_sq =
                (dx + closest_x).powi(2) + (dy + closest_y).powi(2) + (dz + closest_z).powi(2);

            // Collision if the closest point is within the sphere
            closest_dist_sq <= radius.powi(2)
        }

        // Cuboid-sphere collision (reverse of above)
        (Shape3D::Cuboid(w, h, d), Shape3D::Sphere(radius))
        | (Shape3D::BeveledCuboid(w, h, d, _), Shape3D::Sphere(radius)) => {
            // Just use the same logic with positions reversed
            let w_half = w / 2.0;
            let h_half = h / 2.0;
            let d_half = d / 2.0;

            let closest_x = dx.clamp(-w_half, w_half);
            let closest_y = dy.clamp(-h_half, h_half);
            let closest_z = dz.clamp(-d_half, d_half);

            let closest_dist_sq =
                (dx - closest_x).powi(2) + (dy - closest_y).powi(2) + (dz - closest_z).powi(2);

            closest_dist_sq <= radius.powi(2)
        }

        // Cylinder collisions
        (Shape3D::Cylinder(r1, h1), Shape3D::Cylinder(r2, h2)) => {
            // Check if cylinders overlap
            // First check height (y-axis)
            let h1_half = h1 / 2.0;
            let h2_half = h2 / 2.0;

            if dy.abs() > (h1_half + h2_half) {
                return false;
            }

            // Then check radius (xz-plane)
            let xz_dist_sq = dx.powi(2) + dz.powi(2);
            xz_dist_sq <= (r1 + r2).powi(2)
        }

        // For all other combinations, use simpler approximations
        _ => {
            // Fallback to the bounding sphere check we already did
            true
        }
    }
}

/// Calculates the collision normal vector between two colliding shapes
///
/// The normal vector points from the first shape toward the second shape
/// and can be used for collision response calculations.
///
/// # Arguments
/// * `shape1` - Reference to the first shape
/// * `shape2` - Reference to the second shape
/// * `position1` - Position of the first shape as (x, y, z) coordinates
/// * `position2` - Position of the second shape as (x, y, z) coordinates
///
/// # Returns
/// * `Some((nx, ny, nz))` - Normalized collision normal vector if shapes are colliding
/// * `None` - If shapes are not colliding
///
/// # Examples
/// ```
/// use rs_physics::interactions::shape_collisions_3d;
/// use rs_physics::models::Shape3D;
///
/// let sphere1 = Shape3D::new_sphere(1.0);
/// let sphere2 = Shape3D::new_sphere(1.0);
///
/// let normal = shape_collisions_3d::collision_normal(
///     &sphere1, (0.0, 0.0, 0.0),
///     &sphere2, (1.5, 0.0, 0.0)
/// );
///
/// assert!(normal.is_some());
/// let (nx, ny, nz) = normal.unwrap();
/// assert!((nx - 1.0).abs() < 1e-10); // Normal points along x-axis
/// assert!(ny.abs() < 1e-10);
/// assert!(nz.abs() < 1e-10);
/// ```
pub fn collision_normal(
    shape1: &Shape3D,
    position1: (f64, f64, f64),
    shape2: &Shape3D,
    position2: (f64, f64, f64),
) -> Option<(f64, f64, f64)> {
    if !check_collision(shape1, position1, shape2, position2) {
        return None;
    }

    // Vector from position1 to position2
    let dx = position2.0 - position1.0;
    let dy = position2.1 - position1.1;
    let dz = position2.2 - position1.2;

    let distance_squared = dx.powi(2) + dy.powi(2) + dz.powi(2);

    // If centers are at same position, return a default normal
    if distance_squared < 1e-10 {
        return Some((0.0, 0.0, 1.0));
    }

    // For sphere-sphere collision, the normal is the center-to-center vector
    match (shape1, shape2) {
        (Shape3D::Sphere(_), Shape3D::Sphere(_)) => {
            let distance = distance_squared.sqrt();
            Some((dx / distance, dy / distance, dz / distance))
        }

        // For cuboid collisions, find the minimum penetration axis
        (Shape3D::Cuboid(w1, h1, d1), Shape3D::Cuboid(w2, h2, d2))
        | (Shape3D::BeveledCuboid(w1, h1, d1, _), Shape3D::BeveledCuboid(w2, h2, d2, _))
        | (Shape3D::Cuboid(w1, h1, d1), Shape3D::BeveledCuboid(w2, h2, d2, _))
        | (Shape3D::BeveledCuboid(w1, h1, d1, _), Shape3D::Cuboid(w2, h2, d2)) => {
            let w1_half = w1 / 2.0;
            let h1_half = h1 / 2.0;
            let d1_half = d1 / 2.0;

            let w2_half = w2 / 2.0;
            let h2_half = h2 / 2.0;
            let d2_half = d2 / 2.0;

            // Calculate penetration depths in each axis
            let x_overlap = w1_half + w2_half - dx.abs();
            let y_overlap = h1_half + h2_half - dy.abs();
            let z_overlap = d1_half + d2_half - dz.abs();

            // Find minimum penetration axis
            if x_overlap < y_overlap && x_overlap < z_overlap {
                // X-axis has minimum penetration
                Some((dx.signum(), 0.0, 0.0))
            } else if y_overlap < z_overlap {
                // Y-axis has minimum penetration
                Some((0.0, dy.signum(), 0.0))
            } else {
                // Z-axis has minimum penetration
                Some((0.0, 0.0, dz.signum()))
            }
        }

        // For other combinations, use center-to-center as approximation
        _ => {
            // Normalize the vector
            let distance = distance_squared.sqrt();
            Some((dx / distance, dy / distance, dz / distance))
        }
    }
}

/// Checks for overlap between two sets of corners along a given axis
///
/// This is used in the Separating Axis Theorem (SAT) algorithm for oriented
/// bounding box collision detection.
///
/// # Arguments
/// * `corners1` - Slice of (x, y, z) coordinates for the first set of corners
/// * `corners2` - Slice of (x, y, z) coordinates for the second set of corners
/// * `axis` - The axis (as a direction vector) along which to check for overlap
///
/// # Returns
/// `true` if the corners overlap when projected onto the axis, `false` otherwise
pub fn check_overlap_along_axis(
    corners1: &[(f64, f64, f64)],
    corners2: &[(f64, f64, f64)],
    axis: &(f64, f64, f64),
) -> bool {
    // Project all corners onto axis
    let projections1: Vec<f64> = corners1
        .iter()
        .map(|c| c.0 * axis.0 + c.1 * axis.1 + c.2 * axis.2)
        .collect();

    let projections2: Vec<f64> = corners2
        .iter()
        .map(|c| c.0 * axis.0 + c.1 * axis.1 + c.2 * axis.2)
        .collect();

    // Find min and max projections
    let min1 = projections1.iter().fold(f64::MAX, |a, &b| a.min(b));
    let max1 = projections1.iter().fold(f64::MIN, |a, &b| a.max(b));
    let min2 = projections2.iter().fold(f64::MAX, |a, &b| a.min(b));
    let max2 = projections2.iter().fold(f64::MIN, |a, &b| a.max(b));

    // Check for overlap
    max1 >= min2 && max2 >= min1
}

/// Calculates the impact point between two colliding physical objects
///
/// The impact point is used for calculating collision impulses and
/// generating torque effects.
///
/// # Arguments
/// * `obj1` - Reference to the first physical object
/// * `obj2` - Reference to the second physical object
/// * `normal` - The collision normal vector pointing from obj1 to obj2
///
/// # Returns
/// The impact point as (x, y, z) coordinates
pub fn calculate_impact_point(
    obj1: &PhysicalObject3D,
    obj2: &PhysicalObject3D,
    normal: (f64, f64, f64),
) -> (f64, f64, f64) {
    let pos1 = obj1.object.position.to_coord();
    let pos2 = obj2.object.position.to_coord();

    // Default to midpoint if we can't determine better point
    let mut impact: (f64, f64, f64) = (
        (pos1.0 + pos2.0) / 2.0,
        (pos1.1 + pos2.1) / 2.0,
        (pos1.2 + pos2.2) / 2.0,
    );

    match (&obj1.shape, &obj2.shape) {
        // For sphere-sphere, impact point is along the line connecting centers
        (Shape3D::Sphere(r1), Shape3D::Sphere(_r2)) => {
            // Calculate distance between centers
            let dx = pos2.0 - pos1.0;
            let dy = pos2.1 - pos1.1;
            let dz = pos2.2 - pos1.2;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            if dist > 0.001 {
                // Impact point is at surface of first sphere along line to second sphere
                impact = (
                    pos1.0 + dx / dist * r1,
                    pos1.1 + dy / dist * r1,
                    pos1.2 + dz / dist * r1,
                );
            }
        }

        // For cuboid collisions, find the closest points on each cuboid
        (Shape3D::Cuboid(_, _, _), Shape3D::Cuboid(_, _, _))
        | (Shape3D::BeveledCuboid(_, _, _, _), Shape3D::BeveledCuboid(_, _, _, _))
        | (Shape3D::Cuboid(_, _, _), Shape3D::BeveledCuboid(_, _, _, _))
        | (Shape3D::BeveledCuboid(_, _, _, _), Shape3D::Cuboid(_, _, _)) => {
            // For cuboids, the closest corners can give a better impact point
            // Get world corners of both shapes
            let corners1 = obj1.get_corner_positions();
            let corners2 = obj2.get_corner_positions();

            if !corners1.is_empty() && !corners2.is_empty() {
                // Find pair of corners with smallest distance
                let mut min_dist = f64::MAX;
                let mut closest_pair: ((f64, f64, f64), (f64, f64, f64)) =
                    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0));

                for c1 in &corners1 {
                    for c2 in &corners2 {
                        let dx = c2.0 - c1.0;
                        let dy = c2.1 - c1.1;
                        let dz = c2.2 - c1.2;
                        let dist_sq = dx * dx + dy * dy + dz * dz;

                        if dist_sq < min_dist {
                            min_dist = dist_sq;
                            closest_pair = (*c1, *c2);
                        }
                    }
                }

                // Impact point is midway between closest corners
                impact = (
                    (closest_pair.0 .0 + closest_pair.1 .0) / 2.0,
                    (closest_pair.0 .1 + closest_pair.1 .1) / 2.0,
                    (closest_pair.0 .2 + closest_pair.1 .2) / 2.0,
                );
            }
        }

        // For sphere-cuboid, impact is on sphere surface nearest cuboid
        (Shape3D::Sphere(radius), Shape3D::Cuboid(_, _, _))
        | (Shape3D::Sphere(radius), Shape3D::BeveledCuboid(_, _, _, _)) => {
            // Impact point is along normal direction from sphere center
            impact = (
                pos1.0 + normal.0 * radius,
                pos1.1 + normal.1 * radius,
                pos1.2 + normal.2 * radius,
            );
        }

        (Shape3D::Cuboid(_, _, _), Shape3D::Sphere(radius))
        | (Shape3D::BeveledCuboid(_, _, _, _), Shape3D::Sphere(radius)) => {
            // Impact point is along normal direction from sphere center (opposite direction)
            impact = (
                pos2.0 - normal.0 * radius,
                pos2.1 - normal.1 * radius,
                pos2.2 - normal.2 * radius,
            );
        }

        // Default case - use midpoint between centers
        _ => { /* impact already set to midpoint */ }
    }

    impact
}

/// Calculates the velocity at a specific point on a physical object
///
/// The point velocity includes both linear velocity and rotational
/// contributions.
///
/// # Arguments
/// * `obj` - Reference to the physical object
/// * `r` - The relative position vector from object center to the point
///
/// # Returns
/// The total velocity at the point as (vx, vy, vz)
pub fn calculate_point_velocity(obj: &PhysicalObject3D, r: (f64, f64, f64)) -> (f64, f64, f64) {
    // Linear velocity
    let v_linear = (
        obj.object.velocity.x,
        obj.object.velocity.y,
        obj.object.velocity.z,
    );

    // Angular contribution: v_angular = ω × r
    let v_angular = (
        obj.angular_velocity.1 * r.2 - obj.angular_velocity.2 * r.1,
        obj.angular_velocity.2 * r.0 - obj.angular_velocity.0 * r.2,
        obj.angular_velocity.0 * r.1 - obj.angular_velocity.1 * r.0,
    );

    // Total velocity at point
    (
        v_linear.0 + v_angular.0,
        v_linear.1 + v_angular.1,
        v_linear.2 + v_angular.2,
    )
}

/// Calculates the collision impulse magnitude between two physical objects
///
/// This calculation accounts for both linear and angular momentum conservation.
///
/// # Arguments
/// * `obj1` - Reference to the first physical object
/// * `obj2` - Reference to the second physical object
/// * `vrel` - Relative velocity between impact points on the objects
/// * `normal` - The collision normal vector pointing from obj1 to obj2
/// * `restitution` - Coefficient of restitution (0.0 = perfectly inelastic, 1.0 = perfectly elastic)
/// * `r1` - The relative position vector from obj1's center to impact point
/// * `r2` - The relative position vector from obj2's center to impact point
///
/// # Returns
/// The magnitude of the collision impulse
pub fn calculate_collision_impulse(
    obj1: &PhysicalObject3D,
    obj2: &PhysicalObject3D,
    vrel: (f64, f64, f64),
    normal: (f64, f64, f64),
    restitution: f64,
    r1: (f64, f64, f64),
    r2: (f64, f64, f64),
) -> f64 {
    // Normal component of relative velocity
    let vrel_n = dot_product(vrel, normal);

    // Calculate angular contribution to impulse denominator
    let inertia1 = obj1.shape.moment_of_inertia(obj1.object.mass);
    let inertia2 = obj2.shape.moment_of_inertia(obj2.object.mass);

    // Calculate r × n for both objects
    let r1_cross_n = cross_product(r1, normal);
    let r2_cross_n = cross_product(r2, normal);

    // Calculate angular terms
    let angular_term1 = r1_cross_n.0 * r1_cross_n.0 / inertia1[0]
        + r1_cross_n.1 * r1_cross_n.1 / inertia1[1]
        + r1_cross_n.2 * r1_cross_n.2 / inertia1[2];

    let angular_term2 = r2_cross_n.0 * r2_cross_n.0 / inertia2[0]
        + r2_cross_n.1 * r2_cross_n.1 / inertia2[1]
        + r2_cross_n.2 * r2_cross_n.2 / inertia2[2];

    // Calculate full impulse magnitude with rotational components
    let impulse_denom =
        1.0 / obj1.object.mass + 1.0 / obj2.object.mass + angular_term1 + angular_term2;
    let impulse_mag = -(1.0 + restitution) * vrel_n / impulse_denom;

    impulse_mag
}

/// Applies a linear impulse to two physical objects
///
/// The impulse is applied in opposite directions to maintain momentum conservation.
///
/// # Arguments
/// * `obj1` - Mutable reference to the first physical object
/// * `obj2` - Mutable reference to the second physical object
/// * `normal` - The direction of the impulse
/// * `impulse_mag` - The magnitude of the impulse
pub fn apply_linear_impulse(
    obj1: &mut PhysicalObject3D,
    obj2: &mut PhysicalObject3D,
    normal: (f64, f64, f64),
    impulse_mag: f64,
) {
    let impulse = (
        normal.0 * impulse_mag,
        normal.1 * impulse_mag,
        normal.2 * impulse_mag,
    );

    // Apply to first object
    obj1.object.velocity.x += impulse.0 / obj1.object.mass;
    obj1.object.velocity.y += impulse.1 / obj1.object.mass;
    obj1.object.velocity.z += impulse.2 / obj1.object.mass;

    // Apply to second object (opposite direction)
    obj2.object.velocity.x -= impulse.0 / obj2.object.mass;
    obj2.object.velocity.y -= impulse.1 / obj2.object.mass;
    obj2.object.velocity.z -= impulse.2 / obj2.object.mass;
}

/// Applies an angular impulse to two physical objects
///
/// This generates rotational effects based on the collision impulse.
///
/// # Arguments
/// * `obj1` - Mutable reference to the first physical object
/// * `obj2` - Mutable reference to the second physical object
/// * `normal` - The direction of the impulse
/// * `impulse_mag` - The magnitude of the impulse
/// * `r1` - The relative position vector from obj1's center to impact point
/// * `r2` - The relative position vector from obj2's center to impact point
/// * `dt` - The time step duration in seconds
pub fn apply_angular_impulse(
    obj1: &mut PhysicalObject3D,
    obj2: &mut PhysicalObject3D,
    normal: (f64, f64, f64),
    impulse_mag: f64,
    r1: (f64, f64, f64),
    r2: (f64, f64, f64),
    dt: f64,
) {
    let impulse = (
        normal.0 * impulse_mag,
        normal.1 * impulse_mag,
        normal.2 * impulse_mag,
    );

    // Calculate torque from impulse
    let torque1 = cross_product(r1, impulse);
    let torque2 = cross_product(r2, (-impulse.0, -impulse.1, -impulse.2));

    // Get moments of inertia
    let inertia1 = obj1.shape.moment_of_inertia(obj1.object.mass);
    let inertia2 = obj2.shape.moment_of_inertia(obj2.object.mass);

    // Apply angular impulse with appropriate scaling
    let angular_response_factor = 0.8;

    obj1.angular_velocity.0 += torque1.0 / inertia1[0] * dt * angular_response_factor;
    obj1.angular_velocity.1 += torque1.1 / inertia1[1] * dt * angular_response_factor;
    obj1.angular_velocity.2 += torque1.2 / inertia1[2] * dt * angular_response_factor;

    obj2.angular_velocity.0 += torque2.0 / inertia2[0] * dt * angular_response_factor;
    obj2.angular_velocity.1 += torque2.1 / inertia2[1] * dt * angular_response_factor;
    obj2.angular_velocity.2 += torque2.2 / inertia2[2] * dt * angular_response_factor;

    // Add small random perturbation to prevent "stuck" scenarios
    let random_factor = 0.05;

    if impulse_mag > 0.1 {
        // Only add randomness for significant collisions
        obj1.angular_velocity.0 += rng().random_range(-random_factor..random_factor);
        obj1.angular_velocity.1 += rng().random_range(-random_factor..random_factor);
        obj1.angular_velocity.2 += rng().random_range(-random_factor..random_factor);
    }
}

/// Resolves penetration between two colliding objects
///
/// Adjusts the positions of both objects to prevent them from occupying
/// the same space, with adjustments proportional to their masses.
///
/// # Arguments
/// * `obj1` - Mutable reference to the first physical object
/// * `obj2` - Mutable reference to the second physical object
/// * `normal` - The collision normal vector pointing from obj1 to obj2
pub fn resolve_penetration(
    obj1: &mut PhysicalObject3D,
    obj2: &mut PhysicalObject3D,
    normal: (f64, f64, f64),
) {
    let pos1 = obj1.object.position.to_coord();
    let pos2 = obj2.object.position.to_coord();

    // Calculate penetration based on shape types
    let penetration = match (&obj1.shape, &obj2.shape) {
        (Shape3D::BeveledCuboid(w1, h1, d1, _), Shape3D::BeveledCuboid(w2, h2, d2, _))
        | (Shape3D::Cuboid(w1, h1, d1), Shape3D::Cuboid(w2, h2, d2))
        | (Shape3D::BeveledCuboid(w1, h1, d1, _), Shape3D::Cuboid(w2, h2, d2))
        | (Shape3D::Cuboid(w1, h1, d1), Shape3D::BeveledCuboid(w2, h2, d2, _)) => {
            let w1_half = w1 / 2.0;
            let h1_half = h1 / 2.0;
            let d1_half = d1 / 2.0;

            let w2_half = w2 / 2.0;
            let h2_half = h2 / 2.0;
            let d2_half = d2 / 2.0;

            // Vector from obj1 to obj2
            let dx = pos2.0 - pos1.0;
            let dy = pos2.1 - pos1.1;
            let dz = pos2.2 - pos1.2;

            // Calculate overlap in each axis
            let overlap_x = (w1_half + w2_half) - dx.abs();
            let overlap_y = (h1_half + h2_half) - dy.abs();
            let overlap_z = (d1_half + d2_half) - dz.abs();

            // Find minimum overlap
            overlap_x.min(overlap_y).min(overlap_z)
        }

        (Shape3D::Sphere(r1), Shape3D::Sphere(r2)) => {
            // Calculate distance between centers
            let dx = pos2.0 - pos1.0;
            let dy = pos2.1 - pos1.1;
            let dz = pos2.2 - pos1.2;
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();

            // Penetration is overlap amount
            (r1 + r2) - distance
        }

        // For other shape combinations, use a small default value
        _ => 0.01,
    };

    // Apply correction only if there is actual penetration
    let correction_threshold = 0.001;
    let correction_percentage = 0.8;

    if penetration > correction_threshold {
        let correction = penetration * correction_percentage;
        let correction_vector = (
            normal.0 * correction,
            normal.1 * correction,
            normal.2 * correction,
        );

        // Apply correction inversely proportional to mass
        let total_mass = obj1.object.mass + obj2.object.mass;
        let self_ratio = obj2.object.mass / total_mass;
        let other_ratio = obj1.object.mass / total_mass;

        // Move both objects apart
        obj1.object.position.x -= correction_vector.0 * self_ratio;
        obj1.object.position.y -= correction_vector.1 * self_ratio;
        obj1.object.position.z -= correction_vector.2 * self_ratio;

        obj2.object.position.x += correction_vector.0 * other_ratio;
        obj2.object.position.y += correction_vector.1 * other_ratio;
        obj2.object.position.z += correction_vector.2 * other_ratio;
    }
}

/// Handles a collision between two physical objects
///
/// Performs collision detection, calculates response impulses,
/// and resolves penetration between the objects.
///
/// # Arguments
/// * `obj1` - Mutable reference to the first physical object
/// * `obj2` - Mutable reference to the second physical object
/// * `dt` - The time step duration in seconds
///
/// # Returns
/// `true` if a collision was detected and handled, `false` otherwise
pub fn handle_collision(obj1: &mut PhysicalObject3D, obj2: &mut PhysicalObject3D, dt: f64) -> bool {
    let pos1 = obj1.object.position.to_coord();
    let pos2 = obj2.object.position.to_coord();

    // First perform collision detection
    if !check_collision_objects(&obj1, &obj2) {
        return false;
    }

    // Get collision normal
    if let Some(normal) = collision_normal(&obj1.shape, pos1, &obj2.shape, pos2) {
        // Calculate combined coefficient of restitution
        let restitution = (obj1.get_restitution() + obj2.get_restitution()) / 2.0;

        // Calculate impact point
        let impact_point = calculate_impact_point(obj1, obj2, normal);

        // Calculate relative vectors from centers to impact point
        let r1 = (
            impact_point.0 - pos1.0,
            impact_point.1 - pos1.1,
            impact_point.2 - pos1.2,
        );

        let r2 = (
            impact_point.0 - pos2.0,
            impact_point.1 - pos2.1,
            impact_point.2 - pos2.2,
        );

        // Calculate point velocities including rotation
        let v1 = calculate_point_velocity(obj1, r1);
        let v2 = calculate_point_velocity(obj2, r2);

        // Relative velocity at impact point
        let vrel = (v1.0 - v2.0, v1.1 - v2.1, v1.2 - v2.2);

        // Calculate normal component of relative velocity
        let vrel_n = dot_product(vrel, normal);

        // Only proceed with collision response if objects are moving toward each other
        if vrel_n < 0.0 {
            // Calculate collision impulse
            let impulse_mag =
                calculate_collision_impulse(obj1, obj2, vrel, normal, restitution, r1, r2);

            // Apply impulse to linear velocities
            apply_linear_impulse(obj1, obj2, normal, impulse_mag);

            // Apply impulse to angular velocities
            apply_angular_impulse(obj1, obj2, normal, impulse_mag, r1, r2, dt);

            // Resolve penetration
            resolve_penetration(obj1, obj2, normal);

            return true;
        }
    }

    false
}

/// Handles collision between a cuboid and the ground
///
/// Resolves penetration, applies bounce physics, and calculates
/// appropriate torque effects.
///
/// # Arguments
/// * `obj` - Mutable reference to the physical object
/// * `min_y` - The y-coordinate of the lowest point of the cuboid
/// * `corners` - The corners of the cuboid in world space
/// * `dt` - The time step duration in seconds
pub fn handle_cuboid_ground_collision(
    obj: &mut PhysicalObject3D,
    min_y: f64,
    corners: &[(f64, f64, f64)],
    dt: f64,
) {
    // Calculate penetration depth
    let penetration = obj.physics_constants.ground_level - min_y;

    // Adjust position to resolve penetration
    obj.object.position.y += penetration;

    // Apply bounce physics if moving downward
    if obj.object.velocity.y < 0.0 {
        // Calculate which corners are in contact with the ground
        let ground_corners: Vec<(f64, f64, f64)> = corners
            .iter()
            .filter(|(_, y, _)| (*y - obj.physics_constants.ground_level).abs() < 0.01)
            .cloned()
            .collect();

        // Only apply bounce if we have ground contact
        if !ground_corners.is_empty() {
            // Bounce with appropriate energy loss
            let restitution = obj.get_restitution() * 0.8;
            obj.object.velocity.y = -obj.object.velocity.y * restitution;

            // Friction coefficients
            let sliding_friction = 0.7;
            let rolling_friction = 0.4;

            // Apply friction to horizontal velocity
            obj.object.velocity.x *= 1.0 - sliding_friction * dt;
            obj.object.velocity.z *= 1.0 - sliding_friction * dt;

            // Apply friction to angular velocity
            let angular_damping = rolling_friction * dt;
            obj.angular_velocity.0 *= 1.0 - angular_damping;
            obj.angular_velocity.2 *= 1.0 - angular_damping;

            // Calculate torque based on impact with ground
            if obj.object.velocity.y.abs() > 0.1 {
                // Calculate impact point (average of ground corners)
                let impact_point: (f64, f64, f64) = if !ground_corners.is_empty() {
                    let sum = ground_corners.iter().fold((0.0, 0.0, 0.0), |acc, &p| {
                        (acc.0 + p.0, acc.1 + p.1, acc.2 + p.2)
                    });
                    let count = ground_corners.len() as f64;
                    (sum.0 / count, sum.1 / count, sum.2 / count)
                } else {
                    (
                        obj.object.position.x,
                        obj.physics_constants.ground_level,
                        obj.object.position.z,
                    )
                };

                // Calculate relative vector from center to impact point
                let r: (f64, f64, f64) = (
                    impact_point.0 - obj.object.position.x,
                    impact_point.1 - obj.object.position.y,
                    impact_point.2 - obj.object.position.z,
                );

                // Calculate impact force (simplified as vertical impulse)
                let impact_force = (0.0, -obj.object.velocity.y * obj.object.mass, 0.0);

                // Calculate torque as cross product
                let torque: (f64, f64, f64) = cross_product(r, impact_force);

                // Scale torque effect
                let torque_factor = 0.2;
                apply_torque(
                    obj,
                    (
                        torque.0 * torque_factor,
                        torque.1 * torque_factor,
                        torque.2 * torque_factor,
                    ),
                    dt,
                );
            }
        }
    }
}

/// Handles collision between a sphere and the ground
///
/// Resolves penetration, applies bounce physics, and calculates
/// rolling effects.
///
/// # Arguments
/// * `obj` - Mutable reference to the physical object
/// * `radius` - The radius of the sphere
/// * `dt` - The time step duration in seconds
pub fn handle_sphere_ground_collision(obj: &mut PhysicalObject3D, radius: f64, dt: f64) {
    // Calculate penetration depth
    let sphere_bottom = obj.object.position.y - radius;
    let penetration = obj.physics_constants.ground_level - sphere_bottom;

    // Adjust position to resolve penetration
    obj.object.position.y += penetration;

    // Apply bounce physics if moving downward
    if obj.object.velocity.y < 0.0 {
        // Bounce with energy loss
        let restitution = obj.get_restitution();
        obj.object.velocity.y = -obj.object.velocity.y * restitution;

        // Apply rolling friction
        let friction = 0.5; // Rolling friction coefficient

        // Calculate friction force direction (opposite to velocity)
        let speed_sq = obj.object.velocity.x * obj.object.velocity.x
            + obj.object.velocity.z * obj.object.velocity.z;

        if speed_sq > 0.001 {
            let speed = speed_sq.sqrt();
            let friction_force_x = -obj.object.velocity.x / speed * friction;
            let friction_force_z = -obj.object.velocity.z / speed * friction;

            // Apply friction to slow down horizontal motion
            obj.object.velocity.x += friction_force_x * dt;
            obj.object.velocity.z += friction_force_z * dt;

            // Convert linear velocity to angular (rolling without slipping)
            obj.angular_velocity.0 = -obj.object.velocity.z / radius;
            obj.angular_velocity.2 = obj.object.velocity.x / radius;
        }
    }
}

/// Handles collision between a cylinder and the ground
///
/// Resolves penetration, applies bounce physics, and calculates
/// orientation-dependent rolling or stability effects.
///
/// # Arguments
/// * `obj` - Mutable reference to the physical object
/// * `radius` - The radius of the cylinder
/// * `height` - The height of the cylinder
/// * `dt` - The time step duration in seconds
pub fn handle_cylinder_ground_collision(
    obj: &mut PhysicalObject3D,
    radius: f64,
    height: f64,
    dt: f64,
) {
    // Calculate penetration depth
    let half_height = height / 2.0;
    let bottom_y = obj.object.position.y - half_height;
    let penetration = obj.physics_constants.ground_level - bottom_y;

    // Adjust position to resolve penetration
    obj.object.position.y += penetration;

    // Apply bounce physics if moving downward
    if obj.object.velocity.y < 0.0 {
        // Bounce with energy loss
        let restitution = obj.get_restitution() * 0.9; // Cylinders bounce a bit better
        obj.object.velocity.y = -obj.object.velocity.y * restitution;

        // Apply friction based on orientation
        // For a cylinder, the friction depends on whether it's on its side or its end

        // Calculate the up vector in world space
        let cylinder_up = rotate_point((0.0, 1.0, 0.0), obj.orientation.to_tuple());
        let up_dot_world_up = cylinder_up.1; // Dot product with world up (0,1,0)

        // If cylinder is more vertical (on its end)
        if up_dot_world_up.abs() > 0.7 {
            // Higher friction (cylinder standing on end doesn't roll well)
            let friction = 0.8;
            obj.object.velocity.x *= 1.0 - friction * dt;
            obj.object.velocity.z *= 1.0 - friction * dt;

            // Little angular velocity
            obj.angular_velocity.0 *= 0.9;
            obj.angular_velocity.2 *= 0.9;
        } else {
            // Lower friction (cylinder can roll on its side)
            let friction = 0.3;
            obj.object.velocity.x *= 1.0 - friction * dt;
            obj.object.velocity.z *= 1.0 - friction * dt;

            // Calculate rolling axis (perpendicular to both cylinder axis and ground normal)
            let cylinder_axis = rotate_point((0.0, 1.0, 0.0), obj.orientation.to_tuple());
            let ground_normal = (0.0, 1.0, 0.0);

            let roll_axis = cross_product(cylinder_axis, ground_normal);
            let roll_axis_length =
                (roll_axis.0 * roll_axis.0 + roll_axis.1 * roll_axis.1 + roll_axis.2 * roll_axis.2)
                    .sqrt();

            if roll_axis_length > 0.001 {
                // Normalize roll axis
                let roll_dir = (
                    roll_axis.0 / roll_axis_length,
                    roll_axis.1 / roll_axis_length,
                    roll_axis.2 / roll_axis_length,
                );

                // Rolling velocity (scalar)
                let vel_magnitude = (obj.object.velocity.x * obj.object.velocity.x
                    + obj.object.velocity.z * obj.object.velocity.z)
                    .sqrt();

                // Set angular velocity for rolling (proportional to linear velocity)
                let angular_speed = vel_magnitude / radius;
                obj.angular_velocity.0 = roll_dir.0 * angular_speed;
                obj.angular_velocity.1 = roll_dir.1 * angular_speed;
                obj.angular_velocity.2 = roll_dir.2 * angular_speed;
            }
        }
    }
}

/// Handles collision between a polyhedron and the ground
///
/// Resolves penetration, applies bounce physics with appropriate
/// torque for complex shapes.
///
/// # Arguments
/// * `obj` - Mutable reference to the physical object
/// * `vertices` - The vertices of the polyhedron in world space
/// * `dt` - The time step duration in seconds
pub fn handle_polyhedron_ground_collision(
    obj: &mut PhysicalObject3D,
    vertices: &[(f64, f64, f64)],
    dt: f64,
) {
    // Find vertices that are in contact with the ground
    let ground_vertices: Vec<(f64, f64, f64)> = vertices
        .iter()
        .filter(|(_, y, _)| (*y - obj.physics_constants.ground_level).abs() < 0.01)
        .cloned()
        .collect();

    // Find lowest vertex
    let min_y = vertices.iter().map(|(_, y, _)| *y).fold(f64::MAX, f64::min);

    // Calculate penetration depth
    let penetration = obj.physics_constants.ground_level - min_y;

    // Adjust position to resolve penetration
    obj.object.position.y += penetration;

    // Only apply collision response if moving downward
    if obj.object.velocity.y < 0.0 && !ground_vertices.is_empty() {
        // Bounce with energy loss
        let restitution = obj.get_restitution() * 0.7; // Polyhedra tend to bounce less
        obj.object.velocity.y = -obj.object.velocity.y * restitution;

        // Apply friction
        let friction = 0.6;
        obj.object.velocity.x *= 1.0 - friction * dt;
        obj.object.velocity.z *= 1.0 - friction * dt;

        // Apply angular damping
        let angular_damping = 0.5 * dt;
        obj.angular_velocity.0 *= 1.0 - angular_damping;
        obj.angular_velocity.1 *= 1.0 - angular_damping;
        obj.angular_velocity.2 *= 1.0 - angular_damping;

        // Calculate impact torque if we have ground contact
        if !ground_vertices.is_empty() && obj.object.velocity.y.abs() > 0.1 {
            // Calculate impact point (average of ground vertices)
            let sum = ground_vertices.iter().fold((0.0, 0.0, 0.0), |acc, &p| {
                (acc.0 + p.0, acc.1 + p.1, acc.2 + p.2)
            });
            let count = ground_vertices.len() as f64;
            let impact_point = (sum.0 / count, sum.1 / count, sum.2 / count);

            // Calculate relative vector from center to impact
            let r = (
                impact_point.0 - obj.object.position.x,
                impact_point.1 - obj.object.position.y,
                impact_point.2 - obj.object.position.z,
            );

            // Impact force (simplified vertical impulse)
            let impact_force = (0.0, -obj.object.velocity.y * obj.object.mass, 0.0);

            // Calculate torque
            let torque = cross_product(r, impact_force);

            // Apply with scaling factor
            let torque_factor = 0.15; // Slightly less torque for polyhedra
            apply_torque(
                obj,
                (
                    torque.0 * torque_factor,
                    torque.1 * torque_factor,
                    torque.2 * torque_factor,
                ),
                dt,
            );
        }
    }
}

/// Applies torque (rotational force) to a physical object
///
/// # Arguments
/// * `obj` - Mutable reference to the physical object
/// * `torque` - The torque vector as (tx, ty, tz)
/// * `dt` - The time step duration in seconds
pub fn apply_torque(obj: &mut PhysicalObject3D, torque: (f64, f64, f64), dt: f64) {
    // Get the moment of inertia tensor
    let inertia = obj.shape.moment_of_inertia(obj.object.mass);

    // Apply torque
    obj.angular_velocity.0 += torque.0 * dt / inertia[0];
    obj.angular_velocity.1 += torque.1 * dt / inertia[1];
    obj.angular_velocity.2 += torque.2 * dt / inertia[2];
}

/// Updates the physics state of an object for a single time step
///
/// Updates position and orientation based on velocities, and handles
/// collision with the ground if applicable.
///
/// # Arguments
/// * `obj` - Mutable reference to the physical object to update
/// * `dt` - The time step duration in seconds
///
/// # Examples
/// ```
///
/// // Create a sphere with gravity and initial velocity
/// use rs_physics::interactions::shape_collisions_3d;
/// use rs_physics::models::{PhysicalObject3D, Shape3D};
/// use rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS;
/// let mut sphere = PhysicalObject3D::new(
///     1.0, // mass
///     (1.0, 0.0, 0.0), // initial velocity
///     (0.0, 5.0, 0.0), // initial position
///     Shape3D::new_sphere(1.0), // shape
///     None, // material
///     (0.0, 0.0, 0.0), // angular velocity
///     (0.0, 0.0, 0.0), // orientation
///     DEFAULT_PHYSICS_CONSTANTS
/// );
///
/// // Apply gravity and update for 0.1 seconds
/// shape_collisions_3d::apply_gravity(&mut sphere, 9.8, 0.1);
/// shape_collisions_3d::update_physics(&mut sphere, 0.1);
///
/// // Object should have moved and accelerated due to gravity
/// assert!(sphere.object.position.x > 0.0);
/// assert!(sphere.object.position.y < 5.0);
/// assert!(sphere.object.velocity.y < 0.0);
/// ```
pub fn update_physics(obj: &mut PhysicalObject3D, dt: f64) {
    // Update position based on linear velocity
    obj.object.position.x += obj.object.velocity.x * dt;
    obj.object.position.y += obj.object.velocity.y * dt;
    obj.object.position.z += obj.object.velocity.z * dt;

    // Update orientation based on angular velocity
    obj.orientation.roll += obj.angular_velocity.0 * dt;
    obj.orientation.pitch += obj.angular_velocity.1 * dt;
    obj.orientation.yaw += obj.angular_velocity.2 * dt;

    // Normalize angles to [0, 2π)
    obj.orientation.roll = normalize_angle(obj.orientation.roll);
    obj.orientation.pitch = normalize_angle(obj.orientation.pitch);
    obj.orientation.yaw = normalize_angle(obj.orientation.yaw);

    // Handle ground collision based on shape type
    match &obj.shape {
        Shape3D::BeveledCuboid(_width, _height, _depth, _) => {
            // Calculate the positions of the 8 corners of the cube
            let corners: [(f64, f64, f64); 8] = obj.get_corner_positions();

            // Find the lowest point (corner closest to the ground)
            let min_y = corners.iter().map(|(_, y, _)| *y).fold(f64::MAX, f64::min);

            // If the lowest point is below ground level
            if min_y < obj.physics_constants.ground_level {
                handle_cuboid_ground_collision(obj, min_y, &corners, dt);
            }
        }
        Shape3D::Cuboid(_width, _height, _depth) => {
            // Reuse the same code for regular cuboids
            let corners: [(f64, f64, f64); 8] = obj.get_corner_positions();
            let min_y = corners.iter().map(|(_, y, _)| *y).fold(f64::MAX, f64::min);

            if min_y < obj.physics_constants.ground_level {
                handle_cuboid_ground_collision(obj, min_y, &corners, dt);
            }
        }
        Shape3D::Sphere(radius) => {
            // For sphere, just check if the bottom point is below ground
            let sphere_bottom = obj.object.position.y - radius;

            if sphere_bottom < obj.physics_constants.ground_level {
                handle_sphere_ground_collision(obj, *radius, dt);
            }
        }
        Shape3D::Cylinder(radius, height) => {
            // For cylinder, check the bottom rim points
            let half_height = height / 2.0;
            let bottom_y = obj.object.position.y - half_height;

            if bottom_y < obj.physics_constants.ground_level {
                handle_cylinder_ground_collision(obj, *radius, *height, dt);
            }
        }
        Shape3D::Polyhedron(_vertices, _) => {
            // For polyhedron, transform all vertices and find lowest point
            let world_vertices: Vec<(f64, f64, f64)> = world_vertices(obj);
            let min_y = world_vertices
                .iter()
                .map(|(_, y, _)| *y)
                .fold(f64::MAX, f64::min);

            if min_y < obj.physics_constants.ground_level {
                handle_polyhedron_ground_collision(obj, &world_vertices, dt);
            }
        }
    }
}

/// Gets the vertices of a shape in world space
///
/// Transforms the local vertices of the shape to world space using
/// the object's position and orientation.
///
/// # Arguments
/// * `obj` - Reference to the physical object
///
/// # Returns
/// A vector of (x, y, z) coordinates representing the object's vertices in world space
pub fn world_vertices(obj: &PhysicalObject3D) -> Vec<(f64, f64, f64)> {
    let local_vertices = obj.shape.create_vertices();
    let position = obj.object.position.to_coord();

    // Transform local vertices to world space
    local_vertices
        .iter()
        .map(|v| {
            obj.shape
                .transform_point(*v, position, obj.orientation.to_tuple())
        })
        .collect()
}

/// Gets the faces of a shape in world space
///
/// Transforms the shape's faces (collections of vertices) to world space
/// using the object's position and orientation.
///
/// # Arguments
/// * `obj` - Reference to the physical object
///
/// # Returns
/// A vector of faces, where each face is a vector of (x, y, z) coordinates
pub fn world_faces(obj: &PhysicalObject3D) -> Vec<Vec<(f64, f64, f64)>> {
    let local_vertices = obj.shape.create_vertices();
    let faces = obj.shape.create_faces();
    let position = obj.object.position.to_coord();

    // Transform faces to world space
    faces
        .iter()
        .map(|face| {
            face.iter()
                .map(|&idx| {
                    let vertex = local_vertices[idx];
                    obj.shape
                        .transform_point(vertex, position, obj.orientation.to_tuple())
                })
                .collect()
        })
        .collect()
}

/// Determines which face of a die is currently facing up
///
/// This is useful for dice simulation to determine the rolled value.
///
/// # Arguments
/// * `obj` - Reference to the physical object (must be a beveled cuboid/die shape)
///
/// # Returns
/// * `Some(face)` - The face number (1-6) that is pointing up
/// * `None` - If the object is not a die or the orientation is ambiguous
pub fn die_face_up(obj: &PhysicalObject3D) -> Option<u8> {
    // Only applicable to beveled cuboids (dice)
    if let Shape3D::BeveledCuboid(_, _, _, _) = obj.shape {
        // The up direction in world space
        let up: (f64, f64, f64) = (0.0, 1.0, 0.0);

        // Transform the up vector to object space (inverse of orientation)
        let inverted_orientation: (f64, f64, f64) = (
            -obj.orientation.roll,
            -obj.orientation.pitch,
            -obj.orientation.yaw,
        );

        let obj_up: (f64, f64, f64) = rotate_point(up, inverted_orientation);

        // Find which face normal is most aligned with the up direction
        face_from_normal(&obj.shape, obj_up)
    } else {
        None
    }
}

/// Gets the face number that corresponds to a normal direction
///
/// Used to determine which face of a die corresponds to a direction vector.
///
/// # Arguments
/// * `shape` - Reference to the shape (must be a beveled cuboid/die)
/// * `normal` - The normal direction vector to check
///
/// # Returns
/// * `Some(face)` - The face number (1-6) corresponding to the normal direction
/// * `None` - If the shape is not a die
pub fn face_from_normal(shape: &Shape3D, normal: (f64, f64, f64)) -> Option<u8> {
    match shape {
        Shape3D::BeveledCuboid(_, _, _, _) => {
            // For a die, map the normal direction to a face number
            // Standard dice have opposite faces sum to 7
            let (nx, ny, nz) = normal;

            // Find which component has the largest absolute value
            let abs_nx = nx.abs();
            let abs_ny = ny.abs();
            let abs_nz = nz.abs();

            if abs_nx >= abs_ny && abs_nx >= abs_nz {
                // X-axis dominant
                if nx > 0.0 {
                    Some(2) // Right face
                } else {
                    Some(5) // Left face
                }
            } else if abs_ny >= abs_nx && abs_ny >= abs_nz {
                // Y-axis dominant
                if ny > 0.0 {
                    Some(1) // Top face
                } else {
                    Some(6) // Bottom face
                }
            } else {
                // Z-axis dominant
                if nz > 0.0 {
                    Some(4) // Back face
                } else {
                    Some(3) // Front face
                }
            }
        }
        _ => None, // Not applicable for non-die shapes
    }
}

/// Applies damping to an object's linear and angular velocities
///
/// Simulates friction and air resistance to gradually slow down
/// the object's movement.
///
/// # Arguments
/// * `obj` - Mutable reference to the physical object
/// * `linear_damping` - The linear damping coefficient (0.0 = no damping)
/// * `angular_damping` - The angular damping coefficient
/// * `dt` - The time step duration in seconds
pub fn apply_damping(
    obj: &mut PhysicalObject3D,
    linear_damping: f64,
    angular_damping: f64,
    dt: f64,
) {
    // Apply linear damping
    let linear_factor = (1.0 - linear_damping * dt).max(0.0);
    obj.object.velocity.x *= linear_factor;
    obj.object.velocity.y *= linear_factor;
    obj.object.velocity.z *= linear_factor;

    // Apply stronger angular damping
    let angular_factor = (1.0 - angular_damping * dt).max(0.0);
    obj.angular_velocity.0 *= angular_factor;
    obj.angular_velocity.1 *= angular_factor;
    obj.angular_velocity.2 *= angular_factor;

    // Add threshold damping to stop very slow rotations
    let min_angular_velocity = 0.05;
    if obj.angular_velocity.0.abs() < min_angular_velocity {
        obj.angular_velocity.0 = 0.0;
    }
    if obj.angular_velocity.1.abs() < min_angular_velocity {
        obj.angular_velocity.1 = 0.0;
    }
    if obj.angular_velocity.2.abs() < min_angular_velocity {
        obj.angular_velocity.2 = 0.0;
    }

    // Apply additional damping when close to ground to simulate rolling resistance
    let ground_proximity = obj.object.position.y - obj.physics_constants.ground_level;
    if ground_proximity < 0.1 {
        // Increase damping when close to ground
        let ground_damping = 0.6 * dt;
        obj.angular_velocity.0 *= 1.0 - ground_damping;
        obj.angular_velocity.2 *= 1.0 - ground_damping;
    }
}

/// Applies gravity to an object
///
/// Updates the object's vertical velocity component based on
/// gravitational acceleration.
///
/// # Arguments
/// * `obj` - Mutable reference to the physical object
/// * `gravity` - The gravitational acceleration in m/s²
/// * `dt` - The time step duration in seconds
pub fn apply_gravity(obj: &mut PhysicalObject3D, gravity: f64, dt: f64) {
    obj.object.velocity.y -= gravity * dt;
}

/// Checks if an object is at rest (has stopped moving)
///
/// Determines if an object's linear and angular velocities are below
/// specified thresholds.
///
/// # Arguments
/// * `obj` - Reference to the physical object
/// * `linear_threshold` - Maximum linear speed to be considered at rest
/// * `angular_threshold` - Maximum angular speed to be considered at rest
///
/// # Returns
/// `true` if the object is at rest, `false` otherwise
pub fn is_at_rest(obj: &PhysicalObject3D, linear_threshold: f64, angular_threshold: f64) -> bool {
    // Calculate linear and angular speed
    let linear_speed = (obj.object.velocity.x.powi(2)
        + obj.object.velocity.y.powi(2)
        + obj.object.velocity.z.powi(2))
    .sqrt();

    let angular_speed = (obj.angular_velocity.0.powi(2)
        + obj.angular_velocity.1.powi(2)
        + obj.angular_velocity.2.powi(2))
    .sqrt();

    // Check if both are below threshold
    linear_speed < linear_threshold && angular_speed < angular_threshold
}

/// Updates the physics for multiple objects, including inter-object collisions
///
/// This is the main function for simulating a physical system with multiple objects.
///
/// # Arguments
/// * `objects` - Mutable slice of physical objects to update
/// * `dt` - The time step duration in seconds
///
/// # Examples
/// ```
/// use rs_physics::interactions::shape_collisions_3d;
/// use rs_physics::models::{PhysicalObject3D, Shape3D};
/// use rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS;
///
/// // Create two spheres
/// let mut sphere1 = PhysicalObject3D::new(
///     1.0, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
///     Shape3D::new_sphere(0.5), None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
///     DEFAULT_PHYSICS_CONSTANTS
/// );
///
/// let mut sphere2 = PhysicalObject3D::new(
///     1.0, (-0.5, 0.0, 0.0), (2.0, 1.0, 0.0),
///     Shape3D::new_sphere(0.5), None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
///     DEFAULT_PHYSICS_CONSTANTS
/// );
///
/// let mut objects = vec![sphere1, sphere2];
///
/// // Update the physics system
/// shape_collisions_3d::update_physics_system(&mut objects, 0.1);
///
/// // Objects should have moved and potentially collided
/// ```
pub fn update_physics_system(objects: &mut [PhysicalObject3D], dt: f64) {
    // First update all objects individually
    for obj in objects.iter_mut() {
        update_physics(obj, dt);
    }

    // Then check for collisions between objects
    for i in 0..objects.len() {
        for j in (i + 1)..objects.len() {
            // Need to use split_at_mut to get mutable references to two elements
            let (first, second) = objects.split_at_mut(j);
            let obj1 = &mut first[i];
            let obj2 = &mut second[0];

            // Check and handle collision
            handle_collision(obj1, obj2, dt);
        }
    }
}

/// Performs collision detection between two physical objects, respecting orientation.
/// It preserves your bounding-sphere check and specialized sphere/cylinder logic.
/// For oriented cuboid-cuboid, it calls SAT logic instead of AABB.
pub fn check_collision_objects(obj1: &PhysicalObject3D, obj2: &PhysicalObject3D) -> bool {
    // Quick bounding-sphere reject
    let pos1 = obj1.object.position.to_coord();
    let pos2 = obj2.object.position.to_coord();

    let dx = pos2.0 - pos1.0;
    let dy = pos2.1 - pos1.1;
    let dz = pos2.2 - pos1.2;
    let distance_sq = dx * dx + dy * dy + dz * dz;

    let r1 = obj1.shape.bounding_radius();
    let r2 = obj2.shape.bounding_radius();
    if distance_sq > (r1 + r2).powi(2) {
        return false;
    }

    // Next do shape-specific logic (sphere-sphere, sphere-cuboid, cylinder-cyl, etc.).
    // For the "cuboid-cuboid" (or beveled) path, we do the new SAT approach.

    match (&obj1.shape, &obj2.shape) {
        // ---- Spheres ----
        (Shape3D::Sphere(r1), Shape3D::Sphere(r2)) => distance_sq <= (r1 + r2).powi(2),

        // ---- Sphere-cuboid combos ----
        (Shape3D::Sphere(radius), Shape3D::Cuboid(w, h, d))
        | (Shape3D::Sphere(radius), Shape3D::BeveledCuboid(w, h, d, _)) => {
            let dx_clamp = dx.clamp(-w / 2.0, w / 2.0);
            let dy_clamp = dy.clamp(-h / 2.0, h / 2.0);
            let dz_clamp = dz.clamp(-d / 2.0, d / 2.0);
            let closest_dist_sq =
                (dx - dx_clamp).powi(2) + (dy - dy_clamp).powi(2) + (dz - dz_clamp).powi(2);
            closest_dist_sq <= radius.powi(2)
        }

        // ---- Cuboid-sphere (the reverse) ----
        (Shape3D::Cuboid(w, h, d), Shape3D::Sphere(radius))
        | (Shape3D::BeveledCuboid(w, h, d, _), Shape3D::Sphere(radius)) => {
            let dx_clamp = dx.clamp(-w / 2.0, w / 2.0);
            let dy_clamp = dy.clamp(-h / 2.0, h / 2.0);
            let dz_clamp = dz.clamp(-d / 2.0, d / 2.0);
            let closest_dist_sq =
                (dx - dx_clamp).powi(2) + (dy - dy_clamp).powi(2) + (dz - dz_clamp).powi(2);
            closest_dist_sq <= radius.powi(2)
        }

        // ---- Cylinder-cylinders ----
        (Shape3D::Cylinder(r1, h1), Shape3D::Cylinder(r2, h2)) => {
            let half1 = h1 / 2.0;
            let half2 = h2 / 2.0;
            if dy.abs() > (half1 + half2) {
                return false;
            }
            let xz_sq = dx * dx + dz * dz;
            xz_sq <= (r1 + r2).powi(2)
        }

        // ---- Oriented SAT for cuboid-cuboid or beveled combos ----
        (
            Shape3D::Cuboid(..) | Shape3D::BeveledCuboid(..),
            Shape3D::Cuboid(..) | Shape3D::BeveledCuboid(..),
        ) => sat_cuboid_collision(obj1, obj2),

        // ---- Fallback for other combos or unhandled shapes ----
        _ => {
            // default to `true` if no specialized logic was done, or do a bounding-sphere fallback
            true
        }
    }
}

/// Returns true if two cuboid-like objects (cuboid or beveled) overlap using SAT.
pub fn sat_cuboid_collision(obj1: &PhysicalObject3D, obj2: &PhysicalObject3D) -> bool {
    // Collect local vertices
    let local1 = obj1.shape.create_vertices();
    let local2 = obj2.shape.create_vertices();

    // Convert them to world coordinates (orientation + position)
    let world1: Vec<(f64, f64, f64)> = local1
        .iter()
        .map(|&p| {
            obj1.shape.transform_point(
                p,
                obj1.object.position.to_coord(),
                obj1.orientation.to_tuple(),
            )
        })
        .collect();

    let world2: Vec<(f64, f64, f64)> = local2
        .iter()
        .map(|&p| {
            obj2.shape.transform_point(
                p,
                obj2.object.position.to_coord(),
                obj2.orientation.to_tuple(),
            )
        })
        .collect();

    // Compute the primary axes (face normals) from each shape
    // We assume create_vertices() returns at least 8 corners for a cuboid or beveled cuboid.
    let (ax1, ax2, ax3) = compute_box_axes(&world1);
    let (ax4, ax5, ax6) = compute_box_axes(&world2);

    // Build candidate axes: each shape's 3 face normals + cross products among them
    let mut axes = vec![ax1, ax2, ax3, ax4, ax5, ax6];
    let shape1_axes = [ax1, ax2, ax3];
    let shape2_axes = [ax4, ax5, ax6];

    for &a in &shape1_axes {
        for &b in &shape2_axes {
            let cross = cross_product(a, b);
            // skip axis if it's nearly zero length
            if dot_product(cross, cross) > 1e-12 {
                axes.push(normalize(cross));
            }
        }
    }

    // Check for overlap on each axis
    for axis in axes {
        if !check_overlap_along_axis(&world1, &world2, &axis) {
            // Found a separating axis => no collision
            return false;
        }
    }

    // No separating axis => colliding
    true
}

/// Extracts three orthonormal axes from the corner set of a box.
/// We pick corners[0], corners[1], corners[3], corners[4] to get edges
/// that emanate from the same corner. If corners are generated consistently,
/// that yields X, Y, and Z directions (in world space).
fn compute_box_axes(
    corners: &[(f64, f64, f64)],
) -> ((f64, f64, f64), (f64, f64, f64), (f64, f64, f64)) {
    // If something’s off (like not enough corners), return default axes
    if corners.len() < 5 {
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0));
    }

    let edge_x = (
        corners[1].0 - corners[0].0,
        corners[1].1 - corners[0].1,
        corners[1].2 - corners[0].2,
    );
    let edge_y = (
        corners[3].0 - corners[0].0,
        corners[3].1 - corners[0].1,
        corners[3].2 - corners[0].2,
    );
    let edge_z = (
        corners[4].0 - corners[0].0,
        corners[4].1 - corners[0].1,
        corners[4].2 - corners[0].2,
    );

    (normalize(edge_x), normalize(edge_y), normalize(edge_z))
}

/// Normalizes a 3D vector safely, returning (0,0,0) if it’s almost zero-length.
fn normalize(v: (f64, f64, f64)) -> (f64, f64, f64) {
    let len_sq = dot_product(v, v);
    if len_sq < 1e-12 {
        (0.0, 0.0, 0.0)
    } else {
        let inv_len = 1.0 / len_sq.sqrt();
        (v.0 * inv_len, v.1 * inv_len, v.2 * inv_len)
    }
}
