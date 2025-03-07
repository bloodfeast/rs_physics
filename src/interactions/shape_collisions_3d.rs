use crate::interactions::{cross_product, dot_product};
use crate::models::{normalize_angle, rotate_point, PhysicalObject3D, Shape3D, ToCoordinates, Quaternion};
use rand::{rng, Rng};
use crate::interactions::gjk_collision_3d::{epa_contact_points, gjk_collision_detection};

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

    // Apply to first object - this should make the velocity more negative
    obj1.object.velocity.x -= impulse.0 / obj1.object.mass;
    obj1.object.velocity.y -= impulse.1 / obj1.object.mass;
    obj1.object.velocity.z -= impulse.2 / obj1.object.mass;

    // Apply to second object (opposite direction)
    obj2.object.velocity.x += impulse.0 / obj2.object.mass;
    obj2.object.velocity.y += impulse.1 / obj2.object.mass;
    obj2.object.velocity.z += impulse.2 / obj2.object.mass;
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

    // Quick bounding-sphere rejection test
    let dx = pos2.0 - pos1.0;
    let dy = pos2.1 - pos1.1;
    let dz = pos2.2 - pos1.2;
    let distance_sq = dx * dx + dy * dy + dz * dz;

    let r1 = obj1.shape.bounding_radius();
    let r2 = obj2.shape.bounding_radius();
    if distance_sq > (r1 + r2).powi(2) {
        return false;
    }

    // Fast path for sphere-sphere collisions
    if let (Shape3D::Sphere(radius1), Shape3D::Sphere(radius2)) = (&obj1.shape, &obj2.shape) {
        if distance_sq > (radius1 + radius2).powi(2) {
            return false;
        }

        // Calculate normal
        let distance = distance_sq.sqrt();
        let normal = (dx / distance, dy / distance, dz / distance);

        // Create contact points
        let r1 = (normal.0 * radius1, normal.1 * radius1, normal.2 * radius1);
        let r2 = (-normal.0 * radius2, -normal.1 * radius2, -normal.2 * radius2);

        // Calculate penetration depth
        let penetration = radius1 + radius2 - distance;

        // Handle collision response directly for spheres
        if penetration > 0.0 {
            // Calculate point velocities
            let v1 = calculate_point_velocity(obj1, r1);
            let v2 = calculate_point_velocity(obj2, r2);

            // Relative velocity
            let vrel = (v2.0 - v1.0, v2.1 - v1.1, v2.2 - v1.2);

            // Normal component of relative velocity
            let vrel_n = dot_product(vrel, normal);

            if vrel_n < 0.0 {
                // Apply collision response
                let restitution = (obj1.get_restitution() + obj2.get_restitution()) / 2.0;
                let impulse_mag = calculate_collision_impulse(
                    obj1, obj2, vrel, normal, restitution, r1, r2
                );

                apply_linear_impulse(obj1, obj2, normal, impulse_mag);
                apply_angular_impulse(obj1, obj2, normal, impulse_mag, r1, r2, dt);

                // Resolve penetration
                resolve_sphere_penetration(obj1, obj2, normal, penetration);

                return true;
            }
        }

        return false;
    }

    // For other shapes, use GJK-EPA
    let orientation1 = Quaternion::from_euler(
        obj1.orientation.roll,
        obj1.orientation.pitch,
        obj1.orientation.yaw
    );

    let orientation2 = Quaternion::from_euler(
        obj2.orientation.roll,
        obj2.orientation.pitch,
        obj2.orientation.yaw
    );

    // Use GJK for collision detection
    if let Some(simplex) = gjk_collision_detection(
        &obj1.shape, pos1, orientation1,
        &obj2.shape, pos2, orientation2
    ) {
        // Use EPA to get contact information
        if let Some(contact) = epa_contact_points(
            &obj1.shape, pos1, orientation1,
            &obj2.shape, pos2, orientation2,
            &simplex
        ) {
            // Calculate point velocities
            let r1 = (
                contact.point1.0 - pos1.0,
                contact.point1.1 - pos1.1,
                contact.point1.2 - pos1.2
            );

            let r2 = (
                contact.point2.0 - pos2.0,
                contact.point2.1 - pos2.1,
                contact.point2.2 - pos2.2
            );

            let v1 = calculate_point_velocity(obj1, r1);
            let v2 = calculate_point_velocity(obj2, r2);

            let vrel = (v1.0 - v2.0, v1.1 - v2.1, v1.2 - v2.2);
            let vrel_n = dot_product(vrel, contact.normal);

            if vrel_n < 0.0 {
                let restitution = (obj1.get_restitution() + obj2.get_restitution()) / 2.0;
                let impulse_mag = calculate_collision_impulse(
                    obj1, obj2, vrel, contact.normal, restitution, r1, r2
                );

                apply_linear_impulse(obj1, obj2, contact.normal, impulse_mag);
                apply_angular_impulse(obj1, obj2, contact.normal, impulse_mag, r1, r2, dt);

                resolve_penetration_epa(obj1, obj2, contact.normal, contact.penetration);

                return true;
            }
        }
    }

    false
}

/// Resolves penetration using EPA results
pub fn resolve_penetration_epa(
    obj1: &mut PhysicalObject3D,
    obj2: &mut PhysicalObject3D,
    normal: (f64, f64, f64),
    penetration: f64
) {
    let correction_percentage = 0.8;
    let correction = penetration * correction_percentage;
    let correction_vector = (
        normal.0 * correction,
        normal.1 * correction,
        normal.2 * correction
    );

    // Apply correction based on inverse mass ratio
    let total_mass = obj1.object.mass + obj2.object.mass;
    let self_ratio = obj2.object.mass / total_mass;
    let other_ratio = obj1.object.mass / total_mass;

    // Move objects apart
    obj1.object.position.x += correction_vector.0 * self_ratio;
    obj1.object.position.y += correction_vector.1 * self_ratio;
    obj1.object.position.z += correction_vector.2 * self_ratio;

    obj2.object.position.x -= correction_vector.0 * other_ratio;
    obj2.object.position.y -= correction_vector.1 * other_ratio;
    obj2.object.position.z -= correction_vector.2 * other_ratio;
}

/// Resolves penetration for sphere-sphere collisions
pub fn resolve_sphere_penetration(
    obj1: &mut PhysicalObject3D,
    obj2: &mut PhysicalObject3D,
    normal: (f64, f64, f64),
    penetration: f64
) {
    let correction_percentage = 1.0;
    let correction = penetration * correction_percentage;
    let correction_vector = (
        normal.0 * correction,
        normal.1 * correction,
        normal.2 * correction
    );

    // Apply correction based on inverse mass ratio
    let total_mass = obj1.object.mass + obj2.object.mass;
    let obj1_ratio = obj1.object.mass / total_mass;
    let obj2_ratio = obj2.object.mass / total_mass;

    // Move objects apart - obj1 moves opposite to the normal direction
    obj1.object.position.x -= correction_vector.0 * obj2_ratio;
    obj1.object.position.y -= correction_vector.1 * obj2_ratio;
    obj1.object.position.z -= correction_vector.2 * obj2_ratio;

    // obj2 moves in the normal direction
    obj2.object.position.x += correction_vector.0 * obj1_ratio;
    obj2.object.position.y += correction_vector.1 * obj1_ratio;
    obj2.object.position.z += correction_vector.2 * obj1_ratio;
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