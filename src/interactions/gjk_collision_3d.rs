use crate::interactions::{cross_product, dot_product, vector_magnitude};
use crate::models::{Quaternion, Shape3D, Simplex, SupportPoint};

/// A face of the polytopes for EPA algorithm
pub struct Face {
    indices: [usize; 3],
    normal: (f64, f64, f64),
    distance: f64,
}

/// Contact information from collision detection
#[derive(Debug, Clone)]
pub struct ContactInfo {
    pub point1: (f64, f64, f64), // Contact point on shape1
    pub point2: (f64, f64, f64), // Contact point on shape2
    pub normal: (f64, f64, f64), // Contact normal (pointing from shape2 to shape1)
    pub penetration: f64,        // Penetration depth
}

/// GJK (Gilbert-Johnson-Keerthi) algorithm for collision detection between convex shapes
///
/// This implementation detects collisions between two convex 3D shapes and can be used
/// for robust physics simulations. It properly handles rotated objects.
///
/// # Arguments
/// * `shape1` - First shape to check for collision
/// * `position1` - Position of the first shape
/// * `orientation1` - Orientation of the first shape as a quaternion
/// * `shape2` - Second shape to check for collision
/// * `position2` - Position of the second shape
/// * `orientation2` - Orientation of the second shape as a quaternion
///
/// # Returns
/// `Some(Simplex)` if the shapes are colliding, containing the final simplex
/// `None` if the shapes are not colliding
pub fn gjk_collision_detection(
    shape1: &Shape3D,
    position1: (f64, f64, f64),
    orientation1: Quaternion,
    shape2: &Shape3D,
    position2: (f64, f64, f64),
    orientation2: Quaternion
) -> Option<Simplex> {
    // Get the initial search direction (from center of shape1 to center of shape2)
    let mut direction = (
        position2.0 - position1.0,
        position2.1 - position1.1,
        position2.2 - position1.2
    );

    // If the centers are at the same position, use an arbitrary direction
    if vector_magnitude(direction) < 1e-10 {
        direction = (1.0, 0.0, 0.0);
    }

    // Initialize the simplex
    let mut simplex = Simplex::new();

    // Get the first support point
    let first_support = get_support_point(
        shape1, position1, orientation1,
        shape2, position2, orientation2,
        direction
    );
    simplex.add(first_support);

    // Negate the direction for the next iteration
    direction = negate_vector(direction);

    // Main GJK loop
    let max_iterations = 32; // Prevent infinite loops
    let mut iterations = 0;

    // For spheres, we can do a quick distance check before entering the loop
    if let (Shape3D::Sphere(radius1), Shape3D::Sphere(radius2)) = (shape1, shape2) {
        let dx = position2.0 - position1.0;
        let dy = position2.1 - position1.1;
        let dz = position2.2 - position1.2;
        let distance_sq = dx * dx + dy * dy + dz * dz;

        // If distance is greater than sum of radii, definitely no collision
        if distance_sq > (radius1 + radius2).powi(2) {
            return None;
        }
    }

    while iterations < max_iterations {
        iterations += 1;

        // Get a new support point in the current direction
        let support = get_support_point(
            shape1, position1, orientation1,
            shape2, position2, orientation2,
            direction
        );

        // If the support point doesn't pass the origin, shapes are not colliding
        let support_dot_dir = dot_product(support.point, direction);

        // Use a small epsilon to handle floating-point precision issues
        let epsilon = 1e-10;
        if support_dot_dir < epsilon {
            return None;
        }

        // Add the new support point to the simplex
        simplex.add(support);

        // Check if the simplex contains the origin and update the direction
        if do_simplex(&mut simplex, &mut direction) {
            return Some(simplex); // Collision detected, return the simplex
        }
    }

    // If we've reached maximum iterations, consider it a non-collision for stability
    None
}

/// Combined simplex processing function that handles all simplex cases
/// Returns true if the simplex contains the origin
pub fn do_simplex(simplex: &mut Simplex, direction: &mut (f64, f64, f64)) -> bool {
    match simplex.size() {
        2 => handle_line_case(simplex, direction),
        3 => handle_triangle_case(simplex, direction),
        4 => handle_tetrahedron_case(simplex, direction),
        _ => false,
    }
}


/// Gets the support point for GJK algorithm
pub fn get_support_point(
    shape1: &Shape3D,
    position1: (f64, f64, f64),
    orientation1: Quaternion,
    shape2: &Shape3D,
    position2: (f64, f64, f64),
    orientation2: Quaternion,
    direction: (f64, f64, f64)
) -> SupportPoint {
    // Get the farthest point in the given direction for shape1
    let point_a = get_support_point_for_shape(shape1, position1, orientation1, direction);

    // Get the farthest point in the opposite direction for shape2
    let neg_direction = negate_vector(direction);
    let point_b = get_support_point_for_shape(shape2, position2, orientation2, neg_direction);

    // The final support point is the difference between the two points (Minkowski Difference)
    let point = (
        point_a.0 - point_b.0,
        point_a.1 - point_b.1,
        point_a.2 - point_b.2
    );

    SupportPoint { point, point_a, point_b }
}

/// Gets the support point for a specific shape
pub fn get_support_point_for_shape(
    shape: &Shape3D,
    position: (f64, f64, f64),
    orientation: Quaternion,
    direction: (f64, f64, f64)
) -> (f64, f64, f64) {
    // Implementation for various shape types
    match shape {
        Shape3D::Sphere(radius) => {
            // For a sphere, the support point is just radius * normalized direction
            let dir_magnitude = vector_magnitude(direction);
            if dir_magnitude < 1e-10 {
                return position;
            }

            let normalized_dir = (
                direction.0 / dir_magnitude,
                direction.1 / dir_magnitude,
                direction.2 / dir_magnitude
            );

            (
                position.0 + normalized_dir.0 * radius,
                position.1 + normalized_dir.1 * radius,
                position.2 + normalized_dir.2 * radius
            )
        },
        Shape3D::Cuboid(width, height, depth) => {
            // For a cuboid, we need to transform the direction to local space
            let half_width = width / 2.0;
            let half_height = height / 2.0;
            let half_depth = depth / 2.0;

            // Convert direction to shape's local space
            let local_dir = orientation.inverse().rotate_point(direction);

            // Get the farthest point in that direction
            let local_support = (
                if local_dir.0 >= 0.0 { half_width } else { -half_width },
                if local_dir.1 >= 0.0 { half_height } else { -half_height },
                if local_dir.2 >= 0.0 { half_depth } else { -half_depth }
            );

            // Transform back to world space
            let rotated_support = orientation.rotate_point(local_support);

            (
                position.0 + rotated_support.0,
                position.1 + rotated_support.1,
                position.2 + rotated_support.2
            )
        },
        Shape3D::BeveledCuboid(width, height, depth, bevel) => {
            // Similar to cuboid but with beveled edges
            let half_width = width / 2.0;
            let half_height = height / 2.0;
            let half_depth = depth / 2.0;

            // Convert direction to shape's local space
            let local_dir = orientation.inverse().rotate_point(direction);

            // Normalize the local direction
            let dir_magnitude = vector_magnitude(local_dir);
            if dir_magnitude < 1e-10 {
                let rotated_origin = orientation.rotate_point((0.0, 0.0, 0.0));
                return (
                    position.0 + rotated_origin.0,
                    position.1 + rotated_origin.1,
                    position.2 + rotated_origin.2
                );
            }

            let norm_local_dir = (
                local_dir.0 / dir_magnitude,
                local_dir.1 / dir_magnitude,
                local_dir.2 / dir_magnitude
            );

            // Get the farthest point with beveled corners
            let local_support = if norm_local_dir.0.abs() > 0.999 ||
                norm_local_dir.1.abs() > 0.999 ||
                norm_local_dir.2.abs() > 0.999 {
                // Direction is along one of the main axes
                (
                    if norm_local_dir.0 >= 0.0 { half_width } else { -half_width },
                    if norm_local_dir.1 >= 0.0 { half_height } else { -half_height },
                    if norm_local_dir.2 >= 0.0 { half_depth } else { -half_depth }
                )
            } else {
                // Direction is not along the main axes, consider beveling
                let x = if norm_local_dir.0 >= 0.0 { half_width - bevel } else { -half_width + bevel };
                let y = if norm_local_dir.1 >= 0.0 { half_height - bevel } else { -half_height + bevel };
                let z = if norm_local_dir.2 >= 0.0 { half_depth - bevel } else { -half_depth + bevel };

                // Add the beveled part
                let bevel_dir = (
                    norm_local_dir.0 * bevel,
                    norm_local_dir.1 * bevel,
                    norm_local_dir.2 * bevel
                );

                (x + bevel_dir.0, y + bevel_dir.1, z + bevel_dir.2)
            };

            // Transform back to world space
            let rotated_support = orientation.rotate_point(local_support);

            (
                position.0 + rotated_support.0,
                position.1 + rotated_support.1,
                position.2 + rotated_support.2
            )
        },
        Shape3D::Cylinder(radius, height) => {
            // For a cylinder, we need special handling based on the direction
            let half_height = height / 2.0;

            // Convert direction to shape's local space
            let local_dir = orientation.inverse().rotate_point(direction);

            // Decompose into axial and radial components
            let axial_component = local_dir.1;
            let radial_dir = (local_dir.0, 0.0, local_dir.2);
            let radial_magnitude = vector_magnitude(radial_dir);

            // Calculate support point in local space
            let local_support = if radial_magnitude < 1e-10 {
                // Direction is along the cylinder axis
                (
                    0.0,
                    if axial_component >= 0.0 { half_height } else { -half_height },
                    0.0
                )
            } else {
                // Direction has a radial component
                let normalized_radial = (
                    radial_dir.0 / radial_magnitude,
                    0.0,
                    radial_dir.2 / radial_magnitude
                );

                (
                    normalized_radial.0 * radius,
                    if axial_component >= 0.0 { half_height } else { -half_height },
                    normalized_radial.2 * radius
                )
            };

            // Transform back to world space
            let rotated_support = orientation.rotate_point(local_support);

            (
                position.0 + rotated_support.0,
                position.1 + rotated_support.1,
                position.2 + rotated_support.2
            )
        },
        Shape3D::Polyhedron(vertices, _) => {
            // For a polyhedron, find the vertex with maximum dot product with direction
            let local_dir = orientation.inverse().rotate_point(direction);

            let mut max_dot = -f64::INFINITY;
            let mut max_vertex = (0.0, 0.0, 0.0);

            for vertex in vertices {
                let dot = dot_product(*vertex, local_dir);
                if dot > max_dot {
                    max_dot = dot;
                    max_vertex = *vertex;
                }
            }

            // Transform back to world space
            let rotated_vertex = orientation.rotate_point(max_vertex);

            (
                position.0 + rotated_vertex.0,
                position.1 + rotated_vertex.1,
                position.2 + rotated_vertex.2
            )
        }
    }
}

/// returns the inverse of the vector
pub fn negate_vector(v: (f64, f64, f64)) -> (f64, f64, f64) {
    (-v.0, -v.1, -v.2)
}

/// Handles the line case for GJK algorithm
/// Returns true if the origin is on the line segment, false otherwise
pub fn handle_line_case(simplex: &mut Simplex, direction: &mut (f64, f64, f64)) -> bool {
    let a = simplex.get_a().point;
    let b = simplex.get_b().point;

    // Vector from b to a
    let ab = (a.0 - b.0, a.1 - b.1, a.2 - b.2);
    // Vector from b to origin
    let b0 = (0.0 - b.0, 0.0 - b.1, 0.0 - b.2);

    // Ensure direction is always updated (not just leaving y component as 0)
    if ab.1 == 0.0 && direction.0 == 0.0 && direction.2 == 0.0 {
        // If we're in a special case where we might not update y, pick a different direction
        *direction = (0.0, 1.0, 0.0);
        return false;
    }

    // Check if origin is beyond a
    let v = (a.0, a.1, a.2);
    if dot_product(v, ab) >= 0.0 {
        // Origin is beyond a, perpendicular to AB
        *direction = triple_product(ab, b0, ab);

        // If the new direction is too small, we need a fallback
        if vector_magnitude(*direction) < 1e-10 {
            // Use a perpendicular to AB that isn't tiny
            if ab.0.abs() > ab.1.abs() && ab.0.abs() > ab.2.abs() {
                *direction = cross_product(ab, (0.0, 1.0, 0.0));
            } else {
                *direction = cross_product(ab, (1.0, 0.0, 0.0));
            }
        }
    } else {
        // Origin is behind a, remove a and go back toward origin
        simplex.set_ab(simplex.get_b().clone(), simplex.get_a().clone());
        *direction = b0;
    }

    false // Line segment cannot contain the origin
}

/// Handles the triangle case for GJK algorithm
/// Returns true if the origin is in the triangle, false otherwise
pub fn handle_triangle_case(simplex: &mut Simplex, direction: &mut (f64, f64, f64)) -> bool {
    let a = simplex.get_a().point;
    let b = simplex.get_b().point;
    let c = simplex.get_c().point;

    // Vector from c to a and c to b
    let ac = (a.0 - c.0, a.1 - c.1, a.2 - c.2);
    let bc = (b.0 - c.0, b.1 - c.1, b.2 - c.2);

    // Vector from c to origin
    let c0 = (0.0 - c.0, 0.0 - c.1, 0.0 - c.2);

    // Cross product of AC and BC - normal to triangle face
    let abc = cross_product(ac, bc);

    // Check if origin is above or below the triangle
    let abc_dot_c0 = dot_product(abc, c0);

    if abc_dot_c0 > 0.0 {
        // Origin is above the triangle

        // Check edges
        // Edge AC
        let acf = cross_product(ac, abc);
        if dot_product(acf, c0) > 0.0 {
            // Origin is outside AC edge, keep B and C
            simplex.set_abc(simplex.get_b().clone(), simplex.get_c().clone(), simplex.get_a().clone());
            *direction = acf;
            return false;
        }

        // Edge BC
        let bcf = cross_product(abc, bc);
        if dot_product(bcf, c0) > 0.0 {
            // Origin is outside BC edge, keep A and C
            simplex.set_abc(simplex.get_a().clone(), simplex.get_c().clone(), simplex.get_b().clone());
            *direction = bcf;
            return false;
        }

        // Origin is inside both AC and BC, so above the triangle face
        *direction = abc;
    } else {
        // Origin is below the triangle

        // Check edges
        // Edge AC
        let acf = cross_product(abc, ac);
        if dot_product(acf, c0) > 0.0 {
            // Origin is outside AC edge, keep B and C
            simplex.set_abc(simplex.get_b().clone(), simplex.get_c().clone(), simplex.get_a().clone());
            *direction = acf;
            return false;
        }

        // Edge BC
        let bcf = cross_product(bc, abc);
        if dot_product(bcf, c0) > 0.0 {
            // Origin is outside BC edge, keep A and C
            simplex.set_abc(simplex.get_a().clone(), simplex.get_c().clone(), simplex.get_b().clone());
            *direction = bcf;
            return false;
        }

        // Origin is inside both AC and BC, so below the triangle face
        *direction = negate_vector(abc);
    }

    false // The triangle itself cannot contain the origin in 3D
}

/// Handles the tetrahedron case for GJK algorithm
/// Returns true if the origin is inside the tetrahedron, false otherwise
pub fn handle_tetrahedron_case(simplex: &mut Simplex, direction: &mut (f64, f64, f64)) -> bool {
    let a = simplex.get_a().point;
    let b = simplex.get_b().point;
    let c = simplex.get_c().point;
    let d = simplex.get_d().point;


    // Vector from d to origin
    let d0 = (0.0 - d.0, 0.0 - d.1, 0.0 - d.2);

    let normal_abc = cross_product(
        (c.0 - a.0, c.1 - a.1, c.2 - a.2),  // AC
        (b.0 - a.0, b.1 - a.1, b.2 - a.2)   // AB
    );

    // Check if the origin is on the positive side of ABC face
    if dot_product(normal_abc, d0) > 0.0 {
        simplex.set_abc(simplex.get_a().clone(), simplex.get_b().clone(), simplex.get_c().clone());
        *direction = normal_abc;
        return false;
    }

    // If we reach here, the origin is inside the tetrahedron
    return true;
}

// Triple product: A × (B × C)
pub fn triple_product(a: (f64, f64, f64), b: (f64, f64, f64), c: (f64, f64, f64)) -> (f64, f64, f64) {
    let bc = cross_product(b, c);
    cross_product(a, bc)
}

/// EPA (Expanding Polytope Algorithm) for finding contact information
/// Returns contact information if objects are colliding, None otherwise
pub fn epa_contact_points(
    shape1: &Shape3D,
    position1: (f64, f64, f64),
    orientation1: Quaternion,
    shape2: &Shape3D,
    position2: (f64, f64, f64),
    orientation2: Quaternion,
    simplex: &Simplex
) -> Option<ContactInfo> {
    // Special case for sphere-sphere collisions
    if let (Shape3D::Sphere(radius1), Shape3D::Sphere(radius2)) = (shape1, shape2) {
        let direction = (
            position2.0 - position1.0,
            position2.1 - position1.1,
            position2.2 - position1.2
        );

        let distance_sq = direction.0 * direction.0 + direction.1 * direction.1 + direction.2 * direction.2;
        let distance = distance_sq.sqrt();

        if distance < radius1 + radius2 {
            // Normalize the direction
            let normal = if distance > 0.001 {
                (direction.0 / distance, direction.1 / distance, direction.2 / distance)
            } else {
                // Default direction if centers are too close
                (1.0, 0.0, 0.0)
            };

            // Calculate penetration depth
            let penetration = radius1 + radius2 - distance;

            // Calculate contact points on each sphere's surface
            let contact1 = (
                position1.0 + normal.0 * radius1,
                position1.1 + normal.1 * radius1,
                position1.2 + normal.2 * radius1
            );

            let contact2 = (
                position2.0 - normal.0 * radius2,
                position2.1 - normal.1 * radius2,
                position2.2 - normal.2 * radius2
            );

            return Some(ContactInfo {
                point1: contact1,
                point2: contact2,
                normal: (-normal.0, -normal.1, -normal.2), // Normal points from shape2 to shape1
                penetration,
            });
        }
    }

    // Create the initial polytope from the simplex
    let mut polytope = simplex.points.clone();

    // Ensure we have at least 4 points for a 3D polytope
    while polytope.len() < 4 {
        // Choose directions orthogonal to existing points
        let mut direction = (1.0, 0.0, 0.0);

        // If we have 1 point, add in x direction
        if polytope.len() == 1 {
            direction = (1.0, 0.0, 0.0);
        }
        // If we have 2 points, add in y direction
        else if polytope.len() == 2 {
            direction = (0.0, 1.0, 0.0);
        }
        // If we have 3 points, add in z direction
        else if polytope.len() == 3 {
            direction = (0.0, 0.0, 1.0);
        }

        // Get support point
        let support = get_support_point(
            shape1, position1, orientation1,
            shape2, position2, orientation2,
            direction
        );

        polytope.push(support);
    }

    let mut faces = Vec::new();

    // Generate initial faces
    // For a tetrahedron, we have 4 triangular faces
    if polytope.len() == 4 {
        // Create the faces (adjust winding for correct normals)
        let face_indices = [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2]
        ];

        for indices in &face_indices {
            let a = polytope[indices[0]].point;
            let b = polytope[indices[1]].point;
            let c = polytope[indices[2]].point;

            // Calculate face normal and distance
            let ab = (b.0 - a.0, b.1 - a.1, b.2 - a.2);
            let ac = (c.0 - a.0, c.1 - a.1, c.2 - a.2);

            let normal = cross_product(ab, ac);
            let normal_length = vector_magnitude(normal);

            if normal_length < 1e-10 {
                continue; // Skip degenerate faces
            }

            let normalized_normal = (
                normal.0 / normal_length,
                normal.1 / normal_length,
                normal.2 / normal_length
            );

            // Make sure the normal points outward (away from the origin)
            let dot = dot_product(normalized_normal, a);
            let normalized_normal = if dot < 0.0 {
                normalized_normal
            } else {
                negate_vector(normalized_normal)
            };

            // Calculate distance to origin (should be negative if normal points outward)
            let distance = -dot_product(normalized_normal, a);

            faces.push(Face {
                indices: [indices[0], indices[1], indices[2]],
                normal: normalized_normal,
                distance: distance.abs(),
            });
        }
    }

    // Main EPA loop
    const MAX_ITERATIONS: usize = 32;
    const TOLERANCE: f64 = 1e-6;

    for _ in 0..MAX_ITERATIONS {
        // Find the closest face to the origin
        if faces.is_empty() {
            return None;
        }

        let closest_face_idx = faces.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap();

        let closest_face = &faces[closest_face_idx];

        // Get the support point in the direction of the face normal
        let support = get_support_point(
            shape1, position1, orientation1,
            shape2, position2, orientation2,
            closest_face.normal
        );

        // Calculate distance from support point to closest face
        let support_dist = dot_product(support.point, closest_face.normal);

        // Check if we've reached the tolerance
        if (support_dist - closest_face.distance).abs() < TOLERANCE {
            // We've found the closest point, calculate contact information
            return calculate_contact_from_face(closest_face, &polytope, shape1, position1, orientation1, shape2, position2, orientation2);
        }

        // Remove faces that can be seen from the support point
        let mut visible_faces: Vec<usize> = Vec::new();
        let mut visible_edges: Vec<(usize, usize)> = Vec::new();

        for (i, face) in faces.iter().enumerate() {
            // A face is visible if the support point is in front of it
            if dot_product(face.normal, support.point) > 0.0 {
                visible_faces.push(i);

                // Keep track of the edges of visible faces
                let edges = [
                    (face.indices[0], face.indices[1]),
                    (face.indices[1], face.indices[2]),
                    (face.indices[2], face.indices[0])
                ];

                for edge in &edges {
                    let mut found = false;

                    // Check if this edge is already in our list
                    for (j, &existing_edge) in visible_edges.iter().enumerate() {
                        if edge.0 == existing_edge.1 && edge.1 == existing_edge.0 {
                            // This edge is shared with another visible face, remove it
                            visible_edges.swap_remove(j);
                            found = true;
                            break;
                        }
                    }

                    if !found {
                        // This is a new edge of the silhouette
                        visible_edges.push(*edge);
                    }
                }
            }
        }

        // If no faces are visible, something went wrong
        if visible_faces.is_empty() {
            return None;
        }

        // Remove all visible faces
        // Sort in reverse order to avoid invalidating indices
        visible_faces.sort_unstable_by(|a, b| b.cmp(a));
        for &face_index in &visible_faces {
            faces.swap_remove(face_index);
        }

        // Add the new support point to the polytope
        polytope.push(support.clone());
        let new_vertex_index = polytope.len() - 1;

        // Create new faces connecting the supporting point with the boundary edges
        for &(i, j) in &visible_edges {
            let a = polytope[i].point;
            let b = polytope[j].point;
            let c = support.point;

            // Calculate face normal and distance
            let ab = (b.0 - a.0, b.1 - a.1, b.2 - a.2);
            let ac = (c.0 - a.0, c.1 - a.1, c.2 - a.2);

            let normal = cross_product(ab, ac);
            let normal_length = vector_magnitude(normal);

            if normal_length < 1e-10 {
                continue; // Skip degenerate faces
            }

            let normalized_normal = (
                normal.0 / normal_length,
                normal.1 / normal_length,
                normal.2 / normal_length
            );

            // Make sure the normal points outward
            let dot = dot_product(normalized_normal, a);
            let normalized_normal = if dot < 0.0 {
                normalized_normal
            } else {
                negate_vector(normalized_normal)
            };

            // Calculate distance to origin
            let distance = -dot_product(normalized_normal, a);

            faces.push(Face {
                indices: [i, j, new_vertex_index],
                normal: normalized_normal,
                distance: distance.abs(),
            });
        }
    }

    // If we've reached the maximum number of iterations, use the closest face
    if faces.is_empty() {
        return None;
    }

    let closest_face = faces.iter()
        .min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    calculate_contact_from_face(closest_face, &polytope, shape1, position1, orientation1, shape2, position2, orientation2)
}

/// Calculate contact information from the closest face found by EPA
///
/// note: The unused parameters here may be removed later but as I refine the approach
///     I would rather require them now to make refactoring easier later if I end up removing some
///     of them.
pub fn calculate_contact_from_face(
    face: &Face,
    polytope: &[SupportPoint],
    _shape1: &Shape3D,
    _position1: (f64, f64, f64),
    _orientation1: Quaternion,
    _shape2: &Shape3D,
    _position2: (f64, f64, f64),
    _orientation2: Quaternion
) -> Option<ContactInfo> {
    // Get the vertices of the face
    let a = polytope[face.indices[0]].point;
    let b = polytope[face.indices[1]].point;
    let c = polytope[face.indices[2]].point;

    // Barycentric coordinates of the closest point to origin on the face
    let barycentric = barycentric_coordinates_of_closest_point(a, b, c);

    // Points on the original shapes
    let p1_a = polytope[face.indices[0]].point_a;
    let p1_b = polytope[face.indices[1]].point_a;
    let p1_c = polytope[face.indices[2]].point_a;

    let p2_a = polytope[face.indices[0]].point_b;
    let p2_b = polytope[face.indices[1]].point_b;
    let p2_c = polytope[face.indices[2]].point_b;

    // Calculate contact points on both shapes
    let contact_point1: (f64, f64, f64) = (
        barycentric.0 * p1_a.0 + barycentric.1 * p1_b.0 + barycentric.2 * p1_c.0,
        barycentric.0 * p1_a.1 + barycentric.1 * p1_b.1 + barycentric.2 * p1_c.1,
        barycentric.0 * p1_a.2 + barycentric.1 * p1_b.2 + barycentric.2 * p1_c.2
    );

    let contact_point2: (f64, f64, f64) = (
        barycentric.0 * p2_a.0 + barycentric.1 * p2_b.0 + barycentric.2 * p2_c.0,
        barycentric.0 * p2_a.1 + barycentric.1 * p2_b.1 + barycentric.2 * p2_c.1,
        barycentric.0 * p2_a.2 + barycentric.1 * p2_b.2 + barycentric.2 * p2_c.2
    );

    // Normal pointing from shape2 to shape1
    let normal = negate_vector(face.normal);

    // Calculate penetration depth
    let penetration = face.distance;

    Some(ContactInfo {
        point1: contact_point1,
        point2: contact_point2,
        normal,
        penetration,
    })
}

/// Calculate barycentric coordinates of the closest point to origin on a triangle
pub fn barycentric_coordinates_of_closest_point(
    a: (f64, f64, f64),
    b: (f64, f64, f64),
    c: (f64, f64, f64)
) -> (f64, f64, f64) {
    // Calculate the closest point on the face to the origin
    let ab = (b.0 - a.0, b.1 - a.1, b.2 - a.2);
    let ac = (c.0 - a.0, c.1 - a.1, c.2 - a.2);
    let ap = (0.0 - a.0, 0.0 - a.1, 0.0 - a.2); // Vector from a to origin

    // Gram matrix
    let d00 = dot_product(ab, ab);
    let d01 = dot_product(ab, ac);
    let d11 = dot_product(ac, ac);
    let d20 = dot_product(ap, ab);
    let d21 = dot_product(ap, ac);

    let inv_denom = 1.0 / (d00 * d11 - d01 * d01);

    // Calculate barycentric coordinates
    let v = (d11 * d20 - d01 * d21) * inv_denom;
    let w = (d00 * d21 - d01 * d20) * inv_denom;
    let u = 1.0 - v - w;

    // Clamp to the triangle
    if u < 0.0 || v < 0.0 || w < 0.0 {
        // Closest point is not inside the triangle, find closest edge or vertex

        // - note: the lengths here are currently not used but may be in further refinement,
        //         or they will be removed
        //
        // Check edges
        let _ab_length = vector_magnitude(ab);
        let _ac_length = vector_magnitude(ac);
        let bc = (c.0 - b.0, c.1 - b.1, c.2 - b.2);
        let _bc_length = vector_magnitude(bc);

        // Check vertices
        let dist_a = vector_magnitude(ap);
        let bp = (0.0 - b.0, 0.0 - b.1, 0.0 - b.2);
        let dist_b = vector_magnitude(bp);
        let cp = (0.0 - c.0, 0.0 - c.1, 0.0 - c.2);
        let dist_c = vector_magnitude(cp);

        // Find the closest feature
        if dist_a <= dist_b && dist_a <= dist_c {
            // Closest to vertex A
            return (1.0, 0.0, 0.0);
        } else if dist_b <= dist_a && dist_b <= dist_c {
            // Closest to vertex B
            return (0.0, 1.0, 0.0);
        } else {
            // Closest to vertex C
            return (0.0, 0.0, 1.0);
        }
    }

    (u, v, w)
}
